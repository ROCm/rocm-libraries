	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.section	.text.bm224_bn320_bk064_wm2_wn4_mc0,"axG",@progbits,bm224_bn320_bk064_wm2_wn4_mc0,comdat
	.protected	bm224_bn320_bk064_wm2_wn4_mc0 ; -- Begin function bm224_bn320_bk064_wm2_wn4_mc0
	.globl	bm224_bn320_bk064_wm2_wn4_mc0
	.p2align	8
	.type	bm224_bn320_bk064_wm2_wn4_mc0,@function
bm224_bn320_bk064_wm2_wn4_mc0: ; @bm224_bn320_bk064_wm2_wn4_mc0
; %bb.0:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1 ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_mov_b64 s[2:3], src_shared_base
	s_movk_i32 s2, 0x7700
	s_load_b96 s[24:26], s[0:1], 0x78 nv
	s_and_b64 s[2:3], s[2:3], 12
	s_getreg_b32 s6, hwreg(HW_REG_IB_STS2, 6, 4)
	s_sub_co_i32 s4, 16, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_lshr_b32 s4, s4, 2
	s_cmp_lg_u64 s[2:3], 0
	s_cselect_b32 s4, s4, 0
	s_bfe_u32 s2, ttmp6, 0x4000c
	s_bfe_u32 s5, ttmp6, 0x40010
	s_add_co_i32 s2, s2, 1
	s_and_b32 s3, ttmp6, 15
	s_mul_i32 s2, ttmp9, s2
	s_add_co_i32 s5, s5, 1
	s_add_co_i32 s3, s3, s2
	s_mul_i32 s2, ttmp7, s5
	s_bfe_u32 s5, ttmp6, 0x40004
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_co_i32 s5, s5, s2
	s_cmp_eq_u32 s6, 0
	s_wait_kmcnt 0x0
	s_mov_b32 s9, s26
	s_cselect_b32 s38, ttmp9, s3
	s_cselect_b32 s33, ttmp7, s5
	s_add_co_i32 s2, s24, 0xdf
	s_mul_i32 s6, s38, 0xffffff20
	s_mul_hi_i32 s3, s2, 0x92492493
	s_add_co_i32 s7, s24, s6
	s_add_co_i32 s3, s3, s2
	s_add_co_i32 s2, s25, 0x13f
	s_lshr_b32 s5, s3, 31
	s_ashr_i32 s3, s3, 7
	s_mul_hi_i32 s2, s2, 0x66666667
	s_add_co_i32 s5, s3, s5
	s_lshr_b32 s3, s2, 31
	s_ashr_i32 s2, s2, 7
	s_min_i32 s27, s7, 0xe0
	s_add_co_i32 s6, s2, s3
	s_cmp_lt_i32 s38, s5
	s_cselect_b32 s39, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_and_b32 s2, s39, exec_lo
	s_mul_i32 s2, s33, 0xfffffec0
	s_cselect_b32 s3, s27, 0
	s_add_co_i32 s2, s25, s2
	s_min_i32 s2, s2, 0x140
	s_cmp_lt_i32 s33, s6
	s_cselect_b32 s40, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s7, s40, exec_lo
	s_cselect_b32 s29, s2, 0
	s_add_co_i32 s17, s26, 63
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 2, 1), 1
	s_min_i32 s2, s26, 64
	s_cmp_gt_i32 s17, 63
	s_cselect_b32 s16, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_and_b32 s7, s16, exec_lo
	s_cselect_b32 s2, s2, 0
	s_cmp_lt_i32 s3, 0xe0
	s_cselect_b32 s41, -1, 0
	s_and_b32 vcc_lo, exec_lo, s41
	s_mov_b32 s7, s41
	s_cbranch_vccnz .LBB0_2
; %bb.1:
	s_cmp_lt_i32 s29, 0x140
	s_cselect_b32 s7, -1, 0
	s_cmp_lt_i32 s2, 64
	s_cselect_b32 s8, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 s7, s8, s7
.LBB0_2:
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s7
	s_cbranch_vccnz .LBB0_8
; %bb.3:
	v_dual_mov_b32 v3, 0 :: v_dual_lshlrev_b32 v2, 2, v0
	v_or_b32_e32 v1, 0xffffff00, v0
	s_mov_b32 s7, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v4, v2 :: v_dual_mov_b32 v5, v1
.LBB0_4:                                ; =>This Inner Loop Header: Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)
	v_add_nc_u32_e32 v5, 0x100, v5
	ds_store_b32 v4, v3
	v_add_nc_u32_e32 v4, 0x400, v4
	v_cmp_lt_u32_e32 vcc_lo, 0x1cbf, v5
	s_or_b32 s7, vcc_lo, s7
	s_and_not1_b32 exec_lo, exec_lo, s7
	s_cbranch_execnz .LBB0_4
; %bb.5:
	s_or_b32 exec_lo, exec_lo, s7
	v_lshl_add_u32 v2, s4, 2, v2
	v_mov_b32_e32 v3, 0
	s_mov_b32 s7, 0
.LBB0_6:                                ; =>This Inner Loop Header: Depth=1
	v_add_nc_u32_e32 v1, 0x100, v1
	ds_store_b32 v2, v3 offset:30464
	v_add_nc_u32_e32 v2, 0x400, v2
	v_cmp_lt_u32_e32 vcc_lo, 0x297f, v1
	s_or_b32 s7, vcc_lo, s7
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s7
	s_cbranch_execnz .LBB0_6
; %bb.7:
	s_or_b32 exec_lo, exec_lo, s7
	s_barrier_signal -1
	s_barrier_wait -1
.LBB0_8:
	s_clause 0x2
	s_load_b64 s[18:19], s[0:1], 0x0 nv
	s_load_b128 s[12:15], s[0:1], 0x20 nv
	s_load_b128 s[20:23], s[0:1], 0x48 nv
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_lshrrev_b32_e32 v35 /*v291*/, 5, v0
	s_lshl_b32 s0, s4, 2
	s_add_co_i32 s6, s6, -1
	s_mov_b64 s[30:31], src_shared_base
	s_or_b32 s30, s0, 0x7700
	s_add_co_i32 s0, s5, -1
	s_min_i32 s36, s33, s6
	s_mov_b32 s1, exec_lo
	s_set_vgpr_msb 0x4004                   ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_cmpx_lt_i32_e32 0, v35 /*v291*/
	s_xor_b32 s1, exec_lo, s1
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_cbranch_execz .LBB0_12
; %bb.9:
	s_mov_b32 s25, exec_lo
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_cmpx_eq_u32_e32 1, v35 /*v291*/
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_cbranch_execz .LBB0_11
; %bb.10:
	s_cmp_gt_i32 s2, 0
	s_mul_i32 s4, s36, 0x140
	s_cselect_b32 s8, -1, 0
	s_ashr_i32 s5, s4, 31
	s_wait_kmcnt 0x0
	s_bfe_i64 s[6:7], s[20:21], 0x200000
	s_mov_b32 s28, s2
	s_mul_u64 s[4:5], s[6:7], s[4:5]
	s_and_b32 s6, s40, s8
	s_lshl_b64 s[4:5], s[4:5], 1
	v_cndmask_b32_e64 v2, 0, 1, s6
	s_add_nc_u64 s[4:5], s[14:15], s[4:5]
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_mov_b32 v1, s30 :: v_dual_mov_b32 v4, s4
	s_and_b32 s5, s5, 0x1ffffff
	v_readfirstlane_b32 s44, v2
	s_bitset1_b32 s5, 31
	v_readfirstlane_b32 s45, v1
	v_mov_b32_e32 v3, s5
	v_readfirstlane_b32 s46, v4
	s_mov_b32 s10, 0
	s_lshr_b32 s4, s29, 16
	s_lshr_b64 s[6:7], s[28:29], 16
	v_readfirstlane_b32 s47, v3
	s_movk_i32 s8, 0x140
	s_lshl_b32 s5, s2, 16
	s_or_b32 s7, s4, 0x400000
	s_mov_b32 s4, 0x7510000
	s_mov_b32 s11, s10
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[44:47], s[4:11]
.LBB0_11:
	s_or_b32 exec_lo, exec_lo, s25
.LBB0_12:
	s_or_saveexec_b32 s25, s1
	s_min_i32 s0, s38, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mulk_i32 s0, 0xe0
	s_xor_b32 exec_lo, exec_lo, s25
	s_cbranch_execz .LBB0_14
; %bb.13:
	s_cmp_gt_i32 s2, 0
	s_mov_b32 s10, 0
	s_cselect_b32 s6, -1, 0
	s_ashr_i32 s1, s0, 31
	s_wait_kmcnt 0x0
	s_bfe_i64 s[4:5], s[12:13], 0x200000
	s_movk_i32 s8, 0xe0
	s_mul_u64 s[4:5], s[4:5], s[0:1]
	s_mov_b32 s11, s10
	s_lshl_b64 s[4:5], s[4:5], 1
	s_mov_b32 s45, s10
	s_add_nc_u64 s[4:5], s[18:19], s[4:5]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s1, s5, 0x1ffffff
	s_and_b32 s5, s39, s6
	s_bitset1_b32 s1, 31
	v_cndmask_b32_e64 v2, 0, 1, s5
	v_dual_mov_b32 v4, s4 :: v_dual_mov_b32 v3, s1
	s_lshr_b32 s1, s3, 16
	s_lshr_b64 s[6:7], s[2:3], 16
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_readfirstlane_b32 s44, v2
	v_readfirstlane_b32 s46, v4
	v_readfirstlane_b32 s47, v3
	s_lshl_b32 s5, s2, 16
	s_or_b32 s7, s1, 0x400000
	s_mov_b32 s4, 0x7510000
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[44:47], s[4:11]
.LBB0_14:
	s_or_b32 exec_lo, exec_lo, s25
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	v_mov_b32_e32 v9, 0
	s_and_b32 s25, s39, s40
	s_set_vgpr_msb 0x44                     ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_and_b32_e32 v31 /*v287*/, 3, v35 /*v291*/
	v_cndmask_b32_e64 v27 /*v283*/, 0, 1, s25
	s_set_vgpr_msb 0x4440                   ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_lshrrev_b32_e32 v33 /*v289*/, 7, v0
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_dual_mov_b32 v8, v9 :: v_dual_mov_b32 v7, v9
	v_dual_mov_b32 v6, v9 :: v_dual_mov_b32 v5, v9
	v_dual_mov_b32 v4, v9 :: v_dual_mov_b32 v3, v9
	v_dual_mov_b32 v2, v9 :: v_dual_mov_b32 v17, v9
	v_dual_mov_b32 v16, v9 :: v_dual_mov_b32 v15, v9
	v_dual_mov_b32 v14, v9 :: v_dual_mov_b32 v13, v9
	v_dual_mov_b32 v12, v9 :: v_dual_mov_b32 v11, v9
	v_dual_mov_b32 v10, v9 :: v_dual_mov_b32 v25, v9
	v_dual_mov_b32 v24, v9 :: v_dual_mov_b32 v23, v9
	v_dual_mov_b32 v22, v9 :: v_dual_mov_b32 v21, v9
	v_dual_mov_b32 v20, v9 :: v_dual_mov_b32 v19, v9
	v_dual_mov_b32 v18, v9 :: v_dual_mov_b32 v33, v9
	v_dual_mov_b32 v32, v9 :: v_dual_mov_b32 v31, v9
	v_dual_mov_b32 v30, v9 :: v_dual_mov_b32 v29, v9
	v_dual_mov_b32 v28, v9 :: v_dual_mov_b32 v27, v9
	v_dual_mov_b32 v26, v9 :: v_dual_mov_b32 v41, v9
	v_dual_mov_b32 v40, v9 :: v_dual_mov_b32 v39, v9
	v_dual_mov_b32 v38, v9 :: v_dual_mov_b32 v37, v9
	v_dual_mov_b32 v36, v9 :: v_dual_mov_b32 v35, v9
	v_dual_mov_b32 v34, v9 :: v_dual_mov_b32 v49, v9
	v_dual_mov_b32 v48, v9 :: v_dual_mov_b32 v47, v9
	v_dual_mov_b32 v46, v9 :: v_dual_mov_b32 v45, v9
	v_dual_mov_b32 v44, v9 :: v_dual_mov_b32 v43, v9
	v_dual_mov_b32 v42, v9 :: v_dual_mov_b32 v57, v9
	v_dual_mov_b32 v56, v9 :: v_dual_mov_b32 v55, v9
	v_dual_mov_b32 v54, v9 :: v_dual_mov_b32 v53, v9
	v_dual_mov_b32 v52, v9 :: v_dual_mov_b32 v51, v9
	v_dual_mov_b32 v50, v9 :: v_dual_mov_b32 v65, v9
	v_dual_mov_b32 v64, v9 :: v_dual_mov_b32 v63, v9
	v_dual_mov_b32 v62, v9 :: v_dual_mov_b32 v61, v9
	v_dual_mov_b32 v60, v9 :: v_dual_mov_b32 v59, v9
	v_dual_mov_b32 v58, v9 :: v_dual_mov_b32 v73, v9
	v_dual_mov_b32 v72, v9 :: v_dual_mov_b32 v71, v9
	v_dual_mov_b32 v70, v9 :: v_dual_mov_b32 v69, v9
	v_dual_mov_b32 v68, v9 :: v_dual_mov_b32 v67, v9
	v_dual_mov_b32 v66, v9 :: v_dual_mov_b32 v81, v9
	v_dual_mov_b32 v80, v9 :: v_dual_mov_b32 v79, v9
	v_dual_mov_b32 v78, v9 :: v_dual_mov_b32 v77, v9
	v_dual_mov_b32 v76, v9 :: v_dual_mov_b32 v75, v9
	v_dual_mov_b32 v74, v9 :: v_dual_mov_b32 v89, v9
	v_dual_mov_b32 v88, v9 :: v_dual_mov_b32 v87, v9
	v_dual_mov_b32 v86, v9 :: v_dual_mov_b32 v85, v9
	v_dual_mov_b32 v84, v9 :: v_dual_mov_b32 v83, v9
	v_dual_mov_b32 v82, v9 :: v_dual_mov_b32 v97, v9
	v_dual_mov_b32 v96, v9 :: v_dual_mov_b32 v95, v9
	v_dual_mov_b32 v94, v9 :: v_dual_mov_b32 v93, v9
	v_dual_mov_b32 v92, v9 :: v_dual_mov_b32 v91, v9
	v_dual_mov_b32 v90, v9 :: v_dual_mov_b32 v105, v9
	v_dual_mov_b32 v104, v9 :: v_dual_mov_b32 v103, v9
	v_dual_mov_b32 v102, v9 :: v_dual_mov_b32 v101, v9
	v_dual_mov_b32 v100, v9 :: v_dual_mov_b32 v99, v9
	v_dual_mov_b32 v98, v9 :: v_dual_mov_b32 v113, v9
	v_dual_mov_b32 v112, v9 :: v_dual_mov_b32 v111, v9
	v_dual_mov_b32 v110, v9 :: v_dual_mov_b32 v109, v9
	v_dual_mov_b32 v108, v9 :: v_dual_mov_b32 v107, v9
	v_dual_mov_b32 v106, v9 :: v_dual_mov_b32 v121, v9
	v_dual_mov_b32 v120, v9 :: v_dual_mov_b32 v119, v9
	v_dual_mov_b32 v118, v9 :: v_dual_mov_b32 v117, v9
	v_dual_mov_b32 v116, v9 :: v_dual_mov_b32 v115, v9
	v_dual_mov_b32 v114, v9 :: v_dual_mov_b32 v129, v9
	v_dual_mov_b32 v128, v9 :: v_dual_mov_b32 v127, v9
	v_dual_mov_b32 v126, v9 :: v_dual_mov_b32 v125, v9
	v_dual_mov_b32 v124, v9 :: v_dual_mov_b32 v123, v9
	v_dual_mov_b32 v122, v9 :: v_dual_mov_b32 v145, v9
	v_dual_mov_b32 v144, v9 :: v_dual_mov_b32 v143, v9
	v_dual_mov_b32 v142, v9 :: v_dual_mov_b32 v141, v9
	v_dual_mov_b32 v140, v9 :: v_dual_mov_b32 v139, v9
	v_dual_mov_b32 v138, v9 :: v_dual_mov_b32 v153, v9
	v_dual_mov_b32 v152, v9 :: v_dual_mov_b32 v151, v9
	v_dual_mov_b32 v150, v9 :: v_dual_mov_b32 v149, v9
	v_dual_mov_b32 v148, v9 :: v_dual_mov_b32 v147, v9
	v_dual_mov_b32 v146, v9 :: v_dual_mov_b32 v161, v9
	v_dual_mov_b32 v160, v9 :: v_dual_mov_b32 v159, v9
	v_dual_mov_b32 v158, v9 :: v_dual_mov_b32 v157, v9
	v_dual_mov_b32 v156, v9 :: v_dual_mov_b32 v155, v9
	v_dual_mov_b32 v154, v9 :: v_dual_mov_b32 v169, v9
	v_dual_mov_b32 v168, v9 :: v_dual_mov_b32 v167, v9
	v_dual_mov_b32 v166, v9 :: v_dual_mov_b32 v165, v9
	v_dual_mov_b32 v164, v9 :: v_dual_mov_b32 v163, v9
	v_dual_mov_b32 v162, v9 :: v_dual_mov_b32 v177, v9
	v_dual_mov_b32 v176, v9 :: v_dual_mov_b32 v175, v9
	v_dual_mov_b32 v174, v9 :: v_dual_mov_b32 v173, v9
	v_dual_mov_b32 v172, v9 :: v_dual_mov_b32 v171, v9
	v_dual_mov_b32 v170, v9 :: v_dual_mov_b32 v185, v9
	v_dual_mov_b32 v184, v9 :: v_dual_mov_b32 v183, v9
	v_dual_mov_b32 v182, v9 :: v_dual_mov_b32 v181, v9
	v_dual_mov_b32 v180, v9 :: v_dual_mov_b32 v179, v9
	v_dual_mov_b32 v178, v9 :: v_dual_mov_b32 v193, v9
	v_dual_mov_b32 v192, v9 :: v_dual_mov_b32 v191, v9
	v_dual_mov_b32 v190, v9 :: v_dual_mov_b32 v189, v9
	v_dual_mov_b32 v188, v9 :: v_dual_mov_b32 v187, v9
	v_dual_mov_b32 v186, v9 :: v_dual_mov_b32 v201, v9
	v_dual_mov_b32 v200, v9 :: v_dual_mov_b32 v199, v9
	v_dual_mov_b32 v198, v9 :: v_dual_mov_b32 v197, v9
	v_dual_mov_b32 v196, v9 :: v_dual_mov_b32 v195, v9
	v_dual_mov_b32 v194, v9 :: v_dual_mov_b32 v209, v9
	v_dual_mov_b32 v208, v9 :: v_dual_mov_b32 v207, v9
	v_dual_mov_b32 v206, v9 :: v_dual_mov_b32 v205, v9
	v_dual_mov_b32 v204, v9 :: v_dual_mov_b32 v203, v9
	v_dual_mov_b32 v202, v9 :: v_dual_mov_b32 v217, v9
	v_dual_mov_b32 v216, v9 :: v_dual_mov_b32 v215, v9
	v_dual_mov_b32 v214, v9 :: v_dual_mov_b32 v213, v9
	v_dual_mov_b32 v212, v9 :: v_dual_mov_b32 v211, v9
	v_dual_mov_b32 v210, v9 :: v_dual_mov_b32 v225, v9
	v_dual_mov_b32 v224, v9 :: v_dual_mov_b32 v223, v9
	v_dual_mov_b32 v222, v9 :: v_dual_mov_b32 v221, v9
	v_dual_mov_b32 v220, v9 :: v_dual_mov_b32 v219, v9
	v_dual_mov_b32 v218, v9 :: v_dual_mov_b32 v233, v9
	v_dual_mov_b32 v232, v9 :: v_dual_mov_b32 v231, v9
	v_dual_mov_b32 v230, v9 :: v_dual_mov_b32 v229, v9
	v_dual_mov_b32 v228, v9 :: v_dual_mov_b32 v227, v9
	v_dual_mov_b32 v226, v9 :: v_dual_mov_b32 v241, v9
	v_dual_mov_b32 v240, v9 :: v_dual_mov_b32 v239, v9
	v_dual_mov_b32 v238, v9 :: v_dual_mov_b32 v237, v9
	v_dual_mov_b32 v236, v9 :: v_dual_mov_b32 v235, v9
	v_dual_mov_b32 v234, v9 :: v_dual_mov_b32 v249, v9
	v_dual_mov_b32 v248, v9 :: v_dual_mov_b32 v247, v9
	v_dual_mov_b32 v246, v9 :: v_dual_mov_b32 v245, v9
	v_dual_mov_b32 v244, v9 :: v_dual_mov_b32 v243, v9
	v_dual_mov_b32 v242, v9 :: v_dual_mov_b32 v255, v9
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_dual_mov_b32 v1 /*v257*/, v9 :: v_dual_mov_b32 v0 /*v256*/, v9
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_dual_mov_b32 v254, v9 :: v_dual_mov_b32 v253, v9
	v_dual_mov_b32 v252, v9 :: v_dual_mov_b32 v251, v9
	v_dual_mov_b32 v250, v9 :: v_dual_mov_b32 v137, v9
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_dual_mov_b32 v9 /*v265*/, v9 :: v_dual_mov_b32 v8 /*v264*/, v9
	v_dual_mov_b32 v7 /*v263*/, v9 :: v_dual_mov_b32 v6 /*v262*/, v9
	v_dual_mov_b32 v5 /*v261*/, v9 :: v_dual_mov_b32 v4 /*v260*/, v9
	v_dual_mov_b32 v3 /*v259*/, v9 :: v_dual_mov_b32 v2 /*v258*/, v9
	v_dual_mov_b32 v17 /*v273*/, v9 :: v_dual_mov_b32 v16 /*v272*/, v9
	v_dual_mov_b32 v15 /*v271*/, v9 :: v_dual_mov_b32 v14 /*v270*/, v9
	v_dual_mov_b32 v13 /*v269*/, v9 :: v_dual_mov_b32 v12 /*v268*/, v9
	v_dual_mov_b32 v11 /*v267*/, v9 :: v_dual_mov_b32 v10 /*v266*/, v9
	v_dual_mov_b32 v25 /*v281*/, v9 :: v_dual_mov_b32 v24 /*v280*/, v9
	v_dual_mov_b32 v23 /*v279*/, v9 :: v_dual_mov_b32 v22 /*v278*/, v9
	v_dual_mov_b32 v21 /*v277*/, v9 :: v_dual_mov_b32 v20 /*v276*/, v9
	v_dual_mov_b32 v19 /*v275*/, v9 :: v_dual_mov_b32 v18 /*v274*/, v9
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_dual_mov_b32 v136, v9 :: v_dual_mov_b32 v135, v9
	v_dual_mov_b32 v134, v9 :: v_dual_mov_b32 v133, v9
	v_dual_mov_b32 v132, v9 :: v_dual_mov_b32 v131, v9
	v_mov_b32_e32 v130, v9
	s_and_not1_b32 vcc_lo, exec_lo, s16
	s_barrier_wait -1
	s_cbranch_vccnz .LBB0_37
; %bb.15:
	v_dual_lshlrev_b32 v4, 6, v0 :: v_dual_bitop2_b32 v2, 16, v0 bitop3:0x40
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_mul_u32_u24_e32 v3, 0x1c00, v33 /*v289*/
	s_set_vgpr_msb 0x440                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_mov_b32_e32 v29 /*v285*/, 0
	s_set_vgpr_msb 0x4004                   ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_mul_u32_u24_e32 v1, 0x1400, v31 /*v287*/
	s_mov_b64 s[4:5], src_shared_base
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_and_or_b32 v5, 0x3c0, v4, v2
	s_add_co_i32 s6, s30, 0xaa00
	s_mov_b32 s7, s5
	s_mov_b32 s11, 0
	s_and_b64 s[6:7], s[6:7], 15
	v_or_b32_e32 v7, v5, v3
	s_sub_co_i32 s1, 16, s6
	s_mov_b32 s44, s5
	s_lshr_b32 s1, s1, 2
	s_cmp_lg_u64 s[6:7], 0
	v_add_nc_u32_e32 v2, 0x400, v7
	s_cselect_b32 s1, s1, 0
	s_mov_b32 s42, s31
	s_lshl2_add_u32 s1, s1, s30
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_dual_lshrrev_b32 v8, 4, v2 :: v_dual_bitop2_b32 v24, v5, v1 bitop3:0x54
	v_lshrrev_b32_e32 v4, 4, v7
	v_add_nc_u32_e32 v12, 0x1800, v7
	s_add_co_i32 s4, s1, 0x12100
	v_dual_lshrrev_b32 v16, 4, v24 :: v_dual_bitop2_b32 v5, 32, v5 bitop3:0x54
	v_and_b32_e32 v11, 0x3f8, v8
	v_add_nc_u32_e32 v8, 0x1000, v7
	v_and_b32_e32 v6, 0x1f8, v4
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v16, 0x3f8, v16
	v_add_nc_u32_e32 v18, 0x400, v24
	v_add_nc_u32_e32 v19, 0x800, v24
	v_lshrrev_b32_e32 v14, 4, v8
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_dual_add_nc_u32 v26 /*v282*/, v6, v7 :: v_dual_add_nc_u32 v30 /*v286*/, v16, v24
	s_and_b32 s10, s4, 15
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshrrev_b32_e32 v16, 4, v19
	v_and_b32_e32 v17, 0x3f8, v14
	v_dual_lshrrev_b32 v14, 4, v12 :: v_dual_bitop2_b32 v3, v5, v3 bitop3:0x54
	v_add_nc_u32_e32 v19, 0x1000, v24
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_and_b32_e32 v25, 0x7f8, v16
	s_sub_co_i32 s2, 16, s10
	v_and_b32_e32 v21, 0x3f8, v14
	v_lshrrev_b32_e32 v14, 4, v18
	v_add_nc_u32_e32 v18, 0xc00, v24
	v_add_nc_u32_e32 v16, 0x400, v3
	v_lshrrev_b32_e32 v20, 4, v3
	s_add_co_i32 s43, s1, 0xaa00
	v_and_b32_e32 v23, 0x7f8, v14
	v_dual_lshrrev_b32 v14, 4, v18 :: v_dual_lshrrev_b32 v18, 4, v19
	v_add_nc_u32_e32 v19, 0x800, v3
	s_lshr_b32 s1, s2, 2
	v_or_b32_e32 v1, v5, v1
	s_delay_alu instid0(VALU_DEP_3)
	v_and_b32_e32 v26, 0x7f8, v14
	v_lshrrev_b32_e32 v14, 4, v16
	v_and_b32_e32 v27, 0x7f8, v18
	v_add_nc_u32_e32 v18, 0xc00, v3
	v_lshrrev_b32_e32 v16, 4, v19
	v_and_b32_e32 v19, 0x1f8, v20
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_and_b32_e32 v28 /*v284*/, 0x3f8, v14
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_add_nc_u32_e32 v20, 0x1000, v3
	v_lshrrev_b32_e32 v18, 4, v18
	v_and_b32_e32 v14, 0x3f8, v16
	v_add_nc_u32_e32 v16, 0x1400, v3
	v_add_nc_u32_e32 v3, 0x1800, v3
	v_lshrrev_b32_e32 v20, 4, v20
	v_add_nc_u32_e32 v28, 0x400, v1
	v_add_nc_u32_e32 v29, 0xc00, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_dual_lshrrev_b32 v5, 4, v16 :: v_dual_lshrrev_b32 v3, 4, v3
	v_and_b32_e32 v16, 0x3f8, v18
	v_and_b32_e32 v18, 0x3f8, v20
	s_cmp_lg_u64 s[10:11], 0
	v_and_b32_e32 v20, 0x3f8, v5
	v_lshrrev_b32_e32 v5, 4, v1
	v_and_b32_e32 v22, 0x3f8, v3
	v_add_nc_u32_e32 v3, 0x800, v1
	v_add_nc_u32_e32 v1, 0x1000, v1
	s_cselect_b32 s1, s1, 0
	v_and_b32_e32 v30, 0x3f8, v5
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_dual_lshrrev_b32 v5, 4, v29 :: v_dual_lshrrev_b32 v3, 4, v3
	v_sub_nc_u32_e32 v29, 0x1dbf, v0
	v_lshrrev_b32_e32 v1, 4, v1
	s_ashr_i32 s2, s17, 31
	v_and_b32_e32 v32, 0x7f8, v5
	v_and_b32_e32 v31, 0x7f8, v3
	v_sub_nc_u32_e32 v3, 0x2a7f, v0
	v_lshrrev_b32_e32 v5, 8, v29
	v_and_b32_e32 v29, 0x7f8, v1
	s_lshr_b32 s2, s2, 26
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_dual_add_nc_u32 v50 /*v306*/, v27, v24 :: v_dual_add_nc_u32 v66 /*v322*/, v30, v24
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_dual_lshrrev_b32 v1, 8, v3 :: v_dual_add_nc_u32 v3, 1, v5
	s_add_co_i32 s17, s17, s2
	s_lshl_b32 s10, s1, 2
	s_ashr_i32 s45, s17, 6
	s_add_nc_u64 s[34:35], s[4:5], s[10:11]
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_and_b32_e32 v37 /*v293*/, 30, v3
	s_cmp_lt_i32 s29, 0x140
	s_mul_i32 s4, s36, 0x140
	s_cselect_b32 s46, -1, 0
	s_wait_kmcnt 0x0
	s_bfe_i64 s[6:7], s[20:21], 0x200000
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_add_nc_u32_e32 v5, 1, v1
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_dual_add_nc_u32 v42 /*v298*/, v21, v7 :: v_dual_add_nc_u32 v44 /*v300*/, v23, v24
	v_dual_add_nc_u32 v46 /*v302*/, v25, v24 :: v_dual_add_nc_u32 v48 /*v304*/, v26, v24
	v_add3_u32 v70 /*v326*/, v24, v31, 0x800
	v_add3_u32 v72 /*v328*/, v24, v32, 0xc00
	v_add3_u32 v74 /*v330*/, v24, v29, 0x1000
	v_add_nc_u32_e32 v32 /*v288*/, v11, v7
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_lshl_or_b32 v33, v37 /*v293*/, 8, v0
	v_mov_b32_e32 v23, v29 /*v285*/
	s_ashr_i32 s5, s4, 31
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_add_nc_u32_e32 v52 /*v308*/, v19, v7
	s_mul_u64 s[4:5], s[6:7], s[4:5]
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_dual_mov_b32 v19, v29 /*v285*/ :: v_dual_mov_b32 v21, v29 /*v285*/
	s_bfe_i64 s[6:7], s[12:13], 0x200000
	v_mov_b32_e32 v11, v29 /*v285*/
	s_ashr_i32 s1, s0, 31
	v_or_b32_e32 v1, 0x100, v0
	s_mul_u64 s[0:1], s[6:7], s[0:1]
	s_set_vgpr_msb 0x141                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	v_dual_mov_b32 v0 /*v256*/, v29 /*v285*/ :: v_dual_add_nc_u32 v41 /*v297*/, 0xffffff00, v33
	s_set_vgpr_msb 0x4100                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_add_nc_u32_e32 v10, 0x1400, v7
	v_add_nc_u32_e32 v6, 0xc00, v7
	v_add_nc_u32_e32 v4, 0x800, v7
	s_lshl_b64 s[0:1], s[0:1], 1
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_lshlrev_b32_e32 v76 /*v332*/, 2, v33
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_lshrrev_b32_e32 v15, 4, v10
	s_add_nc_u64 s[36:37], s[18:19], s[0:1]
	v_dual_mov_b32 v27, v29 /*v285*/ :: v_dual_lshrrev_b32 v28, 4, v28
	s_set_vgpr_msb 0x141                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	v_dual_mov_b32 v79 /*v335*/, v29 /*v285*/ :: v_dual_mov_b32 v1 /*v257*/, v29 /*v285*/
	s_set_vgpr_msb 0x4101                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_and_b32_e32 v15, 0x3f8, v15
	v_mov_b32_e32 v25, v29 /*v285*/
	v_and_b32_e32 v28, 0x7f8, v28
	v_mov_b32_e32 v26, v29 /*v285*/
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_dual_add_nc_u32 v38 /*v294*/, v17, v7 :: v_dual_add_nc_u32 v40 /*v296*/, v15, v7
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_mov_b32_e32 v15, v29 /*v285*/
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_add3_u32 v68 /*v324*/, v24, v28, 0x400
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_mov_b32_e32 v24, v29 /*v285*/
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_and_b32_e32 v39 /*v295*/, 46, v5
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_dual_mov_b32 v17, v29 /*v285*/ :: v_dual_mov_b32 v29, v29 /*v285*/
	v_dual_mov_b32 v30, v29 /*v285*/ :: v_dual_mov_b32 v28, v29 /*v285*/
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_ne_u32_e64 s1, v39 /*v295*/, v5
	v_mov_b32_e32 v5, v29 /*v285*/
	v_lshl_or_b32 v34, v39 /*v295*/, 8, v0
	v_dual_mov_b32 v31, v29 /*v285*/ :: v_dual_mov_b32 v32, v29 /*v285*/
	v_lshrrev_b32_e32 v13, 4, v6
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_add_nc_u64_e32 v[56:57] /*v[312:313]*/, v[14:15], v[4:5]
	s_set_vgpr_msb 0x4041                   ;  msbs: dst=1 src0=1 src1=0 src2=0
	v_dual_mov_b32 v77 /*v333*/, v29 /*v285*/ :: v_dual_lshlrev_b32 v78 /*v334*/, 2, v34
	s_set_vgpr_msb 0x4100                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshrrev_b32_e32 v9, 4, v4
	v_and_b32_e32 v13, 0x3f8, v13
	s_set_vgpr_msb 0x41                     ;  msbs: dst=1 src0=1 src1=0 src2=0
	v_dual_mov_b32 v2 /*v258*/, v29 /*v285*/ :: v_dual_add_nc_u32 v43 /*v299*/, 0xffffff00, v34
	s_set_vgpr_msb 0x4101                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_mov_b32_e32 v4, v29 /*v285*/
	v_and_b32_e32 v9, 0x3f8, v9
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_add_nc_u32_e32 v36 /*v292*/, v13, v7
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_dual_mov_b32 v13, v29 /*v285*/ :: v_dual_mov_b32 v14, v29 /*v285*/
	v_dual_mov_b32 v33, v29 /*v285*/ :: v_dual_mov_b32 v34, v29 /*v285*/
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_add_nc_u32_e32 v34 /*v290*/, v9, v7
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_dual_mov_b32 v7, v29 /*v285*/ :: v_dual_mov_b32 v9, v29 /*v285*/
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_add_nc_u64_e32 v[64:65] /*v[320:321]*/, v[22:23], v[12:13]
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_mov_b32_e32 v22, v29 /*v285*/
	v_cmp_ne_u32_e64 s0, v37 /*v293*/, v3
	v_mov_b32_e32 v3, v29 /*v285*/
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_add_nc_u64_e32 v[58:59] /*v[314:315]*/, v[16:17], v[6:7]
	v_add_nc_u64_e32 v[60:61] /*v[316:317]*/, v[18:19], v[8:9]
	v_add_nc_u64_e32 v[62:63] /*v[318:319]*/, v[20:21], v[10:11]
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_mov_b32_e32 v6, v29 /*v285*/
	s_set_vgpr_msb 0x141                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	v_add_nc_u64_e32 v[54:55] /*v[310:311]*/, v[28:29] /*v[284:285]*/, v[2:3]
	s_set_vgpr_msb 0x4101                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_dual_mov_b32 v2, v29 /*v285*/ :: v_dual_mov_b32 v8, v29 /*v285*/
	v_dual_mov_b32 v10, v29 /*v285*/ :: v_dual_mov_b32 v12, v29 /*v285*/
	v_dual_mov_b32 v16, v29 /*v285*/ :: v_dual_mov_b32 v18, v29 /*v285*/
	v_dual_mov_b32 v20, v29 /*v285*/ :: v_dual_mov_b32 v35, v29 /*v285*/
	v_dual_mov_b32 v36, v29 /*v285*/ :: v_dual_mov_b32 v37, v29 /*v285*/
	v_dual_mov_b32 v38, v29 /*v285*/ :: v_dual_mov_b32 v39, v29 /*v285*/
	v_dual_mov_b32 v40, v29 /*v285*/ :: v_dual_mov_b32 v41, v29 /*v285*/
	v_dual_mov_b32 v42, v29 /*v285*/ :: v_dual_mov_b32 v43, v29 /*v285*/
	v_dual_mov_b32 v44, v29 /*v285*/ :: v_dual_mov_b32 v45, v29 /*v285*/
	v_dual_mov_b32 v46, v29 /*v285*/ :: v_dual_mov_b32 v47, v29 /*v285*/
	v_dual_mov_b32 v48, v29 /*v285*/ :: v_dual_mov_b32 v49, v29 /*v285*/
	v_dual_mov_b32 v50, v29 /*v285*/ :: v_dual_mov_b32 v51, v29 /*v285*/
	v_dual_mov_b32 v52, v29 /*v285*/ :: v_dual_mov_b32 v53, v29 /*v285*/
	v_dual_mov_b32 v54, v29 /*v285*/ :: v_dual_mov_b32 v55, v29 /*v285*/
	v_dual_mov_b32 v56, v29 /*v285*/ :: v_dual_mov_b32 v57, v29 /*v285*/
	v_dual_mov_b32 v58, v29 /*v285*/ :: v_dual_mov_b32 v59, v29 /*v285*/
	v_dual_mov_b32 v60, v29 /*v285*/ :: v_dual_mov_b32 v61, v29 /*v285*/
	v_dual_mov_b32 v62, v29 /*v285*/ :: v_dual_mov_b32 v63, v29 /*v285*/
	v_dual_mov_b32 v64, v29 /*v285*/ :: v_dual_mov_b32 v65, v29 /*v285*/
	v_dual_mov_b32 v66, v29 /*v285*/ :: v_dual_mov_b32 v67, v29 /*v285*/
	v_dual_mov_b32 v68, v29 /*v285*/ :: v_dual_mov_b32 v69, v29 /*v285*/
	v_dual_mov_b32 v70, v29 /*v285*/ :: v_dual_mov_b32 v71, v29 /*v285*/
	v_dual_mov_b32 v72, v29 /*v285*/ :: v_dual_mov_b32 v73, v29 /*v285*/
	v_dual_mov_b32 v74, v29 /*v285*/ :: v_dual_mov_b32 v75, v29 /*v285*/
	v_dual_mov_b32 v76, v29 /*v285*/ :: v_dual_mov_b32 v77, v29 /*v285*/
	v_dual_mov_b32 v78, v29 /*v285*/ :: v_dual_mov_b32 v79, v29 /*v285*/
	v_dual_mov_b32 v80, v29 /*v285*/ :: v_dual_mov_b32 v81, v29 /*v285*/
	v_dual_mov_b32 v82, v29 /*v285*/ :: v_dual_mov_b32 v83, v29 /*v285*/
	v_dual_mov_b32 v84, v29 /*v285*/ :: v_dual_mov_b32 v85, v29 /*v285*/
	v_dual_mov_b32 v86, v29 /*v285*/ :: v_dual_mov_b32 v87, v29 /*v285*/
	v_dual_mov_b32 v88, v29 /*v285*/ :: v_dual_mov_b32 v89, v29 /*v285*/
	v_dual_mov_b32 v90, v29 /*v285*/ :: v_dual_mov_b32 v91, v29 /*v285*/
	v_dual_mov_b32 v92, v29 /*v285*/ :: v_dual_mov_b32 v93, v29 /*v285*/
	v_dual_mov_b32 v94, v29 /*v285*/ :: v_dual_mov_b32 v95, v29 /*v285*/
	v_dual_mov_b32 v96, v29 /*v285*/ :: v_dual_mov_b32 v97, v29 /*v285*/
	v_dual_mov_b32 v98, v29 /*v285*/ :: v_dual_mov_b32 v99, v29 /*v285*/
	v_dual_mov_b32 v100, v29 /*v285*/ :: v_dual_mov_b32 v101, v29 /*v285*/
	v_dual_mov_b32 v102, v29 /*v285*/ :: v_dual_mov_b32 v103, v29 /*v285*/
	v_dual_mov_b32 v104, v29 /*v285*/ :: v_dual_mov_b32 v105, v29 /*v285*/
	v_dual_mov_b32 v106, v29 /*v285*/ :: v_dual_mov_b32 v107, v29 /*v285*/
	v_dual_mov_b32 v108, v29 /*v285*/ :: v_dual_mov_b32 v109, v29 /*v285*/
	v_dual_mov_b32 v110, v29 /*v285*/ :: v_dual_mov_b32 v111, v29 /*v285*/
	v_dual_mov_b32 v112, v29 /*v285*/ :: v_dual_mov_b32 v113, v29 /*v285*/
	v_dual_mov_b32 v114, v29 /*v285*/ :: v_dual_mov_b32 v115, v29 /*v285*/
	v_dual_mov_b32 v116, v29 /*v285*/ :: v_dual_mov_b32 v117, v29 /*v285*/
	v_dual_mov_b32 v118, v29 /*v285*/ :: v_dual_mov_b32 v119, v29 /*v285*/
	v_dual_mov_b32 v120, v29 /*v285*/ :: v_dual_mov_b32 v121, v29 /*v285*/
	v_dual_mov_b32 v122, v29 /*v285*/ :: v_dual_mov_b32 v123, v29 /*v285*/
	v_dual_mov_b32 v124, v29 /*v285*/ :: v_dual_mov_b32 v125, v29 /*v285*/
	v_dual_mov_b32 v126, v29 /*v285*/ :: v_dual_mov_b32 v127, v29 /*v285*/
	v_dual_mov_b32 v128, v29 /*v285*/ :: v_dual_mov_b32 v129, v29 /*v285*/
	v_dual_mov_b32 v138, v29 /*v285*/ :: v_dual_mov_b32 v139, v29 /*v285*/
	v_dual_mov_b32 v140, v29 /*v285*/ :: v_dual_mov_b32 v141, v29 /*v285*/
	v_dual_mov_b32 v142, v29 /*v285*/ :: v_dual_mov_b32 v143, v29 /*v285*/
	v_dual_mov_b32 v144, v29 /*v285*/ :: v_dual_mov_b32 v145, v29 /*v285*/
	v_dual_mov_b32 v146, v29 /*v285*/ :: v_dual_mov_b32 v147, v29 /*v285*/
	v_dual_mov_b32 v148, v29 /*v285*/ :: v_dual_mov_b32 v149, v29 /*v285*/
	v_dual_mov_b32 v150, v29 /*v285*/ :: v_dual_mov_b32 v151, v29 /*v285*/
	v_dual_mov_b32 v152, v29 /*v285*/ :: v_dual_mov_b32 v153, v29 /*v285*/
	v_dual_mov_b32 v154, v29 /*v285*/ :: v_dual_mov_b32 v155, v29 /*v285*/
	v_dual_mov_b32 v156, v29 /*v285*/ :: v_dual_mov_b32 v157, v29 /*v285*/
	v_dual_mov_b32 v158, v29 /*v285*/ :: v_dual_mov_b32 v159, v29 /*v285*/
	v_dual_mov_b32 v160, v29 /*v285*/ :: v_dual_mov_b32 v161, v29 /*v285*/
	v_dual_mov_b32 v162, v29 /*v285*/ :: v_dual_mov_b32 v163, v29 /*v285*/
	v_dual_mov_b32 v164, v29 /*v285*/ :: v_dual_mov_b32 v165, v29 /*v285*/
	v_dual_mov_b32 v166, v29 /*v285*/ :: v_dual_mov_b32 v167, v29 /*v285*/
	v_dual_mov_b32 v168, v29 /*v285*/ :: v_dual_mov_b32 v169, v29 /*v285*/
	v_dual_mov_b32 v170, v29 /*v285*/ :: v_dual_mov_b32 v171, v29 /*v285*/
	v_dual_mov_b32 v172, v29 /*v285*/ :: v_dual_mov_b32 v173, v29 /*v285*/
	v_dual_mov_b32 v174, v29 /*v285*/ :: v_dual_mov_b32 v175, v29 /*v285*/
	v_dual_mov_b32 v176, v29 /*v285*/ :: v_dual_mov_b32 v177, v29 /*v285*/
	v_dual_mov_b32 v178, v29 /*v285*/ :: v_dual_mov_b32 v179, v29 /*v285*/
	v_dual_mov_b32 v180, v29 /*v285*/ :: v_dual_mov_b32 v181, v29 /*v285*/
	v_dual_mov_b32 v182, v29 /*v285*/ :: v_dual_mov_b32 v183, v29 /*v285*/
	v_dual_mov_b32 v184, v29 /*v285*/ :: v_dual_mov_b32 v185, v29 /*v285*/
	v_dual_mov_b32 v186, v29 /*v285*/ :: v_dual_mov_b32 v187, v29 /*v285*/
	v_dual_mov_b32 v188, v29 /*v285*/ :: v_dual_mov_b32 v189, v29 /*v285*/
	v_dual_mov_b32 v190, v29 /*v285*/ :: v_dual_mov_b32 v191, v29 /*v285*/
	v_dual_mov_b32 v192, v29 /*v285*/ :: v_dual_mov_b32 v193, v29 /*v285*/
	v_dual_mov_b32 v194, v29 /*v285*/ :: v_dual_mov_b32 v195, v29 /*v285*/
	v_dual_mov_b32 v196, v29 /*v285*/ :: v_dual_mov_b32 v197, v29 /*v285*/
	v_dual_mov_b32 v198, v29 /*v285*/ :: v_dual_mov_b32 v199, v29 /*v285*/
	v_dual_mov_b32 v200, v29 /*v285*/ :: v_dual_mov_b32 v201, v29 /*v285*/
	v_dual_mov_b32 v202, v29 /*v285*/ :: v_dual_mov_b32 v203, v29 /*v285*/
	v_dual_mov_b32 v204, v29 /*v285*/ :: v_dual_mov_b32 v205, v29 /*v285*/
	v_dual_mov_b32 v206, v29 /*v285*/ :: v_dual_mov_b32 v207, v29 /*v285*/
	v_dual_mov_b32 v208, v29 /*v285*/ :: v_dual_mov_b32 v209, v29 /*v285*/
	v_dual_mov_b32 v210, v29 /*v285*/ :: v_dual_mov_b32 v211, v29 /*v285*/
	v_dual_mov_b32 v212, v29 /*v285*/ :: v_dual_mov_b32 v213, v29 /*v285*/
	v_dual_mov_b32 v214, v29 /*v285*/ :: v_dual_mov_b32 v215, v29 /*v285*/
	v_dual_mov_b32 v216, v29 /*v285*/ :: v_dual_mov_b32 v217, v29 /*v285*/
	v_dual_mov_b32 v218, v29 /*v285*/ :: v_dual_mov_b32 v219, v29 /*v285*/
	v_dual_mov_b32 v220, v29 /*v285*/ :: v_dual_mov_b32 v221, v29 /*v285*/
	v_dual_mov_b32 v222, v29 /*v285*/ :: v_dual_mov_b32 v223, v29 /*v285*/
	v_dual_mov_b32 v224, v29 /*v285*/ :: v_dual_mov_b32 v225, v29 /*v285*/
	v_dual_mov_b32 v226, v29 /*v285*/ :: v_dual_mov_b32 v227, v29 /*v285*/
	v_dual_mov_b32 v228, v29 /*v285*/ :: v_dual_mov_b32 v229, v29 /*v285*/
	v_dual_mov_b32 v230, v29 /*v285*/ :: v_dual_mov_b32 v231, v29 /*v285*/
	v_dual_mov_b32 v232, v29 /*v285*/ :: v_dual_mov_b32 v233, v29 /*v285*/
	v_dual_mov_b32 v234, v29 /*v285*/ :: v_dual_mov_b32 v235, v29 /*v285*/
	v_dual_mov_b32 v236, v29 /*v285*/ :: v_dual_mov_b32 v237, v29 /*v285*/
	v_dual_mov_b32 v238, v29 /*v285*/ :: v_dual_mov_b32 v239, v29 /*v285*/
	v_dual_mov_b32 v240, v29 /*v285*/ :: v_dual_mov_b32 v241, v29 /*v285*/
	v_dual_mov_b32 v242, v29 /*v285*/ :: v_dual_mov_b32 v243, v29 /*v285*/
	v_dual_mov_b32 v244, v29 /*v285*/ :: v_dual_mov_b32 v245, v29 /*v285*/
	v_dual_mov_b32 v246, v29 /*v285*/ :: v_dual_mov_b32 v247, v29 /*v285*/
	v_dual_mov_b32 v248, v29 /*v285*/ :: v_dual_mov_b32 v249, v29 /*v285*/
	v_dual_mov_b32 v250, v29 /*v285*/ :: v_dual_mov_b32 v251, v29 /*v285*/
	v_dual_mov_b32 v252, v29 /*v285*/ :: v_dual_mov_b32 v253, v29 /*v285*/
	v_dual_mov_b32 v254, v29 /*v285*/ :: v_dual_mov_b32 v255, v29 /*v285*/
	v_mov_b32_e32 v130, v29 /*v285*/
	s_set_vgpr_msb 0x141                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	v_dual_mov_b32 v3 /*v259*/, v29 /*v285*/ :: v_dual_mov_b32 v4 /*v260*/, v29 /*v285*/
	v_dual_mov_b32 v5 /*v261*/, v29 /*v285*/ :: v_dual_mov_b32 v6 /*v262*/, v29 /*v285*/
	v_dual_mov_b32 v7 /*v263*/, v29 /*v285*/ :: v_dual_mov_b32 v8 /*v264*/, v29 /*v285*/
	v_dual_mov_b32 v9 /*v265*/, v29 /*v285*/ :: v_dual_mov_b32 v10 /*v266*/, v29 /*v285*/
	v_dual_mov_b32 v11 /*v267*/, v29 /*v285*/ :: v_dual_mov_b32 v12 /*v268*/, v29 /*v285*/
	v_dual_mov_b32 v13 /*v269*/, v29 /*v285*/ :: v_dual_mov_b32 v14 /*v270*/, v29 /*v285*/
	v_dual_mov_b32 v15 /*v271*/, v29 /*v285*/ :: v_dual_mov_b32 v16 /*v272*/, v29 /*v285*/
	v_dual_mov_b32 v17 /*v273*/, v29 /*v285*/ :: v_dual_mov_b32 v18 /*v274*/, v29 /*v285*/
	v_dual_mov_b32 v19 /*v275*/, v29 /*v285*/ :: v_dual_mov_b32 v20 /*v276*/, v29 /*v285*/
	v_dual_mov_b32 v21 /*v277*/, v29 /*v285*/ :: v_dual_mov_b32 v22 /*v278*/, v29 /*v285*/
	v_dual_mov_b32 v23 /*v279*/, v29 /*v285*/ :: v_dual_mov_b32 v24 /*v280*/, v29 /*v285*/
	v_mov_b32_e32 v25 /*v281*/, v29 /*v285*/
	s_set_vgpr_msb 0x4101                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_dual_mov_b32 v131, v29 /*v285*/ :: v_dual_mov_b32 v132, v29 /*v285*/
	v_dual_mov_b32 v133, v29 /*v285*/ :: v_dual_mov_b32 v134, v29 /*v285*/
	v_dual_mov_b32 v135, v29 /*v285*/ :: v_dual_mov_b32 v136, v29 /*v285*/
	v_mov_b32_e32 v137, v29 /*v285*/
	s_lshr_b32 s47, s29, 16
	s_lshr_b32 s48, s3, 16
	s_lshl_b64 s[4:5], s[4:5], 1
	s_movk_i32 s16, 0x140
	s_bitset1_b32 s47, 22
	s_movk_i32 s8, 0xe0
	s_bitset1_b32 s48, 22
	s_add_nc_u64 s[20:21], s[14:15], s[4:5]
	s_mov_b32 s4, 0x7510000
	s_mov_b32 s49, s11
	s_set_vgpr_msb 0x100                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_branch .LBB0_17
.LBB0_16:                               ;   in Loop: Header=BB0_17 Depth=1
	s_cmp_eq_u32 s49, s45
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_scc1 .LBB0_37
.LBB0_17:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_19 Depth 2
                                        ;     Child Loop BB0_22 Depth 2
                                        ;     Child Loop BB0_24 Depth 2
                                        ;     Child Loop BB0_27 Depth 2
	s_and_b32 s50, s49, 1
	s_add_co_i32 s49, s49, 1
	s_xor_b32 s5, s50, 1
	s_lshl_b32 s2, s49, 6
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_sub_co_i32 s2, s26, s2
	s_min_i32 s2, s2, 64
	s_cmp_lt_i32 s49, s45
	s_cselect_b32 s10, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_and_b32 s6, s10, exec_lo
	s_cselect_b32 s2, s2, 0
	s_cmp_lt_i32 s2, 64
	s_cselect_b32 s6, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b32 s6, s46, s6
	s_or_b32 s6, s41, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s6
	s_cbranch_vccnz .LBB0_29
; %bb.18:                               ;   in Loop: Header=BB0_17 Depth=1
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_mov_b64_e32 v[80:81] /*v[336:337]*/, v[0:1]
	s_set_vgpr_msb 0x4041                   ;  msbs: dst=1 src0=1 src1=0 src2=0
	v_mov_b32_e32 v45 /*v301*/, v37 /*v293*/
	s_cmp_lg_u32 s5, 0
	s_mov_b32 s12, 0
	s_cselect_b32 s7, s44, s31
	s_cselect_b32 s6, s43, 0
	s_set_vgpr_msb 0x4100                   ;  msbs: dst=0 src0=0 src1=0 src2=0
.LBB0_19:                               ;   Parent Loop BB0_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_set_vgpr_msb 0x45                     ;  msbs: dst=1 src0=1 src1=1 src2=0
	v_dual_mov_b32 v28 /*v284*/, v80 /*v336*/ :: v_dual_add_nc_u32 v45 /*v301*/, -2, v45 /*v301*/
	v_dual_mov_b32 v82 /*v338*/, v81 /*v337*/ :: v_dual_mov_b32 v83 /*v339*/, v29 /*v285*/
	v_add_nc_u32_e32 v81 /*v337*/, 0x200, v81 /*v337*/
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u64 v[84:85] /*v[340:341]*/, v[28:29] /*v[284:285]*/, 2, s[6:7]
	v_cmp_eq_u32_e32 vcc_lo, 0, v45 /*v301*/
	v_add_nc_u32_e32 v80 /*v336*/, 0x200, v80 /*v336*/
	v_lshl_add_u64 v[82:83] /*v[338:339]*/, v[82:83] /*v[338:339]*/, 2, s[6:7]
	s_clause 0x1
	flat_store_b32 v[84:85] /*v[340:341]*/, v29 /*v285*/
	flat_store_b32 v[82:83] /*v[338:339]*/, v29 /*v285*/
	s_or_b32 s12, vcc_lo, s12
	s_wait_xcnt 0x0
	s_and_not1_b32 exec_lo, exec_lo, s12
	s_set_vgpr_msb 0x4500                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_cbranch_execnz .LBB0_19
; %bb.20:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s12
	s_and_saveexec_b32 s12, s0
	s_cbranch_execz .LBB0_23
; %bb.21:                               ;   in Loop: Header=BB0_17 Depth=1
	s_set_vgpr_msb 0x45                     ;  msbs: dst=1 src0=1 src1=1 src2=0
	v_add_nc_u64_e32 v[80:81] /*v[336:337]*/, s[6:7], v[76:77] /*v[332:333]*/
	v_mov_b32_e32 v28 /*v284*/, v41 /*v297*/
	s_mov_b32 s6, 0
	s_set_vgpr_msb 0x4500                   ;  msbs: dst=0 src0=0 src1=0 src2=0
.LBB0_22:                               ;   Parent Loop BB0_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_set_vgpr_msb 0x45                     ;  msbs: dst=1 src0=1 src1=1 src2=0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v28 /*v284*/, 0x100, v28 /*v284*/
	flat_store_b32 v[80:81] /*v[336:337]*/, v29 /*v285*/
	s_wait_xcnt 0x0
	v_add_nc_u64_e32 v[80:81] /*v[336:337]*/, 0x400, v[80:81] /*v[336:337]*/
	v_cmp_lt_u32_e32 vcc_lo, 0x1cbf, v28 /*v284*/
	s_or_b32 s6, vcc_lo, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s6
	s_set_vgpr_msb 0x4500                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_cbranch_execnz .LBB0_22
.LBB0_23:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s12
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_mov_b64_e32 v[80:81] /*v[336:337]*/, v[0:1]
	s_set_vgpr_msb 0x4041                   ;  msbs: dst=1 src0=1 src1=0 src2=0
	v_mov_b32_e32 v45 /*v301*/, v39 /*v295*/
	s_cmp_lg_u32 s5, 0
	s_mov_b32 s12, 0
	s_cselect_b32 s7, s35, s42
	s_cselect_b32 s6, s34, s30
	s_set_vgpr_msb 0x4100                   ;  msbs: dst=0 src0=0 src1=0 src2=0
.LBB0_24:                               ;   Parent Loop BB0_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_set_vgpr_msb 0x45                     ;  msbs: dst=1 src0=1 src1=1 src2=0
	v_dual_mov_b32 v28 /*v284*/, v80 /*v336*/ :: v_dual_add_nc_u32 v45 /*v301*/, -2, v45 /*v301*/
	v_dual_mov_b32 v82 /*v338*/, v81 /*v337*/ :: v_dual_mov_b32 v83 /*v339*/, v29 /*v285*/
	v_add_nc_u32_e32 v81 /*v337*/, 0x200, v81 /*v337*/
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u64 v[84:85] /*v[340:341]*/, v[28:29] /*v[284:285]*/, 2, s[6:7]
	v_cmp_eq_u32_e32 vcc_lo, 0, v45 /*v301*/
	v_add_nc_u32_e32 v80 /*v336*/, 0x200, v80 /*v336*/
	v_lshl_add_u64 v[82:83] /*v[338:339]*/, v[82:83] /*v[338:339]*/, 2, s[6:7]
	s_clause 0x1
	flat_store_b32 v[84:85] /*v[340:341]*/, v29 /*v285*/
	flat_store_b32 v[82:83] /*v[338:339]*/, v29 /*v285*/
	s_or_b32 s12, vcc_lo, s12
	s_wait_xcnt 0x0
	s_and_not1_b32 exec_lo, exec_lo, s12
	s_set_vgpr_msb 0x4500                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_cbranch_execnz .LBB0_24
; %bb.25:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s12
	s_and_saveexec_b32 s12, s1
	s_cbranch_execz .LBB0_28
; %bb.26:                               ;   in Loop: Header=BB0_17 Depth=1
	s_set_vgpr_msb 0x45                     ;  msbs: dst=1 src0=1 src1=1 src2=0
	v_add_nc_u64_e32 v[80:81] /*v[336:337]*/, s[6:7], v[78:79] /*v[334:335]*/
	v_mov_b32_e32 v28 /*v284*/, v43 /*v299*/
	s_mov_b32 s6, 0
	s_set_vgpr_msb 0x4500                   ;  msbs: dst=0 src0=0 src1=0 src2=0
.LBB0_27:                               ;   Parent Loop BB0_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_set_vgpr_msb 0x45                     ;  msbs: dst=1 src0=1 src1=1 src2=0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v28 /*v284*/, 0x100, v28 /*v284*/
	flat_store_b32 v[80:81] /*v[336:337]*/, v29 /*v285*/
	s_wait_xcnt 0x0
	v_add_nc_u64_e32 v[80:81] /*v[336:337]*/, 0x400, v[80:81] /*v[336:337]*/
	v_cmp_lt_u32_e32 vcc_lo, 0x297f, v28 /*v284*/
	s_or_b32 s6, vcc_lo, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s6
	s_set_vgpr_msb 0x4500                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_cbranch_execnz .LBB0_27
.LBB0_28:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s12
	s_barrier_signal -1
	s_barrier_wait -1
.LBB0_29:                               ;   in Loop: Header=BB0_17 Depth=1
	s_and_b32 s6, s10, exec_lo
	s_cselect_b32 s6, s49, 0
	s_mov_b32 s7, exec_lo
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_cmpx_lt_i32_e32 0, v35 /*v291*/
	s_xor_b32 s7, exec_lo, s7
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_cbranch_execnz .LBB0_32
; %bb.30:                               ;   in Loop: Header=BB0_17 Depth=1
	s_and_not1_saveexec_b32 s12, s7
	s_cbranch_execnz .LBB0_35
.LBB0_31:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s12
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s25
	s_cbranch_vccnz .LBB0_16
	s_branch .LBB0_36
.LBB0_32:                               ;   in Loop: Header=BB0_17 Depth=1
	s_mov_b32 s51, exec_lo
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_cmpx_eq_u32_e32 1, v35 /*v291*/
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_cbranch_execz .LBB0_34
; %bb.33:                               ;   in Loop: Header=BB0_17 Depth=1
	s_cmp_lg_u32 s5, 0
	s_mov_b32 s28, s2
	s_cselect_b32 s14, s34, s30
	s_cmp_gt_i32 s2, 0
	s_mov_b32 s17, s9
	s_cselect_b32 s15, -1, 0
	s_lshl_b32 s10, s6, 6
	s_mov_b32 s18, s11
	s_lshl_b64 s[12:13], s[10:11], 1
	s_mov_b32 s19, s11
	s_add_nc_u64 s[12:13], s[20:21], s[12:13]
	s_set_vgpr_msb 0x41                     ;  msbs: dst=1 src0=1 src1=0 src2=0
	v_dual_mov_b32 v45 /*v301*/, s14 :: v_dual_mov_b32 v80 /*v336*/, s12
	s_and_b32 s10, s13, 0x1ffffff
	s_and_b32 s13, s40, s15
	s_bitset1_b32 s10, 31
	v_cndmask_b32_e64 v28 /*v284*/, 0, 1, s13
	v_mov_b32_e32 v47 /*v303*/, s10
	v_readfirstlane_b32 s53, v45 /*v301*/
	v_readfirstlane_b32 s54, v80 /*v336*/
	s_lshr_b64 s[14:15], s[28:29], 16
	v_readfirstlane_b32 s52, v28 /*v284*/
	v_readfirstlane_b32 s55, v47 /*v303*/
	s_lshl_b32 s13, s2, 16
	s_mov_b32 s12, s4
	s_mov_b32 s15, s47
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[52:55], s[12:19]
	s_set_vgpr_msb 0x4100                   ;  msbs: dst=0 src0=0 src1=0 src2=0
.LBB0_34:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s51
	s_and_not1_saveexec_b32 s12, s7
	s_cbranch_execz .LBB0_31
.LBB0_35:                               ;   in Loop: Header=BB0_17 Depth=1
	s_cmp_lg_u32 s5, 0
	s_cselect_b32 s5, s43, 0
	s_cmp_gt_i32 s2, 0
	s_cselect_b32 s13, -1, 0
	s_lshl_b32 s10, s6, 6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_lshl_b64 s[6:7], s[10:11], 1
	s_and_b32 s10, s39, s13
	s_add_nc_u64 s[6:7], s[36:37], s[6:7]
	s_set_vgpr_msb 0x41                     ;  msbs: dst=1 src0=1 src1=0 src2=0
	v_cndmask_b32_e64 v28 /*v284*/, 0, 1, s10
	s_and_b32 s7, s7, 0x1ffffff
	v_dual_mov_b32 v45 /*v301*/, s5 :: v_dual_mov_b32 v80 /*v336*/, s6
	s_bitset1_b32 s7, 31
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_readfirstlane_b32 s52, v28 /*v284*/
	v_mov_b32_e32 v47 /*v303*/, s7
	v_readfirstlane_b32 s53, v45 /*v301*/
	v_readfirstlane_b32 s54, v80 /*v336*/
	s_lshr_b64 s[6:7], s[2:3], 16
	s_lshl_b32 s5, s2, 16
	v_readfirstlane_b32 s55, v47 /*v303*/
	s_mov_b32 s7, s48
	s_mov_b32 s10, s11
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	tensor_load_to_lds s[52:55], s[4:11]
	s_or_b32 exec_lo, exec_lo, s12
	s_and_not1_b32 vcc_lo, exec_lo, s25
	s_set_vgpr_msb 0x4100                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_cbranch_vccnz .LBB0_16
.LBB0_36:                               ;   in Loop: Header=BB0_17 Depth=1
	s_cmp_lg_u32 s50, 0
	s_cselect_b32 s5, s43, 0
	s_cselect_b32 s2, s34, s30
	s_set_vgpr_msb 0x55                     ;  msbs: dst=1 src0=1 src1=1 src2=1
	v_lshl_add_u32 v28 /*v284*/, v26 /*v282*/, 1, s5
	v_lshl_add_u32 v45 /*v301*/, v32 /*v288*/, 1, s5
	v_lshl_add_u32 v47 /*v303*/, v34 /*v290*/, 1, s5
	v_lshl_add_u32 v49 /*v305*/, v36 /*v292*/, 1, s5
	ds_load_b128 v[80:83] /*v[336:339]*/, v28 /*v284*/
	ds_load_b128 v[84:87] /*v[340:343]*/, v28 /*v284*/ offset:16
	ds_load_b128 v[88:91] /*v[344:347]*/, v45 /*v301*/ offset:2048
	ds_load_b128 v[92:95] /*v[348:351]*/, v45 /*v301*/ offset:2064
	v_lshl_add_u32 v28 /*v284*/, v30 /*v286*/, 1, s2
	v_lshl_add_u32 v45 /*v301*/, v38 /*v294*/, 1, s5
	ds_load_b128 v[96:99] /*v[352:355]*/, v47 /*v303*/ offset:4096
	ds_load_b128 v[100:103] /*v[356:359]*/, v47 /*v303*/ offset:4112
	v_lshl_add_u32 v47 /*v303*/, v40 /*v296*/, 1, s5
	ds_load_b128 v[112:115] /*v[368:371]*/, v28 /*v284*/
	ds_load_b128 v[116:119] /*v[372:375]*/, v28 /*v284*/ offset:16
	ds_load_b128 v[120:123] /*v[376:379]*/, v45 /*v301*/ offset:8192
	v_lshl_add_u32 v28 /*v284*/, v42 /*v298*/, 1, s5
	ds_load_b128 v[124:127] /*v[380:383]*/, v45 /*v301*/ offset:8208
	v_lshl_add_u32 v45 /*v301*/, v44 /*v300*/, 1, s2
	ds_load_b128 v[128:131] /*v[384:387]*/, v47 /*v303*/ offset:10240
	ds_load_b128 v[132:135] /*v[388:391]*/, v47 /*v303*/ offset:10256
	ds_load_b128 v[136:139] /*v[392:395]*/, v28 /*v284*/ offset:12288
	ds_load_b128 v[140:143] /*v[396:399]*/, v28 /*v284*/ offset:12304
	v_lshl_add_u32 v28 /*v284*/, v46 /*v302*/, 1, s2
	ds_load_b128 v[144:147] /*v[400:403]*/, v45 /*v301*/ offset:2048
	v_lshl_add_u32 v47 /*v303*/, v48 /*v304*/, 1, s2
	ds_load_b128 v[148:151] /*v[404:407]*/, v45 /*v301*/ offset:2064
	v_lshl_add_u32 v45 /*v301*/, v50 /*v306*/, 1, s2
	ds_load_b128 v[152:155] /*v[408:411]*/, v28 /*v284*/ offset:4096
	ds_load_b128 v[156:159] /*v[412:415]*/, v28 /*v284*/ offset:4112
	ds_load_b128 v[160:163] /*v[416:419]*/, v47 /*v303*/ offset:6144
	ds_load_b128 v[164:167] /*v[420:423]*/, v47 /*v303*/ offset:6160
	ds_load_b128 v[168:171] /*v[424:427]*/, v45 /*v301*/ offset:8192
	ds_load_b128 v[172:175] /*v[428:431]*/, v45 /*v301*/ offset:8208
	ds_load_b128 v[104:107] /*v[360:363]*/, v49 /*v305*/ offset:6144
	ds_load_b128 v[108:111] /*v[364:367]*/, v49 /*v305*/ offset:6160
	v_lshl_add_u32 v28 /*v284*/, v52 /*v308*/, 1, s5
	s_wait_dscnt 0x10
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[274:281]*/, v[80:87] /*v[336:343]*/, v[112:119] /*v[368:375]*/, v[18:25] /*v[274:281]*/
	v_lshl_add_u32 v45 /*v301*/, v54 /*v310*/, 1, s5
	v_lshl_add_u32 v47 /*v303*/, v58 /*v314*/, 1, s5
	; sched_group_barrier mask(0x00000100) size(24) SyncID(0)
	s_wait_dscnt 0x8
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[266:273]*/, v[80:87] /*v[336:343]*/, v[144:151] /*v[400:407]*/, v[10:17] /*v[266:273]*/ matrix_b_reuse
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[80:87] /*v[336:343]*/, v[152:159] /*v[408:415]*/, v[2:9] /*v[258:265]*/ matrix_a_reuse
	s_set_vgpr_msb 0x5505                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[250:257], v[80:87] /*v[336:343]*/, v[160:167] /*v[416:423]*/, v[250:257] matrix_b_reuse
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[242:249], v[80:87] /*v[336:343]*/, v[168:175] /*v[424:431]*/, v[242:249] matrix_a_reuse
	s_set_vgpr_msb 0x541                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	ds_load_b128 v[80:83] /*v[336:339]*/, v28 /*v284*/ offset:64
	ds_load_b128 v[84:87] /*v[340:343]*/, v28 /*v284*/ offset:80
	v_lshl_add_u32 v28 /*v284*/, v56 /*v312*/, 1, s5
	s_set_vgpr_msb 0x4105                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[234:241], v[88:95] /*v[344:351]*/, v[112:119] /*v[368:375]*/, v[234:241] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[194:201], v[96:103] /*v[352:359]*/, v[112:119] /*v[368:375]*/, v[194:201] matrix_b_reuse
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[154:161], v[104:111] /*v[360:367]*/, v[112:119] /*v[368:375]*/, v[154:161] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[106:113], v[120:127] /*v[376:383]*/, v[112:119] /*v[368:375]*/, v[106:113] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[66:73], v[128:135] /*v[384:391]*/, v[112:119] /*v[368:375]*/, v[66:73] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[26:33], v[136:143] /*v[392:399]*/, v[112:119] /*v[368:375]*/, v[26:33] matrix_b_reuse
	s_set_vgpr_msb 0x541                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	ds_load_b128 v[112:115] /*v[368:371]*/, v45 /*v301*/ offset:64
	ds_load_b128 v[116:119] /*v[372:375]*/, v45 /*v301*/ offset:80
	v_lshl_add_u32 v45 /*v301*/, v62 /*v318*/, 1, s5
	s_set_vgpr_msb 0x4105                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[226:233], v[88:95] /*v[344:351]*/, v[144:151] /*v[400:407]*/, v[226:233] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[218:225], v[88:95] /*v[344:351]*/, v[152:159] /*v[408:415]*/, v[218:225] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[210:217], v[88:95] /*v[344:351]*/, v[160:167] /*v[416:423]*/, v[210:217] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[202:209], v[88:95] /*v[344:351]*/, v[168:175] /*v[424:431]*/, v[202:209] matrix_b_reuse
	s_set_vgpr_msb 0x541                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	ds_load_b128 v[88:91] /*v[344:347]*/, v28 /*v284*/ offset:64
	ds_load_b128 v[92:95] /*v[348:351]*/, v28 /*v284*/ offset:80
	v_lshl_add_u32 v28 /*v284*/, v60 /*v316*/, 1, s5
	s_set_vgpr_msb 0x4105                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[18:25], v[136:143] /*v[392:399]*/, v[144:151] /*v[400:407]*/, v[18:25] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[58:65], v[128:135] /*v[384:391]*/, v[144:151] /*v[400:407]*/, v[58:65] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[98:105], v[120:127] /*v[376:383]*/, v[144:151] /*v[400:407]*/, v[98:105] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[146:153], v[104:111] /*v[360:367]*/, v[144:151] /*v[400:407]*/, v[146:153] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[186:193], v[96:103] /*v[352:359]*/, v[144:151] /*v[400:407]*/, v[186:193] matrix_b_reuse
	s_set_vgpr_msb 0x541                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	ds_load_b128 v[144:147] /*v[400:403]*/, v47 /*v303*/ offset:64
	ds_load_b128 v[148:151] /*v[404:407]*/, v47 /*v303*/ offset:80
	v_lshl_add_u32 v47 /*v303*/, v68 /*v324*/, 1, s2
	ds_load_b128 v[176:179] /*v[432:435]*/, v47 /*v303*/ offset:64
	s_set_vgpr_msb 0x4105                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[178:185], v[96:103] /*v[352:359]*/, v[152:159] /*v[408:415]*/, v[178:185] matrix_b_reuse
	s_set_vgpr_msb 0x541                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	ds_load_b128 v[180:183] /*v[436:439]*/, v47 /*v303*/ offset:80
	v_lshl_add_u32 v47 /*v303*/, v74 /*v330*/, 1, s2
	ds_load_b128 v[192:195] /*v[448:451]*/, v47 /*v303*/ offset:64
	s_set_vgpr_msb 0x4105                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[138:145], v[104:111] /*v[360:367]*/, v[152:159] /*v[408:415]*/, v[138:145] matrix_b_reuse
	s_set_vgpr_msb 0x541                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	ds_load_b128 v[196:199] /*v[452:455]*/, v47 /*v303*/ offset:80
	s_set_vgpr_msb 0x4105                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[90:97], v[120:127] /*v[376:383]*/, v[152:159] /*v[408:415]*/, v[90:97] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[50:57], v[128:135] /*v[384:391]*/, v[152:159] /*v[408:415]*/, v[50:57] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[10:17], v[136:143] /*v[392:399]*/, v[152:159] /*v[408:415]*/, v[10:17] matrix_b_reuse
	s_set_vgpr_msb 0x541                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	ds_load_b128 v[152:155] /*v[408:411]*/, v45 /*v301*/ offset:64
	ds_load_b128 v[156:159] /*v[412:415]*/, v45 /*v301*/ offset:80
	v_lshl_add_u32 v45 /*v301*/, v66 /*v322*/, 1, s2
	s_set_vgpr_msb 0x4105                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[170:177], v[96:103] /*v[352:359]*/, v[160:167] /*v[416:423]*/, v[170:177] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[162:169], v[96:103] /*v[352:359]*/, v[168:175] /*v[424:431]*/, v[162:169] matrix_b_reuse
	s_set_vgpr_msb 0x541                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	ds_load_b128 v[96:99] /*v[352:355]*/, v28 /*v284*/ offset:64
	ds_load_b128 v[100:103] /*v[356:359]*/, v28 /*v284*/ offset:80
	v_lshl_add_u32 v28 /*v284*/, v64 /*v320*/, 1, s5
	s_set_vgpr_msb 0x4105                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[2:9], v[136:143] /*v[392:399]*/, v[160:167] /*v[416:423]*/, v[2:9] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[42:49], v[128:135] /*v[384:391]*/, v[160:167] /*v[416:423]*/, v[42:49] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[82:89], v[120:127] /*v[376:383]*/, v[160:167] /*v[416:423]*/, v[82:89] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[122:129], v[104:111] /*v[360:367]*/, v[160:167] /*v[416:423]*/, v[122:129] matrix_b_reuse
	s_set_vgpr_msb 0x541                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	ds_load_b128 v[160:163] /*v[416:419]*/, v45 /*v301*/ offset:64
	ds_load_b128 v[164:167] /*v[420:423]*/, v45 /*v301*/ offset:80
	v_lshl_add_u32 v45 /*v301*/, v72 /*v328*/, 1, s2
	ds_load_b128 v[184:187] /*v[440:443]*/, v45 /*v301*/ offset:64
	s_set_vgpr_msb 0x4105                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[114:121], v[104:111] /*v[360:367]*/, v[168:175] /*v[424:431]*/, v[114:121] matrix_b_reuse
	s_set_vgpr_msb 0x541                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	ds_load_b128 v[104:107] /*v[360:363]*/, v28 /*v284*/ offset:64
	ds_load_b128 v[108:111] /*v[364:367]*/, v28 /*v284*/ offset:80
	v_lshl_add_u32 v28 /*v284*/, v70 /*v326*/, 1, s2
	ds_load_b128 v[188:191] /*v[444:447]*/, v45 /*v301*/ offset:80
	s_set_vgpr_msb 0x4105                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[74:81], v[120:127] /*v[376:383]*/, v[168:175] /*v[424:431]*/, v[74:81] matrix_b_reuse
	s_set_vgpr_msb 0x541                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	ds_load_b128 v[120:123] /*v[376:379]*/, v28 /*v284*/ offset:64
	ds_load_b128 v[124:127] /*v[380:383]*/, v28 /*v284*/ offset:80
	s_set_vgpr_msb 0x4105                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[34:41], v[128:135] /*v[384:391]*/, v[168:175] /*v[424:431]*/, v[34:41] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[130:137], v[136:143] /*v[392:399]*/, v[168:175] /*v[424:431]*/, v[130:137] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(35) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x555                    ;  msbs: dst=1 src0=1 src1=1 src2=1
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[274:281]*/, v[80:87] /*v[336:343]*/, v[160:167] /*v[416:423]*/, v[18:25] /*v[274:281]*/
	; sched_group_barrier mask(0x00000100) size(24) SyncID(0)
	s_set_vgpr_msb 0x5505                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[234:241], v[112:119] /*v[368:375]*/, v[160:167] /*v[416:423]*/, v[234:241] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[194:201], v[88:95] /*v[344:351]*/, v[160:167] /*v[416:423]*/, v[194:201] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[154:161], v[144:151] /*v[400:407]*/, v[160:167] /*v[416:423]*/, v[154:161] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[106:113], v[96:103] /*v[352:359]*/, v[160:167] /*v[416:423]*/, v[106:113] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[66:73], v[152:159] /*v[408:415]*/, v[160:167] /*v[416:423]*/, v[66:73] matrix_b_reuse
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x32_bf16 v[26:33], v[104:111] /*v[360:367]*/, v[160:167] /*v[416:423]*/, v[26:33] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[18:25], v[104:111] /*v[360:367]*/, v[176:183] /*v[432:439]*/, v[18:25] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[58:65], v[152:159] /*v[408:415]*/, v[176:183] /*v[432:439]*/, v[58:65] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[98:105], v[96:103] /*v[352:359]*/, v[176:183] /*v[432:439]*/, v[98:105] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[146:153], v[144:151] /*v[400:407]*/, v[176:183] /*v[432:439]*/, v[146:153] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[186:193], v[88:95] /*v[344:351]*/, v[176:183] /*v[432:439]*/, v[186:193] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[226:233], v[112:119] /*v[368:375]*/, v[176:183] /*v[432:439]*/, v[226:233] matrix_b_reuse
	s_set_vgpr_msb 0x555                    ;  msbs: dst=1 src0=1 src1=1 src2=1
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[266:273]*/, v[80:87] /*v[336:343]*/, v[176:183] /*v[432:439]*/, v[10:17] /*v[266:273]*/ matrix_b_reuse
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[80:87] /*v[336:343]*/, v[120:127] /*v[376:383]*/, v[2:9] /*v[258:265]*/ matrix_a_reuse
	s_set_vgpr_msb 0x5505                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[218:225], v[112:119] /*v[368:375]*/, v[120:127] /*v[376:383]*/, v[218:225] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[178:185], v[88:95] /*v[344:351]*/, v[120:127] /*v[376:383]*/, v[178:185] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[138:145], v[144:151] /*v[400:407]*/, v[120:127] /*v[376:383]*/, v[138:145] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[90:97], v[96:103] /*v[352:359]*/, v[120:127] /*v[376:383]*/, v[90:97] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[50:57], v[152:159] /*v[408:415]*/, v[120:127] /*v[376:383]*/, v[50:57] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[10:17], v[104:111] /*v[360:367]*/, v[120:127] /*v[376:383]*/, v[10:17] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[2:9], v[104:111] /*v[360:367]*/, v[184:191] /*v[440:447]*/, v[2:9] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[42:49], v[152:159] /*v[408:415]*/, v[184:191] /*v[440:447]*/, v[42:49] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[82:89], v[96:103] /*v[352:359]*/, v[184:191] /*v[440:447]*/, v[82:89] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[122:129], v[144:151] /*v[400:407]*/, v[184:191] /*v[440:447]*/, v[122:129] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[170:177], v[88:95] /*v[344:351]*/, v[184:191] /*v[440:447]*/, v[170:177] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[210:217], v[112:119] /*v[368:375]*/, v[184:191] /*v[440:447]*/, v[210:217] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[250:257], v[80:87] /*v[336:343]*/, v[184:191] /*v[440:447]*/, v[250:257] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[242:249], v[80:87] /*v[336:343]*/, v[192:199] /*v[448:455]*/, v[242:249] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[202:209], v[112:119] /*v[368:375]*/, v[192:199] /*v[448:455]*/, v[202:209] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[162:169], v[88:95] /*v[344:351]*/, v[192:199] /*v[448:455]*/, v[162:169] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[114:121], v[144:151] /*v[400:407]*/, v[192:199] /*v[448:455]*/, v[114:121] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[74:81], v[96:103] /*v[352:359]*/, v[192:199] /*v[448:455]*/, v[74:81] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[34:41], v[152:159] /*v[408:415]*/, v[192:199] /*v[448:455]*/, v[34:41] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[130:137], v[104:111] /*v[360:367]*/, v[192:199] /*v[448:455]*/, v[130:137] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(35) SyncID(0)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_wait_dscnt 0
	;;#ASMEND
	s_set_vgpr_msb 0x500                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_branch .LBB0_16
.LBB0_37:
	s_wait_tensorcnt 0x0
	;;#ASMSTART
	s_wait_dscnt 0
	;;#ASMEND
	s_and_b32 vcc_lo, exec_lo, s25
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_vccz .LBB0_39
; %bb.38:
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_mul_u32_u24_e32 v1, 0x70, v33 /*v289*/
	s_set_vgpr_msb 0x444                    ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_mul_u32_u24_e32 v26 /*v282*/, 0x50, v31 /*v287*/
	s_set_vgpr_msb 0x4440                   ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_lshrrev_b32_e32 v28 /*v284*/, 1, v0
	s_set_vgpr_msb 0x4045                   ;  msbs: dst=1 src0=1 src1=1 src2=0
	v_cvt_pk_bf16_f32 v17 /*v273*/, v16 /*v272*/, v17 /*v273*/
	v_cvt_pk_bf16_f32 v16 /*v272*/, v14 /*v270*/, v15 /*v271*/
	v_cvt_pk_bf16_f32 v15 /*v271*/, v12 /*v268*/, v13 /*v269*/
	s_set_vgpr_msb 0x4550                   ;  msbs: dst=1 src0=0 src1=0 src2=1
	v_and_or_b32 v26 /*v282*/, v0, 15, v26 /*v282*/
	s_set_vgpr_msb 0x5001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_and_or_b32 v1, v28 /*v284*/, 8, v1
	s_set_vgpr_msb 0x145                    ;  msbs: dst=1 src0=1 src1=1 src2=0
	v_cvt_pk_bf16_f32 v25 /*v281*/, v24 /*v280*/, v25 /*v281*/
	v_cvt_pk_bf16_f32 v24 /*v280*/, v22 /*v278*/, v23 /*v279*/
	v_cvt_pk_bf16_f32 v23 /*v279*/, v20 /*v276*/, v21 /*v277*/
	v_cvt_pk_bf16_f32 v22 /*v278*/, v18 /*v274*/, v19 /*v275*/
	s_set_vgpr_msb 0x4504                   ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_mad_u32_u24 v1, 0xe0, v26 /*v282*/, v1
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_cvt_pk_bf16_f32 v65, v64, v65
	v_cvt_pk_bf16_f32 v64, v62, v63
	v_cvt_pk_bf16_f32 v63, v60, v61
	s_set_vgpr_msb 0x45                     ;  msbs: dst=1 src0=1 src1=1 src2=0
	v_cvt_pk_bf16_f32 v9 /*v265*/, v8 /*v264*/, v9 /*v265*/
	s_set_vgpr_msb 0x4540                   ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_lshrrev_b32_e32 v12 /*v268*/, 3, v1
	v_add_nc_u32_e32 v13 /*v269*/, 0xe00, v1
	v_add_nc_u32_e32 v14 /*v270*/, 0x1c00, v1
	v_lshlrev_b32_e32 v18 /*v274*/, 1, v1
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_add_nc_u32_e32 v60, 0x2a50, v1
	s_set_vgpr_msb 0x45                     ;  msbs: dst=1 src0=1 src1=1 src2=0
	v_and_b32_e32 v12 /*v268*/, 0x3ff0, v12 /*v268*/
	v_dual_lshrrev_b32 v13 /*v269*/, 3, v13 /*v269*/ :: v_dual_lshrrev_b32 v20 /*v276*/, 3, v14 /*v270*/
	v_cvt_pk_bf16_f32 v14 /*v270*/, v10 /*v266*/, v11 /*v267*/
	v_cvt_pk_bf16_f32 v8 /*v264*/, v6 /*v262*/, v7 /*v263*/
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_nc_u32_e32 v11 /*v267*/, v12 /*v268*/, v18 /*v274*/
	v_and_b32_e32 v12 /*v268*/, 0xffffff0, v13 /*v269*/
	v_and_b32_e32 v13 /*v269*/, 0xffffff0, v20 /*v276*/
	v_cvt_pk_bf16_f32 v7 /*v263*/, v4 /*v260*/, v5 /*v261*/
	v_cvt_pk_bf16_f32 v1 /*v257*/, v0 /*v256*/, v1 /*v257*/
	ds_store_b128 v11 /*v267*/, v[22:25] /*v[278:281]*/
	v_dual_add_nc_u32 v11 /*v267*/, v12 /*v268*/, v18 /*v274*/ :: v_dual_add_nc_u32 v12 /*v268*/, v13 /*v269*/, v18 /*v274*/
	s_set_vgpr_msb 0x4540                   ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_add_nc_u32_e32 v13 /*v269*/, 0x3800, v1
	v_cvt_pk_bf16_f32 v0 /*v256*/, v254, v255
	v_add_nc_u32_e32 v4 /*v260*/, 16, v1
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_cvt_pk_bf16_f32 v255, v252, v253
	v_add_nc_u32_e32 v252, 0xe10, v1
	v_cvt_pk_bf16_f32 v241, v240, v241
	v_cvt_pk_bf16_f32 v240, v238, v239
	v_cvt_pk_bf16_f32 v239, v236, v237
	v_add_nc_u32_e32 v236, 0x1c10, v1
	v_cvt_pk_bf16_f32 v233, v232, v233
	v_cvt_pk_bf16_f32 v232, v230, v231
	v_cvt_pk_bf16_f32 v231, v228, v229
	v_cvt_pk_bf16_f32 v225, v224, v225
	v_cvt_pk_bf16_f32 v224, v222, v223
	v_add_nc_u32_e32 v228, 0x3810, v1
	v_cvt_pk_bf16_f32 v223, v220, v221
	v_cvt_pk_bf16_f32 v217, v216, v217
	v_cvt_pk_bf16_f32 v216, v214, v215
	v_add_nc_u32_e32 v220, 32, v1
	v_cvt_pk_bf16_f32 v215, v212, v213
	v_add_nc_u32_e32 v212, 0xe20, v1
	v_cvt_pk_bf16_f32 v201, v200, v201
	v_cvt_pk_bf16_f32 v200, v198, v199
	v_cvt_pk_bf16_f32 v199, v196, v197
	v_add_nc_u32_e32 v196, 0x1c20, v1
	v_cvt_pk_bf16_f32 v193, v192, v193
	v_cvt_pk_bf16_f32 v192, v190, v191
	v_cvt_pk_bf16_f32 v191, v188, v189
	v_cvt_pk_bf16_f32 v185, v184, v185
	v_cvt_pk_bf16_f32 v184, v182, v183
	v_add_nc_u32_e32 v188, 0x3820, v1
	v_cvt_pk_bf16_f32 v183, v180, v181
	v_cvt_pk_bf16_f32 v177, v176, v177
	v_cvt_pk_bf16_f32 v176, v174, v175
	v_add_nc_u32_e32 v180, 48, v1
	v_cvt_pk_bf16_f32 v175, v172, v173
	v_add_nc_u32_e32 v172, 0xe30, v1
	v_cvt_pk_bf16_f32 v161, v160, v161
	v_cvt_pk_bf16_f32 v160, v158, v159
	v_cvt_pk_bf16_f32 v159, v156, v157
	v_add_nc_u32_e32 v156, 0x1c30, v1
	v_cvt_pk_bf16_f32 v153, v152, v153
	v_cvt_pk_bf16_f32 v152, v150, v151
	v_cvt_pk_bf16_f32 v151, v148, v149
	v_cvt_pk_bf16_f32 v145, v144, v145
	v_cvt_pk_bf16_f32 v144, v142, v143
	v_add_nc_u32_e32 v148, 0x3830, v1
	v_cvt_pk_bf16_f32 v143, v140, v141
	v_cvt_pk_bf16_f32 v129, v128, v129
	v_cvt_pk_bf16_f32 v128, v126, v127
	v_add_nc_u32_e32 v140, 64, v1
	v_cvt_pk_bf16_f32 v127, v124, v125
	v_add_nc_u32_e32 v124, 0xe40, v1
	v_cvt_pk_bf16_f32 v113, v112, v113
	v_cvt_pk_bf16_f32 v112, v110, v111
	v_cvt_pk_bf16_f32 v111, v108, v109
	v_add_nc_u32_e32 v108, 0x1c40, v1
	v_add_nc_u32_e32 v109, 0x2a40, v1
	v_cvt_pk_bf16_f32 v97, v96, v97
	v_cvt_pk_bf16_f32 v96, v94, v95
	v_cvt_pk_bf16_f32 v95, v92, v93
	v_add_nc_u32_e32 v92, 0x3840, v1
	v_cvt_pk_bf16_f32 v89, v88, v89
	v_cvt_pk_bf16_f32 v88, v86, v87
	v_cvt_pk_bf16_f32 v87, v84, v85
	v_add_nc_u32_e32 v84, 0x50, v1
	v_add_nc_u32_e32 v85, 0xe50, v1
	v_cvt_pk_bf16_f32 v73, v72, v73
	v_cvt_pk_bf16_f32 v72, v70, v71
	v_cvt_pk_bf16_f32 v71, v68, v69
	v_add_nc_u32_e32 v68, 0x1c50, v1
	v_cvt_pk_bf16_f32 v57, v56, v57
	v_cvt_pk_bf16_f32 v56, v54, v55
	v_cvt_pk_bf16_f32 v55, v52, v53
	v_add_nc_u32_e32 v52, 0x60, v1
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_add_nc_u32_e32 v19 /*v275*/, 0x2a00, v1
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_add_nc_u32_e32 v237, 0x2a10, v1
	v_add_nc_u32_e32 v197, 0x2a20, v1
	v_add_nc_u32_e32 v157, 0x2a30, v1
	v_cvt_pk_bf16_f32 v62, v58, v59
	v_lshrrev_b32_e32 v59, 3, v60
	v_add_nc_u32_e32 v60, 0x3850, v1
	v_cvt_pk_bf16_f32 v49, v48, v49
	v_cvt_pk_bf16_f32 v48, v46, v47
	v_cvt_pk_bf16_f32 v47, v44, v45
	v_add_nc_u32_e32 v44, 0xe60, v1
	v_cvt_pk_bf16_f32 v41, v40, v41
	v_cvt_pk_bf16_f32 v40, v38, v39
	v_cvt_pk_bf16_f32 v39, v36, v37
	v_add_nc_u32_e32 v36, 0x1c60, v1
	v_cvt_pk_bf16_f32 v33, v32, v33
	v_cvt_pk_bf16_f32 v32, v30, v31
	v_cvt_pk_bf16_f32 v31, v28, v29
	v_add_nc_u32_e32 v28, 0x2a60, v1
	v_add_nc_u32_e32 v1, 0x3860, v1
	s_set_vgpr_msb 0x45                     ;  msbs: dst=1 src0=1 src1=1 src2=0
	v_cvt_pk_bf16_f32 v6 /*v262*/, v2 /*v258*/, v3 /*v259*/
	v_lshrrev_b32_e32 v3 /*v259*/, 3, v13 /*v269*/
	s_set_vgpr_msb 0x4500                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_cvt_pk_bf16_f32 v254, v250, v251
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_lshrrev_b32_e32 v251, 3, v4 /*v260*/
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_cvt_pk_bf16_f32 v249, v248, v249
	v_cvt_pk_bf16_f32 v248, v246, v247
	v_cvt_pk_bf16_f32 v246, v242, v243
	v_lshrrev_b32_e32 v243, 3, v252
	v_lshrrev_b32_e32 v236, 3, v236
	v_cvt_pk_bf16_f32 v222, v218, v219
	v_lshrrev_b32_e32 v219, 3, v228
	v_cvt_pk_bf16_f32 v214, v210, v211
	v_lshrrev_b32_e32 v211, 3, v220
	v_cvt_pk_bf16_f32 v209, v208, v209
	v_cvt_pk_bf16_f32 v208, v206, v207
	v_cvt_pk_bf16_f32 v206, v202, v203
	v_lshrrev_b32_e32 v203, 3, v212
	v_lshrrev_b32_e32 v196, 3, v196
	v_cvt_pk_bf16_f32 v182, v178, v179
	v_lshrrev_b32_e32 v179, 3, v188
	v_cvt_pk_bf16_f32 v174, v170, v171
	v_lshrrev_b32_e32 v171, 3, v180
	v_cvt_pk_bf16_f32 v169, v168, v169
	v_cvt_pk_bf16_f32 v168, v166, v167
	v_cvt_pk_bf16_f32 v166, v162, v163
	v_lshrrev_b32_e32 v163, 3, v172
	v_lshrrev_b32_e32 v156, 3, v156
	v_cvt_pk_bf16_f32 v142, v138, v139
	v_lshrrev_b32_e32 v139, 3, v148
	v_cvt_pk_bf16_f32 v126, v122, v123
	v_lshrrev_b32_e32 v123, 3, v140
	v_cvt_pk_bf16_f32 v121, v120, v121
	v_cvt_pk_bf16_f32 v120, v118, v119
	v_cvt_pk_bf16_f32 v118, v114, v115
	v_lshrrev_b32_e32 v115, 3, v124
	v_dual_lshrrev_b32 v108, 3, v108 :: v_dual_lshrrev_b32 v109, 3, v109
	v_cvt_pk_bf16_f32 v94, v90, v91
	v_lshrrev_b32_e32 v91, 3, v92
	v_cvt_pk_bf16_f32 v86, v82, v83
	v_dual_lshrrev_b32 v83, 3, v84 :: v_dual_lshrrev_b32 v85, 3, v85
	v_cvt_pk_bf16_f32 v70, v66, v67
	v_lshrrev_b32_e32 v67, 3, v68
	v_cvt_pk_bf16_f32 v46, v42, v43
	v_lshrrev_b32_e32 v43, 3, v52
	s_set_vgpr_msb 0x44                     ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_lshrrev_b32_e32 v10 /*v266*/, 3, v19 /*v275*/
	s_set_vgpr_msb 0x4400                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshrrev_b32_e32 v237, 3, v237
	v_lshrrev_b32_e32 v197, 3, v197
	v_dual_lshrrev_b32 v157, 3, v157 :: v_dual_lshrrev_b32 v60, 3, v60
	v_cvt_pk_bf16_f32 v38, v34, v35
	v_lshrrev_b32_e32 v35, 3, v44
	v_cvt_pk_bf16_f32 v30, v26, v27
	v_lshrrev_b32_e32 v26, 3, v36
	v_cvt_pk_bf16_f32 v25, v24, v25
	v_cvt_pk_bf16_f32 v24, v22, v23
	v_cvt_pk_bf16_f32 v23, v20, v21
	v_dual_lshrrev_b32 v20, 3, v28 :: v_dual_lshrrev_b32 v1, 3, v1
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_and_b32_e32 v250, 0xffffff0, v3 /*v259*/
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_and_b32_e32 v251, 0x7ff0, v251
	v_cvt_pk_bf16_f32 v238, v234, v235
	v_and_b32_e32 v234, 0xffffff0, v243
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_lshlrev_b32_e32 v235, 1, v4 /*v260*/
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_and_b32_e32 v236, 0xffffff0, v236
	v_and_b32_e32 v210, 0xffffff0, v219
	v_and_b32_e32 v211, 0x7ff0, v211
	v_cvt_pk_bf16_f32 v198, v194, v195
	v_and_b32_e32 v194, 0xffffff0, v203
	v_lshlrev_b32_e32 v195, 1, v220
	v_and_b32_e32 v196, 0xffffff0, v196
	v_and_b32_e32 v170, 0xffffff0, v179
	v_and_b32_e32 v171, 0x7ff0, v171
	v_cvt_pk_bf16_f32 v158, v154, v155
	v_and_b32_e32 v154, 0xffffff0, v163
	v_lshlrev_b32_e32 v155, 1, v180
	v_and_b32_e32 v156, 0xffffff0, v156
	v_and_b32_e32 v122, 0xffffff0, v139
	v_and_b32_e32 v123, 0x7ff0, v123
	v_cvt_pk_bf16_f32 v110, v106, v107
	v_and_b32_e32 v106, 0xffffff0, v115
	v_lshlrev_b32_e32 v107, 1, v140
	v_cvt_pk_bf16_f32 v105, v104, v105
	v_and_b32_e32 v108, 0xffffff0, v108
	v_cvt_pk_bf16_f32 v104, v102, v103
	v_cvt_pk_bf16_f32 v102, v98, v99
	v_and_b32_e32 v99, 0xffffff0, v109
	v_and_b32_e32 v82, 0xffffff0, v91
	v_cvt_pk_bf16_f32 v81, v80, v81
	v_and_b32_e32 v83, 0x7ff0, v83
	v_cvt_pk_bf16_f32 v80, v78, v79
	v_cvt_pk_bf16_f32 v79, v76, v77
	v_cvt_pk_bf16_f32 v78, v74, v75
	v_and_b32_e32 v75, 0xffffff0, v85
	v_lshlrev_b32_e32 v76, 1, v84
	v_and_b32_e32 v58, 0xffffff0, v67
	v_and_b32_e32 v59, 0xffffff0, v59
	v_and_b32_e32 v34, 0x7ff0, v43
	s_set_vgpr_msb 0x44                     ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_and_b32_e32 v10 /*v266*/, 0xffffff0, v10 /*v266*/
	s_set_vgpr_msb 0x4400                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_cvt_pk_bf16_f32 v230, v226, v227
	v_and_b32_e32 v227, 0xffffff0, v237
	v_cvt_pk_bf16_f32 v190, v186, v187
	v_and_b32_e32 v187, 0xffffff0, v197
	v_cvt_pk_bf16_f32 v150, v146, v147
	v_and_b32_e32 v147, 0xffffff0, v157
	v_cvt_pk_bf16_f32 v54, v50, v51
	v_and_b32_e32 v51, 0xffffff0, v60
	v_and_b32_e32 v35, 0xffffff0, v35
	v_lshlrev_b32_e32 v37, 1, v52
	v_and_b32_e32 v26, 0xffffff0, v26
	v_cvt_pk_bf16_f32 v22, v18, v19
	v_and_b32_e32 v19, 0xffffff0, v20
	v_and_b32_e32 v1, 0xffffff0, v1
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_add_nc_u32_e32 v250, v250, v18 /*v274*/
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_cvt_pk_bf16_f32 v247, v244, v245
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_add_nc_u32_e32 v242, v251, v18 /*v274*/
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_dual_add_nc_u32 v234, v234, v235 :: v_dual_add_nc_u32 v226, v236, v235
	v_add_nc_u32_e32 v210, v210, v235
	v_cvt_pk_bf16_f32 v207, v204, v205
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_add_nc_u32_e32 v202, v211, v18 /*v274*/
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_dual_add_nc_u32 v194, v194, v195 :: v_dual_add_nc_u32 v186, v196, v195
	v_add_nc_u32_e32 v170, v170, v195
	v_cvt_pk_bf16_f32 v167, v164, v165
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_add_nc_u32_e32 v162, v171, v18 /*v274*/
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_dual_add_nc_u32 v154, v154, v155 :: v_dual_add_nc_u32 v146, v156, v155
	v_add_nc_u32_e32 v122, v122, v155
	v_cvt_pk_bf16_f32 v119, v116, v117
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_add_nc_u32_e32 v114, v123, v18 /*v274*/
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_add_nc_u32_e32 v106, v106, v107
	v_cvt_pk_bf16_f32 v103, v100, v101
	v_dual_add_nc_u32 v98, v108, v107 :: v_dual_add_nc_u32 v90, v99, v107
	v_add_nc_u32_e32 v82, v82, v107
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_add_nc_u32_e32 v74, v83, v18 /*v274*/
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_dual_add_nc_u32 v66, v75, v76 :: v_dual_add_nc_u32 v58, v58, v76
	v_add_nc_u32_e32 v50, v59, v76
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_add_nc_u32_e32 v34, v34, v18 /*v274*/
	s_set_vgpr_msb 0x445                    ;  msbs: dst=1 src0=1 src1=1 src2=0
	v_add_nc_u32_e32 v2 /*v258*/, v10 /*v266*/, v18 /*v274*/
	s_set_vgpr_msb 0x4500                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_add_nc_u32_e32 v218, v227, v235
	v_add_nc_u32_e32 v178, v187, v195
	v_add_nc_u32_e32 v138, v147, v155
	v_add_nc_u32_e32 v42, v51, v76
	v_dual_add_nc_u32 v27, v35, v37 :: v_dual_add_nc_u32 v18, v26, v37
	v_cvt_pk_bf16_f32 v17, v16, v17
	v_cvt_pk_bf16_f32 v16, v14, v15
	v_cvt_pk_bf16_f32 v15, v12, v13
	v_cvt_pk_bf16_f32 v14, v10, v11
	v_add_nc_u32_e32 v10, v19, v37
	v_cvt_pk_bf16_f32 v9, v8, v9
	v_cvt_pk_bf16_f32 v8, v6, v7
	v_cvt_pk_bf16_f32 v7, v4, v5
	v_cvt_pk_bf16_f32 v6, v2, v3
	v_add_nc_u32_e32 v1, v1, v37
	v_cvt_pk_bf16_f32 v5, v136, v137
	v_cvt_pk_bf16_f32 v4, v134, v135
	v_cvt_pk_bf16_f32 v3, v132, v133
	v_cvt_pk_bf16_f32 v2, v130, v131
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	ds_store_b128 v11 /*v267*/, v[14:17] /*v[270:273]*/ offset:7168
	ds_store_b128 v12 /*v268*/, v[6:9] /*v[262:265]*/ offset:14336
	s_set_vgpr_msb 0x501                    ;  msbs: dst=0 src0=1 src1=0 src2=0
	ds_store_b128 v2 /*v258*/, v[254:257] offset:21504
	s_set_vgpr_msb 0x100                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	ds_store_b128 v250, v[246:249] offset:28672
	ds_store_b128 v242, v[238:241] offset:32
	ds_store_b128 v234, v[230:233] offset:7168
	ds_store_b128 v226, v[222:225] offset:14336
	ds_store_b128 v218, v[214:217] offset:21504
	ds_store_b128 v210, v[206:209] offset:28672
	ds_store_b128 v202, v[198:201] offset:64
	ds_store_b128 v194, v[190:193] offset:7168
	ds_store_b128 v186, v[182:185] offset:14336
	ds_store_b128 v178, v[174:177] offset:21504
	ds_store_b128 v170, v[166:169] offset:28672
	ds_store_b128 v162, v[158:161] offset:96
	ds_store_b128 v154, v[150:153] offset:7168
	ds_store_b128 v146, v[142:145] offset:14336
	ds_store_b128 v138, v[126:129] offset:21504
	ds_store_b128 v122, v[118:121] offset:28672
	ds_store_b128 v114, v[110:113] offset:128
	ds_store_b128 v106, v[102:105] offset:7168
	ds_store_b128 v98, v[94:97] offset:14336
	ds_store_b128 v90, v[86:89] offset:21504
	ds_store_b128 v82, v[78:81] offset:28672
	ds_store_b128 v74, v[70:73] offset:160
	ds_store_b128 v66, v[62:65] offset:7168
	ds_store_b128 v58, v[54:57] offset:14336
	ds_store_b128 v50, v[46:49] offset:21504
	ds_store_b128 v42, v[38:41] offset:28672
	ds_store_b128 v34, v[30:33] offset:192
	ds_store_b128 v27, v[22:25] offset:7168
	ds_store_b128 v18, v[14:17] offset:14336
	ds_store_b128 v10, v[6:9] offset:21504
	ds_store_b128 v1, v[2:5] offset:28672
.LBB0_39:
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_cmp_ne_u32_e32 vcc_lo, 1, v27 /*v283*/
	;;#ASMSTART
	s_wait_dscnt 0
	;;#ASMEND
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_vccnz .LBB0_50
; %bb.40:
	s_mul_i32 s3, s29, s3
	s_mov_b32 s0, exec_lo
	v_cmpx_gt_i32_e64 s3, v0
	s_cbranch_execz .LBB0_50
; %bb.41:
	s_mul_i32 s0, s38, 0xe0
	v_xad_u32 v2, v0, -1, s3
	s_ashr_i32 s1, s0, 31
	s_wait_kmcnt 0x0
	s_mul_i32 s14, s33, 0x140
	s_lshl_b64 s[0:1], s[0:1], 1
	s_ashr_i32 s25, s24, 31
	s_add_nc_u64 s[4:5], s[22:23], s[0:1]
	s_mov_b32 s0, 0
                                        ; implicit-def: $vgpr1
                                        ; implicit-def: $vgpr6
                                        ; implicit-def: $sgpr12_sgpr13
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_u32_e32 0x2ff, v2
	s_xor_b32 s15, exec_lo, s1
	s_cbranch_execnz .LBB0_44
; %bb.42:
	s_or_saveexec_b32 s1, s15
	v_mov_b64_e32 v[2:3], s[12:13]
	s_xor_b32 exec_lo, exec_lo, s1
	s_cbranch_execnz .LBB0_47
.LBB0_43:
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 exec_lo, exec_lo, s0
	s_cbranch_execnz .LBB0_48
	s_branch .LBB0_50
.LBB0_44:
	s_abs_i32 s16, s27
	v_lshrrev_b32_e32 v1, 8, v2
	s_cvt_f32_u32 s0, s16
	v_or_b32_e32 v3, 0x300, v0
	s_sub_co_i32 s1, 0, s16
	v_mov_b32_e32 v7, 0
	v_rcp_iflag_f32_e32 v2, s0
	v_add_nc_u32_e32 v8, 1, v1
	v_or_b32_e32 v1, 0x100, v0
	s_mov_b32 s13, 0
	s_mov_b32 s17, s14
	s_mov_b32 s18, s14
	v_and_b32_e32 v9, 0x1fffffc, v8
	s_mov_b32 s19, s14
	v_readfirstlane_b32 s0, v2
	v_or_b32_e32 v2, 0x200, v0
	s_mov_b32 s6, s24
	v_mov_b32_e32 v10, v9
	s_mov_b32 s7, s25
	s_mul_f32 s0, s0, 0x4f7ffffe
	v_mov_b64_e32 v[4:5], v[2:3]
	v_mov_b64_e32 v[2:3], v[0:1]
	s_mov_b32 s8, s24
	s_cvt_u32_f32 s0, s0
	s_mov_b32 s9, s25
	s_mov_b32 s10, s24
	s_mov_b32 s11, s25
	s_mul_i32 s1, s1, s0
	s_mov_b32 s20, s27
	s_mul_hi_u32 s1, s0, s1
	s_mov_b32 s21, s27
	s_mov_b32 s22, s27
	s_ashr_i32 s23, s27, 31
	s_add_co_i32 s12, s0, s1
	s_mov_b32 s26, s13
.LBB0_45:                               ; =>This Inner Loop Header: Depth=1
	v_dual_sub_nc_u32 v6, 0, v2 :: v_dual_sub_nc_u32 v12, 0, v3
	v_dual_sub_nc_u32 v16, 0, v4 :: v_dual_sub_nc_u32 v19, 0, v5
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_ashrrev_i32 v1, 31, v2 :: v_dual_max_i32 v6, v6, v2
	v_max_i32_e32 v12, v12, v3
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_max_i32_e32 v16, v16, v4
	v_dual_ashrrev_i32 v11, 31, v3 :: v_dual_max_i32 v19, v19, v5
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mul_hi_u32 v20, v6, s12
	v_mul_hi_u32 v21, v12, s12
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mul_hi_u32 v22, v16, s12
	v_mul_hi_u32 v23, v19, s12
	v_dual_ashrrev_i32 v14, 31, v4 :: v_dual_ashrrev_i32 v18, 31, v5
	v_xor_b32_e32 v1, s23, v1
	v_xor_b32_e32 v11, s23, v11
	v_mul_lo_u32 v24, v20, s16
	v_mul_lo_u32 v26, v21, s16
	v_mul_lo_u32 v27, v22, s16
	v_dual_add_nc_u32 v25, 1, v20 :: v_dual_add_nc_u32 v29, 1, v21
	v_mul_lo_u32 v28, v23, s16
	v_dual_add_nc_u32 v30, 1, v22 :: v_dual_add_nc_u32 v31, 1, v23
	v_dual_sub_nc_u32 v6, v6, v24 :: v_dual_sub_nc_u32 v12, v12, v26
	v_dual_sub_nc_u32 v16, v16, v27 :: v_dual_bitop2_b32 v14, s23, v14 bitop3:0x14
	v_xor_b32_e32 v18, s23, v18
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cmp_le_u32_e32 vcc_lo, s16, v6
	v_cmp_le_u32_e64 s0, s16, v12
	v_subrev_nc_u32_e32 v24, s16, v6
	v_cmp_le_u32_e64 s1, s16, v16
	v_subrev_nc_u32_e32 v26, s16, v16
	v_cndmask_b32_e32 v20, v20, v25, vcc_lo
	v_subrev_nc_u32_e32 v25, s16, v12
	v_dual_cndmask_b32 v21, v21, v29, s0 :: v_dual_sub_nc_u32 v19, v19, v28
	v_cndmask_b32_e64 v22, v22, v30, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_cndmask_b32 v6, v6, v24, vcc_lo :: v_dual_cndmask_b32 v12, v12, v25, s0
	v_dual_add_nc_u32 v25, 1, v21 :: v_dual_cndmask_b32 v16, v16, v26, s1
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_cmp_le_u32_e64 s2, s16, v19
	v_subrev_nc_u32_e32 v27, s16, v19
	v_cmp_le_u32_e32 vcc_lo, s16, v12
	v_dual_add_nc_u32 v26, 1, v22 :: v_dual_add_nc_u32 v24, 1, v20
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v23, v23, v31, s2
	v_dual_cndmask_b32 v19, v19, v27, s2 :: v_dual_cndmask_b32 v12, v21, v25, vcc_lo
	v_cmp_le_u32_e32 vcc_lo, s16, v16
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_add_nc_u32 v10, -4, v10 :: v_dual_add_nc_u32 v27, 1, v23
	v_dual_mov_b32 v13, v7 :: v_dual_bitop2_b32 v12, v12, v11 bitop3:0x14
	v_cndmask_b32_e32 v16, v22, v26, vcc_lo
	v_cmp_le_u32_e32 vcc_lo, s16, v6
	v_dual_mov_b32 v15, v7 :: v_dual_mov_b32 v17, v7
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sub_nc_u32_e32 v11, v12, v11
	v_xor_b32_e32 v16, v16, v14
	v_cndmask_b32_e32 v6, v20, v24, vcc_lo
	v_cmp_le_u32_e32 vcc_lo, s16, v19
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mul_lo_u32 v12, v11, s20
	v_dual_sub_nc_u32 v26, v16, v14 :: v_dual_bitop2_b32 v6, v6, v1 bitop3:0x14
	v_cndmask_b32_e32 v19, v23, v27, vcc_lo
	v_add_nc_u32_e32 v20, s17, v11
	v_cmp_eq_u32_e32 vcc_lo, 0, v10
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mul_lo_u32 v14, v26, s21
	v_dual_sub_nc_u32 v1, v6, v1 :: v_dual_bitop2_b32 v19, v19, v18 bitop3:0x14
	v_dual_sub_nc_u32 v12, v3, v12 :: v_dual_add_nc_u32 v22, s18, v26
	v_ashrrev_i32_e32 v21, 31, v20
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mul_lo_u32 v6, v1, s27
	v_dual_sub_nc_u32 v27, v19, v18 :: v_dual_add_nc_u32 v18, s14, v1
	v_sub_nc_u32_e32 v14, v4, v14
	v_mad_u32 v11, 0xe0, v11, v12
	v_ashrrev_i32_e32 v23, 31, v22
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_mul_lo_u32 v16, v27, s22
	v_dual_add_nc_u32 v24, s19, v27 :: v_dual_sub_nc_u32 v6, v2, v6
	v_ashrrev_i32_e32 v19, 31, v18
	v_mad_u32 v26, 0xe0, v26, v14
	v_mul_u64_e32 v[20:21], s[6:7], v[20:21]
	v_ashrrev_i32_e32 v25, 31, v24
	v_mad_u32 v1, 0xe0, v1, v6
	v_sub_nc_u32_e32 v16, v5, v16
	v_mul_u64_e32 v[18:19], s[24:25], v[18:19]
	v_mul_u64_e32 v[22:23], s[8:9], v[22:23]
	v_mul_u64_e32 v[24:25], s[10:11], v[24:25]
	v_ashrrev_i32_e32 v29, 31, v11
	v_mad_u32 v27, 0xe0, v27, v16
	v_dual_ashrrev_i32 v30, 31, v26 :: v_dual_ashrrev_i32 v28, 31, v1
	v_lshlrev_b32_e32 v32, 1, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_lshrrev_b32 v29, 25, v29 :: v_dual_lshlrev_b32 v33, 1, v11
	v_dual_lshrrev_b32 v30, 25, v30 :: v_dual_lshrrev_b32 v28, 25, v28
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_ashrrev_i32 v31, 31, v27 :: v_dual_add_nc_u32 v11, v11, v29
	v_lshlrev_b32_e32 v34, 1, v26
	v_dual_add_nc_u32 v26, v26, v30 :: v_dual_add_nc_u32 v1, v1, v28
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_lshrrev_b32_e32 v31, 25, v31
	v_lshlrev_b32_e32 v35, 1, v27
	v_dual_ashrrev_i32 v11, 7, v11 :: v_dual_ashrrev_i32 v26, 7, v26
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_ashrrev_i32 v1, 7, v1 :: v_dual_add_nc_u32 v27, v27, v31
	v_add_nc_u32_e32 v5, 0x400, v5
	v_lshl_add_u32 v11, v11, 4, v33
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u32 v26, v26, 4, v34
	v_lshl_add_u32 v1, v1, 4, v32
	v_ashrrev_i32_e32 v27, 7, v27
	v_add_nc_u32_e32 v4, 0x400, v4
	v_add_nc_u32_e32 v3, 0x400, v3
	v_add_nc_u32_e32 v2, 0x400, v2
	s_or_b32 s26, vcc_lo, s26
	v_lshl_add_u32 v27, v27, 4, v35
	ds_load_u16 v1, v1
	ds_load_u16 v11, v11
	ds_load_u16 v26, v26
	ds_load_u16 v27, v27
	v_lshl_add_u64 v[20:21], v[20:21], 1, s[4:5]
	v_lshl_add_u64 v[18:19], v[18:19], 1, s[4:5]
	v_lshl_add_u64 v[22:23], v[22:23], 1, s[4:5]
	v_lshl_add_u64 v[24:25], v[24:25], 1, s[4:5]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[20:21]
	v_lshl_add_u64 v[18:19], v[6:7], 1, v[18:19]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u64 v[14:15], v[14:15], 1, v[22:23]
	v_lshl_add_u64 v[16:17], v[16:17], 1, v[24:25]
	s_wait_dscnt 0x3
	global_store_b16 v[18:19], v1, off
	s_wait_dscnt 0x2
	global_store_b16 v[12:13], v11, off
	s_wait_dscnt 0x1
	global_store_b16 v[14:15], v26, off
	s_wait_dscnt 0x0
	global_store_b16 v[16:17], v27, off
	s_wait_xcnt 0x0
	s_and_not1_b32 exec_lo, exec_lo, s26
	s_cbranch_execnz .LBB0_45
; %bb.46:
	s_or_b32 exec_lo, exec_lo, s26
	v_cmp_ne_u32_e32 vcc_lo, v8, v9
	v_lshl_or_b32 v0, v9, 8, v0
	v_dual_mov_b32 v6, s16 :: v_dual_mov_b32 v1, s23
	s_and_b32 s0, vcc_lo, exec_lo
	s_or_saveexec_b32 s1, s15
	v_mov_b64_e32 v[2:3], s[12:13]
	s_xor_b32 exec_lo, exec_lo, s1
	s_cbranch_execz .LBB0_43
.LBB0_47:
	s_abs_i32 s2, s27
	s_ashr_i32 s8, s27, 31
	s_cvt_f32_u32 s6, s2
	s_sub_co_i32 s7, 0, s2
	v_mov_b32_e32 v6, s2
	s_or_b32 s0, s0, exec_lo
	v_rcp_iflag_f32_e32 v1, s6
	v_nop
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_3)
	v_readfirstlane_b32 s6, v1
	v_mov_b32_e32 v1, s8
	s_mul_f32 s6, s6, 0x4f7ffffe
	s_cvt_u32_f32 s6, s6
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_i32 s7, s7, s6
	s_mul_hi_u32 s9, s6, s7
	s_mov_b32 s7, 0
	s_add_co_i32 s6, s6, s9
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_mov_b64_e32 v[2:3], s[6:7]
	s_or_b32 exec_lo, exec_lo, s1
	s_and_b32 exec_lo, exec_lo, s0
	s_cbranch_execz .LBB0_50
.LBB0_48:
	v_mov_b32_e32 v5, 0
	s_mov_b32 s0, 0
	s_sub_co_i32 s1, 0, s27
.LBB0_49:                               ; =>This Inner Loop Header: Depth=1
	v_sub_nc_u32_e32 v4, 0, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_i32_e32 v4, v4, v0
	v_mul_u64_e32 v[8:9], v[4:5], v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v7, v9, v6
	v_dual_add_nc_u32 v8, 1, v9 :: v_dual_sub_nc_u32 v4, v4, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_sub_nc_u32_e32 v7, v4, v6
	v_cmp_ge_u32_e32 vcc_lo, v4, v6
	v_dual_cndmask_b32 v8, v9, v8, vcc_lo :: v_dual_cndmask_b32 v4, v4, v7, vcc_lo
	v_ashrrev_i32_e32 v9, 31, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_nc_u32_e32 v7, 1, v8
	v_cmp_ge_u32_e32 vcc_lo, v4, v6
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_cndmask_b32 v4, v8, v7, vcc_lo :: v_dual_bitop2_b32 v9, v9, v1 bitop3:0x14
	v_xor_b32_e32 v4, v4, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_sub_nc_u32_e32 v7, v4, v9
	v_mul_lo_u32 v4, 0xe0, v4
	v_mul_lo_u32 v9, 0xe0, v9
	v_mul_lo_u32 v8, v7, s27
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_sub_nc_u32 v4, v4, v8 :: v_dual_add_nc_u32 v8, s14, v7
	v_dual_sub_nc_u32 v4, v4, v9 :: v_dual_ashrrev_i32 v9, 31, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_nc_u32_e32 v4, v0, v4
	v_mul_u64_e32 v[8:9], s[24:25], v[8:9]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v10, 31, v4
	v_lshrrev_b32_e32 v10, 25, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_add_nc_u32 v10, v4, v10 :: v_dual_lshlrev_b32 v4, 1, v4
	v_ashrrev_i32_e32 v10, 7, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v4, v10, 4, v4
	ds_load_u16 v10, v4
	v_mad_u32 v4, s1, v7, v0
	v_add_nc_u32_e32 v0, 0x100, v0
	v_cmp_le_i32_e32 vcc_lo, s3, v0
	v_lshl_add_u64 v[8:9], v[8:9], 1, s[4:5]
	s_or_b32 s0, vcc_lo, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u64 v[8:9], v[4:5], 1, v[8:9]
	s_wait_dscnt 0x0
	global_store_b16 v[8:9], v10, off
	s_wait_xcnt 0x0
	s_and_not1_b32 exec_lo, exec_lo, s0
	s_cbranch_execnz .LBB0_49
.LBB0_50:
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end0:
	.size	bm224_bn320_bk064_wm2_wn4_mc0, .Lfunc_end0-bm224_bn320_bk064_wm2_wn4_mc0
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel bm224_bn320_bk064_wm2_wn4_mc0
		.amdhsa_group_segment_fixed_size 152320
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 132
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 456
		.amdhsa_next_free_sgpr 56
		.amdhsa_named_barrier_count 0
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size 88
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text.bm224_bn320_bk064_wm2_wn4_mc0,"axG",@progbits,bm224_bn320_bk064_wm2_wn4_mc0,comdat
                                        ; -- End function
	.set .Lbm224_bn320_bk064_wm2_wn4_mc0.num_vgpr, 456
	.set .Lbm224_bn320_bk064_wm2_wn4_mc0.num_agpr, 0
	.set .Lbm224_bn320_bk064_wm2_wn4_mc0.numbered_sgpr, 56
	.set .Lbm224_bn320_bk064_wm2_wn4_mc0.num_named_barrier, 0
	.set .Lbm224_bn320_bk064_wm2_wn4_mc0.private_seg_size, 0
	.set .Lbm224_bn320_bk064_wm2_wn4_mc0.uses_vcc, 1
	.set .Lbm224_bn320_bk064_wm2_wn4_mc0.uses_flat_scratch, 1
	.set .Lbm224_bn320_bk064_wm2_wn4_mc0.has_dyn_sized_stack, 0
	.set .Lbm224_bn320_bk064_wm2_wn4_mc0.has_recursion, 0
	.set .Lbm224_bn320_bk064_wm2_wn4_mc0.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 11160
; TotalNumSgprs: 58
; NumVgprs: 456
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 152320 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 28
; NumSGPRsForWavesPerEU: 58
; NumVGPRsForWavesPerEU: 456
; NamedBarCnt: 0
; Occupancy: 2
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.section	.AMDGPU.csdata,"",@progbits
	.type	__hip_cuid_b6a324a5f3dd7a72,@object ; @__hip_cuid_b6a324a5f3dd7a72
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_b6a324a5f3dd7a72
__hip_cuid_b6a324a5f3dd7a72:
	.byte	0                               ; 0x0
	.size	__hip_cuid_b6a324a5f3dd7a72, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git aa451e1fe6a793394d6733051b1778633063ae96+PATCHED:d17c5aa0e3ea29cde402f58f27e39b6034effa27)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_b6a324a5f3dd7a72
	.amdgpu_metadata
---
custom.config:
  Source:
    Origin: hipkittens
  Version: 1.0.0
  Features:
    SupportsUserArgs: false
    SupportsBias: false
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
    UseBeta: False
    Batched: True
    UseBias: 0
    Activation: False
    UseScaleAlphaVec: 0
  CustomKernel:
    args: [ { type: address, semantic: AddressA },
            { type: uint64, semantic: ConstantOne },
            { type: uint64, semantic: ConstantOne },
            { type: uint64, semantic: SizeFree0 },
            { type: uint64, semantic: SizeSum },
            { type: address, semantic: AddressB },
            { type: uint64, semantic: ConstantOne },
            { type: uint64, semantic: ConstantOne },
            { type: uint64, semantic: SizeFree1 },
            { type: uint64, semantic: SizeSum },
            { type: address, semantic: AddressD },
            { type: uint64, semantic: ConstantOne },
            { type: uint64, semantic: ConstantOne },
            { type: uint64, semantic: SizeFree0 },
            { type: uint64, semantic: SizeFree1 },
            { type: uint32, semantic: SizeFree0 },
            { type: uint32, semantic: SizeFree1 },
            { type: uint32, semantic: SizeSum } ]
    macrotile: [224, 320, 64]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, One]
  MatrixInstruction: [16, 16, 32, 1]
  EnableMatrixInstruction: True
  MIWaveTile: [7, 5]
  WavefrontSize: 32
amdhsa.kernels:
  - .args:
      - .offset:         0
        .size:           120
        .value_kind:     by_value
      - .offset:         120
        .size:           4
        .value_kind:     by_value
      - .offset:         124
        .size:           4
        .value_kind:     by_value
      - .offset:         128
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 152320
    .kernarg_segment_align: 8
    .kernarg_segment_size: 132
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           bm224_bn320_bk064_wm2_wn4_mc0
    .private_segment_fixed_size: 0
    .sgpr_count:     58
    .sgpr_spill_count: 0
    .symbol:         bm224_bn320_bk064_wm2_wn4_mc0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     456
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
