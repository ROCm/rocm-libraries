	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.section	.text.bm224_bn256_bk128_wm2_wn4_mc0,"axG",@progbits,bm224_bn256_bk128_wm2_wn4_mc0,comdat
	.protected	bm224_bn256_bk128_wm2_wn4_mc0 ; -- Begin function bm224_bn256_bk128_wm2_wn4_mc0
	.globl	bm224_bn256_bk128_wm2_wn4_mc0
	.p2align	8
	.type	bm224_bn256_bk128_wm2_wn4_mc0,@function
bm224_bn256_bk128_wm2_wn4_mc0: ; @bm224_bn256_bk128_wm2_wn4_mc0
; %bb.0:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1 ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_mov_b64 s[2:3], src_shared_base
	s_mov_b32 s2, 0xee00
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
	s_cselect_b32 s36, ttmp9, s3
	s_cselect_b32 s5, ttmp7, s5
	s_add_co_i32 s2, s24, 0xdf
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_hi_i32 s3, s2, 0x92492493
	s_add_co_i32 s3, s3, s2
	s_add_co_i32 s2, s25, 0xff
	s_lshr_b32 s6, s3, 31
	s_ashr_i32 s3, s3, 7
	s_ashr_i32 s7, s2, 31
	s_add_co_i32 s6, s3, s6
	s_lshr_b32 s3, s7, 24
	s_mul_i32 s7, s36, 0xffffff20
	s_add_co_i32 s2, s2, s3
	s_add_co_i32 s3, s24, s7
	s_ashr_i32 s7, s2, 8
	s_min_i32 s27, s3, 0xe0
	s_cmp_lt_i32 s36, s6
	s_cselect_b32 s37, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_and_b32 s2, s37, exec_lo
	s_cselect_b32 s3, s27, 0
	s_lshl_b32 s33, s5, 8
	s_sub_co_i32 s2, s25, s33
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_min_i32 s2, s2, 0x100
	s_cmp_lt_i32 s5, s7
	s_cselect_b32 s38, -1, 0
	s_and_b32 s8, s38, exec_lo
	s_cselect_b32 s29, s2, 0
	s_add_co_i32 s17, s26, 0x7f
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 2, 1), 1
	s_min_i32 s2, s26, 0x80
	s_cmp_gt_i32 s17, 0x7f
	s_cselect_b32 s16, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_and_b32 s8, s16, exec_lo
	s_cselect_b32 s2, s2, 0
	s_cmp_lt_i32 s3, 0xe0
	s_cselect_b32 s39, -1, 0
	s_and_b32 vcc_lo, exec_lo, s39
	s_mov_b32 s8, s39
	s_cbranch_vccnz .LBB0_2
; %bb.1:
	s_cmp_lt_i32 s29, 0x100
	s_cselect_b32 s8, -1, 0
	s_cmp_lt_i32 s2, 0x80
	s_cselect_b32 s10, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 s8, s10, s8
.LBB0_2:
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s8
	s_cbranch_vccnz .LBB0_8
; %bb.3:
	v_dual_mov_b32 v3, 0 :: v_dual_lshlrev_b32 v2, 2, v0
	v_or_b32_e32 v1, 0xffffff00, v0
	s_mov_b32 s8, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v4, v2 :: v_dual_mov_b32 v5, v1
.LBB0_4:                                ; =>This Inner Loop Header: Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)
	v_add_nc_u32_e32 v5, 0x100, v5
	ds_store_b32 v4, v3
	v_add_nc_u32_e32 v4, 0x400, v4
	v_cmp_lt_u32_e32 vcc_lo, 0x3a7f, v5
	s_or_b32 s8, vcc_lo, s8
	s_and_not1_b32 exec_lo, exec_lo, s8
	s_cbranch_execnz .LBB0_4
; %bb.5:
	s_or_b32 exec_lo, exec_lo, s8
	v_lshl_add_u32 v2, s4, 2, v2
	v_mov_b32_e32 v3, 0
	s_mov_b32 s8, 0
.LBB0_6:                                ; =>This Inner Loop Header: Depth=1
	v_add_nc_u32_e32 v1, 0x100, v1
	ds_store_b32 v2, v3 offset:60928
	v_add_nc_u32_e32 v2, 0x400, v2
	v_cmp_lt_u32_e32 vcc_lo, 0x42ff, v1
	s_or_b32 s8, vcc_lo, s8
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s8
	s_cbranch_execnz .LBB0_6
; %bb.7:
	s_or_b32 exec_lo, exec_lo, s8
	s_barrier_signal -1
	s_barrier_wait -1
.LBB0_8:
	s_clause 0x2
	s_load_b64 s[18:19], s[0:1], 0x0 nv
	s_load_b128 s[12:15], s[0:1], 0x20 nv
	s_load_b128 s[20:23], s[0:1], 0x48 nv
	s_wait_xcnt 0x0
	s_mov_b64 s[0:1], src_shared_base
	v_lshrrev_b32_e32 v240, 5, v0
	s_lshl_b32 s0, s4, 2
	s_add_co_i32 s7, s7, -1
	s_or_b32 s40, s0, 0xee00
	s_add_co_i32 s25, s6, -1
	s_min_i32 s0, s5, s7
	s_mov_b32 s4, exec_lo
	v_cmpx_lt_i32_e32 0, v240
	s_xor_b32 s30, exec_lo, s4
	s_cbranch_execz .LBB0_12
; %bb.9:
	s_mov_b32 s31, exec_lo
	v_cmpx_eq_u32_e32 1, v240
	s_cbranch_execz .LBB0_11
; %bb.10:
	s_cmp_gt_i32 s2, 0
	s_mov_b32 s28, s2
	s_cselect_b32 s8, -1, 0
	s_lshl_b32 s4, s0, 8
	s_wait_kmcnt 0x0
	s_bfe_i64 s[6:7], s[20:21], 0x200000
	s_ashr_i32 s5, s4, 31
	s_mov_b32 s10, 0
	s_mul_u64 s[4:5], s[6:7], s[4:5]
	s_mov_b32 s11, s10
	s_lshl_b64 s[4:5], s[4:5], 1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_nc_u64 s[6:7], s[14:15], s[4:5]
	v_dual_mov_b32 v1, s40 :: v_dual_mov_b32 v4, s6
	s_and_b32 s4, s7, 0x1ffffff
	s_and_b32 s7, s38, s8
	s_bitset1_b32 s4, 31
	v_cndmask_b32_e64 v2, 0, 1, s7
	v_mov_b32_e32 v3, s4
	v_readfirstlane_b32 s45, v1
	v_readfirstlane_b32 s46, v4
	s_lshr_b32 s4, s29, 16
	v_readfirstlane_b32 s44, v2
	v_readfirstlane_b32 s47, v3
	s_lshr_b64 s[6:7], s[28:29], 16
	s_lshl_b32 s5, s2, 16
	s_or_b32 s7, s4, 0x800000
	s_movk_i32 s8, 0x100
	s_mov_b32 s4, 0x7510000
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[44:47], s[4:11]
.LBB0_11:
	s_or_b32 exec_lo, exec_lo, s31
.LBB0_12:
	s_or_saveexec_b32 s28, s30
	s_min_i32 s4, s36, s25
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mul_i32 s34, s4, 0xe0
	s_xor_b32 exec_lo, exec_lo, s28
	s_cbranch_execz .LBB0_14
; %bb.13:
	s_cmp_gt_i32 s2, 0
	s_mov_b32 s10, 0
	s_cselect_b32 s6, -1, 0
	s_ashr_i32 s35, s34, 31
	s_wait_kmcnt 0x0
	s_bfe_i64 s[4:5], s[12:13], 0x200000
	s_and_b32 s6, s37, s6
	s_mul_u64 s[4:5], s[4:5], s[34:35]
	v_cndmask_b32_e64 v2, 0, 1, s6
	s_lshl_b64 s[4:5], s[4:5], 1
	s_lshr_b64 s[6:7], s[2:3], 16
	s_add_nc_u64 s[4:5], s[18:19], s[4:5]
	s_movk_i32 s8, 0xe0
	s_and_b32 s5, s5, 0x1ffffff
	v_readfirstlane_b32 s44, v2
	s_bitset1_b32 s5, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v4, s4 :: v_dual_mov_b32 v3, s5
	s_lshr_b32 s4, s3, 16
	s_lshl_b32 s5, s2, 16
	s_or_b32 s7, s4, 0x800000
	v_readfirstlane_b32 s46, v4
	v_readfirstlane_b32 s47, v3
	s_mov_b32 s4, 0x7510000
	s_mov_b32 s11, s10
	s_mov_b32 s45, s10
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[44:47], s[4:11]
.LBB0_14:
	s_or_b32 exec_lo, exec_lo, s28
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	v_dual_lshrrev_b32 v231, 7, v0 :: v_dual_mov_b32 v9, 0
	v_lshlrev_b32_e32 v1, 6, v240
	s_and_b32 s25, s37, s38
	s_and_not1_b32 vcc_lo, exec_lo, s16
	v_cndmask_b32_e64 v227, 0, 1, s25
	v_mov_b32_e32 v8, v9
	v_and_b32_e32 v233, 0xc0, v1
	v_dual_mov_b32 v7, v9 :: v_dual_mov_b32 v6, v9
	v_dual_mov_b32 v5, v9 :: v_dual_mov_b32 v4, v9
	v_dual_mov_b32 v3, v9 :: v_dual_mov_b32 v2, v9
	v_dual_mov_b32 v17, v9 :: v_dual_mov_b32 v16, v9
	v_dual_mov_b32 v15, v9 :: v_dual_mov_b32 v14, v9
	v_dual_mov_b32 v13, v9 :: v_dual_mov_b32 v12, v9
	v_dual_mov_b32 v11, v9 :: v_dual_mov_b32 v10, v9
	v_dual_mov_b32 v25, v9 :: v_dual_mov_b32 v24, v9
	v_dual_mov_b32 v23, v9 :: v_dual_mov_b32 v22, v9
	v_dual_mov_b32 v21, v9 :: v_dual_mov_b32 v20, v9
	v_dual_mov_b32 v19, v9 :: v_dual_mov_b32 v18, v9
	v_dual_mov_b32 v33, v9 :: v_dual_mov_b32 v32, v9
	v_dual_mov_b32 v31, v9 :: v_dual_mov_b32 v30, v9
	v_dual_mov_b32 v29, v9 :: v_dual_mov_b32 v28, v9
	v_dual_mov_b32 v27, v9 :: v_dual_mov_b32 v26, v9
	v_dual_mov_b32 v41, v9 :: v_dual_mov_b32 v40, v9
	v_dual_mov_b32 v39, v9 :: v_dual_mov_b32 v38, v9
	v_dual_mov_b32 v37, v9 :: v_dual_mov_b32 v36, v9
	v_dual_mov_b32 v35, v9 :: v_dual_mov_b32 v34, v9
	v_dual_mov_b32 v49, v9 :: v_dual_mov_b32 v48, v9
	v_dual_mov_b32 v47, v9 :: v_dual_mov_b32 v46, v9
	v_dual_mov_b32 v45, v9 :: v_dual_mov_b32 v44, v9
	v_dual_mov_b32 v43, v9 :: v_dual_mov_b32 v42, v9
	v_dual_mov_b32 v57, v9 :: v_dual_mov_b32 v56, v9
	v_dual_mov_b32 v55, v9 :: v_dual_mov_b32 v54, v9
	v_dual_mov_b32 v53, v9 :: v_dual_mov_b32 v52, v9
	v_dual_mov_b32 v51, v9 :: v_dual_mov_b32 v50, v9
	v_dual_mov_b32 v65, v9 :: v_dual_mov_b32 v64, v9
	v_dual_mov_b32 v63, v9 :: v_dual_mov_b32 v62, v9
	v_dual_mov_b32 v61, v9 :: v_dual_mov_b32 v60, v9
	v_dual_mov_b32 v59, v9 :: v_dual_mov_b32 v58, v9
	v_dual_mov_b32 v73, v9 :: v_dual_mov_b32 v72, v9
	v_dual_mov_b32 v71, v9 :: v_dual_mov_b32 v70, v9
	v_dual_mov_b32 v69, v9 :: v_dual_mov_b32 v68, v9
	v_dual_mov_b32 v67, v9 :: v_dual_mov_b32 v66, v9
	v_dual_mov_b32 v81, v9 :: v_dual_mov_b32 v80, v9
	v_dual_mov_b32 v79, v9 :: v_dual_mov_b32 v78, v9
	v_dual_mov_b32 v77, v9 :: v_dual_mov_b32 v76, v9
	v_dual_mov_b32 v75, v9 :: v_dual_mov_b32 v74, v9
	v_dual_mov_b32 v89, v9 :: v_dual_mov_b32 v88, v9
	v_dual_mov_b32 v87, v9 :: v_dual_mov_b32 v86, v9
	v_dual_mov_b32 v85, v9 :: v_dual_mov_b32 v84, v9
	v_dual_mov_b32 v83, v9 :: v_dual_mov_b32 v82, v9
	v_dual_mov_b32 v97, v9 :: v_dual_mov_b32 v96, v9
	v_dual_mov_b32 v95, v9 :: v_dual_mov_b32 v94, v9
	v_dual_mov_b32 v93, v9 :: v_dual_mov_b32 v92, v9
	v_dual_mov_b32 v91, v9 :: v_dual_mov_b32 v90, v9
	v_dual_mov_b32 v105, v9 :: v_dual_mov_b32 v104, v9
	v_dual_mov_b32 v103, v9 :: v_dual_mov_b32 v102, v9
	v_dual_mov_b32 v101, v9 :: v_dual_mov_b32 v100, v9
	v_dual_mov_b32 v99, v9 :: v_dual_mov_b32 v98, v9
	v_dual_mov_b32 v121, v9 :: v_dual_mov_b32 v120, v9
	v_dual_mov_b32 v119, v9 :: v_dual_mov_b32 v118, v9
	v_dual_mov_b32 v117, v9 :: v_dual_mov_b32 v116, v9
	v_dual_mov_b32 v115, v9 :: v_dual_mov_b32 v114, v9
	v_dual_mov_b32 v129, v9 :: v_dual_mov_b32 v128, v9
	v_dual_mov_b32 v127, v9 :: v_dual_mov_b32 v126, v9
	v_dual_mov_b32 v125, v9 :: v_dual_mov_b32 v124, v9
	v_dual_mov_b32 v123, v9 :: v_dual_mov_b32 v122, v9
	v_dual_mov_b32 v137, v9 :: v_dual_mov_b32 v136, v9
	v_dual_mov_b32 v135, v9 :: v_dual_mov_b32 v134, v9
	v_dual_mov_b32 v133, v9 :: v_dual_mov_b32 v132, v9
	v_dual_mov_b32 v131, v9 :: v_dual_mov_b32 v130, v9
	v_dual_mov_b32 v145, v9 :: v_dual_mov_b32 v144, v9
	v_dual_mov_b32 v143, v9 :: v_dual_mov_b32 v142, v9
	v_dual_mov_b32 v141, v9 :: v_dual_mov_b32 v140, v9
	v_dual_mov_b32 v139, v9 :: v_dual_mov_b32 v138, v9
	v_dual_mov_b32 v153, v9 :: v_dual_mov_b32 v152, v9
	v_dual_mov_b32 v151, v9 :: v_dual_mov_b32 v150, v9
	v_dual_mov_b32 v149, v9 :: v_dual_mov_b32 v148, v9
	v_dual_mov_b32 v147, v9 :: v_dual_mov_b32 v146, v9
	v_dual_mov_b32 v161, v9 :: v_dual_mov_b32 v160, v9
	v_dual_mov_b32 v159, v9 :: v_dual_mov_b32 v158, v9
	v_dual_mov_b32 v157, v9 :: v_dual_mov_b32 v156, v9
	v_dual_mov_b32 v155, v9 :: v_dual_mov_b32 v154, v9
	v_dual_mov_b32 v169, v9 :: v_dual_mov_b32 v168, v9
	v_dual_mov_b32 v167, v9 :: v_dual_mov_b32 v166, v9
	v_dual_mov_b32 v165, v9 :: v_dual_mov_b32 v164, v9
	v_dual_mov_b32 v163, v9 :: v_dual_mov_b32 v162, v9
	v_dual_mov_b32 v177, v9 :: v_dual_mov_b32 v176, v9
	v_dual_mov_b32 v175, v9 :: v_dual_mov_b32 v174, v9
	v_dual_mov_b32 v173, v9 :: v_dual_mov_b32 v172, v9
	v_dual_mov_b32 v171, v9 :: v_dual_mov_b32 v170, v9
	v_dual_mov_b32 v185, v9 :: v_dual_mov_b32 v184, v9
	v_dual_mov_b32 v183, v9 :: v_dual_mov_b32 v182, v9
	v_dual_mov_b32 v181, v9 :: v_dual_mov_b32 v180, v9
	v_dual_mov_b32 v179, v9 :: v_dual_mov_b32 v178, v9
	v_dual_mov_b32 v193, v9 :: v_dual_mov_b32 v192, v9
	v_dual_mov_b32 v191, v9 :: v_dual_mov_b32 v190, v9
	v_dual_mov_b32 v189, v9 :: v_dual_mov_b32 v188, v9
	v_dual_mov_b32 v187, v9 :: v_dual_mov_b32 v186, v9
	v_dual_mov_b32 v201, v9 :: v_dual_mov_b32 v200, v9
	v_dual_mov_b32 v199, v9 :: v_dual_mov_b32 v198, v9
	v_dual_mov_b32 v197, v9 :: v_dual_mov_b32 v196, v9
	v_dual_mov_b32 v195, v9 :: v_dual_mov_b32 v194, v9
	v_dual_mov_b32 v209, v9 :: v_dual_mov_b32 v208, v9
	v_dual_mov_b32 v207, v9 :: v_dual_mov_b32 v206, v9
	v_dual_mov_b32 v205, v9 :: v_dual_mov_b32 v204, v9
	v_dual_mov_b32 v203, v9 :: v_dual_mov_b32 v202, v9
	v_dual_mov_b32 v217, v9 :: v_dual_mov_b32 v216, v9
	v_dual_mov_b32 v215, v9 :: v_dual_mov_b32 v214, v9
	v_dual_mov_b32 v213, v9 :: v_dual_mov_b32 v212, v9
	v_dual_mov_b32 v211, v9 :: v_dual_mov_b32 v210, v9
	v_dual_mov_b32 v225, v9 :: v_dual_mov_b32 v224, v9
	v_dual_mov_b32 v223, v9 :: v_dual_mov_b32 v222, v9
	v_dual_mov_b32 v221, v9 :: v_dual_mov_b32 v220, v9
	v_dual_mov_b32 v219, v9 :: v_dual_mov_b32 v218, v9
	v_dual_mov_b32 v113, v9 :: v_dual_mov_b32 v112, v9
	v_dual_mov_b32 v111, v9 :: v_dual_mov_b32 v110, v9
	v_dual_mov_b32 v109, v9 :: v_dual_mov_b32 v108, v9
	v_dual_mov_b32 v107, v9 :: v_dual_mov_b32 v106, v9
	s_barrier_wait -1
	s_cbranch_vccnz .LBB0_37
; %bb.15:
	s_mov_b64 s[4:5], src_shared_base
	s_add_co_i32 s6, s40, 0x11000
	s_mov_b32 s7, s5
	v_dual_lshlrev_b32 v2, 7, v233 :: v_dual_bitop2_b32 v3, 16, v0 bitop3:0x40
	v_mul_u32_u24_e32 v1, 0x3800, v231
	v_lshlrev_b32_e32 v4, 7, v0
	s_and_b64 s[30:31], s[6:7], 15
	v_dual_mov_b32 v229, 0 :: v_dual_sub_nc_u32 v6, 0x3b7f, v0
	s_sub_co_i32 s2, 16, s30
	s_delay_alu instid0(VALU_DEP_2)
	v_and_or_b32 v3, 0x780, v4, v3
	s_lshr_b32 s2, s2, 2
	s_cmp_lg_u64 s[30:31], 0
	s_mov_b32 s11, 0
	s_cselect_b32 s2, s2, 0
	v_or_b32_e32 v1, v3, v1
	s_lshl2_add_u32 s42, s2, s6
	v_or_b32_e32 v4, v2, v3
	s_add_co_i32 s4, s42, 0xee00
	v_lshrrev_b32_e32 v6, 8, v6
	s_and_b32 s10, s4, 15
	v_lshrrev_b32_e32 v5, 4, v1
	s_sub_co_i32 s2, 16, s10
	v_lshrrev_b32_e32 v7, 4, v4
	s_lshr_b32 s2, s2, 2
	s_cmp_lg_u64 s[10:11], 0
	v_and_b32_e32 v5, 0x3f8, v5
	s_cselect_b32 s2, s2, 0
	s_ashr_i32 s6, s17, 31
	v_and_b32_e32 v7, 0x678, v7
	s_lshr_b32 s6, s6, 25
	v_dual_add_nc_u32 v226, v5, v1 :: v_dual_add_nc_u32 v5, 1, v6
	s_add_co_i32 s17, s17, s6
	s_lshl_b32 s10, s2, 2
	s_ashr_i32 s44, s17, 7
	s_cmp_lt_i32 s29, 0x100
	v_dual_add_nc_u32 v230, v7, v4 :: v_dual_bitop2_b32 v241, 62, v5 bitop3:0x40
	s_add_nc_u64 s[30:31], s[4:5], s[10:11]
	s_cselect_b32 s45, -1, 0
	s_lshl_b32 s4, s0, 8
	s_mov_b32 s43, s5
	s_ashr_i32 s5, s4, 31
	s_wait_kmcnt 0x0
	s_bfe_i64 s[6:7], s[20:21], 0x200000
	v_lshl_or_b32 v4, v241, 8, v0
	s_mul_u64 s[4:5], s[6:7], s[4:5]
	s_ashr_i32 s35, s34, 31
	s_lshl_b64 s[4:5], s[4:5], 1
	s_bfe_i64 s[6:7], s[12:13], 0x200000
	s_add_nc_u64 s[20:21], s[14:15], s[4:5]
	s_mul_u64 s[4:5], s[6:7], s[34:35]
	v_or_b32_e32 v1, 0x100, v0
	v_cmp_ne_u32_e64 s0, v5, v241
	v_add3_u32 v232, v3, v2, v7
	v_dual_mov_b32 v235, v229 :: v_dual_add_nc_u32 v242, 0xffffff00, v4
	v_dual_mov_b32 v237, v229 :: v_dual_lshlrev_b32 v234, 2, v4
	v_or_b32_e32 v243, 0x4300, v0
	v_lshl_or_b32 v236, v0, 2, 0x11000
	v_dual_mov_b32 v2, v229 :: v_dual_mov_b32 v3, v229
	v_dual_mov_b32 v4, v229 :: v_dual_mov_b32 v5, v229
	v_dual_mov_b32 v6, v229 :: v_dual_mov_b32 v7, v229
	v_dual_mov_b32 v8, v229 :: v_dual_mov_b32 v9, v229
	v_dual_mov_b32 v10, v229 :: v_dual_mov_b32 v11, v229
	v_dual_mov_b32 v12, v229 :: v_dual_mov_b32 v13, v229
	v_dual_mov_b32 v14, v229 :: v_dual_mov_b32 v15, v229
	v_dual_mov_b32 v16, v229 :: v_dual_mov_b32 v17, v229
	v_dual_mov_b32 v18, v229 :: v_dual_mov_b32 v19, v229
	v_dual_mov_b32 v20, v229 :: v_dual_mov_b32 v21, v229
	v_dual_mov_b32 v22, v229 :: v_dual_mov_b32 v23, v229
	v_dual_mov_b32 v24, v229 :: v_dual_mov_b32 v25, v229
	v_dual_mov_b32 v26, v229 :: v_dual_mov_b32 v27, v229
	v_dual_mov_b32 v28, v229 :: v_dual_mov_b32 v29, v229
	v_dual_mov_b32 v30, v229 :: v_dual_mov_b32 v31, v229
	v_dual_mov_b32 v32, v229 :: v_dual_mov_b32 v33, v229
	v_dual_mov_b32 v34, v229 :: v_dual_mov_b32 v35, v229
	v_dual_mov_b32 v36, v229 :: v_dual_mov_b32 v37, v229
	v_dual_mov_b32 v38, v229 :: v_dual_mov_b32 v39, v229
	v_dual_mov_b32 v40, v229 :: v_dual_mov_b32 v41, v229
	v_dual_mov_b32 v42, v229 :: v_dual_mov_b32 v43, v229
	v_dual_mov_b32 v44, v229 :: v_dual_mov_b32 v45, v229
	v_dual_mov_b32 v46, v229 :: v_dual_mov_b32 v47, v229
	v_dual_mov_b32 v48, v229 :: v_dual_mov_b32 v49, v229
	v_dual_mov_b32 v50, v229 :: v_dual_mov_b32 v51, v229
	v_dual_mov_b32 v52, v229 :: v_dual_mov_b32 v53, v229
	v_dual_mov_b32 v54, v229 :: v_dual_mov_b32 v55, v229
	v_dual_mov_b32 v56, v229 :: v_dual_mov_b32 v57, v229
	v_dual_mov_b32 v58, v229 :: v_dual_mov_b32 v59, v229
	v_dual_mov_b32 v60, v229 :: v_dual_mov_b32 v61, v229
	v_dual_mov_b32 v62, v229 :: v_dual_mov_b32 v63, v229
	v_dual_mov_b32 v64, v229 :: v_dual_mov_b32 v65, v229
	v_dual_mov_b32 v66, v229 :: v_dual_mov_b32 v67, v229
	v_dual_mov_b32 v68, v229 :: v_dual_mov_b32 v69, v229
	v_dual_mov_b32 v70, v229 :: v_dual_mov_b32 v71, v229
	v_dual_mov_b32 v72, v229 :: v_dual_mov_b32 v73, v229
	v_dual_mov_b32 v74, v229 :: v_dual_mov_b32 v75, v229
	v_dual_mov_b32 v76, v229 :: v_dual_mov_b32 v77, v229
	v_dual_mov_b32 v78, v229 :: v_dual_mov_b32 v79, v229
	v_dual_mov_b32 v80, v229 :: v_dual_mov_b32 v81, v229
	v_dual_mov_b32 v82, v229 :: v_dual_mov_b32 v83, v229
	v_dual_mov_b32 v84, v229 :: v_dual_mov_b32 v85, v229
	v_dual_mov_b32 v86, v229 :: v_dual_mov_b32 v87, v229
	v_dual_mov_b32 v88, v229 :: v_dual_mov_b32 v89, v229
	v_dual_mov_b32 v90, v229 :: v_dual_mov_b32 v91, v229
	v_dual_mov_b32 v92, v229 :: v_dual_mov_b32 v93, v229
	v_dual_mov_b32 v94, v229 :: v_dual_mov_b32 v95, v229
	v_dual_mov_b32 v96, v229 :: v_dual_mov_b32 v97, v229
	v_dual_mov_b32 v98, v229 :: v_dual_mov_b32 v99, v229
	v_dual_mov_b32 v100, v229 :: v_dual_mov_b32 v101, v229
	v_dual_mov_b32 v102, v229 :: v_dual_mov_b32 v103, v229
	v_dual_mov_b32 v104, v229 :: v_dual_mov_b32 v105, v229
	v_dual_mov_b32 v114, v229 :: v_dual_mov_b32 v115, v229
	v_dual_mov_b32 v116, v229 :: v_dual_mov_b32 v117, v229
	v_dual_mov_b32 v118, v229 :: v_dual_mov_b32 v119, v229
	v_dual_mov_b32 v120, v229 :: v_dual_mov_b32 v121, v229
	v_dual_mov_b32 v122, v229 :: v_dual_mov_b32 v123, v229
	v_dual_mov_b32 v124, v229 :: v_dual_mov_b32 v125, v229
	v_dual_mov_b32 v126, v229 :: v_dual_mov_b32 v127, v229
	v_dual_mov_b32 v128, v229 :: v_dual_mov_b32 v129, v229
	v_dual_mov_b32 v130, v229 :: v_dual_mov_b32 v131, v229
	v_dual_mov_b32 v132, v229 :: v_dual_mov_b32 v133, v229
	v_dual_mov_b32 v134, v229 :: v_dual_mov_b32 v135, v229
	v_dual_mov_b32 v136, v229 :: v_dual_mov_b32 v137, v229
	v_dual_mov_b32 v138, v229 :: v_dual_mov_b32 v139, v229
	v_dual_mov_b32 v140, v229 :: v_dual_mov_b32 v141, v229
	v_dual_mov_b32 v142, v229 :: v_dual_mov_b32 v143, v229
	v_dual_mov_b32 v144, v229 :: v_dual_mov_b32 v145, v229
	v_dual_mov_b32 v146, v229 :: v_dual_mov_b32 v147, v229
	v_dual_mov_b32 v148, v229 :: v_dual_mov_b32 v149, v229
	v_dual_mov_b32 v150, v229 :: v_dual_mov_b32 v151, v229
	v_dual_mov_b32 v152, v229 :: v_dual_mov_b32 v153, v229
	v_dual_mov_b32 v154, v229 :: v_dual_mov_b32 v155, v229
	v_dual_mov_b32 v156, v229 :: v_dual_mov_b32 v157, v229
	v_dual_mov_b32 v158, v229 :: v_dual_mov_b32 v159, v229
	v_dual_mov_b32 v160, v229 :: v_dual_mov_b32 v161, v229
	v_dual_mov_b32 v162, v229 :: v_dual_mov_b32 v163, v229
	v_dual_mov_b32 v164, v229 :: v_dual_mov_b32 v165, v229
	v_dual_mov_b32 v166, v229 :: v_dual_mov_b32 v167, v229
	v_dual_mov_b32 v168, v229 :: v_dual_mov_b32 v169, v229
	v_dual_mov_b32 v170, v229 :: v_dual_mov_b32 v171, v229
	v_dual_mov_b32 v172, v229 :: v_dual_mov_b32 v173, v229
	v_dual_mov_b32 v174, v229 :: v_dual_mov_b32 v175, v229
	v_dual_mov_b32 v176, v229 :: v_dual_mov_b32 v177, v229
	v_dual_mov_b32 v178, v229 :: v_dual_mov_b32 v179, v229
	v_dual_mov_b32 v180, v229 :: v_dual_mov_b32 v181, v229
	v_dual_mov_b32 v182, v229 :: v_dual_mov_b32 v183, v229
	v_dual_mov_b32 v184, v229 :: v_dual_mov_b32 v185, v229
	v_dual_mov_b32 v186, v229 :: v_dual_mov_b32 v187, v229
	v_dual_mov_b32 v188, v229 :: v_dual_mov_b32 v189, v229
	v_dual_mov_b32 v190, v229 :: v_dual_mov_b32 v191, v229
	v_dual_mov_b32 v192, v229 :: v_dual_mov_b32 v193, v229
	v_dual_mov_b32 v194, v229 :: v_dual_mov_b32 v195, v229
	v_dual_mov_b32 v196, v229 :: v_dual_mov_b32 v197, v229
	v_dual_mov_b32 v198, v229 :: v_dual_mov_b32 v199, v229
	v_dual_mov_b32 v200, v229 :: v_dual_mov_b32 v201, v229
	v_dual_mov_b32 v202, v229 :: v_dual_mov_b32 v203, v229
	v_dual_mov_b32 v204, v229 :: v_dual_mov_b32 v205, v229
	v_dual_mov_b32 v206, v229 :: v_dual_mov_b32 v207, v229
	v_dual_mov_b32 v208, v229 :: v_dual_mov_b32 v209, v229
	v_dual_mov_b32 v210, v229 :: v_dual_mov_b32 v211, v229
	v_dual_mov_b32 v212, v229 :: v_dual_mov_b32 v213, v229
	v_dual_mov_b32 v214, v229 :: v_dual_mov_b32 v215, v229
	v_dual_mov_b32 v216, v229 :: v_dual_mov_b32 v217, v229
	v_dual_mov_b32 v218, v229 :: v_dual_mov_b32 v219, v229
	v_dual_mov_b32 v220, v229 :: v_dual_mov_b32 v221, v229
	v_dual_mov_b32 v222, v229 :: v_dual_mov_b32 v223, v229
	v_dual_mov_b32 v224, v229 :: v_dual_mov_b32 v225, v229
	v_dual_mov_b32 v106, v229 :: v_dual_mov_b32 v107, v229
	v_dual_mov_b32 v108, v229 :: v_dual_mov_b32 v109, v229
	v_dual_mov_b32 v110, v229 :: v_dual_mov_b32 v111, v229
	v_dual_mov_b32 v112, v229 :: v_dual_mov_b32 v113, v229
	s_lshr_b32 s46, s29, 16
	s_lshr_b32 s47, s3, 16
	s_lshl_b64 s[4:5], s[4:5], 1
	s_mov_b32 s41, s1
	s_movk_i32 s16, 0x100
	s_bitset1_b32 s46, 23
	s_movk_i32 s8, 0xe0
	s_bitset1_b32 s47, 23
	s_add_nc_u64 s[34:35], s[18:19], s[4:5]
	s_mov_b32 s4, 0x7510000
	s_mov_b32 s48, s11
	s_branch .LBB0_17
.LBB0_16:                               ;   in Loop: Header=BB0_17 Depth=1
	s_cmp_eq_u32 s48, s44
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_scc1 .LBB0_37
.LBB0_17:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_19 Depth 2
                                        ;     Child Loop BB0_22 Depth 2
                                        ;     Child Loop BB0_24 Depth 2
                                        ;     Child Loop BB0_27 Depth 2
	s_and_b32 s49, s48, 1
	s_add_co_i32 s48, s48, 1
	s_xor_b32 s5, s49, 1
	s_lshl_b32 s2, s48, 7
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_sub_co_i32 s2, s26, s2
	s_min_i32 s2, s2, 0x80
	s_cmp_lt_i32 s48, s44
	s_cselect_b32 s10, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_and_b32 s6, s10, exec_lo
	s_cselect_b32 s2, s2, 0
	s_cmp_lt_i32 s2, 0x80
	s_cselect_b32 s6, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b32 s6, s45, s6
	s_or_b32 s6, s39, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s6
	s_cbranch_vccnz .LBB0_29
; %bb.18:                               ;   in Loop: Header=BB0_17 Depth=1
	v_mov_b64_e32 v[238:239], v[0:1]
	v_mov_b32_e32 v244, v241
	s_cmp_lg_u32 s5, 0
	s_mov_b32 s12, 0
	s_cselect_b32 s7, s43, s1
	s_cselect_b32 s6, s42, 0
.LBB0_19:                               ;   Parent Loop BB0_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_dual_mov_b32 v228, v238 :: v_dual_add_nc_u32 v244, -2, v244
	v_dual_mov_b32 v246, v239 :: v_dual_mov_b32 v247, v229
	v_add_nc_u32_e32 v239, 0x200, v239
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u64 v[248:249], v[228:229], 2, s[6:7]
	v_cmp_eq_u32_e32 vcc_lo, 0, v244
	v_add_nc_u32_e32 v238, 0x200, v238
	v_lshl_add_u64 v[246:247], v[246:247], 2, s[6:7]
	s_clause 0x1
	flat_store_b32 v[248:249], v229
	flat_store_b32 v[246:247], v229
	s_or_b32 s12, vcc_lo, s12
	s_wait_xcnt 0x0
	s_and_not1_b32 exec_lo, exec_lo, s12
	s_cbranch_execnz .LBB0_19
; %bb.20:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s12
	s_and_saveexec_b32 s12, s0
	s_cbranch_execz .LBB0_23
; %bb.21:                               ;   in Loop: Header=BB0_17 Depth=1
	v_add_nc_u64_e32 v[238:239], s[6:7], v[234:235]
	v_mov_b32_e32 v228, v242
	s_mov_b32 s6, 0
.LBB0_22:                               ;   Parent Loop BB0_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v228, 0x100, v228
	flat_store_b32 v[238:239], v229
	s_wait_xcnt 0x0
	v_add_nc_u64_e32 v[238:239], 0x400, v[238:239]
	v_cmp_lt_u32_e32 vcc_lo, 0x3a7f, v228
	s_or_b32 s6, vcc_lo, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s6
	s_cbranch_execnz .LBB0_22
.LBB0_23:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s12
	v_mov_b64_e32 v[238:239], v[0:1]
	v_mov_b32_e32 v244, 0x44
	s_cmp_lg_u32 s5, 0
	s_mov_b32 s12, 0
	s_cselect_b32 s7, s31, s41
	s_cselect_b32 s6, s30, s40
.LBB0_24:                               ;   Parent Loop BB0_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_dual_mov_b32 v228, v238 :: v_dual_add_nc_u32 v244, -2, v244
	v_dual_mov_b32 v246, v239 :: v_dual_mov_b32 v247, v229
	v_add_nc_u32_e32 v239, 0x200, v239
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u64 v[248:249], v[228:229], 2, s[6:7]
	v_cmp_eq_u32_e32 vcc_lo, 0, v244
	v_add_nc_u32_e32 v238, 0x200, v238
	v_lshl_add_u64 v[246:247], v[246:247], 2, s[6:7]
	s_clause 0x1
	flat_store_b32 v[248:249], v229
	flat_store_b32 v[246:247], v229
	s_or_b32 s12, vcc_lo, s12
	s_wait_xcnt 0x0
	s_and_not1_b32 exec_lo, exec_lo, s12
	s_cbranch_execnz .LBB0_24
; %bb.25:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s12
	s_and_saveexec_b32 s12, s11
	s_cbranch_execz .LBB0_28
; %bb.26:                               ;   in Loop: Header=BB0_17 Depth=1
	v_add_nc_u64_e32 v[238:239], s[6:7], v[236:237]
	v_mov_b32_e32 v228, v243
	s_mov_b32 s6, 0
.LBB0_27:                               ;   Parent Loop BB0_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v228, 0x100, v228
	flat_store_b32 v[238:239], v229
	s_wait_xcnt 0x0
	v_add_nc_u64_e32 v[238:239], 0x400, v[238:239]
	v_cmp_lt_u32_e32 vcc_lo, 0x42ff, v228
	s_or_b32 s6, vcc_lo, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s6
	s_cbranch_execnz .LBB0_27
.LBB0_28:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s12
	s_barrier_signal -1
	s_barrier_wait -1
.LBB0_29:                               ;   in Loop: Header=BB0_17 Depth=1
	s_and_b32 s6, s10, exec_lo
	s_cselect_b32 s6, s48, 0
	s_mov_b32 s7, exec_lo
	v_cmpx_lt_i32_e32 0, v240
	s_xor_b32 s7, exec_lo, s7
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
	s_mov_b32 s50, exec_lo
	v_cmpx_eq_u32_e32 1, v240
	s_cbranch_execz .LBB0_34
; %bb.33:                               ;   in Loop: Header=BB0_17 Depth=1
	s_cmp_lg_u32 s5, 0
	s_mov_b32 s28, s2
	s_cselect_b32 s14, s30, s40
	s_cmp_gt_i32 s2, 0
	s_mov_b32 s17, s9
	s_cselect_b32 s15, -1, 0
	s_lshl_b32 s10, s6, 7
	s_mov_b32 s18, s11
	s_lshl_b64 s[12:13], s[10:11], 1
	s_mov_b32 s19, s11
	s_add_nc_u64 s[12:13], s[20:21], s[12:13]
	s_delay_alu instid0(SALU_CYCLE_1)
	v_dual_mov_b32 v239, s14 :: v_dual_mov_b32 v238, s12
	s_and_b32 s10, s13, 0x1ffffff
	s_and_b32 s13, s38, s15
	s_bitset1_b32 s10, 31
	v_cndmask_b32_e64 v228, 0, 1, s13
	v_mov_b32_e32 v245, s10
	v_readfirstlane_b32 s53, v239
	v_readfirstlane_b32 s54, v238
	s_lshr_b64 s[14:15], s[28:29], 16
	v_readfirstlane_b32 s52, v228
	v_readfirstlane_b32 s55, v245
	s_lshl_b32 s13, s2, 16
	s_mov_b32 s12, s4
	s_mov_b32 s15, s46
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[52:55], s[12:19]
.LBB0_34:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s50
	s_and_not1_saveexec_b32 s12, s7
	s_cbranch_execz .LBB0_31
.LBB0_35:                               ;   in Loop: Header=BB0_17 Depth=1
	s_cmp_lg_u32 s5, 0
	s_cselect_b32 s5, s42, 0
	s_cmp_gt_i32 s2, 0
	s_cselect_b32 s13, -1, 0
	s_lshl_b32 s10, s6, 7
	s_delay_alu instid0(SALU_CYCLE_1)
	s_lshl_b64 s[6:7], s[10:11], 1
	s_and_b32 s10, s37, s13
	s_add_nc_u64 s[6:7], s[34:35], s[6:7]
	v_cndmask_b32_e64 v228, 0, 1, s10
	s_and_b32 s7, s7, 0x1ffffff
	v_dual_mov_b32 v239, s5 :: v_dual_mov_b32 v238, s6
	s_bitset1_b32 s7, 31
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_readfirstlane_b32 s52, v228
	v_mov_b32_e32 v245, s7
	v_readfirstlane_b32 s53, v239
	v_readfirstlane_b32 s54, v238
	s_lshr_b64 s[6:7], s[2:3], 16
	s_lshl_b32 s5, s2, 16
	v_readfirstlane_b32 s55, v245
	s_mov_b32 s7, s47
	s_mov_b32 s10, s11
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	tensor_load_to_lds s[52:55], s[4:11]
	s_or_b32 exec_lo, exec_lo, s12
	s_and_not1_b32 vcc_lo, exec_lo, s25
	s_cbranch_vccnz .LBB0_16
.LBB0_36:                               ;   in Loop: Header=BB0_17 Depth=1
	s_cmp_lg_u32 s49, 0
	s_cselect_b32 s2, s42, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_lshl_add_u32 v228, v226, 1, s2
	s_cselect_b32 s2, s30, s40
	v_lshl_add_u32 v238, v230, 1, s2
	ds_load_b128 v[244:247], v228
	ds_load_b128 v[248:251], v228 offset:16
	ds_load_b128 v[252:255], v228 offset:4352
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[0:3] /*v[256:259]*/, v228 offset:4368
	ds_load_b128 v[4:7] /*v[260:263]*/, v228 offset:8704
	ds_load_b128 v[8:11] /*v[264:267]*/, v228 offset:8720
	ds_load_b128 v[12:15] /*v[268:271]*/, v228 offset:13056
	ds_load_b128 v[16:19] /*v[272:275]*/, v228 offset:13072
	ds_load_b128 v[20:23] /*v[276:279]*/, v228 offset:17408
	ds_load_b128 v[24:27] /*v[280:283]*/, v228 offset:17424
	ds_load_b128 v[28:31] /*v[284:287]*/, v228 offset:21760
	ds_load_b128 v[44:47] /*v[300:303]*/, v238 offset:8704
	ds_load_b128 v[48:51] /*v[304:307]*/, v238 offset:8720
	ds_load_b128 v[52:55] /*v[308:311]*/, v228 offset:26112
	ds_load_b128 v[56:59] /*v[312:315]*/, v228 offset:26128
	ds_load_b128 v[32:35] /*v[288:291]*/, v228 offset:21776
	ds_load_b128 v[60:63] /*v[316:319]*/, v228 offset:21824
	ds_load_b128 v[64:67] /*v[320:323]*/, v228 offset:21840
	ds_load_b128 v[68:71] /*v[324:327]*/, v228 offset:26176
	ds_load_b128 v[72:75] /*v[328:331]*/, v228 offset:26192
	; sched_group_barrier mask(0x00000100) size(11) SyncID(0)
	s_set_vgpr_msb 0x4004                   ;  msbs: dst=0 src0=0 src1=1 src2=0
	s_wait_dscnt 0x7
	v_wmma_f32_16x16x32_bf16 v[202:209], v[244:251], v[44:51] /*v[300:307]*/, v[202:209] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[170:177], v[252:259], v[44:51] /*v[300:307]*/, v[170:177] matrix_b_reuse
	s_set_vgpr_msb 0x405                    ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[138:145], v[4:11] /*v[260:267]*/, v[44:51] /*v[300:307]*/, v[138:145] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[98:105], v[12:19] /*v[268:275]*/, v[44:51] /*v[300:307]*/, v[98:105] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[66:73], v[20:27] /*v[276:283]*/, v[44:51] /*v[300:307]*/, v[66:73] matrix_b_reuse
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[2:9], v[52:59] /*v[308:315]*/, v[44:51] /*v[300:307]*/, v[2:9] matrix_b_reuse
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[34:41], v[28:35] /*v[284:291]*/, v[44:51] /*v[300:307]*/, v[34:41] matrix_b_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[44:47] /*v[300:303]*/, v238 offset:4352
	ds_load_b128 v[48:51] /*v[304:307]*/, v238 offset:4368
	s_set_vgpr_msb 0x4005                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[10:17], v[52:59] /*v[308:315]*/, v[44:51] /*v[300:307]*/, v[10:17] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[42:49], v[28:35] /*v[284:291]*/, v[44:51] /*v[300:307]*/, v[42:49] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[74:81], v[20:27] /*v[276:283]*/, v[44:51] /*v[300:307]*/, v[74:81] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[114:121], v[12:19] /*v[268:275]*/, v[44:51] /*v[300:307]*/, v[114:121] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[146:153], v[4:11] /*v[260:267]*/, v[44:51] /*v[300:307]*/, v[146:153] matrix_b_reuse
	s_set_vgpr_msb 0x504                    ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[178:185], v[252:259], v[44:51] /*v[300:307]*/, v[178:185] matrix_b_reuse
	s_set_vgpr_msb 0x440                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[36:39] /*v[292:295]*/, v238 offset:13056
	ds_load_b128 v[40:43] /*v[296:299]*/, v238 offset:13072
	s_set_vgpr_msb 0x4005                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[58:65], v[20:27] /*v[276:283]*/, v[36:43] /*v[292:299]*/, v[58:65] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[90:97], v[12:19] /*v[268:275]*/, v[36:43] /*v[292:299]*/, v[90:97] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[130:137], v[4:11] /*v[260:267]*/, v[36:43] /*v[292:299]*/, v[130:137] matrix_b_reuse
	s_set_vgpr_msb 0x504                    ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[162:169], v[252:259], v[36:43] /*v[292:299]*/, v[162:169] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[194:201], v[244:251], v[36:43] /*v[292:299]*/, v[194:201] matrix_b_reuse
	s_set_vgpr_msb 0x405                    ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[106:113], v[52:59] /*v[308:315]*/, v[36:43] /*v[292:299]*/, v[106:113] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[26:33], v[28:35] /*v[284:291]*/, v[36:43] /*v[292:299]*/, v[26:33] matrix_b_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[36:39] /*v[292:295]*/, v238
	ds_load_b128 v[40:43] /*v[296:299]*/, v238 offset:16
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v238, v232, 1, s2
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[76:79] /*v[332:335]*/, v238 offset:4416
	s_set_vgpr_msb 0x4004                   ;  msbs: dst=0 src0=0 src1=1 src2=0
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x32_bf16 v[218:225], v[244:251], v[36:43] /*v[292:299]*/, v[218:225]
	s_set_vgpr_msb 0x440                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[80:83] /*v[336:339]*/, v238 offset:4432
	ds_load_b128 v[84:87] /*v[340:343]*/, v238 offset:8768
	ds_load_b128 v[88:91] /*v[344:347]*/, v238 offset:8784
	ds_load_b128 v[92:95] /*v[348:351]*/, v238 offset:13120
	ds_load_b128 v[96:99] /*v[352:355]*/, v238 offset:13136
	s_set_vgpr_msb 0x4004                   ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[186:193], v[252:259], v[36:43] /*v[292:299]*/, v[186:193] matrix_b_reuse
	s_set_vgpr_msb 0x405                    ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[4:11] /*v[260:267]*/, v[36:43] /*v[292:299]*/, v[154:161] matrix_b_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[4:7] /*v[260:263]*/, v238 offset:64
	ds_load_b128 v[8:11] /*v[264:267]*/, v238 offset:80
	s_set_vgpr_msb 0x4005                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[122:129], v[12:19] /*v[268:275]*/, v[36:43] /*v[292:299]*/, v[122:129] matrix_b_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[12:15] /*v[268:271]*/, v228 offset:17472
	ds_load_b128 v[16:19] /*v[272:275]*/, v228 offset:17488
	s_set_vgpr_msb 0x4005                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[82:89], v[20:27] /*v[276:283]*/, v[36:43] /*v[292:299]*/, v[82:89] matrix_b_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[20:23] /*v[276:279]*/, v228 offset:4416
	ds_load_b128 v[24:27] /*v[280:283]*/, v228 offset:4432
	s_set_vgpr_msb 0x4005                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[50:57], v[28:35] /*v[284:291]*/, v[36:43] /*v[292:299]*/, v[50:57] matrix_b_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[28:31] /*v[284:287]*/, v228 offset:64
	ds_load_b128 v[32:35] /*v[288:291]*/, v228 offset:80
	s_set_vgpr_msb 0x4005                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[18:25], v[52:59] /*v[308:315]*/, v[36:43] /*v[292:299]*/, v[18:25] matrix_b_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[36:39] /*v[292:295]*/, v228 offset:8768
	ds_load_b128 v[40:43] /*v[296:299]*/, v228 offset:8784
	ds_load_b128 v[52:55] /*v[308:311]*/, v228 offset:13120
	ds_load_b128 v[56:59] /*v[312:315]*/, v228 offset:13136
	; sched_group_barrier mask(0x00000008) size(14) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(11) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(14) SyncID(0)
	s_set_vgpr_msb 0x4004                   ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[210:217], v[244:251], v[44:51] /*v[300:307]*/, v[210:217] matrix_b_reuse
	; sched_barrier mask(0x00000000)
	ds_load_b128 v[244:247], v228 offset:128
	ds_load_b128 v[248:251], v228 offset:144
	ds_load_b128 v[252:255], v228 offset:4480
	s_set_vgpr_msb 0x440                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[0:3] /*v[256:259]*/, v228 offset:4496
	ds_load_b128 v[44:47] /*v[300:303]*/, v228 offset:8832
	ds_load_b128 v[48:51] /*v[304:307]*/, v228 offset:8848
	ds_load_b128 v[100:103] /*v[356:359]*/, v228 offset:13184
	ds_load_b128 v[104:107] /*v[360:363]*/, v228 offset:13200
	ds_load_b128 v[108:111] /*v[364:367]*/, v228 offset:17536
	ds_load_b128 v[112:115] /*v[368:371]*/, v228 offset:17552
	ds_load_b128 v[116:119] /*v[372:375]*/, v228 offset:21888
	s_set_vgpr_msb 0x4005                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0xf
	v_wmma_f32_16x16x32_bf16 v[218:225], v[28:35] /*v[284:291]*/, v[4:11] /*v[260:267]*/, v[218:225]
	; sched_group_barrier mask(0x00000100) size(11) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[186:193], v[20:27] /*v[276:283]*/, v[4:11] /*v[260:267]*/, v[186:193] matrix_b_reuse
	s_wait_dscnt 0xd
	v_wmma_f32_16x16x32_bf16 v[154:161], v[36:43] /*v[292:299]*/, v[4:11] /*v[260:267]*/, v[154:161] matrix_b_reuse
	s_wait_dscnt 0xb
	v_wmma_f32_16x16x32_bf16 v[122:129], v[52:59] /*v[308:315]*/, v[4:11] /*v[260:267]*/, v[122:129] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[82:89], v[12:19] /*v[268:275]*/, v[4:11] /*v[260:267]*/, v[82:89] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[50:57], v[60:67] /*v[316:323]*/, v[4:11] /*v[260:267]*/, v[50:57] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[18:25], v[68:75] /*v[324:331]*/, v[4:11] /*v[260:267]*/, v[18:25] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[10:17], v[68:75] /*v[324:331]*/, v[76:83] /*v[332:339]*/, v[10:17] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[42:49], v[60:67] /*v[316:323]*/, v[76:83] /*v[332:339]*/, v[42:49] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[74:81], v[12:19] /*v[268:275]*/, v[76:83] /*v[332:339]*/, v[74:81] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[114:121], v[52:59] /*v[308:315]*/, v[76:83] /*v[332:339]*/, v[114:121] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[146:153], v[36:43] /*v[292:299]*/, v[76:83] /*v[332:339]*/, v[146:153] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[178:185], v[20:27] /*v[276:283]*/, v[76:83] /*v[332:339]*/, v[178:185] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[210:217], v[28:35] /*v[284:291]*/, v[76:83] /*v[332:339]*/, v[210:217] matrix_b_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[120:123] /*v[376:379]*/, v228 offset:21904
	ds_load_b128 v[4:7] /*v[260:263]*/, v228 offset:26240
	ds_load_b128 v[8:11] /*v[264:267]*/, v228 offset:26256
	ds_load_b128 v[76:79] /*v[332:335]*/, v238 offset:128
	ds_load_b128 v[80:83] /*v[336:339]*/, v238 offset:144
	ds_load_b128 v[124:127] /*v[380:383]*/, v238 offset:4480
	ds_load_b128 v[128:131] /*v[384:387]*/, v238 offset:4496
	ds_load_b128 v[132:135] /*v[388:391]*/, v238 offset:8832
	ds_load_b128 v[136:139] /*v[392:395]*/, v238 offset:8848
	ds_load_b128 v[140:143] /*v[396:399]*/, v238 offset:13184
	ds_load_b128 v[144:147] /*v[400:403]*/, v238 offset:13200
	s_set_vgpr_msb 0x4005                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[202:209], v[28:35] /*v[284:291]*/, v[84:91] /*v[340:347]*/, v[202:209] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(14) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(11) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[170:177], v[20:27] /*v[276:283]*/, v[84:91] /*v[340:347]*/, v[170:177] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[138:145], v[36:43] /*v[292:299]*/, v[84:91] /*v[340:347]*/, v[138:145] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[98:105], v[52:59] /*v[308:315]*/, v[84:91] /*v[340:347]*/, v[98:105] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[66:73], v[12:19] /*v[268:275]*/, v[84:91] /*v[340:347]*/, v[66:73] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[34:41], v[60:67] /*v[316:323]*/, v[84:91] /*v[340:347]*/, v[34:41] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[2:9], v[68:75] /*v[324:331]*/, v[84:91] /*v[340:347]*/, v[2:9] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[106:113], v[68:75] /*v[324:331]*/, v[92:99] /*v[348:355]*/, v[106:113] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[26:33], v[60:67] /*v[316:323]*/, v[92:99] /*v[348:355]*/, v[26:33] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[58:65], v[12:19] /*v[268:275]*/, v[92:99] /*v[348:355]*/, v[58:65] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[90:97], v[52:59] /*v[308:315]*/, v[92:99] /*v[348:355]*/, v[90:97] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[130:137], v[36:43] /*v[292:299]*/, v[92:99] /*v[348:355]*/, v[130:137] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[162:169], v[20:27] /*v[276:283]*/, v[92:99] /*v[348:355]*/, v[162:169] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[194:201], v[28:35] /*v[284:291]*/, v[92:99] /*v[348:355]*/, v[194:201] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(14) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[12:15] /*v[268:271]*/, v228 offset:192
	ds_load_b128 v[16:19] /*v[272:275]*/, v228 offset:208
	ds_load_b128 v[20:23] /*v[276:279]*/, v228 offset:4544
	ds_load_b128 v[24:27] /*v[280:283]*/, v228 offset:4560
	ds_load_b128 v[28:31] /*v[284:287]*/, v228 offset:8896
	ds_load_b128 v[32:35] /*v[288:291]*/, v228 offset:8912
	ds_load_b128 v[36:39] /*v[292:295]*/, v228 offset:13248
	ds_load_b128 v[40:43] /*v[296:299]*/, v228 offset:13264
	ds_load_b128 v[52:55] /*v[308:311]*/, v228 offset:17600
	ds_load_b128 v[56:59] /*v[312:315]*/, v228 offset:17616
	ds_load_b128 v[60:63] /*v[316:319]*/, v228 offset:21952
	s_set_vgpr_msb 0x4004                   ;  msbs: dst=0 src0=0 src1=1 src2=0
	s_wait_dscnt 0x11
	v_wmma_f32_16x16x32_bf16 v[218:225], v[244:251], v[76:83] /*v[332:339]*/, v[218:225]
	; sched_group_barrier mask(0x00000100) size(11) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[186:193], v[252:259], v[76:83] /*v[332:339]*/, v[186:193] matrix_b_reuse
	s_set_vgpr_msb 0x405                    ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[44:51] /*v[300:307]*/, v[76:83] /*v[332:339]*/, v[154:161] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[122:129], v[100:107] /*v[356:363]*/, v[76:83] /*v[332:339]*/, v[122:129] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[82:89], v[108:115] /*v[364:371]*/, v[76:83] /*v[332:339]*/, v[82:89] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[50:57], v[116:123] /*v[372:379]*/, v[76:83] /*v[332:339]*/, v[50:57] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[18:25], v[4:11] /*v[260:267]*/, v[76:83] /*v[332:339]*/, v[18:25] matrix_b_reuse
	s_wait_dscnt 0xf
	v_wmma_f32_16x16x32_bf16 v[10:17], v[4:11] /*v[260:267]*/, v[124:131] /*v[380:387]*/, v[10:17] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[42:49], v[116:123] /*v[372:379]*/, v[124:131] /*v[380:387]*/, v[42:49] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[74:81], v[108:115] /*v[364:371]*/, v[124:131] /*v[380:387]*/, v[74:81] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[114:121], v[100:107] /*v[356:363]*/, v[124:131] /*v[380:387]*/, v[114:121] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[146:153], v[44:51] /*v[300:307]*/, v[124:131] /*v[380:387]*/, v[146:153] matrix_b_reuse
	s_set_vgpr_msb 0x504                    ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[178:185], v[252:259], v[124:131] /*v[380:387]*/, v[178:185] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[210:217], v[244:251], v[124:131] /*v[380:387]*/, v[210:217] matrix_b_reuse
	s_set_vgpr_msb 0x440                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[64:67] /*v[320:323]*/, v228 offset:21968
	ds_load_b128 v[68:71] /*v[324:327]*/, v228 offset:26304
	ds_load_b128 v[72:75] /*v[328:331]*/, v228 offset:26320
	ds_load_b128 v[76:79] /*v[332:335]*/, v238 offset:192
	ds_load_b128 v[80:83] /*v[336:339]*/, v238 offset:208
	ds_load_b128 v[84:87] /*v[340:343]*/, v238 offset:4544
	ds_load_b128 v[88:91] /*v[344:347]*/, v238 offset:4560
	ds_load_b128 v[92:95] /*v[348:351]*/, v238 offset:8896
	ds_load_b128 v[96:99] /*v[352:355]*/, v238 offset:8912
	ds_load_b128 v[124:127] /*v[380:383]*/, v238 offset:13248
	ds_load_b128 v[128:131] /*v[384:387]*/, v238 offset:13264
	s_set_vgpr_msb 0x4004                   ;  msbs: dst=0 src0=0 src1=1 src2=0
	s_wait_dscnt 0x18
	v_wmma_f32_16x16x32_bf16 v[202:209], v[244:251], v[132:139] /*v[388:395]*/, v[202:209] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(14) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(11) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[170:177], v[252:259], v[132:139] /*v[388:395]*/, v[170:177] matrix_b_reuse
	s_set_vgpr_msb 0x405                    ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[138:145], v[44:51] /*v[300:307]*/, v[132:139] /*v[388:395]*/, v[138:145] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[98:105], v[100:107] /*v[356:363]*/, v[132:139] /*v[388:395]*/, v[98:105] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[66:73], v[108:115] /*v[364:371]*/, v[132:139] /*v[388:395]*/, v[66:73] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[34:41], v[116:123] /*v[372:379]*/, v[132:139] /*v[388:395]*/, v[34:41] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[2:9], v[4:11] /*v[260:267]*/, v[132:139] /*v[388:395]*/, v[2:9] matrix_b_reuse
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[106:113], v[4:11] /*v[260:267]*/, v[140:147] /*v[396:403]*/, v[106:113] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[26:33], v[116:123] /*v[372:379]*/, v[140:147] /*v[396:403]*/, v[26:33] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[58:65], v[108:115] /*v[364:371]*/, v[140:147] /*v[396:403]*/, v[58:65] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[90:97], v[100:107] /*v[356:363]*/, v[140:147] /*v[396:403]*/, v[90:97] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[130:137], v[44:51] /*v[300:307]*/, v[140:147] /*v[396:403]*/, v[130:137] matrix_b_reuse
	s_set_vgpr_msb 0x504                    ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[162:169], v[252:259], v[140:147] /*v[396:403]*/, v[162:169] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[194:201], v[244:251], v[140:147] /*v[396:403]*/, v[194:201] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(14) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x405                    ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[218:225], v[12:19] /*v[268:275]*/, v[76:83] /*v[332:339]*/, v[218:225]
	; sched_group_barrier mask(0x00000100) size(11) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[186:193], v[20:27] /*v[276:283]*/, v[76:83] /*v[332:339]*/, v[186:193] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[154:161], v[28:35] /*v[284:291]*/, v[76:83] /*v[332:339]*/, v[154:161] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[122:129], v[36:43] /*v[292:299]*/, v[76:83] /*v[332:339]*/, v[122:129] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[82:89], v[52:59] /*v[308:315]*/, v[76:83] /*v[332:339]*/, v[82:89] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[50:57], v[60:67] /*v[316:323]*/, v[76:83] /*v[332:339]*/, v[50:57] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[18:25], v[68:75] /*v[324:331]*/, v[76:83] /*v[332:339]*/, v[18:25] matrix_b_reuse
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[10:17], v[68:75] /*v[324:331]*/, v[84:91] /*v[340:347]*/, v[10:17] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[42:49], v[60:67] /*v[316:323]*/, v[84:91] /*v[340:347]*/, v[42:49] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[74:81], v[52:59] /*v[308:315]*/, v[84:91] /*v[340:347]*/, v[74:81] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[114:121], v[36:43] /*v[292:299]*/, v[84:91] /*v[340:347]*/, v[114:121] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[146:153], v[28:35] /*v[284:291]*/, v[84:91] /*v[340:347]*/, v[146:153] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[178:185], v[20:27] /*v[276:283]*/, v[84:91] /*v[340:347]*/, v[178:185] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[210:217], v[12:19] /*v[268:275]*/, v[84:91] /*v[340:347]*/, v[210:217] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(14) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(11) SyncID(0)
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[202:209], v[12:19] /*v[268:275]*/, v[92:99] /*v[348:355]*/, v[202:209] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[170:177], v[20:27] /*v[276:283]*/, v[92:99] /*v[348:355]*/, v[170:177] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[138:145], v[28:35] /*v[284:291]*/, v[92:99] /*v[348:355]*/, v[138:145] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[98:105], v[36:43] /*v[292:299]*/, v[92:99] /*v[348:355]*/, v[98:105] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[66:73], v[52:59] /*v[308:315]*/, v[92:99] /*v[348:355]*/, v[66:73] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[34:41], v[60:67] /*v[316:323]*/, v[92:99] /*v[348:355]*/, v[34:41] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[2:9], v[68:75] /*v[324:331]*/, v[92:99] /*v[348:355]*/, v[2:9] matrix_b_reuse
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[106:113], v[68:75] /*v[324:331]*/, v[124:131] /*v[380:387]*/, v[106:113] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[26:33], v[60:67] /*v[316:323]*/, v[124:131] /*v[380:387]*/, v[26:33] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[58:65], v[52:59] /*v[308:315]*/, v[124:131] /*v[380:387]*/, v[58:65] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[90:97], v[36:43] /*v[292:299]*/, v[124:131] /*v[380:387]*/, v[90:97] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[130:137], v[28:35] /*v[284:291]*/, v[124:131] /*v[380:387]*/, v[130:137] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[162:169], v[20:27] /*v[276:283]*/, v[124:131] /*v[380:387]*/, v[162:169] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[194:201], v[12:19] /*v[268:275]*/, v[124:131] /*v[380:387]*/, v[194:201] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(14) SyncID(0)
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
	v_mul_u32_u24_e32 v1, 0x70, v231
	v_lshrrev_b32_e32 v226, 1, v0
	v_and_or_b32 v228, v0, 15, v233
	v_cvt_pk_bf16_f32 v217, v216, v217
	v_cvt_pk_bf16_f32 v216, v214, v215
	v_cvt_pk_bf16_f32 v215, v212, v213
	v_and_or_b32 v1, v226, 8, v1
	v_cvt_pk_bf16_f32 v214, v210, v211
	v_cvt_pk_bf16_f32 v225, v224, v225
	v_cvt_pk_bf16_f32 v224, v222, v223
	v_cvt_pk_bf16_f32 v223, v220, v221
	v_mad_u32_u24 v1, 0xe0, v228, v1
	v_cvt_pk_bf16_f32 v222, v218, v219
	v_cvt_pk_bf16_f32 v65, v64, v65
	v_cvt_pk_bf16_f32 v64, v62, v63
	v_cvt_pk_bf16_f32 v63, v60, v61
	v_lshrrev_b32_e32 v210, 3, v1
	v_add_nc_u32_e32 v211, 0xe00, v1
	v_add_nc_u32_e32 v212, 0x1c00, v1
	v_lshlrev_b32_e32 v213, 1, v1
	v_add_nc_u32_e32 v60, 0xe50, v1
	v_and_b32_e32 v210, 0x3ff0, v210
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_dual_lshrrev_b32 v211, 3, v211 :: v_dual_lshrrev_b32 v212, 3, v212
	v_cvt_pk_bf16_f32 v201, v200, v201
	v_cvt_pk_bf16_f32 v200, v198, v199
	v_add_nc_u32_e32 v210, v210, v213
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v211, 0x7ff0, v211
	v_and_b32_e32 v212, 0x7ff0, v212
	v_cvt_pk_bf16_f32 v199, v196, v197
	v_add_nc_u32_e32 v196, 0xe10, v1
	v_cvt_pk_bf16_f32 v137, v136, v137
	v_cvt_pk_bf16_f32 v136, v134, v135
	v_cvt_pk_bf16_f32 v135, v132, v133
	v_add_nc_u32_e32 v132, 0xe30, v1
	v_add_nc_u32_e32 v218, 0x2a00, v1
	ds_store_b128 v210, v[222:225]
	v_dual_add_nc_u32 v210, v211, v213 :: v_dual_add_nc_u32 v211, v212, v213
	v_add_nc_u32_e32 v212, 16, v1
	v_cvt_pk_bf16_f32 v193, v192, v193
	v_cvt_pk_bf16_f32 v192, v190, v191
	v_cvt_pk_bf16_f32 v191, v188, v189
	v_cvt_pk_bf16_f32 v185, v184, v185
	v_cvt_pk_bf16_f32 v184, v182, v183
	v_add_nc_u32_e32 v188, 0x2a10, v1
	v_cvt_pk_bf16_f32 v183, v180, v181
	v_cvt_pk_bf16_f32 v177, v176, v177
	v_cvt_pk_bf16_f32 v176, v174, v175
	v_add_nc_u32_e32 v180, 32, v1
	v_cvt_pk_bf16_f32 v175, v172, v173
	v_add_nc_u32_e32 v172, 0xe20, v1
	v_cvt_pk_bf16_f32 v161, v160, v161
	v_cvt_pk_bf16_f32 v160, v158, v159
	v_cvt_pk_bf16_f32 v159, v156, v157
	v_add_nc_u32_e32 v157, 0x2a20, v1
	v_cvt_pk_bf16_f32 v153, v152, v153
	v_cvt_pk_bf16_f32 v152, v150, v151
	v_cvt_pk_bf16_f32 v151, v148, v149
	v_add_nc_u32_e32 v148, 48, v1
	v_cvt_pk_bf16_f32 v129, v128, v129
	v_cvt_pk_bf16_f32 v128, v126, v127
	v_cvt_pk_bf16_f32 v127, v124, v125
	v_cvt_pk_bf16_f32 v121, v120, v121
	v_cvt_pk_bf16_f32 v120, v118, v119
	v_add_nc_u32_e32 v124, 0x2a30, v1
	v_cvt_pk_bf16_f32 v119, v116, v117
	v_cvt_pk_bf16_f32 v105, v104, v105
	v_cvt_pk_bf16_f32 v104, v102, v103
	v_add_nc_u32_e32 v116, 64, v1
	v_cvt_pk_bf16_f32 v103, v100, v101
	v_add_nc_u32_e32 v100, 0xe40, v1
	v_cvt_pk_bf16_f32 v89, v88, v89
	v_cvt_pk_bf16_f32 v88, v86, v87
	v_cvt_pk_bf16_f32 v87, v84, v85
	v_add_nc_u32_e32 v85, 0x2a40, v1
	v_cvt_pk_bf16_f32 v81, v80, v81
	v_cvt_pk_bf16_f32 v80, v78, v79
	v_cvt_pk_bf16_f32 v79, v76, v77
	v_add_nc_u32_e32 v76, 0x50, v1
	v_cvt_pk_bf16_f32 v62, v58, v59
	v_lshrrev_b32_e32 v59, 3, v60
	v_add_nc_u32_e32 v60, 0x1c50, v1
	v_cvt_pk_bf16_f32 v57, v56, v57
	v_cvt_pk_bf16_f32 v56, v54, v55
	v_cvt_pk_bf16_f32 v55, v52, v53
	v_add_nc_u32_e32 v52, 0x2a50, v1
	v_cvt_pk_bf16_f32 v198, v194, v195
	v_lshrrev_b32_e32 v195, 3, v196
	v_add_nc_u32_e32 v196, 0x1c10, v1
	v_add_nc_u32_e32 v156, 0x1c20, v1
	v_cvt_pk_bf16_f32 v134, v130, v131
	v_lshrrev_b32_e32 v131, 3, v132
	v_add_nc_u32_e32 v132, 0x1c30, v1
	v_add_nc_u32_e32 v84, 0x1c40, v1
	v_cvt_pk_bf16_f32 v41, v40, v41
	v_cvt_pk_bf16_f32 v40, v38, v39
	v_cvt_pk_bf16_f32 v39, v36, v37
	v_add_nc_u32_e32 v36, 0x60, v1
	v_add_nc_u32_e32 v37, 0xe60, v1
	v_cvt_pk_bf16_f32 v33, v32, v33
	v_cvt_pk_bf16_f32 v32, v30, v31
	v_cvt_pk_bf16_f32 v31, v28, v29
	v_add_nc_u32_e32 v28, 0x1c60, v1
	v_add_nc_u32_e32 v1, 0x2a60, v1
	v_cvt_pk_bf16_f32 v209, v208, v209
	v_lshrrev_b32_e32 v218, 3, v218
	v_cvt_pk_bf16_f32 v208, v206, v207
	v_cvt_pk_bf16_f32 v206, v202, v203
	v_lshrrev_b32_e32 v203, 3, v212
	v_cvt_pk_bf16_f32 v182, v178, v179
	v_lshrrev_b32_e32 v179, 3, v188
	v_cvt_pk_bf16_f32 v174, v170, v171
	v_lshrrev_b32_e32 v171, 3, v180
	v_cvt_pk_bf16_f32 v169, v168, v169
	v_cvt_pk_bf16_f32 v168, v166, v167
	v_cvt_pk_bf16_f32 v166, v162, v163
	v_dual_lshrrev_b32 v163, 3, v172 :: v_dual_lshrrev_b32 v157, 3, v157
	v_cvt_pk_bf16_f32 v145, v144, v145
	v_cvt_pk_bf16_f32 v144, v142, v143
	v_cvt_pk_bf16_f32 v142, v138, v139
	v_lshrrev_b32_e32 v139, 3, v148
	v_cvt_pk_bf16_f32 v118, v114, v115
	v_lshrrev_b32_e32 v115, 3, v124
	v_cvt_pk_bf16_f32 v102, v98, v99
	v_lshrrev_b32_e32 v99, 3, v116
	v_cvt_pk_bf16_f32 v97, v96, v97
	v_cvt_pk_bf16_f32 v96, v94, v95
	v_cvt_pk_bf16_f32 v94, v90, v91
	v_dual_lshrrev_b32 v91, 3, v100 :: v_dual_lshrrev_b32 v85, 3, v85
	v_cvt_pk_bf16_f32 v73, v72, v73
	v_cvt_pk_bf16_f32 v72, v70, v71
	v_cvt_pk_bf16_f32 v70, v66, v67
	v_lshrrev_b32_e32 v67, 3, v76
	v_lshrrev_b32_e32 v60, 3, v60
	v_cvt_pk_bf16_f32 v49, v48, v49
	v_cvt_pk_bf16_f32 v48, v46, v47
	v_cvt_pk_bf16_f32 v46, v42, v43
	v_lshrrev_b32_e32 v43, 3, v52
	v_lshrrev_b32_e32 v196, 3, v196
	v_lshrrev_b32_e32 v156, 3, v156
	v_lshrrev_b32_e32 v132, 3, v132
	v_lshrrev_b32_e32 v84, 3, v84
	v_cvt_pk_bf16_f32 v38, v34, v35
	v_lshrrev_b32_e32 v35, 3, v36
	v_cvt_pk_bf16_f32 v30, v26, v27
	v_lshrrev_b32_e32 v26, 3, v37
	v_cvt_pk_bf16_f32 v25, v24, v25
	v_cvt_pk_bf16_f32 v24, v22, v23
	v_cvt_pk_bf16_f32 v23, v20, v21
	v_dual_lshrrev_b32 v20, 3, v28 :: v_dual_lshrrev_b32 v1, 3, v1
	v_and_b32_e32 v218, 0x7ff0, v218
	v_and_b32_e32 v194, 0x7ff0, v203
	v_and_b32_e32 v195, 0x7ff0, v195
	v_lshlrev_b32_e32 v197, 1, v212
	v_and_b32_e32 v170, 0x7ff0, v179
	v_and_b32_e32 v171, 0x7ff0, v171
	v_cvt_pk_bf16_f32 v158, v154, v155
	v_and_b32_e32 v154, 0x7ff0, v163
	v_lshlrev_b32_e32 v155, 1, v180
	v_cvt_pk_bf16_f32 v150, v146, v147
	v_and_b32_e32 v147, 0x7ff0, v157
	v_and_b32_e32 v130, 0x7ff0, v139
	v_and_b32_e32 v131, 0x7ff0, v131
	v_lshlrev_b32_e32 v133, 1, v148
	v_and_b32_e32 v98, 0x7ff0, v115
	v_and_b32_e32 v99, 0x7ff0, v99
	v_cvt_pk_bf16_f32 v86, v82, v83
	v_and_b32_e32 v82, 0x7ff0, v91
	v_lshlrev_b32_e32 v83, 1, v116
	v_cvt_pk_bf16_f32 v78, v74, v75
	v_and_b32_e32 v75, 0x7ff0, v85
	v_and_b32_e32 v58, 0x7ff0, v67
	v_lshlrev_b32_e32 v61, 1, v76
	v_cvt_pk_bf16_f32 v54, v50, v51
	v_and_b32_e32 v51, 0x7ff0, v60
	v_and_b32_e32 v34, 0x7ff0, v43
	v_cvt_pk_bf16_f32 v190, v186, v187
	v_and_b32_e32 v187, 0x7ff0, v196
	v_and_b32_e32 v156, 0x7ff0, v156
	v_cvt_pk_bf16_f32 v126, v122, v123
	v_and_b32_e32 v123, 0x7ff0, v132
	v_and_b32_e32 v84, 0x7ff0, v84
	v_and_b32_e32 v59, 0x7ff0, v59
	v_and_b32_e32 v35, 0x7ff0, v35
	v_and_b32_e32 v26, 0x7ff0, v26
	v_lshlrev_b32_e32 v29, 1, v36
	v_cvt_pk_bf16_f32 v22, v18, v19
	v_and_b32_e32 v19, 0x7ff0, v20
	v_and_b32_e32 v1, 0x7ff0, v1
	v_add_nc_u32_e32 v202, v218, v213
	v_add_nc_u32_e32 v194, v194, v213
	v_dual_add_nc_u32 v186, v195, v197 :: v_dual_add_nc_u32 v170, v170, v197
	v_cvt_pk_bf16_f32 v167, v164, v165
	v_dual_add_nc_u32 v162, v171, v213 :: v_dual_add_nc_u32 v154, v154, v155
	v_dual_add_nc_u32 v138, v147, v155 :: v_dual_add_nc_u32 v130, v130, v213
	v_dual_add_nc_u32 v122, v131, v133 :: v_dual_add_nc_u32 v98, v98, v133
	v_cvt_pk_bf16_f32 v95, v92, v93
	v_dual_add_nc_u32 v90, v99, v213 :: v_dual_add_nc_u32 v82, v82, v83
	v_dual_add_nc_u32 v66, v75, v83 :: v_dual_add_nc_u32 v58, v58, v213
	v_dual_add_nc_u32 v42, v51, v61 :: v_dual_add_nc_u32 v34, v34, v61
	v_cvt_pk_bf16_f32 v207, v204, v205
	v_dual_add_nc_u32 v178, v187, v197 :: v_dual_add_nc_u32 v146, v156, v155
	v_cvt_pk_bf16_f32 v143, v140, v141
	v_dual_add_nc_u32 v114, v123, v133 :: v_dual_add_nc_u32 v74, v84, v83
	v_cvt_pk_bf16_f32 v71, v68, v69
	v_add_nc_u32_e32 v50, v59, v61
	v_cvt_pk_bf16_f32 v47, v44, v45
	v_add_nc_u32_e32 v27, v35, v213
	v_add_nc_u32_e32 v18, v26, v29
	v_cvt_pk_bf16_f32 v17, v16, v17
	v_cvt_pk_bf16_f32 v16, v14, v15
	v_cvt_pk_bf16_f32 v15, v12, v13
	v_cvt_pk_bf16_f32 v14, v10, v11
	v_add_nc_u32_e32 v10, v19, v29
	v_cvt_pk_bf16_f32 v9, v8, v9
	v_cvt_pk_bf16_f32 v8, v6, v7
	v_cvt_pk_bf16_f32 v7, v4, v5
	v_cvt_pk_bf16_f32 v6, v2, v3
	v_add_nc_u32_e32 v1, v1, v29
	v_cvt_pk_bf16_f32 v5, v112, v113
	v_cvt_pk_bf16_f32 v4, v110, v111
	v_cvt_pk_bf16_f32 v3, v108, v109
	v_cvt_pk_bf16_f32 v2, v106, v107
	ds_store_b128 v210, v[214:217] offset:7168
	ds_store_b128 v211, v[206:209] offset:14336
	ds_store_b128 v202, v[198:201] offset:21504
	ds_store_b128 v194, v[190:193] offset:32
	ds_store_b128 v186, v[182:185] offset:7168
	ds_store_b128 v178, v[174:177] offset:14336
	ds_store_b128 v170, v[166:169] offset:21504
	ds_store_b128 v162, v[158:161] offset:64
	ds_store_b128 v154, v[150:153] offset:7168
	ds_store_b128 v146, v[142:145] offset:14336
	ds_store_b128 v138, v[134:137] offset:21504
	ds_store_b128 v130, v[126:129] offset:96
	ds_store_b128 v122, v[118:121] offset:7168
	ds_store_b128 v114, v[102:105] offset:14336
	ds_store_b128 v98, v[94:97] offset:21504
	ds_store_b128 v90, v[86:89] offset:128
	ds_store_b128 v82, v[78:81] offset:7168
	ds_store_b128 v74, v[70:73] offset:14336
	ds_store_b128 v66, v[62:65] offset:21504
	ds_store_b128 v58, v[54:57] offset:160
	ds_store_b128 v50, v[46:49] offset:7168
	ds_store_b128 v42, v[38:41] offset:14336
	ds_store_b128 v34, v[30:33] offset:21504
	ds_store_b128 v27, v[22:25] offset:192
	ds_store_b128 v18, v[14:17] offset:7168
	ds_store_b128 v10, v[6:9] offset:14336
	ds_store_b128 v1, v[2:5] offset:21504
.LBB0_39:
	v_cmp_ne_u32_e32 vcc_lo, 1, v227
	;;#ASMSTART
	s_wait_dscnt 0
	;;#ASMEND
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_vccnz .LBB0_50
; %bb.40:
	s_mul_i32 s3, s29, s3
	s_mov_b32 s0, exec_lo
	v_cmpx_gt_i32_e64 s3, v0
	s_cbranch_execz .LBB0_50
; %bb.41:
	s_mul_i32 s0, s36, 0xe0
	v_xad_u32 v2, v0, -1, s3
	s_ashr_i32 s1, s0, 31
	s_ashr_i32 s25, s24, 31
	s_lshl_b64 s[0:1], s[0:1], 1
                                        ; implicit-def: $vgpr1
                                        ; implicit-def: $vgpr6
                                        ; implicit-def: $sgpr12_sgpr13
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[4:5], s[22:23], s[0:1]
	s_mov_b32 s0, 0
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_u32_e32 0x2ff, v2
	s_xor_b32 s14, exec_lo, s1
	s_cbranch_execnz .LBB0_44
; %bb.42:
	s_or_saveexec_b32 s1, s14
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
	s_abs_i32 s15, s27
	v_lshrrev_b32_e32 v1, 8, v2
	s_cvt_f32_u32 s0, s15
	v_or_b32_e32 v3, 0x300, v0
	s_sub_co_i32 s1, 0, s15
	v_mov_b32_e32 v7, 0
	v_rcp_iflag_f32_e32 v2, s0
	v_add_nc_u32_e32 v8, 1, v1
	v_or_b32_e32 v1, 0x100, v0
	s_mov_b32 s13, 0
	s_mov_b32 s6, s24
	s_mov_b32 s7, s25
	v_and_b32_e32 v9, 0x1fffffc, v8
	s_mov_b32 s8, s24
	v_readfirstlane_b32 s0, v2
	v_or_b32_e32 v2, 0x200, v0
	s_mov_b32 s9, s25
	v_mov_b32_e32 v10, v9
	s_mov_b32 s10, s24
	s_mul_f32 s0, s0, 0x4f7ffffe
	v_mov_b64_e32 v[4:5], v[2:3]
	v_mov_b64_e32 v[2:3], v[0:1]
	s_mov_b32 s11, s25
	s_cvt_u32_f32 s0, s0
	s_mov_b32 s16, s27
	s_mov_b32 s17, s27
	s_mov_b32 s18, s27
	s_mul_i32 s1, s1, s0
	s_mov_b32 s19, s33
	s_mul_hi_u32 s1, s0, s1
	s_mov_b32 s20, s33
	s_mov_b32 s21, s33
	s_ashr_i32 s22, s27, 31
	s_add_co_i32 s12, s0, s1
	s_mov_b32 s23, s13
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
	v_xor_b32_e32 v1, s22, v1
	v_xor_b32_e32 v11, s22, v11
	v_mul_lo_u32 v24, v20, s15
	v_mul_lo_u32 v26, v21, s15
	v_mul_lo_u32 v27, v22, s15
	v_dual_add_nc_u32 v25, 1, v20 :: v_dual_add_nc_u32 v29, 1, v21
	v_mul_lo_u32 v28, v23, s15
	v_dual_add_nc_u32 v30, 1, v22 :: v_dual_add_nc_u32 v31, 1, v23
	v_dual_sub_nc_u32 v6, v6, v24 :: v_dual_sub_nc_u32 v12, v12, v26
	v_dual_sub_nc_u32 v16, v16, v27 :: v_dual_bitop2_b32 v14, s22, v14 bitop3:0x14
	v_xor_b32_e32 v18, s22, v18
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cmp_le_u32_e32 vcc_lo, s15, v6
	v_cmp_le_u32_e64 s0, s15, v12
	v_subrev_nc_u32_e32 v24, s15, v6
	v_cmp_le_u32_e64 s1, s15, v16
	v_subrev_nc_u32_e32 v26, s15, v16
	v_cndmask_b32_e32 v20, v20, v25, vcc_lo
	v_subrev_nc_u32_e32 v25, s15, v12
	v_dual_cndmask_b32 v21, v21, v29, s0 :: v_dual_sub_nc_u32 v19, v19, v28
	v_cndmask_b32_e64 v22, v22, v30, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_cndmask_b32 v6, v6, v24, vcc_lo :: v_dual_cndmask_b32 v12, v12, v25, s0
	v_dual_add_nc_u32 v25, 1, v21 :: v_dual_cndmask_b32 v16, v16, v26, s1
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_cmp_le_u32_e64 s2, s15, v19
	v_subrev_nc_u32_e32 v27, s15, v19
	v_cmp_le_u32_e32 vcc_lo, s15, v12
	v_dual_add_nc_u32 v26, 1, v22 :: v_dual_add_nc_u32 v24, 1, v20
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v23, v23, v31, s2
	v_dual_cndmask_b32 v19, v19, v27, s2 :: v_dual_cndmask_b32 v12, v21, v25, vcc_lo
	v_cmp_le_u32_e32 vcc_lo, s15, v16
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_add_nc_u32 v10, -4, v10 :: v_dual_add_nc_u32 v27, 1, v23
	v_dual_mov_b32 v13, v7 :: v_dual_bitop2_b32 v12, v12, v11 bitop3:0x14
	v_cndmask_b32_e32 v16, v22, v26, vcc_lo
	v_cmp_le_u32_e32 vcc_lo, s15, v6
	v_dual_mov_b32 v15, v7 :: v_dual_mov_b32 v17, v7
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sub_nc_u32_e32 v11, v12, v11
	v_xor_b32_e32 v16, v16, v14
	v_cndmask_b32_e32 v6, v20, v24, vcc_lo
	v_cmp_le_u32_e32 vcc_lo, s15, v19
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mul_lo_u32 v12, v11, s16
	v_dual_sub_nc_u32 v26, v16, v14 :: v_dual_bitop2_b32 v6, v6, v1 bitop3:0x14
	v_cndmask_b32_e32 v19, v23, v27, vcc_lo
	v_add_nc_u32_e32 v20, s19, v11
	v_cmp_eq_u32_e32 vcc_lo, 0, v10
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mul_lo_u32 v14, v26, s17
	v_dual_sub_nc_u32 v1, v6, v1 :: v_dual_bitop2_b32 v19, v19, v18 bitop3:0x14
	v_dual_sub_nc_u32 v12, v3, v12 :: v_dual_add_nc_u32 v22, s20, v26
	v_ashrrev_i32_e32 v21, 31, v20
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mul_lo_u32 v6, v1, s27
	v_dual_sub_nc_u32 v27, v19, v18 :: v_dual_add_nc_u32 v18, s33, v1
	v_sub_nc_u32_e32 v14, v4, v14
	v_mad_u32 v11, 0xe0, v11, v12
	v_ashrrev_i32_e32 v23, 31, v22
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_mul_lo_u32 v16, v27, s18
	v_dual_add_nc_u32 v24, s21, v27 :: v_dual_sub_nc_u32 v6, v2, v6
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
	s_or_b32 s23, vcc_lo, s23
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
	s_and_not1_b32 exec_lo, exec_lo, s23
	s_cbranch_execnz .LBB0_45
; %bb.46:
	s_or_b32 exec_lo, exec_lo, s23
	v_cmp_ne_u32_e32 vcc_lo, v8, v9
	v_lshl_or_b32 v0, v9, 8, v0
	v_dual_mov_b32 v6, s15 :: v_dual_mov_b32 v1, s22
	s_and_b32 s0, vcc_lo, exec_lo
	s_or_saveexec_b32 s1, s14
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
	v_dual_sub_nc_u32 v4, v4, v8 :: v_dual_add_nc_u32 v8, s33, v7
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
	.size	bm224_bn256_bk128_wm2_wn4_mc0, .Lfunc_end0-bm224_bn256_bk128_wm2_wn4_mc0
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel bm224_bn256_bk128_wm2_wn4_mc0
		.amdhsa_group_segment_fixed_size 261120
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
		.amdhsa_next_free_vgpr 404
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
	.section	.text.bm224_bn256_bk128_wm2_wn4_mc0,"axG",@progbits,bm224_bn256_bk128_wm2_wn4_mc0,comdat
                                        ; -- End function
	.set .Lbm224_bn256_bk128_wm2_wn4_mc0.num_vgpr, 404
	.set .Lbm224_bn256_bk128_wm2_wn4_mc0.num_agpr, 0
	.set .Lbm224_bn256_bk128_wm2_wn4_mc0.numbered_sgpr, 56
	.set .Lbm224_bn256_bk128_wm2_wn4_mc0.num_named_barrier, 0
	.set .Lbm224_bn256_bk128_wm2_wn4_mc0.private_seg_size, 0
	.set .Lbm224_bn256_bk128_wm2_wn4_mc0.uses_vcc, 1
	.set .Lbm224_bn256_bk128_wm2_wn4_mc0.uses_flat_scratch, 1
	.set .Lbm224_bn256_bk128_wm2_wn4_mc0.has_dyn_sized_stack, 0
	.set .Lbm224_bn256_bk128_wm2_wn4_mc0.has_recursion, 0
	.set .Lbm224_bn256_bk128_wm2_wn4_mc0.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 9664
; TotalNumSgprs: 58
; NumVgprs: 404
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 261120 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 25
; NumSGPRsForWavesPerEU: 58
; NumVGPRsForWavesPerEU: 404
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
	.type	__hip_cuid_dbf712cf7666b467,@object ; @__hip_cuid_dbf712cf7666b467
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_dbf712cf7666b467
__hip_cuid_dbf712cf7666b467:
	.byte	0                               ; 0x0
	.size	__hip_cuid_dbf712cf7666b467, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git aa451e1fe6a793394d6733051b1778633063ae96+PATCHED:d17c5aa0e3ea29cde402f58f27e39b6034effa27)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_dbf712cf7666b467
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
    macrotile: [224, 256, 128]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, One]
  MatrixInstruction: [16, 16, 32, 1]
  EnableMatrixInstruction: True
  MIWaveTile: [7, 4]
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
    .group_segment_fixed_size: 261120
    .kernarg_segment_align: 8
    .kernarg_segment_size: 132
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           bm224_bn256_bk128_wm2_wn4_mc0
    .private_segment_fixed_size: 0
    .sgpr_count:     58
    .sgpr_spill_count: 0
    .symbol:         bm224_bn256_bk128_wm2_wn4_mc0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     404
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
