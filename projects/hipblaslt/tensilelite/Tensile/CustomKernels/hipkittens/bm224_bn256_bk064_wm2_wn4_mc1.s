	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.section	.text.bm224_bn256_bk064_wm2_wn4_mc1,"axG",@progbits,bm224_bn256_bk064_wm2_wn4_mc1,comdat
	.protected	bm224_bn256_bk064_wm2_wn4_mc1 ; -- Begin function bm224_bn256_bk064_wm2_wn4_mc1
	.globl	bm224_bn256_bk064_wm2_wn4_mc1
	.p2align	8
	.type	bm224_bn256_bk064_wm2_wn4_mc1,@function
bm224_bn256_bk064_wm2_wn4_mc1: ; @bm224_bn256_bk064_wm2_wn4_mc1
; %bb.0:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1 ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_load_b96 s[28:30], s[0:1], 0x78 nv
	s_mov_b64 s[2:3], src_shared_base
	s_movk_i32 s2, 0x7700
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_and_b64 s[2:3], s[2:3], 12
	s_sub_co_i32 s4, 16, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_lshr_b32 s4, s4, 2
	s_cmp_lg_u64 s[2:3], 0
	s_cselect_b32 s5, s4, 0
	s_and_b32 s2, ttmp6, 15
	s_bfe_u32 s3, ttmp6, 0x40004
	s_lshl2_add_u32 s40, ttmp9, s2
	s_lshl2_add_u32 s4, ttmp7, s3
	s_mul_i32 s2, s40, 0xffffff20
	s_wait_kmcnt 0x0
	s_add_co_i32 s3, s28, 0xdf
	s_add_co_i32 s6, s29, 0xff
	s_mul_hi_i32 s7, s3, 0x92492493
	s_add_co_i32 s2, s28, s2
	s_ashr_i32 s8, s6, 31
	s_add_co_i32 s7, s7, s3
	s_min_i32 s31, s2, 0xe0
	s_lshr_b32 s2, s8, 24
	s_lshr_b32 s3, s7, 31
	s_ashr_i32 s7, s7, 7
	s_add_co_i32 s2, s6, s2
	s_add_co_i32 s6, s7, s3
	s_ashr_i32 s7, s2, 8
	s_cmp_lt_i32 s40, s6
	s_mov_b32 s9, s30
	s_cselect_b32 s41, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_and_b32 s2, s41, exec_lo
	s_cselect_b32 s3, s31, 0
	s_lshl_b32 s33, s4, 8
	s_sub_co_i32 s2, s29, s33
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_min_i32 s2, s2, 0x100
	s_cmp_lt_i32 s4, s7
	s_cselect_b32 s42, -1, 0
	s_and_b32 s8, s42, exec_lo
	s_cselect_b32 s35, s2, 0
	s_add_co_i32 s12, s30, 63
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 2, 1), 1
	s_min_i32 s2, s30, 64
	s_cmp_gt_i32 s12, 63
	s_cselect_b32 s13, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_and_b32 s8, s13, exec_lo
	s_cselect_b32 s2, s2, 0
	s_cmp_lt_i32 s3, 0xe0
	s_cselect_b32 s43, -1, 0
	s_and_b32 vcc_lo, exec_lo, s43
	s_mov_b32 s8, s43
	s_cbranch_vccnz .LBB0_2
; %bb.1:
	s_cmp_lt_i32 s35, 0x100
	s_cselect_b32 s8, -1, 0
	s_cmp_lt_i32 s2, 64
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
	v_cmp_lt_u32_e32 vcc_lo, 0x1cbf, v5
	s_or_b32 s8, vcc_lo, s8
	s_and_not1_b32 exec_lo, exec_lo, s8
	s_cbranch_execnz .LBB0_4
; %bb.5:
	s_or_b32 exec_lo, exec_lo, s8
	v_lshl_add_u32 v2, s5, 2, v2
	v_mov_b32_e32 v3, 0
	s_mov_b32 s8, 0
.LBB0_6:                                ; =>This Inner Loop Header: Depth=1
	v_add_nc_u32_e32 v1, 0x100, v1
	ds_store_b32 v2, v3 offset:30464
	v_add_nc_u32_e32 v2, 0x400, v2
	v_cmp_lt_u32_e32 vcc_lo, 0x20ff, v1
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
	s_load_b64 s[14:15], s[0:1], 0x0 nv
	s_load_b128 s[24:27], s[0:1], 0x20 nv
	s_load_b128 s[20:23], s[0:1], 0x48 nv
	s_wait_xcnt 0x0
	s_lshl_b32 s1, s4, 2
	v_lshrrev_b32_e32 v235, 5, v0
	s_lshl_b32 s16, s5, 2
	s_add_co_i32 s7, s7, -1
	s_and_b32 s1, s1, 12
	s_mov_b64 s[36:37], src_shared_base
	s_or_b32 s36, s16, 0x7700
	s_add_co_i32 s17, s6, -1
	s_min_i32 s0, s4, s7
	s_and_b32 s18, s40, 3
	s_lshl_b32 s1, 15, s1
	s_mov_b32 s4, exec_lo
	v_cmpx_lt_i32_e32 0, v235
	s_xor_b32 s19, exec_lo, s4
	s_cbranch_execz .LBB0_12
; %bb.9:
	s_mov_b32 s29, exec_lo
	v_cmpx_eq_u32_e32 1, v235
	s_cbranch_execz .LBB0_11
; %bb.10:
	s_cmp_gt_i32 s2, 0
	s_mov_b32 s34, s2
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
	s_add_nc_u64 s[6:7], s[26:27], s[4:5]
	v_dual_mov_b32 v1, s36 :: v_dual_mov_b32 v4, s6
	s_and_b32 s5, s7, 0x1ffffff
	s_and_b32 s7, s42, s8
	s_bitset1_b32 s5, 31
	v_cndmask_b32_e64 v2, 0, 1, s7
	v_mov_b32_e32 v3, s5
	s_lshr_b64 s[6:7], s[34:35], 16
	v_readfirstlane_b32 s45, v1
	v_readfirstlane_b32 s46, v4
	v_readfirstlane_b32 s44, v2
	v_readfirstlane_b32 s47, v3
	s_lshr_b32 s7, s35, 16
	s_or_b32 s4, s1, 0x7510000
	s_lshl_b32 s5, s2, 16
	s_bitset1_b32 s7, 22
	s_movk_i32 s8, 0x100
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[44:47], s[4:11]
.LBB0_11:
	s_or_b32 exec_lo, exec_lo, s29
.LBB0_12:
	s_or_saveexec_b32 s29, s19
	s_min_i32 s4, s40, s17
	s_lshl_b32 s17, 0x1111, s18
	s_mul_i32 s18, s4, 0xe0
	s_xor_b32 exec_lo, exec_lo, s29
	s_cbranch_execz .LBB0_14
; %bb.13:
	s_cmp_gt_i32 s2, 0
	s_mov_b32 s10, 0
	s_cselect_b32 s6, -1, 0
	s_ashr_i32 s19, s18, 31
	s_wait_kmcnt 0x0
	s_bfe_i64 s[4:5], s[24:25], 0x200000
	s_and_b32 s6, s41, s6
	s_mul_u64 s[4:5], s[4:5], s[18:19]
	v_cndmask_b32_e64 v2, 0, 1, s6
	s_lshl_b64 s[4:5], s[4:5], 1
	s_lshr_b64 s[6:7], s[2:3], 16
	s_add_nc_u64 s[4:5], s[14:15], s[4:5]
	s_movk_i32 s8, 0xe0
	s_and_b32 s5, s5, 0x1ffffff
	v_readfirstlane_b32 s44, v2
	s_bitset1_b32 s5, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v4, s4 :: v_dual_mov_b32 v3, s5
	s_lshl_b32 s5, s2, 16
	s_lshr_b32 s2, s3, 16
	s_or_b32 s4, s17, 0x7510000
	v_readfirstlane_b32 s46, v4
	v_readfirstlane_b32 s47, v3
	s_or_b32 s7, s2, 0x400000
	s_mov_b32 s11, s10
	s_mov_b32 s45, s10
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[44:47], s[4:11]
.LBB0_14:
	s_or_b32 exec_lo, exec_lo, s29
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s2, exec_lo
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	v_cmpx_gt_u32_e32 32, v0
	s_cbranch_execz .LBB0_16
; %bb.15:
	s_barrier_signal -3
.LBB0_16:
	s_or_b32 exec_lo, exec_lo, s2
	v_dual_lshlrev_b32 v1, 6, v235 :: v_dual_lshrrev_b32 v231, 7, v0
	v_mov_b32_e32 v9, 0
	s_and_b32 s29, s41, s42
	s_and_not1_b32 vcc_lo, exec_lo, s13
	s_delay_alu instid0(VALU_DEP_2)
	v_and_b32_e32 v233, 0xc0, v1
	v_cndmask_b32_e64 v227, 0, 1, s29
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
	v_dual_mov_b32 v98, v9 :: v_dual_mov_b32 v121, v9
	v_dual_mov_b32 v120, v9 :: v_dual_mov_b32 v119, v9
	v_dual_mov_b32 v118, v9 :: v_dual_mov_b32 v117, v9
	v_dual_mov_b32 v116, v9 :: v_dual_mov_b32 v115, v9
	v_dual_mov_b32 v114, v9 :: v_dual_mov_b32 v129, v9
	v_dual_mov_b32 v128, v9 :: v_dual_mov_b32 v127, v9
	v_dual_mov_b32 v126, v9 :: v_dual_mov_b32 v125, v9
	v_dual_mov_b32 v124, v9 :: v_dual_mov_b32 v123, v9
	v_dual_mov_b32 v122, v9 :: v_dual_mov_b32 v137, v9
	v_dual_mov_b32 v136, v9 :: v_dual_mov_b32 v135, v9
	v_dual_mov_b32 v134, v9 :: v_dual_mov_b32 v133, v9
	v_dual_mov_b32 v132, v9 :: v_dual_mov_b32 v131, v9
	v_dual_mov_b32 v130, v9 :: v_dual_mov_b32 v145, v9
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
	v_dual_mov_b32 v218, v9 :: v_dual_mov_b32 v113, v9
	v_dual_mov_b32 v112, v9 :: v_dual_mov_b32 v111, v9
	v_dual_mov_b32 v110, v9 :: v_dual_mov_b32 v109, v9
	v_dual_mov_b32 v108, v9 :: v_dual_mov_b32 v107, v9
	v_mov_b32_e32 v106, v9
	s_barrier_wait -1
	s_barrier_wait -3
	s_cbranch_vccnz .LBB0_41
; %bb.17:
	v_dual_lshlrev_b32 v24, 6, v233 :: v_dual_bitop2_b32 v2, 16, v0 bitop3:0x40
	v_mul_u32_u24_e32 v1, 0x1c00, v231
	v_lshlrev_b32_e32 v3, 6, v0
	s_mov_b64 s[4:5], src_shared_base
	s_or_b32 s6, s16, 0xff00
	s_mov_b32 s7, s5
	s_mov_b32 s11, 0
	v_and_or_b32 v25, 0x3c0, v3, v2
	s_and_b64 s[6:7], s[6:7], 15
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_lshl_or_b32 v18 /*v274*/, v0, 2, 0x8800
	s_sub_co_i32 s2, 16, s6
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_or_b32_e32 v241, 0x2100, v0
	v_or_b32_e32 v7, v25, v1
	s_lshr_b32 s2, s2, 2
	s_cmp_lg_u64 s[6:7], 0
	s_mov_b32 s44, s37
	s_cselect_b32 s2, s2, 0
	v_lshrrev_b32_e32 v2, 4, v7
	s_lshl2_add_u32 s2, s2, s16
	v_dual_mov_b32 v229, 0 :: v_dual_add_nc_u32 v4, 0x800, v7
	v_add_nc_u32_e32 v10, 0x1400, v7
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_and_b32_e32 v3, 0x1f8, v2
	v_add_nc_u32_e32 v2, 0x400, v7
	v_add_nc_u32_e32 v12, 0x1800, v7
	v_dual_lshrrev_b32 v14, 4, v10 :: v_dual_bitop2_b32 v13, v24, v25 bitop3:0x54
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_add_nc_u32 v226, v3, v7 :: v_dual_lshrrev_b32 v3, 4, v2
	v_dual_lshrrev_b32 v15, 4, v12 :: v_dual_lshrrev_b32 v16, 4, v13
	v_or_b32_e32 v18, 0x400, v13
	v_add_nc_u32_e32 v6, 0xc00, v7
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v3, 0x3f8, v3
	v_or_b32_e32 v17, 32, v25
	v_and_b32_e32 v19, 0x3f8, v14
	v_and_b32_e32 v15, 0x3f8, v15
	v_and_b32_e32 v14, 0x338, v16
	v_dual_mov_b32 v3, v229 :: v_dual_add_nc_u32 v232, v3, v7
	v_dual_lshrrev_b32 v5, 4, v4 :: v_dual_lshrrev_b32 v16, 4, v18
	v_add_nc_u32_e32 v8, 0x1000, v7
	v_or_b32_e32 v18, 0x800, v13
	v_or_b32_e32 v1, v17, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_and_b32_e32 v5, 0x3f8, v5
	v_and_b32_e32 v21, 0x378, v16
	v_dual_mov_b32 v15, v229 :: v_dual_add_nc_u32 v242, v15, v7
	v_or_b32_e32 v20, 0xc00, v13
	v_add_nc_u32_e32 v234, v5, v7
	v_dual_mov_b32 v5, v229 :: v_dual_lshrrev_b32 v9, 4, v6
	v_dual_lshrrev_b32 v11, 4, v8 :: v_dual_add_nc_u32 v230, v14, v13
	v_lshrrev_b32_e32 v16, 4, v18
	v_add_nc_u32_e32 v14, 0x400, v1
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v9, 0x3f8, v9
	v_lshrrev_b32_e32 v18, 4, v20
	v_add_nc_u32_e32 v22, 0x800, v1
	v_dual_mov_b32 v19, v229 :: v_dual_add_nc_u32 v240, v19, v7
	v_dual_lshrrev_b32 v20, 4, v1 :: v_dual_lshrrev_b32 v14, 4, v14
	v_add_nc_u32_e32 v236, v9, v7
	v_and_b32_e32 v23, 0x3b8, v16
	v_and_b32_e32 v26, 0x3f8, v18
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v27, 0x1f8, v20
	v_and_b32_e32 v228, 0x3f8, v14
	v_add_nc_u32_e32 v16, 0xc00, v1
	v_add_nc_u32_e32 v18, 0x1000, v1
	v_add_nc_u32_e32 v20, 0x1400, v1
	v_add_nc_u32_e32 v1, 0x1800, v1
	v_add_nc_u32_e32 v244, v21, v13
	v_dual_mov_b32 v9, v229 :: v_dual_mov_b32 v21, v229
	v_dual_lshrrev_b32 v14, 4, v22 :: v_dual_lshrrev_b32 v16, 4, v16
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_lshrrev_b32 v1, 4, v1 :: v_dual_bitop2_b32 v17, v17, v24 bitop3:0x54
	v_sub_nc_u32_e32 v29, 0x1dbf, v0
	v_and_b32_e32 v14, 0x3f8, v14
	s_add_co_i32 s4, s2, 0x17600
	v_and_b32_e32 v11, 0x3f8, v11
	s_and_b32 s10, s4, 15
	v_and_b32_e32 v22, 0x3f8, v1
	v_or_b32_e32 v1, 0x400, v17
	v_or_b32_e32 v28, 0x800, v17
	v_or_b32_e32 v30, 0xc00, v17
	v_lshrrev_b32_e32 v17, 4, v17
	v_add_nc_u64_e32 v[254:255], v[14:15], v[4:5]
	v_dual_mov_b32 v4, v229 :: v_dual_lshrrev_b32 v29, 8, v29
	v_add_nc_u32_e32 v238, v11, v7
	s_sub_co_i32 s6, 16, s10
	s_add_co_i32 s45, s2, 0xff00
	s_lshr_b32 s2, s6, 2
	s_cmp_lg_u64 s[10:11], 0
	v_lshrrev_b32_e32 v28, 4, v28
	v_and_b32_e32 v31, 0x338, v17
	v_dual_add_nc_u32 v17, 1, v29 :: v_dual_add_nc_u32 v250, v27, v7
	s_cselect_b32 s2, s2, 0
	s_ashr_i32 s6, s12, 31
	v_dual_add_nc_u32 v246, v23, v13 :: v_dual_add_nc_u32 v248, v26, v13
	s_lshr_b32 s6, s6, 26
	v_dual_mov_b32 v13, v229 :: v_dual_bitop2_b32 v237, 30, v17 bitop3:0x40
	s_add_co_i32 s12, s12, s6
	v_dual_mov_b32 v11, v229 :: v_dual_mov_b32 v23, v229
	v_dual_lshrrev_b32 v18, 4, v18 :: v_dual_lshrrev_b32 v20, 4, v20
	v_dual_lshrrev_b32 v1, 4, v1 :: v_dual_lshrrev_b32 v30, 4, v30
	s_lshl_b32 s10, s2, 2
	s_ashr_i32 s47, s12, 6
	s_cmp_lt_i32 s35, 0x100
	v_and_b32_e32 v16, 0x3f8, v16
	s_cselect_b32 s48, -1, 0
	s_or_b32 s12, s1, 0x7510000
	v_and_b32_e32 v18, 0x3f8, v18
	v_and_b32_e32 v20, 0x3f8, v20
	v_cmp_ne_u32_e64 s1, v17, v237
	v_dual_mov_b32 v17, v229 :: v_dual_mov_b32 v7, v229
	s_lshl_b32 s6, s0, 8
	v_and_b32_e32 v32, 0x3f8, v1
	v_and_b32_e32 v28, 0x3f8, v28
	v_and_b32_e32 v29, 0x3f8, v30
	v_lshl_or_b32 v30, v237, 8, v0
	v_add_nc_u64_e32 v[252:253], v[228:229], v[2:3]
	v_add_nc_u32_e32 v2, v25, v24
	s_ashr_i32 s7, s6, 31
	s_ashr_i32 s19, s18, 31
	s_wait_kmcnt 0x0
	s_bfe_i64 s[20:21], s[20:21], 0x200000
	s_bfe_i64 s[24:25], s[24:25], 0x200000
	s_mul_u64 s[6:7], s[20:21], s[6:7]
	s_mul_u64 s[18:19], s[24:25], s[18:19]
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_add_nc_u64_e32 v[0:1] /*v[256:257]*/, v[16:17], v[6:7]
	v_add_nc_u64_e32 v[2:3] /*v[258:259]*/, v[18:19], v[8:9]
	v_add_nc_u64_e32 v[4:5] /*v[260:261]*/, v[20:21], v[10:11]
	v_add_nc_u64_e32 v[6:7] /*v[262:263]*/, v[22:23], v[12:13]
	v_cmp_eq_u32_e64 s0, 0, v235
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_or_b32_e32 v1, 0x100, v0
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_dual_mov_b32 v17 /*v273*/, v229 :: v_dual_add_nc_u32 v8 /*v264*/, v31, v2
	v_add3_u32 v10 /*v266*/, v2, v32, 0x400
	v_add3_u32 v12 /*v268*/, v2, v28, 0x800
	v_add3_u32 v14 /*v270*/, v2, v29, 0xc00
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_dual_mov_b32 v6, v229 :: v_dual_add_nc_u32 v239, 0xffffff00, v30
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_dual_mov_b32 v19 /*v275*/, v229 :: v_dual_lshlrev_b32 v16 /*v272*/, 2, v30
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_dual_mov_b32 v2, v229 :: v_dual_mov_b32 v8, v229
	v_dual_mov_b32 v10, v229 :: v_dual_mov_b32 v12, v229
	v_dual_mov_b32 v14, v229 :: v_dual_mov_b32 v16, v229
	v_dual_mov_b32 v18, v229 :: v_dual_mov_b32 v20, v229
	v_dual_mov_b32 v22, v229 :: v_dual_mov_b32 v24, v229
	v_dual_mov_b32 v25, v229 :: v_dual_mov_b32 v26, v229
	v_dual_mov_b32 v27, v229 :: v_dual_mov_b32 v28, v229
	v_dual_mov_b32 v29, v229 :: v_dual_mov_b32 v30, v229
	v_dual_mov_b32 v31, v229 :: v_dual_mov_b32 v32, v229
	v_dual_mov_b32 v33, v229 :: v_dual_mov_b32 v34, v229
	v_dual_mov_b32 v35, v229 :: v_dual_mov_b32 v36, v229
	v_dual_mov_b32 v37, v229 :: v_dual_mov_b32 v38, v229
	v_dual_mov_b32 v39, v229 :: v_dual_mov_b32 v40, v229
	v_dual_mov_b32 v41, v229 :: v_dual_mov_b32 v42, v229
	v_dual_mov_b32 v43, v229 :: v_dual_mov_b32 v44, v229
	v_dual_mov_b32 v45, v229 :: v_dual_mov_b32 v46, v229
	v_dual_mov_b32 v47, v229 :: v_dual_mov_b32 v48, v229
	v_dual_mov_b32 v49, v229 :: v_dual_mov_b32 v50, v229
	v_dual_mov_b32 v51, v229 :: v_dual_mov_b32 v52, v229
	v_dual_mov_b32 v53, v229 :: v_dual_mov_b32 v54, v229
	v_dual_mov_b32 v55, v229 :: v_dual_mov_b32 v56, v229
	v_dual_mov_b32 v57, v229 :: v_dual_mov_b32 v58, v229
	v_dual_mov_b32 v59, v229 :: v_dual_mov_b32 v60, v229
	v_dual_mov_b32 v61, v229 :: v_dual_mov_b32 v62, v229
	v_dual_mov_b32 v63, v229 :: v_dual_mov_b32 v64, v229
	v_dual_mov_b32 v65, v229 :: v_dual_mov_b32 v66, v229
	v_dual_mov_b32 v67, v229 :: v_dual_mov_b32 v68, v229
	v_dual_mov_b32 v69, v229 :: v_dual_mov_b32 v70, v229
	v_dual_mov_b32 v71, v229 :: v_dual_mov_b32 v72, v229
	v_dual_mov_b32 v73, v229 :: v_dual_mov_b32 v74, v229
	v_dual_mov_b32 v75, v229 :: v_dual_mov_b32 v76, v229
	v_dual_mov_b32 v77, v229 :: v_dual_mov_b32 v78, v229
	v_dual_mov_b32 v79, v229 :: v_dual_mov_b32 v80, v229
	v_dual_mov_b32 v81, v229 :: v_dual_mov_b32 v82, v229
	v_dual_mov_b32 v83, v229 :: v_dual_mov_b32 v84, v229
	v_dual_mov_b32 v85, v229 :: v_dual_mov_b32 v86, v229
	v_dual_mov_b32 v87, v229 :: v_dual_mov_b32 v88, v229
	v_dual_mov_b32 v89, v229 :: v_dual_mov_b32 v90, v229
	v_dual_mov_b32 v91, v229 :: v_dual_mov_b32 v92, v229
	v_dual_mov_b32 v93, v229 :: v_dual_mov_b32 v94, v229
	v_dual_mov_b32 v95, v229 :: v_dual_mov_b32 v96, v229
	v_dual_mov_b32 v97, v229 :: v_dual_mov_b32 v98, v229
	v_dual_mov_b32 v99, v229 :: v_dual_mov_b32 v100, v229
	v_dual_mov_b32 v101, v229 :: v_dual_mov_b32 v102, v229
	v_dual_mov_b32 v103, v229 :: v_dual_mov_b32 v104, v229
	v_dual_mov_b32 v105, v229 :: v_dual_mov_b32 v114, v229
	v_dual_mov_b32 v115, v229 :: v_dual_mov_b32 v116, v229
	v_dual_mov_b32 v117, v229 :: v_dual_mov_b32 v118, v229
	v_dual_mov_b32 v119, v229 :: v_dual_mov_b32 v120, v229
	v_dual_mov_b32 v121, v229 :: v_dual_mov_b32 v122, v229
	v_dual_mov_b32 v123, v229 :: v_dual_mov_b32 v124, v229
	v_dual_mov_b32 v125, v229 :: v_dual_mov_b32 v126, v229
	v_dual_mov_b32 v127, v229 :: v_dual_mov_b32 v128, v229
	v_dual_mov_b32 v129, v229 :: v_dual_mov_b32 v130, v229
	v_dual_mov_b32 v131, v229 :: v_dual_mov_b32 v132, v229
	v_dual_mov_b32 v133, v229 :: v_dual_mov_b32 v134, v229
	v_dual_mov_b32 v135, v229 :: v_dual_mov_b32 v136, v229
	v_dual_mov_b32 v137, v229 :: v_dual_mov_b32 v138, v229
	v_dual_mov_b32 v139, v229 :: v_dual_mov_b32 v140, v229
	v_dual_mov_b32 v141, v229 :: v_dual_mov_b32 v142, v229
	v_dual_mov_b32 v143, v229 :: v_dual_mov_b32 v144, v229
	v_dual_mov_b32 v145, v229 :: v_dual_mov_b32 v146, v229
	v_dual_mov_b32 v147, v229 :: v_dual_mov_b32 v148, v229
	v_dual_mov_b32 v149, v229 :: v_dual_mov_b32 v150, v229
	v_dual_mov_b32 v151, v229 :: v_dual_mov_b32 v152, v229
	v_dual_mov_b32 v153, v229 :: v_dual_mov_b32 v154, v229
	v_dual_mov_b32 v155, v229 :: v_dual_mov_b32 v156, v229
	v_dual_mov_b32 v157, v229 :: v_dual_mov_b32 v158, v229
	v_dual_mov_b32 v159, v229 :: v_dual_mov_b32 v160, v229
	v_dual_mov_b32 v161, v229 :: v_dual_mov_b32 v162, v229
	v_dual_mov_b32 v163, v229 :: v_dual_mov_b32 v164, v229
	v_dual_mov_b32 v165, v229 :: v_dual_mov_b32 v166, v229
	v_dual_mov_b32 v167, v229 :: v_dual_mov_b32 v168, v229
	v_dual_mov_b32 v169, v229 :: v_dual_mov_b32 v170, v229
	v_dual_mov_b32 v171, v229 :: v_dual_mov_b32 v172, v229
	v_dual_mov_b32 v173, v229 :: v_dual_mov_b32 v174, v229
	v_dual_mov_b32 v175, v229 :: v_dual_mov_b32 v176, v229
	v_dual_mov_b32 v177, v229 :: v_dual_mov_b32 v178, v229
	v_dual_mov_b32 v179, v229 :: v_dual_mov_b32 v180, v229
	v_dual_mov_b32 v181, v229 :: v_dual_mov_b32 v182, v229
	v_dual_mov_b32 v183, v229 :: v_dual_mov_b32 v184, v229
	v_dual_mov_b32 v185, v229 :: v_dual_mov_b32 v186, v229
	v_dual_mov_b32 v187, v229 :: v_dual_mov_b32 v188, v229
	v_dual_mov_b32 v189, v229 :: v_dual_mov_b32 v190, v229
	v_dual_mov_b32 v191, v229 :: v_dual_mov_b32 v192, v229
	v_dual_mov_b32 v193, v229 :: v_dual_mov_b32 v194, v229
	v_dual_mov_b32 v195, v229 :: v_dual_mov_b32 v196, v229
	v_dual_mov_b32 v197, v229 :: v_dual_mov_b32 v198, v229
	v_dual_mov_b32 v199, v229 :: v_dual_mov_b32 v200, v229
	v_dual_mov_b32 v201, v229 :: v_dual_mov_b32 v202, v229
	v_dual_mov_b32 v203, v229 :: v_dual_mov_b32 v204, v229
	v_dual_mov_b32 v205, v229 :: v_dual_mov_b32 v206, v229
	v_dual_mov_b32 v207, v229 :: v_dual_mov_b32 v208, v229
	v_dual_mov_b32 v209, v229 :: v_dual_mov_b32 v210, v229
	v_dual_mov_b32 v211, v229 :: v_dual_mov_b32 v212, v229
	v_dual_mov_b32 v213, v229 :: v_dual_mov_b32 v214, v229
	v_dual_mov_b32 v215, v229 :: v_dual_mov_b32 v216, v229
	v_dual_mov_b32 v217, v229 :: v_dual_mov_b32 v218, v229
	v_dual_mov_b32 v219, v229 :: v_dual_mov_b32 v220, v229
	v_dual_mov_b32 v221, v229 :: v_dual_mov_b32 v222, v229
	v_dual_mov_b32 v223, v229 :: v_dual_mov_b32 v224, v229
	v_dual_mov_b32 v225, v229 :: v_dual_mov_b32 v106, v229
	v_dual_mov_b32 v107, v229 :: v_dual_mov_b32 v108, v229
	v_dual_mov_b32 v109, v229 :: v_dual_mov_b32 v110, v229
	v_dual_mov_b32 v111, v229 :: v_dual_mov_b32 v112, v229
	v_mov_b32_e32 v113, v229
	s_lshr_b32 s49, s35, 16
	s_lshr_b32 s50, s3, 16
	s_lshl_b64 s[6:7], s[6:7], 1
	s_lshl_b64 s[18:19], s[18:19], 1
	s_movk_i32 s16, 0x100
	s_mov_b32 s46, s5
	s_add_nc_u64 s[38:39], s[4:5], s[10:11]
	s_bitset1_b32 s49, 22
	s_movk_i32 s8, 0xe0
	s_or_b32 s4, s17, 0x7510000
	s_bitset1_b32 s50, 22
	s_add_nc_u64 s[20:21], s[26:27], s[6:7]
	s_add_nc_u64 s[24:25], s[14:15], s[18:19]
	s_mov_b32 s26, s11
	s_branch .LBB0_19
.LBB0_18:                               ;   in Loop: Header=BB0_19 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
	s_cmp_eq_u32 s26, s47
	s_barrier_wait -1
	s_barrier_wait -3
	s_cbranch_scc1 .LBB0_41
.LBB0_19:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_21 Depth 2
                                        ;     Child Loop BB0_24 Depth 2
                                        ;     Child Loop BB0_26 Depth 2
                                        ;     Child Loop BB0_29 Depth 2
	s_and_b32 s27, s26, 1
	s_add_co_i32 s26, s26, 1
	s_xor_b32 s5, s27, 1
	s_lshl_b32 s2, s26, 6
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_sub_co_i32 s2, s30, s2
	s_min_i32 s2, s2, 64
	s_cmp_lt_i32 s26, s47
	s_cselect_b32 s10, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_and_b32 s6, s10, exec_lo
	s_cselect_b32 s2, s2, 0
	s_cmp_lt_i32 s2, 64
	s_cselect_b32 s6, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b32 s6, s48, s6
	s_or_b32 s6, s43, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s6
	s_cbranch_vccnz .LBB0_31
; %bb.20:                               ;   in Loop: Header=BB0_19 Depth=1
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_mov_b64_e32 v[20:21] /*v[276:277]*/, v[0:1]
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_mov_b32_e32 v243, v237
	s_cmp_lg_u32 s5, 0
	s_mov_b32 s13, 0
	s_cselect_b32 s7, s46, s37
	s_cselect_b32 s6, s45, 0
.LBB0_21:                               ;   Parent Loop BB0_19 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_set_vgpr_msb 1                        ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_dual_mov_b32 v228, v20 /*v276*/ :: v_dual_add_nc_u32 v243, -2, v243
	s_set_vgpr_msb 0x145                    ;  msbs: dst=1 src0=1 src1=1 src2=0
	v_dual_mov_b32 v22 /*v278*/, v21 /*v277*/ :: v_dual_add_nc_u32 v21 /*v277*/, 0x200, v21 /*v277*/
	s_set_vgpr_msb 0x4544                   ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_dual_mov_b32 v23 /*v279*/, v229 :: v_dual_add_nc_u32 v20 /*v276*/, 0x200, v20 /*v276*/
	s_set_vgpr_msb 0x4440                   ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_cmp_eq_u32_e32 vcc_lo, 0, v243
	v_lshl_add_u64 v[24:25] /*v[280:281]*/, v[228:229], 2, s[6:7]
	s_set_vgpr_msb 0x4041                   ;  msbs: dst=1 src0=1 src1=0 src2=0
	v_lshl_add_u64 v[22:23] /*v[278:279]*/, v[22:23] /*v[278:279]*/, 2, s[6:7]
	s_clause 0x1
	flat_store_b32 v[24:25] /*v[280:281]*/, v229
	flat_store_b32 v[22:23] /*v[278:279]*/, v229
	s_or_b32 s13, vcc_lo, s13
	s_wait_xcnt 0x0
	s_and_not1_b32 exec_lo, exec_lo, s13
	s_set_vgpr_msb 0x4100                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_cbranch_execnz .LBB0_21
; %bb.22:                               ;   in Loop: Header=BB0_19 Depth=1
	s_or_b32 exec_lo, exec_lo, s13
	s_and_saveexec_b32 s13, s1
	s_cbranch_execz .LBB0_25
; %bb.23:                               ;   in Loop: Header=BB0_19 Depth=1
	s_set_vgpr_msb 0x44                     ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_add_nc_u64_e32 v[20:21] /*v[276:277]*/, s[6:7], v[16:17] /*v[272:273]*/
	s_set_vgpr_msb 0x4400                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_mov_b32_e32 v228, v239
	s_mov_b32 s6, 0
.LBB0_24:                               ;   Parent Loop BB0_19 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v228, 0x100, v228
	s_set_vgpr_msb 1                        ;  msbs: dst=0 src0=1 src1=0 src2=0
	flat_store_b32 v[20:21] /*v[276:277]*/, v229
	s_set_vgpr_msb 0x144                    ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_add_nc_u64_e32 v[20:21] /*v[276:277]*/, 0x400, v[20:21] /*v[276:277]*/
	s_set_vgpr_msb 0x4400                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_cmp_lt_u32_e32 vcc_lo, 0x1cbf, v228
	s_or_b32 s6, vcc_lo, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s6
	s_cbranch_execnz .LBB0_24
.LBB0_25:                               ;   in Loop: Header=BB0_19 Depth=1
	s_or_b32 exec_lo, exec_lo, s13
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_mov_b64_e32 v[20:21] /*v[276:277]*/, v[0:1]
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_mov_b32_e32 v243, 34
	s_cmp_lg_u32 s5, 0
	s_mov_b32 s13, 0
	s_cselect_b32 s7, s39, s44
	s_cselect_b32 s6, s38, s36
.LBB0_26:                               ;   Parent Loop BB0_19 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_set_vgpr_msb 1                        ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_dual_mov_b32 v228, v20 /*v276*/ :: v_dual_add_nc_u32 v243, -2, v243
	s_set_vgpr_msb 0x145                    ;  msbs: dst=1 src0=1 src1=1 src2=0
	v_dual_mov_b32 v22 /*v278*/, v21 /*v277*/ :: v_dual_add_nc_u32 v21 /*v277*/, 0x200, v21 /*v277*/
	s_set_vgpr_msb 0x4544                   ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_dual_mov_b32 v23 /*v279*/, v229 :: v_dual_add_nc_u32 v20 /*v276*/, 0x200, v20 /*v276*/
	s_set_vgpr_msb 0x4440                   ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_cmp_eq_u32_e32 vcc_lo, 0, v243
	v_lshl_add_u64 v[24:25] /*v[280:281]*/, v[228:229], 2, s[6:7]
	s_set_vgpr_msb 0x4041                   ;  msbs: dst=1 src0=1 src1=0 src2=0
	v_lshl_add_u64 v[22:23] /*v[278:279]*/, v[22:23] /*v[278:279]*/, 2, s[6:7]
	s_clause 0x1
	flat_store_b32 v[24:25] /*v[280:281]*/, v229
	flat_store_b32 v[22:23] /*v[278:279]*/, v229
	s_or_b32 s13, vcc_lo, s13
	s_wait_xcnt 0x0
	s_and_not1_b32 exec_lo, exec_lo, s13
	s_set_vgpr_msb 0x4100                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_cbranch_execnz .LBB0_26
; %bb.27:                               ;   in Loop: Header=BB0_19 Depth=1
	s_or_b32 exec_lo, exec_lo, s13
	s_and_saveexec_b32 s13, s11
	s_cbranch_execz .LBB0_30
; %bb.28:                               ;   in Loop: Header=BB0_19 Depth=1
	s_set_vgpr_msb 0x44                     ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_add_nc_u64_e32 v[20:21] /*v[276:277]*/, s[6:7], v[18:19] /*v[274:275]*/
	s_set_vgpr_msb 0x4400                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_mov_b32_e32 v228, v241
	s_mov_b32 s6, 0
.LBB0_29:                               ;   Parent Loop BB0_19 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v228, 0x100, v228
	s_set_vgpr_msb 1                        ;  msbs: dst=0 src0=1 src1=0 src2=0
	flat_store_b32 v[20:21] /*v[276:277]*/, v229
	s_set_vgpr_msb 0x144                    ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_add_nc_u64_e32 v[20:21] /*v[276:277]*/, 0x400, v[20:21] /*v[276:277]*/
	s_set_vgpr_msb 0x4400                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_cmp_lt_u32_e32 vcc_lo, 0x20ff, v228
	s_or_b32 s6, vcc_lo, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s6
	s_cbranch_execnz .LBB0_29
.LBB0_30:                               ;   in Loop: Header=BB0_19 Depth=1
	s_or_b32 exec_lo, exec_lo, s13
	s_barrier_signal -1
	s_barrier_wait -1
.LBB0_31:                               ;   in Loop: Header=BB0_19 Depth=1
	s_and_b32 s6, s10, exec_lo
	s_cselect_b32 s6, s26, 0
	s_mov_b32 s7, exec_lo
	v_cmpx_lt_i32_e32 0, v235
	s_xor_b32 s7, exec_lo, s7
	s_cbranch_execnz .LBB0_37
; %bb.32:                               ;   in Loop: Header=BB0_19 Depth=1
	s_and_not1_saveexec_b32 s13, s7
	s_cbranch_execnz .LBB0_40
.LBB0_33:                               ;   in Loop: Header=BB0_19 Depth=1
	s_or_b32 exec_lo, exec_lo, s13
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s29
	s_cbranch_vccnz .LBB0_35
.LBB0_34:                               ;   in Loop: Header=BB0_19 Depth=1
	s_cmp_lg_u32 s27, 0
	s_cselect_b32 s5, s45, 0
	s_cselect_b32 s2, s38, s36
	v_lshl_add_u32 v228, v226, 1, s5
	v_lshl_add_u32 v243, v232, 1, s5
	v_lshl_add_u32 v245, v234, 1, s5
	v_lshl_add_u32 v247, v236, 1, s5
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[20:23] /*v[276:279]*/, v228
	ds_load_b128 v[24:27] /*v[280:283]*/, v228 offset:16
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v228, v238, 1, s5
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[28:31] /*v[284:287]*/, v243 offset:2048
	ds_load_b128 v[32:35] /*v[288:291]*/, v243 offset:2064
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v243, v240, 1, s5
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[36:39] /*v[292:295]*/, v245 offset:4096
	ds_load_b128 v[52:55] /*v[308:311]*/, v228 offset:8192
	ds_load_b128 v[56:59] /*v[312:315]*/, v228 offset:8208
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v228, v246, 1, s2
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[40:43] /*v[296:299]*/, v245 offset:4112
	ds_load_b128 v[44:47] /*v[300:303]*/, v247 offset:6144
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v245, v248, 1, s2
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[48:51] /*v[304:307]*/, v247 offset:6160
	ds_load_b128 v[60:63] /*v[316:319]*/, v243 offset:10240
	ds_load_b128 v[76:79] /*v[332:335]*/, v228 offset:4096
	ds_load_b128 v[80:83] /*v[336:339]*/, v228 offset:4112
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v228, v242, 1, s5
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[68:71] /*v[324:327]*/, v245 offset:6144
	ds_load_b128 v[72:75] /*v[328:331]*/, v245 offset:6160
	ds_load_b128 v[64:67] /*v[320:323]*/, v243 offset:10256
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v243, v252, 1, s5
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[84:87] /*v[340:343]*/, v228 offset:12288
	ds_load_b128 v[88:91] /*v[344:347]*/, v228 offset:12304
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v228, v230, 1, s2
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x32_bf16 v[58:65], v[52:59] /*v[308:315]*/, v[68:75] /*v[324:331]*/, v[58:65] matrix_b_reuse
	s_set_vgpr_msb 0x500                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v245, v254, 1, s5
	; sched_group_barrier mask(0x00000100) size(11) SyncID(0)
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[90:97], v[44:51] /*v[300:307]*/, v[68:75] /*v[324:331]*/, v[90:97] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[130:137], v[36:43] /*v[292:299]*/, v[68:75] /*v[324:331]*/, v[130:137] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[162:169], v[28:35] /*v[284:291]*/, v[68:75] /*v[324:331]*/, v[162:169] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[194:201], v[20:27] /*v[276:283]*/, v[68:75] /*v[324:331]*/, v[194:201] matrix_b_reuse
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[106:113], v[84:91] /*v[340:347]*/, v[68:75] /*v[324:331]*/, v[106:113] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[26:33], v[60:67] /*v[316:323]*/, v[68:75] /*v[324:331]*/, v[26:33] matrix_b_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[68:71] /*v[324:327]*/, v228
	ds_load_b128 v[72:75] /*v[328:331]*/, v228 offset:16
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v228, v244, 1, s2
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[218:225], v[20:27] /*v[276:283]*/, v[68:75] /*v[324:331]*/, v[218:225]
	v_wmma_f32_16x16x32_bf16 v[186:193], v[28:35] /*v[284:291]*/, v[68:75] /*v[324:331]*/, v[186:193] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[154:161], v[36:43] /*v[292:299]*/, v[68:75] /*v[324:331]*/, v[154:161] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[122:129], v[44:51] /*v[300:307]*/, v[68:75] /*v[324:331]*/, v[122:129] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[82:89], v[52:59] /*v[308:315]*/, v[68:75] /*v[324:331]*/, v[82:89] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[50:57], v[60:67] /*v[316:323]*/, v[68:75] /*v[324:331]*/, v[50:57] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[18:25], v[84:91] /*v[340:347]*/, v[68:75] /*v[324:331]*/, v[18:25] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[202:209], v[20:27] /*v[276:283]*/, v[76:83] /*v[332:339]*/, v[202:209] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[170:177], v[28:35] /*v[284:291]*/, v[76:83] /*v[332:339]*/, v[170:177] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[138:145], v[36:43] /*v[292:299]*/, v[76:83] /*v[332:339]*/, v[138:145] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[98:105], v[44:51] /*v[300:307]*/, v[76:83] /*v[332:339]*/, v[98:105] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[66:73], v[52:59] /*v[308:315]*/, v[76:83] /*v[332:339]*/, v[66:73] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[2:9], v[84:91] /*v[340:347]*/, v[76:83] /*v[332:339]*/, v[2:9] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[34:41], v[60:67] /*v[316:323]*/, v[76:83] /*v[332:339]*/, v[34:41] matrix_b_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[76:79] /*v[332:335]*/, v228 offset:2048
	ds_load_b128 v[80:83] /*v[336:339]*/, v228 offset:2064
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v228, v250, 1, s5
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[68:71] /*v[324:327]*/, v228 offset:64
	s_set_vgpr_msb 0x4005                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x32_bf16 v[42:49], v[60:67] /*v[316:323]*/, v[76:83] /*v[332:339]*/, v[42:49] matrix_b_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[72:75] /*v[328:331]*/, v228 offset:80
	ds_load_b128 v[60:63] /*v[316:319]*/, v243 offset:64
	ds_load_b128 v[64:67] /*v[320:323]*/, v243 offset:80
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_lshl_add_u32 v228, v0 /*v256*/, 1, s5
	v_lshl_add_u32 v243, v2 /*v258*/, 1, s5
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[92:95] /*v[348:351]*/, v243 offset:64
	s_set_vgpr_msb 0x4005                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[10:17], v[84:91] /*v[340:347]*/, v[76:83] /*v[332:339]*/, v[10:17] matrix_a_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[84:87] /*v[340:343]*/, v245 offset:64
	ds_load_b128 v[88:91] /*v[344:347]*/, v245 offset:80
	ds_load_b128 v[96:99] /*v[352:355]*/, v243 offset:80
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_lshl_add_u32 v243, v6 /*v262*/, 1, s5
	v_lshl_add_u32 v245, v8 /*v264*/, 1, s2
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[100:103] /*v[356:359]*/, v243 offset:64
	s_set_vgpr_msb 0x4005                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[74:81], v[52:59] /*v[308:315]*/, v[76:83] /*v[332:339]*/, v[74:81] matrix_b_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[52:55] /*v[308:311]*/, v228 offset:64
	ds_load_b128 v[56:59] /*v[312:315]*/, v228 offset:80
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_lshl_add_u32 v228, v4 /*v260*/, 1, s5
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[104:107] /*v[360:363]*/, v243 offset:80
	ds_load_b128 v[108:111] /*v[364:367]*/, v245 offset:64
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_lshl_add_u32 v243, v12 /*v268*/, 1, s2
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[112:115] /*v[368:371]*/, v245 offset:80
	s_set_vgpr_msb 0x4005                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[114:121], v[44:51] /*v[300:307]*/, v[76:83] /*v[332:339]*/, v[114:121] matrix_b_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[44:47] /*v[300:303]*/, v228 offset:64
	ds_load_b128 v[48:51] /*v[304:307]*/, v228 offset:80
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_lshl_add_u32 v228, v10 /*v266*/, 1, s2
	v_lshl_add_u32 v245, v14 /*v270*/, 1, s2
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[116:119] /*v[372:375]*/, v243 offset:64
	ds_load_b128 v[120:123] /*v[376:379]*/, v243 offset:80
	ds_load_b128 v[124:127] /*v[380:383]*/, v245 offset:64
	s_set_vgpr_msb 0x4005                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[146:153], v[36:43] /*v[292:299]*/, v[76:83] /*v[332:339]*/, v[146:153] matrix_b_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[36:39] /*v[292:295]*/, v228 offset:64
	ds_load_b128 v[40:43] /*v[296:299]*/, v228 offset:80
	ds_load_b128 v[128:131] /*v[384:387]*/, v245 offset:80
	s_set_vgpr_msb 0x4005                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[178:185], v[28:35] /*v[284:291]*/, v[76:83] /*v[332:339]*/, v[178:185] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(14) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(11) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(14) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[210:217], v[20:27] /*v[276:283]*/, v[76:83] /*v[332:339]*/, v[210:217] matrix_b_reuse
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0x8
	v_wmma_f32_16x16x32_bf16 v[218:225], v[68:75] /*v[324:331]*/, v[108:115] /*v[364:371]*/, v[218:225]
	; sched_group_barrier mask(0x00000100) size(11) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[186:193], v[60:67] /*v[316:323]*/, v[108:115] /*v[364:371]*/, v[186:193] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[154:161], v[84:91] /*v[340:347]*/, v[108:115] /*v[364:371]*/, v[154:161] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[122:129], v[52:59] /*v[308:315]*/, v[108:115] /*v[364:371]*/, v[122:129] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[82:89], v[92:99] /*v[348:355]*/, v[108:115] /*v[364:371]*/, v[82:89] matrix_b_reuse
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[50:57], v[44:51] /*v[300:307]*/, v[108:115] /*v[364:371]*/, v[50:57] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[18:25], v[100:107] /*v[356:363]*/, v[108:115] /*v[364:371]*/, v[18:25] matrix_b_reuse
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x32_bf16 v[10:17], v[100:107] /*v[356:363]*/, v[36:43] /*v[292:299]*/, v[10:17] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[42:49], v[44:51] /*v[300:307]*/, v[36:43] /*v[292:299]*/, v[42:49] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[74:81], v[92:99] /*v[348:355]*/, v[36:43] /*v[292:299]*/, v[74:81] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[114:121], v[52:59] /*v[308:315]*/, v[36:43] /*v[292:299]*/, v[114:121] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[146:153], v[84:91] /*v[340:347]*/, v[36:43] /*v[292:299]*/, v[146:153] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[178:185], v[60:67] /*v[316:323]*/, v[36:43] /*v[292:299]*/, v[178:185] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[210:217], v[68:75] /*v[324:331]*/, v[36:43] /*v[292:299]*/, v[210:217] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(14) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(11) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[202:209], v[68:75] /*v[324:331]*/, v[116:123] /*v[372:379]*/, v[202:209] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[170:177], v[60:67] /*v[316:323]*/, v[116:123] /*v[372:379]*/, v[170:177] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[138:145], v[84:91] /*v[340:347]*/, v[116:123] /*v[372:379]*/, v[138:145] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[98:105], v[52:59] /*v[308:315]*/, v[116:123] /*v[372:379]*/, v[98:105] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[66:73], v[92:99] /*v[348:355]*/, v[116:123] /*v[372:379]*/, v[66:73] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[34:41], v[44:51] /*v[300:307]*/, v[116:123] /*v[372:379]*/, v[34:41] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[2:9], v[100:107] /*v[356:363]*/, v[116:123] /*v[372:379]*/, v[2:9] matrix_b_reuse
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[106:113], v[100:107] /*v[356:363]*/, v[124:131] /*v[380:387]*/, v[106:113] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[26:33], v[44:51] /*v[300:307]*/, v[124:131] /*v[380:387]*/, v[26:33] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[58:65], v[92:99] /*v[348:355]*/, v[124:131] /*v[380:387]*/, v[58:65] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[90:97], v[52:59] /*v[308:315]*/, v[124:131] /*v[380:387]*/, v[90:97] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[130:137], v[84:91] /*v[340:347]*/, v[124:131] /*v[380:387]*/, v[130:137] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[162:169], v[60:67] /*v[316:323]*/, v[124:131] /*v[380:387]*/, v[162:169] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[194:201], v[68:75] /*v[324:331]*/, v[124:131] /*v[380:387]*/, v[194:201] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(14) SyncID(0)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_wait_dscnt 0
	;;#ASMEND
	s_set_vgpr_msb 0x500                    ;  msbs: dst=0 src0=0 src1=0 src2=0
.LBB0_35:                               ;   in Loop: Header=BB0_19 Depth=1
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_and_saveexec_b32 s2, s0
	s_cbranch_execz .LBB0_18
; %bb.36:                               ;   in Loop: Header=BB0_19 Depth=1
	s_barrier_signal -3
	s_branch .LBB0_18
.LBB0_37:                               ;   in Loop: Header=BB0_19 Depth=1
	s_mov_b32 s51, exec_lo
	v_cmpx_eq_u32_e32 1, v235
	s_cbranch_execz .LBB0_39
; %bb.38:                               ;   in Loop: Header=BB0_19 Depth=1
	s_cmp_lg_u32 s5, 0
	s_mov_b32 s34, s2
	s_cselect_b32 s13, s38, s36
	s_cmp_gt_i32 s2, 0
	s_mov_b32 s18, s11
	s_cselect_b32 s17, -1, 0
	s_lshl_b32 s10, s6, 6
	s_mov_b32 s19, s11
	s_lshl_b64 s[14:15], s[10:11], 1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_nc_u64 s[14:15], s[20:21], s[14:15]
	s_and_b32 s10, s15, 0x1ffffff
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_mov_b32_e32 v20 /*v276*/, s14
	s_bitset1_b32 s10, 31
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_dual_mov_b32 v243, s13 :: v_dual_mov_b32 v245, s10
	s_and_b32 s13, s42, s17
	v_readfirstlane_b32 s54, v20 /*v276*/
	v_cndmask_b32_e64 v228, 0, 1, s13
	s_set_vgpr_msb 0x100                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_readfirstlane_b32 s53, v243
	v_readfirstlane_b32 s55, v245
	s_lshr_b64 s[14:15], s[34:35], 16
	s_lshl_b32 s13, s2, 16
	v_readfirstlane_b32 s52, v228
	s_mov_b32 s15, s49
	s_mov_b32 s17, s9
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[52:55], s[12:19]
.LBB0_39:                               ;   in Loop: Header=BB0_19 Depth=1
	s_or_b32 exec_lo, exec_lo, s51
	s_and_not1_saveexec_b32 s13, s7
	s_cbranch_execz .LBB0_33
.LBB0_40:                               ;   in Loop: Header=BB0_19 Depth=1
	s_cmp_lg_u32 s5, 0
	s_cselect_b32 s5, s45, 0
	s_cmp_gt_i32 s2, 0
	s_cselect_b32 s14, -1, 0
	s_lshl_b32 s10, s6, 6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_lshl_b64 s[6:7], s[10:11], 1
	s_and_b32 s10, s41, s14
	s_add_nc_u64 s[6:7], s[24:25], s[6:7]
	v_cndmask_b32_e64 v228, 0, 1, s10
	s_and_b32 s7, s7, 0x1ffffff
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_mov_b32_e32 v20 /*v276*/, s6
	s_bitset1_b32 s7, 31
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_dual_mov_b32 v243, s5 :: v_dual_mov_b32 v245, s7
	v_readfirstlane_b32 s52, v228
	s_set_vgpr_msb 1                        ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_readfirstlane_b32 s54, v20 /*v276*/
	s_lshr_b64 s[6:7], s[2:3], 16
	s_set_vgpr_msb 0x100                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_readfirstlane_b32 s53, v243
	v_readfirstlane_b32 s55, v245
	s_lshl_b32 s5, s2, 16
	s_mov_b32 s7, s50
	s_mov_b32 s10, s11
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	tensor_load_to_lds s[52:55], s[4:11]
	s_or_b32 exec_lo, exec_lo, s13
	s_and_not1_b32 vcc_lo, exec_lo, s29
	s_cbranch_vccz .LBB0_34
	s_branch .LBB0_35
.LBB0_41:
	s_wait_tensorcnt 0x0
	;;#ASMSTART
	s_wait_dscnt 0
	;;#ASMEND
	s_and_b32 vcc_lo, exec_lo, s29
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_vccz .LBB0_43
; %bb.42:
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
.LBB0_43:
	v_cmp_ne_u32_e32 vcc_lo, 1, v227
	;;#ASMSTART
	s_wait_dscnt 0
	;;#ASMEND
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_vccnz .LBB0_54
; %bb.44:
	s_mul_i32 s3, s35, s3
	s_mov_b32 s0, exec_lo
	v_cmpx_gt_i32_e64 s3, v0
	s_cbranch_execz .LBB0_54
; %bb.45:
	s_mul_i32 s0, s40, 0xe0
	v_xad_u32 v2, v0, -1, s3
	s_ashr_i32 s1, s0, 31
	s_ashr_i32 s29, s28, 31
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
	s_cbranch_execnz .LBB0_48
; %bb.46:
	s_or_saveexec_b32 s1, s14
	v_mov_b64_e32 v[2:3], s[12:13]
	s_xor_b32 exec_lo, exec_lo, s1
	s_cbranch_execnz .LBB0_51
.LBB0_47:
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 exec_lo, exec_lo, s0
	s_cbranch_execnz .LBB0_52
	s_branch .LBB0_54
.LBB0_48:
	s_abs_i32 s15, s31
	v_lshrrev_b32_e32 v1, 8, v2
	s_cvt_f32_u32 s0, s15
	v_or_b32_e32 v3, 0x300, v0
	s_sub_co_i32 s1, 0, s15
	v_mov_b32_e32 v7, 0
	v_rcp_iflag_f32_e32 v2, s0
	v_add_nc_u32_e32 v8, 1, v1
	v_or_b32_e32 v1, 0x100, v0
	s_mov_b32 s13, 0
	s_mov_b32 s6, s28
	s_mov_b32 s7, s29
	v_and_b32_e32 v9, 0x1fffffc, v8
	s_mov_b32 s8, s28
	v_readfirstlane_b32 s0, v2
	v_or_b32_e32 v2, 0x200, v0
	s_mov_b32 s9, s29
	v_mov_b32_e32 v10, v9
	s_mov_b32 s10, s28
	s_mul_f32 s0, s0, 0x4f7ffffe
	v_mov_b64_e32 v[4:5], v[2:3]
	v_mov_b64_e32 v[2:3], v[0:1]
	s_mov_b32 s11, s29
	s_cvt_u32_f32 s0, s0
	s_mov_b32 s16, s31
	s_mov_b32 s17, s31
	s_mov_b32 s18, s31
	s_mul_i32 s1, s1, s0
	s_mov_b32 s19, s33
	s_mul_hi_u32 s1, s0, s1
	s_mov_b32 s20, s33
	s_mov_b32 s21, s33
	s_ashr_i32 s22, s31, 31
	s_add_co_i32 s12, s0, s1
	s_mov_b32 s23, s13
.LBB0_49:                               ; =>This Inner Loop Header: Depth=1
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
	v_mul_lo_u32 v6, v1, s31
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
	v_mul_u64_e32 v[18:19], s[28:29], v[18:19]
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
	s_cbranch_execnz .LBB0_49
; %bb.50:
	s_or_b32 exec_lo, exec_lo, s23
	v_cmp_ne_u32_e32 vcc_lo, v8, v9
	v_lshl_or_b32 v0, v9, 8, v0
	v_dual_mov_b32 v6, s15 :: v_dual_mov_b32 v1, s22
	s_and_b32 s0, vcc_lo, exec_lo
	s_or_saveexec_b32 s1, s14
	v_mov_b64_e32 v[2:3], s[12:13]
	s_xor_b32 exec_lo, exec_lo, s1
	s_cbranch_execz .LBB0_47
.LBB0_51:
	s_abs_i32 s2, s31
	s_ashr_i32 s8, s31, 31
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
	s_cbranch_execz .LBB0_54
.LBB0_52:
	v_mov_b32_e32 v5, 0
	s_mov_b32 s0, 0
	s_sub_co_i32 s1, 0, s31
.LBB0_53:                               ; =>This Inner Loop Header: Depth=1
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
	v_mul_lo_u32 v8, v7, s31
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_sub_nc_u32 v4, v4, v8 :: v_dual_add_nc_u32 v8, s33, v7
	v_dual_sub_nc_u32 v4, v4, v9 :: v_dual_ashrrev_i32 v9, 31, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_nc_u32_e32 v4, v0, v4
	v_mul_u64_e32 v[8:9], s[28:29], v[8:9]
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
	s_cbranch_execnz .LBB0_53
.LBB0_54:
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end0:
	.size	bm224_bn256_bk064_wm2_wn4_mc1, .Lfunc_end0-bm224_bn256_bk064_wm2_wn4_mc1
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel bm224_bn256_bk064_wm2_wn4_mc1
		.amdhsa_group_segment_fixed_size 130560
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
		.amdhsa_next_free_vgpr 388
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
		.amdhsa_inst_pref_size 77
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text.bm224_bn256_bk064_wm2_wn4_mc1,"axG",@progbits,bm224_bn256_bk064_wm2_wn4_mc1,comdat
                                        ; -- End function
	.set .Lbm224_bn256_bk064_wm2_wn4_mc1.num_vgpr, 388
	.set .Lbm224_bn256_bk064_wm2_wn4_mc1.num_agpr, 0
	.set .Lbm224_bn256_bk064_wm2_wn4_mc1.numbered_sgpr, 56
	.set .Lbm224_bn256_bk064_wm2_wn4_mc1.num_named_barrier, 0
	.set .Lbm224_bn256_bk064_wm2_wn4_mc1.private_seg_size, 0
	.set .Lbm224_bn256_bk064_wm2_wn4_mc1.uses_vcc, 1
	.set .Lbm224_bn256_bk064_wm2_wn4_mc1.uses_flat_scratch, 1
	.set .Lbm224_bn256_bk064_wm2_wn4_mc1.has_dyn_sized_stack, 0
	.set .Lbm224_bn256_bk064_wm2_wn4_mc1.has_recursion, 0
	.set .Lbm224_bn256_bk064_wm2_wn4_mc1.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 9740
; TotalNumSgprs: 58
; NumVgprs: 388
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 130560 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 24
; NumSGPRsForWavesPerEU: 58
; NumVGPRsForWavesPerEU: 388
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
	.type	__hip_cuid_e8513156521a14a2,@object ; @__hip_cuid_e8513156521a14a2
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_e8513156521a14a2
__hip_cuid_e8513156521a14a2:
	.byte	0                               ; 0x0
	.size	__hip_cuid_e8513156521a14a2, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git aa451e1fe6a793394d6733051b1778633063ae96+PATCHED:d17c5aa0e3ea29cde402f58f27e39b6034effa27)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_e8513156521a14a2
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
    macrotile: [224, 256, 64]
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
    .cluster_dims:
      - 4
      - 4
      - 1
    .gfx1250_revision: B0
    .group_segment_fixed_size: 130560
    .kernarg_segment_align: 8
    .kernarg_segment_size: 132
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           bm224_bn256_bk064_wm2_wn4_mc1
    .private_segment_fixed_size: 0
    .sgpr_count:     58
    .sgpr_spill_count: 0
    .symbol:         bm224_bn256_bk064_wm2_wn4_mc1.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     388
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
