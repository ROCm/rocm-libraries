	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.section	.text.bm256_bn224_bk096_wm4_wn2_mc1,"axG",@progbits,bm256_bn224_bk096_wm4_wn2_mc1,comdat
	.protected	bm256_bn224_bk096_wm4_wn2_mc1 ; -- Begin function bm256_bn224_bk096_wm4_wn2_mc1
	.globl	bm256_bn224_bk096_wm4_wn2_mc1
	.p2align	8
	.type	bm256_bn224_bk096_wm4_wn2_mc1,@function
bm256_bn224_bk096_wm4_wn2_mc1: ; @bm256_bn224_bk096_wm4_wn2_mc1
; %bb.0:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1 ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_load_b96 s[28:30], s[0:1], 0x78 nv
	s_mov_b64 s[2:3], src_shared_base
	s_mov_b32 s2, 0xcc00
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_and_b64 s[2:3], s[2:3], 12
	s_sub_co_i32 s4, 16, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_lshr_b32 s4, s4, 2
	s_cmp_lg_u64 s[2:3], 0
	s_cselect_b32 s4, s4, 0
	s_and_b32 s2, ttmp6, 15
	s_bfe_u32 s5, ttmp6, 0x40004
	s_lshl2_add_u32 s3, ttmp9, s2
	s_lshl2_add_u32 s33, ttmp7, s5
	s_lshl_b32 s2, s3, 8
	s_wait_kmcnt 0x0
	s_add_co_i32 s5, s28, 0xff
	s_add_co_i32 s6, s29, 0xdf
	s_sub_co_i32 s7, s28, s2
	s_ashr_i32 s8, s5, 31
	s_mul_hi_i32 s9, s6, 0x92492493
	s_min_i32 s31, s7, 0x100
	s_lshr_b32 s7, s8, 24
	s_add_co_i32 s9, s9, s6
	s_add_co_i32 s5, s5, s7
	s_lshr_b32 s6, s9, 31
	s_ashr_i32 s7, s9, 7
	s_ashr_i32 s5, s5, 8
	s_add_co_i32 s6, s7, s6
	s_cmp_lt_i32 s3, s5
	s_mul_i32 s7, s33, 0xffffff20
	s_cselect_b32 s42, -1, 0
	s_mov_b32 s9, s30
	s_and_b32 s8, s42, exec_lo
	s_cselect_b32 s35, s31, 0
	s_add_co_i32 s7, s29, s7
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_min_i32 s7, s7, 0xe0
	s_cmp_lt_i32 s33, s6
	s_cselect_b32 s29, -1, 0
	s_and_b32 s8, s29, exec_lo
	s_cselect_b32 s37, s7, 0
	s_add_co_i32 s12, s30, 0x5f
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 2, 1), 1
	s_min_i32 s7, s30, 0x60
	s_cmp_gt_i32 s12, 0x5f
	s_cselect_b32 s13, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_and_b32 s8, s13, exec_lo
	s_cselect_b32 s34, s7, 0
	s_cmp_lt_i32 s35, 0x100
	s_cselect_b32 s43, -1, 0
	s_and_b32 vcc_lo, exec_lo, s43
	s_mov_b32 s7, s43
	s_cbranch_vccnz .LBB0_2
; %bb.1:
	s_cmp_lt_i32 s37, 0xe0
	s_cselect_b32 s7, -1, 0
	s_cmp_lt_i32 s34, 0x60
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
	v_cmp_lt_u32_e32 vcc_lo, 0x31ff, v5
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
	ds_store_b32 v2, v3 offset:52224
	v_add_nc_u32_e32 v2, 0x400, v2
	v_cmp_lt_u32_e32 vcc_lo, 0x2b9f, v1
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
	s_load_b64 s[14:15], s[0:1], 0x0 nv
	s_load_b128 s[24:27], s[0:1], 0x20 nv
	s_load_b128 s[20:23], s[0:1], 0x48 nv
	s_wait_xcnt 0x0
	s_lshl_b32 s1, s33, 2
	v_lshrrev_b32_e32 v235, 5, v0
	s_lshl_b32 s0, s4, 2
	s_add_co_i32 s6, s6, -1
	s_and_b32 s1, s1, 12
	s_mov_b64 s[38:39], src_shared_base
	s_or_b32 s38, s0, 0xcc00
	s_add_co_i32 s16, s5, -1
	s_min_i32 s0, s33, s6
	s_and_b32 s17, s3, 3
	s_lshl_b32 s1, 15, s1
	s_mov_b32 s4, exec_lo
	v_cmpx_lt_i32_e32 0, v235
	s_xor_b32 s18, exec_lo, s4
	s_cbranch_execz .LBB0_12
; %bb.9:
	s_mov_b32 s19, exec_lo
	v_cmpx_eq_u32_e32 1, v235
	s_cbranch_execz .LBB0_11
; %bb.10:
	s_cmp_gt_i32 s34, 0
	s_mul_i32 s4, s0, 0xe0
	s_cselect_b32 s8, -1, 0
	s_ashr_i32 s5, s4, 31
	s_wait_kmcnt 0x0
	s_bfe_i64 s[6:7], s[20:21], 0x200000
	s_mov_b32 s36, s34
	s_mul_u64 s[4:5], s[6:7], s[4:5]
	s_and_b32 s6, s29, s8
	s_lshl_b64 s[4:5], s[4:5], 1
	v_cndmask_b32_e64 v2, 0, 1, s6
	s_add_nc_u64 s[4:5], s[26:27], s[4:5]
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_mov_b32 v1, s38 :: v_dual_mov_b32 v4, s4
	s_and_b32 s5, s5, 0x1ffffff
	v_readfirstlane_b32 s44, v2
	s_bitset1_b32 s5, 31
	v_readfirstlane_b32 s45, v1
	v_mov_b32_e32 v3, s5
	v_readfirstlane_b32 s46, v4
	s_mov_b32 s10, 0
	s_lshr_b32 s11, s37, 16
	s_lshr_b64 s[6:7], s[36:37], 16
	v_readfirstlane_b32 s47, v3
	s_movk_i32 s8, 0xe0
	s_or_b32 s4, s1, 0x7510000
	s_lshl_b32 s5, s34, 16
	s_or_b32 s7, s11, 0x600000
	s_mov_b32 s11, s10
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[44:47], s[4:11]
.LBB0_11:
	s_or_b32 exec_lo, exec_lo, s19
.LBB0_12:
	s_or_saveexec_b32 s18, s18
	s_min_i32 s36, s3, s16
	s_lshl_b32 s17, 0x1111, s17
	s_xor_b32 exec_lo, exec_lo, s18
	s_cbranch_execz .LBB0_14
; %bb.13:
	s_cmp_gt_i32 s34, 0
	s_mov_b32 s10, 0
	s_cselect_b32 s3, -1, 0
	s_lshl_b32 s4, s36, 8
	s_wait_kmcnt 0x0
	s_bfe_i64 s[6:7], s[24:25], 0x200000
	s_ashr_i32 s5, s4, 31
	s_and_b32 s3, s42, s3
	s_mul_u64 s[4:5], s[6:7], s[4:5]
	v_cndmask_b32_e64 v2, 0, 1, s3
	s_lshl_b64 s[6:7], s[4:5], 1
	s_lshr_b32 s3, s35, 16
	s_add_nc_u64 s[6:7], s[14:15], s[6:7]
	s_or_b32 s4, s17, 0x7510000
	s_and_b32 s7, s7, 0x1ffffff
	v_readfirstlane_b32 s44, v2
	s_bitset1_b32 s7, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v4, s6 :: v_dual_mov_b32 v3, s7
	s_lshr_b64 s[6:7], s[34:35], 16
	s_lshl_b32 s5, s34, 16
	s_or_b32 s7, s3, 0x600000
	v_readfirstlane_b32 s46, v4
	v_readfirstlane_b32 s47, v3
	s_movk_i32 s8, 0x100
	s_mov_b32 s11, s10
	s_mov_b32 s45, s10
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[44:47], s[4:11]
.LBB0_14:
	s_or_b32 exec_lo, exec_lo, s18
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s3, exec_lo
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	v_cmpx_gt_u32_e32 32, v0
	s_cbranch_execz .LBB0_16
; %bb.15:
	s_barrier_signal -3
.LBB0_16:
	s_or_b32 exec_lo, exec_lo, s3
	v_dual_mov_b32 v9, 0 :: v_dual_bitop2_b32 v1, 32, v0 bitop3:0x40
	s_and_b32 s3, s42, s29
	v_and_b32_e32 v231, 0xc0, v0
	v_cndmask_b32_e64 v227, 0, 1, s3
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_ne_u32_e32 vcc_lo, 0, v1
	v_dual_mov_b32 v8, v9 :: v_dual_mov_b32 v7, v9
	v_dual_mov_b32 v6, v9 :: v_dual_mov_b32 v5, v9
	v_cndmask_b32_e64 v233, 0, 0x70, vcc_lo
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
	v_dual_mov_b32 v34, v9 :: v_dual_mov_b32 v57, v9
	v_dual_mov_b32 v56, v9 :: v_dual_mov_b32 v55, v9
	v_dual_mov_b32 v54, v9 :: v_dual_mov_b32 v53, v9
	v_dual_mov_b32 v52, v9 :: v_dual_mov_b32 v51, v9
	v_dual_mov_b32 v50, v9 :: v_dual_mov_b32 v49, v9
	v_dual_mov_b32 v48, v9 :: v_dual_mov_b32 v47, v9
	v_dual_mov_b32 v46, v9 :: v_dual_mov_b32 v45, v9
	v_dual_mov_b32 v44, v9 :: v_dual_mov_b32 v43, v9
	v_dual_mov_b32 v42, v9 :: v_dual_mov_b32 v65, v9
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
	v_dual_mov_b32 v90, v9 :: v_dual_mov_b32 v121, v9
	v_dual_mov_b32 v120, v9 :: v_dual_mov_b32 v119, v9
	v_dual_mov_b32 v118, v9 :: v_dual_mov_b32 v117, v9
	v_dual_mov_b32 v116, v9 :: v_dual_mov_b32 v115, v9
	v_dual_mov_b32 v114, v9 :: v_dual_mov_b32 v113, v9
	v_dual_mov_b32 v112, v9 :: v_dual_mov_b32 v111, v9
	v_dual_mov_b32 v110, v9 :: v_dual_mov_b32 v109, v9
	v_dual_mov_b32 v108, v9 :: v_dual_mov_b32 v107, v9
	v_dual_mov_b32 v106, v9 :: v_dual_mov_b32 v129, v9
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
	v_dual_mov_b32 v154, v9 :: v_dual_mov_b32 v177, v9
	v_dual_mov_b32 v176, v9 :: v_dual_mov_b32 v175, v9
	v_dual_mov_b32 v174, v9 :: v_dual_mov_b32 v173, v9
	v_dual_mov_b32 v172, v9 :: v_dual_mov_b32 v171, v9
	v_dual_mov_b32 v170, v9 :: v_dual_mov_b32 v169, v9
	v_dual_mov_b32 v168, v9 :: v_dual_mov_b32 v167, v9
	v_dual_mov_b32 v166, v9 :: v_dual_mov_b32 v165, v9
	v_dual_mov_b32 v164, v9 :: v_dual_mov_b32 v163, v9
	v_dual_mov_b32 v162, v9 :: v_dual_mov_b32 v185, v9
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
	v_dual_mov_b32 v218, v9 :: v_dual_mov_b32 v105, v9
	v_dual_mov_b32 v104, v9 :: v_dual_mov_b32 v103, v9
	v_dual_mov_b32 v102, v9 :: v_dual_mov_b32 v101, v9
	v_dual_mov_b32 v100, v9 :: v_dual_mov_b32 v99, v9
	v_mov_b32_e32 v98, v9
	s_and_not1_b32 vcc_lo, exec_lo, s13
	s_mov_b32 s11, 0
	s_barrier_wait -1
	s_barrier_wait -3
	s_cbranch_vccnz .LBB0_41
; %bb.17:
	s_mov_b64 s[6:7], src_shared_base
	s_add_co_i32 s4, s38, 0xb280
	s_mov_b32 s5, s7
	v_mul_u32_u24_e32 v3, 0x60, v231
	s_and_b64 s[4:5], s[4:5], 15
	v_and_b32_e32 v7, 16, v0
	s_sub_co_i32 s6, 16, s4
	s_mul_i32 s18, s0, 0xe0
	s_lshr_b32 s6, s6, 2
	s_cmp_lg_u64 s[4:5], 0
	v_or_b32_e32 v9, v3, v7
	s_cselect_b32 s5, s6, 0
	v_and_b32_e32 v5, 15, v0
	s_lshl2_add_u32 s5, s5, s38
	s_movk_i32 s0, 0x600
	s_add_co_i32 s6, s5, 0x17e80
	s_add_co_i32 s45, s5, 0xb280
	s_and_b32 s10, s6, 15
	v_mad_u32_u24 v11, 0x60, v5, s0
	s_sub_co_i32 s8, 16, s10
	s_mul_hi_i32 s4, s12, 0x2aaaaaab
	s_lshr_b32 s5, s8, 2
	s_cmp_lg_u64 s[10:11], 0
	v_add_nc_u32_e32 v8, v9, v11
	s_cselect_b32 s5, s5, 0
	s_lshr_b32 s8, s4, 31
	s_ashr_i32 s47, s4, 4
	s_lshl_b32 s10, s5, 2
	s_add_co_i32 s47, s47, s8
	v_mul_u32_u24_e32 v2, 0x60, v5
	s_cmp_lt_i32 s37, 0xe0
	s_movk_i32 s0, 0xc00
	s_cselect_b32 s48, -1, 0
	s_or_b32 s12, s1, 0x7510000
	s_movk_i32 s1, 0x1200
	v_mul_u32_u24_e32 v1, 0x60, v233
	v_mad_u32_u24 v15, 0x60, v5, s1
	v_or_b32_e32 v4, v9, v2
	v_mad_u32_u24 v13, 0x60, v5, s0
	v_or_b32_e32 v23, 0x1800, v2
	s_ashr_i32 s19, s18, 31
	s_lshl_b32 s40, s36, 8
	v_lshrrev_b32_e32 v6, 4, v4
	v_mad_u32_u24 v49, 0x60, v5, v9
	s_wait_kmcnt 0x0
	s_bfe_i64 s[20:21], s[20:21], 0x200000
	s_ashr_i32 s41, s40, 31
	v_mov_b32_e32 v229, 0
	v_and_b32_e32 v6, 0x7f8, v6
	v_add_nc_u32_e32 v40, 0x600, v49
	v_add_nc_u32_e32 v42, 0x1200, v49
	s_mul_u64 s[18:19], s[20:21], s[18:19]
	s_bfe_i64 s[20:21], s[24:25], 0x200000
	v_dual_add_nc_u32 v226, v6, v4 :: v_dual_add_nc_u32 v6, v9, v13
	v_dual_lshrrev_b32 v4, 4, v8 :: v_dual_bitop2_b32 v10, v1, v7 bitop3:0x54
	v_add_nc_u32_e32 v8, v9, v15
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v41, v229 :: v_dual_lshrrev_b32 v6, 4, v6
	v_mad_u32_u24 v31, 0x60, v5, v10
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_and_b32_e32 v17, 0xff8, v4
	v_dual_lshrrev_b32 v8, 4, v8 :: v_dual_add_nc_u32 v12, v10, v11
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_add_nc_u32 v14, v10, v13 :: v_dual_lshrrev_b32 v4, 4, v31
	v_and_b32_e32 v19, 0xff8, v6
	v_and_b32_e32 v21, 0xff8, v8
	s_delay_alu instid0(VALU_DEP_4)
	v_lshrrev_b32_e32 v6, 4, v12
	v_add_nc_u32_e32 v2, 0x1e00, v31
	v_and_b32_e32 v4, 0x7f8, v4
	v_lshrrev_b32_e32 v8, 4, v14
	v_add_nc_u32_e32 v44, 0x600, v31
	v_and_b32_e32 v25, 0x7f8, v6
	v_dual_add_nc_u32 v234, v19, v49 :: v_dual_add_nc_u32 v236, v21, v49
	v_add_nc_u32_e32 v230, v4, v31
	v_add_nc_u32_e32 v4, v10, v15
	v_and_b32_e32 v27, 0x7f8, v8
	v_or_b32_e32 v6, 32, v7
	v_add_nc_u32_e32 v238, v25, v31
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_mov_b32 v21, v229 :: v_dual_lshrrev_b32 v8, 4, v4
	v_add_nc_u32_e32 v4, 0x2400, v31
	v_add_nc_u32_e32 v10, v10, v23
	v_or_b32_e32 v12, v6, v3
	v_add_nc_u32_e32 v240, v27, v31
	v_and_b32_e32 v29, 0x7f8, v8
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_lshrrev_b32 v14, 4, v4 :: v_dual_lshrrev_b32 v8, 4, v10
	v_lshrrev_b32_e32 v10, 4, v2
	v_mad_u32_u24 v16, 0x60, v5, v12
	v_add_nc_u32_e32 v18, v12, v11
	v_and_b32_e32 v37, 0x7f8, v14
	v_or_b32_e32 v14, v1, v6
	v_and_b32_e32 v33, 0x7f8, v8
	v_and_b32_e32 v35, 0x7f8, v10
	v_dual_lshrrev_b32 v8, 4, v16 :: v_dual_lshrrev_b32 v10, 4, v18
	v_dual_add_nc_u32 v6, v12, v13 :: v_dual_add_nc_u32 v12, v12, v15
	v_mad_u32_u24 v16, 0x60, v5, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_and_b32_e32 v39, 0xff8, v8
	v_and_b32_e32 v228, 0xff8, v10
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_dual_lshrrev_b32 v6, 4, v6 :: v_dual_lshrrev_b32 v8, 4, v12
	v_dual_add_nc_u32 v10, v14, v11 :: v_dual_add_nc_u32 v12, v14, v13
	v_dual_lshrrev_b32 v18, 4, v16 :: v_dual_bitop2_b32 v7, 64, v7 bitop3:0x54
	v_dual_mov_b32 v9, v229 :: v_dual_add_nc_u32 v20, 0x1e00, v16
	v_dual_lshrrev_b32 v10, 4, v10 :: v_dual_lshrrev_b32 v12, 4, v12
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_and_b32_e32 v46, 0x7f8, v18
	v_add_nc_u32_e32 v18, v14, v15
	v_add_nc_u32_e32 v14, v14, v23
	v_or_b32_e32 v3, v3, v7
	v_dual_mov_b32 v43, v229 :: v_dual_add_nc_u32 v16, 0x2400, v16
	v_dual_lshrrev_b32 v22, 4, v14 :: v_dual_bitop2_b32 v1, v1, v7 bitop3:0x54
	v_lshrrev_b32_e32 v18, 4, v18
	v_and_b32_e32 v6, 0xff8, v6
	v_and_b32_e32 v8, 0xff8, v8
	s_delay_alu instid0(VALU_DEP_4)
	v_mad_u32_u24 v30, 0x60, v5, v1
	v_and_b32_e32 v10, 0x7f8, v10
	v_and_b32_e32 v14, 0x7f8, v18
	v_lshrrev_b32_e32 v18, 4, v20
	v_mad_u32_u24 v24, 0x60, v5, v3
	v_lshrrev_b32_e32 v20, 4, v16
	v_and_b32_e32 v16, 0x7f8, v22
	v_lshrrev_b32_e32 v28, 4, v30
	v_and_b32_e32 v12, 0x7f8, v12
	v_dual_lshrrev_b32 v22, 4, v24 :: v_dual_add_nc_u32 v24, v3, v11
	v_add_nc_u32_e32 v11, v1, v11
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_and_b32_e32 v48, 0x7f8, v28
	v_mov_b32_e32 v45, v229
	v_and_b32_e32 v47, 0xff8, v22
	v_dual_lshrrev_b32 v7, 4, v24 :: v_dual_add_nc_u32 v24, v3, v13
	v_dual_add_nc_u32 v3, v3, v15 :: v_dual_add_nc_u32 v13, v1, v13
	v_lshrrev_b32_e32 v11, 4, v11
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_and_b32_e32 v22, 0xff8, v7
	v_dual_lshrrev_b32 v7, 4, v24 :: v_dual_lshrrev_b32 v3, 4, v3
	v_and_b32_e32 v18, 0xff8, v18
	v_and_b32_e32 v20, 0xff8, v20
	v_add_nc_u32_e32 v232, v17, v49
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v24, 0xff8, v7
	v_and_b32_e32 v26, 0xff8, v3
	v_lshrrev_b32_e32 v3, 4, v13
	v_and_b32_e32 v28, 0x7f8, v11
	v_add_nc_u32_e32 v7, v1, v15
	v_add_nc_u32_e32 v11, 0x1e00, v30
	v_add_nc_u32_e32 v13, 0x2400, v30
	v_add_nc_u32_e32 v1, v1, v23
	v_and_b32_e32 v30, 0x7f8, v3
	v_dual_add_nc_u32 v242, v29, v31 :: v_dual_add_nc_u32 v246, v35, v31
	v_dual_add_nc_u32 v244, v33, v31 :: v_dual_add_nc_u32 v252, v39, v49
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_lshrrev_b32 v1, 4, v1 :: v_dual_lshrrev_b32 v3, 4, v7
	v_dual_lshrrev_b32 v7, 4, v11 :: v_dual_lshrrev_b32 v11, 4, v13
	v_sub_nc_u32_e32 v13, 0x2c9f, v0
	v_and_b32_e32 v34, 0x7f8, v1
	v_add_nc_u32_e32 v248, v37, v31
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_dual_add_nc_u32 v2 /*v258*/, v46, v31 :: v_dual_add_nc_u32 v16 /*v272*/, v47, v49
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshrrev_b32_e32 v1, 8, v13
	v_and_b32_e32 v32, 0x7f8, v3
	v_and_b32_e32 v38, 0xff8, v11
	v_mov_b32_e32 v11, v229
	v_add_nc_u64_e32 v[250:251], v[228:229], v[40:41]
	v_add_nc_u32_e32 v3, 1, v1
	v_and_b32_e32 v36, 0xff8, v7
	v_dual_mov_b32 v7, v229 :: v_dual_add_nc_u32 v228, 0xc00, v49
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mov_b32 v13, v229 :: v_dual_bitop2_b32 v237, 62, v3 bitop3:0x40
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_add_nc_u64_e32 v[0:1] /*v[256:257]*/, v[8:9], v[42:43]
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_mov_b32_e32 v15, v229
	v_add_nc_u64_e32 v[254:255], v[6:7], v[228:229]
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_add_nc_u64_e32 v[4:5] /*v[260:261]*/, v[10:11], v[44:45]
	v_cmp_ne_u32_e64 s1, v3, v237
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_dual_mov_b32 v17, v229 :: v_dual_add_nc_u32 v6, 0xc00, v31
	v_dual_mov_b32 v19, v229 :: v_dual_add_nc_u32 v8, 0x1200, v31
	v_dual_mov_b32 v3, v229 :: v_dual_add_nc_u32 v10, 0x1800, v31
	v_dual_mov_b32 v5, v229 :: v_dual_mov_b32 v23, v229
	v_dual_mov_b32 v25, v229 :: v_dual_mov_b32 v27, v229
	v_mov_b32_e32 v29, v229
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_dual_mov_b32 v39 /*v295*/, v229 :: v_dual_add_nc_u32 v24 /*v280*/, v48, v31
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_dual_mov_b32 v31, v229 :: v_dual_mov_b32 v33, v229
	v_dual_mov_b32 v35, v229 :: v_dual_mov_b32 v37, v229
	v_mov_b32_e32 v39, v229
	v_lshl_or_b32 v50, v237, 8, v0
	s_lshl_b64 s[18:19], s[18:19], 1
	s_mul_u64 s[24:25], s[20:21], s[40:41]
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_add_nc_u64_e32 v[6:7] /*v[262:263]*/, v[12:13], v[6:7]
	v_add_nc_u64_e32 v[8:9] /*v[264:265]*/, v[14:15], v[8:9]
	v_add_nc_u64_e32 v[10:11] /*v[266:267]*/, v[16:17], v[10:11]
	v_add_nc_u64_e32 v[12:13] /*v[268:269]*/, v[18:19], v[2:3]
	v_add_nc_u64_e32 v[14:15] /*v[270:271]*/, v[20:21], v[4:5]
	v_add_nc_u64_e32 v[18:19] /*v[274:275]*/, v[22:23], v[40:41]
	v_add_nc_u64_e32 v[20:21] /*v[276:277]*/, v[24:25], v[228:229]
	v_add_nc_u64_e32 v[22:23] /*v[278:279]*/, v[26:27], v[42:43]
	v_add_nc_u64_e32 v[26:27] /*v[282:283]*/, v[28:29], v[44:45]
	v_add_nc_u64_e32 v[28:29] /*v[284:285]*/, v[30:31], v[6:7]
	v_add_nc_u64_e32 v[30:31] /*v[286:287]*/, v[32:33], v[8:9]
	v_add_nc_u64_e32 v[32:33] /*v[288:289]*/, v[34:35], v[10:11]
	v_add_nc_u64_e32 v[34:35] /*v[290:291]*/, v[36:37], v[2:3]
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_mov_b32_e32 v8, v229
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_add_nc_u64_e32 v[36:37] /*v[292:293]*/, v[38:39], v[4:5]
	v_cmp_eq_u32_e64 s0, 0, v235
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_or_b32_e32 v1, 0x100, v0
	v_or_b32_e32 v239, 0x3100, v0
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_lshl_or_b32 v38 /*v294*/, v0, 2, 0xc800
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_dual_mov_b32 v10, v229 :: v_dual_add_nc_u32 v241, 0xffffff00, v50
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_dual_mov_b32 v41 /*v297*/, v229 :: v_dual_lshlrev_b32 v40 /*v296*/, 2, v50
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_dual_mov_b32 v2, v229 :: v_dual_mov_b32 v4, v229
	v_dual_mov_b32 v6, v229 :: v_dual_mov_b32 v12, v229
	v_dual_mov_b32 v14, v229 :: v_dual_mov_b32 v16, v229
	v_dual_mov_b32 v18, v229 :: v_dual_mov_b32 v20, v229
	v_dual_mov_b32 v22, v229 :: v_dual_mov_b32 v24, v229
	v_dual_mov_b32 v26, v229 :: v_dual_mov_b32 v28, v229
	v_dual_mov_b32 v30, v229 :: v_dual_mov_b32 v32, v229
	v_dual_mov_b32 v34, v229 :: v_dual_mov_b32 v36, v229
	v_dual_mov_b32 v38, v229 :: v_dual_mov_b32 v40, v229
	v_dual_mov_b32 v50, v229 :: v_dual_mov_b32 v51, v229
	v_dual_mov_b32 v52, v229 :: v_dual_mov_b32 v53, v229
	v_dual_mov_b32 v54, v229 :: v_dual_mov_b32 v55, v229
	v_dual_mov_b32 v56, v229 :: v_dual_mov_b32 v57, v229
	v_dual_mov_b32 v42, v229 :: v_dual_mov_b32 v44, v229
	v_dual_mov_b32 v46, v229 :: v_dual_mov_b32 v47, v229
	v_dual_mov_b32 v48, v229 :: v_dual_mov_b32 v49, v229
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
	v_dual_mov_b32 v114, v229 :: v_dual_mov_b32 v115, v229
	v_dual_mov_b32 v116, v229 :: v_dual_mov_b32 v117, v229
	v_dual_mov_b32 v118, v229 :: v_dual_mov_b32 v119, v229
	v_dual_mov_b32 v120, v229 :: v_dual_mov_b32 v121, v229
	v_dual_mov_b32 v106, v229 :: v_dual_mov_b32 v107, v229
	v_dual_mov_b32 v108, v229 :: v_dual_mov_b32 v109, v229
	v_dual_mov_b32 v110, v229 :: v_dual_mov_b32 v111, v229
	v_dual_mov_b32 v112, v229 :: v_dual_mov_b32 v113, v229
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
	v_dual_mov_b32 v170, v229 :: v_dual_mov_b32 v171, v229
	v_dual_mov_b32 v172, v229 :: v_dual_mov_b32 v173, v229
	v_dual_mov_b32 v174, v229 :: v_dual_mov_b32 v175, v229
	v_dual_mov_b32 v176, v229 :: v_dual_mov_b32 v177, v229
	v_dual_mov_b32 v162, v229 :: v_dual_mov_b32 v163, v229
	v_dual_mov_b32 v164, v229 :: v_dual_mov_b32 v165, v229
	v_dual_mov_b32 v166, v229 :: v_dual_mov_b32 v167, v229
	v_dual_mov_b32 v168, v229 :: v_dual_mov_b32 v169, v229
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
	v_dual_mov_b32 v98, v229 :: v_dual_mov_b32 v99, v229
	v_dual_mov_b32 v100, v229 :: v_dual_mov_b32 v101, v229
	v_dual_mov_b32 v102, v229 :: v_dual_mov_b32 v103, v229
	v_dual_mov_b32 v104, v229 :: v_dual_mov_b32 v105, v229
	s_lshr_b32 s49, s37, 16
	s_lshr_b32 s50, s35, 16
	s_add_nc_u64 s[20:21], s[26:27], s[18:19]
	s_lshl_b64 s[18:19], s[24:25], 1
	s_mov_b32 s44, s39
	s_mov_b32 s46, s7
	s_movk_i32 s16, 0xe0
	s_or_b32 s49, s49, 0x600000
	s_or_b32 s4, s17, 0x7510000
	s_or_b32 s50, s50, 0x600000
	s_add_nc_u64 s[24:25], s[14:15], s[18:19]
	s_mov_b32 s40, -1
	s_add_nc_u64 s[26:27], s[6:7], s[10:11]
	s_movk_i32 s8, 0x100
	s_mov_b32 s41, s11
	s_branch .LBB0_19
.LBB0_18:                               ;   in Loop: Header=BB0_19 Depth=1
	s_or_b32 exec_lo, exec_lo, s5
	s_cmp_eq_u32 s41, s47
	s_barrier_wait -1
	s_barrier_wait -3
	s_cbranch_scc1 .LBB0_41
.LBB0_19:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_21 Depth 2
                                        ;     Child Loop BB0_24 Depth 2
                                        ;     Child Loop BB0_26 Depth 2
                                        ;     Child Loop BB0_29 Depth 2
	s_and_b32 s51, s41, 1
	s_add_co_i32 s41, s41, 1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_i32 s5, s41, 0xffffffa0
	s_add_co_i32 s6, s5, s30
	s_xor_b32 s5, s51, 1
	s_min_i32 s6, s6, 0x60
	s_cmp_lt_i32 s41, s47
	s_cselect_b32 s10, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_and_b32 s7, s10, exec_lo
	s_cselect_b32 s34, s6, 0
	s_cmp_lt_i32 s34, 0x60
	s_cselect_b32 s6, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b32 s6, s48, s6
	s_or_b32 s6, s43, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s6
	s_cbranch_vccnz .LBB0_31
; %bb.20:                               ;   in Loop: Header=BB0_19 Depth=1
	v_nop
	v_nop
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_mov_b64_e32 v[42:43] /*v[298:299]*/, v[0:1]
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_mov_b32_e32 v243, 50
	s_cmp_lg_u32 s5, 0
	s_mov_b32 s13, 0
	s_cselect_b32 s7, s46, s39
	s_cselect_b32 s6, s45, 0
.LBB0_21:                               ;   Parent Loop BB0_19 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_set_vgpr_msb 1                        ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_dual_mov_b32 v228, v42 /*v298*/ :: v_dual_add_nc_u32 v243, -2, v243
	s_set_vgpr_msb 0x145                    ;  msbs: dst=1 src0=1 src1=1 src2=0
	v_dual_mov_b32 v44 /*v300*/, v43 /*v299*/ :: v_dual_add_nc_u32 v43 /*v299*/, 0x200, v43 /*v299*/
	s_set_vgpr_msb 0x4544                   ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_dual_mov_b32 v45 /*v301*/, v229 :: v_dual_add_nc_u32 v42 /*v298*/, 0x200, v42 /*v298*/
	s_set_vgpr_msb 0x4440                   ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_cmp_eq_u32_e32 vcc_lo, 0, v243
	v_lshl_add_u64 v[46:47] /*v[302:303]*/, v[228:229], 2, s[6:7]
	s_set_vgpr_msb 0x4041                   ;  msbs: dst=1 src0=1 src1=0 src2=0
	v_lshl_add_u64 v[44:45] /*v[300:301]*/, v[44:45] /*v[300:301]*/, 2, s[6:7]
	s_clause 0x1
	flat_store_b32 v[46:47] /*v[302:303]*/, v229
	flat_store_b32 v[44:45] /*v[300:301]*/, v229
	s_or_b32 s13, vcc_lo, s13
	s_wait_xcnt 0x0
	s_and_not1_b32 exec_lo, exec_lo, s13
	s_set_vgpr_msb 0x4100                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_cbranch_execnz .LBB0_21
; %bb.22:                               ;   in Loop: Header=BB0_19 Depth=1
	s_or_b32 exec_lo, exec_lo, s13
	s_and_saveexec_b32 s13, s40
	s_cbranch_execz .LBB0_25
; %bb.23:                               ;   in Loop: Header=BB0_19 Depth=1
	s_set_vgpr_msb 0x44                     ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_add_nc_u64_e32 v[42:43] /*v[298:299]*/, s[6:7], v[38:39] /*v[294:295]*/
	s_set_vgpr_msb 0x4400                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_mov_b32_e32 v228, v239
	s_mov_b32 s6, 0
.LBB0_24:                               ;   Parent Loop BB0_19 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v228, 0x100, v228
	s_set_vgpr_msb 1                        ;  msbs: dst=0 src0=1 src1=0 src2=0
	flat_store_b32 v[42:43] /*v[298:299]*/, v229
	s_set_vgpr_msb 0x144                    ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_add_nc_u64_e32 v[42:43] /*v[298:299]*/, 0x400, v[42:43] /*v[298:299]*/
	s_set_vgpr_msb 0x4400                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_cmp_lt_u32_e32 vcc_lo, 0x31ff, v228
	s_or_b32 s6, vcc_lo, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s6
	s_cbranch_execnz .LBB0_24
.LBB0_25:                               ;   in Loop: Header=BB0_19 Depth=1
	s_or_b32 exec_lo, exec_lo, s13
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_mov_b64_e32 v[42:43] /*v[298:299]*/, v[0:1]
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_mov_b32_e32 v243, v237
	s_cmp_lg_u32 s5, 0
	s_mov_b32 s13, 0
	s_cselect_b32 s7, s27, s44
	s_cselect_b32 s6, s26, s38
.LBB0_26:                               ;   Parent Loop BB0_19 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_set_vgpr_msb 1                        ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_dual_mov_b32 v228, v42 /*v298*/ :: v_dual_add_nc_u32 v243, -2, v243
	s_set_vgpr_msb 0x145                    ;  msbs: dst=1 src0=1 src1=1 src2=0
	v_dual_mov_b32 v44 /*v300*/, v43 /*v299*/ :: v_dual_add_nc_u32 v43 /*v299*/, 0x200, v43 /*v299*/
	s_set_vgpr_msb 0x4544                   ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_dual_mov_b32 v45 /*v301*/, v229 :: v_dual_add_nc_u32 v42 /*v298*/, 0x200, v42 /*v298*/
	s_set_vgpr_msb 0x4440                   ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_cmp_eq_u32_e32 vcc_lo, 0, v243
	v_lshl_add_u64 v[46:47] /*v[302:303]*/, v[228:229], 2, s[6:7]
	s_set_vgpr_msb 0x4041                   ;  msbs: dst=1 src0=1 src1=0 src2=0
	v_lshl_add_u64 v[44:45] /*v[300:301]*/, v[44:45] /*v[300:301]*/, 2, s[6:7]
	s_clause 0x1
	flat_store_b32 v[46:47] /*v[302:303]*/, v229
	flat_store_b32 v[44:45] /*v[300:301]*/, v229
	s_or_b32 s13, vcc_lo, s13
	s_wait_xcnt 0x0
	s_and_not1_b32 exec_lo, exec_lo, s13
	s_set_vgpr_msb 0x4100                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_cbranch_execnz .LBB0_26
; %bb.27:                               ;   in Loop: Header=BB0_19 Depth=1
	s_or_b32 exec_lo, exec_lo, s13
	s_and_saveexec_b32 s13, s1
	s_cbranch_execz .LBB0_30
; %bb.28:                               ;   in Loop: Header=BB0_19 Depth=1
	s_set_vgpr_msb 0x44                     ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_add_nc_u64_e32 v[42:43] /*v[298:299]*/, s[6:7], v[40:41] /*v[296:297]*/
	s_set_vgpr_msb 0x4400                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_mov_b32_e32 v228, v241
	s_mov_b32 s6, 0
.LBB0_29:                               ;   Parent Loop BB0_19 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v228, 0x100, v228
	s_set_vgpr_msb 1                        ;  msbs: dst=0 src0=1 src1=0 src2=0
	flat_store_b32 v[42:43] /*v[298:299]*/, v229
	s_set_vgpr_msb 0x144                    ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_add_nc_u64_e32 v[42:43] /*v[298:299]*/, 0x400, v[42:43] /*v[298:299]*/
	s_set_vgpr_msb 0x4400                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_cmp_lt_u32_e32 vcc_lo, 0x2b9f, v228
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
	s_cselect_b32 s6, s41, 0
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
	s_and_not1_b32 vcc_lo, exec_lo, s3
	s_cbranch_vccnz .LBB0_35
.LBB0_34:                               ;   in Loop: Header=BB0_19 Depth=1
	s_cmp_lg_u32 s51, 0
	s_cselect_b32 s6, s45, 0
	s_cselect_b32 s5, s26, s38
	v_lshl_add_u32 v228, v226, 1, s6
	v_lshl_add_u32 v243, v230, 1, s5
	v_lshl_add_u32 v245, v238, 1, s5
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[42:45] /*v[298:301]*/, v228
	ds_load_b128 v[46:49] /*v[302:305]*/, v228 offset:16
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v228, v232, 1, s6
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[50:53] /*v[306:309]*/, v243
	ds_load_b128 v[54:57] /*v[310:313]*/, v243 offset:16
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v243, v236, 1, s6
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[82:85] /*v[338:341]*/, v245 offset:3072
	ds_load_b128 v[58:61] /*v[314:317]*/, v228 offset:3072
	ds_load_b128 v[62:65] /*v[318:321]*/, v228 offset:3088
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v228, v234, 1, s6
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[74:77] /*v[330:333]*/, v243 offset:9216
	ds_load_b128 v[78:81] /*v[334:337]*/, v243 offset:9232
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v243, v250, 1, s6
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[66:69] /*v[322:325]*/, v228 offset:6144
	ds_load_b128 v[70:73] /*v[326:329]*/, v228 offset:6160
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v228, v248, 1, s5
	; sched_group_barrier mask(0x00000100) size(11) SyncID(0)
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x7
	v_wmma_f32_16x16x32_bf16 v[218:225], v[42:49] /*v[298:305]*/, v[50:57] /*v[306:313]*/, v[218:225]
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[170:177], v[58:65] /*v[314:321]*/, v[50:57] /*v[306:313]*/, v[170:177] matrix_a_reuse
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[114:121], v[66:73] /*v[322:329]*/, v[50:57] /*v[306:313]*/, v[114:121]
	v_wmma_f32_16x16x32_bf16 v[50:57], v[74:81] /*v[330:337]*/, v[50:57] /*v[306:313]*/, v[50:57] matrix_a_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[86:89] /*v[342:345]*/, v245 offset:3088
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v245, v254, 1, s6
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[58:65] /*v[314:321]*/, v[82:89] /*v[338:345]*/, v[154:161] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[90:97], v[66:73] /*v[322:329]*/, v[82:89] /*v[338:345]*/, v[90:97] matrix_a_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[50:53] /*v[306:309]*/, v228 offset:18432
	ds_load_b128 v[54:57] /*v[310:313]*/, v228 offset:18448
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v228, v246, 1, s5
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[162:169], v[42:49] /*v[298:305]*/, v[50:57] /*v[306:313]*/, v[162:169] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[106:113], v[58:65] /*v[314:321]*/, v[50:57] /*v[306:313]*/, v[106:113]
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[90:93] /*v[346:349]*/, v228 offset:15360
	ds_load_b128 v[94:97] /*v[350:353]*/, v228 offset:15376
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v228, v244, 1, s5
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[178:185], v[42:49] /*v[298:305]*/, v[90:97] /*v[346:353]*/, v[178:185] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[122:129], v[58:65] /*v[314:321]*/, v[90:97] /*v[346:353]*/, v[122:129] matrix_a_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[98:101] /*v[354:357]*/, v228 offset:12288
	ds_load_b128 v[102:105] /*v[358:361]*/, v228 offset:12304
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v228, v242, 1, s5
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[130:137], v[58:65] /*v[314:321]*/, v[98:105] /*v[354:361]*/, v[130:137] matrix_a_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[106:109] /*v[362:365]*/, v228 offset:9216
	ds_load_b128 v[110:113] /*v[366:369]*/, v228 offset:9232
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v228, v240, 1, s5
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[138:145], v[58:65] /*v[314:321]*/, v[106:113] /*v[362:369]*/, v[138:145] matrix_a_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[114:117] /*v[370:373]*/, v228 offset:6144
	ds_load_b128 v[118:121] /*v[374:377]*/, v228 offset:6160
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v228, v252, 1, s6
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[146:153], v[58:65] /*v[314:321]*/, v[114:121] /*v[370:377]*/, v[146:153] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(14) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(11) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[42:49], v[66:73] /*v[322:329]*/, v[50:57] /*v[306:313]*/, v[42:49] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[98:105], v[74:81] /*v[330:337]*/, v[50:57] /*v[306:313]*/, v[98:105]
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[50:53] /*v[306:309]*/, v243 offset:64
	ds_load_b128 v[54:57] /*v[310:313]*/, v243 offset:80
	s_set_vgpr_msb 0x4005                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_lshl_add_u32 v243, v2 /*v258*/, 1, s5
	v_wmma_f32_16x16x32_bf16 v[210:217], v[42:49] /*v[298:305]*/, v[82:89] /*v[338:345]*/, v[210:217] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[202:209], v[42:49] /*v[298:305]*/, v[114:121] /*v[370:377]*/, v[202:209] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[194:201], v[42:49] /*v[298:305]*/, v[106:113] /*v[362:369]*/, v[194:201] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[186:193], v[42:49] /*v[298:305]*/, v[98:105] /*v[354:361]*/, v[186:193] matrix_a_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[42:45] /*v[298:301]*/, v228 offset:64
	ds_load_b128 v[46:49] /*v[302:305]*/, v228 offset:80
	s_set_vgpr_msb 0x4005                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_lshl_add_u32 v228, v0 /*v256*/, 1, s6
	v_wmma_f32_16x16x32_bf16 v[58:65], v[66:73] /*v[322:329]*/, v[90:97] /*v[346:353]*/, v[58:65] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[2:9], v[74:81] /*v[330:337]*/, v[90:97] /*v[346:353]*/, v[2:9] matrix_a_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[90:93] /*v[346:349]*/, v228 offset:64
	ds_load_b128 v[94:97] /*v[350:353]*/, v228 offset:80
	s_set_vgpr_msb 0x4005                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_lshl_add_u32 v228, v4 /*v260*/, 1, s5
	v_wmma_f32_16x16x32_bf16 v[66:73], v[66:73] /*v[322:329]*/, v[98:105] /*v[354:361]*/, v[66:73] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[10:17], v[74:81] /*v[330:337]*/, v[98:105] /*v[354:361]*/, v[10:17] matrix_a_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[98:101] /*v[354:357]*/, v243 offset:64
	ds_load_b128 v[102:105] /*v[358:361]*/, v243 offset:80
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_lshl_add_u32 v243, v6 /*v262*/, 1, s5
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[122:125] /*v[378:381]*/, v243 offset:64
	s_set_vgpr_msb 0x4005                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[18:25], v[74:81] /*v[330:337]*/, v[106:113] /*v[362:369]*/, v[18:25] matrix_a_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[126:129] /*v[382:385]*/, v243 offset:80
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_lshl_add_u32 v243, v12 /*v268*/, 1, s5
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[138:141] /*v[394:397]*/, v243 offset:64
	s_set_vgpr_msb 0x4005                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[26:33], v[74:81] /*v[330:337]*/, v[114:121] /*v[370:377]*/, v[26:33] matrix_a_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[142:145] /*v[398:401]*/, v243 offset:80
	s_set_vgpr_msb 0x4005                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[34:41], v[74:81] /*v[330:337]*/, v[82:89] /*v[338:345]*/, v[34:41] matrix_a_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[74:77] /*v[330:333]*/, v245 offset:64
	ds_load_b128 v[78:81] /*v[334:337]*/, v245 offset:80
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_lshl_add_u32 v245, v8 /*v264*/, 1, s5
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[130:133] /*v[386:389]*/, v245 offset:64
	s_set_vgpr_msb 0x4005                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[74:81], v[66:73] /*v[322:329]*/, v[106:113] /*v[362:369]*/, v[74:81] matrix_a_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[106:109] /*v[362:365]*/, v228 offset:64
	ds_load_b128 v[110:113] /*v[366:369]*/, v228 offset:80
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_lshl_add_u32 v228, v10 /*v266*/, 1, s5
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[134:137] /*v[390:393]*/, v245 offset:80
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_lshl_add_u32 v245, v14 /*v270*/, 1, s5
	; sched_group_barrier mask(0x00000008) size(14) SyncID(0)
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[58:61] /*v[314:317]*/, v228 offset:64
	ds_load_b128 v[62:65] /*v[318:321]*/, v228 offset:80
	ds_load_b128 v[146:149] /*v[402:405]*/, v245 offset:64
	ds_load_b128 v[150:153] /*v[406:409]*/, v245 offset:80
	s_set_vgpr_msb 0x4005                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[82:89], v[66:73] /*v[322:329]*/, v[114:121] /*v[370:377]*/, v[82:89] matrix_a_reuse
	; sched_barrier mask(0x00000000)
	v_lshl_add_u32 v228, v16 /*v272*/, 1, s6
	v_lshl_add_u32 v243, v18 /*v274*/, 1, s6
	v_lshl_add_u32 v245, v20 /*v276*/, 1, s6
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[66:69] /*v[322:325]*/, v228 offset:128
	ds_load_b128 v[70:73] /*v[326:329]*/, v228 offset:144
	ds_load_b128 v[82:85] /*v[338:341]*/, v243 offset:128
	ds_load_b128 v[86:89] /*v[342:345]*/, v243 offset:144
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_lshl_add_u32 v228, v22 /*v278*/, 1, s6
	v_lshl_add_u32 v243, v24 /*v280*/, 1, s5
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[114:117] /*v[370:373]*/, v245 offset:128
	ds_load_b128 v[118:121] /*v[374:377]*/, v245 offset:144
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_lshl_add_u32 v245, v26 /*v282*/, 1, s5
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[154:157] /*v[410:413]*/, v228 offset:128
	ds_load_b128 v[158:161] /*v[414:417]*/, v228 offset:144
	ds_load_b128 v[162:165] /*v[418:421]*/, v243 offset:128
	ds_load_b128 v[166:169] /*v[422:425]*/, v243 offset:144
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_lshl_add_u32 v228, v28 /*v284*/, 1, s5
	v_lshl_add_u32 v243, v30 /*v286*/, 1, s5
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[170:173] /*v[426:429]*/, v245 offset:128
	s_set_vgpr_msb 0x4005                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x19
	v_wmma_f32_16x16x32_bf16 v[218:225], v[42:49] /*v[298:305]*/, v[98:105] /*v[354:361]*/, v[218:225]
	; sched_group_barrier mask(0x00000100) size(11) SyncID(0)
	s_wait_dscnt 0x10
	v_wmma_f32_16x16x32_bf16 v[210:217], v[42:49] /*v[298:305]*/, v[106:113] /*v[362:369]*/, v[210:217] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[202:209], v[42:49] /*v[298:305]*/, v[122:129] /*v[378:385]*/, v[202:209] matrix_a_reuse
	s_wait_dscnt 0xf
	v_wmma_f32_16x16x32_bf16 v[194:201], v[42:49] /*v[298:305]*/, v[130:137] /*v[386:393]*/, v[194:201] matrix_a_reuse
	s_wait_dscnt 0xd
	v_wmma_f32_16x16x32_bf16 v[186:193], v[42:49] /*v[298:305]*/, v[58:65] /*v[314:321]*/, v[186:193] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[178:185], v[42:49] /*v[298:305]*/, v[138:145] /*v[394:401]*/, v[178:185] matrix_a_reuse
	s_wait_dscnt 0xb
	v_wmma_f32_16x16x32_bf16 v[162:169], v[42:49] /*v[298:305]*/, v[146:153] /*v[402:409]*/, v[162:169] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[106:113], v[50:57] /*v[306:313]*/, v[146:153] /*v[402:409]*/, v[106:113]
	v_wmma_f32_16x16x32_bf16 v[122:129], v[50:57] /*v[306:313]*/, v[138:145] /*v[394:401]*/, v[122:129] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[130:137], v[50:57] /*v[306:313]*/, v[58:65] /*v[314:321]*/, v[130:137] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[138:145], v[50:57] /*v[306:313]*/, v[130:137] /*v[386:393]*/, v[138:145] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[146:153], v[50:57] /*v[306:313]*/, v[122:129] /*v[378:385]*/, v[146:153] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[154:161], v[50:57] /*v[306:313]*/, v[106:113] /*v[362:369]*/, v[154:161] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[170:177], v[50:57] /*v[306:313]*/, v[98:105] /*v[354:361]*/, v[170:177] matrix_a_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[174:177] /*v[430:433]*/, v245 offset:144
	ds_load_b128 v[42:45] /*v[298:301]*/, v228 offset:128
	ds_load_b128 v[46:49] /*v[302:305]*/, v228 offset:144
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_lshl_add_u32 v228, v32 /*v288*/, 1, s5
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[50:53] /*v[306:309]*/, v243 offset:128
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_lshl_add_u32 v245, v34 /*v290*/, 1, s5
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[54:57] /*v[310:313]*/, v243 offset:144
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_lshl_add_u32 v243, v36 /*v292*/, 1, s5
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[178:181] /*v[434:437]*/, v228 offset:128
	ds_load_b128 v[182:185] /*v[438:441]*/, v228 offset:144
	ds_load_b128 v[186:189] /*v[442:445]*/, v245 offset:128
	ds_load_b128 v[190:193] /*v[446:449]*/, v245 offset:144
	ds_load_b128 v[194:197] /*v[450:453]*/, v243 offset:128
	ds_load_b128 v[198:201] /*v[454:457]*/, v243 offset:144
	s_set_vgpr_msb 0x4005                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[114:121], v[74:81] /*v[330:337]*/, v[98:105] /*v[354:361]*/, v[114:121]
	; sched_group_barrier mask(0x00000008) size(14) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(11) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[90:97], v[74:81] /*v[330:337]*/, v[106:113] /*v[362:369]*/, v[90:97] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[82:89], v[74:81] /*v[330:337]*/, v[122:129] /*v[378:385]*/, v[82:89] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[74:81], v[74:81] /*v[330:337]*/, v[130:137] /*v[386:393]*/, v[74:81] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[66:73], v[74:81] /*v[330:337]*/, v[58:65] /*v[314:321]*/, v[66:73] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[58:65], v[74:81] /*v[330:337]*/, v[138:145] /*v[394:401]*/, v[58:65] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[42:49], v[74:81] /*v[330:337]*/, v[146:153] /*v[402:409]*/, v[42:49] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[98:105], v[90:97] /*v[346:353]*/, v[146:153] /*v[402:409]*/, v[98:105]
	v_wmma_f32_16x16x32_bf16 v[2:9], v[90:97] /*v[346:353]*/, v[138:145] /*v[394:401]*/, v[2:9] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[10:17], v[90:97] /*v[346:353]*/, v[58:65] /*v[314:321]*/, v[10:17] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[18:25], v[90:97] /*v[346:353]*/, v[130:137] /*v[386:393]*/, v[18:25] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[26:33], v[90:97] /*v[346:353]*/, v[122:129] /*v[378:385]*/, v[26:33] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[34:41], v[90:97] /*v[346:353]*/, v[106:113] /*v[362:369]*/, v[34:41] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[50:57], v[90:97] /*v[346:353]*/, v[98:105] /*v[354:361]*/, v[50:57] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(14) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[218:225], v[66:73] /*v[322:329]*/, v[162:169] /*v[418:425]*/, v[218:225]
	; sched_group_barrier mask(0x00000100) size(11) SyncID(0)
	s_wait_dscnt 0xa
	v_wmma_f32_16x16x32_bf16 v[210:217], v[66:73] /*v[322:329]*/, v[170:177] /*v[426:433]*/, v[210:217] matrix_a_reuse
	s_wait_dscnt 0x8
	v_wmma_f32_16x16x32_bf16 v[202:209], v[66:73] /*v[322:329]*/, v[42:49] /*v[298:305]*/, v[202:209] matrix_a_reuse
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[194:201], v[66:73] /*v[322:329]*/, v[50:57] /*v[306:313]*/, v[194:201] matrix_a_reuse
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[186:193], v[66:73] /*v[322:329]*/, v[178:185] /*v[434:441]*/, v[186:193] matrix_a_reuse
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[178:185], v[66:73] /*v[322:329]*/, v[186:193] /*v[442:449]*/, v[178:185] matrix_a_reuse
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[162:169], v[66:73] /*v[322:329]*/, v[194:201] /*v[450:457]*/, v[162:169] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[106:113], v[82:89] /*v[338:345]*/, v[194:201] /*v[450:457]*/, v[106:113]
	v_wmma_f32_16x16x32_bf16 v[122:129], v[82:89] /*v[338:345]*/, v[186:193] /*v[442:449]*/, v[122:129] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[130:137], v[82:89] /*v[338:345]*/, v[178:185] /*v[434:441]*/, v[130:137] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[138:145], v[82:89] /*v[338:345]*/, v[50:57] /*v[306:313]*/, v[138:145] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[146:153], v[82:89] /*v[338:345]*/, v[42:49] /*v[298:305]*/, v[146:153] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[154:161], v[82:89] /*v[338:345]*/, v[170:177] /*v[426:433]*/, v[154:161] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[170:177], v[82:89] /*v[338:345]*/, v[162:169] /*v[418:425]*/, v[170:177] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(14) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(11) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[114:121], v[114:121] /*v[370:377]*/, v[162:169] /*v[418:425]*/, v[114:121]
	v_wmma_f32_16x16x32_bf16 v[90:97], v[114:121] /*v[370:377]*/, v[170:177] /*v[426:433]*/, v[90:97] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[82:89], v[114:121] /*v[370:377]*/, v[42:49] /*v[298:305]*/, v[82:89] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[74:81], v[114:121] /*v[370:377]*/, v[50:57] /*v[306:313]*/, v[74:81] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[66:73], v[114:121] /*v[370:377]*/, v[178:185] /*v[434:441]*/, v[66:73] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[58:65], v[114:121] /*v[370:377]*/, v[186:193] /*v[442:449]*/, v[58:65] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[42:49], v[114:121] /*v[370:377]*/, v[194:201] /*v[450:457]*/, v[42:49] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[98:105], v[154:161] /*v[410:417]*/, v[194:201] /*v[450:457]*/, v[98:105]
	v_wmma_f32_16x16x32_bf16 v[2:9], v[154:161] /*v[410:417]*/, v[186:193] /*v[442:449]*/, v[2:9] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[10:17], v[154:161] /*v[410:417]*/, v[178:185] /*v[434:441]*/, v[10:17] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[18:25], v[154:161] /*v[410:417]*/, v[50:57] /*v[306:313]*/, v[18:25] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[26:33], v[154:161] /*v[410:417]*/, v[42:49] /*v[298:305]*/, v[26:33] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[34:41], v[154:161] /*v[410:417]*/, v[170:177] /*v[426:433]*/, v[34:41] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[50:57], v[154:161] /*v[410:417]*/, v[162:169] /*v[418:425]*/, v[50:57] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(14) SyncID(0)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_wait_dscnt 0
	;;#ASMEND
	s_set_vgpr_msb 0x500                    ;  msbs: dst=0 src0=0 src1=0 src2=0
.LBB0_35:                               ;   in Loop: Header=BB0_19 Depth=1
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_and_saveexec_b32 s5, s0
	s_cbranch_execz .LBB0_18
; %bb.36:                               ;   in Loop: Header=BB0_19 Depth=1
	s_barrier_signal -3
	s_branch .LBB0_18
.LBB0_37:                               ;   in Loop: Header=BB0_19 Depth=1
	s_mov_b32 s52, exec_lo
	v_cmpx_eq_u32_e32 1, v235
	s_cbranch_execz .LBB0_39
; %bb.38:                               ;   in Loop: Header=BB0_19 Depth=1
	s_cmp_lg_u32 s5, 0
	s_mul_i32 s10, s6, 0x60
	s_cselect_b32 s13, s26, s38
	s_cmp_gt_i32 s34, 0
	s_mov_b32 s36, s34
	s_cselect_b32 s17, -1, 0
	s_lshl_b64 s[14:15], s[10:11], 1
	s_mov_b32 s18, s11
	s_add_nc_u64 s[14:15], s[20:21], s[14:15]
	s_mov_b32 s19, s11
	s_and_b32 s10, s15, 0x1ffffff
	s_and_b32 s15, s29, s17
	s_bitset1_b32 s10, 31
	v_cndmask_b32_e64 v228, 0, 1, s15
	v_dual_mov_b32 v243, s13 :: v_dual_mov_b32 v245, s10
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_mov_b32_e32 v42 /*v298*/, s14
	s_lshr_b64 s[14:15], s[36:37], 16
	v_readfirstlane_b32 s56, v228
	v_readfirstlane_b32 s57, v243
	v_readfirstlane_b32 s59, v245
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_readfirstlane_b32 s58, v42 /*v298*/
	s_lshl_b32 s13, s34, 16
	s_mov_b32 s15, s49
	s_mov_b32 s17, s9
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[56:59], s[12:19]
	s_set_vgpr_msb 0x100                    ;  msbs: dst=0 src0=0 src1=0 src2=0
.LBB0_39:                               ;   in Loop: Header=BB0_19 Depth=1
	s_or_b32 exec_lo, exec_lo, s52
	s_and_not1_saveexec_b32 s13, s7
	s_cbranch_execz .LBB0_33
.LBB0_40:                               ;   in Loop: Header=BB0_19 Depth=1
	s_cmp_lg_u32 s5, 0
	s_mul_i32 s10, s6, 0x60
	s_cselect_b32 s5, s45, 0
	s_cmp_gt_i32 s34, 0
	s_cselect_b32 s14, -1, 0
	s_lshl_b64 s[6:7], s[10:11], 1
	s_and_b32 s10, s42, s14
	s_add_nc_u64 s[6:7], s[24:25], s[6:7]
	v_cndmask_b32_e64 v228, 0, 1, s10
	s_and_b32 s7, s7, 0x1ffffff
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_mov_b32_e32 v42 /*v298*/, s6
	s_bitset1_b32 s7, 31
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_dual_mov_b32 v243, s5 :: v_dual_mov_b32 v245, s7
	v_readfirstlane_b32 s52, v228
	s_set_vgpr_msb 1                        ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_readfirstlane_b32 s54, v42 /*v298*/
	s_lshr_b64 s[6:7], s[34:35], 16
	s_set_vgpr_msb 0x100                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_readfirstlane_b32 s53, v243
	v_readfirstlane_b32 s55, v245
	s_lshl_b32 s5, s34, 16
	s_mov_b32 s7, s50
	s_mov_b32 s10, s11
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	tensor_load_to_lds s[52:55], s[4:11]
	s_or_b32 exec_lo, exec_lo, s13
	s_and_not1_b32 vcc_lo, exec_lo, s3
	s_cbranch_vccz .LBB0_34
	s_branch .LBB0_35
.LBB0_41:
	s_wait_tensorcnt 0x0
	;;#ASMSTART
	s_wait_dscnt 0
	;;#ASMEND
	s_and_b32 vcc_lo, exec_lo, s3
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_vccz .LBB0_43
; %bb.42:
	v_lshrrev_b32_e32 v1, 1, v0
	v_and_or_b32 v226, v0, 15, v233
	v_cvt_pk_bf16_f32 v209, v208, v209
	v_cvt_pk_bf16_f32 v208, v206, v207
	v_cvt_pk_bf16_f32 v207, v204, v205
	v_and_or_b32 v1, v1, 8, v231
	v_cvt_pk_bf16_f32 v206, v202, v203
	v_cvt_pk_bf16_f32 v217, v216, v217
	v_cvt_pk_bf16_f32 v216, v214, v215
	v_cvt_pk_bf16_f32 v214, v210, v211
	v_lshl_or_b32 v1, v226, 8, v1
	v_cvt_pk_bf16_f32 v225, v224, v225
	v_cvt_pk_bf16_f32 v224, v222, v223
	v_cvt_pk_bf16_f32 v223, v220, v221
	v_cvt_pk_bf16_f32 v222, v218, v219
	v_add_nc_u32_e32 v204, 0x1000, v1
	v_lshrrev_b32_e32 v202, 3, v1
	v_add_nc_u32_e32 v203, 0x2000, v1
	v_lshlrev_b32_e32 v205, 1, v1
	v_add_nc_u32_e32 v210, 0x3000, v1
	v_lshrrev_b32_e32 v204, 3, v204
	v_and_b32_e32 v202, 0xff0, v202
	v_lshrrev_b32_e32 v203, 3, v203
	v_cvt_pk_bf16_f32 v215, v212, v213
	v_lshrrev_b32_e32 v210, 3, v210
	v_and_b32_e32 v204, 0x1ff0, v204
	v_add_nc_u32_e32 v202, v202, v205
	v_and_b32_e32 v203, 0x1ff0, v203
	v_cvt_pk_bf16_f32 v177, v176, v177
	v_cvt_pk_bf16_f32 v176, v174, v175
	v_add_nc_u32_e32 v204, v204, v205
	v_cvt_pk_bf16_f32 v174, v170, v171
	v_add_nc_u32_e32 v170, 0x5000, v1
	v_add_nc_u32_e32 v203, v203, v205
	ds_store_b128 v202, v[222:225]
	ds_store_b128 v204, v[214:217] offset:8192
	v_and_b32_e32 v202, 0x1ff0, v210
	v_cvt_pk_bf16_f32 v175, v172, v173
	v_cvt_pk_bf16_f32 v173, v200, v201
	v_cvt_pk_bf16_f32 v172, v198, v199
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_lshrrev_b32 v198, 3, v170 :: v_dual_add_nc_u32 v202, v202, v205
	v_cvt_pk_bf16_f32 v171, v196, v197
	v_cvt_pk_bf16_f32 v170, v194, v195
	v_add_nc_u32_e32 v211, 16, v1
	v_add_nc_u32_e32 v212, 0x4000, v1
	ds_store_b128 v203, v[206:209] offset:16384
	v_add_nc_u32_e32 v196, 0x6000, v1
	ds_store_b128 v202, v[170:173] offset:24576
	v_add_nc_u32_e32 v173, 0x2010, v1
	v_cvt_pk_bf16_f32 v161, v160, v161
	v_cvt_pk_bf16_f32 v160, v158, v159
	v_cvt_pk_bf16_f32 v159, v156, v157
	v_add_nc_u32_e32 v156, 0x3010, v1
	v_cvt_pk_bf16_f32 v193, v192, v193
	v_cvt_pk_bf16_f32 v192, v190, v191
	v_cvt_pk_bf16_f32 v191, v188, v189
	v_add_nc_u32_e32 v188, 0x1010, v1
	v_dual_lshrrev_b32 v213, 3, v211 :: v_dual_lshrrev_b32 v204, 3, v212
	v_cvt_pk_bf16_f32 v190, v186, v187
	v_lshrrev_b32_e32 v187, 3, v196
	v_cvt_pk_bf16_f32 v169, v168, v169
	v_cvt_pk_bf16_f32 v168, v166, v167
	v_cvt_pk_bf16_f32 v166, v162, v163
	v_dual_lshrrev_b32 v163, 3, v173 :: v_dual_lshrrev_b32 v156, 3, v156
	v_lshrrev_b32_e32 v170, 3, v188
	v_and_b32_e32 v213, 0x1ff0, v213
	v_and_b32_e32 v200, 0x1ff0, v204
	v_cvt_pk_bf16_f32 v185, v184, v185
	v_cvt_pk_bf16_f32 v184, v182, v183
	v_cvt_pk_bf16_f32 v182, v178, v179
	v_and_b32_e32 v178, 0x1ff0, v187
	v_lshlrev_b32_e32 v172, 1, v211
	v_cvt_pk_bf16_f32 v158, v154, v155
	v_and_b32_e32 v154, 0x1ff0, v163
	v_cvt_pk_bf16_f32 v153, v152, v153
	v_cvt_pk_bf16_f32 v152, v150, v151
	v_cvt_pk_bf16_f32 v150, v146, v147
	v_and_b32_e32 v146, 0x1ff0, v156
	v_cvt_pk_bf16_f32 v121, v120, v121
	v_cvt_pk_bf16_f32 v120, v118, v119
	v_cvt_pk_bf16_f32 v118, v114, v115
	v_add_nc_u32_e32 v114, 0x5010, v1
	v_and_b32_e32 v195, 0x1ff0, v198
	v_and_b32_e32 v170, 0x1ff0, v170
	v_dual_add_nc_u32 v210, v213, v205 :: v_dual_add_nc_u32 v194, v200, v205
	v_add_nc_u32_e32 v171, v178, v205
	v_cvt_pk_bf16_f32 v167, v164, v165
	v_add_nc_u32_e32 v154, v154, v172
	v_cvt_pk_bf16_f32 v151, v148, v149
	v_cvt_pk_bf16_f32 v119, v116, v117
	v_add_nc_u32_e32 v146, v146, v172
	v_cvt_pk_bf16_f32 v117, v144, v145
	v_cvt_pk_bf16_f32 v116, v142, v143
	v_lshrrev_b32_e32 v142, 3, v114
	v_cvt_pk_bf16_f32 v115, v140, v141
	v_cvt_pk_bf16_f32 v114, v138, v139
	v_add_nc_u32_e32 v186, v195, v205
	v_cvt_pk_bf16_f32 v183, v180, v181
	v_add_nc_u32_e32 v162, v170, v172
	ds_store_b128 v194, v[190:193] offset:32768
	ds_store_b128 v186, v[182:185] offset:40960
	ds_store_b128 v171, v[166:169] offset:49152
	v_add_nc_u32_e32 v155, 32, v1
	ds_store_b128 v210, v[174:177] offset:32
	ds_store_b128 v162, v[158:161] offset:8192
	v_add_nc_u32_e32 v158, 0x4010, v1
	ds_store_b128 v154, v[150:153] offset:16384
	v_add_nc_u32_e32 v140, 0x6010, v1
	ds_store_b128 v146, v[114:117] offset:24576
	v_add_nc_u32_e32 v117, 0x2020, v1
	v_cvt_pk_bf16_f32 v97, v96, v97
	v_cvt_pk_bf16_f32 v96, v94, v95
	v_cvt_pk_bf16_f32 v95, v92, v93
	v_add_nc_u32_e32 v92, 0x3020, v1
	v_cvt_pk_bf16_f32 v137, v136, v137
	v_cvt_pk_bf16_f32 v136, v134, v135
	v_cvt_pk_bf16_f32 v135, v132, v133
	v_add_nc_u32_e32 v132, 0x1020, v1
	v_dual_lshrrev_b32 v157, 3, v155 :: v_dual_lshrrev_b32 v147, 3, v158
	v_cvt_pk_bf16_f32 v134, v130, v131
	v_lshrrev_b32_e32 v131, 3, v140
	v_cvt_pk_bf16_f32 v113, v112, v113
	v_cvt_pk_bf16_f32 v112, v110, v111
	v_cvt_pk_bf16_f32 v110, v106, v107
	v_dual_lshrrev_b32 v107, 3, v117 :: v_dual_lshrrev_b32 v92, 3, v92
	v_lshrrev_b32_e32 v114, 3, v132
	v_and_b32_e32 v157, 0x1ff0, v157
	v_and_b32_e32 v144, 0x1ff0, v147
	v_cvt_pk_bf16_f32 v129, v128, v129
	v_cvt_pk_bf16_f32 v128, v126, v127
	v_cvt_pk_bf16_f32 v126, v122, v123
	v_and_b32_e32 v122, 0x1ff0, v131
	v_lshlrev_b32_e32 v116, 1, v155
	v_cvt_pk_bf16_f32 v94, v90, v91
	v_and_b32_e32 v90, 0x1ff0, v107
	v_cvt_pk_bf16_f32 v89, v88, v89
	v_cvt_pk_bf16_f32 v88, v86, v87
	v_cvt_pk_bf16_f32 v86, v82, v83
	v_and_b32_e32 v82, 0x1ff0, v92
	v_cvt_pk_bf16_f32 v57, v56, v57
	v_cvt_pk_bf16_f32 v56, v54, v55
	v_cvt_pk_bf16_f32 v54, v50, v51
	v_add_nc_u32_e32 v50, 0x5020, v1
	v_and_b32_e32 v139, 0x1ff0, v142
	v_and_b32_e32 v114, 0x1ff0, v114
	v_dual_add_nc_u32 v148, v157, v205 :: v_dual_add_nc_u32 v138, v144, v172
	v_add_nc_u32_e32 v115, v122, v172
	v_cvt_pk_bf16_f32 v111, v108, v109
	v_add_nc_u32_e32 v90, v90, v116
	v_cvt_pk_bf16_f32 v87, v84, v85
	v_cvt_pk_bf16_f32 v55, v52, v53
	v_add_nc_u32_e32 v82, v82, v116
	v_cvt_pk_bf16_f32 v53, v80, v81
	v_cvt_pk_bf16_f32 v52, v78, v79
	v_lshrrev_b32_e32 v78, 3, v50
	v_cvt_pk_bf16_f32 v51, v76, v77
	v_cvt_pk_bf16_f32 v50, v74, v75
	v_add_nc_u32_e32 v130, v139, v172
	v_cvt_pk_bf16_f32 v127, v124, v125
	v_add_nc_u32_e32 v106, v114, v116
	v_cvt_pk_bf16_f32 v41, v40, v41
	v_cvt_pk_bf16_f32 v40, v38, v39
	v_cvt_pk_bf16_f32 v39, v36, v37
	v_add_nc_u32_e32 v36, 0x3030, v1
	ds_store_b128 v138, v[134:137] offset:32768
	ds_store_b128 v130, v[126:129] offset:40960
	ds_store_b128 v115, v[110:113] offset:49152
	ds_store_b128 v148, v[118:121] offset:64
	ds_store_b128 v106, v[94:97] offset:8192
	v_add_nc_u32_e32 v94, 0x4020, v1
	ds_store_b128 v90, v[86:89] offset:16384
	v_cvt_pk_bf16_f32 v73, v72, v73
	v_cvt_pk_bf16_f32 v72, v70, v71
	v_add_nc_u32_e32 v76, 0x6020, v1
	v_cvt_pk_bf16_f32 v71, v68, v69
	v_add_nc_u32_e32 v68, 0x1030, v1
	ds_store_b128 v82, v[50:53] offset:24576
	v_add_nc_u32_e32 v53, 0x2030, v1
	v_add_nc_u32_e32 v91, 48, v1
	v_cvt_pk_bf16_f32 v38, v34, v35
	v_lshrrev_b32_e32 v35, 3, v36
	v_add_nc_u32_e32 v36, 0x4030, v1
	v_cvt_pk_bf16_f32 v33, v32, v33
	v_cvt_pk_bf16_f32 v32, v30, v31
	v_cvt_pk_bf16_f32 v31, v28, v29
	v_add_nc_u32_e32 v28, 0x5030, v1
	v_add_nc_u32_e32 v1, 0x6030, v1
	v_lshrrev_b32_e32 v83, 3, v94
	v_cvt_pk_bf16_f32 v70, v66, v67
	v_lshrrev_b32_e32 v67, 3, v76
	v_lshrrev_b32_e32 v50, 3, v68
	v_cvt_pk_bf16_f32 v49, v48, v49
	v_cvt_pk_bf16_f32 v48, v46, v47
	v_cvt_pk_bf16_f32 v46, v42, v43
	v_dual_lshrrev_b32 v43, 3, v53 :: v_dual_lshrrev_b32 v93, 3, v91
	v_cvt_pk_bf16_f32 v30, v26, v27
	v_lshrrev_b32_e32 v26, 3, v36
	v_cvt_pk_bf16_f32 v25, v24, v25
	v_cvt_pk_bf16_f32 v24, v22, v23
	v_cvt_pk_bf16_f32 v23, v20, v21
	v_dual_lshrrev_b32 v20, 3, v28 :: v_dual_lshrrev_b32 v1, 3, v1
	v_and_b32_e32 v80, 0x1ff0, v83
	v_cvt_pk_bf16_f32 v65, v64, v65
	v_cvt_pk_bf16_f32 v64, v62, v63
	v_cvt_pk_bf16_f32 v62, v58, v59
	v_and_b32_e32 v58, 0x1ff0, v67
	v_and_b32_e32 v50, 0x1ff0, v50
	v_lshlrev_b32_e32 v52, 1, v91
	v_and_b32_e32 v34, 0x1ff0, v43
	v_and_b32_e32 v93, 0x1ff0, v93
	v_and_b32_e32 v75, 0x1ff0, v78
	v_and_b32_e32 v35, 0x1ff0, v35
	v_and_b32_e32 v26, 0x1ff0, v26
	v_cvt_pk_bf16_f32 v22, v18, v19
	v_and_b32_e32 v19, 0x1ff0, v20
	v_and_b32_e32 v1, 0x1ff0, v1
	v_dual_add_nc_u32 v74, v80, v116 :: v_dual_add_nc_u32 v51, v58, v116
	v_cvt_pk_bf16_f32 v47, v44, v45
	v_add_nc_u32_e32 v42, v50, v52
	v_dual_add_nc_u32 v34, v34, v52 :: v_dual_add_nc_u32 v84, v93, v205
	v_add_nc_u32_e32 v66, v75, v116
	v_cvt_pk_bf16_f32 v63, v60, v61
	v_dual_add_nc_u32 v27, v35, v52 :: v_dual_add_nc_u32 v18, v26, v52
	v_cvt_pk_bf16_f32 v17, v16, v17
	v_cvt_pk_bf16_f32 v16, v14, v15
	v_cvt_pk_bf16_f32 v15, v12, v13
	v_cvt_pk_bf16_f32 v14, v10, v11
	v_add_nc_u32_e32 v10, v19, v52
	v_cvt_pk_bf16_f32 v9, v8, v9
	v_cvt_pk_bf16_f32 v8, v6, v7
	v_cvt_pk_bf16_f32 v7, v4, v5
	v_cvt_pk_bf16_f32 v6, v2, v3
	v_add_nc_u32_e32 v1, v1, v52
	v_cvt_pk_bf16_f32 v5, v104, v105
	v_cvt_pk_bf16_f32 v4, v102, v103
	v_cvt_pk_bf16_f32 v3, v100, v101
	v_cvt_pk_bf16_f32 v2, v98, v99
	ds_store_b128 v74, v[70:73] offset:32768
	ds_store_b128 v66, v[62:65] offset:40960
	ds_store_b128 v51, v[46:49] offset:49152
	ds_store_b128 v84, v[54:57] offset:96
	ds_store_b128 v42, v[38:41] offset:8192
	ds_store_b128 v34, v[30:33] offset:16384
	ds_store_b128 v27, v[22:25] offset:24576
	ds_store_b128 v18, v[14:17] offset:32768
	ds_store_b128 v10, v[6:9] offset:40960
	ds_store_b128 v1, v[2:5] offset:49152
.LBB0_43:
	v_cmp_ne_u32_e32 vcc_lo, 1, v227
	;;#ASMSTART
	s_wait_dscnt 0
	;;#ASMEND
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_vccnz .LBB0_54
; %bb.44:
	s_wait_kmcnt 0x0
	s_mul_i32 s14, s37, s35
	s_mov_b32 s0, exec_lo
	v_cmpx_gt_i32_e64 s14, v0
	s_cbranch_execz .LBB0_54
; %bb.45:
	s_ashr_i32 s3, s2, 31
	v_xad_u32 v2, v0, -1, s14
	s_lshl_b64 s[0:1], s[2:3], 1
	s_mul_i32 s15, s33, 0xe0
	s_ashr_i32 s29, s28, 31
	s_add_nc_u64 s[4:5], s[22:23], s[0:1]
	s_mov_b32 s0, 0
                                        ; implicit-def: $vgpr1
                                        ; implicit-def: $vgpr6
                                        ; implicit-def: $sgpr12_sgpr13
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_u32_e32 0x2ff, v2
	s_xor_b32 s3, exec_lo, s1
	s_cbranch_execnz .LBB0_48
; %bb.46:
	s_or_saveexec_b32 s1, s3
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
	s_abs_i32 s16, s31
	v_lshrrev_b32_e32 v1, 8, v2
	s_cvt_f32_u32 s0, s16
	v_or_b32_e32 v3, 0x300, v0
	s_sub_co_i32 s1, 0, s16
	v_mov_b32_e32 v7, 0
	v_rcp_iflag_f32_e32 v2, s0
	v_add_nc_u32_e32 v8, 1, v1
	v_or_b32_e32 v1, 0x100, v0
	s_mov_b32 s13, 0
	s_mov_b32 s17, s15
	s_mov_b32 s18, s15
	v_and_b32_e32 v9, 0x1fffffc, v8
	s_mov_b32 s19, s15
	v_readfirstlane_b32 s0, v2
	v_or_b32_e32 v2, 0x200, v0
	s_mov_b32 s6, s28
	v_mov_b32_e32 v10, v9
	s_mov_b32 s7, s29
	s_mul_f32 s0, s0, 0x4f7ffffe
	v_mov_b64_e32 v[4:5], v[2:3]
	v_mov_b64_e32 v[2:3], v[0:1]
	s_mov_b32 s8, s28
	s_cvt_u32_f32 s0, s0
	s_mov_b32 s9, s29
	s_mov_b32 s10, s28
	s_mov_b32 s11, s29
	s_mul_i32 s1, s1, s0
	s_mov_b32 s20, s31
	s_mul_hi_u32 s1, s0, s1
	s_mov_b32 s21, s31
	s_mov_b32 s22, s31
	s_ashr_i32 s23, s31, 31
	s_add_co_i32 s12, s0, s1
	s_mov_b32 s24, s13
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
	v_dual_add_nc_u32 v22, s18, v26 :: v_dual_sub_nc_u32 v12, v3, v12
	v_ashrrev_i32_e32 v21, 31, v20
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mul_lo_u32 v6, v1, s31
	v_dual_sub_nc_u32 v27, v19, v18 :: v_dual_add_nc_u32 v18, s15, v1
	v_sub_nc_u32_e32 v14, v4, v14
	v_ashrrev_i32_e32 v23, 31, v22
	v_lshl_add_u32 v11, v11, 8, v12
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_mul_lo_u32 v16, v27, s22
	v_dual_add_nc_u32 v24, s19, v27 :: v_dual_sub_nc_u32 v6, v2, v6
	v_ashrrev_i32_e32 v19, 31, v18
	v_lshl_add_u32 v26, v26, 8, v14
	v_mul_u64_e32 v[20:21], s[6:7], v[20:21]
	v_ashrrev_i32_e32 v25, 31, v24
	v_lshl_add_u32 v1, v1, 8, v6
	v_sub_nc_u32_e32 v16, v5, v16
	v_mul_u64_e32 v[18:19], s[28:29], v[18:19]
	v_mul_u64_e32 v[22:23], s[8:9], v[22:23]
	v_mul_u64_e32 v[24:25], s[10:11], v[24:25]
	v_ashrrev_i32_e32 v28, 31, v1
	v_lshl_add_u32 v27, v27, 8, v16
	v_dual_ashrrev_i32 v29, 31, v11 :: v_dual_ashrrev_i32 v30, 31, v26
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_lshlrev_b32 v32, 1, v1 :: v_dual_lshrrev_b32 v28, 25, v28
	v_dual_ashrrev_i32 v31, 31, v27 :: v_dual_lshrrev_b32 v29, 25, v29
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_lshrrev_b32 v30, 25, v30 :: v_dual_lshlrev_b32 v33, 1, v11
	v_dual_add_nc_u32 v1, v1, v28 :: v_dual_lshrrev_b32 v31, 25, v31
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_add_nc_u32 v11, v11, v29 :: v_dual_lshlrev_b32 v34, 1, v26
	v_dual_add_nc_u32 v26, v26, v30 :: v_dual_lshlrev_b32 v35, 1, v27
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_add_nc_u32 v27, v27, v31 :: v_dual_ashrrev_i32 v1, 7, v1
	v_dual_ashrrev_i32 v11, 7, v11 :: v_dual_ashrrev_i32 v26, 7, v26
	v_add_nc_u32_e32 v5, 0x400, v5
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_ashrrev_i32_e32 v27, 7, v27
	v_lshl_add_u32 v1, v1, 4, v32
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v11, v11, 4, v33
	v_lshl_add_u32 v26, v26, 4, v34
	v_add_nc_u32_e32 v4, 0x400, v4
	v_lshl_add_u32 v27, v27, 4, v35
	ds_load_u16 v1, v1
	ds_load_u16 v11, v11
	ds_load_u16 v26, v26
	ds_load_u16 v27, v27
	v_add_nc_u32_e32 v3, 0x400, v3
	v_lshl_add_u64 v[20:21], v[20:21], 1, s[4:5]
	v_add_nc_u32_e32 v2, 0x400, v2
	s_or_b32 s24, vcc_lo, s24
	v_lshl_add_u64 v[18:19], v[18:19], 1, s[4:5]
	v_lshl_add_u64 v[22:23], v[22:23], 1, s[4:5]
	v_lshl_add_u64 v[24:25], v[24:25], 1, s[4:5]
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[20:21]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u64 v[18:19], v[6:7], 1, v[18:19]
	v_lshl_add_u64 v[14:15], v[14:15], 1, v[22:23]
	s_delay_alu instid0(VALU_DEP_4)
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
	s_and_not1_b32 exec_lo, exec_lo, s24
	s_cbranch_execnz .LBB0_49
; %bb.50:
	s_or_b32 exec_lo, exec_lo, s24
	v_cmp_ne_u32_e32 vcc_lo, v8, v9
	v_lshl_or_b32 v0, v9, 8, v0
	v_dual_mov_b32 v6, s16 :: v_dual_mov_b32 v1, s23
	s_and_b32 s0, vcc_lo, exec_lo
	s_or_saveexec_b32 s1, s3
	v_mov_b64_e32 v[2:3], s[12:13]
	s_xor_b32 exec_lo, exec_lo, s1
	s_cbranch_execz .LBB0_47
.LBB0_51:
	s_abs_i32 s6, s31
	s_ashr_i32 s7, s31, 31
	s_cvt_f32_u32 s2, s6
	s_sub_co_i32 s3, 0, s6
	v_mov_b32_e32 v6, s6
	s_or_b32 s0, s0, exec_lo
	v_rcp_iflag_f32_e32 v1, s2
	v_nop
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_3)
	v_readfirstlane_b32 s2, v1
	v_mov_b32_e32 v1, s7
	s_mul_f32 s2, s2, 0x4f7ffffe
	s_cvt_u32_f32 s2, s2
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_i32 s3, s3, s2
	s_mul_hi_u32 s8, s2, s3
	s_mov_b32 s3, 0
	s_add_co_i32 s2, s2, s8
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_mov_b64_e32 v[2:3], s[2:3]
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
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_sub_nc_u32 v7, v4, v9 :: v_dual_lshlrev_b32 v4, 8, v4
	v_lshlrev_b32_e32 v9, 8, v9
	v_mul_lo_u32 v8, v7, s31
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_sub_nc_u32 v4, v4, v8 :: v_dual_add_nc_u32 v8, s15, v7
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
	v_cmp_le_i32_e32 vcc_lo, s14, v0
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
	.size	bm256_bn224_bk096_wm4_wn2_mc1, .Lfunc_end0-bm256_bn224_bk096_wm4_wn2_mc1
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel bm256_bn224_bk096_wm4_wn2_mc1
		.amdhsa_group_segment_fixed_size 195840
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
		.amdhsa_next_free_vgpr 458
		.amdhsa_next_free_sgpr 60
		.amdhsa_named_barrier_count 0
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size 85
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text.bm256_bn224_bk096_wm4_wn2_mc1,"axG",@progbits,bm256_bn224_bk096_wm4_wn2_mc1,comdat
                                        ; -- End function
	.set .Lbm256_bn224_bk096_wm4_wn2_mc1.num_vgpr, 458
	.set .Lbm256_bn224_bk096_wm4_wn2_mc1.num_agpr, 0
	.set .Lbm256_bn224_bk096_wm4_wn2_mc1.numbered_sgpr, 60
	.set .Lbm256_bn224_bk096_wm4_wn2_mc1.num_named_barrier, 0
	.set .Lbm256_bn224_bk096_wm4_wn2_mc1.private_seg_size, 0
	.set .Lbm256_bn224_bk096_wm4_wn2_mc1.uses_vcc, 1
	.set .Lbm256_bn224_bk096_wm4_wn2_mc1.uses_flat_scratch, 1
	.set .Lbm256_bn224_bk096_wm4_wn2_mc1.has_dyn_sized_stack, 0
	.set .Lbm256_bn224_bk096_wm4_wn2_mc1.has_recursion, 0
	.set .Lbm256_bn224_bk096_wm4_wn2_mc1.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 10872
; TotalNumSgprs: 62
; NumVgprs: 458
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 195840 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 28
; NumSGPRsForWavesPerEU: 62
; NumVGPRsForWavesPerEU: 458
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
	.type	__hip_cuid_c4421495a3106880,@object ; @__hip_cuid_c4421495a3106880
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_c4421495a3106880
__hip_cuid_c4421495a3106880:
	.byte	0                               ; 0x0
	.size	__hip_cuid_c4421495a3106880, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git aa451e1fe6a793394d6733051b1778633063ae96+PATCHED:d17c5aa0e3ea29cde402f58f27e39b6034effa27)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_c4421495a3106880
	.amdgpu_metadata
---
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
    .group_segment_fixed_size: 195840
    .kernarg_segment_align: 8
    .kernarg_segment_size: 132
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           bm256_bn224_bk096_wm4_wn2_mc1
    .private_segment_fixed_size: 0
    .sgpr_count:     62
    .sgpr_spill_count: 0
    .symbol:         bm256_bn224_bk096_wm4_wn2_mc1.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     458
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
