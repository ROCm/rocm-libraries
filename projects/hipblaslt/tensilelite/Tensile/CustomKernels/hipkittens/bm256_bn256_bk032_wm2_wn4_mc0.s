	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.section	.text.bm256_bn256_bk032_wm2_wn4_mc0,"axG",@progbits,bm256_bn256_bk032_wm2_wn4_mc0,comdat
	.protected	bm256_bn256_bk032_wm2_wn4_mc0 ; -- Begin function bm256_bn256_bk032_wm2_wn4_mc0
	.globl	bm256_bn256_bk032_wm2_wn4_mc0
	.p2align	8
	.type	bm256_bn256_bk032_wm2_wn4_mc0,@function
bm256_bn256_bk032_wm2_wn4_mc0: ; @bm256_bn256_bk032_wm2_wn4_mc0
; %bb.0:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1 ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_mov_b64 s[2:3], src_shared_base
	s_movk_i32 s2, 0x4400
	s_load_b96 s[16:18], s[0:1], 0x78 nv
	s_and_b64 s[2:3], s[2:3], 12
	s_getreg_b32 s6, hwreg(HW_REG_IB_STS2, 6, 4)
	s_sub_co_i32 s4, 16, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_lshr_b32 s4, s4, 2
	s_cmp_lg_u64 s[2:3], 0
	s_cselect_b32 s2, s4, 0
	s_bfe_u32 s3, ttmp6, 0x4000c
	s_bfe_u32 s5, ttmp6, 0x40010
	s_add_co_i32 s3, s3, 1
	s_and_b32 s4, ttmp6, 15
	s_mul_i32 s3, ttmp9, s3
	s_add_co_i32 s5, s5, 1
	s_add_co_i32 s4, s4, s3
	s_mul_i32 s3, ttmp7, s5
	s_bfe_u32 s5, ttmp6, 0x40004
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_co_i32 s5, s5, s3
	s_cmp_eq_u32 s6, 0
	s_cselect_b32 s36, ttmp9, s4
	s_cselect_b32 s3, ttmp7, s5
	s_wait_kmcnt 0x0
	s_add_co_i32 s4, s16, 0xff
	s_add_co_i32 s6, s17, 0xff
	s_ashr_i32 s5, s4, 31
	s_lshl_b32 s20, s36, 8
	s_lshr_b32 s5, s5, 24
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s4, s4, s5
	s_ashr_i32 s5, s6, 31
	s_ashr_i32 s4, s4, 8
	s_lshr_b32 s5, s5, 24
	s_add_co_i32 s6, s6, s5
	s_sub_co_i32 s5, s16, s20
	s_ashr_i32 s6, s6, 8
	s_min_i32 s19, s5, 0x100
	s_cmp_lt_i32 s36, s4
	s_cselect_b32 s21, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_and_b32 s5, s21, exec_lo
	s_cselect_b32 s23, s19, 0
	s_lshl_b32 s33, s3, 8
	s_sub_co_i32 s5, s17, s33
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_min_i32 s7, s5, 0x100
	s_cmp_lt_i32 s3, s6
	s_mov_b32 s5, s18
	s_cselect_b32 s17, -1, 0
	s_and_b32 s8, s17, exec_lo
	s_cselect_b32 s25, s7, 0
	s_add_co_i32 s28, s18, 31
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 2, 1), 1
	s_min_i32 s7, s18, 32
	s_cmp_gt_i32 s28, 31
	s_cselect_b32 s29, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_and_b32 s8, s29, exec_lo
	s_cselect_b32 s22, s7, 0
	s_cmp_lt_i32 s23, 0x100
	s_cselect_b32 s34, -1, 0
	s_and_b32 vcc_lo, exec_lo, s34
	s_mov_b32 s7, s34
	s_cbranch_vccnz .LBB0_2
; %bb.1:
	s_cmp_lt_i32 s25, 0x100
	s_cselect_b32 s7, -1, 0
	s_cmp_lt_i32 s22, 32
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
	v_cmp_lt_u32_e32 vcc_lo, 0xfff, v5
	s_or_b32 s7, vcc_lo, s7
	s_and_not1_b32 exec_lo, exec_lo, s7
	s_cbranch_execnz .LBB0_4
; %bb.5:
	s_or_b32 exec_lo, exec_lo, s7
	v_lshl_add_u32 v2, s2, 2, v2
	v_mov_b32_e32 v3, 0
	s_mov_b32 s7, 0
.LBB0_6:                                ; =>This Inner Loop Header: Depth=1
	v_add_nc_u32_e32 v1, 0x100, v1
	ds_store_b32 v2, v3 offset:17408
	v_add_nc_u32_e32 v2, 0x400, v2
	v_cmp_lt_u32_e32 vcc_lo, 0xfff, v1
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
	s_load_b64 s[30:31], s[0:1], 0x0 nv
	s_load_b128 s[12:15], s[0:1], 0x20 nv
	s_load_b128 s[8:11], s[0:1], 0x48 nv
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_lshrrev_b32_e32 v11 /*v267*/, 5, v0
	s_lshl_b32 s37, s2, 2
	s_add_co_i32 s6, s6, -1
	s_mov_b64 s[26:27], src_shared_base
	s_or_b32 s35, s37, 0x4400
	s_add_co_i32 s26, s4, -1
	s_min_i32 s41, s3, s6
	s_mov_b32 s0, exec_lo
	s_set_vgpr_msb 0x4004                   ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_cmpx_lt_i32_e32 0, v11 /*v267*/
	s_xor_b32 s38, exec_lo, s0
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_cbranch_execz .LBB0_12
; %bb.9:
	s_mov_b32 s39, exec_lo
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_cmpx_eq_u32_e32 1, v11 /*v267*/
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_cbranch_execz .LBB0_11
; %bb.10:
	s_cmp_gt_i32 s22, 0
	s_mov_b32 s24, s22
	s_cselect_b32 s4, -1, 0
	s_lshl_b32 s0, s41, 8
	s_wait_kmcnt 0x0
	s_bfe_i64 s[2:3], s[8:9], 0x200000
	s_ashr_i32 s1, s0, 31
	s_mov_b32 s6, 0
	s_mul_u64 s[0:1], s[2:3], s[0:1]
	s_mov_b32 s7, s6
	s_lshl_b64 s[0:1], s[0:1], 1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_nc_u64 s[2:3], s[14:15], s[0:1]
	v_dual_mov_b32 v1, s35 :: v_dual_mov_b32 v4, s2
	s_and_b32 s0, s3, 0x1ffffff
	s_and_b32 s3, s17, s4
	s_bitset1_b32 s0, 31
	v_cndmask_b32_e64 v2, 0, 1, s3
	v_mov_b32_e32 v3, s0
	v_readfirstlane_b32 s45, v1
	v_readfirstlane_b32 s46, v4
	s_lshr_b32 s0, s25, 16
	v_readfirstlane_b32 s44, v2
	v_readfirstlane_b32 s47, v3
	s_lshr_b64 s[2:3], s[24:25], 16
	s_lshl_b32 s1, s22, 16
	s_or_b32 s3, s0, 0x200000
	s_movk_i32 s4, 0x100
	s_mov_b32 s0, 0x7510000
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[44:47], s[0:7]
.LBB0_11:
	s_or_b32 exec_lo, exec_lo, s39
.LBB0_12:
	s_or_saveexec_b32 s38, s38
	s_min_i32 s24, s36, s26
	s_xor_b32 exec_lo, exec_lo, s38
	s_cbranch_execz .LBB0_14
; %bb.13:
	s_cmp_gt_i32 s22, 0
	s_mov_b32 s6, 0
	s_cselect_b32 s4, -1, 0
	s_lshl_b32 s0, s24, 8
	s_wait_kmcnt 0x0
	s_bfe_i64 s[2:3], s[12:13], 0x200000
	s_ashr_i32 s1, s0, 31
	s_and_b32 s4, s21, s4
	s_mul_u64 s[0:1], s[2:3], s[0:1]
	v_cndmask_b32_e64 v2, 0, 1, s4
	s_lshl_b64 s[2:3], s[0:1], 1
	s_lshr_b32 s0, s23, 16
	s_add_nc_u64 s[2:3], s[30:31], s[2:3]
	s_lshl_b32 s1, s22, 16
	s_and_b32 s3, s3, 0x1ffffff
	v_readfirstlane_b32 s44, v2
	s_bitset1_b32 s3, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v4, s2 :: v_dual_mov_b32 v3, s3
	s_lshr_b64 s[2:3], s[22:23], 16
	s_or_b32 s3, s0, 0x200000
	s_movk_i32 s4, 0x100
	v_readfirstlane_b32 s46, v4
	v_readfirstlane_b32 s47, v3
	s_mov_b32 s0, 0x7510000
	s_mov_b32 s7, s6
	s_mov_b32 s45, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[44:47], s[0:7]
.LBB0_14:
	s_or_b32 exec_lo, exec_lo, s38
	s_wait_tensorcnt 0x0
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	s_barrier_signal -1
	v_dual_lshlrev_b32 v1, 6, v11 /*v267*/ :: v_dual_mov_b32 v41, 0
	s_and_b32 s26, s21, s17
	s_set_vgpr_msb 0x440                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_and_b32_e32 v9 /*v265*/, 0x80, v0
	v_cndmask_b32_e64 v3 /*v259*/, 0, 1, s26
	v_and_b32_e32 v7 /*v263*/, 0xc0, v1
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_dual_mov_b32 v40, v41 :: v_dual_mov_b32 v39, v41
	v_dual_mov_b32 v38, v41 :: v_dual_mov_b32 v37, v41
	v_dual_mov_b32 v36, v41 :: v_dual_mov_b32 v35, v41
	v_dual_mov_b32 v34, v41 :: v_dual_mov_b32 v9, v41
	v_dual_mov_b32 v8, v41 :: v_dual_mov_b32 v7, v41
	v_dual_mov_b32 v6, v41 :: v_dual_mov_b32 v5, v41
	v_dual_mov_b32 v4, v41 :: v_dual_mov_b32 v3, v41
	v_dual_mov_b32 v2, v41 :: v_dual_mov_b32 v17, v41
	v_dual_mov_b32 v16, v41 :: v_dual_mov_b32 v15, v41
	v_dual_mov_b32 v14, v41 :: v_dual_mov_b32 v13, v41
	v_dual_mov_b32 v12, v41 :: v_dual_mov_b32 v11, v41
	v_dual_mov_b32 v10, v41 :: v_dual_mov_b32 v25, v41
	v_dual_mov_b32 v24, v41 :: v_dual_mov_b32 v23, v41
	v_dual_mov_b32 v22, v41 :: v_dual_mov_b32 v21, v41
	v_dual_mov_b32 v20, v41 :: v_dual_mov_b32 v19, v41
	v_dual_mov_b32 v18, v41 :: v_dual_mov_b32 v65, v41
	v_dual_mov_b32 v64, v41 :: v_dual_mov_b32 v63, v41
	v_dual_mov_b32 v62, v41 :: v_dual_mov_b32 v61, v41
	v_dual_mov_b32 v60, v41 :: v_dual_mov_b32 v59, v41
	v_dual_mov_b32 v58, v41 :: v_dual_mov_b32 v33, v41
	v_dual_mov_b32 v32, v41 :: v_dual_mov_b32 v31, v41
	v_dual_mov_b32 v30, v41 :: v_dual_mov_b32 v29, v41
	v_dual_mov_b32 v28, v41 :: v_dual_mov_b32 v27, v41
	v_dual_mov_b32 v26, v41 :: v_dual_mov_b32 v49, v41
	v_dual_mov_b32 v48, v41 :: v_dual_mov_b32 v47, v41
	v_dual_mov_b32 v46, v41 :: v_dual_mov_b32 v45, v41
	v_dual_mov_b32 v44, v41 :: v_dual_mov_b32 v43, v41
	v_dual_mov_b32 v42, v41 :: v_dual_mov_b32 v57, v41
	v_dual_mov_b32 v56, v41 :: v_dual_mov_b32 v55, v41
	v_dual_mov_b32 v54, v41 :: v_dual_mov_b32 v53, v41
	v_dual_mov_b32 v52, v41 :: v_dual_mov_b32 v51, v41
	v_dual_mov_b32 v50, v41 :: v_dual_mov_b32 v97, v41
	v_dual_mov_b32 v96, v41 :: v_dual_mov_b32 v95, v41
	v_dual_mov_b32 v94, v41 :: v_dual_mov_b32 v93, v41
	v_dual_mov_b32 v92, v41 :: v_dual_mov_b32 v91, v41
	v_dual_mov_b32 v90, v41 :: v_dual_mov_b32 v73, v41
	v_dual_mov_b32 v72, v41 :: v_dual_mov_b32 v71, v41
	v_dual_mov_b32 v70, v41 :: v_dual_mov_b32 v69, v41
	v_dual_mov_b32 v68, v41 :: v_dual_mov_b32 v67, v41
	v_dual_mov_b32 v66, v41 :: v_dual_mov_b32 v81, v41
	v_dual_mov_b32 v80, v41 :: v_dual_mov_b32 v79, v41
	v_dual_mov_b32 v78, v41 :: v_dual_mov_b32 v77, v41
	v_dual_mov_b32 v76, v41 :: v_dual_mov_b32 v75, v41
	v_dual_mov_b32 v74, v41 :: v_dual_mov_b32 v89, v41
	v_dual_mov_b32 v88, v41 :: v_dual_mov_b32 v87, v41
	v_dual_mov_b32 v86, v41 :: v_dual_mov_b32 v85, v41
	v_dual_mov_b32 v84, v41 :: v_dual_mov_b32 v83, v41
	v_dual_mov_b32 v82, v41 :: v_dual_mov_b32 v137, v41
	v_dual_mov_b32 v136, v41 :: v_dual_mov_b32 v135, v41
	v_dual_mov_b32 v134, v41 :: v_dual_mov_b32 v133, v41
	v_dual_mov_b32 v132, v41 :: v_dual_mov_b32 v131, v41
	v_dual_mov_b32 v130, v41 :: v_dual_mov_b32 v105, v41
	v_dual_mov_b32 v104, v41 :: v_dual_mov_b32 v103, v41
	v_dual_mov_b32 v102, v41 :: v_dual_mov_b32 v101, v41
	v_dual_mov_b32 v100, v41 :: v_dual_mov_b32 v99, v41
	v_dual_mov_b32 v98, v41 :: v_dual_mov_b32 v113, v41
	v_dual_mov_b32 v112, v41 :: v_dual_mov_b32 v111, v41
	v_dual_mov_b32 v110, v41 :: v_dual_mov_b32 v109, v41
	v_dual_mov_b32 v108, v41 :: v_dual_mov_b32 v107, v41
	v_dual_mov_b32 v106, v41 :: v_dual_mov_b32 v121, v41
	v_dual_mov_b32 v120, v41 :: v_dual_mov_b32 v119, v41
	v_dual_mov_b32 v118, v41 :: v_dual_mov_b32 v117, v41
	v_dual_mov_b32 v116, v41 :: v_dual_mov_b32 v115, v41
	v_dual_mov_b32 v114, v41 :: v_dual_mov_b32 v169, v41
	v_dual_mov_b32 v168, v41 :: v_dual_mov_b32 v167, v41
	v_dual_mov_b32 v166, v41 :: v_dual_mov_b32 v165, v41
	v_dual_mov_b32 v164, v41 :: v_dual_mov_b32 v163, v41
	v_dual_mov_b32 v162, v41 :: v_dual_mov_b32 v129, v41
	v_dual_mov_b32 v128, v41 :: v_dual_mov_b32 v127, v41
	v_dual_mov_b32 v126, v41 :: v_dual_mov_b32 v125, v41
	v_dual_mov_b32 v124, v41 :: v_dual_mov_b32 v123, v41
	v_dual_mov_b32 v122, v41 :: v_dual_mov_b32 v177, v41
	v_dual_mov_b32 v176, v41 :: v_dual_mov_b32 v175, v41
	v_dual_mov_b32 v174, v41 :: v_dual_mov_b32 v173, v41
	v_dual_mov_b32 v172, v41 :: v_dual_mov_b32 v171, v41
	v_dual_mov_b32 v170, v41 :: v_dual_mov_b32 v185, v41
	v_dual_mov_b32 v184, v41 :: v_dual_mov_b32 v183, v41
	v_dual_mov_b32 v182, v41 :: v_dual_mov_b32 v181, v41
	v_dual_mov_b32 v180, v41 :: v_dual_mov_b32 v179, v41
	v_dual_mov_b32 v178, v41 :: v_dual_mov_b32 v193, v41
	v_dual_mov_b32 v192, v41 :: v_dual_mov_b32 v191, v41
	v_dual_mov_b32 v190, v41 :: v_dual_mov_b32 v189, v41
	v_dual_mov_b32 v188, v41 :: v_dual_mov_b32 v187, v41
	v_dual_mov_b32 v186, v41 :: v_dual_mov_b32 v201, v41
	v_dual_mov_b32 v200, v41 :: v_dual_mov_b32 v199, v41
	v_dual_mov_b32 v198, v41 :: v_dual_mov_b32 v197, v41
	v_dual_mov_b32 v196, v41 :: v_dual_mov_b32 v195, v41
	v_dual_mov_b32 v194, v41 :: v_dual_mov_b32 v209, v41
	v_dual_mov_b32 v208, v41 :: v_dual_mov_b32 v207, v41
	v_dual_mov_b32 v206, v41 :: v_dual_mov_b32 v205, v41
	v_dual_mov_b32 v204, v41 :: v_dual_mov_b32 v203, v41
	v_dual_mov_b32 v202, v41 :: v_dual_mov_b32 v217, v41
	v_dual_mov_b32 v216, v41 :: v_dual_mov_b32 v215, v41
	v_dual_mov_b32 v214, v41 :: v_dual_mov_b32 v213, v41
	v_dual_mov_b32 v212, v41 :: v_dual_mov_b32 v211, v41
	v_dual_mov_b32 v210, v41 :: v_dual_mov_b32 v225, v41
	v_dual_mov_b32 v224, v41 :: v_dual_mov_b32 v223, v41
	v_dual_mov_b32 v222, v41 :: v_dual_mov_b32 v221, v41
	v_dual_mov_b32 v220, v41 :: v_dual_mov_b32 v219, v41
	v_dual_mov_b32 v218, v41 :: v_dual_mov_b32 v233, v41
	v_dual_mov_b32 v232, v41 :: v_dual_mov_b32 v231, v41
	v_dual_mov_b32 v230, v41 :: v_dual_mov_b32 v229, v41
	v_dual_mov_b32 v228, v41 :: v_dual_mov_b32 v227, v41
	v_dual_mov_b32 v226, v41 :: v_dual_mov_b32 v241, v41
	v_dual_mov_b32 v240, v41 :: v_dual_mov_b32 v239, v41
	v_dual_mov_b32 v238, v41 :: v_dual_mov_b32 v237, v41
	v_dual_mov_b32 v236, v41 :: v_dual_mov_b32 v235, v41
	v_dual_mov_b32 v234, v41 :: v_dual_mov_b32 v249, v41
	v_dual_mov_b32 v248, v41 :: v_dual_mov_b32 v247, v41
	v_dual_mov_b32 v246, v41 :: v_dual_mov_b32 v245, v41
	v_dual_mov_b32 v244, v41 :: v_dual_mov_b32 v243, v41
	v_dual_mov_b32 v242, v41 :: v_dual_mov_b32 v255, v41
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_dual_mov_b32 v1 /*v257*/, v41 :: v_dual_mov_b32 v0 /*v256*/, v41
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_dual_mov_b32 v254, v41 :: v_dual_mov_b32 v253, v41
	v_dual_mov_b32 v252, v41 :: v_dual_mov_b32 v251, v41
	v_dual_mov_b32 v250, v41 :: v_dual_mov_b32 v145, v41
	v_dual_mov_b32 v144, v41 :: v_dual_mov_b32 v143, v41
	v_dual_mov_b32 v142, v41 :: v_dual_mov_b32 v141, v41
	v_dual_mov_b32 v140, v41 :: v_dual_mov_b32 v139, v41
	v_dual_mov_b32 v138, v41 :: v_dual_mov_b32 v153, v41
	v_dual_mov_b32 v152, v41 :: v_dual_mov_b32 v151, v41
	v_dual_mov_b32 v150, v41 :: v_dual_mov_b32 v149, v41
	v_dual_mov_b32 v148, v41 :: v_dual_mov_b32 v147, v41
	v_dual_mov_b32 v146, v41 :: v_dual_mov_b32 v161, v41
	v_dual_mov_b32 v160, v41 :: v_dual_mov_b32 v159, v41
	v_dual_mov_b32 v158, v41 :: v_dual_mov_b32 v157, v41
	v_dual_mov_b32 v156, v41 :: v_dual_mov_b32 v155, v41
	v_mov_b32_e32 v154, v41
	s_and_not1_b32 vcc_lo, exec_lo, s29
	s_barrier_wait -1
	s_cbranch_vccnz .LBB0_37
; %bb.15:
	v_dual_lshlrev_b32 v4, 5, v0 :: v_dual_bitop2_b32 v3, 16, v0 bitop3:0x40
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_dual_lshlrev_b32 v1, 5, v7 /*v263*/ :: v_dual_lshlrev_b32 v2, 5, v9 /*v265*/
	s_mov_b64 s[0:1], src_shared_base
	s_or_b32 s2, s37, 0x8800
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_and_b32_e32 v4, 0x1e0, v4
	s_mov_b32 s3, s1
	s_mov_b32 s7, 0
	s_and_b64 s[2:3], s[2:3], 15
	s_mov_b32 s38, s1
	v_or3_b32 v2, v2, v3, v4
	s_sub_co_i32 s0, 16, s2
	v_or3_b32 v3, v1, v3, v4
	s_lshr_b32 s0, s0, 2
	s_cmp_lg_u64 s[2:3], 0
	v_or_b32_e32 v6, 0x200, v2
	v_or_b32_e32 v7, 0x400, v2
	v_lshrrev_b32_e32 v5, 4, v2
	v_or_b32_e32 v8, 0x800, v2
	s_cselect_b32 s0, s0, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_lshrrev_b32 v6, 4, v6 :: v_dual_lshrrev_b32 v7, 4, v7
	v_and_b32_e32 v5, 0x118, v5
	s_lshl2_add_u32 s2, s0, s37
	v_or_b32_e32 v9, 0xa00, v2
	v_or_b32_e32 v10, 0xc00, v2
	v_lshrrev_b32_e32 v8, 4, v8
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_dual_mov_b32 v5 /*v261*/, 0 :: v_dual_add_nc_u32 v2 /*v258*/, v5, v2
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_and_b32_e32 v5, 0x1f8, v6
	s_add_co_i32 s0, s2, 0xcc00
	v_lshrrev_b32_e32 v9, 4, v9
	s_and_b32 s6, s0, 15
	v_or_b32_e32 v1, 0xe00, v2
	s_sub_co_i32 s3, 16, s6
	v_or_b32_e32 v6, 0x600, v2
	s_add_co_i32 s37, s2, 0x8800
	s_lshr_b32 s2, s3, 2
	s_cmp_lg_u64 s[6:7], 0
	v_and_b32_e32 v4, 0x1f8, v8
	v_lshrrev_b32_e32 v6, 4, v6
	v_and_b32_e32 v8, 0x1f8, v9
	v_dual_lshrrev_b32 v10, 4, v10 :: v_dual_lshrrev_b32 v1, 4, v1
	v_or_b32_e32 v11, 0x200, v3
	v_or_b32_e32 v12, 0x400, v3
	v_or_b32_e32 v13, 0x600, v3
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_and_b32_e32 v9, 0x1f8, v10
	s_cselect_b32 s2, s2, 0
	v_and_b32_e32 v7, 0x1f8, v7
	v_dual_lshrrev_b32 v11, 4, v11 :: v_dual_lshrrev_b32 v13, 4, v13
	s_ashr_i32 s3, s28, 31
	v_dual_lshrrev_b32 v12, 4, v12 :: v_dual_lshrrev_b32 v10, 4, v3
	s_lshr_b32 s3, s3, 27
	s_lshl_b32 s6, s2, 2
	s_add_co_i32 s28, s28, s3
	v_and_b32_e32 v6, 0x1f8, v6
	v_and_b32_e32 v10, 0x198, v10
	s_ashr_i32 s39, s28, 5
	s_cmp_lt_i32 s25, 0x100
	v_and_b32_e32 v14, 0x1f8, v1
	s_cselect_b32 s40, -1, 0
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_dual_add_nc_u32 v6 /*v262*/, v10, v3 :: v_dual_add_nc_u32 v8 /*v264*/, v5, v2
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_and_b32_e32 v10, 0x1f8, v11
	v_and_b32_e32 v11, 0x1f8, v12
	s_wait_kmcnt 0x0
	s_bfe_i64 s[12:13], s[12:13], 0x200000
	s_add_nc_u64 s[28:29], s[0:1], s[6:7]
	s_lshl_b32 s0, s24, 8
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_dual_add_nc_u32 v10 /*v266*/, v7, v2 :: v_dual_add_nc_u32 v12 /*v268*/, v6, v2
	s_ashr_i32 s1, s0, 31
	v_dual_add_nc_u32 v14 /*v270*/, v4, v2 :: v_dual_add_nc_u32 v18 /*v274*/, v9, v2
	s_mul_u64 s[0:1], s[12:13], s[0:1]
	v_dual_add_nc_u32 v20 /*v276*/, v14, v2 :: v_dual_add_nc_u32 v24 /*v280*/, v11, v3
	v_or_b32_e32 v13 /*v269*/, 0xf00, v0
	v_add_nc_u32_e32 v22 /*v278*/, v10, v3
	v_lshl_or_b32 v28 /*v284*/, v0, 2, 0x4000
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_and_b32_e32 v12, 0x1f8, v13
	s_set_vgpr_msb 0x41                     ;  msbs: dst=1 src0=1 src1=0 src2=0
	v_dual_mov_b32 v29 /*v285*/, v5 /*v261*/ :: v_dual_mov_b32 v0 /*v256*/, v5 /*v261*/
	s_lshl_b32 s2, s41, 8
	s_set_vgpr_msb 0x4101                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_dual_mov_b32 v34, v5 /*v261*/ :: v_dual_mov_b32 v35, v5 /*v261*/
	s_ashr_i32 s3, s2, 31
	s_bfe_i64 s[8:9], s[8:9], 0x200000
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_add_nc_u32_e32 v26 /*v282*/, v12, v3
	s_mul_u64 s[2:3], s[8:9], s[2:3]
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_dual_mov_b32 v36, v5 /*v261*/ :: v_dual_mov_b32 v37, v5 /*v261*/
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_add_nc_u32_e32 v16 /*v272*/, v8, v2
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_dual_mov_b32 v38, v5 /*v261*/ :: v_dual_mov_b32 v39, v5 /*v261*/
	v_or_b32_e32 v1, 0x100, v0
	v_dual_mov_b32 v40, v5 /*v261*/ :: v_dual_mov_b32 v41, v5 /*v261*/
	v_dual_mov_b32 v2, v5 /*v261*/ :: v_dual_mov_b32 v3, v5 /*v261*/
	v_dual_mov_b32 v4, v5 /*v261*/ :: v_dual_mov_b32 v5, v5 /*v261*/
	v_dual_mov_b32 v6, v5 /*v261*/ :: v_dual_mov_b32 v7, v5 /*v261*/
	v_dual_mov_b32 v8, v5 /*v261*/ :: v_dual_mov_b32 v9, v5 /*v261*/
	v_dual_mov_b32 v10, v5 /*v261*/ :: v_dual_mov_b32 v11, v5 /*v261*/
	v_dual_mov_b32 v12, v5 /*v261*/ :: v_dual_mov_b32 v13, v5 /*v261*/
	v_dual_mov_b32 v14, v5 /*v261*/ :: v_dual_mov_b32 v15, v5 /*v261*/
	v_dual_mov_b32 v16, v5 /*v261*/ :: v_dual_mov_b32 v17, v5 /*v261*/
	v_dual_mov_b32 v18, v5 /*v261*/ :: v_dual_mov_b32 v19, v5 /*v261*/
	v_dual_mov_b32 v20, v5 /*v261*/ :: v_dual_mov_b32 v21, v5 /*v261*/
	v_dual_mov_b32 v22, v5 /*v261*/ :: v_dual_mov_b32 v23, v5 /*v261*/
	v_dual_mov_b32 v24, v5 /*v261*/ :: v_dual_mov_b32 v25, v5 /*v261*/
	v_dual_mov_b32 v58, v5 /*v261*/ :: v_dual_mov_b32 v59, v5 /*v261*/
	v_dual_mov_b32 v60, v5 /*v261*/ :: v_dual_mov_b32 v61, v5 /*v261*/
	v_dual_mov_b32 v62, v5 /*v261*/ :: v_dual_mov_b32 v63, v5 /*v261*/
	v_dual_mov_b32 v64, v5 /*v261*/ :: v_dual_mov_b32 v65, v5 /*v261*/
	v_dual_mov_b32 v26, v5 /*v261*/ :: v_dual_mov_b32 v27, v5 /*v261*/
	v_dual_mov_b32 v28, v5 /*v261*/ :: v_dual_mov_b32 v29, v5 /*v261*/
	v_dual_mov_b32 v30, v5 /*v261*/ :: v_dual_mov_b32 v31, v5 /*v261*/
	v_dual_mov_b32 v32, v5 /*v261*/ :: v_dual_mov_b32 v33, v5 /*v261*/
	v_dual_mov_b32 v42, v5 /*v261*/ :: v_dual_mov_b32 v43, v5 /*v261*/
	v_dual_mov_b32 v44, v5 /*v261*/ :: v_dual_mov_b32 v45, v5 /*v261*/
	v_dual_mov_b32 v46, v5 /*v261*/ :: v_dual_mov_b32 v47, v5 /*v261*/
	v_dual_mov_b32 v48, v5 /*v261*/ :: v_dual_mov_b32 v49, v5 /*v261*/
	v_dual_mov_b32 v50, v5 /*v261*/ :: v_dual_mov_b32 v51, v5 /*v261*/
	v_dual_mov_b32 v52, v5 /*v261*/ :: v_dual_mov_b32 v53, v5 /*v261*/
	v_dual_mov_b32 v54, v5 /*v261*/ :: v_dual_mov_b32 v55, v5 /*v261*/
	v_dual_mov_b32 v56, v5 /*v261*/ :: v_dual_mov_b32 v57, v5 /*v261*/
	v_dual_mov_b32 v90, v5 /*v261*/ :: v_dual_mov_b32 v91, v5 /*v261*/
	v_dual_mov_b32 v92, v5 /*v261*/ :: v_dual_mov_b32 v93, v5 /*v261*/
	v_dual_mov_b32 v94, v5 /*v261*/ :: v_dual_mov_b32 v95, v5 /*v261*/
	v_dual_mov_b32 v96, v5 /*v261*/ :: v_dual_mov_b32 v97, v5 /*v261*/
	v_dual_mov_b32 v66, v5 /*v261*/ :: v_dual_mov_b32 v67, v5 /*v261*/
	v_dual_mov_b32 v68, v5 /*v261*/ :: v_dual_mov_b32 v69, v5 /*v261*/
	v_dual_mov_b32 v70, v5 /*v261*/ :: v_dual_mov_b32 v71, v5 /*v261*/
	v_dual_mov_b32 v72, v5 /*v261*/ :: v_dual_mov_b32 v73, v5 /*v261*/
	v_dual_mov_b32 v74, v5 /*v261*/ :: v_dual_mov_b32 v75, v5 /*v261*/
	v_dual_mov_b32 v76, v5 /*v261*/ :: v_dual_mov_b32 v77, v5 /*v261*/
	v_dual_mov_b32 v78, v5 /*v261*/ :: v_dual_mov_b32 v79, v5 /*v261*/
	v_dual_mov_b32 v80, v5 /*v261*/ :: v_dual_mov_b32 v81, v5 /*v261*/
	v_dual_mov_b32 v82, v5 /*v261*/ :: v_dual_mov_b32 v83, v5 /*v261*/
	v_dual_mov_b32 v84, v5 /*v261*/ :: v_dual_mov_b32 v85, v5 /*v261*/
	v_dual_mov_b32 v86, v5 /*v261*/ :: v_dual_mov_b32 v87, v5 /*v261*/
	v_dual_mov_b32 v88, v5 /*v261*/ :: v_dual_mov_b32 v89, v5 /*v261*/
	v_dual_mov_b32 v130, v5 /*v261*/ :: v_dual_mov_b32 v131, v5 /*v261*/
	v_dual_mov_b32 v132, v5 /*v261*/ :: v_dual_mov_b32 v133, v5 /*v261*/
	v_dual_mov_b32 v134, v5 /*v261*/ :: v_dual_mov_b32 v135, v5 /*v261*/
	v_dual_mov_b32 v136, v5 /*v261*/ :: v_dual_mov_b32 v137, v5 /*v261*/
	v_dual_mov_b32 v98, v5 /*v261*/ :: v_dual_mov_b32 v99, v5 /*v261*/
	v_dual_mov_b32 v100, v5 /*v261*/ :: v_dual_mov_b32 v101, v5 /*v261*/
	v_dual_mov_b32 v102, v5 /*v261*/ :: v_dual_mov_b32 v103, v5 /*v261*/
	v_dual_mov_b32 v104, v5 /*v261*/ :: v_dual_mov_b32 v105, v5 /*v261*/
	v_dual_mov_b32 v106, v5 /*v261*/ :: v_dual_mov_b32 v107, v5 /*v261*/
	v_dual_mov_b32 v108, v5 /*v261*/ :: v_dual_mov_b32 v109, v5 /*v261*/
	v_dual_mov_b32 v110, v5 /*v261*/ :: v_dual_mov_b32 v111, v5 /*v261*/
	v_dual_mov_b32 v112, v5 /*v261*/ :: v_dual_mov_b32 v113, v5 /*v261*/
	v_dual_mov_b32 v114, v5 /*v261*/ :: v_dual_mov_b32 v115, v5 /*v261*/
	v_dual_mov_b32 v116, v5 /*v261*/ :: v_dual_mov_b32 v117, v5 /*v261*/
	v_dual_mov_b32 v118, v5 /*v261*/ :: v_dual_mov_b32 v119, v5 /*v261*/
	v_dual_mov_b32 v120, v5 /*v261*/ :: v_dual_mov_b32 v121, v5 /*v261*/
	v_dual_mov_b32 v162, v5 /*v261*/ :: v_dual_mov_b32 v163, v5 /*v261*/
	v_dual_mov_b32 v164, v5 /*v261*/ :: v_dual_mov_b32 v165, v5 /*v261*/
	v_dual_mov_b32 v166, v5 /*v261*/ :: v_dual_mov_b32 v167, v5 /*v261*/
	v_dual_mov_b32 v168, v5 /*v261*/ :: v_dual_mov_b32 v169, v5 /*v261*/
	v_dual_mov_b32 v122, v5 /*v261*/ :: v_dual_mov_b32 v123, v5 /*v261*/
	v_dual_mov_b32 v124, v5 /*v261*/ :: v_dual_mov_b32 v125, v5 /*v261*/
	v_dual_mov_b32 v126, v5 /*v261*/ :: v_dual_mov_b32 v127, v5 /*v261*/
	v_dual_mov_b32 v128, v5 /*v261*/ :: v_dual_mov_b32 v129, v5 /*v261*/
	v_dual_mov_b32 v170, v5 /*v261*/ :: v_dual_mov_b32 v171, v5 /*v261*/
	v_dual_mov_b32 v172, v5 /*v261*/ :: v_dual_mov_b32 v173, v5 /*v261*/
	v_dual_mov_b32 v174, v5 /*v261*/ :: v_dual_mov_b32 v175, v5 /*v261*/
	v_dual_mov_b32 v176, v5 /*v261*/ :: v_dual_mov_b32 v177, v5 /*v261*/
	v_dual_mov_b32 v178, v5 /*v261*/ :: v_dual_mov_b32 v179, v5 /*v261*/
	v_dual_mov_b32 v180, v5 /*v261*/ :: v_dual_mov_b32 v181, v5 /*v261*/
	v_dual_mov_b32 v182, v5 /*v261*/ :: v_dual_mov_b32 v183, v5 /*v261*/
	v_dual_mov_b32 v184, v5 /*v261*/ :: v_dual_mov_b32 v185, v5 /*v261*/
	v_dual_mov_b32 v186, v5 /*v261*/ :: v_dual_mov_b32 v187, v5 /*v261*/
	v_dual_mov_b32 v188, v5 /*v261*/ :: v_dual_mov_b32 v189, v5 /*v261*/
	v_dual_mov_b32 v190, v5 /*v261*/ :: v_dual_mov_b32 v191, v5 /*v261*/
	v_dual_mov_b32 v192, v5 /*v261*/ :: v_dual_mov_b32 v193, v5 /*v261*/
	v_dual_mov_b32 v194, v5 /*v261*/ :: v_dual_mov_b32 v195, v5 /*v261*/
	v_dual_mov_b32 v196, v5 /*v261*/ :: v_dual_mov_b32 v197, v5 /*v261*/
	v_dual_mov_b32 v198, v5 /*v261*/ :: v_dual_mov_b32 v199, v5 /*v261*/
	v_dual_mov_b32 v200, v5 /*v261*/ :: v_dual_mov_b32 v201, v5 /*v261*/
	v_dual_mov_b32 v202, v5 /*v261*/ :: v_dual_mov_b32 v203, v5 /*v261*/
	v_dual_mov_b32 v204, v5 /*v261*/ :: v_dual_mov_b32 v205, v5 /*v261*/
	v_dual_mov_b32 v206, v5 /*v261*/ :: v_dual_mov_b32 v207, v5 /*v261*/
	v_dual_mov_b32 v208, v5 /*v261*/ :: v_dual_mov_b32 v209, v5 /*v261*/
	v_dual_mov_b32 v210, v5 /*v261*/ :: v_dual_mov_b32 v211, v5 /*v261*/
	v_dual_mov_b32 v212, v5 /*v261*/ :: v_dual_mov_b32 v213, v5 /*v261*/
	v_dual_mov_b32 v214, v5 /*v261*/ :: v_dual_mov_b32 v215, v5 /*v261*/
	v_dual_mov_b32 v216, v5 /*v261*/ :: v_dual_mov_b32 v217, v5 /*v261*/
	v_dual_mov_b32 v218, v5 /*v261*/ :: v_dual_mov_b32 v219, v5 /*v261*/
	v_dual_mov_b32 v220, v5 /*v261*/ :: v_dual_mov_b32 v221, v5 /*v261*/
	v_dual_mov_b32 v222, v5 /*v261*/ :: v_dual_mov_b32 v223, v5 /*v261*/
	v_dual_mov_b32 v224, v5 /*v261*/ :: v_dual_mov_b32 v225, v5 /*v261*/
	v_dual_mov_b32 v226, v5 /*v261*/ :: v_dual_mov_b32 v227, v5 /*v261*/
	v_dual_mov_b32 v228, v5 /*v261*/ :: v_dual_mov_b32 v229, v5 /*v261*/
	v_dual_mov_b32 v230, v5 /*v261*/ :: v_dual_mov_b32 v231, v5 /*v261*/
	v_dual_mov_b32 v232, v5 /*v261*/ :: v_dual_mov_b32 v233, v5 /*v261*/
	v_dual_mov_b32 v234, v5 /*v261*/ :: v_dual_mov_b32 v235, v5 /*v261*/
	v_dual_mov_b32 v236, v5 /*v261*/ :: v_dual_mov_b32 v237, v5 /*v261*/
	v_dual_mov_b32 v238, v5 /*v261*/ :: v_dual_mov_b32 v239, v5 /*v261*/
	v_dual_mov_b32 v240, v5 /*v261*/ :: v_dual_mov_b32 v241, v5 /*v261*/
	v_dual_mov_b32 v242, v5 /*v261*/ :: v_dual_mov_b32 v243, v5 /*v261*/
	v_dual_mov_b32 v244, v5 /*v261*/ :: v_dual_mov_b32 v245, v5 /*v261*/
	v_dual_mov_b32 v246, v5 /*v261*/ :: v_dual_mov_b32 v247, v5 /*v261*/
	v_dual_mov_b32 v248, v5 /*v261*/ :: v_dual_mov_b32 v249, v5 /*v261*/
	v_dual_mov_b32 v250, v5 /*v261*/ :: v_dual_mov_b32 v251, v5 /*v261*/
	v_dual_mov_b32 v252, v5 /*v261*/ :: v_dual_mov_b32 v253, v5 /*v261*/
	v_dual_mov_b32 v254, v5 /*v261*/ :: v_dual_mov_b32 v255, v5 /*v261*/
	s_set_vgpr_msb 0x141                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	v_mov_b32_e32 v1 /*v257*/, v5 /*v261*/
	s_set_vgpr_msb 0x4101                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_dual_mov_b32 v138, v5 /*v261*/ :: v_dual_mov_b32 v139, v5 /*v261*/
	v_dual_mov_b32 v140, v5 /*v261*/ :: v_dual_mov_b32 v141, v5 /*v261*/
	v_dual_mov_b32 v142, v5 /*v261*/ :: v_dual_mov_b32 v143, v5 /*v261*/
	v_dual_mov_b32 v144, v5 /*v261*/ :: v_dual_mov_b32 v145, v5 /*v261*/
	v_dual_mov_b32 v146, v5 /*v261*/ :: v_dual_mov_b32 v147, v5 /*v261*/
	v_dual_mov_b32 v148, v5 /*v261*/ :: v_dual_mov_b32 v149, v5 /*v261*/
	v_dual_mov_b32 v150, v5 /*v261*/ :: v_dual_mov_b32 v151, v5 /*v261*/
	v_dual_mov_b32 v152, v5 /*v261*/ :: v_dual_mov_b32 v153, v5 /*v261*/
	v_dual_mov_b32 v154, v5 /*v261*/ :: v_dual_mov_b32 v155, v5 /*v261*/
	v_dual_mov_b32 v156, v5 /*v261*/ :: v_dual_mov_b32 v157, v5 /*v261*/
	v_dual_mov_b32 v158, v5 /*v261*/ :: v_dual_mov_b32 v159, v5 /*v261*/
	v_dual_mov_b32 v160, v5 /*v261*/ :: v_dual_mov_b32 v161, v5 /*v261*/
	s_lshr_b32 s41, s25, 16
	s_lshr_b32 s42, s23, 16
	s_lshl_b64 s[2:3], s[2:3], 1
	s_lshl_b64 s[0:1], s[0:1], 1
	s_mov_b32 s36, s27
	s_movk_i32 s4, 0x100
	s_bitset1_b32 s41, 21
	s_bitset1_b32 s42, 21
	s_add_nc_u64 s[8:9], s[14:15], s[2:3]
	s_add_nc_u64 s[12:13], s[30:31], s[0:1]
	s_mov_b32 s0, 0x7510000
	s_mov_b32 s14, -1
	s_mov_b32 s15, s7
	s_set_vgpr_msb 0x100                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_branch .LBB0_17
.LBB0_16:                               ;   in Loop: Header=BB0_17 Depth=1
	s_cmp_eq_u32 s15, s39
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_scc1 .LBB0_37
.LBB0_17:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_19 Depth 2
                                        ;     Child Loop BB0_22 Depth 2
                                        ;     Child Loop BB0_24 Depth 2
                                        ;     Child Loop BB0_27 Depth 2
	s_and_b32 s30, s15, 1
	s_add_co_i32 s15, s15, 1
	s_xor_b32 s31, s30, 1
	s_lshl_b32 s1, s15, 5
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_sub_co_i32 s1, s18, s1
	s_min_i32 s2, s1, 32
	s_cmp_lt_i32 s15, s39
	s_cselect_b32 s1, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_and_b32 s3, s1, exec_lo
	s_cselect_b32 s22, s2, 0
	s_cmp_lt_i32 s22, 32
	s_cselect_b32 s2, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b32 s2, s40, s2
	s_or_b32 s2, s34, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB0_29
; %bb.18:                               ;   in Loop: Header=BB0_17 Depth=1
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_mov_b64_e32 v[30:31] /*v[286:287]*/, v[0:1]
	v_mov_b32_e32 v15 /*v271*/, 16
	s_cmp_lg_u32 s31, 0
	s_mov_b32 s6, 0
	s_cselect_b32 s3, s38, s27
	s_cselect_b32 s2, s37, 0
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
.LBB0_19:                               ;   Parent Loop BB0_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_set_vgpr_msb 0x45                     ;  msbs: dst=1 src0=1 src1=1 src2=0
	v_dual_mov_b32 v4 /*v260*/, v30 /*v286*/ :: v_dual_add_nc_u32 v15 /*v271*/, -2, v15 /*v271*/
	v_dual_mov_b32 v32 /*v288*/, v31 /*v287*/ :: v_dual_mov_b32 v33 /*v289*/, v5 /*v261*/
	v_add_nc_u32_e32 v31 /*v287*/, 0x200, v31 /*v287*/
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u64 v[34:35] /*v[290:291]*/, v[4:5] /*v[260:261]*/, 2, s[2:3]
	v_cmp_eq_u32_e32 vcc_lo, 0, v15 /*v271*/
	v_add_nc_u32_e32 v30 /*v286*/, 0x200, v30 /*v286*/
	v_lshl_add_u64 v[32:33] /*v[288:289]*/, v[32:33] /*v[288:289]*/, 2, s[2:3]
	s_clause 0x1
	flat_store_b32 v[34:35] /*v[290:291]*/, v5 /*v261*/
	flat_store_b32 v[32:33] /*v[288:289]*/, v5 /*v261*/
	s_or_b32 s6, vcc_lo, s6
	s_wait_xcnt 0x0
	s_and_not1_b32 exec_lo, exec_lo, s6
	s_set_vgpr_msb 0x4500                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_cbranch_execnz .LBB0_19
; %bb.20:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s6
	s_and_saveexec_b32 s6, s14
	s_cbranch_execz .LBB0_23
; %bb.21:                               ;   in Loop: Header=BB0_17 Depth=1
	s_set_vgpr_msb 0x45                     ;  msbs: dst=1 src0=1 src1=1 src2=0
	v_add_nc_u64_e32 v[30:31] /*v[286:287]*/, s[2:3], v[28:29] /*v[284:285]*/
	v_mov_b32_e32 v4 /*v260*/, v13 /*v269*/
	s_mov_b32 s2, 0
	s_set_vgpr_msb 0x4500                   ;  msbs: dst=0 src0=0 src1=0 src2=0
.LBB0_22:                               ;   Parent Loop BB0_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_set_vgpr_msb 0x45                     ;  msbs: dst=1 src0=1 src1=1 src2=0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v4 /*v260*/, 0x100, v4 /*v260*/
	flat_store_b32 v[30:31] /*v[286:287]*/, v5 /*v261*/
	s_wait_xcnt 0x0
	v_add_nc_u64_e32 v[30:31] /*v[286:287]*/, 0x400, v[30:31] /*v[286:287]*/
	v_cmp_lt_u32_e32 vcc_lo, 0xfff, v4 /*v260*/
	s_or_b32 s2, vcc_lo, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s2
	s_set_vgpr_msb 0x4500                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_cbranch_execnz .LBB0_22
.LBB0_23:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s6
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_mov_b64_e32 v[30:31] /*v[286:287]*/, v[0:1]
	v_mov_b32_e32 v15 /*v271*/, 16
	s_cmp_lg_u32 s31, 0
	s_mov_b32 s6, 0
	s_cselect_b32 s3, s29, s36
	s_cselect_b32 s2, s28, s35
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
.LBB0_24:                               ;   Parent Loop BB0_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_set_vgpr_msb 0x45                     ;  msbs: dst=1 src0=1 src1=1 src2=0
	v_dual_mov_b32 v4 /*v260*/, v30 /*v286*/ :: v_dual_add_nc_u32 v15 /*v271*/, -2, v15 /*v271*/
	v_dual_mov_b32 v32 /*v288*/, v31 /*v287*/ :: v_dual_mov_b32 v33 /*v289*/, v5 /*v261*/
	v_add_nc_u32_e32 v31 /*v287*/, 0x200, v31 /*v287*/
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u64 v[34:35] /*v[290:291]*/, v[4:5] /*v[260:261]*/, 2, s[2:3]
	v_cmp_eq_u32_e32 vcc_lo, 0, v15 /*v271*/
	v_add_nc_u32_e32 v30 /*v286*/, 0x200, v30 /*v286*/
	v_lshl_add_u64 v[32:33] /*v[288:289]*/, v[32:33] /*v[288:289]*/, 2, s[2:3]
	s_clause 0x1
	flat_store_b32 v[34:35] /*v[290:291]*/, v5 /*v261*/
	flat_store_b32 v[32:33] /*v[288:289]*/, v5 /*v261*/
	s_or_b32 s6, vcc_lo, s6
	s_wait_xcnt 0x0
	s_and_not1_b32 exec_lo, exec_lo, s6
	s_set_vgpr_msb 0x4500                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_cbranch_execnz .LBB0_24
; %bb.25:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s6
	s_and_saveexec_b32 s6, s14
	s_cbranch_execz .LBB0_28
; %bb.26:                               ;   in Loop: Header=BB0_17 Depth=1
	s_set_vgpr_msb 0x45                     ;  msbs: dst=1 src0=1 src1=1 src2=0
	v_add_nc_u64_e32 v[30:31] /*v[286:287]*/, s[2:3], v[28:29] /*v[284:285]*/
	v_mov_b32_e32 v4 /*v260*/, v13 /*v269*/
	s_mov_b32 s2, 0
	s_set_vgpr_msb 0x4500                   ;  msbs: dst=0 src0=0 src1=0 src2=0
.LBB0_27:                               ;   Parent Loop BB0_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_set_vgpr_msb 0x45                     ;  msbs: dst=1 src0=1 src1=1 src2=0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v4 /*v260*/, 0x100, v4 /*v260*/
	flat_store_b32 v[30:31] /*v[286:287]*/, v5 /*v261*/
	s_wait_xcnt 0x0
	v_add_nc_u64_e32 v[30:31] /*v[286:287]*/, 0x400, v[30:31] /*v[286:287]*/
	v_cmp_lt_u32_e32 vcc_lo, 0xfff, v4 /*v260*/
	s_or_b32 s2, vcc_lo, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s2
	s_set_vgpr_msb 0x4500                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_cbranch_execnz .LBB0_27
.LBB0_28:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s6
	s_barrier_signal -1
	s_barrier_wait -1
.LBB0_29:                               ;   in Loop: Header=BB0_17 Depth=1
	s_and_b32 s1, s1, exec_lo
	s_cselect_b32 s43, s15, 0
	s_mov_b32 s1, exec_lo
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_cmpx_lt_i32_e32 0, v11 /*v267*/
	s_xor_b32 s44, exec_lo, s1
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_cbranch_execnz .LBB0_32
; %bb.30:                               ;   in Loop: Header=BB0_17 Depth=1
	s_and_not1_saveexec_b32 s24, s44
	s_cbranch_execnz .LBB0_35
.LBB0_31:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s24
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s26
	s_cbranch_vccnz .LBB0_16
	s_branch .LBB0_36
.LBB0_32:                               ;   in Loop: Header=BB0_17 Depth=1
	s_mov_b32 s45, exec_lo
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_cmpx_eq_u32_e32 1, v11 /*v267*/
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_cbranch_execz .LBB0_34
; %bb.33:                               ;   in Loop: Header=BB0_17 Depth=1
	s_cmp_lg_u32 s31, 0
	s_cselect_b32 s1, s28, s35
	s_cmp_gt_i32 s22, 0
	s_cselect_b32 s24, -1, 0
	s_lshl_b32 s6, s43, 5
	s_delay_alu instid0(SALU_CYCLE_1)
	s_lshl_b64 s[2:3], s[6:7], 1
	s_mov_b32 s6, s7
	s_add_nc_u64 s[2:3], s[8:9], s[2:3]
	v_nop
	v_nop
	s_set_vgpr_msb 0x41                     ;  msbs: dst=1 src0=1 src1=0 src2=0
	v_dual_mov_b32 v15 /*v271*/, s1 :: v_dual_mov_b32 v30 /*v286*/, s2
	s_and_b32 s1, s3, 0x1ffffff
	s_and_b32 s3, s17, s24
	s_bitset1_b32 s1, 31
	v_cndmask_b32_e64 v4 /*v260*/, 0, 1, s3
	v_mov_b32_e32 v17 /*v273*/, s1
	s_mov_b32 s24, s22
	v_readfirstlane_b32 s49, v15 /*v271*/
	v_readfirstlane_b32 s50, v30 /*v286*/
	v_readfirstlane_b32 s48, v4 /*v260*/
	v_readfirstlane_b32 s51, v17 /*v273*/
	s_lshr_b64 s[2:3], s[24:25], 16
	s_lshl_b32 s1, s22, 16
	s_mov_b32 s3, s41
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[48:51], s[0:7]
	s_set_vgpr_msb 0x4100                   ;  msbs: dst=0 src0=0 src1=0 src2=0
.LBB0_34:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s45
	s_and_not1_saveexec_b32 s24, s44
	s_cbranch_execz .LBB0_31
.LBB0_35:                               ;   in Loop: Header=BB0_17 Depth=1
	s_cmp_lg_u32 s31, 0
	s_cselect_b32 s1, s37, 0
	s_cmp_gt_i32 s22, 0
	s_cselect_b32 s31, -1, 0
	s_lshl_b32 s6, s43, 5
	s_delay_alu instid0(SALU_CYCLE_1)
	s_lshl_b64 s[2:3], s[6:7], 1
	s_and_b32 s6, s21, s31
	s_add_nc_u64 s[2:3], s[12:13], s[2:3]
	s_set_vgpr_msb 0x41                     ;  msbs: dst=1 src0=1 src1=0 src2=0
	v_cndmask_b32_e64 v4 /*v260*/, 0, 1, s6
	s_and_b32 s3, s3, 0x1ffffff
	v_nop
	v_nop
	v_dual_mov_b32 v15 /*v271*/, s1 :: v_dual_mov_b32 v30 /*v286*/, s2
	s_bitset1_b32 s3, 31
	v_readfirstlane_b32 s44, v4 /*v260*/
	v_mov_b32_e32 v17 /*v273*/, s3
	s_delay_alu instid0(VALU_DEP_3)
	v_readfirstlane_b32 s45, v15 /*v271*/
	v_readfirstlane_b32 s46, v30 /*v286*/
	s_lshr_b64 s[2:3], s[22:23], 16
	s_lshl_b32 s1, s22, 16
	v_readfirstlane_b32 s47, v17 /*v273*/
	s_mov_b32 s3, s42
	s_mov_b32 s6, s7
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	tensor_load_to_lds s[44:47], s[0:7]
	s_or_b32 exec_lo, exec_lo, s24
	s_and_not1_b32 vcc_lo, exec_lo, s26
	s_set_vgpr_msb 0x4100                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_cbranch_vccnz .LBB0_16
.LBB0_36:                               ;   in Loop: Header=BB0_17 Depth=1
	s_cmp_lg_u32 s30, 0
	s_cselect_b32 s2, s37, 0
	s_cselect_b32 s1, s28, s35
	s_set_vgpr_msb 0x41                     ;  msbs: dst=1 src0=1 src1=0 src2=0
	v_lshl_add_u32 v4 /*v260*/, v2 /*v258*/, 1, s2
	v_lshl_add_u32 v15 /*v271*/, v8 /*v264*/, 1, s2
	v_lshl_add_u32 v17 /*v273*/, v10 /*v266*/, 1, s2
	v_lshl_add_u32 v19 /*v275*/, v24 /*v280*/, 1, s1
	ds_load_b128 v[30:33] /*v[286:289]*/, v4 /*v260*/
	ds_load_b128 v[34:37] /*v[290:293]*/, v4 /*v260*/ offset:16
	v_lshl_add_u32 v4 /*v260*/, v20 /*v276*/, 1, s2
	ds_load_b128 v[38:41] /*v[294:297]*/, v15 /*v271*/ offset:1024
	ds_load_b128 v[42:45] /*v[298:301]*/, v15 /*v271*/ offset:1040
	ds_load_b128 v[46:49] /*v[302:305]*/, v17 /*v273*/ offset:2048
	ds_load_b128 v[50:53] /*v[306:309]*/, v17 /*v273*/ offset:2064
	ds_load_b128 v[62:65] /*v[318:321]*/, v4 /*v260*/ offset:7168
	ds_load_b128 v[66:69] /*v[322:325]*/, v4 /*v260*/ offset:7184
	v_lshl_add_u32 v4 /*v260*/, v18 /*v274*/, 1, s2
	ds_load_b128 v[54:57] /*v[310:313]*/, v19 /*v275*/ offset:2048
	ds_load_b128 v[58:61] /*v[314:317]*/, v19 /*v275*/ offset:2064
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	ds_load_b128 v[70:73] /*v[326:329]*/, v4 /*v260*/ offset:6144
	ds_load_b128 v[74:77] /*v[330:333]*/, v4 /*v260*/ offset:6160
	v_lshl_add_u32 v4 /*v260*/, v16 /*v272*/, 1, s2
	s_set_vgpr_msb 0x4105                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[170:177], v[46:53] /*v[302:309]*/, v[54:61] /*v[310:317]*/, v[170:177] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[234:241], v[30:37] /*v[286:293]*/, v[54:61] /*v[310:317]*/, v[234:241] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[146:153], v[62:69] /*v[318:325]*/, v[54:61] /*v[310:317]*/, v[146:153] matrix_b_reuse
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[10:17], v[70:77] /*v[326:333]*/, v[54:61] /*v[310:317]*/, v[10:17] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[202:209], v[38:45] /*v[294:301]*/, v[54:61] /*v[310:317]*/, v[202:209] matrix_b_reuse
	s_set_vgpr_msb 0x541                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	ds_load_b128 v[78:81] /*v[334:337]*/, v4 /*v260*/ offset:5120
	ds_load_b128 v[82:85] /*v[338:341]*/, v4 /*v260*/ offset:5136
	v_lshl_add_u32 v4 /*v260*/, v14 /*v270*/, 1, s2
	ds_load_b128 v[86:89] /*v[342:345]*/, v4 /*v260*/ offset:4096
	ds_load_b128 v[90:93] /*v[346:349]*/, v4 /*v260*/ offset:4112
	v_lshl_add_u32 v4 /*v260*/, v12 /*v268*/, 1, s2
	ds_load_b128 v[94:97] /*v[350:353]*/, v4 /*v260*/ offset:3072
	ds_load_b128 v[98:101] /*v[354:357]*/, v4 /*v260*/ offset:3088
	v_lshl_add_u32 v4 /*v260*/, v22 /*v278*/, 1, s1
	s_set_vgpr_msb 0x4105                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[42:49], v[78:85] /*v[334:341]*/, v[54:61] /*v[310:317]*/, v[42:49] matrix_b_reuse
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[74:81], v[86:93] /*v[342:349]*/, v[54:61] /*v[310:317]*/, v[74:81] matrix_b_reuse
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[106:113], v[94:101] /*v[350:357]*/, v[54:61] /*v[310:317]*/, v[106:113] matrix_b_reuse
	s_set_vgpr_msb 0x541                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	ds_load_b128 v[102:105] /*v[358:361]*/, v4 /*v260*/ offset:1024
	ds_load_b128 v[106:109] /*v[362:365]*/, v4 /*v260*/ offset:1040
	v_lshl_add_u32 v4 /*v260*/, v6 /*v262*/, 1, s1
	ds_load_b128 v[54:57] /*v[310:313]*/, v4 /*v260*/
	ds_load_b128 v[58:61] /*v[314:317]*/, v4 /*v260*/ offset:16
	v_lshl_add_u32 v4 /*v260*/, v26 /*v282*/, 1, s1
	s_set_vgpr_msb 0x4105                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[90:97], v[78:85] /*v[334:341]*/, v[54:61] /*v[310:317]*/, v[90:97] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[58:65], v[70:77] /*v[326:333]*/, v[54:61] /*v[310:317]*/, v[58:65] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[34:41], v[62:69] /*v[318:325]*/, v[54:61] /*v[310:317]*/, v[34:41] matrix_b_reuse
	s_set_vgpr_msb 0x541                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	ds_load_b128 v[110:113] /*v[366:369]*/, v4 /*v260*/ offset:3072
	ds_load_b128 v[114:117] /*v[370:373]*/, v4 /*v260*/ offset:3088
	s_set_vgpr_msb 0x4105                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[194:201], v[38:45] /*v[294:301]*/, v[110:117] /*v[366:373]*/, v[194:201] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[226:233], v[30:37] /*v[286:293]*/, v[110:117] /*v[366:373]*/, v[226:233] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[154:161], v[62:69] /*v[318:325]*/, v[110:117] /*v[366:373]*/, v[154:161] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[2:9], v[70:77] /*v[326:333]*/, v[110:117] /*v[366:373]*/, v[2:9] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[66:73], v[86:93] /*v[342:349]*/, v[110:117] /*v[366:373]*/, v[66:73] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[98:105], v[94:101] /*v[350:357]*/, v[110:117] /*v[366:373]*/, v[98:105] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[122:129], v[46:53] /*v[302:309]*/, v[110:117] /*v[366:373]*/, v[122:129] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[18:25], v[70:77] /*v[326:333]*/, v[102:109] /*v[358:365]*/, v[18:25] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[138:145], v[62:69] /*v[318:325]*/, v[102:109] /*v[358:365]*/, v[138:145] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[250:257], v[30:37] /*v[286:293]*/, v[54:61] /*v[310:317]*/, v[250:257]
	v_wmma_f32_16x16x32_bf16 v[218:225], v[38:45] /*v[294:301]*/, v[54:61] /*v[310:317]*/, v[218:225] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[26:33], v[78:85] /*v[334:341]*/, v[110:117] /*v[366:373]*/, v[26:33] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[178:185], v[46:53] /*v[302:309]*/, v[102:109] /*v[358:365]*/, v[178:185] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[210:217], v[38:45] /*v[294:301]*/, v[102:109] /*v[358:365]*/, v[210:217] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[114:121], v[94:101] /*v[350:357]*/, v[102:109] /*v[358:365]*/, v[114:121] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[82:89], v[86:93] /*v[342:349]*/, v[102:109] /*v[358:365]*/, v[82:89] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[186:193], v[46:53] /*v[302:309]*/, v[54:61] /*v[310:317]*/, v[186:193] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[162:169], v[94:101] /*v[350:357]*/, v[54:61] /*v[310:317]*/, v[162:169] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[130:137], v[86:93] /*v[342:349]*/, v[54:61] /*v[310:317]*/, v[130:137] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[50:57], v[78:85] /*v[334:341]*/, v[102:109] /*v[358:365]*/, v[50:57] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[242:249], v[30:37] /*v[286:293]*/, v[102:109] /*v[358:365]*/, v[242:249] matrix_b_reuse
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
	s_and_b32 vcc_lo, exec_lo, s26
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_vccz .LBB0_39
; %bb.38:
	v_lshrrev_b32_e32 v1, 1, v0
	s_set_vgpr_msb 0x50                     ;  msbs: dst=1 src0=0 src1=0 src2=1
	v_and_or_b32 v2 /*v258*/, v0, 15, v7 /*v263*/
	s_set_vgpr_msb 0x5010                   ;  msbs: dst=0 src0=0 src1=0 src2=1
	v_cvt_pk_bf16_f32 v241, v240, v241
	v_cvt_pk_bf16_f32 v240, v238, v239
	v_cvt_pk_bf16_f32 v239, v236, v237
	v_and_or_b32 v1, v1, 8, v9 /*v265*/
	v_cvt_pk_bf16_f32 v249, v248, v249
	v_cvt_pk_bf16_f32 v248, v246, v247
	v_cvt_pk_bf16_f32 v246, v242, v243
	s_set_vgpr_msb 0x1045                   ;  msbs: dst=1 src0=1 src1=1 src2=0
	v_cvt_pk_bf16_f32 v1 /*v257*/, v0 /*v256*/, v1 /*v257*/
	s_set_vgpr_msb 0x4501                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_lshl_or_b32 v1, v2 /*v258*/, 8, v1
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_cvt_pk_bf16_f32 v0 /*v256*/, v254, v255
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_cvt_pk_bf16_f32 v255, v252, v253
	v_cvt_pk_bf16_f32 v254, v250, v251
	v_cvt_pk_bf16_f32 v247, v244, v245
	v_add_nc_u32_e32 v236, 0x1000, v1
	v_dual_lshrrev_b32 v237, 3, v1 :: v_dual_lshlrev_b32 v242, 1, v1
	v_cvt_pk_bf16_f32 v225, v224, v225
	v_cvt_pk_bf16_f32 v224, v222, v223
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshrrev_b32_e32 v236, 3, v236
	v_and_b32_e32 v237, 0x1ffe, v237
	v_cvt_pk_bf16_f32 v223, v220, v221
	v_add_nc_u32_e32 v221, 0x3010, v1
	v_cvt_pk_bf16_f32 v238, v234, v235
	v_and_b32_e32 v236, 0x3ffe, v236
	v_add_nc_u32_e32 v237, v237, v242
	v_add_nc_u32_e32 v234, 0x2000, v1
	v_add_nc_u32_e32 v235, 0x3000, v1
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_nc_u32 v243, 16, v1 :: v_dual_add_nc_u32 v236, v236, v242
	v_cvt_pk_bf16_f32 v217, v216, v217
	v_cvt_pk_bf16_f32 v216, v214, v215
	v_cvt_pk_bf16_f32 v215, v212, v213
	ds_store_b128 v237, v[254:257]
	ds_store_b128 v236, v[246:249] offset:8192
	v_add_nc_u32_e32 v237, 0x1010, v1
	v_lshrrev_b32_e32 v212, 3, v221
	v_add_nc_u32_e32 v213, 32, v1
	v_cvt_pk_bf16_f32 v193, v192, v193
	v_cvt_pk_bf16_f32 v192, v190, v191
	v_cvt_pk_bf16_f32 v191, v188, v189
	v_dual_add_nc_u32 v189, 48, v1 :: v_dual_lshrrev_b32 v234, 3, v234
	v_lshrrev_b32_e32 v235, 3, v235
	v_cvt_pk_bf16_f32 v233, v232, v233
	v_cvt_pk_bf16_f32 v232, v230, v231
	v_cvt_pk_bf16_f32 v231, v228, v229
	v_dual_lshrrev_b32 v229, 3, v237 :: v_dual_lshrrev_b32 v244, 3, v243
	v_cvt_pk_bf16_f32 v214, v210, v211
	v_and_b32_e32 v211, 0x3ff0, v212
	v_lshrrev_b32_e32 v212, 3, v213
	v_cvt_pk_bf16_f32 v185, v184, v185
	v_cvt_pk_bf16_f32 v184, v182, v183
	v_cvt_pk_bf16_f32 v182, v178, v179
	v_lshrrev_b32_e32 v178, 3, v189
	v_and_b32_e32 v234, 0x3ffe, v234
	v_and_b32_e32 v235, 0x3ffe, v235
	v_cvt_pk_bf16_f32 v222, v218, v219
	v_and_b32_e32 v218, 0x3ff0, v229
	v_lshlrev_b32_e32 v220, 1, v243
	v_and_b32_e32 v236, 0x1bf0, v244
	v_cvt_pk_bf16_f32 v209, v208, v209
	v_cvt_pk_bf16_f32 v208, v206, v207
	v_cvt_pk_bf16_f32 v206, v202, v203
	v_and_b32_e32 v203, 0x1bf0, v212
	v_and_b32_e32 v178, 0x1bf0, v178
	v_dual_add_nc_u32 v234, v234, v242 :: v_dual_add_nc_u32 v235, v235, v242
	v_cvt_pk_bf16_f32 v230, v226, v227
	v_dual_add_nc_u32 v218, v218, v220 :: v_dual_add_nc_u32 v228, v236, v242
	v_cvt_pk_bf16_f32 v201, v200, v201
	v_cvt_pk_bf16_f32 v200, v198, v199
	v_cvt_pk_bf16_f32 v199, v196, v197
	v_add_nc_u32_e32 v196, v203, v242
	v_cvt_pk_bf16_f32 v190, v186, v187
	v_cvt_pk_bf16_f32 v177, v176, v177
	v_cvt_pk_bf16_f32 v176, v174, v175
	v_cvt_pk_bf16_f32 v174, v170, v171
	v_add_nc_u32_e32 v171, v178, v242
	v_cvt_pk_bf16_f32 v169, v168, v169
	v_cvt_pk_bf16_f32 v168, v166, v167
	v_cvt_pk_bf16_f32 v167, v164, v165
	v_cvt_pk_bf16_f32 v166, v162, v163
	v_add_nc_u32_e32 v219, 0x2010, v1
	v_cvt_pk_bf16_f32 v175, v172, v173
	v_add_nc_u32_e32 v172, 64, v1
	v_add_nc_u32_e32 v164, 0x50, v1
	v_cvt_pk_bf16_f32 v207, v204, v205
	v_add_nc_u32_e32 v204, 0x1020, v1
	ds_store_b128 v234, v[238:241] offset:16384
	ds_store_b128 v228, v[222:225] offset:32
	ds_store_b128 v235, v[230:233] offset:24576
	ds_store_b128 v196, v[190:193] offset:64
	ds_store_b128 v218, v[214:217] offset:8192
	ds_store_b128 v171, v[166:169] offset:96
	v_add_nc_u32_e32 v167, 0x60, v1
	v_dual_lshrrev_b32 v219, 3, v219 :: v_dual_lshrrev_b32 v173, 3, v172
	v_cvt_pk_bf16_f32 v137, v136, v137
	v_lshrrev_b32_e32 v166, 3, v164
	v_cvt_pk_bf16_f32 v136, v134, v135
	v_cvt_pk_bf16_f32 v135, v132, v133
	v_add_nc_u32_e32 v133, 0x70, v1
	v_dual_lshrrev_b32 v197, 3, v204 :: v_dual_lshrrev_b32 v132, 3, v167
	v_and_b32_e32 v219, 0x3ff0, v219
	v_add_nc_u32_e32 v188, 0x2020, v1
	v_cvt_pk_bf16_f32 v183, v180, v181
	v_add_nc_u32_e32 v180, 0x3020, v1
	v_and_b32_e32 v163, 0x1bf0, v173
	v_add_nc_u32_e32 v165, 0x1030, v1
	v_cvt_pk_bf16_f32 v134, v130, v131
	v_and_b32_e32 v130, 0x1bf0, v166
	v_cvt_pk_bf16_f32 v97, v96, v97
	v_cvt_pk_bf16_f32 v96, v94, v95
	v_cvt_pk_bf16_f32 v94, v90, v91
	v_lshrrev_b32_e32 v91, 3, v133
	v_and_b32_e32 v186, 0x3ff0, v197
	v_lshlrev_b32_e32 v187, 1, v213
	v_and_b32_e32 v132, 0x1bf0, v132
	v_cvt_pk_bf16_f32 v41, v40, v41
	v_cvt_pk_bf16_f32 v40, v38, v39
	v_cvt_pk_bf16_f32 v38, v34, v35
	v_add_nc_u32_e32 v34, 0x2030, v1
	v_add_nc_u32_e32 v210, v219, v220
	v_add_nc_u32_e32 v202, v211, v220
	v_cvt_pk_bf16_f32 v198, v194, v195
	v_lshrrev_b32_e32 v188, 3, v188
	v_dual_lshrrev_b32 v170, 3, v180 :: v_dual_add_nc_u32 v163, v163, v242
	v_dual_lshrrev_b32 v131, 3, v165 :: v_dual_add_nc_u32 v130, v130, v242
	v_cvt_pk_bf16_f32 v95, v92, v93
	v_cvt_pk_bf16_f32 v65, v64, v65
	v_cvt_pk_bf16_f32 v64, v62, v63
	v_cvt_pk_bf16_f32 v62, v58, v59
	v_and_b32_e32 v58, 0x1bf0, v91
	v_dual_add_nc_u32 v186, v186, v187 :: v_dual_add_nc_u32 v90, v132, v242
	v_cvt_pk_bf16_f32 v63, v60, v61
	v_lshrrev_b32_e32 v59, 3, v34
	v_and_b32_e32 v188, 0x3ff0, v188
	v_and_b32_e32 v162, 0x3ff0, v170
	ds_store_b128 v210, v[206:209] offset:16384
	ds_store_b128 v163, v[134:137] offset:128
	ds_store_b128 v202, v[198:201] offset:24576
	ds_store_b128 v130, v[94:97] offset:160
	ds_store_b128 v186, v[182:185] offset:8192
	ds_store_b128 v90, v[62:65] offset:192
	v_add_nc_u32_e32 v63, 0x3030, v1
	v_add_nc_u32_e32 v90, v58, v242
	v_and_b32_e32 v58, 0x3ff0, v131
	v_lshlrev_b32_e32 v92, 1, v189
	v_and_b32_e32 v62, 0x3ff0, v59
	v_add_nc_u32_e32 v179, v188, v187
	v_cvt_pk_bf16_f32 v39, v36, v37
	v_add_nc_u32_e32 v91, v162, v187
	v_cvt_pk_bf16_f32 v37, v128, v129
	v_cvt_pk_bf16_f32 v36, v126, v127
	v_cvt_pk_bf16_f32 v35, v124, v125
	v_cvt_pk_bf16_f32 v34, v122, v123
	v_lshrrev_b32_e32 v95, 3, v63
	v_add_nc_u32_e32 v96, 0x1040, v1
	v_add_nc_u32_e32 v93, v58, v92
	v_cvt_pk_bf16_f32 v61, v120, v121
	v_cvt_pk_bf16_f32 v60, v118, v119
	v_cvt_pk_bf16_f32 v59, v116, v117
	v_cvt_pk_bf16_f32 v58, v114, v115
	v_add_nc_u32_e32 v94, v62, v92
	v_cvt_pk_bf16_f32 v65, v112, v113
	v_cvt_pk_bf16_f32 v64, v110, v111
	v_cvt_pk_bf16_f32 v63, v108, v109
	v_cvt_pk_bf16_f32 v62, v106, v107
	ds_store_b128 v179, v[174:177] offset:16384
	v_and_b32_e32 v95, 0x3ff0, v95
	ds_store_b128 v91, v[34:37] offset:24576
	ds_store_b128 v93, v[58:61] offset:8192
	v_lshrrev_b32_e32 v34, 3, v96
	v_add_nc_u32_e32 v35, 0x2040, v1
	ds_store_b128 v94, v[62:65] offset:16384
	v_add_nc_u32_e32 v63, 0x3040, v1
	v_add_nc_u32_e32 v91, v95, v92
	v_and_b32_e32 v58, 0x3ff0, v34
	v_dual_lshlrev_b32 v92, 1, v172 :: v_dual_lshrrev_b32 v59, 3, v35
	v_cvt_pk_bf16_f32 v37, v104, v105
	v_cvt_pk_bf16_f32 v36, v102, v103
	v_cvt_pk_bf16_f32 v35, v100, v101
	v_cvt_pk_bf16_f32 v34, v98, v99
	v_cvt_pk_bf16_f32 v65, v80, v81
	v_lshrrev_b32_e32 v80, 3, v63
	v_cvt_pk_bf16_f32 v63, v76, v77
	v_add_nc_u32_e32 v76, 0x1050, v1
	v_add_nc_u32_e32 v93, v58, v92
	v_cvt_pk_bf16_f32 v61, v88, v89
	v_and_b32_e32 v62, 0x3ff0, v59
	v_cvt_pk_bf16_f32 v60, v86, v87
	v_cvt_pk_bf16_f32 v59, v84, v85
	v_cvt_pk_bf16_f32 v58, v82, v83
	ds_store_b128 v91, v[34:37] offset:24576
	ds_store_b128 v93, v[58:61] offset:8192
	v_lshrrev_b32_e32 v34, 3, v76
	v_add_nc_u32_e32 v82, v62, v92
	v_cvt_pk_bf16_f32 v62, v74, v75
	v_and_b32_e32 v74, 0x3ff0, v80
	v_add_nc_u32_e32 v35, 0x2050, v1
	v_and_b32_e32 v59, 0x3ff0, v34
	v_lshlrev_b32_e32 v60, 1, v164
	v_cvt_pk_bf16_f32 v64, v78, v79
	v_add_nc_u32_e32 v58, v74, v92
	v_cvt_pk_bf16_f32 v37, v72, v73
	v_cvt_pk_bf16_f32 v36, v70, v71
	v_lshrrev_b32_e32 v61, 3, v35
	v_cvt_pk_bf16_f32 v35, v68, v69
	v_cvt_pk_bf16_f32 v34, v66, v67
	v_add_nc_u32_e32 v59, v59, v60
	v_cvt_pk_bf16_f32 v57, v56, v57
	v_cvt_pk_bf16_f32 v56, v54, v55
	v_cvt_pk_bf16_f32 v55, v52, v53
	v_cvt_pk_bf16_f32 v54, v50, v51
	ds_store_b128 v82, v[62:65] offset:16384
	v_add_nc_u32_e32 v62, 0x3050, v1
	v_cvt_pk_bf16_f32 v49, v48, v49
	v_cvt_pk_bf16_f32 v48, v46, v47
	v_cvt_pk_bf16_f32 v47, v44, v45
	v_add_nc_u32_e32 v44, 0x1060, v1
	ds_store_b128 v58, v[34:37] offset:24576
	ds_store_b128 v59, v[54:57] offset:8192
	v_add_nc_u32_e32 v36, 0x2060, v1
	v_lshrrev_b32_e32 v51, 3, v62
	v_cvt_pk_bf16_f32 v33, v32, v33
	v_cvt_pk_bf16_f32 v32, v30, v31
	v_cvt_pk_bf16_f32 v31, v28, v29
	v_add_nc_u32_e32 v28, 0x3060, v1
	v_lshrrev_b32_e32 v34, 3, v44
	v_lshrrev_b32_e32 v36, 3, v36
	v_and_b32_e32 v61, 0x3ff0, v61
	v_cvt_pk_bf16_f32 v46, v42, v43
	v_and_b32_e32 v42, 0x3ff0, v51
	v_cvt_pk_bf16_f32 v25, v24, v25
	v_cvt_pk_bf16_f32 v24, v22, v23
	v_cvt_pk_bf16_f32 v22, v18, v19
	v_lshrrev_b32_e32 v19, 3, v28
	v_and_b32_e32 v34, 0x3ff0, v34
	v_lshlrev_b32_e32 v37, 1, v167
	v_cvt_pk_bf16_f32 v30, v26, v27
	v_and_b32_e32 v27, 0x3ff0, v36
	v_dual_add_nc_u32 v50, v61, v60 :: v_dual_add_nc_u32 v35, v42, v60
	v_cvt_pk_bf16_f32 v17, v16, v17
	v_cvt_pk_bf16_f32 v16, v14, v15
	v_cvt_pk_bf16_f32 v14, v10, v11
	v_add_nc_u32_e32 v10, 0x1070, v1
	v_and_b32_e32 v11, 0x3ff0, v19
	v_add_nc_u32_e32 v26, v34, v37
	v_cvt_pk_bf16_f32 v23, v20, v21
	v_add_nc_u32_e32 v18, v27, v37
	v_cvt_pk_bf16_f32 v15, v12, v13
	ds_store_b128 v50, v[46:49] offset:16384
	ds_store_b128 v35, v[30:33] offset:24576
	ds_store_b128 v26, v[22:25] offset:8192
	ds_store_b128 v18, v[14:17] offset:16384
	v_dual_lshrrev_b32 v10, 3, v10 :: v_dual_add_nc_u32 v18, v11, v37
	v_add_nc_u32_e32 v11, 0x2070, v1
	v_add_nc_u32_e32 v1, 0x3070, v1
	v_cvt_pk_bf16_f32 v9, v8, v9
	v_cvt_pk_bf16_f32 v8, v6, v7
	v_and_b32_e32 v10, 0x3ff0, v10
	v_lshlrev_b32_e32 v14, 1, v133
	v_cvt_pk_bf16_f32 v7, v4, v5
	v_dual_lshrrev_b32 v4, 3, v11 :: v_dual_lshrrev_b32 v1, 3, v1
	v_cvt_pk_bf16_f32 v6, v2, v3
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_nc_u32_e32 v19, v10, v14
	v_cvt_pk_bf16_f32 v5, v144, v145
	v_and_b32_e32 v10, 0x3ff0, v4
	v_and_b32_e32 v1, 0x3ff0, v1
	v_cvt_pk_bf16_f32 v4, v142, v143
	v_cvt_pk_bf16_f32 v3, v140, v141
	v_cvt_pk_bf16_f32 v2, v138, v139
	v_add_nc_u32_e32 v20, v10, v14
	v_cvt_pk_bf16_f32 v13, v152, v153
	v_cvt_pk_bf16_f32 v12, v150, v151
	v_cvt_pk_bf16_f32 v11, v148, v149
	v_cvt_pk_bf16_f32 v10, v146, v147
	v_add_nc_u32_e32 v1, v1, v14
	v_cvt_pk_bf16_f32 v17, v160, v161
	v_cvt_pk_bf16_f32 v16, v158, v159
	v_cvt_pk_bf16_f32 v15, v156, v157
	v_cvt_pk_bf16_f32 v14, v154, v155
	ds_store_b128 v18, v[6:9] offset:24576
	ds_store_b128 v90, v[38:41] offset:224
	ds_store_b128 v19, v[2:5] offset:8192
	ds_store_b128 v20, v[10:13] offset:16384
	ds_store_b128 v1, v[14:17] offset:24576
.LBB0_39:
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_cmp_ne_u32_e32 vcc_lo, 1, v3 /*v259*/
	;;#ASMSTART
	s_wait_dscnt 0
	;;#ASMEND
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_vccnz .LBB0_50
; %bb.40:
	s_mul_i32 s3, s25, s23
	s_mov_b32 s0, exec_lo
	v_cmpx_gt_i32_e64 s3, v0
	s_cbranch_execz .LBB0_50
; %bb.41:
	s_ashr_i32 s21, s20, 31
	v_xad_u32 v2, v0, -1, s3
	s_lshl_b64 s[0:1], s[20:21], 1
	s_ashr_i32 s17, s16, 31
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[4:5], s[10:11], s[0:1]
	s_mov_b32 s0, 0
                                        ; implicit-def: $vgpr1
                                        ; implicit-def: $vgpr6
                                        ; implicit-def: $sgpr12_sgpr13
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
	s_abs_i32 s15, s19
	v_lshrrev_b32_e32 v1, 8, v2
	s_cvt_f32_u32 s0, s15
	v_or_b32_e32 v3, 0x300, v0
	s_sub_co_i32 s1, 0, s15
	v_mov_b32_e32 v7, 0
	v_rcp_iflag_f32_e32 v2, s0
	v_add_nc_u32_e32 v8, 1, v1
	v_or_b32_e32 v1, 0x100, v0
	s_mov_b32 s13, 0
	s_mov_b32 s6, s16
	s_mov_b32 s7, s17
	v_and_b32_e32 v9, 0x1fffffc, v8
	s_mov_b32 s8, s16
	v_readfirstlane_b32 s0, v2
	v_or_b32_e32 v2, 0x200, v0
	s_mov_b32 s9, s17
	v_mov_b32_e32 v10, v9
	s_mov_b32 s10, s16
	s_mul_f32 s0, s0, 0x4f7ffffe
	v_mov_b64_e32 v[4:5], v[2:3]
	v_mov_b64_e32 v[2:3], v[0:1]
	s_mov_b32 s11, s17
	s_cvt_u32_f32 s0, s0
	s_mov_b32 s18, s19
	s_mov_b32 s20, s19
	s_mov_b32 s21, s19
	s_mul_i32 s1, s1, s0
	s_mov_b32 s22, s33
	s_mul_hi_u32 s1, s0, s1
	s_mov_b32 s23, s33
	s_mov_b32 s24, s33
	s_ashr_i32 s25, s19, 31
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
	v_xor_b32_e32 v1, s25, v1
	v_xor_b32_e32 v11, s25, v11
	v_mul_lo_u32 v24, v20, s15
	v_mul_lo_u32 v26, v21, s15
	v_mul_lo_u32 v27, v22, s15
	v_dual_add_nc_u32 v25, 1, v20 :: v_dual_add_nc_u32 v29, 1, v21
	v_mul_lo_u32 v28, v23, s15
	v_dual_add_nc_u32 v30, 1, v22 :: v_dual_add_nc_u32 v31, 1, v23
	v_dual_sub_nc_u32 v6, v6, v24 :: v_dual_sub_nc_u32 v12, v12, v26
	v_dual_sub_nc_u32 v16, v16, v27 :: v_dual_bitop2_b32 v14, s25, v14 bitop3:0x14
	v_xor_b32_e32 v18, s25, v18
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
	v_mul_lo_u32 v12, v11, s18
	v_dual_sub_nc_u32 v26, v16, v14 :: v_dual_bitop2_b32 v6, v6, v1 bitop3:0x14
	v_cndmask_b32_e32 v19, v23, v27, vcc_lo
	v_add_nc_u32_e32 v20, s22, v11
	v_cmp_eq_u32_e32 vcc_lo, 0, v10
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mul_lo_u32 v14, v26, s20
	v_dual_sub_nc_u32 v1, v6, v1 :: v_dual_bitop2_b32 v19, v19, v18 bitop3:0x14
	v_dual_add_nc_u32 v22, s23, v26 :: v_dual_sub_nc_u32 v12, v3, v12
	v_ashrrev_i32_e32 v21, 31, v20
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mul_lo_u32 v6, v1, s19
	v_dual_sub_nc_u32 v27, v19, v18 :: v_dual_add_nc_u32 v18, s33, v1
	v_sub_nc_u32_e32 v14, v4, v14
	v_ashrrev_i32_e32 v23, 31, v22
	v_lshl_add_u32 v11, v11, 8, v12
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_mul_lo_u32 v16, v27, s21
	v_dual_add_nc_u32 v24, s24, v27 :: v_dual_sub_nc_u32 v6, v2, v6
	v_ashrrev_i32_e32 v19, 31, v18
	v_lshl_add_u32 v26, v26, 8, v14
	v_mul_u64_e32 v[20:21], s[6:7], v[20:21]
	v_ashrrev_i32_e32 v25, 31, v24
	v_lshl_add_u32 v1, v1, 8, v6
	v_sub_nc_u32_e32 v16, v5, v16
	v_mul_u64_e32 v[18:19], s[16:17], v[18:19]
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
	s_or_b32 s26, vcc_lo, s26
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
	s_and_not1_b32 exec_lo, exec_lo, s26
	s_cbranch_execnz .LBB0_45
; %bb.46:
	s_or_b32 exec_lo, exec_lo, s26
	v_cmp_ne_u32_e32 vcc_lo, v8, v9
	v_lshl_or_b32 v0, v9, 8, v0
	v_dual_mov_b32 v6, s15 :: v_dual_mov_b32 v1, s25
	s_and_b32 s0, vcc_lo, exec_lo
	s_or_saveexec_b32 s1, s14
	v_mov_b64_e32 v[2:3], s[12:13]
	s_xor_b32 exec_lo, exec_lo, s1
	s_cbranch_execz .LBB0_43
.LBB0_47:
	s_abs_i32 s2, s19
	s_ashr_i32 s8, s19, 31
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
	s_sub_co_i32 s1, 0, s19
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
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_sub_nc_u32 v7, v4, v9 :: v_dual_lshlrev_b32 v4, 8, v4
	v_lshlrev_b32_e32 v9, 8, v9
	v_mul_lo_u32 v8, v7, s19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_sub_nc_u32 v4, v4, v8 :: v_dual_add_nc_u32 v8, s33, v7
	v_dual_sub_nc_u32 v4, v4, v9 :: v_dual_ashrrev_i32 v9, 31, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_nc_u32_e32 v4, v0, v4
	v_mul_u64_e32 v[8:9], s[16:17], v[8:9]
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
	.size	bm256_bn256_bk032_wm2_wn4_mc0, .Lfunc_end0-bm256_bn256_bk032_wm2_wn4_mc0
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel bm256_bn256_bk032_wm2_wn4_mc0
		.amdhsa_group_segment_fixed_size 139264
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
		.amdhsa_next_free_vgpr 374
		.amdhsa_next_free_sgpr 52
		.amdhsa_named_barrier_count 0
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size 73
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text.bm256_bn256_bk032_wm2_wn4_mc0,"axG",@progbits,bm256_bn256_bk032_wm2_wn4_mc0,comdat
                                        ; -- End function
	.set .Lbm256_bn256_bk032_wm2_wn4_mc0.num_vgpr, 374
	.set .Lbm256_bn256_bk032_wm2_wn4_mc0.num_agpr, 0
	.set .Lbm256_bn256_bk032_wm2_wn4_mc0.numbered_sgpr, 52
	.set .Lbm256_bn256_bk032_wm2_wn4_mc0.num_named_barrier, 0
	.set .Lbm256_bn256_bk032_wm2_wn4_mc0.private_seg_size, 0
	.set .Lbm256_bn256_bk032_wm2_wn4_mc0.uses_vcc, 1
	.set .Lbm256_bn256_bk032_wm2_wn4_mc0.uses_flat_scratch, 1
	.set .Lbm256_bn256_bk032_wm2_wn4_mc0.has_dyn_sized_stack, 0
	.set .Lbm256_bn256_bk032_wm2_wn4_mc0.has_recursion, 0
	.set .Lbm256_bn256_bk032_wm2_wn4_mc0.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 9320
; TotalNumSgprs: 54
; NumVgprs: 374
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 139264 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 23
; NumSGPRsForWavesPerEU: 54
; NumVGPRsForWavesPerEU: 374
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
	.type	__hip_cuid_41071ac3e10689e0,@object ; @__hip_cuid_41071ac3e10689e0
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_41071ac3e10689e0
__hip_cuid_41071ac3e10689e0:
	.byte	0                               ; 0x0
	.size	__hip_cuid_41071ac3e10689e0, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git aa451e1fe6a793394d6733051b1778633063ae96+PATCHED:d17c5aa0e3ea29cde402f58f27e39b6034effa27)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_41071ac3e10689e0
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
    .gfx1250_revision: B0
    .group_segment_fixed_size: 139264
    .kernarg_segment_align: 8
    .kernarg_segment_size: 132
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           bm256_bn256_bk032_wm2_wn4_mc0
    .private_segment_fixed_size: 0
    .sgpr_count:     54
    .sgpr_spill_count: 0
    .symbol:         bm256_bn256_bk032_wm2_wn4_mc0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     374
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
