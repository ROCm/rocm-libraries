	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.section	.text.bm256_bn256_bk128_wm2_wn4_mc0,"axG",@progbits,bm256_bn256_bk128_wm2_wn4_mc0,comdat
	.protected	bm256_bn256_bk128_wm2_wn4_mc0 ; -- Begin function bm256_bn256_bk128_wm2_wn4_mc0
	.globl	bm256_bn256_bk128_wm2_wn4_mc0
	.p2align	8
	.type	bm256_bn256_bk128_wm2_wn4_mc0,@function
bm256_bn256_bk128_wm2_wn4_mc0: ; @bm256_bn256_bk128_wm2_wn4_mc0
; %bb.0:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1 ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_mov_b64 s[2:3], src_shared_base
	s_mov_b32 s2, 0x11000
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
	s_cselect_b32 s37, ttmp9, s4
	s_cselect_b32 s3, ttmp7, s5
	s_wait_kmcnt 0x0
	s_add_co_i32 s4, s16, 0xff
	s_add_co_i32 s6, s17, 0xff
	s_ashr_i32 s5, s4, 31
	s_lshl_b32 s20, s37, 8
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
	s_cmp_lt_i32 s37, s4
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
	s_cselect_b32 s34, -1, 0
	s_and_b32 s8, s34, exec_lo
	s_cselect_b32 s25, s7, 0
	s_add_co_i32 s28, s18, 0x7f
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 2, 1), 1
	s_min_i32 s7, s18, 0x80
	s_cmp_gt_i32 s28, 0x7f
	s_cselect_b32 s36, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_and_b32 s8, s36, exec_lo
	s_cselect_b32 s22, s7, 0
	s_cmp_lt_i32 s23, 0x100
	s_cselect_b32 s35, -1, 0
	s_and_b32 vcc_lo, exec_lo, s35
	s_mov_b32 s7, s35
	s_cbranch_vccnz .LBB0_2
; %bb.1:
	s_cmp_lt_i32 s25, 0x100
	s_cselect_b32 s7, -1, 0
	s_cmp_lt_i32 s22, 0x80
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
	v_cmp_lt_u32_e32 vcc_lo, 0x42ff, v5
	s_or_b32 s7, vcc_lo, s7
	s_and_not1_b32 exec_lo, exec_lo, s7
	s_cbranch_execnz .LBB0_4
; %bb.5:
	s_or_b32 exec_lo, exec_lo, s7
	v_lshl_add_u32 v2, s2, 2, v2
	v_mov_b32_e32 v3, 0
	s_mov_b32 s7, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_or_b32_e32 v2, 0x11000, v2
.LBB0_6:                                ; =>This Inner Loop Header: Depth=1
	v_add_nc_u32_e32 v1, 0x100, v1
	ds_store_b32 v2, v3
	v_add_nc_u32_e32 v2, 0x400, v2
	v_cmp_lt_u32_e32 vcc_lo, 0x42ff, v1
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
	s_add_co_i32 s6, s6, -1
	s_mov_b64 s[26:27], src_shared_base
	s_lshl2_add_u32 s26, s2, 0x11000
	s_add_co_i32 s17, s4, -1
	s_min_i32 s29, s3, s6
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
	s_lshl_b32 s0, s29, 8
	s_wait_kmcnt 0x0
	s_bfe_i64 s[2:3], s[8:9], 0x200000
	s_ashr_i32 s1, s0, 31
	s_mov_b32 s6, 0
	s_mul_u64 s[0:1], s[2:3], s[0:1]
	s_mov_b32 s7, s6
	s_lshl_b64 s[0:1], s[0:1], 1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_nc_u64 s[2:3], s[14:15], s[0:1]
	v_dual_mov_b32 v1, s26 :: v_dual_mov_b32 v4, s2
	s_and_b32 s0, s3, 0x1ffffff
	s_and_b32 s3, s34, s4
	s_bitset1_b32 s0, 31
	v_cndmask_b32_e64 v2, 0, 1, s3
	v_mov_b32_e32 v3, s0
	v_readfirstlane_b32 s41, v1
	v_readfirstlane_b32 s42, v4
	s_lshr_b32 s0, s25, 16
	v_readfirstlane_b32 s40, v2
	v_readfirstlane_b32 s43, v3
	s_lshr_b64 s[2:3], s[24:25], 16
	s_lshl_b32 s1, s22, 16
	s_or_b32 s3, s0, 0x800000
	s_movk_i32 s4, 0x100
	s_mov_b32 s0, 0x7510000
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[40:43], s[0:7]
.LBB0_11:
	s_or_b32 exec_lo, exec_lo, s39
.LBB0_12:
	s_or_saveexec_b32 s38, s38
	s_min_i32 s24, s37, s17
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
	v_readfirstlane_b32 s40, v2
	s_bitset1_b32 s3, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v4, s2 :: v_dual_mov_b32 v3, s3
	s_lshr_b64 s[2:3], s[22:23], 16
	s_or_b32 s3, s0, 0x800000
	s_movk_i32 s4, 0x100
	v_readfirstlane_b32 s42, v4
	v_readfirstlane_b32 s43, v3
	s_mov_b32 s0, 0x7510000
	s_mov_b32 s7, s6
	s_mov_b32 s41, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[40:43], s[0:7]
.LBB0_14:
	s_or_b32 exec_lo, exec_lo, s38
	s_wait_tensorcnt 0x0
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	s_barrier_signal -1
	v_dual_lshlrev_b32 v1, 6, v11 /*v267*/ :: v_dual_mov_b32 v9, 0
	s_and_b32 s17, s21, s34
	s_set_vgpr_msb 0x440                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_and_b32_e32 v7 /*v263*/, 0x80, v0
	v_cndmask_b32_e64 v3 /*v259*/, 0, 1, s17
	v_and_b32_e32 v9 /*v265*/, 0xc0, v1
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_dual_mov_b32 v8, v9 :: v_dual_mov_b32 v7, v9
	v_dual_mov_b32 v6, v9 :: v_dual_mov_b32 v5, v9
	v_dual_mov_b32 v4, v9 :: v_dual_mov_b32 v3, v9
	v_dual_mov_b32 v2, v9 :: v_dual_mov_b32 v17, v9
	v_dual_mov_b32 v16, v9 :: v_dual_mov_b32 v15, v9
	v_dual_mov_b32 v14, v9 :: v_dual_mov_b32 v13, v9
	v_dual_mov_b32 v12, v9 :: v_dual_mov_b32 v11, v9
	v_dual_mov_b32 v10, v9 :: v_dual_mov_b32 v57, v9
	v_dual_mov_b32 v56, v9 :: v_dual_mov_b32 v55, v9
	v_dual_mov_b32 v54, v9 :: v_dual_mov_b32 v53, v9
	v_dual_mov_b32 v52, v9 :: v_dual_mov_b32 v51, v9
	v_dual_mov_b32 v50, v9 :: v_dual_mov_b32 v25, v9
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
	v_dual_mov_b32 v34, v9 :: v_dual_mov_b32 v81, v9
	v_dual_mov_b32 v80, v9 :: v_dual_mov_b32 v79, v9
	v_dual_mov_b32 v78, v9 :: v_dual_mov_b32 v77, v9
	v_dual_mov_b32 v76, v9 :: v_dual_mov_b32 v75, v9
	v_dual_mov_b32 v74, v9 :: v_dual_mov_b32 v49, v9
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
	v_dual_mov_b32 v66, v9 :: v_dual_mov_b32 v113, v9
	v_dual_mov_b32 v112, v9 :: v_dual_mov_b32 v111, v9
	v_dual_mov_b32 v110, v9 :: v_dual_mov_b32 v109, v9
	v_dual_mov_b32 v108, v9 :: v_dual_mov_b32 v107, v9
	v_dual_mov_b32 v106, v9 :: v_dual_mov_b32 v89, v9
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
	v_dual_mov_b32 v98, v9 :: v_dual_mov_b32 v153, v9
	v_dual_mov_b32 v152, v9 :: v_dual_mov_b32 v151, v9
	v_dual_mov_b32 v150, v9 :: v_dual_mov_b32 v149, v9
	v_dual_mov_b32 v148, v9 :: v_dual_mov_b32 v147, v9
	v_dual_mov_b32 v146, v9 :: v_dual_mov_b32 v121, v9
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
	v_dual_mov_b32 v130, v9 :: v_dual_mov_b32 v169, v9
	v_dual_mov_b32 v168, v9 :: v_dual_mov_b32 v167, v9
	v_dual_mov_b32 v166, v9 :: v_dual_mov_b32 v165, v9
	v_dual_mov_b32 v164, v9 :: v_dual_mov_b32 v163, v9
	v_dual_mov_b32 v162, v9 :: v_dual_mov_b32 v145, v9
	v_dual_mov_b32 v144, v9 :: v_dual_mov_b32 v143, v9
	v_dual_mov_b32 v142, v9 :: v_dual_mov_b32 v141, v9
	v_dual_mov_b32 v140, v9 :: v_dual_mov_b32 v139, v9
	v_dual_mov_b32 v138, v9 :: v_dual_mov_b32 v177, v9
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
	v_dual_mov_b32 v250, v9 :: v_dual_mov_b32 v161, v9
	v_dual_mov_b32 v160, v9 :: v_dual_mov_b32 v159, v9
	v_dual_mov_b32 v158, v9 :: v_dual_mov_b32 v157, v9
	v_dual_mov_b32 v156, v9 :: v_dual_mov_b32 v155, v9
	v_mov_b32_e32 v154, v9
	s_and_not1_b32 vcc_lo, exec_lo, s36
	s_barrier_wait -1
	s_cbranch_vccnz .LBB0_37
; %bb.15:
	s_mov_b64 s[0:1], src_shared_base
	s_add_co_i32 s2, s26, 0x11000
	s_mov_b32 s3, s1
	v_dual_lshlrev_b32 v4, 7, v0 :: v_dual_bitop2_b32 v1, 16, v0 bitop3:0x40
	s_and_b64 s[36:37], s[2:3], 15
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_dual_lshlrev_b32 v2, 7, v9 /*v265*/ :: v_dual_lshlrev_b32 v3, 7, v7 /*v263*/
	s_sub_co_i32 s0, 16, s36
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_and_or_b32 v4, 0x780, v4, v1
	s_lshr_b32 s0, s0, 2
	s_cmp_lg_u64 s[36:37], 0
	s_mov_b32 s7, 0
	s_cselect_b32 s0, s0, 0
	v_or_b32_e32 v5, v2, v4
	s_lshl2_add_u32 s37, s0, s2
	v_or_b32_e32 v1, v4, v3
	s_add_co_i32 s0, s37, 0x11000
	s_mov_b32 s38, s1
	s_and_b32 s6, s0, 15
	v_lshrrev_b32_e32 v7, 4, v5
	s_sub_co_i32 s2, 16, s6
	v_lshrrev_b32_e32 v6, 4, v1
	s_lshr_b32 s2, s2, 2
	s_cmp_lg_u64 s[6:7], 0
	v_and_b32_e32 v7, 0x678, v7
	s_cselect_b32 s2, s2, 0
	s_ashr_i32 s3, s28, 31
	v_and_b32_e32 v6, 0x478, v6
	s_lshr_b32 s3, s3, 25
	s_lshl_b32 s6, s2, 2
	s_add_co_i32 s28, s28, s3
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_dual_mov_b32 v5 /*v261*/, 0 :: v_dual_add_nc_u32 v2 /*v258*/, v6, v1
	s_ashr_i32 s39, s28, 7
	s_cmp_lt_i32 s25, 0x100
	v_add_nc_u32_e32 v6 /*v262*/, v7, v5
	s_cselect_b32 s40, -1, 0
	s_wait_kmcnt 0x0
	s_bfe_i64 s[12:13], s[12:13], 0x200000
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_or_b32_e32 v1, 0x100, v0
	s_lshl_b32 s2, s29, 8
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_add3_u32 v8 /*v264*/, v4, v3, v6
	s_ashr_i32 s3, s2, 31
	v_add3_u32 v10 /*v266*/, v4, v2, v7
	v_or_b32_e32 v16 /*v272*/, 0x4300, v0
	v_lshl_or_b32 v12 /*v268*/, v0, 2, 0x11000
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_dual_mov_b32 v3, v5 /*v261*/ :: v_dual_mov_b32 v2, v5 /*v261*/
	s_add_nc_u64 s[28:29], s[0:1], s[6:7]
	s_lshl_b32 s0, s24, 8
	v_dual_mov_b32 v4, v5 /*v261*/ :: v_dual_mov_b32 v5, v5 /*v261*/
	s_ashr_i32 s1, s0, 31
	s_bfe_i64 s[8:9], s[8:9], 0x200000
	s_mul_u64 s[0:1], s[12:13], s[0:1]
	s_mul_u64 s[2:3], s[8:9], s[2:3]
	v_dual_mov_b32 v6, v5 /*v261*/ :: v_dual_mov_b32 v7, v5 /*v261*/
	s_set_vgpr_msb 0x141                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	v_dual_mov_b32 v13 /*v269*/, v5 /*v261*/ :: v_dual_mov_b32 v0 /*v256*/, v5 /*v261*/
	s_set_vgpr_msb 0x4101                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_dual_mov_b32 v8, v5 /*v261*/ :: v_dual_mov_b32 v9, v5 /*v261*/
	v_dual_mov_b32 v10, v5 /*v261*/ :: v_dual_mov_b32 v11, v5 /*v261*/
	v_dual_mov_b32 v12, v5 /*v261*/ :: v_dual_mov_b32 v13, v5 /*v261*/
	v_dual_mov_b32 v14, v5 /*v261*/ :: v_dual_mov_b32 v15, v5 /*v261*/
	v_dual_mov_b32 v16, v5 /*v261*/ :: v_dual_mov_b32 v17, v5 /*v261*/
	v_dual_mov_b32 v50, v5 /*v261*/ :: v_dual_mov_b32 v51, v5 /*v261*/
	v_dual_mov_b32 v52, v5 /*v261*/ :: v_dual_mov_b32 v53, v5 /*v261*/
	v_dual_mov_b32 v54, v5 /*v261*/ :: v_dual_mov_b32 v55, v5 /*v261*/
	v_dual_mov_b32 v56, v5 /*v261*/ :: v_dual_mov_b32 v57, v5 /*v261*/
	v_dual_mov_b32 v18, v5 /*v261*/ :: v_dual_mov_b32 v19, v5 /*v261*/
	v_dual_mov_b32 v20, v5 /*v261*/ :: v_dual_mov_b32 v21, v5 /*v261*/
	v_dual_mov_b32 v22, v5 /*v261*/ :: v_dual_mov_b32 v23, v5 /*v261*/
	v_dual_mov_b32 v24, v5 /*v261*/ :: v_dual_mov_b32 v25, v5 /*v261*/
	v_dual_mov_b32 v26, v5 /*v261*/ :: v_dual_mov_b32 v27, v5 /*v261*/
	v_dual_mov_b32 v28, v5 /*v261*/ :: v_dual_mov_b32 v29, v5 /*v261*/
	v_dual_mov_b32 v30, v5 /*v261*/ :: v_dual_mov_b32 v31, v5 /*v261*/
	v_dual_mov_b32 v32, v5 /*v261*/ :: v_dual_mov_b32 v33, v5 /*v261*/
	v_dual_mov_b32 v34, v5 /*v261*/ :: v_dual_mov_b32 v35, v5 /*v261*/
	v_dual_mov_b32 v36, v5 /*v261*/ :: v_dual_mov_b32 v37, v5 /*v261*/
	v_dual_mov_b32 v38, v5 /*v261*/ :: v_dual_mov_b32 v39, v5 /*v261*/
	v_dual_mov_b32 v40, v5 /*v261*/ :: v_dual_mov_b32 v41, v5 /*v261*/
	v_dual_mov_b32 v74, v5 /*v261*/ :: v_dual_mov_b32 v75, v5 /*v261*/
	v_dual_mov_b32 v76, v5 /*v261*/ :: v_dual_mov_b32 v77, v5 /*v261*/
	v_dual_mov_b32 v78, v5 /*v261*/ :: v_dual_mov_b32 v79, v5 /*v261*/
	v_dual_mov_b32 v80, v5 /*v261*/ :: v_dual_mov_b32 v81, v5 /*v261*/
	v_dual_mov_b32 v42, v5 /*v261*/ :: v_dual_mov_b32 v43, v5 /*v261*/
	v_dual_mov_b32 v44, v5 /*v261*/ :: v_dual_mov_b32 v45, v5 /*v261*/
	v_dual_mov_b32 v46, v5 /*v261*/ :: v_dual_mov_b32 v47, v5 /*v261*/
	v_dual_mov_b32 v48, v5 /*v261*/ :: v_dual_mov_b32 v49, v5 /*v261*/
	v_dual_mov_b32 v58, v5 /*v261*/ :: v_dual_mov_b32 v59, v5 /*v261*/
	v_dual_mov_b32 v60, v5 /*v261*/ :: v_dual_mov_b32 v61, v5 /*v261*/
	v_dual_mov_b32 v62, v5 /*v261*/ :: v_dual_mov_b32 v63, v5 /*v261*/
	v_dual_mov_b32 v64, v5 /*v261*/ :: v_dual_mov_b32 v65, v5 /*v261*/
	v_dual_mov_b32 v66, v5 /*v261*/ :: v_dual_mov_b32 v67, v5 /*v261*/
	v_dual_mov_b32 v68, v5 /*v261*/ :: v_dual_mov_b32 v69, v5 /*v261*/
	v_dual_mov_b32 v70, v5 /*v261*/ :: v_dual_mov_b32 v71, v5 /*v261*/
	v_dual_mov_b32 v72, v5 /*v261*/ :: v_dual_mov_b32 v73, v5 /*v261*/
	v_dual_mov_b32 v106, v5 /*v261*/ :: v_dual_mov_b32 v107, v5 /*v261*/
	v_dual_mov_b32 v108, v5 /*v261*/ :: v_dual_mov_b32 v109, v5 /*v261*/
	v_dual_mov_b32 v110, v5 /*v261*/ :: v_dual_mov_b32 v111, v5 /*v261*/
	v_dual_mov_b32 v112, v5 /*v261*/ :: v_dual_mov_b32 v113, v5 /*v261*/
	v_dual_mov_b32 v82, v5 /*v261*/ :: v_dual_mov_b32 v83, v5 /*v261*/
	v_dual_mov_b32 v84, v5 /*v261*/ :: v_dual_mov_b32 v85, v5 /*v261*/
	v_dual_mov_b32 v86, v5 /*v261*/ :: v_dual_mov_b32 v87, v5 /*v261*/
	v_dual_mov_b32 v88, v5 /*v261*/ :: v_dual_mov_b32 v89, v5 /*v261*/
	v_dual_mov_b32 v90, v5 /*v261*/ :: v_dual_mov_b32 v91, v5 /*v261*/
	v_dual_mov_b32 v92, v5 /*v261*/ :: v_dual_mov_b32 v93, v5 /*v261*/
	v_dual_mov_b32 v94, v5 /*v261*/ :: v_dual_mov_b32 v95, v5 /*v261*/
	v_dual_mov_b32 v96, v5 /*v261*/ :: v_dual_mov_b32 v97, v5 /*v261*/
	v_dual_mov_b32 v98, v5 /*v261*/ :: v_dual_mov_b32 v99, v5 /*v261*/
	v_dual_mov_b32 v100, v5 /*v261*/ :: v_dual_mov_b32 v101, v5 /*v261*/
	v_dual_mov_b32 v102, v5 /*v261*/ :: v_dual_mov_b32 v103, v5 /*v261*/
	v_dual_mov_b32 v104, v5 /*v261*/ :: v_dual_mov_b32 v105, v5 /*v261*/
	v_dual_mov_b32 v146, v5 /*v261*/ :: v_dual_mov_b32 v147, v5 /*v261*/
	v_dual_mov_b32 v148, v5 /*v261*/ :: v_dual_mov_b32 v149, v5 /*v261*/
	v_dual_mov_b32 v150, v5 /*v261*/ :: v_dual_mov_b32 v151, v5 /*v261*/
	v_dual_mov_b32 v152, v5 /*v261*/ :: v_dual_mov_b32 v153, v5 /*v261*/
	v_dual_mov_b32 v114, v5 /*v261*/ :: v_dual_mov_b32 v115, v5 /*v261*/
	v_dual_mov_b32 v116, v5 /*v261*/ :: v_dual_mov_b32 v117, v5 /*v261*/
	v_dual_mov_b32 v118, v5 /*v261*/ :: v_dual_mov_b32 v119, v5 /*v261*/
	v_dual_mov_b32 v120, v5 /*v261*/ :: v_dual_mov_b32 v121, v5 /*v261*/
	v_dual_mov_b32 v122, v5 /*v261*/ :: v_dual_mov_b32 v123, v5 /*v261*/
	v_dual_mov_b32 v124, v5 /*v261*/ :: v_dual_mov_b32 v125, v5 /*v261*/
	v_dual_mov_b32 v126, v5 /*v261*/ :: v_dual_mov_b32 v127, v5 /*v261*/
	v_dual_mov_b32 v128, v5 /*v261*/ :: v_dual_mov_b32 v129, v5 /*v261*/
	v_dual_mov_b32 v130, v5 /*v261*/ :: v_dual_mov_b32 v131, v5 /*v261*/
	v_dual_mov_b32 v132, v5 /*v261*/ :: v_dual_mov_b32 v133, v5 /*v261*/
	v_dual_mov_b32 v134, v5 /*v261*/ :: v_dual_mov_b32 v135, v5 /*v261*/
	v_dual_mov_b32 v136, v5 /*v261*/ :: v_dual_mov_b32 v137, v5 /*v261*/
	v_dual_mov_b32 v162, v5 /*v261*/ :: v_dual_mov_b32 v163, v5 /*v261*/
	v_dual_mov_b32 v164, v5 /*v261*/ :: v_dual_mov_b32 v165, v5 /*v261*/
	v_dual_mov_b32 v166, v5 /*v261*/ :: v_dual_mov_b32 v167, v5 /*v261*/
	v_dual_mov_b32 v168, v5 /*v261*/ :: v_dual_mov_b32 v169, v5 /*v261*/
	v_dual_mov_b32 v138, v5 /*v261*/ :: v_dual_mov_b32 v139, v5 /*v261*/
	v_dual_mov_b32 v140, v5 /*v261*/ :: v_dual_mov_b32 v141, v5 /*v261*/
	v_dual_mov_b32 v142, v5 /*v261*/ :: v_dual_mov_b32 v143, v5 /*v261*/
	v_dual_mov_b32 v144, v5 /*v261*/ :: v_dual_mov_b32 v145, v5 /*v261*/
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
	s_bitset1_b32 s41, 23
	s_bitset1_b32 s42, 23
	s_add_nc_u64 s[8:9], s[14:15], s[2:3]
	s_add_nc_u64 s[12:13], s[30:31], s[0:1]
	s_mov_b32 s0, 0x7510000
	s_mov_b32 s14, s7
	s_set_vgpr_msb 0x100                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_branch .LBB0_17
.LBB0_16:                               ;   in Loop: Header=BB0_17 Depth=1
	s_cmp_eq_u32 s14, s39
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_scc1 .LBB0_37
.LBB0_17:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_19 Depth 2
                                        ;     Child Loop BB0_22 Depth 2
                                        ;     Child Loop BB0_24 Depth 2
                                        ;     Child Loop BB0_27 Depth 2
	s_and_b32 s15, s14, 1
	s_add_co_i32 s14, s14, 1
	s_xor_b32 s30, s15, 1
	s_lshl_b32 s1, s14, 7
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_sub_co_i32 s1, s18, s1
	s_min_i32 s2, s1, 0x80
	s_cmp_lt_i32 s14, s39
	s_cselect_b32 s1, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_and_b32 s3, s1, exec_lo
	s_cselect_b32 s22, s2, 0
	s_cmp_lt_i32 s22, 0x80
	s_cselect_b32 s2, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b32 s2, s40, s2
	s_or_b32 s2, s35, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB0_29
; %bb.18:                               ;   in Loop: Header=BB0_17 Depth=1
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_mov_b64_e32 v[14:15] /*v[270:271]*/, v[0:1]
	v_mov_b32_e32 v17 /*v273*/, 0x44
	s_cmp_lg_u32 s30, 0
	s_mov_b32 s6, 0
	s_cselect_b32 s3, s38, s27
	s_cselect_b32 s2, s37, 0
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
.LBB0_19:                               ;   Parent Loop BB0_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_set_vgpr_msb 0x45                     ;  msbs: dst=1 src0=1 src1=1 src2=0
	v_dual_mov_b32 v4 /*v260*/, v14 /*v270*/ :: v_dual_add_nc_u32 v17 /*v273*/, -2, v17 /*v273*/
	v_dual_mov_b32 v18 /*v274*/, v15 /*v271*/ :: v_dual_mov_b32 v19 /*v275*/, v5 /*v261*/
	v_add_nc_u32_e32 v15 /*v271*/, 0x200, v15 /*v271*/
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u64 v[20:21] /*v[276:277]*/, v[4:5] /*v[260:261]*/, 2, s[2:3]
	v_cmp_eq_u32_e32 vcc_lo, 0, v17 /*v273*/
	v_add_nc_u32_e32 v14 /*v270*/, 0x200, v14 /*v270*/
	v_lshl_add_u64 v[18:19] /*v[274:275]*/, v[18:19] /*v[274:275]*/, 2, s[2:3]
	s_clause 0x1
	flat_store_b32 v[20:21] /*v[276:277]*/, v5 /*v261*/
	flat_store_b32 v[18:19] /*v[274:275]*/, v5 /*v261*/
	s_or_b32 s6, vcc_lo, s6
	s_wait_xcnt 0x0
	s_and_not1_b32 exec_lo, exec_lo, s6
	s_set_vgpr_msb 0x4500                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_cbranch_execnz .LBB0_19
; %bb.20:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s6
	s_and_saveexec_b32 s6, s7
	s_cbranch_execz .LBB0_23
; %bb.21:                               ;   in Loop: Header=BB0_17 Depth=1
	s_set_vgpr_msb 0x45                     ;  msbs: dst=1 src0=1 src1=1 src2=0
	v_add_nc_u64_e32 v[14:15] /*v[270:271]*/, s[2:3], v[12:13] /*v[268:269]*/
	v_mov_b32_e32 v4 /*v260*/, v16 /*v272*/
	s_mov_b32 s2, 0
	s_set_vgpr_msb 0x4500                   ;  msbs: dst=0 src0=0 src1=0 src2=0
.LBB0_22:                               ;   Parent Loop BB0_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_set_vgpr_msb 0x45                     ;  msbs: dst=1 src0=1 src1=1 src2=0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v4 /*v260*/, 0x100, v4 /*v260*/
	flat_store_b32 v[14:15] /*v[270:271]*/, v5 /*v261*/
	s_wait_xcnt 0x0
	v_add_nc_u64_e32 v[14:15] /*v[270:271]*/, 0x400, v[14:15] /*v[270:271]*/
	v_cmp_lt_u32_e32 vcc_lo, 0x42ff, v4 /*v260*/
	s_or_b32 s2, vcc_lo, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s2
	s_set_vgpr_msb 0x4500                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_cbranch_execnz .LBB0_22
.LBB0_23:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s6
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_mov_b64_e32 v[14:15] /*v[270:271]*/, v[0:1]
	v_mov_b32_e32 v17 /*v273*/, 0x44
	s_cmp_lg_u32 s30, 0
	s_mov_b32 s6, 0
	s_cselect_b32 s3, s29, s36
	s_cselect_b32 s2, s28, s26
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
.LBB0_24:                               ;   Parent Loop BB0_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_set_vgpr_msb 0x45                     ;  msbs: dst=1 src0=1 src1=1 src2=0
	v_dual_mov_b32 v4 /*v260*/, v14 /*v270*/ :: v_dual_add_nc_u32 v17 /*v273*/, -2, v17 /*v273*/
	v_dual_mov_b32 v18 /*v274*/, v15 /*v271*/ :: v_dual_mov_b32 v19 /*v275*/, v5 /*v261*/
	v_add_nc_u32_e32 v15 /*v271*/, 0x200, v15 /*v271*/
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u64 v[20:21] /*v[276:277]*/, v[4:5] /*v[260:261]*/, 2, s[2:3]
	v_cmp_eq_u32_e32 vcc_lo, 0, v17 /*v273*/
	v_add_nc_u32_e32 v14 /*v270*/, 0x200, v14 /*v270*/
	v_lshl_add_u64 v[18:19] /*v[274:275]*/, v[18:19] /*v[274:275]*/, 2, s[2:3]
	s_clause 0x1
	flat_store_b32 v[20:21] /*v[276:277]*/, v5 /*v261*/
	flat_store_b32 v[18:19] /*v[274:275]*/, v5 /*v261*/
	s_or_b32 s6, vcc_lo, s6
	s_wait_xcnt 0x0
	s_and_not1_b32 exec_lo, exec_lo, s6
	s_set_vgpr_msb 0x4500                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_cbranch_execnz .LBB0_24
; %bb.25:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s6
	s_and_saveexec_b32 s6, s7
	s_cbranch_execz .LBB0_28
; %bb.26:                               ;   in Loop: Header=BB0_17 Depth=1
	s_set_vgpr_msb 0x45                     ;  msbs: dst=1 src0=1 src1=1 src2=0
	v_add_nc_u64_e32 v[14:15] /*v[270:271]*/, s[2:3], v[12:13] /*v[268:269]*/
	v_mov_b32_e32 v4 /*v260*/, v16 /*v272*/
	s_mov_b32 s2, 0
	s_set_vgpr_msb 0x4500                   ;  msbs: dst=0 src0=0 src1=0 src2=0
.LBB0_27:                               ;   Parent Loop BB0_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_set_vgpr_msb 0x45                     ;  msbs: dst=1 src0=1 src1=1 src2=0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v4 /*v260*/, 0x100, v4 /*v260*/
	flat_store_b32 v[14:15] /*v[270:271]*/, v5 /*v261*/
	s_wait_xcnt 0x0
	v_add_nc_u64_e32 v[14:15] /*v[270:271]*/, 0x400, v[14:15] /*v[270:271]*/
	v_cmp_lt_u32_e32 vcc_lo, 0x42ff, v4 /*v260*/
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
	s_cselect_b32 s31, s14, 0
	s_mov_b32 s1, exec_lo
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_cmpx_lt_i32_e32 0, v11 /*v267*/
	s_xor_b32 s43, exec_lo, s1
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_cbranch_execnz .LBB0_32
; %bb.30:                               ;   in Loop: Header=BB0_17 Depth=1
	s_and_not1_saveexec_b32 s24, s43
	s_cbranch_execnz .LBB0_35
.LBB0_31:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s24
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s17
	s_cbranch_vccnz .LBB0_16
	s_branch .LBB0_36
.LBB0_32:                               ;   in Loop: Header=BB0_17 Depth=1
	s_mov_b32 s44, exec_lo
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_cmpx_eq_u32_e32 1, v11 /*v267*/
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_cbranch_execz .LBB0_34
; %bb.33:                               ;   in Loop: Header=BB0_17 Depth=1
	s_cmp_lg_u32 s30, 0
	s_cselect_b32 s1, s28, s26
	s_cmp_gt_i32 s22, 0
	s_cselect_b32 s24, -1, 0
	s_lshl_b32 s6, s31, 7
	s_delay_alu instid0(SALU_CYCLE_1)
	s_lshl_b64 s[2:3], s[6:7], 1
	s_mov_b32 s6, s7
	s_add_nc_u64 s[2:3], s[8:9], s[2:3]
	s_set_vgpr_msb 0x41                     ;  msbs: dst=1 src0=1 src1=0 src2=0
	v_dual_mov_b32 v15 /*v271*/, s1 :: v_dual_mov_b32 v14 /*v270*/, s2
	s_and_b32 s1, s3, 0x1ffffff
	s_and_b32 s3, s34, s24
	s_bitset1_b32 s1, 31
	v_cndmask_b32_e64 v4 /*v260*/, 0, 1, s3
	v_mov_b32_e32 v17 /*v273*/, s1
	s_mov_b32 s24, s22
	v_readfirstlane_b32 s49, v15 /*v271*/
	v_readfirstlane_b32 s50, v14 /*v270*/
	v_readfirstlane_b32 s48, v4 /*v260*/
	v_readfirstlane_b32 s51, v17 /*v273*/
	s_lshr_b64 s[2:3], s[24:25], 16
	s_lshl_b32 s1, s22, 16
	s_mov_b32 s3, s41
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[48:51], s[0:7]
	s_set_vgpr_msb 0x4100                   ;  msbs: dst=0 src0=0 src1=0 src2=0
.LBB0_34:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s44
	s_and_not1_saveexec_b32 s24, s43
	s_cbranch_execz .LBB0_31
.LBB0_35:                               ;   in Loop: Header=BB0_17 Depth=1
	s_cmp_lg_u32 s30, 0
	s_cselect_b32 s1, s37, 0
	s_cmp_gt_i32 s22, 0
	s_cselect_b32 s30, -1, 0
	s_lshl_b32 s6, s31, 7
	s_delay_alu instid0(SALU_CYCLE_1)
	s_lshl_b64 s[2:3], s[6:7], 1
	s_and_b32 s6, s21, s30
	s_add_nc_u64 s[2:3], s[12:13], s[2:3]
	s_set_vgpr_msb 0x41                     ;  msbs: dst=1 src0=1 src1=0 src2=0
	v_cndmask_b32_e64 v4 /*v260*/, 0, 1, s6
	s_and_b32 s3, s3, 0x1ffffff
	v_dual_mov_b32 v15 /*v271*/, s1 :: v_dual_mov_b32 v14 /*v270*/, s2
	s_bitset1_b32 s3, 31
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_readfirstlane_b32 s44, v4 /*v260*/
	v_mov_b32_e32 v17 /*v273*/, s3
	v_readfirstlane_b32 s45, v15 /*v271*/
	v_readfirstlane_b32 s46, v14 /*v270*/
	s_lshr_b64 s[2:3], s[22:23], 16
	s_lshl_b32 s1, s22, 16
	v_readfirstlane_b32 s47, v17 /*v273*/
	s_mov_b32 s3, s42
	s_mov_b32 s6, s7
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	tensor_load_to_lds s[44:47], s[0:7]
	s_or_b32 exec_lo, exec_lo, s24
	s_and_not1_b32 vcc_lo, exec_lo, s17
	s_set_vgpr_msb 0x4100                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_cbranch_vccnz .LBB0_16
.LBB0_36:                               ;   in Loop: Header=BB0_17 Depth=1
	s_cmp_lg_u32 s15, 0
	s_cselect_b32 s2, s37, 0
	s_cselect_b32 s1, s28, s26
	s_set_vgpr_msb 0x41                     ;  msbs: dst=1 src0=1 src1=0 src2=0
	v_lshl_add_u32 v4 /*v260*/, v2 /*v258*/, 1, s2
	v_lshl_add_u32 v14 /*v270*/, v6 /*v262*/, 1, s1
	ds_load_b128 v[18:21] /*v[274:277]*/, v4 /*v260*/
	ds_load_b128 v[22:25] /*v[278:281]*/, v4 /*v260*/ offset:16
	ds_load_b128 v[26:29] /*v[282:285]*/, v4 /*v260*/ offset:4352
	ds_load_b128 v[30:33] /*v[286:289]*/, v4 /*v260*/ offset:4368
	ds_load_b128 v[34:37] /*v[290:293]*/, v4 /*v260*/ offset:8704
	ds_load_b128 v[38:41] /*v[294:297]*/, v4 /*v260*/ offset:8720
	ds_load_b128 v[42:45] /*v[298:301]*/, v14 /*v270*/ offset:13056
	ds_load_b128 v[46:49] /*v[302:305]*/, v14 /*v270*/ offset:13072
	ds_load_b128 v[50:53] /*v[306:309]*/, v4 /*v260*/ offset:30464
	ds_load_b128 v[54:57] /*v[310:313]*/, v4 /*v260*/ offset:30480
	ds_load_b128 v[58:61] /*v[314:317]*/, v4 /*v260*/ offset:26112
	ds_load_b128 v[62:65] /*v[318:321]*/, v4 /*v260*/ offset:26128
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	s_set_vgpr_msb 0x4105                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[138:145], v[34:41] /*v[290:297]*/, v[42:49] /*v[298:305]*/, v[138:145] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[194:201], v[26:33] /*v[282:289]*/, v[42:49] /*v[298:305]*/, v[194:201] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[226:233], v[18:25] /*v[274:281]*/, v[42:49] /*v[298:305]*/, v[226:233] matrix_b_reuse
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[154:161], v[50:57] /*v[306:313]*/, v[42:49] /*v[298:305]*/, v[154:161] matrix_a_reuse
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[18:25], v[58:65] /*v[314:321]*/, v[42:49] /*v[298:305]*/, v[18:25] matrix_b_reuse
	s_set_vgpr_msb 0x541                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	ds_load_b128 v[66:69] /*v[322:325]*/, v4 /*v260*/ offset:21760
	ds_load_b128 v[70:73] /*v[326:329]*/, v4 /*v260*/ offset:21776
	ds_load_b128 v[74:77] /*v[330:333]*/, v4 /*v260*/ offset:17408
	ds_load_b128 v[78:81] /*v[334:337]*/, v4 /*v260*/ offset:17424
	ds_load_b128 v[82:85] /*v[338:341]*/, v4 /*v260*/ offset:13056
	ds_load_b128 v[86:89] /*v[342:345]*/, v4 /*v260*/ offset:13072
	v_lshl_add_u32 v4 /*v260*/, v8 /*v264*/, 1, s2
	s_set_vgpr_msb 0x4105                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[42:49], v[66:73] /*v[322:329]*/, v[42:49] /*v[298:305]*/, v[42:49] matrix_b_reuse
	s_set_vgpr_msb 0x541                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	ds_load_b128 v[90:93] /*v[346:349]*/, v4 /*v260*/ offset:17472
	ds_load_b128 v[94:97] /*v[350:353]*/, v4 /*v260*/ offset:17488
	ds_load_b128 v[98:101] /*v[354:357]*/, v4 /*v260*/ offset:26176
	ds_load_b128 v[102:105] /*v[358:361]*/, v4 /*v260*/ offset:26192
	ds_load_b128 v[106:109] /*v[362:365]*/, v4 /*v260*/ offset:30528
	ds_load_b128 v[110:113] /*v[366:369]*/, v4 /*v260*/ offset:30544
	s_set_vgpr_msb 0x4105                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x8
	v_wmma_f32_16x16x32_bf16 v[82:89], v[74:81] /*v[330:337]*/, v[42:49] /*v[298:305]*/, v[82:89] matrix_b_reuse
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[114:121], v[82:89] /*v[338:345]*/, v[42:49] /*v[298:305]*/, v[114:121] matrix_b_reuse
	s_set_vgpr_msb 0x541                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	ds_load_b128 v[42:45] /*v[298:301]*/, v14 /*v270*/ offset:8704
	ds_load_b128 v[46:49] /*v[302:305]*/, v14 /*v270*/ offset:8720
	s_set_vgpr_msb 0x4105                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[234:241], v[18:25] /*v[274:281]*/, v[42:49] /*v[298:305]*/, v[234:241] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[202:209], v[26:33] /*v[282:289]*/, v[42:49] /*v[298:305]*/, v[202:209] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[170:177], v[34:41] /*v[290:297]*/, v[42:49] /*v[298:305]*/, v[170:177] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[122:129], v[82:89] /*v[338:345]*/, v[42:49] /*v[298:305]*/, v[122:129] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[90:97], v[74:81] /*v[330:337]*/, v[42:49] /*v[298:305]*/, v[90:97] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[58:65], v[66:73] /*v[322:329]*/, v[42:49] /*v[298:305]*/, v[58:65] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[26:33], v[58:65] /*v[314:321]*/, v[42:49] /*v[298:305]*/, v[26:33] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[2:9], v[50:57] /*v[306:313]*/, v[42:49] /*v[298:305]*/, v[2:9] matrix_b_reuse
	s_set_vgpr_msb 0x541                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	ds_load_b128 v[42:45] /*v[298:301]*/, v14 /*v270*/
	ds_load_b128 v[46:49] /*v[302:305]*/, v14 /*v270*/ offset:16
	s_set_vgpr_msb 0x4105                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[106:113], v[66:73] /*v[322:329]*/, v[42:49] /*v[298:305]*/, v[106:113] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[74:81], v[58:65] /*v[314:321]*/, v[42:49] /*v[298:305]*/, v[74:81] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[50:57], v[50:57] /*v[306:313]*/, v[42:49] /*v[298:305]*/, v[50:57] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[250:257], v[18:25] /*v[274:281]*/, v[42:49] /*v[298:305]*/, v[250:257]
	v_wmma_f32_16x16x32_bf16 v[218:225], v[26:33] /*v[282:289]*/, v[42:49] /*v[298:305]*/, v[218:225] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[186:193], v[34:41] /*v[290:297]*/, v[42:49] /*v[298:305]*/, v[186:193] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[162:169], v[82:89] /*v[338:345]*/, v[42:49] /*v[298:305]*/, v[162:169] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[146:153], v[74:81] /*v[330:337]*/, v[42:49] /*v[298:305]*/, v[146:153] matrix_b_reuse
	s_set_vgpr_msb 0x541                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	ds_load_b128 v[42:45] /*v[298:301]*/, v14 /*v270*/ offset:4352
	ds_load_b128 v[46:49] /*v[302:305]*/, v14 /*v270*/ offset:4368
	v_lshl_add_u32 v14 /*v270*/, v10 /*v266*/, 1, s1
	ds_load_b128 v[114:117] /*v[370:373]*/, v14 /*v270*/ offset:4416
	s_set_vgpr_msb 0x4105                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x32_bf16 v[10:17], v[50:57] /*v[306:313]*/, v[42:49] /*v[298:305]*/, v[10:17] matrix_a_reuse
	s_set_vgpr_msb 0x541                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	ds_load_b128 v[50:53] /*v[306:309]*/, v4 /*v260*/ offset:64
	ds_load_b128 v[54:57] /*v[310:313]*/, v4 /*v260*/ offset:80
	ds_load_b128 v[118:121] /*v[374:377]*/, v14 /*v270*/ offset:4432
	ds_load_b128 v[122:125] /*v[378:381]*/, v14 /*v270*/ offset:8768
	ds_load_b128 v[126:129] /*v[382:385]*/, v14 /*v270*/ offset:8784
	ds_load_b128 v[130:133] /*v[386:389]*/, v14 /*v270*/ offset:13120
	ds_load_b128 v[134:137] /*v[390:393]*/, v14 /*v270*/ offset:13136
	s_set_vgpr_msb 0x4105                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[34:41], v[58:65] /*v[314:321]*/, v[42:49] /*v[298:305]*/, v[34:41] matrix_b_reuse
	s_set_vgpr_msb 0x541                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	ds_load_b128 v[58:61] /*v[314:317]*/, v4 /*v260*/ offset:4416
	ds_load_b128 v[62:65] /*v[318:321]*/, v4 /*v260*/ offset:4432
	s_set_vgpr_msb 0x4105                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[66:73], v[66:73] /*v[322:329]*/, v[42:49] /*v[298:305]*/, v[66:73] matrix_b_reuse
	s_set_vgpr_msb 0x541                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	ds_load_b128 v[66:69] /*v[322:325]*/, v4 /*v260*/ offset:8768
	ds_load_b128 v[70:73] /*v[326:329]*/, v4 /*v260*/ offset:8784
	s_set_vgpr_msb 0x4105                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[98:105], v[74:81] /*v[330:337]*/, v[42:49] /*v[298:305]*/, v[98:105] matrix_b_reuse
	s_set_vgpr_msb 0x541                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	ds_load_b128 v[74:77] /*v[330:333]*/, v4 /*v260*/ offset:13120
	ds_load_b128 v[78:81] /*v[334:337]*/, v4 /*v260*/ offset:13136
	s_set_vgpr_msb 0x4105                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[130:137], v[82:89] /*v[338:345]*/, v[42:49] /*v[298:305]*/, v[130:137] matrix_b_reuse
	s_set_vgpr_msb 0x541                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	ds_load_b128 v[82:85] /*v[338:341]*/, v4 /*v260*/ offset:21824
	ds_load_b128 v[86:89] /*v[342:345]*/, v4 /*v260*/ offset:21840
	s_set_vgpr_msb 0x4105                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[178:185], v[34:41] /*v[290:297]*/, v[42:49] /*v[298:305]*/, v[178:185] matrix_b_reuse
	s_set_vgpr_msb 0x541                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	ds_load_b128 v[34:37] /*v[290:293]*/, v14 /*v270*/ offset:64
	ds_load_b128 v[38:41] /*v[294:297]*/, v14 /*v270*/ offset:80
	s_set_vgpr_msb 0x4105                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[210:217], v[26:33] /*v[282:289]*/, v[42:49] /*v[298:305]*/, v[210:217] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[242:249], v[18:25] /*v[274:281]*/, v[42:49] /*v[298:305]*/, v[242:249] matrix_b_reuse
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x541                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	ds_load_b128 v[18:21] /*v[274:277]*/, v4 /*v260*/ offset:128
	ds_load_b128 v[22:25] /*v[278:281]*/, v4 /*v260*/ offset:144
	ds_load_b128 v[26:29] /*v[282:285]*/, v4 /*v260*/ offset:4480
	ds_load_b128 v[30:33] /*v[286:289]*/, v4 /*v260*/ offset:4496
	ds_load_b128 v[42:45] /*v[298:301]*/, v4 /*v260*/ offset:8832
	ds_load_b128 v[46:49] /*v[302:305]*/, v4 /*v260*/ offset:8848
	s_set_vgpr_msb 0x4105                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[250:257], v[50:57] /*v[306:313]*/, v[34:41] /*v[290:297]*/, v[250:257]
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[218:225], v[58:65] /*v[314:321]*/, v[34:41] /*v[290:297]*/, v[218:225] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[186:193], v[66:73] /*v[322:329]*/, v[34:41] /*v[290:297]*/, v[186:193] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[162:169], v[74:81] /*v[330:337]*/, v[34:41] /*v[290:297]*/, v[162:169] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[146:153], v[90:97] /*v[346:353]*/, v[34:41] /*v[290:297]*/, v[146:153] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[106:113], v[82:89] /*v[338:345]*/, v[34:41] /*v[290:297]*/, v[106:113] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[74:81], v[98:105] /*v[354:361]*/, v[34:41] /*v[290:297]*/, v[74:81] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[50:57], v[106:113] /*v[362:369]*/, v[34:41] /*v[290:297]*/, v[50:57] matrix_b_reuse
	s_set_vgpr_msb 0x541                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	ds_load_b128 v[34:37] /*v[290:293]*/, v4 /*v260*/ offset:13184
	ds_load_b128 v[38:41] /*v[294:297]*/, v4 /*v260*/ offset:13200
	ds_load_b128 v[138:141] /*v[394:397]*/, v4 /*v260*/ offset:17536
	ds_load_b128 v[142:145] /*v[398:401]*/, v4 /*v260*/ offset:17552
	ds_load_b128 v[146:149] /*v[402:405]*/, v4 /*v260*/ offset:21888
	ds_load_b128 v[150:153] /*v[406:409]*/, v4 /*v260*/ offset:21904
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	s_set_vgpr_msb 0x4105                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[10:17], v[106:113] /*v[362:369]*/, v[114:121] /*v[370:377]*/, v[10:17] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[34:41], v[98:105] /*v[354:361]*/, v[114:121] /*v[370:377]*/, v[34:41] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[66:73], v[82:89] /*v[338:345]*/, v[114:121] /*v[370:377]*/, v[66:73] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[98:105], v[90:97] /*v[346:353]*/, v[114:121] /*v[370:377]*/, v[98:105] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[130:137], v[74:81] /*v[330:337]*/, v[114:121] /*v[370:377]*/, v[130:137] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[178:185], v[66:73] /*v[322:329]*/, v[114:121] /*v[370:377]*/, v[178:185] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[210:217], v[58:65] /*v[314:321]*/, v[114:121] /*v[370:377]*/, v[210:217] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[242:249], v[50:57] /*v[306:313]*/, v[114:121] /*v[370:377]*/, v[242:249] matrix_b_reuse
	s_set_vgpr_msb 0x541                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	ds_load_b128 v[114:117] /*v[370:373]*/, v4 /*v260*/ offset:26240
	ds_load_b128 v[118:121] /*v[374:377]*/, v4 /*v260*/ offset:26256
	ds_load_b128 v[154:157] /*v[410:413]*/, v4 /*v260*/ offset:30592
	ds_load_b128 v[158:161] /*v[414:417]*/, v4 /*v260*/ offset:30608
	ds_load_b128 v[162:165] /*v[418:421]*/, v14 /*v270*/ offset:128
	ds_load_b128 v[166:169] /*v[422:425]*/, v14 /*v270*/ offset:144
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	s_set_vgpr_msb 0x4105                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[234:241], v[50:57] /*v[306:313]*/, v[122:129] /*v[378:385]*/, v[234:241] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[202:209], v[58:65] /*v[314:321]*/, v[122:129] /*v[378:385]*/, v[202:209] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[170:177], v[66:73] /*v[322:329]*/, v[122:129] /*v[378:385]*/, v[170:177] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[122:129], v[74:81] /*v[330:337]*/, v[122:129] /*v[378:385]*/, v[122:129] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[90:97], v[90:97] /*v[346:353]*/, v[122:129] /*v[378:385]*/, v[90:97] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[58:65], v[82:89] /*v[338:345]*/, v[122:129] /*v[378:385]*/, v[58:65] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[26:33], v[98:105] /*v[354:361]*/, v[122:129] /*v[378:385]*/, v[26:33] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[2:9], v[106:113] /*v[362:369]*/, v[122:129] /*v[378:385]*/, v[2:9] matrix_b_reuse
	s_set_vgpr_msb 0x541                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	ds_load_b128 v[122:125] /*v[378:381]*/, v14 /*v270*/ offset:4480
	ds_load_b128 v[126:129] /*v[382:385]*/, v14 /*v270*/ offset:4496
	ds_load_b128 v[170:173] /*v[426:429]*/, v14 /*v270*/ offset:8832
	ds_load_b128 v[174:177] /*v[430:433]*/, v14 /*v270*/ offset:8848
	ds_load_b128 v[178:181] /*v[434:437]*/, v14 /*v270*/ offset:13184
	ds_load_b128 v[182:185] /*v[438:441]*/, v14 /*v270*/ offset:13200
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	s_set_vgpr_msb 0x4105                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[106:113] /*v[362:369]*/, v[130:137] /*v[386:393]*/, v[154:161] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[18:25], v[98:105] /*v[354:361]*/, v[130:137] /*v[386:393]*/, v[18:25] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[42:49], v[82:89] /*v[338:345]*/, v[130:137] /*v[386:393]*/, v[42:49] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[82:89], v[90:97] /*v[346:353]*/, v[130:137] /*v[386:393]*/, v[82:89] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[114:121], v[74:81] /*v[330:337]*/, v[130:137] /*v[386:393]*/, v[114:121] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[138:145], v[66:73] /*v[322:329]*/, v[130:137] /*v[386:393]*/, v[138:145] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[194:201], v[58:65] /*v[314:321]*/, v[130:137] /*v[386:393]*/, v[194:201] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[226:233], v[50:57] /*v[306:313]*/, v[130:137] /*v[386:393]*/, v[226:233] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x541                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	ds_load_b128 v[50:53] /*v[306:309]*/, v4 /*v260*/ offset:192
	ds_load_b128 v[54:57] /*v[310:313]*/, v4 /*v260*/ offset:208
	ds_load_b128 v[58:61] /*v[314:317]*/, v4 /*v260*/ offset:4544
	ds_load_b128 v[62:65] /*v[318:321]*/, v4 /*v260*/ offset:4560
	ds_load_b128 v[66:69] /*v[322:325]*/, v4 /*v260*/ offset:8896
	ds_load_b128 v[70:73] /*v[326:329]*/, v4 /*v260*/ offset:8912
	s_set_vgpr_msb 0x4105                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[250:257], v[18:25] /*v[274:281]*/, v[162:169] /*v[418:425]*/, v[250:257]
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[218:225], v[26:33] /*v[282:289]*/, v[162:169] /*v[418:425]*/, v[218:225] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[186:193], v[42:49] /*v[298:305]*/, v[162:169] /*v[418:425]*/, v[186:193] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[162:169], v[34:41] /*v[290:297]*/, v[162:169] /*v[418:425]*/, v[162:169] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[146:153], v[138:145] /*v[394:401]*/, v[162:169] /*v[418:425]*/, v[146:153] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[106:113], v[146:153] /*v[402:409]*/, v[162:169] /*v[418:425]*/, v[106:113] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[74:81], v[114:121] /*v[370:377]*/, v[162:169] /*v[418:425]*/, v[74:81] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[50:57], v[154:161] /*v[410:417]*/, v[162:169] /*v[418:425]*/, v[50:57] matrix_b_reuse
	s_set_vgpr_msb 0x541                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	ds_load_b128 v[74:77] /*v[330:333]*/, v4 /*v260*/ offset:13248
	ds_load_b128 v[78:81] /*v[334:337]*/, v4 /*v260*/ offset:13264
	ds_load_b128 v[82:85] /*v[338:341]*/, v4 /*v260*/ offset:17600
	ds_load_b128 v[86:89] /*v[342:345]*/, v4 /*v260*/ offset:17616
	ds_load_b128 v[90:93] /*v[346:349]*/, v4 /*v260*/ offset:21952
	ds_load_b128 v[94:97] /*v[350:353]*/, v4 /*v260*/ offset:21968
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	s_set_vgpr_msb 0x4105                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x10
	v_wmma_f32_16x16x32_bf16 v[10:17], v[154:161] /*v[410:417]*/, v[122:129] /*v[378:385]*/, v[10:17] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[34:41], v[114:121] /*v[370:377]*/, v[122:129] /*v[378:385]*/, v[34:41] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[66:73], v[146:153] /*v[402:409]*/, v[122:129] /*v[378:385]*/, v[66:73] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[98:105], v[138:145] /*v[394:401]*/, v[122:129] /*v[378:385]*/, v[98:105] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[130:137], v[34:41] /*v[290:297]*/, v[122:129] /*v[378:385]*/, v[130:137] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[178:185], v[42:49] /*v[298:305]*/, v[122:129] /*v[378:385]*/, v[178:185] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[210:217], v[26:33] /*v[282:289]*/, v[122:129] /*v[378:385]*/, v[210:217] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[242:249], v[18:25] /*v[274:281]*/, v[122:129] /*v[378:385]*/, v[242:249] matrix_b_reuse
	s_set_vgpr_msb 0x541                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	ds_load_b128 v[98:101] /*v[354:357]*/, v4 /*v260*/ offset:26304
	ds_load_b128 v[102:105] /*v[358:361]*/, v4 /*v260*/ offset:26320
	ds_load_b128 v[106:109] /*v[362:365]*/, v4 /*v260*/ offset:30656
	ds_load_b128 v[110:113] /*v[366:369]*/, v4 /*v260*/ offset:30672
	ds_load_b128 v[122:125] /*v[378:381]*/, v14 /*v270*/ offset:192
	ds_load_b128 v[126:129] /*v[382:385]*/, v14 /*v270*/ offset:208
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	s_set_vgpr_msb 0x4105                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x14
	v_wmma_f32_16x16x32_bf16 v[234:241], v[18:25] /*v[274:281]*/, v[170:177] /*v[426:433]*/, v[234:241] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[202:209], v[26:33] /*v[282:289]*/, v[170:177] /*v[426:433]*/, v[202:209] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[170:177], v[42:49] /*v[298:305]*/, v[170:177] /*v[426:433]*/, v[170:177] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[122:129], v[34:41] /*v[290:297]*/, v[170:177] /*v[426:433]*/, v[122:129] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[90:97], v[138:145] /*v[394:401]*/, v[170:177] /*v[426:433]*/, v[90:97] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[58:65], v[146:153] /*v[402:409]*/, v[170:177] /*v[426:433]*/, v[58:65] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[26:33], v[114:121] /*v[370:377]*/, v[170:177] /*v[426:433]*/, v[26:33] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[2:9], v[154:161] /*v[410:417]*/, v[170:177] /*v[426:433]*/, v[2:9] matrix_b_reuse
	s_set_vgpr_msb 0x541                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	ds_load_b128 v[130:133] /*v[386:389]*/, v14 /*v270*/ offset:4544
	ds_load_b128 v[134:137] /*v[390:393]*/, v14 /*v270*/ offset:4560
	ds_load_b128 v[162:165] /*v[418:421]*/, v14 /*v270*/ offset:8896
	ds_load_b128 v[166:169] /*v[422:425]*/, v14 /*v270*/ offset:8912
	ds_load_b128 v[170:173] /*v[426:429]*/, v14 /*v270*/ offset:13248
	ds_load_b128 v[174:177] /*v[430:433]*/, v14 /*v270*/ offset:13264
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	s_set_vgpr_msb 0x4105                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x18
	v_wmma_f32_16x16x32_bf16 v[154:161], v[154:161] /*v[410:417]*/, v[178:185] /*v[434:441]*/, v[154:161] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[18:25], v[114:121] /*v[370:377]*/, v[178:185] /*v[434:441]*/, v[18:25] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[42:49], v[146:153] /*v[402:409]*/, v[178:185] /*v[434:441]*/, v[42:49] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[82:89], v[138:145] /*v[394:401]*/, v[178:185] /*v[434:441]*/, v[82:89] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[114:121], v[34:41] /*v[290:297]*/, v[178:185] /*v[434:441]*/, v[114:121] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[138:145], v[42:49] /*v[298:305]*/, v[178:185] /*v[434:441]*/, v[138:145] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[194:201], v[26:33] /*v[282:289]*/, v[178:185] /*v[434:441]*/, v[194:201] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[226:233], v[18:25] /*v[274:281]*/, v[178:185] /*v[434:441]*/, v[226:233] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[250:257], v[50:57] /*v[306:313]*/, v[122:129] /*v[378:385]*/, v[250:257]
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[218:225], v[58:65] /*v[314:321]*/, v[122:129] /*v[378:385]*/, v[218:225] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[186:193], v[66:73] /*v[322:329]*/, v[122:129] /*v[378:385]*/, v[186:193] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[162:169], v[74:81] /*v[330:337]*/, v[122:129] /*v[378:385]*/, v[162:169] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[146:153], v[82:89] /*v[338:345]*/, v[122:129] /*v[378:385]*/, v[146:153] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[106:113], v[90:97] /*v[346:353]*/, v[122:129] /*v[378:385]*/, v[106:113] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[74:81], v[98:105] /*v[354:361]*/, v[122:129] /*v[378:385]*/, v[74:81] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[50:57], v[106:113] /*v[362:369]*/, v[122:129] /*v[378:385]*/, v[50:57] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[10:17], v[106:113] /*v[362:369]*/, v[130:137] /*v[386:393]*/, v[10:17] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[34:41], v[98:105] /*v[354:361]*/, v[130:137] /*v[386:393]*/, v[34:41] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[66:73], v[90:97] /*v[346:353]*/, v[130:137] /*v[386:393]*/, v[66:73] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[98:105], v[82:89] /*v[338:345]*/, v[130:137] /*v[386:393]*/, v[98:105] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[130:137], v[74:81] /*v[330:337]*/, v[130:137] /*v[386:393]*/, v[130:137] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[178:185], v[66:73] /*v[322:329]*/, v[130:137] /*v[386:393]*/, v[178:185] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[210:217], v[58:65] /*v[314:321]*/, v[130:137] /*v[386:393]*/, v[210:217] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[242:249], v[50:57] /*v[306:313]*/, v[130:137] /*v[386:393]*/, v[242:249] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[234:241], v[50:57] /*v[306:313]*/, v[162:169] /*v[418:425]*/, v[234:241] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[202:209], v[58:65] /*v[314:321]*/, v[162:169] /*v[418:425]*/, v[202:209] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[170:177], v[66:73] /*v[322:329]*/, v[162:169] /*v[418:425]*/, v[170:177] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[122:129], v[74:81] /*v[330:337]*/, v[162:169] /*v[418:425]*/, v[122:129] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[90:97], v[82:89] /*v[338:345]*/, v[162:169] /*v[418:425]*/, v[90:97] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[58:65], v[90:97] /*v[346:353]*/, v[162:169] /*v[418:425]*/, v[58:65] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[26:33], v[98:105] /*v[354:361]*/, v[162:169] /*v[418:425]*/, v[26:33] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[2:9], v[106:113] /*v[362:369]*/, v[162:169] /*v[418:425]*/, v[2:9] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[106:113] /*v[362:369]*/, v[170:177] /*v[426:433]*/, v[154:161] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[18:25], v[98:105] /*v[354:361]*/, v[170:177] /*v[426:433]*/, v[18:25] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[42:49], v[90:97] /*v[346:353]*/, v[170:177] /*v[426:433]*/, v[42:49] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[82:89], v[82:89] /*v[338:345]*/, v[170:177] /*v[426:433]*/, v[82:89] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[114:121], v[74:81] /*v[330:337]*/, v[170:177] /*v[426:433]*/, v[114:121] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[138:145], v[66:73] /*v[322:329]*/, v[170:177] /*v[426:433]*/, v[138:145] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[194:201], v[58:65] /*v[314:321]*/, v[170:177] /*v[426:433]*/, v[194:201] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[226:233], v[50:57] /*v[306:313]*/, v[170:177] /*v[426:433]*/, v[226:233] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
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
	s_and_b32 vcc_lo, exec_lo, s17
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_vccz .LBB0_39
; %bb.38:
	v_lshrrev_b32_e32 v1, 1, v0
	s_set_vgpr_msb 0x50                     ;  msbs: dst=1 src0=0 src1=0 src2=1
	v_and_or_b32 v2 /*v258*/, v0, 15, v9 /*v265*/
	s_set_vgpr_msb 0x5010                   ;  msbs: dst=0 src0=0 src1=0 src2=1
	v_cvt_pk_bf16_f32 v241, v240, v241
	v_cvt_pk_bf16_f32 v240, v238, v239
	v_cvt_pk_bf16_f32 v239, v236, v237
	v_and_or_b32 v1, v1, 8, v7 /*v263*/
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
	v_cvt_pk_bf16_f32 v153, v152, v153
	v_lshrrev_b32_e32 v166, 3, v164
	v_cvt_pk_bf16_f32 v152, v150, v151
	v_cvt_pk_bf16_f32 v151, v148, v149
	v_add_nc_u32_e32 v149, 0x70, v1
	v_dual_lshrrev_b32 v197, 3, v204 :: v_dual_lshrrev_b32 v148, 3, v167
	v_and_b32_e32 v219, 0x3ff0, v219
	v_add_nc_u32_e32 v188, 0x2020, v1
	v_cvt_pk_bf16_f32 v183, v180, v181
	v_add_nc_u32_e32 v180, 0x3020, v1
	v_and_b32_e32 v163, 0x1bf0, v173
	v_add_nc_u32_e32 v165, 0x1030, v1
	v_cvt_pk_bf16_f32 v150, v146, v147
	v_and_b32_e32 v146, 0x1bf0, v166
	v_cvt_pk_bf16_f32 v113, v112, v113
	v_cvt_pk_bf16_f32 v112, v110, v111
	v_cvt_pk_bf16_f32 v110, v106, v107
	v_lshrrev_b32_e32 v107, 3, v149
	v_and_b32_e32 v186, 0x3ff0, v197
	v_lshlrev_b32_e32 v187, 1, v213
	v_and_b32_e32 v148, 0x1bf0, v148
	v_cvt_pk_bf16_f32 v57, v56, v57
	v_cvt_pk_bf16_f32 v56, v54, v55
	v_cvt_pk_bf16_f32 v54, v50, v51
	v_add_nc_u32_e32 v50, 0x2030, v1
	v_add_nc_u32_e32 v210, v219, v220
	v_add_nc_u32_e32 v202, v211, v220
	v_cvt_pk_bf16_f32 v198, v194, v195
	v_lshrrev_b32_e32 v188, 3, v188
	v_dual_lshrrev_b32 v170, 3, v180 :: v_dual_add_nc_u32 v163, v163, v242
	v_dual_lshrrev_b32 v147, 3, v165 :: v_dual_add_nc_u32 v146, v146, v242
	v_cvt_pk_bf16_f32 v111, v108, v109
	v_cvt_pk_bf16_f32 v81, v80, v81
	v_cvt_pk_bf16_f32 v80, v78, v79
	v_cvt_pk_bf16_f32 v78, v74, v75
	v_and_b32_e32 v74, 0x1bf0, v107
	v_dual_add_nc_u32 v186, v186, v187 :: v_dual_add_nc_u32 v106, v148, v242
	v_cvt_pk_bf16_f32 v79, v76, v77
	v_lshrrev_b32_e32 v75, 3, v50
	v_and_b32_e32 v188, 0x3ff0, v188
	v_and_b32_e32 v162, 0x3ff0, v170
	ds_store_b128 v210, v[206:209] offset:16384
	ds_store_b128 v163, v[150:153] offset:128
	ds_store_b128 v202, v[198:201] offset:24576
	ds_store_b128 v146, v[110:113] offset:160
	ds_store_b128 v186, v[182:185] offset:8192
	ds_store_b128 v106, v[78:81] offset:192
	v_add_nc_u32_e32 v79, 0x3030, v1
	v_add_nc_u32_e32 v106, v74, v242
	v_and_b32_e32 v74, 0x3ff0, v147
	v_lshlrev_b32_e32 v108, 1, v189
	v_and_b32_e32 v78, 0x3ff0, v75
	v_add_nc_u32_e32 v179, v188, v187
	v_cvt_pk_bf16_f32 v55, v52, v53
	v_add_nc_u32_e32 v107, v162, v187
	v_cvt_pk_bf16_f32 v53, v144, v145
	v_cvt_pk_bf16_f32 v52, v142, v143
	v_cvt_pk_bf16_f32 v51, v140, v141
	v_cvt_pk_bf16_f32 v50, v138, v139
	v_lshrrev_b32_e32 v111, 3, v79
	v_add_nc_u32_e32 v112, 0x1040, v1
	v_add_nc_u32_e32 v109, v74, v108
	v_cvt_pk_bf16_f32 v77, v136, v137
	v_cvt_pk_bf16_f32 v76, v134, v135
	v_cvt_pk_bf16_f32 v75, v132, v133
	v_cvt_pk_bf16_f32 v74, v130, v131
	v_add_nc_u32_e32 v110, v78, v108
	v_cvt_pk_bf16_f32 v81, v128, v129
	v_cvt_pk_bf16_f32 v80, v126, v127
	v_cvt_pk_bf16_f32 v79, v124, v125
	v_cvt_pk_bf16_f32 v78, v122, v123
	ds_store_b128 v179, v[174:177] offset:16384
	v_and_b32_e32 v111, 0x3ff0, v111
	ds_store_b128 v107, v[50:53] offset:24576
	ds_store_b128 v109, v[74:77] offset:8192
	v_lshrrev_b32_e32 v50, 3, v112
	v_add_nc_u32_e32 v51, 0x2040, v1
	ds_store_b128 v110, v[78:81] offset:16384
	v_add_nc_u32_e32 v79, 0x3040, v1
	v_add_nc_u32_e32 v107, v111, v108
	v_and_b32_e32 v74, 0x3ff0, v50
	v_dual_lshlrev_b32 v108, 1, v172 :: v_dual_lshrrev_b32 v75, 3, v51
	v_cvt_pk_bf16_f32 v53, v120, v121
	v_cvt_pk_bf16_f32 v52, v118, v119
	v_cvt_pk_bf16_f32 v51, v116, v117
	v_cvt_pk_bf16_f32 v50, v114, v115
	v_cvt_pk_bf16_f32 v81, v96, v97
	v_lshrrev_b32_e32 v96, 3, v79
	v_cvt_pk_bf16_f32 v79, v92, v93
	v_add_nc_u32_e32 v92, 0x1050, v1
	v_add_nc_u32_e32 v109, v74, v108
	v_cvt_pk_bf16_f32 v77, v104, v105
	v_and_b32_e32 v78, 0x3ff0, v75
	v_cvt_pk_bf16_f32 v76, v102, v103
	v_cvt_pk_bf16_f32 v75, v100, v101
	v_cvt_pk_bf16_f32 v74, v98, v99
	ds_store_b128 v107, v[50:53] offset:24576
	ds_store_b128 v109, v[74:77] offset:8192
	v_lshrrev_b32_e32 v50, 3, v92
	v_add_nc_u32_e32 v98, v78, v108
	v_cvt_pk_bf16_f32 v78, v90, v91
	v_and_b32_e32 v90, 0x3ff0, v96
	v_add_nc_u32_e32 v51, 0x2050, v1
	v_and_b32_e32 v75, 0x3ff0, v50
	v_lshlrev_b32_e32 v76, 1, v164
	v_cvt_pk_bf16_f32 v80, v94, v95
	v_add_nc_u32_e32 v74, v90, v108
	v_cvt_pk_bf16_f32 v53, v88, v89
	v_cvt_pk_bf16_f32 v52, v86, v87
	v_lshrrev_b32_e32 v77, 3, v51
	v_cvt_pk_bf16_f32 v51, v84, v85
	v_cvt_pk_bf16_f32 v50, v82, v83
	v_add_nc_u32_e32 v75, v75, v76
	v_cvt_pk_bf16_f32 v73, v72, v73
	v_cvt_pk_bf16_f32 v72, v70, v71
	v_cvt_pk_bf16_f32 v71, v68, v69
	v_cvt_pk_bf16_f32 v70, v66, v67
	ds_store_b128 v98, v[78:81] offset:16384
	v_add_nc_u32_e32 v78, 0x3050, v1
	v_cvt_pk_bf16_f32 v49, v48, v49
	v_cvt_pk_bf16_f32 v48, v46, v47
	v_cvt_pk_bf16_f32 v47, v44, v45
	v_add_nc_u32_e32 v44, 0x3060, v1
	v_cvt_pk_bf16_f32 v65, v64, v65
	v_cvt_pk_bf16_f32 v64, v62, v63
	v_cvt_pk_bf16_f32 v63, v60, v61
	v_add_nc_u32_e32 v60, 0x1060, v1
	ds_store_b128 v74, v[50:53] offset:24576
	ds_store_b128 v75, v[70:73] offset:8192
	v_add_nc_u32_e32 v52, 0x2060, v1
	v_cvt_pk_bf16_f32 v33, v32, v33
	v_cvt_pk_bf16_f32 v32, v30, v31
	v_cvt_pk_bf16_f32 v30, v26, v27
	v_add_nc_u32_e32 v26, 0x1070, v1
	v_cvt_pk_bf16_f32 v31, v28, v29
	v_add_nc_u32_e32 v28, 0x2070, v1
	v_add_nc_u32_e32 v1, 0x3070, v1
	v_lshrrev_b32_e32 v67, 3, v78
	v_cvt_pk_bf16_f32 v41, v40, v41
	v_cvt_pk_bf16_f32 v40, v38, v39
	v_cvt_pk_bf16_f32 v38, v34, v35
	v_lshrrev_b32_e32 v35, 3, v44
	v_lshrrev_b32_e32 v50, 3, v60
	v_dual_lshrrev_b32 v52, 3, v52 :: v_dual_lshrrev_b32 v26, 3, v26
	v_cvt_pk_bf16_f32 v25, v24, v25
	v_cvt_pk_bf16_f32 v24, v22, v23
	v_cvt_pk_bf16_f32 v23, v20, v21
	v_dual_lshrrev_b32 v20, 3, v28 :: v_dual_lshrrev_b32 v1, 3, v1
	v_and_b32_e32 v77, 0x3ff0, v77
	v_cvt_pk_bf16_f32 v62, v58, v59
	v_and_b32_e32 v58, 0x3ff0, v67
	v_lshlrev_b32_e32 v53, 1, v167
	v_and_b32_e32 v27, 0x3ff0, v35
	v_and_b32_e32 v50, 0x3ff0, v50
	v_cvt_pk_bf16_f32 v46, v42, v43
	v_and_b32_e32 v43, 0x3ff0, v52
	v_and_b32_e32 v26, 0x3ff0, v26
	v_lshlrev_b32_e32 v29, 1, v149
	v_cvt_pk_bf16_f32 v22, v18, v19
	v_and_b32_e32 v19, 0x3ff0, v20
	v_and_b32_e32 v1, 0x3ff0, v1
	v_dual_add_nc_u32 v66, v77, v76 :: v_dual_add_nc_u32 v51, v58, v76
	v_dual_add_nc_u32 v27, v27, v53 :: v_dual_add_nc_u32 v42, v50, v53
	v_cvt_pk_bf16_f32 v39, v36, v37
	v_add_nc_u32_e32 v34, v43, v53
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
	v_cvt_pk_bf16_f32 v5, v160, v161
	v_cvt_pk_bf16_f32 v4, v158, v159
	v_cvt_pk_bf16_f32 v3, v156, v157
	v_cvt_pk_bf16_f32 v2, v154, v155
	ds_store_b128 v66, v[62:65] offset:16384
	ds_store_b128 v51, v[46:49] offset:24576
	ds_store_b128 v42, v[38:41] offset:8192
	ds_store_b128 v34, v[30:33] offset:16384
	ds_store_b128 v27, v[22:25] offset:24576
	ds_store_b128 v106, v[54:57] offset:224
	ds_store_b128 v18, v[14:17] offset:8192
	ds_store_b128 v10, v[6:9] offset:16384
	ds_store_b128 v1, v[2:5] offset:24576
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
	.size	bm256_bn256_bk128_wm2_wn4_mc0, .Lfunc_end0-bm256_bn256_bk128_wm2_wn4_mc0
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel bm256_bn256_bk128_wm2_wn4_mc0
		.amdhsa_group_segment_fixed_size 278528
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
		.amdhsa_next_free_vgpr 442
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
		.amdhsa_inst_pref_size 82
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text.bm256_bn256_bk128_wm2_wn4_mc0,"axG",@progbits,bm256_bn256_bk128_wm2_wn4_mc0,comdat
                                        ; -- End function
	.set .Lbm256_bn256_bk128_wm2_wn4_mc0.num_vgpr, 442
	.set .Lbm256_bn256_bk128_wm2_wn4_mc0.num_agpr, 0
	.set .Lbm256_bn256_bk128_wm2_wn4_mc0.numbered_sgpr, 52
	.set .Lbm256_bn256_bk128_wm2_wn4_mc0.num_named_barrier, 0
	.set .Lbm256_bn256_bk128_wm2_wn4_mc0.private_seg_size, 0
	.set .Lbm256_bn256_bk128_wm2_wn4_mc0.uses_vcc, 1
	.set .Lbm256_bn256_bk128_wm2_wn4_mc0.uses_flat_scratch, 1
	.set .Lbm256_bn256_bk128_wm2_wn4_mc0.has_dyn_sized_stack, 0
	.set .Lbm256_bn256_bk128_wm2_wn4_mc0.has_recursion, 0
	.set .Lbm256_bn256_bk128_wm2_wn4_mc0.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 10480
; TotalNumSgprs: 54
; NumVgprs: 442
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 278528 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 27
; NumSGPRsForWavesPerEU: 54
; NumVGPRsForWavesPerEU: 442
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
	.type	__hip_cuid_c3c58847898bb86d,@object ; @__hip_cuid_c3c58847898bb86d
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_c3c58847898bb86d
__hip_cuid_c3c58847898bb86d:
	.byte	0                               ; 0x0
	.size	__hip_cuid_c3c58847898bb86d, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git aa451e1fe6a793394d6733051b1778633063ae96+PATCHED:d17c5aa0e3ea29cde402f58f27e39b6034effa27)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_c3c58847898bb86d
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
    .group_segment_fixed_size: 278528
    .kernarg_segment_align: 8
    .kernarg_segment_size: 132
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           bm256_bn256_bk128_wm2_wn4_mc0
    .private_segment_fixed_size: 0
    .sgpr_count:     54
    .sgpr_spill_count: 0
    .symbol:         bm256_bn256_bk128_wm2_wn4_mc0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     442
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
