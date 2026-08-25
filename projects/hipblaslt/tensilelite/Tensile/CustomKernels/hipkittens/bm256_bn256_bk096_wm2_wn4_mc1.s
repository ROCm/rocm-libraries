	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.section	.text.bm256_bn256_bk096_wm2_wn4_mc1,"axG",@progbits,bm256_bn256_bk096_wm2_wn4_mc1,comdat
	.protected	bm256_bn256_bk096_wm2_wn4_mc1 ; -- Begin function bm256_bn256_bk096_wm2_wn4_mc1
	.globl	bm256_bn256_bk096_wm2_wn4_mc1
	.p2align	8
	.type	bm256_bn256_bk096_wm2_wn4_mc1,@function
bm256_bn256_bk096_wm2_wn4_mc1: ; @bm256_bn256_bk096_wm2_wn4_mc1
; %bb.0:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1 ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_load_b96 s[24:26], s[0:1], 0x78 nv
	s_mov_b64 s[2:3], src_shared_base
	s_mov_b32 s2, 0xcc00
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_and_b64 s[2:3], s[2:3], 12
	s_sub_co_i32 s4, 16, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_lshr_b32 s4, s4, 2
	s_cmp_lg_u64 s[2:3], 0
	s_cselect_b32 s5, s4, 0
	s_and_b32 s2, ttmp6, 15
	s_bfe_u32 s4, ttmp6, 0x40004
	s_lshl2_add_u32 s3, ttmp9, s2
	s_lshl2_add_u32 s4, ttmp7, s4
	s_lshl_b32 s2, s3, 8
	s_wait_kmcnt 0x0
	s_add_co_i32 s6, s24, 0xff
	s_add_co_i32 s7, s25, 0xff
	s_sub_co_i32 s8, s24, s2
	s_ashr_i32 s9, s6, 31
	s_ashr_i32 s10, s7, 31
	s_min_i32 s27, s8, 0x100
	s_lshr_b32 s8, s9, 24
	s_lshr_b32 s9, s10, 24
	s_add_co_i32 s6, s6, s8
	s_add_co_i32 s7, s7, s9
	s_ashr_i32 s6, s6, 8
	s_ashr_i32 s7, s7, 8
	s_cmp_lt_i32 s3, s6
	s_mov_b32 s9, s26
	s_cselect_b32 s38, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_and_b32 s8, s38, exec_lo
	s_cselect_b32 s29, s27, 0
	s_lshl_b32 s33, s4, 8
	s_sub_co_i32 s8, s25, s33
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_min_i32 s8, s8, 0x100
	s_cmp_lt_i32 s4, s7
	s_cselect_b32 s25, -1, 0
	s_and_b32 s10, s25, exec_lo
	s_cselect_b32 s31, s8, 0
	s_add_co_i32 s12, s26, 0x5f
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 2, 1), 1
	s_min_i32 s8, s26, 0x60
	s_cmp_gt_i32 s12, 0x5f
	s_cselect_b32 s34, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_and_b32 s10, s34, exec_lo
	s_cselect_b32 s28, s8, 0
	s_cmp_lt_i32 s29, 0x100
	s_cselect_b32 s39, -1, 0
	s_and_b32 vcc_lo, exec_lo, s39
	s_mov_b32 s8, s39
	s_cbranch_vccnz .LBB0_2
; %bb.1:
	s_cmp_lt_i32 s31, 0x100
	s_cselect_b32 s8, -1, 0
	s_cmp_lt_i32 s28, 0x60
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
	v_cmp_lt_u32_e32 vcc_lo, 0x31ff, v5
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
	ds_store_b32 v2, v3 offset:52224
	v_add_nc_u32_e32 v2, 0x400, v2
	v_cmp_lt_u32_e32 vcc_lo, 0x31ff, v1
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
	s_load_b128 s[16:19], s[0:1], 0x20 nv
	s_load_b128 s[20:23], s[0:1], 0x48 nv
	s_wait_xcnt 0x0
	s_mov_b64 s[0:1], src_shared_base
	s_lshl_b32 s0, s5, 2
	s_add_co_i32 s7, s7, -1
	s_or_b32 s40, s0, 0xcc00
	s_min_i32 s0, s4, s7
	s_lshl_b32 s4, s4, 2
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_lshrrev_b32_e32 v11 /*v267*/, 5, v0
	s_and_b32 s4, s4, 12
	s_add_co_i32 s35, s6, -1
	s_and_b32 s37, s3, 3
	s_lshl_b32 s13, 15, s4
	s_mov_b32 s4, exec_lo
	s_set_vgpr_msb 0x4004                   ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_cmpx_lt_i32_e32 0, v11 /*v267*/
	s_xor_b32 s36, exec_lo, s4
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_cbranch_execz .LBB0_12
; %bb.9:
	s_mov_b32 s41, exec_lo
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_cmpx_eq_u32_e32 1, v11 /*v267*/
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_cbranch_execz .LBB0_11
; %bb.10:
	s_cmp_gt_i32 s28, 0
	s_mov_b32 s30, s28
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
	s_add_nc_u64 s[6:7], s[18:19], s[4:5]
	v_dual_mov_b32 v1, s40 :: v_dual_mov_b32 v4, s6
	s_and_b32 s5, s7, 0x1ffffff
	s_and_b32 s7, s25, s8
	s_bitset1_b32 s5, 31
	v_cndmask_b32_e64 v2, 0, 1, s7
	v_mov_b32_e32 v3, s5
	s_lshr_b64 s[6:7], s[30:31], 16
	v_readfirstlane_b32 s45, v1
	v_readfirstlane_b32 s46, v4
	v_readfirstlane_b32 s44, v2
	v_readfirstlane_b32 s47, v3
	s_lshr_b32 s7, s31, 16
	s_or_b32 s4, s13, 0x7510000
	s_lshl_b32 s5, s28, 16
	s_or_b32 s7, s7, 0x600000
	s_movk_i32 s8, 0x100
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[44:47], s[4:11]
.LBB0_11:
	s_or_b32 exec_lo, exec_lo, s41
.LBB0_12:
	s_or_saveexec_b32 s41, s36
	s_min_i32 s36, s3, s35
	s_lshl_b32 s30, 0x1111, s37
	s_xor_b32 exec_lo, exec_lo, s41
	s_cbranch_execz .LBB0_14
; %bb.13:
	s_cmp_gt_i32 s28, 0
	s_mov_b32 s10, 0
	s_cselect_b32 s3, -1, 0
	s_lshl_b32 s4, s36, 8
	s_wait_kmcnt 0x0
	s_bfe_i64 s[6:7], s[16:17], 0x200000
	s_ashr_i32 s5, s4, 31
	s_and_b32 s3, s38, s3
	s_mul_u64 s[4:5], s[6:7], s[4:5]
	v_cndmask_b32_e64 v2, 0, 1, s3
	s_lshl_b64 s[6:7], s[4:5], 1
	s_lshr_b32 s3, s29, 16
	s_add_nc_u64 s[6:7], s[14:15], s[6:7]
	s_or_b32 s4, s30, 0x7510000
	s_and_b32 s7, s7, 0x1ffffff
	v_readfirstlane_b32 s44, v2
	s_bitset1_b32 s7, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v4, s6 :: v_dual_mov_b32 v3, s7
	s_lshr_b64 s[6:7], s[28:29], 16
	s_lshl_b32 s5, s28, 16
	s_or_b32 s7, s3, 0x600000
	v_readfirstlane_b32 s46, v4
	v_readfirstlane_b32 s47, v3
	s_movk_i32 s8, 0x100
	s_mov_b32 s11, s10
	s_mov_b32 s45, s10
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[44:47], s[4:11]
.LBB0_14:
	s_or_b32 exec_lo, exec_lo, s41
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
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_dual_lshlrev_b32 v1, 6, v11 /*v267*/ :: v_dual_mov_b32 v9, 0
	s_and_b32 s3, s38, s25
	s_set_vgpr_msb 0x440                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_and_b32_e32 v7 /*v263*/, 0x80, v0
	v_cndmask_b32_e64 v3 /*v259*/, 0, 1, s3
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
	s_and_not1_b32 vcc_lo, exec_lo, s34
	s_barrier_wait -1
	s_barrier_wait -3
	s_cbranch_vccnz .LBB0_41
; %bb.17:
	s_mov_b64 s[4:5], src_shared_base
	s_add_co_i32 s6, s40, 0xcc00
	s_mov_b32 s7, s5
	s_mov_b32 s11, 0
	s_and_b64 s[6:7], s[6:7], 15
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_mul_u32_u24_e32 v3, 0x60, v7 /*v263*/
	s_sub_co_i32 s4, 16, s6
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_and_b32_e32 v5, 16, v0
	s_lshr_b32 s4, s4, 2
	s_cmp_lg_u64 s[6:7], 0
	s_mul_hi_i32 s6, s12, 0x2aaaaaab
	s_cselect_b32 s4, s4, 0
	v_or_b32_e32 v9, v3, v5
	s_lshl2_add_u32 s7, s4, s40
	v_and_b32_e32 v7, 15, v0
	s_add_co_i32 s4, s7, 0x19800
	s_add_co_i32 s42, s7, 0xcc00
	s_and_b32 s10, s4, 15
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_mul_u32_u24_e32 v1, 0x60, v9 /*v265*/
	s_sub_co_i32 s8, 16, s10
	s_set_vgpr_msb 0x440                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_or_b32_e32 v13 /*v269*/, 0x3100, v0
	s_lshr_b32 s7, s8, 2
	s_cmp_lg_u64 s[10:11], 0
	v_lshl_or_b32 v76 /*v332*/, v0, 2, 0xc800
	s_cselect_b32 s7, s7, 0
	s_lshr_b32 s8, s6, 31
	s_ashr_i32 s44, s6, 4
	s_lshl_b32 s10, s7, 2
	s_add_co_i32 s44, s44, s8
	s_cmp_lt_i32 s31, 0x100
	s_mov_b32 s41, s1
	s_cselect_b32 s45, -1, 0
	s_lshl_b32 s6, s0, 8
	s_movk_i32 s0, 0x600
	s_lshl_b32 s36, s36, 8
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_mad_u32_u24 v13, 0x60, v7, s0
	s_movk_i32 s0, 0xc00
	s_ashr_i32 s7, s6, 31
	v_mad_u32_u24 v15, 0x60, v7, s0
	s_movk_i32 s0, 0x1200
	s_ashr_i32 s37, s36, 31
	s_wait_kmcnt 0x0
	s_bfe_i64 s[20:21], s[20:21], 0x200000
	s_bfe_i64 s[16:17], s[16:17], 0x200000
	v_add_nc_u32_e32 v8, v9, v15
	v_mul_u32_u24_e32 v2, 0x60, v7
	v_mad_u32_u24 v17, 0x60, v7, s0
	v_or_b32_e32 v6, v3, v13
	s_mul_u64 s[6:7], s[20:21], s[6:7]
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_lshrrev_b32 v8, 4, v8 :: v_dual_bitop2_b32 v11, v9, v2 bitop3:0x54
	v_or_b32_e32 v19, 0x1800, v2
	s_mul_u64 s[16:17], s[16:17], s[36:37]
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_cmp_eq_u32_e64 s0, 0, v11 /*v267*/
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_and_b32_e32 v23, 0xff8, v8
	v_lshrrev_b32_e32 v4, 4, v11
	s_lshr_b32 s46, s31, 16
	s_lshr_b32 s47, s29, 16
	s_lshl_b64 s[6:7], s[6:7], 1
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_mov_b32_e32 v5 /*v261*/, 0
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_and_b32_e32 v4, 0x778, v4
	s_lshl_b64 s[16:17], s[16:17], 1
	s_mov_b32 s43, s5
	s_add_nc_u64 s[34:35], s[4:5], s[10:11]
	v_mov_b32_e32 v41, v5 /*v261*/
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_add_nc_u32_e32 v2 /*v258*/, v4, v11
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_dual_lshrrev_b32 v4, 4, v6 :: v_dual_add_nc_u32 v6, v9, v17
	v_add_nc_u32_e32 v8, 0x1e00, v11
	v_add_nc_u32_e32 v10, 0x2400, v11
	v_mad_u32_u24 v53, 0x60, v7, v9
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v21, 0x7f8, v4
	v_dual_lshrrev_b32 v4, 4, v6 :: v_dual_add_nc_u32 v6, v9, v19
	v_or_b32_e32 v27, v1, v5
	s_set_vgpr_msb 1                        ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_dual_mov_b32 v47, v5 /*v261*/ :: v_dual_add_nc_u32 v40, 0x600, v53
	s_delay_alu instid0(VALU_DEP_3)
	v_and_b32_e32 v25, 0xff8, v4
	v_dual_lshrrev_b32 v4, 4, v6 :: v_dual_lshrrev_b32 v6, 4, v8
	v_lshrrev_b32_e32 v8, 4, v10
	v_add_nc_u32_e32 v10, 0x2a00, v11
	s_set_vgpr_msb 0x100                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_or_b32_e32 v2, v27, v2
	v_and_b32_e32 v29, 0xff8, v4
	v_and_b32_e32 v31, 0xff8, v6
	v_and_b32_e32 v33, 0xff8, v8
	v_dual_lshrrev_b32 v4, 4, v10 :: v_dual_bitop2_b32 v10, 32, v5 bitop3:0x54
	v_dual_add_nc_u32 v8, v27, v13 :: v_dual_lshrrev_b32 v6, 4, v2
	v_add_nc_u32_e32 v12, v27, v15
	s_delay_alu instid0(VALU_DEP_3)
	v_and_b32_e32 v35, 0xff8, v4
	v_add_nc_u32_e32 v14, v27, v17
	v_mad_u32_u24 v54, 0x60, v7, v27
	v_and_b32_e32 v4, 0x7f8, v6
	v_dual_lshrrev_b32 v6, 4, v8 :: v_dual_bitop2_b32 v8, v10, v3 bitop3:0x54
	v_lshrrev_b32_e32 v12, 4, v12
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_dual_add_nc_u32 v10 /*v266*/, v23, v53 :: v_dual_add_nc_u32 v12 /*v268*/, v25, v53
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_and_b32_e32 v37, 0xff8, v6
	v_mad_u32_u24 v6, 0x60, v7, v8
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_dual_add_nc_u32 v6 /*v262*/, v4, v2 :: v_dual_add_nc_u32 v8 /*v264*/, v21, v53
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_and_b32_e32 v39, 0xff8, v12
	v_dual_lshrrev_b32 v2, 4, v14 :: v_dual_add_nc_u32 v4, v8, v13
	v_dual_add_nc_u32 v12, v8, v15 :: v_dual_lshrrev_b32 v14, 4, v6
	v_add_nc_u32_e32 v16, v8, v17
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_and_b32_e32 v42, 0xff8, v2
	v_lshrrev_b32_e32 v2, 4, v4
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_lshrrev_b32 v4, 4, v12 :: v_dual_add_nc_u32 v8, v8, v19
	v_and_b32_e32 v43, 0xff8, v14
	v_dual_lshrrev_b32 v12, 4, v16 :: v_dual_bitop2_b32 v16, v1, v10 bitop3:0x54
	v_add_nc_u32_e32 v10, 0x2400, v6
	v_or_b32_e32 v5, 64, v5
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_dual_add_nc_u32 v28 /*v284*/, v42, v54 :: v_dual_add_nc_u32 v30 /*v286*/, v43, v53
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_dual_mov_b32 v43, v5 /*v261*/ :: v_dual_lshrrev_b32 v10, 4, v10
	s_set_vgpr_msb 0x100                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_or_b32_e32 v3, v3, v5
	v_add_nc_u32_e32 v14, 0x1e00, v6
	v_or_b32_e32 v1, v1, v5
	s_set_vgpr_msb 1                        ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_mov_b32_e32 v45, v5 /*v261*/
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_and_b32_e32 v4 /*v260*/, 0xff8, v2
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_mad_u32_u24 v32, 0x60, v7, v3
	v_and_b32_e32 v2, 0xff8, v4
	v_and_b32_e32 v4, 0xff8, v12
	v_mad_u32_u24 v18, 0x60, v7, v16
	v_mad_u32_u24 v5, 0x60, v7, v1
	v_dual_mov_b32 v7, v5 /*v261*/ :: v_dual_add_nc_u32 v26, 0x2400, v32
	v_dual_lshrrev_b32 v8, 4, v8 :: v_dual_lshrrev_b32 v12, 4, v14
	v_dual_mov_b32 v9, v5 /*v261*/ :: v_dual_add_nc_u32 v14, 0x2a00, v6
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshrrev_b32_e32 v30, 4, v26
	v_and_b32_e32 v6, 0xff8, v8
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_and_b32_e32 v8, 0x1ff8, v12
	v_lshrrev_b32_e32 v12, 4, v14
	s_set_vgpr_msb 0x100                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_dual_lshrrev_b32 v14, 4, v18 :: v_dual_add_nc_u32 v18, v16, v13
	s_set_vgpr_msb 0x41                     ;  msbs: dst=1 src0=1 src1=0 src2=0
	v_add_nc_u64_e32 v[22:23] /*v[278:279]*/, v[4:5] /*v[260:261]*/, v[40:41]
	v_dual_mov_b32 v77 /*v333*/, v5 /*v261*/ :: v_dual_add_nc_u32 v4 /*v260*/, 0xc00, v53
	s_set_vgpr_msb 0x4101                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_dual_mov_b32 v25, v5 /*v261*/ :: v_dual_lshrrev_b32 v18, 4, v18
	v_and_b32_e32 v50, 0xff8, v14
	s_set_vgpr_msb 0x100                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_dual_add_nc_u32 v14, v16, v15 :: v_dual_add_nc_u32 v16, v16, v17
	v_lshrrev_b32_e32 v24, 4, v32
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_dual_add_nc_u32 v14 /*v270*/, v29, v53 :: v_dual_add_nc_u32 v16 /*v272*/, v31, v11
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_dual_lshrrev_b32 v22, 4, v14 :: v_dual_add_nc_u32 v20, v3, v13
	v_and_b32_e32 v14, 0xff8, v18
	s_set_vgpr_msb 1                        ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_dual_mov_b32 v29, v5 /*v261*/ :: v_dual_lshrrev_b32 v18, 4, v16
	s_delay_alu instid0(VALU_DEP_3)
	v_and_b32_e32 v16, 0xff8, v22
	s_set_vgpr_msb 0x100                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_dual_add_nc_u32 v22, v3, v15 :: v_dual_lshrrev_b32 v20, 4, v20
	v_and_b32_e32 v51, 0xff8, v24
	v_dual_add_nc_u32 v24, v3, v17 :: v_dual_add_nc_u32 v3, v3, v19
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_lshrrev_b32_e32 v22, 4, v22
	v_add_nc_u32_e32 v19, 0x1e00, v32
	v_dual_lshrrev_b32 v5, 4, v5 :: v_dual_lshrrev_b32 v24, 4, v24
	s_delay_alu instid0(VALU_DEP_4)
	v_lshrrev_b32_e32 v3, 4, v3
	v_add_nc_u32_e32 v15, v1, v15
	s_set_vgpr_msb 1                        ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_dual_mov_b32 v49, v5 /*v261*/ :: v_dual_add_nc_u32 v42, 0x1800, v53
	v_and_b32_e32 v24, 0xff8, v24
	v_and_b32_e32 v26, 0xff8, v3
	v_add_nc_u32_e32 v3, 0x2a00, v32
	v_and_b32_e32 v52, 0xff8, v5
	v_mov_b32_e32 v5, v5 /*v261*/
	s_set_vgpr_msb 0x100                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_add_nc_u32_e32 v13, v1, v13
	v_dual_add_nc_u32 v1, v1, v17 :: v_dual_lshrrev_b32 v3, 4, v3
	v_add_nc_u32_e32 v44, 0x1e00, v53
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_lshrrev_b32 v15, 4, v15 :: v_dual_lshrrev_b32 v13, 4, v13
	v_lshrrev_b32_e32 v1, 4, v1
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v32, 0x1ff8, v3
	s_set_vgpr_msb 1                        ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_mov_b32_e32 v3, v5 /*v261*/
	v_and_b32_e32 v10, 0x1ff8, v10
	v_and_b32_e32 v12, 0x1ff8, v12
	v_and_b32_e32 v18, 0xff8, v18
	v_and_b32_e32 v20, 0xff8, v20
	s_set_vgpr_msb 0x144                    ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_add_nc_u64_e32 v[32:33] /*v[288:289]*/, v[2:3], v[4:5] /*v[260:261]*/
	s_set_vgpr_msb 0x4400                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_add_nc_u32_e32 v2, 0x1200, v53
	v_and_b32_e32 v22, 0xff8, v22
	v_and_b32_e32 v30, 0x1ff8, v30
	v_and_b32_e32 v34, 0xff8, v13
	v_and_b32_e32 v36, 0xff8, v15
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_add_nc_u64_e32 v[34:35] /*v[290:291]*/, v[4:5], v[2:3]
	v_add_nc_u64_e32 v[58:59] /*v[314:315]*/, v[24:25], v[2:3]
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_dual_mov_b32 v2, v5 /*v261*/ :: v_dual_lshrrev_b32 v19, 4, v19
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_dual_add_nc_u32 v18 /*v274*/, v33, v11 :: v_dual_add_nc_u32 v20 /*v276*/, v35, v11
	v_dual_add_nc_u32 v24 /*v280*/, v37, v54 :: v_dual_add_nc_u32 v26 /*v282*/, v39, v54
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_and_b32_e32 v28, 0x1ff8, v19
	v_mov_b32_e32 v11, v5 /*v261*/
	v_and_b32_e32 v38, 0xff8, v1
	v_dual_mov_b32 v17, v5 /*v261*/ :: v_dual_add_nc_u32 v46, 0x2400, v53
	v_dual_mov_b32 v13, v5 /*v261*/ :: v_dual_add_nc_u32 v48, 0x2a00, v53
	v_mov_b32_e32 v19, v5 /*v261*/
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_add_nc_u64_e32 v[36:37] /*v[292:293]*/, v[6:7], v[42:43]
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_mov_b32_e32 v15, v5 /*v261*/
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_add_nc_u64_e32 v[38:39] /*v[294:295]*/, v[8:9], v[44:45]
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_dual_mov_b32 v21, v5 /*v261*/ :: v_dual_add_nc_u32 v4, 0x600, v54
	v_dual_mov_b32 v23, v5 /*v261*/ :: v_dual_add_nc_u32 v6, 0xc00, v54
	v_dual_mov_b32 v27, v5 /*v261*/ :: v_dual_add_nc_u32 v8, 0x1200, v54
	v_dual_mov_b32 v31, v5 /*v261*/ :: v_dual_mov_b32 v33, v5 /*v261*/
	v_dual_mov_b32 v35, v5 /*v261*/ :: v_dual_mov_b32 v37, v5 /*v261*/
	v_mov_b32_e32 v39, v5 /*v261*/
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_add_nc_u64_e32 v[40:41] /*v[296:297]*/, v[10:11], v[46:47]
	v_add_nc_u64_e32 v[42:43] /*v[298:299]*/, v[12:13], v[48:49]
	v_add_nc_u64_e32 v[46:47] /*v[302:303]*/, v[14:15], v[4:5]
	v_add_nc_u64_e32 v[48:49] /*v[304:305]*/, v[16:17], v[6:7]
	v_add_nc_u64_e32 v[50:51] /*v[306:307]*/, v[18:19], v[8:9]
	v_add_nc_u64_e32 v[54:55] /*v[310:311]*/, v[20:21], v[40:41]
	s_set_vgpr_msb 0x4044                   ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_add_nc_u64_e32 v[56:57] /*v[312:313]*/, v[22:23], v[4:5] /*v[260:261]*/
	s_set_vgpr_msb 0x4440                   ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_add_nc_u64_e32 v[60:61] /*v[316:317]*/, v[26:27], v[42:43]
	v_add_nc_u64_e32 v[62:63] /*v[318:319]*/, v[28:29], v[44:45]
	v_add_nc_u64_e32 v[64:65] /*v[320:321]*/, v[30:31], v[46:47]
	v_add_nc_u64_e32 v[66:67] /*v[322:323]*/, v[32:33], v[48:49]
	v_add_nc_u64_e32 v[70:71] /*v[326:327]*/, v[34:35], v[4:5]
	v_add_nc_u64_e32 v[72:73] /*v[328:329]*/, v[36:37], v[6:7]
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_mov_b32_e32 v6, v5 /*v261*/
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_add_nc_u64_e32 v[74:75] /*v[330:331]*/, v[38:39], v[8:9]
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_or_b32_e32 v1, 0x100, v0
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_dual_add_nc_u32 v44 /*v300*/, v50, v54 :: v_dual_add_nc_u32 v52 /*v308*/, v51, v53
	v_add_nc_u32_e32 v68 /*v324*/, v52, v54
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_dual_mov_b32 v4, v5 /*v261*/ :: v_dual_mov_b32 v8, v5 /*v261*/
	v_dual_mov_b32 v10, v5 /*v261*/ :: v_dual_mov_b32 v12, v5 /*v261*/
	v_dual_mov_b32 v14, v5 /*v261*/ :: v_dual_mov_b32 v16, v5 /*v261*/
	v_dual_mov_b32 v50, v5 /*v261*/ :: v_dual_mov_b32 v51, v5 /*v261*/
	v_dual_mov_b32 v52, v5 /*v261*/ :: v_dual_mov_b32 v53, v5 /*v261*/
	v_dual_mov_b32 v54, v5 /*v261*/ :: v_dual_mov_b32 v55, v5 /*v261*/
	v_dual_mov_b32 v56, v5 /*v261*/ :: v_dual_mov_b32 v57, v5 /*v261*/
	v_dual_mov_b32 v18, v5 /*v261*/ :: v_dual_mov_b32 v20, v5 /*v261*/
	v_dual_mov_b32 v22, v5 /*v261*/ :: v_dual_mov_b32 v24, v5 /*v261*/
	v_dual_mov_b32 v26, v5 /*v261*/ :: v_dual_mov_b32 v28, v5 /*v261*/
	v_dual_mov_b32 v30, v5 /*v261*/ :: v_dual_mov_b32 v32, v5 /*v261*/
	v_dual_mov_b32 v34, v5 /*v261*/ :: v_dual_mov_b32 v36, v5 /*v261*/
	v_dual_mov_b32 v38, v5 /*v261*/ :: v_dual_mov_b32 v40, v5 /*v261*/
	v_dual_mov_b32 v74, v5 /*v261*/ :: v_dual_mov_b32 v75, v5 /*v261*/
	v_dual_mov_b32 v76, v5 /*v261*/ :: v_dual_mov_b32 v77, v5 /*v261*/
	v_dual_mov_b32 v78, v5 /*v261*/ :: v_dual_mov_b32 v79, v5 /*v261*/
	v_dual_mov_b32 v80, v5 /*v261*/ :: v_dual_mov_b32 v81, v5 /*v261*/
	v_dual_mov_b32 v42, v5 /*v261*/ :: v_dual_mov_b32 v44, v5 /*v261*/
	v_dual_mov_b32 v46, v5 /*v261*/ :: v_dual_mov_b32 v48, v5 /*v261*/
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
	v_mov_b32_e32 v154, v5 /*v261*/
	s_set_vgpr_msb 0x141                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	v_dual_mov_b32 v0 /*v256*/, v5 /*v261*/ :: v_dual_mov_b32 v1 /*v257*/, v5 /*v261*/
	s_set_vgpr_msb 0x4101                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_dual_mov_b32 v155, v5 /*v261*/ :: v_dual_mov_b32 v156, v5 /*v261*/
	v_dual_mov_b32 v157, v5 /*v261*/ :: v_dual_mov_b32 v158, v5 /*v261*/
	v_dual_mov_b32 v159, v5 /*v261*/ :: v_dual_mov_b32 v160, v5 /*v261*/
	v_mov_b32_e32 v161, v5 /*v261*/
	s_movk_i32 s8, 0x100
	s_or_b32 s12, s13, 0x7510000
	s_or_b32 s46, s46, 0x600000
	s_or_b32 s4, s30, 0x7510000
	s_or_b32 s47, s47, 0x600000
	s_add_nc_u64 s[20:21], s[18:19], s[6:7]
	s_add_nc_u64 s[36:37], s[14:15], s[16:17]
	s_mov_b32 s48, -1
	s_mov_b32 s49, s11
	s_set_vgpr_msb 0x100                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_branch .LBB0_19
.LBB0_18:                               ;   in Loop: Header=BB0_19 Depth=1
	s_or_b32 exec_lo, exec_lo, s5
	s_cmp_eq_u32 s49, s44
	s_barrier_wait -1
	s_barrier_wait -3
	s_cbranch_scc1 .LBB0_41
.LBB0_19:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_21 Depth 2
                                        ;     Child Loop BB0_24 Depth 2
                                        ;     Child Loop BB0_26 Depth 2
                                        ;     Child Loop BB0_29 Depth 2
	s_and_b32 s50, s49, 1
	s_add_co_i32 s49, s49, 1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_i32 s5, s49, 0xffffffa0
	s_add_co_i32 s6, s5, s26
	s_xor_b32 s5, s50, 1
	s_min_i32 s6, s6, 0x60
	s_cmp_lt_i32 s49, s44
	s_cselect_b32 s10, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_and_b32 s7, s10, exec_lo
	s_cselect_b32 s28, s6, 0
	s_cmp_lt_i32 s28, 0x60
	s_cselect_b32 s6, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b32 s6, s45, s6
	s_or_b32 s6, s39, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s6
	s_cbranch_vccnz .LBB0_31
; %bb.20:                               ;   in Loop: Header=BB0_19 Depth=1
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_mov_b64_e32 v[78:79] /*v[334:335]*/, v[0:1]
	v_mov_b32_e32 v15 /*v271*/, 50
	s_cmp_lg_u32 s5, 0
	s_mov_b32 s13, 0
	s_cselect_b32 s7, s43, s1
	s_cselect_b32 s6, s42, 0
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
.LBB0_21:                               ;   Parent Loop BB0_19 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_set_vgpr_msb 0x45                     ;  msbs: dst=1 src0=1 src1=1 src2=0
	v_dual_mov_b32 v4 /*v260*/, v78 /*v334*/ :: v_dual_add_nc_u32 v15 /*v271*/, -2, v15 /*v271*/
	v_dual_mov_b32 v80 /*v336*/, v79 /*v335*/ :: v_dual_mov_b32 v81 /*v337*/, v5 /*v261*/
	v_add_nc_u32_e32 v79 /*v335*/, 0x200, v79 /*v335*/
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u64 v[82:83] /*v[338:339]*/, v[4:5] /*v[260:261]*/, 2, s[6:7]
	v_cmp_eq_u32_e32 vcc_lo, 0, v15 /*v271*/
	v_add_nc_u32_e32 v78 /*v334*/, 0x200, v78 /*v334*/
	v_lshl_add_u64 v[80:81] /*v[336:337]*/, v[80:81] /*v[336:337]*/, 2, s[6:7]
	s_clause 0x1
	flat_store_b32 v[82:83] /*v[338:339]*/, v5 /*v261*/
	flat_store_b32 v[80:81] /*v[336:337]*/, v5 /*v261*/
	s_or_b32 s13, vcc_lo, s13
	s_wait_xcnt 0x0
	s_and_not1_b32 exec_lo, exec_lo, s13
	s_set_vgpr_msb 0x4500                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_cbranch_execnz .LBB0_21
; %bb.22:                               ;   in Loop: Header=BB0_19 Depth=1
	s_or_b32 exec_lo, exec_lo, s13
	s_and_saveexec_b32 s13, s48
	s_cbranch_execz .LBB0_25
; %bb.23:                               ;   in Loop: Header=BB0_19 Depth=1
	s_set_vgpr_msb 0x45                     ;  msbs: dst=1 src0=1 src1=1 src2=0
	v_add_nc_u64_e32 v[78:79] /*v[334:335]*/, s[6:7], v[76:77] /*v[332:333]*/
	v_mov_b32_e32 v4 /*v260*/, v13 /*v269*/
	s_mov_b32 s6, 0
	s_set_vgpr_msb 0x4500                   ;  msbs: dst=0 src0=0 src1=0 src2=0
.LBB0_24:                               ;   Parent Loop BB0_19 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_set_vgpr_msb 0x45                     ;  msbs: dst=1 src0=1 src1=1 src2=0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v4 /*v260*/, 0x100, v4 /*v260*/
	flat_store_b32 v[78:79] /*v[334:335]*/, v5 /*v261*/
	s_wait_xcnt 0x0
	v_add_nc_u64_e32 v[78:79] /*v[334:335]*/, 0x400, v[78:79] /*v[334:335]*/
	v_cmp_lt_u32_e32 vcc_lo, 0x31ff, v4 /*v260*/
	s_or_b32 s6, vcc_lo, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s6
	s_set_vgpr_msb 0x4500                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_cbranch_execnz .LBB0_24
.LBB0_25:                               ;   in Loop: Header=BB0_19 Depth=1
	s_or_b32 exec_lo, exec_lo, s13
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_mov_b64_e32 v[78:79] /*v[334:335]*/, v[0:1]
	v_mov_b32_e32 v15 /*v271*/, 50
	s_cmp_lg_u32 s5, 0
	s_mov_b32 s13, 0
	s_cselect_b32 s7, s35, s41
	s_cselect_b32 s6, s34, s40
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
.LBB0_26:                               ;   Parent Loop BB0_19 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_set_vgpr_msb 0x45                     ;  msbs: dst=1 src0=1 src1=1 src2=0
	v_dual_mov_b32 v4 /*v260*/, v78 /*v334*/ :: v_dual_add_nc_u32 v15 /*v271*/, -2, v15 /*v271*/
	v_dual_mov_b32 v80 /*v336*/, v79 /*v335*/ :: v_dual_mov_b32 v81 /*v337*/, v5 /*v261*/
	v_add_nc_u32_e32 v79 /*v335*/, 0x200, v79 /*v335*/
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u64 v[82:83] /*v[338:339]*/, v[4:5] /*v[260:261]*/, 2, s[6:7]
	v_cmp_eq_u32_e32 vcc_lo, 0, v15 /*v271*/
	v_add_nc_u32_e32 v78 /*v334*/, 0x200, v78 /*v334*/
	v_lshl_add_u64 v[80:81] /*v[336:337]*/, v[80:81] /*v[336:337]*/, 2, s[6:7]
	s_clause 0x1
	flat_store_b32 v[82:83] /*v[338:339]*/, v5 /*v261*/
	flat_store_b32 v[80:81] /*v[336:337]*/, v5 /*v261*/
	s_or_b32 s13, vcc_lo, s13
	s_wait_xcnt 0x0
	s_and_not1_b32 exec_lo, exec_lo, s13
	s_set_vgpr_msb 0x4500                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_cbranch_execnz .LBB0_26
; %bb.27:                               ;   in Loop: Header=BB0_19 Depth=1
	s_or_b32 exec_lo, exec_lo, s13
	s_and_saveexec_b32 s13, s48
	s_cbranch_execz .LBB0_30
; %bb.28:                               ;   in Loop: Header=BB0_19 Depth=1
	s_set_vgpr_msb 0x45                     ;  msbs: dst=1 src0=1 src1=1 src2=0
	v_add_nc_u64_e32 v[78:79] /*v[334:335]*/, s[6:7], v[76:77] /*v[332:333]*/
	v_mov_b32_e32 v4 /*v260*/, v13 /*v269*/
	s_mov_b32 s6, 0
	s_set_vgpr_msb 0x4500                   ;  msbs: dst=0 src0=0 src1=0 src2=0
.LBB0_29:                               ;   Parent Loop BB0_19 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_set_vgpr_msb 0x45                     ;  msbs: dst=1 src0=1 src1=1 src2=0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v4 /*v260*/, 0x100, v4 /*v260*/
	flat_store_b32 v[78:79] /*v[334:335]*/, v5 /*v261*/
	s_wait_xcnt 0x0
	v_add_nc_u64_e32 v[78:79] /*v[334:335]*/, 0x400, v[78:79] /*v[334:335]*/
	v_cmp_lt_u32_e32 vcc_lo, 0x31ff, v4 /*v260*/
	s_or_b32 s6, vcc_lo, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s6
	s_set_vgpr_msb 0x4500                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_cbranch_execnz .LBB0_29
.LBB0_30:                               ;   in Loop: Header=BB0_19 Depth=1
	s_or_b32 exec_lo, exec_lo, s13
	s_barrier_signal -1
	s_barrier_wait -1
.LBB0_31:                               ;   in Loop: Header=BB0_19 Depth=1
	s_and_b32 s6, s10, exec_lo
	s_cselect_b32 s6, s49, 0
	s_mov_b32 s7, exec_lo
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_cmpx_lt_i32_e32 0, v11 /*v267*/
	s_xor_b32 s7, exec_lo, s7
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
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
	s_cmp_lg_u32 s50, 0
	s_cselect_b32 s6, s42, 0
	s_cselect_b32 s5, s34, s40
	s_set_vgpr_msb 0x41                     ;  msbs: dst=1 src0=1 src1=0 src2=0
	v_lshl_add_u32 v4 /*v260*/, v2 /*v258*/, 1, s6
	v_lshl_add_u32 v15 /*v271*/, v8 /*v264*/, 1, s6
	v_lshl_add_u32 v17 /*v273*/, v10 /*v266*/, 1, s6
	v_lshl_add_u32 v19 /*v275*/, v28 /*v284*/, 1, s5
	ds_load_b128 v[78:81] /*v[334:337]*/, v4 /*v260*/
	ds_load_b128 v[82:85] /*v[338:341]*/, v4 /*v260*/ offset:16
	v_lshl_add_u32 v4 /*v260*/, v20 /*v276*/, 1, s6
	ds_load_b128 v[86:89] /*v[342:345]*/, v15 /*v271*/ offset:3072
	ds_load_b128 v[90:93] /*v[346:349]*/, v15 /*v271*/ offset:3088
	ds_load_b128 v[94:97] /*v[350:353]*/, v17 /*v273*/ offset:6144
	ds_load_b128 v[98:101] /*v[354:357]*/, v17 /*v273*/ offset:6160
	ds_load_b128 v[110:113] /*v[366:369]*/, v4 /*v260*/ offset:21504
	ds_load_b128 v[114:117] /*v[370:373]*/, v4 /*v260*/ offset:21520
	v_lshl_add_u32 v4 /*v260*/, v18 /*v274*/, 1, s6
	ds_load_b128 v[102:105] /*v[358:361]*/, v19 /*v275*/ offset:9216
	ds_load_b128 v[106:109] /*v[362:365]*/, v19 /*v275*/ offset:9232
	v_lshl_add_u32 v15 /*v271*/, v22 /*v278*/, 1, s6
	v_lshl_add_u32 v17 /*v273*/, v34 /*v290*/, 1, s6
	ds_load_b128 v[118:121] /*v[374:377]*/, v4 /*v260*/ offset:18432
	ds_load_b128 v[122:125] /*v[378:381]*/, v4 /*v260*/ offset:18448
	v_lshl_add_u32 v4 /*v260*/, v16 /*v272*/, 1, s6
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	s_set_vgpr_msb 0x4105                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[138:145], v[94:101] /*v[350:357]*/, v[102:109] /*v[358:365]*/, v[138:145] matrix_b_reuse
	s_set_vgpr_msb 0x541                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	ds_load_b128 v[150:153] /*v[406:409]*/, v17 /*v273*/ offset:64
	ds_load_b128 v[154:157] /*v[410:413]*/, v17 /*v273*/ offset:80
	v_lshl_add_u32 v17 /*v273*/, v44 /*v300*/, 1, s5
	ds_load_b128 v[174:177] /*v[430:433]*/, v17 /*v273*/ offset:64
	s_set_vgpr_msb 0x4105                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[194:201], v[86:93] /*v[342:349]*/, v[102:109] /*v[358:365]*/, v[194:201] matrix_b_reuse
	s_set_vgpr_msb 0x541                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	ds_load_b128 v[178:181] /*v[434:437]*/, v17 /*v273*/ offset:80
	v_lshl_add_u32 v17 /*v273*/, v50 /*v306*/, 1, s5
	ds_load_b128 v[190:193] /*v[446:449]*/, v17 /*v273*/ offset:64
	s_set_vgpr_msb 0x4105                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[226:233], v[78:85] /*v[334:341]*/, v[102:109] /*v[358:365]*/, v[226:233] matrix_b_reuse
	s_set_vgpr_msb 0x541                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	ds_load_b128 v[194:197] /*v[450:453]*/, v17 /*v273*/ offset:80
	s_set_vgpr_msb 0x4105                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[110:117] /*v[366:373]*/, v[102:109] /*v[358:365]*/, v[154:161] matrix_a_reuse
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[18:25], v[118:125] /*v[374:381]*/, v[102:109] /*v[358:365]*/, v[18:25] matrix_b_reuse
	s_set_vgpr_msb 0x541                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	ds_load_b128 v[126:129] /*v[382:385]*/, v4 /*v260*/ offset:15360
	ds_load_b128 v[130:133] /*v[386:389]*/, v4 /*v260*/ offset:15376
	v_lshl_add_u32 v4 /*v260*/, v14 /*v270*/, 1, s6
	ds_load_b128 v[134:137] /*v[390:393]*/, v4 /*v260*/ offset:12288
	ds_load_b128 v[138:141] /*v[394:397]*/, v4 /*v260*/ offset:12304
	v_lshl_add_u32 v4 /*v260*/, v12 /*v268*/, 1, s6
	ds_load_b128 v[142:145] /*v[398:401]*/, v4 /*v260*/ offset:9216
	ds_load_b128 v[146:149] /*v[402:405]*/, v4 /*v260*/ offset:9232
	v_lshl_add_u32 v4 /*v260*/, v26 /*v282*/, 1, s5
	s_set_vgpr_msb 0x4105                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[42:49], v[126:133] /*v[382:389]*/, v[102:109] /*v[358:365]*/, v[42:49] matrix_b_reuse
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[82:89], v[134:141] /*v[390:397]*/, v[102:109] /*v[358:365]*/, v[82:89] matrix_b_reuse
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[114:121], v[142:149] /*v[398:405]*/, v[102:109] /*v[358:365]*/, v[114:121] matrix_b_reuse
	s_set_vgpr_msb 0x541                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	ds_load_b128 v[102:105] /*v[358:361]*/, v4 /*v260*/ offset:6144
	ds_load_b128 v[106:109] /*v[362:365]*/, v4 /*v260*/ offset:6160
	v_lshl_add_u32 v4 /*v260*/, v6 /*v262*/, 1, s5
	s_set_vgpr_msb 0x4105                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[234:241], v[78:85] /*v[334:341]*/, v[102:109] /*v[358:365]*/, v[234:241] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[202:209], v[86:93] /*v[342:349]*/, v[102:109] /*v[358:365]*/, v[202:209] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[170:177], v[94:101] /*v[350:357]*/, v[102:109] /*v[358:365]*/, v[170:177] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[122:129], v[142:149] /*v[398:405]*/, v[102:109] /*v[358:365]*/, v[122:129] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[90:97], v[134:141] /*v[390:397]*/, v[102:109] /*v[358:365]*/, v[90:97] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[58:65], v[126:133] /*v[382:389]*/, v[102:109] /*v[358:365]*/, v[58:65] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[26:33], v[118:125] /*v[374:381]*/, v[102:109] /*v[358:365]*/, v[26:33] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[2:9], v[110:117] /*v[366:373]*/, v[102:109] /*v[358:365]*/, v[2:9] matrix_b_reuse
	s_set_vgpr_msb 0x541                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	ds_load_b128 v[102:105] /*v[358:361]*/, v4 /*v260*/
	ds_load_b128 v[106:109] /*v[362:365]*/, v4 /*v260*/ offset:16
	v_lshl_add_u32 v4 /*v260*/, v24 /*v280*/, 1, s5
	s_set_vgpr_msb 0x4105                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[106:113], v[126:133] /*v[382:389]*/, v[102:109] /*v[358:365]*/, v[106:113] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[74:81], v[118:125] /*v[374:381]*/, v[102:109] /*v[358:365]*/, v[74:81] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[50:57], v[110:117] /*v[366:373]*/, v[102:109] /*v[358:365]*/, v[50:57] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[250:257], v[78:85] /*v[334:341]*/, v[102:109] /*v[358:365]*/, v[250:257]
	v_wmma_f32_16x16x32_bf16 v[218:225], v[86:93] /*v[342:349]*/, v[102:109] /*v[358:365]*/, v[218:225] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[186:193], v[94:101] /*v[350:357]*/, v[102:109] /*v[358:365]*/, v[186:193] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[162:169], v[142:149] /*v[398:405]*/, v[102:109] /*v[358:365]*/, v[162:169] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[146:153], v[134:141] /*v[390:397]*/, v[102:109] /*v[358:365]*/, v[146:153] matrix_b_reuse
	s_set_vgpr_msb 0x541                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	ds_load_b128 v[102:105] /*v[358:361]*/, v4 /*v260*/ offset:3072
	ds_load_b128 v[106:109] /*v[362:365]*/, v4 /*v260*/ offset:3088
	v_lshl_add_u32 v4 /*v260*/, v30 /*v286*/, 1, s6
	s_set_vgpr_msb 0x4105                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[10:17], v[110:117] /*v[366:373]*/, v[102:109] /*v[358:365]*/, v[10:17] matrix_a_reuse
	s_set_vgpr_msb 0x541                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	ds_load_b128 v[110:113] /*v[366:369]*/, v4 /*v260*/ offset:64
	ds_load_b128 v[114:117] /*v[370:373]*/, v4 /*v260*/ offset:80
	v_lshl_add_u32 v4 /*v260*/, v32 /*v288*/, 1, s6
	s_set_vgpr_msb 0x4105                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[34:41], v[118:125] /*v[374:381]*/, v[102:109] /*v[358:365]*/, v[34:41] matrix_b_reuse
	s_set_vgpr_msb 0x541                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	ds_load_b128 v[118:121] /*v[374:377]*/, v15 /*v271*/ offset:64
	ds_load_b128 v[122:125] /*v[378:381]*/, v15 /*v271*/ offset:80
	v_lshl_add_u32 v15 /*v271*/, v38 /*v294*/, 1, s6
	ds_load_b128 v[158:161] /*v[414:417]*/, v15 /*v271*/ offset:64
	s_set_vgpr_msb 0x4105                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[66:73], v[126:133] /*v[382:389]*/, v[102:109] /*v[358:365]*/, v[66:73] matrix_b_reuse
	s_set_vgpr_msb 0x541                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	ds_load_b128 v[126:129] /*v[382:385]*/, v4 /*v260*/ offset:64
	ds_load_b128 v[130:133] /*v[386:389]*/, v4 /*v260*/ offset:80
	v_lshl_add_u32 v4 /*v260*/, v36 /*v292*/, 1, s6
	ds_load_b128 v[162:165] /*v[418:421]*/, v15 /*v271*/ offset:80
	v_lshl_add_u32 v15 /*v271*/, v42 /*v298*/, 1, s6
	ds_load_b128 v[166:169] /*v[422:425]*/, v15 /*v271*/ offset:64
	s_set_vgpr_msb 0x4105                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[98:105], v[134:141] /*v[390:397]*/, v[102:109] /*v[358:365]*/, v[98:105] matrix_b_reuse
	s_set_vgpr_msb 0x541                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	ds_load_b128 v[134:137] /*v[390:393]*/, v4 /*v260*/ offset:64
	ds_load_b128 v[138:141] /*v[394:397]*/, v4 /*v260*/ offset:80
	v_lshl_add_u32 v4 /*v260*/, v40 /*v296*/, 1, s6
	ds_load_b128 v[170:173] /*v[426:429]*/, v15 /*v271*/ offset:80
	v_lshl_add_u32 v15 /*v271*/, v48 /*v304*/, 1, s5
	ds_load_b128 v[182:185] /*v[438:441]*/, v15 /*v271*/ offset:64
	s_set_vgpr_msb 0x4105                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[130:137], v[142:149] /*v[398:405]*/, v[102:109] /*v[358:365]*/, v[130:137] matrix_b_reuse
	s_set_vgpr_msb 0x541                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	ds_load_b128 v[142:145] /*v[398:401]*/, v4 /*v260*/ offset:64
	ds_load_b128 v[146:149] /*v[402:405]*/, v4 /*v260*/ offset:80
	v_lshl_add_u32 v4 /*v260*/, v46 /*v302*/, 1, s5
	ds_load_b128 v[186:189] /*v[442:445]*/, v15 /*v271*/ offset:80
	s_set_vgpr_msb 0x4105                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[178:185], v[94:101] /*v[350:357]*/, v[102:109] /*v[358:365]*/, v[178:185] matrix_b_reuse
	s_set_vgpr_msb 0x541                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	ds_load_b128 v[94:97] /*v[350:353]*/, v4 /*v260*/ offset:64
	ds_load_b128 v[98:101] /*v[354:357]*/, v4 /*v260*/ offset:80
	s_set_vgpr_msb 0x4105                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[210:217], v[86:93] /*v[342:349]*/, v[102:109] /*v[358:365]*/, v[210:217] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[242:249], v[78:85] /*v[334:341]*/, v[102:109] /*v[358:365]*/, v[242:249] matrix_b_reuse
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x541                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	v_lshl_add_u32 v4 /*v260*/, v52 /*v308*/, 1, s6
	v_lshl_add_u32 v15 /*v271*/, v54 /*v310*/, 1, s6
	v_lshl_add_u32 v17 /*v273*/, v56 /*v312*/, 1, s6
	ds_load_b128 v[78:81] /*v[334:337]*/, v4 /*v260*/ offset:128
	ds_load_b128 v[82:85] /*v[338:341]*/, v4 /*v260*/ offset:144
	ds_load_b128 v[86:89] /*v[342:345]*/, v15 /*v271*/ offset:128
	ds_load_b128 v[90:93] /*v[346:349]*/, v15 /*v271*/ offset:144
	ds_load_b128 v[102:105] /*v[358:361]*/, v17 /*v273*/ offset:128
	ds_load_b128 v[106:109] /*v[362:365]*/, v17 /*v273*/ offset:144
	v_lshl_add_u32 v4 /*v260*/, v58 /*v314*/, 1, s6
	v_lshl_add_u32 v15 /*v271*/, v60 /*v316*/, 1, s6
	v_lshl_add_u32 v17 /*v273*/, v62 /*v318*/, 1, s6
	s_set_vgpr_msb 0x4105                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[250:257], v[110:117] /*v[366:373]*/, v[174:181] /*v[430:437]*/, v[250:257]
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	s_wait_dscnt 0x14
	v_wmma_f32_16x16x32_bf16 v[218:225], v[118:125] /*v[374:381]*/, v[174:181] /*v[430:437]*/, v[218:225] matrix_b_reuse
	s_wait_dscnt 0x11
	v_wmma_f32_16x16x32_bf16 v[186:193], v[126:133] /*v[382:389]*/, v[174:181] /*v[430:437]*/, v[186:193] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[162:169], v[150:157] /*v[406:413]*/, v[174:181] /*v[430:437]*/, v[162:169] matrix_b_reuse
	s_wait_dscnt 0xd
	v_wmma_f32_16x16x32_bf16 v[146:153], v[134:141] /*v[390:397]*/, v[174:181] /*v[430:437]*/, v[146:153] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[106:113], v[158:165] /*v[414:421]*/, v[174:181] /*v[430:437]*/, v[106:113] matrix_b_reuse
	s_wait_dscnt 0x9
	v_wmma_f32_16x16x32_bf16 v[74:81], v[142:149] /*v[398:405]*/, v[174:181] /*v[430:437]*/, v[74:81] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[50:57], v[166:173] /*v[422:429]*/, v[174:181] /*v[430:437]*/, v[50:57] matrix_b_reuse
	s_set_vgpr_msb 0x541                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	ds_load_b128 v[174:177] /*v[430:433]*/, v4 /*v260*/ offset:128
	ds_load_b128 v[178:181] /*v[434:437]*/, v4 /*v260*/ offset:144
	ds_load_b128 v[198:201] /*v[454:457]*/, v15 /*v271*/ offset:128
	ds_load_b128 v[202:205] /*v[458:461]*/, v15 /*v271*/ offset:144
	ds_load_b128 v[206:209] /*v[462:465]*/, v17 /*v273*/ offset:128
	ds_load_b128 v[210:213] /*v[466:469]*/, v17 /*v273*/ offset:144
	v_lshl_add_u32 v4 /*v260*/, v64 /*v320*/, 1, s6
	v_lshl_add_u32 v15 /*v271*/, v66 /*v322*/, 1, s6
	v_lshl_add_u32 v17 /*v273*/, v68 /*v324*/, 1, s5
	s_set_vgpr_msb 0x4105                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[10:17], v[166:173] /*v[422:429]*/, v[94:101] /*v[350:357]*/, v[10:17] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[34:41], v[142:149] /*v[398:405]*/, v[94:101] /*v[350:357]*/, v[34:41] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[66:73], v[158:165] /*v[414:421]*/, v[94:101] /*v[350:357]*/, v[66:73] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[98:105], v[134:141] /*v[390:397]*/, v[94:101] /*v[350:357]*/, v[98:105] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[130:137], v[150:157] /*v[406:413]*/, v[94:101] /*v[350:357]*/, v[130:137] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[178:185], v[126:133] /*v[382:389]*/, v[94:101] /*v[350:357]*/, v[178:185] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[210:217], v[118:125] /*v[374:381]*/, v[94:101] /*v[350:357]*/, v[210:217] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[242:249], v[110:117] /*v[366:373]*/, v[94:101] /*v[350:357]*/, v[242:249] matrix_b_reuse
	s_set_vgpr_msb 0x541                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	ds_load_b128 v[94:97] /*v[350:353]*/, v4 /*v260*/ offset:128
	ds_load_b128 v[98:101] /*v[354:357]*/, v4 /*v260*/ offset:144
	ds_load_b128 v[214:217] /*v[470:473]*/, v15 /*v271*/ offset:128
	ds_load_b128 v[218:221] /*v[474:477]*/, v15 /*v271*/ offset:144
	ds_load_b128 v[222:225] /*v[478:481]*/, v17 /*v273*/ offset:128
	ds_load_b128 v[226:229] /*v[482:485]*/, v17 /*v273*/ offset:144
	v_lshl_add_u32 v4 /*v260*/, v70 /*v326*/, 1, s5
	v_lshl_add_u32 v15 /*v271*/, v72 /*v328*/, 1, s5
	v_lshl_add_u32 v17 /*v273*/, v74 /*v330*/, 1, s5
	s_set_vgpr_msb 0x4105                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[234:241], v[110:117] /*v[366:373]*/, v[182:189] /*v[438:445]*/, v[234:241] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[202:209], v[118:125] /*v[374:381]*/, v[182:189] /*v[438:445]*/, v[202:209] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[170:177], v[126:133] /*v[382:389]*/, v[182:189] /*v[438:445]*/, v[170:177] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[122:129], v[150:157] /*v[406:413]*/, v[182:189] /*v[438:445]*/, v[122:129] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[90:97], v[134:141] /*v[390:397]*/, v[182:189] /*v[438:445]*/, v[90:97] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[58:65], v[158:165] /*v[414:421]*/, v[182:189] /*v[438:445]*/, v[58:65] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[26:33], v[142:149] /*v[398:405]*/, v[182:189] /*v[438:445]*/, v[26:33] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[2:9], v[166:173] /*v[422:429]*/, v[182:189] /*v[438:445]*/, v[2:9] matrix_b_reuse
	s_set_vgpr_msb 0x541                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	ds_load_b128 v[182:185] /*v[438:441]*/, v4 /*v260*/ offset:128
	ds_load_b128 v[186:189] /*v[442:445]*/, v4 /*v260*/ offset:144
	ds_load_b128 v[230:233] /*v[486:489]*/, v15 /*v271*/ offset:128
	ds_load_b128 v[234:237] /*v[490:493]*/, v15 /*v271*/ offset:144
	ds_load_b128 v[238:241] /*v[494:497]*/, v17 /*v273*/ offset:128
	ds_load_b128 v[242:245] /*v[498:501]*/, v17 /*v273*/ offset:144
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	s_set_vgpr_msb 0x4105                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[166:173] /*v[422:429]*/, v[190:197] /*v[446:453]*/, v[154:161] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[18:25], v[142:149] /*v[398:405]*/, v[190:197] /*v[446:453]*/, v[18:25] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[42:49], v[158:165] /*v[414:421]*/, v[190:197] /*v[446:453]*/, v[42:49] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[82:89], v[134:141] /*v[390:397]*/, v[190:197] /*v[446:453]*/, v[82:89] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[114:121], v[150:157] /*v[406:413]*/, v[190:197] /*v[446:453]*/, v[114:121] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[138:145], v[126:133] /*v[382:389]*/, v[190:197] /*v[446:453]*/, v[138:145] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[194:201], v[118:125] /*v[374:381]*/, v[190:197] /*v[446:453]*/, v[194:201] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[226:233], v[110:117] /*v[366:373]*/, v[190:197] /*v[446:453]*/, v[226:233] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[250:257], v[78:85] /*v[334:341]*/, v[222:229] /*v[478:485]*/, v[250:257]
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[218:225], v[86:93] /*v[342:349]*/, v[222:229] /*v[478:485]*/, v[218:225] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[186:193], v[102:109] /*v[358:365]*/, v[222:229] /*v[478:485]*/, v[186:193] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[162:169], v[174:181] /*v[430:437]*/, v[222:229] /*v[478:485]*/, v[162:169] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[146:153], v[198:205] /*v[454:461]*/, v[222:229] /*v[478:485]*/, v[146:153] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[106:113], v[206:213] /*v[462:469]*/, v[222:229] /*v[478:485]*/, v[106:113] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[74:81], v[94:101] /*v[350:357]*/, v[222:229] /*v[478:485]*/, v[74:81] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[50:57], v[214:221] /*v[470:477]*/, v[222:229] /*v[478:485]*/, v[50:57] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[10:17], v[214:221] /*v[470:477]*/, v[182:189] /*v[438:445]*/, v[10:17] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[34:41], v[94:101] /*v[350:357]*/, v[182:189] /*v[438:445]*/, v[34:41] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[66:73], v[206:213] /*v[462:469]*/, v[182:189] /*v[438:445]*/, v[66:73] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[98:105], v[198:205] /*v[454:461]*/, v[182:189] /*v[438:445]*/, v[98:105] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[130:137], v[174:181] /*v[430:437]*/, v[182:189] /*v[438:445]*/, v[130:137] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[178:185], v[102:109] /*v[358:365]*/, v[182:189] /*v[438:445]*/, v[178:185] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[210:217], v[86:93] /*v[342:349]*/, v[182:189] /*v[438:445]*/, v[210:217] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[242:249], v[78:85] /*v[334:341]*/, v[182:189] /*v[438:445]*/, v[242:249] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[234:241], v[78:85] /*v[334:341]*/, v[230:237] /*v[486:493]*/, v[234:241] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[202:209], v[86:93] /*v[342:349]*/, v[230:237] /*v[486:493]*/, v[202:209] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[170:177], v[102:109] /*v[358:365]*/, v[230:237] /*v[486:493]*/, v[170:177] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[122:129], v[174:181] /*v[430:437]*/, v[230:237] /*v[486:493]*/, v[122:129] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[90:97], v[198:205] /*v[454:461]*/, v[230:237] /*v[486:493]*/, v[90:97] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[58:65], v[206:213] /*v[462:469]*/, v[230:237] /*v[486:493]*/, v[58:65] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[26:33], v[94:101] /*v[350:357]*/, v[230:237] /*v[486:493]*/, v[26:33] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[2:9], v[214:221] /*v[470:477]*/, v[230:237] /*v[486:493]*/, v[2:9] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[214:221] /*v[470:477]*/, v[238:245] /*v[494:501]*/, v[154:161] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[18:25], v[94:101] /*v[350:357]*/, v[238:245] /*v[494:501]*/, v[18:25] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[42:49], v[206:213] /*v[462:469]*/, v[238:245] /*v[494:501]*/, v[42:49] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[82:89], v[198:205] /*v[454:461]*/, v[238:245] /*v[494:501]*/, v[82:89] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[114:121], v[174:181] /*v[430:437]*/, v[238:245] /*v[494:501]*/, v[114:121] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[138:145], v[102:109] /*v[358:365]*/, v[238:245] /*v[494:501]*/, v[138:145] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[194:201], v[86:93] /*v[342:349]*/, v[238:245] /*v[494:501]*/, v[194:201] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[226:233], v[78:85] /*v[334:341]*/, v[238:245] /*v[494:501]*/, v[226:233] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
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
	s_mov_b32 s51, exec_lo
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_cmpx_eq_u32_e32 1, v11 /*v267*/
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_cbranch_execz .LBB0_39
; %bb.38:                               ;   in Loop: Header=BB0_19 Depth=1
	s_cmp_lg_u32 s5, 0
	s_mul_i32 s10, s6, 0x60
	s_cselect_b32 s13, s34, s40
	s_cmp_gt_i32 s28, 0
	s_mov_b32 s30, s28
	s_cselect_b32 s16, -1, 0
	s_lshl_b64 s[14:15], s[10:11], 1
	s_mov_b32 s17, s9
	s_add_nc_u64 s[14:15], s[20:21], s[14:15]
	v_nop
	v_nop
	s_set_vgpr_msb 0x41                     ;  msbs: dst=1 src0=1 src1=0 src2=0
	v_dual_mov_b32 v15 /*v271*/, s13 :: v_dual_mov_b32 v78 /*v334*/, s14
	s_and_b32 s10, s15, 0x1ffffff
	s_and_b32 s15, s25, s16
	s_bitset1_b32 s10, 31
	v_cndmask_b32_e64 v4 /*v260*/, 0, 1, s15
	v_mov_b32_e32 v17 /*v273*/, s10
	v_readfirstlane_b32 s53, v15 /*v271*/
	v_readfirstlane_b32 s54, v78 /*v334*/
	s_lshr_b64 s[14:15], s[30:31], 16
	v_readfirstlane_b32 s52, v4 /*v260*/
	v_readfirstlane_b32 s55, v17 /*v273*/
	s_lshl_b32 s13, s28, 16
	s_mov_b32 s15, s46
	s_mov_b32 s16, s8
	s_mov_b32 s18, s11
	s_mov_b32 s19, s11
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[52:55], s[12:19]
	s_set_vgpr_msb 0x4100                   ;  msbs: dst=0 src0=0 src1=0 src2=0
.LBB0_39:                               ;   in Loop: Header=BB0_19 Depth=1
	s_or_b32 exec_lo, exec_lo, s51
	s_and_not1_saveexec_b32 s13, s7
	s_cbranch_execz .LBB0_33
.LBB0_40:                               ;   in Loop: Header=BB0_19 Depth=1
	s_cmp_lg_u32 s5, 0
	s_mul_i32 s10, s6, 0x60
	s_cselect_b32 s5, s42, 0
	s_cmp_gt_i32 s28, 0
	s_cselect_b32 s14, -1, 0
	s_lshl_b64 s[6:7], s[10:11], 1
	s_and_b32 s10, s38, s14
	s_add_nc_u64 s[6:7], s[36:37], s[6:7]
	s_set_vgpr_msb 0x41                     ;  msbs: dst=1 src0=1 src1=0 src2=0
	v_cndmask_b32_e64 v4 /*v260*/, 0, 1, s10
	s_and_b32 s7, s7, 0x1ffffff
	v_nop
	v_nop
	v_dual_mov_b32 v15 /*v271*/, s5 :: v_dual_mov_b32 v78 /*v334*/, s6
	s_bitset1_b32 s7, 31
	v_readfirstlane_b32 s16, v4 /*v260*/
	v_mov_b32_e32 v17 /*v273*/, s7
	s_delay_alu instid0(VALU_DEP_3)
	v_readfirstlane_b32 s17, v15 /*v271*/
	v_readfirstlane_b32 s18, v78 /*v334*/
	s_lshr_b64 s[6:7], s[28:29], 16
	s_lshl_b32 s5, s28, 16
	v_readfirstlane_b32 s19, v17 /*v273*/
	s_mov_b32 s7, s47
	s_mov_b32 s10, s11
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	tensor_load_to_lds s[16:19], s[4:11]
	s_or_b32 exec_lo, exec_lo, s13
	s_and_not1_b32 vcc_lo, exec_lo, s3
	s_set_vgpr_msb 0x4100                   ;  msbs: dst=0 src0=0 src1=0 src2=0
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
.LBB0_43:
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_cmp_ne_u32_e32 vcc_lo, 1, v3 /*v259*/
	;;#ASMSTART
	s_wait_dscnt 0
	;;#ASMEND
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_vccnz .LBB0_54
; %bb.44:
	s_wait_kmcnt 0x0
	s_mul_i32 s14, s31, s29
	s_mov_b32 s0, exec_lo
	v_cmpx_gt_i32_e64 s14, v0
	s_cbranch_execz .LBB0_54
; %bb.45:
	s_ashr_i32 s3, s2, 31
	v_xad_u32 v2, v0, -1, s14
	s_lshl_b64 s[0:1], s[2:3], 1
	s_ashr_i32 s25, s24, 31
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
	v_dual_add_nc_u32 v22, s20, v26 :: v_dual_sub_nc_u32 v12, v3, v12
	v_ashrrev_i32_e32 v21, 31, v20
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mul_lo_u32 v6, v1, s27
	v_dual_sub_nc_u32 v27, v19, v18 :: v_dual_add_nc_u32 v18, s33, v1
	v_sub_nc_u32_e32 v14, v4, v14
	v_ashrrev_i32_e32 v23, 31, v22
	v_lshl_add_u32 v11, v11, 8, v12
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_mul_lo_u32 v16, v27, s18
	v_dual_add_nc_u32 v24, s21, v27 :: v_dual_sub_nc_u32 v6, v2, v6
	v_ashrrev_i32_e32 v19, 31, v18
	v_lshl_add_u32 v26, v26, 8, v14
	v_mul_u64_e32 v[20:21], s[6:7], v[20:21]
	v_ashrrev_i32_e32 v25, 31, v24
	v_lshl_add_u32 v1, v1, 8, v6
	v_sub_nc_u32_e32 v16, v5, v16
	v_mul_u64_e32 v[18:19], s[24:25], v[18:19]
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
	s_or_b32 s23, vcc_lo, s23
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
	s_and_not1_b32 exec_lo, exec_lo, s23
	s_cbranch_execnz .LBB0_49
; %bb.50:
	s_or_b32 exec_lo, exec_lo, s23
	v_cmp_ne_u32_e32 vcc_lo, v8, v9
	v_lshl_or_b32 v0, v9, 8, v0
	v_dual_mov_b32 v6, s15 :: v_dual_mov_b32 v1, s22
	s_and_b32 s0, vcc_lo, exec_lo
	s_or_saveexec_b32 s1, s3
	v_mov_b64_e32 v[2:3], s[12:13]
	s_xor_b32 exec_lo, exec_lo, s1
	s_cbranch_execz .LBB0_47
.LBB0_51:
	s_abs_i32 s6, s27
	s_ashr_i32 s7, s27, 31
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
	s_sub_co_i32 s1, 0, s27
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
	.size	bm256_bn256_bk096_wm2_wn4_mc1, .Lfunc_end0-bm256_bn256_bk096_wm2_wn4_mc1
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel bm256_bn256_bk096_wm2_wn4_mc1
		.amdhsa_group_segment_fixed_size 208896
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
		.amdhsa_next_free_vgpr 502
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
		.amdhsa_inst_pref_size 92
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text.bm256_bn256_bk096_wm2_wn4_mc1,"axG",@progbits,bm256_bn256_bk096_wm2_wn4_mc1,comdat
                                        ; -- End function
	.set .Lbm256_bn256_bk096_wm2_wn4_mc1.num_vgpr, 502
	.set .Lbm256_bn256_bk096_wm2_wn4_mc1.num_agpr, 0
	.set .Lbm256_bn256_bk096_wm2_wn4_mc1.numbered_sgpr, 56
	.set .Lbm256_bn256_bk096_wm2_wn4_mc1.num_named_barrier, 0
	.set .Lbm256_bn256_bk096_wm2_wn4_mc1.private_seg_size, 0
	.set .Lbm256_bn256_bk096_wm2_wn4_mc1.uses_vcc, 1
	.set .Lbm256_bn256_bk096_wm2_wn4_mc1.uses_flat_scratch, 1
	.set .Lbm256_bn256_bk096_wm2_wn4_mc1.has_dyn_sized_stack, 0
	.set .Lbm256_bn256_bk096_wm2_wn4_mc1.has_recursion, 0
	.set .Lbm256_bn256_bk096_wm2_wn4_mc1.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 11676
; TotalNumSgprs: 58
; NumVgprs: 502
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 208896 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 31
; NumSGPRsForWavesPerEU: 58
; NumVGPRsForWavesPerEU: 502
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
	.type	__hip_cuid_7893e7e1c7abac6d,@object ; @__hip_cuid_7893e7e1c7abac6d
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_7893e7e1c7abac6d
__hip_cuid_7893e7e1c7abac6d:
	.byte	0                               ; 0x0
	.size	__hip_cuid_7893e7e1c7abac6d, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git aa451e1fe6a793394d6733051b1778633063ae96+PATCHED:d17c5aa0e3ea29cde402f58f27e39b6034effa27)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_7893e7e1c7abac6d
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
    .group_segment_fixed_size: 208896
    .kernarg_segment_align: 8
    .kernarg_segment_size: 132
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           bm256_bn256_bk096_wm2_wn4_mc1
    .private_segment_fixed_size: 0
    .sgpr_count:     58
    .sgpr_spill_count: 0
    .symbol:         bm256_bn256_bk096_wm2_wn4_mc1.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     502
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
