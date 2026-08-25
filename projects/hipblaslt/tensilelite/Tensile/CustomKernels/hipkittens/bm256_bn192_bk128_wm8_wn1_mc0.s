	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.section	.text.bm256_bn192_bk128_wm8_wn1_mc0,"axG",@progbits,bm256_bn192_bk128_wm8_wn1_mc0,comdat
	.protected	bm256_bn192_bk128_wm8_wn1_mc0 ; -- Begin function bm256_bn192_bk128_wm8_wn1_mc0
	.globl	bm256_bn192_bk128_wm8_wn1_mc0
	.p2align	8
	.type	bm256_bn192_bk128_wm8_wn1_mc0,@function
bm256_bn192_bk128_wm8_wn1_mc0: ; @bm256_bn192_bk128_wm8_wn1_mc0
; %bb.0:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1 ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_mov_b64 s[2:3], src_shared_base
	s_mov_b32 s2, 0x11000
	s_load_b96 s[20:22], s[0:1], 0x78 nv
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
	s_cselect_b32 s35, ttmp9, s4
	s_cselect_b32 s33, ttmp7, s5
	s_wait_kmcnt 0x0
	s_add_co_i32 s3, s20, 0xff
	s_add_co_i32 s5, s21, 0xbf
	s_ashr_i32 s4, s3, 31
	s_lshl_b32 s24, s35, 8
	s_lshr_b32 s4, s4, 24
	s_sub_co_i32 s6, s20, s24
	s_add_co_i32 s3, s3, s4
	s_mul_hi_i32 s4, s5, 0x2aaaaaab
	s_ashr_i32 s3, s3, 8
	s_lshr_b32 s5, s4, 31
	s_ashr_i32 s4, s4, 5
	s_min_i32 s23, s6, 0x100
	s_add_co_i32 s4, s4, s5
	s_cmp_lt_i32 s35, s3
	s_cselect_b32 s25, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_and_b32 s5, s25, exec_lo
	s_mul_i32 s5, s33, 0xffffff40
	s_cselect_b32 s27, s23, 0
	s_add_co_i32 s5, s21, s5
	s_min_i32 s6, s5, 0xc0
	s_cmp_lt_i32 s33, s4
	s_mov_b32 s5, s22
	s_cselect_b32 s38, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s7, s38, exec_lo
	s_cselect_b32 s29, s6, 0
	s_add_co_i32 s13, s22, 0x7f
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 2, 1), 1
	s_min_i32 s6, s22, 0x80
	s_cmp_gt_i32 s13, 0x7f
	s_cselect_b32 s12, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_and_b32 s7, s12, exec_lo
	s_cselect_b32 s26, s6, 0
	s_cmp_lt_i32 s27, 0x100
	s_cselect_b32 s39, -1, 0
	s_and_b32 vcc_lo, exec_lo, s39
	s_mov_b32 s6, s39
	s_cbranch_vccnz .LBB0_2
; %bb.1:
	s_cmp_lt_i32 s29, 0xc0
	s_cselect_b32 s6, -1, 0
	s_cmp_lt_i32 s26, 0x80
	s_cselect_b32 s7, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 s6, s7, s6
.LBB0_2:
	v_lshlrev_b32_e32 v194, 2, v0
	s_and_not1_b32 vcc_lo, exec_lo, s6
	s_cbranch_vccnz .LBB0_8
; %bb.3:
	v_or_b32_e32 v1, 0xffffff00, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v3, v194
	s_mov_b32 s6, 0
	v_mov_b32_e32 v4, v1
.LBB0_4:                                ; =>This Inner Loop Header: Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)
	v_add_nc_u32_e32 v4, 0x100, v4
	ds_store_b32 v3, v2
	v_add_nc_u32_e32 v3, 0x400, v3
	v_cmp_lt_u32_e32 vcc_lo, 0x42ff, v4
	s_or_b32 s6, vcc_lo, s6
	s_and_not1_b32 exec_lo, exec_lo, s6
	s_cbranch_execnz .LBB0_4
; %bb.5:
	s_or_b32 exec_lo, exec_lo, s6
	v_lshl_add_u32 v2, s2, 2, v194
	v_mov_b32_e32 v3, 0
	s_mov_b32 s6, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_or_b32_e32 v2, 0x11000, v2
.LBB0_6:                                ; =>This Inner Loop Header: Depth=1
	v_add_nc_u32_e32 v1, 0x100, v1
	ds_store_b32 v2, v3
	v_add_nc_u32_e32 v2, 0x400, v2
	v_cmp_lt_u32_e32 vcc_lo, 0x31ff, v1
	s_or_b32 s6, vcc_lo, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s6
	s_cbranch_execnz .LBB0_6
; %bb.7:
	s_or_b32 exec_lo, exec_lo, s6
	s_barrier_signal -1
	s_barrier_wait -1
.LBB0_8:
	s_clause 0x2
	s_load_b64 s[14:15], s[0:1], 0x0 nv
	s_load_b128 s[8:11], s[0:1], 0x20 nv
	s_load_b128 s[16:19], s[0:1], 0x48 nv
	v_lshrrev_b32_e32 v199, 5, v0
	s_add_co_i32 s4, s4, -1
	s_mov_b64 s[30:31], src_shared_base
	s_lshl2_add_u32 s30, s2, 0x11000
	s_add_co_i32 s21, s3, -1
	s_min_i32 s34, s33, s4
	s_wait_xcnt 0x0
	s_mov_b32 s0, exec_lo
	v_cmpx_lt_i32_e32 0, v199
	s_xor_b32 s36, exec_lo, s0
	s_cbranch_execz .LBB0_12
; %bb.9:
	s_mov_b32 s37, exec_lo
	v_cmpx_eq_u32_e32 1, v199
	s_cbranch_execz .LBB0_11
; %bb.10:
	s_cmp_gt_i32 s26, 0
	s_mul_i32 s0, s34, 0xc0
	s_cselect_b32 s4, -1, 0
	s_ashr_i32 s1, s0, 31
	s_wait_kmcnt 0x0
	s_bfe_i64 s[2:3], s[16:17], 0x200000
	s_mov_b32 s28, s26
	s_mul_u64 s[0:1], s[2:3], s[0:1]
	s_and_b32 s2, s38, s4
	s_lshl_b64 s[0:1], s[0:1], 1
	v_cndmask_b32_e64 v2, 0, 1, s2
	s_add_nc_u64 s[0:1], s[10:11], s[0:1]
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_mov_b32 v1, s30 :: v_dual_mov_b32 v4, s0
	s_and_b32 s1, s1, 0x1ffffff
	v_readfirstlane_b32 s40, v2
	s_bitset1_b32 s1, 31
	v_readfirstlane_b32 s41, v1
	v_mov_b32_e32 v3, s1
	v_readfirstlane_b32 s42, v4
	s_mov_b32 s6, 0
	s_lshr_b32 s0, s29, 16
	s_lshr_b64 s[2:3], s[28:29], 16
	v_readfirstlane_b32 s43, v3
	s_movk_i32 s4, 0xc0
	s_lshl_b32 s1, s26, 16
	s_or_b32 s3, s0, 0x800000
	s_mov_b32 s0, 0x7510000
	s_mov_b32 s7, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[40:43], s[0:7]
.LBB0_11:
	s_or_b32 exec_lo, exec_lo, s37
.LBB0_12:
	s_or_saveexec_b32 s36, s36
	s_min_i32 s28, s35, s21
	s_xor_b32 exec_lo, exec_lo, s36
	s_cbranch_execz .LBB0_14
; %bb.13:
	s_cmp_gt_i32 s26, 0
	s_mov_b32 s6, 0
	s_cselect_b32 s4, -1, 0
	s_lshl_b32 s0, s28, 8
	s_wait_kmcnt 0x0
	s_bfe_i64 s[2:3], s[8:9], 0x200000
	s_ashr_i32 s1, s0, 31
	s_and_b32 s4, s25, s4
	s_mul_u64 s[0:1], s[2:3], s[0:1]
	v_cndmask_b32_e64 v2, 0, 1, s4
	s_lshl_b64 s[2:3], s[0:1], 1
	s_lshr_b32 s0, s27, 16
	s_add_nc_u64 s[2:3], s[14:15], s[2:3]
	s_lshl_b32 s1, s26, 16
	s_and_b32 s3, s3, 0x1ffffff
	v_readfirstlane_b32 s40, v2
	s_bitset1_b32 s3, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v4, s2 :: v_dual_mov_b32 v3, s3
	s_lshr_b64 s[2:3], s[26:27], 16
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
	s_or_b32 exec_lo, exec_lo, s36
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	v_mov_b32_e32 v9, 0
	s_and_b32 s21, s25, s38
	s_and_not1_b32 vcc_lo, exec_lo, s12
	v_cndmask_b32_e64 v197, 0, 1, s21
	s_delay_alu instid0(VALU_DEP_2)
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
	v_dual_mov_b32 v186, v9 :: v_dual_mov_b32 v113, v9
	v_dual_mov_b32 v112, v9 :: v_dual_mov_b32 v111, v9
	v_dual_mov_b32 v110, v9 :: v_dual_mov_b32 v109, v9
	v_dual_mov_b32 v108, v9 :: v_dual_mov_b32 v107, v9
	v_mov_b32_e32 v106, v9
	s_barrier_wait -1
	s_cbranch_vccnz .LBB0_37
; %bb.15:
	s_mov_b64 s[0:1], src_shared_base
	s_add_co_i32 s2, s30, 0xcc00
	s_mov_b32 s3, s1
	v_lshlrev_b32_e32 v1, 7, v0
	s_and_b64 s[2:3], s[2:3], 15
	s_mov_b32 s7, 0
	s_sub_co_i32 s0, 16, s2
	s_mov_b32 s42, s1
	s_lshr_b32 s0, s0, 2
	s_cmp_lg_u64 s[2:3], 0
	v_and_b32_e32 v1, 0x780, v1
	s_cselect_b32 s0, s0, 0
	s_mul_i32 s2, s34, 0xc0
	s_lshl2_add_u32 s3, s0, s30
	v_mov_b32_e32 v195, 0
	s_add_co_i32 s0, s3, 0x1dc00
	v_and_or_b32 v2, v0, 16, v1
	s_and_b32 s6, s0, 15
	s_add_co_i32 s41, s3, 0xcc00
	s_sub_co_i32 s4, 16, s6
	v_lshrrev_b32_e32 v1, 4, v1
	s_lshr_b32 s3, s4, 2
	s_cmp_lg_u64 s[6:7], 0
	v_lshl_or_b32 v3, v199, 12, v2
	s_cselect_b32 s3, s3, 0
	s_ashr_i32 s4, s13, 31
	s_lshl_b32 s6, s3, 2
	s_lshr_b32 s4, s4, 25
	v_lshrrev_b32_e32 v4, 4, v3
	s_add_co_i32 s13, s13, s4
	s_add_nc_u64 s[34:35], s[0:1], s[6:7]
	s_ashr_i32 s43, s13, 7
	s_cmp_lt_i32 s29, 0xc0
	v_and_b32_e32 v4, 0x778, v4
	s_cselect_b32 s44, -1, 0
	s_lshl_b32 s0, s28, 8
	s_ashr_i32 s3, s2, 31
	s_ashr_i32 s1, s0, 31
	s_wait_kmcnt 0x0
	s_bfe_i64 s[16:17], s[16:17], 0x200000
	s_bfe_i64 s[8:9], s[8:9], 0x200000
	s_mul_u64 s[2:3], s[16:17], s[2:3]
	s_mul_u64 s[0:1], s[8:9], s[0:1]
	v_dual_add_nc_u32 v196, v4, v3 :: v_dual_add_nc_u32 v198, v2, v1
	v_or_b32_e32 v1, 0x100, v0
	v_or_b32_e32 v206, 0x4300, v0
	v_or_b32_e32 v200, 0x11000, v194
	v_dual_mov_b32 v201, v195 :: v_dual_mov_b32 v203, v195
	v_or_b32_e32 v207, 0x3100, v0
	v_or_b32_e32 v202, 0xc800, v194
	v_dual_mov_b32 v2, v195 :: v_dual_mov_b32 v3, v195
	v_dual_mov_b32 v4, v195 :: v_dual_mov_b32 v5, v195
	v_dual_mov_b32 v6, v195 :: v_dual_mov_b32 v7, v195
	v_dual_mov_b32 v8, v195 :: v_dual_mov_b32 v9, v195
	v_dual_mov_b32 v10, v195 :: v_dual_mov_b32 v11, v195
	v_dual_mov_b32 v12, v195 :: v_dual_mov_b32 v13, v195
	v_dual_mov_b32 v14, v195 :: v_dual_mov_b32 v15, v195
	v_dual_mov_b32 v16, v195 :: v_dual_mov_b32 v17, v195
	v_dual_mov_b32 v18, v195 :: v_dual_mov_b32 v19, v195
	v_dual_mov_b32 v20, v195 :: v_dual_mov_b32 v21, v195
	v_dual_mov_b32 v22, v195 :: v_dual_mov_b32 v23, v195
	v_dual_mov_b32 v24, v195 :: v_dual_mov_b32 v25, v195
	v_dual_mov_b32 v26, v195 :: v_dual_mov_b32 v27, v195
	v_dual_mov_b32 v28, v195 :: v_dual_mov_b32 v29, v195
	v_dual_mov_b32 v30, v195 :: v_dual_mov_b32 v31, v195
	v_dual_mov_b32 v32, v195 :: v_dual_mov_b32 v33, v195
	v_dual_mov_b32 v34, v195 :: v_dual_mov_b32 v35, v195
	v_dual_mov_b32 v36, v195 :: v_dual_mov_b32 v37, v195
	v_dual_mov_b32 v38, v195 :: v_dual_mov_b32 v39, v195
	v_dual_mov_b32 v40, v195 :: v_dual_mov_b32 v41, v195
	v_dual_mov_b32 v42, v195 :: v_dual_mov_b32 v43, v195
	v_dual_mov_b32 v44, v195 :: v_dual_mov_b32 v45, v195
	v_dual_mov_b32 v46, v195 :: v_dual_mov_b32 v47, v195
	v_dual_mov_b32 v48, v195 :: v_dual_mov_b32 v49, v195
	v_dual_mov_b32 v50, v195 :: v_dual_mov_b32 v51, v195
	v_dual_mov_b32 v52, v195 :: v_dual_mov_b32 v53, v195
	v_dual_mov_b32 v54, v195 :: v_dual_mov_b32 v55, v195
	v_dual_mov_b32 v56, v195 :: v_dual_mov_b32 v57, v195
	v_dual_mov_b32 v58, v195 :: v_dual_mov_b32 v59, v195
	v_dual_mov_b32 v60, v195 :: v_dual_mov_b32 v61, v195
	v_dual_mov_b32 v62, v195 :: v_dual_mov_b32 v63, v195
	v_dual_mov_b32 v64, v195 :: v_dual_mov_b32 v65, v195
	v_dual_mov_b32 v66, v195 :: v_dual_mov_b32 v67, v195
	v_dual_mov_b32 v68, v195 :: v_dual_mov_b32 v69, v195
	v_dual_mov_b32 v70, v195 :: v_dual_mov_b32 v71, v195
	v_dual_mov_b32 v72, v195 :: v_dual_mov_b32 v73, v195
	v_dual_mov_b32 v74, v195 :: v_dual_mov_b32 v75, v195
	v_dual_mov_b32 v76, v195 :: v_dual_mov_b32 v77, v195
	v_dual_mov_b32 v78, v195 :: v_dual_mov_b32 v79, v195
	v_dual_mov_b32 v80, v195 :: v_dual_mov_b32 v81, v195
	v_dual_mov_b32 v82, v195 :: v_dual_mov_b32 v83, v195
	v_dual_mov_b32 v84, v195 :: v_dual_mov_b32 v85, v195
	v_dual_mov_b32 v86, v195 :: v_dual_mov_b32 v87, v195
	v_dual_mov_b32 v88, v195 :: v_dual_mov_b32 v89, v195
	v_dual_mov_b32 v90, v195 :: v_dual_mov_b32 v91, v195
	v_dual_mov_b32 v92, v195 :: v_dual_mov_b32 v93, v195
	v_dual_mov_b32 v94, v195 :: v_dual_mov_b32 v95, v195
	v_dual_mov_b32 v96, v195 :: v_dual_mov_b32 v97, v195
	v_dual_mov_b32 v98, v195 :: v_dual_mov_b32 v99, v195
	v_dual_mov_b32 v100, v195 :: v_dual_mov_b32 v101, v195
	v_dual_mov_b32 v102, v195 :: v_dual_mov_b32 v103, v195
	v_dual_mov_b32 v104, v195 :: v_dual_mov_b32 v105, v195
	v_dual_mov_b32 v114, v195 :: v_dual_mov_b32 v115, v195
	v_dual_mov_b32 v116, v195 :: v_dual_mov_b32 v117, v195
	v_dual_mov_b32 v118, v195 :: v_dual_mov_b32 v119, v195
	v_dual_mov_b32 v120, v195 :: v_dual_mov_b32 v121, v195
	v_dual_mov_b32 v122, v195 :: v_dual_mov_b32 v123, v195
	v_dual_mov_b32 v124, v195 :: v_dual_mov_b32 v125, v195
	v_dual_mov_b32 v126, v195 :: v_dual_mov_b32 v127, v195
	v_dual_mov_b32 v128, v195 :: v_dual_mov_b32 v129, v195
	v_dual_mov_b32 v130, v195 :: v_dual_mov_b32 v131, v195
	v_dual_mov_b32 v132, v195 :: v_dual_mov_b32 v133, v195
	v_dual_mov_b32 v134, v195 :: v_dual_mov_b32 v135, v195
	v_dual_mov_b32 v136, v195 :: v_dual_mov_b32 v137, v195
	v_dual_mov_b32 v138, v195 :: v_dual_mov_b32 v139, v195
	v_dual_mov_b32 v140, v195 :: v_dual_mov_b32 v141, v195
	v_dual_mov_b32 v142, v195 :: v_dual_mov_b32 v143, v195
	v_dual_mov_b32 v144, v195 :: v_dual_mov_b32 v145, v195
	v_dual_mov_b32 v146, v195 :: v_dual_mov_b32 v147, v195
	v_dual_mov_b32 v148, v195 :: v_dual_mov_b32 v149, v195
	v_dual_mov_b32 v150, v195 :: v_dual_mov_b32 v151, v195
	v_dual_mov_b32 v152, v195 :: v_dual_mov_b32 v153, v195
	v_dual_mov_b32 v154, v195 :: v_dual_mov_b32 v155, v195
	v_dual_mov_b32 v156, v195 :: v_dual_mov_b32 v157, v195
	v_dual_mov_b32 v158, v195 :: v_dual_mov_b32 v159, v195
	v_dual_mov_b32 v160, v195 :: v_dual_mov_b32 v161, v195
	v_dual_mov_b32 v162, v195 :: v_dual_mov_b32 v163, v195
	v_dual_mov_b32 v164, v195 :: v_dual_mov_b32 v165, v195
	v_dual_mov_b32 v166, v195 :: v_dual_mov_b32 v167, v195
	v_dual_mov_b32 v168, v195 :: v_dual_mov_b32 v169, v195
	v_dual_mov_b32 v170, v195 :: v_dual_mov_b32 v171, v195
	v_dual_mov_b32 v172, v195 :: v_dual_mov_b32 v173, v195
	v_dual_mov_b32 v174, v195 :: v_dual_mov_b32 v175, v195
	v_dual_mov_b32 v176, v195 :: v_dual_mov_b32 v177, v195
	v_dual_mov_b32 v178, v195 :: v_dual_mov_b32 v179, v195
	v_dual_mov_b32 v180, v195 :: v_dual_mov_b32 v181, v195
	v_dual_mov_b32 v182, v195 :: v_dual_mov_b32 v183, v195
	v_dual_mov_b32 v184, v195 :: v_dual_mov_b32 v185, v195
	v_dual_mov_b32 v186, v195 :: v_dual_mov_b32 v187, v195
	v_dual_mov_b32 v188, v195 :: v_dual_mov_b32 v189, v195
	v_dual_mov_b32 v190, v195 :: v_dual_mov_b32 v191, v195
	v_dual_mov_b32 v192, v195 :: v_dual_mov_b32 v193, v195
	v_dual_mov_b32 v106, v195 :: v_dual_mov_b32 v107, v195
	v_dual_mov_b32 v108, v195 :: v_dual_mov_b32 v109, v195
	v_dual_mov_b32 v110, v195 :: v_dual_mov_b32 v111, v195
	v_dual_mov_b32 v112, v195 :: v_dual_mov_b32 v113, v195
	s_lshr_b32 s45, s29, 16
	s_lshr_b32 s46, s27, 16
	s_lshl_b64 s[2:3], s[2:3], 1
	s_lshl_b64 s[0:1], s[0:1], 1
	s_mov_b32 s40, s31
	s_movk_i32 s12, 0xc0
	s_bitset1_b32 s45, 23
	s_bitset1_b32 s46, 23
	s_add_nc_u64 s[16:17], s[10:11], s[2:3]
	s_add_nc_u64 s[36:37], s[14:15], s[0:1]
	s_mov_b32 s47, -1
	s_movk_i32 s4, 0x100
	s_mov_b32 s0, 0x7510000
	s_mov_b32 s48, s7
	s_branch .LBB0_17
.LBB0_16:                               ;   in Loop: Header=BB0_17 Depth=1
	s_cmp_eq_u32 s48, s43
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
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshl_b32 s1, s48, 7
	s_sub_co_i32 s2, s22, s1
	s_xor_b32 s1, s49, 1
	s_min_i32 s2, s2, 0x80
	s_cmp_lt_i32 s48, s43
	s_cselect_b32 s6, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_and_b32 s3, s6, exec_lo
	s_cselect_b32 s26, s2, 0
	s_cmp_lt_i32 s26, 0x80
	s_cselect_b32 s2, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b32 s2, s44, s2
	s_or_b32 s2, s39, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB0_29
; %bb.18:                               ;   in Loop: Header=BB0_17 Depth=1
	v_mov_b64_e32 v[204:205], v[0:1]
	v_mov_b32_e32 v208, 0x44
	s_cmp_lg_u32 s1, 0
	s_mov_b32 s8, 0
	s_cselect_b32 s3, s42, s31
	s_cselect_b32 s2, s41, 0
.LBB0_19:                               ;   Parent Loop BB0_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_dual_mov_b32 v194, v204 :: v_dual_add_nc_u32 v208, -2, v208
	v_dual_mov_b32 v210, v205 :: v_dual_mov_b32 v211, v195
	v_add_nc_u32_e32 v205, 0x200, v205
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u64 v[212:213], v[194:195], 2, s[2:3]
	v_cmp_eq_u32_e32 vcc_lo, 0, v208
	v_add_nc_u32_e32 v204, 0x200, v204
	v_lshl_add_u64 v[210:211], v[210:211], 2, s[2:3]
	s_clause 0x1
	flat_store_b32 v[212:213], v195
	flat_store_b32 v[210:211], v195
	s_or_b32 s8, vcc_lo, s8
	s_wait_xcnt 0x0
	s_and_not1_b32 exec_lo, exec_lo, s8
	s_cbranch_execnz .LBB0_19
; %bb.20:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s8
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_23
; %bb.21:                               ;   in Loop: Header=BB0_17 Depth=1
	v_add_nc_u64_e32 v[204:205], s[2:3], v[200:201]
	v_mov_b32_e32 v194, v206
	s_mov_b32 s2, 0
.LBB0_22:                               ;   Parent Loop BB0_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v194, 0x100, v194
	flat_store_b32 v[204:205], v195
	s_wait_xcnt 0x0
	v_add_nc_u64_e32 v[204:205], 0x400, v[204:205]
	v_cmp_lt_u32_e32 vcc_lo, 0x42ff, v194
	s_or_b32 s2, vcc_lo, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s2
	s_cbranch_execnz .LBB0_22
.LBB0_23:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s8
	v_mov_b64_e32 v[204:205], v[0:1]
	v_mov_b32_e32 v208, 50
	s_cmp_lg_u32 s1, 0
	s_mov_b32 s8, 0
	s_cselect_b32 s3, s35, s40
	s_cselect_b32 s2, s34, s30
.LBB0_24:                               ;   Parent Loop BB0_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_dual_mov_b32 v194, v204 :: v_dual_add_nc_u32 v208, -2, v208
	v_dual_mov_b32 v210, v205 :: v_dual_mov_b32 v211, v195
	v_add_nc_u32_e32 v205, 0x200, v205
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u64 v[212:213], v[194:195], 2, s[2:3]
	v_cmp_eq_u32_e32 vcc_lo, 0, v208
	v_add_nc_u32_e32 v204, 0x200, v204
	v_lshl_add_u64 v[210:211], v[210:211], 2, s[2:3]
	s_clause 0x1
	flat_store_b32 v[212:213], v195
	flat_store_b32 v[210:211], v195
	s_or_b32 s8, vcc_lo, s8
	s_wait_xcnt 0x0
	s_and_not1_b32 exec_lo, exec_lo, s8
	s_cbranch_execnz .LBB0_24
; %bb.25:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s8
	s_and_saveexec_b32 s8, s47
	s_cbranch_execz .LBB0_28
; %bb.26:                               ;   in Loop: Header=BB0_17 Depth=1
	v_add_nc_u64_e32 v[204:205], s[2:3], v[202:203]
	v_mov_b32_e32 v194, v207
	s_mov_b32 s2, 0
.LBB0_27:                               ;   Parent Loop BB0_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v194, 0x100, v194
	flat_store_b32 v[204:205], v195
	s_wait_xcnt 0x0
	v_add_nc_u64_e32 v[204:205], 0x400, v[204:205]
	v_cmp_lt_u32_e32 vcc_lo, 0x31ff, v194
	s_or_b32 s2, vcc_lo, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s2
	s_cbranch_execnz .LBB0_27
.LBB0_28:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s8
	s_barrier_signal -1
	s_barrier_wait -1
.LBB0_29:                               ;   in Loop: Header=BB0_17 Depth=1
	s_and_b32 s2, s6, exec_lo
	s_cselect_b32 s2, s48, 0
	s_mov_b32 s3, exec_lo
	v_cmpx_lt_i32_e32 0, v199
	s_xor_b32 s3, exec_lo, s3
	s_cbranch_execnz .LBB0_32
; %bb.30:                               ;   in Loop: Header=BB0_17 Depth=1
	s_and_not1_saveexec_b32 s8, s3
	s_cbranch_execnz .LBB0_35
.LBB0_31:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s8
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s21
	s_cbranch_vccnz .LBB0_16
	s_branch .LBB0_36
.LBB0_32:                               ;   in Loop: Header=BB0_17 Depth=1
	s_mov_b32 s50, exec_lo
	v_cmpx_eq_u32_e32 1, v199
	s_cbranch_execz .LBB0_34
; %bb.33:                               ;   in Loop: Header=BB0_17 Depth=1
	s_cmp_lg_u32 s1, 0
	s_mov_b32 s28, s26
	s_cselect_b32 s10, s34, s30
	s_cmp_gt_i32 s26, 0
	s_mov_b32 s13, s5
	s_cselect_b32 s11, -1, 0
	s_lshl_b32 s6, s2, 7
	s_mov_b32 s14, s7
	s_lshl_b64 s[8:9], s[6:7], 1
	s_mov_b32 s15, s7
	s_add_nc_u64 s[8:9], s[16:17], s[8:9]
	s_delay_alu instid0(SALU_CYCLE_1)
	v_dual_mov_b32 v205, s10 :: v_dual_mov_b32 v204, s8
	s_and_b32 s6, s9, 0x1ffffff
	s_and_b32 s9, s38, s11
	s_bitset1_b32 s6, 31
	v_cndmask_b32_e64 v194, 0, 1, s9
	v_mov_b32_e32 v209, s6
	v_readfirstlane_b32 s53, v205
	v_readfirstlane_b32 s54, v204
	s_lshr_b64 s[10:11], s[28:29], 16
	v_readfirstlane_b32 s52, v194
	v_readfirstlane_b32 s55, v209
	s_lshl_b32 s9, s26, 16
	s_mov_b32 s8, s0
	s_mov_b32 s11, s45
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[52:55], s[8:15]
.LBB0_34:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s50
	s_and_not1_saveexec_b32 s8, s3
	s_cbranch_execz .LBB0_31
.LBB0_35:                               ;   in Loop: Header=BB0_17 Depth=1
	s_cmp_lg_u32 s1, 0
	s_cselect_b32 s1, s41, 0
	s_cmp_gt_i32 s26, 0
	s_cselect_b32 s9, -1, 0
	s_lshl_b32 s6, s2, 7
	s_delay_alu instid0(SALU_CYCLE_1)
	s_lshl_b64 s[2:3], s[6:7], 1
	s_and_b32 s6, s25, s9
	s_add_nc_u64 s[2:3], s[36:37], s[2:3]
	v_cndmask_b32_e64 v194, 0, 1, s6
	s_and_b32 s3, s3, 0x1ffffff
	v_dual_mov_b32 v205, s1 :: v_dual_mov_b32 v204, s2
	s_bitset1_b32 s3, 31
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_readfirstlane_b32 s52, v194
	v_mov_b32_e32 v209, s3
	v_readfirstlane_b32 s53, v205
	v_readfirstlane_b32 s54, v204
	s_lshr_b64 s[2:3], s[26:27], 16
	s_lshl_b32 s1, s26, 16
	v_readfirstlane_b32 s55, v209
	s_mov_b32 s3, s46
	s_mov_b32 s6, s7
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	tensor_load_to_lds s[52:55], s[0:7]
	s_or_b32 exec_lo, exec_lo, s8
	s_and_not1_b32 vcc_lo, exec_lo, s21
	s_cbranch_vccnz .LBB0_16
.LBB0_36:                               ;   in Loop: Header=BB0_17 Depth=1
	s_cmp_lg_u32 s49, 0
	s_cselect_b32 s1, s41, 0
	s_cselect_b32 s2, s34, s30
	v_lshl_add_u32 v194, v196, 1, s1
	v_lshl_add_u32 v204, v198, 1, s2
	ds_load_b128 v[208:211], v194
	ds_load_b128 v[212:215], v194 offset:16
	ds_load_b128 v[216:219], v204
	ds_load_b128 v[220:223], v204 offset:16
	ds_load_b128 v[224:227], v194 offset:4352
	ds_load_b128 v[228:231], v194 offset:4368
	ds_load_b128 v[232:235], v204 offset:4352
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[0:3] /*v[256:259]*/, v194 offset:4416
	ds_load_b128 v[4:7] /*v[260:263]*/, v194 offset:4432
	ds_load_b128 v[8:11] /*v[264:267]*/, v204 offset:64
	ds_load_b128 v[12:15] /*v[268:271]*/, v204 offset:80
	ds_load_b128 v[16:19] /*v[272:275]*/, v204 offset:8768
	ds_load_b128 v[20:23] /*v[276:279]*/, v204 offset:8784
	ds_load_b128 v[24:27] /*v[280:283]*/, v204 offset:13120
	ds_load_b128 v[28:31] /*v[284:287]*/, v204 offset:13136
	ds_load_b128 v[32:35] /*v[288:291]*/, v204 offset:17472
	ds_load_b128 v[36:39] /*v[292:295]*/, v204 offset:17488
	ds_load_b128 v[40:43] /*v[296:299]*/, v204 offset:21824
	ds_load_b128 v[44:47] /*v[300:303]*/, v204 offset:21840
	ds_load_b128 v[48:51] /*v[304:307]*/, v204 offset:26176
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_wait_dscnt 0x10
	v_wmma_f32_16x16x32_bf16 v[186:193], v[208:215], v[216:223], v[186:193]
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[52:55] /*v[308:311]*/, v204 offset:26192
	ds_load_b128 v[56:59] /*v[312:315]*/, v204 offset:30528
	ds_load_b128 v[60:63] /*v[316:319]*/, v204 offset:30544
	ds_load_b128 v[64:67] /*v[320:323]*/, v204 offset:39232
	ds_load_b128 v[68:71] /*v[324:327]*/, v204 offset:39248
	ds_load_b128 v[72:75] /*v[328:331]*/, v204 offset:43584
	ds_load_b128 v[76:79] /*v[332:335]*/, v204 offset:43600
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_wait_dscnt 0x15
	v_wmma_f32_16x16x32_bf16 v[82:89], v[224:231], v[216:223], v[82:89] matrix_a_reuse
	ds_load_b128 v[236:239], v204 offset:4368
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[80:83] /*v[336:339]*/, v204 offset:47936
	ds_load_b128 v[84:87] /*v[340:343]*/, v204 offset:47952
	; sched_group_barrier mask(0x00000100) size(7) SyncID(0)
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[74:81], v[224:231], v[232:239], v[74:81] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[178:185], v[208:215], v[232:239], v[178:185] matrix_a_reuse
	ds_load_b128 v[232:235], v204 offset:21760
	ds_load_b128 v[236:239], v204 offset:21776
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[42:49], v[224:231], v[232:239], v[42:49] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[146:153], v[208:215], v[232:239], v[146:153] matrix_a_reuse
	ds_load_b128 v[236:239], v204 offset:43536
	ds_load_b128 v[232:235], v204 offset:43520
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[98:105], v[208:215], v[232:239], v[98:105] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[2:9], v[224:231], v[232:239], v[2:9] matrix_a_reuse
	ds_load_b128 v[240:243], v204 offset:13056
	ds_load_b128 v[244:247], v204 offset:13072
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[58:65], v[224:231], v[240:247], v[58:65] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[162:169], v[208:215], v[240:247], v[162:169] matrix_a_reuse
	ds_load_b128 v[240:243], v204 offset:30464
	ds_load_b128 v[244:247], v204 offset:30480
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[130:137], v[208:215], v[240:247], v[130:137] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[26:33], v[224:231], v[240:247], v[26:33] matrix_a_reuse
	ds_load_b128 v[240:243], v204 offset:47872
	ds_load_b128 v[244:247], v204 offset:47888
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[90:97], v[208:215], v[240:247], v[90:97] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[106:113], v[224:231], v[240:247], v[106:113]
	ds_load_b128 v[216:219], v204 offset:17408
	ds_load_b128 v[220:223], v204 offset:17424
	ds_load_b128 v[248:251], v204 offset:8704
	ds_load_b128 v[252:255], v204 offset:8720
	; sched_group_barrier mask(0x00000008) size(6) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(7) SyncID(0)
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[50:57], v[224:231], v[216:223], v[50:57] matrix_a_reuse
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[66:73], v[224:231], v[248:255], v[66:73] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[170:177], v[208:215], v[248:255], v[170:177] matrix_a_reuse
	ds_load_b128 v[248:251], v204 offset:34816
	ds_load_b128 v[252:255], v204 offset:34832
	; sched_group_barrier mask(0x00000008) size(6) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(7) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[154:161], v[208:215], v[216:223], v[154:161] matrix_a_reuse
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[18:25], v[224:231], v[248:255], v[18:25] matrix_a_reuse
	ds_load_b128 v[216:219], v204 offset:26112
	ds_load_b128 v[220:223], v204 offset:26128
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[34:41], v[224:231], v[216:223], v[34:41] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[138:145], v[208:215], v[216:223], v[138:145] matrix_a_reuse
	ds_load_b128 v[216:219], v204 offset:39168
	ds_load_b128 v[220:223], v204 offset:39184
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[114:121], v[208:215], v[216:223], v[114:121] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(6) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(7) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[10:17], v[224:231], v[216:223], v[10:17] matrix_a_reuse
	ds_load_b128 v[216:219], v204 offset:4416
	ds_load_b128 v[220:223], v204 offset:4432
	v_wmma_f32_16x16x32_bf16 v[122:129], v[208:215], v[248:255], v[122:129] matrix_a_reuse
	ds_load_b128 v[248:251], v194 offset:64
	ds_load_b128 v[252:255], v194 offset:80
	ds_load_b128 v[208:211], v204 offset:34880
	ds_load_b128 v[212:215], v204 offset:34896
	; sched_group_barrier mask(0x00000008) size(6) SyncID(0)
	; sched_barrier mask(0x00000000)
	ds_load_b128 v[224:227], v194 offset:128
	ds_load_b128 v[228:231], v194 offset:144
	ds_load_b128 v[232:235], v194 offset:4480
	ds_load_b128 v[236:239], v194 offset:4496
	ds_load_b128 v[240:243], v204 offset:128
	ds_load_b128 v[244:247], v204 offset:144
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[88:91] /*v[344:347]*/, v204 offset:4480
	s_set_vgpr_msb 0x4004                   ;  msbs: dst=0 src0=0 src1=1 src2=0
	s_wait_dscnt 0x9
	v_wmma_f32_16x16x32_bf16 v[186:193], v[248:255], v[8:15] /*v[264:271]*/, v[186:193]
	; sched_group_barrier mask(0x00000100) size(7) SyncID(0)
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_wmma_f32_16x16x32_bf16 v[178:185], v[248:255], v[216:223], v[178:185] matrix_a_reuse
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[170:177], v[248:255], v[16:23] /*v[272:279]*/, v[170:177] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[162:169], v[248:255], v[24:31] /*v[280:287]*/, v[162:169] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[154:161], v[248:255], v[32:39] /*v[288:295]*/, v[154:161] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[146:153], v[248:255], v[40:47] /*v[296:303]*/, v[146:153] matrix_a_reuse
	s_set_vgpr_msb 0x440                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[92:95] /*v[348:351]*/, v204 offset:4496
	ds_load_b128 v[96:99] /*v[352:355]*/, v204 offset:8832
	ds_load_b128 v[100:103] /*v[356:359]*/, v204 offset:8848
	ds_load_b128 v[104:107] /*v[360:363]*/, v204 offset:13184
	ds_load_b128 v[108:111] /*v[364:367]*/, v204 offset:13200
	ds_load_b128 v[112:115] /*v[368:371]*/, v204 offset:17536
	ds_load_b128 v[116:119] /*v[372:375]*/, v204 offset:17552
	s_set_vgpr_msb 0x4004                   ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[138:145], v[248:255], v[48:55] /*v[304:311]*/, v[138:145] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(6) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(7) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[130:137], v[248:255], v[56:63] /*v[312:319]*/, v[130:137] matrix_a_reuse
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[122:129], v[248:255], v[208:215], v[122:129] matrix_a_reuse
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[114:121], v[248:255], v[64:71] /*v[320:327]*/, v[114:121] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[98:105], v[248:255], v[72:79] /*v[328:335]*/, v[98:105] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[90:97], v[248:255], v[80:87] /*v[336:343]*/, v[90:97] matrix_a_reuse
	ds_load_b128 v[248:251], v204 offset:21888
	ds_load_b128 v[252:255], v204 offset:21904
	s_set_vgpr_msb 0x440                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[120:123] /*v[376:379]*/, v204 offset:26240
	ds_load_b128 v[124:127] /*v[380:383]*/, v204 offset:26256
	ds_load_b128 v[128:131] /*v[384:387]*/, v204 offset:30592
	ds_load_b128 v[132:135] /*v[388:391]*/, v204 offset:30608
	ds_load_b128 v[136:139] /*v[392:395]*/, v204 offset:34944
	s_set_vgpr_msb 0x4005                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[106:113], v[0:7] /*v[256:263]*/, v[80:87] /*v[336:343]*/, v[106:113]
	; sched_group_barrier mask(0x00000008) size(6) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(7) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[2:9], v[0:7] /*v[256:263]*/, v[72:79] /*v[328:335]*/, v[2:9] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[10:17], v[0:7] /*v[256:263]*/, v[64:71] /*v[320:327]*/, v[10:17] matrix_a_reuse
	s_set_vgpr_msb 0x501                    ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_wmma_f32_16x16x32_bf16 v[18:25], v[0:7] /*v[256:263]*/, v[208:215], v[18:25] matrix_a_reuse
	s_set_vgpr_msb 0x105                    ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[26:33], v[0:7] /*v[256:263]*/, v[56:63] /*v[312:319]*/, v[26:33] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[34:41], v[0:7] /*v[256:263]*/, v[48:55] /*v[304:311]*/, v[34:41] matrix_a_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[140:143] /*v[396:399]*/, v204 offset:34960
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	ds_load_b128 v[208:211], v204 offset:39296
	ds_load_b128 v[212:215], v204 offset:39312
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[48:51] /*v[304:307]*/, v204 offset:43648
	ds_load_b128 v[52:55] /*v[308:311]*/, v204 offset:43664
	ds_load_b128 v[56:59] /*v[312:315]*/, v204 offset:48000
	ds_load_b128 v[60:63] /*v[316:319]*/, v204 offset:48016
	s_set_vgpr_msb 0x4005                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[42:49], v[0:7] /*v[256:263]*/, v[40:47] /*v[296:303]*/, v[42:49] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(6) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(7) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[50:57], v[0:7] /*v[256:263]*/, v[32:39] /*v[288:295]*/, v[50:57] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[58:65], v[0:7] /*v[256:263]*/, v[24:31] /*v[280:287]*/, v[58:65] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[66:73], v[0:7] /*v[256:263]*/, v[16:23] /*v[272:279]*/, v[66:73] matrix_a_reuse
	s_set_vgpr_msb 0x501                    ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_wmma_f32_16x16x32_bf16 v[74:81], v[0:7] /*v[256:263]*/, v[216:223], v[74:81] matrix_a_reuse
	s_set_vgpr_msb 0x105                    ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[82:89], v[0:7] /*v[256:263]*/, v[8:15] /*v[264:271]*/, v[82:89] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(6) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x500                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	ds_load_b128 v[216:219], v194 offset:192
	ds_load_b128 v[220:223], v194 offset:208
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[0:3] /*v[256:259]*/, v194 offset:4544
	ds_load_b128 v[4:7] /*v[260:263]*/, v194 offset:4560
	ds_load_b128 v[8:11] /*v[264:267]*/, v204 offset:192
	ds_load_b128 v[12:15] /*v[268:271]*/, v204 offset:208
	ds_load_b128 v[16:19] /*v[272:275]*/, v204 offset:4544
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_wait_dscnt 0x1d
	v_wmma_f32_16x16x32_bf16 v[186:193], v[224:231], v[240:247], v[186:193]
	; sched_group_barrier mask(0x00000100) size(7) SyncID(0)
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	s_wait_dscnt 0x1b
	v_wmma_f32_16x16x32_bf16 v[178:185], v[224:231], v[88:95] /*v[344:351]*/, v[178:185] matrix_a_reuse
	s_wait_dscnt 0x19
	v_wmma_f32_16x16x32_bf16 v[170:177], v[224:231], v[96:103] /*v[352:359]*/, v[170:177] matrix_a_reuse
	s_wait_dscnt 0x17
	v_wmma_f32_16x16x32_bf16 v[162:169], v[224:231], v[104:111] /*v[360:367]*/, v[162:169] matrix_a_reuse
	s_wait_dscnt 0x15
	v_wmma_f32_16x16x32_bf16 v[154:161], v[224:231], v[112:119] /*v[368:375]*/, v[154:161] matrix_a_reuse
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_wait_dscnt 0x13
	v_wmma_f32_16x16x32_bf16 v[146:153], v[224:231], v[248:255], v[146:153] matrix_a_reuse
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[20:23] /*v[276:279]*/, v204 offset:4560
	ds_load_b128 v[24:27] /*v[280:283]*/, v204 offset:8896
	ds_load_b128 v[28:31] /*v[284:287]*/, v204 offset:8912
	ds_load_b128 v[32:35] /*v[288:291]*/, v204 offset:13248
	ds_load_b128 v[36:39] /*v[292:295]*/, v204 offset:13264
	ds_load_b128 v[40:43] /*v[296:299]*/, v204 offset:17600
	ds_load_b128 v[44:47] /*v[300:303]*/, v204 offset:17616
	s_set_vgpr_msb 0x4004                   ;  msbs: dst=0 src0=0 src1=1 src2=0
	s_wait_dscnt 0x18
	v_wmma_f32_16x16x32_bf16 v[138:145], v[224:231], v[120:127] /*v[376:383]*/, v[138:145] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(6) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(7) SyncID(0)
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[130:137], v[224:231], v[128:135] /*v[384:391]*/, v[130:137] matrix_a_reuse
	s_wait_dscnt 0x14
	v_wmma_f32_16x16x32_bf16 v[122:129], v[224:231], v[136:143] /*v[392:399]*/, v[122:129] matrix_a_reuse
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[114:121], v[224:231], v[208:215], v[114:121] matrix_a_reuse
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	s_wait_dscnt 0x10
	v_wmma_f32_16x16x32_bf16 v[98:105], v[224:231], v[48:55] /*v[304:311]*/, v[98:105] matrix_a_reuse
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[90:97], v[224:231], v[56:63] /*v[312:319]*/, v[90:97] matrix_a_reuse
	ds_load_b128 v[224:227], v204 offset:21952
	ds_load_b128 v[228:231], v204 offset:21968
	s_set_vgpr_msb 0x440                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[64:67] /*v[320:323]*/, v204 offset:26304
	ds_load_b128 v[68:71] /*v[324:327]*/, v204 offset:26320
	ds_load_b128 v[72:75] /*v[328:331]*/, v204 offset:30656
	ds_load_b128 v[76:79] /*v[332:335]*/, v204 offset:30672
	ds_load_b128 v[80:83] /*v[336:339]*/, v204 offset:35008
	s_set_vgpr_msb 0x4004                   ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[106:113], v[232:239], v[56:63] /*v[312:319]*/, v[106:113]
	; sched_group_barrier mask(0x00000008) size(6) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(7) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[2:9], v[232:239], v[48:55] /*v[304:311]*/, v[2:9] matrix_a_reuse
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_wmma_f32_16x16x32_bf16 v[10:17], v[232:239], v[208:215], v[10:17] matrix_a_reuse
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[18:25], v[232:239], v[136:143] /*v[392:399]*/, v[18:25] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[26:33], v[232:239], v[128:135] /*v[384:391]*/, v[26:33] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[34:41], v[232:239], v[120:127] /*v[376:383]*/, v[34:41] matrix_a_reuse
	s_set_vgpr_msb 0x440                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[84:87] /*v[340:343]*/, v204 offset:35024
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	ds_load_b128 v[208:211], v204 offset:39360
	ds_load_b128 v[212:215], v204 offset:39376
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[48:51] /*v[304:307]*/, v204 offset:43712
	ds_load_b128 v[52:55] /*v[308:311]*/, v204 offset:43728
	ds_load_b128 v[56:59] /*v[312:315]*/, v204 offset:48064
	ds_load_b128 v[60:63] /*v[316:319]*/, v204 offset:48080
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_wmma_f32_16x16x32_bf16 v[42:49], v[232:239], v[248:255], v[42:49] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(6) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(7) SyncID(0)
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[50:57], v[232:239], v[112:119] /*v[368:375]*/, v[50:57] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[58:65], v[232:239], v[104:111] /*v[360:367]*/, v[58:65] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[66:73], v[232:239], v[96:103] /*v[352:359]*/, v[66:73] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[74:81], v[232:239], v[88:95] /*v[344:351]*/, v[74:81] matrix_a_reuse
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_wmma_f32_16x16x32_bf16 v[82:89], v[232:239], v[240:247], v[82:89] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(6) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[186:193], v[216:223], v[8:15] /*v[264:271]*/, v[186:193]
	; sched_group_barrier mask(0x00000100) size(7) SyncID(0)
	s_wait_dscnt 0x14
	v_wmma_f32_16x16x32_bf16 v[178:185], v[216:223], v[16:23] /*v[272:279]*/, v[178:185] matrix_a_reuse
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[170:177], v[216:223], v[24:31] /*v[280:287]*/, v[170:177] matrix_a_reuse
	s_wait_dscnt 0x10
	v_wmma_f32_16x16x32_bf16 v[162:169], v[216:223], v[32:39] /*v[288:295]*/, v[162:169] matrix_a_reuse
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[154:161], v[216:223], v[40:47] /*v[296:303]*/, v[154:161] matrix_a_reuse
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[146:153], v[216:223], v[224:231], v[146:153] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(6) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(7) SyncID(0)
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	s_wait_dscnt 0xa
	v_wmma_f32_16x16x32_bf16 v[138:145], v[216:223], v[64:71] /*v[320:327]*/, v[138:145] matrix_a_reuse
	s_wait_dscnt 0x8
	v_wmma_f32_16x16x32_bf16 v[130:137], v[216:223], v[72:79] /*v[328:335]*/, v[130:137] matrix_a_reuse
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[122:129], v[216:223], v[80:87] /*v[336:343]*/, v[122:129] matrix_a_reuse
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[114:121], v[216:223], v[208:215], v[114:121] matrix_a_reuse
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[98:105], v[216:223], v[48:55] /*v[304:311]*/, v[98:105] matrix_a_reuse
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[90:97], v[216:223], v[56:63] /*v[312:319]*/, v[90:97] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(6) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(7) SyncID(0)
	s_set_vgpr_msb 0x405                    ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[106:113], v[0:7] /*v[256:263]*/, v[56:63] /*v[312:319]*/, v[106:113]
	v_wmma_f32_16x16x32_bf16 v[2:9], v[0:7] /*v[256:263]*/, v[48:55] /*v[304:311]*/, v[2:9] matrix_a_reuse
	s_set_vgpr_msb 0x501                    ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_wmma_f32_16x16x32_bf16 v[10:17], v[0:7] /*v[256:263]*/, v[208:215], v[10:17] matrix_a_reuse
	s_set_vgpr_msb 0x105                    ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[18:25], v[0:7] /*v[256:263]*/, v[80:87] /*v[336:343]*/, v[18:25] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[26:33], v[0:7] /*v[256:263]*/, v[72:79] /*v[328:335]*/, v[26:33] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[34:41], v[0:7] /*v[256:263]*/, v[64:71] /*v[320:327]*/, v[34:41] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(6) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(7) SyncID(0)
	s_set_vgpr_msb 0x501                    ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_wmma_f32_16x16x32_bf16 v[42:49], v[0:7] /*v[256:263]*/, v[224:231], v[42:49] matrix_a_reuse
	s_set_vgpr_msb 0x105                    ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[50:57], v[0:7] /*v[256:263]*/, v[40:47] /*v[296:303]*/, v[50:57] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[58:65], v[0:7] /*v[256:263]*/, v[32:39] /*v[288:295]*/, v[58:65] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[66:73], v[0:7] /*v[256:263]*/, v[24:31] /*v[280:287]*/, v[66:73] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[74:81], v[0:7] /*v[256:263]*/, v[16:23] /*v[272:279]*/, v[74:81] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[82:89], v[0:7] /*v[256:263]*/, v[8:15] /*v[264:271]*/, v[82:89] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(6) SyncID(0)
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
	s_and_b32 vcc_lo, exec_lo, s21
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_vccz .LBB0_39
; %bb.38:
	v_lshrrev_b32_e32 v1, 1, v0
	v_and_b32_e32 v194, 0xe0, v0
	v_cvt_pk_bf16_f32 v185, v184, v185
	v_cvt_pk_bf16_f32 v184, v182, v183
	v_cvt_pk_bf16_f32 v182, v178, v179
	v_lshl_or_b32 v1, v0, 8, v1
	v_cvt_pk_bf16_f32 v169, v168, v169
	v_cvt_pk_bf16_f32 v168, v166, v167
	v_cvt_pk_bf16_f32 v167, v164, v165
	v_cvt_pk_bf16_f32 v166, v162, v163
	v_and_or_b32 v1, 0xf08, v1, v194
	v_cvt_pk_bf16_f32 v153, v152, v153
	v_cvt_pk_bf16_f32 v152, v150, v151
	v_cvt_pk_bf16_f32 v151, v148, v149
	v_cvt_pk_bf16_f32 v145, v144, v145
	v_or_b32_e32 v178, 0x1000, v1
	v_or_b32_e32 v179, 0x2000, v1
	v_or_b32_e32 v163, 0x5000, v1
	v_or_b32_e32 v165, 0x6000, v1
	v_or_b32_e32 v148, 0x8000, v1
	v_cvt_pk_bf16_f32 v144, v142, v143
	v_cvt_pk_bf16_f32 v142, v138, v139
	v_or_b32_e32 v138, 0x7000, v1
	v_cvt_pk_bf16_f32 v137, v136, v137
	v_cvt_pk_bf16_f32 v136, v134, v135
	v_cvt_pk_bf16_f32 v135, v132, v133
	v_or_b32_e32 v133, 0xa000, v1
	v_cvt_pk_bf16_f32 v33, v32, v33
	v_cvt_pk_bf16_f32 v32, v30, v31
	v_cvt_pk_bf16_f32 v31, v28, v29
	v_or_b32_e32 v28, 0xa010, v1
	v_cvt_pk_bf16_f32 v143, v140, v141
	v_or_b32_e32 v140, 0x9000, v1
	v_cvt_pk_bf16_f32 v129, v128, v129
	v_cvt_pk_bf16_f32 v128, v126, v127
	v_cvt_pk_bf16_f32 v127, v124, v125
	v_or_b32_e32 v124, 0xb000, v1
	v_cvt_pk_bf16_f32 v177, v176, v177
	v_cvt_pk_bf16_f32 v176, v174, v175
	v_cvt_pk_bf16_f32 v174, v170, v171
	v_lshrrev_b32_e32 v171, 3, v178
	v_or_b32_e32 v178, 0x3000, v1
	v_cvt_pk_bf16_f32 v49, v48, v49
	v_cvt_pk_bf16_f32 v48, v46, v47
	v_cvt_pk_bf16_f32 v46, v42, v43
	v_or_b32_e32 v42, 0x8010, v1
	v_cvt_pk_bf16_f32 v41, v40, v41
	v_cvt_pk_bf16_f32 v40, v38, v39
	v_cvt_pk_bf16_f32 v38, v34, v35
	v_or_b32_e32 v35, 0x9010, v1
	v_cvt_pk_bf16_f32 v175, v172, v173
	v_lshrrev_b32_e32 v172, 3, v179
	v_or_b32_e32 v179, 0x4000, v1
	v_dual_lshrrev_b32 v170, 3, v1 :: v_dual_lshlrev_b32 v173, 1, v1
	v_dual_lshrrev_b32 v163, 3, v163 :: v_dual_lshrrev_b32 v165, 3, v165
	v_dual_lshrrev_b32 v139, 3, v148 :: v_dual_lshrrev_b32 v138, 3, v138
	v_cvt_pk_bf16_f32 v126, v122, v123
	v_lshrrev_b32_e32 v123, 3, v133
	v_cvt_pk_bf16_f32 v25, v24, v25
	v_cvt_pk_bf16_f32 v24, v22, v23
	v_cvt_pk_bf16_f32 v23, v20, v21
	v_lshrrev_b32_e32 v20, 3, v28
	v_or_b32_e32 v1, 0xb010, v1
	v_lshrrev_b32_e32 v132, 3, v140
	v_cvt_pk_bf16_f32 v121, v120, v121
	v_cvt_pk_bf16_f32 v120, v118, v119
	v_cvt_pk_bf16_f32 v118, v114, v115
	v_dual_lshrrev_b32 v115, 3, v124 :: v_dual_lshrrev_b32 v178, 3, v178
	v_lshrrev_b32_e32 v34, 3, v42
	v_cvt_pk_bf16_f32 v30, v26, v27
	v_lshrrev_b32_e32 v26, 3, v35
	v_lshrrev_b32_e32 v179, 3, v179
	v_and_b32_e32 v172, 0x5f0, v172
	v_cvt_pk_bf16_f32 v161, v160, v161
	v_cvt_pk_bf16_f32 v160, v158, v159
	v_cvt_pk_bf16_f32 v158, v154, v155
	v_and_b32_e32 v154, 0xbf0, v163
	v_and_b32_e32 v155, 0xdf0, v165
	v_and_b32_e32 v139, 0x11f0, v139
	v_cvt_pk_bf16_f32 v134, v130, v131
	v_and_b32_e32 v130, 0xff0, v138
	v_and_b32_e32 v114, 0x15f0, v123
	v_cvt_pk_bf16_f32 v22, v18, v19
	v_and_b32_e32 v19, 0x15f0, v20
	v_lshrrev_b32_e32 v20, 3, v1
	v_and_b32_e32 v132, 0x13f0, v132
	v_and_b32_e32 v115, 0x17f0, v115
	v_and_b32_e32 v170, 0x1f0, v170
	v_and_b32_e32 v178, 0x7f0, v178
	v_and_b32_e32 v34, 0x11f0, v34
	v_and_b32_e32 v26, 0x13f0, v26
	v_and_b32_e32 v171, 0x3f0, v171
	v_and_b32_e32 v164, 0x9f0, v179
	v_add_nc_u32_e32 v172, v172, v173
	v_cvt_pk_bf16_f32 v150, v146, v147
	v_dual_add_nc_u32 v146, v154, v173 :: v_dual_add_nc_u32 v147, v155, v173
	v_lshl_add_u32 v131, v148, 1, v139
	v_cvt_pk_bf16_f32 v119, v116, v117
	v_add_nc_u32_e32 v116, v130, v173
	v_lshl_add_u32 v114, v133, 1, v114
	v_cvt_pk_bf16_f32 v105, v104, v105
	v_cvt_pk_bf16_f32 v104, v102, v103
	v_cvt_pk_bf16_f32 v103, v100, v101
	v_cvt_pk_bf16_f32 v102, v98, v99
	v_cvt_pk_bf16_f32 v17, v16, v17
	v_cvt_pk_bf16_f32 v16, v14, v15
	v_cvt_pk_bf16_f32 v14, v10, v11
	v_and_b32_e32 v11, 0x17f0, v20
	v_lshl_add_u32 v122, v140, 1, v132
	v_lshl_add_u32 v98, v124, 1, v115
	v_cvt_pk_bf16_f32 v97, v96, v97
	v_cvt_pk_bf16_f32 v96, v94, v95
	v_cvt_pk_bf16_f32 v95, v92, v93
	v_cvt_pk_bf16_f32 v94, v90, v91
	v_cvt_pk_bf16_f32 v73, v72, v73
	v_cvt_pk_bf16_f32 v72, v70, v71
	v_cvt_pk_bf16_f32 v71, v68, v69
	v_cvt_pk_bf16_f32 v70, v66, v67
	v_cvt_pk_bf16_f32 v47, v44, v45
	v_cvt_pk_bf16_f32 v39, v36, v37
	v_cvt_pk_bf16_f32 v193, v192, v193
	v_cvt_pk_bf16_f32 v192, v190, v191
	v_cvt_pk_bf16_f32 v191, v188, v189
	v_cvt_pk_bf16_f32 v190, v186, v187
	v_add_nc_u32_e32 v170, v170, v173
	v_add_nc_u32_e32 v162, v178, v173
	v_lshl_add_u32 v27, v42, 1, v34
	v_cvt_pk_bf16_f32 v89, v88, v89
	v_cvt_pk_bf16_f32 v88, v86, v87
	v_cvt_pk_bf16_f32 v87, v84, v85
	v_cvt_pk_bf16_f32 v86, v82, v83
	v_cvt_pk_bf16_f32 v65, v64, v65
	v_cvt_pk_bf16_f32 v64, v62, v63
	v_cvt_pk_bf16_f32 v63, v60, v61
	v_cvt_pk_bf16_f32 v62, v58, v59
	v_lshl_add_u32 v18, v35, 1, v26
	v_cvt_pk_bf16_f32 v15, v12, v13
	v_cvt_pk_bf16_f32 v183, v180, v181
	v_dual_add_nc_u32 v171, v171, v173 :: v_dual_add_nc_u32 v164, v164, v173
	v_cvt_pk_bf16_f32 v159, v156, v157
	v_lshl_add_u32 v10, v28, 1, v19
	v_cvt_pk_bf16_f32 v9, v8, v9
	v_cvt_pk_bf16_f32 v8, v6, v7
	v_cvt_pk_bf16_f32 v7, v4, v5
	v_cvt_pk_bf16_f32 v6, v2, v3
	ds_store_b128 v131, v[126:129]
	ds_store_b128 v122, v[118:121]
	v_cvt_pk_bf16_f32 v81, v80, v81
	v_cvt_pk_bf16_f32 v80, v78, v79
	v_cvt_pk_bf16_f32 v79, v76, v77
	v_cvt_pk_bf16_f32 v78, v74, v75
	ds_store_b128 v114, v[102:105]
	ds_store_b128 v98, v[94:97]
	ds_store_b128 v170, v[190:193]
	ds_store_b128 v170, v[86:89] offset:32
	ds_store_b128 v171, v[182:185] offset:8192
	ds_store_b128 v171, v[78:81] offset:8224
	v_cvt_pk_bf16_f32 v57, v56, v57
	v_cvt_pk_bf16_f32 v56, v54, v55
	v_cvt_pk_bf16_f32 v55, v52, v53
	v_cvt_pk_bf16_f32 v54, v50, v51
	ds_store_b128 v172, v[174:177] offset:16384
	ds_store_b128 v172, v[70:73] offset:16416
	ds_store_b128 v162, v[166:169] offset:24576
	ds_store_b128 v162, v[62:65] offset:24608
	ds_store_b128 v164, v[158:161] offset:32768
	ds_store_b128 v164, v[54:57] offset:32800
	ds_store_b128 v146, v[150:153] offset:40960
	ds_store_b128 v146, v[46:49] offset:40992
	ds_store_b128 v147, v[142:145] offset:49152
	ds_store_b128 v147, v[38:41] offset:49184
	v_lshl_add_u32 v1, v1, 1, v11
	v_cvt_pk_bf16_f32 v5, v112, v113
	v_cvt_pk_bf16_f32 v4, v110, v111
	v_cvt_pk_bf16_f32 v3, v108, v109
	v_cvt_pk_bf16_f32 v2, v106, v107
	ds_store_b128 v116, v[134:137] offset:57344
	ds_store_b128 v116, v[30:33] offset:57376
	ds_store_b128 v27, v[22:25]
	ds_store_b128 v18, v[14:17]
	ds_store_b128 v10, v[6:9]
	ds_store_b128 v1, v[2:5]
.LBB0_39:
	v_cmp_ne_u32_e32 vcc_lo, 1, v197
	;;#ASMSTART
	s_wait_dscnt 0
	;;#ASMEND
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_vccnz .LBB0_50
; %bb.40:
	s_mul_i32 s3, s29, s27
	s_mov_b32 s0, exec_lo
	v_cmpx_gt_i32_e64 s3, v0
	s_cbranch_execz .LBB0_50
; %bb.41:
	s_ashr_i32 s25, s24, 31
	v_xad_u32 v2, v0, -1, s3
	s_lshl_b64 s[0:1], s[24:25], 1
	s_wait_kmcnt 0x0
	s_mul_i32 s14, s33, 0xc0
	s_ashr_i32 s21, s20, 31
	s_add_nc_u64 s[4:5], s[18:19], s[0:1]
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
	s_abs_i32 s16, s23
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
	s_mov_b32 s6, s20
	v_mov_b32_e32 v10, v9
	s_mov_b32 s7, s21
	s_mul_f32 s0, s0, 0x4f7ffffe
	v_mov_b64_e32 v[4:5], v[2:3]
	v_mov_b64_e32 v[2:3], v[0:1]
	s_mov_b32 s8, s20
	s_cvt_u32_f32 s0, s0
	s_mov_b32 s9, s21
	s_mov_b32 s10, s20
	s_mov_b32 s11, s21
	s_mul_i32 s1, s1, s0
	s_mov_b32 s22, s23
	s_mul_hi_u32 s1, s0, s1
	s_mov_b32 s24, s23
	s_mov_b32 s25, s23
	s_ashr_i32 s26, s23, 31
	s_add_co_i32 s12, s0, s1
	s_mov_b32 s27, s13
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
	v_xor_b32_e32 v1, s26, v1
	v_xor_b32_e32 v11, s26, v11
	v_mul_lo_u32 v24, v20, s16
	v_mul_lo_u32 v26, v21, s16
	v_mul_lo_u32 v27, v22, s16
	v_dual_add_nc_u32 v25, 1, v20 :: v_dual_add_nc_u32 v29, 1, v21
	v_mul_lo_u32 v28, v23, s16
	v_dual_add_nc_u32 v30, 1, v22 :: v_dual_add_nc_u32 v31, 1, v23
	v_dual_sub_nc_u32 v6, v6, v24 :: v_dual_sub_nc_u32 v12, v12, v26
	v_dual_sub_nc_u32 v16, v16, v27 :: v_dual_bitop2_b32 v14, s26, v14 bitop3:0x14
	v_xor_b32_e32 v18, s26, v18
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
	v_mul_lo_u32 v12, v11, s22
	v_dual_sub_nc_u32 v26, v16, v14 :: v_dual_bitop2_b32 v6, v6, v1 bitop3:0x14
	v_cndmask_b32_e32 v19, v23, v27, vcc_lo
	v_add_nc_u32_e32 v20, s17, v11
	v_cmp_eq_u32_e32 vcc_lo, 0, v10
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mul_lo_u32 v14, v26, s24
	v_dual_sub_nc_u32 v1, v6, v1 :: v_dual_bitop2_b32 v19, v19, v18 bitop3:0x14
	v_dual_add_nc_u32 v22, s18, v26 :: v_dual_sub_nc_u32 v12, v3, v12
	v_ashrrev_i32_e32 v21, 31, v20
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mul_lo_u32 v6, v1, s23
	v_dual_sub_nc_u32 v27, v19, v18 :: v_dual_add_nc_u32 v18, s14, v1
	v_sub_nc_u32_e32 v14, v4, v14
	v_ashrrev_i32_e32 v23, 31, v22
	v_lshl_add_u32 v11, v11, 8, v12
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_mul_lo_u32 v16, v27, s25
	v_dual_add_nc_u32 v24, s19, v27 :: v_dual_sub_nc_u32 v6, v2, v6
	v_ashrrev_i32_e32 v19, 31, v18
	v_lshl_add_u32 v26, v26, 8, v14
	v_mul_u64_e32 v[20:21], s[6:7], v[20:21]
	v_ashrrev_i32_e32 v25, 31, v24
	v_lshl_add_u32 v1, v1, 8, v6
	v_sub_nc_u32_e32 v16, v5, v16
	v_mul_u64_e32 v[18:19], s[20:21], v[18:19]
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
	s_or_b32 s27, vcc_lo, s27
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
	s_and_not1_b32 exec_lo, exec_lo, s27
	s_cbranch_execnz .LBB0_45
; %bb.46:
	s_or_b32 exec_lo, exec_lo, s27
	v_cmp_ne_u32_e32 vcc_lo, v8, v9
	v_lshl_or_b32 v0, v9, 8, v0
	v_dual_mov_b32 v6, s16 :: v_dual_mov_b32 v1, s26
	s_and_b32 s0, vcc_lo, exec_lo
	s_or_saveexec_b32 s1, s15
	v_mov_b64_e32 v[2:3], s[12:13]
	s_xor_b32 exec_lo, exec_lo, s1
	s_cbranch_execz .LBB0_43
.LBB0_47:
	s_abs_i32 s2, s23
	s_ashr_i32 s8, s23, 31
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
	s_sub_co_i32 s1, 0, s23
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
	v_mul_lo_u32 v8, v7, s23
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_sub_nc_u32 v4, v4, v8 :: v_dual_add_nc_u32 v8, s14, v7
	v_dual_sub_nc_u32 v4, v4, v9 :: v_dual_ashrrev_i32 v9, 31, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_nc_u32_e32 v4, v0, v4
	v_mul_u64_e32 v[8:9], s[20:21], v[8:9]
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
	.size	bm256_bn192_bk128_wm8_wn1_mc0, .Lfunc_end0-bm256_bn192_bk128_wm8_wn1_mc0
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel bm256_bn192_bk128_wm8_wn1_mc0
		.amdhsa_group_segment_fixed_size 243712
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
		.amdhsa_next_free_vgpr 400
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
		.amdhsa_inst_pref_size 71
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text.bm256_bn192_bk128_wm8_wn1_mc0,"axG",@progbits,bm256_bn192_bk128_wm8_wn1_mc0,comdat
                                        ; -- End function
	.set .Lbm256_bn192_bk128_wm8_wn1_mc0.num_vgpr, 400
	.set .Lbm256_bn192_bk128_wm8_wn1_mc0.num_agpr, 0
	.set .Lbm256_bn192_bk128_wm8_wn1_mc0.numbered_sgpr, 56
	.set .Lbm256_bn192_bk128_wm8_wn1_mc0.num_named_barrier, 0
	.set .Lbm256_bn192_bk128_wm8_wn1_mc0.private_seg_size, 0
	.set .Lbm256_bn192_bk128_wm8_wn1_mc0.uses_vcc, 1
	.set .Lbm256_bn192_bk128_wm8_wn1_mc0.uses_flat_scratch, 1
	.set .Lbm256_bn192_bk128_wm8_wn1_mc0.has_dyn_sized_stack, 0
	.set .Lbm256_bn192_bk128_wm8_wn1_mc0.has_recursion, 0
	.set .Lbm256_bn192_bk128_wm8_wn1_mc0.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 9036
; TotalNumSgprs: 58
; NumVgprs: 400
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 243712 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 24
; NumSGPRsForWavesPerEU: 58
; NumVGPRsForWavesPerEU: 400
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
	.type	__hip_cuid_369a10d5474b9f25,@object ; @__hip_cuid_369a10d5474b9f25
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_369a10d5474b9f25
__hip_cuid_369a10d5474b9f25:
	.byte	0                               ; 0x0
	.size	__hip_cuid_369a10d5474b9f25, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git aa451e1fe6a793394d6733051b1778633063ae96+PATCHED:d17c5aa0e3ea29cde402f58f27e39b6034effa27)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_369a10d5474b9f25
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
    macrotile: [256, 192, 128]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, One]
  MatrixInstruction: [16, 16, 32, 1]
  EnableMatrixInstruction: True
  MIWaveTile: [8, 3]
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
    .group_segment_fixed_size: 243712
    .kernarg_segment_align: 8
    .kernarg_segment_size: 132
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           bm256_bn192_bk128_wm8_wn1_mc0
    .private_segment_fixed_size: 0
    .sgpr_count:     58
    .sgpr_spill_count: 0
    .symbol:         bm256_bn192_bk128_wm8_wn1_mc0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     400
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
