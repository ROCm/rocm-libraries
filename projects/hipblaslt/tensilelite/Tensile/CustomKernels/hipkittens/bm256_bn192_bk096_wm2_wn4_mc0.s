	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.section	.text.bm256_bn192_bk096_wm2_wn4_mc0,"axG",@progbits,bm256_bn192_bk096_wm2_wn4_mc0,comdat
	.protected	bm256_bn192_bk096_wm2_wn4_mc0 ; -- Begin function bm256_bn192_bk096_wm2_wn4_mc0
	.globl	bm256_bn192_bk096_wm2_wn4_mc0
	.p2align	8
	.type	bm256_bn192_bk096_wm2_wn4_mc0,@function
bm256_bn192_bk096_wm2_wn4_mc0: ; @bm256_bn192_bk096_wm2_wn4_mc0
; %bb.0:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1 ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_mov_b64 s[2:3], src_shared_base
	s_mov_b32 s2, 0xcc00
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
	s_cselect_b32 s3, ttmp9, s3
	s_cselect_b32 s33, ttmp7, s5
	s_add_co_i32 s2, s24, 0xff
	s_add_co_i32 s6, s25, 0xbf
	s_ashr_i32 s5, s2, 31
	s_mul_hi_i32 s6, s6, 0x2aaaaaab
	s_lshr_b32 s5, s5, 24
	s_lshr_b32 s7, s6, 31
	s_add_co_i32 s2, s2, s5
	s_ashr_i32 s6, s6, 5
	s_ashr_i32 s5, s2, 8
	s_lshl_b32 s2, s3, 8
	s_add_co_i32 s6, s6, s7
	s_sub_co_i32 s8, s24, s2
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_min_i32 s27, s8, 0x100
	s_cmp_lt_i32 s3, s5
	s_cselect_b32 s38, -1, 0
	s_and_b32 s7, s38, exec_lo
	s_mul_i32 s7, s33, 0xffffff40
	s_cselect_b32 s29, s27, 0
	s_add_co_i32 s7, s25, s7
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_min_i32 s7, s7, 0xc0
	s_cmp_lt_i32 s33, s6
	s_cselect_b32 s25, -1, 0
	s_and_b32 s8, s25, exec_lo
	s_cselect_b32 s31, s7, 0
	s_add_co_i32 s16, s26, 0x5f
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 2, 1), 1
	s_min_i32 s7, s26, 0x60
	s_cmp_gt_i32 s16, 0x5f
	s_cselect_b32 s17, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_and_b32 s8, s17, exec_lo
	s_cselect_b32 s28, s7, 0
	s_cmp_lt_i32 s29, 0x100
	s_cselect_b32 s39, -1, 0
	s_and_b32 vcc_lo, exec_lo, s39
	s_mov_b32 s7, s39
	s_cbranch_vccnz .LBB0_2
; %bb.1:
	s_cmp_lt_i32 s31, 0xc0
	s_cselect_b32 s7, -1, 0
	s_cmp_lt_i32 s28, 0x60
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
	v_cmp_lt_u32_e32 vcc_lo, 0x253f, v1
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
	s_wait_xcnt 0x0
	s_mov_b64 s[0:1], src_shared_base
	v_lshrrev_b32_e32 v203, 5, v0
	s_lshl_b32 s0, s4, 2
	s_add_co_i32 s6, s6, -1
	s_or_b32 s40, s0, 0xcc00
	s_add_co_i32 s34, s5, -1
	s_min_i32 s0, s33, s6
	s_mov_b32 s4, exec_lo
	v_cmpx_lt_i32_e32 0, v203
	s_xor_b32 s35, exec_lo, s4
	s_cbranch_execz .LBB0_12
; %bb.9:
	s_mov_b32 s36, exec_lo
	v_cmpx_eq_u32_e32 1, v203
	s_cbranch_execz .LBB0_11
; %bb.10:
	s_cmp_gt_i32 s28, 0
	s_mul_i32 s4, s0, 0xc0
	s_cselect_b32 s8, -1, 0
	s_ashr_i32 s5, s4, 31
	s_wait_kmcnt 0x0
	s_bfe_i64 s[6:7], s[20:21], 0x200000
	s_mov_b32 s30, s28
	s_mul_u64 s[4:5], s[6:7], s[4:5]
	s_and_b32 s6, s25, s8
	s_lshl_b64 s[4:5], s[4:5], 1
	v_cndmask_b32_e64 v2, 0, 1, s6
	s_add_nc_u64 s[4:5], s[14:15], s[4:5]
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_mov_b32 v1, s40 :: v_dual_mov_b32 v4, s4
	s_and_b32 s5, s5, 0x1ffffff
	v_readfirstlane_b32 s44, v2
	s_bitset1_b32 s5, 31
	v_readfirstlane_b32 s45, v1
	v_mov_b32_e32 v3, s5
	v_readfirstlane_b32 s46, v4
	s_mov_b32 s10, 0
	s_lshr_b32 s4, s31, 16
	s_lshr_b64 s[6:7], s[30:31], 16
	v_readfirstlane_b32 s47, v3
	s_movk_i32 s8, 0xc0
	s_lshl_b32 s5, s28, 16
	s_or_b32 s7, s4, 0x600000
	s_mov_b32 s4, 0x7510000
	s_mov_b32 s11, s10
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[44:47], s[4:11]
.LBB0_11:
	s_or_b32 exec_lo, exec_lo, s36
.LBB0_12:
	s_or_saveexec_b32 s35, s35
	s_min_i32 s30, s3, s34
	s_xor_b32 exec_lo, exec_lo, s35
	s_cbranch_execz .LBB0_14
; %bb.13:
	s_cmp_gt_i32 s28, 0
	s_mov_b32 s10, 0
	s_cselect_b32 s3, -1, 0
	s_lshl_b32 s4, s30, 8
	s_wait_kmcnt 0x0
	s_bfe_i64 s[6:7], s[12:13], 0x200000
	s_ashr_i32 s5, s4, 31
	s_and_b32 s3, s38, s3
	s_mul_u64 s[4:5], s[6:7], s[4:5]
	v_cndmask_b32_e64 v2, 0, 1, s3
	s_lshl_b64 s[6:7], s[4:5], 1
	s_lshr_b32 s4, s29, 16
	s_add_nc_u64 s[6:7], s[18:19], s[6:7]
	s_lshl_b32 s5, s28, 16
	s_and_b32 s7, s7, 0x1ffffff
	v_readfirstlane_b32 s44, v2
	s_bitset1_b32 s7, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v4, s6 :: v_dual_mov_b32 v3, s7
	s_lshr_b64 s[6:7], s[28:29], 16
	s_or_b32 s7, s4, 0x600000
	s_movk_i32 s8, 0x100
	v_readfirstlane_b32 s46, v4
	v_readfirstlane_b32 s47, v3
	s_mov_b32 s4, 0x7510000
	s_mov_b32 s11, s10
	s_mov_b32 s45, s10
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[44:47], s[4:11]
.LBB0_14:
	s_or_b32 exec_lo, exec_lo, s35
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	v_dual_mov_b32 v9, 0 :: v_dual_bitop2_b32 v201, 3, v203 bitop3:0x40
	s_and_b32 s3, s38, s25
	v_and_b32_e32 v199, 0x80, v0
	v_cndmask_b32_e64 v195, 0, 1, s3
	s_and_not1_b32 vcc_lo, exec_lo, s17
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
	v_dual_mov_b32 v82, v9 :: v_dual_mov_b32 v105, v9
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
	v_dual_mov_b32 v186, v9 :: v_dual_mov_b32 v97, v9
	v_dual_mov_b32 v96, v9 :: v_dual_mov_b32 v95, v9
	v_dual_mov_b32 v94, v9 :: v_dual_mov_b32 v93, v9
	v_dual_mov_b32 v92, v9 :: v_dual_mov_b32 v91, v9
	v_mov_b32_e32 v90, v9
	s_barrier_wait -1
	s_cbranch_vccnz .LBB0_37
; %bb.15:
	s_mov_b64 s[4:5], src_shared_base
	s_add_co_i32 s6, s40, 0x9900
	s_mov_b32 s7, s5
	v_mul_u32_u24_e32 v3, 0x60, v199
	s_and_b64 s[6:7], s[6:7], 15
	v_and_b32_e32 v5, 15, v0
	s_sub_co_i32 s4, 16, s6
	v_and_b32_e32 v7, 16, v0
	s_lshr_b32 s4, s4, 2
	s_cmp_lg_u64 s[6:7], 0
	s_mov_b32 s11, 0
	s_cselect_b32 s4, s4, 0
	v_or_b32_e32 v9, v3, v7
	s_lshl2_add_u32 s7, s4, s40
	v_mul_u32_u24_e32 v2, 0x60, v5
	s_add_co_i32 s4, s7, 0x16500
	s_add_co_i32 s42, s7, 0x9900
	s_and_b32 s10, s4, 15
	v_mul_u32_u24_e32 v1, 0x1200, v201
	s_sub_co_i32 s8, 16, s10
	v_or_b32_e32 v11, v9, v2
	s_lshr_b32 s7, s8, 2
	s_cmp_lg_u64 s[10:11], 0
	v_or_b32_e32 v17, 0x1800, v2
	s_cselect_b32 s7, s7, 0
	v_lshrrev_b32_e32 v4, 4, v11
	s_lshl_b32 s10, s7, 2
	s_movk_i32 s7, 0xc00
	s_add_nc_u64 s[34:35], s[4:5], s[10:11]
	s_mul_i32 s4, s0, 0xc0
	s_movk_i32 s0, 0x600
	v_mad_u32_u24 v15, 0x60, v5, s7
	v_mad_u32_u24 v13, 0x60, v5, s0
	v_and_b32_e32 v4, 0x778, v4
	s_mul_hi_i32 s6, s16, 0x2aaaaaab
	s_mov_b32 s43, s5
	s_lshr_b32 s8, s6, 31
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_dual_add_nc_u32 v194, v4, v11 :: v_dual_bitop2_b32 v6, v3, v13 bitop3:0x54
	s_ashr_i32 s44, s6, 4
	v_or_b32_e32 v207, 0x3100, v0
	s_add_co_i32 s44, s44, s8
	v_dual_lshrrev_b32 v2, 4, v6 :: v_dual_add_nc_u32 v8, v9, v15
	v_mad_u32_u24 v49, 0x60, v5, v9
	v_mov_b32_e32 v197, 0
	s_cmp_lt_i32 s31, 0xc0
	s_delay_alu instid0(VALU_DEP_3)
	v_and_b32_e32 v19, 0x7f8, v2
	v_lshrrev_b32_e32 v4, 4, v8
	v_add_nc_u32_e32 v6, 0x1200, v11
	v_add_nc_u32_e32 v8, v9, v17
	v_or_b32_e32 v10, v1, v7
	v_mov_b32_e32 v37, v197
	v_and_b32_e32 v21, 0xff8, v4
	v_lshrrev_b32_e32 v2, 4, v6
	v_add_nc_u32_e32 v4, 0x1e00, v11
	v_lshrrev_b32_e32 v6, 4, v8
	v_add_nc_u32_e32 v8, 0x2400, v11
	v_mad_u32_u24 v44, 0x60, v5, v10
	v_and_b32_e32 v23, 0xff8, v2
	v_lshrrev_b32_e32 v2, 4, v4
	v_and_b32_e32 v25, 0xff8, v6
	v_lshrrev_b32_e32 v4, 4, v8
	v_add_nc_u32_e32 v6, 0x2a00, v11
	v_or_b32_e32 v8, 32, v7
	v_and_b32_e32 v27, 0xff8, v2
	v_lshrrev_b32_e32 v2, 4, v44
	v_and_b32_e32 v29, 0xff8, v4
	v_dual_lshrrev_b32 v4, 4, v6 :: v_dual_add_nc_u32 v6, v10, v13
	v_or_b32_e32 v12, v8, v3
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_and_b32_e32 v2, 0x7f8, v2
	v_mov_b32_e32 v39, v197
	v_and_b32_e32 v31, 0xff8, v4
	v_dual_lshrrev_b32 v4, 4, v6 :: v_dual_add_nc_u32 v10, v10, v15
	v_mad_u32_u24 v14, 0x60, v5, v12
	v_dual_add_nc_u32 v198, v2, v44 :: v_dual_add_nc_u32 v2, v12, v13
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_and_b32_e32 v33, 0x7f8, v4
	v_lshrrev_b32_e32 v6, 4, v10
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_lshrrev_b32_e32 v4, 4, v14
	v_add_nc_u32_e32 v10, 0x1200, v14
	v_lshrrev_b32_e32 v2, 4, v2
	v_dual_mov_b32 v41, v197 :: v_dual_add_nc_u32 v18, 0x2400, v14
	v_and_b32_e32 v45, 0xff8, v4
	v_add_nc_u32_e32 v4, v12, v17
	v_and_b32_e32 v35, 0x7f8, v6
	v_dual_add_nc_u32 v6, v12, v15 :: v_dual_lshrrev_b32 v10, 4, v10
	v_add_nc_u32_e32 v12, 0x1e00, v14
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_lshrrev_b32_e32 v16, 4, v4
	v_and_b32_e32 v196, 0xff8, v2
	v_lshrrev_b32_e32 v6, 4, v6
	v_and_b32_e32 v4, 0x1ff8, v10
	v_lshrrev_b32_e32 v10, 4, v12
	v_or_b32_e32 v12, v1, v8
	v_dual_mov_b32 v9, v197 :: v_dual_add_nc_u32 v14, 0x2a00, v14
	v_and_b32_e32 v2, 0xff8, v6
	v_and_b32_e32 v6, 0xff8, v16
	v_dual_lshrrev_b32 v16, 4, v18 :: v_dual_bitop2_b32 v7, 64, v7 bitop3:0x54
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_lshrrev_b32_e32 v14, 4, v14
	v_and_b32_e32 v8, 0x1ff8, v10
	v_mad_u32_u24 v18, 0x60, v5, v12
	v_and_b32_e32 v10, 0x1ff8, v16
	v_dual_add_nc_u32 v16, v12, v13 :: v_dual_bitop2_b32 v3, v3, v7 bitop3:0x54
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_add_nc_u32 v20, v12, v15 :: v_dual_lshrrev_b32 v18, 4, v18
	v_and_b32_e32 v12, 0x1ff8, v14
	v_lshrrev_b32_e32 v14, 4, v16
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mad_u32_u24 v26, 0x60, v5, v3
	v_lshrrev_b32_e32 v16, 4, v20
	v_and_b32_e32 v46, 0x7f8, v18
	v_dual_add_nc_u32 v18, v3, v13 :: v_dual_add_nc_u32 v22, v3, v15
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_lshrrev_b32_e32 v20, 4, v26
	v_dual_mov_b32 v43, v197 :: v_dual_add_nc_u32 v24, 0x1200, v26
	v_dual_lshrrev_b32 v18, 4, v18 :: v_dual_add_nc_u32 v3, v3, v17
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_and_b32_e32 v47, 0xff8, v20
	v_dual_lshrrev_b32 v20, 4, v22 :: v_dual_lshrrev_b32 v17, 4, v24
	v_add_nc_u32_e32 v24, 0x1e00, v26
	s_delay_alu instid0(VALU_DEP_4)
	v_lshrrev_b32_e32 v3, 4, v3
	v_or_b32_e32 v1, v1, v7
	v_dual_add_nc_u32 v204, v23, v11 :: v_dual_add_nc_u32 v206, v25, v49
	v_and_b32_e32 v22, 0x1ff8, v17
	v_dual_add_nc_u32 v208, v27, v11 :: v_dual_add_nc_u32 v210, v29, v11
	v_dual_add_nc_u32 v212, v31, v11 :: v_dual_add_nc_u32 v214, v33, v44
	v_dual_mov_b32 v11, v197 :: v_dual_lshrrev_b32 v17, 4, v24
	v_add_nc_u32_e32 v7, 0x2a00, v26
	v_add_nc_u32_e32 v13, v1, v13
	v_add_nc_u32_e32 v28, 0x2400, v26
	v_and_b32_e32 v24, 0xff8, v3
	v_and_b32_e32 v26, 0x1ff8, v17
	v_mad_u32_u24 v17, 0x60, v5, v1
	v_dual_lshrrev_b32 v13, 4, v13 :: v_dual_add_nc_u32 v1, v1, v15
	v_dual_mov_b32 v5, v197 :: v_dual_add_nc_u32 v38, 0x1200, v49
	v_dual_mov_b32 v23, v197 :: v_dual_add_nc_u32 v42, 0x1e00, v49
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_and_b32_e32 v32, 0x7f8, v13
	v_dual_mov_b32 v13, v197 :: v_dual_lshrrev_b32 v3, 4, v28
	v_sub_nc_u32_e32 v15, 0x263f, v0
	v_dual_add_nc_u32 v200, v19, v49 :: v_dual_add_nc_u32 v202, v21, v49
	v_add_nc_u64_e32 v[228:229], v[8:9], v[42:43]
	v_and_b32_e32 v28, 0x1ff8, v3
	v_lshrrev_b32_e32 v3, 4, v7
	v_dual_mov_b32 v31, v197 :: v_dual_add_nc_u32 v8, 0xc00, v44
	v_mov_b32_e32 v19, v197
	v_add_nc_u64_e32 v[224:225], v[4:5], v[38:39]
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_and_b32_e32 v30, 0x1ff8, v3
	v_dual_mov_b32 v27, v197 :: v_dual_add_nc_u32 v4, 0x2a00, v49
	v_add_nc_u32_e32 v36, 0x600, v49
	v_dual_mov_b32 v21, v197 :: v_dual_add_nc_u32 v40, 0x1800, v49
	v_add_nc_u64_e32 v[232:233], v[12:13], v[4:5]
	v_add_nc_u64_e32 v[254:255], v[30:31], v[4:5]
	v_dual_mov_b32 v4, v197 :: v_dual_lshrrev_b32 v7, 4, v17
	v_dual_lshrrev_b32 v3, 8, v15 :: v_dual_mov_b32 v15, v197
	v_add_nc_u64_e32 v[218:219], v[196:197], v[36:37]
	v_dual_mov_b32 v17, v197 :: v_dual_add_nc_u32 v196, 0xc00, v49
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_add_nc_u32 v3, 1, v3 :: v_dual_lshrrev_b32 v1, 4, v1
	v_and_b32_e32 v48, 0x7f8, v7
	v_mov_b32_e32 v7, v197
	v_and_b32_e32 v14, 0x7f8, v14
	v_and_b32_e32 v205, 46, v3
	v_and_b32_e32 v16, 0x7f8, v16
	v_and_b32_e32 v18, 0xff8, v18
	v_and_b32_e32 v20, 0xff8, v20
	v_and_b32_e32 v34, 0x7f8, v1
	v_cmp_ne_u32_e64 s0, v3, v205
	v_dual_mov_b32 v3, v197 :: v_dual_add_nc_u32 v216, v35, v44
	v_add_nc_u32_e32 v220, v45, v49
	v_add_nc_u64_e32 v[226:227], v[6:7], v[40:41]
	v_dual_mov_b32 v29, v197 :: v_dual_add_nc_u32 v6, 0x600, v44
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u64_e32 v[222:223], v[2:3], v[196:197]
	v_dual_mov_b32 v25, v197 :: v_dual_add_nc_u32 v2, 0x2400, v49
	v_dual_mov_b32 v33, v197 :: v_dual_mov_b32 v35, v197
	s_cselect_b32 s45, -1, 0
	s_ashr_i32 s5, s4, 31
	s_lshl_b32 s6, s30, 8
	s_wait_kmcnt 0x0
	s_bfe_i64 s[20:21], s[20:21], 0x200000
	v_lshl_or_b32 v50, v205, 8, v0
	s_ashr_i32 s7, s6, 31
	s_mul_u64 s[4:5], s[20:21], s[4:5]
	s_bfe_i64 s[12:13], s[12:13], 0x200000
	s_lshl_b64 s[4:5], s[4:5], 1
	s_mul_u64 s[6:7], s[12:13], s[6:7]
	v_add_nc_u64_e32 v[230:231], v[10:11], v[2:3]
	v_add_nc_u64_e32 v[236:237], v[14:15], v[6:7]
	v_add_nc_u64_e32 v[238:239], v[16:17], v[8:9]
	v_add_nc_u64_e32 v[242:243], v[18:19], v[36:37]
	v_add_nc_u64_e32 v[244:245], v[20:21], v[196:197]
	v_add_nc_u64_e32 v[246:247], v[22:23], v[38:39]
	v_add_nc_u64_e32 v[248:249], v[24:25], v[40:41]
	v_add_nc_u64_e32 v[250:251], v[26:27], v[42:43]
	v_add_nc_u64_e32 v[252:253], v[28:29], v[2:3]
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_add_nc_u64_e32 v[2:3] /*v[258:259]*/, v[32:33], v[6:7]
	v_add_nc_u64_e32 v[4:5] /*v[260:261]*/, v[34:35], v[8:9]
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_or_b32_e32 v1, 0x100, v0
	v_dual_add_nc_u32 v234, v46, v44 :: v_dual_add_nc_u32 v240, v47, v49
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_dual_mov_b32 v7 /*v263*/, v197 :: v_dual_add_nc_u32 v0 /*v256*/, v48, v44
	v_lshl_or_b32 v6 /*v262*/, v0, 2, 0xc800
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_dual_mov_b32 v6, v197 :: v_dual_add_nc_u32 v209, 0xffffff00, v50
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_dual_mov_b32 v9 /*v265*/, v197 :: v_dual_lshlrev_b32 v8 /*v264*/, 2, v50
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_dual_mov_b32 v2, v197 :: v_dual_mov_b32 v8, v197
	v_dual_mov_b32 v10, v197 :: v_dual_mov_b32 v12, v197
	v_dual_mov_b32 v14, v197 :: v_dual_mov_b32 v16, v197
	v_dual_mov_b32 v18, v197 :: v_dual_mov_b32 v20, v197
	v_dual_mov_b32 v22, v197 :: v_dual_mov_b32 v24, v197
	v_dual_mov_b32 v26, v197 :: v_dual_mov_b32 v28, v197
	v_dual_mov_b32 v30, v197 :: v_dual_mov_b32 v32, v197
	v_dual_mov_b32 v34, v197 :: v_dual_mov_b32 v36, v197
	v_dual_mov_b32 v38, v197 :: v_dual_mov_b32 v40, v197
	v_dual_mov_b32 v42, v197 :: v_dual_mov_b32 v44, v197
	v_dual_mov_b32 v45, v197 :: v_dual_mov_b32 v46, v197
	v_dual_mov_b32 v47, v197 :: v_dual_mov_b32 v48, v197
	v_dual_mov_b32 v49, v197 :: v_dual_mov_b32 v50, v197
	v_dual_mov_b32 v51, v197 :: v_dual_mov_b32 v52, v197
	v_dual_mov_b32 v53, v197 :: v_dual_mov_b32 v54, v197
	v_dual_mov_b32 v55, v197 :: v_dual_mov_b32 v56, v197
	v_dual_mov_b32 v57, v197 :: v_dual_mov_b32 v58, v197
	v_dual_mov_b32 v59, v197 :: v_dual_mov_b32 v60, v197
	v_dual_mov_b32 v61, v197 :: v_dual_mov_b32 v62, v197
	v_dual_mov_b32 v63, v197 :: v_dual_mov_b32 v64, v197
	v_dual_mov_b32 v65, v197 :: v_dual_mov_b32 v66, v197
	v_dual_mov_b32 v67, v197 :: v_dual_mov_b32 v68, v197
	v_dual_mov_b32 v69, v197 :: v_dual_mov_b32 v70, v197
	v_dual_mov_b32 v71, v197 :: v_dual_mov_b32 v72, v197
	v_dual_mov_b32 v73, v197 :: v_dual_mov_b32 v74, v197
	v_dual_mov_b32 v75, v197 :: v_dual_mov_b32 v76, v197
	v_dual_mov_b32 v77, v197 :: v_dual_mov_b32 v78, v197
	v_dual_mov_b32 v79, v197 :: v_dual_mov_b32 v80, v197
	v_dual_mov_b32 v81, v197 :: v_dual_mov_b32 v82, v197
	v_dual_mov_b32 v83, v197 :: v_dual_mov_b32 v84, v197
	v_dual_mov_b32 v85, v197 :: v_dual_mov_b32 v86, v197
	v_dual_mov_b32 v87, v197 :: v_dual_mov_b32 v88, v197
	v_dual_mov_b32 v89, v197 :: v_dual_mov_b32 v98, v197
	v_dual_mov_b32 v99, v197 :: v_dual_mov_b32 v100, v197
	v_dual_mov_b32 v101, v197 :: v_dual_mov_b32 v102, v197
	v_dual_mov_b32 v103, v197 :: v_dual_mov_b32 v104, v197
	v_dual_mov_b32 v105, v197 :: v_dual_mov_b32 v106, v197
	v_dual_mov_b32 v107, v197 :: v_dual_mov_b32 v108, v197
	v_dual_mov_b32 v109, v197 :: v_dual_mov_b32 v110, v197
	v_dual_mov_b32 v111, v197 :: v_dual_mov_b32 v112, v197
	v_dual_mov_b32 v113, v197 :: v_dual_mov_b32 v114, v197
	v_dual_mov_b32 v115, v197 :: v_dual_mov_b32 v116, v197
	v_dual_mov_b32 v117, v197 :: v_dual_mov_b32 v118, v197
	v_dual_mov_b32 v119, v197 :: v_dual_mov_b32 v120, v197
	v_dual_mov_b32 v121, v197 :: v_dual_mov_b32 v122, v197
	v_dual_mov_b32 v123, v197 :: v_dual_mov_b32 v124, v197
	v_dual_mov_b32 v125, v197 :: v_dual_mov_b32 v126, v197
	v_dual_mov_b32 v127, v197 :: v_dual_mov_b32 v128, v197
	v_dual_mov_b32 v129, v197 :: v_dual_mov_b32 v130, v197
	v_dual_mov_b32 v131, v197 :: v_dual_mov_b32 v132, v197
	v_dual_mov_b32 v133, v197 :: v_dual_mov_b32 v134, v197
	v_dual_mov_b32 v135, v197 :: v_dual_mov_b32 v136, v197
	v_dual_mov_b32 v137, v197 :: v_dual_mov_b32 v138, v197
	v_dual_mov_b32 v139, v197 :: v_dual_mov_b32 v140, v197
	v_dual_mov_b32 v141, v197 :: v_dual_mov_b32 v142, v197
	v_dual_mov_b32 v143, v197 :: v_dual_mov_b32 v144, v197
	v_dual_mov_b32 v145, v197 :: v_dual_mov_b32 v146, v197
	v_dual_mov_b32 v147, v197 :: v_dual_mov_b32 v148, v197
	v_dual_mov_b32 v149, v197 :: v_dual_mov_b32 v150, v197
	v_dual_mov_b32 v151, v197 :: v_dual_mov_b32 v152, v197
	v_dual_mov_b32 v153, v197 :: v_dual_mov_b32 v154, v197
	v_dual_mov_b32 v155, v197 :: v_dual_mov_b32 v156, v197
	v_dual_mov_b32 v157, v197 :: v_dual_mov_b32 v158, v197
	v_dual_mov_b32 v159, v197 :: v_dual_mov_b32 v160, v197
	v_dual_mov_b32 v161, v197 :: v_dual_mov_b32 v162, v197
	v_dual_mov_b32 v163, v197 :: v_dual_mov_b32 v164, v197
	v_dual_mov_b32 v165, v197 :: v_dual_mov_b32 v166, v197
	v_dual_mov_b32 v167, v197 :: v_dual_mov_b32 v168, v197
	v_dual_mov_b32 v169, v197 :: v_dual_mov_b32 v170, v197
	v_dual_mov_b32 v171, v197 :: v_dual_mov_b32 v172, v197
	v_dual_mov_b32 v173, v197 :: v_dual_mov_b32 v174, v197
	v_dual_mov_b32 v175, v197 :: v_dual_mov_b32 v176, v197
	v_dual_mov_b32 v177, v197 :: v_dual_mov_b32 v178, v197
	v_dual_mov_b32 v179, v197 :: v_dual_mov_b32 v180, v197
	v_dual_mov_b32 v181, v197 :: v_dual_mov_b32 v182, v197
	v_dual_mov_b32 v183, v197 :: v_dual_mov_b32 v184, v197
	v_dual_mov_b32 v185, v197 :: v_dual_mov_b32 v186, v197
	v_dual_mov_b32 v187, v197 :: v_dual_mov_b32 v188, v197
	v_dual_mov_b32 v189, v197 :: v_dual_mov_b32 v190, v197
	v_dual_mov_b32 v191, v197 :: v_dual_mov_b32 v192, v197
	v_dual_mov_b32 v193, v197 :: v_dual_mov_b32 v90, v197
	v_dual_mov_b32 v91, v197 :: v_dual_mov_b32 v92, v197
	v_dual_mov_b32 v93, v197 :: v_dual_mov_b32 v94, v197
	v_dual_mov_b32 v95, v197 :: v_dual_mov_b32 v96, v197
	v_mov_b32_e32 v97, v197
	s_lshr_b32 s46, s31, 16
	s_lshr_b32 s47, s29, 16
	s_add_nc_u64 s[20:21], s[14:15], s[4:5]
	s_lshl_b64 s[4:5], s[6:7], 1
	s_mov_b32 s41, s1
	s_movk_i32 s16, 0xc0
	s_or_b32 s46, s46, 0x600000
	s_or_b32 s47, s47, 0x600000
	s_add_nc_u64 s[36:37], s[18:19], s[4:5]
	s_mov_b32 s48, -1
	s_movk_i32 s8, 0x100
	s_mov_b32 s4, 0x7510000
	s_mov_b32 s49, s11
	s_branch .LBB0_17
.LBB0_16:                               ;   in Loop: Header=BB0_17 Depth=1
	s_cmp_eq_u32 s49, s44
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
	s_cbranch_vccnz .LBB0_29
; %bb.18:                               ;   in Loop: Header=BB0_17 Depth=1
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_mov_b64_e32 v[10:11] /*v[266:267]*/, v[0:1]
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_mov_b32_e32 v211, 50
	s_cmp_lg_u32 s5, 0
	s_mov_b32 s12, 0
	s_cselect_b32 s7, s43, s1
	s_cselect_b32 s6, s42, 0
.LBB0_19:                               ;   Parent Loop BB0_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_set_vgpr_msb 1                        ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_dual_mov_b32 v196, v10 /*v266*/ :: v_dual_add_nc_u32 v211, -2, v211
	s_set_vgpr_msb 0x145                    ;  msbs: dst=1 src0=1 src1=1 src2=0
	v_dual_mov_b32 v12 /*v268*/, v11 /*v267*/ :: v_dual_add_nc_u32 v11 /*v267*/, 0x200, v11 /*v267*/
	s_set_vgpr_msb 0x4544                   ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_dual_mov_b32 v13 /*v269*/, v197 :: v_dual_add_nc_u32 v10 /*v266*/, 0x200, v10 /*v266*/
	s_set_vgpr_msb 0x4440                   ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_cmp_eq_u32_e32 vcc_lo, 0, v211
	v_lshl_add_u64 v[14:15] /*v[270:271]*/, v[196:197], 2, s[6:7]
	s_set_vgpr_msb 0x4041                   ;  msbs: dst=1 src0=1 src1=0 src2=0
	v_lshl_add_u64 v[12:13] /*v[268:269]*/, v[12:13] /*v[268:269]*/, 2, s[6:7]
	s_clause 0x1
	flat_store_b32 v[14:15] /*v[270:271]*/, v197
	flat_store_b32 v[12:13] /*v[268:269]*/, v197
	s_or_b32 s12, vcc_lo, s12
	s_wait_xcnt 0x0
	s_and_not1_b32 exec_lo, exec_lo, s12
	s_set_vgpr_msb 0x4100                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_cbranch_execnz .LBB0_19
; %bb.20:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s12
	s_and_saveexec_b32 s12, s48
	s_cbranch_execz .LBB0_23
; %bb.21:                               ;   in Loop: Header=BB0_17 Depth=1
	s_set_vgpr_msb 0x44                     ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_add_nc_u64_e32 v[10:11] /*v[266:267]*/, s[6:7], v[6:7] /*v[262:263]*/
	s_set_vgpr_msb 0x4400                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_mov_b32_e32 v196, v207
	s_mov_b32 s6, 0
.LBB0_22:                               ;   Parent Loop BB0_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v196, 0x100, v196
	s_set_vgpr_msb 1                        ;  msbs: dst=0 src0=1 src1=0 src2=0
	flat_store_b32 v[10:11] /*v[266:267]*/, v197
	s_set_vgpr_msb 0x144                    ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_add_nc_u64_e32 v[10:11] /*v[266:267]*/, 0x400, v[10:11] /*v[266:267]*/
	s_set_vgpr_msb 0x4400                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_cmp_lt_u32_e32 vcc_lo, 0x31ff, v196
	s_or_b32 s6, vcc_lo, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s6
	s_cbranch_execnz .LBB0_22
.LBB0_23:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s12
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_mov_b64_e32 v[10:11] /*v[266:267]*/, v[0:1]
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_mov_b32_e32 v211, v205
	s_cmp_lg_u32 s5, 0
	s_mov_b32 s12, 0
	s_cselect_b32 s7, s35, s41
	s_cselect_b32 s6, s34, s40
.LBB0_24:                               ;   Parent Loop BB0_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_set_vgpr_msb 1                        ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_dual_mov_b32 v196, v10 /*v266*/ :: v_dual_add_nc_u32 v211, -2, v211
	s_set_vgpr_msb 0x145                    ;  msbs: dst=1 src0=1 src1=1 src2=0
	v_dual_mov_b32 v12 /*v268*/, v11 /*v267*/ :: v_dual_add_nc_u32 v11 /*v267*/, 0x200, v11 /*v267*/
	s_set_vgpr_msb 0x4544                   ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_dual_mov_b32 v13 /*v269*/, v197 :: v_dual_add_nc_u32 v10 /*v266*/, 0x200, v10 /*v266*/
	s_set_vgpr_msb 0x4440                   ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_cmp_eq_u32_e32 vcc_lo, 0, v211
	v_lshl_add_u64 v[14:15] /*v[270:271]*/, v[196:197], 2, s[6:7]
	s_set_vgpr_msb 0x4041                   ;  msbs: dst=1 src0=1 src1=0 src2=0
	v_lshl_add_u64 v[12:13] /*v[268:269]*/, v[12:13] /*v[268:269]*/, 2, s[6:7]
	s_clause 0x1
	flat_store_b32 v[14:15] /*v[270:271]*/, v197
	flat_store_b32 v[12:13] /*v[268:269]*/, v197
	s_or_b32 s12, vcc_lo, s12
	s_wait_xcnt 0x0
	s_and_not1_b32 exec_lo, exec_lo, s12
	s_set_vgpr_msb 0x4100                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_cbranch_execnz .LBB0_24
; %bb.25:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s12
	s_and_saveexec_b32 s12, s0
	s_cbranch_execz .LBB0_28
; %bb.26:                               ;   in Loop: Header=BB0_17 Depth=1
	s_set_vgpr_msb 0x44                     ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_add_nc_u64_e32 v[10:11] /*v[266:267]*/, s[6:7], v[8:9] /*v[264:265]*/
	s_set_vgpr_msb 0x4400                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_mov_b32_e32 v196, v209
	s_mov_b32 s6, 0
.LBB0_27:                               ;   Parent Loop BB0_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v196, 0x100, v196
	s_set_vgpr_msb 1                        ;  msbs: dst=0 src0=1 src1=0 src2=0
	flat_store_b32 v[10:11] /*v[266:267]*/, v197
	s_set_vgpr_msb 0x144                    ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_add_nc_u64_e32 v[10:11] /*v[266:267]*/, 0x400, v[10:11] /*v[266:267]*/
	s_set_vgpr_msb 0x4400                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_cmp_lt_u32_e32 vcc_lo, 0x253f, v196
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
	s_cselect_b32 s6, s49, 0
	s_mov_b32 s7, exec_lo
	v_cmpx_lt_i32_e32 0, v203
	s_xor_b32 s7, exec_lo, s7
	s_cbranch_execnz .LBB0_32
; %bb.30:                               ;   in Loop: Header=BB0_17 Depth=1
	s_and_not1_saveexec_b32 s12, s7
	s_cbranch_execnz .LBB0_35
.LBB0_31:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s12
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s3
	s_cbranch_vccnz .LBB0_16
	s_branch .LBB0_36
.LBB0_32:                               ;   in Loop: Header=BB0_17 Depth=1
	s_mov_b32 s51, exec_lo
	v_cmpx_eq_u32_e32 1, v203
	s_cbranch_execz .LBB0_34
; %bb.33:                               ;   in Loop: Header=BB0_17 Depth=1
	s_cmp_lg_u32 s5, 0
	s_mul_i32 s10, s6, 0x60
	s_cselect_b32 s14, s34, s40
	s_cmp_gt_i32 s28, 0
	s_mov_b32 s30, s28
	s_cselect_b32 s15, -1, 0
	s_lshl_b64 s[12:13], s[10:11], 1
	s_mov_b32 s17, s9
	s_add_nc_u64 s[12:13], s[20:21], s[12:13]
	s_mov_b32 s18, s11
	s_and_b32 s10, s13, 0x1ffffff
	s_and_b32 s13, s25, s15
	s_bitset1_b32 s10, 31
	v_cndmask_b32_e64 v196, 0, 1, s13
	v_dual_mov_b32 v211, s14 :: v_dual_mov_b32 v213, s10
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_mov_b32_e32 v10 /*v266*/, s12
	s_lshr_b64 s[14:15], s[30:31], 16
	v_readfirstlane_b32 s52, v196
	v_readfirstlane_b32 s53, v211
	v_readfirstlane_b32 s55, v213
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_readfirstlane_b32 s54, v10 /*v266*/
	s_lshl_b32 s13, s28, 16
	s_mov_b32 s12, s4
	s_mov_b32 s15, s46
	s_mov_b32 s19, s11
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[52:55], s[12:19]
	s_set_vgpr_msb 0x100                    ;  msbs: dst=0 src0=0 src1=0 src2=0
.LBB0_34:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s51
	s_and_not1_saveexec_b32 s12, s7
	s_cbranch_execz .LBB0_31
.LBB0_35:                               ;   in Loop: Header=BB0_17 Depth=1
	s_cmp_lg_u32 s5, 0
	s_mul_i32 s10, s6, 0x60
	s_cselect_b32 s5, s42, 0
	s_cmp_gt_i32 s28, 0
	s_cselect_b32 s13, -1, 0
	s_lshl_b64 s[6:7], s[10:11], 1
	s_and_b32 s10, s38, s13
	s_add_nc_u64 s[6:7], s[36:37], s[6:7]
	v_cndmask_b32_e64 v196, 0, 1, s10
	s_and_b32 s7, s7, 0x1ffffff
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_mov_b32_e32 v10 /*v266*/, s6
	s_bitset1_b32 s7, 31
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_dual_mov_b32 v211, s5 :: v_dual_mov_b32 v213, s7
	v_readfirstlane_b32 s52, v196
	s_set_vgpr_msb 1                        ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_readfirstlane_b32 s54, v10 /*v266*/
	s_lshr_b64 s[6:7], s[28:29], 16
	s_set_vgpr_msb 0x100                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_readfirstlane_b32 s53, v211
	v_readfirstlane_b32 s55, v213
	s_lshl_b32 s5, s28, 16
	s_mov_b32 s7, s47
	s_mov_b32 s10, s11
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	tensor_load_to_lds s[52:55], s[4:11]
	s_or_b32 exec_lo, exec_lo, s12
	s_and_not1_b32 vcc_lo, exec_lo, s3
	s_cbranch_vccnz .LBB0_16
.LBB0_36:                               ;   in Loop: Header=BB0_17 Depth=1
	s_cmp_lg_u32 s50, 0
	s_cselect_b32 s6, s42, 0
	s_cselect_b32 s5, s34, s40
	v_lshl_add_u32 v196, v194, 1, s6
	v_lshl_add_u32 v211, v200, 1, s6
	v_lshl_add_u32 v213, v202, 1, s6
	v_lshl_add_u32 v215, v204, 1, s6
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[10:13] /*v[266:269]*/, v196
	ds_load_b128 v[14:17] /*v[270:273]*/, v196 offset:16
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v196, v206, 1, s6
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[18:21] /*v[274:277]*/, v211 offset:3072
	ds_load_b128 v[22:25] /*v[278:281]*/, v211 offset:3088
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v211, v208, 1, s6
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[26:29] /*v[282:285]*/, v213 offset:6144
	ds_load_b128 v[42:45] /*v[298:301]*/, v196 offset:12288
	ds_load_b128 v[46:49] /*v[302:305]*/, v196 offset:12304
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v196, v214, 1, s5
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[30:33] /*v[286:289]*/, v213 offset:6160
	ds_load_b128 v[34:37] /*v[290:293]*/, v215 offset:9216
	ds_load_b128 v[38:41] /*v[294:297]*/, v215 offset:9232
	ds_load_b128 v[50:53] /*v[306:309]*/, v211 offset:15360
	ds_load_b128 v[66:69] /*v[322:325]*/, v196 offset:3072
	ds_load_b128 v[70:73] /*v[326:329]*/, v196 offset:3088
	ds_load_b128 v[54:57] /*v[310:313]*/, v211 offset:15376
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v196, v212, 1, s6
	v_lshl_add_u32 v213, v216, 1, s5
	v_lshl_add_u32 v211, v218, 1, s6
	; sched_group_barrier mask(0x00000100) size(11) SyncID(0)
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[50:57], v[50:57] /*v[306:313]*/, v[66:73] /*v[322:329]*/, v[50:57] matrix_b_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[74:77] /*v[330:333]*/, v196 offset:21504
	ds_load_b128 v[78:81] /*v[334:337]*/, v196 offset:21520
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v196, v210, 1, s6
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[74:81], v[42:49] /*v[298:305]*/, v[66:73] /*v[322:329]*/, v[74:81] matrix_b_reuse
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[2:9], v[74:81] /*v[330:337]*/, v[66:73] /*v[322:329]*/, v[2:9] matrix_a_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[82:85] /*v[338:341]*/, v196 offset:18432
	ds_load_b128 v[86:89] /*v[342:345]*/, v196 offset:18448
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v196, v198, 1, s5
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[26:33], v[82:89] /*v[338:345]*/, v[66:73] /*v[322:329]*/, v[26:33] matrix_b_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[58:61] /*v[314:317]*/, v213 offset:6144
	ds_load_b128 v[62:65] /*v[318:321]*/, v213 offset:6160
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v213, v222, 1, s6
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[170:177], v[10:17] /*v[266:273]*/, v[58:65] /*v[314:321]*/, v[170:177] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[146:153], v[18:25] /*v[274:281]*/, v[58:65] /*v[314:321]*/, v[146:153] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[122:129], v[26:33] /*v[282:289]*/, v[58:65] /*v[314:321]*/, v[122:129] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[98:105], v[34:41] /*v[290:297]*/, v[58:65] /*v[314:321]*/, v[98:105] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[66:73], v[42:49] /*v[298:305]*/, v[58:65] /*v[314:321]*/, v[66:73] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[90:97], v[74:81] /*v[330:337]*/, v[58:65] /*v[314:321]*/, v[90:97] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[18:25], v[82:89] /*v[338:345]*/, v[58:65] /*v[314:321]*/, v[18:25] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[42:49], v[50:57] /*v[306:313]*/, v[58:65] /*v[314:321]*/, v[42:49] matrix_b_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[58:61] /*v[314:317]*/, v196
	ds_load_b128 v[62:65] /*v[318:321]*/, v196 offset:16
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v196, v220, 1, s6
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[186:193], v[10:17] /*v[266:273]*/, v[58:65] /*v[314:321]*/, v[186:193]
	v_wmma_f32_16x16x32_bf16 v[162:169], v[18:25] /*v[274:281]*/, v[58:65] /*v[314:321]*/, v[162:169] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[138:145], v[26:33] /*v[282:289]*/, v[58:65] /*v[314:321]*/, v[138:145] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[114:121], v[34:41] /*v[290:297]*/, v[58:65] /*v[314:321]*/, v[114:121] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[82:89], v[42:49] /*v[298:305]*/, v[58:65] /*v[314:321]*/, v[82:89] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[58:65], v[50:57] /*v[306:313]*/, v[58:65] /*v[314:321]*/, v[58:65] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[34:41], v[82:89] /*v[338:345]*/, v[58:65] /*v[314:321]*/, v[34:41] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[10:17], v[74:81] /*v[330:337]*/, v[58:65] /*v[314:321]*/, v[10:17] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(12) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(11) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[154:161], v[18:25] /*v[274:281]*/, v[66:73] /*v[322:329]*/, v[154:161] matrix_b_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[18:21] /*v[274:277]*/, v211 offset:64
	ds_load_b128 v[22:25] /*v[278:281]*/, v211 offset:80
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v211, v226, 1, s6
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[58:61] /*v[314:317]*/, v211 offset:64
	s_set_vgpr_msb 0x4005                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[178:185], v[10:17] /*v[266:273]*/, v[66:73] /*v[322:329]*/, v[178:185] matrix_b_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[10:13] /*v[266:269]*/, v196 offset:64
	ds_load_b128 v[14:17] /*v[270:273]*/, v196 offset:80
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v196, v224, 1, s6
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[62:65] /*v[318:321]*/, v211 offset:80
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v211, v230, 1, s6
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[90:93] /*v[346:349]*/, v211 offset:64
	s_set_vgpr_msb 0x4005                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[106:113], v[34:41] /*v[290:297]*/, v[66:73] /*v[322:329]*/, v[106:113] matrix_b_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[34:37] /*v[290:293]*/, v196 offset:64
	ds_load_b128 v[38:41] /*v[294:297]*/, v196 offset:80
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v196, v228, 1, s6
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[94:97] /*v[350:353]*/, v211 offset:80
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v211, v236, 1, s5
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[74:77] /*v[330:333]*/, v196 offset:64
	s_set_vgpr_msb 0x4005                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[130:137], v[26:33] /*v[282:289]*/, v[66:73] /*v[322:329]*/, v[130:137] matrix_b_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[26:29] /*v[282:285]*/, v213 offset:64
	ds_load_b128 v[30:33] /*v[286:289]*/, v213 offset:80
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v213, v232, 1, s6
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[78:81] /*v[334:337]*/, v196 offset:80
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v196, v234, 1, s5
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[106:109] /*v[362:365]*/, v211 offset:64
	ds_load_b128 v[110:113] /*v[366:369]*/, v211 offset:80
	ds_load_b128 v[98:101] /*v[354:357]*/, v213 offset:64
	ds_load_b128 v[102:105] /*v[358:361]*/, v213 offset:80
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v213, v238, 1, s5
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[82:85] /*v[338:341]*/, v196 offset:64
	ds_load_b128 v[86:89] /*v[342:345]*/, v196 offset:80
	; sched_group_barrier mask(0x00000008) size(12) SyncID(0)
	ds_load_b128 v[114:117] /*v[370:373]*/, v213 offset:64
	ds_load_b128 v[118:121] /*v[374:377]*/, v213 offset:80
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v196, v240, 1, s6
	v_lshl_add_u32 v211, v242, 1, s6
	v_lshl_add_u32 v213, v244, 1, s6
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[42:45] /*v[298:301]*/, v196 offset:128
	ds_load_b128 v[46:49] /*v[302:305]*/, v196 offset:144
	ds_load_b128 v[50:53] /*v[306:309]*/, v211 offset:128
	ds_load_b128 v[54:57] /*v[310:313]*/, v211 offset:144
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v196, v246, 1, s6
	v_lshl_add_u32 v211, v248, 1, s6
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[66:69] /*v[322:325]*/, v213 offset:128
	ds_load_b128 v[70:73] /*v[326:329]*/, v213 offset:144
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v213, v250, 1, s6
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[122:125] /*v[378:381]*/, v196 offset:128
	ds_load_b128 v[126:129] /*v[382:385]*/, v196 offset:144
	ds_load_b128 v[130:133] /*v[386:389]*/, v211 offset:128
	ds_load_b128 v[134:137] /*v[390:393]*/, v211 offset:144
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v196, v252, 1, s6
	v_lshl_add_u32 v211, v254, 1, s6
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[138:141] /*v[394:397]*/, v213 offset:128
	s_set_vgpr_msb 0x4005                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0xd
	v_wmma_f32_16x16x32_bf16 v[186:193], v[10:17] /*v[266:273]*/, v[82:89] /*v[338:345]*/, v[186:193]
	; sched_group_barrier mask(0x00000100) size(11) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[162:169], v[18:25] /*v[274:281]*/, v[82:89] /*v[338:345]*/, v[162:169] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[138:145], v[26:33] /*v[282:289]*/, v[82:89] /*v[338:345]*/, v[138:145] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[114:121], v[34:41] /*v[290:297]*/, v[82:89] /*v[338:345]*/, v[114:121] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[82:89], v[58:65] /*v[314:321]*/, v[82:89] /*v[338:345]*/, v[82:89] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[58:65], v[74:81] /*v[330:337]*/, v[82:89] /*v[338:345]*/, v[58:65] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[34:41], v[90:97] /*v[346:353]*/, v[82:89] /*v[338:345]*/, v[34:41] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[10:17], v[98:105] /*v[354:361]*/, v[82:89] /*v[338:345]*/, v[10:17] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[2:9], v[98:105] /*v[354:361]*/, v[106:113] /*v[362:369]*/, v[2:9] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[26:33], v[90:97] /*v[346:353]*/, v[106:113] /*v[362:369]*/, v[26:33] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[50:57], v[74:81] /*v[330:337]*/, v[106:113] /*v[362:369]*/, v[50:57] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[74:81], v[58:65] /*v[314:321]*/, v[106:113] /*v[362:369]*/, v[74:81] matrix_b_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[142:145] /*v[398:401]*/, v213 offset:144
	ds_load_b128 v[82:85] /*v[338:341]*/, v196 offset:128
	ds_load_b128 v[86:89] /*v[342:345]*/, v196 offset:144
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_lshl_add_u32 v196, v0 /*v256*/, 1, s5
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[146:149] /*v[402:405]*/, v211 offset:128
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_lshl_add_u32 v213, v2 /*v258*/, 1, s5
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[150:153] /*v[406:409]*/, v211 offset:144
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_lshl_add_u32 v211, v4 /*v260*/, 1, s5
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[154:157] /*v[410:413]*/, v196 offset:128
	ds_load_b128 v[158:161] /*v[414:417]*/, v196 offset:144
	ds_load_b128 v[162:165] /*v[418:421]*/, v213 offset:128
	ds_load_b128 v[166:169] /*v[422:425]*/, v213 offset:144
	ds_load_b128 v[170:173] /*v[426:429]*/, v211 offset:128
	ds_load_b128 v[174:177] /*v[430:433]*/, v211 offset:144
	s_set_vgpr_msb 0x4005                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[106:113], v[34:41] /*v[290:297]*/, v[106:113] /*v[362:369]*/, v[106:113] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(12) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(11) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[130:137], v[26:33] /*v[282:289]*/, v[106:113] /*v[362:369]*/, v[130:137] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[154:161], v[18:25] /*v[274:281]*/, v[106:113] /*v[362:369]*/, v[154:161] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[178:185], v[10:17] /*v[266:273]*/, v[106:113] /*v[362:369]*/, v[178:185] matrix_b_reuse
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[170:177], v[10:17] /*v[266:273]*/, v[114:121] /*v[370:377]*/, v[170:177] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[146:153], v[18:25] /*v[274:281]*/, v[114:121] /*v[370:377]*/, v[146:153] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[122:129], v[26:33] /*v[282:289]*/, v[114:121] /*v[370:377]*/, v[122:129] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[98:105], v[34:41] /*v[290:297]*/, v[114:121] /*v[370:377]*/, v[98:105] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[66:73], v[58:65] /*v[314:321]*/, v[114:121] /*v[370:377]*/, v[66:73] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[42:49], v[74:81] /*v[330:337]*/, v[114:121] /*v[370:377]*/, v[42:49] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[18:25], v[90:97] /*v[346:353]*/, v[114:121] /*v[370:377]*/, v[18:25] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[90:97], v[98:105] /*v[354:361]*/, v[114:121] /*v[370:377]*/, v[90:97] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(12) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[186:193], v[42:49] /*v[298:305]*/, v[154:161] /*v[410:417]*/, v[186:193]
	; sched_group_barrier mask(0x00000100) size(11) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[162:169], v[50:57] /*v[306:313]*/, v[154:161] /*v[410:417]*/, v[162:169] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[138:145], v[66:73] /*v[322:329]*/, v[154:161] /*v[410:417]*/, v[138:145] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[114:121], v[122:129] /*v[378:385]*/, v[154:161] /*v[410:417]*/, v[114:121] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[82:89], v[130:137] /*v[386:393]*/, v[154:161] /*v[410:417]*/, v[82:89] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[58:65], v[138:145] /*v[394:401]*/, v[154:161] /*v[410:417]*/, v[58:65] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[34:41], v[82:89] /*v[338:345]*/, v[154:161] /*v[410:417]*/, v[34:41] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[10:17], v[146:153] /*v[402:409]*/, v[154:161] /*v[410:417]*/, v[10:17] matrix_b_reuse
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[2:9], v[146:153] /*v[402:409]*/, v[162:169] /*v[418:425]*/, v[2:9] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[26:33], v[82:89] /*v[338:345]*/, v[162:169] /*v[418:425]*/, v[26:33] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[50:57], v[138:145] /*v[394:401]*/, v[162:169] /*v[418:425]*/, v[50:57] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[74:81], v[130:137] /*v[386:393]*/, v[162:169] /*v[418:425]*/, v[74:81] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(12) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(11) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[106:113], v[122:129] /*v[378:385]*/, v[162:169] /*v[418:425]*/, v[106:113] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[130:137], v[66:73] /*v[322:329]*/, v[162:169] /*v[418:425]*/, v[130:137] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[154:161], v[50:57] /*v[306:313]*/, v[162:169] /*v[418:425]*/, v[154:161] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[178:185], v[42:49] /*v[298:305]*/, v[162:169] /*v[418:425]*/, v[178:185] matrix_b_reuse
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[170:177], v[42:49] /*v[298:305]*/, v[170:177] /*v[426:433]*/, v[170:177] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[146:153], v[50:57] /*v[306:313]*/, v[170:177] /*v[426:433]*/, v[146:153] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[122:129], v[66:73] /*v[322:329]*/, v[170:177] /*v[426:433]*/, v[122:129] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[98:105], v[122:129] /*v[378:385]*/, v[170:177] /*v[426:433]*/, v[98:105] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[66:73], v[130:137] /*v[386:393]*/, v[170:177] /*v[426:433]*/, v[66:73] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[42:49], v[138:145] /*v[394:401]*/, v[170:177] /*v[426:433]*/, v[42:49] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[18:25], v[82:89] /*v[338:345]*/, v[170:177] /*v[426:433]*/, v[18:25] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[90:97], v[146:153] /*v[402:409]*/, v[170:177] /*v[426:433]*/, v[90:97] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(12) SyncID(0)
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
	s_and_b32 vcc_lo, exec_lo, s3
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_vccz .LBB0_39
; %bb.38:
	v_mul_u32_u24_e32 v1, 48, v201
	v_lshrrev_b32_e32 v194, 1, v0
	v_cvt_pk_bf16_f32 v185, v184, v185
	v_cvt_pk_bf16_f32 v184, v182, v183
	v_cvt_pk_bf16_f32 v182, v178, v179
	v_and_or_b32 v1, v0, 15, v1
	v_and_or_b32 v194, v194, 8, v199
	v_cvt_pk_bf16_f32 v177, v176, v177
	v_cvt_pk_bf16_f32 v176, v174, v175
	v_cvt_pk_bf16_f32 v175, v172, v173
	v_cvt_pk_bf16_f32 v174, v170, v171
	v_lshl_or_b32 v1, v1, 8, v194
	v_cvt_pk_bf16_f32 v183, v180, v181
	v_cvt_pk_bf16_f32 v169, v168, v169
	v_cvt_pk_bf16_f32 v168, v166, v167
	v_cvt_pk_bf16_f32 v167, v164, v165
	v_add_nc_u32_e32 v178, 0x1000, v1
	v_lshrrev_b32_e32 v170, 3, v1
	v_add_nc_u32_e32 v171, 0x2000, v1
	v_add_nc_u32_e32 v173, 16, v1
	v_add_nc_u32_e32 v180, 0x1010, v1
	v_dual_lshrrev_b32 v172, 3, v178 :: v_dual_lshlrev_b32 v178, 1, v1
	v_and_b32_e32 v170, 0x1ff0, v170
	v_add_nc_u32_e32 v164, 0x2010, v1
	v_cvt_pk_bf16_f32 v161, v160, v161
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v172, 0x1ffffffe, v172
	v_cvt_pk_bf16_f32 v160, v158, v159
	v_cvt_pk_bf16_f32 v159, v156, v157
	v_add_nc_u32_e32 v156, 32, v1
	v_add_nc_u32_e32 v157, 0x1020, v1
	v_cvt_pk_bf16_f32 v145, v144, v145
	v_cvt_pk_bf16_f32 v144, v142, v143
	v_cvt_pk_bf16_f32 v143, v140, v141
	v_add_nc_u32_e32 v140, 0x2020, v1
	v_cvt_pk_bf16_f32 v137, v136, v137
	v_cvt_pk_bf16_f32 v136, v134, v135
	v_cvt_pk_bf16_f32 v135, v132, v133
	v_add_nc_u32_e32 v132, 48, v1
	v_add_nc_u32_e32 v133, 0x1030, v1
	v_cvt_pk_bf16_f32 v121, v120, v121
	v_cvt_pk_bf16_f32 v120, v118, v119
	v_cvt_pk_bf16_f32 v119, v116, v117
	v_add_nc_u32_e32 v116, 0x2030, v1
	v_cvt_pk_bf16_f32 v113, v112, v113
	v_cvt_pk_bf16_f32 v112, v110, v111
	v_cvt_pk_bf16_f32 v111, v108, v109
	v_add_nc_u32_e32 v108, 64, v1
	v_cvt_pk_bf16_f32 v105, v104, v105
	v_cvt_pk_bf16_f32 v104, v102, v103
	v_cvt_pk_bf16_f32 v103, v100, v101
	v_cvt_pk_bf16_f32 v89, v88, v89
	v_cvt_pk_bf16_f32 v88, v86, v87
	v_add_nc_u32_e32 v101, 0x2040, v1
	v_cvt_pk_bf16_f32 v87, v84, v85
	v_add_nc_u32_e32 v84, 0x50, v1
	v_add_nc_u32_e32 v109, 0x1040, v1
	v_cvt_pk_bf16_f32 v193, v192, v193
	v_cvt_pk_bf16_f32 v192, v190, v191
	v_cvt_pk_bf16_f32 v191, v188, v189
	v_cvt_pk_bf16_f32 v190, v186, v187
	v_dual_lshrrev_b32 v171, 3, v171 :: v_dual_lshrrev_b32 v179, 3, v173
	v_dual_add_nc_u32 v170, v170, v178 :: v_dual_add_nc_u32 v172, v172, v178
	v_lshrrev_b32_e32 v180, 3, v180
	v_cvt_pk_bf16_f32 v166, v162, v163
	v_lshrrev_b32_e32 v163, 3, v164
	v_cvt_pk_bf16_f32 v158, v154, v155
	v_dual_lshrrev_b32 v155, 3, v156 :: v_dual_lshrrev_b32 v157, 3, v157
	v_cvt_pk_bf16_f32 v142, v138, v139
	v_lshrrev_b32_e32 v139, 3, v140
	v_cvt_pk_bf16_f32 v134, v130, v131
	v_dual_lshrrev_b32 v131, 3, v132 :: v_dual_lshrrev_b32 v133, 3, v133
	v_cvt_pk_bf16_f32 v118, v114, v115
	v_lshrrev_b32_e32 v115, 3, v116
	v_cvt_pk_bf16_f32 v110, v106, v107
	v_lshrrev_b32_e32 v107, 3, v108
	v_cvt_pk_bf16_f32 v86, v82, v83
	v_cvt_pk_bf16_f32 v81, v80, v81
	v_lshrrev_b32_e32 v83, 3, v101
	v_cvt_pk_bf16_f32 v80, v78, v79
	v_cvt_pk_bf16_f32 v78, v74, v75
	v_dual_lshrrev_b32 v75, 3, v84 :: v_dual_lshrrev_b32 v109, 3, v109
	v_and_b32_e32 v171, 0x1ffffffe, v171
	v_and_b32_e32 v179, 0x7ff0, v179
	ds_store_b128 v170, v[190:193]
	ds_store_b128 v172, v[182:185] offset:8192
	v_and_b32_e32 v172, 0xffffff0, v180
	v_lshlrev_b32_e32 v173, 1, v173
	v_and_b32_e32 v154, 0xffffff0, v163
	v_cvt_pk_bf16_f32 v153, v152, v153
	v_and_b32_e32 v155, 0x7ff0, v155
	v_cvt_pk_bf16_f32 v152, v150, v151
	v_cvt_pk_bf16_f32 v151, v148, v149
	v_cvt_pk_bf16_f32 v150, v146, v147
	v_and_b32_e32 v147, 0xffffff0, v157
	v_lshlrev_b32_e32 v148, 1, v156
	v_and_b32_e32 v130, 0xffffff0, v139
	v_cvt_pk_bf16_f32 v129, v128, v129
	v_and_b32_e32 v131, 0x7ff0, v131
	v_cvt_pk_bf16_f32 v128, v126, v127
	v_cvt_pk_bf16_f32 v127, v124, v125
	v_cvt_pk_bf16_f32 v126, v122, v123
	v_and_b32_e32 v123, 0xffffff0, v133
	v_lshlrev_b32_e32 v124, 1, v132
	v_and_b32_e32 v106, 0xffffff0, v115
	v_and_b32_e32 v107, 0x7ff0, v107
	v_lshlrev_b32_e32 v100, 1, v108
	v_and_b32_e32 v74, 0xffffff0, v83
	v_and_b32_e32 v75, 0x7ff0, v75
	v_cvt_pk_bf16_f32 v102, v98, v99
	v_and_b32_e32 v99, 0xffffff0, v109
	v_add_nc_u32_e32 v170, v171, v178
	v_dual_add_nc_u32 v171, v179, v178 :: v_dual_add_nc_u32 v162, v172, v173
	v_dual_add_nc_u32 v154, v154, v173 :: v_dual_add_nc_u32 v146, v155, v178
	v_dual_add_nc_u32 v138, v147, v148 :: v_dual_add_nc_u32 v130, v130, v148
	v_add_nc_u32_e32 v122, v131, v178
	v_dual_add_nc_u32 v114, v123, v124 :: v_dual_add_nc_u32 v106, v106, v124
	v_add_nc_u32_e32 v98, v107, v178
	v_cvt_pk_bf16_f32 v79, v76, v77
	v_add_nc_u32_e32 v74, v74, v100
	v_cvt_pk_bf16_f32 v73, v72, v73
	v_cvt_pk_bf16_f32 v72, v70, v71
	v_add_nc_u32_e32 v76, 0x1050, v1
	v_cvt_pk_bf16_f32 v71, v68, v69
	v_cvt_pk_bf16_f32 v70, v66, v67
	v_add_nc_u32_e32 v66, v75, v178
	v_cvt_pk_bf16_f32 v65, v64, v65
	v_cvt_pk_bf16_f32 v64, v62, v63
	v_cvt_pk_bf16_f32 v63, v60, v61
	v_cvt_pk_bf16_f32 v62, v58, v59
	v_add_nc_u32_e32 v61, 0x60, v1
	v_cvt_pk_bf16_f32 v57, v56, v57
	v_cvt_pk_bf16_f32 v56, v54, v55
	v_cvt_pk_bf16_f32 v55, v52, v53
	v_add_nc_u32_e32 v52, 0x1060, v1
	v_add_nc_u32_e32 v82, v99, v100
	v_add_nc_u32_e32 v60, 0x2050, v1
	v_cvt_pk_bf16_f32 v41, v40, v41
	v_cvt_pk_bf16_f32 v40, v38, v39
	v_cvt_pk_bf16_f32 v38, v34, v35
	v_add_nc_u32_e32 v34, 0x2060, v1
	v_cvt_pk_bf16_f32 v39, v36, v37
	v_add_nc_u32_e32 v37, 0x70, v1
	v_cvt_pk_bf16_f32 v33, v32, v33
	v_cvt_pk_bf16_f32 v32, v30, v31
	v_cvt_pk_bf16_f32 v31, v28, v29
	v_add_nc_u32_e32 v28, 0x1070, v1
	v_add_nc_u32_e32 v1, 0x2070, v1
	ds_store_b128 v170, v[174:177] offset:16384
	ds_store_b128 v171, v[166:169] offset:32
	ds_store_b128 v162, v[158:161] offset:8192
	ds_store_b128 v154, v[150:153] offset:16384
	ds_store_b128 v146, v[142:145] offset:64
	ds_store_b128 v138, v[134:137] offset:8192
	ds_store_b128 v130, v[126:129] offset:16384
	ds_store_b128 v122, v[118:121] offset:96
	ds_store_b128 v114, v[110:113] offset:8192
	ds_store_b128 v106, v[102:105] offset:16384
	ds_store_b128 v98, v[86:89] offset:128
	ds_store_b128 v82, v[78:81] offset:8192
	v_lshrrev_b32_e32 v67, 3, v76
	ds_store_b128 v74, v[70:73] offset:16384
	ds_store_b128 v66, v[62:65] offset:160
	v_lshrrev_b32_e32 v62, 3, v61
	v_cvt_pk_bf16_f32 v49, v48, v49
	v_cvt_pk_bf16_f32 v48, v46, v47
	v_cvt_pk_bf16_f32 v46, v42, v43
	v_lshrrev_b32_e32 v43, 3, v52
	v_dual_lshrrev_b32 v60, 3, v60 :: v_dual_lshrrev_b32 v34, 3, v34
	v_cvt_pk_bf16_f32 v30, v26, v27
	v_lshrrev_b32_e32 v26, 3, v37
	v_cvt_pk_bf16_f32 v25, v24, v25
	v_cvt_pk_bf16_f32 v24, v22, v23
	v_cvt_pk_bf16_f32 v23, v20, v21
	v_dual_lshrrev_b32 v20, 3, v28 :: v_dual_lshrrev_b32 v1, 3, v1
	v_and_b32_e32 v58, 0xffffff0, v67
	v_lshlrev_b32_e32 v59, 1, v84
	v_cvt_pk_bf16_f32 v54, v50, v51
	v_and_b32_e32 v51, 0x7ff0, v62
	v_and_b32_e32 v35, 0xffffff0, v43
	v_lshlrev_b32_e32 v36, 1, v61
	v_and_b32_e32 v60, 0xffffff0, v60
	v_and_b32_e32 v34, 0xffffff0, v34
	v_and_b32_e32 v26, 0x7ff0, v26
	v_cvt_pk_bf16_f32 v22, v18, v19
	v_and_b32_e32 v19, 0xffffff0, v20
	v_lshlrev_b32_e32 v20, 1, v37
	v_and_b32_e32 v1, 0xffffff0, v1
	v_dual_add_nc_u32 v58, v58, v59 :: v_dual_add_nc_u32 v42, v51, v178
	v_dual_add_nc_u32 v35, v35, v36 :: v_dual_add_nc_u32 v50, v60, v59
	v_cvt_pk_bf16_f32 v47, v44, v45
	v_add_nc_u32_e32 v27, v34, v36
	v_add_nc_u32_e32 v18, v26, v178
	v_cvt_pk_bf16_f32 v17, v16, v17
	v_cvt_pk_bf16_f32 v16, v14, v15
	v_cvt_pk_bf16_f32 v15, v12, v13
	v_cvt_pk_bf16_f32 v14, v10, v11
	v_add_nc_u32_e32 v10, v19, v20
	v_cvt_pk_bf16_f32 v9, v8, v9
	v_cvt_pk_bf16_f32 v8, v6, v7
	v_cvt_pk_bf16_f32 v7, v4, v5
	v_cvt_pk_bf16_f32 v6, v2, v3
	v_add_nc_u32_e32 v1, v1, v20
	v_cvt_pk_bf16_f32 v5, v96, v97
	v_cvt_pk_bf16_f32 v4, v94, v95
	v_cvt_pk_bf16_f32 v3, v92, v93
	v_cvt_pk_bf16_f32 v2, v90, v91
	ds_store_b128 v58, v[54:57] offset:8192
	ds_store_b128 v50, v[46:49] offset:16384
	ds_store_b128 v42, v[38:41] offset:192
	ds_store_b128 v35, v[30:33] offset:8192
	ds_store_b128 v27, v[22:25] offset:16384
	ds_store_b128 v18, v[14:17] offset:224
	ds_store_b128 v10, v[6:9] offset:8192
	ds_store_b128 v1, v[2:5] offset:16384
.LBB0_39:
	v_cmp_ne_u32_e32 vcc_lo, 1, v195
	;;#ASMSTART
	s_wait_dscnt 0
	;;#ASMEND
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_vccnz .LBB0_50
; %bb.40:
	s_wait_kmcnt 0x0
	s_mul_i32 s14, s31, s29
	s_mov_b32 s0, exec_lo
	v_cmpx_gt_i32_e64 s14, v0
	s_cbranch_execz .LBB0_50
; %bb.41:
	s_ashr_i32 s3, s2, 31
	v_xad_u32 v2, v0, -1, s14
	s_lshl_b64 s[0:1], s[2:3], 1
	s_mul_i32 s15, s33, 0xc0
	s_ashr_i32 s25, s24, 31
	s_add_nc_u64 s[4:5], s[22:23], s[0:1]
	s_mov_b32 s0, 0
                                        ; implicit-def: $vgpr1
                                        ; implicit-def: $vgpr6
                                        ; implicit-def: $sgpr12_sgpr13
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_u32_e32 0x2ff, v2
	s_xor_b32 s3, exec_lo, s1
	s_cbranch_execnz .LBB0_44
; %bb.42:
	s_or_saveexec_b32 s1, s3
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
	s_mov_b32 s17, s15
	s_mov_b32 s18, s15
	v_and_b32_e32 v9, 0x1fffffc, v8
	s_mov_b32 s19, s15
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
	v_dual_add_nc_u32 v22, s18, v26 :: v_dual_sub_nc_u32 v12, v3, v12
	v_ashrrev_i32_e32 v21, 31, v20
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mul_lo_u32 v6, v1, s27
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
	v_dual_mov_b32 v6, s16 :: v_dual_mov_b32 v1, s23
	s_and_b32 s0, vcc_lo, exec_lo
	s_or_saveexec_b32 s1, s3
	v_mov_b64_e32 v[2:3], s[12:13]
	s_xor_b32 exec_lo, exec_lo, s1
	s_cbranch_execz .LBB0_43
.LBB0_47:
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
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_sub_nc_u32 v7, v4, v9 :: v_dual_lshlrev_b32 v4, 8, v4
	v_lshlrev_b32_e32 v9, 8, v9
	v_mul_lo_u32 v8, v7, s27
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_sub_nc_u32 v4, v4, v8 :: v_dual_add_nc_u32 v8, s15, v7
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
	s_cbranch_execnz .LBB0_49
.LBB0_50:
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end0:
	.size	bm256_bn192_bk096_wm2_wn4_mc0, .Lfunc_end0-bm256_bn192_bk096_wm2_wn4_mc0
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel bm256_bn192_bk096_wm2_wn4_mc0
		.amdhsa_group_segment_fixed_size 182784
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
		.amdhsa_next_free_vgpr 434
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
		.amdhsa_inst_pref_size 79
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text.bm256_bn192_bk096_wm2_wn4_mc0,"axG",@progbits,bm256_bn192_bk096_wm2_wn4_mc0,comdat
                                        ; -- End function
	.set .Lbm256_bn192_bk096_wm2_wn4_mc0.num_vgpr, 434
	.set .Lbm256_bn192_bk096_wm2_wn4_mc0.num_agpr, 0
	.set .Lbm256_bn192_bk096_wm2_wn4_mc0.numbered_sgpr, 56
	.set .Lbm256_bn192_bk096_wm2_wn4_mc0.num_named_barrier, 0
	.set .Lbm256_bn192_bk096_wm2_wn4_mc0.private_seg_size, 0
	.set .Lbm256_bn192_bk096_wm2_wn4_mc0.uses_vcc, 1
	.set .Lbm256_bn192_bk096_wm2_wn4_mc0.uses_flat_scratch, 1
	.set .Lbm256_bn192_bk096_wm2_wn4_mc0.has_dyn_sized_stack, 0
	.set .Lbm256_bn192_bk096_wm2_wn4_mc0.has_recursion, 0
	.set .Lbm256_bn192_bk096_wm2_wn4_mc0.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 10112
; TotalNumSgprs: 58
; NumVgprs: 434
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 182784 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 27
; NumSGPRsForWavesPerEU: 58
; NumVGPRsForWavesPerEU: 434
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
	.type	__hip_cuid_5a41ce6d94381b4b,@object ; @__hip_cuid_5a41ce6d94381b4b
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_5a41ce6d94381b4b
__hip_cuid_5a41ce6d94381b4b:
	.byte	0                               ; 0x0
	.size	__hip_cuid_5a41ce6d94381b4b, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git aa451e1fe6a793394d6733051b1778633063ae96+PATCHED:d17c5aa0e3ea29cde402f58f27e39b6034effa27)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_5a41ce6d94381b4b
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
    macrotile: [256, 192, 96]
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
    .group_segment_fixed_size: 182784
    .kernarg_segment_align: 8
    .kernarg_segment_size: 132
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           bm256_bn192_bk096_wm2_wn4_mc0
    .private_segment_fixed_size: 0
    .sgpr_count:     58
    .sgpr_spill_count: 0
    .symbol:         bm256_bn192_bk096_wm2_wn4_mc0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     434
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
