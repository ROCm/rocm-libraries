	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.section	.text.bm256_bn192_bk064_wm4_wn2_mc1,"axG",@progbits,bm256_bn192_bk064_wm4_wn2_mc1,comdat
	.protected	bm256_bn192_bk064_wm4_wn2_mc1 ; -- Begin function bm256_bn192_bk064_wm4_wn2_mc1
	.globl	bm256_bn192_bk064_wm4_wn2_mc1
	.p2align	8
	.type	bm256_bn192_bk064_wm4_wn2_mc1,@function
bm256_bn192_bk064_wm4_wn2_mc1: ; @bm256_bn192_bk064_wm4_wn2_mc1
; %bb.0:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1 ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_load_b96 s[28:30], s[0:1], 0x78 nv
	s_mov_b64 s[2:3], src_shared_base
	s_mov_b32 s2, 0x8800
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
	s_sub_co_i32 s7, s28, s2
	s_ashr_i32 s8, s5, 31
	s_add_co_i32 s6, s29, 0xbf
	s_min_i32 s31, s7, 0x100
	s_lshr_b32 s7, s8, 24
	s_mul_hi_i32 s6, s6, 0x2aaaaaab
	s_add_co_i32 s5, s5, s7
	s_lshr_b32 s8, s6, 31
	s_ashr_i32 s6, s6, 5
	s_ashr_i32 s5, s5, 8
	s_add_co_i32 s6, s6, s8
	s_cmp_lt_i32 s3, s5
	s_mul_i32 s7, s33, 0xffffff40
	s_cselect_b32 s42, -1, 0
	s_mov_b32 s9, s30
	s_and_b32 s8, s42, exec_lo
	s_cselect_b32 s35, s31, 0
	s_add_co_i32 s7, s29, s7
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_min_i32 s7, s7, 0xc0
	s_cmp_lt_i32 s33, s6
	s_cselect_b32 s29, -1, 0
	s_and_b32 s8, s29, exec_lo
	s_cselect_b32 s37, s7, 0
	s_add_co_i32 s12, s30, 63
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 2, 1), 1
	s_min_i32 s7, s30, 64
	s_cmp_gt_i32 s12, 63
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
	s_cmp_lt_i32 s37, 0xc0
	s_cselect_b32 s7, -1, 0
	s_cmp_lt_i32 s34, 64
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
	v_cmp_lt_u32_e32 vcc_lo, 0x20ff, v5
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
	ds_store_b32 v2, v3 offset:34816
	v_add_nc_u32_e32 v2, 0x400, v2
	v_cmp_lt_u32_e32 vcc_lo, 0x187f, v1
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
	s_lshl_b32 s0, s4, 2
	s_lshl_b32 s4, s33, 2
	v_lshrrev_b32_e32 v203, 5, v0
	s_add_co_i32 s6, s6, -1
	s_and_b32 s4, s4, 12
	s_mov_b64 s[38:39], src_shared_base
	s_or_b32 s38, s0, 0x8800
	s_add_co_i32 s16, s5, -1
	s_min_i32 s1, s33, s6
	s_and_b32 s18, s3, 3
	s_lshl_b32 s17, 15, s4
	s_mov_b32 s4, exec_lo
	v_cmpx_lt_i32_e32 0, v203
	s_xor_b32 s19, exec_lo, s4
	s_cbranch_execz .LBB0_12
; %bb.9:
	s_mov_b32 s40, exec_lo
	v_cmpx_eq_u32_e32 1, v203
	s_cbranch_execz .LBB0_11
; %bb.10:
	s_cmp_gt_i32 s34, 0
	s_mul_i32 s4, s1, 0xc0
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
	s_movk_i32 s8, 0xc0
	s_or_b32 s4, s17, 0x7510000
	s_lshl_b32 s5, s34, 16
	s_or_b32 s7, s11, 0x400000
	s_mov_b32 s11, s10
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[44:47], s[4:11]
.LBB0_11:
	s_or_b32 exec_lo, exec_lo, s40
.LBB0_12:
	s_or_saveexec_b32 s36, s19
	s_min_i32 s19, s3, s16
	s_lshl_b32 s18, 0x1111, s18
	s_xor_b32 exec_lo, exec_lo, s36
	s_cbranch_execz .LBB0_14
; %bb.13:
	s_cmp_gt_i32 s34, 0
	s_mov_b32 s10, 0
	s_cselect_b32 s3, -1, 0
	s_lshl_b32 s4, s19, 8
	s_wait_kmcnt 0x0
	s_bfe_i64 s[6:7], s[24:25], 0x200000
	s_ashr_i32 s5, s4, 31
	s_and_b32 s3, s42, s3
	s_mul_u64 s[4:5], s[6:7], s[4:5]
	v_cndmask_b32_e64 v2, 0, 1, s3
	s_lshl_b64 s[6:7], s[4:5], 1
	s_lshr_b32 s3, s35, 16
	s_add_nc_u64 s[6:7], s[14:15], s[6:7]
	s_or_b32 s4, s18, 0x7510000
	s_and_b32 s7, s7, 0x1ffffff
	v_readfirstlane_b32 s44, v2
	s_bitset1_b32 s7, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v4, s6 :: v_dual_mov_b32 v3, s7
	s_lshr_b64 s[6:7], s[34:35], 16
	s_lshl_b32 s5, s34, 16
	s_or_b32 s7, s3, 0x400000
	v_readfirstlane_b32 s46, v4
	v_readfirstlane_b32 s47, v3
	s_movk_i32 s8, 0x100
	s_mov_b32 s11, s10
	s_mov_b32 s45, s10
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[44:47], s[4:11]
.LBB0_14:
	s_or_b32 exec_lo, exec_lo, s36
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
	v_and_b32_e32 v199, 0xc0, v0
	v_cndmask_b32_e64 v195, 0, 1, s3
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_ne_u32_e32 vcc_lo, 0, v1
	v_dual_mov_b32 v8, v9 :: v_dual_mov_b32 v7, v9
	v_dual_mov_b32 v6, v9 :: v_dual_mov_b32 v5, v9
	v_cndmask_b32_e64 v201, 0, 0x60, vcc_lo
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
	v_dual_mov_b32 v26, v9 :: v_dual_mov_b32 v73, v9
	v_dual_mov_b32 v72, v9 :: v_dual_mov_b32 v71, v9
	v_dual_mov_b32 v70, v9 :: v_dual_mov_b32 v69, v9
	v_dual_mov_b32 v68, v9 :: v_dual_mov_b32 v67, v9
	v_dual_mov_b32 v66, v9 :: v_dual_mov_b32 v41, v9
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
	v_dual_mov_b32 v58, v9 :: v_dual_mov_b32 v81, v9
	v_dual_mov_b32 v80, v9 :: v_dual_mov_b32 v79, v9
	v_dual_mov_b32 v78, v9 :: v_dual_mov_b32 v77, v9
	v_dual_mov_b32 v76, v9 :: v_dual_mov_b32 v75, v9
	v_dual_mov_b32 v74, v9 :: v_dual_mov_b32 v105, v9
	v_dual_mov_b32 v104, v9 :: v_dual_mov_b32 v103, v9
	v_dual_mov_b32 v102, v9 :: v_dual_mov_b32 v101, v9
	v_dual_mov_b32 v100, v9 :: v_dual_mov_b32 v99, v9
	v_dual_mov_b32 v98, v9 :: v_dual_mov_b32 v89, v9
	v_dual_mov_b32 v88, v9 :: v_dual_mov_b32 v87, v9
	v_dual_mov_b32 v86, v9 :: v_dual_mov_b32 v85, v9
	v_dual_mov_b32 v84, v9 :: v_dual_mov_b32 v83, v9
	v_dual_mov_b32 v82, v9 :: v_dual_mov_b32 v97, v9
	v_dual_mov_b32 v96, v9 :: v_dual_mov_b32 v95, v9
	v_dual_mov_b32 v94, v9 :: v_dual_mov_b32 v93, v9
	v_dual_mov_b32 v92, v9 :: v_dual_mov_b32 v91, v9
	v_dual_mov_b32 v90, v9 :: v_dual_mov_b32 v113, v9
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
	v_dual_mov_b32 v186, v9 :: v_dual_mov_b32 v121, v9
	v_dual_mov_b32 v120, v9 :: v_dual_mov_b32 v119, v9
	v_dual_mov_b32 v118, v9 :: v_dual_mov_b32 v117, v9
	v_dual_mov_b32 v116, v9 :: v_dual_mov_b32 v115, v9
	v_mov_b32_e32 v114, v9
	s_movk_i32 s16, 0xc0
	s_and_not1_b32 vcc_lo, exec_lo, s13
	s_mov_b32 s11, 0
	s_barrier_wait -1
	s_barrier_wait -3
	s_cbranch_vccnz .LBB0_41
; %bb.17:
	v_dual_lshlrev_b32 v4, 6, v0 :: v_dual_bitop2_b32 v1, 16, v0 bitop3:0x40
	v_dual_lshlrev_b32 v2, 6, v201 :: v_dual_lshlrev_b32 v3, 6, v199
	s_mov_b64 s[4:5], src_shared_base
	s_or_b32 s6, s0, 0xee00
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_and_or_b32 v4, 0x3c0, v4, v1
	s_mov_b32 s7, s5
	v_or_b32_e32 v207, 0x2100, v0
	s_and_b64 s[6:7], s[6:7], 15
	v_dual_mov_b32 v197, 0 :: v_dual_bitop2_b32 v5, v4, v3 bitop3:0x54
	s_sub_co_i32 s4, 16, s6
	v_sub_nc_u32_e32 v24, 0x197f, v0
	s_lshr_b32 s4, s4, 2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_dual_lshrrev_b32 v1, 4, v5 :: v_dual_bitop2_b32 v6, v2, v4 bitop3:0x54
	v_or_b32_e32 v7, 0x800, v5
	v_or_b32_e32 v8, 0xc00, v5
	s_cmp_lg_u64 s[6:7], 0
	v_or_b32_e32 v10, 0x400, v6
	v_and_b32_e32 v1, 0x338, v1
	s_cselect_b32 s4, s4, 0
	v_lshrrev_b32_e32 v8, 4, v8
	s_lshl2_add_u32 s6, s4, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_lshrrev_b32 v9, 4, v6 :: v_dual_add_nc_u32 v194, v1, v5
	v_or_b32_e32 v1, 0x400, v5
	v_lshrrev_b32_e32 v10, 4, v10
	v_dual_mov_b32 v239, v197 :: v_dual_add_nc_u32 v12, 0x1000, v6
	v_and_b32_e32 v9, 0x1b8, v9
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_lshrrev_b32 v1, 4, v1 :: v_dual_lshrrev_b32 v7, 4, v7
	v_add_nc_u32_e32 v13, 0x1400, v6
	v_dual_add_nc_u32 v198, v9, v6 :: v_dual_bitop2_b32 v14, 32, v4 bitop3:0x54
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_and_b32_e32 v11, 0x378, v1
	v_add_nc_u32_e32 v1, 0x800, v6
	v_and_b32_e32 v9, 0x1f8, v10
	v_dual_mov_b32 v237, v197 :: v_dual_add_nc_u32 v10, 0xc00, v6
	v_dual_lshrrev_b32 v1, 4, v1 :: v_dual_bitop2_b32 v15, v14, v3 bitop3:0x54
	v_and_b32_e32 v7, 0x3b8, v7
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_lshrrev_b32_e32 v10, 4, v10
	v_or_b32_e32 v14, v14, v2
	v_or_b32_e32 v17, 0x400, v15
	v_and_b32_e32 v16, 0x3b8, v1
	v_lshrrev_b32_e32 v1, 4, v15
	v_and_b32_e32 v10, 0x3f8, v10
	s_add_co_i32 s4, s6, 0x17600
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_add_nc_u32 v202, v7, v5 :: v_dual_add_nc_u32 v208, v16, v6
	v_and_b32_e32 v19, 0x338, v1
	v_lshrrev_b32_e32 v1, 4, v17
	v_or_b32_e32 v18, 0x800, v15
	v_or_b32_e32 v15, 0xc00, v15
	v_or_b32_e32 v17, 0x400, v14
	s_and_b32 s10, s4, 15
	v_lshrrev_b32_e32 v20, 4, v14
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_lshrrev_b32 v18, 4, v18 :: v_dual_lshrrev_b32 v15, 4, v15
	v_and_b32_e32 v21, 0x3f8, v1
	v_dual_lshrrev_b32 v1, 4, v17 :: v_dual_add_nc_u32 v3, v4, v3
	v_add_nc_u32_e32 v210, v10, v6
	v_dual_add_nc_u32 v2, v4, v2 :: v_dual_mov_b32 v4, v197
	v_dual_lshrrev_b32 v12, 4, v12 :: v_dual_lshrrev_b32 v13, 4, v13
	v_dual_mov_b32 v7, v197 :: v_dual_lshrrev_b32 v24, 8, v24
	v_add_nc_u32_e32 v200, v11, v5
	s_mul_i32 s0, s1, 0xc0
	s_sub_co_i32 s1, 16, s10
	s_add_co_i32 s45, s6, 0xee00
	s_lshr_b32 s1, s1, 2
	s_cmp_lg_u64 s[10:11], 0
	v_and_b32_e32 v8, 0x3f8, v8
	v_and_b32_e32 v12, 0x3b8, v12
	v_and_b32_e32 v13, 0x3f8, v13
	v_and_b32_e32 v17, 0x3f8, v18
	v_and_b32_e32 v15, 0x3f8, v15
	v_and_b32_e32 v18, 0x1b8, v20
	v_add_nc_u32_e32 v20, 0x800, v14
	v_and_b32_e32 v22, 0x1f8, v1
	v_add_nc_u32_e32 v1, 0xc00, v14
	v_add_nc_u32_e32 v23, 0x1000, v14
	v_add_nc_u32_e32 v14, 0x1400, v14
	v_dual_add_nc_u32 v24, 1, v24 :: v_dual_add_nc_u32 v204, v8, v5
	s_cselect_b32 s1, s1, 0
	s_ashr_i32 s6, s12, 31
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_nc_u32 v206, v9, v6 :: v_dual_bitop2_b32 v205, 26, v24 bitop3:0x40
	s_lshr_b32 s6, s6, 26
	v_dual_add_nc_u32 v212, v12, v6 :: v_dual_add_nc_u32 v214, v13, v6
	s_add_co_i32 s12, s12, s6
	v_add_nc_u32_e32 v216, v19, v3
	v_add3_u32 v218, v3, v21, 0x400
	v_add3_u32 v220, v3, v17, 0x800
	v_add3_u32 v222, v3, v15, 0xc00
	v_dual_mov_b32 v3, v197 :: v_dual_mov_b32 v5, v197
	v_dual_lshrrev_b32 v14, 4, v14 :: v_dual_mov_b32 v6, v197
	v_dual_lshrrev_b32 v20, 4, v20 :: v_dual_lshrrev_b32 v1, 4, v1
	v_dual_mov_b32 v8, v197 :: v_dual_lshrrev_b32 v23, 4, v23
	s_lshl_b32 s10, s1, 2
	s_ashr_i32 s47, s12, 6
	s_cmp_lt_i32 s37, 0xc0
	v_and_b32_e32 v20, 0x3f8, v20
	s_cselect_b32 s48, -1, 0
	s_lshl_b32 s6, s19, 8
	v_and_b32_e32 v25, 0x3f8, v1
	v_and_b32_e32 v23, 0x3f8, v23
	v_and_b32_e32 v14, 0x3f8, v14
	v_lshl_or_b32 v26, v205, 8, v0
	s_add_nc_u64 s[40:41], s[4:5], s[10:11]
	s_ashr_i32 s1, s0, 31
	s_ashr_i32 s7, s6, 31
	s_or_b32 s4, s18, 0x7510000
	s_wait_kmcnt 0x0
	s_bfe_i64 s[18:19], s[20:21], 0x200000
	s_bfe_i64 s[20:21], s[24:25], 0x200000
	s_mul_u64 s[18:19], s[18:19], s[0:1]
	s_mul_u64 s[6:7], s[20:21], s[6:7]
	v_cmp_eq_u32_e64 s0, 0, v203
	v_or_b32_e32 v1, 0x100, v0
	v_cmp_ne_u32_e64 s1, v24, v205
	v_dual_mov_b32 v9, v197 :: v_dual_add_nc_u32 v224, v18, v2
	v_add3_u32 v226, v2, v22, 0x400
	v_add3_u32 v228, v2, v20, 0x800
	v_add3_u32 v230, v2, v25, 0xc00
	v_add3_u32 v232, v2, v23, 0x1000
	v_add3_u32 v234, v2, v14, 0x1400
	v_lshl_or_b32 v236, v0, 2, 0x8800
	v_dual_mov_b32 v10, v197 :: v_dual_add_nc_u32 v209, 0xffffff00, v26
	v_dual_mov_b32 v11, v197 :: v_dual_lshlrev_b32 v238, 2, v26
	v_dual_mov_b32 v2, v197 :: v_dual_mov_b32 v12, v197
	v_dual_mov_b32 v13, v197 :: v_dual_mov_b32 v14, v197
	v_dual_mov_b32 v15, v197 :: v_dual_mov_b32 v16, v197
	v_dual_mov_b32 v17, v197 :: v_dual_mov_b32 v18, v197
	v_dual_mov_b32 v19, v197 :: v_dual_mov_b32 v20, v197
	v_dual_mov_b32 v21, v197 :: v_dual_mov_b32 v22, v197
	v_dual_mov_b32 v23, v197 :: v_dual_mov_b32 v24, v197
	v_dual_mov_b32 v25, v197 :: v_dual_mov_b32 v26, v197
	v_dual_mov_b32 v27, v197 :: v_dual_mov_b32 v28, v197
	v_dual_mov_b32 v29, v197 :: v_dual_mov_b32 v30, v197
	v_dual_mov_b32 v31, v197 :: v_dual_mov_b32 v32, v197
	v_dual_mov_b32 v33, v197 :: v_dual_mov_b32 v66, v197
	v_dual_mov_b32 v67, v197 :: v_dual_mov_b32 v68, v197
	v_dual_mov_b32 v69, v197 :: v_dual_mov_b32 v70, v197
	v_dual_mov_b32 v71, v197 :: v_dual_mov_b32 v72, v197
	v_dual_mov_b32 v73, v197 :: v_dual_mov_b32 v34, v197
	v_dual_mov_b32 v35, v197 :: v_dual_mov_b32 v36, v197
	v_dual_mov_b32 v37, v197 :: v_dual_mov_b32 v38, v197
	v_dual_mov_b32 v39, v197 :: v_dual_mov_b32 v40, v197
	v_dual_mov_b32 v41, v197 :: v_dual_mov_b32 v42, v197
	v_dual_mov_b32 v43, v197 :: v_dual_mov_b32 v44, v197
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
	v_dual_mov_b32 v65, v197 :: v_dual_mov_b32 v74, v197
	v_dual_mov_b32 v75, v197 :: v_dual_mov_b32 v76, v197
	v_dual_mov_b32 v77, v197 :: v_dual_mov_b32 v78, v197
	v_dual_mov_b32 v79, v197 :: v_dual_mov_b32 v80, v197
	v_dual_mov_b32 v81, v197 :: v_dual_mov_b32 v98, v197
	v_dual_mov_b32 v99, v197 :: v_dual_mov_b32 v100, v197
	v_dual_mov_b32 v101, v197 :: v_dual_mov_b32 v102, v197
	v_dual_mov_b32 v103, v197 :: v_dual_mov_b32 v104, v197
	v_dual_mov_b32 v105, v197 :: v_dual_mov_b32 v82, v197
	v_dual_mov_b32 v83, v197 :: v_dual_mov_b32 v84, v197
	v_dual_mov_b32 v85, v197 :: v_dual_mov_b32 v86, v197
	v_dual_mov_b32 v87, v197 :: v_dual_mov_b32 v88, v197
	v_dual_mov_b32 v89, v197 :: v_dual_mov_b32 v90, v197
	v_dual_mov_b32 v91, v197 :: v_dual_mov_b32 v92, v197
	v_dual_mov_b32 v93, v197 :: v_dual_mov_b32 v94, v197
	v_dual_mov_b32 v95, v197 :: v_dual_mov_b32 v96, v197
	v_dual_mov_b32 v97, v197 :: v_dual_mov_b32 v106, v197
	v_dual_mov_b32 v107, v197 :: v_dual_mov_b32 v108, v197
	v_dual_mov_b32 v109, v197 :: v_dual_mov_b32 v110, v197
	v_dual_mov_b32 v111, v197 :: v_dual_mov_b32 v112, v197
	v_dual_mov_b32 v113, v197 :: v_dual_mov_b32 v122, v197
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
	v_dual_mov_b32 v193, v197 :: v_dual_mov_b32 v114, v197
	v_dual_mov_b32 v115, v197 :: v_dual_mov_b32 v116, v197
	v_dual_mov_b32 v117, v197 :: v_dual_mov_b32 v118, v197
	v_dual_mov_b32 v119, v197 :: v_dual_mov_b32 v120, v197
	v_mov_b32_e32 v121, v197
	s_lshr_b32 s49, s37, 16
	s_lshr_b32 s50, s35, 16
	s_lshl_b64 s[18:19], s[18:19], 1
	s_lshl_b64 s[6:7], s[6:7], 1
	s_mov_b32 s44, s39
	s_mov_b32 s46, s5
	s_or_b32 s12, s17, 0x7510000
	s_bitset1_b32 s49, 22
	s_bitset1_b32 s50, 22
	s_add_nc_u64 s[20:21], s[26:27], s[18:19]
	s_add_nc_u64 s[24:25], s[14:15], s[6:7]
	s_movk_i32 s8, 0x100
	s_mov_b32 s26, s11
	s_branch .LBB0_19
.LBB0_18:                               ;   in Loop: Header=BB0_19 Depth=1
	s_or_b32 exec_lo, exec_lo, s5
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
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshl_b32 s5, s26, 6
	s_sub_co_i32 s6, s30, s5
	s_xor_b32 s5, s27, 1
	s_min_i32 s6, s6, 64
	s_cmp_lt_i32 s26, s47
	s_cselect_b32 s10, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_and_b32 s7, s10, exec_lo
	s_cselect_b32 s34, s6, 0
	s_cmp_lt_i32 s34, 64
	s_cselect_b32 s6, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b32 s6, s48, s6
	s_or_b32 s6, s43, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s6
	s_cbranch_vccnz .LBB0_31
; %bb.20:                               ;   in Loop: Header=BB0_19 Depth=1
	v_mov_b64_e32 v[240:241], v[0:1]
	v_mov_b32_e32 v211, 34
	s_cmp_lg_u32 s5, 0
	s_mov_b32 s13, 0
	s_cselect_b32 s7, s46, s39
	s_cselect_b32 s6, s45, 0
.LBB0_21:                               ;   Parent Loop BB0_19 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_dual_mov_b32 v196, v240 :: v_dual_add_nc_u32 v211, -2, v211
	v_dual_mov_b32 v242, v241 :: v_dual_mov_b32 v243, v197
	v_add_nc_u32_e32 v241, 0x200, v241
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u64 v[244:245], v[196:197], 2, s[6:7]
	v_cmp_eq_u32_e32 vcc_lo, 0, v211
	v_add_nc_u32_e32 v240, 0x200, v240
	v_lshl_add_u64 v[242:243], v[242:243], 2, s[6:7]
	s_clause 0x1
	flat_store_b32 v[244:245], v197
	flat_store_b32 v[242:243], v197
	s_or_b32 s13, vcc_lo, s13
	s_wait_xcnt 0x0
	s_and_not1_b32 exec_lo, exec_lo, s13
	s_cbranch_execnz .LBB0_21
; %bb.22:                               ;   in Loop: Header=BB0_19 Depth=1
	s_or_b32 exec_lo, exec_lo, s13
	s_and_saveexec_b32 s13, s11
	s_cbranch_execz .LBB0_25
; %bb.23:                               ;   in Loop: Header=BB0_19 Depth=1
	v_add_nc_u64_e32 v[240:241], s[6:7], v[236:237]
	v_mov_b32_e32 v196, v207
	s_mov_b32 s6, 0
.LBB0_24:                               ;   Parent Loop BB0_19 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v196, 0x100, v196
	flat_store_b32 v[240:241], v197
	s_wait_xcnt 0x0
	v_add_nc_u64_e32 v[240:241], 0x400, v[240:241]
	v_cmp_lt_u32_e32 vcc_lo, 0x20ff, v196
	s_or_b32 s6, vcc_lo, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s6
	s_cbranch_execnz .LBB0_24
.LBB0_25:                               ;   in Loop: Header=BB0_19 Depth=1
	s_or_b32 exec_lo, exec_lo, s13
	v_mov_b64_e32 v[240:241], v[0:1]
	v_mov_b32_e32 v211, v205
	s_cmp_lg_u32 s5, 0
	s_mov_b32 s13, 0
	s_cselect_b32 s7, s41, s44
	s_cselect_b32 s6, s40, s38
.LBB0_26:                               ;   Parent Loop BB0_19 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_dual_mov_b32 v196, v240 :: v_dual_add_nc_u32 v211, -2, v211
	v_dual_mov_b32 v242, v241 :: v_dual_mov_b32 v243, v197
	v_add_nc_u32_e32 v241, 0x200, v241
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u64 v[244:245], v[196:197], 2, s[6:7]
	v_cmp_eq_u32_e32 vcc_lo, 0, v211
	v_add_nc_u32_e32 v240, 0x200, v240
	v_lshl_add_u64 v[242:243], v[242:243], 2, s[6:7]
	s_clause 0x1
	flat_store_b32 v[244:245], v197
	flat_store_b32 v[242:243], v197
	s_or_b32 s13, vcc_lo, s13
	s_wait_xcnt 0x0
	s_and_not1_b32 exec_lo, exec_lo, s13
	s_cbranch_execnz .LBB0_26
; %bb.27:                               ;   in Loop: Header=BB0_19 Depth=1
	s_or_b32 exec_lo, exec_lo, s13
	s_and_saveexec_b32 s13, s1
	s_cbranch_execz .LBB0_30
; %bb.28:                               ;   in Loop: Header=BB0_19 Depth=1
	v_add_nc_u64_e32 v[240:241], s[6:7], v[238:239]
	v_mov_b32_e32 v196, v209
	s_mov_b32 s6, 0
.LBB0_29:                               ;   Parent Loop BB0_19 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v196, 0x100, v196
	flat_store_b32 v[240:241], v197
	s_wait_xcnt 0x0
	v_add_nc_u64_e32 v[240:241], 0x400, v[240:241]
	v_cmp_lt_u32_e32 vcc_lo, 0x187f, v196
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
	v_cmpx_lt_i32_e32 0, v203
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
	s_cmp_lg_u32 s27, 0
	s_cselect_b32 s6, s45, 0
	s_cselect_b32 s5, s40, s38
	v_lshl_add_u32 v196, v194, 1, s6
	v_lshl_add_u32 v211, v200, 1, s6
	v_lshl_add_u32 v213, v202, 1, s6
	v_lshl_add_u32 v215, v210, 1, s5
	v_lshl_add_u32 v217, v224, 1, s5
	ds_load_b128 v[240:243], v196
	ds_load_b128 v[244:247], v196 offset:16
	v_lshl_add_u32 v196, v212, 1, s5
	ds_load_b128 v[248:251], v211 offset:2048
	ds_load_b128 v[252:255], v211 offset:2064
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[0:3] /*v[256:259]*/, v213 offset:4096
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v211, v218, 1, s6
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[16:19] /*v[272:275]*/, v196 offset:8192
	ds_load_b128 v[20:23] /*v[276:279]*/, v196 offset:8208
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v196, v214, 1, s5
	v_lshl_add_u32 v219, v226, 1, s5
	; sched_group_barrier mask(0x00000100) size(5) SyncID(0)
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[24:27] /*v[280:283]*/, v196 offset:10240
	s_set_vgpr_msb 0x4004                   ;  msbs: dst=0 src0=0 src1=1 src2=0
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x32_bf16 v[154:161], v[240:247], v[16:23] /*v[272:279]*/, v[154:161] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[90:97], v[248:255], v[16:23] /*v[272:279]*/, v[90:97] matrix_a_reuse
	s_set_vgpr_msb 0x440                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[28:31] /*v[284:287]*/, v196 offset:10256
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v196, v204, 1, s6
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[32:35] /*v[288:291]*/, v196 offset:6144
	ds_load_b128 v[36:39] /*v[292:295]*/, v196 offset:6160
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v196, v206, 1, s5
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[40:43] /*v[296:299]*/, v196 offset:2048
	ds_load_b128 v[44:47] /*v[300:303]*/, v196 offset:2064
	s_set_vgpr_msb 0x4005                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[2:9], v[32:39] /*v[288:295]*/, v[16:23] /*v[272:279]*/, v[2:9] matrix_a_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[4:7] /*v[260:263]*/, v213 offset:4112
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v196, v208, 1, s5
	v_lshl_add_u32 v213, v220, 1, s6
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[48:51] /*v[304:307]*/, v196 offset:4096
	ds_load_b128 v[52:55] /*v[308:311]*/, v196 offset:4112
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v196, v198, 1, s5
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[42:49], v[0:7] /*v[256:263]*/, v[16:23] /*v[272:279]*/, v[42:49] matrix_a_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[16:19] /*v[272:275]*/, v196
	ds_load_b128 v[20:23] /*v[276:279]*/, v196 offset:16
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v196, v216, 1, s6
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[66:73], v[32:39] /*v[288:295]*/, v[16:23] /*v[272:279]*/, v[66:73] matrix_a_reuse
	s_set_vgpr_msb 0x504                    ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[186:193], v[240:247], v[16:23] /*v[272:279]*/, v[186:193]
	v_wmma_f32_16x16x32_bf16 v[138:145], v[248:255], v[16:23] /*v[272:279]*/, v[138:145] matrix_a_reuse
	s_set_vgpr_msb 0x405                    ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[98:105], v[0:7] /*v[256:263]*/, v[16:23] /*v[272:279]*/, v[98:105]
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[8:11] /*v[264:267]*/, v215 offset:6144
	ds_load_b128 v[12:15] /*v[268:271]*/, v215 offset:6160
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v215, v222, 1, s6
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[16:19] /*v[272:275]*/, v219 offset:64
	ds_load_b128 v[20:23] /*v[276:279]*/, v219 offset:80
	s_set_vgpr_msb 0x4005                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[50:57], v[0:7] /*v[256:263]*/, v[8:15] /*v[264:271]*/, v[50:57] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(6) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(5) SyncID(0)
	s_set_vgpr_msb 0x504                    ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[162:169], v[240:247], v[8:15] /*v[264:271]*/, v[162:169] matrix_a_reuse
	s_set_vgpr_msb 0x405                    ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[34:41], v[0:7] /*v[256:263]*/, v[24:31] /*v[280:287]*/, v[34:41] matrix_a_reuse
	s_set_vgpr_msb 0x504                    ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[178:185], v[240:247], v[40:47] /*v[296:303]*/, v[178:185] matrix_a_reuse
	s_set_vgpr_msb 0x405                    ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[26:33], v[32:39] /*v[288:295]*/, v[40:47] /*v[296:303]*/, v[26:33] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(6) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(5) SyncID(0)
	s_set_vgpr_msb 0x504                    ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[146:153], v[240:247], v[24:31] /*v[280:287]*/, v[146:153] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[82:89], v[248:255], v[24:31] /*v[280:287]*/, v[82:89]
	s_set_vgpr_msb 0x405                    ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[114:121], v[32:39] /*v[288:295]*/, v[24:31] /*v[280:287]*/, v[114:121]
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[24:27] /*v[280:283]*/, v211 offset:64
	ds_load_b128 v[28:31] /*v[284:287]*/, v211 offset:80
	s_set_vgpr_msb 0x4004                   ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_lshl_add_u32 v211, v230, 1, s5
	v_wmma_f32_16x16x32_bf16 v[130:137], v[248:255], v[40:47] /*v[296:303]*/, v[130:137] matrix_a_reuse
	s_set_vgpr_msb 0x405                    ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[10:17], v[32:39] /*v[288:295]*/, v[8:15] /*v[264:271]*/, v[10:17] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(6) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(5) SyncID(0)
	s_set_vgpr_msb 0x504                    ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[170:177], v[240:247], v[48:55] /*v[304:311]*/, v[170:177] matrix_a_reuse
	ds_load_b128 v[240:243], v196 offset:64
	ds_load_b128 v[244:247], v196 offset:80
	v_lshl_add_u32 v196, v228, 1, s5
	s_set_vgpr_msb 0x440                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[56:59] /*v[312:315]*/, v196 offset:64
	s_set_vgpr_msb 0x4005                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[18:25], v[32:39] /*v[288:295]*/, v[48:55] /*v[304:311]*/, v[18:25] matrix_a_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[32:35] /*v[288:291]*/, v213 offset:64
	ds_load_b128 v[36:39] /*v[292:295]*/, v213 offset:80
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v213, v232, 1, s5
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[60:63] /*v[316:319]*/, v196 offset:80
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v196, v234, 1, s5
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[64:67] /*v[320:323]*/, v213 offset:64
	s_set_vgpr_msb 0x4005                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[74:81], v[0:7] /*v[256:263]*/, v[40:47] /*v[296:303]*/, v[74:81] matrix_a_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[40:43] /*v[296:299]*/, v215 offset:64
	ds_load_b128 v[44:47] /*v[300:303]*/, v215 offset:80
	ds_load_b128 v[68:71] /*v[324:327]*/, v213 offset:80
	ds_load_b128 v[72:75] /*v[328:331]*/, v196 offset:64
	ds_load_b128 v[76:79] /*v[332:335]*/, v196 offset:80
	s_set_vgpr_msb 0x4004                   ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[122:129], v[248:255], v[48:55] /*v[304:311]*/, v[122:129] matrix_a_reuse
	s_set_vgpr_msb 0x405                    ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[58:65], v[0:7] /*v[256:263]*/, v[48:55] /*v[304:311]*/, v[58:65] matrix_a_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[48:51] /*v[304:307]*/, v217 offset:64
	ds_load_b128 v[52:55] /*v[308:311]*/, v217 offset:80
	ds_load_b128 v[0:3] /*v[256:259]*/, v211 offset:64
	ds_load_b128 v[4:7] /*v[260:263]*/, v211 offset:80
	; sched_group_barrier mask(0x00000008) size(6) SyncID(0)
	s_set_vgpr_msb 0x4004                   ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[106:113], v[248:255], v[8:15] /*v[264:271]*/, v[106:113] matrix_a_reuse
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[186:193], v[240:247], v[48:55] /*v[304:311]*/, v[186:193]
	; sched_group_barrier mask(0x00000100) size(5) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[178:185], v[240:247], v[16:23] /*v[272:279]*/, v[178:185] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[170:177], v[240:247], v[56:63] /*v[312:319]*/, v[170:177] matrix_a_reuse
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[162:169], v[240:247], v[0:7] /*v[256:263]*/, v[162:169] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[154:161], v[240:247], v[64:71] /*v[320:327]*/, v[154:161] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[146:153], v[240:247], v[72:79] /*v[328:335]*/, v[146:153] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(6) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(5) SyncID(0)
	s_set_vgpr_msb 0x405                    ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[82:89], v[24:31] /*v[280:287]*/, v[72:79] /*v[328:335]*/, v[82:89]
	v_wmma_f32_16x16x32_bf16 v[90:97], v[24:31] /*v[280:287]*/, v[64:71] /*v[320:327]*/, v[90:97] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[106:113], v[24:31] /*v[280:287]*/, v[0:7] /*v[256:263]*/, v[106:113] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[122:129], v[24:31] /*v[280:287]*/, v[56:63] /*v[312:319]*/, v[122:129] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[130:137], v[24:31] /*v[280:287]*/, v[16:23] /*v[272:279]*/, v[130:137] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[138:145], v[24:31] /*v[280:287]*/, v[48:55] /*v[304:311]*/, v[138:145] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(6) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(5) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[98:105], v[32:39] /*v[288:295]*/, v[48:55] /*v[304:311]*/, v[98:105]
	v_wmma_f32_16x16x32_bf16 v[74:81], v[32:39] /*v[288:295]*/, v[16:23] /*v[272:279]*/, v[74:81] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[58:65], v[32:39] /*v[288:295]*/, v[56:63] /*v[312:319]*/, v[58:65] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[50:57], v[32:39] /*v[288:295]*/, v[0:7] /*v[256:263]*/, v[50:57] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[42:49], v[32:39] /*v[288:295]*/, v[64:71] /*v[320:327]*/, v[42:49] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[34:41], v[32:39] /*v[288:295]*/, v[72:79] /*v[328:335]*/, v[34:41] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(6) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(5) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[114:121], v[40:47] /*v[296:303]*/, v[72:79] /*v[328:335]*/, v[114:121]
	v_wmma_f32_16x16x32_bf16 v[2:9], v[40:47] /*v[296:303]*/, v[64:71] /*v[320:327]*/, v[2:9] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[10:17], v[40:47] /*v[296:303]*/, v[0:7] /*v[256:263]*/, v[10:17] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[18:25], v[40:47] /*v[296:303]*/, v[56:63] /*v[312:319]*/, v[18:25] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[26:33], v[40:47] /*v[296:303]*/, v[16:23] /*v[272:279]*/, v[26:33] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[66:73], v[40:47] /*v[296:303]*/, v[48:55] /*v[304:311]*/, v[66:73] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(6) SyncID(0)
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
	v_cmpx_eq_u32_e32 1, v203
	s_cbranch_execz .LBB0_39
; %bb.38:                               ;   in Loop: Header=BB0_19 Depth=1
	s_cmp_lg_u32 s5, 0
	s_mov_b32 s36, s34
	s_cselect_b32 s13, s40, s38
	s_cmp_gt_i32 s34, 0
	s_mov_b32 s18, s11
	s_cselect_b32 s17, -1, 0
	s_lshl_b32 s10, s6, 6
	s_mov_b32 s19, s11
	s_lshl_b64 s[14:15], s[10:11], 1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_nc_u64 s[14:15], s[20:21], s[14:15]
	v_dual_mov_b32 v211, s13 :: v_dual_mov_b32 v240, s14
	s_and_b32 s10, s15, 0x1ffffff
	s_and_b32 s13, s29, s17
	s_bitset1_b32 s10, 31
	v_cndmask_b32_e64 v196, 0, 1, s13
	v_mov_b32_e32 v213, s10
	v_readfirstlane_b32 s53, v211
	v_readfirstlane_b32 s54, v240
	s_lshr_b64 s[14:15], s[36:37], 16
	v_readfirstlane_b32 s52, v196
	v_readfirstlane_b32 s55, v213
	s_lshl_b32 s13, s34, 16
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
	s_cmp_gt_i32 s34, 0
	s_cselect_b32 s14, -1, 0
	s_lshl_b32 s10, s6, 6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_lshl_b64 s[6:7], s[10:11], 1
	s_and_b32 s10, s42, s14
	s_add_nc_u64 s[6:7], s[24:25], s[6:7]
	v_cndmask_b32_e64 v196, 0, 1, s10
	s_and_b32 s7, s7, 0x1ffffff
	v_dual_mov_b32 v211, s5 :: v_dual_mov_b32 v240, s6
	s_bitset1_b32 s7, 31
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_readfirstlane_b32 s52, v196
	v_mov_b32_e32 v213, s7
	v_readfirstlane_b32 s53, v211
	v_readfirstlane_b32 s54, v240
	s_lshr_b64 s[6:7], s[34:35], 16
	s_lshl_b32 s5, s34, 16
	v_readfirstlane_b32 s55, v213
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
	v_and_or_b32 v194, v0, 15, v201
	v_cvt_pk_bf16_f32 v177, v176, v177
	v_cvt_pk_bf16_f32 v176, v174, v175
	v_cvt_pk_bf16_f32 v175, v172, v173
	v_and_or_b32 v1, v1, 8, v199
	v_cvt_pk_bf16_f32 v185, v184, v185
	v_cvt_pk_bf16_f32 v184, v182, v183
	v_cvt_pk_bf16_f32 v182, v178, v179
	v_cvt_pk_bf16_f32 v174, v170, v171
	v_lshl_or_b32 v1, v194, 8, v1
	v_cvt_pk_bf16_f32 v169, v168, v169
	v_cvt_pk_bf16_f32 v168, v166, v167
	v_cvt_pk_bf16_f32 v166, v162, v163
	v_cvt_pk_bf16_f32 v153, v152, v153
	v_add_nc_u32_e32 v172, 0x1000, v1
	v_lshrrev_b32_e32 v173, 3, v1
	v_add_nc_u32_e32 v170, 0x2000, v1
	v_add_nc_u32_e32 v171, 0x3000, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_lshlrev_b32 v178, 1, v1 :: v_dual_lshrrev_b32 v172, 3, v172
	v_and_b32_e32 v173, 0xff0, v173
	v_add_nc_u32_e32 v162, 16, v1
	v_cvt_pk_bf16_f32 v152, v150, v151
	v_cvt_pk_bf16_f32 v151, v148, v149
	v_and_b32_e32 v172, 0x1ff0, v172
	v_add_nc_u32_e32 v148, 32, v1
	v_cvt_pk_bf16_f32 v193, v192, v193
	v_cvt_pk_bf16_f32 v192, v190, v191
	v_cvt_pk_bf16_f32 v191, v188, v189
	v_cvt_pk_bf16_f32 v190, v186, v187
	v_dual_lshrrev_b32 v170, 3, v170 :: v_dual_lshrrev_b32 v171, 3, v171
	v_add_nc_u32_e32 v173, v173, v178
	v_cvt_pk_bf16_f32 v183, v180, v181
	v_add_nc_u32_e32 v172, v172, v178
	v_cvt_pk_bf16_f32 v167, v164, v165
	v_dual_lshrrev_b32 v165, 3, v162 :: v_dual_lshrrev_b32 v149, 3, v148
	v_add_nc_u32_e32 v179, 0x4000, v1
	v_and_b32_e32 v170, 0x1ff0, v170
	v_and_b32_e32 v171, 0x1ff0, v171
	ds_store_b128 v173, v[190:193]
	ds_store_b128 v172, v[182:185] offset:8192
	v_add_nc_u32_e32 v172, 0x5000, v1
	v_cvt_pk_bf16_f32 v150, v146, v147
	v_cvt_pk_bf16_f32 v145, v144, v145
	v_add_nc_u32_e32 v147, 0x1010, v1
	v_cvt_pk_bf16_f32 v144, v142, v143
	v_cvt_pk_bf16_f32 v143, v140, v141
	v_add_nc_u32_e32 v140, 48, v1
	v_cvt_pk_bf16_f32 v161, v160, v161
	v_cvt_pk_bf16_f32 v160, v158, v159
	v_cvt_pk_bf16_f32 v158, v154, v155
	v_and_b32_e32 v155, 0xff0, v165
	v_cvt_pk_bf16_f32 v142, v138, v139
	v_and_b32_e32 v139, 0xff0, v149
	v_dual_lshrrev_b32 v179, 3, v179 :: v_dual_add_nc_u32 v170, v170, v178
	v_dual_add_nc_u32 v171, v171, v178 :: v_dual_lshrrev_b32 v164, 3, v172
	v_dual_lshrrev_b32 v138, 3, v147 :: v_dual_lshrrev_b32 v141, 3, v140
	v_cvt_pk_bf16_f32 v105, v104, v105
	v_cvt_pk_bf16_f32 v104, v102, v103
	v_cvt_pk_bf16_f32 v102, v98, v99
	v_add_nc_u32_e32 v99, 0x2010, v1
	v_cvt_pk_bf16_f32 v73, v72, v73
	v_cvt_pk_bf16_f32 v72, v70, v71
	v_cvt_pk_bf16_f32 v70, v66, v67
	v_add_nc_u32_e32 v67, 0x3010, v1
	v_add_nc_u32_e32 v146, v155, v178
	v_add_nc_u32_e32 v139, v139, v178
	v_cvt_pk_bf16_f32 v103, v100, v101
	v_and_b32_e32 v173, 0x1ff0, v179
	v_and_b32_e32 v164, 0x1ff0, v164
	ds_store_b128 v170, v[174:177] offset:16384
	ds_store_b128 v146, v[142:145] offset:32
	v_and_b32_e32 v138, 0x1ff0, v138
	v_and_b32_e32 v98, 0xff0, v141
	v_lshlrev_b32_e32 v141, 1, v162
	ds_store_b128 v171, v[166:169] offset:24576
	ds_store_b128 v139, v[102:105] offset:64
	v_lshrrev_b32_e32 v66, 3, v99
	v_lshrrev_b32_e32 v99, 3, v67
	v_add_nc_u32_e32 v103, 0x4010, v1
	v_add_nc_u32_e32 v163, v173, v178
	v_cvt_pk_bf16_f32 v159, v156, v157
	v_dual_add_nc_u32 v154, v164, v178 :: v_dual_add_nc_u32 v139, v98, v178
	v_cvt_pk_bf16_f32 v71, v68, v69
	v_add_nc_u32_e32 v138, v138, v141
	v_cvt_pk_bf16_f32 v69, v136, v137
	v_and_b32_e32 v98, 0x1ff0, v66
	v_cvt_pk_bf16_f32 v68, v134, v135
	v_cvt_pk_bf16_f32 v67, v132, v133
	v_cvt_pk_bf16_f32 v66, v130, v131
	v_and_b32_e32 v102, 0x1ff0, v99
	v_cvt_pk_bf16_f32 v105, v112, v113
	v_lshrrev_b32_e32 v112, 3, v103
	v_cvt_pk_bf16_f32 v104, v110, v111
	v_add_nc_u32_e32 v110, 0x5010, v1
	ds_store_b128 v163, v[158:161] offset:32768
	ds_store_b128 v154, v[150:153] offset:40960
	v_add_nc_u32_e32 v130, v98, v141
	v_cvt_pk_bf16_f32 v101, v128, v129
	v_cvt_pk_bf16_f32 v100, v126, v127
	v_cvt_pk_bf16_f32 v99, v124, v125
	v_cvt_pk_bf16_f32 v98, v122, v123
	v_add_nc_u32_e32 v122, v102, v141
	v_cvt_pk_bf16_f32 v102, v106, v107
	v_and_b32_e32 v106, 0x1ff0, v112
	ds_store_b128 v138, v[66:69] offset:8192
	v_lshrrev_b32_e32 v66, 3, v110
	v_cvt_pk_bf16_f32 v103, v108, v109
	ds_store_b128 v130, v[98:101] offset:16384
	ds_store_b128 v122, v[102:105] offset:24576
	v_add_nc_u32_e32 v98, v106, v141
	v_cvt_pk_bf16_f32 v69, v96, v97
	v_and_b32_e32 v96, 0x1ff0, v66
	v_cvt_pk_bf16_f32 v68, v94, v95
	v_cvt_pk_bf16_f32 v67, v92, v93
	v_cvt_pk_bf16_f32 v66, v90, v91
	v_add_nc_u32_e32 v94, 0x1020, v1
	v_cvt_pk_bf16_f32 v89, v88, v89
	v_cvt_pk_bf16_f32 v88, v86, v87
	v_cvt_pk_bf16_f32 v87, v84, v85
	ds_store_b128 v98, v[66:69] offset:32768
	v_add_nc_u32_e32 v67, 0x3020, v1
	v_add_nc_u32_e32 v84, 0x2020, v1
	v_cvt_pk_bf16_f32 v68, v78, v79
	v_cvt_pk_bf16_f32 v57, v56, v57
	v_cvt_pk_bf16_f32 v56, v54, v55
	v_lshrrev_b32_e32 v78, 3, v67
	v_cvt_pk_bf16_f32 v67, v76, v77
	v_add_nc_u32_e32 v76, 0x4020, v1
	v_cvt_pk_bf16_f32 v55, v52, v53
	v_add_nc_u32_e32 v52, 0x1030, v1
	v_cvt_pk_bf16_f32 v65, v64, v65
	v_cvt_pk_bf16_f32 v64, v62, v63
	v_cvt_pk_bf16_f32 v63, v60, v61
	v_add_nc_u32_e32 v60, 0x5020, v1
	v_cvt_pk_bf16_f32 v41, v40, v41
	v_cvt_pk_bf16_f32 v40, v38, v39
	v_cvt_pk_bf16_f32 v38, v34, v35
	v_add_nc_u32_e32 v34, 0x2030, v1
	v_cvt_pk_bf16_f32 v39, v36, v37
	v_add_nc_u32_e32 v37, 0x3030, v1
	v_cvt_pk_bf16_f32 v33, v32, v33
	v_cvt_pk_bf16_f32 v32, v30, v31
	v_cvt_pk_bf16_f32 v31, v28, v29
	v_add_nc_u32_e32 v28, 0x4030, v1
	v_add_nc_u32_e32 v1, 0x5030, v1
	v_dual_lshrrev_b32 v91, 3, v94 :: v_dual_lshrrev_b32 v66, 3, v84
	v_cvt_pk_bf16_f32 v62, v58, v59
	v_lshrrev_b32_e32 v59, 3, v76
	v_cvt_pk_bf16_f32 v49, v48, v49
	v_cvt_pk_bf16_f32 v48, v46, v47
	v_cvt_pk_bf16_f32 v46, v42, v43
	v_lshrrev_b32_e32 v43, 3, v52
	v_cvt_pk_bf16_f32 v54, v50, v51
	v_dual_lshrrev_b32 v51, 3, v60 :: v_dual_lshrrev_b32 v34, 3, v34
	v_cvt_pk_bf16_f32 v30, v26, v27
	v_lshrrev_b32_e32 v26, 3, v37
	v_cvt_pk_bf16_f32 v25, v24, v25
	v_cvt_pk_bf16_f32 v24, v22, v23
	v_cvt_pk_bf16_f32 v23, v20, v21
	v_dual_lshrrev_b32 v20, 3, v28 :: v_dual_lshrrev_b32 v1, 3, v1
	v_cvt_pk_bf16_f32 v86, v82, v83
	v_and_b32_e32 v82, 0x1ff0, v91
	v_lshlrev_b32_e32 v83, 1, v148
	v_cvt_pk_bf16_f32 v69, v80, v81
	v_and_b32_e32 v80, 0x1ff0, v66
	v_and_b32_e32 v50, 0x1ff0, v59
	v_and_b32_e32 v35, 0x1ff0, v43
	v_lshlrev_b32_e32 v36, 1, v140
	v_cvt_pk_bf16_f32 v66, v74, v75
	v_and_b32_e32 v75, 0x1ff0, v78
	v_and_b32_e32 v51, 0x1ff0, v51
	v_and_b32_e32 v34, 0x1ff0, v34
	v_and_b32_e32 v26, 0x1ff0, v26
	v_cvt_pk_bf16_f32 v22, v18, v19
	v_and_b32_e32 v19, 0x1ff0, v20
	v_and_b32_e32 v1, 0x1ff0, v1
	v_dual_add_nc_u32 v90, v96, v141 :: v_dual_add_nc_u32 v82, v82, v83
	v_dual_add_nc_u32 v74, v80, v83 :: v_dual_add_nc_u32 v50, v50, v83
	v_cvt_pk_bf16_f32 v47, v44, v45
	v_add_nc_u32_e32 v35, v35, v36
	v_add_nc_u32_e32 v58, v75, v83
	v_dual_add_nc_u32 v42, v51, v83 :: v_dual_add_nc_u32 v27, v34, v36
	v_add_nc_u32_e32 v18, v26, v36
	v_cvt_pk_bf16_f32 v17, v16, v17
	v_cvt_pk_bf16_f32 v16, v14, v15
	v_cvt_pk_bf16_f32 v15, v12, v13
	v_cvt_pk_bf16_f32 v14, v10, v11
	v_add_nc_u32_e32 v10, v19, v36
	v_cvt_pk_bf16_f32 v9, v8, v9
	v_cvt_pk_bf16_f32 v8, v6, v7
	v_cvt_pk_bf16_f32 v7, v4, v5
	v_cvt_pk_bf16_f32 v6, v2, v3
	v_add_nc_u32_e32 v1, v1, v36
	v_cvt_pk_bf16_f32 v5, v120, v121
	v_cvt_pk_bf16_f32 v4, v118, v119
	v_cvt_pk_bf16_f32 v3, v116, v117
	v_cvt_pk_bf16_f32 v2, v114, v115
	ds_store_b128 v90, v[86:89] offset:40960
	ds_store_b128 v82, v[66:69] offset:8192
	ds_store_b128 v74, v[62:65] offset:16384
	ds_store_b128 v58, v[54:57] offset:24576
	ds_store_b128 v50, v[46:49] offset:32768
	ds_store_b128 v42, v[38:41] offset:40960
	ds_store_b128 v139, v[70:73] offset:96
	ds_store_b128 v35, v[30:33] offset:8192
	ds_store_b128 v27, v[22:25] offset:16384
	ds_store_b128 v18, v[14:17] offset:24576
	ds_store_b128 v10, v[6:9] offset:32768
	ds_store_b128 v1, v[2:5] offset:40960
.LBB0_43:
	v_cmp_ne_u32_e32 vcc_lo, 1, v195
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
	s_mul_i32 s15, s33, 0xc0
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
	.size	bm256_bn192_bk064_wm4_wn2_mc1, .Lfunc_end0-bm256_bn192_bk064_wm4_wn2_mc1
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel bm256_bn192_bk064_wm4_wn2_mc1
		.amdhsa_group_segment_fixed_size 121856
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
		.amdhsa_next_free_vgpr 336
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
		.amdhsa_inst_pref_size 70
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text.bm256_bn192_bk064_wm4_wn2_mc1,"axG",@progbits,bm256_bn192_bk064_wm4_wn2_mc1,comdat
                                        ; -- End function
	.set .Lbm256_bn192_bk064_wm4_wn2_mc1.num_vgpr, 336
	.set .Lbm256_bn192_bk064_wm4_wn2_mc1.num_agpr, 0
	.set .Lbm256_bn192_bk064_wm4_wn2_mc1.numbered_sgpr, 56
	.set .Lbm256_bn192_bk064_wm4_wn2_mc1.num_named_barrier, 0
	.set .Lbm256_bn192_bk064_wm4_wn2_mc1.private_seg_size, 0
	.set .Lbm256_bn192_bk064_wm4_wn2_mc1.uses_vcc, 1
	.set .Lbm256_bn192_bk064_wm4_wn2_mc1.uses_flat_scratch, 1
	.set .Lbm256_bn192_bk064_wm4_wn2_mc1.has_dyn_sized_stack, 0
	.set .Lbm256_bn192_bk064_wm4_wn2_mc1.has_recursion, 0
	.set .Lbm256_bn192_bk064_wm4_wn2_mc1.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 8948
; TotalNumSgprs: 58
; NumVgprs: 336
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 121856 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 20
; NumSGPRsForWavesPerEU: 58
; NumVGPRsForWavesPerEU: 336
; NamedBarCnt: 0
; Occupancy: 3
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
	.type	__hip_cuid_74ca04bd41a709ed,@object ; @__hip_cuid_74ca04bd41a709ed
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_74ca04bd41a709ed
__hip_cuid_74ca04bd41a709ed:
	.byte	0                               ; 0x0
	.size	__hip_cuid_74ca04bd41a709ed, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git aa451e1fe6a793394d6733051b1778633063ae96+PATCHED:d17c5aa0e3ea29cde402f58f27e39b6034effa27)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_74ca04bd41a709ed
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
    macrotile: [256, 192, 64]
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
    .cluster_dims:
      - 4
      - 4
      - 1
    .gfx1250_revision: B0
    .group_segment_fixed_size: 121856
    .kernarg_segment_align: 8
    .kernarg_segment_size: 132
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           bm256_bn192_bk064_wm4_wn2_mc1
    .private_segment_fixed_size: 0
    .sgpr_count:     58
    .sgpr_spill_count: 0
    .symbol:         bm256_bn192_bk064_wm4_wn2_mc1.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     336
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
