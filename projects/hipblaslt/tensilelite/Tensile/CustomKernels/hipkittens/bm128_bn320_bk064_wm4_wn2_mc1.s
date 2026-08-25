	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.section	.text.bm128_bn320_bk064_wm4_wn2_mc1,"axG",@progbits,bm128_bn320_bk064_wm4_wn2_mc1,comdat
	.protected	bm128_bn320_bk064_wm4_wn2_mc1 ; -- Begin function bm128_bn320_bk064_wm4_wn2_mc1
	.globl	bm128_bn320_bk064_wm4_wn2_mc1
	.p2align	8
	.type	bm128_bn320_bk064_wm4_wn2_mc1,@function
bm128_bn320_bk064_wm4_wn2_mc1: ; @bm128_bn320_bk064_wm4_wn2_mc1
; %bb.0:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1 ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_load_b96 s[28:30], s[0:1], 0x78 nv
	s_mov_b64 s[2:3], src_shared_base
	s_movk_i32 s2, 0x4400
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
	s_lshl_b32 s2, s3, 7
	s_wait_kmcnt 0x0
	s_add_co_i32 s5, s28, 0x7f
	s_sub_co_i32 s7, s28, s2
	s_ashr_i32 s8, s5, 31
	s_add_co_i32 s6, s29, 0x13f
	s_min_i32 s31, s7, 0x80
	s_lshr_b32 s7, s8, 25
	s_mul_hi_i32 s6, s6, 0x66666667
	s_add_co_i32 s5, s5, s7
	s_lshr_b32 s8, s6, 31
	s_ashr_i32 s6, s6, 7
	s_ashr_i32 s5, s5, 7
	s_add_co_i32 s6, s6, s8
	s_cmp_lt_i32 s3, s5
	s_mul_i32 s7, s33, 0xfffffec0
	s_cselect_b32 s42, -1, 0
	s_mov_b32 s9, s30
	s_and_b32 s8, s42, exec_lo
	s_cselect_b32 s35, s31, 0
	s_add_co_i32 s7, s29, s7
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_min_i32 s7, s7, 0x140
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
	s_cmp_lt_i32 s35, 0x80
	s_cselect_b32 s43, -1, 0
	s_and_b32 vcc_lo, exec_lo, s43
	s_mov_b32 s7, s43
	s_cbranch_vccnz .LBB0_2
; %bb.1:
	s_cmp_lt_i32 s37, 0x140
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
	v_cmp_lt_u32_e32 vcc_lo, 0xfff, v5
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
	ds_store_b32 v2, v3 offset:17408
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
	s_load_b64 s[14:15], s[0:1], 0x0 nv
	s_load_b128 s[24:27], s[0:1], 0x20 nv
	s_load_b128 s[20:23], s[0:1], 0x48 nv
	s_wait_xcnt 0x0
	s_lshl_b32 s0, s33, 2
	v_lshrrev_b32_e32 v171, 5, v0
	s_lshl_b32 s1, s4, 2
	s_add_co_i32 s6, s6, -1
	s_and_b32 s0, s0, 12
	s_mov_b64 s[38:39], src_shared_base
	s_or_b32 s38, s1, 0x4400
	s_add_co_i32 s16, s5, -1
	s_min_i32 s17, s33, s6
	s_and_b32 s19, s3, 3
	s_lshl_b32 s0, 15, s0
	s_mov_b32 s4, exec_lo
	v_cmpx_lt_i32_e32 0, v171
	s_xor_b32 s18, exec_lo, s4
	s_cbranch_execz .LBB0_12
; %bb.9:
	s_mov_b32 s40, exec_lo
	v_cmpx_eq_u32_e32 1, v171
	s_cbranch_execz .LBB0_11
; %bb.10:
	s_cmp_gt_i32 s34, 0
	s_mul_i32 s4, s17, 0x140
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
	s_movk_i32 s8, 0x140
	s_or_b32 s4, s0, 0x7510000
	s_lshl_b32 s5, s34, 16
	s_or_b32 s7, s11, 0x400000
	s_mov_b32 s11, s10
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[44:47], s[4:11]
.LBB0_11:
	s_or_b32 exec_lo, exec_lo, s40
.LBB0_12:
	s_or_saveexec_b32 s40, s18
	s_min_i32 s18, s3, s16
	s_lshl_b32 s36, 0x1111, s19
	s_xor_b32 exec_lo, exec_lo, s40
	s_cbranch_execz .LBB0_14
; %bb.13:
	s_cmp_gt_i32 s34, 0
	s_mov_b32 s10, 0
	s_cselect_b32 s3, -1, 0
	s_lshl_b32 s4, s18, 7
	s_wait_kmcnt 0x0
	s_bfe_i64 s[6:7], s[24:25], 0x200000
	s_ashr_i32 s5, s4, 31
	s_and_b32 s3, s42, s3
	s_mul_u64 s[4:5], s[6:7], s[4:5]
	v_cndmask_b32_e64 v2, 0, 1, s3
	s_lshl_b64 s[6:7], s[4:5], 1
	s_lshr_b32 s3, s35, 16
	s_add_nc_u64 s[6:7], s[14:15], s[6:7]
	s_or_b32 s4, s36, 0x7510000
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
	s_movk_i32 s8, 0x80
	s_mov_b32 s11, s10
	s_mov_b32 s45, s10
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[44:47], s[4:11]
.LBB0_14:
	s_or_b32 exec_lo, exec_lo, s40
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
	v_dual_lshlrev_b32 v1, 4, v171 :: v_dual_bitop2_b32 v2, 32, v0 bitop3:0x40
	v_mov_b32_e32 v9, 0
	s_and_b32 s3, s42, s29
	s_mov_b32 s11, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_and_b32_e32 v167, 0x60, v1
	v_cmp_ne_u32_e32 vcc_lo, 0, v2
	v_cndmask_b32_e64 v163, 0, 1, s3
	v_dual_mov_b32 v8, v9 :: v_dual_mov_b32 v7, v9
	v_dual_mov_b32 v6, v9 :: v_dual_mov_b32 v5, v9
	v_cndmask_b32_e64 v169, 0, 0xa0, vcc_lo
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
	v_dual_mov_b32 v58, v9 :: v_dual_mov_b32 v97, v9
	v_dual_mov_b32 v96, v9 :: v_dual_mov_b32 v95, v9
	v_dual_mov_b32 v94, v9 :: v_dual_mov_b32 v93, v9
	v_dual_mov_b32 v92, v9 :: v_dual_mov_b32 v91, v9
	v_dual_mov_b32 v90, v9 :: v_dual_mov_b32 v73, v9
	v_dual_mov_b32 v72, v9 :: v_dual_mov_b32 v71, v9
	v_dual_mov_b32 v70, v9 :: v_dual_mov_b32 v69, v9
	v_dual_mov_b32 v68, v9 :: v_dual_mov_b32 v67, v9
	v_dual_mov_b32 v66, v9 :: v_dual_mov_b32 v89, v9
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
	v_dual_mov_b32 v154, v9 :: v_dual_mov_b32 v81, v9
	v_dual_mov_b32 v80, v9 :: v_dual_mov_b32 v79, v9
	v_dual_mov_b32 v78, v9 :: v_dual_mov_b32 v77, v9
	v_dual_mov_b32 v76, v9 :: v_dual_mov_b32 v75, v9
	v_mov_b32_e32 v74, v9
	s_and_not1_b32 vcc_lo, exec_lo, s13
	s_barrier_wait -1
	s_barrier_wait -3
	s_cbranch_vccnz .LBB0_41
; %bb.17:
	v_dual_lshlrev_b32 v4, 6, v0 :: v_dual_bitop2_b32 v1, 16, v0 bitop3:0x40
	v_dual_lshlrev_b32 v2, 6, v169 :: v_dual_lshlrev_b32 v3, 6, v167
	s_mov_b64 s[4:5], src_shared_base
	s_or_b32 s6, s1, 0xee00
	s_delay_alu instid0(VALU_DEP_2)
	v_and_or_b32 v4, 0x3c0, v4, v1
	s_mov_b32 s7, s5
	v_or_b32_e32 v175, 0xf00, v0
	s_and_b64 s[6:7], s[6:7], 15
	v_lshl_or_b32 v212, v0, 2, 0x4000
	v_or_b32_e32 v6, v2, v4
	v_or_b32_e32 v5, v3, v4
	s_sub_co_i32 s4, 16, s6
	s_mov_b32 s44, s39
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mov_b32 v165, 0 :: v_dual_lshrrev_b32 v8, 4, v6
	v_lshrrev_b32_e32 v1, 4, v5
	v_or_b32_e32 v7, 0x400, v5
	v_sub_nc_u32_e32 v28, 0x2a7f, v0
	s_lshr_b32 s4, s4, 2
	s_cmp_lg_u64 s[6:7], 0
	v_and_b32_e32 v1, 0x1b8, v1
	s_cselect_b32 s4, s4, 0
	s_mul_i32 s6, s17, 0x140
	v_mov_b32_e32 v213, v165
	s_lshl2_add_u32 s1, s4, s1
	v_dual_add_nc_u32 v162, v1, v5 :: v_dual_lshrrev_b32 v1, 4, v7
	v_and_b32_e32 v7, 0x2b8, v8
	v_or_b32_e32 v8, 0x400, v6
	v_add_nc_u32_e32 v9, 0x800, v6
	v_add_nc_u32_e32 v10, 0xc00, v6
	v_add_nc_u32_e32 v13, 0x1c00, v6
	v_add_nc_u32_e32 v166, v7, v6
	v_and_b32_e32 v11, 0x1f8, v1
	v_dual_lshrrev_b32 v1, 4, v8 :: v_dual_lshrrev_b32 v7, 4, v9
	v_or_b32_e32 v9, 0x1000, v6
	v_lshrrev_b32_e32 v8, 4, v10
	v_or_b32_e32 v10, 0x1400, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_and_b32_e32 v12, 0x2f8, v1
	v_dual_lshrrev_b32 v1, 4, v9 :: v_dual_bitop2_b32 v14, 32, v4 bitop3:0x54
	v_add_nc_u32_e32 v16, 0x2000, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_lshrrev_b32_e32 v9, 4, v10
	v_dual_mov_b32 v215, v165 :: v_dual_add_nc_u32 v10, 0x1800, v6
	v_and_b32_e32 v15, 0x3b8, v1
	v_add_nc_u32_e32 v17, 0x2400, v6
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_lshrrev_b32 v16, 4, v16 :: v_dual_lshrrev_b32 v1, 4, v10
	v_dual_lshrrev_b32 v10, 4, v13 :: v_dual_bitop2_b32 v13, v14, v3 bitop3:0x54
	v_dual_lshrrev_b32 v17, 4, v17 :: v_dual_bitop2_b32 v14, v14, v2 bitop3:0x54
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_and_b32_e32 v18, 0x7b8, v1
	v_and_b32_e32 v10, 0x7f8, v10
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_or_b32_e32 v1, 0x400, v13
	v_dual_lshrrev_b32 v13, 4, v13 :: v_dual_lshrrev_b32 v19, 4, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_add_nc_u32 v3, v4, v3 :: v_dual_add_nc_u32 v182, v10, v6
	v_add_nc_u32_e32 v22, 0xc00, v14
	v_and_b32_e32 v13, 0x1b8, v13
	v_dual_add_nc_u32 v168, v11, v5 :: v_dual_add_nc_u32 v2, v4, v2
	v_dual_mov_b32 v5, v165 :: v_dual_lshrrev_b32 v1, 4, v1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_lshrrev_b32 v22, 4, v22 :: v_dual_add_nc_u32 v188, v13, v3
	v_dual_mov_b32 v13, v165 :: v_dual_lshrrev_b32 v28, 8, v28
	v_and_b32_e32 v20, 0x1f8, v1
	v_or_b32_e32 v1, 0x400, v14
	v_and_b32_e32 v8, 0x3f8, v8
	v_add_nc_u32_e32 v21, 0x800, v14
	v_or_b32_e32 v23, 0x1000, v14
	v_or_b32_e32 v24, 0x1400, v14
	v_lshrrev_b32_e32 v1, 4, v1
	s_add_co_i32 s4, s1, 0x13200
	v_dual_add_nc_u32 v174, v8, v6 :: v_dual_add_nc_u32 v176, v15, v6
	s_and_b32 s10, s4, 15
	v_lshrrev_b32_e32 v23, 4, v23
	v_and_b32_e32 v25, 0x3f8, v1
	v_dual_mov_b32 v8, v165 :: v_dual_lshrrev_b32 v1, 4, v24
	v_lshrrev_b32_e32 v21, 4, v21
	s_sub_co_i32 s7, 16, s10
	s_add_co_i32 s45, s1, 0xee00
	s_lshr_b32 s1, s7, 2
	s_cmp_lg_u64 s[10:11], 0
	v_and_b32_e32 v7, 0x3b8, v7
	v_and_b32_e32 v9, 0x3f8, v9
	v_add_nc_u32_e32 v24, 0x1800, v14
	v_add_nc_u32_e32 v26, 0x1c00, v14
	v_and_b32_e32 v27, 0x3f8, v1
	v_dual_mov_b32 v10, v165 :: v_dual_add_nc_u32 v1, 0x2000, v14
	v_dual_mov_b32 v11, v165 :: v_dual_add_nc_u32 v14, 0x2400, v14
	v_dual_add_nc_u32 v28, 1, v28 :: v_dual_add_nc_u32 v170, v12, v6
	s_cselect_b32 s1, s1, 0
	s_ashr_i32 s7, s12, 31
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_lshrrev_b32 v1, 4, v1 :: v_dual_lshrrev_b32 v14, 4, v14
	s_lshr_b32 s7, s7, 26
	v_dual_add_nc_u32 v172, v7, v6 :: v_dual_bitop2_b32 v173, 46, v28 bitop3:0x40
	s_add_co_i32 s12, s12, s7
	v_dual_add_nc_u32 v178, v9, v6 :: v_dual_add_nc_u32 v180, v18, v6
	v_dual_mov_b32 v7, v165 :: v_dual_mov_b32 v9, v165
	v_dual_lshrrev_b32 v24, 4, v24 :: v_dual_lshrrev_b32 v26, 4, v26
	s_lshl_b32 s10, s1, 2
	s_ashr_i32 s47, s12, 6
	s_cmp_lt_i32 s37, 0x140
	v_and_b32_e32 v16, 0x6b8, v16
	s_cselect_b32 s48, -1, 0
	s_ashr_i32 s7, s6, 31
	s_lshl_b32 s18, s18, 7
	v_and_b32_e32 v17, 0x6f8, v17
	v_and_b32_e32 v19, 0x2b8, v19
	v_and_b32_e32 v21, 0x7f8, v21
	v_and_b32_e32 v22, 0x7f8, v22
	v_and_b32_e32 v23, 0x3f8, v23
	v_and_b32_e32 v24, 0x7f8, v24
	v_and_b32_e32 v26, 0x7f8, v26
	v_and_b32_e32 v29, 0x7f8, v1
	v_and_b32_e32 v14, 0x7f8, v14
	s_wait_kmcnt 0x0
	s_bfe_i64 s[20:21], s[20:21], 0x200000
	v_lshl_or_b32 v30, v173, 8, v0
	s_ashr_i32 s19, s18, 31
	s_mul_u64 s[6:7], s[20:21], s[6:7]
	s_bfe_i64 s[20:21], s[24:25], 0x200000
	s_or_b32 s12, s0, 0x7510000
	s_mul_u64 s[18:19], s[20:21], s[18:19]
	v_cmp_eq_u32_e64 s0, 0, v171
	v_or_b32_e32 v1, 0x100, v0
	v_cmp_ne_u32_e64 s1, v28, v173
	v_dual_add_nc_u32 v184, v16, v6 :: v_dual_add_nc_u32 v186, v17, v6
	v_add3_u32 v190, v3, v20, 0x400
	v_dual_add_nc_u32 v192, v19, v2 :: v_dual_mov_b32 v12, v165
	v_add3_u32 v194, v2, v25, 0x400
	v_add3_u32 v196, v2, v21, 0x800
	v_add3_u32 v198, v2, v22, 0xc00
	v_add3_u32 v200, v2, v23, 0x1000
	v_add3_u32 v202, v2, v27, 0x1400
	v_add3_u32 v204, v2, v24, 0x1800
	v_add3_u32 v206, v2, v26, 0x1c00
	v_add3_u32 v208, v2, v29, 0x2000
	v_add3_u32 v210, v2, v14, 0x2400
	v_dual_mov_b32 v14, v165 :: v_dual_add_nc_u32 v177, 0xffffff00, v30
	v_dual_mov_b32 v15, v165 :: v_dual_lshlrev_b32 v214, 2, v30
	v_dual_mov_b32 v2, v165 :: v_dual_mov_b32 v3, v165
	v_dual_mov_b32 v4, v165 :: v_dual_mov_b32 v6, v165
	v_dual_mov_b32 v16, v165 :: v_dual_mov_b32 v17, v165
	v_dual_mov_b32 v18, v165 :: v_dual_mov_b32 v19, v165
	v_dual_mov_b32 v20, v165 :: v_dual_mov_b32 v21, v165
	v_dual_mov_b32 v22, v165 :: v_dual_mov_b32 v23, v165
	v_dual_mov_b32 v24, v165 :: v_dual_mov_b32 v25, v165
	v_dual_mov_b32 v26, v165 :: v_dual_mov_b32 v27, v165
	v_dual_mov_b32 v28, v165 :: v_dual_mov_b32 v29, v165
	v_dual_mov_b32 v30, v165 :: v_dual_mov_b32 v31, v165
	v_dual_mov_b32 v32, v165 :: v_dual_mov_b32 v33, v165
	v_dual_mov_b32 v34, v165 :: v_dual_mov_b32 v35, v165
	v_dual_mov_b32 v36, v165 :: v_dual_mov_b32 v37, v165
	v_dual_mov_b32 v38, v165 :: v_dual_mov_b32 v39, v165
	v_dual_mov_b32 v40, v165 :: v_dual_mov_b32 v41, v165
	v_dual_mov_b32 v42, v165 :: v_dual_mov_b32 v43, v165
	v_dual_mov_b32 v44, v165 :: v_dual_mov_b32 v45, v165
	v_dual_mov_b32 v46, v165 :: v_dual_mov_b32 v47, v165
	v_dual_mov_b32 v48, v165 :: v_dual_mov_b32 v49, v165
	v_dual_mov_b32 v50, v165 :: v_dual_mov_b32 v51, v165
	v_dual_mov_b32 v52, v165 :: v_dual_mov_b32 v53, v165
	v_dual_mov_b32 v54, v165 :: v_dual_mov_b32 v55, v165
	v_dual_mov_b32 v56, v165 :: v_dual_mov_b32 v57, v165
	v_dual_mov_b32 v58, v165 :: v_dual_mov_b32 v59, v165
	v_dual_mov_b32 v60, v165 :: v_dual_mov_b32 v61, v165
	v_dual_mov_b32 v62, v165 :: v_dual_mov_b32 v63, v165
	v_dual_mov_b32 v64, v165 :: v_dual_mov_b32 v65, v165
	v_dual_mov_b32 v90, v165 :: v_dual_mov_b32 v91, v165
	v_dual_mov_b32 v92, v165 :: v_dual_mov_b32 v93, v165
	v_dual_mov_b32 v94, v165 :: v_dual_mov_b32 v95, v165
	v_dual_mov_b32 v96, v165 :: v_dual_mov_b32 v97, v165
	v_dual_mov_b32 v66, v165 :: v_dual_mov_b32 v67, v165
	v_dual_mov_b32 v68, v165 :: v_dual_mov_b32 v69, v165
	v_dual_mov_b32 v70, v165 :: v_dual_mov_b32 v71, v165
	v_dual_mov_b32 v72, v165 :: v_dual_mov_b32 v73, v165
	v_dual_mov_b32 v82, v165 :: v_dual_mov_b32 v83, v165
	v_dual_mov_b32 v84, v165 :: v_dual_mov_b32 v85, v165
	v_dual_mov_b32 v86, v165 :: v_dual_mov_b32 v87, v165
	v_dual_mov_b32 v88, v165 :: v_dual_mov_b32 v89, v165
	v_dual_mov_b32 v98, v165 :: v_dual_mov_b32 v99, v165
	v_dual_mov_b32 v100, v165 :: v_dual_mov_b32 v101, v165
	v_dual_mov_b32 v102, v165 :: v_dual_mov_b32 v103, v165
	v_dual_mov_b32 v104, v165 :: v_dual_mov_b32 v105, v165
	v_dual_mov_b32 v106, v165 :: v_dual_mov_b32 v107, v165
	v_dual_mov_b32 v108, v165 :: v_dual_mov_b32 v109, v165
	v_dual_mov_b32 v110, v165 :: v_dual_mov_b32 v111, v165
	v_dual_mov_b32 v112, v165 :: v_dual_mov_b32 v113, v165
	v_dual_mov_b32 v114, v165 :: v_dual_mov_b32 v115, v165
	v_dual_mov_b32 v116, v165 :: v_dual_mov_b32 v117, v165
	v_dual_mov_b32 v118, v165 :: v_dual_mov_b32 v119, v165
	v_dual_mov_b32 v120, v165 :: v_dual_mov_b32 v121, v165
	v_dual_mov_b32 v122, v165 :: v_dual_mov_b32 v123, v165
	v_dual_mov_b32 v124, v165 :: v_dual_mov_b32 v125, v165
	v_dual_mov_b32 v126, v165 :: v_dual_mov_b32 v127, v165
	v_dual_mov_b32 v128, v165 :: v_dual_mov_b32 v129, v165
	v_dual_mov_b32 v130, v165 :: v_dual_mov_b32 v131, v165
	v_dual_mov_b32 v132, v165 :: v_dual_mov_b32 v133, v165
	v_dual_mov_b32 v134, v165 :: v_dual_mov_b32 v135, v165
	v_dual_mov_b32 v136, v165 :: v_dual_mov_b32 v137, v165
	v_dual_mov_b32 v138, v165 :: v_dual_mov_b32 v139, v165
	v_dual_mov_b32 v140, v165 :: v_dual_mov_b32 v141, v165
	v_dual_mov_b32 v142, v165 :: v_dual_mov_b32 v143, v165
	v_dual_mov_b32 v144, v165 :: v_dual_mov_b32 v145, v165
	v_dual_mov_b32 v146, v165 :: v_dual_mov_b32 v147, v165
	v_dual_mov_b32 v148, v165 :: v_dual_mov_b32 v149, v165
	v_dual_mov_b32 v150, v165 :: v_dual_mov_b32 v151, v165
	v_dual_mov_b32 v152, v165 :: v_dual_mov_b32 v153, v165
	v_dual_mov_b32 v154, v165 :: v_dual_mov_b32 v155, v165
	v_dual_mov_b32 v156, v165 :: v_dual_mov_b32 v157, v165
	v_dual_mov_b32 v158, v165 :: v_dual_mov_b32 v159, v165
	v_dual_mov_b32 v160, v165 :: v_dual_mov_b32 v161, v165
	v_dual_mov_b32 v74, v165 :: v_dual_mov_b32 v75, v165
	v_dual_mov_b32 v76, v165 :: v_dual_mov_b32 v77, v165
	v_dual_mov_b32 v78, v165 :: v_dual_mov_b32 v79, v165
	v_dual_mov_b32 v80, v165 :: v_dual_mov_b32 v81, v165
	s_lshr_b32 s49, s37, 16
	s_lshr_b32 s50, s35, 16
	s_lshl_b64 s[6:7], s[6:7], 1
	s_lshl_b64 s[18:19], s[18:19], 1
	s_movk_i32 s16, 0x140
	s_mov_b32 s46, s5
	s_add_nc_u64 s[40:41], s[4:5], s[10:11]
	s_bitset1_b32 s49, 22
	s_or_b32 s4, s36, 0x7510000
	s_bitset1_b32 s50, 22
	s_add_nc_u64 s[20:21], s[26:27], s[6:7]
	s_add_nc_u64 s[24:25], s[14:15], s[18:19]
	s_mov_b32 s26, -1
	s_movk_i32 s8, 0x80
	s_mov_b32 s27, s11
	s_branch .LBB0_19
.LBB0_18:                               ;   in Loop: Header=BB0_19 Depth=1
	s_or_b32 exec_lo, exec_lo, s5
	s_cmp_eq_u32 s27, s47
	s_barrier_wait -1
	s_barrier_wait -3
	s_cbranch_scc1 .LBB0_41
.LBB0_19:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_21 Depth 2
                                        ;     Child Loop BB0_24 Depth 2
                                        ;     Child Loop BB0_26 Depth 2
                                        ;     Child Loop BB0_29 Depth 2
	s_and_b32 s51, s27, 1
	s_add_co_i32 s27, s27, 1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshl_b32 s5, s27, 6
	s_sub_co_i32 s6, s30, s5
	s_xor_b32 s5, s51, 1
	s_min_i32 s6, s6, 64
	s_cmp_lt_i32 s27, s47
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
	v_mov_b64_e32 v[216:217], v[0:1]
	v_mov_b32_e32 v179, 16
	s_cmp_lg_u32 s5, 0
	s_mov_b32 s13, 0
	s_cselect_b32 s7, s46, s39
	s_cselect_b32 s6, s45, 0
.LBB0_21:                               ;   Parent Loop BB0_19 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_dual_mov_b32 v164, v216 :: v_dual_add_nc_u32 v179, -2, v179
	v_dual_mov_b32 v218, v217 :: v_dual_mov_b32 v219, v165
	v_add_nc_u32_e32 v217, 0x200, v217
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u64 v[220:221], v[164:165], 2, s[6:7]
	v_cmp_eq_u32_e32 vcc_lo, 0, v179
	v_add_nc_u32_e32 v216, 0x200, v216
	v_lshl_add_u64 v[218:219], v[218:219], 2, s[6:7]
	s_clause 0x1
	flat_store_b32 v[220:221], v165
	flat_store_b32 v[218:219], v165
	s_or_b32 s13, vcc_lo, s13
	s_wait_xcnt 0x0
	s_and_not1_b32 exec_lo, exec_lo, s13
	s_cbranch_execnz .LBB0_21
; %bb.22:                               ;   in Loop: Header=BB0_19 Depth=1
	s_or_b32 exec_lo, exec_lo, s13
	s_and_saveexec_b32 s13, s26
	s_cbranch_execz .LBB0_25
; %bb.23:                               ;   in Loop: Header=BB0_19 Depth=1
	v_add_nc_u64_e32 v[216:217], s[6:7], v[212:213]
	v_mov_b32_e32 v164, v175
	s_mov_b32 s6, 0
.LBB0_24:                               ;   Parent Loop BB0_19 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v164, 0x100, v164
	flat_store_b32 v[216:217], v165
	s_wait_xcnt 0x0
	v_add_nc_u64_e32 v[216:217], 0x400, v[216:217]
	v_cmp_lt_u32_e32 vcc_lo, 0xfff, v164
	s_or_b32 s6, vcc_lo, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s6
	s_cbranch_execnz .LBB0_24
.LBB0_25:                               ;   in Loop: Header=BB0_19 Depth=1
	s_or_b32 exec_lo, exec_lo, s13
	v_mov_b64_e32 v[216:217], v[0:1]
	v_mov_b32_e32 v179, v173
	s_cmp_lg_u32 s5, 0
	s_mov_b32 s13, 0
	s_cselect_b32 s7, s41, s44
	s_cselect_b32 s6, s40, s38
.LBB0_26:                               ;   Parent Loop BB0_19 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_dual_mov_b32 v164, v216 :: v_dual_add_nc_u32 v179, -2, v179
	v_dual_mov_b32 v218, v217 :: v_dual_mov_b32 v219, v165
	v_add_nc_u32_e32 v217, 0x200, v217
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u64 v[220:221], v[164:165], 2, s[6:7]
	v_cmp_eq_u32_e32 vcc_lo, 0, v179
	v_add_nc_u32_e32 v216, 0x200, v216
	v_lshl_add_u64 v[218:219], v[218:219], 2, s[6:7]
	s_clause 0x1
	flat_store_b32 v[220:221], v165
	flat_store_b32 v[218:219], v165
	s_or_b32 s13, vcc_lo, s13
	s_wait_xcnt 0x0
	s_and_not1_b32 exec_lo, exec_lo, s13
	s_cbranch_execnz .LBB0_26
; %bb.27:                               ;   in Loop: Header=BB0_19 Depth=1
	s_or_b32 exec_lo, exec_lo, s13
	s_and_saveexec_b32 s13, s1
	s_cbranch_execz .LBB0_30
; %bb.28:                               ;   in Loop: Header=BB0_19 Depth=1
	v_add_nc_u64_e32 v[216:217], s[6:7], v[214:215]
	v_mov_b32_e32 v164, v177
	s_mov_b32 s6, 0
.LBB0_29:                               ;   Parent Loop BB0_19 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v164, 0x100, v164
	flat_store_b32 v[216:217], v165
	s_wait_xcnt 0x0
	v_add_nc_u64_e32 v[216:217], 0x400, v[216:217]
	v_cmp_lt_u32_e32 vcc_lo, 0x297f, v164
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
	s_cselect_b32 s6, s27, 0
	s_mov_b32 s7, exec_lo
	v_cmpx_lt_i32_e32 0, v171
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
	s_cselect_b32 s5, s40, s38
	v_lshl_add_u32 v164, v162, 1, s6
	v_lshl_add_u32 v179, v166, 1, s5
	v_lshl_add_u32 v181, v192, 1, s5
	v_lshl_add_u32 v183, v194, 1, s5
	v_lshl_add_u32 v185, v196, 1, s5
	ds_load_b128 v[216:219], v164
	ds_load_b128 v[220:223], v164 offset:16
	v_lshl_add_u32 v164, v168, 1, s6
	ds_load_b128 v[224:227], v179
	ds_load_b128 v[228:231], v179 offset:16
	v_lshl_add_u32 v179, v180, 1, s5
	v_lshl_add_u32 v187, v198, 1, s5
	ds_load_b128 v[232:235], v164 offset:2048
	ds_load_b128 v[236:239], v164 offset:2064
	v_lshl_add_u32 v164, v186, 1, s5
	v_lshl_add_u32 v189, v200, 1, s5
	v_lshl_add_u32 v191, v202, 1, s5
	v_lshl_add_u32 v193, v204, 1, s5
	v_lshl_add_u32 v195, v206, 1, s5
	v_lshl_add_u32 v197, v208, 1, s5
	v_lshl_add_u32 v199, v210, 1, s5
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[16:19] /*v[272:275]*/, v183 offset:64
	ds_load_b128 v[20:23] /*v[276:279]*/, v183 offset:80
	ds_load_b128 v[24:27] /*v[280:283]*/, v185 offset:64
	ds_load_b128 v[28:31] /*v[284:287]*/, v185 offset:80
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[154:161], v[216:223], v[224:231], v[154:161]
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[32:35] /*v[288:291]*/, v189 offset:64
	ds_load_b128 v[36:39] /*v[292:295]*/, v189 offset:80
	ds_load_b128 v[40:43] /*v[296:299]*/, v191 offset:64
	ds_load_b128 v[44:47] /*v[300:303]*/, v191 offset:80
	ds_load_b128 v[48:51] /*v[304:307]*/, v195 offset:64
	ds_load_b128 v[52:55] /*v[308:311]*/, v195 offset:80
	ds_load_b128 v[56:59] /*v[312:315]*/, v197 offset:64
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_wait_dscnt 0xb
	v_wmma_f32_16x16x32_bf16 v[90:97], v[232:239], v[224:231], v[90:97] matrix_a_reuse
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[8:11] /*v[264:267]*/, v179 offset:12288
	ds_load_b128 v[12:15] /*v[268:271]*/, v179 offset:12304
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v179, v190, 1, s6
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[60:63] /*v[316:319]*/, v197 offset:80
	ds_load_b128 v[64:67] /*v[320:323]*/, v199 offset:64
	ds_load_b128 v[68:71] /*v[324:327]*/, v199 offset:80
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	s_set_vgpr_msb 0x4004                   ;  msbs: dst=0 src0=0 src1=1 src2=0
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x32_bf16 v[106:113], v[216:223], v[8:15] /*v[264:271]*/, v[106:113] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[18:25], v[232:239], v[8:15] /*v[264:271]*/, v[18:25] matrix_a_reuse
	ds_load_b128 v[224:227], v164 offset:18432
	ds_load_b128 v[228:231], v164 offset:18448
	v_lshl_add_u32 v164, v184, 1, s5
	s_set_vgpr_msb 0x440                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[8:11] /*v[264:267]*/, v181 offset:64
	ds_load_b128 v[12:15] /*v[268:271]*/, v181 offset:80
	; sched_group_barrier mask(0x00000008) size(5) SyncID(0)
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[66:73], v[216:223], v[224:231], v[66:73] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[74:81], v[232:239], v[224:231], v[74:81]
	ds_load_b128 v[224:227], v164 offset:16384
	ds_load_b128 v[228:231], v164 offset:16400
	v_lshl_add_u32 v164, v174, 1, s5
	ds_load_b128 v[240:243], v164 offset:6144
	ds_load_b128 v[244:247], v164 offset:6160
	v_lshl_add_u32 v164, v172, 1, s5
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[82:89], v[216:223], v[224:231], v[82:89] matrix_a_reuse
	ds_load_b128 v[248:251], v164 offset:4096
	ds_load_b128 v[252:255], v164 offset:4112
	v_lshl_add_u32 v164, v170, 1, s5
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[42:49], v[232:239], v[240:247], v[42:49] matrix_a_reuse
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[50:57], v[232:239], v[248:255], v[50:57] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[138:145], v[216:223], v[248:255], v[138:145] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[130:137], v[216:223], v[240:247], v[130:137] matrix_a_reuse
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[0:3] /*v[256:259]*/, v164 offset:2048
	ds_load_b128 v[4:7] /*v[260:263]*/, v164 offset:2064
	s_set_vgpr_msb 0x4004                   ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_lshl_add_u32 v164, v178, 1, s5
	ds_load_b128 v[248:251], v164 offset:10240
	ds_load_b128 v[252:255], v164 offset:10256
	v_lshl_add_u32 v164, v176, 1, s5
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[58:65], v[232:239], v[0:7] /*v[256:263]*/, v[58:65] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[146:153], v[216:223], v[0:7] /*v[256:263]*/, v[146:153] matrix_a_reuse
	s_set_vgpr_msb 0x440                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[0:3] /*v[256:259]*/, v164 offset:8192
	ds_load_b128 v[4:7] /*v[260:263]*/, v164 offset:8208
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v164, v182, 1, s5
	ds_load_b128 v[240:243], v164 offset:14336
	ds_load_b128 v[244:247], v164 offset:14352
	v_lshl_add_u32 v164, v188, 1, s6
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[114:121], v[216:223], v[248:255], v[114:121] matrix_a_reuse
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[10:17], v[232:239], v[240:247], v[10:17] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[98:105], v[216:223], v[240:247], v[98:105] matrix_a_reuse
	ds_load_b128 v[240:243], v193 offset:64
	ds_load_b128 v[244:247], v193 offset:80
	; sched_group_barrier mask(0x00000008) size(5) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[2:9], v[232:239], v[224:231], v[2:9] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(5) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[26:33], v[232:239], v[248:255], v[26:33] matrix_a_reuse
	ds_load_b128 v[248:251], v164 offset:64
	ds_load_b128 v[252:255], v164 offset:80
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[34:41], v[232:239], v[0:7] /*v[256:263]*/, v[34:41] matrix_a_reuse
	ds_load_b128 v[232:235], v187 offset:64
	ds_load_b128 v[236:239], v187 offset:80
	v_wmma_f32_16x16x32_bf16 v[122:129], v[216:223], v[0:7] /*v[256:263]*/, v[122:129] matrix_a_reuse
	s_set_vgpr_msb 0x440                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[0:3] /*v[256:259]*/, v179 offset:64
	ds_load_b128 v[4:7] /*v[260:263]*/, v179 offset:80
	; sched_group_barrier mask(0x00000008) size(5) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x4004                   ;  msbs: dst=0 src0=0 src1=1 src2=0
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[154:161], v[248:255], v[8:15] /*v[264:271]*/, v[154:161]
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[146:153], v[248:255], v[16:23] /*v[272:279]*/, v[146:153] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[138:145], v[248:255], v[24:31] /*v[280:287]*/, v[138:145] matrix_a_reuse
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[130:137], v[248:255], v[232:239], v[130:137] matrix_a_reuse
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[122:129], v[248:255], v[32:39] /*v[288:295]*/, v[122:129] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(5) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[114:121], v[248:255], v[40:47] /*v[296:303]*/, v[114:121] matrix_a_reuse
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_wmma_f32_16x16x32_bf16 v[106:113], v[248:255], v[240:247], v[106:113] matrix_a_reuse
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[98:105], v[248:255], v[48:55] /*v[304:311]*/, v[98:105] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[82:89], v[248:255], v[56:63] /*v[312:319]*/, v[82:89] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[66:73], v[248:255], v[64:71] /*v[320:327]*/, v[66:73] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(5) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	s_set_vgpr_msb 0x405                    ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[74:81], v[0:7] /*v[256:263]*/, v[64:71] /*v[320:327]*/, v[74:81]
	v_wmma_f32_16x16x32_bf16 v[2:9], v[0:7] /*v[256:263]*/, v[56:63] /*v[312:319]*/, v[2:9] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[10:17], v[0:7] /*v[256:263]*/, v[48:55] /*v[304:311]*/, v[10:17] matrix_a_reuse
	s_set_vgpr_msb 0x501                    ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_wmma_f32_16x16x32_bf16 v[18:25], v[0:7] /*v[256:263]*/, v[240:247], v[18:25] matrix_a_reuse
	s_set_vgpr_msb 0x105                    ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[26:33], v[0:7] /*v[256:263]*/, v[40:47] /*v[296:303]*/, v[26:33] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(5) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[34:41], v[0:7] /*v[256:263]*/, v[32:39] /*v[288:295]*/, v[34:41] matrix_a_reuse
	s_set_vgpr_msb 0x501                    ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_wmma_f32_16x16x32_bf16 v[42:49], v[0:7] /*v[256:263]*/, v[232:239], v[42:49] matrix_a_reuse
	s_set_vgpr_msb 0x105                    ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[50:57], v[0:7] /*v[256:263]*/, v[24:31] /*v[280:287]*/, v[50:57] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[58:65], v[0:7] /*v[256:263]*/, v[16:23] /*v[272:279]*/, v[58:65] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[90:97], v[0:7] /*v[256:263]*/, v[8:15] /*v[264:271]*/, v[90:97] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(5) SyncID(0)
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
	v_cmpx_eq_u32_e32 1, v171
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
	v_dual_mov_b32 v179, s13 :: v_dual_mov_b32 v216, s14
	s_and_b32 s10, s15, 0x1ffffff
	s_and_b32 s13, s29, s17
	s_bitset1_b32 s10, 31
	v_cndmask_b32_e64 v164, 0, 1, s13
	v_mov_b32_e32 v181, s10
	v_readfirstlane_b32 s57, v179
	v_readfirstlane_b32 s58, v216
	s_lshr_b64 s[14:15], s[36:37], 16
	v_readfirstlane_b32 s56, v164
	v_readfirstlane_b32 s59, v181
	s_lshl_b32 s13, s34, 16
	s_mov_b32 s15, s49
	s_mov_b32 s17, s9
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[56:59], s[12:19]
.LBB0_39:                               ;   in Loop: Header=BB0_19 Depth=1
	s_or_b32 exec_lo, exec_lo, s52
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
	v_cndmask_b32_e64 v164, 0, 1, s10
	s_and_b32 s7, s7, 0x1ffffff
	v_dual_mov_b32 v179, s5 :: v_dual_mov_b32 v216, s6
	s_bitset1_b32 s7, 31
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_readfirstlane_b32 s52, v164
	v_mov_b32_e32 v181, s7
	v_readfirstlane_b32 s53, v179
	v_readfirstlane_b32 s54, v216
	s_lshr_b64 s[6:7], s[34:35], 16
	s_lshl_b32 s5, s34, 16
	v_readfirstlane_b32 s55, v181
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
	v_and_or_b32 v162, v0, 15, v169
	v_cvt_pk_bf16_f32 v153, v152, v153
	v_cvt_pk_bf16_f32 v152, v150, v151
	v_cvt_pk_bf16_f32 v150, v146, v147
	v_and_or_b32 v1, v1, 8, v167
	v_cvt_pk_bf16_f32 v145, v144, v145
	v_cvt_pk_bf16_f32 v144, v142, v143
	v_lshlrev_b32_e32 v146, 4, v162
	v_cvt_pk_bf16_f32 v142, v138, v139
	v_lshl_or_b32 v1, v162, 7, v1
	v_cvt_pk_bf16_f32 v143, v140, v141
	v_cvt_pk_bf16_f32 v161, v160, v161
	v_and_b32_e32 v139, 0xbf0, v146
	v_cvt_pk_bf16_f32 v160, v158, v159
	v_add_nc_u32_e32 v147, 0x800, v1
	v_add_nc_u32_e32 v138, 0x1000, v1
	v_lshlrev_b32_e32 v141, 1, v1
	v_cvt_pk_bf16_f32 v159, v156, v157
	v_cvt_pk_bf16_f32 v158, v154, v155
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_lshrrev_b32 v140, 3, v147 :: v_dual_lshrrev_b32 v138, 3, v138
	v_add_nc_u32_e32 v147, 0x1800, v1
	v_add_nc_u32_e32 v139, v139, v141
	v_cvt_pk_bf16_f32 v151, v148, v149
	v_and_b32_e32 v140, 0x1ff0, v140
	v_and_b32_e32 v138, 0x1ff0, v138
	v_lshrrev_b32_e32 v147, 3, v147
	v_cvt_pk_bf16_f32 v97, v96, v97
	v_cvt_pk_bf16_f32 v96, v94, v95
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_nc_u32 v140, v140, v141 :: v_dual_add_nc_u32 v138, v138, v141
	v_cvt_pk_bf16_f32 v94, v90, v91
	v_add_nc_u32_e32 v90, 0x2800, v1
	ds_store_b128 v139, v[158:161]
	ds_store_b128 v140, v[150:153] offset:4096
	v_and_b32_e32 v139, 0x1ff0, v147
	v_cvt_pk_bf16_f32 v95, v92, v93
	ds_store_b128 v138, v[142:145] offset:8192
	v_cvt_pk_bf16_f32 v93, v136, v137
	v_cvt_pk_bf16_f32 v92, v134, v135
	v_dual_add_nc_u32 v138, v139, v141 :: v_dual_lshrrev_b32 v134, 3, v90
	v_cvt_pk_bf16_f32 v91, v132, v133
	v_cvt_pk_bf16_f32 v90, v130, v131
	v_cvt_pk_bf16_f32 v121, v120, v121
	v_cvt_pk_bf16_f32 v120, v118, v119
	v_cvt_pk_bf16_f32 v119, v116, v117
	v_add_nc_u32_e32 v116, 0x3800, v1
	ds_store_b128 v138, v[90:93] offset:12288
	v_cvt_pk_bf16_f32 v65, v64, v65
	v_cvt_pk_bf16_f32 v64, v62, v63
	v_cvt_pk_bf16_f32 v63, v60, v61
	v_lshrrev_b32_e32 v91, 3, v116
	v_add_nc_u32_e32 v60, 0x1810, v1
	v_cvt_pk_bf16_f32 v129, v128, v129
	v_cvt_pk_bf16_f32 v128, v126, v127
	v_cvt_pk_bf16_f32 v127, v124, v125
	v_add_nc_u32_e32 v124, 0x3000, v1
	v_cvt_pk_bf16_f32 v41, v40, v41
	v_cvt_pk_bf16_f32 v40, v38, v39
	v_cvt_pk_bf16_f32 v39, v36, v37
	v_add_nc_u32_e32 v36, 0x3010, v1
	v_add_nc_u32_e32 v146, 16, v1
	v_add_nc_u32_e32 v148, 0x2000, v1
	v_cvt_pk_bf16_f32 v93, v112, v113
	v_and_b32_e32 v112, 0x1ff0, v91
	v_cvt_pk_bf16_f32 v91, v108, v109
	v_cvt_pk_bf16_f32 v105, v104, v105
	v_cvt_pk_bf16_f32 v104, v102, v103
	v_add_nc_u32_e32 v108, 0x4800, v1
	v_cvt_pk_bf16_f32 v103, v100, v101
	v_add_nc_u32_e32 v100, 0x810, v1
	v_cvt_pk_bf16_f32 v73, v72, v73
	v_cvt_pk_bf16_f32 v72, v70, v71
	v_cvt_pk_bf16_f32 v71, v68, v69
	v_add_nc_u32_e32 v68, 0x1010, v1
	v_cvt_pk_bf16_f32 v62, v58, v59
	v_lshrrev_b32_e32 v59, 3, v60
	v_add_nc_u32_e32 v60, 0x2010, v1
	v_cvt_pk_bf16_f32 v57, v56, v57
	v_cvt_pk_bf16_f32 v56, v54, v55
	v_cvt_pk_bf16_f32 v55, v52, v53
	v_add_nc_u32_e32 v52, 0x2810, v1
	v_cvt_pk_bf16_f32 v126, v122, v123
	v_lshrrev_b32_e32 v123, 3, v124
	v_cvt_pk_bf16_f32 v118, v114, v115
	v_add_nc_u32_e32 v114, 0x4000, v1
	v_cvt_pk_bf16_f32 v38, v34, v35
	v_lshrrev_b32_e32 v35, 3, v36
	v_add_nc_u32_e32 v36, 0x3810, v1
	v_cvt_pk_bf16_f32 v33, v32, v33
	v_cvt_pk_bf16_f32 v32, v30, v31
	v_cvt_pk_bf16_f32 v31, v28, v29
	v_add_nc_u32_e32 v28, 0x4010, v1
	v_add_nc_u32_e32 v1, 0x4810, v1
	v_dual_lshrrev_b32 v149, 3, v146 :: v_dual_lshrrev_b32 v140, 3, v148
	v_cvt_pk_bf16_f32 v102, v98, v99
	v_cvt_pk_bf16_f32 v89, v88, v89
	v_lshrrev_b32_e32 v99, 3, v108
	v_cvt_pk_bf16_f32 v88, v86, v87
	v_cvt_pk_bf16_f32 v86, v82, v83
	v_lshrrev_b32_e32 v83, 3, v100
	v_cvt_pk_bf16_f32 v70, v66, v67
	v_lshrrev_b32_e32 v67, 3, v68
	v_lshrrev_b32_e32 v60, 3, v60
	v_cvt_pk_bf16_f32 v49, v48, v49
	v_cvt_pk_bf16_f32 v48, v46, v47
	v_cvt_pk_bf16_f32 v46, v42, v43
	v_lshrrev_b32_e32 v43, 3, v52
	v_and_b32_e32 v90, 0x1ff0, v123
	v_cvt_pk_bf16_f32 v92, v110, v111
	v_lshrrev_b32_e32 v110, 3, v114
	v_cvt_pk_bf16_f32 v30, v26, v27
	v_lshrrev_b32_e32 v26, 3, v36
	v_cvt_pk_bf16_f32 v25, v24, v25
	v_cvt_pk_bf16_f32 v24, v22, v23
	v_cvt_pk_bf16_f32 v23, v20, v21
	v_dual_lshrrev_b32 v20, 3, v28 :: v_dual_lshrrev_b32 v1, 3, v1
	v_and_b32_e32 v149, 0xbf0, v149
	v_and_b32_e32 v136, 0x1ff0, v140
	v_and_b32_e32 v131, 0x1ff0, v134
	v_cvt_pk_bf16_f32 v87, v84, v85
	v_and_b32_e32 v82, 0x1ff0, v99
	v_and_b32_e32 v83, 0x1ff0, v83
	v_lshlrev_b32_e32 v84, 1, v146
	v_and_b32_e32 v58, 0x1ff0, v67
	v_cvt_pk_bf16_f32 v54, v50, v51
	v_and_b32_e32 v51, 0x1ff0, v60
	v_and_b32_e32 v34, 0x1ff0, v43
	v_add_nc_u32_e32 v115, v90, v141
	v_cvt_pk_bf16_f32 v90, v106, v107
	v_and_b32_e32 v107, 0x1ff0, v110
	v_and_b32_e32 v59, 0x1ff0, v59
	v_and_b32_e32 v35, 0x1ff0, v35
	v_and_b32_e32 v26, 0x1ff0, v26
	v_cvt_pk_bf16_f32 v22, v18, v19
	v_and_b32_e32 v19, 0x1ff0, v20
	v_and_b32_e32 v1, 0x1ff0, v1
	v_dual_add_nc_u32 v147, v149, v141 :: v_dual_add_nc_u32 v130, v136, v141
	v_dual_add_nc_u32 v122, v131, v141 :: v_dual_add_nc_u32 v106, v112, v141
	v_dual_add_nc_u32 v82, v82, v141 :: v_dual_add_nc_u32 v66, v83, v84
	v_dual_add_nc_u32 v58, v58, v84 :: v_dual_add_nc_u32 v42, v51, v84
	v_dual_add_nc_u32 v34, v34, v84 :: v_dual_add_nc_u32 v98, v107, v141
	v_add_nc_u32_e32 v50, v59, v84
	v_cvt_pk_bf16_f32 v47, v44, v45
	v_dual_add_nc_u32 v27, v35, v84 :: v_dual_add_nc_u32 v18, v26, v84
	v_cvt_pk_bf16_f32 v17, v16, v17
	v_cvt_pk_bf16_f32 v16, v14, v15
	v_cvt_pk_bf16_f32 v15, v12, v13
	v_cvt_pk_bf16_f32 v14, v10, v11
	v_add_nc_u32_e32 v10, v19, v84
	v_cvt_pk_bf16_f32 v9, v8, v9
	v_cvt_pk_bf16_f32 v8, v6, v7
	v_cvt_pk_bf16_f32 v7, v4, v5
	v_cvt_pk_bf16_f32 v6, v2, v3
	v_add_nc_u32_e32 v1, v1, v84
	v_cvt_pk_bf16_f32 v5, v80, v81
	v_cvt_pk_bf16_f32 v4, v78, v79
	v_cvt_pk_bf16_f32 v3, v76, v77
	v_cvt_pk_bf16_f32 v2, v74, v75
	ds_store_b128 v130, v[126:129] offset:16384
	ds_store_b128 v122, v[118:121] offset:20480
	ds_store_b128 v115, v[90:93] offset:24576
	ds_store_b128 v106, v[102:105] offset:28672
	ds_store_b128 v98, v[86:89] offset:32768
	ds_store_b128 v82, v[70:73] offset:36864
	ds_store_b128 v147, v[94:97] offset:32
	ds_store_b128 v66, v[62:65] offset:4096
	ds_store_b128 v58, v[54:57] offset:8192
	ds_store_b128 v50, v[46:49] offset:12288
	ds_store_b128 v42, v[38:41] offset:16384
	ds_store_b128 v34, v[30:33] offset:20480
	ds_store_b128 v27, v[22:25] offset:24576
	ds_store_b128 v18, v[14:17] offset:28672
	ds_store_b128 v10, v[6:9] offset:32768
	ds_store_b128 v1, v[2:5] offset:36864
.LBB0_43:
	v_cmp_ne_u32_e32 vcc_lo, 1, v163
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
	s_mul_i32 s15, s33, 0x140
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
	v_lshl_add_u32 v11, v11, 7, v12
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_mul_lo_u32 v16, v27, s22
	v_dual_add_nc_u32 v24, s19, v27 :: v_dual_sub_nc_u32 v6, v2, v6
	v_ashrrev_i32_e32 v19, 31, v18
	v_lshl_add_u32 v26, v26, 7, v14
	v_mul_u64_e32 v[20:21], s[6:7], v[20:21]
	v_ashrrev_i32_e32 v25, 31, v24
	v_lshl_add_u32 v1, v1, 7, v6
	v_sub_nc_u32_e32 v16, v5, v16
	v_mul_u64_e32 v[18:19], s[28:29], v[18:19]
	v_mul_u64_e32 v[22:23], s[8:9], v[22:23]
	v_mul_u64_e32 v[24:25], s[10:11], v[24:25]
	v_ashrrev_i32_e32 v28, 31, v1
	v_lshl_add_u32 v27, v27, 7, v16
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
	v_dual_sub_nc_u32 v7, v4, v9 :: v_dual_lshlrev_b32 v4, 7, v4
	v_lshlrev_b32_e32 v9, 7, v9
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
	.size	bm128_bn320_bk064_wm4_wn2_mc1, .Lfunc_end0-bm128_bn320_bk064_wm4_wn2_mc1
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel bm128_bn320_bk064_wm4_wn2_mc1
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
		.amdhsa_next_free_vgpr 328
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
		.amdhsa_inst_pref_size 67
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text.bm128_bn320_bk064_wm4_wn2_mc1,"axG",@progbits,bm128_bn320_bk064_wm4_wn2_mc1,comdat
                                        ; -- End function
	.set .Lbm128_bn320_bk064_wm4_wn2_mc1.num_vgpr, 328
	.set .Lbm128_bn320_bk064_wm4_wn2_mc1.num_agpr, 0
	.set .Lbm128_bn320_bk064_wm4_wn2_mc1.numbered_sgpr, 60
	.set .Lbm128_bn320_bk064_wm4_wn2_mc1.num_named_barrier, 0
	.set .Lbm128_bn320_bk064_wm4_wn2_mc1.private_seg_size, 0
	.set .Lbm128_bn320_bk064_wm4_wn2_mc1.uses_vcc, 1
	.set .Lbm128_bn320_bk064_wm4_wn2_mc1.uses_flat_scratch, 1
	.set .Lbm128_bn320_bk064_wm4_wn2_mc1.has_dyn_sized_stack, 0
	.set .Lbm128_bn320_bk064_wm4_wn2_mc1.has_recursion, 0
	.set .Lbm128_bn320_bk064_wm4_wn2_mc1.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 8524
; TotalNumSgprs: 62
; NumVgprs: 328
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 121856 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 20
; NumSGPRsForWavesPerEU: 62
; NumVGPRsForWavesPerEU: 328
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
	.type	__hip_cuid_3d3227ae7577e781,@object ; @__hip_cuid_3d3227ae7577e781
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_3d3227ae7577e781
__hip_cuid_3d3227ae7577e781:
	.byte	0                               ; 0x0
	.size	__hip_cuid_3d3227ae7577e781, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git aa451e1fe6a793394d6733051b1778633063ae96+PATCHED:d17c5aa0e3ea29cde402f58f27e39b6034effa27)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_3d3227ae7577e781
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
    .group_segment_fixed_size: 121856
    .kernarg_segment_align: 8
    .kernarg_segment_size: 132
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           bm128_bn320_bk064_wm4_wn2_mc1
    .private_segment_fixed_size: 0
    .sgpr_count:     62
    .sgpr_spill_count: 0
    .symbol:         bm128_bn320_bk064_wm4_wn2_mc1.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     328
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
