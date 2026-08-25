	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.section	.text.bm224_bn128_bk064_wm2_wn4_mc0,"axG",@progbits,bm224_bn128_bk064_wm2_wn4_mc0,comdat
	.protected	bm224_bn128_bk064_wm2_wn4_mc0 ; -- Begin function bm224_bn128_bk064_wm2_wn4_mc0
	.globl	bm224_bn128_bk064_wm2_wn4_mc0
	.p2align	8
	.type	bm224_bn128_bk064_wm2_wn4_mc0,@function
bm224_bn128_bk064_wm2_wn4_mc0: ; @bm224_bn128_bk064_wm2_wn4_mc0
; %bb.0:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1 ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_mov_b64 s[2:3], src_shared_base
	s_movk_i32 s2, 0x7700
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
	s_add_co_i32 s2, s25, 0x7f
	s_lshr_b32 s6, s3, 31
	s_ashr_i32 s3, s3, 7
	s_ashr_i32 s7, s2, 31
	s_add_co_i32 s6, s3, s6
	s_lshr_b32 s3, s7, 25
	s_mul_i32 s7, s36, 0xffffff20
	s_add_co_i32 s2, s2, s3
	s_add_co_i32 s3, s24, s7
	s_ashr_i32 s7, s2, 7
	s_min_i32 s27, s3, 0xe0
	s_cmp_lt_i32 s36, s6
	s_cselect_b32 s37, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_and_b32 s2, s37, exec_lo
	s_cselect_b32 s3, s27, 0
	s_lshl_b32 s33, s5, 7
	s_sub_co_i32 s2, s25, s33
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_min_i32 s2, s2, 0x80
	s_cmp_lt_i32 s5, s7
	s_cselect_b32 s25, -1, 0
	s_and_b32 s8, s25, exec_lo
	s_cselect_b32 s29, s2, 0
	s_add_co_i32 s17, s26, 63
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 2, 1), 1
	s_min_i32 s2, s26, 64
	s_cmp_gt_i32 s17, 63
	s_cselect_b32 s16, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_and_b32 s8, s16, exec_lo
	s_cselect_b32 s2, s2, 0
	s_cmp_lt_i32 s3, 0xe0
	s_cselect_b32 s38, -1, 0
	s_and_b32 vcc_lo, exec_lo, s38
	s_mov_b32 s8, s38
	s_cbranch_vccnz .LBB0_2
; %bb.1:
	s_cmp_lt_i32 s29, 0x80
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
	v_lshl_add_u32 v2, s4, 2, v2
	v_mov_b32_e32 v3, 0
	s_mov_b32 s8, 0
.LBB0_6:                                ; =>This Inner Loop Header: Depth=1
	v_add_nc_u32_e32 v1, 0x100, v1
	ds_store_b32 v2, v3 offset:30464
	v_add_nc_u32_e32 v2, 0x400, v2
	v_cmp_lt_u32_e32 vcc_lo, 0xfff, v1
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
	v_lshrrev_b32_e32 v121, 5, v0
	s_lshl_b32 s30, s4, 2
	s_add_co_i32 s7, s7, -1
	s_wait_xcnt 0x0
	s_mov_b64 s[0:1], src_shared_base
	s_or_b32 s40, s30, 0x7700
	s_add_co_i32 s31, s6, -1
	s_min_i32 s0, s5, s7
	s_mov_b32 s4, exec_lo
	v_cmpx_lt_i32_e32 0, v121
	s_xor_b32 s34, exec_lo, s4
	s_cbranch_execz .LBB0_12
; %bb.9:
	s_mov_b32 s35, exec_lo
	v_cmpx_eq_u32_e32 1, v121
	s_cbranch_execz .LBB0_11
; %bb.10:
	s_cmp_gt_i32 s2, 0
	s_mov_b32 s28, s2
	s_cselect_b32 s8, -1, 0
	s_lshl_b32 s4, s0, 7
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
	s_and_b32 s7, s25, s8
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
	s_or_b32 s7, s4, 0x400000
	s_movk_i32 s8, 0x80
	s_mov_b32 s4, 0x7510000
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[44:47], s[4:11]
.LBB0_11:
	s_or_b32 exec_lo, exec_lo, s35
.LBB0_12:
	s_or_saveexec_b32 s28, s34
	s_min_i32 s4, s36, s31
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
	s_or_b32 s7, s4, 0x400000
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
	v_dual_lshrrev_b32 v119, 7, v0 :: v_dual_mov_b32 v9, 0
	s_and_b32 s39, s37, s25
	s_and_not1_b32 vcc_lo, exec_lo, s16
	v_cndmask_b32_e64 v115, 0, 1, s39
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
	v_dual_mov_b32 v50, v9 :: v_dual_mov_b32 v73, v9
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
	v_dual_mov_b32 v98, v9 :: v_dual_mov_b32 v113, v9
	v_dual_mov_b32 v112, v9 :: v_dual_mov_b32 v111, v9
	v_dual_mov_b32 v110, v9 :: v_dual_mov_b32 v109, v9
	v_dual_mov_b32 v108, v9 :: v_dual_mov_b32 v107, v9
	v_dual_mov_b32 v106, v9 :: v_dual_mov_b32 v65, v9
	v_dual_mov_b32 v64, v9 :: v_dual_mov_b32 v63, v9
	v_dual_mov_b32 v62, v9 :: v_dual_mov_b32 v61, v9
	v_dual_mov_b32 v60, v9 :: v_dual_mov_b32 v59, v9
	v_mov_b32_e32 v58, v9
	s_barrier_wait -1
	s_cbranch_vccnz .LBB0_37
; %bb.15:
	v_dual_lshlrev_b32 v1, 6, v0 :: v_dual_bitop2_b32 v2, 16, v0 bitop3:0x40
	v_mul_u32_u24_e32 v3, 0x1c00, v119
	s_mov_b64 s[4:5], src_shared_base
	s_or_b32 s6, s30, 0xbb00
	s_mov_b32 s7, s5
	v_and_or_b32 v5, 0x3c0, v1, v2
	v_and_b32_e32 v1, 0x1800, v1
	s_and_b64 s[6:7], s[6:7], 15
	s_mov_b32 s11, 0
	s_sub_co_i32 s2, 16, s6
	v_or_b32_e32 v7, v5, v3
	s_lshr_b32 s2, s2, 2
	s_cmp_lg_u64 s[6:7], 0
	s_mov_b32 s42, s5
	s_cselect_b32 s2, s2, 0
	v_lshrrev_b32_e32 v2, 4, v7
	s_lshl2_add_u32 s2, s2, s30
	v_lshl_or_b32 v154, v0, 2, 0x4000
	s_add_co_i32 s4, s2, 0x13200
	v_mov_b32_e32 v117, 0
	v_and_b32_e32 v6, 0x1f8, v2
	v_add_nc_u32_e32 v2, 0x400, v7
	s_and_b32 s10, s4, 15
	s_add_co_i32 s43, s2, 0xbb00
	s_sub_co_i32 s6, 16, s10
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v155, v117 :: v_dual_lshrrev_b32 v9, 4, v2
	v_add_nc_u32_e32 v4, 0x800, v7
	v_add_nc_u32_e32 v114, v6, v7
	v_add_nc_u32_e32 v6, 0xc00, v7
	v_add_nc_u32_e32 v8, 0x1000, v7
	v_and_b32_e32 v9, 0x3f8, v9
	v_lshrrev_b32_e32 v10, 4, v4
	s_lshr_b32 s2, s6, 2
	s_cmp_lg_u64 s[10:11], 0
	v_dual_mov_b32 v29, v117 :: v_dual_mov_b32 v30, v117
	s_delay_alu instid0(VALU_DEP_2)
	v_and_b32_e32 v13, 0x3f8, v10
	v_add_nc_u32_e32 v10, 0x1400, v7
	v_or_b32_e32 v24, v5, v1
	v_or_b32_e32 v5, 32, v5
	s_cselect_b32 s2, s2, 0
	s_ashr_i32 s6, s17, 31
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_lshrrev_b32 v16, 4, v10 :: v_dual_lshrrev_b32 v14, 4, v24
	v_or_b32_e32 v17, 0x400, v24
	v_or_b32_e32 v3, v5, v3
	v_or_b32_e32 v1, v5, v1
	v_and_b32_e32 v19, 0x3f8, v16
	v_and_b32_e32 v14, 0x1b8, v14
	v_lshrrev_b32_e32 v16, 4, v17
	v_add_nc_u32_e32 v20, 0xc00, v3
	v_add_nc_u32_e32 v5, 0x1400, v3
	v_add_nc_u32_e32 v22, 0x1000, v3
	v_add_nc_u32_e32 v118, v14, v24
	v_and_b32_e32 v21, 0x1f8, v16
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_lshrrev_b32 v16, 4, v3 :: v_dual_lshrrev_b32 v5, 4, v5
	v_add_nc_u32_e32 v128, v19, v7
	v_dual_mov_b32 v19, v117 :: v_dual_add_nc_u32 v14, 0x400, v3
	v_and_b32_e32 v23, 0x1f8, v16
	v_add_nc_u32_e32 v132, v21, v24
	v_dual_mov_b32 v21, v117 :: v_dual_lshrrev_b32 v16, 4, v20
	v_or_b32_e32 v20, 0x400, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_dual_lshrrev_b32 v14, 4, v14 :: v_dual_add_nc_u32 v134, v23, v7
	s_lshr_b32 s6, s6, 26
	s_lshl_b32 s10, s2, 2
	v_lshrrev_b32_e32 v25, 4, v20
	v_and_b32_e32 v20, 0x3f8, v5
	v_dual_mov_b32 v5, v117 :: v_dual_lshrrev_b32 v12, 4, v8
	v_and_b32_e32 v116, 0x3f8, v14
	s_add_co_i32 s17, s17, s6
	s_add_nc_u64 s[30:31], s[4:5], s[10:11]
	s_ashr_i32 s44, s17, 6
	v_and_b32_e32 v15, 0x3f8, v12
	v_add_nc_u32_e32 v12, 0x1800, v7
	v_lshrrev_b32_e32 v1, 4, v1
	s_cmp_lt_i32 s29, 0x80
	v_and_b32_e32 v25, 0x1f8, v25
	s_cselect_b32 s45, -1, 0
	v_dual_mov_b32 v23, v117 :: v_dual_lshrrev_b32 v18, 4, v12
	s_lshl_b32 s4, s0, 7
	v_and_b32_e32 v27, 0x1b8, v1
	s_ashr_i32 s5, s4, 31
	s_delay_alu instid0(VALU_DEP_2)
	v_and_b32_e32 v17, 0x3f8, v18
	v_add_nc_u32_e32 v18, 0x800, v3
	v_dual_mov_b32 v15, v117 :: v_dual_add_nc_u32 v126, v15, v7
	v_lshrrev_b32_e32 v11, 4, v6
	v_add_nc_u32_e32 v3, 0x1800, v3
	v_add_nc_u32_e32 v122, v13, v7
	v_add_nc_u32_e32 v130, v17, v7
	v_mov_b32_e32 v17, v117
	v_and_b32_e32 v11, 0x3f8, v11
	v_add_nc_u32_e32 v120, v9, v7
	v_dual_mov_b32 v9, v117 :: v_dual_lshrrev_b32 v18, 4, v18
	v_lshrrev_b32_e32 v3, 4, v3
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v11, v117 :: v_dual_add_nc_u32 v124, v11, v7
	v_mov_b32_e32 v7, v117
	v_and_b32_e32 v14, 0x3f8, v18
	v_lshrrev_b32_e32 v18, 4, v22
	v_sub_nc_u32_e32 v22, 0x1dbf, v0
	v_and_b32_e32 v16, 0x3f8, v16
	v_mov_b32_e32 v13, v117
	s_ashr_i32 s35, s34, 31
	v_and_b32_e32 v18, 0x3f8, v18
	v_lshrrev_b32_e32 v26, 8, v22
	v_and_b32_e32 v22, 0x3f8, v3
	s_wait_kmcnt 0x0
	s_bfe_i64 s[6:7], s[20:21], 0x200000
	s_bfe_i64 s[12:13], s[12:13], 0x200000
	s_mul_u64 s[4:5], s[6:7], s[4:5]
	v_add_nc_u32_e32 v3, 1, v26
	s_mul_u64 s[6:7], s[12:13], s[34:35]
	v_add_nc_u64_e32 v[140:141], v[16:17], v[6:7]
	v_add_nc_u64_e32 v[142:143], v[18:19], v[8:9]
	v_add_nc_u64_e32 v[144:145], v[20:21], v[10:11]
	v_and_b32_e32 v123, 30, v3
	v_add_nc_u64_e32 v[146:147], v[22:23], v[12:13]
	v_or_b32_e32 v1, 0x100, v0
	v_dual_mov_b32 v153, v117 :: v_dual_add_nc_u32 v148, v27, v24
	s_delay_alu instid0(VALU_DEP_4)
	v_cmp_ne_u32_e64 s0, v3, v123
	v_mov_b32_e32 v3, v117
	v_lshl_or_b32 v26, v123, 8, v0
	v_add_nc_u64_e32 v[138:139], v[14:15], v[4:5]
	v_add3_u32 v150, v24, v25, 0x400
	v_mov_b32_e32 v4, v117
	v_add_nc_u64_e32 v[136:137], v[116:117], v[2:3]
	v_dual_mov_b32 v6, v117 :: v_dual_add_nc_u32 v125, 0xffffff00, v26
	v_dual_lshlrev_b32 v152, 2, v26 :: v_dual_mov_b32 v2, v117
	v_dual_mov_b32 v8, v117 :: v_dual_mov_b32 v10, v117
	v_dual_mov_b32 v12, v117 :: v_dual_mov_b32 v14, v117
	v_dual_mov_b32 v16, v117 :: v_dual_mov_b32 v18, v117
	v_dual_mov_b32 v20, v117 :: v_dual_mov_b32 v22, v117
	v_dual_mov_b32 v24, v117 :: v_dual_mov_b32 v25, v117
	v_dual_mov_b32 v26, v117 :: v_dual_mov_b32 v27, v117
	v_dual_mov_b32 v28, v117 :: v_dual_mov_b32 v31, v117
	v_dual_mov_b32 v32, v117 :: v_dual_mov_b32 v33, v117
	v_dual_mov_b32 v34, v117 :: v_dual_mov_b32 v35, v117
	v_dual_mov_b32 v36, v117 :: v_dual_mov_b32 v37, v117
	v_dual_mov_b32 v38, v117 :: v_dual_mov_b32 v39, v117
	v_dual_mov_b32 v40, v117 :: v_dual_mov_b32 v41, v117
	v_dual_mov_b32 v42, v117 :: v_dual_mov_b32 v43, v117
	v_dual_mov_b32 v44, v117 :: v_dual_mov_b32 v45, v117
	v_dual_mov_b32 v46, v117 :: v_dual_mov_b32 v47, v117
	v_dual_mov_b32 v48, v117 :: v_dual_mov_b32 v49, v117
	v_dual_mov_b32 v50, v117 :: v_dual_mov_b32 v51, v117
	v_dual_mov_b32 v52, v117 :: v_dual_mov_b32 v53, v117
	v_dual_mov_b32 v54, v117 :: v_dual_mov_b32 v55, v117
	v_dual_mov_b32 v56, v117 :: v_dual_mov_b32 v57, v117
	v_dual_mov_b32 v66, v117 :: v_dual_mov_b32 v67, v117
	v_dual_mov_b32 v68, v117 :: v_dual_mov_b32 v69, v117
	v_dual_mov_b32 v70, v117 :: v_dual_mov_b32 v71, v117
	v_dual_mov_b32 v72, v117 :: v_dual_mov_b32 v73, v117
	v_dual_mov_b32 v74, v117 :: v_dual_mov_b32 v75, v117
	v_dual_mov_b32 v76, v117 :: v_dual_mov_b32 v77, v117
	v_dual_mov_b32 v78, v117 :: v_dual_mov_b32 v79, v117
	v_dual_mov_b32 v80, v117 :: v_dual_mov_b32 v81, v117
	v_dual_mov_b32 v82, v117 :: v_dual_mov_b32 v83, v117
	v_dual_mov_b32 v84, v117 :: v_dual_mov_b32 v85, v117
	v_dual_mov_b32 v86, v117 :: v_dual_mov_b32 v87, v117
	v_dual_mov_b32 v88, v117 :: v_dual_mov_b32 v89, v117
	v_dual_mov_b32 v90, v117 :: v_dual_mov_b32 v91, v117
	v_dual_mov_b32 v92, v117 :: v_dual_mov_b32 v93, v117
	v_dual_mov_b32 v94, v117 :: v_dual_mov_b32 v95, v117
	v_dual_mov_b32 v96, v117 :: v_dual_mov_b32 v97, v117
	v_dual_mov_b32 v98, v117 :: v_dual_mov_b32 v99, v117
	v_dual_mov_b32 v100, v117 :: v_dual_mov_b32 v101, v117
	v_dual_mov_b32 v102, v117 :: v_dual_mov_b32 v103, v117
	v_dual_mov_b32 v104, v117 :: v_dual_mov_b32 v105, v117
	v_dual_mov_b32 v106, v117 :: v_dual_mov_b32 v107, v117
	v_dual_mov_b32 v108, v117 :: v_dual_mov_b32 v109, v117
	v_dual_mov_b32 v110, v117 :: v_dual_mov_b32 v111, v117
	v_dual_mov_b32 v112, v117 :: v_dual_mov_b32 v113, v117
	v_dual_mov_b32 v58, v117 :: v_dual_mov_b32 v59, v117
	v_dual_mov_b32 v60, v117 :: v_dual_mov_b32 v61, v117
	v_dual_mov_b32 v62, v117 :: v_dual_mov_b32 v63, v117
	v_dual_mov_b32 v64, v117 :: v_dual_mov_b32 v65, v117
	v_or_b32_e32 v127, 0xf00, v0
	s_lshr_b32 s46, s29, 16
	s_lshr_b32 s47, s3, 16
	s_lshl_b64 s[4:5], s[4:5], 1
	s_lshl_b64 s[6:7], s[6:7], 1
	s_mov_b32 s41, s1
	s_movk_i32 s16, 0x80
	s_bitset1_b32 s46, 22
	s_movk_i32 s8, 0xe0
	s_bitset1_b32 s47, 22
	s_add_nc_u64 s[20:21], s[14:15], s[4:5]
	s_add_nc_u64 s[34:35], s[18:19], s[6:7]
	s_mov_b32 s48, -1
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
	s_xor_b32 s5, s50, 1
	s_lshl_b32 s2, s49, 6
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_sub_co_i32 s2, s26, s2
	s_min_i32 s2, s2, 64
	s_cmp_lt_i32 s49, s44
	s_cselect_b32 s10, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_and_b32 s6, s10, exec_lo
	s_cselect_b32 s2, s2, 0
	s_cmp_lt_i32 s2, 64
	s_cselect_b32 s6, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b32 s6, s45, s6
	s_or_b32 s6, s38, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s6
	s_cbranch_vccnz .LBB0_29
; %bb.18:                               ;   in Loop: Header=BB0_17 Depth=1
	v_nop
	v_nop
	v_nop
	v_nop
	v_mov_b64_e32 v[156:157], v[0:1]
	v_mov_b32_e32 v129, v123
	s_cmp_lg_u32 s5, 0
	s_mov_b32 s12, 0
	s_cselect_b32 s7, s42, s1
	s_cselect_b32 s6, s43, 0
.LBB0_19:                               ;   Parent Loop BB0_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_dual_mov_b32 v116, v156 :: v_dual_add_nc_u32 v129, -2, v129
	v_dual_mov_b32 v158, v157 :: v_dual_mov_b32 v159, v117
	v_add_nc_u32_e32 v157, 0x200, v157
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u64 v[160:161], v[116:117], 2, s[6:7]
	v_cmp_eq_u32_e32 vcc_lo, 0, v129
	v_add_nc_u32_e32 v156, 0x200, v156
	v_lshl_add_u64 v[158:159], v[158:159], 2, s[6:7]
	s_clause 0x1
	flat_store_b32 v[160:161], v117
	flat_store_b32 v[158:159], v117
	s_or_b32 s12, vcc_lo, s12
	s_wait_xcnt 0x0
	s_and_not1_b32 exec_lo, exec_lo, s12
	s_cbranch_execnz .LBB0_19
; %bb.20:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s12
	s_and_saveexec_b32 s12, s0
	s_cbranch_execz .LBB0_23
; %bb.21:                               ;   in Loop: Header=BB0_17 Depth=1
	v_add_nc_u64_e32 v[156:157], s[6:7], v[152:153]
	v_mov_b32_e32 v116, v125
	s_mov_b32 s6, 0
.LBB0_22:                               ;   Parent Loop BB0_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v116, 0x100, v116
	flat_store_b32 v[156:157], v117
	s_wait_xcnt 0x0
	v_add_nc_u64_e32 v[156:157], 0x400, v[156:157]
	v_cmp_lt_u32_e32 vcc_lo, 0x1cbf, v116
	s_or_b32 s6, vcc_lo, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s6
	s_cbranch_execnz .LBB0_22
.LBB0_23:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s12
	v_mov_b64_e32 v[156:157], v[0:1]
	v_mov_b32_e32 v129, 16
	s_cmp_lg_u32 s5, 0
	s_mov_b32 s12, 0
	s_cselect_b32 s7, s31, s41
	s_cselect_b32 s6, s30, s40
.LBB0_24:                               ;   Parent Loop BB0_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_dual_mov_b32 v116, v156 :: v_dual_add_nc_u32 v129, -2, v129
	v_dual_mov_b32 v158, v157 :: v_dual_mov_b32 v159, v117
	v_add_nc_u32_e32 v157, 0x200, v157
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u64 v[160:161], v[116:117], 2, s[6:7]
	v_cmp_eq_u32_e32 vcc_lo, 0, v129
	v_add_nc_u32_e32 v156, 0x200, v156
	v_lshl_add_u64 v[158:159], v[158:159], 2, s[6:7]
	s_clause 0x1
	flat_store_b32 v[160:161], v117
	flat_store_b32 v[158:159], v117
	s_or_b32 s12, vcc_lo, s12
	s_wait_xcnt 0x0
	s_and_not1_b32 exec_lo, exec_lo, s12
	s_cbranch_execnz .LBB0_24
; %bb.25:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s12
	s_and_saveexec_b32 s12, s48
	s_cbranch_execz .LBB0_28
; %bb.26:                               ;   in Loop: Header=BB0_17 Depth=1
	v_add_nc_u64_e32 v[156:157], s[6:7], v[154:155]
	v_mov_b32_e32 v116, v127
	s_mov_b32 s6, 0
.LBB0_27:                               ;   Parent Loop BB0_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v116, 0x100, v116
	flat_store_b32 v[156:157], v117
	s_wait_xcnt 0x0
	v_add_nc_u64_e32 v[156:157], 0x400, v[156:157]
	v_cmp_lt_u32_e32 vcc_lo, 0xfff, v116
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
	v_cmpx_lt_i32_e32 0, v121
	s_xor_b32 s7, exec_lo, s7
	s_cbranch_execnz .LBB0_32
; %bb.30:                               ;   in Loop: Header=BB0_17 Depth=1
	s_and_not1_saveexec_b32 s12, s7
	s_cbranch_execnz .LBB0_35
.LBB0_31:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s12
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s39
	s_cbranch_vccnz .LBB0_16
	s_branch .LBB0_36
.LBB0_32:                               ;   in Loop: Header=BB0_17 Depth=1
	s_mov_b32 s51, exec_lo
	v_cmpx_eq_u32_e32 1, v121
	s_cbranch_execz .LBB0_34
; %bb.33:                               ;   in Loop: Header=BB0_17 Depth=1
	s_cmp_lg_u32 s5, 0
	s_mov_b32 s28, s2
	s_cselect_b32 s14, s30, s40
	s_cmp_gt_i32 s2, 0
	s_mov_b32 s17, s9
	s_cselect_b32 s15, -1, 0
	s_lshl_b32 s10, s6, 6
	s_mov_b32 s18, s11
	s_lshl_b64 s[12:13], s[10:11], 1
	s_mov_b32 s19, s11
	s_add_nc_u64 s[12:13], s[20:21], s[12:13]
	v_nop
	v_nop
	v_dual_mov_b32 v129, s14 :: v_dual_mov_b32 v156, s12
	s_and_b32 s10, s13, 0x1ffffff
	s_and_b32 s13, s25, s15
	s_bitset1_b32 s10, 31
	v_cndmask_b32_e64 v116, 0, 1, s13
	v_mov_b32_e32 v131, s10
	v_readfirstlane_b32 s53, v129
	v_readfirstlane_b32 s54, v156
	s_lshr_b64 s[14:15], s[28:29], 16
	v_readfirstlane_b32 s52, v116
	v_readfirstlane_b32 s55, v131
	s_lshl_b32 s13, s2, 16
	s_mov_b32 s12, s4
	s_mov_b32 s15, s46
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[52:55], s[12:19]
.LBB0_34:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s51
	s_and_not1_saveexec_b32 s12, s7
	s_cbranch_execz .LBB0_31
.LBB0_35:                               ;   in Loop: Header=BB0_17 Depth=1
	s_cmp_lg_u32 s5, 0
	s_cselect_b32 s5, s43, 0
	s_cmp_gt_i32 s2, 0
	s_cselect_b32 s13, -1, 0
	s_lshl_b32 s10, s6, 6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_lshl_b64 s[6:7], s[10:11], 1
	s_and_b32 s10, s37, s13
	s_add_nc_u64 s[6:7], s[34:35], s[6:7]
	v_cndmask_b32_e64 v116, 0, 1, s10
	s_and_b32 s7, s7, 0x1ffffff
	v_nop
	v_nop
	v_dual_mov_b32 v129, s5 :: v_dual_mov_b32 v156, s6
	s_bitset1_b32 s7, 31
	v_readfirstlane_b32 s52, v116
	v_mov_b32_e32 v131, s7
	s_delay_alu instid0(VALU_DEP_3)
	v_readfirstlane_b32 s53, v129
	v_readfirstlane_b32 s54, v156
	s_lshr_b64 s[6:7], s[2:3], 16
	s_lshl_b32 s5, s2, 16
	v_readfirstlane_b32 s55, v131
	s_mov_b32 s7, s47
	s_mov_b32 s10, s11
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	tensor_load_to_lds s[52:55], s[4:11]
	s_or_b32 exec_lo, exec_lo, s12
	s_and_not1_b32 vcc_lo, exec_lo, s39
	s_cbranch_vccnz .LBB0_16
.LBB0_36:                               ;   in Loop: Header=BB0_17 Depth=1
	s_cmp_lg_u32 s50, 0
	s_cselect_b32 s2, s43, 0
	s_cselect_b32 s5, s30, s40
	v_lshl_add_u32 v116, v114, 1, s2
	v_lshl_add_u32 v129, v120, 1, s2
	v_lshl_add_u32 v131, v122, 1, s2
	v_lshl_add_u32 v133, v118, 1, s5
	v_lshl_add_u32 v135, v142, 1, s2
	ds_load_b128 v[156:159], v116
	ds_load_b128 v[160:163], v116 offset:16
	v_lshl_add_u32 v116, v124, 1, s2
	ds_load_b128 v[164:167], v129 offset:2048
	ds_load_b128 v[168:171], v129 offset:2064
	v_lshl_add_u32 v129, v126, 1, s2
	ds_load_b128 v[172:175], v131 offset:4096
	ds_load_b128 v[180:183], v116 offset:6144
	ds_load_b128 v[184:187], v116 offset:6160
	v_lshl_add_u32 v116, v128, 1, s2
	ds_load_b128 v[176:179], v131 offset:4112
	ds_load_b128 v[188:191], v129 offset:8192
	v_lshl_add_u32 v131, v138, 1, s2
	v_lshl_add_u32 v137, v144, 1, s2
	ds_load_b128 v[204:207], v116 offset:10240
	ds_load_b128 v[208:211], v116 offset:10256
	v_lshl_add_u32 v116, v132, 1, s5
	v_lshl_add_u32 v139, v146, 1, s2
	v_lshl_add_u32 v141, v148, 1, s5
	v_lshl_add_u32 v143, v150, 1, s5
	; sched_group_barrier mask(0x00000100) size(9) SyncID(0)
	ds_load_b128 v[212:215], v116 offset:2048
	ds_load_b128 v[216:219], v116 offset:2064
	v_lshl_add_u32 v116, v130, 1, s2
	ds_load_b128 v[228:231], v141 offset:64
	ds_load_b128 v[232:235], v141 offset:80
	ds_load_b128 v[236:239], v143 offset:64
	ds_load_b128 v[240:243], v143 offset:80
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[66:73], v[172:179], v[212:219], v[66:73] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[82:89], v[164:171], v[212:219], v[82:89] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[98:105], v[156:163], v[212:219], v[98:105] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[10:17], v[204:211], v[212:219], v[10:17] matrix_b_reuse
	ds_load_b128 v[220:223], v116 offset:12288
	ds_load_b128 v[224:227], v116 offset:12304
	v_lshl_add_u32 v116, v134, 1, s2
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[58:65], v[220:227], v[212:219], v[58:65] matrix_a_reuse
	ds_load_b128 v[192:195], v129 offset:8208
	v_lshl_add_u32 v129, v136, 1, s2
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[26:33], v[188:195], v[212:219], v[26:33] matrix_b_reuse
	ds_load_b128 v[196:199], v133
	ds_load_b128 v[200:203], v133 offset:16
	v_lshl_add_u32 v133, v140, 1, s2
	; sched_group_barrier mask(0x00000008) size(7) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(9) SyncID(0)
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[106:113], v[156:163], v[196:203], v[106:113]
	ds_load_b128 v[156:159], v116 offset:64
	ds_load_b128 v[160:163], v116 offset:80
	v_wmma_f32_16x16x32_bf16 v[90:97], v[164:171], v[196:203], v[90:97] matrix_b_reuse
	ds_load_b128 v[164:167], v129 offset:64
	ds_load_b128 v[168:171], v129 offset:80
	v_wmma_f32_16x16x32_bf16 v[74:81], v[172:179], v[196:203], v[74:81] matrix_b_reuse
	ds_load_b128 v[172:175], v131 offset:64
	ds_load_b128 v[176:179], v131 offset:80
	v_wmma_f32_16x16x32_bf16 v[50:57], v[180:187], v[196:203], v[50:57] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[34:41], v[188:195], v[196:203], v[34:41] matrix_b_reuse
	ds_load_b128 v[188:191], v137 offset:64
	ds_load_b128 v[192:195], v137 offset:80
	v_wmma_f32_16x16x32_bf16 v[18:25], v[204:211], v[196:203], v[18:25] matrix_b_reuse
	ds_load_b128 v[204:207], v135 offset:64
	ds_load_b128 v[208:211], v135 offset:80
	v_wmma_f32_16x16x32_bf16 v[42:49], v[180:187], v[212:219], v[42:49] matrix_b_reuse
	ds_load_b128 v[180:183], v133 offset:64
	ds_load_b128 v[184:187], v133 offset:80
	ds_load_b128 v[212:215], v139 offset:64
	ds_load_b128 v[216:219], v139 offset:80
	; sched_group_barrier mask(0x00000008) size(7) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[2:9], v[220:227], v[196:203], v[2:9] matrix_b_reuse
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[106:113], v[156:163], v[228:235], v[106:113]
	; sched_group_barrier mask(0x00000100) size(9) SyncID(0)
	s_wait_dscnt 0xa
	v_wmma_f32_16x16x32_bf16 v[90:97], v[164:171], v[228:235], v[90:97] matrix_b_reuse
	s_wait_dscnt 0x8
	v_wmma_f32_16x16x32_bf16 v[74:81], v[172:179], v[228:235], v[74:81] matrix_b_reuse
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[50:57], v[180:187], v[228:235], v[50:57] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[34:41], v[204:211], v[228:235], v[34:41] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[18:25], v[188:195], v[228:235], v[18:25] matrix_b_reuse
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[2:9], v[212:219], v[228:235], v[2:9] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(7) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(9) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[58:65], v[212:219], v[236:243], v[58:65] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[10:17], v[188:195], v[236:243], v[10:17] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[26:33], v[204:211], v[236:243], v[26:33] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[42:49], v[180:187], v[236:243], v[42:49] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[66:73], v[172:179], v[236:243], v[66:73] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[82:89], v[164:171], v[236:243], v[82:89] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[98:105], v[156:163], v[236:243], v[98:105] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(7) SyncID(0)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_wait_dscnt 0
	;;#ASMEND
	s_branch .LBB0_16
.LBB0_37:
	s_wait_tensorcnt 0x0
	;;#ASMSTART
	s_wait_dscnt 0
	;;#ASMEND
	s_and_b32 vcc_lo, exec_lo, s39
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_vccz .LBB0_39
; %bb.38:
	v_mul_u32_u24_e32 v1, 0x70, v119
	v_lshrrev_b32_e32 v114, 1, v0
	v_and_b32_e32 v116, 0x6f, v0
	v_nop
	v_cvt_pk_bf16_f32 v105, v104, v105
	v_cvt_pk_bf16_f32 v104, v102, v103
	v_cvt_pk_bf16_f32 v102, v98, v99
	v_and_or_b32 v1, v114, 8, v1
	v_cvt_pk_bf16_f32 v97, v96, v97
	v_cvt_pk_bf16_f32 v96, v94, v95
	v_cvt_pk_bf16_f32 v94, v90, v91
	v_cvt_pk_bf16_f32 v95, v92, v93
	v_mad_u32_u24 v1, 0xe0, v116, v1
	v_cvt_pk_bf16_f32 v113, v112, v113
	v_cvt_pk_bf16_f32 v112, v110, v111
	v_cvt_pk_bf16_f32 v111, v108, v109
	v_cvt_pk_bf16_f32 v110, v106, v107
	v_add_nc_u32_e32 v98, 0xe00, v1
	v_dual_lshrrev_b32 v90, 3, v1 :: v_dual_lshlrev_b32 v93, 1, v1
	v_add_nc_u32_e32 v92, 16, v1
	v_cvt_pk_bf16_f32 v73, v72, v73
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshrrev_b32_e32 v91, 3, v98
	v_and_b32_e32 v90, 0x1ff0, v90
	v_cvt_pk_bf16_f32 v72, v70, v71
	v_cvt_pk_bf16_f32 v71, v68, v69
	v_add_nc_u32_e32 v68, 0xe30, v1
	v_and_b32_e32 v91, 0x3ff0, v91
	v_add_nc_u32_e32 v90, v90, v93
	v_cvt_pk_bf16_f32 v103, v100, v101
	v_cvt_pk_bf16_f32 v41, v40, v41
	v_cvt_pk_bf16_f32 v40, v38, v39
	v_add_nc_u32_e32 v91, v91, v93
	v_cvt_pk_bf16_f32 v39, v36, v37
	v_add_nc_u32_e32 v36, 0x50, v1
	v_lshrrev_b32_e32 v92, 3, v92
	v_add_nc_u32_e32 v99, 0xe10, v1
	v_add_nc_u32_e32 v100, 32, v1
	ds_store_b128 v90, v[110:113]
	ds_store_b128 v91, v[102:105] offset:7168
	v_add_nc_u32_e32 v91, 0xe20, v1
	v_cvt_pk_bf16_f32 v81, v80, v81
	v_cvt_pk_bf16_f32 v80, v78, v79
	v_cvt_pk_bf16_f32 v79, v76, v77
	v_add_nc_u32_e32 v76, 48, v1
	v_cvt_pk_bf16_f32 v70, v66, v67
	v_dual_lshrrev_b32 v67, 3, v68 :: v_dual_add_nc_u32 v68, 64, v1
	v_cvt_pk_bf16_f32 v57, v56, v57
	v_cvt_pk_bf16_f32 v56, v54, v55
	v_cvt_pk_bf16_f32 v55, v52, v53
	v_add_nc_u32_e32 v52, 0xe40, v1
	v_cvt_pk_bf16_f32 v38, v34, v35
	v_lshrrev_b32_e32 v35, 3, v36
	v_add_nc_u32_e32 v36, 0xe50, v1
	v_cvt_pk_bf16_f32 v33, v32, v33
	v_cvt_pk_bf16_f32 v32, v30, v31
	v_cvt_pk_bf16_f32 v31, v28, v29
	v_add_nc_u32_e32 v28, 0x60, v1
	v_add_nc_u32_e32 v1, 0xe60, v1
	v_and_b32_e32 v92, 0x3ff0, v92
	v_dual_lshrrev_b32 v99, 3, v99 :: v_dual_lshrrev_b32 v90, 3, v100
	v_lshrrev_b32_e32 v91, 3, v91
	v_cvt_pk_bf16_f32 v78, v74, v75
	v_lshrrev_b32_e32 v75, 3, v76
	v_lshrrev_b32_e32 v68, 3, v68
	v_cvt_pk_bf16_f32 v49, v48, v49
	v_cvt_pk_bf16_f32 v48, v46, v47
	v_cvt_pk_bf16_f32 v46, v42, v43
	v_lshrrev_b32_e32 v43, 3, v52
	v_cvt_pk_bf16_f32 v30, v26, v27
	v_lshrrev_b32_e32 v26, 3, v36
	v_cvt_pk_bf16_f32 v25, v24, v25
	v_cvt_pk_bf16_f32 v24, v22, v23
	v_cvt_pk_bf16_f32 v23, v20, v21
	v_dual_lshrrev_b32 v20, 3, v28 :: v_dual_lshrrev_b32 v1, 3, v1
	v_dual_lshlrev_b32 v98, 1, v98 :: v_dual_add_nc_u32 v92, v92, v93
	v_and_b32_e32 v99, 0x3ff0, v99
	v_cvt_pk_bf16_f32 v89, v88, v89
	v_and_b32_e32 v90, 0x3ff0, v90
	v_cvt_pk_bf16_f32 v88, v86, v87
	v_cvt_pk_bf16_f32 v86, v82, v83
	v_and_b32_e32 v83, 0x3ff0, v91
	v_and_b32_e32 v66, 0x3ff0, v75
	v_cvt_pk_bf16_f32 v54, v50, v51
	v_and_b32_e32 v51, 0x3ff0, v68
	v_and_b32_e32 v34, 0x3ff0, v43
	v_and_b32_e32 v67, 0x3ff0, v67
	v_and_b32_e32 v35, 0x3ff0, v35
	v_and_b32_e32 v26, 0x3ff0, v26
	v_cvt_pk_bf16_f32 v22, v18, v19
	v_and_b32_e32 v19, 0x3ff0, v20
	v_and_b32_e32 v1, 0x3ff0, v1
	ds_store_b128 v92, v[94:97] offset:32
	v_add_nc_u32_e32 v92, v99, v98
	v_cvt_pk_bf16_f32 v87, v84, v85
	v_dual_add_nc_u32 v82, v90, v93 :: v_dual_add_nc_u32 v74, v83, v98
	v_dual_add_nc_u32 v66, v66, v93 :: v_dual_add_nc_u32 v42, v51, v93
	v_dual_add_nc_u32 v34, v34, v98 :: v_dual_add_nc_u32 v50, v67, v98
	v_cvt_pk_bf16_f32 v47, v44, v45
	v_dual_add_nc_u32 v27, v35, v93 :: v_dual_add_nc_u32 v18, v26, v98
	v_cvt_pk_bf16_f32 v17, v16, v17
	v_cvt_pk_bf16_f32 v16, v14, v15
	v_cvt_pk_bf16_f32 v15, v12, v13
	v_cvt_pk_bf16_f32 v14, v10, v11
	v_add_nc_u32_e32 v10, v19, v93
	v_cvt_pk_bf16_f32 v9, v8, v9
	v_cvt_pk_bf16_f32 v8, v6, v7
	v_cvt_pk_bf16_f32 v7, v4, v5
	v_cvt_pk_bf16_f32 v6, v2, v3
	v_add_nc_u32_e32 v1, v1, v98
	v_cvt_pk_bf16_f32 v5, v64, v65
	v_cvt_pk_bf16_f32 v4, v62, v63
	v_cvt_pk_bf16_f32 v3, v60, v61
	v_cvt_pk_bf16_f32 v2, v58, v59
	ds_store_b128 v92, v[86:89] offset:32
	ds_store_b128 v82, v[78:81] offset:64
	ds_store_b128 v74, v[70:73] offset:64
	ds_store_b128 v66, v[54:57] offset:96
	ds_store_b128 v50, v[46:49] offset:96
	ds_store_b128 v42, v[38:41] offset:128
	ds_store_b128 v34, v[30:33] offset:128
	ds_store_b128 v27, v[22:25] offset:160
	ds_store_b128 v18, v[14:17] offset:160
	ds_store_b128 v10, v[6:9] offset:192
	ds_store_b128 v1, v[2:5] offset:192
.LBB0_39:
	v_cmp_ne_u32_e32 vcc_lo, 1, v115
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
	.size	bm224_bn128_bk064_wm2_wn4_mc0, .Lfunc_end0-bm224_bn128_bk064_wm2_wn4_mc0
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel bm224_bn128_bk064_wm2_wn4_mc0
		.amdhsa_group_segment_fixed_size 95744
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
		.amdhsa_next_free_vgpr 244
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
		.amdhsa_inst_pref_size 56
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text.bm224_bn128_bk064_wm2_wn4_mc0,"axG",@progbits,bm224_bn128_bk064_wm2_wn4_mc0,comdat
                                        ; -- End function
	.set .Lbm224_bn128_bk064_wm2_wn4_mc0.num_vgpr, 244
	.set .Lbm224_bn128_bk064_wm2_wn4_mc0.num_agpr, 0
	.set .Lbm224_bn128_bk064_wm2_wn4_mc0.numbered_sgpr, 56
	.set .Lbm224_bn128_bk064_wm2_wn4_mc0.num_named_barrier, 0
	.set .Lbm224_bn128_bk064_wm2_wn4_mc0.private_seg_size, 0
	.set .Lbm224_bn128_bk064_wm2_wn4_mc0.uses_vcc, 1
	.set .Lbm224_bn128_bk064_wm2_wn4_mc0.uses_flat_scratch, 1
	.set .Lbm224_bn128_bk064_wm2_wn4_mc0.has_dyn_sized_stack, 0
	.set .Lbm224_bn128_bk064_wm2_wn4_mc0.has_recursion, 0
	.set .Lbm224_bn128_bk064_wm2_wn4_mc0.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 7096
; TotalNumSgprs: 58
; NumVgprs: 244
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 95744 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 15
; NumSGPRsForWavesPerEU: 58
; NumVGPRsForWavesPerEU: 244
; NamedBarCnt: 0
; Occupancy: 4
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
	.type	__hip_cuid_22058d6de2dafa6e,@object ; @__hip_cuid_22058d6de2dafa6e
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_22058d6de2dafa6e
__hip_cuid_22058d6de2dafa6e:
	.byte	0                               ; 0x0
	.size	__hip_cuid_22058d6de2dafa6e, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git aa451e1fe6a793394d6733051b1778633063ae96+PATCHED:d17c5aa0e3ea29cde402f58f27e39b6034effa27)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_22058d6de2dafa6e
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
    .group_segment_fixed_size: 95744
    .kernarg_segment_align: 8
    .kernarg_segment_size: 132
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           bm224_bn128_bk064_wm2_wn4_mc0
    .private_segment_fixed_size: 0
    .sgpr_count:     58
    .sgpr_spill_count: 0
    .symbol:         bm224_bn128_bk064_wm2_wn4_mc0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     244
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
