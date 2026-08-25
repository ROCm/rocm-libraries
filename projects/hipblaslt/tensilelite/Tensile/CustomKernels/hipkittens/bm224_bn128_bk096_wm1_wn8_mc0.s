	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.section	.text.bm224_bn128_bk096_wm1_wn8_mc0,"axG",@progbits,bm224_bn128_bk096_wm1_wn8_mc0,comdat
	.protected	bm224_bn128_bk096_wm1_wn8_mc0 ; -- Begin function bm224_bn128_bk096_wm1_wn8_mc0
	.globl	bm224_bn128_bk096_wm1_wn8_mc0
	.p2align	8
	.type	bm224_bn128_bk096_wm1_wn8_mc0,@function
bm224_bn128_bk096_wm1_wn8_mc0: ; @bm224_bn128_bk096_wm1_wn8_mc0
; %bb.0:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1 ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_mov_b64 s[2:3], src_shared_base
	s_mov_b32 s2, 0xb280
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
	s_cselect_b32 s38, ttmp9, s3
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
	s_mul_i32 s7, s38, 0xffffff20
	s_add_co_i32 s2, s2, s3
	s_add_co_i32 s3, s24, s7
	s_ashr_i32 s7, s2, 7
	s_min_i32 s27, s3, 0xe0
	s_cmp_lt_i32 s38, s6
	s_cselect_b32 s39, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_and_b32 s2, s39, exec_lo
	s_cselect_b32 s3, s27, 0
	s_lshl_b32 s33, s5, 7
	s_sub_co_i32 s2, s25, s33
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_min_i32 s2, s2, 0x80
	s_cmp_lt_i32 s5, s7
	s_cselect_b32 s25, -1, 0
	s_and_b32 s8, s25, exec_lo
	s_cselect_b32 s29, s2, 0
	s_add_co_i32 s16, s26, 0x5f
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 2, 1), 1
	s_min_i32 s2, s26, 0x60
	s_cmp_gt_i32 s16, 0x5f
	s_cselect_b32 s17, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_and_b32 s8, s17, exec_lo
	s_cselect_b32 s2, s2, 0
	s_cmp_lt_i32 s3, 0xe0
	s_cselect_b32 s40, -1, 0
	s_and_b32 vcc_lo, exec_lo, s40
	s_mov_b32 s8, s40
	s_cbranch_vccnz .LBB0_2
; %bb.1:
	s_cmp_lt_i32 s29, 0x80
	s_cselect_b32 s8, -1, 0
	s_cmp_lt_i32 s2, 0x60
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
	v_cmp_lt_u32_e32 vcc_lo, 0x2b9f, v5
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
	ds_store_b32 v2, v3 offset:45696
	v_add_nc_u32_e32 v2, 0x400, v2
	v_cmp_lt_u32_e32 vcc_lo, 0x187f, v1
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
	v_lshrrev_b32_e32 v123, 5, v0
	s_wait_xcnt 0x0
	s_lshl_b32 s0, s4, 2
	s_add_co_i32 s7, s7, -1
	s_mov_b64 s[30:31], src_shared_base
	s_or_b32 s41, s0, 0xb280
	s_add_co_i32 s0, s6, -1
	s_min_i32 s36, s5, s7
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_i32_e32 0, v123
	s_xor_b32 s1, exec_lo, s1
	s_cbranch_execz .LBB0_12
; %bb.9:
	s_mov_b32 s30, exec_lo
	v_cmpx_eq_u32_e32 1, v123
	s_cbranch_execz .LBB0_11
; %bb.10:
	s_cmp_gt_i32 s2, 0
	s_mov_b32 s28, s2
	s_cselect_b32 s8, -1, 0
	s_lshl_b32 s4, s36, 7
	s_wait_kmcnt 0x0
	s_bfe_i64 s[6:7], s[20:21], 0x200000
	s_ashr_i32 s5, s4, 31
	s_mov_b32 s10, 0
	s_mul_u64 s[4:5], s[6:7], s[4:5]
	s_mov_b32 s11, s10
	s_lshl_b64 s[4:5], s[4:5], 1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_nc_u64 s[6:7], s[14:15], s[4:5]
	v_dual_mov_b32 v1, s41 :: v_dual_mov_b32 v4, s6
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
	s_or_b32 s7, s4, 0x600000
	s_movk_i32 s8, 0x80
	s_mov_b32 s4, 0x7510000
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[44:47], s[4:11]
.LBB0_11:
	s_or_b32 exec_lo, exec_lo, s30
.LBB0_12:
	s_or_saveexec_b32 s28, s1
	s_min_i32 s0, s38, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mulk_i32 s0, 0xe0
	s_xor_b32 exec_lo, exec_lo, s28
	s_cbranch_execz .LBB0_14
; %bb.13:
	s_cmp_gt_i32 s2, 0
	s_mov_b32 s10, 0
	s_cselect_b32 s6, -1, 0
	s_ashr_i32 s1, s0, 31
	s_wait_kmcnt 0x0
	s_bfe_i64 s[4:5], s[12:13], 0x200000
	s_movk_i32 s8, 0xe0
	s_mul_u64 s[4:5], s[4:5], s[0:1]
	s_mov_b32 s11, s10
	s_lshl_b64 s[4:5], s[4:5], 1
	s_mov_b32 s45, s10
	s_add_nc_u64 s[4:5], s[18:19], s[4:5]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s1, s5, 0x1ffffff
	s_and_b32 s5, s39, s6
	s_bitset1_b32 s1, 31
	v_cndmask_b32_e64 v2, 0, 1, s5
	v_dual_mov_b32 v4, s4 :: v_dual_mov_b32 v3, s1
	s_lshr_b32 s1, s3, 16
	s_lshr_b64 s[6:7], s[2:3], 16
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_readfirstlane_b32 s44, v2
	v_readfirstlane_b32 s46, v4
	v_readfirstlane_b32 s47, v3
	s_lshl_b32 s5, s2, 16
	s_or_b32 s7, s1, 0x600000
	s_mov_b32 s4, 0x7510000
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[44:47], s[4:11]
.LBB0_14:
	s_or_b32 exec_lo, exec_lo, s28
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	v_dual_lshlrev_b32 v119, 4, v123 :: v_dual_mov_b32 v9, 0
	s_and_b32 s30, s39, s25
	v_and_b32_e32 v121, 15, v0
	v_cndmask_b32_e64 v117, 0, 1, s30
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
	v_dual_mov_b32 v115, 0 :: v_dual_bitop2_b32 v37, 16, v0 bitop3:0x40
	v_mul_u32_u24_e64 v3, 0x5400, 0
	v_mul_u32_u24_e32 v5, 0x60, v121
	v_mul_u32_u24_e32 v1, 0x60, v119
	s_mov_b64 s[4:5], src_shared_base
	v_dual_mov_b32 v61, v115 :: v_dual_mov_b32 v63, v115
	s_delay_alu instid0(VALU_DEP_3)
	v_or3_b32 v114, v3, v37, v5
	v_or_b32_e32 v7, 0x1800, v5
	v_or_b32_e32 v17, 0x3000, v5
	v_or_b32_e32 v29, 0x4800, v5
	v_mad_u32_u24 v39, 0x60, v119, v5
	v_add_nc_u32_e32 v4, 0x600, v114
	v_lshrrev_b32_e32 v2, 4, v5
	v_add_nc_u32_e32 v6, 0xc00, v114
	v_add_nc_u32_e32 v8, 0x1e00, v114
	v_add_nc_u32_e32 v10, 0x2a00, v114
	v_lshrrev_b32_e32 v4, 4, v4
	v_and_b32_e32 v2, 0x78, v2
	v_lshrrev_b32_e32 v6, 4, v6
	v_or_b32_e32 v64, 0x3000, v114
	s_add_co_i32 s6, s41, 0x6600
	v_and_b32_e32 v9, 0xf8, v4
	v_dual_add_nc_u32 v116, v2, v114 :: v_dual_lshrrev_b32 v4, 4, v7
	v_add_nc_u32_e32 v2, 0x1200, v114
	v_and_b32_e32 v11, 0x1f8, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v9, v115 :: v_dual_add_nc_u32 v120, v9, v114
	v_dual_lshrrev_b32 v6, 4, v8 :: v_dual_lshrrev_b32 v2, 4, v2
	v_add_nc_u32_e32 v8, 0x2400, v114
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_nc_u32_e32 v122, v11, v114
	v_mov_b32_e32 v11, v115
	v_and_b32_e32 v15, 0x3f8, v6
	v_and_b32_e32 v13, 0x1f8, v2
	v_and_b32_e32 v2, 0x1f8, v4
	v_lshrrev_b32_e32 v4, 4, v8
	v_add_nc_u32_e32 v8, 0x3600, v114
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_lshrrev_b32 v12, 4, v17 :: v_dual_add_nc_u32 v124, v13, v114
	v_add_nc_u32_e32 v128, v15, v114
	v_dual_mov_b32 v13, v115 :: v_dual_lshrrev_b32 v6, 4, v10
	v_add_nc_u32_e32 v10, 0x3c00, v114
	v_and_b32_e32 v19, 0x2f8, v4
	v_mov_b32_e32 v15, v115
	v_and_b32_e32 v4, 0x378, v12
	v_and_b32_e32 v21, 0x3f8, v6
	v_dual_lshrrev_b32 v6, 4, v8 :: v_dual_lshrrev_b32 v8, 4, v10
	v_add_nc_u32_e32 v130, v19, v114
	v_dual_mov_b32 v19, v115 :: v_dual_add_nc_u32 v10, 0x4e00, v114
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_and_b32_e32 v25, 0x3f8, v6
	v_add_nc_u32_e32 v6, 0x4200, v114
	v_or_b32_e32 v1, v1, v37
	v_dual_mov_b32 v21, v115 :: v_dual_add_nc_u32 v132, v21, v114
	v_dual_lshrrev_b32 v10, 4, v10 :: v_dual_bitop2_b32 v23, 32, v37 bitop3:0x54
	v_lshrrev_b32_e32 v6, 4, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_mad_u32_u24 v1, 0x60, v121, v1
	v_and_b32_e32 v27, 0x7f8, v8
	v_or_b32_e32 v8, v23, v3
	v_and_b32_e32 v33, 0x5f8, v10
	v_and_b32_e32 v31, 0x4f8, v6
	v_dual_lshrrev_b32 v12, 4, v1 :: v_dual_add_nc_u32 v136, v25, v114
	v_add_nc_u32_e32 v138, v27, v114
	v_mad_u32_u24 v30, 0x60, v121, v8
	v_lshrrev_b32_e32 v8, 4, v29
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_and_b32_e32 v12, 0x7f8, v12
	v_dual_mov_b32 v25, v115 :: v_dual_mov_b32 v65, v115
	v_add_nc_u32_e32 v16, 0x1e00, v30
	v_lshrrev_b32_e32 v14, 4, v30
	v_add_nc_u32_e32 v10, 0x600, v30
	v_add_nc_u32_e32 v118, v12, v1
	v_add_nc_u32_e32 v1, 0xc00, v30
	v_and_b32_e32 v6, 0x4f8, v8
	v_and_b32_e32 v8, 0xf8, v14
	v_add_nc_u32_e32 v12, 0x1200, v30
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_add_nc_u32 v14, v23, v7 :: v_dual_lshrrev_b32 v1, 4, v1
	v_add_nc_u32_e32 v24, 0x3600, v30
	v_add_nc_u32_e32 v26, 0x3c00, v30
	v_dual_lshrrev_b32 v18, 4, v12 :: v_dual_lshrrev_b32 v20, 4, v14
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_and_b32_e32 v12, 0x1f8, v1
	v_add_nc_u32_e32 v1, 0x2400, v30
	v_lshrrev_b32_e32 v22, 4, v16
	v_and_b32_e32 v14, 0x3f8, v18
	v_and_b32_e32 v16, 0x3f8, v20
	v_dual_mov_b32 v27, v115 :: v_dual_add_nc_u32 v20, 0x2a00, v30
	v_lshrrev_b32_e32 v1, 4, v1
	v_and_b32_e32 v18, 0x3f8, v22
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_dual_add_nc_u32 v22, v23, v17 :: v_dual_lshrrev_b32 v28, 4, v20
	v_or_b32_e32 v35, 64, v37
	v_and_b32_e32 v20, 0x3f8, v1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_lshrrev_b32 v1, 4, v24 :: v_dual_lshrrev_b32 v32, 4, v22
	v_dual_add_nc_u32 v140, v31, v114 :: v_dual_add_nc_u32 v144, v33, v114
	v_dual_mov_b32 v31, v115 :: v_dual_lshrrev_b32 v34, 4, v26
	v_and_b32_e32 v26, 0x7f8, v1
	v_add_nc_u32_e32 v1, 0x4200, v30
	v_or_b32_e32 v3, v3, v35
	v_and_b32_e32 v24, 0x3f8, v32
	v_dual_mov_b32 v33, v115 :: v_dual_add_nc_u32 v30, 0x4e00, v30
	v_add_nc_u32_e32 v32, v23, v29
	v_lshrrev_b32_e32 v1, 4, v1
	v_mad_u32_u24 v3, 0x60, v121, v3
	v_and_b32_e32 v22, 0x3f8, v28
	v_and_b32_e32 v28, 0x7f8, v34
	v_dual_lshrrev_b32 v34, 4, v30 :: v_dual_lshrrev_b32 v5, 4, v32
	v_add_nc_u32_e32 v23, v39, v23
	v_and_b32_e32 v30, 0x5f8, v1
	v_add_nc_u32_e32 v1, 0x600, v3
	v_add_nc_u32_e32 v38, 0xc00, v3
	v_and_b32_e32 v32, 0x5f8, v5
	v_lshrrev_b32_e32 v5, 4, v23
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_dual_lshrrev_b32 v23, 4, v3 :: v_dual_lshrrev_b32 v1, 4, v1
	v_add_nc_u32_e32 v40, 0x1200, v3
	v_add_nc_u32_e32 v7, v35, v7
	v_and_b32_e32 v45, 0x7f8, v5
	v_lshrrev_b32_e32 v5, 4, v38
	v_and_b32_e32 v36, 0xf8, v23
	v_and_b32_e32 v38, 0x1f8, v1
	v_lshrrev_b32_e32 v1, 4, v40
	v_add_nc_u32_e32 v23, 0x1e00, v3
	v_and_b32_e32 v40, 0x1f8, v5
	v_add_nc_u32_e32 v5, 0x2400, v3
	v_add_nc_u32_e32 v17, v35, v17
	v_and_b32_e32 v42, 0x3f8, v1
	v_lshrrev_b32_e32 v1, 4, v7
	v_lshrrev_b32_e32 v7, 4, v23
	v_add_nc_u32_e32 v23, 0x2a00, v3
	v_lshrrev_b32_e32 v5, 4, v5
	v_add_nc_u64_e32 v[162:163], v[24:25], v[64:65]
	v_and_b32_e32 v44, 0x3f8, v1
	v_and_b32_e32 v46, 0x3f8, v7
	v_lshrrev_b32_e32 v1, 4, v23
	v_and_b32_e32 v48, 0x3f8, v5
	v_lshrrev_b32_e32 v5, 4, v17
	v_add_nc_u32_e32 v7, 0x3600, v3
	v_add_nc_u32_e32 v17, 0x3c00, v3
	v_and_b32_e32 v50, 0x3f8, v1
	v_add_nc_u32_e32 v1, 0x4200, v3
	v_and_b32_e32 v52, 0x3f8, v5
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_dual_lshrrev_b32 v5, 4, v7 :: v_dual_lshrrev_b32 v7, 4, v17
	v_add_nc_u32_e32 v3, 0x4e00, v3
	v_lshrrev_b32_e32 v1, 4, v1
	v_add_nc_u32_e32 v17, v35, v29
	v_add_nc_u64_e32 v[24:25], 0x4e00, v[114:115]
	v_and_b32_e32 v56, 0x7f8, v7
	v_add_nc_u32_e32 v7, v39, v35
	v_and_b32_e32 v58, 0x5f8, v1
	v_lshrrev_b32_e32 v1, 4, v3
	v_and_b32_e32 v34, 0x5f8, v34
	v_mov_b32_e32 v35, v115
	s_mov_b32 s7, s5
	v_and_b32_e32 v54, 0x7f8, v5
	v_and_b32_e32 v62, 0x5f8, v1
	s_and_b64 s[6:7], s[6:7], 15
	v_sub_nc_u32_e32 v3, 0x2c9f, v0
	s_sub_co_i32 s1, 16, s6
	v_add_nc_u64_e32 v[146:147], v[8:9], v[114:115]
	v_or_b32_e32 v8, 0x1800, v114
	v_add_nc_u64_e32 v[172:173], v[34:35], v[24:25]
	v_add_nc_u64_e32 v[202:203], v[62:63], v[24:25]
	v_dual_mov_b32 v24, v115 :: v_dual_lshrrev_b32 v5, 4, v17
	v_dual_lshrrev_b32 v1, 8, v3 :: v_dual_mov_b32 v17, v115
	s_lshr_b32 s1, s1, 2
	s_cmp_lg_u64 s[6:7], 0
	s_delay_alu instid0(VALU_DEP_2)
	v_and_b32_e32 v60, 0x5f8, v5
	s_cselect_b32 s1, s1, 0
	v_add_nc_u64_e32 v[154:155], v[16:17], v[8:9]
	v_add_nc_u64_e32 v[16:17], 0x3600, v[114:115]
	v_sub_nc_u32_e32 v5, 0x197f, v0
	v_lshrrev_b32_e32 v3, 4, v7
	s_lshl2_add_u32 s1, s1, s41
	s_mov_b32 s11, 0
	s_add_co_i32 s4, s1, 0x11880
	v_add_nc_u64_e32 v[164:165], v[26:27], v[16:17]
	s_and_b32 s10, s4, 15
	v_and_b32_e32 v66, 0x7f8, v3
	v_or_b32_e32 v26, v39, v37
	s_sub_co_i32 s6, 16, s10
	s_add_co_i32 s43, s1, 0x6600
	s_lshr_b32 s1, s6, 2
	s_mul_hi_i32 s2, s16, 0x2aaaaaab
	s_cmp_lg_u64 s[10:11], 0
	v_dual_mov_b32 v43, v115 :: v_dual_add_nc_u32 v174, v45, v26
	v_dual_mov_b32 v207, v115 :: v_dual_add_nc_u32 v204, v66, v26
	v_dual_mov_b32 v26, v115 :: v_dual_lshrrev_b32 v5, 8, v5
	s_cselect_b32 s1, s1, 0
	s_lshr_b32 s6, s2, 31
	s_ashr_i32 s45, s2, 4
	s_lshl_b32 s10, s1, 2
	s_add_co_i32 s45, s45, s6
	s_cmp_lt_i32 s29, 0x80
	v_dual_add_nc_u32 v7, 1, v1 :: v_dual_mov_b32 v39, v115
	v_add_nc_u32_e32 v5, 1, v5
	s_add_nc_u64 s[34:35], s[4:5], s[10:11]
	s_cselect_b32 s46, -1, 0
	s_lshl_b32 s4, s36, 7
	s_mov_b32 s44, s5
	s_ashr_i32 s5, s4, 31
	s_wait_kmcnt 0x0
	s_bfe_i64 s[6:7], s[20:21], 0x200000
	v_and_b32_e32 v125, 62, v7
	v_dual_mov_b32 v41, v115 :: v_dual_bitop2_b32 v129, 26, v5 bitop3:0x40
	s_ashr_i32 s1, s0, 31
	s_mul_u64 s[4:5], s[6:7], s[4:5]
	s_bfe_i64 s[6:7], s[12:13], 0x200000
	s_lshl_b64 s[4:5], s[4:5], 1
	s_mul_u64 s[0:1], s[6:7], s[0:1]
	s_add_nc_u64 s[20:21], s[14:15], s[4:5]
	s_lshl_b64 s[4:5], s[0:1], 1
	v_mov_b32_e32 v3, v115
	v_cmp_ne_u32_e64 s1, v5, v129
	v_mov_b32_e32 v5, v115
	v_cmp_ne_u32_e64 s0, v7, v125
	v_dual_mov_b32 v7, v115 :: v_dual_mov_b32 v23, v115
	v_lshrrev_b32_e32 v10, 4, v10
	v_add_nc_u64_e32 v[126:127], v[2:3], v[114:115]
	v_add_nc_u64_e32 v[134:135], v[4:5], v[114:115]
	v_add_nc_u64_e32 v[2:3], 0x600, v[114:115]
	v_add_nc_u64_e32 v[4:5], 0xc00, v[114:115]
	v_and_b32_e32 v10, 0x1f8, v10
	v_add_nc_u64_e32 v[142:143], v[6:7], v[114:115]
	v_add_nc_u64_e32 v[6:7], 0x1200, v[114:115]
	v_dual_mov_b32 v29, v115 :: v_dual_mov_b32 v37, v115
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u64_e32 v[148:149], v[10:11], v[2:3]
	v_add_nc_u64_e32 v[150:151], v[12:13], v[4:5]
	v_add_nc_u64_e32 v[10:11], 0x1e00, v[114:115]
	v_add_nc_u64_e32 v[12:13], 0x2400, v[114:115]
	v_add_nc_u64_e32 v[152:153], v[14:15], v[6:7]
	v_add_nc_u64_e32 v[14:15], 0x2a00, v[114:115]
	v_dual_mov_b32 v45, v115 :: v_dual_mov_b32 v47, v115
	v_add_nc_u64_e32 v[156:157], v[18:19], v[10:11]
	v_add_nc_u64_e32 v[158:159], v[20:21], v[12:13]
	v_add_nc_u64_e32 v[18:19], 0x3c00, v[114:115]
	v_add_nc_u64_e32 v[20:21], 0x4200, v[114:115]
	v_add_nc_u64_e32 v[160:161], v[22:23], v[14:15]
	v_or_b32_e32 v22, 0x4800, v114
	v_dual_mov_b32 v49, v115 :: v_dual_mov_b32 v51, v115
	v_dual_mov_b32 v53, v115 :: v_dual_mov_b32 v55, v115
	v_dual_mov_b32 v57, v115 :: v_dual_mov_b32 v59, v115
	v_lshl_or_b32 v67, v125, 8, v0
	v_lshl_or_b32 v68, v129, 8, v0
	v_add_nc_u64_e32 v[166:167], v[28:29], v[18:19]
	v_add_nc_u64_e32 v[168:169], v[30:31], v[20:21]
	v_add_nc_u64_e32 v[170:171], v[32:33], v[22:23]
	v_add_nc_u64_e32 v[176:177], v[36:37], v[114:115]
	v_add_nc_u64_e32 v[178:179], v[38:39], v[2:3]
	v_add_nc_u64_e32 v[180:181], v[40:41], v[4:5]
	v_add_nc_u64_e32 v[182:183], v[42:43], v[6:7]
	v_add_nc_u64_e32 v[184:185], v[44:45], v[8:9]
	v_add_nc_u64_e32 v[186:187], v[46:47], v[10:11]
	v_add_nc_u64_e32 v[188:189], v[48:49], v[12:13]
	v_add_nc_u64_e32 v[190:191], v[50:51], v[14:15]
	v_add_nc_u64_e32 v[192:193], v[52:53], v[64:65]
	v_add_nc_u64_e32 v[194:195], v[54:55], v[16:17]
	v_add_nc_u64_e32 v[196:197], v[56:57], v[18:19]
	v_add_nc_u64_e32 v[198:199], v[58:59], v[20:21]
	v_add_nc_u64_e32 v[200:201], v[60:61], v[22:23]
	v_or_b32_e32 v1, 0x100, v0
	v_dual_mov_b32 v28, v115 :: v_dual_add_nc_u32 v127, 0xffffff00, v67
	v_lshlrev_b32_e32 v206, 2, v67
	v_dual_mov_b32 v30, v115 :: v_dual_add_nc_u32 v131, 0xffffff00, v68
	v_dual_mov_b32 v209, v115 :: v_dual_lshlrev_b32 v208, 2, v68
	v_dual_mov_b32 v2, v115 :: v_dual_mov_b32 v3, v115
	v_dual_mov_b32 v5, v115 :: v_dual_mov_b32 v4, v115
	v_dual_mov_b32 v6, v115 :: v_dual_mov_b32 v7, v115
	v_dual_mov_b32 v11, v115 :: v_dual_mov_b32 v8, v115
	v_dual_mov_b32 v10, v115 :: v_dual_mov_b32 v12, v115
	v_dual_mov_b32 v13, v115 :: v_dual_mov_b32 v15, v115
	v_dual_mov_b32 v14, v115 :: v_dual_mov_b32 v16, v115
	v_dual_mov_b32 v17, v115 :: v_dual_mov_b32 v19, v115
	v_dual_mov_b32 v18, v115 :: v_dual_mov_b32 v20, v115
	v_dual_mov_b32 v21, v115 :: v_dual_mov_b32 v25, v115
	v_dual_mov_b32 v22, v115 :: v_dual_mov_b32 v32, v115
	v_dual_mov_b32 v34, v115 :: v_dual_mov_b32 v36, v115
	v_dual_mov_b32 v38, v115 :: v_dual_mov_b32 v40, v115
	v_dual_mov_b32 v42, v115 :: v_dual_mov_b32 v44, v115
	v_dual_mov_b32 v46, v115 :: v_dual_mov_b32 v48, v115
	v_dual_mov_b32 v50, v115 :: v_dual_mov_b32 v52, v115
	v_dual_mov_b32 v54, v115 :: v_dual_mov_b32 v56, v115
	v_dual_mov_b32 v66, v115 :: v_dual_mov_b32 v67, v115
	v_dual_mov_b32 v68, v115 :: v_dual_mov_b32 v69, v115
	v_dual_mov_b32 v70, v115 :: v_dual_mov_b32 v71, v115
	v_dual_mov_b32 v72, v115 :: v_dual_mov_b32 v73, v115
	v_dual_mov_b32 v74, v115 :: v_dual_mov_b32 v75, v115
	v_dual_mov_b32 v76, v115 :: v_dual_mov_b32 v77, v115
	v_dual_mov_b32 v78, v115 :: v_dual_mov_b32 v79, v115
	v_dual_mov_b32 v80, v115 :: v_dual_mov_b32 v81, v115
	v_dual_mov_b32 v82, v115 :: v_dual_mov_b32 v83, v115
	v_dual_mov_b32 v84, v115 :: v_dual_mov_b32 v85, v115
	v_dual_mov_b32 v86, v115 :: v_dual_mov_b32 v87, v115
	v_dual_mov_b32 v88, v115 :: v_dual_mov_b32 v89, v115
	v_dual_mov_b32 v90, v115 :: v_dual_mov_b32 v91, v115
	v_dual_mov_b32 v92, v115 :: v_dual_mov_b32 v93, v115
	v_dual_mov_b32 v94, v115 :: v_dual_mov_b32 v95, v115
	v_dual_mov_b32 v96, v115 :: v_dual_mov_b32 v97, v115
	v_dual_mov_b32 v98, v115 :: v_dual_mov_b32 v99, v115
	v_dual_mov_b32 v100, v115 :: v_dual_mov_b32 v101, v115
	v_dual_mov_b32 v102, v115 :: v_dual_mov_b32 v103, v115
	v_dual_mov_b32 v104, v115 :: v_dual_mov_b32 v105, v115
	v_dual_mov_b32 v106, v115 :: v_dual_mov_b32 v107, v115
	v_dual_mov_b32 v108, v115 :: v_dual_mov_b32 v109, v115
	v_dual_mov_b32 v110, v115 :: v_dual_mov_b32 v111, v115
	v_dual_mov_b32 v112, v115 :: v_dual_mov_b32 v113, v115
	v_dual_mov_b32 v58, v115 :: v_dual_mov_b32 v60, v115
	v_dual_mov_b32 v62, v115 :: v_dual_mov_b32 v64, v115
	s_lshr_b32 s47, s29, 16
	s_lshr_b32 s48, s3, 16
	s_mov_b32 s42, s31
	s_movk_i32 s16, 0x80
	s_or_b32 s47, s47, 0x600000
	s_movk_i32 s8, 0xe0
	s_or_b32 s48, s48, 0x600000
	s_add_nc_u64 s[36:37], s[18:19], s[4:5]
	s_mov_b32 s4, 0x7510000
	s_mov_b32 s49, s11
	s_branch .LBB0_17
.LBB0_16:                               ;   in Loop: Header=BB0_17 Depth=1
	s_cmp_eq_u32 s49, s45
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
	s_mul_i32 s2, s49, 0xffffffa0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s2, s2, s26
	s_min_i32 s2, s2, 0x60
	s_cmp_lt_i32 s49, s45
	s_cselect_b32 s10, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_and_b32 s6, s10, exec_lo
	s_cselect_b32 s2, s2, 0
	s_cmp_lt_i32 s2, 0x60
	s_cselect_b32 s6, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b32 s6, s46, s6
	s_or_b32 s6, s40, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s6
	s_cbranch_vccnz .LBB0_29
; %bb.18:                               ;   in Loop: Header=BB0_17 Depth=1
	v_mov_b64_e32 v[210:211], v[0:1]
	v_mov_b32_e32 v133, v125
	s_cmp_lg_u32 s5, 0
	s_mov_b32 s12, 0
	s_cselect_b32 s7, s44, s31
	s_cselect_b32 s6, s43, 0
.LBB0_19:                               ;   Parent Loop BB0_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_dual_mov_b32 v114, v210 :: v_dual_add_nc_u32 v133, -2, v133
	v_dual_mov_b32 v212, v211 :: v_dual_mov_b32 v213, v115
	v_add_nc_u32_e32 v211, 0x200, v211
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u64 v[214:215], v[114:115], 2, s[6:7]
	v_cmp_eq_u32_e32 vcc_lo, 0, v133
	v_add_nc_u32_e32 v210, 0x200, v210
	v_lshl_add_u64 v[212:213], v[212:213], 2, s[6:7]
	s_clause 0x1
	flat_store_b32 v[214:215], v115
	flat_store_b32 v[212:213], v115
	s_or_b32 s12, vcc_lo, s12
	s_wait_xcnt 0x0
	s_and_not1_b32 exec_lo, exec_lo, s12
	s_cbranch_execnz .LBB0_19
; %bb.20:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s12
	s_and_saveexec_b32 s12, s0
	s_cbranch_execz .LBB0_23
; %bb.21:                               ;   in Loop: Header=BB0_17 Depth=1
	v_add_nc_u64_e32 v[210:211], s[6:7], v[206:207]
	v_mov_b32_e32 v114, v127
	s_mov_b32 s6, 0
.LBB0_22:                               ;   Parent Loop BB0_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v114, 0x100, v114
	flat_store_b32 v[210:211], v115
	s_wait_xcnt 0x0
	v_add_nc_u64_e32 v[210:211], 0x400, v[210:211]
	v_cmp_lt_u32_e32 vcc_lo, 0x2b9f, v114
	s_or_b32 s6, vcc_lo, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s6
	s_cbranch_execnz .LBB0_22
.LBB0_23:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s12
	v_mov_b64_e32 v[210:211], v[0:1]
	v_mov_b32_e32 v133, v129
	s_cmp_lg_u32 s5, 0
	s_mov_b32 s12, 0
	s_cselect_b32 s7, s35, s42
	s_cselect_b32 s6, s34, s41
.LBB0_24:                               ;   Parent Loop BB0_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_dual_mov_b32 v114, v210 :: v_dual_add_nc_u32 v133, -2, v133
	v_dual_mov_b32 v212, v211 :: v_dual_mov_b32 v213, v115
	v_add_nc_u32_e32 v211, 0x200, v211
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u64 v[214:215], v[114:115], 2, s[6:7]
	v_cmp_eq_u32_e32 vcc_lo, 0, v133
	v_add_nc_u32_e32 v210, 0x200, v210
	v_lshl_add_u64 v[212:213], v[212:213], 2, s[6:7]
	s_clause 0x1
	flat_store_b32 v[214:215], v115
	flat_store_b32 v[212:213], v115
	s_or_b32 s12, vcc_lo, s12
	s_wait_xcnt 0x0
	s_and_not1_b32 exec_lo, exec_lo, s12
	s_cbranch_execnz .LBB0_24
; %bb.25:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s12
	s_and_saveexec_b32 s12, s1
	s_cbranch_execz .LBB0_28
; %bb.26:                               ;   in Loop: Header=BB0_17 Depth=1
	v_add_nc_u64_e32 v[210:211], s[6:7], v[208:209]
	v_mov_b32_e32 v114, v131
	s_mov_b32 s6, 0
.LBB0_27:                               ;   Parent Loop BB0_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v114, 0x100, v114
	flat_store_b32 v[210:211], v115
	s_wait_xcnt 0x0
	v_add_nc_u64_e32 v[210:211], 0x400, v[210:211]
	v_cmp_lt_u32_e32 vcc_lo, 0x187f, v114
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
	v_cmpx_lt_i32_e32 0, v123
	s_xor_b32 s7, exec_lo, s7
	s_cbranch_execnz .LBB0_32
; %bb.30:                               ;   in Loop: Header=BB0_17 Depth=1
	s_and_not1_saveexec_b32 s12, s7
	s_cbranch_execnz .LBB0_35
.LBB0_31:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s12
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s30
	s_cbranch_vccnz .LBB0_16
	s_branch .LBB0_36
.LBB0_32:                               ;   in Loop: Header=BB0_17 Depth=1
	s_mov_b32 s51, exec_lo
	v_cmpx_eq_u32_e32 1, v123
	s_cbranch_execz .LBB0_34
; %bb.33:                               ;   in Loop: Header=BB0_17 Depth=1
	s_cmp_lg_u32 s5, 0
	s_mul_i32 s10, s6, 0x60
	s_cselect_b32 s14, s34, s41
	s_cmp_gt_i32 s2, 0
	s_mov_b32 s28, s2
	s_cselect_b32 s15, -1, 0
	s_lshl_b64 s[12:13], s[10:11], 1
	s_mov_b32 s17, s9
	s_add_nc_u64 s[12:13], s[20:21], s[12:13]
	s_delay_alu instid0(SALU_CYCLE_1)
	v_dual_mov_b32 v133, s14 :: v_dual_mov_b32 v210, s12
	s_and_b32 s10, s13, 0x1ffffff
	s_and_b32 s13, s25, s15
	s_bitset1_b32 s10, 31
	v_cndmask_b32_e64 v114, 0, 1, s13
	v_mov_b32_e32 v135, s10
	v_readfirstlane_b32 s53, v133
	v_readfirstlane_b32 s54, v210
	s_lshr_b64 s[14:15], s[28:29], 16
	v_readfirstlane_b32 s52, v114
	v_readfirstlane_b32 s55, v135
	s_lshl_b32 s13, s2, 16
	s_mov_b32 s12, s4
	s_mov_b32 s15, s47
	s_mov_b32 s18, s11
	s_mov_b32 s19, s11
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[52:55], s[12:19]
.LBB0_34:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s51
	s_and_not1_saveexec_b32 s12, s7
	s_cbranch_execz .LBB0_31
.LBB0_35:                               ;   in Loop: Header=BB0_17 Depth=1
	s_cmp_lg_u32 s5, 0
	s_mul_i32 s10, s6, 0x60
	s_cselect_b32 s5, s43, 0
	s_cmp_gt_i32 s2, 0
	s_cselect_b32 s13, -1, 0
	s_lshl_b64 s[6:7], s[10:11], 1
	s_and_b32 s10, s39, s13
	s_add_nc_u64 s[6:7], s[36:37], s[6:7]
	v_cndmask_b32_e64 v114, 0, 1, s10
	s_and_b32 s7, s7, 0x1ffffff
	v_dual_mov_b32 v133, s5 :: v_dual_mov_b32 v210, s6
	s_bitset1_b32 s7, 31
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_readfirstlane_b32 s52, v114
	v_mov_b32_e32 v135, s7
	v_readfirstlane_b32 s53, v133
	v_readfirstlane_b32 s54, v210
	s_lshr_b64 s[6:7], s[2:3], 16
	s_lshl_b32 s5, s2, 16
	v_readfirstlane_b32 s55, v135
	s_mov_b32 s7, s48
	s_mov_b32 s10, s11
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	tensor_load_to_lds s[52:55], s[4:11]
	s_or_b32 exec_lo, exec_lo, s12
	s_and_not1_b32 vcc_lo, exec_lo, s30
	s_cbranch_vccnz .LBB0_16
.LBB0_36:                               ;   in Loop: Header=BB0_17 Depth=1
	s_cmp_lg_u32 s50, 0
	s_cselect_b32 s2, s43, 0
	s_cselect_b32 s5, s34, s41
	v_lshl_add_u32 v114, v116, 1, s2
	v_lshl_add_u32 v133, v120, 1, s2
	v_lshl_add_u32 v137, v124, 1, s2
	v_lshl_add_u32 v135, v122, 1, s2
	v_lshl_add_u32 v139, v154, 1, s2
	ds_load_b128 v[210:213], v114
	ds_load_b128 v[214:217], v114 offset:16
	ds_load_b128 v[218:221], v133 offset:3072
	ds_load_b128 v[222:225], v133 offset:3088
	v_lshl_add_u32 v114, v126, 1, s2
	v_lshl_add_u32 v133, v128, 1, s2
	ds_load_b128 v[234:237], v137 offset:9216
	ds_load_b128 v[238:241], v137 offset:9232
	v_lshl_add_u32 v137, v118, 1, s5
	ds_load_b128 v[242:245], v114 offset:12288
	ds_load_b128 v[246:249], v114 offset:12304
	v_lshl_add_u32 v114, v130, 1, s2
	ds_load_b128 v[250:253], v133 offset:15360
	ds_load_b128 v[254:257], v133 offset:15376
	v_lshl_add_u32 v133, v132, 1, s2
	ds_load_b128 v[226:229], v135 offset:6144
	ds_load_b128 v[230:233], v135 offset:6160
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[2:5] /*v[258:261]*/, v114 offset:18432
	ds_load_b128 v[6:9] /*v[262:265]*/, v114 offset:18448
	ds_load_b128 v[10:13] /*v[266:269]*/, v133 offset:21504
	ds_load_b128 v[22:25] /*v[278:281]*/, v137
	ds_load_b128 v[26:29] /*v[282:285]*/, v137 offset:16
	s_set_vgpr_msb 0x4004                   ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_lshl_add_u32 v135, v144, 1, s2
	v_lshl_add_u32 v114, v142, 1, s2
	v_lshl_add_u32 v137, v152, 1, s2
	v_lshl_add_u32 v141, v156, 1, s2
	v_lshl_add_u32 v143, v158, 1, s2
	v_lshl_add_u32 v145, v160, 1, s2
	v_lshl_add_u32 v147, v162, 1, s2
	v_lshl_add_u32 v149, v164, 1, s2
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[106:113], v[210:217], v[22:29] /*v[278:285]*/, v[106:113]
	v_lshl_add_u32 v151, v166, 1, s2
	v_lshl_add_u32 v153, v168, 1, s2
	v_lshl_add_u32 v155, v170, 1, s2
	v_lshl_add_u32 v157, v172, 1, s2
	v_lshl_add_u32 v159, v174, 1, s5
	s_set_vgpr_msb 0x440                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[30:33] /*v[286:289]*/, v137 offset:64
	ds_load_b128 v[34:37] /*v[290:293]*/, v137 offset:80
	s_set_vgpr_msb 0x4004                   ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[98:105], v[218:225], v[22:29] /*v[278:285]*/, v[98:105] matrix_b_reuse
	s_set_vgpr_msb 0x440                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[38:41] /*v[294:297]*/, v141 offset:64
	ds_load_b128 v[42:45] /*v[298:301]*/, v141 offset:80
	ds_load_b128 v[46:49] /*v[302:305]*/, v143 offset:64
	ds_load_b128 v[50:53] /*v[306:309]*/, v143 offset:80
	ds_load_b128 v[54:57] /*v[310:313]*/, v145 offset:64
	ds_load_b128 v[58:61] /*v[314:317]*/, v145 offset:80
	ds_load_b128 v[62:65] /*v[318:321]*/, v149 offset:64
	s_set_vgpr_msb 0x4004                   ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[90:97], v[226:233], v[22:29] /*v[278:285]*/, v[90:97] matrix_b_reuse
	ds_load_b128 v[226:229], v139 offset:64
	ds_load_b128 v[230:233], v139 offset:80
	s_set_vgpr_msb 0x440                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[66:69] /*v[322:325]*/, v149 offset:80
	ds_load_b128 v[70:73] /*v[326:329]*/, v151 offset:64
	ds_load_b128 v[74:77] /*v[330:333]*/, v151 offset:80
	ds_load_b128 v[78:81] /*v[334:337]*/, v155 offset:64
	ds_load_b128 v[82:85] /*v[338:341]*/, v155 offset:80
	s_set_vgpr_msb 0x4004                   ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[82:89], v[234:241], v[22:29] /*v[278:285]*/, v[82:89] matrix_b_reuse
	ds_load_b128 v[234:237], v147 offset:64
	ds_load_b128 v[238:241], v147 offset:80
	s_set_vgpr_msb 0x440                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[86:89] /*v[342:345]*/, v157 offset:64
	ds_load_b128 v[90:93] /*v[346:349]*/, v157 offset:80
	ds_load_b128 v[94:97] /*v[350:353]*/, v159 offset:64
	ds_load_b128 v[98:101] /*v[354:357]*/, v159 offset:80
	; sched_group_barrier mask(0x00000100) size(15) SyncID(0)
	s_set_vgpr_msb 0x4004                   ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[74:81], v[242:249], v[22:29] /*v[278:285]*/, v[74:81] matrix_b_reuse
	ds_load_b128 v[242:245], v153 offset:64
	ds_load_b128 v[246:249], v153 offset:80
	v_wmma_f32_16x16x32_bf16 v[66:73], v[250:257], v[22:29] /*v[278:285]*/, v[66:73] matrix_b_reuse
	s_set_vgpr_msb 0x405                    ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[50:57], v[2:9] /*v[258:265]*/, v[22:29] /*v[278:285]*/, v[50:57] matrix_b_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[14:17] /*v[270:273]*/, v135 offset:39936
	ds_load_b128 v[18:21] /*v[274:277]*/, v135 offset:39952
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v135, v150, 1, s2
	; sched_group_barrier mask(0x00000008) size(7) SyncID(0)
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[58:65], v[14:21] /*v[270:277]*/, v[22:29] /*v[278:285]*/, v[58:65] matrix_b_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[14:17] /*v[270:273]*/, v114 offset:36864
	ds_load_b128 v[18:21] /*v[274:277]*/, v114 offset:36880
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v114, v140, 1, s2
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[2:9], v[14:21] /*v[270:277]*/, v[22:29] /*v[278:285]*/, v[2:9] matrix_b_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[14:17] /*v[270:273]*/, v114 offset:33792
	ds_load_b128 v[18:21] /*v[274:277]*/, v114 offset:33808
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v114, v138, 1, s2
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[10:17], v[14:21] /*v[270:277]*/, v[22:29] /*v[278:285]*/, v[10:17] matrix_b_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[14:17] /*v[270:273]*/, v114 offset:30720
	ds_load_b128 v[18:21] /*v[274:277]*/, v114 offset:30736
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v114, v136, 1, s2
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[18:25], v[14:21] /*v[270:277]*/, v[22:29] /*v[278:285]*/, v[18:25] matrix_b_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[14:17] /*v[270:273]*/, v114 offset:27648
	ds_load_b128 v[18:21] /*v[274:277]*/, v114 offset:27664
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v114, v134, 1, s2
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[26:33], v[14:21] /*v[270:277]*/, v[22:29] /*v[278:285]*/, v[26:33] matrix_b_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[14:17] /*v[270:273]*/, v114 offset:24576
	ds_load_b128 v[18:21] /*v[274:277]*/, v114 offset:24592
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v114, v146, 1, s2
	ds_load_b128 v[210:213], v114 offset:64
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x32_bf16 v[34:41], v[14:21] /*v[270:277]*/, v[22:29] /*v[278:285]*/, v[34:41] matrix_b_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[14:17] /*v[270:273]*/, v133 offset:21520
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v133, v148, 1, s2
	ds_load_b128 v[214:217], v114 offset:80
	; sched_group_barrier mask(0x00000100) size(15) SyncID(0)
	ds_load_b128 v[218:221], v133 offset:64
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[42:49], v[10:17] /*v[266:273]*/, v[22:29] /*v[278:285]*/, v[42:49] matrix_b_reuse
	s_set_vgpr_msb 0x500                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	ds_load_b128 v[222:225], v133 offset:80
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[10:13] /*v[266:269]*/, v135 offset:64
	ds_load_b128 v[14:17] /*v[270:273]*/, v135 offset:80
	; sched_group_barrier mask(0x00000008) size(7) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v114, v176, 1, s2
	v_lshl_add_u32 v133, v178, 1, s2
	v_lshl_add_u32 v135, v180, 1, s2
	ds_load_b128 v[250:253], v114 offset:128
	ds_load_b128 v[254:257], v114 offset:144
	v_lshl_add_u32 v114, v182, 1, s2
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[2:5] /*v[258:261]*/, v133 offset:128
	ds_load_b128 v[6:9] /*v[262:265]*/, v133 offset:144
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v133, v184, 1, s2
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[18:21] /*v[274:277]*/, v135 offset:128
	ds_load_b128 v[102:105] /*v[358:361]*/, v114 offset:128
	ds_load_b128 v[106:109] /*v[362:365]*/, v114 offset:144
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v114, v186, 1, s2
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[22:25] /*v[278:281]*/, v135 offset:144
	ds_load_b128 v[110:113] /*v[366:369]*/, v133 offset:128
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v135, v188, 1, s2
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[114:117] /*v[370:373]*/, v133 offset:144
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v133, v190, 1, s2
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[118:121] /*v[374:377]*/, v114 offset:128
	ds_load_b128 v[122:125] /*v[378:381]*/, v114 offset:144
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v114, v192, 1, s2
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[126:129] /*v[382:385]*/, v135 offset:128
	ds_load_b128 v[130:133] /*v[386:389]*/, v135 offset:144
	ds_load_b128 v[26:29] /*v[282:285]*/, v133 offset:128
	s_set_vgpr_msb 0x4004                   ;  msbs: dst=0 src0=0 src1=1 src2=0
	s_wait_dscnt 0x13
	v_wmma_f32_16x16x32_bf16 v[106:113], v[210:217], v[94:101] /*v[350:357]*/, v[106:113]
	v_lshl_add_u32 v135, v194, 1, s2
	; sched_group_barrier mask(0x00000100) size(15) SyncID(0)
	s_wait_dscnt 0x11
	v_wmma_f32_16x16x32_bf16 v[98:105], v[218:225], v[94:101] /*v[350:357]*/, v[98:105] matrix_b_reuse
	s_set_vgpr_msb 0x405                    ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0xf
	v_wmma_f32_16x16x32_bf16 v[90:97], v[10:17] /*v[266:273]*/, v[94:101] /*v[350:357]*/, v[90:97] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[82:89], v[30:37] /*v[286:293]*/, v[94:101] /*v[350:357]*/, v[82:89] matrix_b_reuse
	s_set_vgpr_msb 0x504                    ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[74:81], v[226:233], v[94:101] /*v[350:357]*/, v[74:81] matrix_b_reuse
	s_set_vgpr_msb 0x405                    ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[66:73], v[38:45] /*v[294:301]*/, v[94:101] /*v[350:357]*/, v[66:73] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[50:57], v[46:53] /*v[302:309]*/, v[94:101] /*v[350:357]*/, v[50:57] matrix_b_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[30:33] /*v[286:289]*/, v133 offset:144
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	ds_load_b128 v[210:213], v114 offset:128
	v_lshl_add_u32 v133, v196, 1, s2
	ds_load_b128 v[214:217], v114 offset:144
	v_lshl_add_u32 v114, v198, 1, s2
	ds_load_b128 v[218:221], v135 offset:128
	ds_load_b128 v[222:225], v135 offset:144
	ds_load_b128 v[226:229], v133 offset:128
	ds_load_b128 v[230:233], v133 offset:144
	v_lshl_add_u32 v133, v200, 1, s2
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[10:13] /*v[266:269]*/, v114 offset:128
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v135, v202, 1, s2
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[14:17] /*v[270:273]*/, v114 offset:144
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v114, v204, 1, s5
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[34:37] /*v[290:293]*/, v133 offset:128
	ds_load_b128 v[38:41] /*v[294:297]*/, v133 offset:144
	ds_load_b128 v[42:45] /*v[298:301]*/, v135 offset:128
	ds_load_b128 v[46:49] /*v[302:305]*/, v135 offset:144
	ds_load_b128 v[134:137] /*v[390:393]*/, v114 offset:128
	ds_load_b128 v[138:141] /*v[394:397]*/, v114 offset:144
	s_set_vgpr_msb 0x4005                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[42:49], v[54:61] /*v[310:317]*/, v[94:101] /*v[350:357]*/, v[42:49] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(7) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(15) SyncID(0)
	s_set_vgpr_msb 0x504                    ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[34:41], v[234:241], v[94:101] /*v[350:357]*/, v[34:41] matrix_b_reuse
	s_set_vgpr_msb 0x405                    ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[26:33], v[62:69] /*v[318:325]*/, v[94:101] /*v[350:357]*/, v[26:33] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[18:25], v[70:77] /*v[326:333]*/, v[94:101] /*v[350:357]*/, v[18:25] matrix_b_reuse
	s_set_vgpr_msb 0x504                    ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[10:17], v[242:249], v[94:101] /*v[350:357]*/, v[10:17] matrix_b_reuse
	s_set_vgpr_msb 0x405                    ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[2:9], v[78:85] /*v[334:341]*/, v[94:101] /*v[350:357]*/, v[2:9] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[58:65], v[86:93] /*v[342:349]*/, v[94:101] /*v[350:357]*/, v[58:65] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(7) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x504                    ;  msbs: dst=0 src0=0 src1=1 src2=0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[106:113], v[250:257], v[134:141] /*v[390:397]*/, v[106:113]
	; sched_group_barrier mask(0x00000100) size(15) SyncID(0)
	s_set_vgpr_msb 0x405                    ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[98:105], v[2:9] /*v[258:265]*/, v[134:141] /*v[390:397]*/, v[98:105] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[90:97], v[18:25] /*v[274:281]*/, v[134:141] /*v[390:397]*/, v[90:97] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[82:89], v[102:109] /*v[358:365]*/, v[134:141] /*v[390:397]*/, v[82:89] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[74:81], v[110:117] /*v[366:373]*/, v[134:141] /*v[390:397]*/, v[74:81] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[66:73], v[118:125] /*v[374:381]*/, v[134:141] /*v[390:397]*/, v[66:73] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[50:57], v[126:133] /*v[382:389]*/, v[134:141] /*v[390:397]*/, v[50:57] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(7) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(15) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[42:49], v[26:33] /*v[282:289]*/, v[134:141] /*v[390:397]*/, v[42:49] matrix_b_reuse
	s_set_vgpr_msb 0x504                    ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[34:41], v[210:217], v[134:141] /*v[390:397]*/, v[34:41] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[26:33], v[218:225], v[134:141] /*v[390:397]*/, v[26:33] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[18:25], v[226:233], v[134:141] /*v[390:397]*/, v[18:25] matrix_b_reuse
	s_set_vgpr_msb 0x405                    ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[10:17], v[10:17] /*v[266:273]*/, v[134:141] /*v[390:397]*/, v[10:17] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[2:9], v[34:41] /*v[290:297]*/, v[134:141] /*v[390:397]*/, v[2:9] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[58:65], v[42:49] /*v[298:305]*/, v[134:141] /*v[390:397]*/, v[58:65] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(7) SyncID(0)
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
	s_and_b32 vcc_lo, exec_lo, s30
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_vccz .LBB0_39
; %bb.38:
	v_dual_add_nc_u32 v1, v121, v119 :: v_dual_lshrrev_b32 v114, 1, v0
	v_cvt_pk_bf16_f32 v105, v104, v105
	v_cvt_pk_bf16_f32 v104, v102, v103
	v_cvt_pk_bf16_f32 v103, v100, v101
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_u32_u24_e32 v1, 0xe0, v1
	v_cvt_pk_bf16_f32 v102, v98, v99
	v_cvt_pk_bf16_f32 v97, v96, v97
	v_cvt_pk_bf16_f32 v96, v94, v95
	v_cvt_pk_bf16_f32 v95, v92, v93
	v_and_or_b32 v100, v114, 8, v1
	v_lshrrev_b32_e32 v1, 3, v1
	v_cvt_pk_bf16_f32 v94, v90, v91
	v_cvt_pk_bf16_f32 v89, v88, v89
	v_cvt_pk_bf16_f32 v113, v112, v113
	v_add_nc_u32_e32 v98, 32, v100
	v_and_b32_e32 v1, 0x1ff0, v1
	v_dual_add_nc_u32 v90, 48, v100 :: v_dual_lshlrev_b32 v92, 1, v100
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_add_nc_u32 v88, 64, v100 :: v_dual_lshrrev_b32 v91, 3, v98
	v_cvt_pk_bf16_f32 v112, v110, v111
	v_cvt_pk_bf16_f32 v111, v108, v109
	v_cvt_pk_bf16_f32 v110, v106, v107
	v_dual_lshrrev_b32 v90, 3, v90 :: v_dual_add_nc_u32 v1, v1, v92
	v_lshrrev_b32_e32 v88, 3, v88
	v_and_b32_e32 v91, 0x3ff0, v91
	ds_store_b128 v1, v[110:113]
	ds_store_b128 v1, v[102:105] offset:32
	v_and_b32_e32 v90, 0x3ff0, v90
	v_and_b32_e32 v1, 0x3ff0, v88
	v_add_nc_u32_e32 v93, 0x50, v100
	v_add_nc_u32_e32 v91, v91, v92
	v_cvt_pk_bf16_f32 v81, v80, v81
	v_cvt_pk_bf16_f32 v80, v78, v79
	v_cvt_pk_bf16_f32 v79, v76, v77
	v_add_nc_u32_e32 v76, 0x60, v100
	v_add_nc_u32_e32 v90, v90, v92
	v_cvt_pk_bf16_f32 v88, v86, v87
	v_cvt_pk_bf16_f32 v87, v84, v85
	v_cvt_pk_bf16_f32 v86, v82, v83
	v_add_nc_u32_e32 v1, v1, v92
	v_cvt_pk_bf16_f32 v78, v74, v75
	v_cvt_pk_bf16_f32 v73, v72, v73
	v_cvt_pk_bf16_f32 v72, v70, v71
	v_cvt_pk_bf16_f32 v71, v68, v69
	v_add_nc_u32_e32 v68, 0x70, v100
	ds_store_b128 v91, v[94:97] offset:64
	v_dual_lshrrev_b32 v91, 3, v93 :: v_dual_lshrrev_b32 v75, 3, v76
	ds_store_b128 v90, v[86:89] offset:96
	ds_store_b128 v1, v[78:81] offset:128
	v_lshrrev_b32_e32 v1, 3, v68
	v_and_b32_e32 v82, 0x3ff0, v91
	v_cvt_pk_bf16_f32 v70, v66, v67
	v_and_b32_e32 v66, 0x3ff0, v75
	v_cvt_pk_bf16_f32 v33, v32, v33
	v_and_b32_e32 v1, 0x3ff0, v1
	v_cvt_pk_bf16_f32 v32, v30, v31
	v_cvt_pk_bf16_f32 v30, v26, v27
	v_add_nc_u32_e32 v27, 0xc0, v100
	v_add_nc_u32_e32 v67, 0x80, v100
	v_cvt_pk_bf16_f32 v57, v56, v57
	v_cvt_pk_bf16_f32 v56, v54, v55
	v_cvt_pk_bf16_f32 v54, v50, v51
	v_add_nc_u32_e32 v51, 0x90, v100
	v_add_nc_u32_e32 v74, v82, v92
	v_add_nc_u32_e32 v66, v66, v92
	v_cvt_pk_bf16_f32 v55, v52, v53
	v_cvt_pk_bf16_f32 v41, v40, v41
	v_cvt_pk_bf16_f32 v40, v38, v39
	v_cvt_pk_bf16_f32 v39, v36, v37
	v_add_nc_u32_e32 v36, 0xa0, v100
	v_add_nc_u32_e32 v1, v1, v92
	v_cvt_pk_bf16_f32 v49, v48, v49
	v_cvt_pk_bf16_f32 v48, v46, v47
	v_cvt_pk_bf16_f32 v47, v44, v45
	v_cvt_pk_bf16_f32 v46, v42, v43
	v_cvt_pk_bf16_f32 v38, v34, v35
	v_add_nc_u32_e32 v35, 0xb0, v100
	v_cvt_pk_bf16_f32 v25, v24, v25
	v_cvt_pk_bf16_f32 v24, v22, v23
	v_cvt_pk_bf16_f32 v23, v20, v21
	v_lshrrev_b32_e32 v20, 3, v27
	v_add_nc_u32_e32 v21, 0xd0, v100
	v_lshrrev_b32_e32 v67, 3, v67
	v_lshrrev_b32_e32 v43, 3, v51
	ds_store_b128 v74, v[70:73] offset:160
	ds_store_b128 v66, v[54:57] offset:192
	ds_store_b128 v1, v[46:49] offset:224
	v_dual_lshrrev_b32 v1, 3, v36 :: v_dual_lshrrev_b32 v26, 3, v35
	v_cvt_pk_bf16_f32 v22, v18, v19
	v_and_b32_e32 v19, 0x3ff0, v20
	v_lshrrev_b32_e32 v20, 3, v21
	v_and_b32_e32 v50, 0x3ff0, v67
	v_and_b32_e32 v34, 0x3ff0, v43
	v_and_b32_e32 v1, 0x3ff0, v1
	v_and_b32_e32 v26, 0x3ff0, v26
	v_cvt_pk_bf16_f32 v17, v16, v17
	v_cvt_pk_bf16_f32 v16, v14, v15
	v_cvt_pk_bf16_f32 v14, v10, v11
	v_and_b32_e32 v11, 0x3ff0, v20
	v_add_nc_u32_e32 v42, v50, v92
	v_add_nc_u32_e32 v34, v34, v92
	v_cvt_pk_bf16_f32 v31, v28, v29
	v_dual_add_nc_u32 v1, v1, v92 :: v_dual_add_nc_u32 v18, v26, v92
	v_cvt_pk_bf16_f32 v15, v12, v13
	v_add_nc_u32_e32 v10, v19, v92
	v_cvt_pk_bf16_f32 v9, v8, v9
	v_cvt_pk_bf16_f32 v8, v6, v7
	v_cvt_pk_bf16_f32 v7, v4, v5
	v_cvt_pk_bf16_f32 v6, v2, v3
	v_add_nc_u32_e32 v11, v11, v92
	v_cvt_pk_bf16_f32 v5, v64, v65
	v_cvt_pk_bf16_f32 v4, v62, v63
	v_cvt_pk_bf16_f32 v3, v60, v61
	v_cvt_pk_bf16_f32 v2, v58, v59
	ds_store_b128 v42, v[38:41] offset:256
	ds_store_b128 v34, v[30:33] offset:288
	ds_store_b128 v1, v[22:25] offset:320
	ds_store_b128 v18, v[14:17] offset:352
	ds_store_b128 v10, v[6:9] offset:384
	ds_store_b128 v11, v[2:5] offset:416
.LBB0_39:
	v_cmp_ne_u32_e32 vcc_lo, 1, v117
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
	s_mul_i32 s0, s38, 0xe0
	v_nop
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
	.size	bm224_bn128_bk096_wm1_wn8_mc0, .Lfunc_end0-bm224_bn128_bk096_wm1_wn8_mc0
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel bm224_bn128_bk096_wm1_wn8_mc0
		.amdhsa_group_segment_fixed_size 143616
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
		.amdhsa_next_free_vgpr 398
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
		.amdhsa_inst_pref_size 72
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text.bm224_bn128_bk096_wm1_wn8_mc0,"axG",@progbits,bm224_bn128_bk096_wm1_wn8_mc0,comdat
                                        ; -- End function
	.set .Lbm224_bn128_bk096_wm1_wn8_mc0.num_vgpr, 398
	.set .Lbm224_bn128_bk096_wm1_wn8_mc0.num_agpr, 0
	.set .Lbm224_bn128_bk096_wm1_wn8_mc0.numbered_sgpr, 56
	.set .Lbm224_bn128_bk096_wm1_wn8_mc0.num_named_barrier, 0
	.set .Lbm224_bn128_bk096_wm1_wn8_mc0.private_seg_size, 0
	.set .Lbm224_bn128_bk096_wm1_wn8_mc0.uses_vcc, 1
	.set .Lbm224_bn128_bk096_wm1_wn8_mc0.uses_flat_scratch, 1
	.set .Lbm224_bn128_bk096_wm1_wn8_mc0.has_dyn_sized_stack, 0
	.set .Lbm224_bn128_bk096_wm1_wn8_mc0.has_recursion, 0
	.set .Lbm224_bn128_bk096_wm1_wn8_mc0.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 9096
; TotalNumSgprs: 58
; NumVgprs: 398
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 143616 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 24
; NumSGPRsForWavesPerEU: 58
; NumVGPRsForWavesPerEU: 398
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
	.type	__hip_cuid_5e086a16ebdb857d,@object ; @__hip_cuid_5e086a16ebdb857d
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_5e086a16ebdb857d
__hip_cuid_5e086a16ebdb857d:
	.byte	0                               ; 0x0
	.size	__hip_cuid_5e086a16ebdb857d, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git aa451e1fe6a793394d6733051b1778633063ae96+PATCHED:d17c5aa0e3ea29cde402f58f27e39b6034effa27)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_5e086a16ebdb857d
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
    .group_segment_fixed_size: 143616
    .kernarg_segment_align: 8
    .kernarg_segment_size: 132
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           bm224_bn128_bk096_wm1_wn8_mc0
    .private_segment_fixed_size: 0
    .sgpr_count:     58
    .sgpr_spill_count: 0
    .symbol:         bm224_bn128_bk096_wm1_wn8_mc0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     398
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
