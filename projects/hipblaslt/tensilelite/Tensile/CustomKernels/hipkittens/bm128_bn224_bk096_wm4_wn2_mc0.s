	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.section	.text.bm128_bn224_bk096_wm4_wn2_mc0,"axG",@progbits,bm128_bn224_bk096_wm4_wn2_mc0,comdat
	.protected	bm128_bn224_bk096_wm4_wn2_mc0 ; -- Begin function bm128_bn224_bk096_wm4_wn2_mc0
	.globl	bm128_bn224_bk096_wm4_wn2_mc0
	.p2align	8
	.type	bm128_bn224_bk096_wm4_wn2_mc0,@function
bm128_bn224_bk096_wm4_wn2_mc0: ; @bm128_bn224_bk096_wm4_wn2_mc0
; %bb.0:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1 ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_mov_b64 s[2:3], src_shared_base
	s_movk_i32 s2, 0x6600
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
	s_cselect_b32 s33, ttmp7, s5
	s_add_co_i32 s2, s24, 0x7f
	s_add_co_i32 s5, s25, 0xdf
	s_ashr_i32 s3, s2, 31
	s_mul_hi_i32 s6, s5, 0x92492493
	s_lshr_b32 s3, s3, 25
	s_add_co_i32 s6, s6, s5
	s_add_co_i32 s2, s2, s3
	s_lshr_b32 s3, s6, 31
	s_ashr_i32 s5, s2, 7
	s_lshl_b32 s2, s36, 7
	s_ashr_i32 s6, s6, 7
	s_sub_co_i32 s7, s24, s2
	s_add_co_i32 s6, s6, s3
	s_min_i32 s27, s7, 0x80
	s_cmp_lt_i32 s36, s5
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_and_b32 s7, s3, exec_lo
	s_mul_i32 s7, s33, 0xffffff20
	s_cselect_b32 s29, s27, 0
	s_add_co_i32 s7, s25, s7
	s_min_i32 s7, s7, 0xe0
	s_cmp_lt_i32 s33, s6
	s_cselect_b32 s25, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
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
	s_cmp_lt_i32 s29, 0x80
	s_cselect_b32 s40, -1, 0
	s_and_b32 vcc_lo, exec_lo, s40
	s_mov_b32 s7, s40
	s_cbranch_vccnz .LBB0_2
; %bb.1:
	s_cmp_lt_i32 s31, 0xe0
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
	v_cmp_lt_u32_e32 vcc_lo, 0x187f, v5
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
	ds_store_b32 v2, v3 offset:26112
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
	s_load_b64 s[18:19], s[0:1], 0x0 nv
	s_load_b128 s[12:15], s[0:1], 0x20 nv
	s_load_b128 s[20:23], s[0:1], 0x48 nv
	v_lshrrev_b32_e32 v123, 5, v0
	s_wait_xcnt 0x0
	s_lshl_b32 s0, s4, 2
	s_add_co_i32 s6, s6, -1
	s_mov_b64 s[34:35], src_shared_base
	s_or_b32 s41, s0, 0x6600
	s_add_co_i32 s1, s5, -1
	s_min_i32 s0, s33, s6
	s_mov_b32 s4, exec_lo
	v_cmpx_lt_i32_e32 0, v123
	s_xor_b32 s34, exec_lo, s4
	s_cbranch_execz .LBB0_12
; %bb.9:
	s_mov_b32 s37, exec_lo
	v_cmpx_eq_u32_e32 1, v123
	s_cbranch_execz .LBB0_11
; %bb.10:
	s_cmp_gt_i32 s28, 0
	s_mul_i32 s4, s0, 0xe0
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
	v_dual_mov_b32 v1, s41 :: v_dual_mov_b32 v4, s4
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
	s_movk_i32 s8, 0xe0
	s_lshl_b32 s5, s28, 16
	s_or_b32 s7, s4, 0x600000
	s_mov_b32 s4, 0x7510000
	s_mov_b32 s11, s10
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[44:47], s[4:11]
.LBB0_11:
	s_or_b32 exec_lo, exec_lo, s37
.LBB0_12:
	s_or_saveexec_b32 s34, s34
	s_min_i32 s30, s36, s1
	s_xor_b32 exec_lo, exec_lo, s34
	s_cbranch_execz .LBB0_14
; %bb.13:
	s_cmp_gt_i32 s28, 0
	s_mov_b32 s10, 0
	s_cselect_b32 s1, -1, 0
	s_lshl_b32 s4, s30, 7
	s_wait_kmcnt 0x0
	s_bfe_i64 s[6:7], s[12:13], 0x200000
	s_ashr_i32 s5, s4, 31
	s_and_b32 s1, s3, s1
	s_mul_u64 s[4:5], s[6:7], s[4:5]
	v_cndmask_b32_e64 v2, 0, 1, s1
	s_lshl_b64 s[6:7], s[4:5], 1
	s_lshr_b32 s4, s29, 16
	s_add_nc_u64 s[6:7], s[18:19], s[6:7]
	s_lshl_b32 s5, s28, 16
	s_and_b32 s7, s7, 0x1ffffff
	v_readfirstlane_b32 s36, v2
	s_bitset1_b32 s7, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v4, s6 :: v_dual_mov_b32 v3, s7
	s_lshr_b64 s[6:7], s[28:29], 16
	s_or_b32 s7, s4, 0x600000
	s_movk_i32 s8, 0x80
	v_readfirstlane_b32 s38, v4
	v_readfirstlane_b32 s39, v3
	s_mov_b32 s4, 0x7510000
	s_mov_b32 s11, s10
	s_mov_b32 s37, s10
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[36:39], s[4:11]
.LBB0_14:
	s_or_b32 exec_lo, exec_lo, s34
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	v_dual_lshlrev_b32 v1, 4, v123 :: v_dual_bitop2_b32 v2, 32, v0 bitop3:0x40
	v_mov_b32_e32 v9, 0
	s_and_b32 s34, s3, s25
	s_mov_b32 s11, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_and_b32_e32 v119, 0x60, v1
	v_cmp_ne_u32_e32 vcc_lo, 0, v2
	v_cndmask_b32_e64 v115, 0, 1, s34
	v_dual_mov_b32 v8, v9 :: v_dual_mov_b32 v7, v9
	v_dual_mov_b32 v6, v9 :: v_dual_mov_b32 v5, v9
	v_cndmask_b32_e64 v121, 0, 0x70, vcc_lo
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
	v_dual_mov_b32 v34, v9 :: v_dual_mov_b32 v65, v9
	v_dual_mov_b32 v64, v9 :: v_dual_mov_b32 v63, v9
	v_dual_mov_b32 v62, v9 :: v_dual_mov_b32 v61, v9
	v_dual_mov_b32 v60, v9 :: v_dual_mov_b32 v59, v9
	v_dual_mov_b32 v58, v9 :: v_dual_mov_b32 v49, v9
	v_dual_mov_b32 v48, v9 :: v_dual_mov_b32 v47, v9
	v_dual_mov_b32 v46, v9 :: v_dual_mov_b32 v45, v9
	v_dual_mov_b32 v44, v9 :: v_dual_mov_b32 v43, v9
	v_dual_mov_b32 v42, v9 :: v_dual_mov_b32 v73, v9
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
	v_dual_mov_b32 v106, v9 :: v_dual_mov_b32 v57, v9
	v_dual_mov_b32 v56, v9 :: v_dual_mov_b32 v55, v9
	v_dual_mov_b32 v54, v9 :: v_dual_mov_b32 v53, v9
	v_dual_mov_b32 v52, v9 :: v_dual_mov_b32 v51, v9
	v_mov_b32_e32 v50, v9
	s_and_not1_b32 vcc_lo, exec_lo, s17
	s_barrier_wait -1
	s_cbranch_vccnz .LBB0_37
; %bb.15:
	s_mov_b64 s[4:5], src_shared_base
	s_add_co_i32 s6, s41, 0xb280
	s_mov_b32 s7, s5
	v_mul_u32_u24_e32 v3, 0x60, v119
	s_and_b64 s[6:7], s[6:7], 15
	v_and_b32_e32 v5, 16, v0
	s_sub_co_i32 s1, 16, s6
	s_mov_b32 s44, s5
	s_lshr_b32 s1, s1, 2
	s_cmp_lg_u64 s[6:7], 0
	v_or_b32_e32 v2, v3, v5
	s_cselect_b32 s1, s1, 0
	v_and_b32_e32 v7, 15, v0
	s_lshl2_add_u32 s1, s1, s41
	v_mul_u32_u24_e32 v1, 0x60, v121
	s_add_co_i32 s4, s1, 0x11880
	s_add_co_i32 s43, s1, 0xb280
	s_and_b32 s10, s4, 15
	v_mad_u32_u24 v27, 0x60, v7, v2
	s_sub_co_i32 s7, 16, s10
	s_mul_hi_i32 s6, s16, 0x2aaaaaab
	s_lshr_b32 s1, s7, 2
	s_cmp_lg_u64 s[10:11], 0
	v_lshrrev_b32_e32 v4, 4, v27
	s_cselect_b32 s1, s1, 0
	s_lshr_b32 s7, s6, 31
	s_lshl_b32 s10, s1, 2
	s_ashr_i32 s45, s6, 4
	s_add_nc_u64 s[36:37], s[4:5], s[10:11]
	s_movk_i32 s5, 0x600
	v_and_b32_e32 v4, 0x7f8, v4
	v_mad_u32_u24 v9, 0x60, v7, s5
	s_add_co_i32 s45, s45, s7
	s_cmp_lt_i32 s31, 0xe0
	s_mulk_i32 s0, 0xe0
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_nc_u32 v114, v4, v27 :: v_dual_add_nc_u32 v2, v2, v9
	v_or_b32_e32 v6, v1, v5
	v_mul_u32_u24_e32 v8, 0x60, v7
	s_cselect_b32 s46, -1, 0
	s_ashr_i32 s1, s0, 31
	v_lshrrev_b32_e32 v2, 4, v2
	v_mad_u32_u24 v29, 0x60, v7, v6
	v_or_b32_e32 v13, 0x1800, v8
	s_lshl_b32 s4, s30, 7
	v_mov_b32_e32 v117, 0
	v_and_b32_e32 v11, 0x7f8, v2
	v_lshrrev_b32_e32 v10, 4, v29
	v_add_nc_u32_e32 v2, 0xc00, v29
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v37, v117 :: v_dual_add_nc_u32 v36, 0x600, v27
	v_add_nc_u32_e32 v4, v6, v9
	v_and_b32_e32 v10, 0x7f8, v10
	s_delay_alu instid0(VALU_DEP_4)
	v_lshrrev_b32_e32 v8, 4, v2
	s_wait_kmcnt 0x0
	s_bfe_i64 s[6:7], s[20:21], 0x200000
	s_ashr_i32 s5, s4, 31
	v_dual_lshrrev_b32 v12, 4, v4 :: v_dual_add_nc_u32 v118, v10, v29
	v_add_nc_u32_e32 v4, 0x1200, v29
	v_and_b32_e32 v17, 0x7f8, v8
	v_add_nc_u32_e32 v120, v11, v27
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_and_b32_e32 v15, 0x7f8, v12
	v_dual_mov_b32 v11, v117 :: v_dual_add_nc_u32 v10, v6, v13
	v_dual_lshrrev_b32 v12, 4, v4 :: v_dual_bitop2_b32 v14, 32, v5 bitop3:0x54
	v_add_nc_u32_e32 v6, 0x1e00, v29
	v_add_nc_u32_e32 v8, 0x2400, v29
	v_lshrrev_b32_e32 v10, 4, v10
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_and_b32_e32 v19, 0x7f8, v12
	v_dual_add_nc_u32 v124, v17, v29 :: v_dual_bitop2_b32 v12, v3, v14 bitop3:0x54
	v_dual_lshrrev_b32 v18, 4, v8 :: v_dual_lshrrev_b32 v16, 4, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_and_b32_e32 v21, 0x7f8, v10
	v_mad_u32_u24 v10, 0x60, v7, v12
	v_dual_add_nc_u32 v12, v12, v9 :: v_dual_bitop2_b32 v14, v1, v14 bitop3:0x54
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_and_b32_e32 v23, 0x7f8, v16
	v_and_b32_e32 v25, 0x7f8, v18
	v_dual_add_nc_u32 v122, v15, v29 :: v_dual_add_nc_u32 v128, v21, v29
	v_lshrrev_b32_e32 v12, 4, v12
	v_mad_u32_u24 v16, 0x60, v7, v14
	v_dual_lshrrev_b32 v10, 4, v10 :: v_dual_add_nc_u32 v18, v14, v9
	v_add_nc_u32_e32 v14, v14, v13
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_and_b32_e32 v116, 0x7f8, v12
	v_lshrrev_b32_e32 v20, 4, v16
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_and_b32_e32 v31, 0x7f8, v10
	v_dual_mov_b32 v15, v117 :: v_dual_add_nc_u32 v10, 0xc00, v16
	v_lshrrev_b32_e32 v18, 4, v18
	v_add_nc_u32_e32 v12, 0x1200, v16
	v_and_b32_e32 v33, 0x7f8, v20
	v_dual_lshrrev_b32 v20, 4, v10 :: v_dual_bitop2_b32 v5, 64, v5 bitop3:0x54
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_and_b32_e32 v10, 0x7f8, v18
	v_dual_lshrrev_b32 v18, 4, v12 :: v_dual_lshrrev_b32 v22, 4, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_and_b32_e32 v12, 0xff8, v20
	v_dual_mov_b32 v17, v117 :: v_dual_add_nc_u32 v20, 0x1e00, v16
	v_or_b32_e32 v3, v3, v5
	v_and_b32_e32 v14, 0xff8, v18
	v_add_nc_u32_e32 v18, 0x2400, v16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_lshrrev_b32 v20, 4, v20 :: v_dual_bitop2_b32 v1, v1, v5 bitop3:0x54
	v_and_b32_e32 v16, 0x7f8, v22
	v_mad_u32_u24 v22, 0x60, v7, v3
	v_add_nc_u32_e32 v3, v3, v9
	v_dual_add_nc_u32 v126, v19, v29 :: v_dual_add_nc_u32 v132, v25, v29
	v_dual_mov_b32 v19, v117 :: v_dual_lshrrev_b32 v5, 4, v18
	v_and_b32_e32 v18, 0xff8, v20
	v_lshrrev_b32_e32 v22, 4, v22
	v_mad_u32_u24 v7, 0x60, v7, v1
	v_mov_b32_e32 v21, v117
	v_and_b32_e32 v20, 0xff8, v5
	v_dual_add_nc_u32 v5, v1, v9 :: v_dual_lshrrev_b32 v3, 4, v3
	v_and_b32_e32 v35, 0x7f8, v22
	v_dual_lshrrev_b32 v9, 4, v7 :: v_dual_add_nc_u32 v1, v1, v13
	v_add_nc_u32_e32 v24, 0xc00, v7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshrrev_b32_e32 v5, 4, v5
	v_and_b32_e32 v22, 0x7f8, v3
	v_add_nc_u32_e32 v3, 0x1200, v7
	v_and_b32_e32 v38, 0x7f8, v9
	v_lshrrev_b32_e32 v9, 4, v24
	v_and_b32_e32 v24, 0x7f8, v5
	v_add_nc_u32_e32 v5, 0x1e00, v7
	v_lshrrev_b32_e32 v3, 4, v3
	v_add_nc_u32_e32 v7, 0x2400, v7
	v_lshrrev_b32_e32 v1, 4, v1
	v_add_nc_u64_e32 v[134:135], v[116:117], v[36:37]
	v_add_nc_u32_e32 v116, 0x600, v29
	v_and_b32_e32 v28, 0xff8, v3
	v_lshrrev_b32_e32 v3, 4, v5
	v_sub_nc_u32_e32 v5, 0x197f, v0
	v_and_b32_e32 v26, 0xff8, v9
	v_lshrrev_b32_e32 v7, 4, v7
	v_and_b32_e32 v30, 0x7f8, v1
	v_sub_nc_u32_e32 v1, 0x2c9f, v0
	v_and_b32_e32 v32, 0xff8, v3
	v_dual_lshrrev_b32 v3, 8, v5 :: v_dual_add_nc_u32 v136, v31, v27
	v_dual_add_nc_u32 v130, v23, v29 :: v_dual_add_nc_u32 v138, v33, v29
	v_mov_b32_e32 v9, v117
	v_add_nc_u64_e32 v[140:141], v[10:11], v[116:117]
	v_dual_mov_b32 v23, v117 :: v_dual_add_nc_u32 v10, 0x1800, v29
	v_dual_mov_b32 v29, v117 :: v_dual_add_nc_u32 v156, v38, v29
	v_mov_b32_e32 v31, v117
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u64_e32 v[146:147], v[16:17], v[10:11]
	v_add_nc_u32_e32 v3, 1, v3
	s_mul_u64 s[0:1], s[6:7], s[0:1]
	s_bfe_i64 s[6:7], s[12:13], 0x200000
	v_add_nc_u64_e32 v[164:165], v[30:31], v[10:11]
	v_dual_mov_b32 v10, v117 :: v_dual_lshrrev_b32 v1, 8, v1
	v_and_b32_e32 v125, 26, v3
	s_lshl_b64 s[0:1], s[0:1], 1
	s_mul_u64 s[4:5], s[6:7], s[4:5]
	s_add_nc_u64 s[20:21], s[14:15], s[0:1]
	v_add_nc_u32_e32 v5, 1, v1
	s_lshl_b64 s[0:1], s[4:5], 1
	v_and_b32_e32 v34, 0xff8, v7
	s_add_nc_u64 s[38:39], s[18:19], s[0:1]
	v_cmp_ne_u32_e64 s0, v3, v125
	v_dual_mov_b32 v13, v117 :: v_dual_bitop2_b32 v127, 62, v5 bitop3:0x40
	v_dual_mov_b32 v3, v117 :: v_dual_mov_b32 v7, v117
	v_dual_mov_b32 v25, v117 :: v_dual_add_nc_u32 v152, v35, v27
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_ne_u32_e64 s1, v5, v127
	v_dual_mov_b32 v5, v117 :: v_dual_mov_b32 v27, v117
	v_dual_mov_b32 v33, v117 :: v_dual_mov_b32 v35, v117
	v_lshl_or_b32 v39, v125, 8, v0
	v_lshl_or_b32 v40, v127, 8, v0
	v_add_nc_u64_e32 v[142:143], v[12:13], v[2:3]
	v_add_nc_u64_e32 v[144:145], v[14:15], v[4:5]
	v_add_nc_u64_e32 v[148:149], v[18:19], v[6:7]
	v_add_nc_u64_e32 v[150:151], v[20:21], v[8:9]
	v_add_nc_u64_e32 v[154:155], v[22:23], v[36:37]
	v_add_nc_u64_e32 v[158:159], v[24:25], v[116:117]
	v_add_nc_u64_e32 v[160:161], v[26:27], v[2:3]
	v_add_nc_u64_e32 v[162:163], v[28:29], v[4:5]
	v_add_nc_u64_e32 v[166:167], v[32:33], v[6:7]
	v_add_nc_u64_e32 v[168:169], v[34:35], v[8:9]
	v_or_b32_e32 v1, 0x100, v0
	v_dual_mov_b32 v12, v117 :: v_dual_add_nc_u32 v129, 0xffffff00, v39
	v_dual_mov_b32 v171, v117 :: v_dual_lshlrev_b32 v170, 2, v39
	v_dual_mov_b32 v14, v117 :: v_dual_add_nc_u32 v131, 0xffffff00, v40
	v_dual_mov_b32 v173, v117 :: v_dual_lshlrev_b32 v172, 2, v40
	v_dual_mov_b32 v2, v117 :: v_dual_mov_b32 v4, v117
	v_dual_mov_b32 v6, v117 :: v_dual_mov_b32 v8, v117
	v_dual_mov_b32 v16, v117 :: v_dual_mov_b32 v18, v117
	v_dual_mov_b32 v20, v117 :: v_dual_mov_b32 v22, v117
	v_dual_mov_b32 v24, v117 :: v_dual_mov_b32 v26, v117
	v_dual_mov_b32 v28, v117 :: v_dual_mov_b32 v30, v117
	v_dual_mov_b32 v32, v117 :: v_dual_mov_b32 v34, v117
	v_dual_mov_b32 v36, v117 :: v_dual_mov_b32 v38, v117
	v_dual_mov_b32 v39, v117 :: v_dual_mov_b32 v40, v117
	v_dual_mov_b32 v41, v117 :: v_dual_mov_b32 v58, v117
	v_dual_mov_b32 v59, v117 :: v_dual_mov_b32 v60, v117
	v_dual_mov_b32 v61, v117 :: v_dual_mov_b32 v62, v117
	v_dual_mov_b32 v63, v117 :: v_dual_mov_b32 v64, v117
	v_dual_mov_b32 v65, v117 :: v_dual_mov_b32 v42, v117
	v_dual_mov_b32 v43, v117 :: v_dual_mov_b32 v44, v117
	v_dual_mov_b32 v45, v117 :: v_dual_mov_b32 v46, v117
	v_dual_mov_b32 v47, v117 :: v_dual_mov_b32 v48, v117
	v_dual_mov_b32 v49, v117 :: v_dual_mov_b32 v66, v117
	v_dual_mov_b32 v67, v117 :: v_dual_mov_b32 v68, v117
	v_dual_mov_b32 v69, v117 :: v_dual_mov_b32 v70, v117
	v_dual_mov_b32 v71, v117 :: v_dual_mov_b32 v72, v117
	v_dual_mov_b32 v73, v117 :: v_dual_mov_b32 v74, v117
	v_dual_mov_b32 v75, v117 :: v_dual_mov_b32 v76, v117
	v_dual_mov_b32 v77, v117 :: v_dual_mov_b32 v78, v117
	v_dual_mov_b32 v79, v117 :: v_dual_mov_b32 v80, v117
	v_dual_mov_b32 v81, v117 :: v_dual_mov_b32 v82, v117
	v_dual_mov_b32 v83, v117 :: v_dual_mov_b32 v84, v117
	v_dual_mov_b32 v85, v117 :: v_dual_mov_b32 v86, v117
	v_dual_mov_b32 v87, v117 :: v_dual_mov_b32 v88, v117
	v_dual_mov_b32 v89, v117 :: v_dual_mov_b32 v90, v117
	v_dual_mov_b32 v91, v117 :: v_dual_mov_b32 v92, v117
	v_dual_mov_b32 v93, v117 :: v_dual_mov_b32 v94, v117
	v_dual_mov_b32 v95, v117 :: v_dual_mov_b32 v96, v117
	v_dual_mov_b32 v97, v117 :: v_dual_mov_b32 v98, v117
	v_dual_mov_b32 v99, v117 :: v_dual_mov_b32 v100, v117
	v_dual_mov_b32 v101, v117 :: v_dual_mov_b32 v102, v117
	v_dual_mov_b32 v103, v117 :: v_dual_mov_b32 v104, v117
	v_dual_mov_b32 v105, v117 :: v_dual_mov_b32 v106, v117
	v_dual_mov_b32 v107, v117 :: v_dual_mov_b32 v108, v117
	v_dual_mov_b32 v109, v117 :: v_dual_mov_b32 v110, v117
	v_dual_mov_b32 v111, v117 :: v_dual_mov_b32 v112, v117
	v_dual_mov_b32 v113, v117 :: v_dual_mov_b32 v50, v117
	v_dual_mov_b32 v51, v117 :: v_dual_mov_b32 v52, v117
	v_dual_mov_b32 v53, v117 :: v_dual_mov_b32 v54, v117
	v_dual_mov_b32 v55, v117 :: v_dual_mov_b32 v56, v117
	v_mov_b32_e32 v57, v117
	s_lshr_b32 s47, s31, 16
	s_lshr_b32 s48, s29, 16
	s_mov_b32 s42, s35
	s_movk_i32 s16, 0xe0
	s_or_b32 s47, s47, 0x600000
	s_or_b32 s48, s48, 0x600000
	s_mov_b32 s4, 0x7510000
	s_movk_i32 s8, 0x80
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
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_i32 s5, s49, 0xffffffa0
	s_add_co_i32 s6, s5, s26
	s_xor_b32 s5, s50, 1
	s_min_i32 s6, s6, 0x60
	s_cmp_lt_i32 s49, s45
	s_cselect_b32 s10, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_and_b32 s7, s10, exec_lo
	s_cselect_b32 s28, s6, 0
	s_cmp_lt_i32 s28, 0x60
	s_cselect_b32 s6, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b32 s6, s46, s6
	s_or_b32 s6, s40, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s6
	s_cbranch_vccnz .LBB0_29
; %bb.18:                               ;   in Loop: Header=BB0_17 Depth=1
	v_mov_b64_e32 v[174:175], v[0:1]
	v_mov_b32_e32 v133, v125
	s_cmp_lg_u32 s5, 0
	s_mov_b32 s12, 0
	s_cselect_b32 s7, s44, s35
	s_cselect_b32 s6, s43, 0
.LBB0_19:                               ;   Parent Loop BB0_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_dual_mov_b32 v116, v174 :: v_dual_add_nc_u32 v133, -2, v133
	v_dual_mov_b32 v176, v175 :: v_dual_mov_b32 v177, v117
	v_add_nc_u32_e32 v175, 0x200, v175
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u64 v[178:179], v[116:117], 2, s[6:7]
	v_cmp_eq_u32_e32 vcc_lo, 0, v133
	v_add_nc_u32_e32 v174, 0x200, v174
	v_lshl_add_u64 v[176:177], v[176:177], 2, s[6:7]
	s_clause 0x1
	flat_store_b32 v[178:179], v117
	flat_store_b32 v[176:177], v117
	s_or_b32 s12, vcc_lo, s12
	s_wait_xcnt 0x0
	s_and_not1_b32 exec_lo, exec_lo, s12
	s_cbranch_execnz .LBB0_19
; %bb.20:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s12
	s_and_saveexec_b32 s12, s0
	s_cbranch_execz .LBB0_23
; %bb.21:                               ;   in Loop: Header=BB0_17 Depth=1
	v_add_nc_u64_e32 v[174:175], s[6:7], v[170:171]
	v_mov_b32_e32 v116, v129
	s_mov_b32 s6, 0
.LBB0_22:                               ;   Parent Loop BB0_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v116, 0x100, v116
	flat_store_b32 v[174:175], v117
	s_wait_xcnt 0x0
	v_add_nc_u64_e32 v[174:175], 0x400, v[174:175]
	v_cmp_lt_u32_e32 vcc_lo, 0x187f, v116
	s_or_b32 s6, vcc_lo, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s6
	s_cbranch_execnz .LBB0_22
.LBB0_23:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s12
	v_mov_b64_e32 v[174:175], v[0:1]
	v_mov_b32_e32 v133, v127
	s_cmp_lg_u32 s5, 0
	s_mov_b32 s12, 0
	s_cselect_b32 s7, s37, s42
	s_cselect_b32 s6, s36, s41
.LBB0_24:                               ;   Parent Loop BB0_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_dual_mov_b32 v116, v174 :: v_dual_add_nc_u32 v133, -2, v133
	v_dual_mov_b32 v176, v175 :: v_dual_mov_b32 v177, v117
	v_add_nc_u32_e32 v175, 0x200, v175
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u64 v[178:179], v[116:117], 2, s[6:7]
	v_cmp_eq_u32_e32 vcc_lo, 0, v133
	v_add_nc_u32_e32 v174, 0x200, v174
	v_lshl_add_u64 v[176:177], v[176:177], 2, s[6:7]
	s_clause 0x1
	flat_store_b32 v[178:179], v117
	flat_store_b32 v[176:177], v117
	s_or_b32 s12, vcc_lo, s12
	s_wait_xcnt 0x0
	s_and_not1_b32 exec_lo, exec_lo, s12
	s_cbranch_execnz .LBB0_24
; %bb.25:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s12
	s_and_saveexec_b32 s12, s1
	s_cbranch_execz .LBB0_28
; %bb.26:                               ;   in Loop: Header=BB0_17 Depth=1
	v_add_nc_u64_e32 v[174:175], s[6:7], v[172:173]
	v_mov_b32_e32 v116, v131
	s_mov_b32 s6, 0
.LBB0_27:                               ;   Parent Loop BB0_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v116, 0x100, v116
	flat_store_b32 v[174:175], v117
	s_wait_xcnt 0x0
	v_add_nc_u64_e32 v[174:175], 0x400, v[174:175]
	v_cmp_lt_u32_e32 vcc_lo, 0x2b9f, v116
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
	s_and_not1_b32 vcc_lo, exec_lo, s34
	s_cbranch_vccnz .LBB0_16
	s_branch .LBB0_36
.LBB0_32:                               ;   in Loop: Header=BB0_17 Depth=1
	s_mov_b32 s51, exec_lo
	v_cmpx_eq_u32_e32 1, v123
	s_cbranch_execz .LBB0_34
; %bb.33:                               ;   in Loop: Header=BB0_17 Depth=1
	s_cmp_lg_u32 s5, 0
	s_mul_i32 s10, s6, 0x60
	s_cselect_b32 s14, s36, s41
	s_cmp_gt_i32 s28, 0
	s_mov_b32 s30, s28
	s_cselect_b32 s15, -1, 0
	s_lshl_b64 s[12:13], s[10:11], 1
	s_mov_b32 s17, s9
	s_add_nc_u64 s[12:13], s[20:21], s[12:13]
	s_delay_alu instid0(SALU_CYCLE_1)
	v_dual_mov_b32 v133, s14 :: v_dual_mov_b32 v174, s12
	s_and_b32 s10, s13, 0x1ffffff
	s_and_b32 s13, s25, s15
	s_bitset1_b32 s10, 31
	v_cndmask_b32_e64 v116, 0, 1, s13
	v_mov_b32_e32 v135, s10
	v_readfirstlane_b32 s53, v133
	v_readfirstlane_b32 s54, v174
	s_lshr_b64 s[14:15], s[30:31], 16
	v_readfirstlane_b32 s52, v116
	v_readfirstlane_b32 s55, v135
	s_lshl_b32 s13, s28, 16
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
	s_cmp_gt_i32 s28, 0
	s_cselect_b32 s13, -1, 0
	s_lshl_b64 s[6:7], s[10:11], 1
	s_and_b32 s10, s3, s13
	s_add_nc_u64 s[6:7], s[38:39], s[6:7]
	v_cndmask_b32_e64 v116, 0, 1, s10
	s_and_b32 s7, s7, 0x1ffffff
	v_dual_mov_b32 v133, s5 :: v_dual_mov_b32 v174, s6
	s_bitset1_b32 s7, 31
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_readfirstlane_b32 s52, v116
	v_mov_b32_e32 v135, s7
	v_readfirstlane_b32 s53, v133
	v_readfirstlane_b32 s54, v174
	s_lshr_b64 s[6:7], s[28:29], 16
	s_lshl_b32 s5, s28, 16
	v_readfirstlane_b32 s55, v135
	s_mov_b32 s7, s48
	s_mov_b32 s10, s11
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	tensor_load_to_lds s[52:55], s[4:11]
	s_or_b32 exec_lo, exec_lo, s12
	s_and_not1_b32 vcc_lo, exec_lo, s34
	s_cbranch_vccnz .LBB0_16
.LBB0_36:                               ;   in Loop: Header=BB0_17 Depth=1
	s_cmp_lg_u32 s50, 0
	s_cselect_b32 s6, s43, 0
	s_cselect_b32 s5, s36, s41
	v_lshl_add_u32 v116, v114, 1, s6
	v_lshl_add_u32 v133, v118, 1, s5
	v_lshl_add_u32 v135, v138, 1, s5
	v_lshl_add_u32 v137, v140, 1, s5
	v_lshl_add_u32 v139, v142, 1, s5
	ds_load_b128 v[174:177], v116
	ds_load_b128 v[178:181], v116 offset:16
	v_lshl_add_u32 v116, v122, 1, s5
	ds_load_b128 v[182:185], v133
	ds_load_b128 v[186:189], v133 offset:16
	v_lshl_add_u32 v133, v124, 1, s5
	v_lshl_add_u32 v141, v144, 1, s5
	ds_load_b128 v[190:193], v116 offset:3072
	ds_load_b128 v[194:197], v116 offset:3088
	v_lshl_add_u32 v116, v120, 1, s6
	ds_load_b128 v[206:209], v133 offset:6144
	v_lshl_add_u32 v143, v146, 1, s5
	v_lshl_add_u32 v145, v148, 1, s5
	v_lshl_add_u32 v147, v150, 1, s5
	ds_load_b128 v[198:201], v116 offset:3072
	ds_load_b128 v[202:205], v116 offset:3088
	v_lshl_add_u32 v116, v128, 1, s5
	ds_load_b128 v[230:233], v137 offset:64
	ds_load_b128 v[234:237], v137 offset:80
	ds_load_b128 v[238:241], v139 offset:64
	s_wait_dscnt 0x8
	v_wmma_f32_16x16x32_bf16 v[106:113], v[174:181], v[182:189], v[106:113]
	ds_load_b128 v[242:245], v139 offset:80
	ds_load_b128 v[246:249], v143 offset:64
	ds_load_b128 v[250:253], v143 offset:80
	ds_load_b128 v[254:257], v145 offset:64
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[2:5] /*v[258:261]*/, v145 offset:80
	ds_load_b128 v[6:9] /*v[262:265]*/, v147 offset:64
	ds_load_b128 v[10:13] /*v[266:269]*/, v147 offset:80
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_wait_dscnt 0xd
	v_wmma_f32_16x16x32_bf16 v[98:105], v[174:181], v[190:197], v[98:105] matrix_a_reuse
	; sched_group_barrier mask(0x00000100) size(9) SyncID(0)
	s_wait_dscnt 0xa
	v_wmma_f32_16x16x32_bf16 v[34:41], v[198:205], v[190:197], v[34:41] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[58:65], v[198:205], v[182:189], v[58:65] matrix_a_reuse
	ds_load_b128 v[210:213], v133 offset:6160
	v_lshl_add_u32 v133, v132, 1, s5
	ds_load_b128 v[222:225], v133 offset:18432
	ds_load_b128 v[226:229], v133 offset:18448
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[42:49], v[174:181], v[222:229], v[42:49] matrix_a_reuse
	v_lshl_add_u32 v133, v134, 1, s6
	v_wmma_f32_16x16x32_bf16 v[50:57], v[198:205], v[222:229], v[50:57]
	ds_load_b128 v[182:185], v116 offset:12288
	ds_load_b128 v[186:189], v116 offset:12304
	v_lshl_add_u32 v116, v126, 1, s5
	; sched_group_barrier mask(0x00000008) size(7) SyncID(0)
	ds_load_b128 v[190:193], v116 offset:9216
	ds_load_b128 v[194:197], v116 offset:9232
	v_lshl_add_u32 v116, v130, 1, s5
	ds_load_b128 v[214:217], v116 offset:15360
	ds_load_b128 v[218:221], v116 offset:15376
	v_lshl_add_u32 v116, v136, 1, s6
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[10:17], v[198:205], v[182:189], v[10:17] matrix_a_reuse
	; sched_group_barrier mask(0x00000100) size(9) SyncID(0)
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[18:25], v[198:205], v[190:197], v[18:25] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[26:33], v[198:205], v[206:213], v[26:33] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[90:97], v[174:181], v[206:213], v[90:97] matrix_a_reuse
	ds_load_b128 v[206:209], v135 offset:64
	ds_load_b128 v[210:213], v135 offset:80
	v_wmma_f32_16x16x32_bf16 v[82:89], v[174:181], v[190:197], v[82:89] matrix_a_reuse
	ds_load_b128 v[190:193], v133 offset:64
	ds_load_b128 v[194:197], v133 offset:80
	v_wmma_f32_16x16x32_bf16 v[74:81], v[174:181], v[182:189], v[74:81] matrix_a_reuse
	ds_load_b128 v[182:185], v116 offset:64
	ds_load_b128 v[186:189], v116 offset:80
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[66:73], v[174:181], v[214:221], v[66:73] matrix_a_reuse
	ds_load_b128 v[174:177], v141 offset:64
	ds_load_b128 v[178:181], v141 offset:80
	; sched_group_barrier mask(0x00000008) size(7) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[2:9], v[198:205], v[214:221], v[2:9] matrix_a_reuse
	; sched_barrier mask(0x00000000)
	v_lshl_add_u32 v116, v152, 1, s6
	v_lshl_add_u32 v133, v154, 1, s6
	v_lshl_add_u32 v135, v156, 1, s5
	ds_load_b128 v[198:201], v116 offset:128
	ds_load_b128 v[202:205], v116 offset:144
	v_lshl_add_u32 v116, v158, 1, s5
	ds_load_b128 v[214:217], v133 offset:128
	ds_load_b128 v[218:221], v133 offset:144
	v_lshl_add_u32 v133, v160, 1, s5
	ds_load_b128 v[222:225], v135 offset:128
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[14:17] /*v[270:273]*/, v116 offset:128
	ds_load_b128 v[18:21] /*v[274:277]*/, v116 offset:144
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v116, v162, 1, s5
	ds_load_b128 v[226:229], v135 offset:144
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[22:25] /*v[278:281]*/, v133 offset:128
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_wait_dscnt 0xb
	v_wmma_f32_16x16x32_bf16 v[106:113], v[182:189], v[206:213], v[106:113]
	v_lshl_add_u32 v135, v164, 1, s5
	; sched_group_barrier mask(0x00000100) size(9) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[98:105], v[182:189], v[230:237], v[98:105] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[90:97], v[182:189], v[238:245], v[90:97] matrix_a_reuse
	s_wait_dscnt 0x9
	v_wmma_f32_16x16x32_bf16 v[82:89], v[182:189], v[174:181], v[82:89] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[74:81], v[182:189], v[246:253], v[74:81] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[66:73], v[182:189], v[254:261], v[66:73] matrix_a_reuse
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[42:49], v[182:189], v[6:13] /*v[262:269]*/, v[42:49] matrix_a_reuse
	s_set_vgpr_msb 0x440                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[26:29] /*v[282:285]*/, v133 offset:144
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	ds_load_b128 v[182:185], v116 offset:128
	v_lshl_add_u32 v133, v166, 1, s5
	ds_load_b128 v[186:189], v116 offset:144
	v_lshl_add_u32 v116, v168, 1, s5
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[30:33] /*v[286:289]*/, v135 offset:128
	ds_load_b128 v[34:37] /*v[290:293]*/, v135 offset:144
	ds_load_b128 v[38:41] /*v[294:297]*/, v133 offset:128
	ds_load_b128 v[42:45] /*v[298:301]*/, v133 offset:144
	ds_load_b128 v[46:49] /*v[302:305]*/, v116 offset:128
	ds_load_b128 v[50:53] /*v[306:309]*/, v116 offset:144
	s_set_vgpr_msb 0x4004                   ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[50:57], v[190:197], v[6:13] /*v[262:269]*/, v[50:57]
	; sched_group_barrier mask(0x00000008) size(7) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(9) SyncID(0)
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_wmma_f32_16x16x32_bf16 v[2:9], v[190:197], v[254:261], v[2:9] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[10:17], v[190:197], v[246:253], v[10:17] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[18:25], v[190:197], v[174:181], v[18:25] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[26:33], v[190:197], v[238:245], v[26:33] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[34:41], v[190:197], v[230:237], v[34:41] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[58:65], v[190:197], v[206:213], v[58:65] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(7) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0xa
	v_wmma_f32_16x16x32_bf16 v[106:113], v[198:205], v[222:229], v[106:113]
	; sched_group_barrier mask(0x00000100) size(9) SyncID(0)
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[98:105], v[198:205], v[14:21] /*v[270:277]*/, v[98:105] matrix_a_reuse
	s_wait_dscnt 0x8
	v_wmma_f32_16x16x32_bf16 v[90:97], v[198:205], v[22:29] /*v[278:285]*/, v[90:97] matrix_a_reuse
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[82:89], v[198:205], v[182:189], v[82:89] matrix_a_reuse
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[74:81], v[198:205], v[30:37] /*v[286:293]*/, v[74:81] matrix_a_reuse
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[66:73], v[198:205], v[38:45] /*v[294:301]*/, v[66:73] matrix_a_reuse
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[42:49], v[198:205], v[46:53] /*v[302:309]*/, v[42:49] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(7) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(9) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[50:57], v[214:221], v[46:53] /*v[302:309]*/, v[50:57]
	v_wmma_f32_16x16x32_bf16 v[2:9], v[214:221], v[38:45] /*v[294:301]*/, v[2:9] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[10:17], v[214:221], v[30:37] /*v[286:293]*/, v[10:17] matrix_a_reuse
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_wmma_f32_16x16x32_bf16 v[18:25], v[214:221], v[182:189], v[18:25] matrix_a_reuse
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[26:33], v[214:221], v[22:29] /*v[278:285]*/, v[26:33] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[34:41], v[214:221], v[14:21] /*v[270:277]*/, v[34:41] matrix_a_reuse
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_wmma_f32_16x16x32_bf16 v[58:65], v[214:221], v[222:229], v[58:65] matrix_a_reuse
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
	s_and_b32 vcc_lo, exec_lo, s34
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_vccz .LBB0_39
; %bb.38:
	v_lshrrev_b32_e32 v1, 1, v0
	v_and_or_b32 v114, v0, 15, v121
	v_cvt_pk_bf16_f32 v105, v104, v105
	v_cvt_pk_bf16_f32 v104, v102, v103
	v_cvt_pk_bf16_f32 v102, v98, v99
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_lshlrev_b32 v114, 7, v114 :: v_dual_bitop2_b32 v1, 8, v1 bitop3:0x40
	v_cvt_pk_bf16_f32 v97, v96, v97
	v_cvt_pk_bf16_f32 v96, v94, v95
	v_cvt_pk_bf16_f32 v94, v90, v91
	v_cvt_pk_bf16_f32 v95, v92, v93
	v_or3_b32 v1, v119, v1, v114
	v_lshrrev_b32_e32 v91, 3, v114
	v_cvt_pk_bf16_f32 v113, v112, v113
	v_cvt_pk_bf16_f32 v112, v110, v111
	v_cvt_pk_bf16_f32 v111, v108, v109
	v_add_nc_u32_e32 v98, 0x800, v1
	v_add_nc_u32_e32 v90, 0x1000, v1
	v_lshlrev_b32_e32 v93, 1, v1
	v_add_nc_u32_e32 v99, 0x1800, v1
	v_cvt_pk_bf16_f32 v110, v106, v107
	v_lshrrev_b32_e32 v92, 3, v98
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_lshrrev_b32 v90, 3, v90 :: v_dual_add_nc_u32 v91, v91, v93
	v_lshrrev_b32_e32 v99, 3, v99
	v_cvt_pk_bf16_f32 v103, v100, v101
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_and_b32_e32 v92, 0xff0, v92
	v_and_b32_e32 v90, 0xff0, v90
	v_cvt_pk_bf16_f32 v65, v64, v65
	v_cvt_pk_bf16_f32 v64, v62, v63
	v_cvt_pk_bf16_f32 v62, v58, v59
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_nc_u32 v92, v92, v93 :: v_dual_add_nc_u32 v90, v90, v93
	v_add_nc_u32_e32 v58, 0x2800, v1
	ds_store_b128 v91, v[110:113]
	ds_store_b128 v92, v[102:105] offset:4096
	v_and_b32_e32 v91, 0xff0, v99
	v_cvt_pk_bf16_f32 v63, v60, v61
	ds_store_b128 v90, v[94:97] offset:8192
	v_cvt_pk_bf16_f32 v61, v88, v89
	v_cvt_pk_bf16_f32 v60, v86, v87
	v_dual_add_nc_u32 v90, v91, v93 :: v_dual_lshrrev_b32 v86, 3, v58
	v_cvt_pk_bf16_f32 v59, v84, v85
	v_cvt_pk_bf16_f32 v58, v82, v83
	v_cvt_pk_bf16_f32 v41, v40, v41
	v_cvt_pk_bf16_f32 v40, v38, v39
	v_cvt_pk_bf16_f32 v39, v36, v37
	v_add_nc_u32_e32 v36, 0x1810, v1
	v_add_nc_u32_e32 v100, 0x2000, v1
	v_cvt_pk_bf16_f32 v81, v80, v81
	v_cvt_pk_bf16_f32 v80, v78, v79
	v_add_nc_u32_e32 v84, 0x3000, v1
	v_cvt_pk_bf16_f32 v79, v76, v77
	v_add_nc_u32_e32 v76, 0x810, v1
	ds_store_b128 v90, v[58:61] offset:12288
	v_add_nc_u32_e32 v61, 0x1010, v1
	v_add_nc_u32_e32 v98, 16, v1
	v_cvt_pk_bf16_f32 v38, v34, v35
	v_lshrrev_b32_e32 v35, 3, v36
	v_add_nc_u32_e32 v36, 0x2010, v1
	v_cvt_pk_bf16_f32 v33, v32, v33
	v_cvt_pk_bf16_f32 v32, v30, v31
	v_cvt_pk_bf16_f32 v31, v28, v29
	v_add_nc_u32_e32 v28, 0x2810, v1
	v_add_nc_u32_e32 v1, 0x3010, v1
	v_lshrrev_b32_e32 v92, 3, v100
	v_cvt_pk_bf16_f32 v78, v74, v75
	v_lshrrev_b32_e32 v75, 3, v84
	v_lshrrev_b32_e32 v58, 3, v76
	v_cvt_pk_bf16_f32 v49, v48, v49
	v_cvt_pk_bf16_f32 v48, v46, v47
	v_cvt_pk_bf16_f32 v46, v42, v43
	v_dual_lshrrev_b32 v43, 3, v61 :: v_dual_lshrrev_b32 v101, 3, v98
	v_cvt_pk_bf16_f32 v30, v26, v27
	v_lshrrev_b32_e32 v26, 3, v36
	v_cvt_pk_bf16_f32 v25, v24, v25
	v_cvt_pk_bf16_f32 v24, v22, v23
	v_cvt_pk_bf16_f32 v23, v20, v21
	v_dual_lshrrev_b32 v20, 3, v28 :: v_dual_lshrrev_b32 v1, 3, v1
	v_and_b32_e32 v88, 0xff0, v92
	v_cvt_pk_bf16_f32 v73, v72, v73
	v_cvt_pk_bf16_f32 v72, v70, v71
	v_cvt_pk_bf16_f32 v70, v66, v67
	v_and_b32_e32 v66, 0xff0, v75
	v_and_b32_e32 v58, 0xff0, v58
	v_lshlrev_b32_e32 v60, 1, v98
	v_and_b32_e32 v34, 0xff0, v43
	v_and_b32_e32 v101, 0xff0, v101
	v_and_b32_e32 v83, 0xff0, v86
	v_and_b32_e32 v35, 0xff0, v35
	v_and_b32_e32 v26, 0xff0, v26
	v_cvt_pk_bf16_f32 v22, v18, v19
	v_and_b32_e32 v19, 0xff0, v20
	v_and_b32_e32 v1, 0xff0, v1
	v_dual_add_nc_u32 v82, v88, v93 :: v_dual_add_nc_u32 v59, v66, v93
	v_cvt_pk_bf16_f32 v47, v44, v45
	v_add_nc_u32_e32 v42, v58, v60
	v_dual_add_nc_u32 v34, v34, v60 :: v_dual_add_nc_u32 v99, v101, v93
	v_add_nc_u32_e32 v74, v83, v93
	v_cvt_pk_bf16_f32 v71, v68, v69
	v_dual_add_nc_u32 v27, v35, v60 :: v_dual_add_nc_u32 v18, v26, v60
	v_cvt_pk_bf16_f32 v17, v16, v17
	v_cvt_pk_bf16_f32 v16, v14, v15
	v_cvt_pk_bf16_f32 v15, v12, v13
	v_cvt_pk_bf16_f32 v14, v10, v11
	v_add_nc_u32_e32 v10, v19, v60
	v_cvt_pk_bf16_f32 v9, v8, v9
	v_cvt_pk_bf16_f32 v8, v6, v7
	v_cvt_pk_bf16_f32 v7, v4, v5
	v_cvt_pk_bf16_f32 v6, v2, v3
	v_add_nc_u32_e32 v1, v1, v60
	v_cvt_pk_bf16_f32 v5, v56, v57
	v_cvt_pk_bf16_f32 v4, v54, v55
	v_cvt_pk_bf16_f32 v3, v52, v53
	v_cvt_pk_bf16_f32 v2, v50, v51
	ds_store_b128 v82, v[78:81] offset:16384
	ds_store_b128 v74, v[70:73] offset:20480
	ds_store_b128 v59, v[46:49] offset:24576
	ds_store_b128 v99, v[62:65] offset:32
	ds_store_b128 v42, v[38:41] offset:4096
	ds_store_b128 v34, v[30:33] offset:8192
	ds_store_b128 v27, v[22:25] offset:12288
	ds_store_b128 v18, v[14:17] offset:16384
	ds_store_b128 v10, v[6:9] offset:20480
	ds_store_b128 v1, v[2:5] offset:24576
.LBB0_39:
	v_cmp_ne_u32_e32 vcc_lo, 1, v115
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
	s_mul_i32 s15, s33, 0xe0
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
	v_mul_u64_e32 v[18:19], s[24:25], v[18:19]
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
	v_dual_sub_nc_u32 v7, v4, v9 :: v_dual_lshlrev_b32 v4, 7, v4
	v_lshlrev_b32_e32 v9, 7, v9
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
	.size	bm128_bn224_bk096_wm4_wn2_mc0, .Lfunc_end0-bm128_bn224_bk096_wm4_wn2_mc0
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel bm128_bn224_bk096_wm4_wn2_mc0
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
		.amdhsa_next_free_vgpr 310
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
		.amdhsa_inst_pref_size 63
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text.bm128_bn224_bk096_wm4_wn2_mc0,"axG",@progbits,bm128_bn224_bk096_wm4_wn2_mc0,comdat
                                        ; -- End function
	.set .Lbm128_bn224_bk096_wm4_wn2_mc0.num_vgpr, 310
	.set .Lbm128_bn224_bk096_wm4_wn2_mc0.num_agpr, 0
	.set .Lbm128_bn224_bk096_wm4_wn2_mc0.numbered_sgpr, 56
	.set .Lbm128_bn224_bk096_wm4_wn2_mc0.num_named_barrier, 0
	.set .Lbm128_bn224_bk096_wm4_wn2_mc0.private_seg_size, 0
	.set .Lbm128_bn224_bk096_wm4_wn2_mc0.uses_vcc, 1
	.set .Lbm128_bn224_bk096_wm4_wn2_mc0.uses_flat_scratch, 1
	.set .Lbm128_bn224_bk096_wm4_wn2_mc0.has_dyn_sized_stack, 0
	.set .Lbm128_bn224_bk096_wm4_wn2_mc0.has_recursion, 0
	.set .Lbm128_bn224_bk096_wm4_wn2_mc0.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 7956
; TotalNumSgprs: 58
; NumVgprs: 310
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 143616 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 19
; NumSGPRsForWavesPerEU: 58
; NumVGPRsForWavesPerEU: 310
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
	.type	__hip_cuid_4514cbf632f24bc9,@object ; @__hip_cuid_4514cbf632f24bc9
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_4514cbf632f24bc9
__hip_cuid_4514cbf632f24bc9:
	.byte	0                               ; 0x0
	.size	__hip_cuid_4514cbf632f24bc9, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git aa451e1fe6a793394d6733051b1778633063ae96+PATCHED:d17c5aa0e3ea29cde402f58f27e39b6034effa27)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_4514cbf632f24bc9
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
    macrotile: [128, 224, 96]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, One]
  MatrixInstruction: [16, 16, 32, 1]
  EnableMatrixInstruction: True
  MIWaveTile: [4, 3]
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
    .group_segment_fixed_size: 143616
    .kernarg_segment_align: 8
    .kernarg_segment_size: 132
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           bm128_bn224_bk096_wm4_wn2_mc0
    .private_segment_fixed_size: 0
    .sgpr_count:     58
    .sgpr_spill_count: 0
    .symbol:         bm128_bn224_bk096_wm4_wn2_mc0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     310
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
