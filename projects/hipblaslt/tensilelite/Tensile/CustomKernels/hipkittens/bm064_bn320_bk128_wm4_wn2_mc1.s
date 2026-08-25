	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.section	.text.bm064_bn320_bk128_wm4_wn2_mc1,"axG",@progbits,bm064_bn320_bk128_wm4_wn2_mc1,comdat
	.protected	bm064_bn320_bk128_wm4_wn2_mc1 ; -- Begin function bm064_bn320_bk128_wm4_wn2_mc1
	.globl	bm064_bn320_bk128_wm4_wn2_mc1
	.p2align	8
	.type	bm064_bn320_bk128_wm4_wn2_mc1,@function
bm064_bn320_bk128_wm4_wn2_mc1: ; @bm064_bn320_bk128_wm4_wn2_mc1
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
	s_bfe_u32 s3, ttmp6, 0x40004
	s_lshl2_add_u32 s16, ttmp9, s2
	s_lshl2_add_u32 s33, ttmp7, s3
	s_lshl_b32 s2, s16, 6
	s_wait_kmcnt 0x0
	s_add_co_i32 s3, s28, 63
	s_sub_co_i32 s6, s28, s2
	s_ashr_i32 s7, s3, 31
	s_add_co_i32 s5, s29, 0x13f
	s_min_i32 s31, s6, 64
	s_lshr_b32 s6, s7, 26
	s_mul_hi_i32 s5, s5, 0x66666667
	s_add_co_i32 s3, s3, s6
	s_lshr_b32 s7, s5, 31
	s_ashr_i32 s6, s5, 7
	s_ashr_i32 s5, s3, 6
	s_add_co_i32 s6, s6, s7
	s_cmp_lt_i32 s16, s5
	s_mul_i32 s7, s33, 0xfffffec0
	s_cselect_b32 s3, -1, 0
	s_mov_b32 s9, s30
	s_and_b32 s8, s3, exec_lo
	s_cselect_b32 s35, s31, 0
	s_add_co_i32 s7, s29, s7
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_min_i32 s7, s7, 0x140
	s_cmp_lt_i32 s33, s6
	s_cselect_b32 s29, -1, 0
	s_and_b32 s8, s29, exec_lo
	s_cselect_b32 s37, s7, 0
	s_add_co_i32 s12, s30, 0x7f
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 2, 1), 1
	s_min_i32 s7, s30, 0x80
	s_cmp_gt_i32 s12, 0x7f
	s_cselect_b32 s13, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_and_b32 s8, s13, exec_lo
	s_cselect_b32 s34, s7, 0
	s_cmp_lt_i32 s35, 64
	s_cselect_b32 s40, -1, 0
	s_and_b32 vcc_lo, exec_lo, s40
	s_mov_b32 s7, s40
	s_cbranch_vccnz .LBB0_2
; %bb.1:
	s_cmp_lt_i32 s37, 0x140
	s_cselect_b32 s7, -1, 0
	s_cmp_lt_i32 s34, 0x80
	s_cselect_b32 s8, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 s7, s8, s7
.LBB0_2:
	v_lshlrev_b32_e32 v84, 2, v0
	s_and_not1_b32 vcc_lo, exec_lo, s7
	s_cbranch_vccnz .LBB0_8
; %bb.3:
	v_or_b32_e32 v1, 0xffffff00, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v3, v84
	s_mov_b32 s7, 0
	v_mov_b32_e32 v4, v1
.LBB0_4:                                ; =>This Inner Loop Header: Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)
	v_add_nc_u32_e32 v4, 0x100, v4
	ds_store_b32 v3, v2
	v_add_nc_u32_e32 v3, 0x400, v3
	v_cmp_lt_u32_e32 vcc_lo, 0xfff, v4
	s_or_b32 s7, vcc_lo, s7
	s_and_not1_b32 exec_lo, exec_lo, s7
	s_cbranch_execnz .LBB0_4
; %bb.5:
	s_or_b32 exec_lo, exec_lo, s7
	v_lshl_add_u32 v2, s4, 2, v84
	v_mov_b32_e32 v3, 0
	s_mov_b32 s7, 0
.LBB0_6:                                ; =>This Inner Loop Header: Depth=1
	v_add_nc_u32_e32 v1, 0x100, v1
	ds_store_b32 v2, v3 offset:17408
	v_add_nc_u32_e32 v2, 0x400, v2
	v_cmp_lt_u32_e32 vcc_lo, 0x53ff, v1
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
	s_mov_b64 s[0:1], src_shared_base
	s_lshl_b32 s0, s4, 2
	v_lshrrev_b32_e32 v89, 5, v0
	s_or_b32 s41, s0, 0x4400
	s_lshl_b32 s0, s33, 2
	s_add_co_i32 s6, s6, -1
	s_and_b32 s0, s0, 12
	s_add_co_i32 s18, s5, -1
	s_min_i32 s17, s33, s6
	s_and_b32 s38, s16, 3
	s_lshl_b32 s0, 15, s0
	s_mov_b32 s4, exec_lo
	v_cmpx_lt_i32_e32 0, v89
	s_xor_b32 s19, exec_lo, s4
	s_cbranch_execz .LBB0_12
; %bb.9:
	s_mov_b32 s39, exec_lo
	v_cmpx_eq_u32_e32 1, v89
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
	v_dual_mov_b32 v1, s41 :: v_dual_mov_b32 v4, s4
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
	s_or_b32 s7, s11, 0x800000
	s_mov_b32 s11, s10
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[44:47], s[4:11]
.LBB0_11:
	s_or_b32 exec_lo, exec_lo, s39
.LBB0_12:
	s_or_saveexec_b32 s36, s19
	s_min_i32 s19, s16, s18
	s_lshl_b32 s18, 0x1111, s38
	s_xor_b32 exec_lo, exec_lo, s36
	s_cbranch_execz .LBB0_14
; %bb.13:
	s_cmp_gt_i32 s34, 0
	s_mov_b32 s10, 0
	s_cselect_b32 s8, -1, 0
	s_lshl_b32 s4, s19, 6
	s_wait_kmcnt 0x0
	s_bfe_i64 s[6:7], s[24:25], 0x200000
	s_ashr_i32 s5, s4, 31
	s_and_b32 s8, s3, s8
	s_mul_u64 s[4:5], s[6:7], s[4:5]
	v_cndmask_b32_e64 v2, 0, 1, s8
	s_lshl_b64 s[6:7], s[4:5], 1
	s_lshr_b32 s8, s35, 16
	s_add_nc_u64 s[6:7], s[14:15], s[6:7]
	s_or_b32 s4, s18, 0x7510000
	s_and_b32 s7, s7, 0x1ffffff
	v_readfirstlane_b32 s44, v2
	s_bitset1_b32 s7, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v4, s6 :: v_dual_mov_b32 v3, s7
	s_lshr_b64 s[6:7], s[34:35], 16
	s_lshl_b32 s5, s34, 16
	s_or_b32 s7, s8, 0x800000
	v_readfirstlane_b32 s46, v4
	v_readfirstlane_b32 s47, v3
	s_mov_b32 s8, 64
	s_mov_b32 s11, s10
	s_mov_b32 s45, s10
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[44:47], s[4:11]
.LBB0_14:
	s_or_b32 exec_lo, exec_lo, s36
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s4, exec_lo
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	v_cmpx_gt_u32_e32 32, v0
	s_cbranch_execz .LBB0_16
; %bb.15:
	s_barrier_signal -3
.LBB0_16:
	s_or_b32 exec_lo, exec_lo, s4
	v_dual_lshlrev_b32 v1, 3, v89 :: v_dual_bitop2_b32 v2, 32, v0 bitop3:0x40
	v_mov_b32_e32 v9, 0
	s_and_b32 s42, s3, s29
	s_mov_b32 s11, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_and_b32_e32 v87, 48, v1
	v_cmp_ne_u32_e32 vcc_lo, 0, v2
	v_cndmask_b32_e64 v83, 0, 1, s42
	v_dual_mov_b32 v8, v9 :: v_dual_mov_b32 v7, v9
	v_dual_mov_b32 v6, v9 :: v_dual_mov_b32 v5, v9
	v_cndmask_b32_e64 v91, 0, 0xa0, vcc_lo
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
	v_dual_mov_b32 v34, v9 :: v_dual_mov_b32 v57, v9
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
	v_dual_mov_b32 v74, v9 :: v_dual_mov_b32 v49, v9
	v_dual_mov_b32 v48, v9 :: v_dual_mov_b32 v47, v9
	v_dual_mov_b32 v46, v9 :: v_dual_mov_b32 v45, v9
	v_dual_mov_b32 v44, v9 :: v_dual_mov_b32 v43, v9
	v_mov_b32_e32 v42, v9
	s_and_not1_b32 vcc_lo, exec_lo, s13
	s_barrier_wait -1
	s_barrier_wait -3
	s_cbranch_vccnz .LBB0_41
; %bb.17:
	s_mov_b64 s[4:5], src_shared_base
	s_add_co_i32 s6, s41, 0x15400
	s_mov_b32 s7, s5
	v_dual_lshlrev_b32 v2, 7, v91 :: v_dual_lshlrev_b32 v1, 7, v0
	s_and_b64 s[38:39], s[6:7], 15
	v_or_b32_e32 v93, 0xf00, v0
	s_sub_co_i32 s4, 16, s38
	v_or_b32_e32 v94, 0x4000, v84
	s_lshr_b32 s4, s4, 2
	v_and_b32_e32 v1, 0x780, v1
	s_cmp_lg_u64 s[38:39], 0
	v_or_b32_e32 v100, 0x5300, v0
	s_cselect_b32 s4, s4, 0
	v_or_b32_e32 v96, 0x15000, v84
	s_lshl2_add_u32 s44, s4, s6
	v_and_or_b32 v4, v0, 16, v1
	v_lshlrev_b32_e32 v3, 7, v87
	s_add_co_i32 s4, s44, 0x4400
	s_mul_i32 s6, s17, 0x140
	s_and_b32 s10, s4, 15
	v_or_b32_e32 v6, v2, v4
	v_or_b32_e32 v5, v3, v4
	s_sub_co_i32 s7, 16, s10
	v_or_b32_e32 v1, v1, v3
	s_lshr_b32 s7, s7, 2
	s_cmp_lg_u64 s[10:11], 0
	v_dual_lshrrev_b32 v8, 4, v6 :: v_dual_lshrrev_b32 v7, 4, v5
	s_cselect_b32 s7, s7, 0
	s_ashr_i32 s8, s12, 31
	s_lshl_b32 s10, s7, 2
	s_lshr_b32 s8, s8, 25
	v_and_b32_e32 v7, 0x1f8, v7
	s_add_co_i32 s12, s12, s8
	v_and_b32_e32 v8, 0x578, v8
	s_ashr_i32 s46, s12, 7
	s_cmp_lt_i32 s37, 0x140
	s_add_nc_u64 s[38:39], s[4:5], s[10:11]
	s_cselect_b32 s47, -1, 0
	s_ashr_i32 s7, s6, 31
	s_lshl_b32 s50, s19, 6
	s_or_b32 s4, s18, 0x7510000
	v_dual_mov_b32 v85, 0 :: v_dual_add_nc_u32 v82, v7, v5
	v_dual_add_nc_u32 v86, v8, v6 :: v_dual_lshrrev_b32 v5, 4, v1
	s_wait_kmcnt 0x0
	s_bfe_i64 s[18:19], s[20:21], 0x200000
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_nc_u32 v3, v4, v3 :: v_dual_mov_b32 v95, v85
	s_ashr_i32 s51, s50, 31
	s_mul_u64 s[6:7], s[18:19], s[6:7]
	s_bfe_i64 s[18:19], s[24:25], 0x200000
	s_or_b32 s12, s0, 0x7510000
	s_mul_u64 s[18:19], s[18:19], s[50:51]
	v_cmp_eq_u32_e64 s0, 0, v89
	v_or_b32_e32 v1, 0x100, v0
	v_dual_add_nc_u32 v88, v5, v3 :: v_dual_add_nc_u32 v92, v7, v3
	v_add3_u32 v90, v4, v2, v8
	v_dual_mov_b32 v97, v85 :: v_dual_mov_b32 v2, v85
	v_dual_mov_b32 v3, v85 :: v_dual_mov_b32 v4, v85
	v_dual_mov_b32 v5, v85 :: v_dual_mov_b32 v6, v85
	v_dual_mov_b32 v7, v85 :: v_dual_mov_b32 v8, v85
	v_dual_mov_b32 v9, v85 :: v_dual_mov_b32 v10, v85
	v_dual_mov_b32 v11, v85 :: v_dual_mov_b32 v12, v85
	v_dual_mov_b32 v13, v85 :: v_dual_mov_b32 v14, v85
	v_dual_mov_b32 v15, v85 :: v_dual_mov_b32 v16, v85
	v_dual_mov_b32 v17, v85 :: v_dual_mov_b32 v18, v85
	v_dual_mov_b32 v19, v85 :: v_dual_mov_b32 v20, v85
	v_dual_mov_b32 v21, v85 :: v_dual_mov_b32 v22, v85
	v_dual_mov_b32 v23, v85 :: v_dual_mov_b32 v24, v85
	v_dual_mov_b32 v25, v85 :: v_dual_mov_b32 v26, v85
	v_dual_mov_b32 v27, v85 :: v_dual_mov_b32 v28, v85
	v_dual_mov_b32 v29, v85 :: v_dual_mov_b32 v30, v85
	v_dual_mov_b32 v31, v85 :: v_dual_mov_b32 v32, v85
	v_dual_mov_b32 v33, v85 :: v_dual_mov_b32 v34, v85
	v_dual_mov_b32 v35, v85 :: v_dual_mov_b32 v36, v85
	v_dual_mov_b32 v37, v85 :: v_dual_mov_b32 v38, v85
	v_dual_mov_b32 v39, v85 :: v_dual_mov_b32 v40, v85
	v_dual_mov_b32 v41, v85 :: v_dual_mov_b32 v50, v85
	v_dual_mov_b32 v51, v85 :: v_dual_mov_b32 v52, v85
	v_dual_mov_b32 v53, v85 :: v_dual_mov_b32 v54, v85
	v_dual_mov_b32 v55, v85 :: v_dual_mov_b32 v56, v85
	v_dual_mov_b32 v57, v85 :: v_dual_mov_b32 v58, v85
	v_dual_mov_b32 v59, v85 :: v_dual_mov_b32 v60, v85
	v_dual_mov_b32 v61, v85 :: v_dual_mov_b32 v62, v85
	v_dual_mov_b32 v63, v85 :: v_dual_mov_b32 v64, v85
	v_dual_mov_b32 v65, v85 :: v_dual_mov_b32 v66, v85
	v_dual_mov_b32 v67, v85 :: v_dual_mov_b32 v68, v85
	v_dual_mov_b32 v69, v85 :: v_dual_mov_b32 v70, v85
	v_dual_mov_b32 v71, v85 :: v_dual_mov_b32 v72, v85
	v_dual_mov_b32 v73, v85 :: v_dual_mov_b32 v74, v85
	v_dual_mov_b32 v75, v85 :: v_dual_mov_b32 v76, v85
	v_dual_mov_b32 v77, v85 :: v_dual_mov_b32 v78, v85
	v_dual_mov_b32 v79, v85 :: v_dual_mov_b32 v80, v85
	v_dual_mov_b32 v81, v85 :: v_dual_mov_b32 v42, v85
	v_dual_mov_b32 v43, v85 :: v_dual_mov_b32 v44, v85
	v_dual_mov_b32 v45, v85 :: v_dual_mov_b32 v46, v85
	v_dual_mov_b32 v47, v85 :: v_dual_mov_b32 v48, v85
	v_mov_b32_e32 v49, v85
	s_lshr_b32 s48, s37, 16
	s_lshr_b32 s49, s35, 16
	s_lshl_b64 s[6:7], s[6:7], 1
	s_lshl_b64 s[18:19], s[18:19], 1
	s_mov_b32 s43, s1
	s_movk_i32 s16, 0x140
	s_mov_b32 s45, s5
	s_bitset1_b32 s48, 23
	s_bitset1_b32 s49, 23
	s_add_nc_u64 s[20:21], s[26:27], s[6:7]
	s_add_nc_u64 s[24:25], s[14:15], s[18:19]
	s_mov_b32 s26, -1
	s_mov_b32 s8, 64
	s_mov_b32 s27, s11
	s_branch .LBB0_19
.LBB0_18:                               ;   in Loop: Header=BB0_19 Depth=1
	s_or_b32 exec_lo, exec_lo, s5
	s_cmp_eq_u32 s27, s46
	s_barrier_wait -1
	s_barrier_wait -3
	s_cbranch_scc1 .LBB0_41
.LBB0_19:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_21 Depth 2
                                        ;     Child Loop BB0_24 Depth 2
                                        ;     Child Loop BB0_26 Depth 2
                                        ;     Child Loop BB0_29 Depth 2
	s_and_b32 s50, s27, 1
	s_add_co_i32 s27, s27, 1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshl_b32 s5, s27, 7
	s_sub_co_i32 s6, s30, s5
	s_xor_b32 s5, s50, 1
	s_min_i32 s6, s6, 0x80
	s_cmp_lt_i32 s27, s46
	s_cselect_b32 s10, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_and_b32 s7, s10, exec_lo
	s_cselect_b32 s34, s6, 0
	s_cmp_lt_i32 s34, 0x80
	s_cselect_b32 s6, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b32 s6, s47, s6
	s_or_b32 s6, s40, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s6
	s_cbranch_vccnz .LBB0_31
; %bb.20:                               ;   in Loop: Header=BB0_19 Depth=1
	v_mov_b64_e32 v[98:99], v[0:1]
	v_mov_b32_e32 v101, 16
	s_cmp_lg_u32 s5, 0
	s_mov_b32 s13, 0
	s_cselect_b32 s7, s45, s1
	s_cselect_b32 s6, s44, 0
.LBB0_21:                               ;   Parent Loop BB0_19 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_dual_mov_b32 v84, v98 :: v_dual_add_nc_u32 v101, -2, v101
	v_dual_mov_b32 v102, v99 :: v_dual_mov_b32 v103, v85
	v_add_nc_u32_e32 v99, 0x200, v99
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u64 v[104:105], v[84:85], 2, s[6:7]
	v_cmp_eq_u32_e32 vcc_lo, 0, v101
	v_add_nc_u32_e32 v98, 0x200, v98
	v_lshl_add_u64 v[102:103], v[102:103], 2, s[6:7]
	s_clause 0x1
	flat_store_b32 v[104:105], v85
	flat_store_b32 v[102:103], v85
	s_or_b32 s13, vcc_lo, s13
	s_wait_xcnt 0x0
	s_and_not1_b32 exec_lo, exec_lo, s13
	s_cbranch_execnz .LBB0_21
; %bb.22:                               ;   in Loop: Header=BB0_19 Depth=1
	s_or_b32 exec_lo, exec_lo, s13
	s_and_saveexec_b32 s13, s26
	s_cbranch_execz .LBB0_25
; %bb.23:                               ;   in Loop: Header=BB0_19 Depth=1
	v_add_nc_u64_e32 v[98:99], s[6:7], v[94:95]
	v_mov_b32_e32 v84, v93
	s_mov_b32 s6, 0
.LBB0_24:                               ;   Parent Loop BB0_19 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v84, 0x100, v84
	flat_store_b32 v[98:99], v85
	s_wait_xcnt 0x0
	v_add_nc_u64_e32 v[98:99], 0x400, v[98:99]
	v_cmp_lt_u32_e32 vcc_lo, 0xfff, v84
	s_or_b32 s6, vcc_lo, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s6
	s_cbranch_execnz .LBB0_24
.LBB0_25:                               ;   in Loop: Header=BB0_19 Depth=1
	s_or_b32 exec_lo, exec_lo, s13
	v_mov_b64_e32 v[98:99], v[0:1]
	v_mov_b32_e32 v101, 0x54
	s_cmp_lg_u32 s5, 0
	s_mov_b32 s13, 0
	s_cselect_b32 s7, s39, s43
	s_cselect_b32 s6, s38, s41
.LBB0_26:                               ;   Parent Loop BB0_19 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_dual_mov_b32 v84, v98 :: v_dual_add_nc_u32 v101, -2, v101
	v_dual_mov_b32 v102, v99 :: v_dual_mov_b32 v103, v85
	v_add_nc_u32_e32 v99, 0x200, v99
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u64 v[104:105], v[84:85], 2, s[6:7]
	v_cmp_eq_u32_e32 vcc_lo, 0, v101
	v_add_nc_u32_e32 v98, 0x200, v98
	v_lshl_add_u64 v[102:103], v[102:103], 2, s[6:7]
	s_clause 0x1
	flat_store_b32 v[104:105], v85
	flat_store_b32 v[102:103], v85
	s_or_b32 s13, vcc_lo, s13
	s_wait_xcnt 0x0
	s_and_not1_b32 exec_lo, exec_lo, s13
	s_cbranch_execnz .LBB0_26
; %bb.27:                               ;   in Loop: Header=BB0_19 Depth=1
	s_or_b32 exec_lo, exec_lo, s13
	s_and_saveexec_b32 s13, s26
	s_cbranch_execz .LBB0_30
; %bb.28:                               ;   in Loop: Header=BB0_19 Depth=1
	v_add_nc_u64_e32 v[98:99], s[6:7], v[96:97]
	v_mov_b32_e32 v84, v100
	s_mov_b32 s6, 0
.LBB0_29:                               ;   Parent Loop BB0_19 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v84, 0x100, v84
	flat_store_b32 v[98:99], v85
	s_wait_xcnt 0x0
	v_add_nc_u64_e32 v[98:99], 0x400, v[98:99]
	v_cmp_lt_u32_e32 vcc_lo, 0x53ff, v84
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
	v_cmpx_lt_i32_e32 0, v89
	s_xor_b32 s7, exec_lo, s7
	s_cbranch_execnz .LBB0_37
; %bb.32:                               ;   in Loop: Header=BB0_19 Depth=1
	s_and_not1_saveexec_b32 s13, s7
	s_cbranch_execnz .LBB0_40
.LBB0_33:                               ;   in Loop: Header=BB0_19 Depth=1
	s_or_b32 exec_lo, exec_lo, s13
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s42
	s_cbranch_vccnz .LBB0_35
.LBB0_34:                               ;   in Loop: Header=BB0_19 Depth=1
	s_cmp_lg_u32 s50, 0
	s_cselect_b32 s5, s44, 0
	s_cselect_b32 s6, s38, s41
	v_lshl_add_u32 v84, v82, 1, s5
	v_lshl_add_u32 v98, v86, 1, s6
	v_lshl_add_u32 v99, v90, 1, s6
	ds_load_b128 v[102:105], v84
	ds_load_b128 v[106:109], v84 offset:16
	ds_load_b128 v[110:113], v98
	ds_load_b128 v[114:117], v98 offset:16
	ds_load_b128 v[118:121], v98 offset:4352
	ds_load_b128 v[122:125], v98 offset:4368
	ds_load_b128 v[126:129], v98 offset:8704
	ds_load_b128 v[130:133], v98 offset:8720
	ds_load_b128 v[134:137], v98 offset:13056
	ds_load_b128 v[138:141], v98 offset:13072
	ds_load_b128 v[142:145], v98 offset:17408
	v_lshl_add_u32 v84, v88, 1, s5
	ds_load_b128 v[150:153], v99 offset:64
	ds_load_b128 v[154:157], v99 offset:80
	ds_load_b128 v[158:161], v99 offset:4416
	ds_load_b128 v[162:165], v99 offset:4432
	ds_load_b128 v[166:169], v99 offset:8768
	ds_load_b128 v[170:173], v99 offset:8784
	ds_load_b128 v[174:177], v99 offset:17472
	ds_load_b128 v[178:181], v99 offset:17488
	s_wait_dscnt 0xf
	v_wmma_f32_16x16x32_bf16 v[74:81], v[102:109], v[110:117], v[74:81]
	ds_load_b128 v[182:185], v99 offset:21824
	ds_load_b128 v[186:189], v99 offset:21840
	ds_load_b128 v[190:193], v99 offset:30528
	ds_load_b128 v[194:197], v99 offset:30544
	ds_load_b128 v[198:201], v99 offset:34880
	ds_load_b128 v[202:205], v99 offset:34896
	ds_load_b128 v[206:209], v99 offset:39232
	s_wait_dscnt 0x14
	v_wmma_f32_16x16x32_bf16 v[66:73], v[102:109], v[118:125], v[66:73] matrix_a_reuse
	ds_load_b128 v[210:213], v99 offset:39248
	; sched_group_barrier mask(0x00000100) size(11) SyncID(0)
	s_wait_dscnt 0x13
	v_wmma_f32_16x16x32_bf16 v[58:65], v[102:109], v[126:133], v[58:65] matrix_a_reuse
	s_wait_dscnt 0x11
	v_wmma_f32_16x16x32_bf16 v[50:57], v[102:109], v[134:141], v[50:57] matrix_a_reuse
	ds_load_b128 v[110:113], v98 offset:39168
	ds_load_b128 v[114:117], v98 offset:39184
	ds_load_b128 v[146:149], v98 offset:17424
	ds_load_b128 v[118:121], v98 offset:26112
	ds_load_b128 v[122:125], v98 offset:26128
	ds_load_b128 v[126:129], v98 offset:30464
	ds_load_b128 v[130:133], v98 offset:30480
	ds_load_b128 v[134:137], v98 offset:34816
	ds_load_b128 v[138:141], v98 offset:34832
	s_wait_dscnt 0x7
	v_wmma_f32_16x16x32_bf16 v[42:49], v[102:109], v[110:117], v[42:49] matrix_a_reuse
	ds_load_b128 v[110:113], v98 offset:21760
	ds_load_b128 v[114:117], v98 offset:21776
	; sched_group_barrier mask(0x00000008) size(5) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(11) SyncID(0)
	s_wait_dscnt 0x8
	v_wmma_f32_16x16x32_bf16 v[34:41], v[102:109], v[142:149], v[34:41] matrix_a_reuse
	ds_load_b128 v[142:145], v84 offset:64
	ds_load_b128 v[146:149], v84 offset:80
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[26:33], v[102:109], v[110:117], v[26:33] matrix_a_reuse
	ds_load_b128 v[110:113], v99 offset:13120
	ds_load_b128 v[114:117], v99 offset:13136
	v_wmma_f32_16x16x32_bf16 v[18:25], v[102:109], v[118:125], v[18:25] matrix_a_reuse
	ds_load_b128 v[118:121], v99 offset:26176
	ds_load_b128 v[122:125], v99 offset:26192
	v_wmma_f32_16x16x32_bf16 v[10:17], v[102:109], v[126:133], v[10:17] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(5) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[2:9], v[102:109], v[134:141], v[2:9] matrix_a_reuse
	; sched_barrier mask(0x00000000)
	v_lshl_add_u32 v84, v92, 1, s5
	ds_load_b128 v[102:105], v99 offset:128
	ds_load_b128 v[106:109], v99 offset:144
	ds_load_b128 v[126:129], v99 offset:4480
	ds_load_b128 v[130:133], v99 offset:4496
	ds_load_b128 v[134:137], v84 offset:128
	ds_load_b128 v[138:141], v84 offset:144
	ds_load_b128 v[214:217], v99 offset:8832
	ds_load_b128 v[218:221], v99 offset:8848
	ds_load_b128 v[222:225], v99 offset:13184
	ds_load_b128 v[226:229], v99 offset:13200
	ds_load_b128 v[230:233], v99 offset:17536
	s_wait_dscnt 0xf
	v_wmma_f32_16x16x32_bf16 v[74:81], v[142:149], v[150:157], v[74:81]
	; sched_group_barrier mask(0x00000100) size(11) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[66:73], v[142:149], v[158:165], v[66:73] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[58:65], v[142:149], v[166:173], v[58:65] matrix_a_reuse
	s_wait_dscnt 0xd
	v_wmma_f32_16x16x32_bf16 v[50:57], v[142:149], v[110:117], v[50:57] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[34:41], v[142:149], v[174:181], v[34:41] matrix_a_reuse
	ds_load_b128 v[234:237], v99 offset:17552
	ds_load_b128 v[110:113], v99 offset:21888
	ds_load_b128 v[114:117], v99 offset:21904
	ds_load_b128 v[150:153], v99 offset:26240
	ds_load_b128 v[154:157], v99 offset:26256
	ds_load_b128 v[158:161], v99 offset:30592
	ds_load_b128 v[162:165], v99 offset:30608
	ds_load_b128 v[166:169], v99 offset:34944
	ds_load_b128 v[170:173], v99 offset:34960
	ds_load_b128 v[174:177], v99 offset:39296
	ds_load_b128 v[178:181], v99 offset:39312
	v_wmma_f32_16x16x32_bf16 v[26:33], v[142:149], v[182:189], v[26:33] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(5) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(11) SyncID(0)
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[18:25], v[142:149], v[118:125], v[18:25] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[10:17], v[142:149], v[190:197], v[10:17] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[2:9], v[142:149], v[198:205], v[2:9] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[42:49], v[142:149], v[206:213], v[42:49] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(5) SyncID(0)
	; sched_barrier mask(0x00000000)
	ds_load_b128 v[118:121], v84 offset:192
	ds_load_b128 v[122:125], v84 offset:208
	ds_load_b128 v[142:145], v99 offset:192
	ds_load_b128 v[146:149], v99 offset:208
	ds_load_b128 v[182:185], v99 offset:4544
	ds_load_b128 v[186:189], v99 offset:4560
	ds_load_b128 v[190:193], v99 offset:8896
	ds_load_b128 v[194:197], v99 offset:8912
	ds_load_b128 v[198:201], v99 offset:13248
	ds_load_b128 v[202:205], v99 offset:13264
	ds_load_b128 v[206:209], v99 offset:17600
	s_wait_dscnt 0x1b
	v_wmma_f32_16x16x32_bf16 v[74:81], v[134:141], v[102:109], v[74:81]
	; sched_group_barrier mask(0x00000100) size(11) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[66:73], v[134:141], v[126:133], v[66:73] matrix_a_reuse
	s_wait_dscnt 0x19
	v_wmma_f32_16x16x32_bf16 v[58:65], v[134:141], v[214:221], v[58:65] matrix_a_reuse
	s_wait_dscnt 0x17
	v_wmma_f32_16x16x32_bf16 v[50:57], v[134:141], v[222:229], v[50:57] matrix_a_reuse
	s_wait_dscnt 0x15
	v_wmma_f32_16x16x32_bf16 v[34:41], v[134:141], v[230:237], v[34:41] matrix_a_reuse
	ds_load_b128 v[210:213], v99 offset:17616
	ds_load_b128 v[102:105], v99 offset:21952
	ds_load_b128 v[106:109], v99 offset:21968
	ds_load_b128 v[126:129], v99 offset:26304
	ds_load_b128 v[130:133], v99 offset:26320
	ds_load_b128 v[214:217], v99 offset:30656
	ds_load_b128 v[218:221], v99 offset:30672
	ds_load_b128 v[222:225], v99 offset:35008
	ds_load_b128 v[226:229], v99 offset:35024
	ds_load_b128 v[230:233], v99 offset:39360
	ds_load_b128 v[234:237], v99 offset:39376
	s_wait_dscnt 0x1e
	v_wmma_f32_16x16x32_bf16 v[26:33], v[134:141], v[110:117], v[26:33] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(5) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(11) SyncID(0)
	s_wait_dscnt 0x1c
	v_wmma_f32_16x16x32_bf16 v[18:25], v[134:141], v[150:157], v[18:25] matrix_a_reuse
	s_wait_dscnt 0x1a
	v_wmma_f32_16x16x32_bf16 v[10:17], v[134:141], v[158:165], v[10:17] matrix_a_reuse
	s_wait_dscnt 0x18
	v_wmma_f32_16x16x32_bf16 v[2:9], v[134:141], v[166:173], v[2:9] matrix_a_reuse
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[42:49], v[134:141], v[174:181], v[42:49] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(5) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[74:81], v[118:125], v[142:149], v[74:81]
	; sched_group_barrier mask(0x00000100) size(11) SyncID(0)
	s_wait_dscnt 0x10
	v_wmma_f32_16x16x32_bf16 v[66:73], v[118:125], v[182:189], v[66:73] matrix_a_reuse
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[58:65], v[118:125], v[190:197], v[58:65] matrix_a_reuse
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[50:57], v[118:125], v[198:205], v[50:57] matrix_a_reuse
	s_wait_dscnt 0xa
	v_wmma_f32_16x16x32_bf16 v[34:41], v[118:125], v[206:213], v[34:41] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(5) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(11) SyncID(0)
	s_wait_dscnt 0x8
	v_wmma_f32_16x16x32_bf16 v[26:33], v[118:125], v[102:109], v[26:33] matrix_a_reuse
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[18:25], v[118:125], v[126:133], v[18:25] matrix_a_reuse
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[10:17], v[118:125], v[214:221], v[10:17] matrix_a_reuse
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[2:9], v[118:125], v[222:229], v[2:9] matrix_a_reuse
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[42:49], v[118:125], v[230:237], v[42:49] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(5) SyncID(0)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_wait_dscnt 0
	;;#ASMEND
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
	v_cmpx_eq_u32_e32 1, v89
	s_cbranch_execz .LBB0_39
; %bb.38:                               ;   in Loop: Header=BB0_19 Depth=1
	s_cmp_lg_u32 s5, 0
	s_mov_b32 s36, s34
	s_cselect_b32 s13, s38, s41
	s_cmp_gt_i32 s34, 0
	s_mov_b32 s18, s11
	s_cselect_b32 s17, -1, 0
	s_lshl_b32 s10, s6, 7
	s_mov_b32 s19, s11
	s_lshl_b64 s[14:15], s[10:11], 1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_nc_u64 s[14:15], s[20:21], s[14:15]
	v_dual_mov_b32 v99, s13 :: v_dual_mov_b32 v98, s14
	s_and_b32 s10, s15, 0x1ffffff
	s_and_b32 s13, s29, s17
	s_bitset1_b32 s10, 31
	v_cndmask_b32_e64 v84, 0, 1, s13
	v_mov_b32_e32 v101, s10
	v_readfirstlane_b32 s53, v99
	v_readfirstlane_b32 s54, v98
	s_lshr_b64 s[14:15], s[36:37], 16
	v_readfirstlane_b32 s52, v84
	v_readfirstlane_b32 s55, v101
	s_lshl_b32 s13, s34, 16
	s_mov_b32 s15, s48
	s_mov_b32 s17, s9
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[52:55], s[12:19]
.LBB0_39:                               ;   in Loop: Header=BB0_19 Depth=1
	s_or_b32 exec_lo, exec_lo, s51
	s_and_not1_saveexec_b32 s13, s7
	s_cbranch_execz .LBB0_33
.LBB0_40:                               ;   in Loop: Header=BB0_19 Depth=1
	s_cmp_lg_u32 s5, 0
	s_cselect_b32 s5, s44, 0
	s_cmp_gt_i32 s34, 0
	s_cselect_b32 s14, -1, 0
	s_lshl_b32 s10, s6, 7
	s_delay_alu instid0(SALU_CYCLE_1)
	s_lshl_b64 s[6:7], s[10:11], 1
	s_and_b32 s10, s3, s14
	s_add_nc_u64 s[6:7], s[24:25], s[6:7]
	v_cndmask_b32_e64 v84, 0, 1, s10
	s_and_b32 s7, s7, 0x1ffffff
	v_dual_mov_b32 v99, s5 :: v_dual_mov_b32 v98, s6
	s_bitset1_b32 s7, 31
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_readfirstlane_b32 s52, v84
	v_mov_b32_e32 v101, s7
	v_readfirstlane_b32 s53, v99
	v_readfirstlane_b32 s54, v98
	s_lshr_b64 s[6:7], s[34:35], 16
	s_lshl_b32 s5, s34, 16
	v_readfirstlane_b32 s55, v101
	s_mov_b32 s7, s49
	s_mov_b32 s10, s11
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	tensor_load_to_lds s[52:55], s[4:11]
	s_or_b32 exec_lo, exec_lo, s13
	s_and_not1_b32 vcc_lo, exec_lo, s42
	s_cbranch_vccz .LBB0_34
	s_branch .LBB0_35
.LBB0_41:
	s_wait_tensorcnt 0x0
	;;#ASMSTART
	s_wait_dscnt 0
	;;#ASMEND
	s_and_b32 vcc_lo, exec_lo, s42
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_vccz .LBB0_43
; %bb.42:
	v_lshrrev_b32_e32 v1, 1, v0
	v_and_or_b32 v82, v0, 15, v91
	v_cvt_pk_bf16_f32 v65, v64, v65
	v_cvt_pk_bf16_f32 v64, v62, v63
	v_cvt_pk_bf16_f32 v63, v60, v61
	v_and_or_b32 v1, v1, 8, v87
	v_cvt_pk_bf16_f32 v73, v72, v73
	v_cvt_pk_bf16_f32 v72, v70, v71
	v_cvt_pk_bf16_f32 v70, v66, v67
	v_lshlrev_b32_e32 v61, 3, v82
	v_lshl_or_b32 v1, v82, 6, v1
	v_cvt_pk_bf16_f32 v62, v58, v59
	v_cvt_pk_bf16_f32 v81, v80, v81
	v_cvt_pk_bf16_f32 v80, v78, v79
	v_and_b32_e32 v59, 0x5f0, v61
	v_add_nc_u32_e32 v60, 0x400, v1
	v_add_nc_u32_e32 v66, 0x800, v1
	v_cvt_pk_bf16_f32 v79, v76, v77
	v_cvt_pk_bf16_f32 v78, v74, v75
	v_add_nc_u32_e32 v67, 0xc00, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_lshrrev_b32 v58, 3, v60 :: v_dual_lshrrev_b32 v60, 3, v66
	v_lshlrev_b32_e32 v66, 1, v1
	v_cvt_pk_bf16_f32 v71, v68, v69
	v_cvt_pk_bf16_f32 v41, v40, v41
	v_and_b32_e32 v58, 0xff0, v58
	v_cvt_pk_bf16_f32 v40, v38, v39
	v_add_nc_u32_e32 v59, v59, v66
	v_cvt_pk_bf16_f32 v39, v36, v37
	v_add_nc_u32_e32 v36, 0x1800, v1
	v_dual_add_nc_u32 v58, v58, v66 :: v_dual_lshrrev_b32 v61, 3, v67
	v_add_nc_u32_e32 v67, 0x1000, v1
	v_cvt_pk_bf16_f32 v38, v34, v35
	ds_store_b128 v59, v[78:81]
	ds_store_b128 v58, v[70:73] offset:2048
	v_add_nc_u32_e32 v59, 0x1400, v1
	v_lshrrev_b32_e32 v35, 3, v36
	v_add_nc_u32_e32 v36, 0x1c00, v1
	v_cvt_pk_bf16_f32 v33, v32, v33
	v_cvt_pk_bf16_f32 v32, v30, v31
	v_cvt_pk_bf16_f32 v31, v28, v29
	v_add_nc_u32_e32 v28, 0x2000, v1
	v_add_nc_u32_e32 v1, 0x2400, v1
	v_lshrrev_b32_e32 v67, 3, v67
	v_cvt_pk_bf16_f32 v57, v56, v57
	v_cvt_pk_bf16_f32 v56, v54, v55
	v_cvt_pk_bf16_f32 v54, v50, v51
	v_lshrrev_b32_e32 v51, 3, v59
	v_cvt_pk_bf16_f32 v30, v26, v27
	v_lshrrev_b32_e32 v26, 3, v36
	v_cvt_pk_bf16_f32 v25, v24, v25
	v_cvt_pk_bf16_f32 v24, v22, v23
	v_cvt_pk_bf16_f32 v23, v20, v21
	v_dual_lshrrev_b32 v20, 3, v28 :: v_dual_lshrrev_b32 v1, 3, v1
	v_and_b32_e32 v60, 0xff0, v60
	v_and_b32_e32 v58, 0xff0, v67
	v_and_b32_e32 v34, 0xff0, v51
	v_and_b32_e32 v61, 0xff0, v61
	v_and_b32_e32 v35, 0xff0, v35
	v_and_b32_e32 v26, 0xff0, v26
	v_cvt_pk_bf16_f32 v22, v18, v19
	v_and_b32_e32 v19, 0xff0, v20
	v_and_b32_e32 v1, 0xff0, v1
	v_dual_add_nc_u32 v60, v60, v66 :: v_dual_add_nc_u32 v50, v58, v66
	v_dual_add_nc_u32 v34, v34, v66 :: v_dual_add_nc_u32 v61, v61, v66
	v_cvt_pk_bf16_f32 v55, v52, v53
	v_dual_add_nc_u32 v27, v35, v66 :: v_dual_add_nc_u32 v18, v26, v66
	v_cvt_pk_bf16_f32 v17, v16, v17
	v_cvt_pk_bf16_f32 v16, v14, v15
	v_cvt_pk_bf16_f32 v15, v12, v13
	v_cvt_pk_bf16_f32 v14, v10, v11
	v_add_nc_u32_e32 v10, v19, v66
	v_cvt_pk_bf16_f32 v9, v8, v9
	v_cvt_pk_bf16_f32 v8, v6, v7
	v_cvt_pk_bf16_f32 v7, v4, v5
	v_cvt_pk_bf16_f32 v6, v2, v3
	v_add_nc_u32_e32 v1, v1, v66
	v_cvt_pk_bf16_f32 v5, v48, v49
	v_cvt_pk_bf16_f32 v4, v46, v47
	v_cvt_pk_bf16_f32 v3, v44, v45
	v_cvt_pk_bf16_f32 v2, v42, v43
	ds_store_b128 v60, v[62:65] offset:4096
	ds_store_b128 v61, v[54:57] offset:6144
	ds_store_b128 v50, v[38:41] offset:8192
	ds_store_b128 v34, v[30:33] offset:10240
	ds_store_b128 v27, v[22:25] offset:12288
	ds_store_b128 v18, v[14:17] offset:14336
	ds_store_b128 v10, v[6:9] offset:16384
	ds_store_b128 v1, v[2:5] offset:18432
.LBB0_43:
	v_cmp_ne_u32_e32 vcc_lo, 1, v83
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
	v_nop
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
	v_lshl_add_u32 v11, v11, 6, v12
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_mul_lo_u32 v16, v27, s22
	v_dual_add_nc_u32 v24, s19, v27 :: v_dual_sub_nc_u32 v6, v2, v6
	v_ashrrev_i32_e32 v19, 31, v18
	v_lshl_add_u32 v26, v26, 6, v14
	v_mul_u64_e32 v[20:21], s[6:7], v[20:21]
	v_ashrrev_i32_e32 v25, 31, v24
	v_lshl_add_u32 v1, v1, 6, v6
	v_sub_nc_u32_e32 v16, v5, v16
	v_mul_u64_e32 v[18:19], s[28:29], v[18:19]
	v_mul_u64_e32 v[22:23], s[8:9], v[22:23]
	v_mul_u64_e32 v[24:25], s[10:11], v[24:25]
	v_ashrrev_i32_e32 v28, 31, v1
	v_lshl_add_u32 v27, v27, 6, v16
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
	v_dual_sub_nc_u32 v7, v4, v9 :: v_dual_lshlrev_b32 v4, 6, v4
	v_lshlrev_b32_e32 v9, 6, v9
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
	.size	bm064_bn320_bk128_wm4_wn2_mc1, .Lfunc_end0-bm064_bn320_bk128_wm4_wn2_mc1
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel bm064_bn320_bk128_wm4_wn2_mc1
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
		.amdhsa_next_free_vgpr 337
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
		.amdhsa_inst_pref_size 52
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text.bm064_bn320_bk128_wm4_wn2_mc1,"axG",@progbits,bm064_bn320_bk128_wm4_wn2_mc1,comdat
                                        ; -- End function
	.set .Lbm064_bn320_bk128_wm4_wn2_mc1.num_vgpr, 238
	.set .Lbm064_bn320_bk128_wm4_wn2_mc1.num_agpr, 0
	.set .Lbm064_bn320_bk128_wm4_wn2_mc1.numbered_sgpr, 56
	.set .Lbm064_bn320_bk128_wm4_wn2_mc1.num_named_barrier, 0
	.set .Lbm064_bn320_bk128_wm4_wn2_mc1.private_seg_size, 0
	.set .Lbm064_bn320_bk128_wm4_wn2_mc1.uses_vcc, 1
	.set .Lbm064_bn320_bk128_wm4_wn2_mc1.uses_flat_scratch, 1
	.set .Lbm064_bn320_bk128_wm4_wn2_mc1.has_dyn_sized_stack, 0
	.set .Lbm064_bn320_bk128_wm4_wn2_mc1.has_recursion, 0
	.set .Lbm064_bn320_bk128_wm4_wn2_mc1.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 6620
; TotalNumSgprs: 58
; NumVgprs: 238
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 208896 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 21
; NumSGPRsForWavesPerEU: 58
; NumVGPRsForWavesPerEU: 337
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
	.type	__hip_cuid_a5c429f419a99dd5,@object ; @__hip_cuid_a5c429f419a99dd5
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_a5c429f419a99dd5
__hip_cuid_a5c429f419a99dd5:
	.byte	0                               ; 0x0
	.size	__hip_cuid_a5c429f419a99dd5, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git aa451e1fe6a793394d6733051b1778633063ae96+PATCHED:d17c5aa0e3ea29cde402f58f27e39b6034effa27)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_a5c429f419a99dd5
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
    .name:           bm064_bn320_bk128_wm4_wn2_mc1
    .private_segment_fixed_size: 0
    .sgpr_count:     58
    .sgpr_spill_count: 0
    .symbol:         bm064_bn320_bk128_wm4_wn2_mc1.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     238
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
