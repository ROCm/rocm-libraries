	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.section	.text.bm128_bn128_bk096_wm8_wn1_mc0,"axG",@progbits,bm128_bn128_bk096_wm8_wn1_mc0,comdat
	.protected	bm128_bn128_bk096_wm8_wn1_mc0 ; -- Begin function bm128_bn128_bk096_wm8_wn1_mc0
	.globl	bm128_bn128_bk096_wm8_wn1_mc0
	.p2align	8
	.type	bm128_bn128_bk096_wm8_wn1_mc0,@function
bm128_bn128_bk096_wm8_wn1_mc0: ; @bm128_bn128_bk096_wm8_wn1_mc0
; %bb.0:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1 ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_mov_b64 s[2:3], src_shared_base
	s_movk_i32 s2, 0x6600
	s_load_b96 s[20:22], s[0:1], 0x78 nv
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
	s_mov_b32 s9, s22
	s_cselect_b32 s36, ttmp9, s3
	s_cselect_b32 s5, ttmp7, s5
	s_add_co_i32 s2, s20, 0x7f
	s_add_co_i32 s7, s21, 0x7f
	s_ashr_i32 s3, s2, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshr_b32 s3, s3, 25
	s_add_co_i32 s2, s2, s3
	s_ashr_i32 s3, s7, 31
	s_ashr_i32 s6, s2, 7
	s_lshr_b32 s3, s3, 25
	s_lshl_b32 s2, s36, 7
	s_add_co_i32 s7, s7, s3
	s_sub_co_i32 s3, s20, s2
	s_ashr_i32 s7, s7, 7
	s_min_i32 s23, s3, 0x80
	s_cmp_lt_i32 s36, s6
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_and_b32 s8, s3, exec_lo
	s_cselect_b32 s25, s23, 0
	s_lshl_b32 s33, s5, 7
	s_sub_co_i32 s8, s21, s33
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_min_i32 s8, s8, 0x80
	s_cmp_lt_i32 s5, s7
	s_cselect_b32 s21, -1, 0
	s_and_b32 s10, s21, exec_lo
	s_cselect_b32 s27, s8, 0
	s_add_co_i32 s28, s22, 0x5f
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 2, 1), 1
	s_min_i32 s8, s22, 0x60
	s_cmp_gt_i32 s28, 0x5f
	s_cselect_b32 s29, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_and_b32 s10, s29, exec_lo
	s_cselect_b32 s24, s8, 0
	s_cmp_lt_i32 s25, 0x80
	s_cselect_b32 s34, -1, 0
	s_and_b32 vcc_lo, exec_lo, s34
	s_mov_b32 s8, s34
	s_cbranch_vccnz .LBB0_2
; %bb.1:
	s_cmp_lt_i32 s27, 0x80
	s_cselect_b32 s8, -1, 0
	s_cmp_lt_i32 s24, 0x60
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
	v_cmp_lt_u32_e32 vcc_lo, 0x187f, v5
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
	ds_store_b32 v2, v3 offset:26112
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
	s_load_b64 s[30:31], s[0:1], 0x0 nv
	s_load_b128 s[16:19], s[0:1], 0x20 nv
	s_load_b128 s[12:15], s[0:1], 0x48 nv
	v_lshrrev_b32_e32 v67, 5, v0
	s_lshl_b32 s38, s4, 2
	s_add_co_i32 s7, s7, -1
	s_wait_xcnt 0x0
	s_mov_b64 s[0:1], src_shared_base
	s_or_b32 s35, s38, 0x6600
	s_add_co_i32 s37, s6, -1
	s_min_i32 s0, s5, s7
	s_mov_b32 s4, exec_lo
	v_cmpx_lt_i32_e32 0, v67
	s_xor_b32 s39, exec_lo, s4
	s_cbranch_execz .LBB0_12
; %bb.9:
	s_mov_b32 s40, exec_lo
	v_cmpx_eq_u32_e32 1, v67
	s_cbranch_execz .LBB0_11
; %bb.10:
	s_cmp_gt_i32 s24, 0
	s_mov_b32 s26, s24
	s_cselect_b32 s8, -1, 0
	s_lshl_b32 s4, s0, 7
	s_wait_kmcnt 0x0
	s_bfe_i64 s[6:7], s[12:13], 0x200000
	s_ashr_i32 s5, s4, 31
	s_mov_b32 s10, 0
	s_mul_u64 s[4:5], s[6:7], s[4:5]
	s_mov_b32 s11, s10
	s_lshl_b64 s[4:5], s[4:5], 1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_nc_u64 s[6:7], s[18:19], s[4:5]
	v_dual_mov_b32 v1, s35 :: v_dual_mov_b32 v4, s6
	s_and_b32 s4, s7, 0x1ffffff
	s_and_b32 s7, s21, s8
	s_bitset1_b32 s4, 31
	v_cndmask_b32_e64 v2, 0, 1, s7
	v_mov_b32_e32 v3, s4
	v_readfirstlane_b32 s45, v1
	v_readfirstlane_b32 s46, v4
	s_lshr_b32 s4, s27, 16
	v_readfirstlane_b32 s44, v2
	v_readfirstlane_b32 s47, v3
	s_lshr_b64 s[6:7], s[26:27], 16
	s_lshl_b32 s5, s24, 16
	s_or_b32 s7, s4, 0x600000
	s_movk_i32 s8, 0x80
	s_mov_b32 s4, 0x7510000
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[44:47], s[4:11]
.LBB0_11:
	s_or_b32 exec_lo, exec_lo, s40
.LBB0_12:
	s_or_saveexec_b32 s39, s39
	s_min_i32 s26, s36, s37
	s_xor_b32 exec_lo, exec_lo, s39
	s_cbranch_execz .LBB0_14
; %bb.13:
	s_cmp_gt_i32 s24, 0
	s_mov_b32 s10, 0
	s_cselect_b32 s8, -1, 0
	s_lshl_b32 s4, s26, 7
	s_wait_kmcnt 0x0
	s_bfe_i64 s[6:7], s[16:17], 0x200000
	s_ashr_i32 s5, s4, 31
	s_and_b32 s8, s3, s8
	s_mul_u64 s[4:5], s[6:7], s[4:5]
	v_cndmask_b32_e64 v2, 0, 1, s8
	s_lshl_b64 s[6:7], s[4:5], 1
	s_lshr_b32 s4, s25, 16
	s_add_nc_u64 s[6:7], s[30:31], s[6:7]
	s_lshl_b32 s5, s24, 16
	s_and_b32 s7, s7, 0x1ffffff
	v_readfirstlane_b32 s40, v2
	s_bitset1_b32 s7, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v4, s6 :: v_dual_mov_b32 v3, s7
	s_lshr_b64 s[6:7], s[24:25], 16
	s_or_b32 s7, s4, 0x600000
	s_movk_i32 s8, 0x80
	v_readfirstlane_b32 s42, v4
	v_readfirstlane_b32 s43, v3
	s_mov_b32 s4, 0x7510000
	s_mov_b32 s11, s10
	s_mov_b32 s41, s10
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[40:43], s[4:11]
.LBB0_14:
	s_or_b32 exec_lo, exec_lo, s39
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	v_mov_b32_e32 v9, 0
	s_and_b32 s36, s3, s21
	s_and_not1_b32 vcc_lo, exec_lo, s29
	v_cndmask_b32_e64 v71, 0, 1, s36
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
	v_dual_mov_b32 v26, v9 :: v_dual_mov_b32 v49, v9
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
	v_dual_mov_b32 v58, v9 :: v_dual_mov_b32 v41, v9
	v_dual_mov_b32 v40, v9 :: v_dual_mov_b32 v39, v9
	v_dual_mov_b32 v38, v9 :: v_dual_mov_b32 v37, v9
	v_dual_mov_b32 v36, v9 :: v_dual_mov_b32 v35, v9
	v_mov_b32_e32 v34, v9
	s_barrier_wait -1
	s_cbranch_vccnz .LBB0_37
; %bb.15:
	v_mul_u32_u24_e32 v1, 0x600, v67
	v_and_b32_e32 v3, 16, v0
	v_and_b32_e32 v2, 15, v0
	s_mov_b64 s[4:5], src_shared_base
	s_or_b32 s6, s38, 0xcc00
	s_mov_b32 s7, s5
	v_or_b32_e32 v1, v1, v3
	s_and_b64 s[6:7], s[6:7], 15
	s_mov_b32 s11, 0
	s_sub_co_i32 s4, 16, s6
	s_mov_b32 s37, s1
	v_mad_u32_u24 v1, 0x60, v2, v1
	s_lshr_b32 s4, s4, 2
	s_cmp_lg_u64 s[6:7], 0
	s_mul_hi_i32 s6, s28, 0x2aaaaaab
	s_cselect_b32 s4, s4, 0
	v_lshrrev_b32_e32 v4, 4, v1
	v_mul_u32_u24_e32 v5, 0x60, v2
	s_lshl2_add_u32 s7, s4, s38
	s_mov_b32 s38, s5
	s_add_co_i32 s4, s7, 0x13200
	v_and_b32_e32 v4, 0x7f8, v4
	s_and_b32 s10, s4, 15
	s_add_co_i32 s39, s7, 0xcc00
	s_sub_co_i32 s8, 16, s10
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v66, v4, v1
	v_dual_lshrrev_b32 v1, 4, v5 :: v_dual_bitop2_b32 v8, v5, v3 bitop3:0x54
	s_lshr_b32 s7, s8, 2
	s_cmp_lg_u64 s[10:11], 0
	v_and_b32_e32 v1, 0x78, v1
	s_cselect_b32 s7, s7, 0
	s_lshr_b32 s8, s6, 31
	s_ashr_i32 s40, s6, 4
	s_lshl_b32 s10, s7, 2
	s_add_co_i32 s40, s40, s8
	s_cmp_lt_i32 s27, 0x80
	v_dual_mov_b32 v69, 0 :: v_dual_add_nc_u32 v70, v8, v1
	v_or_b32_e32 v1, 0x1800, v5
	v_or_b32_e32 v10, 32, v3
	s_add_nc_u64 s[28:29], s[4:5], s[10:11]
	s_cselect_b32 s41, -1, 0
	s_lshl_b32 s4, s0, 7
	s_movk_i32 s0, 0x60
	v_lshrrev_b32_e32 v12, 4, v1
	v_mad_u32_u24 v4, v2, s0, 0x600
	s_movk_i32 s0, 0xc00
	v_mad_u32_u24 v13, 0x60, v2, v10
	v_dual_add_nc_u32 v10, v1, v10 :: v_dual_bitop2_b32 v3, 64, v3 bitop3:0x54
	s_movk_i32 s10, 0x1200
	v_mad_u32_u24 v6, 0x60, v2, s0
	s_movk_i32 s0, 0x1e00
	v_mad_u32_u24 v7, 0x60, v2, s10
	s_movk_i32 s10, 0x2400
	v_mad_u32_u24 v5, 0x60, v2, s0
	s_movk_i32 s0, 0x2a00
	v_mad_u32_u24 v9, 0x60, v2, s10
	v_mad_u32_u24 v11, 0x60, v2, s0
	v_mad_u32_u24 v2, 0x60, v2, v3
	v_mad_u32_u24 v14, 0x600, v67, v13
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_dual_add_nc_u32 v1, v1, v3 :: v_dual_lshrrev_b32 v9, 4, v9
	v_sub_nc_u32_e32 v27, 0x197f, v0
	v_add_nc_u32_e32 v24, 0x1200, v2
	v_mad_u32_u24 v21, 0x600, v67, v2
	v_add_nc_u32_e32 v26, 0x2400, v2
	v_dual_lshrrev_b32 v4, 4, v4 :: v_dual_lshrrev_b32 v5, 4, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_lshrrev_b32 v11, 4, v11 :: v_dual_lshrrev_b32 v24, 4, v24
	v_lshrrev_b32_e32 v14, 4, v14
	v_and_b32_e32 v9, 0x2f8, v9
	v_add_nc_u32_e32 v22, 0x600, v2
	v_add_nc_u32_e32 v23, 0xc00, v2
	v_and_b32_e32 v3, 0x3f8, v24
	v_add_nc_u32_e32 v24, 0x1e00, v2
	v_dual_lshrrev_b32 v21, 4, v21 :: v_dual_lshrrev_b32 v25, 4, v2
	v_lshrrev_b32_e32 v1, 4, v1
	v_add_nc_u32_e32 v2, 0x2a00, v2
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_lshrrev_b32 v24, 4, v24 :: v_dual_lshrrev_b32 v26, 4, v26
	v_lshrrev_b32_e32 v27, 8, v27
	v_and_b32_e32 v4, 0xf8, v4
	v_and_b32_e32 v11, 0x3f8, v11
	v_add_nc_u32_e32 v15, 0x600, v13
	v_and_b32_e32 v28, 0x3f8, v1
	v_lshrrev_b32_e32 v1, 4, v2
	v_and_b32_e32 v2, 0x3f8, v24
	v_and_b32_e32 v24, 0x3f8, v26
	v_dual_add_nc_u32 v26, 1, v27 :: v_dual_add_nc_u32 v82, v8, v9
	v_add_nc_u32_e32 v84, v8, v11
	v_add_nc_u32_e32 v9, 0x1e00, v8
	v_dual_lshrrev_b32 v6, 4, v6 :: v_dual_lshrrev_b32 v7, 4, v7
	v_and_b32_e32 v12, 0x1f8, v12
	v_and_b32_e32 v5, 0x3f8, v5
	v_dual_mov_b32 v123, v69 :: v_dual_add_nc_u32 v16, 0xc00, v13
	v_and_b32_e32 v14, 0x7f8, v14
	v_dual_lshrrev_b32 v22, 4, v22 :: v_dual_lshrrev_b32 v23, 4, v23
	v_and_b32_e32 v21, 0x7f8, v21
	v_dual_add_nc_u32 v72, v8, v4 :: v_dual_bitop2_b32 v73, 26, v26 bitop3:0x40
	v_mad_u32_u24 v4, 0x600, v67, v8
	v_add_nc_u32_e32 v11, 0x2400, v8
	v_dual_add_nc_u32 v116, v2, v9 :: v_dual_mov_b32 v2, v69
	v_dual_lshrrev_b32 v18, 4, v13 :: v_dual_lshrrev_b32 v15, 4, v15
	v_and_b32_e32 v6, 0x1f8, v6
	v_add_nc_u32_e32 v17, 0x1200, v13
	v_and_b32_e32 v22, 0x1f8, v22
	v_dual_add_nc_u32 v78, v8, v12 :: v_dual_add_nc_u32 v80, v8, v5
	v_and_b32_e32 v15, 0x1f8, v15
	v_add_nc_u32_e32 v5, 0x600, v8
	v_dual_add_nc_u32 v86, v14, v4 :: v_dual_mov_b32 v14, v69
	v_add_nc_u32_e32 v104, v21, v4
	v_dual_mov_b32 v4, v69 :: v_dual_lshrrev_b32 v16, 4, v16
	v_lshrrev_b32_e32 v17, 4, v17
	v_and_b32_e32 v7, 0x1f8, v7
	v_and_b32_e32 v18, 0xf8, v18
	v_and_b32_e32 v25, 0xf8, v25
	v_and_b32_e32 v16, 0x1f8, v16
	v_and_b32_e32 v23, 0x1f8, v23
	v_dual_add_nc_u32 v74, v8, v6 :: v_dual_add_nc_u32 v76, v8, v7
	v_add_nc_u32_e32 v6, 0xc00, v8
	v_add_nc_u32_e32 v19, 0x1e00, v13
	v_dual_mov_b32 v15, v69 :: v_dual_add_nc_u32 v90, v15, v5
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_add_nc_u32 v88, v18, v8 :: v_dual_add_nc_u32 v92, v16, v6
	v_dual_add_nc_u32 v106, v25, v8 :: v_dual_add_nc_u32 v110, v23, v6
	v_dual_mov_b32 v6, v69 :: v_dual_lshrrev_b32 v10, 4, v10
	v_lshrrev_b32_e32 v19, 4, v19
	v_add_nc_u32_e32 v20, 0x2400, v13
	v_add_nc_u32_e32 v13, 0x2a00, v13
	v_dual_mov_b32 v16, v69 :: v_dual_add_nc_u32 v7, 0x1200, v8
	v_and_b32_e32 v10, 0x3f8, v10
	v_lshl_or_b32 v29, v73, 8, v0
	v_add_nc_u32_e32 v12, 0x2a00, v8
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v112, v3, v7
	v_add3_u32 v114, v8, v28, 64
	v_add3_u32 v96, v8, v10, 32
	v_dual_mov_b32 v3, v69 :: v_dual_mov_b32 v8, v69
	v_dual_lshrrev_b32 v20, 4, v20 :: v_dual_lshrrev_b32 v13, 4, v13
	s_ashr_i32 s5, s4, 31
	s_lshl_b32 s6, s26, 7
	v_and_b32_e32 v17, 0x3f8, v17
	v_and_b32_e32 v19, 0x3f8, v19
	v_and_b32_e32 v20, 0x3f8, v20
	v_and_b32_e32 v13, 0x3f8, v13
	s_wait_kmcnt 0x0
	s_bfe_i64 s[12:13], s[12:13], 0x200000
	v_and_b32_e32 v27, 0x3f8, v1
	s_ashr_i32 s7, s6, 31
	s_mul_u64 s[4:5], s[12:13], s[4:5]
	s_bfe_i64 s[12:13], s[16:17], 0x200000
	s_lshl_b64 s[4:5], s[4:5], 1
	s_mul_u64 s[6:7], s[12:13], s[6:7]
	v_or_b32_e32 v1, 0x100, v0
	v_cmp_ne_u32_e64 s0, v26, v73
	v_add_nc_u32_e32 v108, v22, v5
	v_dual_mov_b32 v22, v69 :: v_dual_add_nc_u32 v75, 0xffffff00, v29
	v_dual_mov_b32 v5, v69 :: v_dual_lshlrev_b32 v122, 2, v29
	v_dual_mov_b32 v10, v69 :: v_dual_mov_b32 v18, v69
	v_dual_add_nc_u32 v94, v17, v7 :: v_dual_add_nc_u32 v98, v19, v9
	v_dual_mov_b32 v7, v69 :: v_dual_mov_b32 v9, v69
	v_dual_mov_b32 v17, v69 :: v_dual_add_nc_u32 v100, v20, v11
	v_dual_add_nc_u32 v102, v13, v12 :: v_dual_add_nc_u32 v120, v27, v12
	v_dual_mov_b32 v19, v69 :: v_dual_mov_b32 v12, v69
	v_dual_mov_b32 v13, v69 :: v_dual_mov_b32 v20, v69
	v_dual_mov_b32 v11, v69 :: v_dual_add_nc_u32 v118, v24, v11
	v_dual_mov_b32 v21, v69 :: v_dual_mov_b32 v23, v69
	v_dual_mov_b32 v24, v69 :: v_dual_mov_b32 v25, v69
	v_dual_mov_b32 v26, v69 :: v_dual_mov_b32 v27, v69
	v_dual_mov_b32 v28, v69 :: v_dual_mov_b32 v29, v69
	v_dual_mov_b32 v30, v69 :: v_dual_mov_b32 v31, v69
	v_dual_mov_b32 v32, v69 :: v_dual_mov_b32 v33, v69
	v_dual_mov_b32 v42, v69 :: v_dual_mov_b32 v43, v69
	v_dual_mov_b32 v44, v69 :: v_dual_mov_b32 v45, v69
	v_dual_mov_b32 v46, v69 :: v_dual_mov_b32 v47, v69
	v_dual_mov_b32 v48, v69 :: v_dual_mov_b32 v49, v69
	v_dual_mov_b32 v50, v69 :: v_dual_mov_b32 v51, v69
	v_dual_mov_b32 v52, v69 :: v_dual_mov_b32 v53, v69
	v_dual_mov_b32 v54, v69 :: v_dual_mov_b32 v55, v69
	v_dual_mov_b32 v56, v69 :: v_dual_mov_b32 v57, v69
	v_dual_mov_b32 v58, v69 :: v_dual_mov_b32 v59, v69
	v_dual_mov_b32 v60, v69 :: v_dual_mov_b32 v61, v69
	v_dual_mov_b32 v62, v69 :: v_dual_mov_b32 v63, v69
	v_dual_mov_b32 v64, v69 :: v_dual_mov_b32 v65, v69
	v_dual_mov_b32 v34, v69 :: v_dual_mov_b32 v35, v69
	v_dual_mov_b32 v36, v69 :: v_dual_mov_b32 v37, v69
	v_dual_mov_b32 v38, v69 :: v_dual_mov_b32 v39, v69
	v_dual_mov_b32 v40, v69 :: v_dual_mov_b32 v41, v69
	s_lshr_b32 s42, s27, 16
	s_lshr_b32 s43, s25, 16
	s_add_nc_u64 s[12:13], s[18:19], s[4:5]
	s_lshl_b64 s[4:5], s[6:7], 1
	s_movk_i32 s8, 0x80
	s_or_b32 s42, s42, 0x600000
	s_or_b32 s43, s43, 0x600000
	s_add_nc_u64 s[16:17], s[30:31], s[4:5]
	s_mov_b32 s4, 0x7510000
	s_mov_b32 s18, s11
	s_branch .LBB0_17
.LBB0_16:                               ;   in Loop: Header=BB0_17 Depth=1
	s_cmp_eq_u32 s18, s40
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_scc1 .LBB0_37
.LBB0_17:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_19 Depth 2
                                        ;     Child Loop BB0_22 Depth 2
                                        ;     Child Loop BB0_24 Depth 2
                                        ;     Child Loop BB0_27 Depth 2
	s_and_b32 s19, s18, 1
	s_add_co_i32 s18, s18, 1
	s_xor_b32 s30, s19, 1
	s_mul_i32 s5, s18, 0xffffffa0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s5, s5, s22
	s_min_i32 s6, s5, 0x60
	s_cmp_lt_i32 s18, s40
	s_cselect_b32 s5, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_and_b32 s7, s5, exec_lo
	s_cselect_b32 s24, s6, 0
	s_cmp_lt_i32 s24, 0x60
	s_cselect_b32 s6, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b32 s6, s41, s6
	s_or_b32 s6, s34, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s6
	s_cbranch_vccnz .LBB0_29
; %bb.18:                               ;   in Loop: Header=BB0_17 Depth=1
	v_nop
	v_nop
	v_nop
	v_nop
	v_mov_b64_e32 v[124:125], v[0:1]
	v_mov_b32_e32 v77, v73
	s_cmp_lg_u32 s30, 0
	s_mov_b32 s10, 0
	s_cselect_b32 s7, s38, s1
	s_cselect_b32 s6, s39, 0
.LBB0_19:                               ;   Parent Loop BB0_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_dual_mov_b32 v68, v124 :: v_dual_add_nc_u32 v77, -2, v77
	v_dual_mov_b32 v126, v125 :: v_dual_mov_b32 v127, v69
	v_add_nc_u32_e32 v125, 0x200, v125
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u64 v[128:129], v[68:69], 2, s[6:7]
	v_cmp_eq_u32_e32 vcc_lo, 0, v77
	v_add_nc_u32_e32 v124, 0x200, v124
	v_lshl_add_u64 v[126:127], v[126:127], 2, s[6:7]
	s_clause 0x1
	flat_store_b32 v[128:129], v69
	flat_store_b32 v[126:127], v69
	s_or_b32 s10, vcc_lo, s10
	s_wait_xcnt 0x0
	s_and_not1_b32 exec_lo, exec_lo, s10
	s_cbranch_execnz .LBB0_19
; %bb.20:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s10
	s_and_saveexec_b32 s10, s0
	s_cbranch_execz .LBB0_23
; %bb.21:                               ;   in Loop: Header=BB0_17 Depth=1
	v_add_nc_u64_e32 v[124:125], s[6:7], v[122:123]
	v_mov_b32_e32 v68, v75
	s_mov_b32 s6, 0
.LBB0_22:                               ;   Parent Loop BB0_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v68, 0x100, v68
	flat_store_b32 v[124:125], v69
	s_wait_xcnt 0x0
	v_add_nc_u64_e32 v[124:125], 0x400, v[124:125]
	v_cmp_lt_u32_e32 vcc_lo, 0x187f, v68
	s_or_b32 s6, vcc_lo, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s6
	s_cbranch_execnz .LBB0_22
.LBB0_23:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s10
	v_mov_b64_e32 v[124:125], v[0:1]
	v_mov_b32_e32 v77, v73
	s_cmp_lg_u32 s30, 0
	s_mov_b32 s10, 0
	s_cselect_b32 s7, s29, s37
	s_cselect_b32 s6, s28, s35
.LBB0_24:                               ;   Parent Loop BB0_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_dual_mov_b32 v68, v124 :: v_dual_add_nc_u32 v77, -2, v77
	v_dual_mov_b32 v126, v125 :: v_dual_mov_b32 v127, v69
	v_add_nc_u32_e32 v125, 0x200, v125
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u64 v[128:129], v[68:69], 2, s[6:7]
	v_cmp_eq_u32_e32 vcc_lo, 0, v77
	v_add_nc_u32_e32 v124, 0x200, v124
	v_lshl_add_u64 v[126:127], v[126:127], 2, s[6:7]
	s_clause 0x1
	flat_store_b32 v[128:129], v69
	flat_store_b32 v[126:127], v69
	s_or_b32 s10, vcc_lo, s10
	s_wait_xcnt 0x0
	s_and_not1_b32 exec_lo, exec_lo, s10
	s_cbranch_execnz .LBB0_24
; %bb.25:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s10
	s_and_saveexec_b32 s10, s0
	s_cbranch_execz .LBB0_28
; %bb.26:                               ;   in Loop: Header=BB0_17 Depth=1
	v_add_nc_u64_e32 v[124:125], s[6:7], v[122:123]
	v_mov_b32_e32 v68, v75
	s_mov_b32 s6, 0
.LBB0_27:                               ;   Parent Loop BB0_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v68, 0x100, v68
	flat_store_b32 v[124:125], v69
	s_wait_xcnt 0x0
	v_add_nc_u64_e32 v[124:125], 0x400, v[124:125]
	v_cmp_lt_u32_e32 vcc_lo, 0x187f, v68
	s_or_b32 s6, vcc_lo, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s6
	s_cbranch_execnz .LBB0_27
.LBB0_28:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s10
	s_barrier_signal -1
	s_barrier_wait -1
.LBB0_29:                               ;   in Loop: Header=BB0_17 Depth=1
	s_and_b32 s5, s5, exec_lo
	s_cselect_b32 s31, s18, 0
	s_mov_b32 s5, exec_lo
	v_cmpx_lt_i32_e32 0, v67
	s_xor_b32 s44, exec_lo, s5
	s_cbranch_execnz .LBB0_32
; %bb.30:                               ;   in Loop: Header=BB0_17 Depth=1
	s_and_not1_saveexec_b32 s26, s44
	s_cbranch_execnz .LBB0_35
.LBB0_31:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s26
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s36
	s_cbranch_vccnz .LBB0_16
	s_branch .LBB0_36
.LBB0_32:                               ;   in Loop: Header=BB0_17 Depth=1
	s_mov_b32 s45, exec_lo
	v_cmpx_eq_u32_e32 1, v67
	s_cbranch_execz .LBB0_34
; %bb.33:                               ;   in Loop: Header=BB0_17 Depth=1
	s_cmp_lg_u32 s30, 0
	s_mul_i32 s10, s31, 0x60
	s_cselect_b32 s5, s28, s35
	s_cmp_gt_i32 s24, 0
	s_cselect_b32 s26, -1, 0
	s_lshl_b64 s[6:7], s[10:11], 1
	s_and_b32 s10, s21, s26
	s_add_nc_u64 s[6:7], s[12:13], s[6:7]
	v_cndmask_b32_e64 v68, 0, 1, s10
	s_and_b32 s7, s7, 0x1ffffff
	v_nop
	v_dual_mov_b32 v77, s5 :: v_dual_mov_b32 v124, s6
	s_bitset1_b32 s7, 31
	s_mov_b32 s26, s24
	v_mov_b32_e32 v79, s7
	v_readfirstlane_b32 s48, v68
	v_readfirstlane_b32 s49, v77
	v_readfirstlane_b32 s50, v124
	s_lshr_b64 s[6:7], s[26:27], 16
	v_readfirstlane_b32 s51, v79
	s_lshl_b32 s5, s24, 16
	s_mov_b32 s7, s42
	s_mov_b32 s10, s11
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[48:51], s[4:11]
.LBB0_34:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s45
	s_and_not1_saveexec_b32 s26, s44
	s_cbranch_execz .LBB0_31
.LBB0_35:                               ;   in Loop: Header=BB0_17 Depth=1
	s_cmp_lg_u32 s30, 0
	s_mul_i32 s10, s31, 0x60
	s_cselect_b32 s5, s39, 0
	s_cmp_gt_i32 s24, 0
	s_cselect_b32 s30, -1, 0
	s_lshl_b64 s[6:7], s[10:11], 1
	s_and_b32 s10, s3, s30
	s_add_nc_u64 s[6:7], s[16:17], s[6:7]
	v_cndmask_b32_e64 v68, 0, 1, s10
	s_and_b32 s7, s7, 0x1ffffff
	v_nop
	v_nop
	v_dual_mov_b32 v77, s5 :: v_dual_mov_b32 v124, s6
	s_bitset1_b32 s7, 31
	v_readfirstlane_b32 s44, v68
	v_mov_b32_e32 v79, s7
	s_delay_alu instid0(VALU_DEP_3)
	v_readfirstlane_b32 s45, v77
	v_readfirstlane_b32 s46, v124
	s_lshr_b64 s[6:7], s[24:25], 16
	s_lshl_b32 s5, s24, 16
	v_readfirstlane_b32 s47, v79
	s_mov_b32 s7, s43
	s_mov_b32 s10, s11
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	tensor_load_to_lds s[44:47], s[4:11]
	s_or_b32 exec_lo, exec_lo, s26
	s_and_not1_b32 vcc_lo, exec_lo, s36
	s_cbranch_vccnz .LBB0_16
.LBB0_36:                               ;   in Loop: Header=BB0_17 Depth=1
	s_cmp_lg_u32 s19, 0
	s_cselect_b32 s6, s39, 0
	s_cselect_b32 s5, s28, s35
	v_lshl_add_u32 v68, v66, 1, s6
	v_lshl_add_u32 v77, v70, 1, s5
	v_lshl_add_u32 v79, v80, 1, s5
	v_lshl_add_u32 v81, v86, 1, s6
	v_lshl_add_u32 v83, v88, 1, s5
	ds_load_b128 v[124:127], v68
	ds_load_b128 v[128:131], v68 offset:16
	v_lshl_add_u32 v68, v72, 1, s5
	ds_load_b128 v[132:135], v77
	ds_load_b128 v[136:139], v77 offset:16
	v_lshl_add_u32 v77, v76, 1, s5
	v_lshl_add_u32 v85, v90, 1, s5
	ds_load_b128 v[140:143], v68 offset:3072
	ds_load_b128 v[144:147], v68 offset:3088
	v_lshl_add_u32 v68, v74, 1, s5
	ds_load_b128 v[156:159], v77 offset:9216
	v_lshl_add_u32 v87, v92, 1, s5
	v_lshl_add_u32 v89, v94, 1, s5
	v_lshl_add_u32 v91, v98, 1, s5
	ds_load_b128 v[148:151], v68 offset:6144
	ds_load_b128 v[152:155], v68 offset:6160
	v_lshl_add_u32 v68, v84, 1, s5
	v_lshl_add_u32 v93, v100, 1, s5
	v_lshl_add_u32 v95, v102, 1, s5
	ds_load_b128 v[164:167], v83 offset:64
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[58:65], v[124:131], v[132:139], v[58:65]
	ds_load_b128 v[168:171], v83 offset:80
	ds_load_b128 v[180:183], v87 offset:64
	ds_load_b128 v[184:187], v87 offset:80
	ds_load_b128 v[188:191], v89 offset:64
	ds_load_b128 v[192:195], v89 offset:80
	ds_load_b128 v[196:199], v91 offset:64
	ds_load_b128 v[200:203], v91 offset:80
	s_wait_dscnt 0xb
	v_wmma_f32_16x16x32_bf16 v[50:57], v[124:131], v[140:147], v[50:57] matrix_a_reuse
	ds_load_b128 v[204:207], v93 offset:64
	ds_load_b128 v[208:211], v93 offset:80
	ds_load_b128 v[212:215], v95 offset:64
	ds_load_b128 v[216:219], v95 offset:80
	; sched_group_barrier mask(0x00000100) size(9) SyncID(0)
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[42:49], v[124:131], v[148:155], v[42:49] matrix_a_reuse
	ds_load_b128 v[160:163], v77 offset:9232
	v_lshl_add_u32 v77, v82, 1, s5
	ds_load_b128 v[172:175], v77 offset:18432
	ds_load_b128 v[176:179], v77 offset:18448
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[2:9], v[124:131], v[172:179], v[2:9] matrix_a_reuse
	ds_load_b128 v[132:135], v68 offset:21504
	ds_load_b128 v[136:139], v68 offset:21520
	v_lshl_add_u32 v68, v78, 1, s5
	ds_load_b128 v[140:143], v81 offset:64
	ds_load_b128 v[144:147], v81 offset:80
	ds_load_b128 v[148:151], v79 offset:15360
	ds_load_b128 v[152:155], v79 offset:15376
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[34:41], v[124:131], v[132:139], v[34:41] matrix_a_reuse
	ds_load_b128 v[132:135], v68 offset:12288
	ds_load_b128 v[136:139], v68 offset:12304
	v_lshl_add_u32 v68, v96, 1, s5
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(9) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[26:33], v[124:131], v[156:163], v[26:33] matrix_a_reuse
	ds_load_b128 v[156:159], v85 offset:64
	ds_load_b128 v[160:163], v85 offset:80
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[18:25], v[124:131], v[132:139], v[18:25] matrix_a_reuse
	ds_load_b128 v[132:135], v68 offset:12288
	ds_load_b128 v[136:139], v68 offset:12304
	v_wmma_f32_16x16x32_bf16 v[10:17], v[124:131], v[148:155], v[10:17] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_barrier mask(0x00000000)
	v_lshl_add_u32 v68, v104, 1, s6
	v_lshl_add_u32 v77, v106, 1, s5
	v_lshl_add_u32 v79, v108, 1, s5
	ds_load_b128 v[124:127], v68 offset:128
	ds_load_b128 v[128:131], v68 offset:144
	v_lshl_add_u32 v68, v110, 1, s5
	ds_load_b128 v[148:151], v77 offset:128
	ds_load_b128 v[152:155], v77 offset:144
	v_lshl_add_u32 v77, v112, 1, s5
	ds_load_b128 v[172:175], v79 offset:128
	ds_load_b128 v[220:223], v68 offset:128
	ds_load_b128 v[224:227], v68 offset:144
	v_lshl_add_u32 v68, v114, 1, s5
	ds_load_b128 v[176:179], v79 offset:144
	ds_load_b128 v[228:231], v77 offset:128
	v_wmma_f32_16x16x32_bf16 v[58:65], v[140:147], v[164:171], v[58:65]
	v_lshl_add_u32 v79, v118, 1, s5
	; sched_group_barrier mask(0x00000100) size(9) SyncID(0)
	s_wait_dscnt 0xb
	v_wmma_f32_16x16x32_bf16 v[50:57], v[140:147], v[156:163], v[50:57] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[42:49], v[140:147], v[180:187], v[42:49] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[26:33], v[140:147], v[188:195], v[26:33] matrix_a_reuse
	ds_load_b128 v[232:235], v77 offset:144
	v_lshl_add_u32 v77, v116, 1, s5
	ds_load_b128 v[156:159], v68 offset:12288
	ds_load_b128 v[160:163], v68 offset:12304
	v_lshl_add_u32 v68, v120, 1, s5
	ds_load_b128 v[180:183], v79 offset:128
	ds_load_b128 v[164:167], v77 offset:128
	ds_load_b128 v[168:171], v77 offset:144
	ds_load_b128 v[184:187], v79 offset:144
	ds_load_b128 v[188:191], v68 offset:128
	ds_load_b128 v[192:195], v68 offset:144
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[18:25], v[140:147], v[132:139], v[18:25] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(9) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[10:17], v[140:147], v[196:203], v[10:17] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[2:9], v[140:147], v[204:211], v[2:9] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[34:41], v[140:147], v[212:219], v[34:41] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[58:65], v[124:131], v[148:155], v[58:65]
	; sched_group_barrier mask(0x00000100) size(9) SyncID(0)
	s_wait_dscnt 0xa
	v_wmma_f32_16x16x32_bf16 v[50:57], v[124:131], v[172:179], v[50:57] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[42:49], v[124:131], v[220:227], v[42:49] matrix_a_reuse
	s_wait_dscnt 0x8
	v_wmma_f32_16x16x32_bf16 v[26:33], v[124:131], v[228:235], v[26:33] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(9) SyncID(0)
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[18:25], v[124:131], v[156:163], v[18:25] matrix_a_reuse
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x32_bf16 v[10:17], v[124:131], v[164:171], v[10:17] matrix_a_reuse
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[2:9], v[124:131], v[180:187], v[2:9] matrix_a_reuse
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[34:41], v[124:131], v[188:195], v[34:41] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
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
	s_and_b32 vcc_lo, exec_lo, s36
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_vccz .LBB0_39
; %bb.38:
	v_dual_lshrrev_b32 v1, 1, v0 :: v_dual_lshlrev_b32 v66, 7, v0
	v_cvt_pk_bf16_f32 v65, v64, v65
	v_cvt_pk_bf16_f32 v64, v62, v63
	v_cvt_pk_bf16_f32 v62, v58, v59
	v_cvt_pk_bf16_f32 v57, v56, v57
	v_bitop3_b32 v1, v66, 0x788, v1 bitop3:0xc8
	v_cvt_pk_bf16_f32 v56, v54, v55
	v_cvt_pk_bf16_f32 v55, v52, v53
	v_cvt_pk_bf16_f32 v49, v48, v49
	v_cvt_pk_bf16_f32 v54, v50, v51
	v_or_b32_e32 v58, 0x800, v1
	v_or_b32_e32 v50, 0x1000, v1
	v_cvt_pk_bf16_f32 v33, v32, v33
	v_cvt_pk_bf16_f32 v32, v30, v31
	v_cvt_pk_bf16_f32 v31, v28, v29
	v_lshrrev_b32_e32 v52, 3, v58
	v_or_b32_e32 v58, 0x1800, v1
	v_or_b32_e32 v28, 0x3000, v1
	v_dual_lshlrev_b32 v68, 4, v0 :: v_dual_lshlrev_b32 v51, 1, v1
	v_lshrrev_b32_e32 v50, 3, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_lshrrev_b32_e32 v48, 3, v58
	v_or_b32_e32 v58, 0x2000, v1
	v_cvt_pk_bf16_f32 v30, v26, v27
	v_cvt_pk_bf16_f32 v25, v24, v25
	v_cvt_pk_bf16_f32 v24, v22, v23
	v_and_b32_e32 v59, 0x3f0, v48
	v_cvt_pk_bf16_f32 v48, v46, v47
	v_cvt_pk_bf16_f32 v46, v42, v43
	v_or_b32_e32 v43, 0x2800, v1
	v_or_b32_e32 v1, 0x3800, v1
	v_cvt_pk_bf16_f32 v47, v44, v45
	v_lshrrev_b32_e32 v44, 3, v58
	v_cvt_pk_bf16_f32 v23, v20, v21
	v_dual_lshrrev_b32 v26, 3, v43 :: v_dual_lshrrev_b32 v20, 3, v28
	v_and_b32_e32 v53, 0xf0, v68
	v_lshl_or_b32 v51, v67, 5, v51
	v_and_b32_e32 v50, 0x2f0, v50
	v_lshrrev_b32_e32 v1, 3, v1
	v_and_b32_e32 v52, 0x1f0, v52
	v_and_b32_e32 v44, 0x4f0, v44
	v_and_b32_e32 v26, 0x5f0, v26
	v_cvt_pk_bf16_f32 v22, v18, v19
	v_and_b32_e32 v19, 0x6f0, v20
	v_cvt_pk_bf16_f32 v63, v60, v61
	v_dual_add_nc_u32 v53, v53, v51 :: v_dual_add_nc_u32 v50, v50, v51
	v_and_b32_e32 v1, 0x7f0, v1
	v_dual_add_nc_u32 v52, v52, v51 :: v_dual_add_nc_u32 v42, v59, v51
	v_dual_add_nc_u32 v27, v44, v51 :: v_dual_add_nc_u32 v18, v26, v51
	v_cvt_pk_bf16_f32 v17, v16, v17
	v_cvt_pk_bf16_f32 v16, v14, v15
	v_cvt_pk_bf16_f32 v15, v12, v13
	v_cvt_pk_bf16_f32 v14, v10, v11
	v_add_nc_u32_e32 v10, v19, v51
	v_cvt_pk_bf16_f32 v9, v8, v9
	v_cvt_pk_bf16_f32 v8, v6, v7
	v_cvt_pk_bf16_f32 v7, v4, v5
	v_cvt_pk_bf16_f32 v6, v2, v3
	ds_store_b128 v53, v[62:65]
	ds_store_b128 v52, v[54:57] offset:4096
	v_add_nc_u32_e32 v1, v1, v51
	v_cvt_pk_bf16_f32 v5, v40, v41
	v_cvt_pk_bf16_f32 v4, v38, v39
	v_cvt_pk_bf16_f32 v3, v36, v37
	v_cvt_pk_bf16_f32 v2, v34, v35
	ds_store_b128 v50, v[46:49] offset:8192
	ds_store_b128 v42, v[30:33] offset:12288
	ds_store_b128 v27, v[22:25] offset:16384
	ds_store_b128 v18, v[14:17] offset:20480
	ds_store_b128 v10, v[6:9] offset:24576
	ds_store_b128 v1, v[2:5] offset:28672
.LBB0_39:
	v_cmp_ne_u32_e32 vcc_lo, 1, v71
	;;#ASMSTART
	s_wait_dscnt 0
	;;#ASMEND
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_vccnz .LBB0_50
; %bb.40:
	s_wait_kmcnt 0x0
	s_mul_i32 s16, s27, s25
	s_mov_b32 s0, exec_lo
	v_cmpx_gt_i32_e64 s16, v0
	s_cbranch_execz .LBB0_50
; %bb.41:
	s_ashr_i32 s3, s2, 31
	v_nop
	v_xad_u32 v2, v0, -1, s16
	s_lshl_b64 s[0:1], s[2:3], 1
	s_ashr_i32 s21, s20, 31
	s_add_nc_u64 s[4:5], s[14:15], s[0:1]
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
	s_abs_i32 s14, s23
	v_lshrrev_b32_e32 v1, 8, v2
	s_cvt_f32_u32 s0, s14
	v_or_b32_e32 v3, 0x300, v0
	s_sub_co_i32 s1, 0, s14
	v_mov_b32_e32 v7, 0
	v_rcp_iflag_f32_e32 v2, s0
	v_add_nc_u32_e32 v8, 1, v1
	v_or_b32_e32 v1, 0x100, v0
	s_mov_b32 s13, 0
	s_mov_b32 s6, s20
	s_mov_b32 s7, s21
	v_and_b32_e32 v9, 0x1fffffc, v8
	s_mov_b32 s8, s20
	v_readfirstlane_b32 s0, v2
	v_or_b32_e32 v2, 0x200, v0
	s_mov_b32 s9, s21
	v_mov_b32_e32 v10, v9
	s_mov_b32 s10, s20
	s_mul_f32 s0, s0, 0x4f7ffffe
	v_mov_b64_e32 v[4:5], v[2:3]
	v_mov_b64_e32 v[2:3], v[0:1]
	s_mov_b32 s11, s21
	s_cvt_u32_f32 s0, s0
	s_mov_b32 s15, s23
	s_mov_b32 s17, s23
	s_mov_b32 s18, s23
	s_mul_i32 s1, s1, s0
	s_mov_b32 s19, s33
	s_mul_hi_u32 s1, s0, s1
	s_mov_b32 s22, s33
	s_mov_b32 s24, s33
	s_ashr_i32 s25, s23, 31
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
	v_mul_lo_u32 v24, v20, s14
	v_mul_lo_u32 v26, v21, s14
	v_mul_lo_u32 v27, v22, s14
	v_dual_add_nc_u32 v25, 1, v20 :: v_dual_add_nc_u32 v29, 1, v21
	v_mul_lo_u32 v28, v23, s14
	v_dual_add_nc_u32 v30, 1, v22 :: v_dual_add_nc_u32 v31, 1, v23
	v_dual_sub_nc_u32 v6, v6, v24 :: v_dual_sub_nc_u32 v12, v12, v26
	v_dual_sub_nc_u32 v16, v16, v27 :: v_dual_bitop2_b32 v14, s25, v14 bitop3:0x14
	v_xor_b32_e32 v18, s25, v18
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cmp_le_u32_e32 vcc_lo, s14, v6
	v_cmp_le_u32_e64 s0, s14, v12
	v_subrev_nc_u32_e32 v24, s14, v6
	v_cmp_le_u32_e64 s1, s14, v16
	v_subrev_nc_u32_e32 v26, s14, v16
	v_cndmask_b32_e32 v20, v20, v25, vcc_lo
	v_subrev_nc_u32_e32 v25, s14, v12
	v_dual_cndmask_b32 v21, v21, v29, s0 :: v_dual_sub_nc_u32 v19, v19, v28
	v_cndmask_b32_e64 v22, v22, v30, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_cndmask_b32 v6, v6, v24, vcc_lo :: v_dual_cndmask_b32 v12, v12, v25, s0
	v_dual_add_nc_u32 v25, 1, v21 :: v_dual_cndmask_b32 v16, v16, v26, s1
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_cmp_le_u32_e64 s2, s14, v19
	v_subrev_nc_u32_e32 v27, s14, v19
	v_cmp_le_u32_e32 vcc_lo, s14, v12
	v_dual_add_nc_u32 v26, 1, v22 :: v_dual_add_nc_u32 v24, 1, v20
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v23, v23, v31, s2
	v_dual_cndmask_b32 v19, v19, v27, s2 :: v_dual_cndmask_b32 v12, v21, v25, vcc_lo
	v_cmp_le_u32_e32 vcc_lo, s14, v16
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_add_nc_u32 v10, -4, v10 :: v_dual_add_nc_u32 v27, 1, v23
	v_dual_mov_b32 v13, v7 :: v_dual_bitop2_b32 v12, v12, v11 bitop3:0x14
	v_cndmask_b32_e32 v16, v22, v26, vcc_lo
	v_cmp_le_u32_e32 vcc_lo, s14, v6
	v_dual_mov_b32 v15, v7 :: v_dual_mov_b32 v17, v7
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sub_nc_u32_e32 v11, v12, v11
	v_xor_b32_e32 v16, v16, v14
	v_cndmask_b32_e32 v6, v20, v24, vcc_lo
	v_cmp_le_u32_e32 vcc_lo, s14, v19
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mul_lo_u32 v12, v11, s15
	v_dual_sub_nc_u32 v26, v16, v14 :: v_dual_bitop2_b32 v6, v6, v1 bitop3:0x14
	v_cndmask_b32_e32 v19, v23, v27, vcc_lo
	v_add_nc_u32_e32 v20, s19, v11
	v_cmp_eq_u32_e32 vcc_lo, 0, v10
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mul_lo_u32 v14, v26, s17
	v_dual_sub_nc_u32 v1, v6, v1 :: v_dual_bitop2_b32 v19, v19, v18 bitop3:0x14
	v_dual_add_nc_u32 v22, s22, v26 :: v_dual_sub_nc_u32 v12, v3, v12
	v_ashrrev_i32_e32 v21, 31, v20
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mul_lo_u32 v6, v1, s23
	v_dual_sub_nc_u32 v27, v19, v18 :: v_dual_add_nc_u32 v18, s33, v1
	v_sub_nc_u32_e32 v14, v4, v14
	v_ashrrev_i32_e32 v23, 31, v22
	v_lshl_add_u32 v11, v11, 7, v12
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_mul_lo_u32 v16, v27, s18
	v_dual_add_nc_u32 v24, s24, v27 :: v_dual_sub_nc_u32 v6, v2, v6
	v_ashrrev_i32_e32 v19, 31, v18
	v_lshl_add_u32 v26, v26, 7, v14
	v_mul_u64_e32 v[20:21], s[6:7], v[20:21]
	v_ashrrev_i32_e32 v25, 31, v24
	v_lshl_add_u32 v1, v1, 7, v6
	v_sub_nc_u32_e32 v16, v5, v16
	v_mul_u64_e32 v[18:19], s[20:21], v[18:19]
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
	v_dual_mov_b32 v6, s14 :: v_dual_mov_b32 v1, s25
	s_and_b32 s0, vcc_lo, exec_lo
	s_or_saveexec_b32 s1, s3
	v_mov_b64_e32 v[2:3], s[12:13]
	s_xor_b32 exec_lo, exec_lo, s1
	s_cbranch_execz .LBB0_43
.LBB0_47:
	s_abs_i32 s6, s23
	s_ashr_i32 s7, s23, 31
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
	v_dual_sub_nc_u32 v7, v4, v9 :: v_dual_lshlrev_b32 v4, 7, v4
	v_lshlrev_b32_e32 v9, 7, v9
	v_mul_lo_u32 v8, v7, s23
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_sub_nc_u32 v4, v4, v8 :: v_dual_add_nc_u32 v8, s33, v7
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
	v_cmp_le_i32_e32 vcc_lo, s16, v0
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
	.size	bm128_bn128_bk096_wm8_wn1_mc0, .Lfunc_end0-bm128_bn128_bk096_wm8_wn1_mc0
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel bm128_bn128_bk096_wm8_wn1_mc0
		.amdhsa_group_segment_fixed_size 104448
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
		.amdhsa_next_free_vgpr 236
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
		.amdhsa_inst_pref_size 54
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text.bm128_bn128_bk096_wm8_wn1_mc0,"axG",@progbits,bm128_bn128_bk096_wm8_wn1_mc0,comdat
                                        ; -- End function
	.set .Lbm128_bn128_bk096_wm8_wn1_mc0.num_vgpr, 236
	.set .Lbm128_bn128_bk096_wm8_wn1_mc0.num_agpr, 0
	.set .Lbm128_bn128_bk096_wm8_wn1_mc0.numbered_sgpr, 52
	.set .Lbm128_bn128_bk096_wm8_wn1_mc0.num_named_barrier, 0
	.set .Lbm128_bn128_bk096_wm8_wn1_mc0.private_seg_size, 0
	.set .Lbm128_bn128_bk096_wm8_wn1_mc0.uses_vcc, 1
	.set .Lbm128_bn128_bk096_wm8_wn1_mc0.uses_flat_scratch, 1
	.set .Lbm128_bn128_bk096_wm8_wn1_mc0.has_dyn_sized_stack, 0
	.set .Lbm128_bn128_bk096_wm8_wn1_mc0.has_recursion, 0
	.set .Lbm128_bn128_bk096_wm8_wn1_mc0.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 6872
; TotalNumSgprs: 54
; NumVgprs: 236
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 104448 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 14
; NumSGPRsForWavesPerEU: 54
; NumVGPRsForWavesPerEU: 236
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
	.type	__hip_cuid_eae1cb2e98f91d9b,@object ; @__hip_cuid_eae1cb2e98f91d9b
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_eae1cb2e98f91d9b
__hip_cuid_eae1cb2e98f91d9b:
	.byte	0                               ; 0x0
	.size	__hip_cuid_eae1cb2e98f91d9b, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git aa451e1fe6a793394d6733051b1778633063ae96+PATCHED:d17c5aa0e3ea29cde402f58f27e39b6034effa27)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_eae1cb2e98f91d9b
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
    .group_segment_fixed_size: 104448
    .kernarg_segment_align: 8
    .kernarg_segment_size: 132
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           bm128_bn128_bk096_wm8_wn1_mc0
    .private_segment_fixed_size: 0
    .sgpr_count:     54
    .sgpr_spill_count: 0
    .symbol:         bm128_bn128_bk096_wm8_wn1_mc0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     236
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
