	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.section	.text.bm128_bn128_bk096_wm1_wn8_mc1,"axG",@progbits,bm128_bn128_bk096_wm1_wn8_mc1,comdat
	.protected	bm128_bn128_bk096_wm1_wn8_mc1 ; -- Begin function bm128_bn128_bk096_wm1_wn8_mc1
	.globl	bm128_bn128_bk096_wm1_wn8_mc1
	.p2align	8
	.type	bm128_bn128_bk096_wm1_wn8_mc1,@function
bm128_bn128_bk096_wm1_wn8_mc1: ; @bm128_bn128_bk096_wm1_wn8_mc1
; %bb.0:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1 ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_load_b96 s[24:26], s[0:1], 0x78 nv
	s_mov_b64 s[2:3], src_shared_base
	s_movk_i32 s2, 0x6600
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_and_b64 s[2:3], s[2:3], 12
	s_sub_co_i32 s4, 16, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_lshr_b32 s4, s4, 2
	s_cmp_lg_u64 s[2:3], 0
	s_cselect_b32 s5, s4, 0
	s_and_b32 s2, ttmp6, 15
	s_bfe_u32 s3, ttmp6, 0x40004
	s_lshl2_add_u32 s37, ttmp9, s2
	s_lshl2_add_u32 s4, ttmp7, s3
	s_lshl_b32 s2, s37, 7
	s_wait_kmcnt 0x0
	s_add_co_i32 s3, s24, 0x7f
	s_add_co_i32 s6, s25, 0x7f
	s_sub_co_i32 s7, s24, s2
	s_ashr_i32 s8, s3, 31
	s_ashr_i32 s9, s6, 31
	s_min_i32 s27, s7, 0x80
	s_lshr_b32 s7, s8, 25
	s_lshr_b32 s8, s9, 25
	s_add_co_i32 s3, s3, s7
	s_add_co_i32 s7, s6, s8
	s_ashr_i32 s6, s3, 7
	s_ashr_i32 s7, s7, 7
	s_cmp_lt_i32 s37, s6
	s_mov_b32 s9, s26
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_and_b32 s8, s3, exec_lo
	s_cselect_b32 s29, s27, 0
	s_lshl_b32 s33, s4, 7
	s_sub_co_i32 s8, s25, s33
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_min_i32 s8, s8, 0x80
	s_cmp_lt_i32 s4, s7
	s_cselect_b32 s25, -1, 0
	s_and_b32 s10, s25, exec_lo
	s_cselect_b32 s31, s8, 0
	s_add_co_i32 s12, s26, 0x5f
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 2, 1), 1
	s_min_i32 s8, s26, 0x60
	s_cmp_gt_i32 s12, 0x5f
	s_cselect_b32 s13, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_and_b32 s10, s13, exec_lo
	s_cselect_b32 s28, s8, 0
	s_cmp_lt_i32 s29, 0x80
	s_cselect_b32 s40, -1, 0
	s_and_b32 vcc_lo, exec_lo, s40
	s_mov_b32 s8, s40
	s_cbranch_vccnz .LBB0_2
; %bb.1:
	s_cmp_lt_i32 s31, 0x80
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
	v_cmp_lt_u32_e32 vcc_lo, 0x187f, v5
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
	s_load_b64 s[14:15], s[0:1], 0x0 nv
	s_load_b128 s[16:19], s[0:1], 0x20 nv
	s_load_b128 s[20:23], s[0:1], 0x48 nv
	s_wait_xcnt 0x0
	s_lshl_b32 s1, s4, 2
	v_lshrrev_b32_e32 v71, 5, v0
	s_lshl_b32 s36, s5, 2
	s_add_co_i32 s7, s7, -1
	s_and_b32 s1, s1, 12
	s_mov_b64 s[34:35], src_shared_base
	s_or_b32 s34, s36, 0x6600
	s_add_co_i32 s38, s6, -1
	s_min_i32 s0, s4, s7
	s_and_b32 s39, s37, 3
	s_lshl_b32 s1, 15, s1
	s_mov_b32 s4, exec_lo
	v_cmpx_lt_i32_e32 0, v71
	s_xor_b32 s41, exec_lo, s4
	s_cbranch_execz .LBB0_12
; %bb.9:
	s_mov_b32 s42, exec_lo
	v_cmpx_eq_u32_e32 1, v71
	s_cbranch_execz .LBB0_11
; %bb.10:
	s_cmp_gt_i32 s28, 0
	s_mov_b32 s30, s28
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
	s_add_nc_u64 s[6:7], s[18:19], s[4:5]
	v_dual_mov_b32 v1, s34 :: v_dual_mov_b32 v4, s6
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
	s_or_b32 s4, s1, 0x7510000
	s_lshl_b32 s5, s28, 16
	s_or_b32 s7, s7, 0x600000
	s_movk_i32 s8, 0x80
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[44:47], s[4:11]
.LBB0_11:
	s_or_b32 exec_lo, exec_lo, s42
.LBB0_12:
	s_or_saveexec_b32 s41, s41
	s_min_i32 s38, s37, s38
	s_lshl_b32 s30, 0x1111, s39
	s_xor_b32 exec_lo, exec_lo, s41
	s_cbranch_execz .LBB0_14
; %bb.13:
	s_cmp_gt_i32 s28, 0
	s_mov_b32 s10, 0
	s_cselect_b32 s8, -1, 0
	s_lshl_b32 s4, s38, 7
	s_wait_kmcnt 0x0
	s_bfe_i64 s[6:7], s[16:17], 0x200000
	s_ashr_i32 s5, s4, 31
	s_and_b32 s8, s3, s8
	s_mul_u64 s[4:5], s[6:7], s[4:5]
	v_cndmask_b32_e64 v2, 0, 1, s8
	s_lshl_b64 s[6:7], s[4:5], 1
	s_lshr_b32 s8, s29, 16
	s_add_nc_u64 s[6:7], s[14:15], s[6:7]
	s_or_b32 s4, s30, 0x7510000
	s_and_b32 s7, s7, 0x1ffffff
	v_readfirstlane_b32 s44, v2
	s_bitset1_b32 s7, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v4, s6 :: v_dual_mov_b32 v3, s7
	s_lshr_b64 s[6:7], s[28:29], 16
	s_lshl_b32 s5, s28, 16
	s_or_b32 s7, s8, 0x600000
	v_readfirstlane_b32 s46, v4
	v_readfirstlane_b32 s47, v3
	s_movk_i32 s8, 0x80
	s_mov_b32 s11, s10
	s_mov_b32 s45, s10
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[44:47], s[4:11]
.LBB0_14:
	s_or_b32 exec_lo, exec_lo, s41
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
	v_dual_mov_b32 v9, 0 :: v_dual_lshlrev_b32 v73, 4, v71
	s_and_b32 s41, s3, s25
	s_and_not1_b32 vcc_lo, exec_lo, s13
	v_cndmask_b32_e64 v67, 0, 1, s41
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
	v_dual_mov_b32 v34, v9 :: v_dual_mov_b32 v57, v9
	v_dual_mov_b32 v56, v9 :: v_dual_mov_b32 v55, v9
	v_dual_mov_b32 v54, v9 :: v_dual_mov_b32 v53, v9
	v_dual_mov_b32 v52, v9 :: v_dual_mov_b32 v51, v9
	v_dual_mov_b32 v50, v9 :: v_dual_mov_b32 v65, v9
	v_dual_mov_b32 v64, v9 :: v_dual_mov_b32 v63, v9
	v_dual_mov_b32 v62, v9 :: v_dual_mov_b32 v61, v9
	v_dual_mov_b32 v60, v9 :: v_dual_mov_b32 v59, v9
	v_dual_mov_b32 v58, v9 :: v_dual_mov_b32 v49, v9
	v_dual_mov_b32 v48, v9 :: v_dual_mov_b32 v47, v9
	v_dual_mov_b32 v46, v9 :: v_dual_mov_b32 v45, v9
	v_dual_mov_b32 v44, v9 :: v_dual_mov_b32 v43, v9
	v_mov_b32_e32 v42, v9
	s_barrier_wait -1
	s_barrier_wait -3
	s_cbranch_vccnz .LBB0_41
; %bb.17:
	s_mov_b64 s[4:5], src_shared_base
	s_or_b32 s6, s36, 0xcc00
	s_mov_b32 s7, s5
	v_and_b32_e32 v5, 15, v0
	s_and_b64 s[6:7], s[6:7], 15
	s_mov_b32 s11, 0
	s_sub_co_i32 s4, 16, s6
	v_mul_u32_u24_e64 v3, 0x60, 0
	s_lshr_b32 s4, s4, 2
	s_cmp_lg_u64 s[6:7], 0
	v_mul_u32_u24_e32 v7, 0x60, v5
	s_cselect_b32 s4, s4, 0
	s_mul_hi_i32 s6, s12, 0x2aaaaaab
	s_lshl2_add_u32 s7, s4, s36
	v_mul_u32_u24_e32 v1, 0x60, v73
	s_add_co_i32 s4, s7, 0x13200
	v_lshrrev_b32_e32 v2, 4, v7
	s_and_b32 s10, s4, 15
	s_add_co_i32 s43, s7, 0xcc00
	s_sub_co_i32 s8, 16, s10
	v_or_b32_e32 v9, 0x1800, v7
	s_lshr_b32 s7, s8, 2
	v_and_b32_e32 v2, 0x78, v2
	v_and_b32_e32 v25, 16, v0
	s_cmp_lg_u64 s[10:11], 0
	v_mov_b32_e32 v69, 0
	s_cselect_b32 s7, s7, 0
	s_lshr_b32 s8, s6, 31
	s_ashr_i32 s45, s6, 4
	s_lshl_b32 s10, s7, 2
	s_add_co_i32 s45, s45, s8
	v_or3_b32 v29, v3, v25, v7
	s_cmp_lt_i32 s31, 0x80
	v_mov_b32_e32 v39, v69
	s_cselect_b32 s46, -1, 0
	s_lshl_b32 s6, s0, 7
	s_movk_i32 s0, 0x600
	v_add_nc_u32_e32 v66, v2, v29
	v_mad_u32_u24 v6, 0x60, v5, s0
	v_add_nc_u32_e32 v2, 0xc00, v29
	v_add_nc_u32_e32 v4, 0x1200, v29
	v_lshrrev_b32_e32 v11, 4, v9
	v_mad_u32_u24 v33, 0x60, v73, v7
	v_lshrrev_b32_e32 v6, 4, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_lshrrev_b32 v8, 4, v2 :: v_dual_lshrrev_b32 v10, 4, v4
	v_and_b32_e32 v11, 0x1f8, v11
	v_add_nc_u32_e32 v38, 0x600, v29
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_and_b32_e32 v13, 0xf8, v6
	v_and_b32_e32 v15, 0x1f8, v8
	v_add_nc_u32_e32 v6, 0x1e00, v29
	v_add_nc_u32_e32 v8, 0x2400, v29
	v_or_b32_e32 v18, 32, v25
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_dual_add_nc_u32 v72, v13, v29 :: v_dual_add_nc_u32 v74, v15, v29
	v_mov_b32_e32 v13, v69
	v_dual_lshrrev_b32 v16, 4, v6 :: v_dual_bitop2_b32 v1, v1, v25 bitop3:0x54
	v_dual_lshrrev_b32 v19, 4, v8 :: v_dual_bitop2_b32 v14, v3, v18 bitop3:0x54
	v_and_b32_e32 v17, 0x1f8, v10
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_mad_u32_u24 v1, 0x60, v5, v1
	v_dual_mov_b32 v15, v69 :: v_dual_add_nc_u32 v10, 0x2a00, v29
	v_mad_u32_u24 v21, 0x60, v5, v14
	v_and_b32_e32 v23, 0x3f8, v16
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_lshrrev_b32_e32 v12, 4, v1
	v_dual_add_nc_u32 v76, v17, v29 :: v_dual_add_nc_u32 v78, v11, v29
	v_add_nc_u32_e32 v7, 0x1e00, v21
	v_dual_mov_b32 v17, v69 :: v_dual_add_nc_u32 v14, 0x1200, v21
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_and_b32_e32 v12, 0x7f8, v12
	v_lshrrev_b32_e32 v20, 4, v10
	v_and_b32_e32 v19, 0x2f8, v19
	v_dual_lshrrev_b32 v14, 4, v14 :: v_dual_add_nc_u32 v80, v23, v29
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v11, v69 :: v_dual_add_nc_u32 v70, v12, v1
	v_add_nc_u32_e32 v1, 0x600, v21
	v_add_nc_u32_e32 v12, 0xc00, v21
	v_dual_add_nc_u32 v16, v18, v9 :: v_dual_add_nc_u32 v18, v33, v18
	v_and_b32_e32 v27, 0x3f8, v20
	v_lshrrev_b32_e32 v1, 4, v1
	v_lshrrev_b32_e32 v20, 4, v21
	v_dual_mov_b32 v19, v69 :: v_dual_add_nc_u32 v82, v19, v29
	v_and_b32_e32 v14, 0x3f8, v14
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_and_b32_e32 v68, 0x1f8, v1
	v_or_b32_e32 v1, 64, v25
	v_and_b32_e32 v31, 0xf8, v20
	v_add_nc_u32_e32 v20, 0x2400, v21
	v_add_nc_u32_e32 v21, 0x2a00, v21
	v_dual_lshrrev_b32 v24, 4, v18 :: v_dual_add_nc_u32 v9, v1, v9
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_lshrrev_b32 v20, 4, v20 :: v_dual_bitop2_b32 v3, v3, v1 bitop3:0x54
	v_lshrrev_b32_e32 v21, 4, v21
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_and_b32_e32 v35, 0x7f8, v24
	v_dual_mov_b32 v27, v69 :: v_dual_add_nc_u32 v84, v27, v29
	v_mad_u32_u24 v3, 0x60, v5, v3
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v22, 0x3f8, v21
	v_mov_b32_e32 v23, v69
	v_add_nc_u64_e32 v[88:89], v[68:69], v[38:39]
	v_or_b32_e32 v68, 0x1800, v29
	v_add_nc_u32_e32 v5, 0x600, v3
	v_lshrrev_b32_e32 v24, 4, v3
	v_add_nc_u32_e32 v21, 0x1200, v3
	s_or_b32 s12, s1, 0x7510000
	v_and_b32_e32 v20, 0x3f8, v20
	v_lshrrev_b32_e32 v5, 4, v5
	v_and_b32_e32 v37, 0xf8, v24
	v_lshrrev_b32_e32 v21, 4, v21
	v_add_nc_u32_e32 v86, v31, v29
	s_lshl_b32 s38, s38, 7
	v_and_b32_e32 v24, 0x1f8, v5
	v_add_nc_u32_e32 v5, 0x1e00, v3
	v_and_b32_e32 v28, 0x3f8, v21
	v_add_nc_u32_e32 v104, v37, v29
	v_mov_b32_e32 v29, v69
	s_ashr_i32 s7, s6, 31
	v_lshrrev_b32_e32 v5, 4, v5
	s_ashr_i32 s39, s38, 31
	s_wait_kmcnt 0x0
	s_bfe_i64 s[20:21], s[20:21], 0x200000
	v_lshrrev_b32_e32 v9, 4, v9
	s_bfe_i64 s[16:17], s[16:17], 0x200000
	v_and_b32_e32 v32, 0x3f8, v5
	v_dual_mov_b32 v5, v69 :: v_dual_lshrrev_b32 v12, 4, v12
	v_lshrrev_b32_e32 v16, 4, v16
	v_and_b32_e32 v30, 0x3f8, v9
	v_sub_nc_u32_e32 v9, 0x197f, v0
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u64_e32 v[92:93], v[14:15], v[4:5]
	v_add_nc_u64_e32 v[110:111], v[28:29], v[4:5]
	v_dual_mov_b32 v4, v69 :: v_dual_lshrrev_b32 v7, 4, v7
	v_and_b32_e32 v16, 0x3f8, v16
	v_lshrrev_b32_e32 v9, 8, v9
	v_and_b32_e32 v12, 0x1f8, v12
	s_mul_u64 s[6:7], s[20:21], s[6:7]
	v_and_b32_e32 v18, 0x3f8, v7
	v_add_nc_u32_e32 v7, 0xc00, v3
	v_add_nc_u64_e32 v[94:95], v[16:17], v[68:69]
	v_mov_b32_e32 v16, v69
	s_mul_u64 s[16:17], s[16:17], s[38:39]
	v_add_nc_u64_e32 v[100:101], v[22:23], v[10:11]
	v_lshrrev_b32_e32 v7, 4, v7
	v_cmp_eq_u32_e64 s0, 0, v71
	v_dual_mov_b32 v123, v69 :: v_dual_mov_b32 v21, v69
	v_mov_b32_e32 v14, v69
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_and_b32_e32 v26, 0x1f8, v7
	v_add_nc_u32_e32 v7, 0x2400, v3
	v_add_nc_u32_e32 v3, 0x2a00, v3
	v_dual_mov_b32 v51, v69 :: v_dual_mov_b32 v52, v69
	v_dual_mov_b32 v53, v69 :: v_dual_mov_b32 v54, v69
	v_dual_lshrrev_b32 v7, 4, v7 :: v_dual_mov_b32 v55, v69
	v_dual_mov_b32 v56, v69 :: v_dual_mov_b32 v57, v69
	v_mov_b32_e32 v58, v69
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_and_b32_e32 v34, 0x3f8, v7
	v_dual_mov_b32 v7, v69 :: v_dual_mov_b32 v22, v69
	v_dual_mov_b32 v28, v69 :: v_dual_mov_b32 v59, v69
	v_mov_b32_e32 v60, v69
	v_add_nc_u64_e32 v[96:97], v[18:19], v[6:7]
	v_mov_b32_e32 v18, v69
	v_dual_add_nc_u32 v1, v33, v1 :: v_dual_lshrrev_b32 v3, 4, v3
	v_dual_mov_b32 v31, v69 :: v_dual_mov_b32 v37, v69
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v50, v69 :: v_dual_lshrrev_b32 v1, 4, v1
	v_and_b32_e32 v36, 0x3f8, v3
	v_dual_add_nc_u32 v3, 1, v9 :: v_dual_mov_b32 v9, v69
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_nc_u64_e32 v[112:113], v[30:31], v[68:69]
	v_and_b32_e32 v40, 0x7f8, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_nc_u64_e32 v[118:119], v[36:37], v[10:11]
	v_and_b32_e32 v75, 26, v3
	v_or_b32_e32 v1, 0x100, v0
	v_dual_mov_b32 v10, v69 :: v_dual_mov_b32 v30, v69
	v_mov_b32_e32 v36, v69
	s_delay_alu instid0(VALU_DEP_4)
	v_cmp_ne_u32_e64 s1, v3, v75
	v_mov_b32_e32 v3, v69
	v_lshl_or_b32 v41, v75, 8, v0
	v_add_nc_u64_e32 v[98:99], v[20:21], v[8:9]
	v_dual_mov_b32 v20, v69 :: v_dual_mov_b32 v61, v69
	v_mov_b32_e32 v62, v69
	v_add_nc_u64_e32 v[90:91], v[12:13], v[2:3]
	v_or_b32_e32 v12, v33, v25
	v_dual_mov_b32 v25, v69 :: v_dual_mov_b32 v33, v69
	v_add_nc_u64_e32 v[108:109], v[26:27], v[2:3]
	v_add_nc_u32_e32 v77, 0xffffff00, v41
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v35, v69 :: v_dual_add_nc_u32 v102, v35, v12
	v_add_nc_u64_e32 v[106:107], v[24:25], v[38:39]
	v_add_nc_u64_e32 v[114:115], v[32:33], v[6:7]
	v_dual_add_nc_u32 v120, v40, v12 :: v_dual_lshlrev_b32 v122, 2, v41
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u64_e32 v[116:117], v[34:35], v[8:9]
	v_dual_mov_b32 v2, v69 :: v_dual_mov_b32 v6, v69
	v_dual_mov_b32 v8, v69 :: v_dual_mov_b32 v12, v69
	v_dual_mov_b32 v24, v69 :: v_dual_mov_b32 v26, v69
	v_dual_mov_b32 v32, v69 :: v_dual_mov_b32 v34, v69
	v_dual_mov_b32 v38, v69 :: v_dual_mov_b32 v40, v69
	v_dual_mov_b32 v41, v69 :: v_dual_mov_b32 v63, v69
	v_dual_mov_b32 v64, v69 :: v_dual_mov_b32 v65, v69
	v_dual_mov_b32 v42, v69 :: v_dual_mov_b32 v43, v69
	v_dual_mov_b32 v44, v69 :: v_dual_mov_b32 v45, v69
	v_dual_mov_b32 v46, v69 :: v_dual_mov_b32 v47, v69
	v_dual_mov_b32 v48, v69 :: v_dual_mov_b32 v49, v69
	s_lshr_b32 s47, s31, 16
	s_lshr_b32 s48, s29, 16
	s_lshl_b64 s[6:7], s[6:7], 1
	s_lshl_b64 s[16:17], s[16:17], 1
	s_mov_b32 s42, s35
	s_mov_b32 s44, s5
	s_add_nc_u64 s[36:37], s[4:5], s[10:11]
	s_movk_i32 s8, 0x80
	s_or_b32 s47, s47, 0x600000
	s_or_b32 s4, s30, 0x7510000
	s_or_b32 s48, s48, 0x600000
	s_add_nc_u64 s[20:21], s[18:19], s[6:7]
	s_add_nc_u64 s[38:39], s[14:15], s[16:17]
	s_mov_b32 s49, s11
	s_branch .LBB0_19
.LBB0_18:                               ;   in Loop: Header=BB0_19 Depth=1
	s_or_b32 exec_lo, exec_lo, s5
	s_cmp_eq_u32 s49, s45
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
	s_cbranch_vccnz .LBB0_31
; %bb.20:                               ;   in Loop: Header=BB0_19 Depth=1
	v_nop
	v_nop
	v_mov_b64_e32 v[124:125], v[0:1]
	v_mov_b32_e32 v79, v75
	s_cmp_lg_u32 s5, 0
	s_mov_b32 s13, 0
	s_cselect_b32 s7, s44, s35
	s_cselect_b32 s6, s43, 0
.LBB0_21:                               ;   Parent Loop BB0_19 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_dual_mov_b32 v68, v124 :: v_dual_add_nc_u32 v79, -2, v79
	v_dual_mov_b32 v126, v125 :: v_dual_mov_b32 v127, v69
	v_add_nc_u32_e32 v125, 0x200, v125
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u64 v[128:129], v[68:69], 2, s[6:7]
	v_cmp_eq_u32_e32 vcc_lo, 0, v79
	v_add_nc_u32_e32 v124, 0x200, v124
	v_lshl_add_u64 v[126:127], v[126:127], 2, s[6:7]
	s_clause 0x1
	flat_store_b32 v[128:129], v69
	flat_store_b32 v[126:127], v69
	s_or_b32 s13, vcc_lo, s13
	s_wait_xcnt 0x0
	s_and_not1_b32 exec_lo, exec_lo, s13
	s_cbranch_execnz .LBB0_21
; %bb.22:                               ;   in Loop: Header=BB0_19 Depth=1
	s_or_b32 exec_lo, exec_lo, s13
	s_and_saveexec_b32 s13, s1
	s_cbranch_execz .LBB0_25
; %bb.23:                               ;   in Loop: Header=BB0_19 Depth=1
	v_add_nc_u64_e32 v[124:125], s[6:7], v[122:123]
	v_mov_b32_e32 v68, v77
	s_mov_b32 s6, 0
.LBB0_24:                               ;   Parent Loop BB0_19 Depth=1
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
	s_cbranch_execnz .LBB0_24
.LBB0_25:                               ;   in Loop: Header=BB0_19 Depth=1
	s_or_b32 exec_lo, exec_lo, s13
	v_mov_b64_e32 v[124:125], v[0:1]
	v_mov_b32_e32 v79, v75
	s_cmp_lg_u32 s5, 0
	s_mov_b32 s13, 0
	s_cselect_b32 s7, s37, s42
	s_cselect_b32 s6, s36, s34
.LBB0_26:                               ;   Parent Loop BB0_19 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_dual_mov_b32 v68, v124 :: v_dual_add_nc_u32 v79, -2, v79
	v_dual_mov_b32 v126, v125 :: v_dual_mov_b32 v127, v69
	v_add_nc_u32_e32 v125, 0x200, v125
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u64 v[128:129], v[68:69], 2, s[6:7]
	v_cmp_eq_u32_e32 vcc_lo, 0, v79
	v_add_nc_u32_e32 v124, 0x200, v124
	v_lshl_add_u64 v[126:127], v[126:127], 2, s[6:7]
	s_clause 0x1
	flat_store_b32 v[128:129], v69
	flat_store_b32 v[126:127], v69
	s_or_b32 s13, vcc_lo, s13
	s_wait_xcnt 0x0
	s_and_not1_b32 exec_lo, exec_lo, s13
	s_cbranch_execnz .LBB0_26
; %bb.27:                               ;   in Loop: Header=BB0_19 Depth=1
	s_or_b32 exec_lo, exec_lo, s13
	s_and_saveexec_b32 s13, s1
	s_cbranch_execz .LBB0_30
; %bb.28:                               ;   in Loop: Header=BB0_19 Depth=1
	v_add_nc_u64_e32 v[124:125], s[6:7], v[122:123]
	v_mov_b32_e32 v68, v77
	s_mov_b32 s6, 0
.LBB0_29:                               ;   Parent Loop BB0_19 Depth=1
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
	s_cbranch_execnz .LBB0_29
.LBB0_30:                               ;   in Loop: Header=BB0_19 Depth=1
	s_or_b32 exec_lo, exec_lo, s13
	s_barrier_signal -1
	s_barrier_wait -1
.LBB0_31:                               ;   in Loop: Header=BB0_19 Depth=1
	s_and_b32 s6, s10, exec_lo
	s_cselect_b32 s6, s49, 0
	s_mov_b32 s7, exec_lo
	v_cmpx_lt_i32_e32 0, v71
	s_xor_b32 s7, exec_lo, s7
	s_cbranch_execnz .LBB0_37
; %bb.32:                               ;   in Loop: Header=BB0_19 Depth=1
	s_and_not1_saveexec_b32 s13, s7
	s_cbranch_execnz .LBB0_40
.LBB0_33:                               ;   in Loop: Header=BB0_19 Depth=1
	s_or_b32 exec_lo, exec_lo, s13
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s41
	s_cbranch_vccnz .LBB0_35
.LBB0_34:                               ;   in Loop: Header=BB0_19 Depth=1
	s_cmp_lg_u32 s50, 0
	s_cselect_b32 s5, s43, 0
	s_cselect_b32 s6, s36, s34
	v_lshl_add_u32 v68, v66, 1, s5
	v_lshl_add_u32 v79, v72, 1, s5
	v_lshl_add_u32 v81, v74, 1, s5
	v_lshl_add_u32 v83, v70, 1, s6
	v_lshl_add_u32 v85, v94, 1, s5
	ds_load_b128 v[124:127], v68
	ds_load_b128 v[128:131], v68 offset:16
	ds_load_b128 v[132:135], v79 offset:3072
	ds_load_b128 v[136:139], v79 offset:3088
	v_lshl_add_u32 v68, v76, 1, s5
	v_lshl_add_u32 v79, v78, 1, s5
	ds_load_b128 v[140:143], v81 offset:6144
	ds_load_b128 v[144:147], v81 offset:6160
	v_lshl_add_u32 v81, v84, 1, s5
	ds_load_b128 v[148:151], v68 offset:9216
	ds_load_b128 v[152:155], v68 offset:9232
	ds_load_b128 v[156:159], v79 offset:12288
	ds_load_b128 v[168:171], v83
	ds_load_b128 v[172:175], v83 offset:16
	v_lshl_add_u32 v68, v82, 1, s5
	v_lshl_add_u32 v83, v92, 1, s5
	v_lshl_add_u32 v87, v96, 1, s5
	v_lshl_add_u32 v89, v98, 1, s5
	v_lshl_add_u32 v91, v100, 1, s5
	v_lshl_add_u32 v93, v102, 1, s6
	ds_load_b128 v[184:187], v83 offset:64
	ds_load_b128 v[188:191], v83 offset:80
	ds_load_b128 v[192:195], v85 offset:64
	ds_load_b128 v[196:199], v85 offset:80
	ds_load_b128 v[200:203], v89 offset:64
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[58:65], v[124:131], v[168:175], v[58:65]
	ds_load_b128 v[204:207], v89 offset:80
	ds_load_b128 v[208:211], v91 offset:64
	ds_load_b128 v[212:215], v91 offset:80
	ds_load_b128 v[216:219], v93 offset:64
	ds_load_b128 v[220:223], v93 offset:80
	; sched_group_barrier mask(0x00000100) size(9) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[50:57], v[132:139], v[168:175], v[50:57] matrix_b_reuse
	ds_load_b128 v[132:135], v87 offset:64
	ds_load_b128 v[136:139], v87 offset:80
	v_wmma_f32_16x16x32_bf16 v[34:41], v[140:147], v[168:175], v[34:41] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[26:33], v[148:155], v[168:175], v[26:33] matrix_b_reuse
	ds_load_b128 v[160:163], v81 offset:21504
	ds_load_b128 v[164:167], v81 offset:21520
	v_lshl_add_u32 v81, v90, 1, s5
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	ds_load_b128 v[124:127], v81 offset:64
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x32_bf16 v[42:49], v[160:167], v[168:175], v[42:49] matrix_b_reuse
	ds_load_b128 v[160:163], v68 offset:18432
	ds_load_b128 v[164:167], v68 offset:18448
	v_lshl_add_u32 v68, v80, 1, s5
	ds_load_b128 v[128:131], v81 offset:80
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x32_bf16 v[2:9], v[160:167], v[168:175], v[2:9] matrix_b_reuse
	ds_load_b128 v[160:163], v68 offset:15360
	ds_load_b128 v[164:167], v68 offset:15376
	v_lshl_add_u32 v68, v86, 1, s5
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[10:17], v[160:167], v[168:175], v[10:17] matrix_b_reuse
	ds_load_b128 v[160:163], v79 offset:12304
	v_lshl_add_u32 v79, v88, 1, s5
	; sched_group_barrier mask(0x00000100) size(9) SyncID(0)
	ds_load_b128 v[176:179], v79 offset:64
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x32_bf16 v[18:25], v[156:163], v[168:175], v[18:25] matrix_b_reuse
	ds_load_b128 v[156:159], v68 offset:64
	ds_load_b128 v[160:163], v68 offset:80
	ds_load_b128 v[180:183], v79 offset:80
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_barrier mask(0x00000000)
	v_lshl_add_u32 v68, v104, 1, s5
	v_lshl_add_u32 v79, v106, 1, s5
	v_lshl_add_u32 v81, v108, 1, s5
	ds_load_b128 v[140:143], v68 offset:128
	ds_load_b128 v[144:147], v68 offset:144
	v_lshl_add_u32 v68, v110, 1, s5
	ds_load_b128 v[148:151], v79 offset:128
	ds_load_b128 v[152:155], v79 offset:144
	v_lshl_add_u32 v79, v112, 1, s5
	ds_load_b128 v[164:167], v81 offset:128
	ds_load_b128 v[224:227], v68 offset:128
	ds_load_b128 v[228:231], v68 offset:144
	v_lshl_add_u32 v68, v114, 1, s5
	ds_load_b128 v[168:171], v81 offset:144
	ds_load_b128 v[172:175], v79 offset:128
	s_wait_dscnt 0xa
	v_wmma_f32_16x16x32_bf16 v[58:65], v[156:163], v[216:223], v[58:65]
	v_lshl_add_u32 v81, v118, 1, s5
	; sched_group_barrier mask(0x00000100) size(9) SyncID(0)
	s_wait_dscnt 0x9
	v_wmma_f32_16x16x32_bf16 v[50:57], v[176:183], v[216:223], v[50:57] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[34:41], v[124:131], v[216:223], v[34:41] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[26:33], v[184:191], v[216:223], v[26:33] matrix_b_reuse
	ds_load_b128 v[176:179], v79 offset:144
	v_lshl_add_u32 v79, v116, 1, s5
	ds_load_b128 v[124:127], v68 offset:128
	ds_load_b128 v[128:131], v68 offset:144
	v_lshl_add_u32 v68, v120, 1, s6
	ds_load_b128 v[180:183], v81 offset:128
	ds_load_b128 v[156:159], v79 offset:128
	ds_load_b128 v[160:163], v79 offset:144
	ds_load_b128 v[184:187], v81 offset:144
	ds_load_b128 v[232:235], v68 offset:128
	ds_load_b128 v[236:239], v68 offset:144
	v_wmma_f32_16x16x32_bf16 v[18:25], v[192:199], v[216:223], v[18:25] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(9) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[10:17], v[132:139], v[216:223], v[10:17] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[2:9], v[200:207], v[216:223], v[2:9] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[42:49], v[208:215], v[216:223], v[42:49] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[58:65], v[140:147], v[232:239], v[58:65]
	; sched_group_barrier mask(0x00000100) size(9) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[50:57], v[148:155], v[232:239], v[50:57] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[34:41], v[164:171], v[232:239], v[34:41] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[26:33], v[224:231], v[232:239], v[26:33] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(9) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[18:25], v[172:179], v[232:239], v[18:25] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[10:17], v[124:131], v[232:239], v[10:17] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[2:9], v[156:163], v[232:239], v[2:9] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[42:49], v[180:187], v[232:239], v[42:49] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
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
	v_cmpx_eq_u32_e32 1, v71
	s_cbranch_execz .LBB0_39
; %bb.38:                               ;   in Loop: Header=BB0_19 Depth=1
	s_cmp_lg_u32 s5, 0
	s_mul_i32 s10, s6, 0x60
	s_cselect_b32 s13, s36, s34
	s_cmp_gt_i32 s28, 0
	s_mov_b32 s30, s28
	s_cselect_b32 s16, -1, 0
	s_lshl_b64 s[14:15], s[10:11], 1
	s_mov_b32 s17, s9
	s_add_nc_u64 s[14:15], s[20:21], s[14:15]
	s_delay_alu instid0(SALU_CYCLE_1)
	v_dual_mov_b32 v79, s13 :: v_dual_mov_b32 v124, s14
	s_and_b32 s10, s15, 0x1ffffff
	s_and_b32 s15, s25, s16
	s_bitset1_b32 s10, 31
	v_cndmask_b32_e64 v68, 0, 1, s15
	v_mov_b32_e32 v81, s10
	v_readfirstlane_b32 s53, v79
	v_readfirstlane_b32 s54, v124
	s_lshr_b64 s[14:15], s[30:31], 16
	v_readfirstlane_b32 s52, v68
	v_readfirstlane_b32 s55, v81
	s_lshl_b32 s13, s28, 16
	s_mov_b32 s15, s47
	s_mov_b32 s16, s8
	s_mov_b32 s18, s11
	s_mov_b32 s19, s11
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[52:55], s[12:19]
.LBB0_39:                               ;   in Loop: Header=BB0_19 Depth=1
	s_or_b32 exec_lo, exec_lo, s51
	s_and_not1_saveexec_b32 s13, s7
	s_cbranch_execz .LBB0_33
.LBB0_40:                               ;   in Loop: Header=BB0_19 Depth=1
	s_cmp_lg_u32 s5, 0
	s_mul_i32 s10, s6, 0x60
	s_cselect_b32 s5, s43, 0
	s_cmp_gt_i32 s28, 0
	s_cselect_b32 s14, -1, 0
	s_lshl_b64 s[6:7], s[10:11], 1
	s_and_b32 s10, s3, s14
	s_add_nc_u64 s[6:7], s[38:39], s[6:7]
	v_cndmask_b32_e64 v68, 0, 1, s10
	s_and_b32 s7, s7, 0x1ffffff
	v_dual_mov_b32 v79, s5 :: v_dual_mov_b32 v124, s6
	s_bitset1_b32 s7, 31
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_readfirstlane_b32 s16, v68
	v_mov_b32_e32 v81, s7
	v_readfirstlane_b32 s17, v79
	v_readfirstlane_b32 s18, v124
	s_lshr_b64 s[6:7], s[28:29], 16
	s_lshl_b32 s5, s28, 16
	v_readfirstlane_b32 s19, v81
	s_mov_b32 s7, s48
	s_mov_b32 s10, s11
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	tensor_load_to_lds s[16:19], s[4:11]
	s_or_b32 exec_lo, exec_lo, s13
	s_and_not1_b32 vcc_lo, exec_lo, s41
	s_cbranch_vccz .LBB0_34
	s_branch .LBB0_35
.LBB0_41:
	s_wait_tensorcnt 0x0
	;;#ASMSTART
	s_wait_dscnt 0
	;;#ASMEND
	s_and_b32 vcc_lo, exec_lo, s41
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_vccz .LBB0_43
; %bb.42:
	v_and_or_b32 v1, v0, 15, v73
	v_lshrrev_b32_e32 v66, 1, v0
	v_cvt_pk_bf16_f32 v65, v64, v65
	v_cvt_pk_bf16_f32 v64, v62, v63
	v_cvt_pk_bf16_f32 v62, v58, v59
	v_lshlrev_b32_e32 v1, 7, v1
	v_cvt_pk_bf16_f32 v63, v60, v61
	v_cvt_pk_bf16_f32 v41, v40, v41
	v_cvt_pk_bf16_f32 v40, v38, v39
	v_cvt_pk_bf16_f32 v39, v36, v37
	v_and_or_b32 v58, v66, 8, v1
	v_lshrrev_b32_e32 v1, 3, v1
	v_cvt_pk_bf16_f32 v38, v34, v35
	v_cvt_pk_bf16_f32 v57, v56, v57
	v_cvt_pk_bf16_f32 v56, v54, v55
	v_cvt_pk_bf16_f32 v55, v52, v53
	v_lshl_add_u32 v1, v58, 1, v1
	v_cvt_pk_bf16_f32 v54, v50, v51
	v_cvt_pk_bf16_f32 v33, v32, v33
	v_cvt_pk_bf16_f32 v32, v30, v31
	v_cvt_pk_bf16_f32 v31, v28, v29
	v_cvt_pk_bf16_f32 v30, v26, v27
	v_cvt_pk_bf16_f32 v25, v24, v25
	v_cvt_pk_bf16_f32 v24, v22, v23
	v_cvt_pk_bf16_f32 v23, v20, v21
	v_cvt_pk_bf16_f32 v22, v18, v19
	v_cvt_pk_bf16_f32 v17, v16, v17
	v_cvt_pk_bf16_f32 v16, v14, v15
	v_cvt_pk_bf16_f32 v15, v12, v13
	v_cvt_pk_bf16_f32 v14, v10, v11
	v_cvt_pk_bf16_f32 v9, v8, v9
	v_cvt_pk_bf16_f32 v8, v6, v7
	v_cvt_pk_bf16_f32 v7, v4, v5
	v_cvt_pk_bf16_f32 v6, v2, v3
	ds_store_b128 v1, v[62:65]
	ds_store_b128 v1, v[54:57] offset:32
	v_cvt_pk_bf16_f32 v5, v48, v49
	v_cvt_pk_bf16_f32 v4, v46, v47
	v_cvt_pk_bf16_f32 v3, v44, v45
	v_cvt_pk_bf16_f32 v2, v42, v43
	ds_store_b128 v1, v[38:41] offset:64
	ds_store_b128 v1, v[30:33] offset:96
	ds_store_b128 v1, v[22:25] offset:128
	ds_store_b128 v1, v[14:17] offset:160
	ds_store_b128 v1, v[6:9] offset:192
	ds_store_b128 v1, v[2:5] offset:224
.LBB0_43:
	v_cmp_ne_u32_e32 vcc_lo, 1, v67
	;;#ASMSTART
	s_wait_dscnt 0
	;;#ASMEND
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
	v_nop
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
	v_lshl_add_u32 v11, v11, 7, v12
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_mul_lo_u32 v16, v27, s18
	v_dual_add_nc_u32 v24, s21, v27 :: v_dual_sub_nc_u32 v6, v2, v6
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
	v_dual_sub_nc_u32 v7, v4, v9 :: v_dual_lshlrev_b32 v4, 7, v4
	v_lshlrev_b32_e32 v9, 7, v9
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
	.size	bm128_bn128_bk096_wm1_wn8_mc1, .Lfunc_end0-bm128_bn128_bk096_wm1_wn8_mc1
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel bm128_bn128_bk096_wm1_wn8_mc1
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
		.amdhsa_next_free_vgpr 240
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
	.section	.text.bm128_bn128_bk096_wm1_wn8_mc1,"axG",@progbits,bm128_bn128_bk096_wm1_wn8_mc1,comdat
                                        ; -- End function
	.set .Lbm128_bn128_bk096_wm1_wn8_mc1.num_vgpr, 240
	.set .Lbm128_bn128_bk096_wm1_wn8_mc1.num_agpr, 0
	.set .Lbm128_bn128_bk096_wm1_wn8_mc1.numbered_sgpr, 56
	.set .Lbm128_bn128_bk096_wm1_wn8_mc1.num_named_barrier, 0
	.set .Lbm128_bn128_bk096_wm1_wn8_mc1.private_seg_size, 0
	.set .Lbm128_bn128_bk096_wm1_wn8_mc1.uses_vcc, 1
	.set .Lbm128_bn128_bk096_wm1_wn8_mc1.uses_flat_scratch, 1
	.set .Lbm128_bn128_bk096_wm1_wn8_mc1.has_dyn_sized_stack, 0
	.set .Lbm128_bn128_bk096_wm1_wn8_mc1.has_recursion, 0
	.set .Lbm128_bn128_bk096_wm1_wn8_mc1.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 6652
; TotalNumSgprs: 58
; NumVgprs: 240
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 104448 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 14
; NumSGPRsForWavesPerEU: 58
; NumVGPRsForWavesPerEU: 240
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
	.type	__hip_cuid_7829365880a385e0,@object ; @__hip_cuid_7829365880a385e0
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_7829365880a385e0
__hip_cuid_7829365880a385e0:
	.byte	0                               ; 0x0
	.size	__hip_cuid_7829365880a385e0, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git aa451e1fe6a793394d6733051b1778633063ae96+PATCHED:d17c5aa0e3ea29cde402f58f27e39b6034effa27)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_7829365880a385e0
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
    macrotile: [128, 128, 96]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, One]
  MatrixInstruction: [16, 16, 32, 1]
  EnableMatrixInstruction: True
  MIWaveTile: [4, 2]
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
    .group_segment_fixed_size: 104448
    .kernarg_segment_align: 8
    .kernarg_segment_size: 132
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           bm128_bn128_bk096_wm1_wn8_mc1
    .private_segment_fixed_size: 0
    .sgpr_count:     58
    .sgpr_spill_count: 0
    .symbol:         bm128_bn128_bk096_wm1_wn8_mc1.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     240
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
