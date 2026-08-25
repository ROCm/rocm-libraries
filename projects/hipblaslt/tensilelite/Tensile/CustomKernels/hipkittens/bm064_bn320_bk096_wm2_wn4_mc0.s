	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.section	.text.bm064_bn320_bk096_wm2_wn4_mc0,"axG",@progbits,bm064_bn320_bk096_wm2_wn4_mc0,comdat
	.protected	bm064_bn320_bk096_wm2_wn4_mc0 ; -- Begin function bm064_bn320_bk096_wm2_wn4_mc0
	.globl	bm064_bn320_bk096_wm2_wn4_mc0
	.p2align	8
	.type	bm064_bn320_bk096_wm2_wn4_mc0,@function
bm064_bn320_bk096_wm2_wn4_mc0: ; @bm064_bn320_bk096_wm2_wn4_mc0
; %bb.0:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1 ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_mov_b64 s[2:3], src_shared_base
	s_movk_i32 s2, 0x3300
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
	s_add_co_i32 s2, s24, 63
	s_add_co_i32 s5, s25, 0x13f
	s_ashr_i32 s3, s2, 31
	s_lshl_b32 s28, s36, 6
	s_lshr_b32 s3, s3, 26
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_co_i32 s2, s2, s3
	s_mul_hi_i32 s3, s5, 0x66666667
	s_ashr_i32 s5, s2, 6
	s_lshr_b32 s2, s3, 31
	s_ashr_i32 s6, s3, 7
	s_sub_co_i32 s3, s24, s28
	s_add_co_i32 s6, s6, s2
	s_min_i32 s27, s3, 64
	s_cmp_lt_i32 s36, s5
	s_cselect_b32 s29, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_and_b32 s2, s29, exec_lo
	s_mul_i32 s2, s33, 0xfffffec0
	s_cselect_b32 s31, s27, 0
	s_add_co_i32 s2, s25, s2
	s_min_i32 s2, s2, 0x140
	s_cmp_lt_i32 s33, s6
	s_cselect_b32 s25, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s3, s25, exec_lo
	s_cselect_b32 s3, s2, 0
	s_add_co_i32 s16, s26, 0x5f
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 2, 1), 1
	s_min_i32 s2, s26, 0x60
	s_cmp_gt_i32 s16, 0x5f
	s_cselect_b32 s17, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)
	s_and_b32 s7, s17, exec_lo
	s_cselect_b32 s30, s2, 0
	s_cmp_lt_i32 s31, 64
	s_mov_b32 s2, -1
	s_cselect_b32 s40, -1, 0
	s_and_b32 vcc_lo, exec_lo, s40
	s_cbranch_vccnz .LBB0_2
; %bb.1:
	s_cmp_lt_i32 s3, 0x140
	s_cselect_b32 s2, -1, 0
	s_cmp_lt_i32 s30, 0x60
	s_cselect_b32 s7, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 s2, s7, s2
.LBB0_2:
	v_sub_nc_u32_e32 v83, 0xcbf, v0
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB0_12
; %bb.3:
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_dual_lshrrev_b32 v2, 8, v83 :: v_dual_lshlrev_b32 v3, 2, v0
	s_mov_b32 s7, 0
	s_mov_b32 s8, 0
	v_dual_mov_b32 v4, 0 :: v_dual_add_nc_u32 v5, 2, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mov_b32 v6, v3 :: v_dual_add_nc_u32 v1, -1, v2
	v_and_b32_e32 v5, 30, v5
	s_branch .LBB0_5
.LBB0_4:                                ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
	s_add_co_i32 s8, s8, 2
	v_add_nc_u32_e32 v6, 0x800, v6
	v_cmp_eq_u32_e32 vcc_lo, s8, v5
	s_or_b32 s7, vcc_lo, s7
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s7
	s_cbranch_execz .LBB0_9
.LBB0_5:                                ; =>This Inner Loop Header: Depth=1
	s_mov_b32 s10, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_le_u32_e32 vcc_lo, s8, v1
	v_cmpx_le_u32_e64 s8, v2
; %bb.6:                                ;   in Loop: Header=BB0_5 Depth=1
	ds_store_b32 v6, v4
; %bb.7:                                ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s10
	s_and_saveexec_b32 s2, vcc_lo
	s_cbranch_execz .LBB0_4
; %bb.8:                                ;   in Loop: Header=BB0_5 Depth=1
	ds_store_b32 v6, v4 offset:1024
	s_branch .LBB0_4
.LBB0_9:
	s_or_b32 exec_lo, exec_lo, s7
	v_lshl_add_u32 v1, s4, 2, v3
	v_or_b32_e32 v2, 0xffffff00, v0
	v_mov_b32_e32 v3, 0
	s_mov_b32 s2, 0
.LBB0_10:                               ; =>This Inner Loop Header: Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)
	v_add_nc_u32_e32 v2, 0x100, v2
	ds_store_b32 v1, v3 offset:13056
	v_add_nc_u32_e32 v1, 0x400, v1
	v_cmp_lt_u32_e32 vcc_lo, 0x3ebf, v2
	s_or_b32 s2, vcc_lo, s2
	s_and_not1_b32 exec_lo, exec_lo, s2
	s_cbranch_execnz .LBB0_10
; %bb.11:
	s_or_b32 exec_lo, exec_lo, s2
	s_barrier_signal -1
	s_barrier_wait -1
.LBB0_12:
	s_clause 0x2
	s_load_b64 s[18:19], s[0:1], 0x0 nv
	s_load_b128 s[12:15], s[0:1], 0x20 nv
	s_load_b128 s[20:23], s[0:1], 0x48 nv
	v_lshrrev_b32_e32 v93, 5, v0
	s_wait_xcnt 0x0
	s_lshl_b32 s0, s4, 2
	s_add_co_i32 s6, s6, -1
	s_mov_b64 s[34:35], src_shared_base
	s_or_b32 s41, s0, 0x3300
	s_add_co_i32 s1, s5, -1
	s_min_i32 s0, s33, s6
	s_mov_b32 s2, exec_lo
	v_cmpx_lt_i32_e32 0, v93
	s_xor_b32 s34, exec_lo, s2
	s_cbranch_execz .LBB0_16
; %bb.13:
	s_mov_b32 s37, exec_lo
	v_cmpx_eq_u32_e32 1, v93
	s_cbranch_execz .LBB0_15
; %bb.14:
	s_cmp_gt_i32 s30, 0
	s_mul_i32 s4, s0, 0x140
	s_cselect_b32 s2, -1, 0
	s_ashr_i32 s5, s4, 31
	s_wait_kmcnt 0x0
	s_bfe_i64 s[6:7], s[20:21], 0x200000
	s_and_b32 s2, s25, s2
	s_mul_u64 s[4:5], s[6:7], s[4:5]
	v_cndmask_b32_e64 v2, 0, 1, s2
	s_lshl_b64 s[4:5], s[4:5], 1
	s_mov_b32 s10, 0
	s_add_nc_u64 s[4:5], s[14:15], s[4:5]
	s_delay_alu instid0(SALU_CYCLE_1)
	v_dual_mov_b32 v1, s41 :: v_dual_mov_b32 v4, s4
	s_and_b32 s2, s5, 0x1ffffff
	v_readfirstlane_b32 s44, v2
	s_bitset1_b32 s2, 31
	v_readfirstlane_b32 s45, v1
	v_mov_b32_e32 v3, s2
	s_mov_b32 s2, s30
	v_readfirstlane_b32 s46, v4
	s_lshr_b64 s[6:7], s[2:3], 16
	s_lshr_b32 s2, s3, 16
	v_readfirstlane_b32 s47, v3
	s_movk_i32 s8, 0x140
	s_lshl_b32 s5, s30, 16
	s_or_b32 s7, s2, 0x600000
	s_mov_b32 s4, 0x7510000
	s_mov_b32 s11, s10
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[44:47], s[4:11]
.LBB0_15:
	s_or_b32 exec_lo, exec_lo, s37
.LBB0_16:
	s_or_saveexec_b32 s2, s34
	s_min_i32 s1, s36, s1
	s_xor_b32 exec_lo, exec_lo, s2
	s_cbranch_execz .LBB0_18
; %bb.17:
	s_cmp_gt_i32 s30, 0
	s_mov_b32 s10, 0
	s_cselect_b32 s8, -1, 0
	s_lshl_b32 s4, s1, 6
	s_wait_kmcnt 0x0
	s_bfe_i64 s[6:7], s[12:13], 0x200000
	s_ashr_i32 s5, s4, 31
	s_and_b32 s8, s29, s8
	s_mul_u64 s[4:5], s[6:7], s[4:5]
	v_cndmask_b32_e64 v2, 0, 1, s8
	s_lshl_b64 s[6:7], s[4:5], 1
	s_lshr_b32 s4, s31, 16
	s_add_nc_u64 s[6:7], s[18:19], s[6:7]
	s_lshl_b32 s5, s30, 16
	s_and_b32 s7, s7, 0x1ffffff
	v_readfirstlane_b32 s36, v2
	s_bitset1_b32 s7, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v4, s6 :: v_dual_mov_b32 v3, s7
	s_lshr_b64 s[6:7], s[30:31], 16
	s_or_b32 s7, s4, 0x600000
	s_mov_b32 s8, 64
	v_readfirstlane_b32 s38, v4
	v_readfirstlane_b32 s39, v3
	s_mov_b32 s4, 0x7510000
	s_mov_b32 s11, s10
	s_mov_b32 s37, s10
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[36:39], s[4:11]
.LBB0_18:
	s_or_b32 exec_lo, exec_lo, s2
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	v_dual_lshlrev_b32 v1, 3, v93 :: v_dual_bitop2_b32 v89, 3, v93 bitop3:0x40
	v_mov_b32_e32 v9, 0
	s_and_b32 s34, s29, s25
	s_and_not1_b32 vcc_lo, exec_lo, s17
	v_cndmask_b32_e64 v87, 0, 1, s34
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v8, v9 :: v_dual_bitop2_b32 v91, 32, v1 bitop3:0x40
	v_dual_mov_b32 v7, v9 :: v_dual_mov_b32 v6, v9
	v_dual_mov_b32 v5, v9 :: v_dual_mov_b32 v4, v9
	v_dual_mov_b32 v3, v9 :: v_dual_mov_b32 v2, v9
	v_dual_mov_b32 v17, v9 :: v_dual_mov_b32 v16, v9
	v_dual_mov_b32 v15, v9 :: v_dual_mov_b32 v14, v9
	v_dual_mov_b32 v13, v9 :: v_dual_mov_b32 v12, v9
	v_dual_mov_b32 v11, v9 :: v_dual_mov_b32 v10, v9
	v_dual_mov_b32 v25, v9 :: v_dual_mov_b32 v24, v9
	v_dual_mov_b32 v23, v9 :: v_dual_mov_b32 v22, v9
	v_dual_mov_b32 v21, v9 :: v_dual_mov_b32 v20, v9
	v_dual_mov_b32 v19, v9 :: v_dual_mov_b32 v18, v9
	v_dual_mov_b32 v33, v9 :: v_dual_mov_b32 v32, v9
	v_dual_mov_b32 v31, v9 :: v_dual_mov_b32 v30, v9
	v_dual_mov_b32 v29, v9 :: v_dual_mov_b32 v28, v9
	v_dual_mov_b32 v27, v9 :: v_dual_mov_b32 v26, v9
	v_dual_mov_b32 v41, v9 :: v_dual_mov_b32 v40, v9
	v_dual_mov_b32 v39, v9 :: v_dual_mov_b32 v38, v9
	v_dual_mov_b32 v37, v9 :: v_dual_mov_b32 v36, v9
	v_dual_mov_b32 v35, v9 :: v_dual_mov_b32 v34, v9
	v_dual_mov_b32 v57, v9 :: v_dual_mov_b32 v56, v9
	v_dual_mov_b32 v55, v9 :: v_dual_mov_b32 v54, v9
	v_dual_mov_b32 v53, v9 :: v_dual_mov_b32 v52, v9
	v_dual_mov_b32 v51, v9 :: v_dual_mov_b32 v50, v9
	v_dual_mov_b32 v65, v9 :: v_dual_mov_b32 v64, v9
	v_dual_mov_b32 v63, v9 :: v_dual_mov_b32 v62, v9
	v_dual_mov_b32 v61, v9 :: v_dual_mov_b32 v60, v9
	v_dual_mov_b32 v59, v9 :: v_dual_mov_b32 v58, v9
	v_dual_mov_b32 v73, v9 :: v_dual_mov_b32 v72, v9
	v_dual_mov_b32 v71, v9 :: v_dual_mov_b32 v70, v9
	v_dual_mov_b32 v69, v9 :: v_dual_mov_b32 v68, v9
	v_dual_mov_b32 v67, v9 :: v_dual_mov_b32 v66, v9
	v_dual_mov_b32 v81, v9 :: v_dual_mov_b32 v80, v9
	v_dual_mov_b32 v79, v9 :: v_dual_mov_b32 v78, v9
	v_dual_mov_b32 v77, v9 :: v_dual_mov_b32 v76, v9
	v_dual_mov_b32 v75, v9 :: v_dual_mov_b32 v74, v9
	v_dual_mov_b32 v49, v9 :: v_dual_mov_b32 v48, v9
	v_dual_mov_b32 v47, v9 :: v_dual_mov_b32 v46, v9
	v_dual_mov_b32 v45, v9 :: v_dual_mov_b32 v44, v9
	v_dual_mov_b32 v43, v9 :: v_dual_mov_b32 v42, v9
	s_barrier_wait -1
	s_cbranch_vccnz .LBB0_42
; %bb.19:
	s_mov_b64 s[4:5], src_shared_base
	s_add_co_i32 s6, s41, 0xff00
	s_mov_b32 s7, s5
	v_mul_u32_u24_e32 v3, 0x60, v91
	s_and_b64 s[6:7], s[6:7], 15
	v_and_b32_e32 v5, 16, v0
	s_sub_co_i32 s2, 16, s6
	s_mov_b32 s11, 0
	s_lshr_b32 s2, s2, 2
	s_cmp_lg_u64 s[6:7], 0
	v_or_b32_e32 v2, v3, v5
	s_cselect_b32 s2, s2, 0
	v_and_b32_e32 v7, 15, v0
	s_lshl2_add_u32 s2, s2, s41
	s_mul_hi_i32 s6, s16, 0x2aaaaaab
	s_add_co_i32 s4, s2, 0x13200
	s_add_co_i32 s43, s2, 0xff00
	s_and_b32 s10, s4, 15
	v_mad_u32_u24 v17, 0x60, v7, v2
	s_sub_co_i32 s7, 16, s10
	v_mul_u32_u24_e32 v1, 0x1e00, v89
	s_lshr_b32 s2, s7, 2
	s_cmp_lg_u64 s[10:11], 0
	v_lshrrev_b32_e32 v4, 4, v17
	s_cselect_b32 s2, s2, 0
	s_lshr_b32 s7, s6, 31
	s_ashr_i32 s45, s6, 4
	s_lshl_b32 s10, s2, 2
	s_add_co_i32 s45, s45, s7
	s_cmp_lt_i32 s3, 0x140
	s_add_nc_u64 s[36:37], s[4:5], s[10:11]
	s_mul_i32 s4, s0, 0x140
	s_cselect_b32 s46, -1, 0
	s_lshl_b32 s0, s1, 6
	s_movk_i32 s1, 0x600
	v_and_b32_e32 v4, 0x3f8, v4
	v_mad_u32_u24 v9, 0x60, v7, s1
	v_dual_mov_b32 v85, 0 :: v_dual_lshrrev_b32 v88, 8, v83
	s_mov_b32 s44, s5
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_nc_u32_e32 v82, v4, v17
	v_add_nc_u32_e32 v2, v2, v9
	v_or_b32_e32 v6, v1, v5
	v_mul_u32_u24_e32 v8, 0x60, v7
	s_ashr_i32 s5, s4, 31
	s_wait_kmcnt 0x0
	s_bfe_i64 s[6:7], s[20:21], 0x200000
	v_lshrrev_b32_e32 v4, 4, v2
	v_mad_u32_u24 v28, 0x60, v7, v6
	v_mov_b32_e32 v25, v85
	s_ashr_i32 s1, s0, 31
	s_mul_u64 s[4:5], s[6:7], s[4:5]
	v_and_b32_e32 v13, 0x3f8, v4
	v_lshrrev_b32_e32 v10, 4, v28
	v_add_nc_u32_e32 v2, 0xc00, v28
	v_add_nc_u32_e32 v11, v6, v9
	s_bfe_i64 s[6:7], s[12:13], 0x200000
	s_lshl_b64 s[4:5], s[4:5], 1
	v_and_b32_e32 v10, 0xff8, v10
	s_mul_u64 s[0:1], s[6:7], s[0:1]
	s_add_nc_u64 s[20:21], s[14:15], s[4:5]
	s_lshl_b64 s[4:5], s[0:1], 1
	v_add_nc_u32_e32 v24, 0x600, v17
	v_dual_add_nc_u32 v86, v10, v28 :: v_dual_lshrrev_b32 v10, 4, v2
	v_dual_lshrrev_b32 v4, 4, v11 :: v_dual_bitop2_b32 v12, 32, v5 bitop3:0x54
	v_or_b32_e32 v11, 0x1800, v8
	v_mov_b32_e32 v129, v85
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_and_b32_e32 v19, 0xff8, v10
	v_or_b32_e32 v8, v3, v12
	v_and_b32_e32 v15, 0xff8, v4
	v_add_nc_u32_e32 v4, 0x1200, v28
	v_dual_add_nc_u32 v6, v6, v11 :: v_dual_bitop2_b32 v12, v1, v12 bitop3:0x54
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mad_u32_u24 v10, 0x60, v7, v8
	v_dual_add_nc_u32 v8, v8, v9 :: v_dual_lshrrev_b32 v14, 4, v4
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshrrev_b32_e32 v6, 4, v6
	v_mad_u32_u24 v16, 0x60, v7, v12
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_dual_lshrrev_b32 v10, 4, v10 :: v_dual_lshrrev_b32 v8, 4, v8
	v_or_b32_e32 v5, 64, v5
	v_and_b32_e32 v23, 0xff8, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshrrev_b32_e32 v6, 4, v16
	v_and_b32_e32 v26, 0x3f8, v10
	v_add_nc_u32_e32 v10, v12, v9
	v_or_b32_e32 v3, v3, v5
	v_and_b32_e32 v21, 0xff8, v14
	v_and_b32_e32 v27, 0xff8, v6
	v_and_b32_e32 v84, 0x3f8, v8
	v_lshrrev_b32_e32 v6, 4, v10
	v_mad_u32_u24 v14, 0x60, v7, v3
	v_or_b32_e32 v1, v1, v5
	v_add_nc_u32_e32 v3, v3, v9
	v_add_nc_u32_e32 v8, 0xc00, v16
	v_add_nc_u32_e32 v10, 0x1200, v16
	v_lshrrev_b32_e32 v5, 4, v14
	v_mad_u32_u24 v7, 0x60, v7, v1
	v_dual_add_nc_u32 v94, v19, v28 :: v_dual_add_nc_u32 v100, v26, v17
	v_mov_b32_e32 v19, v85
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v29, 0x3f8, v5
	v_add_nc_u32_e32 v5, v1, v9
	v_add_nc_u32_e32 v16, 0xc00, v7
	v_lshrrev_b32_e32 v3, 4, v3
	v_lshrrev_b32_e32 v9, 4, v7
	v_dual_add_nc_u32 v12, v12, v11 :: v_dual_add_nc_u32 v1, v1, v11
	v_add_nc_u32_e32 v83, -1, v88
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_and_b32_e32 v14, 0x3f8, v3
	v_lshrrev_b32_e32 v3, 4, v5
	v_and_b32_e32 v30, 0xff8, v9
	v_dual_mov_b32 v9, v85 :: v_dual_lshrrev_b32 v8, 4, v8
	v_dual_mov_b32 v11, v85 :: v_dual_lshrrev_b32 v5, 4, v16
	v_and_b32_e32 v16, 0xff8, v3
	v_sub_nc_u32_e32 v3, 0x3fbf, v0
	v_add_nc_u32_e32 v90, v13, v17
	v_and_b32_e32 v8, 0x1ff8, v8
	v_add_nc_u32_e32 v7, 0x1200, v7
	v_and_b32_e32 v18, 0x1ff8, v5
	v_dual_lshrrev_b32 v3, 8, v3 :: v_dual_add_nc_u32 v92, v15, v28
	v_add_nc_u32_e32 v96, v21, v28
	v_dual_mov_b32 v21, v85 :: v_dual_lshrrev_b32 v12, 4, v12
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_add_nc_u32_e32 v3, 1, v3
	v_and_b32_e32 v6, 0xff8, v6
	v_dual_mov_b32 v13, v85 :: v_dual_add_nc_u32 v98, v23, v28
	v_lshrrev_b32_e32 v1, 4, v1
	v_and_b32_e32 v95, 0x7e, v3
	v_and_b32_e32 v12, 0xff8, v12
	v_dual_mov_b32 v27, v85 :: v_dual_add_nc_u32 v104, v27, v28
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_and_b32_e32 v22, 0xff8, v1
	v_cmp_ne_u32_e64 s0, v3, v95
	v_dual_mov_b32 v3, v85 :: v_dual_lshrrev_b32 v10, 4, v10
	v_lshl_or_b32 v31, v95, 8, v0
	v_mov_b32_e32 v5, v85
	v_add_nc_u64_e32 v[102:103], v[84:85], v[24:25]
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u64_e32 v[108:109], v[8:9], v[2:3]
	v_add_nc_u64_e32 v[122:123], v[18:19], v[2:3]
	v_dual_mov_b32 v2, v85 :: v_dual_lshrrev_b32 v7, 4, v7
	v_and_b32_e32 v10, 0x1ff8, v10
	v_dual_mov_b32 v15, v85 :: v_dual_add_nc_u32 v84, 0x600, v28
	v_add_nc_u32_e32 v26, 0x1800, v28
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_and_b32_e32 v20, 0x1ff8, v7
	v_mov_b32_e32 v7, v85
	v_dual_add_nc_u32 v114, v29, v17 :: v_dual_add_nc_u32 v118, v30, v28
	v_dual_mov_b32 v17, v85 :: v_dual_mov_b32 v23, v85
	v_dual_lshlrev_b32 v128, 2, v31 :: v_dual_add_nc_u32 v1, 2, v88
	v_add_nc_u64_e32 v[106:107], v[6:7], v[84:85]
	v_add_nc_u64_e32 v[110:111], v[10:11], v[4:5]
	v_add_nc_u64_e32 v[112:113], v[12:13], v[26:27]
	v_add_nc_u64_e32 v[116:117], v[14:15], v[24:25]
	v_add_nc_u64_e32 v[120:121], v[16:17], v[84:85]
	v_add_nc_u64_e32 v[124:125], v[20:21], v[4:5]
	v_add_nc_u64_e32 v[126:127], v[22:23], v[26:27]
	v_and_b32_e32 v97, 30, v1
	v_or_b32_e32 v1, 0x100, v0
	v_dual_mov_b32 v6, v85 :: v_dual_add_nc_u32 v99, 0xffffff00, v31
	v_dual_mov_b32 v4, v85 :: v_dual_mov_b32 v8, v85
	v_dual_mov_b32 v10, v85 :: v_dual_mov_b32 v12, v85
	v_dual_mov_b32 v14, v85 :: v_dual_mov_b32 v16, v85
	v_dual_mov_b32 v18, v85 :: v_dual_mov_b32 v20, v85
	v_dual_mov_b32 v22, v85 :: v_dual_mov_b32 v24, v85
	v_dual_mov_b32 v26, v85 :: v_dual_mov_b32 v28, v85
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
	s_lshr_b32 s47, s3, 16
	s_lshr_b32 s48, s31, 16
	s_mov_b32 s42, s35
	s_movk_i32 s16, 0x140
	s_or_b32 s47, s47, 0x600000
	s_or_b32 s48, s48, 0x600000
	s_mov_b32 s8, 64
	s_add_nc_u64 s[38:39], s[18:19], s[4:5]
	s_mov_b32 s4, 0x7510000
	s_mov_b32 s49, s11
	s_branch .LBB0_21
.LBB0_20:                               ;   in Loop: Header=BB0_21 Depth=1
	s_cmp_eq_u32 s49, s45
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_scc1 .LBB0_42
.LBB0_21:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_24 Depth 2
                                        ;     Child Loop BB0_29 Depth 2
                                        ;     Child Loop BB0_32 Depth 2
	s_and_b32 s50, s49, 1
	s_add_co_i32 s49, s49, 1
	s_xor_b32 s5, s50, 1
	s_mul_i32 s1, s49, 0xffffffa0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s1, s1, s26
	s_min_i32 s1, s1, 0x60
	s_cmp_lt_i32 s49, s45
	s_cselect_b32 s2, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_and_b32 s6, s2, exec_lo
	s_cselect_b32 s30, s1, 0
	s_cmp_lt_i32 s30, 0x60
	s_cselect_b32 s1, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b32 s1, s46, s1
	s_or_b32 s1, s40, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_cbranch_vccnz .LBB0_34
; %bb.22:                               ;   in Loop: Header=BB0_21 Depth=1
	v_mov_b64_e32 v[130:131], v[0:1]
	s_cmp_lg_u32 s5, 0
	s_mov_b32 s10, 0
	s_cselect_b32 s7, s44, s35
	s_cselect_b32 s6, s43, 0
	s_mov_b32 s12, 0
	s_branch .LBB0_24
.LBB0_23:                               ;   in Loop: Header=BB0_24 Depth=2
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s1
	s_add_co_i32 s12, s12, 2
	v_add_nc_u32_e32 v131, 0x200, v131
	v_cmp_eq_u32_e32 vcc_lo, s12, v97
	v_add_nc_u32_e32 v130, 0x200, v130
	s_or_b32 s10, vcc_lo, s10
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s10
	s_cbranch_execz .LBB0_28
.LBB0_24:                               ;   Parent Loop BB0_21 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_mov_b32 s13, exec_lo
	v_cmp_le_u32_e32 vcc_lo, s12, v83
	v_cmpx_le_u32_e64 s12, v88
	s_cbranch_execz .LBB0_26
; %bb.25:                               ;   in Loop: Header=BB0_24 Depth=2
	v_mov_b32_e32 v84, v130
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u64 v[132:133], v[84:85], 2, s[6:7]
	flat_store_b32 v[132:133], v85
.LBB0_26:                               ;   in Loop: Header=BB0_24 Depth=2
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s13
	s_and_saveexec_b32 s1, vcc_lo
	s_cbranch_execz .LBB0_23
; %bb.27:                               ;   in Loop: Header=BB0_24 Depth=2
	v_mov_b32_e32 v84, v131
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u64 v[132:133], v[84:85], 2, s[6:7]
	flat_store_b32 v[132:133], v85
	s_branch .LBB0_23
.LBB0_28:                               ;   in Loop: Header=BB0_21 Depth=1
	s_or_b32 exec_lo, exec_lo, s10
	v_mov_b64_e32 v[130:131], v[0:1]
	v_mov_b32_e32 v101, v95
	s_cmp_lg_u32 s5, 0
	s_mov_b32 s1, 0
	s_cselect_b32 s7, s37, s42
	s_cselect_b32 s6, s36, s41
.LBB0_29:                               ;   Parent Loop BB0_21 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_dual_mov_b32 v84, v130 :: v_dual_add_nc_u32 v101, -2, v101
	v_dual_mov_b32 v132, v131 :: v_dual_mov_b32 v133, v85
	v_add_nc_u32_e32 v131, 0x200, v131
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u64 v[134:135], v[84:85], 2, s[6:7]
	v_cmp_eq_u32_e32 vcc_lo, 0, v101
	v_add_nc_u32_e32 v130, 0x200, v130
	v_lshl_add_u64 v[132:133], v[132:133], 2, s[6:7]
	s_clause 0x1
	flat_store_b32 v[134:135], v85
	flat_store_b32 v[132:133], v85
	s_or_b32 s1, vcc_lo, s1
	s_wait_xcnt 0x0
	s_and_not1_b32 exec_lo, exec_lo, s1
	s_cbranch_execnz .LBB0_29
; %bb.30:                               ;   in Loop: Header=BB0_21 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_33
; %bb.31:                               ;   in Loop: Header=BB0_21 Depth=1
	v_add_nc_u64_e32 v[130:131], s[6:7], v[128:129]
	v_mov_b32_e32 v84, v99
	s_mov_b32 s6, 0
.LBB0_32:                               ;   Parent Loop BB0_21 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v84, 0x100, v84
	flat_store_b32 v[130:131], v85
	s_wait_xcnt 0x0
	v_add_nc_u64_e32 v[130:131], 0x400, v[130:131]
	v_cmp_lt_u32_e32 vcc_lo, 0x3ebf, v84
	s_or_b32 s6, vcc_lo, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s6
	s_cbranch_execnz .LBB0_32
.LBB0_33:                               ;   in Loop: Header=BB0_21 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	s_barrier_signal -1
	s_barrier_wait -1
.LBB0_34:                               ;   in Loop: Header=BB0_21 Depth=1
	s_and_b32 s1, s2, exec_lo
	s_cselect_b32 s1, s49, 0
	s_mov_b32 s2, exec_lo
	v_cmpx_lt_i32_e32 0, v93
	s_xor_b32 s6, exec_lo, s2
	s_cbranch_execnz .LBB0_37
; %bb.35:                               ;   in Loop: Header=BB0_21 Depth=1
	s_and_not1_saveexec_b32 s2, s6
	s_cbranch_execnz .LBB0_40
.LBB0_36:                               ;   in Loop: Header=BB0_21 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s34
	s_cbranch_vccnz .LBB0_20
	s_branch .LBB0_41
.LBB0_37:                               ;   in Loop: Header=BB0_21 Depth=1
	s_mov_b32 s7, exec_lo
	v_cmpx_eq_u32_e32 1, v93
	s_cbranch_execz .LBB0_39
; %bb.38:                               ;   in Loop: Header=BB0_21 Depth=1
	s_cmp_lg_u32 s5, 0
	s_mul_i32 s10, s1, 0x60
	s_cselect_b32 s2, s36, s41
	s_cmp_gt_i32 s30, 0
	s_mov_b32 s17, s9
	s_cselect_b32 s14, -1, 0
	s_lshl_b64 s[12:13], s[10:11], 1
	s_mov_b32 s18, s11
	s_add_nc_u64 s[12:13], s[20:21], s[12:13]
	s_delay_alu instid0(SALU_CYCLE_1)
	v_dual_mov_b32 v101, s2 :: v_dual_mov_b32 v130, s12
	s_and_b32 s10, s13, 0x1ffffff
	s_and_b32 s13, s25, s14
	s_bitset1_b32 s10, 31
	v_cndmask_b32_e64 v84, 0, 1, s13
	v_mov_b32_e32 v103, s10
	s_mov_b32 s2, s30
	v_readfirstlane_b32 s53, v101
	v_readfirstlane_b32 s54, v130
	v_readfirstlane_b32 s52, v84
	v_readfirstlane_b32 s55, v103
	s_lshr_b64 s[14:15], s[2:3], 16
	s_lshl_b32 s13, s30, 16
	s_mov_b32 s12, s4
	s_mov_b32 s15, s47
	s_mov_b32 s19, s11
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[52:55], s[12:19]
.LBB0_39:                               ;   in Loop: Header=BB0_21 Depth=1
	s_or_b32 exec_lo, exec_lo, s7
	s_and_not1_saveexec_b32 s2, s6
	s_cbranch_execz .LBB0_36
.LBB0_40:                               ;   in Loop: Header=BB0_21 Depth=1
	s_cmp_lg_u32 s5, 0
	s_mul_i32 s10, s1, 0x60
	s_cselect_b32 s5, s43, 0
	s_cmp_gt_i32 s30, 0
	s_cselect_b32 s1, -1, 0
	s_lshl_b64 s[6:7], s[10:11], 1
	s_and_b32 s1, s29, s1
	s_add_nc_u64 s[6:7], s[38:39], s[6:7]
	v_cndmask_b32_e64 v84, 0, 1, s1
	s_and_b32 s7, s7, 0x1ffffff
	v_dual_mov_b32 v101, s5 :: v_dual_mov_b32 v130, s6
	s_bitset1_b32 s7, 31
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_readfirstlane_b32 s12, v84
	v_mov_b32_e32 v103, s7
	v_readfirstlane_b32 s13, v101
	v_readfirstlane_b32 s14, v130
	s_lshr_b64 s[6:7], s[30:31], 16
	s_lshl_b32 s5, s30, 16
	v_readfirstlane_b32 s15, v103
	s_mov_b32 s7, s48
	s_mov_b32 s10, s11
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	tensor_load_to_lds s[12:15], s[4:11]
	s_or_b32 exec_lo, exec_lo, s2
	s_and_not1_b32 vcc_lo, exec_lo, s34
	s_cbranch_vccnz .LBB0_20
.LBB0_41:                               ;   in Loop: Header=BB0_21 Depth=1
	s_cmp_lg_u32 s50, 0
	s_cselect_b32 s2, s43, 0
	s_cselect_b32 s1, s36, s41
	v_lshl_add_u32 v84, v82, 1, s2
	v_lshl_add_u32 v101, v86, 1, s1
	v_lshl_add_u32 v103, v104, 1, s1
	v_lshl_add_u32 v105, v106, 1, s1
	v_lshl_add_u32 v107, v108, 1, s1
	ds_load_b128 v[130:133], v84
	ds_load_b128 v[134:137], v84 offset:16
	v_lshl_add_u32 v84, v90, 1, s2
	ds_load_b128 v[138:141], v101
	ds_load_b128 v[142:145], v101 offset:16
	v_lshl_add_u32 v101, v96, 1, s1
	v_lshl_add_u32 v109, v110, 1, s1
	ds_load_b128 v[146:149], v84 offset:3072
	ds_load_b128 v[150:153], v84 offset:3088
	v_lshl_add_u32 v84, v92, 1, s1
	v_lshl_add_u32 v111, v112, 1, s1
	ds_load_b128 v[178:181], v103 offset:64
	ds_load_b128 v[182:185], v103 offset:80
	ds_load_b128 v[186:189], v107 offset:64
	ds_load_b128 v[154:157], v84 offset:3072
	ds_load_b128 v[190:193], v107 offset:80
	ds_load_b128 v[194:197], v109 offset:64
	ds_load_b128 v[198:201], v109 offset:80
	ds_load_b128 v[202:205], v111 offset:64
	ds_load_b128 v[206:209], v111 offset:80
	s_wait_dscnt 0xb
	v_wmma_f32_16x16x32_bf16 v[74:81], v[130:137], v[138:145], v[74:81]
	; sched_group_barrier mask(0x00000100) size(7) SyncID(0)
	s_wait_dscnt 0x9
	v_wmma_f32_16x16x32_bf16 v[26:33], v[146:153], v[138:145], v[26:33] matrix_a_reuse
	ds_load_b128 v[158:161], v84 offset:3088
	v_lshl_add_u32 v84, v98, 1, s1
	ds_load_b128 v[170:173], v84 offset:12288
	ds_load_b128 v[174:177], v84 offset:12304
	v_lshl_add_u32 v84, v100, 1, s2
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[34:41], v[130:137], v[170:177], v[34:41] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[42:49], v[146:153], v[170:177], v[42:49]
	ds_load_b128 v[138:141], v101 offset:9216
	ds_load_b128 v[142:145], v101 offset:9232
	v_lshl_add_u32 v101, v94, 1, s1
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[50:57], v[130:137], v[138:145], v[50:57] matrix_a_reuse
	ds_load_b128 v[162:165], v101 offset:6144
	ds_load_b128 v[166:169], v101 offset:6160
	v_lshl_add_u32 v101, v102, 1, s2
	; sched_group_barrier mask(0x00000008) size(5) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(7) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[2:9], v[146:153], v[138:145], v[2:9] matrix_a_reuse
	ds_load_b128 v[138:141], v105 offset:64
	ds_load_b128 v[142:145], v105 offset:80
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[10:17], v[146:153], v[162:169], v[10:17] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[18:25], v[146:153], v[154:161], v[18:25] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[66:73], v[130:137], v[154:161], v[66:73] matrix_a_reuse
	ds_load_b128 v[154:157], v84 offset:64
	ds_load_b128 v[158:161], v84 offset:80
	v_wmma_f32_16x16x32_bf16 v[58:65], v[130:137], v[162:169], v[58:65] matrix_a_reuse
	ds_load_b128 v[162:165], v101 offset:64
	ds_load_b128 v[166:169], v101 offset:80
	; sched_group_barrier mask(0x00000008) size(5) SyncID(0)
	; sched_barrier mask(0x00000000)
	v_lshl_add_u32 v84, v114, 1, s2
	v_lshl_add_u32 v101, v116, 1, s2
	v_lshl_add_u32 v103, v118, 1, s1
	v_lshl_add_u32 v105, v120, 1, s1
	ds_load_b128 v[130:133], v84 offset:128
	ds_load_b128 v[134:137], v84 offset:144
	ds_load_b128 v[146:149], v101 offset:128
	ds_load_b128 v[150:153], v101 offset:144
	ds_load_b128 v[170:173], v103 offset:128
	ds_load_b128 v[174:177], v103 offset:144
	v_lshl_add_u32 v84, v122, 1, s1
	v_lshl_add_u32 v101, v124, 1, s1
	v_lshl_add_u32 v103, v126, 1, s1
	ds_load_b128 v[210:213], v105 offset:128
	s_wait_dscnt 0x9
	v_wmma_f32_16x16x32_bf16 v[74:81], v[154:161], v[178:185], v[74:81]
	; sched_group_barrier mask(0x00000100) size(7) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[66:73], v[154:161], v[138:145], v[66:73] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[58:65], v[154:161], v[186:193], v[58:65] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[50:57], v[154:161], v[194:201], v[50:57] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[34:41], v[154:161], v[202:209], v[34:41] matrix_a_reuse
	ds_load_b128 v[214:217], v105 offset:144
	ds_load_b128 v[154:157], v84 offset:128
	ds_load_b128 v[158:161], v84 offset:144
	ds_load_b128 v[218:221], v101 offset:128
	ds_load_b128 v[222:225], v101 offset:144
	ds_load_b128 v[226:229], v103 offset:128
	ds_load_b128 v[230:233], v103 offset:144
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[42:49], v[162:169], v[202:209], v[42:49]
	; sched_group_barrier mask(0x00000008) size(5) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(7) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[2:9], v[162:169], v[194:201], v[2:9] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[10:17], v[162:169], v[186:193], v[10:17] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[18:25], v[162:169], v[138:145], v[18:25] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[26:33], v[162:169], v[178:185], v[26:33] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(5) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0x8
	v_wmma_f32_16x16x32_bf16 v[74:81], v[130:137], v[170:177], v[74:81]
	; sched_group_barrier mask(0x00000100) size(7) SyncID(0)
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[66:73], v[130:137], v[210:217], v[66:73] matrix_a_reuse
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[58:65], v[130:137], v[154:161], v[58:65] matrix_a_reuse
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[50:57], v[130:137], v[218:225], v[50:57] matrix_a_reuse
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[34:41], v[130:137], v[226:233], v[34:41] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(5) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(7) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[42:49], v[146:153], v[226:233], v[42:49]
	v_wmma_f32_16x16x32_bf16 v[2:9], v[146:153], v[218:225], v[2:9] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[10:17], v[146:153], v[154:161], v[10:17] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[18:25], v[146:153], v[210:217], v[18:25] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[26:33], v[146:153], v[170:177], v[26:33] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(5) SyncID(0)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_wait_dscnt 0
	;;#ASMEND
	s_branch .LBB0_20
.LBB0_42:
	s_wait_tensorcnt 0x0
	;;#ASMSTART
	s_wait_dscnt 0
	;;#ASMEND
	s_and_b32 vcc_lo, exec_lo, s34
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_vccz .LBB0_44
; %bb.43:
	v_mul_u32_u24_e32 v1, 0x50, v89
	v_lshrrev_b32_e32 v82, 1, v0
	v_cvt_pk_bf16_f32 v81, v80, v81
	v_cvt_pk_bf16_f32 v80, v78, v79
	v_cvt_pk_bf16_f32 v78, v74, v75
	v_and_or_b32 v1, v0, 15, v1
	v_and_or_b32 v82, v82, 8, v91
	v_cvt_pk_bf16_f32 v65, v64, v65
	v_cvt_pk_bf16_f32 v64, v62, v63
	v_cvt_pk_bf16_f32 v63, v60, v61
	v_cvt_pk_bf16_f32 v62, v58, v59
	v_lshl_or_b32 v74, v1, 6, v82
	v_lshlrev_b32_e32 v1, 3, v1
	v_cvt_pk_bf16_f32 v73, v72, v73
	v_cvt_pk_bf16_f32 v72, v70, v71
	v_cvt_pk_bf16_f32 v70, v66, v67
	v_add_nc_u32_e32 v58, 0x800, v74
	v_add_nc_u32_e32 v60, 0xc00, v74
	v_add_nc_u32_e32 v66, 0x400, v74
	v_and_b32_e32 v1, 0x7f0, v1
	v_lshlrev_b32_e32 v61, 1, v74
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_lshrrev_b32 v58, 3, v58 :: v_dual_lshrrev_b32 v60, 3, v60
	v_lshrrev_b32_e32 v59, 3, v66
	v_cvt_pk_bf16_f32 v79, v76, v77
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_nc_u32_e32 v1, v1, v61
	v_and_b32_e32 v58, 0xffffff0, v58
	v_and_b32_e32 v60, 0xffffff0, v60
	v_and_b32_e32 v59, 0xffffff0, v59
	v_cvt_pk_bf16_f32 v33, v32, v33
	v_cvt_pk_bf16_f32 v32, v30, v31
	v_cvt_pk_bf16_f32 v30, v26, v27
	v_add_nc_u32_e32 v27, 0xc10, v74
	ds_store_b128 v1, v[78:81]
	v_dual_add_nc_u32 v1, v58, v61 :: v_dual_add_nc_u32 v58, v60, v61
	v_add_nc_u32_e32 v60, 16, v74
	v_cvt_pk_bf16_f32 v71, v68, v69
	v_add_nc_u32_e32 v66, 0x1000, v74
	v_add_nc_u32_e32 v59, v59, v61
	v_cvt_pk_bf16_f32 v57, v56, v57
	v_cvt_pk_bf16_f32 v56, v54, v55
	v_cvt_pk_bf16_f32 v55, v52, v53
	v_add_nc_u32_e32 v52, 0x410, v74
	v_cvt_pk_bf16_f32 v41, v40, v41
	v_cvt_pk_bf16_f32 v40, v38, v39
	v_cvt_pk_bf16_f32 v38, v34, v35
	v_add_nc_u32_e32 v35, 0x810, v74
	v_cvt_pk_bf16_f32 v25, v24, v25
	v_cvt_pk_bf16_f32 v24, v22, v23
	v_cvt_pk_bf16_f32 v23, v20, v21
	v_lshrrev_b32_e32 v20, 3, v27
	v_add_nc_u32_e32 v21, 0x1010, v74
	v_cvt_pk_bf16_f32 v54, v50, v51
	v_dual_lshrrev_b32 v51, 3, v60 :: v_dual_lshrrev_b32 v66, 3, v66
	ds_store_b128 v59, v[70:73] offset:2048
	ds_store_b128 v1, v[62:65] offset:4096
	v_dual_lshrrev_b32 v1, 3, v52 :: v_dual_lshrrev_b32 v26, 3, v35
	v_cvt_pk_bf16_f32 v22, v18, v19
	v_and_b32_e32 v19, 0xffffff0, v20
	v_lshrrev_b32_e32 v20, 3, v21
	v_and_b32_e32 v34, 0x1ff0, v51
	v_and_b32_e32 v59, 0xffffff0, v66
	v_cvt_pk_bf16_f32 v39, v36, v37
	v_and_b32_e32 v1, 0xffffff0, v1
	v_lshlrev_b32_e32 v36, 1, v60
	v_and_b32_e32 v26, 0xffffff0, v26
	v_cvt_pk_bf16_f32 v17, v16, v17
	v_cvt_pk_bf16_f32 v16, v14, v15
	v_cvt_pk_bf16_f32 v14, v10, v11
	v_and_b32_e32 v11, 0xffffff0, v20
	v_add_nc_u32_e32 v34, v34, v61
	v_cvt_pk_bf16_f32 v31, v28, v29
	v_dual_add_nc_u32 v50, v59, v61 :: v_dual_add_nc_u32 v1, v1, v36
	v_add_nc_u32_e32 v18, v26, v36
	v_cvt_pk_bf16_f32 v15, v12, v13
	v_add_nc_u32_e32 v10, v19, v36
	v_cvt_pk_bf16_f32 v9, v8, v9
	v_cvt_pk_bf16_f32 v8, v6, v7
	v_cvt_pk_bf16_f32 v7, v4, v5
	v_cvt_pk_bf16_f32 v6, v2, v3
	v_add_nc_u32_e32 v11, v11, v36
	v_cvt_pk_bf16_f32 v5, v48, v49
	v_cvt_pk_bf16_f32 v4, v46, v47
	v_cvt_pk_bf16_f32 v3, v44, v45
	v_cvt_pk_bf16_f32 v2, v42, v43
	ds_store_b128 v58, v[54:57] offset:6144
	ds_store_b128 v50, v[38:41] offset:8192
	ds_store_b128 v34, v[30:33] offset:32
	ds_store_b128 v1, v[22:25] offset:2048
	ds_store_b128 v18, v[14:17] offset:4096
	ds_store_b128 v10, v[6:9] offset:6144
	ds_store_b128 v11, v[2:5] offset:8192
.LBB0_44:
	v_cmp_ne_u32_e32 vcc_lo, 1, v87
	;;#ASMSTART
	s_wait_dscnt 0
	;;#ASMEND
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_vccnz .LBB0_55
; %bb.45:
	s_mul_i32 s3, s3, s31
	s_mov_b32 s0, exec_lo
	v_cmpx_gt_i32_e64 s3, v0
	s_cbranch_execz .LBB0_55
; %bb.46:
	s_ashr_i32 s29, s28, 31
	v_xad_u32 v2, v0, -1, s3
	s_lshl_b64 s[0:1], s[28:29], 1
	s_wait_kmcnt 0x0
	s_mul_i32 s14, s33, 0x140
	s_ashr_i32 s25, s24, 31
	s_add_nc_u64 s[4:5], s[22:23], s[0:1]
	s_mov_b32 s0, 0
                                        ; implicit-def: $vgpr1
                                        ; implicit-def: $vgpr6
                                        ; implicit-def: $sgpr12_sgpr13
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_u32_e32 0x2ff, v2
	s_xor_b32 s15, exec_lo, s1
	s_cbranch_execnz .LBB0_49
; %bb.47:
	s_or_saveexec_b32 s1, s15
	v_mov_b64_e32 v[2:3], s[12:13]
	s_xor_b32 exec_lo, exec_lo, s1
	s_cbranch_execnz .LBB0_52
.LBB0_48:
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 exec_lo, exec_lo, s0
	s_cbranch_execnz .LBB0_53
	s_branch .LBB0_55
.LBB0_49:
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
	s_mov_b32 s17, s14
	s_mov_b32 s18, s14
	v_and_b32_e32 v9, 0x1fffffc, v8
	s_mov_b32 s19, s14
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
.LBB0_50:                               ; =>This Inner Loop Header: Depth=1
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
	v_dual_sub_nc_u32 v27, v19, v18 :: v_dual_add_nc_u32 v18, s14, v1
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
	v_mul_u64_e32 v[18:19], s[24:25], v[18:19]
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
	s_cbranch_execnz .LBB0_50
; %bb.51:
	s_or_b32 exec_lo, exec_lo, s26
	v_cmp_ne_u32_e32 vcc_lo, v8, v9
	v_lshl_or_b32 v0, v9, 8, v0
	v_dual_mov_b32 v6, s16 :: v_dual_mov_b32 v1, s23
	s_and_b32 s0, vcc_lo, exec_lo
	s_or_saveexec_b32 s1, s15
	v_mov_b64_e32 v[2:3], s[12:13]
	s_xor_b32 exec_lo, exec_lo, s1
	s_cbranch_execz .LBB0_48
.LBB0_52:
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
	s_cbranch_execz .LBB0_55
.LBB0_53:
	v_mov_b32_e32 v5, 0
	s_mov_b32 s0, 0
	s_sub_co_i32 s1, 0, s27
.LBB0_54:                               ; =>This Inner Loop Header: Depth=1
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
	v_mul_lo_u32 v8, v7, s27
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_sub_nc_u32 v4, v4, v8 :: v_dual_add_nc_u32 v8, s14, v7
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
	s_cbranch_execnz .LBB0_54
.LBB0_55:
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end0:
	.size	bm064_bn320_bk096_wm2_wn4_mc0, .Lfunc_end0-bm064_bn320_bk096_wm2_wn4_mc0
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel bm064_bn320_bk096_wm2_wn4_mc0
		.amdhsa_group_segment_fixed_size 156672
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
		.amdhsa_next_free_vgpr 234
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
	.section	.text.bm064_bn320_bk096_wm2_wn4_mc0,"axG",@progbits,bm064_bn320_bk096_wm2_wn4_mc0,comdat
                                        ; -- End function
	.set .Lbm064_bn320_bk096_wm2_wn4_mc0.num_vgpr, 234
	.set .Lbm064_bn320_bk096_wm2_wn4_mc0.num_agpr, 0
	.set .Lbm064_bn320_bk096_wm2_wn4_mc0.numbered_sgpr, 56
	.set .Lbm064_bn320_bk096_wm2_wn4_mc0.num_named_barrier, 0
	.set .Lbm064_bn320_bk096_wm2_wn4_mc0.private_seg_size, 0
	.set .Lbm064_bn320_bk096_wm2_wn4_mc0.uses_vcc, 1
	.set .Lbm064_bn320_bk096_wm2_wn4_mc0.uses_flat_scratch, 1
	.set .Lbm064_bn320_bk096_wm2_wn4_mc0.has_dyn_sized_stack, 0
	.set .Lbm064_bn320_bk096_wm2_wn4_mc0.has_recursion, 0
	.set .Lbm064_bn320_bk096_wm2_wn4_mc0.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 6892
; TotalNumSgprs: 58
; NumVgprs: 234
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 156672 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 14
; NumSGPRsForWavesPerEU: 58
; NumVGPRsForWavesPerEU: 234
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
	.type	__hip_cuid_435ed6335b5c41bc,@object ; @__hip_cuid_435ed6335b5c41bc
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_435ed6335b5c41bc
__hip_cuid_435ed6335b5c41bc:
	.byte	0                               ; 0x0
	.size	__hip_cuid_435ed6335b5c41bc, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git aa451e1fe6a793394d6733051b1778633063ae96+PATCHED:d17c5aa0e3ea29cde402f58f27e39b6034effa27)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_435ed6335b5c41bc
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
    macrotile: [64, 320, 96]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, One]
  MatrixInstruction: [16, 16, 32, 1]
  EnableMatrixInstruction: True
  MIWaveTile: [2, 5]
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
    .group_segment_fixed_size: 156672
    .kernarg_segment_align: 8
    .kernarg_segment_size: 132
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           bm064_bn320_bk096_wm2_wn4_mc0
    .private_segment_fixed_size: 0
    .sgpr_count:     58
    .sgpr_spill_count: 0
    .symbol:         bm064_bn320_bk096_wm2_wn4_mc0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     234
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
