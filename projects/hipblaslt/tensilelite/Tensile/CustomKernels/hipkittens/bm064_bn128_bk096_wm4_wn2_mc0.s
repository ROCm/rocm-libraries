	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.section	.text.bm064_bn128_bk096_wm4_wn2_mc0,"axG",@progbits,bm064_bn128_bk096_wm4_wn2_mc0,comdat
	.protected	bm064_bn128_bk096_wm4_wn2_mc0 ; -- Begin function bm064_bn128_bk096_wm4_wn2_mc0
	.globl	bm064_bn128_bk096_wm4_wn2_mc0
	.p2align	8
	.type	bm064_bn128_bk096_wm4_wn2_mc0,@function
bm064_bn128_bk096_wm4_wn2_mc0: ; @bm064_bn128_bk096_wm4_wn2_mc0
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
	s_cselect_b32 s5, ttmp7, s5
	s_add_co_i32 s2, s24, 63
	s_add_co_i32 s7, s25, 0x7f
	s_ashr_i32 s3, s2, 31
	s_lshl_b32 s28, s36, 6
	s_lshr_b32 s3, s3, 26
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s2, s2, s3
	s_ashr_i32 s3, s7, 31
	s_ashr_i32 s6, s2, 6
	s_lshr_b32 s2, s3, 25
	s_add_co_i32 s7, s7, s2
	s_sub_co_i32 s2, s24, s28
	s_ashr_i32 s7, s7, 7
	s_min_i32 s27, s2, 64
	s_cmp_lt_i32 s36, s6
	s_cselect_b32 s29, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_and_b32 s2, s29, exec_lo
	s_cselect_b32 s31, s27, 0
	s_lshl_b32 s33, s5, 7
	s_sub_co_i32 s2, s25, s33
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_min_i32 s2, s2, 0x80
	s_cmp_lt_i32 s5, s7
	s_cselect_b32 s25, -1, 0
	s_and_b32 s3, s25, exec_lo
	s_cselect_b32 s3, s2, 0
	s_add_co_i32 s16, s26, 0x5f
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 2, 1), 1
	s_min_i32 s2, s26, 0x60
	s_cmp_gt_i32 s16, 0x5f
	s_cselect_b32 s17, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)
	s_and_b32 s8, s17, exec_lo
	s_cselect_b32 s30, s2, 0
	s_cmp_lt_i32 s31, 64
	s_mov_b32 s2, -1
	s_cselect_b32 s40, -1, 0
	s_and_b32 vcc_lo, exec_lo, s40
	s_cbranch_vccnz .LBB0_2
; %bb.1:
	s_cmp_lt_i32 s3, 0x80
	s_cselect_b32 s2, -1, 0
	s_cmp_lt_i32 s30, 0x60
	s_cselect_b32 s8, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 s2, s8, s2
.LBB0_2:
	v_sub_nc_u32_e32 v35, 0xcbf, v0
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB0_12
; %bb.3:
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_dual_lshrrev_b32 v2, 8, v35 :: v_dual_lshlrev_b32 v3, 2, v0
	s_mov_b32 s8, 0
	s_mov_b32 s10, 0
	v_dual_mov_b32 v4, 0 :: v_dual_add_nc_u32 v5, 2, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mov_b32 v6, v3 :: v_dual_add_nc_u32 v1, -1, v2
	v_and_b32_e32 v5, 30, v5
	s_branch .LBB0_5
.LBB0_4:                                ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
	s_add_co_i32 s10, s10, 2
	v_add_nc_u32_e32 v6, 0x800, v6
	v_cmp_eq_u32_e32 vcc_lo, s10, v5
	s_or_b32 s8, vcc_lo, s8
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s8
	s_cbranch_execz .LBB0_9
.LBB0_5:                                ; =>This Inner Loop Header: Depth=1
	s_mov_b32 s11, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_le_u32_e32 vcc_lo, s10, v1
	v_cmpx_le_u32_e64 s10, v2
; %bb.6:                                ;   in Loop: Header=BB0_5 Depth=1
	ds_store_b32 v6, v4
; %bb.7:                                ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s11
	s_and_saveexec_b32 s2, vcc_lo
	s_cbranch_execz .LBB0_4
; %bb.8:                                ;   in Loop: Header=BB0_5 Depth=1
	ds_store_b32 v6, v4 offset:1024
	s_branch .LBB0_4
.LBB0_9:
	s_or_b32 exec_lo, exec_lo, s8
	v_lshl_add_u32 v1, s4, 2, v3
	v_or_b32_e32 v2, 0xffffff00, v0
	v_mov_b32_e32 v3, 0
	s_mov_b32 s2, 0
.LBB0_10:                               ; =>This Inner Loop Header: Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)
	v_add_nc_u32_e32 v2, 0x100, v2
	ds_store_b32 v1, v3 offset:13056
	v_add_nc_u32_e32 v1, 0x400, v1
	v_cmp_lt_u32_e32 vcc_lo, 0x187f, v2
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
	v_lshrrev_b32_e32 v39, 5, v0
	s_wait_xcnt 0x0
	s_lshl_b32 s1, s4, 2
	s_add_co_i32 s7, s7, -1
	s_mov_b64 s[34:35], src_shared_base
	s_or_b32 s34, s1, 0x3300
	s_add_co_i32 s37, s6, -1
	s_min_i32 s0, s5, s7
	s_mov_b32 s2, exec_lo
	v_cmpx_lt_i32_e32 0, v39
	s_xor_b32 s38, exec_lo, s2
	s_cbranch_execz .LBB0_16
; %bb.13:
	s_mov_b32 s39, exec_lo
	v_cmpx_eq_u32_e32 1, v39
	s_cbranch_execz .LBB0_15
; %bb.14:
	s_cmp_gt_i32 s30, 0
	s_mov_b32 s10, 0
	s_cselect_b32 s2, -1, 0
	s_lshl_b32 s4, s0, 7
	s_wait_kmcnt 0x0
	s_bfe_i64 s[6:7], s[20:21], 0x200000
	s_ashr_i32 s5, s4, 31
	s_and_b32 s2, s25, s2
	s_mul_u64 s[4:5], s[6:7], s[4:5]
	v_cndmask_b32_e64 v2, 0, 1, s2
	s_lshl_b64 s[4:5], s[4:5], 1
	s_mov_b32 s2, s30
	s_add_nc_u64 s[6:7], s[14:15], s[4:5]
	s_delay_alu instid0(SALU_CYCLE_1)
	v_dual_mov_b32 v1, s34 :: v_dual_mov_b32 v4, s6
	s_and_b32 s4, s7, 0x1ffffff
	v_readfirstlane_b32 s44, v2
	s_bitset1_b32 s4, 31
	v_readfirstlane_b32 s45, v1
	v_mov_b32_e32 v3, s4
	v_readfirstlane_b32 s46, v4
	s_lshr_b32 s4, s3, 16
	s_lshr_b64 s[6:7], s[2:3], 16
	s_lshl_b32 s5, s30, 16
	v_readfirstlane_b32 s47, v3
	s_or_b32 s7, s4, 0x600000
	s_movk_i32 s8, 0x80
	s_mov_b32 s4, 0x7510000
	s_mov_b32 s11, s10
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[44:47], s[4:11]
.LBB0_15:
	s_or_b32 exec_lo, exec_lo, s39
.LBB0_16:
	s_or_saveexec_b32 s38, s38
	s_min_i32 s2, s36, s37
	s_xor_b32 exec_lo, exec_lo, s38
	s_cbranch_execz .LBB0_18
; %bb.17:
	s_cmp_gt_i32 s30, 0
	s_mov_b32 s10, 0
	s_cselect_b32 s8, -1, 0
	s_lshl_b32 s4, s2, 6
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
	v_readfirstlane_b32 s44, v2
	s_bitset1_b32 s7, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v4, s6 :: v_dual_mov_b32 v3, s7
	s_lshr_b64 s[6:7], s[30:31], 16
	s_or_b32 s7, s4, 0x600000
	s_mov_b32 s8, 64
	v_readfirstlane_b32 s46, v4
	v_readfirstlane_b32 s47, v3
	s_mov_b32 s4, 0x7510000
	s_mov_b32 s11, s10
	s_mov_b32 s45, s10
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[44:47], s[4:11]
.LBB0_18:
	s_or_b32 exec_lo, exec_lo, s38
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	v_dual_lshlrev_b32 v1, 3, v39 :: v_dual_lshlrev_b32 v2, 6, v39
	s_and_b32 s41, s29, s25
	v_mov_b32_e32 v9, 0
	v_cndmask_b32_e64 v41, 0, 1, s41
	s_delay_alu instid0(VALU_DEP_3)
	v_and_b32_e32 v43, 48, v1
	v_and_b32_e32 v45, 64, v2
	s_and_not1_b32 vcc_lo, exec_lo, s17
	v_dual_mov_b32 v8, v9 :: v_dual_mov_b32 v7, v9
	v_dual_mov_b32 v6, v9 :: v_dual_mov_b32 v5, v9
	v_dual_mov_b32 v4, v9 :: v_dual_mov_b32 v3, v9
	v_dual_mov_b32 v2, v9 :: v_dual_mov_b32 v25, v9
	v_dual_mov_b32 v24, v9 :: v_dual_mov_b32 v23, v9
	v_dual_mov_b32 v22, v9 :: v_dual_mov_b32 v21, v9
	v_dual_mov_b32 v20, v9 :: v_dual_mov_b32 v19, v9
	v_dual_mov_b32 v18, v9 :: v_dual_mov_b32 v33, v9
	v_dual_mov_b32 v32, v9 :: v_dual_mov_b32 v31, v9
	v_dual_mov_b32 v30, v9 :: v_dual_mov_b32 v29, v9
	v_dual_mov_b32 v28, v9 :: v_dual_mov_b32 v27, v9
	v_dual_mov_b32 v26, v9 :: v_dual_mov_b32 v17, v9
	v_dual_mov_b32 v16, v9 :: v_dual_mov_b32 v15, v9
	v_dual_mov_b32 v14, v9 :: v_dual_mov_b32 v13, v9
	v_dual_mov_b32 v12, v9 :: v_dual_mov_b32 v11, v9
	v_mov_b32_e32 v10, v9
	s_mov_b32 s8, 64
	s_barrier_wait -1
	s_cbranch_vccnz .LBB0_42
; %bb.19:
	v_and_b32_e32 v2, 15, v0
	v_and_b32_e32 v3, 16, v0
	v_mul_u32_u24_e32 v5, 0x60, v45
	v_mul_u32_u24_e32 v1, 0x60, v43
	s_mov_b64 s[4:5], src_shared_base
	v_mul_u32_u24_e32 v4, 0x60, v2
	s_or_b32 s6, s1, 0x9900
	s_mov_b32 s7, s5
	s_mov_b32 s11, 0
	s_and_b64 s[6:7], s[6:7], 15
	v_or_b32_e32 v7, v4, v3
	s_sub_co_i32 s4, 16, s6
	s_mov_b32 s43, s5
	s_lshr_b32 s4, s4, 2
	s_cmp_lg_u64 s[6:7], 0
	v_or_b32_e32 v9, v7, v5
	s_cselect_b32 s4, s4, 0
	s_mul_hi_i32 s6, s16, 0x2aaaaaab
	s_lshl2_add_u32 s1, s4, s1
	s_mov_b32 s42, s35
	v_dual_lshrrev_b32 v6, 4, v9 :: v_dual_bitop2_b32 v1, v1, v3 bitop3:0x54
	v_mov_b32_e32 v37, 0
	s_add_co_i32 s4, s1, 0xcc00
	v_sub_nc_u32_e32 v13, 0x197f, v0
	s_delay_alu instid0(VALU_DEP_3)
	v_and_b32_e32 v6, 0x1f8, v6
	v_mad_u32_u24 v1, 0x60, v2, v1
	s_and_b32 s10, s4, 15
	s_add_co_i32 s44, s1, 0x9900
	s_sub_co_i32 s7, 16, s10
	v_dual_mov_b32 v24, v37 :: v_dual_mov_b32 v26, v37
	v_lshrrev_b32_e32 v4, 4, v1
	s_lshr_b32 s1, s7, 2
	s_cmp_lg_u64 s[10:11], 0
	v_dual_mov_b32 v27, v37 :: v_dual_mov_b32 v28, v37
	s_delay_alu instid0(VALU_DEP_2)
	v_and_b32_e32 v4, 0x3f8, v4
	s_cselect_b32 s1, s1, 0
	s_lshr_b32 s7, s6, 31
	s_ashr_i32 s45, s6, 4
	s_lshl_b32 s10, s1, 2
	v_add_nc_u32_e32 v34, v4, v1
	v_add_nc_u32_e32 v4, 0x600, v9
	v_mad_u32_u24 v20, 0x60, v45, v7
	v_mad_u32_u24 v22, 0x60, v43, v7
	s_add_co_i32 s45, s45, s7
	s_cmp_lt_i32 s3, 0x80
	v_dual_lshrrev_b32 v4, 4, v4 :: v_dual_bitop2_b32 v3, 32, v3 bitop3:0x54
	v_add_nc_u32_e32 v38, v6, v9
	v_add_nc_u32_e32 v6, 0xc00, v9
	v_add_nc_u32_e32 v8, 0x1200, v9
	s_cselect_b32 s46, -1, 0
	v_mad_u32_u24 v1, 0x60, v2, v3
	v_or_b32_e32 v3, v5, v3
	s_lshl_b32 s0, s0, 7
	s_wait_kmcnt 0x0
	s_bfe_i64 s[6:7], s[20:21], 0x200000
	s_ashr_i32 s1, s0, 31
	v_mad_u32_u24 v1, 0x60, v43, v1
	v_mad_u32_u24 v2, 0x60, v2, v3
	v_and_b32_e32 v3, 0x3f8, v4
	v_mov_b32_e32 v15, v37
	s_mul_u64 s[0:1], s[6:7], s[0:1]
	s_add_nc_u64 s[36:37], s[4:5], s[10:11]
	v_add_nc_u32_e32 v4, 0x600, v2
	v_dual_mov_b32 v3, v37 :: v_dual_add_nc_u32 v42, v3, v9
	v_lshrrev_b32_e32 v5, 4, v6
	s_lshl_b64 s[0:1], s[0:1], 1
	v_lshrrev_b32_e32 v1, 4, v1
	s_add_nc_u64 s[20:21], s[14:15], s[0:1]
	s_lshl_b32 s4, s2, 6
	v_and_b32_e32 v5, 0x3f8, v5
	s_ashr_i32 s5, s4, 31
	v_and_b32_e32 v16, 0x3f8, v1
	v_lshrrev_b32_e32 v1, 4, v2
	s_bfe_i64 s[6:7], s[12:13], 0x200000
	v_add_nc_u32_e32 v44, v5, v9
	v_dual_mov_b32 v5, v37 :: v_dual_lshrrev_b32 v6, 4, v8
	v_add_nc_u32_e32 v8, 64, v7
	v_and_b32_e32 v17, 0x3f8, v1
	v_mov_b32_e32 v7, v37
	s_mul_u64 s[4:5], s[6:7], s[4:5]
	v_and_b32_e32 v11, 0x3f8, v6
	v_add_nc_u32_e32 v6, 0xc00, v2
	v_add_nc_u32_e32 v2, 0x1200, v2
	v_lshrrev_b32_e32 v1, 4, v4
	v_mad_u32_u24 v10, 0x60, v43, v8
	v_mad_u32_u24 v8, 0x60, v45, v8
	v_dual_lshrrev_b32 v4, 4, v6 :: v_dual_add_nc_u32 v48, v11, v9
	v_add_nc_u32_e32 v52, v17, v20
	v_dual_mov_b32 v9, v37 :: v_dual_lshrrev_b32 v6, 4, v2
	v_and_b32_e32 v36, 0x7f8, v1
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v2, 0x7f8, v4
	v_lshrrev_b32_e32 v1, 4, v10
	v_dual_mov_b32 v69, v37 :: v_dual_add_nc_u32 v10, 0xc00, v8
	v_and_b32_e32 v4, 0x7f8, v6
	v_dual_mov_b32 v11, v37 :: v_dual_add_nc_u32 v6, 0x600, v8
	v_add_nc_u32_e32 v12, 0x1200, v8
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_lshrrev_b32 v8, 4, v8 :: v_dual_lshrrev_b32 v10, 4, v10
	v_and_b32_e32 v18, 0x3f8, v1
	v_dual_mov_b32 v23, v37 :: v_dual_lshrrev_b32 v40, 8, v35
	v_and_b32_e32 v19, 0x3f8, v8
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_and_b32_e32 v8, 0x7f8, v10
	v_dual_add_nc_u32 v58, v18, v22 :: v_dual_add_nc_u32 v35, -1, v40
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_dual_add_nc_u32 v50, v16, v22 :: v_dual_add_nc_u32 v60, v19, v20
	v_dual_mov_b32 v19, v37 :: v_dual_lshrrev_b32 v1, 4, v6
	v_dual_mov_b32 v25, v37 :: v_dual_lshrrev_b32 v12, 4, v12
	v_dual_mov_b32 v18, v37 :: v_dual_mov_b32 v22, v37
	v_and_b32_e32 v6, 0x7f8, v1
	v_lshrrev_b32_e32 v1, 8, v13
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_and_b32_e32 v10, 0x7f8, v12
	v_dual_mov_b32 v13, v37 :: v_dual_add_nc_u32 v12, 0x600, v20
	v_dual_mov_b32 v29, v37 :: v_dual_mov_b32 v30, v37
	v_dual_add_nc_u32 v14, 1, v1 :: v_dual_add_nc_u32 v1, 2, v40
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_nc_u64_e32 v[46:47], v[36:37], v[12:13]
	v_dual_mov_b32 v31, v37 :: v_dual_add_nc_u32 v36, 0xc00, v20
	v_dual_mov_b32 v32, v37 :: v_dual_bitop2_b32 v49, 26, v14 bitop3:0x40
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_and_b32_e32 v51, 30, v1
	v_add_nc_u64_e32 v[54:55], v[2:3], v[36:37]
	v_add_nc_u64_e32 v[64:65], v[8:9], v[36:37]
	v_or_b32_e32 v1, 0x100, v0
	v_lshl_or_b32 v21, v49, 8, v0
	v_cmp_ne_u32_e64 s0, v14, v49
	v_add_nc_u32_e32 v14, 0x1200, v20
	v_dual_mov_b32 v2, v37 :: v_dual_mov_b32 v8, v37
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_nc_u32_e32 v47, 0xffffff00, v21
	v_lshlrev_b32_e32 v68, 2, v21
	v_add_nc_u64_e32 v[56:57], v[4:5], v[14:15]
	v_add_nc_u64_e32 v[66:67], v[10:11], v[14:15]
	v_mov_b32_e32 v21, v37
	v_add_nc_u64_e32 v[62:63], v[6:7], v[12:13]
	v_dual_mov_b32 v4, v37 :: v_dual_mov_b32 v6, v37
	v_dual_mov_b32 v20, v37 :: v_dual_mov_b32 v33, v37
	v_dual_mov_b32 v10, v37 :: v_dual_mov_b32 v12, v37
	v_dual_mov_b32 v14, v37 :: v_dual_mov_b32 v16, v37
	v_mov_b32_e32 v17, v37
	s_lshr_b32 s47, s3, 16
	s_lshr_b32 s48, s31, 16
	s_lshl_b64 s[4:5], s[4:5], 1
	s_movk_i32 s16, 0x80
	s_or_b32 s47, s47, 0x600000
	s_or_b32 s48, s48, 0x600000
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
	v_nop
	v_nop
	v_nop
	v_nop
	v_mov_b64_e32 v[70:71], v[0:1]
	s_cmp_lg_u32 s5, 0
	s_mov_b32 s10, 0
	s_cselect_b32 s7, s43, s35
	s_cselect_b32 s6, s44, 0
	s_mov_b32 s12, 0
	s_branch .LBB0_24
.LBB0_23:                               ;   in Loop: Header=BB0_24 Depth=2
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s1
	s_add_co_i32 s12, s12, 2
	v_add_nc_u32_e32 v71, 0x200, v71
	v_cmp_eq_u32_e32 vcc_lo, s12, v51
	v_add_nc_u32_e32 v70, 0x200, v70
	s_or_b32 s10, vcc_lo, s10
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s10
	s_cbranch_execz .LBB0_28
.LBB0_24:                               ;   Parent Loop BB0_21 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_mov_b32 s13, exec_lo
	v_cmp_le_u32_e32 vcc_lo, s12, v35
	v_cmpx_le_u32_e64 s12, v40
	s_cbranch_execz .LBB0_26
; %bb.25:                               ;   in Loop: Header=BB0_24 Depth=2
	v_mov_b32_e32 v36, v70
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u64 v[72:73], v[36:37], 2, s[6:7]
	flat_store_b32 v[72:73], v37
.LBB0_26:                               ;   in Loop: Header=BB0_24 Depth=2
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s13
	s_and_saveexec_b32 s1, vcc_lo
	s_cbranch_execz .LBB0_23
; %bb.27:                               ;   in Loop: Header=BB0_24 Depth=2
	v_mov_b32_e32 v36, v71
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u64 v[72:73], v[36:37], 2, s[6:7]
	flat_store_b32 v[72:73], v37
	s_branch .LBB0_23
.LBB0_28:                               ;   in Loop: Header=BB0_21 Depth=1
	s_or_b32 exec_lo, exec_lo, s10
	v_mov_b64_e32 v[70:71], v[0:1]
	v_mov_b32_e32 v53, v49
	s_cmp_lg_u32 s5, 0
	s_mov_b32 s1, 0
	s_cselect_b32 s7, s37, s42
	s_cselect_b32 s6, s36, s34
.LBB0_29:                               ;   Parent Loop BB0_21 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_dual_mov_b32 v36, v70 :: v_dual_add_nc_u32 v53, -2, v53
	v_dual_mov_b32 v72, v71 :: v_dual_mov_b32 v73, v37
	v_add_nc_u32_e32 v71, 0x200, v71
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u64 v[74:75], v[36:37], 2, s[6:7]
	v_cmp_eq_u32_e32 vcc_lo, 0, v53
	v_add_nc_u32_e32 v70, 0x200, v70
	v_lshl_add_u64 v[72:73], v[72:73], 2, s[6:7]
	s_clause 0x1
	flat_store_b32 v[74:75], v37
	flat_store_b32 v[72:73], v37
	s_or_b32 s1, vcc_lo, s1
	s_wait_xcnt 0x0
	s_and_not1_b32 exec_lo, exec_lo, s1
	s_cbranch_execnz .LBB0_29
; %bb.30:                               ;   in Loop: Header=BB0_21 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_33
; %bb.31:                               ;   in Loop: Header=BB0_21 Depth=1
	v_add_nc_u64_e32 v[70:71], s[6:7], v[68:69]
	v_mov_b32_e32 v36, v47
	s_mov_b32 s6, 0
.LBB0_32:                               ;   Parent Loop BB0_21 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v36, 0x100, v36
	flat_store_b32 v[70:71], v37
	s_wait_xcnt 0x0
	v_add_nc_u64_e32 v[70:71], 0x400, v[70:71]
	v_cmp_lt_u32_e32 vcc_lo, 0x187f, v36
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
	v_cmpx_lt_i32_e32 0, v39
	s_xor_b32 s6, exec_lo, s2
	s_cbranch_execnz .LBB0_37
; %bb.35:                               ;   in Loop: Header=BB0_21 Depth=1
	s_and_not1_saveexec_b32 s2, s6
	s_cbranch_execnz .LBB0_40
.LBB0_36:                               ;   in Loop: Header=BB0_21 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s41
	s_cbranch_vccnz .LBB0_20
	s_branch .LBB0_41
.LBB0_37:                               ;   in Loop: Header=BB0_21 Depth=1
	s_mov_b32 s7, exec_lo
	v_cmpx_eq_u32_e32 1, v39
	s_cbranch_execz .LBB0_39
; %bb.38:                               ;   in Loop: Header=BB0_21 Depth=1
	s_cmp_lg_u32 s5, 0
	s_mul_i32 s10, s1, 0x60
	s_cselect_b32 s2, s36, s34
	s_cmp_gt_i32 s30, 0
	s_mov_b32 s17, s9
	s_cselect_b32 s14, -1, 0
	s_lshl_b64 s[12:13], s[10:11], 1
	s_mov_b32 s18, s11
	s_add_nc_u64 s[12:13], s[20:21], s[12:13]
	v_nop
	v_nop
	v_dual_mov_b32 v53, s2 :: v_dual_mov_b32 v70, s12
	s_and_b32 s10, s13, 0x1ffffff
	s_and_b32 s13, s25, s14
	s_bitset1_b32 s10, 31
	v_cndmask_b32_e64 v36, 0, 1, s13
	v_mov_b32_e32 v55, s10
	s_mov_b32 s2, s30
	v_readfirstlane_b32 s53, v53
	v_readfirstlane_b32 s54, v70
	v_readfirstlane_b32 s52, v36
	v_readfirstlane_b32 s55, v55
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
	s_cselect_b32 s5, s44, 0
	s_cmp_gt_i32 s30, 0
	s_cselect_b32 s1, -1, 0
	s_lshl_b64 s[6:7], s[10:11], 1
	s_and_b32 s1, s29, s1
	s_add_nc_u64 s[6:7], s[38:39], s[6:7]
	v_cndmask_b32_e64 v36, 0, 1, s1
	s_and_b32 s7, s7, 0x1ffffff
	v_nop
	v_nop
	v_dual_mov_b32 v53, s5 :: v_dual_mov_b32 v70, s6
	s_bitset1_b32 s7, 31
	v_readfirstlane_b32 s12, v36
	v_mov_b32_e32 v55, s7
	s_delay_alu instid0(VALU_DEP_3)
	v_readfirstlane_b32 s13, v53
	v_readfirstlane_b32 s14, v70
	s_lshr_b64 s[6:7], s[30:31], 16
	s_lshl_b32 s5, s30, 16
	v_readfirstlane_b32 s15, v55
	s_mov_b32 s7, s48
	s_mov_b32 s10, s11
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	tensor_load_to_lds s[12:15], s[4:11]
	s_or_b32 exec_lo, exec_lo, s2
	s_and_not1_b32 vcc_lo, exec_lo, s41
	s_cbranch_vccnz .LBB0_20
.LBB0_41:                               ;   in Loop: Header=BB0_21 Depth=1
	s_cmp_lg_u32 s50, 0
	s_cselect_b32 s1, s44, 0
	s_cselect_b32 s2, s36, s34
	v_lshl_add_u32 v36, v34, 1, s1
	v_lshl_add_u32 v53, v38, 1, s2
	v_lshl_add_u32 v55, v52, 1, s2
	v_lshl_add_u32 v57, v46, 1, s2
	v_lshl_add_u32 v59, v54, 1, s2
	ds_load_b128 v[70:73], v36
	ds_load_b128 v[74:77], v36 offset:16
	ds_load_b128 v[78:81], v53
	ds_load_b128 v[82:85], v53 offset:16
	v_lshl_add_u32 v36, v42, 1, s2
	v_lshl_add_u32 v53, v48, 1, s2
	v_lshl_add_u32 v61, v56, 1, s2
	ds_load_b128 v[94:97], v55 offset:64
	ds_load_b128 v[98:101], v55 offset:80
	ds_load_b128 v[86:89], v36 offset:3072
	ds_load_b128 v[110:113], v57 offset:64
	ds_load_b128 v[114:117], v57 offset:80
	ds_load_b128 v[118:121], v59 offset:64
	ds_load_b128 v[122:125], v59 offset:80
	ds_load_b128 v[126:129], v61 offset:64
	ds_load_b128 v[130:133], v61 offset:80
	; sched_group_barrier mask(0x00000100) size(5) SyncID(0)
	s_wait_dscnt 0x9
	v_wmma_f32_16x16x32_bf16 v[26:33], v[70:77], v[78:85], v[26:33]
	ds_load_b128 v[90:93], v36 offset:3088
	v_lshl_add_u32 v36, v44, 1, s2
	ds_load_b128 v[102:105], v36 offset:6144
	ds_load_b128 v[106:109], v36 offset:6160
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[2:9], v[70:77], v[102:109], v[2:9] matrix_a_reuse
	ds_load_b128 v[78:81], v53 offset:9216
	ds_load_b128 v[82:85], v53 offset:9232
	v_lshl_add_u32 v53, v50, 1, s1
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[10:17], v[70:77], v[78:85], v[10:17] matrix_a_reuse
	ds_load_b128 v[78:81], v53 offset:64
	ds_load_b128 v[82:85], v53 offset:80
	; sched_group_barrier mask(0x00000100) size(5) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[18:25], v[70:77], v[86:93], v[18:25] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_barrier mask(0x00000000)
	v_lshl_add_u32 v36, v58, 1, s1
	v_lshl_add_u32 v53, v60, 1, s2
	v_lshl_add_u32 v55, v62, 1, s2
	ds_load_b128 v[70:73], v36 offset:128
	ds_load_b128 v[74:77], v36 offset:144
	ds_load_b128 v[86:89], v53 offset:128
	ds_load_b128 v[90:93], v53 offset:144
	v_lshl_add_u32 v36, v64, 1, s2
	v_lshl_add_u32 v53, v66, 1, s2
	ds_load_b128 v[102:105], v55 offset:128
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[26:33], v[78:85], v[94:101], v[26:33]
	; sched_group_barrier mask(0x00000100) size(5) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[18:25], v[78:85], v[110:117], v[18:25] matrix_a_reuse
	ds_load_b128 v[106:109], v55 offset:144
	ds_load_b128 v[94:97], v36 offset:128
	ds_load_b128 v[98:101], v36 offset:144
	ds_load_b128 v[110:113], v53 offset:128
	ds_load_b128 v[114:117], v53 offset:144
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(5) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[2:9], v[78:85], v[118:125], v[2:9] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[10:17], v[78:85], v[126:133], v[10:17] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[26:33], v[70:77], v[86:93], v[26:33]
	; sched_group_barrier mask(0x00000100) size(5) SyncID(0)
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[18:25], v[70:77], v[102:109], v[18:25] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(5) SyncID(0)
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[2:9], v[70:77], v[94:101], v[2:9] matrix_a_reuse
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[10:17], v[70:77], v[110:117], v[10:17] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
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
	s_and_b32 vcc_lo, exec_lo, s41
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_vccz .LBB0_44
; %bb.43:
	v_lshrrev_b32_e32 v1, 1, v0
	v_and_or_b32 v34, v0, 15, v45
	v_nop
	v_cvt_pk_bf16_f32 v9, v8, v9
	v_cvt_pk_bf16_f32 v8, v6, v7
	v_cvt_pk_bf16_f32 v25, v24, v25
	v_and_or_b32 v1, v1, 8, v43
	v_cvt_pk_bf16_f32 v24, v22, v23
	v_cvt_pk_bf16_f32 v22, v18, v19
	v_cvt_pk_bf16_f32 v23, v20, v21
	v_lshlrev_b32_e32 v20, 3, v34
	v_lshl_or_b32 v1, v34, 6, v1
	v_cvt_pk_bf16_f32 v7, v4, v5
	v_cvt_pk_bf16_f32 v33, v32, v33
	v_cvt_pk_bf16_f32 v32, v30, v31
	v_cvt_pk_bf16_f32 v31, v28, v29
	v_add_nc_u32_e32 v6, 0x400, v1
	v_add_nc_u32_e32 v18, 0x800, v1
	v_add_nc_u32_e32 v19, 0xc00, v1
	v_lshlrev_b32_e32 v1, 1, v1
	v_cvt_pk_bf16_f32 v30, v26, v27
	v_lshrrev_b32_e32 v4, 3, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_lshrrev_b32 v5, 3, v18 :: v_dual_lshrrev_b32 v6, 3, v19
	v_and_b32_e32 v18, 0x2f0, v20
	v_and_b32_e32 v4, 0x7f0, v4
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_and_b32_e32 v5, 0x7f0, v5
	v_and_b32_e32 v19, 0x7f0, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_nc_u32_e32 v18, v18, v1
	v_cvt_pk_bf16_f32 v6, v2, v3
	v_dual_add_nc_u32 v20, v4, v1 :: v_dual_add_nc_u32 v21, v5, v1
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v1, v19, v1
	v_cvt_pk_bf16_f32 v5, v16, v17
	v_cvt_pk_bf16_f32 v4, v14, v15
	v_cvt_pk_bf16_f32 v3, v12, v13
	v_cvt_pk_bf16_f32 v2, v10, v11
	ds_store_b128 v18, v[30:33]
	ds_store_b128 v20, v[22:25] offset:2048
	ds_store_b128 v21, v[6:9] offset:4096
	ds_store_b128 v1, v[2:5] offset:6144
.LBB0_44:
	v_cmp_ne_u32_e32 vcc_lo, 1, v41
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
	v_nop
	v_xad_u32 v2, v0, -1, s3
	s_lshl_b64 s[0:1], s[28:29], 1
	s_ashr_i32 s25, s24, 31
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[4:5], s[22:23], s[0:1]
	s_mov_b32 s0, 0
                                        ; implicit-def: $vgpr1
                                        ; implicit-def: $vgpr6
                                        ; implicit-def: $sgpr12_sgpr13
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_u32_e32 0x2ff, v2
	s_xor_b32 s14, exec_lo, s1
	s_cbranch_execnz .LBB0_49
; %bb.47:
	s_or_saveexec_b32 s1, s14
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
	v_lshl_add_u32 v11, v11, 6, v12
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_mul_lo_u32 v16, v27, s18
	v_dual_add_nc_u32 v24, s21, v27 :: v_dual_sub_nc_u32 v6, v2, v6
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
	s_cbranch_execnz .LBB0_50
; %bb.51:
	s_or_b32 exec_lo, exec_lo, s23
	v_cmp_ne_u32_e32 vcc_lo, v8, v9
	v_lshl_or_b32 v0, v9, 8, v0
	v_dual_mov_b32 v6, s15 :: v_dual_mov_b32 v1, s22
	s_and_b32 s0, vcc_lo, exec_lo
	s_or_saveexec_b32 s1, s14
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
	s_cbranch_execnz .LBB0_54
.LBB0_55:
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end0:
	.size	bm064_bn128_bk096_wm4_wn2_mc0, .Lfunc_end0-bm064_bn128_bk096_wm4_wn2_mc0
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel bm064_bn128_bk096_wm4_wn2_mc0
		.amdhsa_group_segment_fixed_size 78336
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
		.amdhsa_next_free_vgpr 134
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
		.amdhsa_inst_pref_size 45
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text.bm064_bn128_bk096_wm4_wn2_mc0,"axG",@progbits,bm064_bn128_bk096_wm4_wn2_mc0,comdat
                                        ; -- End function
	.set .Lbm064_bn128_bk096_wm4_wn2_mc0.num_vgpr, 134
	.set .Lbm064_bn128_bk096_wm4_wn2_mc0.num_agpr, 0
	.set .Lbm064_bn128_bk096_wm4_wn2_mc0.numbered_sgpr, 56
	.set .Lbm064_bn128_bk096_wm4_wn2_mc0.num_named_barrier, 0
	.set .Lbm064_bn128_bk096_wm4_wn2_mc0.private_seg_size, 0
	.set .Lbm064_bn128_bk096_wm4_wn2_mc0.uses_vcc, 1
	.set .Lbm064_bn128_bk096_wm4_wn2_mc0.uses_flat_scratch, 1
	.set .Lbm064_bn128_bk096_wm4_wn2_mc0.has_dyn_sized_stack, 0
	.set .Lbm064_bn128_bk096_wm4_wn2_mc0.has_recursion, 0
	.set .Lbm064_bn128_bk096_wm4_wn2_mc0.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 5680
; TotalNumSgprs: 58
; NumVgprs: 134
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 78336 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 8
; NumSGPRsForWavesPerEU: 58
; NumVGPRsForWavesPerEU: 134
; NamedBarCnt: 0
; Occupancy: 7
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
	.type	__hip_cuid_fce36e4da1d2429a,@object ; @__hip_cuid_fce36e4da1d2429a
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_fce36e4da1d2429a
__hip_cuid_fce36e4da1d2429a:
	.byte	0                               ; 0x0
	.size	__hip_cuid_fce36e4da1d2429a, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git aa451e1fe6a793394d6733051b1778633063ae96+PATCHED:d17c5aa0e3ea29cde402f58f27e39b6034effa27)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_fce36e4da1d2429a
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
    .group_segment_fixed_size: 78336
    .kernarg_segment_align: 8
    .kernarg_segment_size: 132
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           bm064_bn128_bk096_wm4_wn2_mc0
    .private_segment_fixed_size: 0
    .sgpr_count:     58
    .sgpr_spill_count: 0
    .symbol:         bm064_bn128_bk096_wm4_wn2_mc0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     134
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
