	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.section	.text.bm064_bn064_bk096_wm4_wn2_mc1,"axG",@progbits,bm064_bn064_bk096_wm4_wn2_mc1,comdat
	.protected	bm064_bn064_bk096_wm4_wn2_mc1 ; -- Begin function bm064_bn064_bk096_wm4_wn2_mc1
	.globl	bm064_bn064_bk096_wm4_wn2_mc1
	.p2align	8
	.type	bm064_bn064_bk096_wm4_wn2_mc1,@function
bm064_bn064_bk096_wm4_wn2_mc1: ; @bm064_bn064_bk096_wm4_wn2_mc1
; %bb.0:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1 ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_load_b96 s[24:26], s[0:1], 0x78 nv
	s_mov_b64 s[2:3], src_shared_base
	s_movk_i32 s2, 0x3300
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_and_b64 s[2:3], s[2:3], 12
	s_sub_co_i32 s4, 16, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_lshr_b32 s4, s4, 2
	s_cmp_lg_u64 s[2:3], 0
	s_cselect_b32 s5, s4, 0
	s_and_b32 s2, ttmp6, 15
	s_bfe_u32 s3, ttmp6, 0x40004
	s_lshl2_add_u32 s36, ttmp9, s2
	s_lshl2_add_u32 s4, ttmp7, s3
	s_lshl_b32 s28, s36, 6
	s_wait_kmcnt 0x0
	s_add_co_i32 s2, s24, 63
	s_add_co_i32 s3, s25, 63
	s_sub_co_i32 s6, s24, s28
	s_ashr_i32 s7, s2, 31
	s_ashr_i32 s8, s3, 31
	s_min_i32 s27, s6, 64
	s_lshr_b32 s6, s7, 26
	s_lshr_b32 s7, s8, 26
	s_add_co_i32 s2, s2, s6
	s_add_co_i32 s3, s3, s7
	s_ashr_i32 s6, s2, 6
	s_ashr_i32 s7, s3, 6
	s_cmp_lt_i32 s36, s6
	s_mov_b32 s9, s26
	s_cselect_b32 s29, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_and_b32 s2, s29, exec_lo
	s_cselect_b32 s31, s27, 0
	s_lshl_b32 s33, s4, 6
	s_sub_co_i32 s2, s25, s33
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_min_i32 s2, s2, 64
	s_cmp_lt_i32 s4, s7
	s_cselect_b32 s25, -1, 0
	s_and_b32 s3, s25, exec_lo
	s_cselect_b32 s3, s2, 0
	s_add_co_i32 s12, s26, 0x5f
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 2, 1), 1
	s_min_i32 s2, s26, 0x60
	s_cmp_gt_i32 s12, 0x5f
	s_cselect_b32 s13, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_and_b32 s8, s13, exec_lo
	s_cselect_b32 s30, s2, 0
	s_cmp_lt_i32 s31, 64
	s_cselect_b32 s40, -1, 0
	s_and_b32 vcc_lo, exec_lo, s40
	s_mov_b32 s2, s40
	s_cbranch_vccnz .LBB0_2
; %bb.1:
	s_cmp_lt_i32 s3, 64
	s_cselect_b32 s2, -1, 0
	s_cmp_lt_i32 s30, 0x60
	s_cselect_b32 s8, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 s2, s8, s2
.LBB0_2:
	v_sub_nc_u32_e32 v19, 0xcbf, v0
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB0_16
; %bb.3:
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_dual_lshrrev_b32 v2, 8, v19 :: v_dual_lshlrev_b32 v4, 2, v0
	s_mov_b32 s8, 0
	s_mov_b32 s10, 0
	v_dual_mov_b32 v5, 0 :: v_dual_add_nc_u32 v3, 2, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mov_b32 v6, v4 :: v_dual_add_nc_u32 v1, -1, v2
	v_and_b32_e32 v3, 30, v3
	s_branch .LBB0_5
.LBB0_4:                                ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
	s_add_co_i32 s10, s10, 2
	v_add_nc_u32_e32 v6, 0x800, v6
	v_cmp_eq_u32_e32 vcc_lo, s10, v3
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
	ds_store_b32 v6, v5
; %bb.7:                                ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s11
	s_and_saveexec_b32 s2, vcc_lo
	s_cbranch_execz .LBB0_4
; %bb.8:                                ;   in Loop: Header=BB0_5 Depth=1
	ds_store_b32 v6, v5 offset:1024
	s_branch .LBB0_4
.LBB0_9:
	s_or_b32 exec_lo, exec_lo, s8
	v_lshl_add_u32 v4, s5, 2, v4
	v_mov_b32_e32 v5, 0
	s_mov_b32 s8, 0
	s_mov_b32 s10, 0
	s_branch .LBB0_11
.LBB0_10:                               ;   in Loop: Header=BB0_11 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
	s_add_co_i32 s10, s10, 2
	v_add_nc_u32_e32 v4, 0x800, v4
	v_cmp_eq_u32_e32 vcc_lo, s10, v3
	s_or_b32 s8, vcc_lo, s8
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s8
	s_cbranch_execz .LBB0_15
.LBB0_11:                               ; =>This Inner Loop Header: Depth=1
	s_mov_b32 s11, exec_lo
	v_cmp_le_u32_e32 vcc_lo, s10, v1
	v_cmpx_le_u32_e64 s10, v2
; %bb.12:                               ;   in Loop: Header=BB0_11 Depth=1
	ds_store_b32 v4, v5 offset:13056
; %bb.13:                               ;   in Loop: Header=BB0_11 Depth=1
	s_or_b32 exec_lo, exec_lo, s11
	s_and_saveexec_b32 s2, vcc_lo
	s_cbranch_execz .LBB0_10
; %bb.14:                               ;   in Loop: Header=BB0_11 Depth=1
	ds_store_b32 v4, v5 offset:14080
	s_branch .LBB0_10
.LBB0_15:
	s_or_b32 exec_lo, exec_lo, s8
	s_barrier_signal -1
	s_barrier_wait -1
.LBB0_16:
	s_clause 0x2
	s_load_b64 s[14:15], s[0:1], 0x0 nv
	s_load_b128 s[16:19], s[0:1], 0x20 nv
	s_load_b128 s[20:23], s[0:1], 0x48 nv
	s_wait_xcnt 0x0
	s_lshl_b32 s1, s4, 2
	v_lshrrev_b32_e32 v23, 5, v0
	s_lshl_b32 s37, s5, 2
	s_add_co_i32 s7, s7, -1
	s_and_b32 s1, s1, 12
	s_mov_b64 s[34:35], src_shared_base
	s_or_b32 s34, s37, 0x3300
	s_add_co_i32 s38, s6, -1
	s_min_i32 s0, s4, s7
	s_and_b32 s39, s36, 3
	s_lshl_b32 s1, 15, s1
	s_mov_b32 s2, exec_lo
	v_cmpx_lt_i32_e32 0, v23
	s_xor_b32 s41, exec_lo, s2
	s_cbranch_execz .LBB0_20
; %bb.17:
	s_mov_b32 s42, exec_lo
	v_cmpx_eq_u32_e32 1, v23
	s_cbranch_execz .LBB0_19
; %bb.18:
	s_cmp_gt_i32 s30, 0
	s_mov_b32 s10, 0
	s_cselect_b32 s2, -1, 0
	s_lshl_b32 s4, s0, 6
	s_wait_kmcnt 0x0
	s_bfe_i64 s[6:7], s[20:21], 0x200000
	s_ashr_i32 s5, s4, 31
	s_and_b32 s2, s25, s2
	s_mul_u64 s[4:5], s[6:7], s[4:5]
	v_cndmask_b32_e64 v2, 0, 1, s2
	s_lshl_b64 s[4:5], s[4:5], 1
	s_mov_b32 s2, s30
	s_add_nc_u64 s[6:7], s[18:19], s[4:5]
	s_delay_alu instid0(SALU_CYCLE_1)
	v_dual_mov_b32 v1, s34 :: v_dual_mov_b32 v4, s6
	s_and_b32 s5, s7, 0x1ffffff
	v_readfirstlane_b32 s44, v2
	s_bitset1_b32 s5, 31
	v_readfirstlane_b32 s45, v1
	v_mov_b32_e32 v3, s5
	v_readfirstlane_b32 s46, v4
	s_lshr_b64 s[6:7], s[2:3], 16
	s_lshr_b32 s2, s3, 16
	s_or_b32 s4, s1, 0x7510000
	v_readfirstlane_b32 s47, v3
	s_lshl_b32 s5, s30, 16
	s_or_b32 s7, s2, 0x600000
	s_mov_b32 s8, 64
	s_mov_b32 s11, s10
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[44:47], s[4:11]
.LBB0_19:
	s_or_b32 exec_lo, exec_lo, s42
.LBB0_20:
	s_or_saveexec_b32 s41, s41
	s_min_i32 s38, s36, s38
	s_lshl_b32 s2, 0x1111, s39
	s_xor_b32 exec_lo, exec_lo, s41
	s_cbranch_execz .LBB0_22
; %bb.21:
	s_cmp_gt_i32 s30, 0
	s_mov_b32 s10, 0
	s_cselect_b32 s8, -1, 0
	s_lshl_b32 s4, s38, 6
	s_wait_kmcnt 0x0
	s_bfe_i64 s[6:7], s[16:17], 0x200000
	s_ashr_i32 s5, s4, 31
	s_and_b32 s8, s29, s8
	s_mul_u64 s[4:5], s[6:7], s[4:5]
	v_cndmask_b32_e64 v2, 0, 1, s8
	s_lshl_b64 s[6:7], s[4:5], 1
	s_lshr_b32 s8, s31, 16
	s_add_nc_u64 s[6:7], s[14:15], s[6:7]
	s_or_b32 s4, s2, 0x7510000
	s_and_b32 s7, s7, 0x1ffffff
	v_readfirstlane_b32 s44, v2
	s_bitset1_b32 s7, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v4, s6 :: v_dual_mov_b32 v3, s7
	s_lshr_b64 s[6:7], s[30:31], 16
	s_lshl_b32 s5, s30, 16
	s_or_b32 s7, s8, 0x600000
	v_readfirstlane_b32 s46, v4
	v_readfirstlane_b32 s47, v3
	s_mov_b32 s8, 64
	s_mov_b32 s11, s10
	s_mov_b32 s45, s10
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[44:47], s[4:11]
.LBB0_22:
	s_or_b32 exec_lo, exec_lo, s41
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s4, exec_lo
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	v_cmpx_gt_u32_e32 32, v0
	s_cbranch_execz .LBB0_24
; %bb.23:
	s_barrier_signal -3
.LBB0_24:
	s_or_b32 exec_lo, exec_lo, s4
	v_dual_lshlrev_b32 v1, 3, v23 :: v_dual_mov_b32 v9, 0
	s_and_b32 s41, s29, s25
	s_and_not1_b32 vcc_lo, exec_lo, s13
	v_cndmask_b32_e64 v25, 0, 1, s41
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v8, v9 :: v_dual_bitop2_b32 v27, 48, v1 bitop3:0x40
	v_dual_mov_b32 v7, v9 :: v_dual_mov_b32 v6, v9
	v_dual_mov_b32 v5, v9 :: v_dual_mov_b32 v4, v9
	v_dual_mov_b32 v3, v9 :: v_dual_mov_b32 v2, v9
	v_dual_mov_b32 v17, v9 :: v_dual_mov_b32 v16, v9
	v_dual_mov_b32 v15, v9 :: v_dual_mov_b32 v14, v9
	v_dual_mov_b32 v13, v9 :: v_dual_mov_b32 v12, v9
	v_dual_mov_b32 v11, v9 :: v_dual_mov_b32 v10, v9
	s_barrier_wait -1
	s_barrier_wait -3
	s_cbranch_vccnz .LBB0_51
; %bb.25:
	v_mul_u32_u24_e32 v2, 0x60, v27
	v_and_b32_e32 v3, 16, v0
	v_dual_mov_b32 v21, 0 :: v_dual_bitop2_b32 v5, 15, v0 bitop3:0x40
	s_mov_b64 s[4:5], src_shared_base
	s_or_b32 s6, s37, 0x6600
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or_b32_e32 v2, v2, v3
	v_mul_u32_u24_e32 v9, 0x60, v5
	s_mov_b32 s7, s5
	s_mov_b32 s11, 0
	s_and_b64 s[6:7], s[6:7], 15
	v_mad_u32_u24 v2, 0x60, v5, v2
	v_and_b32_e32 v1, 32, v0
	s_sub_co_i32 s4, 16, s6
	v_dual_mov_b32 v14, v21 :: v_dual_mov_b32 v15, v21
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshrrev_b32_e32 v6, 4, v2
	v_mul_u32_u24_e32 v4, 0x60, v1
	s_lshr_b32 s4, s4, 2
	s_cmp_lg_u64 s[6:7], 0
	s_mul_hi_i32 s6, s12, 0x2aaaaaab
	v_and_b32_e32 v6, 0x3f8, v6
	s_cselect_b32 s4, s4, 0
	v_dual_mov_b32 v16, v21 :: v_dual_mov_b32 v17, v21
	s_lshl2_add_u32 s7, s4, s37
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_nc_u32 v18, v6, v2 :: v_dual_bitop2_b32 v7, v4, v3 bitop3:0x54
	v_or_b32_e32 v8, 32, v3
	s_add_co_i32 s4, s7, 0x9900
	s_add_co_i32 s44, s7, 0x6600
	s_and_b32 s10, s4, 15
	v_mad_u32_u24 v7, 0x60, v5, v7
	v_mad_u32_u24 v6, 0x60, v5, v8
	s_sub_co_i32 s8, 16, s10
	v_or_b32_e32 v3, v3, v9
	s_lshr_b32 s7, s8, 2
	v_lshrrev_b32_e32 v2, 4, v7
	v_mad_u32_u24 v6, 0x60, v27, v6
	v_add_nc_u32_e32 v9, 0x600, v7
	s_cmp_lg_u64 s[10:11], 0
	v_mad_u32_u24 v12, 0x60, v1, v3
	v_and_b32_e32 v2, 0x1f8, v2
	s_cselect_b32 s7, s7, 0
	v_lshrrev_b32_e32 v9, 4, v9
	s_lshr_b32 s8, s6, 31
	s_ashr_i32 s45, s6, 4
	v_dual_add_nc_u32 v22, v2, v7 :: v_dual_lshrrev_b32 v2, 4, v6
	v_dual_add_nc_u32 v8, 64, v3 :: v_dual_bitop2_b32 v4, v8, v4 bitop3:0x54
	v_and_b32_e32 v6, 0x1f8, v9
	s_lshl_b32 s10, s7, 2
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_and_b32_e32 v10, 0x3f8, v2
	v_mad_u32_u24 v4, 0x60, v5, v4
	v_mad_u32_u24 v5, 0x60, v1, v8
	v_mad_u32_u24 v8, 0x60, v27, v8
	s_add_co_i32 s45, s45, s8
	s_cmp_lt_i32 s3, 64
	v_add_nc_u32_e32 v9, 0x600, v4
	v_add_nc_u32_e32 v2, 0x600, v5
	v_dual_lshrrev_b32 v8, 4, v8 :: v_dual_lshrrev_b32 v5, 4, v5
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_dual_lshrrev_b32 v4, 4, v4 :: v_dual_lshrrev_b32 v9, 4, v9
	v_lshrrev_b32_e32 v24, 8, v19
	v_mad_u32_u24 v13, 0x60, v27, v3
	s_cselect_b32 s46, -1, 0
	v_and_b32_e32 v11, 0x1f8, v4
	v_and_b32_e32 v20, 0x3f8, v9
	v_and_b32_e32 v9, 0x1f8, v5
	v_dual_mov_b32 v5, v21 :: v_dual_lshrrev_b32 v2, 4, v2
	v_dual_add_nc_u32 v4, 2, v24 :: v_dual_add_nc_u32 v19, -1, v24
	v_mov_b32_e32 v3, v21
	s_lshl_b32 s6, s0, 6
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_and_b32_e32 v2, 0x3f8, v2
	v_dual_add_nc_u32 v26, v6, v7 :: v_dual_bitop2_b32 v29, 30, v4 bitop3:0x40
	v_add_nc_u32_e32 v4, 0x600, v12
	s_lshl_b32 s38, s38, 6
	v_and_b32_e32 v8, 0x3f8, v8
	s_ashr_i32 s7, s6, 31
	s_ashr_i32 s39, s38, 31
	s_wait_kmcnt 0x0
	s_bfe_i64 s[20:21], s[20:21], 0x200000
	s_bfe_i64 s[16:17], s[16:17], 0x200000
	s_mul_u64 s[6:7], s[20:21], s[6:7]
	s_mul_u64 s[16:17], s[16:17], s[38:39]
	v_add_nc_u64_e32 v[32:33], v[20:21], v[4:5]
	v_add_nc_u64_e32 v[38:39], v[2:3], v[4:5]
	v_cmp_eq_u32_e64 s0, 0, v23
	v_or_b32_e32 v1, 0x100, v0
	v_dual_add_nc_u32 v28, v10, v13 :: v_dual_add_nc_u32 v30, v11, v12
	v_dual_add_nc_u32 v34, v8, v13 :: v_dual_add_nc_u32 v36, v9, v12
	v_dual_mov_b32 v2, v21 :: v_dual_mov_b32 v4, v21
	v_dual_mov_b32 v6, v21 :: v_dual_mov_b32 v7, v21
	v_dual_mov_b32 v8, v21 :: v_dual_mov_b32 v9, v21
	v_dual_mov_b32 v10, v21 :: v_dual_mov_b32 v11, v21
	v_dual_mov_b32 v12, v21 :: v_dual_mov_b32 v13, v21
	s_lshr_b32 s47, s3, 16
	s_lshr_b32 s48, s31, 16
	s_lshl_b64 s[6:7], s[6:7], 1
	s_lshl_b64 s[16:17], s[16:17], 1
	s_mov_b32 s42, s35
	s_mov_b32 s43, s5
	s_add_nc_u64 s[36:37], s[4:5], s[10:11]
	s_mov_b32 s8, 64
	s_or_b32 s12, s1, 0x7510000
	s_or_b32 s47, s47, 0x600000
	s_or_b32 s4, s2, 0x7510000
	s_or_b32 s48, s48, 0x600000
	s_add_nc_u64 s[20:21], s[18:19], s[6:7]
	s_add_nc_u64 s[38:39], s[14:15], s[16:17]
	s_mov_b32 s49, s11
	s_branch .LBB0_27
.LBB0_26:                               ;   in Loop: Header=BB0_27 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	s_cmp_eq_u32 s49, s45
	s_barrier_wait -1
	s_barrier_wait -3
	s_cbranch_scc1 .LBB0_51
.LBB0_27:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_30 Depth 2
                                        ;     Child Loop BB0_36 Depth 2
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
	s_cbranch_vccnz .LBB0_41
; %bb.28:                               ;   in Loop: Header=BB0_27 Depth=1
	v_nop
	v_nop
	v_nop
	v_nop
	v_mov_b64_e32 v[40:41], v[0:1]
	s_cmp_lg_u32 s5, 0
	s_mov_b32 s10, 0
	s_cselect_b32 s7, s43, s35
	s_cselect_b32 s6, s44, 0
	s_mov_b32 s13, 0
	s_branch .LBB0_30
.LBB0_29:                               ;   in Loop: Header=BB0_30 Depth=2
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s1
	s_add_co_i32 s13, s13, 2
	v_add_nc_u32_e32 v41, 0x200, v41
	v_cmp_eq_u32_e32 vcc_lo, s13, v29
	v_add_nc_u32_e32 v40, 0x200, v40
	s_or_b32 s10, vcc_lo, s10
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s10
	s_cbranch_execz .LBB0_34
.LBB0_30:                               ;   Parent Loop BB0_27 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_mov_b32 s14, exec_lo
	v_cmp_le_u32_e32 vcc_lo, s13, v19
	v_cmpx_le_u32_e64 s13, v24
	s_cbranch_execz .LBB0_32
; %bb.31:                               ;   in Loop: Header=BB0_30 Depth=2
	v_mov_b32_e32 v20, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u64 v[42:43], v[20:21], 2, s[6:7]
	flat_store_b32 v[42:43], v21
.LBB0_32:                               ;   in Loop: Header=BB0_30 Depth=2
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s14
	s_and_saveexec_b32 s1, vcc_lo
	s_cbranch_execz .LBB0_29
; %bb.33:                               ;   in Loop: Header=BB0_30 Depth=2
	v_mov_b32_e32 v20, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u64 v[42:43], v[20:21], 2, s[6:7]
	flat_store_b32 v[42:43], v21
	s_branch .LBB0_29
.LBB0_34:                               ;   in Loop: Header=BB0_27 Depth=1
	s_or_b32 exec_lo, exec_lo, s10
	v_mov_b64_e32 v[40:41], v[0:1]
	s_cmp_lg_u32 s5, 0
	s_mov_b32 s10, 0
	s_cselect_b32 s7, s37, s42
	s_cselect_b32 s6, s36, s34
	s_mov_b32 s13, 0
	s_branch .LBB0_36
.LBB0_35:                               ;   in Loop: Header=BB0_36 Depth=2
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s1
	s_add_co_i32 s13, s13, 2
	v_add_nc_u32_e32 v41, 0x200, v41
	v_cmp_eq_u32_e32 vcc_lo, s13, v29
	v_add_nc_u32_e32 v40, 0x200, v40
	s_or_b32 s10, vcc_lo, s10
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s10
	s_cbranch_execz .LBB0_40
.LBB0_36:                               ;   Parent Loop BB0_27 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_mov_b32 s14, exec_lo
	v_cmp_le_u32_e32 vcc_lo, s13, v19
	v_cmpx_le_u32_e64 s13, v24
	s_cbranch_execz .LBB0_38
; %bb.37:                               ;   in Loop: Header=BB0_36 Depth=2
	v_mov_b32_e32 v20, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u64 v[42:43], v[20:21], 2, s[6:7]
	flat_store_b32 v[42:43], v21
.LBB0_38:                               ;   in Loop: Header=BB0_36 Depth=2
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s14
	s_and_saveexec_b32 s1, vcc_lo
	s_cbranch_execz .LBB0_35
; %bb.39:                               ;   in Loop: Header=BB0_36 Depth=2
	v_mov_b32_e32 v20, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u64 v[42:43], v[20:21], 2, s[6:7]
	flat_store_b32 v[42:43], v21
	s_branch .LBB0_35
.LBB0_40:                               ;   in Loop: Header=BB0_27 Depth=1
	s_or_b32 exec_lo, exec_lo, s10
	s_barrier_signal -1
	s_barrier_wait -1
.LBB0_41:                               ;   in Loop: Header=BB0_27 Depth=1
	s_and_b32 s1, s2, exec_lo
	s_cselect_b32 s1, s49, 0
	s_mov_b32 s2, exec_lo
	v_cmpx_lt_i32_e32 0, v23
	s_xor_b32 s6, exec_lo, s2
	s_cbranch_execnz .LBB0_47
; %bb.42:                               ;   in Loop: Header=BB0_27 Depth=1
	s_and_not1_saveexec_b32 s2, s6
	s_cbranch_execnz .LBB0_50
.LBB0_43:                               ;   in Loop: Header=BB0_27 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s41
	s_cbranch_vccnz .LBB0_45
.LBB0_44:                               ;   in Loop: Header=BB0_27 Depth=1
	s_cmp_lg_u32 s50, 0
	s_cselect_b32 s1, s44, 0
	s_cselect_b32 s2, s36, s34
	v_lshl_add_u32 v20, v18, 1, s1
	v_lshl_add_u32 v31, v22, 1, s2
	v_lshl_add_u32 v33, v26, 1, s2
	ds_load_b128 v[40:43], v20
	ds_load_b128 v[44:47], v20 offset:16
	ds_load_b128 v[48:51], v31
	ds_load_b128 v[52:55], v33 offset:3072
	ds_load_b128 v[56:59], v33 offset:3088
	v_lshl_add_u32 v20, v28, 1, s1
	v_lshl_add_u32 v33, v32, 1, s2
	; sched_group_barrier mask(0x00000100) size(3) SyncID(0)
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[10:17], v[40:47], v[52:59], v[10:17] matrix_a_reuse
	ds_load_b128 v[52:55], v31 offset:16
	v_lshl_add_u32 v31, v30, 1, s2
	ds_load_b128 v[56:59], v20 offset:64
	ds_load_b128 v[60:63], v20 offset:80
	ds_load_b128 v[72:75], v33 offset:64
	ds_load_b128 v[76:79], v33 offset:80
	ds_load_b128 v[64:67], v31 offset:64
	ds_load_b128 v[68:71], v31 offset:80
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[2:9], v[40:47], v[48:55], v[2:9]
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(3) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	v_lshl_add_u32 v20, v34, 1, s1
	v_lshl_add_u32 v31, v36, 1, s2
	v_lshl_add_u32 v33, v38, 1, s2
	ds_load_b128 v[40:43], v20 offset:128
	ds_load_b128 v[44:47], v20 offset:144
	ds_load_b128 v[48:51], v31 offset:128
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x32_bf16 v[2:9], v[56:63], v[64:71], v[2:9]
	ds_load_b128 v[52:55], v31 offset:144
	ds_load_b128 v[64:67], v33 offset:128
	ds_load_b128 v[68:71], v33 offset:144
	; sched_group_barrier mask(0x00000100) size(3) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(3) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[10:17], v[56:63], v[72:79], v[10:17] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[2:9], v[40:47], v[48:55], v[2:9]
	; sched_group_barrier mask(0x00000100) size(3) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(3) SyncID(0)
	s_wait_dscnt 0x0
	s_delay_alu instid0(TRANS32_DEP_2)
	v_wmma_f32_16x16x32_bf16 v[10:17], v[40:47], v[64:71], v[10:17] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_wait_dscnt 0
	;;#ASMEND
.LBB0_45:                               ;   in Loop: Header=BB0_27 Depth=1
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_26
; %bb.46:                               ;   in Loop: Header=BB0_27 Depth=1
	s_barrier_signal -3
	s_branch .LBB0_26
.LBB0_47:                               ;   in Loop: Header=BB0_27 Depth=1
	s_mov_b32 s7, exec_lo
	v_cmpx_eq_u32_e32 1, v23
	s_cbranch_execz .LBB0_49
; %bb.48:                               ;   in Loop: Header=BB0_27 Depth=1
	s_cmp_lg_u32 s5, 0
	s_mul_i32 s10, s1, 0x60
	s_cselect_b32 s2, s36, s34
	s_cmp_gt_i32 s30, 0
	s_mov_b32 s16, s8
	s_cselect_b32 s13, -1, 0
	s_lshl_b64 s[14:15], s[10:11], 1
	s_and_b32 s13, s25, s13
	s_add_nc_u64 s[14:15], s[20:21], s[14:15]
	v_cndmask_b32_e64 v20, 0, 1, s13
	s_and_b32 s10, s15, 0x1ffffff
	v_nop
	v_dual_mov_b32 v31, s2 :: v_dual_mov_b32 v40, s14
	s_bitset1_b32 s10, 31
	s_mov_b32 s2, s30
	v_mov_b32_e32 v33, s10
	v_readfirstlane_b32 s52, v20
	v_readfirstlane_b32 s53, v31
	v_readfirstlane_b32 s54, v40
	s_lshr_b64 s[14:15], s[2:3], 16
	v_readfirstlane_b32 s55, v33
	s_lshl_b32 s13, s30, 16
	s_mov_b32 s15, s47
	s_mov_b32 s17, s9
	s_mov_b32 s18, s11
	s_mov_b32 s19, s11
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[52:55], s[12:19]
.LBB0_49:                               ;   in Loop: Header=BB0_27 Depth=1
	s_or_b32 exec_lo, exec_lo, s7
	s_and_not1_saveexec_b32 s2, s6
	s_cbranch_execz .LBB0_43
.LBB0_50:                               ;   in Loop: Header=BB0_27 Depth=1
	s_cmp_lg_u32 s5, 0
	s_mul_i32 s10, s1, 0x60
	s_cselect_b32 s5, s44, 0
	s_cmp_gt_i32 s30, 0
	s_cselect_b32 s1, -1, 0
	s_lshl_b64 s[6:7], s[10:11], 1
	s_and_b32 s1, s29, s1
	s_add_nc_u64 s[6:7], s[38:39], s[6:7]
	v_cndmask_b32_e64 v20, 0, 1, s1
	s_and_b32 s7, s7, 0x1ffffff
	v_nop
	v_nop
	v_dual_mov_b32 v31, s5 :: v_dual_mov_b32 v40, s6
	s_bitset1_b32 s7, 31
	v_readfirstlane_b32 s16, v20
	v_mov_b32_e32 v33, s7
	s_delay_alu instid0(VALU_DEP_3)
	v_readfirstlane_b32 s17, v31
	v_readfirstlane_b32 s18, v40
	s_lshr_b64 s[6:7], s[30:31], 16
	s_lshl_b32 s5, s30, 16
	v_readfirstlane_b32 s19, v33
	s_mov_b32 s7, s48
	s_mov_b32 s10, s11
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	tensor_load_to_lds s[16:19], s[4:11]
	s_or_b32 exec_lo, exec_lo, s2
	s_and_not1_b32 vcc_lo, exec_lo, s41
	s_cbranch_vccz .LBB0_44
	s_branch .LBB0_45
.LBB0_51:
	s_wait_tensorcnt 0x0
	;;#ASMSTART
	s_wait_dscnt 0
	;;#ASMEND
	s_and_b32 vcc_lo, exec_lo, s41
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_vccz .LBB0_53
; %bb.52:
	v_dual_lshlrev_b32 v1, 6, v0 :: v_dual_lshrrev_b32 v18, 1, v0
	v_lshlrev_b32_e32 v20, 3, v0
	v_nop
	v_cvt_pk_bf16_f32 v9, v8, v9
	v_cvt_pk_bf16_f32 v8, v6, v7
	v_and_b32_e32 v1, 0xbc0, v1
	v_and_b32_e32 v18, 8, v18
	v_cvt_pk_bf16_f32 v6, v2, v3
	v_and_b32_e32 v2, 0x170, v20
	v_cvt_pk_bf16_f32 v7, v4, v5
	v_or_b32_e32 v19, 0x400, v1
	v_or3_b32 v1, v27, v18, v1
	v_cvt_pk_bf16_f32 v3, v12, v13
	v_cvt_pk_bf16_f32 v5, v16, v17
	v_cvt_pk_bf16_f32 v4, v14, v15
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_lshrrev_b32 v18, 3, v19 :: v_dual_lshlrev_b32 v1, 1, v1
	v_and_b32_e32 v18, 0x1f0, v18
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_nc_u32_e32 v12, v2, v1
	v_cvt_pk_bf16_f32 v2, v10, v11
	v_add_nc_u32_e32 v1, v18, v1
	ds_store_b128 v12, v[6:9]
	ds_store_b128 v1, v[2:5] offset:2048
.LBB0_53:
	v_cmp_ne_u32_e32 vcc_lo, 1, v25
	;;#ASMSTART
	s_wait_dscnt 0
	;;#ASMEND
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_vccnz .LBB0_64
; %bb.54:
	s_mul_i32 s3, s3, s31
	s_mov_b32 s0, exec_lo
	v_cmpx_gt_i32_e64 s3, v0
	s_cbranch_execz .LBB0_64
; %bb.55:
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
	s_cbranch_execnz .LBB0_58
; %bb.56:
	s_or_saveexec_b32 s1, s14
	v_mov_b64_e32 v[2:3], s[12:13]
	s_xor_b32 exec_lo, exec_lo, s1
	s_cbranch_execnz .LBB0_61
.LBB0_57:
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 exec_lo, exec_lo, s0
	s_cbranch_execnz .LBB0_62
	s_branch .LBB0_64
.LBB0_58:
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
.LBB0_59:                               ; =>This Inner Loop Header: Depth=1
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
	s_cbranch_execnz .LBB0_59
; %bb.60:
	s_or_b32 exec_lo, exec_lo, s23
	v_cmp_ne_u32_e32 vcc_lo, v8, v9
	v_lshl_or_b32 v0, v9, 8, v0
	v_dual_mov_b32 v6, s15 :: v_dual_mov_b32 v1, s22
	s_and_b32 s0, vcc_lo, exec_lo
	s_or_saveexec_b32 s1, s14
	v_mov_b64_e32 v[2:3], s[12:13]
	s_xor_b32 exec_lo, exec_lo, s1
	s_cbranch_execz .LBB0_57
.LBB0_61:
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
	s_cbranch_execz .LBB0_64
.LBB0_62:
	v_mov_b32_e32 v5, 0
	s_mov_b32 s0, 0
	s_sub_co_i32 s1, 0, s27
.LBB0_63:                               ; =>This Inner Loop Header: Depth=1
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
	s_cbranch_execnz .LBB0_63
.LBB0_64:
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end0:
	.size	bm064_bn064_bk096_wm4_wn2_mc1, .Lfunc_end0-bm064_bn064_bk096_wm4_wn2_mc1
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel bm064_bn064_bk096_wm4_wn2_mc1
		.amdhsa_group_segment_fixed_size 52224
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
		.amdhsa_next_free_vgpr 80
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
		.amdhsa_inst_pref_size 40
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text.bm064_bn064_bk096_wm4_wn2_mc1,"axG",@progbits,bm064_bn064_bk096_wm4_wn2_mc1,comdat
                                        ; -- End function
	.set .Lbm064_bn064_bk096_wm4_wn2_mc1.num_vgpr, 80
	.set .Lbm064_bn064_bk096_wm4_wn2_mc1.num_agpr, 0
	.set .Lbm064_bn064_bk096_wm4_wn2_mc1.numbered_sgpr, 56
	.set .Lbm064_bn064_bk096_wm4_wn2_mc1.num_named_barrier, 0
	.set .Lbm064_bn064_bk096_wm4_wn2_mc1.private_seg_size, 0
	.set .Lbm064_bn064_bk096_wm4_wn2_mc1.uses_vcc, 1
	.set .Lbm064_bn064_bk096_wm4_wn2_mc1.uses_flat_scratch, 0
	.set .Lbm064_bn064_bk096_wm4_wn2_mc1.has_dyn_sized_stack, 0
	.set .Lbm064_bn064_bk096_wm4_wn2_mc1.has_recursion, 0
	.set .Lbm064_bn064_bk096_wm4_wn2_mc1.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 5000
; TotalNumSgprs: 58
; NumVgprs: 80
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 52224 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 4
; NumSGPRsForWavesPerEU: 58
; NumVGPRsForWavesPerEU: 80
; NamedBarCnt: 0
; Occupancy: 12
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
	.type	__hip_cuid_eecf10a66da4e57a,@object ; @__hip_cuid_eecf10a66da4e57a
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_eecf10a66da4e57a
__hip_cuid_eecf10a66da4e57a:
	.byte	0                               ; 0x0
	.size	__hip_cuid_eecf10a66da4e57a, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git aa451e1fe6a793394d6733051b1778633063ae96+PATCHED:d17c5aa0e3ea29cde402f58f27e39b6034effa27)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_eecf10a66da4e57a
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
    macrotile: [64, 64, 96]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, One]
  MatrixInstruction: [16, 16, 32, 1]
  EnableMatrixInstruction: True
  MIWaveTile: [2, 1]
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
    .group_segment_fixed_size: 52224
    .kernarg_segment_align: 8
    .kernarg_segment_size: 132
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           bm064_bn064_bk096_wm4_wn2_mc1
    .private_segment_fixed_size: 0
    .sgpr_count:     58
    .sgpr_spill_count: 0
    .symbol:         bm064_bn064_bk096_wm4_wn2_mc1.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     80
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
