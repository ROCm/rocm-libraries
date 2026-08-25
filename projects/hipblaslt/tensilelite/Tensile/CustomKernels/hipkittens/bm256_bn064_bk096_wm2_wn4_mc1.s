	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.section	.text.bm256_bn064_bk096_wm2_wn4_mc1,"axG",@progbits,bm256_bn064_bk096_wm2_wn4_mc1,comdat
	.protected	bm256_bn064_bk096_wm2_wn4_mc1 ; -- Begin function bm256_bn064_bk096_wm2_wn4_mc1
	.globl	bm256_bn064_bk096_wm2_wn4_mc1
	.p2align	8
	.type	bm256_bn064_bk096_wm2_wn4_mc1,@function
bm256_bn064_bk096_wm2_wn4_mc1: ; @bm256_bn064_bk096_wm2_wn4_mc1
; %bb.0:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1 ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_load_b96 s[28:30], s[0:1], 0x78 nv
	s_mov_b64 s[2:3], src_shared_base
	s_mov_b32 s2, 0xcc00
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_and_b64 s[2:3], s[2:3], 12
	s_sub_co_i32 s4, 16, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_lshr_b32 s4, s4, 2
	s_cmp_lg_u64 s[2:3], 0
	s_cselect_b32 s5, s4, 0
	s_and_b32 s2, ttmp6, 15
	s_bfe_u32 s3, ttmp6, 0x40004
	s_lshl2_add_u32 s17, ttmp9, s2
	s_lshl2_add_u32 s4, ttmp7, s3
	s_lshl_b32 s34, s17, 8
	s_wait_kmcnt 0x0
	s_add_co_i32 s2, s28, 0xff
	s_add_co_i32 s3, s29, 63
	s_sub_co_i32 s6, s28, s34
	s_ashr_i32 s7, s2, 31
	s_ashr_i32 s8, s3, 31
	s_min_i32 s31, s6, 0x100
	s_lshr_b32 s6, s7, 24
	s_lshr_b32 s7, s8, 26
	s_add_co_i32 s2, s2, s6
	s_add_co_i32 s3, s3, s7
	s_ashr_i32 s6, s2, 8
	s_ashr_i32 s7, s3, 6
	s_cmp_lt_i32 s17, s6
	s_mov_b32 s9, s30
	s_cselect_b32 s35, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_and_b32 s2, s35, exec_lo
	s_cselect_b32 s37, s31, 0
	s_lshl_b32 s33, s4, 6
	s_sub_co_i32 s2, s29, s33
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_min_i32 s2, s2, 64
	s_cmp_lt_i32 s4, s7
	s_cselect_b32 s29, -1, 0
	s_and_b32 s3, s29, exec_lo
	s_cselect_b32 s3, s2, 0
	s_add_co_i32 s12, s30, 0x5f
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 2, 1), 1
	s_min_i32 s2, s30, 0x60
	s_cmp_gt_i32 s12, 0x5f
	s_cselect_b32 s13, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_and_b32 s8, s13, exec_lo
	s_cselect_b32 s36, s2, 0
	s_cmp_lt_i32 s37, 0x100
	s_cselect_b32 s42, -1, 0
	s_and_b32 vcc_lo, exec_lo, s42
	s_mov_b32 s2, s42
	s_cbranch_vccnz .LBB0_2
; %bb.1:
	s_cmp_lt_i32 s3, 64
	s_cselect_b32 s2, -1, 0
	s_cmp_lt_i32 s36, 0x60
	s_cselect_b32 s8, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 s2, s8, s2
.LBB0_2:
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB0_12
; %bb.3:
	v_dual_lshlrev_b32 v1, 2, v0 :: v_dual_mov_b32 v3, 0
	v_or_b32_e32 v2, 0xffffff00, v0
	s_mov_b32 s2, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_e32 v4, v1
.LBB0_4:                                ; =>This Inner Loop Header: Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)
	v_add_nc_u32_e32 v2, 0x100, v2
	ds_store_b32 v4, v3
	v_add_nc_u32_e32 v4, 0x400, v4
	v_cmp_lt_u32_e32 vcc_lo, 0x31ff, v2
	s_or_b32 s2, vcc_lo, s2
	s_and_not1_b32 exec_lo, exec_lo, s2
	s_cbranch_execnz .LBB0_4
; %bb.5:
	s_or_b32 exec_lo, exec_lo, s2
	v_dual_mov_b32 v5, 0 :: v_dual_sub_nc_u32 v2, 0xcbf, v0
	v_lshl_add_u32 v3, s5, 2, v1
	s_mov_b32 s8, 0
	s_mov_b32 s10, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshrrev_b32_e32 v2, 8, v2
	v_add_nc_u32_e32 v4, 2, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_nc_u32 v1, -1, v2 :: v_dual_bitop2_b32 v4, 30, v4 bitop3:0x40
	s_branch .LBB0_7
.LBB0_6:                                ;   in Loop: Header=BB0_7 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
	s_add_co_i32 s10, s10, 2
	v_add_nc_u32_e32 v3, 0x800, v3
	v_cmp_eq_u32_e32 vcc_lo, s10, v4
	s_or_b32 s8, vcc_lo, s8
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s8
	s_cbranch_execz .LBB0_11
.LBB0_7:                                ; =>This Inner Loop Header: Depth=1
	s_mov_b32 s11, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e32 vcc_lo, s10, v1
	v_cmpx_le_u32_e64 s10, v2
; %bb.8:                                ;   in Loop: Header=BB0_7 Depth=1
	ds_store_b32 v3, v5 offset:52224
; %bb.9:                                ;   in Loop: Header=BB0_7 Depth=1
	s_or_b32 exec_lo, exec_lo, s11
	s_and_saveexec_b32 s2, vcc_lo
	s_cbranch_execz .LBB0_6
; %bb.10:                               ;   in Loop: Header=BB0_7 Depth=1
	ds_store_b32 v3, v5 offset:53248
	s_branch .LBB0_6
.LBB0_11:
	s_or_b32 exec_lo, exec_lo, s8
	s_barrier_signal -1
	s_barrier_wait -1
.LBB0_12:
	s_clause 0x2
	s_load_b64 s[14:15], s[0:1], 0x0 nv
	s_load_b128 s[24:27], s[0:1], 0x20 nv
	s_load_b128 s[20:23], s[0:1], 0x48 nv
	s_wait_xcnt 0x0
	s_lshl_b32 s1, s4, 2
	v_lshrrev_b32_e32 v73, 5, v0
	s_lshl_b32 s16, s5, 2
	s_add_co_i32 s7, s7, -1
	s_and_b32 s1, s1, 12
	s_mov_b64 s[38:39], src_shared_base
	s_or_b32 s38, s16, 0xcc00
	s_add_co_i32 s18, s6, -1
	s_min_i32 s0, s4, s7
	s_and_b32 s19, s17, 3
	s_lshl_b32 s1, 15, s1
	s_mov_b32 s2, exec_lo
	v_cmpx_lt_i32_e32 0, v73
	s_xor_b32 s40, exec_lo, s2
	s_cbranch_execz .LBB0_16
; %bb.13:
	s_mov_b32 s41, exec_lo
	v_cmpx_eq_u32_e32 1, v73
	s_cbranch_execz .LBB0_15
; %bb.14:
	s_cmp_gt_i32 s36, 0
	s_mov_b32 s10, 0
	s_cselect_b32 s2, -1, 0
	s_lshl_b32 s4, s0, 6
	s_wait_kmcnt 0x0
	s_bfe_i64 s[6:7], s[20:21], 0x200000
	s_ashr_i32 s5, s4, 31
	s_and_b32 s2, s29, s2
	s_mul_u64 s[4:5], s[6:7], s[4:5]
	v_cndmask_b32_e64 v2, 0, 1, s2
	s_lshl_b64 s[4:5], s[4:5], 1
	s_mov_b32 s2, s36
	s_add_nc_u64 s[6:7], s[26:27], s[4:5]
	s_delay_alu instid0(SALU_CYCLE_1)
	v_dual_mov_b32 v1, s38 :: v_dual_mov_b32 v4, s6
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
	s_lshl_b32 s5, s36, 16
	s_or_b32 s7, s2, 0x600000
	s_mov_b32 s8, 64
	s_mov_b32 s11, s10
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[44:47], s[4:11]
.LBB0_15:
	s_or_b32 exec_lo, exec_lo, s41
.LBB0_16:
	s_or_saveexec_b32 s40, s40
	s_min_i32 s17, s17, s18
	s_lshl_b32 s2, 0x1111, s19
	s_xor_b32 exec_lo, exec_lo, s40
	s_cbranch_execz .LBB0_18
; %bb.17:
	s_cmp_gt_i32 s36, 0
	s_mov_b32 s10, 0
	s_cselect_b32 s8, -1, 0
	s_lshl_b32 s4, s17, 8
	s_wait_kmcnt 0x0
	s_bfe_i64 s[6:7], s[24:25], 0x200000
	s_ashr_i32 s5, s4, 31
	s_and_b32 s8, s35, s8
	s_mul_u64 s[4:5], s[6:7], s[4:5]
	v_cndmask_b32_e64 v2, 0, 1, s8
	s_lshl_b64 s[6:7], s[4:5], 1
	s_lshr_b32 s8, s37, 16
	s_add_nc_u64 s[6:7], s[14:15], s[6:7]
	s_or_b32 s4, s2, 0x7510000
	s_and_b32 s7, s7, 0x1ffffff
	v_readfirstlane_b32 s44, v2
	s_bitset1_b32 s7, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v4, s6 :: v_dual_mov_b32 v3, s7
	s_lshr_b64 s[6:7], s[36:37], 16
	s_lshl_b32 s5, s36, 16
	s_or_b32 s7, s8, 0x600000
	v_readfirstlane_b32 s46, v4
	v_readfirstlane_b32 s47, v3
	s_movk_i32 s8, 0x100
	s_mov_b32 s11, s10
	s_mov_b32 s45, s10
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[44:47], s[4:11]
.LBB0_18:
	s_or_b32 exec_lo, exec_lo, s40
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s4, exec_lo
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	v_cmpx_gt_u32_e32 32, v0
	s_cbranch_execz .LBB0_20
; %bb.19:
	s_barrier_signal -3
.LBB0_20:
	s_or_b32 exec_lo, exec_lo, s4
	v_dual_lshlrev_b32 v1, 4, v73 :: v_dual_mov_b32 v9, 0
	s_and_b32 s43, s35, s29
	v_and_b32_e32 v75, 0x80, v0
	v_cndmask_b32_e64 v71, 0, 1, s43
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mov_b32 v8, v9 :: v_dual_bitop2_b32 v77, 48, v1 bitop3:0x40
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
	v_dual_mov_b32 v49, v9 :: v_dual_mov_b32 v48, v9
	v_dual_mov_b32 v47, v9 :: v_dual_mov_b32 v46, v9
	v_dual_mov_b32 v45, v9 :: v_dual_mov_b32 v44, v9
	v_dual_mov_b32 v43, v9 :: v_dual_mov_b32 v42, v9
	v_dual_mov_b32 v57, v9 :: v_dual_mov_b32 v56, v9
	v_dual_mov_b32 v55, v9 :: v_dual_mov_b32 v54, v9
	v_dual_mov_b32 v53, v9 :: v_dual_mov_b32 v52, v9
	v_dual_mov_b32 v51, v9 :: v_dual_mov_b32 v50, v9
	v_dual_mov_b32 v65, v9 :: v_dual_mov_b32 v64, v9
	v_dual_mov_b32 v63, v9 :: v_dual_mov_b32 v62, v9
	v_dual_mov_b32 v61, v9 :: v_dual_mov_b32 v60, v9
	v_dual_mov_b32 v59, v9 :: v_dual_mov_b32 v58, v9
	v_dual_mov_b32 v41, v9 :: v_dual_mov_b32 v40, v9
	v_dual_mov_b32 v39, v9 :: v_dual_mov_b32 v38, v9
	v_dual_mov_b32 v37, v9 :: v_dual_mov_b32 v36, v9
	v_dual_mov_b32 v35, v9 :: v_dual_mov_b32 v34, v9
	s_and_not1_b32 vcc_lo, exec_lo, s13
	s_barrier_wait -1
	s_barrier_wait -3
	s_cbranch_vccnz .LBB0_46
; %bb.21:
	s_mov_b64 s[4:5], src_shared_base
	s_or_b32 s6, s16, 0xff00
	s_mov_b32 s7, s5
	v_mul_u32_u24_e32 v3, 0x60, v75
	s_and_b64 s[6:7], s[6:7], 15
	v_and_b32_e32 v5, 15, v0
	s_sub_co_i32 s4, 16, s6
	v_and_b32_e32 v15, 16, v0
	s_lshr_b32 s4, s4, 2
	s_cmp_lg_u64 s[6:7], 0
	s_mov_b32 s11, 0
	s_cselect_b32 s4, s4, 0
	v_or_b32_e32 v7, v3, v15
	s_lshl2_add_u32 s7, s4, s16
	v_mul_u32_u24_e32 v8, 0x60, v5
	s_add_co_i32 s4, s7, 0x1cb00
	s_add_co_i32 s45, s7, 0xff00
	s_and_b32 s10, s4, 15
	s_mul_hi_i32 s6, s12, 0x2aaaaaab
	s_sub_co_i32 s8, 16, s10
	v_or_b32_e32 v11, 0x1800, v8
	s_lshr_b32 s7, s8, 2
	s_cmp_lg_u64 s[10:11], 0
	v_or_b32_e32 v9, v7, v8
	s_cselect_b32 s7, s7, 0
	s_lshr_b32 s8, s6, 31
	s_ashr_i32 s47, s6, 4
	s_lshl_b32 s10, s7, 2
	s_add_co_i32 s47, s47, s8
	s_cmp_lt_i32 s3, 64
	v_lshrrev_b32_e32 v2, 4, v9
	s_cselect_b32 s48, -1, 0
	s_lshl_b32 s6, s0, 6
	s_movk_i32 s0, 0x600
	v_mul_u32_u24_e32 v1, 0x60, v77
	v_mad_u32_u24 v4, 0x60, v5, s0
	v_and_b32_e32 v2, 0x778, v2
	v_mov_b32_e32 v69, 0
	v_mad_u32_u24 v37, 0x60, v77, v8
	s_ashr_i32 s7, s6, 31
	v_or_b32_e32 v4, v3, v4
	s_wait_kmcnt 0x0
	s_bfe_i64 s[20:21], s[20:21], 0x200000
	v_mov_b32_e32 v29, v69
	s_lshl_b32 s18, s17, 8
	s_mul_u64 s[6:7], s[20:21], s[6:7]
	v_dual_lshrrev_b32 v4, 4, v4 :: v_dual_add_nc_u32 v10, v7, v11
	v_add_nc_u32_e32 v66, v2, v9
	v_add_nc_u32_e32 v2, 0xc00, v9
	v_or_b32_e32 v1, v1, v15
	v_add_nc_u32_e32 v6, 0x1200, v9
	v_and_b32_e32 v13, 0x7f8, v4
	v_dual_mov_b32 v31, v69 :: v_dual_add_nc_u32 v4, 0x2400, v9
	v_lshrrev_b32_e32 v2, 4, v2
	v_mad_u32_u24 v1, 0x60, v5, v1
	v_lshrrev_b32_e32 v10, 4, v10
	v_lshrrev_b32_e32 v6, 4, v6
	v_mad_u32_u24 v41, 0x60, v5, v7
	v_and_b32_e32 v17, 0xff8, v2
	v_lshrrev_b32_e32 v14, 4, v1
	v_and_b32_e32 v21, 0xff8, v10
	v_or_b32_e32 v10, 32, v15
	v_and_b32_e32 v19, 0xff8, v6
	v_add_nc_u32_e32 v2, 0x1e00, v9
	v_dual_mov_b32 v7, v69 :: v_dual_add_nc_u32 v6, 0x2a00, v9
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_lshrrev_b32 v4, 4, v4 :: v_dual_bitop2_b32 v12, v10, v3 bitop3:0x54
	v_lshrrev_b32_e32 v2, 4, v2
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_lshrrev_b32_e32 v6, 4, v6
	v_and_b32_e32 v14, 0x3f8, v14
	v_mad_u32_u24 v16, 0x60, v5, v12
	v_and_b32_e32 v25, 0xff8, v4
	v_and_b32_e32 v23, 0xff8, v2
	v_and_b32_e32 v27, 0xff8, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_dual_add_nc_u32 v70, v14, v1 :: v_dual_lshrrev_b32 v6, 4, v16
	v_dual_mov_b32 v33, v69 :: v_dual_add_nc_u32 v2, 0x600, v16
	v_add_nc_u32_e32 v1, 0xc00, v16
	v_add_nc_u32_e32 v4, 0x1200, v16
	v_dual_add_nc_u32 v12, v12, v11 :: v_dual_lshrrev_b32 v2, 4, v2
	v_dual_add_nc_u32 v76, v17, v9 :: v_dual_add_nc_u32 v78, v19, v9
	v_dual_add_nc_u32 v82, v23, v9 :: v_dual_add_nc_u32 v84, v25, v9
	v_dual_mov_b32 v9, v69 :: v_dual_add_nc_u32 v86, v27, v9
	v_dual_lshrrev_b32 v1, 4, v1 :: v_dual_lshrrev_b32 v4, 4, v4
	v_and_b32_e32 v36, 0xff8, v6
	v_and_b32_e32 v68, 0x1ff8, v2
	v_lshrrev_b32_e32 v6, 4, v12
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_and_b32_e32 v2, 0x1ff8, v1
	v_or_b32_e32 v1, 64, v15
	v_dual_mov_b32 v35, v69 :: v_dual_add_nc_u32 v12, 0x1e00, v16
	v_mov_b32_e32 v19, v69
	v_and_b32_e32 v6, 0xff8, v6
	v_dual_lshrrev_b32 v12, 4, v12 :: v_dual_bitop2_b32 v3, v3, v1 bitop3:0x54
	v_add_nc_u32_e32 v14, 0x2400, v16
	v_dual_mov_b32 v43, v69 :: v_dual_add_nc_u32 v32, 0x1800, v41
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_mad_u32_u24 v20, 0x60, v5, v3
	v_dual_add_nc_u32 v18, v37, v10 :: v_dual_add_nc_u32 v3, v3, v11
	v_add_nc_u32_e32 v16, 0x2a00, v16
	v_add_nc_u64_e32 v[96:97], v[6:7], v[32:33]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_nc_u32_e32 v22, 0x1200, v20
	v_dual_lshrrev_b32 v24, 4, v20 :: v_dual_lshrrev_b32 v18, 4, v18
	v_add_nc_u32_e32 v11, 0x1e00, v20
	v_dual_add_nc_u32 v1, v37, v1 :: v_dual_bitop2_b32 v6, v37, v15 bitop3:0x54
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_and_b32_e32 v39, 0xff8, v24
	v_and_b32_e32 v38, 0x3f8, v18
	v_lshrrev_b32_e32 v18, 4, v22
	v_add_nc_u32_e32 v22, 0x2400, v20
	v_dual_lshrrev_b32 v11, 4, v11 :: v_dual_add_nc_u32 v74, v13, v41
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_dual_add_nc_u32 v104, v38, v6 :: v_dual_add_nc_u32 v80, v21, v41
	v_dual_mov_b32 v13, v69 :: v_dual_add_nc_u32 v106, v39, v41
	v_dual_mov_b32 v21, v69 :: v_dual_lshrrev_b32 v24, 4, v22
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_and_b32_e32 v22, 0x1ff8, v11
	v_dual_mov_b32 v11, v69 :: v_dual_lshrrev_b32 v14, 4, v14
	v_lshrrev_b32_e32 v16, 4, v16
	v_and_b32_e32 v8, 0x1ff8, v12
	v_dual_lshrrev_b32 v3, 4, v3 :: v_dual_mov_b32 v15, v69
	v_and_b32_e32 v10, 0x1ff8, v14
	v_add_nc_u32_e32 v14, 0x600, v20
	v_and_b32_e32 v12, 0x1ff8, v16
	v_dual_mov_b32 v17, v69 :: v_dual_add_nc_u32 v16, 0xc00, v20
	v_dual_mov_b32 v23, v69 :: v_dual_add_nc_u32 v20, 0x2a00, v20
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_lshrrev_b32_e32 v14, 4, v14
	v_dual_mov_b32 v25, v69 :: v_dual_add_nc_u32 v28, 0x600, v41
	v_lshrrev_b32_e32 v26, 4, v20
	v_and_b32_e32 v20, 0xff8, v3
	v_sub_nc_u32_e32 v3, 0xcbf, v0
	v_lshrrev_b32_e32 v1, 4, v1
	v_and_b32_e32 v4, 0x1ff8, v4
	v_lshrrev_b32_e32 v16, 4, v16
	v_add_nc_u64_e32 v[88:89], v[68:69], v[28:29]
	v_lshrrev_b32_e32 v72, 8, v3
	v_dual_mov_b32 v27, v69 :: v_dual_add_nc_u32 v68, 0xc00, v41
	v_dual_mov_b32 v5, v69 :: v_dual_add_nc_u32 v30, 0x1200, v41
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mov_b32 v125, v69 :: v_dual_add_nc_u32 v3, 2, v72
	v_add_nc_u32_e32 v67, -1, v72
	v_and_b32_e32 v14, 0x1ff8, v14
	v_and_b32_e32 v16, 0x1ff8, v16
	v_and_b32_e32 v18, 0x1ff8, v18
	v_dual_add_nc_u32 v90, v36, v41 :: v_dual_bitop2_b32 v79, 30, v3 bitop3:0x40
	v_mov_b32_e32 v3, v69
	v_and_b32_e32 v24, 0x1ff8, v24
	v_and_b32_e32 v26, 0x1ff8, v26
	v_dual_mov_b32 v45, v69 :: v_dual_add_nc_u32 v34, 0x1e00, v41
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u64_e32 v[92:93], v[2:3], v[68:69]
	v_add_nc_u64_e32 v[94:95], v[4:5], v[30:31]
	v_dual_mov_b32 v47, v69 :: v_dual_add_nc_u32 v2, 0x2400, v41
	v_dual_mov_b32 v49, v69 :: v_dual_add_nc_u32 v4, 0x2a00, v41
	v_and_b32_e32 v40, 0x3f8, v1
	s_ashr_i32 s19, s18, 31
	s_lshl_b64 s[6:7], s[6:7], 1
	s_bfe_i64 s[24:25], s[24:25], 0x200000
	s_add_nc_u64 s[20:21], s[26:27], s[6:7]
	s_mul_u64 s[6:7], s[24:25], s[18:19]
	v_add_nc_u64_e32 v[98:99], v[8:9], v[34:35]
	v_add_nc_u64_e32 v[100:101], v[10:11], v[2:3]
	v_add_nc_u64_e32 v[102:103], v[12:13], v[4:5]
	v_add_nc_u64_e32 v[108:109], v[14:15], v[28:29]
	v_add_nc_u64_e32 v[110:111], v[16:17], v[68:69]
	v_add_nc_u64_e32 v[112:113], v[18:19], v[30:31]
	v_add_nc_u64_e32 v[114:115], v[20:21], v[32:33]
	v_add_nc_u64_e32 v[116:117], v[22:23], v[34:35]
	v_add_nc_u64_e32 v[118:119], v[24:25], v[2:3]
	v_add_nc_u64_e32 v[120:121], v[26:27], v[4:5]
	v_cmp_eq_u32_e64 s0, 0, v73
	v_or_b32_e32 v1, 0x100, v0
	v_dual_add_nc_u32 v122, v40, v6 :: v_dual_mov_b32 v8, v69
	v_or_b32_e32 v81, 0x3100, v0
	v_lshl_or_b32 v124, v0, 2, 0xc800
	v_dual_mov_b32 v2, v69 :: v_dual_mov_b32 v4, v69
	v_dual_mov_b32 v6, v69 :: v_dual_mov_b32 v10, v69
	v_dual_mov_b32 v12, v69 :: v_dual_mov_b32 v14, v69
	v_dual_mov_b32 v16, v69 :: v_dual_mov_b32 v18, v69
	v_dual_mov_b32 v20, v69 :: v_dual_mov_b32 v22, v69
	v_dual_mov_b32 v24, v69 :: v_dual_mov_b32 v26, v69
	v_dual_mov_b32 v28, v69 :: v_dual_mov_b32 v30, v69
	v_dual_mov_b32 v32, v69 :: v_dual_mov_b32 v42, v69
	v_dual_mov_b32 v44, v69 :: v_dual_mov_b32 v46, v69
	v_dual_mov_b32 v48, v69 :: v_dual_mov_b32 v50, v69
	v_dual_mov_b32 v51, v69 :: v_dual_mov_b32 v52, v69
	v_dual_mov_b32 v53, v69 :: v_dual_mov_b32 v54, v69
	v_dual_mov_b32 v55, v69 :: v_dual_mov_b32 v56, v69
	v_dual_mov_b32 v57, v69 :: v_dual_mov_b32 v58, v69
	v_dual_mov_b32 v59, v69 :: v_dual_mov_b32 v60, v69
	v_dual_mov_b32 v61, v69 :: v_dual_mov_b32 v62, v69
	v_dual_mov_b32 v63, v69 :: v_dual_mov_b32 v64, v69
	v_dual_mov_b32 v65, v69 :: v_dual_mov_b32 v34, v69
	v_dual_mov_b32 v36, v69 :: v_dual_mov_b32 v37, v69
	v_dual_mov_b32 v38, v69 :: v_dual_mov_b32 v39, v69
	v_dual_mov_b32 v40, v69 :: v_dual_mov_b32 v41, v69
	s_lshr_b32 s49, s3, 16
	s_lshr_b32 s50, s37, 16
	s_lshl_b64 s[6:7], s[6:7], 1
	s_mov_b32 s44, s39
	s_mov_b32 s46, s5
	s_add_nc_u64 s[40:41], s[4:5], s[10:11]
	s_mov_b32 s16, 64
	s_or_b32 s12, s1, 0x7510000
	s_or_b32 s49, s49, 0x600000
	s_or_b32 s4, s2, 0x7510000
	s_or_b32 s50, s50, 0x600000
	s_add_nc_u64 s[24:25], s[14:15], s[6:7]
	s_mov_b32 s26, -1
	s_movk_i32 s8, 0x100
	s_mov_b32 s27, s11
	s_branch .LBB0_23
.LBB0_22:                               ;   in Loop: Header=BB0_23 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	s_cmp_eq_u32 s27, s47
	s_barrier_wait -1
	s_barrier_wait -3
	s_cbranch_scc1 .LBB0_46
.LBB0_23:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_25 Depth 2
                                        ;     Child Loop BB0_28 Depth 2
                                        ;     Child Loop BB0_31 Depth 2
	s_and_b32 s51, s27, 1
	s_add_co_i32 s27, s27, 1
	s_xor_b32 s5, s51, 1
	s_mul_i32 s1, s27, 0xffffffa0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s1, s1, s30
	s_min_i32 s1, s1, 0x60
	s_cmp_lt_i32 s27, s47
	s_cselect_b32 s2, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_and_b32 s6, s2, exec_lo
	s_cselect_b32 s36, s1, 0
	s_cmp_lt_i32 s36, 0x60
	s_cselect_b32 s1, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b32 s1, s48, s1
	s_or_b32 s1, s42, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_cbranch_vccnz .LBB0_36
; %bb.24:                               ;   in Loop: Header=BB0_23 Depth=1
	v_nop
	v_nop
	v_mov_b64_e32 v[126:127], v[0:1]
	v_mov_b32_e32 v83, 50
	s_cmp_lg_u32 s5, 0
	s_mov_b32 s1, 0
	s_cselect_b32 s7, s46, s39
	s_cselect_b32 s6, s45, 0
.LBB0_25:                               ;   Parent Loop BB0_23 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_dual_mov_b32 v68, v126 :: v_dual_add_nc_u32 v83, -2, v83
	v_dual_mov_b32 v128, v127 :: v_dual_mov_b32 v129, v69
	v_add_nc_u32_e32 v127, 0x200, v127
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u64 v[130:131], v[68:69], 2, s[6:7]
	v_cmp_eq_u32_e32 vcc_lo, 0, v83
	v_add_nc_u32_e32 v126, 0x200, v126
	v_lshl_add_u64 v[128:129], v[128:129], 2, s[6:7]
	s_clause 0x1
	flat_store_b32 v[130:131], v69
	flat_store_b32 v[128:129], v69
	s_or_b32 s1, vcc_lo, s1
	s_wait_xcnt 0x0
	s_and_not1_b32 exec_lo, exec_lo, s1
	s_cbranch_execnz .LBB0_25
; %bb.26:                               ;   in Loop: Header=BB0_23 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	s_and_saveexec_b32 s1, s26
	s_cbranch_execz .LBB0_29
; %bb.27:                               ;   in Loop: Header=BB0_23 Depth=1
	v_add_nc_u64_e32 v[126:127], s[6:7], v[124:125]
	v_mov_b32_e32 v68, v81
	s_mov_b32 s6, 0
.LBB0_28:                               ;   Parent Loop BB0_23 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v68, 0x100, v68
	flat_store_b32 v[126:127], v69
	s_wait_xcnt 0x0
	v_add_nc_u64_e32 v[126:127], 0x400, v[126:127]
	v_cmp_lt_u32_e32 vcc_lo, 0x31ff, v68
	s_or_b32 s6, vcc_lo, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s6
	s_cbranch_execnz .LBB0_28
.LBB0_29:                               ;   in Loop: Header=BB0_23 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	v_mov_b64_e32 v[126:127], v[0:1]
	s_cmp_lg_u32 s5, 0
	s_mov_b32 s10, 0
	s_cselect_b32 s7, s41, s44
	s_cselect_b32 s6, s40, s38
	s_mov_b32 s13, 0
	s_branch .LBB0_31
.LBB0_30:                               ;   in Loop: Header=BB0_31 Depth=2
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s1
	s_add_co_i32 s13, s13, 2
	v_add_nc_u32_e32 v127, 0x200, v127
	v_cmp_eq_u32_e32 vcc_lo, s13, v79
	v_add_nc_u32_e32 v126, 0x200, v126
	s_or_b32 s10, vcc_lo, s10
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s10
	s_cbranch_execz .LBB0_35
.LBB0_31:                               ;   Parent Loop BB0_23 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_mov_b32 s14, exec_lo
	v_cmp_le_u32_e32 vcc_lo, s13, v67
	v_cmpx_le_u32_e64 s13, v72
	s_cbranch_execz .LBB0_33
; %bb.32:                               ;   in Loop: Header=BB0_31 Depth=2
	v_mov_b32_e32 v68, v126
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u64 v[128:129], v[68:69], 2, s[6:7]
	flat_store_b32 v[128:129], v69
.LBB0_33:                               ;   in Loop: Header=BB0_31 Depth=2
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s14
	s_and_saveexec_b32 s1, vcc_lo
	s_cbranch_execz .LBB0_30
; %bb.34:                               ;   in Loop: Header=BB0_31 Depth=2
	v_mov_b32_e32 v68, v127
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u64 v[128:129], v[68:69], 2, s[6:7]
	flat_store_b32 v[128:129], v69
	s_branch .LBB0_30
.LBB0_35:                               ;   in Loop: Header=BB0_23 Depth=1
	s_or_b32 exec_lo, exec_lo, s10
	s_barrier_signal -1
	s_barrier_wait -1
.LBB0_36:                               ;   in Loop: Header=BB0_23 Depth=1
	s_and_b32 s1, s2, exec_lo
	s_cselect_b32 s1, s27, 0
	s_mov_b32 s2, exec_lo
	v_cmpx_lt_i32_e32 0, v73
	s_xor_b32 s6, exec_lo, s2
	s_cbranch_execnz .LBB0_42
; %bb.37:                               ;   in Loop: Header=BB0_23 Depth=1
	s_and_not1_saveexec_b32 s2, s6
	s_cbranch_execnz .LBB0_45
.LBB0_38:                               ;   in Loop: Header=BB0_23 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s43
	s_cbranch_vccnz .LBB0_40
.LBB0_39:                               ;   in Loop: Header=BB0_23 Depth=1
	s_cmp_lg_u32 s51, 0
	s_cselect_b32 s1, s45, 0
	s_cselect_b32 s2, s40, s38
	v_lshl_add_u32 v68, v66, 1, s1
	v_lshl_add_u32 v83, v74, 1, s1
	v_lshl_add_u32 v85, v76, 1, s1
	v_lshl_add_u32 v87, v70, 1, s2
	v_lshl_add_u32 v89, v96, 1, s1
	ds_load_b128 v[126:129], v68
	ds_load_b128 v[130:133], v68 offset:16
	ds_load_b128 v[134:137], v83 offset:3072
	ds_load_b128 v[138:141], v83 offset:3088
	v_lshl_add_u32 v68, v78, 1, s1
	v_lshl_add_u32 v83, v80, 1, s1
	ds_load_b128 v[142:145], v85 offset:6144
	ds_load_b128 v[146:149], v85 offset:6160
	v_lshl_add_u32 v85, v86, 1, s1
	ds_load_b128 v[150:153], v68 offset:9216
	ds_load_b128 v[154:157], v68 offset:9232
	ds_load_b128 v[158:161], v83 offset:12288
	ds_load_b128 v[170:173], v87
	ds_load_b128 v[174:177], v87 offset:16
	v_lshl_add_u32 v68, v84, 1, s1
	v_lshl_add_u32 v87, v94, 1, s1
	v_lshl_add_u32 v91, v98, 1, s1
	v_lshl_add_u32 v93, v100, 1, s1
	v_lshl_add_u32 v95, v102, 1, s1
	v_lshl_add_u32 v97, v104, 1, s2
	ds_load_b128 v[186:189], v87 offset:64
	ds_load_b128 v[190:193], v87 offset:80
	ds_load_b128 v[194:197], v89 offset:64
	ds_load_b128 v[198:201], v89 offset:80
	ds_load_b128 v[202:205], v93 offset:64
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[58:65], v[126:133], v[170:177], v[58:65]
	ds_load_b128 v[206:209], v93 offset:80
	ds_load_b128 v[210:213], v95 offset:64
	ds_load_b128 v[214:217], v95 offset:80
	ds_load_b128 v[218:221], v97 offset:64
	ds_load_b128 v[222:225], v97 offset:80
	; sched_group_barrier mask(0x00000100) size(9) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[50:57], v[134:141], v[170:177], v[50:57] matrix_b_reuse
	ds_load_b128 v[134:137], v91 offset:64
	ds_load_b128 v[138:141], v91 offset:80
	v_wmma_f32_16x16x32_bf16 v[42:49], v[142:149], v[170:177], v[42:49] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[26:33], v[150:157], v[170:177], v[26:33] matrix_b_reuse
	ds_load_b128 v[162:165], v85 offset:21504
	ds_load_b128 v[166:169], v85 offset:21520
	v_lshl_add_u32 v85, v92, 1, s1
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	ds_load_b128 v[126:129], v85 offset:64
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x32_bf16 v[34:41], v[162:169], v[170:177], v[34:41] matrix_b_reuse
	ds_load_b128 v[162:165], v68 offset:18432
	ds_load_b128 v[166:169], v68 offset:18448
	v_lshl_add_u32 v68, v82, 1, s1
	ds_load_b128 v[130:133], v85 offset:80
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x32_bf16 v[2:9], v[162:169], v[170:177], v[2:9] matrix_b_reuse
	ds_load_b128 v[162:165], v68 offset:15360
	ds_load_b128 v[166:169], v68 offset:15376
	v_lshl_add_u32 v68, v90, 1, s1
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[10:17], v[162:169], v[170:177], v[10:17] matrix_b_reuse
	ds_load_b128 v[162:165], v83 offset:12304
	v_lshl_add_u32 v83, v88, 1, s1
	; sched_group_barrier mask(0x00000100) size(9) SyncID(0)
	ds_load_b128 v[178:181], v83 offset:64
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x32_bf16 v[18:25], v[158:165], v[170:177], v[18:25] matrix_b_reuse
	ds_load_b128 v[158:161], v68 offset:64
	ds_load_b128 v[162:165], v68 offset:80
	ds_load_b128 v[182:185], v83 offset:80
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_barrier mask(0x00000000)
	v_lshl_add_u32 v68, v106, 1, s1
	v_lshl_add_u32 v83, v108, 1, s1
	v_lshl_add_u32 v85, v110, 1, s1
	ds_load_b128 v[142:145], v68 offset:128
	ds_load_b128 v[146:149], v68 offset:144
	v_lshl_add_u32 v68, v112, 1, s1
	ds_load_b128 v[150:153], v83 offset:128
	ds_load_b128 v[154:157], v83 offset:144
	v_lshl_add_u32 v83, v114, 1, s1
	ds_load_b128 v[166:169], v85 offset:128
	ds_load_b128 v[226:229], v68 offset:128
	ds_load_b128 v[230:233], v68 offset:144
	v_lshl_add_u32 v68, v116, 1, s1
	ds_load_b128 v[170:173], v85 offset:144
	ds_load_b128 v[174:177], v83 offset:128
	s_wait_dscnt 0xa
	v_wmma_f32_16x16x32_bf16 v[58:65], v[158:165], v[218:225], v[58:65]
	v_lshl_add_u32 v85, v120, 1, s1
	; sched_group_barrier mask(0x00000100) size(9) SyncID(0)
	s_wait_dscnt 0x9
	v_wmma_f32_16x16x32_bf16 v[50:57], v[178:185], v[218:225], v[50:57] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[42:49], v[126:133], v[218:225], v[42:49] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[26:33], v[186:193], v[218:225], v[26:33] matrix_b_reuse
	ds_load_b128 v[178:181], v83 offset:144
	v_lshl_add_u32 v83, v118, 1, s1
	ds_load_b128 v[126:129], v68 offset:128
	ds_load_b128 v[130:133], v68 offset:144
	v_lshl_add_u32 v68, v122, 1, s2
	ds_load_b128 v[182:185], v85 offset:128
	ds_load_b128 v[158:161], v83 offset:128
	ds_load_b128 v[162:165], v83 offset:144
	ds_load_b128 v[186:189], v85 offset:144
	ds_load_b128 v[234:237], v68 offset:128
	ds_load_b128 v[238:241], v68 offset:144
	v_wmma_f32_16x16x32_bf16 v[18:25], v[194:201], v[218:225], v[18:25] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(9) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[10:17], v[134:141], v[218:225], v[10:17] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[2:9], v[202:209], v[218:225], v[2:9] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[34:41], v[210:217], v[218:225], v[34:41] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[58:65], v[142:149], v[234:241], v[58:65]
	; sched_group_barrier mask(0x00000100) size(9) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[50:57], v[150:157], v[234:241], v[50:57] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[42:49], v[166:173], v[234:241], v[42:49] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[26:33], v[226:233], v[234:241], v[26:33] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(9) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[18:25], v[174:181], v[234:241], v[18:25] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[10:17], v[126:133], v[234:241], v[10:17] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[2:9], v[158:165], v[234:241], v[2:9] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[34:41], v[182:189], v[234:241], v[34:41] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_wait_dscnt 0
	;;#ASMEND
.LBB0_40:                               ;   in Loop: Header=BB0_23 Depth=1
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_22
; %bb.41:                               ;   in Loop: Header=BB0_23 Depth=1
	s_barrier_signal -3
	s_branch .LBB0_22
.LBB0_42:                               ;   in Loop: Header=BB0_23 Depth=1
	s_mov_b32 s7, exec_lo
	v_cmpx_eq_u32_e32 1, v73
	s_cbranch_execz .LBB0_44
; %bb.43:                               ;   in Loop: Header=BB0_23 Depth=1
	s_cmp_lg_u32 s5, 0
	s_mul_i32 s10, s1, 0x60
	s_cselect_b32 s2, s40, s38
	s_cmp_gt_i32 s36, 0
	s_mov_b32 s17, s9
	s_cselect_b32 s13, -1, 0
	s_lshl_b64 s[14:15], s[10:11], 1
	s_and_b32 s13, s29, s13
	s_add_nc_u64 s[14:15], s[20:21], s[14:15]
	v_cndmask_b32_e64 v68, 0, 1, s13
	s_and_b32 s10, s15, 0x1ffffff
	v_dual_mov_b32 v83, s2 :: v_dual_mov_b32 v126, s14
	s_bitset1_b32 s10, 31
	s_mov_b32 s2, s36
	v_mov_b32_e32 v85, s10
	v_readfirstlane_b32 s52, v68
	v_readfirstlane_b32 s53, v83
	v_readfirstlane_b32 s54, v126
	s_lshr_b64 s[14:15], s[2:3], 16
	v_readfirstlane_b32 s55, v85
	s_lshl_b32 s13, s36, 16
	s_mov_b32 s15, s49
	s_mov_b32 s18, s11
	s_mov_b32 s19, s11
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[52:55], s[12:19]
.LBB0_44:                               ;   in Loop: Header=BB0_23 Depth=1
	s_or_b32 exec_lo, exec_lo, s7
	s_and_not1_saveexec_b32 s2, s6
	s_cbranch_execz .LBB0_38
.LBB0_45:                               ;   in Loop: Header=BB0_23 Depth=1
	s_cmp_lg_u32 s5, 0
	s_mul_i32 s10, s1, 0x60
	s_cselect_b32 s5, s45, 0
	s_cmp_gt_i32 s36, 0
	s_cselect_b32 s1, -1, 0
	s_lshl_b64 s[6:7], s[10:11], 1
	s_and_b32 s1, s35, s1
	s_add_nc_u64 s[6:7], s[24:25], s[6:7]
	v_cndmask_b32_e64 v68, 0, 1, s1
	s_and_b32 s7, s7, 0x1ffffff
	v_dual_mov_b32 v83, s5 :: v_dual_mov_b32 v126, s6
	s_bitset1_b32 s7, 31
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_readfirstlane_b32 s52, v68
	v_mov_b32_e32 v85, s7
	v_readfirstlane_b32 s53, v83
	v_readfirstlane_b32 s54, v126
	s_lshr_b64 s[6:7], s[36:37], 16
	s_lshl_b32 s5, s36, 16
	v_readfirstlane_b32 s55, v85
	s_mov_b32 s7, s50
	s_mov_b32 s10, s11
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	tensor_load_to_lds s[52:55], s[4:11]
	s_or_b32 exec_lo, exec_lo, s2
	s_and_not1_b32 vcc_lo, exec_lo, s43
	s_cbranch_vccz .LBB0_39
	s_branch .LBB0_40
.LBB0_46:
	s_wait_tensorcnt 0x0
	;;#ASMSTART
	s_wait_dscnt 0
	;;#ASMEND
	s_and_b32 vcc_lo, exec_lo, s43
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_vccz .LBB0_48
; %bb.47:
	v_lshrrev_b32_e32 v1, 1, v0
	v_and_or_b32 v66, v0, 15, v77
	v_cvt_pk_bf16_f32 v57, v56, v57
	v_cvt_pk_bf16_f32 v56, v54, v55
	v_cvt_pk_bf16_f32 v54, v50, v51
	v_and_or_b32 v1, v1, 8, v75
	v_cvt_pk_bf16_f32 v49, v48, v49
	v_cvt_pk_bf16_f32 v48, v46, v47
	v_cvt_pk_bf16_f32 v46, v42, v43
	v_cvt_pk_bf16_f32 v65, v64, v65
	v_lshl_or_b32 v1, v66, 8, v1
	v_cvt_pk_bf16_f32 v64, v62, v63
	v_cvt_pk_bf16_f32 v63, v60, v61
	v_cvt_pk_bf16_f32 v62, v58, v59
	v_cvt_pk_bf16_f32 v47, v44, v45
	v_lshrrev_b32_e32 v50, 3, v1
	v_cvt_pk_bf16_f32 v55, v52, v53
	v_cvt_pk_bf16_f32 v33, v32, v33
	v_cvt_pk_bf16_f32 v32, v30, v31
	v_cvt_pk_bf16_f32 v31, v28, v29
	v_and_b32_e32 v42, 0x7fe, v50
	v_cvt_pk_bf16_f32 v30, v26, v27
	v_cvt_pk_bf16_f32 v25, v24, v25
	v_cvt_pk_bf16_f32 v24, v22, v23
	v_cvt_pk_bf16_f32 v23, v20, v21
	v_lshl_add_u32 v1, v1, 1, v42
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
	v_cvt_pk_bf16_f32 v5, v40, v41
	v_cvt_pk_bf16_f32 v4, v38, v39
	v_cvt_pk_bf16_f32 v3, v36, v37
	v_cvt_pk_bf16_f32 v2, v34, v35
	ds_store_b128 v1, v[46:49] offset:64
	ds_store_b128 v1, v[30:33] offset:96
	ds_store_b128 v1, v[22:25] offset:128
	ds_store_b128 v1, v[14:17] offset:160
	ds_store_b128 v1, v[6:9] offset:192
	ds_store_b128 v1, v[2:5] offset:224
.LBB0_48:
	v_cmp_ne_u32_e32 vcc_lo, 1, v71
	;;#ASMSTART
	s_wait_dscnt 0
	;;#ASMEND
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_vccnz .LBB0_59
; %bb.49:
	s_mul_i32 s3, s3, s37
	s_mov_b32 s0, exec_lo
	v_cmpx_gt_i32_e64 s3, v0
	s_cbranch_execz .LBB0_59
; %bb.50:
	s_ashr_i32 s35, s34, 31
	v_nop
	v_xad_u32 v2, v0, -1, s3
	s_lshl_b64 s[0:1], s[34:35], 1
	s_ashr_i32 s29, s28, 31
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[4:5], s[22:23], s[0:1]
	s_mov_b32 s0, 0
                                        ; implicit-def: $vgpr1
                                        ; implicit-def: $vgpr6
                                        ; implicit-def: $sgpr12_sgpr13
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_u32_e32 0x2ff, v2
	s_xor_b32 s14, exec_lo, s1
	s_cbranch_execnz .LBB0_53
; %bb.51:
	s_or_saveexec_b32 s1, s14
	v_mov_b64_e32 v[2:3], s[12:13]
	s_xor_b32 exec_lo, exec_lo, s1
	s_cbranch_execnz .LBB0_56
.LBB0_52:
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 exec_lo, exec_lo, s0
	s_cbranch_execnz .LBB0_57
	s_branch .LBB0_59
.LBB0_53:
	s_abs_i32 s15, s31
	v_lshrrev_b32_e32 v1, 8, v2
	s_cvt_f32_u32 s0, s15
	v_or_b32_e32 v3, 0x300, v0
	s_sub_co_i32 s1, 0, s15
	v_mov_b32_e32 v7, 0
	v_rcp_iflag_f32_e32 v2, s0
	v_add_nc_u32_e32 v8, 1, v1
	v_or_b32_e32 v1, 0x100, v0
	s_mov_b32 s13, 0
	s_mov_b32 s6, s28
	s_mov_b32 s7, s29
	v_and_b32_e32 v9, 0x1fffffc, v8
	s_mov_b32 s8, s28
	v_readfirstlane_b32 s0, v2
	v_or_b32_e32 v2, 0x200, v0
	s_mov_b32 s9, s29
	v_mov_b32_e32 v10, v9
	s_mov_b32 s10, s28
	s_mul_f32 s0, s0, 0x4f7ffffe
	v_mov_b64_e32 v[4:5], v[2:3]
	v_mov_b64_e32 v[2:3], v[0:1]
	s_mov_b32 s11, s29
	s_cvt_u32_f32 s0, s0
	s_mov_b32 s16, s31
	s_mov_b32 s17, s31
	s_mov_b32 s18, s31
	s_mul_i32 s1, s1, s0
	s_mov_b32 s19, s33
	s_mul_hi_u32 s1, s0, s1
	s_mov_b32 s20, s33
	s_mov_b32 s21, s33
	s_ashr_i32 s22, s31, 31
	s_add_co_i32 s12, s0, s1
	s_mov_b32 s23, s13
.LBB0_54:                               ; =>This Inner Loop Header: Depth=1
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
	v_mul_lo_u32 v6, v1, s31
	v_dual_sub_nc_u32 v27, v19, v18 :: v_dual_add_nc_u32 v18, s33, v1
	v_sub_nc_u32_e32 v14, v4, v14
	v_ashrrev_i32_e32 v23, 31, v22
	v_lshl_add_u32 v11, v11, 8, v12
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_mul_lo_u32 v16, v27, s18
	v_dual_add_nc_u32 v24, s21, v27 :: v_dual_sub_nc_u32 v6, v2, v6
	v_ashrrev_i32_e32 v19, 31, v18
	v_lshl_add_u32 v26, v26, 8, v14
	v_mul_u64_e32 v[20:21], s[6:7], v[20:21]
	v_ashrrev_i32_e32 v25, 31, v24
	v_lshl_add_u32 v1, v1, 8, v6
	v_sub_nc_u32_e32 v16, v5, v16
	v_mul_u64_e32 v[18:19], s[28:29], v[18:19]
	v_mul_u64_e32 v[22:23], s[8:9], v[22:23]
	v_mul_u64_e32 v[24:25], s[10:11], v[24:25]
	v_ashrrev_i32_e32 v28, 31, v1
	v_lshl_add_u32 v27, v27, 8, v16
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
	s_cbranch_execnz .LBB0_54
; %bb.55:
	s_or_b32 exec_lo, exec_lo, s23
	v_cmp_ne_u32_e32 vcc_lo, v8, v9
	v_lshl_or_b32 v0, v9, 8, v0
	v_dual_mov_b32 v6, s15 :: v_dual_mov_b32 v1, s22
	s_and_b32 s0, vcc_lo, exec_lo
	s_or_saveexec_b32 s1, s14
	v_mov_b64_e32 v[2:3], s[12:13]
	s_xor_b32 exec_lo, exec_lo, s1
	s_cbranch_execz .LBB0_52
.LBB0_56:
	s_abs_i32 s2, s31
	s_ashr_i32 s8, s31, 31
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
	s_cbranch_execz .LBB0_59
.LBB0_57:
	v_mov_b32_e32 v5, 0
	s_mov_b32 s0, 0
	s_sub_co_i32 s1, 0, s31
.LBB0_58:                               ; =>This Inner Loop Header: Depth=1
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
	v_dual_sub_nc_u32 v7, v4, v9 :: v_dual_lshlrev_b32 v4, 8, v4
	v_lshlrev_b32_e32 v9, 8, v9
	v_mul_lo_u32 v8, v7, s31
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_sub_nc_u32 v4, v4, v8 :: v_dual_add_nc_u32 v8, s33, v7
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
	v_cmp_le_i32_e32 vcc_lo, s3, v0
	v_lshl_add_u64 v[8:9], v[8:9], 1, s[4:5]
	s_or_b32 s0, vcc_lo, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u64 v[8:9], v[4:5], 1, v[8:9]
	s_wait_dscnt 0x0
	global_store_b16 v[8:9], v10, off
	s_wait_xcnt 0x0
	s_and_not1_b32 exec_lo, exec_lo, s0
	s_cbranch_execnz .LBB0_58
.LBB0_59:
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end0:
	.size	bm256_bn064_bk096_wm2_wn4_mc1, .Lfunc_end0-bm256_bn064_bk096_wm2_wn4_mc1
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel bm256_bn064_bk096_wm2_wn4_mc1
		.amdhsa_group_segment_fixed_size 130560
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
		.amdhsa_next_free_vgpr 242
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
		.amdhsa_inst_pref_size 53
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text.bm256_bn064_bk096_wm2_wn4_mc1,"axG",@progbits,bm256_bn064_bk096_wm2_wn4_mc1,comdat
                                        ; -- End function
	.set .Lbm256_bn064_bk096_wm2_wn4_mc1.num_vgpr, 242
	.set .Lbm256_bn064_bk096_wm2_wn4_mc1.num_agpr, 0
	.set .Lbm256_bn064_bk096_wm2_wn4_mc1.numbered_sgpr, 56
	.set .Lbm256_bn064_bk096_wm2_wn4_mc1.num_named_barrier, 0
	.set .Lbm256_bn064_bk096_wm2_wn4_mc1.private_seg_size, 0
	.set .Lbm256_bn064_bk096_wm2_wn4_mc1.uses_vcc, 1
	.set .Lbm256_bn064_bk096_wm2_wn4_mc1.uses_flat_scratch, 1
	.set .Lbm256_bn064_bk096_wm2_wn4_mc1.has_dyn_sized_stack, 0
	.set .Lbm256_bn064_bk096_wm2_wn4_mc1.has_recursion, 0
	.set .Lbm256_bn064_bk096_wm2_wn4_mc1.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 6752
; TotalNumSgprs: 58
; NumVgprs: 242
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 130560 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 15
; NumSGPRsForWavesPerEU: 58
; NumVGPRsForWavesPerEU: 242
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
	.type	__hip_cuid_bd32a85d5ea0ef63,@object ; @__hip_cuid_bd32a85d5ea0ef63
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_bd32a85d5ea0ef63
__hip_cuid_bd32a85d5ea0ef63:
	.byte	0                               ; 0x0
	.size	__hip_cuid_bd32a85d5ea0ef63, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git aa451e1fe6a793394d6733051b1778633063ae96+PATCHED:d17c5aa0e3ea29cde402f58f27e39b6034effa27)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_bd32a85d5ea0ef63
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
    macrotile: [256, 64, 96]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, One]
  MatrixInstruction: [16, 16, 32, 1]
  EnableMatrixInstruction: True
  MIWaveTile: [8, 1]
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
    .group_segment_fixed_size: 130560
    .kernarg_segment_align: 8
    .kernarg_segment_size: 132
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           bm256_bn064_bk096_wm2_wn4_mc1
    .private_segment_fixed_size: 0
    .sgpr_count:     58
    .sgpr_spill_count: 0
    .symbol:         bm256_bn064_bk096_wm2_wn4_mc1.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     242
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
