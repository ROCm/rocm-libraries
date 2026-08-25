	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.section	.text.bm128_bn096_bk096_wm8_wn1_mc1,"axG",@progbits,bm128_bn096_bk096_wm8_wn1_mc1,comdat
	.protected	bm128_bn096_bk096_wm8_wn1_mc1 ; -- Begin function bm128_bn096_bk096_wm8_wn1_mc1
	.globl	bm128_bn096_bk096_wm8_wn1_mc1
	.p2align	8
	.type	bm128_bn096_bk096_wm8_wn1_mc1,@function
bm128_bn096_bk096_wm8_wn1_mc1: ; @bm128_bn096_bk096_wm8_wn1_mc1
; %bb.0:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1 ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_load_b96 s[28:30], s[0:1], 0x78 nv
	s_mov_b64 s[2:3], src_shared_base
	s_movk_i32 s2, 0x6600
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
	s_lshl_b32 s34, s16, 7
	s_wait_kmcnt 0x0
	s_add_co_i32 s2, s28, 0x7f
	s_sub_co_i32 s5, s28, s34
	s_ashr_i32 s6, s2, 31
	s_add_co_i32 s3, s29, 0x5f
	s_min_i32 s31, s5, 0x80
	s_lshr_b32 s5, s6, 25
	s_mul_hi_i32 s3, s3, 0x2aaaaaab
	s_add_co_i32 s2, s2, s5
	s_lshr_b32 s6, s3, 31
	s_ashr_i32 s3, s3, 4
	s_ashr_i32 s5, s2, 7
	s_add_co_i32 s6, s3, s6
	s_cmp_lt_i32 s16, s5
	s_mul_i32 s2, s33, 0xffffffa0
	s_cselect_b32 s35, -1, 0
	s_mov_b32 s9, s30
	s_and_b32 s3, s35, exec_lo
	s_cselect_b32 s37, s31, 0
	s_add_co_i32 s2, s29, s2
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_min_i32 s2, s2, 0x60
	s_cmp_lt_i32 s33, s6
	s_cselect_b32 s29, -1, 0
	s_and_b32 s3, s29, exec_lo
	s_cselect_b32 s39, s2, 0
	s_add_co_i32 s12, s30, 0x5f
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 2, 1), 1
	s_min_i32 s2, s30, 0x60
	s_cmp_gt_i32 s12, 0x5f
	s_cselect_b32 s13, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_and_b32 s3, s13, exec_lo
	s_cselect_b32 s36, s2, 0
	s_cmp_lt_i32 s37, 0x80
	s_cselect_b32 s42, -1, 0
	s_and_b32 vcc_lo, exec_lo, s42
	s_mov_b32 s2, s42
	s_cbranch_vccnz .LBB0_2
; %bb.1:
	s_min_i32 s2, s36, s39
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lt_i32 s2, 0x60
	s_cselect_b32 s2, -1, 0
.LBB0_2:
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB0_8
; %bb.3:
	v_dual_mov_b32 v3, 0 :: v_dual_lshlrev_b32 v2, 2, v0
	v_or_b32_e32 v1, 0xffffff00, v0
	s_mov_b32 s2, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v4, v2 :: v_dual_mov_b32 v5, v1
.LBB0_4:                                ; =>This Inner Loop Header: Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)
	v_add_nc_u32_e32 v5, 0x100, v5
	ds_store_b32 v4, v3
	v_add_nc_u32_e32 v4, 0x400, v4
	v_cmp_lt_u32_e32 vcc_lo, 0x187f, v5
	s_or_b32 s2, vcc_lo, s2
	s_and_not1_b32 exec_lo, exec_lo, s2
	s_cbranch_execnz .LBB0_4
; %bb.5:
	s_or_b32 exec_lo, exec_lo, s2
	v_lshl_add_u32 v2, s4, 2, v2
	v_mov_b32_e32 v3, 0
	s_mov_b32 s2, 0
.LBB0_6:                                ; =>This Inner Loop Header: Depth=1
	v_add_nc_u32_e32 v1, 0x100, v1
	ds_store_b32 v2, v3 offset:26112
	v_add_nc_u32_e32 v2, 0x400, v2
	v_cmp_lt_u32_e32 vcc_lo, 0x121f, v1
	s_or_b32 s2, vcc_lo, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s2
	s_cbranch_execnz .LBB0_6
; %bb.7:
	s_or_b32 exec_lo, exec_lo, s2
	s_barrier_signal -1
	s_barrier_wait -1
.LBB0_8:
	s_clause 0x2
	s_load_b64 s[14:15], s[0:1], 0x0 nv
	s_load_b128 s[24:27], s[0:1], 0x20 nv
	s_load_b128 s[20:23], s[0:1], 0x48 nv
	s_mov_b64 s[2:3], src_shared_base
	s_lshl_b32 s2, s33, 2
	v_lshrrev_b32_e32 v51, 5, v0
	s_wait_xcnt 0x0
	s_lshl_b32 s1, s4, 2
	s_add_co_i32 s6, s6, -1
	s_and_b32 s2, s2, 12
	s_or_b32 s43, s1, 0x6600
	s_add_co_i32 s17, s5, -1
	s_min_i32 s0, s33, s6
	s_and_b32 s19, s16, 3
	s_lshl_b32 s2, 15, s2
	s_mov_b32 s4, exec_lo
	v_cmpx_lt_i32_e32 0, v51
	s_xor_b32 s18, exec_lo, s4
	s_cbranch_execz .LBB0_12
; %bb.9:
	s_mov_b32 s40, exec_lo
	v_cmpx_eq_u32_e32 1, v51
	s_cbranch_execz .LBB0_11
; %bb.10:
	s_cmp_gt_i32 s36, 0
	s_mul_i32 s4, s0, 0x60
	s_cselect_b32 s8, -1, 0
	s_ashr_i32 s5, s4, 31
	s_wait_kmcnt 0x0
	s_bfe_i64 s[6:7], s[20:21], 0x200000
	s_mov_b32 s38, s36
	s_mul_u64 s[4:5], s[6:7], s[4:5]
	s_and_b32 s6, s29, s8
	s_lshl_b64 s[4:5], s[4:5], 1
	v_cndmask_b32_e64 v2, 0, 1, s6
	s_add_nc_u64 s[4:5], s[26:27], s[4:5]
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_mov_b32 v1, s43 :: v_dual_mov_b32 v4, s4
	s_and_b32 s5, s5, 0x1ffffff
	v_readfirstlane_b32 s44, v2
	s_bitset1_b32 s5, 31
	v_readfirstlane_b32 s45, v1
	v_mov_b32_e32 v3, s5
	v_readfirstlane_b32 s46, v4
	s_mov_b32 s10, 0
	s_lshr_b32 s11, s39, 16
	s_lshr_b64 s[6:7], s[38:39], 16
	v_readfirstlane_b32 s47, v3
	s_movk_i32 s8, 0x60
	s_or_b32 s4, s2, 0x7510000
	s_lshl_b32 s5, s36, 16
	s_or_b32 s7, s11, 0x600000
	s_mov_b32 s11, s10
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[44:47], s[4:11]
.LBB0_11:
	s_or_b32 exec_lo, exec_lo, s40
.LBB0_12:
	s_or_saveexec_b32 s38, s18
	s_min_i32 s18, s16, s17
	s_lshl_b32 s17, 0x1111, s19
	s_xor_b32 exec_lo, exec_lo, s38
	s_cbranch_execz .LBB0_14
; %bb.13:
	s_cmp_gt_i32 s36, 0
	s_mov_b32 s10, 0
	s_cselect_b32 s8, -1, 0
	s_lshl_b32 s4, s18, 7
	s_wait_kmcnt 0x0
	s_bfe_i64 s[6:7], s[24:25], 0x200000
	s_ashr_i32 s5, s4, 31
	s_and_b32 s8, s35, s8
	s_mul_u64 s[4:5], s[6:7], s[4:5]
	v_cndmask_b32_e64 v2, 0, 1, s8
	s_lshl_b64 s[6:7], s[4:5], 1
	s_lshr_b32 s8, s37, 16
	s_add_nc_u64 s[6:7], s[14:15], s[6:7]
	s_or_b32 s4, s17, 0x7510000
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
	s_movk_i32 s8, 0x80
	s_mov_b32 s11, s10
	s_mov_b32 s45, s10
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[44:47], s[4:11]
.LBB0_14:
	s_or_b32 exec_lo, exec_lo, s38
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
	v_mov_b32_e32 v9, 0
	s_and_b32 s44, s35, s29
	s_and_not1_b32 vcc_lo, exec_lo, s13
	v_cndmask_b32_e64 v55, 0, 1, s44
	s_barrier_wait -1
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
	v_dual_mov_b32 v18, v9 :: v_dual_mov_b32 v41, v9
	v_dual_mov_b32 v40, v9 :: v_dual_mov_b32 v39, v9
	v_dual_mov_b32 v38, v9 :: v_dual_mov_b32 v37, v9
	v_dual_mov_b32 v36, v9 :: v_dual_mov_b32 v35, v9
	v_dual_mov_b32 v34, v9 :: v_dual_mov_b32 v49, v9
	v_dual_mov_b32 v48, v9 :: v_dual_mov_b32 v47, v9
	v_dual_mov_b32 v46, v9 :: v_dual_mov_b32 v45, v9
	v_dual_mov_b32 v44, v9 :: v_dual_mov_b32 v43, v9
	v_dual_mov_b32 v42, v9 :: v_dual_mov_b32 v33, v9
	v_dual_mov_b32 v32, v9 :: v_dual_mov_b32 v31, v9
	v_dual_mov_b32 v30, v9 :: v_dual_mov_b32 v29, v9
	v_dual_mov_b32 v28, v9 :: v_dual_mov_b32 v27, v9
	v_mov_b32_e32 v26, v9
	s_barrier_wait -3
	s_cbranch_vccnz .LBB0_41
; %bb.17:
	s_mov_b64 s[4:5], src_shared_base
	v_mul_u32_u24_e32 v1, 0x600, v51
	v_and_b32_e32 v3, 16, v0
	s_or_b32 s6, s1, 0xb280
	s_mov_b32 s7, s5
	v_and_b32_e32 v2, 15, v0
	s_and_b64 s[6:7], s[6:7], 15
	v_dual_mov_b32 v53, 0 :: v_dual_bitop2_b32 v1, v1, v3 bitop3:0x54
	s_sub_co_i32 s4, 16, s6
	s_mov_b32 s11, 0
	s_lshr_b32 s4, s4, 2
	s_cmp_lg_u64 s[6:7], 0
	v_mad_u32_u24 v1, 0x60, v2, v1
	s_cselect_b32 s4, s4, 0
	s_mul_hi_i32 s6, s12, 0x2aaaaaab
	s_lshl2_add_u32 s1, s4, s1
	s_movk_i32 s16, 0x60
	s_add_co_i32 s4, s1, 0x11880
	v_lshrrev_b32_e32 v4, 4, v1
	s_and_b32 s10, s4, 15
	s_add_co_i32 s47, s1, 0xb280
	s_sub_co_i32 s7, 16, s10
	v_mul_u32_u24_e32 v5, 0x60, v2
	s_lshr_b32 s1, s7, 2
	v_and_b32_e32 v4, 0x7f8, v4
	s_cmp_lg_u64 s[10:11], 0
	v_sub_nc_u32_e32 v22, 0x131f, v0
	s_cselect_b32 s1, s1, 0
	s_lshr_b32 s7, s6, 31
	s_ashr_i32 s48, s6, 4
	s_lshl_b32 s10, s1, 2
	s_add_co_i32 s48, s48, s7
	v_lshrrev_b32_e32 v6, 4, v5
	v_dual_add_nc_u32 v50, v4, v1 :: v_dual_bitop2_b32 v4, v5, v3 bitop3:0x54
	v_mad_u32_u24 v1, v2, s16, 0x600
	s_cmp_lt_i32 s39, 0x60
	s_delay_alu instid0(VALU_DEP_3)
	v_and_b32_e32 v6, 0x78, v6
	s_cselect_b32 s49, -1, 0
	s_or_b32 s12, s2, 0x7510000
	s_movk_i32 s2, 0xc00
	v_dual_lshrrev_b32 v1, 4, v1 :: v_dual_bitop2_b32 v8, 32, v3 bitop3:0x54
	v_mad_u32_u24 v7, 0x60, v2, s2
	v_add_nc_u32_e32 v54, v4, v6
	v_or_b32_e32 v5, 0x1800, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v6, 0xf8, v1
	v_or_b32_e32 v3, 64, v3
	v_lshrrev_b32_e32 v1, 4, v7
	v_mad_u32_u24 v7, 0x60, v2, v8
	s_movk_i32 s2, 0x1200
	v_mov_b32_e32 v95, v53
	v_mad_u32_u24 v9, 0x60, v2, s2
	v_and_b32_e32 v12, 0x1f8, v1
	v_mad_u32_u24 v11, 0x600, v51, v7
	v_lshrrev_b32_e32 v1, 4, v5
	v_add_nc_u32_e32 v13, 0x600, v7
	v_lshrrev_b32_e32 v14, 4, v7
	s_movk_i32 s2, 0x1e00
	v_lshrrev_b32_e32 v11, 4, v11
	v_and_b32_e32 v15, 0x1f8, v1
	v_lshrrev_b32_e32 v1, 4, v13
	v_mad_u32_u24 v10, 0x60, v2, s2
	v_and_b32_e32 v13, 0xf8, v14
	v_add_nc_u32_e32 v14, 0xc00, v7
	v_dual_mov_b32 v97, v53 :: v_dual_add_nc_u32 v16, 0x1200, v7
	v_and_b32_e32 v17, 0x1f8, v1
	v_add_nc_u32_e32 v1, v5, v8
	v_mad_u32_u24 v2, 0x60, v2, v3
	v_dual_add_nc_u32 v3, v5, v3 :: v_dual_lshrrev_b32 v8, 4, v14
	v_lshrrev_b32_e32 v14, 4, v16
	v_add_nc_u32_e32 v7, 0x1e00, v7
	v_lshrrev_b32_e32 v1, 4, v1
	v_mad_u32_u24 v16, 0x600, v51, v2
	v_dual_lshrrev_b32 v3, 4, v3 :: v_dual_lshrrev_b32 v9, 4, v9
	v_lshrrev_b32_e32 v10, 4, v10
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_and_b32_e32 v18, 0x3f8, v1
	v_dual_lshrrev_b32 v1, 4, v16 :: v_dual_lshrrev_b32 v19, 4, v2
	v_add_nc_u32_e32 v16, 0x600, v2
	v_and_b32_e32 v3, 0x3f8, v3
	v_and_b32_e32 v9, 0x1f8, v9
	v_and_b32_e32 v10, 0x3f8, v10
	v_and_b32_e32 v11, 0x7f8, v11
	v_and_b32_e32 v21, 0x7f8, v1
	v_dual_add_nc_u32 v56, v4, v6 :: v_dual_add_nc_u32 v58, v4, v12
	v_mad_u32_u24 v6, 0x600, v51, v4
	v_add3_u32 v90, v4, v3, 64
	v_dual_mov_b32 v3, v53 :: v_dual_lshrrev_b32 v1, 4, v16
	v_and_b32_e32 v16, 0xf8, v19
	v_and_b32_e32 v8, 0x1f8, v8
	v_add_nc_u32_e32 v20, 0xc00, v2
	v_dual_add_nc_u32 v60, v4, v9 :: v_dual_add_nc_u32 v62, v4, v15
	v_and_b32_e32 v5, 0x1f8, v1
	v_dual_add_nc_u32 v64, v4, v10 :: v_dual_add_nc_u32 v68, v13, v4
	v_add_nc_u32_e32 v9, 0x600, v4
	v_add_nc_u32_e32 v66, v11, v6
	v_add_nc_u32_e32 v10, 0xc00, v4
	v_add_nc_u32_e32 v11, 0x1200, v4
	v_dual_mov_b32 v13, v53 :: v_dual_add_nc_u32 v12, 0x1e00, v4
	v_add3_u32 v76, v4, v18, 32
	v_dual_add_nc_u32 v80, v21, v6 :: v_dual_add_nc_u32 v82, v16, v4
	v_dual_mov_b32 v4, v53 :: v_dual_lshrrev_b32 v7, 4, v7
	v_dual_mov_b32 v6, v53 :: v_dual_add_nc_u32 v19, 0x1200, v2
	v_add_nc_u32_e32 v1, 0x1e00, v2
	v_add_nc_u32_e32 v84, v5, v9
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_and_b32_e32 v7, 0x3f8, v7
	v_dual_mov_b32 v5, v53 :: v_dual_add_nc_u32 v72, v8, v10
	v_dual_mov_b32 v8, v53 :: v_dual_lshrrev_b32 v20, 4, v20
	v_dual_lshrrev_b32 v1, 4, v1 :: v_dual_lshrrev_b32 v2, 4, v19
	v_dual_mov_b32 v7, v53 :: v_dual_add_nc_u32 v78, v7, v12
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_and_b32_e32 v19, 0x1f8, v20
	v_sub_nc_u32_e32 v20, 0x197f, v0
	v_and_b32_e32 v23, 0x3f8, v1
	v_and_b32_e32 v14, 0x3f8, v14
	v_and_b32_e32 v2, 0x3f8, v2
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_add_nc_u32 v86, v19, v10 :: v_dual_lshrrev_b32 v20, 8, v20
	v_mov_b32_e32 v10, v53
	v_add_nc_u32_e32 v70, v17, v9
	v_dual_mov_b32 v9, v53 :: v_dual_lshrrev_b32 v1, 8, v22
	v_add_nc_u32_e32 v20, 1, v20
	v_dual_add_nc_u32 v92, v23, v12 :: v_dual_mov_b32 v12, v53
	s_mulk_i32 s0, 0x60
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_dual_add_nc_u32 v22, 1, v1 :: v_dual_bitop2_b32 v57, 26, v20 bitop3:0x40
	v_add_nc_u32_e32 v74, v14, v11
	s_ashr_i32 s1, s0, 31
	s_lshl_b32 s6, s18, 7
	v_dual_add_nc_u32 v88, v2, v11 :: v_dual_bitop2_b32 v59, 22, v22 bitop3:0x40
	s_wait_kmcnt 0x0
	s_bfe_i64 s[18:19], s[20:21], 0x200000
	v_lshl_or_b32 v24, v57, 8, v0
	s_mul_u64 s[18:19], s[18:19], s[0:1]
	v_cmp_ne_u32_e64 s1, v20, v57
	v_lshl_or_b32 v20, v59, 8, v0
	s_ashr_i32 s7, s6, 31
	s_bfe_i64 s[24:25], s[24:25], 0x200000
	v_cmp_eq_u32_e64 s0, 0, v51
	s_mul_u64 s[6:7], s[24:25], s[6:7]
	v_or_b32_e32 v1, 0x100, v0
	v_cmp_ne_u32_e64 s2, v22, v59
	v_dual_mov_b32 v16, v53 :: v_dual_add_nc_u32 v61, 0xffffff00, v24
	v_dual_mov_b32 v15, v53 :: v_dual_lshlrev_b32 v94, 2, v24
	v_dual_mov_b32 v18, v53 :: v_dual_add_nc_u32 v63, 0xffffff00, v20
	v_dual_mov_b32 v17, v53 :: v_dual_lshlrev_b32 v96, 2, v20
	v_dual_mov_b32 v2, v53 :: v_dual_mov_b32 v11, v53
	v_dual_mov_b32 v14, v53 :: v_dual_mov_b32 v19, v53
	v_dual_mov_b32 v20, v53 :: v_dual_mov_b32 v21, v53
	v_dual_mov_b32 v22, v53 :: v_dual_mov_b32 v23, v53
	v_dual_mov_b32 v24, v53 :: v_dual_mov_b32 v25, v53
	v_dual_mov_b32 v34, v53 :: v_dual_mov_b32 v35, v53
	v_dual_mov_b32 v36, v53 :: v_dual_mov_b32 v37, v53
	v_dual_mov_b32 v38, v53 :: v_dual_mov_b32 v39, v53
	v_dual_mov_b32 v40, v53 :: v_dual_mov_b32 v41, v53
	v_dual_mov_b32 v42, v53 :: v_dual_mov_b32 v43, v53
	v_dual_mov_b32 v44, v53 :: v_dual_mov_b32 v45, v53
	v_dual_mov_b32 v46, v53 :: v_dual_mov_b32 v47, v53
	v_dual_mov_b32 v48, v53 :: v_dual_mov_b32 v49, v53
	v_dual_mov_b32 v26, v53 :: v_dual_mov_b32 v27, v53
	v_dual_mov_b32 v28, v53 :: v_dual_mov_b32 v29, v53
	v_dual_mov_b32 v30, v53 :: v_dual_mov_b32 v31, v53
	v_dual_mov_b32 v32, v53 :: v_dual_mov_b32 v33, v53
	s_lshr_b32 s50, s39, 16
	s_lshr_b32 s51, s37, 16
	s_lshl_b64 s[18:19], s[18:19], 1
	s_lshl_b64 s[6:7], s[6:7], 1
	s_mov_b32 s45, s3
	s_mov_b32 s46, s5
	s_add_nc_u64 s[40:41], s[4:5], s[10:11]
	s_or_b32 s50, s50, 0x600000
	s_or_b32 s4, s17, 0x7510000
	s_or_b32 s51, s51, 0x600000
	s_add_nc_u64 s[20:21], s[26:27], s[18:19]
	s_add_nc_u64 s[24:25], s[14:15], s[6:7]
	s_movk_i32 s8, 0x80
	s_mov_b32 s26, s11
	s_branch .LBB0_19
.LBB0_18:                               ;   in Loop: Header=BB0_19 Depth=1
	s_or_b32 exec_lo, exec_lo, s5
	s_cmp_eq_u32 s26, s48
	s_barrier_wait -1
	s_barrier_wait -3
	s_cbranch_scc1 .LBB0_41
.LBB0_19:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_21 Depth 2
                                        ;     Child Loop BB0_24 Depth 2
                                        ;     Child Loop BB0_26 Depth 2
                                        ;     Child Loop BB0_29 Depth 2
	s_and_b32 s27, s26, 1
	s_add_co_i32 s26, s26, 1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_i32 s5, s26, 0xffffffa0
	s_add_co_i32 s6, s5, s30
	s_xor_b32 s5, s27, 1
	s_min_i32 s6, s6, 0x60
	s_cmp_lt_i32 s26, s48
	s_cselect_b32 s10, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_and_b32 s7, s10, exec_lo
	s_cselect_b32 s36, s6, 0
	s_cmp_lt_i32 s36, 0x60
	s_cselect_b32 s6, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b32 s6, s49, s6
	s_or_b32 s6, s42, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s6
	s_cbranch_vccnz .LBB0_31
; %bb.20:                               ;   in Loop: Header=BB0_19 Depth=1
	v_nop
	v_nop
	v_nop
	v_nop
	v_mov_b64_e32 v[98:99], v[0:1]
	v_mov_b32_e32 v65, v57
	s_cmp_lg_u32 s5, 0
	s_mov_b32 s13, 0
	s_cselect_b32 s7, s46, s3
	s_cselect_b32 s6, s47, 0
.LBB0_21:                               ;   Parent Loop BB0_19 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_dual_mov_b32 v52, v98 :: v_dual_add_nc_u32 v65, -2, v65
	v_dual_mov_b32 v100, v99 :: v_dual_mov_b32 v101, v53
	v_add_nc_u32_e32 v99, 0x200, v99
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u64 v[102:103], v[52:53], 2, s[6:7]
	v_cmp_eq_u32_e32 vcc_lo, 0, v65
	v_add_nc_u32_e32 v98, 0x200, v98
	v_lshl_add_u64 v[100:101], v[100:101], 2, s[6:7]
	s_clause 0x1
	flat_store_b32 v[102:103], v53
	flat_store_b32 v[100:101], v53
	s_or_b32 s13, vcc_lo, s13
	s_wait_xcnt 0x0
	s_and_not1_b32 exec_lo, exec_lo, s13
	s_cbranch_execnz .LBB0_21
; %bb.22:                               ;   in Loop: Header=BB0_19 Depth=1
	s_or_b32 exec_lo, exec_lo, s13
	s_and_saveexec_b32 s13, s1
	s_cbranch_execz .LBB0_25
; %bb.23:                               ;   in Loop: Header=BB0_19 Depth=1
	v_add_nc_u64_e32 v[98:99], s[6:7], v[94:95]
	v_mov_b32_e32 v52, v61
	s_mov_b32 s6, 0
.LBB0_24:                               ;   Parent Loop BB0_19 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v52, 0x100, v52
	flat_store_b32 v[98:99], v53
	s_wait_xcnt 0x0
	v_add_nc_u64_e32 v[98:99], 0x400, v[98:99]
	v_cmp_lt_u32_e32 vcc_lo, 0x187f, v52
	s_or_b32 s6, vcc_lo, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s6
	s_cbranch_execnz .LBB0_24
.LBB0_25:                               ;   in Loop: Header=BB0_19 Depth=1
	s_or_b32 exec_lo, exec_lo, s13
	v_mov_b64_e32 v[98:99], v[0:1]
	v_mov_b32_e32 v65, v59
	s_cmp_lg_u32 s5, 0
	s_mov_b32 s13, 0
	s_cselect_b32 s7, s41, s45
	s_cselect_b32 s6, s40, s43
.LBB0_26:                               ;   Parent Loop BB0_19 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_dual_mov_b32 v52, v98 :: v_dual_add_nc_u32 v65, -2, v65
	v_dual_mov_b32 v100, v99 :: v_dual_mov_b32 v101, v53
	v_add_nc_u32_e32 v99, 0x200, v99
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u64 v[102:103], v[52:53], 2, s[6:7]
	v_cmp_eq_u32_e32 vcc_lo, 0, v65
	v_add_nc_u32_e32 v98, 0x200, v98
	v_lshl_add_u64 v[100:101], v[100:101], 2, s[6:7]
	s_clause 0x1
	flat_store_b32 v[102:103], v53
	flat_store_b32 v[100:101], v53
	s_or_b32 s13, vcc_lo, s13
	s_wait_xcnt 0x0
	s_and_not1_b32 exec_lo, exec_lo, s13
	s_cbranch_execnz .LBB0_26
; %bb.27:                               ;   in Loop: Header=BB0_19 Depth=1
	s_or_b32 exec_lo, exec_lo, s13
	s_and_saveexec_b32 s13, s2
	s_cbranch_execz .LBB0_30
; %bb.28:                               ;   in Loop: Header=BB0_19 Depth=1
	v_add_nc_u64_e32 v[98:99], s[6:7], v[96:97]
	v_mov_b32_e32 v52, v63
	s_mov_b32 s6, 0
.LBB0_29:                               ;   Parent Loop BB0_19 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v52, 0x100, v52
	flat_store_b32 v[98:99], v53
	s_wait_xcnt 0x0
	v_add_nc_u64_e32 v[98:99], 0x400, v[98:99]
	v_cmp_lt_u32_e32 vcc_lo, 0x121f, v52
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
	s_cselect_b32 s6, s26, 0
	s_mov_b32 s7, exec_lo
	v_cmpx_lt_i32_e32 0, v51
	s_xor_b32 s7, exec_lo, s7
	s_cbranch_execnz .LBB0_37
; %bb.32:                               ;   in Loop: Header=BB0_19 Depth=1
	s_and_not1_saveexec_b32 s13, s7
	s_cbranch_execnz .LBB0_40
.LBB0_33:                               ;   in Loop: Header=BB0_19 Depth=1
	s_or_b32 exec_lo, exec_lo, s13
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s44
	s_cbranch_vccnz .LBB0_35
.LBB0_34:                               ;   in Loop: Header=BB0_19 Depth=1
	s_cmp_lg_u32 s27, 0
	s_cselect_b32 s6, s47, 0
	s_cselect_b32 s5, s40, s43
	v_lshl_add_u32 v52, v50, 1, s6
	v_lshl_add_u32 v65, v54, 1, s5
	v_lshl_add_u32 v67, v66, 1, s6
	v_lshl_add_u32 v69, v68, 1, s5
	v_lshl_add_u32 v71, v70, 1, s5
	ds_load_b128 v[98:101], v52
	ds_load_b128 v[102:105], v52 offset:16
	v_lshl_add_u32 v52, v56, 1, s5
	ds_load_b128 v[106:109], v65
	ds_load_b128 v[110:113], v65 offset:16
	v_lshl_add_u32 v65, v64, 1, s5
	v_lshl_add_u32 v73, v72, 1, s5
	ds_load_b128 v[114:117], v52 offset:3072
	ds_load_b128 v[118:121], v52 offset:3088
	v_lshl_add_u32 v52, v58, 1, s5
	v_lshl_add_u32 v75, v76, 1, s5
	v_lshl_add_u32 v77, v78, 1, s5
	ds_load_b128 v[130:133], v69 offset:64
	ds_load_b128 v[134:137], v69 offset:80
	ds_load_b128 v[122:125], v52 offset:6144
	ds_load_b128 v[138:141], v71 offset:64
	ds_load_b128 v[142:145], v71 offset:80
	ds_load_b128 v[162:165], v75 offset:12288
	ds_load_b128 v[166:169], v75 offset:12304
	ds_load_b128 v[170:173], v77 offset:64
	s_wait_dscnt 0xa
	v_wmma_f32_16x16x32_bf16 v[42:49], v[98:105], v[106:113], v[42:49]
	ds_load_b128 v[174:177], v77 offset:80
	; sched_group_barrier mask(0x00000100) size(7) SyncID(0)
	s_wait_dscnt 0x9
	v_wmma_f32_16x16x32_bf16 v[34:41], v[98:105], v[114:121], v[34:41] matrix_a_reuse
	ds_load_b128 v[106:109], v65 offset:15360
	ds_load_b128 v[110:113], v65 offset:15376
	v_lshl_add_u32 v65, v62, 1, s5
	ds_load_b128 v[146:149], v65 offset:12288
	ds_load_b128 v[150:153], v65 offset:12304
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[2:9], v[98:105], v[146:153], v[2:9] matrix_a_reuse
	ds_load_b128 v[126:129], v52 offset:6160
	v_lshl_add_u32 v52, v60, 1, s5
	ds_load_b128 v[114:117], v67 offset:64
	ds_load_b128 v[118:121], v67 offset:80
	; sched_group_barrier mask(0x00000008) size(3) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[26:33], v[98:105], v[106:113], v[26:33] matrix_a_reuse
	ds_load_b128 v[106:109], v52 offset:9216
	ds_load_b128 v[110:113], v52 offset:9232
	v_lshl_add_u32 v52, v74, 1, s5
	; sched_group_barrier mask(0x00000100) size(7) SyncID(0)
	ds_load_b128 v[154:157], v52 offset:64
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[18:25], v[98:105], v[122:129], v[18:25] matrix_a_reuse
	ds_load_b128 v[122:125], v73 offset:64
	ds_load_b128 v[126:129], v73 offset:80
	ds_load_b128 v[158:161], v52 offset:80
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[10:17], v[98:105], v[106:113], v[10:17] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(3) SyncID(0)
	; sched_barrier mask(0x00000000)
	v_lshl_add_u32 v52, v80, 1, s6
	v_lshl_add_u32 v65, v82, 1, s5
	v_lshl_add_u32 v67, v84, 1, s5
	v_lshl_add_u32 v69, v86, 1, s5
	ds_load_b128 v[98:101], v52 offset:128
	ds_load_b128 v[102:105], v52 offset:144
	ds_load_b128 v[106:109], v65 offset:128
	ds_load_b128 v[110:113], v65 offset:144
	ds_load_b128 v[146:149], v67 offset:128
	ds_load_b128 v[150:153], v67 offset:144
	v_lshl_add_u32 v52, v88, 1, s5
	v_lshl_add_u32 v65, v90, 1, s5
	v_lshl_add_u32 v67, v92, 1, s5
	ds_load_b128 v[178:181], v69 offset:128
	v_wmma_f32_16x16x32_bf16 v[42:49], v[114:121], v[130:137], v[42:49]
	; sched_group_barrier mask(0x00000100) size(7) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[34:41], v[114:121], v[138:145], v[34:41] matrix_a_reuse
	s_wait_dscnt 0x8
	v_wmma_f32_16x16x32_bf16 v[18:25], v[114:121], v[122:129], v[18:25] matrix_a_reuse
	ds_load_b128 v[182:185], v69 offset:144
	ds_load_b128 v[122:125], v52 offset:128
	ds_load_b128 v[126:129], v52 offset:144
	ds_load_b128 v[130:133], v65 offset:12288
	ds_load_b128 v[134:137], v65 offset:12304
	ds_load_b128 v[138:141], v67 offset:128
	ds_load_b128 v[142:145], v67 offset:144
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[10:17], v[114:121], v[154:161], v[10:17] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(3) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(7) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[2:9], v[114:121], v[162:169], v[2:9] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[26:33], v[114:121], v[170:177], v[26:33] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(3) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0xa
	v_wmma_f32_16x16x32_bf16 v[42:49], v[98:105], v[106:113], v[42:49]
	; sched_group_barrier mask(0x00000100) size(7) SyncID(0)
	s_wait_dscnt 0x8
	v_wmma_f32_16x16x32_bf16 v[34:41], v[98:105], v[146:153], v[34:41] matrix_a_reuse
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[18:25], v[98:105], v[178:185], v[18:25] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(3) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(7) SyncID(0)
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[10:17], v[98:105], v[122:129], v[10:17] matrix_a_reuse
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[2:9], v[98:105], v[130:137], v[2:9] matrix_a_reuse
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[26:33], v[98:105], v[138:145], v[26:33] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(3) SyncID(0)
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
	s_mov_b32 s52, exec_lo
	v_cmpx_eq_u32_e32 1, v51
	s_cbranch_execz .LBB0_39
; %bb.38:                               ;   in Loop: Header=BB0_19 Depth=1
	s_cmp_lg_u32 s5, 0
	s_mul_i32 s10, s6, 0x60
	s_cselect_b32 s13, s40, s43
	s_cmp_gt_i32 s36, 0
	s_mov_b32 s38, s36
	s_cselect_b32 s17, -1, 0
	s_lshl_b64 s[14:15], s[10:11], 1
	s_mov_b32 s18, s11
	s_add_nc_u64 s[14:15], s[20:21], s[14:15]
	v_nop
	v_nop
	v_dual_mov_b32 v65, s13 :: v_dual_mov_b32 v98, s14
	s_and_b32 s10, s15, 0x1ffffff
	s_and_b32 s15, s29, s17
	s_bitset1_b32 s10, 31
	v_cndmask_b32_e64 v52, 0, 1, s15
	v_mov_b32_e32 v67, s10
	v_readfirstlane_b32 s57, v65
	v_readfirstlane_b32 s58, v98
	s_lshr_b64 s[14:15], s[38:39], 16
	v_readfirstlane_b32 s56, v52
	v_readfirstlane_b32 s59, v67
	s_lshl_b32 s13, s36, 16
	s_mov_b32 s15, s50
	s_mov_b32 s17, s9
	s_mov_b32 s19, s11
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[56:59], s[12:19]
.LBB0_39:                               ;   in Loop: Header=BB0_19 Depth=1
	s_or_b32 exec_lo, exec_lo, s52
	s_and_not1_saveexec_b32 s13, s7
	s_cbranch_execz .LBB0_33
.LBB0_40:                               ;   in Loop: Header=BB0_19 Depth=1
	s_cmp_lg_u32 s5, 0
	s_mul_i32 s10, s6, 0x60
	s_cselect_b32 s5, s47, 0
	s_cmp_gt_i32 s36, 0
	s_cselect_b32 s14, -1, 0
	s_lshl_b64 s[6:7], s[10:11], 1
	s_and_b32 s10, s35, s14
	s_add_nc_u64 s[6:7], s[24:25], s[6:7]
	v_cndmask_b32_e64 v52, 0, 1, s10
	s_and_b32 s7, s7, 0x1ffffff
	v_nop
	v_nop
	v_dual_mov_b32 v65, s5 :: v_dual_mov_b32 v98, s6
	s_bitset1_b32 s7, 31
	v_readfirstlane_b32 s52, v52
	v_mov_b32_e32 v67, s7
	s_delay_alu instid0(VALU_DEP_3)
	v_readfirstlane_b32 s53, v65
	v_readfirstlane_b32 s54, v98
	s_lshr_b64 s[6:7], s[36:37], 16
	s_lshl_b32 s5, s36, 16
	v_readfirstlane_b32 s55, v67
	s_mov_b32 s7, s51
	s_mov_b32 s10, s11
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	tensor_load_to_lds s[52:55], s[4:11]
	s_or_b32 exec_lo, exec_lo, s13
	s_and_not1_b32 vcc_lo, exec_lo, s44
	s_cbranch_vccz .LBB0_34
	s_branch .LBB0_35
.LBB0_41:
	s_wait_tensorcnt 0x0
	;;#ASMSTART
	s_wait_dscnt 0
	;;#ASMEND
	s_and_b32 vcc_lo, exec_lo, s44
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_vccz .LBB0_43
; %bb.42:
	v_dual_lshrrev_b32 v1, 1, v0 :: v_dual_lshlrev_b32 v50, 7, v0
	v_cvt_pk_bf16_f32 v49, v48, v49
	v_cvt_pk_bf16_f32 v48, v46, v47
	v_cvt_pk_bf16_f32 v46, v42, v43
	v_cvt_pk_bf16_f32 v41, v40, v41
	v_bitop3_b32 v1, v50, 0x788, v1 bitop3:0xc8
	v_cvt_pk_bf16_f32 v40, v38, v39
	v_cvt_pk_bf16_f32 v47, v44, v45
	v_lshlrev_b32_e32 v52, 4, v0
	v_cvt_pk_bf16_f32 v39, v36, v37
	v_or_b32_e32 v43, 0x1000, v1
	v_or_b32_e32 v42, 0x800, v1
	v_lshlrev_b32_e32 v36, 1, v1
	v_cvt_pk_bf16_f32 v25, v24, v25
	v_cvt_pk_bf16_f32 v24, v22, v23
	v_lshrrev_b32_e32 v38, 3, v43
	v_or_b32_e32 v43, 0x1800, v1
	v_lshrrev_b32_e32 v37, 3, v42
	v_cvt_pk_bf16_f32 v23, v20, v21
	v_and_b32_e32 v42, 0xf0, v52
	v_and_b32_e32 v44, 0x2f0, v38
	v_cvt_pk_bf16_f32 v38, v34, v35
	v_lshrrev_b32_e32 v34, 3, v43
	v_or_b32_e32 v43, 0x2000, v1
	v_or_b32_e32 v1, 0x2800, v1
	v_lshl_or_b32 v36, v51, 5, v36
	v_and_b32_e32 v37, 0x1f0, v37
	v_and_b32_e32 v34, 0x3f0, v34
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_dual_lshrrev_b32 v20, 3, v43 :: v_dual_lshrrev_b32 v1, 3, v1
	v_cvt_pk_bf16_f32 v22, v18, v19
	v_dual_add_nc_u32 v35, v42, v36 :: v_dual_add_nc_u32 v37, v37, v36
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_and_b32_e32 v19, 0x4f0, v20
	v_and_b32_e32 v1, 0x5f0, v1
	v_dual_add_nc_u32 v42, v44, v36 :: v_dual_add_nc_u32 v18, v34, v36
	v_cvt_pk_bf16_f32 v17, v16, v17
	v_cvt_pk_bf16_f32 v16, v14, v15
	v_cvt_pk_bf16_f32 v15, v12, v13
	v_cvt_pk_bf16_f32 v14, v10, v11
	v_add_nc_u32_e32 v10, v19, v36
	v_cvt_pk_bf16_f32 v9, v8, v9
	v_cvt_pk_bf16_f32 v8, v6, v7
	v_cvt_pk_bf16_f32 v7, v4, v5
	v_cvt_pk_bf16_f32 v6, v2, v3
	v_add_nc_u32_e32 v1, v1, v36
	v_cvt_pk_bf16_f32 v5, v32, v33
	v_cvt_pk_bf16_f32 v4, v30, v31
	v_cvt_pk_bf16_f32 v3, v28, v29
	v_cvt_pk_bf16_f32 v2, v26, v27
	ds_store_b128 v35, v[46:49]
	ds_store_b128 v37, v[38:41] offset:4096
	ds_store_b128 v42, v[22:25] offset:8192
	ds_store_b128 v18, v[14:17] offset:12288
	ds_store_b128 v10, v[6:9] offset:16384
	ds_store_b128 v1, v[2:5] offset:20480
.LBB0_43:
	v_cmp_ne_u32_e32 vcc_lo, 1, v55
	;;#ASMSTART
	s_wait_dscnt 0
	;;#ASMEND
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_vccnz .LBB0_54
; %bb.44:
	s_mul_i32 s3, s39, s37
	s_mov_b32 s0, exec_lo
	v_cmpx_gt_i32_e64 s3, v0
	s_cbranch_execz .LBB0_54
; %bb.45:
	s_ashr_i32 s35, s34, 31
	v_nop
	v_xad_u32 v2, v0, -1, s3
	s_lshl_b64 s[0:1], s[34:35], 1
	s_wait_kmcnt 0x0
	s_mul_i32 s14, s33, 0x60
	s_ashr_i32 s29, s28, 31
	s_add_nc_u64 s[4:5], s[22:23], s[0:1]
	s_mov_b32 s0, 0
                                        ; implicit-def: $vgpr1
                                        ; implicit-def: $vgpr6
                                        ; implicit-def: $sgpr12_sgpr13
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_u32_e32 0x2ff, v2
	s_xor_b32 s15, exec_lo, s1
	s_cbranch_execnz .LBB0_48
; %bb.46:
	s_or_saveexec_b32 s1, s15
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
	s_mov_b32 s17, s14
	s_mov_b32 s18, s14
	v_and_b32_e32 v9, 0x1fffffc, v8
	s_mov_b32 s19, s14
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
	v_dual_sub_nc_u32 v27, v19, v18 :: v_dual_add_nc_u32 v18, s14, v1
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
	v_mul_u64_e32 v[18:19], s[28:29], v[18:19]
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
	s_or_saveexec_b32 s1, s15
	v_mov_b64_e32 v[2:3], s[12:13]
	s_xor_b32 exec_lo, exec_lo, s1
	s_cbranch_execz .LBB0_47
.LBB0_51:
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
	v_dual_sub_nc_u32 v7, v4, v9 :: v_dual_lshlrev_b32 v4, 7, v4
	v_lshlrev_b32_e32 v9, 7, v9
	v_mul_lo_u32 v8, v7, s31
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_sub_nc_u32 v4, v4, v8 :: v_dual_add_nc_u32 v8, s14, v7
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
	s_cbranch_execnz .LBB0_53
.LBB0_54:
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end0:
	.size	bm128_bn096_bk096_wm8_wn1_mc1, .Lfunc_end0-bm128_bn096_bk096_wm8_wn1_mc1
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel bm128_bn096_bk096_wm8_wn1_mc1
		.amdhsa_group_segment_fixed_size 91392
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
		.amdhsa_next_free_vgpr 186
		.amdhsa_next_free_sgpr 60
		.amdhsa_named_barrier_count 0
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size 50
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text.bm128_bn096_bk096_wm8_wn1_mc1,"axG",@progbits,bm128_bn096_bk096_wm8_wn1_mc1,comdat
                                        ; -- End function
	.set .Lbm128_bn096_bk096_wm8_wn1_mc1.num_vgpr, 186
	.set .Lbm128_bn096_bk096_wm8_wn1_mc1.num_agpr, 0
	.set .Lbm128_bn096_bk096_wm8_wn1_mc1.numbered_sgpr, 60
	.set .Lbm128_bn096_bk096_wm8_wn1_mc1.num_named_barrier, 0
	.set .Lbm128_bn096_bk096_wm8_wn1_mc1.private_seg_size, 0
	.set .Lbm128_bn096_bk096_wm8_wn1_mc1.uses_vcc, 1
	.set .Lbm128_bn096_bk096_wm8_wn1_mc1.uses_flat_scratch, 1
	.set .Lbm128_bn096_bk096_wm8_wn1_mc1.has_dyn_sized_stack, 0
	.set .Lbm128_bn096_bk096_wm8_wn1_mc1.has_recursion, 0
	.set .Lbm128_bn096_bk096_wm8_wn1_mc1.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 6308
; TotalNumSgprs: 62
; NumVgprs: 186
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 91392 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 11
; NumSGPRsForWavesPerEU: 62
; NumVGPRsForWavesPerEU: 186
; NamedBarCnt: 0
; Occupancy: 5
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
	.type	__hip_cuid_35b42a8bf438461e,@object ; @__hip_cuid_35b42a8bf438461e
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_35b42a8bf438461e
__hip_cuid_35b42a8bf438461e:
	.byte	0                               ; 0x0
	.size	__hip_cuid_35b42a8bf438461e, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git aa451e1fe6a793394d6733051b1778633063ae96+PATCHED:d17c5aa0e3ea29cde402f58f27e39b6034effa27)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_35b42a8bf438461e
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
    .group_segment_fixed_size: 91392
    .kernarg_segment_align: 8
    .kernarg_segment_size: 132
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           bm128_bn096_bk096_wm8_wn1_mc1
    .private_segment_fixed_size: 0
    .sgpr_count:     62
    .sgpr_spill_count: 0
    .symbol:         bm128_bn096_bk096_wm8_wn1_mc1.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     186
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
