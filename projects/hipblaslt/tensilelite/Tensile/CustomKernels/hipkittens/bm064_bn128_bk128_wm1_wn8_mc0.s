	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.section	.text.bm064_bn128_bk128_wm1_wn8_mc0,"axG",@progbits,bm064_bn128_bk128_wm1_wn8_mc0,comdat
	.protected	bm064_bn128_bk128_wm1_wn8_mc0 ; -- Begin function bm064_bn128_bk128_wm1_wn8_mc0
	.globl	bm064_bn128_bk128_wm1_wn8_mc0
	.p2align	8
	.type	bm064_bn128_bk128_wm1_wn8_mc0,@function
bm064_bn128_bk128_wm1_wn8_mc0: ; @bm064_bn128_bk128_wm1_wn8_mc0
; %bb.0:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1 ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_mov_b64 s[2:3], src_shared_base
	s_movk_i32 s2, 0x4400
	s_load_b96 s[20:22], s[0:1], 0x78 nv
	s_and_b64 s[2:3], s[2:3], 12
	s_getreg_b32 s6, hwreg(HW_REG_IB_STS2, 6, 4)
	s_sub_co_i32 s4, 16, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_lshr_b32 s4, s4, 2
	s_cmp_lg_u64 s[2:3], 0
	s_cselect_b32 s2, s4, 0
	s_bfe_u32 s3, ttmp6, 0x4000c
	s_bfe_u32 s5, ttmp6, 0x40010
	s_add_co_i32 s3, s3, 1
	s_and_b32 s4, ttmp6, 15
	s_mul_i32 s3, ttmp9, s3
	s_add_co_i32 s5, s5, 1
	s_add_co_i32 s4, s4, s3
	s_mul_i32 s3, ttmp7, s5
	s_bfe_u32 s5, ttmp6, 0x40004
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_co_i32 s5, s5, s3
	s_cmp_eq_u32 s6, 0
	s_cselect_b32 s35, ttmp9, s4
	s_cselect_b32 s3, ttmp7, s5
	s_wait_kmcnt 0x0
	s_add_co_i32 s4, s20, 63
	s_add_co_i32 s6, s21, 0x7f
	s_ashr_i32 s5, s4, 31
	s_lshl_b32 s24, s35, 6
	s_lshr_b32 s5, s5, 26
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s4, s4, s5
	s_ashr_i32 s5, s6, 31
	s_ashr_i32 s4, s4, 6
	s_lshr_b32 s5, s5, 25
	s_add_co_i32 s6, s6, s5
	s_sub_co_i32 s5, s20, s24
	s_ashr_i32 s6, s6, 7
	s_min_i32 s23, s5, 64
	s_cmp_lt_i32 s35, s4
	s_cselect_b32 s25, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_and_b32 s5, s25, exec_lo
	s_cselect_b32 s27, s23, 0
	s_lshl_b32 s33, s3, 7
	s_sub_co_i32 s5, s21, s33
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_min_i32 s7, s5, 0x80
	s_cmp_lt_i32 s3, s6
	s_mov_b32 s5, s22
	s_cselect_b32 s21, -1, 0
	s_and_b32 s8, s21, exec_lo
	s_cselect_b32 s29, s7, 0
	s_add_co_i32 s13, s22, 0x7f
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 2, 1), 1
	s_min_i32 s7, s22, 0x80
	s_cmp_gt_i32 s13, 0x7f
	s_cselect_b32 s12, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_and_b32 s8, s12, exec_lo
	s_cselect_b32 s26, s7, 0
	s_cmp_lt_i32 s27, 64
	s_cselect_b32 s38, -1, 0
	s_and_b32 vcc_lo, exec_lo, s38
	s_mov_b32 s7, s38
	s_cbranch_vccnz .LBB0_2
; %bb.1:
	s_min_i32 s7, s26, s29
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lt_i32 s7, 0x80
	s_cselect_b32 s7, -1, 0
.LBB0_2:
	v_lshlrev_b32_e32 v34, 2, v0
	s_and_not1_b32 vcc_lo, exec_lo, s7
	s_cbranch_vccnz .LBB0_8
; %bb.3:
	v_or_b32_e32 v1, 0xffffff00, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v3, v34
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
	v_lshl_add_u32 v2, s2, 2, v34
	v_mov_b32_e32 v3, 0
	s_mov_b32 s7, 0
.LBB0_6:                                ; =>This Inner Loop Header: Depth=1
	v_add_nc_u32_e32 v1, 0x100, v1
	ds_store_b32 v2, v3 offset:17408
	v_add_nc_u32_e32 v2, 0x400, v2
	v_cmp_lt_u32_e32 vcc_lo, 0x20ff, v1
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
	s_load_b128 s[8:11], s[0:1], 0x20 nv
	s_load_b128 s[16:19], s[0:1], 0x48 nv
	v_lshrrev_b32_e32 v37, 5, v0
	s_lshl_b32 s34, s2, 2
	s_add_co_i32 s6, s6, -1
	s_mov_b64 s[30:31], src_shared_base
	s_or_b32 s30, s34, 0x4400
	s_add_co_i32 s37, s4, -1
	s_min_i32 s36, s3, s6
	s_wait_xcnt 0x0
	s_mov_b32 s0, exec_lo
	v_cmpx_lt_i32_e32 0, v37
	s_xor_b32 s39, exec_lo, s0
	s_cbranch_execz .LBB0_12
; %bb.9:
	s_mov_b32 s40, exec_lo
	v_cmpx_eq_u32_e32 1, v37
	s_cbranch_execz .LBB0_11
; %bb.10:
	s_cmp_gt_i32 s26, 0
	s_mov_b32 s28, s26
	s_cselect_b32 s4, -1, 0
	s_lshl_b32 s0, s36, 7
	s_wait_kmcnt 0x0
	s_bfe_i64 s[2:3], s[16:17], 0x200000
	s_ashr_i32 s1, s0, 31
	s_mov_b32 s6, 0
	s_mul_u64 s[0:1], s[2:3], s[0:1]
	s_mov_b32 s7, s6
	s_lshl_b64 s[0:1], s[0:1], 1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_nc_u64 s[2:3], s[10:11], s[0:1]
	v_dual_mov_b32 v1, s30 :: v_dual_mov_b32 v4, s2
	s_and_b32 s0, s3, 0x1ffffff
	s_and_b32 s3, s21, s4
	s_bitset1_b32 s0, 31
	v_cndmask_b32_e64 v2, 0, 1, s3
	v_mov_b32_e32 v3, s0
	v_readfirstlane_b32 s45, v1
	v_readfirstlane_b32 s46, v4
	s_lshr_b32 s0, s29, 16
	v_readfirstlane_b32 s44, v2
	v_readfirstlane_b32 s47, v3
	s_lshr_b64 s[2:3], s[28:29], 16
	s_lshl_b32 s1, s26, 16
	s_or_b32 s3, s0, 0x800000
	s_movk_i32 s4, 0x80
	s_mov_b32 s0, 0x7510000
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[44:47], s[0:7]
.LBB0_11:
	s_or_b32 exec_lo, exec_lo, s40
.LBB0_12:
	s_or_saveexec_b32 s39, s39
	s_min_i32 s28, s35, s37
	s_xor_b32 exec_lo, exec_lo, s39
	s_cbranch_execz .LBB0_14
; %bb.13:
	s_cmp_gt_i32 s26, 0
	s_mov_b32 s6, 0
	s_cselect_b32 s4, -1, 0
	s_lshl_b32 s0, s28, 6
	s_wait_kmcnt 0x0
	s_bfe_i64 s[2:3], s[8:9], 0x200000
	s_ashr_i32 s1, s0, 31
	s_and_b32 s4, s25, s4
	s_mul_u64 s[0:1], s[2:3], s[0:1]
	v_cndmask_b32_e64 v2, 0, 1, s4
	s_lshl_b64 s[2:3], s[0:1], 1
	s_lshr_b32 s0, s27, 16
	s_add_nc_u64 s[2:3], s[14:15], s[2:3]
	s_lshl_b32 s1, s26, 16
	s_and_b32 s3, s3, 0x1ffffff
	v_readfirstlane_b32 s40, v2
	s_bitset1_b32 s3, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v4, s2 :: v_dual_mov_b32 v3, s3
	s_lshr_b64 s[2:3], s[26:27], 16
	s_or_b32 s3, s0, 0x800000
	s_mov_b32 s4, 64
	v_readfirstlane_b32 s42, v4
	v_readfirstlane_b32 s43, v3
	s_mov_b32 s0, 0x7510000
	s_mov_b32 s7, s6
	s_mov_b32 s41, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[40:43], s[0:7]
.LBB0_14:
	s_or_b32 exec_lo, exec_lo, s39
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	v_dual_mov_b32 v9, 0 :: v_dual_lshlrev_b32 v41, 4, v37
	s_and_b32 s39, s25, s21
	s_and_not1_b32 vcc_lo, exec_lo, s12
	v_cndmask_b32_e64 v39, 0, 1, s39
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v8, v9 :: v_dual_mov_b32 v7, v9
	v_dual_mov_b32 v6, v9 :: v_dual_mov_b32 v5, v9
	v_dual_mov_b32 v4, v9 :: v_dual_mov_b32 v3, v9
	v_dual_mov_b32 v2, v9 :: v_dual_mov_b32 v17, v9
	v_dual_mov_b32 v16, v9 :: v_dual_mov_b32 v15, v9
	v_dual_mov_b32 v14, v9 :: v_dual_mov_b32 v13, v9
	v_dual_mov_b32 v12, v9 :: v_dual_mov_b32 v11, v9
	v_dual_mov_b32 v10, v9 :: v_dual_mov_b32 v33, v9
	v_dual_mov_b32 v32, v9 :: v_dual_mov_b32 v31, v9
	v_dual_mov_b32 v30, v9 :: v_dual_mov_b32 v29, v9
	v_dual_mov_b32 v28, v9 :: v_dual_mov_b32 v27, v9
	v_dual_mov_b32 v26, v9 :: v_dual_mov_b32 v25, v9
	v_dual_mov_b32 v24, v9 :: v_dual_mov_b32 v23, v9
	v_dual_mov_b32 v22, v9 :: v_dual_mov_b32 v21, v9
	v_dual_mov_b32 v20, v9 :: v_dual_mov_b32 v19, v9
	v_mov_b32_e32 v18, v9
	s_barrier_wait -1
	s_cbranch_vccnz .LBB0_37
; %bb.15:
	s_mov_b64 s[0:1], src_shared_base
	s_or_b32 s2, s34, 0xcc00
	s_mov_b32 s3, s1
	v_dual_lshlrev_b32 v1, 7, v0 :: v_dual_lshlrev_b32 v2, 7, v41
	s_and_b64 s[2:3], s[2:3], 15
	s_mov_b32 s7, 0
	s_sub_co_i32 s0, 16, s2
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v1, 0x780, v1
	s_lshr_b32 s0, s0, 2
	s_cmp_lg_u64 s[2:3], 0
	s_mov_b32 s42, s1
	s_cselect_b32 s0, s0, 0
	v_and_or_b32 v3, v0, 16, v1
	s_lshl2_add_u32 s2, s0, s34
	v_lshrrev_b32_e32 v1, 4, v1
	s_add_co_i32 s0, s2, 0x11000
	s_add_co_i32 s41, s2, 0xcc00
	s_and_b32 s6, s0, 15
	v_or_b32_e32 v4, v2, v3
	s_sub_co_i32 s3, 16, s6
	v_mov_b32_e32 v35, 0
	s_lshr_b32 s2, s3, 2
	s_cmp_lg_u64 s[6:7], 0
	v_lshrrev_b32_e32 v5, 4, v4
	s_cselect_b32 s2, s2, 0
	s_ashr_i32 s3, s13, 31
	s_lshl_b32 s6, s2, 2
	s_lshr_b32 s3, s3, 25
	s_add_nc_u64 s[34:35], s[0:1], s[6:7]
	s_add_co_i32 s13, s13, s3
	v_and_b32_e32 v5, 0x3f8, v5
	s_ashr_i32 s43, s13, 7
	s_cmp_lt_i32 s29, 0x80
	v_dual_mov_b32 v43, v35 :: v_dual_add_nc_u32 v36, v1, v3
	s_cselect_b32 s44, -1, 0
	s_lshl_b32 s0, s36, 7
	s_lshl_b32 s2, s28, 6
	s_ashr_i32 s1, s0, 31
	s_ashr_i32 s3, s2, 31
	s_wait_kmcnt 0x0
	s_bfe_i64 s[16:17], s[16:17], 0x200000
	s_bfe_i64 s[8:9], s[8:9], 0x200000
	s_mul_u64 s[0:1], s[16:17], s[0:1]
	s_mul_u64 s[2:3], s[8:9], s[2:3]
	v_dual_mov_b32 v45, v35 :: v_dual_add_nc_u32 v38, v5, v4
	v_or_b32_e32 v1, 0x100, v0
	v_add3_u32 v40, v3, v2, v5
	v_or_b32_e32 v48, 0xf00, v0
	v_or_b32_e32 v42, 0x4000, v34
	v_or_b32_e32 v49, 0x2100, v0
	v_or_b32_e32 v44, 0x8800, v34
	v_dual_mov_b32 v2, v35 :: v_dual_mov_b32 v3, v35
	v_dual_mov_b32 v4, v35 :: v_dual_mov_b32 v5, v35
	v_dual_mov_b32 v6, v35 :: v_dual_mov_b32 v7, v35
	v_dual_mov_b32 v8, v35 :: v_dual_mov_b32 v9, v35
	v_dual_mov_b32 v10, v35 :: v_dual_mov_b32 v11, v35
	v_dual_mov_b32 v12, v35 :: v_dual_mov_b32 v13, v35
	v_dual_mov_b32 v14, v35 :: v_dual_mov_b32 v15, v35
	v_dual_mov_b32 v16, v35 :: v_dual_mov_b32 v17, v35
	v_dual_mov_b32 v26, v35 :: v_dual_mov_b32 v27, v35
	v_dual_mov_b32 v28, v35 :: v_dual_mov_b32 v29, v35
	v_dual_mov_b32 v30, v35 :: v_dual_mov_b32 v31, v35
	v_dual_mov_b32 v32, v35 :: v_dual_mov_b32 v33, v35
	v_dual_mov_b32 v18, v35 :: v_dual_mov_b32 v19, v35
	v_dual_mov_b32 v20, v35 :: v_dual_mov_b32 v21, v35
	v_dual_mov_b32 v22, v35 :: v_dual_mov_b32 v23, v35
	v_dual_mov_b32 v24, v35 :: v_dual_mov_b32 v25, v35
	s_lshr_b32 s45, s29, 16
	s_lshr_b32 s46, s27, 16
	s_lshl_b64 s[0:1], s[0:1], 1
	s_lshl_b64 s[2:3], s[2:3], 1
	s_mov_b32 s40, s31
	s_movk_i32 s12, 0x80
	s_bitset1_b32 s45, 23
	s_bitset1_b32 s46, 23
	s_add_nc_u64 s[16:17], s[10:11], s[0:1]
	s_add_nc_u64 s[36:37], s[14:15], s[2:3]
	s_mov_b32 s47, -1
	s_mov_b32 s0, 0x7510000
	s_mov_b32 s4, 64
	s_mov_b32 s48, s7
	s_branch .LBB0_17
.LBB0_16:                               ;   in Loop: Header=BB0_17 Depth=1
	s_cmp_eq_u32 s48, s43
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_scc1 .LBB0_37
.LBB0_17:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_19 Depth 2
                                        ;     Child Loop BB0_22 Depth 2
                                        ;     Child Loop BB0_24 Depth 2
                                        ;     Child Loop BB0_27 Depth 2
	s_and_b32 s49, s48, 1
	s_add_co_i32 s48, s48, 1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshl_b32 s1, s48, 7
	s_sub_co_i32 s2, s22, s1
	s_xor_b32 s1, s49, 1
	s_min_i32 s2, s2, 0x80
	s_cmp_lt_i32 s48, s43
	s_cselect_b32 s6, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_and_b32 s3, s6, exec_lo
	s_cselect_b32 s26, s2, 0
	s_cmp_lt_i32 s26, 0x80
	s_cselect_b32 s2, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b32 s2, s44, s2
	s_or_b32 s2, s38, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB0_29
; %bb.18:                               ;   in Loop: Header=BB0_17 Depth=1
	v_mov_b64_e32 v[46:47], v[0:1]
	v_nop
	v_nop
	v_nop
	v_mov_b32_e32 v50, 16
	s_cmp_lg_u32 s1, 0
	s_mov_b32 s8, 0
	s_cselect_b32 s3, s42, s31
	s_cselect_b32 s2, s41, 0
.LBB0_19:                               ;   Parent Loop BB0_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_dual_mov_b32 v34, v46 :: v_dual_add_nc_u32 v50, -2, v50
	v_dual_mov_b32 v52, v47 :: v_dual_mov_b32 v53, v35
	v_add_nc_u32_e32 v47, 0x200, v47
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u64 v[54:55], v[34:35], 2, s[2:3]
	v_cmp_eq_u32_e32 vcc_lo, 0, v50
	v_add_nc_u32_e32 v46, 0x200, v46
	v_lshl_add_u64 v[52:53], v[52:53], 2, s[2:3]
	s_clause 0x1
	flat_store_b32 v[54:55], v35
	flat_store_b32 v[52:53], v35
	s_or_b32 s8, vcc_lo, s8
	s_wait_xcnt 0x0
	s_and_not1_b32 exec_lo, exec_lo, s8
	s_cbranch_execnz .LBB0_19
; %bb.20:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s8
	s_and_saveexec_b32 s8, s47
	s_cbranch_execz .LBB0_23
; %bb.21:                               ;   in Loop: Header=BB0_17 Depth=1
	v_add_nc_u64_e32 v[46:47], s[2:3], v[42:43]
	v_mov_b32_e32 v34, v48
	s_mov_b32 s2, 0
.LBB0_22:                               ;   Parent Loop BB0_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v34, 0x100, v34
	flat_store_b32 v[46:47], v35
	s_wait_xcnt 0x0
	v_add_nc_u64_e32 v[46:47], 0x400, v[46:47]
	v_cmp_lt_u32_e32 vcc_lo, 0xfff, v34
	s_or_b32 s2, vcc_lo, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s2
	s_cbranch_execnz .LBB0_22
.LBB0_23:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s8
	v_mov_b64_e32 v[46:47], v[0:1]
	v_mov_b32_e32 v50, 34
	s_cmp_lg_u32 s1, 0
	s_mov_b32 s8, 0
	s_cselect_b32 s3, s35, s40
	s_cselect_b32 s2, s34, s30
.LBB0_24:                               ;   Parent Loop BB0_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_dual_mov_b32 v34, v46 :: v_dual_add_nc_u32 v50, -2, v50
	v_dual_mov_b32 v52, v47 :: v_dual_mov_b32 v53, v35
	v_add_nc_u32_e32 v47, 0x200, v47
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u64 v[54:55], v[34:35], 2, s[2:3]
	v_cmp_eq_u32_e32 vcc_lo, 0, v50
	v_add_nc_u32_e32 v46, 0x200, v46
	v_lshl_add_u64 v[52:53], v[52:53], 2, s[2:3]
	s_clause 0x1
	flat_store_b32 v[54:55], v35
	flat_store_b32 v[52:53], v35
	s_or_b32 s8, vcc_lo, s8
	s_wait_xcnt 0x0
	s_and_not1_b32 exec_lo, exec_lo, s8
	s_cbranch_execnz .LBB0_24
; %bb.25:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s8
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_28
; %bb.26:                               ;   in Loop: Header=BB0_17 Depth=1
	v_add_nc_u64_e32 v[46:47], s[2:3], v[44:45]
	v_mov_b32_e32 v34, v49
	s_mov_b32 s2, 0
.LBB0_27:                               ;   Parent Loop BB0_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v34, 0x100, v34
	flat_store_b32 v[46:47], v35
	s_wait_xcnt 0x0
	v_add_nc_u64_e32 v[46:47], 0x400, v[46:47]
	v_cmp_lt_u32_e32 vcc_lo, 0x20ff, v34
	s_or_b32 s2, vcc_lo, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s2
	s_cbranch_execnz .LBB0_27
.LBB0_28:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s8
	s_barrier_signal -1
	s_barrier_wait -1
.LBB0_29:                               ;   in Loop: Header=BB0_17 Depth=1
	s_and_b32 s2, s6, exec_lo
	s_cselect_b32 s2, s48, 0
	s_mov_b32 s3, exec_lo
	v_cmpx_lt_i32_e32 0, v37
	s_xor_b32 s3, exec_lo, s3
	s_cbranch_execnz .LBB0_32
; %bb.30:                               ;   in Loop: Header=BB0_17 Depth=1
	s_and_not1_saveexec_b32 s8, s3
	s_cbranch_execnz .LBB0_35
.LBB0_31:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s8
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s39
	s_cbranch_vccnz .LBB0_16
	s_branch .LBB0_36
.LBB0_32:                               ;   in Loop: Header=BB0_17 Depth=1
	s_mov_b32 s50, exec_lo
	v_cmpx_eq_u32_e32 1, v37
	s_cbranch_execz .LBB0_34
; %bb.33:                               ;   in Loop: Header=BB0_17 Depth=1
	s_cmp_lg_u32 s1, 0
	s_mov_b32 s28, s26
	s_cselect_b32 s10, s34, s30
	s_cmp_gt_i32 s26, 0
	s_mov_b32 s13, s5
	s_cselect_b32 s11, -1, 0
	s_lshl_b32 s6, s2, 7
	s_mov_b32 s14, s7
	s_lshl_b64 s[8:9], s[6:7], 1
	s_mov_b32 s15, s7
	s_add_nc_u64 s[8:9], s[16:17], s[8:9]
	s_delay_alu instid0(SALU_CYCLE_1)
	v_dual_mov_b32 v47, s10 :: v_dual_mov_b32 v46, s8
	s_and_b32 s6, s9, 0x1ffffff
	s_and_b32 s9, s21, s11
	s_bitset1_b32 s6, 31
	v_cndmask_b32_e64 v34, 0, 1, s9
	v_mov_b32_e32 v51, s6
	v_readfirstlane_b32 s53, v47
	v_readfirstlane_b32 s54, v46
	s_lshr_b64 s[10:11], s[28:29], 16
	v_readfirstlane_b32 s52, v34
	v_readfirstlane_b32 s55, v51
	s_lshl_b32 s9, s26, 16
	s_mov_b32 s8, s0
	s_mov_b32 s11, s45
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[52:55], s[8:15]
.LBB0_34:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s50
	s_and_not1_saveexec_b32 s8, s3
	s_cbranch_execz .LBB0_31
.LBB0_35:                               ;   in Loop: Header=BB0_17 Depth=1
	s_cmp_lg_u32 s1, 0
	s_cselect_b32 s1, s41, 0
	s_cmp_gt_i32 s26, 0
	s_cselect_b32 s9, -1, 0
	s_lshl_b32 s6, s2, 7
	s_delay_alu instid0(SALU_CYCLE_1)
	s_lshl_b64 s[2:3], s[6:7], 1
	s_and_b32 s6, s25, s9
	s_add_nc_u64 s[2:3], s[36:37], s[2:3]
	v_cndmask_b32_e64 v34, 0, 1, s6
	s_and_b32 s3, s3, 0x1ffffff
	v_dual_mov_b32 v47, s1 :: v_dual_mov_b32 v46, s2
	s_bitset1_b32 s3, 31
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_readfirstlane_b32 s52, v34
	v_mov_b32_e32 v51, s3
	v_readfirstlane_b32 s53, v47
	v_readfirstlane_b32 s54, v46
	s_lshr_b64 s[2:3], s[26:27], 16
	s_lshl_b32 s1, s26, 16
	v_readfirstlane_b32 s55, v51
	s_mov_b32 s3, s46
	s_mov_b32 s6, s7
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	tensor_load_to_lds s[52:55], s[0:7]
	s_or_b32 exec_lo, exec_lo, s8
	s_and_not1_b32 vcc_lo, exec_lo, s39
	s_cbranch_vccnz .LBB0_16
.LBB0_36:                               ;   in Loop: Header=BB0_17 Depth=1
	s_cmp_lg_u32 s49, 0
	s_cselect_b32 s1, s41, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_lshl_add_u32 v34, v36, 1, s1
	s_cselect_b32 s1, s34, s30
	v_lshl_add_u32 v46, v38, 1, s1
	ds_load_b128 v[50:53], v34
	ds_load_b128 v[54:57], v34 offset:16
	ds_load_b128 v[58:61], v34 offset:4352
	ds_load_b128 v[62:65], v34 offset:4368
	ds_load_b128 v[66:69], v34 offset:8704
	ds_load_b128 v[78:81], v46
	ds_load_b128 v[82:85], v46 offset:16
	v_lshl_add_u32 v46, v40, 1, s1
	ds_load_b128 v[86:89], v34 offset:64
	ds_load_b128 v[90:93], v34 offset:80
	ds_load_b128 v[94:97], v34 offset:8768
	ds_load_b128 v[98:101], v34 offset:8784
	ds_load_b128 v[102:105], v34 offset:13120
	ds_load_b128 v[106:109], v34 offset:13136
	ds_load_b128 v[110:113], v46 offset:64
	ds_load_b128 v[114:117], v46 offset:80
	; sched_group_barrier mask(0x00000100) size(5) SyncID(0)
	s_wait_dscnt 0x8
	v_wmma_f32_16x16x32_bf16 v[26:33], v[50:57], v[78:85], v[26:33]
	v_wmma_f32_16x16x32_bf16 v[10:17], v[58:65], v[78:85], v[10:17] matrix_b_reuse
	ds_load_b128 v[70:73], v34 offset:13056
	ds_load_b128 v[74:77], v34 offset:13072
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[18:25], v[70:77], v[78:85], v[18:25] matrix_b_reuse
	ds_load_b128 v[70:73], v34 offset:8720
	; sched_group_barrier mask(0x00000100) size(5) SyncID(0)
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[2:9], v[66:73], v[78:85], v[2:9] matrix_b_reuse
	ds_load_b128 v[66:69], v34 offset:4416
	ds_load_b128 v[70:73], v34 offset:4432
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_barrier mask(0x00000000)
	ds_load_b128 v[50:53], v34 offset:128
	ds_load_b128 v[54:57], v34 offset:144
	ds_load_b128 v[58:61], v34 offset:4480
	ds_load_b128 v[62:65], v34 offset:4496
	ds_load_b128 v[74:77], v34 offset:8832
	v_wmma_f32_16x16x32_bf16 v[26:33], v[86:93], v[110:117], v[26:33]
	; sched_group_barrier mask(0x00000100) size(5) SyncID(0)
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[10:17], v[66:73], v[110:117], v[10:17] matrix_b_reuse
	ds_load_b128 v[78:81], v34 offset:8848
	ds_load_b128 v[66:69], v34 offset:13184
	ds_load_b128 v[70:73], v34 offset:13200
	ds_load_b128 v[82:85], v46 offset:128
	ds_load_b128 v[86:89], v46 offset:144
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(5) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[2:9], v[94:101], v[110:117], v[2:9] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[18:25], v[102:109], v[110:117], v[18:25] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_barrier mask(0x00000000)
	ds_load_b128 v[90:93], v34 offset:192
	ds_load_b128 v[94:97], v34 offset:208
	ds_load_b128 v[98:101], v34 offset:4544
	ds_load_b128 v[102:105], v34 offset:4560
	ds_load_b128 v[106:109], v34 offset:8896
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[26:33], v[50:57], v[82:89], v[26:33]
	; sched_group_barrier mask(0x00000100) size(5) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[10:17], v[58:65], v[82:89], v[10:17] matrix_b_reuse
	ds_load_b128 v[110:113], v34 offset:8912
	ds_load_b128 v[50:53], v34 offset:13248
	ds_load_b128 v[54:57], v34 offset:13264
	ds_load_b128 v[58:61], v46 offset:192
	ds_load_b128 v[62:65], v46 offset:208
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(5) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[2:9], v[74:81], v[82:89], v[2:9] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[18:25], v[66:73], v[82:89], v[18:25] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[26:33], v[90:97], v[58:65], v[26:33]
	; sched_group_barrier mask(0x00000100) size(5) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[10:17], v[98:105], v[58:65], v[10:17] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(5) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[2:9], v[106:113], v[58:65], v[2:9] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[18:25], v[50:57], v[58:65], v[18:25] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
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
	s_and_b32 vcc_lo, exec_lo, s39
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_vccz .LBB0_39
; %bb.38:
	v_and_or_b32 v1, v0, 15, v41
	v_and_b32_e32 v34, 16, v0
	v_cvt_pk_bf16_f32 v33, v32, v33
	v_cvt_pk_bf16_f32 v32, v30, v31
	v_cvt_pk_bf16_f32 v31, v28, v29
	v_lshlrev_b32_e32 v28, 3, v1
	v_cvt_pk_bf16_f32 v30, v26, v27
	v_lshl_or_b32 v1, v1, 7, v34
	v_cvt_pk_bf16_f32 v17, v16, v17
	v_cvt_pk_bf16_f32 v16, v14, v15
	v_and_b32_e32 v26, 0x3f0, v28
	v_cvt_pk_bf16_f32 v15, v12, v13
	v_cvt_pk_bf16_f32 v14, v10, v11
	v_cvt_pk_bf16_f32 v9, v8, v9
	v_cvt_pk_bf16_f32 v8, v6, v7
	v_add_nc_u32_e32 v1, v26, v1
	v_cvt_pk_bf16_f32 v7, v4, v5
	v_cvt_pk_bf16_f32 v6, v2, v3
	v_cvt_pk_bf16_f32 v5, v24, v25
	v_cvt_pk_bf16_f32 v4, v22, v23
	v_cvt_pk_bf16_f32 v3, v20, v21
	v_cvt_pk_bf16_f32 v2, v18, v19
	ds_store_b128 v1, v[30:33]
	ds_store_b128 v1, v[14:17] offset:32
	ds_store_b128 v1, v[6:9] offset:64
	ds_store_b128 v1, v[2:5] offset:96
.LBB0_39:
	v_cmp_ne_u32_e32 vcc_lo, 1, v39
	;;#ASMSTART
	s_wait_dscnt 0
	;;#ASMEND
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_vccnz .LBB0_50
; %bb.40:
	s_mul_i32 s3, s29, s27
	s_mov_b32 s0, exec_lo
	v_cmpx_gt_i32_e64 s3, v0
	s_cbranch_execz .LBB0_50
; %bb.41:
	s_ashr_i32 s25, s24, 31
	v_nop
	v_xad_u32 v2, v0, -1, s3
	s_lshl_b64 s[0:1], s[24:25], 1
	s_ashr_i32 s21, s20, 31
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[4:5], s[18:19], s[0:1]
	s_mov_b32 s0, 0
                                        ; implicit-def: $vgpr1
                                        ; implicit-def: $vgpr6
                                        ; implicit-def: $sgpr12_sgpr13
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
	s_abs_i32 s15, s23
	v_lshrrev_b32_e32 v1, 8, v2
	s_cvt_f32_u32 s0, s15
	v_or_b32_e32 v3, 0x300, v0
	s_sub_co_i32 s1, 0, s15
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
	s_mov_b32 s16, s23
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
	v_mul_lo_u32 v24, v20, s15
	v_mul_lo_u32 v26, v21, s15
	v_mul_lo_u32 v27, v22, s15
	v_dual_add_nc_u32 v25, 1, v20 :: v_dual_add_nc_u32 v29, 1, v21
	v_mul_lo_u32 v28, v23, s15
	v_dual_add_nc_u32 v30, 1, v22 :: v_dual_add_nc_u32 v31, 1, v23
	v_dual_sub_nc_u32 v6, v6, v24 :: v_dual_sub_nc_u32 v12, v12, v26
	v_dual_sub_nc_u32 v16, v16, v27 :: v_dual_bitop2_b32 v14, s25, v14 bitop3:0x14
	v_xor_b32_e32 v18, s25, v18
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
	v_dual_add_nc_u32 v22, s22, v26 :: v_dual_sub_nc_u32 v12, v3, v12
	v_ashrrev_i32_e32 v21, 31, v20
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mul_lo_u32 v6, v1, s23
	v_dual_sub_nc_u32 v27, v19, v18 :: v_dual_add_nc_u32 v18, s33, v1
	v_sub_nc_u32_e32 v14, v4, v14
	v_ashrrev_i32_e32 v23, 31, v22
	v_lshl_add_u32 v11, v11, 6, v12
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_mul_lo_u32 v16, v27, s18
	v_dual_add_nc_u32 v24, s24, v27 :: v_dual_sub_nc_u32 v6, v2, v6
	v_ashrrev_i32_e32 v19, 31, v18
	v_lshl_add_u32 v26, v26, 6, v14
	v_mul_u64_e32 v[20:21], s[6:7], v[20:21]
	v_ashrrev_i32_e32 v25, 31, v24
	v_lshl_add_u32 v1, v1, 6, v6
	v_sub_nc_u32_e32 v16, v5, v16
	v_mul_u64_e32 v[18:19], s[20:21], v[18:19]
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
	s_cbranch_execnz .LBB0_45
; %bb.46:
	s_or_b32 exec_lo, exec_lo, s26
	v_cmp_ne_u32_e32 vcc_lo, v8, v9
	v_lshl_or_b32 v0, v9, 8, v0
	v_dual_mov_b32 v6, s15 :: v_dual_mov_b32 v1, s25
	s_and_b32 s0, vcc_lo, exec_lo
	s_or_saveexec_b32 s1, s14
	v_mov_b64_e32 v[2:3], s[12:13]
	s_xor_b32 exec_lo, exec_lo, s1
	s_cbranch_execz .LBB0_43
.LBB0_47:
	s_abs_i32 s2, s23
	s_ashr_i32 s8, s23, 31
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
	v_dual_sub_nc_u32 v7, v4, v9 :: v_dual_lshlrev_b32 v4, 6, v4
	v_lshlrev_b32_e32 v9, 6, v9
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
	.size	bm064_bn128_bk128_wm1_wn8_mc0, .Lfunc_end0-bm064_bn128_bk128_wm1_wn8_mc0
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel bm064_bn128_bk128_wm1_wn8_mc0
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
		.amdhsa_next_free_vgpr 145
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
		.amdhsa_inst_pref_size 39
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text.bm064_bn128_bk128_wm1_wn8_mc0,"axG",@progbits,bm064_bn128_bk128_wm1_wn8_mc0,comdat
                                        ; -- End function
	.set .Lbm064_bn128_bk128_wm1_wn8_mc0.num_vgpr, 118
	.set .Lbm064_bn128_bk128_wm1_wn8_mc0.num_agpr, 0
	.set .Lbm064_bn128_bk128_wm1_wn8_mc0.numbered_sgpr, 56
	.set .Lbm064_bn128_bk128_wm1_wn8_mc0.num_named_barrier, 0
	.set .Lbm064_bn128_bk128_wm1_wn8_mc0.private_seg_size, 0
	.set .Lbm064_bn128_bk128_wm1_wn8_mc0.uses_vcc, 1
	.set .Lbm064_bn128_bk128_wm1_wn8_mc0.uses_flat_scratch, 1
	.set .Lbm064_bn128_bk128_wm1_wn8_mc0.has_dyn_sized_stack, 0
	.set .Lbm064_bn128_bk128_wm1_wn8_mc0.has_recursion, 0
	.set .Lbm064_bn128_bk128_wm1_wn8_mc0.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 4944
; TotalNumSgprs: 58
; NumVgprs: 118
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 104448 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 9
; NumSGPRsForWavesPerEU: 58
; NumVGPRsForWavesPerEU: 145
; NamedBarCnt: 0
; Occupancy: 6
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
	.type	__hip_cuid_b63cb72a321943c5,@object ; @__hip_cuid_b63cb72a321943c5
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_b63cb72a321943c5
__hip_cuid_b63cb72a321943c5:
	.byte	0                               ; 0x0
	.size	__hip_cuid_b63cb72a321943c5, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git aa451e1fe6a793394d6733051b1778633063ae96+PATCHED:d17c5aa0e3ea29cde402f58f27e39b6034effa27)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_b63cb72a321943c5
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
    macrotile: [64, 128, 128]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, One]
  MatrixInstruction: [16, 16, 32, 1]
  EnableMatrixInstruction: True
  MIWaveTile: [2, 2]
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
    .group_segment_fixed_size: 104448
    .kernarg_segment_align: 8
    .kernarg_segment_size: 132
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           bm064_bn128_bk128_wm1_wn8_mc0
    .private_segment_fixed_size: 0
    .sgpr_count:     58
    .sgpr_spill_count: 0
    .symbol:         bm064_bn128_bk128_wm1_wn8_mc0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     118
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
