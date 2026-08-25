	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.section	.text.bm064_bn064_bk064_wm4_wn2_mc0,"axG",@progbits,bm064_bn064_bk064_wm4_wn2_mc0,comdat
	.protected	bm064_bn064_bk064_wm4_wn2_mc0 ; -- Begin function bm064_bn064_bk064_wm4_wn2_mc0
	.globl	bm064_bn064_bk064_wm4_wn2_mc0
	.p2align	8
	.type	bm064_bn064_bk064_wm4_wn2_mc0,@function
bm064_bn064_bk064_wm4_wn2_mc0: ; @bm064_bn064_bk064_wm4_wn2_mc0
; %bb.0:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1 ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_mov_b64 s[2:3], src_shared_base
	s_movk_i32 s2, 0x2200
	s_load_b96 s[16:18], s[0:1], 0x78 nv
	s_and_b64 s[2:3], s[2:3], 12
	s_getreg_b32 s5, hwreg(HW_REG_IB_STS2, 6, 4)
	s_sub_co_i32 s4, 16, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_lshr_b32 s4, s4, 2
	s_cmp_lg_u64 s[2:3], 0
	s_cselect_b32 s10, s4, 0
	s_bfe_u32 s2, ttmp6, 0x4000c
	s_bfe_u32 s4, ttmp6, 0x40010
	s_add_co_i32 s2, s2, 1
	s_and_b32 s3, ttmp6, 15
	s_mul_i32 s2, ttmp9, s2
	s_add_co_i32 s4, s4, 1
	s_add_co_i32 s3, s3, s2
	s_mul_i32 s2, ttmp7, s4
	s_bfe_u32 s4, ttmp6, 0x40004
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_co_i32 s4, s4, s2
	s_cmp_eq_u32 s5, 0
	s_wait_kmcnt 0x0
	s_mov_b32 s9, s18
	s_cselect_b32 s8, ttmp9, s3
	s_cselect_b32 s11, ttmp7, s4
	s_add_co_i32 s2, s16, 63
	s_add_co_i32 s4, s17, 63
	s_ashr_i32 s3, s2, 31
	s_lshl_b32 s20, s8, 6
	s_lshr_b32 s3, s3, 26
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s2, s2, s3
	s_ashr_i32 s3, s4, 31
	s_ashr_i32 s22, s2, 6
	s_lshr_b32 s2, s3, 26
	s_add_co_i32 s4, s4, s2
	s_sub_co_i32 s2, s16, s20
	s_ashr_i32 s24, s4, 6
	s_min_i32 s19, s2, 64
	s_cmp_lt_i32 s8, s22
	s_cselect_b32 s21, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_and_b32 s2, s21, exec_lo
	s_cselect_b32 s3, s19, 0
	s_lshl_b32 s30, s11, 6
	s_sub_co_i32 s2, s17, s30
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_min_i32 s2, s2, 64
	s_cmp_lt_i32 s11, s24
	s_cselect_b32 s17, -1, 0
	s_and_b32 s4, s17, exec_lo
	s_cselect_b32 s23, s2, 0
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 2, 1), 1
	s_cmp_lt_i32 s3, 64
	s_cselect_b32 s31, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 vcc_lo, exec_lo, s31
	s_mov_b32 s2, s31
	s_cbranch_vccnz .LBB0_2
; %bb.1:
	s_min_i32 s2, s18, s23
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lt_i32 s2, 64
	s_cselect_b32 s2, -1, 0
.LBB0_2:
	v_sub_nc_u32_e32 v3, 0x87f, v0
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB0_16
; %bb.3:
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_dual_lshrrev_b32 v2, 8, v3 :: v_dual_lshlrev_b32 v5, 2, v0
	s_mov_b32 s4, 0
	s_mov_b32 s6, 0
	v_dual_mov_b32 v6, 0 :: v_dual_add_nc_u32 v4, 2, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mov_b32 v1, v2 :: v_dual_mov_b32 v7, v5
	v_and_b32_e32 v4, 30, v4
	s_branch .LBB0_5
.LBB0_4:                                ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
	s_add_co_i32 s4, s4, 2
	v_add_nc_u32_e32 v7, 0x800, v7
	v_cmp_eq_u32_e32 vcc_lo, s4, v4
	s_or_b32 s6, vcc_lo, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s6
	s_cbranch_execz .LBB0_9
.LBB0_5:                                ; =>This Inner Loop Header: Depth=1
	s_mov_b32 s5, s4
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b64 s[12:13], s[4:5], 0x100000000
	s_mov_b32 s5, exec_lo
	v_cmp_le_u32_e32 vcc_lo, s13, v1
	v_cmpx_le_u32_e64 s12, v2
; %bb.6:                                ;   in Loop: Header=BB0_5 Depth=1
	ds_store_b32 v7, v6
; %bb.7:                                ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s5
	s_and_saveexec_b32 s2, vcc_lo
	s_cbranch_execz .LBB0_4
; %bb.8:                                ;   in Loop: Header=BB0_5 Depth=1
	ds_store_b32 v7, v6 offset:1024
	s_branch .LBB0_4
.LBB0_9:
	s_or_b32 exec_lo, exec_lo, s6
	v_lshl_add_u32 v5, s10, 2, v5
	v_mov_b32_e32 v6, 0
	s_mov_b32 s4, 0
	s_mov_b32 s6, 0
	s_branch .LBB0_11
.LBB0_10:                               ;   in Loop: Header=BB0_11 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
	s_add_co_i32 s4, s4, 2
	v_add_nc_u32_e32 v5, 0x800, v5
	v_cmp_eq_u32_e32 vcc_lo, s4, v4
	s_or_b32 s6, vcc_lo, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s6
	s_cbranch_execz .LBB0_15
.LBB0_11:                               ; =>This Inner Loop Header: Depth=1
	s_mov_b32 s5, s4
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b64 s[12:13], s[4:5], 0x100000000
	s_mov_b32 s5, exec_lo
	v_cmp_le_u32_e32 vcc_lo, s13, v1
	v_cmpx_le_u32_e64 s12, v2
; %bb.12:                               ;   in Loop: Header=BB0_11 Depth=1
	ds_store_b32 v5, v6 offset:8704
; %bb.13:                               ;   in Loop: Header=BB0_11 Depth=1
	s_or_b32 exec_lo, exec_lo, s5
	s_and_saveexec_b32 s2, vcc_lo
	s_cbranch_execz .LBB0_10
; %bb.14:                               ;   in Loop: Header=BB0_11 Depth=1
	ds_store_b32 v5, v6 offset:9728
	s_branch .LBB0_10
.LBB0_15:
	s_or_b32 exec_lo, exec_lo, s6
	s_barrier_signal -1
	s_barrier_wait -1
.LBB0_16:
	s_clause 0x2
	s_load_b64 s[26:27], s[0:1], 0x0 nv
	s_load_b128 s[4:7], s[0:1], 0x20 nv
	s_load_b128 s[12:15], s[0:1], 0x48 nv
	v_lshrrev_b32_e32 v23, 5, v0
	v_med3_i32 v1, s18, 0, 64
	s_lshl_b32 s2, s10, 2
	s_add_co_i32 s24, s24, -1
	s_wait_xcnt 0x0
	s_mov_b64 s[0:1], src_shared_base
	s_or_b32 s33, s2, 0x2200
	s_add_co_i32 s22, s22, -1
	s_min_i32 s0, s11, s24
	s_mov_b32 s10, exec_lo
	v_cmpx_lt_i32_e32 0, v23
	s_xor_b32 s10, exec_lo, s10
	s_cbranch_execz .LBB0_20
; %bb.17:
	s_mov_b32 s11, exec_lo
	v_cmpx_eq_u32_e32 1, v23
	s_cbranch_execz .LBB0_19
; %bb.18:
	s_cmp_gt_i32 s18, 0
	v_dual_mov_b32 v9, s33 :: v_dual_lshlrev_b32 v1, 16, v1
	s_cselect_b32 s34, -1, 0
	s_lshl_b32 s24, s0, 6
	s_wait_kmcnt 0x0
	s_bfe_i64 s[28:29], s[12:13], 0x200000
	s_ashr_i32 s25, s24, 31
	v_mov_b32_e32 v7, s18
	s_mul_u64 s[24:25], s[28:29], s[24:25]
	s_lshr_b32 s28, s23, 16
	s_lshl_b64 s[24:25], s[24:25], 1
	s_and_b32 s29, s34, s17
	s_add_nc_u64 s[24:25], s[6:7], s[24:25]
	s_bitset1_b32 s28, 22
	s_and_b32 s25, s25, 0x1ffffff
	v_cndmask_b32_e64 v4, 0, 1, s29
	s_bitset1_b32 s25, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_dual_mov_b32 v6, s24 :: v_dual_mov_b32 v11, s25
	s_lshl_b32 s24, s23, 16
	v_dual_mov_b32 v5, s28 :: v_dual_mov_b32 v2, s24
	v_readfirstlane_b32 s44, v4
	v_readfirstlane_b32 s45, v9
	v_readfirstlane_b32 s46, v6
	v_readfirstlane_b32 s47, v11
	v_readfirstlane_b32 s37, v1
	v_readfirstlane_b32 s38, v2
	v_readfirstlane_b32 s39, v5
	v_readfirstlane_b32 s41, v7
	s_mov_b32 s42, 0
	s_mov_b32 s36, 0x7510000
	s_mov_b32 s40, 64
	s_mov_b32 s43, s42
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[44:47], s[36:43]
.LBB0_19:
	s_or_b32 exec_lo, exec_lo, s11
                                        ; implicit-def: $vgpr1
.LBB0_20:
	s_or_saveexec_b32 s10, s10
	s_min_i32 s22, s8, s22
	s_xor_b32 exec_lo, exec_lo, s10
	s_cbranch_execz .LBB0_22
; %bb.21:
	s_cmp_gt_i32 s18, 0
	s_mov_b32 s45, 0
	s_cselect_b32 s8, -1, 0
	s_lshl_b32 s24, s22, 6
	s_wait_kmcnt 0x0
	s_bfe_i64 s[28:29], s[4:5], 0x200000
	s_ashr_i32 s25, s24, 31
	s_lshl_b32 s11, s3, 16
	s_mul_u64 s[24:25], s[28:29], s[24:25]
	s_lshr_b32 s28, s3, 16
	s_lshl_b64 s[24:25], s[24:25], 1
	s_and_b32 s8, s8, s21
	s_add_nc_u64 s[24:25], s[26:27], s[24:25]
	s_bitset1_b32 s28, 22
	s_and_b32 s25, s25, 0x1ffffff
	v_dual_mov_b32 v6, s24 :: v_dual_lshlrev_b32 v1, 16, v1
	s_bitset1_b32 s25, 31
	v_cndmask_b32_e64 v4, 0, 1, s8
	v_dual_mov_b32 v9, s25 :: v_dual_mov_b32 v2, s11
	v_dual_mov_b32 v5, s28 :: v_dual_mov_b32 v7, s18
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_readfirstlane_b32 s44, v4
	v_readfirstlane_b32 s46, v6
	v_readfirstlane_b32 s47, v9
	v_readfirstlane_b32 s37, v1
	v_readfirstlane_b32 s38, v2
	v_readfirstlane_b32 s39, v5
	v_readfirstlane_b32 s41, v7
	s_mov_b32 s36, 0x7510000
	s_mov_b32 s40, 64
	s_mov_b32 s42, s45
	s_mov_b32 s43, s45
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[44:47], s[36:43]
.LBB0_22:
	s_or_b32 exec_lo, exec_lo, s10
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	v_dual_lshlrev_b32 v1, 3, v23 :: v_dual_lshlrev_b32 v29, 6, v0
	s_and_b32 s34, s21, s17
	s_cmp_lt_i32 s18, 1
	v_cndmask_b32_e64 v25, 0, 1, s34
	s_delay_alu instid0(VALU_DEP_2)
	v_and_b32_e32 v27, 48, v1
	s_barrier_wait -1
	s_cbranch_scc1 .LBB0_47
; %bb.23:
	s_mov_b64 s[24:25], src_shared_base
	s_or_b32 s10, s2, 0x4400
	s_mov_b32 s11, s25
	v_dual_lshlrev_b32 v2, 6, v27 :: v_dual_bitop2_b32 v1, 16, v0 bitop3:0x40
	s_and_b64 s[10:11], s[10:11], 15
	s_mov_b32 s35, s1
	s_sub_co_i32 s8, 16, s10
	s_mov_b32 s37, s25
	s_lshr_b32 s8, s8, 2
	s_cmp_lg_u64 s[10:11], 0
	v_and_or_b32 v4, 0x3c0, v29, v1
	s_cselect_b32 s8, s8, 0
	s_mov_b32 s11, 0
	s_lshl2_add_u32 s2, s8, s2
	s_mov_b32 s8, 64
	s_add_co_i32 s24, s2, 0x6600
	v_and_or_b32 v5, 0x800, v29, v4
	v_or_b32_e32 v1, v2, v4
	s_and_b32 s10, s24, 15
	s_add_co_i32 s36, s2, 0x4400
	s_sub_co_i32 s28, 16, s10
	v_lshrrev_b32_e32 v7, 4, v5
	v_or_b32_e32 v8, 0x400, v5
	v_lshrrev_b32_e32 v6, 4, v1
	s_lshr_b32 s2, s28, 2
	s_cmp_lg_u64 s[10:11], 0
	v_and_b32_e32 v7, 0xb8, v7
	s_cselect_b32 s2, s2, 0
	s_add_co_i32 s10, s18, 63
	v_lshrrev_b32_e32 v8, 4, v8
	v_and_b32_e32 v6, 0xf8, v6
	s_ashr_i32 s28, s10, 31
	v_dual_add_nc_u32 v22, v7, v5 :: v_dual_lshrrev_b32 v24, 8, v3
	s_lshr_b32 s28, s28, 26
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v21, 0 :: v_dual_add_nc_u32 v18, v6, v1
	s_add_co_i32 s28, s10, s28
	v_and_b32_e32 v1, 0xf8, v8
	s_lshl_b32 s10, s2, 2
	s_ashr_i32 s38, s28, 6
	s_cmp_lt_i32 s23, 64
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_nc_u32 v3, 2, v24 :: v_dual_add_nc_u32 v26, v5, v1
	s_cselect_b32 s39, -1, 0
	s_lshl_b32 s28, s0, 6
	s_lshl_b32 s44, s22, 6
	s_ashr_i32 s29, s28, 31
	s_ashr_i32 s45, s44, 31
	s_wait_kmcnt 0x0
	s_bfe_i64 s[12:13], s[12:13], 0x200000
	s_bfe_i64 s[4:5], s[4:5], 0x200000
	s_mul_u64 s[12:13], s[12:13], s[28:29]
	s_mul_u64 s[4:5], s[4:5], s[44:45]
	v_dual_mov_b32 v19, v24 :: v_dual_bitop2_b32 v31, 30, v3 bitop3:0x40
	v_or_b32_e32 v1, 0x100, v0
	v_add3_u32 v28, v4, v2, v6
	v_dual_mov_b32 v3, v21 :: v_dual_add_nc_u32 v30, 0x400, v26
	v_dual_mov_b32 v2, v21 :: v_dual_mov_b32 v4, v21
	v_dual_mov_b32 v5, v21 :: v_dual_mov_b32 v6, v21
	v_dual_mov_b32 v7, v21 :: v_dual_mov_b32 v8, v21
	v_dual_mov_b32 v9, v21 :: v_dual_mov_b32 v10, v21
	v_dual_mov_b32 v11, v21 :: v_dual_mov_b32 v12, v21
	v_dual_mov_b32 v13, v21 :: v_dual_mov_b32 v14, v21
	v_dual_mov_b32 v15, v21 :: v_dual_mov_b32 v16, v21
	v_mov_b32_e32 v17, v21
	s_lshr_b32 s40, s23, 16
	s_lshr_b32 s41, s3, 16
	s_lshl_b64 s[12:13], s[12:13], 1
	s_lshl_b64 s[4:5], s[4:5], 1
	s_add_nc_u64 s[24:25], s[24:25], s[10:11]
	s_bitset1_b32 s40, 22
	s_bitset1_b32 s41, 22
	s_max_i32 s42, s38, 1
	s_add_nc_u64 s[12:13], s[6:7], s[12:13]
	s_add_nc_u64 s[26:27], s[26:27], s[4:5]
	s_mov_b32 s4, 0x7510000
	s_mov_b32 s43, s11
	s_branch .LBB0_25
.LBB0_24:                               ;   in Loop: Header=BB0_25 Depth=1
	s_cmp_eq_u32 s43, s42
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_scc1 .LBB0_48
.LBB0_25:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_28 Depth 2
                                        ;     Child Loop BB0_34 Depth 2
	s_and_b32 s44, s43, 1
	s_add_co_i32 s43, s43, 1
	s_xor_b32 s45, s44, 1
	s_lshl_b32 s0, s43, 6
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_sub_co_i32 s0, s18, s0
	s_min_i32 s0, s0, 64
	s_cmp_lt_i32 s43, s38
	s_cselect_b32 s5, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_and_b32 s2, s5, exec_lo
	s_cselect_b32 s2, s0, 0
	s_cmp_lt_i32 s2, 64
	s_cselect_b32 s0, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b32 s0, s39, s0
	s_or_b32 s0, s31, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s0
	s_cbranch_vccnz .LBB0_39
; %bb.26:                               ;   in Loop: Header=BB0_25 Depth=1
	v_nop
	v_nop
	v_nop
	v_mov_b64_e32 v[32:33], v[0:1]
	s_cmp_lg_u32 s45, 0
	s_mov_b32 s6, 0
	s_cselect_b32 s29, s37, s1
	s_cselect_b32 s28, s36, 0
	s_mov_b32 s10, 0
	s_branch .LBB0_28
.LBB0_27:                               ;   in Loop: Header=BB0_28 Depth=2
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s0
	s_add_co_i32 s6, s6, 2
	v_add_nc_u32_e32 v33, 0x200, v33
	v_cmp_eq_u32_e32 vcc_lo, s6, v31
	v_add_nc_u32_e32 v32, 0x200, v32
	s_or_b32 s10, vcc_lo, s10
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s10
	s_cbranch_execz .LBB0_32
.LBB0_28:                               ;   Parent Loop BB0_25 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_mov_b32 s7, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b64 s[46:47], s[6:7], 0x100000000
	s_mov_b32 s7, exec_lo
	v_cmp_le_u32_e32 vcc_lo, s47, v19
	v_cmpx_le_u32_e64 s46, v24
	s_cbranch_execz .LBB0_30
; %bb.29:                               ;   in Loop: Header=BB0_28 Depth=2
	v_mov_b32_e32 v20, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u64 v[34:35], v[20:21], 2, s[28:29]
	flat_store_b32 v[34:35], v21
.LBB0_30:                               ;   in Loop: Header=BB0_28 Depth=2
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s7
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB0_27
; %bb.31:                               ;   in Loop: Header=BB0_28 Depth=2
	v_mov_b32_e32 v20, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u64 v[34:35], v[20:21], 2, s[28:29]
	flat_store_b32 v[34:35], v21
	s_branch .LBB0_27
.LBB0_32:                               ;   in Loop: Header=BB0_25 Depth=1
	s_or_b32 exec_lo, exec_lo, s10
	v_mov_b64_e32 v[32:33], v[0:1]
	s_cmp_lg_u32 s45, 0
	s_mov_b32 s6, 0
	s_cselect_b32 s29, s25, s35
	s_cselect_b32 s28, s24, s33
	s_mov_b32 s10, 0
	s_branch .LBB0_34
.LBB0_33:                               ;   in Loop: Header=BB0_34 Depth=2
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s0
	s_add_co_i32 s6, s6, 2
	v_add_nc_u32_e32 v33, 0x200, v33
	v_cmp_eq_u32_e32 vcc_lo, s6, v31
	v_add_nc_u32_e32 v32, 0x200, v32
	s_or_b32 s10, vcc_lo, s10
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s10
	s_cbranch_execz .LBB0_38
.LBB0_34:                               ;   Parent Loop BB0_25 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_mov_b32 s7, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b64 s[46:47], s[6:7], 0x100000000
	s_mov_b32 s7, exec_lo
	v_cmp_le_u32_e32 vcc_lo, s47, v19
	v_cmpx_le_u32_e64 s46, v24
	s_cbranch_execz .LBB0_36
; %bb.35:                               ;   in Loop: Header=BB0_34 Depth=2
	v_mov_b32_e32 v20, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u64 v[34:35], v[20:21], 2, s[28:29]
	flat_store_b32 v[34:35], v21
.LBB0_36:                               ;   in Loop: Header=BB0_34 Depth=2
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s7
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB0_33
; %bb.37:                               ;   in Loop: Header=BB0_34 Depth=2
	v_mov_b32_e32 v20, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u64 v[34:35], v[20:21], 2, s[28:29]
	flat_store_b32 v[34:35], v21
	s_branch .LBB0_33
.LBB0_38:                               ;   in Loop: Header=BB0_25 Depth=1
	s_or_b32 exec_lo, exec_lo, s10
	s_barrier_signal -1
	s_barrier_wait -1
.LBB0_39:                               ;   in Loop: Header=BB0_25 Depth=1
	s_and_b32 s0, s5, exec_lo
	s_cselect_b32 s0, s43, 0
	s_mov_b32 s5, exec_lo
	v_cmpx_lt_i32_e32 0, v23
	s_xor_b32 s28, exec_lo, s5
	s_cbranch_execnz .LBB0_42
; %bb.40:                               ;   in Loop: Header=BB0_25 Depth=1
	s_and_not1_saveexec_b32 s22, s28
	s_cbranch_execnz .LBB0_45
.LBB0_41:                               ;   in Loop: Header=BB0_25 Depth=1
	s_or_b32 exec_lo, exec_lo, s22
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s34
	s_cbranch_vccnz .LBB0_24
	s_branch .LBB0_46
.LBB0_42:                               ;   in Loop: Header=BB0_25 Depth=1
	s_mov_b32 s29, exec_lo
	v_cmpx_eq_u32_e32 1, v23
	s_cbranch_execz .LBB0_44
; %bb.43:                               ;   in Loop: Header=BB0_25 Depth=1
	s_cmp_lg_u32 s45, 0
	s_cselect_b32 s5, s24, s33
	s_cmp_gt_i32 s2, 0
	s_cselect_b32 s22, -1, 0
	s_lshl_b32 s10, s0, 6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_lshl_b64 s[6:7], s[10:11], 1
	s_mov_b32 s10, s11
	s_add_nc_u64 s[6:7], s[12:13], s[6:7]
	v_nop
	v_dual_mov_b32 v33, s5 :: v_dual_mov_b32 v32, s6
	s_and_b32 s5, s7, 0x1ffffff
	s_and_b32 s7, s17, s22
	s_bitset1_b32 s5, 31
	v_cndmask_b32_e64 v20, 0, 1, s7
	v_mov_b32_e32 v35, s5
	s_mov_b32 s22, s2
	v_readfirstlane_b32 s49, v33
	v_readfirstlane_b32 s50, v32
	v_readfirstlane_b32 s48, v20
	v_readfirstlane_b32 s51, v35
	s_lshr_b64 s[6:7], s[22:23], 16
	s_lshl_b32 s5, s2, 16
	s_mov_b32 s7, s40
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[48:51], s[4:11]
.LBB0_44:                               ;   in Loop: Header=BB0_25 Depth=1
	s_or_b32 exec_lo, exec_lo, s29
	s_and_not1_saveexec_b32 s22, s28
	s_cbranch_execz .LBB0_41
.LBB0_45:                               ;   in Loop: Header=BB0_25 Depth=1
	s_cmp_lg_u32 s45, 0
	s_cselect_b32 s5, s36, 0
	s_cmp_gt_i32 s2, 0
	s_cselect_b32 s28, -1, 0
	s_lshl_b32 s10, s0, 6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_lshl_b64 s[6:7], s[10:11], 1
	s_mov_b32 s10, s11
	s_add_nc_u64 s[6:7], s[26:27], s[6:7]
	v_nop
	v_nop
	v_dual_mov_b32 v33, s5 :: v_dual_mov_b32 v32, s6
	s_and_b32 s0, s7, 0x1ffffff
	s_and_b32 s7, s21, s28
	s_bitset1_b32 s0, 31
	v_cndmask_b32_e64 v20, 0, 1, s7
	v_mov_b32_e32 v35, s0
	v_readfirstlane_b32 s49, v33
	v_readfirstlane_b32 s50, v32
	s_lshr_b64 s[6:7], s[2:3], 16
	v_readfirstlane_b32 s48, v20
	v_readfirstlane_b32 s51, v35
	s_lshl_b32 s5, s2, 16
	s_mov_b32 s7, s41
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	tensor_load_to_lds s[48:51], s[4:11]
	s_or_b32 exec_lo, exec_lo, s22
	s_and_not1_b32 vcc_lo, exec_lo, s34
	s_cbranch_vccnz .LBB0_24
.LBB0_46:                               ;   in Loop: Header=BB0_25 Depth=1
	s_cmp_lg_u32 s44, 0
	s_cselect_b32 s0, s36, 0
	s_cselect_b32 s2, s24, s33
	v_lshl_add_u32 v20, v18, 1, s0
	v_lshl_add_u32 v56, v22, 1, s2
	v_lshl_add_u32 v57, v30, 1, s2
	ds_load_b128 v[32:35], v20
	ds_load_b128 v[36:39], v20 offset:16
	ds_load_b128 v[40:43], v56
	ds_load_b128 v[44:47], v56 offset:16
	v_lshl_add_u32 v20, v26, 1, s2
	; sched_group_barrier mask(0x00000100) size(3) SyncID(0)
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[2:9], v[32:39], v[40:47], v[2:9]
	ds_load_b128 v[40:43], v20 offset:2048
	ds_load_b128 v[44:47], v20 offset:2064
	v_lshl_add_u32 v20, v28, 1, s0
	ds_load_b128 v[48:51], v20 offset:64
	ds_load_b128 v[52:55], v20 offset:80
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[10:17], v[32:39], v[40:47], v[10:17] matrix_a_reuse
	ds_load_b128 v[32:35], v56 offset:64
	ds_load_b128 v[36:39], v56 offset:80
	ds_load_b128 v[40:43], v57 offset:64
	ds_load_b128 v[44:47], v57 offset:80
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(3) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[2:9], v[48:55], v[32:39], v[2:9]
	; sched_group_barrier mask(0x00000100) size(3) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(3) SyncID(0)
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[10:17], v[48:55], v[40:47], v[10:17] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_wait_dscnt 0
	;;#ASMEND
	s_branch .LBB0_24
.LBB0_47:
	v_mov_b32_e32 v9, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v8, v9 :: v_dual_mov_b32 v7, v9
	v_dual_mov_b32 v6, v9 :: v_dual_mov_b32 v5, v9
	v_dual_mov_b32 v4, v9 :: v_dual_mov_b32 v3, v9
	v_dual_mov_b32 v2, v9 :: v_dual_mov_b32 v17, v9
	v_dual_mov_b32 v16, v9 :: v_dual_mov_b32 v15, v9
	v_dual_mov_b32 v14, v9 :: v_dual_mov_b32 v13, v9
	v_dual_mov_b32 v12, v9 :: v_dual_mov_b32 v11, v9
	v_mov_b32_e32 v10, v9
.LBB0_48:
	s_wait_tensorcnt 0x0
	;;#ASMSTART
	s_wait_dscnt 0
	;;#ASMEND
	s_barrier_signal -1
	s_and_b32 vcc_lo, exec_lo, s34
	s_barrier_wait -1
	s_cbranch_vccz .LBB0_50
; %bb.49:
	v_dual_lshrrev_b32 v1, 1, v0 :: v_dual_lshlrev_b32 v19, 3, v0
	v_and_b32_e32 v18, 0xbc0, v29
	v_nop
	v_cvt_pk_bf16_f32 v9, v8, v9
	v_cvt_pk_bf16_f32 v8, v6, v7
	v_cvt_pk_bf16_f32 v7, v4, v5
	v_cvt_pk_bf16_f32 v5, v16, v17
	v_or_b32_e32 v20, 0x400, v18
	v_and_b32_e32 v1, 8, v1
	v_and_b32_e32 v16, 0x170, v19
	v_cvt_pk_bf16_f32 v6, v2, v3
	v_cvt_pk_bf16_f32 v2, v10, v11
	v_lshrrev_b32_e32 v4, 3, v20
	v_or3_b32 v1, v27, v1, v18
	v_cvt_pk_bf16_f32 v3, v12, v13
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_and_b32_e32 v17, 0x1f0, v4
	v_lshlrev_b32_e32 v1, 1, v1
	v_cvt_pk_bf16_f32 v4, v14, v15
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_nc_u32 v10, v16, v1 :: v_dual_add_nc_u32 v1, v17, v1
	ds_store_b128 v10, v[6:9]
	ds_store_b128 v1, v[2:5] offset:2048
.LBB0_50:
	;;#ASMSTART
	s_wait_dscnt 0
	;;#ASMEND
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc_lo, 1, v25
	s_barrier_wait -1
	s_cbranch_vccnz .LBB0_61
; %bb.51:
	s_mul_i32 s3, s23, s3
	s_mov_b32 s0, exec_lo
	v_cmpx_gt_i32_e64 s3, v0
	s_cbranch_execz .LBB0_61
; %bb.52:
	s_ashr_i32 s21, s20, 31
	v_nop
	v_xad_u32 v2, v0, -1, s3
	s_lshl_b64 s[0:1], s[20:21], 1
	s_ashr_i32 s17, s16, 31
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[4:5], s[14:15], s[0:1]
	s_mov_b32 s0, 0
                                        ; implicit-def: $vgpr1
                                        ; implicit-def: $vgpr6
                                        ; implicit-def: $sgpr12_sgpr13
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_u32_e32 0x2ff, v2
	s_xor_b32 s14, exec_lo, s1
	s_cbranch_execnz .LBB0_55
; %bb.53:
	s_or_saveexec_b32 s1, s14
	v_mov_b64_e32 v[2:3], s[12:13]
	s_xor_b32 exec_lo, exec_lo, s1
	s_cbranch_execnz .LBB0_58
.LBB0_54:
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 exec_lo, exec_lo, s0
	s_cbranch_execnz .LBB0_59
	s_branch .LBB0_61
.LBB0_55:
	s_abs_i32 s15, s19
	v_dual_lshrrev_b32 v1, 8, v2 :: v_dual_mov_b32 v7, 0
	s_cvt_f32_u32 s0, s15
	v_or_b32_e32 v3, 0x300, v0
	s_sub_co_i32 s1, 0, s15
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v8, 1, v1
	v_rcp_iflag_f32_e32 v2, s0
	v_or_b32_e32 v1, 0x100, v0
	s_mov_b32 s13, 0
	s_mov_b32 s6, s16
	v_and_b32_e32 v9, 0x1fffffc, v8
	s_mov_b32 s7, s17
	s_mov_b32 s8, s16
	s_mov_b32 s9, s17
	v_readfirstlane_b32 s0, v2
	v_or_b32_e32 v2, 0x200, v0
	v_mov_b32_e32 v10, v9
	s_mov_b32 s10, s16
	s_mov_b32 s11, s17
	s_mul_f32 s0, s0, 0x4f7ffffe
	v_mov_b64_e32 v[4:5], v[2:3]
	v_mov_b64_e32 v[2:3], v[0:1]
	s_mov_b32 s18, s19
	s_cvt_u32_f32 s0, s0
	s_mov_b32 s20, s19
	s_mov_b32 s21, s19
	s_mov_b32 s22, s30
	s_mul_i32 s1, s1, s0
	s_mov_b32 s23, s30
	s_mul_hi_u32 s1, s0, s1
	s_mov_b32 s24, s30
	s_ashr_i32 s25, s19, 31
	s_add_co_i32 s12, s0, s1
	s_mov_b32 s26, s13
.LBB0_56:                               ; =>This Inner Loop Header: Depth=1
	v_dual_ashrrev_i32 v11, 31, v3 :: v_dual_sub_nc_u32 v12, 0, v3
	v_dual_ashrrev_i32 v1, 31, v2 :: v_dual_sub_nc_u32 v6, 0, v2
	v_dual_ashrrev_i32 v18, 31, v5 :: v_dual_sub_nc_u32 v19, 0, v5
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_ashrrev_i32 v14, 31, v4 :: v_dual_max_i32 v12, v12, v3
	v_dual_mov_b32 v17, v7 :: v_dual_sub_nc_u32 v16, 0, v4
	v_add_nc_u32_e32 v10, -4, v10
	v_mul_hi_u32 v21, v12, s12
	v_dual_mov_b32 v13, v7 :: v_dual_max_i32 v6, v6, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mov_b32 v15, v7 :: v_dual_add_nc_u32 v29, 1, v21
	v_mul_hi_u32 v20, v6, s12
	v_mul_lo_u32 v26, v21, s15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_add_nc_u32 v25, 1, v20 :: v_dual_bitop2_b32 v1, s25, v1 bitop3:0x14
	v_mul_lo_u32 v24, v20, s15
	v_dual_sub_nc_u32 v6, v6, v24 :: v_dual_max_i32 v19, v19, v5
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_sub_nc_u32_e32 v12, v12, v26
	v_cmp_le_u32_e32 vcc_lo, s15, v6
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mul_hi_u32 v23, v19, s12
	v_cmp_le_u32_e64 s0, s15, v12
	v_max_i32_e32 v16, v16, v4
	v_subrev_nc_u32_e32 v24, s15, v6
	v_dual_cndmask_b32 v20, v20, v25, vcc_lo :: v_dual_bitop2_b32 v14, s25, v14 bitop3:0x14
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v21, v21, v29, s0
	v_mul_hi_u32 v22, v16, s12
	v_mul_lo_u32 v28, v23, s15
	v_subrev_nc_u32_e32 v25, s15, v12
	v_xor_b32_e32 v11, s25, v11
	v_dual_cndmask_b32 v6, v6, v24 :: v_dual_add_nc_u32 v31, 1, v23
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_add_nc_u32 v24, 1, v20 :: v_dual_cndmask_b32 v12, v12, v25, s0
	v_mul_lo_u32 v27, v22, s15
	v_dual_sub_nc_u32 v19, v19, v28 :: v_dual_bitop2_b32 v18, s25, v18 bitop3:0x14
	v_dual_add_nc_u32 v30, 1, v22 :: v_dual_add_nc_u32 v25, 1, v21
	v_cmp_le_u32_e32 vcc_lo, s15, v12
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cmp_le_u32_e64 s2, s15, v19
	v_sub_nc_u32_e32 v16, v16, v27
	v_subrev_nc_u32_e32 v27, s15, v19
	v_cndmask_b32_e64 v23, v23, v31, s2
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_cmp_le_u32_e64 s1, s15, v16
	v_subrev_nc_u32_e32 v26, s15, v16
	v_cndmask_b32_e64 v19, v19, v27, s2
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_add_nc_u32 v27, 1, v23 :: v_dual_cndmask_b32 v22, v22, v30, s1
	v_cndmask_b32_e64 v16, v16, v26, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_add_nc_u32 v26, 1, v22 :: v_dual_cndmask_b32 v12, v21, v25, vcc_lo
	v_cmp_le_u32_e32 vcc_lo, s15, v16
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v16, v22, v26, vcc_lo
	v_cmp_le_u32_e32 vcc_lo, s15, v6
	v_cndmask_b32_e32 v6, v20, v24, vcc_lo
	v_cmp_le_u32_e32 vcc_lo, s15, v19
	v_dual_cndmask_b32 v19, v23, v27, vcc_lo :: v_dual_bitop2_b32 v6, v6, v1 bitop3:0x14
	v_xor_b32_e32 v16, v16, v14
	v_cmp_eq_u32_e32 vcc_lo, 0, v10
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_xor_b32_e32 v19, v19, v18
	v_dual_sub_nc_u32 v1, v6, v1 :: v_dual_bitop2_b32 v12, v12, v11 bitop3:0x14
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_sub_nc_u32_e32 v26, v16, v14
	s_or_b32 s26, vcc_lo, s26
	v_dual_sub_nc_u32 v27, v19, v18 :: v_dual_sub_nc_u32 v11, v12, v11
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mul_lo_u32 v6, v1, s19
	v_mul_lo_u32 v14, v26, s20
	v_add_nc_u32_e32 v18, s30, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_mul_lo_u32 v16, v27, s21
	v_mul_lo_u32 v12, v11, s18
	v_dual_add_nc_u32 v20, s22, v11 :: v_dual_add_nc_u32 v22, s23, v26
	v_dual_add_nc_u32 v24, s24, v27 :: v_dual_sub_nc_u32 v6, v2, v6
	v_sub_nc_u32_e32 v14, v4, v14
	v_dual_ashrrev_i32 v19, 31, v18 :: v_dual_ashrrev_i32 v21, 31, v20
	v_sub_nc_u32_e32 v12, v3, v12
	v_dual_sub_nc_u32 v16, v5, v16 :: v_dual_ashrrev_i32 v23, 31, v22
	v_ashrrev_i32_e32 v25, 31, v24
	v_lshl_add_u32 v1, v1, 6, v6
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v11, v11, 6, v12
	v_lshl_add_u32 v26, v26, 6, v14
	v_mul_u64_e32 v[18:19], s[16:17], v[18:19]
	v_lshl_add_u32 v27, v27, 6, v16
	v_mul_u64_e32 v[20:21], s[6:7], v[20:21]
	v_mul_u64_e32 v[22:23], s[8:9], v[22:23]
	v_dual_ashrrev_i32 v28, 31, v1 :: v_dual_ashrrev_i32 v29, 31, v11
	v_mul_u64_e32 v[24:25], s[10:11], v[24:25]
	v_dual_ashrrev_i32 v30, 31, v26 :: v_dual_ashrrev_i32 v31, 31, v27
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_lshrrev_b32 v28, 25, v28 :: v_dual_lshrrev_b32 v29, 25, v29
	v_dual_lshlrev_b32 v32, 1, v1 :: v_dual_lshlrev_b32 v33, 1, v11
	v_dual_lshrrev_b32 v30, 25, v30 :: v_dual_lshrrev_b32 v31, 25, v31
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_add_nc_u32 v1, v1, v28 :: v_dual_add_nc_u32 v11, v11, v29
	v_dual_lshlrev_b32 v34, 1, v26 :: v_dual_lshlrev_b32 v35, 1, v27
	v_dual_add_nc_u32 v26, v26, v30 :: v_dual_add_nc_u32 v27, v27, v31
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_ashrrev_i32 v1, 7, v1 :: v_dual_ashrrev_i32 v11, 7, v11
	v_add_nc_u32_e32 v5, 0x400, v5
	v_dual_ashrrev_i32 v26, 7, v26 :: v_dual_ashrrev_i32 v27, 7, v27
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u32 v1, v1, 4, v32
	v_lshl_add_u32 v11, v11, 4, v33
	v_add_nc_u32_e32 v4, 0x400, v4
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v26, v26, 4, v34
	v_lshl_add_u32 v27, v27, 4, v35
	ds_load_u16 v1, v1
	ds_load_u16 v11, v11
	ds_load_u16 v26, v26
	ds_load_u16 v27, v27
	v_add_nc_u32_e32 v3, 0x400, v3
	v_add_nc_u32_e32 v2, 0x400, v2
	v_lshl_add_u64 v[18:19], v[18:19], 1, s[4:5]
	v_lshl_add_u64 v[20:21], v[20:21], 1, s[4:5]
	v_lshl_add_u64 v[22:23], v[22:23], 1, s[4:5]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_lshl_add_u64 v[18:19], v[6:7], 1, v[18:19]
	v_lshl_add_u64 v[24:25], v[24:25], 1, s[4:5]
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[20:21]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u64 v[14:15], v[14:15], 1, v[22:23]
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
	s_cbranch_execnz .LBB0_56
; %bb.57:
	s_or_b32 exec_lo, exec_lo, s26
	v_cmp_ne_u32_e32 vcc_lo, v8, v9
	v_lshl_or_b32 v0, v9, 8, v0
	v_dual_mov_b32 v6, s15 :: v_dual_mov_b32 v1, s25
	s_and_b32 s0, vcc_lo, exec_lo
	s_or_saveexec_b32 s1, s14
	v_mov_b64_e32 v[2:3], s[12:13]
	s_xor_b32 exec_lo, exec_lo, s1
	s_cbranch_execz .LBB0_54
.LBB0_58:
	s_abs_i32 s2, s19
	s_ashr_i32 s8, s19, 31
	s_cvt_f32_u32 s6, s2
	s_sub_co_i32 s7, 0, s2
	s_or_b32 s0, s0, exec_lo
	v_mov_b32_e32 v6, s2
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
	s_cbranch_execz .LBB0_61
.LBB0_59:
	v_mov_b32_e32 v5, 0
	s_mov_b32 s0, 0
	s_sub_co_i32 s1, 0, s19
.LBB0_60:                               ; =>This Inner Loop Header: Depth=1
	v_sub_nc_u32_e32 v4, 0, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_i32_e32 v4, v4, v0
	v_mul_u64_e32 v[8:9], v[4:5], v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v7, v9, v6
	v_dual_add_nc_u32 v8, 1, v9 :: v_dual_sub_nc_u32 v4, v4, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_ge_u32_e32 vcc_lo, v4, v6
	v_cndmask_b32_e32 v8, v9, v8, vcc_lo
	v_dual_ashrrev_i32 v9, 31, v0 :: v_dual_sub_nc_u32 v7, v4, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_cndmask_b32 v4, v4, v7, vcc_lo :: v_dual_bitop2_b32 v9, v9, v1 bitop3:0x14
	v_add_nc_u32_e32 v7, 1, v8
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_ge_u32_e32 vcc_lo, v4, v6
	v_cndmask_b32_e32 v4, v8, v7, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_xor_b32_e32 v4, v4, v9
	v_dual_sub_nc_u32 v7, v4, v9 :: v_dual_lshlrev_b32 v4, 6, v4
	v_lshlrev_b32_e32 v9, 6, v9
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v8, v7, s19
	v_dual_sub_nc_u32 v4, v4, v8 :: v_dual_add_nc_u32 v8, s30, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_sub_nc_u32 v4, v4, v9 :: v_dual_ashrrev_i32 v9, 31, v8
	v_add_nc_u32_e32 v4, v0, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_u64_e32 v[8:9], s[16:17], v[8:9]
	v_ashrrev_i32_e32 v10, 31, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshrrev_b32_e32 v10, 25, v10
	v_add_nc_u32_e32 v10, v4, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_ashrrev_i32 v10, 7, v10 :: v_dual_lshlrev_b32 v4, 1, v4
	v_lshl_add_u32 v4, v10, 4, v4
	ds_load_u16 v10, v4
	v_mad_u32 v4, s1, v7, v0
	v_add_nc_u32_e32 v0, 0x100, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_le_i32_e32 vcc_lo, s3, v0
	v_lshl_add_u64 v[8:9], v[8:9], 1, s[4:5]
	s_or_b32 s0, vcc_lo, s0
	v_lshl_add_u64 v[8:9], v[4:5], 1, v[8:9]
	s_wait_dscnt 0x0
	global_store_b16 v[8:9], v10, off
	s_wait_xcnt 0x0
	s_and_not1_b32 exec_lo, exec_lo, s0
	s_cbranch_execnz .LBB0_60
.LBB0_61:
	s_endpgm
.Lfunc_end0:
	.size	bm064_bn064_bk064_wm4_wn2_mc0, .Lfunc_end0-bm064_bn064_bk064_wm4_wn2_mc0
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel bm064_bn064_bk064_wm4_wn2_mc0
		.amdhsa_group_segment_fixed_size 34816
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
		.amdhsa_next_free_vgpr 58
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
		.amdhsa_inst_pref_size 37
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text.bm064_bn064_bk064_wm4_wn2_mc0,"axG",@progbits,bm064_bn064_bk064_wm4_wn2_mc0,comdat
                                        ; -- End function
	.set .Lbm064_bn064_bk064_wm4_wn2_mc0.num_vgpr, 58
	.set .Lbm064_bn064_bk064_wm4_wn2_mc0.num_agpr, 0
	.set .Lbm064_bn064_bk064_wm4_wn2_mc0.numbered_sgpr, 52
	.set .Lbm064_bn064_bk064_wm4_wn2_mc0.num_named_barrier, 0
	.set .Lbm064_bn064_bk064_wm4_wn2_mc0.private_seg_size, 0
	.set .Lbm064_bn064_bk064_wm4_wn2_mc0.uses_vcc, 1
	.set .Lbm064_bn064_bk064_wm4_wn2_mc0.uses_flat_scratch, 0
	.set .Lbm064_bn064_bk064_wm4_wn2_mc0.has_dyn_sized_stack, 0
	.set .Lbm064_bn064_bk064_wm4_wn2_mc0.has_recursion, 0
	.set .Lbm064_bn064_bk064_wm4_wn2_mc0.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 4656
; TotalNumSgprs: 54
; NumVgprs: 58
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 34816 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 3
; NumSGPRsForWavesPerEU: 54
; NumVGPRsForWavesPerEU: 58
; NamedBarCnt: 0
; Occupancy: 16
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
	.type	__hip_cuid_e3b1d5c4ccec3c52,@object ; @__hip_cuid_e3b1d5c4ccec3c52
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_e3b1d5c4ccec3c52
__hip_cuid_e3b1d5c4ccec3c52:
	.byte	0                               ; 0x0
	.size	__hip_cuid_e3b1d5c4ccec3c52, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git aa451e1fe6a793394d6733051b1778633063ae96+PATCHED:d17c5aa0e3ea29cde402f58f27e39b6034effa27)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_e3b1d5c4ccec3c52
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
    .group_segment_fixed_size: 34816
    .kernarg_segment_align: 8
    .kernarg_segment_size: 132
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           bm064_bn064_bk064_wm4_wn2_mc0
    .private_segment_fixed_size: 0
    .sgpr_count:     54
    .sgpr_spill_count: 0
    .symbol:         bm064_bn064_bk064_wm4_wn2_mc0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     58
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
