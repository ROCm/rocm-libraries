	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.section	.text.bm256_bn096_bk032_wm8_wn1_mc0,"axG",@progbits,bm256_bn096_bk032_wm8_wn1_mc0,comdat
	.protected	bm256_bn096_bk032_wm8_wn1_mc0 ; -- Begin function bm256_bn096_bk032_wm8_wn1_mc0
	.globl	bm256_bn096_bk032_wm8_wn1_mc0
	.p2align	8
	.type	bm256_bn096_bk032_wm8_wn1_mc0,@function
bm256_bn096_bk032_wm8_wn1_mc0: ; @bm256_bn096_bk032_wm8_wn1_mc0
; %bb.0:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1 ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_mov_b64 s[2:3], src_shared_base
	s_movk_i32 s2, 0x4400
	s_load_b96 s[24:26], s[0:1], 0x78 nv
	s_and_b64 s[2:3], s[2:3], 12
	s_getreg_b32 s5, hwreg(HW_REG_IB_STS2, 6, 4)
	s_sub_co_i32 s4, 16, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_lshr_b32 s4, s4, 2
	s_cmp_lg_u64 s[2:3], 0
	s_cselect_b32 s6, s4, 0
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
	s_mov_b32 s9, s26
	s_cselect_b32 s35, ttmp9, s3
	s_cselect_b32 s33, ttmp7, s4
	s_add_co_i32 s2, s24, 0xff
	s_add_co_i32 s4, s25, 0x5f
	s_ashr_i32 s3, s2, 31
	s_lshl_b32 s28, s35, 8
	s_lshr_b32 s3, s3, 24
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_co_i32 s2, s2, s3
	s_mul_hi_i32 s3, s4, 0x2aaaaaab
	s_ashr_i32 s7, s2, 8
	s_lshr_b32 s2, s3, 31
	s_ashr_i32 s8, s3, 4
	s_sub_co_i32 s3, s24, s28
	s_add_co_i32 s8, s8, s2
	s_min_i32 s27, s3, 0x100
	s_cmp_lt_i32 s35, s7
	s_cselect_b32 s29, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_and_b32 s2, s29, exec_lo
	s_mul_i32 s2, s33, 0xffffffa0
	s_cselect_b32 s31, s27, 0
	s_add_co_i32 s2, s25, s2
	s_min_i32 s2, s2, 0x60
	s_cmp_lt_i32 s33, s8
	s_cselect_b32 s25, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s3, s25, exec_lo
	s_cselect_b32 s3, s2, 0
	s_add_co_i32 s17, s26, 31
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 2, 1), 1
	s_min_i32 s2, s26, 32
	s_cmp_gt_i32 s17, 31
	s_cselect_b32 s16, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_and_b32 s4, s16, exec_lo
	s_cselect_b32 s30, s2, 0
	s_cmp_lt_i32 s31, 0x100
	s_cselect_b32 s38, -1, 0
	s_and_b32 vcc_lo, exec_lo, s38
	s_mov_b32 s2, s38
	s_cbranch_vccnz .LBB0_2
; %bb.1:
	s_cmp_lt_i32 s3, 0x60
	s_cselect_b32 s2, -1, 0
	s_cmp_lt_i32 s30, 32
	s_cselect_b32 s4, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 s2, s4, s2
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
	v_cmp_lt_u32_e32 vcc_lo, 0xfff, v2
	s_or_b32 s2, vcc_lo, s2
	s_and_not1_b32 exec_lo, exec_lo, s2
	s_cbranch_execnz .LBB0_4
; %bb.5:
	s_or_b32 exec_lo, exec_lo, s2
	v_dual_mov_b32 v5, 0 :: v_dual_sub_nc_u32 v2, 0x65f, v0
	v_lshl_add_u32 v3, s6, 2, v1
	s_mov_b32 s4, 0
	s_mov_b32 s10, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshrrev_b32_e32 v2, 8, v2
	v_add_nc_u32_e32 v4, 2, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v1, v2 :: v_dual_bitop2_b32 v4, 14, v4 bitop3:0x40
	s_branch .LBB0_7
.LBB0_6:                                ;   in Loop: Header=BB0_7 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
	s_add_co_i32 s4, s4, 2
	v_add_nc_u32_e32 v3, 0x800, v3
	v_cmp_eq_u32_e32 vcc_lo, s4, v4
	s_or_b32 s10, vcc_lo, s10
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s10
	s_cbranch_execz .LBB0_11
.LBB0_7:                                ; =>This Inner Loop Header: Depth=1
	s_mov_b32 s5, s4
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b64 s[12:13], s[4:5], 0x100000000
	s_mov_b32 s5, exec_lo
	v_cmp_le_u32_e32 vcc_lo, s13, v1
	v_cmpx_le_u32_e64 s12, v2
; %bb.8:                                ;   in Loop: Header=BB0_7 Depth=1
	ds_store_b32 v3, v5 offset:17408
; %bb.9:                                ;   in Loop: Header=BB0_7 Depth=1
	s_or_b32 exec_lo, exec_lo, s5
	s_and_saveexec_b32 s2, vcc_lo
	s_cbranch_execz .LBB0_6
; %bb.10:                               ;   in Loop: Header=BB0_7 Depth=1
	ds_store_b32 v3, v5 offset:18432
	s_branch .LBB0_6
.LBB0_11:
	s_or_b32 exec_lo, exec_lo, s10
	s_barrier_signal -1
	s_barrier_wait -1
.LBB0_12:
	s_clause 0x2
	s_load_b64 s[18:19], s[0:1], 0x0 nv
	s_load_b128 s[12:15], s[0:1], 0x20 nv
	s_load_b128 s[20:23], s[0:1], 0x48 nv
	s_wait_xcnt 0x0
	s_mov_b64 s[0:1], src_shared_base
	v_lshrrev_b32_e32 v105, 5, v0
	s_lshl_b32 s0, s6, 2
	s_add_co_i32 s8, s8, -1
	s_or_b32 s39, s0, 0x4400
	s_add_co_i32 s36, s7, -1
	s_min_i32 s34, s33, s8
	s_mov_b32 s2, exec_lo
	v_cmpx_lt_i32_e32 0, v105
	s_xor_b32 s37, exec_lo, s2
	s_cbranch_execz .LBB0_16
; %bb.13:
	s_mov_b32 s40, exec_lo
	v_cmpx_eq_u32_e32 1, v105
	s_cbranch_execz .LBB0_15
; %bb.14:
	s_cmp_gt_i32 s30, 0
	s_mul_i32 s4, s34, 0x60
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
	v_dual_mov_b32 v1, s39 :: v_dual_mov_b32 v4, s4
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
	s_movk_i32 s8, 0x60
	s_lshl_b32 s5, s30, 16
	s_or_b32 s7, s2, 0x200000
	s_mov_b32 s4, 0x7510000
	s_mov_b32 s11, s10
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[44:47], s[4:11]
.LBB0_15:
	s_or_b32 exec_lo, exec_lo, s40
.LBB0_16:
	s_or_saveexec_b32 s37, s37
	s_min_i32 s2, s35, s36
	s_xor_b32 exec_lo, exec_lo, s37
	s_cbranch_execz .LBB0_18
; %bb.17:
	s_cmp_gt_i32 s30, 0
	s_mov_b32 s10, 0
	s_cselect_b32 s8, -1, 0
	s_lshl_b32 s4, s2, 8
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
	v_readfirstlane_b32 s40, v2
	s_bitset1_b32 s7, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v4, s6 :: v_dual_mov_b32 v3, s7
	s_lshr_b64 s[6:7], s[30:31], 16
	s_or_b32 s7, s4, 0x200000
	s_movk_i32 s8, 0x100
	v_readfirstlane_b32 s42, v4
	v_readfirstlane_b32 s43, v3
	s_mov_b32 s4, 0x7510000
	s_mov_b32 s11, s10
	s_mov_b32 s41, s10
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[40:43], s[4:11]
.LBB0_18:
	s_or_b32 exec_lo, exec_lo, s37
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	v_mov_b32_e32 v9, 0
	s_and_b32 s40, s29, s25
	s_and_not1_b32 vcc_lo, exec_lo, s16
	v_cndmask_b32_e64 v103, 0, 1, s40
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
	v_dual_mov_b32 v34, v9 :: v_dual_mov_b32 v49, v9
	v_dual_mov_b32 v48, v9 :: v_dual_mov_b32 v47, v9
	v_dual_mov_b32 v46, v9 :: v_dual_mov_b32 v45, v9
	v_dual_mov_b32 v44, v9 :: v_dual_mov_b32 v43, v9
	v_dual_mov_b32 v42, v9 :: v_dual_mov_b32 v81, v9
	v_dual_mov_b32 v80, v9 :: v_dual_mov_b32 v79, v9
	v_dual_mov_b32 v78, v9 :: v_dual_mov_b32 v77, v9
	v_dual_mov_b32 v76, v9 :: v_dual_mov_b32 v75, v9
	v_dual_mov_b32 v74, v9 :: v_dual_mov_b32 v89, v9
	v_dual_mov_b32 v88, v9 :: v_dual_mov_b32 v87, v9
	v_dual_mov_b32 v86, v9 :: v_dual_mov_b32 v85, v9
	v_dual_mov_b32 v84, v9 :: v_dual_mov_b32 v83, v9
	v_dual_mov_b32 v82, v9 :: v_dual_mov_b32 v97, v9
	v_dual_mov_b32 v96, v9 :: v_dual_mov_b32 v95, v9
	v_dual_mov_b32 v94, v9 :: v_dual_mov_b32 v93, v9
	v_dual_mov_b32 v92, v9 :: v_dual_mov_b32 v91, v9
	v_dual_mov_b32 v90, v9 :: v_dual_mov_b32 v57, v9
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
	v_mov_b32_e32 v66, v9
	s_barrier_wait -1
	s_cbranch_vccnz .LBB0_42
; %bb.19:
	v_dual_lshlrev_b32 v3, 5, v0 :: v_dual_lshlrev_b32 v6, 1, v0
	v_dual_lshlrev_b32 v1, 10, v105 :: v_dual_bitop2_b32 v2, 16, v0 bitop3:0x40
	s_mov_b64 s[4:5], src_shared_base
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_and_b32_e32 v3, 0x1e0, v3
	s_or_b32 s6, s0, 0x5d80
	s_mov_b32 s7, s5
	v_dual_mov_b32 v101, 0 :: v_dual_bitop2_b32 v6, 24, v6 bitop3:0x40
	v_or3_b32 v4, v1, v2, v3
	s_and_b64 s[6:7], s[6:7], 15
	v_or_b32_e32 v8, 0x800, v3
	s_sub_co_i32 s4, 16, s6
	s_mov_b32 s11, 0
	v_or_b32_e32 v5, 0x200, v4
	v_lshrrev_b32_e32 v1, 4, v4
	s_lshr_b32 s4, s4, 2
	s_cmp_lg_u64 s[6:7], 0
	v_or_b32_e32 v7, 0x600, v3
	v_lshrrev_b32_e32 v5, 4, v5
	v_and_b32_e32 v1, 0x1d8, v1
	s_cselect_b32 s4, s4, 0
	v_or_b32_e32 v2, v3, v2
	s_lshl2_add_u32 s0, s4, s0
	s_mul_i32 s6, s34, 0x60
	v_add_nc_u32_e32 v98, v1, v4
	v_or_b32_e32 v1, 0x200, v3
	s_add_co_i32 s4, s0, 0xa180
	v_add_nc_u32_e32 v102, v2, v6
	s_and_b32 s10, s4, 15
	s_add_co_i32 s42, s0, 0x5d80
	v_lshrrev_b32_e32 v1, 4, v1
	s_sub_co_i32 s7, 16, s10
	v_or_b32_e32 v6, 0x400, v3
	s_lshr_b32 s0, s7, 2
	s_cmp_lg_u64 s[10:11], 0
	v_or_b32_e32 v3, 0xa00, v3
	v_dual_lshrrev_b32 v1, 4, v8 :: v_dual_bitop2_b32 v9, 56, v1 bitop3:0x40
	v_dual_mov_b32 v119, v101 :: v_dual_sub_nc_u32 v8, 0x65f, v0
	s_cselect_b32 s0, s0, 0
	s_ashr_i32 s7, s17, 31
	v_dual_lshrrev_b32 v6, 4, v6 :: v_dual_lshrrev_b32 v7, 4, v7
	s_lshr_b32 s7, s7, 27
	v_dual_lshrrev_b32 v3, 4, v3 :: v_dual_lshrrev_b32 v104, 8, v8
	s_add_co_i32 s17, s17, s7
	s_lshl_b32 s10, s0, 2
	s_ashr_i32 s44, s17, 5
	s_cmp_lt_i32 s3, 0x60
	s_add_nc_u64 s[34:35], s[4:5], s[10:11]
	s_cselect_b32 s45, -1, 0
	s_lshl_b32 s4, s2, 8
	v_and_b32_e32 v5, 0x1f8, v5
	v_and_b32_e32 v6, 0x58, v6
	v_and_b32_e32 v7, 0x78, v7
	v_and_b32_e32 v10, 0x98, v1
	v_and_b32_e32 v3, 0xb8, v3
	v_dual_add_nc_u32 v8, 2, v104 :: v_dual_add_nc_u32 v108, v2, v9
	s_mov_b32 s43, s5
	s_ashr_i32 s7, s6, 31
	s_ashr_i32 s5, s4, 31
	s_wait_kmcnt 0x0
	s_bfe_i64 s[20:21], s[20:21], 0x200000
	s_bfe_i64 s[12:13], s[12:13], 0x200000
	s_mul_u64 s[6:7], s[20:21], s[6:7]
	s_mul_u64 s[4:5], s[12:13], s[4:5]
	v_or_b32_e32 v1, 0x100, v0
	v_dual_add_nc_u32 v110, v2, v6 :: v_dual_bitop2_b32 v107, 14, v8 bitop3:0x40
	v_dual_add_nc_u32 v106, v5, v4 :: v_dual_add_nc_u32 v112, v2, v7
	v_dual_add_nc_u32 v114, v2, v10 :: v_dual_add_nc_u32 v116, v2, v3
	v_lshl_or_b32 v118, v0, 2, 0x4000
	v_dual_mov_b32 v2, v101 :: v_dual_mov_b32 v3, v101
	v_dual_mov_b32 v4, v101 :: v_dual_mov_b32 v5, v101
	v_dual_mov_b32 v6, v101 :: v_dual_mov_b32 v7, v101
	v_dual_mov_b32 v8, v101 :: v_dual_mov_b32 v9, v101
	v_dual_mov_b32 v10, v101 :: v_dual_mov_b32 v11, v101
	v_dual_mov_b32 v12, v101 :: v_dual_mov_b32 v13, v101
	v_dual_mov_b32 v14, v101 :: v_dual_mov_b32 v15, v101
	v_dual_mov_b32 v16, v101 :: v_dual_mov_b32 v17, v101
	v_dual_mov_b32 v18, v101 :: v_dual_mov_b32 v19, v101
	v_dual_mov_b32 v20, v101 :: v_dual_mov_b32 v21, v101
	v_dual_mov_b32 v22, v101 :: v_dual_mov_b32 v23, v101
	v_dual_mov_b32 v24, v101 :: v_dual_mov_b32 v25, v101
	v_dual_mov_b32 v26, v101 :: v_dual_mov_b32 v27, v101
	v_dual_mov_b32 v28, v101 :: v_dual_mov_b32 v29, v101
	v_dual_mov_b32 v30, v101 :: v_dual_mov_b32 v31, v101
	v_dual_mov_b32 v32, v101 :: v_dual_mov_b32 v33, v101
	v_dual_mov_b32 v34, v101 :: v_dual_mov_b32 v35, v101
	v_dual_mov_b32 v36, v101 :: v_dual_mov_b32 v37, v101
	v_dual_mov_b32 v38, v101 :: v_dual_mov_b32 v39, v101
	v_dual_mov_b32 v40, v101 :: v_dual_mov_b32 v41, v101
	v_dual_mov_b32 v42, v101 :: v_dual_mov_b32 v43, v101
	v_dual_mov_b32 v44, v101 :: v_dual_mov_b32 v45, v101
	v_dual_mov_b32 v46, v101 :: v_dual_mov_b32 v47, v101
	v_dual_mov_b32 v48, v101 :: v_dual_mov_b32 v49, v101
	v_dual_mov_b32 v74, v101 :: v_dual_mov_b32 v75, v101
	v_dual_mov_b32 v76, v101 :: v_dual_mov_b32 v77, v101
	v_dual_mov_b32 v78, v101 :: v_dual_mov_b32 v79, v101
	v_dual_mov_b32 v80, v101 :: v_dual_mov_b32 v81, v101
	v_dual_mov_b32 v82, v101 :: v_dual_mov_b32 v83, v101
	v_dual_mov_b32 v84, v101 :: v_dual_mov_b32 v85, v101
	v_dual_mov_b32 v86, v101 :: v_dual_mov_b32 v87, v101
	v_dual_mov_b32 v88, v101 :: v_dual_mov_b32 v89, v101
	v_dual_mov_b32 v90, v101 :: v_dual_mov_b32 v91, v101
	v_dual_mov_b32 v92, v101 :: v_dual_mov_b32 v93, v101
	v_dual_mov_b32 v94, v101 :: v_dual_mov_b32 v95, v101
	v_dual_mov_b32 v96, v101 :: v_dual_mov_b32 v97, v101
	v_dual_mov_b32 v50, v101 :: v_dual_mov_b32 v51, v101
	v_dual_mov_b32 v52, v101 :: v_dual_mov_b32 v53, v101
	v_dual_mov_b32 v54, v101 :: v_dual_mov_b32 v55, v101
	v_dual_mov_b32 v56, v101 :: v_dual_mov_b32 v57, v101
	v_dual_mov_b32 v58, v101 :: v_dual_mov_b32 v59, v101
	v_dual_mov_b32 v60, v101 :: v_dual_mov_b32 v61, v101
	v_dual_mov_b32 v62, v101 :: v_dual_mov_b32 v63, v101
	v_dual_mov_b32 v64, v101 :: v_dual_mov_b32 v65, v101
	v_dual_mov_b32 v66, v101 :: v_dual_mov_b32 v67, v101
	v_dual_mov_b32 v68, v101 :: v_dual_mov_b32 v69, v101
	v_dual_mov_b32 v70, v101 :: v_dual_mov_b32 v71, v101
	v_dual_mov_b32 v72, v101 :: v_dual_mov_b32 v73, v101
	v_or_b32_e32 v109, 0xf00, v0
	v_mov_b32_e32 v99, v104
	s_lshr_b32 s46, s3, 16
	s_lshr_b32 s47, s31, 16
	s_lshl_b64 s[6:7], s[6:7], 1
	s_lshl_b64 s[4:5], s[4:5], 1
	s_mov_b32 s41, s1
	s_movk_i32 s16, 0x60
	s_bitset1_b32 s46, 21
	s_bitset1_b32 s47, 21
	s_add_nc_u64 s[20:21], s[14:15], s[6:7]
	s_add_nc_u64 s[36:37], s[18:19], s[4:5]
	s_mov_b32 s48, -1
	s_movk_i32 s8, 0x100
	s_mov_b32 s4, 0x7510000
	s_mov_b32 s49, s11
	s_branch .LBB0_21
.LBB0_20:                               ;   in Loop: Header=BB0_21 Depth=1
	s_cmp_eq_u32 s49, s44
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_scc1 .LBB0_42
.LBB0_21:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_23 Depth 2
                                        ;     Child Loop BB0_26 Depth 2
                                        ;     Child Loop BB0_29 Depth 2
	s_and_b32 s50, s49, 1
	s_add_co_i32 s49, s49, 1
	s_xor_b32 s5, s50, 1
	s_lshl_b32 s0, s49, 5
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_sub_co_i32 s0, s26, s0
	s_min_i32 s0, s0, 32
	s_cmp_lt_i32 s49, s44
	s_cselect_b32 s2, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_and_b32 s6, s2, exec_lo
	s_cselect_b32 s30, s0, 0
	s_cmp_lt_i32 s30, 32
	s_cselect_b32 s0, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b32 s0, s45, s0
	s_or_b32 s0, s38, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s0
	s_cbranch_vccnz .LBB0_34
; %bb.22:                               ;   in Loop: Header=BB0_21 Depth=1
	v_nop
	v_nop
	v_nop
	v_mov_b64_e32 v[120:121], v[0:1]
	v_mov_b32_e32 v111, 16
	s_cmp_lg_u32 s5, 0
	s_mov_b32 s0, 0
	s_cselect_b32 s7, s43, s1
	s_cselect_b32 s6, s42, 0
.LBB0_23:                               ;   Parent Loop BB0_21 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_dual_mov_b32 v100, v120 :: v_dual_add_nc_u32 v111, -2, v111
	v_dual_mov_b32 v122, v121 :: v_dual_mov_b32 v123, v101
	v_add_nc_u32_e32 v121, 0x200, v121
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u64 v[124:125], v[100:101], 2, s[6:7]
	v_cmp_eq_u32_e32 vcc_lo, 0, v111
	v_add_nc_u32_e32 v120, 0x200, v120
	v_lshl_add_u64 v[122:123], v[122:123], 2, s[6:7]
	s_clause 0x1
	flat_store_b32 v[124:125], v101
	flat_store_b32 v[122:123], v101
	s_or_b32 s0, vcc_lo, s0
	s_wait_xcnt 0x0
	s_and_not1_b32 exec_lo, exec_lo, s0
	s_cbranch_execnz .LBB0_23
; %bb.24:                               ;   in Loop: Header=BB0_21 Depth=1
	s_or_b32 exec_lo, exec_lo, s0
	s_and_saveexec_b32 s0, s48
	s_cbranch_execz .LBB0_27
; %bb.25:                               ;   in Loop: Header=BB0_21 Depth=1
	v_add_nc_u64_e32 v[120:121], s[6:7], v[118:119]
	v_mov_b32_e32 v100, v109
	s_mov_b32 s6, 0
.LBB0_26:                               ;   Parent Loop BB0_21 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v100, 0x100, v100
	flat_store_b32 v[120:121], v101
	s_wait_xcnt 0x0
	v_add_nc_u64_e32 v[120:121], 0x400, v[120:121]
	v_cmp_lt_u32_e32 vcc_lo, 0xfff, v100
	s_or_b32 s6, vcc_lo, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s6
	s_cbranch_execnz .LBB0_26
.LBB0_27:                               ;   in Loop: Header=BB0_21 Depth=1
	s_or_b32 exec_lo, exec_lo, s0
	v_mov_b64_e32 v[120:121], v[0:1]
	s_cmp_lg_u32 s5, 0
	s_mov_b32 s6, 0
	s_cselect_b32 s13, s35, s41
	s_cselect_b32 s12, s34, s39
	s_mov_b32 s10, 0
	s_branch .LBB0_29
.LBB0_28:                               ;   in Loop: Header=BB0_29 Depth=2
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s0
	s_add_co_i32 s6, s6, 2
	v_add_nc_u32_e32 v121, 0x200, v121
	v_cmp_eq_u32_e32 vcc_lo, s6, v107
	v_add_nc_u32_e32 v120, 0x200, v120
	s_or_b32 s10, vcc_lo, s10
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s10
	s_cbranch_execz .LBB0_33
.LBB0_29:                               ;   Parent Loop BB0_21 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_mov_b32 s7, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b64 s[14:15], s[6:7], 0x100000000
	s_mov_b32 s7, exec_lo
	v_cmp_le_u32_e32 vcc_lo, s15, v99
	v_cmpx_le_u32_e64 s14, v104
	s_cbranch_execz .LBB0_31
; %bb.30:                               ;   in Loop: Header=BB0_29 Depth=2
	v_mov_b32_e32 v100, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u64 v[122:123], v[100:101], 2, s[12:13]
	flat_store_b32 v[122:123], v101
.LBB0_31:                               ;   in Loop: Header=BB0_29 Depth=2
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s7
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB0_28
; %bb.32:                               ;   in Loop: Header=BB0_29 Depth=2
	v_mov_b32_e32 v100, v121
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u64 v[122:123], v[100:101], 2, s[12:13]
	flat_store_b32 v[122:123], v101
	s_branch .LBB0_28
.LBB0_33:                               ;   in Loop: Header=BB0_21 Depth=1
	s_or_b32 exec_lo, exec_lo, s10
	s_barrier_signal -1
	s_barrier_wait -1
.LBB0_34:                               ;   in Loop: Header=BB0_21 Depth=1
	s_and_b32 s0, s2, exec_lo
	s_cselect_b32 s0, s49, 0
	s_mov_b32 s2, exec_lo
	v_cmpx_lt_i32_e32 0, v105
	s_xor_b32 s6, exec_lo, s2
	s_cbranch_execnz .LBB0_37
; %bb.35:                               ;   in Loop: Header=BB0_21 Depth=1
	s_and_not1_saveexec_b32 s2, s6
	s_cbranch_execnz .LBB0_40
.LBB0_36:                               ;   in Loop: Header=BB0_21 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s40
	s_cbranch_vccnz .LBB0_20
	s_branch .LBB0_41
.LBB0_37:                               ;   in Loop: Header=BB0_21 Depth=1
	s_mov_b32 s7, exec_lo
	v_cmpx_eq_u32_e32 1, v105
	s_cbranch_execz .LBB0_39
; %bb.38:                               ;   in Loop: Header=BB0_21 Depth=1
	s_cmp_lg_u32 s5, 0
	s_mov_b32 s2, s30
	s_cselect_b32 s14, s34, s39
	s_cmp_gt_i32 s30, 0
	s_mov_b32 s17, s9
	s_cselect_b32 s15, -1, 0
	s_lshl_b32 s10, s0, 5
	s_mov_b32 s18, s11
	s_lshl_b64 s[12:13], s[10:11], 1
	s_mov_b32 s19, s11
	s_add_nc_u64 s[12:13], s[20:21], s[12:13]
	v_nop
	v_dual_mov_b32 v111, s14 :: v_dual_mov_b32 v120, s12
	s_and_b32 s10, s13, 0x1ffffff
	s_and_b32 s13, s25, s15
	s_bitset1_b32 s10, 31
	v_cndmask_b32_e64 v100, 0, 1, s13
	v_mov_b32_e32 v113, s10
	v_readfirstlane_b32 s53, v111
	v_readfirstlane_b32 s54, v120
	s_lshr_b64 s[14:15], s[2:3], 16
	v_readfirstlane_b32 s52, v100
	v_readfirstlane_b32 s55, v113
	s_lshl_b32 s13, s30, 16
	s_mov_b32 s12, s4
	s_mov_b32 s15, s46
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[52:55], s[12:19]
.LBB0_39:                               ;   in Loop: Header=BB0_21 Depth=1
	s_or_b32 exec_lo, exec_lo, s7
	s_and_not1_saveexec_b32 s2, s6
	s_cbranch_execz .LBB0_36
.LBB0_40:                               ;   in Loop: Header=BB0_21 Depth=1
	s_cmp_lg_u32 s5, 0
	s_cselect_b32 s5, s42, 0
	s_cmp_gt_i32 s30, 0
	s_cselect_b32 s12, -1, 0
	s_lshl_b32 s10, s0, 5
	s_delay_alu instid0(SALU_CYCLE_1)
	s_lshl_b64 s[6:7], s[10:11], 1
	s_mov_b32 s10, s11
	s_add_nc_u64 s[6:7], s[36:37], s[6:7]
	v_nop
	v_nop
	v_dual_mov_b32 v111, s5 :: v_dual_mov_b32 v120, s6
	s_and_b32 s0, s7, 0x1ffffff
	s_and_b32 s7, s29, s12
	s_bitset1_b32 s0, 31
	v_cndmask_b32_e64 v100, 0, 1, s7
	v_mov_b32_e32 v113, s0
	v_readfirstlane_b32 s13, v111
	v_readfirstlane_b32 s14, v120
	s_lshr_b64 s[6:7], s[30:31], 16
	v_readfirstlane_b32 s12, v100
	v_readfirstlane_b32 s15, v113
	s_lshl_b32 s5, s30, 16
	s_mov_b32 s7, s47
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	tensor_load_to_lds s[12:15], s[4:11]
	s_or_b32 exec_lo, exec_lo, s2
	s_and_not1_b32 vcc_lo, exec_lo, s40
	s_cbranch_vccnz .LBB0_20
.LBB0_41:                               ;   in Loop: Header=BB0_21 Depth=1
	s_cmp_lg_u32 s50, 0
	s_cselect_b32 s0, s42, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	v_lshl_add_u32 v100, v98, 1, s0
	v_lshl_add_u32 v111, v106, 1, s0
	s_cselect_b32 s0, s34, s39
	ds_load_b128 v[120:123], v100
	ds_load_b128 v[124:127], v100 offset:16
	v_lshl_add_u32 v100, v110, 1, s0
	ds_load_b128 v[128:131], v111 offset:1024
	ds_load_b128 v[132:135], v111 offset:1040
	; sched_group_barrier mask(0x00000100) size(4) SyncID(0)
	ds_load_b128 v[136:139], v100 offset:2048
	ds_load_b128 v[140:143], v100 offset:2064
	v_lshl_add_u32 v100, v108, 1, s0
	ds_load_b128 v[144:147], v100 offset:1024
	ds_load_b128 v[148:151], v100 offset:1040
	v_lshl_add_u32 v100, v102, 1, s0
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[2:9], v[128:135], v[136:143], v[2:9] matrix_a_reuse
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[10:17], v[128:135], v[144:151], v[10:17] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[82:89], v[120:127], v[144:151], v[82:89] matrix_a_reuse
	ds_load_b128 v[152:155], v100
	ds_load_b128 v[156:159], v100 offset:16
	v_lshl_add_u32 v100, v112, 1, s0
	ds_load_b128 v[144:147], v100 offset:3072
	ds_load_b128 v[148:151], v100 offset:3088
	v_lshl_add_u32 v100, v114, 1, s0
	v_wmma_f32_16x16x32_bf16 v[74:81], v[120:127], v[136:143], v[74:81] matrix_a_reuse
	ds_load_b128 v[136:139], v100 offset:4096
	ds_load_b128 v[140:143], v100 offset:4112
	v_lshl_add_u32 v100, v116, 1, s0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[34:41], v[120:127], v[136:143], v[34:41] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[50:57], v[128:135], v[144:151], v[50:57] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[42:49], v[120:127], v[144:151], v[42:49] matrix_a_reuse
	ds_load_b128 v[144:147], v100 offset:5120
	ds_load_b128 v[148:151], v100 offset:5136
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[26:33], v[120:127], v[144:151], v[26:33] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[66:73], v[128:135], v[144:151], v[66:73]
	; sched_group_barrier mask(0x00000008) size(3) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(3) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(4) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[18:25], v[128:135], v[152:159], v[18:25] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[90:97], v[120:127], v[152:159], v[90:97]
	; sched_group_barrier mask(0x00000008) size(3) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(4) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[58:65], v[128:135], v[136:143], v[58:65] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(3) SyncID(0)
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
	s_and_b32 vcc_lo, exec_lo, s40
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_vccz .LBB0_44
; %bb.43:
	v_lshrrev_b32_e32 v1, 1, v0
	v_and_b32_e32 v98, 0xe0, v0
	v_cvt_pk_bf16_f32 v81, v80, v81
	v_cvt_pk_bf16_f32 v80, v78, v79
	v_cvt_pk_bf16_f32 v89, v88, v89
	v_lshl_or_b32 v1, v0, 8, v1
	v_cvt_pk_bf16_f32 v88, v86, v87
	v_cvt_pk_bf16_f32 v86, v82, v83
	v_cvt_pk_bf16_f32 v87, v84, v85
	v_cvt_pk_bf16_f32 v79, v76, v77
	v_and_or_b32 v1, 0xf08, v1, v98
	v_cvt_pk_bf16_f32 v97, v96, v97
	v_cvt_pk_bf16_f32 v96, v94, v95
	v_cvt_pk_bf16_f32 v95, v92, v93
	v_cvt_pk_bf16_f32 v94, v90, v91
	v_or_b32_e32 v78, 0x1000, v1
	v_or_b32_e32 v83, 0x3000, v1
	v_or_b32_e32 v82, 0x2000, v1
	v_or_b32_e32 v84, 0x4000, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_lshrrev_b32 v76, 3, v1 :: v_dual_lshrrev_b32 v78, 3, v78
	v_lshlrev_b32_e32 v77, 1, v1
	v_or_b32_e32 v1, 0x5000, v1
	v_dual_lshrrev_b32 v83, 3, v83 :: v_dual_lshrrev_b32 v82, 3, v82
	v_and_b32_e32 v78, 0x3f0, v78
	v_lshrrev_b32_e32 v84, 3, v84
	v_and_b32_e32 v76, 0x1f0, v76
	v_lshrrev_b32_e32 v1, 3, v1
	v_and_b32_e32 v82, 0x5f0, v82
	v_add_nc_u32_e32 v85, v78, v77
	v_cvt_pk_bf16_f32 v78, v74, v75
	v_and_b32_e32 v74, 0x7f0, v83
	v_and_b32_e32 v75, 0x9f0, v84
	v_add_nc_u32_e32 v76, v76, v77
	v_and_b32_e32 v1, 0xbf0, v1
	v_cvt_pk_bf16_f32 v25, v24, v25
	v_cvt_pk_bf16_f32 v24, v22, v23
	v_cvt_pk_bf16_f32 v23, v20, v21
	v_cvt_pk_bf16_f32 v22, v18, v19
	v_cvt_pk_bf16_f32 v49, v48, v49
	v_cvt_pk_bf16_f32 v48, v46, v47
	v_cvt_pk_bf16_f32 v47, v44, v45
	v_cvt_pk_bf16_f32 v46, v42, v43
	v_add_nc_u32_e32 v42, v74, v77
	v_cvt_pk_bf16_f32 v17, v16, v17
	v_cvt_pk_bf16_f32 v16, v14, v15
	v_cvt_pk_bf16_f32 v15, v12, v13
	v_cvt_pk_bf16_f32 v14, v10, v11
	v_cvt_pk_bf16_f32 v9, v8, v9
	v_cvt_pk_bf16_f32 v8, v6, v7
	v_cvt_pk_bf16_f32 v7, v4, v5
	v_cvt_pk_bf16_f32 v6, v2, v3
	v_cvt_pk_bf16_f32 v5, v56, v57
	v_cvt_pk_bf16_f32 v4, v54, v55
	v_cvt_pk_bf16_f32 v3, v52, v53
	v_cvt_pk_bf16_f32 v2, v50, v51
	v_dual_add_nc_u32 v82, v82, v77 :: v_dual_add_nc_u32 v43, v75, v77
	v_cvt_pk_bf16_f32 v41, v40, v41
	v_cvt_pk_bf16_f32 v40, v38, v39
	v_cvt_pk_bf16_f32 v39, v36, v37
	v_cvt_pk_bf16_f32 v38, v34, v35
	ds_store_b128 v76, v[94:97]
	ds_store_b128 v76, v[22:25] offset:32
	ds_store_b128 v85, v[86:89] offset:8192
	ds_store_b128 v85, v[14:17] offset:8224
	ds_store_b128 v82, v[78:81] offset:16384
	ds_store_b128 v82, v[6:9] offset:16416
	v_cvt_pk_bf16_f32 v9, v64, v65
	v_cvt_pk_bf16_f32 v8, v62, v63
	v_cvt_pk_bf16_f32 v7, v60, v61
	v_cvt_pk_bf16_f32 v6, v58, v59
	v_cvt_pk_bf16_f32 v33, v32, v33
	v_cvt_pk_bf16_f32 v32, v30, v31
	v_cvt_pk_bf16_f32 v31, v28, v29
	v_cvt_pk_bf16_f32 v30, v26, v27
	v_add_nc_u32_e32 v1, v1, v77
	v_cvt_pk_bf16_f32 v13, v72, v73
	v_cvt_pk_bf16_f32 v12, v70, v71
	v_cvt_pk_bf16_f32 v11, v68, v69
	v_cvt_pk_bf16_f32 v10, v66, v67
	ds_store_b128 v42, v[46:49] offset:24576
	ds_store_b128 v42, v[2:5] offset:24608
	ds_store_b128 v43, v[38:41] offset:32768
	ds_store_b128 v43, v[6:9] offset:32800
	ds_store_b128 v1, v[30:33] offset:40960
	ds_store_b128 v1, v[10:13] offset:40992
.LBB0_44:
	v_cmp_ne_u32_e32 vcc_lo, 1, v103
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
	s_mul_i32 s14, s33, 0x60
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
	v_lshl_add_u32 v11, v11, 8, v12
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_mul_lo_u32 v16, v27, s22
	v_dual_add_nc_u32 v24, s19, v27 :: v_dual_sub_nc_u32 v6, v2, v6
	v_ashrrev_i32_e32 v19, 31, v18
	v_lshl_add_u32 v26, v26, 8, v14
	v_mul_u64_e32 v[20:21], s[6:7], v[20:21]
	v_ashrrev_i32_e32 v25, 31, v24
	v_lshl_add_u32 v1, v1, 8, v6
	v_sub_nc_u32_e32 v16, v5, v16
	v_mul_u64_e32 v[18:19], s[24:25], v[18:19]
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
	v_dual_sub_nc_u32 v7, v4, v9 :: v_dual_lshlrev_b32 v4, 8, v4
	v_lshlrev_b32_e32 v9, 8, v9
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
	.size	bm256_bn096_bk032_wm8_wn1_mc0, .Lfunc_end0-bm256_bn096_bk032_wm8_wn1_mc0
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel bm256_bn096_bk032_wm8_wn1_mc0
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
		.amdhsa_next_free_vgpr 160
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
		.amdhsa_inst_pref_size 47
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text.bm256_bn096_bk032_wm8_wn1_mc0,"axG",@progbits,bm256_bn096_bk032_wm8_wn1_mc0,comdat
                                        ; -- End function
	.set .Lbm256_bn096_bk032_wm8_wn1_mc0.num_vgpr, 160
	.set .Lbm256_bn096_bk032_wm8_wn1_mc0.num_agpr, 0
	.set .Lbm256_bn096_bk032_wm8_wn1_mc0.numbered_sgpr, 56
	.set .Lbm256_bn096_bk032_wm8_wn1_mc0.num_named_barrier, 0
	.set .Lbm256_bn096_bk032_wm8_wn1_mc0.private_seg_size, 0
	.set .Lbm256_bn096_bk032_wm8_wn1_mc0.uses_vcc, 1
	.set .Lbm256_bn096_bk032_wm8_wn1_mc0.uses_flat_scratch, 1
	.set .Lbm256_bn096_bk032_wm8_wn1_mc0.has_dyn_sized_stack, 0
	.set .Lbm256_bn096_bk032_wm8_wn1_mc0.has_recursion, 0
	.set .Lbm256_bn096_bk032_wm8_wn1_mc0.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 6016
; TotalNumSgprs: 58
; NumVgprs: 160
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 52224 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 9
; NumSGPRsForWavesPerEU: 58
; NumVGPRsForWavesPerEU: 160
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
	.type	__hip_cuid_9f354a54ef8150d5,@object ; @__hip_cuid_9f354a54ef8150d5
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_9f354a54ef8150d5
__hip_cuid_9f354a54ef8150d5:
	.byte	0                               ; 0x0
	.size	__hip_cuid_9f354a54ef8150d5, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git aa451e1fe6a793394d6733051b1778633063ae96+PATCHED:d17c5aa0e3ea29cde402f58f27e39b6034effa27)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_9f354a54ef8150d5
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
    .group_segment_fixed_size: 52224
    .kernarg_segment_align: 8
    .kernarg_segment_size: 132
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           bm256_bn096_bk032_wm8_wn1_mc0
    .private_segment_fixed_size: 0
    .sgpr_count:     58
    .sgpr_spill_count: 0
    .symbol:         bm256_bn096_bk032_wm8_wn1_mc0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     160
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
