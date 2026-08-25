	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.section	.text.bm256_bn128_bk032_wm1_wn8_mc0,"axG",@progbits,bm256_bn128_bk032_wm1_wn8_mc0,comdat
	.protected	bm256_bn128_bk032_wm1_wn8_mc0 ; -- Begin function bm256_bn128_bk032_wm1_wn8_mc0
	.globl	bm256_bn128_bk032_wm1_wn8_mc0
	.p2align	8
	.type	bm256_bn128_bk032_wm1_wn8_mc0,@function
bm256_bn128_bk032_wm1_wn8_mc0: ; @bm256_bn128_bk032_wm1_wn8_mc0
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
	s_cselect_b32 s7, ttmp7, s4
	s_add_co_i32 s2, s24, 0xff
	s_add_co_i32 s4, s25, 0x7f
	s_ashr_i32 s3, s2, 31
	s_lshl_b32 s28, s35, 8
	s_lshr_b32 s3, s3, 24
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s2, s2, s3
	s_ashr_i32 s3, s4, 31
	s_ashr_i32 s8, s2, 8
	s_lshr_b32 s2, s3, 25
	s_add_co_i32 s4, s4, s2
	s_sub_co_i32 s2, s24, s28
	s_ashr_i32 s10, s4, 7
	s_min_i32 s27, s2, 0x100
	s_cmp_lt_i32 s35, s8
	s_cselect_b32 s29, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_and_b32 s2, s29, exec_lo
	s_cselect_b32 s31, s27, 0
	s_lshl_b32 s33, s7, 7
	s_sub_co_i32 s2, s25, s33
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_min_i32 s2, s2, 0x80
	s_cmp_lt_i32 s7, s10
	s_cselect_b32 s25, -1, 0
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
	s_cmp_lt_i32 s3, 0x80
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
	v_dual_mov_b32 v5, 0 :: v_dual_sub_nc_u32 v2, 0x87f, v0
	v_lshl_add_u32 v3, s6, 2, v1
	s_mov_b32 s4, 0
	s_mov_b32 s11, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshrrev_b32_e32 v2, 8, v2
	v_add_nc_u32_e32 v4, 2, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v1, v2 :: v_dual_bitop2_b32 v4, 30, v4 bitop3:0x40
	s_branch .LBB0_7
.LBB0_6:                                ;   in Loop: Header=BB0_7 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
	s_add_co_i32 s4, s4, 2
	v_add_nc_u32_e32 v3, 0x800, v3
	v_cmp_eq_u32_e32 vcc_lo, s4, v4
	s_or_b32 s11, vcc_lo, s11
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s11
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
	s_or_b32 exec_lo, exec_lo, s11
	s_barrier_signal -1
	s_barrier_wait -1
.LBB0_12:
	s_clause 0x2
	s_load_b64 s[18:19], s[0:1], 0x0 nv
	s_load_b128 s[12:15], s[0:1], 0x20 nv
	s_load_b128 s[20:23], s[0:1], 0x48 nv
	v_lshrrev_b32_e32 v137, 5, v0
	s_lshl_b32 s34, s6, 2
	s_add_co_i32 s10, s10, -1
	s_wait_xcnt 0x0
	s_mov_b64 s[0:1], src_shared_base
	s_or_b32 s39, s34, 0x4400
	s_add_co_i32 s36, s8, -1
	s_min_i32 s0, s7, s10
	s_mov_b32 s2, exec_lo
	v_cmpx_lt_i32_e32 0, v137
	s_xor_b32 s37, exec_lo, s2
	s_cbranch_execz .LBB0_16
; %bb.13:
	s_mov_b32 s40, exec_lo
	v_cmpx_eq_u32_e32 1, v137
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
	v_dual_mov_b32 v1, s39 :: v_dual_mov_b32 v4, s6
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
	s_or_b32 s7, s4, 0x200000
	s_movk_i32 s8, 0x80
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
	v_dual_mov_b32 v9, 0 :: v_dual_lshlrev_b32 v139, 4, v137
	s_and_b32 s40, s29, s25
	s_and_not1_b32 vcc_lo, exec_lo, s16
	v_cndmask_b32_e64 v135, 0, 1, s40
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
	v_dual_mov_b32 v42, v9 :: v_dual_mov_b32 v57, v9
	v_dual_mov_b32 v56, v9 :: v_dual_mov_b32 v55, v9
	v_dual_mov_b32 v54, v9 :: v_dual_mov_b32 v53, v9
	v_dual_mov_b32 v52, v9 :: v_dual_mov_b32 v51, v9
	v_dual_mov_b32 v50, v9 :: v_dual_mov_b32 v65, v9
	v_dual_mov_b32 v64, v9 :: v_dual_mov_b32 v63, v9
	v_dual_mov_b32 v62, v9 :: v_dual_mov_b32 v61, v9
	v_dual_mov_b32 v60, v9 :: v_dual_mov_b32 v59, v9
	v_dual_mov_b32 v58, v9 :: v_dual_mov_b32 v81, v9
	v_dual_mov_b32 v80, v9 :: v_dual_mov_b32 v79, v9
	v_dual_mov_b32 v78, v9 :: v_dual_mov_b32 v77, v9
	v_dual_mov_b32 v76, v9 :: v_dual_mov_b32 v75, v9
	v_dual_mov_b32 v74, v9 :: v_dual_mov_b32 v105, v9
	v_dual_mov_b32 v104, v9 :: v_dual_mov_b32 v103, v9
	v_dual_mov_b32 v102, v9 :: v_dual_mov_b32 v101, v9
	v_dual_mov_b32 v100, v9 :: v_dual_mov_b32 v99, v9
	v_dual_mov_b32 v98, v9 :: v_dual_mov_b32 v113, v9
	v_dual_mov_b32 v112, v9 :: v_dual_mov_b32 v111, v9
	v_dual_mov_b32 v110, v9 :: v_dual_mov_b32 v109, v9
	v_dual_mov_b32 v108, v9 :: v_dual_mov_b32 v107, v9
	v_dual_mov_b32 v106, v9 :: v_dual_mov_b32 v121, v9
	v_dual_mov_b32 v120, v9 :: v_dual_mov_b32 v119, v9
	v_dual_mov_b32 v118, v9 :: v_dual_mov_b32 v117, v9
	v_dual_mov_b32 v116, v9 :: v_dual_mov_b32 v115, v9
	v_dual_mov_b32 v114, v9 :: v_dual_mov_b32 v129, v9
	v_dual_mov_b32 v128, v9 :: v_dual_mov_b32 v127, v9
	v_dual_mov_b32 v126, v9 :: v_dual_mov_b32 v125, v9
	v_dual_mov_b32 v124, v9 :: v_dual_mov_b32 v123, v9
	v_dual_mov_b32 v122, v9 :: v_dual_mov_b32 v73, v9
	v_dual_mov_b32 v72, v9 :: v_dual_mov_b32 v71, v9
	v_dual_mov_b32 v70, v9 :: v_dual_mov_b32 v69, v9
	v_dual_mov_b32 v68, v9 :: v_dual_mov_b32 v67, v9
	v_dual_mov_b32 v66, v9 :: v_dual_mov_b32 v89, v9
	v_dual_mov_b32 v88, v9 :: v_dual_mov_b32 v87, v9
	v_dual_mov_b32 v86, v9 :: v_dual_mov_b32 v85, v9
	v_dual_mov_b32 v84, v9 :: v_dual_mov_b32 v83, v9
	v_dual_mov_b32 v82, v9 :: v_dual_mov_b32 v97, v9
	v_dual_mov_b32 v96, v9 :: v_dual_mov_b32 v95, v9
	v_dual_mov_b32 v94, v9 :: v_dual_mov_b32 v93, v9
	v_dual_mov_b32 v92, v9 :: v_dual_mov_b32 v91, v9
	v_mov_b32_e32 v90, v9
	s_barrier_wait -1
	s_cbranch_vccnz .LBB0_42
; %bb.19:
	v_dual_lshlrev_b32 v1, 5, v0 :: v_dual_lshlrev_b32 v2, 1, v0
	s_mov_b64 s[4:5], src_shared_base
	s_or_b32 s6, s34, 0x6600
	s_mov_b32 s7, s5
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v1, 0x1e0, v1
	s_and_b64 s[6:7], s[6:7], 15
	s_mov_b32 s11, 0
	s_sub_co_i32 s4, 16, s6
	s_mov_b32 s43, s5
	s_lshr_b32 s4, s4, 2
	v_or_b32_e32 v3, 0x200, v1
	v_and_or_b32 v4, v0, 16, v1
	v_and_b32_e32 v2, 24, v2
	s_cmp_lg_u64 s[6:7], 0
	v_or_b32_e32 v5, 0x600, v1
	s_cselect_b32 s4, s4, 0
	v_lshrrev_b32_e32 v3, 4, v3
	s_lshl2_add_u32 s6, s4, s34
	v_dual_mov_b32 v133, 0 :: v_dual_add_nc_u32 v130, v2, v4
	v_or_b32_e32 v2, 0x400, v1
	s_add_co_i32 s4, s6, 0xaa00
	v_and_b32_e32 v3, 56, v3
	s_and_b32 s10, s4, 15
	v_lshl_or_b32 v15, v139, 5, v4
	v_lshrrev_b32_e32 v2, 4, v2
	s_sub_co_i32 s7, 16, s10
	s_add_co_i32 s42, s6, 0x6600
	s_lshr_b32 s6, s7, 2
	s_cmp_lg_u64 s[10:11], 0
	v_or_b32_e32 v6, 0x800, v1
	v_or_b32_e32 v7, 0xa00, v1
	v_or_b32_e32 v8, 0xc00, v1
	v_lshrrev_b32_e32 v5, 4, v5
	v_and_b32_e32 v2, 0x58, v2
	v_or_b32_e32 v9, 0xe00, v1
	v_or_b32_e32 v10, 0x1000, v1
	v_or_b32_e32 v11, 0x1200, v1
	v_or_b32_e32 v12, 0x1400, v1
	v_or_b32_e32 v13, 0x1600, v1
	v_or_b32_e32 v14, 0x1800, v1
	v_or_b32_e32 v16, 0x1a00, v1
	v_or_b32_e32 v17, 0x1c00, v1
	v_or_b32_e32 v1, 0x1e00, v1
	v_lshrrev_b32_e32 v18, 4, v15
	v_sub_nc_u32_e32 v19, 0x87f, v0
	s_cselect_b32 s6, s6, 0
	s_ashr_i32 s7, s17, 31
	v_lshrrev_b32_e32 v6, 4, v6
	s_lshr_b32 s7, s7, 27
	v_dual_lshrrev_b32 v7, 4, v7 :: v_dual_lshrrev_b32 v8, 4, v8
	s_add_co_i32 s17, s17, s7
	v_and_b32_e32 v5, 0x78, v5
	v_dual_lshrrev_b32 v9, 4, v9 :: v_dual_lshrrev_b32 v10, 4, v10
	v_dual_lshrrev_b32 v11, 4, v11 :: v_dual_lshrrev_b32 v12, 4, v12
	v_dual_lshrrev_b32 v13, 4, v13 :: v_dual_lshrrev_b32 v14, 4, v14
	v_lshrrev_b32_e32 v16, 4, v16
	v_and_b32_e32 v18, 0xf8, v18
	v_dual_lshrrev_b32 v17, 4, v17 :: v_dual_add_nc_u32 v140, v2, v4
	v_add_nc_u32_e32 v142, v5, v4
	v_dual_mov_b32 v2, v133 :: v_dual_lshrrev_b32 v1, 4, v1
	v_lshrrev_b32_e32 v136, 8, v19
	s_lshl_b32 s10, s6, 2
	s_ashr_i32 s44, s17, 5
	s_cmp_lt_i32 s3, 0x80
	s_add_nc_u64 s[34:35], s[4:5], s[10:11]
	s_cselect_b32 s45, -1, 0
	s_lshl_b32 s4, s0, 7
	s_lshl_b32 s6, s2, 8
	v_and_b32_e32 v6, 0x98, v6
	v_and_b32_e32 v7, 0xb8, v7
	v_and_b32_e32 v8, 0xd8, v8
	v_and_b32_e32 v9, 0xf8, v9
	v_and_b32_e32 v10, 0x118, v10
	v_and_b32_e32 v11, 0x138, v11
	v_and_b32_e32 v12, 0x158, v12
	v_and_b32_e32 v13, 0x178, v13
	v_and_b32_e32 v14, 0x198, v14
	v_and_b32_e32 v16, 0x1b8, v16
	v_and_b32_e32 v20, 0x1f8, v1
	v_add_nc_u32_e32 v134, v18, v15
	v_and_b32_e32 v17, 0x1d8, v17
	v_dual_add_nc_u32 v15, 2, v136 :: v_dual_mov_b32 v169, v133
	s_ashr_i32 s5, s4, 31
	s_ashr_i32 s7, s6, 31
	s_wait_kmcnt 0x0
	s_bfe_i64 s[20:21], s[20:21], 0x200000
	s_bfe_i64 s[12:13], s[12:13], 0x200000
	s_mul_u64 s[4:5], s[20:21], s[4:5]
	s_mul_u64 s[6:7], s[12:13], s[6:7]
	v_or_b32_e32 v1, 0x100, v0
	v_dual_add_nc_u32 v138, v3, v4 :: v_dual_bitop2_b32 v141, 30, v15 bitop3:0x40
	v_dual_add_nc_u32 v144, v6, v4 :: v_dual_add_nc_u32 v146, v7, v4
	v_dual_add_nc_u32 v148, v8, v4 :: v_dual_add_nc_u32 v150, v9, v4
	v_dual_add_nc_u32 v152, v10, v4 :: v_dual_add_nc_u32 v154, v11, v4
	v_dual_add_nc_u32 v156, v12, v4 :: v_dual_add_nc_u32 v158, v13, v4
	v_dual_add_nc_u32 v160, v14, v4 :: v_dual_add_nc_u32 v162, v16, v4
	v_dual_add_nc_u32 v164, v17, v4 :: v_dual_add_nc_u32 v166, v20, v4
	v_lshl_or_b32 v168, v0, 2, 0x4000
	v_dual_mov_b32 v3, v133 :: v_dual_mov_b32 v4, v133
	v_dual_mov_b32 v5, v133 :: v_dual_mov_b32 v6, v133
	v_dual_mov_b32 v7, v133 :: v_dual_mov_b32 v8, v133
	v_dual_mov_b32 v9, v133 :: v_dual_mov_b32 v10, v133
	v_dual_mov_b32 v11, v133 :: v_dual_mov_b32 v12, v133
	v_dual_mov_b32 v13, v133 :: v_dual_mov_b32 v14, v133
	v_dual_mov_b32 v15, v133 :: v_dual_mov_b32 v16, v133
	v_dual_mov_b32 v17, v133 :: v_dual_mov_b32 v18, v133
	v_dual_mov_b32 v19, v133 :: v_dual_mov_b32 v20, v133
	v_dual_mov_b32 v21, v133 :: v_dual_mov_b32 v22, v133
	v_dual_mov_b32 v23, v133 :: v_dual_mov_b32 v24, v133
	v_dual_mov_b32 v25, v133 :: v_dual_mov_b32 v26, v133
	v_dual_mov_b32 v27, v133 :: v_dual_mov_b32 v28, v133
	v_dual_mov_b32 v29, v133 :: v_dual_mov_b32 v30, v133
	v_dual_mov_b32 v31, v133 :: v_dual_mov_b32 v32, v133
	v_dual_mov_b32 v33, v133 :: v_dual_mov_b32 v34, v133
	v_dual_mov_b32 v35, v133 :: v_dual_mov_b32 v36, v133
	v_dual_mov_b32 v37, v133 :: v_dual_mov_b32 v38, v133
	v_dual_mov_b32 v39, v133 :: v_dual_mov_b32 v40, v133
	v_dual_mov_b32 v41, v133 :: v_dual_mov_b32 v42, v133
	v_dual_mov_b32 v43, v133 :: v_dual_mov_b32 v44, v133
	v_dual_mov_b32 v45, v133 :: v_dual_mov_b32 v46, v133
	v_dual_mov_b32 v47, v133 :: v_dual_mov_b32 v48, v133
	v_dual_mov_b32 v49, v133 :: v_dual_mov_b32 v50, v133
	v_dual_mov_b32 v51, v133 :: v_dual_mov_b32 v52, v133
	v_dual_mov_b32 v53, v133 :: v_dual_mov_b32 v54, v133
	v_dual_mov_b32 v55, v133 :: v_dual_mov_b32 v56, v133
	v_dual_mov_b32 v57, v133 :: v_dual_mov_b32 v58, v133
	v_dual_mov_b32 v59, v133 :: v_dual_mov_b32 v60, v133
	v_dual_mov_b32 v61, v133 :: v_dual_mov_b32 v62, v133
	v_dual_mov_b32 v63, v133 :: v_dual_mov_b32 v64, v133
	v_dual_mov_b32 v65, v133 :: v_dual_mov_b32 v74, v133
	v_dual_mov_b32 v75, v133 :: v_dual_mov_b32 v76, v133
	v_dual_mov_b32 v77, v133 :: v_dual_mov_b32 v78, v133
	v_dual_mov_b32 v79, v133 :: v_dual_mov_b32 v80, v133
	v_dual_mov_b32 v81, v133 :: v_dual_mov_b32 v98, v133
	v_dual_mov_b32 v99, v133 :: v_dual_mov_b32 v100, v133
	v_dual_mov_b32 v101, v133 :: v_dual_mov_b32 v102, v133
	v_dual_mov_b32 v103, v133 :: v_dual_mov_b32 v104, v133
	v_dual_mov_b32 v105, v133 :: v_dual_mov_b32 v106, v133
	v_dual_mov_b32 v107, v133 :: v_dual_mov_b32 v108, v133
	v_dual_mov_b32 v109, v133 :: v_dual_mov_b32 v110, v133
	v_dual_mov_b32 v111, v133 :: v_dual_mov_b32 v112, v133
	v_dual_mov_b32 v113, v133 :: v_dual_mov_b32 v114, v133
	v_dual_mov_b32 v115, v133 :: v_dual_mov_b32 v116, v133
	v_dual_mov_b32 v117, v133 :: v_dual_mov_b32 v118, v133
	v_dual_mov_b32 v119, v133 :: v_dual_mov_b32 v120, v133
	v_dual_mov_b32 v121, v133 :: v_dual_mov_b32 v122, v133
	v_dual_mov_b32 v123, v133 :: v_dual_mov_b32 v124, v133
	v_dual_mov_b32 v125, v133 :: v_dual_mov_b32 v126, v133
	v_dual_mov_b32 v127, v133 :: v_dual_mov_b32 v128, v133
	v_dual_mov_b32 v129, v133 :: v_dual_mov_b32 v66, v133
	v_dual_mov_b32 v67, v133 :: v_dual_mov_b32 v68, v133
	v_dual_mov_b32 v69, v133 :: v_dual_mov_b32 v70, v133
	v_dual_mov_b32 v71, v133 :: v_dual_mov_b32 v72, v133
	v_dual_mov_b32 v73, v133 :: v_dual_mov_b32 v82, v133
	v_dual_mov_b32 v83, v133 :: v_dual_mov_b32 v84, v133
	v_dual_mov_b32 v85, v133 :: v_dual_mov_b32 v86, v133
	v_dual_mov_b32 v87, v133 :: v_dual_mov_b32 v88, v133
	v_dual_mov_b32 v89, v133 :: v_dual_mov_b32 v90, v133
	v_dual_mov_b32 v91, v133 :: v_dual_mov_b32 v92, v133
	v_dual_mov_b32 v93, v133 :: v_dual_mov_b32 v94, v133
	v_dual_mov_b32 v95, v133 :: v_dual_mov_b32 v96, v133
	v_dual_mov_b32 v97, v133 :: v_dual_mov_b32 v131, v136
	v_or_b32_e32 v143, 0xf00, v0
	s_lshr_b32 s46, s3, 16
	s_lshr_b32 s47, s31, 16
	s_lshl_b64 s[4:5], s[4:5], 1
	s_lshl_b64 s[6:7], s[6:7], 1
	s_mov_b32 s41, s1
	s_movk_i32 s16, 0x80
	s_bitset1_b32 s46, 21
	s_bitset1_b32 s47, 21
	s_add_nc_u64 s[20:21], s[14:15], s[4:5]
	s_add_nc_u64 s[36:37], s[18:19], s[6:7]
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
	v_mov_b64_e32 v[170:171], v[0:1]
	v_mov_b32_e32 v145, 16
	s_cmp_lg_u32 s5, 0
	s_mov_b32 s0, 0
	s_cselect_b32 s7, s43, s1
	s_cselect_b32 s6, s42, 0
.LBB0_23:                               ;   Parent Loop BB0_21 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_dual_mov_b32 v132, v170 :: v_dual_add_nc_u32 v145, -2, v145
	v_dual_mov_b32 v172, v171 :: v_dual_mov_b32 v173, v133
	v_add_nc_u32_e32 v171, 0x200, v171
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u64 v[174:175], v[132:133], 2, s[6:7]
	v_cmp_eq_u32_e32 vcc_lo, 0, v145
	v_add_nc_u32_e32 v170, 0x200, v170
	v_lshl_add_u64 v[172:173], v[172:173], 2, s[6:7]
	s_clause 0x1
	flat_store_b32 v[174:175], v133
	flat_store_b32 v[172:173], v133
	s_or_b32 s0, vcc_lo, s0
	s_wait_xcnt 0x0
	s_and_not1_b32 exec_lo, exec_lo, s0
	s_cbranch_execnz .LBB0_23
; %bb.24:                               ;   in Loop: Header=BB0_21 Depth=1
	s_or_b32 exec_lo, exec_lo, s0
	s_and_saveexec_b32 s0, s48
	s_cbranch_execz .LBB0_27
; %bb.25:                               ;   in Loop: Header=BB0_21 Depth=1
	v_add_nc_u64_e32 v[170:171], s[6:7], v[168:169]
	v_mov_b32_e32 v132, v143
	s_mov_b32 s6, 0
.LBB0_26:                               ;   Parent Loop BB0_21 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v132, 0x100, v132
	flat_store_b32 v[170:171], v133
	s_wait_xcnt 0x0
	v_add_nc_u64_e32 v[170:171], 0x400, v[170:171]
	v_cmp_lt_u32_e32 vcc_lo, 0xfff, v132
	s_or_b32 s6, vcc_lo, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s6
	s_cbranch_execnz .LBB0_26
.LBB0_27:                               ;   in Loop: Header=BB0_21 Depth=1
	s_or_b32 exec_lo, exec_lo, s0
	v_mov_b64_e32 v[170:171], v[0:1]
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
	v_add_nc_u32_e32 v171, 0x200, v171
	v_cmp_eq_u32_e32 vcc_lo, s6, v141
	v_add_nc_u32_e32 v170, 0x200, v170
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
	v_cmp_le_u32_e32 vcc_lo, s15, v131
	v_cmpx_le_u32_e64 s14, v136
	s_cbranch_execz .LBB0_31
; %bb.30:                               ;   in Loop: Header=BB0_29 Depth=2
	v_mov_b32_e32 v132, v170
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u64 v[172:173], v[132:133], 2, s[12:13]
	flat_store_b32 v[172:173], v133
.LBB0_31:                               ;   in Loop: Header=BB0_29 Depth=2
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s7
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB0_28
; %bb.32:                               ;   in Loop: Header=BB0_29 Depth=2
	v_mov_b32_e32 v132, v171
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u64 v[172:173], v[132:133], 2, s[12:13]
	flat_store_b32 v[172:173], v133
	s_branch .LBB0_28
.LBB0_33:                               ;   in Loop: Header=BB0_21 Depth=1
	s_or_b32 exec_lo, exec_lo, s10
	s_barrier_signal -1
	s_barrier_wait -1
.LBB0_34:                               ;   in Loop: Header=BB0_21 Depth=1
	s_and_b32 s0, s2, exec_lo
	s_cselect_b32 s0, s49, 0
	s_mov_b32 s2, exec_lo
	v_cmpx_lt_i32_e32 0, v137
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
	v_cmpx_eq_u32_e32 1, v137
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
	v_dual_mov_b32 v145, s14 :: v_dual_mov_b32 v170, s12
	s_and_b32 s10, s13, 0x1ffffff
	s_and_b32 s13, s25, s15
	s_bitset1_b32 s10, 31
	v_cndmask_b32_e64 v132, 0, 1, s13
	v_mov_b32_e32 v147, s10
	v_readfirstlane_b32 s53, v145
	v_readfirstlane_b32 s54, v170
	s_lshr_b64 s[14:15], s[2:3], 16
	v_readfirstlane_b32 s52, v132
	v_readfirstlane_b32 s55, v147
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
	v_dual_mov_b32 v145, s5 :: v_dual_mov_b32 v170, s6
	s_and_b32 s0, s7, 0x1ffffff
	s_and_b32 s7, s29, s12
	s_bitset1_b32 s0, 31
	v_cndmask_b32_e64 v132, 0, 1, s7
	v_mov_b32_e32 v147, s0
	v_readfirstlane_b32 s13, v145
	v_readfirstlane_b32 s14, v170
	s_lshr_b64 s[6:7], s[30:31], 16
	v_readfirstlane_b32 s12, v132
	v_readfirstlane_b32 s15, v147
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
	s_cselect_b32 s2, s34, s39
	v_lshl_add_u32 v132, v130, 1, s0
	v_lshl_add_u32 v145, v138, 1, s0
	v_lshl_add_u32 v147, v140, 1, s0
	v_lshl_add_u32 v149, v142, 1, s0
	ds_load_b128 v[170:173], v132
	ds_load_b128 v[174:177], v132 offset:16
	ds_load_b128 v[178:181], v145 offset:1024
	ds_load_b128 v[182:185], v145 offset:1040
	ds_load_b128 v[186:189], v147 offset:2048
	ds_load_b128 v[190:193], v147 offset:2064
	v_lshl_add_u32 v132, v144, 1, s0
	v_lshl_add_u32 v145, v146, 1, s0
	v_lshl_add_u32 v147, v148, 1, s0
	ds_load_b128 v[194:197], v149 offset:3072
	ds_load_b128 v[198:201], v149 offset:3088
	ds_load_b128 v[202:205], v132 offset:4096
	ds_load_b128 v[206:209], v132 offset:4112
	ds_load_b128 v[210:213], v145 offset:5120
	ds_load_b128 v[214:217], v145 offset:5136
	ds_load_b128 v[218:221], v147 offset:6144
	v_lshl_add_u32 v132, v150, 1, s0
	ds_load_b128 v[222:225], v147 offset:6160
	v_lshl_add_u32 v145, v152, 1, s0
	v_lshl_add_u32 v147, v134, 1, s2
	v_lshl_add_u32 v149, v154, 1, s0
	ds_load_b128 v[226:229], v132 offset:7168
	ds_load_b128 v[230:233], v132 offset:7184
	ds_load_b128 v[234:237], v145 offset:8192
	ds_load_b128 v[242:245], v147
	ds_load_b128 v[246:249], v147 offset:16
	v_lshl_add_u32 v132, v156, 1, s0
	; sched_group_barrier mask(0x00000100) size(17) SyncID(0)
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[114:121], v[178:185], v[242:249], v[114:121] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[106:113], v[186:193], v[242:249], v[106:113] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[98:105], v[194:201], v[242:249], v[98:105] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[74:81], v[202:209], v[242:249], v[74:81] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[58:65], v[210:217], v[242:249], v[58:65] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[50:57], v[218:225], v[242:249], v[50:57] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[42:49], v[226:233], v[242:249], v[42:49] matrix_b_reuse
	ds_load_b128 v[250:253], v149 offset:9216
	ds_load_b128 v[254:257], v149 offset:9232
	ds_load_b128 v[238:241], v145 offset:8208
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x32_bf16 v[26:33], v[250:257], v[242:249], v[26:33] matrix_b_reuse
	ds_load_b128 v[250:253], v132 offset:10240
	ds_load_b128 v[254:257], v132 offset:10256
	v_lshl_add_u32 v132, v158, 1, s0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[18:25], v[250:257], v[242:249], v[18:25] matrix_b_reuse
	ds_load_b128 v[250:253], v132 offset:11264
	ds_load_b128 v[254:257], v132 offset:11280
	v_lshl_add_u32 v132, v160, 1, s0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[10:17], v[250:257], v[242:249], v[10:17] matrix_b_reuse
	ds_load_b128 v[250:253], v132 offset:12288
	ds_load_b128 v[254:257], v132 offset:12304
	v_lshl_add_u32 v132, v162, 1, s0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[2:9], v[250:257], v[242:249], v[2:9] matrix_b_reuse
	ds_load_b128 v[250:253], v132 offset:13312
	ds_load_b128 v[254:257], v132 offset:13328
	v_lshl_add_u32 v132, v164, 1, s0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[66:73], v[250:257], v[242:249], v[66:73] matrix_b_reuse
	ds_load_b128 v[250:253], v132 offset:14336
	ds_load_b128 v[254:257], v132 offset:14352
	v_lshl_add_u32 v132, v166, 1, s0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[82:89], v[250:257], v[242:249], v[82:89] matrix_b_reuse
	ds_load_b128 v[250:253], v132 offset:15360
	ds_load_b128 v[254:257], v132 offset:15376
	; sched_group_barrier mask(0x00000100) size(17) SyncID(0)
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[90:97], v[250:257], v[242:249], v[90:97] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[122:129], v[170:177], v[242:249], v[122:129]
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[34:41], v[234:241], v[242:249], v[34:41] matrix_b_reuse
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
	v_and_or_b32 v1, v0, 15, v139
	v_lshrrev_b32_e32 v130, 1, v0
	v_nop
	v_cvt_pk_bf16_f32 v129, v128, v129
	v_cvt_pk_bf16_f32 v128, v126, v127
	v_cvt_pk_bf16_f32 v126, v122, v123
	v_lshlrev_b32_e32 v1, 8, v1
	v_cvt_pk_bf16_f32 v127, v124, v125
	v_cvt_pk_bf16_f32 v81, v80, v81
	v_cvt_pk_bf16_f32 v80, v78, v79
	v_cvt_pk_bf16_f32 v79, v76, v77
	v_and_or_b32 v122, v130, 8, v1
	v_lshrrev_b32_e32 v1, 3, v1
	v_cvt_pk_bf16_f32 v78, v74, v75
	v_cvt_pk_bf16_f32 v25, v24, v25
	v_cvt_pk_bf16_f32 v24, v22, v23
	v_cvt_pk_bf16_f32 v23, v20, v21
	v_lshl_add_u32 v1, v122, 1, v1
	v_cvt_pk_bf16_f32 v22, v18, v19
	v_cvt_pk_bf16_f32 v121, v120, v121
	v_cvt_pk_bf16_f32 v120, v118, v119
	v_cvt_pk_bf16_f32 v119, v116, v117
	v_cvt_pk_bf16_f32 v118, v114, v115
	v_cvt_pk_bf16_f32 v65, v64, v65
	v_cvt_pk_bf16_f32 v64, v62, v63
	v_cvt_pk_bf16_f32 v63, v60, v61
	v_cvt_pk_bf16_f32 v62, v58, v59
	v_cvt_pk_bf16_f32 v17, v16, v17
	v_cvt_pk_bf16_f32 v16, v14, v15
	v_cvt_pk_bf16_f32 v15, v12, v13
	v_cvt_pk_bf16_f32 v14, v10, v11
	v_cvt_pk_bf16_f32 v113, v112, v113
	v_cvt_pk_bf16_f32 v112, v110, v111
	v_cvt_pk_bf16_f32 v111, v108, v109
	v_cvt_pk_bf16_f32 v110, v106, v107
	v_cvt_pk_bf16_f32 v57, v56, v57
	v_cvt_pk_bf16_f32 v56, v54, v55
	v_cvt_pk_bf16_f32 v55, v52, v53
	v_cvt_pk_bf16_f32 v54, v50, v51
	v_cvt_pk_bf16_f32 v9, v8, v9
	v_cvt_pk_bf16_f32 v8, v6, v7
	v_cvt_pk_bf16_f32 v7, v4, v5
	v_cvt_pk_bf16_f32 v6, v2, v3
	v_cvt_pk_bf16_f32 v105, v104, v105
	v_cvt_pk_bf16_f32 v104, v102, v103
	v_cvt_pk_bf16_f32 v103, v100, v101
	v_cvt_pk_bf16_f32 v102, v98, v99
	v_cvt_pk_bf16_f32 v49, v48, v49
	v_cvt_pk_bf16_f32 v48, v46, v47
	v_cvt_pk_bf16_f32 v47, v44, v45
	v_cvt_pk_bf16_f32 v46, v42, v43
	v_cvt_pk_bf16_f32 v5, v72, v73
	v_cvt_pk_bf16_f32 v4, v70, v71
	v_cvt_pk_bf16_f32 v3, v68, v69
	v_cvt_pk_bf16_f32 v2, v66, v67
	v_cvt_pk_bf16_f32 v41, v40, v41
	v_cvt_pk_bf16_f32 v40, v38, v39
	v_cvt_pk_bf16_f32 v39, v36, v37
	v_cvt_pk_bf16_f32 v38, v34, v35
	v_cvt_pk_bf16_f32 v13, v88, v89
	v_cvt_pk_bf16_f32 v12, v86, v87
	v_cvt_pk_bf16_f32 v11, v84, v85
	v_cvt_pk_bf16_f32 v10, v82, v83
	ds_store_b128 v1, v[126:129]
	ds_store_b128 v1, v[118:121] offset:32
	ds_store_b128 v1, v[110:113] offset:64
	ds_store_b128 v1, v[102:105] offset:96
	v_cvt_pk_bf16_f32 v33, v32, v33
	v_cvt_pk_bf16_f32 v32, v30, v31
	v_cvt_pk_bf16_f32 v31, v28, v29
	v_cvt_pk_bf16_f32 v30, v26, v27
	ds_store_b128 v1, v[78:81] offset:128
	ds_store_b128 v1, v[62:65] offset:160
	ds_store_b128 v1, v[54:57] offset:192
	ds_store_b128 v1, v[46:49] offset:224
	ds_store_b128 v1, v[38:41] offset:272
	ds_store_b128 v1, v[30:33] offset:304
	v_cvt_pk_bf16_f32 v21, v96, v97
	v_cvt_pk_bf16_f32 v20, v94, v95
	v_cvt_pk_bf16_f32 v19, v92, v93
	v_cvt_pk_bf16_f32 v18, v90, v91
	ds_store_b128 v1, v[22:25] offset:336
	ds_store_b128 v1, v[14:17] offset:368
	ds_store_b128 v1, v[6:9] offset:400
	ds_store_b128 v1, v[2:5] offset:432
	ds_store_b128 v1, v[10:13] offset:464
	ds_store_b128 v1, v[18:21] offset:496
.LBB0_44:
	v_cmp_ne_u32_e32 vcc_lo, 1, v135
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
	v_dual_sub_nc_u32 v7, v4, v9 :: v_dual_lshlrev_b32 v4, 8, v4
	v_lshlrev_b32_e32 v9, 8, v9
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
	.size	bm256_bn128_bk032_wm1_wn8_mc0, .Lfunc_end0-bm256_bn128_bk032_wm1_wn8_mc0
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel bm256_bn128_bk032_wm1_wn8_mc0
		.amdhsa_group_segment_fixed_size 69632
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
		.amdhsa_next_free_vgpr 258
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
	.section	.text.bm256_bn128_bk032_wm1_wn8_mc0,"axG",@progbits,bm256_bn128_bk032_wm1_wn8_mc0,comdat
                                        ; -- End function
	.set .Lbm256_bn128_bk032_wm1_wn8_mc0.num_vgpr, 258
	.set .Lbm256_bn128_bk032_wm1_wn8_mc0.num_agpr, 0
	.set .Lbm256_bn128_bk032_wm1_wn8_mc0.numbered_sgpr, 56
	.set .Lbm256_bn128_bk032_wm1_wn8_mc0.num_named_barrier, 0
	.set .Lbm256_bn128_bk032_wm1_wn8_mc0.private_seg_size, 0
	.set .Lbm256_bn128_bk032_wm1_wn8_mc0.uses_vcc, 1
	.set .Lbm256_bn128_bk032_wm1_wn8_mc0.uses_flat_scratch, 1
	.set .Lbm256_bn128_bk032_wm1_wn8_mc0.has_dyn_sized_stack, 0
	.set .Lbm256_bn128_bk032_wm1_wn8_mc0.has_recursion, 0
	.set .Lbm256_bn128_bk032_wm1_wn8_mc0.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 6760
; TotalNumSgprs: 58
; NumVgprs: 258
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 69632 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 16
; NumSGPRsForWavesPerEU: 58
; NumVGPRsForWavesPerEU: 258
; NamedBarCnt: 0
; Occupancy: 3
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
	.type	__hip_cuid_9ffd4dc3c2e3e582,@object ; @__hip_cuid_9ffd4dc3c2e3e582
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_9ffd4dc3c2e3e582
__hip_cuid_9ffd4dc3c2e3e582:
	.byte	0                               ; 0x0
	.size	__hip_cuid_9ffd4dc3c2e3e582, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git aa451e1fe6a793394d6733051b1778633063ae96+PATCHED:d17c5aa0e3ea29cde402f58f27e39b6034effa27)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_9ffd4dc3c2e3e582
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
    .group_segment_fixed_size: 69632
    .kernarg_segment_align: 8
    .kernarg_segment_size: 132
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           bm256_bn128_bk032_wm1_wn8_mc0
    .private_segment_fixed_size: 0
    .sgpr_count:     58
    .sgpr_spill_count: 0
    .symbol:         bm256_bn128_bk032_wm1_wn8_mc0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     258
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
