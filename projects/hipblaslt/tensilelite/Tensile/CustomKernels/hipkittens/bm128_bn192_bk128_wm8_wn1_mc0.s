	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.section	.text.bm128_bn192_bk128_wm8_wn1_mc0,"axG",@progbits,bm128_bn192_bk128_wm8_wn1_mc0,comdat
	.protected	bm128_bn192_bk128_wm8_wn1_mc0 ; -- Begin function bm128_bn192_bk128_wm8_wn1_mc0
	.globl	bm128_bn192_bk128_wm8_wn1_mc0
	.p2align	8
	.type	bm128_bn192_bk128_wm8_wn1_mc0,@function
bm128_bn192_bk128_wm8_wn1_mc0: ; @bm128_bn192_bk128_wm8_wn1_mc0
; %bb.0:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1 ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_mov_b64 s[2:3], src_shared_base
	s_mov_b32 s2, 0x8800
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
	s_cselect_b32 s33, ttmp7, s5
	s_wait_kmcnt 0x0
	s_add_co_i32 s3, s20, 0x7f
	s_add_co_i32 s5, s21, 0xbf
	s_ashr_i32 s4, s3, 31
	s_lshl_b32 s24, s35, 7
	s_lshr_b32 s4, s4, 25
	s_sub_co_i32 s6, s20, s24
	s_add_co_i32 s3, s3, s4
	s_mul_hi_i32 s4, s5, 0x2aaaaaab
	s_ashr_i32 s3, s3, 7
	s_lshr_b32 s5, s4, 31
	s_ashr_i32 s4, s4, 5
	s_min_i32 s23, s6, 0x80
	s_add_co_i32 s4, s4, s5
	s_cmp_lt_i32 s35, s3
	s_cselect_b32 s25, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_and_b32 s5, s25, exec_lo
	s_mul_i32 s5, s33, 0xffffff40
	s_cselect_b32 s27, s23, 0
	s_add_co_i32 s5, s21, s5
	s_min_i32 s6, s5, 0xc0
	s_cmp_lt_i32 s33, s4
	s_mov_b32 s5, s22
	s_cselect_b32 s21, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s7, s21, exec_lo
	s_cselect_b32 s29, s6, 0
	s_add_co_i32 s13, s22, 0x7f
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 2, 1), 1
	s_min_i32 s6, s22, 0x80
	s_cmp_gt_i32 s13, 0x7f
	s_cselect_b32 s12, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_and_b32 s7, s12, exec_lo
	s_cselect_b32 s26, s6, 0
	s_cmp_lt_i32 s27, 0x80
	s_cselect_b32 s38, -1, 0
	s_and_b32 vcc_lo, exec_lo, s38
	s_mov_b32 s6, s38
	s_cbranch_vccnz .LBB0_2
; %bb.1:
	s_cmp_lt_i32 s29, 0xc0
	s_cselect_b32 s6, -1, 0
	s_cmp_lt_i32 s26, 0x80
	s_cselect_b32 s7, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 s6, s7, s6
.LBB0_2:
	v_lshlrev_b32_e32 v98, 2, v0
	s_and_not1_b32 vcc_lo, exec_lo, s6
	s_cbranch_vccnz .LBB0_8
; %bb.3:
	v_or_b32_e32 v1, 0xffffff00, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v3, v98
	s_mov_b32 s6, 0
	v_mov_b32_e32 v4, v1
.LBB0_4:                                ; =>This Inner Loop Header: Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)
	v_add_nc_u32_e32 v4, 0x100, v4
	ds_store_b32 v3, v2
	v_add_nc_u32_e32 v3, 0x400, v3
	v_cmp_lt_u32_e32 vcc_lo, 0x20ff, v4
	s_or_b32 s6, vcc_lo, s6
	s_and_not1_b32 exec_lo, exec_lo, s6
	s_cbranch_execnz .LBB0_4
; %bb.5:
	s_or_b32 exec_lo, exec_lo, s6
	v_lshl_add_u32 v2, s2, 2, v98
	v_mov_b32_e32 v3, 0
	s_mov_b32 s6, 0
.LBB0_6:                                ; =>This Inner Loop Header: Depth=1
	v_add_nc_u32_e32 v1, 0x100, v1
	ds_store_b32 v2, v3 offset:34816
	v_add_nc_u32_e32 v2, 0x400, v2
	v_cmp_lt_u32_e32 vcc_lo, 0x31ff, v1
	s_or_b32 s6, vcc_lo, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s6
	s_cbranch_execnz .LBB0_6
; %bb.7:
	s_or_b32 exec_lo, exec_lo, s6
	s_barrier_signal -1
	s_barrier_wait -1
.LBB0_8:
	s_clause 0x2
	s_load_b64 s[14:15], s[0:1], 0x0 nv
	s_load_b128 s[8:11], s[0:1], 0x20 nv
	s_load_b128 s[16:19], s[0:1], 0x48 nv
	v_lshrrev_b32_e32 v103, 5, v0
	s_wait_xcnt 0x0
	s_lshl_b32 s0, s2, 2
	s_add_co_i32 s4, s4, -1
	s_mov_b64 s[30:31], src_shared_base
	s_or_b32 s39, s0, 0x8800
	s_add_co_i32 s30, s3, -1
	s_min_i32 s34, s33, s4
	s_mov_b32 s0, exec_lo
	v_cmpx_lt_i32_e32 0, v103
	s_xor_b32 s36, exec_lo, s0
	s_cbranch_execz .LBB0_12
; %bb.9:
	s_mov_b32 s37, exec_lo
	v_cmpx_eq_u32_e32 1, v103
	s_cbranch_execz .LBB0_11
; %bb.10:
	s_cmp_gt_i32 s26, 0
	s_mul_i32 s0, s34, 0xc0
	s_cselect_b32 s4, -1, 0
	s_ashr_i32 s1, s0, 31
	s_wait_kmcnt 0x0
	s_bfe_i64 s[2:3], s[16:17], 0x200000
	s_mov_b32 s28, s26
	s_mul_u64 s[0:1], s[2:3], s[0:1]
	s_and_b32 s2, s21, s4
	s_lshl_b64 s[0:1], s[0:1], 1
	v_cndmask_b32_e64 v2, 0, 1, s2
	s_add_nc_u64 s[0:1], s[10:11], s[0:1]
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_mov_b32 v1, s39 :: v_dual_mov_b32 v4, s0
	s_and_b32 s1, s1, 0x1ffffff
	v_readfirstlane_b32 s40, v2
	s_bitset1_b32 s1, 31
	v_readfirstlane_b32 s41, v1
	v_mov_b32_e32 v3, s1
	v_readfirstlane_b32 s42, v4
	s_mov_b32 s6, 0
	s_lshr_b32 s0, s29, 16
	s_lshr_b64 s[2:3], s[28:29], 16
	v_readfirstlane_b32 s43, v3
	s_movk_i32 s4, 0xc0
	s_lshl_b32 s1, s26, 16
	s_or_b32 s3, s0, 0x800000
	s_mov_b32 s0, 0x7510000
	s_mov_b32 s7, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[40:43], s[0:7]
.LBB0_11:
	s_or_b32 exec_lo, exec_lo, s37
.LBB0_12:
	s_or_saveexec_b32 s36, s36
	s_min_i32 s28, s35, s30
	s_xor_b32 exec_lo, exec_lo, s36
	s_cbranch_execz .LBB0_14
; %bb.13:
	s_cmp_gt_i32 s26, 0
	s_mov_b32 s6, 0
	s_cselect_b32 s4, -1, 0
	s_lshl_b32 s0, s28, 7
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
	s_movk_i32 s4, 0x80
	v_readfirstlane_b32 s42, v4
	v_readfirstlane_b32 s43, v3
	s_mov_b32 s0, 0x7510000
	s_mov_b32 s7, s6
	s_mov_b32 s41, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[40:43], s[0:7]
.LBB0_14:
	s_or_b32 exec_lo, exec_lo, s36
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	v_dual_mov_b32 v9, 0 :: v_dual_lshlrev_b32 v110, 7, v0
	s_and_b32 s30, s25, s21
	s_and_not1_b32 vcc_lo, exec_lo, s12
	v_cndmask_b32_e64 v101, 0, 1, s30
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
	v_dual_mov_b32 v42, v9 :: v_dual_mov_b32 v65, v9
	v_dual_mov_b32 v64, v9 :: v_dual_mov_b32 v63, v9
	v_dual_mov_b32 v62, v9 :: v_dual_mov_b32 v61, v9
	v_dual_mov_b32 v60, v9 :: v_dual_mov_b32 v59, v9
	v_dual_mov_b32 v58, v9 :: v_dual_mov_b32 v73, v9
	v_dual_mov_b32 v72, v9 :: v_dual_mov_b32 v71, v9
	v_dual_mov_b32 v70, v9 :: v_dual_mov_b32 v69, v9
	v_dual_mov_b32 v68, v9 :: v_dual_mov_b32 v67, v9
	v_dual_mov_b32 v66, v9 :: v_dual_mov_b32 v81, v9
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
	v_mov_b32_e32 v50, v9
	s_barrier_wait -1
	s_cbranch_vccnz .LBB0_37
; %bb.15:
	s_mov_b64 s[0:1], src_shared_base
	s_add_co_i32 s2, s39, 0xcc00
	s_mov_b32 s3, s1
	v_and_b32_e32 v1, 0x780, v110
	s_and_b64 s[2:3], s[2:3], 15
	s_mov_b32 s7, 0
	s_sub_co_i32 s0, 16, s2
	s_mov_b32 s42, s1
	s_lshr_b32 s0, s0, 2
	s_cmp_lg_u64 s[2:3], 0
	v_and_or_b32 v2, v0, 16, v1
	s_cselect_b32 s0, s0, 0
	s_mul_i32 s2, s34, 0xc0
	s_lshl2_add_u32 s3, s0, s39
	v_or_b32_e32 v111, 0x2100, v0
	s_add_co_i32 s0, s3, 0x15400
	s_add_co_i32 s41, s3, 0xcc00
	s_and_b32 s6, s0, 15
	v_lshl_or_b32 v3, v103, 11, v2
	s_sub_co_i32 s4, 16, s6
	v_lshrrev_b32_e32 v1, 4, v1
	s_lshr_b32 s3, s4, 2
	s_cmp_lg_u64 s[6:7], 0
	v_lshrrev_b32_e32 v4, 4, v3
	s_cselect_b32 s3, s3, 0
	s_ashr_i32 s4, s13, 31
	s_lshl_b32 s6, s3, 2
	s_lshr_b32 s4, s4, 25
	s_add_nc_u64 s[34:35], s[0:1], s[6:7]
	s_add_co_i32 s13, s13, s4
	v_and_b32_e32 v4, 0x3f8, v4
	s_ashr_i32 s43, s13, 7
	s_cmp_lt_i32 s29, 0xc0
	v_mov_b32_e32 v99, 0
	s_cselect_b32 s44, -1, 0
	s_lshl_b32 s0, s28, 7
	s_ashr_i32 s3, s2, 31
	s_ashr_i32 s1, s0, 31
	s_wait_kmcnt 0x0
	s_bfe_i64 s[16:17], s[16:17], 0x200000
	s_bfe_i64 s[8:9], s[8:9], 0x200000
	s_mul_u64 s[2:3], s[16:17], s[2:3]
	s_mul_u64 s[0:1], s[8:9], s[0:1]
	v_dual_add_nc_u32 v100, v4, v3 :: v_dual_add_nc_u32 v102, v2, v1
	v_mov_b32_e32 v105, v99
	v_or_b32_e32 v1, 0x100, v0
	v_or_b32_e32 v104, 0x8800, v98
	v_or_b32_e32 v112, 0x3100, v0
	v_or_b32_e32 v106, 0xc800, v98
	v_dual_mov_b32 v107, v99 :: v_dual_mov_b32 v2, v99
	v_dual_mov_b32 v3, v99 :: v_dual_mov_b32 v4, v99
	v_dual_mov_b32 v5, v99 :: v_dual_mov_b32 v6, v99
	v_dual_mov_b32 v7, v99 :: v_dual_mov_b32 v8, v99
	v_dual_mov_b32 v9, v99 :: v_dual_mov_b32 v10, v99
	v_dual_mov_b32 v11, v99 :: v_dual_mov_b32 v12, v99
	v_dual_mov_b32 v13, v99 :: v_dual_mov_b32 v14, v99
	v_dual_mov_b32 v15, v99 :: v_dual_mov_b32 v16, v99
	v_dual_mov_b32 v17, v99 :: v_dual_mov_b32 v18, v99
	v_dual_mov_b32 v19, v99 :: v_dual_mov_b32 v20, v99
	v_dual_mov_b32 v21, v99 :: v_dual_mov_b32 v22, v99
	v_dual_mov_b32 v23, v99 :: v_dual_mov_b32 v24, v99
	v_dual_mov_b32 v25, v99 :: v_dual_mov_b32 v26, v99
	v_dual_mov_b32 v27, v99 :: v_dual_mov_b32 v28, v99
	v_dual_mov_b32 v29, v99 :: v_dual_mov_b32 v30, v99
	v_dual_mov_b32 v31, v99 :: v_dual_mov_b32 v32, v99
	v_dual_mov_b32 v33, v99 :: v_dual_mov_b32 v34, v99
	v_dual_mov_b32 v35, v99 :: v_dual_mov_b32 v36, v99
	v_dual_mov_b32 v37, v99 :: v_dual_mov_b32 v38, v99
	v_dual_mov_b32 v39, v99 :: v_dual_mov_b32 v40, v99
	v_dual_mov_b32 v41, v99 :: v_dual_mov_b32 v42, v99
	v_dual_mov_b32 v43, v99 :: v_dual_mov_b32 v44, v99
	v_dual_mov_b32 v45, v99 :: v_dual_mov_b32 v46, v99
	v_dual_mov_b32 v47, v99 :: v_dual_mov_b32 v48, v99
	v_dual_mov_b32 v49, v99 :: v_dual_mov_b32 v58, v99
	v_dual_mov_b32 v59, v99 :: v_dual_mov_b32 v60, v99
	v_dual_mov_b32 v61, v99 :: v_dual_mov_b32 v62, v99
	v_dual_mov_b32 v63, v99 :: v_dual_mov_b32 v64, v99
	v_dual_mov_b32 v65, v99 :: v_dual_mov_b32 v66, v99
	v_dual_mov_b32 v67, v99 :: v_dual_mov_b32 v68, v99
	v_dual_mov_b32 v69, v99 :: v_dual_mov_b32 v70, v99
	v_dual_mov_b32 v71, v99 :: v_dual_mov_b32 v72, v99
	v_dual_mov_b32 v73, v99 :: v_dual_mov_b32 v74, v99
	v_dual_mov_b32 v75, v99 :: v_dual_mov_b32 v76, v99
	v_dual_mov_b32 v77, v99 :: v_dual_mov_b32 v78, v99
	v_dual_mov_b32 v79, v99 :: v_dual_mov_b32 v80, v99
	v_dual_mov_b32 v81, v99 :: v_dual_mov_b32 v82, v99
	v_dual_mov_b32 v83, v99 :: v_dual_mov_b32 v84, v99
	v_dual_mov_b32 v85, v99 :: v_dual_mov_b32 v86, v99
	v_dual_mov_b32 v87, v99 :: v_dual_mov_b32 v88, v99
	v_dual_mov_b32 v89, v99 :: v_dual_mov_b32 v90, v99
	v_dual_mov_b32 v91, v99 :: v_dual_mov_b32 v92, v99
	v_dual_mov_b32 v93, v99 :: v_dual_mov_b32 v94, v99
	v_dual_mov_b32 v95, v99 :: v_dual_mov_b32 v96, v99
	v_dual_mov_b32 v97, v99 :: v_dual_mov_b32 v50, v99
	v_dual_mov_b32 v51, v99 :: v_dual_mov_b32 v52, v99
	v_dual_mov_b32 v53, v99 :: v_dual_mov_b32 v54, v99
	v_dual_mov_b32 v55, v99 :: v_dual_mov_b32 v56, v99
	v_mov_b32_e32 v57, v99
	s_lshr_b32 s45, s29, 16
	s_lshr_b32 s46, s27, 16
	s_lshl_b64 s[2:3], s[2:3], 1
	s_lshl_b64 s[0:1], s[0:1], 1
	s_mov_b32 s40, s31
	s_movk_i32 s12, 0xc0
	s_bitset1_b32 s45, 23
	s_bitset1_b32 s46, 23
	s_add_nc_u64 s[16:17], s[10:11], s[2:3]
	s_add_nc_u64 s[36:37], s[14:15], s[0:1]
	s_mov_b32 s47, -1
	s_movk_i32 s4, 0x80
	s_mov_b32 s0, 0x7510000
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
	v_mov_b64_e32 v[108:109], v[0:1]
	v_mov_b32_e32 v113, 34
	s_cmp_lg_u32 s1, 0
	s_mov_b32 s8, 0
	s_cselect_b32 s3, s42, s31
	s_cselect_b32 s2, s41, 0
.LBB0_19:                               ;   Parent Loop BB0_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_dual_mov_b32 v98, v108 :: v_dual_add_nc_u32 v113, -2, v113
	v_dual_mov_b32 v114, v109 :: v_dual_mov_b32 v115, v99
	v_add_nc_u32_e32 v109, 0x200, v109
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u64 v[116:117], v[98:99], 2, s[2:3]
	v_cmp_eq_u32_e32 vcc_lo, 0, v113
	v_add_nc_u32_e32 v108, 0x200, v108
	v_lshl_add_u64 v[114:115], v[114:115], 2, s[2:3]
	s_clause 0x1
	flat_store_b32 v[116:117], v99
	flat_store_b32 v[114:115], v99
	s_or_b32 s8, vcc_lo, s8
	s_wait_xcnt 0x0
	s_and_not1_b32 exec_lo, exec_lo, s8
	s_cbranch_execnz .LBB0_19
; %bb.20:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s8
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_23
; %bb.21:                               ;   in Loop: Header=BB0_17 Depth=1
	v_add_nc_u64_e32 v[108:109], s[2:3], v[104:105]
	v_mov_b32_e32 v98, v111
	s_mov_b32 s2, 0
.LBB0_22:                               ;   Parent Loop BB0_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v98, 0x100, v98
	flat_store_b32 v[108:109], v99
	s_wait_xcnt 0x0
	v_add_nc_u64_e32 v[108:109], 0x400, v[108:109]
	v_cmp_lt_u32_e32 vcc_lo, 0x20ff, v98
	s_or_b32 s2, vcc_lo, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s2
	s_cbranch_execnz .LBB0_22
.LBB0_23:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s8
	v_mov_b64_e32 v[108:109], v[0:1]
	v_mov_b32_e32 v113, 50
	s_cmp_lg_u32 s1, 0
	s_mov_b32 s8, 0
	s_cselect_b32 s3, s35, s40
	s_cselect_b32 s2, s34, s39
.LBB0_24:                               ;   Parent Loop BB0_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_dual_mov_b32 v98, v108 :: v_dual_add_nc_u32 v113, -2, v113
	v_dual_mov_b32 v114, v109 :: v_dual_mov_b32 v115, v99
	v_add_nc_u32_e32 v109, 0x200, v109
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u64 v[116:117], v[98:99], 2, s[2:3]
	v_cmp_eq_u32_e32 vcc_lo, 0, v113
	v_add_nc_u32_e32 v108, 0x200, v108
	v_lshl_add_u64 v[114:115], v[114:115], 2, s[2:3]
	s_clause 0x1
	flat_store_b32 v[116:117], v99
	flat_store_b32 v[114:115], v99
	s_or_b32 s8, vcc_lo, s8
	s_wait_xcnt 0x0
	s_and_not1_b32 exec_lo, exec_lo, s8
	s_cbranch_execnz .LBB0_24
; %bb.25:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s8
	s_and_saveexec_b32 s8, s47
	s_cbranch_execz .LBB0_28
; %bb.26:                               ;   in Loop: Header=BB0_17 Depth=1
	v_add_nc_u64_e32 v[108:109], s[2:3], v[106:107]
	v_mov_b32_e32 v98, v112
	s_mov_b32 s2, 0
.LBB0_27:                               ;   Parent Loop BB0_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v98, 0x100, v98
	flat_store_b32 v[108:109], v99
	s_wait_xcnt 0x0
	v_add_nc_u64_e32 v[108:109], 0x400, v[108:109]
	v_cmp_lt_u32_e32 vcc_lo, 0x31ff, v98
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
	v_cmpx_lt_i32_e32 0, v103
	s_xor_b32 s3, exec_lo, s3
	s_cbranch_execnz .LBB0_32
; %bb.30:                               ;   in Loop: Header=BB0_17 Depth=1
	s_and_not1_saveexec_b32 s8, s3
	s_cbranch_execnz .LBB0_35
.LBB0_31:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s8
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s30
	s_cbranch_vccnz .LBB0_16
	s_branch .LBB0_36
.LBB0_32:                               ;   in Loop: Header=BB0_17 Depth=1
	s_mov_b32 s50, exec_lo
	v_cmpx_eq_u32_e32 1, v103
	s_cbranch_execz .LBB0_34
; %bb.33:                               ;   in Loop: Header=BB0_17 Depth=1
	s_cmp_lg_u32 s1, 0
	s_mov_b32 s28, s26
	s_cselect_b32 s10, s34, s39
	s_cmp_gt_i32 s26, 0
	s_mov_b32 s13, s5
	s_cselect_b32 s11, -1, 0
	s_lshl_b32 s6, s2, 7
	s_mov_b32 s14, s7
	s_lshl_b64 s[8:9], s[6:7], 1
	s_mov_b32 s15, s7
	s_add_nc_u64 s[8:9], s[16:17], s[8:9]
	s_delay_alu instid0(SALU_CYCLE_1)
	v_dual_mov_b32 v109, s10 :: v_dual_mov_b32 v108, s8
	s_and_b32 s6, s9, 0x1ffffff
	s_and_b32 s9, s21, s11
	s_bitset1_b32 s6, 31
	v_cndmask_b32_e64 v98, 0, 1, s9
	v_mov_b32_e32 v113, s6
	v_readfirstlane_b32 s53, v109
	v_readfirstlane_b32 s54, v108
	s_lshr_b64 s[10:11], s[28:29], 16
	v_readfirstlane_b32 s52, v98
	v_readfirstlane_b32 s55, v113
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
	v_cndmask_b32_e64 v98, 0, 1, s6
	s_and_b32 s3, s3, 0x1ffffff
	v_dual_mov_b32 v109, s1 :: v_dual_mov_b32 v108, s2
	s_bitset1_b32 s3, 31
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_readfirstlane_b32 s52, v98
	v_mov_b32_e32 v113, s3
	v_readfirstlane_b32 s53, v109
	v_readfirstlane_b32 s54, v108
	s_lshr_b64 s[2:3], s[26:27], 16
	s_lshl_b32 s1, s26, 16
	v_readfirstlane_b32 s55, v113
	s_mov_b32 s3, s46
	s_mov_b32 s6, s7
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	tensor_load_to_lds s[52:55], s[0:7]
	s_or_b32 exec_lo, exec_lo, s8
	s_and_not1_b32 vcc_lo, exec_lo, s30
	s_cbranch_vccnz .LBB0_16
.LBB0_36:                               ;   in Loop: Header=BB0_17 Depth=1
	s_cmp_lg_u32 s49, 0
	s_cselect_b32 s1, s41, 0
	s_cselect_b32 s2, s34, s39
	v_lshl_add_u32 v98, v100, 1, s1
	v_lshl_add_u32 v108, v102, 1, s2
	ds_load_b128 v[114:117], v98
	ds_load_b128 v[118:121], v98 offset:16
	ds_load_b128 v[122:125], v108
	ds_load_b128 v[126:129], v108 offset:16
	ds_load_b128 v[130:133], v108 offset:4352
	ds_load_b128 v[134:137], v108 offset:4368
	ds_load_b128 v[138:141], v108 offset:8704
	ds_load_b128 v[142:145], v108 offset:8720
	ds_load_b128 v[146:149], v108 offset:13056
	ds_load_b128 v[150:153], v108 offset:13072
	ds_load_b128 v[154:157], v108 offset:17408
	ds_load_b128 v[158:161], v108 offset:17424
	ds_load_b128 v[162:165], v108 offset:21760
	ds_load_b128 v[170:173], v108 offset:64
	ds_load_b128 v[174:177], v108 offset:80
	ds_load_b128 v[178:181], v108 offset:8768
	ds_load_b128 v[182:185], v108 offset:8784
	ds_load_b128 v[186:189], v108 offset:13120
	ds_load_b128 v[190:193], v108 offset:13136
	ds_load_b128 v[194:197], v108 offset:17472
	s_wait_dscnt 0x10
	v_wmma_f32_16x16x32_bf16 v[90:97], v[114:121], v[122:129], v[90:97]
	ds_load_b128 v[198:201], v108 offset:17488
	ds_load_b128 v[202:205], v108 offset:26176
	ds_load_b128 v[206:209], v108 offset:26192
	ds_load_b128 v[210:213], v108 offset:30528
	ds_load_b128 v[214:217], v108 offset:30544
	ds_load_b128 v[218:221], v108 offset:39232
	ds_load_b128 v[222:225], v108 offset:39248
	s_wait_dscnt 0x15
	v_wmma_f32_16x16x32_bf16 v[82:89], v[114:121], v[130:137], v[82:89] matrix_a_reuse
	ds_load_b128 v[226:229], v108 offset:43584
	ds_load_b128 v[230:233], v108 offset:43600
	ds_load_b128 v[234:237], v108 offset:47936
	ds_load_b128 v[238:241], v108 offset:47952
	; sched_group_barrier mask(0x00000100) size(13) SyncID(0)
	s_wait_dscnt 0x17
	v_wmma_f32_16x16x32_bf16 v[74:81], v[114:121], v[138:145], v[74:81] matrix_a_reuse
	s_wait_dscnt 0x15
	v_wmma_f32_16x16x32_bf16 v[66:73], v[114:121], v[146:153], v[66:73] matrix_a_reuse
	s_wait_dscnt 0x13
	v_wmma_f32_16x16x32_bf16 v[58:65], v[114:121], v[154:161], v[58:65] matrix_a_reuse
	ds_load_b128 v[122:125], v108 offset:47872
	ds_load_b128 v[126:129], v108 offset:47888
	ds_load_b128 v[166:169], v108 offset:21776
	ds_load_b128 v[130:133], v108 offset:30464
	ds_load_b128 v[134:137], v108 offset:30480
	ds_load_b128 v[138:141], v108 offset:34816
	ds_load_b128 v[142:145], v108 offset:34832
	ds_load_b128 v[146:149], v108 offset:39168
	ds_load_b128 v[150:153], v108 offset:39184
	ds_load_b128 v[154:157], v108 offset:43520
	ds_load_b128 v[158:161], v108 offset:43536
	s_wait_dscnt 0x9
	v_wmma_f32_16x16x32_bf16 v[50:57], v[114:121], v[122:129], v[50:57] matrix_a_reuse
	ds_load_b128 v[122:125], v108 offset:26112
	ds_load_b128 v[126:129], v108 offset:26128
	; sched_group_barrier mask(0x00000008) size(6) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(13) SyncID(0)
	s_wait_dscnt 0xa
	v_wmma_f32_16x16x32_bf16 v[42:49], v[114:121], v[162:169], v[42:49] matrix_a_reuse
	ds_load_b128 v[162:165], v98 offset:64
	ds_load_b128 v[166:169], v98 offset:80
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[34:41], v[114:121], v[122:129], v[34:41] matrix_a_reuse
	ds_load_b128 v[122:125], v108 offset:4416
	ds_load_b128 v[126:129], v108 offset:4432
	v_wmma_f32_16x16x32_bf16 v[26:33], v[114:121], v[130:137], v[26:33] matrix_a_reuse
	ds_load_b128 v[130:133], v108 offset:21824
	ds_load_b128 v[134:137], v108 offset:21840
	v_wmma_f32_16x16x32_bf16 v[18:25], v[114:121], v[138:145], v[18:25] matrix_a_reuse
	ds_load_b128 v[138:141], v108 offset:34880
	ds_load_b128 v[142:145], v108 offset:34896
	v_wmma_f32_16x16x32_bf16 v[10:17], v[114:121], v[146:153], v[10:17] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(6) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[2:9], v[114:121], v[154:161], v[2:9] matrix_a_reuse
	; sched_barrier mask(0x00000000)
	ds_load_b128 v[114:117], v98 offset:128
	ds_load_b128 v[118:121], v98 offset:144
	ds_load_b128 v[146:149], v108 offset:128
	ds_load_b128 v[150:153], v108 offset:144
	ds_load_b128 v[154:157], v108 offset:4480
	ds_load_b128 v[158:161], v108 offset:4496
	ds_load_b128 v[242:245], v108 offset:8832
	ds_load_b128 v[246:249], v108 offset:8848
	ds_load_b128 v[250:253], v108 offset:13184
	ds_load_b128 v[254:257], v108 offset:13200
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[2:5] /*v[258:261]*/, v108 offset:17536
	ds_load_b128 v[6:9] /*v[262:265]*/, v108 offset:17552
	ds_load_b128 v[10:13] /*v[266:269]*/, v108 offset:21888
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_wait_dscnt 0x13
	v_wmma_f32_16x16x32_bf16 v[90:97], v[162:169], v[170:177], v[90:97]
	; sched_group_barrier mask(0x00000100) size(13) SyncID(0)
	s_wait_dscnt 0x11
	v_wmma_f32_16x16x32_bf16 v[82:89], v[162:169], v[122:129], v[82:89] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[74:81], v[162:169], v[178:185], v[74:81] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[66:73], v[162:169], v[186:193], v[66:73] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[58:65], v[162:169], v[194:201], v[58:65] matrix_a_reuse
	s_wait_dscnt 0xf
	v_wmma_f32_16x16x32_bf16 v[42:49], v[162:169], v[130:137], v[42:49] matrix_a_reuse
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[14:17] /*v[270:273]*/, v108 offset:21904
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	ds_load_b128 v[122:125], v108 offset:26240
	ds_load_b128 v[126:129], v108 offset:26256
	ds_load_b128 v[130:133], v108 offset:30592
	ds_load_b128 v[134:137], v108 offset:30608
	ds_load_b128 v[170:173], v108 offset:34944
	ds_load_b128 v[174:177], v108 offset:34960
	ds_load_b128 v[178:181], v108 offset:39296
	ds_load_b128 v[182:185], v108 offset:39312
	ds_load_b128 v[186:189], v108 offset:43648
	ds_load_b128 v[190:193], v108 offset:43664
	ds_load_b128 v[194:197], v108 offset:48000
	ds_load_b128 v[198:201], v108 offset:48016
	v_wmma_f32_16x16x32_bf16 v[34:41], v[162:169], v[202:209], v[34:41] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(6) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(13) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[26:33], v[162:169], v[210:217], v[26:33] matrix_a_reuse
	s_wait_dscnt 0x1a
	v_wmma_f32_16x16x32_bf16 v[18:25], v[162:169], v[138:145], v[18:25] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[10:17], v[162:169], v[218:225], v[10:17] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[2:9], v[162:169], v[226:233], v[2:9] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[50:57], v[162:169], v[234:241], v[50:57] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(6) SyncID(0)
	; sched_barrier mask(0x00000000)
	ds_load_b128 v[138:141], v98 offset:192
	ds_load_b128 v[142:145], v98 offset:208
	ds_load_b128 v[162:165], v108 offset:192
	ds_load_b128 v[166:169], v108 offset:208
	ds_load_b128 v[202:205], v108 offset:4544
	ds_load_b128 v[206:209], v108 offset:4560
	ds_load_b128 v[210:213], v108 offset:8896
	ds_load_b128 v[214:217], v108 offset:8912
	ds_load_b128 v[218:221], v108 offset:13248
	ds_load_b128 v[222:225], v108 offset:13264
	ds_load_b128 v[226:229], v108 offset:17600
	ds_load_b128 v[230:233], v108 offset:17616
	ds_load_b128 v[234:237], v108 offset:21952
	s_wait_dscnt 0x23
	v_wmma_f32_16x16x32_bf16 v[90:97], v[114:121], v[146:153], v[90:97]
	; sched_group_barrier mask(0x00000100) size(13) SyncID(0)
	s_wait_dscnt 0x21
	v_wmma_f32_16x16x32_bf16 v[82:89], v[114:121], v[154:161], v[82:89] matrix_a_reuse
	s_wait_dscnt 0x1f
	v_wmma_f32_16x16x32_bf16 v[74:81], v[114:121], v[242:249], v[74:81] matrix_a_reuse
	s_wait_dscnt 0x1d
	v_wmma_f32_16x16x32_bf16 v[66:73], v[114:121], v[250:257], v[66:73] matrix_a_reuse
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	s_wait_dscnt 0x1b
	v_wmma_f32_16x16x32_bf16 v[58:65], v[114:121], v[2:9] /*v[258:265]*/, v[58:65] matrix_a_reuse
	s_wait_dscnt 0x19
	v_wmma_f32_16x16x32_bf16 v[42:49], v[114:121], v[10:17] /*v[266:273]*/, v[42:49] matrix_a_reuse
	ds_load_b128 v[238:241], v108 offset:21968
	ds_load_b128 v[146:149], v108 offset:26304
	ds_load_b128 v[150:153], v108 offset:26320
	ds_load_b128 v[154:157], v108 offset:30656
	ds_load_b128 v[158:161], v108 offset:30672
	ds_load_b128 v[242:245], v108 offset:35008
	ds_load_b128 v[246:249], v108 offset:35024
	ds_load_b128 v[250:253], v108 offset:39360
	ds_load_b128 v[254:257], v108 offset:39376
	s_set_vgpr_msb 0x440                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[2:5] /*v[258:261]*/, v108 offset:43712
	ds_load_b128 v[6:9] /*v[262:265]*/, v108 offset:43728
	ds_load_b128 v[10:13] /*v[266:269]*/, v108 offset:48064
	ds_load_b128 v[14:17] /*v[270:273]*/, v108 offset:48080
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_wait_dscnt 0x24
	v_wmma_f32_16x16x32_bf16 v[34:41], v[114:121], v[122:129], v[34:41] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(6) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(13) SyncID(0)
	s_wait_dscnt 0x22
	v_wmma_f32_16x16x32_bf16 v[26:33], v[114:121], v[130:137], v[26:33] matrix_a_reuse
	s_wait_dscnt 0x20
	v_wmma_f32_16x16x32_bf16 v[18:25], v[114:121], v[170:177], v[18:25] matrix_a_reuse
	s_wait_dscnt 0x1e
	v_wmma_f32_16x16x32_bf16 v[10:17], v[114:121], v[178:185], v[10:17] matrix_a_reuse
	s_wait_dscnt 0x1c
	v_wmma_f32_16x16x32_bf16 v[2:9], v[114:121], v[186:193], v[2:9] matrix_a_reuse
	s_wait_dscnt 0x1a
	v_wmma_f32_16x16x32_bf16 v[50:57], v[114:121], v[194:201], v[50:57] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(6) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[90:97], v[138:145], v[162:169], v[90:97]
	; sched_group_barrier mask(0x00000100) size(13) SyncID(0)
	s_wait_dscnt 0x14
	v_wmma_f32_16x16x32_bf16 v[82:89], v[138:145], v[202:209], v[82:89] matrix_a_reuse
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[74:81], v[138:145], v[210:217], v[74:81] matrix_a_reuse
	s_wait_dscnt 0x10
	v_wmma_f32_16x16x32_bf16 v[66:73], v[138:145], v[218:225], v[66:73] matrix_a_reuse
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[58:65], v[138:145], v[226:233], v[58:65] matrix_a_reuse
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[42:49], v[138:145], v[234:241], v[42:49] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(6) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(13) SyncID(0)
	s_wait_dscnt 0xa
	v_wmma_f32_16x16x32_bf16 v[34:41], v[138:145], v[146:153], v[34:41] matrix_a_reuse
	s_wait_dscnt 0x8
	v_wmma_f32_16x16x32_bf16 v[26:33], v[138:145], v[154:161], v[26:33] matrix_a_reuse
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[18:25], v[138:145], v[242:249], v[18:25] matrix_a_reuse
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[10:17], v[138:145], v[250:257], v[10:17] matrix_a_reuse
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[2:9], v[138:145], v[2:9] /*v[258:265]*/, v[2:9] matrix_a_reuse
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[50:57], v[138:145], v[10:17] /*v[266:273]*/, v[50:57] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(6) SyncID(0)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_wait_dscnt 0
	;;#ASMEND
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_branch .LBB0_16
.LBB0_37:
	s_wait_tensorcnt 0x0
	;;#ASMSTART
	s_wait_dscnt 0
	;;#ASMEND
	s_and_b32 vcc_lo, exec_lo, s30
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_vccz .LBB0_39
; %bb.38:
	v_lshrrev_b32_e32 v1, 1, v0
	v_cvt_pk_bf16_f32 v81, v80, v81
	v_cvt_pk_bf16_f32 v89, v88, v89
	v_cvt_pk_bf16_f32 v88, v86, v87
	v_cvt_pk_bf16_f32 v87, v84, v85
	v_bitop3_b32 v1, v110, 0x788, v1 bitop3:0xc8
	v_cvt_pk_bf16_f32 v97, v96, v97
	v_cvt_pk_bf16_f32 v96, v94, v95
	v_cvt_pk_bf16_f32 v94, v90, v91
	v_cvt_pk_bf16_f32 v86, v82, v83
	v_or_b32_e32 v80, 0x1000, v1
	v_or_b32_e32 v85, 0x1800, v1
	v_lshlrev_b32_e32 v82, 1, v1
	v_cvt_pk_bf16_f32 v49, v48, v49
	v_cvt_pk_bf16_f32 v48, v46, v47
	v_lshrrev_b32_e32 v90, 3, v80
	v_cvt_pk_bf16_f32 v80, v78, v79
	v_lshrrev_b32_e32 v78, 3, v85
	v_cvt_pk_bf16_f32 v47, v44, v45
	v_or_b32_e32 v44, 0x3800, v1
	v_lshl_or_b32 v82, v103, 5, v82
	v_and_b32_e32 v79, 0x2f0, v90
	v_or_b32_e32 v85, 0x2000, v1
	v_cvt_pk_bf16_f32 v73, v72, v73
	v_cvt_pk_bf16_f32 v72, v70, v71
	v_cvt_pk_bf16_f32 v71, v68, v69
	v_or_b32_e32 v68, 0x3000, v1
	v_or_b32_e32 v84, 0x800, v1
	v_and_b32_e32 v90, 0x3f0, v78
	v_cvt_pk_bf16_f32 v78, v74, v75
	v_or_b32_e32 v74, 0x2800, v1
	v_cvt_pk_bf16_f32 v46, v42, v43
	v_lshrrev_b32_e32 v43, 3, v44
	v_or_b32_e32 v44, 0x4000, v1
	v_cvt_pk_bf16_f32 v41, v40, v41
	v_cvt_pk_bf16_f32 v40, v38, v39
	v_cvt_pk_bf16_f32 v38, v34, v35
	v_or_b32_e32 v35, 0x4800, v1
	v_cvt_pk_bf16_f32 v33, v32, v33
	v_cvt_pk_bf16_f32 v32, v30, v31
	v_cvt_pk_bf16_f32 v31, v28, v29
	v_or_b32_e32 v28, 0x5000, v1
	v_dual_lshlrev_b32 v98, 4, v0 :: v_dual_add_nc_u32 v91, v79, v82
	v_cvt_pk_bf16_f32 v79, v76, v77
	v_lshrrev_b32_e32 v76, 3, v85
	v_cvt_pk_bf16_f32 v65, v64, v65
	v_cvt_pk_bf16_f32 v64, v62, v63
	v_cvt_pk_bf16_f32 v62, v58, v59
	v_lshrrev_b32_e32 v59, 3, v68
	v_or_b32_e32 v1, 0x5800, v1
	v_dual_lshrrev_b32 v83, 3, v84 :: v_dual_lshrrev_b32 v74, 3, v74
	v_cvt_pk_bf16_f32 v39, v36, v37
	v_lshrrev_b32_e32 v36, 3, v44
	v_cvt_pk_bf16_f32 v30, v26, v27
	v_lshrrev_b32_e32 v26, 3, v35
	v_cvt_pk_bf16_f32 v25, v24, v25
	v_cvt_pk_bf16_f32 v24, v22, v23
	v_cvt_pk_bf16_f32 v23, v20, v21
	v_lshrrev_b32_e32 v20, 3, v28
	v_and_b32_e32 v84, 0xf0, v98
	v_and_b32_e32 v76, 0x4f0, v76
	v_and_b32_e32 v42, 0x6f0, v59
	v_lshrrev_b32_e32 v1, 3, v1
	v_and_b32_e32 v83, 0x1f0, v83
	v_cvt_pk_bf16_f32 v70, v66, v67
	v_and_b32_e32 v67, 0x5f0, v74
	v_and_b32_e32 v43, 0x7f0, v43
	v_and_b32_e32 v36, 0x8f0, v36
	v_and_b32_e32 v26, 0x9f0, v26
	v_cvt_pk_bf16_f32 v22, v18, v19
	v_and_b32_e32 v19, 0xaf0, v20
	v_cvt_pk_bf16_f32 v95, v92, v93
	v_add_nc_u32_e32 v84, v84, v82
	v_add_nc_u32_e32 v66, v76, v82
	v_cvt_pk_bf16_f32 v63, v60, v61
	v_add_nc_u32_e32 v42, v42, v82
	v_and_b32_e32 v1, 0xbf0, v1
	v_dual_add_nc_u32 v83, v83, v82 :: v_dual_add_nc_u32 v75, v90, v82
	v_add_nc_u32_e32 v58, v67, v82
	v_dual_add_nc_u32 v34, v43, v82 :: v_dual_add_nc_u32 v27, v36, v82
	v_add_nc_u32_e32 v18, v26, v82
	v_cvt_pk_bf16_f32 v17, v16, v17
	v_cvt_pk_bf16_f32 v16, v14, v15
	v_cvt_pk_bf16_f32 v15, v12, v13
	v_cvt_pk_bf16_f32 v14, v10, v11
	v_add_nc_u32_e32 v10, v19, v82
	v_cvt_pk_bf16_f32 v9, v8, v9
	v_cvt_pk_bf16_f32 v8, v6, v7
	v_cvt_pk_bf16_f32 v7, v4, v5
	v_cvt_pk_bf16_f32 v6, v2, v3
	ds_store_b128 v84, v[94:97]
	ds_store_b128 v83, v[86:89] offset:4096
	ds_store_b128 v91, v[78:81] offset:8192
	ds_store_b128 v75, v[70:73] offset:12288
	ds_store_b128 v66, v[62:65] offset:16384
	ds_store_b128 v58, v[46:49] offset:20480
	v_add_nc_u32_e32 v1, v1, v82
	v_cvt_pk_bf16_f32 v5, v56, v57
	v_cvt_pk_bf16_f32 v4, v54, v55
	v_cvt_pk_bf16_f32 v3, v52, v53
	v_cvt_pk_bf16_f32 v2, v50, v51
	ds_store_b128 v42, v[38:41] offset:24576
	ds_store_b128 v34, v[30:33] offset:28672
	ds_store_b128 v27, v[22:25] offset:32768
	ds_store_b128 v18, v[14:17] offset:36864
	ds_store_b128 v10, v[6:9] offset:40960
	ds_store_b128 v1, v[2:5] offset:45056
.LBB0_39:
	v_cmp_ne_u32_e32 vcc_lo, 1, v101
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
	s_wait_kmcnt 0x0
	s_mul_i32 s14, s33, 0xc0
	s_ashr_i32 s21, s20, 31
	s_add_nc_u64 s[4:5], s[18:19], s[0:1]
	s_mov_b32 s0, 0
                                        ; implicit-def: $vgpr1
                                        ; implicit-def: $vgpr6
                                        ; implicit-def: $sgpr12_sgpr13
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_u32_e32 0x2ff, v2
	s_xor_b32 s15, exec_lo, s1
	s_cbranch_execnz .LBB0_44
; %bb.42:
	s_or_saveexec_b32 s1, s15
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
	s_abs_i32 s16, s23
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
	s_mov_b32 s6, s20
	v_mov_b32_e32 v10, v9
	s_mov_b32 s7, s21
	s_mul_f32 s0, s0, 0x4f7ffffe
	v_mov_b64_e32 v[4:5], v[2:3]
	v_mov_b64_e32 v[2:3], v[0:1]
	s_mov_b32 s8, s20
	s_cvt_u32_f32 s0, s0
	s_mov_b32 s9, s21
	s_mov_b32 s10, s20
	s_mov_b32 s11, s21
	s_mul_i32 s1, s1, s0
	s_mov_b32 s22, s23
	s_mul_hi_u32 s1, s0, s1
	s_mov_b32 s24, s23
	s_mov_b32 s25, s23
	s_ashr_i32 s26, s23, 31
	s_add_co_i32 s12, s0, s1
	s_mov_b32 s27, s13
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
	v_xor_b32_e32 v1, s26, v1
	v_xor_b32_e32 v11, s26, v11
	v_mul_lo_u32 v24, v20, s16
	v_mul_lo_u32 v26, v21, s16
	v_mul_lo_u32 v27, v22, s16
	v_dual_add_nc_u32 v25, 1, v20 :: v_dual_add_nc_u32 v29, 1, v21
	v_mul_lo_u32 v28, v23, s16
	v_dual_add_nc_u32 v30, 1, v22 :: v_dual_add_nc_u32 v31, 1, v23
	v_dual_sub_nc_u32 v6, v6, v24 :: v_dual_sub_nc_u32 v12, v12, v26
	v_dual_sub_nc_u32 v16, v16, v27 :: v_dual_bitop2_b32 v14, s26, v14 bitop3:0x14
	v_xor_b32_e32 v18, s26, v18
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
	v_mul_lo_u32 v12, v11, s22
	v_dual_sub_nc_u32 v26, v16, v14 :: v_dual_bitop2_b32 v6, v6, v1 bitop3:0x14
	v_cndmask_b32_e32 v19, v23, v27, vcc_lo
	v_add_nc_u32_e32 v20, s17, v11
	v_cmp_eq_u32_e32 vcc_lo, 0, v10
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mul_lo_u32 v14, v26, s24
	v_dual_sub_nc_u32 v1, v6, v1 :: v_dual_bitop2_b32 v19, v19, v18 bitop3:0x14
	v_dual_add_nc_u32 v22, s18, v26 :: v_dual_sub_nc_u32 v12, v3, v12
	v_ashrrev_i32_e32 v21, 31, v20
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mul_lo_u32 v6, v1, s23
	v_dual_sub_nc_u32 v27, v19, v18 :: v_dual_add_nc_u32 v18, s14, v1
	v_sub_nc_u32_e32 v14, v4, v14
	v_ashrrev_i32_e32 v23, 31, v22
	v_lshl_add_u32 v11, v11, 7, v12
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_mul_lo_u32 v16, v27, s25
	v_dual_add_nc_u32 v24, s19, v27 :: v_dual_sub_nc_u32 v6, v2, v6
	v_ashrrev_i32_e32 v19, 31, v18
	v_lshl_add_u32 v26, v26, 7, v14
	v_mul_u64_e32 v[20:21], s[6:7], v[20:21]
	v_ashrrev_i32_e32 v25, 31, v24
	v_lshl_add_u32 v1, v1, 7, v6
	v_sub_nc_u32_e32 v16, v5, v16
	v_mul_u64_e32 v[18:19], s[20:21], v[18:19]
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
	s_or_b32 s27, vcc_lo, s27
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
	s_and_not1_b32 exec_lo, exec_lo, s27
	s_cbranch_execnz .LBB0_45
; %bb.46:
	s_or_b32 exec_lo, exec_lo, s27
	v_cmp_ne_u32_e32 vcc_lo, v8, v9
	v_lshl_or_b32 v0, v9, 8, v0
	v_dual_mov_b32 v6, s16 :: v_dual_mov_b32 v1, s26
	s_and_b32 s0, vcc_lo, exec_lo
	s_or_saveexec_b32 s1, s15
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
	v_dual_sub_nc_u32 v7, v4, v9 :: v_dual_lshlrev_b32 v4, 7, v4
	v_lshlrev_b32_e32 v9, 7, v9
	v_mul_lo_u32 v8, v7, s23
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_sub_nc_u32 v4, v4, v8 :: v_dual_add_nc_u32 v8, s14, v7
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
	.size	bm128_bn192_bk128_wm8_wn1_mc0, .Lfunc_end0-bm128_bn192_bk128_wm8_wn1_mc0
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel bm128_bn192_bk128_wm8_wn1_mc0
		.amdhsa_group_segment_fixed_size 174080
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
		.amdhsa_next_free_vgpr 337
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
		.amdhsa_inst_pref_size 55
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text.bm128_bn192_bk128_wm8_wn1_mc0,"axG",@progbits,bm128_bn192_bk128_wm8_wn1_mc0,comdat
                                        ; -- End function
	.set .Lbm128_bn192_bk128_wm8_wn1_mc0.num_vgpr, 274
	.set .Lbm128_bn192_bk128_wm8_wn1_mc0.num_agpr, 0
	.set .Lbm128_bn192_bk128_wm8_wn1_mc0.numbered_sgpr, 56
	.set .Lbm128_bn192_bk128_wm8_wn1_mc0.num_named_barrier, 0
	.set .Lbm128_bn192_bk128_wm8_wn1_mc0.private_seg_size, 0
	.set .Lbm128_bn192_bk128_wm8_wn1_mc0.uses_vcc, 1
	.set .Lbm128_bn192_bk128_wm8_wn1_mc0.uses_flat_scratch, 1
	.set .Lbm128_bn192_bk128_wm8_wn1_mc0.has_dyn_sized_stack, 0
	.set .Lbm128_bn192_bk128_wm8_wn1_mc0.has_recursion, 0
	.set .Lbm128_bn192_bk128_wm8_wn1_mc0.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 6984
; TotalNumSgprs: 58
; NumVgprs: 274
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 174080 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 21
; NumSGPRsForWavesPerEU: 58
; NumVGPRsForWavesPerEU: 337
; NamedBarCnt: 0
; Occupancy: 2
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
	.type	__hip_cuid_a63b5b987ab371e3,@object ; @__hip_cuid_a63b5b987ab371e3
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_a63b5b987ab371e3
__hip_cuid_a63b5b987ab371e3:
	.byte	0                               ; 0x0
	.size	__hip_cuid_a63b5b987ab371e3, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git aa451e1fe6a793394d6733051b1778633063ae96+PATCHED:d17c5aa0e3ea29cde402f58f27e39b6034effa27)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_a63b5b987ab371e3
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
    macrotile: [128, 192, 128]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, One]
  MatrixInstruction: [16, 16, 32, 1]
  EnableMatrixInstruction: True
  MIWaveTile: [4, 3]
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
    .group_segment_fixed_size: 174080
    .kernarg_segment_align: 8
    .kernarg_segment_size: 132
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           bm128_bn192_bk128_wm8_wn1_mc0
    .private_segment_fixed_size: 0
    .sgpr_count:     58
    .sgpr_spill_count: 0
    .symbol:         bm128_bn192_bk128_wm8_wn1_mc0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     274
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
