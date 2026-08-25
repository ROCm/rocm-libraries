	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.section	.text.bm256_bn128_bk064_wm2_wn4_mc0,"axG",@progbits,bm256_bn128_bk064_wm2_wn4_mc0,comdat
	.protected	bm256_bn128_bk064_wm2_wn4_mc0 ; -- Begin function bm256_bn128_bk064_wm2_wn4_mc0
	.globl	bm256_bn128_bk064_wm2_wn4_mc0
	.p2align	8
	.type	bm256_bn128_bk064_wm2_wn4_mc0,@function
bm256_bn128_bk064_wm2_wn4_mc0: ; @bm256_bn128_bk064_wm2_wn4_mc0
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
	s_cselect_b32 s3, ttmp7, s5
	s_wait_kmcnt 0x0
	s_add_co_i32 s4, s20, 0xff
	s_add_co_i32 s6, s21, 0x7f
	s_ashr_i32 s5, s4, 31
	s_lshl_b32 s24, s35, 8
	s_lshr_b32 s5, s5, 24
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s4, s4, s5
	s_ashr_i32 s5, s6, 31
	s_ashr_i32 s4, s4, 8
	s_lshr_b32 s5, s5, 25
	s_add_co_i32 s6, s6, s5
	s_sub_co_i32 s5, s20, s24
	s_ashr_i32 s6, s6, 7
	s_min_i32 s23, s5, 0x100
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
	s_add_co_i32 s13, s22, 63
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 2, 1), 1
	s_min_i32 s7, s22, 64
	s_cmp_gt_i32 s13, 63
	s_cselect_b32 s12, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_and_b32 s8, s12, exec_lo
	s_cselect_b32 s26, s7, 0
	s_cmp_lt_i32 s27, 0x100
	s_cselect_b32 s38, -1, 0
	s_and_b32 vcc_lo, exec_lo, s38
	s_mov_b32 s7, s38
	s_cbranch_vccnz .LBB0_2
; %bb.1:
	s_cmp_lt_i32 s29, 0x80
	s_cselect_b32 s7, -1, 0
	s_cmp_lt_i32 s26, 64
	s_cselect_b32 s8, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 s7, s8, s7
.LBB0_2:
	v_lshlrev_b32_e32 v132, 2, v0
	s_and_not1_b32 vcc_lo, exec_lo, s7
	s_cbranch_vccnz .LBB0_8
; %bb.3:
	v_or_b32_e32 v1, 0xffffff00, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v3, v132
	s_mov_b32 s7, 0
	v_mov_b32_e32 v4, v1
.LBB0_4:                                ; =>This Inner Loop Header: Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)
	v_add_nc_u32_e32 v4, 0x100, v4
	ds_store_b32 v3, v2
	v_add_nc_u32_e32 v3, 0x400, v3
	v_cmp_lt_u32_e32 vcc_lo, 0x20ff, v4
	s_or_b32 s7, vcc_lo, s7
	s_and_not1_b32 exec_lo, exec_lo, s7
	s_cbranch_execnz .LBB0_4
; %bb.5:
	s_or_b32 exec_lo, exec_lo, s7
	v_lshl_add_u32 v2, s2, 2, v132
	v_mov_b32_e32 v3, 0
	s_mov_b32 s7, 0
.LBB0_6:                                ; =>This Inner Loop Header: Depth=1
	v_add_nc_u32_e32 v1, 0x100, v1
	ds_store_b32 v2, v3 offset:34816
	v_add_nc_u32_e32 v2, 0x400, v2
	v_cmp_lt_u32_e32 vcc_lo, 0xfff, v1
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
	v_lshrrev_b32_e32 v137, 5, v0
	s_lshl_b32 s34, s2, 2
	s_add_co_i32 s6, s6, -1
	s_mov_b64 s[30:31], src_shared_base
	s_or_b32 s39, s34, 0x8800
	s_add_co_i32 s30, s4, -1
	s_min_i32 s36, s3, s6
	s_wait_xcnt 0x0
	s_mov_b32 s0, exec_lo
	v_cmpx_lt_i32_e32 0, v137
	s_xor_b32 s37, exec_lo, s0
	s_cbranch_execz .LBB0_12
; %bb.9:
	s_mov_b32 s40, exec_lo
	v_cmpx_eq_u32_e32 1, v137
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
	v_dual_mov_b32 v1, s39 :: v_dual_mov_b32 v4, s2
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
	s_or_b32 s3, s0, 0x400000
	s_movk_i32 s4, 0x80
	s_mov_b32 s0, 0x7510000
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[44:47], s[0:7]
.LBB0_11:
	s_or_b32 exec_lo, exec_lo, s40
.LBB0_12:
	s_or_saveexec_b32 s37, s37
	s_min_i32 s28, s35, s30
	s_xor_b32 exec_lo, exec_lo, s37
	s_cbranch_execz .LBB0_14
; %bb.13:
	s_cmp_gt_i32 s26, 0
	s_mov_b32 s6, 0
	s_cselect_b32 s4, -1, 0
	s_lshl_b32 s0, s28, 8
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
	s_or_b32 s3, s0, 0x400000
	s_movk_i32 s4, 0x100
	v_readfirstlane_b32 s42, v4
	v_readfirstlane_b32 s43, v3
	s_mov_b32 s0, 0x7510000
	s_mov_b32 s7, s6
	s_mov_b32 s41, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[40:43], s[0:7]
.LBB0_14:
	s_or_b32 exec_lo, exec_lo, s37
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	v_mov_b32_e32 v9, 0
	s_and_b32 s30, s25, s21
	v_and_b32_e32 v135, 0x80, v0
	v_cndmask_b32_e64 v131, 0, 1, s30
	s_and_not1_b32 vcc_lo, exec_lo, s12
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
	v_dual_mov_b32 v50, v9 :: v_dual_mov_b32 v73, v9
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
	v_dual_mov_b32 v90, v9 :: v_dual_mov_b32 v105, v9
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
	v_dual_mov_b32 v122, v9 :: v_dual_mov_b32 v65, v9
	v_dual_mov_b32 v64, v9 :: v_dual_mov_b32 v63, v9
	v_dual_mov_b32 v62, v9 :: v_dual_mov_b32 v61, v9
	v_dual_mov_b32 v60, v9 :: v_dual_mov_b32 v59, v9
	v_mov_b32_e32 v58, v9
	s_movk_i32 s12, 0x80
	s_barrier_wait -1
	s_cbranch_vccnz .LBB0_37
; %bb.15:
	v_dual_lshlrev_b32 v1, 6, v0 :: v_dual_lshlrev_b32 v2, 6, v135
	v_dual_mov_b32 v133, 0 :: v_dual_bitop2_b32 v3, 16, v0 bitop3:0x40
	s_mov_b64 s[0:1], src_shared_base
	s_or_b32 s2, s34, 0xcc00
	s_mov_b32 s3, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_and_or_b32 v3, 0x3c0, v1, v3
	v_and_b32_e32 v1, 0x1800, v1
	s_and_b64 s[2:3], s[2:3], 15
	s_mov_b32 s7, 0
	s_sub_co_i32 s0, 16, s2
	v_or_b32_e32 v4, v3, v2
	s_lshr_b32 s0, s0, 2
	s_cmp_lg_u64 s[2:3], 0
	s_mov_b32 s41, s1
	s_cselect_b32 s0, s0, 0
	v_lshrrev_b32_e32 v5, 4, v4
	s_lshl2_add_u32 s2, s0, s34
	v_or_b32_e32 v172, 0x8800, v132
	s_add_co_i32 s0, s2, 0x15400
	s_add_co_i32 s42, s2, 0xcc00
	v_and_b32_e32 v5, 0x238, v5
	s_and_b32 s6, s0, 15
	v_or_b32_e32 v174, 0x4000, v132
	s_sub_co_i32 s3, 16, s6
	v_dual_mov_b32 v24, v133 :: v_dual_mov_b32 v25, v133
	v_add_nc_u32_e32 v130, v5, v4
	v_or_b32_e32 v5, 0x800, v4
	s_lshr_b32 s2, s3, 2
	s_cmp_lg_u64 s[6:7], 0
	v_dual_mov_b32 v26, v133 :: v_dual_mov_b32 v27, v133
	s_delay_alu instid0(VALU_DEP_2)
	v_lshrrev_b32_e32 v5, 4, v5
	v_or_b32_e32 v11, 0x1800, v4
	v_or_b32_e32 v10, v3, v1
	v_or_b32_e32 v12, 0x1c00, v4
	v_or_b32_e32 v14, 32, v3
	v_or_b32_e32 v6, 0x400, v4
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_lshrrev_b32 v11, 4, v11 :: v_dual_lshrrev_b32 v13, 4, v10
	v_dual_lshrrev_b32 v12, 4, v12 :: v_dual_bitop2_b32 v16, v14, v2 bitop3:0x54
	v_or_b32_e32 v8, 0x1000, v4
	v_or_b32_e32 v15, 0x400, v10
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v13, 0x1b8, v13
	v_or_b32_e32 v7, 0xc00, v4
	v_or_b32_e32 v20, 0x1000, v16
	v_dual_lshrrev_b32 v6, 4, v6 :: v_dual_bitop2_b32 v1, v14, v1 bitop3:0x54
	v_or_b32_e32 v9, 0x1400, v4
	v_lshrrev_b32_e32 v8, 4, v8
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_lshrrev_b32 v20, 4, v20 :: v_dual_lshrrev_b32 v15, 4, v15
	v_add_nc_u32_e32 v134, v13, v10
	v_or_b32_e32 v13, 0x400, v16
	v_or_b32_e32 v17, 0x800, v16
	v_or_b32_e32 v18, 0xc00, v16
	v_lshrrev_b32_e32 v19, 4, v16
	v_or_b32_e32 v14, 0x1400, v16
	v_or_b32_e32 v21, 0x1800, v16
	v_or_b32_e32 v16, 0x1c00, v16
	v_or_b32_e32 v22, 0x400, v1
	s_cselect_b32 s2, s2, 0
	s_ashr_i32 s3, s13, 31
	v_lshrrev_b32_e32 v7, 4, v7
	s_lshr_b32 s3, s3, 26
	v_and_b32_e32 v6, 0x278, v6
	s_add_co_i32 s13, s13, s3
	v_lshrrev_b32_e32 v9, 4, v9
	v_and_b32_e32 v8, 0x338, v8
	v_lshrrev_b32_e32 v13, 4, v13
	v_dual_lshrrev_b32 v17, 4, v17 :: v_dual_lshrrev_b32 v18, 4, v18
	v_dual_lshrrev_b32 v14, 4, v14 :: v_dual_lshrrev_b32 v21, 4, v21
	v_dual_lshrrev_b32 v16, 4, v16 :: v_dual_lshrrev_b32 v1, 4, v1
	v_dual_lshrrev_b32 v22, 4, v22 :: v_dual_add_nc_u32 v136, v6, v4
	s_lshl_b32 s6, s2, 2
	s_ashr_i32 s43, s13, 6
	s_cmp_lt_i32 s29, 0x80
	s_add_nc_u64 s[34:35], s[0:1], s[6:7]
	s_cselect_b32 s44, -1, 0
	s_lshl_b32 s0, s36, 7
	s_lshl_b32 s2, s28, 8
	v_and_b32_e32 v5, 0x2b8, v5
	v_and_b32_e32 v7, 0x2f8, v7
	v_and_b32_e32 v9, 0x378, v9
	v_and_b32_e32 v11, 0x3b8, v11
	v_and_b32_e32 v12, 0x3f8, v12
	v_and_b32_e32 v15, 0x1f8, v15
	v_and_b32_e32 v19, 0x238, v19
	v_and_b32_e32 v13, 0x3f8, v13
	v_and_b32_e32 v17, 0x3f8, v17
	v_and_b32_e32 v18, 0x3f8, v18
	v_and_b32_e32 v20, 0x3f8, v20
	v_and_b32_e32 v14, 0x3f8, v14
	v_and_b32_e32 v21, 0x3f8, v21
	v_and_b32_e32 v16, 0x3f8, v16
	v_and_b32_e32 v23, 0x1b8, v1
	v_and_b32_e32 v22, 0x1f8, v22
	v_dual_add_nc_u32 v142, v8, v4 :: v_dual_add_nc_u32 v2, v3, v2
	s_ashr_i32 s1, s0, 31
	s_ashr_i32 s3, s2, 31
	s_wait_kmcnt 0x0
	s_bfe_i64 s[16:17], s[16:17], 0x200000
	s_bfe_i64 s[8:9], s[8:9], 0x200000
	s_mul_u64 s[0:1], s[16:17], s[0:1]
	s_mul_u64 s[2:3], s[8:9], s[2:3]
	v_or_b32_e32 v1, 0x100, v0
	v_dual_add_nc_u32 v138, v5, v4 :: v_dual_add_nc_u32 v140, v7, v4
	v_dual_add_nc_u32 v144, v9, v4 :: v_dual_add_nc_u32 v146, v11, v4
	v_dual_add_nc_u32 v148, v12, v4 :: v_dual_add_nc_u32 v150, v15, v10
	v_dual_mov_b32 v173, v133 :: v_dual_add_nc_u32 v152, v19, v2
	v_add3_u32 v154, v2, v13, 0x400
	v_add3_u32 v156, v2, v17, 0x800
	v_add3_u32 v158, v2, v18, 0xc00
	v_add3_u32 v160, v2, v20, 0x1000
	v_add3_u32 v162, v2, v14, 0x1400
	v_add3_u32 v164, v2, v21, 0x1800
	v_add3_u32 v166, v2, v16, 0x1c00
	v_dual_mov_b32 v175, v133 :: v_dual_add_nc_u32 v168, v23, v10
	v_add3_u32 v170, v10, v22, 0x400
	v_dual_mov_b32 v2, v133 :: v_dual_mov_b32 v3, v133
	v_dual_mov_b32 v4, v133 :: v_dual_mov_b32 v5, v133
	v_dual_mov_b32 v6, v133 :: v_dual_mov_b32 v7, v133
	v_dual_mov_b32 v8, v133 :: v_dual_mov_b32 v9, v133
	v_dual_mov_b32 v10, v133 :: v_dual_mov_b32 v11, v133
	v_dual_mov_b32 v12, v133 :: v_dual_mov_b32 v13, v133
	v_dual_mov_b32 v14, v133 :: v_dual_mov_b32 v15, v133
	v_dual_mov_b32 v16, v133 :: v_dual_mov_b32 v17, v133
	v_dual_mov_b32 v18, v133 :: v_dual_mov_b32 v19, v133
	v_dual_mov_b32 v20, v133 :: v_dual_mov_b32 v21, v133
	v_dual_mov_b32 v22, v133 :: v_dual_mov_b32 v23, v133
	v_dual_mov_b32 v28, v133 :: v_dual_mov_b32 v29, v133
	v_dual_mov_b32 v30, v133 :: v_dual_mov_b32 v31, v133
	v_dual_mov_b32 v32, v133 :: v_dual_mov_b32 v33, v133
	v_dual_mov_b32 v34, v133 :: v_dual_mov_b32 v35, v133
	v_dual_mov_b32 v36, v133 :: v_dual_mov_b32 v37, v133
	v_dual_mov_b32 v38, v133 :: v_dual_mov_b32 v39, v133
	v_dual_mov_b32 v40, v133 :: v_dual_mov_b32 v41, v133
	v_dual_mov_b32 v42, v133 :: v_dual_mov_b32 v43, v133
	v_dual_mov_b32 v44, v133 :: v_dual_mov_b32 v45, v133
	v_dual_mov_b32 v46, v133 :: v_dual_mov_b32 v47, v133
	v_dual_mov_b32 v48, v133 :: v_dual_mov_b32 v49, v133
	v_dual_mov_b32 v50, v133 :: v_dual_mov_b32 v51, v133
	v_dual_mov_b32 v52, v133 :: v_dual_mov_b32 v53, v133
	v_dual_mov_b32 v54, v133 :: v_dual_mov_b32 v55, v133
	v_dual_mov_b32 v56, v133 :: v_dual_mov_b32 v57, v133
	v_dual_mov_b32 v66, v133 :: v_dual_mov_b32 v67, v133
	v_dual_mov_b32 v68, v133 :: v_dual_mov_b32 v69, v133
	v_dual_mov_b32 v70, v133 :: v_dual_mov_b32 v71, v133
	v_dual_mov_b32 v72, v133 :: v_dual_mov_b32 v73, v133
	v_dual_mov_b32 v74, v133 :: v_dual_mov_b32 v75, v133
	v_dual_mov_b32 v76, v133 :: v_dual_mov_b32 v77, v133
	v_dual_mov_b32 v78, v133 :: v_dual_mov_b32 v79, v133
	v_dual_mov_b32 v80, v133 :: v_dual_mov_b32 v81, v133
	v_dual_mov_b32 v82, v133 :: v_dual_mov_b32 v83, v133
	v_dual_mov_b32 v84, v133 :: v_dual_mov_b32 v85, v133
	v_dual_mov_b32 v86, v133 :: v_dual_mov_b32 v87, v133
	v_dual_mov_b32 v88, v133 :: v_dual_mov_b32 v89, v133
	v_dual_mov_b32 v90, v133 :: v_dual_mov_b32 v91, v133
	v_dual_mov_b32 v92, v133 :: v_dual_mov_b32 v93, v133
	v_dual_mov_b32 v94, v133 :: v_dual_mov_b32 v95, v133
	v_dual_mov_b32 v96, v133 :: v_dual_mov_b32 v97, v133
	v_dual_mov_b32 v98, v133 :: v_dual_mov_b32 v99, v133
	v_dual_mov_b32 v100, v133 :: v_dual_mov_b32 v101, v133
	v_dual_mov_b32 v102, v133 :: v_dual_mov_b32 v103, v133
	v_dual_mov_b32 v104, v133 :: v_dual_mov_b32 v105, v133
	v_dual_mov_b32 v106, v133 :: v_dual_mov_b32 v107, v133
	v_dual_mov_b32 v108, v133 :: v_dual_mov_b32 v109, v133
	v_dual_mov_b32 v110, v133 :: v_dual_mov_b32 v111, v133
	v_dual_mov_b32 v112, v133 :: v_dual_mov_b32 v113, v133
	v_dual_mov_b32 v114, v133 :: v_dual_mov_b32 v115, v133
	v_dual_mov_b32 v116, v133 :: v_dual_mov_b32 v117, v133
	v_dual_mov_b32 v118, v133 :: v_dual_mov_b32 v119, v133
	v_dual_mov_b32 v120, v133 :: v_dual_mov_b32 v121, v133
	v_dual_mov_b32 v122, v133 :: v_dual_mov_b32 v123, v133
	v_dual_mov_b32 v124, v133 :: v_dual_mov_b32 v125, v133
	v_dual_mov_b32 v126, v133 :: v_dual_mov_b32 v127, v133
	v_dual_mov_b32 v128, v133 :: v_dual_mov_b32 v129, v133
	v_dual_mov_b32 v58, v133 :: v_dual_mov_b32 v59, v133
	v_dual_mov_b32 v60, v133 :: v_dual_mov_b32 v61, v133
	v_dual_mov_b32 v62, v133 :: v_dual_mov_b32 v63, v133
	v_dual_mov_b32 v64, v133 :: v_dual_mov_b32 v65, v133
	v_or_b32_e32 v139, 0x2100, v0
	v_or_b32_e32 v141, 0xf00, v0
	s_lshr_b32 s45, s29, 16
	s_lshr_b32 s46, s27, 16
	s_lshl_b64 s[0:1], s[0:1], 1
	s_lshl_b64 s[2:3], s[2:3], 1
	s_mov_b32 s40, s31
	s_bitset1_b32 s45, 22
	s_bitset1_b32 s46, 22
	s_add_nc_u64 s[16:17], s[10:11], s[0:1]
	s_add_nc_u64 s[36:37], s[14:15], s[2:3]
	s_mov_b32 s47, -1
	s_movk_i32 s4, 0x100
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
	s_lshl_b32 s1, s48, 6
	s_sub_co_i32 s2, s22, s1
	s_xor_b32 s1, s49, 1
	s_min_i32 s2, s2, 64
	s_cmp_lt_i32 s48, s43
	s_cselect_b32 s6, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_and_b32 s3, s6, exec_lo
	s_cselect_b32 s26, s2, 0
	s_cmp_lt_i32 s26, 64
	s_cselect_b32 s2, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b32 s2, s44, s2
	s_or_b32 s2, s38, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB0_29
; %bb.18:                               ;   in Loop: Header=BB0_17 Depth=1
	v_nop
	v_nop
	v_nop
	v_mov_b64_e32 v[176:177], v[0:1]
	v_mov_b32_e32 v143, 34
	s_cmp_lg_u32 s1, 0
	s_mov_b32 s8, 0
	s_cselect_b32 s3, s41, s31
	s_cselect_b32 s2, s42, 0
.LBB0_19:                               ;   Parent Loop BB0_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_dual_mov_b32 v132, v176 :: v_dual_add_nc_u32 v143, -2, v143
	v_dual_mov_b32 v178, v177 :: v_dual_mov_b32 v179, v133
	v_add_nc_u32_e32 v177, 0x200, v177
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u64 v[180:181], v[132:133], 2, s[2:3]
	v_cmp_eq_u32_e32 vcc_lo, 0, v143
	v_add_nc_u32_e32 v176, 0x200, v176
	v_lshl_add_u64 v[178:179], v[178:179], 2, s[2:3]
	s_clause 0x1
	flat_store_b32 v[180:181], v133
	flat_store_b32 v[178:179], v133
	s_or_b32 s8, vcc_lo, s8
	s_wait_xcnt 0x0
	s_and_not1_b32 exec_lo, exec_lo, s8
	s_cbranch_execnz .LBB0_19
; %bb.20:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s8
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_23
; %bb.21:                               ;   in Loop: Header=BB0_17 Depth=1
	v_add_nc_u64_e32 v[176:177], s[2:3], v[172:173]
	v_mov_b32_e32 v132, v139
	s_mov_b32 s2, 0
.LBB0_22:                               ;   Parent Loop BB0_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v132, 0x100, v132
	flat_store_b32 v[176:177], v133
	s_wait_xcnt 0x0
	v_add_nc_u64_e32 v[176:177], 0x400, v[176:177]
	v_cmp_lt_u32_e32 vcc_lo, 0x20ff, v132
	s_or_b32 s2, vcc_lo, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s2
	s_cbranch_execnz .LBB0_22
.LBB0_23:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s8
	v_mov_b64_e32 v[176:177], v[0:1]
	v_mov_b32_e32 v143, 16
	s_cmp_lg_u32 s1, 0
	s_mov_b32 s8, 0
	s_cselect_b32 s3, s35, s40
	s_cselect_b32 s2, s34, s39
.LBB0_24:                               ;   Parent Loop BB0_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_dual_mov_b32 v132, v176 :: v_dual_add_nc_u32 v143, -2, v143
	v_dual_mov_b32 v178, v177 :: v_dual_mov_b32 v179, v133
	v_add_nc_u32_e32 v177, 0x200, v177
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u64 v[180:181], v[132:133], 2, s[2:3]
	v_cmp_eq_u32_e32 vcc_lo, 0, v143
	v_add_nc_u32_e32 v176, 0x200, v176
	v_lshl_add_u64 v[178:179], v[178:179], 2, s[2:3]
	s_clause 0x1
	flat_store_b32 v[180:181], v133
	flat_store_b32 v[178:179], v133
	s_or_b32 s8, vcc_lo, s8
	s_wait_xcnt 0x0
	s_and_not1_b32 exec_lo, exec_lo, s8
	s_cbranch_execnz .LBB0_24
; %bb.25:                               ;   in Loop: Header=BB0_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s8
	s_and_saveexec_b32 s8, s47
	s_cbranch_execz .LBB0_28
; %bb.26:                               ;   in Loop: Header=BB0_17 Depth=1
	v_add_nc_u64_e32 v[176:177], s[2:3], v[174:175]
	v_mov_b32_e32 v132, v141
	s_mov_b32 s2, 0
.LBB0_27:                               ;   Parent Loop BB0_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v132, 0x100, v132
	flat_store_b32 v[176:177], v133
	s_wait_xcnt 0x0
	v_add_nc_u64_e32 v[176:177], 0x400, v[176:177]
	v_cmp_lt_u32_e32 vcc_lo, 0xfff, v132
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
	v_cmpx_lt_i32_e32 0, v137
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
	v_cmpx_eq_u32_e32 1, v137
	s_cbranch_execz .LBB0_34
; %bb.33:                               ;   in Loop: Header=BB0_17 Depth=1
	s_cmp_lg_u32 s1, 0
	s_mov_b32 s28, s26
	s_cselect_b32 s10, s34, s39
	s_cmp_gt_i32 s26, 0
	s_mov_b32 s13, s5
	s_cselect_b32 s11, -1, 0
	s_lshl_b32 s6, s2, 6
	s_mov_b32 s14, s7
	s_lshl_b64 s[8:9], s[6:7], 1
	s_mov_b32 s15, s7
	s_add_nc_u64 s[8:9], s[16:17], s[8:9]
	v_nop
	v_dual_mov_b32 v143, s10 :: v_dual_mov_b32 v176, s8
	s_and_b32 s6, s9, 0x1ffffff
	s_and_b32 s9, s21, s11
	s_bitset1_b32 s6, 31
	v_cndmask_b32_e64 v132, 0, 1, s9
	v_mov_b32_e32 v145, s6
	v_readfirstlane_b32 s53, v143
	v_readfirstlane_b32 s54, v176
	s_lshr_b64 s[10:11], s[28:29], 16
	v_readfirstlane_b32 s52, v132
	v_readfirstlane_b32 s55, v145
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
	s_cselect_b32 s1, s42, 0
	s_cmp_gt_i32 s26, 0
	s_cselect_b32 s9, -1, 0
	s_lshl_b32 s6, s2, 6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_lshl_b64 s[2:3], s[6:7], 1
	s_and_b32 s6, s25, s9
	s_add_nc_u64 s[2:3], s[36:37], s[2:3]
	v_cndmask_b32_e64 v132, 0, 1, s6
	s_and_b32 s3, s3, 0x1ffffff
	v_nop
	v_dual_mov_b32 v143, s1 :: v_dual_mov_b32 v176, s2
	s_bitset1_b32 s3, 31
	v_readfirstlane_b32 s52, v132
	v_mov_b32_e32 v145, s3
	s_delay_alu instid0(VALU_DEP_3)
	v_readfirstlane_b32 s53, v143
	v_readfirstlane_b32 s54, v176
	s_lshr_b64 s[2:3], s[26:27], 16
	s_lshl_b32 s1, s26, 16
	v_readfirstlane_b32 s55, v145
	s_mov_b32 s3, s46
	s_mov_b32 s6, s7
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	tensor_load_to_lds s[52:55], s[0:7]
	s_or_b32 exec_lo, exec_lo, s8
	s_and_not1_b32 vcc_lo, exec_lo, s30
	s_cbranch_vccnz .LBB0_16
.LBB0_36:                               ;   in Loop: Header=BB0_17 Depth=1
	s_cmp_lg_u32 s49, 0
	s_cselect_b32 s1, s42, 0
	s_cselect_b32 s2, s34, s39
	v_lshl_add_u32 v132, v130, 1, s1
	v_lshl_add_u32 v143, v136, 1, s1
	v_lshl_add_u32 v145, v138, 1, s1
	v_lshl_add_u32 v147, v134, 1, s2
	v_lshl_add_u32 v149, v160, 1, s1
	ds_load_b128 v[176:179], v132
	ds_load_b128 v[180:183], v132 offset:16
	ds_load_b128 v[184:187], v143 offset:2048
	ds_load_b128 v[188:191], v143 offset:2064
	ds_load_b128 v[192:195], v145 offset:4096
	ds_load_b128 v[200:203], v147
	ds_load_b128 v[204:207], v147 offset:16
	v_lshl_add_u32 v132, v140, 1, s1
	v_lshl_add_u32 v143, v152, 1, s1
	v_lshl_add_u32 v147, v158, 1, s1
	v_lshl_add_u32 v151, v162, 1, s1
	v_lshl_add_u32 v153, v164, 1, s1
	ds_load_b128 v[208:211], v132 offset:6144
	ds_load_b128 v[212:215], v132 offset:6160
	v_lshl_add_u32 v132, v150, 1, s2
	v_lshl_add_u32 v155, v166, 1, s1
	v_lshl_add_u32 v157, v168, 1, s2
	v_lshl_add_u32 v159, v170, 1, s2
	ds_load_b128 v[224:227], v147 offset:64
	ds_load_b128 v[228:231], v147 offset:80
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[122:129], v[176:183], v[200:207], v[122:129]
	ds_load_b128 v[216:219], v132 offset:2048
	ds_load_b128 v[220:223], v132 offset:2064
	v_lshl_add_u32 v132, v142, 1, s1
	ds_load_b128 v[232:235], v149 offset:64
	ds_load_b128 v[236:239], v149 offset:80
	ds_load_b128 v[240:243], v153 offset:64
	ds_load_b128 v[244:247], v153 offset:80
	ds_load_b128 v[248:251], v155 offset:64
	ds_load_b128 v[252:255], v155 offset:80
	; sched_group_barrier mask(0x00000100) size(5) SyncID(0)
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[114:121], v[176:183], v[216:223], v[114:121] matrix_b_reuse
	ds_load_b128 v[176:179], v132 offset:8192
	ds_load_b128 v[180:183], v132 offset:8208
	v_lshl_add_u32 v132, v148, 1, s1
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[50:57], v[176:183], v[200:207], v[50:57] matrix_b_reuse
	ds_load_b128 v[196:199], v145 offset:4112
	v_lshl_add_u32 v145, v154, 1, s1
	v_wmma_f32_16x16x32_bf16 v[106:113], v[184:191], v[200:207], v[106:113] matrix_b_reuse
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[90:97], v[192:199], v[200:207], v[90:97] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[98:105], v[184:191], v[216:223], v[98:105] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[82:89], v[192:199], v[216:223], v[82:89] matrix_b_reuse
	ds_load_b128 v[184:187], v132 offset:14336
	ds_load_b128 v[188:191], v132 offset:14352
	v_lshl_add_u32 v132, v146, 1, s1
	ds_load_b128 v[196:199], v132 offset:12304
	ds_load_b128 v[192:195], v132 offset:12288
	v_lshl_add_u32 v132, v144, 1, s1
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[10:17], v[192:199], v[216:223], v[10:17] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[58:65], v[184:191], v[216:223], v[58:65] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[2:9], v[184:191], v[200:207], v[2:9] matrix_b_reuse
	ds_load_b128 v[184:187], v132 offset:10240
	ds_load_b128 v[188:191], v132 offset:10256
	v_lshl_add_u32 v132, v156, 1, s1
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[26:33], v[184:191], v[216:223], v[26:33] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[34:41], v[184:191], v[200:207], v[34:41] matrix_b_reuse
	ds_load_b128 v[184:187], v157 offset:64
	ds_load_b128 v[188:191], v157 offset:80
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(5) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[74:81], v[208:215], v[200:207], v[74:81] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(5) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[66:73], v[208:215], v[216:223], v[66:73] matrix_b_reuse
	ds_load_b128 v[208:211], v132 offset:64
	ds_load_b128 v[212:215], v132 offset:80
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(5) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[42:49], v[176:183], v[216:223], v[42:49] matrix_b_reuse
	ds_load_b128 v[176:179], v145 offset:64
	ds_load_b128 v[180:183], v145 offset:80
	ds_load_b128 v[216:219], v151 offset:64
	ds_load_b128 v[220:223], v151 offset:80
	v_wmma_f32_16x16x32_bf16 v[18:25], v[192:199], v[200:207], v[18:25] matrix_b_reuse
	ds_load_b128 v[192:195], v143 offset:64
	ds_load_b128 v[196:199], v143 offset:80
	ds_load_b128 v[200:203], v159 offset:64
	ds_load_b128 v[204:207], v159 offset:80
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[122:129], v[192:199], v[184:191], v[122:129]
	; sched_group_barrier mask(0x00000100) size(5) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[106:113], v[176:183], v[184:191], v[106:113] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[90:97], v[208:215], v[184:191], v[90:97] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[74:81], v[224:231], v[184:191], v[74:81] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(5) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[50:57], v[232:239], v[184:191], v[50:57] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[34:41], v[216:223], v[184:191], v[34:41] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[18:25], v[240:247], v[184:191], v[18:25] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[2:9], v[248:255], v[184:191], v[2:9] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(5) SyncID(0)
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[58:65], v[248:255], v[200:207], v[58:65] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[10:17], v[240:247], v[200:207], v[10:17] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[26:33], v[216:223], v[200:207], v[26:33] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[42:49], v[232:239], v[200:207], v[42:49] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(5) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[66:73], v[224:231], v[200:207], v[66:73] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[82:89], v[208:215], v[200:207], v[82:89] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[98:105], v[176:183], v[200:207], v[98:105] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[114:121], v[192:199], v[200:207], v[114:121] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
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
	s_and_b32 vcc_lo, exec_lo, s30
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_vccz .LBB0_39
; %bb.38:
	v_dual_lshrrev_b32 v1, 1, v0 :: v_dual_lshlrev_b32 v130, 8, v0
	v_cvt_pk_bf16_f32 v113, v112, v113
	v_cvt_pk_bf16_f32 v112, v110, v111
	v_cvt_pk_bf16_f32 v111, v108, v109
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v1, 8, v1
	v_and_b32_e32 v130, 0x6f00, v130
	v_cvt_pk_bf16_f32 v121, v120, v121
	v_cvt_pk_bf16_f32 v120, v118, v119
	v_cvt_pk_bf16_f32 v118, v114, v115
	v_cvt_pk_bf16_f32 v49, v48, v49
	v_or3_b32 v1, v1, v135, v130
	v_cvt_pk_bf16_f32 v48, v46, v47
	v_cvt_pk_bf16_f32 v47, v44, v45
	v_cvt_pk_bf16_f32 v129, v128, v129
	v_cvt_pk_bf16_f32 v128, v126, v127
	v_or_b32_e32 v110, 0x1000, v1
	v_dual_lshrrev_b32 v108, 3, v1 :: v_dual_lshlrev_b32 v114, 1, v1
	v_add_nc_u32_e32 v44, 0x1050, v1
	v_cvt_pk_bf16_f32 v127, v124, v125
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshrrev_b32_e32 v115, 3, v110
	v_and_b32_e32 v108, 0xdfe, v108
	v_cvt_pk_bf16_f32 v126, v122, v123
	v_cvt_pk_bf16_f32 v119, v116, v117
	v_lshlrev_b32_e32 v122, 1, v110
	v_and_b32_e32 v115, 0xffe, v115
	v_dual_add_nc_u32 v108, v108, v114 :: v_dual_add_nc_u32 v109, 16, v1
	v_cvt_pk_bf16_f32 v105, v104, v105
	v_cvt_pk_bf16_f32 v104, v102, v103
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v110, v115, v114
	v_cvt_pk_bf16_f32 v103, v100, v101
	v_add_nc_u32_e32 v100, 48, v1
	v_cvt_pk_bf16_f32 v89, v88, v89
	v_cvt_pk_bf16_f32 v88, v86, v87
	v_cvt_pk_bf16_f32 v87, v84, v85
	v_add_nc_u32_e32 v84, 64, v1
	v_cvt_pk_bf16_f32 v73, v72, v73
	v_cvt_pk_bf16_f32 v72, v70, v71
	v_cvt_pk_bf16_f32 v71, v68, v69
	v_add_nc_u32_e32 v68, 0x50, v1
	v_add_nc_u32_e32 v116, 0x1010, v1
	v_cvt_pk_bf16_f32 v86, v82, v83
	v_add_nc_u32_e32 v83, 0x1030, v1
	v_cvt_pk_bf16_f32 v81, v80, v81
	v_cvt_pk_bf16_f32 v80, v78, v79
	v_cvt_pk_bf16_f32 v78, v74, v75
	v_add_nc_u32_e32 v74, 0x1040, v1
	v_add_nc_u32_e32 v117, 32, v1
	v_cvt_pk_bf16_f32 v46, v42, v43
	v_lshrrev_b32_e32 v43, 3, v44
	v_add_nc_u32_e32 v44, 0x60, v1
	ds_store_b128 v108, v[126:129]
	ds_store_b128 v110, v[118:121] offset:8192
	v_cvt_pk_bf16_f32 v110, v106, v107
	v_add_nc_u32_e32 v106, 0x1020, v1
	v_cvt_pk_bf16_f32 v41, v40, v41
	v_cvt_pk_bf16_f32 v40, v38, v39
	v_cvt_pk_bf16_f32 v38, v34, v35
	v_add_nc_u32_e32 v35, 0x1060, v1
	v_cvt_pk_bf16_f32 v33, v32, v33
	v_cvt_pk_bf16_f32 v32, v30, v31
	v_cvt_pk_bf16_f32 v31, v28, v29
	v_add_nc_u32_e32 v28, 0x70, v1
	v_lshrrev_b32_e32 v109, 3, v109
	v_cvt_pk_bf16_f32 v97, v96, v97
	v_cvt_pk_bf16_f32 v96, v94, v95
	v_cvt_pk_bf16_f32 v94, v90, v91
	v_lshrrev_b32_e32 v91, 3, v100
	v_cvt_pk_bf16_f32 v79, v76, v77
	v_lshrrev_b32_e32 v76, 3, v84
	v_cvt_pk_bf16_f32 v57, v56, v57
	v_cvt_pk_bf16_f32 v56, v54, v55
	v_cvt_pk_bf16_f32 v54, v50, v51
	v_lshrrev_b32_e32 v51, 3, v68
	v_add_nc_u32_e32 v1, 0x1070, v1
	v_dual_lshrrev_b32 v116, 3, v116 :: v_dual_lshrrev_b32 v83, 3, v83
	v_dual_lshrrev_b32 v74, 3, v74 :: v_dual_lshrrev_b32 v115, 3, v117
	v_cvt_pk_bf16_f32 v39, v36, v37
	v_dual_lshrrev_b32 v36, 3, v44 :: v_dual_lshrrev_b32 v106, 3, v106
	v_cvt_pk_bf16_f32 v30, v26, v27
	v_lshrrev_b32_e32 v26, 3, v35
	v_cvt_pk_bf16_f32 v25, v24, v25
	v_cvt_pk_bf16_f32 v24, v22, v23
	v_cvt_pk_bf16_f32 v23, v20, v21
	v_lshrrev_b32_e32 v20, 3, v28
	v_and_b32_e32 v109, 0xff0, v109
	v_and_b32_e32 v82, 0xff0, v91
	v_and_b32_e32 v76, 0xff0, v76
	v_and_b32_e32 v42, 0xff0, v51
	v_lshrrev_b32_e32 v1, 3, v1
	v_and_b32_e32 v116, 0x1ff0, v116
	v_and_b32_e32 v83, 0x1ff0, v83
	v_cvt_pk_bf16_f32 v70, v66, v67
	v_and_b32_e32 v67, 0x1ff0, v74
	v_and_b32_e32 v43, 0x1ff0, v43
	v_and_b32_e32 v108, 0xff0, v115
	v_and_b32_e32 v36, 0xff0, v36
	v_cvt_pk_bf16_f32 v102, v98, v99
	v_and_b32_e32 v99, 0x1ff0, v106
	v_and_b32_e32 v26, 0x1ff0, v26
	v_cvt_pk_bf16_f32 v22, v18, v19
	v_and_b32_e32 v19, 0xff0, v20
	v_dual_add_nc_u32 v109, v109, v114 :: v_dual_add_nc_u32 v82, v82, v114
	v_add_nc_u32_e32 v66, v76, v114
	v_cvt_pk_bf16_f32 v55, v52, v53
	v_add_nc_u32_e32 v42, v42, v114
	v_and_b32_e32 v1, 0x1ff0, v1
	v_dual_add_nc_u32 v107, v116, v122 :: v_dual_add_nc_u32 v75, v83, v122
	v_add_nc_u32_e32 v50, v67, v122
	v_add_nc_u32_e32 v34, v43, v122
	v_add_nc_u32_e32 v98, v108, v114
	v_cvt_pk_bf16_f32 v95, v92, v93
	v_add_nc_u32_e32 v27, v36, v114
	v_dual_add_nc_u32 v90, v99, v122 :: v_dual_add_nc_u32 v18, v26, v122
	v_cvt_pk_bf16_f32 v17, v16, v17
	v_cvt_pk_bf16_f32 v16, v14, v15
	v_cvt_pk_bf16_f32 v15, v12, v13
	v_cvt_pk_bf16_f32 v14, v10, v11
	v_add_nc_u32_e32 v10, v19, v114
	v_cvt_pk_bf16_f32 v9, v8, v9
	v_cvt_pk_bf16_f32 v8, v6, v7
	v_cvt_pk_bf16_f32 v7, v4, v5
	v_cvt_pk_bf16_f32 v6, v2, v3
	ds_store_b128 v109, v[110:113] offset:32
	ds_store_b128 v107, v[102:105] offset:32
	ds_store_b128 v98, v[94:97] offset:64
	ds_store_b128 v90, v[86:89] offset:64
	ds_store_b128 v82, v[78:81] offset:96
	ds_store_b128 v75, v[70:73] offset:96
	ds_store_b128 v66, v[54:57] offset:128
	ds_store_b128 v50, v[46:49] offset:128
	v_add_nc_u32_e32 v1, v1, v122
	v_cvt_pk_bf16_f32 v5, v64, v65
	v_cvt_pk_bf16_f32 v4, v62, v63
	v_cvt_pk_bf16_f32 v3, v60, v61
	v_cvt_pk_bf16_f32 v2, v58, v59
	ds_store_b128 v42, v[38:41] offset:160
	ds_store_b128 v34, v[30:33] offset:160
	ds_store_b128 v27, v[22:25] offset:192
	ds_store_b128 v18, v[14:17] offset:192
	ds_store_b128 v10, v[6:9] offset:224
	ds_store_b128 v1, v[2:5] offset:224
.LBB0_39:
	v_cmp_ne_u32_e32 vcc_lo, 1, v131
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
	v_lshl_add_u32 v11, v11, 8, v12
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_mul_lo_u32 v16, v27, s18
	v_dual_add_nc_u32 v24, s24, v27 :: v_dual_sub_nc_u32 v6, v2, v6
	v_ashrrev_i32_e32 v19, 31, v18
	v_lshl_add_u32 v26, v26, 8, v14
	v_mul_u64_e32 v[20:21], s[6:7], v[20:21]
	v_ashrrev_i32_e32 v25, 31, v24
	v_lshl_add_u32 v1, v1, 8, v6
	v_sub_nc_u32_e32 v16, v5, v16
	v_mul_u64_e32 v[18:19], s[20:21], v[18:19]
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
	v_dual_sub_nc_u32 v7, v4, v9 :: v_dual_lshlrev_b32 v4, 8, v4
	v_lshlrev_b32_e32 v9, 8, v9
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
	.size	bm256_bn128_bk064_wm2_wn4_mc0, .Lfunc_end0-bm256_bn128_bk064_wm2_wn4_mc0
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel bm256_bn128_bk064_wm2_wn4_mc0
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
		.amdhsa_next_free_vgpr 256
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
		.amdhsa_inst_pref_size 59
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text.bm256_bn128_bk064_wm2_wn4_mc0,"axG",@progbits,bm256_bn128_bk064_wm2_wn4_mc0,comdat
                                        ; -- End function
	.set .Lbm256_bn128_bk064_wm2_wn4_mc0.num_vgpr, 256
	.set .Lbm256_bn128_bk064_wm2_wn4_mc0.num_agpr, 0
	.set .Lbm256_bn128_bk064_wm2_wn4_mc0.numbered_sgpr, 56
	.set .Lbm256_bn128_bk064_wm2_wn4_mc0.num_named_barrier, 0
	.set .Lbm256_bn128_bk064_wm2_wn4_mc0.private_seg_size, 0
	.set .Lbm256_bn128_bk064_wm2_wn4_mc0.uses_vcc, 1
	.set .Lbm256_bn128_bk064_wm2_wn4_mc0.uses_flat_scratch, 1
	.set .Lbm256_bn128_bk064_wm2_wn4_mc0.has_dyn_sized_stack, 0
	.set .Lbm256_bn128_bk064_wm2_wn4_mc0.has_recursion, 0
	.set .Lbm256_bn128_bk064_wm2_wn4_mc0.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 7436
; TotalNumSgprs: 58
; NumVgprs: 256
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 104448 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 15
; NumSGPRsForWavesPerEU: 58
; NumVGPRsForWavesPerEU: 256
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
	.type	__hip_cuid_2756295065ba1bf3,@object ; @__hip_cuid_2756295065ba1bf3
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_2756295065ba1bf3
__hip_cuid_2756295065ba1bf3:
	.byte	0                               ; 0x0
	.size	__hip_cuid_2756295065ba1bf3, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git aa451e1fe6a793394d6733051b1778633063ae96+PATCHED:d17c5aa0e3ea29cde402f58f27e39b6034effa27)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_2756295065ba1bf3
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
    .group_segment_fixed_size: 104448
    .kernarg_segment_align: 8
    .kernarg_segment_size: 132
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           bm256_bn128_bk064_wm2_wn4_mc0
    .private_segment_fixed_size: 0
    .sgpr_count:     58
    .sgpr_spill_count: 0
    .symbol:         bm256_bn128_bk064_wm2_wn4_mc0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     256
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
