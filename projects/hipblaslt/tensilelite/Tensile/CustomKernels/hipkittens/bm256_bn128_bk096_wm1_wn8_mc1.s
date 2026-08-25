	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.section	.text.bm256_bn128_bk096_wm1_wn8_mc1,"axG",@progbits,bm256_bn128_bk096_wm1_wn8_mc1,comdat
	.protected	bm256_bn128_bk096_wm1_wn8_mc1 ; -- Begin function bm256_bn128_bk096_wm1_wn8_mc1
	.globl	bm256_bn128_bk096_wm1_wn8_mc1
	.p2align	8
	.type	bm256_bn128_bk096_wm1_wn8_mc1,@function
bm256_bn128_bk096_wm1_wn8_mc1: ; @bm256_bn128_bk096_wm1_wn8_mc1
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
	s_lshl2_add_u32 s16, ttmp9, s2
	s_lshl2_add_u32 s4, ttmp7, s3
	s_lshl_b32 s2, s16, 8
	s_wait_kmcnt 0x0
	s_add_co_i32 s3, s28, 0xff
	s_add_co_i32 s6, s29, 0x7f
	s_sub_co_i32 s7, s28, s2
	s_ashr_i32 s8, s3, 31
	s_ashr_i32 s9, s6, 31
	s_min_i32 s31, s7, 0x100
	s_lshr_b32 s7, s8, 24
	s_lshr_b32 s8, s9, 25
	s_add_co_i32 s3, s3, s7
	s_add_co_i32 s7, s6, s8
	s_ashr_i32 s6, s3, 8
	s_ashr_i32 s7, s7, 7
	s_cmp_lt_i32 s16, s6
	s_mov_b32 s9, s30
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_and_b32 s8, s3, exec_lo
	s_cselect_b32 s35, s31, 0
	s_lshl_b32 s33, s4, 7
	s_sub_co_i32 s8, s29, s33
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_min_i32 s8, s8, 0x80
	s_cmp_lt_i32 s4, s7
	s_cselect_b32 s29, -1, 0
	s_and_b32 s10, s29, exec_lo
	s_cselect_b32 s37, s8, 0
	s_add_co_i32 s12, s30, 0x5f
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 2, 1), 1
	s_min_i32 s8, s30, 0x60
	s_cmp_gt_i32 s12, 0x5f
	s_cselect_b32 s13, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_and_b32 s10, s13, exec_lo
	s_cselect_b32 s34, s8, 0
	s_cmp_lt_i32 s35, 0x100
	s_cselect_b32 s42, -1, 0
	s_and_b32 vcc_lo, exec_lo, s42
	s_mov_b32 s8, s42
	s_cbranch_vccnz .LBB0_2
; %bb.1:
	s_cmp_lt_i32 s37, 0x80
	s_cselect_b32 s8, -1, 0
	s_cmp_lt_i32 s34, 0x60
	s_cselect_b32 s10, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 s8, s10, s8
.LBB0_2:
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s8
	s_cbranch_vccnz .LBB0_8
; %bb.3:
	v_dual_mov_b32 v3, 0 :: v_dual_lshlrev_b32 v2, 2, v0
	v_or_b32_e32 v1, 0xffffff00, v0
	s_mov_b32 s8, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v4, v2 :: v_dual_mov_b32 v5, v1
.LBB0_4:                                ; =>This Inner Loop Header: Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)
	v_add_nc_u32_e32 v5, 0x100, v5
	ds_store_b32 v4, v3
	v_add_nc_u32_e32 v4, 0x400, v4
	v_cmp_lt_u32_e32 vcc_lo, 0x31ff, v5
	s_or_b32 s8, vcc_lo, s8
	s_and_not1_b32 exec_lo, exec_lo, s8
	s_cbranch_execnz .LBB0_4
; %bb.5:
	s_or_b32 exec_lo, exec_lo, s8
	v_lshl_add_u32 v2, s5, 2, v2
	v_mov_b32_e32 v3, 0
	s_mov_b32 s8, 0
.LBB0_6:                                ; =>This Inner Loop Header: Depth=1
	v_add_nc_u32_e32 v1, 0x100, v1
	ds_store_b32 v2, v3 offset:52224
	v_add_nc_u32_e32 v2, 0x400, v2
	v_cmp_lt_u32_e32 vcc_lo, 0x187f, v1
	s_or_b32 s8, vcc_lo, s8
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s8
	s_cbranch_execnz .LBB0_6
; %bb.7:
	s_or_b32 exec_lo, exec_lo, s8
	s_barrier_signal -1
	s_barrier_wait -1
.LBB0_8:
	s_clause 0x2
	s_load_b64 s[14:15], s[0:1], 0x0 nv
	s_load_b128 s[24:27], s[0:1], 0x20 nv
	s_load_b128 s[20:23], s[0:1], 0x48 nv
	s_wait_xcnt 0x0
	s_lshl_b32 s1, s4, 2
	v_lshrrev_b32_e32 v137, 5, v0
	s_lshl_b32 s0, s5, 2
	s_add_co_i32 s7, s7, -1
	s_and_b32 s1, s1, 12
	s_mov_b64 s[38:39], src_shared_base
	s_or_b32 s43, s0, 0xcc00
	s_add_co_i32 s17, s6, -1
	s_min_i32 s0, s4, s7
	s_and_b32 s19, s16, 3
	s_lshl_b32 s1, 15, s1
	s_mov_b32 s4, exec_lo
	v_cmpx_lt_i32_e32 0, v137
	s_xor_b32 s18, exec_lo, s4
	s_cbranch_execz .LBB0_12
; %bb.9:
	s_mov_b32 s38, exec_lo
	v_cmpx_eq_u32_e32 1, v137
	s_cbranch_execz .LBB0_11
; %bb.10:
	s_cmp_gt_i32 s34, 0
	s_mov_b32 s36, s34
	s_cselect_b32 s8, -1, 0
	s_lshl_b32 s4, s0, 7
	s_wait_kmcnt 0x0
	s_bfe_i64 s[6:7], s[20:21], 0x200000
	s_ashr_i32 s5, s4, 31
	s_mov_b32 s10, 0
	s_mul_u64 s[4:5], s[6:7], s[4:5]
	s_mov_b32 s11, s10
	s_lshl_b64 s[4:5], s[4:5], 1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_nc_u64 s[6:7], s[26:27], s[4:5]
	v_dual_mov_b32 v1, s43 :: v_dual_mov_b32 v4, s6
	s_and_b32 s5, s7, 0x1ffffff
	s_and_b32 s7, s29, s8
	s_bitset1_b32 s5, 31
	v_cndmask_b32_e64 v2, 0, 1, s7
	v_mov_b32_e32 v3, s5
	s_lshr_b64 s[6:7], s[36:37], 16
	v_readfirstlane_b32 s45, v1
	v_readfirstlane_b32 s46, v4
	v_readfirstlane_b32 s44, v2
	v_readfirstlane_b32 s47, v3
	s_lshr_b32 s7, s37, 16
	s_or_b32 s4, s1, 0x7510000
	s_lshl_b32 s5, s34, 16
	s_or_b32 s7, s7, 0x600000
	s_movk_i32 s8, 0x80
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[44:47], s[4:11]
.LBB0_11:
	s_or_b32 exec_lo, exec_lo, s38
.LBB0_12:
	s_or_saveexec_b32 s36, s18
	s_min_i32 s18, s16, s17
	s_lshl_b32 s17, 0x1111, s19
	s_xor_b32 exec_lo, exec_lo, s36
	s_cbranch_execz .LBB0_14
; %bb.13:
	s_cmp_gt_i32 s34, 0
	s_mov_b32 s10, 0
	s_cselect_b32 s8, -1, 0
	s_lshl_b32 s4, s18, 8
	s_wait_kmcnt 0x0
	s_bfe_i64 s[6:7], s[24:25], 0x200000
	s_ashr_i32 s5, s4, 31
	s_and_b32 s8, s3, s8
	s_mul_u64 s[4:5], s[6:7], s[4:5]
	v_cndmask_b32_e64 v2, 0, 1, s8
	s_lshl_b64 s[6:7], s[4:5], 1
	s_lshr_b32 s8, s35, 16
	s_add_nc_u64 s[6:7], s[14:15], s[6:7]
	s_or_b32 s4, s17, 0x7510000
	s_and_b32 s7, s7, 0x1ffffff
	v_readfirstlane_b32 s44, v2
	s_bitset1_b32 s7, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v4, s6 :: v_dual_mov_b32 v3, s7
	s_lshr_b64 s[6:7], s[34:35], 16
	s_lshl_b32 s5, s34, 16
	s_or_b32 s7, s8, 0x600000
	v_readfirstlane_b32 s46, v4
	v_readfirstlane_b32 s47, v3
	s_movk_i32 s8, 0x100
	s_mov_b32 s11, s10
	s_mov_b32 s45, s10
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[44:47], s[4:11]
.LBB0_14:
	s_or_b32 exec_lo, exec_lo, s36
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
	v_dual_mov_b32 v9, 0 :: v_dual_lshlrev_b32 v135, 4, v137
	s_and_b32 s38, s3, s29
	s_and_not1_b32 vcc_lo, exec_lo, s13
	v_cndmask_b32_e64 v131, 0, 1, s38
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
	v_dual_mov_b32 v58, v9 :: v_dual_mov_b32 v73, v9
	v_dual_mov_b32 v72, v9 :: v_dual_mov_b32 v71, v9
	v_dual_mov_b32 v70, v9 :: v_dual_mov_b32 v69, v9
	v_dual_mov_b32 v68, v9 :: v_dual_mov_b32 v67, v9
	v_dual_mov_b32 v66, v9 :: v_dual_mov_b32 v81, v9
	v_dual_mov_b32 v80, v9 :: v_dual_mov_b32 v79, v9
	v_dual_mov_b32 v78, v9 :: v_dual_mov_b32 v77, v9
	v_dual_mov_b32 v76, v9 :: v_dual_mov_b32 v75, v9
	v_dual_mov_b32 v74, v9 :: v_dual_mov_b32 v97, v9
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
	v_dual_mov_b32 v122, v9 :: v_dual_mov_b32 v89, v9
	v_dual_mov_b32 v88, v9 :: v_dual_mov_b32 v87, v9
	v_dual_mov_b32 v86, v9 :: v_dual_mov_b32 v85, v9
	v_dual_mov_b32 v84, v9 :: v_dual_mov_b32 v83, v9
	v_mov_b32_e32 v82, v9
	s_barrier_wait -1
	s_barrier_wait -3
	s_cbranch_vccnz .LBB0_41
; %bb.17:
	s_mov_b64 s[4:5], src_shared_base
	s_add_co_i32 s6, s43, 0x6600
	s_mov_b32 s7, s5
	v_and_b32_e32 v5, 15, v0
	s_and_b64 s[6:7], s[6:7], 15
	s_mov_b32 s11, 0
	s_sub_co_i32 s4, 16, s6
	v_mul_u32_u24_e64 v3, 0x60, 0
	s_lshr_b32 s4, s4, 2
	s_cmp_lg_u64 s[6:7], 0
	v_mul_u32_u24_e32 v7, 0x60, v5
	s_cselect_b32 s4, s4, 0
	s_mul_hi_i32 s6, s12, 0x2aaaaaab
	s_lshl2_add_u32 s7, s4, s43
	v_mul_u32_u24_e32 v1, 0x60, v135
	s_add_co_i32 s4, s7, 0x13200
	s_add_co_i32 s45, s7, 0x6600
	s_and_b32 s10, s4, 15
	v_lshrrev_b32_e32 v2, 4, v7
	s_sub_co_i32 s8, 16, s10
	v_or_b32_e32 v11, 0x1800, v7
	s_lshr_b32 s7, s8, 2
	s_cmp_lg_u64 s[10:11], 0
	v_and_b32_e32 v2, 0x78, v2
	s_cselect_b32 s7, s7, 0
	s_lshr_b32 s8, s6, 31
	s_ashr_i32 s47, s6, 4
	s_lshl_b32 s10, s7, 2
	s_add_co_i32 s47, s47, s8
	s_cmp_lt_i32 s37, 0x80
	v_and_b32_e32 v49, 16, v0
	s_cselect_b32 s48, -1, 0
	s_lshl_b32 s6, s0, 7
	s_movk_i32 s0, 0x600
	v_or_b32_e32 v23, 0x3000, v7
	v_mad_u32_u24 v4, 0x60, v5, s0
	s_movk_i32 s0, 0xc00
	v_or3_b32 v51, v3, v49, v7
	v_mad_u32_u24 v6, 0x60, v5, s0
	s_movk_i32 s0, 0x1200
	v_dual_mov_b32 v133, 0 :: v_dual_lshrrev_b32 v4, 4, v4
	v_mad_u32_u24 v8, 0x60, v5, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_add_nc_u32 v130, v2, v51 :: v_dual_lshrrev_b32 v6, 4, v6
	v_and_b32_e32 v9, 0xf8, v4
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v79, v133 :: v_dual_add_nc_u32 v2, 0x1e00, v51
	v_add_nc_u32_e32 v4, 0x2400, v51
	v_lshrrev_b32_e32 v8, 4, v8
	v_and_b32_e32 v13, 0x1f8, v6
	v_dual_lshrrev_b32 v10, 4, v11 :: v_dual_lshrrev_b32 v12, 4, v2
	v_add_nc_u32_e32 v6, 0x2a00, v51
	v_lshrrev_b32_e32 v14, 4, v4
	v_and_b32_e32 v15, 0x1f8, v8
	v_dual_mov_b32 v81, v133 :: v_dual_add_nc_u32 v8, 0x3600, v51
	v_and_b32_e32 v19, 0x3f8, v12
	v_lshrrev_b32_e32 v12, 4, v6
	v_and_b32_e32 v21, 0x2f8, v14
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_lshrrev_b32 v14, 4, v23 :: v_dual_lshrrev_b32 v16, 4, v8
	v_or_b32_e32 v27, 0x4800, v7
	v_and_b32_e32 v17, 0x1f8, v10
	v_add_nc_u32_e32 v10, 0x3c00, v51
	v_and_b32_e32 v25, 0x3f8, v12
	v_dual_mov_b32 v83, v133 :: v_dual_add_nc_u32 v12, 0x4200, v51
	v_and_b32_e32 v31, 0x3f8, v16
	v_dual_lshrrev_b32 v16, 4, v27 :: v_dual_bitop2_b32 v37, 32, v49 bitop3:0x54
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_lshrrev_b32_e32 v20, 4, v12
	v_and_b32_e32 v29, 0x378, v14
	v_add_nc_u32_e32 v14, 0x4e00, v51
	v_and_b32_e32 v39, 0x4f8, v16
	v_add_nc_u32_e32 v148, v25, v51
	v_dual_mov_b32 v25, v133 :: v_dual_lshrrev_b32 v18, 4, v10
	v_and_b32_e32 v35, 0x4f8, v20
	v_dual_lshrrev_b32 v22, 4, v14 :: v_dual_bitop2_b32 v1, v1, v49 bitop3:0x54
	v_add_nc_u32_e32 v16, 0x5400, v51
	v_or_b32_e32 v20, v37, v3
	v_and_b32_e32 v33, 0x7f8, v18
	s_delay_alu instid0(VALU_DEP_4)
	v_mad_u32_u24 v1, 0x60, v5, v1
	v_add_nc_u32_e32 v18, 0x5a00, v51
	v_lshrrev_b32_e32 v24, 4, v16
	v_mad_u32_u24 v41, 0x60, v5, v20
	v_and_b32_e32 v43, 0x5f8, v22
	v_add_nc_u32_e32 v150, v29, v51
	v_dual_mov_b32 v29, v133 :: v_dual_lshrrev_b32 v20, 4, v1
	v_lshrrev_b32_e32 v22, 4, v18
	v_and_b32_e32 v45, 0x5f8, v24
	v_add_nc_u32_e32 v24, 0x600, v41
	v_lshrrev_b32_e32 v26, 4, v41
	v_and_b32_e32 v20, 0x7f8, v20
	v_and_b32_e32 v47, 0x7f8, v22
	v_add_nc_u32_e32 v22, 0xc00, v41
	v_dual_lshrrev_b32 v24, 4, v24 :: v_dual_add_nc_u32 v142, v17, v51
	v_dual_mov_b32 v31, v133 :: v_dual_add_nc_u32 v152, v31, v51
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_dual_add_nc_u32 v134, v20, v1 :: v_dual_lshrrev_b32 v20, 4, v22
	v_add_nc_u32_e32 v1, 0x1200, v41
	v_and_b32_e32 v132, 0x1f8, v24
	v_add_nc_u32_e32 v22, v37, v11
	v_add_nc_u32_e32 v24, 0x1e00, v41
	v_and_b32_e32 v53, 0xf8, v26
	v_lshrrev_b32_e32 v1, 4, v1
	v_add_nc_u32_e32 v26, 0x2400, v41
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_dual_lshrrev_b32 v28, 4, v22 :: v_dual_lshrrev_b32 v30, 4, v24
	v_add_nc_u32_e32 v34, 0x3c00, v41
	v_dual_add_nc_u32 v154, v33, v51 :: v_dual_add_nc_u32 v164, v47, v51
	v_dual_mov_b32 v33, v133 :: v_dual_lshrrev_b32 v32, 4, v26
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v26, 0x3f8, v30
	v_add_nc_u32_e32 v30, v37, v23
	v_and_b32_e32 v24, 0x3f8, v28
	v_add_nc_u32_e32 v42, v37, v27
	v_and_b32_e32 v28, 0x3f8, v32
	v_add_nc_u32_e32 v32, 0x3600, v41
	v_dual_add_nc_u32 v136, v9, v51 :: v_dual_bitop2_b32 v55, 64, v49 bitop3:0x54
	v_dual_add_nc_u32 v140, v15, v51 :: v_dual_add_nc_u32 v146, v21, v51
	v_dual_mov_b32 v35, v133 :: v_dual_add_nc_u32 v156, v35, v51
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_lshrrev_b32 v36, 4, v30 :: v_dual_lshrrev_b32 v38, 4, v32
	v_dual_mov_b32 v9, v133 :: v_dual_lshrrev_b32 v40, 4, v34
	v_mad_u32_u24 v57, 0x60, v135, v7
	v_and_b32_e32 v32, 0x3f8, v36
	v_add_nc_u32_e32 v78, 0x600, v51
	v_and_b32_e32 v20, 0x1f8, v20
	v_and_b32_e32 v36, 0x7f8, v40
	v_lshrrev_b32_e32 v40, 4, v42
	v_add_nc_u32_e32 v42, 0x5400, v41
	v_dual_add_nc_u32 v37, v57, v37 :: v_dual_bitop2_b32 v3, v3, v55 bitop3:0x54
	v_dual_mov_b32 v21, v133 :: v_dual_add_nc_u32 v158, v39, v51
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshrrev_b32_e32 v7, 4, v42
	v_mad_u32_u24 v3, 0x60, v5, v3
	v_add_nc_u32_e32 v5, 0x5a00, v41
	v_mov_b32_e32 v39, v133
	v_add_nc_u64_e32 v[168:169], v[132:133], v[78:79]
	v_and_b32_e32 v44, 0x7f8, v7
	v_lshrrev_b32_e32 v7, 4, v37
	v_lshrrev_b32_e32 v5, 4, v5
	v_add_nc_u32_e32 v37, 0xc00, v3
	v_add_nc_u32_e32 v132, 0xc00, v51
	v_dual_add_nc_u32 v138, v13, v51 :: v_dual_add_nc_u32 v144, v19, v51
	v_and_b32_e32 v59, 0x7f8, v7
	v_add_nc_u32_e32 v7, v55, v11
	v_and_b32_e32 v46, 0x7f8, v5
	v_lshrrev_b32_e32 v5, 4, v37
	v_add_nc_u32_e32 v11, 0x2400, v3
	v_or_b32_e32 v82, 0x1800, v51
	v_mov_b32_e32 v13, v133
	v_add_nc_u64_e32 v[170:171], v[20:21], v[132:133]
	v_and_b32_e32 v50, 0x1f8, v5
	v_add_nc_u32_e32 v5, 0x1e00, v3
	v_or_b32_e32 v20, 0x3000, v51
	v_dual_mov_b32 v63, v133 :: v_dual_mov_b32 v65, v133
	v_add_nc_u64_e32 v[174:175], v[24:25], v[82:83]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshrrev_b32_e32 v5, 4, v5
	v_add_nc_u64_e32 v[182:183], v[32:33], v[20:21]
	v_or_b32_e32 v24, v57, v49
	v_and_b32_e32 v22, 0x3f8, v1
	v_add_nc_u32_e32 v1, 0x2a00, v41
	v_and_b32_e32 v56, 0x3f8, v5
	v_add_nc_u32_e32 v5, v55, v23
	v_add_nc_u32_e32 v23, v55, v27
	v_dual_add_nc_u32 v162, v45, v51 :: v_dual_add_nc_u32 v198, v59, v24
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v59, v133 :: v_dual_lshrrev_b32 v5, 4, v5
	v_lshrrev_b32_e32 v7, 4, v7
	v_and_b32_e32 v34, 0x7f8, v38
	v_dual_mov_b32 v69, v133 :: v_dual_mov_b32 v71, v133
	v_and_b32_e32 v62, 0x3f8, v5
	v_lshrrev_b32_e32 v5, 4, v23
	v_and_b32_e32 v54, 0x3f8, v7
	v_add_nc_u32_e32 v7, 0x3600, v3
	v_add_nc_u64_e32 v[184:185], v[34:35], v[8:9]
	v_add_nc_u64_e32 v[216:217], v[62:63], v[20:21]
	v_and_b32_e32 v70, 0x5f8, v5
	v_add_nc_u32_e32 v5, v57, v55
	v_dual_mov_b32 v20, v133 :: v_dual_lshrrev_b32 v11, 4, v11
	v_dual_mov_b32 v57, v133 :: v_dual_mov_b32 v32, v133
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_lshrrev_b32 v5, 4, v5 :: v_dual_mov_b32 v23, v133
	v_and_b32_e32 v58, 0x3f8, v11
	v_add_nc_u32_e32 v11, 0x4200, v3
	v_add_nc_u32_e32 v80, 0x1200, v51
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v84, 0x7f8, v5
	v_mov_b32_e32 v5, v133
	v_add_nc_u32_e32 v166, v53, v51
	v_and_b32_e32 v40, 0x5f8, v40
	v_add_nc_u32_e32 v37, 0x2a00, v3
	v_add_nc_u32_e32 v232, v84, v24
	v_add_nc_u64_e32 v[178:179], v[28:29], v[4:5]
	v_add_nc_u64_e32 v[212:213], v[58:59], v[4:5]
	v_dual_mov_b32 v4, v133 :: v_dual_lshrrev_b32 v1, 4, v1
	v_dual_mov_b32 v24, v133 :: v_dual_lshrrev_b32 v7, 4, v7
	v_dual_mov_b32 v28, v133 :: v_dual_lshrrev_b32 v11, 4, v11
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_and_b32_e32 v30, 0x3f8, v1
	v_add_nc_u32_e32 v1, 0x4200, v41
	v_and_b32_e32 v64, 0x7f8, v7
	v_add_nc_u32_e32 v7, 0x5400, v3
	v_and_b32_e32 v68, 0x5f8, v11
	v_sub_nc_u32_e32 v11, 0x197f, v0
	v_lshrrev_b32_e32 v1, 4, v1
	v_add_nc_u64_e32 v[218:219], v[64:65], v[8:9]
	v_mov_b32_e32 v8, v133
	v_add_nc_u64_e32 v[222:223], v[68:69], v[12:13]
	v_lshrrev_b32_e32 v7, 4, v7
	v_and_b32_e32 v38, 0x5f8, v1
	v_add_nc_u32_e32 v1, 0x4e00, v41
	v_lshrrev_b32_e32 v41, 4, v3
	s_or_b32 s12, s1, 0x7510000
	v_and_b32_e32 v74, 0x7f8, v7
	v_add_nc_u64_e32 v[188:189], v[38:39], v[12:13]
	v_dual_mov_b32 v12, v133 :: v_dual_lshrrev_b32 v1, 4, v1
	v_and_b32_e32 v61, 0xf8, v41
	v_mov_b32_e32 v41, v133
	v_add_nc_u64_e32 v[172:173], v[22:23], v[80:81]
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_and_b32_e32 v42, 0x5f8, v1
	v_add_nc_u32_e32 v1, 0x600, v3
	v_or_b32_e32 v22, 0x4800, v51
	v_dual_mov_b32 v7, v133 :: v_dual_mov_b32 v34, v133
	v_dual_add_nc_u32 v160, v43, v51 :: v_dual_lshrrev_b32 v1, 4, v1
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u64_e32 v[190:191], v[40:41], v[22:23]
	v_add_nc_u64_e32 v[224:225], v[70:71], v[22:23]
	v_mov_b32_e32 v22, v133
	v_add_nc_u64_e32 v[180:181], v[30:31], v[6:7]
	v_and_b32_e32 v48, 0x1f8, v1
	v_dual_mov_b32 v30, v133 :: v_dual_add_nc_u32 v1, 0x1200, v3
	v_dual_mov_b32 v27, v133 :: v_dual_mov_b32 v43, v133
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v15, v133 :: v_dual_lshrrev_b32 v1, 4, v1
	v_dual_mov_b32 v45, v133 :: v_dual_mov_b32 v17, v133
	v_dual_mov_b32 v47, v133 :: v_dual_mov_b32 v19, v133
	v_and_b32_e32 v52, 0x3f8, v1
	v_dual_lshrrev_b32 v1, 4, v37 :: v_dual_mov_b32 v37, v133
	v_dual_mov_b32 v49, v133 :: v_dual_mov_b32 v53, v133
	v_mov_b32_e32 v55, v133
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_and_b32_e32 v60, 0x3f8, v1
	v_add_nc_u32_e32 v1, 0x3c00, v3
	v_dual_mov_b32 v67, v133 :: v_dual_mov_b32 v73, v133
	v_dual_mov_b32 v75, v133 :: v_dual_mov_b32 v77, v133
	v_dual_mov_b32 v235, v133 :: v_dual_lshrrev_b32 v1, 4, v1
	s_ashr_i32 s7, s6, 31
	s_lshl_b32 s18, s18, 8
	s_wait_kmcnt 0x0
	s_bfe_i64 s[20:21], s[20:21], 0x200000
	s_ashr_i32 s19, s18, 31
	v_and_b32_e32 v66, 0x7f8, v1
	v_add_nc_u32_e32 v1, 0x4e00, v3
	v_add_nc_u32_e32 v3, 0x5a00, v3
	s_mul_u64 s[6:7], s[20:21], s[6:7]
	s_bfe_i64 s[20:21], s[24:25], 0x200000
	v_add_nc_u64_e32 v[192:193], v[42:43], v[14:15]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_dual_lshrrev_b32 v1, 4, v1 :: v_dual_lshrrev_b32 v3, 4, v3
	s_mul_u64 s[18:19], s[20:21], s[18:19]
	v_add_nc_u64_e32 v[196:197], v[46:47], v[18:19]
	v_add_nc_u64_e32 v[206:207], v[52:53], v[80:81]
	v_and_b32_e32 v72, 0x5f8, v1
	v_lshrrev_b32_e32 v1, 8, v11
	v_and_b32_e32 v76, 0x7f8, v3
	v_mov_b32_e32 v11, v133
	v_add_nc_u64_e32 v[208:209], v[54:55], v[82:83]
	v_add_nc_u64_e32 v[226:227], v[72:73], v[14:15]
	v_add_nc_u32_e32 v3, 1, v1
	v_add_nc_u64_e32 v[230:231], v[76:77], v[18:19]
	v_add_nc_u64_e32 v[186:187], v[36:37], v[10:11]
	v_add_nc_u64_e32 v[220:221], v[66:67], v[10:11]
	v_cmp_eq_u32_e64 s0, 0, v137
	v_and_b32_e32 v139, 26, v3
	v_or_b32_e32 v1, 0x100, v0
	v_or_b32_e32 v141, 0x3100, v0
	v_lshl_or_b32 v234, v0, 2, 0xc800
	v_dual_mov_b32 v237, v133 :: v_dual_mov_b32 v40, v133
	v_lshl_or_b32 v85, v139, 8, v0
	v_cmp_ne_u32_e64 s1, v3, v139
	v_mov_b32_e32 v3, v133
	v_add_nc_u32_e32 v200, v61, v51
	v_dual_mov_b32 v51, v133 :: v_dual_mov_b32 v61, v133
	v_add_nc_u64_e32 v[194:195], v[44:45], v[16:17]
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_nc_u64_e32 v[176:177], v[26:27], v[2:3]
	v_add_nc_u64_e32 v[202:203], v[48:49], v[78:79]
	v_add_nc_u64_e32 v[204:205], v[50:51], v[132:133]
	v_add_nc_u64_e32 v[210:211], v[56:57], v[2:3]
	v_add_nc_u64_e32 v[214:215], v[60:61], v[6:7]
	v_add_nc_u64_e32 v[228:229], v[74:75], v[16:17]
	v_dual_mov_b32 v38, v133 :: v_dual_add_nc_u32 v143, 0xffffff00, v85
	v_dual_lshlrev_b32 v236, 2, v85 :: v_dual_mov_b32 v2, v133
	v_dual_mov_b32 v6, v133 :: v_dual_mov_b32 v10, v133
	v_dual_mov_b32 v14, v133 :: v_dual_mov_b32 v16, v133
	v_dual_mov_b32 v18, v133 :: v_dual_mov_b32 v26, v133
	v_dual_mov_b32 v36, v133 :: v_dual_mov_b32 v42, v133
	v_dual_mov_b32 v44, v133 :: v_dual_mov_b32 v46, v133
	v_dual_mov_b32 v48, v133 :: v_dual_mov_b32 v50, v133
	v_dual_mov_b32 v52, v133 :: v_dual_mov_b32 v54, v133
	v_dual_mov_b32 v56, v133 :: v_dual_mov_b32 v58, v133
	v_dual_mov_b32 v60, v133 :: v_dual_mov_b32 v62, v133
	v_dual_mov_b32 v64, v133 :: v_dual_mov_b32 v66, v133
	v_dual_mov_b32 v68, v133 :: v_dual_mov_b32 v70, v133
	v_dual_mov_b32 v72, v133 :: v_dual_mov_b32 v74, v133
	v_dual_mov_b32 v76, v133 :: v_dual_mov_b32 v78, v133
	v_dual_mov_b32 v80, v133 :: v_dual_mov_b32 v90, v133
	v_dual_mov_b32 v91, v133 :: v_dual_mov_b32 v92, v133
	v_dual_mov_b32 v93, v133 :: v_dual_mov_b32 v94, v133
	v_dual_mov_b32 v95, v133 :: v_dual_mov_b32 v96, v133
	v_dual_mov_b32 v97, v133 :: v_dual_mov_b32 v98, v133
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
	v_dual_mov_b32 v129, v133 :: v_dual_mov_b32 v82, v133
	v_dual_mov_b32 v84, v133 :: v_dual_mov_b32 v85, v133
	v_dual_mov_b32 v86, v133 :: v_dual_mov_b32 v87, v133
	v_dual_mov_b32 v88, v133 :: v_dual_mov_b32 v89, v133
	s_lshr_b32 s49, s37, 16
	s_lshr_b32 s50, s35, 16
	s_lshl_b64 s[6:7], s[6:7], 1
	s_lshl_b64 s[18:19], s[18:19], 1
	s_mov_b32 s44, s39
	s_mov_b32 s46, s5
	s_add_nc_u64 s[40:41], s[4:5], s[10:11]
	s_movk_i32 s16, 0x80
	s_or_b32 s49, s49, 0x600000
	s_or_b32 s4, s17, 0x7510000
	s_or_b32 s50, s50, 0x600000
	s_add_nc_u64 s[20:21], s[26:27], s[6:7]
	s_add_nc_u64 s[24:25], s[14:15], s[18:19]
	s_mov_b32 s26, -1
	s_movk_i32 s8, 0x100
	s_mov_b32 s27, s11
	s_branch .LBB0_19
.LBB0_18:                               ;   in Loop: Header=BB0_19 Depth=1
	s_or_b32 exec_lo, exec_lo, s5
	s_cmp_eq_u32 s27, s47
	s_barrier_wait -1
	s_barrier_wait -3
	s_cbranch_scc1 .LBB0_41
.LBB0_19:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_21 Depth 2
                                        ;     Child Loop BB0_24 Depth 2
                                        ;     Child Loop BB0_26 Depth 2
                                        ;     Child Loop BB0_29 Depth 2
	s_and_b32 s51, s27, 1
	s_add_co_i32 s27, s27, 1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_i32 s5, s27, 0xffffffa0
	s_add_co_i32 s6, s5, s30
	s_xor_b32 s5, s51, 1
	s_min_i32 s6, s6, 0x60
	s_cmp_lt_i32 s27, s47
	s_cselect_b32 s10, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_and_b32 s7, s10, exec_lo
	s_cselect_b32 s34, s6, 0
	s_cmp_lt_i32 s34, 0x60
	s_cselect_b32 s6, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b32 s6, s48, s6
	s_or_b32 s6, s42, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s6
	s_cbranch_vccnz .LBB0_31
; %bb.20:                               ;   in Loop: Header=BB0_19 Depth=1
	v_mov_b64_e32 v[238:239], v[0:1]
	v_mov_b32_e32 v145, 50
	s_cmp_lg_u32 s5, 0
	s_mov_b32 s13, 0
	s_cselect_b32 s7, s46, s39
	s_cselect_b32 s6, s45, 0
.LBB0_21:                               ;   Parent Loop BB0_19 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_dual_mov_b32 v132, v238 :: v_dual_add_nc_u32 v145, -2, v145
	v_dual_mov_b32 v240, v239 :: v_dual_mov_b32 v241, v133
	v_add_nc_u32_e32 v239, 0x200, v239
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u64 v[242:243], v[132:133], 2, s[6:7]
	v_cmp_eq_u32_e32 vcc_lo, 0, v145
	v_add_nc_u32_e32 v238, 0x200, v238
	v_lshl_add_u64 v[240:241], v[240:241], 2, s[6:7]
	s_clause 0x1
	flat_store_b32 v[242:243], v133
	flat_store_b32 v[240:241], v133
	s_or_b32 s13, vcc_lo, s13
	s_wait_xcnt 0x0
	s_and_not1_b32 exec_lo, exec_lo, s13
	s_cbranch_execnz .LBB0_21
; %bb.22:                               ;   in Loop: Header=BB0_19 Depth=1
	s_or_b32 exec_lo, exec_lo, s13
	s_and_saveexec_b32 s13, s26
	s_cbranch_execz .LBB0_25
; %bb.23:                               ;   in Loop: Header=BB0_19 Depth=1
	v_add_nc_u64_e32 v[238:239], s[6:7], v[234:235]
	v_mov_b32_e32 v132, v141
	s_mov_b32 s6, 0
.LBB0_24:                               ;   Parent Loop BB0_19 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v132, 0x100, v132
	flat_store_b32 v[238:239], v133
	s_wait_xcnt 0x0
	v_add_nc_u64_e32 v[238:239], 0x400, v[238:239]
	v_cmp_lt_u32_e32 vcc_lo, 0x31ff, v132
	s_or_b32 s6, vcc_lo, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s6
	s_cbranch_execnz .LBB0_24
.LBB0_25:                               ;   in Loop: Header=BB0_19 Depth=1
	s_or_b32 exec_lo, exec_lo, s13
	v_mov_b64_e32 v[238:239], v[0:1]
	v_mov_b32_e32 v145, v139
	s_cmp_lg_u32 s5, 0
	s_mov_b32 s13, 0
	s_cselect_b32 s7, s41, s44
	s_cselect_b32 s6, s40, s43
.LBB0_26:                               ;   Parent Loop BB0_19 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_dual_mov_b32 v132, v238 :: v_dual_add_nc_u32 v145, -2, v145
	v_dual_mov_b32 v240, v239 :: v_dual_mov_b32 v241, v133
	v_add_nc_u32_e32 v239, 0x200, v239
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u64 v[242:243], v[132:133], 2, s[6:7]
	v_cmp_eq_u32_e32 vcc_lo, 0, v145
	v_add_nc_u32_e32 v238, 0x200, v238
	v_lshl_add_u64 v[240:241], v[240:241], 2, s[6:7]
	s_clause 0x1
	flat_store_b32 v[242:243], v133
	flat_store_b32 v[240:241], v133
	s_or_b32 s13, vcc_lo, s13
	s_wait_xcnt 0x0
	s_and_not1_b32 exec_lo, exec_lo, s13
	s_cbranch_execnz .LBB0_26
; %bb.27:                               ;   in Loop: Header=BB0_19 Depth=1
	s_or_b32 exec_lo, exec_lo, s13
	s_and_saveexec_b32 s13, s1
	s_cbranch_execz .LBB0_30
; %bb.28:                               ;   in Loop: Header=BB0_19 Depth=1
	v_add_nc_u64_e32 v[238:239], s[6:7], v[236:237]
	v_mov_b32_e32 v132, v143
	s_mov_b32 s6, 0
.LBB0_29:                               ;   Parent Loop BB0_19 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v132, 0x100, v132
	flat_store_b32 v[238:239], v133
	s_wait_xcnt 0x0
	v_add_nc_u64_e32 v[238:239], 0x400, v[238:239]
	v_cmp_lt_u32_e32 vcc_lo, 0x187f, v132
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
	s_cselect_b32 s6, s27, 0
	s_mov_b32 s7, exec_lo
	v_cmpx_lt_i32_e32 0, v137
	s_xor_b32 s7, exec_lo, s7
	s_cbranch_execnz .LBB0_37
; %bb.32:                               ;   in Loop: Header=BB0_19 Depth=1
	s_and_not1_saveexec_b32 s13, s7
	s_cbranch_execnz .LBB0_40
.LBB0_33:                               ;   in Loop: Header=BB0_19 Depth=1
	s_or_b32 exec_lo, exec_lo, s13
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s38
	s_cbranch_vccnz .LBB0_35
.LBB0_34:                               ;   in Loop: Header=BB0_19 Depth=1
	s_cmp_lg_u32 s51, 0
	s_cselect_b32 s6, s45, 0
	s_cselect_b32 s5, s40, s43
	v_lshl_add_u32 v132, v130, 1, s6
	v_lshl_add_u32 v145, v136, 1, s6
	v_lshl_add_u32 v147, v138, 1, s6
	v_lshl_add_u32 v149, v140, 1, s6
	v_lshl_add_u32 v151, v174, 1, s6
	ds_load_b128 v[238:241], v132
	ds_load_b128 v[242:245], v132 offset:16
	ds_load_b128 v[246:249], v145 offset:3072
	ds_load_b128 v[250:253], v145 offset:3088
	v_lshl_add_u32 v132, v142, 1, s6
	v_lshl_add_u32 v145, v144, 1, s6
	ds_load_b128 v[254:257], v147 offset:6144
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[2:5] /*v[258:261]*/, v147 offset:6160
	ds_load_b128 v[6:9] /*v[262:265]*/, v149 offset:9216
	ds_load_b128 v[10:13] /*v[266:269]*/, v149 offset:9232
	ds_load_b128 v[14:17] /*v[270:273]*/, v132 offset:12288
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v147, v146, 1, s6
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[18:21] /*v[274:277]*/, v132 offset:12304
	ds_load_b128 v[22:25] /*v[278:281]*/, v145 offset:15360
	ds_load_b128 v[26:29] /*v[282:285]*/, v145 offset:15376
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v132, v148, 1, s6
	v_lshl_add_u32 v145, v150, 1, s6
	v_lshl_add_u32 v149, v134, 1, s5
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[30:33] /*v[286:289]*/, v147 offset:18432
	ds_load_b128 v[34:37] /*v[290:293]*/, v147 offset:18448
	ds_load_b128 v[38:41] /*v[294:297]*/, v132 offset:21504
	ds_load_b128 v[42:45] /*v[298:301]*/, v132 offset:21520
	ds_load_b128 v[46:49] /*v[302:305]*/, v145 offset:24576
	ds_load_b128 v[58:61] /*v[314:317]*/, v149
	ds_load_b128 v[62:65] /*v[318:321]*/, v149 offset:16
	s_set_vgpr_msb 0x4004                   ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_lshl_add_u32 v147, v164, 1, s6
	v_lshl_add_u32 v132, v162, 1, s6
	v_lshl_add_u32 v149, v172, 1, s6
	v_lshl_add_u32 v153, v176, 1, s6
	v_lshl_add_u32 v155, v178, 1, s6
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[122:129], v[238:245], v[58:65] /*v[314:321]*/, v[122:129]
	v_lshl_add_u32 v157, v180, 1, s6
	v_lshl_add_u32 v159, v182, 1, s6
	v_lshl_add_u32 v161, v184, 1, s6
	v_lshl_add_u32 v163, v186, 1, s6
	v_lshl_add_u32 v165, v188, 1, s6
	v_lshl_add_u32 v167, v190, 1, s6
	v_lshl_add_u32 v169, v192, 1, s6
	v_wmma_f32_16x16x32_bf16 v[114:121], v[246:253], v[58:65] /*v[314:321]*/, v[114:121] matrix_b_reuse
	v_lshl_add_u32 v171, v194, 1, s6
	v_lshl_add_u32 v173, v196, 1, s6
	v_lshl_add_u32 v175, v198, 1, s5
	s_set_vgpr_msb 0x440                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[66:69] /*v[322:325]*/, v151 offset:64
	ds_load_b128 v[70:73] /*v[326:329]*/, v151 offset:80
	ds_load_b128 v[74:77] /*v[330:333]*/, v153 offset:64
	ds_load_b128 v[78:81] /*v[334:337]*/, v153 offset:80
	s_set_vgpr_msb 0x4004                   ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[106:113], v[254:261], v[58:65] /*v[314:321]*/, v[106:113] matrix_b_reuse
	ds_load_b128 v[254:257], v149 offset:64
	s_set_vgpr_msb 0x440                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[2:5] /*v[258:261]*/, v149 offset:80
	ds_load_b128 v[82:85] /*v[338:341]*/, v157 offset:64
	ds_load_b128 v[86:89] /*v[342:345]*/, v157 offset:80
	ds_load_b128 v[90:93] /*v[346:349]*/, v159 offset:64
	ds_load_b128 v[94:97] /*v[350:353]*/, v159 offset:80
	ds_load_b128 v[98:101] /*v[354:357]*/, v161 offset:64
	s_set_vgpr_msb 0x4005                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[98:105], v[6:13] /*v[262:269]*/, v[58:65] /*v[314:321]*/, v[98:105] matrix_b_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[6:9] /*v[262:265]*/, v155 offset:64
	ds_load_b128 v[10:13] /*v[266:269]*/, v155 offset:80
	ds_load_b128 v[102:105] /*v[358:361]*/, v161 offset:80
	ds_load_b128 v[106:109] /*v[362:365]*/, v165 offset:64
	ds_load_b128 v[110:113] /*v[366:369]*/, v165 offset:80
	ds_load_b128 v[114:117] /*v[370:373]*/, v167 offset:64
	ds_load_b128 v[118:121] /*v[374:377]*/, v167 offset:80
	s_set_vgpr_msb 0x4005                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[90:97], v[14:21] /*v[270:277]*/, v[58:65] /*v[314:321]*/, v[90:97] matrix_b_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[14:17] /*v[270:273]*/, v163 offset:64
	ds_load_b128 v[18:21] /*v[274:277]*/, v163 offset:80
	ds_load_b128 v[122:125] /*v[378:381]*/, v171 offset:64
	ds_load_b128 v[126:129] /*v[382:385]*/, v171 offset:80
	ds_load_b128 v[130:133] /*v[386:389]*/, v173 offset:64
	ds_load_b128 v[134:137] /*v[390:393]*/, v173 offset:80
	ds_load_b128 v[138:141] /*v[394:397]*/, v175 offset:64
	s_set_vgpr_msb 0x4005                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[74:81], v[22:29] /*v[278:285]*/, v[58:65] /*v[314:321]*/, v[74:81] matrix_b_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[22:25] /*v[278:281]*/, v169 offset:64
	ds_load_b128 v[26:29] /*v[282:285]*/, v169 offset:80
	ds_load_b128 v[142:145] /*v[398:401]*/, v175 offset:80
	; sched_group_barrier mask(0x00000100) size(17) SyncID(0)
	s_set_vgpr_msb 0x4005                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[66:73], v[30:37] /*v[286:293]*/, v[58:65] /*v[314:321]*/, v[66:73] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[58:65], v[38:45] /*v[294:301]*/, v[58:65] /*v[314:321]*/, v[58:65] matrix_b_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[50:53] /*v[306:309]*/, v147 offset:46080
	ds_load_b128 v[54:57] /*v[310:313]*/, v147 offset:46096
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v147, v170, 1, s6
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[82:89], v[50:57] /*v[306:313]*/, v[58:65] /*v[314:321]*/, v[82:89] matrix_b_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[50:53] /*v[306:309]*/, v132 offset:43008
	ds_load_b128 v[54:57] /*v[310:313]*/, v132 offset:43024
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v132, v160, 1, s6
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[2:9], v[50:57] /*v[306:313]*/, v[58:65] /*v[314:321]*/, v[2:9] matrix_b_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[50:53] /*v[306:309]*/, v132 offset:39936
	ds_load_b128 v[54:57] /*v[310:313]*/, v132 offset:39952
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v132, v158, 1, s6
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[10:17], v[50:57] /*v[306:313]*/, v[58:65] /*v[314:321]*/, v[10:17] matrix_b_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[50:53] /*v[306:309]*/, v132 offset:36864
	ds_load_b128 v[54:57] /*v[310:313]*/, v132 offset:36880
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v132, v156, 1, s6
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[18:25], v[50:57] /*v[306:313]*/, v[58:65] /*v[314:321]*/, v[18:25] matrix_b_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[50:53] /*v[306:309]*/, v132 offset:33792
	ds_load_b128 v[54:57] /*v[310:313]*/, v132 offset:33808
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v132, v154, 1, s6
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[26:33], v[50:57] /*v[306:313]*/, v[58:65] /*v[314:321]*/, v[26:33] matrix_b_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[50:53] /*v[306:309]*/, v132 offset:30720
	ds_load_b128 v[54:57] /*v[310:313]*/, v132 offset:30736
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v132, v152, 1, s6
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[34:41], v[50:57] /*v[306:313]*/, v[58:65] /*v[314:321]*/, v[34:41] matrix_b_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[50:53] /*v[306:309]*/, v132 offset:27648
	ds_load_b128 v[54:57] /*v[310:313]*/, v132 offset:27664
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v132, v166, 1, s6
	ds_load_b128 v[238:241], v132 offset:64
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x32_bf16 v[42:49], v[50:57] /*v[306:313]*/, v[58:65] /*v[314:321]*/, v[42:49] matrix_b_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[50:53] /*v[306:309]*/, v145 offset:24592
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v145, v168, 1, s6
	ds_load_b128 v[242:245], v132 offset:80
	; sched_group_barrier mask(0x00000100) size(17) SyncID(0)
	ds_load_b128 v[246:249], v145 offset:64
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[50:57], v[46:53] /*v[302:309]*/, v[58:65] /*v[314:321]*/, v[50:57] matrix_b_reuse
	s_set_vgpr_msb 0x500                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	ds_load_b128 v[250:253], v145 offset:80
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[46:49] /*v[302:305]*/, v147 offset:64
	ds_load_b128 v[50:53] /*v[306:309]*/, v147 offset:80
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v132, v200, 1, s6
	v_lshl_add_u32 v145, v202, 1, s6
	v_lshl_add_u32 v147, v204, 1, s6
	v_lshl_add_u32 v149, v206, 1, s6
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[30:33] /*v[286:289]*/, v132 offset:128
	ds_load_b128 v[34:37] /*v[290:293]*/, v132 offset:144
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v132, v208, 1, s6
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[38:41] /*v[294:297]*/, v145 offset:128
	ds_load_b128 v[42:45] /*v[298:301]*/, v145 offset:144
	ds_load_b128 v[54:57] /*v[310:313]*/, v147 offset:128
	ds_load_b128 v[58:61] /*v[314:317]*/, v147 offset:144
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v145, v210, 1, s6
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[154:157] /*v[410:413]*/, v132 offset:128
	ds_load_b128 v[158:161] /*v[414:417]*/, v132 offset:144
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v132, v212, 1, s6
	v_lshl_add_u32 v147, v214, 1, s6
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[162:165] /*v[418:421]*/, v145 offset:128
	ds_load_b128 v[166:169] /*v[422:425]*/, v145 offset:144
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v145, v216, 1, s6
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[170:173] /*v[426:429]*/, v132 offset:128
	ds_load_b128 v[174:177] /*v[430:433]*/, v132 offset:144
	ds_load_b128 v[178:181] /*v[434:437]*/, v147 offset:128
	ds_load_b128 v[182:185] /*v[438:441]*/, v147 offset:144
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v132, v218, 1, s6
	v_lshl_add_u32 v147, v220, 1, s6
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[146:149] /*v[402:405]*/, v149 offset:128
	ds_load_b128 v[150:153] /*v[406:409]*/, v149 offset:144
	ds_load_b128 v[62:65] /*v[318:321]*/, v145 offset:128
	s_set_vgpr_msb 0x4004                   ;  msbs: dst=0 src0=0 src1=1 src2=0
	s_wait_dscnt 0x15
	v_wmma_f32_16x16x32_bf16 v[122:129], v[238:245], v[138:145] /*v[394:401]*/, v[122:129]
	; sched_group_barrier mask(0x00000100) size(17) SyncID(0)
	s_wait_dscnt 0x13
	v_wmma_f32_16x16x32_bf16 v[114:121], v[246:253], v[138:145] /*v[394:401]*/, v[114:121] matrix_b_reuse
	s_set_vgpr_msb 0x405                    ;  msbs: dst=0 src0=1 src1=1 src2=0
	s_wait_dscnt 0x11
	v_wmma_f32_16x16x32_bf16 v[106:113], v[46:53] /*v[302:309]*/, v[138:145] /*v[394:401]*/, v[106:113] matrix_b_reuse
	s_set_vgpr_msb 0x504                    ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[98:105], v[254:261], v[138:145] /*v[394:401]*/, v[98:105] matrix_b_reuse
	s_set_vgpr_msb 0x405                    ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[90:97], v[66:73] /*v[322:329]*/, v[138:145] /*v[394:401]*/, v[90:97] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[74:81], v[74:81] /*v[330:337]*/, v[138:145] /*v[394:401]*/, v[74:81] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[66:73], v[6:13] /*v[262:269]*/, v[138:145] /*v[394:401]*/, v[66:73] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[58:65], v[82:89] /*v[338:345]*/, v[138:145] /*v[394:401]*/, v[58:65] matrix_b_reuse
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[66:69] /*v[322:325]*/, v145 offset:144
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v145, v222, 1, s6
	ds_load_b128 v[238:241], v132 offset:128
	ds_load_b128 v[242:245], v132 offset:144
	ds_load_b128 v[246:249], v147 offset:128
	v_lshl_add_u32 v132, v224, 1, s6
	ds_load_b128 v[250:253], v147 offset:144
	v_lshl_add_u32 v147, v226, 1, s6
	ds_load_b128 v[254:257], v145 offset:128
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[2:5] /*v[258:261]*/, v145 offset:144
	ds_load_b128 v[6:9] /*v[262:265]*/, v132 offset:128
	ds_load_b128 v[10:13] /*v[266:269]*/, v132 offset:144
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v132, v228, 1, s6
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[46:49] /*v[302:305]*/, v147 offset:128
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v145, v230, 1, s6
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[50:53] /*v[306:309]*/, v147 offset:144
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u32 v147, v232, 1, s5
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[70:73] /*v[326:329]*/, v132 offset:128
	ds_load_b128 v[74:77] /*v[330:333]*/, v132 offset:144
	ds_load_b128 v[78:81] /*v[334:337]*/, v145 offset:128
	ds_load_b128 v[82:85] /*v[338:341]*/, v145 offset:144
	ds_load_b128 v[186:189] /*v[442:445]*/, v147 offset:128
	ds_load_b128 v[190:193] /*v[446:449]*/, v147 offset:144
	s_set_vgpr_msb 0x4005                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[50:57], v[90:97] /*v[346:353]*/, v[138:145] /*v[394:401]*/, v[50:57] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(17) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[42:49], v[98:105] /*v[354:361]*/, v[138:145] /*v[394:401]*/, v[42:49] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[34:41], v[14:21] /*v[270:277]*/, v[138:145] /*v[394:401]*/, v[34:41] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[26:33], v[106:113] /*v[362:369]*/, v[138:145] /*v[394:401]*/, v[26:33] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[18:25], v[114:121] /*v[370:377]*/, v[138:145] /*v[394:401]*/, v[18:25] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[10:17], v[22:29] /*v[278:285]*/, v[138:145] /*v[394:401]*/, v[10:17] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[2:9], v[122:129] /*v[378:385]*/, v[138:145] /*v[394:401]*/, v[2:9] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[82:89], v[130:137] /*v[386:393]*/, v[138:145] /*v[394:401]*/, v[82:89] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[122:129], v[30:37] /*v[286:293]*/, v[186:193] /*v[442:449]*/, v[122:129]
	; sched_group_barrier mask(0x00000100) size(17) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[114:121], v[38:45] /*v[294:301]*/, v[186:193] /*v[442:449]*/, v[114:121] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[106:113], v[54:61] /*v[310:317]*/, v[186:193] /*v[442:449]*/, v[106:113] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[98:105], v[146:153] /*v[402:409]*/, v[186:193] /*v[442:449]*/, v[98:105] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[90:97], v[154:161] /*v[410:417]*/, v[186:193] /*v[442:449]*/, v[90:97] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[74:81], v[162:169] /*v[418:425]*/, v[186:193] /*v[442:449]*/, v[74:81] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[66:73], v[170:177] /*v[426:433]*/, v[186:193] /*v[442:449]*/, v[66:73] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[58:65], v[178:185] /*v[434:441]*/, v[186:193] /*v[442:449]*/, v[58:65] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(17) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[50:57], v[62:69] /*v[318:325]*/, v[186:193] /*v[442:449]*/, v[50:57] matrix_b_reuse
	s_set_vgpr_msb 0x504                    ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[42:49], v[238:245], v[186:193] /*v[442:449]*/, v[42:49] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[34:41], v[246:253], v[186:193] /*v[442:449]*/, v[34:41] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[26:33], v[254:261], v[186:193] /*v[442:449]*/, v[26:33] matrix_b_reuse
	s_set_vgpr_msb 0x405                    ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_wmma_f32_16x16x32_bf16 v[18:25], v[6:13] /*v[262:269]*/, v[186:193] /*v[442:449]*/, v[18:25] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[10:17], v[46:53] /*v[302:309]*/, v[186:193] /*v[442:449]*/, v[10:17] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[2:9], v[70:77] /*v[326:333]*/, v[186:193] /*v[442:449]*/, v[2:9] matrix_b_reuse
	v_wmma_f32_16x16x32_bf16 v[82:89], v[78:85] /*v[334:341]*/, v[186:193] /*v[442:449]*/, v[82:89] matrix_b_reuse
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_wait_dscnt 0
	;;#ASMEND
	s_set_vgpr_msb 0x500                    ;  msbs: dst=0 src0=0 src1=0 src2=0
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
	v_cmpx_eq_u32_e32 1, v137
	s_cbranch_execz .LBB0_39
; %bb.38:                               ;   in Loop: Header=BB0_19 Depth=1
	s_cmp_lg_u32 s5, 0
	s_mul_i32 s10, s6, 0x60
	s_cselect_b32 s13, s40, s43
	s_cmp_gt_i32 s34, 0
	s_mov_b32 s36, s34
	s_cselect_b32 s17, -1, 0
	s_lshl_b64 s[14:15], s[10:11], 1
	s_mov_b32 s18, s11
	s_add_nc_u64 s[14:15], s[20:21], s[14:15]
	s_delay_alu instid0(SALU_CYCLE_1)
	v_dual_mov_b32 v145, s13 :: v_dual_mov_b32 v238, s14
	s_and_b32 s10, s15, 0x1ffffff
	s_and_b32 s15, s29, s17
	s_bitset1_b32 s10, 31
	v_cndmask_b32_e64 v132, 0, 1, s15
	v_mov_b32_e32 v147, s10
	v_readfirstlane_b32 s57, v145
	v_readfirstlane_b32 s58, v238
	s_lshr_b64 s[14:15], s[36:37], 16
	v_readfirstlane_b32 s56, v132
	v_readfirstlane_b32 s59, v147
	s_lshl_b32 s13, s34, 16
	s_mov_b32 s15, s49
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
	s_cselect_b32 s5, s45, 0
	s_cmp_gt_i32 s34, 0
	s_cselect_b32 s14, -1, 0
	s_lshl_b64 s[6:7], s[10:11], 1
	s_and_b32 s10, s3, s14
	s_add_nc_u64 s[6:7], s[24:25], s[6:7]
	v_cndmask_b32_e64 v132, 0, 1, s10
	s_and_b32 s7, s7, 0x1ffffff
	v_dual_mov_b32 v145, s5 :: v_dual_mov_b32 v238, s6
	s_bitset1_b32 s7, 31
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_readfirstlane_b32 s52, v132
	v_mov_b32_e32 v147, s7
	v_readfirstlane_b32 s53, v145
	v_readfirstlane_b32 s54, v238
	s_lshr_b64 s[6:7], s[34:35], 16
	s_lshl_b32 s5, s34, 16
	v_readfirstlane_b32 s55, v147
	s_mov_b32 s7, s50
	s_mov_b32 s10, s11
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	tensor_load_to_lds s[52:55], s[4:11]
	s_or_b32 exec_lo, exec_lo, s13
	s_and_not1_b32 vcc_lo, exec_lo, s38
	s_cbranch_vccz .LBB0_34
	s_branch .LBB0_35
.LBB0_41:
	s_wait_tensorcnt 0x0
	;;#ASMSTART
	s_wait_dscnt 0
	;;#ASMEND
	s_and_b32 vcc_lo, exec_lo, s38
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_vccz .LBB0_43
; %bb.42:
	v_and_or_b32 v1, v0, 15, v135
	v_lshrrev_b32_e32 v130, 1, v0
	v_cvt_pk_bf16_f32 v129, v128, v129
	v_cvt_pk_bf16_f32 v128, v126, v127
	v_cvt_pk_bf16_f32 v126, v122, v123
	v_lshlrev_b32_e32 v1, 8, v1
	v_cvt_pk_bf16_f32 v127, v124, v125
	v_cvt_pk_bf16_f32 v97, v96, v97
	v_cvt_pk_bf16_f32 v96, v94, v95
	v_cvt_pk_bf16_f32 v95, v92, v93
	v_and_or_b32 v122, v130, 8, v1
	v_lshrrev_b32_e32 v1, 3, v1
	v_cvt_pk_bf16_f32 v94, v90, v91
	v_cvt_pk_bf16_f32 v41, v40, v41
	v_cvt_pk_bf16_f32 v40, v38, v39
	v_cvt_pk_bf16_f32 v39, v36, v37
	v_lshl_add_u32 v1, v122, 1, v1
	v_cvt_pk_bf16_f32 v38, v34, v35
	v_cvt_pk_bf16_f32 v121, v120, v121
	v_cvt_pk_bf16_f32 v120, v118, v119
	v_cvt_pk_bf16_f32 v119, v116, v117
	v_cvt_pk_bf16_f32 v118, v114, v115
	v_cvt_pk_bf16_f32 v81, v80, v81
	v_cvt_pk_bf16_f32 v80, v78, v79
	v_cvt_pk_bf16_f32 v79, v76, v77
	v_cvt_pk_bf16_f32 v78, v74, v75
	v_cvt_pk_bf16_f32 v33, v32, v33
	v_cvt_pk_bf16_f32 v32, v30, v31
	v_cvt_pk_bf16_f32 v31, v28, v29
	v_cvt_pk_bf16_f32 v30, v26, v27
	v_cvt_pk_bf16_f32 v113, v112, v113
	v_cvt_pk_bf16_f32 v112, v110, v111
	v_cvt_pk_bf16_f32 v111, v108, v109
	v_cvt_pk_bf16_f32 v110, v106, v107
	v_cvt_pk_bf16_f32 v73, v72, v73
	v_cvt_pk_bf16_f32 v72, v70, v71
	v_cvt_pk_bf16_f32 v71, v68, v69
	v_cvt_pk_bf16_f32 v70, v66, v67
	v_cvt_pk_bf16_f32 v25, v24, v25
	v_cvt_pk_bf16_f32 v24, v22, v23
	v_cvt_pk_bf16_f32 v23, v20, v21
	v_cvt_pk_bf16_f32 v22, v18, v19
	v_cvt_pk_bf16_f32 v105, v104, v105
	v_cvt_pk_bf16_f32 v104, v102, v103
	v_cvt_pk_bf16_f32 v103, v100, v101
	v_cvt_pk_bf16_f32 v102, v98, v99
	v_cvt_pk_bf16_f32 v65, v64, v65
	v_cvt_pk_bf16_f32 v64, v62, v63
	v_cvt_pk_bf16_f32 v63, v60, v61
	v_cvt_pk_bf16_f32 v62, v58, v59
	v_cvt_pk_bf16_f32 v17, v16, v17
	v_cvt_pk_bf16_f32 v16, v14, v15
	v_cvt_pk_bf16_f32 v15, v12, v13
	v_cvt_pk_bf16_f32 v14, v10, v11
	v_cvt_pk_bf16_f32 v57, v56, v57
	v_cvt_pk_bf16_f32 v56, v54, v55
	v_cvt_pk_bf16_f32 v55, v52, v53
	v_cvt_pk_bf16_f32 v54, v50, v51
	v_cvt_pk_bf16_f32 v9, v8, v9
	v_cvt_pk_bf16_f32 v8, v6, v7
	v_cvt_pk_bf16_f32 v7, v4, v5
	v_cvt_pk_bf16_f32 v6, v2, v3
	ds_store_b128 v1, v[126:129]
	ds_store_b128 v1, v[118:121] offset:32
	ds_store_b128 v1, v[110:113] offset:64
	ds_store_b128 v1, v[102:105] offset:96
	v_cvt_pk_bf16_f32 v49, v48, v49
	v_cvt_pk_bf16_f32 v48, v46, v47
	v_cvt_pk_bf16_f32 v47, v44, v45
	v_cvt_pk_bf16_f32 v46, v42, v43
	ds_store_b128 v1, v[94:97] offset:128
	ds_store_b128 v1, v[78:81] offset:160
	ds_store_b128 v1, v[70:73] offset:192
	ds_store_b128 v1, v[62:65] offset:224
	ds_store_b128 v1, v[54:57] offset:272
	ds_store_b128 v1, v[46:49] offset:304
	v_cvt_pk_bf16_f32 v5, v88, v89
	v_cvt_pk_bf16_f32 v4, v86, v87
	v_cvt_pk_bf16_f32 v3, v84, v85
	v_cvt_pk_bf16_f32 v2, v82, v83
	ds_store_b128 v1, v[38:41] offset:336
	ds_store_b128 v1, v[30:33] offset:368
	ds_store_b128 v1, v[22:25] offset:400
	ds_store_b128 v1, v[14:17] offset:432
	ds_store_b128 v1, v[6:9] offset:464
	ds_store_b128 v1, v[2:5] offset:496
.LBB0_43:
	v_cmp_ne_u32_e32 vcc_lo, 1, v131
	;;#ASMSTART
	s_wait_dscnt 0
	;;#ASMEND
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_vccnz .LBB0_54
; %bb.44:
	s_wait_kmcnt 0x0
	s_mul_i32 s14, s37, s35
	s_mov_b32 s0, exec_lo
	v_cmpx_gt_i32_e64 s14, v0
	s_cbranch_execz .LBB0_54
; %bb.45:
	s_ashr_i32 s3, s2, 31
	v_nop
	v_xad_u32 v2, v0, -1, s14
	s_lshl_b64 s[0:1], s[2:3], 1
	s_ashr_i32 s29, s28, 31
	s_add_nc_u64 s[4:5], s[22:23], s[0:1]
	s_mov_b32 s0, 0
                                        ; implicit-def: $vgpr1
                                        ; implicit-def: $vgpr6
                                        ; implicit-def: $sgpr12_sgpr13
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_u32_e32 0x2ff, v2
	s_xor_b32 s3, exec_lo, s1
	s_cbranch_execnz .LBB0_48
; %bb.46:
	s_or_saveexec_b32 s1, s3
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
	s_cbranch_execnz .LBB0_49
; %bb.50:
	s_or_b32 exec_lo, exec_lo, s23
	v_cmp_ne_u32_e32 vcc_lo, v8, v9
	v_lshl_or_b32 v0, v9, 8, v0
	v_dual_mov_b32 v6, s15 :: v_dual_mov_b32 v1, s22
	s_and_b32 s0, vcc_lo, exec_lo
	s_or_saveexec_b32 s1, s3
	v_mov_b64_e32 v[2:3], s[12:13]
	s_xor_b32 exec_lo, exec_lo, s1
	s_cbranch_execz .LBB0_47
.LBB0_51:
	s_abs_i32 s6, s31
	s_ashr_i32 s7, s31, 31
	s_cvt_f32_u32 s2, s6
	s_sub_co_i32 s3, 0, s6
	v_mov_b32_e32 v6, s6
	s_or_b32 s0, s0, exec_lo
	v_rcp_iflag_f32_e32 v1, s2
	v_nop
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_3)
	v_readfirstlane_b32 s2, v1
	v_mov_b32_e32 v1, s7
	s_mul_f32 s2, s2, 0x4f7ffffe
	s_cvt_u32_f32 s2, s2
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_i32 s3, s3, s2
	s_mul_hi_u32 s8, s2, s3
	s_mov_b32 s3, 0
	s_add_co_i32 s2, s2, s8
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_mov_b64_e32 v[2:3], s[2:3]
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
	v_cmp_le_i32_e32 vcc_lo, s14, v0
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
	.size	bm256_bn128_bk096_wm1_wn8_mc1, .Lfunc_end0-bm256_bn128_bk096_wm1_wn8_mc1
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel bm256_bn128_bk096_wm1_wn8_mc1
		.amdhsa_group_segment_fixed_size 156672
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
		.amdhsa_next_free_vgpr 450
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
		.amdhsa_inst_pref_size 73
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text.bm256_bn128_bk096_wm1_wn8_mc1,"axG",@progbits,bm256_bn128_bk096_wm1_wn8_mc1,comdat
                                        ; -- End function
	.set .Lbm256_bn128_bk096_wm1_wn8_mc1.num_vgpr, 450
	.set .Lbm256_bn128_bk096_wm1_wn8_mc1.num_agpr, 0
	.set .Lbm256_bn128_bk096_wm1_wn8_mc1.numbered_sgpr, 60
	.set .Lbm256_bn128_bk096_wm1_wn8_mc1.num_named_barrier, 0
	.set .Lbm256_bn128_bk096_wm1_wn8_mc1.private_seg_size, 0
	.set .Lbm256_bn128_bk096_wm1_wn8_mc1.uses_vcc, 1
	.set .Lbm256_bn128_bk096_wm1_wn8_mc1.uses_flat_scratch, 1
	.set .Lbm256_bn128_bk096_wm1_wn8_mc1.has_dyn_sized_stack, 0
	.set .Lbm256_bn128_bk096_wm1_wn8_mc1.has_recursion, 0
	.set .Lbm256_bn128_bk096_wm1_wn8_mc1.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 9284
; TotalNumSgprs: 62
; NumVgprs: 450
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 156672 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 28
; NumSGPRsForWavesPerEU: 62
; NumVGPRsForWavesPerEU: 450
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
	.type	__hip_cuid_f3d755ca12bb891f,@object ; @__hip_cuid_f3d755ca12bb891f
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_f3d755ca12bb891f
__hip_cuid_f3d755ca12bb891f:
	.byte	0                               ; 0x0
	.size	__hip_cuid_f3d755ca12bb891f, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git aa451e1fe6a793394d6733051b1778633063ae96+PATCHED:d17c5aa0e3ea29cde402f58f27e39b6034effa27)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_f3d755ca12bb891f
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
    macrotile: [256, 128, 96]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, One]
  MatrixInstruction: [16, 16, 32, 1]
  EnableMatrixInstruction: True
  MIWaveTile: [8, 2]
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
    .group_segment_fixed_size: 156672
    .kernarg_segment_align: 8
    .kernarg_segment_size: 132
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           bm256_bn128_bk096_wm1_wn8_mc1
    .private_segment_fixed_size: 0
    .sgpr_count:     62
    .sgpr_spill_count: 0
    .symbol:         bm256_bn128_bk096_wm1_wn8_mc1.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     450
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
