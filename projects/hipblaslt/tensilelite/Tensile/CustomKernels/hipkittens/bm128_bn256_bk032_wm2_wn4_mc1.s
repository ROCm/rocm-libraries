	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.section	.text.bm128_bn256_bk032_wm2_wn4_mc1,"axG",@progbits,bm128_bn256_bk032_wm2_wn4_mc1,comdat
	.protected	bm128_bn256_bk032_wm2_wn4_mc1 ; -- Begin function bm128_bn256_bk032_wm2_wn4_mc1
	.globl	bm128_bn256_bk032_wm2_wn4_mc1
	.p2align	8
	.type	bm128_bn256_bk032_wm2_wn4_mc1,@function
bm128_bn256_bk032_wm2_wn4_mc1: ; @bm128_bn256_bk032_wm2_wn4_mc1
; %bb.0:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1 ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_load_b96 s[28:30], s[0:1], 0x78 nv
	s_mov_b64 s[2:3], src_shared_base
	s_movk_i32 s2, 0x2200
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_and_b64 s[2:3], s[2:3], 12
	s_sub_co_i32 s4, 16, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_lshr_b32 s4, s4, 2
	s_cmp_lg_u64 s[2:3], 0
	s_cselect_b32 s7, s4, 0
	s_and_b32 s2, ttmp6, 15
	s_bfe_u32 s3, ttmp6, 0x40004
	s_lshl2_add_u32 s17, ttmp9, s2
	s_lshl2_add_u32 s6, ttmp7, s3
	s_lshl_b32 s34, s17, 7
	s_wait_kmcnt 0x0
	s_add_co_i32 s2, s28, 0x7f
	s_add_co_i32 s3, s29, 0xff
	s_sub_co_i32 s4, s28, s34
	s_ashr_i32 s5, s2, 31
	s_ashr_i32 s8, s3, 31
	s_min_i32 s31, s4, 0x80
	s_lshr_b32 s4, s5, 25
	s_lshr_b32 s5, s8, 24
	s_add_co_i32 s2, s2, s4
	s_add_co_i32 s3, s3, s5
	s_ashr_i32 s8, s2, 7
	s_ashr_i32 s10, s3, 8
	s_cmp_lt_i32 s17, s8
	s_mov_b32 s9, s30
	s_cselect_b32 s35, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_and_b32 s2, s35, exec_lo
	s_cselect_b32 s37, s31, 0
	s_lshl_b32 s33, s6, 8
	s_sub_co_i32 s2, s29, s33
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_min_i32 s2, s2, 0x100
	s_cmp_lt_i32 s6, s10
	s_cselect_b32 s29, -1, 0
	s_and_b32 s3, s29, exec_lo
	s_cselect_b32 s3, s2, 0
	s_add_co_i32 s12, s30, 31
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 2, 1), 1
	s_min_i32 s2, s30, 32
	s_cmp_gt_i32 s12, 31
	s_cselect_b32 s13, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)
	s_and_b32 s4, s13, exec_lo
	s_cselect_b32 s36, s2, 0
	s_cmp_lt_i32 s37, 0x80
	s_mov_b32 s2, -1
	s_cselect_b32 s42, -1, 0
	s_and_b32 vcc_lo, exec_lo, s42
	s_cbranch_vccnz .LBB0_2
; %bb.1:
	s_cmp_lt_i32 s3, 0x100
	s_cselect_b32 s2, -1, 0
	s_cmp_lt_i32 s36, 32
	s_cselect_b32 s4, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 s2, s4, s2
.LBB0_2:
	v_sub_nc_u32_e32 v131, 0x87f, v0
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB0_12
; %bb.3:
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_dual_lshrrev_b32 v2, 8, v131 :: v_dual_lshlrev_b32 v3, 2, v0
	s_mov_b32 s4, 0
	s_mov_b32 s11, 0
	v_dual_mov_b32 v4, 0 :: v_dual_add_nc_u32 v5, 2, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mov_b32 v1, v2 :: v_dual_mov_b32 v6, v3
	v_and_b32_e32 v5, 30, v5
	s_branch .LBB0_5
.LBB0_4:                                ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
	s_add_co_i32 s4, s4, 2
	v_add_nc_u32_e32 v6, 0x800, v6
	v_cmp_eq_u32_e32 vcc_lo, s4, v5
	s_or_b32 s11, vcc_lo, s11
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s11
	s_cbranch_execz .LBB0_9
.LBB0_5:                                ; =>This Inner Loop Header: Depth=1
	s_mov_b32 s5, s4
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b64 s[14:15], s[4:5], 0x100000000
	s_mov_b32 s5, exec_lo
	v_cmp_le_u32_e32 vcc_lo, s15, v1
	v_cmpx_le_u32_e64 s14, v2
; %bb.6:                                ;   in Loop: Header=BB0_5 Depth=1
	ds_store_b32 v6, v4
; %bb.7:                                ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s5
	s_and_saveexec_b32 s2, vcc_lo
	s_cbranch_execz .LBB0_4
; %bb.8:                                ;   in Loop: Header=BB0_5 Depth=1
	ds_store_b32 v6, v4 offset:1024
	s_branch .LBB0_4
.LBB0_9:
	s_or_b32 exec_lo, exec_lo, s11
	v_lshl_add_u32 v1, s7, 2, v3
	v_or_b32_e32 v2, 0xffffff00, v0
	v_mov_b32_e32 v3, 0
	s_mov_b32 s2, 0
.LBB0_10:                               ; =>This Inner Loop Header: Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)
	v_add_nc_u32_e32 v2, 0x100, v2
	ds_store_b32 v1, v3 offset:8704
	v_add_nc_u32_e32 v1, 0x400, v1
	v_cmp_lt_u32_e32 vcc_lo, 0xfff, v2
	s_or_b32 s2, vcc_lo, s2
	s_and_not1_b32 exec_lo, exec_lo, s2
	s_cbranch_execnz .LBB0_10
; %bb.11:
	s_or_b32 exec_lo, exec_lo, s2
	s_barrier_signal -1
	s_barrier_wait -1
.LBB0_12:
	s_clause 0x2
	s_load_b64 s[14:15], s[0:1], 0x0 nv
	s_load_b128 s[24:27], s[0:1], 0x20 nv
	s_load_b128 s[20:23], s[0:1], 0x48 nv
	s_wait_xcnt 0x0
	s_lshl_b32 s1, s6, 2
	v_lshrrev_b32_e32 v141, 5, v0
	s_lshl_b32 s16, s7, 2
	s_add_co_i32 s10, s10, -1
	s_and_b32 s1, s1, 12
	s_mov_b64 s[38:39], src_shared_base
	s_or_b32 s38, s16, 0x2200
	s_add_co_i32 s18, s8, -1
	s_min_i32 s0, s6, s10
	s_and_b32 s19, s17, 3
	s_lshl_b32 s1, 15, s1
	s_mov_b32 s2, exec_lo
	v_cmpx_lt_i32_e32 0, v141
	s_xor_b32 s40, exec_lo, s2
	s_cbranch_execz .LBB0_16
; %bb.13:
	s_mov_b32 s41, exec_lo
	v_cmpx_eq_u32_e32 1, v141
	s_cbranch_execz .LBB0_15
; %bb.14:
	s_cmp_gt_i32 s36, 0
	s_mov_b32 s10, 0
	s_cselect_b32 s2, -1, 0
	s_lshl_b32 s4, s0, 8
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
	s_or_b32 s7, s2, 0x200000
	s_movk_i32 s8, 0x100
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
	s_lshl_b32 s4, s17, 7
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
	s_or_b32 s7, s8, 0x200000
	v_readfirstlane_b32 s46, v4
	v_readfirstlane_b32 s47, v3
	s_movk_i32 s8, 0x80
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
	v_dual_lshlrev_b32 v1, 4, v141 :: v_dual_lshlrev_b32 v2, 6, v141
	v_mov_b32_e32 v17, 0
	s_and_b32 s43, s35, s29
	s_and_not1_b32 vcc_lo, exec_lo, s13
	s_delay_alu instid0(VALU_DEP_2)
	v_and_b32_e32 v137, 64, v1
	v_and_b32_e32 v139, 0xc0, v2
	v_dual_mov_b32 v16, v17 :: v_dual_mov_b32 v15, v17
	v_dual_mov_b32 v14, v17 :: v_dual_mov_b32 v13, v17
	v_dual_mov_b32 v12, v17 :: v_dual_mov_b32 v11, v17
	v_dual_mov_b32 v10, v17 :: v_dual_mov_b32 v9, v17
	v_dual_mov_b32 v8, v17 :: v_dual_mov_b32 v7, v17
	v_dual_mov_b32 v6, v17 :: v_dual_mov_b32 v5, v17
	v_dual_mov_b32 v4, v17 :: v_dual_mov_b32 v3, v17
	v_dual_mov_b32 v2, v17 :: v_dual_mov_b32 v25, v17
	v_dual_mov_b32 v24, v17 :: v_dual_mov_b32 v23, v17
	v_dual_mov_b32 v22, v17 :: v_dual_mov_b32 v21, v17
	v_dual_mov_b32 v20, v17 :: v_dual_mov_b32 v19, v17
	v_dual_mov_b32 v18, v17 :: v_dual_mov_b32 v33, v17
	v_dual_mov_b32 v32, v17 :: v_dual_mov_b32 v31, v17
	v_dual_mov_b32 v30, v17 :: v_dual_mov_b32 v29, v17
	v_dual_mov_b32 v28, v17 :: v_dual_mov_b32 v27, v17
	v_dual_mov_b32 v26, v17 :: v_dual_mov_b32 v49, v17
	v_dual_mov_b32 v48, v17 :: v_dual_mov_b32 v47, v17
	v_dual_mov_b32 v46, v17 :: v_dual_mov_b32 v45, v17
	v_dual_mov_b32 v44, v17 :: v_dual_mov_b32 v43, v17
	v_dual_mov_b32 v42, v17 :: v_dual_mov_b32 v41, v17
	v_dual_mov_b32 v40, v17 :: v_dual_mov_b32 v39, v17
	v_dual_mov_b32 v38, v17 :: v_dual_mov_b32 v37, v17
	v_dual_mov_b32 v36, v17 :: v_dual_mov_b32 v35, v17
	v_dual_mov_b32 v34, v17 :: v_dual_mov_b32 v57, v17
	v_dual_mov_b32 v56, v17 :: v_dual_mov_b32 v55, v17
	v_dual_mov_b32 v54, v17 :: v_dual_mov_b32 v53, v17
	v_dual_mov_b32 v52, v17 :: v_dual_mov_b32 v51, v17
	v_dual_mov_b32 v50, v17 :: v_dual_mov_b32 v89, v17
	v_dual_mov_b32 v88, v17 :: v_dual_mov_b32 v87, v17
	v_dual_mov_b32 v86, v17 :: v_dual_mov_b32 v85, v17
	v_dual_mov_b32 v84, v17 :: v_dual_mov_b32 v83, v17
	v_dual_mov_b32 v82, v17 :: v_dual_mov_b32 v97, v17
	v_dual_mov_b32 v96, v17 :: v_dual_mov_b32 v95, v17
	v_dual_mov_b32 v94, v17 :: v_dual_mov_b32 v93, v17
	v_dual_mov_b32 v92, v17 :: v_dual_mov_b32 v91, v17
	v_dual_mov_b32 v90, v17 :: v_dual_mov_b32 v105, v17
	v_dual_mov_b32 v104, v17 :: v_dual_mov_b32 v103, v17
	v_dual_mov_b32 v102, v17 :: v_dual_mov_b32 v101, v17
	v_dual_mov_b32 v100, v17 :: v_dual_mov_b32 v99, v17
	v_dual_mov_b32 v98, v17 :: v_dual_mov_b32 v113, v17
	v_dual_mov_b32 v112, v17 :: v_dual_mov_b32 v111, v17
	v_dual_mov_b32 v110, v17 :: v_dual_mov_b32 v109, v17
	v_dual_mov_b32 v108, v17 :: v_dual_mov_b32 v107, v17
	v_dual_mov_b32 v106, v17 :: v_dual_mov_b32 v121, v17
	v_dual_mov_b32 v120, v17 :: v_dual_mov_b32 v119, v17
	v_dual_mov_b32 v118, v17 :: v_dual_mov_b32 v117, v17
	v_dual_mov_b32 v116, v17 :: v_dual_mov_b32 v115, v17
	v_dual_mov_b32 v114, v17 :: v_dual_mov_b32 v129, v17
	v_dual_mov_b32 v128, v17 :: v_dual_mov_b32 v127, v17
	v_dual_mov_b32 v126, v17 :: v_dual_mov_b32 v125, v17
	v_dual_mov_b32 v124, v17 :: v_dual_mov_b32 v123, v17
	v_dual_mov_b32 v122, v17 :: v_dual_mov_b32 v65, v17
	v_dual_mov_b32 v64, v17 :: v_dual_mov_b32 v63, v17
	v_dual_mov_b32 v62, v17 :: v_dual_mov_b32 v61, v17
	v_dual_mov_b32 v60, v17 :: v_dual_mov_b32 v59, v17
	v_dual_mov_b32 v58, v17 :: v_dual_mov_b32 v73, v17
	v_dual_mov_b32 v72, v17 :: v_dual_mov_b32 v71, v17
	v_dual_mov_b32 v70, v17 :: v_dual_mov_b32 v69, v17
	v_dual_mov_b32 v68, v17 :: v_dual_mov_b32 v67, v17
	v_dual_mov_b32 v66, v17 :: v_dual_mov_b32 v81, v17
	v_dual_mov_b32 v80, v17 :: v_dual_mov_b32 v79, v17
	v_dual_mov_b32 v78, v17 :: v_dual_mov_b32 v77, v17
	v_dual_mov_b32 v76, v17 :: v_dual_mov_b32 v75, v17
	v_mov_b32_e32 v74, v17
	v_cndmask_b32_e64 v135, 0, 1, s43
	s_barrier_wait -1
	s_barrier_wait -3
	s_cbranch_vccnz .LBB0_46
; %bb.21:
	v_dual_lshlrev_b32 v4, 5, v0 :: v_dual_bitop2_b32 v3, 16, v0 bitop3:0x40
	s_mov_b64 s[4:5], src_shared_base
	s_or_b32 s6, s16, 0x6600
	s_mov_b32 s7, s5
	v_dual_lshlrev_b32 v1, 5, v139 :: v_dual_lshlrev_b32 v2, 5, v137
	v_and_b32_e32 v4, 0x1e0, v4
	s_and_b64 s[6:7], s[6:7], 15
	s_mov_b32 s11, 0
	s_sub_co_i32 s4, 16, s6
	v_lshrrev_b32_e32 v136, 8, v131
	s_lshr_b32 s4, s4, 2
	v_or3_b32 v2, v2, v3, v4
	s_cmp_lg_u64 s[6:7], 0
	v_or3_b32 v3, v1, v3, v4
	s_cselect_b32 s4, s4, 0
	v_lshl_or_b32 v150, v0, 2, 0x4000
	s_lshl2_add_u32 s6, s4, s16
	v_or_b32_e32 v6, 0x200, v2
	v_or_b32_e32 v7, 0x400, v2
	v_lshrrev_b32_e32 v5, 4, v2
	s_add_co_i32 s4, s6, 0x8800
	s_add_co_i32 s45, s6, 0x6600
	s_and_b32 s10, s4, 15
	v_dual_mov_b32 v133, 0 :: v_dual_lshrrev_b32 v6, 4, v6
	v_lshrrev_b32_e32 v7, 4, v7
	v_and_b32_e32 v5, 0x98, v5
	s_sub_co_i32 s7, 16, s10
	s_delay_alu instid0(VALU_DEP_3)
	v_and_b32_e32 v4, 0xf8, v6
	s_lshr_b32 s6, s7, 2
	s_cmp_lg_u64 s[10:11], 0
	v_dual_add_nc_u32 v130, v5, v2 :: v_dual_lshrrev_b32 v6, 4, v3
	v_and_b32_e32 v5, 0xf8, v7
	v_or_b32_e32 v1, 0x600, v2
	v_or_b32_e32 v7, 0x200, v3
	v_or_b32_e32 v8, 0x400, v3
	v_or_b32_e32 v9, 0x600, v3
	s_cselect_b32 s6, s6, 0
	s_ashr_i32 s7, s12, 31
	v_dual_lshrrev_b32 v1, 4, v1 :: v_dual_lshrrev_b32 v7, 4, v7
	s_lshr_b32 s7, s7, 27
	v_and_b32_e32 v6, 0x198, v6
	s_add_co_i32 s12, s12, s7
	v_dual_lshrrev_b32 v8, 4, v8 :: v_dual_lshrrev_b32 v9, 4, v9
	s_lshl_b32 s10, s6, 2
	s_ashr_i32 s47, s12, 5
	s_cmp_lt_i32 s3, 0x100
	v_and_b32_e32 v10, 0xf8, v1
	s_cselect_b32 s48, -1, 0
	s_lshl_b32 s6, s0, 8
	s_lshl_b32 s18, s17, 7
	v_dual_add_nc_u32 v134, v6, v3 :: v_dual_add_nc_u32 v1, 2, v136
	v_and_b32_e32 v6, 0x1f8, v7
	v_and_b32_e32 v7, 0x1f8, v8
	v_and_b32_e32 v8, 0x1f8, v9
	s_ashr_i32 s7, s6, 31
	s_ashr_i32 s19, s18, 31
	s_wait_kmcnt 0x0
	s_bfe_i64 s[20:21], s[20:21], 0x200000
	s_bfe_i64 s[24:25], s[24:25], 0x200000
	s_mul_u64 s[6:7], s[20:21], s[6:7]
	s_mul_u64 s[18:19], s[24:25], s[18:19]
	v_cmp_eq_u32_e64 s0, 0, v141
	v_and_b32_e32 v143, 30, v1
	v_or_b32_e32 v1, 0x100, v0
	v_dual_add_nc_u32 v138, v4, v2 :: v_dual_add_nc_u32 v140, v5, v2
	v_dual_add_nc_u32 v142, v10, v2 :: v_dual_add_nc_u32 v146, v7, v3
	v_dual_add_nc_u32 v144, v6, v3 :: v_dual_add_nc_u32 v148, v8, v3
	v_dual_mov_b32 v151, v133 :: v_dual_mov_b32 v10, v133
	v_dual_mov_b32 v11, v133 :: v_dual_mov_b32 v12, v133
	v_dual_mov_b32 v13, v133 :: v_dual_mov_b32 v14, v133
	v_dual_mov_b32 v15, v133 :: v_dual_mov_b32 v16, v133
	v_dual_mov_b32 v17, v133 :: v_dual_mov_b32 v2, v133
	v_dual_mov_b32 v3, v133 :: v_dual_mov_b32 v4, v133
	v_dual_mov_b32 v5, v133 :: v_dual_mov_b32 v6, v133
	v_dual_mov_b32 v7, v133 :: v_dual_mov_b32 v8, v133
	v_dual_mov_b32 v9, v133 :: v_dual_mov_b32 v18, v133
	v_dual_mov_b32 v19, v133 :: v_dual_mov_b32 v20, v133
	v_dual_mov_b32 v21, v133 :: v_dual_mov_b32 v22, v133
	v_dual_mov_b32 v23, v133 :: v_dual_mov_b32 v24, v133
	v_dual_mov_b32 v25, v133 :: v_dual_mov_b32 v26, v133
	v_dual_mov_b32 v27, v133 :: v_dual_mov_b32 v28, v133
	v_dual_mov_b32 v29, v133 :: v_dual_mov_b32 v30, v133
	v_dual_mov_b32 v31, v133 :: v_dual_mov_b32 v32, v133
	v_dual_mov_b32 v33, v133 :: v_dual_mov_b32 v42, v133
	v_dual_mov_b32 v43, v133 :: v_dual_mov_b32 v44, v133
	v_dual_mov_b32 v45, v133 :: v_dual_mov_b32 v46, v133
	v_dual_mov_b32 v47, v133 :: v_dual_mov_b32 v48, v133
	v_dual_mov_b32 v49, v133 :: v_dual_mov_b32 v34, v133
	v_dual_mov_b32 v35, v133 :: v_dual_mov_b32 v36, v133
	v_dual_mov_b32 v37, v133 :: v_dual_mov_b32 v38, v133
	v_dual_mov_b32 v39, v133 :: v_dual_mov_b32 v40, v133
	v_dual_mov_b32 v41, v133 :: v_dual_mov_b32 v50, v133
	v_dual_mov_b32 v51, v133 :: v_dual_mov_b32 v52, v133
	v_dual_mov_b32 v53, v133 :: v_dual_mov_b32 v54, v133
	v_dual_mov_b32 v55, v133 :: v_dual_mov_b32 v56, v133
	v_dual_mov_b32 v57, v133 :: v_dual_mov_b32 v82, v133
	v_dual_mov_b32 v83, v133 :: v_dual_mov_b32 v84, v133
	v_dual_mov_b32 v85, v133 :: v_dual_mov_b32 v86, v133
	v_dual_mov_b32 v87, v133 :: v_dual_mov_b32 v88, v133
	v_dual_mov_b32 v89, v133 :: v_dual_mov_b32 v90, v133
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
	v_dual_mov_b32 v129, v133 :: v_dual_mov_b32 v58, v133
	v_dual_mov_b32 v59, v133 :: v_dual_mov_b32 v60, v133
	v_dual_mov_b32 v61, v133 :: v_dual_mov_b32 v62, v133
	v_dual_mov_b32 v63, v133 :: v_dual_mov_b32 v64, v133
	v_dual_mov_b32 v65, v133 :: v_dual_mov_b32 v66, v133
	v_dual_mov_b32 v67, v133 :: v_dual_mov_b32 v68, v133
	v_dual_mov_b32 v69, v133 :: v_dual_mov_b32 v70, v133
	v_dual_mov_b32 v71, v133 :: v_dual_mov_b32 v72, v133
	v_dual_mov_b32 v73, v133 :: v_dual_mov_b32 v74, v133
	v_dual_mov_b32 v75, v133 :: v_dual_mov_b32 v76, v133
	v_dual_mov_b32 v77, v133 :: v_dual_mov_b32 v78, v133
	v_dual_mov_b32 v79, v133 :: v_dual_mov_b32 v80, v133
	v_dual_mov_b32 v81, v133 :: v_dual_mov_b32 v131, v136
	v_or_b32_e32 v145, 0xf00, v0
	s_lshr_b32 s49, s3, 16
	s_lshr_b32 s50, s37, 16
	s_lshl_b64 s[6:7], s[6:7], 1
	s_lshl_b64 s[18:19], s[18:19], 1
	s_mov_b32 s44, s39
	s_movk_i32 s16, 0x100
	s_mov_b32 s46, s5
	s_add_nc_u64 s[40:41], s[4:5], s[10:11]
	s_or_b32 s12, s1, 0x7510000
	s_bitset1_b32 s49, 21
	s_or_b32 s4, s2, 0x7510000
	s_bitset1_b32 s50, 21
	s_add_nc_u64 s[20:21], s[26:27], s[6:7]
	s_add_nc_u64 s[24:25], s[14:15], s[18:19]
	s_mov_b32 s26, -1
	s_movk_i32 s8, 0x80
	s_mov_b32 s27, s11
	s_branch .LBB0_23
.LBB0_22:                               ;   in Loop: Header=BB0_23 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	s_cmp_eq_u32 s27, s47
	s_barrier_wait -1
	s_barrier_wait -3
	s_cbranch_scc1 .LBB0_46
.LBB0_23:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_26 Depth 2
                                        ;     Child Loop BB0_31 Depth 2
                                        ;     Child Loop BB0_34 Depth 2
	s_and_b32 s51, s27, 1
	s_add_co_i32 s27, s27, 1
	s_xor_b32 s5, s51, 1
	s_lshl_b32 s1, s27, 5
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_sub_co_i32 s1, s30, s1
	s_min_i32 s1, s1, 32
	s_cmp_lt_i32 s27, s47
	s_cselect_b32 s2, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_and_b32 s6, s2, exec_lo
	s_cselect_b32 s36, s1, 0
	s_cmp_lt_i32 s36, 32
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
	v_nop
	v_mov_b64_e32 v[152:153], v[0:1]
	s_cmp_lg_u32 s5, 0
	s_mov_b32 s6, 0
	s_cselect_b32 s15, s46, s39
	s_cselect_b32 s14, s45, 0
	s_mov_b32 s10, 0
	s_branch .LBB0_26
.LBB0_25:                               ;   in Loop: Header=BB0_26 Depth=2
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s1
	s_add_co_i32 s6, s6, 2
	v_add_nc_u32_e32 v153, 0x200, v153
	v_cmp_eq_u32_e32 vcc_lo, s6, v143
	v_add_nc_u32_e32 v152, 0x200, v152
	s_or_b32 s10, vcc_lo, s10
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s10
	s_cbranch_execz .LBB0_30
.LBB0_26:                               ;   Parent Loop BB0_23 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_mov_b32 s7, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b64 s[18:19], s[6:7], 0x100000000
	s_mov_b32 s7, exec_lo
	v_cmp_le_u32_e32 vcc_lo, s19, v131
	v_cmpx_le_u32_e64 s18, v136
	s_cbranch_execz .LBB0_28
; %bb.27:                               ;   in Loop: Header=BB0_26 Depth=2
	v_mov_b32_e32 v132, v152
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u64 v[154:155], v[132:133], 2, s[14:15]
	flat_store_b32 v[154:155], v133
.LBB0_28:                               ;   in Loop: Header=BB0_26 Depth=2
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s7
	s_and_saveexec_b32 s1, vcc_lo
	s_cbranch_execz .LBB0_25
; %bb.29:                               ;   in Loop: Header=BB0_26 Depth=2
	v_mov_b32_e32 v132, v153
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u64 v[154:155], v[132:133], 2, s[14:15]
	flat_store_b32 v[154:155], v133
	s_branch .LBB0_25
.LBB0_30:                               ;   in Loop: Header=BB0_23 Depth=1
	s_or_b32 exec_lo, exec_lo, s10
	v_mov_b64_e32 v[152:153], v[0:1]
	v_mov_b32_e32 v147, 16
	s_cmp_lg_u32 s5, 0
	s_mov_b32 s1, 0
	s_cselect_b32 s7, s41, s44
	s_cselect_b32 s6, s40, s38
.LBB0_31:                               ;   Parent Loop BB0_23 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_dual_mov_b32 v132, v152 :: v_dual_add_nc_u32 v147, -2, v147
	v_dual_mov_b32 v154, v153 :: v_dual_mov_b32 v155, v133
	v_add_nc_u32_e32 v153, 0x200, v153
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u64 v[156:157], v[132:133], 2, s[6:7]
	v_cmp_eq_u32_e32 vcc_lo, 0, v147
	v_add_nc_u32_e32 v152, 0x200, v152
	v_lshl_add_u64 v[154:155], v[154:155], 2, s[6:7]
	s_clause 0x1
	flat_store_b32 v[156:157], v133
	flat_store_b32 v[154:155], v133
	s_or_b32 s1, vcc_lo, s1
	s_wait_xcnt 0x0
	s_and_not1_b32 exec_lo, exec_lo, s1
	s_cbranch_execnz .LBB0_31
; %bb.32:                               ;   in Loop: Header=BB0_23 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	s_and_saveexec_b32 s1, s26
	s_cbranch_execz .LBB0_35
; %bb.33:                               ;   in Loop: Header=BB0_23 Depth=1
	v_add_nc_u64_e32 v[152:153], s[6:7], v[150:151]
	v_mov_b32_e32 v132, v145
	s_mov_b32 s6, 0
.LBB0_34:                               ;   Parent Loop BB0_23 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v132, 0x100, v132
	flat_store_b32 v[152:153], v133
	s_wait_xcnt 0x0
	v_add_nc_u64_e32 v[152:153], 0x400, v[152:153]
	v_cmp_lt_u32_e32 vcc_lo, 0xfff, v132
	s_or_b32 s6, vcc_lo, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s6
	s_cbranch_execnz .LBB0_34
.LBB0_35:                               ;   in Loop: Header=BB0_23 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	s_barrier_signal -1
	s_barrier_wait -1
.LBB0_36:                               ;   in Loop: Header=BB0_23 Depth=1
	s_and_b32 s1, s2, exec_lo
	s_cselect_b32 s1, s27, 0
	s_mov_b32 s2, exec_lo
	v_cmpx_lt_i32_e32 0, v141
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
	v_lshl_add_u32 v132, v130, 1, s1
	v_lshl_add_u32 v147, v138, 1, s1
	v_lshl_add_u32 v149, v144, 1, s2
	v_lshl_add_u32 v180, v140, 1, s1
	ds_load_b128 v[152:155], v132
	ds_load_b128 v[156:159], v132 offset:16
	ds_load_b128 v[160:163], v147 offset:1024
	ds_load_b128 v[164:167], v147 offset:1040
	ds_load_b128 v[168:171], v149 offset:1024
	ds_load_b128 v[172:175], v149 offset:1040
	v_lshl_add_u32 v132, v134, 1, s2
	; sched_group_barrier mask(0x00000100) size(4) SyncID(0)
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[82:89], v[160:167], v[168:175], v[82:89] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[114:121], v[152:159], v[168:175], v[114:121] matrix_a_reuse
	ds_load_b128 v[184:187], v132
	ds_load_b128 v[188:191], v132 offset:16
	v_lshl_add_u32 v132, v142, 1, s1
	ds_load_b128 v[192:195], v132 offset:3072
	ds_load_b128 v[196:199], v132 offset:3088
	v_lshl_add_u32 v132, v146, 1, s2
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[58:65], v[192:199], v[168:175], v[58:65] matrix_a_reuse
	ds_load_b128 v[176:179], v180 offset:2048
	ds_load_b128 v[180:183], v180 offset:2064
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(4) SyncID(0)
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[26:33], v[176:183], v[168:175], v[26:33] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[122:129], v[152:159], v[184:191], v[122:129]
	v_wmma_f32_16x16x32_bf16 v[90:97], v[160:167], v[184:191], v[90:97] matrix_a_reuse
	ds_load_b128 v[200:203], v132 offset:2048
	ds_load_b128 v[204:207], v132 offset:2064
	v_lshl_add_u32 v132, v148, 1, s2
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[66:73], v[192:199], v[200:207], v[66:73] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(4) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[42:49], v[176:183], v[184:191], v[42:49]
	v_wmma_f32_16x16x32_bf16 v[10:17], v[192:199], v[184:191], v[10:17] matrix_a_reuse
	ds_load_b128 v[208:211], v132 offset:3072
	ds_load_b128 v[212:215], v132 offset:3088
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[34:41], v[160:167], v[208:215], v[34:41]
	v_wmma_f32_16x16x32_bf16 v[74:81], v[192:199], v[208:215], v[74:81]
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(4) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[18:25], v[176:183], v[200:207], v[18:25] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[106:113], v[152:159], v[200:207], v[106:113] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[50:57], v[160:167], v[200:207], v[50:57] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[98:105], v[152:159], v[208:215], v[98:105] matrix_a_reuse
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[2:9], v[176:183], v[208:215], v[2:9] matrix_a_reuse
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
	v_cmpx_eq_u32_e32 1, v141
	s_cbranch_execz .LBB0_44
; %bb.43:                               ;   in Loop: Header=BB0_23 Depth=1
	s_cmp_lg_u32 s5, 0
	s_mov_b32 s2, s36
	s_cselect_b32 s13, s40, s38
	s_cmp_gt_i32 s36, 0
	s_mov_b32 s18, s11
	s_cselect_b32 s17, -1, 0
	s_lshl_b32 s10, s1, 5
	s_mov_b32 s19, s11
	s_lshl_b64 s[14:15], s[10:11], 1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_nc_u64 s[14:15], s[20:21], s[14:15]
	v_nop
	v_dual_mov_b32 v147, s13 :: v_dual_mov_b32 v152, s14
	s_and_b32 s10, s15, 0x1ffffff
	s_and_b32 s13, s29, s17
	s_bitset1_b32 s10, 31
	v_cndmask_b32_e64 v132, 0, 1, s13
	v_mov_b32_e32 v149, s10
	v_readfirstlane_b32 s53, v147
	v_readfirstlane_b32 s54, v152
	s_lshr_b64 s[14:15], s[2:3], 16
	v_readfirstlane_b32 s52, v132
	v_readfirstlane_b32 s55, v149
	s_lshl_b32 s13, s36, 16
	s_mov_b32 s15, s49
	s_mov_b32 s17, s9
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[52:55], s[12:19]
.LBB0_44:                               ;   in Loop: Header=BB0_23 Depth=1
	s_or_b32 exec_lo, exec_lo, s7
	s_and_not1_saveexec_b32 s2, s6
	s_cbranch_execz .LBB0_38
.LBB0_45:                               ;   in Loop: Header=BB0_23 Depth=1
	s_cmp_lg_u32 s5, 0
	s_cselect_b32 s5, s45, 0
	s_cmp_gt_i32 s36, 0
	s_cselect_b32 s13, -1, 0
	s_lshl_b32 s10, s1, 5
	s_delay_alu instid0(SALU_CYCLE_1)
	s_lshl_b64 s[6:7], s[10:11], 1
	s_mov_b32 s10, s11
	s_add_nc_u64 s[6:7], s[24:25], s[6:7]
	v_nop
	v_nop
	v_dual_mov_b32 v147, s5 :: v_dual_mov_b32 v152, s6
	s_and_b32 s1, s7, 0x1ffffff
	s_and_b32 s7, s35, s13
	s_bitset1_b32 s1, 31
	v_cndmask_b32_e64 v132, 0, 1, s7
	v_mov_b32_e32 v149, s1
	v_readfirstlane_b32 s53, v147
	v_readfirstlane_b32 s54, v152
	s_lshr_b64 s[6:7], s[36:37], 16
	v_readfirstlane_b32 s52, v132
	v_readfirstlane_b32 s55, v149
	s_lshl_b32 s5, s36, 16
	s_mov_b32 s7, s50
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
	v_and_or_b32 v130, v0, 15, v139
	v_cvt_pk_bf16_f32 v113, v112, v113
	v_cvt_pk_bf16_f32 v112, v110, v111
	v_cvt_pk_bf16_f32 v111, v108, v109
	v_and_or_b32 v1, v1, 8, v137
	v_cvt_pk_bf16_f32 v121, v120, v121
	v_cvt_pk_bf16_f32 v120, v118, v119
	v_cvt_pk_bf16_f32 v118, v114, v115
	v_lshlrev_b32_e32 v109, 4, v130
	v_lshl_or_b32 v1, v130, 7, v1
	v_cvt_pk_bf16_f32 v110, v106, v107
	v_cvt_pk_bf16_f32 v129, v128, v129
	v_cvt_pk_bf16_f32 v128, v126, v127
	v_and_b32_e32 v107, 0xdf0, v109
	v_add_nc_u32_e32 v108, 0x800, v1
	v_add_nc_u32_e32 v114, 0x1000, v1
	v_cvt_pk_bf16_f32 v127, v124, v125
	v_cvt_pk_bf16_f32 v126, v122, v123
	v_cvt_pk_bf16_f32 v119, v116, v117
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_lshrrev_b32 v106, 3, v108 :: v_dual_lshrrev_b32 v108, 3, v114
	v_lshlrev_b32_e32 v114, 1, v1
	v_add_nc_u32_e32 v115, 0x1800, v1
	v_cvt_pk_bf16_f32 v97, v96, v97
	v_and_b32_e32 v106, 0x1ff0, v106
	v_cvt_pk_bf16_f32 v96, v94, v95
	v_add_nc_u32_e32 v107, v107, v114
	v_cvt_pk_bf16_f32 v95, v92, v93
	v_add_nc_u32_e32 v92, 0x1010, v1
	v_add_nc_u32_e32 v106, v106, v114
	v_cvt_pk_bf16_f32 v89, v88, v89
	v_cvt_pk_bf16_f32 v88, v86, v87
	v_cvt_pk_bf16_f32 v86, v82, v83
	ds_store_b128 v107, v[126:129]
	ds_store_b128 v106, v[118:121] offset:4096
	v_add_nc_u32_e32 v107, 0x810, v1
	v_add_nc_u32_e32 v83, 0x1810, v1
	v_dual_lshrrev_b32 v109, 3, v115 :: v_dual_add_nc_u32 v115, 16, v1
	v_add_nc_u32_e32 v93, 32, v1
	v_cvt_pk_bf16_f32 v49, v48, v49
	v_cvt_pk_bf16_f32 v48, v46, v47
	v_cvt_pk_bf16_f32 v47, v44, v45
	v_add_nc_u32_e32 v44, 0x820, v1
	v_cvt_pk_bf16_f32 v17, v16, v17
	v_cvt_pk_bf16_f32 v16, v14, v15
	v_cvt_pk_bf16_f32 v14, v10, v11
	v_add_nc_u32_e32 v11, 0x1020, v1
	v_cvt_pk_bf16_f32 v105, v104, v105
	v_cvt_pk_bf16_f32 v104, v102, v103
	v_cvt_pk_bf16_f32 v103, v100, v101
	v_dual_lshrrev_b32 v101, 3, v107 :: v_dual_lshrrev_b32 v92, 3, v92
	v_cvt_pk_bf16_f32 v57, v56, v57
	v_cvt_pk_bf16_f32 v56, v54, v55
	v_cvt_pk_bf16_f32 v55, v52, v53
	v_lshrrev_b32_e32 v52, 3, v83
	v_lshrrev_b32_e32 v116, 3, v115
	v_cvt_pk_bf16_f32 v87, v84, v85
	v_dual_lshrrev_b32 v84, 3, v93 :: v_dual_lshrrev_b32 v10, 3, v44
	v_cvt_pk_bf16_f32 v15, v12, v13
	v_cvt_pk_bf16_f32 v12, v38, v39
	v_lshrrev_b32_e32 v38, 3, v11
	v_and_b32_e32 v108, 0x1ff0, v108
	v_and_b32_e32 v109, 0x1ff0, v109
	v_cvt_pk_bf16_f32 v94, v90, v91
	v_and_b32_e32 v90, 0x1ff0, v101
	v_lshlrev_b32_e32 v91, 1, v115
	v_and_b32_e32 v92, 0x1ff0, v92
	v_cvt_pk_bf16_f32 v46, v42, v43
	v_and_b32_e32 v43, 0x1ff0, v52
	v_and_b32_e32 v106, 0xdf0, v116
	v_and_b32_e32 v84, 0xdf0, v84
	v_cvt_pk_bf16_f32 v13, v40, v41
	v_and_b32_e32 v40, 0x1ff0, v10
	v_lshlrev_b32_e32 v41, 1, v93
	v_cvt_pk_bf16_f32 v10, v34, v35
	v_and_b32_e32 v35, 0x1ff0, v38
	v_dual_add_nc_u32 v108, v108, v114 :: v_dual_add_nc_u32 v109, v109, v114
	v_cvt_pk_bf16_f32 v102, v98, v99
	v_dual_add_nc_u32 v90, v90, v91 :: v_dual_add_nc_u32 v82, v92, v91
	v_cvt_pk_bf16_f32 v54, v50, v51
	v_add_nc_u32_e32 v43, v43, v91
	v_cvt_pk_bf16_f32 v11, v36, v37
	v_cvt_pk_bf16_f32 v25, v24, v25
	v_cvt_pk_bf16_f32 v24, v22, v23
	v_cvt_pk_bf16_f32 v22, v18, v19
	v_add_nc_u32_e32 v18, 0x830, v1
	v_dual_add_nc_u32 v100, v106, v114 :: v_dual_add_nc_u32 v53, v84, v114
	v_add_nc_u32_e32 v34, v40, v41
	v_cvt_pk_bf16_f32 v33, v32, v33
	v_cvt_pk_bf16_f32 v32, v30, v31
	v_cvt_pk_bf16_f32 v31, v28, v29
	v_cvt_pk_bf16_f32 v30, v26, v27
	v_add_nc_u32_e32 v36, 0x1820, v1
	v_add_nc_u32_e32 v26, v35, v41
	v_cvt_pk_bf16_f32 v23, v20, v21
	v_add_nc_u32_e32 v85, 48, v1
	ds_store_b128 v108, v[110:113] offset:8192
	ds_store_b128 v100, v[94:97] offset:32
	ds_store_b128 v109, v[102:105] offset:12288
	ds_store_b128 v53, v[46:49] offset:64
	ds_store_b128 v90, v[86:89] offset:4096
	ds_store_b128 v82, v[54:57] offset:8192
	ds_store_b128 v43, v[10:13] offset:12288
	ds_store_b128 v34, v[30:33] offset:4096
	ds_store_b128 v26, v[22:25] offset:8192
	v_lshrrev_b32_e32 v10, 3, v18
	v_add_nc_u32_e32 v11, 0x1030, v1
	v_add_nc_u32_e32 v1, 0x1830, v1
	v_dual_lshrrev_b32 v27, 3, v36 :: v_dual_lshrrev_b32 v83, 3, v85
	v_cvt_pk_bf16_f32 v9, v8, v9
	v_cvt_pk_bf16_f32 v8, v6, v7
	v_and_b32_e32 v10, 0x1ff0, v10
	v_lshlrev_b32_e32 v18, 1, v85
	v_cvt_pk_bf16_f32 v7, v4, v5
	v_dual_lshrrev_b32 v4, 3, v11 :: v_dual_lshrrev_b32 v1, 3, v1
	v_and_b32_e32 v19, 0x1ff0, v27
	v_and_b32_e32 v42, 0xdf0, v83
	v_add_nc_u32_e32 v23, v10, v18
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v10, 0x1ff0, v4
	v_and_b32_e32 v1, 0x1ff0, v1
	v_add_nc_u32_e32 v22, v19, v41
	v_cvt_pk_bf16_f32 v6, v2, v3
	v_add_nc_u32_e32 v42, v42, v114
	v_cvt_pk_bf16_f32 v5, v64, v65
	v_cvt_pk_bf16_f32 v4, v62, v63
	v_cvt_pk_bf16_f32 v3, v60, v61
	v_cvt_pk_bf16_f32 v2, v58, v59
	v_add_nc_u32_e32 v24, v10, v18
	v_cvt_pk_bf16_f32 v13, v72, v73
	v_cvt_pk_bf16_f32 v12, v70, v71
	v_cvt_pk_bf16_f32 v11, v68, v69
	v_cvt_pk_bf16_f32 v10, v66, v67
	v_add_nc_u32_e32 v1, v1, v18
	v_cvt_pk_bf16_f32 v21, v80, v81
	v_cvt_pk_bf16_f32 v20, v78, v79
	v_cvt_pk_bf16_f32 v19, v76, v77
	v_cvt_pk_bf16_f32 v18, v74, v75
	ds_store_b128 v22, v[6:9] offset:12288
	ds_store_b128 v42, v[14:17] offset:96
	ds_store_b128 v23, v[2:5] offset:4096
	ds_store_b128 v24, v[10:13] offset:8192
	ds_store_b128 v1, v[18:21] offset:12288
.LBB0_48:
	v_cmp_ne_u32_e32 vcc_lo, 1, v135
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
	v_lshl_add_u32 v11, v11, 7, v12
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_mul_lo_u32 v16, v27, s18
	v_dual_add_nc_u32 v24, s21, v27 :: v_dual_sub_nc_u32 v6, v2, v6
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
	v_dual_sub_nc_u32 v7, v4, v9 :: v_dual_lshlrev_b32 v4, 7, v4
	v_lshlrev_b32_e32 v9, 7, v9
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
	.size	bm128_bn256_bk032_wm2_wn4_mc1, .Lfunc_end0-bm128_bn256_bk032_wm2_wn4_mc1
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel bm128_bn256_bk032_wm2_wn4_mc1
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
		.amdhsa_next_free_vgpr 216
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
	.section	.text.bm128_bn256_bk032_wm2_wn4_mc1,"axG",@progbits,bm128_bn256_bk032_wm2_wn4_mc1,comdat
                                        ; -- End function
	.set .Lbm128_bn256_bk032_wm2_wn4_mc1.num_vgpr, 216
	.set .Lbm128_bn256_bk032_wm2_wn4_mc1.num_agpr, 0
	.set .Lbm128_bn256_bk032_wm2_wn4_mc1.numbered_sgpr, 56
	.set .Lbm128_bn256_bk032_wm2_wn4_mc1.num_named_barrier, 0
	.set .Lbm128_bn256_bk032_wm2_wn4_mc1.private_seg_size, 0
	.set .Lbm128_bn256_bk032_wm2_wn4_mc1.uses_vcc, 1
	.set .Lbm128_bn256_bk032_wm2_wn4_mc1.uses_flat_scratch, 1
	.set .Lbm128_bn256_bk032_wm2_wn4_mc1.has_dyn_sized_stack, 0
	.set .Lbm128_bn256_bk032_wm2_wn4_mc1.has_recursion, 0
	.set .Lbm128_bn256_bk032_wm2_wn4_mc1.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 6768
; TotalNumSgprs: 58
; NumVgprs: 216
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 69632 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 13
; NumSGPRsForWavesPerEU: 58
; NumVGPRsForWavesPerEU: 216
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
	.type	__hip_cuid_a47d1cce12f74907,@object ; @__hip_cuid_a47d1cce12f74907
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_a47d1cce12f74907
__hip_cuid_a47d1cce12f74907:
	.byte	0                               ; 0x0
	.size	__hip_cuid_a47d1cce12f74907, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git aa451e1fe6a793394d6733051b1778633063ae96+PATCHED:d17c5aa0e3ea29cde402f58f27e39b6034effa27)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_a47d1cce12f74907
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
    macrotile: [128, 256, 32]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, One]
  MatrixInstruction: [16, 16, 32, 1]
  EnableMatrixInstruction: True
  MIWaveTile: [4, 4]
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
    .group_segment_fixed_size: 69632
    .kernarg_segment_align: 8
    .kernarg_segment_size: 132
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           bm128_bn256_bk032_wm2_wn4_mc1
    .private_segment_fixed_size: 0
    .sgpr_count:     58
    .sgpr_spill_count: 0
    .symbol:         bm128_bn256_bk032_wm2_wn4_mc1.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     216
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
