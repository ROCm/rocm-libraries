	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 4
	.text
	.protected	wvSplitKQ_vllm_f8_f16_tn_n4_y1u4_gfx1201 ; -- Begin function wvSplitKQ_vllm_f8_f16_tn_n4_y1u4_gfx1201
	.globl	wvSplitKQ_vllm_f8_f16_tn_n4_y1u4_gfx1201
	.p2align	8
	.type	wvSplitKQ_vllm_f8_f16_tn_n4_y1u4_gfx1201,@function
wvSplitKQ_vllm_f8_f16_tn_n4_y1u4_gfx1201: ; @wvSplitKQ_vllm_f8_f16_tn_n4_y1u4_gfx1201
; %bb.0:
	s_clause 0x1
	s_load_b64 s[2:3], s[0:1], 0x0
	s_load_b64 s[18:19], s[0:1], 0x10
	v_and_b32_e32 v86, 0x3ff, v0
	v_bfe_u32 v0, v0, 10, 10
	s_mov_b32 s21, 0
	s_mov_b32 s4, exec_lo
	s_mov_b32 s9, s21
	v_lshlrev_b32_e32 v85, 4, v86
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)
	v_lshl_add_u32 v3, v0, 9, v85
	s_wait_kmcnt 0x0
	s_cmp_lt_u32 s2, 0x10000
	s_mov_b32 s8, s2
	s_cselect_b32 s33, s2, 0x10000
	v_cmpx_gt_u32_e64 s33, v3
	s_cbranch_execz .LBB0_9
; %bb.1:
	global_load_b128 v[4:7], v3, s[18:19]
	v_add_nc_u32_e32 v1, 0x2000, v3
	s_mov_b32 s5, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b128 v3, v[4:7]
	v_cmpx_gt_u32_e64 s33, v1
	s_xor_b32 s5, exec_lo, s5
	s_cbranch_execz .LBB0_9
; %bb.2:
	v_add_co_u32 v1, s5, s18, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v2, null, s19, 0, s5
	v_add_nc_u32_e32 v8, 0x4000, v3
	s_mov_b32 s5, exec_lo
	global_load_b128 v[4:7], v[1:2], off offset:8192
	s_wait_loadcnt 0x0
	ds_store_b128 v3, v[4:7] offset:8192
	v_cmpx_gt_u32_e64 s33, v8
	s_wait_alu 0xfffe
	s_xor_b32 s5, exec_lo, s5
	s_cbranch_execz .LBB0_9
; %bb.3:
	global_load_b128 v[4:7], v[1:2], off offset:16384
	v_add_nc_u32_e32 v8, 0x6000, v3
	s_mov_b32 s5, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b128 v3, v[4:7] offset:16384
	v_cmpx_gt_u32_e64 s33, v8
	s_wait_alu 0xfffe
	s_xor_b32 s5, exec_lo, s5
	s_cbranch_execz .LBB0_9
; %bb.4:
	global_load_b128 v[4:7], v[1:2], off offset:24576
	v_add_nc_u32_e32 v8, 0x8000, v3
	s_mov_b32 s5, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b128 v3, v[4:7] offset:24576
	v_cmpx_gt_u32_e64 s33, v8
	s_wait_alu 0xfffe
	s_xor_b32 s5, exec_lo, s5
	s_cbranch_execz .LBB0_9
; %bb.5:
	global_load_b128 v[4:7], v[1:2], off offset:32768
	v_add_nc_u32_e32 v8, 0xa000, v3
	s_mov_b32 s5, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b128 v3, v[4:7] offset:32768
	v_cmpx_gt_u32_e64 s33, v8
	s_wait_alu 0xfffe
	s_xor_b32 s5, exec_lo, s5
	s_cbranch_execz .LBB0_9
; %bb.6:
	global_load_b128 v[4:7], v[1:2], off offset:40960
	v_add_nc_u32_e32 v8, 0xc000, v3
	s_mov_b32 s5, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b128 v3, v[4:7] offset:40960
	v_cmpx_gt_u32_e64 s33, v8
	s_wait_alu 0xfffe
	s_xor_b32 s5, exec_lo, s5
	s_cbranch_execz .LBB0_9
; %bb.7:
	global_load_b128 v[4:7], v[1:2], off offset:49152
	v_add_nc_u32_e32 v8, 0xe000, v3
	s_mov_b32 s5, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b128 v3, v[4:7] offset:49152
	v_cmpx_gt_u32_e64 s33, v8
	s_wait_alu 0xfffe
	s_xor_b32 s5, exec_lo, s5
	s_cbranch_execz .LBB0_9
; %bb.8:
	global_load_b128 v[4:7], v[1:2], off offset:57344
	s_wait_loadcnt 0x0
	ds_store_b128 v3, v[4:7] offset:57344
.LBB0_9:
	s_or_b32 exec_lo, exec_lo, s4
	s_clause 0x2
	s_load_b64 s[12:13], s[0:1], 0x20
	s_load_b128 s[4:7], s[0:1], 0x48
	s_load_b64 s[14:15], s[0:1], 0x58
	s_sub_nc_u64 s[10:11], 0x10000, s[8:9]
	s_delay_alu instid0(SALU_CYCLE_1)
	v_cmp_lt_u64_e64 s11, s[10:11], s[8:9]
	s_and_b32 s11, s11, exec_lo
	s_cselect_b32 s10, s10, s8
	s_cmp_lt_u32 s2, 0x10000
	s_wait_alu 0xfffe
	s_cselect_b32 s38, s10, 0
	s_wait_kmcnt 0x0
	s_mov_b32 s20, s13
	s_mov_b32 s13, exec_lo
	s_add_nc_u64 s[10:11], s[18:19], s[20:21]
	v_cmpx_gt_u32_e64 s38, v3
	s_cbranch_execz .LBB0_12
; %bb.10:
	v_mov_b32_e32 v1, v3
	s_mov_b32 s16, 0
.LBB0_11:                               ; =>This Inner Loop Header: Depth=1
	global_load_b128 v[4:7], v1, s[10:11]
	v_add_nc_u32_e32 v2, s2, v1
	v_add_nc_u32_e32 v1, 0x2000, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e32 vcc_lo, s38, v1
	s_or_b32 s16, vcc_lo, s16
	s_wait_loadcnt 0x0
	ds_store_b128 v2, v[4:7]
	s_and_not1_b32 exec_lo, exec_lo, s16
	s_cbranch_execnz .LBB0_11
.LBB0_12:
	s_or_b32 exec_lo, exec_lo, s13
	s_load_b64 s[16:17], s[0:1], 0x8
	s_lshl_b64 s[22:23], s[8:9], 1
	s_add_nc_u64 s[10:11], s[10:11], s[20:21]
	s_sub_nc_u64 s[22:23], 0x10000, s[22:23]
	s_delay_alu instid0(SALU_CYCLE_1)
	v_cmp_lt_u64_e64 s13, s[22:23], s[8:9]
	s_and_b32 s13, s13, exec_lo
	s_cselect_b32 s13, s22, s8
	s_cmp_lt_u32 s2, 0x8000
	s_mov_b32 s22, exec_lo
	s_cselect_b32 s40, s13, 0
	s_mov_b32 s13, 0
	s_lshl_b32 s39, s2, 1
	v_cmpx_gt_u32_e64 s40, v3
	s_cbranch_execz .LBB0_15
; %bb.13:
	v_mov_b32_e32 v1, v3
.LBB0_14:                               ; =>This Inner Loop Header: Depth=1
	global_load_b128 v[4:7], v1, s[10:11]
	s_wait_alu 0xfffe
	v_add_nc_u32_e32 v2, s39, v1
	v_add_nc_u32_e32 v1, 0x2000, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e32 vcc_lo, s40, v1
	s_or_b32 s13, vcc_lo, s13
	s_wait_loadcnt 0x0
	ds_store_b128 v2, v[4:7]
	s_and_not1_b32 exec_lo, exec_lo, s13
	s_cbranch_execnz .LBB0_14
.LBB0_15:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s22
	s_mul_u64 s[22:23], s[8:9], -3
	s_mov_b32 s13, 0
	s_wait_alu 0xfffe
	s_add_nc_u64 s[22:23], s[22:23], 0x10000
	s_mul_i32 s41, s2, 3
	s_wait_alu 0xfffe
	v_cmp_lt_u64_e64 s9, s[22:23], s[8:9]
	s_and_b32 s9, s9, exec_lo
	s_cselect_b32 s8, s22, s8
	s_cmp_lt_u32 s2, 0x5556
	s_mov_b32 s22, exec_lo
	s_wait_alu 0xfffe
	s_cselect_b32 s42, s8, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	v_cmpx_gt_u32_e64 s42, v3
	s_cbranch_execz .LBB0_18
; %bb.16:
	s_add_nc_u64 s[8:9], s[10:11], s[20:21]
.LBB0_17:                               ; =>This Inner Loop Header: Depth=1
	global_load_b128 v[4:7], v3, s[8:9]
	v_add_nc_u32_e32 v1, s41, v3
	v_add_nc_u32_e32 v3, 0x2000, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e32 vcc_lo, s42, v3
	s_or_b32 s13, vcc_lo, s13
	s_wait_loadcnt 0x0
	ds_store_b128 v1, v[4:7]
	s_and_not1_b32 exec_lo, exec_lo, s13
	s_cbranch_execnz .LBB0_17
.LBB0_18:
	s_or_b32 exec_lo, exec_lo, s22
	s_load_b32 s13, s[0:1], 0x18
	s_add_co_i32 s8, s3, -1
	s_mov_b32 s22, 4
	s_wait_kmcnt 0x0
	s_lshl_b32 s9, s13, 4
	s_wait_alu 0xfffe
	v_add_co_u32 v1, s11, s9, s8
	s_and_not1_b32 vcc_lo, exec_lo, s11
	v_readfirstlane_b32 s10, v1
	s_cbranch_vccz .LBB0_20
; %bb.19:
	s_cvt_f32_u32 s11, s9
	s_sub_co_i32 s22, 0, s9
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_iflag_f32_e32 v1, s11
	v_readfirstlane_b32 s11, v1
	s_mul_f32 s11, s11, 0x4f7ffffe
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_cvt_u32_f32 s11, s11
	s_wait_alu 0xfffe
	s_mul_i32 s22, s22, s11
	s_wait_alu 0xfffe
	s_mul_hi_u32 s22, s11, s22
	s_wait_alu 0xfffe
	s_add_co_i32 s11, s11, s22
	s_wait_alu 0xfffe
	s_mul_hi_u32 s11, s10, s11
	s_wait_alu 0xfffe
	s_mul_i32 s22, s11, s9
	s_wait_alu 0xfffe
	s_sub_co_i32 s10, s10, s22
	s_add_co_i32 s22, s11, 1
	s_wait_alu 0xfffe
	s_sub_co_i32 s23, s10, s9
	s_cmp_ge_u32 s10, s9
	s_cselect_b32 s11, s22, s11
	s_wait_alu 0xfffe
	s_cselect_b32 s10, s23, s10
	s_add_co_i32 s22, s11, 1
	s_wait_alu 0xfffe
	s_cmp_ge_u32 s10, s9
	s_cselect_b32 s9, s22, s11
	s_wait_alu 0xfffe
	s_mul_i32 s9, s9, s13
	s_wait_alu 0xfffe
	s_cvt_f32_u32 s10, s9
	s_sub_co_i32 s11, 0, s9
	s_add_co_i32 s8, s9, s8
	s_wait_alu 0xfffe
	v_rcp_iflag_f32_e32 v1, s10
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_2)
	v_readfirstlane_b32 s10, v1
	s_mul_f32 s10, s10, 0x4f7ffffe
	s_wait_alu 0xfffe
	s_cvt_u32_f32 s10, s10
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_2)
	s_mul_i32 s11, s11, s10
	s_wait_alu 0xfffe
	s_mul_hi_u32 s11, s10, s11
	s_wait_alu 0xfffe
	s_add_co_i32 s10, s10, s11
	s_wait_alu 0xfffe
	s_mul_hi_u32 s10, s8, s10
	s_wait_alu 0xfffe
	s_mul_i32 s11, s10, s9
	s_wait_alu 0xfffe
	s_sub_co_i32 s8, s8, s11
	s_add_co_i32 s11, s10, 1
	s_wait_alu 0xfffe
	s_sub_co_i32 s22, s8, s9
	s_cmp_ge_u32 s8, s9
	s_cselect_b32 s10, s11, s10
	s_wait_alu 0xfffe
	s_cselect_b32 s8, s22, s8
	s_add_co_i32 s11, s10, 1
	s_wait_alu 0xfffe
	s_cmp_ge_u32 s8, s9
	s_cselect_b32 s8, s11, s10
	s_wait_alu 0xfffe
	s_max_u32 s22, s8, 4
.LBB0_20:
	s_mov_b32 s23, 1.0
	s_cmp_eq_u64 s[4:5], 0
	s_mov_b32 s24, 1.0
	s_cbranch_scc1 .LBB0_22
; %bb.21:
	s_load_b32 s24, s[4:5], 0x0
.LBB0_22:
	s_cmp_eq_u64 s[6:7], 0
	s_cbranch_scc1 .LBB0_24
; %bb.23:
	s_load_b32 s23, s[6:7], 0x0
.LBB0_24:
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_mov_b32 s4, exec_lo
	s_wait_alu 0xfffe
	v_cmpx_gt_u32_e64 s22, v0
	s_cbranch_execz .LBB0_146
; %bb.25:
	s_load_b256 s[4:11], s[0:1], 0x28
	s_cmp_eq_u32 s38, s2
	v_mad_co_u64_u32 v[80:81], null, s22, ttmp9, v[0:1]
	s_cselect_b32 s0, -1, 0
	s_cmp_eq_u32 s40, s2
	s_mul_i32 s13, s22, s13
	s_cselect_b32 s1, -1, 0
	s_cmp_eq_u32 s42, s2
	s_cselect_b32 s25, -1, 0
	s_cmp_eq_u32 s2, s33
	s_cselect_b32 s26, -1, 0
	s_wait_alu 0xfffe
	s_and_b32 s0, s26, s0
	s_wait_alu 0xfffe
	s_and_b32 s0, s0, s1
	s_wait_alu 0xfffe
	s_and_b32 s1, s0, s25
	s_wait_kmcnt 0x0
	s_mul_f32 s8, s8, s24
	v_cmp_gt_u32_e64 s0, s3, v80
	s_wait_alu 0xfffe
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_mov_b32 s1, -1
	s_mul_f32 s8, s8, s23
	s_cbranch_vccz .LBB0_118
; %bb.26:
	s_and_saveexec_b32 s43, s0
	s_cbranch_execz .LBB0_117
; %bb.27:
	s_lshl_b64 s[22:23], s[20:21], 1
	s_cmp_eq_u64 s[14:15], 0
	s_mov_b32 s27, 0
	s_cselect_b32 s0, -1, 0
	s_cmp_lg_u64 s[14:15], 0
	v_cmp_eq_u32_e64 s1, 31, v86
	s_cselect_b32 s44, -1, 0
	s_cmp_lg_u32 s2, 0
	v_dual_mov_b32 v82, 0 :: v_dual_add_nc_u32 v87, s2, v85
	s_cselect_b32 s45, -1, 0
	s_add_co_i32 s46, s2, -16
	s_cmp_neq_f32 s9, 0
	v_dual_mov_b32 v81, v80 :: v_dual_add_nc_u32 v88, s39, v85
	v_add_nc_u32_e32 v89, s41, v85
	v_mbcnt_lo_u32_b32 v90, -1, 0
	s_mov_b32 s26, s10
	s_mov_b32 s34, s11
	s_mov_b32 s35, s27
	s_mul_u64 s[24:25], s[20:21], 3
	s_cselect_b32 s21, -1, 0
	s_mul_u64 s[28:29], s[26:27], 6
	s_lshl_b64 s[30:31], s[26:27], 2
	s_lshl_b64 s[34:35], s[34:35], 1
	s_lshl_b64 s[36:37], s[26:27], 1
	s_branch .LBB0_30
.LBB0_28:                               ;   in Loop: Header=BB0_30 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v0, v4, v6
	v_cndmask_b32_e64 v4, v0, v4, s0
	v_add_co_u32 v0, vcc_lo, v2, s34
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s35, v3, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f16_f32_e32 v2, v4
	global_store_b16 v[0:1], v2, off
.LBB0_29:                               ;   in Loop: Header=BB0_30 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s26
	v_add_nc_u32_e32 v81, s13, v81
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e32 vcc_lo, s3, v81
	s_or_b32 s27, vcc_lo, s27
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s27
	s_cbranch_execz .LBB0_117
.LBB0_30:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_39 Depth 2
	v_mov_b32_e32 v91, 0
	s_and_not1_b32 vcc_lo, exec_lo, s44
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_32
; %bb.31:                               ;   in Loop: Header=BB0_30 Depth=1
	s_wait_dscnt 0x3
	v_lshlrev_b64_e32 v[0:1], 1, v[81:82]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, s14, v0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s15, v1, vcc_lo
	global_load_u16 v91, v[0:1], off
.LBB0_32:                               ;   in Loop: Header=BB0_30 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s45
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_107
; %bb.33:                               ;   in Loop: Header=BB0_30 Depth=1
	v_mad_co_u64_u32 v[83:84], null, v81, s12, s[16:17]
	v_dual_mov_b32 v95, 0 :: v_dual_mov_b32 v94, 0
	v_dual_mov_b32 v93, 0 :: v_dual_mov_b32 v92, 0
	s_mov_b32 s26, 0
	s_branch .LBB0_39
.LBB0_34:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s51
.LBB0_35:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s50
.LBB0_36:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s49
.LBB0_37:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s48
.LBB0_38:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s47
	s_wait_loadcnt_dscnt 0x0
	v_dot4_f32_fp8_fp8 v16, v16, v12, v95
	v_dot4_f32_fp8_fp8 v20, v20, v12, v94
	v_dot4_f32_fp8_fp8 v24, v24, v12, v93
	v_dot4_f32_fp8_fp8 v12, v28, v12, v92
	s_addk_co_i32 s26, 0x800
	v_dot4_f32_fp8_fp8 v16, v17, v13, v16
	v_dot4_f32_fp8_fp8 v17, v21, v13, v20
	v_dot4_f32_fp8_fp8 v20, v25, v13, v24
	v_dot4_f32_fp8_fp8 v12, v29, v13, v12
	s_wait_alu 0xfffe
	s_cmp_lt_u32 s26, s2
	v_dot4_f32_fp8_fp8 v13, v18, v14, v16
	v_dot4_f32_fp8_fp8 v16, v22, v14, v17
	v_dot4_f32_fp8_fp8 v17, v26, v14, v20
	v_dot4_f32_fp8_fp8 v12, v30, v14, v12
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v13, v19, v15, v13
	v_dot4_f32_fp8_fp8 v14, v23, v15, v16
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v16, v27, v15, v17
	v_dot4_f32_fp8_fp8 v12, v31, v15, v12
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v13, v32, v8, v13
	v_dot4_f32_fp8_fp8 v14, v36, v8, v14
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v15, v40, v8, v16
	v_dot4_f32_fp8_fp8 v8, v44, v8, v12
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v12, v33, v9, v13
	v_dot4_f32_fp8_fp8 v13, v37, v9, v14
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v14, v41, v9, v15
	v_dot4_f32_fp8_fp8 v8, v45, v9, v8
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v9, v34, v10, v12
	v_dot4_f32_fp8_fp8 v12, v38, v10, v13
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v13, v42, v10, v14
	v_dot4_f32_fp8_fp8 v8, v46, v10, v8
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v9, v35, v11, v9
	v_dot4_f32_fp8_fp8 v10, v39, v11, v12
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v12, v43, v11, v13
	v_dot4_f32_fp8_fp8 v8, v47, v11, v8
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v9, v48, v4, v9
	v_dot4_f32_fp8_fp8 v10, v52, v4, v10
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v11, v56, v4, v12
	v_dot4_f32_fp8_fp8 v4, v60, v4, v8
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v8, v49, v5, v9
	v_dot4_f32_fp8_fp8 v9, v53, v5, v10
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v10, v57, v5, v11
	v_dot4_f32_fp8_fp8 v4, v61, v5, v4
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v5, v50, v6, v8
	v_dot4_f32_fp8_fp8 v8, v54, v6, v9
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v9, v58, v6, v10
	v_dot4_f32_fp8_fp8 v4, v62, v6, v4
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v5, v51, v7, v5
	v_dot4_f32_fp8_fp8 v6, v55, v7, v8
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v8, v59, v7, v9
	v_dot4_f32_fp8_fp8 v4, v63, v7, v4
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v5, v64, v0, v5
	v_dot4_f32_fp8_fp8 v6, v68, v0, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v7, v72, v0, v8
	v_dot4_f32_fp8_fp8 v0, v76, v0, v4
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v4, v65, v1, v5
	v_dot4_f32_fp8_fp8 v5, v69, v1, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v6, v73, v1, v7
	v_dot4_f32_fp8_fp8 v0, v77, v1, v0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v1, v66, v2, v4
	v_dot4_f32_fp8_fp8 v4, v70, v2, v5
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v5, v74, v2, v6
	v_dot4_f32_fp8_fp8 v0, v78, v2, v0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v95, v67, v3, v1
	v_dot4_f32_fp8_fp8 v94, v71, v3, v4
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v93, v75, v3, v5
	v_dot4_f32_fp8_fp8 v92, v79, v3, v0
	s_cbranch_scc0 .LBB0_108
.LBB0_39:                               ;   Parent Loop BB0_30 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu 0xfffe
	v_add_nc_u32_e32 v97, s26, v85
	v_dual_mov_b32 v79, 0 :: v_dual_mov_b32 v78, 0
	v_dual_mov_b32 v77, 0 :: v_dual_mov_b32 v76, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v18, 0 :: v_dual_add_nc_u32 v99, 0x200, v97
	v_add_nc_u32_e32 v98, 0x400, v97
	v_min_u32_e32 v0, s46, v97
	v_dual_mov_b32 v19, 0 :: v_dual_add_nc_u32 v96, 0x600, v97
	v_min_u32_e32 v2, s46, v99
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_min_u32_e32 v4, s46, v98
	v_add_co_u32 v0, vcc_lo, v83, v0
	s_delay_alu instid0(VALU_DEP_4)
	v_min_u32_e32 v6, s46, v96
	s_wait_dscnt 0x3
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, 0, v84, vcc_lo
	v_add_co_u32 v2, vcc_lo, v83, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, 0, v84, vcc_lo
	v_add_co_u32 v4, vcc_lo, v83, v4
	s_wait_dscnt 0x0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v5, null, 0, v84, vcc_lo
	v_add_co_u32 v16, vcc_lo, v83, v6
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v17, null, 0, v84, vcc_lo
	s_clause 0x3
	global_load_b128 v[12:15], v[0:1], off
	global_load_b128 v[8:11], v[2:3], off
	global_load_b128 v[4:7], v[4:5], off
	global_load_b128 v[0:3], v[16:17], off
	v_mov_b32_e32 v16, 0
	v_cmp_gt_u32_e32 vcc_lo, s2, v97
	v_mov_b32_e32 v17, 0
	v_dual_mov_b32 v63, 0 :: v_dual_mov_b32 v62, 0
	v_dual_mov_b32 v61, 0 :: v_dual_mov_b32 v60, 0
	v_dual_mov_b32 v47, 0 :: v_dual_mov_b32 v46, 0
	v_dual_mov_b32 v45, 0 :: v_dual_mov_b32 v44, 0
	v_dual_mov_b32 v31, 0 :: v_dual_mov_b32 v30, 0
	v_dual_mov_b32 v29, 0 :: v_dual_mov_b32 v28, 0
	v_dual_mov_b32 v75, 0 :: v_dual_mov_b32 v74, 0
	v_dual_mov_b32 v73, 0 :: v_dual_mov_b32 v72, 0
	v_dual_mov_b32 v59, 0 :: v_dual_mov_b32 v58, 0
	v_dual_mov_b32 v57, 0 :: v_dual_mov_b32 v56, 0
	v_dual_mov_b32 v43, 0 :: v_dual_mov_b32 v42, 0
	v_dual_mov_b32 v41, 0 :: v_dual_mov_b32 v40, 0
	v_dual_mov_b32 v27, 0 :: v_dual_mov_b32 v26, 0
	v_dual_mov_b32 v25, 0 :: v_dual_mov_b32 v24, 0
	v_dual_mov_b32 v71, 0 :: v_dual_mov_b32 v70, 0
	v_dual_mov_b32 v69, 0 :: v_dual_mov_b32 v68, 0
	v_dual_mov_b32 v55, 0 :: v_dual_mov_b32 v54, 0
	v_dual_mov_b32 v53, 0 :: v_dual_mov_b32 v52, 0
	v_dual_mov_b32 v39, 0 :: v_dual_mov_b32 v38, 0
	v_dual_mov_b32 v37, 0 :: v_dual_mov_b32 v36, 0
	v_dual_mov_b32 v23, 0 :: v_dual_mov_b32 v22, 0
	v_dual_mov_b32 v21, 0 :: v_dual_mov_b32 v20, 0
	v_dual_mov_b32 v67, 0 :: v_dual_mov_b32 v66, 0
	v_dual_mov_b32 v65, 0 :: v_dual_mov_b32 v64, 0
	v_dual_mov_b32 v51, 0 :: v_dual_mov_b32 v50, 0
	v_dual_mov_b32 v49, 0 :: v_dual_mov_b32 v48, 0
	v_dual_mov_b32 v35, 0 :: v_dual_mov_b32 v34, 0
	v_dual_mov_b32 v33, 0 :: v_dual_mov_b32 v32, 0
	s_and_saveexec_b32 s47, vcc_lo
	s_cbranch_execz .LBB0_38
; %bb.40:                               ;   in Loop: Header=BB0_39 Depth=2
	v_add_co_u32 v28, s48, s18, v97
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v29, null, s19, 0, s48
	s_mov_b32 s48, exec_lo
                                        ; implicit-def: $vgpr16
	v_cmpx_le_u32_e64 s33, v97
	s_wait_alu 0xfffe
	s_xor_b32 s48, exec_lo, s48
	s_cbranch_execz .LBB0_42
; %bb.41:                               ;   in Loop: Header=BB0_39 Depth=2
	global_load_b128 v[16:19], v[28:29], off
.LBB0_42:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s48, s48
	s_cbranch_execz .LBB0_44
; %bb.43:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_loadcnt 0x0
	ds_load_b128 v[16:19], v97
.LBB0_44:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s48
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s48, exec_lo
                                        ; implicit-def: $vgpr23
	v_cmpx_le_u32_e64 s38, v97
	s_wait_alu 0xfffe
	s_xor_b32 s48, exec_lo, s48
	s_cbranch_execz .LBB0_46
; %bb.45:                               ;   in Loop: Header=BB0_39 Depth=2
	v_add_co_u32 v20, vcc_lo, v28, s20
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v21, null, 0, v29, vcc_lo
	global_load_b128 v[20:23], v[20:21], off
.LBB0_46:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s48, s48
	s_cbranch_execz .LBB0_48
; %bb.47:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v20, s26, v87
	ds_load_b128 v[20:23], v20
.LBB0_48:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s48
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s48, exec_lo
                                        ; implicit-def: $vgpr27
	v_cmpx_le_u32_e64 s40, v97
	s_wait_alu 0xfffe
	s_xor_b32 s48, exec_lo, s48
	s_cbranch_execz .LBB0_50
; %bb.49:                               ;   in Loop: Header=BB0_39 Depth=2
	v_add_co_u32 v24, vcc_lo, v28, s22
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v25, null, s23, v29, vcc_lo
	global_load_b128 v[24:27], v[24:25], off
.LBB0_50:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s48, s48
	s_cbranch_execz .LBB0_52
; %bb.51:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v24, s26, v88
	ds_load_b128 v[24:27], v24
.LBB0_52:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s48
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s48, exec_lo
                                        ; implicit-def: $vgpr31
	v_cmpx_le_u32_e64 s42, v97
	s_wait_alu 0xfffe
	s_xor_b32 s48, exec_lo, s48
	s_cbranch_execz .LBB0_54
; %bb.53:                               ;   in Loop: Header=BB0_39 Depth=2
	v_add_co_u32 v28, vcc_lo, v28, s24
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v29, null, s25, v29, vcc_lo
	global_load_b128 v[28:31], v[28:29], off
.LBB0_54:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s48, s48
	s_cbranch_execz .LBB0_56
; %bb.55:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v28, s26, v89
	ds_load_b128 v[28:31], v28
.LBB0_56:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s48
	v_cmp_gt_u32_e32 vcc_lo, s2, v99
	v_dual_mov_b32 v79, 0 :: v_dual_mov_b32 v78, 0
	v_dual_mov_b32 v77, 0 :: v_dual_mov_b32 v76, 0
	v_dual_mov_b32 v63, 0 :: v_dual_mov_b32 v62, 0
	v_dual_mov_b32 v61, 0 :: v_dual_mov_b32 v60, 0
	v_dual_mov_b32 v47, 0 :: v_dual_mov_b32 v46, 0
	v_dual_mov_b32 v45, 0 :: v_dual_mov_b32 v44, 0
	v_dual_mov_b32 v75, 0 :: v_dual_mov_b32 v74, 0
	v_dual_mov_b32 v73, 0 :: v_dual_mov_b32 v72, 0
	v_dual_mov_b32 v59, 0 :: v_dual_mov_b32 v58, 0
	v_dual_mov_b32 v57, 0 :: v_dual_mov_b32 v56, 0
	v_dual_mov_b32 v43, 0 :: v_dual_mov_b32 v42, 0
	v_dual_mov_b32 v41, 0 :: v_dual_mov_b32 v40, 0
	v_dual_mov_b32 v71, 0 :: v_dual_mov_b32 v70, 0
	v_dual_mov_b32 v69, 0 :: v_dual_mov_b32 v68, 0
	v_dual_mov_b32 v55, 0 :: v_dual_mov_b32 v54, 0
	v_dual_mov_b32 v53, 0 :: v_dual_mov_b32 v52, 0
	v_dual_mov_b32 v39, 0 :: v_dual_mov_b32 v38, 0
	v_dual_mov_b32 v37, 0 :: v_dual_mov_b32 v36, 0
	v_dual_mov_b32 v67, 0 :: v_dual_mov_b32 v66, 0
	v_dual_mov_b32 v65, 0 :: v_dual_mov_b32 v64, 0
	v_dual_mov_b32 v51, 0 :: v_dual_mov_b32 v50, 0
	v_dual_mov_b32 v49, 0 :: v_dual_mov_b32 v48, 0
	v_dual_mov_b32 v35, 0 :: v_dual_mov_b32 v34, 0
	v_dual_mov_b32 v33, 0 :: v_dual_mov_b32 v32, 0
	s_and_saveexec_b32 s48, vcc_lo
	s_cbranch_execz .LBB0_37
; %bb.57:                               ;   in Loop: Header=BB0_39 Depth=2
	v_add_co_u32 v44, s49, s18, v99
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v45, null, s19, 0, s49
	s_mov_b32 s49, exec_lo
                                        ; implicit-def: $vgpr35
	v_cmpx_le_u32_e64 s33, v99
	s_wait_alu 0xfffe
	s_xor_b32 s49, exec_lo, s49
	s_cbranch_execz .LBB0_59
; %bb.58:                               ;   in Loop: Header=BB0_39 Depth=2
	global_load_b128 v[32:35], v[44:45], off
.LBB0_59:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s49, s49
	s_cbranch_execz .LBB0_61
; %bb.60:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_loadcnt 0x0
	ds_load_b128 v[32:35], v97 offset:512
.LBB0_61:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s49
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s49, exec_lo
                                        ; implicit-def: $vgpr39
	v_cmpx_le_u32_e64 s38, v99
	s_wait_alu 0xfffe
	s_xor_b32 s49, exec_lo, s49
	s_cbranch_execz .LBB0_63
; %bb.62:                               ;   in Loop: Header=BB0_39 Depth=2
	v_add_co_u32 v36, vcc_lo, v44, s20
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v37, null, 0, v45, vcc_lo
	global_load_b128 v[36:39], v[36:37], off
.LBB0_63:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s49, s49
	s_cbranch_execz .LBB0_65
; %bb.64:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v36, s26, v87
	ds_load_b128 v[36:39], v36 offset:512
.LBB0_65:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s49
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s49, exec_lo
                                        ; implicit-def: $vgpr43
	v_cmpx_le_u32_e64 s40, v99
	s_wait_alu 0xfffe
	s_xor_b32 s49, exec_lo, s49
	s_cbranch_execz .LBB0_67
; %bb.66:                               ;   in Loop: Header=BB0_39 Depth=2
	v_add_co_u32 v40, vcc_lo, v44, s22
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v41, null, s23, v45, vcc_lo
	global_load_b128 v[40:43], v[40:41], off
.LBB0_67:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s49, s49
	s_cbranch_execz .LBB0_69
; %bb.68:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v40, s26, v88
	ds_load_b128 v[40:43], v40 offset:512
.LBB0_69:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s49
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s49, exec_lo
                                        ; implicit-def: $vgpr47
	v_cmpx_le_u32_e64 s42, v99
	s_wait_alu 0xfffe
	s_xor_b32 s49, exec_lo, s49
	s_cbranch_execz .LBB0_71
; %bb.70:                               ;   in Loop: Header=BB0_39 Depth=2
	v_add_co_u32 v44, vcc_lo, v44, s24
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v45, null, s25, v45, vcc_lo
	global_load_b128 v[44:47], v[44:45], off
.LBB0_71:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s49, s49
	s_cbranch_execz .LBB0_73
; %bb.72:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v44, s26, v89
	ds_load_b128 v[44:47], v44 offset:512
.LBB0_73:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s49
	v_cmp_gt_u32_e32 vcc_lo, s2, v98
	v_dual_mov_b32 v79, 0 :: v_dual_mov_b32 v78, 0
	v_dual_mov_b32 v77, 0 :: v_dual_mov_b32 v76, 0
	v_dual_mov_b32 v63, 0 :: v_dual_mov_b32 v62, 0
	v_dual_mov_b32 v61, 0 :: v_dual_mov_b32 v60, 0
	v_dual_mov_b32 v75, 0 :: v_dual_mov_b32 v74, 0
	v_dual_mov_b32 v73, 0 :: v_dual_mov_b32 v72, 0
	v_dual_mov_b32 v59, 0 :: v_dual_mov_b32 v58, 0
	v_dual_mov_b32 v57, 0 :: v_dual_mov_b32 v56, 0
	v_dual_mov_b32 v71, 0 :: v_dual_mov_b32 v70, 0
	v_dual_mov_b32 v69, 0 :: v_dual_mov_b32 v68, 0
	v_dual_mov_b32 v55, 0 :: v_dual_mov_b32 v54, 0
	v_dual_mov_b32 v53, 0 :: v_dual_mov_b32 v52, 0
	v_dual_mov_b32 v67, 0 :: v_dual_mov_b32 v66, 0
	v_dual_mov_b32 v65, 0 :: v_dual_mov_b32 v64, 0
	v_dual_mov_b32 v51, 0 :: v_dual_mov_b32 v50, 0
	v_dual_mov_b32 v49, 0 :: v_dual_mov_b32 v48, 0
	s_and_saveexec_b32 s49, vcc_lo
	s_cbranch_execz .LBB0_36
; %bb.74:                               ;   in Loop: Header=BB0_39 Depth=2
	v_add_co_u32 v60, s50, s18, v98
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v61, null, s19, 0, s50
	s_mov_b32 s50, exec_lo
                                        ; implicit-def: $vgpr51
	v_cmpx_le_u32_e64 s33, v98
	s_wait_alu 0xfffe
	s_xor_b32 s50, exec_lo, s50
	s_cbranch_execz .LBB0_76
; %bb.75:                               ;   in Loop: Header=BB0_39 Depth=2
	global_load_b128 v[48:51], v[60:61], off
.LBB0_76:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s50, s50
	s_cbranch_execz .LBB0_78
; %bb.77:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_loadcnt 0x0
	ds_load_b128 v[48:51], v97 offset:1024
.LBB0_78:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s50
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s50, exec_lo
                                        ; implicit-def: $vgpr55
	v_cmpx_le_u32_e64 s38, v98
	s_wait_alu 0xfffe
	s_xor_b32 s50, exec_lo, s50
	s_cbranch_execz .LBB0_80
; %bb.79:                               ;   in Loop: Header=BB0_39 Depth=2
	v_add_co_u32 v52, vcc_lo, v60, s20
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v53, null, 0, v61, vcc_lo
	global_load_b128 v[52:55], v[52:53], off
.LBB0_80:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s50, s50
	s_cbranch_execz .LBB0_82
; %bb.81:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v52, s26, v87
	ds_load_b128 v[52:55], v52 offset:1024
.LBB0_82:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s50
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s50, exec_lo
                                        ; implicit-def: $vgpr59
	v_cmpx_le_u32_e64 s40, v98
	s_wait_alu 0xfffe
	s_xor_b32 s50, exec_lo, s50
	s_cbranch_execz .LBB0_84
; %bb.83:                               ;   in Loop: Header=BB0_39 Depth=2
	v_add_co_u32 v56, vcc_lo, v60, s22
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v57, null, s23, v61, vcc_lo
	global_load_b128 v[56:59], v[56:57], off
.LBB0_84:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s50, s50
	s_cbranch_execz .LBB0_86
; %bb.85:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v56, s26, v88
	ds_load_b128 v[56:59], v56 offset:1024
.LBB0_86:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s50
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s50, exec_lo
                                        ; implicit-def: $vgpr63
	v_cmpx_le_u32_e64 s42, v98
	s_wait_alu 0xfffe
	s_xor_b32 s50, exec_lo, s50
	s_cbranch_execz .LBB0_88
; %bb.87:                               ;   in Loop: Header=BB0_39 Depth=2
	v_add_co_u32 v60, vcc_lo, v60, s24
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v61, null, s25, v61, vcc_lo
	global_load_b128 v[60:63], v[60:61], off
.LBB0_88:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s50, s50
	s_cbranch_execz .LBB0_90
; %bb.89:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v60, s26, v89
	ds_load_b128 v[60:63], v60 offset:1024
.LBB0_90:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s50
	v_dual_mov_b32 v79, 0 :: v_dual_mov_b32 v78, 0
	v_dual_mov_b32 v77, 0 :: v_dual_mov_b32 v76, 0
	v_dual_mov_b32 v75, 0 :: v_dual_mov_b32 v74, 0
	v_dual_mov_b32 v73, 0 :: v_dual_mov_b32 v72, 0
	v_dual_mov_b32 v71, 0 :: v_dual_mov_b32 v70, 0
	v_dual_mov_b32 v69, 0 :: v_dual_mov_b32 v68, 0
	v_dual_mov_b32 v67, 0 :: v_dual_mov_b32 v66, 0
	v_dual_mov_b32 v65, 0 :: v_dual_mov_b32 v64, 0
	s_mov_b32 s50, exec_lo
	v_cmpx_gt_u32_e64 s2, v96
	s_cbranch_execz .LBB0_35
; %bb.91:                               ;   in Loop: Header=BB0_39 Depth=2
	v_add_co_u32 v76, s51, s18, v96
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v77, null, s19, 0, s51
	s_mov_b32 s51, exec_lo
                                        ; implicit-def: $vgpr67
	v_cmpx_le_u32_e64 s33, v96
	s_wait_alu 0xfffe
	s_xor_b32 s51, exec_lo, s51
	s_cbranch_execz .LBB0_93
; %bb.92:                               ;   in Loop: Header=BB0_39 Depth=2
	global_load_b128 v[64:67], v[76:77], off
                                        ; implicit-def: $vgpr97
.LBB0_93:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s51, s51
	s_cbranch_execz .LBB0_95
; %bb.94:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_loadcnt 0x0
	ds_load_b128 v[64:67], v97 offset:1536
.LBB0_95:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s51
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s51, exec_lo
                                        ; implicit-def: $vgpr71
	v_cmpx_le_u32_e64 s38, v96
	s_wait_alu 0xfffe
	s_xor_b32 s51, exec_lo, s51
	s_cbranch_execz .LBB0_97
; %bb.96:                               ;   in Loop: Header=BB0_39 Depth=2
	v_add_co_u32 v68, vcc_lo, v76, s20
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v69, null, 0, v77, vcc_lo
	global_load_b128 v[68:71], v[68:69], off
.LBB0_97:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s51, s51
	s_cbranch_execz .LBB0_99
; %bb.98:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v68, s26, v87
	ds_load_b128 v[68:71], v68 offset:1536
.LBB0_99:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s51
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s51, exec_lo
                                        ; implicit-def: $vgpr75
	v_cmpx_le_u32_e64 s40, v96
	s_wait_alu 0xfffe
	s_xor_b32 s51, exec_lo, s51
	s_cbranch_execz .LBB0_101
; %bb.100:                              ;   in Loop: Header=BB0_39 Depth=2
	v_add_co_u32 v72, vcc_lo, v76, s22
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v73, null, s23, v77, vcc_lo
	global_load_b128 v[72:75], v[72:73], off
.LBB0_101:                              ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s51, s51
	s_cbranch_execz .LBB0_103
; %bb.102:                              ;   in Loop: Header=BB0_39 Depth=2
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v72, s26, v88
	ds_load_b128 v[72:75], v72 offset:1536
.LBB0_103:                              ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s51
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s51, exec_lo
                                        ; implicit-def: $vgpr79
	v_cmpx_le_u32_e64 s42, v96
	s_wait_alu 0xfffe
	s_xor_b32 s51, exec_lo, s51
	s_cbranch_execz .LBB0_105
; %bb.104:                              ;   in Loop: Header=BB0_39 Depth=2
	v_add_co_u32 v76, vcc_lo, v76, s24
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v77, null, s25, v77, vcc_lo
	global_load_b128 v[76:79], v[76:77], off
.LBB0_105:                              ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s51, s51
	s_cbranch_execz .LBB0_34
; %bb.106:                              ;   in Loop: Header=BB0_39 Depth=2
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v76, s26, v89
	ds_load_b128 v[76:79], v76 offset:1536
	s_branch .LBB0_34
.LBB0_107:                              ;   in Loop: Header=BB0_30 Depth=1
	v_dual_mov_b32 v92, 0 :: v_dual_mov_b32 v93, 0
	v_dual_mov_b32 v94, 0 :: v_dual_mov_b32 v95, 0
.LBB0_108:                              ;   in Loop: Header=BB0_30 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_dpp v0, v95, v95 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_dscnt 0x3
	v_add_f32_dpp v1, v94, v94 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v93, v93 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v92, v92 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_xor_b32_e32 v4, 16, v90
	v_add_f32_dpp v0, v0, v0 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v1, v1, v1 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v3, v3 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cmp_gt_i32_e32 vcc_lo, 32, v4
	v_add_f32_dpp v0, v0, v0 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v1, v1, v1 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v3, v3 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v4, v90, v4, vcc_lo
	v_add_f32_dpp v0, v0, v0 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v9, v1, v1 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v2, v2 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_loadcnt 0x0
	;;#ASMSTART
	;;#ASMEND
	s_wait_dscnt 0x0
	v_lshlrev_b32_e32 v5, 2, v4
	v_add_f32_dpp v4, v3, v3 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	ds_bpermute_b32 v1, v5, v0
	ds_bpermute_b32 v10, v5, v9
	ds_bpermute_b32 v8, v5, v7
	ds_bpermute_b32 v5, v5, v4
	s_and_saveexec_b32 s26, s1
	s_cbranch_execz .LBB0_29
; %bb.109:                              ;   in Loop: Header=BB0_30 Depth=1
	v_lshlrev_b64_e32 v[2:3], 1, v[81:82]
	s_wait_dscnt 0x3
	v_add_f32_e32 v6, v0, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v11, s8, v6
	v_add_co_u32 v0, vcc_lo, s4, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s5, v3, vcc_lo
	s_and_not1_b32 vcc_lo, exec_lo, s21
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_111
; %bb.110:                              ;   in Loop: Header=BB0_30 Depth=1
	global_load_u16 v6, v[0:1], off
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v11, s9, v6, v11 op_sel_hi:[0,1,0]
.LBB0_111:                              ;   in Loop: Header=BB0_30 Depth=1
	v_cvt_f32_f16_e32 v6, v91
	s_wait_dscnt 0x2
	v_add_f32_e32 v9, v9, v10
	v_add_co_u32 v2, vcc_lo, s6, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s7, v3, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_add_f32 v12, v11, v6 :: v_dual_mul_f32 v9, s8, v9
	s_and_not1_b32 vcc_lo, exec_lo, s21
	v_cndmask_b32_e64 v10, v12, v11, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v10, v10
	global_store_b16 v[2:3], v10, off
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_113
; %bb.112:                              ;   in Loop: Header=BB0_30 Depth=1
	v_add_co_u32 v10, vcc_lo, v0, s36
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v11, null, s37, v1, vcc_lo
	global_load_u16 v10, v[10:11], off
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v9, s9, v10, v9 op_sel_hi:[0,1,0]
.LBB0_113:                              ;   in Loop: Header=BB0_30 Depth=1
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_dual_add_f32 v10, v9, v6 :: v_dual_add_f32 v7, v7, v8
	v_add_co_u32 v2, vcc_lo, v2, s34
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s35, v3, vcc_lo
	v_cndmask_b32_e64 v8, v10, v9, s0
	v_mul_f32_e32 v7, s8, v7
	s_and_not1_b32 vcc_lo, exec_lo, s21
	s_delay_alu instid0(VALU_DEP_2)
	v_cvt_f16_f32_e32 v8, v8
	global_store_b16 v[2:3], v8, off
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_115
; %bb.114:                              ;   in Loop: Header=BB0_30 Depth=1
	v_add_co_u32 v8, vcc_lo, v0, s30
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v9, null, s31, v1, vcc_lo
	global_load_u16 v8, v[8:9], off
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v7, s9, v8, v7 op_sel_hi:[0,1,0]
.LBB0_115:                              ;   in Loop: Header=BB0_30 Depth=1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v8, v7, v6
	s_wait_dscnt 0x0
	v_add_f32_e32 v4, v4, v5
	v_add_co_u32 v2, vcc_lo, v2, s34
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s35, v3, vcc_lo
	v_cndmask_b32_e64 v5, v8, v7, s0
	v_mul_f32_e32 v4, s8, v4
	s_and_not1_b32 vcc_lo, exec_lo, s21
	s_delay_alu instid0(VALU_DEP_2)
	v_cvt_f16_f32_e32 v5, v5
	global_store_b16 v[2:3], v5, off
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_28
; %bb.116:                              ;   in Loop: Header=BB0_30 Depth=1
	v_add_co_u32 v0, vcc_lo, v0, s28
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s29, v1, vcc_lo
	global_load_u16 v0, v[0:1], off
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v4, s9, v0, v4 op_sel_hi:[0,1,0]
	s_branch .LBB0_28
.LBB0_117:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s43
	s_mov_b32 s1, 0
.LBB0_118:
	s_wait_alu 0xfffe
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_146
; %bb.119:
	v_cmp_gt_u32_e32 vcc_lo, s3, v80
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_146
; %bb.120:
	s_cmp_eq_u64 s[14:15], 0
	s_mov_b32 s19, 0
	s_cselect_b32 s0, -1, 0
	s_cmp_lg_u64 s[14:15], 0
	v_cmp_eq_u32_e64 s1, 31, v86
	s_cselect_b32 s26, -1, 0
	s_cmp_lg_u32 s2, 0
	v_dual_mov_b32 v81, 0 :: v_dual_add_nc_u32 v84, s41, v85
	s_cselect_b32 s27, -1, 0
	s_add_co_i32 s28, s2, -16
	s_cmp_neq_f32 s9, 0
	v_add_nc_u32_e32 v86, s39, v85
	v_add_nc_u32_e32 v87, s2, v85
	v_mbcnt_lo_u32_b32 v88, -1, 0
	s_mov_b32 s18, s10
	s_mov_b32 s22, s11
	s_wait_alu 0xfffe
	s_mov_b32 s23, s19
	s_cselect_b32 s29, -1, 0
	s_mul_u64 s[10:11], s[18:19], 6
	s_lshl_b64 s[20:21], s[18:19], 2
	s_wait_alu 0xfffe
	s_lshl_b64 s[22:23], s[22:23], 1
	s_lshl_b64 s[24:25], s[18:19], 1
	s_branch .LBB0_123
.LBB0_121:                              ;   in Loop: Header=BB0_123 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v0, v4, v6
	v_cndmask_b32_e64 v4, v0, v4, s0
	v_add_co_u32 v0, vcc_lo, v2, s22
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s23, v3, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f16_f32_e32 v2, v4
	global_store_b16 v[0:1], v2, off
.LBB0_122:                              ;   in Loop: Header=BB0_123 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s18
	v_add_nc_u32_e32 v80, s13, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e32 vcc_lo, s3, v80
	s_or_b32 s19, vcc_lo, s19
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s19
	s_cbranch_execz .LBB0_146
.LBB0_123:                              ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_131 Depth 2
	v_mov_b32_e32 v89, 0
	s_and_not1_b32 vcc_lo, exec_lo, s26
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_125
; %bb.124:                              ;   in Loop: Header=BB0_123 Depth=1
	s_wait_dscnt 0x3
	v_lshlrev_b64_e32 v[0:1], 1, v[80:81]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, s14, v0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s15, v1, vcc_lo
	global_load_u16 v89, v[0:1], off
.LBB0_125:                              ;   in Loop: Header=BB0_123 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s27
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_136
; %bb.126:                              ;   in Loop: Header=BB0_123 Depth=1
	v_mad_co_u64_u32 v[82:83], null, v80, s12, s[16:17]
	v_dual_mov_b32 v93, 0 :: v_dual_mov_b32 v92, 0
	v_dual_mov_b32 v91, 0 :: v_dual_mov_b32 v90, 0
	s_mov_b32 s18, 0
	s_branch .LBB0_131
.LBB0_127:                              ;   in Loop: Header=BB0_131 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s34
.LBB0_128:                              ;   in Loop: Header=BB0_131 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s33
.LBB0_129:                              ;   in Loop: Header=BB0_131 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s31
.LBB0_130:                              ;   in Loop: Header=BB0_131 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s30
	s_wait_loadcnt_dscnt 0x303
	v_dot4_f32_fp8_fp8 v76, v76, v12, v93
	s_wait_dscnt 0x2
	v_dot4_f32_fp8_fp8 v20, v20, v12, v92
	s_wait_dscnt 0x1
	v_dot4_f32_fp8_fp8 v44, v44, v12, v91
	s_wait_dscnt 0x0
	v_dot4_f32_fp8_fp8 v12, v72, v12, v90
	s_addk_co_i32 s18, 0x800
	v_dot4_f32_fp8_fp8 v72, v77, v13, v76
	v_dot4_f32_fp8_fp8 v20, v21, v13, v20
	v_dot4_f32_fp8_fp8 v21, v45, v13, v44
	v_dot4_f32_fp8_fp8 v12, v73, v13, v12
	s_wait_alu 0xfffe
	s_cmp_lt_u32 s18, s2
	v_dot4_f32_fp8_fp8 v13, v78, v14, v72
	v_dot4_f32_fp8_fp8 v20, v22, v14, v20
	v_dot4_f32_fp8_fp8 v21, v46, v14, v21
	v_dot4_f32_fp8_fp8 v12, v74, v14, v12
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v13, v79, v15, v13
	v_dot4_f32_fp8_fp8 v14, v23, v15, v20
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v20, v47, v15, v21
	v_dot4_f32_fp8_fp8 v12, v75, v15, v12
	s_wait_loadcnt 0x2
	v_dot4_f32_fp8_fp8 v13, v40, v8, v13
	v_dot4_f32_fp8_fp8 v14, v52, v8, v14
	v_dot4_f32_fp8_fp8 v15, v60, v8, v20
	v_dot4_f32_fp8_fp8 v8, v68, v8, v12
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v12, v41, v9, v13
	v_dot4_f32_fp8_fp8 v13, v53, v9, v14
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v14, v61, v9, v15
	v_dot4_f32_fp8_fp8 v8, v69, v9, v8
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v9, v42, v10, v12
	v_dot4_f32_fp8_fp8 v12, v54, v10, v13
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v13, v62, v10, v14
	v_dot4_f32_fp8_fp8 v8, v70, v10, v8
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v9, v43, v11, v9
	v_dot4_f32_fp8_fp8 v10, v55, v11, v12
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v12, v63, v11, v13
	v_dot4_f32_fp8_fp8 v8, v71, v11, v8
	s_wait_loadcnt 0x1
	v_dot4_f32_fp8_fp8 v9, v64, v4, v9
	v_dot4_f32_fp8_fp8 v10, v56, v4, v10
	v_dot4_f32_fp8_fp8 v11, v48, v4, v12
	v_dot4_f32_fp8_fp8 v4, v36, v4, v8
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v8, v65, v5, v9
	v_dot4_f32_fp8_fp8 v9, v57, v5, v10
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v10, v49, v5, v11
	v_dot4_f32_fp8_fp8 v4, v37, v5, v4
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v5, v66, v6, v8
	v_dot4_f32_fp8_fp8 v8, v58, v6, v9
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v9, v50, v6, v10
	v_dot4_f32_fp8_fp8 v4, v38, v6, v4
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v5, v67, v7, v5
	v_dot4_f32_fp8_fp8 v6, v59, v7, v8
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v8, v51, v7, v9
	v_dot4_f32_fp8_fp8 v4, v39, v7, v4
	s_wait_loadcnt 0x0
	v_dot4_f32_fp8_fp8 v5, v32, v0, v5
	v_dot4_f32_fp8_fp8 v6, v28, v0, v6
	v_dot4_f32_fp8_fp8 v7, v24, v0, v8
	v_dot4_f32_fp8_fp8 v0, v16, v0, v4
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v4, v33, v1, v5
	v_dot4_f32_fp8_fp8 v5, v29, v1, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v6, v25, v1, v7
	v_dot4_f32_fp8_fp8 v0, v17, v1, v0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v1, v34, v2, v4
	v_dot4_f32_fp8_fp8 v4, v30, v2, v5
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v5, v26, v2, v6
	v_dot4_f32_fp8_fp8 v0, v18, v2, v0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v93, v35, v3, v1
	v_dot4_f32_fp8_fp8 v92, v31, v3, v4
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v91, v27, v3, v5
	v_dot4_f32_fp8_fp8 v90, v19, v3, v0
	s_cbranch_scc0 .LBB0_137
.LBB0_131:                              ;   Parent Loop BB0_123 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu 0xfffe
	v_dual_mov_b32 v41, 0 :: v_dual_add_nc_u32 v94, s18, v85
	v_dual_mov_b32 v42, 0 :: v_dual_mov_b32 v67, 0
	v_dual_mov_b32 v64, 0 :: v_dual_mov_b32 v33, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v43, 0 :: v_dual_add_nc_u32 v100, 0x200, v94
	v_dual_mov_b32 v65, 0 :: v_dual_add_nc_u32 v96, 0x400, v94
	v_min_u32_e32 v0, s28, v94
	v_dual_mov_b32 v40, 0 :: v_dual_add_nc_u32 v95, 0x600, v94
	v_min_u32_e32 v2, s28, v100
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_min_u32_e32 v4, s28, v96
	v_add_co_u32 v0, vcc_lo, v82, v0
	s_delay_alu instid0(VALU_DEP_4)
	v_min_u32_e32 v6, s28, v95
	s_wait_dscnt 0x3
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, 0, v83, vcc_lo
	v_add_co_u32 v2, vcc_lo, v82, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, 0, v83, vcc_lo
	v_add_co_u32 v4, vcc_lo, v82, v4
	s_wait_dscnt 0x0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v5, null, 0, v83, vcc_lo
	v_add_co_u32 v16, vcc_lo, v82, v6
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v17, null, 0, v83, vcc_lo
	s_clause 0x3
	global_load_b128 v[12:15], v[0:1], off
	global_load_b128 v[8:11], v[2:3], off
	global_load_b128 v[4:7], v[4:5], off
	global_load_b128 v[0:3], v[16:17], off
	v_cmp_gt_u32_e32 vcc_lo, s2, v94
	v_dual_mov_b32 v66, 0 :: v_dual_mov_b32 v35, 0
	v_dual_mov_b32 v32, 0 :: v_dual_mov_b32 v21, 0
	v_dual_mov_b32 v34, 0 :: v_dual_mov_b32 v23, 0
	v_dual_mov_b32 v20, 0 :: v_dual_mov_b32 v53, 0
	v_dual_mov_b32 v22, 0 :: v_dual_mov_b32 v55, 0
	v_dual_mov_b32 v52, 0 :: v_dual_mov_b32 v57, 0
	v_dual_mov_b32 v54, 0 :: v_dual_mov_b32 v59, 0
	v_dual_mov_b32 v56, 0 :: v_dual_mov_b32 v29, 0
	v_dual_mov_b32 v58, 0 :: v_dual_mov_b32 v31, 0
	v_dual_mov_b32 v28, 0 :: v_dual_mov_b32 v45, 0
	v_dual_mov_b32 v30, 0 :: v_dual_mov_b32 v47, 0
	v_dual_mov_b32 v44, 0 :: v_dual_mov_b32 v61, 0
	v_dual_mov_b32 v46, 0 :: v_dual_mov_b32 v63, 0
	v_dual_mov_b32 v60, 0 :: v_dual_mov_b32 v49, 0
	v_dual_mov_b32 v62, 0 :: v_dual_mov_b32 v51, 0
	v_dual_mov_b32 v48, 0 :: v_dual_mov_b32 v25, 0
	v_dual_mov_b32 v50, 0 :: v_dual_mov_b32 v27, 0
	v_dual_mov_b32 v24, 0 :: v_dual_mov_b32 v73, 0
	v_dual_mov_b32 v26, 0 :: v_dual_mov_b32 v75, 0
	v_dual_mov_b32 v72, 0 :: v_dual_mov_b32 v69, 0
	v_dual_mov_b32 v74, 0 :: v_dual_mov_b32 v71, 0
	v_dual_mov_b32 v68, 0 :: v_dual_mov_b32 v37, 0
	v_dual_mov_b32 v70, 0 :: v_dual_mov_b32 v39, 0
	v_dual_mov_b32 v36, 0 :: v_dual_mov_b32 v17, 0
	v_dual_mov_b32 v38, 0 :: v_dual_mov_b32 v19, 0
	v_dual_mov_b32 v16, 0 :: v_dual_mov_b32 v79, 0
	v_dual_mov_b32 v18, 0 :: v_dual_mov_b32 v77, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v76, 0
	s_and_saveexec_b32 s30, vcc_lo
	s_cbranch_execz .LBB0_130
; %bb.132:                              ;   in Loop: Header=BB0_131 Depth=2
	v_dual_mov_b32 v18, 0 :: v_dual_add_nc_u32 v99, s18, v87
	v_dual_mov_b32 v19, 0 :: v_dual_add_nc_u32 v98, s18, v86
	v_dual_mov_b32 v16, 0 :: v_dual_add_nc_u32 v97, s18, v84
	ds_load_b128 v[76:79], v94
	ds_load_b128 v[20:23], v99
	ds_load_b128 v[44:47], v98
	ds_load_b128 v[72:75], v97
	v_cmp_gt_u32_e32 vcc_lo, s2, v100
	v_dual_mov_b32 v17, 0 :: v_dual_mov_b32 v38, 0
	v_dual_mov_b32 v39, 0 :: v_dual_mov_b32 v36, 0
	v_dual_mov_b32 v37, 0 :: v_dual_mov_b32 v70, 0
	v_dual_mov_b32 v71, 0 :: v_dual_mov_b32 v68, 0
	v_dual_mov_b32 v69, 0 :: v_dual_mov_b32 v26, 0
	v_dual_mov_b32 v27, 0 :: v_dual_mov_b32 v24, 0
	v_dual_mov_b32 v25, 0 :: v_dual_mov_b32 v50, 0
	v_dual_mov_b32 v51, 0 :: v_dual_mov_b32 v48, 0
	v_dual_mov_b32 v49, 0 :: v_dual_mov_b32 v62, 0
	v_dual_mov_b32 v63, 0 :: v_dual_mov_b32 v60, 0
	v_dual_mov_b32 v61, 0 :: v_dual_mov_b32 v30, 0
	v_dual_mov_b32 v31, 0 :: v_dual_mov_b32 v28, 0
	v_dual_mov_b32 v29, 0 :: v_dual_mov_b32 v58, 0
	v_dual_mov_b32 v59, 0 :: v_dual_mov_b32 v56, 0
	v_dual_mov_b32 v57, 0 :: v_dual_mov_b32 v54, 0
	v_dual_mov_b32 v55, 0 :: v_dual_mov_b32 v52, 0
	v_dual_mov_b32 v53, 0 :: v_dual_mov_b32 v34, 0
	v_dual_mov_b32 v35, 0 :: v_dual_mov_b32 v32, 0
	v_dual_mov_b32 v33, 0 :: v_dual_mov_b32 v66, 0
	v_dual_mov_b32 v67, 0 :: v_dual_mov_b32 v64, 0
	v_dual_mov_b32 v65, 0 :: v_dual_mov_b32 v42, 0
	v_dual_mov_b32 v43, 0 :: v_dual_mov_b32 v40, 0
	v_mov_b32_e32 v41, 0
	s_and_saveexec_b32 s31, vcc_lo
	s_cbranch_execz .LBB0_129
; %bb.133:                              ;   in Loop: Header=BB0_131 Depth=2
	ds_load_b128 v[40:43], v94 offset:512
	ds_load_b128 v[52:55], v99 offset:512
	ds_load_b128 v[60:63], v98 offset:512
	ds_load_b128 v[68:71], v97 offset:512
	v_cmp_gt_u32_e32 vcc_lo, s2, v96
	v_dual_mov_b32 v19, 0 :: v_dual_mov_b32 v18, 0
	v_dual_mov_b32 v17, 0 :: v_dual_mov_b32 v16, 0
	v_dual_mov_b32 v39, 0 :: v_dual_mov_b32 v38, 0
	v_dual_mov_b32 v37, 0 :: v_dual_mov_b32 v36, 0
	v_dual_mov_b32 v27, 0 :: v_dual_mov_b32 v26, 0
	v_dual_mov_b32 v25, 0 :: v_dual_mov_b32 v24, 0
	v_dual_mov_b32 v51, 0 :: v_dual_mov_b32 v50, 0
	v_dual_mov_b32 v49, 0 :: v_dual_mov_b32 v48, 0
	v_dual_mov_b32 v31, 0 :: v_dual_mov_b32 v30, 0
	v_dual_mov_b32 v29, 0 :: v_dual_mov_b32 v28, 0
	v_dual_mov_b32 v59, 0 :: v_dual_mov_b32 v58, 0
	v_dual_mov_b32 v57, 0 :: v_dual_mov_b32 v56, 0
	v_dual_mov_b32 v35, 0 :: v_dual_mov_b32 v34, 0
	v_dual_mov_b32 v33, 0 :: v_dual_mov_b32 v32, 0
	v_dual_mov_b32 v67, 0 :: v_dual_mov_b32 v66, 0
	v_dual_mov_b32 v65, 0 :: v_dual_mov_b32 v64, 0
	s_and_saveexec_b32 s33, vcc_lo
	s_cbranch_execz .LBB0_128
; %bb.134:                              ;   in Loop: Header=BB0_131 Depth=2
	ds_load_b128 v[64:67], v94 offset:1024
	ds_load_b128 v[56:59], v99 offset:1024
	ds_load_b128 v[48:51], v98 offset:1024
	ds_load_b128 v[36:39], v97 offset:1024
	v_dual_mov_b32 v19, 0 :: v_dual_mov_b32 v18, 0
	v_dual_mov_b32 v17, 0 :: v_dual_mov_b32 v16, 0
	v_dual_mov_b32 v27, 0 :: v_dual_mov_b32 v26, 0
	v_dual_mov_b32 v25, 0 :: v_dual_mov_b32 v24, 0
	v_dual_mov_b32 v31, 0 :: v_dual_mov_b32 v30, 0
	v_dual_mov_b32 v29, 0 :: v_dual_mov_b32 v28, 0
	v_dual_mov_b32 v35, 0 :: v_dual_mov_b32 v34, 0
	v_dual_mov_b32 v33, 0 :: v_dual_mov_b32 v32, 0
	s_mov_b32 s34, exec_lo
	v_cmpx_gt_u32_e64 s2, v95
	s_cbranch_execz .LBB0_127
; %bb.135:                              ;   in Loop: Header=BB0_131 Depth=2
	ds_load_b128 v[32:35], v94 offset:1536
	ds_load_b128 v[28:31], v99 offset:1536
	ds_load_b128 v[24:27], v98 offset:1536
	ds_load_b128 v[16:19], v97 offset:1536
	s_branch .LBB0_127
.LBB0_136:                              ;   in Loop: Header=BB0_123 Depth=1
	v_dual_mov_b32 v90, 0 :: v_dual_mov_b32 v91, 0
	v_dual_mov_b32 v92, 0 :: v_dual_mov_b32 v93, 0
.LBB0_137:                              ;   in Loop: Header=BB0_123 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_dpp v0, v93, v93 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_dscnt 0x3
	v_add_f32_dpp v1, v92, v92 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v91, v91 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v90, v90 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_xor_b32_e32 v4, 16, v88
	v_add_f32_dpp v0, v0, v0 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v1, v1, v1 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v3, v3 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cmp_gt_i32_e32 vcc_lo, 32, v4
	v_add_f32_dpp v0, v0, v0 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v1, v1, v1 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v3, v3 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v4, v88, v4, vcc_lo
	v_add_f32_dpp v0, v0, v0 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v9, v1, v1 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v2, v2 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_loadcnt 0x0
	;;#ASMSTART
	;;#ASMEND
	s_wait_dscnt 0x0
	v_lshlrev_b32_e32 v5, 2, v4
	v_add_f32_dpp v4, v3, v3 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	ds_bpermute_b32 v1, v5, v0
	ds_bpermute_b32 v10, v5, v9
	ds_bpermute_b32 v8, v5, v7
	ds_bpermute_b32 v5, v5, v4
	s_and_saveexec_b32 s18, s1
	s_cbranch_execz .LBB0_122
; %bb.138:                              ;   in Loop: Header=BB0_123 Depth=1
	v_lshlrev_b64_e32 v[2:3], 1, v[80:81]
	s_wait_dscnt 0x3
	v_add_f32_e32 v6, v0, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v11, s8, v6
	v_add_co_u32 v0, vcc_lo, s4, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s5, v3, vcc_lo
	s_and_not1_b32 vcc_lo, exec_lo, s29
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_140
; %bb.139:                              ;   in Loop: Header=BB0_123 Depth=1
	global_load_u16 v6, v[0:1], off
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v11, s9, v6, v11 op_sel_hi:[0,1,0]
.LBB0_140:                              ;   in Loop: Header=BB0_123 Depth=1
	v_cvt_f32_f16_e32 v6, v89
	s_wait_dscnt 0x2
	v_add_f32_e32 v9, v9, v10
	v_add_co_u32 v2, vcc_lo, s6, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s7, v3, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_add_f32 v12, v11, v6 :: v_dual_mul_f32 v9, s8, v9
	s_and_not1_b32 vcc_lo, exec_lo, s29
	v_cndmask_b32_e64 v10, v12, v11, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v10, v10
	global_store_b16 v[2:3], v10, off
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_142
; %bb.141:                              ;   in Loop: Header=BB0_123 Depth=1
	v_add_co_u32 v10, vcc_lo, v0, s24
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v11, null, s25, v1, vcc_lo
	global_load_u16 v10, v[10:11], off
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v9, s9, v10, v9 op_sel_hi:[0,1,0]
.LBB0_142:                              ;   in Loop: Header=BB0_123 Depth=1
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_dual_add_f32 v10, v9, v6 :: v_dual_add_f32 v7, v7, v8
	v_add_co_u32 v2, vcc_lo, v2, s22
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s23, v3, vcc_lo
	v_cndmask_b32_e64 v8, v10, v9, s0
	v_mul_f32_e32 v7, s8, v7
	s_and_not1_b32 vcc_lo, exec_lo, s29
	s_delay_alu instid0(VALU_DEP_2)
	v_cvt_f16_f32_e32 v8, v8
	global_store_b16 v[2:3], v8, off
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_144
; %bb.143:                              ;   in Loop: Header=BB0_123 Depth=1
	v_add_co_u32 v8, vcc_lo, v0, s20
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v9, null, s21, v1, vcc_lo
	global_load_u16 v8, v[8:9], off
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v7, s9, v8, v7 op_sel_hi:[0,1,0]
.LBB0_144:                              ;   in Loop: Header=BB0_123 Depth=1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v8, v7, v6
	s_wait_dscnt 0x0
	v_add_f32_e32 v4, v4, v5
	v_add_co_u32 v2, vcc_lo, v2, s22
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s23, v3, vcc_lo
	v_cndmask_b32_e64 v5, v8, v7, s0
	v_mul_f32_e32 v4, s8, v4
	s_and_not1_b32 vcc_lo, exec_lo, s29
	s_delay_alu instid0(VALU_DEP_2)
	v_cvt_f16_f32_e32 v5, v5
	global_store_b16 v[2:3], v5, off
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_121
; %bb.145:                              ;   in Loop: Header=BB0_123 Depth=1
	v_add_co_u32 v0, vcc_lo, v0, s10
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s11, v1, vcc_lo
	global_load_u16 v0, v[0:1], off
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v4, s9, v0, v4 op_sel_hi:[0,1,0]
	s_branch .LBB0_121
.LBB0_146:
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wvSplitKQ_vllm_f8_f16_tn_n4_y1u4_gfx1201
		.amdhsa_group_segment_fixed_size 65536
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 96
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 1
		.amdhsa_next_free_vgpr 169
		.amdhsa_next_free_sgpr 52
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size 66
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	wvSplitKQ_vllm_f8_f16_tn_n4_y1u4_gfx1201, .Lfunc_end0-wvSplitKQ_vllm_f8_f16_tn_n4_y1u4_gfx1201
                                        ; -- End function
	.set wvSplitKQ_vllm_f8_f16_tn_n4_y1u4_gfx1201.num_vgpr, 101
	.set wvSplitKQ_vllm_f8_f16_tn_n4_y1u4_gfx1201.num_agpr, 0
	.set wvSplitKQ_vllm_f8_f16_tn_n4_y1u4_gfx1201.numbered_sgpr, 52
	.set wvSplitKQ_vllm_f8_f16_tn_n4_y1u4_gfx1201.num_named_barrier, 0
	.set wvSplitKQ_vllm_f8_f16_tn_n4_y1u4_gfx1201.private_seg_size, 0
	.set wvSplitKQ_vllm_f8_f16_tn_n4_y1u4_gfx1201.uses_vcc, 1
	.set wvSplitKQ_vllm_f8_f16_tn_n4_y1u4_gfx1201.uses_flat_scratch, 0
	.set wvSplitKQ_vllm_f8_f16_tn_n4_y1u4_gfx1201.has_dyn_sized_stack, 0
	.set wvSplitKQ_vllm_f8_f16_tn_n4_y1u4_gfx1201.has_recursion, 0
	.set wvSplitKQ_vllm_f8_f16_tn_n4_y1u4_gfx1201.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 8344
; TotalNumSgprs: 54
; NumVgprs: 101
; ScratchSize: 0
; MemoryBound: 1
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 65536 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 21
; NumSGPRsForWavesPerEU: 54
; NumVGPRsForWavesPerEU: 169
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 1
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_35ffd18c920e0139,@object ; @__hip_cuid_35ffd18c920e0139
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_35ffd18c920e0139
__hip_cuid_35ffd18c920e0139:
	.byte	0                               ; 0x0
	.size	__hip_cuid_35ffd18c920e0139, 1

	.ident	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.0 26014 7b800a19466229b8479a78de19143dc33c3ab9b5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_35ffd18c920e0139
	.amdgpu_metadata
---
custom.config:
  Source:
    Origin: vllm
    Repository: "https://github.com/vllm-project/vllm"
  Version: 392ed873c7fed4c41fafaad022f83f7d0fe9f57a
  Features:
    SupportsUserArgs: false
    SupportsBias: true
    SupportsActivation: false
    SupportsScaleAlpha: false
    SupportsGSU: false
  InternalSupportParams:
    KernArgsVersion: 0
  ProblemType:
    OperationType: GEMM
    DataType: F8
    DestDataType: h
    ComputeDataType: s
    HighPrecisionAccumulate: True
    TransposeA: True
    TransposeB: False
    UseBeta: True
    Batched: True
    UseScaleAB: Scalar
    UseBias: 1
    BiasSrc: D
    BiasDataTypeList: ['h']
    Activation: False
    UseScaleAlphaVec: 0
  CustomKernel:
    args: [ { type: int32, semantic: SizeSum },
            { type: int32, semantic: SizeFree0 },
            { type: address, semantic: AddressA },
            { type: address, semantic: AddressB },
            { type: int32, semantic: ComputeUnits },
            { type: int32, semantic: SizeFree1 },
            { type: int32, semantic: StrideA0 },
            { type: int32, semantic: StrideB0 },
            { type: address, semantic: AddressC },
            { type: address, semantic: AddressD },
            { type: float32, semantic: Alpha },
            { type: float32, semantic: Beta },
            { type: int32, semantic: StrideC0 },
            { type: int32, semantic: StrideD0 },
            { type: address, semantic: AddressScaleA },
            { type: address, semantic: AddressScaleB },
            { type: address, semantic: AddressBias } ]
    macrotile: [32, 16, 16]
    threads: [32, 16, 1]
    grid: [ComputeUnits, One, One]
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 16, 1]
  EnableMatrixInstruction: True
  MIWaveTile: [1, 1]
  AssertSummationElementMultiple: 16
  AssertSizeEqual: { 1: 4, 2: 1 }
  AssertStrideAEqual: { 0: 1 }
  AssertStrideBEqual: { 0: 1 }
  AssertStrideCEqual: { 0: 1 }
  AssertStrideDEqual: { 0: 1 }
  StaggerU: 0
  WavefrontSize: 32
amdhsa.kernels:
  - .args:
      - .offset:         0
        .size:           4
        .value_kind:     by_value
      - .offset:         4
        .size:           4
        .value_kind:     by_value
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           4
        .value_kind:     by_value
      - .offset:         28
        .size:           4
        .value_kind:     by_value
      - .offset:         32
        .size:           4
        .value_kind:     by_value
      - .offset:         36
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         40
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         48
        .size:           8
        .value_kind:     global_buffer
      - .offset:         56
        .size:           4
        .value_kind:     by_value
      - .offset:         60
        .size:           4
        .value_kind:     by_value
      - .offset:         64
        .size:           4
        .value_kind:     by_value
      - .offset:         68
        .size:           4
        .value_kind:     by_value
      - .actual_access:  read_only
        .address_space:  global
        .offset:         72
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         80
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         88
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 65536
    .kernarg_segment_align: 8
    .kernarg_segment_size: 96
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 512
    .name:           wvSplitKQ_vllm_f8_f16_tn_n4_y1u4_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count:     54
    .sgpr_spill_count: 0
    .symbol:         wvSplitKQ_vllm_f8_f16_tn_n4_y1u4_gfx1201.kd
    .vgpr_count:     101
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 1
...

	.end_amdgpu_metadata
