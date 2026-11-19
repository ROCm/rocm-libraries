	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 4
	.text
	.protected	wvSplitKQ_vllm_f8_bf16_tn_n4_y2u4_gfx1201 ; -- Begin function wvSplitKQ_vllm_f8_bf16_tn_n4_y2u4_gfx1201
	.globl	wvSplitKQ_vllm_f8_bf16_tn_n4_y2u4_gfx1201
	.p2align	8
	.type	wvSplitKQ_vllm_f8_bf16_tn_n4_y2u4_gfx1201,@function
wvSplitKQ_vllm_f8_bf16_tn_n4_y2u4_gfx1201: ; @wvSplitKQ_vllm_f8_bf16_tn_n4_y2u4_gfx1201
; %bb.0:
	s_clause 0x1
	s_load_b64 s[12:13], s[0:1], 0x0
	s_load_b64 s[20:21], s[0:1], 0x10
	v_and_b32_e32 v104, 0x3ff, v0
	v_bfe_u32 v2, v0, 10, 10
	s_mov_b32 s23, 0
	s_mov_b32 s4, exec_lo
	s_mov_b32 s3, s23
	v_lshlrev_b32_e32 v103, 4, v104
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)
	v_lshl_add_u32 v3, v2, 9, v103
	s_wait_kmcnt 0x0
	s_cmp_lt_u32 s12, 0x10000
	s_mov_b32 s2, s12
	s_cselect_b32 s34, s12, 0x10000
	v_cmpx_gt_u32_e64 s34, v3
	s_cbranch_execz .LBB0_9
; %bb.1:
	global_load_b128 v[4:7], v3, s[20:21]
	v_add_nc_u32_e32 v0, 0x2000, v3
	s_mov_b32 s5, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b128 v3, v[4:7]
	v_cmpx_gt_u32_e64 s34, v0
	s_xor_b32 s5, exec_lo, s5
	s_cbranch_execz .LBB0_9
; %bb.2:
	v_add_co_u32 v0, s5, s20, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v1, null, s21, 0, s5
	v_add_nc_u32_e32 v8, 0x4000, v3
	s_mov_b32 s5, exec_lo
	global_load_b128 v[4:7], v[0:1], off offset:8192
	s_wait_loadcnt 0x0
	ds_store_b128 v3, v[4:7] offset:8192
	v_cmpx_gt_u32_e64 s34, v8
	s_wait_alu 0xfffe
	s_xor_b32 s5, exec_lo, s5
	s_cbranch_execz .LBB0_9
; %bb.3:
	global_load_b128 v[4:7], v[0:1], off offset:16384
	v_add_nc_u32_e32 v8, 0x6000, v3
	s_mov_b32 s5, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b128 v3, v[4:7] offset:16384
	v_cmpx_gt_u32_e64 s34, v8
	s_wait_alu 0xfffe
	s_xor_b32 s5, exec_lo, s5
	s_cbranch_execz .LBB0_9
; %bb.4:
	global_load_b128 v[4:7], v[0:1], off offset:24576
	v_add_nc_u32_e32 v8, 0x8000, v3
	s_mov_b32 s5, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b128 v3, v[4:7] offset:24576
	v_cmpx_gt_u32_e64 s34, v8
	s_wait_alu 0xfffe
	s_xor_b32 s5, exec_lo, s5
	s_cbranch_execz .LBB0_9
; %bb.5:
	global_load_b128 v[4:7], v[0:1], off offset:32768
	v_add_nc_u32_e32 v8, 0xa000, v3
	s_mov_b32 s5, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b128 v3, v[4:7] offset:32768
	v_cmpx_gt_u32_e64 s34, v8
	s_wait_alu 0xfffe
	s_xor_b32 s5, exec_lo, s5
	s_cbranch_execz .LBB0_9
; %bb.6:
	global_load_b128 v[4:7], v[0:1], off offset:40960
	v_add_nc_u32_e32 v8, 0xc000, v3
	s_mov_b32 s5, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b128 v3, v[4:7] offset:40960
	v_cmpx_gt_u32_e64 s34, v8
	s_wait_alu 0xfffe
	s_xor_b32 s5, exec_lo, s5
	s_cbranch_execz .LBB0_9
; %bb.7:
	global_load_b128 v[4:7], v[0:1], off offset:49152
	v_add_nc_u32_e32 v8, 0xe000, v3
	s_mov_b32 s5, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b128 v3, v[4:7] offset:49152
	v_cmpx_gt_u32_e64 s34, v8
	s_wait_alu 0xfffe
	s_xor_b32 s5, exec_lo, s5
	s_cbranch_execz .LBB0_9
; %bb.8:
	global_load_b128 v[4:7], v[0:1], off offset:57344
	s_wait_loadcnt 0x0
	ds_store_b128 v3, v[4:7] offset:57344
.LBB0_9:
	s_or_b32 exec_lo, exec_lo, s4
	s_clause 0x3
	s_load_b64 s[14:15], s[0:1], 0x20
	s_load_b64 s[16:17], s[0:1], 0x8
	s_load_b64 s[18:19], s[0:1], 0x58
	s_load_b128 s[4:7], s[0:1], 0x48
	s_sub_nc_u64 s[8:9], 0x10000, s[2:3]
	s_mov_b32 s10, exec_lo
	v_cmp_lt_u64_e64 s9, s[8:9], s[2:3]
	s_and_b32 s9, s9, exec_lo
	s_cselect_b32 s8, s8, s2
	s_cmp_lt_u32 s12, 0x10000
	s_wait_alu 0xfffe
	s_cselect_b32 s35, s8, 0
	s_wait_kmcnt 0x0
	s_mov_b32 s22, s15
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_nc_u64 s[8:9], s[20:21], s[22:23]
	s_wait_alu 0xfffe
	v_cmpx_gt_u32_e64 s35, v3
	s_cbranch_execz .LBB0_12
; %bb.10:
	v_mov_b32_e32 v0, v3
	s_mov_b32 s11, 0
.LBB0_11:                               ; =>This Inner Loop Header: Depth=1
	global_load_b128 v[4:7], v0, s[8:9]
	v_add_nc_u32_e32 v1, s12, v0
	v_add_nc_u32_e32 v0, 0x2000, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e32 vcc_lo, s35, v0
	s_or_b32 s11, vcc_lo, s11
	s_wait_loadcnt 0x0
	ds_store_b128 v1, v[4:7]
	s_and_not1_b32 exec_lo, exec_lo, s11
	s_cbranch_execnz .LBB0_11
.LBB0_12:
	s_or_b32 exec_lo, exec_lo, s10
	s_lshl_b64 s[10:11], s[2:3], 1
	s_add_nc_u64 s[8:9], s[8:9], s[22:23]
	s_sub_nc_u64 s[10:11], 0x10000, s[10:11]
	s_delay_alu instid0(SALU_CYCLE_1)
	v_cmp_lt_u64_e64 s11, s[10:11], s[2:3]
	s_and_b32 s11, s11, exec_lo
	s_cselect_b32 s10, s10, s2
	s_cmp_lt_u32 s12, 0x8000
	s_mov_b32 s11, exec_lo
	s_wait_alu 0xfffe
	s_cselect_b32 s37, s10, 0
	s_mov_b32 s10, 0
	s_lshl_b32 s36, s12, 1
	v_cmpx_gt_u32_e64 s37, v3
	s_cbranch_execz .LBB0_15
; %bb.13:
	v_mov_b32_e32 v0, v3
.LBB0_14:                               ; =>This Inner Loop Header: Depth=1
	global_load_b128 v[4:7], v0, s[8:9]
	v_add_nc_u32_e32 v1, s36, v0
	v_add_nc_u32_e32 v0, 0x2000, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e32 vcc_lo, s37, v0
	s_wait_alu 0xfffe
	s_or_b32 s10, vcc_lo, s10
	s_wait_loadcnt 0x0
	ds_store_b128 v1, v[4:7]
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s10
	s_cbranch_execnz .LBB0_14
.LBB0_15:
	s_or_b32 exec_lo, exec_lo, s11
	s_mul_u64 s[10:11], s[2:3], -3
	s_mul_i32 s38, s12, 3
	s_wait_alu 0xfffe
	s_add_nc_u64 s[10:11], s[10:11], 0x10000
	s_wait_alu 0xfffe
	v_cmp_lt_u64_e64 s3, s[10:11], s[2:3]
	s_mov_b32 s11, exec_lo
	s_and_b32 s3, s3, exec_lo
	s_cselect_b32 s2, s10, s2
	s_cmp_lt_u32 s12, 0x5556
	s_mov_b32 s10, 0
	s_wait_alu 0xfffe
	s_cselect_b32 s39, s2, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	v_cmpx_gt_u32_e64 s39, v3
	s_cbranch_execz .LBB0_18
; %bb.16:
	s_add_nc_u64 s[2:3], s[8:9], s[22:23]
.LBB0_17:                               ; =>This Inner Loop Header: Depth=1
	global_load_b128 v[4:7], v3, s[2:3]
	v_add_nc_u32_e32 v0, s38, v3
	v_add_nc_u32_e32 v3, 0x2000, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e32 vcc_lo, s39, v3
	s_or_b32 s10, vcc_lo, s10
	s_wait_loadcnt 0x0
	ds_store_b128 v0, v[4:7]
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s10
	s_cbranch_execnz .LBB0_17
.LBB0_18:
	s_or_b32 exec_lo, exec_lo, s11
	s_load_b32 s2, s[0:1], 0x18
	s_add_co_i32 s15, s13, -1
	s_wait_kmcnt 0x0
	s_lshl_b32 s8, s2, 5
	s_wait_alu 0xfffe
	v_add_co_u32 v0, s3, s8, s15
	s_and_not1_b32 vcc_lo, exec_lo, s3
	s_mov_b32 s3, 4
	v_readfirstlane_b32 s9, v0
	s_cbranch_vccz .LBB0_20
; %bb.19:
	s_cvt_f32_u32 s3, s8
	s_sub_co_i32 s10, 0, s8
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_iflag_f32_e32 v0, s3
	v_readfirstlane_b32 s3, v0
	s_mul_f32 s3, s3, 0x4f7ffffe
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_cvt_u32_f32 s3, s3
	s_wait_alu 0xfffe
	s_mul_i32 s10, s10, s3
	s_wait_alu 0xfffe
	s_mul_hi_u32 s10, s3, s10
	s_wait_alu 0xfffe
	s_add_co_i32 s3, s3, s10
	s_wait_alu 0xfffe
	s_mul_hi_u32 s3, s9, s3
	s_wait_alu 0xfffe
	s_mul_i32 s10, s3, s8
	s_wait_alu 0xfffe
	s_sub_co_i32 s9, s9, s10
	s_add_co_i32 s10, s3, 1
	s_wait_alu 0xfffe
	s_sub_co_i32 s11, s9, s8
	s_cmp_ge_u32 s9, s8
	s_cselect_b32 s3, s10, s3
	s_wait_alu 0xfffe
	s_cselect_b32 s9, s11, s9
	s_add_co_i32 s10, s3, 1
	s_wait_alu 0xfffe
	s_cmp_ge_u32 s9, s8
	s_cselect_b32 s3, s10, s3
	s_wait_alu 0xfffe
	s_mul_i32 s3, s2, s3
	s_wait_alu 0xfffe
	s_lshl_b32 s3, s3, 1
	s_wait_alu 0xfffe
	s_cvt_f32_u32 s8, s3
	s_sub_co_i32 s9, 0, s3
	s_add_co_i32 s10, s3, s15
	s_wait_alu 0xfffe
	v_rcp_iflag_f32_e32 v0, s8
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_2)
	v_readfirstlane_b32 s8, v0
	s_mul_f32 s8, s8, 0x4f7ffffe
	s_wait_alu 0xfffe
	s_cvt_u32_f32 s8, s8
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_2)
	s_mul_i32 s9, s9, s8
	s_wait_alu 0xfffe
	s_mul_hi_u32 s9, s8, s9
	s_wait_alu 0xfffe
	s_add_co_i32 s8, s8, s9
	s_wait_alu 0xfffe
	s_mul_hi_u32 s8, s10, s8
	s_wait_alu 0xfffe
	s_mul_i32 s9, s8, s3
	s_wait_alu 0xfffe
	s_sub_co_i32 s9, s10, s9
	s_add_co_i32 s10, s8, 1
	s_wait_alu 0xfffe
	s_sub_co_i32 s11, s9, s3
	s_cmp_ge_u32 s9, s3
	s_cselect_b32 s8, s10, s8
	s_wait_alu 0xfffe
	s_cselect_b32 s9, s11, s9
	s_add_co_i32 s10, s8, 1
	s_wait_alu 0xfffe
	s_cmp_ge_u32 s9, s3
	s_cselect_b32 s3, s10, s8
	s_wait_alu 0xfffe
	s_max_u32 s3, s3, 4
.LBB0_20:
	s_mov_b32 s24, 1.0
	s_cmp_eq_u64 s[4:5], 0
	s_mov_b32 s25, 1.0
	s_cbranch_scc1 .LBB0_22
; %bb.21:
	s_load_b32 s25, s[4:5], 0x0
.LBB0_22:
	s_cmp_eq_u64 s[6:7], 0
	s_cbranch_scc1 .LBB0_24
; %bb.23:
	s_load_b32 s24, s[6:7], 0x0
.LBB0_24:
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_mov_b32 s4, exec_lo
	s_wait_alu 0xfffe
	v_cmpx_gt_u32_e64 s3, v2
	s_cbranch_execz .LBB0_176
; %bb.25:
	s_load_b256 s[4:11], s[0:1], 0x28
	s_cmp_eq_u32 s35, s12
	s_mul_i32 s2, s2, s3
	s_cselect_b32 s26, -1, 0
	s_cmp_eq_u32 s37, s12
	s_cselect_b32 s0, -1, 0
	s_cmp_eq_u32 s39, s12
	s_cselect_b32 s1, -1, 0
	s_cmp_eq_u32 s12, s34
	s_cselect_b32 s27, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_and_b32 s26, s27, s26
	s_and_b32 s0, s26, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s1, s0, s1
	s_mul_i32 s0, s3, ttmp9
	s_wait_kmcnt 0x0
	s_mul_f32 s3, s8, s25
	v_add_lshl_u32 v96, s0, v2, 1
	s_wait_alu 0xfffe
	s_lshl_b32 s8, s2, 1
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_mul_f32 s33, s3, s24
	s_mov_b32 s1, -1
	v_cmp_gt_u32_e64 s0, s13, v96
	s_cbranch_vccz .LBB0_133
; %bb.26:
	s_and_saveexec_b32 s40, s0
	s_cbranch_execz .LBB0_132
; %bb.27:
	s_lshl_b64 s[24:25], s[22:23], 1
	s_cmp_eq_u64 s[18:19], 0
	s_mov_b32 s3, 0
	s_cselect_b32 s0, -1, 0
	s_cmp_lg_u64 s[18:19], 0
	v_cmp_eq_u32_e64 s1, 31, v104
	s_cselect_b32 s41, -1, 0
	s_cmp_lg_u32 s12, 0
	v_dual_mov_b32 v98, 0 :: v_dual_add_nc_u32 v105, s12, v103
	s_cselect_b32 s42, -1, 0
	s_add_co_i32 s43, s12, -16
	s_cmp_neq_f32 s9, 0
	v_dual_mov_b32 v97, v96 :: v_dual_add_nc_u32 v106, s36, v103
	v_add_nc_u32_e32 v107, s38, v103
	v_mbcnt_lo_u32_b32 v108, -1, 0
	s_mov_b32 s2, s10
	s_mov_b32 s30, s11
	s_wait_alu 0xfffe
	s_mov_b32 s31, s3
	s_mul_u64 s[26:27], s[22:23], 3
	s_cselect_b32 s23, -1, 0
	s_lshl_b64 s[28:29], s[2:3], 1
	s_lshl_b64 s[30:31], s[30:31], 1
	s_branch .LBB0_30
.LBB0_28:                               ;   in Loop: Header=BB0_30 Depth=1
	v_lshlrev_b32_e32 v0, 16, v109
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v0, v4, v0
	v_cndmask_b32_e64 v0, v0, v4, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_bfe_u32 v1, v0, 16, 1
	v_or_b32_e32 v4, 0x400000, v0
	v_cmp_u_f32_e32 vcc_lo, v0, v0
	v_add3_u32 v1, v1, v0, 0x7fff
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v0, v1, v4, vcc_lo
	global_store_d16_hi_b16 v[2:3], v0, off offset:2
.LBB0_29:                               ;   in Loop: Header=BB0_30 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s44
	v_add_nc_u32_e32 v97, s8, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e32 vcc_lo, s13, v97
	s_or_b32 s3, vcc_lo, s3
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s3
	s_cbranch_execz .LBB0_132
.LBB0_30:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_39 Depth 2
	v_or_b32_e32 v110, 1, v97
	v_min_u32_e32 v2, s15, v97
	v_mov_b32_e32 v111, 0
	v_mov_b32_e32 v109, 0
	s_and_not1_b32 vcc_lo, exec_lo, s41
	v_min_u32_e32 v0, s15, v110
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_32
; %bb.31:                               ;   in Loop: Header=BB0_30 Depth=1
	v_mov_b32_e32 v3, v98
	s_wait_dscnt 0x7
	v_mov_b32_e32 v1, v98
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[3:4], 1, v[2:3]
	s_wait_dscnt 0x0
	v_lshlrev_b64_e32 v[5:6], 1, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v3, vcc_lo, s18, v3
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v4, null, s19, v4, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v5, vcc_lo, s18, v5
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v6, null, s19, v6, vcc_lo
	s_clause 0x1
	global_load_u16 v111, v[3:4], off
	global_load_u16 v109, v[5:6], off
.LBB0_32:                               ;   in Loop: Header=BB0_30 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s42
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_107
; %bb.33:                               ;   in Loop: Header=BB0_30 Depth=1
	v_mad_co_u64_u32 v[99:100], null, v2, s14, 0
	v_mad_co_u64_u32 v[101:102], null, v0, s14, 0
	v_dual_mov_b32 v113, 0 :: v_dual_mov_b32 v112, 0
	v_dual_mov_b32 v114, 0 :: v_dual_mov_b32 v115, 0
	v_dual_mov_b32 v116, 0 :: v_dual_mov_b32 v117, 0
	v_dual_mov_b32 v118, 0 :: v_dual_mov_b32 v119, 0
	s_mov_b32 s2, 0
	s_branch .LBB0_39
.LBB0_34:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s48
.LBB0_35:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s47
.LBB0_36:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s46
.LBB0_37:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s45
.LBB0_38:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s44
	s_wait_loadcnt_dscnt 0x0
	v_dot4_f32_fp8_fp8 v119, v32, v28, v119
	v_dot4_f32_fp8_fp8 v32, v32, v16, v117
	v_dot4_f32_fp8_fp8 v115, v36, v28, v115
	v_dot4_f32_fp8_fp8 v36, v36, v16, v118
	s_addk_co_i32 s2, 0x800
	v_dot4_f32_fp8_fp8 v117, v33, v29, v119
	v_dot4_f32_fp8_fp8 v32, v33, v17, v32
	v_dot4_f32_fp8_fp8 v33, v40, v28, v116
	v_dot4_f32_fp8_fp8 v28, v44, v28, v113
	v_dot4_f32_fp8_fp8 v36, v37, v17, v36
	v_dot4_f32_fp8_fp8 v113, v34, v30, v117
	v_dot4_f32_fp8_fp8 v32, v34, v18, v32
	v_dot4_f32_fp8_fp8 v34, v37, v29, v115
	v_dot4_f32_fp8_fp8 v33, v41, v29, v33
	v_dot4_f32_fp8_fp8 v28, v45, v29, v28
	v_dot4_f32_fp8_fp8 v40, v40, v16, v114
	v_dot4_f32_fp8_fp8 v37, v35, v31, v113
	v_dot4_f32_fp8_fp8 v29, v38, v30, v34
	v_dot4_f32_fp8_fp8 v34, v38, v18, v36
	v_dot4_f32_fp8_fp8 v32, v35, v19, v32
	v_dot4_f32_fp8_fp8 v33, v42, v30, v33
	v_dot4_f32_fp8_fp8 v28, v46, v30, v28
	v_dot4_f32_fp8_fp8 v29, v39, v31, v29
	v_dot4_f32_fp8_fp8 v30, v39, v19, v34
	v_dot4_f32_fp8_fp8 v16, v44, v16, v112
	v_dot4_f32_fp8_fp8 v35, v41, v17, v40
	v_dot4_f32_fp8_fp8 v33, v43, v31, v33
	v_dot4_f32_fp8_fp8 v28, v47, v31, v28
	v_dot4_f32_fp8_fp8 v31, v48, v20, v37
	v_dot4_f32_fp8_fp8 v32, v48, v24, v32
	v_dot4_f32_fp8_fp8 v29, v56, v20, v29
	v_dot4_f32_fp8_fp8 v30, v56, v24, v30
	v_dot4_f32_fp8_fp8 v16, v45, v17, v16
	v_dot4_f32_fp8_fp8 v35, v42, v18, v35
	v_dot4_f32_fp8_fp8 v17, v49, v21, v31
	v_dot4_f32_fp8_fp8 v31, v49, v25, v32
	v_dot4_f32_fp8_fp8 v29, v57, v21, v29
	v_dot4_f32_fp8_fp8 v30, v57, v25, v30
	v_dot4_f32_fp8_fp8 v16, v46, v18, v16
	v_dot4_f32_fp8_fp8 v34, v43, v19, v35
	v_dot4_f32_fp8_fp8 v17, v50, v22, v17
	v_dot4_f32_fp8_fp8 v18, v50, v26, v31
	v_dot4_f32_fp8_fp8 v29, v58, v22, v29
	v_dot4_f32_fp8_fp8 v30, v58, v26, v30
	v_dot4_f32_fp8_fp8 v16, v47, v19, v16
	v_dot4_f32_fp8_fp8 v17, v51, v23, v17
	v_dot4_f32_fp8_fp8 v18, v51, v27, v18
	v_dot4_f32_fp8_fp8 v19, v59, v23, v29
	v_dot4_f32_fp8_fp8 v29, v59, v27, v30
	v_dot4_f32_fp8_fp8 v30, v52, v20, v33
	v_dot4_f32_fp8_fp8 v31, v52, v24, v34
	v_dot4_f32_fp8_fp8 v20, v60, v20, v28
	v_dot4_f32_fp8_fp8 v16, v60, v24, v16
	v_dot4_f32_fp8_fp8 v17, v64, v8, v17
	v_dot4_f32_fp8_fp8 v24, v53, v21, v30
	v_dot4_f32_fp8_fp8 v28, v53, v25, v31
	v_dot4_f32_fp8_fp8 v20, v61, v21, v20
	v_dot4_f32_fp8_fp8 v16, v61, v25, v16
	v_dot4_f32_fp8_fp8 v18, v64, v12, v18
	v_dot4_f32_fp8_fp8 v17, v65, v9, v17
	v_dot4_f32_fp8_fp8 v21, v54, v22, v24
	v_dot4_f32_fp8_fp8 v24, v54, v26, v28
	v_dot4_f32_fp8_fp8 v20, v62, v22, v20
	v_dot4_f32_fp8_fp8 v16, v62, v26, v16
	v_dot4_f32_fp8_fp8 v18, v65, v13, v18
	v_dot4_f32_fp8_fp8 v17, v66, v10, v17
	v_dot4_f32_fp8_fp8 v21, v55, v23, v21
	v_dot4_f32_fp8_fp8 v22, v55, v27, v24
	v_dot4_f32_fp8_fp8 v20, v63, v23, v20
	v_dot4_f32_fp8_fp8 v16, v63, v27, v16
	v_dot4_f32_fp8_fp8 v18, v66, v14, v18
	v_dot4_f32_fp8_fp8 v17, v67, v11, v17
	v_dot4_f32_fp8_fp8 v19, v68, v8, v19
	v_dot4_f32_fp8_fp8 v23, v68, v12, v29
	v_dot4_f32_fp8_fp8 v21, v76, v8, v21
	v_dot4_f32_fp8_fp8 v22, v76, v12, v22
	v_dot4_f32_fp8_fp8 v18, v67, v15, v18
	v_dot4_f32_fp8_fp8 v8, v72, v8, v20
	v_dot4_f32_fp8_fp8 v12, v72, v12, v16
	v_dot4_f32_fp8_fp8 v19, v69, v9, v19
	v_dot4_f32_fp8_fp8 v23, v69, v13, v23
	v_dot4_f32_fp8_fp8 v21, v77, v9, v21
	v_dot4_f32_fp8_fp8 v22, v77, v13, v22
	v_dot4_f32_fp8_fp8 v16, v84, v0, v17
	v_dot4_f32_fp8_fp8 v17, v84, v4, v18
	v_dot4_f32_fp8_fp8 v8, v73, v9, v8
	v_dot4_f32_fp8_fp8 v9, v73, v13, v12
	v_dot4_f32_fp8_fp8 v19, v70, v10, v19
	v_dot4_f32_fp8_fp8 v23, v70, v14, v23
	v_dot4_f32_fp8_fp8 v21, v78, v10, v21
	v_dot4_f32_fp8_fp8 v22, v78, v14, v22
	v_dot4_f32_fp8_fp8 v12, v85, v1, v16
	v_dot4_f32_fp8_fp8 v13, v85, v5, v17
	v_dot4_f32_fp8_fp8 v8, v74, v10, v8
	v_dot4_f32_fp8_fp8 v9, v74, v14, v9
	v_dot4_f32_fp8_fp8 v19, v71, v11, v19
	v_dot4_f32_fp8_fp8 v23, v71, v15, v23
	v_dot4_f32_fp8_fp8 v21, v79, v11, v21
	v_dot4_f32_fp8_fp8 v22, v79, v15, v22
	v_dot4_f32_fp8_fp8 v10, v86, v2, v12
	v_dot4_f32_fp8_fp8 v12, v86, v6, v13
	v_dot4_f32_fp8_fp8 v8, v75, v11, v8
	v_dot4_f32_fp8_fp8 v9, v75, v15, v9
	v_dot4_f32_fp8_fp8 v18, v80, v0, v19
	v_dot4_f32_fp8_fp8 v119, v87, v3, v10
	v_dot4_f32_fp8_fp8 v117, v87, v7, v12
	v_dot4_f32_fp8_fp8 v10, v80, v4, v23
	v_dot4_f32_fp8_fp8 v11, v88, v0, v21
	v_dot4_f32_fp8_fp8 v12, v88, v4, v22
	v_dot4_f32_fp8_fp8 v0, v92, v0, v8
	v_dot4_f32_fp8_fp8 v4, v92, v4, v9
	v_dot4_f32_fp8_fp8 v16, v81, v1, v18
	v_dot4_f32_fp8_fp8 v8, v81, v5, v10
	v_dot4_f32_fp8_fp8 v9, v89, v1, v11
	v_dot4_f32_fp8_fp8 v10, v89, v5, v12
	v_dot4_f32_fp8_fp8 v0, v93, v1, v0
	v_dot4_f32_fp8_fp8 v1, v93, v5, v4
	v_dot4_f32_fp8_fp8 v13, v82, v2, v16
	v_dot4_f32_fp8_fp8 v4, v82, v6, v8
	v_dot4_f32_fp8_fp8 v5, v90, v2, v9
	v_dot4_f32_fp8_fp8 v8, v90, v6, v10
	v_dot4_f32_fp8_fp8 v0, v94, v2, v0
	v_dot4_f32_fp8_fp8 v1, v94, v6, v1
	v_dot4_f32_fp8_fp8 v115, v83, v3, v13
	v_dot4_f32_fp8_fp8 v118, v83, v7, v4
	v_dot4_f32_fp8_fp8 v116, v91, v3, v5
	v_dot4_f32_fp8_fp8 v114, v91, v7, v8
	v_dot4_f32_fp8_fp8 v113, v95, v3, v0
	v_dot4_f32_fp8_fp8 v112, v95, v7, v1
	s_wait_alu 0xfffe
	s_cmp_ge_u32 s2, s12
	s_cbranch_scc1 .LBB0_108
.LBB0_39:                               ;   Parent Loop BB0_30 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu 0xfffe
	v_dual_mov_b32 v94, 0 :: v_dual_add_nc_u32 v121, s2, v103
	v_dual_mov_b32 v75, 0 :: v_dual_mov_b32 v74, 0
	v_dual_mov_b32 v73, 0 :: v_dual_mov_b32 v72, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_min_u32_e32 v0, s43, v121
	v_dual_mov_b32 v92, 0 :: v_dual_add_nc_u32 v123, 0x200, v121
	v_dual_mov_b32 v95, 0 :: v_dual_add_nc_u32 v122, 0x400, v121
	v_add_co_u32 v2, s44, s16, v0
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v3, null, s17, 0, s44
	v_min_u32_e32 v4, s43, v123
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v0, vcc_lo, v2, v99
	s_wait_dscnt 0x7
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, v3, v100, vcc_lo
	v_add_co_u32 v2, vcc_lo, v2, v101
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, v3, v102, vcc_lo
	v_dual_mov_b32 v93, 0 :: v_dual_add_nc_u32 v120, 0x600, v121
	global_load_b128 v[28:31], v[0:1], off
	s_wait_dscnt 0x5
	global_load_b128 v[16:19], v[2:3], off
	v_add_co_u32 v2, s44, s16, v4
	v_min_u32_e32 v4, s43, v122
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v3, null, s17, 0, s44
	v_min_u32_e32 v8, s43, v120
	v_add_co_u32 v0, vcc_lo, v2, v99
	v_add_co_u32 v6, s44, s16, v4
	s_wait_dscnt 0x1
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v7, null, s17, 0, s44
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, v3, v100, vcc_lo
	v_add_co_u32 v2, vcc_lo, v2, v101
	v_add_co_u32 v8, s44, s16, v8
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, v3, v102, vcc_lo
	v_add_co_u32 v4, vcc_lo, v6, v99
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v9, null, s17, 0, s44
	s_wait_dscnt 0x0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v5, null, v7, v100, vcc_lo
	v_add_co_u32 v6, vcc_lo, v6, v101
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v7, null, v7, v102, vcc_lo
	v_add_co_u32 v32, vcc_lo, v8, v99
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v33, null, v9, v100, vcc_lo
	v_add_co_u32 v34, vcc_lo, v8, v101
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v35, null, v9, v102, vcc_lo
	s_clause 0x5
	global_load_b128 v[20:23], v[0:1], off
	global_load_b128 v[24:27], v[2:3], off
	global_load_b128 v[8:11], v[4:5], off
	global_load_b128 v[12:15], v[6:7], off
	global_load_b128 v[0:3], v[32:33], off
	global_load_b128 v[4:7], v[34:35], off
	v_cmp_gt_u32_e32 vcc_lo, s12, v121
	v_dual_mov_b32 v63, 0 :: v_dual_mov_b32 v62, 0
	v_dual_mov_b32 v61, 0 :: v_dual_mov_b32 v60, 0
	v_dual_mov_b32 v47, 0 :: v_dual_mov_b32 v46, 0
	v_dual_mov_b32 v45, 0 :: v_dual_mov_b32 v44, 0
	v_dual_mov_b32 v91, 0 :: v_dual_mov_b32 v90, 0
	v_dual_mov_b32 v89, 0 :: v_dual_mov_b32 v88, 0
	v_dual_mov_b32 v79, 0 :: v_dual_mov_b32 v78, 0
	v_dual_mov_b32 v77, 0 :: v_dual_mov_b32 v76, 0
	v_dual_mov_b32 v55, 0 :: v_dual_mov_b32 v54, 0
	v_dual_mov_b32 v53, 0 :: v_dual_mov_b32 v52, 0
	v_dual_mov_b32 v43, 0 :: v_dual_mov_b32 v42, 0
	v_dual_mov_b32 v41, 0 :: v_dual_mov_b32 v40, 0
	v_dual_mov_b32 v83, 0 :: v_dual_mov_b32 v82, 0
	v_dual_mov_b32 v81, 0 :: v_dual_mov_b32 v80, 0
	v_dual_mov_b32 v71, 0 :: v_dual_mov_b32 v70, 0
	v_dual_mov_b32 v69, 0 :: v_dual_mov_b32 v68, 0
	v_dual_mov_b32 v59, 0 :: v_dual_mov_b32 v58, 0
	v_dual_mov_b32 v57, 0 :: v_dual_mov_b32 v56, 0
	v_dual_mov_b32 v39, 0 :: v_dual_mov_b32 v38, 0
	v_dual_mov_b32 v37, 0 :: v_dual_mov_b32 v36, 0
	v_dual_mov_b32 v87, 0 :: v_dual_mov_b32 v86, 0
	v_dual_mov_b32 v85, 0 :: v_dual_mov_b32 v84, 0
	v_dual_mov_b32 v67, 0 :: v_dual_mov_b32 v66, 0
	v_dual_mov_b32 v65, 0 :: v_dual_mov_b32 v64, 0
	v_dual_mov_b32 v51, 0 :: v_dual_mov_b32 v50, 0
	v_dual_mov_b32 v49, 0 :: v_dual_mov_b32 v48, 0
	v_dual_mov_b32 v35, 0 :: v_dual_mov_b32 v34, 0
	v_dual_mov_b32 v33, 0 :: v_dual_mov_b32 v32, 0
	s_and_saveexec_b32 s44, vcc_lo
	s_cbranch_execz .LBB0_38
; %bb.40:                               ;   in Loop: Header=BB0_39 Depth=2
	v_add_co_u32 v44, s45, s20, v121
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v45, null, s21, 0, s45
	s_mov_b32 s45, exec_lo
                                        ; implicit-def: $vgpr35
	v_cmpx_le_u32_e64 s34, v121
	s_wait_alu 0xfffe
	s_xor_b32 s45, exec_lo, s45
	s_cbranch_execz .LBB0_42
; %bb.41:                               ;   in Loop: Header=BB0_39 Depth=2
	global_load_b128 v[32:35], v[44:45], off
.LBB0_42:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s45, s45
	s_cbranch_execz .LBB0_44
; %bb.43:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_loadcnt 0x0
	ds_load_b128 v[32:35], v121
.LBB0_44:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s45
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s45, exec_lo
                                        ; implicit-def: $vgpr39
	v_cmpx_le_u32_e64 s35, v121
	s_wait_alu 0xfffe
	s_xor_b32 s45, exec_lo, s45
	s_cbranch_execz .LBB0_46
; %bb.45:                               ;   in Loop: Header=BB0_39 Depth=2
	v_add_co_u32 v36, vcc_lo, v44, s22
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v37, null, 0, v45, vcc_lo
	global_load_b128 v[36:39], v[36:37], off
.LBB0_46:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s45, s45
	s_cbranch_execz .LBB0_48
; %bb.47:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v36, s2, v105
	ds_load_b128 v[36:39], v36
.LBB0_48:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s45
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s45, exec_lo
                                        ; implicit-def: $vgpr43
	v_cmpx_le_u32_e64 s37, v121
	s_wait_alu 0xfffe
	s_xor_b32 s45, exec_lo, s45
	s_cbranch_execz .LBB0_50
; %bb.49:                               ;   in Loop: Header=BB0_39 Depth=2
	v_add_co_u32 v40, vcc_lo, v44, s24
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v41, null, s25, v45, vcc_lo
	global_load_b128 v[40:43], v[40:41], off
.LBB0_50:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s45, s45
	s_cbranch_execz .LBB0_52
; %bb.51:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v40, s2, v106
	ds_load_b128 v[40:43], v40
.LBB0_52:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s45
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s45, exec_lo
                                        ; implicit-def: $vgpr47
	v_cmpx_le_u32_e64 s39, v121
	s_wait_alu 0xfffe
	s_xor_b32 s45, exec_lo, s45
	s_cbranch_execz .LBB0_54
; %bb.53:                               ;   in Loop: Header=BB0_39 Depth=2
	v_add_co_u32 v44, vcc_lo, v44, s26
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v45, null, s27, v45, vcc_lo
	global_load_b128 v[44:47], v[44:45], off
.LBB0_54:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s45, s45
	s_cbranch_execz .LBB0_56
; %bb.55:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v44, s2, v107
	ds_load_b128 v[44:47], v44
.LBB0_56:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s45
	v_cmp_gt_u32_e32 vcc_lo, s12, v123
	v_dual_mov_b32 v95, 0 :: v_dual_mov_b32 v94, 0
	v_dual_mov_b32 v93, 0 :: v_dual_mov_b32 v92, 0
	v_dual_mov_b32 v75, 0 :: v_dual_mov_b32 v74, 0
	v_dual_mov_b32 v73, 0 :: v_dual_mov_b32 v72, 0
	v_dual_mov_b32 v63, 0 :: v_dual_mov_b32 v62, 0
	v_dual_mov_b32 v61, 0 :: v_dual_mov_b32 v60, 0
	v_dual_mov_b32 v91, 0 :: v_dual_mov_b32 v90, 0
	v_dual_mov_b32 v89, 0 :: v_dual_mov_b32 v88, 0
	v_dual_mov_b32 v79, 0 :: v_dual_mov_b32 v78, 0
	v_dual_mov_b32 v77, 0 :: v_dual_mov_b32 v76, 0
	v_dual_mov_b32 v55, 0 :: v_dual_mov_b32 v54, 0
	v_dual_mov_b32 v53, 0 :: v_dual_mov_b32 v52, 0
	v_dual_mov_b32 v83, 0 :: v_dual_mov_b32 v82, 0
	v_dual_mov_b32 v81, 0 :: v_dual_mov_b32 v80, 0
	v_dual_mov_b32 v71, 0 :: v_dual_mov_b32 v70, 0
	v_dual_mov_b32 v69, 0 :: v_dual_mov_b32 v68, 0
	v_dual_mov_b32 v59, 0 :: v_dual_mov_b32 v58, 0
	v_dual_mov_b32 v57, 0 :: v_dual_mov_b32 v56, 0
	v_dual_mov_b32 v87, 0 :: v_dual_mov_b32 v86, 0
	v_dual_mov_b32 v85, 0 :: v_dual_mov_b32 v84, 0
	v_dual_mov_b32 v67, 0 :: v_dual_mov_b32 v66, 0
	v_dual_mov_b32 v65, 0 :: v_dual_mov_b32 v64, 0
	v_dual_mov_b32 v51, 0 :: v_dual_mov_b32 v50, 0
	v_dual_mov_b32 v49, 0 :: v_dual_mov_b32 v48, 0
	s_and_saveexec_b32 s45, vcc_lo
	s_cbranch_execz .LBB0_37
; %bb.57:                               ;   in Loop: Header=BB0_39 Depth=2
	v_add_co_u32 v60, s46, s20, v123
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v61, null, s21, 0, s46
	s_mov_b32 s46, exec_lo
                                        ; implicit-def: $vgpr51
	v_cmpx_le_u32_e64 s34, v123
	s_wait_alu 0xfffe
	s_xor_b32 s46, exec_lo, s46
	s_cbranch_execz .LBB0_59
; %bb.58:                               ;   in Loop: Header=BB0_39 Depth=2
	global_load_b128 v[48:51], v[60:61], off
.LBB0_59:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s46, s46
	s_cbranch_execz .LBB0_61
; %bb.60:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_loadcnt 0x0
	ds_load_b128 v[48:51], v121 offset:512
.LBB0_61:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s46
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s46, exec_lo
                                        ; implicit-def: $vgpr59
	v_cmpx_le_u32_e64 s35, v123
	s_wait_alu 0xfffe
	s_xor_b32 s46, exec_lo, s46
	s_cbranch_execz .LBB0_63
; %bb.62:                               ;   in Loop: Header=BB0_39 Depth=2
	v_add_co_u32 v52, vcc_lo, v60, s22
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v53, null, 0, v61, vcc_lo
	global_load_b128 v[56:59], v[52:53], off
.LBB0_63:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s46, s46
	s_cbranch_execz .LBB0_65
; %bb.64:                               ;   in Loop: Header=BB0_39 Depth=2
	v_add_nc_u32_e32 v52, s2, v105
	s_wait_loadcnt 0x0
	ds_load_b128 v[56:59], v52 offset:512
.LBB0_65:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s46
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s46, exec_lo
                                        ; implicit-def: $vgpr55
	v_cmpx_le_u32_e64 s37, v123
	s_wait_alu 0xfffe
	s_xor_b32 s46, exec_lo, s46
	s_cbranch_execz .LBB0_67
; %bb.66:                               ;   in Loop: Header=BB0_39 Depth=2
	v_add_co_u32 v52, vcc_lo, v60, s24
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v53, null, s25, v61, vcc_lo
	global_load_b128 v[52:55], v[52:53], off
.LBB0_67:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s46, s46
	s_cbranch_execz .LBB0_69
; %bb.68:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v52, s2, v106
	ds_load_b128 v[52:55], v52 offset:512
.LBB0_69:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s46
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s46, exec_lo
                                        ; implicit-def: $vgpr63
	v_cmpx_le_u32_e64 s39, v123
	s_wait_alu 0xfffe
	s_xor_b32 s46, exec_lo, s46
	s_cbranch_execz .LBB0_71
; %bb.70:                               ;   in Loop: Header=BB0_39 Depth=2
	v_add_co_u32 v60, vcc_lo, v60, s26
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v61, null, s27, v61, vcc_lo
	global_load_b128 v[60:63], v[60:61], off
.LBB0_71:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s46, s46
	s_cbranch_execz .LBB0_73
; %bb.72:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v60, s2, v107
	ds_load_b128 v[60:63], v60 offset:512
.LBB0_73:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s46
	v_cmp_gt_u32_e32 vcc_lo, s12, v122
	v_dual_mov_b32 v95, 0 :: v_dual_mov_b32 v94, 0
	v_dual_mov_b32 v93, 0 :: v_dual_mov_b32 v92, 0
	v_dual_mov_b32 v75, 0 :: v_dual_mov_b32 v74, 0
	v_dual_mov_b32 v73, 0 :: v_dual_mov_b32 v72, 0
	v_dual_mov_b32 v91, 0 :: v_dual_mov_b32 v90, 0
	v_dual_mov_b32 v89, 0 :: v_dual_mov_b32 v88, 0
	v_dual_mov_b32 v79, 0 :: v_dual_mov_b32 v78, 0
	v_dual_mov_b32 v77, 0 :: v_dual_mov_b32 v76, 0
	v_dual_mov_b32 v83, 0 :: v_dual_mov_b32 v82, 0
	v_dual_mov_b32 v81, 0 :: v_dual_mov_b32 v80, 0
	v_dual_mov_b32 v71, 0 :: v_dual_mov_b32 v70, 0
	v_dual_mov_b32 v69, 0 :: v_dual_mov_b32 v68, 0
	v_dual_mov_b32 v87, 0 :: v_dual_mov_b32 v86, 0
	v_dual_mov_b32 v85, 0 :: v_dual_mov_b32 v84, 0
	v_dual_mov_b32 v67, 0 :: v_dual_mov_b32 v66, 0
	v_dual_mov_b32 v65, 0 :: v_dual_mov_b32 v64, 0
	s_and_saveexec_b32 s46, vcc_lo
	s_cbranch_execz .LBB0_36
; %bb.74:                               ;   in Loop: Header=BB0_39 Depth=2
	v_add_co_u32 v72, s47, s20, v122
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v73, null, s21, 0, s47
	s_mov_b32 s47, exec_lo
                                        ; implicit-def: $vgpr67
	v_cmpx_le_u32_e64 s34, v122
	s_wait_alu 0xfffe
	s_xor_b32 s47, exec_lo, s47
	s_cbranch_execz .LBB0_76
; %bb.75:                               ;   in Loop: Header=BB0_39 Depth=2
	global_load_b128 v[64:67], v[72:73], off
.LBB0_76:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s47, s47
	s_cbranch_execz .LBB0_78
; %bb.77:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_loadcnt 0x0
	ds_load_b128 v[64:67], v121 offset:1024
.LBB0_78:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s47
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s47, exec_lo
                                        ; implicit-def: $vgpr71
	v_cmpx_le_u32_e64 s35, v122
	s_wait_alu 0xfffe
	s_xor_b32 s47, exec_lo, s47
	s_cbranch_execz .LBB0_80
; %bb.79:                               ;   in Loop: Header=BB0_39 Depth=2
	v_add_co_u32 v68, vcc_lo, v72, s22
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v69, null, 0, v73, vcc_lo
	global_load_b128 v[68:71], v[68:69], off
.LBB0_80:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s47, s47
	s_cbranch_execz .LBB0_82
; %bb.81:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v68, s2, v105
	ds_load_b128 v[68:71], v68 offset:1024
.LBB0_82:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s47
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s47, exec_lo
                                        ; implicit-def: $vgpr79
	v_cmpx_le_u32_e64 s37, v122
	s_wait_alu 0xfffe
	s_xor_b32 s47, exec_lo, s47
	s_cbranch_execz .LBB0_84
; %bb.83:                               ;   in Loop: Header=BB0_39 Depth=2
	v_add_co_u32 v74, vcc_lo, v72, s24
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v75, null, s25, v73, vcc_lo
	global_load_b128 v[76:79], v[74:75], off
.LBB0_84:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s47, s47
	s_cbranch_execz .LBB0_86
; %bb.85:                               ;   in Loop: Header=BB0_39 Depth=2
	v_add_nc_u32_e32 v74, s2, v106
	s_wait_loadcnt 0x0
	ds_load_b128 v[76:79], v74 offset:1024
.LBB0_86:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s47
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s47, exec_lo
                                        ; implicit-def: $vgpr75
	v_cmpx_le_u32_e64 s39, v122
	s_wait_alu 0xfffe
	s_xor_b32 s47, exec_lo, s47
	s_cbranch_execz .LBB0_88
; %bb.87:                               ;   in Loop: Header=BB0_39 Depth=2
	v_add_co_u32 v72, vcc_lo, v72, s26
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v73, null, s27, v73, vcc_lo
	global_load_b128 v[72:75], v[72:73], off
.LBB0_88:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s47, s47
	s_cbranch_execz .LBB0_90
; %bb.89:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v72, s2, v107
	ds_load_b128 v[72:75], v72 offset:1024
.LBB0_90:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s47
	v_dual_mov_b32 v95, 0 :: v_dual_mov_b32 v94, 0
	v_dual_mov_b32 v93, 0 :: v_dual_mov_b32 v92, 0
	v_dual_mov_b32 v91, 0 :: v_dual_mov_b32 v90, 0
	v_dual_mov_b32 v89, 0 :: v_dual_mov_b32 v88, 0
	v_dual_mov_b32 v83, 0 :: v_dual_mov_b32 v82, 0
	v_dual_mov_b32 v81, 0 :: v_dual_mov_b32 v80, 0
	v_dual_mov_b32 v87, 0 :: v_dual_mov_b32 v86, 0
	v_dual_mov_b32 v85, 0 :: v_dual_mov_b32 v84, 0
	s_mov_b32 s47, exec_lo
	v_cmpx_gt_u32_e64 s12, v120
	s_cbranch_execz .LBB0_35
; %bb.91:                               ;   in Loop: Header=BB0_39 Depth=2
	v_add_co_u32 v92, s48, s20, v120
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v93, null, s21, 0, s48
	s_mov_b32 s48, exec_lo
                                        ; implicit-def: $vgpr87
	v_cmpx_le_u32_e64 s34, v120
	s_wait_alu 0xfffe
	s_xor_b32 s48, exec_lo, s48
	s_cbranch_execz .LBB0_93
; %bb.92:                               ;   in Loop: Header=BB0_39 Depth=2
	global_load_b128 v[84:87], v[92:93], off
                                        ; implicit-def: $vgpr121
.LBB0_93:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s48, s48
	s_cbranch_execz .LBB0_95
; %bb.94:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_loadcnt 0x0
	ds_load_b128 v[84:87], v121 offset:1536
.LBB0_95:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s48
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s48, exec_lo
                                        ; implicit-def: $vgpr83
	v_cmpx_le_u32_e64 s35, v120
	s_wait_alu 0xfffe
	s_xor_b32 s48, exec_lo, s48
	s_cbranch_execz .LBB0_97
; %bb.96:                               ;   in Loop: Header=BB0_39 Depth=2
	v_add_co_u32 v80, vcc_lo, v92, s22
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v81, null, 0, v93, vcc_lo
	global_load_b128 v[80:83], v[80:81], off
.LBB0_97:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s48, s48
	s_cbranch_execz .LBB0_99
; %bb.98:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v80, s2, v105
	ds_load_b128 v[80:83], v80 offset:1536
.LBB0_99:                               ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s48
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s48, exec_lo
                                        ; implicit-def: $vgpr91
	v_cmpx_le_u32_e64 s37, v120
	s_wait_alu 0xfffe
	s_xor_b32 s48, exec_lo, s48
	s_cbranch_execz .LBB0_101
; %bb.100:                              ;   in Loop: Header=BB0_39 Depth=2
	v_add_co_u32 v88, vcc_lo, v92, s24
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v89, null, s25, v93, vcc_lo
	global_load_b128 v[88:91], v[88:89], off
.LBB0_101:                              ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s48, s48
	s_cbranch_execz .LBB0_103
; %bb.102:                              ;   in Loop: Header=BB0_39 Depth=2
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v88, s2, v106
	ds_load_b128 v[88:91], v88 offset:1536
.LBB0_103:                              ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s48
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s48, exec_lo
                                        ; implicit-def: $vgpr95
	v_cmpx_le_u32_e64 s39, v120
	s_wait_alu 0xfffe
	s_xor_b32 s48, exec_lo, s48
	s_cbranch_execz .LBB0_105
; %bb.104:                              ;   in Loop: Header=BB0_39 Depth=2
	v_add_co_u32 v92, vcc_lo, v92, s26
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v93, null, s27, v93, vcc_lo
	global_load_b128 v[92:95], v[92:93], off
.LBB0_105:                              ;   in Loop: Header=BB0_39 Depth=2
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s48, s48
	s_cbranch_execz .LBB0_34
; %bb.106:                              ;   in Loop: Header=BB0_39 Depth=2
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v92, s2, v107
	ds_load_b128 v[92:95], v92 offset:1536
	s_branch .LBB0_34
.LBB0_107:                              ;   in Loop: Header=BB0_30 Depth=1
	v_dual_mov_b32 v119, 0 :: v_dual_mov_b32 v118, 0
	v_dual_mov_b32 v117, 0 :: v_dual_mov_b32 v116, 0
	v_dual_mov_b32 v115, 0 :: v_dual_mov_b32 v114, 0
	v_dual_mov_b32 v112, 0 :: v_dual_mov_b32 v113, 0
.LBB0_108:                              ;   in Loop: Header=BB0_30 Depth=1
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_dpp v2, v115, v115 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v119, v119 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_dscnt 0x7
	v_add_f32_dpp v1, v117, v117 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_xor_b32_e32 v3, 16, v108
	v_add_f32_dpp v4, v116, v116 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_dscnt 0x0
	v_add_f32_dpp v5, v114, v114 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v113, v113 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v112, v112 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v0, v0 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v1, v1, v1 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cmp_gt_i32_e32 vcc_lo, 32, v3
	v_add_f32_dpp v4, v4, v4 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v5, v5 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v15, v2, v2 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v118, v118 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v6, v6 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v7, v7 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v3, v108, v3, vcc_lo
	v_add_f32_dpp v0, v0, v0 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v1, v1, v1 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v5, v5 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v6, v6 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v7, v7 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshlrev_b32_e32 v3, 2, v3
	v_add_f32_dpp v0, v0, v0 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v17, v1, v1 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v13, v2, v2 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v11, v4, v4 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v9, v5, v5 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v6, v6 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v7, v7 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	ds_bpermute_b32 v1, v3, v0
	ds_bpermute_b32 v18, v3, v17
	ds_bpermute_b32 v16, v3, v15
	ds_bpermute_b32 v14, v3, v13
	ds_bpermute_b32 v12, v3, v11
	ds_bpermute_b32 v10, v3, v9
	ds_bpermute_b32 v7, v3, v6
	ds_bpermute_b32 v5, v3, v4
	s_wait_loadcnt 0x1
	;;#ASMSTART
	;;#ASMEND
	s_wait_loadcnt 0x0
	;;#ASMSTART
	;;#ASMEND
	s_and_saveexec_b32 s44, s1
	s_cbranch_execz .LBB0_29
; %bb.109:                              ;   in Loop: Header=BB0_30 Depth=1
	v_lshlrev_b64_e32 v[2:3], 1, v[97:98]
	s_wait_dscnt 0x7
	v_add_f32_e32 v8, v0, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v19, s33, v8
	v_add_co_u32 v0, vcc_lo, s4, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s5, v3, vcc_lo
	s_and_not1_b32 vcc_lo, exec_lo, s23
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_111
; %bb.110:                              ;   in Loop: Header=BB0_30 Depth=1
	global_load_u16 v8, v[0:1], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v8, 16, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v19, s9, v8
.LBB0_111:                              ;   in Loop: Header=BB0_30 Depth=1
	v_lshlrev_b32_e32 v8, 16, v111
	v_add_co_u32 v2, vcc_lo, s6, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s7, v3, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v20, v19, v8
	v_cmp_gt_u32_e64 s2, s13, v110
	v_cndmask_b32_e64 v19, v20, v19, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_bfe_u32 v20, v19, 16, 1
	v_or_b32_e32 v21, 0x400000, v19
	v_cmp_u_f32_e32 vcc_lo, v19, v19
	v_add3_u32 v20, v20, v19, 0x7fff
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v19, v20, v21, vcc_lo
	global_store_d16_hi_b16 v[2:3], v19, off
	s_and_saveexec_b32 s45, s2
	s_cbranch_execz .LBB0_115
; %bb.112:                              ;   in Loop: Header=BB0_30 Depth=1
	s_wait_dscnt 0x6
	v_add_f32_e32 v17, v17, v18
	s_and_not1_b32 vcc_lo, exec_lo, s23
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v17, s33, v17
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_114
; %bb.113:                              ;   in Loop: Header=BB0_30 Depth=1
	global_load_u16 v18, v[0:1], off offset:2
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v18, 16, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, s9, v18
.LBB0_114:                              ;   in Loop: Header=BB0_30 Depth=1
	v_lshlrev_b32_e32 v18, 16, v109
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v18, v17, v18
	v_cndmask_b32_e64 v17, v18, v17, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_bfe_u32 v18, v17, 16, 1
	v_or_b32_e32 v19, 0x400000, v17
	v_cmp_u_f32_e32 vcc_lo, v17, v17
	v_add3_u32 v18, v18, v17, 0x7fff
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v17, v18, v19, vcc_lo
	global_store_d16_hi_b16 v[2:3], v17, off offset:2
.LBB0_115:                              ;   in Loop: Header=BB0_30 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s45
	s_wait_dscnt 0x5
	v_add_f32_e32 v15, v15, v16
	v_add_co_u32 v0, vcc_lo, v0, s28
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s29, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_f32_e32 v15, s33, v15
	s_and_not1_b32 vcc_lo, exec_lo, s23
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_117
; %bb.116:                              ;   in Loop: Header=BB0_30 Depth=1
	global_load_u16 v16, v[0:1], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v16, 16, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v15, s9, v16
.LBB0_117:                              ;   in Loop: Header=BB0_30 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_add_f32_e32 v16, v15, v8
	v_add_co_u32 v2, vcc_lo, v2, s30
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s31, v3, vcc_lo
	v_cndmask_b32_e64 v15, v16, v15, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_bfe_u32 v16, v15, 16, 1
	v_or_b32_e32 v17, 0x400000, v15
	v_cmp_u_f32_e32 vcc_lo, v15, v15
	v_add3_u32 v16, v16, v15, 0x7fff
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v15, v16, v17, vcc_lo
	global_store_d16_hi_b16 v[2:3], v15, off
	s_and_saveexec_b32 s45, s2
	s_cbranch_execz .LBB0_121
; %bb.118:                              ;   in Loop: Header=BB0_30 Depth=1
	s_wait_dscnt 0x4
	v_add_f32_e32 v13, v13, v14
	s_and_not1_b32 vcc_lo, exec_lo, s23
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v13, s33, v13
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_120
; %bb.119:                              ;   in Loop: Header=BB0_30 Depth=1
	global_load_u16 v14, v[0:1], off offset:2
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v14, 16, v14
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, s9, v14
.LBB0_120:                              ;   in Loop: Header=BB0_30 Depth=1
	v_lshlrev_b32_e32 v14, 16, v109
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v14, v13, v14
	v_cndmask_b32_e64 v13, v14, v13, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_bfe_u32 v14, v13, 16, 1
	v_or_b32_e32 v15, 0x400000, v13
	v_cmp_u_f32_e32 vcc_lo, v13, v13
	v_add3_u32 v14, v14, v13, 0x7fff
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v13, v14, v15, vcc_lo
	global_store_d16_hi_b16 v[2:3], v13, off offset:2
.LBB0_121:                              ;   in Loop: Header=BB0_30 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s45
	s_wait_dscnt 0x3
	v_add_f32_e32 v11, v11, v12
	v_add_co_u32 v0, vcc_lo, v0, s28
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s29, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_f32_e32 v11, s33, v11
	s_and_not1_b32 vcc_lo, exec_lo, s23
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_123
; %bb.122:                              ;   in Loop: Header=BB0_30 Depth=1
	global_load_u16 v12, v[0:1], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v12, 16, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, s9, v12
.LBB0_123:                              ;   in Loop: Header=BB0_30 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_add_f32_e32 v12, v11, v8
	v_add_co_u32 v2, vcc_lo, v2, s30
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s31, v3, vcc_lo
	v_cndmask_b32_e64 v11, v12, v11, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_bfe_u32 v12, v11, 16, 1
	v_or_b32_e32 v13, 0x400000, v11
	v_cmp_u_f32_e32 vcc_lo, v11, v11
	v_add3_u32 v12, v12, v11, 0x7fff
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v11, v12, v13, vcc_lo
	global_store_d16_hi_b16 v[2:3], v11, off
	s_and_saveexec_b32 s45, s2
	s_cbranch_execz .LBB0_127
; %bb.124:                              ;   in Loop: Header=BB0_30 Depth=1
	s_wait_dscnt 0x2
	v_add_f32_e32 v9, v9, v10
	s_and_not1_b32 vcc_lo, exec_lo, s23
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v9, s33, v9
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_126
; %bb.125:                              ;   in Loop: Header=BB0_30 Depth=1
	global_load_u16 v10, v[0:1], off offset:2
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v10, 16, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, s9, v10
.LBB0_126:                              ;   in Loop: Header=BB0_30 Depth=1
	v_lshlrev_b32_e32 v10, 16, v109
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v10, v9, v10
	v_cndmask_b32_e64 v9, v10, v9, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_bfe_u32 v10, v9, 16, 1
	v_or_b32_e32 v11, 0x400000, v9
	v_cmp_u_f32_e32 vcc_lo, v9, v9
	v_add3_u32 v10, v10, v9, 0x7fff
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v9, v10, v11, vcc_lo
	global_store_d16_hi_b16 v[2:3], v9, off offset:2
.LBB0_127:                              ;   in Loop: Header=BB0_30 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s45
	s_wait_dscnt 0x1
	v_add_f32_e32 v6, v6, v7
	v_add_co_u32 v0, vcc_lo, v0, s28
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s29, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_f32_e32 v6, s33, v6
	s_and_not1_b32 vcc_lo, exec_lo, s23
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_129
; %bb.128:                              ;   in Loop: Header=BB0_30 Depth=1
	global_load_u16 v7, v[0:1], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v7, 16, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v6, s9, v7
.LBB0_129:                              ;   in Loop: Header=BB0_30 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_add_f32_e32 v7, v6, v8
	v_add_co_u32 v2, vcc_lo, v2, s30
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s31, v3, vcc_lo
	v_cndmask_b32_e64 v6, v7, v6, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_bfe_u32 v7, v6, 16, 1
	v_or_b32_e32 v8, 0x400000, v6
	v_cmp_u_f32_e32 vcc_lo, v6, v6
	v_add3_u32 v7, v7, v6, 0x7fff
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v6, v7, v8, vcc_lo
	global_store_d16_hi_b16 v[2:3], v6, off
	s_and_b32 exec_lo, exec_lo, s2
	s_cbranch_execz .LBB0_29
; %bb.130:                              ;   in Loop: Header=BB0_30 Depth=1
	s_wait_dscnt 0x0
	v_add_f32_e32 v4, v4, v5
	s_and_not1_b32 vcc_lo, exec_lo, s23
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v4, s33, v4
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_28
; %bb.131:                              ;   in Loop: Header=BB0_30 Depth=1
	global_load_u16 v0, v[0:1], off offset:2
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v0, 16, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v4, s9, v0
	s_branch .LBB0_28
.LBB0_132:
	s_or_b32 exec_lo, exec_lo, s40
	s_mov_b32 s1, 0
.LBB0_133:
	s_wait_alu 0xfffe
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_176
; %bb.134:
	v_cmp_gt_u32_e32 vcc_lo, s13, v96
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_176
; %bb.135:
	s_cmp_eq_u64 s[18:19], 0
	s_mov_b32 s3, 0
	s_cselect_b32 s0, -1, 0
	s_cmp_lg_u64 s[18:19], 0
	v_cmp_eq_u32_e64 s1, 31, v104
	s_cselect_b32 s22, -1, 0
	s_cmp_lg_u32 s12, 0
	v_dual_mov_b32 v97, 0 :: v_dual_add_nc_u32 v102, s38, v103
	s_cselect_b32 s23, -1, 0
	s_add_co_i32 s24, s12, -16
	s_cmp_neq_f32 s9, 0
	v_add_nc_u32_e32 v104, s36, v103
	v_add_nc_u32_e32 v105, s12, v103
	v_mbcnt_lo_u32_b32 v106, -1, 0
	s_mov_b32 s2, s10
	s_mov_b32 s20, s11
	s_wait_alu 0xfffe
	s_mov_b32 s21, s3
	s_cselect_b32 s25, -1, 0
	s_lshl_b64 s[10:11], s[2:3], 1
	s_wait_alu 0xfffe
	s_lshl_b64 s[20:21], s[20:21], 1
	s_branch .LBB0_138
.LBB0_136:                              ;   in Loop: Header=BB0_138 Depth=1
	v_lshlrev_b32_e32 v0, 16, v107
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v0, v4, v0
	v_cndmask_b32_e64 v0, v0, v4, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_bfe_u32 v1, v0, 16, 1
	v_or_b32_e32 v4, 0x400000, v0
	v_cmp_u_f32_e32 vcc_lo, v0, v0
	v_add3_u32 v1, v1, v0, 0x7fff
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v0, v1, v4, vcc_lo
	global_store_d16_hi_b16 v[2:3], v0, off offset:2
.LBB0_137:                              ;   in Loop: Header=BB0_138 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s26
	v_add_nc_u32_e32 v96, s8, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e32 vcc_lo, s13, v96
	s_or_b32 s3, vcc_lo, s3
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s3
	s_cbranch_execz .LBB0_176
.LBB0_138:                              ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_146 Depth 2
	v_or_b32_e32 v108, 1, v96
	v_min_u32_e32 v2, s15, v96
	v_mov_b32_e32 v109, 0
	v_mov_b32_e32 v107, 0
	s_and_not1_b32 vcc_lo, exec_lo, s22
	v_min_u32_e32 v0, s15, v108
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_140
; %bb.139:                              ;   in Loop: Header=BB0_138 Depth=1
	v_mov_b32_e32 v3, v97
	s_wait_dscnt 0x7
	v_mov_b32_e32 v1, v97
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[3:4], 1, v[2:3]
	s_wait_dscnt 0x0
	v_lshlrev_b64_e32 v[5:6], 1, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v3, vcc_lo, s18, v3
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v4, null, s19, v4, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v5, vcc_lo, s18, v5
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v6, null, s19, v6, vcc_lo
	s_clause 0x1
	global_load_u16 v109, v[3:4], off
	global_load_u16 v107, v[5:6], off
.LBB0_140:                              ;   in Loop: Header=BB0_138 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s23
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_151
; %bb.141:                              ;   in Loop: Header=BB0_138 Depth=1
	v_mad_co_u64_u32 v[98:99], null, v2, s14, 0
	v_mad_co_u64_u32 v[100:101], null, v0, s14, 0
	v_dual_mov_b32 v111, 0 :: v_dual_mov_b32 v110, 0
	v_dual_mov_b32 v112, 0 :: v_dual_mov_b32 v113, 0
	v_dual_mov_b32 v114, 0 :: v_dual_mov_b32 v115, 0
	v_dual_mov_b32 v116, 0 :: v_dual_mov_b32 v117, 0
	s_mov_b32 s2, 0
	s_branch .LBB0_146
.LBB0_142:                              ;   in Loop: Header=BB0_146 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s29
.LBB0_143:                              ;   in Loop: Header=BB0_146 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s28
.LBB0_144:                              ;   in Loop: Header=BB0_146 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s27
.LBB0_145:                              ;   in Loop: Header=BB0_146 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s26
	s_wait_loadcnt_dscnt 0x703
	v_dot4_f32_fp8_fp8 v117, v72, v36, v117
	s_wait_loadcnt 0x6
	v_dot4_f32_fp8_fp8 v72, v72, v16, v115
	s_wait_dscnt 0x2
	v_dot4_f32_fp8_fp8 v113, v76, v36, v113
	v_dot4_f32_fp8_fp8 v76, v76, v16, v116
	s_addk_co_i32 s2, 0x800
	v_dot4_f32_fp8_fp8 v115, v73, v37, v117
	v_dot4_f32_fp8_fp8 v72, v73, v17, v72
	s_wait_dscnt 0x1
	v_dot4_f32_fp8_fp8 v73, v92, v36, v114
	s_wait_dscnt 0x0
	v_dot4_f32_fp8_fp8 v36, v80, v36, v111
	v_dot4_f32_fp8_fp8 v76, v77, v17, v76
	v_dot4_f32_fp8_fp8 v111, v74, v38, v115
	v_dot4_f32_fp8_fp8 v72, v74, v18, v72
	v_dot4_f32_fp8_fp8 v74, v77, v37, v113
	v_dot4_f32_fp8_fp8 v73, v93, v37, v73
	v_dot4_f32_fp8_fp8 v36, v81, v37, v36
	v_dot4_f32_fp8_fp8 v92, v92, v16, v112
	v_dot4_f32_fp8_fp8 v77, v75, v39, v111
	v_dot4_f32_fp8_fp8 v37, v78, v38, v74
	v_dot4_f32_fp8_fp8 v74, v78, v18, v76
	v_dot4_f32_fp8_fp8 v72, v75, v19, v72
	v_dot4_f32_fp8_fp8 v73, v94, v38, v73
	v_dot4_f32_fp8_fp8 v36, v82, v38, v36
	v_dot4_f32_fp8_fp8 v37, v79, v39, v37
	v_dot4_f32_fp8_fp8 v38, v79, v19, v74
	v_dot4_f32_fp8_fp8 v16, v80, v16, v110
	v_dot4_f32_fp8_fp8 v75, v93, v17, v92
	v_dot4_f32_fp8_fp8 v73, v95, v39, v73
	v_dot4_f32_fp8_fp8 v36, v83, v39, v36
	s_wait_loadcnt 0x5
	v_dot4_f32_fp8_fp8 v39, v84, v20, v77
	s_wait_loadcnt 0x4
	v_dot4_f32_fp8_fp8 v72, v84, v24, v72
	v_dot4_f32_fp8_fp8 v37, v88, v20, v37
	v_dot4_f32_fp8_fp8 v38, v88, v24, v38
	v_dot4_f32_fp8_fp8 v16, v81, v17, v16
	v_dot4_f32_fp8_fp8 v75, v94, v18, v75
	v_dot4_f32_fp8_fp8 v17, v85, v21, v39
	v_dot4_f32_fp8_fp8 v39, v85, v25, v72
	v_dot4_f32_fp8_fp8 v37, v89, v21, v37
	v_dot4_f32_fp8_fp8 v38, v89, v25, v38
	v_dot4_f32_fp8_fp8 v16, v82, v18, v16
	v_dot4_f32_fp8_fp8 v74, v95, v19, v75
	v_dot4_f32_fp8_fp8 v17, v86, v22, v17
	v_dot4_f32_fp8_fp8 v18, v86, v26, v39
	v_dot4_f32_fp8_fp8 v37, v90, v22, v37
	v_dot4_f32_fp8_fp8 v38, v90, v26, v38
	v_dot4_f32_fp8_fp8 v16, v83, v19, v16
	v_dot4_f32_fp8_fp8 v17, v87, v23, v17
	v_dot4_f32_fp8_fp8 v18, v87, v27, v18
	v_dot4_f32_fp8_fp8 v19, v91, v23, v37
	v_dot4_f32_fp8_fp8 v37, v91, v27, v38
	v_dot4_f32_fp8_fp8 v38, v64, v20, v73
	v_dot4_f32_fp8_fp8 v39, v64, v24, v74
	v_dot4_f32_fp8_fp8 v20, v68, v20, v36
	v_dot4_f32_fp8_fp8 v16, v68, v24, v16
	s_wait_loadcnt 0x3
	v_dot4_f32_fp8_fp8 v17, v60, v8, v17
	v_dot4_f32_fp8_fp8 v24, v65, v21, v38
	v_dot4_f32_fp8_fp8 v36, v65, v25, v39
	v_dot4_f32_fp8_fp8 v20, v69, v21, v20
	v_dot4_f32_fp8_fp8 v16, v69, v25, v16
	s_wait_loadcnt 0x2
	v_dot4_f32_fp8_fp8 v18, v60, v12, v18
	v_dot4_f32_fp8_fp8 v17, v61, v9, v17
	v_dot4_f32_fp8_fp8 v21, v66, v22, v24
	v_dot4_f32_fp8_fp8 v24, v66, v26, v36
	v_dot4_f32_fp8_fp8 v20, v70, v22, v20
	v_dot4_f32_fp8_fp8 v16, v70, v26, v16
	v_dot4_f32_fp8_fp8 v18, v61, v13, v18
	v_dot4_f32_fp8_fp8 v17, v62, v10, v17
	v_dot4_f32_fp8_fp8 v21, v67, v23, v21
	v_dot4_f32_fp8_fp8 v22, v67, v27, v24
	v_dot4_f32_fp8_fp8 v20, v71, v23, v20
	v_dot4_f32_fp8_fp8 v16, v71, v27, v16
	v_dot4_f32_fp8_fp8 v18, v62, v14, v18
	v_dot4_f32_fp8_fp8 v17, v63, v11, v17
	v_dot4_f32_fp8_fp8 v19, v56, v8, v19
	v_dot4_f32_fp8_fp8 v23, v56, v12, v37
	v_dot4_f32_fp8_fp8 v21, v52, v8, v21
	v_dot4_f32_fp8_fp8 v22, v52, v12, v22
	v_dot4_f32_fp8_fp8 v18, v63, v15, v18
	v_dot4_f32_fp8_fp8 v8, v44, v8, v20
	v_dot4_f32_fp8_fp8 v12, v44, v12, v16
	v_dot4_f32_fp8_fp8 v19, v57, v9, v19
	v_dot4_f32_fp8_fp8 v23, v57, v13, v23
	v_dot4_f32_fp8_fp8 v21, v53, v9, v21
	v_dot4_f32_fp8_fp8 v22, v53, v13, v22
	s_wait_loadcnt 0x1
	v_dot4_f32_fp8_fp8 v16, v48, v0, v17
	s_wait_loadcnt 0x0
	v_dot4_f32_fp8_fp8 v17, v48, v4, v18
	v_dot4_f32_fp8_fp8 v8, v45, v9, v8
	v_dot4_f32_fp8_fp8 v9, v45, v13, v12
	v_dot4_f32_fp8_fp8 v19, v58, v10, v19
	v_dot4_f32_fp8_fp8 v23, v58, v14, v23
	v_dot4_f32_fp8_fp8 v21, v54, v10, v21
	v_dot4_f32_fp8_fp8 v22, v54, v14, v22
	v_dot4_f32_fp8_fp8 v12, v49, v1, v16
	v_dot4_f32_fp8_fp8 v13, v49, v5, v17
	v_dot4_f32_fp8_fp8 v8, v46, v10, v8
	v_dot4_f32_fp8_fp8 v9, v46, v14, v9
	v_dot4_f32_fp8_fp8 v19, v59, v11, v19
	v_dot4_f32_fp8_fp8 v23, v59, v15, v23
	v_dot4_f32_fp8_fp8 v21, v55, v11, v21
	v_dot4_f32_fp8_fp8 v22, v55, v15, v22
	v_dot4_f32_fp8_fp8 v10, v50, v2, v12
	v_dot4_f32_fp8_fp8 v12, v50, v6, v13
	v_dot4_f32_fp8_fp8 v8, v47, v11, v8
	v_dot4_f32_fp8_fp8 v9, v47, v15, v9
	v_dot4_f32_fp8_fp8 v18, v40, v0, v19
	v_dot4_f32_fp8_fp8 v117, v51, v3, v10
	v_dot4_f32_fp8_fp8 v115, v51, v7, v12
	v_dot4_f32_fp8_fp8 v10, v40, v4, v23
	v_dot4_f32_fp8_fp8 v11, v32, v0, v21
	v_dot4_f32_fp8_fp8 v12, v32, v4, v22
	v_dot4_f32_fp8_fp8 v0, v28, v0, v8
	v_dot4_f32_fp8_fp8 v4, v28, v4, v9
	v_dot4_f32_fp8_fp8 v16, v41, v1, v18
	v_dot4_f32_fp8_fp8 v8, v41, v5, v10
	v_dot4_f32_fp8_fp8 v9, v33, v1, v11
	v_dot4_f32_fp8_fp8 v10, v33, v5, v12
	v_dot4_f32_fp8_fp8 v0, v29, v1, v0
	v_dot4_f32_fp8_fp8 v1, v29, v5, v4
	v_dot4_f32_fp8_fp8 v13, v42, v2, v16
	v_dot4_f32_fp8_fp8 v4, v42, v6, v8
	v_dot4_f32_fp8_fp8 v5, v34, v2, v9
	v_dot4_f32_fp8_fp8 v8, v34, v6, v10
	v_dot4_f32_fp8_fp8 v0, v30, v2, v0
	v_dot4_f32_fp8_fp8 v1, v30, v6, v1
	v_dot4_f32_fp8_fp8 v113, v43, v3, v13
	v_dot4_f32_fp8_fp8 v116, v43, v7, v4
	v_dot4_f32_fp8_fp8 v114, v35, v3, v5
	v_dot4_f32_fp8_fp8 v112, v35, v7, v8
	v_dot4_f32_fp8_fp8 v111, v31, v3, v0
	v_dot4_f32_fp8_fp8 v110, v31, v7, v1
	s_wait_alu 0xfffe
	s_cmp_ge_u32 s2, s12
	s_cbranch_scc1 .LBB0_152
.LBB0_146:                              ;   Parent Loop BB0_138 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu 0xfffe
	v_dual_mov_b32 v73, 0 :: v_dual_add_nc_u32 v118, s2, v103
	v_dual_mov_b32 v84, 0 :: v_dual_mov_b32 v85, 0
	v_dual_mov_b32 v86, 0 :: v_dual_mov_b32 v87, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_min_u32_e32 v0, s24, v118
	v_dual_mov_b32 v72, 0 :: v_dual_add_nc_u32 v123, 0x200, v118
	v_dual_mov_b32 v75, 0 :: v_dual_add_nc_u32 v120, 0x400, v118
	v_add_co_u32 v2, s26, s16, v0
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v3, null, s17, 0, s26
	v_min_u32_e32 v4, s24, v123
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v0, vcc_lo, v2, v98
	s_wait_dscnt 0x7
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, v3, v99, vcc_lo
	v_add_co_u32 v2, vcc_lo, v2, v100
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, v3, v101, vcc_lo
	v_dual_mov_b32 v74, 0 :: v_dual_add_nc_u32 v119, 0x600, v118
	global_load_b128 v[36:39], v[0:1], off
	s_wait_dscnt 0x5
	global_load_b128 v[16:19], v[2:3], off
	v_add_co_u32 v2, s26, s16, v4
	v_min_u32_e32 v4, s24, v120
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v3, null, s17, 0, s26
	v_min_u32_e32 v8, s24, v119
	v_add_co_u32 v0, vcc_lo, v2, v98
	v_add_co_u32 v6, s26, s16, v4
	s_wait_dscnt 0x1
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v7, null, s17, 0, s26
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, v3, v99, vcc_lo
	v_add_co_u32 v2, vcc_lo, v2, v100
	v_add_co_u32 v8, s26, s16, v8
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, v3, v101, vcc_lo
	v_add_co_u32 v4, vcc_lo, v6, v98
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v9, null, s17, 0, s26
	s_wait_dscnt 0x0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v5, null, v7, v99, vcc_lo
	v_add_co_u32 v6, vcc_lo, v6, v100
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v7, null, v7, v101, vcc_lo
	v_add_co_u32 v28, vcc_lo, v8, v98
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v29, null, v9, v99, vcc_lo
	v_add_co_u32 v30, vcc_lo, v8, v100
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v31, null, v9, v101, vcc_lo
	s_clause 0x5
	global_load_b128 v[20:23], v[0:1], off
	global_load_b128 v[24:27], v[2:3], off
	global_load_b128 v[8:11], v[4:5], off
	global_load_b128 v[12:15], v[6:7], off
	global_load_b128 v[0:3], v[28:29], off
	global_load_b128 v[4:7], v[30:31], off
	v_cmp_gt_u32_e32 vcc_lo, s12, v118
	v_dual_mov_b32 v60, 0 :: v_dual_mov_b32 v61, 0
	v_dual_mov_b32 v62, 0 :: v_dual_mov_b32 v63, 0
	v_dual_mov_b32 v48, 0 :: v_dual_mov_b32 v49, 0
	v_dual_mov_b32 v50, 0 :: v_dual_mov_b32 v51, 0
	v_dual_mov_b32 v76, 0 :: v_dual_mov_b32 v77, 0
	v_dual_mov_b32 v78, 0 :: v_dual_mov_b32 v79, 0
	v_dual_mov_b32 v88, 0 :: v_dual_mov_b32 v89, 0
	v_dual_mov_b32 v90, 0 :: v_dual_mov_b32 v91, 0
	v_dual_mov_b32 v56, 0 :: v_dual_mov_b32 v57, 0
	v_dual_mov_b32 v58, 0 :: v_dual_mov_b32 v59, 0
	v_dual_mov_b32 v40, 0 :: v_dual_mov_b32 v41, 0
	v_dual_mov_b32 v42, 0 :: v_dual_mov_b32 v43, 0
	v_dual_mov_b32 v92, 0 :: v_dual_mov_b32 v93, 0
	v_dual_mov_b32 v94, 0 :: v_dual_mov_b32 v95, 0
	v_dual_mov_b32 v64, 0 :: v_dual_mov_b32 v65, 0
	v_dual_mov_b32 v66, 0 :: v_dual_mov_b32 v67, 0
	v_dual_mov_b32 v52, 0 :: v_dual_mov_b32 v53, 0
	v_dual_mov_b32 v54, 0 :: v_dual_mov_b32 v55, 0
	v_dual_mov_b32 v32, 0 :: v_dual_mov_b32 v33, 0
	v_dual_mov_b32 v34, 0 :: v_dual_mov_b32 v35, 0
	v_dual_mov_b32 v80, 0 :: v_dual_mov_b32 v81, 0
	v_dual_mov_b32 v82, 0 :: v_dual_mov_b32 v83, 0
	v_dual_mov_b32 v68, 0 :: v_dual_mov_b32 v69, 0
	v_dual_mov_b32 v70, 0 :: v_dual_mov_b32 v71, 0
	v_dual_mov_b32 v44, 0 :: v_dual_mov_b32 v45, 0
	v_dual_mov_b32 v46, 0 :: v_dual_mov_b32 v47, 0
	v_dual_mov_b32 v28, 0 :: v_dual_mov_b32 v29, 0
	v_dual_mov_b32 v30, 0 :: v_dual_mov_b32 v31, 0
	s_and_saveexec_b32 s26, vcc_lo
	s_cbranch_execz .LBB0_145
; %bb.147:                              ;   in Loop: Header=BB0_146 Depth=2
	v_dual_mov_b32 v31, 0 :: v_dual_add_nc_u32 v124, s2, v105
	v_dual_mov_b32 v29, 0 :: v_dual_add_nc_u32 v122, s2, v104
	v_dual_mov_b32 v30, 0 :: v_dual_add_nc_u32 v121, s2, v102
	ds_load_b128 v[72:75], v118
	ds_load_b128 v[76:79], v124
	ds_load_b128 v[92:95], v122
	ds_load_b128 v[80:83], v121
	v_cmp_gt_u32_e32 vcc_lo, s12, v123
	v_dual_mov_b32 v28, 0 :: v_dual_mov_b32 v47, 0
	v_dual_mov_b32 v46, 0 :: v_dual_mov_b32 v45, 0
	v_dual_mov_b32 v44, 0 :: v_dual_mov_b32 v71, 0
	v_dual_mov_b32 v70, 0 :: v_dual_mov_b32 v69, 0
	v_dual_mov_b32 v68, 0 :: v_dual_mov_b32 v35, 0
	v_dual_mov_b32 v34, 0 :: v_dual_mov_b32 v33, 0
	v_dual_mov_b32 v32, 0 :: v_dual_mov_b32 v55, 0
	v_dual_mov_b32 v54, 0 :: v_dual_mov_b32 v53, 0
	v_dual_mov_b32 v52, 0 :: v_dual_mov_b32 v67, 0
	v_dual_mov_b32 v66, 0 :: v_dual_mov_b32 v65, 0
	v_dual_mov_b32 v64, 0 :: v_dual_mov_b32 v43, 0
	v_dual_mov_b32 v42, 0 :: v_dual_mov_b32 v41, 0
	v_dual_mov_b32 v40, 0 :: v_dual_mov_b32 v59, 0
	v_dual_mov_b32 v58, 0 :: v_dual_mov_b32 v57, 0
	v_dual_mov_b32 v56, 0 :: v_dual_mov_b32 v91, 0
	v_dual_mov_b32 v90, 0 :: v_dual_mov_b32 v89, 0
	v_dual_mov_b32 v88, 0 :: v_dual_mov_b32 v51, 0
	v_dual_mov_b32 v50, 0 :: v_dual_mov_b32 v49, 0
	v_dual_mov_b32 v48, 0 :: v_dual_mov_b32 v63, 0
	v_dual_mov_b32 v62, 0 :: v_dual_mov_b32 v61, 0
	v_dual_mov_b32 v60, 0 :: v_dual_mov_b32 v87, 0
	v_dual_mov_b32 v86, 0 :: v_dual_mov_b32 v85, 0
	v_mov_b32_e32 v84, 0
	s_and_saveexec_b32 s27, vcc_lo
	s_cbranch_execz .LBB0_144
; %bb.148:                              ;   in Loop: Header=BB0_146 Depth=2
	ds_load_b128 v[84:87], v118 offset:512
	ds_load_b128 v[88:91], v124 offset:512
	ds_load_b128 v[64:67], v122 offset:512
	ds_load_b128 v[68:71], v121 offset:512
	v_cmp_gt_u32_e32 vcc_lo, s12, v120
	v_dual_mov_b32 v31, 0 :: v_dual_mov_b32 v30, 0
	v_dual_mov_b32 v29, 0 :: v_dual_mov_b32 v28, 0
	v_dual_mov_b32 v47, 0 :: v_dual_mov_b32 v46, 0
	v_dual_mov_b32 v45, 0 :: v_dual_mov_b32 v44, 0
	v_dual_mov_b32 v35, 0 :: v_dual_mov_b32 v34, 0
	v_dual_mov_b32 v33, 0 :: v_dual_mov_b32 v32, 0
	v_dual_mov_b32 v55, 0 :: v_dual_mov_b32 v54, 0
	v_dual_mov_b32 v53, 0 :: v_dual_mov_b32 v52, 0
	v_dual_mov_b32 v43, 0 :: v_dual_mov_b32 v42, 0
	v_dual_mov_b32 v41, 0 :: v_dual_mov_b32 v40, 0
	v_dual_mov_b32 v59, 0 :: v_dual_mov_b32 v58, 0
	v_dual_mov_b32 v57, 0 :: v_dual_mov_b32 v56, 0
	v_dual_mov_b32 v51, 0 :: v_dual_mov_b32 v50, 0
	v_dual_mov_b32 v49, 0 :: v_dual_mov_b32 v48, 0
	v_dual_mov_b32 v63, 0 :: v_dual_mov_b32 v62, 0
	v_dual_mov_b32 v61, 0 :: v_dual_mov_b32 v60, 0
	s_and_saveexec_b32 s28, vcc_lo
	s_cbranch_execz .LBB0_143
; %bb.149:                              ;   in Loop: Header=BB0_146 Depth=2
	ds_load_b128 v[60:63], v118 offset:1024
	ds_load_b128 v[56:59], v124 offset:1024
	ds_load_b128 v[52:55], v122 offset:1024
	ds_load_b128 v[44:47], v121 offset:1024
	v_dual_mov_b32 v31, 0 :: v_dual_mov_b32 v30, 0
	v_dual_mov_b32 v29, 0 :: v_dual_mov_b32 v28, 0
	v_dual_mov_b32 v35, 0 :: v_dual_mov_b32 v34, 0
	v_dual_mov_b32 v33, 0 :: v_dual_mov_b32 v32, 0
	v_dual_mov_b32 v43, 0 :: v_dual_mov_b32 v42, 0
	v_dual_mov_b32 v41, 0 :: v_dual_mov_b32 v40, 0
	v_dual_mov_b32 v51, 0 :: v_dual_mov_b32 v50, 0
	v_dual_mov_b32 v49, 0 :: v_dual_mov_b32 v48, 0
	s_mov_b32 s29, exec_lo
	v_cmpx_gt_u32_e64 s12, v119
	s_cbranch_execz .LBB0_142
; %bb.150:                              ;   in Loop: Header=BB0_146 Depth=2
	ds_load_b128 v[48:51], v118 offset:1536
	ds_load_b128 v[40:43], v124 offset:1536
	ds_load_b128 v[32:35], v122 offset:1536
	ds_load_b128 v[28:31], v121 offset:1536
	s_branch .LBB0_142
.LBB0_151:                              ;   in Loop: Header=BB0_138 Depth=1
	v_dual_mov_b32 v117, 0 :: v_dual_mov_b32 v116, 0
	v_dual_mov_b32 v115, 0 :: v_dual_mov_b32 v114, 0
	v_dual_mov_b32 v113, 0 :: v_dual_mov_b32 v112, 0
	v_dual_mov_b32 v110, 0 :: v_dual_mov_b32 v111, 0
.LBB0_152:                              ;   in Loop: Header=BB0_138 Depth=1
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_dpp v2, v113, v113 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v117, v117 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_dscnt 0x7
	v_add_f32_dpp v1, v115, v115 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_xor_b32_e32 v3, 16, v106
	v_add_f32_dpp v4, v114, v114 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_dscnt 0x0
	v_add_f32_dpp v5, v112, v112 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v111, v111 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v110, v110 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v0, v0 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v1, v1, v1 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cmp_gt_i32_e32 vcc_lo, 32, v3
	v_add_f32_dpp v4, v4, v4 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v5, v5 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v15, v2, v2 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v116, v116 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v6, v6 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v7, v7 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v3, v106, v3, vcc_lo
	v_add_f32_dpp v0, v0, v0 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v1, v1, v1 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v5, v5 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v6, v6 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v7, v7 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshlrev_b32_e32 v3, 2, v3
	v_add_f32_dpp v0, v0, v0 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v17, v1, v1 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v13, v2, v2 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v11, v4, v4 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v9, v5, v5 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v6, v6 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v7, v7 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	ds_bpermute_b32 v1, v3, v0
	ds_bpermute_b32 v18, v3, v17
	ds_bpermute_b32 v16, v3, v15
	ds_bpermute_b32 v14, v3, v13
	ds_bpermute_b32 v12, v3, v11
	ds_bpermute_b32 v10, v3, v9
	ds_bpermute_b32 v7, v3, v6
	ds_bpermute_b32 v5, v3, v4
	s_wait_loadcnt 0x1
	;;#ASMSTART
	;;#ASMEND
	s_wait_loadcnt 0x0
	;;#ASMSTART
	;;#ASMEND
	s_and_saveexec_b32 s26, s1
	s_cbranch_execz .LBB0_137
; %bb.153:                              ;   in Loop: Header=BB0_138 Depth=1
	v_lshlrev_b64_e32 v[2:3], 1, v[96:97]
	s_wait_dscnt 0x7
	v_add_f32_e32 v8, v0, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v19, s33, v8
	v_add_co_u32 v0, vcc_lo, s4, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s5, v3, vcc_lo
	s_and_not1_b32 vcc_lo, exec_lo, s25
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_155
; %bb.154:                              ;   in Loop: Header=BB0_138 Depth=1
	global_load_u16 v8, v[0:1], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v8, 16, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v19, s9, v8
.LBB0_155:                              ;   in Loop: Header=BB0_138 Depth=1
	v_lshlrev_b32_e32 v8, 16, v109
	v_add_co_u32 v2, vcc_lo, s6, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s7, v3, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v20, v19, v8
	v_cmp_gt_u32_e64 s2, s13, v108
	v_cndmask_b32_e64 v19, v20, v19, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_bfe_u32 v20, v19, 16, 1
	v_or_b32_e32 v21, 0x400000, v19
	v_cmp_u_f32_e32 vcc_lo, v19, v19
	v_add3_u32 v20, v20, v19, 0x7fff
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v19, v20, v21, vcc_lo
	global_store_d16_hi_b16 v[2:3], v19, off
	s_and_saveexec_b32 s27, s2
	s_cbranch_execz .LBB0_159
; %bb.156:                              ;   in Loop: Header=BB0_138 Depth=1
	s_wait_dscnt 0x6
	v_add_f32_e32 v17, v17, v18
	s_and_not1_b32 vcc_lo, exec_lo, s25
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v17, s33, v17
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_158
; %bb.157:                              ;   in Loop: Header=BB0_138 Depth=1
	global_load_u16 v18, v[0:1], off offset:2
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v18, 16, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, s9, v18
.LBB0_158:                              ;   in Loop: Header=BB0_138 Depth=1
	v_lshlrev_b32_e32 v18, 16, v107
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v18, v17, v18
	v_cndmask_b32_e64 v17, v18, v17, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_bfe_u32 v18, v17, 16, 1
	v_or_b32_e32 v19, 0x400000, v17
	v_cmp_u_f32_e32 vcc_lo, v17, v17
	v_add3_u32 v18, v18, v17, 0x7fff
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v17, v18, v19, vcc_lo
	global_store_d16_hi_b16 v[2:3], v17, off offset:2
.LBB0_159:                              ;   in Loop: Header=BB0_138 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s27
	s_wait_dscnt 0x5
	v_add_f32_e32 v15, v15, v16
	v_add_co_u32 v0, vcc_lo, v0, s10
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s11, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_f32_e32 v15, s33, v15
	s_and_not1_b32 vcc_lo, exec_lo, s25
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_161
; %bb.160:                              ;   in Loop: Header=BB0_138 Depth=1
	global_load_u16 v16, v[0:1], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v16, 16, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v15, s9, v16
.LBB0_161:                              ;   in Loop: Header=BB0_138 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_add_f32_e32 v16, v15, v8
	v_add_co_u32 v2, vcc_lo, v2, s20
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s21, v3, vcc_lo
	v_cndmask_b32_e64 v15, v16, v15, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_bfe_u32 v16, v15, 16, 1
	v_or_b32_e32 v17, 0x400000, v15
	v_cmp_u_f32_e32 vcc_lo, v15, v15
	v_add3_u32 v16, v16, v15, 0x7fff
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v15, v16, v17, vcc_lo
	global_store_d16_hi_b16 v[2:3], v15, off
	s_and_saveexec_b32 s27, s2
	s_cbranch_execz .LBB0_165
; %bb.162:                              ;   in Loop: Header=BB0_138 Depth=1
	s_wait_dscnt 0x4
	v_add_f32_e32 v13, v13, v14
	s_and_not1_b32 vcc_lo, exec_lo, s25
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v13, s33, v13
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_164
; %bb.163:                              ;   in Loop: Header=BB0_138 Depth=1
	global_load_u16 v14, v[0:1], off offset:2
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v14, 16, v14
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, s9, v14
.LBB0_164:                              ;   in Loop: Header=BB0_138 Depth=1
	v_lshlrev_b32_e32 v14, 16, v107
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v14, v13, v14
	v_cndmask_b32_e64 v13, v14, v13, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_bfe_u32 v14, v13, 16, 1
	v_or_b32_e32 v15, 0x400000, v13
	v_cmp_u_f32_e32 vcc_lo, v13, v13
	v_add3_u32 v14, v14, v13, 0x7fff
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v13, v14, v15, vcc_lo
	global_store_d16_hi_b16 v[2:3], v13, off offset:2
.LBB0_165:                              ;   in Loop: Header=BB0_138 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s27
	s_wait_dscnt 0x3
	v_add_f32_e32 v11, v11, v12
	v_add_co_u32 v0, vcc_lo, v0, s10
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s11, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_f32_e32 v11, s33, v11
	s_and_not1_b32 vcc_lo, exec_lo, s25
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_167
; %bb.166:                              ;   in Loop: Header=BB0_138 Depth=1
	global_load_u16 v12, v[0:1], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v12, 16, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, s9, v12
.LBB0_167:                              ;   in Loop: Header=BB0_138 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_add_f32_e32 v12, v11, v8
	v_add_co_u32 v2, vcc_lo, v2, s20
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s21, v3, vcc_lo
	v_cndmask_b32_e64 v11, v12, v11, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_bfe_u32 v12, v11, 16, 1
	v_or_b32_e32 v13, 0x400000, v11
	v_cmp_u_f32_e32 vcc_lo, v11, v11
	v_add3_u32 v12, v12, v11, 0x7fff
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v11, v12, v13, vcc_lo
	global_store_d16_hi_b16 v[2:3], v11, off
	s_and_saveexec_b32 s27, s2
	s_cbranch_execz .LBB0_171
; %bb.168:                              ;   in Loop: Header=BB0_138 Depth=1
	s_wait_dscnt 0x2
	v_add_f32_e32 v9, v9, v10
	s_and_not1_b32 vcc_lo, exec_lo, s25
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v9, s33, v9
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_170
; %bb.169:                              ;   in Loop: Header=BB0_138 Depth=1
	global_load_u16 v10, v[0:1], off offset:2
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v10, 16, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, s9, v10
.LBB0_170:                              ;   in Loop: Header=BB0_138 Depth=1
	v_lshlrev_b32_e32 v10, 16, v107
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v10, v9, v10
	v_cndmask_b32_e64 v9, v10, v9, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_bfe_u32 v10, v9, 16, 1
	v_or_b32_e32 v11, 0x400000, v9
	v_cmp_u_f32_e32 vcc_lo, v9, v9
	v_add3_u32 v10, v10, v9, 0x7fff
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v9, v10, v11, vcc_lo
	global_store_d16_hi_b16 v[2:3], v9, off offset:2
.LBB0_171:                              ;   in Loop: Header=BB0_138 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s27
	s_wait_dscnt 0x1
	v_add_f32_e32 v6, v6, v7
	v_add_co_u32 v0, vcc_lo, v0, s10
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s11, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_f32_e32 v6, s33, v6
	s_and_not1_b32 vcc_lo, exec_lo, s25
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_173
; %bb.172:                              ;   in Loop: Header=BB0_138 Depth=1
	global_load_u16 v7, v[0:1], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v7, 16, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v6, s9, v7
.LBB0_173:                              ;   in Loop: Header=BB0_138 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_add_f32_e32 v7, v6, v8
	v_add_co_u32 v2, vcc_lo, v2, s20
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s21, v3, vcc_lo
	v_cndmask_b32_e64 v6, v7, v6, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_bfe_u32 v7, v6, 16, 1
	v_or_b32_e32 v8, 0x400000, v6
	v_cmp_u_f32_e32 vcc_lo, v6, v6
	v_add3_u32 v7, v7, v6, 0x7fff
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v6, v7, v8, vcc_lo
	global_store_d16_hi_b16 v[2:3], v6, off
	s_and_b32 exec_lo, exec_lo, s2
	s_cbranch_execz .LBB0_137
; %bb.174:                              ;   in Loop: Header=BB0_138 Depth=1
	s_wait_dscnt 0x0
	v_add_f32_e32 v4, v4, v5
	s_and_not1_b32 vcc_lo, exec_lo, s25
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v4, s33, v4
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_136
; %bb.175:                              ;   in Loop: Header=BB0_138 Depth=1
	global_load_u16 v0, v[0:1], off offset:2
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v0, 16, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v4, s9, v0
	s_branch .LBB0_136
.LBB0_176:
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wvSplitKQ_vllm_f8_bf16_tn_n4_y2u4_gfx1201
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
		.amdhsa_next_free_sgpr 49
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size 93
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
	.size	wvSplitKQ_vllm_f8_bf16_tn_n4_y2u4_gfx1201, .Lfunc_end0-wvSplitKQ_vllm_f8_bf16_tn_n4_y2u4_gfx1201
                                        ; -- End function
	.set wvSplitKQ_vllm_f8_bf16_tn_n4_y2u4_gfx1201.num_vgpr, 125
	.set wvSplitKQ_vllm_f8_bf16_tn_n4_y2u4_gfx1201.num_agpr, 0
	.set wvSplitKQ_vllm_f8_bf16_tn_n4_y2u4_gfx1201.numbered_sgpr, 49
	.set wvSplitKQ_vllm_f8_bf16_tn_n4_y2u4_gfx1201.num_named_barrier, 0
	.set wvSplitKQ_vllm_f8_bf16_tn_n4_y2u4_gfx1201.private_seg_size, 0
	.set wvSplitKQ_vllm_f8_bf16_tn_n4_y2u4_gfx1201.uses_vcc, 1
	.set wvSplitKQ_vllm_f8_bf16_tn_n4_y2u4_gfx1201.uses_flat_scratch, 0
	.set wvSplitKQ_vllm_f8_bf16_tn_n4_y2u4_gfx1201.has_dyn_sized_stack, 0
	.set wvSplitKQ_vllm_f8_bf16_tn_n4_y2u4_gfx1201.has_recursion, 0
	.set wvSplitKQ_vllm_f8_bf16_tn_n4_y2u4_gfx1201.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 11784
; TotalNumSgprs: 51
; NumVgprs: 125
; ScratchSize: 0
; MemoryBound: 1
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 65536 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 21
; NumSGPRsForWavesPerEU: 51
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
	.type	__hip_cuid_1b139d1b9194e41f,@object ; @__hip_cuid_1b139d1b9194e41f
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_1b139d1b9194e41f
__hip_cuid_1b139d1b9194e41f:
	.byte	0                               ; 0x0
	.size	__hip_cuid_1b139d1b9194e41f, 1

	.ident	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.0 26014 7b800a19466229b8479a78de19143dc33c3ab9b5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_1b139d1b9194e41f
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
    DestDataType: b
    ComputeDataType: s
    HighPrecisionAccumulate: True
    TransposeA: True
    TransposeB: False
    UseBeta: True
    Batched: True
    UseScaleAB: Scalar
    UseBias: 1
    BiasSrc: D
    BiasDataTypeList: ['b']
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
    .name:           wvSplitKQ_vllm_f8_bf16_tn_n4_y2u4_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count:     51
    .sgpr_spill_count: 0
    .symbol:         wvSplitKQ_vllm_f8_bf16_tn_n4_y2u4_gfx1201.kd
    .vgpr_count:     125
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 1
...

	.end_amdgpu_metadata
