	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 4
	.text
	.protected	wvSplitKQ_vllm_f8_bf16_tn_n1_y1u4_gfx1201 ; -- Begin function wvSplitKQ_vllm_f8_bf16_tn_n1_y1u4_gfx1201
	.globl	wvSplitKQ_vllm_f8_bf16_tn_n1_y1u4_gfx1201
	.p2align	8
	.type	wvSplitKQ_vllm_f8_bf16_tn_n1_y1u4_gfx1201,@function
wvSplitKQ_vllm_f8_bf16_tn_n1_y1u4_gfx1201: ; @wvSplitKQ_vllm_f8_bf16_tn_n1_y1u4_gfx1201
; %bb.0:
	s_clause 0x1
	s_load_b64 s[2:3], s[0:1], 0x0
	s_load_b64 s[14:15], s[0:1], 0x10
	v_and_b32_e32 v40, 0x3ff, v0
	v_bfe_u32 v0, v0, 10, 10
	s_mov_b32 s17, 4
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v39, 4, v40
	v_lshl_add_u32 v3, v0, 9, v39
	s_wait_kmcnt 0x0
	s_min_u32 s5, s2, 0x10000
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)
	v_cmpx_gt_u32_e64 s5, v3
	s_cbranch_execz .LBB0_9
; %bb.1:
	global_load_b128 v[4:7], v3, s[14:15]
	v_add_nc_u32_e32 v1, 0x2000, v3
	s_mov_b32 s6, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b128 v3, v[4:7]
	v_cmpx_gt_u32_e64 s5, v1
	s_xor_b32 s6, exec_lo, s6
	s_cbranch_execz .LBB0_9
; %bb.2:
	v_add_co_u32 v1, s6, s14, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v2, null, s15, 0, s6
	v_add_nc_u32_e32 v8, 0x4000, v3
	s_mov_b32 s6, exec_lo
	global_load_b128 v[4:7], v[1:2], off offset:8192
	s_wait_loadcnt 0x0
	ds_store_b128 v3, v[4:7] offset:8192
	v_cmpx_gt_u32_e64 s5, v8
	s_wait_alu 0xfffe
	s_xor_b32 s6, exec_lo, s6
	s_cbranch_execz .LBB0_9
; %bb.3:
	global_load_b128 v[4:7], v[1:2], off offset:16384
	v_add_nc_u32_e32 v8, 0x6000, v3
	s_mov_b32 s6, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b128 v3, v[4:7] offset:16384
	v_cmpx_gt_u32_e64 s5, v8
	s_wait_alu 0xfffe
	s_xor_b32 s6, exec_lo, s6
	s_cbranch_execz .LBB0_9
; %bb.4:
	global_load_b128 v[4:7], v[1:2], off offset:24576
	v_add_nc_u32_e32 v8, 0x8000, v3
	s_mov_b32 s6, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b128 v3, v[4:7] offset:24576
	v_cmpx_gt_u32_e64 s5, v8
	s_wait_alu 0xfffe
	s_xor_b32 s6, exec_lo, s6
	s_cbranch_execz .LBB0_9
; %bb.5:
	global_load_b128 v[4:7], v[1:2], off offset:32768
	v_add_nc_u32_e32 v8, 0xa000, v3
	s_mov_b32 s6, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b128 v3, v[4:7] offset:32768
	v_cmpx_gt_u32_e64 s5, v8
	s_wait_alu 0xfffe
	s_xor_b32 s6, exec_lo, s6
	s_cbranch_execz .LBB0_9
; %bb.6:
	global_load_b128 v[4:7], v[1:2], off offset:40960
	v_add_nc_u32_e32 v8, 0xc000, v3
	s_mov_b32 s6, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b128 v3, v[4:7] offset:40960
	v_cmpx_gt_u32_e64 s5, v8
	s_wait_alu 0xfffe
	s_xor_b32 s6, exec_lo, s6
	s_cbranch_execz .LBB0_9
; %bb.7:
	global_load_b128 v[4:7], v[1:2], off offset:49152
	v_add_nc_u32_e32 v8, 0xe000, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_u32_e32 vcc_lo, s5, v8
	s_wait_loadcnt 0x0
	ds_store_b128 v3, v[4:7] offset:49152
	s_and_saveexec_b32 s5, vcc_lo
	s_wait_alu 0xfffe
	s_xor_b32 s5, exec_lo, s5
	s_cbranch_execz .LBB0_9
; %bb.8:
	global_load_b128 v[4:7], v[1:2], off offset:57344
	s_wait_loadcnt 0x0
	ds_store_b128 v3, v[4:7] offset:57344
.LBB0_9:
	s_or_b32 exec_lo, exec_lo, s4
	s_clause 0x1
	s_load_b32 s18, s[0:1], 0x18
	s_load_b64 s[4:5], s[0:1], 0x48
	s_add_co_i32 s6, s3, -1
	s_wait_kmcnt 0x0
	s_lshl_b32 s7, s18, 4
	s_wait_alu 0xfffe
	v_add_co_u32 v1, s9, s7, s6
	s_and_not1_b32 vcc_lo, exec_lo, s9
	v_readfirstlane_b32 s8, v1
	s_cbranch_vccz .LBB0_11
; %bb.10:
	s_cvt_f32_u32 s9, s7
	s_sub_co_i32 s10, 0, s7
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_iflag_f32_e32 v1, s9
	v_readfirstlane_b32 s9, v1
	s_mul_f32 s9, s9, 0x4f7ffffe
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_cvt_u32_f32 s9, s9
	s_wait_alu 0xfffe
	s_mul_i32 s10, s10, s9
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_hi_u32 s10, s9, s10
	s_add_co_i32 s9, s9, s10
	s_wait_alu 0xfffe
	s_mul_hi_u32 s9, s8, s9
	s_wait_alu 0xfffe
	s_mul_i32 s10, s9, s7
	s_delay_alu instid0(SALU_CYCLE_1)
	s_sub_co_i32 s8, s8, s10
	s_add_co_i32 s10, s9, 1
	s_wait_alu 0xfffe
	s_sub_co_i32 s11, s8, s7
	s_cmp_ge_u32 s8, s7
	s_cselect_b32 s9, s10, s9
	s_cselect_b32 s8, s11, s8
	s_wait_alu 0xfffe
	s_add_co_i32 s10, s9, 1
	s_cmp_ge_u32 s8, s7
	s_cselect_b32 s7, s10, s9
	s_wait_alu 0xfffe
	s_mul_i32 s7, s7, s18
	s_wait_alu 0xfffe
	s_cvt_f32_u32 s8, s7
	s_sub_co_i32 s9, 0, s7
	s_add_co_i32 s6, s7, s6
	s_wait_alu 0xfffe
	v_rcp_iflag_f32_e32 v1, s8
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_2)
	v_readfirstlane_b32 s8, v1
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
	s_mul_hi_u32 s8, s6, s8
	s_wait_alu 0xfffe
	s_mul_i32 s9, s8, s7
	s_wait_alu 0xfffe
	s_sub_co_i32 s6, s6, s9
	s_add_co_i32 s9, s8, 1
	s_wait_alu 0xfffe
	s_sub_co_i32 s10, s6, s7
	s_cmp_ge_u32 s6, s7
	s_cselect_b32 s8, s9, s8
	s_cselect_b32 s6, s10, s6
	s_wait_alu 0xfffe
	s_add_co_i32 s9, s8, 1
	s_cmp_ge_u32 s6, s7
	s_wait_alu 0xfffe
	s_cselect_b32 s6, s9, s8
	s_wait_alu 0xfffe
	s_max_u32 s17, s6, 4
.LBB0_11:
	s_load_b64 s[6:7], s[0:1], 0x50
	s_mov_b32 s19, 1.0
	s_cmp_eq_u64 s[4:5], 0
	s_mov_b32 s20, 1.0
	s_cbranch_scc1 .LBB0_13
; %bb.12:
	s_load_b32 s20, s[4:5], 0x0
.LBB0_13:
	s_wait_kmcnt 0x0
	s_cmp_eq_u64 s[6:7], 0
	s_cbranch_scc1 .LBB0_15
; %bb.14:
	s_load_b32 s19, s[6:7], 0x0
.LBB0_15:
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_mov_b32 s4, exec_lo
	v_cmpx_gt_u32_e64 s17, v0
	s_cbranch_execz .LBB0_75
; %bb.16:
	s_clause 0x4
	s_load_b64 s[8:9], s[0:1], 0x38
	s_load_b128 s[4:7], s[0:1], 0x28
	s_load_b64 s[10:11], s[0:1], 0x58
	s_load_b64 s[12:13], s[0:1], 0x8
	s_load_b32 s16, s[0:1], 0x20
	v_mad_co_u64_u32 v[32:33], null, s17, ttmp9, v[0:1]
	s_cmp_gt_u32 s2, 0x10000
	s_mul_i32 s17, s17, s18
	v_cmp_gt_u32_e32 vcc_lo, s3, v32
	s_wait_kmcnt 0x0
	s_mul_f32 s0, s8, s20
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_2)
	s_mul_f32 s8, s0, s19
	s_mov_b32 s0, -1
	s_cbranch_scc0 .LBB0_53
; %bb.17:
	s_and_saveexec_b32 s18, vcc_lo
	s_cbranch_execz .LBB0_52
; %bb.18:
	s_cmp_eq_u64 s[10:11], 0
	v_mbcnt_lo_u32_b32 v41, -1, 0
	s_cselect_b32 s0, -1, 0
	s_cmp_lg_u64 s[10:11], 0
	v_cmp_eq_u32_e64 s1, 31, v40
	s_cselect_b32 s19, -1, 0
	s_add_co_i32 s20, s2, -16
	s_cmp_neq_f32 s9, 0
	v_dual_mov_b32 v34, 0 :: v_dual_mov_b32 v33, v32
	v_xor_b32_e32 v42, 16, v41
	s_cselect_b32 s21, -1, 0
	s_mov_b32 s22, 0
	s_branch .LBB0_21
.LBB0_19:                               ;   in Loop: Header=BB0_21 Depth=1
	v_lshlrev_b32_e32 v1, 16, v43
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v0, v1
	v_cndmask_b32_e64 v2, v1, v0, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_bfe_u32 v0, v2, 16, 1
	v_or_b32_e32 v3, 0x400000, v2
	v_add3_u32 v4, v0, v2, 0x7fff
	v_add_co_u32 v0, vcc_lo, s6, v35
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s7, v36, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v2, v2
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v2, v4, v3, vcc_lo
	global_store_d16_hi_b16 v[0:1], v2, off
.LBB0_20:                               ;   in Loop: Header=BB0_21 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s23
	v_add_nc_u32_e32 v33, s17, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e32 vcc_lo, s3, v33
	s_or_b32 s22, vcc_lo, s22
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s22
	s_cbranch_execz .LBB0_52
.LBB0_21:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_29 Depth 2
	v_lshlrev_b64_e32 v[35:36], 1, v[33:34]
	v_mov_b32_e32 v43, v34
	s_and_not1_b32 vcc_lo, exec_lo, s19
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_23
; %bb.22:                               ;   in Loop: Header=BB0_21 Depth=1
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, s10, v35
	s_wait_dscnt 0x0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s11, v36, vcc_lo
	global_load_u16 v43, v[0:1], off
.LBB0_23:                               ;   in Loop: Header=BB0_21 Depth=1
	v_mad_co_u64_u32 v[37:38], null, v33, s16, s[12:13]
	v_mov_b32_e32 v44, 0
	s_mov_b32 s23, 0
	s_branch .LBB0_29
.LBB0_24:                               ;   in Loop: Header=BB0_29 Depth=2
	s_or_b32 exec_lo, exec_lo, s28
.LBB0_25:                               ;   in Loop: Header=BB0_29 Depth=2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s27
.LBB0_26:                               ;   in Loop: Header=BB0_29 Depth=2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s26
.LBB0_27:                               ;   in Loop: Header=BB0_29 Depth=2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s25
.LBB0_28:                               ;   in Loop: Header=BB0_29 Depth=2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s24
	s_wait_loadcnt_dscnt 0x0
	v_dot4_f32_fp8_fp8 v12, v16, v12, v44
	s_addk_co_i32 s23, 0x800
	s_wait_alu 0xfffe
	s_cmp_lt_u32 s23, s2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dot4_f32_fp8_fp8 v12, v17, v13, v12
	v_dot4_f32_fp8_fp8 v12, v18, v14, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dot4_f32_fp8_fp8 v12, v19, v15, v12
	v_dot4_f32_fp8_fp8 v8, v20, v8, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dot4_f32_fp8_fp8 v8, v21, v9, v8
	v_dot4_f32_fp8_fp8 v8, v22, v10, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dot4_f32_fp8_fp8 v8, v23, v11, v8
	v_dot4_f32_fp8_fp8 v4, v24, v4, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dot4_f32_fp8_fp8 v4, v25, v5, v4
	v_dot4_f32_fp8_fp8 v4, v26, v6, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dot4_f32_fp8_fp8 v4, v27, v7, v4
	v_dot4_f32_fp8_fp8 v0, v28, v0, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dot4_f32_fp8_fp8 v0, v29, v1, v0
	v_dot4_f32_fp8_fp8 v0, v30, v2, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_dot4_f32_fp8_fp8 v44, v31, v3, v0
	s_cbranch_scc0 .LBB0_49
.LBB0_29:                               ;   Parent Loop BB0_21 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu 0xfffe
	v_add_nc_u32_e32 v45, s23, v39
	v_dual_mov_b32 v31, 0 :: v_dual_mov_b32 v30, 0
	v_dual_mov_b32 v29, 0 :: v_dual_mov_b32 v28, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_add_nc_u32_e32 v48, 0x200, v45
	v_dual_mov_b32 v18, 0 :: v_dual_add_nc_u32 v47, 0x400, v45
	v_min_u32_e32 v0, s20, v45
	v_dual_mov_b32 v19, 0 :: v_dual_add_nc_u32 v46, 0x600, v45
	v_min_u32_e32 v2, s20, v48
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_min_u32_e32 v4, s20, v47
	v_add_co_u32 v0, vcc_lo, v37, v0
	s_delay_alu instid0(VALU_DEP_4)
	v_min_u32_e32 v6, s20, v46
	s_wait_dscnt 0x0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, 0, v38, vcc_lo
	v_add_co_u32 v2, vcc_lo, v37, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, 0, v38, vcc_lo
	v_add_co_u32 v4, vcc_lo, v37, v4
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v5, null, 0, v38, vcc_lo
	v_add_co_u32 v16, vcc_lo, v37, v6
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v17, null, 0, v38, vcc_lo
	s_clause 0x3
	global_load_b128 v[12:15], v[0:1], off
	global_load_b128 v[8:11], v[2:3], off
	global_load_b128 v[4:7], v[4:5], off
	global_load_b128 v[0:3], v[16:17], off
	v_dual_mov_b32 v16, 0 :: v_dual_mov_b32 v17, 0
	v_dual_mov_b32 v27, 0 :: v_dual_mov_b32 v26, 0
	v_dual_mov_b32 v25, 0 :: v_dual_mov_b32 v24, 0
	v_dual_mov_b32 v23, 0 :: v_dual_mov_b32 v22, 0
	v_dual_mov_b32 v21, 0 :: v_dual_mov_b32 v20, 0
	s_mov_b32 s24, exec_lo
	v_cmpx_gt_u32_e64 s2, v45
	s_cbranch_execz .LBB0_28
; %bb.30:                               ;   in Loop: Header=BB0_29 Depth=2
	s_mov_b32 s25, exec_lo
                                        ; implicit-def: $vgpr16
	v_cmpx_lt_u32_e32 0xffff, v45
	s_xor_b32 s25, exec_lo, s25
	s_cbranch_execz .LBB0_32
; %bb.31:                               ;   in Loop: Header=BB0_29 Depth=2
	global_load_b128 v[16:19], v45, s[14:15]
.LBB0_32:                               ;   in Loop: Header=BB0_29 Depth=2
	s_and_not1_saveexec_b32 s25, s25
	s_cbranch_execz .LBB0_34
; %bb.33:                               ;   in Loop: Header=BB0_29 Depth=2
	s_wait_loadcnt 0x0
	ds_load_b128 v[16:19], v45
.LBB0_34:                               ;   in Loop: Header=BB0_29 Depth=2
	s_or_b32 exec_lo, exec_lo, s25
	v_dual_mov_b32 v31, 0 :: v_dual_mov_b32 v30, 0
	v_dual_mov_b32 v29, 0 :: v_dual_mov_b32 v28, 0
	v_dual_mov_b32 v27, 0 :: v_dual_mov_b32 v26, 0
	v_dual_mov_b32 v25, 0 :: v_dual_mov_b32 v24, 0
	v_dual_mov_b32 v23, 0 :: v_dual_mov_b32 v22, 0
	v_dual_mov_b32 v21, 0 :: v_dual_mov_b32 v20, 0
	s_mov_b32 s25, exec_lo
	v_cmpx_gt_u32_e64 s2, v48
	s_cbranch_execz .LBB0_27
; %bb.35:                               ;   in Loop: Header=BB0_29 Depth=2
	s_mov_b32 s26, exec_lo
                                        ; implicit-def: $vgpr23
	v_cmpx_lt_u32_e32 0xffff, v48
	s_xor_b32 s26, exec_lo, s26
	s_cbranch_execz .LBB0_37
; %bb.36:                               ;   in Loop: Header=BB0_29 Depth=2
	global_load_b128 v[20:23], v48, s[14:15]
.LBB0_37:                               ;   in Loop: Header=BB0_29 Depth=2
	s_and_not1_saveexec_b32 s26, s26
	s_cbranch_execz .LBB0_39
; %bb.38:                               ;   in Loop: Header=BB0_29 Depth=2
	s_wait_loadcnt 0x0
	ds_load_b128 v[20:23], v45 offset:512
.LBB0_39:                               ;   in Loop: Header=BB0_29 Depth=2
	s_or_b32 exec_lo, exec_lo, s26
	v_dual_mov_b32 v31, 0 :: v_dual_mov_b32 v30, 0
	v_dual_mov_b32 v29, 0 :: v_dual_mov_b32 v28, 0
	v_dual_mov_b32 v27, 0 :: v_dual_mov_b32 v26, 0
	v_dual_mov_b32 v25, 0 :: v_dual_mov_b32 v24, 0
	s_mov_b32 s26, exec_lo
	v_cmpx_gt_u32_e64 s2, v47
	s_cbranch_execz .LBB0_26
; %bb.40:                               ;   in Loop: Header=BB0_29 Depth=2
	s_mov_b32 s27, exec_lo
                                        ; implicit-def: $vgpr27
	v_cmpx_lt_u32_e32 0xffff, v47
	s_xor_b32 s27, exec_lo, s27
	s_cbranch_execz .LBB0_42
; %bb.41:                               ;   in Loop: Header=BB0_29 Depth=2
	global_load_b128 v[24:27], v47, s[14:15]
.LBB0_42:                               ;   in Loop: Header=BB0_29 Depth=2
	s_and_not1_saveexec_b32 s27, s27
	s_cbranch_execz .LBB0_44
; %bb.43:                               ;   in Loop: Header=BB0_29 Depth=2
	s_wait_loadcnt 0x0
	ds_load_b128 v[24:27], v45 offset:1024
.LBB0_44:                               ;   in Loop: Header=BB0_29 Depth=2
	s_or_b32 exec_lo, exec_lo, s27
	v_dual_mov_b32 v31, 0 :: v_dual_mov_b32 v30, 0
	v_dual_mov_b32 v29, 0 :: v_dual_mov_b32 v28, 0
	s_mov_b32 s27, exec_lo
	v_cmpx_gt_u32_e64 s2, v46
	s_cbranch_execz .LBB0_25
; %bb.45:                               ;   in Loop: Header=BB0_29 Depth=2
	s_mov_b32 s28, exec_lo
                                        ; implicit-def: $vgpr31
	v_cmpx_lt_u32_e32 0xffff, v46
	s_xor_b32 s28, exec_lo, s28
	s_cbranch_execz .LBB0_47
; %bb.46:                               ;   in Loop: Header=BB0_29 Depth=2
	global_load_b128 v[28:31], v46, s[14:15]
                                        ; implicit-def: $vgpr45
.LBB0_47:                               ;   in Loop: Header=BB0_29 Depth=2
	s_and_not1_saveexec_b32 s28, s28
	s_cbranch_execz .LBB0_24
; %bb.48:                               ;   in Loop: Header=BB0_29 Depth=2
	s_wait_loadcnt 0x0
	ds_load_b128 v[28:31], v45 offset:1536
	s_branch .LBB0_24
.LBB0_49:                               ;   in Loop: Header=BB0_21 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_add_f32_dpp v0, v44, v44 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cmp_gt_i32_e32 vcc_lo, 32, v42
	;;#ASMSTART
	;;#ASMEND
	v_add_f32_dpp v0, v0, v0 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v1, v41, v42, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_dpp v0, v0, v0 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshlrev_b32_e32 v1, 2, v1
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_dpp v0, v0, v0 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	ds_bpermute_b32 v1, v1, v0
	s_and_saveexec_b32 s23, s1
	s_cbranch_execz .LBB0_20
; %bb.50:                               ;   in Loop: Header=BB0_21 Depth=1
	s_wait_dscnt 0x0
	v_add_f32_e32 v0, v0, v1
	s_and_not1_b32 vcc_lo, exec_lo, s21
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v0, s8, v0
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_19
; %bb.51:                               ;   in Loop: Header=BB0_21 Depth=1
	v_add_co_u32 v1, vcc_lo, s4, v35
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v2, null, s5, v36, vcc_lo
	global_load_u16 v1, v[1:2], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v1, 16, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v0, s9, v1
	s_branch .LBB0_19
.LBB0_52:
	s_or_b32 exec_lo, exec_lo, s18
	s_mov_b32 s0, 0
.LBB0_53:
	s_wait_alu 0xfffe
	s_and_not1_b32 vcc_lo, exec_lo, s0
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_75
; %bb.54:
	v_cmp_gt_u32_e32 vcc_lo, s3, v32
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_75
; %bb.55:
	s_cmp_eq_u64 s[10:11], 0
	v_mbcnt_lo_u32_b32 v36, -1, 0
	s_cselect_b32 s0, -1, 0
	s_cmp_lg_u64 s[10:11], 0
	v_cmp_eq_u32_e64 s1, 31, v40
	s_cselect_b32 s14, -1, 0
	s_cmp_lg_u32 s2, 0
	v_mov_b32_e32 v33, 0
	s_cselect_b32 s15, -1, 0
	s_add_co_i32 s18, s2, -16
	s_cmp_neq_f32 s9, 0
	v_xor_b32_e32 v37, 16, v36
	s_mov_b32 s20, 0
	s_cselect_b32 s19, -1, 0
	s_branch .LBB0_58
.LBB0_56:                               ;   in Loop: Header=BB0_58 Depth=1
	v_lshlrev_b32_e32 v3, 16, v38
	v_add_co_u32 v0, vcc_lo, s6, v0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s7, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v3, v2, v3
	v_cndmask_b32_e64 v2, v3, v2, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_bfe_u32 v3, v2, 16, 1
	v_or_b32_e32 v4, 0x400000, v2
	v_cmp_u_f32_e32 vcc_lo, v2, v2
	v_add3_u32 v3, v3, v2, 0x7fff
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v2, v3, v4, vcc_lo
	global_store_d16_hi_b16 v[0:1], v2, off
.LBB0_57:                               ;   in Loop: Header=BB0_58 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s21
	v_add_nc_u32_e32 v32, s17, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e32 vcc_lo, s3, v32
	s_or_b32 s20, vcc_lo, s20
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s20
	s_cbranch_execz .LBB0_75
.LBB0_58:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_66 Depth 2
	v_mov_b32_e32 v38, 0
	s_wait_alu 0xfffe
	s_and_not1_b32 vcc_lo, exec_lo, s14
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_60
; %bb.59:                               ;   in Loop: Header=BB0_58 Depth=1
	s_wait_dscnt 0x0
	v_lshlrev_b64_e32 v[0:1], 1, v[32:33]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, s10, v0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s11, v1, vcc_lo
	global_load_u16 v38, v[0:1], off
.LBB0_60:                               ;   in Loop: Header=BB0_58 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s15
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_71
; %bb.61:                               ;   in Loop: Header=BB0_58 Depth=1
	v_mad_co_u64_u32 v[34:35], null, v32, s16, s[12:13]
	v_mov_b32_e32 v40, 0
	s_mov_b32 s21, 0
	s_branch .LBB0_66
.LBB0_62:                               ;   in Loop: Header=BB0_66 Depth=2
	s_or_b32 exec_lo, exec_lo, s25
.LBB0_63:                               ;   in Loop: Header=BB0_66 Depth=2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s24
.LBB0_64:                               ;   in Loop: Header=BB0_66 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s23
.LBB0_65:                               ;   in Loop: Header=BB0_66 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s22
	s_wait_loadcnt_dscnt 0x300
	v_dot4_f32_fp8_fp8 v12, v24, v12, v40
	s_addk_co_i32 s21, 0x800
	s_wait_alu 0xfffe
	s_cmp_lt_u32 s21, s2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dot4_f32_fp8_fp8 v12, v25, v13, v12
	v_dot4_f32_fp8_fp8 v12, v26, v14, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dot4_f32_fp8_fp8 v12, v27, v15, v12
	s_wait_loadcnt 0x2
	v_dot4_f32_fp8_fp8 v8, v20, v8, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dot4_f32_fp8_fp8 v8, v21, v9, v8
	v_dot4_f32_fp8_fp8 v8, v22, v10, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dot4_f32_fp8_fp8 v8, v23, v11, v8
	s_wait_loadcnt 0x1
	v_dot4_f32_fp8_fp8 v4, v28, v4, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dot4_f32_fp8_fp8 v4, v29, v5, v4
	v_dot4_f32_fp8_fp8 v4, v30, v6, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dot4_f32_fp8_fp8 v4, v31, v7, v4
	s_wait_loadcnt 0x0
	v_dot4_f32_fp8_fp8 v0, v16, v0, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dot4_f32_fp8_fp8 v0, v17, v1, v0
	v_dot4_f32_fp8_fp8 v0, v18, v2, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_dot4_f32_fp8_fp8 v40, v19, v3, v0
	s_cbranch_scc0 .LBB0_72
.LBB0_66:                               ;   Parent Loop BB0_58 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu 0xfffe
	v_dual_mov_b32 v20, 0 :: v_dual_add_nc_u32 v41, s21, v39
	v_dual_mov_b32 v28, 0 :: v_dual_mov_b32 v29, 0
	v_dual_mov_b32 v30, 0 :: v_dual_mov_b32 v31, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v21, 0 :: v_dual_add_nc_u32 v44, 0x200, v41
	v_dual_mov_b32 v22, 0 :: v_dual_add_nc_u32 v43, 0x400, v41
	v_min_u32_e32 v0, s18, v41
	v_dual_mov_b32 v23, 0 :: v_dual_add_nc_u32 v42, 0x600, v41
	v_min_u32_e32 v2, s18, v44
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_min_u32_e32 v4, s18, v43
	v_add_co_u32 v0, vcc_lo, v34, v0
	s_delay_alu instid0(VALU_DEP_4)
	v_min_u32_e32 v6, s18, v42
	s_wait_dscnt 0x0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, 0, v35, vcc_lo
	v_add_co_u32 v2, vcc_lo, v34, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, 0, v35, vcc_lo
	v_add_co_u32 v4, vcc_lo, v34, v4
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v5, null, 0, v35, vcc_lo
	v_add_co_u32 v16, vcc_lo, v34, v6
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v17, null, 0, v35, vcc_lo
	s_clause 0x3
	global_load_b128 v[12:15], v[0:1], off
	global_load_b128 v[8:11], v[2:3], off
	global_load_b128 v[4:7], v[4:5], off
	global_load_b128 v[0:3], v[16:17], off
	v_dual_mov_b32 v16, 0 :: v_dual_mov_b32 v17, 0
	v_dual_mov_b32 v18, 0 :: v_dual_mov_b32 v19, 0
	v_dual_mov_b32 v27, 0 :: v_dual_mov_b32 v26, 0
	v_dual_mov_b32 v25, 0 :: v_dual_mov_b32 v24, 0
	s_mov_b32 s22, exec_lo
	v_cmpx_gt_u32_e64 s2, v41
	s_cbranch_execz .LBB0_65
; %bb.67:                               ;   in Loop: Header=BB0_66 Depth=2
	ds_load_b128 v[24:27], v41
	v_dual_mov_b32 v19, 0 :: v_dual_mov_b32 v18, 0
	v_dual_mov_b32 v17, 0 :: v_dual_mov_b32 v16, 0
	v_dual_mov_b32 v31, 0 :: v_dual_mov_b32 v30, 0
	v_dual_mov_b32 v29, 0 :: v_dual_mov_b32 v28, 0
	v_dual_mov_b32 v23, 0 :: v_dual_mov_b32 v22, 0
	v_dual_mov_b32 v21, 0 :: v_dual_mov_b32 v20, 0
	s_mov_b32 s23, exec_lo
	v_cmpx_gt_u32_e64 s2, v44
	s_cbranch_execz .LBB0_64
; %bb.68:                               ;   in Loop: Header=BB0_66 Depth=2
	ds_load_b128 v[20:23], v41 offset:512
	v_dual_mov_b32 v19, 0 :: v_dual_mov_b32 v18, 0
	v_dual_mov_b32 v17, 0 :: v_dual_mov_b32 v16, 0
	v_dual_mov_b32 v31, 0 :: v_dual_mov_b32 v30, 0
	v_dual_mov_b32 v29, 0 :: v_dual_mov_b32 v28, 0
	s_mov_b32 s24, exec_lo
	v_cmpx_gt_u32_e64 s2, v43
	s_cbranch_execz .LBB0_63
; %bb.69:                               ;   in Loop: Header=BB0_66 Depth=2
	ds_load_b128 v[28:31], v41 offset:1024
	v_dual_mov_b32 v19, 0 :: v_dual_mov_b32 v18, 0
	v_dual_mov_b32 v17, 0 :: v_dual_mov_b32 v16, 0
	s_mov_b32 s25, exec_lo
	v_cmpx_gt_u32_e64 s2, v42
	s_cbranch_execz .LBB0_62
; %bb.70:                               ;   in Loop: Header=BB0_66 Depth=2
	ds_load_b128 v[16:19], v41 offset:1536
	s_branch .LBB0_62
.LBB0_71:                               ;   in Loop: Header=BB0_58 Depth=1
	v_mov_b32_e32 v40, 0
.LBB0_72:                               ;   in Loop: Header=BB0_58 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_add_f32_dpp v0, v40, v40 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cmp_gt_i32_e32 vcc_lo, 32, v37
	s_wait_loadcnt 0x0
	;;#ASMSTART
	;;#ASMEND
	v_add_f32_dpp v0, v0, v0 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_dscnt 0x0
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v1, v36, v37, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_dpp v0, v0, v0 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshlrev_b32_e32 v1, 2, v1
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_dpp v0, v0, v0 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	ds_bpermute_b32 v1, v1, v0
	s_and_saveexec_b32 s21, s1
	s_cbranch_execz .LBB0_57
; %bb.73:                               ;   in Loop: Header=BB0_58 Depth=1
	s_wait_dscnt 0x0
	v_add_f32_e32 v2, v0, v1
	v_lshlrev_b64_e32 v[0:1], 1, v[32:33]
	s_and_not1_b32 vcc_lo, exec_lo, s19
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v2, s8, v2
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_56
; %bb.74:                               ;   in Loop: Header=BB0_58 Depth=1
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_u32 v3, vcc_lo, s4, v0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v4, null, s5, v1, vcc_lo
	global_load_u16 v3, v[3:4], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v3, 16, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v2, s9, v3
	s_branch .LBB0_56
.LBB0_75:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wvSplitKQ_vllm_f8_bf16_tn_n1_y1u4_gfx1201
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
		.amdhsa_next_free_sgpr 29
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size 28
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
	.size	wvSplitKQ_vllm_f8_bf16_tn_n1_y1u4_gfx1201, .Lfunc_end0-wvSplitKQ_vllm_f8_bf16_tn_n1_y1u4_gfx1201
                                        ; -- End function
	.set wvSplitKQ_vllm_f8_bf16_tn_n1_y1u4_gfx1201.num_vgpr, 49
	.set wvSplitKQ_vllm_f8_bf16_tn_n1_y1u4_gfx1201.num_agpr, 0
	.set wvSplitKQ_vllm_f8_bf16_tn_n1_y1u4_gfx1201.numbered_sgpr, 29
	.set wvSplitKQ_vllm_f8_bf16_tn_n1_y1u4_gfx1201.num_named_barrier, 0
	.set wvSplitKQ_vllm_f8_bf16_tn_n1_y1u4_gfx1201.private_seg_size, 0
	.set wvSplitKQ_vllm_f8_bf16_tn_n1_y1u4_gfx1201.uses_vcc, 1
	.set wvSplitKQ_vllm_f8_bf16_tn_n1_y1u4_gfx1201.uses_flat_scratch, 0
	.set wvSplitKQ_vllm_f8_bf16_tn_n1_y1u4_gfx1201.has_dyn_sized_stack, 0
	.set wvSplitKQ_vllm_f8_bf16_tn_n1_y1u4_gfx1201.has_recursion, 0
	.set wvSplitKQ_vllm_f8_bf16_tn_n1_y1u4_gfx1201.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 3544
; TotalNumSgprs: 31
; NumVgprs: 49
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 65536 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 21
; NumSGPRsForWavesPerEU: 31
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
	.type	__hip_cuid_4c4eed39468262a,@object ; @__hip_cuid_4c4eed39468262a
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_4c4eed39468262a
__hip_cuid_4c4eed39468262a:
	.byte	0                               ; 0x0
	.size	__hip_cuid_4c4eed39468262a, 1

	.ident	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.0 26014 7b800a19466229b8479a78de19143dc33c3ab9b5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_4c4eed39468262a
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
  AssertSizeEqual: { 1: 1, 2: 1 }
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
    .name:           wvSplitKQ_vllm_f8_bf16_tn_n1_y1u4_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count:     31
    .sgpr_spill_count: 0
    .symbol:         wvSplitKQ_vllm_f8_bf16_tn_n1_y1u4_gfx1201.kd
    .vgpr_count:     49
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 1
...

	.end_amdgpu_metadata
