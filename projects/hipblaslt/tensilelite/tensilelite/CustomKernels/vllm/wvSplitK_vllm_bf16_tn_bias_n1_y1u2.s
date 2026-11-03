	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 4
	.text
	.protected	wvSplitK_vllm_bf16_tn_bias_n1_y1u2 ; -- Begin function wvSplitK_vllm_bf16_tn_bias_n1_y1u2
	.globl	wvSplitK_vllm_bf16_tn_bias_n1_y1u2
	.p2align	8
	.type	wvSplitK_vllm_bf16_tn_bias_n1_y1u2,@function
wvSplitK_vllm_bf16_tn_bias_n1_y1u2:     ; @wvSplitK_vllm_bf16_tn_bias_n1_y1u2
; %bb.70:
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.71:
.LBB0_0:
	s_load_dwordx2 s[24:25], s[0:1], 0x38
	s_load_dwordx2 s[26:27], s[0:1], 0x48
	s_mov_b64 s[0:1], -1
	s_cmp_gt_i32 s2, 0x14000
	v_bfe_u32 v28, v0, 10, 10
	s_cbranch_scc0 .LBB0_32
; %bb.1:
	s_lshl_b32 s0, s8, 4
	s_abs_i32 s1, s0
	v_cvt_f32_u32_e32 v1, s1
	s_sub_i32 s18, 0, s1
	s_add_i32 s9, s3, -1
	s_add_i32 s17, s0, s9
	v_rcp_iflag_f32_e32 v1, v1
	s_xor_b32 s0, s17, s0
	s_abs_i32 s17, s17
	s_ashr_i32 s0, s0, 31
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	s_nop 0
	v_readfirstlane_b32 s19, v1
	s_mul_i32 s18, s18, s19
	s_mul_hi_u32 s18, s19, s18
	s_add_i32 s19, s19, s18
	s_mul_hi_u32 s18, s17, s19
	s_mul_i32 s19, s18, s1
	s_sub_i32 s17, s17, s19
	s_add_i32 s20, s18, 1
	s_sub_i32 s19, s17, s1
	s_cmp_ge_u32 s17, s1
	s_cselect_b32 s18, s20, s18
	s_cselect_b32 s17, s19, s17
	s_add_i32 s19, s18, 1
	s_cmp_ge_u32 s17, s1
	s_cselect_b32 s1, s19, s18
	s_xor_b32 s1, s1, s0
	s_sub_i32 s0, s1, s0
	s_mul_i32 s0, s0, s8
	s_abs_i32 s1, s0
	v_cvt_f32_u32_e32 v1, s1
	s_sub_i32 s17, 0, s1
	s_add_i32 s9, s0, s9
	s_xor_b32 s0, s9, s0
	v_rcp_iflag_f32_e32 v1, v1
	s_abs_i32 s9, s9
	s_ashr_i32 s0, s0, 31
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	s_nop 0
	v_readfirstlane_b32 s18, v1
	s_mul_i32 s17, s17, s18
	s_mul_hi_u32 s17, s18, s17
	s_add_i32 s18, s18, s17
	s_mul_hi_u32 s17, s9, s18
	s_mul_i32 s18, s17, s1
	s_sub_i32 s9, s9, s18
	s_add_i32 s19, s17, 1
	s_sub_i32 s18, s9, s1
	s_cmp_ge_u32 s9, s1
	s_cselect_b32 s17, s19, s17
	s_cselect_b32 s9, s18, s9
	s_add_i32 s18, s17, 1
	s_cmp_ge_u32 s9, s1
	s_cselect_b32 s1, s18, s17
	s_xor_b32 s1, s1, s0
	s_sub_i32 s0, s1, s0
	s_max_i32 s20, s0, 4
	v_cmp_gt_u32_e32 vcc, s20, v28
	s_and_saveexec_b64 s[28:29], vcc
	s_cbranch_execz .LBB0_31
; %bb.2:
	v_cvt_f32_u32_e32 v1, s20
	s_mul_i32 s0, s20, s16
	s_sub_i32 s17, 0, s20
	v_add_u32_e32 v24, s0, v28
	v_rcp_iflag_f32_e32 v1, v1
	s_abs_i32 s9, s3
	s_ashr_i32 s1, s3, 31
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	s_nop 0
	v_readfirstlane_b32 s0, v1
	s_mul_i32 s17, s17, s0
	s_mul_hi_u32 s17, s0, s17
	s_add_i32 s0, s0, s17
	s_mul_hi_u32 s0, s9, s0
	s_mul_i32 s0, s0, s20
	s_sub_i32 s0, s9, s0
	s_sub_i32 s9, s0, s20
	s_cmp_ge_u32 s0, s20
	s_cselect_b32 s0, s9, s0
	s_sub_i32 s9, s0, s20
	s_cmp_ge_u32 s0, s20
	s_cselect_b32 s0, s9, s0
	s_xor_b32 s0, s0, s1
	s_sub_i32 s0, s0, s1
	s_add_i32 s1, s20, s3
	s_sub_i32 s1, s1, s0
	s_cmp_eq_u32 s0, 0
	s_cselect_b32 s9, s3, s1
	v_cmp_gt_u32_e32 vcc, s9, v24
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_31
; %bb.3:
	s_and_b32 s0, s11, 7
	s_cmp_lg_u32 s0, 0
	s_mov_b32 s31, 0
	s_cselect_b64 s[0:1], -1, 0
	s_and_b32 s30, s6, 15
	s_cmp_lg_u64 s[30:31], 0
	s_cselect_b64 s[18:19], -1, 0
	s_or_b64 s[36:37], s[18:19], s[0:1]
	s_waitcnt lgkmcnt(0)
	s_cmp_eq_u64 s[26:27], 0
	v_and_b32_e32 v2, 0x3ff, v0
	s_cselect_b64 s[0:1], -1, 0
	v_lshlrev_b32_e32 v1, 3, v2
	s_lshl_b32 s17, s20, 9
	v_cmp_eq_u32_e64 s[18:19], 63, v2
	s_mul_i32 s51, s20, s8
	v_lshlrev_b32_e32 v2, 4, v2
	s_lshl_b32 s52, s20, 10
	v_cmp_neq_f32_e64 s[20:21], s25, 0
	v_lshl_add_u32 v30, v28, 10, v2
	s_mov_b64 s[34:35], 0
	v_cndmask_b32_e64 v2, 0, 1, s[20:21]
	v_lshl_add_u32 v29, v28, 9, v1
	s_add_i32 s33, s2, -8
	s_ashr_i32 s50, s10, 31
	s_xor_b64 s[38:39], s[0:1], -1
	v_mov_b32_e32 v3, 0
	v_cmp_ne_u32_e64 s[20:21], 1, v2
                                        ; implicit-def: $vgpr31
	s_branch .LBB0_5
.LBB0_4:                                ;   in Loop: Header=BB0_5 Depth=1
	s_or_b64 exec, exec, s[40:41]
	v_add_u32_e32 v24, s51, v24
	v_cmp_le_u32_e32 vcc, s9, v24
	s_or_b64 s[34:35], vcc, s[34:35]
	s_andn2_b64 exec, exec, s[34:35]
	s_cbranch_execz .LBB0_31
.LBB0_5:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_11 Depth 2
                                        ;       Child Loop BB0_16 Depth 3
	v_cmp_gt_u32_e64 s[22:23], s3, v24
	s_and_b64 s[42:43], s[38:39], s[22:23]
	v_mov_b32_e32 v25, v3
	s_and_saveexec_b64 s[40:41], s[42:43]
	s_cbranch_execz .LBB0_7
; %bb.6:                                ;   in Loop: Header=BB0_5 Depth=1
	v_lshl_add_u64 v[4:5], v[24:25], 1, s[26:27]
	global_load_ushort v31, v[4:5], off
.LBB0_7:                                ;   in Loop: Header=BB0_5 Depth=1
	s_or_b64 exec, exec, s[40:41]
	v_mad_u64_u32 v[4:5], s[40:41], v24, s10, 0
	v_mov_b32_e32 v2, v5
	v_mad_u64_u32 v[6:7], s[40:41], v24, s50, v[2:3]
	v_mov_b32_e32 v5, v6
	v_lshl_add_u64 v[26:27], v[4:5], 1, s[4:5]
	v_mov_b32_e32 v4, v3
	v_mov_b32_e32 v5, v3
	v_mov_b32_e32 v2, v3
	s_mov_b32 s30, 0
	v_mov_b64_e32 v[6:7], v[4:5]
	s_mov_b32 s53, s30
	v_mov_b64_e32 v[4:5], v[2:3]
	s_branch .LBB0_11
.LBB0_8:                                ;   in Loop: Header=BB0_11 Depth=2
	s_or_b64 exec, exec, s[44:45]
.LBB0_9:                                ;   in Loop: Header=BB0_11 Depth=2
	s_or_b64 exec, exec, s[42:43]
	s_waitcnt vmcnt(1) lgkmcnt(0)
	v_mfma_f32_4x4x4_16b_bf16 v[4:7], v[16:17], v[12:13], v[4:7]
	s_nop 1
	v_mfma_f32_4x4x4_16b_bf16 v[4:7], v[18:19], v[14:15], v[4:7]
	s_waitcnt vmcnt(0)
	s_nop 0
	v_mfma_f32_4x4x4_16b_bf16 v[4:7], v[20:21], v[8:9], v[4:7]
	s_nop 1
	v_mfma_f32_4x4x4_16b_bf16 v[4:7], v[22:23], v[10:11], v[4:7]
.LBB0_10:                               ;   in Loop: Header=BB0_11 Depth=2
	s_or_b64 exec, exec, s[40:41]
	s_addk_i32 s53, 0x400
	s_cmp_ge_u32 s53, s2
	s_cbranch_scc1 .LBB0_25
.LBB0_11:                               ;   Parent Loop BB0_5 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_16 Depth 3
	s_cmp_eq_u32 s53, 0
	s_cselect_b64 s[40:41], -1, 0
	s_add_i32 s42, s30, 0x14000
	s_cmp_eq_u32 s53, s42
	s_cselect_b64 s[44:45], -1, 0
	s_or_b64 s[44:45], s[40:41], s[44:45]
	s_andn2_b64 vcc, exec, s[44:45]
	s_cbranch_vccz .LBB0_13
; %bb.12:                               ;   in Loop: Header=BB0_11 Depth=2
	s_and_saveexec_b64 s[40:41], s[22:23]
	s_cbranch_execz .LBB0_10
	s_branch .LBB0_22
.LBB0_13:                               ;   in Loop: Header=BB0_11 Depth=2
	s_and_b64 s[40:41], s[40:41], exec
	s_cselect_b32 s30, s30, s42
	s_sub_i32 s40, s2, s30
	s_min_u32 s54, s40, 0x14000
	s_lshl_b64 s[40:41], s[30:31], 1
	s_add_u32 s40, s6, s40
	s_addc_u32 s41, s7, s41
	s_mov_b32 s55, 0
	s_mov_b64 s[42:43], 0
	v_mov_b32_e32 v10, v30
	s_barrier
                                        ; implicit-def: $sgpr44_sgpr45
	s_branch .LBB0_16
.LBB0_14:                               ;   in Loop: Header=BB0_16 Depth=3
	s_add_i32 s55, s55, s17
	s_cmp_gt_u32 s55, 0x13fff
	s_cselect_b64 s[48:49], -1, 0
	s_andn2_b64 s[44:45], s[44:45], exec
	s_and_b64 s[48:49], s[48:49], exec
	v_add_u32_e32 v10, s52, v10
	s_or_b64 s[44:45], s[44:45], s[48:49]
.LBB0_15:                               ;   in Loop: Header=BB0_16 Depth=3
	s_or_b64 exec, exec, s[46:47]
	s_and_b64 s[46:47], exec, s[44:45]
	s_or_b64 s[42:43], s[46:47], s[42:43]
	s_andn2_b64 exec, exec, s[42:43]
	s_cbranch_execz .LBB0_21
.LBB0_16:                               ;   Parent Loop BB0_5 Depth=1
                                        ;     Parent Loop BB0_11 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	v_add_u32_e32 v2, s55, v29
	v_cmp_gt_u32_e32 vcc, s54, v2
	s_or_b64 s[44:45], s[44:45], exec
	s_and_saveexec_b64 s[46:47], vcc
	s_cbranch_execz .LBB0_15
; %bb.17:                               ;   in Loop: Header=BB0_16 Depth=3
	v_lshl_add_u64 v[8:9], v[2:3], 1, s[40:41]
	s_mov_b64 s[48:49], -1
	s_and_b64 vcc, exec, s[36:37]
	s_cbranch_vccz .LBB0_19
; %bb.18:                               ;   in Loop: Header=BB0_16 Depth=3
	global_load_dwordx4 v[12:15], v[8:9], off
	s_mov_b64 s[48:49], 0
	s_waitcnt vmcnt(0)
	ds_write_b128 v10, v[12:15]
.LBB0_19:                               ;   in Loop: Header=BB0_16 Depth=3
	s_andn2_b64 vcc, exec, s[48:49]
	s_cbranch_vccnz .LBB0_14
; %bb.20:                               ;   in Loop: Header=BB0_16 Depth=3
	v_readfirstlane_b32 s48, v10
	s_mov_b32 m0, s48
	s_nop 0
	global_load_lds_dwordx4 v[8:9], off
	s_branch .LBB0_14
.LBB0_21:                               ;   in Loop: Header=BB0_11 Depth=2
	s_or_b64 exec, exec, s[42:43]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	s_and_saveexec_b64 s[40:41], s[22:23]
	s_cbranch_execz .LBB0_10
.LBB0_22:                               ;   in Loop: Header=BB0_11 Depth=2
	v_add_u32_e32 v32, s53, v1
	v_min_u32_e32 v2, s33, v32
	v_add_u32_e32 v33, 0x200, v32
	v_lshl_add_u64 v[16:17], v[2:3], 1, v[26:27]
	v_min_u32_e32 v2, s33, v33
	v_lshl_add_u64 v[18:19], v[2:3], 1, v[26:27]
	global_load_dwordx4 v[12:15], v[16:17], off nt
	global_load_dwordx4 v[8:11], v[18:19], off nt
	v_cmp_gt_u32_e32 vcc, s2, v32
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v16, 0
	v_mov_b32_e32 v17, 0
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v21, 0
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v23, 0
	s_and_saveexec_b64 s[42:43], vcc
	s_cbranch_execz .LBB0_9
; %bb.23:                               ;   in Loop: Header=BB0_11 Depth=2
	v_subrev_u32_e32 v2, s30, v32
	v_lshlrev_b32_e32 v2, 1, v2
	ds_read_b128 v[16:19], v2
	v_cmp_gt_u32_e32 vcc, s2, v33
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v21, 0
	v_mov_b32_e32 v20, 0
	s_and_saveexec_b64 s[44:45], vcc
	s_cbranch_execz .LBB0_8
; %bb.24:                               ;   in Loop: Header=BB0_11 Depth=2
	ds_read_b128 v[20:23], v2 offset:1024
	s_branch .LBB0_8
.LBB0_25:                               ;   in Loop: Header=BB0_5 Depth=1
	s_and_saveexec_b64 s[40:41], s[22:23]
	s_cbranch_execz .LBB0_4
; %bb.26:                               ;   in Loop: Header=BB0_5 Depth=1
	v_add_f32_dpp v2, v5, v4 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v2, v6, v2 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v2, v7, v2 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v2, v2, v2 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v2, v2, v2 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_mov_b32_dpp v2, v2 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v2, v2, v2 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_mov_b32_dpp v4, v2 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_and_b64 exec, exec, s[18:19]
	s_cbranch_execz .LBB0_4
; %bb.27:                               ;   in Loop: Header=BB0_5 Depth=1
	v_add_f32_e32 v2, v2, v4
	v_mul_f32_e32 v2, s24, v2
	s_and_b64 vcc, exec, s[20:21]
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v4, 16, v31
	s_cbranch_vccnz .LBB0_29
; %bb.28:                               ;   in Loop: Header=BB0_5 Depth=1
	v_lshlrev_b64 v[6:7], 1, v[24:25]
	v_lshl_add_u64 v[8:9], s[12:13], 0, v[6:7]
	global_load_ushort v5, v[8:9], off
	v_lshl_add_u64 v[6:7], s[14:15], 0, v[6:7]
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v5, 16, v5
	v_fma_f32 v5, s25, v5, v2
	v_add_f32_e32 v8, v5, v4
	v_cndmask_b32_e64 v5, v8, v5, s[0:1]
	v_cvt_pk_bf16_f32 v5, v5, s0
	global_store_short v[6:7], v5, off
	s_cbranch_execnz .LBB0_4
	s_branch .LBB0_30
.LBB0_29:                               ;   in Loop: Header=BB0_5 Depth=1
.LBB0_30:                               ;   in Loop: Header=BB0_5 Depth=1
	v_add_f32_e32 v4, v2, v4
	v_cndmask_b32_e64 v2, v4, v2, s[0:1]
	v_cvt_pk_bf16_f32 v2, v2, s0
	v_lshl_add_u64 v[4:5], v[24:25], 1, s[14:15]
	global_store_short v[4:5], v2, off
	s_branch .LBB0_4
.LBB0_31:
	s_or_b64 exec, exec, s[28:29]
	s_mov_b64 s[0:1], 0
.LBB0_32:
	s_andn2_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB0_69
; %bb.33:
	v_and_b32_e32 v2, 0x3ff, v0
	v_lshlrev_b32_e32 v26, 3, v2
	s_cmp_eq_u32 s11, s2
	s_mov_b64 s[18:19], 0
	s_cbranch_scc1 .LBB0_35
; %bb.34:
	s_and_b32 s20, s6, 15
	s_mov_b32 s21, 0
	s_mov_b64 s[0:1], -1
	s_min_u32 s9, s2, 0x14000
	v_lshl_add_u32 v3, v28, 9, v26
	s_cbranch_execz .LBB0_36
	s_branch .LBB0_37
.LBB0_35:
                                        ; implicit-def: $sgpr20_sgpr21
	s_mov_b64 s[0:1], 0
	s_min_u32 s9, s2, 0x14000
	v_lshl_add_u32 v3, v28, 9, v26
.LBB0_36:
	s_and_b32 s0, s2, 7
	s_cmp_lg_u32 s0, 0
	s_mov_b32 s21, 0
	s_cselect_b64 s[0:1], -1, 0
	s_and_b32 s20, s6, 15
	s_cmp_lg_u64 s[20:21], 0
	s_cselect_b64 s[18:19], -1, 0
	s_or_b64 s[0:1], s[0:1], s[18:19]
	s_mov_b64 s[18:19], -1
.LBB0_37:
	s_andn2_b64 vcc, exec, s[0:1]
	v_cmp_gt_u32_e64 s[0:1], s9, v3
	s_cbranch_vccz .LBB0_39
; %bb.38:
	s_andn2_b64 vcc, exec, s[18:19]
	s_cbranch_vccz .LBB0_47
	s_branch .LBB0_51
.LBB0_39:
	s_and_saveexec_b64 s[18:19], s[0:1]
	s_cbranch_execz .LBB0_46
; %bb.40:
	s_and_b32 s0, s11, 7
	s_cmp_lg_u32 s0, 0
	s_cselect_b64 s[22:23], -1, 0
	s_cmp_lg_u64 s[20:21], 0
	v_lshlrev_b32_e32 v0, 4, v2
	s_cselect_b64 s[20:21], -1, 0
	v_lshl_add_u32 v4, v28, 10, v0
	v_lshlrev_b32_e32 v0, 1, v3
	v_mov_b32_e32 v1, 0
	s_mov_b64 s[0:1], 0
	s_or_b64 s[20:21], s[22:23], s[20:21]
	v_lshl_add_u64 v[0:1], s[6:7], 0, v[0:1]
	s_mov_b64 s[22:23], 0x4000
	v_mov_b32_e32 v5, v3
	s_branch .LBB0_42
.LBB0_41:                               ;   in Loop: Header=BB0_42 Depth=1
	v_add_u32_e32 v5, 0x2000, v5
	v_cmp_le_u32_e32 vcc, s9, v5
	v_add_u32_e32 v4, 0x4000, v4
	s_or_b64 s[0:1], vcc, s[0:1]
	v_lshl_add_u64 v[0:1], v[0:1], 0, s[22:23]
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB0_46
.LBB0_42:                               ; =>This Inner Loop Header: Depth=1
	s_mov_b64 s[28:29], -1
	s_and_b64 vcc, exec, s[20:21]
	s_cbranch_vccz .LBB0_44
; %bb.43:                               ;   in Loop: Header=BB0_42 Depth=1
	global_load_dwordx4 v[6:9], v[0:1], off
	s_mov_b64 s[28:29], 0
	s_waitcnt vmcnt(0)
	ds_write_b128 v4, v[6:9]
.LBB0_44:                               ;   in Loop: Header=BB0_42 Depth=1
	s_andn2_b64 vcc, exec, s[28:29]
	s_cbranch_vccnz .LBB0_41
; %bb.45:                               ;   in Loop: Header=BB0_42 Depth=1
	v_readfirstlane_b32 s11, v4
	s_mov_b32 m0, s11
	s_nop 0
	global_load_lds_dwordx4 v[0:1], off
	s_branch .LBB0_41
.LBB0_46:
	s_or_b64 exec, exec, s[18:19]
	s_cbranch_execnz .LBB0_51
.LBB0_47:
	v_cmp_gt_u32_e32 vcc, s9, v3
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_50
; %bb.48:
	v_lshlrev_b32_e32 v0, 4, v2
	v_lshl_add_u32 v4, v28, 10, v0
	v_lshlrev_b32_e32 v0, 1, v3
	v_mov_b32_e32 v1, 0
	v_lshl_add_u64 v[0:1], s[6:7], 0, v[0:1]
	s_mov_b64 s[6:7], 0
	s_mov_b64 s[18:19], 0x4000
.LBB0_49:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s11, v4
	s_mov_b32 m0, s11
	v_add_u32_e32 v3, 0x2000, v3
	global_load_lds_dwordx4 v[0:1], off
	v_cmp_le_u32_e32 vcc, s9, v3
	v_add_u32_e32 v4, 0x4000, v4
	s_or_b64 s[6:7], vcc, s[6:7]
	v_lshl_add_u64 v[0:1], v[0:1], 0, s[18:19]
	s_andn2_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB0_49
.LBB0_50:
	s_or_b64 exec, exec, s[0:1]
.LBB0_51:
	s_lshl_b32 s0, s8, 4
	s_abs_i32 s1, s0
	v_cvt_f32_u32_e32 v0, s1
	s_mov_b32 s6, s3
	;;#ASMSTART
	;;#ASMEND
	s_sub_i32 s7, 0, s1
	v_rcp_iflag_f32_e32 v0, v0
	s_add_i32 s6, s6, -1
	s_add_i32 s9, s6, s0
	s_xor_b32 s0, s9, s0
	v_mul_f32_e32 v0, 0x4f7ffffe, v0
	v_cvt_u32_f32_e32 v0, v0
	s_abs_i32 s9, s9
	s_ashr_i32 s0, s0, 31
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_readfirstlane_b32 s11, v0
	s_mul_i32 s7, s7, s11
	s_mul_hi_u32 s7, s11, s7
	s_add_i32 s11, s11, s7
	s_mul_hi_u32 s7, s9, s11
	s_mul_i32 s11, s7, s1
	s_sub_i32 s9, s9, s11
	s_add_i32 s17, s7, 1
	s_sub_i32 s11, s9, s1
	s_cmp_ge_u32 s9, s1
	s_cselect_b32 s7, s17, s7
	s_cselect_b32 s9, s11, s9
	s_add_i32 s11, s7, 1
	s_cmp_ge_u32 s9, s1
	s_cselect_b32 s1, s11, s7
	s_xor_b32 s1, s1, s0
	s_sub_i32 s0, s1, s0
	s_mul_i32 s0, s0, s8
	s_abs_i32 s1, s0
	v_cvt_f32_u32_e32 v0, s1
	s_sub_i32 s7, 0, s1
	s_add_i32 s6, s0, s6
	s_xor_b32 s0, s6, s0
	v_rcp_iflag_f32_e32 v0, v0
	s_abs_i32 s6, s6
	s_ashr_i32 s0, s0, 31
	s_barrier
	v_mul_f32_e32 v0, 0x4f7ffffe, v0
	v_cvt_u32_f32_e32 v0, v0
	s_nop 0
	v_readfirstlane_b32 s9, v0
	s_mul_i32 s7, s7, s9
	s_mul_hi_u32 s7, s9, s7
	s_add_i32 s9, s9, s7
	s_mul_hi_u32 s7, s6, s9
	s_mul_i32 s9, s7, s1
	s_sub_i32 s6, s6, s9
	s_add_i32 s11, s7, 1
	s_sub_i32 s9, s6, s1
	s_cmp_ge_u32 s6, s1
	s_cselect_b32 s7, s11, s7
	s_cselect_b32 s6, s9, s6
	s_add_i32 s9, s7, 1
	s_cmp_ge_u32 s6, s1
	s_cselect_b32 s1, s9, s7
	s_xor_b32 s1, s1, s0
	s_sub_i32 s0, s1, s0
	s_max_i32 s11, s0, 4
	s_mul_i32 s0, s11, s16
	v_add_u32_e32 v20, s0, v28
	v_cmp_gt_u32_e32 vcc, s11, v28
	v_cmp_gt_u32_e64 s[0:1], s3, v20
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_69
; %bb.52:
	s_cmp_eq_u64 s[26:27], 0
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lg_u64 s[26:27], 0
	s_cselect_b64 s[6:7], -1, 0
	s_cmp_lg_u32 s2, 0
	s_cselect_b64 s[16:17], -1, 0
	v_cndmask_b32_e64 v0, 0, 1, s[6:7]
	v_cmp_ne_u32_e64 s[6:7], 1, v0
	v_cndmask_b32_e64 v0, 0, 1, s[16:17]
	v_cmp_neq_f32_e64 s[16:17], s25, 0
	s_mul_i32 s11, s11, s8
	v_cmp_ne_u32_e64 s[8:9], 1, v0
	v_cndmask_b32_e64 v0, 0, 1, s[16:17]
	s_mov_b64 s[20:21], 0
	v_cmp_eq_u32_e64 s[18:19], 63, v2
	s_add_i32 s30, s2, -8
	s_add_i32 s31, s3, -1
	s_ashr_i32 s33, s10, 31
	v_lshlrev_b32_e32 v27, 4, v2
	v_mov_b32_e32 v23, 0
	v_cmp_ne_u32_e64 s[16:17], 1, v0
                                        ; implicit-def: $vgpr28
	s_branch .LBB0_54
.LBB0_53:                               ;   in Loop: Header=BB0_54 Depth=1
	s_or_b64 exec, exec, s[22:23]
	v_add_u32_e32 v20, s11, v20
	v_cmp_le_u32_e32 vcc, s3, v20
	s_or_b64 s[20:21], vcc, s[20:21]
	s_andn2_b64 exec, exec, s[20:21]
	s_cbranch_execz .LBB0_69
.LBB0_54:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_60 Depth 2
	s_and_b64 vcc, exec, s[6:7]
	s_cbranch_vccnz .LBB0_56
; %bb.55:                               ;   in Loop: Header=BB0_54 Depth=1
	v_mov_b32_e32 v21, v23
	v_lshl_add_u64 v[0:1], v[20:21], 1, s[26:27]
	global_load_ushort v28, v[0:1], off
.LBB0_56:                               ;   in Loop: Header=BB0_54 Depth=1
	s_and_b64 vcc, exec, s[8:9]
	s_cbranch_vccnz .LBB0_63
; %bb.57:                               ;   in Loop: Header=BB0_54 Depth=1
	v_min_u32_e32 v3, s31, v20
	v_mad_u64_u32 v[0:1], s[22:23], v3, s10, 0
	v_mov_b32_e32 v2, v1
	v_mad_u64_u32 v[2:3], s[22:23], v3, s33, v[2:3]
	v_mov_b32_e32 v1, v2
	v_lshl_add_u64 v[24:25], v[0:1], 1, s[4:5]
	v_mov_b32_e32 v0, 0
	s_mov_b32 s34, 0
	v_mov_b32_e32 v21, v27
	v_mov_b32_e32 v1, v0
	v_mov_b32_e32 v2, v0
	v_mov_b32_e32 v3, v0
	s_branch .LBB0_60
.LBB0_58:                               ;   in Loop: Header=BB0_60 Depth=2
	s_or_b64 exec, exec, s[28:29]
.LBB0_59:                               ;   in Loop: Header=BB0_60 Depth=2
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(1) lgkmcnt(0)
	v_mfma_f32_4x4x4_16b_bf16 v[0:3], v[16:17], v[12:13], v[0:3]
	s_addk_i32 s34, 0x400
	v_add_u32_e32 v21, 0x800, v21
	v_mfma_f32_4x4x4_16b_bf16 v[0:3], v[18:19], v[14:15], v[0:3]
	s_cmp_ge_u32 s34, s2
	s_waitcnt vmcnt(0)
	v_mfma_f32_4x4x4_16b_bf16 v[0:3], v[8:9], v[4:5], v[0:3]
	s_nop 1
	v_mfma_f32_4x4x4_16b_bf16 v[0:3], v[10:11], v[6:7], v[0:3]
	s_cbranch_scc1 .LBB0_64
.LBB0_60:                               ;   Parent Loop BB0_54 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_add_u32_e32 v16, s34, v26
	v_min_u32_e32 v22, s30, v16
	v_add_u32_e32 v29, 0x200, v16
	v_lshl_add_u64 v[8:9], v[22:23], 1, v[24:25]
	v_min_u32_e32 v22, s30, v29
	v_lshl_add_u64 v[10:11], v[22:23], 1, v[24:25]
	global_load_dwordx4 v[12:15], v[8:9], off nt
	global_load_dwordx4 v[4:7], v[10:11], off nt
	v_cmp_gt_u32_e32 vcc, s2, v16
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v16, 0
	v_mov_b32_e32 v17, 0
	v_mov_b32_e32 v8, 0
	v_mov_b32_e32 v9, 0
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v11, 0
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_59
; %bb.61:                               ;   in Loop: Header=BB0_60 Depth=2
	ds_read_b128 v[16:19], v21
	v_cmp_gt_u32_e32 vcc, s2, v29
	v_mov_b32_e32 v11, 0
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v9, 0
	v_mov_b32_e32 v8, 0
	s_and_saveexec_b64 s[28:29], vcc
	s_cbranch_execz .LBB0_58
; %bb.62:                               ;   in Loop: Header=BB0_60 Depth=2
	ds_read_b128 v[8:11], v21 offset:1024
	s_branch .LBB0_58
.LBB0_63:                               ;   in Loop: Header=BB0_54 Depth=1
	v_mov_b32_e32 v3, 0
	v_mov_b32_e32 v2, v3
	v_mov_b32_e32 v1, v3
	v_mov_b32_e32 v0, v3
.LBB0_64:                               ;   in Loop: Header=BB0_54 Depth=1
	; sched_barrier mask(0x00000000)
	s_nop 3
	v_add_f32_dpp v0, v1, v0 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v0, v2, v0 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v0, v3, v0 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v0, v0, v0 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v0, v0, v0 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_mov_b32_dpp v0, v0 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v0, v0, v0 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_mov_b32_dpp v1, v0 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_and_saveexec_b64 s[22:23], s[18:19]
	s_cbranch_execz .LBB0_53
; %bb.65:                               ;   in Loop: Header=BB0_54 Depth=1
	v_add_f32_e32 v0, v0, v1
	v_mul_f32_e32 v0, s24, v0
	s_and_b64 vcc, exec, s[16:17]
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v1, 16, v28
	s_cbranch_vccnz .LBB0_67
; %bb.66:                               ;   in Loop: Header=BB0_54 Depth=1
	v_mov_b32_e32 v21, v23
	v_lshlrev_b64 v[2:3], 1, v[20:21]
	v_lshl_add_u64 v[4:5], s[12:13], 0, v[2:3]
	global_load_ushort v4, v[4:5], off
	v_lshl_add_u64 v[2:3], s[14:15], 0, v[2:3]
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v4, 16, v4
	v_fma_f32 v4, s25, v4, v0
	v_add_f32_e32 v5, v4, v1
	v_cndmask_b32_e64 v4, v5, v4, s[0:1]
	v_cvt_pk_bf16_f32 v4, v4, s0
	global_store_short v[2:3], v4, off
	s_cbranch_execnz .LBB0_53
	s_branch .LBB0_68
.LBB0_67:                               ;   in Loop: Header=BB0_54 Depth=1
.LBB0_68:                               ;   in Loop: Header=BB0_54 Depth=1
	v_add_f32_e32 v1, v0, v1
	v_cndmask_b32_e64 v0, v1, v0, s[0:1]
	v_mov_b32_e32 v21, v23
	v_cvt_pk_bf16_f32 v2, v0, s0
	v_lshl_add_u64 v[0:1], v[20:21], 1, s[14:15]
	global_store_short v[0:1], v2, off
	s_branch .LBB0_53
.LBB0_69:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wvSplitK_vllm_bf16_tn_bias_n1_y1u2
		.amdhsa_group_segment_fixed_size 163840
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 80
		.amdhsa_user_sgpr_count 16
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 14
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 1
		.amdhsa_next_free_vgpr 97
		.amdhsa_next_free_sgpr 96
		.amdhsa_accum_offset 36
		.amdhsa_reserve_vcc 1
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
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
	.size	wvSplitK_vllm_bf16_tn_bias_n1_y1u2, .Lfunc_end0-wvSplitK_vllm_bf16_tn_bias_n1_y1u2
                                        ; -- End function
	.set wvSplitK_vllm_bf16_tn_bias_n1_y1u2.num_vgpr, 34
	.set wvSplitK_vllm_bf16_tn_bias_n1_y1u2.num_agpr, 0
	.set wvSplitK_vllm_bf16_tn_bias_n1_y1u2.numbered_sgpr, 56
	.set wvSplitK_vllm_bf16_tn_bias_n1_y1u2.num_named_barrier, 0
	.set wvSplitK_vllm_bf16_tn_bias_n1_y1u2.private_seg_size, 0
	.set wvSplitK_vllm_bf16_tn_bias_n1_y1u2.uses_vcc, 1
	.set wvSplitK_vllm_bf16_tn_bias_n1_y1u2.uses_flat_scratch, 0
	.set wvSplitK_vllm_bf16_tn_bias_n1_y1u2.has_dyn_sized_stack, 0
	.set wvSplitK_vllm_bf16_tn_bias_n1_y1u2.has_recursion, 0
	.set wvSplitK_vllm_bf16_tn_bias_n1_y1u2.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 3212
; TotalNumSgprs: 62
; NumVgprs: 34
; NumAgprs: 0
; TotalNumVgprs: 34
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 163840 bytes/workgroup (compile time only)
; SGPRBlocks: 12
; VGPRBlocks: 12
; NumSGPRsForWavesPerEU: 102
; NumVGPRsForWavesPerEU: 97
; AccumOffset: 36
; Occupancy: 4
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 1
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 8
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_73e34e140ecc4cd9,@object ; @__hip_cuid_73e34e140ecc4cd9
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_73e34e140ecc4cd9
__hip_cuid_73e34e140ecc4cd9:
	.byte	0                               ; 0x0
	.size	__hip_cuid_73e34e140ecc4cd9, 1

	.ident	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.4 26084 f58b06dce1f9c15707c5f808fd002e18c2accf7e)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_73e34e140ecc4cd9
	.amdgpu_metadata
---
custom.config:
  Source:
    Origin: vllm
    Repository: "https://github.com/vllm-project/vllm"
  Version: 9163190dda009a310d8c175d63860ac7c671ca90
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
    DataType: b
    DestDataType: b
    ComputeDataType: s
    HighPrecisionAccumulate: True
    TransposeA: True
    TransposeB: False
    UseBeta: True
    Batched: True
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
            { type: address, semantic: AddressBias } ]
    macrotile: [64, 16, 8]
    threads: [64, 16, 1]
    grid: [ComputeUnits, One, One]
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 16, 1]
  EnableMatrixInstruction: True
  MIWaveTile: [1, 1]
  AssertSummationElementMultiple: 8
  AssertSizeEqual: { 1: 1, 2: 1 }
  AssertSizeGreaterThan: { 0: 8 }
  AssertStrideAEqual: { 0: 1 }
  AssertStrideBEqual: { 0: 1 }
  AssertStrideCEqual: { 0: 1 }
  AssertStrideDEqual: { 0: 1 }
  StaggerU: 0
  WavefrontSize: 64
amdhsa.kernels:
  - .agpr_count:     0
    .args:
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
      - .address_space:  global
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
    .group_segment_fixed_size: 163840
    .kernarg_segment_align: 8
    .kernarg_segment_size: 80
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           wvSplitK_vllm_bf16_tn_bias_n1_y1u2
    .private_segment_fixed_size: 0
    .sgpr_count:     62
    .sgpr_spill_count: 0
    .symbol:         wvSplitK_vllm_bf16_tn_bias_n1_y1u2.kd
    .vgpr_count:     34
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 1
...

	.end_amdgpu_metadata
