	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 4
	.text
	.protected	wvSplitK_vllm_bf16_tn_n4_y1u2 ; -- Begin function wvSplitK_vllm_bf16_tn_n4_y1u2
	.globl	wvSplitK_vllm_bf16_tn_n4_y1u2
	.p2align	8
	.type	wvSplitK_vllm_bf16_tn_n4_y1u2,@function
wvSplitK_vllm_bf16_tn_n4_y1u2:          ; @wvSplitK_vllm_bf16_tn_n4_y1u2
; %bb.77:
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.78:
.LBB0_0:
	s_load_dwordx4 s[24:27], s[0:1], 0x38
	s_mov_b32 s22, s11
	s_mov_b64 s[0:1], -1
	s_cmpk_gt_i32 s2, 0x5000
	v_bfe_u32 v64, v0, 10, 10
	s_cbranch_scc0 .LBB0_42
; %bb.1:
	s_lshl_b32 s0, s8, 4
	s_abs_i32 s1, s0
	v_cvt_f32_u32_e32 v1, s1
	s_sub_i32 s17, 0, s1
	s_add_i32 s9, s3, -1
	s_add_i32 s11, s0, s9
	v_rcp_iflag_f32_e32 v1, v1
	s_xor_b32 s0, s11, s0
	s_abs_i32 s11, s11
	s_ashr_i32 s0, s0, 31
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	s_nop 0
	v_readfirstlane_b32 s18, v1
	s_mul_i32 s17, s17, s18
	s_mul_hi_u32 s17, s18, s17
	s_add_i32 s18, s18, s17
	s_mul_hi_u32 s17, s11, s18
	s_mul_i32 s18, s17, s1
	s_sub_i32 s11, s11, s18
	s_add_i32 s19, s17, 1
	s_sub_i32 s18, s11, s1
	s_cmp_ge_u32 s11, s1
	s_cselect_b32 s17, s19, s17
	s_cselect_b32 s11, s18, s11
	s_add_i32 s18, s17, 1
	s_cmp_ge_u32 s11, s1
	s_cselect_b32 s1, s18, s17
	s_xor_b32 s1, s1, s0
	s_sub_i32 s0, s1, s0
	s_mul_i32 s0, s0, s8
	s_abs_i32 s1, s0
	v_cvt_f32_u32_e32 v1, s1
	s_sub_i32 s11, 0, s1
	s_add_i32 s9, s0, s9
	s_xor_b32 s0, s9, s0
	v_rcp_iflag_f32_e32 v1, v1
	s_abs_i32 s9, s9
	s_ashr_i32 s0, s0, 31
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	s_nop 0
	v_readfirstlane_b32 s17, v1
	s_mul_i32 s11, s11, s17
	s_mul_hi_u32 s11, s17, s11
	s_add_i32 s17, s17, s11
	s_mul_hi_u32 s11, s9, s17
	s_mul_i32 s17, s11, s1
	s_sub_i32 s9, s9, s17
	s_add_i32 s18, s11, 1
	s_sub_i32 s17, s9, s1
	s_cmp_ge_u32 s9, s1
	s_cselect_b32 s11, s18, s11
	s_cselect_b32 s9, s17, s9
	s_add_i32 s17, s11, 1
	s_cmp_ge_u32 s9, s1
	s_cselect_b32 s1, s17, s11
	s_xor_b32 s1, s1, s0
	s_sub_i32 s0, s1, s0
	s_max_i32 s18, s0, 4
	v_cmp_gt_u32_e32 vcc, s18, v64
	s_and_saveexec_b64 s[28:29], vcc
	s_cbranch_execz .LBB0_41
; %bb.2:
	v_cvt_f32_u32_e32 v1, s18
	s_mul_i32 s0, s18, s16
	s_sub_i32 s11, 0, s18
	v_add_u32_e32 v58, s0, v64
	v_rcp_iflag_f32_e32 v1, v1
	s_abs_i32 s9, s3
	s_ashr_i32 s1, s3, 31
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	s_nop 0
	v_readfirstlane_b32 s0, v1
	s_mul_i32 s11, s11, s0
	s_mul_hi_u32 s11, s0, s11
	s_add_i32 s0, s0, s11
	s_mul_hi_u32 s0, s9, s0
	s_mul_i32 s0, s0, s18
	s_sub_i32 s0, s9, s0
	s_sub_i32 s9, s0, s18
	s_cmp_ge_u32 s0, s18
	s_cselect_b32 s0, s9, s0
	s_sub_i32 s9, s0, s18
	s_cmp_ge_u32 s0, s18
	s_cselect_b32 s0, s9, s0
	s_xor_b32 s0, s0, s1
	s_sub_i32 s0, s0, s1
	s_add_i32 s1, s18, s3
	s_sub_i32 s1, s1, s0
	s_cmp_eq_u32 s0, 0
	s_cselect_b32 s9, s3, s1
	v_cmp_gt_u32_e32 vcc, s9, v58
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_41
; %bb.3:
	s_and_b32 s0, s22, 7
	s_cmp_lg_u32 s0, 0
	s_mov_b32 s31, 0
	s_cselect_b64 s[20:21], -1, 0
	s_and_b32 s30, s6, 15
	s_cmp_lg_u64 s[30:31], 0
	s_cselect_b64 s[48:49], -1, 0
	v_and_b32_e32 v2, 0x3ff, v0
	v_lshlrev_b32_e32 v1, 3, v2
	v_cmp_eq_u32_e64 s[0:1], 63, v2
	v_lshlrev_b32_e32 v2, 4, v2
	s_or_b64 s[48:49], s[48:49], s[20:21]
	s_ashr_i32 s23, s22, 31
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s39, s26, 31
	s_mov_b32 s38, s26
	s_ashr_i32 s41, s27, 31
	s_mov_b32 s40, s27
	v_lshl_add_u32 v66, v64, 10, v2
	v_cndmask_b32_e64 v2, 0, 1, s[48:49]
	s_mov_b64 s[34:35], 0
	v_lshl_add_u32 v65, v64, 9, v1
	s_lshl_b32 s11, s18, 9
	s_add_i32 s17, s2, -8
	s_ashr_i32 s33, s10, 31
	v_cmp_neq_f32_e64 s[36:37], s25, 0
	s_mul_i32 s60, s18, s8
	s_lshl_b64 s[42:43], s[22:23], 1
	s_lshl_b64 s[44:45], s[38:39], 1
	s_lshl_b64 s[46:47], s[40:41], 1
	s_lshl_b32 s61, s18, 10
	v_mov_b32_e32 v61, 0
	v_cmp_ne_u32_e64 s[20:21], 1, v2
	s_branch .LBB0_5
.LBB0_4:                                ;   in Loop: Header=BB0_5 Depth=1
	s_or_b64 exec, exec, s[50:51]
	v_add_u32_e32 v58, s60, v58
	v_cmp_le_u32_e32 vcc, s9, v58
	s_or_b64 s[34:35], vcc, s[34:35]
	s_andn2_b64 exec, exec, s[34:35]
	s_cbranch_execz .LBB0_41
.LBB0_5:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_9 Depth 2
                                        ;       Child Loop BB0_14 Depth 3
	v_mad_u64_u32 v[2:3], s[50:51], v58, s10, 0
	v_mov_b32_e32 v4, v3
	v_mad_u64_u32 v[4:5], s[50:51], v58, s33, v[4:5]
	v_mov_b32_e32 v3, v4
	v_cmp_gt_u32_e64 s[18:19], s3, v58
	v_mov_b32_e32 v59, v61
	v_lshl_add_u64 v[62:63], v[2:3], 1, s[4:5]
	s_mov_b32 s30, s31
	s_mov_b32 s62, s31
	v_mov_b32_e32 v14, v61
	v_mov_b32_e32 v15, v61
	v_mov_b32_e32 v16, v61
	v_mov_b32_e32 v17, v61
	v_mov_b32_e32 v10, v61
	v_mov_b32_e32 v11, v61
	v_mov_b32_e32 v12, v61
	v_mov_b32_e32 v13, v61
	v_mov_b32_e32 v6, v61
	v_mov_b32_e32 v7, v61
	v_mov_b32_e32 v8, v61
	v_mov_b32_e32 v9, v61
	v_mov_b32_e32 v2, v61
	v_mov_b32_e32 v3, v61
	v_mov_b32_e32 v4, v61
	v_mov_b32_e32 v5, v61
	s_branch .LBB0_9
.LBB0_6:                                ;   in Loop: Header=BB0_9 Depth=2
	s_or_b64 exec, exec, s[54:55]
.LBB0_7:                                ;   in Loop: Header=BB0_9 Depth=2
	s_or_b64 exec, exec, s[52:53]
	s_waitcnt vmcnt(1) lgkmcnt(3)
	v_mfma_f32_4x4x4_16b_bf16 v[14:17], v[50:51], v[22:23], v[14:17]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_4x4x4_16b_bf16 v[10:13], v[42:43], v[22:23], v[10:13]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_4x4x4_16b_bf16 v[6:9], v[30:31], v[22:23], v[6:9]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_4x4x4_16b_bf16 v[2:5], v[26:27], v[22:23], v[2:5]
	v_mfma_f32_4x4x4_16b_bf16 v[14:17], v[52:53], v[24:25], v[14:17]
	v_mfma_f32_4x4x4_16b_bf16 v[10:13], v[44:45], v[24:25], v[10:13]
	v_mfma_f32_4x4x4_16b_bf16 v[6:9], v[32:33], v[24:25], v[6:9]
	v_mfma_f32_4x4x4_16b_bf16 v[2:5], v[28:29], v[24:25], v[2:5]
	s_waitcnt vmcnt(0)
	v_mfma_f32_4x4x4_16b_bf16 v[14:17], v[38:39], v[18:19], v[14:17]
	v_mfma_f32_4x4x4_16b_bf16 v[10:13], v[34:35], v[18:19], v[10:13]
	v_mfma_f32_4x4x4_16b_bf16 v[6:9], v[46:47], v[18:19], v[6:9]
	v_mfma_f32_4x4x4_16b_bf16 v[2:5], v[54:55], v[18:19], v[2:5]
	v_mfma_f32_4x4x4_16b_bf16 v[14:17], v[40:41], v[20:21], v[14:17]
	v_mfma_f32_4x4x4_16b_bf16 v[10:13], v[36:37], v[20:21], v[10:13]
	v_mfma_f32_4x4x4_16b_bf16 v[6:9], v[48:49], v[20:21], v[6:9]
	v_mfma_f32_4x4x4_16b_bf16 v[2:5], v[56:57], v[20:21], v[2:5]
.LBB0_8:                                ;   in Loop: Header=BB0_9 Depth=2
	s_or_b64 exec, exec, s[50:51]
	s_addk_i32 s62, 0x400
	s_cmp_ge_u32 s62, s2
	s_cbranch_scc1 .LBB0_35
.LBB0_9:                                ;   Parent Loop BB0_5 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_14 Depth 3
	s_cmp_eq_u32 s62, 0
	s_cselect_b64 s[50:51], -1, 0
	s_add_i32 s52, s30, 0x5000
	s_cmp_eq_u32 s62, s52
	s_cselect_b64 s[54:55], -1, 0
	s_or_b64 s[54:55], s[50:51], s[54:55]
	s_andn2_b64 vcc, exec, s[54:55]
	s_cbranch_vccz .LBB0_11
; %bb.10:                               ;   in Loop: Header=BB0_9 Depth=2
	s_and_saveexec_b64 s[50:51], s[18:19]
	s_cbranch_execz .LBB0_8
	s_branch .LBB0_32
.LBB0_11:                               ;   in Loop: Header=BB0_9 Depth=2
	s_and_b64 s[50:51], s[50:51], exec
	s_cselect_b32 s30, s30, s52
	s_sub_i32 s50, s2, s30
	s_min_u32 s63, s50, 0x5000
	s_lshl_b64 s[50:51], s[30:31], 1
	s_add_u32 s50, s6, s50
	s_addc_u32 s51, s7, s51
	s_mov_b32 s64, 0
	s_mov_b64 s[52:53], 0
	v_mov_b32_e32 v20, v66
	s_barrier
                                        ; implicit-def: $sgpr54_sgpr55
	s_branch .LBB0_14
.LBB0_12:                               ;   in Loop: Header=BB0_14 Depth=3
	s_add_i32 s64, s64, s11
	s_cmpk_gt_u32 s64, 0x4fff
	s_cselect_b64 s[58:59], -1, 0
	s_andn2_b64 s[54:55], s[54:55], exec
	s_and_b64 s[58:59], s[58:59], exec
	v_add_u32_e32 v20, s61, v20
	s_or_b64 s[54:55], s[54:55], s[58:59]
.LBB0_13:                               ;   in Loop: Header=BB0_14 Depth=3
	s_or_b64 exec, exec, s[56:57]
	s_and_b64 s[56:57], exec, s[54:55]
	s_or_b64 s[52:53], s[56:57], s[52:53]
	s_andn2_b64 exec, exec, s[52:53]
	s_cbranch_execz .LBB0_31
.LBB0_14:                               ;   Parent Loop BB0_5 Depth=1
                                        ;     Parent Loop BB0_9 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	v_add_u32_e32 v60, s64, v65
	v_cmp_gt_u32_e32 vcc, s63, v60
	s_or_b64 s[54:55], s[54:55], exec
	s_and_saveexec_b64 s[56:57], vcc
	s_cbranch_execz .LBB0_13
; %bb.15:                               ;   in Loop: Header=BB0_14 Depth=3
	v_lshl_add_u64 v[18:19], v[60:61], 1, s[50:51]
	s_mov_b64 s[58:59], -1
	s_and_b64 vcc, exec, s[48:49]
	s_cbranch_vccz .LBB0_17
; %bb.16:                               ;   in Loop: Header=BB0_14 Depth=3
	global_load_dwordx4 v[22:25], v[18:19], off
	s_mov_b64 s[58:59], 0
	s_waitcnt vmcnt(0)
	ds_write_b128 v20, v[22:25]
.LBB0_17:                               ;   in Loop: Header=BB0_14 Depth=3
	s_andn2_b64 vcc, exec, s[58:59]
	s_cbranch_vccnz .LBB0_19
; %bb.18:                               ;   in Loop: Header=BB0_14 Depth=3
	v_readfirstlane_b32 s58, v20
	s_mov_b32 m0, s58
	s_nop 0
	global_load_lds_dwordx4 v[18:19], off
.LBB0_19:                               ;   in Loop: Header=BB0_14 Depth=3
	v_lshl_add_u64 v[18:19], s[22:23], 1, v[18:19]
	v_add_u32_e32 v21, 0xa000, v20
	s_and_b64 vcc, exec, s[20:21]
	s_mov_b64 s[58:59], -1
	s_cbranch_vccnz .LBB0_21
; %bb.20:                               ;   in Loop: Header=BB0_14 Depth=3
	global_load_dwordx4 v[22:25], v[18:19], off
	s_mov_b64 s[58:59], 0
	s_waitcnt vmcnt(0)
	ds_write_b128 v21, v[22:25]
.LBB0_21:                               ;   in Loop: Header=BB0_14 Depth=3
	s_andn2_b64 vcc, exec, s[58:59]
	s_cbranch_vccnz .LBB0_23
; %bb.22:                               ;   in Loop: Header=BB0_14 Depth=3
	v_readfirstlane_b32 s58, v21
	s_mov_b32 m0, s58
	s_nop 0
	global_load_lds_dwordx4 v[18:19], off
.LBB0_23:                               ;   in Loop: Header=BB0_14 Depth=3
	v_lshl_add_u64 v[18:19], v[18:19], 0, s[42:43]
	v_add_u32_e32 v21, 0x14000, v20
	s_and_b64 vcc, exec, s[20:21]
	s_mov_b64 s[58:59], -1
	s_cbranch_vccnz .LBB0_25
; %bb.24:                               ;   in Loop: Header=BB0_14 Depth=3
	global_load_dwordx4 v[22:25], v[18:19], off
	s_mov_b64 s[58:59], 0
	s_waitcnt vmcnt(0)
	ds_write_b128 v21, v[22:25]
.LBB0_25:                               ;   in Loop: Header=BB0_14 Depth=3
	s_andn2_b64 vcc, exec, s[58:59]
	s_cbranch_vccnz .LBB0_27
; %bb.26:                               ;   in Loop: Header=BB0_14 Depth=3
	v_readfirstlane_b32 s58, v21
	s_mov_b32 m0, s58
	s_nop 0
	global_load_lds_dwordx4 v[18:19], off
.LBB0_27:                               ;   in Loop: Header=BB0_14 Depth=3
	v_lshl_add_u64 v[18:19], v[18:19], 0, s[42:43]
	v_add_u32_e32 v21, 0x1e000, v20
	s_and_b64 vcc, exec, s[20:21]
	s_mov_b64 s[58:59], -1
	s_cbranch_vccnz .LBB0_29
; %bb.28:                               ;   in Loop: Header=BB0_14 Depth=3
	global_load_dwordx4 v[22:25], v[18:19], off
	s_mov_b64 s[58:59], 0
	s_waitcnt vmcnt(0)
	ds_write_b128 v21, v[22:25]
.LBB0_29:                               ;   in Loop: Header=BB0_14 Depth=3
	s_andn2_b64 vcc, exec, s[58:59]
	s_cbranch_vccnz .LBB0_12
; %bb.30:                               ;   in Loop: Header=BB0_14 Depth=3
	v_readfirstlane_b32 s58, v21
	s_mov_b32 m0, s58
	s_nop 0
	global_load_lds_dwordx4 v[18:19], off
	s_branch .LBB0_12
.LBB0_31:                               ;   in Loop: Header=BB0_9 Depth=2
	s_or_b64 exec, exec, s[52:53]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	s_and_saveexec_b64 s[50:51], s[18:19]
	s_cbranch_execz .LBB0_8
.LBB0_32:                               ;   in Loop: Header=BB0_9 Depth=2
	v_add_u32_e32 v68, s62, v1
	v_min_u32_e32 v60, s17, v68
	v_add_u32_e32 v67, 0x200, v68
	v_lshl_add_u64 v[18:19], v[60:61], 1, v[62:63]
	v_min_u32_e32 v60, s17, v67
	v_lshl_add_u64 v[20:21], v[60:61], 1, v[62:63]
	global_load_dwordx4 v[22:25], v[18:19], off nt
	s_nop 0
	global_load_dwordx4 v[18:21], v[20:21], off nt
	v_cmp_gt_u32_e32 vcc, s2, v68
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v39, 0
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v56, 0
	v_mov_b32_e32 v57, 0
	s_and_saveexec_b64 s[52:53], vcc
	s_cbranch_execz .LBB0_7
; %bb.33:                               ;   in Loop: Header=BB0_9 Depth=2
	v_subrev_u32_e32 v26, s30, v68
	v_lshlrev_b32_e32 v60, 1, v26
	v_add_u32_e32 v26, 0x14000, v60
	v_add_u32_e32 v27, 0x1e000, v60
	ds_read_b128 v[50:53], v60
	ds_read_b128 v[42:45], v60 offset:40960
	ds_read_b128 v[30:33], v26
	ds_read_b128 v[26:29], v27
	v_cmp_gt_u32_e32 vcc, s2, v67
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v56, 0
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v39, 0
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v48, 0
	s_and_saveexec_b64 s[54:55], vcc
	s_cbranch_execz .LBB0_6
; %bb.34:                               ;   in Loop: Header=BB0_9 Depth=2
	v_add_u32_e32 v46, 0x14400, v60
	v_add_u32_e32 v54, 0x1e400, v60
	ds_read_b128 v[38:41], v60 offset:1024
	ds_read_b128 v[34:37], v60 offset:41984
	ds_read_b128 v[46:49], v46
	ds_read_b128 v[54:57], v54
	s_branch .LBB0_6
.LBB0_35:                               ;   in Loop: Header=BB0_5 Depth=1
	s_and_saveexec_b64 s[50:51], s[18:19]
	s_cbranch_execz .LBB0_4
; %bb.36:                               ;   in Loop: Header=BB0_5 Depth=1
	v_add_f32_dpp v14, v15, v14 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v10, v11, v10 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v7, v6 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v3, v2 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v14, v16, v14 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v10, v12, v10 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v8, v6 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v4, v2 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v14, v17, v14 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v10, v13, v10 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v9, v6 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v5, v2 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v14, v14, v14 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v10, v10, v10 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v6, v6 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v14, v14, v14 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v10, v10, v10 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v6, v6 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v14, v14 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v10, v10 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v6, v6 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v2, v2 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v14, v14, v14 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v10, v10, v10 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v6, v6 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v2, v2 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v15, v14 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v11, v10 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v7, v6 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v8, v3 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_and_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB0_4
; %bb.37:                               ;   in Loop: Header=BB0_5 Depth=1
	v_add_f32_e32 v5, v14, v15
	v_add_f32_e32 v4, v10, v11
	v_add_f32_e32 v2, v6, v7
	s_andn2_b64 vcc, exec, s[36:37]
	v_add_f32_e32 v3, v3, v8
	s_cbranch_vccnz .LBB0_39
; %bb.38:                               ;   in Loop: Header=BB0_5 Depth=1
	v_lshlrev_b64 v[6:7], 1, v[58:59]
	v_lshl_add_u64 v[8:9], s[12:13], 0, v[6:7]
	global_load_ushort v10, v[8:9], off
	v_lshl_add_u64 v[6:7], s[14:15], 0, v[6:7]
	v_lshl_add_u64 v[8:9], s[38:39], 1, v[8:9]
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v10, 16, v10
	v_mul_f32_e32 v10, s25, v10
	v_fmac_f32_e32 v10, s24, v5
	v_cvt_pk_bf16_f32 v10, v10, s0
	global_store_short v[6:7], v10, off
	global_load_ushort v10, v[8:9], off
	v_lshl_add_u64 v[6:7], s[40:41], 1, v[6:7]
	v_lshl_add_u64 v[8:9], v[8:9], 0, s[44:45]
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v10, 16, v10
	v_mul_f32_e32 v10, s25, v10
	v_fmac_f32_e32 v10, s24, v4
	v_cvt_pk_bf16_f32 v10, v10, s0
	global_store_short v[6:7], v10, off
	global_load_ushort v10, v[8:9], off
	v_lshl_add_u64 v[6:7], v[6:7], 0, s[46:47]
	v_lshl_add_u64 v[8:9], v[8:9], 0, s[44:45]
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v10, 16, v10
	v_mul_f32_e32 v10, s25, v10
	v_fmac_f32_e32 v10, s24, v2
	v_cvt_pk_bf16_f32 v10, v10, s0
	global_store_short v[6:7], v10, off
	global_load_ushort v8, v[8:9], off
	v_lshl_add_u64 v[6:7], v[6:7], 0, s[46:47]
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v8, 16, v8
	v_mul_f32_e32 v8, s25, v8
	v_fmac_f32_e32 v8, s24, v3
	v_cvt_pk_bf16_f32 v8, v8, s0
	global_store_short v[6:7], v8, off
	s_cbranch_execnz .LBB0_4
	s_branch .LBB0_40
.LBB0_39:                               ;   in Loop: Header=BB0_5 Depth=1
.LBB0_40:                               ;   in Loop: Header=BB0_5 Depth=1
	v_mul_f32_e32 v5, s24, v5
	v_lshl_add_u64 v[6:7], v[58:59], 1, s[14:15]
	v_cvt_pk_bf16_f32 v5, v5, s0
	v_mul_f32_e32 v4, s24, v4
	global_store_short v[6:7], v5, off
	v_lshl_add_u64 v[6:7], s[40:41], 1, v[6:7]
	v_cvt_pk_bf16_f32 v4, v4, s0
	v_mul_f32_e32 v2, s24, v2
	global_store_short v[6:7], v4, off
	v_lshl_add_u64 v[4:5], v[6:7], 0, s[46:47]
	v_cvt_pk_bf16_f32 v2, v2, s0
	global_store_short v[4:5], v2, off
	v_mul_f32_e32 v2, s24, v3
	v_lshl_add_u64 v[4:5], v[4:5], 0, s[46:47]
	v_cvt_pk_bf16_f32 v2, v2, s0
	global_store_short v[4:5], v2, off
	s_branch .LBB0_4
.LBB0_41:
	s_or_b64 exec, exec, s[28:29]
	s_mov_b64 s[0:1], 0
.LBB0_42:
	s_andn2_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB0_76
; %bb.43:
	s_lshl_b32 s11, s2, 2
	v_and_b32_e32 v8, 0x3ff, v0
	v_lshlrev_b32_e32 v62, 3, v8
	s_cmp_eq_u32 s22, s2
	s_mov_b64 s[18:19], 0
	s_cbranch_scc1 .LBB0_45
; %bb.44:
	s_and_b32 s20, s6, 15
	s_mov_b32 s21, 0
	s_mov_b64 s[0:1], -1
	s_min_u32 s9, s11, 0x14000
	v_lshl_add_u32 v9, v64, 9, v62
	s_cbranch_execz .LBB0_46
	s_branch .LBB0_47
.LBB0_45:
                                        ; implicit-def: $sgpr20_sgpr21
	s_mov_b64 s[0:1], 0
	s_min_u32 s9, s11, 0x14000
	v_lshl_add_u32 v9, v64, 9, v62
.LBB0_46:
	s_and_b32 s0, s2, 7
	s_cmp_lg_u32 s0, 0
	s_mov_b32 s21, 0
	s_cselect_b64 s[0:1], -1, 0
	s_and_b32 s20, s6, 15
	s_cmp_lg_u64 s[20:21], 0
	s_cselect_b64 s[18:19], -1, 0
	s_or_b64 s[0:1], s[0:1], s[18:19]
	s_mov_b64 s[18:19], -1
.LBB0_47:
	s_andn2_b64 vcc, exec, s[0:1]
	v_cmp_gt_u32_e64 s[0:1], s9, v9
	s_cbranch_vccz .LBB0_49
; %bb.48:
	s_andn2_b64 vcc, exec, s[18:19]
	s_cbranch_vccz .LBB0_57
	s_branch .LBB0_61
.LBB0_49:
	s_and_saveexec_b64 s[18:19], s[0:1]
	s_cbranch_execz .LBB0_56
; %bb.50:
	v_cvt_f32_u32_e32 v0, s2
	s_and_b32 s17, s22, 7
	s_cmp_lg_u32 s17, 0
	s_cselect_b64 s[28:29], -1, 0
	v_rcp_iflag_f32_e32 v0, v0
	s_cmp_lg_u64 s[20:21], 0
	s_cselect_b64 s[20:21], -1, 0
	s_sub_i32 s23, 0, s2
	v_mul_f32_e32 v0, 0x4f7ffffe, v0
	v_cvt_u32_f32_e32 v0, v0
	s_or_b64 s[20:21], s[28:29], s[20:21]
	s_mov_b64 s[0:1], 0
	s_ashr_i32 s17, s22, 31
	v_mul_lo_u32 v1, s23, v0
	v_mul_hi_u32 v1, v0, v1
	v_add_u32_e32 v0, v0, v1
	v_mov_b32_e32 v1, 0
	v_mad_u64_u32 v[2:3], s[28:29], v9, v0, 0
	v_lshlrev_b64 v[4:5], 13, v[0:1]
	v_lshlrev_b32_e32 v0, 4, v8
	v_lshl_add_u32 v10, v64, 10, v0
	v_mov_b32_e32 v0, v9
	s_branch .LBB0_52
.LBB0_51:                               ;   in Loop: Header=BB0_52 Depth=1
	v_add_u32_e32 v0, 0x2000, v0
	v_cmp_le_u32_e32 vcc, s9, v0
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[4:5]
	s_or_b64 s[0:1], vcc, s[0:1]
	v_add_u32_e32 v10, 0x4000, v10
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB0_56
.LBB0_52:                               ; =>This Inner Loop Header: Depth=1
	v_not_b32_e32 v11, v3
	v_mad_u64_u32 v[6:7], s[28:29], s23, v3, v[0:1]
	v_add_u32_e32 v7, 1, v3
	v_cmp_le_u32_e32 vcc, s2, v6
	v_mad_u64_u32 v[12:13], s[28:29], s2, v11, v[0:1]
	s_nop 0
	v_cndmask_b32_e32 v7, v3, v7, vcc
	v_cndmask_b32_e32 v6, v6, v12, vcc
	v_add_u32_e32 v11, 1, v7
	v_cmp_le_u32_e32 vcc, s2, v6
	s_nop 1
	v_cndmask_b32_e32 v11, v7, v11, vcc
	v_mad_u64_u32 v[12:13], s[28:29], v11, s22, 0
	v_mov_b32_e32 v14, v13
	v_mad_u64_u32 v[14:15], s[28:29], v11, s17, v[14:15]
	v_mad_u64_u32 v[6:7], s[28:29], s23, v11, v[0:1]
	v_mov_b32_e32 v13, v14
	v_mov_b32_e32 v7, v1
	v_lshl_add_u64 v[12:13], v[12:13], 1, s[6:7]
	v_lshl_add_u64 v[6:7], v[6:7], 1, v[12:13]
	s_mov_b64 s[28:29], -1
	s_and_b64 vcc, exec, s[20:21]
	s_cbranch_vccz .LBB0_54
; %bb.53:                               ;   in Loop: Header=BB0_52 Depth=1
	global_load_dwordx4 v[12:15], v[6:7], off
	s_mov_b64 s[28:29], 0
	s_waitcnt vmcnt(0)
	ds_write_b128 v10, v[12:15]
.LBB0_54:                               ;   in Loop: Header=BB0_52 Depth=1
	s_andn2_b64 vcc, exec, s[28:29]
	s_cbranch_vccnz .LBB0_51
; %bb.55:                               ;   in Loop: Header=BB0_52 Depth=1
	v_readfirstlane_b32 s28, v10
	s_mov_b32 m0, s28
	s_nop 0
	global_load_lds_dwordx4 v[6:7], off
	s_branch .LBB0_51
.LBB0_56:
	s_or_b64 exec, exec, s[18:19]
	s_cbranch_execnz .LBB0_61
.LBB0_57:
	v_cmp_gt_u32_e32 vcc, s9, v9
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_60
; %bb.58:
	v_lshlrev_b32_e32 v0, 4, v8
	v_lshl_add_u32 v2, v64, 10, v0
	v_lshlrev_b32_e32 v0, 1, v9
	v_mov_b32_e32 v1, 0
	v_lshl_add_u64 v[0:1], s[6:7], 0, v[0:1]
	s_mov_b64 s[6:7], 0
	s_mov_b64 s[18:19], 0x4000
.LBB0_59:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s17, v2
	s_mov_b32 m0, s17
	v_add_u32_e32 v9, 0x2000, v9
	global_load_lds_dwordx4 v[0:1], off
	v_cmp_le_u32_e32 vcc, s9, v9
	v_add_u32_e32 v2, 0x4000, v2
	s_or_b64 s[6:7], vcc, s[6:7]
	v_lshl_add_u64 v[0:1], v[0:1], 0, s[18:19]
	s_andn2_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB0_59
.LBB0_60:
	s_or_b64 exec, exec, s[0:1]
.LBB0_61:
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
	v_readfirstlane_b32 s17, v0
	s_mul_i32 s7, s7, s17
	s_mul_hi_u32 s7, s17, s7
	s_add_i32 s17, s17, s7
	s_mul_hi_u32 s7, s9, s17
	s_mul_i32 s17, s7, s1
	s_sub_i32 s9, s9, s17
	s_add_i32 s18, s7, 1
	s_sub_i32 s17, s9, s1
	s_cmp_ge_u32 s9, s1
	s_cselect_b32 s7, s18, s7
	s_cselect_b32 s9, s17, s9
	s_add_i32 s17, s7, 1
	s_cmp_ge_u32 s9, s1
	s_cselect_b32 s1, s17, s7
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
	s_add_i32 s17, s7, 1
	s_sub_i32 s9, s6, s1
	s_cmp_ge_u32 s6, s1
	s_cselect_b32 s7, s17, s7
	s_cselect_b32 s6, s9, s6
	s_add_i32 s9, s7, 1
	s_cmp_ge_u32 s6, s1
	s_cselect_b32 s1, s9, s7
	s_xor_b32 s1, s1, s0
	s_sub_i32 s0, s1, s0
	s_max_i32 s33, s0, 4
	s_mul_i32 s0, s33, s16
	v_add_u32_e32 v56, s0, v64
	v_cmp_gt_u32_e32 vcc, s33, v64
	v_cmp_gt_u32_e64 s[0:1], s3, v56
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_76
; %bb.62:
	s_cmp_lg_u32 s2, 0
	s_cselect_b64 s[6:7], -1, 0
	s_ashr_i32 s17, s26, 31
	s_mov_b32 s16, s26
	s_ashr_i32 s19, s27, 31
	s_mov_b32 s18, s27
	v_cndmask_b32_e64 v0, 0, 1, s[6:7]
	v_cmp_eq_u32_e64 s[0:1], 63, v8
	s_mul_i32 s33, s33, s8
	s_add_i32 s34, s2, -8
	s_add_i32 s35, s3, -1
	s_ashr_i32 s36, s10, 31
	v_cmp_neq_f32_e64 s[8:9], s25, 0
	s_lshl_b32 s37, s2, 1
	s_lshl_b64 s[20:21], s[16:17], 1
	s_lshl_b64 s[22:23], s[18:19], 1
	v_lshlrev_b32_e32 v63, 4, v8
	s_mul_i32 s38, s2, 6
	s_mov_b64 s[26:27], 0
	v_cmp_ne_u32_e64 s[6:7], 1, v0
	v_mov_b32_e32 v59, 0
	s_branch .LBB0_64
.LBB0_63:                               ;   in Loop: Header=BB0_64 Depth=1
	s_or_b64 exec, exec, s[28:29]
	v_add_u32_e32 v56, s33, v56
	v_cmp_le_u32_e32 vcc, s3, v56
	s_or_b64 s[26:27], vcc, s[26:27]
	s_andn2_b64 exec, exec, s[26:27]
	s_cbranch_execz .LBB0_76
.LBB0_64:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_68 Depth 2
	s_and_b64 vcc, exec, s[6:7]
	v_mov_b32_e32 v7, v59
	v_mov_b32_e32 v6, v59
	v_mov_b32_e32 v5, v59
	v_mov_b32_e32 v4, v59
	v_mov_b32_e32 v3, v59
	v_mov_b32_e32 v2, v59
	v_mov_b32_e32 v1, v59
	v_mov_b32_e32 v0, v59
	v_mov_b32_e32 v11, v59
	v_mov_b32_e32 v10, v59
	v_mov_b32_e32 v9, v59
	v_mov_b32_e32 v8, v59
	v_mov_b32_e32 v15, v59
	v_mov_b32_e32 v14, v59
	v_mov_b32_e32 v13, v59
	v_mov_b32_e32 v12, v59
	s_cbranch_vccnz .LBB0_71
; %bb.65:                               ;   in Loop: Header=BB0_64 Depth=1
	v_min_u32_e32 v3, s35, v56
	v_mad_u64_u32 v[0:1], s[28:29], v3, s10, 0
	v_mov_b32_e32 v2, v1
	v_mad_u64_u32 v[2:3], s[28:29], v3, s36, v[2:3]
	v_mov_b32_e32 v1, v2
	v_mov_b32_e32 v12, 0
	v_lshl_add_u64 v[60:61], v[0:1], 1, s[4:5]
	s_mov_b32 s39, 0
	v_mov_b32_e32 v57, v63
	v_mov_b32_e32 v13, v12
	v_mov_b32_e32 v14, v12
	v_mov_b32_e32 v15, v12
	v_mov_b32_e32 v8, v12
	v_mov_b32_e32 v9, v12
	v_mov_b32_e32 v10, v12
	v_mov_b32_e32 v11, v12
	v_mov_b32_e32 v0, v12
	v_mov_b32_e32 v1, v12
	v_mov_b32_e32 v2, v12
	v_mov_b32_e32 v3, v12
	v_mov_b32_e32 v4, v12
	v_mov_b32_e32 v5, v12
	v_mov_b32_e32 v6, v12
	v_mov_b32_e32 v7, v12
	s_branch .LBB0_68
.LBB0_66:                               ;   in Loop: Header=BB0_68 Depth=2
	s_or_b64 exec, exec, s[30:31]
.LBB0_67:                               ;   in Loop: Header=BB0_68 Depth=2
	s_or_b64 exec, exec, s[28:29]
	s_waitcnt vmcnt(1) lgkmcnt(1)
	v_mfma_f32_4x4x4_16b_bf16 v[12:15], v[52:53], v[20:21], v[12:15]
	s_addk_i32 s39, 0x400
	v_mfma_f32_4x4x4_16b_bf16 v[8:11], v[44:45], v[20:21], v[8:11]
	v_add_u32_e32 v57, 0x800, v57
	v_mfma_f32_4x4x4_16b_bf16 v[0:3], v[28:29], v[20:21], v[0:3]
	s_cmp_ge_u32 s39, s2
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_4x4x4_16b_bf16 v[4:7], v[24:25], v[20:21], v[4:7]
	v_mfma_f32_4x4x4_16b_bf16 v[12:15], v[54:55], v[22:23], v[12:15]
	v_mfma_f32_4x4x4_16b_bf16 v[8:11], v[46:47], v[22:23], v[8:11]
	v_mfma_f32_4x4x4_16b_bf16 v[0:3], v[30:31], v[22:23], v[0:3]
	v_mfma_f32_4x4x4_16b_bf16 v[4:7], v[26:27], v[22:23], v[4:7]
	s_waitcnt vmcnt(0)
	v_mfma_f32_4x4x4_16b_bf16 v[12:15], v[48:49], v[16:17], v[12:15]
	v_mfma_f32_4x4x4_16b_bf16 v[8:11], v[40:41], v[16:17], v[8:11]
	v_mfma_f32_4x4x4_16b_bf16 v[0:3], v[36:37], v[16:17], v[0:3]
	v_mfma_f32_4x4x4_16b_bf16 v[4:7], v[32:33], v[16:17], v[4:7]
	v_mfma_f32_4x4x4_16b_bf16 v[12:15], v[50:51], v[18:19], v[12:15]
	v_mfma_f32_4x4x4_16b_bf16 v[8:11], v[42:43], v[18:19], v[8:11]
	v_mfma_f32_4x4x4_16b_bf16 v[0:3], v[38:39], v[18:19], v[0:3]
	v_mfma_f32_4x4x4_16b_bf16 v[4:7], v[34:35], v[18:19], v[4:7]
	s_cbranch_scc1 .LBB0_71
.LBB0_68:                               ;   Parent Loop BB0_64 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_add_u32_e32 v28, s39, v62
	v_min_u32_e32 v58, s34, v28
	v_add_u32_e32 v64, 0x200, v28
	v_lshl_add_u64 v[24:25], v[58:59], 1, v[60:61]
	v_min_u32_e32 v58, s34, v64
	v_lshl_add_u64 v[26:27], v[58:59], 1, v[60:61]
	global_load_dwordx4 v[20:23], v[24:25], off nt
	global_load_dwordx4 v[16:19], v[26:27], off nt
	v_cmp_gt_u32_e32 vcc, s2, v28
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v24, 0
	v_mov_b32_e32 v25, 0
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v39, 0
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v35, 0
	s_and_saveexec_b64 s[28:29], vcc
	s_cbranch_execz .LBB0_67
; %bb.69:                               ;   in Loop: Header=BB0_68 Depth=2
	v_add_u32_e32 v66, s37, v57
	v_add_u32_e32 v65, s11, v57
	ds_read_b128 v[44:47], v66
	ds_read_b128 v[28:31], v65
	v_add_u32_e32 v58, s38, v57
	ds_read_b128 v[52:55], v57
	ds_read_b128 v[24:27], v58
	v_cmp_gt_u32_e32 vcc, s2, v64
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v39, 0
	v_mov_b32_e32 v38, 0
	s_and_saveexec_b64 s[30:31], vcc
	s_cbranch_execz .LBB0_66
; %bb.70:                               ;   in Loop: Header=BB0_68 Depth=2
	ds_read_b128 v[40:43], v66 offset:1024
	ds_read_b128 v[36:39], v65 offset:1024
	ds_read_b128 v[48:51], v57 offset:1024
	ds_read_b128 v[32:35], v58 offset:1024
	s_branch .LBB0_66
.LBB0_71:                               ;   in Loop: Header=BB0_64 Depth=1
	; sched_barrier mask(0x00000000)
	s_nop 2
	v_add_f32_dpp v0, v1, v0 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v12, v13, v12 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v9, v8 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v2, v0 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v5, v4 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v12, v14, v12 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v10, v8 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v6, v2 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v12, v15, v12 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v11, v8 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v3, v0 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v7, v2 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v12, v12, v12 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v8, v8 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v0, v0 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v12, v12, v12 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v8, v8 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v0, v0 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v12, v12 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v8, v8 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v0, v0 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v2, v2 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v12, v12, v12 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v8, v8 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v0, v0 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v2, v2 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v13, v12 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v9, v8 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v1, v0 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v5, v4 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_and_saveexec_b64 s[28:29], s[0:1]
	s_cbranch_execz .LBB0_63
; %bb.72:                               ;   in Loop: Header=BB0_64 Depth=1
	v_add_f32_e32 v3, v12, v13
	v_add_f32_e32 v2, v8, v9
	v_add_f32_e32 v1, v0, v1
	v_add_f32_e32 v0, v4, v5
	s_andn2_b64 vcc, exec, s[8:9]
	v_mov_b32_e32 v57, v59
	s_cbranch_vccnz .LBB0_74
; %bb.73:                               ;   in Loop: Header=BB0_64 Depth=1
	v_lshlrev_b64 v[4:5], 1, v[56:57]
	v_lshl_add_u64 v[6:7], s[12:13], 0, v[4:5]
	global_load_ushort v8, v[6:7], off
	v_lshl_add_u64 v[4:5], s[14:15], 0, v[4:5]
	v_lshl_add_u64 v[6:7], s[16:17], 1, v[6:7]
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v8, 16, v8
	v_mul_f32_e32 v8, s25, v8
	v_fmac_f32_e32 v8, s24, v3
	v_cvt_pk_bf16_f32 v8, v8, s0
	global_store_short v[4:5], v8, off
	global_load_ushort v8, v[6:7], off
	v_lshl_add_u64 v[4:5], s[18:19], 1, v[4:5]
	v_lshl_add_u64 v[6:7], v[6:7], 0, s[20:21]
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v8, 16, v8
	v_mul_f32_e32 v8, s25, v8
	v_fmac_f32_e32 v8, s24, v2
	v_cvt_pk_bf16_f32 v8, v8, s0
	global_store_short v[4:5], v8, off
	global_load_ushort v8, v[6:7], off
	v_lshl_add_u64 v[4:5], v[4:5], 0, s[22:23]
	v_lshl_add_u64 v[6:7], v[6:7], 0, s[20:21]
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v8, 16, v8
	v_mul_f32_e32 v8, s25, v8
	v_fmac_f32_e32 v8, s24, v1
	v_cvt_pk_bf16_f32 v8, v8, s0
	global_store_short v[4:5], v8, off
	global_load_ushort v6, v[6:7], off
	v_lshl_add_u64 v[4:5], v[4:5], 0, s[22:23]
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v6, 16, v6
	v_mul_f32_e32 v6, s25, v6
	v_fmac_f32_e32 v6, s24, v0
	v_cvt_pk_bf16_f32 v6, v6, s0
	global_store_short v[4:5], v6, off
	s_cbranch_execnz .LBB0_63
	s_branch .LBB0_75
.LBB0_74:                               ;   in Loop: Header=BB0_64 Depth=1
.LBB0_75:                               ;   in Loop: Header=BB0_64 Depth=1
	v_mul_f32_e32 v3, s24, v3
	v_lshl_add_u64 v[4:5], v[56:57], 1, s[14:15]
	v_cvt_pk_bf16_f32 v3, v3, s0
	v_mul_f32_e32 v2, s24, v2
	global_store_short v[4:5], v3, off
	v_lshl_add_u64 v[4:5], s[18:19], 1, v[4:5]
	v_cvt_pk_bf16_f32 v2, v2, s0
	v_mul_f32_e32 v1, s24, v1
	global_store_short v[4:5], v2, off
	v_lshl_add_u64 v[2:3], v[4:5], 0, s[22:23]
	v_cvt_pk_bf16_f32 v1, v1, s0
	v_mul_f32_e32 v0, s24, v0
	global_store_short v[2:3], v1, off
	v_lshl_add_u64 v[2:3], v[2:3], 0, s[22:23]
	v_cvt_pk_bf16_f32 v0, v0, s0
	global_store_short v[2:3], v0, off
	s_branch .LBB0_63
.LBB0_76:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wvSplitK_vllm_bf16_tn_n4_y1u2
		.amdhsa_group_segment_fixed_size 163840
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 72
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
		.amdhsa_accum_offset 72
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
	.size	wvSplitK_vllm_bf16_tn_n4_y1u2, .Lfunc_end0-wvSplitK_vllm_bf16_tn_n4_y1u2
                                        ; -- End function
	.set wvSplitK_vllm_bf16_tn_n4_y1u2.num_vgpr, 69
	.set wvSplitK_vllm_bf16_tn_n4_y1u2.num_agpr, 0
	.set wvSplitK_vllm_bf16_tn_n4_y1u2.numbered_sgpr, 65
	.set wvSplitK_vllm_bf16_tn_n4_y1u2.num_named_barrier, 0
	.set wvSplitK_vllm_bf16_tn_n4_y1u2.private_seg_size, 0
	.set wvSplitK_vllm_bf16_tn_n4_y1u2.uses_vcc, 1
	.set wvSplitK_vllm_bf16_tn_n4_y1u2.uses_flat_scratch, 0
	.set wvSplitK_vllm_bf16_tn_n4_y1u2.has_dyn_sized_stack, 0
	.set wvSplitK_vllm_bf16_tn_n4_y1u2.has_recursion, 0
	.set wvSplitK_vllm_bf16_tn_n4_y1u2.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 5092
; TotalNumSgprs: 71
; NumVgprs: 69
; NumAgprs: 0
; TotalNumVgprs: 69
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 163840 bytes/workgroup (compile time only)
; SGPRBlocks: 12
; VGPRBlocks: 12
; NumSGPRsForWavesPerEU: 102
; NumVGPRsForWavesPerEU: 97
; AccumOffset: 72
; Occupancy: 4
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 1
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 17
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_6243a8c6a1c71994,@object ; @__hip_cuid_6243a8c6a1c71994
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_6243a8c6a1c71994
__hip_cuid_6243a8c6a1c71994:
	.byte	0                               ; 0x0
	.size	__hip_cuid_6243a8c6a1c71994, 1

	.ident	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.4 26084 f58b06dce1f9c15707c5f808fd002e18c2accf7e)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_6243a8c6a1c71994
	.amdgpu_metadata
---
custom.config:
  Source:
    Origin: vllm
    Repository: "https://github.com/vllm-project/vllm"
  Version: 9163190dda009a310d8c175d63860ac7c671ca90
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
    UseBeta: True
    Batched: True
    UseBias: 0
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
            { type: int32, semantic: StrideD0 } ]
    macrotile: [64, 16, 8]
    threads: [64, 16, 1]
    grid: [ComputeUnits, One, One]
  MatrixInstruction: [16, 16, 16, 1]
  EnableMatrixInstruction: True
  MIWaveTile: [1, 1]
  AssertSummationElementMultiple: 8
  AssertSizeEqual: { 1: 4, 2: 1 }
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
    .group_segment_fixed_size: 163840
    .kernarg_segment_align: 8
    .kernarg_segment_size: 72
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           wvSplitK_vllm_bf16_tn_n4_y1u2
    .private_segment_fixed_size: 0
    .sgpr_count:     71
    .sgpr_spill_count: 0
    .symbol:         wvSplitK_vllm_bf16_tn_n4_y1u2.kd
    .vgpr_count:     69
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 1
...

	.end_amdgpu_metadata
