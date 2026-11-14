	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 4
	.text
	.protected	wvSplitKQ_vllm_f8_bf16_tn_n2_y2u2 ; -- Begin function wvSplitKQ_vllm_f8_bf16_tn_n2_y2u2
	.globl	wvSplitKQ_vllm_f8_bf16_tn_n2_y2u2
	.p2align	8
	.type	wvSplitKQ_vllm_f8_bf16_tn_n2_y2u2,@function
wvSplitKQ_vllm_f8_bf16_tn_n2_y2u2:      ; @wvSplitKQ_vllm_f8_bf16_tn_n2_y2u2
; %bb.87:
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.88:
.LBB0_0:
	s_mov_b32 s24, s11
	s_mov_b32 s11, s3
	s_mov_b32 s3, 0
	s_cmp_lt_u32 s2, 0x28000
	s_cselect_b32 s42, s2, 0x28000
	s_and_b32 s18, s6, 15
	s_mov_b32 s19, s3
	s_load_dwordx4 s[20:23], s[0:1], 0x48
	s_cmp_lg_u64 s[18:19], 0
	v_and_b32_e32 v51, 0x3ff, v0
	s_cselect_b64 s[18:19], -1, 0
	s_and_b32 s9, s24, 15
	v_bfe_u32 v4, v0, 10, 10
	s_cmp_lg_u32 s9, 0
	v_lshlrev_b32_e32 v48, 4, v51
	s_cselect_b64 s[28:29], -1, 0
	v_lshl_add_u32 v0, v4, 10, v48
	s_mov_b64 s[26:27], 0
	s_or_b64 s[18:19], s[18:19], s[28:29]
	v_cmp_gt_u32_e32 vcc, s42, v0
	s_and_saveexec_b64 s[28:29], vcc
	s_cbranch_execz .LBB0_7
; %bb.1:
	v_mov_b32_e32 v3, 0
	v_lshlrev_b32_e32 v2, 10, v4
	v_mov_b32_e32 v49, v3
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[48:49]
	v_lshl_add_u64 v[2:3], s[6:7], 0, v[2:3]
	s_mov_b64 s[30:31], 0x4000
	v_mov_b32_e32 v1, v0
	s_branch .LBB0_3
.LBB0_2:                                ;   in Loop: Header=BB0_3 Depth=1
	v_add_u32_e32 v1, 0x4000, v1
	v_cmp_le_u32_e32 vcc, s42, v1
	s_or_b64 s[26:27], vcc, s[26:27]
	v_lshl_add_u64 v[2:3], v[2:3], 0, s[30:31]
	s_andn2_b64 exec, exec, s[26:27]
	s_cbranch_execz .LBB0_7
.LBB0_3:                                ; =>This Inner Loop Header: Depth=1
	s_mov_b64 s[34:35], -1
	s_and_b64 vcc, exec, s[18:19]
	s_cbranch_vccz .LBB0_5
; %bb.4:                                ;   in Loop: Header=BB0_3 Depth=1
	global_load_dwordx4 v[6:9], v[2:3], off
	s_mov_b64 s[34:35], 0
	s_waitcnt vmcnt(0)
	ds_write_b128 v1, v[6:9]
.LBB0_5:                                ;   in Loop: Header=BB0_3 Depth=1
	s_andn2_b64 vcc, exec, s[34:35]
	s_cbranch_vccnz .LBB0_2
; %bb.6:                                ;   in Loop: Header=BB0_3 Depth=1
	v_readfirstlane_b32 s9, v1
	s_mov_b32 m0, s9
	s_nop 0
	global_load_lds_dwordx4 v[2:3], off
	s_branch .LBB0_2
.LBB0_7:
	s_or_b64 exec, exec, s[28:29]
	s_sub_u32 s26, 0x28000, s2
	s_subb_u32 s27, 0, 0
	v_mov_b64_e32 v[2:3], s[2:3]
	v_cmp_lt_u64_e32 vcc, s[26:27], v[2:3]
	s_and_b64 s[28:29], vcc, exec
	s_cselect_b32 s3, s26, s2
	s_cmp_lt_u32 s2, 0x28000
	s_cselect_b32 s43, s3, 0
	s_mov_b32 s25, 0
	v_cmp_gt_u32_e32 vcc, s43, v0
	s_and_saveexec_b64 s[26:27], vcc
	s_cbranch_execz .LBB0_14
; %bb.8:
	s_add_u32 s28, s6, s24
	v_cndmask_b32_e64 v2, 0, 1, s[18:19]
	s_addc_u32 s29, s7, 0
	s_mov_b64 s[30:31], 0
	v_mov_b32_e32 v1, 0
	v_cmp_ne_u32_e64 s[18:19], 1, v2
	s_branch .LBB0_10
.LBB0_9:                                ;   in Loop: Header=BB0_10 Depth=1
	v_add_u32_e32 v0, 0x4000, v0
	v_cmp_le_u32_e32 vcc, s43, v0
	s_or_b64 s[30:31], vcc, s[30:31]
	s_andn2_b64 exec, exec, s[30:31]
	s_cbranch_execz .LBB0_14
.LBB0_10:                               ; =>This Inner Loop Header: Depth=1
	v_lshl_add_u64 v[2:3], s[28:29], 0, v[0:1]
	v_add_u32_e32 v5, s2, v0
	s_and_b64 vcc, exec, s[18:19]
	s_mov_b64 s[34:35], -1
	s_cbranch_vccnz .LBB0_12
; %bb.11:                               ;   in Loop: Header=BB0_10 Depth=1
	global_load_dwordx4 v[6:9], v[2:3], off
	s_mov_b64 s[34:35], 0
	s_waitcnt vmcnt(0)
	ds_write_b128 v5, v[6:9]
.LBB0_12:                               ;   in Loop: Header=BB0_10 Depth=1
	s_andn2_b64 vcc, exec, s[34:35]
	s_cbranch_vccnz .LBB0_9
; %bb.13:                               ;   in Loop: Header=BB0_10 Depth=1
	v_readfirstlane_b32 s3, v5
	s_mov_b32 m0, s3
	s_nop 0
	global_load_lds_dwordx4 v[2:3], off
	s_branch .LBB0_9
.LBB0_14:
	s_or_b64 exec, exec, s[26:27]
	s_add_i32 s3, s11, -1
	s_lshl_b32 s17, s8, 5
	v_mov_b32_e32 v0, s3
	v_add_co_u32_e32 v0, vcc, s17, v0
	s_nop 0
	v_readfirstlane_b32 s18, v0
	s_andn2_b64 vcc, exec, vcc
	s_mov_b32 s9, 4
	s_cbranch_vccz .LBB0_16
; %bb.15:
	v_cvt_f32_u32_e32 v0, s17
	s_sub_i32 s9, 0, s17
	v_rcp_iflag_f32_e32 v0, v0
	s_nop 0
	v_mul_f32_e32 v0, 0x4f7ffffe, v0
	v_cvt_u32_f32_e32 v0, v0
	s_nop 0
	v_readfirstlane_b32 s19, v0
	s_mul_i32 s9, s9, s19
	s_mul_hi_u32 s9, s19, s9
	s_add_i32 s19, s19, s9
	s_mul_hi_u32 s9, s18, s19
	s_mul_i32 s19, s9, s17
	s_sub_i32 s18, s18, s19
	s_add_i32 s26, s9, 1
	s_sub_i32 s19, s18, s17
	s_cmp_ge_u32 s18, s17
	s_cselect_b32 s9, s26, s9
	s_cselect_b32 s18, s19, s18
	s_add_i32 s19, s9, 1
	s_cmp_ge_u32 s18, s17
	s_cselect_b32 s9, s19, s9
	s_mul_i32 s9, s8, s9
	s_lshl_b32 s9, s9, 1
	v_cvt_f32_u32_e32 v0, s9
	s_sub_i32 s18, 0, s9
	s_add_i32 s17, s9, s3
	v_rcp_iflag_f32_e32 v0, v0
	s_nop 0
	v_mul_f32_e32 v0, 0x4f7ffffe, v0
	v_cvt_u32_f32_e32 v0, v0
	s_nop 0
	v_readfirstlane_b32 s19, v0
	s_mul_i32 s18, s18, s19
	s_mul_hi_u32 s18, s19, s18
	s_add_i32 s19, s19, s18
	s_mul_hi_u32 s18, s17, s19
	s_mul_i32 s19, s18, s9
	s_sub_i32 s17, s17, s19
	s_add_i32 s26, s18, 1
	s_sub_i32 s19, s17, s9
	s_cmp_ge_u32 s17, s9
	s_cselect_b32 s18, s26, s18
	s_cselect_b32 s17, s19, s17
	s_add_i32 s19, s18, 1
	s_cmp_ge_u32 s17, s9
	s_cselect_b32 s9, s19, s18
	s_max_u32 s9, s9, 4
.LBB0_16:
	s_mov_b32 s17, 1.0
	s_waitcnt lgkmcnt(0)
	s_cmp_eq_u64 s[20:21], 0
	s_mov_b32 s18, 1.0
	s_cbranch_scc1 .LBB0_18
; %bb.17:
	s_load_dword s18, s[20:21], 0x0
.LBB0_18:
	s_cmp_eq_u64 s[22:23], 0
	s_cbranch_scc1 .LBB0_20
; %bb.19:
	s_load_dword s17, s[22:23], 0x0
.LBB0_20:
	v_cmp_gt_u32_e32 vcc, s9, v4
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_86
; %bb.21:
	s_cmp_eq_u32 s43, s2
	s_cselect_b64 s[20:21], -1, 0
	s_cmp_eq_u32 s2, s42
	s_cselect_b64 s[22:23], -1, 0
	s_and_b64 s[26:27], s[22:23], s[20:21]
	s_load_dwordx4 s[20:23], s[0:1], 0x38
	v_mov_b32_e32 v0, s18
	s_mul_i32 s0, s9, s16
	s_mul_i32 s8, s8, s9
	v_add_lshl_u32 v50, s0, v4, 1
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v0, s20, v0
	s_lshl_b32 s33, s8, 1
	v_mul_f32_e32 v49, s17, v0
	s_mov_b64 s[8:9], -1
	s_andn2_b64 vcc, exec, s[26:27]
	v_cmp_gt_u32_e64 s[0:1], s11, v50
	s_cbranch_vccz .LBB0_62
; %bb.22:
	s_and_saveexec_b64 s[26:27], s[0:1]
	s_cbranch_execz .LBB0_61
; %bb.23:
	v_mbcnt_lo_u32_b32 v0, -1, 0
	v_mbcnt_hi_u32_b32 v0, -1, v0
	v_and_b32_e32 v1, 63, v0
	v_cmp_gt_u32_e32 vcc, 44, v1
	s_cmp_lg_u32 s2, 0
	s_cselect_b64 s[8:9], -1, 0
	v_cndmask_b32_e64 v2, 0, 20, vcc
	v_cmp_gt_u32_e32 vcc, 24, v1
	s_mov_b32 s29, 0
	v_add_lshl_u32 v64, v2, v0, 2
	v_cndmask_b32_e64 v1, 0, 40, vcc
	v_add_lshl_u32 v65, v1, v0, 2
	v_cndmask_b32_e64 v0, 0, 1, s[8:9]
	s_add_i32 s20, s2, -16
	v_cmp_eq_u32_e64 s[0:1], 0, v51
	v_cmp_neq_f32_e64 s[30:31], s21, 0
	s_mov_b32 s28, s22
	s_mov_b32 s34, s23
	s_mov_b32 s35, s29
	v_add_u32_e32 v66, s2, v48
	s_mov_b64 s[36:37], 0
	v_cmp_ne_u32_e64 s[8:9], 1, v0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v54, v50
	s_branch .LBB0_26
.LBB0_24:                               ;   in Loop: Header=BB0_26 Depth=1
	v_cvt_pk_bf16_f32 v0, v4, s0
	global_store_short v[2:3], v0, off offset:2
.LBB0_25:                               ;   in Loop: Header=BB0_26 Depth=1
	s_or_b64 exec, exec, s[38:39]
	v_add_u32_e32 v54, s33, v54
	v_cmp_le_u32_e32 vcc, s11, v54
	s_or_b64 s[36:37], vcc, s[36:37]
	s_andn2_b64 exec, exec, s[36:37]
	s_cbranch_execz .LBB0_61
.LBB0_26:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_31 Depth 2
	s_and_b64 vcc, exec, s[8:9]
	v_or_b32_e32 v67, 1, v54
	v_mov_b32_e32 v15, v53
	v_mov_b32_e32 v14, v53
	v_mov_b32_e32 v13, v53
	v_mov_b32_e32 v12, v53
	s_waitcnt lgkmcnt(3)
	v_mov_b32_e32 v11, v53
	v_mov_b32_e32 v10, v53
	s_waitcnt lgkmcnt(2)
	v_mov_b32_e32 v9, v53
	v_mov_b32_e32 v8, v53
	v_mov_b32_e32 v3, v53
	v_mov_b32_e32 v2, v53
	v_mov_b32_e32 v1, v53
	v_mov_b32_e32 v0, v53
	s_waitcnt lgkmcnt(1)
	v_mov_b32_e32 v7, v53
	v_mov_b32_e32 v6, v53
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v5, v53
	v_mov_b32_e32 v4, v53
	s_cbranch_vccnz .LBB0_49
; %bb.27:                               ;   in Loop: Header=BB0_26 Depth=1
	v_min_u32_e32 v0, s3, v54
	v_mad_u64_u32 v[56:57], s[16:17], v0, s10, 0
	v_min_u32_e32 v0, s3, v67
	v_mov_b32_e32 v4, 0
	v_mad_u64_u32 v[58:59], s[16:17], v0, s10, 0
	s_mov_b32 s40, 0
	v_mov_b32_e32 v5, v4
	v_mov_b32_e32 v6, v4
	v_mov_b32_e32 v7, v4
	v_mov_b32_e32 v0, v4
	v_mov_b32_e32 v1, v4
	v_mov_b32_e32 v2, v4
	v_mov_b32_e32 v3, v4
	v_mov_b32_e32 v8, v4
	v_mov_b32_e32 v9, v4
	v_mov_b32_e32 v10, v4
	v_mov_b32_e32 v11, v4
	v_mov_b32_e32 v12, v4
	v_mov_b32_e32 v13, v4
	v_mov_b32_e32 v14, v4
	v_mov_b32_e32 v15, v4
	s_branch .LBB0_31
.LBB0_28:                               ;   in Loop: Header=BB0_31 Depth=2
	s_or_b64 exec, exec, s[38:39]
.LBB0_29:                               ;   in Loop: Header=BB0_31 Depth=2
	s_or_b64 exec, exec, s[18:19]
.LBB0_30:                               ;   in Loop: Header=BB0_31 Depth=2
	s_or_b64 exec, exec, s[16:17]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_mfma_f32_16x16x32_fp8_fp8 v[12:15], v[32:33], v[24:25], v[12:15]
	s_addk_i32 s40, 0x800
	s_cmp_ge_u32 s40, s2
	v_mfma_f32_16x16x32_fp8_fp8 v[8:11], v[32:33], v[28:29], v[8:11]
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[36:37], v[24:25], v[0:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[4:7], v[36:37], v[28:29], v[4:7]
	v_mfma_f32_16x16x32_fp8_fp8 v[12:15], v[34:35], v[26:27], v[12:15]
	v_mfma_f32_16x16x32_fp8_fp8 v[8:11], v[34:35], v[30:31], v[8:11]
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[38:39], v[26:27], v[0:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[4:7], v[38:39], v[30:31], v[4:7]
	v_mfma_f32_16x16x32_fp8_fp8 v[12:15], v[40:41], v[20:21], v[12:15]
	v_mfma_f32_16x16x32_fp8_fp8 v[8:11], v[40:41], v[16:17], v[8:11]
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[44:45], v[20:21], v[0:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[4:7], v[44:45], v[16:17], v[4:7]
	v_mfma_f32_16x16x32_fp8_fp8 v[12:15], v[42:43], v[22:23], v[12:15]
	v_mfma_f32_16x16x32_fp8_fp8 v[8:11], v[42:43], v[18:19], v[8:11]
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[46:47], v[22:23], v[0:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[4:7], v[46:47], v[18:19], v[4:7]
	s_cbranch_scc1 .LBB0_49
.LBB0_31:                               ;   Parent Loop BB0_26 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_add_u32_e32 v62, s40, v48
	v_min_u32_e32 v52, s20, v62
	v_lshl_add_u64 v[16:17], s[4:5], 0, v[52:53]
	v_add_u32_e32 v60, 0x400, v62
	v_lshl_add_u64 v[18:19], v[16:17], 0, v[56:57]
	v_lshl_add_u64 v[16:17], v[16:17], 0, v[58:59]
	v_min_u32_e32 v52, s20, v60
	global_load_dwordx4 v[24:27], v[18:19], off nt
	global_load_dwordx4 v[28:31], v[16:17], off nt
	v_lshl_add_u64 v[16:17], s[4:5], 0, v[52:53]
	v_lshl_add_u64 v[32:33], v[16:17], 0, v[56:57]
	v_lshl_add_u64 v[34:35], v[16:17], 0, v[58:59]
	global_load_dwordx4 v[20:23], v[32:33], off nt
	global_load_dwordx4 v[16:19], v[34:35], off nt
	v_cmp_gt_u32_e32 vcc, s2, v62
	v_mov_b64_e32 v[46:47], 0
	v_mov_b64_e32 v[44:45], 0
	v_mov_b64_e32 v[38:39], 0
	v_mov_b64_e32 v[36:37], 0
	v_mov_b64_e32 v[42:43], 0
	v_mov_b64_e32 v[40:41], 0
	v_mov_b64_e32 v[34:35], 0
	v_mov_b64_e32 v[32:33], 0
	s_and_saveexec_b64 s[16:17], vcc
	s_cbranch_execz .LBB0_30
; %bb.32:                               ;   in Loop: Header=BB0_31 Depth=2
	v_mov_b32_e32 v63, v53
	v_lshl_add_u64 v[36:37], s[6:7], 0, v[62:63]
	v_cmp_le_u32_e32 vcc, s42, v62
                                        ; implicit-def: $vgpr34_vgpr35
	s_and_saveexec_b64 s[18:19], vcc
	s_xor_b64 s[18:19], exec, s[18:19]
	s_cbranch_execz .LBB0_34
; %bb.33:                               ;   in Loop: Header=BB0_31 Depth=2
	global_load_dwordx4 v[32:35], v[36:37], off
.LBB0_34:                               ;   in Loop: Header=BB0_31 Depth=2
	s_andn2_saveexec_b64 s[18:19], s[18:19]
	s_cbranch_execz .LBB0_36
; %bb.35:                               ;   in Loop: Header=BB0_31 Depth=2
	s_waitcnt vmcnt(0)
	ds_read_b128 v[32:35], v62
.LBB0_36:                               ;   in Loop: Header=BB0_31 Depth=2
	s_or_b64 exec, exec, s[18:19]
	v_cmp_le_u32_e32 vcc, s43, v62
                                        ; implicit-def: $vgpr38_vgpr39
	s_and_saveexec_b64 s[18:19], vcc
	s_xor_b64 s[18:19], exec, s[18:19]
	s_cbranch_execz .LBB0_38
; %bb.37:                               ;   in Loop: Header=BB0_31 Depth=2
	v_lshl_add_u64 v[36:37], v[36:37], 0, s[24:25]
	global_load_dwordx4 v[36:39], v[36:37], off
.LBB0_38:                               ;   in Loop: Header=BB0_31 Depth=2
	s_andn2_saveexec_b64 s[18:19], s[18:19]
	s_cbranch_execz .LBB0_40
; %bb.39:                               ;   in Loop: Header=BB0_31 Depth=2
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v36, s40, v66
	ds_read_b128 v[36:39], v36
.LBB0_40:                               ;   in Loop: Header=BB0_31 Depth=2
	s_or_b64 exec, exec, s[18:19]
	v_cmp_gt_u32_e32 vcc, s2, v60
	v_mov_b64_e32 v[46:47], 0
	v_mov_b64_e32 v[44:45], 0
	v_mov_b64_e32 v[42:43], 0
	v_mov_b64_e32 v[40:41], 0
	s_and_saveexec_b64 s[18:19], vcc
	s_cbranch_execz .LBB0_29
; %bb.41:                               ;   in Loop: Header=BB0_31 Depth=2
	v_mov_b32_e32 v61, v53
	v_lshl_add_u64 v[44:45], s[6:7], 0, v[60:61]
	v_cmp_le_u32_e32 vcc, s42, v60
                                        ; implicit-def: $vgpr42_vgpr43
	s_and_saveexec_b64 s[38:39], vcc
	s_xor_b64 s[38:39], exec, s[38:39]
	s_cbranch_execz .LBB0_43
; %bb.42:                               ;   in Loop: Header=BB0_31 Depth=2
	global_load_dwordx4 v[40:43], v[44:45], off
                                        ; implicit-def: $vgpr62
.LBB0_43:                               ;   in Loop: Header=BB0_31 Depth=2
	s_andn2_saveexec_b64 s[38:39], s[38:39]
	s_cbranch_execz .LBB0_45
; %bb.44:                               ;   in Loop: Header=BB0_31 Depth=2
	s_waitcnt vmcnt(0)
	ds_read_b128 v[40:43], v62 offset:1024
.LBB0_45:                               ;   in Loop: Header=BB0_31 Depth=2
	s_or_b64 exec, exec, s[38:39]
	v_cmp_le_u32_e32 vcc, s43, v60
                                        ; implicit-def: $vgpr46_vgpr47
	s_and_saveexec_b64 s[38:39], vcc
	s_xor_b64 s[38:39], exec, s[38:39]
	s_cbranch_execz .LBB0_47
; %bb.46:                               ;   in Loop: Header=BB0_31 Depth=2
	v_lshl_add_u64 v[44:45], v[44:45], 0, s[24:25]
	global_load_dwordx4 v[44:47], v[44:45], off
.LBB0_47:                               ;   in Loop: Header=BB0_31 Depth=2
	s_andn2_saveexec_b64 s[38:39], s[38:39]
	s_cbranch_execz .LBB0_28
; %bb.48:                               ;   in Loop: Header=BB0_31 Depth=2
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v44, s40, v66
	ds_read_b128 v[44:47], v44 offset:1024
	s_branch .LBB0_28
.LBB0_49:                               ;   in Loop: Header=BB0_26 Depth=1
	s_nop 5
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
	ds_bpermute_b32 v13, v64, v12
	ds_bpermute_b32 v9, v64, v8
	ds_bpermute_b32 v1, v64, v0
	ds_bpermute_b32 v3, v64, v2
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v10, v12, v13
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v8, v8, v9
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v6, v0, v1
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v4, v2, v3
	ds_bpermute_b32 v11, v65, v10
	ds_bpermute_b32 v9, v65, v8
	ds_bpermute_b32 v7, v65, v6
	ds_bpermute_b32 v5, v65, v4
	s_and_saveexec_b64 s[38:39], s[0:1]
	s_cbranch_execz .LBB0_25
; %bb.50:                               ;   in Loop: Header=BB0_26 Depth=1
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v2, v10, v11
	v_mov_b32_e32 v55, v53
	v_cndmask_b32_e64 v3, 0, 1, s[30:31]
	v_lshl_add_u64 v[0:1], v[54:55], 1, s[12:13]
	v_cmp_ne_u32_e64 s[18:19], 1, v3
	s_andn2_b64 vcc, exec, s[30:31]
	v_mul_f32_e32 v10, v49, v2
	s_cbranch_vccnz .LBB0_52
; %bb.51:                               ;   in Loop: Header=BB0_26 Depth=1
	global_load_ushort v2, v[0:1], off
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v2, 16, v2
	v_fmac_f32_e32 v10, s21, v2
.LBB0_52:                               ;   in Loop: Header=BB0_26 Depth=1
	v_lshl_add_u64 v[2:3], v[54:55], 1, s[14:15]
	v_cvt_pk_bf16_f32 v10, v10, s0
	v_cmp_gt_u32_e64 s[16:17], s11, v67
	global_store_short v[2:3], v10, off
	s_and_saveexec_b64 s[40:41], s[16:17]
	s_cbranch_execz .LBB0_56
; %bb.53:                               ;   in Loop: Header=BB0_26 Depth=1
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v8, v8, v9
	s_and_b64 vcc, exec, s[18:19]
	v_mul_f32_e32 v8, v49, v8
	s_cbranch_vccnz .LBB0_55
; %bb.54:                               ;   in Loop: Header=BB0_26 Depth=1
	global_load_ushort v9, v[0:1], off offset:2
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v9, 16, v9
	v_fmac_f32_e32 v8, s21, v9
.LBB0_55:                               ;   in Loop: Header=BB0_26 Depth=1
	v_cvt_pk_bf16_f32 v8, v8, s0
	global_store_short v[2:3], v8, off offset:2
.LBB0_56:                               ;   in Loop: Header=BB0_26 Depth=1
	s_or_b64 exec, exec, s[40:41]
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v6, v6, v7
	v_lshl_add_u64 v[0:1], s[28:29], 1, v[0:1]
	s_and_b64 vcc, exec, s[18:19]
	v_mul_f32_e32 v6, v49, v6
	s_cbranch_vccnz .LBB0_58
; %bb.57:                               ;   in Loop: Header=BB0_26 Depth=1
	global_load_ushort v7, v[0:1], off
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v7, 16, v7
	v_fmac_f32_e32 v6, s21, v7
.LBB0_58:                               ;   in Loop: Header=BB0_26 Depth=1
	v_lshl_add_u64 v[2:3], s[34:35], 1, v[2:3]
	v_cvt_pk_bf16_f32 v6, v6, s0
	global_store_short v[2:3], v6, off
	s_and_b64 exec, exec, s[16:17]
	s_cbranch_execz .LBB0_25
; %bb.59:                               ;   in Loop: Header=BB0_26 Depth=1
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v4, v4, v5
	s_and_b64 vcc, exec, s[18:19]
	v_mul_f32_e32 v4, v49, v4
	s_cbranch_vccnz .LBB0_24
; %bb.60:                               ;   in Loop: Header=BB0_26 Depth=1
	global_load_ushort v0, v[0:1], off offset:2
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v0, 16, v0
	v_fmac_f32_e32 v4, s21, v0
	s_branch .LBB0_24
.LBB0_61:
	s_or_b64 exec, exec, s[26:27]
	s_mov_b64 s[8:9], 0
.LBB0_62:
	s_andn2_b64 vcc, exec, s[8:9]
	s_cbranch_vccnz .LBB0_86
; %bb.63:
	v_cmp_gt_u32_e32 vcc, s11, v50
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_86
; %bb.64:
	v_mbcnt_lo_u32_b32 v0, -1, 0
	v_mbcnt_hi_u32_b32 v0, -1, v0
	v_and_b32_e32 v1, 63, v0
	v_cmp_gt_u32_e32 vcc, 44, v1
	s_cmp_lg_u32 s2, 0
	s_cselect_b64 s[6:7], -1, 0
	v_cndmask_b32_e64 v2, 0, 20, vcc
	v_cmp_gt_u32_e32 vcc, 24, v1
	s_mov_b32 s19, 0
	v_add_lshl_u32 v58, v2, v0, 2
	v_cndmask_b32_e64 v1, 0, 40, vcc
	v_add_lshl_u32 v59, v1, v0, 2
	v_cndmask_b32_e64 v0, 0, 1, s[6:7]
	s_add_i32 s20, s2, -16
	v_cmp_eq_u32_e64 s[0:1], 0, v51
	v_cmp_neq_f32_e64 s[24:25], s21, 0
	s_mov_b32 s18, s22
	s_mov_b32 s22, s23
	s_mov_b32 s23, s19
	v_add_u32_e32 v60, s2, v48
	s_mov_b64 s[26:27], 0
	v_cmp_ne_u32_e64 s[6:7], 1, v0
	v_mov_b32_e32 v53, 0
	s_branch .LBB0_67
.LBB0_65:                               ;   in Loop: Header=BB0_67 Depth=1
	v_cvt_pk_bf16_f32 v0, v4, s0
	global_store_short v[2:3], v0, off offset:2
.LBB0_66:                               ;   in Loop: Header=BB0_67 Depth=1
	s_or_b64 exec, exec, s[28:29]
	v_add_u32_e32 v50, s33, v50
	v_cmp_le_u32_e32 vcc, s11, v50
	s_or_b64 s[26:27], vcc, s[26:27]
	s_andn2_b64 exec, exec, s[26:27]
	s_cbranch_execz .LBB0_86
.LBB0_67:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_71 Depth 2
	s_and_b64 vcc, exec, s[6:7]
	v_or_b32_e32 v61, 1, v50
	v_mov_b32_e32 v15, v53
	v_mov_b32_e32 v14, v53
	v_mov_b32_e32 v13, v53
	v_mov_b32_e32 v12, v53
	s_waitcnt lgkmcnt(3)
	v_mov_b32_e32 v11, v53
	v_mov_b32_e32 v10, v53
	s_waitcnt lgkmcnt(2)
	v_mov_b32_e32 v9, v53
	v_mov_b32_e32 v8, v53
	v_mov_b32_e32 v3, v53
	v_mov_b32_e32 v2, v53
	v_mov_b32_e32 v1, v53
	v_mov_b32_e32 v0, v53
	s_waitcnt lgkmcnt(1)
	v_mov_b32_e32 v7, v53
	v_mov_b32_e32 v6, v53
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v5, v53
	v_mov_b32_e32 v4, v53
	s_cbranch_vccnz .LBB0_74
; %bb.68:                               ;   in Loop: Header=BB0_67 Depth=1
	v_min_u32_e32 v0, s3, v50
	v_mad_u64_u32 v[54:55], s[8:9], v0, s10, 0
	v_min_u32_e32 v0, s3, v61
	v_mov_b32_e32 v4, 0
	v_mad_u64_u32 v[56:57], s[8:9], v0, s10, 0
	s_mov_b32 s28, 0
	v_mov_b32_e32 v5, v4
	v_mov_b32_e32 v6, v4
	v_mov_b32_e32 v7, v4
	v_mov_b32_e32 v0, v4
	v_mov_b32_e32 v1, v4
	v_mov_b32_e32 v2, v4
	v_mov_b32_e32 v3, v4
	v_mov_b32_e32 v8, v4
	v_mov_b32_e32 v9, v4
	v_mov_b32_e32 v10, v4
	v_mov_b32_e32 v11, v4
	v_mov_b32_e32 v12, v4
	v_mov_b32_e32 v13, v4
	v_mov_b32_e32 v14, v4
	v_mov_b32_e32 v15, v4
	s_branch .LBB0_71
.LBB0_69:                               ;   in Loop: Header=BB0_71 Depth=2
	s_or_b64 exec, exec, s[16:17]
.LBB0_70:                               ;   in Loop: Header=BB0_71 Depth=2
	s_or_b64 exec, exec, s[8:9]
	s_waitcnt vmcnt(3) lgkmcnt(1)
	v_mfma_f32_16x16x32_fp8_fp8 v[12:15], v[40:41], v[24:25], v[12:15]
	s_addk_i32 s28, 0x800
	s_cmp_ge_u32 s28, s2
	s_waitcnt vmcnt(2)
	v_mfma_f32_16x16x32_fp8_fp8 v[8:11], v[40:41], v[28:29], v[8:11]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[44:45], v[24:25], v[0:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[4:7], v[44:45], v[28:29], v[4:7]
	v_mfma_f32_16x16x32_fp8_fp8 v[12:15], v[42:43], v[26:27], v[12:15]
	v_mfma_f32_16x16x32_fp8_fp8 v[8:11], v[42:43], v[30:31], v[8:11]
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[46:47], v[26:27], v[0:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[4:7], v[46:47], v[30:31], v[4:7]
	s_waitcnt vmcnt(1)
	v_mfma_f32_16x16x32_fp8_fp8 v[12:15], v[36:37], v[20:21], v[12:15]
	s_waitcnt vmcnt(0)
	v_mfma_f32_16x16x32_fp8_fp8 v[8:11], v[36:37], v[16:17], v[8:11]
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[32:33], v[20:21], v[0:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[4:7], v[32:33], v[16:17], v[4:7]
	v_mfma_f32_16x16x32_fp8_fp8 v[12:15], v[38:39], v[22:23], v[12:15]
	v_mfma_f32_16x16x32_fp8_fp8 v[8:11], v[38:39], v[18:19], v[8:11]
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[34:35], v[22:23], v[0:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[4:7], v[34:35], v[18:19], v[4:7]
	s_cbranch_scc1 .LBB0_74
.LBB0_71:                               ;   Parent Loop BB0_67 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_add_u32_e32 v51, s28, v48
	v_min_u32_e32 v52, s20, v51
	v_lshl_add_u64 v[16:17], s[4:5], 0, v[52:53]
	v_add_u32_e32 v62, 0x400, v51
	v_lshl_add_u64 v[18:19], v[16:17], 0, v[54:55]
	v_lshl_add_u64 v[16:17], v[16:17], 0, v[56:57]
	v_min_u32_e32 v52, s20, v62
	global_load_dwordx4 v[24:27], v[18:19], off nt
	global_load_dwordx4 v[28:31], v[16:17], off nt
	v_lshl_add_u64 v[16:17], s[4:5], 0, v[52:53]
	v_lshl_add_u64 v[32:33], v[16:17], 0, v[54:55]
	v_lshl_add_u64 v[34:35], v[16:17], 0, v[56:57]
	global_load_dwordx4 v[20:23], v[32:33], off nt
	global_load_dwordx4 v[16:19], v[34:35], off nt
	v_cmp_gt_u32_e32 vcc, s2, v51
	v_mov_b64_e32 v[40:41], 0
	v_mov_b64_e32 v[42:43], 0
	v_mov_b64_e32 v[36:37], 0
	v_mov_b64_e32 v[38:39], 0
	v_mov_b64_e32 v[44:45], 0
	v_mov_b64_e32 v[46:47], 0
	v_mov_b64_e32 v[32:33], 0
	v_mov_b64_e32 v[34:35], 0
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_70
; %bb.72:                               ;   in Loop: Header=BB0_71 Depth=2
	v_add_u32_e32 v52, s28, v60
	ds_read_b128 v[40:43], v51
	ds_read_b128 v[44:47], v52
	v_cmp_gt_u32_e32 vcc, s2, v62
	v_mov_b64_e32 v[34:35], 0
	v_mov_b64_e32 v[32:33], 0
	v_mov_b64_e32 v[38:39], 0
	v_mov_b64_e32 v[36:37], 0
	s_and_saveexec_b64 s[16:17], vcc
	s_cbranch_execz .LBB0_69
; %bb.73:                               ;   in Loop: Header=BB0_71 Depth=2
	ds_read_b128 v[36:39], v51 offset:1024
	ds_read_b128 v[32:35], v52 offset:1024
	s_branch .LBB0_69
.LBB0_74:                               ;   in Loop: Header=BB0_67 Depth=1
	s_nop 5
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
	ds_bpermute_b32 v13, v58, v12
	ds_bpermute_b32 v9, v58, v8
	ds_bpermute_b32 v1, v58, v0
	ds_bpermute_b32 v3, v58, v2
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v10, v12, v13
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v8, v8, v9
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v6, v0, v1
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v4, v2, v3
	ds_bpermute_b32 v11, v59, v10
	ds_bpermute_b32 v9, v59, v8
	ds_bpermute_b32 v7, v59, v6
	ds_bpermute_b32 v5, v59, v4
	s_and_saveexec_b64 s[28:29], s[0:1]
	s_cbranch_execz .LBB0_66
; %bb.75:                               ;   in Loop: Header=BB0_67 Depth=1
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v2, v10, v11
	v_mov_b32_e32 v51, v53
	v_cndmask_b32_e64 v3, 0, 1, s[24:25]
	v_lshl_add_u64 v[0:1], v[50:51], 1, s[12:13]
	v_cmp_ne_u32_e64 s[8:9], 1, v3
	s_andn2_b64 vcc, exec, s[24:25]
	v_mul_f32_e32 v10, v49, v2
	s_cbranch_vccnz .LBB0_77
; %bb.76:                               ;   in Loop: Header=BB0_67 Depth=1
	global_load_ushort v2, v[0:1], off
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v2, 16, v2
	v_fmac_f32_e32 v10, s21, v2
.LBB0_77:                               ;   in Loop: Header=BB0_67 Depth=1
	v_lshl_add_u64 v[2:3], v[50:51], 1, s[14:15]
	v_cvt_pk_bf16_f32 v10, v10, s0
	v_cmp_gt_u32_e64 s[16:17], s11, v61
	global_store_short v[2:3], v10, off
	s_and_saveexec_b64 s[30:31], s[16:17]
	s_cbranch_execz .LBB0_81
; %bb.78:                               ;   in Loop: Header=BB0_67 Depth=1
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v8, v8, v9
	s_and_b64 vcc, exec, s[8:9]
	v_mul_f32_e32 v8, v49, v8
	s_cbranch_vccnz .LBB0_80
; %bb.79:                               ;   in Loop: Header=BB0_67 Depth=1
	global_load_ushort v9, v[0:1], off offset:2
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v9, 16, v9
	v_fmac_f32_e32 v8, s21, v9
.LBB0_80:                               ;   in Loop: Header=BB0_67 Depth=1
	v_cvt_pk_bf16_f32 v8, v8, s0
	global_store_short v[2:3], v8, off offset:2
.LBB0_81:                               ;   in Loop: Header=BB0_67 Depth=1
	s_or_b64 exec, exec, s[30:31]
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v6, v6, v7
	v_lshl_add_u64 v[0:1], s[18:19], 1, v[0:1]
	s_and_b64 vcc, exec, s[8:9]
	v_mul_f32_e32 v6, v49, v6
	s_cbranch_vccnz .LBB0_83
; %bb.82:                               ;   in Loop: Header=BB0_67 Depth=1
	global_load_ushort v7, v[0:1], off
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v7, 16, v7
	v_fmac_f32_e32 v6, s21, v7
.LBB0_83:                               ;   in Loop: Header=BB0_67 Depth=1
	v_lshl_add_u64 v[2:3], s[22:23], 1, v[2:3]
	v_cvt_pk_bf16_f32 v6, v6, s0
	global_store_short v[2:3], v6, off
	s_and_b64 exec, exec, s[16:17]
	s_cbranch_execz .LBB0_66
; %bb.84:                               ;   in Loop: Header=BB0_67 Depth=1
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v4, v4, v5
	s_and_b64 vcc, exec, s[8:9]
	v_mul_f32_e32 v4, v49, v4
	s_cbranch_vccnz .LBB0_65
; %bb.85:                               ;   in Loop: Header=BB0_67 Depth=1
	global_load_ushort v0, v[0:1], off offset:2
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v0, 16, v0
	v_fmac_f32_e32 v4, s21, v0
	s_branch .LBB0_65
.LBB0_86:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wvSplitKQ_vllm_f8_bf16_tn_n2_y2u2
		.amdhsa_group_segment_fixed_size 163840
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 88
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
		.amdhsa_accum_offset 68
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
	.size	wvSplitKQ_vllm_f8_bf16_tn_n2_y2u2, .Lfunc_end0-wvSplitKQ_vllm_f8_bf16_tn_n2_y2u2
                                        ; -- End function
	.set wvSplitKQ_vllm_f8_bf16_tn_n2_y2u2.num_vgpr, 68
	.set wvSplitKQ_vllm_f8_bf16_tn_n2_y2u2.num_agpr, 0
	.set wvSplitKQ_vllm_f8_bf16_tn_n2_y2u2.numbered_sgpr, 44
	.set wvSplitKQ_vllm_f8_bf16_tn_n2_y2u2.num_named_barrier, 0
	.set wvSplitKQ_vllm_f8_bf16_tn_n2_y2u2.private_seg_size, 0
	.set wvSplitKQ_vllm_f8_bf16_tn_n2_y2u2.uses_vcc, 1
	.set wvSplitKQ_vllm_f8_bf16_tn_n2_y2u2.uses_flat_scratch, 0
	.set wvSplitKQ_vllm_f8_bf16_tn_n2_y2u2.has_dyn_sized_stack, 0
	.set wvSplitKQ_vllm_f8_bf16_tn_n2_y2u2.has_recursion, 0
	.set wvSplitKQ_vllm_f8_bf16_tn_n2_y2u2.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 3828
; TotalNumSgprs: 50
; NumVgprs: 68
; NumAgprs: 0
; TotalNumVgprs: 68
; ScratchSize: 0
; MemoryBound: 1
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 163840 bytes/workgroup (compile time only)
; SGPRBlocks: 12
; VGPRBlocks: 12
; NumSGPRsForWavesPerEU: 102
; NumVGPRsForWavesPerEU: 97
; AccumOffset: 68
; Occupancy: 4
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 1
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 16
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_f4a48629165cf143,@object ; @__hip_cuid_f4a48629165cf143
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_f4a48629165cf143
__hip_cuid_f4a48629165cf143:
	.byte	0                               ; 0x0
	.size	__hip_cuid_f4a48629165cf143, 1

	.ident	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.4 26084 f58b06dce1f9c15707c5f808fd002e18c2accf7e)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_f4a48629165cf143
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
    DataType: F8
    DestDataType: b
    ComputeDataType: s
    HighPrecisionAccumulate: True
    TransposeA: True
    TransposeB: False
    UseBeta: True
    Batched: True
    UseScaleAB: Scalar
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
            { type: int32, semantic: StrideD0 },
            { type: address, semantic: AddressScaleA },
            { type: address, semantic: AddressScaleB } ]
    macrotile: [64, 16, 16]
    threads: [64, 16, 1]
    grid: [ComputeUnits, One, One]
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 32, 1]
  EnableMatrixInstruction: True
  MIWaveTile: [1, 1]
  AssertSummationElementMultiple: 16
  AssertSizeEqual: { 1: 2, 2: 1 }
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
      - .actual_access:  read_only
        .address_space:  global
        .offset:         80
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 163840
    .kernarg_segment_align: 8
    .kernarg_segment_size: 88
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           wvSplitKQ_vllm_f8_bf16_tn_n2_y2u2
    .private_segment_fixed_size: 0
    .sgpr_count:     50
    .sgpr_spill_count: 0
    .symbol:         wvSplitKQ_vllm_f8_bf16_tn_n2_y2u2.kd
    .vgpr_count:     68
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 1
...

	.end_amdgpu_metadata
