	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 4
	.text
	.protected	wvSplitK_vllm_bf16_tn_bias_n5_y1u2 ; -- Begin function wvSplitK_vllm_bf16_tn_bias_n5_y1u2
	.globl	wvSplitK_vllm_bf16_tn_bias_n5_y1u2
	.p2align	8
	.type	wvSplitK_vllm_bf16_tn_bias_n5_y1u2,@function
wvSplitK_vllm_bf16_tn_bias_n5_y1u2:     ; @wvSplitK_vllm_bf16_tn_bias_n5_y1u2
; %bb.86:
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.87:
.LBB0_0:
	s_load_dwordx4 s[24:27], s[0:1], 0x38
	s_load_dwordx2 s[28:29], s[0:1], 0x48
	s_mov_b32 s30, s11
	s_mov_b64 s[0:1], -1
	s_cmpk_gt_i32 s2, 0x4000
	v_bfe_u32 v76, v0, 10, 10
	s_cbranch_scc0 .LBB0_48
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
	s_max_i32 s20, s0, 4
	v_cmp_gt_u32_e32 vcc, s20, v76
	s_and_saveexec_b64 s[34:35], vcc
	s_cbranch_execz .LBB0_47
; %bb.2:
	v_cvt_f32_u32_e32 v1, s20
	s_mul_i32 s0, s20, s16
	s_sub_i32 s11, 0, s20
	v_add_u32_e32 v70, s0, v76
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
	v_cmp_gt_u32_e32 vcc, s9, v70
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_47
; %bb.3:
	s_and_b32 s0, s30, 7
	s_cmp_lg_u32 s0, 0
	s_mov_b32 s37, 0
	s_cselect_b64 s[0:1], -1, 0
	s_and_b32 s36, s6, 15
	s_cmp_lg_u64 s[36:37], 0
	s_cselect_b64 s[18:19], -1, 0
	s_or_b64 s[40:41], s[18:19], s[0:1]
	s_waitcnt lgkmcnt(0)
	s_cmp_eq_u64 s[28:29], 0
	v_and_b32_e32 v2, 0x3ff, v0
	s_cselect_b64 s[0:1], -1, 0
	v_lshlrev_b32_e32 v1, 3, v2
	s_ashr_i32 s31, s30, 31
	v_cmp_eq_u32_e64 s[18:19], 63, v2
	s_ashr_i32 s45, s26, 31
	s_mov_b32 s44, s26
	s_ashr_i32 s47, s27, 31
	s_mov_b32 s46, s27
	v_lshlrev_b32_e32 v2, 4, v2
	s_mov_b64 s[38:39], 0
	v_lshl_add_u32 v77, v76, 9, v1
	s_lshl_b32 s11, s20, 9
	s_add_i32 s17, s2, -8
	s_ashr_i32 s33, s10, 31
	v_cmp_neq_f32_e64 s[42:43], s25, 0
	s_mul_i32 s66, s20, s8
	s_lshl_b64 s[48:49], s[30:31], 1
	s_lshl_b64 s[50:51], s[44:45], 1
	s_lshl_b64 s[52:53], s[46:47], 1
	v_lshl_add_u32 v78, v76, 10, v2
	s_lshl_b32 s67, s20, 10
	s_xor_b64 s[54:55], s[0:1], -1
	v_mov_b32_e32 v73, 0
	v_cndmask_b32_e64 v79, 0, 1, s[40:41]
                                        ; implicit-def: $vgpr80
	s_branch .LBB0_5
.LBB0_4:                                ;   in Loop: Header=BB0_5 Depth=1
	s_or_b64 exec, exec, s[22:23]
	v_add_u32_e32 v70, s66, v70
	v_cmp_le_u32_e32 vcc, s9, v70
	s_or_b64 s[38:39], vcc, s[38:39]
	s_andn2_b64 exec, exec, s[38:39]
	s_cbranch_execz .LBB0_47
.LBB0_5:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_11 Depth 2
                                        ;       Child Loop BB0_16 Depth 3
	v_cmp_gt_u32_e64 s[20:21], s3, v70
	s_and_b64 s[56:57], s[54:55], s[20:21]
	v_mov_b32_e32 v71, v73
	s_and_saveexec_b64 s[22:23], s[56:57]
	s_cbranch_execz .LBB0_7
; %bb.6:                                ;   in Loop: Header=BB0_5 Depth=1
	v_lshl_add_u64 v[2:3], v[70:71], 1, s[28:29]
	global_load_ushort v80, v[2:3], off
.LBB0_7:                                ;   in Loop: Header=BB0_5 Depth=1
	s_or_b64 exec, exec, s[22:23]
	v_mad_u64_u32 v[2:3], s[22:23], v70, s10, 0
	v_mov_b32_e32 v4, v3
	v_mad_u64_u32 v[4:5], s[22:23], v70, s33, v[4:5]
	v_mov_b32_e32 v3, v4
	v_mov_b32_e32 v18, 0
	s_mov_b32 s36, 0
	v_lshl_add_u64 v[74:75], v[2:3], 1, s[4:5]
	s_mov_b32 s68, s36
	v_mov_b32_e32 v19, v18
	v_mov_b32_e32 v20, v18
	v_mov_b32_e32 v21, v18
	v_mov_b32_e32 v14, v18
	v_mov_b32_e32 v15, v18
	v_mov_b32_e32 v16, v18
	v_mov_b32_e32 v17, v18
	v_mov_b32_e32 v10, v18
	v_mov_b32_e32 v11, v18
	v_mov_b32_e32 v12, v18
	v_mov_b32_e32 v13, v18
	v_mov_b32_e32 v6, v18
	v_mov_b32_e32 v7, v18
	v_mov_b32_e32 v8, v18
	v_mov_b32_e32 v9, v18
	v_mov_b32_e32 v2, v18
	v_mov_b32_e32 v3, v18
	v_mov_b32_e32 v4, v18
	v_mov_b32_e32 v5, v18
	s_branch .LBB0_11
.LBB0_8:                                ;   in Loop: Header=BB0_11 Depth=2
	s_or_b64 exec, exec, s[58:59]
.LBB0_9:                                ;   in Loop: Header=BB0_11 Depth=2
	s_or_b64 exec, exec, s[56:57]
	s_waitcnt vmcnt(1) lgkmcnt(4)
	v_mfma_f32_4x4x4_16b_bf16 v[18:21], v[62:63], v[26:27], v[18:21]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_4x4x4_16b_bf16 v[14:17], v[54:55], v[26:27], v[14:17]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_4x4x4_16b_bf16 v[10:13], v[38:39], v[26:27], v[10:13]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_4x4x4_16b_bf16 v[6:9], v[34:35], v[26:27], v[6:9]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_4x4x4_16b_bf16 v[2:5], v[30:31], v[26:27], v[2:5]
	v_mfma_f32_4x4x4_16b_bf16 v[18:21], v[64:65], v[28:29], v[18:21]
	v_mfma_f32_4x4x4_16b_bf16 v[14:17], v[56:57], v[28:29], v[14:17]
	v_mfma_f32_4x4x4_16b_bf16 v[10:13], v[40:41], v[28:29], v[10:13]
	v_mfma_f32_4x4x4_16b_bf16 v[6:9], v[36:37], v[28:29], v[6:9]
	v_mfma_f32_4x4x4_16b_bf16 v[2:5], v[32:33], v[28:29], v[2:5]
	s_waitcnt vmcnt(0)
	v_mfma_f32_4x4x4_16b_bf16 v[18:21], v[46:47], v[22:23], v[18:21]
	v_mfma_f32_4x4x4_16b_bf16 v[14:17], v[42:43], v[22:23], v[14:17]
	v_mfma_f32_4x4x4_16b_bf16 v[10:13], v[58:59], v[22:23], v[10:13]
	v_mfma_f32_4x4x4_16b_bf16 v[6:9], v[50:51], v[22:23], v[6:9]
	v_mfma_f32_4x4x4_16b_bf16 v[2:5], v[66:67], v[22:23], v[2:5]
	v_mfma_f32_4x4x4_16b_bf16 v[18:21], v[48:49], v[24:25], v[18:21]
	v_mfma_f32_4x4x4_16b_bf16 v[14:17], v[44:45], v[24:25], v[14:17]
	v_mfma_f32_4x4x4_16b_bf16 v[10:13], v[60:61], v[24:25], v[10:13]
	v_mfma_f32_4x4x4_16b_bf16 v[6:9], v[52:53], v[24:25], v[6:9]
	v_mfma_f32_4x4x4_16b_bf16 v[2:5], v[68:69], v[24:25], v[2:5]
.LBB0_10:                               ;   in Loop: Header=BB0_11 Depth=2
	s_or_b64 exec, exec, s[22:23]
	s_addk_i32 s68, 0x400
	s_cmp_ge_u32 s68, s2
	s_cbranch_scc1 .LBB0_41
.LBB0_11:                               ;   Parent Loop BB0_5 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_16 Depth 3
	s_cmp_eq_u32 s68, 0
	s_cselect_b64 s[22:23], -1, 0
	s_add_i32 s56, s36, 0x4000
	s_cmp_eq_u32 s68, s56
	s_cselect_b64 s[58:59], -1, 0
	s_or_b64 s[58:59], s[22:23], s[58:59]
	s_andn2_b64 vcc, exec, s[58:59]
	s_cbranch_vccz .LBB0_13
; %bb.12:                               ;   in Loop: Header=BB0_11 Depth=2
	s_and_saveexec_b64 s[22:23], s[20:21]
	s_cbranch_execz .LBB0_10
	s_branch .LBB0_38
.LBB0_13:                               ;   in Loop: Header=BB0_11 Depth=2
	s_and_b64 s[22:23], s[22:23], exec
	s_cselect_b32 s36, s36, s56
	s_sub_i32 s22, s2, s36
	s_min_u32 s69, s22, 0x4000
	s_lshl_b64 s[22:23], s[36:37], 1
	s_add_u32 s56, s6, s22
	s_addc_u32 s57, s7, s23
	s_mov_b32 s70, 0
	s_mov_b64 s[58:59], 0
	v_mov_b32_e32 v24, v78
	s_barrier
                                        ; implicit-def: $sgpr60_sgpr61
	s_branch .LBB0_16
.LBB0_14:                               ;   in Loop: Header=BB0_16 Depth=3
	s_add_i32 s70, s70, s11
	s_cmpk_gt_u32 s70, 0x3fff
	s_cselect_b64 s[22:23], -1, 0
	s_andn2_b64 s[60:61], s[60:61], exec
	s_and_b64 s[22:23], s[22:23], exec
	v_add_u32_e32 v24, s67, v24
	s_or_b64 s[60:61], s[60:61], s[22:23]
.LBB0_15:                               ;   in Loop: Header=BB0_16 Depth=3
	s_or_b64 exec, exec, s[62:63]
	s_and_b64 s[22:23], exec, s[60:61]
	s_or_b64 s[58:59], s[22:23], s[58:59]
	s_andn2_b64 exec, exec, s[58:59]
	s_cbranch_execz .LBB0_37
.LBB0_16:                               ;   Parent Loop BB0_5 Depth=1
                                        ;     Parent Loop BB0_11 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	v_add_u32_e32 v72, s70, v77
	v_cmp_gt_u32_e32 vcc, s69, v72
	s_or_b64 s[60:61], s[60:61], exec
	s_and_saveexec_b64 s[62:63], vcc
	s_cbranch_execz .LBB0_15
; %bb.17:                               ;   in Loop: Header=BB0_16 Depth=3
	v_lshl_add_u64 v[22:23], v[72:73], 1, s[56:57]
	s_mov_b64 s[22:23], -1
	s_and_b64 vcc, exec, s[40:41]
	s_cbranch_vccz .LBB0_19
; %bb.18:                               ;   in Loop: Header=BB0_16 Depth=3
	global_load_dwordx4 v[26:29], v[22:23], off
	s_mov_b64 s[22:23], 0
	s_waitcnt vmcnt(0)
	ds_write_b128 v24, v[26:29]
.LBB0_19:                               ;   in Loop: Header=BB0_16 Depth=3
	s_andn2_b64 vcc, exec, s[22:23]
	s_cbranch_vccnz .LBB0_21
; %bb.20:                               ;   in Loop: Header=BB0_16 Depth=3
	v_readfirstlane_b32 s22, v24
	s_mov_b32 m0, s22
	s_nop 0
	global_load_lds_dwordx4 v[22:23], off
.LBB0_21:                               ;   in Loop: Header=BB0_16 Depth=3
	v_lshl_add_u64 v[22:23], s[30:31], 1, v[22:23]
	v_add_u32_e32 v25, 0x8000, v24
	v_cmp_ne_u32_e64 s[22:23], 1, v79
	s_andn2_b64 vcc, exec, s[40:41]
	s_mov_b64 s[64:65], -1
	s_cbranch_vccnz .LBB0_23
; %bb.22:                               ;   in Loop: Header=BB0_16 Depth=3
	global_load_dwordx4 v[26:29], v[22:23], off
	s_mov_b64 s[64:65], 0
	s_waitcnt vmcnt(0)
	ds_write_b128 v25, v[26:29]
.LBB0_23:                               ;   in Loop: Header=BB0_16 Depth=3
	s_andn2_b64 vcc, exec, s[64:65]
	s_cbranch_vccnz .LBB0_25
; %bb.24:                               ;   in Loop: Header=BB0_16 Depth=3
	v_readfirstlane_b32 s64, v25
	s_mov_b32 m0, s64
	s_nop 0
	global_load_lds_dwordx4 v[22:23], off
.LBB0_25:                               ;   in Loop: Header=BB0_16 Depth=3
	v_lshl_add_u64 v[22:23], v[22:23], 0, s[48:49]
	v_add_u32_e32 v25, 0x10000, v24
	s_and_b64 vcc, exec, s[22:23]
	s_mov_b64 s[64:65], -1
	s_cbranch_vccnz .LBB0_27
; %bb.26:                               ;   in Loop: Header=BB0_16 Depth=3
	global_load_dwordx4 v[26:29], v[22:23], off
	s_mov_b64 s[64:65], 0
	s_waitcnt vmcnt(0)
	ds_write_b128 v25, v[26:29]
.LBB0_27:                               ;   in Loop: Header=BB0_16 Depth=3
	s_andn2_b64 vcc, exec, s[64:65]
	s_cbranch_vccnz .LBB0_29
; %bb.28:                               ;   in Loop: Header=BB0_16 Depth=3
	v_readfirstlane_b32 s64, v25
	s_mov_b32 m0, s64
	s_nop 0
	global_load_lds_dwordx4 v[22:23], off
.LBB0_29:                               ;   in Loop: Header=BB0_16 Depth=3
	v_lshl_add_u64 v[22:23], v[22:23], 0, s[48:49]
	v_add_u32_e32 v25, 0x18000, v24
	s_and_b64 vcc, exec, s[22:23]
	s_mov_b64 s[64:65], -1
	s_cbranch_vccnz .LBB0_31
; %bb.30:                               ;   in Loop: Header=BB0_16 Depth=3
	global_load_dwordx4 v[26:29], v[22:23], off
	s_mov_b64 s[64:65], 0
	s_waitcnt vmcnt(0)
	ds_write_b128 v25, v[26:29]
.LBB0_31:                               ;   in Loop: Header=BB0_16 Depth=3
	s_andn2_b64 vcc, exec, s[64:65]
	s_cbranch_vccnz .LBB0_33
; %bb.32:                               ;   in Loop: Header=BB0_16 Depth=3
	v_readfirstlane_b32 s64, v25
	s_mov_b32 m0, s64
	s_nop 0
	global_load_lds_dwordx4 v[22:23], off
.LBB0_33:                               ;   in Loop: Header=BB0_16 Depth=3
	v_lshl_add_u64 v[22:23], v[22:23], 0, s[48:49]
	v_add_u32_e32 v25, 0x20000, v24
	s_and_b64 vcc, exec, s[22:23]
	s_mov_b64 s[22:23], -1
	s_cbranch_vccnz .LBB0_35
; %bb.34:                               ;   in Loop: Header=BB0_16 Depth=3
	global_load_dwordx4 v[26:29], v[22:23], off
	s_mov_b64 s[22:23], 0
	s_waitcnt vmcnt(0)
	ds_write_b128 v25, v[26:29]
.LBB0_35:                               ;   in Loop: Header=BB0_16 Depth=3
	s_andn2_b64 vcc, exec, s[22:23]
	s_cbranch_vccnz .LBB0_14
; %bb.36:                               ;   in Loop: Header=BB0_16 Depth=3
	v_readfirstlane_b32 s22, v25
	s_mov_b32 m0, s22
	s_nop 0
	global_load_lds_dwordx4 v[22:23], off
	s_branch .LBB0_14
.LBB0_37:                               ;   in Loop: Header=BB0_11 Depth=2
	s_or_b64 exec, exec, s[58:59]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	s_and_saveexec_b64 s[22:23], s[20:21]
	s_cbranch_execz .LBB0_10
.LBB0_38:                               ;   in Loop: Header=BB0_11 Depth=2
	v_add_u32_e32 v82, s68, v1
	v_min_u32_e32 v72, s17, v82
	v_add_u32_e32 v81, 0x200, v82
	v_lshl_add_u64 v[22:23], v[72:73], 1, v[74:75]
	v_min_u32_e32 v72, s17, v81
	v_lshl_add_u64 v[24:25], v[72:73], 1, v[74:75]
	global_load_dwordx4 v[26:29], v[22:23], off nt
	s_nop 0
	global_load_dwordx4 v[22:25], v[24:25], off nt
	v_cmp_gt_u32_e32 vcc, s2, v82
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v58, 0
	v_mov_b32_e32 v59, 0
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v39, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v56, 0
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v63, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	s_and_saveexec_b64 s[56:57], vcc
	s_cbranch_execz .LBB0_9
; %bb.39:                               ;   in Loop: Header=BB0_11 Depth=2
	v_subrev_u32_e32 v30, s36, v82
	v_lshlrev_b32_e32 v72, 1, v30
	v_add_u32_e32 v30, 0x10000, v72
	ds_read_b128 v[62:65], v72
	ds_read_b128 v[54:57], v72 offset:32768
	v_add_u32_e32 v31, 0x18000, v72
	ds_read_b128 v[38:41], v30
	ds_read_b128 v[34:37], v31
	v_add_u32_e32 v30, 0x20000, v72
	ds_read_b128 v[30:33], v30
	v_cmp_gt_u32_e32 vcc, s2, v81
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v59, 0
	v_mov_b32_e32 v58, 0
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v52, 0
	s_and_saveexec_b64 s[58:59], vcc
	s_cbranch_execz .LBB0_8
; %bb.40:                               ;   in Loop: Header=BB0_11 Depth=2
	v_add_u32_e32 v50, 0x10400, v72
	v_add_u32_e32 v51, 0x18400, v72
	v_add_u32_e32 v66, 0x20400, v72
	ds_read_b128 v[46:49], v72 offset:1024
	ds_read_b128 v[42:45], v72 offset:33792
	ds_read_b128 v[58:61], v50
	ds_read_b128 v[50:53], v51
	ds_read_b128 v[66:69], v66
	s_branch .LBB0_8
.LBB0_41:                               ;   in Loop: Header=BB0_5 Depth=1
	s_and_saveexec_b64 s[22:23], s[20:21]
	s_cbranch_execz .LBB0_4
; %bb.42:                               ;   in Loop: Header=BB0_5 Depth=1
	v_add_f32_dpp v18, v19, v18 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v14, v15, v14 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v10, v11, v10 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v7, v6 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v3, v2 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v18, v20, v18 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v14, v16, v14 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v10, v12, v10 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v8, v6 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v4, v2 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v18, v21, v18 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v14, v17, v14 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v10, v13, v10 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v9, v6 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v5, v2 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v18, v18, v18 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v14, v14, v14 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v10, v10, v10 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v6, v6 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v18, v18, v18 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v14, v14, v14 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v10, v10, v10 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v6, v6 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v18, v18 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v14, v14 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v10, v10 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v6, v6 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v2, v2 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v18, v18, v18 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v14, v14, v14 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v10, v10, v10 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v6, v6 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v19, v18 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v15, v14 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v11, v10 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v9, v8 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v3, v2 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_and_b64 exec, exec, s[18:19]
	s_cbranch_execz .LBB0_4
; %bb.43:                               ;   in Loop: Header=BB0_5 Depth=1
	v_add_f32_e32 v7, v18, v19
	v_add_f32_e32 v6, v14, v15
	v_add_f32_e32 v5, v10, v11
	v_add_f32_e32 v4, v8, v9
	v_add_f32_e32 v2, v2, v3
	s_andn2_b64 vcc, exec, s[42:43]
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v3, 16, v80
	s_cbranch_vccnz .LBB0_45
; %bb.44:                               ;   in Loop: Header=BB0_5 Depth=1
	v_lshlrev_b64 v[8:9], 1, v[70:71]
	v_lshl_add_u64 v[10:11], s[12:13], 0, v[8:9]
	global_load_ushort v12, v[10:11], off
	v_lshl_add_u64 v[8:9], s[14:15], 0, v[8:9]
	v_lshl_add_u64 v[10:11], s[44:45], 1, v[10:11]
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v12, 16, v12
	v_mul_f32_e32 v12, s25, v12
	v_fmac_f32_e32 v12, s24, v7
	v_add_f32_e32 v13, v12, v3
	v_cndmask_b32_e64 v12, v13, v12, s[0:1]
	v_cvt_pk_bf16_f32 v12, v12, s0
	global_store_short v[8:9], v12, off
	global_load_ushort v12, v[10:11], off
	v_lshl_add_u64 v[8:9], s[46:47], 1, v[8:9]
	v_lshl_add_u64 v[10:11], v[10:11], 0, s[50:51]
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v12, 16, v12
	v_mul_f32_e32 v12, s25, v12
	v_fmac_f32_e32 v12, s24, v6
	v_add_f32_e32 v13, v12, v3
	v_cndmask_b32_e64 v12, v13, v12, s[0:1]
	v_cvt_pk_bf16_f32 v12, v12, s0
	global_store_short v[8:9], v12, off
	global_load_ushort v12, v[10:11], off
	v_lshl_add_u64 v[8:9], v[8:9], 0, s[52:53]
	v_lshl_add_u64 v[10:11], v[10:11], 0, s[50:51]
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v12, 16, v12
	v_mul_f32_e32 v12, s25, v12
	v_fmac_f32_e32 v12, s24, v5
	v_add_f32_e32 v13, v12, v3
	v_cndmask_b32_e64 v12, v13, v12, s[0:1]
	v_cvt_pk_bf16_f32 v12, v12, s0
	global_store_short v[8:9], v12, off
	global_load_ushort v12, v[10:11], off
	v_lshl_add_u64 v[8:9], v[8:9], 0, s[52:53]
	v_lshl_add_u64 v[10:11], v[10:11], 0, s[50:51]
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v12, 16, v12
	v_mul_f32_e32 v12, s25, v12
	v_fmac_f32_e32 v12, s24, v4
	v_add_f32_e32 v13, v12, v3
	v_cndmask_b32_e64 v12, v13, v12, s[0:1]
	v_cvt_pk_bf16_f32 v12, v12, s0
	global_store_short v[8:9], v12, off
	global_load_ushort v10, v[10:11], off
	v_lshl_add_u64 v[8:9], v[8:9], 0, s[52:53]
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v10, 16, v10
	v_mul_f32_e32 v10, s25, v10
	v_fmac_f32_e32 v10, s24, v2
	v_add_f32_e32 v11, v10, v3
	v_cndmask_b32_e64 v10, v11, v10, s[0:1]
	v_cvt_pk_bf16_f32 v10, v10, s0
	global_store_short v[8:9], v10, off
	s_cbranch_execnz .LBB0_4
	s_branch .LBB0_46
.LBB0_45:                               ;   in Loop: Header=BB0_5 Depth=1
.LBB0_46:                               ;   in Loop: Header=BB0_5 Depth=1
	v_mul_f32_e32 v10, s24, v7
	v_fma_f32 v7, s24, v7, v3
	v_cndmask_b32_e64 v7, v7, v10, s[0:1]
	v_lshl_add_u64 v[8:9], v[70:71], 1, s[14:15]
	v_cvt_pk_bf16_f32 v7, v7, s0
	global_store_short v[8:9], v7, off
	v_mul_f32_e32 v7, s24, v6
	v_fma_f32 v6, s24, v6, v3
	v_cndmask_b32_e64 v6, v6, v7, s[0:1]
	v_lshl_add_u64 v[8:9], s[46:47], 1, v[8:9]
	v_cvt_pk_bf16_f32 v6, v6, s0
	global_store_short v[8:9], v6, off
	v_lshl_add_u64 v[6:7], v[8:9], 0, s[52:53]
	v_mul_f32_e32 v8, s24, v5
	v_fma_f32 v5, s24, v5, v3
	v_cndmask_b32_e64 v5, v5, v8, s[0:1]
	v_cvt_pk_bf16_f32 v5, v5, s0
	global_store_short v[6:7], v5, off
	v_mul_f32_e32 v5, s24, v4
	v_fma_f32 v4, s24, v4, v3
	v_cndmask_b32_e64 v4, v4, v5, s[0:1]
	v_lshl_add_u64 v[6:7], v[6:7], 0, s[52:53]
	v_cvt_pk_bf16_f32 v4, v4, s0
	global_store_short v[6:7], v4, off
	v_lshl_add_u64 v[4:5], v[6:7], 0, s[52:53]
	v_mul_f32_e32 v6, s24, v2
	v_fmac_f32_e32 v3, s24, v2
	v_cndmask_b32_e64 v2, v3, v6, s[0:1]
	v_cvt_pk_bf16_f32 v2, v2, s0
	global_store_short v[4:5], v2, off
	s_branch .LBB0_4
.LBB0_47:
	s_or_b64 exec, exec, s[34:35]
	s_mov_b64 s[0:1], 0
.LBB0_48:
	s_andn2_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB0_85
; %bb.49:
	v_and_b32_e32 v8, 0x3ff, v0
	s_mul_i32 s9, s2, 5
	v_lshlrev_b32_e32 v74, 3, v8
	s_cmp_eq_u32 s30, s2
	s_mov_b64 s[18:19], 0
	s_cbranch_scc1 .LBB0_51
; %bb.50:
	s_and_b32 s20, s6, 15
	s_mov_b32 s21, 0
	s_mov_b64 s[0:1], -1
	s_min_u32 s9, s9, 0x14000
	v_lshl_add_u32 v9, v76, 9, v74
	s_cbranch_execz .LBB0_52
	s_branch .LBB0_53
.LBB0_51:
                                        ; implicit-def: $sgpr20_sgpr21
	s_mov_b64 s[0:1], 0
	s_min_u32 s9, s9, 0x14000
	v_lshl_add_u32 v9, v76, 9, v74
.LBB0_52:
	s_and_b32 s0, s2, 7
	s_cmp_lg_u32 s0, 0
	s_mov_b32 s21, 0
	s_cselect_b64 s[0:1], -1, 0
	s_and_b32 s20, s6, 15
	s_cmp_lg_u64 s[20:21], 0
	s_cselect_b64 s[18:19], -1, 0
	s_or_b64 s[0:1], s[0:1], s[18:19]
	s_mov_b64 s[18:19], -1
.LBB0_53:
	s_andn2_b64 vcc, exec, s[0:1]
	v_cmp_gt_u32_e64 s[0:1], s9, v9
	s_cbranch_vccz .LBB0_55
; %bb.54:
	s_andn2_b64 vcc, exec, s[18:19]
	s_cbranch_vccz .LBB0_63
	s_branch .LBB0_67
.LBB0_55:
	s_and_saveexec_b64 s[18:19], s[0:1]
	s_cbranch_execz .LBB0_62
; %bb.56:
	v_cvt_f32_u32_e32 v0, s2
	s_and_b32 s11, s30, 7
	s_cmp_lg_u32 s11, 0
	s_cselect_b64 s[22:23], -1, 0
	v_rcp_iflag_f32_e32 v0, v0
	s_cmp_lg_u64 s[20:21], 0
	s_cselect_b64 s[20:21], -1, 0
	s_sub_i32 s17, 0, s2
	v_mul_f32_e32 v0, 0x4f7ffffe, v0
	v_cvt_u32_f32_e32 v0, v0
	s_or_b64 s[20:21], s[22:23], s[20:21]
	s_mov_b64 s[0:1], 0
	s_ashr_i32 s11, s30, 31
	v_mul_lo_u32 v1, s17, v0
	v_mul_hi_u32 v1, v0, v1
	v_add_u32_e32 v0, v0, v1
	v_mov_b32_e32 v1, 0
	v_mad_u64_u32 v[2:3], s[22:23], v9, v0, 0
	v_lshlrev_b64 v[4:5], 13, v[0:1]
	v_lshlrev_b32_e32 v0, 4, v8
	v_lshl_add_u32 v10, v76, 10, v0
	v_mov_b32_e32 v0, v9
	s_branch .LBB0_58
.LBB0_57:                               ;   in Loop: Header=BB0_58 Depth=1
	v_add_u32_e32 v0, 0x2000, v0
	v_cmp_le_u32_e32 vcc, s9, v0
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[4:5]
	s_or_b64 s[0:1], vcc, s[0:1]
	v_add_u32_e32 v10, 0x4000, v10
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB0_62
.LBB0_58:                               ; =>This Inner Loop Header: Depth=1
	v_not_b32_e32 v11, v3
	v_mad_u64_u32 v[6:7], s[22:23], s17, v3, v[0:1]
	v_add_u32_e32 v7, 1, v3
	v_cmp_le_u32_e32 vcc, s2, v6
	v_mad_u64_u32 v[12:13], s[22:23], s2, v11, v[0:1]
	s_nop 0
	v_cndmask_b32_e32 v7, v3, v7, vcc
	v_cndmask_b32_e32 v6, v6, v12, vcc
	v_add_u32_e32 v11, 1, v7
	v_cmp_le_u32_e32 vcc, s2, v6
	s_nop 1
	v_cndmask_b32_e32 v11, v7, v11, vcc
	v_mad_u64_u32 v[12:13], s[22:23], v11, s30, 0
	v_mov_b32_e32 v14, v13
	v_mad_u64_u32 v[14:15], s[22:23], v11, s11, v[14:15]
	v_mad_u64_u32 v[6:7], s[22:23], s17, v11, v[0:1]
	v_mov_b32_e32 v13, v14
	v_mov_b32_e32 v7, v1
	v_lshl_add_u64 v[12:13], v[12:13], 1, s[6:7]
	v_lshl_add_u64 v[6:7], v[6:7], 1, v[12:13]
	s_mov_b64 s[22:23], -1
	s_and_b64 vcc, exec, s[20:21]
	s_cbranch_vccz .LBB0_60
; %bb.59:                               ;   in Loop: Header=BB0_58 Depth=1
	global_load_dwordx4 v[12:15], v[6:7], off
	s_mov_b64 s[22:23], 0
	s_waitcnt vmcnt(0)
	ds_write_b128 v10, v[12:15]
.LBB0_60:                               ;   in Loop: Header=BB0_58 Depth=1
	s_andn2_b64 vcc, exec, s[22:23]
	s_cbranch_vccnz .LBB0_57
; %bb.61:                               ;   in Loop: Header=BB0_58 Depth=1
	v_readfirstlane_b32 s22, v10
	s_mov_b32 m0, s22
	s_nop 0
	global_load_lds_dwordx4 v[6:7], off
	s_branch .LBB0_57
.LBB0_62:
	s_or_b64 exec, exec, s[18:19]
	s_cbranch_execnz .LBB0_67
.LBB0_63:
	v_cmp_gt_u32_e32 vcc, s9, v9
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_66
; %bb.64:
	v_lshlrev_b32_e32 v0, 4, v8
	v_lshl_add_u32 v2, v76, 10, v0
	v_lshlrev_b32_e32 v0, 1, v9
	v_mov_b32_e32 v1, 0
	v_lshl_add_u64 v[0:1], s[6:7], 0, v[0:1]
	s_mov_b64 s[6:7], 0
	s_mov_b64 s[18:19], 0x4000
.LBB0_65:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s11, v2
	s_mov_b32 m0, s11
	v_add_u32_e32 v9, 0x2000, v9
	global_load_lds_dwordx4 v[0:1], off
	v_cmp_le_u32_e32 vcc, s9, v9
	v_add_u32_e32 v2, 0x4000, v2
	s_or_b64 s[6:7], vcc, s[6:7]
	v_lshl_add_u64 v[0:1], v[0:1], 0, s[18:19]
	s_andn2_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB0_65
.LBB0_66:
	s_or_b64 exec, exec, s[0:1]
.LBB0_67:
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
	v_add_u32_e32 v68, s0, v76
	v_cmp_gt_u32_e32 vcc, s11, v76
	v_cmp_gt_u32_e64 s[0:1], s3, v68
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_85
; %bb.68:
	s_cmp_eq_u64 s[28:29], 0
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lg_u64 s[28:29], 0
	s_cselect_b64 s[6:7], -1, 0
	s_cmp_lg_u32 s2, 0
	s_cselect_b64 s[36:37], -1, 0
	v_cndmask_b32_e64 v0, 0, 1, s[6:7]
	s_ashr_i32 s23, s26, 31
	s_mov_b32 s22, s26
	s_ashr_i32 s31, s27, 31
	s_mov_b32 s30, s27
	v_cmp_ne_u32_e64 s[6:7], 1, v0
	v_cndmask_b32_e64 v0, 0, 1, s[36:37]
	s_mov_b64 s[18:19], 0
	v_cmp_eq_u32_e64 s[16:17], 63, v8
	s_mul_i32 s11, s11, s8
	s_add_i32 s33, s2, -8
	s_add_i32 s40, s3, -1
	s_ashr_i32 s41, s10, 31
	v_cmp_neq_f32_e64 s[20:21], s25, 0
	s_lshl_b32 s42, s2, 1
	s_lshl_b64 s[26:27], s[22:23], 1
	s_lshl_b64 s[34:35], s[30:31], 1
	v_lshlrev_b32_e32 v75, 4, v8
	s_lshl_b32 s43, s2, 2
	s_mul_i32 s44, s2, 6
	s_lshl_b32 s45, s2, 3
	v_mov_b32_e32 v71, 0
	v_cmp_ne_u32_e64 s[8:9], 1, v0
                                        ; implicit-def: $vgpr76
	s_branch .LBB0_70
.LBB0_69:                               ;   in Loop: Header=BB0_70 Depth=1
	s_or_b64 exec, exec, s[36:37]
	v_add_u32_e32 v68, s11, v68
	v_cmp_le_u32_e32 vcc, s3, v68
	s_or_b64 s[18:19], vcc, s[18:19]
	s_andn2_b64 exec, exec, s[18:19]
	s_cbranch_execz .LBB0_85
.LBB0_70:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_76 Depth 2
	s_and_b64 vcc, exec, s[6:7]
	s_cbranch_vccnz .LBB0_72
; %bb.71:                               ;   in Loop: Header=BB0_70 Depth=1
	v_mov_b32_e32 v69, v71
	v_lshl_add_u64 v[0:1], v[68:69], 1, s[28:29]
	global_load_ushort v76, v[0:1], off
.LBB0_72:                               ;   in Loop: Header=BB0_70 Depth=1
	s_and_b64 vcc, exec, s[8:9]
	s_cbranch_vccnz .LBB0_79
; %bb.73:                               ;   in Loop: Header=BB0_70 Depth=1
	v_min_u32_e32 v3, s40, v68
	v_mad_u64_u32 v[0:1], s[36:37], v3, s10, 0
	v_mov_b32_e32 v2, v1
	v_mad_u64_u32 v[2:3], s[36:37], v3, s41, v[2:3]
	v_mov_b32_e32 v1, v2
	v_mov_b32_e32 v16, 0
	v_lshl_add_u64 v[72:73], v[0:1], 1, s[4:5]
	s_mov_b32 s46, 0
	v_mov_b32_e32 v69, v75
	v_mov_b32_e32 v17, v16
	v_mov_b32_e32 v18, v16
	v_mov_b32_e32 v19, v16
	v_mov_b32_e32 v12, v16
	v_mov_b32_e32 v13, v16
	v_mov_b32_e32 v14, v16
	v_mov_b32_e32 v15, v16
	v_mov_b32_e32 v8, v16
	v_mov_b32_e32 v9, v16
	v_mov_b32_e32 v10, v16
	v_mov_b32_e32 v11, v16
	v_mov_b32_e32 v0, v16
	v_mov_b32_e32 v1, v16
	v_mov_b32_e32 v2, v16
	v_mov_b32_e32 v3, v16
	v_mov_b32_e32 v4, v16
	v_mov_b32_e32 v5, v16
	v_mov_b32_e32 v6, v16
	v_mov_b32_e32 v7, v16
	s_branch .LBB0_76
.LBB0_74:                               ;   in Loop: Header=BB0_76 Depth=2
	s_or_b64 exec, exec, s[38:39]
.LBB0_75:                               ;   in Loop: Header=BB0_76 Depth=2
	s_or_b64 exec, exec, s[36:37]
	s_waitcnt vmcnt(1) lgkmcnt(4)
	v_mfma_f32_4x4x4_16b_bf16 v[16:19], v[64:65], v[24:25], v[16:19]
	s_addk_i32 s46, 0x400
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_4x4x4_16b_bf16 v[12:15], v[60:61], v[24:25], v[12:15]
	v_add_u32_e32 v69, 0x800, v69
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_4x4x4_16b_bf16 v[8:11], v[40:41], v[24:25], v[8:11]
	s_cmp_ge_u32 s46, s2
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_4x4x4_16b_bf16 v[0:3], v[32:33], v[24:25], v[0:3]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_4x4x4_16b_bf16 v[4:7], v[28:29], v[24:25], v[4:7]
	v_mfma_f32_4x4x4_16b_bf16 v[16:19], v[66:67], v[26:27], v[16:19]
	v_mfma_f32_4x4x4_16b_bf16 v[12:15], v[62:63], v[26:27], v[12:15]
	v_mfma_f32_4x4x4_16b_bf16 v[8:11], v[42:43], v[26:27], v[8:11]
	v_mfma_f32_4x4x4_16b_bf16 v[0:3], v[34:35], v[26:27], v[0:3]
	v_mfma_f32_4x4x4_16b_bf16 v[4:7], v[30:31], v[26:27], v[4:7]
	s_waitcnt vmcnt(0)
	v_mfma_f32_4x4x4_16b_bf16 v[16:19], v[56:57], v[20:21], v[16:19]
	v_mfma_f32_4x4x4_16b_bf16 v[12:15], v[52:53], v[20:21], v[12:15]
	v_mfma_f32_4x4x4_16b_bf16 v[8:11], v[48:49], v[20:21], v[8:11]
	v_mfma_f32_4x4x4_16b_bf16 v[0:3], v[44:45], v[20:21], v[0:3]
	v_mfma_f32_4x4x4_16b_bf16 v[4:7], v[36:37], v[20:21], v[4:7]
	v_mfma_f32_4x4x4_16b_bf16 v[16:19], v[58:59], v[22:23], v[16:19]
	v_mfma_f32_4x4x4_16b_bf16 v[12:15], v[54:55], v[22:23], v[12:15]
	v_mfma_f32_4x4x4_16b_bf16 v[8:11], v[50:51], v[22:23], v[8:11]
	v_mfma_f32_4x4x4_16b_bf16 v[0:3], v[46:47], v[22:23], v[0:3]
	v_mfma_f32_4x4x4_16b_bf16 v[4:7], v[38:39], v[22:23], v[4:7]
	s_cbranch_scc1 .LBB0_80
.LBB0_76:                               ;   Parent Loop BB0_70 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_add_u32_e32 v32, s46, v74
	v_min_u32_e32 v70, s33, v32
	v_add_u32_e32 v77, 0x200, v32
	v_lshl_add_u64 v[28:29], v[70:71], 1, v[72:73]
	v_min_u32_e32 v70, s33, v77
	v_lshl_add_u64 v[30:31], v[70:71], 1, v[72:73]
	global_load_dwordx4 v[24:27], v[28:29], off nt
	global_load_dwordx4 v[20:23], v[30:31], off nt
	v_cmp_gt_u32_e32 vcc, s2, v32
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v63, 0
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v58, 0
	v_mov_b32_e32 v59, 0
	v_mov_b32_e32 v56, 0
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v39, 0
	s_and_saveexec_b64 s[36:37], vcc
	s_cbranch_execz .LBB0_75
; %bb.77:                               ;   in Loop: Header=BB0_76 Depth=2
	v_add_u32_e32 v80, s42, v69
	v_add_u32_e32 v78, s44, v69
	ds_read_b128 v[64:67], v69
	v_add_u32_e32 v79, s43, v69
	ds_read_b128 v[60:63], v80
	ds_read_b128 v[40:43], v79
	v_add_u32_e32 v70, s45, v69
	ds_read_b128 v[32:35], v78
	ds_read_b128 v[28:31], v70
	v_cmp_gt_u32_e32 vcc, s2, v77
	v_mov_b32_e32 v39, 0
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v56, 0
	v_mov_b32_e32 v59, 0
	v_mov_b32_e32 v58, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v46, 0
	s_and_saveexec_b64 s[38:39], vcc
	s_cbranch_execz .LBB0_74
; %bb.78:                               ;   in Loop: Header=BB0_76 Depth=2
	ds_read_b128 v[56:59], v69 offset:1024
	ds_read_b128 v[52:55], v80 offset:1024
	ds_read_b128 v[48:51], v79 offset:1024
	ds_read_b128 v[44:47], v78 offset:1024
	ds_read_b128 v[36:39], v70 offset:1024
	s_branch .LBB0_74
.LBB0_79:                               ;   in Loop: Header=BB0_70 Depth=1
	v_mov_b32_e32 v7, 0
	v_mov_b32_e32 v6, v7
	v_mov_b32_e32 v5, v7
	v_mov_b32_e32 v4, v7
	v_mov_b32_e32 v3, v7
	v_mov_b32_e32 v2, v7
	v_mov_b32_e32 v1, v7
	v_mov_b32_e32 v0, v7
	v_mov_b32_e32 v11, v7
	v_mov_b32_e32 v10, v7
	v_mov_b32_e32 v9, v7
	v_mov_b32_e32 v8, v7
	v_mov_b32_e32 v15, v7
	v_mov_b32_e32 v14, v7
	v_mov_b32_e32 v13, v7
	v_mov_b32_e32 v12, v7
	v_mov_b32_e32 v19, v7
	v_mov_b32_e32 v18, v7
	v_mov_b32_e32 v17, v7
	v_mov_b32_e32 v16, v7
.LBB0_80:                               ;   in Loop: Header=BB0_70 Depth=1
	; sched_barrier mask(0x00000000)
	s_nop 2
	v_add_f32_dpp v0, v1, v0 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v16, v17, v16 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v12, v13, v12 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v9, v8 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v2, v0 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v5, v4 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v16, v18, v16 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v12, v14, v12 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v10, v8 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v6, v2 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v16, v19, v16 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v12, v15, v12 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v11, v8 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v3, v0 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v7, v2 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v16, v16, v16 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v12, v12, v12 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v8, v8 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v0, v0 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v16, v16, v16 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v12, v12, v12 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v8, v8 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v0, v0 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v16, v16 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v12, v12 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v8, v8 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v0, v0 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v2, v2 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v16, v16, v16 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v12, v12, v12 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v8, v8 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v0, v0 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v2, v2 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v17, v16 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v13, v12 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v9, v8 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v1, v0 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v7, v6 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_and_saveexec_b64 s[36:37], s[16:17]
	s_cbranch_execz .LBB0_69
; %bb.81:                               ;   in Loop: Header=BB0_70 Depth=1
	v_add_f32_e32 v5, v16, v17
	v_add_f32_e32 v4, v12, v13
	v_add_f32_e32 v3, v8, v9
	v_add_f32_e32 v2, v0, v1
	v_add_f32_e32 v0, v6, v7
	s_andn2_b64 vcc, exec, s[20:21]
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v1, 16, v76
	s_cbranch_vccnz .LBB0_83
; %bb.82:                               ;   in Loop: Header=BB0_70 Depth=1
	v_mov_b32_e32 v69, v71
	v_lshlrev_b64 v[6:7], 1, v[68:69]
	v_lshl_add_u64 v[8:9], s[12:13], 0, v[6:7]
	global_load_ushort v10, v[8:9], off
	v_lshl_add_u64 v[6:7], s[14:15], 0, v[6:7]
	v_lshl_add_u64 v[8:9], s[22:23], 1, v[8:9]
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v10, 16, v10
	v_mul_f32_e32 v10, s25, v10
	v_fmac_f32_e32 v10, s24, v5
	v_add_f32_e32 v11, v10, v1
	v_cndmask_b32_e64 v10, v11, v10, s[0:1]
	v_cvt_pk_bf16_f32 v10, v10, s0
	global_store_short v[6:7], v10, off
	global_load_ushort v10, v[8:9], off
	v_lshl_add_u64 v[6:7], s[30:31], 1, v[6:7]
	v_lshl_add_u64 v[8:9], v[8:9], 0, s[26:27]
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v10, 16, v10
	v_mul_f32_e32 v10, s25, v10
	v_fmac_f32_e32 v10, s24, v4
	v_add_f32_e32 v11, v10, v1
	v_cndmask_b32_e64 v10, v11, v10, s[0:1]
	v_cvt_pk_bf16_f32 v10, v10, s0
	global_store_short v[6:7], v10, off
	global_load_ushort v10, v[8:9], off
	v_lshl_add_u64 v[6:7], v[6:7], 0, s[34:35]
	v_lshl_add_u64 v[8:9], v[8:9], 0, s[26:27]
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v10, 16, v10
	v_mul_f32_e32 v10, s25, v10
	v_fmac_f32_e32 v10, s24, v3
	v_add_f32_e32 v11, v10, v1
	v_cndmask_b32_e64 v10, v11, v10, s[0:1]
	v_cvt_pk_bf16_f32 v10, v10, s0
	global_store_short v[6:7], v10, off
	global_load_ushort v10, v[8:9], off
	v_lshl_add_u64 v[6:7], v[6:7], 0, s[34:35]
	v_lshl_add_u64 v[8:9], v[8:9], 0, s[26:27]
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v10, 16, v10
	v_mul_f32_e32 v10, s25, v10
	v_fmac_f32_e32 v10, s24, v2
	v_add_f32_e32 v11, v10, v1
	v_cndmask_b32_e64 v10, v11, v10, s[0:1]
	v_cvt_pk_bf16_f32 v10, v10, s0
	global_store_short v[6:7], v10, off
	global_load_ushort v8, v[8:9], off
	v_lshl_add_u64 v[6:7], v[6:7], 0, s[34:35]
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v8, 16, v8
	v_mul_f32_e32 v8, s25, v8
	v_fmac_f32_e32 v8, s24, v0
	v_add_f32_e32 v9, v8, v1
	v_cndmask_b32_e64 v8, v9, v8, s[0:1]
	v_cvt_pk_bf16_f32 v8, v8, s0
	global_store_short v[6:7], v8, off
	s_cbranch_execnz .LBB0_69
	s_branch .LBB0_84
.LBB0_83:                               ;   in Loop: Header=BB0_70 Depth=1
.LBB0_84:                               ;   in Loop: Header=BB0_70 Depth=1
	v_mul_f32_e32 v8, s24, v5
	v_fma_f32 v5, s24, v5, v1
	v_mov_b32_e32 v69, v71
	v_cndmask_b32_e64 v5, v5, v8, s[0:1]
	v_lshl_add_u64 v[6:7], v[68:69], 1, s[14:15]
	v_cvt_pk_bf16_f32 v5, v5, s0
	global_store_short v[6:7], v5, off
	v_mul_f32_e32 v5, s24, v4
	v_fma_f32 v4, s24, v4, v1
	v_cndmask_b32_e64 v4, v4, v5, s[0:1]
	v_lshl_add_u64 v[6:7], s[30:31], 1, v[6:7]
	v_cvt_pk_bf16_f32 v4, v4, s0
	global_store_short v[6:7], v4, off
	v_lshl_add_u64 v[4:5], v[6:7], 0, s[34:35]
	v_mul_f32_e32 v6, s24, v3
	v_fma_f32 v3, s24, v3, v1
	v_cndmask_b32_e64 v3, v3, v6, s[0:1]
	v_cvt_pk_bf16_f32 v3, v3, s0
	global_store_short v[4:5], v3, off
	v_mul_f32_e32 v3, s24, v2
	v_fma_f32 v2, s24, v2, v1
	v_cndmask_b32_e64 v2, v2, v3, s[0:1]
	v_lshl_add_u64 v[4:5], v[4:5], 0, s[34:35]
	v_cvt_pk_bf16_f32 v2, v2, s0
	global_store_short v[4:5], v2, off
	v_lshl_add_u64 v[2:3], v[4:5], 0, s[34:35]
	v_mul_f32_e32 v4, s24, v0
	v_fmac_f32_e32 v1, s24, v0
	v_cndmask_b32_e64 v0, v1, v4, s[0:1]
	v_cvt_pk_bf16_f32 v0, v0, s0
	global_store_short v[2:3], v0, off
	s_branch .LBB0_69
.LBB0_85:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wvSplitK_vllm_bf16_tn_bias_n5_y1u2
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
		.amdhsa_accum_offset 84
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
	.size	wvSplitK_vllm_bf16_tn_bias_n5_y1u2, .Lfunc_end0-wvSplitK_vllm_bf16_tn_bias_n5_y1u2
                                        ; -- End function
	.set wvSplitK_vllm_bf16_tn_bias_n5_y1u2.num_vgpr, 83
	.set wvSplitK_vllm_bf16_tn_bias_n5_y1u2.num_agpr, 0
	.set wvSplitK_vllm_bf16_tn_bias_n5_y1u2.numbered_sgpr, 71
	.set wvSplitK_vllm_bf16_tn_bias_n5_y1u2.num_named_barrier, 0
	.set wvSplitK_vllm_bf16_tn_bias_n5_y1u2.private_seg_size, 0
	.set wvSplitK_vllm_bf16_tn_bias_n5_y1u2.uses_vcc, 1
	.set wvSplitK_vllm_bf16_tn_bias_n5_y1u2.uses_flat_scratch, 0
	.set wvSplitK_vllm_bf16_tn_bias_n5_y1u2.has_dyn_sized_stack, 0
	.set wvSplitK_vllm_bf16_tn_bias_n5_y1u2.has_recursion, 0
	.set wvSplitK_vllm_bf16_tn_bias_n5_y1u2.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 6164
; TotalNumSgprs: 77
; NumVgprs: 83
; NumAgprs: 0
; TotalNumVgprs: 83
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 163840 bytes/workgroup (compile time only)
; SGPRBlocks: 12
; VGPRBlocks: 12
; NumSGPRsForWavesPerEU: 102
; NumVGPRsForWavesPerEU: 97
; AccumOffset: 84
; Occupancy: 4
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 1
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 20
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_4d36c5f5227dbf7f,@object ; @__hip_cuid_4d36c5f5227dbf7f
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_4d36c5f5227dbf7f
__hip_cuid_4d36c5f5227dbf7f:
	.byte	0                               ; 0x0
	.size	__hip_cuid_4d36c5f5227dbf7f, 1

	.ident	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.4 26084 f58b06dce1f9c15707c5f808fd002e18c2accf7e)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_4d36c5f5227dbf7f
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
  AssertSizeEqual: { 1: 5, 2: 1 }
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
    .name:           wvSplitK_vllm_bf16_tn_bias_n5_y1u2
    .private_segment_fixed_size: 0
    .sgpr_count:     77
    .sgpr_spill_count: 0
    .symbol:         wvSplitK_vllm_bf16_tn_bias_n5_y1u2.kd
    .vgpr_count:     83
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 1
...

	.end_amdgpu_metadata
