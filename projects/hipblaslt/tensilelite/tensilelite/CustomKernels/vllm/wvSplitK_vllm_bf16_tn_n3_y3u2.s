	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 4
	.text
	.protected	wvSplitK_vllm_bf16_tn_n3_y3u2 ; -- Begin function wvSplitK_vllm_bf16_tn_n3_y3u2
	.globl	wvSplitK_vllm_bf16_tn_n3_y3u2
	.p2align	8
	.type	wvSplitK_vllm_bf16_tn_n3_y3u2,@function
wvSplitK_vllm_bf16_tn_n3_y3u2:          ; @wvSplitK_vllm_bf16_tn_n3_y3u2
; %bb.158:
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.159:
.LBB0_0:
	s_load_dwordx4 s[28:31], s[0:1], 0x38
	s_mov_b32 s26, s11
	s_mov_b64 s[0:1], -1
	s_cmpk_gt_i32 s2, 0x6aaa
	v_bfe_u32 v98, v0, 10, 10
	s_mul_i32 s11, s8, 3
	s_mul_i32 s17, s8, 48
	s_cbranch_scc0 .LBB0_85
; %bb.1:
	s_abs_i32 s0, s17
	v_cvt_f32_u32_e32 v1, s0
	s_sub_i32 s8, 0, s0
	s_add_i32 s33, s3, -1
	s_add_i32 s1, s17, s33
	v_rcp_iflag_f32_e32 v1, v1
	s_xor_b32 s9, s1, s17
	s_abs_i32 s1, s1
	s_ashr_i32 s9, s9, 31
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	s_nop 0
	v_readfirstlane_b32 s18, v1
	s_mul_i32 s8, s8, s18
	s_mul_hi_u32 s8, s18, s8
	s_add_i32 s18, s18, s8
	s_mul_hi_u32 s8, s1, s18
	s_mul_i32 s18, s8, s0
	s_sub_i32 s1, s1, s18
	s_add_i32 s19, s8, 1
	s_sub_i32 s18, s1, s0
	s_cmp_ge_u32 s1, s0
	s_cselect_b32 s8, s19, s8
	s_cselect_b32 s1, s18, s1
	s_add_i32 s18, s8, 1
	s_cmp_ge_u32 s1, s0
	s_cselect_b32 s0, s18, s8
	s_xor_b32 s0, s0, s9
	s_sub_i32 s0, s0, s9
	s_mul_i32 s0, s0, s11
	s_abs_i32 s1, s0
	v_cvt_f32_u32_e32 v1, s1
	s_sub_i32 s9, 0, s1
	s_add_i32 s8, s0, s33
	s_xor_b32 s0, s8, s0
	v_rcp_iflag_f32_e32 v1, v1
	s_abs_i32 s8, s8
	s_ashr_i32 s0, s0, 31
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	s_nop 0
	v_readfirstlane_b32 s18, v1
	s_mul_i32 s9, s9, s18
	s_mul_hi_u32 s9, s18, s9
	s_add_i32 s18, s18, s9
	s_mul_hi_u32 s9, s8, s18
	s_mul_i32 s18, s9, s1
	s_sub_i32 s8, s8, s18
	s_add_i32 s19, s9, 1
	s_sub_i32 s18, s8, s1
	s_cmp_ge_u32 s8, s1
	s_cselect_b32 s9, s19, s9
	s_cselect_b32 s8, s18, s8
	s_add_i32 s18, s9, 1
	s_cmp_ge_u32 s8, s1
	s_cselect_b32 s1, s18, s9
	s_xor_b32 s1, s1, s0
	s_sub_i32 s0, s1, s0
	s_max_i32 s8, s0, 4
	v_cmp_gt_u32_e32 vcc, s8, v98
	s_and_saveexec_b64 s[34:35], vcc
	s_cbranch_execz .LBB0_84
; %bb.2:
	s_mul_i32 s9, s8, 3
	v_cvt_f32_u32_e32 v3, s9
	s_mul_i32 s0, s8, s16
	v_add_u32_e32 v1, s0, v98
	v_lshl_add_u32 v2, v1, 1, v1
	v_rcp_iflag_f32_e32 v3, v3
	v_add_u32_e32 v1, 3, v2
	v_cmp_gt_u32_e32 vcc, s3, v2
	v_cmp_lt_u32_e64 s[0:1], s3, v1
	s_add_i32 s66, s3, -3
	v_mov_b32_e32 v4, s66
	s_and_b64 s[0:1], vcc, s[0:1]
	v_cndmask_b32_e64 v88, v2, v4, s[0:1]
	v_mul_f32_e32 v2, 0x4f7ffffe, v3
	v_cvt_u32_f32_e32 v2, v2
	s_sub_i32 s18, 0, s9
	s_abs_i32 s1, s3
	s_ashr_i32 s0, s3, 31
	v_readfirstlane_b32 s19, v2
	s_mul_i32 s18, s18, s19
	s_mul_hi_u32 s18, s19, s18
	s_add_i32 s19, s19, s18
	s_mul_hi_u32 s18, s1, s19
	s_mul_i32 s18, s18, s9
	s_sub_i32 s1, s1, s18
	s_sub_i32 s18, s1, s9
	s_cmp_ge_u32 s1, s9
	s_cselect_b32 s1, s18, s1
	s_sub_i32 s18, s1, s9
	s_cmp_ge_u32 s1, s9
	s_cselect_b32 s1, s18, s1
	s_xor_b32 s1, s1, s0
	s_sub_i32 s0, s1, s0
	s_add_i32 s9, s9, s3
	s_sub_i32 s1, s9, s0
	s_cmp_eq_u32 s0, 0
	s_cselect_b32 s67, s3, s1
	v_cmp_gt_u32_e64 s[0:1], s67, v88
	s_and_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB0_84
; %bb.3:
	s_and_b32 s0, s26, 7
	s_cmp_lg_u32 s0, 0
	s_mov_b32 s37, 0
	s_cselect_b64 s[0:1], -1, 0
	s_and_b32 s36, s6, 15
	s_cmp_lg_u64 s[36:37], 0
	s_cselect_b64 s[18:19], -1, 0
	s_or_b64 s[40:41], s[18:19], s[0:1]
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s19, s30, 31
	s_mov_b32 s18, s30
	s_ashr_i32 s27, s26, 31
	s_lshl_b32 s68, s8, 9
	s_add_i32 s69, s2, -8
	s_ashr_i32 s70, s10, 31
	s_ashr_i32 s21, s31, 31
	s_lshl_b64 s[18:19], s[18:19], 1
	s_mov_b32 s20, s31
	s_add_u32 s46, s12, s18
	s_addc_u32 s47, s13, s19
	s_lshl_b64 s[20:21], s[20:21], 1
	s_add_u32 s48, s14, s20
	s_addc_u32 s49, s15, s21
	s_add_u32 s50, s46, s18
	v_and_b32_e32 v2, 0x3ff, v0
	s_addc_u32 s51, s47, s19
	v_sub_u32_e64 v1, v1, s3 clamp
	v_lshlrev_b32_e32 v99, 3, v2
	v_cmp_eq_u32_e64 s[0:1], 63, v2
	s_add_u32 s52, s48, s20
	v_lshlrev_b32_e32 v2, 4, v2
	s_mov_b64 s[38:39], 0
	v_mov_b32_e32 v3, 0
	v_cndmask_b32_e32 v1, 0, v1, vcc
	v_lshl_add_u32 v100, v98, 9, v99
	s_mul_i32 s71, s8, s11
	v_cmp_eq_f32_e64 s[42:43], s29, 0
	v_cmp_neq_f32_e64 s[44:45], s29, 0
	s_addc_u32 s53, s49, s21
	s_lshl_b64 s[54:55], s[26:27], 1
	v_lshl_add_u32 v101, v98, 10, v2
	s_lshl_b32 s72, s8, 10
	v_cndmask_b32_e64 v102, 0, 1, s[40:41]
	s_branch .LBB0_6
.LBB0_4:                                ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[58:59]
	v_add_u32_e32 v4, 3, v2
	v_cmp_gt_u32_e32 vcc, s3, v2
	v_cmp_lt_u32_e64 s[8:9], s3, v4
	v_subrev_u32_e32 v4, s3, v4
	v_mov_b32_e32 v5, s66
	s_and_b64 vcc, vcc, s[8:9]
	v_cndmask_b32_e32 v2, v2, v5, vcc
	v_cndmask_b32_e32 v1, v1, v4, vcc
.LBB0_5:                                ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[56:57]
	v_cmp_le_u32_e32 vcc, s67, v2
	s_or_b64 s[38:39], vcc, s[38:39]
	v_mov_b32_e32 v88, v2
	s_andn2_b64 exec, exec, s[38:39]
	s_cbranch_execz .LBB0_84
.LBB0_6:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_10 Depth 2
                                        ;       Child Loop BB0_15 Depth 3
	v_min_u32_e32 v4, s33, v88
	v_mad_u64_u32 v[6:7], s[20:21], v4, s10, 0
	v_mov_b32_e32 v2, v7
	v_mad_u64_u32 v[4:5], s[20:21], v4, s70, v[2:3]
	v_add_u32_e32 v2, 1, v88
	v_mov_b32_e32 v7, v4
	v_min_u32_e32 v4, s33, v2
	v_mad_u64_u32 v[8:9], s[20:21], v4, s10, 0
	v_mov_b32_e32 v2, v9
	v_mad_u64_u32 v[4:5], s[20:21], v4, s70, v[2:3]
	v_add_u32_e32 v90, 2, v88
	v_mov_b32_e32 v9, v4
	v_min_u32_e32 v4, s33, v90
	v_mad_u64_u32 v[10:11], s[20:21], v4, s10, 0
	v_mov_b32_e32 v2, v11
	v_mad_u64_u32 v[4:5], s[20:21], v4, s70, v[2:3]
	v_mov_b32_e32 v11, v4
	v_mov_b32_e32 v4, v3
	v_mov_b32_e32 v5, v3
	v_mov_b32_e32 v2, v3
	v_lshlrev_b64 v[94:95], 1, v[8:9]
	v_lshlrev_b64 v[96:97], 1, v[10:11]
	v_mov_b64_e32 v[30:31], v[4:5]
	v_mov_b64_e32 v[34:35], v[4:5]
	v_mov_b64_e32 v[38:39], v[4:5]
	v_mov_b64_e32 v[26:27], v[4:5]
	v_mov_b64_e32 v[22:23], v[4:5]
	v_mov_b64_e32 v[18:19], v[4:5]
	v_mov_b64_e32 v[14:15], v[4:5]
	v_mov_b64_e32 v[10:11], v[4:5]
	v_lshlrev_b64 v[92:93], 1, v[6:7]
	v_mov_b64_e32 v[28:29], v[2:3]
	v_mov_b64_e32 v[32:33], v[2:3]
	v_mov_b64_e32 v[36:37], v[2:3]
	v_mov_b64_e32 v[24:25], v[2:3]
	v_mov_b64_e32 v[20:21], v[2:3]
	v_mov_b64_e32 v[16:17], v[2:3]
	v_mov_b64_e32 v[12:13], v[2:3]
	v_mov_b64_e32 v[8:9], v[2:3]
	v_mov_b64_e32 v[6:7], v[4:5]
	v_cmp_gt_u32_e64 s[8:9], s3, v88
	v_cmp_le_u32_e64 s[18:19], s3, v88
	v_mov_b64_e32 v[4:5], v[2:3]
	s_mov_b32 s36, s37
	s_mov_b32 s62, s37
	s_branch .LBB0_10
.LBB0_7:                                ;   in Loop: Header=BB0_10 Depth=2
	s_or_b64 exec, exec, s[24:25]
.LBB0_8:                                ;   in Loop: Header=BB0_10 Depth=2
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(5) lgkmcnt(2)
	v_mfma_f32_4x4x4_16b_bf16 v[36:39], v[84:85], v[56:57], v[36:39]
	s_waitcnt vmcnt(4)
	v_mfma_f32_4x4x4_16b_bf16 v[32:35], v[84:85], v[52:53], v[32:35]
	s_waitcnt vmcnt(3)
	v_mfma_f32_4x4x4_16b_bf16 v[28:31], v[84:85], v[60:61], v[28:31]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_4x4x4_16b_bf16 v[24:27], v[80:81], v[56:57], v[24:27]
	v_mfma_f32_4x4x4_16b_bf16 v[20:23], v[80:81], v[52:53], v[20:23]
	v_mfma_f32_4x4x4_16b_bf16 v[16:19], v[80:81], v[60:61], v[16:19]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_4x4x4_16b_bf16 v[12:15], v[76:77], v[56:57], v[12:15]
	v_mfma_f32_4x4x4_16b_bf16 v[8:11], v[76:77], v[52:53], v[8:11]
	v_mfma_f32_4x4x4_16b_bf16 v[4:7], v[76:77], v[60:61], v[4:7]
	v_mfma_f32_4x4x4_16b_bf16 v[36:39], v[86:87], v[58:59], v[36:39]
	v_mfma_f32_4x4x4_16b_bf16 v[32:35], v[86:87], v[54:55], v[32:35]
	v_mfma_f32_4x4x4_16b_bf16 v[28:31], v[86:87], v[62:63], v[28:31]
	v_mfma_f32_4x4x4_16b_bf16 v[24:27], v[82:83], v[58:59], v[24:27]
	v_mfma_f32_4x4x4_16b_bf16 v[20:23], v[82:83], v[54:55], v[20:23]
	v_mfma_f32_4x4x4_16b_bf16 v[16:19], v[82:83], v[62:63], v[16:19]
	v_mfma_f32_4x4x4_16b_bf16 v[12:15], v[78:79], v[58:59], v[12:15]
	v_mfma_f32_4x4x4_16b_bf16 v[8:11], v[78:79], v[54:55], v[8:11]
	v_mfma_f32_4x4x4_16b_bf16 v[4:7], v[78:79], v[62:63], v[4:7]
	s_waitcnt vmcnt(2)
	v_mfma_f32_4x4x4_16b_bf16 v[36:39], v[72:73], v[40:41], v[36:39]
	s_waitcnt vmcnt(1)
	v_mfma_f32_4x4x4_16b_bf16 v[32:35], v[72:73], v[48:49], v[32:35]
	s_waitcnt vmcnt(0)
	v_mfma_f32_4x4x4_16b_bf16 v[28:31], v[72:73], v[44:45], v[28:31]
	v_mfma_f32_4x4x4_16b_bf16 v[24:27], v[68:69], v[40:41], v[24:27]
	v_mfma_f32_4x4x4_16b_bf16 v[20:23], v[68:69], v[48:49], v[20:23]
	v_mfma_f32_4x4x4_16b_bf16 v[16:19], v[68:69], v[44:45], v[16:19]
	v_mfma_f32_4x4x4_16b_bf16 v[12:15], v[64:65], v[40:41], v[12:15]
	v_mfma_f32_4x4x4_16b_bf16 v[8:11], v[64:65], v[48:49], v[8:11]
	v_mfma_f32_4x4x4_16b_bf16 v[4:7], v[64:65], v[44:45], v[4:7]
	v_mfma_f32_4x4x4_16b_bf16 v[36:39], v[74:75], v[42:43], v[36:39]
	v_mfma_f32_4x4x4_16b_bf16 v[32:35], v[74:75], v[50:51], v[32:35]
	v_mfma_f32_4x4x4_16b_bf16 v[28:31], v[74:75], v[46:47], v[28:31]
	v_mfma_f32_4x4x4_16b_bf16 v[24:27], v[70:71], v[42:43], v[24:27]
	v_mfma_f32_4x4x4_16b_bf16 v[20:23], v[70:71], v[50:51], v[20:23]
	v_mfma_f32_4x4x4_16b_bf16 v[16:19], v[70:71], v[46:47], v[16:19]
	v_mfma_f32_4x4x4_16b_bf16 v[12:15], v[66:67], v[42:43], v[12:15]
	v_mfma_f32_4x4x4_16b_bf16 v[8:11], v[66:67], v[50:51], v[8:11]
	v_mfma_f32_4x4x4_16b_bf16 v[4:7], v[66:67], v[46:47], v[4:7]
.LBB0_9:                                ;   in Loop: Header=BB0_10 Depth=2
	s_or_b64 exec, exec, s[20:21]
	s_addk_i32 s62, 0x400
	s_cmp_ge_u32 s62, s2
	s_cbranch_scc1 .LBB0_32
.LBB0_10:                               ;   Parent Loop BB0_6 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_15 Depth 3
	s_cmp_eq_u32 s62, 0
	s_cselect_b64 s[20:21], -1, 0
	s_add_i32 s22, s36, 0x6800
	s_cmp_eq_u32 s62, s22
	s_cselect_b64 s[24:25], -1, 0
	s_or_b64 s[24:25], s[20:21], s[24:25]
	s_andn2_b64 vcc, exec, s[24:25]
	s_cbranch_vccz .LBB0_12
; %bb.11:                               ;   in Loop: Header=BB0_10 Depth=2
	s_and_saveexec_b64 s[20:21], s[8:9]
	s_cbranch_execz .LBB0_9
	s_branch .LBB0_29
.LBB0_12:                               ;   in Loop: Header=BB0_10 Depth=2
	s_and_b64 s[20:21], s[20:21], exec
	s_cselect_b32 s36, s36, s22
	s_sub_i32 s20, s2, s36
	s_min_u32 s63, s20, 0x6800
	s_lshl_b64 s[20:21], s[36:37], 1
	s_add_u32 s22, s6, s20
	s_addc_u32 s23, s7, s21
	s_mov_b32 s64, 0
	s_mov_b64 s[24:25], 0
	v_mov_b32_e32 v42, v101
	s_barrier
                                        ; implicit-def: $sgpr56_sgpr57
	s_branch .LBB0_15
.LBB0_13:                               ;   in Loop: Header=BB0_15 Depth=3
	s_add_i32 s64, s64, s68
	s_cmpk_gt_u32 s64, 0x67ff
	s_cselect_b64 s[20:21], -1, 0
	s_andn2_b64 s[56:57], s[56:57], exec
	s_and_b64 s[20:21], s[20:21], exec
	v_add_u32_e32 v42, s72, v42
	s_or_b64 s[56:57], s[56:57], s[20:21]
.LBB0_14:                               ;   in Loop: Header=BB0_15 Depth=3
	s_or_b64 exec, exec, s[58:59]
	s_and_b64 s[20:21], exec, s[56:57]
	s_or_b64 s[24:25], s[20:21], s[24:25]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execz .LBB0_28
.LBB0_15:                               ;   Parent Loop BB0_6 Depth=1
                                        ;     Parent Loop BB0_10 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	v_add_u32_e32 v2, s64, v100
	v_cmp_gt_u32_e32 vcc, s63, v2
	s_or_b64 s[56:57], s[56:57], exec
	s_and_saveexec_b64 s[58:59], vcc
	s_cbranch_execz .LBB0_14
; %bb.16:                               ;   in Loop: Header=BB0_15 Depth=3
	v_lshl_add_u64 v[40:41], v[2:3], 1, s[22:23]
	s_mov_b64 s[20:21], -1
	s_and_b64 vcc, exec, s[40:41]
	s_cbranch_vccz .LBB0_18
; %bb.17:                               ;   in Loop: Header=BB0_15 Depth=3
	global_load_dwordx4 v[44:47], v[40:41], off
	s_mov_b64 s[20:21], 0
	s_waitcnt vmcnt(0)
	ds_write_b128 v42, v[44:47]
.LBB0_18:                               ;   in Loop: Header=BB0_15 Depth=3
	s_andn2_b64 vcc, exec, s[20:21]
	s_cbranch_vccnz .LBB0_20
; %bb.19:                               ;   in Loop: Header=BB0_15 Depth=3
	v_readfirstlane_b32 s20, v42
	s_mov_b32 m0, s20
	s_nop 0
	global_load_lds_dwordx4 v[40:41], off
.LBB0_20:                               ;   in Loop: Header=BB0_15 Depth=3
	v_lshl_add_u64 v[40:41], s[26:27], 1, v[40:41]
	v_add_u32_e32 v2, 0xd000, v42
	v_cmp_ne_u32_e64 s[20:21], 1, v102
	s_andn2_b64 vcc, exec, s[40:41]
	s_mov_b64 s[60:61], -1
	s_cbranch_vccnz .LBB0_22
; %bb.21:                               ;   in Loop: Header=BB0_15 Depth=3
	global_load_dwordx4 v[44:47], v[40:41], off
	s_mov_b64 s[60:61], 0
	s_waitcnt vmcnt(0)
	ds_write_b128 v2, v[44:47]
.LBB0_22:                               ;   in Loop: Header=BB0_15 Depth=3
	s_andn2_b64 vcc, exec, s[60:61]
	s_cbranch_vccnz .LBB0_24
; %bb.23:                               ;   in Loop: Header=BB0_15 Depth=3
	v_readfirstlane_b32 s60, v2
	s_mov_b32 m0, s60
	s_nop 0
	global_load_lds_dwordx4 v[40:41], off
.LBB0_24:                               ;   in Loop: Header=BB0_15 Depth=3
	v_lshl_add_u64 v[40:41], v[40:41], 0, s[54:55]
	v_add_u32_e32 v2, 0x1a000, v42
	s_and_b64 vcc, exec, s[20:21]
	s_mov_b64 s[20:21], -1
	s_cbranch_vccnz .LBB0_26
; %bb.25:                               ;   in Loop: Header=BB0_15 Depth=3
	global_load_dwordx4 v[44:47], v[40:41], off
	s_mov_b64 s[20:21], 0
	s_waitcnt vmcnt(0)
	ds_write_b128 v2, v[44:47]
.LBB0_26:                               ;   in Loop: Header=BB0_15 Depth=3
	s_andn2_b64 vcc, exec, s[20:21]
	s_cbranch_vccnz .LBB0_13
; %bb.27:                               ;   in Loop: Header=BB0_15 Depth=3
	v_readfirstlane_b32 s20, v2
	s_mov_b32 m0, s20
	s_nop 0
	global_load_lds_dwordx4 v[40:41], off
	s_branch .LBB0_13
.LBB0_28:                               ;   in Loop: Header=BB0_10 Depth=2
	s_or_b64 exec, exec, s[24:25]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	s_and_saveexec_b64 s[20:21], s[8:9]
	s_cbranch_execz .LBB0_9
.LBB0_29:                               ;   in Loop: Header=BB0_10 Depth=2
	v_add_u32_e32 v89, s62, v99
	v_min_u32_e32 v2, s69, v89
	v_lshl_add_u64 v[40:41], v[2:3], 1, s[4:5]
	v_add_u32_e32 v91, 0x200, v89
	v_lshl_add_u64 v[42:43], v[40:41], 0, v[92:93]
	v_lshl_add_u64 v[44:45], v[40:41], 0, v[94:95]
	v_min_u32_e32 v2, s69, v91
	global_load_dwordx4 v[56:59], v[42:43], off nt
	global_load_dwordx4 v[52:55], v[44:45], off nt
	v_lshl_add_u64 v[44:45], v[2:3], 1, s[4:5]
	v_lshl_add_u64 v[40:41], v[40:41], 0, v[96:97]
	v_lshl_add_u64 v[42:43], v[44:45], 0, v[92:93]
	v_lshl_add_u64 v[46:47], v[44:45], 0, v[94:95]
	v_lshl_add_u64 v[44:45], v[44:45], 0, v[96:97]
	global_load_dwordx4 v[60:63], v[40:41], off nt
	s_nop 0
	global_load_dwordx4 v[40:43], v[42:43], off nt
	s_nop 0
	global_load_dwordx4 v[48:51], v[46:47], off nt
	s_nop 0
	global_load_dwordx4 v[44:47], v[44:45], off nt
	v_cmp_gt_u32_e32 vcc, s2, v89
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v79, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v75, 0
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v73, 0
	v_mov_b32_e32 v86, 0
	v_mov_b32_e32 v87, 0
	v_mov_b32_e32 v84, 0
	v_mov_b32_e32 v85, 0
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_8
; %bb.30:                               ;   in Loop: Header=BB0_10 Depth=2
	v_subrev_u32_e32 v2, s36, v89
	v_lshlrev_b32_e32 v2, 1, v2
	v_add_u32_e32 v64, 0x1a000, v2
	ds_read_b128 v[84:87], v2
	ds_read_b128 v[80:83], v2 offset:53248
	ds_read_b128 v[76:79], v64
	v_cmp_gt_u32_e32 vcc, s2, v91
	v_mov_b32_e32 v73, 0
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v75, 0
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v66, 0
	s_and_saveexec_b64 s[24:25], vcc
	s_cbranch_execz .LBB0_7
; %bb.31:                               ;   in Loop: Header=BB0_10 Depth=2
	ds_read_b128 v[72:75], v2 offset:1024
	ds_read_b128 v[68:71], v2 offset:54272
	v_add_u32_e32 v2, 0x1a400, v2
	ds_read_b128 v[64:67], v2
	s_branch .LBB0_7
.LBB0_32:                               ;   in Loop: Header=BB0_6 Depth=1
	v_add_u32_e32 v2, s71, v88
	s_and_saveexec_b64 s[8:9], s[18:19]
	s_xor_b64 s[8:9], exec, s[8:9]
; %bb.33:                               ;   in Loop: Header=BB0_6 Depth=1
                                        ; implicit-def: $vgpr88
                                        ; implicit-def: $vgpr90
; %bb.34:                               ;   in Loop: Header=BB0_6 Depth=1
	s_andn2_saveexec_b64 s[56:57], s[8:9]
	s_cbranch_execz .LBB0_5
; %bb.35:                               ;   in Loop: Header=BB0_6 Depth=1
	v_add_f32_dpp v36, v37, v36 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v32, v33, v32 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v28, v29, v28 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v24, v25, v24 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v20, v21, v20 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v16, v17, v16 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v12, v13, v12 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v9, v8 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v5, v4 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v36, v38, v36 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v32, v34, v32 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v28, v30, v28 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v24, v26, v24 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v20, v22, v20 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v16, v18, v16 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v12, v14, v12 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v10, v8 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v6, v4 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v36, v39, v36 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v32, v35, v32 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v28, v31, v28 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v24, v27, v24 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v20, v23, v20 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v16, v19, v16 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v12, v15, v12 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v11, v8 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v7, v4 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v36, v36, v36 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v32, v32, v32 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v28, v28, v28 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v24, v24, v24 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v20, v20, v20 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v16, v16, v16 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v12, v12, v12 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v8, v8 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v36, v36, v36 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v32, v32, v32 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v28, v28, v28 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v24, v24, v24 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v20, v20, v20 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v16, v16, v16 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v12, v12, v12 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v8, v8 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v36, v36 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v32, v32 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v28, v28 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v24, v24 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v20, v20 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v16, v16 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v12, v12 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v8, v8 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v4, v4 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v36, v36, v36 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v32, v32, v32 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v28, v28, v28 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v24, v24, v24 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v20, v20, v20 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v16, v16, v16 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v12, v12, v12 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v15, v8, v8 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v37, v36 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v33, v32 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v29, v28 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v25, v24 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v21, v20 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v17, v16 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v14, v12 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v18, v15 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v19, v4 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_and_saveexec_b64 s[58:59], s[0:1]
	s_cbranch_execz .LBB0_4
; %bb.36:                               ;   in Loop: Header=BB0_6 Depth=1
	v_cmp_eq_u32_e64 s[8:9], 0, v1
	s_and_b64 s[18:19], s[42:43], s[8:9]
	v_add_f32_e32 v13, v36, v37
	v_add_f32_e32 v11, v32, v33
	v_add_f32_e32 v10, v28, v29
	v_add_f32_e32 v9, v24, v25
	v_add_f32_e32 v8, v20, v21
	v_add_f32_e32 v7, v16, v17
	v_add_f32_e32 v6, v12, v14
	v_add_f32_e32 v5, v15, v18
	v_add_f32_e32 v4, v4, v19
	s_xor_b64 s[20:21], s[18:19], -1
	v_mov_b32_e32 v91, v3
	s_mov_b64 s[18:19], 0
                                        ; implicit-def: $vgpr12
	s_and_saveexec_b64 s[22:23], s[20:21]
	s_xor_b64 s[60:61], exec, s[22:23]
	s_cbranch_execnz .LBB0_39
; %bb.37:                               ;   in Loop: Header=BB0_6 Depth=1
	s_andn2_saveexec_b64 s[8:9], s[60:61]
	s_cbranch_execnz .LBB0_82
.LBB0_38:                               ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[8:9]
	s_and_b64 exec, exec, s[18:19]
	s_cbranch_execz .LBB0_4
	s_branch .LBB0_83
.LBB0_39:                               ;   in Loop: Header=BB0_6 Depth=1
	v_cndmask_b32_e64 v12, 0, 1, s[44:45]
	v_cmp_ne_u32_e64 s[22:23], 1, v12
	s_and_saveexec_b64 s[18:19], s[8:9]
	s_cbranch_execz .LBB0_43
; %bb.40:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v12, s28, v13
	s_and_b64 vcc, exec, s[22:23]
	v_mov_b32_e32 v89, v3
	s_cbranch_vccnz .LBB0_42
; %bb.41:                               ;   in Loop: Header=BB0_6 Depth=1
	v_lshl_add_u64 v[14:15], v[88:89], 1, s[12:13]
	global_load_ushort v13, v[14:15], off
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v13, 16, v13
	v_fmac_f32_e32 v12, s29, v13
.LBB0_42:                               ;   in Loop: Header=BB0_6 Depth=1
	v_cvt_pk_bf16_f32 v14, v12, s0
	v_lshl_add_u64 v[12:13], v[88:89], 1, s[14:15]
	global_store_short v[12:13], v14, off
.LBB0_43:                               ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[18:19]
	v_cndmask_b32_e64 v12, 0, 1, s[42:43]
	v_cmp_gt_u32_e64 s[18:19], 2, v1
	v_cmp_ne_u32_e64 s[20:21], 1, v12
	s_and_saveexec_b64 s[24:25], s[18:19]
	s_cbranch_execnz .LBB0_50
; %bb.44:                               ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[24:25]
	v_cmp_gt_u32_e64 s[24:25], 3, v1
	s_and_saveexec_b64 s[62:63], s[24:25]
	s_cbranch_execnz .LBB0_55
.LBB0_45:                               ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[62:63]
	s_and_saveexec_b64 s[62:63], s[8:9]
	s_cbranch_execnz .LBB0_58
.LBB0_46:                               ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[62:63]
	s_and_saveexec_b64 s[62:63], s[18:19]
	s_cbranch_execnz .LBB0_61
.LBB0_47:                               ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[62:63]
	s_and_saveexec_b64 s[62:63], s[24:25]
	s_cbranch_execnz .LBB0_66
.LBB0_48:                               ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[62:63]
	s_and_saveexec_b64 s[62:63], s[8:9]
	s_cbranch_execnz .LBB0_69
.LBB0_49:                               ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[62:63]
	s_and_saveexec_b64 s[8:9], s[18:19]
	s_cbranch_execnz .LBB0_72
	s_branch .LBB0_77
.LBB0_50:                               ;   in Loop: Header=BB0_6 Depth=1
	s_and_b64 vcc, exec, s[20:21]
	v_mov_b32_e32 v89, v3
	s_cbranch_vccnz .LBB0_52
; %bb.51:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v11, s28, v11
	s_cbranch_execz .LBB0_53
	s_branch .LBB0_54
.LBB0_52:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v11, s28, v11
.LBB0_53:                               ;   in Loop: Header=BB0_6 Depth=1
	v_lshl_add_u64 v[12:13], v[88:89], 1, s[12:13]
	global_load_ushort v12, v[12:13], off offset:2
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v12, 16, v12
	v_fmac_f32_e32 v11, s29, v12
.LBB0_54:                               ;   in Loop: Header=BB0_6 Depth=1
	v_cvt_pk_bf16_f32 v11, v11, s0
	v_lshl_add_u64 v[12:13], v[88:89], 1, s[14:15]
	global_store_short v[12:13], v11, off offset:2
	s_or_b64 exec, exec, s[24:25]
	v_cmp_gt_u32_e64 s[24:25], 3, v1
	s_and_saveexec_b64 s[62:63], s[24:25]
	s_cbranch_execz .LBB0_45
.LBB0_55:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v10, s28, v10
	s_and_b64 vcc, exec, s[22:23]
	v_mov_b32_e32 v91, v3
	s_cbranch_vccnz .LBB0_57
; %bb.56:                               ;   in Loop: Header=BB0_6 Depth=1
	v_lshl_add_u64 v[12:13], v[90:91], 1, s[12:13]
	global_load_ushort v11, v[12:13], off
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v11, 16, v11
	v_fmac_f32_e32 v10, s29, v11
.LBB0_57:                               ;   in Loop: Header=BB0_6 Depth=1
	v_cvt_pk_bf16_f32 v12, v10, s0
	v_lshl_add_u64 v[10:11], v[90:91], 1, s[14:15]
	global_store_short v[10:11], v12, off
	s_or_b64 exec, exec, s[62:63]
	s_and_saveexec_b64 s[62:63], s[8:9]
	s_cbranch_execz .LBB0_46
.LBB0_58:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v9, s28, v9
	s_and_b64 vcc, exec, s[22:23]
	v_mov_b32_e32 v89, v3
	s_cbranch_vccnz .LBB0_60
; %bb.59:                               ;   in Loop: Header=BB0_6 Depth=1
	v_lshl_add_u64 v[10:11], v[88:89], 1, s[46:47]
	global_load_ushort v10, v[10:11], off
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v10, 16, v10
	v_fmac_f32_e32 v9, s29, v10
.LBB0_60:                               ;   in Loop: Header=BB0_6 Depth=1
	v_cvt_pk_bf16_f32 v9, v9, s0
	v_lshl_add_u64 v[10:11], v[88:89], 1, s[48:49]
	global_store_short v[10:11], v9, off
	s_or_b64 exec, exec, s[62:63]
	s_and_saveexec_b64 s[62:63], s[18:19]
	s_cbranch_execz .LBB0_47
.LBB0_61:                               ;   in Loop: Header=BB0_6 Depth=1
	s_and_b64 vcc, exec, s[20:21]
	v_mov_b32_e32 v89, v3
	s_cbranch_vccnz .LBB0_63
; %bb.62:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v8, s28, v8
	s_cbranch_execz .LBB0_64
	s_branch .LBB0_65
.LBB0_63:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v8, s28, v8
.LBB0_64:                               ;   in Loop: Header=BB0_6 Depth=1
	v_lshl_add_u64 v[10:11], v[88:89], 1, s[46:47]
	global_load_ushort v9, v[10:11], off offset:2
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v9, 16, v9
	v_fmac_f32_e32 v8, s29, v9
.LBB0_65:                               ;   in Loop: Header=BB0_6 Depth=1
	v_cvt_pk_bf16_f32 v10, v8, s0
	v_lshl_add_u64 v[8:9], v[88:89], 1, s[48:49]
	global_store_short v[8:9], v10, off offset:2
	s_or_b64 exec, exec, s[62:63]
	s_and_saveexec_b64 s[62:63], s[24:25]
	s_cbranch_execz .LBB0_48
.LBB0_66:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v7, s28, v7
	s_and_b64 vcc, exec, s[22:23]
	v_mov_b32_e32 v91, v3
	s_cbranch_vccnz .LBB0_68
; %bb.67:                               ;   in Loop: Header=BB0_6 Depth=1
	v_lshl_add_u64 v[8:9], v[90:91], 1, s[46:47]
	global_load_ushort v8, v[8:9], off
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v8, 16, v8
	v_fmac_f32_e32 v7, s29, v8
.LBB0_68:                               ;   in Loop: Header=BB0_6 Depth=1
	v_cvt_pk_bf16_f32 v7, v7, s0
	v_lshl_add_u64 v[8:9], v[90:91], 1, s[48:49]
	global_store_short v[8:9], v7, off
	s_or_b64 exec, exec, s[62:63]
	s_and_saveexec_b64 s[62:63], s[8:9]
	s_cbranch_execz .LBB0_49
.LBB0_69:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v6, s28, v6
	s_and_b64 vcc, exec, s[22:23]
	v_mov_b32_e32 v89, v3
	s_cbranch_vccnz .LBB0_71
; %bb.70:                               ;   in Loop: Header=BB0_6 Depth=1
	v_lshl_add_u64 v[8:9], v[88:89], 1, s[50:51]
	global_load_ushort v7, v[8:9], off
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v7, 16, v7
	v_fmac_f32_e32 v6, s29, v7
.LBB0_71:                               ;   in Loop: Header=BB0_6 Depth=1
	v_cvt_pk_bf16_f32 v8, v6, s0
	v_lshl_add_u64 v[6:7], v[88:89], 1, s[52:53]
	global_store_short v[6:7], v8, off
	s_or_b64 exec, exec, s[62:63]
	s_and_saveexec_b64 s[8:9], s[18:19]
	s_cbranch_execz .LBB0_77
.LBB0_72:                               ;   in Loop: Header=BB0_6 Depth=1
	s_and_b64 vcc, exec, s[20:21]
	v_mov_b32_e32 v89, v3
	s_cbranch_vccnz .LBB0_74
; %bb.73:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v5, s28, v5
	s_cbranch_execz .LBB0_75
	s_branch .LBB0_76
.LBB0_74:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v5, s28, v5
.LBB0_75:                               ;   in Loop: Header=BB0_6 Depth=1
	v_lshl_add_u64 v[6:7], v[88:89], 1, s[50:51]
	global_load_ushort v6, v[6:7], off offset:2
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v6, 16, v6
	v_fmac_f32_e32 v5, s29, v6
.LBB0_76:                               ;   in Loop: Header=BB0_6 Depth=1
	v_cvt_pk_bf16_f32 v5, v5, s0
	v_lshl_add_u64 v[6:7], v[88:89], 1, s[52:53]
	global_store_short v[6:7], v5, off offset:2
.LBB0_77:                               ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[8:9]
	s_mov_b64 s[18:19], 0
                                        ; implicit-def: $vgpr12
	s_and_saveexec_b64 s[8:9], s[24:25]
	s_xor_b64 s[8:9], exec, s[8:9]
	s_cbranch_execz .LBB0_81
; %bb.78:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v12, s28, v4
	s_and_b64 vcc, exec, s[22:23]
	v_mov_b32_e32 v91, v3
	s_cbranch_vccnz .LBB0_80
; %bb.79:                               ;   in Loop: Header=BB0_6 Depth=1
	v_lshl_add_u64 v[4:5], v[90:91], 1, s[50:51]
	global_load_ushort v4, v[4:5], off
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v4, 16, v4
	v_fmac_f32_e32 v12, s29, v4
.LBB0_80:                               ;   in Loop: Header=BB0_6 Depth=1
	s_mov_b64 s[18:19], exec
.LBB0_81:                               ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[8:9]
	s_and_b64 s[18:19], s[18:19], exec
                                        ; implicit-def: $vgpr13
                                        ; implicit-def: $vgpr11
                                        ; implicit-def: $vgpr10
                                        ; implicit-def: $vgpr9
                                        ; implicit-def: $vgpr8
                                        ; implicit-def: $vgpr7
                                        ; implicit-def: $vgpr6
                                        ; implicit-def: $vgpr5
                                        ; implicit-def: $vgpr4
                                        ; implicit-def: $vgpr88
	s_andn2_saveexec_b64 s[8:9], s[60:61]
	s_cbranch_execz .LBB0_38
.LBB0_82:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mov_b32_e32 v89, v3
	v_mul_f32_e32 v16, s28, v13
	v_lshlrev_b64 v[12:13], 1, v[88:89]
	v_mul_f32_e32 v11, s28, v11
	v_lshl_add_u64 v[14:15], s[14:15], 0, v[12:13]
	v_cvt_pk_bf16_f32 v11, v16, v11
	v_mul_f32_e32 v10, s28, v10
	global_store_dword v[14:15], v11, off
	v_cvt_pk_bf16_f32 v16, v10, s0
	v_lshlrev_b64 v[10:11], 1, v[90:91]
	v_lshl_add_u64 v[14:15], s[14:15], 0, v[10:11]
	v_mul_f32_e32 v9, s28, v9
	v_mul_f32_e32 v8, s28, v8
	global_store_short v[14:15], v16, off
	v_lshl_add_u64 v[14:15], s[48:49], 0, v[12:13]
	v_cvt_pk_bf16_f32 v8, v9, v8
	v_mul_f32_e32 v7, s28, v7
	global_store_dword v[14:15], v8, off
	v_cvt_pk_bf16_f32 v7, v7, s0
	v_lshl_add_u64 v[8:9], s[48:49], 0, v[10:11]
	global_store_short v[8:9], v7, off
	v_mul_f32_e32 v8, s28, v6
	v_mul_f32_e32 v5, s28, v5
	v_lshl_add_u64 v[6:7], s[52:53], 0, v[12:13]
	v_cvt_pk_bf16_f32 v5, v8, v5
	global_store_dword v[6:7], v5, off
	v_mul_f32_e32 v12, s28, v4
	s_or_b64 s[18:19], s[18:19], exec
	s_or_b64 exec, exec, s[8:9]
	s_and_b64 exec, exec, s[18:19]
	s_cbranch_execz .LBB0_4
.LBB0_83:                               ;   in Loop: Header=BB0_6 Depth=1
	v_cvt_pk_bf16_f32 v6, v12, s0
	v_lshl_add_u64 v[4:5], v[90:91], 1, s[52:53]
	global_store_short v[4:5], v6, off
	s_branch .LBB0_4
.LBB0_84:
	s_or_b64 exec, exec, s[34:35]
	s_mov_b64 s[0:1], 0
.LBB0_85:
	s_andn2_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB0_157
; %bb.86:
	v_and_b32_e32 v8, 0x3ff, v0
	s_mul_i32 s22, s2, 3
	v_lshlrev_b32_e32 v94, 3, v8
	s_cmp_eq_u32 s26, s2
	s_mov_b64 s[8:9], 0
	s_cbranch_scc1 .LBB0_88
; %bb.87:
	s_and_b32 s18, s6, 15
	s_mov_b32 s19, 0
	s_mov_b64 s[0:1], -1
	s_min_u32 s22, s22, 0x14000
	v_lshl_add_u32 v9, v98, 9, v94
	s_cbranch_execz .LBB0_89
	s_branch .LBB0_90
.LBB0_88:
                                        ; implicit-def: $sgpr18_sgpr19
	s_mov_b64 s[0:1], 0
	s_min_u32 s22, s22, 0x14000
	v_lshl_add_u32 v9, v98, 9, v94
.LBB0_89:
	s_and_b32 s0, s2, 7
	s_cmp_lg_u32 s0, 0
	s_mov_b32 s19, 0
	s_cselect_b64 s[0:1], -1, 0
	s_and_b32 s18, s6, 15
	s_cmp_lg_u64 s[18:19], 0
	s_cselect_b64 s[8:9], -1, 0
	s_or_b64 s[0:1], s[0:1], s[8:9]
	s_mov_b64 s[8:9], -1
.LBB0_90:
	s_andn2_b64 vcc, exec, s[0:1]
	v_cmp_gt_u32_e64 s[0:1], s22, v9
	s_cbranch_vccz .LBB0_92
; %bb.91:
	s_andn2_b64 vcc, exec, s[8:9]
	s_cbranch_vccz .LBB0_100
	s_branch .LBB0_104
.LBB0_92:
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_99
; %bb.93:
	v_cvt_f32_u32_e32 v0, s2
	s_and_b32 s20, s26, 7
	s_cmp_lg_u32 s20, 0
	s_cselect_b64 s[20:21], -1, 0
	v_rcp_iflag_f32_e32 v0, v0
	s_cmp_lg_u64 s[18:19], 0
	s_cselect_b64 s[18:19], -1, 0
	s_sub_i32 s24, 0, s2
	v_mul_f32_e32 v0, 0x4f7ffffe, v0
	v_cvt_u32_f32_e32 v0, v0
	s_or_b64 s[18:19], s[20:21], s[18:19]
	s_mov_b64 s[0:1], 0
	s_ashr_i32 s23, s26, 31
	v_mul_lo_u32 v1, s24, v0
	v_mul_hi_u32 v1, v0, v1
	v_add_u32_e32 v0, v0, v1
	v_mov_b32_e32 v1, 0
	v_mad_u64_u32 v[2:3], s[20:21], v9, v0, 0
	v_lshlrev_b64 v[4:5], 13, v[0:1]
	v_lshlrev_b32_e32 v0, 4, v8
	v_lshl_add_u32 v10, v98, 10, v0
	v_mov_b32_e32 v0, v9
	s_branch .LBB0_95
.LBB0_94:                               ;   in Loop: Header=BB0_95 Depth=1
	v_add_u32_e32 v0, 0x2000, v0
	v_cmp_le_u32_e32 vcc, s22, v0
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[4:5]
	s_or_b64 s[0:1], vcc, s[0:1]
	v_add_u32_e32 v10, 0x4000, v10
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB0_99
.LBB0_95:                               ; =>This Inner Loop Header: Depth=1
	v_not_b32_e32 v11, v3
	v_mad_u64_u32 v[6:7], s[20:21], s24, v3, v[0:1]
	v_add_u32_e32 v7, 1, v3
	v_cmp_le_u32_e32 vcc, s2, v6
	v_mad_u64_u32 v[12:13], s[20:21], s2, v11, v[0:1]
	s_nop 0
	v_cndmask_b32_e32 v7, v3, v7, vcc
	v_cndmask_b32_e32 v6, v6, v12, vcc
	v_add_u32_e32 v11, 1, v7
	v_cmp_le_u32_e32 vcc, s2, v6
	s_nop 1
	v_cndmask_b32_e32 v11, v7, v11, vcc
	v_mad_u64_u32 v[12:13], s[20:21], v11, s26, 0
	v_mov_b32_e32 v14, v13
	v_mad_u64_u32 v[14:15], s[20:21], v11, s23, v[14:15]
	v_mad_u64_u32 v[6:7], s[20:21], s24, v11, v[0:1]
	v_mov_b32_e32 v13, v14
	v_mov_b32_e32 v7, v1
	v_lshl_add_u64 v[12:13], v[12:13], 1, s[6:7]
	v_lshl_add_u64 v[6:7], v[6:7], 1, v[12:13]
	s_mov_b64 s[20:21], -1
	s_and_b64 vcc, exec, s[18:19]
	s_cbranch_vccz .LBB0_97
; %bb.96:                               ;   in Loop: Header=BB0_95 Depth=1
	global_load_dwordx4 v[12:15], v[6:7], off
	s_mov_b64 s[20:21], 0
	s_waitcnt vmcnt(0)
	ds_write_b128 v10, v[12:15]
.LBB0_97:                               ;   in Loop: Header=BB0_95 Depth=1
	s_andn2_b64 vcc, exec, s[20:21]
	s_cbranch_vccnz .LBB0_94
; %bb.98:                               ;   in Loop: Header=BB0_95 Depth=1
	v_readfirstlane_b32 s20, v10
	s_mov_b32 m0, s20
	s_nop 0
	global_load_lds_dwordx4 v[6:7], off
	s_branch .LBB0_94
.LBB0_99:
	s_or_b64 exec, exec, s[8:9]
	s_cbranch_execnz .LBB0_104
.LBB0_100:
	v_cmp_gt_u32_e32 vcc, s22, v9
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_103
; %bb.101:
	v_lshlrev_b32_e32 v0, 4, v8
	v_lshl_add_u32 v2, v98, 10, v0
	v_lshlrev_b32_e32 v0, 1, v9
	v_mov_b32_e32 v1, 0
	v_lshl_add_u64 v[0:1], s[6:7], 0, v[0:1]
	s_mov_b64 s[6:7], 0
	s_mov_b64 s[8:9], 0x4000
.LBB0_102:                              ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s18, v2
	s_mov_b32 m0, s18
	v_add_u32_e32 v9, 0x2000, v9
	global_load_lds_dwordx4 v[0:1], off
	v_cmp_le_u32_e32 vcc, s22, v9
	v_add_u32_e32 v2, 0x4000, v2
	s_or_b64 s[6:7], vcc, s[6:7]
	v_lshl_add_u64 v[0:1], v[0:1], 0, s[8:9]
	s_andn2_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB0_102
.LBB0_103:
	s_or_b64 exec, exec, s[0:1]
.LBB0_104:
	s_abs_i32 s0, s17
	v_cvt_f32_u32_e32 v0, s0
	s_mov_b32 s1, s3
	;;#ASMSTART
	;;#ASMEND
	s_sub_i32 s6, 0, s0
	v_rcp_iflag_f32_e32 v0, v0
	s_add_i32 s1, s1, -1
	s_add_i32 s7, s1, s17
	s_xor_b32 s8, s7, s17
	v_mul_f32_e32 v0, 0x4f7ffffe, v0
	v_cvt_u32_f32_e32 v0, v0
	s_abs_i32 s7, s7
	s_ashr_i32 s8, s8, 31
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_readfirstlane_b32 s9, v0
	s_mul_i32 s6, s6, s9
	s_mul_hi_u32 s6, s9, s6
	s_add_i32 s9, s9, s6
	s_mul_hi_u32 s6, s7, s9
	s_mul_i32 s9, s6, s0
	s_sub_i32 s7, s7, s9
	s_add_i32 s17, s6, 1
	s_sub_i32 s9, s7, s0
	s_cmp_ge_u32 s7, s0
	s_cselect_b32 s6, s17, s6
	s_cselect_b32 s7, s9, s7
	s_add_i32 s9, s6, 1
	s_cmp_ge_u32 s7, s0
	s_cselect_b32 s0, s9, s6
	s_xor_b32 s0, s0, s8
	s_sub_i32 s0, s0, s8
	s_mul_i32 s0, s0, s11
	s_abs_i32 s6, s0
	v_cvt_f32_u32_e32 v0, s6
	s_sub_i32 s7, 0, s6
	s_add_i32 s1, s0, s1
	s_xor_b32 s0, s1, s0
	v_rcp_iflag_f32_e32 v0, v0
	s_abs_i32 s1, s1
	s_ashr_i32 s0, s0, 31
	s_barrier
	v_mul_f32_e32 v0, 0x4f7ffffe, v0
	v_cvt_u32_f32_e32 v0, v0
	s_nop 0
	v_readfirstlane_b32 s8, v0
	s_mul_i32 s7, s7, s8
	s_mul_hi_u32 s7, s8, s7
	s_add_i32 s8, s8, s7
	s_mul_hi_u32 s7, s1, s8
	s_mul_i32 s8, s7, s6
	s_sub_i32 s1, s1, s8
	s_add_i32 s9, s7, 1
	s_sub_i32 s8, s1, s6
	s_cmp_ge_u32 s1, s6
	s_cselect_b32 s7, s9, s7
	s_cselect_b32 s1, s8, s1
	s_add_i32 s8, s7, 1
	s_cmp_ge_u32 s1, s6
	s_cselect_b32 s1, s8, s7
	s_xor_b32 s1, s1, s0
	s_sub_i32 s0, s1, s0
	s_max_i32 s8, s0, 4
	s_mul_i32 s0, s8, s16
	v_add_u32_e32 v0, s0, v98
	v_lshl_add_u32 v1, v0, 1, v0
	v_add_u32_e32 v0, 3, v1
	v_cmp_gt_u32_e32 vcc, s3, v1
	v_cmp_lt_u32_e64 s[0:1], s3, v0
	s_add_i32 s33, s3, -3
	v_mov_b32_e32 v2, s33
	s_and_b64 s[0:1], vcc, s[0:1]
	v_cndmask_b32_e64 v84, v1, v2, s[0:1]
	v_cmp_gt_u32_e64 s[0:1], s8, v98
	v_cmp_gt_u32_e64 s[6:7], s3, v84
	s_and_b64 s[0:1], s[0:1], s[6:7]
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_157
; %bb.105:
	s_cmp_lg_u32 s2, 0
	s_cselect_b64 s[6:7], -1, 0
	s_mul_i32 s11, s8, s11
	s_ashr_i32 s9, s30, 31
	s_mov_b32 s8, s30
	s_add_i32 s46, s2, -8
	s_add_i32 s47, s3, -1
	s_ashr_i32 s48, s10, 31
	s_ashr_i32 s17, s31, 31
	s_lshl_b64 s[8:9], s[8:9], 1
	s_mov_b32 s16, s31
	s_add_u32 s26, s12, s8
	s_addc_u32 s27, s13, s9
	s_lshl_b64 s[16:17], s[16:17], 1
	s_add_u32 s30, s14, s16
	s_addc_u32 s31, s15, s17
	s_add_u32 s34, s26, s8
	v_sub_u32_e64 v0, v0, s3 clamp
	s_addc_u32 s35, s27, s9
	v_cndmask_b32_e32 v95, 0, v0, vcc
	s_add_u32 s36, s30, s16
	v_cndmask_b32_e64 v0, 0, 1, s[6:7]
	v_cmp_eq_u32_e64 s[0:1], 63, v8
	v_cmp_eq_f32_e64 s[22:23], s29, 0
	v_cmp_neq_f32_e64 s[24:25], s29, 0
	s_addc_u32 s37, s31, s17
	s_lshl_b32 s49, s2, 1
	v_lshlrev_b32_e32 v96, 4, v8
	s_lshl_b32 s50, s2, 2
	s_mov_b64 s[38:39], 0
	v_cmp_ne_u32_e64 s[20:21], 1, v0
	v_mov_b32_e32 v87, 0
	s_branch .LBB0_107
.LBB0_106:                              ;   in Loop: Header=BB0_107 Depth=1
	s_or_b64 exec, exec, s[40:41]
	v_add_u32_e32 v0, s11, v84
	v_add_u32_e32 v1, 3, v0
	v_cmp_gt_u32_e32 vcc, s3, v0
	v_cmp_lt_u32_e64 s[6:7], s3, v1
	v_mov_b32_e32 v2, s33
	s_and_b64 vcc, vcc, s[6:7]
	v_cndmask_b32_e32 v84, v0, v2, vcc
	v_subrev_u32_e32 v1, s3, v1
	v_cmp_le_u32_e64 s[6:7], s3, v84
	s_or_b64 s[38:39], s[6:7], s[38:39]
	v_cndmask_b32_e32 v95, v95, v1, vcc
	s_andn2_b64 exec, exec, s[38:39]
	s_cbranch_execz .LBB0_157
.LBB0_107:                              ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_111 Depth 2
	s_and_b64 vcc, exec, s[20:21]
	v_mov_b32_e32 v23, v87
	v_mov_b32_e32 v22, v87
	v_mov_b32_e32 v21, v87
	v_mov_b32_e32 v20, v87
	v_mov_b32_e32 v3, v87
	v_mov_b32_e32 v2, v87
	v_mov_b32_e32 v1, v87
	v_mov_b32_e32 v0, v87
	v_mov_b32_e32 v7, v87
	v_mov_b32_e32 v6, v87
	v_mov_b32_e32 v5, v87
	v_mov_b32_e32 v4, v87
	v_mov_b32_e32 v11, v87
	v_mov_b32_e32 v10, v87
	v_mov_b32_e32 v9, v87
	v_mov_b32_e32 v8, v87
	v_mov_b32_e32 v15, v87
	v_mov_b32_e32 v14, v87
	v_mov_b32_e32 v13, v87
	v_mov_b32_e32 v12, v87
	v_mov_b32_e32 v19, v87
	v_mov_b32_e32 v18, v87
	v_mov_b32_e32 v17, v87
	v_mov_b32_e32 v16, v87
	v_mov_b32_e32 v35, v87
	v_mov_b32_e32 v34, v87
	v_mov_b32_e32 v33, v87
	v_mov_b32_e32 v32, v87
	v_mov_b32_e32 v31, v87
	v_mov_b32_e32 v30, v87
	v_mov_b32_e32 v29, v87
	v_mov_b32_e32 v28, v87
	v_mov_b32_e32 v27, v87
	v_mov_b32_e32 v26, v87
	v_mov_b32_e32 v25, v87
	v_mov_b32_e32 v24, v87
	s_cbranch_vccnz .LBB0_114
; %bb.108:                              ;   in Loop: Header=BB0_107 Depth=1
	v_min_u32_e32 v3, s47, v84
	v_mad_u64_u32 v[0:1], s[6:7], v3, s10, 0
	v_mov_b32_e32 v2, v1
	v_mad_u64_u32 v[2:3], s[6:7], v3, s48, v[2:3]
	v_mov_b32_e32 v1, v2
	v_add_u32_e32 v2, 1, v84
	v_min_u32_e32 v5, s47, v2
	v_mad_u64_u32 v[2:3], s[6:7], v5, s10, 0
	v_mov_b32_e32 v4, v3
	v_mad_u64_u32 v[4:5], s[6:7], v5, s48, v[4:5]
	v_mov_b32_e32 v3, v4
	v_add_u32_e32 v4, 2, v84
	v_min_u32_e32 v7, s47, v4
	v_mad_u64_u32 v[4:5], s[6:7], v7, s10, 0
	v_mov_b32_e32 v6, v5
	v_mad_u64_u32 v[6:7], s[6:7], v7, s48, v[6:7]
	v_mov_b32_e32 v5, v6
	v_mov_b32_e32 v24, 0
	s_mov_b32 s16, 0
	v_lshlrev_b64 v[88:89], 1, v[0:1]
	v_lshlrev_b64 v[90:91], 1, v[2:3]
	v_lshlrev_b64 v[92:93], 1, v[4:5]
	v_mov_b32_e32 v85, v96
	v_mov_b32_e32 v25, v24
	v_mov_b32_e32 v26, v24
	v_mov_b32_e32 v27, v24
	v_mov_b32_e32 v28, v24
	v_mov_b32_e32 v29, v24
	v_mov_b32_e32 v30, v24
	v_mov_b32_e32 v31, v24
	v_mov_b32_e32 v32, v24
	v_mov_b32_e32 v33, v24
	v_mov_b32_e32 v34, v24
	v_mov_b32_e32 v35, v24
	v_mov_b32_e32 v16, v24
	v_mov_b32_e32 v17, v24
	v_mov_b32_e32 v18, v24
	v_mov_b32_e32 v19, v24
	v_mov_b32_e32 v12, v24
	v_mov_b32_e32 v13, v24
	v_mov_b32_e32 v14, v24
	v_mov_b32_e32 v15, v24
	v_mov_b32_e32 v8, v24
	v_mov_b32_e32 v9, v24
	v_mov_b32_e32 v10, v24
	v_mov_b32_e32 v11, v24
	v_mov_b32_e32 v4, v24
	v_mov_b32_e32 v5, v24
	v_mov_b32_e32 v6, v24
	v_mov_b32_e32 v7, v24
	v_mov_b32_e32 v0, v24
	v_mov_b32_e32 v1, v24
	v_mov_b32_e32 v2, v24
	v_mov_b32_e32 v3, v24
	v_mov_b32_e32 v20, v24
	v_mov_b32_e32 v21, v24
	v_mov_b32_e32 v22, v24
	v_mov_b32_e32 v23, v24
	s_branch .LBB0_111
.LBB0_109:                              ;   in Loop: Header=BB0_111 Depth=2
	s_or_b64 exec, exec, s[8:9]
.LBB0_110:                              ;   in Loop: Header=BB0_111 Depth=2
	s_or_b64 exec, exec, s[6:7]
	s_waitcnt vmcnt(5) lgkmcnt(2)
	v_mfma_f32_4x4x4_16b_bf16 v[32:35], v[80:81], v[56:57], v[32:35]
	s_addk_i32 s16, 0x400
	s_waitcnt vmcnt(4)
	v_mfma_f32_4x4x4_16b_bf16 v[28:31], v[80:81], v[52:53], v[28:31]
	v_add_u32_e32 v85, 0x800, v85
	s_waitcnt vmcnt(3)
	v_mfma_f32_4x4x4_16b_bf16 v[24:27], v[80:81], v[60:61], v[24:27]
	s_cmp_ge_u32 s16, s2
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_4x4x4_16b_bf16 v[16:19], v[76:77], v[56:57], v[16:19]
	v_mfma_f32_4x4x4_16b_bf16 v[12:15], v[76:77], v[52:53], v[12:15]
	v_mfma_f32_4x4x4_16b_bf16 v[8:11], v[76:77], v[60:61], v[8:11]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_4x4x4_16b_bf16 v[4:7], v[72:73], v[56:57], v[4:7]
	v_mfma_f32_4x4x4_16b_bf16 v[0:3], v[72:73], v[52:53], v[0:3]
	v_mfma_f32_4x4x4_16b_bf16 v[20:23], v[72:73], v[60:61], v[20:23]
	v_mfma_f32_4x4x4_16b_bf16 v[32:35], v[82:83], v[58:59], v[32:35]
	v_mfma_f32_4x4x4_16b_bf16 v[28:31], v[82:83], v[54:55], v[28:31]
	v_mfma_f32_4x4x4_16b_bf16 v[24:27], v[82:83], v[62:63], v[24:27]
	v_mfma_f32_4x4x4_16b_bf16 v[16:19], v[78:79], v[58:59], v[16:19]
	v_mfma_f32_4x4x4_16b_bf16 v[12:15], v[78:79], v[54:55], v[12:15]
	v_mfma_f32_4x4x4_16b_bf16 v[8:11], v[78:79], v[62:63], v[8:11]
	v_mfma_f32_4x4x4_16b_bf16 v[4:7], v[74:75], v[58:59], v[4:7]
	v_mfma_f32_4x4x4_16b_bf16 v[0:3], v[74:75], v[54:55], v[0:3]
	v_mfma_f32_4x4x4_16b_bf16 v[20:23], v[74:75], v[62:63], v[20:23]
	s_waitcnt vmcnt(2)
	v_mfma_f32_4x4x4_16b_bf16 v[32:35], v[68:69], v[44:45], v[32:35]
	s_waitcnt vmcnt(1)
	v_mfma_f32_4x4x4_16b_bf16 v[28:31], v[68:69], v[48:49], v[28:31]
	s_waitcnt vmcnt(0)
	v_mfma_f32_4x4x4_16b_bf16 v[24:27], v[68:69], v[36:37], v[24:27]
	v_mfma_f32_4x4x4_16b_bf16 v[16:19], v[64:65], v[44:45], v[16:19]
	v_mfma_f32_4x4x4_16b_bf16 v[12:15], v[64:65], v[48:49], v[12:15]
	v_mfma_f32_4x4x4_16b_bf16 v[8:11], v[64:65], v[36:37], v[8:11]
	v_mfma_f32_4x4x4_16b_bf16 v[4:7], v[40:41], v[44:45], v[4:7]
	v_mfma_f32_4x4x4_16b_bf16 v[0:3], v[40:41], v[48:49], v[0:3]
	v_mfma_f32_4x4x4_16b_bf16 v[20:23], v[40:41], v[36:37], v[20:23]
	v_mfma_f32_4x4x4_16b_bf16 v[32:35], v[70:71], v[46:47], v[32:35]
	v_mfma_f32_4x4x4_16b_bf16 v[28:31], v[70:71], v[50:51], v[28:31]
	v_mfma_f32_4x4x4_16b_bf16 v[24:27], v[70:71], v[38:39], v[24:27]
	v_mfma_f32_4x4x4_16b_bf16 v[16:19], v[66:67], v[46:47], v[16:19]
	v_mfma_f32_4x4x4_16b_bf16 v[12:15], v[66:67], v[50:51], v[12:15]
	v_mfma_f32_4x4x4_16b_bf16 v[8:11], v[66:67], v[38:39], v[8:11]
	v_mfma_f32_4x4x4_16b_bf16 v[4:7], v[42:43], v[46:47], v[4:7]
	v_mfma_f32_4x4x4_16b_bf16 v[0:3], v[42:43], v[50:51], v[0:3]
	v_mfma_f32_4x4x4_16b_bf16 v[20:23], v[42:43], v[38:39], v[20:23]
	s_cbranch_scc1 .LBB0_114
.LBB0_111:                              ;   Parent Loop BB0_107 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_add_u32_e32 v42, s16, v94
	v_min_u32_e32 v86, s46, v42
	v_lshl_add_u64 v[36:37], v[86:87], 1, s[4:5]
	v_add_u32_e32 v97, 0x200, v42
	v_lshl_add_u64 v[38:39], v[36:37], 0, v[88:89]
	v_min_u32_e32 v86, s46, v97
	v_lshl_add_u64 v[40:41], v[36:37], 0, v[90:91]
	global_load_dwordx4 v[56:59], v[38:39], off nt
	global_load_dwordx4 v[52:55], v[40:41], off nt
	v_lshl_add_u64 v[36:37], v[36:37], 0, v[92:93]
	v_lshl_add_u64 v[38:39], v[86:87], 1, s[4:5]
	v_lshl_add_u64 v[40:41], v[38:39], 0, v[88:89]
	global_load_dwordx4 v[60:63], v[36:37], off nt
	global_load_dwordx4 v[44:47], v[40:41], off nt
	v_lshl_add_u64 v[36:37], v[38:39], 0, v[90:91]
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[92:93]
	global_load_dwordx4 v[48:51], v[36:37], off nt
	s_nop 0
	global_load_dwordx4 v[36:39], v[38:39], off nt
	v_cmp_gt_u32_e32 vcc, s2, v42
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v75, 0
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v73, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v79, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v81, 0
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_110
; %bb.112:                              ;   in Loop: Header=BB0_111 Depth=2
	v_add_u32_e32 v98, s49, v85
	ds_read_b128 v[80:83], v85
	v_add_u32_e32 v86, s50, v85
	ds_read_b128 v[76:79], v98
	ds_read_b128 v[72:75], v86
	v_cmp_gt_u32_e32 vcc, s2, v97
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v42, 0
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_109
; %bb.113:                              ;   in Loop: Header=BB0_111 Depth=2
	ds_read_b128 v[68:71], v85 offset:1024
	ds_read_b128 v[64:67], v98 offset:1024
	ds_read_b128 v[40:43], v86 offset:1024
	s_branch .LBB0_109
.LBB0_114:                              ;   in Loop: Header=BB0_107 Depth=1
	; sched_barrier mask(0x00000000)
	s_nop 2
	v_add_f32_dpp v0, v1, v0 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v32, v33, v32 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v28, v29, v28 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v24, v25, v24 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v16, v17, v16 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v12, v13, v12 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v9, v8 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v5, v4 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v2, v0 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v21, v20 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v32, v34, v32 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v28, v30, v28 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v24, v26, v24 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v16, v18, v16 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v12, v14, v12 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v10, v8 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v6, v4 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v22, v2 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v32, v35, v32 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v28, v31, v28 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v24, v27, v24 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v16, v19, v16 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v12, v15, v12 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v11, v8 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v7, v4 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v3, v0 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v23, v2 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v32, v32, v32 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v28, v28, v28 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v24, v24, v24 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v16, v16, v16 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v12, v12, v12 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v8, v8 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v0, v0 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v32, v32, v32 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v28, v28, v28 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v24, v24, v24 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v16, v16, v16 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v12, v12, v12 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v8, v8 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v0, v0 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v32, v32 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v28, v28 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v24, v24 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v16, v16 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v12, v12 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v8, v8 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v4, v4 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v0, v0 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v2, v2 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v32, v32, v32 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v28, v28, v28 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v24, v24, v24 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v16, v16, v16 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v12, v12, v12 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v8, v8 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v11, v4, v4 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v0, v0 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v15, v2, v2 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v33, v32 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v29, v28 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v25, v24 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v17, v16 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v13, v12 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v10, v8 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v14, v11 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v1, v0 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v18, v15 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_and_saveexec_b64 s[40:41], s[0:1]
	s_cbranch_execz .LBB0_106
; %bb.115:                              ;   in Loop: Header=BB0_107 Depth=1
	v_cmp_eq_u32_e64 s[8:9], 0, v95
	s_and_b64 s[6:7], s[22:23], s[8:9]
	v_add_f32_e32 v9, v32, v33
	v_add_f32_e32 v7, v28, v29
	v_add_f32_e32 v6, v24, v25
	v_add_f32_e32 v5, v16, v17
	v_add_f32_e32 v4, v12, v13
	v_add_f32_e32 v3, v8, v10
	v_add_f32_e32 v2, v11, v14
	v_add_f32_e32 v1, v0, v1
	v_add_f32_e32 v0, v15, v18
	s_xor_b64 s[16:17], s[6:7], -1
	s_mov_b64 s[6:7], 0
                                        ; implicit-def: $vgpr8
	s_and_saveexec_b64 s[18:19], s[16:17]
	s_xor_b64 s[42:43], exec, s[18:19]
	s_cbranch_execnz .LBB0_118
; %bb.116:                              ;   in Loop: Header=BB0_107 Depth=1
	s_andn2_saveexec_b64 s[8:9], s[42:43]
	s_cbranch_execnz .LBB0_155
.LBB0_117:                              ;   in Loop: Header=BB0_107 Depth=1
	s_or_b64 exec, exec, s[8:9]
	s_and_b64 exec, exec, s[6:7]
	s_cbranch_execz .LBB0_106
	s_branch .LBB0_156
.LBB0_118:                              ;   in Loop: Header=BB0_107 Depth=1
	v_cndmask_b32_e64 v8, 0, 1, s[24:25]
	v_cmp_ne_u32_e64 s[6:7], 1, v8
	s_and_saveexec_b64 s[16:17], s[8:9]
	s_cbranch_execnz .LBB0_126
; %bb.119:                              ;   in Loop: Header=BB0_107 Depth=1
	s_or_b64 exec, exec, s[16:17]
	v_cmp_gt_u32_e64 s[16:17], 2, v95
	s_and_saveexec_b64 s[18:19], s[16:17]
	s_cbranch_execnz .LBB0_129
.LBB0_120:                              ;   in Loop: Header=BB0_107 Depth=1
	s_or_b64 exec, exec, s[18:19]
	v_cmp_gt_u32_e64 s[18:19], 3, v95
	s_and_saveexec_b64 s[44:45], s[18:19]
	s_cbranch_execnz .LBB0_132
.LBB0_121:                              ;   in Loop: Header=BB0_107 Depth=1
	s_or_b64 exec, exec, s[44:45]
	s_and_saveexec_b64 s[44:45], s[8:9]
	s_cbranch_execnz .LBB0_135
.LBB0_122:                              ;   in Loop: Header=BB0_107 Depth=1
	s_or_b64 exec, exec, s[44:45]
	s_and_saveexec_b64 s[44:45], s[16:17]
	s_cbranch_execnz .LBB0_138
.LBB0_123:                              ;   in Loop: Header=BB0_107 Depth=1
	s_or_b64 exec, exec, s[44:45]
	s_and_saveexec_b64 s[44:45], s[18:19]
	s_cbranch_execnz .LBB0_141
.LBB0_124:                              ;   in Loop: Header=BB0_107 Depth=1
	s_or_b64 exec, exec, s[44:45]
	s_and_saveexec_b64 s[44:45], s[8:9]
	s_cbranch_execnz .LBB0_144
.LBB0_125:                              ;   in Loop: Header=BB0_107 Depth=1
	s_or_b64 exec, exec, s[44:45]
	s_and_saveexec_b64 s[8:9], s[16:17]
	s_cbranch_execnz .LBB0_147
	s_branch .LBB0_150
.LBB0_126:                              ;   in Loop: Header=BB0_107 Depth=1
	v_mul_f32_e32 v8, s28, v9
	s_and_b64 vcc, exec, s[6:7]
	v_mov_b32_e32 v85, v87
	s_cbranch_vccnz .LBB0_128
; %bb.127:                              ;   in Loop: Header=BB0_107 Depth=1
	v_lshl_add_u64 v[10:11], v[84:85], 1, s[12:13]
	global_load_ushort v9, v[10:11], off
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v9, 16, v9
	v_fmac_f32_e32 v8, s29, v9
.LBB0_128:                              ;   in Loop: Header=BB0_107 Depth=1
	v_cvt_pk_bf16_f32 v10, v8, s0
	v_lshl_add_u64 v[8:9], v[84:85], 1, s[14:15]
	global_store_short v[8:9], v10, off
	s_or_b64 exec, exec, s[16:17]
	v_cmp_gt_u32_e64 s[16:17], 2, v95
	s_and_saveexec_b64 s[18:19], s[16:17]
	s_cbranch_execz .LBB0_120
.LBB0_129:                              ;   in Loop: Header=BB0_107 Depth=1
	v_mul_f32_e32 v7, s28, v7
	s_and_b64 vcc, exec, s[6:7]
	v_add_u32_e32 v86, 1, v84
	s_cbranch_vccnz .LBB0_131
; %bb.130:                              ;   in Loop: Header=BB0_107 Depth=1
	v_lshl_add_u64 v[8:9], v[86:87], 1, s[12:13]
	global_load_ushort v8, v[8:9], off
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v8, 16, v8
	v_fmac_f32_e32 v7, s29, v8
.LBB0_131:                              ;   in Loop: Header=BB0_107 Depth=1
	v_cvt_pk_bf16_f32 v7, v7, s0
	v_lshl_add_u64 v[8:9], v[86:87], 1, s[14:15]
	global_store_short v[8:9], v7, off
	s_or_b64 exec, exec, s[18:19]
	v_cmp_gt_u32_e64 s[18:19], 3, v95
	s_and_saveexec_b64 s[44:45], s[18:19]
	s_cbranch_execz .LBB0_121
.LBB0_132:                              ;   in Loop: Header=BB0_107 Depth=1
	v_mul_f32_e32 v6, s28, v6
	s_and_b64 vcc, exec, s[6:7]
	v_add_u32_e32 v86, 2, v84
	s_cbranch_vccnz .LBB0_134
; %bb.133:                              ;   in Loop: Header=BB0_107 Depth=1
	v_lshl_add_u64 v[8:9], v[86:87], 1, s[12:13]
	global_load_ushort v7, v[8:9], off
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v7, 16, v7
	v_fmac_f32_e32 v6, s29, v7
.LBB0_134:                              ;   in Loop: Header=BB0_107 Depth=1
	v_cvt_pk_bf16_f32 v8, v6, s0
	v_lshl_add_u64 v[6:7], v[86:87], 1, s[14:15]
	global_store_short v[6:7], v8, off
	s_or_b64 exec, exec, s[44:45]
	s_and_saveexec_b64 s[44:45], s[8:9]
	s_cbranch_execz .LBB0_122
.LBB0_135:                              ;   in Loop: Header=BB0_107 Depth=1
	v_mul_f32_e32 v5, s28, v5
	s_and_b64 vcc, exec, s[6:7]
	v_mov_b32_e32 v85, v87
	s_cbranch_vccnz .LBB0_137
; %bb.136:                              ;   in Loop: Header=BB0_107 Depth=1
	v_lshl_add_u64 v[6:7], v[84:85], 1, s[26:27]
	global_load_ushort v6, v[6:7], off
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v6, 16, v6
	v_fmac_f32_e32 v5, s29, v6
.LBB0_137:                              ;   in Loop: Header=BB0_107 Depth=1
	v_cvt_pk_bf16_f32 v5, v5, s0
	v_lshl_add_u64 v[6:7], v[84:85], 1, s[30:31]
	global_store_short v[6:7], v5, off
	s_or_b64 exec, exec, s[44:45]
	s_and_saveexec_b64 s[44:45], s[16:17]
	s_cbranch_execz .LBB0_123
.LBB0_138:                              ;   in Loop: Header=BB0_107 Depth=1
	v_mul_f32_e32 v4, s28, v4
	s_and_b64 vcc, exec, s[6:7]
	v_add_u32_e32 v86, 1, v84
	s_cbranch_vccnz .LBB0_140
; %bb.139:                              ;   in Loop: Header=BB0_107 Depth=1
	v_lshl_add_u64 v[6:7], v[86:87], 1, s[26:27]
	global_load_ushort v5, v[6:7], off
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v5, 16, v5
	v_fmac_f32_e32 v4, s29, v5
.LBB0_140:                              ;   in Loop: Header=BB0_107 Depth=1
	v_cvt_pk_bf16_f32 v6, v4, s0
	v_lshl_add_u64 v[4:5], v[86:87], 1, s[30:31]
	global_store_short v[4:5], v6, off
	s_or_b64 exec, exec, s[44:45]
	s_and_saveexec_b64 s[44:45], s[18:19]
	s_cbranch_execz .LBB0_124
.LBB0_141:                              ;   in Loop: Header=BB0_107 Depth=1
	v_mul_f32_e32 v3, s28, v3
	s_and_b64 vcc, exec, s[6:7]
	v_add_u32_e32 v86, 2, v84
	s_cbranch_vccnz .LBB0_143
; %bb.142:                              ;   in Loop: Header=BB0_107 Depth=1
	v_lshl_add_u64 v[4:5], v[86:87], 1, s[26:27]
	global_load_ushort v4, v[4:5], off
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v4, 16, v4
	v_fmac_f32_e32 v3, s29, v4
.LBB0_143:                              ;   in Loop: Header=BB0_107 Depth=1
	v_cvt_pk_bf16_f32 v3, v3, s0
	v_lshl_add_u64 v[4:5], v[86:87], 1, s[30:31]
	global_store_short v[4:5], v3, off
	s_or_b64 exec, exec, s[44:45]
	s_and_saveexec_b64 s[44:45], s[8:9]
	s_cbranch_execz .LBB0_125
.LBB0_144:                              ;   in Loop: Header=BB0_107 Depth=1
	v_mul_f32_e32 v2, s28, v2
	s_and_b64 vcc, exec, s[6:7]
	v_mov_b32_e32 v85, v87
	s_cbranch_vccnz .LBB0_146
; %bb.145:                              ;   in Loop: Header=BB0_107 Depth=1
	v_lshl_add_u64 v[4:5], v[84:85], 1, s[34:35]
	global_load_ushort v3, v[4:5], off
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v3, 16, v3
	v_fmac_f32_e32 v2, s29, v3
.LBB0_146:                              ;   in Loop: Header=BB0_107 Depth=1
	v_cvt_pk_bf16_f32 v4, v2, s0
	v_lshl_add_u64 v[2:3], v[84:85], 1, s[36:37]
	global_store_short v[2:3], v4, off
	s_or_b64 exec, exec, s[44:45]
	s_and_saveexec_b64 s[8:9], s[16:17]
	s_cbranch_execz .LBB0_150
.LBB0_147:                              ;   in Loop: Header=BB0_107 Depth=1
	v_mul_f32_e32 v1, s28, v1
	s_and_b64 vcc, exec, s[6:7]
	v_add_u32_e32 v86, 1, v84
	s_cbranch_vccnz .LBB0_149
; %bb.148:                              ;   in Loop: Header=BB0_107 Depth=1
	v_lshl_add_u64 v[2:3], v[86:87], 1, s[34:35]
	global_load_ushort v2, v[2:3], off
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v2, 16, v2
	v_fmac_f32_e32 v1, s29, v2
.LBB0_149:                              ;   in Loop: Header=BB0_107 Depth=1
	v_cvt_pk_bf16_f32 v1, v1, s0
	v_lshl_add_u64 v[2:3], v[86:87], 1, s[36:37]
	global_store_short v[2:3], v1, off
.LBB0_150:                              ;   in Loop: Header=BB0_107 Depth=1
	s_or_b64 exec, exec, s[8:9]
	s_mov_b64 s[16:17], 0
                                        ; implicit-def: $vgpr8
	s_and_saveexec_b64 s[8:9], s[18:19]
	s_xor_b64 s[8:9], exec, s[8:9]
	s_cbranch_execz .LBB0_154
; %bb.151:                              ;   in Loop: Header=BB0_107 Depth=1
	v_mul_f32_e32 v8, s28, v0
	s_and_b64 vcc, exec, s[6:7]
	v_add_u32_e32 v86, 2, v84
	s_cbranch_vccnz .LBB0_153
; %bb.152:                              ;   in Loop: Header=BB0_107 Depth=1
	v_lshl_add_u64 v[0:1], v[86:87], 1, s[34:35]
	global_load_ushort v0, v[0:1], off
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v0, 16, v0
	v_fmac_f32_e32 v8, s29, v0
.LBB0_153:                              ;   in Loop: Header=BB0_107 Depth=1
	s_mov_b64 s[16:17], exec
.LBB0_154:                              ;   in Loop: Header=BB0_107 Depth=1
	s_or_b64 exec, exec, s[8:9]
	s_and_b64 s[6:7], s[16:17], exec
                                        ; implicit-def: $vgpr9
                                        ; implicit-def: $vgpr7
                                        ; implicit-def: $vgpr6
                                        ; implicit-def: $vgpr5
                                        ; implicit-def: $vgpr4
                                        ; implicit-def: $vgpr3
                                        ; implicit-def: $vgpr2
                                        ; implicit-def: $vgpr1
                                        ; implicit-def: $vgpr0
	s_andn2_saveexec_b64 s[8:9], s[42:43]
	s_cbranch_execz .LBB0_117
.LBB0_155:                              ;   in Loop: Header=BB0_107 Depth=1
	v_mul_f32_e32 v8, s28, v9
	v_mov_b32_e32 v85, v87
	v_cvt_pk_bf16_f32 v12, v8, s0
	v_lshlrev_b64 v[8:9], 1, v[84:85]
	v_lshl_add_u64 v[10:11], s[14:15], 0, v[8:9]
	v_add_u32_e32 v86, 1, v84
	global_store_short v[10:11], v12, off
	v_mul_f32_e32 v7, s28, v7
	v_lshlrev_b64 v[10:11], 1, v[86:87]
	v_cvt_pk_bf16_f32 v7, v7, s0
	v_lshl_add_u64 v[12:13], s[14:15], 0, v[10:11]
	v_mul_f32_e32 v6, s28, v6
	v_add_u32_e32 v86, 2, v84
	global_store_short v[12:13], v7, off
	v_cvt_pk_bf16_f32 v14, v6, s0
	v_lshlrev_b64 v[6:7], 1, v[86:87]
	v_lshl_add_u64 v[12:13], s[14:15], 0, v[6:7]
	v_mul_f32_e32 v5, s28, v5
	global_store_short v[12:13], v14, off
	v_cvt_pk_bf16_f32 v5, v5, s0
	v_lshl_add_u64 v[12:13], s[30:31], 0, v[8:9]
	v_mul_f32_e32 v4, s28, v4
	global_store_short v[12:13], v5, off
	v_cvt_pk_bf16_f32 v12, v4, s0
	v_lshl_add_u64 v[4:5], s[30:31], 0, v[10:11]
	v_mul_f32_e32 v3, s28, v3
	global_store_short v[4:5], v12, off
	v_cvt_pk_bf16_f32 v3, v3, s0
	v_lshl_add_u64 v[4:5], s[30:31], 0, v[6:7]
	v_mul_f32_e32 v2, s28, v2
	global_store_short v[4:5], v3, off
	v_cvt_pk_bf16_f32 v4, v2, s0
	v_lshl_add_u64 v[2:3], s[36:37], 0, v[8:9]
	v_mul_f32_e32 v1, s28, v1
	global_store_short v[2:3], v4, off
	v_cvt_pk_bf16_f32 v1, v1, s0
	v_lshl_add_u64 v[2:3], s[36:37], 0, v[10:11]
	global_store_short v[2:3], v1, off
	v_mul_f32_e32 v8, s28, v0
	s_or_b64 s[6:7], s[6:7], exec
	s_or_b64 exec, exec, s[8:9]
	s_and_b64 exec, exec, s[6:7]
	s_cbranch_execz .LBB0_106
.LBB0_156:                              ;   in Loop: Header=BB0_107 Depth=1
	v_cvt_pk_bf16_f32 v2, v8, s0
	v_lshl_add_u64 v[0:1], v[86:87], 1, s[36:37]
	global_store_short v[0:1], v2, off
	s_branch .LBB0_106
.LBB0_157:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wvSplitK_vllm_bf16_tn_n3_y3u2
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
		.amdhsa_next_free_vgpr 103
		.amdhsa_next_free_sgpr 96
		.amdhsa_accum_offset 104
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
	.size	wvSplitK_vllm_bf16_tn_n3_y3u2, .Lfunc_end0-wvSplitK_vllm_bf16_tn_n3_y3u2
                                        ; -- End function
	.set wvSplitK_vllm_bf16_tn_n3_y3u2.num_vgpr, 103
	.set wvSplitK_vllm_bf16_tn_n3_y3u2.num_agpr, 0
	.set wvSplitK_vllm_bf16_tn_n3_y3u2.numbered_sgpr, 73
	.set wvSplitK_vllm_bf16_tn_n3_y3u2.num_named_barrier, 0
	.set wvSplitK_vllm_bf16_tn_n3_y3u2.private_seg_size, 0
	.set wvSplitK_vllm_bf16_tn_n3_y3u2.uses_vcc, 1
	.set wvSplitK_vllm_bf16_tn_n3_y3u2.uses_flat_scratch, 0
	.set wvSplitK_vllm_bf16_tn_n3_y3u2.has_dyn_sized_stack, 0
	.set wvSplitK_vllm_bf16_tn_n3_y3u2.has_recursion, 0
	.set wvSplitK_vllm_bf16_tn_n3_y3u2.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 8364
; TotalNumSgprs: 79
; NumVgprs: 103
; NumAgprs: 0
; TotalNumVgprs: 103
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 163840 bytes/workgroup (compile time only)
; SGPRBlocks: 12
; VGPRBlocks: 12
; NumSGPRsForWavesPerEU: 102
; NumVGPRsForWavesPerEU: 103
; AccumOffset: 104
; Occupancy: 4
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 1
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 25
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_a19e57bfdfabeab4,@object ; @__hip_cuid_a19e57bfdfabeab4
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_a19e57bfdfabeab4
__hip_cuid_a19e57bfdfabeab4:
	.byte	0                               ; 0x0
	.size	__hip_cuid_a19e57bfdfabeab4, 1

	.ident	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.4 26084 f58b06dce1f9c15707c5f808fd002e18c2accf7e)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_a19e57bfdfabeab4
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
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 16, 1]
  EnableMatrixInstruction: True
  MIWaveTile: [1, 1]
  AssertSummationElementMultiple: 8
  AssertSizeEqual: { 1: 3, 2: 1 }
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
    .name:           wvSplitK_vllm_bf16_tn_n3_y3u2
    .private_segment_fixed_size: 0
    .sgpr_count:     79
    .sgpr_spill_count: 0
    .symbol:         wvSplitK_vllm_bf16_tn_n3_y3u2.kd
    .vgpr_count:     103
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 1
...

	.end_amdgpu_metadata
