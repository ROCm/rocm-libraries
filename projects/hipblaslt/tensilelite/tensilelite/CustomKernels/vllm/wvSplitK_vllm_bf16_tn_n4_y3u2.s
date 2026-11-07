	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 4
	.text
	.protected	wvSplitK_vllm_bf16_tn_n4_y3u2 ; -- Begin function wvSplitK_vllm_bf16_tn_n4_y3u2
	.globl	wvSplitK_vllm_bf16_tn_n4_y3u2
	.p2align	8
	.type	wvSplitK_vllm_bf16_tn_n4_y3u2,@function
wvSplitK_vllm_bf16_tn_n4_y3u2:          ; @wvSplitK_vllm_bf16_tn_n4_y3u2
; %bb.188:
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.189:
.LBB0_0:
	s_load_dwordx4 s[28:31], s[0:1], 0x38
	s_mov_b32 s26, s11
	s_mov_b64 s[0:1], -1
	s_cmpk_gt_i32 s2, 0x5000
	v_bfe_u32 v118, v0, 10, 10
	s_mul_i32 s11, s8, 3
	s_mul_i32 s17, s8, 48
	s_cbranch_scc0 .LBB0_103
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
	v_cmp_gt_u32_e32 vcc, s8, v118
	s_and_saveexec_b64 s[34:35], vcc
	s_cbranch_execz .LBB0_102
; %bb.2:
	s_mul_i32 s9, s8, 3
	v_cvt_f32_u32_e32 v3, s9
	s_mul_i32 s0, s8, s16
	v_add_u32_e32 v1, s0, v118
	v_lshl_add_u32 v2, v1, 1, v1
	v_rcp_iflag_f32_e32 v3, v3
	v_add_u32_e32 v1, 3, v2
	v_cmp_gt_u32_e32 vcc, s3, v2
	v_cmp_lt_u32_e64 s[0:1], s3, v1
	s_add_i32 s70, s3, -3
	v_mov_b32_e32 v4, s70
	s_and_b64 s[0:1], vcc, s[0:1]
	v_cndmask_b32_e64 v108, v2, v4, s[0:1]
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
	s_cselect_b32 s71, s3, s1
	v_cmp_gt_u32_e64 s[0:1], s71, v108
	s_and_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB0_102
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
	s_lshl_b32 s72, s8, 9
	s_add_i32 s73, s2, -8
	s_ashr_i32 s74, s10, 31
	s_ashr_i32 s21, s31, 31
	s_lshl_b64 s[18:19], s[18:19], 1
	s_mov_b32 s20, s31
	s_add_u32 s46, s12, s18
	s_addc_u32 s47, s13, s19
	s_lshl_b64 s[20:21], s[20:21], 1
	s_add_u32 s48, s14, s20
	s_addc_u32 s49, s15, s21
	s_add_u32 s50, s46, s18
	s_addc_u32 s51, s47, s19
	s_add_u32 s52, s48, s20
	s_addc_u32 s53, s49, s21
	s_add_u32 s54, s50, s18
	v_and_b32_e32 v2, 0x3ff, v0
	s_addc_u32 s55, s51, s19
	v_sub_u32_e64 v1, v1, s3 clamp
	v_lshlrev_b32_e32 v119, 3, v2
	v_cmp_eq_u32_e64 s[0:1], 63, v2
	s_add_u32 s56, s52, s20
	v_lshlrev_b32_e32 v2, 4, v2
	s_mov_b64 s[38:39], 0
	v_mov_b32_e32 v3, 0
	v_cndmask_b32_e32 v1, 0, v1, vcc
	v_lshl_add_u32 v120, v118, 9, v119
	s_mul_i32 s75, s8, s11
	v_cmp_eq_f32_e64 s[42:43], s29, 0
	v_cmp_neq_f32_e64 s[44:45], s29, 0
	s_addc_u32 s57, s53, s21
	s_lshl_b64 s[58:59], s[26:27], 1
	v_lshl_add_u32 v121, v118, 10, v2
	s_lshl_b32 s76, s8, 10
	v_cndmask_b32_e64 v122, 0, 1, s[40:41]
	s_branch .LBB0_6
.LBB0_4:                                ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[62:63]
	v_add_u32_e32 v4, 3, v2
	v_cmp_gt_u32_e32 vcc, s3, v2
	v_cmp_lt_u32_e64 s[8:9], s3, v4
	v_subrev_u32_e32 v4, s3, v4
	v_mov_b32_e32 v5, s70
	s_and_b64 vcc, vcc, s[8:9]
	v_cndmask_b32_e32 v2, v2, v5, vcc
	v_cndmask_b32_e32 v1, v1, v4, vcc
.LBB0_5:                                ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[60:61]
	v_cmp_le_u32_e32 vcc, s71, v2
	s_or_b64 s[38:39], vcc, s[38:39]
	v_mov_b32_e32 v108, v2
	s_andn2_b64 exec, exec, s[38:39]
	s_cbranch_execz .LBB0_102
.LBB0_6:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_10 Depth 2
                                        ;       Child Loop BB0_15 Depth 3
	v_min_u32_e32 v4, s33, v108
	v_mad_u64_u32 v[6:7], s[20:21], v4, s10, 0
	v_mov_b32_e32 v2, v7
	v_mad_u64_u32 v[4:5], s[20:21], v4, s74, v[2:3]
	v_add_u32_e32 v2, 1, v108
	v_mov_b32_e32 v7, v4
	v_min_u32_e32 v4, s33, v2
	v_mad_u64_u32 v[8:9], s[20:21], v4, s10, 0
	v_mov_b32_e32 v2, v9
	v_mad_u64_u32 v[4:5], s[20:21], v4, s74, v[2:3]
	v_add_u32_e32 v110, 2, v108
	v_mov_b32_e32 v9, v4
	v_min_u32_e32 v4, s33, v110
	v_mad_u64_u32 v[10:11], s[20:21], v4, s10, 0
	v_mov_b32_e32 v2, v11
	v_mad_u64_u32 v[4:5], s[20:21], v4, s74, v[2:3]
	v_mov_b32_e32 v11, v4
	v_mov_b32_e32 v4, v3
	v_mov_b32_e32 v5, v3
	v_mov_b32_e32 v2, v3
	v_lshlrev_b64 v[114:115], 1, v[8:9]
	v_lshlrev_b64 v[116:117], 1, v[10:11]
	v_mov_b64_e32 v[30:31], v[4:5]
	v_mov_b64_e32 v[34:35], v[4:5]
	v_mov_b64_e32 v[38:39], v[4:5]
	v_mov_b64_e32 v[42:43], v[4:5]
	v_mov_b64_e32 v[46:47], v[4:5]
	v_mov_b64_e32 v[50:51], v[4:5]
	v_mov_b64_e32 v[26:27], v[4:5]
	v_mov_b64_e32 v[22:23], v[4:5]
	v_mov_b64_e32 v[18:19], v[4:5]
	v_mov_b64_e32 v[14:15], v[4:5]
	v_mov_b64_e32 v[10:11], v[4:5]
	v_lshlrev_b64 v[112:113], 1, v[6:7]
	v_mov_b64_e32 v[28:29], v[2:3]
	v_mov_b64_e32 v[32:33], v[2:3]
	v_mov_b64_e32 v[36:37], v[2:3]
	v_mov_b64_e32 v[40:41], v[2:3]
	v_mov_b64_e32 v[44:45], v[2:3]
	v_mov_b64_e32 v[48:49], v[2:3]
	v_mov_b64_e32 v[24:25], v[2:3]
	v_mov_b64_e32 v[20:21], v[2:3]
	v_mov_b64_e32 v[16:17], v[2:3]
	v_mov_b64_e32 v[12:13], v[2:3]
	v_mov_b64_e32 v[8:9], v[2:3]
	v_mov_b64_e32 v[6:7], v[4:5]
	v_cmp_gt_u32_e64 s[8:9], s3, v108
	v_cmp_le_u32_e64 s[18:19], s3, v108
	v_mov_b64_e32 v[4:5], v[2:3]
	s_mov_b32 s36, s37
	s_mov_b32 s66, s37
	s_branch .LBB0_10
.LBB0_7:                                ;   in Loop: Header=BB0_10 Depth=2
	s_or_b64 exec, exec, s[24:25]
.LBB0_8:                                ;   in Loop: Header=BB0_10 Depth=2
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(5) lgkmcnt(3)
	v_mfma_f32_4x4x4_16b_bf16 v[48:51], v[104:105], v[68:69], v[48:51]
	s_waitcnt vmcnt(4)
	v_mfma_f32_4x4x4_16b_bf16 v[44:47], v[104:105], v[64:65], v[44:47]
	s_waitcnt vmcnt(3)
	v_mfma_f32_4x4x4_16b_bf16 v[40:43], v[104:105], v[72:73], v[40:43]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_4x4x4_16b_bf16 v[36:39], v[100:101], v[68:69], v[36:39]
	v_mfma_f32_4x4x4_16b_bf16 v[32:35], v[100:101], v[64:65], v[32:35]
	v_mfma_f32_4x4x4_16b_bf16 v[28:31], v[100:101], v[72:73], v[28:31]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_4x4x4_16b_bf16 v[24:27], v[96:97], v[68:69], v[24:27]
	v_mfma_f32_4x4x4_16b_bf16 v[20:23], v[96:97], v[64:65], v[20:23]
	v_mfma_f32_4x4x4_16b_bf16 v[16:19], v[96:97], v[72:73], v[16:19]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_4x4x4_16b_bf16 v[12:15], v[80:81], v[68:69], v[12:15]
	v_mfma_f32_4x4x4_16b_bf16 v[8:11], v[80:81], v[64:65], v[8:11]
	v_mfma_f32_4x4x4_16b_bf16 v[4:7], v[80:81], v[72:73], v[4:7]
	v_mfma_f32_4x4x4_16b_bf16 v[48:51], v[106:107], v[70:71], v[48:51]
	v_mfma_f32_4x4x4_16b_bf16 v[44:47], v[106:107], v[66:67], v[44:47]
	v_mfma_f32_4x4x4_16b_bf16 v[40:43], v[106:107], v[74:75], v[40:43]
	v_mfma_f32_4x4x4_16b_bf16 v[36:39], v[102:103], v[70:71], v[36:39]
	v_mfma_f32_4x4x4_16b_bf16 v[32:35], v[102:103], v[66:67], v[32:35]
	v_mfma_f32_4x4x4_16b_bf16 v[28:31], v[102:103], v[74:75], v[28:31]
	v_mfma_f32_4x4x4_16b_bf16 v[24:27], v[98:99], v[70:71], v[24:27]
	v_mfma_f32_4x4x4_16b_bf16 v[20:23], v[98:99], v[66:67], v[20:23]
	v_mfma_f32_4x4x4_16b_bf16 v[16:19], v[98:99], v[74:75], v[16:19]
	v_mfma_f32_4x4x4_16b_bf16 v[12:15], v[82:83], v[70:71], v[12:15]
	v_mfma_f32_4x4x4_16b_bf16 v[8:11], v[82:83], v[66:67], v[8:11]
	v_mfma_f32_4x4x4_16b_bf16 v[4:7], v[82:83], v[74:75], v[4:7]
	s_waitcnt vmcnt(2)
	v_mfma_f32_4x4x4_16b_bf16 v[48:51], v[92:93], v[52:53], v[48:51]
	s_waitcnt vmcnt(1)
	v_mfma_f32_4x4x4_16b_bf16 v[44:47], v[92:93], v[60:61], v[44:47]
	s_waitcnt vmcnt(0)
	v_mfma_f32_4x4x4_16b_bf16 v[40:43], v[92:93], v[56:57], v[40:43]
	v_mfma_f32_4x4x4_16b_bf16 v[36:39], v[84:85], v[52:53], v[36:39]
	v_mfma_f32_4x4x4_16b_bf16 v[32:35], v[84:85], v[60:61], v[32:35]
	v_mfma_f32_4x4x4_16b_bf16 v[28:31], v[84:85], v[56:57], v[28:31]
	v_mfma_f32_4x4x4_16b_bf16 v[24:27], v[88:89], v[52:53], v[24:27]
	v_mfma_f32_4x4x4_16b_bf16 v[20:23], v[88:89], v[60:61], v[20:23]
	v_mfma_f32_4x4x4_16b_bf16 v[16:19], v[88:89], v[56:57], v[16:19]
	v_mfma_f32_4x4x4_16b_bf16 v[12:15], v[76:77], v[52:53], v[12:15]
	v_mfma_f32_4x4x4_16b_bf16 v[8:11], v[76:77], v[60:61], v[8:11]
	v_mfma_f32_4x4x4_16b_bf16 v[4:7], v[76:77], v[56:57], v[4:7]
	v_mfma_f32_4x4x4_16b_bf16 v[48:51], v[94:95], v[54:55], v[48:51]
	v_mfma_f32_4x4x4_16b_bf16 v[44:47], v[94:95], v[62:63], v[44:47]
	v_mfma_f32_4x4x4_16b_bf16 v[40:43], v[94:95], v[58:59], v[40:43]
	v_mfma_f32_4x4x4_16b_bf16 v[36:39], v[86:87], v[54:55], v[36:39]
	v_mfma_f32_4x4x4_16b_bf16 v[32:35], v[86:87], v[62:63], v[32:35]
	v_mfma_f32_4x4x4_16b_bf16 v[28:31], v[86:87], v[58:59], v[28:31]
	v_mfma_f32_4x4x4_16b_bf16 v[24:27], v[90:91], v[54:55], v[24:27]
	v_mfma_f32_4x4x4_16b_bf16 v[20:23], v[90:91], v[62:63], v[20:23]
	v_mfma_f32_4x4x4_16b_bf16 v[16:19], v[90:91], v[58:59], v[16:19]
	v_mfma_f32_4x4x4_16b_bf16 v[12:15], v[78:79], v[54:55], v[12:15]
	v_mfma_f32_4x4x4_16b_bf16 v[8:11], v[78:79], v[62:63], v[8:11]
	v_mfma_f32_4x4x4_16b_bf16 v[4:7], v[78:79], v[58:59], v[4:7]
.LBB0_9:                                ;   in Loop: Header=BB0_10 Depth=2
	s_or_b64 exec, exec, s[20:21]
	s_addk_i32 s66, 0x400
	s_cmp_ge_u32 s66, s2
	s_cbranch_scc1 .LBB0_36
.LBB0_10:                               ;   Parent Loop BB0_6 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_15 Depth 3
	s_cmp_eq_u32 s66, 0
	s_cselect_b64 s[20:21], -1, 0
	s_add_i32 s22, s36, 0x5000
	s_cmp_eq_u32 s66, s22
	s_cselect_b64 s[24:25], -1, 0
	s_or_b64 s[24:25], s[20:21], s[24:25]
	s_andn2_b64 vcc, exec, s[24:25]
	s_cbranch_vccz .LBB0_12
; %bb.11:                               ;   in Loop: Header=BB0_10 Depth=2
	s_and_saveexec_b64 s[20:21], s[8:9]
	s_cbranch_execz .LBB0_9
	s_branch .LBB0_33
.LBB0_12:                               ;   in Loop: Header=BB0_10 Depth=2
	s_and_b64 s[20:21], s[20:21], exec
	s_cselect_b32 s36, s36, s22
	s_sub_i32 s20, s2, s36
	s_min_u32 s67, s20, 0x5000
	s_lshl_b64 s[20:21], s[36:37], 1
	s_add_u32 s22, s6, s20
	s_addc_u32 s23, s7, s21
	s_mov_b32 s68, 0
	s_mov_b64 s[24:25], 0
	v_mov_b32_e32 v54, v121
	s_barrier
                                        ; implicit-def: $sgpr60_sgpr61
	s_branch .LBB0_15
.LBB0_13:                               ;   in Loop: Header=BB0_15 Depth=3
	s_add_i32 s68, s68, s72
	s_cmpk_gt_u32 s68, 0x4fff
	s_cselect_b64 s[20:21], -1, 0
	s_andn2_b64 s[60:61], s[60:61], exec
	s_and_b64 s[20:21], s[20:21], exec
	v_add_u32_e32 v54, s76, v54
	s_or_b64 s[60:61], s[60:61], s[20:21]
.LBB0_14:                               ;   in Loop: Header=BB0_15 Depth=3
	s_or_b64 exec, exec, s[62:63]
	s_and_b64 s[20:21], exec, s[60:61]
	s_or_b64 s[24:25], s[20:21], s[24:25]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execz .LBB0_32
.LBB0_15:                               ;   Parent Loop BB0_6 Depth=1
                                        ;     Parent Loop BB0_10 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	v_add_u32_e32 v2, s68, v120
	v_cmp_gt_u32_e32 vcc, s67, v2
	s_or_b64 s[60:61], s[60:61], exec
	s_and_saveexec_b64 s[62:63], vcc
	s_cbranch_execz .LBB0_14
; %bb.16:                               ;   in Loop: Header=BB0_15 Depth=3
	v_lshl_add_u64 v[52:53], v[2:3], 1, s[22:23]
	s_mov_b64 s[20:21], -1
	s_and_b64 vcc, exec, s[40:41]
	s_cbranch_vccz .LBB0_18
; %bb.17:                               ;   in Loop: Header=BB0_15 Depth=3
	global_load_dwordx4 v[56:59], v[52:53], off
	s_mov_b64 s[20:21], 0
	s_waitcnt vmcnt(0)
	ds_write_b128 v54, v[56:59]
.LBB0_18:                               ;   in Loop: Header=BB0_15 Depth=3
	s_andn2_b64 vcc, exec, s[20:21]
	s_cbranch_vccnz .LBB0_20
; %bb.19:                               ;   in Loop: Header=BB0_15 Depth=3
	v_readfirstlane_b32 s20, v54
	s_mov_b32 m0, s20
	s_nop 0
	global_load_lds_dwordx4 v[52:53], off
.LBB0_20:                               ;   in Loop: Header=BB0_15 Depth=3
	v_lshl_add_u64 v[52:53], s[26:27], 1, v[52:53]
	v_add_u32_e32 v2, 0xa000, v54
	v_cmp_ne_u32_e64 s[20:21], 1, v122
	s_andn2_b64 vcc, exec, s[40:41]
	s_mov_b64 s[64:65], -1
	s_cbranch_vccnz .LBB0_22
; %bb.21:                               ;   in Loop: Header=BB0_15 Depth=3
	global_load_dwordx4 v[56:59], v[52:53], off
	s_mov_b64 s[64:65], 0
	s_waitcnt vmcnt(0)
	ds_write_b128 v2, v[56:59]
.LBB0_22:                               ;   in Loop: Header=BB0_15 Depth=3
	s_andn2_b64 vcc, exec, s[64:65]
	s_cbranch_vccnz .LBB0_24
; %bb.23:                               ;   in Loop: Header=BB0_15 Depth=3
	v_readfirstlane_b32 s64, v2
	s_mov_b32 m0, s64
	s_nop 0
	global_load_lds_dwordx4 v[52:53], off
.LBB0_24:                               ;   in Loop: Header=BB0_15 Depth=3
	v_lshl_add_u64 v[52:53], v[52:53], 0, s[58:59]
	v_add_u32_e32 v2, 0x14000, v54
	s_and_b64 vcc, exec, s[20:21]
	s_mov_b64 s[64:65], -1
	s_cbranch_vccnz .LBB0_26
; %bb.25:                               ;   in Loop: Header=BB0_15 Depth=3
	global_load_dwordx4 v[56:59], v[52:53], off
	s_mov_b64 s[64:65], 0
	s_waitcnt vmcnt(0)
	ds_write_b128 v2, v[56:59]
.LBB0_26:                               ;   in Loop: Header=BB0_15 Depth=3
	s_andn2_b64 vcc, exec, s[64:65]
	s_cbranch_vccnz .LBB0_28
; %bb.27:                               ;   in Loop: Header=BB0_15 Depth=3
	v_readfirstlane_b32 s64, v2
	s_mov_b32 m0, s64
	s_nop 0
	global_load_lds_dwordx4 v[52:53], off
.LBB0_28:                               ;   in Loop: Header=BB0_15 Depth=3
	v_lshl_add_u64 v[52:53], v[52:53], 0, s[58:59]
	v_add_u32_e32 v2, 0x1e000, v54
	s_and_b64 vcc, exec, s[20:21]
	s_mov_b64 s[20:21], -1
	s_cbranch_vccnz .LBB0_30
; %bb.29:                               ;   in Loop: Header=BB0_15 Depth=3
	global_load_dwordx4 v[56:59], v[52:53], off
	s_mov_b64 s[20:21], 0
	s_waitcnt vmcnt(0)
	ds_write_b128 v2, v[56:59]
.LBB0_30:                               ;   in Loop: Header=BB0_15 Depth=3
	s_andn2_b64 vcc, exec, s[20:21]
	s_cbranch_vccnz .LBB0_13
; %bb.31:                               ;   in Loop: Header=BB0_15 Depth=3
	v_readfirstlane_b32 s20, v2
	s_mov_b32 m0, s20
	s_nop 0
	global_load_lds_dwordx4 v[52:53], off
	s_branch .LBB0_13
.LBB0_32:                               ;   in Loop: Header=BB0_10 Depth=2
	s_or_b64 exec, exec, s[24:25]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	s_and_saveexec_b64 s[20:21], s[8:9]
	s_cbranch_execz .LBB0_9
.LBB0_33:                               ;   in Loop: Header=BB0_10 Depth=2
	v_add_u32_e32 v109, s66, v119
	v_min_u32_e32 v2, s73, v109
	v_lshl_add_u64 v[52:53], v[2:3], 1, s[4:5]
	v_add_u32_e32 v111, 0x200, v109
	v_lshl_add_u64 v[54:55], v[52:53], 0, v[112:113]
	v_lshl_add_u64 v[56:57], v[52:53], 0, v[114:115]
	v_min_u32_e32 v2, s73, v111
	global_load_dwordx4 v[68:71], v[54:55], off nt
	global_load_dwordx4 v[64:67], v[56:57], off nt
	v_lshl_add_u64 v[56:57], v[2:3], 1, s[4:5]
	v_lshl_add_u64 v[52:53], v[52:53], 0, v[116:117]
	v_lshl_add_u64 v[54:55], v[56:57], 0, v[112:113]
	v_lshl_add_u64 v[58:59], v[56:57], 0, v[114:115]
	v_lshl_add_u64 v[56:57], v[56:57], 0, v[116:117]
	global_load_dwordx4 v[72:75], v[52:53], off nt
	s_nop 0
	global_load_dwordx4 v[52:55], v[54:55], off nt
	s_nop 0
	global_load_dwordx4 v[60:63], v[58:59], off nt
	s_nop 0
	global_load_dwordx4 v[56:59], v[56:57], off nt
	v_cmp_gt_u32_e32 vcc, s2, v109
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v79, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v90, 0
	v_mov_b32_e32 v91, 0
	v_mov_b32_e32 v88, 0
	v_mov_b32_e32 v89, 0
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v99, 0
	v_mov_b32_e32 v96, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v86, 0
	v_mov_b32_e32 v87, 0
	v_mov_b32_e32 v84, 0
	v_mov_b32_e32 v85, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v103, 0
	v_mov_b32_e32 v100, 0
	v_mov_b32_e32 v101, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v105, 0
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_8
; %bb.34:                               ;   in Loop: Header=BB0_10 Depth=2
	v_subrev_u32_e32 v2, s36, v109
	v_lshlrev_b32_e32 v2, 1, v2
	v_add_u32_e32 v76, 0x14000, v2
	ds_read_b128 v[104:107], v2
	ds_read_b128 v[100:103], v2 offset:40960
	v_add_u32_e32 v77, 0x1e000, v2
	ds_read_b128 v[96:99], v76
	ds_read_b128 v[80:83], v77
	v_cmp_gt_u32_e32 vcc, s2, v111
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v85, 0
	v_mov_b32_e32 v84, 0
	v_mov_b32_e32 v87, 0
	v_mov_b32_e32 v86, 0
	v_mov_b32_e32 v89, 0
	v_mov_b32_e32 v88, 0
	v_mov_b32_e32 v91, 0
	v_mov_b32_e32 v90, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v79, 0
	v_mov_b32_e32 v78, 0
	s_and_saveexec_b64 s[24:25], vcc
	s_cbranch_execz .LBB0_7
; %bb.35:                               ;   in Loop: Header=BB0_10 Depth=2
	v_add_u32_e32 v76, 0x14400, v2
	ds_read_b128 v[92:95], v2 offset:1024
	ds_read_b128 v[84:87], v2 offset:41984
	v_add_u32_e32 v2, 0x1e400, v2
	ds_read_b128 v[88:91], v76
	ds_read_b128 v[76:79], v2
	s_branch .LBB0_7
.LBB0_36:                               ;   in Loop: Header=BB0_6 Depth=1
	v_add_u32_e32 v2, s75, v108
	s_and_saveexec_b64 s[8:9], s[18:19]
	s_xor_b64 s[8:9], exec, s[8:9]
; %bb.37:                               ;   in Loop: Header=BB0_6 Depth=1
                                        ; implicit-def: $vgpr108
                                        ; implicit-def: $vgpr110
; %bb.38:                               ;   in Loop: Header=BB0_6 Depth=1
	s_andn2_saveexec_b64 s[60:61], s[8:9]
	s_cbranch_execz .LBB0_5
; %bb.39:                               ;   in Loop: Header=BB0_6 Depth=1
	v_add_f32_dpp v48, v49, v48 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v44, v45, v44 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v40, v41, v40 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v36, v37, v36 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v32, v33, v32 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v28, v29, v28 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v24, v25, v24 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v20, v21, v20 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v16, v17, v16 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v12, v13, v12 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v9, v8 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v5, v4 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v48, v50, v48 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v44, v46, v44 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v40, v42, v40 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v36, v38, v36 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v32, v34, v32 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v28, v30, v28 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v24, v26, v24 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v20, v22, v20 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v16, v18, v16 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v12, v14, v12 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v10, v8 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v6, v4 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v48, v51, v48 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v44, v47, v44 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v40, v43, v40 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v36, v39, v36 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v32, v35, v32 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v28, v31, v28 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v24, v27, v24 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v20, v23, v20 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v16, v19, v16 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v12, v15, v12 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v11, v8 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v7, v4 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v48, v48, v48 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v44, v44, v44 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v40, v40, v40 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v36, v36, v36 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v32, v32, v32 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v28, v28, v28 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v24, v24, v24 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v20, v20, v20 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v16, v16, v16 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v12, v12, v12 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v8, v8 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v48, v48, v48 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v44, v44, v44 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v40, v40, v40 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v36, v36, v36 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v32, v32, v32 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v28, v28, v28 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v24, v24, v24 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v20, v20, v20 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v16, v16, v16 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v12, v12, v12 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v8, v8 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v48, v48 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v44, v44 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v40, v40 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v36, v36 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v32, v32 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v28, v28 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v24, v24 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v20, v20 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v16, v16 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v12, v12 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v8, v8 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v4, v4 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v48, v48, v48 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v44, v44, v44 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v40, v40, v40 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v36, v36, v36 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v32, v32, v32 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v28, v28, v28 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v24, v24, v24 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v20, v20, v20 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v17, v16, v16 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v15, v12, v12 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v22, v8, v8 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v49, v48 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v45, v44 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v41, v40 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v37, v36 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v33, v32 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v29, v28 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v25, v24 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v21, v20 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v18, v17 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v19, v15 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v23, v22 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v26, v4 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_and_saveexec_b64 s[62:63], s[0:1]
	s_cbranch_execz .LBB0_4
; %bb.40:                               ;   in Loop: Header=BB0_6 Depth=1
	v_cmp_eq_u32_e64 s[8:9], 0, v1
	s_and_b64 s[18:19], s[42:43], s[8:9]
	v_add_f32_e32 v16, v48, v49
	v_add_f32_e32 v14, v44, v45
	v_add_f32_e32 v13, v40, v41
	v_add_f32_e32 v12, v36, v37
	v_add_f32_e32 v11, v32, v33
	v_add_f32_e32 v10, v28, v29
	v_add_f32_e32 v9, v24, v25
	v_add_f32_e32 v8, v20, v21
	v_add_f32_e32 v7, v17, v18
	v_add_f32_e32 v6, v15, v19
	v_add_f32_e32 v5, v22, v23
	v_add_f32_e32 v4, v4, v26
	s_xor_b64 s[20:21], s[18:19], -1
	v_mov_b32_e32 v111, v3
	s_mov_b64 s[18:19], 0
                                        ; implicit-def: $vgpr15
	s_and_saveexec_b64 s[22:23], s[20:21]
	s_xor_b64 s[64:65], exec, s[22:23]
	s_cbranch_execnz .LBB0_43
; %bb.41:                               ;   in Loop: Header=BB0_6 Depth=1
	s_andn2_saveexec_b64 s[8:9], s[64:65]
	s_cbranch_execnz .LBB0_100
.LBB0_42:                               ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[8:9]
	s_and_b64 exec, exec, s[18:19]
	s_cbranch_execz .LBB0_4
	s_branch .LBB0_101
.LBB0_43:                               ;   in Loop: Header=BB0_6 Depth=1
	v_cndmask_b32_e64 v15, 0, 1, s[44:45]
	v_cmp_ne_u32_e64 s[22:23], 1, v15
	s_and_saveexec_b64 s[18:19], s[8:9]
	s_cbranch_execz .LBB0_47
; %bb.44:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v15, s28, v16
	s_and_b64 vcc, exec, s[22:23]
	v_mov_b32_e32 v109, v3
	s_cbranch_vccnz .LBB0_46
; %bb.45:                               ;   in Loop: Header=BB0_6 Depth=1
	v_lshl_add_u64 v[16:17], v[108:109], 1, s[12:13]
	global_load_ushort v16, v[16:17], off
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v16, 16, v16
	v_fmac_f32_e32 v15, s29, v16
.LBB0_46:                               ;   in Loop: Header=BB0_6 Depth=1
	v_cvt_pk_bf16_f32 v15, v15, s0
	v_lshl_add_u64 v[16:17], v[108:109], 1, s[14:15]
	global_store_short v[16:17], v15, off
.LBB0_47:                               ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[18:19]
	v_cndmask_b32_e64 v15, 0, 1, s[42:43]
	v_cmp_gt_u32_e64 s[18:19], 2, v1
	v_cmp_ne_u32_e64 s[20:21], 1, v15
	s_and_saveexec_b64 s[24:25], s[18:19]
	s_cbranch_execnz .LBB0_57
; %bb.48:                               ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[24:25]
	v_cmp_gt_u32_e64 s[24:25], 3, v1
	s_and_saveexec_b64 s[66:67], s[24:25]
	s_cbranch_execnz .LBB0_62
.LBB0_49:                               ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[66:67]
	s_and_saveexec_b64 s[66:67], s[8:9]
	s_cbranch_execnz .LBB0_65
.LBB0_50:                               ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[66:67]
	s_and_saveexec_b64 s[66:67], s[18:19]
	s_cbranch_execnz .LBB0_68
.LBB0_51:                               ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[66:67]
	s_and_saveexec_b64 s[66:67], s[24:25]
	s_cbranch_execnz .LBB0_73
.LBB0_52:                               ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[66:67]
	s_and_saveexec_b64 s[66:67], s[8:9]
	s_cbranch_execnz .LBB0_76
.LBB0_53:                               ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[66:67]
	s_and_saveexec_b64 s[66:67], s[18:19]
	s_cbranch_execnz .LBB0_79
.LBB0_54:                               ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[66:67]
	s_and_saveexec_b64 s[66:67], s[24:25]
	s_cbranch_execnz .LBB0_84
.LBB0_55:                               ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[66:67]
	s_and_saveexec_b64 s[66:67], s[8:9]
	s_cbranch_execnz .LBB0_87
.LBB0_56:                               ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[66:67]
	s_and_saveexec_b64 s[8:9], s[18:19]
	s_cbranch_execnz .LBB0_90
	s_branch .LBB0_95
.LBB0_57:                               ;   in Loop: Header=BB0_6 Depth=1
	s_and_b64 vcc, exec, s[20:21]
	v_mov_b32_e32 v109, v3
	s_cbranch_vccnz .LBB0_59
; %bb.58:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v14, s28, v14
	s_cbranch_execz .LBB0_60
	s_branch .LBB0_61
.LBB0_59:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v14, s28, v14
.LBB0_60:                               ;   in Loop: Header=BB0_6 Depth=1
	v_lshl_add_u64 v[16:17], v[108:109], 1, s[12:13]
	global_load_ushort v15, v[16:17], off offset:2
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v15, 16, v15
	v_fmac_f32_e32 v14, s29, v15
.LBB0_61:                               ;   in Loop: Header=BB0_6 Depth=1
	v_cvt_pk_bf16_f32 v16, v14, s0
	v_lshl_add_u64 v[14:15], v[108:109], 1, s[14:15]
	global_store_short v[14:15], v16, off offset:2
	s_or_b64 exec, exec, s[24:25]
	v_cmp_gt_u32_e64 s[24:25], 3, v1
	s_and_saveexec_b64 s[66:67], s[24:25]
	s_cbranch_execz .LBB0_49
.LBB0_62:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v13, s28, v13
	s_and_b64 vcc, exec, s[22:23]
	v_mov_b32_e32 v111, v3
	s_cbranch_vccnz .LBB0_64
; %bb.63:                               ;   in Loop: Header=BB0_6 Depth=1
	v_lshl_add_u64 v[14:15], v[110:111], 1, s[12:13]
	global_load_ushort v14, v[14:15], off
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v14, 16, v14
	v_fmac_f32_e32 v13, s29, v14
.LBB0_64:                               ;   in Loop: Header=BB0_6 Depth=1
	v_cvt_pk_bf16_f32 v13, v13, s0
	v_lshl_add_u64 v[14:15], v[110:111], 1, s[14:15]
	global_store_short v[14:15], v13, off
	s_or_b64 exec, exec, s[66:67]
	s_and_saveexec_b64 s[66:67], s[8:9]
	s_cbranch_execz .LBB0_50
.LBB0_65:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v12, s28, v12
	s_and_b64 vcc, exec, s[22:23]
	v_mov_b32_e32 v109, v3
	s_cbranch_vccnz .LBB0_67
; %bb.66:                               ;   in Loop: Header=BB0_6 Depth=1
	v_lshl_add_u64 v[14:15], v[108:109], 1, s[46:47]
	global_load_ushort v13, v[14:15], off
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v13, 16, v13
	v_fmac_f32_e32 v12, s29, v13
.LBB0_67:                               ;   in Loop: Header=BB0_6 Depth=1
	v_cvt_pk_bf16_f32 v14, v12, s0
	v_lshl_add_u64 v[12:13], v[108:109], 1, s[48:49]
	global_store_short v[12:13], v14, off
	s_or_b64 exec, exec, s[66:67]
	s_and_saveexec_b64 s[66:67], s[18:19]
	s_cbranch_execz .LBB0_51
.LBB0_68:                               ;   in Loop: Header=BB0_6 Depth=1
	s_and_b64 vcc, exec, s[20:21]
	v_mov_b32_e32 v109, v3
	s_cbranch_vccnz .LBB0_70
; %bb.69:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v11, s28, v11
	s_cbranch_execz .LBB0_71
	s_branch .LBB0_72
.LBB0_70:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v11, s28, v11
.LBB0_71:                               ;   in Loop: Header=BB0_6 Depth=1
	v_lshl_add_u64 v[12:13], v[108:109], 1, s[46:47]
	global_load_ushort v12, v[12:13], off offset:2
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v12, 16, v12
	v_fmac_f32_e32 v11, s29, v12
.LBB0_72:                               ;   in Loop: Header=BB0_6 Depth=1
	v_cvt_pk_bf16_f32 v11, v11, s0
	v_lshl_add_u64 v[12:13], v[108:109], 1, s[48:49]
	global_store_short v[12:13], v11, off offset:2
	s_or_b64 exec, exec, s[66:67]
	s_and_saveexec_b64 s[66:67], s[24:25]
	s_cbranch_execz .LBB0_52
.LBB0_73:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v10, s28, v10
	s_and_b64 vcc, exec, s[22:23]
	v_mov_b32_e32 v111, v3
	s_cbranch_vccnz .LBB0_75
; %bb.74:                               ;   in Loop: Header=BB0_6 Depth=1
	v_lshl_add_u64 v[12:13], v[110:111], 1, s[46:47]
	global_load_ushort v11, v[12:13], off
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v11, 16, v11
	v_fmac_f32_e32 v10, s29, v11
.LBB0_75:                               ;   in Loop: Header=BB0_6 Depth=1
	v_cvt_pk_bf16_f32 v12, v10, s0
	v_lshl_add_u64 v[10:11], v[110:111], 1, s[48:49]
	global_store_short v[10:11], v12, off
	s_or_b64 exec, exec, s[66:67]
	s_and_saveexec_b64 s[66:67], s[8:9]
	s_cbranch_execz .LBB0_53
.LBB0_76:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v9, s28, v9
	s_and_b64 vcc, exec, s[22:23]
	v_mov_b32_e32 v109, v3
	s_cbranch_vccnz .LBB0_78
; %bb.77:                               ;   in Loop: Header=BB0_6 Depth=1
	v_lshl_add_u64 v[10:11], v[108:109], 1, s[50:51]
	global_load_ushort v10, v[10:11], off
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v10, 16, v10
	v_fmac_f32_e32 v9, s29, v10
.LBB0_78:                               ;   in Loop: Header=BB0_6 Depth=1
	v_cvt_pk_bf16_f32 v9, v9, s0
	v_lshl_add_u64 v[10:11], v[108:109], 1, s[52:53]
	global_store_short v[10:11], v9, off
	s_or_b64 exec, exec, s[66:67]
	s_and_saveexec_b64 s[66:67], s[18:19]
	s_cbranch_execz .LBB0_54
.LBB0_79:                               ;   in Loop: Header=BB0_6 Depth=1
	s_and_b64 vcc, exec, s[20:21]
	v_mov_b32_e32 v109, v3
	s_cbranch_vccnz .LBB0_81
; %bb.80:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v8, s28, v8
	s_cbranch_execz .LBB0_82
	s_branch .LBB0_83
.LBB0_81:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v8, s28, v8
.LBB0_82:                               ;   in Loop: Header=BB0_6 Depth=1
	v_lshl_add_u64 v[10:11], v[108:109], 1, s[50:51]
	global_load_ushort v9, v[10:11], off offset:2
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v9, 16, v9
	v_fmac_f32_e32 v8, s29, v9
.LBB0_83:                               ;   in Loop: Header=BB0_6 Depth=1
	v_cvt_pk_bf16_f32 v10, v8, s0
	v_lshl_add_u64 v[8:9], v[108:109], 1, s[52:53]
	global_store_short v[8:9], v10, off offset:2
	s_or_b64 exec, exec, s[66:67]
	s_and_saveexec_b64 s[66:67], s[24:25]
	s_cbranch_execz .LBB0_55
.LBB0_84:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v7, s28, v7
	s_and_b64 vcc, exec, s[22:23]
	v_mov_b32_e32 v111, v3
	s_cbranch_vccnz .LBB0_86
; %bb.85:                               ;   in Loop: Header=BB0_6 Depth=1
	v_lshl_add_u64 v[8:9], v[110:111], 1, s[50:51]
	global_load_ushort v8, v[8:9], off
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v8, 16, v8
	v_fmac_f32_e32 v7, s29, v8
.LBB0_86:                               ;   in Loop: Header=BB0_6 Depth=1
	v_cvt_pk_bf16_f32 v7, v7, s0
	v_lshl_add_u64 v[8:9], v[110:111], 1, s[52:53]
	global_store_short v[8:9], v7, off
	s_or_b64 exec, exec, s[66:67]
	s_and_saveexec_b64 s[66:67], s[8:9]
	s_cbranch_execz .LBB0_56
.LBB0_87:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v6, s28, v6
	s_and_b64 vcc, exec, s[22:23]
	v_mov_b32_e32 v109, v3
	s_cbranch_vccnz .LBB0_89
; %bb.88:                               ;   in Loop: Header=BB0_6 Depth=1
	v_lshl_add_u64 v[8:9], v[108:109], 1, s[54:55]
	global_load_ushort v7, v[8:9], off
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v7, 16, v7
	v_fmac_f32_e32 v6, s29, v7
.LBB0_89:                               ;   in Loop: Header=BB0_6 Depth=1
	v_cvt_pk_bf16_f32 v8, v6, s0
	v_lshl_add_u64 v[6:7], v[108:109], 1, s[56:57]
	global_store_short v[6:7], v8, off
	s_or_b64 exec, exec, s[66:67]
	s_and_saveexec_b64 s[8:9], s[18:19]
	s_cbranch_execz .LBB0_95
.LBB0_90:                               ;   in Loop: Header=BB0_6 Depth=1
	s_and_b64 vcc, exec, s[20:21]
	v_mov_b32_e32 v109, v3
	s_cbranch_vccnz .LBB0_92
; %bb.91:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v5, s28, v5
	s_cbranch_execz .LBB0_93
	s_branch .LBB0_94
.LBB0_92:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v5, s28, v5
.LBB0_93:                               ;   in Loop: Header=BB0_6 Depth=1
	v_lshl_add_u64 v[6:7], v[108:109], 1, s[54:55]
	global_load_ushort v6, v[6:7], off offset:2
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v6, 16, v6
	v_fmac_f32_e32 v5, s29, v6
.LBB0_94:                               ;   in Loop: Header=BB0_6 Depth=1
	v_cvt_pk_bf16_f32 v5, v5, s0
	v_lshl_add_u64 v[6:7], v[108:109], 1, s[56:57]
	global_store_short v[6:7], v5, off offset:2
.LBB0_95:                               ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[8:9]
	s_mov_b64 s[18:19], 0
                                        ; implicit-def: $vgpr15
	s_and_saveexec_b64 s[8:9], s[24:25]
	s_xor_b64 s[8:9], exec, s[8:9]
	s_cbranch_execz .LBB0_99
; %bb.96:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v15, s28, v4
	s_and_b64 vcc, exec, s[22:23]
	v_mov_b32_e32 v111, v3
	s_cbranch_vccnz .LBB0_98
; %bb.97:                               ;   in Loop: Header=BB0_6 Depth=1
	v_lshl_add_u64 v[4:5], v[110:111], 1, s[54:55]
	global_load_ushort v4, v[4:5], off
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v4, 16, v4
	v_fmac_f32_e32 v15, s29, v4
.LBB0_98:                               ;   in Loop: Header=BB0_6 Depth=1
	s_mov_b64 s[18:19], exec
.LBB0_99:                               ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[8:9]
	s_and_b64 s[18:19], s[18:19], exec
                                        ; implicit-def: $vgpr16
                                        ; implicit-def: $vgpr14
                                        ; implicit-def: $vgpr13
                                        ; implicit-def: $vgpr12
                                        ; implicit-def: $vgpr11
                                        ; implicit-def: $vgpr10
                                        ; implicit-def: $vgpr9
                                        ; implicit-def: $vgpr8
                                        ; implicit-def: $vgpr7
                                        ; implicit-def: $vgpr6
                                        ; implicit-def: $vgpr5
                                        ; implicit-def: $vgpr4
                                        ; implicit-def: $vgpr108
	s_andn2_saveexec_b64 s[8:9], s[64:65]
	s_cbranch_execz .LBB0_42
.LBB0_100:                              ;   in Loop: Header=BB0_6 Depth=1
	v_mov_b32_e32 v109, v3
	v_mul_f32_e32 v15, s28, v16
	v_lshlrev_b64 v[16:17], 1, v[108:109]
	v_mul_f32_e32 v14, s28, v14
	v_lshl_add_u64 v[18:19], s[14:15], 0, v[16:17]
	v_cvt_pk_bf16_f32 v14, v15, v14
	global_store_dword v[18:19], v14, off
	v_mul_f32_e32 v13, s28, v13
	v_lshlrev_b64 v[14:15], 1, v[110:111]
	v_cvt_pk_bf16_f32 v13, v13, s0
	v_lshl_add_u64 v[18:19], s[14:15], 0, v[14:15]
	global_store_short v[18:19], v13, off
	v_mul_f32_e32 v18, s28, v12
	v_mul_f32_e32 v11, s28, v11
	v_lshl_add_u64 v[12:13], s[48:49], 0, v[16:17]
	v_cvt_pk_bf16_f32 v11, v18, v11
	v_mul_f32_e32 v10, s28, v10
	global_store_dword v[12:13], v11, off
	v_cvt_pk_bf16_f32 v12, v10, s0
	v_lshl_add_u64 v[10:11], s[48:49], 0, v[14:15]
	v_mul_f32_e32 v9, s28, v9
	v_mul_f32_e32 v8, s28, v8
	global_store_short v[10:11], v12, off
	v_lshl_add_u64 v[10:11], s[52:53], 0, v[16:17]
	v_cvt_pk_bf16_f32 v8, v9, v8
	v_mul_f32_e32 v7, s28, v7
	global_store_dword v[10:11], v8, off
	v_cvt_pk_bf16_f32 v7, v7, s0
	v_lshl_add_u64 v[8:9], s[52:53], 0, v[14:15]
	global_store_short v[8:9], v7, off
	v_mul_f32_e32 v8, s28, v6
	v_mul_f32_e32 v5, s28, v5
	v_lshl_add_u64 v[6:7], s[56:57], 0, v[16:17]
	v_cvt_pk_bf16_f32 v5, v8, v5
	global_store_dword v[6:7], v5, off
	v_mul_f32_e32 v15, s28, v4
	s_or_b64 s[18:19], s[18:19], exec
	s_or_b64 exec, exec, s[8:9]
	s_and_b64 exec, exec, s[18:19]
	s_cbranch_execz .LBB0_4
.LBB0_101:                              ;   in Loop: Header=BB0_6 Depth=1
	v_cvt_pk_bf16_f32 v6, v15, s0
	v_lshl_add_u64 v[4:5], v[110:111], 1, s[56:57]
	global_store_short v[4:5], v6, off
	s_branch .LBB0_4
.LBB0_102:
	s_or_b64 exec, exec, s[34:35]
	s_mov_b64 s[0:1], 0
.LBB0_103:
	s_andn2_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB0_187
; %bb.104:
	s_lshl_b32 s33, s2, 2
	v_and_b32_e32 v8, 0x3ff, v0
	v_lshlrev_b32_e32 v114, 3, v8
	s_cmp_eq_u32 s26, s2
	s_mov_b64 s[8:9], 0
	s_cbranch_scc1 .LBB0_106
; %bb.105:
	s_and_b32 s18, s6, 15
	s_mov_b32 s19, 0
	s_mov_b64 s[0:1], -1
	s_min_u32 s22, s33, 0x14000
	v_lshl_add_u32 v9, v118, 9, v114
	s_cbranch_execz .LBB0_107
	s_branch .LBB0_108
.LBB0_106:
                                        ; implicit-def: $sgpr18_sgpr19
	s_mov_b64 s[0:1], 0
	s_min_u32 s22, s33, 0x14000
	v_lshl_add_u32 v9, v118, 9, v114
.LBB0_107:
	s_and_b32 s0, s2, 7
	s_cmp_lg_u32 s0, 0
	s_mov_b32 s19, 0
	s_cselect_b64 s[0:1], -1, 0
	s_and_b32 s18, s6, 15
	s_cmp_lg_u64 s[18:19], 0
	s_cselect_b64 s[8:9], -1, 0
	s_or_b64 s[0:1], s[0:1], s[8:9]
	s_mov_b64 s[8:9], -1
.LBB0_108:
	s_andn2_b64 vcc, exec, s[0:1]
	v_cmp_gt_u32_e64 s[0:1], s22, v9
	s_cbranch_vccz .LBB0_110
; %bb.109:
	s_andn2_b64 vcc, exec, s[8:9]
	s_cbranch_vccz .LBB0_118
	s_branch .LBB0_122
.LBB0_110:
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_117
; %bb.111:
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
	v_lshl_add_u32 v10, v118, 10, v0
	v_mov_b32_e32 v0, v9
	s_branch .LBB0_113
.LBB0_112:                              ;   in Loop: Header=BB0_113 Depth=1
	v_add_u32_e32 v0, 0x2000, v0
	v_cmp_le_u32_e32 vcc, s22, v0
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[4:5]
	s_or_b64 s[0:1], vcc, s[0:1]
	v_add_u32_e32 v10, 0x4000, v10
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB0_117
.LBB0_113:                              ; =>This Inner Loop Header: Depth=1
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
	s_cbranch_vccz .LBB0_115
; %bb.114:                              ;   in Loop: Header=BB0_113 Depth=1
	global_load_dwordx4 v[12:15], v[6:7], off
	s_mov_b64 s[20:21], 0
	s_waitcnt vmcnt(0)
	ds_write_b128 v10, v[12:15]
.LBB0_115:                              ;   in Loop: Header=BB0_113 Depth=1
	s_andn2_b64 vcc, exec, s[20:21]
	s_cbranch_vccnz .LBB0_112
; %bb.116:                              ;   in Loop: Header=BB0_113 Depth=1
	v_readfirstlane_b32 s20, v10
	s_mov_b32 m0, s20
	s_nop 0
	global_load_lds_dwordx4 v[6:7], off
	s_branch .LBB0_112
.LBB0_117:
	s_or_b64 exec, exec, s[8:9]
	s_cbranch_execnz .LBB0_122
.LBB0_118:
	v_cmp_gt_u32_e32 vcc, s22, v9
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_121
; %bb.119:
	v_lshlrev_b32_e32 v0, 4, v8
	v_lshl_add_u32 v2, v118, 10, v0
	v_lshlrev_b32_e32 v0, 1, v9
	v_mov_b32_e32 v1, 0
	v_lshl_add_u64 v[0:1], s[6:7], 0, v[0:1]
	s_mov_b64 s[6:7], 0
	s_mov_b64 s[8:9], 0x4000
.LBB0_120:                              ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s18, v2
	s_mov_b32 m0, s18
	v_add_u32_e32 v9, 0x2000, v9
	global_load_lds_dwordx4 v[0:1], off
	v_cmp_le_u32_e32 vcc, s22, v9
	v_add_u32_e32 v2, 0x4000, v2
	s_or_b64 s[6:7], vcc, s[6:7]
	v_lshl_add_u64 v[0:1], v[0:1], 0, s[8:9]
	s_andn2_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB0_120
.LBB0_121:
	s_or_b64 exec, exec, s[0:1]
.LBB0_122:
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
	v_add_u32_e32 v0, s0, v118
	v_lshl_add_u32 v1, v0, 1, v0
	v_add_u32_e32 v0, 3, v1
	v_cmp_gt_u32_e32 vcc, s3, v1
	v_cmp_lt_u32_e64 s[0:1], s3, v0
	s_add_i32 s50, s3, -3
	v_mov_b32_e32 v2, s50
	s_and_b64 s[0:1], vcc, s[0:1]
	v_cndmask_b32_e64 v104, v1, v2, s[0:1]
	v_cmp_gt_u32_e64 s[0:1], s8, v118
	v_cmp_gt_u32_e64 s[6:7], s3, v104
	s_and_b64 s[0:1], s[0:1], s[6:7]
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_187
; %bb.123:
	s_cmp_lg_u32 s2, 0
	s_cselect_b64 s[6:7], -1, 0
	s_mul_i32 s11, s8, s11
	s_ashr_i32 s9, s30, 31
	s_mov_b32 s8, s30
	s_add_i32 s51, s2, -8
	s_add_i32 s52, s3, -1
	s_ashr_i32 s53, s10, 31
	s_ashr_i32 s17, s31, 31
	s_lshl_b64 s[8:9], s[8:9], 1
	s_mov_b32 s16, s31
	s_add_u32 s26, s12, s8
	s_addc_u32 s27, s13, s9
	s_lshl_b64 s[16:17], s[16:17], 1
	s_add_u32 s30, s14, s16
	s_addc_u32 s31, s15, s17
	s_add_u32 s34, s26, s8
	s_addc_u32 s35, s27, s9
	s_add_u32 s36, s30, s16
	s_addc_u32 s37, s31, s17
	s_add_u32 s38, s34, s8
	v_sub_u32_e64 v0, v0, s3 clamp
	s_addc_u32 s39, s35, s9
	v_cndmask_b32_e32 v115, 0, v0, vcc
	s_add_u32 s40, s36, s16
	v_cndmask_b32_e64 v0, 0, 1, s[6:7]
	v_cmp_eq_u32_e64 s[0:1], 63, v8
	v_cmp_eq_f32_e64 s[22:23], s29, 0
	v_cmp_neq_f32_e64 s[24:25], s29, 0
	s_addc_u32 s41, s37, s17
	s_lshl_b32 s54, s2, 1
	v_lshlrev_b32_e32 v116, 4, v8
	s_mul_i32 s55, s2, 6
	s_mov_b64 s[42:43], 0
	v_cmp_ne_u32_e64 s[20:21], 1, v0
	v_mov_b32_e32 v107, 0
	s_branch .LBB0_125
.LBB0_124:                              ;   in Loop: Header=BB0_125 Depth=1
	s_or_b64 exec, exec, s[44:45]
	v_add_u32_e32 v0, s11, v104
	v_add_u32_e32 v1, 3, v0
	v_cmp_gt_u32_e32 vcc, s3, v0
	v_cmp_lt_u32_e64 s[6:7], s3, v1
	v_mov_b32_e32 v2, s50
	s_and_b64 vcc, vcc, s[6:7]
	v_cndmask_b32_e32 v104, v0, v2, vcc
	v_subrev_u32_e32 v1, s3, v1
	v_cmp_le_u32_e64 s[6:7], s3, v104
	s_or_b64 s[42:43], s[6:7], s[42:43]
	v_cndmask_b32_e32 v115, v115, v1, vcc
	s_andn2_b64 exec, exec, s[42:43]
	s_cbranch_execz .LBB0_187
.LBB0_125:                              ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_129 Depth 2
	s_and_b64 vcc, exec, s[20:21]
	v_mov_b32_e32 v27, v107
	v_mov_b32_e32 v26, v107
	v_mov_b32_e32 v25, v107
	v_mov_b32_e32 v24, v107
	v_mov_b32_e32 v3, v107
	v_mov_b32_e32 v2, v107
	v_mov_b32_e32 v1, v107
	v_mov_b32_e32 v0, v107
	v_mov_b32_e32 v7, v107
	v_mov_b32_e32 v6, v107
	v_mov_b32_e32 v5, v107
	v_mov_b32_e32 v4, v107
	v_mov_b32_e32 v11, v107
	v_mov_b32_e32 v10, v107
	v_mov_b32_e32 v9, v107
	v_mov_b32_e32 v8, v107
	v_mov_b32_e32 v15, v107
	v_mov_b32_e32 v14, v107
	v_mov_b32_e32 v13, v107
	v_mov_b32_e32 v12, v107
	v_mov_b32_e32 v19, v107
	v_mov_b32_e32 v18, v107
	v_mov_b32_e32 v17, v107
	v_mov_b32_e32 v16, v107
	v_mov_b32_e32 v47, v107
	v_mov_b32_e32 v46, v107
	v_mov_b32_e32 v45, v107
	v_mov_b32_e32 v44, v107
	v_mov_b32_e32 v43, v107
	v_mov_b32_e32 v42, v107
	v_mov_b32_e32 v41, v107
	v_mov_b32_e32 v40, v107
	v_mov_b32_e32 v39, v107
	v_mov_b32_e32 v38, v107
	v_mov_b32_e32 v37, v107
	v_mov_b32_e32 v36, v107
	v_mov_b32_e32 v35, v107
	v_mov_b32_e32 v34, v107
	v_mov_b32_e32 v33, v107
	v_mov_b32_e32 v32, v107
	v_mov_b32_e32 v31, v107
	v_mov_b32_e32 v30, v107
	v_mov_b32_e32 v29, v107
	v_mov_b32_e32 v28, v107
	v_mov_b32_e32 v23, v107
	v_mov_b32_e32 v22, v107
	v_mov_b32_e32 v21, v107
	v_mov_b32_e32 v20, v107
	s_cbranch_vccnz .LBB0_132
; %bb.126:                              ;   in Loop: Header=BB0_125 Depth=1
	v_min_u32_e32 v3, s52, v104
	v_mad_u64_u32 v[0:1], s[6:7], v3, s10, 0
	v_mov_b32_e32 v2, v1
	v_mad_u64_u32 v[2:3], s[6:7], v3, s53, v[2:3]
	v_mov_b32_e32 v1, v2
	v_add_u32_e32 v2, 1, v104
	v_min_u32_e32 v5, s52, v2
	v_mad_u64_u32 v[2:3], s[6:7], v5, s10, 0
	v_mov_b32_e32 v4, v3
	v_mad_u64_u32 v[4:5], s[6:7], v5, s53, v[4:5]
	v_mov_b32_e32 v3, v4
	v_add_u32_e32 v4, 2, v104
	v_min_u32_e32 v7, s52, v4
	v_mad_u64_u32 v[4:5], s[6:7], v7, s10, 0
	v_mov_b32_e32 v6, v5
	v_mad_u64_u32 v[6:7], s[6:7], v7, s53, v[6:7]
	v_mov_b32_e32 v5, v6
	v_mov_b32_e32 v20, 0
	s_mov_b32 s16, 0
	v_lshlrev_b64 v[108:109], 1, v[0:1]
	v_lshlrev_b64 v[110:111], 1, v[2:3]
	v_lshlrev_b64 v[112:113], 1, v[4:5]
	v_mov_b32_e32 v105, v116
	v_mov_b32_e32 v21, v20
	v_mov_b32_e32 v22, v20
	v_mov_b32_e32 v23, v20
	v_mov_b32_e32 v28, v20
	v_mov_b32_e32 v29, v20
	v_mov_b32_e32 v30, v20
	v_mov_b32_e32 v31, v20
	v_mov_b32_e32 v32, v20
	v_mov_b32_e32 v33, v20
	v_mov_b32_e32 v34, v20
	v_mov_b32_e32 v35, v20
	v_mov_b32_e32 v36, v20
	v_mov_b32_e32 v37, v20
	v_mov_b32_e32 v38, v20
	v_mov_b32_e32 v39, v20
	v_mov_b32_e32 v40, v20
	v_mov_b32_e32 v41, v20
	v_mov_b32_e32 v42, v20
	v_mov_b32_e32 v43, v20
	v_mov_b32_e32 v44, v20
	v_mov_b32_e32 v45, v20
	v_mov_b32_e32 v46, v20
	v_mov_b32_e32 v47, v20
	v_mov_b32_e32 v16, v20
	v_mov_b32_e32 v17, v20
	v_mov_b32_e32 v18, v20
	v_mov_b32_e32 v19, v20
	v_mov_b32_e32 v12, v20
	v_mov_b32_e32 v13, v20
	v_mov_b32_e32 v14, v20
	v_mov_b32_e32 v15, v20
	v_mov_b32_e32 v8, v20
	v_mov_b32_e32 v9, v20
	v_mov_b32_e32 v10, v20
	v_mov_b32_e32 v11, v20
	v_mov_b32_e32 v4, v20
	v_mov_b32_e32 v5, v20
	v_mov_b32_e32 v6, v20
	v_mov_b32_e32 v7, v20
	v_mov_b32_e32 v0, v20
	v_mov_b32_e32 v1, v20
	v_mov_b32_e32 v2, v20
	v_mov_b32_e32 v3, v20
	v_mov_b32_e32 v24, v20
	v_mov_b32_e32 v25, v20
	v_mov_b32_e32 v26, v20
	v_mov_b32_e32 v27, v20
	s_branch .LBB0_129
.LBB0_127:                              ;   in Loop: Header=BB0_129 Depth=2
	s_or_b64 exec, exec, s[8:9]
.LBB0_128:                              ;   in Loop: Header=BB0_129 Depth=2
	s_or_b64 exec, exec, s[6:7]
	s_waitcnt vmcnt(5) lgkmcnt(1)
	v_mfma_f32_4x4x4_16b_bf16 v[44:47], v[100:101], v[68:69], v[44:47]
	s_addk_i32 s16, 0x400
	s_waitcnt vmcnt(4)
	v_mfma_f32_4x4x4_16b_bf16 v[40:43], v[100:101], v[64:65], v[40:43]
	v_add_u32_e32 v105, 0x800, v105
	s_waitcnt vmcnt(3)
	v_mfma_f32_4x4x4_16b_bf16 v[36:39], v[100:101], v[72:73], v[36:39]
	s_cmp_ge_u32 s16, s2
	v_mfma_f32_4x4x4_16b_bf16 v[32:35], v[96:97], v[68:69], v[32:35]
	v_mfma_f32_4x4x4_16b_bf16 v[28:31], v[96:97], v[64:65], v[28:31]
	v_mfma_f32_4x4x4_16b_bf16 v[20:23], v[96:97], v[72:73], v[20:23]
	v_mfma_f32_4x4x4_16b_bf16 v[16:19], v[88:89], v[68:69], v[16:19]
	v_mfma_f32_4x4x4_16b_bf16 v[12:15], v[88:89], v[64:65], v[12:15]
	v_mfma_f32_4x4x4_16b_bf16 v[8:11], v[88:89], v[72:73], v[8:11]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_4x4x4_16b_bf16 v[4:7], v[80:81], v[68:69], v[4:7]
	v_mfma_f32_4x4x4_16b_bf16 v[0:3], v[80:81], v[64:65], v[0:3]
	v_mfma_f32_4x4x4_16b_bf16 v[24:27], v[80:81], v[72:73], v[24:27]
	v_mfma_f32_4x4x4_16b_bf16 v[44:47], v[102:103], v[70:71], v[44:47]
	v_mfma_f32_4x4x4_16b_bf16 v[40:43], v[102:103], v[66:67], v[40:43]
	v_mfma_f32_4x4x4_16b_bf16 v[36:39], v[102:103], v[74:75], v[36:39]
	v_mfma_f32_4x4x4_16b_bf16 v[32:35], v[98:99], v[70:71], v[32:35]
	v_mfma_f32_4x4x4_16b_bf16 v[28:31], v[98:99], v[66:67], v[28:31]
	v_mfma_f32_4x4x4_16b_bf16 v[20:23], v[98:99], v[74:75], v[20:23]
	v_mfma_f32_4x4x4_16b_bf16 v[16:19], v[90:91], v[70:71], v[16:19]
	v_mfma_f32_4x4x4_16b_bf16 v[12:15], v[90:91], v[66:67], v[12:15]
	v_mfma_f32_4x4x4_16b_bf16 v[8:11], v[90:91], v[74:75], v[8:11]
	v_mfma_f32_4x4x4_16b_bf16 v[4:7], v[82:83], v[70:71], v[4:7]
	v_mfma_f32_4x4x4_16b_bf16 v[0:3], v[82:83], v[66:67], v[0:3]
	v_mfma_f32_4x4x4_16b_bf16 v[24:27], v[82:83], v[74:75], v[24:27]
	s_waitcnt vmcnt(2)
	v_mfma_f32_4x4x4_16b_bf16 v[44:47], v[92:93], v[56:57], v[44:47]
	s_waitcnt vmcnt(1)
	v_mfma_f32_4x4x4_16b_bf16 v[40:43], v[92:93], v[60:61], v[40:43]
	s_waitcnt vmcnt(0)
	v_mfma_f32_4x4x4_16b_bf16 v[36:39], v[92:93], v[48:49], v[36:39]
	v_mfma_f32_4x4x4_16b_bf16 v[32:35], v[84:85], v[56:57], v[32:35]
	v_mfma_f32_4x4x4_16b_bf16 v[28:31], v[84:85], v[60:61], v[28:31]
	v_mfma_f32_4x4x4_16b_bf16 v[20:23], v[84:85], v[48:49], v[20:23]
	v_mfma_f32_4x4x4_16b_bf16 v[16:19], v[76:77], v[56:57], v[16:19]
	v_mfma_f32_4x4x4_16b_bf16 v[12:15], v[76:77], v[60:61], v[12:15]
	v_mfma_f32_4x4x4_16b_bf16 v[8:11], v[76:77], v[48:49], v[8:11]
	v_mfma_f32_4x4x4_16b_bf16 v[4:7], v[52:53], v[56:57], v[4:7]
	v_mfma_f32_4x4x4_16b_bf16 v[0:3], v[52:53], v[60:61], v[0:3]
	v_mfma_f32_4x4x4_16b_bf16 v[24:27], v[52:53], v[48:49], v[24:27]
	v_mfma_f32_4x4x4_16b_bf16 v[44:47], v[94:95], v[58:59], v[44:47]
	v_mfma_f32_4x4x4_16b_bf16 v[40:43], v[94:95], v[62:63], v[40:43]
	v_mfma_f32_4x4x4_16b_bf16 v[36:39], v[94:95], v[50:51], v[36:39]
	v_mfma_f32_4x4x4_16b_bf16 v[32:35], v[86:87], v[58:59], v[32:35]
	v_mfma_f32_4x4x4_16b_bf16 v[28:31], v[86:87], v[62:63], v[28:31]
	v_mfma_f32_4x4x4_16b_bf16 v[20:23], v[86:87], v[50:51], v[20:23]
	v_mfma_f32_4x4x4_16b_bf16 v[16:19], v[78:79], v[58:59], v[16:19]
	v_mfma_f32_4x4x4_16b_bf16 v[12:15], v[78:79], v[62:63], v[12:15]
	v_mfma_f32_4x4x4_16b_bf16 v[8:11], v[78:79], v[50:51], v[8:11]
	v_mfma_f32_4x4x4_16b_bf16 v[4:7], v[54:55], v[58:59], v[4:7]
	v_mfma_f32_4x4x4_16b_bf16 v[0:3], v[54:55], v[62:63], v[0:3]
	v_mfma_f32_4x4x4_16b_bf16 v[24:27], v[54:55], v[50:51], v[24:27]
	s_cbranch_scc1 .LBB0_132
.LBB0_129:                              ;   Parent Loop BB0_125 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_add_u32_e32 v54, s16, v114
	v_min_u32_e32 v106, s51, v54
	v_lshl_add_u64 v[48:49], v[106:107], 1, s[4:5]
	v_add_u32_e32 v117, 0x200, v54
	v_lshl_add_u64 v[50:51], v[48:49], 0, v[108:109]
	v_min_u32_e32 v106, s51, v117
	v_lshl_add_u64 v[52:53], v[48:49], 0, v[110:111]
	global_load_dwordx4 v[68:71], v[50:51], off nt
	global_load_dwordx4 v[64:67], v[52:53], off nt
	v_lshl_add_u64 v[48:49], v[48:49], 0, v[112:113]
	v_lshl_add_u64 v[50:51], v[106:107], 1, s[4:5]
	v_lshl_add_u64 v[52:53], v[50:51], 0, v[108:109]
	global_load_dwordx4 v[72:75], v[48:49], off nt
	global_load_dwordx4 v[56:59], v[52:53], off nt
	v_lshl_add_u64 v[48:49], v[50:51], 0, v[110:111]
	v_lshl_add_u64 v[50:51], v[50:51], 0, v[112:113]
	global_load_dwordx4 v[60:63], v[48:49], off nt
	s_nop 0
	global_load_dwordx4 v[48:51], v[50:51], off nt
	v_cmp_gt_u32_e32 vcc, s2, v54
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v79, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v90, 0
	v_mov_b32_e32 v91, 0
	v_mov_b32_e32 v88, 0
	v_mov_b32_e32 v89, 0
	v_mov_b32_e32 v86, 0
	v_mov_b32_e32 v87, 0
	v_mov_b32_e32 v84, 0
	v_mov_b32_e32 v85, 0
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v99, 0
	v_mov_b32_e32 v96, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v103, 0
	v_mov_b32_e32 v100, 0
	v_mov_b32_e32 v101, 0
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_128
; %bb.130:                              ;   in Loop: Header=BB0_129 Depth=2
	v_add_u32_e32 v119, s54, v105
	v_add_u32_e32 v118, s33, v105
	ds_read_b128 v[96:99], v119
	ds_read_b128 v[88:91], v118
	v_add_u32_e32 v106, s55, v105
	ds_read_b128 v[100:103], v105
	ds_read_b128 v[80:83], v106
	v_cmp_gt_u32_e32 vcc, s2, v117
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v85, 0
	v_mov_b32_e32 v84, 0
	v_mov_b32_e32 v87, 0
	v_mov_b32_e32 v86, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v79, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v54, 0
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_127
; %bb.131:                              ;   in Loop: Header=BB0_129 Depth=2
	ds_read_b128 v[84:87], v119 offset:1024
	ds_read_b128 v[76:79], v118 offset:1024
	ds_read_b128 v[92:95], v105 offset:1024
	ds_read_b128 v[52:55], v106 offset:1024
	s_branch .LBB0_127
.LBB0_132:                              ;   in Loop: Header=BB0_125 Depth=1
	; sched_barrier mask(0x00000000)
	s_nop 2
	v_add_f32_dpp v0, v1, v0 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v44, v45, v44 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v40, v41, v40 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v36, v37, v36 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v32, v33, v32 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v28, v29, v28 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v20, v21, v20 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v16, v17, v16 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v12, v13, v12 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v9, v8 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v5, v4 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v2, v0 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v25, v24 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v44, v46, v44 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v40, v42, v40 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v36, v38, v36 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v32, v34, v32 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v28, v30, v28 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v20, v22, v20 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v16, v18, v16 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v12, v14, v12 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v10, v8 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v6, v4 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v26, v2 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v44, v47, v44 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v40, v43, v40 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v36, v39, v36 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v32, v35, v32 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v28, v31, v28 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v20, v23, v20 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v16, v19, v16 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v12, v15, v12 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v11, v8 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v7, v4 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v3, v0 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v27, v2 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v44, v44, v44 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v40, v40, v40 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v36, v36, v36 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v32, v32, v32 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v28, v28, v28 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v20, v20, v20 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v16, v16, v16 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v12, v12, v12 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v8, v8 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v0, v0 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v44, v44, v44 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v40, v40, v40 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v36, v36, v36 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v32, v32, v32 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v28, v28, v28 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v20, v20, v20 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v16, v16, v16 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v12, v12, v12 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v8, v8 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v0, v0 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v44, v44 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v40, v40 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v36, v36 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v32, v32 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v28, v28 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v20, v20 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v16, v16 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v12, v12 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v8, v8 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v4, v4 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v0, v0 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v2, v2 row_shr:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v44, v44, v44 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v40, v40, v40 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v36, v36, v36 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v32, v32, v32 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v28, v28, v28 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v20, v20, v20 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v16, v16, v16 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v13, v12, v12 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v11, v8, v8 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v18, v4, v4 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v0, v0 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v22, v2, v2 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v45, v44 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v41, v40 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v37, v36 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v33, v32 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v29, v28 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v21, v20 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v17, v16 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v14, v13 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v15, v11 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v19, v18 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v1, v0 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v23, v22 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_and_saveexec_b64 s[44:45], s[0:1]
	s_cbranch_execz .LBB0_124
; %bb.133:                              ;   in Loop: Header=BB0_125 Depth=1
	v_cmp_eq_u32_e64 s[8:9], 0, v115
	s_and_b64 s[6:7], s[22:23], s[8:9]
	v_add_f32_e32 v12, v44, v45
	v_add_f32_e32 v10, v40, v41
	v_add_f32_e32 v9, v36, v37
	v_add_f32_e32 v8, v32, v33
	v_add_f32_e32 v7, v28, v29
	v_add_f32_e32 v6, v20, v21
	v_add_f32_e32 v5, v16, v17
	v_add_f32_e32 v4, v13, v14
	v_add_f32_e32 v3, v11, v15
	v_add_f32_e32 v2, v18, v19
	v_add_f32_e32 v1, v0, v1
	v_add_f32_e32 v0, v22, v23
	s_xor_b64 s[16:17], s[6:7], -1
	s_mov_b64 s[6:7], 0
                                        ; implicit-def: $vgpr11
	s_and_saveexec_b64 s[18:19], s[16:17]
	s_xor_b64 s[46:47], exec, s[18:19]
	s_cbranch_execnz .LBB0_136
; %bb.134:                              ;   in Loop: Header=BB0_125 Depth=1
	s_andn2_saveexec_b64 s[8:9], s[46:47]
	s_cbranch_execnz .LBB0_185
.LBB0_135:                              ;   in Loop: Header=BB0_125 Depth=1
	s_or_b64 exec, exec, s[8:9]
	s_and_b64 exec, exec, s[6:7]
	s_cbranch_execz .LBB0_124
	s_branch .LBB0_186
.LBB0_136:                              ;   in Loop: Header=BB0_125 Depth=1
	v_cndmask_b32_e64 v11, 0, 1, s[24:25]
	v_cmp_ne_u32_e64 s[6:7], 1, v11
	s_and_saveexec_b64 s[16:17], s[8:9]
	s_cbranch_execnz .LBB0_147
; %bb.137:                              ;   in Loop: Header=BB0_125 Depth=1
	s_or_b64 exec, exec, s[16:17]
	v_cmp_gt_u32_e64 s[16:17], 2, v115
	s_and_saveexec_b64 s[18:19], s[16:17]
	s_cbranch_execnz .LBB0_150
.LBB0_138:                              ;   in Loop: Header=BB0_125 Depth=1
	s_or_b64 exec, exec, s[18:19]
	v_cmp_gt_u32_e64 s[18:19], 3, v115
	s_and_saveexec_b64 s[48:49], s[18:19]
	s_cbranch_execnz .LBB0_153
.LBB0_139:                              ;   in Loop: Header=BB0_125 Depth=1
	s_or_b64 exec, exec, s[48:49]
	s_and_saveexec_b64 s[48:49], s[8:9]
	s_cbranch_execnz .LBB0_156
.LBB0_140:                              ;   in Loop: Header=BB0_125 Depth=1
	s_or_b64 exec, exec, s[48:49]
	s_and_saveexec_b64 s[48:49], s[16:17]
	s_cbranch_execnz .LBB0_159
.LBB0_141:                              ;   in Loop: Header=BB0_125 Depth=1
	s_or_b64 exec, exec, s[48:49]
	s_and_saveexec_b64 s[48:49], s[18:19]
	s_cbranch_execnz .LBB0_162
.LBB0_142:                              ;   in Loop: Header=BB0_125 Depth=1
	s_or_b64 exec, exec, s[48:49]
	s_and_saveexec_b64 s[48:49], s[8:9]
	s_cbranch_execnz .LBB0_165
.LBB0_143:                              ;   in Loop: Header=BB0_125 Depth=1
	s_or_b64 exec, exec, s[48:49]
	s_and_saveexec_b64 s[48:49], s[16:17]
	s_cbranch_execnz .LBB0_168
.LBB0_144:                              ;   in Loop: Header=BB0_125 Depth=1
	s_or_b64 exec, exec, s[48:49]
	s_and_saveexec_b64 s[48:49], s[18:19]
	s_cbranch_execnz .LBB0_171
.LBB0_145:                              ;   in Loop: Header=BB0_125 Depth=1
	s_or_b64 exec, exec, s[48:49]
	s_and_saveexec_b64 s[48:49], s[8:9]
	s_cbranch_execnz .LBB0_174
.LBB0_146:                              ;   in Loop: Header=BB0_125 Depth=1
	s_or_b64 exec, exec, s[48:49]
	s_and_saveexec_b64 s[8:9], s[16:17]
	s_cbranch_execnz .LBB0_177
	s_branch .LBB0_180
.LBB0_147:                              ;   in Loop: Header=BB0_125 Depth=1
	v_mul_f32_e32 v11, s28, v12
	s_and_b64 vcc, exec, s[6:7]
	v_mov_b32_e32 v105, v107
	s_cbranch_vccnz .LBB0_149
; %bb.148:                              ;   in Loop: Header=BB0_125 Depth=1
	v_lshl_add_u64 v[12:13], v[104:105], 1, s[12:13]
	global_load_ushort v12, v[12:13], off
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v12, 16, v12
	v_fmac_f32_e32 v11, s29, v12
.LBB0_149:                              ;   in Loop: Header=BB0_125 Depth=1
	v_cvt_pk_bf16_f32 v11, v11, s0
	v_lshl_add_u64 v[12:13], v[104:105], 1, s[14:15]
	global_store_short v[12:13], v11, off
	s_or_b64 exec, exec, s[16:17]
	v_cmp_gt_u32_e64 s[16:17], 2, v115
	s_and_saveexec_b64 s[18:19], s[16:17]
	s_cbranch_execz .LBB0_138
.LBB0_150:                              ;   in Loop: Header=BB0_125 Depth=1
	v_mul_f32_e32 v10, s28, v10
	s_and_b64 vcc, exec, s[6:7]
	v_add_u32_e32 v106, 1, v104
	s_cbranch_vccnz .LBB0_152
; %bb.151:                              ;   in Loop: Header=BB0_125 Depth=1
	v_lshl_add_u64 v[12:13], v[106:107], 1, s[12:13]
	global_load_ushort v11, v[12:13], off
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v11, 16, v11
	v_fmac_f32_e32 v10, s29, v11
.LBB0_152:                              ;   in Loop: Header=BB0_125 Depth=1
	v_cvt_pk_bf16_f32 v12, v10, s0
	v_lshl_add_u64 v[10:11], v[106:107], 1, s[14:15]
	global_store_short v[10:11], v12, off
	s_or_b64 exec, exec, s[18:19]
	v_cmp_gt_u32_e64 s[18:19], 3, v115
	s_and_saveexec_b64 s[48:49], s[18:19]
	s_cbranch_execz .LBB0_139
.LBB0_153:                              ;   in Loop: Header=BB0_125 Depth=1
	v_mul_f32_e32 v9, s28, v9
	s_and_b64 vcc, exec, s[6:7]
	v_add_u32_e32 v106, 2, v104
	s_cbranch_vccnz .LBB0_155
; %bb.154:                              ;   in Loop: Header=BB0_125 Depth=1
	v_lshl_add_u64 v[10:11], v[106:107], 1, s[12:13]
	global_load_ushort v10, v[10:11], off
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v10, 16, v10
	v_fmac_f32_e32 v9, s29, v10
.LBB0_155:                              ;   in Loop: Header=BB0_125 Depth=1
	v_cvt_pk_bf16_f32 v9, v9, s0
	v_lshl_add_u64 v[10:11], v[106:107], 1, s[14:15]
	global_store_short v[10:11], v9, off
	s_or_b64 exec, exec, s[48:49]
	s_and_saveexec_b64 s[48:49], s[8:9]
	s_cbranch_execz .LBB0_140
.LBB0_156:                              ;   in Loop: Header=BB0_125 Depth=1
	v_mul_f32_e32 v8, s28, v8
	s_and_b64 vcc, exec, s[6:7]
	v_mov_b32_e32 v105, v107
	s_cbranch_vccnz .LBB0_158
; %bb.157:                              ;   in Loop: Header=BB0_125 Depth=1
	v_lshl_add_u64 v[10:11], v[104:105], 1, s[26:27]
	global_load_ushort v9, v[10:11], off
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v9, 16, v9
	v_fmac_f32_e32 v8, s29, v9
.LBB0_158:                              ;   in Loop: Header=BB0_125 Depth=1
	v_cvt_pk_bf16_f32 v10, v8, s0
	v_lshl_add_u64 v[8:9], v[104:105], 1, s[30:31]
	global_store_short v[8:9], v10, off
	s_or_b64 exec, exec, s[48:49]
	s_and_saveexec_b64 s[48:49], s[16:17]
	s_cbranch_execz .LBB0_141
.LBB0_159:                              ;   in Loop: Header=BB0_125 Depth=1
	v_mul_f32_e32 v7, s28, v7
	s_and_b64 vcc, exec, s[6:7]
	v_add_u32_e32 v106, 1, v104
	s_cbranch_vccnz .LBB0_161
; %bb.160:                              ;   in Loop: Header=BB0_125 Depth=1
	v_lshl_add_u64 v[8:9], v[106:107], 1, s[26:27]
	global_load_ushort v8, v[8:9], off
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v8, 16, v8
	v_fmac_f32_e32 v7, s29, v8
.LBB0_161:                              ;   in Loop: Header=BB0_125 Depth=1
	v_cvt_pk_bf16_f32 v7, v7, s0
	v_lshl_add_u64 v[8:9], v[106:107], 1, s[30:31]
	global_store_short v[8:9], v7, off
	s_or_b64 exec, exec, s[48:49]
	s_and_saveexec_b64 s[48:49], s[18:19]
	s_cbranch_execz .LBB0_142
.LBB0_162:                              ;   in Loop: Header=BB0_125 Depth=1
	v_mul_f32_e32 v6, s28, v6
	s_and_b64 vcc, exec, s[6:7]
	v_add_u32_e32 v106, 2, v104
	s_cbranch_vccnz .LBB0_164
; %bb.163:                              ;   in Loop: Header=BB0_125 Depth=1
	v_lshl_add_u64 v[8:9], v[106:107], 1, s[26:27]
	global_load_ushort v7, v[8:9], off
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v7, 16, v7
	v_fmac_f32_e32 v6, s29, v7
.LBB0_164:                              ;   in Loop: Header=BB0_125 Depth=1
	v_cvt_pk_bf16_f32 v8, v6, s0
	v_lshl_add_u64 v[6:7], v[106:107], 1, s[30:31]
	global_store_short v[6:7], v8, off
	s_or_b64 exec, exec, s[48:49]
	s_and_saveexec_b64 s[48:49], s[8:9]
	s_cbranch_execz .LBB0_143
.LBB0_165:                              ;   in Loop: Header=BB0_125 Depth=1
	v_mul_f32_e32 v5, s28, v5
	s_and_b64 vcc, exec, s[6:7]
	v_mov_b32_e32 v105, v107
	s_cbranch_vccnz .LBB0_167
; %bb.166:                              ;   in Loop: Header=BB0_125 Depth=1
	v_lshl_add_u64 v[6:7], v[104:105], 1, s[34:35]
	global_load_ushort v6, v[6:7], off
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v6, 16, v6
	v_fmac_f32_e32 v5, s29, v6
.LBB0_167:                              ;   in Loop: Header=BB0_125 Depth=1
	v_cvt_pk_bf16_f32 v5, v5, s0
	v_lshl_add_u64 v[6:7], v[104:105], 1, s[36:37]
	global_store_short v[6:7], v5, off
	s_or_b64 exec, exec, s[48:49]
	s_and_saveexec_b64 s[48:49], s[16:17]
	s_cbranch_execz .LBB0_144
.LBB0_168:                              ;   in Loop: Header=BB0_125 Depth=1
	v_mul_f32_e32 v4, s28, v4
	s_and_b64 vcc, exec, s[6:7]
	v_add_u32_e32 v106, 1, v104
	s_cbranch_vccnz .LBB0_170
; %bb.169:                              ;   in Loop: Header=BB0_125 Depth=1
	v_lshl_add_u64 v[6:7], v[106:107], 1, s[34:35]
	global_load_ushort v5, v[6:7], off
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v5, 16, v5
	v_fmac_f32_e32 v4, s29, v5
.LBB0_170:                              ;   in Loop: Header=BB0_125 Depth=1
	v_cvt_pk_bf16_f32 v6, v4, s0
	v_lshl_add_u64 v[4:5], v[106:107], 1, s[36:37]
	global_store_short v[4:5], v6, off
	s_or_b64 exec, exec, s[48:49]
	s_and_saveexec_b64 s[48:49], s[18:19]
	s_cbranch_execz .LBB0_145
.LBB0_171:                              ;   in Loop: Header=BB0_125 Depth=1
	v_mul_f32_e32 v3, s28, v3
	s_and_b64 vcc, exec, s[6:7]
	v_add_u32_e32 v106, 2, v104
	s_cbranch_vccnz .LBB0_173
; %bb.172:                              ;   in Loop: Header=BB0_125 Depth=1
	v_lshl_add_u64 v[4:5], v[106:107], 1, s[34:35]
	global_load_ushort v4, v[4:5], off
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v4, 16, v4
	v_fmac_f32_e32 v3, s29, v4
.LBB0_173:                              ;   in Loop: Header=BB0_125 Depth=1
	v_cvt_pk_bf16_f32 v3, v3, s0
	v_lshl_add_u64 v[4:5], v[106:107], 1, s[36:37]
	global_store_short v[4:5], v3, off
	s_or_b64 exec, exec, s[48:49]
	s_and_saveexec_b64 s[48:49], s[8:9]
	s_cbranch_execz .LBB0_146
.LBB0_174:                              ;   in Loop: Header=BB0_125 Depth=1
	v_mul_f32_e32 v2, s28, v2
	s_and_b64 vcc, exec, s[6:7]
	v_mov_b32_e32 v105, v107
	s_cbranch_vccnz .LBB0_176
; %bb.175:                              ;   in Loop: Header=BB0_125 Depth=1
	v_lshl_add_u64 v[4:5], v[104:105], 1, s[38:39]
	global_load_ushort v3, v[4:5], off
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v3, 16, v3
	v_fmac_f32_e32 v2, s29, v3
.LBB0_176:                              ;   in Loop: Header=BB0_125 Depth=1
	v_cvt_pk_bf16_f32 v4, v2, s0
	v_lshl_add_u64 v[2:3], v[104:105], 1, s[40:41]
	global_store_short v[2:3], v4, off
	s_or_b64 exec, exec, s[48:49]
	s_and_saveexec_b64 s[8:9], s[16:17]
	s_cbranch_execz .LBB0_180
.LBB0_177:                              ;   in Loop: Header=BB0_125 Depth=1
	v_mul_f32_e32 v1, s28, v1
	s_and_b64 vcc, exec, s[6:7]
	v_add_u32_e32 v106, 1, v104
	s_cbranch_vccnz .LBB0_179
; %bb.178:                              ;   in Loop: Header=BB0_125 Depth=1
	v_lshl_add_u64 v[2:3], v[106:107], 1, s[38:39]
	global_load_ushort v2, v[2:3], off
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v2, 16, v2
	v_fmac_f32_e32 v1, s29, v2
.LBB0_179:                              ;   in Loop: Header=BB0_125 Depth=1
	v_cvt_pk_bf16_f32 v1, v1, s0
	v_lshl_add_u64 v[2:3], v[106:107], 1, s[40:41]
	global_store_short v[2:3], v1, off
.LBB0_180:                              ;   in Loop: Header=BB0_125 Depth=1
	s_or_b64 exec, exec, s[8:9]
	s_mov_b64 s[16:17], 0
                                        ; implicit-def: $vgpr11
	s_and_saveexec_b64 s[8:9], s[18:19]
	s_xor_b64 s[8:9], exec, s[8:9]
	s_cbranch_execz .LBB0_184
; %bb.181:                              ;   in Loop: Header=BB0_125 Depth=1
	v_mul_f32_e32 v11, s28, v0
	s_and_b64 vcc, exec, s[6:7]
	v_add_u32_e32 v106, 2, v104
	s_cbranch_vccnz .LBB0_183
; %bb.182:                              ;   in Loop: Header=BB0_125 Depth=1
	v_lshl_add_u64 v[0:1], v[106:107], 1, s[38:39]
	global_load_ushort v0, v[0:1], off
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v0, 16, v0
	v_fmac_f32_e32 v11, s29, v0
.LBB0_183:                              ;   in Loop: Header=BB0_125 Depth=1
	s_mov_b64 s[16:17], exec
.LBB0_184:                              ;   in Loop: Header=BB0_125 Depth=1
	s_or_b64 exec, exec, s[8:9]
	s_and_b64 s[6:7], s[16:17], exec
                                        ; implicit-def: $vgpr12
                                        ; implicit-def: $vgpr10
                                        ; implicit-def: $vgpr9
                                        ; implicit-def: $vgpr8
                                        ; implicit-def: $vgpr7
                                        ; implicit-def: $vgpr6
                                        ; implicit-def: $vgpr5
                                        ; implicit-def: $vgpr4
                                        ; implicit-def: $vgpr3
                                        ; implicit-def: $vgpr2
                                        ; implicit-def: $vgpr1
                                        ; implicit-def: $vgpr0
	s_andn2_saveexec_b64 s[8:9], s[46:47]
	s_cbranch_execz .LBB0_135
.LBB0_185:                              ;   in Loop: Header=BB0_125 Depth=1
	v_mov_b32_e32 v105, v107
	v_mul_f32_e32 v11, s28, v12
	v_lshlrev_b64 v[12:13], 1, v[104:105]
	v_cvt_pk_bf16_f32 v11, v11, s0
	v_lshl_add_u64 v[14:15], s[14:15], 0, v[12:13]
	v_mul_f32_e32 v10, s28, v10
	v_add_u32_e32 v106, 1, v104
	global_store_short v[14:15], v11, off
	v_cvt_pk_bf16_f32 v16, v10, s0
	v_lshlrev_b64 v[10:11], 1, v[106:107]
	v_lshl_add_u64 v[14:15], s[14:15], 0, v[10:11]
	v_add_u32_e32 v106, 2, v104
	global_store_short v[14:15], v16, off
	v_mul_f32_e32 v9, s28, v9
	v_lshlrev_b64 v[14:15], 1, v[106:107]
	v_cvt_pk_bf16_f32 v9, v9, s0
	v_lshl_add_u64 v[16:17], s[14:15], 0, v[14:15]
	v_mul_f32_e32 v8, s28, v8
	global_store_short v[16:17], v9, off
	v_cvt_pk_bf16_f32 v16, v8, s0
	v_lshl_add_u64 v[8:9], s[30:31], 0, v[12:13]
	v_mul_f32_e32 v7, s28, v7
	global_store_short v[8:9], v16, off
	v_cvt_pk_bf16_f32 v7, v7, s0
	v_lshl_add_u64 v[8:9], s[30:31], 0, v[10:11]
	v_mul_f32_e32 v6, s28, v6
	global_store_short v[8:9], v7, off
	v_cvt_pk_bf16_f32 v8, v6, s0
	v_lshl_add_u64 v[6:7], s[30:31], 0, v[14:15]
	v_mul_f32_e32 v5, s28, v5
	global_store_short v[6:7], v8, off
	v_cvt_pk_bf16_f32 v5, v5, s0
	v_lshl_add_u64 v[6:7], s[36:37], 0, v[12:13]
	v_mul_f32_e32 v4, s28, v4
	global_store_short v[6:7], v5, off
	v_cvt_pk_bf16_f32 v6, v4, s0
	v_lshl_add_u64 v[4:5], s[36:37], 0, v[10:11]
	v_mul_f32_e32 v3, s28, v3
	global_store_short v[4:5], v6, off
	v_cvt_pk_bf16_f32 v3, v3, s0
	v_lshl_add_u64 v[4:5], s[36:37], 0, v[14:15]
	v_mul_f32_e32 v2, s28, v2
	global_store_short v[4:5], v3, off
	v_cvt_pk_bf16_f32 v4, v2, s0
	v_lshl_add_u64 v[2:3], s[40:41], 0, v[12:13]
	v_mul_f32_e32 v1, s28, v1
	global_store_short v[2:3], v4, off
	v_cvt_pk_bf16_f32 v1, v1, s0
	v_lshl_add_u64 v[2:3], s[40:41], 0, v[10:11]
	global_store_short v[2:3], v1, off
	v_mul_f32_e32 v11, s28, v0
	s_or_b64 s[6:7], s[6:7], exec
	s_or_b64 exec, exec, s[8:9]
	s_and_b64 exec, exec, s[6:7]
	s_cbranch_execz .LBB0_124
.LBB0_186:                              ;   in Loop: Header=BB0_125 Depth=1
	v_cvt_pk_bf16_f32 v2, v11, s0
	v_lshl_add_u64 v[0:1], v[106:107], 1, s[40:41]
	global_store_short v[0:1], v2, off
	s_branch .LBB0_124
.LBB0_187:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wvSplitK_vllm_bf16_tn_n4_y3u2
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
		.amdhsa_next_free_vgpr 123
		.amdhsa_next_free_sgpr 96
		.amdhsa_accum_offset 124
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
	.size	wvSplitK_vllm_bf16_tn_n4_y3u2, .Lfunc_end0-wvSplitK_vllm_bf16_tn_n4_y3u2
                                        ; -- End function
	.set wvSplitK_vllm_bf16_tn_n4_y3u2.num_vgpr, 123
	.set wvSplitK_vllm_bf16_tn_n4_y3u2.num_agpr, 0
	.set wvSplitK_vllm_bf16_tn_n4_y3u2.numbered_sgpr, 77
	.set wvSplitK_vllm_bf16_tn_n4_y3u2.num_named_barrier, 0
	.set wvSplitK_vllm_bf16_tn_n4_y3u2.private_seg_size, 0
	.set wvSplitK_vllm_bf16_tn_n4_y3u2.uses_vcc, 1
	.set wvSplitK_vllm_bf16_tn_n4_y3u2.uses_flat_scratch, 0
	.set wvSplitK_vllm_bf16_tn_n4_y3u2.has_dyn_sized_stack, 0
	.set wvSplitK_vllm_bf16_tn_n4_y3u2.has_recursion, 0
	.set wvSplitK_vllm_bf16_tn_n4_y3u2.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 10052
; TotalNumSgprs: 83
; NumVgprs: 123
; NumAgprs: 0
; TotalNumVgprs: 123
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 163840 bytes/workgroup (compile time only)
; SGPRBlocks: 12
; VGPRBlocks: 15
; NumSGPRsForWavesPerEU: 102
; NumVGPRsForWavesPerEU: 123
; AccumOffset: 124
; Occupancy: 4
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 1
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 30
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_bed05943042ecda5,@object ; @__hip_cuid_bed05943042ecda5
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_bed05943042ecda5
__hip_cuid_bed05943042ecda5:
	.byte	0                               ; 0x0
	.size	__hip_cuid_bed05943042ecda5, 1

	.ident	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.4 26084 f58b06dce1f9c15707c5f808fd002e18c2accf7e)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_bed05943042ecda5
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
    .name:           wvSplitK_vllm_bf16_tn_n4_y3u2
    .private_segment_fixed_size: 0
    .sgpr_count:     83
    .sgpr_spill_count: 0
    .symbol:         wvSplitK_vllm_bf16_tn_n4_y3u2.kd
    .vgpr_count:     123
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 1
...

	.end_amdgpu_metadata
