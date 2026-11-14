	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 4
	.text
	.protected	wvSplitKQ_vllm_f8_f16_tn_n3_y2u1 ; -- Begin function wvSplitKQ_vllm_f8_f16_tn_n3_y2u1
	.globl	wvSplitKQ_vllm_f8_f16_tn_n3_y2u1
	.p2align	8
	.type	wvSplitKQ_vllm_f8_f16_tn_n3_y2u1,@function
wvSplitKQ_vllm_f8_f16_tn_n3_y2u1:       ; @wvSplitKQ_vllm_f8_f16_tn_n3_y2u1
; %bb.98:
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.99:
.LBB0_0:
	s_mov_b32 s24, s11
	s_mov_b32 s11, s3
	s_mov_b32 s3, 0
	s_cmp_lt_u32 s2, 0x28000
	s_cselect_b32 s48, s2, 0x28000
	s_and_b32 s18, s6, 15
	s_mov_b32 s19, s3
	s_load_dwordx4 s[20:23], s[0:1], 0x48
	s_cmp_lg_u64 s[18:19], 0
	v_and_b32_e32 v47, 0x3ff, v0
	s_cselect_b64 s[18:19], -1, 0
	s_and_b32 s9, s24, 15
	v_bfe_u32 v6, v0, 10, 10
	s_cmp_lg_u32 s9, 0
	v_lshlrev_b32_e32 v44, 4, v47
	s_cselect_b64 s[28:29], -1, 0
	v_lshl_add_u32 v0, v6, 10, v44
	s_mov_b64 s[26:27], 0
	s_or_b64 s[18:19], s[18:19], s[28:29]
	v_cmp_gt_u32_e32 vcc, s48, v0
	s_and_saveexec_b64 s[28:29], vcc
	s_cbranch_execz .LBB0_7
; %bb.1:
	v_mov_b32_e32 v3, 0
	v_lshlrev_b32_e32 v2, 10, v6
	v_mov_b32_e32 v45, v3
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[44:45]
	v_lshl_add_u64 v[2:3], s[6:7], 0, v[2:3]
	s_mov_b64 s[30:31], 0x4000
	v_mov_b32_e32 v1, v0
	s_branch .LBB0_3
.LBB0_2:                                ;   in Loop: Header=BB0_3 Depth=1
	v_add_u32_e32 v1, 0x4000, v1
	v_cmp_le_u32_e32 vcc, s48, v1
	s_or_b64 s[26:27], vcc, s[26:27]
	v_lshl_add_u64 v[2:3], v[2:3], 0, s[30:31]
	s_andn2_b64 exec, exec, s[26:27]
	s_cbranch_execz .LBB0_7
.LBB0_3:                                ; =>This Inner Loop Header: Depth=1
	s_mov_b64 s[34:35], -1
	s_and_b64 vcc, exec, s[18:19]
	s_cbranch_vccz .LBB0_5
; %bb.4:                                ;   in Loop: Header=BB0_3 Depth=1
	global_load_dwordx4 v[8:11], v[2:3], off
	s_mov_b64 s[34:35], 0
	s_waitcnt vmcnt(0)
	ds_write_b128 v1, v[8:11]
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
	s_cselect_b32 s9, s26, s2
	s_cmp_lt_u32 s2, 0x28000
	s_cselect_b32 s49, s9, 0
	s_add_u32 s26, s6, s24
	v_cndmask_b32_e64 v1, 0, 1, s[18:19]
	s_mov_b32 s25, 0
	s_addc_u32 s27, s7, 0
	v_cmp_gt_u32_e32 vcc, s49, v0
	v_cmp_ne_u32_e64 s[18:19], 1, v1
	s_and_saveexec_b64 s[28:29], vcc
	s_cbranch_execz .LBB0_14
; %bb.8:
	s_mov_b64 s[30:31], 0
	v_mov_b32_e32 v3, 0
	v_mov_b32_e32 v2, v0
	s_branch .LBB0_10
.LBB0_9:                                ;   in Loop: Header=BB0_10 Depth=1
	v_add_u32_e32 v2, 0x4000, v2
	v_cmp_le_u32_e32 vcc, s49, v2
	s_or_b64 s[30:31], vcc, s[30:31]
	s_andn2_b64 exec, exec, s[30:31]
	s_cbranch_execz .LBB0_14
.LBB0_10:                               ; =>This Inner Loop Header: Depth=1
	v_lshl_add_u64 v[4:5], s[26:27], 0, v[2:3]
	v_add_u32_e32 v1, s2, v2
	s_and_b64 vcc, exec, s[18:19]
	s_mov_b64 s[34:35], -1
	s_cbranch_vccnz .LBB0_12
; %bb.11:                               ;   in Loop: Header=BB0_10 Depth=1
	global_load_dwordx4 v[8:11], v[4:5], off
	s_mov_b64 s[34:35], 0
	s_waitcnt vmcnt(0)
	ds_write_b128 v1, v[8:11]
.LBB0_12:                               ;   in Loop: Header=BB0_10 Depth=1
	s_andn2_b64 vcc, exec, s[34:35]
	s_cbranch_vccnz .LBB0_9
; %bb.13:                               ;   in Loop: Header=BB0_10 Depth=1
	v_readfirstlane_b32 s9, v1
	s_mov_b32 m0, s9
	s_nop 0
	global_load_lds_dwordx4 v[4:5], off
	s_branch .LBB0_9
.LBB0_14:
	s_or_b64 exec, exec, s[28:29]
	s_lshl_b64 s[28:29], s[2:3], 1
	s_sub_u32 s28, 0x28000, s28
	s_subb_u32 s29, 0, s29
	v_mov_b64_e32 v[2:3], s[2:3]
	v_cmp_lt_u64_e32 vcc, s[28:29], v[2:3]
	s_and_b64 s[30:31], vcc, exec
	s_cselect_b32 s3, s28, s2
	s_cmp_lt_u32 s2, 0x14000
	s_cselect_b32 s51, s3, 0
	s_lshl_b32 s50, s2, 1
	v_cmp_gt_u32_e32 vcc, s51, v0
	s_and_saveexec_b64 s[28:29], vcc
	s_cbranch_execz .LBB0_21
; %bb.15:
	s_add_u32 s26, s26, s24
	s_addc_u32 s27, s27, 0
	s_mov_b64 s[30:31], 0
	v_mov_b32_e32 v1, 0
	s_branch .LBB0_17
.LBB0_16:                               ;   in Loop: Header=BB0_17 Depth=1
	v_add_u32_e32 v0, 0x4000, v0
	v_cmp_le_u32_e32 vcc, s51, v0
	s_or_b64 s[30:31], vcc, s[30:31]
	s_andn2_b64 exec, exec, s[30:31]
	s_cbranch_execz .LBB0_21
.LBB0_17:                               ; =>This Inner Loop Header: Depth=1
	v_lshl_add_u64 v[2:3], s[26:27], 0, v[0:1]
	v_add_u32_e32 v4, s50, v0
	s_and_b64 vcc, exec, s[18:19]
	s_mov_b64 s[34:35], -1
	s_cbranch_vccnz .LBB0_19
; %bb.18:                               ;   in Loop: Header=BB0_17 Depth=1
	global_load_dwordx4 v[8:11], v[2:3], off
	s_mov_b64 s[34:35], 0
	s_waitcnt vmcnt(0)
	ds_write_b128 v4, v[8:11]
.LBB0_19:                               ;   in Loop: Header=BB0_17 Depth=1
	s_andn2_b64 vcc, exec, s[34:35]
	s_cbranch_vccnz .LBB0_16
; %bb.20:                               ;   in Loop: Header=BB0_17 Depth=1
	v_readfirstlane_b32 s3, v4
	s_mov_b32 m0, s3
	s_nop 0
	global_load_lds_dwordx4 v[2:3], off
	s_branch .LBB0_16
.LBB0_21:
	s_or_b64 exec, exec, s[28:29]
	s_add_i32 s3, s11, -1
	s_lshl_b32 s17, s8, 5
	v_mov_b32_e32 v0, s3
	v_add_co_u32_e32 v0, vcc, s17, v0
	s_nop 0
	v_readfirstlane_b32 s18, v0
	s_andn2_b64 vcc, exec, vcc
	s_mov_b32 s9, 4
	s_cbranch_vccz .LBB0_23
; %bb.22:
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
.LBB0_23:
	s_mov_b32 s17, 1.0
	s_waitcnt lgkmcnt(0)
	s_cmp_eq_u64 s[20:21], 0
	s_mov_b32 s18, 1.0
	s_cbranch_scc1 .LBB0_25
; %bb.24:
	s_load_dword s18, s[20:21], 0x0
.LBB0_25:
	s_cmp_eq_u64 s[22:23], 0
	s_cbranch_scc1 .LBB0_27
; %bb.26:
	s_load_dword s17, s[22:23], 0x0
.LBB0_27:
	v_cmp_gt_u32_e32 vcc, s9, v6
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_97
; %bb.28:
	s_cmp_eq_u32 s49, s2
	s_cselect_b64 s[20:21], -1, 0
	s_cmp_eq_u32 s51, s2
	s_cselect_b64 s[22:23], -1, 0
	s_cmp_eq_u32 s2, s48
	s_cselect_b64 s[26:27], -1, 0
	s_and_b64 s[20:21], s[26:27], s[20:21]
	s_and_b64 s[26:27], s[20:21], s[22:23]
	s_load_dwordx4 s[20:23], s[0:1], 0x38
	v_mov_b32_e32 v0, s18
	s_mul_i32 s0, s9, s16
	s_mul_i32 s8, s8, s9
	v_add_lshl_u32 v46, s0, v6, 1
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v0, s20, v0
	s_lshl_b32 s33, s8, 1
	v_mul_f32_e32 v45, s17, v0
	s_mov_b64 s[8:9], -1
	s_andn2_b64 vcc, exec, s[26:27]
	v_cmp_gt_u32_e64 s[0:1], s11, v46
	s_cbranch_vccz .LBB0_69
; %bb.29:
	s_and_saveexec_b64 s[26:27], s[0:1]
	s_cbranch_execz .LBB0_68
; %bb.30:
	v_mbcnt_lo_u32_b32 v0, -1, 0
	v_mbcnt_hi_u32_b32 v0, -1, v0
	v_and_b32_e32 v1, 63, v0
	s_lshl_b64 s[28:29], s[24:25], 1
	v_cmp_gt_u32_e32 vcc, 44, v1
	s_cmp_lg_u32 s2, 0
	s_mov_b32 s31, 0
	v_cndmask_b32_e64 v2, 0, 20, vcc
	v_cmp_gt_u32_e32 vcc, 24, v1
	s_cselect_b64 s[8:9], -1, 0
	v_add_lshl_u32 v58, v2, v0, 2
	v_cndmask_b32_e64 v1, 0, 40, vcc
	v_add_lshl_u32 v59, v1, v0, 2
	s_mov_b32 s30, s22
	s_mov_b32 s36, s23
	s_mov_b32 s37, s31
	v_cndmask_b32_e64 v0, 0, 1, s[8:9]
	s_add_i32 s20, s2, -16
	v_cmp_eq_u32_e64 s[0:1], 0, v47
	v_cmp_neq_f32_e64 s[34:35], s21, 0
	s_lshl_b64 s[38:39], s[30:31], 1
	s_lshl_b64 s[40:41], s[36:37], 1
	v_add_u32_e32 v60, s2, v44
	v_add_u32_e32 v61, s50, v44
	s_mov_b64 s[42:43], 0
	v_cmp_ne_u32_e64 s[8:9], 1, v0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v50, v46
	s_branch .LBB0_33
.LBB0_31:                               ;   in Loop: Header=BB0_33 Depth=1
	v_cvt_f16_f32_e32 v0, v4
	global_store_short v[2:3], v0, off offset:2
.LBB0_32:                               ;   in Loop: Header=BB0_33 Depth=1
	s_or_b64 exec, exec, s[44:45]
	v_add_u32_e32 v50, s33, v50
	v_cmp_le_u32_e32 vcc, s11, v50
	s_or_b64 s[42:43], vcc, s[42:43]
	s_andn2_b64 exec, exec, s[42:43]
	s_cbranch_execz .LBB0_68
.LBB0_33:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_37 Depth 2
	s_and_b64 vcc, exec, s[8:9]
	v_or_b32_e32 v62, 1, v50
	v_mov_b32_e32 v23, v49
	v_mov_b32_e32 v22, v49
	v_mov_b32_e32 v21, v49
	v_mov_b32_e32 v20, v49
	s_waitcnt lgkmcnt(5)
	v_mov_b32_e32 v19, v49
	v_mov_b32_e32 v18, v49
	s_waitcnt lgkmcnt(4)
	v_mov_b32_e32 v17, v49
	v_mov_b32_e32 v16, v49
	s_waitcnt lgkmcnt(3)
	v_mov_b32_e32 v11, v49
	v_mov_b32_e32 v10, v49
	s_waitcnt lgkmcnt(2)
	v_mov_b32_e32 v9, v49
	v_mov_b32_e32 v8, v49
	s_waitcnt lgkmcnt(1)
	v_mov_b32_e32 v7, v49
	v_mov_b32_e32 v6, v49
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v5, v49
	v_mov_b32_e32 v4, v49
	v_mov_b32_e32 v3, v49
	v_mov_b32_e32 v2, v49
	v_mov_b32_e32 v1, v49
	v_mov_b32_e32 v0, v49
	v_mov_b32_e32 v15, v49
	v_mov_b32_e32 v14, v49
	v_mov_b32_e32 v13, v49
	v_mov_b32_e32 v12, v49
	s_cbranch_vccnz .LBB0_50
; %bb.34:                               ;   in Loop: Header=BB0_33 Depth=1
	v_min_u32_e32 v0, s3, v50
	v_mad_u64_u32 v[52:53], s[16:17], v0, s10, 0
	v_min_u32_e32 v0, s3, v62
	v_mov_b32_e32 v12, 0
	v_mad_u64_u32 v[54:55], s[16:17], v0, s10, 0
	s_mov_b32 s44, 0
	v_mov_b32_e32 v13, v12
	v_mov_b32_e32 v14, v12
	v_mov_b32_e32 v15, v12
	v_mov_b32_e32 v0, v12
	v_mov_b32_e32 v1, v12
	v_mov_b32_e32 v2, v12
	v_mov_b32_e32 v3, v12
	v_mov_b32_e32 v4, v12
	v_mov_b32_e32 v5, v12
	v_mov_b32_e32 v6, v12
	v_mov_b32_e32 v7, v12
	v_mov_b32_e32 v8, v12
	v_mov_b32_e32 v9, v12
	v_mov_b32_e32 v10, v12
	v_mov_b32_e32 v11, v12
	v_mov_b32_e32 v16, v12
	v_mov_b32_e32 v17, v12
	v_mov_b32_e32 v18, v12
	v_mov_b32_e32 v19, v12
	v_mov_b32_e32 v20, v12
	v_mov_b32_e32 v21, v12
	v_mov_b32_e32 v22, v12
	v_mov_b32_e32 v23, v12
	s_branch .LBB0_37
.LBB0_35:                               ;   in Loop: Header=BB0_37 Depth=2
	s_or_b64 exec, exec, s[18:19]
.LBB0_36:                               ;   in Loop: Header=BB0_37 Depth=2
	s_or_b64 exec, exec, s[16:17]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_mfma_f32_16x16x32_fp8_fp8 v[20:23], v[32:33], v[28:29], v[20:23]
	s_addk_i32 s44, 0x400
	s_cmp_ge_u32 s44, s2
	v_mfma_f32_16x16x32_fp8_fp8 v[16:19], v[32:33], v[24:25], v[16:19]
	v_mfma_f32_16x16x32_fp8_fp8 v[8:11], v[36:37], v[28:29], v[8:11]
	v_mfma_f32_16x16x32_fp8_fp8 v[4:7], v[36:37], v[24:25], v[4:7]
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[40:41], v[28:29], v[0:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[12:15], v[40:41], v[24:25], v[12:15]
	v_mfma_f32_16x16x32_fp8_fp8 v[20:23], v[34:35], v[30:31], v[20:23]
	v_mfma_f32_16x16x32_fp8_fp8 v[16:19], v[34:35], v[26:27], v[16:19]
	v_mfma_f32_16x16x32_fp8_fp8 v[8:11], v[38:39], v[30:31], v[8:11]
	v_mfma_f32_16x16x32_fp8_fp8 v[4:7], v[38:39], v[26:27], v[4:7]
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[42:43], v[30:31], v[0:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[12:15], v[42:43], v[26:27], v[12:15]
	s_cbranch_scc1 .LBB0_50
.LBB0_37:                               ;   Parent Loop BB0_33 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_add_u32_e32 v56, s44, v44
	v_min_u32_e32 v48, s20, v56
	v_lshl_add_u64 v[24:25], s[4:5], 0, v[48:49]
	v_lshl_add_u64 v[32:33], v[24:25], 0, v[52:53]
	v_lshl_add_u64 v[34:35], v[24:25], 0, v[54:55]
	global_load_dwordx4 v[28:31], v[32:33], off nt
	global_load_dwordx4 v[24:27], v[34:35], off nt
	v_cmp_gt_u32_e32 vcc, s2, v56
	v_mov_b64_e32 v[42:43], 0
	v_mov_b64_e32 v[40:41], 0
	v_mov_b64_e32 v[38:39], 0
	v_mov_b64_e32 v[36:37], 0
	v_mov_b64_e32 v[34:35], 0
	v_mov_b64_e32 v[32:33], 0
	s_and_saveexec_b64 s[16:17], vcc
	s_cbranch_execz .LBB0_36
; %bb.38:                               ;   in Loop: Header=BB0_37 Depth=2
	v_mov_b32_e32 v57, v49
	v_lshl_add_u64 v[40:41], s[6:7], 0, v[56:57]
	v_cmp_le_u32_e32 vcc, s48, v56
                                        ; implicit-def: $vgpr34_vgpr35
	s_and_saveexec_b64 s[18:19], vcc
	s_xor_b64 s[18:19], exec, s[18:19]
	s_cbranch_execz .LBB0_40
; %bb.39:                               ;   in Loop: Header=BB0_37 Depth=2
	global_load_dwordx4 v[32:35], v[40:41], off
.LBB0_40:                               ;   in Loop: Header=BB0_37 Depth=2
	s_andn2_saveexec_b64 s[18:19], s[18:19]
	s_cbranch_execz .LBB0_42
; %bb.41:                               ;   in Loop: Header=BB0_37 Depth=2
	s_waitcnt vmcnt(0)
	ds_read_b128 v[32:35], v56
.LBB0_42:                               ;   in Loop: Header=BB0_37 Depth=2
	s_or_b64 exec, exec, s[18:19]
	v_cmp_le_u32_e32 vcc, s49, v56
                                        ; implicit-def: $vgpr38_vgpr39
	s_and_saveexec_b64 s[18:19], vcc
	s_xor_b64 s[18:19], exec, s[18:19]
	s_cbranch_execz .LBB0_44
; %bb.43:                               ;   in Loop: Header=BB0_37 Depth=2
	v_lshl_add_u64 v[36:37], v[40:41], 0, s[24:25]
	global_load_dwordx4 v[36:39], v[36:37], off
.LBB0_44:                               ;   in Loop: Header=BB0_37 Depth=2
	s_andn2_saveexec_b64 s[18:19], s[18:19]
	s_cbranch_execz .LBB0_46
; %bb.45:                               ;   in Loop: Header=BB0_37 Depth=2
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v36, s44, v60
	ds_read_b128 v[36:39], v36
.LBB0_46:                               ;   in Loop: Header=BB0_37 Depth=2
	s_or_b64 exec, exec, s[18:19]
	v_cmp_le_u32_e32 vcc, s51, v56
                                        ; implicit-def: $vgpr42_vgpr43
	s_and_saveexec_b64 s[18:19], vcc
	s_xor_b64 s[18:19], exec, s[18:19]
	s_cbranch_execz .LBB0_48
; %bb.47:                               ;   in Loop: Header=BB0_37 Depth=2
	v_lshl_add_u64 v[40:41], v[40:41], 0, s[28:29]
	global_load_dwordx4 v[40:43], v[40:41], off
.LBB0_48:                               ;   in Loop: Header=BB0_37 Depth=2
	s_andn2_saveexec_b64 s[18:19], s[18:19]
	s_cbranch_execz .LBB0_35
; %bb.49:                               ;   in Loop: Header=BB0_37 Depth=2
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v40, s44, v61
	ds_read_b128 v[40:43], v40
	s_branch .LBB0_35
.LBB0_50:                               ;   in Loop: Header=BB0_33 Depth=1
	s_nop 5
	v_add_f32_dpp v0, v1, v0 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v20, v21, v20 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v16, v17, v16 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v9, v8 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v5, v4 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v2, v0 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v13, v12 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v20, v22, v20 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v16, v18, v16 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v10, v8 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v6, v4 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v14, v2 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v20, v23, v20 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v16, v19, v16 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v11, v8 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v7, v4 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v3, v0 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v15, v2 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	ds_bpermute_b32 v21, v58, v20
	ds_bpermute_b32 v17, v58, v16
	ds_bpermute_b32 v9, v58, v8
	ds_bpermute_b32 v5, v58, v4
	ds_bpermute_b32 v1, v58, v0
	ds_bpermute_b32 v3, v58, v2
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v18, v20, v21
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v16, v16, v17
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v10, v8, v9
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v8, v4, v5
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v6, v0, v1
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v4, v2, v3
	ds_bpermute_b32 v19, v59, v18
	ds_bpermute_b32 v17, v59, v16
	ds_bpermute_b32 v11, v59, v10
	ds_bpermute_b32 v9, v59, v8
	ds_bpermute_b32 v7, v59, v6
	ds_bpermute_b32 v5, v59, v4
	s_and_saveexec_b64 s[44:45], s[0:1]
	s_cbranch_execz .LBB0_32
; %bb.51:                               ;   in Loop: Header=BB0_33 Depth=1
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v2, v18, v19
	v_mov_b32_e32 v51, v49
	v_cndmask_b32_e64 v3, 0, 1, s[34:35]
	v_lshl_add_u64 v[0:1], v[50:51], 1, s[12:13]
	v_cmp_ne_u32_e64 s[18:19], 1, v3
	s_andn2_b64 vcc, exec, s[34:35]
	v_mul_f32_e32 v2, v45, v2
	s_cbranch_vccnz .LBB0_53
; %bb.52:                               ;   in Loop: Header=BB0_33 Depth=1
	global_load_ushort v3, v[0:1], off
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v2, s21, v3, v2 op_sel_hi:[0,1,0]
.LBB0_53:                               ;   in Loop: Header=BB0_33 Depth=1
	v_cvt_f16_f32_e32 v12, v2
	v_lshl_add_u64 v[2:3], v[50:51], 1, s[14:15]
	v_cmp_gt_u32_e64 s[16:17], s11, v62
	global_store_short v[2:3], v12, off
	s_and_saveexec_b64 s[46:47], s[16:17]
	s_cbranch_execz .LBB0_57
; %bb.54:                               ;   in Loop: Header=BB0_33 Depth=1
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v12, v16, v17
	s_and_b64 vcc, exec, s[18:19]
	v_mul_f32_e32 v12, v45, v12
	s_cbranch_vccnz .LBB0_56
; %bb.55:                               ;   in Loop: Header=BB0_33 Depth=1
	global_load_ushort v13, v[0:1], off offset:2
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v12, s21, v13, v12 op_sel_hi:[0,1,0]
.LBB0_56:                               ;   in Loop: Header=BB0_33 Depth=1
	v_cvt_f16_f32_e32 v12, v12
	global_store_short v[2:3], v12, off offset:2
.LBB0_57:                               ;   in Loop: Header=BB0_33 Depth=1
	s_or_b64 exec, exec, s[46:47]
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v10, v10, v11
	v_lshl_add_u64 v[0:1], s[30:31], 1, v[0:1]
	s_and_b64 vcc, exec, s[18:19]
	v_mul_f32_e32 v10, v45, v10
	s_cbranch_vccnz .LBB0_59
; %bb.58:                               ;   in Loop: Header=BB0_33 Depth=1
	global_load_ushort v11, v[0:1], off
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v10, s21, v11, v10 op_sel_hi:[0,1,0]
.LBB0_59:                               ;   in Loop: Header=BB0_33 Depth=1
	v_cvt_f16_f32_e32 v10, v10
	v_lshl_add_u64 v[2:3], s[36:37], 1, v[2:3]
	global_store_short v[2:3], v10, off
	s_and_saveexec_b64 s[46:47], s[16:17]
	s_cbranch_execz .LBB0_63
; %bb.60:                               ;   in Loop: Header=BB0_33 Depth=1
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v8, v8, v9
	s_and_b64 vcc, exec, s[18:19]
	v_mul_f32_e32 v8, v45, v8
	s_cbranch_vccnz .LBB0_62
; %bb.61:                               ;   in Loop: Header=BB0_33 Depth=1
	global_load_ushort v9, v[0:1], off offset:2
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v8, s21, v9, v8 op_sel_hi:[0,1,0]
.LBB0_62:                               ;   in Loop: Header=BB0_33 Depth=1
	v_cvt_f16_f32_e32 v8, v8
	global_store_short v[2:3], v8, off offset:2
.LBB0_63:                               ;   in Loop: Header=BB0_33 Depth=1
	s_or_b64 exec, exec, s[46:47]
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v6, v6, v7
	v_lshl_add_u64 v[0:1], v[0:1], 0, s[38:39]
	s_and_b64 vcc, exec, s[18:19]
	v_mul_f32_e32 v6, v45, v6
	s_cbranch_vccnz .LBB0_65
; %bb.64:                               ;   in Loop: Header=BB0_33 Depth=1
	global_load_ushort v7, v[0:1], off
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v6, s21, v7, v6 op_sel_hi:[0,1,0]
.LBB0_65:                               ;   in Loop: Header=BB0_33 Depth=1
	v_cvt_f16_f32_e32 v6, v6
	v_lshl_add_u64 v[2:3], v[2:3], 0, s[40:41]
	global_store_short v[2:3], v6, off
	s_and_b64 exec, exec, s[16:17]
	s_cbranch_execz .LBB0_32
; %bb.66:                               ;   in Loop: Header=BB0_33 Depth=1
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v4, v4, v5
	s_and_b64 vcc, exec, s[18:19]
	v_mul_f32_e32 v4, v45, v4
	s_cbranch_vccnz .LBB0_31
; %bb.67:                               ;   in Loop: Header=BB0_33 Depth=1
	global_load_ushort v0, v[0:1], off offset:2
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v4, s21, v0, v4 op_sel_hi:[0,1,0]
	s_branch .LBB0_31
.LBB0_68:
	s_or_b64 exec, exec, s[26:27]
	s_mov_b64 s[8:9], 0
.LBB0_69:
	s_andn2_b64 vcc, exec, s[8:9]
	s_cbranch_vccnz .LBB0_97
; %bb.70:
	v_cmp_gt_u32_e32 vcc, s11, v46
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_97
; %bb.71:
	v_mbcnt_lo_u32_b32 v0, -1, 0
	v_mbcnt_hi_u32_b32 v0, -1, v0
	v_and_b32_e32 v1, 63, v0
	v_cmp_gt_u32_e32 vcc, 44, v1
	s_cmp_lg_u32 s2, 0
	s_mov_b32 s19, 0
	v_cndmask_b32_e64 v2, 0, 20, vcc
	v_cmp_gt_u32_e32 vcc, 24, v1
	s_cselect_b64 s[6:7], -1, 0
	v_add_lshl_u32 v54, v2, v0, 2
	v_cndmask_b32_e64 v1, 0, 40, vcc
	v_add_lshl_u32 v55, v1, v0, 2
	s_mov_b32 s18, s22
	s_mov_b32 s22, s23
	s_mov_b32 s23, s19
	v_cndmask_b32_e64 v0, 0, 1, s[6:7]
	s_add_i32 s20, s2, -16
	v_cmp_eq_u32_e64 s[0:1], 0, v47
	v_cmp_neq_f32_e64 s[24:25], s21, 0
	s_lshl_b64 s[26:27], s[18:19], 1
	s_lshl_b64 s[28:29], s[22:23], 1
	v_add_u32_e32 v56, s50, v44
	v_add_u32_e32 v57, s2, v44
	s_mov_b64 s[30:31], 0
	v_cmp_ne_u32_e64 s[6:7], 1, v0
	v_mov_b32_e32 v49, 0
	s_branch .LBB0_74
.LBB0_72:                               ;   in Loop: Header=BB0_74 Depth=1
	v_cvt_f16_f32_e32 v0, v4
	global_store_short v[2:3], v0, off offset:2
.LBB0_73:                               ;   in Loop: Header=BB0_74 Depth=1
	s_or_b64 exec, exec, s[34:35]
	v_add_u32_e32 v46, s33, v46
	v_cmp_le_u32_e32 vcc, s11, v46
	s_or_b64 s[30:31], vcc, s[30:31]
	s_andn2_b64 exec, exec, s[30:31]
	s_cbranch_execz .LBB0_97
.LBB0_74:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_77 Depth 2
	s_and_b64 vcc, exec, s[6:7]
	v_or_b32_e32 v58, 1, v46
	v_mov_b32_e32 v23, v49
	v_mov_b32_e32 v22, v49
	v_mov_b32_e32 v21, v49
	v_mov_b32_e32 v20, v49
	s_waitcnt lgkmcnt(5)
	v_mov_b32_e32 v19, v49
	v_mov_b32_e32 v18, v49
	s_waitcnt lgkmcnt(4)
	v_mov_b32_e32 v17, v49
	v_mov_b32_e32 v16, v49
	s_waitcnt lgkmcnt(3)
	v_mov_b32_e32 v11, v49
	v_mov_b32_e32 v10, v49
	s_waitcnt lgkmcnt(2)
	v_mov_b32_e32 v9, v49
	v_mov_b32_e32 v8, v49
	s_waitcnt lgkmcnt(1)
	v_mov_b32_e32 v7, v49
	v_mov_b32_e32 v6, v49
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v5, v49
	v_mov_b32_e32 v4, v49
	v_mov_b32_e32 v3, v49
	v_mov_b32_e32 v2, v49
	v_mov_b32_e32 v1, v49
	v_mov_b32_e32 v0, v49
	v_mov_b32_e32 v15, v49
	v_mov_b32_e32 v14, v49
	v_mov_b32_e32 v13, v49
	v_mov_b32_e32 v12, v49
	s_cbranch_vccnz .LBB0_79
; %bb.75:                               ;   in Loop: Header=BB0_74 Depth=1
	v_min_u32_e32 v0, s3, v46
	v_mad_u64_u32 v[50:51], s[8:9], v0, s10, 0
	v_min_u32_e32 v0, s3, v58
	v_mov_b32_e32 v12, 0
	v_mad_u64_u32 v[52:53], s[8:9], v0, s10, 0
	s_mov_b32 s16, 0
	v_mov_b32_e32 v13, v12
	v_mov_b32_e32 v14, v12
	v_mov_b32_e32 v15, v12
	v_mov_b32_e32 v0, v12
	v_mov_b32_e32 v1, v12
	v_mov_b32_e32 v2, v12
	v_mov_b32_e32 v3, v12
	v_mov_b32_e32 v4, v12
	v_mov_b32_e32 v5, v12
	v_mov_b32_e32 v6, v12
	v_mov_b32_e32 v7, v12
	v_mov_b32_e32 v8, v12
	v_mov_b32_e32 v9, v12
	v_mov_b32_e32 v10, v12
	v_mov_b32_e32 v11, v12
	v_mov_b32_e32 v16, v12
	v_mov_b32_e32 v17, v12
	v_mov_b32_e32 v18, v12
	v_mov_b32_e32 v19, v12
	v_mov_b32_e32 v20, v12
	v_mov_b32_e32 v21, v12
	v_mov_b32_e32 v22, v12
	v_mov_b32_e32 v23, v12
	s_branch .LBB0_77
.LBB0_76:                               ;   in Loop: Header=BB0_77 Depth=2
	s_or_b64 exec, exec, s[8:9]
	s_waitcnt vmcnt(1) lgkmcnt(2)
	v_mfma_f32_16x16x32_fp8_fp8 v[20:23], v[36:37], v[28:29], v[20:23]
	s_addk_i32 s16, 0x400
	s_cmp_ge_u32 s16, s2
	s_waitcnt vmcnt(0)
	v_mfma_f32_16x16x32_fp8_fp8 v[16:19], v[36:37], v[24:25], v[16:19]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_fp8_fp8 v[8:11], v[40:41], v[28:29], v[8:11]
	v_mfma_f32_16x16x32_fp8_fp8 v[4:7], v[40:41], v[24:25], v[4:7]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[32:33], v[28:29], v[0:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[12:15], v[32:33], v[24:25], v[12:15]
	v_mfma_f32_16x16x32_fp8_fp8 v[20:23], v[38:39], v[30:31], v[20:23]
	v_mfma_f32_16x16x32_fp8_fp8 v[16:19], v[38:39], v[26:27], v[16:19]
	v_mfma_f32_16x16x32_fp8_fp8 v[8:11], v[42:43], v[30:31], v[8:11]
	v_mfma_f32_16x16x32_fp8_fp8 v[4:7], v[42:43], v[26:27], v[4:7]
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[34:35], v[30:31], v[0:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[12:15], v[34:35], v[26:27], v[12:15]
	s_cbranch_scc1 .LBB0_79
.LBB0_77:                               ;   Parent Loop BB0_74 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_add_u32_e32 v47, s16, v44
	v_min_u32_e32 v48, s20, v47
	v_lshl_add_u64 v[24:25], s[4:5], 0, v[48:49]
	v_lshl_add_u64 v[32:33], v[24:25], 0, v[50:51]
	v_lshl_add_u64 v[34:35], v[24:25], 0, v[52:53]
	global_load_dwordx4 v[28:31], v[32:33], off nt
	global_load_dwordx4 v[24:27], v[34:35], off nt
	v_cmp_gt_u32_e32 vcc, s2, v47
	v_mov_b64_e32 v[36:37], 0
	v_mov_b64_e32 v[38:39], 0
	v_mov_b64_e32 v[40:41], 0
	v_mov_b64_e32 v[42:43], 0
	v_mov_b64_e32 v[32:33], 0
	v_mov_b64_e32 v[34:35], 0
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_76
; %bb.78:                               ;   in Loop: Header=BB0_77 Depth=2
	v_add_u32_e32 v32, s16, v57
	ds_read_b128 v[36:39], v47
	ds_read_b128 v[40:43], v32
	v_add_u32_e32 v32, s16, v56
	ds_read_b128 v[32:35], v32
	s_branch .LBB0_76
.LBB0_79:                               ;   in Loop: Header=BB0_74 Depth=1
	s_nop 5
	v_add_f32_dpp v0, v1, v0 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v20, v21, v20 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v16, v17, v16 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v9, v8 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v5, v4 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v2, v0 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v13, v12 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v20, v22, v20 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v16, v18, v16 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v10, v8 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v6, v4 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v14, v2 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v20, v23, v20 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v16, v19, v16 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v11, v8 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v7, v4 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v3, v0 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v15, v2 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	ds_bpermute_b32 v21, v54, v20
	ds_bpermute_b32 v17, v54, v16
	ds_bpermute_b32 v9, v54, v8
	ds_bpermute_b32 v5, v54, v4
	ds_bpermute_b32 v1, v54, v0
	ds_bpermute_b32 v3, v54, v2
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v18, v20, v21
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v16, v16, v17
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v10, v8, v9
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v8, v4, v5
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v6, v0, v1
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v4, v2, v3
	ds_bpermute_b32 v19, v55, v18
	ds_bpermute_b32 v17, v55, v16
	ds_bpermute_b32 v11, v55, v10
	ds_bpermute_b32 v9, v55, v8
	ds_bpermute_b32 v7, v55, v6
	ds_bpermute_b32 v5, v55, v4
	s_and_saveexec_b64 s[34:35], s[0:1]
	s_cbranch_execz .LBB0_73
; %bb.80:                               ;   in Loop: Header=BB0_74 Depth=1
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v2, v18, v19
	v_mov_b32_e32 v47, v49
	v_cndmask_b32_e64 v3, 0, 1, s[24:25]
	v_lshl_add_u64 v[0:1], v[46:47], 1, s[12:13]
	v_cmp_ne_u32_e64 s[8:9], 1, v3
	s_andn2_b64 vcc, exec, s[24:25]
	v_mul_f32_e32 v2, v45, v2
	s_cbranch_vccnz .LBB0_82
; %bb.81:                               ;   in Loop: Header=BB0_74 Depth=1
	global_load_ushort v3, v[0:1], off
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v2, s21, v3, v2 op_sel_hi:[0,1,0]
.LBB0_82:                               ;   in Loop: Header=BB0_74 Depth=1
	v_cvt_f16_f32_e32 v12, v2
	v_lshl_add_u64 v[2:3], v[46:47], 1, s[14:15]
	v_cmp_gt_u32_e64 s[16:17], s11, v58
	global_store_short v[2:3], v12, off
	s_and_saveexec_b64 s[36:37], s[16:17]
	s_cbranch_execz .LBB0_86
; %bb.83:                               ;   in Loop: Header=BB0_74 Depth=1
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v12, v16, v17
	s_and_b64 vcc, exec, s[8:9]
	v_mul_f32_e32 v12, v45, v12
	s_cbranch_vccnz .LBB0_85
; %bb.84:                               ;   in Loop: Header=BB0_74 Depth=1
	global_load_ushort v13, v[0:1], off offset:2
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v12, s21, v13, v12 op_sel_hi:[0,1,0]
.LBB0_85:                               ;   in Loop: Header=BB0_74 Depth=1
	v_cvt_f16_f32_e32 v12, v12
	global_store_short v[2:3], v12, off offset:2
.LBB0_86:                               ;   in Loop: Header=BB0_74 Depth=1
	s_or_b64 exec, exec, s[36:37]
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v10, v10, v11
	v_lshl_add_u64 v[0:1], s[18:19], 1, v[0:1]
	s_and_b64 vcc, exec, s[8:9]
	v_mul_f32_e32 v10, v45, v10
	s_cbranch_vccnz .LBB0_88
; %bb.87:                               ;   in Loop: Header=BB0_74 Depth=1
	global_load_ushort v11, v[0:1], off
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v10, s21, v11, v10 op_sel_hi:[0,1,0]
.LBB0_88:                               ;   in Loop: Header=BB0_74 Depth=1
	v_cvt_f16_f32_e32 v10, v10
	v_lshl_add_u64 v[2:3], s[22:23], 1, v[2:3]
	global_store_short v[2:3], v10, off
	s_and_saveexec_b64 s[36:37], s[16:17]
	s_cbranch_execz .LBB0_92
; %bb.89:                               ;   in Loop: Header=BB0_74 Depth=1
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v8, v8, v9
	s_and_b64 vcc, exec, s[8:9]
	v_mul_f32_e32 v8, v45, v8
	s_cbranch_vccnz .LBB0_91
; %bb.90:                               ;   in Loop: Header=BB0_74 Depth=1
	global_load_ushort v9, v[0:1], off offset:2
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v8, s21, v9, v8 op_sel_hi:[0,1,0]
.LBB0_91:                               ;   in Loop: Header=BB0_74 Depth=1
	v_cvt_f16_f32_e32 v8, v8
	global_store_short v[2:3], v8, off offset:2
.LBB0_92:                               ;   in Loop: Header=BB0_74 Depth=1
	s_or_b64 exec, exec, s[36:37]
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v6, v6, v7
	v_lshl_add_u64 v[0:1], v[0:1], 0, s[26:27]
	s_and_b64 vcc, exec, s[8:9]
	v_mul_f32_e32 v6, v45, v6
	s_cbranch_vccnz .LBB0_94
; %bb.93:                               ;   in Loop: Header=BB0_74 Depth=1
	global_load_ushort v7, v[0:1], off
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v6, s21, v7, v6 op_sel_hi:[0,1,0]
.LBB0_94:                               ;   in Loop: Header=BB0_74 Depth=1
	v_cvt_f16_f32_e32 v6, v6
	v_lshl_add_u64 v[2:3], v[2:3], 0, s[28:29]
	global_store_short v[2:3], v6, off
	s_and_b64 exec, exec, s[16:17]
	s_cbranch_execz .LBB0_73
; %bb.95:                               ;   in Loop: Header=BB0_74 Depth=1
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v4, v4, v5
	s_and_b64 vcc, exec, s[8:9]
	v_mul_f32_e32 v4, v45, v4
	s_cbranch_vccnz .LBB0_72
; %bb.96:                               ;   in Loop: Header=BB0_74 Depth=1
	global_load_ushort v0, v[0:1], off offset:2
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v4, s21, v0, v4 op_sel_hi:[0,1,0]
	s_branch .LBB0_72
.LBB0_97:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wvSplitKQ_vllm_f8_f16_tn_n3_y2u1
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
		.amdhsa_accum_offset 64
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
	.size	wvSplitKQ_vllm_f8_f16_tn_n3_y2u1, .Lfunc_end0-wvSplitKQ_vllm_f8_f16_tn_n3_y2u1
                                        ; -- End function
	.set wvSplitKQ_vllm_f8_f16_tn_n3_y2u1.num_vgpr, 63
	.set wvSplitKQ_vllm_f8_f16_tn_n3_y2u1.num_agpr, 0
	.set wvSplitKQ_vllm_f8_f16_tn_n3_y2u1.numbered_sgpr, 52
	.set wvSplitKQ_vllm_f8_f16_tn_n3_y2u1.num_named_barrier, 0
	.set wvSplitKQ_vllm_f8_f16_tn_n3_y2u1.private_seg_size, 0
	.set wvSplitKQ_vllm_f8_f16_tn_n3_y2u1.uses_vcc, 1
	.set wvSplitKQ_vllm_f8_f16_tn_n3_y2u1.uses_flat_scratch, 0
	.set wvSplitKQ_vllm_f8_f16_tn_n3_y2u1.has_dyn_sized_stack, 0
	.set wvSplitKQ_vllm_f8_f16_tn_n3_y2u1.has_recursion, 0
	.set wvSplitKQ_vllm_f8_f16_tn_n3_y2u1.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 4312
; TotalNumSgprs: 58
; NumVgprs: 63
; NumAgprs: 0
; TotalNumVgprs: 63
; ScratchSize: 0
; MemoryBound: 1
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 163840 bytes/workgroup (compile time only)
; SGPRBlocks: 12
; VGPRBlocks: 12
; NumSGPRsForWavesPerEU: 102
; NumVGPRsForWavesPerEU: 97
; AccumOffset: 64
; Occupancy: 4
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 1
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 15
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_8d987c28af5765f4,@object ; @__hip_cuid_8d987c28af5765f4
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_8d987c28af5765f4
__hip_cuid_8d987c28af5765f4:
	.byte	0                               ; 0x0
	.size	__hip_cuid_8d987c28af5765f4, 1

	.ident	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.4 26084 f58b06dce1f9c15707c5f808fd002e18c2accf7e)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_8d987c28af5765f4
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
    DestDataType: h
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
  AssertSizeEqual: { 1: 3, 2: 1 }
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
    .name:           wvSplitKQ_vllm_f8_f16_tn_n3_y2u1
    .private_segment_fixed_size: 0
    .sgpr_count:     58
    .sgpr_spill_count: 0
    .symbol:         wvSplitKQ_vllm_f8_f16_tn_n3_y2u1.kd
    .vgpr_count:     63
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 1
...

	.end_amdgpu_metadata
