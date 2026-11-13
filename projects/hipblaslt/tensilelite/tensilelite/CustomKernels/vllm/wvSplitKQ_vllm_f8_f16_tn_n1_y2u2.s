	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 4
	.text
	.protected	wvSplitKQ_vllm_f8_f16_tn_n1_y2u2 ; -- Begin function wvSplitKQ_vllm_f8_f16_tn_n1_y2u2
	.globl	wvSplitKQ_vllm_f8_f16_tn_n1_y2u2
	.p2align	8
	.type	wvSplitKQ_vllm_f8_f16_tn_n1_y2u2,@function
wvSplitKQ_vllm_f8_f16_tn_n1_y2u2:       ; @wvSplitKQ_vllm_f8_f16_tn_n1_y2u2
; %bb.59:
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.60:
.LBB0_0:
	s_load_dwordx4 s[20:23], s[0:1], 0x48
	v_and_b32_e32 v35, 0x3ff, v0
	v_bfe_u32 v2, v0, 10, 10
	v_lshlrev_b32_e32 v32, 4, v35
	s_min_u32 s9, s2, 0x28000
	v_lshl_add_u32 v3, v2, 10, v32
	v_cmp_gt_u32_e32 vcc, s9, v3
	s_and_saveexec_b64 s[18:19], vcc
	s_cbranch_execz .LBB0_7
; %bb.1:
	s_and_b32 s26, s6, 15
	s_mov_b32 s27, 0
	s_cmp_lg_u64 s[26:27], 0
	s_cselect_b64 s[26:27], -1, 0
	s_and_b32 s11, s11, 15
	v_mov_b32_e32 v1, 0
	s_cmp_lg_u32 s11, 0
	v_lshlrev_b32_e32 v0, 10, v2
	v_mov_b32_e32 v33, v1
	s_cselect_b64 s[28:29], -1, 0
	v_lshl_add_u64 v[0:1], v[0:1], 0, v[32:33]
	s_mov_b64 s[24:25], 0
	s_or_b64 s[26:27], s[26:27], s[28:29]
	v_lshl_add_u64 v[0:1], s[6:7], 0, v[0:1]
	s_mov_b64 s[28:29], 0x4000
	s_branch .LBB0_3
.LBB0_2:                                ;   in Loop: Header=BB0_3 Depth=1
	v_add_u32_e32 v3, 0x4000, v3
	v_cmp_le_u32_e32 vcc, s9, v3
	s_or_b64 s[24:25], vcc, s[24:25]
	v_lshl_add_u64 v[0:1], v[0:1], 0, s[28:29]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execz .LBB0_7
.LBB0_3:                                ; =>This Inner Loop Header: Depth=1
	s_mov_b64 s[30:31], -1
	s_and_b64 vcc, exec, s[26:27]
	s_cbranch_vccz .LBB0_5
; %bb.4:                                ;   in Loop: Header=BB0_3 Depth=1
	global_load_dwordx4 v[4:7], v[0:1], off
	s_mov_b64 s[30:31], 0
	s_waitcnt vmcnt(0)
	ds_write_b128 v3, v[4:7]
.LBB0_5:                                ;   in Loop: Header=BB0_3 Depth=1
	s_andn2_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB0_2
; %bb.6:                                ;   in Loop: Header=BB0_3 Depth=1
	v_readfirstlane_b32 s11, v3
	s_mov_b32 m0, s11
	s_nop 0
	global_load_lds_dwordx4 v[0:1], off
	s_branch .LBB0_2
.LBB0_7:
	s_or_b64 exec, exec, s[18:19]
	s_add_i32 s11, s3, -1
	s_lshl_b32 s17, s8, 5
	v_mov_b32_e32 v0, s11
	v_add_co_u32_e32 v0, vcc, s17, v0
	s_nop 0
	v_readfirstlane_b32 s18, v0
	s_andn2_b64 vcc, exec, vcc
	s_mov_b32 s9, 4
	s_cbranch_vccz .LBB0_9
; %bb.8:
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
	s_add_i32 s24, s9, 1
	s_sub_i32 s19, s18, s17
	s_cmp_ge_u32 s18, s17
	s_cselect_b32 s9, s24, s9
	s_cselect_b32 s18, s19, s18
	s_add_i32 s19, s9, 1
	s_cmp_ge_u32 s18, s17
	s_cselect_b32 s9, s19, s9
	s_mul_i32 s9, s8, s9
	s_lshl_b32 s9, s9, 1
	v_cvt_f32_u32_e32 v0, s9
	s_sub_i32 s18, 0, s9
	s_add_i32 s17, s9, s11
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
	s_add_i32 s24, s18, 1
	s_sub_i32 s19, s17, s9
	s_cmp_ge_u32 s17, s9
	s_cselect_b32 s18, s24, s18
	s_cselect_b32 s17, s19, s17
	s_add_i32 s19, s18, 1
	s_cmp_ge_u32 s17, s9
	s_cselect_b32 s9, s19, s18
	s_max_u32 s9, s9, 4
.LBB0_9:
	s_mov_b32 s17, 1.0
	s_waitcnt lgkmcnt(0)
	s_cmp_eq_u64 s[20:21], 0
	s_mov_b32 s24, 1.0
	s_cbranch_scc1 .LBB0_11
; %bb.10:
	s_load_dword s24, s[20:21], 0x0
.LBB0_11:
	s_cmp_eq_u64 s[22:23], 0
	s_cbranch_scc1 .LBB0_13
; %bb.12:
	s_load_dword s17, s[22:23], 0x0
.LBB0_13:
	v_cmp_gt_u32_e32 vcc, s9, v2
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	s_and_saveexec_b64 s[18:19], vcc
	s_cbranch_execz .LBB0_58
; %bb.14:
	s_load_dwordx2 s[18:19], s[0:1], 0x38
	s_mul_i32 s8, s8, s9
	v_mov_b32_e32 v0, s24
	s_mul_i32 s0, s9, s16
	s_lshl_b32 s28, s8, 1
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v0, s18, v0
	v_add_lshl_u32 v34, s0, v2, 1
	s_cmp_gt_u32 s2, 0x28000
	v_mul_f32_e32 v33, s17, v0
	s_mov_b64 s[0:1], -1
	v_cmp_gt_u32_e32 vcc, s3, v34
	s_cbranch_scc0 .LBB0_40
; %bb.15:
	s_and_saveexec_b64 s[16:17], vcc
	s_cbranch_execz .LBB0_39
; %bb.16:
	v_mbcnt_lo_u32_b32 v0, -1, 0
	v_mbcnt_hi_u32_b32 v0, -1, v0
	v_and_b32_e32 v1, 63, v0
	v_cmp_gt_u32_e32 vcc, 44, v1
	s_add_i32 s18, s2, -16
	v_cmp_eq_u32_e64 s[0:1], 0, v35
	v_cndmask_b32_e64 v2, 0, 20, vcc
	v_cmp_gt_u32_e32 vcc, 24, v1
	v_add_lshl_u32 v44, v2, v0, 2
	v_cmp_neq_f32_e64 s[20:21], s19, 0
	v_cndmask_b32_e64 v1, 0, 40, vcc
	v_add_lshl_u32 v45, v1, v0, 2
	s_mov_b64 s[22:23], 0
	v_mov_b32_e32 v37, 0
	s_mov_b32 s29, 0x27fff
	v_mov_b32_e32 v38, v34
	s_branch .LBB0_19
.LBB0_17:                               ;   in Loop: Header=BB0_19 Depth=1
	v_cvt_f16_f32_e32 v0, v4
	global_store_short v[2:3], v0, off offset:2
.LBB0_18:                               ;   in Loop: Header=BB0_19 Depth=1
	s_or_b64 exec, exec, s[24:25]
	v_add_u32_e32 v38, s28, v38
	v_cmp_le_u32_e32 vcc, s3, v38
	s_or_b64 s[22:23], vcc, s[22:23]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execz .LBB0_39
.LBB0_19:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_23 Depth 2
	v_min_u32_e32 v0, s11, v38
	v_or_b32_e32 v46, 1, v38
	v_mad_u64_u32 v[40:41], s[8:9], v0, s10, 0
	v_min_u32_e32 v0, s11, v46
	v_mad_u64_u32 v[42:43], s[8:9], v0, s10, 0
	s_mov_b32 s30, 0
	v_mov_b32_e32 v0, v37
	s_waitcnt lgkmcnt(1)
	v_mov_b32_e32 v1, v37
	v_mov_b32_e32 v2, v37
	v_mov_b32_e32 v3, v37
	v_mov_b32_e32 v4, v37
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v5, v37
	v_mov_b32_e32 v6, v37
	v_mov_b32_e32 v7, v37
	s_branch .LBB0_23
.LBB0_20:                               ;   in Loop: Header=BB0_23 Depth=2
	s_or_b64 exec, exec, s[26:27]
.LBB0_21:                               ;   in Loop: Header=BB0_23 Depth=2
	s_or_b64 exec, exec, s[24:25]
.LBB0_22:                               ;   in Loop: Header=BB0_23 Depth=2
	s_or_b64 exec, exec, s[8:9]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[24:25], v[16:17], v[0:3]
	s_addk_i32 s30, 0x800
	s_cmp_lt_u32 s30, s2
	v_mfma_f32_16x16x32_fp8_fp8 v[4:7], v[24:25], v[12:13], v[4:7]
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[26:27], v[18:19], v[0:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[4:7], v[26:27], v[14:15], v[4:7]
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[28:29], v[20:21], v[0:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[4:7], v[28:29], v[8:9], v[4:7]
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[30:31], v[22:23], v[0:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[4:7], v[30:31], v[10:11], v[4:7]
	s_cbranch_scc0 .LBB0_33
.LBB0_23:                               ;   Parent Loop BB0_19 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_add_u32_e32 v39, s30, v32
	v_min_u32_e32 v36, s18, v39
	v_lshl_add_u64 v[8:9], s[4:5], 0, v[36:37]
	v_add_u32_e32 v47, 0x400, v39
	v_lshl_add_u64 v[10:11], v[8:9], 0, v[40:41]
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[42:43]
	v_min_u32_e32 v36, s18, v47
	global_load_dwordx4 v[16:19], v[10:11], off nt
	global_load_dwordx4 v[12:15], v[8:9], off nt
	v_lshl_add_u64 v[8:9], s[4:5], 0, v[36:37]
	v_lshl_add_u64 v[24:25], v[8:9], 0, v[40:41]
	v_lshl_add_u64 v[26:27], v[8:9], 0, v[42:43]
	global_load_dwordx4 v[20:23], v[24:25], off nt
	global_load_dwordx4 v[8:11], v[26:27], off nt
	v_cmp_gt_u32_e32 vcc, s2, v39
	v_mov_b64_e32 v[30:31], 0
	v_mov_b64_e32 v[28:29], 0
	v_mov_b64_e32 v[26:27], 0
	v_mov_b64_e32 v[24:25], 0
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_22
; %bb.24:                               ;   in Loop: Header=BB0_23 Depth=2
	v_cmp_lt_u32_e32 vcc, s29, v39
                                        ; implicit-def: $vgpr26_vgpr27
	s_and_saveexec_b64 s[24:25], vcc
	s_xor_b64 s[24:25], exec, s[24:25]
	s_cbranch_execz .LBB0_26
; %bb.25:                               ;   in Loop: Header=BB0_23 Depth=2
	global_load_dwordx4 v[24:27], v39, s[6:7]
.LBB0_26:                               ;   in Loop: Header=BB0_23 Depth=2
	s_andn2_saveexec_b64 s[24:25], s[24:25]
	s_cbranch_execz .LBB0_28
; %bb.27:                               ;   in Loop: Header=BB0_23 Depth=2
	s_waitcnt vmcnt(0)
	ds_read_b128 v[24:27], v39
.LBB0_28:                               ;   in Loop: Header=BB0_23 Depth=2
	s_or_b64 exec, exec, s[24:25]
	v_cmp_gt_u32_e32 vcc, s2, v47
	v_mov_b64_e32 v[30:31], 0
	v_mov_b64_e32 v[28:29], 0
	s_and_saveexec_b64 s[24:25], vcc
	s_cbranch_execz .LBB0_21
; %bb.29:                               ;   in Loop: Header=BB0_23 Depth=2
	v_cmp_lt_u32_e32 vcc, s29, v47
                                        ; implicit-def: $vgpr30_vgpr31
	s_and_saveexec_b64 s[26:27], vcc
	s_xor_b64 s[26:27], exec, s[26:27]
	s_cbranch_execz .LBB0_31
; %bb.30:                               ;   in Loop: Header=BB0_23 Depth=2
	global_load_dwordx4 v[28:31], v47, s[6:7]
                                        ; implicit-def: $vgpr39
.LBB0_31:                               ;   in Loop: Header=BB0_23 Depth=2
	s_andn2_saveexec_b64 s[26:27], s[26:27]
	s_cbranch_execz .LBB0_20
; %bb.32:                               ;   in Loop: Header=BB0_23 Depth=2
	s_waitcnt vmcnt(0)
	ds_read_b128 v[28:31], v39 offset:1024
	s_branch .LBB0_20
.LBB0_33:                               ;   in Loop: Header=BB0_19 Depth=1
	s_nop 5
	v_add_f32_dpp v0, v1, v0 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v0, v2, v0 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v5, v4 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_add_f32_dpp v0, v3, v0 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v6, v2 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	ds_bpermute_b32 v1, v44, v0
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v0, v0, v1
	v_add_f32_dpp v2, v7, v2 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	ds_bpermute_b32 v3, v44, v2
	ds_bpermute_b32 v1, v45, v0
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v4, v2, v3
	ds_bpermute_b32 v5, v45, v4
	s_and_saveexec_b64 s[24:25], s[0:1]
	s_cbranch_execz .LBB0_18
; %bb.34:                               ;   in Loop: Header=BB0_19 Depth=1
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v2, v0, v1
	v_mov_b32_e32 v39, v37
	v_cndmask_b32_e64 v3, 0, 1, s[20:21]
	v_lshl_add_u64 v[0:1], v[38:39], 1, s[12:13]
	v_cmp_ne_u32_e64 s[8:9], 1, v3
	s_andn2_b64 vcc, exec, s[20:21]
	v_mul_f32_e32 v2, v33, v2
	s_cbranch_vccnz .LBB0_36
; %bb.35:                               ;   in Loop: Header=BB0_19 Depth=1
	global_load_ushort v3, v[0:1], off
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v2, s19, v3, v2 op_sel_hi:[0,1,0]
.LBB0_36:                               ;   in Loop: Header=BB0_19 Depth=1
	v_cvt_f16_f32_e32 v6, v2
	v_lshl_add_u64 v[2:3], v[38:39], 1, s[14:15]
	v_cmp_gt_u32_e32 vcc, s3, v46
	global_store_short v[2:3], v6, off
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_18
; %bb.37:                               ;   in Loop: Header=BB0_19 Depth=1
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v4, v4, v5
	s_and_b64 vcc, exec, s[8:9]
	v_mul_f32_e32 v4, v33, v4
	s_cbranch_vccnz .LBB0_17
; %bb.38:                               ;   in Loop: Header=BB0_19 Depth=1
	global_load_ushort v0, v[0:1], off offset:2
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v4, s19, v0, v4 op_sel_hi:[0,1,0]
	s_branch .LBB0_17
.LBB0_39:
	s_or_b64 exec, exec, s[16:17]
	s_mov_b64 s[0:1], 0
.LBB0_40:
	s_andn2_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB0_58
; %bb.41:
	v_cmp_gt_u32_e32 vcc, s3, v34
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_58
; %bb.42:
	v_mbcnt_lo_u32_b32 v0, -1, 0
	v_mbcnt_hi_u32_b32 v0, -1, v0
	s_waitcnt lgkmcnt(1)
	v_and_b32_e32 v1, 63, v0
	v_cmp_gt_u32_e32 vcc, 44, v1
	s_cmp_lg_u32 s2, 0
	s_cselect_b64 s[6:7], -1, 0
	v_cndmask_b32_e64 v2, 0, 20, vcc
	v_cmp_gt_u32_e32 vcc, 24, v1
	v_add_lshl_u32 v42, v2, v0, 2
	s_add_i32 s18, s2, -16
	v_cndmask_b32_e64 v1, 0, 40, vcc
	v_add_lshl_u32 v43, v1, v0, 2
	v_cndmask_b32_e64 v0, 0, 1, s[6:7]
	v_cmp_neq_f32_e64 s[6:7], s19, 0
	v_cmp_ne_u32_e64 s[8:9], 1, v0
	v_cmp_eq_u32_e64 s[0:1], 0, v35
	v_cndmask_b32_e64 v0, 0, 1, s[6:7]
	s_mov_b64 s[16:17], 0
	v_mov_b32_e32 v37, 0
	v_cmp_ne_u32_e64 s[6:7], 1, v0
	s_branch .LBB0_45
.LBB0_43:                               ;   in Loop: Header=BB0_45 Depth=1
	v_cvt_f16_f32_e32 v0, v4
	global_store_short v[2:3], v0, off offset:2
.LBB0_44:                               ;   in Loop: Header=BB0_45 Depth=1
	s_or_b64 exec, exec, s[20:21]
	v_add_u32_e32 v34, s28, v34
	v_cmp_le_u32_e32 vcc, s3, v34
	s_or_b64 s[16:17], vcc, s[16:17]
	s_andn2_b64 exec, exec, s[16:17]
	s_cbranch_execz .LBB0_58
.LBB0_45:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_49 Depth 2
	s_and_b64 vcc, exec, s[8:9]
	v_or_b32_e32 v44, 1, v34
	v_mov_b32_e32 v7, v37
	v_mov_b32_e32 v6, v37
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v5, v37
	v_mov_b32_e32 v4, v37
	v_mov_b32_e32 v3, v37
	v_mov_b32_e32 v2, v37
	v_mov_b32_e32 v1, v37
	v_mov_b32_e32 v0, v37
	s_cbranch_vccnz .LBB0_52
; %bb.46:                               ;   in Loop: Header=BB0_45 Depth=1
	v_min_u32_e32 v0, s11, v34
	v_mad_u64_u32 v[38:39], s[20:21], v0, s10, 0
	v_min_u32_e32 v0, s11, v44
	v_mad_u64_u32 v[40:41], s[20:21], v0, s10, 0
	v_mov_b32_e32 v0, 0
	s_mov_b32 s24, 0
	v_mov_b32_e32 v1, v0
	v_mov_b32_e32 v2, v0
	v_mov_b32_e32 v3, v0
	v_mov_b32_e32 v4, v0
	v_mov_b32_e32 v5, v0
	v_mov_b32_e32 v6, v0
	v_mov_b32_e32 v7, v0
	s_branch .LBB0_49
.LBB0_47:                               ;   in Loop: Header=BB0_49 Depth=2
	s_or_b64 exec, exec, s[22:23]
.LBB0_48:                               ;   in Loop: Header=BB0_49 Depth=2
	s_or_b64 exec, exec, s[20:21]
	s_waitcnt vmcnt(3) lgkmcnt(0)
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[28:29], v[20:21], v[0:3]
	s_addk_i32 s24, 0x800
	s_cmp_lt_u32 s24, s2
	s_waitcnt vmcnt(2)
	v_mfma_f32_16x16x32_fp8_fp8 v[4:7], v[28:29], v[12:13], v[4:7]
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[30:31], v[22:23], v[0:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[4:7], v[30:31], v[14:15], v[4:7]
	s_waitcnt vmcnt(1)
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[16:17], v[24:25], v[0:3]
	s_waitcnt vmcnt(0)
	v_mfma_f32_16x16x32_fp8_fp8 v[4:7], v[16:17], v[8:9], v[4:7]
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[18:19], v[26:27], v[0:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[4:7], v[18:19], v[10:11], v[4:7]
	s_cbranch_scc0 .LBB0_52
.LBB0_49:                               ;   Parent Loop BB0_45 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_add_u32_e32 v35, s24, v32
	v_min_u32_e32 v36, s18, v35
	v_lshl_add_u64 v[8:9], s[4:5], 0, v[36:37]
	v_add_u32_e32 v45, 0x400, v35
	v_lshl_add_u64 v[10:11], v[8:9], 0, v[38:39]
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[40:41]
	v_min_u32_e32 v36, s18, v45
	global_load_dwordx4 v[20:23], v[10:11], off nt
	global_load_dwordx4 v[12:15], v[8:9], off nt
	v_lshl_add_u64 v[8:9], s[4:5], 0, v[36:37]
	v_lshl_add_u64 v[16:17], v[8:9], 0, v[38:39]
	v_lshl_add_u64 v[18:19], v[8:9], 0, v[40:41]
	global_load_dwordx4 v[24:27], v[16:17], off nt
	global_load_dwordx4 v[8:11], v[18:19], off nt
	v_cmp_gt_u32_e32 vcc, s2, v35
	v_mov_b64_e32 v[28:29], 0
	v_mov_b64_e32 v[30:31], 0
	v_mov_b64_e32 v[16:17], 0
	v_mov_b64_e32 v[18:19], 0
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_48
; %bb.50:                               ;   in Loop: Header=BB0_49 Depth=2
	ds_read_b128 v[28:31], v35
	v_cmp_gt_u32_e32 vcc, s2, v45
	v_mov_b64_e32 v[18:19], 0
	v_mov_b64_e32 v[16:17], 0
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_47
; %bb.51:                               ;   in Loop: Header=BB0_49 Depth=2
	ds_read_b128 v[16:19], v35 offset:1024
	s_branch .LBB0_47
.LBB0_52:                               ;   in Loop: Header=BB0_45 Depth=1
	s_nop 5
	v_add_f32_dpp v0, v1, v0 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v0, v2, v0 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v5, v4 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_add_f32_dpp v0, v3, v0 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v6, v2 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	ds_bpermute_b32 v1, v42, v0
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v0, v0, v1
	v_add_f32_dpp v2, v7, v2 row_shl:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
	ds_bpermute_b32 v3, v42, v2
	ds_bpermute_b32 v1, v43, v0
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v4, v2, v3
	ds_bpermute_b32 v5, v43, v4
	s_and_saveexec_b64 s[20:21], s[0:1]
	s_cbranch_execz .LBB0_44
; %bb.53:                               ;   in Loop: Header=BB0_45 Depth=1
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v2, v0, v1
	v_mov_b32_e32 v35, v37
	v_lshl_add_u64 v[0:1], v[34:35], 1, s[12:13]
	s_and_b64 vcc, exec, s[6:7]
	v_mul_f32_e32 v2, v33, v2
	s_cbranch_vccnz .LBB0_55
; %bb.54:                               ;   in Loop: Header=BB0_45 Depth=1
	global_load_ushort v3, v[0:1], off
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v2, s19, v3, v2 op_sel_hi:[0,1,0]
.LBB0_55:                               ;   in Loop: Header=BB0_45 Depth=1
	v_cvt_f16_f32_e32 v6, v2
	v_lshl_add_u64 v[2:3], v[34:35], 1, s[14:15]
	v_cmp_gt_u32_e32 vcc, s3, v44
	global_store_short v[2:3], v6, off
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_44
; %bb.56:                               ;   in Loop: Header=BB0_45 Depth=1
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v4, v4, v5
	s_and_b64 vcc, exec, s[6:7]
	v_mul_f32_e32 v4, v33, v4
	s_cbranch_vccnz .LBB0_43
; %bb.57:                               ;   in Loop: Header=BB0_45 Depth=1
	global_load_ushort v0, v[0:1], off offset:2
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v4, s19, v0, v4 op_sel_hi:[0,1,0]
	s_branch .LBB0_43
.LBB0_58:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wvSplitKQ_vllm_f8_f16_tn_n1_y2u2
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
		.amdhsa_accum_offset 48
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
	.size	wvSplitKQ_vllm_f8_f16_tn_n1_y2u2, .Lfunc_end0-wvSplitKQ_vllm_f8_f16_tn_n1_y2u2
                                        ; -- End function
	.set wvSplitKQ_vllm_f8_f16_tn_n1_y2u2.num_vgpr, 48
	.set wvSplitKQ_vllm_f8_f16_tn_n1_y2u2.num_agpr, 0
	.set wvSplitKQ_vllm_f8_f16_tn_n1_y2u2.numbered_sgpr, 32
	.set wvSplitKQ_vllm_f8_f16_tn_n1_y2u2.num_named_barrier, 0
	.set wvSplitKQ_vllm_f8_f16_tn_n1_y2u2.private_seg_size, 0
	.set wvSplitKQ_vllm_f8_f16_tn_n1_y2u2.uses_vcc, 1
	.set wvSplitKQ_vllm_f8_f16_tn_n1_y2u2.uses_flat_scratch, 0
	.set wvSplitKQ_vllm_f8_f16_tn_n1_y2u2.has_dyn_sized_stack, 0
	.set wvSplitKQ_vllm_f8_f16_tn_n1_y2u2.has_recursion, 0
	.set wvSplitKQ_vllm_f8_f16_tn_n1_y2u2.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 2520
; TotalNumSgprs: 38
; NumVgprs: 48
; NumAgprs: 0
; TotalNumVgprs: 48
; ScratchSize: 0
; MemoryBound: 1
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 163840 bytes/workgroup (compile time only)
; SGPRBlocks: 12
; VGPRBlocks: 12
; NumSGPRsForWavesPerEU: 102
; NumVGPRsForWavesPerEU: 97
; AccumOffset: 48
; Occupancy: 4
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 1
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 11
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_cd224954bbdbd40a,@object ; @__hip_cuid_cd224954bbdbd40a
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_cd224954bbdbd40a
__hip_cuid_cd224954bbdbd40a:
	.byte	0                               ; 0x0
	.size	__hip_cuid_cd224954bbdbd40a, 1

	.ident	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.4 26084 f58b06dce1f9c15707c5f808fd002e18c2accf7e)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_cd224954bbdbd40a
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
  AssertSizeEqual: { 1: 1, 2: 1 }
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
    .name:           wvSplitKQ_vllm_f8_f16_tn_n1_y2u2
    .private_segment_fixed_size: 0
    .sgpr_count:     38
    .sgpr_spill_count: 0
    .symbol:         wvSplitKQ_vllm_f8_f16_tn_n1_y2u2.kd
    .vgpr_count:     48
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 1
...

	.end_amdgpu_metadata
