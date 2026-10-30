	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 4
	.text
	.protected	wvSplitK_vllm_f16_tn_n2_y1u2 ; -- Begin function wvSplitK_vllm_f16_tn_n2_y1u2
	.globl	wvSplitK_vllm_f16_tn_n2_y1u2
	.p2align	8
	.type	wvSplitK_vllm_f16_tn_n2_y1u2,@function
wvSplitK_vllm_f16_tn_n2_y1u2:           ; @wvSplitK_vllm_f16_tn_n2_y1u2
; %bb.69:
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.70:
.LBB0_0:
	s_load_dwordx4 s[24:27], s[0:1], 0x38
	s_mov_b32 s22, s11
	s_mov_b64 s[0:1], -1
	s_cmp_gt_i32 s2, 0xa000
	v_bfe_u32 v32, v0, 10, 10
	s_cbranch_scc0 .LBB0_34
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
	v_cmp_gt_u32_e32 vcc, s18, v32
	s_and_saveexec_b64 s[28:29], vcc
	s_cbranch_execz .LBB0_33
; %bb.2:
	v_cvt_f32_u32_e32 v1, s18
	s_mul_i32 s0, s18, s16
	s_sub_i32 s11, 0, s18
	v_add_u32_e32 v26, s0, v32
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
	v_cmp_gt_u32_e32 vcc, s9, v26
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_33
; %bb.3:
	s_and_b32 s0, s22, 7
	s_cmp_lg_u32 s0, 0
	s_mov_b32 s31, 0
	s_cselect_b64 s[20:21], -1, 0
	s_and_b32 s30, s6, 15
	s_cmp_lg_u64 s[30:31], 0
	s_cselect_b64 s[42:43], -1, 0
	v_and_b32_e32 v2, 0x3ff, v0
	v_lshlrev_b32_e32 v1, 3, v2
	v_cmp_eq_u32_e64 s[0:1], 63, v2
	v_lshlrev_b32_e32 v2, 4, v2
	s_or_b64 s[42:43], s[42:43], s[20:21]
	v_lshl_add_u32 v34, v32, 10, v2
	v_cndmask_b32_e64 v2, 0, 1, s[42:43]
	s_mov_b64 s[34:35], 0
	v_lshl_add_u32 v33, v32, 9, v1
	s_ashr_i32 s23, s22, 31
	s_lshl_b32 s11, s18, 9
	s_add_i32 s17, s2, -8
	s_ashr_i32 s33, s10, 31
	s_waitcnt lgkmcnt(0)
	v_cmp_neq_f32_e64 s[36:37], s25, 0
	s_ashr_i32 s39, s26, 31
	s_mov_b32 s38, s26
	s_ashr_i32 s41, s27, 31
	s_mov_b32 s40, s27
	s_mul_i32 s54, s18, s8
	s_lshl_b32 s55, s18, 10
	v_mov_b32_e32 v29, 0
	v_cmp_ne_u32_e64 s[20:21], 1, v2
	s_branch .LBB0_5
.LBB0_4:                                ;   in Loop: Header=BB0_5 Depth=1
	s_or_b64 exec, exec, s[44:45]
	v_add_u32_e32 v26, s54, v26
	v_cmp_le_u32_e32 vcc, s9, v26
	s_or_b64 s[34:35], vcc, s[34:35]
	s_andn2_b64 exec, exec, s[34:35]
	s_cbranch_execz .LBB0_33
.LBB0_5:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_9 Depth 2
                                        ;       Child Loop BB0_14 Depth 3
	v_mad_u64_u32 v[2:3], s[44:45], v26, s10, 0
	v_mov_b32_e32 v4, v3
	v_mad_u64_u32 v[4:5], s[44:45], v26, s33, v[4:5]
	v_mov_b32_e32 v3, v4
	v_cmp_gt_u32_e64 s[18:19], s3, v26
	v_mov_b32_e32 v27, v29
	v_lshl_add_u64 v[30:31], v[2:3], 1, s[4:5]
	s_mov_b32 s30, s31
	s_mov_b32 s56, s31
	v_mov_b32_e32 v36, v29
	v_mov_b32_e32 v35, v29
	s_branch .LBB0_9
.LBB0_6:                                ;   in Loop: Header=BB0_9 Depth=2
	s_or_b64 exec, exec, s[48:49]
.LBB0_7:                                ;   in Loop: Header=BB0_9 Depth=2
	s_or_b64 exec, exec, s[46:47]
	s_waitcnt vmcnt(1) lgkmcnt(1)
	;;#ASMSTART
	v_dot2c_f32_f16 v36, v22, v10
	;;#ASMEND
	s_waitcnt lgkmcnt(0)
	;;#ASMSTART
	v_dot2c_f32_f16 v35, v14, v10
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v36, v23, v11
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v35, v15, v11
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v36, v24, v12
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v35, v16, v12
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v36, v25, v13
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v35, v17, v13
	;;#ASMEND
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	v_dot2c_f32_f16 v36, v18, v2
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v35, v6, v2
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v36, v19, v3
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v35, v7, v3
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v36, v20, v4
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v35, v8, v4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v36, v21, v5
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v35, v9, v5
	;;#ASMEND
.LBB0_8:                                ;   in Loop: Header=BB0_9 Depth=2
	s_or_b64 exec, exec, s[44:45]
	s_addk_i32 s56, 0x400
	s_cmp_ge_u32 s56, s2
	s_cbranch_scc1 .LBB0_27
.LBB0_9:                                ;   Parent Loop BB0_5 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_14 Depth 3
	s_cmp_eq_u32 s56, 0
	s_cselect_b64 s[44:45], -1, 0
	s_add_i32 s46, s30, 0xa000
	s_cmp_eq_u32 s56, s46
	s_cselect_b64 s[48:49], -1, 0
	s_or_b64 s[48:49], s[44:45], s[48:49]
	s_andn2_b64 vcc, exec, s[48:49]
	s_cbranch_vccz .LBB0_11
; %bb.10:                               ;   in Loop: Header=BB0_9 Depth=2
	s_and_saveexec_b64 s[44:45], s[18:19]
	s_cbranch_execz .LBB0_8
	s_branch .LBB0_24
.LBB0_11:                               ;   in Loop: Header=BB0_9 Depth=2
	s_and_b64 s[44:45], s[44:45], exec
	s_cselect_b32 s30, s30, s46
	s_sub_i32 s44, s2, s30
	s_min_u32 s57, s44, 0xa000
	s_lshl_b64 s[44:45], s[30:31], 1
	s_add_u32 s44, s6, s44
	s_addc_u32 s45, s7, s45
	s_mov_b32 s58, 0
	s_mov_b64 s[46:47], 0
	v_mov_b32_e32 v4, v34
	s_barrier
                                        ; implicit-def: $sgpr48_sgpr49
	s_branch .LBB0_14
.LBB0_12:                               ;   in Loop: Header=BB0_14 Depth=3
	s_add_i32 s58, s58, s11
	s_cmpk_gt_u32 s58, 0x9fff
	s_cselect_b64 s[52:53], -1, 0
	s_andn2_b64 s[48:49], s[48:49], exec
	s_and_b64 s[52:53], s[52:53], exec
	v_add_u32_e32 v4, s55, v4
	s_or_b64 s[48:49], s[48:49], s[52:53]
.LBB0_13:                               ;   in Loop: Header=BB0_14 Depth=3
	s_or_b64 exec, exec, s[50:51]
	s_and_b64 s[50:51], exec, s[48:49]
	s_or_b64 s[46:47], s[50:51], s[46:47]
	s_andn2_b64 exec, exec, s[46:47]
	s_cbranch_execz .LBB0_23
.LBB0_14:                               ;   Parent Loop BB0_5 Depth=1
                                        ;     Parent Loop BB0_9 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	v_add_u32_e32 v28, s58, v33
	v_cmp_gt_u32_e32 vcc, s57, v28
	s_or_b64 s[48:49], s[48:49], exec
	s_and_saveexec_b64 s[50:51], vcc
	s_cbranch_execz .LBB0_13
; %bb.15:                               ;   in Loop: Header=BB0_14 Depth=3
	v_lshl_add_u64 v[2:3], v[28:29], 1, s[44:45]
	s_mov_b64 s[52:53], -1
	s_and_b64 vcc, exec, s[42:43]
	s_cbranch_vccz .LBB0_17
; %bb.16:                               ;   in Loop: Header=BB0_14 Depth=3
	global_load_dwordx4 v[6:9], v[2:3], off
	s_mov_b64 s[52:53], 0
	s_waitcnt vmcnt(0)
	ds_write_b128 v4, v[6:9]
.LBB0_17:                               ;   in Loop: Header=BB0_14 Depth=3
	s_andn2_b64 vcc, exec, s[52:53]
	s_cbranch_vccnz .LBB0_19
; %bb.18:                               ;   in Loop: Header=BB0_14 Depth=3
	v_readfirstlane_b32 s52, v4
	s_mov_b32 m0, s52
	s_nop 0
	global_load_lds_dwordx4 v[2:3], off
.LBB0_19:                               ;   in Loop: Header=BB0_14 Depth=3
	v_lshl_add_u64 v[2:3], s[22:23], 1, v[2:3]
	v_add_u32_e32 v5, 0x14000, v4
	s_and_b64 vcc, exec, s[20:21]
	s_mov_b64 s[52:53], -1
	s_cbranch_vccnz .LBB0_21
; %bb.20:                               ;   in Loop: Header=BB0_14 Depth=3
	global_load_dwordx4 v[6:9], v[2:3], off
	s_mov_b64 s[52:53], 0
	s_waitcnt vmcnt(0)
	ds_write_b128 v5, v[6:9]
.LBB0_21:                               ;   in Loop: Header=BB0_14 Depth=3
	s_andn2_b64 vcc, exec, s[52:53]
	s_cbranch_vccnz .LBB0_12
; %bb.22:                               ;   in Loop: Header=BB0_14 Depth=3
	v_readfirstlane_b32 s52, v5
	s_mov_b32 m0, s52
	s_nop 0
	global_load_lds_dwordx4 v[2:3], off
	s_branch .LBB0_12
.LBB0_23:                               ;   in Loop: Header=BB0_9 Depth=2
	s_or_b64 exec, exec, s[46:47]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	s_and_saveexec_b64 s[44:45], s[18:19]
	s_cbranch_execz .LBB0_8
.LBB0_24:                               ;   in Loop: Header=BB0_9 Depth=2
	v_add_u32_e32 v37, s56, v1
	v_min_u32_e32 v28, s17, v37
	v_add_u32_e32 v38, 0x200, v37
	v_lshl_add_u64 v[6:7], v[28:29], 1, v[30:31]
	v_min_u32_e32 v28, s17, v38
	v_lshl_add_u64 v[8:9], v[28:29], 1, v[30:31]
	global_load_dwordx4 v[10:13], v[6:7], off nt
	global_load_dwordx4 v[2:5], v[8:9], off nt
	v_cmp_gt_u32_e32 vcc, s2, v37
	v_mov_b32_e32 v24, 0
	v_mov_b32_e32 v25, 0
	v_mov_b32_e32 v14, 0
	v_mov_b32_e32 v15, 0
	v_mov_b32_e32 v16, 0
	v_mov_b32_e32 v17, 0
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v21, 0
	v_mov_b32_e32 v6, 0
	v_mov_b32_e32 v7, 0
	v_mov_b32_e32 v8, 0
	v_mov_b32_e32 v9, 0
	s_and_saveexec_b64 s[46:47], vcc
	s_cbranch_execz .LBB0_7
; %bb.25:                               ;   in Loop: Header=BB0_9 Depth=2
	v_subrev_u32_e32 v6, s30, v37
	v_lshlrev_b32_e32 v28, 1, v6
	v_add_u32_e32 v6, 0x14000, v28
	ds_read_b128 v[22:25], v28
	ds_read_b128 v[14:17], v6
	v_cmp_gt_u32_e32 vcc, s2, v38
	v_mov_b32_e32 v9, 0
	v_mov_b32_e32 v8, 0
	v_mov_b32_e32 v7, 0
	v_mov_b32_e32 v6, 0
	v_mov_b32_e32 v21, 0
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v18, 0
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_6
; %bb.26:                               ;   in Loop: Header=BB0_9 Depth=2
	v_add_u32_e32 v6, 0x14400, v28
	ds_read_b128 v[18:21], v28 offset:1024
	ds_read_b128 v[6:9], v6
	s_branch .LBB0_6
.LBB0_27:                               ;   in Loop: Header=BB0_5 Depth=1
	s_and_saveexec_b64 s[44:45], s[18:19]
	s_cbranch_execz .LBB0_4
; %bb.28:                               ;   in Loop: Header=BB0_5 Depth=1
	v_add_f32_dpp v2, v36, v36 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v35, v35 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_add_f32_dpp v2, v2, v2 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_add_f32_dpp v2, v2, v2 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_add_f32_dpp v2, v2, v2 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_add_f32_dpp v2, v2, v2 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v3, v2 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v5, v4 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_and_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB0_4
; %bb.29:                               ;   in Loop: Header=BB0_5 Depth=1
	v_add_f32_e32 v2, v2, v3
	s_andn2_b64 vcc, exec, s[36:37]
	v_add_f32_e32 v3, v4, v5
	s_cbranch_vccnz .LBB0_31
; %bb.30:                               ;   in Loop: Header=BB0_5 Depth=1
	v_lshlrev_b64 v[4:5], 1, v[26:27]
	v_lshl_add_u64 v[6:7], s[12:13], 0, v[4:5]
	global_load_ushort v8, v[6:7], off
	v_lshl_add_u64 v[4:5], s[14:15], 0, v[4:5]
	v_lshl_add_u64 v[6:7], s[38:39], 1, v[6:7]
	s_waitcnt vmcnt(0)
	v_cvt_f32_f16_e32 v8, v8
	v_mul_f32_e32 v8, s25, v8
	v_fma_mixlo_f16 v8, s24, v2, v8
	global_store_short v[4:5], v8, off
	global_load_ushort v6, v[6:7], off
	v_lshl_add_u64 v[4:5], s[40:41], 1, v[4:5]
	s_waitcnt vmcnt(0)
	v_cvt_f32_f16_e32 v6, v6
	v_mul_f32_e32 v6, s25, v6
	v_fma_mixlo_f16 v6, s24, v3, v6
	global_store_short v[4:5], v6, off
	s_cbranch_execnz .LBB0_4
	s_branch .LBB0_32
.LBB0_31:                               ;   in Loop: Header=BB0_5 Depth=1
.LBB0_32:                               ;   in Loop: Header=BB0_5 Depth=1
	v_lshl_add_u64 v[4:5], v[26:27], 1, s[14:15]
	v_fma_mixlo_f16 v2, s24, v2, 0
	global_store_short v[4:5], v2, off
	v_lshl_add_u64 v[4:5], s[40:41], 1, v[4:5]
	v_fma_mixlo_f16 v2, s24, v3, 0
	global_store_short v[4:5], v2, off
	s_branch .LBB0_4
.LBB0_33:
	s_or_b64 exec, exec, s[28:29]
	s_mov_b64 s[0:1], 0
.LBB0_34:
	s_andn2_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB0_68
; %bb.35:
	s_lshl_b32 s11, s2, 1
	v_and_b32_e32 v8, 0x3ff, v0
	v_lshlrev_b32_e32 v30, 3, v8
	s_cmp_eq_u32 s22, s2
	s_mov_b64 s[18:19], 0
	s_cbranch_scc1 .LBB0_37
; %bb.36:
	s_and_b32 s20, s6, 15
	s_mov_b32 s21, 0
	s_mov_b64 s[0:1], -1
	s_min_u32 s9, s11, 0x14000
	v_lshl_add_u32 v9, v32, 9, v30
	s_cbranch_execz .LBB0_38
	s_branch .LBB0_39
.LBB0_37:
                                        ; implicit-def: $sgpr20_sgpr21
	s_mov_b64 s[0:1], 0
	s_min_u32 s9, s11, 0x14000
	v_lshl_add_u32 v9, v32, 9, v30
.LBB0_38:
	s_and_b32 s0, s2, 7
	s_cmp_lg_u32 s0, 0
	s_mov_b32 s21, 0
	s_cselect_b64 s[0:1], -1, 0
	s_and_b32 s20, s6, 15
	s_cmp_lg_u64 s[20:21], 0
	s_cselect_b64 s[18:19], -1, 0
	s_or_b64 s[0:1], s[0:1], s[18:19]
	s_mov_b64 s[18:19], -1
.LBB0_39:
	s_andn2_b64 vcc, exec, s[0:1]
	v_cmp_gt_u32_e64 s[0:1], s9, v9
	s_cbranch_vccz .LBB0_41
; %bb.40:
	s_andn2_b64 vcc, exec, s[18:19]
	s_cbranch_vccz .LBB0_49
	s_branch .LBB0_53
.LBB0_41:
	s_and_saveexec_b64 s[18:19], s[0:1]
	s_cbranch_execz .LBB0_48
; %bb.42:
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
	v_lshl_add_u32 v10, v32, 10, v0
	v_mov_b32_e32 v0, v9
	s_branch .LBB0_44
.LBB0_43:                               ;   in Loop: Header=BB0_44 Depth=1
	v_add_u32_e32 v0, 0x2000, v0
	v_cmp_le_u32_e32 vcc, s9, v0
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[4:5]
	s_or_b64 s[0:1], vcc, s[0:1]
	v_add_u32_e32 v10, 0x4000, v10
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB0_48
.LBB0_44:                               ; =>This Inner Loop Header: Depth=1
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
	s_cbranch_vccz .LBB0_46
; %bb.45:                               ;   in Loop: Header=BB0_44 Depth=1
	global_load_dwordx4 v[12:15], v[6:7], off
	s_mov_b64 s[28:29], 0
	s_waitcnt vmcnt(0)
	ds_write_b128 v10, v[12:15]
.LBB0_46:                               ;   in Loop: Header=BB0_44 Depth=1
	s_andn2_b64 vcc, exec, s[28:29]
	s_cbranch_vccnz .LBB0_43
; %bb.47:                               ;   in Loop: Header=BB0_44 Depth=1
	v_readfirstlane_b32 s28, v10
	s_mov_b32 m0, s28
	s_nop 0
	global_load_lds_dwordx4 v[6:7], off
	s_branch .LBB0_43
.LBB0_48:
	s_or_b64 exec, exec, s[18:19]
	s_cbranch_execnz .LBB0_53
.LBB0_49:
	v_cmp_gt_u32_e32 vcc, s9, v9
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_52
; %bb.50:
	v_lshlrev_b32_e32 v0, 4, v8
	v_lshl_add_u32 v2, v32, 10, v0
	v_lshlrev_b32_e32 v0, 1, v9
	v_mov_b32_e32 v1, 0
	v_lshl_add_u64 v[0:1], s[6:7], 0, v[0:1]
	s_mov_b64 s[6:7], 0
	s_mov_b64 s[18:19], 0x4000
.LBB0_51:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s17, v2
	s_mov_b32 m0, s17
	v_add_u32_e32 v9, 0x2000, v9
	global_load_lds_dwordx4 v[0:1], off
	v_cmp_le_u32_e32 vcc, s9, v9
	v_add_u32_e32 v2, 0x4000, v2
	s_or_b64 s[6:7], vcc, s[6:7]
	v_lshl_add_u64 v[0:1], v[0:1], 0, s[18:19]
	s_andn2_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB0_51
.LBB0_52:
	s_or_b64 exec, exec, s[0:1]
.LBB0_53:
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
	s_max_i32 s31, s0, 4
	s_mul_i32 s0, s31, s16
	v_add_u32_e32 v24, s0, v32
	v_cmp_gt_u32_e32 vcc, s31, v32
	v_cmp_gt_u32_e64 s[0:1], s3, v24
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_68
; %bb.54:
	s_cmp_lg_u32 s2, 0
	s_cselect_b64 s[6:7], -1, 0
	v_cndmask_b32_e64 v0, 0, 1, s[6:7]
	v_cmp_neq_f32_e64 s[6:7], s25, 0
	s_mul_i32 s31, s31, s8
	v_cmp_ne_u32_e64 s[8:9], 1, v0
	v_cndmask_b32_e64 v0, 0, 1, s[6:7]
	s_add_i32 s28, s2, -8
	s_add_i32 s29, s3, -1
	s_ashr_i32 s30, s10, 31
	v_cmp_eq_u32_e64 s[0:1], 63, v8
	s_ashr_i32 s17, s26, 31
	s_mov_b32 s16, s26
	s_ashr_i32 s19, s27, 31
	s_mov_b32 s18, s27
	v_lshlrev_b32_e32 v31, 4, v8
	s_mov_b64 s[20:21], 0
	v_mov_b32_e32 v27, 0
	v_cmp_ne_u32_e64 s[6:7], 1, v0
	s_branch .LBB0_56
.LBB0_55:                               ;   in Loop: Header=BB0_56 Depth=1
	s_or_b64 exec, exec, s[22:23]
	v_add_u32_e32 v24, s31, v24
	v_cmp_le_u32_e32 vcc, s3, v24
	s_or_b64 s[20:21], vcc, s[20:21]
	s_andn2_b64 exec, exec, s[20:21]
	s_cbranch_execz .LBB0_68
.LBB0_56:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_60 Depth 2
	s_and_b64 vcc, exec, s[8:9]
	v_mov_b32_e32 v25, v27
	v_mov_b32_e32 v32, v27
	s_cbranch_vccnz .LBB0_63
; %bb.57:                               ;   in Loop: Header=BB0_56 Depth=1
	v_min_u32_e32 v3, s29, v24
	v_mad_u64_u32 v[0:1], s[22:23], v3, s10, 0
	v_mov_b32_e32 v2, v1
	v_mad_u64_u32 v[2:3], s[22:23], v3, s30, v[2:3]
	v_mov_b32_e32 v1, v2
	v_lshl_add_u64 v[28:29], v[0:1], 1, s[4:5]
	v_mov_b32_e32 v32, 0
	s_mov_b32 s33, 0
	v_mov_b32_e32 v33, v31
	v_mov_b32_e32 v25, 0
	s_branch .LBB0_60
.LBB0_58:                               ;   in Loop: Header=BB0_60 Depth=2
	s_or_b64 exec, exec, s[26:27]
.LBB0_59:                               ;   in Loop: Header=BB0_60 Depth=2
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(1) lgkmcnt(1)
	;;#ASMSTART
	v_dot2c_f32_f16 v32, v20, v4
	;;#ASMEND
	s_waitcnt lgkmcnt(0)
	;;#ASMSTART
	v_dot2c_f32_f16 v25, v12, v4
	;;#ASMEND
	s_addk_i32 s33, 0x400
	;;#ASMSTART
	v_dot2c_f32_f16 v32, v21, v5
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v25, v13, v5
	;;#ASMEND
	s_cmp_ge_u32 s33, s2
	;;#ASMSTART
	v_dot2c_f32_f16 v32, v22, v6
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v25, v14, v6
	;;#ASMEND
	v_add_u32_e32 v33, 0x800, v33
	;;#ASMSTART
	v_dot2c_f32_f16 v32, v23, v7
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v25, v15, v7
	;;#ASMEND
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	v_dot2c_f32_f16 v32, v16, v0
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v25, v8, v0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v32, v17, v1
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v25, v9, v1
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v32, v18, v2
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v25, v10, v2
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v32, v19, v3
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v25, v11, v3
	;;#ASMEND
	s_cbranch_scc1 .LBB0_63
.LBB0_60:                               ;   Parent Loop BB0_56 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_add_u32_e32 v12, s33, v30
	v_min_u32_e32 v26, s28, v12
	v_add_u32_e32 v34, 0x200, v12
	v_lshl_add_u64 v[8:9], v[26:27], 1, v[28:29]
	v_min_u32_e32 v26, s28, v34
	v_lshl_add_u64 v[10:11], v[26:27], 1, v[28:29]
	global_load_dwordx4 v[4:7], v[8:9], off nt
	global_load_dwordx4 v[0:3], v[10:11], off nt
	v_cmp_gt_u32_e32 vcc, s2, v12
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v12, 0
	v_mov_b32_e32 v13, 0
	v_mov_b32_e32 v14, 0
	v_mov_b32_e32 v15, 0
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v21, 0
	v_mov_b32_e32 v16, 0
	v_mov_b32_e32 v17, 0
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v8, 0
	v_mov_b32_e32 v9, 0
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v11, 0
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_59
; %bb.61:                               ;   in Loop: Header=BB0_60 Depth=2
	v_add_u32_e32 v26, s11, v33
	ds_read_b128 v[20:23], v33
	ds_read_b128 v[12:15], v26
	v_cmp_gt_u32_e32 vcc, s2, v34
	v_mov_b32_e32 v11, 0
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v9, 0
	v_mov_b32_e32 v8, 0
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v17, 0
	v_mov_b32_e32 v16, 0
	s_and_saveexec_b64 s[26:27], vcc
	s_cbranch_execz .LBB0_58
; %bb.62:                               ;   in Loop: Header=BB0_60 Depth=2
	ds_read_b128 v[16:19], v33 offset:1024
	ds_read_b128 v[8:11], v26 offset:1024
	s_branch .LBB0_58
.LBB0_63:                               ;   in Loop: Header=BB0_56 Depth=1
	; sched_barrier mask(0x00000000)
	s_nop 0
	v_add_f32_dpp v0, v32, v32 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v25, v25 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_add_f32_dpp v0, v0, v0 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_add_f32_dpp v0, v0, v0 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_add_f32_dpp v0, v0, v0 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_add_f32_dpp v0, v0, v0 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v1, v0 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v3, v2 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_and_saveexec_b64 s[22:23], s[0:1]
	s_cbranch_execz .LBB0_55
; %bb.64:                               ;   in Loop: Header=BB0_56 Depth=1
	v_add_f32_e32 v1, v0, v1
	v_add_f32_e32 v0, v2, v3
	s_and_b64 vcc, exec, s[6:7]
	v_mov_b32_e32 v25, v27
	s_cbranch_vccnz .LBB0_66
; %bb.65:                               ;   in Loop: Header=BB0_56 Depth=1
	v_lshlrev_b64 v[2:3], 1, v[24:25]
	v_lshl_add_u64 v[4:5], s[12:13], 0, v[2:3]
	global_load_ushort v6, v[4:5], off
	v_lshl_add_u64 v[2:3], s[14:15], 0, v[2:3]
	v_lshl_add_u64 v[4:5], s[16:17], 1, v[4:5]
	s_waitcnt vmcnt(0)
	v_cvt_f32_f16_e32 v6, v6
	v_mul_f32_e32 v6, s25, v6
	v_fma_mixlo_f16 v6, s24, v1, v6
	global_store_short v[2:3], v6, off
	global_load_ushort v4, v[4:5], off
	v_lshl_add_u64 v[2:3], s[18:19], 1, v[2:3]
	s_waitcnt vmcnt(0)
	v_cvt_f32_f16_e32 v4, v4
	v_mul_f32_e32 v4, s25, v4
	v_fma_mixlo_f16 v4, s24, v0, v4
	global_store_short v[2:3], v4, off
	s_cbranch_execnz .LBB0_55
	s_branch .LBB0_67
.LBB0_66:                               ;   in Loop: Header=BB0_56 Depth=1
.LBB0_67:                               ;   in Loop: Header=BB0_56 Depth=1
	v_lshl_add_u64 v[2:3], v[24:25], 1, s[14:15]
	v_fma_mixlo_f16 v1, s24, v1, 0
	global_store_short v[2:3], v1, off
	v_lshl_add_u64 v[2:3], s[18:19], 1, v[2:3]
	v_fma_mixlo_f16 v0, s24, v0, 0
	global_store_short v[2:3], v0, off
	s_branch .LBB0_55
.LBB0_68:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wvSplitK_vllm_f16_tn_n2_y1u2
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
		.amdhsa_accum_offset 40
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
	.size	wvSplitK_vllm_f16_tn_n2_y1u2, .Lfunc_end0-wvSplitK_vllm_f16_tn_n2_y1u2
                                        ; -- End function
	.set wvSplitK_vllm_f16_tn_n2_y1u2.num_vgpr, 39
	.set wvSplitK_vllm_f16_tn_n2_y1u2.num_agpr, 0
	.set wvSplitK_vllm_f16_tn_n2_y1u2.numbered_sgpr, 59
	.set wvSplitK_vllm_f16_tn_n2_y1u2.num_named_barrier, 0
	.set wvSplitK_vllm_f16_tn_n2_y1u2.private_seg_size, 0
	.set wvSplitK_vllm_f16_tn_n2_y1u2.uses_vcc, 1
	.set wvSplitK_vllm_f16_tn_n2_y1u2.uses_flat_scratch, 0
	.set wvSplitK_vllm_f16_tn_n2_y1u2.has_dyn_sized_stack, 0
	.set wvSplitK_vllm_f16_tn_n2_y1u2.has_recursion, 0
	.set wvSplitK_vllm_f16_tn_n2_y1u2.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 4092
; TotalNumSgprs: 65
; NumVgprs: 39
; NumAgprs: 0
; TotalNumVgprs: 39
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 163840 bytes/workgroup (compile time only)
; SGPRBlocks: 12
; VGPRBlocks: 12
; NumSGPRsForWavesPerEU: 102
; NumVGPRsForWavesPerEU: 97
; AccumOffset: 40
; Occupancy: 4
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 1
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 9
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_9d33b45712f57307,@object ; @__hip_cuid_9d33b45712f57307
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_9d33b45712f57307
__hip_cuid_9d33b45712f57307:
	.byte	0                               ; 0x0
	.size	__hip_cuid_9d33b45712f57307, 1

	.ident	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.4 26084 f58b06dce1f9c15707c5f808fd002e18c2accf7e)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_9d33b45712f57307
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
    DataType: h
    DestDataType: h
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
  AssertSizeEqual: { 1: 2, 2: 1 }
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
    .name:           wvSplitK_vllm_f16_tn_n2_y1u2
    .private_segment_fixed_size: 0
    .sgpr_count:     65
    .sgpr_spill_count: 0
    .symbol:         wvSplitK_vllm_f16_tn_n2_y1u2.kd
    .vgpr_count:     39
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 1
...

	.end_amdgpu_metadata
