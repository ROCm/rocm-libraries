	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 4
	.text
	.protected	wvSplitK_vllm_f16_tn_n1_y1u2 ; -- Begin function wvSplitK_vllm_f16_tn_n1_y1u2
	.globl	wvSplitK_vllm_f16_tn_n1_y1u2
	.p2align	8
	.type	wvSplitK_vllm_f16_tn_n1_y1u2,@function
wvSplitK_vllm_f16_tn_n1_y1u2:           ; @wvSplitK_vllm_f16_tn_n1_y1u2
; %bb.64:
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.65:
.LBB0_0:
	s_load_dwordx2 s[22:23], s[0:1], 0x38
	s_mov_b64 s[0:1], -1
	s_cmp_gt_i32 s2, 0x14000
	v_bfe_u32 v24, v0, 10, 10
	s_cbranch_scc0 .LBB0_29
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
	s_max_i32 s18, s0, 4
	v_cmp_gt_u32_e32 vcc, s18, v24
	s_and_saveexec_b64 s[24:25], vcc
	s_cbranch_execz .LBB0_28
; %bb.2:
	v_cvt_f32_u32_e32 v1, s18
	s_mul_i32 s0, s18, s16
	s_sub_i32 s17, 0, s18
	v_add_u32_e32 v18, s0, v24
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
	v_cmp_gt_u32_e32 vcc, s9, v18
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_28
; %bb.3:
	s_and_b32 s0, s11, 7
	s_cmp_lg_u32 s0, 0
	s_mov_b32 s27, 0
	s_cselect_b64 s[0:1], -1, 0
	s_and_b32 s26, s6, 15
	s_cmp_lg_u64 s[26:27], 0
	s_cselect_b64 s[20:21], -1, 0
	v_and_b32_e32 v2, 0x3ff, v0
	s_or_b64 s[30:31], s[20:21], s[0:1]
	v_lshlrev_b32_e32 v1, 3, v2
	s_lshl_b32 s17, s18, 9
	v_cmp_eq_u32_e64 s[0:1], 63, v2
	s_mul_i32 s45, s18, s8
	v_lshlrev_b32_e32 v2, 4, v2
	s_lshl_b32 s46, s18, 10
	s_waitcnt lgkmcnt(0)
	v_cmp_neq_f32_e64 s[18:19], s23, 0
	v_lshl_add_u32 v26, v24, 10, v2
	s_mov_b64 s[28:29], 0
	v_cndmask_b32_e64 v2, 0, 1, s[18:19]
	v_lshl_add_u32 v25, v24, 9, v1
	s_add_i32 s33, s2, -8
	s_ashr_i32 s44, s10, 31
	v_mov_b32_e32 v21, 0
	v_cmp_ne_u32_e64 s[20:21], 1, v2
	s_branch .LBB0_6
.LBB0_4:                                ;   in Loop: Header=BB0_6 Depth=1
	v_cvt_f16_f32_e32 v4, v2
	v_lshl_add_u64 v[2:3], v[18:19], 1, s[14:15]
	global_store_short v[2:3], v4, off
.LBB0_5:                                ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[34:35]
	v_add_u32_e32 v18, s45, v18
	v_cmp_le_u32_e32 vcc, s9, v18
	s_or_b64 s[28:29], vcc, s[28:29]
	s_andn2_b64 exec, exec, s[28:29]
	s_cbranch_execz .LBB0_28
.LBB0_6:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_10 Depth 2
                                        ;       Child Loop BB0_15 Depth 3
	v_mad_u64_u32 v[2:3], s[34:35], v18, s10, 0
	v_mov_b32_e32 v4, v3
	v_mad_u64_u32 v[4:5], s[34:35], v18, s44, v[4:5]
	v_mov_b32_e32 v3, v4
	v_cmp_gt_u32_e64 s[18:19], s3, v18
	v_mov_b32_e32 v19, v21
	v_lshl_add_u64 v[22:23], v[2:3], 1, s[4:5]
	s_mov_b32 s26, s27
	s_mov_b32 s47, s27
	v_mov_b32_e32 v27, v21
	s_branch .LBB0_10
.LBB0_7:                                ;   in Loop: Header=BB0_10 Depth=2
	s_or_b64 exec, exec, s[38:39]
.LBB0_8:                                ;   in Loop: Header=BB0_10 Depth=2
	s_or_b64 exec, exec, s[36:37]
	s_waitcnt vmcnt(1) lgkmcnt(0)
	;;#ASMSTART
	v_dot2c_f32_f16 v27, v14, v10
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v27, v15, v11
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v27, v16, v12
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v27, v17, v13
	;;#ASMEND
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	v_dot2c_f32_f16 v27, v6, v2
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v27, v7, v3
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v27, v8, v4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v27, v9, v5
	;;#ASMEND
.LBB0_9:                                ;   in Loop: Header=BB0_10 Depth=2
	s_or_b64 exec, exec, s[34:35]
	s_addk_i32 s47, 0x400
	s_cmp_ge_u32 s47, s2
	s_cbranch_scc1 .LBB0_24
.LBB0_10:                               ;   Parent Loop BB0_6 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_15 Depth 3
	s_cmp_eq_u32 s47, 0
	s_cselect_b64 s[34:35], -1, 0
	s_add_i32 s36, s26, 0x14000
	s_cmp_eq_u32 s47, s36
	s_cselect_b64 s[38:39], -1, 0
	s_or_b64 s[38:39], s[34:35], s[38:39]
	s_andn2_b64 vcc, exec, s[38:39]
	s_cbranch_vccz .LBB0_12
; %bb.11:                               ;   in Loop: Header=BB0_10 Depth=2
	s_and_saveexec_b64 s[34:35], s[18:19]
	s_cbranch_execz .LBB0_9
	s_branch .LBB0_21
.LBB0_12:                               ;   in Loop: Header=BB0_10 Depth=2
	s_and_b64 s[34:35], s[34:35], exec
	s_cselect_b32 s26, s26, s36
	s_sub_i32 s34, s2, s26
	s_min_u32 s48, s34, 0x14000
	s_lshl_b64 s[34:35], s[26:27], 1
	s_add_u32 s34, s6, s34
	s_addc_u32 s35, s7, s35
	s_mov_b32 s49, 0
	s_mov_b64 s[36:37], 0
	v_mov_b32_e32 v4, v26
	s_barrier
                                        ; implicit-def: $sgpr38_sgpr39
	s_branch .LBB0_15
.LBB0_13:                               ;   in Loop: Header=BB0_15 Depth=3
	s_add_i32 s49, s49, s17
	s_cmp_gt_u32 s49, 0x13fff
	s_cselect_b64 s[42:43], -1, 0
	s_andn2_b64 s[38:39], s[38:39], exec
	s_and_b64 s[42:43], s[42:43], exec
	v_add_u32_e32 v4, s46, v4
	s_or_b64 s[38:39], s[38:39], s[42:43]
.LBB0_14:                               ;   in Loop: Header=BB0_15 Depth=3
	s_or_b64 exec, exec, s[40:41]
	s_and_b64 s[40:41], exec, s[38:39]
	s_or_b64 s[36:37], s[40:41], s[36:37]
	s_andn2_b64 exec, exec, s[36:37]
	s_cbranch_execz .LBB0_20
.LBB0_15:                               ;   Parent Loop BB0_6 Depth=1
                                        ;     Parent Loop BB0_10 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	v_add_u32_e32 v20, s49, v25
	v_cmp_gt_u32_e32 vcc, s48, v20
	s_or_b64 s[38:39], s[38:39], exec
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_14
; %bb.16:                               ;   in Loop: Header=BB0_15 Depth=3
	v_lshl_add_u64 v[2:3], v[20:21], 1, s[34:35]
	s_mov_b64 s[42:43], -1
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccz .LBB0_18
; %bb.17:                               ;   in Loop: Header=BB0_15 Depth=3
	global_load_dwordx4 v[6:9], v[2:3], off
	s_mov_b64 s[42:43], 0
	s_waitcnt vmcnt(0)
	ds_write_b128 v4, v[6:9]
.LBB0_18:                               ;   in Loop: Header=BB0_15 Depth=3
	s_andn2_b64 vcc, exec, s[42:43]
	s_cbranch_vccnz .LBB0_13
; %bb.19:                               ;   in Loop: Header=BB0_15 Depth=3
	v_readfirstlane_b32 s42, v4
	s_mov_b32 m0, s42
	s_nop 0
	global_load_lds_dwordx4 v[2:3], off
	s_branch .LBB0_13
.LBB0_20:                               ;   in Loop: Header=BB0_10 Depth=2
	s_or_b64 exec, exec, s[36:37]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	s_and_saveexec_b64 s[34:35], s[18:19]
	s_cbranch_execz .LBB0_9
.LBB0_21:                               ;   in Loop: Header=BB0_10 Depth=2
	v_add_u32_e32 v28, s47, v1
	v_min_u32_e32 v20, s33, v28
	v_add_u32_e32 v29, 0x200, v28
	v_lshl_add_u64 v[6:7], v[20:21], 1, v[22:23]
	v_min_u32_e32 v20, s33, v29
	v_lshl_add_u64 v[8:9], v[20:21], 1, v[22:23]
	global_load_dwordx4 v[10:13], v[6:7], off nt
	global_load_dwordx4 v[2:5], v[8:9], off nt
	v_cmp_gt_u32_e32 vcc, s2, v28
	v_mov_b32_e32 v16, 0
	v_mov_b32_e32 v17, 0
	v_mov_b32_e32 v14, 0
	v_mov_b32_e32 v15, 0
	v_mov_b32_e32 v6, 0
	v_mov_b32_e32 v7, 0
	v_mov_b32_e32 v8, 0
	v_mov_b32_e32 v9, 0
	s_and_saveexec_b64 s[36:37], vcc
	s_cbranch_execz .LBB0_8
; %bb.22:                               ;   in Loop: Header=BB0_10 Depth=2
	v_subrev_u32_e32 v6, s26, v28
	v_lshlrev_b32_e32 v20, 1, v6
	ds_read_b128 v[14:17], v20
	v_cmp_gt_u32_e32 vcc, s2, v29
	v_mov_b32_e32 v9, 0
	v_mov_b32_e32 v8, 0
	v_mov_b32_e32 v7, 0
	v_mov_b32_e32 v6, 0
	s_and_saveexec_b64 s[38:39], vcc
	s_cbranch_execz .LBB0_7
; %bb.23:                               ;   in Loop: Header=BB0_10 Depth=2
	ds_read_b128 v[6:9], v20 offset:1024
	s_branch .LBB0_7
.LBB0_24:                               ;   in Loop: Header=BB0_6 Depth=1
	s_and_saveexec_b64 s[34:35], s[18:19]
	s_cbranch_execz .LBB0_5
; %bb.25:                               ;   in Loop: Header=BB0_6 Depth=1
	v_add_f32_dpp v2, v27, v27 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v2, v2, v2 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v2, v2, v2 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v2, v2, v2 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v2, v2, v2 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_mov_b32_dpp v3, v2 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_and_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB0_5
; %bb.26:                               ;   in Loop: Header=BB0_6 Depth=1
	v_add_f32_e32 v2, v2, v3
	s_and_b64 vcc, exec, s[20:21]
	v_mul_f32_e32 v2, s22, v2
	s_cbranch_vccnz .LBB0_4
; %bb.27:                               ;   in Loop: Header=BB0_6 Depth=1
	v_lshl_add_u64 v[4:5], v[18:19], 1, s[12:13]
	global_load_ushort v3, v[4:5], off
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v2, s23, v3, v2 op_sel_hi:[0,1,0]
	s_branch .LBB0_4
.LBB0_28:
	s_or_b64 exec, exec, s[24:25]
	s_mov_b64 s[0:1], 0
.LBB0_29:
	s_andn2_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB0_63
; %bb.30:
	v_and_b32_e32 v2, 0x3ff, v0
	v_lshlrev_b32_e32 v22, 3, v2
	s_cmp_eq_u32 s11, s2
	s_mov_b64 s[18:19], 0
	s_cbranch_scc1 .LBB0_32
; %bb.31:
	s_and_b32 s20, s6, 15
	s_mov_b32 s21, 0
	s_mov_b64 s[0:1], -1
	s_min_u32 s9, s2, 0x14000
	v_lshl_add_u32 v3, v24, 9, v22
	s_cbranch_execz .LBB0_33
	s_branch .LBB0_34
.LBB0_32:
                                        ; implicit-def: $sgpr20_sgpr21
	s_mov_b64 s[0:1], 0
	s_min_u32 s9, s2, 0x14000
	v_lshl_add_u32 v3, v24, 9, v22
.LBB0_33:
	s_and_b32 s0, s2, 7
	s_cmp_lg_u32 s0, 0
	s_mov_b32 s21, 0
	s_cselect_b64 s[0:1], -1, 0
	s_and_b32 s20, s6, 15
	s_cmp_lg_u64 s[20:21], 0
	s_cselect_b64 s[18:19], -1, 0
	s_or_b64 s[0:1], s[0:1], s[18:19]
	s_mov_b64 s[18:19], -1
.LBB0_34:
	s_andn2_b64 vcc, exec, s[0:1]
	v_cmp_gt_u32_e64 s[0:1], s9, v3
	s_cbranch_vccz .LBB0_36
; %bb.35:
	s_andn2_b64 vcc, exec, s[18:19]
	s_cbranch_vccz .LBB0_44
	s_branch .LBB0_48
.LBB0_36:
	s_and_saveexec_b64 s[18:19], s[0:1]
	s_cbranch_execz .LBB0_43
; %bb.37:
	s_and_b32 s0, s11, 7
	s_cmp_lg_u32 s0, 0
	s_cselect_b64 s[24:25], -1, 0
	s_cmp_lg_u64 s[20:21], 0
	v_lshlrev_b32_e32 v0, 4, v2
	s_cselect_b64 s[20:21], -1, 0
	v_lshl_add_u32 v4, v24, 10, v0
	v_lshlrev_b32_e32 v0, 1, v3
	v_mov_b32_e32 v1, 0
	s_mov_b64 s[0:1], 0
	s_or_b64 s[20:21], s[24:25], s[20:21]
	v_lshl_add_u64 v[0:1], s[6:7], 0, v[0:1]
	s_mov_b64 s[24:25], 0x4000
	v_mov_b32_e32 v5, v3
	s_branch .LBB0_39
.LBB0_38:                               ;   in Loop: Header=BB0_39 Depth=1
	v_add_u32_e32 v5, 0x2000, v5
	v_cmp_le_u32_e32 vcc, s9, v5
	v_add_u32_e32 v4, 0x4000, v4
	s_or_b64 s[0:1], vcc, s[0:1]
	v_lshl_add_u64 v[0:1], v[0:1], 0, s[24:25]
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB0_43
.LBB0_39:                               ; =>This Inner Loop Header: Depth=1
	s_mov_b64 s[26:27], -1
	s_and_b64 vcc, exec, s[20:21]
	s_cbranch_vccz .LBB0_41
; %bb.40:                               ;   in Loop: Header=BB0_39 Depth=1
	global_load_dwordx4 v[6:9], v[0:1], off
	s_mov_b64 s[26:27], 0
	s_waitcnt vmcnt(0)
	ds_write_b128 v4, v[6:9]
.LBB0_41:                               ;   in Loop: Header=BB0_39 Depth=1
	s_andn2_b64 vcc, exec, s[26:27]
	s_cbranch_vccnz .LBB0_38
; %bb.42:                               ;   in Loop: Header=BB0_39 Depth=1
	v_readfirstlane_b32 s11, v4
	s_mov_b32 m0, s11
	s_nop 0
	global_load_lds_dwordx4 v[0:1], off
	s_branch .LBB0_38
.LBB0_43:
	s_or_b64 exec, exec, s[18:19]
	s_cbranch_execnz .LBB0_48
.LBB0_44:
	v_cmp_gt_u32_e32 vcc, s9, v3
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_47
; %bb.45:
	v_lshlrev_b32_e32 v0, 4, v2
	v_lshl_add_u32 v4, v24, 10, v0
	v_lshlrev_b32_e32 v0, 1, v3
	v_mov_b32_e32 v1, 0
	v_lshl_add_u64 v[0:1], s[6:7], 0, v[0:1]
	s_mov_b64 s[6:7], 0
	s_mov_b64 s[18:19], 0x4000
.LBB0_46:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s11, v4
	s_mov_b32 m0, s11
	v_add_u32_e32 v3, 0x2000, v3
	global_load_lds_dwordx4 v[0:1], off
	v_cmp_le_u32_e32 vcc, s9, v3
	v_add_u32_e32 v4, 0x4000, v4
	s_or_b64 s[6:7], vcc, s[6:7]
	v_lshl_add_u64 v[0:1], v[0:1], 0, s[18:19]
	s_andn2_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB0_46
.LBB0_47:
	s_or_b64 exec, exec, s[0:1]
.LBB0_48:
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
	s_max_i32 s26, s0, 4
	s_mul_i32 s0, s26, s16
	v_add_u32_e32 v16, s0, v24
	v_cmp_gt_u32_e32 vcc, s26, v24
	v_cmp_gt_u32_e64 s[0:1], s3, v16
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_63
; %bb.49:
	s_cmp_lg_u32 s2, 0
	s_cselect_b64 s[6:7], -1, 0
	v_cndmask_b32_e64 v0, 0, 1, s[6:7]
	v_cmp_neq_f32_e64 s[6:7], s23, 0
	s_mul_i32 s26, s26, s8
	v_cmp_ne_u32_e64 s[8:9], 1, v0
	v_cndmask_b32_e64 v0, 0, 1, s[6:7]
	s_add_i32 s11, s2, -8
	s_add_i32 s24, s3, -1
	s_ashr_i32 s25, s10, 31
	v_cmp_eq_u32_e64 s[0:1], 63, v2
	v_lshlrev_b32_e32 v23, 4, v2
	s_mov_b64 s[16:17], 0
	v_mov_b32_e32 v19, 0
	v_cmp_ne_u32_e64 s[6:7], 1, v0
	s_branch .LBB0_51
.LBB0_50:                               ;   in Loop: Header=BB0_51 Depth=1
	s_or_b64 exec, exec, s[18:19]
	v_add_u32_e32 v16, s26, v16
	v_cmp_le_u32_e32 vcc, s3, v16
	s_or_b64 s[16:17], vcc, s[16:17]
	s_andn2_b64 exec, exec, s[16:17]
	s_cbranch_execz .LBB0_63
.LBB0_51:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_55 Depth 2
	s_and_b64 vcc, exec, s[8:9]
	v_mov_b32_e32 v17, v19
	s_cbranch_vccnz .LBB0_58
; %bb.52:                               ;   in Loop: Header=BB0_51 Depth=1
	v_min_u32_e32 v3, s24, v16
	v_mad_u64_u32 v[0:1], s[18:19], v3, s10, 0
	v_mov_b32_e32 v2, v1
	v_mad_u64_u32 v[2:3], s[18:19], v3, s25, v[2:3]
	v_mov_b32_e32 v1, v2
	v_lshl_add_u64 v[20:21], v[0:1], 1, s[4:5]
	v_mov_b32_e32 v17, 0
	s_mov_b32 s27, 0
	v_mov_b32_e32 v24, v23
	s_branch .LBB0_55
.LBB0_53:                               ;   in Loop: Header=BB0_55 Depth=2
	s_or_b64 exec, exec, s[20:21]
.LBB0_54:                               ;   in Loop: Header=BB0_55 Depth=2
	s_or_b64 exec, exec, s[18:19]
	s_waitcnt vmcnt(1) lgkmcnt(0)
	;;#ASMSTART
	v_dot2c_f32_f16 v17, v12, v4
	;;#ASMEND
	s_addk_i32 s27, 0x400
	;;#ASMSTART
	v_dot2c_f32_f16 v17, v13, v5
	;;#ASMEND
	s_cmp_ge_u32 s27, s2
	;;#ASMSTART
	v_dot2c_f32_f16 v17, v14, v6
	;;#ASMEND
	v_add_u32_e32 v24, 0x800, v24
	;;#ASMSTART
	v_dot2c_f32_f16 v17, v15, v7
	;;#ASMEND
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	v_dot2c_f32_f16 v17, v8, v0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v17, v9, v1
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v17, v10, v2
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v17, v11, v3
	;;#ASMEND
	s_cbranch_scc1 .LBB0_58
.LBB0_55:                               ;   Parent Loop BB0_51 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_add_u32_e32 v12, s27, v22
	v_min_u32_e32 v18, s11, v12
	v_add_u32_e32 v25, 0x200, v12
	v_lshl_add_u64 v[8:9], v[18:19], 1, v[20:21]
	v_min_u32_e32 v18, s11, v25
	v_lshl_add_u64 v[10:11], v[18:19], 1, v[20:21]
	global_load_dwordx4 v[4:7], v[8:9], off nt
	global_load_dwordx4 v[0:3], v[10:11], off nt
	v_cmp_gt_u32_e32 vcc, s2, v12
	v_mov_b32_e32 v14, 0
	v_mov_b32_e32 v15, 0
	v_mov_b32_e32 v12, 0
	v_mov_b32_e32 v13, 0
	v_mov_b32_e32 v8, 0
	v_mov_b32_e32 v9, 0
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v11, 0
	s_and_saveexec_b64 s[18:19], vcc
	s_cbranch_execz .LBB0_54
; %bb.56:                               ;   in Loop: Header=BB0_55 Depth=2
	ds_read_b128 v[12:15], v24
	v_cmp_gt_u32_e32 vcc, s2, v25
	v_mov_b32_e32 v11, 0
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v9, 0
	v_mov_b32_e32 v8, 0
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_53
; %bb.57:                               ;   in Loop: Header=BB0_55 Depth=2
	ds_read_b128 v[8:11], v24 offset:1024
	s_branch .LBB0_53
.LBB0_58:                               ;   in Loop: Header=BB0_51 Depth=1
	; sched_barrier mask(0x00000000)
	s_nop 0
	v_add_f32_dpp v0, v17, v17 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v0, v0, v0 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v0, v0, v0 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v0, v0, v0 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v0, v0, v0 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_mov_b32_dpp v1, v0 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_and_saveexec_b64 s[18:19], s[0:1]
	s_cbranch_execz .LBB0_50
; %bb.59:                               ;   in Loop: Header=BB0_51 Depth=1
	v_add_f32_e32 v0, v0, v1
	s_and_b64 vcc, exec, s[6:7]
	v_mul_f32_e32 v0, s22, v0
	s_cbranch_vccnz .LBB0_61
; %bb.60:                               ;   in Loop: Header=BB0_51 Depth=1
	v_mov_b32_e32 v17, v19
	v_lshlrev_b64 v[2:3], 1, v[16:17]
	v_lshl_add_u64 v[4:5], s[12:13], 0, v[2:3]
	global_load_ushort v1, v[4:5], off
	v_lshl_add_u64 v[2:3], s[14:15], 0, v[2:3]
	s_waitcnt vmcnt(0)
	v_fma_mixlo_f16 v1, s23, v1, v0 op_sel_hi:[0,1,0]
	global_store_short v[2:3], v1, off
	s_cbranch_execnz .LBB0_50
	s_branch .LBB0_62
.LBB0_61:                               ;   in Loop: Header=BB0_51 Depth=1
.LBB0_62:                               ;   in Loop: Header=BB0_51 Depth=1
	v_cvt_f16_f32_e32 v2, v0
	v_mov_b32_e32 v17, v19
	v_lshl_add_u64 v[0:1], v[16:17], 1, s[14:15]
	global_store_short v[0:1], v2, off
	s_branch .LBB0_50
.LBB0_63:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wvSplitK_vllm_f16_tn_n1_y1u2
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
		.amdhsa_accum_offset 32
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
	.size	wvSplitK_vllm_f16_tn_n1_y1u2, .Lfunc_end0-wvSplitK_vllm_f16_tn_n1_y1u2
                                        ; -- End function
	.set wvSplitK_vllm_f16_tn_n1_y1u2.num_vgpr, 30
	.set wvSplitK_vllm_f16_tn_n1_y1u2.num_agpr, 0
	.set wvSplitK_vllm_f16_tn_n1_y1u2.numbered_sgpr, 50
	.set wvSplitK_vllm_f16_tn_n1_y1u2.num_named_barrier, 0
	.set wvSplitK_vllm_f16_tn_n1_y1u2.private_seg_size, 0
	.set wvSplitK_vllm_f16_tn_n1_y1u2.uses_vcc, 1
	.set wvSplitK_vllm_f16_tn_n1_y1u2.uses_flat_scratch, 0
	.set wvSplitK_vllm_f16_tn_n1_y1u2.has_dyn_sized_stack, 0
	.set wvSplitK_vllm_f16_tn_n1_y1u2.has_recursion, 0
	.set wvSplitK_vllm_f16_tn_n1_y1u2.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 3096
; TotalNumSgprs: 56
; NumVgprs: 30
; NumAgprs: 0
; TotalNumVgprs: 30
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 163840 bytes/workgroup (compile time only)
; SGPRBlocks: 12
; VGPRBlocks: 12
; NumSGPRsForWavesPerEU: 102
; NumVGPRsForWavesPerEU: 97
; AccumOffset: 32
; Occupancy: 4
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 1
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 7
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_bd990d5c8a6b0c93,@object ; @__hip_cuid_bd990d5c8a6b0c93
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_bd990d5c8a6b0c93
__hip_cuid_bd990d5c8a6b0c93:
	.byte	0                               ; 0x0
	.size	__hip_cuid_bd990d5c8a6b0c93, 1

	.ident	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.4 26084 f58b06dce1f9c15707c5f808fd002e18c2accf7e)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_bd990d5c8a6b0c93
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
    .group_segment_fixed_size: 163840
    .kernarg_segment_align: 8
    .kernarg_segment_size: 72
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           wvSplitK_vllm_f16_tn_n1_y1u2
    .private_segment_fixed_size: 0
    .sgpr_count:     56
    .sgpr_spill_count: 0
    .symbol:         wvSplitK_vllm_f16_tn_n1_y1u2.kd
    .vgpr_count:     30
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 1
...

	.end_amdgpu_metadata
