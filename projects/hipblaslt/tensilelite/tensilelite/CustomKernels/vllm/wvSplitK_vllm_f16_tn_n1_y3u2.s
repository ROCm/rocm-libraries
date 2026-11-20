	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 4
	.text
	.protected	wvSplitK_vllm_f16_tn_n1_y3u2 ; -- Begin function wvSplitK_vllm_f16_tn_n1_y3u2
	.globl	wvSplitK_vllm_f16_tn_n1_y3u2
	.p2align	8
	.type	wvSplitK_vllm_f16_tn_n1_y3u2,@function
wvSplitK_vllm_f16_tn_n1_y3u2:           ; @wvSplitK_vllm_f16_tn_n1_y3u2
; %bb.98:
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.99:
.LBB0_0:
	s_load_dwordx2 s[20:21], s[0:1], 0x38
	s_mov_b64 s[0:1], -1
	s_cmp_gt_i32 s2, 0x14000
	v_bfe_u32 v46, v0, 10, 10
	s_mul_i32 s33, s8, 3
	s_mul_i32 s17, s8, 48
	s_cbranch_scc0 .LBB0_49
; %bb.1:
	s_abs_i32 s0, s17
	v_cvt_f32_u32_e32 v1, s0
	s_sub_i32 s8, 0, s0
	s_add_i32 s46, s3, -1
	s_add_i32 s1, s17, s46
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
	s_mul_i32 s0, s0, s33
	s_abs_i32 s1, s0
	v_cvt_f32_u32_e32 v1, s1
	s_sub_i32 s9, 0, s1
	s_add_i32 s8, s0, s46
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
	v_cmp_gt_u32_e32 vcc, s8, v46
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_48
; %bb.2:
	s_mul_i32 s9, s8, 3
	v_cvt_f32_u32_e32 v3, s9
	s_mul_i32 s0, s8, s16
	v_add_u32_e32 v1, s0, v46
	v_lshl_add_u32 v2, v1, 1, v1
	v_rcp_iflag_f32_e32 v3, v3
	v_add_u32_e32 v1, 3, v2
	v_cmp_gt_u32_e32 vcc, s3, v2
	v_cmp_lt_u32_e64 s[0:1], s3, v1
	s_add_i32 s47, s3, -3
	v_mov_b32_e32 v4, s47
	s_and_b64 s[0:1], vcc, s[0:1]
	v_cndmask_b32_e64 v36, v2, v4, s[0:1]
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
	s_cselect_b32 s48, s3, s1
	v_cmp_gt_u32_e64 s[0:1], s48, v36
	s_and_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB0_48
; %bb.3:
	s_and_b32 s0, s11, 7
	s_cmp_lg_u32 s0, 0
	s_mov_b32 s25, 0
	s_cselect_b64 s[0:1], -1, 0
	s_and_b32 s24, s6, 15
	s_cmp_lg_u64 s[24:25], 0
	s_cselect_b64 s[18:19], -1, 0
	v_and_b32_e32 v2, 0x3ff, v0
	s_or_b64 s[28:29], s[18:19], s[0:1]
	v_sub_u32_e64 v1, v1, s3 clamp
	v_lshlrev_b32_e32 v47, 3, v2
	v_cmp_eq_u32_e64 s[0:1], 63, v2
	v_lshlrev_b32_e32 v2, 4, v2
	s_mov_b64 s[26:27], 0
	v_cndmask_b32_e32 v1, 0, v1, vcc
	v_lshl_add_u32 v48, v46, 9, v47
	s_lshl_b32 s49, s8, 9
	s_add_i32 s50, s2, -8
	s_ashr_i32 s51, s10, 31
	s_mul_i32 s52, s8, s33
	s_waitcnt lgkmcnt(0)
	v_cmp_eq_f32_e64 s[30:31], s21, 0
	v_cmp_neq_f32_e64 s[34:35], s21, 0
	v_lshl_add_u32 v49, v46, 10, v2
	s_lshl_b32 s53, s8, 10
	v_mov_b32_e32 v35, 0
	s_branch .LBB0_6
.LBB0_4:                                ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[36:37]
	v_add_u32_e32 v3, 3, v2
	v_cmp_gt_u32_e32 vcc, s3, v2
	v_cmp_lt_u32_e64 s[8:9], s3, v3
	v_subrev_u32_e32 v3, s3, v3
	v_mov_b32_e32 v4, s47
	s_and_b64 vcc, vcc, s[8:9]
	v_cndmask_b32_e32 v2, v2, v4, vcc
	v_cndmask_b32_e32 v1, v1, v3, vcc
.LBB0_5:                                ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[18:19]
	v_cmp_le_u32_e32 vcc, s48, v2
	s_or_b64 s[26:27], vcc, s[26:27]
	v_mov_b32_e32 v36, v2
	s_andn2_b64 exec, exec, s[26:27]
	s_cbranch_execz .LBB0_48
.LBB0_6:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_10 Depth 2
                                        ;       Child Loop BB0_15 Depth 3
	v_min_u32_e32 v5, s46, v36
	v_mad_u64_u32 v[2:3], s[36:37], v5, s10, 0
	v_mov_b32_e32 v4, v3
	v_mad_u64_u32 v[4:5], s[36:37], v5, s51, v[4:5]
	v_mov_b32_e32 v3, v4
	v_add_u32_e32 v4, 1, v36
	v_min_u32_e32 v7, s46, v4
	v_mad_u64_u32 v[4:5], s[36:37], v7, s10, 0
	v_mov_b32_e32 v6, v5
	v_add_u32_e32 v38, 2, v36
	v_mad_u64_u32 v[6:7], s[36:37], v7, s51, v[6:7]
	v_min_u32_e32 v9, s46, v38
	v_mov_b32_e32 v5, v6
	v_mad_u64_u32 v[6:7], s[36:37], v9, s10, 0
	v_mov_b32_e32 v8, v7
	v_mad_u64_u32 v[8:9], s[36:37], v9, s51, v[8:9]
	v_mov_b32_e32 v7, v8
	v_cmp_gt_u32_e64 s[8:9], s3, v36
	v_cmp_le_u32_e64 s[18:19], s3, v36
	v_lshlrev_b64 v[40:41], 1, v[2:3]
	v_lshlrev_b64 v[42:43], 1, v[4:5]
	v_lshlrev_b64 v[44:45], 1, v[6:7]
	s_mov_b32 s24, s25
	s_mov_b32 s54, s25
	v_mov_b32_e32 v50, v35
	v_mov_b32_e32 v39, v35
	v_mov_b32_e32 v37, v35
	s_branch .LBB0_10
.LBB0_7:                                ;   in Loop: Header=BB0_10 Depth=2
	s_or_b64 exec, exec, s[40:41]
.LBB0_8:                                ;   in Loop: Header=BB0_10 Depth=2
	s_or_b64 exec, exec, s[38:39]
	s_waitcnt vmcnt(5) lgkmcnt(0)
	;;#ASMSTART
	v_dot2c_f32_f16 v50, v26, v30
	;;#ASMEND
	s_waitcnt vmcnt(4)
	;;#ASMSTART
	v_dot2c_f32_f16 v39, v26, v22
	;;#ASMEND
	s_waitcnt vmcnt(3)
	;;#ASMSTART
	v_dot2c_f32_f16 v37, v26, v18
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v50, v27, v31
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v39, v27, v23
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v37, v27, v19
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v50, v28, v32
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v39, v28, v24
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v37, v28, v20
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v50, v29, v33
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v39, v29, v25
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v37, v29, v21
	;;#ASMEND
	s_waitcnt vmcnt(2)
	;;#ASMSTART
	v_dot2c_f32_f16 v50, v6, v14
	;;#ASMEND
	s_waitcnt vmcnt(1)
	;;#ASMSTART
	v_dot2c_f32_f16 v39, v6, v10
	;;#ASMEND
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	v_dot2c_f32_f16 v37, v6, v2
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v50, v7, v15
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v39, v7, v11
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v37, v7, v3
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v50, v8, v16
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v39, v8, v12
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v37, v8, v4
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v50, v9, v17
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v39, v9, v13
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v37, v9, v5
	;;#ASMEND
.LBB0_9:                                ;   in Loop: Header=BB0_10 Depth=2
	s_or_b64 exec, exec, s[36:37]
	s_addk_i32 s54, 0x400
	s_cmp_ge_u32 s54, s2
	s_cbranch_scc1 .LBB0_24
.LBB0_10:                               ;   Parent Loop BB0_6 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_15 Depth 3
	s_cmp_eq_u32 s54, 0
	s_cselect_b64 s[36:37], -1, 0
	s_add_i32 s38, s24, 0x14000
	s_cmp_eq_u32 s54, s38
	s_cselect_b64 s[40:41], -1, 0
	s_or_b64 s[40:41], s[36:37], s[40:41]
	s_andn2_b64 vcc, exec, s[40:41]
	s_cbranch_vccz .LBB0_12
; %bb.11:                               ;   in Loop: Header=BB0_10 Depth=2
	s_and_saveexec_b64 s[36:37], s[8:9]
	s_cbranch_execz .LBB0_9
	s_branch .LBB0_21
.LBB0_12:                               ;   in Loop: Header=BB0_10 Depth=2
	s_and_b64 s[36:37], s[36:37], exec
	s_cselect_b32 s24, s24, s38
	s_sub_i32 s36, s2, s24
	s_min_u32 s55, s36, 0x14000
	s_lshl_b64 s[36:37], s[24:25], 1
	s_add_u32 s36, s6, s36
	s_addc_u32 s37, s7, s37
	s_mov_b32 s56, 0
	s_mov_b64 s[38:39], 0
	v_mov_b32_e32 v4, v49
	s_barrier
                                        ; implicit-def: $sgpr40_sgpr41
	s_branch .LBB0_15
.LBB0_13:                               ;   in Loop: Header=BB0_15 Depth=3
	s_add_i32 s56, s56, s49
	s_cmp_gt_u32 s56, 0x13fff
	s_cselect_b64 s[44:45], -1, 0
	s_andn2_b64 s[40:41], s[40:41], exec
	s_and_b64 s[44:45], s[44:45], exec
	v_add_u32_e32 v4, s53, v4
	s_or_b64 s[40:41], s[40:41], s[44:45]
.LBB0_14:                               ;   in Loop: Header=BB0_15 Depth=3
	s_or_b64 exec, exec, s[42:43]
	s_and_b64 s[42:43], exec, s[40:41]
	s_or_b64 s[38:39], s[42:43], s[38:39]
	s_andn2_b64 exec, exec, s[38:39]
	s_cbranch_execz .LBB0_20
.LBB0_15:                               ;   Parent Loop BB0_6 Depth=1
                                        ;     Parent Loop BB0_10 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	v_add_u32_e32 v34, s56, v48
	v_cmp_gt_u32_e32 vcc, s55, v34
	s_or_b64 s[40:41], s[40:41], exec
	s_and_saveexec_b64 s[42:43], vcc
	s_cbranch_execz .LBB0_14
; %bb.16:                               ;   in Loop: Header=BB0_15 Depth=3
	v_lshl_add_u64 v[2:3], v[34:35], 1, s[36:37]
	s_mov_b64 s[44:45], -1
	s_and_b64 vcc, exec, s[28:29]
	s_cbranch_vccz .LBB0_18
; %bb.17:                               ;   in Loop: Header=BB0_15 Depth=3
	global_load_dwordx4 v[6:9], v[2:3], off
	s_mov_b64 s[44:45], 0
	s_waitcnt vmcnt(0)
	ds_write_b128 v4, v[6:9]
.LBB0_18:                               ;   in Loop: Header=BB0_15 Depth=3
	s_andn2_b64 vcc, exec, s[44:45]
	s_cbranch_vccnz .LBB0_13
; %bb.19:                               ;   in Loop: Header=BB0_15 Depth=3
	v_readfirstlane_b32 s44, v4
	s_mov_b32 m0, s44
	s_nop 0
	global_load_lds_dwordx4 v[2:3], off
	s_branch .LBB0_13
.LBB0_20:                               ;   in Loop: Header=BB0_10 Depth=2
	s_or_b64 exec, exec, s[38:39]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	s_and_saveexec_b64 s[36:37], s[8:9]
	s_cbranch_execz .LBB0_9
.LBB0_21:                               ;   in Loop: Header=BB0_10 Depth=2
	v_add_u32_e32 v51, s54, v47
	v_min_u32_e32 v34, s50, v51
	v_lshl_add_u64 v[2:3], v[34:35], 1, s[4:5]
	v_add_u32_e32 v52, 0x200, v51
	v_lshl_add_u64 v[4:5], v[2:3], 0, v[40:41]
	v_min_u32_e32 v34, s50, v52
	v_lshl_add_u64 v[6:7], v[2:3], 0, v[42:43]
	global_load_dwordx4 v[30:33], v[4:5], off nt
	global_load_dwordx4 v[22:25], v[6:7], off nt
	v_lshl_add_u64 v[4:5], v[34:35], 1, s[4:5]
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[44:45]
	v_lshl_add_u64 v[6:7], v[4:5], 0, v[40:41]
	global_load_dwordx4 v[18:21], v[2:3], off nt
	global_load_dwordx4 v[14:17], v[6:7], off nt
	v_lshl_add_u64 v[6:7], v[4:5], 0, v[42:43]
	v_lshl_add_u64 v[8:9], v[4:5], 0, v[44:45]
	global_load_dwordx4 v[10:13], v[6:7], off nt
	global_load_dwordx4 v[2:5], v[8:9], off nt
	v_cmp_gt_u32_e32 vcc, s2, v51
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v8, 0
	v_mov_b32_e32 v9, 0
	v_mov_b32_e32 v6, 0
	v_mov_b32_e32 v7, 0
	s_and_saveexec_b64 s[38:39], vcc
	s_cbranch_execz .LBB0_8
; %bb.22:                               ;   in Loop: Header=BB0_10 Depth=2
	v_subrev_u32_e32 v6, s24, v51
	v_lshlrev_b32_e32 v34, 1, v6
	ds_read_b128 v[26:29], v34
	v_cmp_gt_u32_e32 vcc, s2, v52
	v_mov_b32_e32 v7, 0
	v_mov_b32_e32 v6, 0
	v_mov_b32_e32 v9, 0
	v_mov_b32_e32 v8, 0
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_7
; %bb.23:                               ;   in Loop: Header=BB0_10 Depth=2
	ds_read_b128 v[6:9], v34 offset:1024
	s_branch .LBB0_7
.LBB0_24:                               ;   in Loop: Header=BB0_6 Depth=1
	v_add_u32_e32 v2, s52, v36
	s_and_saveexec_b64 s[8:9], s[18:19]
	s_xor_b64 s[8:9], exec, s[8:9]
; %bb.25:                               ;   in Loop: Header=BB0_6 Depth=1
                                        ; implicit-def: $vgpr36
                                        ; implicit-def: $vgpr38
; %bb.26:                               ;   in Loop: Header=BB0_6 Depth=1
	s_andn2_saveexec_b64 s[18:19], s[8:9]
	s_cbranch_execz .LBB0_5
; %bb.27:                               ;   in Loop: Header=BB0_6 Depth=1
	v_add_f32_dpp v3, v50, v50 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v39, v39 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v37, v37 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v3, v3 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v5, v5 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v6, v6 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v3, v3 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v5, v5 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v6, v6 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v3, v3 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v5, v5 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v6, v6 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v3, v3 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v5, v5 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v6, v6 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v4, v3 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v7, v5 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v9, v8 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_and_saveexec_b64 s[36:37], s[0:1]
	s_cbranch_execz .LBB0_4
; %bb.28:                               ;   in Loop: Header=BB0_6 Depth=1
	v_cmp_eq_u32_e32 vcc, 0, v1
	s_and_b64 s[8:9], s[30:31], vcc
	v_add_f32_e32 v6, v3, v4
	v_add_f32_e32 v4, v5, v7
	v_add_f32_e32 v3, v8, v9
	s_xor_b64 s[38:39], s[8:9], -1
	v_mov_b32_e32 v39, v35
	s_mov_b64 s[8:9], 0
                                        ; implicit-def: $vgpr5
	s_and_saveexec_b64 s[40:41], s[38:39]
	s_xor_b64 s[38:39], exec, s[40:41]
	s_cbranch_execnz .LBB0_31
; %bb.29:                               ;   in Loop: Header=BB0_6 Depth=1
	s_andn2_saveexec_b64 s[38:39], s[38:39]
	s_cbranch_execnz .LBB0_46
.LBB0_30:                               ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[38:39]
	s_and_b64 exec, exec, s[8:9]
	s_cbranch_execz .LBB0_4
	s_branch .LBB0_47
.LBB0_31:                               ;   in Loop: Header=BB0_6 Depth=1
	v_cndmask_b32_e64 v5, 0, 1, s[34:35]
	v_cmp_ne_u32_e64 s[8:9], 1, v5
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_35
; %bb.32:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v5, s20, v6
	s_and_b64 vcc, exec, s[8:9]
	v_mov_b32_e32 v37, v35
	s_cbranch_vccnz .LBB0_34
; %bb.33:                               ;   in Loop: Header=BB0_6 Depth=1
	v_lshl_add_u64 v[6:7], v[36:37], 1, s[12:13]
	global_load_ushort v6, v[6:7], off
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v5, s21, v6, v5 op_sel_hi:[0,1,0]
.LBB0_34:                               ;   in Loop: Header=BB0_6 Depth=1
	v_cvt_f16_f32_e32 v5, v5
	v_lshl_add_u64 v[6:7], v[36:37], 1, s[14:15]
	global_store_short v[6:7], v5, off
.LBB0_35:                               ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[40:41]
	v_cmp_gt_u32_e32 vcc, 2, v1
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_41
; %bb.36:                               ;   in Loop: Header=BB0_6 Depth=1
	s_andn2_b64 vcc, exec, s[30:31]
	v_mov_b32_e32 v37, v35
	s_cbranch_vccnz .LBB0_38
; %bb.37:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v4, s20, v4
	s_cbranch_execz .LBB0_39
	s_branch .LBB0_40
.LBB0_38:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v4, s20, v4
.LBB0_39:                               ;   in Loop: Header=BB0_6 Depth=1
	v_lshl_add_u64 v[6:7], v[36:37], 1, s[12:13]
	global_load_ushort v5, v[6:7], off offset:2
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v4, s21, v5, v4 op_sel_hi:[0,1,0]
.LBB0_40:                               ;   in Loop: Header=BB0_6 Depth=1
	v_cvt_f16_f32_e32 v6, v4
	v_lshl_add_u64 v[4:5], v[36:37], 1, s[14:15]
	global_store_short v[4:5], v6, off offset:2
.LBB0_41:                               ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[40:41]
	v_cmp_gt_u32_e32 vcc, 3, v1
	s_mov_b64 s[42:43], 0
                                        ; implicit-def: $vgpr5
	s_and_saveexec_b64 s[40:41], vcc
	s_xor_b64 s[40:41], exec, s[40:41]
	s_cbranch_execz .LBB0_45
; %bb.42:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v5, s20, v3
	s_and_b64 vcc, exec, s[8:9]
	v_mov_b32_e32 v39, v35
	s_cbranch_vccnz .LBB0_44
; %bb.43:                               ;   in Loop: Header=BB0_6 Depth=1
	v_lshl_add_u64 v[6:7], v[38:39], 1, s[12:13]
	global_load_ushort v3, v[6:7], off
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v5, s21, v3, v5 op_sel_hi:[0,1,0]
.LBB0_44:                               ;   in Loop: Header=BB0_6 Depth=1
	s_mov_b64 s[42:43], exec
.LBB0_45:                               ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[40:41]
	s_and_b64 s[8:9], s[42:43], exec
                                        ; implicit-def: $vgpr6
                                        ; implicit-def: $vgpr36
                                        ; implicit-def: $vgpr4
                                        ; implicit-def: $vgpr3
	s_andn2_saveexec_b64 s[38:39], s[38:39]
	s_cbranch_execz .LBB0_30
.LBB0_46:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v5, s20, v6
	v_mov_b32_e32 v37, v35
	v_mul_f32_e32 v4, s20, v4
	v_lshl_add_u64 v[6:7], v[36:37], 1, s[14:15]
	v_cvt_pk_f16_f32 v4, v5, v4
	global_store_dword v[6:7], v4, off
	v_mul_f32_e32 v5, s20, v3
	s_or_b64 s[8:9], s[8:9], exec
	s_or_b64 exec, exec, s[38:39]
	s_and_b64 exec, exec, s[8:9]
	s_cbranch_execz .LBB0_4
.LBB0_47:                               ;   in Loop: Header=BB0_6 Depth=1
	v_cvt_f16_f32_e32 v3, v5
	v_lshl_add_u64 v[4:5], v[38:39], 1, s[14:15]
	global_store_short v[4:5], v3, off
	s_branch .LBB0_4
.LBB0_48:
	s_or_b64 exec, exec, s[22:23]
	s_mov_b64 s[0:1], 0
.LBB0_49:
	s_andn2_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB0_97
; %bb.50:
	v_and_b32_e32 v2, 0x3ff, v0
	v_lshlrev_b32_e32 v42, 3, v2
	s_cmp_eq_u32 s11, s2
	s_mov_b64 s[8:9], 0
	s_cbranch_scc1 .LBB0_52
; %bb.51:
	s_and_b32 s18, s6, 15
	s_mov_b32 s19, 0
	s_mov_b64 s[0:1], -1
	s_min_u32 s26, s2, 0x14000
	v_lshl_add_u32 v3, v46, 9, v42
	s_cbranch_execz .LBB0_53
	s_branch .LBB0_54
.LBB0_52:
                                        ; implicit-def: $sgpr18_sgpr19
	s_mov_b64 s[0:1], 0
	s_min_u32 s26, s2, 0x14000
	v_lshl_add_u32 v3, v46, 9, v42
.LBB0_53:
	s_and_b32 s0, s2, 7
	s_cmp_lg_u32 s0, 0
	s_mov_b32 s19, 0
	s_cselect_b64 s[0:1], -1, 0
	s_and_b32 s18, s6, 15
	s_cmp_lg_u64 s[18:19], 0
	s_cselect_b64 s[8:9], -1, 0
	s_or_b64 s[0:1], s[0:1], s[8:9]
	s_mov_b64 s[8:9], -1
.LBB0_54:
	s_andn2_b64 vcc, exec, s[0:1]
	v_cmp_gt_u32_e64 s[0:1], s26, v3
	s_cbranch_vccz .LBB0_56
; %bb.55:
	s_andn2_b64 vcc, exec, s[8:9]
	s_cbranch_vccz .LBB0_64
	s_branch .LBB0_68
.LBB0_56:
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_63
; %bb.57:
	s_and_b32 s0, s11, 7
	s_cmp_lg_u32 s0, 0
	s_cselect_b64 s[22:23], -1, 0
	s_cmp_lg_u64 s[18:19], 0
	v_lshlrev_b32_e32 v0, 4, v2
	s_cselect_b64 s[18:19], -1, 0
	v_lshl_add_u32 v4, v46, 10, v0
	v_lshlrev_b32_e32 v0, 1, v3
	v_mov_b32_e32 v1, 0
	s_mov_b64 s[0:1], 0
	s_or_b64 s[18:19], s[22:23], s[18:19]
	v_lshl_add_u64 v[0:1], s[6:7], 0, v[0:1]
	s_mov_b64 s[22:23], 0x4000
	v_mov_b32_e32 v5, v3
	s_branch .LBB0_59
.LBB0_58:                               ;   in Loop: Header=BB0_59 Depth=1
	v_add_u32_e32 v5, 0x2000, v5
	v_cmp_le_u32_e32 vcc, s26, v5
	v_add_u32_e32 v4, 0x4000, v4
	s_or_b64 s[0:1], vcc, s[0:1]
	v_lshl_add_u64 v[0:1], v[0:1], 0, s[22:23]
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB0_63
.LBB0_59:                               ; =>This Inner Loop Header: Depth=1
	s_mov_b64 s[24:25], -1
	s_and_b64 vcc, exec, s[18:19]
	s_cbranch_vccz .LBB0_61
; %bb.60:                               ;   in Loop: Header=BB0_59 Depth=1
	global_load_dwordx4 v[6:9], v[0:1], off
	s_mov_b64 s[24:25], 0
	s_waitcnt vmcnt(0)
	ds_write_b128 v4, v[6:9]
.LBB0_61:                               ;   in Loop: Header=BB0_59 Depth=1
	s_andn2_b64 vcc, exec, s[24:25]
	s_cbranch_vccnz .LBB0_58
; %bb.62:                               ;   in Loop: Header=BB0_59 Depth=1
	v_readfirstlane_b32 s11, v4
	s_mov_b32 m0, s11
	s_nop 0
	global_load_lds_dwordx4 v[0:1], off
	s_branch .LBB0_58
.LBB0_63:
	s_or_b64 exec, exec, s[8:9]
	s_cbranch_execnz .LBB0_68
.LBB0_64:
	v_cmp_gt_u32_e32 vcc, s26, v3
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_67
; %bb.65:
	v_lshlrev_b32_e32 v0, 4, v2
	v_lshl_add_u32 v4, v46, 10, v0
	v_lshlrev_b32_e32 v0, 1, v3
	v_mov_b32_e32 v1, 0
	v_lshl_add_u64 v[0:1], s[6:7], 0, v[0:1]
	s_mov_b64 s[6:7], 0
	s_mov_b64 s[8:9], 0x4000
.LBB0_66:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s11, v4
	s_mov_b32 m0, s11
	v_add_u32_e32 v3, 0x2000, v3
	global_load_lds_dwordx4 v[0:1], off
	v_cmp_le_u32_e32 vcc, s26, v3
	v_add_u32_e32 v4, 0x4000, v4
	s_or_b64 s[6:7], vcc, s[6:7]
	v_lshl_add_u64 v[0:1], v[0:1], 0, s[8:9]
	s_andn2_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB0_66
.LBB0_67:
	s_or_b64 exec, exec, s[0:1]
.LBB0_68:
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
	s_add_i32 s11, s6, 1
	s_sub_i32 s9, s7, s0
	s_cmp_ge_u32 s7, s0
	s_cselect_b32 s6, s11, s6
	s_cselect_b32 s7, s9, s7
	s_add_i32 s9, s6, 1
	s_cmp_ge_u32 s7, s0
	s_cselect_b32 s0, s9, s6
	s_xor_b32 s0, s0, s8
	s_sub_i32 s0, s0, s8
	s_mul_i32 s0, s0, s33
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
	v_add_u32_e32 v0, s0, v46
	v_lshl_add_u32 v1, v0, 1, v0
	v_add_u32_e32 v0, 3, v1
	v_cmp_gt_u32_e32 vcc, s3, v1
	v_cmp_lt_u32_e64 s[0:1], s3, v0
	s_add_i32 s11, s3, -3
	v_mov_b32_e32 v3, s11
	s_and_b64 s[0:1], vcc, s[0:1]
	v_cndmask_b32_e64 v32, v1, v3, s[0:1]
	v_cmp_gt_u32_e64 s[0:1], s8, v46
	v_cmp_gt_u32_e64 s[6:7], s3, v32
	s_and_b64 s[0:1], s[0:1], s[6:7]
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_97
; %bb.69:
	s_cmp_lg_u32 s2, 0
	v_sub_u32_e64 v0, v0, s3 clamp
	s_cselect_b64 s[6:7], -1, 0
	v_cndmask_b32_e32 v43, 0, v0, vcc
	v_cndmask_b32_e64 v0, 0, 1, s[6:7]
	s_add_i32 s34, s2, -8
	s_add_i32 s35, s3, -1
	s_ashr_i32 s36, s10, 31
	v_cmp_eq_u32_e64 s[0:1], 63, v2
	v_cmp_eq_f32_e64 s[16:17], s21, 0
	v_cmp_neq_f32_e64 s[18:19], s21, 0
	s_mul_i32 s33, s8, s33
	v_lshlrev_b32_e32 v44, 4, v2
	s_mov_b64 s[22:23], 0
	v_cmp_ne_u32_e64 s[8:9], 1, v0
	v_mov_b32_e32 v35, 0
	s_branch .LBB0_71
.LBB0_70:                               ;   in Loop: Header=BB0_71 Depth=1
	s_or_b64 exec, exec, s[24:25]
	v_add_u32_e32 v0, s33, v32
	v_add_u32_e32 v1, 3, v0
	v_cmp_gt_u32_e32 vcc, s3, v0
	v_cmp_lt_u32_e64 s[6:7], s3, v1
	v_mov_b32_e32 v2, s11
	s_and_b64 vcc, vcc, s[6:7]
	v_cndmask_b32_e32 v32, v0, v2, vcc
	v_subrev_u32_e32 v1, s3, v1
	v_cmp_le_u32_e64 s[6:7], s3, v32
	s_or_b64 s[22:23], s[6:7], s[22:23]
	v_cndmask_b32_e32 v43, v43, v1, vcc
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execz .LBB0_97
.LBB0_71:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_75 Depth 2
	s_and_b64 vcc, exec, s[8:9]
	v_mov_b32_e32 v33, v35
	v_mov_b32_e32 v46, v35
	v_mov_b32_e32 v45, v35
	s_cbranch_vccnz .LBB0_78
; %bb.72:                               ;   in Loop: Header=BB0_71 Depth=1
	v_min_u32_e32 v3, s35, v32
	v_mad_u64_u32 v[0:1], s[6:7], v3, s10, 0
	v_mov_b32_e32 v2, v1
	v_mad_u64_u32 v[2:3], s[6:7], v3, s36, v[2:3]
	v_mov_b32_e32 v1, v2
	v_add_u32_e32 v2, 1, v32
	v_min_u32_e32 v5, s35, v2
	v_mad_u64_u32 v[2:3], s[6:7], v5, s10, 0
	v_mov_b32_e32 v4, v3
	v_mad_u64_u32 v[4:5], s[6:7], v5, s36, v[4:5]
	v_mov_b32_e32 v3, v4
	v_add_u32_e32 v4, 2, v32
	v_min_u32_e32 v7, s35, v4
	v_mad_u64_u32 v[4:5], s[6:7], v7, s10, 0
	v_mov_b32_e32 v6, v5
	v_mad_u64_u32 v[6:7], s[6:7], v7, s36, v[6:7]
	v_mov_b32_e32 v5, v6
	v_mov_b32_e32 v45, 0
	s_mov_b32 s26, 0
	v_lshlrev_b64 v[36:37], 1, v[0:1]
	v_lshlrev_b64 v[38:39], 1, v[2:3]
	v_lshlrev_b64 v[40:41], 1, v[4:5]
	v_mov_b32_e32 v47, v44
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v33, 0
	s_branch .LBB0_75
.LBB0_73:                               ;   in Loop: Header=BB0_75 Depth=2
	s_or_b64 exec, exec, s[24:25]
.LBB0_74:                               ;   in Loop: Header=BB0_75 Depth=2
	s_or_b64 exec, exec, s[6:7]
	s_waitcnt vmcnt(5) lgkmcnt(0)
	;;#ASMSTART
	v_dot2c_f32_f16 v45, v28, v24
	;;#ASMEND
	s_waitcnt vmcnt(4)
	;;#ASMSTART
	v_dot2c_f32_f16 v46, v28, v20
	;;#ASMEND
	s_waitcnt vmcnt(3)
	;;#ASMSTART
	v_dot2c_f32_f16 v33, v28, v16
	;;#ASMEND
	s_addk_i32 s26, 0x400
	;;#ASMSTART
	v_dot2c_f32_f16 v45, v29, v25
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v46, v29, v21
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v33, v29, v17
	;;#ASMEND
	s_cmp_ge_u32 s26, s2
	;;#ASMSTART
	v_dot2c_f32_f16 v45, v30, v26
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v46, v30, v22
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v33, v30, v18
	;;#ASMEND
	v_add_u32_e32 v47, 0x800, v47
	;;#ASMSTART
	v_dot2c_f32_f16 v45, v31, v27
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v46, v31, v23
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v33, v31, v19
	;;#ASMEND
	s_waitcnt vmcnt(2)
	;;#ASMSTART
	v_dot2c_f32_f16 v45, v8, v12
	;;#ASMEND
	s_waitcnt vmcnt(1)
	;;#ASMSTART
	v_dot2c_f32_f16 v46, v8, v4
	;;#ASMEND
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	v_dot2c_f32_f16 v33, v8, v0
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v45, v9, v13
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v46, v9, v5
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v33, v9, v1
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v45, v10, v14
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v46, v10, v6
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v33, v10, v2
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v45, v11, v15
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v46, v11, v7
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v33, v11, v3
	;;#ASMEND
	s_cbranch_scc1 .LBB0_78
.LBB0_75:                               ;   Parent Loop BB0_71 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_add_u32_e32 v28, s26, v42
	v_min_u32_e32 v34, s34, v28
	v_lshl_add_u64 v[0:1], v[34:35], 1, s[4:5]
	v_add_u32_e32 v48, 0x200, v28
	v_lshl_add_u64 v[2:3], v[0:1], 0, v[36:37]
	v_min_u32_e32 v34, s34, v48
	v_lshl_add_u64 v[4:5], v[0:1], 0, v[38:39]
	global_load_dwordx4 v[24:27], v[2:3], off nt
	global_load_dwordx4 v[20:23], v[4:5], off nt
	v_lshl_add_u64 v[2:3], v[34:35], 1, s[4:5]
	v_lshl_add_u64 v[0:1], v[0:1], 0, v[40:41]
	v_lshl_add_u64 v[4:5], v[2:3], 0, v[36:37]
	v_lshl_add_u64 v[8:9], v[2:3], 0, v[38:39]
	global_load_dwordx4 v[16:19], v[0:1], off nt
	global_load_dwordx4 v[12:15], v[4:5], off nt
	v_lshl_add_u64 v[10:11], v[2:3], 0, v[40:41]
	global_load_dwordx4 v[4:7], v[8:9], off nt
	global_load_dwordx4 v[0:3], v[10:11], off nt
	v_cmp_gt_u32_e32 vcc, s2, v28
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v11, 0
	v_mov_b32_e32 v8, 0
	v_mov_b32_e32 v9, 0
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_74
; %bb.76:                               ;   in Loop: Header=BB0_75 Depth=2
	ds_read_b128 v[28:31], v47
	v_cmp_gt_u32_e32 vcc, s2, v48
	v_mov_b32_e32 v9, 0
	v_mov_b32_e32 v8, 0
	v_mov_b32_e32 v11, 0
	v_mov_b32_e32 v10, 0
	s_and_saveexec_b64 s[24:25], vcc
	s_cbranch_execz .LBB0_73
; %bb.77:                               ;   in Loop: Header=BB0_75 Depth=2
	ds_read_b128 v[8:11], v47 offset:1024
	s_branch .LBB0_73
.LBB0_78:                               ;   in Loop: Header=BB0_71 Depth=1
	; sched_barrier mask(0x00000000)
	s_nop 0
	v_add_f32_dpp v0, v45, v45 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v46, v46 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v33, v33 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v0, v0 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v3, v3 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v0, v0 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v3, v3 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v0, v0 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v3, v3 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v0, v0 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v3, v3 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v1, v0 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v4, v2 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v6, v5 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_and_saveexec_b64 s[24:25], s[0:1]
	s_cbranch_execz .LBB0_70
; %bb.79:                               ;   in Loop: Header=BB0_71 Depth=1
	v_cmp_eq_u32_e32 vcc, 0, v43
	s_and_b64 s[6:7], s[16:17], vcc
	v_add_f32_e32 v3, v0, v1
	v_add_f32_e32 v2, v2, v4
	v_add_f32_e32 v0, v5, v6
	s_xor_b64 s[26:27], s[6:7], -1
	s_mov_b64 s[6:7], 0
                                        ; implicit-def: $vgpr1
	s_and_saveexec_b64 s[28:29], s[26:27]
	s_xor_b64 s[26:27], exec, s[28:29]
	s_cbranch_execnz .LBB0_82
; %bb.80:                               ;   in Loop: Header=BB0_71 Depth=1
	s_andn2_saveexec_b64 s[26:27], s[26:27]
	s_cbranch_execnz .LBB0_95
.LBB0_81:                               ;   in Loop: Header=BB0_71 Depth=1
	s_or_b64 exec, exec, s[26:27]
	s_and_b64 exec, exec, s[6:7]
	s_cbranch_execz .LBB0_70
	s_branch .LBB0_96
.LBB0_82:                               ;   in Loop: Header=BB0_71 Depth=1
	v_cndmask_b32_e64 v1, 0, 1, s[18:19]
	v_cmp_ne_u32_e64 s[6:7], 1, v1
	s_and_saveexec_b64 s[28:29], vcc
	s_cbranch_execz .LBB0_86
; %bb.83:                               ;   in Loop: Header=BB0_71 Depth=1
	v_mul_f32_e32 v1, s20, v3
	s_and_b64 vcc, exec, s[6:7]
	v_mov_b32_e32 v33, v35
	s_cbranch_vccnz .LBB0_85
; %bb.84:                               ;   in Loop: Header=BB0_71 Depth=1
	v_lshl_add_u64 v[4:5], v[32:33], 1, s[12:13]
	global_load_ushort v3, v[4:5], off
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v1, s21, v3, v1 op_sel_hi:[0,1,0]
.LBB0_85:                               ;   in Loop: Header=BB0_71 Depth=1
	v_cvt_f16_f32_e32 v1, v1
	v_lshl_add_u64 v[4:5], v[32:33], 1, s[14:15]
	global_store_short v[4:5], v1, off
.LBB0_86:                               ;   in Loop: Header=BB0_71 Depth=1
	s_or_b64 exec, exec, s[28:29]
	v_cmp_gt_u32_e32 vcc, 2, v43
	s_and_saveexec_b64 s[28:29], vcc
	s_cbranch_execz .LBB0_90
; %bb.87:                               ;   in Loop: Header=BB0_71 Depth=1
	v_mul_f32_e32 v1, s20, v2
	s_and_b64 vcc, exec, s[6:7]
	v_add_u32_e32 v34, 1, v32
	s_cbranch_vccnz .LBB0_89
; %bb.88:                               ;   in Loop: Header=BB0_71 Depth=1
	v_lshl_add_u64 v[2:3], v[34:35], 1, s[12:13]
	global_load_ushort v2, v[2:3], off
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v1, s21, v2, v1 op_sel_hi:[0,1,0]
.LBB0_89:                               ;   in Loop: Header=BB0_71 Depth=1
	v_cvt_f16_f32_e32 v1, v1
	v_lshl_add_u64 v[2:3], v[34:35], 1, s[14:15]
	global_store_short v[2:3], v1, off
.LBB0_90:                               ;   in Loop: Header=BB0_71 Depth=1
	s_or_b64 exec, exec, s[28:29]
	v_cmp_gt_u32_e32 vcc, 3, v43
	s_mov_b64 s[30:31], 0
                                        ; implicit-def: $vgpr1
	s_and_saveexec_b64 s[28:29], vcc
	s_xor_b64 s[28:29], exec, s[28:29]
	s_cbranch_execz .LBB0_94
; %bb.91:                               ;   in Loop: Header=BB0_71 Depth=1
	v_mul_f32_e32 v1, s20, v0
	s_and_b64 vcc, exec, s[6:7]
	v_add_u32_e32 v34, 2, v32
	s_cbranch_vccnz .LBB0_93
; %bb.92:                               ;   in Loop: Header=BB0_71 Depth=1
	v_lshl_add_u64 v[2:3], v[34:35], 1, s[12:13]
	global_load_ushort v0, v[2:3], off
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v1, s21, v0, v1 op_sel_hi:[0,1,0]
.LBB0_93:                               ;   in Loop: Header=BB0_71 Depth=1
	s_mov_b64 s[30:31], exec
.LBB0_94:                               ;   in Loop: Header=BB0_71 Depth=1
	s_or_b64 exec, exec, s[28:29]
	s_and_b64 s[6:7], s[30:31], exec
                                        ; implicit-def: $vgpr3
                                        ; implicit-def: $vgpr2
                                        ; implicit-def: $vgpr0
	s_andn2_saveexec_b64 s[26:27], s[26:27]
	s_cbranch_execz .LBB0_81
.LBB0_95:                               ;   in Loop: Header=BB0_71 Depth=1
	v_mov_b32_e32 v33, v35
	v_fma_mixlo_f16 v1, s20, v3, 0
	v_lshl_add_u64 v[4:5], v[32:33], 1, s[14:15]
	v_add_u32_e32 v34, 1, v32
	global_store_short v[4:5], v1, off
	v_fma_mixlo_f16 v1, s20, v2, 0
	v_lshl_add_u64 v[2:3], v[34:35], 1, s[14:15]
	global_store_short v[2:3], v1, off
	v_mul_f32_e32 v1, s20, v0
	v_add_u32_e32 v34, 2, v32
	s_or_b64 s[6:7], s[6:7], exec
	s_or_b64 exec, exec, s[26:27]
	s_and_b64 exec, exec, s[6:7]
	s_cbranch_execz .LBB0_70
.LBB0_96:                               ;   in Loop: Header=BB0_71 Depth=1
	v_cvt_f16_f32_e32 v2, v1
	v_lshl_add_u64 v[0:1], v[34:35], 1, s[14:15]
	global_store_short v[0:1], v2, off
	s_branch .LBB0_70
.LBB0_97:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wvSplitK_vllm_f16_tn_n1_y3u2
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
		.amdhsa_accum_offset 56
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
	.size	wvSplitK_vllm_f16_tn_n1_y3u2, .Lfunc_end0-wvSplitK_vllm_f16_tn_n1_y3u2
                                        ; -- End function
	.set wvSplitK_vllm_f16_tn_n1_y3u2.num_vgpr, 53
	.set wvSplitK_vllm_f16_tn_n1_y3u2.num_agpr, 0
	.set wvSplitK_vllm_f16_tn_n1_y3u2.numbered_sgpr, 57
	.set wvSplitK_vllm_f16_tn_n1_y3u2.num_named_barrier, 0
	.set wvSplitK_vllm_f16_tn_n1_y3u2.private_seg_size, 0
	.set wvSplitK_vllm_f16_tn_n1_y3u2.uses_vcc, 1
	.set wvSplitK_vllm_f16_tn_n1_y3u2.uses_flat_scratch, 0
	.set wvSplitK_vllm_f16_tn_n1_y3u2.has_dyn_sized_stack, 0
	.set wvSplitK_vllm_f16_tn_n1_y3u2.has_recursion, 0
	.set wvSplitK_vllm_f16_tn_n1_y3u2.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 5004
; TotalNumSgprs: 63
; NumVgprs: 53
; NumAgprs: 0
; TotalNumVgprs: 53
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 163840 bytes/workgroup (compile time only)
; SGPRBlocks: 12
; VGPRBlocks: 12
; NumSGPRsForWavesPerEU: 102
; NumVGPRsForWavesPerEU: 97
; AccumOffset: 56
; Occupancy: 4
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 1
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 13
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_df40b9efada65cc0,@object ; @__hip_cuid_df40b9efada65cc0
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_df40b9efada65cc0
__hip_cuid_df40b9efada65cc0:
	.byte	0                               ; 0x0
	.size	__hip_cuid_df40b9efada65cc0, 1

	.ident	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.4 26084 f58b06dce1f9c15707c5f808fd002e18c2accf7e)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_df40b9efada65cc0
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
    .name:           wvSplitK_vllm_f16_tn_n1_y3u2
    .private_segment_fixed_size: 0
    .sgpr_count:     63
    .sgpr_spill_count: 0
    .symbol:         wvSplitK_vllm_f16_tn_n1_y3u2.kd
    .vgpr_count:     53
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 1
...

	.end_amdgpu_metadata
