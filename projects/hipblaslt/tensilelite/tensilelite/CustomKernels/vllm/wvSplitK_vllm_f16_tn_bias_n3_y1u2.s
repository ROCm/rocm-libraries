	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 4
	.text
	.protected	wvSplitK_vllm_f16_tn_bias_n3_y1u2 ; -- Begin function wvSplitK_vllm_f16_tn_bias_n3_y1u2
	.globl	wvSplitK_vllm_f16_tn_bias_n3_y1u2
	.p2align	8
	.type	wvSplitK_vllm_f16_tn_bias_n3_y1u2,@function
wvSplitK_vllm_f16_tn_bias_n3_y1u2:      ; @wvSplitK_vllm_f16_tn_bias_n3_y1u2
; %bb.78:
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.79:
.LBB0_0:
	s_load_dwordx4 s[24:27], s[0:1], 0x38
	s_load_dwordx2 s[28:29], s[0:1], 0x48
	s_mov_b32 s30, s11
	s_mov_b64 s[0:1], -1
	s_cmpk_gt_i32 s2, 0x6aaa
	v_bfe_u32 v40, v0, 10, 10
	s_cbranch_scc0 .LBB0_40
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
	v_cmp_gt_u32_e32 vcc, s20, v40
	s_and_saveexec_b64 s[34:35], vcc
	s_cbranch_execz .LBB0_39
; %bb.2:
	v_cvt_f32_u32_e32 v1, s20
	s_mul_i32 s0, s20, s16
	s_sub_i32 s11, 0, s20
	v_add_u32_e32 v34, s0, v40
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
	v_cmp_gt_u32_e32 vcc, s9, v34
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_39
; %bb.3:
	s_and_b32 s0, s30, 7
	s_cmp_lg_u32 s0, 0
	s_mov_b32 s37, 0
	s_cselect_b64 s[0:1], -1, 0
	s_and_b32 s36, s6, 15
	s_cmp_lg_u64 s[36:37], 0
	s_cselect_b64 s[18:19], -1, 0
	s_or_b64 s[40:41], s[18:19], s[0:1]
	v_and_b32_e32 v2, 0x3ff, v0
	s_waitcnt lgkmcnt(0)
	s_cmp_eq_u64 s[28:29], 0
	v_lshlrev_b32_e32 v1, 3, v2
	v_cmp_eq_u32_e64 s[18:19], 63, v2
	v_lshlrev_b32_e32 v2, 4, v2
	s_cselect_b64 s[0:1], -1, 0
	s_ashr_i32 s31, s30, 31
	s_ashr_i32 s45, s26, 31
	s_mov_b32 s44, s26
	s_ashr_i32 s47, s27, 31
	s_mov_b32 s46, s27
	v_lshl_add_u32 v42, v40, 10, v2
	v_cndmask_b32_e64 v2, 0, 1, s[40:41]
	s_mov_b64 s[38:39], 0
	v_lshl_add_u32 v41, v40, 9, v1
	s_lshl_b32 s11, s20, 9
	s_add_i32 s17, s2, -8
	s_ashr_i32 s33, s10, 31
	v_cmp_neq_f32_e64 s[42:43], s25, 0
	s_mul_i32 s66, s20, s8
	s_lshl_b64 s[48:49], s[30:31], 1
	s_lshl_b64 s[50:51], s[44:45], 1
	s_lshl_b64 s[52:53], s[46:47], 1
	s_lshl_b32 s67, s20, 10
	s_xor_b64 s[54:55], s[0:1], -1
	v_mov_b32_e32 v37, 0
	v_cmp_ne_u32_e64 s[20:21], 1, v2
                                        ; implicit-def: $vgpr43
	s_branch .LBB0_5
.LBB0_4:                                ;   in Loop: Header=BB0_5 Depth=1
	s_or_b64 exec, exec, s[56:57]
	v_add_u32_e32 v34, s66, v34
	v_cmp_le_u32_e32 vcc, s9, v34
	s_or_b64 s[38:39], vcc, s[38:39]
	s_andn2_b64 exec, exec, s[38:39]
	s_cbranch_execz .LBB0_39
.LBB0_5:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_11 Depth 2
                                        ;       Child Loop BB0_16 Depth 3
	v_cmp_gt_u32_e64 s[22:23], s3, v34
	s_and_b64 s[58:59], s[54:55], s[22:23]
	v_mov_b32_e32 v35, v37
	s_and_saveexec_b64 s[56:57], s[58:59]
	s_cbranch_execz .LBB0_7
; %bb.6:                                ;   in Loop: Header=BB0_5 Depth=1
	v_lshl_add_u64 v[2:3], v[34:35], 1, s[28:29]
	global_load_ushort v43, v[2:3], off
.LBB0_7:                                ;   in Loop: Header=BB0_5 Depth=1
	s_or_b64 exec, exec, s[56:57]
	v_mad_u64_u32 v[2:3], s[56:57], v34, s10, 0
	v_mov_b32_e32 v4, v3
	v_mad_u64_u32 v[4:5], s[56:57], v34, s33, v[4:5]
	v_mov_b32_e32 v3, v4
	s_mov_b32 s36, 0
	v_lshl_add_u64 v[38:39], v[2:3], 1, s[4:5]
	v_mov_b32_e32 v46, 0
	s_mov_b32 s68, s36
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v44, 0
	s_branch .LBB0_11
.LBB0_8:                                ;   in Loop: Header=BB0_11 Depth=2
	s_or_b64 exec, exec, s[60:61]
.LBB0_9:                                ;   in Loop: Header=BB0_11 Depth=2
	s_or_b64 exec, exec, s[58:59]
	s_waitcnt vmcnt(1) lgkmcnt(2)
	;;#ASMSTART
	v_dot2c_f32_f16 v46, v30, v14
	;;#ASMEND
	s_waitcnt lgkmcnt(1)
	;;#ASMSTART
	v_dot2c_f32_f16 v45, v26, v14
	;;#ASMEND
	s_waitcnt lgkmcnt(0)
	;;#ASMSTART
	v_dot2c_f32_f16 v44, v22, v14
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v46, v31, v15
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v45, v27, v15
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v44, v23, v15
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v46, v32, v16
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v45, v28, v16
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v44, v24, v16
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v46, v33, v17
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v45, v29, v17
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v44, v25, v17
	;;#ASMEND
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	v_dot2c_f32_f16 v46, v18, v2
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v45, v10, v2
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v44, v6, v2
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v46, v19, v3
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v45, v11, v3
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v44, v7, v3
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v46, v20, v4
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v45, v12, v4
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v44, v8, v4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v46, v21, v5
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v45, v13, v5
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v44, v9, v5
	;;#ASMEND
.LBB0_10:                               ;   in Loop: Header=BB0_11 Depth=2
	s_or_b64 exec, exec, s[56:57]
	s_addk_i32 s68, 0x400
	s_cmp_ge_u32 s68, s2
	s_cbranch_scc1 .LBB0_33
.LBB0_11:                               ;   Parent Loop BB0_5 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_16 Depth 3
	s_cmp_eq_u32 s68, 0
	s_cselect_b64 s[56:57], -1, 0
	s_add_i32 s58, s36, 0x6800
	s_cmp_eq_u32 s68, s58
	s_cselect_b64 s[60:61], -1, 0
	s_or_b64 s[60:61], s[56:57], s[60:61]
	s_andn2_b64 vcc, exec, s[60:61]
	s_cbranch_vccz .LBB0_13
; %bb.12:                               ;   in Loop: Header=BB0_11 Depth=2
	s_and_saveexec_b64 s[56:57], s[22:23]
	s_cbranch_execz .LBB0_10
	s_branch .LBB0_30
.LBB0_13:                               ;   in Loop: Header=BB0_11 Depth=2
	s_and_b64 s[56:57], s[56:57], exec
	s_cselect_b32 s36, s36, s58
	s_sub_i32 s56, s2, s36
	s_min_u32 s69, s56, 0x6800
	s_lshl_b64 s[56:57], s[36:37], 1
	s_add_u32 s56, s6, s56
	s_addc_u32 s57, s7, s57
	s_mov_b32 s70, 0
	s_mov_b64 s[58:59], 0
	v_mov_b32_e32 v4, v42
	s_barrier
                                        ; implicit-def: $sgpr60_sgpr61
	s_branch .LBB0_16
.LBB0_14:                               ;   in Loop: Header=BB0_16 Depth=3
	s_add_i32 s70, s70, s11
	s_cmpk_gt_u32 s70, 0x67ff
	s_cselect_b64 s[64:65], -1, 0
	s_andn2_b64 s[60:61], s[60:61], exec
	s_and_b64 s[64:65], s[64:65], exec
	v_add_u32_e32 v4, s67, v4
	s_or_b64 s[60:61], s[60:61], s[64:65]
.LBB0_15:                               ;   in Loop: Header=BB0_16 Depth=3
	s_or_b64 exec, exec, s[62:63]
	s_and_b64 s[62:63], exec, s[60:61]
	s_or_b64 s[58:59], s[62:63], s[58:59]
	s_andn2_b64 exec, exec, s[58:59]
	s_cbranch_execz .LBB0_29
.LBB0_16:                               ;   Parent Loop BB0_5 Depth=1
                                        ;     Parent Loop BB0_11 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	v_add_u32_e32 v36, s70, v41
	v_cmp_gt_u32_e32 vcc, s69, v36
	s_or_b64 s[60:61], s[60:61], exec
	s_and_saveexec_b64 s[62:63], vcc
	s_cbranch_execz .LBB0_15
; %bb.17:                               ;   in Loop: Header=BB0_16 Depth=3
	v_lshl_add_u64 v[2:3], v[36:37], 1, s[56:57]
	s_mov_b64 s[64:65], -1
	s_and_b64 vcc, exec, s[40:41]
	s_cbranch_vccz .LBB0_19
; %bb.18:                               ;   in Loop: Header=BB0_16 Depth=3
	global_load_dwordx4 v[6:9], v[2:3], off
	s_mov_b64 s[64:65], 0
	s_waitcnt vmcnt(0)
	ds_write_b128 v4, v[6:9]
.LBB0_19:                               ;   in Loop: Header=BB0_16 Depth=3
	s_andn2_b64 vcc, exec, s[64:65]
	s_cbranch_vccnz .LBB0_21
; %bb.20:                               ;   in Loop: Header=BB0_16 Depth=3
	v_readfirstlane_b32 s64, v4
	s_mov_b32 m0, s64
	s_nop 0
	global_load_lds_dwordx4 v[2:3], off
.LBB0_21:                               ;   in Loop: Header=BB0_16 Depth=3
	v_lshl_add_u64 v[2:3], s[30:31], 1, v[2:3]
	v_add_u32_e32 v5, 0xd000, v4
	s_and_b64 vcc, exec, s[20:21]
	s_mov_b64 s[64:65], -1
	s_cbranch_vccnz .LBB0_23
; %bb.22:                               ;   in Loop: Header=BB0_16 Depth=3
	global_load_dwordx4 v[6:9], v[2:3], off
	s_mov_b64 s[64:65], 0
	s_waitcnt vmcnt(0)
	ds_write_b128 v5, v[6:9]
.LBB0_23:                               ;   in Loop: Header=BB0_16 Depth=3
	s_andn2_b64 vcc, exec, s[64:65]
	s_cbranch_vccnz .LBB0_25
; %bb.24:                               ;   in Loop: Header=BB0_16 Depth=3
	v_readfirstlane_b32 s64, v5
	s_mov_b32 m0, s64
	s_nop 0
	global_load_lds_dwordx4 v[2:3], off
.LBB0_25:                               ;   in Loop: Header=BB0_16 Depth=3
	v_lshl_add_u64 v[2:3], v[2:3], 0, s[48:49]
	v_add_u32_e32 v5, 0x1a000, v4
	s_and_b64 vcc, exec, s[20:21]
	s_mov_b64 s[64:65], -1
	s_cbranch_vccnz .LBB0_27
; %bb.26:                               ;   in Loop: Header=BB0_16 Depth=3
	global_load_dwordx4 v[6:9], v[2:3], off
	s_mov_b64 s[64:65], 0
	s_waitcnt vmcnt(0)
	ds_write_b128 v5, v[6:9]
.LBB0_27:                               ;   in Loop: Header=BB0_16 Depth=3
	s_andn2_b64 vcc, exec, s[64:65]
	s_cbranch_vccnz .LBB0_14
; %bb.28:                               ;   in Loop: Header=BB0_16 Depth=3
	v_readfirstlane_b32 s64, v5
	s_mov_b32 m0, s64
	s_nop 0
	global_load_lds_dwordx4 v[2:3], off
	s_branch .LBB0_14
.LBB0_29:                               ;   in Loop: Header=BB0_11 Depth=2
	s_or_b64 exec, exec, s[58:59]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	s_and_saveexec_b64 s[56:57], s[22:23]
	s_cbranch_execz .LBB0_10
.LBB0_30:                               ;   in Loop: Header=BB0_11 Depth=2
	v_add_u32_e32 v47, s68, v1
	v_min_u32_e32 v36, s17, v47
	v_add_u32_e32 v48, 0x200, v47
	v_lshl_add_u64 v[2:3], v[36:37], 1, v[38:39]
	v_min_u32_e32 v36, s17, v48
	v_lshl_add_u64 v[4:5], v[36:37], 1, v[38:39]
	global_load_dwordx4 v[14:17], v[2:3], off nt
	s_nop 0
	global_load_dwordx4 v[2:5], v[4:5], off nt
	v_cmp_gt_u32_e32 vcc, s2, v47
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v24, 0
	v_mov_b32_e32 v25, 0
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v21, 0
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v11, 0
	v_mov_b32_e32 v12, 0
	v_mov_b32_e32 v13, 0
	v_mov_b32_e32 v6, 0
	v_mov_b32_e32 v7, 0
	v_mov_b32_e32 v8, 0
	v_mov_b32_e32 v9, 0
	s_and_saveexec_b64 s[58:59], vcc
	s_cbranch_execz .LBB0_9
; %bb.31:                               ;   in Loop: Header=BB0_11 Depth=2
	v_subrev_u32_e32 v6, s36, v47
	v_lshlrev_b32_e32 v36, 1, v6
	v_add_u32_e32 v6, 0x1a000, v36
	ds_read_b128 v[30:33], v36
	ds_read_b128 v[26:29], v36 offset:53248
	ds_read_b128 v[22:25], v6
	v_cmp_gt_u32_e32 vcc, s2, v48
	v_mov_b32_e32 v9, 0
	v_mov_b32_e32 v8, 0
	v_mov_b32_e32 v7, 0
	v_mov_b32_e32 v6, 0
	v_mov_b32_e32 v13, 0
	v_mov_b32_e32 v12, 0
	v_mov_b32_e32 v11, 0
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v21, 0
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v18, 0
	s_and_saveexec_b64 s[60:61], vcc
	s_cbranch_execz .LBB0_8
; %bb.32:                               ;   in Loop: Header=BB0_11 Depth=2
	v_add_u32_e32 v6, 0x1a400, v36
	ds_read_b128 v[18:21], v36 offset:1024
	ds_read_b128 v[10:13], v36 offset:54272
	ds_read_b128 v[6:9], v6
	s_branch .LBB0_8
.LBB0_33:                               ;   in Loop: Header=BB0_5 Depth=1
	s_and_saveexec_b64 s[56:57], s[22:23]
	s_cbranch_execz .LBB0_4
; %bb.34:                               ;   in Loop: Header=BB0_5 Depth=1
	v_add_f32_dpp v4, v45, v45 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v46, v46 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_add_f32_dpp v4, v4, v4 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_add_f32_dpp v4, v4, v4 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_add_f32_dpp v4, v4, v4 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_add_f32_dpp v5, v4, v4 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v44, v44 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v6, v5 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v3, v2 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_add_f32_dpp v4, v4, v4 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v4, v4, v4 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v7, v4, v4 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_mov_b32_dpp v8, v7 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_and_b64 exec, exec, s[18:19]
	s_cbranch_execz .LBB0_4
; %bb.35:                               ;   in Loop: Header=BB0_5 Depth=1
	v_add_f32_e32 v4, v2, v3
	v_add_f32_e32 v3, v5, v6
	s_andn2_b64 vcc, exec, s[42:43]
	v_add_f32_e32 v2, v7, v8
	s_cbranch_vccnz .LBB0_37
; %bb.36:                               ;   in Loop: Header=BB0_5 Depth=1
	v_lshlrev_b64 v[6:7], 1, v[34:35]
	v_lshl_add_u64 v[8:9], s[12:13], 0, v[6:7]
	global_load_ushort v5, v[8:9], off
	s_waitcnt vmcnt(1)
	v_cvt_f32_f16_e32 v10, v43
	v_lshl_add_u64 v[6:7], s[14:15], 0, v[6:7]
	v_lshl_add_u64 v[8:9], s[44:45], 1, v[8:9]
	s_waitcnt vmcnt(0)
	v_cvt_f32_f16_e32 v5, v5
	v_mul_f32_e32 v5, s25, v5
	v_fmac_f32_e32 v5, s24, v4
	v_add_f32_e32 v11, v5, v10
	v_cndmask_b32_e64 v5, v11, v5, s[0:1]
	v_cvt_f16_f32_e32 v5, v5
	global_store_short v[6:7], v5, off
	global_load_ushort v5, v[8:9], off
	v_lshl_add_u64 v[6:7], s[46:47], 1, v[6:7]
	v_lshl_add_u64 v[8:9], v[8:9], 0, s[50:51]
	s_waitcnt vmcnt(0)
	v_cvt_f32_f16_e32 v5, v5
	v_mul_f32_e32 v5, s25, v5
	v_fmac_f32_e32 v5, s24, v3
	v_add_f32_e32 v11, v5, v10
	v_cndmask_b32_e64 v5, v11, v5, s[0:1]
	v_cvt_f16_f32_e32 v5, v5
	global_store_short v[6:7], v5, off
	global_load_ushort v5, v[8:9], off
	v_lshl_add_u64 v[6:7], v[6:7], 0, s[52:53]
	s_waitcnt vmcnt(0)
	v_cvt_f32_f16_e32 v5, v5
	v_mul_f32_e32 v5, s25, v5
	v_fmac_f32_e32 v5, s24, v2
	v_add_f32_e32 v8, v5, v10
	v_cndmask_b32_e64 v5, v8, v5, s[0:1]
	v_cvt_f16_f32_e32 v5, v5
	global_store_short v[6:7], v5, off
	s_cbranch_execnz .LBB0_4
	s_branch .LBB0_38
.LBB0_37:                               ;   in Loop: Header=BB0_5 Depth=1
.LBB0_38:                               ;   in Loop: Header=BB0_5 Depth=1
	v_mul_f32_e32 v5, s24, v4
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v4, s24, v4, v43 op_sel_hi:[0,0,1]
	v_cndmask_b32_e64 v4, v4, v5, s[0:1]
	v_cvt_f16_f32_e32 v6, v4
	v_mul_f32_e32 v7, s24, v3
	v_fma_mix_f32 v3, s24, v3, v43 op_sel_hi:[0,0,1]
	v_lshl_add_u64 v[4:5], v[34:35], 1, s[14:15]
	v_cndmask_b32_e64 v3, v3, v7, s[0:1]
	global_store_short v[4:5], v6, off
	v_cvt_f16_f32_e32 v6, v3
	v_mul_f32_e32 v3, s24, v2
	v_fma_mix_f32 v2, s24, v2, v43 op_sel_hi:[0,0,1]
	v_cndmask_b32_e64 v2, v2, v3, s[0:1]
	v_cvt_f16_f32_e32 v7, v2
	v_lshl_add_u64 v[2:3], s[46:47], 1, v[4:5]
	global_store_short v[2:3], v6, off
	v_lshl_add_u64 v[2:3], v[2:3], 0, s[52:53]
	global_store_short v[2:3], v7, off
	s_branch .LBB0_4
.LBB0_39:
	s_or_b64 exec, exec, s[34:35]
	s_mov_b64 s[0:1], 0
.LBB0_40:
	s_andn2_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB0_77
; %bb.41:
	v_and_b32_e32 v8, 0x3ff, v0
	s_mul_i32 s9, s2, 3
	v_lshlrev_b32_e32 v38, 3, v8
	s_cmp_eq_u32 s30, s2
	s_mov_b64 s[18:19], 0
	s_cbranch_scc1 .LBB0_43
; %bb.42:
	s_and_b32 s20, s6, 15
	s_mov_b32 s21, 0
	s_mov_b64 s[0:1], -1
	s_min_u32 s9, s9, 0x14000
	v_lshl_add_u32 v9, v40, 9, v38
	s_cbranch_execz .LBB0_44
	s_branch .LBB0_45
.LBB0_43:
                                        ; implicit-def: $sgpr20_sgpr21
	s_mov_b64 s[0:1], 0
	s_min_u32 s9, s9, 0x14000
	v_lshl_add_u32 v9, v40, 9, v38
.LBB0_44:
	s_and_b32 s0, s2, 7
	s_cmp_lg_u32 s0, 0
	s_mov_b32 s21, 0
	s_cselect_b64 s[0:1], -1, 0
	s_and_b32 s20, s6, 15
	s_cmp_lg_u64 s[20:21], 0
	s_cselect_b64 s[18:19], -1, 0
	s_or_b64 s[0:1], s[0:1], s[18:19]
	s_mov_b64 s[18:19], -1
.LBB0_45:
	s_andn2_b64 vcc, exec, s[0:1]
	v_cmp_gt_u32_e64 s[0:1], s9, v9
	s_cbranch_vccz .LBB0_47
; %bb.46:
	s_andn2_b64 vcc, exec, s[18:19]
	s_cbranch_vccz .LBB0_55
	s_branch .LBB0_59
.LBB0_47:
	s_and_saveexec_b64 s[18:19], s[0:1]
	s_cbranch_execz .LBB0_54
; %bb.48:
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
	v_lshl_add_u32 v10, v40, 10, v0
	v_mov_b32_e32 v0, v9
	s_branch .LBB0_50
.LBB0_49:                               ;   in Loop: Header=BB0_50 Depth=1
	v_add_u32_e32 v0, 0x2000, v0
	v_cmp_le_u32_e32 vcc, s9, v0
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[4:5]
	s_or_b64 s[0:1], vcc, s[0:1]
	v_add_u32_e32 v10, 0x4000, v10
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB0_54
.LBB0_50:                               ; =>This Inner Loop Header: Depth=1
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
	s_cbranch_vccz .LBB0_52
; %bb.51:                               ;   in Loop: Header=BB0_50 Depth=1
	global_load_dwordx4 v[12:15], v[6:7], off
	s_mov_b64 s[22:23], 0
	s_waitcnt vmcnt(0)
	ds_write_b128 v10, v[12:15]
.LBB0_52:                               ;   in Loop: Header=BB0_50 Depth=1
	s_andn2_b64 vcc, exec, s[22:23]
	s_cbranch_vccnz .LBB0_49
; %bb.53:                               ;   in Loop: Header=BB0_50 Depth=1
	v_readfirstlane_b32 s22, v10
	s_mov_b32 m0, s22
	s_nop 0
	global_load_lds_dwordx4 v[6:7], off
	s_branch .LBB0_49
.LBB0_54:
	s_or_b64 exec, exec, s[18:19]
	s_cbranch_execnz .LBB0_59
.LBB0_55:
	v_cmp_gt_u32_e32 vcc, s9, v9
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_58
; %bb.56:
	v_lshlrev_b32_e32 v0, 4, v8
	v_lshl_add_u32 v2, v40, 10, v0
	v_lshlrev_b32_e32 v0, 1, v9
	v_mov_b32_e32 v1, 0
	v_lshl_add_u64 v[0:1], s[6:7], 0, v[0:1]
	s_mov_b64 s[6:7], 0
	s_mov_b64 s[18:19], 0x4000
.LBB0_57:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s11, v2
	s_mov_b32 m0, s11
	v_add_u32_e32 v9, 0x2000, v9
	global_load_lds_dwordx4 v[0:1], off
	v_cmp_le_u32_e32 vcc, s9, v9
	v_add_u32_e32 v2, 0x4000, v2
	s_or_b64 s[6:7], vcc, s[6:7]
	v_lshl_add_u64 v[0:1], v[0:1], 0, s[18:19]
	s_andn2_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB0_57
.LBB0_58:
	s_or_b64 exec, exec, s[0:1]
.LBB0_59:
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
	s_max_i32 s41, s0, 4
	s_mul_i32 s0, s41, s16
	v_add_u32_e32 v32, s0, v40
	v_cmp_gt_u32_e32 vcc, s41, v40
	v_cmp_gt_u32_e64 s[0:1], s3, v32
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_77
; %bb.60:
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
	s_add_i32 s11, s2, -8
	s_add_i32 s33, s3, -1
	s_ashr_i32 s40, s10, 31
	v_cmp_eq_u32_e64 s[16:17], 63, v8
	v_cmp_neq_f32_e64 s[20:21], s25, 0
	s_mul_i32 s41, s41, s8
	s_lshl_b32 s42, s2, 1
	s_lshl_b64 s[26:27], s[22:23], 1
	s_lshl_b64 s[34:35], s[30:31], 1
	v_lshlrev_b32_e32 v39, 4, v8
	s_lshl_b32 s43, s2, 2
	v_mov_b32_e32 v35, 0
	v_cmp_ne_u32_e64 s[8:9], 1, v0
                                        ; implicit-def: $vgpr40
	s_branch .LBB0_62
.LBB0_61:                               ;   in Loop: Header=BB0_62 Depth=1
	s_or_b64 exec, exec, s[36:37]
	v_add_u32_e32 v32, s41, v32
	v_cmp_le_u32_e32 vcc, s3, v32
	s_or_b64 s[18:19], vcc, s[18:19]
	s_andn2_b64 exec, exec, s[18:19]
	s_cbranch_execz .LBB0_77
.LBB0_62:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_68 Depth 2
	s_and_b64 vcc, exec, s[6:7]
	s_cbranch_vccnz .LBB0_64
; %bb.63:                               ;   in Loop: Header=BB0_62 Depth=1
	v_mov_b32_e32 v33, v35
	v_lshl_add_u64 v[0:1], v[32:33], 1, s[28:29]
	global_load_ushort v40, v[0:1], off
.LBB0_64:                               ;   in Loop: Header=BB0_62 Depth=1
	s_and_b64 vcc, exec, s[8:9]
	s_cbranch_vccnz .LBB0_71
; %bb.65:                               ;   in Loop: Header=BB0_62 Depth=1
	v_min_u32_e32 v3, s33, v32
	v_mad_u64_u32 v[0:1], s[36:37], v3, s10, 0
	v_mov_b32_e32 v2, v1
	v_mad_u64_u32 v[2:3], s[36:37], v3, s40, v[2:3]
	v_mov_b32_e32 v1, v2
	v_lshl_add_u64 v[36:37], v[0:1], 1, s[4:5]
	v_mov_b32_e32 v42, 0
	s_mov_b32 s44, 0
	v_mov_b32_e32 v43, v39
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v33, 0
	s_branch .LBB0_68
.LBB0_66:                               ;   in Loop: Header=BB0_68 Depth=2
	s_or_b64 exec, exec, s[38:39]
.LBB0_67:                               ;   in Loop: Header=BB0_68 Depth=2
	s_or_b64 exec, exec, s[36:37]
	s_waitcnt vmcnt(1) lgkmcnt(2)
	;;#ASMSTART
	v_dot2c_f32_f16 v42, v28, v4
	;;#ASMEND
	s_waitcnt lgkmcnt(1)
	;;#ASMSTART
	v_dot2c_f32_f16 v41, v24, v4
	;;#ASMEND
	s_waitcnt lgkmcnt(0)
	;;#ASMSTART
	v_dot2c_f32_f16 v33, v16, v4
	;;#ASMEND
	s_addk_i32 s44, 0x400
	;;#ASMSTART
	v_dot2c_f32_f16 v42, v29, v5
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v41, v25, v5
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v33, v17, v5
	;;#ASMEND
	s_cmp_ge_u32 s44, s2
	;;#ASMSTART
	v_dot2c_f32_f16 v42, v30, v6
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v41, v26, v6
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v33, v18, v6
	;;#ASMEND
	v_add_u32_e32 v43, 0x800, v43
	;;#ASMSTART
	v_dot2c_f32_f16 v42, v31, v7
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v41, v27, v7
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v33, v19, v7
	;;#ASMEND
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	v_dot2c_f32_f16 v42, v20, v0
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v41, v12, v0
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v33, v8, v0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v42, v21, v1
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v41, v13, v1
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v33, v9, v1
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v42, v22, v2
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v41, v14, v2
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v33, v10, v2
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v42, v23, v3
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v41, v15, v3
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v33, v11, v3
	;;#ASMEND
	s_cbranch_scc1 .LBB0_72
.LBB0_68:                               ;   Parent Loop BB0_62 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_add_u32_e32 v12, s44, v38
	v_min_u32_e32 v34, s11, v12
	v_add_u32_e32 v44, 0x200, v12
	v_lshl_add_u64 v[8:9], v[34:35], 1, v[36:37]
	v_min_u32_e32 v34, s11, v44
	v_lshl_add_u64 v[10:11], v[34:35], 1, v[36:37]
	global_load_dwordx4 v[4:7], v[8:9], off nt
	global_load_dwordx4 v[0:3], v[10:11], off nt
	v_cmp_gt_u32_e32 vcc, s2, v12
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v24, 0
	v_mov_b32_e32 v25, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v16, 0
	v_mov_b32_e32 v17, 0
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v21, 0
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v12, 0
	v_mov_b32_e32 v13, 0
	v_mov_b32_e32 v14, 0
	v_mov_b32_e32 v15, 0
	v_mov_b32_e32 v8, 0
	v_mov_b32_e32 v9, 0
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v11, 0
	s_and_saveexec_b64 s[36:37], vcc
	s_cbranch_execz .LBB0_67
; %bb.69:                               ;   in Loop: Header=BB0_68 Depth=2
	v_add_u32_e32 v45, s42, v43
	ds_read_b128 v[28:31], v43
	v_add_u32_e32 v34, s43, v43
	ds_read_b128 v[24:27], v45
	ds_read_b128 v[16:19], v34
	v_cmp_gt_u32_e32 vcc, s2, v44
	v_mov_b32_e32 v11, 0
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v9, 0
	v_mov_b32_e32 v8, 0
	v_mov_b32_e32 v15, 0
	v_mov_b32_e32 v14, 0
	v_mov_b32_e32 v13, 0
	v_mov_b32_e32 v12, 0
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v21, 0
	v_mov_b32_e32 v20, 0
	s_and_saveexec_b64 s[38:39], vcc
	s_cbranch_execz .LBB0_66
; %bb.70:                               ;   in Loop: Header=BB0_68 Depth=2
	ds_read_b128 v[20:23], v43 offset:1024
	ds_read_b128 v[12:15], v45 offset:1024
	ds_read_b128 v[8:11], v34 offset:1024
	s_branch .LBB0_66
.LBB0_71:                               ;   in Loop: Header=BB0_62 Depth=1
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v42, 0
.LBB0_72:                               ;   in Loop: Header=BB0_62 Depth=1
	; sched_barrier mask(0x00000000)
	s_nop 0
	v_add_f32_dpp v2, v41, v41 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v42, v42 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_add_f32_dpp v2, v2, v2 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v0, v0 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_add_f32_dpp v2, v2, v2 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v0, v0 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_add_f32_dpp v2, v2, v2 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v0, v0 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_add_f32_dpp v3, v2, v2 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v33, v33 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v0, v0 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v4, v3 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v1, v0 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_add_f32_dpp v2, v2, v2 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v2, v2, v2 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v5, v2, v2 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_mov_b32_dpp v6, v5 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_and_saveexec_b64 s[36:37], s[16:17]
	s_cbranch_execz .LBB0_61
; %bb.73:                               ;   in Loop: Header=BB0_62 Depth=1
	v_add_f32_e32 v2, v0, v1
	v_add_f32_e32 v1, v3, v4
	s_andn2_b64 vcc, exec, s[20:21]
	v_add_f32_e32 v0, v5, v6
	s_cbranch_vccnz .LBB0_75
; %bb.74:                               ;   in Loop: Header=BB0_62 Depth=1
	v_mov_b32_e32 v33, v35
	v_lshlrev_b64 v[4:5], 1, v[32:33]
	v_lshl_add_u64 v[6:7], s[12:13], 0, v[4:5]
	global_load_ushort v3, v[6:7], off
	s_waitcnt vmcnt(1)
	v_cvt_f32_f16_e32 v8, v40
	v_lshl_add_u64 v[4:5], s[14:15], 0, v[4:5]
	v_lshl_add_u64 v[6:7], s[22:23], 1, v[6:7]
	s_waitcnt vmcnt(0)
	v_cvt_f32_f16_e32 v3, v3
	v_mul_f32_e32 v3, s25, v3
	v_fmac_f32_e32 v3, s24, v2
	v_add_f32_e32 v9, v3, v8
	v_cndmask_b32_e64 v3, v9, v3, s[0:1]
	v_cvt_f16_f32_e32 v3, v3
	global_store_short v[4:5], v3, off
	global_load_ushort v3, v[6:7], off
	v_lshl_add_u64 v[4:5], s[30:31], 1, v[4:5]
	v_lshl_add_u64 v[6:7], v[6:7], 0, s[26:27]
	s_waitcnt vmcnt(0)
	v_cvt_f32_f16_e32 v3, v3
	v_mul_f32_e32 v3, s25, v3
	v_fmac_f32_e32 v3, s24, v1
	v_add_f32_e32 v9, v3, v8
	v_cndmask_b32_e64 v3, v9, v3, s[0:1]
	v_cvt_f16_f32_e32 v3, v3
	global_store_short v[4:5], v3, off
	global_load_ushort v3, v[6:7], off
	v_lshl_add_u64 v[4:5], v[4:5], 0, s[34:35]
	s_waitcnt vmcnt(0)
	v_cvt_f32_f16_e32 v3, v3
	v_mul_f32_e32 v3, s25, v3
	v_fmac_f32_e32 v3, s24, v0
	v_add_f32_e32 v6, v3, v8
	v_cndmask_b32_e64 v3, v6, v3, s[0:1]
	v_cvt_f16_f32_e32 v3, v3
	global_store_short v[4:5], v3, off
	s_cbranch_execnz .LBB0_61
	s_branch .LBB0_76
.LBB0_75:                               ;   in Loop: Header=BB0_62 Depth=1
.LBB0_76:                               ;   in Loop: Header=BB0_62 Depth=1
	v_mul_f32_e32 v3, s24, v2
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v2, s24, v2, v40 op_sel_hi:[0,0,1]
	v_cndmask_b32_e64 v2, v2, v3, s[0:1]
	v_cvt_f16_f32_e32 v4, v2
	v_mov_b32_e32 v33, v35
	v_lshl_add_u64 v[2:3], v[32:33], 1, s[14:15]
	global_store_short v[2:3], v4, off
	v_mul_f32_e32 v4, s24, v1
	v_fma_mix_f32 v1, s24, v1, v40 op_sel_hi:[0,0,1]
	v_cndmask_b32_e64 v1, v1, v4, s[0:1]
	v_cvt_f16_f32_e32 v4, v1
	v_mul_f32_e32 v1, s24, v0
	v_fma_mix_f32 v0, s24, v0, v40 op_sel_hi:[0,0,1]
	v_cndmask_b32_e64 v0, v0, v1, s[0:1]
	v_cvt_f16_f32_e32 v5, v0
	v_lshl_add_u64 v[0:1], s[30:31], 1, v[2:3]
	global_store_short v[0:1], v4, off
	v_lshl_add_u64 v[0:1], v[0:1], 0, s[34:35]
	global_store_short v[0:1], v5, off
	s_branch .LBB0_61
.LBB0_77:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wvSplitK_vllm_f16_tn_bias_n3_y1u2
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
		.amdhsa_accum_offset 52
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
	.size	wvSplitK_vllm_f16_tn_bias_n3_y1u2, .Lfunc_end0-wvSplitK_vllm_f16_tn_bias_n3_y1u2
                                        ; -- End function
	.set wvSplitK_vllm_f16_tn_bias_n3_y1u2.num_vgpr, 49
	.set wvSplitK_vllm_f16_tn_bias_n3_y1u2.num_agpr, 0
	.set wvSplitK_vllm_f16_tn_bias_n3_y1u2.numbered_sgpr, 71
	.set wvSplitK_vllm_f16_tn_bias_n3_y1u2.num_named_barrier, 0
	.set wvSplitK_vllm_f16_tn_bias_n3_y1u2.private_seg_size, 0
	.set wvSplitK_vllm_f16_tn_bias_n3_y1u2.uses_vcc, 1
	.set wvSplitK_vllm_f16_tn_bias_n3_y1u2.uses_flat_scratch, 0
	.set wvSplitK_vllm_f16_tn_bias_n3_y1u2.has_dyn_sized_stack, 0
	.set wvSplitK_vllm_f16_tn_bias_n3_y1u2.has_recursion, 0
	.set wvSplitK_vllm_f16_tn_bias_n3_y1u2.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 5180
; TotalNumSgprs: 77
; NumVgprs: 49
; NumAgprs: 0
; TotalNumVgprs: 49
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 163840 bytes/workgroup (compile time only)
; SGPRBlocks: 12
; VGPRBlocks: 12
; NumSGPRsForWavesPerEU: 102
; NumVGPRsForWavesPerEU: 97
; AccumOffset: 52
; Occupancy: 4
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 1
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 12
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_2d6457f51b751c83,@object ; @__hip_cuid_2d6457f51b751c83
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_2d6457f51b751c83
__hip_cuid_2d6457f51b751c83:
	.byte	0                               ; 0x0
	.size	__hip_cuid_2d6457f51b751c83, 1

	.ident	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.4 26084 f58b06dce1f9c15707c5f808fd002e18c2accf7e)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_2d6457f51b751c83
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
    DataType: h
    DestDataType: h
    ComputeDataType: s
    HighPrecisionAccumulate: True
    TransposeA: True
    TransposeB: False
    UseBeta: True
    Batched: True
    UseBias: 1
    BiasSrc: D
    BiasDataTypeList: ['h']
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
    .name:           wvSplitK_vllm_f16_tn_bias_n3_y1u2
    .private_segment_fixed_size: 0
    .sgpr_count:     77
    .sgpr_spill_count: 0
    .symbol:         wvSplitK_vllm_f16_tn_bias_n3_y1u2.kd
    .vgpr_count:     49
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 1
...

	.end_amdgpu_metadata
