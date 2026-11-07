	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 4
	.text
	.protected	wvSplitK_vllm_f16_tn_n2_y3u2 ; -- Begin function wvSplitK_vllm_f16_tn_n2_y3u2
	.globl	wvSplitK_vllm_f16_tn_n2_y3u2
	.p2align	8
	.type	wvSplitK_vllm_f16_tn_n2_y3u2,@function
wvSplitK_vllm_f16_tn_n2_y3u2:           ; @wvSplitK_vllm_f16_tn_n2_y3u2
; %bb.128:
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.129:
.LBB0_0:
	s_load_dwordx4 s[28:31], s[0:1], 0x38
	s_mov_b32 s26, s11
	s_mov_b64 s[0:1], -1
	s_cmp_gt_i32 s2, 0xa000
	v_bfe_u32 v54, v0, 10, 10
	s_mul_i32 s11, s8, 3
	s_mul_i32 s17, s8, 48
	s_cbranch_scc0 .LBB0_67
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
	v_cmp_gt_u32_e32 vcc, s8, v54
	s_and_saveexec_b64 s[34:35], vcc
	s_cbranch_execz .LBB0_66
; %bb.2:
	s_mul_i32 s9, s8, 3
	v_cvt_f32_u32_e32 v3, s9
	s_mul_i32 s0, s8, s16
	v_add_u32_e32 v1, s0, v54
	v_lshl_add_u32 v2, v1, 1, v1
	v_rcp_iflag_f32_e32 v3, v3
	v_add_u32_e32 v1, 3, v2
	v_cmp_gt_u32_e32 vcc, s3, v2
	v_cmp_lt_u32_e64 s[0:1], s3, v1
	s_add_i32 s58, s3, -3
	v_mov_b32_e32 v4, s58
	s_and_b64 s[0:1], vcc, s[0:1]
	v_cndmask_b32_e64 v44, v2, v4, s[0:1]
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
	s_cselect_b32 s59, s3, s1
	v_cmp_gt_u32_e64 s[0:1], s59, v44
	s_and_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB0_66
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
	s_lshl_b32 s60, s8, 9
	s_add_i32 s61, s2, -8
	s_ashr_i32 s62, s10, 31
	s_ashr_i32 s21, s31, 31
	s_lshl_b64 s[18:19], s[18:19], 1
	s_mov_b32 s20, s31
	s_add_u32 s46, s12, s18
	v_and_b32_e32 v2, 0x3ff, v0
	s_addc_u32 s47, s13, s19
	s_lshl_b64 s[18:19], s[20:21], 1
	v_sub_u32_e64 v1, v1, s3 clamp
	v_lshlrev_b32_e32 v55, 3, v2
	v_cmp_eq_u32_e64 s[0:1], 63, v2
	s_add_u32 s48, s14, s18
	v_lshlrev_b32_e32 v2, 4, v2
	s_mov_b64 s[38:39], 0
	v_cndmask_b32_e32 v1, 0, v1, vcc
	v_lshl_add_u32 v56, v54, 9, v55
	s_mul_i32 s63, s8, s11
	v_cmp_eq_f32_e64 s[42:43], s29, 0
	v_cmp_neq_f32_e64 s[44:45], s29, 0
	s_addc_u32 s49, s15, s19
	v_lshl_add_u32 v57, v54, 10, v2
	s_lshl_b32 s64, s8, 10
	v_mov_b32_e32 v43, 0
	s_branch .LBB0_6
.LBB0_4:                                ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[52:53]
	v_add_u32_e32 v3, 3, v2
	v_cmp_gt_u32_e32 vcc, s3, v2
	v_cmp_lt_u32_e64 s[8:9], s3, v3
	v_subrev_u32_e32 v3, s3, v3
	v_mov_b32_e32 v4, s58
	s_and_b64 vcc, vcc, s[8:9]
	v_cndmask_b32_e32 v2, v2, v4, vcc
	v_cndmask_b32_e32 v1, v1, v3, vcc
.LBB0_5:                                ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[50:51]
	v_cmp_le_u32_e32 vcc, s59, v2
	s_or_b64 s[38:39], vcc, s[38:39]
	v_mov_b32_e32 v44, v2
	s_andn2_b64 exec, exec, s[38:39]
	s_cbranch_execz .LBB0_66
.LBB0_6:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_10 Depth 2
                                        ;       Child Loop BB0_15 Depth 3
	v_min_u32_e32 v5, s33, v44
	v_mad_u64_u32 v[2:3], s[20:21], v5, s10, 0
	v_mov_b32_e32 v4, v3
	v_mad_u64_u32 v[4:5], s[20:21], v5, s62, v[4:5]
	v_mov_b32_e32 v3, v4
	v_add_u32_e32 v4, 1, v44
	v_min_u32_e32 v7, s33, v4
	v_mad_u64_u32 v[4:5], s[20:21], v7, s10, 0
	v_mov_b32_e32 v6, v5
	v_add_u32_e32 v46, 2, v44
	v_mad_u64_u32 v[6:7], s[20:21], v7, s62, v[6:7]
	v_min_u32_e32 v9, s33, v46
	v_mov_b32_e32 v5, v6
	v_mad_u64_u32 v[6:7], s[20:21], v9, s10, 0
	v_mov_b32_e32 v8, v7
	v_mad_u64_u32 v[8:9], s[20:21], v9, s62, v[8:9]
	v_mov_b32_e32 v7, v8
	v_cmp_gt_u32_e64 s[8:9], s3, v44
	v_cmp_le_u32_e64 s[18:19], s3, v44
	v_lshlrev_b64 v[48:49], 1, v[2:3]
	v_lshlrev_b64 v[50:51], 1, v[4:5]
	v_lshlrev_b64 v[52:53], 1, v[6:7]
	v_mov_b32_e32 v45, v43
	v_mov_b32_e32 v47, v43
	v_mov_b32_e32 v58, v43
	v_mov_b32_e32 v59, v43
	v_mov_b32_e32 v60, v43
	v_mov_b32_e32 v61, v43
	s_mov_b32 s36, s37
	s_mov_b32 s54, s37
	s_branch .LBB0_10
.LBB0_7:                                ;   in Loop: Header=BB0_10 Depth=2
	s_or_b64 exec, exec, s[24:25]
.LBB0_8:                                ;   in Loop: Header=BB0_10 Depth=2
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(5) lgkmcnt(1)
	;;#ASMSTART
	v_dot2c_f32_f16 v61, v38, v34
	;;#ASMEND
	s_waitcnt vmcnt(4)
	;;#ASMSTART
	v_dot2c_f32_f16 v60, v38, v26
	;;#ASMEND
	s_waitcnt vmcnt(3)
	;;#ASMSTART
	v_dot2c_f32_f16 v59, v38, v22
	;;#ASMEND
	s_waitcnt lgkmcnt(0)
	;;#ASMSTART
	v_dot2c_f32_f16 v58, v30, v34
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v47, v30, v26
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v45, v30, v22
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v61, v39, v35
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v60, v39, v27
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v59, v39, v23
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v58, v31, v35
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v47, v31, v27
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v45, v31, v23
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v61, v40, v36
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v60, v40, v28
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v59, v40, v24
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v58, v32, v36
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v47, v32, v28
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v45, v32, v24
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v61, v41, v37
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v60, v41, v29
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v59, v41, v25
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v58, v33, v37
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v47, v33, v29
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v45, v33, v25
	;;#ASMEND
	s_waitcnt vmcnt(2)
	;;#ASMSTART
	v_dot2c_f32_f16 v61, v18, v10
	;;#ASMEND
	s_waitcnt vmcnt(1)
	;;#ASMSTART
	v_dot2c_f32_f16 v60, v18, v6
	;;#ASMEND
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	v_dot2c_f32_f16 v59, v18, v2
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v58, v14, v10
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v47, v14, v6
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v45, v14, v2
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v61, v19, v11
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v60, v19, v7
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v59, v19, v3
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v58, v15, v11
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v47, v15, v7
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v45, v15, v3
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v61, v20, v12
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v60, v20, v8
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v59, v20, v4
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v58, v16, v12
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v47, v16, v8
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v45, v16, v4
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v61, v21, v13
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v60, v21, v9
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v59, v21, v5
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v58, v17, v13
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v47, v17, v9
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v45, v17, v5
	;;#ASMEND
.LBB0_9:                                ;   in Loop: Header=BB0_10 Depth=2
	s_or_b64 exec, exec, s[20:21]
	s_addk_i32 s54, 0x400
	s_cmp_ge_u32 s54, s2
	s_cbranch_scc1 .LBB0_28
.LBB0_10:                               ;   Parent Loop BB0_6 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_15 Depth 3
	s_cmp_eq_u32 s54, 0
	s_cselect_b64 s[20:21], -1, 0
	s_add_i32 s22, s36, 0xa000
	s_cmp_eq_u32 s54, s22
	s_cselect_b64 s[24:25], -1, 0
	s_or_b64 s[24:25], s[20:21], s[24:25]
	s_andn2_b64 vcc, exec, s[24:25]
	s_cbranch_vccz .LBB0_12
; %bb.11:                               ;   in Loop: Header=BB0_10 Depth=2
	s_and_saveexec_b64 s[20:21], s[8:9]
	s_cbranch_execz .LBB0_9
	s_branch .LBB0_25
.LBB0_12:                               ;   in Loop: Header=BB0_10 Depth=2
	s_and_b64 s[20:21], s[20:21], exec
	s_cselect_b32 s36, s36, s22
	s_sub_i32 s20, s2, s36
	s_min_u32 s55, s20, 0xa000
	s_lshl_b64 s[20:21], s[36:37], 1
	s_add_u32 s20, s6, s20
	s_addc_u32 s21, s7, s21
	s_mov_b32 s56, 0
	s_mov_b64 s[22:23], 0
	v_mov_b32_e32 v4, v57
	s_barrier
                                        ; implicit-def: $sgpr24_sgpr25
	s_branch .LBB0_15
.LBB0_13:                               ;   in Loop: Header=BB0_15 Depth=3
	s_add_i32 s56, s56, s60
	s_cmpk_gt_u32 s56, 0x9fff
	s_cselect_b64 s[52:53], -1, 0
	s_andn2_b64 s[24:25], s[24:25], exec
	s_and_b64 s[52:53], s[52:53], exec
	v_add_u32_e32 v4, s64, v4
	s_or_b64 s[24:25], s[24:25], s[52:53]
.LBB0_14:                               ;   in Loop: Header=BB0_15 Depth=3
	s_or_b64 exec, exec, s[50:51]
	s_and_b64 s[50:51], exec, s[24:25]
	s_or_b64 s[22:23], s[50:51], s[22:23]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execz .LBB0_24
.LBB0_15:                               ;   Parent Loop BB0_6 Depth=1
                                        ;     Parent Loop BB0_10 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	v_add_u32_e32 v42, s56, v56
	v_cmp_gt_u32_e32 vcc, s55, v42
	s_or_b64 s[24:25], s[24:25], exec
	s_and_saveexec_b64 s[50:51], vcc
	s_cbranch_execz .LBB0_14
; %bb.16:                               ;   in Loop: Header=BB0_15 Depth=3
	v_lshl_add_u64 v[2:3], v[42:43], 1, s[20:21]
	s_mov_b64 s[52:53], -1
	s_and_b64 vcc, exec, s[40:41]
	s_cbranch_vccz .LBB0_18
; %bb.17:                               ;   in Loop: Header=BB0_15 Depth=3
	global_load_dwordx4 v[6:9], v[2:3], off
	s_mov_b64 s[52:53], 0
	s_waitcnt vmcnt(0)
	ds_write_b128 v4, v[6:9]
.LBB0_18:                               ;   in Loop: Header=BB0_15 Depth=3
	s_andn2_b64 vcc, exec, s[52:53]
	s_cbranch_vccnz .LBB0_20
; %bb.19:                               ;   in Loop: Header=BB0_15 Depth=3
	v_readfirstlane_b32 s52, v4
	s_mov_b32 m0, s52
	s_nop 0
	global_load_lds_dwordx4 v[2:3], off
.LBB0_20:                               ;   in Loop: Header=BB0_15 Depth=3
	v_lshl_add_u64 v[2:3], s[26:27], 1, v[2:3]
	v_add_u32_e32 v5, 0x14000, v4
	s_andn2_b64 vcc, exec, s[40:41]
	s_mov_b64 s[52:53], -1
	s_cbranch_vccnz .LBB0_22
; %bb.21:                               ;   in Loop: Header=BB0_15 Depth=3
	global_load_dwordx4 v[6:9], v[2:3], off
	s_mov_b64 s[52:53], 0
	s_waitcnt vmcnt(0)
	ds_write_b128 v5, v[6:9]
.LBB0_22:                               ;   in Loop: Header=BB0_15 Depth=3
	s_andn2_b64 vcc, exec, s[52:53]
	s_cbranch_vccnz .LBB0_13
; %bb.23:                               ;   in Loop: Header=BB0_15 Depth=3
	v_readfirstlane_b32 s52, v5
	s_mov_b32 m0, s52
	s_nop 0
	global_load_lds_dwordx4 v[2:3], off
	s_branch .LBB0_13
.LBB0_24:                               ;   in Loop: Header=BB0_10 Depth=2
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	s_and_saveexec_b64 s[20:21], s[8:9]
	s_cbranch_execz .LBB0_9
.LBB0_25:                               ;   in Loop: Header=BB0_10 Depth=2
	v_add_u32_e32 v62, s54, v55
	v_min_u32_e32 v42, s61, v62
	v_lshl_add_u64 v[2:3], v[42:43], 1, s[4:5]
	v_add_u32_e32 v63, 0x200, v62
	v_lshl_add_u64 v[4:5], v[2:3], 0, v[48:49]
	v_min_u32_e32 v42, s61, v63
	v_lshl_add_u64 v[6:7], v[2:3], 0, v[50:51]
	global_load_dwordx4 v[34:37], v[4:5], off nt
	global_load_dwordx4 v[26:29], v[6:7], off nt
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[52:53]
	v_lshl_add_u64 v[4:5], v[42:43], 1, s[4:5]
	v_lshl_add_u64 v[6:7], v[4:5], 0, v[48:49]
	global_load_dwordx4 v[22:25], v[2:3], off nt
	global_load_dwordx4 v[10:13], v[6:7], off nt
	v_lshl_add_u64 v[2:3], v[4:5], 0, v[50:51]
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[52:53]
	global_load_dwordx4 v[6:9], v[2:3], off nt
	s_nop 0
	global_load_dwordx4 v[2:5], v[4:5], off nt
	v_cmp_gt_u32_e32 vcc, s2, v62
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v39, 0
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v21, 0
	v_mov_b32_e32 v14, 0
	v_mov_b32_e32 v15, 0
	v_mov_b32_e32 v16, 0
	v_mov_b32_e32 v17, 0
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_8
; %bb.26:                               ;   in Loop: Header=BB0_10 Depth=2
	v_subrev_u32_e32 v14, s36, v62
	v_lshlrev_b32_e32 v42, 1, v14
	v_add_u32_e32 v14, 0x14000, v42
	ds_read_b128 v[38:41], v42
	ds_read_b128 v[30:33], v14
	v_cmp_gt_u32_e32 vcc, s2, v63
	v_mov_b32_e32 v17, 0
	v_mov_b32_e32 v16, 0
	v_mov_b32_e32 v15, 0
	v_mov_b32_e32 v14, 0
	v_mov_b32_e32 v21, 0
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v18, 0
	s_and_saveexec_b64 s[24:25], vcc
	s_cbranch_execz .LBB0_7
; %bb.27:                               ;   in Loop: Header=BB0_10 Depth=2
	v_add_u32_e32 v14, 0x14400, v42
	ds_read_b128 v[18:21], v42 offset:1024
	ds_read_b128 v[14:17], v14
	s_branch .LBB0_7
.LBB0_28:                               ;   in Loop: Header=BB0_6 Depth=1
	v_add_u32_e32 v2, s63, v44
	s_and_saveexec_b64 s[8:9], s[18:19]
	s_xor_b64 s[8:9], exec, s[8:9]
; %bb.29:                               ;   in Loop: Header=BB0_6 Depth=1
                                        ; implicit-def: $vgpr44
                                        ; implicit-def: $vgpr46
; %bb.30:                               ;   in Loop: Header=BB0_6 Depth=1
	s_andn2_saveexec_b64 s[50:51], s[8:9]
	s_cbranch_execz .LBB0_5
; %bb.31:                               ;   in Loop: Header=BB0_6 Depth=1
	v_add_f32_dpp v7, v59, v59 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v61, v61 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v60, v60 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v7, v7 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v3, v3 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v5, v5 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v7, v7 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v3, v3 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v5, v5 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v7, v7 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v3, v3 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v5, v5 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v7, v7 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v58, v58 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v3, v3 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v5, v5 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v7, v7 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v4, v3 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v6, v5 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v7, v7 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v10, v8 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_add_f32_dpp v7, v7, v7 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v11, v7, v7 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v47, v47 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v12, v11 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v7, v7 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v7, v7, v7 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v7, v7, v7 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v13, v7, v7 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v45, v45 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v14, v13 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v7, v7 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v7, v7, v7 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v7, v7, v7 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v15, v7, v7 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_mov_b32_dpp v16, v15 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_and_saveexec_b64 s[52:53], s[0:1]
	s_cbranch_execz .LBB0_4
; %bb.32:                               ;   in Loop: Header=BB0_6 Depth=1
	v_cmp_eq_u32_e64 s[8:9], 0, v1
	s_and_b64 s[18:19], s[42:43], s[8:9]
	v_add_f32_e32 v9, v3, v4
	v_add_f32_e32 v7, v5, v6
	v_add_f32_e32 v6, v8, v10
	v_add_f32_e32 v5, v11, v12
	v_add_f32_e32 v4, v13, v14
	v_add_f32_e32 v3, v15, v16
	s_xor_b64 s[20:21], s[18:19], -1
	v_mov_b32_e32 v47, v43
	s_mov_b64 s[18:19], 0
                                        ; implicit-def: $vgpr8
	s_and_saveexec_b64 s[22:23], s[20:21]
	s_xor_b64 s[54:55], exec, s[22:23]
	s_cbranch_execnz .LBB0_35
; %bb.33:                               ;   in Loop: Header=BB0_6 Depth=1
	s_andn2_saveexec_b64 s[8:9], s[54:55]
	s_cbranch_execnz .LBB0_64
.LBB0_34:                               ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[8:9]
	s_and_b64 exec, exec, s[18:19]
	s_cbranch_execz .LBB0_4
	s_branch .LBB0_65
.LBB0_35:                               ;   in Loop: Header=BB0_6 Depth=1
	v_cndmask_b32_e64 v8, 0, 1, s[44:45]
	v_cmp_ne_u32_e64 s[22:23], 1, v8
	s_and_saveexec_b64 s[18:19], s[8:9]
	s_cbranch_execz .LBB0_39
; %bb.36:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v8, s28, v9
	s_and_b64 vcc, exec, s[22:23]
	v_mov_b32_e32 v45, v43
	s_cbranch_vccnz .LBB0_38
; %bb.37:                               ;   in Loop: Header=BB0_6 Depth=1
	v_lshl_add_u64 v[10:11], v[44:45], 1, s[12:13]
	global_load_ushort v9, v[10:11], off
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v8, s29, v9, v8 op_sel_hi:[0,1,0]
.LBB0_38:                               ;   in Loop: Header=BB0_6 Depth=1
	v_cvt_f16_f32_e32 v10, v8
	v_lshl_add_u64 v[8:9], v[44:45], 1, s[14:15]
	global_store_short v[8:9], v10, off
.LBB0_39:                               ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[18:19]
	v_cndmask_b32_e64 v8, 0, 1, s[42:43]
	v_cmp_gt_u32_e64 s[18:19], 2, v1
	v_cmp_ne_u32_e64 s[20:21], 1, v8
	s_and_saveexec_b64 s[24:25], s[18:19]
	s_cbranch_execnz .LBB0_43
; %bb.40:                               ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[24:25]
	v_cmp_gt_u32_e64 s[24:25], 3, v1
	s_and_saveexec_b64 s[56:57], s[24:25]
	s_cbranch_execnz .LBB0_48
.LBB0_41:                               ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[56:57]
	s_and_saveexec_b64 s[56:57], s[8:9]
	s_cbranch_execnz .LBB0_51
.LBB0_42:                               ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[56:57]
	s_and_saveexec_b64 s[8:9], s[18:19]
	s_cbranch_execnz .LBB0_54
	s_branch .LBB0_59
.LBB0_43:                               ;   in Loop: Header=BB0_6 Depth=1
	s_and_b64 vcc, exec, s[20:21]
	v_mov_b32_e32 v45, v43
	s_cbranch_vccnz .LBB0_45
; %bb.44:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v7, s28, v7
	s_cbranch_execz .LBB0_46
	s_branch .LBB0_47
.LBB0_45:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v7, s28, v7
.LBB0_46:                               ;   in Loop: Header=BB0_6 Depth=1
	v_lshl_add_u64 v[8:9], v[44:45], 1, s[12:13]
	global_load_ushort v8, v[8:9], off offset:2
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v7, s29, v8, v7 op_sel_hi:[0,1,0]
.LBB0_47:                               ;   in Loop: Header=BB0_6 Depth=1
	v_cvt_f16_f32_e32 v7, v7
	v_lshl_add_u64 v[8:9], v[44:45], 1, s[14:15]
	global_store_short v[8:9], v7, off offset:2
	s_or_b64 exec, exec, s[24:25]
	v_cmp_gt_u32_e64 s[24:25], 3, v1
	s_and_saveexec_b64 s[56:57], s[24:25]
	s_cbranch_execz .LBB0_41
.LBB0_48:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v6, s28, v6
	s_and_b64 vcc, exec, s[22:23]
	v_mov_b32_e32 v47, v43
	s_cbranch_vccnz .LBB0_50
; %bb.49:                               ;   in Loop: Header=BB0_6 Depth=1
	v_lshl_add_u64 v[8:9], v[46:47], 1, s[12:13]
	global_load_ushort v7, v[8:9], off
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v6, s29, v7, v6 op_sel_hi:[0,1,0]
.LBB0_50:                               ;   in Loop: Header=BB0_6 Depth=1
	v_cvt_f16_f32_e32 v8, v6
	v_lshl_add_u64 v[6:7], v[46:47], 1, s[14:15]
	global_store_short v[6:7], v8, off
	s_or_b64 exec, exec, s[56:57]
	s_and_saveexec_b64 s[56:57], s[8:9]
	s_cbranch_execz .LBB0_42
.LBB0_51:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v5, s28, v5
	s_and_b64 vcc, exec, s[22:23]
	v_mov_b32_e32 v45, v43
	s_cbranch_vccnz .LBB0_53
; %bb.52:                               ;   in Loop: Header=BB0_6 Depth=1
	v_lshl_add_u64 v[6:7], v[44:45], 1, s[46:47]
	global_load_ushort v6, v[6:7], off
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v5, s29, v6, v5 op_sel_hi:[0,1,0]
.LBB0_53:                               ;   in Loop: Header=BB0_6 Depth=1
	v_cvt_f16_f32_e32 v5, v5
	v_lshl_add_u64 v[6:7], v[44:45], 1, s[48:49]
	global_store_short v[6:7], v5, off
	s_or_b64 exec, exec, s[56:57]
	s_and_saveexec_b64 s[8:9], s[18:19]
	s_cbranch_execz .LBB0_59
.LBB0_54:                               ;   in Loop: Header=BB0_6 Depth=1
	s_and_b64 vcc, exec, s[20:21]
	v_mov_b32_e32 v45, v43
	s_cbranch_vccnz .LBB0_56
; %bb.55:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v4, s28, v4
	s_cbranch_execz .LBB0_57
	s_branch .LBB0_58
.LBB0_56:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v4, s28, v4
.LBB0_57:                               ;   in Loop: Header=BB0_6 Depth=1
	v_lshl_add_u64 v[6:7], v[44:45], 1, s[46:47]
	global_load_ushort v5, v[6:7], off offset:2
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v4, s29, v5, v4 op_sel_hi:[0,1,0]
.LBB0_58:                               ;   in Loop: Header=BB0_6 Depth=1
	v_cvt_f16_f32_e32 v6, v4
	v_lshl_add_u64 v[4:5], v[44:45], 1, s[48:49]
	global_store_short v[4:5], v6, off offset:2
.LBB0_59:                               ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[8:9]
	s_mov_b64 s[18:19], 0
                                        ; implicit-def: $vgpr8
	s_and_saveexec_b64 s[8:9], s[24:25]
	s_xor_b64 s[8:9], exec, s[8:9]
	s_cbranch_execz .LBB0_63
; %bb.60:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v8, s28, v3
	s_and_b64 vcc, exec, s[22:23]
	v_mov_b32_e32 v47, v43
	s_cbranch_vccnz .LBB0_62
; %bb.61:                               ;   in Loop: Header=BB0_6 Depth=1
	v_lshl_add_u64 v[4:5], v[46:47], 1, s[46:47]
	global_load_ushort v3, v[4:5], off
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v8, s29, v3, v8 op_sel_hi:[0,1,0]
.LBB0_62:                               ;   in Loop: Header=BB0_6 Depth=1
	s_mov_b64 s[18:19], exec
.LBB0_63:                               ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[8:9]
	s_and_b64 s[18:19], s[18:19], exec
                                        ; implicit-def: $vgpr9
                                        ; implicit-def: $vgpr44
                                        ; implicit-def: $vgpr7
                                        ; implicit-def: $vgpr6
                                        ; implicit-def: $vgpr5
                                        ; implicit-def: $vgpr4
                                        ; implicit-def: $vgpr3
	s_andn2_saveexec_b64 s[8:9], s[54:55]
	s_cbranch_execz .LBB0_34
.LBB0_64:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mov_b32_e32 v45, v43
	v_mul_f32_e32 v12, s28, v9
	v_lshlrev_b64 v[8:9], 1, v[44:45]
	v_mul_f32_e32 v7, s28, v7
	v_lshl_add_u64 v[10:11], s[14:15], 0, v[8:9]
	v_cvt_pk_f16_f32 v7, v12, v7
	global_store_dword v[10:11], v7, off
	v_fma_mixlo_f16 v10, s28, v6, 0
	v_lshl_add_u64 v[6:7], v[46:47], 1, s[14:15]
	v_mul_f32_e32 v5, s28, v5
	v_mul_f32_e32 v4, s28, v4
	global_store_short v[6:7], v10, off
	v_lshl_add_u64 v[6:7], s[48:49], 0, v[8:9]
	v_cvt_pk_f16_f32 v4, v5, v4
	global_store_dword v[6:7], v4, off
	v_mul_f32_e32 v8, s28, v3
	s_or_b64 s[18:19], s[18:19], exec
	s_or_b64 exec, exec, s[8:9]
	s_and_b64 exec, exec, s[18:19]
	s_cbranch_execz .LBB0_4
.LBB0_65:                               ;   in Loop: Header=BB0_6 Depth=1
	v_cvt_f16_f32_e32 v3, v8
	v_lshl_add_u64 v[4:5], v[46:47], 1, s[48:49]
	global_store_short v[4:5], v3, off
	s_branch .LBB0_4
.LBB0_66:
	s_or_b64 exec, exec, s[34:35]
	s_mov_b64 s[0:1], 0
.LBB0_67:
	s_andn2_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB0_127
; %bb.68:
	s_lshl_b32 s33, s2, 1
	v_and_b32_e32 v8, 0x3ff, v0
	v_lshlrev_b32_e32 v50, 3, v8
	s_cmp_eq_u32 s26, s2
	s_mov_b64 s[8:9], 0
	s_cbranch_scc1 .LBB0_70
; %bb.69:
	s_and_b32 s18, s6, 15
	s_mov_b32 s19, 0
	s_mov_b64 s[0:1], -1
	s_min_u32 s22, s33, 0x14000
	v_lshl_add_u32 v9, v54, 9, v50
	s_cbranch_execz .LBB0_71
	s_branch .LBB0_72
.LBB0_70:
                                        ; implicit-def: $sgpr18_sgpr19
	s_mov_b64 s[0:1], 0
	s_min_u32 s22, s33, 0x14000
	v_lshl_add_u32 v9, v54, 9, v50
.LBB0_71:
	s_and_b32 s0, s2, 7
	s_cmp_lg_u32 s0, 0
	s_mov_b32 s19, 0
	s_cselect_b64 s[0:1], -1, 0
	s_and_b32 s18, s6, 15
	s_cmp_lg_u64 s[18:19], 0
	s_cselect_b64 s[8:9], -1, 0
	s_or_b64 s[0:1], s[0:1], s[8:9]
	s_mov_b64 s[8:9], -1
.LBB0_72:
	s_andn2_b64 vcc, exec, s[0:1]
	v_cmp_gt_u32_e64 s[0:1], s22, v9
	s_cbranch_vccz .LBB0_74
; %bb.73:
	s_andn2_b64 vcc, exec, s[8:9]
	s_cbranch_vccz .LBB0_82
	s_branch .LBB0_86
.LBB0_74:
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_81
; %bb.75:
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
	v_lshl_add_u32 v10, v54, 10, v0
	v_mov_b32_e32 v0, v9
	s_branch .LBB0_77
.LBB0_76:                               ;   in Loop: Header=BB0_77 Depth=1
	v_add_u32_e32 v0, 0x2000, v0
	v_cmp_le_u32_e32 vcc, s22, v0
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[4:5]
	s_or_b64 s[0:1], vcc, s[0:1]
	v_add_u32_e32 v10, 0x4000, v10
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB0_81
.LBB0_77:                               ; =>This Inner Loop Header: Depth=1
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
	s_cbranch_vccz .LBB0_79
; %bb.78:                               ;   in Loop: Header=BB0_77 Depth=1
	global_load_dwordx4 v[12:15], v[6:7], off
	s_mov_b64 s[20:21], 0
	s_waitcnt vmcnt(0)
	ds_write_b128 v10, v[12:15]
.LBB0_79:                               ;   in Loop: Header=BB0_77 Depth=1
	s_andn2_b64 vcc, exec, s[20:21]
	s_cbranch_vccnz .LBB0_76
; %bb.80:                               ;   in Loop: Header=BB0_77 Depth=1
	v_readfirstlane_b32 s20, v10
	s_mov_b32 m0, s20
	s_nop 0
	global_load_lds_dwordx4 v[6:7], off
	s_branch .LBB0_76
.LBB0_81:
	s_or_b64 exec, exec, s[8:9]
	s_cbranch_execnz .LBB0_86
.LBB0_82:
	v_cmp_gt_u32_e32 vcc, s22, v9
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_85
; %bb.83:
	v_lshlrev_b32_e32 v0, 4, v8
	v_lshl_add_u32 v2, v54, 10, v0
	v_lshlrev_b32_e32 v0, 1, v9
	v_mov_b32_e32 v1, 0
	v_lshl_add_u64 v[0:1], s[6:7], 0, v[0:1]
	s_mov_b64 s[6:7], 0
	s_mov_b64 s[8:9], 0x4000
.LBB0_84:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s18, v2
	s_mov_b32 m0, s18
	v_add_u32_e32 v9, 0x2000, v9
	global_load_lds_dwordx4 v[0:1], off
	v_cmp_le_u32_e32 vcc, s22, v9
	v_add_u32_e32 v2, 0x4000, v2
	s_or_b64 s[6:7], vcc, s[6:7]
	v_lshl_add_u64 v[0:1], v[0:1], 0, s[8:9]
	s_andn2_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB0_84
.LBB0_85:
	s_or_b64 exec, exec, s[0:1]
.LBB0_86:
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
	v_add_u32_e32 v0, s0, v54
	v_lshl_add_u32 v1, v0, 1, v0
	v_add_u32_e32 v0, 3, v1
	v_cmp_gt_u32_e32 vcc, s3, v1
	v_cmp_lt_u32_e64 s[0:1], s3, v0
	s_add_i32 s42, s3, -3
	v_mov_b32_e32 v2, s42
	s_and_b64 s[0:1], vcc, s[0:1]
	v_cndmask_b32_e64 v40, v1, v2, s[0:1]
	v_cmp_gt_u32_e64 s[0:1], s8, v54
	v_cmp_gt_u32_e64 s[6:7], s3, v40
	s_and_b64 s[0:1], s[0:1], s[6:7]
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_127
; %bb.87:
	s_cmp_lg_u32 s2, 0
	s_cselect_b64 s[6:7], -1, 0
	s_ashr_i32 s17, s30, 31
	s_mov_b32 s16, s30
	s_add_i32 s43, s2, -8
	s_add_i32 s44, s3, -1
	s_ashr_i32 s45, s10, 31
	s_ashr_i32 s19, s31, 31
	s_lshl_b64 s[16:17], s[16:17], 1
	s_mov_b32 s18, s31
	s_add_u32 s26, s12, s16
	v_sub_u32_e64 v0, v0, s3 clamp
	s_addc_u32 s27, s13, s17
	s_lshl_b64 s[16:17], s[18:19], 1
	v_cndmask_b32_e32 v51, 0, v0, vcc
	s_add_u32 s30, s14, s16
	v_cndmask_b32_e64 v0, 0, 1, s[6:7]
	v_cmp_eq_u32_e64 s[0:1], 63, v8
	v_cmp_eq_f32_e64 s[22:23], s29, 0
	v_cmp_neq_f32_e64 s[24:25], s29, 0
	s_addc_u32 s31, s15, s17
	s_mul_i32 s11, s8, s11
	v_lshlrev_b32_e32 v52, 4, v8
	s_mov_b64 s[34:35], 0
	v_cmp_ne_u32_e64 s[20:21], 1, v0
	v_mov_b32_e32 v43, 0
	s_branch .LBB0_89
.LBB0_88:                               ;   in Loop: Header=BB0_89 Depth=1
	s_or_b64 exec, exec, s[36:37]
	v_add_u32_e32 v0, s11, v40
	v_add_u32_e32 v1, 3, v0
	v_cmp_gt_u32_e32 vcc, s3, v0
	v_cmp_lt_u32_e64 s[6:7], s3, v1
	v_mov_b32_e32 v2, s42
	s_and_b64 vcc, vcc, s[6:7]
	v_cndmask_b32_e32 v40, v0, v2, vcc
	v_subrev_u32_e32 v1, s3, v1
	v_cmp_le_u32_e64 s[6:7], s3, v40
	s_or_b64 s[34:35], s[6:7], s[34:35]
	v_cndmask_b32_e32 v51, v51, v1, vcc
	s_andn2_b64 exec, exec, s[34:35]
	s_cbranch_execz .LBB0_127
.LBB0_89:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_93 Depth 2
	s_and_b64 vcc, exec, s[20:21]
	v_mov_b32_e32 v57, v43
	v_mov_b32_e32 v56, v43
	v_mov_b32_e32 v55, v43
	v_mov_b32_e32 v54, v43
	v_mov_b32_e32 v53, v43
	v_mov_b32_e32 v41, v43
	s_cbranch_vccnz .LBB0_96
; %bb.90:                               ;   in Loop: Header=BB0_89 Depth=1
	v_min_u32_e32 v3, s44, v40
	v_mad_u64_u32 v[0:1], s[6:7], v3, s10, 0
	v_mov_b32_e32 v2, v1
	v_mad_u64_u32 v[2:3], s[6:7], v3, s45, v[2:3]
	v_mov_b32_e32 v1, v2
	v_add_u32_e32 v2, 1, v40
	v_min_u32_e32 v5, s44, v2
	v_mad_u64_u32 v[2:3], s[6:7], v5, s10, 0
	v_mov_b32_e32 v4, v3
	v_mad_u64_u32 v[4:5], s[6:7], v5, s45, v[4:5]
	v_mov_b32_e32 v3, v4
	v_add_u32_e32 v4, 2, v40
	v_min_u32_e32 v7, s44, v4
	v_mad_u64_u32 v[4:5], s[6:7], v7, s10, 0
	v_mov_b32_e32 v6, v5
	v_mad_u64_u32 v[6:7], s[6:7], v7, s45, v[6:7]
	v_mov_b32_e32 v5, v6
	s_mov_b32 s16, 0
	v_mov_b32_e32 v41, 0
	v_lshlrev_b64 v[44:45], 1, v[0:1]
	v_lshlrev_b64 v[46:47], 1, v[2:3]
	v_lshlrev_b64 v[48:49], 1, v[4:5]
	v_mov_b32_e32 v58, v52
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v56, 0
	v_mov_b32_e32 v57, 0
	s_branch .LBB0_93
.LBB0_91:                               ;   in Loop: Header=BB0_93 Depth=2
	s_or_b64 exec, exec, s[8:9]
.LBB0_92:                               ;   in Loop: Header=BB0_93 Depth=2
	s_or_b64 exec, exec, s[6:7]
	s_waitcnt vmcnt(5) lgkmcnt(1)
	;;#ASMSTART
	v_dot2c_f32_f16 v57, v36, v28
	;;#ASMEND
	s_waitcnt vmcnt(4)
	;;#ASMSTART
	v_dot2c_f32_f16 v56, v36, v24
	;;#ASMEND
	s_waitcnt vmcnt(3)
	;;#ASMSTART
	v_dot2c_f32_f16 v55, v36, v20
	;;#ASMEND
	s_waitcnt lgkmcnt(0)
	;;#ASMSTART
	v_dot2c_f32_f16 v54, v32, v28
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v53, v32, v24
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v41, v32, v20
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v57, v37, v29
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v56, v37, v25
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v55, v37, v21
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v54, v33, v29
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v53, v33, v25
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v41, v33, v21
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v57, v38, v30
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v56, v38, v26
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v55, v38, v22
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v54, v34, v30
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v53, v34, v26
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v41, v34, v22
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v57, v39, v31
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v56, v39, v27
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v55, v39, v23
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v54, v35, v31
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v53, v35, v27
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v41, v35, v23
	;;#ASMEND
	s_waitcnt vmcnt(2)
	;;#ASMSTART
	v_dot2c_f32_f16 v57, v16, v8
	;;#ASMEND
	s_waitcnt vmcnt(1)
	;;#ASMSTART
	v_dot2c_f32_f16 v56, v16, v4
	;;#ASMEND
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	v_dot2c_f32_f16 v55, v16, v0
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v54, v12, v8
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v53, v12, v4
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v41, v12, v0
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v57, v17, v9
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v56, v17, v5
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v55, v17, v1
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v54, v13, v9
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v53, v13, v5
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v41, v13, v1
	;;#ASMEND
	s_addk_i32 s16, 0x400
	;;#ASMSTART
	v_dot2c_f32_f16 v57, v18, v10
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v56, v18, v6
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v55, v18, v2
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v54, v14, v10
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v53, v14, v6
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v41, v14, v2
	;;#ASMEND
	s_cmp_ge_u32 s16, s2
	v_add_u32_e32 v58, 0x800, v58
	;;#ASMSTART
	v_dot2c_f32_f16 v57, v19, v11
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v56, v19, v7
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v55, v19, v3
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v54, v15, v11
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v53, v15, v7
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v41, v15, v3
	;;#ASMEND
	s_cbranch_scc1 .LBB0_96
.LBB0_93:                               ;   Parent Loop BB0_89 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_add_u32_e32 v16, s16, v50
	v_min_u32_e32 v42, s43, v16
	v_lshl_add_u64 v[0:1], v[42:43], 1, s[4:5]
	v_add_u32_e32 v59, 0x200, v16
	v_lshl_add_u64 v[2:3], v[0:1], 0, v[44:45]
	v_min_u32_e32 v42, s43, v59
	v_lshl_add_u64 v[4:5], v[0:1], 0, v[46:47]
	global_load_dwordx4 v[28:31], v[2:3], off nt
	global_load_dwordx4 v[24:27], v[4:5], off nt
	v_lshl_add_u64 v[2:3], v[42:43], 1, s[4:5]
	v_lshl_add_u64 v[0:1], v[0:1], 0, v[48:49]
	v_lshl_add_u64 v[4:5], v[2:3], 0, v[44:45]
	v_lshl_add_u64 v[12:13], v[2:3], 0, v[46:47]
	global_load_dwordx4 v[20:23], v[0:1], off nt
	global_load_dwordx4 v[8:11], v[4:5], off nt
	v_lshl_add_u64 v[14:15], v[2:3], 0, v[48:49]
	global_load_dwordx4 v[4:7], v[12:13], off nt
	global_load_dwordx4 v[0:3], v[14:15], off nt
	v_cmp_gt_u32_e32 vcc, s2, v16
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v39, 0
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v16, 0
	v_mov_b32_e32 v17, 0
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v12, 0
	v_mov_b32_e32 v13, 0
	v_mov_b32_e32 v14, 0
	v_mov_b32_e32 v15, 0
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_92
; %bb.94:                               ;   in Loop: Header=BB0_93 Depth=2
	v_add_u32_e32 v42, s33, v58
	ds_read_b128 v[36:39], v58
	ds_read_b128 v[32:35], v42
	v_cmp_gt_u32_e32 vcc, s2, v59
	v_mov_b32_e32 v15, 0
	v_mov_b32_e32 v14, 0
	v_mov_b32_e32 v13, 0
	v_mov_b32_e32 v12, 0
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v17, 0
	v_mov_b32_e32 v16, 0
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_91
; %bb.95:                               ;   in Loop: Header=BB0_93 Depth=2
	ds_read_b128 v[16:19], v58 offset:1024
	ds_read_b128 v[12:15], v42 offset:1024
	s_branch .LBB0_91
.LBB0_96:                               ;   in Loop: Header=BB0_89 Depth=1
	; sched_barrier mask(0x00000000)
	s_nop 0
	v_add_f32_dpp v4, v55, v55 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v57, v57 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v56, v56 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v0, v0 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v0, v0 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v0, v0 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v4, v4 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v54, v54 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v0, v0 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v1, v0 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v3, v2 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v7, v5 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_add_f32_dpp v4, v4, v4 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v8, v4, v4 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v53, v53 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v9, v8 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v4, v4, v4 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v4, v4, v4 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v10, v4, v4 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v41, v41 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v11, v10 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v4, v4, v4 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v4, v4, v4 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v12, v4, v4 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_mov_b32_dpp v13, v12 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_and_saveexec_b64 s[36:37], s[0:1]
	s_cbranch_execz .LBB0_88
; %bb.97:                               ;   in Loop: Header=BB0_89 Depth=1
	v_cmp_eq_u32_e64 s[8:9], 0, v51
	s_and_b64 s[6:7], s[22:23], s[8:9]
	v_add_f32_e32 v6, v0, v1
	v_add_f32_e32 v4, v2, v3
	v_add_f32_e32 v3, v5, v7
	v_add_f32_e32 v2, v8, v9
	v_add_f32_e32 v1, v10, v11
	v_add_f32_e32 v0, v12, v13
	s_xor_b64 s[16:17], s[6:7], -1
	s_mov_b64 s[6:7], 0
                                        ; implicit-def: $vgpr5
	s_and_saveexec_b64 s[18:19], s[16:17]
	s_xor_b64 s[38:39], exec, s[18:19]
	s_cbranch_execnz .LBB0_100
; %bb.98:                               ;   in Loop: Header=BB0_89 Depth=1
	s_andn2_saveexec_b64 s[8:9], s[38:39]
	s_cbranch_execnz .LBB0_125
.LBB0_99:                               ;   in Loop: Header=BB0_89 Depth=1
	s_or_b64 exec, exec, s[8:9]
	s_and_b64 exec, exec, s[6:7]
	s_cbranch_execz .LBB0_88
	s_branch .LBB0_126
.LBB0_100:                              ;   in Loop: Header=BB0_89 Depth=1
	v_cndmask_b32_e64 v5, 0, 1, s[24:25]
	v_cmp_ne_u32_e64 s[6:7], 1, v5
	s_and_saveexec_b64 s[16:17], s[8:9]
	s_cbranch_execnz .LBB0_105
; %bb.101:                              ;   in Loop: Header=BB0_89 Depth=1
	s_or_b64 exec, exec, s[16:17]
	v_cmp_gt_u32_e64 s[16:17], 2, v51
	s_and_saveexec_b64 s[18:19], s[16:17]
	s_cbranch_execnz .LBB0_108
.LBB0_102:                              ;   in Loop: Header=BB0_89 Depth=1
	s_or_b64 exec, exec, s[18:19]
	v_cmp_gt_u32_e64 s[18:19], 3, v51
	s_and_saveexec_b64 s[40:41], s[18:19]
	s_cbranch_execnz .LBB0_111
.LBB0_103:                              ;   in Loop: Header=BB0_89 Depth=1
	s_or_b64 exec, exec, s[40:41]
	s_and_saveexec_b64 s[40:41], s[8:9]
	s_cbranch_execnz .LBB0_114
.LBB0_104:                              ;   in Loop: Header=BB0_89 Depth=1
	s_or_b64 exec, exec, s[40:41]
	s_and_saveexec_b64 s[8:9], s[16:17]
	s_cbranch_execnz .LBB0_117
	s_branch .LBB0_120
.LBB0_105:                              ;   in Loop: Header=BB0_89 Depth=1
	v_mul_f32_e32 v5, s28, v6
	s_and_b64 vcc, exec, s[6:7]
	v_mov_b32_e32 v41, v43
	s_cbranch_vccnz .LBB0_107
; %bb.106:                              ;   in Loop: Header=BB0_89 Depth=1
	v_lshl_add_u64 v[6:7], v[40:41], 1, s[12:13]
	global_load_ushort v6, v[6:7], off
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v5, s29, v6, v5 op_sel_hi:[0,1,0]
.LBB0_107:                              ;   in Loop: Header=BB0_89 Depth=1
	v_cvt_f16_f32_e32 v5, v5
	v_lshl_add_u64 v[6:7], v[40:41], 1, s[14:15]
	global_store_short v[6:7], v5, off
	s_or_b64 exec, exec, s[16:17]
	v_cmp_gt_u32_e64 s[16:17], 2, v51
	s_and_saveexec_b64 s[18:19], s[16:17]
	s_cbranch_execz .LBB0_102
.LBB0_108:                              ;   in Loop: Header=BB0_89 Depth=1
	v_mul_f32_e32 v4, s28, v4
	s_and_b64 vcc, exec, s[6:7]
	v_add_u32_e32 v42, 1, v40
	s_cbranch_vccnz .LBB0_110
; %bb.109:                              ;   in Loop: Header=BB0_89 Depth=1
	v_lshl_add_u64 v[6:7], v[42:43], 1, s[12:13]
	global_load_ushort v5, v[6:7], off
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v4, s29, v5, v4 op_sel_hi:[0,1,0]
.LBB0_110:                              ;   in Loop: Header=BB0_89 Depth=1
	v_cvt_f16_f32_e32 v6, v4
	v_lshl_add_u64 v[4:5], v[42:43], 1, s[14:15]
	global_store_short v[4:5], v6, off
	s_or_b64 exec, exec, s[18:19]
	v_cmp_gt_u32_e64 s[18:19], 3, v51
	s_and_saveexec_b64 s[40:41], s[18:19]
	s_cbranch_execz .LBB0_103
.LBB0_111:                              ;   in Loop: Header=BB0_89 Depth=1
	v_mul_f32_e32 v3, s28, v3
	s_and_b64 vcc, exec, s[6:7]
	v_add_u32_e32 v42, 2, v40
	s_cbranch_vccnz .LBB0_113
; %bb.112:                              ;   in Loop: Header=BB0_89 Depth=1
	v_lshl_add_u64 v[4:5], v[42:43], 1, s[12:13]
	global_load_ushort v4, v[4:5], off
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v3, s29, v4, v3 op_sel_hi:[0,1,0]
.LBB0_113:                              ;   in Loop: Header=BB0_89 Depth=1
	v_cvt_f16_f32_e32 v3, v3
	v_lshl_add_u64 v[4:5], v[42:43], 1, s[14:15]
	global_store_short v[4:5], v3, off
	s_or_b64 exec, exec, s[40:41]
	s_and_saveexec_b64 s[40:41], s[8:9]
	s_cbranch_execz .LBB0_104
.LBB0_114:                              ;   in Loop: Header=BB0_89 Depth=1
	v_mul_f32_e32 v2, s28, v2
	s_and_b64 vcc, exec, s[6:7]
	v_mov_b32_e32 v41, v43
	s_cbranch_vccnz .LBB0_116
; %bb.115:                              ;   in Loop: Header=BB0_89 Depth=1
	v_lshl_add_u64 v[4:5], v[40:41], 1, s[26:27]
	global_load_ushort v3, v[4:5], off
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v2, s29, v3, v2 op_sel_hi:[0,1,0]
.LBB0_116:                              ;   in Loop: Header=BB0_89 Depth=1
	v_cvt_f16_f32_e32 v4, v2
	v_lshl_add_u64 v[2:3], v[40:41], 1, s[30:31]
	global_store_short v[2:3], v4, off
	s_or_b64 exec, exec, s[40:41]
	s_and_saveexec_b64 s[8:9], s[16:17]
	s_cbranch_execz .LBB0_120
.LBB0_117:                              ;   in Loop: Header=BB0_89 Depth=1
	v_mul_f32_e32 v1, s28, v1
	s_and_b64 vcc, exec, s[6:7]
	v_add_u32_e32 v42, 1, v40
	s_cbranch_vccnz .LBB0_119
; %bb.118:                              ;   in Loop: Header=BB0_89 Depth=1
	v_lshl_add_u64 v[2:3], v[42:43], 1, s[26:27]
	global_load_ushort v2, v[2:3], off
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v1, s29, v2, v1 op_sel_hi:[0,1,0]
.LBB0_119:                              ;   in Loop: Header=BB0_89 Depth=1
	v_cvt_f16_f32_e32 v1, v1
	v_lshl_add_u64 v[2:3], v[42:43], 1, s[30:31]
	global_store_short v[2:3], v1, off
.LBB0_120:                              ;   in Loop: Header=BB0_89 Depth=1
	s_or_b64 exec, exec, s[8:9]
	s_mov_b64 s[16:17], 0
                                        ; implicit-def: $vgpr5
	s_and_saveexec_b64 s[8:9], s[18:19]
	s_xor_b64 s[8:9], exec, s[8:9]
	s_cbranch_execz .LBB0_124
; %bb.121:                              ;   in Loop: Header=BB0_89 Depth=1
	v_mul_f32_e32 v5, s28, v0
	s_and_b64 vcc, exec, s[6:7]
	v_add_u32_e32 v42, 2, v40
	s_cbranch_vccnz .LBB0_123
; %bb.122:                              ;   in Loop: Header=BB0_89 Depth=1
	v_lshl_add_u64 v[0:1], v[42:43], 1, s[26:27]
	global_load_ushort v0, v[0:1], off
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v5, s29, v0, v5 op_sel_hi:[0,1,0]
.LBB0_123:                              ;   in Loop: Header=BB0_89 Depth=1
	s_mov_b64 s[16:17], exec
.LBB0_124:                              ;   in Loop: Header=BB0_89 Depth=1
	s_or_b64 exec, exec, s[8:9]
	s_and_b64 s[6:7], s[16:17], exec
                                        ; implicit-def: $vgpr6
                                        ; implicit-def: $vgpr4
                                        ; implicit-def: $vgpr3
                                        ; implicit-def: $vgpr2
                                        ; implicit-def: $vgpr1
                                        ; implicit-def: $vgpr0
	s_andn2_saveexec_b64 s[8:9], s[38:39]
	s_cbranch_execz .LBB0_99
.LBB0_125:                              ;   in Loop: Header=BB0_89 Depth=1
	v_mov_b32_e32 v41, v43
	v_fma_mixlo_f16 v5, s28, v6, 0
	v_lshlrev_b64 v[6:7], 1, v[40:41]
	v_lshl_add_u64 v[8:9], s[14:15], 0, v[6:7]
	v_add_u32_e32 v42, 1, v40
	global_store_short v[8:9], v5, off
	v_fma_mixlo_f16 v10, s28, v4, 0
	v_lshlrev_b64 v[4:5], 1, v[42:43]
	v_lshl_add_u64 v[8:9], s[14:15], 0, v[4:5]
	v_add_u32_e32 v42, 2, v40
	global_store_short v[8:9], v10, off
	v_fma_mixlo_f16 v3, s28, v3, 0
	v_lshl_add_u64 v[8:9], v[42:43], 1, s[14:15]
	global_store_short v[8:9], v3, off
	v_fma_mixlo_f16 v8, s28, v2, 0
	v_lshl_add_u64 v[2:3], s[30:31], 0, v[6:7]
	global_store_short v[2:3], v8, off
	v_fma_mixlo_f16 v1, s28, v1, 0
	v_lshl_add_u64 v[2:3], s[30:31], 0, v[4:5]
	global_store_short v[2:3], v1, off
	v_mul_f32_e32 v5, s28, v0
	s_or_b64 s[6:7], s[6:7], exec
	s_or_b64 exec, exec, s[8:9]
	s_and_b64 exec, exec, s[6:7]
	s_cbranch_execz .LBB0_88
.LBB0_126:                              ;   in Loop: Header=BB0_89 Depth=1
	v_cvt_f16_f32_e32 v2, v5
	v_lshl_add_u64 v[0:1], v[42:43], 1, s[30:31]
	global_store_short v[0:1], v2, off
	s_branch .LBB0_88
.LBB0_127:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wvSplitK_vllm_f16_tn_n2_y3u2
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
	.size	wvSplitK_vllm_f16_tn_n2_y3u2, .Lfunc_end0-wvSplitK_vllm_f16_tn_n2_y3u2
                                        ; -- End function
	.set wvSplitK_vllm_f16_tn_n2_y3u2.num_vgpr, 64
	.set wvSplitK_vllm_f16_tn_n2_y3u2.num_agpr, 0
	.set wvSplitK_vllm_f16_tn_n2_y3u2.numbered_sgpr, 65
	.set wvSplitK_vllm_f16_tn_n2_y3u2.num_named_barrier, 0
	.set wvSplitK_vllm_f16_tn_n2_y3u2.private_seg_size, 0
	.set wvSplitK_vllm_f16_tn_n2_y3u2.uses_vcc, 1
	.set wvSplitK_vllm_f16_tn_n2_y3u2.uses_flat_scratch, 0
	.set wvSplitK_vllm_f16_tn_n2_y3u2.has_dyn_sized_stack, 0
	.set wvSplitK_vllm_f16_tn_n2_y3u2.has_recursion, 0
	.set wvSplitK_vllm_f16_tn_n2_y3u2.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 7468
; TotalNumSgprs: 71
; NumVgprs: 64
; NumAgprs: 0
; TotalNumVgprs: 64
; ScratchSize: 0
; MemoryBound: 0
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
	.type	__hip_cuid_ebc22cd3fb672b3d,@object ; @__hip_cuid_ebc22cd3fb672b3d
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_ebc22cd3fb672b3d
__hip_cuid_ebc22cd3fb672b3d:
	.byte	0                               ; 0x0
	.size	__hip_cuid_ebc22cd3fb672b3d, 1

	.ident	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.4 26084 f58b06dce1f9c15707c5f808fd002e18c2accf7e)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_ebc22cd3fb672b3d
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
    .name:           wvSplitK_vllm_f16_tn_n2_y3u2
    .private_segment_fixed_size: 0
    .sgpr_count:     71
    .sgpr_spill_count: 0
    .symbol:         wvSplitK_vllm_f16_tn_n2_y3u2.kd
    .vgpr_count:     64
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 1
...

	.end_amdgpu_metadata
