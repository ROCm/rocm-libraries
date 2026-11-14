// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Generated from the .hip beside it with amdclang++ -x hip --offload-arch=gfx950
// --cuda-device-only -O3 -S -mcode-object-version=4, then custom.config added.
	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 4
	.text
	.protected	c_hip_sk_parallel_Cijk_Ailk_Bjlk_HHS_BH_MT64x64x32_gfx950 ; -- Begin function c_hip_sk_parallel_Cijk_Ailk_Bjlk_HHS_BH_MT64x64x32_gfx950
	.globl	c_hip_sk_parallel_Cijk_Ailk_Bjlk_HHS_BH_MT64x64x32_gfx950
	.p2align	8
	.type	c_hip_sk_parallel_Cijk_Ailk_Bjlk_HHS_BH_MT64x64x32_gfx950,@function
c_hip_sk_parallel_Cijk_Ailk_Bjlk_HHS_BH_MT64x64x32_gfx950: ; @c_hip_sk_parallel_Cijk_Ailk_Bjlk_HHS_BH_MT64x64x32_gfx950
; %bb.0:
	s_load_dwordx16 s[36:51], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	v_cvt_f32_u32_e32 v1, s49
	s_sub_i32 s3, 0, s49
	v_rcp_iflag_f32_e32 v1, v1
	s_nop 0
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	s_nop 0
	v_readfirstlane_b32 s4, v1
	s_mul_i32 s3, s3, s4
	s_mul_hi_u32 s3, s4, s3
	s_add_i32 s4, s4, s3
	s_mul_hi_u32 s3, s2, s4
	s_mul_i32 s5, s3, s49
	s_sub_i32 s5, s2, s5
	s_add_i32 s6, s3, 1
	s_sub_i32 s7, s5, s49
	s_cmp_ge_u32 s5, s49
	s_cselect_b32 s3, s6, s3
	s_cselect_b32 s5, s7, s5
	s_add_i32 s6, s3, 1
	s_cmp_ge_u32 s5, s49
	s_cselect_b32 s5, s6, s3
	s_cmp_ge_u32 s5, s50
	s_mov_b32 s3, 0
	s_cbranch_scc1 .LBB0_51
; %bb.1:
	s_add_i32 s6, s36, 63
	s_add_i32 s7, s37, 63
	s_lshr_b32 s6, s6, 6
	s_lshr_b32 s7, s7, 6
	s_mul_i32 s7, s7, s6
	v_cvt_f32_u32_e32 v1, s7
	s_load_dwordx8 s[52:59], s[0:1], 0x0
	v_cvt_f32_u32_e32 v2, s6
	s_load_dwordx2 s[60:61], s[0:1], 0x20
	v_rcp_iflag_f32_e32 v1, v1
	s_sub_i32 s0, 0, s7
	v_rcp_iflag_f32_e32 v2, v2
	v_and_b32_e32 v24, 63, v0
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	v_lshrrev_b32_e32 v19, 2, v0
	v_readfirstlane_b32 s1, v1
	s_mul_i32 s0, s0, s1
	s_mul_hi_u32 s0, s1, s0
	s_add_i32 s1, s1, s0
	s_mul_hi_u32 s0, s5, s1
	s_mul_i32 s1, s0, s7
	s_sub_i32 s1, s5, s1
	s_add_i32 s8, s0, 1
	s_sub_i32 s9, s1, s7
	s_cmp_ge_u32 s1, s7
	v_mul_f32_e32 v1, 0x4f7ffffe, v2
	s_cselect_b32 s0, s8, s0
	v_cvt_u32_f32_e32 v1, v1
	s_cselect_b32 s1, s9, s1
	s_add_i32 s8, s0, 1
	s_cmp_ge_u32 s1, s7
	s_cselect_b32 s33, s8, s0
	s_mul_i32 s0, s33, s7
	s_sub_i32 s1, 0, s6
	v_readfirstlane_b32 s7, v1
	s_mul_i32 s1, s1, s7
	s_mul_hi_u32 s1, s7, s1
	s_sub_i32 s0, s5, s0
	s_add_i32 s7, s7, s1
	s_mul_hi_u32 s1, s0, s7
	s_mul_i32 s7, s1, s6
	s_sub_i32 s7, s0, s7
	s_add_i32 s8, s1, 1
	s_sub_i32 s9, s7, s6
	s_cmp_ge_u32 s7, s6
	s_cselect_b32 s1, s8, s1
	s_cselect_b32 s7, s9, s7
	s_add_i32 s8, s1, 1
	s_cmp_ge_u32 s7, s6
	s_cselect_b32 s1, s8, s1
	s_mul_i32 s6, s1, s6
	s_sub_i32 s0, s0, s6
	s_lshl_b32 s66, s0, 6
	v_or_b32_e32 v18, s66, v24
	v_mov_b32_e32 v2, 0
	s_lshl_b32 s62, s1, 6
	v_and_b32_e32 v1, 48, v19
	v_cmp_gt_u32_e64 s[0:1], s36, v18
	v_mov_b32_e32 v3, v2
	v_mov_b32_e32 v4, v2
	v_mov_b32_e32 v5, v2
	v_mov_b32_e32 v6, v2
	v_mov_b32_e32 v7, v2
	v_mov_b32_e32 v8, v2
	v_mov_b32_e32 v9, v2
	v_mov_b32_e32 v10, v2
	v_mov_b32_e32 v11, v2
	v_mov_b32_e32 v12, v2
	v_mov_b32_e32 v13, v2
	v_mov_b32_e32 v14, v2
	v_mov_b32_e32 v15, v2
	v_mov_b32_e32 v16, v2
	v_mov_b32_e32 v17, v2
	s_and_saveexec_b64 s[64:65], s[0:1]
	s_cbranch_execz .LBB0_38
; %bb.2:
	s_mul_i32 s5, s5, s49
	s_sub_i32 s5, s2, s5
	s_add_i32 s6, s38, 31
	s_lshr_b32 s6, s6, 5
	s_add_i32 s7, s5, 1
	s_mul_i32 s7, s7, s6
	s_mul_hi_u32 s8, s7, s4
	s_mul_i32 s9, s8, s49
	s_sub_i32 s7, s7, s9
	s_add_i32 s9, s8, 1
	s_sub_i32 s10, s7, s49
	s_cmp_ge_u32 s7, s49
	s_cselect_b32 s8, s9, s8
	s_cselect_b32 s7, s10, s7
	s_add_i32 s9, s8, 1
	s_mul_i32 s5, s5, s6
	s_cmp_ge_u32 s7, s49
	s_mul_hi_u32 s4, s5, s4
	s_cselect_b32 s7, s9, s8
	s_mul_i32 s6, s4, s49
	s_lshl_b32 s7, s7, 5
	s_sub_i32 s5, s5, s6
	s_min_u32 s36, s7, s38
	s_add_i32 s6, s4, 1
	s_sub_i32 s7, s5, s49
	s_cmp_ge_u32 s5, s49
	s_cselect_b32 s4, s6, s4
	s_cselect_b32 s5, s7, s5
	s_add_i32 s6, s4, 1
	s_cmp_ge_u32 s5, s49
	s_cselect_b32 s4, s6, s4
	s_lshl_b32 s50, s4, 5
	s_cmp_ge_u32 s50, s36
	s_mov_b32 s67, 0
	s_cbranch_scc1 .LBB0_37
; %bb.3:
	v_add_u32_e32 v2, s62, v1
	v_or_b32_e32 v3, 1, v2
	v_cmp_gt_u32_e64 s[4:5], s37, v3
	v_or_b32_e32 v3, 2, v2
	v_cmp_gt_u32_e64 s[6:7], s37, v3
	v_or_b32_e32 v3, 3, v2
	v_cmp_gt_u32_e64 s[8:9], s37, v3
	v_or_b32_e32 v3, 4, v2
	v_cmp_gt_u32_e64 s[10:11], s37, v3
	v_or_b32_e32 v3, 5, v2
	v_cmp_gt_u32_e64 s[12:13], s37, v3
	v_or_b32_e32 v3, 6, v2
	v_cmp_gt_u32_e64 s[14:15], s37, v3
	v_or_b32_e32 v3, 7, v2
	v_cmp_gt_u32_e64 s[16:17], s37, v3
	v_or_b32_e32 v3, 8, v2
	s_mul_hi_u32 s73, s33, s40
	s_mul_i32 s72, s33, s40
	s_mul_hi_u32 s75, s39, s50
	s_mul_i32 s74, s39, s50
	s_mov_b32 s68, s39
	v_cmp_gt_u32_e64 s[18:19], s37, v3
	v_or_b32_e32 v3, 9, v2
	s_lshl_b64 s[72:73], s[72:73], 1
	s_lshl_b64 s[38:39], s[74:75], 1
	v_cmp_gt_u32_e64 s[20:21], s37, v3
	v_or_b32_e32 v3, 10, v2
	s_add_u32 s40, s72, s38
	v_cmp_gt_u32_e64 s[22:23], s37, v3
	v_or_b32_e32 v3, 11, v2
	s_addc_u32 s72, s73, s39
	s_lshl_b64 s[38:39], s[66:67], 1
	v_cmp_gt_u32_e64 s[24:25], s37, v3
	v_or_b32_e32 v3, 12, v2
	s_add_u32 s38, s40, s38
	v_cmp_gt_u32_e64 s[26:27], s37, v3
	v_or_b32_e32 v3, 13, v2
	s_addc_u32 s39, s72, s39
	v_cmp_gt_u32_e32 vcc, s37, v2
	v_cmp_gt_u32_e64 s[28:29], s37, v3
	v_or_b32_e32 v3, 14, v2
	v_or_b32_e32 v2, 15, v2
	s_waitcnt lgkmcnt(0)
	s_add_u32 s38, s56, s38
	s_mov_b32 s63, s67
	s_mov_b32 s69, s67
	s_mov_b32 s71, s67
	v_cmp_gt_u32_e64 s[30:31], s37, v3
	v_cmp_gt_u32_e64 s[34:35], s37, v2
	v_lshlrev_b32_e32 v2, 1, v24
	v_mov_b32_e32 v3, 0
	s_addc_u32 s39, s57, s39
	s_mul_hi_u32 s57, s33, s42
	s_mul_i32 s56, s33, s42
	s_mul_hi_u32 s67, s41, s50
	s_mul_i32 s66, s41, s50
	s_mov_b32 s70, s41
	v_lshl_add_u64 v[20:21], s[38:39], 0, v[2:3]
	s_lshl_b64 s[38:39], s[68:69], 1
	s_lshl_b64 s[56:57], s[56:57], 1
	s_lshl_b64 s[40:41], s[66:67], 1
	s_add_u32 s42, s56, s40
	s_addc_u32 s56, s57, s41
	s_lshl_b64 s[40:41], s[62:63], 1
	s_add_u32 s40, s58, s40
	s_addc_u32 s41, s59, s41
	v_lshrrev_b32_e32 v2, 1, v0
	s_add_u32 s40, s40, s42
	v_and_b32_e32 v2, 0x60, v2
	s_addc_u32 s41, s41, s56
	v_lshl_add_u64 v[4:5], s[40:41], 0, v[2:3]
	v_lshl_add_u64 v[22:23], v[4:5], 0, 30
	s_lshl_b64 s[40:41], s[70:71], 1
	v_mov_b32_e32 v2, v3
	v_mov_b32_e32 v4, v3
	v_mov_b32_e32 v5, v3
	v_mov_b32_e32 v6, v3
	v_mov_b32_e32 v7, v3
	v_mov_b32_e32 v8, v3
	v_mov_b32_e32 v9, v3
	v_mov_b32_e32 v10, v3
	v_mov_b32_e32 v11, v3
	v_mov_b32_e32 v12, v3
	v_mov_b32_e32 v13, v3
	v_mov_b32_e32 v14, v3
	v_mov_b32_e32 v15, v3
	v_mov_b32_e32 v16, v3
	v_mov_b32_e32 v17, v3
	s_branch .LBB0_5
	.p2align	5
.LBB0_4:                                ;   in Loop: Header=BB0_5 Depth=1
	s_or_b64 exec, exec, s[56:57]
	s_add_i32 s50, s50, 1
	v_lshl_add_u64 v[20:21], v[20:21], 0, s[38:39]
	s_cmp_lt_u32 s50, s36
	v_lshl_add_u64 v[22:23], v[22:23], 0, s[40:41]
	s_cbranch_scc0 .LBB0_38
.LBB0_5:                                ; =>This Inner Loop Header: Depth=1
	global_load_ushort v25, v[20:21], off
	s_waitcnt vmcnt(0)
	v_cvt_f32_f16_e32 v25, v25
	s_and_saveexec_b64 s[56:57], vcc
	s_cbranch_execnz .LBB0_21
; %bb.6:                                ;   in Loop: Header=BB0_5 Depth=1
	s_or_b64 exec, exec, s[56:57]
	s_and_saveexec_b64 s[56:57], s[4:5]
	s_cbranch_execnz .LBB0_22
.LBB0_7:                                ;   in Loop: Header=BB0_5 Depth=1
	s_or_b64 exec, exec, s[56:57]
	s_and_saveexec_b64 s[56:57], s[6:7]
	s_cbranch_execnz .LBB0_23
.LBB0_8:                                ;   in Loop: Header=BB0_5 Depth=1
	s_or_b64 exec, exec, s[56:57]
	s_and_saveexec_b64 s[56:57], s[8:9]
	s_cbranch_execnz .LBB0_24
.LBB0_9:                                ;   in Loop: Header=BB0_5 Depth=1
	s_or_b64 exec, exec, s[56:57]
	s_and_saveexec_b64 s[56:57], s[10:11]
	s_cbranch_execnz .LBB0_25
.LBB0_10:                               ;   in Loop: Header=BB0_5 Depth=1
	s_or_b64 exec, exec, s[56:57]
	s_and_saveexec_b64 s[56:57], s[12:13]
	s_cbranch_execnz .LBB0_26
.LBB0_11:                               ;   in Loop: Header=BB0_5 Depth=1
	s_or_b64 exec, exec, s[56:57]
	s_and_saveexec_b64 s[56:57], s[14:15]
	s_cbranch_execnz .LBB0_27
.LBB0_12:                               ;   in Loop: Header=BB0_5 Depth=1
	s_or_b64 exec, exec, s[56:57]
	s_and_saveexec_b64 s[56:57], s[16:17]
	s_cbranch_execnz .LBB0_28
.LBB0_13:                               ;   in Loop: Header=BB0_5 Depth=1
	s_or_b64 exec, exec, s[56:57]
	s_and_saveexec_b64 s[56:57], s[18:19]
	s_cbranch_execnz .LBB0_29
.LBB0_14:                               ;   in Loop: Header=BB0_5 Depth=1
	s_or_b64 exec, exec, s[56:57]
	s_and_saveexec_b64 s[56:57], s[20:21]
	s_cbranch_execnz .LBB0_30
.LBB0_15:                               ;   in Loop: Header=BB0_5 Depth=1
	s_or_b64 exec, exec, s[56:57]
	s_and_saveexec_b64 s[56:57], s[22:23]
	s_cbranch_execnz .LBB0_31
.LBB0_16:                               ;   in Loop: Header=BB0_5 Depth=1
	s_or_b64 exec, exec, s[56:57]
	s_and_saveexec_b64 s[56:57], s[24:25]
	s_cbranch_execnz .LBB0_32
.LBB0_17:                               ;   in Loop: Header=BB0_5 Depth=1
	s_or_b64 exec, exec, s[56:57]
	s_and_saveexec_b64 s[56:57], s[26:27]
	s_cbranch_execnz .LBB0_33
.LBB0_18:                               ;   in Loop: Header=BB0_5 Depth=1
	s_or_b64 exec, exec, s[56:57]
	s_and_saveexec_b64 s[56:57], s[28:29]
	s_cbranch_execnz .LBB0_34
.LBB0_19:                               ;   in Loop: Header=BB0_5 Depth=1
	s_or_b64 exec, exec, s[56:57]
	s_and_saveexec_b64 s[56:57], s[30:31]
	s_cbranch_execnz .LBB0_35
.LBB0_20:                               ;   in Loop: Header=BB0_5 Depth=1
	s_or_b64 exec, exec, s[56:57]
	s_and_saveexec_b64 s[56:57], s[34:35]
	s_cbranch_execz .LBB0_4
	s_branch .LBB0_36
	.p2align	5, , 4
.LBB0_21:                               ;   in Loop: Header=BB0_5 Depth=1
	global_load_ushort v26, v[22:23], off offset:-30
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v2, v25, v26, v2 op_sel_hi:[0,1,0]
	s_or_b64 exec, exec, s[56:57]
	s_and_saveexec_b64 s[56:57], s[4:5]
	s_cbranch_execz .LBB0_7
.LBB0_22:                               ;   in Loop: Header=BB0_5 Depth=1
	global_load_ushort v26, v[22:23], off offset:-28
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v3, v25, v26, v3 op_sel_hi:[0,1,0]
	s_or_b64 exec, exec, s[56:57]
	s_and_saveexec_b64 s[56:57], s[6:7]
	s_cbranch_execz .LBB0_8
.LBB0_23:                               ;   in Loop: Header=BB0_5 Depth=1
	global_load_ushort v26, v[22:23], off offset:-26
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v4, v25, v26, v4 op_sel_hi:[0,1,0]
	s_or_b64 exec, exec, s[56:57]
	s_and_saveexec_b64 s[56:57], s[8:9]
	s_cbranch_execz .LBB0_9
.LBB0_24:                               ;   in Loop: Header=BB0_5 Depth=1
	global_load_ushort v26, v[22:23], off offset:-24
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v5, v25, v26, v5 op_sel_hi:[0,1,0]
	s_or_b64 exec, exec, s[56:57]
	s_and_saveexec_b64 s[56:57], s[10:11]
	s_cbranch_execz .LBB0_10
.LBB0_25:                               ;   in Loop: Header=BB0_5 Depth=1
	global_load_ushort v26, v[22:23], off offset:-22
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v6, v25, v26, v6 op_sel_hi:[0,1,0]
	s_or_b64 exec, exec, s[56:57]
	s_and_saveexec_b64 s[56:57], s[12:13]
	s_cbranch_execz .LBB0_11
.LBB0_26:                               ;   in Loop: Header=BB0_5 Depth=1
	global_load_ushort v26, v[22:23], off offset:-20
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v7, v25, v26, v7 op_sel_hi:[0,1,0]
	s_or_b64 exec, exec, s[56:57]
	s_and_saveexec_b64 s[56:57], s[14:15]
	s_cbranch_execz .LBB0_12
.LBB0_27:                               ;   in Loop: Header=BB0_5 Depth=1
	global_load_ushort v26, v[22:23], off offset:-18
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v8, v25, v26, v8 op_sel_hi:[0,1,0]
	s_or_b64 exec, exec, s[56:57]
	s_and_saveexec_b64 s[56:57], s[16:17]
	s_cbranch_execz .LBB0_13
.LBB0_28:                               ;   in Loop: Header=BB0_5 Depth=1
	global_load_ushort v26, v[22:23], off offset:-16
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v9, v25, v26, v9 op_sel_hi:[0,1,0]
	s_or_b64 exec, exec, s[56:57]
	s_and_saveexec_b64 s[56:57], s[18:19]
	s_cbranch_execz .LBB0_14
.LBB0_29:                               ;   in Loop: Header=BB0_5 Depth=1
	global_load_ushort v26, v[22:23], off offset:-14
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v10, v25, v26, v10 op_sel_hi:[0,1,0]
	s_or_b64 exec, exec, s[56:57]
	s_and_saveexec_b64 s[56:57], s[20:21]
	s_cbranch_execz .LBB0_15
.LBB0_30:                               ;   in Loop: Header=BB0_5 Depth=1
	global_load_ushort v26, v[22:23], off offset:-12
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v11, v25, v26, v11 op_sel_hi:[0,1,0]
	s_or_b64 exec, exec, s[56:57]
	s_and_saveexec_b64 s[56:57], s[22:23]
	s_cbranch_execz .LBB0_16
.LBB0_31:                               ;   in Loop: Header=BB0_5 Depth=1
	global_load_ushort v26, v[22:23], off offset:-10
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v12, v25, v26, v12 op_sel_hi:[0,1,0]
	s_or_b64 exec, exec, s[56:57]
	s_and_saveexec_b64 s[56:57], s[24:25]
	s_cbranch_execz .LBB0_17
.LBB0_32:                               ;   in Loop: Header=BB0_5 Depth=1
	global_load_ushort v26, v[22:23], off offset:-8
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v13, v25, v26, v13 op_sel_hi:[0,1,0]
	s_or_b64 exec, exec, s[56:57]
	s_and_saveexec_b64 s[56:57], s[26:27]
	s_cbranch_execz .LBB0_18
.LBB0_33:                               ;   in Loop: Header=BB0_5 Depth=1
	global_load_ushort v26, v[22:23], off offset:-6
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v14, v25, v26, v14 op_sel_hi:[0,1,0]
	s_or_b64 exec, exec, s[56:57]
	s_and_saveexec_b64 s[56:57], s[28:29]
	s_cbranch_execz .LBB0_19
.LBB0_34:                               ;   in Loop: Header=BB0_5 Depth=1
	global_load_ushort v26, v[22:23], off offset:-4
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v15, v25, v26, v15 op_sel_hi:[0,1,0]
	s_or_b64 exec, exec, s[56:57]
	s_and_saveexec_b64 s[56:57], s[30:31]
	s_cbranch_execz .LBB0_20
.LBB0_35:                               ;   in Loop: Header=BB0_5 Depth=1
	global_load_ushort v26, v[22:23], off offset:-2
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v16, v25, v26, v16 op_sel_hi:[0,1,0]
	s_or_b64 exec, exec, s[56:57]
	s_and_saveexec_b64 s[56:57], s[34:35]
	s_cbranch_execz .LBB0_4
.LBB0_36:                               ;   in Loop: Header=BB0_5 Depth=1
	global_load_ushort v26, v[22:23], off
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v17, v25, v26, v17 op_sel_hi:[0,1,0]
	s_branch .LBB0_4
.LBB0_37:
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v3, v2
	v_mov_b32_e32 v4, v2
	v_mov_b32_e32 v5, v2
	v_mov_b32_e32 v6, v2
	v_mov_b32_e32 v7, v2
	v_mov_b32_e32 v8, v2
	v_mov_b32_e32 v9, v2
	v_mov_b32_e32 v10, v2
	v_mov_b32_e32 v11, v2
	v_mov_b32_e32 v12, v2
	v_mov_b32_e32 v13, v2
	v_mov_b32_e32 v14, v2
	v_mov_b32_e32 v15, v2
	v_mov_b32_e32 v16, v2
	v_mov_b32_e32 v17, v2
.LBB0_38:
	s_or_b64 exec, exec, s[64:65]
	s_cmp_lg_u32 s49, 1
	s_mov_b64 s[4:5], -1
	s_cbranch_scc0 .LBB0_40
; %bb.39:
	s_waitcnt lgkmcnt(0)
	s_add_u32 s4, s60, s51
	s_addc_u32 s5, s61, 0
	s_lshl_b64 s[2:3], s[2:3], 14
	s_add_u32 s2, s4, s2
	v_lshlrev_b32_e32 v20, 2, v24
	v_lshlrev_b32_e32 v0, 2, v0
	v_lshlrev_b32_e32 v19, 8, v19
	s_movk_i32 s4, 0xf00
	s_addc_u32 s3, s5, s3
	v_lshl_or_b32 v20, v1, 8, v20
	v_or3_b32 v0, v19, v0, s4
	global_store_dword v20, v2, s[2:3]
	global_store_dword v20, v3, s[2:3] offset:256
	global_store_dword v20, v4, s[2:3] offset:512
	global_store_dword v20, v5, s[2:3] offset:768
	global_store_dword v20, v6, s[2:3] offset:1024
	global_store_dword v20, v7, s[2:3] offset:1280
	global_store_dword v20, v8, s[2:3] offset:1536
	global_store_dword v20, v9, s[2:3] offset:1792
	global_store_dword v20, v10, s[2:3] offset:2048
	global_store_dword v20, v11, s[2:3] offset:2304
	global_store_dword v20, v12, s[2:3] offset:2560
	global_store_dword v20, v13, s[2:3] offset:2816
	global_store_dword v20, v14, s[2:3] offset:3072
	global_store_dword v20, v15, s[2:3] offset:3328
	global_store_dword v20, v16, s[2:3] offset:3584
	global_store_dword v0, v17, s[2:3]
	s_mov_b64 s[4:5], 0
.LBB0_40:
	s_andn2_b64 vcc, exec, s[4:5]
	s_cbranch_vccnz .LBB0_51
; %bb.41:
	s_mul_hi_u32 s5, s33, s44
	s_mul_i32 s4, s33, s44
	s_lshl_b64 s[4:5], s[4:5], 1
	s_mul_hi_u32 s3, s33, s46
	s_mul_i32 s2, s33, s46
	s_waitcnt lgkmcnt(0)
	s_add_u32 s4, s54, s4
	s_addc_u32 s5, s55, s5
	s_lshl_b64 s[2:3], s[2:3], 1
	v_mov_b32_e32 v19, 0
	s_add_u32 s2, s52, s2
	v_lshlrev_b64 v[18:19], 1, v[18:19]
	s_addc_u32 s3, s53, s3
	v_add_u32_e32 v21, s62, v1
	v_lshl_add_u64 v[0:1], s[4:5], 0, v[18:19]
	v_lshl_add_u64 v[18:19], s[2:3], 0, v[18:19]
	v_cmp_neq_f32_e64 s[2:3], s48, 0
	s_mov_b32 s6, 0
	s_mov_b32 s7, s45
	v_cndmask_b32_e64 v20, 0, 1, s[2:3]
	s_mov_b32 s8, s43
	v_cmp_ne_u32_e64 s[2:3], 1, v20
	s_branch .LBB0_44
.LBB0_42:                               ;   in Loop: Header=BB0_44 Depth=1
	v_cvt_f16_f32_e32 v24, v22
	v_mad_u64_u32 v[22:23], s[10:11], v20, s7, 0
	v_lshl_add_u64 v[22:23], v[22:23], 1, v[18:19]
	global_store_short v[22:23], v24, off
.LBB0_43:                               ;   in Loop: Header=BB0_44 Depth=1
	s_or_b64 exec, exec, s[4:5]
	s_add_i32 s6, s6, 2
	s_cmp_eq_u32 s6, 16
	s_cbranch_scc1 .LBB0_51
.LBB0_44:                               ; =>This Inner Loop Header: Depth=1
	v_add_u32_e32 v20, s6, v21
	v_cmp_gt_u32_e32 vcc, s37, v20
	s_and_b64 s[10:11], s[0:1], vcc
	s_and_saveexec_b64 s[4:5], s[10:11]
	s_cbranch_execz .LBB0_48
; %bb.45:                               ;   in Loop: Header=BB0_44 Depth=1
	s_set_gpr_idx_on s6, gpr_idx(SRC0)
	v_mov_b32_e32 v22, v2
	s_set_gpr_idx_off
	s_and_b64 vcc, exec, s[2:3]
	v_mul_f32_e32 v22, s47, v22
	s_cbranch_vccnz .LBB0_47
; %bb.46:                               ;   in Loop: Header=BB0_44 Depth=1
	v_mad_u64_u32 v[24:25], s[10:11], v20, s8, 0
	v_lshl_add_u64 v[24:25], v[24:25], 1, v[0:1]
	global_load_ushort v23, v[24:25], off
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v22, s48, v23, v22 op_sel_hi:[0,1,0]
.LBB0_47:                               ;   in Loop: Header=BB0_44 Depth=1
	v_cvt_f16_f32_e32 v24, v22
	v_mad_u64_u32 v[22:23], s[10:11], v20, s7, 0
	v_lshl_add_u64 v[22:23], v[22:23], 1, v[18:19]
	global_store_short v[22:23], v24, off
.LBB0_48:                               ;   in Loop: Header=BB0_44 Depth=1
	s_or_b64 exec, exec, s[4:5]
	v_add_u32_e32 v20, 1, v20
	v_cmp_gt_u32_e32 vcc, s37, v20
	s_and_b64 s[10:11], s[0:1], vcc
	s_and_saveexec_b64 s[4:5], s[10:11]
	s_cbranch_execz .LBB0_43
; %bb.49:                               ;   in Loop: Header=BB0_44 Depth=1
	s_set_gpr_idx_on s6, gpr_idx(SRC0)
	v_mov_b32_e32 v22, v3
	s_set_gpr_idx_off
	s_and_b64 vcc, exec, s[2:3]
	v_mul_f32_e32 v22, s47, v22
	s_cbranch_vccnz .LBB0_42
; %bb.50:                               ;   in Loop: Header=BB0_44 Depth=1
	v_mad_u64_u32 v[24:25], s[10:11], v20, s8, 0
	v_lshl_add_u64 v[24:25], v[24:25], 1, v[0:1]
	global_load_ushort v23, v[24:25], off
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v22, s48, v23, v22 op_sel_hi:[0,1,0]
	s_branch .LBB0_42
.LBB0_51:
	s_endpgm
.Lfunc_end0:
	.size	c_hip_sk_parallel_Cijk_Ailk_Bjlk_HHS_BH_MT64x64x32_gfx950, .Lfunc_end0-c_hip_sk_parallel_Cijk_Ailk_Bjlk_HHS_BH_MT64x64x32_gfx950
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel c_hip_sk_parallel_Cijk_Ailk_Bjlk_HHS_BH_MT64x64x32_gfx950
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 104
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 27
		.amdhsa_next_free_sgpr 76
		.amdhsa_accum_offset 28
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
                                        ; -- End function
	.set .Lc_hip_sk_parallel_Cijk_Ailk_Bjlk_HHS_BH_MT64x64x32_gfx950.num_vgpr, 27
	.set .Lc_hip_sk_parallel_Cijk_Ailk_Bjlk_HHS_BH_MT64x64x32_gfx950.num_agpr, 0
	.set .Lc_hip_sk_parallel_Cijk_Ailk_Bjlk_HHS_BH_MT64x64x32_gfx950.numbered_sgpr, 76
	.set .Lc_hip_sk_parallel_Cijk_Ailk_Bjlk_HHS_BH_MT64x64x32_gfx950.num_named_barrier, 0
	.set .Lc_hip_sk_parallel_Cijk_Ailk_Bjlk_HHS_BH_MT64x64x32_gfx950.private_seg_size, 0
	.set .Lc_hip_sk_parallel_Cijk_Ailk_Bjlk_HHS_BH_MT64x64x32_gfx950.uses_vcc, 1
	.set .Lc_hip_sk_parallel_Cijk_Ailk_Bjlk_HHS_BH_MT64x64x32_gfx950.uses_flat_scratch, 0
	.set .Lc_hip_sk_parallel_Cijk_Ailk_Bjlk_HHS_BH_MT64x64x32_gfx950.has_dyn_sized_stack, 0
	.set .Lc_hip_sk_parallel_Cijk_Ailk_Bjlk_HHS_BH_MT64x64x32_gfx950.has_recursion, 0
	.set .Lc_hip_sk_parallel_Cijk_Ailk_Bjlk_HHS_BH_MT64x64x32_gfx950.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 2420
; TotalNumSgprs: 82
; NumVgprs: 27
; NumAgprs: 0
; TotalNumVgprs: 27
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 10
; VGPRBlocks: 3
; NumSGPRsForWavesPerEU: 82
; NumVGPRsForWavesPerEU: 27
; AccumOffset: 28
; Occupancy: 8
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 6
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.protected	c_hip_sk_parallel_Cijk_Ailk_Bjlk_HHS_BH_MT64x64x32_gfx950_streamKReduce ; -- Begin function c_hip_sk_parallel_Cijk_Ailk_Bjlk_HHS_BH_MT64x64x32_gfx950_streamKReduce
	.globl	c_hip_sk_parallel_Cijk_Ailk_Bjlk_HHS_BH_MT64x64x32_gfx950_streamKReduce
	.p2align	8
	.type	c_hip_sk_parallel_Cijk_Ailk_Bjlk_HHS_BH_MT64x64x32_gfx950_streamKReduce,@function
c_hip_sk_parallel_Cijk_Ailk_Bjlk_HHS_BH_MT64x64x32_gfx950_streamKReduce: ; @c_hip_sk_parallel_Cijk_Ailk_Bjlk_HHS_BH_MT64x64x32_gfx950_streamKReduce
; %bb.0:
	s_load_dwordx4 s[16:19], s[0:1], 0x38
	s_lshr_b32 s3, s2, 1
	s_waitcnt lgkmcnt(0)
	s_cmp_ge_u32 s3, s17
	s_cbranch_scc1 .LBB1_10
; %bb.1:
	s_load_dwordx8 s[8:15], s[0:1], 0x18
	v_mov_b32_e32 v5, 0
	s_waitcnt lgkmcnt(0)
	s_add_i32 s4, s8, 63
	s_add_i32 s5, s9, 63
	s_lshr_b32 s17, s4, 6
	s_lshr_b32 s4, s5, 6
	s_mul_i32 s19, s4, s17
	v_cvt_f32_u32_e32 v2, s19
	v_cvt_f32_u32_e32 v1, s17
	s_load_dwordx4 s[4:7], s[0:1], 0x0
	s_load_dwordx2 s[20:21], s[0:1], 0x10
	s_sub_i32 s0, 0, s19
	v_rcp_iflag_f32_e32 v2, v2
	v_rcp_iflag_f32_e32 v1, v1
	v_mul_f32_e32 v2, 0x4f7ffffe, v2
	v_cvt_u32_f32_e32 v2, v2
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	v_readfirstlane_b32 s22, v2
	s_mul_i32 s0, s0, s22
	s_mul_hi_u32 s0, s22, s0
	s_add_i32 s22, s22, s0
	s_mul_hi_u32 s0, s3, s22
	s_mul_i32 s22, s0, s19
	s_sub_i32 s22, s3, s22
	s_add_i32 s23, s0, 1
	s_sub_i32 s24, s22, s19
	s_cmp_ge_u32 s22, s19
	s_cselect_b32 s0, s23, s0
	s_cselect_b32 s22, s24, s22
	s_add_i32 s23, s0, 1
	s_cmp_ge_u32 s22, s19
	v_readfirstlane_b32 s1, v1
	s_cselect_b32 s25, s23, s0
	s_sub_i32 s0, 0, s17
	s_mul_i32 s0, s0, s1
	s_mul_i32 s19, s25, s19
	s_mul_hi_u32 s0, s1, s0
	s_sub_i32 s19, s3, s19
	s_add_i32 s1, s1, s0
	s_mul_hi_u32 s0, s19, s1
	s_mul_i32 s1, s0, s17
	s_sub_i32 s1, s19, s1
	s_add_i32 s22, s0, 1
	s_sub_i32 s23, s1, s17
	s_cmp_ge_u32 s1, s17
	s_cselect_b32 s0, s22, s0
	s_cselect_b32 s1, s23, s1
	s_add_i32 s22, s0, 1
	s_cmp_ge_u32 s1, s17
	s_cselect_b32 s0, s22, s0
	s_mul_i32 s1, s0, s17
	s_sub_i32 s1, s19, s1
	s_lshl_b32 s19, s1, 6
	s_lshl_b32 s17, s0, 6
	s_mul_hi_u32 s1, s3, s16
	s_mul_i32 s0, s3, s16
	s_lshl_b64 s[22:23], s[0:1], 14
	s_lshl_b32 s0, s2, 11
	s_and_b32 s0, s0, 0x800
	s_add_i32 s24, s0, 0x800
	s_cmp_lg_u32 s16, 0
	s_mul_hi_u32 s27, s25, s11
	s_mul_i32 s26, s25, s11
	s_cselect_b64 s[2:3], -1, 0
	s_lshl_b64 s[26:27], s[26:27], 1
	v_or_b32_e32 v7, s0, v0
	s_mul_hi_u32 s1, s25, s13
	s_mul_i32 s0, s25, s13
	s_waitcnt lgkmcnt(0)
	s_add_u32 s6, s6, s26
	s_addc_u32 s7, s7, s27
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s4, s4, s0
	v_and_or_b32 v4, v0, 63, s19
	s_addc_u32 s5, s5, s1
	v_lshlrev_b64 v[2:3], 1, v[4:5]
	v_lshl_add_u64 v[0:1], s[6:7], 0, v[2:3]
	v_lshl_add_u64 v[2:3], s[4:5], 0, v[2:3]
	s_add_u32 s4, s20, s18
	s_addc_u32 s5, s21, 0
	s_add_u32 s4, s4, s22
	v_cndmask_b32_e64 v6, 0, 1, s[2:3]
	v_cmp_neq_f32_e64 s[2:3], s15, 0
	v_cmp_gt_u32_e64 s[0:1], s8, v4
	v_lshlrev_b32_e32 v4, 2, v7
	s_addc_u32 s5, s5, s23
	v_cndmask_b32_e64 v8, 0, 1, s[2:3]
	v_lshl_add_u64 v[4:5], s[4:5], 0, v[4:5]
	s_mov_b64 s[6:7], 0
	s_mov_b64 s[18:19], 0x4000
	s_mov_b64 s[20:21], 0x400
	v_cmp_ne_u32_e64 s[2:3], 1, v6
	v_cmp_ne_u32_e64 s[4:5], 1, v8
	s_branch .LBB1_4
.LBB1_2:                                ;   in Loop: Header=BB1_4 Depth=1
	v_cvt_f16_f32_e32 v10, v8
	v_mad_u64_u32 v[8:9], s[26:27], v6, s12, 0
	v_lshl_add_u64 v[8:9], v[8:9], 1, v[2:3]
	global_store_short v[8:9], v10, off
.LBB1_3:                                ;   in Loop: Header=BB1_4 Depth=1
	s_or_b64 exec, exec, s[22:23]
	v_add_u32_e32 v7, 0x100, v7
	v_cmp_le_u32_e32 vcc, s24, v7
	s_or_b64 s[6:7], vcc, s[6:7]
	v_lshl_add_u64 v[4:5], v[4:5], 0, s[20:21]
	s_andn2_b64 exec, exec, s[6:7]
	s_cbranch_execz .LBB1_10
.LBB1_4:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB1_7 Depth 2
	v_lshrrev_b32_e32 v6, 6, v7
	v_add_u32_e32 v6, s17, v6
	v_cmp_gt_u32_e32 vcc, s9, v6
	s_and_b64 s[26:27], s[0:1], vcc
	s_and_saveexec_b64 s[22:23], s[26:27]
	s_cbranch_execz .LBB1_3
; %bb.5:                                ;   in Loop: Header=BB1_4 Depth=1
	s_and_b64 vcc, exec, s[2:3]
	v_mov_b32_e32 v10, 0
	s_cbranch_vccnz .LBB1_8
; %bb.6:                                ;   in Loop: Header=BB1_4 Depth=1
	v_mov_b32_e32 v10, 0
	v_mov_b64_e32 v[8:9], v[4:5]
	s_mov_b32 s8, s16
	.p2align	5, , 4
.LBB1_7:                                ;   Parent Loop BB1_4 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	global_load_dword v11, v[8:9], off
	s_add_i32 s8, s8, -1
	v_lshl_add_u64 v[8:9], v[8:9], 0, s[18:19]
	s_cmp_eq_u32 s8, 0
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v10, v10, v11
	s_cbranch_scc0 .LBB1_7
.LBB1_8:                                ;   in Loop: Header=BB1_4 Depth=1
	s_and_b64 vcc, exec, s[4:5]
	v_mul_f32_e32 v8, s14, v10
	s_cbranch_vccnz .LBB1_2
; %bb.9:                                ;   in Loop: Header=BB1_4 Depth=1
	v_mad_u64_u32 v[10:11], s[26:27], v6, s10, 0
	v_lshl_add_u64 v[10:11], v[10:11], 1, v[0:1]
	global_load_ushort v9, v[10:11], off
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v8, s15, v9, v8 op_sel_hi:[0,1,0]
	s_branch .LBB1_2
.LBB1_10:
	s_endpgm
.Lfunc_end1:
	.size	c_hip_sk_parallel_Cijk_Ailk_Bjlk_HHS_BH_MT64x64x32_gfx950_streamKReduce, .Lfunc_end1-c_hip_sk_parallel_Cijk_Ailk_Bjlk_HHS_BH_MT64x64x32_gfx950_streamKReduce
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel c_hip_sk_parallel_Cijk_Ailk_Bjlk_HHS_BH_MT64x64x32_gfx950_streamKReduce
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 68
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 12
		.amdhsa_next_free_sgpr 28
		.amdhsa_accum_offset 12
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
                                        ; -- End function
	.set .Lc_hip_sk_parallel_Cijk_Ailk_Bjlk_HHS_BH_MT64x64x32_gfx950_streamKReduce.num_vgpr, 12
	.set .Lc_hip_sk_parallel_Cijk_Ailk_Bjlk_HHS_BH_MT64x64x32_gfx950_streamKReduce.num_agpr, 0
	.set .Lc_hip_sk_parallel_Cijk_Ailk_Bjlk_HHS_BH_MT64x64x32_gfx950_streamKReduce.numbered_sgpr, 28
	.set .Lc_hip_sk_parallel_Cijk_Ailk_Bjlk_HHS_BH_MT64x64x32_gfx950_streamKReduce.num_named_barrier, 0
	.set .Lc_hip_sk_parallel_Cijk_Ailk_Bjlk_HHS_BH_MT64x64x32_gfx950_streamKReduce.private_seg_size, 0
	.set .Lc_hip_sk_parallel_Cijk_Ailk_Bjlk_HHS_BH_MT64x64x32_gfx950_streamKReduce.uses_vcc, 1
	.set .Lc_hip_sk_parallel_Cijk_Ailk_Bjlk_HHS_BH_MT64x64x32_gfx950_streamKReduce.uses_flat_scratch, 0
	.set .Lc_hip_sk_parallel_Cijk_Ailk_Bjlk_HHS_BH_MT64x64x32_gfx950_streamKReduce.has_dyn_sized_stack, 0
	.set .Lc_hip_sk_parallel_Cijk_Ailk_Bjlk_HHS_BH_MT64x64x32_gfx950_streamKReduce.has_recursion, 0
	.set .Lc_hip_sk_parallel_Cijk_Ailk_Bjlk_HHS_BH_MT64x64x32_gfx950_streamKReduce.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 700
; TotalNumSgprs: 34
; NumVgprs: 12
; NumAgprs: 0
; TotalNumVgprs: 12
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 4
; VGPRBlocks: 1
; NumSGPRsForWavesPerEU: 34
; NumVGPRsForWavesPerEU: 12
; AccumOffset: 12
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 2
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.text
	.type	__hip_cuid_838620b059b3757e,@object ; @__hip_cuid_838620b059b3757e
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_838620b059b3757e
__hip_cuid_838620b059b3757e:
	.byte	0                               ; 0x0
	.size	__hip_cuid_838620b059b3757e, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 46fcb339fb61119b337f973c7ca9e710a319fdd0+PATCHED:440716f8b87be9d8e20ed910e10e5b6d14d57cf6)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_838620b059b3757e
	.amdgpu_metadata
---
custom.config:
  Source:
    Origin: test
  Version: 1.0.0
  Features:
    SupportsUserArgs: false
    SupportsBias: false
    SupportsActivation: false
    SupportsScaleAlpha: false
    SupportsGSU: false
  InternalSupportParams:
    KernArgsVersion: 3
    PersistentLoopArgsVersion: 3
  ProblemType:
    OperationType: GEMM
    DataType: h
    DestDataType: h
    ComputeDataType: s
    HighPrecisionAccumulate: True
    TransposeA: False
    TransposeB: True
    UseBeta: True
    Batched: True
    UseBias: 0
    Activation: False
    UseScaleAlphaVec: 0
  TileProcessingStrategy: StreamK
  WorkAssignment: StaticGrid
  CustomKernel:
    args: [ { type: address, semantic: AddressD },
            { type: address, semantic: AddressC },
            { type: address, semantic: AddressA },
            { type: address, semantic: AddressB },
            { type: address, semantic: AddressWorkspace },
            { type: uint32, semantic: SizeFree0 },
            { type: uint32, semantic: SizeFree1 },
            { type: uint32, semantic: SizeSum },
            { type: uint32, semantic: StrideA0 },
            { type: uint32, semantic: StrideA1 },
            { type: uint32, semantic: StrideB0 },
            { type: uint32, semantic: StrideB1 },
            { type: uint32, semantic: StrideC0 },
            { type: uint32, semantic: StrideC1 },
            { type: uint32, semantic: StrideD0 },
            { type: uint32, semantic: StrideD1 },
            { type: float32, semantic: Alpha },
            { type: float32, semantic: Beta },
            { type: uint32, semantic: SKSplitFactor },
            { type: uint32, semantic: NumTiles },
            { type: uint32, semantic: WorkspaceOffsetPartials } ]
    macrotile: [64, 64, 32]
    threads: [256, 1, 1]
    grid: [StreamKNoBatch, One, One]
    workspaceType: StreamK
    workspaceSizePerElemC: 4
    reductions: [parallel]
    postKernels:
      - kind: streamKReduce
        version: 1
        name: c_hip_sk_parallel_Cijk_Ailk_Bjlk_HHS_BH_MT64x64x32_gfx950_streamKReduce
        args: [ { type: address, semantic: AddressD },
                { type: address, semantic: AddressC },
                { type: address, semantic: AddressWorkspace },
                { type: uint32, semantic: SizeFree0 },
                { type: uint32, semantic: SizeFree1 },
                { type: uint32, semantic: StrideC0 },
                { type: uint32, semantic: StrideC1 },
                { type: uint32, semantic: StrideD0 },
                { type: uint32, semantic: StrideD1 },
                { type: float32, semantic: Alpha },
                { type: float32, semantic: Beta },
                { type: uint32, semantic: SKSplitFactor },
                { type: uint32, semantic: NumTiles },
                { type: uint32, semantic: WorkspaceOffsetPartials } ]
        threads: [256, 1, 1]
        grid: [TilesBands, One, One]
        bands: 2
  MatrixInstruction: [16, 16, 16, 1]
  EnableMatrixInstruction: True
  StaggerU: 0
  WavefrontSize: 64
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .offset:         40
        .size:           4
        .value_kind:     by_value
      - .offset:         44
        .size:           4
        .value_kind:     by_value
      - .offset:         48
        .size:           4
        .value_kind:     by_value
      - .offset:         52
        .size:           4
        .value_kind:     by_value
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
      - .offset:         72
        .size:           4
        .value_kind:     by_value
      - .offset:         76
        .size:           4
        .value_kind:     by_value
      - .offset:         80
        .size:           4
        .value_kind:     by_value
      - .offset:         84
        .size:           4
        .value_kind:     by_value
      - .offset:         88
        .size:           4
        .value_kind:     by_value
      - .offset:         92
        .size:           4
        .value_kind:     by_value
      - .offset:         96
        .size:           4
        .value_kind:     by_value
      - .offset:         100
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 104
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           c_hip_sk_parallel_Cijk_Ailk_Bjlk_HHS_BH_MT64x64x32_gfx950
    .private_segment_fixed_size: 0
    .sgpr_count:     82
    .sgpr_spill_count: 0
    .symbol:         c_hip_sk_parallel_Cijk_Ailk_Bjlk_HHS_BH_MT64x64x32_gfx950.kd
    .vgpr_count:     27
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
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
      - .offset:         40
        .size:           4
        .value_kind:     by_value
      - .offset:         44
        .size:           4
        .value_kind:     by_value
      - .offset:         48
        .size:           4
        .value_kind:     by_value
      - .offset:         52
        .size:           4
        .value_kind:     by_value
      - .offset:         56
        .size:           4
        .value_kind:     by_value
      - .offset:         60
        .size:           4
        .value_kind:     by_value
      - .offset:         64
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 68
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           c_hip_sk_parallel_Cijk_Ailk_Bjlk_HHS_BH_MT64x64x32_gfx950_streamKReduce
    .private_segment_fixed_size: 0
    .sgpr_count:     34
    .sgpr_spill_count: 0
    .symbol:         c_hip_sk_parallel_Cijk_Ailk_Bjlk_HHS_BH_MT64x64x32_gfx950_streamKReduce.kd
    .vgpr_count:     12
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 1
...

	.end_amdgpu_metadata
