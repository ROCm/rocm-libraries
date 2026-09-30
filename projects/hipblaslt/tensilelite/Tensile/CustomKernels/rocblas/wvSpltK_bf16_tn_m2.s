	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.protected	wvSpltK_bf16_tn_m2      ; -- Begin function wvSpltK_bf16_tn_m2
	.globl	wvSpltK_bf16_tn_m2
	.p2align	8
	.type	wvSpltK_bf16_tn_m2,@function
wvSpltK_bf16_tn_m2:                     ; @wvSpltK_bf16_tn_m2
; %bb.75:
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.76:
.LBB0_0:
	v_bfe_u32 v4, v0, 10, 10
	v_lshlrev_b32_e32 v1, 1, v4
	v_lshl_add_u32 v34, s16, 5, v1
	v_mov_b32_e32 v35, 0
	s_mov_b32 s24, s3
	s_ashr_i32 s25, s3, 31
	v_lshl_add_u64 v[2:3], v[34:35], 0, 2
	v_cmp_gt_u64_e32 vcc, s[24:25], v[34:35]
	v_cmp_le_u64_e64 s[18:19], s[24:25], v[2:3]
	v_mov_b32_e32 v32, 1
	s_mov_b32 s17, s11
	s_and_b64 s[20:21], vcc, s[18:19]
	v_mov_b32_e32 v33, v32
	s_and_saveexec_b64 s[18:19], s[20:21]
	s_cbranch_execz .LBB0_6
; %bb.1:
	s_add_i32 s20, s24, -2
	s_mov_b32 s23, 0
	s_mov_b32 s21, s23
	v_cmp_ne_u32_e32 vcc, s20, v34
	s_mov_b32 s28, 1
	v_mov_b32_e32 v33, v32
	s_and_saveexec_b64 s[26:27], vcc
	s_cbranch_execz .LBB0_5
; %bb.2:
	v_subrev_co_u32_e32 v2, vcc, s20, v34
	s_mov_b64 s[34:35], 0
	s_nop 0
	v_subb_co_u32_e64 v3, s[30:31], 0, 0, vcc
	s_mov_b64 s[30:31], 0
	s_mov_b32 s29, s28
.LBB0_3:                                ; =>This Inner Loop Header: Depth=1
	s_cmp_lg_u32 s34, 1
	s_cselect_b32 s29, s29, 0
	s_cmp_lg_u32 s34, 0
	s_cselect_b32 s28, s28, 0
	s_add_u32 s34, s34, 1
	s_mov_b32 s22, s34
	s_addc_u32 s35, s35, 0
	v_cmp_ge_u64_e32 vcc, s[22:23], v[2:3]
	s_or_b64 s[30:31], vcc, s[30:31]
	v_mov_b64_e32 v[32:33], s[28:29]
	s_andn2_b64 exec, exec, s[30:31]
	s_cbranch_execnz .LBB0_3
; %bb.4:
	s_or_b64 exec, exec, s[30:31]
.LBB0_5:
	s_or_b64 exec, exec, s[26:27]
	v_mov_b64_e32 v[34:35], s[20:21]
.LBB0_6:
	s_or_b64 exec, exec, s[18:19]
	v_and_b32_e32 v1, 0x3ff, v0
	s_mul_i32 s3, s9, s2
	v_lshlrev_b32_e32 v42, 3, v1
	s_min_i32 s3, s3, 0x10000
	v_lshl_add_u32 v0, v4, 9, v42
	v_cmp_gt_u32_e32 vcc, s3, v0
	s_and_saveexec_b64 s[18:19], vcc
	s_cbranch_execz .LBB0_9
; %bb.7:
	v_cvt_f32_u32_e32 v2, s2
	s_sub_i32 s16, 0, s2
	v_lshlrev_b32_e32 v3, 4, v1
	v_lshl_add_u32 v3, v4, 10, v3
	v_rcp_iflag_f32_e32 v2, v2
	s_ashr_i32 s11, s17, 31
	s_mov_b64 s[20:21], 0
	v_mul_f32_e32 v2, 0x4f7ffffe, v2
	v_cvt_u32_f32_e32 v5, v2
	v_mov_b32_e32 v2, 0
	v_mul_lo_u32 v4, s16, v5
	v_mul_hi_u32 v4, v5, v4
	v_add_u32_e32 v4, v5, v4
.LBB0_8:                                ; =>This Inner Loop Header: Depth=1
	v_mul_hi_u32 v5, v4, v0
	v_mul_lo_u32 v6, s2, v5
	v_not_b32_e32 v7, v5
	v_sub_u32_e32 v9, v0, v6
	v_add_u32_e32 v8, 1, v5
	v_mad_u64_u32 v[6:7], s[22:23], s2, v7, v[0:1]
	v_cmp_le_u32_e32 vcc, s2, v9
	s_nop 1
	v_cndmask_b32_e32 v5, v5, v8, vcc
	v_cndmask_b32_e32 v6, v9, v6, vcc
	v_add_u32_e32 v7, 1, v5
	v_cmp_le_u32_e32 vcc, s2, v6
	s_nop 1
	v_cndmask_b32_e32 v5, v5, v7, vcc
	v_mad_u64_u32 v[8:9], s[22:23], v5, s17, 0
	v_mov_b32_e32 v10, v9
	v_mad_u64_u32 v[10:11], s[22:23], v5, s11, v[10:11]
	v_mad_u64_u32 v[6:7], s[22:23], s16, v5, v[0:1]
	v_mov_b32_e32 v9, v10
	v_mov_b32_e32 v7, v2
	v_lshl_add_u64 v[8:9], v[8:9], 1, s[6:7]
	v_lshl_add_u64 v[6:7], v[6:7], 1, v[8:9]
	global_load_dwordx4 v[6:9], v[6:7], off
	v_add_u32_e32 v0, 0x2000, v0
	v_cmp_le_u32_e32 vcc, s3, v0
	s_or_b64 s[20:21], vcc, s[20:21]
	s_waitcnt vmcnt(0)
	ds_write_b128 v3, v[6:9]
	v_add_u32_e32 v3, 0x4000, v3
	s_andn2_b64 exec, exec, s[20:21]
	s_cbranch_execnz .LBB0_8
.LBB0_9:
	s_or_b64 exec, exec, s[18:19]
	v_cmp_gt_u64_e32 vcc, s[24:25], v[34:35]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_and_saveexec_b64 s[18:19], vcc
	s_cbranch_execz .LBB0_74
; %bb.10:
	s_load_dwordx4 s[20:23], s[0:1], 0x38
	s_add_i32 s0, s9, -1
	s_min_i32 s3, s0, 1
	s_min_i32 s16, s0, 0
	s_lshl_b32 s26, s8, 5
	s_cmp_lg_u32 s2, 0
	s_cselect_b64 s[46:47], -1, 0
	s_ashr_i32 s8, s17, 31
	s_mul_i32 s33, s16, s8
	s_mul_hi_u32 s36, s16, s17
	s_add_i32 s37, s36, s33
	s_mul_i32 s8, s3, s8
	s_mul_hi_u32 s33, s3, s17
	s_ashr_i32 s11, s10, 31
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s31, s22, 31
	s_mov_b32 s30, s22
	s_ashr_i32 s35, s23, 31
	s_ashr_i32 s27, s26, 31
	s_add_i32 s22, s24, -2
	s_add_i32 s39, s33, s8
	s_mul_i32 s18, s16, s2
	s_mul_i32 s36, s16, s17
	s_mul_i32 s16, s3, s2
	s_cmp_gt_i32 s9, 0
	v_lshlrev_b32_e32 v0, 4, v1
	s_mov_b32 s19, 0
	s_cselect_b64 s[40:41], -1, 0
	s_cmp_gt_i32 s9, 1
	v_lshl_add_u32 v43, s18, 1, v0
	v_lshl_add_u32 v45, s16, 1, v0
	v_cndmask_b32_e64 v0, 0, 1, s[46:47]
	v_cmp_eq_u32_e64 s[0:1], 63, v1
	v_cmp_neq_f32_e64 s[28:29], s21, 0
	s_mov_b32 s34, s23
	s_mov_b32 s23, s19
	s_mul_i32 s38, s3, s17
	s_cselect_b64 s[42:43], -1, 0
	v_add_u32_e32 v44, s18, v42
	v_add_u32_e32 v46, s16, v42
	s_mov_b64 s[44:45], 0
	v_cmp_ne_u32_e64 s[16:17], 1, v0
	v_mov_b32_e32 v37, 0
	s_mov_b32 s3, 0xffff
                                        ; implicit-def: $vgpr0_vgpr1_vgpr2_vgpr3
                                        ; implicit-def: $vgpr4_vgpr5_vgpr6_vgpr7
                                        ; implicit-def: $vgpr8_vgpr9_vgpr10_vgpr11
                                        ; implicit-def: $vgpr12_vgpr13_vgpr14_vgpr15
                                        ; implicit-def: $vgpr18_vgpr19
                                        ; implicit-def: $vgpr26_vgpr27
                                        ; implicit-def: $vgpr22_vgpr23
                                        ; implicit-def: $vgpr30_vgpr31
	s_branch .LBB0_13
.LBB0_11:                               ;   in Loop: Header=BB0_13 Depth=1
	s_or_b64 exec, exec, s[48:49]
	v_mov_b64_e32 v[34:35], s[22:23]
.LBB0_12:                               ;   in Loop: Header=BB0_13 Depth=1
	s_or_b64 exec, exec, s[46:47]
	v_cmp_le_u64_e32 vcc, s[24:25], v[34:35]
	s_or_b64 s[44:45], vcc, s[44:45]
	s_andn2_b64 exec, exec, s[44:45]
	s_cbranch_execz .LBB0_74
.LBB0_13:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_17 Depth 2
                                        ;     Child Loop BB0_72 Depth 2
	s_and_b64 vcc, exec, s[16:17]
	s_cbranch_vccnz .LBB0_48
; %bb.14:                               ;   in Loop: Header=BB0_13 Depth=1
	v_mul_lo_u32 v36, v35, s10
	v_mul_lo_u32 v40, v34, s11
	v_mad_u64_u32 v[38:39], s[8:9], v34, s10, 0
	v_add3_u32 v39, v39, v40, v36
	v_lshl_add_u64 v[38:39], v[38:39], 1, s[4:5]
	s_mov_b32 s18, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v51, v45
	v_mov_b32_e32 v52, v43
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v50, 0
	s_branch .LBB0_17
.LBB0_15:                               ;   in Loop: Header=BB0_17 Depth=2
	s_or_b64 exec, exec, s[46:47]
.LBB0_16:                               ;   in Loop: Header=BB0_17 Depth=2
	s_or_b64 exec, exec, s[8:9]
	s_addk_i32 s18, 0x400
	v_add_u32_e32 v52, 0x800, v52
	s_cmp_ge_u32 s18, s2
	v_add_u32_e32 v51, 0x800, v51
	s_cbranch_scc1 .LBB0_49
.LBB0_17:                               ;   Parent Loop BB0_13 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_add_u32_e32 v36, s18, v42
	v_cmp_gt_u32_e32 vcc, s2, v36
	v_add_u32_e32 v40, 0x200, v36
	s_and_saveexec_b64 s[46:47], vcc
	s_cbranch_execnz .LBB0_21
; %bb.18:                               ;   in Loop: Header=BB0_17 Depth=2
	s_or_b64 exec, exec, s[46:47]
	s_and_saveexec_b64 s[46:47], vcc
	s_cbranch_execnz .LBB0_24
.LBB0_19:                               ;   in Loop: Header=BB0_17 Depth=2
	s_or_b64 exec, exec, s[46:47]
	s_and_saveexec_b64 s[46:47], vcc
	s_cbranch_execnz .LBB0_43
.LBB0_20:                               ;   in Loop: Header=BB0_17 Depth=2
	s_or_b64 exec, exec, s[46:47]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_16
	s_branch .LBB0_46
.LBB0_21:                               ;   in Loop: Header=BB0_17 Depth=2
	v_lshl_add_u64 v[54:55], v[36:37], 1, v[38:39]
	v_lshl_add_u64 v[56:57], s[10:11], 1, v[54:55]
	global_load_dwordx4 v[12:15], v[54:55], off nt
	global_load_dwordx4 v[4:7], v[56:57], off nt
	v_cmp_gt_u32_e64 s[8:9], s2, v40
	s_and_saveexec_b64 s[48:49], s[8:9]
	s_cbranch_execz .LBB0_23
; %bb.22:                               ;   in Loop: Header=BB0_17 Depth=2
	v_mov_b32_e32 v41, v37
	v_lshl_add_u64 v[54:55], v[40:41], 1, v[38:39]
	v_lshl_add_u64 v[56:57], s[10:11], 1, v[54:55]
	global_load_dwordx4 v[8:11], v[54:55], off nt
	global_load_dwordx4 v[0:3], v[56:57], off nt
.LBB0_23:                               ;   in Loop: Header=BB0_17 Depth=2
	s_or_b64 exec, exec, s[48:49]
	s_or_b64 exec, exec, s[46:47]
	s_and_saveexec_b64 s[46:47], vcc
	s_cbranch_execz .LBB0_19
.LBB0_24:                               ;   in Loop: Header=BB0_17 Depth=2
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshl_add_u64 v[26:27], v[36:37], 1, s[6:7]
	v_add_u32_e32 v36, s18, v44
	v_cmp_lt_u32_e64 s[8:9], s3, v36
                                        ; implicit-def: $vgpr16_vgpr17
	s_and_saveexec_b64 s[48:49], s[8:9]
	s_xor_b64 s[8:9], exec, s[48:49]
	s_cbranch_execz .LBB0_26
; %bb.25:                               ;   in Loop: Header=BB0_17 Depth=2
	v_lshl_add_u64 v[16:17], s[36:37], 1, v[26:27]
	global_load_dwordx4 v[16:19], v[16:17], off
.LBB0_26:                               ;   in Loop: Header=BB0_17 Depth=2
	s_andn2_saveexec_b64 s[8:9], s[8:9]
	s_cbranch_execz .LBB0_28
; %bb.27:                               ;   in Loop: Header=BB0_17 Depth=2
	s_waitcnt vmcnt(0)
	ds_read_b128 v[16:19], v52
.LBB0_28:                               ;   in Loop: Header=BB0_17 Depth=2
	s_or_b64 exec, exec, s[8:9]
	v_add_u32_e32 v53, s18, v46
	v_cmp_lt_u32_e64 s[8:9], s3, v53
                                        ; implicit-def: $vgpr24_vgpr25
	s_and_saveexec_b64 s[48:49], s[8:9]
	s_xor_b64 s[8:9], exec, s[48:49]
	s_cbranch_execnz .LBB0_31
; %bb.29:                               ;   in Loop: Header=BB0_17 Depth=2
	s_andn2_saveexec_b64 s[8:9], s[8:9]
	s_cbranch_execnz .LBB0_32
.LBB0_30:                               ;   in Loop: Header=BB0_17 Depth=2
	s_or_b64 exec, exec, s[8:9]
	v_cmp_gt_u32_e64 s[8:9], s2, v40
	s_and_saveexec_b64 s[48:49], s[8:9]
	s_cbranch_execnz .LBB0_33
	s_branch .LBB0_42
.LBB0_31:                               ;   in Loop: Header=BB0_17 Depth=2
	v_lshl_add_u64 v[24:25], s[38:39], 1, v[26:27]
	global_load_dwordx4 v[24:27], v[24:25], off
	s_andn2_saveexec_b64 s[8:9], s[8:9]
	s_cbranch_execz .LBB0_30
.LBB0_32:                               ;   in Loop: Header=BB0_17 Depth=2
	s_waitcnt vmcnt(0)
	ds_read_b128 v[24:27], v51
	s_or_b64 exec, exec, s[8:9]
	v_cmp_gt_u32_e64 s[8:9], s2, v40
	s_and_saveexec_b64 s[48:49], s[8:9]
	s_cbranch_execz .LBB0_42
.LBB0_33:                               ;   in Loop: Header=BB0_17 Depth=2
	v_mov_b32_e32 v41, v37
	v_add_u32_e32 v20, 0x200, v36
	v_lshl_add_u64 v[30:31], v[40:41], 1, s[6:7]
	v_cmp_lt_u32_e64 s[8:9], s3, v20
                                        ; implicit-def: $vgpr20_vgpr21
	s_and_saveexec_b64 s[50:51], s[8:9]
	s_xor_b64 s[8:9], exec, s[50:51]
	s_cbranch_execz .LBB0_35
; %bb.34:                               ;   in Loop: Header=BB0_17 Depth=2
	v_lshl_add_u64 v[20:21], s[36:37], 1, v[30:31]
	global_load_dwordx4 v[20:23], v[20:21], off
.LBB0_35:                               ;   in Loop: Header=BB0_17 Depth=2
	s_andn2_saveexec_b64 s[8:9], s[8:9]
	s_cbranch_execz .LBB0_37
; %bb.36:                               ;   in Loop: Header=BB0_17 Depth=2
	s_waitcnt vmcnt(0)
	ds_read_b128 v[20:23], v52 offset:1024
.LBB0_37:                               ;   in Loop: Header=BB0_17 Depth=2
	s_or_b64 exec, exec, s[8:9]
	v_add_u32_e32 v28, 0x200, v53
	v_cmp_lt_u32_e64 s[8:9], s3, v28
                                        ; implicit-def: $vgpr28_vgpr29
	s_and_saveexec_b64 s[50:51], s[8:9]
	s_xor_b64 s[8:9], exec, s[50:51]
	s_cbranch_execz .LBB0_39
; %bb.38:                               ;   in Loop: Header=BB0_17 Depth=2
	v_lshl_add_u64 v[28:29], s[38:39], 1, v[30:31]
	global_load_dwordx4 v[28:31], v[28:29], off
.LBB0_39:                               ;   in Loop: Header=BB0_17 Depth=2
	s_andn2_saveexec_b64 s[8:9], s[8:9]
	s_cbranch_execz .LBB0_41
; %bb.40:                               ;   in Loop: Header=BB0_17 Depth=2
	s_waitcnt vmcnt(0)
	ds_read_b128 v[28:31], v51 offset:1024
.LBB0_41:                               ;   in Loop: Header=BB0_17 Depth=2
	s_or_b64 exec, exec, s[8:9]
.LBB0_42:                               ;   in Loop: Header=BB0_17 Depth=2
	s_or_b64 exec, exec, s[48:49]
	s_or_b64 exec, exec, s[46:47]
	s_and_saveexec_b64 s[46:47], vcc
	s_cbranch_execz .LBB0_20
.LBB0_43:                               ;   in Loop: Header=BB0_17 Depth=2
	s_waitcnt vmcnt(0) lgkmcnt(0)
	;;#ASMSTART
	v_dot2c_f32_bf16 v50, v16, v12
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_bf16 v49, v16, v4
	;;#ASMEND
	v_cmp_gt_u32_e64 s[8:9], s2, v40
	;;#ASMSTART
	v_dot2c_f32_bf16 v50, v17, v13
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_bf16 v49, v17, v5
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_bf16 v50, v18, v14
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_bf16 v49, v18, v6
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_bf16 v50, v19, v15
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_bf16 v49, v19, v7
	;;#ASMEND
	s_and_saveexec_b64 s[48:49], s[8:9]
	s_cbranch_execz .LBB0_45
; %bb.44:                               ;   in Loop: Header=BB0_17 Depth=2
	;;#ASMSTART
	v_dot2c_f32_bf16 v50, v20, v8
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_bf16 v49, v20, v0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_bf16 v50, v21, v9
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_bf16 v49, v21, v1
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_bf16 v50, v22, v10
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_bf16 v49, v22, v2
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_bf16 v50, v23, v11
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_bf16 v49, v23, v3
	;;#ASMEND
.LBB0_45:                               ;   in Loop: Header=BB0_17 Depth=2
	s_or_b64 exec, exec, s[48:49]
	s_or_b64 exec, exec, s[46:47]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_16
.LBB0_46:                               ;   in Loop: Header=BB0_17 Depth=2
	s_waitcnt vmcnt(0) lgkmcnt(0)
	;;#ASMSTART
	v_dot2c_f32_bf16 v48, v24, v12
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_bf16 v47, v24, v4
	;;#ASMEND
	v_cmp_gt_u32_e32 vcc, s2, v40
	;;#ASMSTART
	v_dot2c_f32_bf16 v48, v25, v13
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_bf16 v47, v25, v5
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_bf16 v48, v26, v14
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_bf16 v47, v26, v6
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_bf16 v48, v27, v15
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_bf16 v47, v27, v7
	;;#ASMEND
	s_and_saveexec_b64 s[46:47], vcc
	s_cbranch_execz .LBB0_15
; %bb.47:                               ;   in Loop: Header=BB0_17 Depth=2
	;;#ASMSTART
	v_dot2c_f32_bf16 v48, v28, v8
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_bf16 v47, v28, v0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_bf16 v48, v29, v9
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_bf16 v47, v29, v1
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_bf16 v48, v30, v10
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_bf16 v47, v30, v2
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_bf16 v48, v31, v11
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_bf16 v47, v31, v3
	;;#ASMEND
	s_branch .LBB0_15
.LBB0_48:                               ;   in Loop: Header=BB0_13 Depth=1
	v_mov_b32_e32 v50, v37
	v_mov_b32_e32 v49, v37
	v_mov_b32_e32 v48, v37
	v_mov_b32_e32 v47, v37
.LBB0_49:                               ;   in Loop: Header=BB0_13 Depth=1
	;;#ASMSTART
	s_nop 0
	v_add_f32 v50, v50, v50 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v49, v49, v49 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v48, v48, v48 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v47, v47, v47 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v50, v50, v50 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v49, v49, v49 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v48, v48, v48 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v47, v47, v47 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v50, v50, v50 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v49, v49, v49 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v48, v48, v48 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v47, v47, v47 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v50, v50, v50 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v49, v49, v49 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v48, v48, v48 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v47, v47, v47 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v50, v50, v50 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v49, v49, v49 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v48, v48, v48 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v47, v47, v47 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v50, v50, v50 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v49, v49, v49 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v48, v48, v48 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v47, v47, v47 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	s_and_saveexec_b64 s[46:47], s[0:1]
	s_cbranch_execz .LBB0_69
; %bb.50:                               ;   in Loop: Header=BB0_13 Depth=1
	v_lshlrev_b64 v[40:41], 1, v[34:35]
	v_lshl_add_u64 v[38:39], s[14:15], 0, v[40:41]
	v_lshl_add_u64 v[40:41], s[12:13], 0, v[40:41]
	s_andn2_b64 vcc, exec, s[40:41]
	v_cmp_ne_u32_e64 s[8:9], 0, v32
	s_cbranch_vccnz .LBB0_60
; %bb.51:                               ;   in Loop: Header=BB0_13 Depth=1
	s_and_saveexec_b64 s[48:49], s[8:9]
	s_cbranch_execz .LBB0_55
; %bb.52:                               ;   in Loop: Header=BB0_13 Depth=1
	s_andn2_b64 vcc, exec, s[28:29]
	v_mul_f32_e32 v36, s20, v50
	s_cbranch_vccnz .LBB0_54
; %bb.53:                               ;   in Loop: Header=BB0_13 Depth=1
	global_load_ushort v50, v[40:41], off
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v50, 16, v50
	v_fmac_f32_e32 v36, s21, v50
.LBB0_54:                               ;   in Loop: Header=BB0_13 Depth=1
	v_cvt_pk_bf16_f32 v36, v36, s0
	global_store_short v[38:39], v36, off
.LBB0_55:                               ;   in Loop: Header=BB0_13 Depth=1
	s_or_b64 exec, exec, s[48:49]
	v_cmp_ne_u32_e32 vcc, 0, v33
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_59
; %bb.56:                               ;   in Loop: Header=BB0_13 Depth=1
	s_andn2_b64 vcc, exec, s[28:29]
	v_mul_f32_e32 v36, s20, v49
	s_cbranch_vccnz .LBB0_58
; %bb.57:                               ;   in Loop: Header=BB0_13 Depth=1
	global_load_ushort v49, v[40:41], off offset:2
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v49, 16, v49
	v_fmac_f32_e32 v36, s21, v49
.LBB0_58:                               ;   in Loop: Header=BB0_13 Depth=1
	v_cvt_pk_bf16_f32 v36, v36, s0
	global_store_short v[38:39], v36, off offset:2
.LBB0_59:                               ;   in Loop: Header=BB0_13 Depth=1
	s_or_b64 exec, exec, s[8:9]
.LBB0_60:                               ;   in Loop: Header=BB0_13 Depth=1
	s_andn2_b64 vcc, exec, s[42:43]
	s_cbranch_vccnz .LBB0_69
; %bb.61:                               ;   in Loop: Header=BB0_13 Depth=1
	v_cndmask_b32_e64 v36, 0, 1, s[28:29]
	v_lshl_add_u64 v[40:41], s[30:31], 1, v[40:41]
	v_lshl_add_u64 v[38:39], s[34:35], 1, v[38:39]
	v_cmp_ne_u32_e32 vcc, 0, v32
	v_cmp_ne_u32_e64 s[8:9], 1, v36
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_65
; %bb.62:                               ;   in Loop: Header=BB0_13 Depth=1
	s_and_b64 vcc, exec, s[8:9]
	v_mul_f32_e32 v36, s20, v48
	s_cbranch_vccnz .LBB0_64
; %bb.63:                               ;   in Loop: Header=BB0_13 Depth=1
	global_load_ushort v48, v[40:41], off
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v48, 16, v48
	v_fmac_f32_e32 v36, s21, v48
.LBB0_64:                               ;   in Loop: Header=BB0_13 Depth=1
	v_cvt_pk_bf16_f32 v36, v36, s0
	global_store_short v[38:39], v36, off
.LBB0_65:                               ;   in Loop: Header=BB0_13 Depth=1
	s_or_b64 exec, exec, s[48:49]
	v_cmp_ne_u32_e32 vcc, 0, v33
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_69
; %bb.66:                               ;   in Loop: Header=BB0_13 Depth=1
	s_and_b64 vcc, exec, s[8:9]
	v_mul_f32_e32 v36, s20, v47
	s_cbranch_vccnz .LBB0_68
; %bb.67:                               ;   in Loop: Header=BB0_13 Depth=1
	global_load_ushort v40, v[40:41], off offset:2
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v40, 16, v40
	v_fmac_f32_e32 v36, s21, v40
.LBB0_68:                               ;   in Loop: Header=BB0_13 Depth=1
	v_cvt_pk_bf16_f32 v36, v36, s0
	global_store_short v[38:39], v36, off offset:2
.LBB0_69:                               ;   in Loop: Header=BB0_13 Depth=1
	s_or_b64 exec, exec, s[46:47]
	v_lshl_add_u64 v[34:35], v[34:35], 0, s[26:27]
	v_lshl_add_u64 v[38:39], v[34:35], 0, 2
	v_cmp_gt_u64_e32 vcc, s[24:25], v[34:35]
	v_cmp_le_u64_e64 s[8:9], s[24:25], v[38:39]
	s_and_b64 s[8:9], vcc, s[8:9]
	s_and_saveexec_b64 s[46:47], s[8:9]
	s_cbranch_execz .LBB0_12
; %bb.70:                               ;   in Loop: Header=BB0_13 Depth=1
	v_cmp_ne_u64_e32 vcc, s[22:23], v[34:35]
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_11
; %bb.71:                               ;   in Loop: Header=BB0_13 Depth=1
	v_subrev_co_u32_e32 v34, vcc, s22, v34
	s_mov_b64 s[50:51], 0
	s_nop 0
	v_subbrev_co_u32_e32 v35, vcc, 0, v35, vcc
	s_mov_b64 s[52:53], 0
.LBB0_72:                               ;   Parent Loop BB0_13 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_cmp_lg_u32 s52, 1
	s_cselect_b64 vcc, -1, 0
	s_cmp_lg_u32 s52, 0
	v_cndmask_b32_e32 v33, 0, v33, vcc
	s_cselect_b64 vcc, -1, 0
	s_add_u32 s52, s52, 1
	s_mov_b32 s18, s52
	s_addc_u32 s53, s53, 0
	v_cmp_ge_u64_e64 s[8:9], s[18:19], v[34:35]
	s_or_b64 s[50:51], s[8:9], s[50:51]
	v_cndmask_b32_e32 v32, 0, v32, vcc
	s_andn2_b64 exec, exec, s[50:51]
	s_cbranch_execnz .LBB0_72
; %bb.73:                               ;   in Loop: Header=BB0_13 Depth=1
	s_or_b64 exec, exec, s[50:51]
	s_branch .LBB0_11
.LBB0_74:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wvSpltK_bf16_tn_m2
		.amdhsa_group_segment_fixed_size 131072
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
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 1
		.amdhsa_next_free_vgpr 97
		.amdhsa_next_free_sgpr 96
		.amdhsa_accum_offset 60
		.amdhsa_reserve_vcc 1
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
	.size	wvSpltK_bf16_tn_m2, .Lfunc_end0-wvSpltK_bf16_tn_m2
                                        ; -- End function
	.set wvSpltK_bf16_tn_m2.num_vgpr, 58
	.set wvSpltK_bf16_tn_m2.num_agpr, 0
	.set wvSpltK_bf16_tn_m2.numbered_sgpr, 54
	.set wvSpltK_bf16_tn_m2.num_named_barrier, 0
	.set wvSpltK_bf16_tn_m2.private_seg_size, 0
	.set wvSpltK_bf16_tn_m2.uses_vcc, 1
	.set wvSpltK_bf16_tn_m2.uses_flat_scratch, 0
	.set wvSpltK_bf16_tn_m2.has_dyn_sized_stack, 0
	.set wvSpltK_bf16_tn_m2.has_recursion, 0
	.set wvSpltK_bf16_tn_m2.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 3528
; TotalNumSgprs: 60
; NumVgprs: 58
; NumAgprs: 0
; TotalNumVgprs: 58
; ScratchSize: 0
; MemoryBound: 1
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 131072 bytes/workgroup (compile time only)
; SGPRBlocks: 12
; VGPRBlocks: 12
; NumSGPRsForWavesPerEU: 102
; NumVGPRsForWavesPerEU: 97
; AccumOffset: 60
; Occupancy: 4
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 1
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 14
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_e30d5cb7b9cce401,@object ; @__hip_cuid_e30d5cb7b9cce401
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_e30d5cb7b9cce401
__hip_cuid_e30d5cb7b9cce401:
	.byte	0                               ; 0x0
	.size	__hip_cuid_e30d5cb7b9cce401, 1

	.ident	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.4 26084 f58b06dce1f9c15707c5f808fd002e18c2accf7e)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_e30d5cb7b9cce401
	.amdgpu_metadata
---
custom.config:
  Source:
    Origin: rocblas
    Repository: "https://github.com/ROCm/rocBLAS-internal"
  Version: 1.0.0
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
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
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
    .group_segment_fixed_size: 131072
    .kernarg_segment_align: 8
    .kernarg_segment_size: 72
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           wvSpltK_bf16_tn_m2
    .private_segment_fixed_size: 0
    .sgpr_count:     60
    .sgpr_spill_count: 0
    .symbol:         wvSpltK_bf16_tn_m2.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     58
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
