	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.protected	wvSpltK_bf16_tn_m4      ; -- Begin function wvSpltK_bf16_tn_m4
	.globl	wvSpltK_bf16_tn_m4
	.p2align	8
	.type	wvSpltK_bf16_tn_m4,@function
wvSpltK_bf16_tn_m4:                     ; @wvSpltK_bf16_tn_m4
; %bb.147:
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.148:
.LBB0_0:
	v_and_b32_e32 v4, 0x3ff, v0
	s_mov_b32 s24, s3
	v_bfe_u32 v1, v0, 10, 10
	s_mul_i32 s3, s9, s2
	v_lshlrev_b32_e32 v94, 3, v4
	s_min_i32 s3, s3, 0x10000
	v_lshl_add_u32 v0, v1, 9, v94
	v_cmp_gt_u32_e32 vcc, s3, v0
	s_and_saveexec_b64 s[18:19], vcc
	s_cbranch_execz .LBB0_3
; %bb.1:
	v_cvt_f32_u32_e32 v2, s2
	s_sub_i32 s22, 0, s2
	v_lshlrev_b32_e32 v3, 4, v4
	s_ashr_i32 s17, s11, 31
	v_rcp_iflag_f32_e32 v2, v2
	v_lshl_add_u32 v3, v1, 10, v3
	s_mov_b64 s[20:21], 0
	v_mul_f32_e32 v2, 0x4f7ffffe, v2
	v_cvt_u32_f32_e32 v5, v2
	v_mov_b32_e32 v2, 0
	v_mul_lo_u32 v6, s22, v5
	v_mul_hi_u32 v6, v5, v6
	v_add_u32_e32 v5, v5, v6
.LBB0_2:                                ; =>This Inner Loop Header: Depth=1
	v_mul_hi_u32 v8, v5, v0
	v_mul_lo_u32 v6, s2, v8
	v_not_b32_e32 v7, v8
	v_sub_u32_e32 v10, v0, v6
	v_add_u32_e32 v9, 1, v8
	v_mad_u64_u32 v[6:7], s[26:27], s2, v7, v[0:1]
	v_cmp_le_u32_e32 vcc, s2, v10
	s_nop 1
	v_cndmask_b32_e32 v7, v8, v9, vcc
	v_cndmask_b32_e32 v6, v10, v6, vcc
	v_add_u32_e32 v8, 1, v7
	v_cmp_le_u32_e32 vcc, s2, v6
	s_nop 1
	v_cndmask_b32_e32 v11, v7, v8, vcc
	v_mad_u64_u32 v[8:9], s[26:27], v11, s11, 0
	v_mov_b32_e32 v10, v9
	v_mad_u64_u32 v[6:7], s[26:27], s22, v11, v[0:1]
	v_mad_u64_u32 v[10:11], s[26:27], v11, s17, v[10:11]
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
	s_cbranch_execnz .LBB0_2
.LBB0_3:
	s_or_b64 exec, exec, s[18:19]
	v_lshl_add_u32 v2, s16, 4, v1
	v_mov_b32_e32 v1, 0
	v_mov_b32_e32 v3, v1
	s_ashr_i32 s25, s24, 31
	v_lshl_add_u64 v[6:7], v[2:3], 0, 1
	v_cmp_le_u64_e32 vcc, s[24:25], v[2:3]
	v_cmp_gt_u64_e64 s[16:17], s[24:25], v[6:7]
	s_add_i32 s26, s24, -1
	v_mov_b32_e32 v0, s26
	s_or_b64 vcc, vcc, s[16:17]
	v_cndmask_b32_e32 v0, v0, v2, vcc
	s_mov_b32 s27, 0
	v_cmp_gt_u64_e64 s[16:17], s[24:25], v[0:1]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_and_saveexec_b64 s[18:19], s[16:17]
	s_cbranch_execz .LBB0_146
; %bb.4:
	s_load_dwordx4 s[20:23], s[0:1], 0x38
	v_cmp_eq_u32_e64 s[0:1], s26, v2
	s_or_b64 s[0:1], vcc, s[0:1]
	s_lshl_b32 s28, s8, 4
	v_cndmask_b32_e64 v95, 0, 1, s[0:1]
	s_add_i32 s0, s9, -1
	s_min_i32 s18, s0, 3
	s_min_i32 s19, s0, 2
	s_min_i32 s33, s0, 1
	s_min_i32 s38, s0, 0
	s_cmp_lg_u32 s2, 0
	s_cselect_b64 s[16:17], -1, 0
	s_ashr_i32 s8, s11, 31
	s_mul_i32 s39, s38, s8
	s_mul_hi_u32 s40, s38, s11
	s_add_i32 s39, s40, s39
	s_mul_i32 s40, s33, s8
	s_mul_hi_u32 s41, s33, s11
	s_mul_i32 s42, s19, s8
	s_mul_hi_u32 s43, s19, s11
	s_mul_i32 s8, s18, s8
	s_mul_hi_u32 s44, s18, s11
	s_ashr_i32 s3, s10, 31
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s35, s22, 31
	s_ashr_i32 s37, s23, 31
	s_ashr_i32 s29, s28, 31
	s_add_i32 s41, s41, s40
	s_add_i32 s43, s43, s42
	s_add_i32 s45, s44, s8
	s_cmp_gt_i32 s9, 0
	s_cselect_b64 s[46:47], -1, 0
	s_cmp_gt_i32 s9, 1
	s_mov_b32 s34, s22
	s_mov_b32 s36, s23
	s_cselect_b64 s[48:49], -1, 0
	s_cmp_gt_i32 s9, 2
	s_mul_i32 s62, s38, s2
	s_mul_i32 s63, s33, s2
	s_mul_i32 s40, s33, s11
	s_mul_i32 s33, s19, s2
	s_mul_i32 s42, s19, s11
	s_mul_i32 s19, s18, s2
	s_cselect_b64 s[50:51], -1, 0
	s_lshl_b64 s[52:53], s[34:35], 2
	s_lshl_b64 s[54:55], s[36:37], 2
	v_lshlrev_b32_e32 v2, 4, v4
	s_cmp_gt_i32 s9, 3
	v_lshl_add_u32 v96, s62, 1, v2
	v_lshl_add_u32 v98, s63, 1, v2
	v_lshl_add_u32 v100, s33, 1, v2
	v_lshl_add_u32 v102, s19, 1, v2
	v_cndmask_b32_e64 v2, 0, 1, s[16:17]
	v_cmp_eq_u32_e64 s[0:1], 63, v4
	v_cmp_neq_f32_e64 s[30:31], s21, 0
	s_mul_i32 s38, s38, s11
	s_mul_i32 s44, s18, s11
	s_cselect_b64 s[56:57], -1, 0
	s_mul_hi_i32 s59, s22, 6
	s_mul_i32 s58, s22, 6
	s_mul_hi_i32 s61, s23, 6
	s_mul_i32 s60, s23, 6
	v_add_u32_e32 v97, s62, v94
	v_add_u32_e32 v99, s63, v94
	v_add_u32_e32 v101, s33, v94
	v_add_u32_e32 v103, s19, v94
	s_mov_b64 s[22:23], 0
	v_cmp_ne_u32_e64 s[16:17], 1, v2
	s_mov_b32 s11, 0xffff
	v_mov_b64_e32 v[88:89], v[0:1]
                                        ; implicit-def: $vgpr4_vgpr5_vgpr6_vgpr7
                                        ; implicit-def: $vgpr8_vgpr9_vgpr10_vgpr11
                                        ; implicit-def: $vgpr12_vgpr13_vgpr14_vgpr15
                                        ; implicit-def: $vgpr16_vgpr17_vgpr18_vgpr19
                                        ; implicit-def: $vgpr22_vgpr23
                                        ; implicit-def: $vgpr38_vgpr39
                                        ; implicit-def: $vgpr54_vgpr55
                                        ; implicit-def: $vgpr70_vgpr71
                                        ; implicit-def: $vgpr26_vgpr27
                                        ; implicit-def: $vgpr42_vgpr43
                                        ; implicit-def: $vgpr58_vgpr59
                                        ; implicit-def: $vgpr74_vgpr75
                                        ; implicit-def: $vgpr30_vgpr31
                                        ; implicit-def: $vgpr46_vgpr47
                                        ; implicit-def: $vgpr62_vgpr63
                                        ; implicit-def: $vgpr78_vgpr79
                                        ; implicit-def: $vgpr34_vgpr35
                                        ; implicit-def: $vgpr50_vgpr51
                                        ; implicit-def: $vgpr66_vgpr67
                                        ; implicit-def: $vgpr82_vgpr83
	s_branch .LBB0_7
.LBB0_5:                                ;   in Loop: Header=BB0_7 Depth=1
	v_lshl_add_u64 v[2:3], v[2:3], 0, s[60:61]
	v_cvt_pk_bf16_f32 v0, v0, s0
	global_store_short v[2:3], v0, off
.LBB0_6:                                ;   in Loop: Header=BB0_7 Depth=1
	s_or_b64 exec, exec, s[18:19]
	v_lshl_add_u64 v[2:3], v[88:89], 0, s[28:29]
	v_lshl_add_u64 v[84:85], v[2:3], 0, 1
	v_cmp_le_u64_e32 vcc, s[24:25], v[2:3]
	v_cmp_gt_u64_e64 s[8:9], s[24:25], v[84:85]
	s_or_b64 vcc, vcc, s[8:9]
	v_mov_b32_e32 v0, s26
	v_cmp_eq_u64_e64 s[18:19], s[26:27], v[2:3]
	v_cndmask_b32_e32 v89, 0, v3, vcc
	v_cndmask_b32_e32 v88, v0, v2, vcc
	s_or_b64 s[8:9], vcc, s[18:19]
	v_cmp_le_u64_e32 vcc, s[24:25], v[88:89]
	s_or_b64 s[22:23], vcc, s[22:23]
	v_cndmask_b32_e64 v95, 0, v95, s[8:9]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execz .LBB0_146
.LBB0_7:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_13 Depth 2
	s_and_b64 vcc, exec, s[16:17]
	s_cbranch_vccnz .LBB0_122
; %bb.8:                                ;   in Loop: Header=BB0_7 Depth=1
	v_mul_lo_u32 v0, v89, s10
	v_mul_lo_u32 v84, v88, s3
	v_mad_u64_u32 v[2:3], s[8:9], v88, s10, 0
	v_add3_u32 v3, v3, v84, v0
	v_lshl_add_u64 v[90:91], v[2:3], 1, s[4:5]
	v_mov_b32_e32 v2, v1
	v_mov_b32_e32 v3, v1
	v_mov_b32_e32 v0, v1
	v_mov_b64_e32 v[86:87], v[2:3]
	s_mov_b32 s33, 0
	v_mov_b32_e32 v104, v102
	v_mov_b32_e32 v105, v100
	v_mov_b32_e32 v106, v98
	v_mov_b32_e32 v107, v96
	v_mov_b64_e32 v[84:85], v[0:1]
	s_branch .LBB0_13
.LBB0_9:                                ;   in Loop: Header=BB0_13 Depth=2
	s_or_b64 exec, exec, s[64:65]
.LBB0_10:                               ;   in Loop: Header=BB0_13 Depth=2
	s_or_b64 exec, exec, s[62:63]
.LBB0_11:                               ;   in Loop: Header=BB0_13 Depth=2
	s_or_b64 exec, exec, s[18:19]
.LBB0_12:                               ;   in Loop: Header=BB0_13 Depth=2
	s_or_b64 exec, exec, s[8:9]
	s_addk_i32 s33, 0x800
	v_add_u32_e32 v107, 0x1000, v107
	v_add_u32_e32 v106, 0x1000, v106
	v_add_u32_e32 v105, 0x1000, v105
	s_cmp_ge_u32 s33, s2
	v_add_u32_e32 v104, 0x1000, v104
	s_cbranch_scc1 .LBB0_123
.LBB0_13:                               ;   Parent Loop BB0_7 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_add_u32_e32 v0, s33, v94
	v_cmp_gt_u32_e32 vcc, s2, v0
	v_add_u32_e32 v2, 0x200, v0
	s_and_saveexec_b64 s[18:19], vcc
	s_cbranch_execnz .LBB0_19
; %bb.14:                               ;   in Loop: Header=BB0_13 Depth=2
	s_or_b64 exec, exec, s[18:19]
	s_and_saveexec_b64 s[18:19], vcc
	s_cbranch_execnz .LBB0_26
.LBB0_15:                               ;   in Loop: Header=BB0_13 Depth=2
	s_or_b64 exec, exec, s[18:19]
	s_and_saveexec_b64 s[18:19], vcc
	s_cbranch_execnz .LBB0_97
.LBB0_16:                               ;   in Loop: Header=BB0_13 Depth=2
	s_or_b64 exec, exec, s[18:19]
	s_and_saveexec_b64 s[18:19], vcc
	s_cbranch_execnz .LBB0_104
.LBB0_17:                               ;   in Loop: Header=BB0_13 Depth=2
	s_or_b64 exec, exec, s[18:19]
	s_and_saveexec_b64 s[18:19], vcc
	s_cbranch_execnz .LBB0_111
.LBB0_18:                               ;   in Loop: Header=BB0_13 Depth=2
	s_or_b64 exec, exec, s[18:19]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_12
	s_branch .LBB0_118
.LBB0_19:                               ;   in Loop: Header=BB0_13 Depth=2
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[16:17], v[0:1], 1, v[90:91]
	global_load_dwordx4 v[16:19], v[16:17], off nt
	v_cmp_gt_u32_e64 s[8:9], s2, v2
	s_and_saveexec_b64 s[62:63], s[8:9]
	s_cbranch_execz .LBB0_25
; %bb.20:                               ;   in Loop: Header=BB0_13 Depth=2
	v_mov_b32_e32 v3, v1
	v_lshl_add_u64 v[12:13], v[2:3], 1, v[90:91]
	global_load_dwordx4 v[12:15], v[12:13], off nt
	v_add_u32_e32 v92, 0x400, v0
	v_cmp_gt_u32_e64 s[8:9], s2, v92
	s_and_saveexec_b64 s[64:65], s[8:9]
	s_cbranch_execz .LBB0_24
; %bb.21:                               ;   in Loop: Header=BB0_13 Depth=2
	v_mov_b32_e32 v93, v1
	v_lshl_add_u64 v[8:9], v[92:93], 1, v[90:91]
	global_load_dwordx4 v[8:11], v[8:9], off nt
	v_add_u32_e32 v92, 0x600, v0
	v_cmp_gt_u32_e64 s[8:9], s2, v92
	s_and_saveexec_b64 s[66:67], s[8:9]
	s_cbranch_execz .LBB0_23
; %bb.22:                               ;   in Loop: Header=BB0_13 Depth=2
	v_mov_b32_e32 v93, v1
	v_lshl_add_u64 v[4:5], v[92:93], 1, v[90:91]
	global_load_dwordx4 v[4:7], v[4:5], off nt
.LBB0_23:                               ;   in Loop: Header=BB0_13 Depth=2
	s_or_b64 exec, exec, s[66:67]
.LBB0_24:                               ;   in Loop: Header=BB0_13 Depth=2
	s_or_b64 exec, exec, s[64:65]
.LBB0_25:                               ;   in Loop: Header=BB0_13 Depth=2
	s_or_b64 exec, exec, s[62:63]
	s_or_b64 exec, exec, s[18:19]
	s_and_saveexec_b64 s[18:19], vcc
	s_cbranch_execz .LBB0_15
.LBB0_26:                               ;   in Loop: Header=BB0_13 Depth=2
	v_add_u32_e32 v108, s33, v97
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshl_add_u64 v[70:71], v[0:1], 1, s[6:7]
	v_cmp_lt_u32_e64 s[8:9], s11, v108
                                        ; implicit-def: $vgpr20_vgpr21
	s_and_saveexec_b64 s[62:63], s[8:9]
	s_xor_b64 s[8:9], exec, s[62:63]
	s_cbranch_execz .LBB0_28
; %bb.27:                               ;   in Loop: Header=BB0_13 Depth=2
	v_lshl_add_u64 v[20:21], s[38:39], 1, v[70:71]
	global_load_dwordx4 v[20:23], v[20:21], off
.LBB0_28:                               ;   in Loop: Header=BB0_13 Depth=2
	s_andn2_saveexec_b64 s[8:9], s[8:9]
	s_cbranch_execz .LBB0_30
; %bb.29:                               ;   in Loop: Header=BB0_13 Depth=2
	s_waitcnt vmcnt(0)
	ds_read_b128 v[20:23], v107
.LBB0_30:                               ;   in Loop: Header=BB0_13 Depth=2
	s_or_b64 exec, exec, s[8:9]
	v_add_u32_e32 v109, s33, v99
	v_cmp_lt_u32_e64 s[8:9], s11, v109
                                        ; implicit-def: $vgpr36_vgpr37
	s_and_saveexec_b64 s[62:63], s[8:9]
	s_xor_b64 s[8:9], exec, s[62:63]
	s_cbranch_execz .LBB0_32
; %bb.31:                               ;   in Loop: Header=BB0_13 Depth=2
	v_lshl_add_u64 v[36:37], s[40:41], 1, v[70:71]
	global_load_dwordx4 v[36:39], v[36:37], off
.LBB0_32:                               ;   in Loop: Header=BB0_13 Depth=2
	s_andn2_saveexec_b64 s[8:9], s[8:9]
	s_cbranch_execz .LBB0_34
; %bb.33:                               ;   in Loop: Header=BB0_13 Depth=2
	s_waitcnt vmcnt(0)
	ds_read_b128 v[36:39], v106
.LBB0_34:                               ;   in Loop: Header=BB0_13 Depth=2
	s_or_b64 exec, exec, s[8:9]
	v_add_u32_e32 v110, s33, v101
	v_cmp_lt_u32_e64 s[8:9], s11, v110
                                        ; implicit-def: $vgpr52_vgpr53
	s_and_saveexec_b64 s[62:63], s[8:9]
	s_xor_b64 s[8:9], exec, s[62:63]
	s_cbranch_execz .LBB0_36
; %bb.35:                               ;   in Loop: Header=BB0_13 Depth=2
	v_lshl_add_u64 v[52:53], s[42:43], 1, v[70:71]
	global_load_dwordx4 v[52:55], v[52:53], off
.LBB0_36:                               ;   in Loop: Header=BB0_13 Depth=2
	s_andn2_saveexec_b64 s[8:9], s[8:9]
	s_cbranch_execz .LBB0_38
; %bb.37:                               ;   in Loop: Header=BB0_13 Depth=2
	s_waitcnt vmcnt(0)
	ds_read_b128 v[52:55], v105
.LBB0_38:                               ;   in Loop: Header=BB0_13 Depth=2
	s_or_b64 exec, exec, s[8:9]
	v_add_u32_e32 v111, s33, v103
	v_cmp_lt_u32_e64 s[8:9], s11, v111
                                        ; implicit-def: $vgpr68_vgpr69
	s_and_saveexec_b64 s[62:63], s[8:9]
	s_xor_b64 s[8:9], exec, s[62:63]
	s_cbranch_execnz .LBB0_41
; %bb.39:                               ;   in Loop: Header=BB0_13 Depth=2
	s_andn2_saveexec_b64 s[8:9], s[8:9]
	s_cbranch_execnz .LBB0_42
.LBB0_40:                               ;   in Loop: Header=BB0_13 Depth=2
	s_or_b64 exec, exec, s[8:9]
	v_cmp_gt_u32_e64 s[8:9], s2, v2
	s_and_saveexec_b64 s[62:63], s[8:9]
	s_cbranch_execnz .LBB0_43
	s_branch .LBB0_96
.LBB0_41:                               ;   in Loop: Header=BB0_13 Depth=2
	v_lshl_add_u64 v[68:69], s[44:45], 1, v[70:71]
	global_load_dwordx4 v[68:71], v[68:69], off
	s_andn2_saveexec_b64 s[8:9], s[8:9]
	s_cbranch_execz .LBB0_40
.LBB0_42:                               ;   in Loop: Header=BB0_13 Depth=2
	s_waitcnt vmcnt(0)
	ds_read_b128 v[68:71], v104
	s_or_b64 exec, exec, s[8:9]
	v_cmp_gt_u32_e64 s[8:9], s2, v2
	s_and_saveexec_b64 s[62:63], s[8:9]
	s_cbranch_execz .LBB0_96
.LBB0_43:                               ;   in Loop: Header=BB0_13 Depth=2
	v_mov_b32_e32 v3, v1
	v_lshl_add_u64 v[74:75], v[2:3], 1, s[6:7]
	v_add_u32_e32 v3, 0x200, v108
	v_cmp_lt_u32_e64 s[8:9], s11, v3
                                        ; implicit-def: $vgpr24_vgpr25
	s_and_saveexec_b64 s[64:65], s[8:9]
	s_xor_b64 s[8:9], exec, s[64:65]
	s_cbranch_execz .LBB0_45
; %bb.44:                               ;   in Loop: Header=BB0_13 Depth=2
	v_lshl_add_u64 v[24:25], s[38:39], 1, v[74:75]
	global_load_dwordx4 v[24:27], v[24:25], off
.LBB0_45:                               ;   in Loop: Header=BB0_13 Depth=2
	s_andn2_saveexec_b64 s[8:9], s[8:9]
	s_cbranch_execz .LBB0_47
; %bb.46:                               ;   in Loop: Header=BB0_13 Depth=2
	s_waitcnt vmcnt(0)
	ds_read_b128 v[24:27], v107 offset:1024
.LBB0_47:                               ;   in Loop: Header=BB0_13 Depth=2
	s_or_b64 exec, exec, s[8:9]
	v_add_u32_e32 v3, 0x200, v109
	v_cmp_lt_u32_e64 s[8:9], s11, v3
                                        ; implicit-def: $vgpr40_vgpr41
	s_and_saveexec_b64 s[64:65], s[8:9]
	s_xor_b64 s[8:9], exec, s[64:65]
	s_cbranch_execz .LBB0_49
; %bb.48:                               ;   in Loop: Header=BB0_13 Depth=2
	v_lshl_add_u64 v[40:41], s[40:41], 1, v[74:75]
	global_load_dwordx4 v[40:43], v[40:41], off
.LBB0_49:                               ;   in Loop: Header=BB0_13 Depth=2
	s_andn2_saveexec_b64 s[8:9], s[8:9]
	s_cbranch_execz .LBB0_51
; %bb.50:                               ;   in Loop: Header=BB0_13 Depth=2
	s_waitcnt vmcnt(0)
	ds_read_b128 v[40:43], v106 offset:1024
.LBB0_51:                               ;   in Loop: Header=BB0_13 Depth=2
	s_or_b64 exec, exec, s[8:9]
	v_add_u32_e32 v3, 0x200, v110
	v_cmp_lt_u32_e64 s[8:9], s11, v3
                                        ; implicit-def: $vgpr56_vgpr57
	s_and_saveexec_b64 s[64:65], s[8:9]
	s_xor_b64 s[8:9], exec, s[64:65]
	s_cbranch_execz .LBB0_53
; %bb.52:                               ;   in Loop: Header=BB0_13 Depth=2
	v_lshl_add_u64 v[56:57], s[42:43], 1, v[74:75]
	global_load_dwordx4 v[56:59], v[56:57], off
.LBB0_53:                               ;   in Loop: Header=BB0_13 Depth=2
	s_andn2_saveexec_b64 s[8:9], s[8:9]
	s_cbranch_execz .LBB0_55
; %bb.54:                               ;   in Loop: Header=BB0_13 Depth=2
	s_waitcnt vmcnt(0)
	ds_read_b128 v[56:59], v105 offset:1024
.LBB0_55:                               ;   in Loop: Header=BB0_13 Depth=2
	s_or_b64 exec, exec, s[8:9]
	v_add_u32_e32 v3, 0x200, v111
	v_cmp_lt_u32_e64 s[8:9], s11, v3
                                        ; implicit-def: $vgpr72_vgpr73
	s_and_saveexec_b64 s[64:65], s[8:9]
	s_xor_b64 s[8:9], exec, s[64:65]
	s_cbranch_execz .LBB0_57
; %bb.56:                               ;   in Loop: Header=BB0_13 Depth=2
	v_lshl_add_u64 v[72:73], s[44:45], 1, v[74:75]
	global_load_dwordx4 v[72:75], v[72:73], off
.LBB0_57:                               ;   in Loop: Header=BB0_13 Depth=2
	s_andn2_saveexec_b64 s[8:9], s[8:9]
	s_cbranch_execz .LBB0_59
; %bb.58:                               ;   in Loop: Header=BB0_13 Depth=2
	s_waitcnt vmcnt(0)
	ds_read_b128 v[72:75], v104 offset:1024
.LBB0_59:                               ;   in Loop: Header=BB0_13 Depth=2
	s_or_b64 exec, exec, s[8:9]
	v_add_u32_e32 v92, 0x400, v0
	v_cmp_gt_u32_e64 s[8:9], s2, v92
	s_and_saveexec_b64 s[64:65], s[8:9]
	s_cbranch_execz .LBB0_95
; %bb.60:                               ;   in Loop: Header=BB0_13 Depth=2
	v_mov_b32_e32 v93, v1
	v_add_u32_e32 v3, 0x400, v108
	v_lshl_add_u64 v[78:79], v[92:93], 1, s[6:7]
	v_cmp_lt_u32_e64 s[8:9], s11, v3
                                        ; implicit-def: $vgpr28_vgpr29
	s_and_saveexec_b64 s[66:67], s[8:9]
	s_xor_b64 s[8:9], exec, s[66:67]
	s_cbranch_execz .LBB0_62
; %bb.61:                               ;   in Loop: Header=BB0_13 Depth=2
	v_lshl_add_u64 v[28:29], s[38:39], 1, v[78:79]
	global_load_dwordx4 v[28:31], v[28:29], off
.LBB0_62:                               ;   in Loop: Header=BB0_13 Depth=2
	s_andn2_saveexec_b64 s[8:9], s[8:9]
	s_cbranch_execz .LBB0_64
; %bb.63:                               ;   in Loop: Header=BB0_13 Depth=2
	s_waitcnt vmcnt(0)
	ds_read_b128 v[28:31], v107 offset:2048
.LBB0_64:                               ;   in Loop: Header=BB0_13 Depth=2
	s_or_b64 exec, exec, s[8:9]
	v_add_u32_e32 v3, 0x400, v109
	v_cmp_lt_u32_e64 s[8:9], s11, v3
                                        ; implicit-def: $vgpr44_vgpr45
	s_and_saveexec_b64 s[66:67], s[8:9]
	s_xor_b64 s[8:9], exec, s[66:67]
	s_cbranch_execz .LBB0_66
; %bb.65:                               ;   in Loop: Header=BB0_13 Depth=2
	v_lshl_add_u64 v[44:45], s[40:41], 1, v[78:79]
	global_load_dwordx4 v[44:47], v[44:45], off
.LBB0_66:                               ;   in Loop: Header=BB0_13 Depth=2
	s_andn2_saveexec_b64 s[8:9], s[8:9]
	s_cbranch_execz .LBB0_68
; %bb.67:                               ;   in Loop: Header=BB0_13 Depth=2
	s_waitcnt vmcnt(0)
	ds_read_b128 v[44:47], v106 offset:2048
.LBB0_68:                               ;   in Loop: Header=BB0_13 Depth=2
	s_or_b64 exec, exec, s[8:9]
	v_add_u32_e32 v3, 0x400, v110
	v_cmp_lt_u32_e64 s[8:9], s11, v3
                                        ; implicit-def: $vgpr60_vgpr61
	s_and_saveexec_b64 s[66:67], s[8:9]
	s_xor_b64 s[8:9], exec, s[66:67]
	s_cbranch_execz .LBB0_70
; %bb.69:                               ;   in Loop: Header=BB0_13 Depth=2
	v_lshl_add_u64 v[60:61], s[42:43], 1, v[78:79]
	global_load_dwordx4 v[60:63], v[60:61], off
.LBB0_70:                               ;   in Loop: Header=BB0_13 Depth=2
	s_andn2_saveexec_b64 s[8:9], s[8:9]
	s_cbranch_execz .LBB0_72
; %bb.71:                               ;   in Loop: Header=BB0_13 Depth=2
	s_waitcnt vmcnt(0)
	ds_read_b128 v[60:63], v105 offset:2048
.LBB0_72:                               ;   in Loop: Header=BB0_13 Depth=2
	s_or_b64 exec, exec, s[8:9]
	v_add_u32_e32 v3, 0x400, v111
	v_cmp_lt_u32_e64 s[8:9], s11, v3
                                        ; implicit-def: $vgpr76_vgpr77
	s_and_saveexec_b64 s[66:67], s[8:9]
	s_xor_b64 s[8:9], exec, s[66:67]
	s_cbranch_execz .LBB0_74
; %bb.73:                               ;   in Loop: Header=BB0_13 Depth=2
	v_lshl_add_u64 v[76:77], s[44:45], 1, v[78:79]
	global_load_dwordx4 v[76:79], v[76:77], off
.LBB0_74:                               ;   in Loop: Header=BB0_13 Depth=2
	s_andn2_saveexec_b64 s[8:9], s[8:9]
	s_cbranch_execz .LBB0_76
; %bb.75:                               ;   in Loop: Header=BB0_13 Depth=2
	s_waitcnt vmcnt(0)
	ds_read_b128 v[76:79], v104 offset:2048
.LBB0_76:                               ;   in Loop: Header=BB0_13 Depth=2
	s_or_b64 exec, exec, s[8:9]
	v_add_u32_e32 v92, 0x600, v0
	v_cmp_gt_u32_e64 s[8:9], s2, v92
	s_and_saveexec_b64 s[66:67], s[8:9]
	s_cbranch_execz .LBB0_94
; %bb.77:                               ;   in Loop: Header=BB0_13 Depth=2
	v_mov_b32_e32 v93, v1
	v_add_u32_e32 v3, 0x600, v108
	v_lshl_add_u64 v[82:83], v[92:93], 1, s[6:7]
	v_cmp_lt_u32_e64 s[8:9], s11, v3
                                        ; implicit-def: $vgpr32_vgpr33
	s_and_saveexec_b64 s[68:69], s[8:9]
	s_xor_b64 s[8:9], exec, s[68:69]
	s_cbranch_execz .LBB0_79
; %bb.78:                               ;   in Loop: Header=BB0_13 Depth=2
	v_lshl_add_u64 v[32:33], s[38:39], 1, v[82:83]
	global_load_dwordx4 v[32:35], v[32:33], off
.LBB0_79:                               ;   in Loop: Header=BB0_13 Depth=2
	s_andn2_saveexec_b64 s[8:9], s[8:9]
	s_cbranch_execz .LBB0_81
; %bb.80:                               ;   in Loop: Header=BB0_13 Depth=2
	s_waitcnt vmcnt(0)
	ds_read_b128 v[32:35], v107 offset:3072
.LBB0_81:                               ;   in Loop: Header=BB0_13 Depth=2
	s_or_b64 exec, exec, s[8:9]
	v_add_u32_e32 v3, 0x600, v109
	v_cmp_lt_u32_e64 s[8:9], s11, v3
                                        ; implicit-def: $vgpr48_vgpr49
	s_and_saveexec_b64 s[68:69], s[8:9]
	s_xor_b64 s[8:9], exec, s[68:69]
	s_cbranch_execz .LBB0_83
; %bb.82:                               ;   in Loop: Header=BB0_13 Depth=2
	v_lshl_add_u64 v[48:49], s[40:41], 1, v[82:83]
	global_load_dwordx4 v[48:51], v[48:49], off
.LBB0_83:                               ;   in Loop: Header=BB0_13 Depth=2
	s_andn2_saveexec_b64 s[8:9], s[8:9]
	s_cbranch_execz .LBB0_85
; %bb.84:                               ;   in Loop: Header=BB0_13 Depth=2
	s_waitcnt vmcnt(0)
	ds_read_b128 v[48:51], v106 offset:3072
.LBB0_85:                               ;   in Loop: Header=BB0_13 Depth=2
	s_or_b64 exec, exec, s[8:9]
	v_add_u32_e32 v3, 0x600, v110
	v_cmp_lt_u32_e64 s[8:9], s11, v3
                                        ; implicit-def: $vgpr64_vgpr65
	s_and_saveexec_b64 s[68:69], s[8:9]
	s_xor_b64 s[8:9], exec, s[68:69]
	s_cbranch_execz .LBB0_87
; %bb.86:                               ;   in Loop: Header=BB0_13 Depth=2
	v_lshl_add_u64 v[64:65], s[42:43], 1, v[82:83]
	global_load_dwordx4 v[64:67], v[64:65], off
.LBB0_87:                               ;   in Loop: Header=BB0_13 Depth=2
	s_andn2_saveexec_b64 s[8:9], s[8:9]
	s_cbranch_execz .LBB0_89
; %bb.88:                               ;   in Loop: Header=BB0_13 Depth=2
	s_waitcnt vmcnt(0)
	ds_read_b128 v[64:67], v105 offset:3072
.LBB0_89:                               ;   in Loop: Header=BB0_13 Depth=2
	s_or_b64 exec, exec, s[8:9]
	v_add_u32_e32 v3, 0x600, v111
	v_cmp_lt_u32_e64 s[8:9], s11, v3
                                        ; implicit-def: $vgpr80_vgpr81
	s_and_saveexec_b64 s[68:69], s[8:9]
	s_xor_b64 s[8:9], exec, s[68:69]
	s_cbranch_execz .LBB0_91
; %bb.90:                               ;   in Loop: Header=BB0_13 Depth=2
	v_lshl_add_u64 v[80:81], s[44:45], 1, v[82:83]
	global_load_dwordx4 v[80:83], v[80:81], off
.LBB0_91:                               ;   in Loop: Header=BB0_13 Depth=2
	s_andn2_saveexec_b64 s[8:9], s[8:9]
	s_cbranch_execz .LBB0_93
; %bb.92:                               ;   in Loop: Header=BB0_13 Depth=2
	s_waitcnt vmcnt(0)
	ds_read_b128 v[80:83], v104 offset:3072
.LBB0_93:                               ;   in Loop: Header=BB0_13 Depth=2
	s_or_b64 exec, exec, s[8:9]
.LBB0_94:                               ;   in Loop: Header=BB0_13 Depth=2
	s_or_b64 exec, exec, s[66:67]
.LBB0_95:                               ;   in Loop: Header=BB0_13 Depth=2
	s_or_b64 exec, exec, s[64:65]
.LBB0_96:                               ;   in Loop: Header=BB0_13 Depth=2
	s_or_b64 exec, exec, s[62:63]
	s_or_b64 exec, exec, s[18:19]
	s_and_saveexec_b64 s[18:19], vcc
	s_cbranch_execz .LBB0_16
.LBB0_97:                               ;   in Loop: Header=BB0_13 Depth=2
	s_waitcnt vmcnt(0) lgkmcnt(0)
	;;#ASMSTART
	v_dot2c_f32_bf16 v84, v20, v16
	;;#ASMEND
	v_cmp_gt_u32_e64 s[8:9], s2, v2
	;;#ASMSTART
	v_dot2c_f32_bf16 v84, v21, v17
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_bf16 v84, v22, v18
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_bf16 v84, v23, v19
	;;#ASMEND
	s_and_saveexec_b64 s[62:63], s[8:9]
	s_cbranch_execz .LBB0_103
; %bb.98:                               ;   in Loop: Header=BB0_13 Depth=2
	;;#ASMSTART
	v_dot2c_f32_bf16 v84, v24, v12
	;;#ASMEND
	v_add_u32_e32 v3, 0x400, v0
	;;#ASMSTART
	v_dot2c_f32_bf16 v84, v25, v13
	;;#ASMEND
	v_cmp_gt_u32_e64 s[8:9], s2, v3
	;;#ASMSTART
	v_dot2c_f32_bf16 v84, v26, v14
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_bf16 v84, v27, v15
	;;#ASMEND
	s_and_saveexec_b64 s[64:65], s[8:9]
	s_cbranch_execz .LBB0_102
; %bb.99:                               ;   in Loop: Header=BB0_13 Depth=2
	;;#ASMSTART
	v_dot2c_f32_bf16 v84, v28, v8
	;;#ASMEND
	v_add_u32_e32 v3, 0x600, v0
	;;#ASMSTART
	v_dot2c_f32_bf16 v84, v29, v9
	;;#ASMEND
	v_cmp_gt_u32_e64 s[8:9], s2, v3
	;;#ASMSTART
	v_dot2c_f32_bf16 v84, v30, v10
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_bf16 v84, v31, v11
	;;#ASMEND
	s_and_saveexec_b64 s[66:67], s[8:9]
	s_cbranch_execz .LBB0_101
; %bb.100:                              ;   in Loop: Header=BB0_13 Depth=2
	;;#ASMSTART
	v_dot2c_f32_bf16 v84, v32, v4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_bf16 v84, v33, v5
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_bf16 v84, v34, v6
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_bf16 v84, v35, v7
	;;#ASMEND
.LBB0_101:                              ;   in Loop: Header=BB0_13 Depth=2
	s_or_b64 exec, exec, s[66:67]
.LBB0_102:                              ;   in Loop: Header=BB0_13 Depth=2
	s_or_b64 exec, exec, s[64:65]
.LBB0_103:                              ;   in Loop: Header=BB0_13 Depth=2
	s_or_b64 exec, exec, s[62:63]
	s_or_b64 exec, exec, s[18:19]
	s_and_saveexec_b64 s[18:19], vcc
	s_cbranch_execz .LBB0_17
.LBB0_104:                              ;   in Loop: Header=BB0_13 Depth=2
	s_waitcnt vmcnt(0) lgkmcnt(0)
	;;#ASMSTART
	v_dot2c_f32_bf16 v85, v36, v16
	;;#ASMEND
	v_cmp_gt_u32_e64 s[8:9], s2, v2
	;;#ASMSTART
	v_dot2c_f32_bf16 v85, v37, v17
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_bf16 v85, v38, v18
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_bf16 v85, v39, v19
	;;#ASMEND
	s_and_saveexec_b64 s[62:63], s[8:9]
	s_cbranch_execz .LBB0_110
; %bb.105:                              ;   in Loop: Header=BB0_13 Depth=2
	;;#ASMSTART
	v_dot2c_f32_bf16 v85, v40, v12
	;;#ASMEND
	v_add_u32_e32 v3, 0x400, v0
	;;#ASMSTART
	v_dot2c_f32_bf16 v85, v41, v13
	;;#ASMEND
	v_cmp_gt_u32_e64 s[8:9], s2, v3
	;;#ASMSTART
	v_dot2c_f32_bf16 v85, v42, v14
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_bf16 v85, v43, v15
	;;#ASMEND
	s_and_saveexec_b64 s[64:65], s[8:9]
	s_cbranch_execz .LBB0_109
; %bb.106:                              ;   in Loop: Header=BB0_13 Depth=2
	;;#ASMSTART
	v_dot2c_f32_bf16 v85, v44, v8
	;;#ASMEND
	v_add_u32_e32 v3, 0x600, v0
	;;#ASMSTART
	v_dot2c_f32_bf16 v85, v45, v9
	;;#ASMEND
	v_cmp_gt_u32_e64 s[8:9], s2, v3
	;;#ASMSTART
	v_dot2c_f32_bf16 v85, v46, v10
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_bf16 v85, v47, v11
	;;#ASMEND
	s_and_saveexec_b64 s[66:67], s[8:9]
	s_cbranch_execz .LBB0_108
; %bb.107:                              ;   in Loop: Header=BB0_13 Depth=2
	;;#ASMSTART
	v_dot2c_f32_bf16 v85, v48, v4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_bf16 v85, v49, v5
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_bf16 v85, v50, v6
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_bf16 v85, v51, v7
	;;#ASMEND
.LBB0_108:                              ;   in Loop: Header=BB0_13 Depth=2
	s_or_b64 exec, exec, s[66:67]
.LBB0_109:                              ;   in Loop: Header=BB0_13 Depth=2
	s_or_b64 exec, exec, s[64:65]
.LBB0_110:                              ;   in Loop: Header=BB0_13 Depth=2
	s_or_b64 exec, exec, s[62:63]
	s_or_b64 exec, exec, s[18:19]
	s_and_saveexec_b64 s[18:19], vcc
	s_cbranch_execz .LBB0_18
.LBB0_111:                              ;   in Loop: Header=BB0_13 Depth=2
	s_waitcnt vmcnt(0) lgkmcnt(0)
	;;#ASMSTART
	v_dot2c_f32_bf16 v86, v52, v16
	;;#ASMEND
	v_cmp_gt_u32_e64 s[8:9], s2, v2
	;;#ASMSTART
	v_dot2c_f32_bf16 v86, v53, v17
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_bf16 v86, v54, v18
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_bf16 v86, v55, v19
	;;#ASMEND
	s_and_saveexec_b64 s[62:63], s[8:9]
	s_cbranch_execz .LBB0_117
; %bb.112:                              ;   in Loop: Header=BB0_13 Depth=2
	;;#ASMSTART
	v_dot2c_f32_bf16 v86, v56, v12
	;;#ASMEND
	v_add_u32_e32 v3, 0x400, v0
	;;#ASMSTART
	v_dot2c_f32_bf16 v86, v57, v13
	;;#ASMEND
	v_cmp_gt_u32_e64 s[8:9], s2, v3
	;;#ASMSTART
	v_dot2c_f32_bf16 v86, v58, v14
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_bf16 v86, v59, v15
	;;#ASMEND
	s_and_saveexec_b64 s[64:65], s[8:9]
	s_cbranch_execz .LBB0_116
; %bb.113:                              ;   in Loop: Header=BB0_13 Depth=2
	;;#ASMSTART
	v_dot2c_f32_bf16 v86, v60, v8
	;;#ASMEND
	v_add_u32_e32 v3, 0x600, v0
	;;#ASMSTART
	v_dot2c_f32_bf16 v86, v61, v9
	;;#ASMEND
	v_cmp_gt_u32_e64 s[8:9], s2, v3
	;;#ASMSTART
	v_dot2c_f32_bf16 v86, v62, v10
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_bf16 v86, v63, v11
	;;#ASMEND
	s_and_saveexec_b64 s[66:67], s[8:9]
	s_cbranch_execz .LBB0_115
; %bb.114:                              ;   in Loop: Header=BB0_13 Depth=2
	;;#ASMSTART
	v_dot2c_f32_bf16 v86, v64, v4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_bf16 v86, v65, v5
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_bf16 v86, v66, v6
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_bf16 v86, v67, v7
	;;#ASMEND
.LBB0_115:                              ;   in Loop: Header=BB0_13 Depth=2
	s_or_b64 exec, exec, s[66:67]
.LBB0_116:                              ;   in Loop: Header=BB0_13 Depth=2
	s_or_b64 exec, exec, s[64:65]
.LBB0_117:                              ;   in Loop: Header=BB0_13 Depth=2
	s_or_b64 exec, exec, s[62:63]
	s_or_b64 exec, exec, s[18:19]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_12
.LBB0_118:                              ;   in Loop: Header=BB0_13 Depth=2
	s_waitcnt vmcnt(0) lgkmcnt(0)
	;;#ASMSTART
	v_dot2c_f32_bf16 v87, v68, v16
	;;#ASMEND
	v_cmp_gt_u32_e32 vcc, s2, v2
	;;#ASMSTART
	v_dot2c_f32_bf16 v87, v69, v17
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_bf16 v87, v70, v18
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_bf16 v87, v71, v19
	;;#ASMEND
	s_and_saveexec_b64 s[18:19], vcc
	s_cbranch_execz .LBB0_11
; %bb.119:                              ;   in Loop: Header=BB0_13 Depth=2
	;;#ASMSTART
	v_dot2c_f32_bf16 v87, v72, v12
	;;#ASMEND
	v_add_u32_e32 v2, 0x400, v0
	;;#ASMSTART
	v_dot2c_f32_bf16 v87, v73, v13
	;;#ASMEND
	v_cmp_gt_u32_e32 vcc, s2, v2
	;;#ASMSTART
	v_dot2c_f32_bf16 v87, v74, v14
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_bf16 v87, v75, v15
	;;#ASMEND
	s_and_saveexec_b64 s[62:63], vcc
	s_cbranch_execz .LBB0_10
; %bb.120:                              ;   in Loop: Header=BB0_13 Depth=2
	;;#ASMSTART
	v_dot2c_f32_bf16 v87, v76, v8
	;;#ASMEND
	v_add_u32_e32 v0, 0x600, v0
	;;#ASMSTART
	v_dot2c_f32_bf16 v87, v77, v9
	;;#ASMEND
	v_cmp_gt_u32_e32 vcc, s2, v0
	;;#ASMSTART
	v_dot2c_f32_bf16 v87, v78, v10
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_bf16 v87, v79, v11
	;;#ASMEND
	s_and_saveexec_b64 s[64:65], vcc
	s_cbranch_execz .LBB0_9
; %bb.121:                              ;   in Loop: Header=BB0_13 Depth=2
	;;#ASMSTART
	v_dot2c_f32_bf16 v87, v80, v4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_bf16 v87, v81, v5
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_bf16 v87, v82, v6
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_bf16 v87, v83, v7
	;;#ASMEND
	s_branch .LBB0_9
.LBB0_122:                              ;   in Loop: Header=BB0_7 Depth=1
	v_mov_b32_e32 v2, v1
	v_mov_b32_e32 v3, v1
	v_mov_b32_e32 v0, v1
	v_mov_b64_e32 v[86:87], v[2:3]
	v_mov_b64_e32 v[84:85], v[0:1]
.LBB0_123:                              ;   in Loop: Header=BB0_7 Depth=1
	;;#ASMSTART
	s_nop 0
	v_add_f32 v84, v84, v84 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v85, v85, v85 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v86, v86, v86 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v87, v87, v87 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v84, v84, v84 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v85, v85, v85 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v86, v86, v86 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v87, v87, v87 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v84, v84, v84 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v85, v85, v85 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v86, v86, v86 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v87, v87, v87 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v84, v84, v84 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v85, v85, v85 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v86, v86, v86 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v87, v87, v87 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v84, v84, v84 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v85, v85, v85 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v86, v86, v86 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v87, v87, v87 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v84, v84, v84 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v85, v85, v85 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v86, v86, v86 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v87, v87, v87 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	s_and_saveexec_b64 s[18:19], s[0:1]
	s_cbranch_execz .LBB0_6
; %bb.124:                              ;   in Loop: Header=BB0_7 Depth=1
	v_lshlrev_b64 v[90:91], 1, v[88:89]
	v_lshl_add_u64 v[2:3], s[14:15], 0, v[90:91]
	v_lshl_add_u64 v[90:91], s[12:13], 0, v[90:91]
	s_andn2_b64 vcc, exec, s[46:47]
	v_cmp_ne_u32_e64 s[8:9], 0, v95
	s_cbranch_vccnz .LBB0_130
; %bb.125:                              ;   in Loop: Header=BB0_7 Depth=1
	s_and_saveexec_b64 s[62:63], s[8:9]
	s_cbranch_execz .LBB0_129
; %bb.126:                              ;   in Loop: Header=BB0_7 Depth=1
	s_andn2_b64 vcc, exec, s[30:31]
	v_mul_f32_e32 v0, s20, v84
	s_cbranch_vccnz .LBB0_128
; %bb.127:                              ;   in Loop: Header=BB0_7 Depth=1
	global_load_ushort v84, v[90:91], off
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v84, 16, v84
	v_fmac_f32_e32 v0, s21, v84
.LBB0_128:                              ;   in Loop: Header=BB0_7 Depth=1
	v_cvt_pk_bf16_f32 v0, v0, s0
	global_store_short v[2:3], v0, off
.LBB0_129:                              ;   in Loop: Header=BB0_7 Depth=1
	s_or_b64 exec, exec, s[62:63]
.LBB0_130:                              ;   in Loop: Header=BB0_7 Depth=1
	s_andn2_b64 vcc, exec, s[48:49]
	s_cbranch_vccz .LBB0_133
; %bb.131:                              ;   in Loop: Header=BB0_7 Depth=1
	s_andn2_b64 vcc, exec, s[50:51]
	s_cbranch_vccz .LBB0_138
.LBB0_132:                              ;   in Loop: Header=BB0_7 Depth=1
	s_andn2_b64 vcc, exec, s[56:57]
	s_cbranch_vccnz .LBB0_6
	s_branch .LBB0_143
.LBB0_133:                              ;   in Loop: Header=BB0_7 Depth=1
	s_and_saveexec_b64 s[62:63], s[8:9]
	s_cbranch_execz .LBB0_137
; %bb.134:                              ;   in Loop: Header=BB0_7 Depth=1
	s_andn2_b64 vcc, exec, s[30:31]
	v_mul_f32_e32 v0, s20, v85
	s_cbranch_vccnz .LBB0_136
; %bb.135:                              ;   in Loop: Header=BB0_7 Depth=1
	v_lshl_add_u64 v[84:85], s[34:35], 1, v[90:91]
	global_load_ushort v84, v[84:85], off
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v84, 16, v84
	v_fmac_f32_e32 v0, s21, v84
.LBB0_136:                              ;   in Loop: Header=BB0_7 Depth=1
	v_lshl_add_u64 v[84:85], s[36:37], 1, v[2:3]
	v_cvt_pk_bf16_f32 v0, v0, s0
	global_store_short v[84:85], v0, off
.LBB0_137:                              ;   in Loop: Header=BB0_7 Depth=1
	s_or_b64 exec, exec, s[62:63]
	s_andn2_b64 vcc, exec, s[50:51]
	s_cbranch_vccnz .LBB0_132
.LBB0_138:                              ;   in Loop: Header=BB0_7 Depth=1
	s_and_saveexec_b64 s[62:63], s[8:9]
	s_cbranch_execz .LBB0_142
; %bb.139:                              ;   in Loop: Header=BB0_7 Depth=1
	s_andn2_b64 vcc, exec, s[30:31]
	v_mul_f32_e32 v0, s20, v86
	s_cbranch_vccnz .LBB0_141
; %bb.140:                              ;   in Loop: Header=BB0_7 Depth=1
	v_lshl_add_u64 v[84:85], v[90:91], 0, s[52:53]
	global_load_ushort v84, v[84:85], off
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v84, 16, v84
	v_fmac_f32_e32 v0, s21, v84
.LBB0_141:                              ;   in Loop: Header=BB0_7 Depth=1
	v_lshl_add_u64 v[84:85], v[2:3], 0, s[54:55]
	v_cvt_pk_bf16_f32 v0, v0, s0
	global_store_short v[84:85], v0, off
.LBB0_142:                              ;   in Loop: Header=BB0_7 Depth=1
	s_or_b64 exec, exec, s[62:63]
	s_andn2_b64 vcc, exec, s[56:57]
	s_cbranch_vccnz .LBB0_6
.LBB0_143:                              ;   in Loop: Header=BB0_7 Depth=1
	s_and_b64 exec, exec, s[8:9]
	s_cbranch_execz .LBB0_6
; %bb.144:                              ;   in Loop: Header=BB0_7 Depth=1
	s_andn2_b64 vcc, exec, s[30:31]
	v_mul_f32_e32 v0, s20, v87
	s_cbranch_vccnz .LBB0_5
; %bb.145:                              ;   in Loop: Header=BB0_7 Depth=1
	v_lshl_add_u64 v[84:85], v[90:91], 0, s[58:59]
	global_load_ushort v84, v[84:85], off
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v84, 16, v84
	v_fmac_f32_e32 v0, s21, v84
	s_branch .LBB0_5
.LBB0_146:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wvSpltK_bf16_tn_m4
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
		.amdhsa_next_free_vgpr 112
		.amdhsa_next_free_sgpr 96
		.amdhsa_accum_offset 112
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
	.size	wvSpltK_bf16_tn_m4, .Lfunc_end0-wvSpltK_bf16_tn_m4
                                        ; -- End function
	.set wvSpltK_bf16_tn_m4.num_vgpr, 112
	.set wvSpltK_bf16_tn_m4.num_agpr, 0
	.set wvSpltK_bf16_tn_m4.numbered_sgpr, 70
	.set wvSpltK_bf16_tn_m4.num_named_barrier, 0
	.set wvSpltK_bf16_tn_m4.private_seg_size, 0
	.set wvSpltK_bf16_tn_m4.uses_vcc, 1
	.set wvSpltK_bf16_tn_m4.uses_flat_scratch, 0
	.set wvSpltK_bf16_tn_m4.has_dyn_sized_stack, 0
	.set wvSpltK_bf16_tn_m4.has_recursion, 0
	.set wvSpltK_bf16_tn_m4.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 5368
; TotalNumSgprs: 76
; NumVgprs: 112
; NumAgprs: 0
; TotalNumVgprs: 112
; ScratchSize: 0
; MemoryBound: 1
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 131072 bytes/workgroup (compile time only)
; SGPRBlocks: 12
; VGPRBlocks: 13
; NumSGPRsForWavesPerEU: 102
; NumVGPRsForWavesPerEU: 112
; AccumOffset: 112
; Occupancy: 4
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 1
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 27
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_529f8428354d040b,@object ; @__hip_cuid_529f8428354d040b
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_529f8428354d040b
__hip_cuid_529f8428354d040b:
	.byte	0                               ; 0x0
	.size	__hip_cuid_529f8428354d040b, 1

	.ident	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.4 26084 f58b06dce1f9c15707c5f808fd002e18c2accf7e)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_529f8428354d040b
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
  AssertSizeEqual: { 2: 1 }
  AssertSizeGreaterThan: { 0: 8 }
  AssertSizeLessThan: { 1: 5 }
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
    .name:           wvSpltK_bf16_tn_m4
    .private_segment_fixed_size: 0
    .sgpr_count:     76
    .sgpr_spill_count: 0
    .symbol:         wvSpltK_bf16_tn_m4.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     112
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
