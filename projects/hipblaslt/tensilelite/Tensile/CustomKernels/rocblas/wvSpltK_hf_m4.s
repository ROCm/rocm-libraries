	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.protected	wvSpltK_hf_m4           ; -- Begin function wvSpltK_hf_m4
	.globl	wvSpltK_hf_m4
	.p2align	8
	.type	wvSpltK_hf_m4,@function
wvSpltK_hf_m4:                          ; @wvSpltK_hf_m4
; %bb.0:
	s_load_dwordx2 s[10:11], s[0:1], 0x0
	v_bfe_u32 v1, v0, 10, 10
	v_lshl_add_u32 v2, s2, 4, v1
	v_lshl_add_u32 v52, v2, 1, v2
	v_mov_b32_e32 v53, 0
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s21, s11, 31
	s_mov_b32 s20, s11
	v_lshl_add_u64 v[2:3], v[52:53], 0, 3
	v_cmp_gt_u64_e32 vcc, s[20:21], v[52:53]
	v_cmp_le_u64_e64 s[2:3], s[20:21], v[2:3]
	v_mov_b32_e32 v48, 1
	s_and_b64 s[4:5], vcc, s[2:3]
	v_mov_b32_e32 v49, v48
	v_mov_b32_e32 v50, v48
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_6
; %bb.1:
	s_add_i32 s8, s20, -3
	s_mov_b32 s13, 0
	s_mov_b32 s9, s13
	v_cmp_ne_u32_e32 vcc, s8, v52
	s_mov_b32 s4, 1
	v_mov_b32_e32 v49, v48
	v_mov_b32_e32 v50, v48
	s_and_saveexec_b64 s[14:15], vcc
	s_cbranch_execz .LBB0_5
; %bb.2:
	v_subrev_co_u32_e32 v2, vcc, s8, v52
	s_mov_b64 s[16:17], 0
	s_nop 0
	v_subb_co_u32_e64 v3, s[6:7], 0, 0, vcc
	s_mov_b64 s[18:19], 0
	s_mov_b32 s5, s4
	s_mov_b32 s6, s4
.LBB0_3:                                ; =>This Inner Loop Header: Depth=1
	s_cmp_lg_u32 s18, 2
	s_cselect_b32 s6, s6, 0
	s_cmp_lg_u32 s18, 1
	s_cselect_b32 s5, s5, 0
	s_cmp_lg_u32 s18, 0
	s_cselect_b32 s4, s4, 0
	s_add_u32 s18, s18, 1
	s_mov_b32 s12, s18
	s_addc_u32 s19, s19, 0
	v_cmp_ge_u64_e32 vcc, s[12:13], v[2:3]
	v_mov_b32_e32 v50, s6
	s_or_b64 s[16:17], vcc, s[16:17]
	v_mov_b32_e32 v49, s5
	v_mov_b32_e32 v48, s4
	s_andn2_b64 exec, exec, s[16:17]
	s_cbranch_execnz .LBB0_3
; %bb.4:
	s_or_b64 exec, exec, s[16:17]
.LBB0_5:
	s_or_b64 exec, exec, s[14:15]
	v_mov_b64_e32 v[52:53], s[8:9]
.LBB0_6:
	s_or_b64 exec, exec, s[2:3]
	s_lshl_b32 s33, s10, 2
	s_cmp_lg_u32 s10, 0
	v_and_b32_e32 v2, 0x3ff, v0
	s_cselect_b64 s[2:3], -1, 0
	s_cmp_eq_u32 s10, 0
	s_mov_b32 s11, 0
	s_cbranch_scc1 .LBB0_12
; %bb.7:
	s_load_dwordx2 s[4:5], s[0:1], 0x10
	v_and_b32_e32 v0, 3, v2
	v_mul_lo_u32 v0, s10, v0
	s_min_i32 s14, s33, 0x8000
	v_lshlrev_b32_e32 v3, 1, v0
	v_lshl_add_u32 v4, v1, 6, v2
	s_mov_b64 s[6:7], 0
	v_mov_b32_e32 v1, 0
                                        ; implicit-def: $sgpr8_sgpr9
	s_branch .LBB0_9
.LBB0_8:                                ;   in Loop: Header=BB0_9 Depth=1
	s_or_b64 exec, exec, s[12:13]
	s_and_b64 s[12:13], exec, s[8:9]
	s_or_b64 s[6:7], s[12:13], s[6:7]
	s_andn2_b64 exec, exec, s[6:7]
	s_cbranch_execz .LBB0_11
.LBB0_9:                                ; =>This Inner Loop Header: Depth=1
	v_add_u32_e32 v0, s11, v4
	v_cmp_gt_u32_e32 vcc, s14, v0
	s_or_b64 s[8:9], s[8:9], exec
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_8
; %bb.10:                               ;   in Loop: Header=BB0_9 Depth=1
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[6:7], v[0:1], 1, s[4:5]
	global_load_ushort v5, v[6:7], off
	s_addk_i32 s11, 0x400
	s_cmp_ge_u32 s11, s14
	v_lshrrev_b32_e32 v0, 1, v0
	s_cselect_b64 s[16:17], -1, 0
	v_and_b32_e32 v0, 0x7ffffffe, v0
	s_andn2_b64 s[8:9], s[8:9], exec
	s_and_b64 s[16:17], s[16:17], exec
	v_add_u32_e32 v0, v3, v0
	s_or_b64 s[8:9], s[8:9], s[16:17]
	s_waitcnt vmcnt(0)
	ds_write_b16 v0, v5
	s_branch .LBB0_8
.LBB0_11:
	s_or_b64 exec, exec, s[6:7]
.LBB0_12:
	v_cmp_gt_u64_e32 vcc, s[20:21], v[52:53]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_110
; %bb.13:
	s_load_dwordx8 s[12:19], s[0:1], 0x18
	s_load_dwordx2 s[22:23], s[0:1], 0x8
	s_ashr_i32 s11, s10, 31
	s_mov_b32 s25, 0
	v_cndmask_b32_e64 v0, 0, 1, s[2:3]
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s18, s18, 48
	v_lshlrev_b32_e32 v51, 3, v2
	v_cmp_eq_u32_e64 s[0:1], 63, v2
	v_cmp_neq_f32_e64 s[26:27], s17, 0
	s_ashr_i32 s19, s18, 31
	s_add_i32 s28, s20, -3
	s_mov_b32 s29, s25
	s_lshl_b64 s[30:31], s[10:11], 1
	s_lshl_b32 s42, s10, 1
	v_lshlrev_b32_e32 v68, 4, v2
	s_mul_i32 s43, s10, 6
	s_mov_b64 s[34:35], 0
	v_cmp_ne_u32_e64 s[2:3], 1, v0
	v_mov_b32_e32 v55, 0
                                        ; implicit-def: $vgpr0_vgpr1_vgpr2_vgpr3
                                        ; implicit-def: $vgpr4_vgpr5_vgpr6_vgpr7
                                        ; implicit-def: $vgpr8_vgpr9_vgpr10_vgpr11
                                        ; implicit-def: $vgpr12_vgpr13_vgpr14_vgpr15
                                        ; implicit-def: $vgpr16_vgpr17_vgpr18_vgpr19
                                        ; implicit-def: $vgpr20_vgpr21_vgpr22_vgpr23
                                        ; implicit-def: $vgpr42_vgpr43
                                        ; implicit-def: $vgpr60_vgpr61
                                        ; implicit-def: $vgpr62_vgpr63
                                        ; implicit-def: $vgpr46_vgpr47
                                        ; implicit-def: $vgpr26_vgpr27
                                        ; implicit-def: $vgpr38_vgpr39
                                        ; implicit-def: $vgpr58_vgpr59
                                        ; implicit-def: $vgpr56_vgpr57
                                        ; implicit-def: $vgpr34_vgpr35
                                        ; implicit-def: $vgpr30_vgpr31
	s_branch .LBB0_16
.LBB0_14:                               ;   in Loop: Header=BB0_16 Depth=1
	s_or_b64 exec, exec, s[8:9]
	v_mov_b64_e32 v[52:53], s[28:29]
.LBB0_15:                               ;   in Loop: Header=BB0_16 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_cmp_le_u64_e32 vcc, s[20:21], v[52:53]
	s_or_b64 s[34:35], vcc, s[34:35]
	s_andn2_b64 exec, exec, s[34:35]
	s_cbranch_execz .LBB0_110
.LBB0_16:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_20 Depth 2
                                        ;     Child Loop BB0_96 Depth 2
	s_and_b64 vcc, exec, s[2:3]
	s_cbranch_vccnz .LBB0_43
; %bb.17:                               ;   in Loop: Header=BB0_16 Depth=1
	v_mul_lo_u32 v54, v53, s10
	v_mul_lo_u32 v66, v52, s11
	v_mad_u64_u32 v[64:65], s[4:5], v52, s10, 0
	v_add3_u32 v65, v65, v66, v54
	v_lshl_add_u64 v[64:65], v[64:65], 1, s[22:23]
	s_mov_b32 s24, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v81, v68
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v73, 0
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v75, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v79, 0
	v_mov_b32_e32 v80, 0
	s_branch .LBB0_20
.LBB0_18:                               ;   in Loop: Header=BB0_20 Depth=2
	s_or_b64 exec, exec, s[6:7]
.LBB0_19:                               ;   in Loop: Header=BB0_20 Depth=2
	s_or_b64 exec, exec, s[4:5]
	s_addk_i32 s24, 0x400
	s_cmp_ge_u32 s24, s10
	v_add_u32_e32 v81, 0x800, v81
	s_cbranch_scc1 .LBB0_44
.LBB0_20:                               ;   Parent Loop BB0_16 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_add_u32_e32 v54, s24, v51
	v_cmp_gt_u32_e32 vcc, s10, v54
	v_add_u32_e32 v66, 0x200, v54
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execnz .LBB0_26
; %bb.21:                               ;   in Loop: Header=BB0_20 Depth=2
	s_or_b64 exec, exec, s[6:7]
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execnz .LBB0_29
.LBB0_22:                               ;   in Loop: Header=BB0_20 Depth=2
	s_or_b64 exec, exec, s[6:7]
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execnz .LBB0_32
.LBB0_23:                               ;   in Loop: Header=BB0_20 Depth=2
	s_or_b64 exec, exec, s[6:7]
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execnz .LBB0_35
.LBB0_24:                               ;   in Loop: Header=BB0_20 Depth=2
	s_or_b64 exec, exec, s[6:7]
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execnz .LBB0_38
.LBB0_25:                               ;   in Loop: Header=BB0_20 Depth=2
	s_or_b64 exec, exec, s[6:7]
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_19
	s_branch .LBB0_41
.LBB0_26:                               ;   in Loop: Header=BB0_20 Depth=2
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[4:5], v[54:55], 1, v[64:65]
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[20:21], s[10:11], 1, v[4:5]
	global_load_dwordx4 v[4:7], v[4:5], off nt
	s_nop 0
	global_load_dwordx4 v[12:15], v[20:21], off nt
	v_lshl_add_u64 v[20:21], v[20:21], 0, s[30:31]
	global_load_dwordx4 v[20:23], v[20:21], off nt
	v_cmp_gt_u32_e64 s[4:5], s10, v66
	s_and_saveexec_b64 s[8:9], s[4:5]
	s_cbranch_execz .LBB0_28
; %bb.27:                               ;   in Loop: Header=BB0_20 Depth=2
	v_mov_b32_e32 v67, v55
	v_lshl_add_u64 v[0:1], v[66:67], 1, v[64:65]
	v_lshl_add_u64 v[16:17], s[10:11], 1, v[0:1]
	global_load_dwordx4 v[0:3], v[0:1], off nt
	s_nop 0
	global_load_dwordx4 v[8:11], v[16:17], off nt
	v_lshl_add_u64 v[16:17], v[16:17], 0, s[30:31]
	global_load_dwordx4 v[16:19], v[16:17], off nt
.LBB0_28:                               ;   in Loop: Header=BB0_20 Depth=2
	s_or_b64 exec, exec, s[8:9]
	s_or_b64 exec, exec, s[6:7]
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_22
.LBB0_29:                               ;   in Loop: Header=BB0_20 Depth=2
	v_add_u32_e32 v54, s42, v81
	v_add_u32_e32 v67, s33, v81
	s_waitcnt lgkmcnt(4)
	ds_read_b128 v[28:31], v81
	s_waitcnt lgkmcnt(4)
	ds_read_b128 v[32:35], v54
	s_waitcnt lgkmcnt(4)
	ds_read2_b32 v[56:57], v67 offset1:1
	v_add_u32_e32 v82, s43, v81
	s_waitcnt lgkmcnt(4)
	ds_read2_b32 v[58:59], v67 offset0:2 offset1:3
	s_waitcnt lgkmcnt(4)
	ds_read_b128 v[36:39], v82
	v_cmp_gt_u32_e64 s[4:5], s10, v66
	s_and_saveexec_b64 s[8:9], s[4:5]
	s_cbranch_execz .LBB0_31
; %bb.30:                               ;   in Loop: Header=BB0_20 Depth=2
	v_add_u32_e32 v40, 0x400, v67
	v_add_u32_e32 v41, 0x408, v67
	ds_read_b128 v[24:27], v81 offset:1024
	ds_read2_b32 v[62:63], v40 offset1:1
	ds_read2_b32 v[60:61], v41 offset1:1
	ds_read_b128 v[44:47], v54 offset:1024
	ds_read_b128 v[40:43], v82 offset:1024
.LBB0_31:                               ;   in Loop: Header=BB0_20 Depth=2
	s_or_b64 exec, exec, s[8:9]
	s_or_b64 exec, exec, s[6:7]
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_23
.LBB0_32:                               ;   in Loop: Header=BB0_20 Depth=2
	s_waitcnt vmcnt(2) lgkmcnt(4)
	;;#ASMSTART
	v_dot2c_f32_f16 v80, v28, v4
	;;#ASMEND
	s_waitcnt vmcnt(1)
	;;#ASMSTART
	v_dot2c_f32_f16 v79, v28, v12
	;;#ASMEND
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	v_dot2c_f32_f16 v78, v28, v20
	;;#ASMEND
	v_cmp_gt_u32_e64 s[4:5], s10, v66
	;;#ASMSTART
	v_dot2c_f32_f16 v80, v29, v5
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v79, v29, v13
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v78, v29, v21
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v80, v30, v6
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v79, v30, v14
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v78, v30, v22
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v80, v31, v7
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v79, v31, v15
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v78, v31, v23
	;;#ASMEND
	s_and_saveexec_b64 s[8:9], s[4:5]
	s_cbranch_execz .LBB0_34
; %bb.33:                               ;   in Loop: Header=BB0_20 Depth=2
	;;#ASMSTART
	v_dot2c_f32_f16 v80, v24, v0
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v79, v24, v8
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v78, v24, v16
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v80, v25, v1
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v79, v25, v9
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v78, v25, v17
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v80, v26, v2
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v79, v26, v10
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v78, v26, v18
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v80, v27, v3
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v79, v27, v11
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v78, v27, v19
	;;#ASMEND
.LBB0_34:                               ;   in Loop: Header=BB0_20 Depth=2
	s_or_b64 exec, exec, s[8:9]
	s_or_b64 exec, exec, s[6:7]
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_24
.LBB0_35:                               ;   in Loop: Header=BB0_20 Depth=2
	s_waitcnt vmcnt(2) lgkmcnt(3)
	;;#ASMSTART
	v_dot2c_f32_f16 v77, v32, v4
	;;#ASMEND
	s_waitcnt vmcnt(1)
	;;#ASMSTART
	v_dot2c_f32_f16 v76, v32, v12
	;;#ASMEND
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	v_dot2c_f32_f16 v75, v32, v20
	;;#ASMEND
	v_cmp_gt_u32_e64 s[4:5], s10, v66
	;;#ASMSTART
	v_dot2c_f32_f16 v77, v33, v5
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v76, v33, v13
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v75, v33, v21
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v77, v34, v6
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v76, v34, v14
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v75, v34, v22
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v77, v35, v7
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v76, v35, v15
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v75, v35, v23
	;;#ASMEND
	s_and_saveexec_b64 s[8:9], s[4:5]
	s_cbranch_execz .LBB0_37
; %bb.36:                               ;   in Loop: Header=BB0_20 Depth=2
	s_waitcnt lgkmcnt(1)
	;;#ASMSTART
	v_dot2c_f32_f16 v77, v44, v0
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v76, v44, v8
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v75, v44, v16
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v77, v45, v1
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v76, v45, v9
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v75, v45, v17
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v77, v46, v2
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v76, v46, v10
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v75, v46, v18
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v77, v47, v3
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v76, v47, v11
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v75, v47, v19
	;;#ASMEND
.LBB0_37:                               ;   in Loop: Header=BB0_20 Depth=2
	s_or_b64 exec, exec, s[8:9]
	s_or_b64 exec, exec, s[6:7]
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_25
.LBB0_38:                               ;   in Loop: Header=BB0_20 Depth=2
	s_waitcnt vmcnt(2) lgkmcnt(2)
	;;#ASMSTART
	v_dot2c_f32_f16 v74, v56, v4
	;;#ASMEND
	s_waitcnt vmcnt(1)
	;;#ASMSTART
	v_dot2c_f32_f16 v73, v56, v12
	;;#ASMEND
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	v_dot2c_f32_f16 v72, v56, v20
	;;#ASMEND
	v_cmp_gt_u32_e64 s[4:5], s10, v66
	;;#ASMSTART
	v_dot2c_f32_f16 v74, v57, v5
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v73, v57, v13
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v72, v57, v21
	;;#ASMEND
	s_waitcnt lgkmcnt(1)
	;;#ASMSTART
	v_dot2c_f32_f16 v74, v58, v6
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v73, v58, v14
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v72, v58, v22
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v74, v59, v7
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v73, v59, v15
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v72, v59, v23
	;;#ASMEND
	s_and_saveexec_b64 s[8:9], s[4:5]
	s_cbranch_execz .LBB0_40
; %bb.39:                               ;   in Loop: Header=BB0_20 Depth=2
	;;#ASMSTART
	v_dot2c_f32_f16 v74, v62, v0
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v73, v62, v8
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v72, v62, v16
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v74, v63, v1
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v73, v63, v9
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v72, v63, v17
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v74, v60, v2
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v73, v60, v10
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v72, v60, v18
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v74, v61, v3
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v73, v61, v11
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v72, v61, v19
	;;#ASMEND
.LBB0_40:                               ;   in Loop: Header=BB0_20 Depth=2
	s_or_b64 exec, exec, s[8:9]
	s_or_b64 exec, exec, s[6:7]
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_19
.LBB0_41:                               ;   in Loop: Header=BB0_20 Depth=2
	s_waitcnt vmcnt(2) lgkmcnt(0)
	;;#ASMSTART
	v_dot2c_f32_f16 v71, v36, v4
	;;#ASMEND
	s_waitcnt vmcnt(1)
	;;#ASMSTART
	v_dot2c_f32_f16 v70, v36, v12
	;;#ASMEND
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	v_dot2c_f32_f16 v69, v36, v20
	;;#ASMEND
	v_cmp_gt_u32_e32 vcc, s10, v66
	;;#ASMSTART
	v_dot2c_f32_f16 v71, v37, v5
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v70, v37, v13
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v69, v37, v21
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v71, v38, v6
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v70, v38, v14
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v69, v38, v22
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v71, v39, v7
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v70, v39, v15
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v69, v39, v23
	;;#ASMEND
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_18
; %bb.42:                               ;   in Loop: Header=BB0_20 Depth=2
	;;#ASMSTART
	v_dot2c_f32_f16 v71, v40, v0
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v70, v40, v8
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v69, v40, v16
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v71, v41, v1
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v70, v41, v9
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v69, v41, v17
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v71, v42, v2
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v70, v42, v10
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v69, v42, v18
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v71, v43, v3
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v70, v43, v11
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v69, v43, v19
	;;#ASMEND
	s_branch .LBB0_18
.LBB0_43:                               ;   in Loop: Header=BB0_16 Depth=1
	v_mov_b32_e32 v80, v55
	v_mov_b32_e32 v79, v55
	v_mov_b32_e32 v78, v55
	v_mov_b32_e32 v77, v55
	v_mov_b32_e32 v76, v55
	v_mov_b32_e32 v75, v55
	v_mov_b32_e32 v74, v55
	v_mov_b32_e32 v73, v55
	v_mov_b32_e32 v72, v55
	v_mov_b32_e32 v71, v55
	v_mov_b32_e32 v70, v55
	v_mov_b32_e32 v69, v55
.LBB0_44:                               ;   in Loop: Header=BB0_16 Depth=1
	;;#ASMSTART
	s_nop 0
	v_add_f32 v80, v80, v80 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v79, v79, v79 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v78, v78, v78 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v77, v77, v77 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v76, v76, v76 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v75, v75, v75 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v74, v74, v74 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v73, v73, v73 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v72, v72, v72 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v71, v71, v71 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v70, v70, v70 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v69, v69, v69 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v80, v80, v80 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v79, v79, v79 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v78, v78, v78 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v77, v77, v77 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v76, v76, v76 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v75, v75, v75 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v74, v74, v74 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v73, v73, v73 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v72, v72, v72 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v71, v71, v71 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v70, v70, v70 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v69, v69, v69 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v80, v80, v80 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v79, v79, v79 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v78, v78, v78 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v77, v77, v77 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v76, v76, v76 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v75, v75, v75 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v74, v74, v74 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v73, v73, v73 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v72, v72, v72 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v71, v71, v71 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v70, v70, v70 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v69, v69, v69 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v80, v80, v80 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v79, v79, v79 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v78, v78, v78 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v77, v77, v77 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v76, v76, v76 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v75, v75, v75 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v74, v74, v74 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v73, v73, v73 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v72, v72, v72 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v71, v71, v71 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v70, v70, v70 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v69, v69, v69 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v80, v80, v80 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v79, v79, v79 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v78, v78, v78 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v77, v77, v77 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v76, v76, v76 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v75, v75, v75 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v74, v74, v74 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v73, v73, v73 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v72, v72, v72 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v71, v71, v71 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v70, v70, v70 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v69, v69, v69 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v80, v80, v80 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v79, v79, v79 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v78, v78, v78 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v77, v77, v77 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v76, v76, v76 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v75, v75, v75 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v74, v74, v74 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v73, v73, v73 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v72, v72, v72 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v71, v71, v71 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v70, v70, v70 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v69, v69, v69 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	s_and_saveexec_b64 s[36:37], s[0:1]
	s_cbranch_execz .LBB0_93
; %bb.45:                               ;   in Loop: Header=BB0_16 Depth=1
	v_cmp_ne_u32_e64 s[8:9], 0, v48
	s_and_saveexec_b64 s[4:5], s[8:9]
	s_cbranch_execz .LBB0_49
; %bb.46:                               ;   in Loop: Header=BB0_16 Depth=1
	v_mul_f32_e32 v54, s16, v80
	s_and_b64 vcc, exec, s[26:27]
	s_cbranch_vccz .LBB0_98
; %bb.47:                               ;   in Loop: Header=BB0_16 Depth=1
	v_lshlrev_b64 v[64:65], 3, v[52:53]
	v_lshl_add_u64 v[66:67], s[12:13], 0, v[64:65]
	global_load_ushort v66, v[66:67], off
	v_lshl_add_u64 v[64:65], s[14:15], 0, v[64:65]
	s_waitcnt vmcnt(0)
	v_fma_mixlo_f16 v66, s17, v66, v54 op_sel_hi:[0,1,0]
	global_store_short v[64:65], v66, off
	s_cbranch_execnz .LBB0_49
.LBB0_48:                               ;   in Loop: Header=BB0_16 Depth=1
	v_cvt_f16_f32_e32 v54, v54
	v_lshl_add_u64 v[64:65], v[52:53], 3, s[14:15]
	global_store_short v[64:65], v54, off
.LBB0_49:                               ;   in Loop: Header=BB0_16 Depth=1
	s_or_b64 exec, exec, s[4:5]
	v_cmp_ne_u32_e64 s[4:5], 0, v49
	s_and_saveexec_b64 s[6:7], s[4:5]
	s_cbranch_execz .LBB0_53
; %bb.50:                               ;   in Loop: Header=BB0_16 Depth=1
	s_andn2_b64 vcc, exec, s[26:27]
	v_mul_f32_e32 v54, s16, v79
	s_cbranch_vccnz .LBB0_99
; %bb.51:                               ;   in Loop: Header=BB0_16 Depth=1
	v_lshlrev_b64 v[64:65], 3, v[52:53]
	v_lshl_add_u64 v[66:67], s[12:13], 0, v[64:65]
	global_load_ushort v66, v[66:67], off offset:8
	v_lshl_add_u64 v[64:65], s[14:15], 0, v[64:65]
	s_waitcnt vmcnt(0)
	v_fma_mixlo_f16 v66, s17, v66, v54 op_sel_hi:[0,1,0]
	global_store_short v[64:65], v66, off offset:8
	s_cbranch_execnz .LBB0_53
.LBB0_52:                               ;   in Loop: Header=BB0_16 Depth=1
	v_cvt_f16_f32_e32 v54, v54
	v_lshl_add_u64 v[64:65], v[52:53], 3, s[14:15]
	global_store_short v[64:65], v54, off offset:8
.LBB0_53:                               ;   in Loop: Header=BB0_16 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_cmp_ne_u32_e64 s[6:7], 0, v50
	s_and_saveexec_b64 s[38:39], s[6:7]
	s_cbranch_execz .LBB0_57
; %bb.54:                               ;   in Loop: Header=BB0_16 Depth=1
	s_andn2_b64 vcc, exec, s[26:27]
	v_mul_f32_e32 v54, s16, v78
	s_cbranch_vccnz .LBB0_100
; %bb.55:                               ;   in Loop: Header=BB0_16 Depth=1
	v_lshlrev_b64 v[64:65], 3, v[52:53]
	v_lshl_add_u64 v[66:67], s[12:13], 0, v[64:65]
	global_load_ushort v66, v[66:67], off offset:16
	v_lshl_add_u64 v[64:65], s[14:15], 0, v[64:65]
	s_waitcnt vmcnt(0)
	v_fma_mixlo_f16 v66, s17, v66, v54 op_sel_hi:[0,1,0]
	global_store_short v[64:65], v66, off offset:16
	s_cbranch_execnz .LBB0_57
.LBB0_56:                               ;   in Loop: Header=BB0_16 Depth=1
	v_cvt_f16_f32_e32 v54, v54
	v_lshl_add_u64 v[64:65], v[52:53], 3, s[14:15]
	global_store_short v[64:65], v54, off offset:16
.LBB0_57:                               ;   in Loop: Header=BB0_16 Depth=1
	s_or_b64 exec, exec, s[38:39]
	s_and_saveexec_b64 s[38:39], s[8:9]
	s_cbranch_execz .LBB0_61
; %bb.58:                               ;   in Loop: Header=BB0_16 Depth=1
	s_andn2_b64 vcc, exec, s[26:27]
	v_mul_f32_e32 v54, s16, v77
	s_cbranch_vccnz .LBB0_101
; %bb.59:                               ;   in Loop: Header=BB0_16 Depth=1
	v_lshlrev_b64 v[64:65], 3, v[52:53]
	v_lshl_add_u64 v[66:67], s[12:13], 0, v[64:65]
	global_load_ushort v66, v[66:67], off offset:2
	v_lshl_add_u64 v[64:65], s[14:15], 0, v[64:65]
	s_waitcnt vmcnt(0)
	v_fma_mixlo_f16 v66, s17, v66, v54 op_sel_hi:[0,1,0]
	global_store_short v[64:65], v66, off offset:2
	s_cbranch_execnz .LBB0_61
.LBB0_60:                               ;   in Loop: Header=BB0_16 Depth=1
	v_cvt_f16_f32_e32 v54, v54
	v_lshl_add_u64 v[64:65], v[52:53], 3, s[14:15]
	global_store_short v[64:65], v54, off offset:2
.LBB0_61:                               ;   in Loop: Header=BB0_16 Depth=1
	s_or_b64 exec, exec, s[38:39]
	s_and_saveexec_b64 s[38:39], s[4:5]
	s_cbranch_execz .LBB0_65
; %bb.62:                               ;   in Loop: Header=BB0_16 Depth=1
	s_andn2_b64 vcc, exec, s[26:27]
	v_mul_f32_e32 v54, s16, v76
	s_cbranch_vccnz .LBB0_102
; %bb.63:                               ;   in Loop: Header=BB0_16 Depth=1
	v_lshlrev_b64 v[64:65], 3, v[52:53]
	v_lshl_add_u64 v[66:67], s[12:13], 0, v[64:65]
	global_load_ushort v66, v[66:67], off offset:10
	v_lshl_add_u64 v[64:65], s[14:15], 0, v[64:65]
	s_waitcnt vmcnt(0)
	v_fma_mixlo_f16 v66, s17, v66, v54 op_sel_hi:[0,1,0]
	global_store_short v[64:65], v66, off offset:10
	s_cbranch_execnz .LBB0_65
.LBB0_64:                               ;   in Loop: Header=BB0_16 Depth=1
	v_cvt_f16_f32_e32 v54, v54
	v_lshl_add_u64 v[64:65], v[52:53], 3, s[14:15]
	global_store_short v[64:65], v54, off offset:10
.LBB0_65:                               ;   in Loop: Header=BB0_16 Depth=1
	s_or_b64 exec, exec, s[38:39]
	s_and_saveexec_b64 s[38:39], s[6:7]
	s_cbranch_execz .LBB0_69
; %bb.66:                               ;   in Loop: Header=BB0_16 Depth=1
	s_andn2_b64 vcc, exec, s[26:27]
	v_mul_f32_e32 v54, s16, v75
	s_cbranch_vccnz .LBB0_103
; %bb.67:                               ;   in Loop: Header=BB0_16 Depth=1
	v_lshlrev_b64 v[64:65], 3, v[52:53]
	v_lshl_add_u64 v[66:67], s[12:13], 0, v[64:65]
	global_load_ushort v66, v[66:67], off offset:18
	v_lshl_add_u64 v[64:65], s[14:15], 0, v[64:65]
	s_waitcnt vmcnt(0)
	v_fma_mixlo_f16 v66, s17, v66, v54 op_sel_hi:[0,1,0]
	global_store_short v[64:65], v66, off offset:18
	s_cbranch_execnz .LBB0_69
.LBB0_68:                               ;   in Loop: Header=BB0_16 Depth=1
	v_cvt_f16_f32_e32 v54, v54
	v_lshl_add_u64 v[64:65], v[52:53], 3, s[14:15]
	global_store_short v[64:65], v54, off offset:18
.LBB0_69:                               ;   in Loop: Header=BB0_16 Depth=1
	s_or_b64 exec, exec, s[38:39]
	s_and_saveexec_b64 s[38:39], s[8:9]
	s_cbranch_execz .LBB0_73
; %bb.70:                               ;   in Loop: Header=BB0_16 Depth=1
	s_andn2_b64 vcc, exec, s[26:27]
	v_mul_f32_e32 v54, s16, v74
	s_cbranch_vccnz .LBB0_104
; %bb.71:                               ;   in Loop: Header=BB0_16 Depth=1
	v_lshlrev_b64 v[64:65], 3, v[52:53]
	v_lshl_add_u64 v[66:67], s[12:13], 0, v[64:65]
	global_load_ushort v66, v[66:67], off offset:4
	v_lshl_add_u64 v[64:65], s[14:15], 0, v[64:65]
	s_waitcnt vmcnt(0)
	v_fma_mixlo_f16 v66, s17, v66, v54 op_sel_hi:[0,1,0]
	global_store_short v[64:65], v66, off offset:4
	s_cbranch_execnz .LBB0_73
.LBB0_72:                               ;   in Loop: Header=BB0_16 Depth=1
	v_cvt_f16_f32_e32 v54, v54
	v_lshl_add_u64 v[64:65], v[52:53], 3, s[14:15]
	global_store_short v[64:65], v54, off offset:4
.LBB0_73:                               ;   in Loop: Header=BB0_16 Depth=1
	s_or_b64 exec, exec, s[38:39]
	s_and_saveexec_b64 s[38:39], s[4:5]
	s_cbranch_execz .LBB0_77
; %bb.74:                               ;   in Loop: Header=BB0_16 Depth=1
	s_andn2_b64 vcc, exec, s[26:27]
	v_mul_f32_e32 v54, s16, v73
	s_cbranch_vccnz .LBB0_105
; %bb.75:                               ;   in Loop: Header=BB0_16 Depth=1
	v_lshlrev_b64 v[64:65], 3, v[52:53]
	v_lshl_add_u64 v[66:67], s[12:13], 0, v[64:65]
	global_load_ushort v66, v[66:67], off offset:12
	v_lshl_add_u64 v[64:65], s[14:15], 0, v[64:65]
	s_waitcnt vmcnt(0)
	v_fma_mixlo_f16 v66, s17, v66, v54 op_sel_hi:[0,1,0]
	global_store_short v[64:65], v66, off offset:12
	s_cbranch_execnz .LBB0_77
.LBB0_76:                               ;   in Loop: Header=BB0_16 Depth=1
	v_cvt_f16_f32_e32 v54, v54
	v_lshl_add_u64 v[64:65], v[52:53], 3, s[14:15]
	global_store_short v[64:65], v54, off offset:12
.LBB0_77:                               ;   in Loop: Header=BB0_16 Depth=1
	s_or_b64 exec, exec, s[38:39]
	s_and_saveexec_b64 s[38:39], s[6:7]
	s_cbranch_execz .LBB0_81
; %bb.78:                               ;   in Loop: Header=BB0_16 Depth=1
	s_andn2_b64 vcc, exec, s[26:27]
	v_mul_f32_e32 v54, s16, v72
	s_cbranch_vccnz .LBB0_106
; %bb.79:                               ;   in Loop: Header=BB0_16 Depth=1
	v_lshlrev_b64 v[64:65], 3, v[52:53]
	v_lshl_add_u64 v[66:67], s[12:13], 0, v[64:65]
	global_load_ushort v66, v[66:67], off offset:20
	v_lshl_add_u64 v[64:65], s[14:15], 0, v[64:65]
	s_waitcnt vmcnt(0)
	v_fma_mixlo_f16 v66, s17, v66, v54 op_sel_hi:[0,1,0]
	global_store_short v[64:65], v66, off offset:20
	s_cbranch_execnz .LBB0_81
.LBB0_80:                               ;   in Loop: Header=BB0_16 Depth=1
	v_cvt_f16_f32_e32 v54, v54
	v_lshl_add_u64 v[64:65], v[52:53], 3, s[14:15]
	global_store_short v[64:65], v54, off offset:20
.LBB0_81:                               ;   in Loop: Header=BB0_16 Depth=1
	s_or_b64 exec, exec, s[38:39]
	s_and_saveexec_b64 s[38:39], s[8:9]
	s_cbranch_execz .LBB0_85
; %bb.82:                               ;   in Loop: Header=BB0_16 Depth=1
	s_andn2_b64 vcc, exec, s[26:27]
	v_mul_f32_e32 v54, s16, v71
	s_cbranch_vccnz .LBB0_107
; %bb.83:                               ;   in Loop: Header=BB0_16 Depth=1
	v_lshlrev_b64 v[64:65], 3, v[52:53]
	v_lshl_add_u64 v[66:67], s[12:13], 0, v[64:65]
	global_load_ushort v66, v[66:67], off offset:6
	v_lshl_add_u64 v[64:65], s[14:15], 0, v[64:65]
	s_waitcnt vmcnt(0)
	v_fma_mixlo_f16 v66, s17, v66, v54 op_sel_hi:[0,1,0]
	global_store_short v[64:65], v66, off offset:6
	s_cbranch_execnz .LBB0_85
.LBB0_84:                               ;   in Loop: Header=BB0_16 Depth=1
	v_cvt_f16_f32_e32 v54, v54
	v_lshl_add_u64 v[64:65], v[52:53], 3, s[14:15]
	global_store_short v[64:65], v54, off offset:6
.LBB0_85:                               ;   in Loop: Header=BB0_16 Depth=1
	s_or_b64 exec, exec, s[38:39]
	s_and_saveexec_b64 s[8:9], s[4:5]
	s_cbranch_execz .LBB0_89
; %bb.86:                               ;   in Loop: Header=BB0_16 Depth=1
	s_andn2_b64 vcc, exec, s[26:27]
	v_mul_f32_e32 v54, s16, v70
	s_cbranch_vccnz .LBB0_108
; %bb.87:                               ;   in Loop: Header=BB0_16 Depth=1
	v_lshlrev_b64 v[64:65], 3, v[52:53]
	v_lshl_add_u64 v[66:67], s[12:13], 0, v[64:65]
	global_load_ushort v66, v[66:67], off offset:14
	v_lshl_add_u64 v[64:65], s[14:15], 0, v[64:65]
	s_waitcnt vmcnt(0)
	v_fma_mixlo_f16 v66, s17, v66, v54 op_sel_hi:[0,1,0]
	global_store_short v[64:65], v66, off offset:14
	s_cbranch_execnz .LBB0_89
.LBB0_88:                               ;   in Loop: Header=BB0_16 Depth=1
	v_cvt_f16_f32_e32 v54, v54
	v_lshl_add_u64 v[64:65], v[52:53], 3, s[14:15]
	global_store_short v[64:65], v54, off offset:14
.LBB0_89:                               ;   in Loop: Header=BB0_16 Depth=1
	s_or_b64 exec, exec, s[8:9]
	s_and_b64 exec, exec, s[6:7]
	s_cbranch_execz .LBB0_93
; %bb.90:                               ;   in Loop: Header=BB0_16 Depth=1
	s_andn2_b64 vcc, exec, s[26:27]
	v_mul_f32_e32 v54, s16, v69
	s_cbranch_vccnz .LBB0_109
; %bb.91:                               ;   in Loop: Header=BB0_16 Depth=1
	v_lshlrev_b64 v[64:65], 3, v[52:53]
	v_lshl_add_u64 v[66:67], s[12:13], 0, v[64:65]
	global_load_ushort v66, v[66:67], off offset:22
	v_lshl_add_u64 v[64:65], s[14:15], 0, v[64:65]
	s_waitcnt vmcnt(0)
	v_fma_mixlo_f16 v66, s17, v66, v54 op_sel_hi:[0,1,0]
	global_store_short v[64:65], v66, off offset:22
	s_cbranch_execnz .LBB0_93
.LBB0_92:                               ;   in Loop: Header=BB0_16 Depth=1
	v_cvt_f16_f32_e32 v54, v54
	v_lshl_add_u64 v[64:65], v[52:53], 3, s[14:15]
	global_store_short v[64:65], v54, off offset:22
.LBB0_93:                               ;   in Loop: Header=BB0_16 Depth=1
	s_or_b64 exec, exec, s[36:37]
	v_lshl_add_u64 v[52:53], v[52:53], 0, s[18:19]
	v_lshl_add_u64 v[64:65], v[52:53], 0, 3
	v_cmp_gt_u64_e32 vcc, s[20:21], v[52:53]
	v_cmp_le_u64_e64 s[4:5], s[20:21], v[64:65]
	s_and_b64 s[4:5], vcc, s[4:5]
	s_and_saveexec_b64 s[6:7], s[4:5]
	s_cbranch_execz .LBB0_15
; %bb.94:                               ;   in Loop: Header=BB0_16 Depth=1
	v_cmp_ne_u64_e32 vcc, s[28:29], v[52:53]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_14
; %bb.95:                               ;   in Loop: Header=BB0_16 Depth=1
	v_subrev_co_u32_e32 v52, vcc, s28, v52
	s_mov_b64 s[36:37], 0
	s_nop 0
	v_subbrev_co_u32_e32 v53, vcc, 0, v53, vcc
	s_mov_b64 s[38:39], 0
.LBB0_96:                               ;   Parent Loop BB0_16 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_cmp_lg_u32 s38, 2
	s_cselect_b64 vcc, -1, 0
	s_cmp_lg_u32 s38, 1
	v_cndmask_b32_e32 v50, 0, v50, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_lg_u32 s38, 0
	v_cndmask_b32_e32 v49, 0, v49, vcc
	s_cselect_b64 vcc, -1, 0
	s_add_u32 s38, s38, 1
	s_mov_b32 s24, s38
	s_addc_u32 s39, s39, 0
	v_cmp_ge_u64_e64 s[4:5], s[24:25], v[52:53]
	s_or_b64 s[36:37], s[4:5], s[36:37]
	v_cndmask_b32_e32 v48, 0, v48, vcc
	s_andn2_b64 exec, exec, s[36:37]
	s_cbranch_execnz .LBB0_96
; %bb.97:                               ;   in Loop: Header=BB0_16 Depth=1
	s_or_b64 exec, exec, s[36:37]
	s_branch .LBB0_14
.LBB0_98:                               ;   in Loop: Header=BB0_16 Depth=1
	s_branch .LBB0_48
.LBB0_99:                               ;   in Loop: Header=BB0_16 Depth=1
	s_branch .LBB0_52
.LBB0_100:                              ;   in Loop: Header=BB0_16 Depth=1
	s_branch .LBB0_56
.LBB0_101:                              ;   in Loop: Header=BB0_16 Depth=1
	s_branch .LBB0_60
.LBB0_102:                              ;   in Loop: Header=BB0_16 Depth=1
	s_branch .LBB0_64
.LBB0_103:                              ;   in Loop: Header=BB0_16 Depth=1
	s_branch .LBB0_68
.LBB0_104:                              ;   in Loop: Header=BB0_16 Depth=1
	s_branch .LBB0_72
.LBB0_105:                              ;   in Loop: Header=BB0_16 Depth=1
	s_branch .LBB0_76
.LBB0_106:                              ;   in Loop: Header=BB0_16 Depth=1
	s_branch .LBB0_80
.LBB0_107:                              ;   in Loop: Header=BB0_16 Depth=1
	s_branch .LBB0_84
.LBB0_108:                              ;   in Loop: Header=BB0_16 Depth=1
	s_branch .LBB0_88
.LBB0_109:                              ;   in Loop: Header=BB0_16 Depth=1
	s_branch .LBB0_92
.LBB0_110:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wvSpltK_hf_m4
		.amdhsa_group_segment_fixed_size 65536
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 52
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
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
		.amdhsa_accum_offset 84
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
	.size	wvSpltK_hf_m4, .Lfunc_end0-wvSpltK_hf_m4
                                        ; -- End function
	.set wvSpltK_hf_m4.num_vgpr, 83
	.set wvSpltK_hf_m4.num_agpr, 0
	.set wvSpltK_hf_m4.numbered_sgpr, 44
	.set wvSpltK_hf_m4.num_named_barrier, 0
	.set wvSpltK_hf_m4.private_seg_size, 0
	.set wvSpltK_hf_m4.uses_vcc, 1
	.set wvSpltK_hf_m4.uses_flat_scratch, 0
	.set wvSpltK_hf_m4.has_dyn_sized_stack, 0
	.set wvSpltK_hf_m4.has_recursion, 0
	.set wvSpltK_hf_m4.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 4796
; TotalNumSgprs: 50
; NumVgprs: 83
; NumAgprs: 0
; TotalNumVgprs: 83
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 65536 bytes/workgroup (compile time only)
; SGPRBlocks: 12
; VGPRBlocks: 12
; NumSGPRsForWavesPerEU: 102
; NumVGPRsForWavesPerEU: 97
; AccumOffset: 84
; Occupancy: 4
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 1
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 20
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_ff6c1fb4e79dcd95,@object ; @__hip_cuid_ff6c1fb4e79dcd95
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_ff6c1fb4e79dcd95
__hip_cuid_ff6c1fb4e79dcd95:
	.byte	0                               ; 0x0
	.size	__hip_cuid_ff6c1fb4e79dcd95, 1

	.ident	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.4 26084 f58b06dce1f9c15707c5f808fd002e18c2accf7e)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_ff6c1fb4e79dcd95
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
    DataType: h
    DestDataType: h
    ComputeDataType: s
    HighPrecisionAccumulate: True
    TransposeA: False
    TransposeB: False
    UseBeta: True
    Batched: True
    UseBias: 0
    Activation: False
    UseScaleAlphaVec: 0
  CustomKernel:
    args: [ { type: int32, semantic: SizeSum },
            { type: int32, semantic: SizeFree1 },
            { type: address, semantic: AddressB },
            { type: address, semantic: AddressA },
            { type: address, semantic: AddressC },
            { type: address, semantic: AddressD },
            { type: float32, semantic: Alpha },
            { type: float32, semantic: Beta },
            { type: int32, semantic: ComputeUnits } ]
    macrotile: [64, 16, 8]
    threads: [64, 16, 1]
    grid: [ComputeUnits, One, One]
  MatrixInstruction: [16, 16, 16, 1]
  EnableMatrixInstruction: True
  MIWaveTile: [1, 1]
  AssertSummationElementMultiple: 8
  AssertSizeEqual: { 0: 4, 2: 1 }
  AssertSizeGreaterThan: { 1: 8 }
  AssertSizeLessThan: { 3: 8193 }
  AssertStrideAEqual: { 0: 1, 1: 4 }
  AssertStrideBEqual: { 0: 1 }
  AssertStrideCEqual: { 0: 1, 1: 4 }
  AssertStrideDEqual: { 0: 1, 1: 4 }
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
    .group_segment_fixed_size: 65536
    .kernarg_segment_align: 8
    .kernarg_segment_size: 52
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           wvSpltK_hf_m4
    .private_segment_fixed_size: 0
    .sgpr_count:     50
    .sgpr_spill_count: 0
    .symbol:         wvSpltK_hf_m4.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     83
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
