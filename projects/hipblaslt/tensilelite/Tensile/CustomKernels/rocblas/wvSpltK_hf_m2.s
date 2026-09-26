	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.protected	wvSpltK_hf_m2           ; -- Begin function wvSpltK_hf_m2
	.globl	wvSpltK_hf_m2
	.p2align	8
	.type	wvSpltK_hf_m2,@function
wvSpltK_hf_m2:                          ; @wvSpltK_hf_m2
; %bb.0:
	s_load_dwordx2 s[16:17], s[0:1], 0x0
	v_bfe_u32 v1, v0, 10, 10
	v_lshlrev_b32_e32 v2, 1, v1
	v_lshl_add_u32 v34, s2, 5, v2
	v_mov_b32_e32 v35, 0
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s19, s17, 31
	s_mov_b32 s18, s17
	v_lshl_add_u64 v[2:3], v[34:35], 0, 2
	v_cmp_gt_u64_e32 vcc, s[18:19], v[34:35]
	v_cmp_le_u64_e64 s[2:3], s[18:19], v[2:3]
	v_mov_b32_e32 v32, 1
	s_and_b64 s[4:5], vcc, s[2:3]
	v_mov_b32_e32 v33, v32
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_6
; %bb.1:
	s_add_i32 s4, s18, -2
	s_mov_b32 s7, 0
	s_mov_b32 s5, s7
	v_cmp_ne_u32_e32 vcc, s4, v34
	s_mov_b32 s10, 1
	v_mov_b32_e32 v33, v32
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_5
; %bb.2:
	v_subrev_co_u32_e32 v2, vcc, s4, v34
	s_mov_b64 s[14:15], 0
	s_nop 0
	v_subb_co_u32_e64 v3, s[12:13], 0, 0, vcc
	s_mov_b64 s[12:13], 0
	s_mov_b32 s11, s10
.LBB0_3:                                ; =>This Inner Loop Header: Depth=1
	s_cmp_lg_u32 s14, 1
	s_cselect_b32 s11, s11, 0
	s_cmp_lg_u32 s14, 0
	s_cselect_b32 s10, s10, 0
	s_add_u32 s14, s14, 1
	s_mov_b32 s6, s14
	s_addc_u32 s15, s15, 0
	v_cmp_ge_u64_e32 vcc, s[6:7], v[2:3]
	s_or_b64 s[12:13], vcc, s[12:13]
	v_mov_b64_e32 v[32:33], s[10:11]
	s_andn2_b64 exec, exec, s[12:13]
	s_cbranch_execnz .LBB0_3
; %bb.4:
	s_or_b64 exec, exec, s[12:13]
.LBB0_5:
	s_or_b64 exec, exec, s[8:9]
	v_mov_b64_e32 v[34:35], s[4:5]
.LBB0_6:
	s_or_b64 exec, exec, s[2:3]
	s_lshl_b32 s33, s16, 1
	s_cmp_lg_u32 s16, 0
	v_and_b32_e32 v2, 0x3ff, v0
	s_cselect_b64 s[2:3], -1, 0
	s_cmp_eq_u32 s16, 0
	s_mov_b32 s12, 0
	s_cbranch_scc1 .LBB0_12
; %bb.7:
	s_load_dwordx2 s[4:5], s[0:1], 0x10
	v_and_b32_e32 v0, 1, v2
	v_mov_b32_e32 v3, s16
	v_cmp_eq_u32_e32 vcc, 1, v0
	s_min_i32 s13, s33, 0x8000
	v_lshl_add_u32 v4, v1, 6, v2
	v_cndmask_b32_e32 v0, 0, v3, vcc
	v_lshlrev_b32_e32 v3, 1, v0
	s_mov_b64 s[6:7], 0
	v_mov_b32_e32 v1, 0
                                        ; implicit-def: $sgpr8_sgpr9
	s_branch .LBB0_9
.LBB0_8:                                ;   in Loop: Header=BB0_9 Depth=1
	s_or_b64 exec, exec, s[10:11]
	s_and_b64 s[10:11], exec, s[8:9]
	s_or_b64 s[6:7], s[10:11], s[6:7]
	s_andn2_b64 exec, exec, s[6:7]
	s_cbranch_execz .LBB0_11
.LBB0_9:                                ; =>This Inner Loop Header: Depth=1
	v_add_u32_e32 v0, s12, v4
	v_cmp_gt_u32_e32 vcc, s13, v0
	s_or_b64 s[8:9], s[8:9], exec
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_8
; %bb.10:                               ;   in Loop: Header=BB0_9 Depth=1
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[6:7], v[0:1], 1, s[4:5]
	global_load_ushort v5, v[6:7], off
	s_addk_i32 s12, 0x400
	s_cmp_ge_u32 s12, s13
	s_cselect_b64 s[14:15], -1, 0
	v_and_b32_e32 v0, -2, v0
	s_andn2_b64 s[8:9], s[8:9], exec
	s_and_b64 s[14:15], s[14:15], exec
	v_add_u32_e32 v0, v3, v0
	s_or_b64 s[8:9], s[8:9], s[14:15]
	s_waitcnt vmcnt(0)
	ds_write_b16 v0, v5
	s_branch .LBB0_8
.LBB0_11:
	s_or_b64 exec, exec, s[6:7]
.LBB0_12:
	v_cmp_gt_u64_e32 vcc, s[18:19], v[34:35]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_62
; %bb.13:
	s_load_dwordx8 s[8:15], s[0:1], 0x18
	s_load_dwordx2 s[20:21], s[0:1], 0x8
	s_mov_b32 s23, 0
	v_cndmask_b32_e64 v0, 0, 1, s[2:3]
	v_lshlrev_b32_e32 v42, 3, v2
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s14, s14, 5
	s_ashr_i32 s17, s16, 31
	v_cmp_eq_u32_e64 s[0:1], 63, v2
	v_cmp_neq_f32_e64 s[24:25], s13, 0
	s_ashr_i32 s15, s14, 31
	s_add_i32 s26, s18, -2
	s_mov_b32 s27, s23
	v_lshlrev_b32_e32 v43, 4, v2
	s_mov_b64 s[28:29], 0
	v_cmp_ne_u32_e64 s[2:3], 1, v0
	v_mov_b32_e32 v37, 0
                                        ; implicit-def: $vgpr0_vgpr1_vgpr2_vgpr3
                                        ; implicit-def: $vgpr4_vgpr5_vgpr6_vgpr7
                                        ; implicit-def: $vgpr12_vgpr13_vgpr14_vgpr15
                                        ; implicit-def: $vgpr16_vgpr17_vgpr18_vgpr19
                                        ; implicit-def: $vgpr26_vgpr27
                                        ; implicit-def: $vgpr10_vgpr11
                                        ; implicit-def: $vgpr22_vgpr23
                                        ; implicit-def: $vgpr30_vgpr31
	s_branch .LBB0_16
.LBB0_14:                               ;   in Loop: Header=BB0_16 Depth=1
	s_or_b64 exec, exec, s[30:31]
	v_mov_b64_e32 v[34:35], s[26:27]
.LBB0_15:                               ;   in Loop: Header=BB0_16 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_cmp_le_u64_e32 vcc, s[18:19], v[34:35]
	s_or_b64 s[28:29], vcc, s[28:29]
	s_andn2_b64 exec, exec, s[28:29]
	s_cbranch_execz .LBB0_62
.LBB0_16:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_20 Depth 2
                                        ;     Child Loop BB0_56 Depth 2
	s_and_b64 vcc, exec, s[2:3]
	s_cbranch_vccnz .LBB0_35
; %bb.17:                               ;   in Loop: Header=BB0_16 Depth=1
	v_mul_lo_u32 v36, v35, s16
	v_mul_lo_u32 v40, v34, s17
	v_mad_u64_u32 v[38:39], s[4:5], v34, s16, 0
	v_add3_u32 v39, v39, v40, v36
	v_lshl_add_u64 v[38:39], v[38:39], 1, s[20:21]
	s_mov_b32 s22, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v48, v43
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v47, 0
	s_branch .LBB0_20
.LBB0_18:                               ;   in Loop: Header=BB0_20 Depth=2
	s_or_b64 exec, exec, s[6:7]
.LBB0_19:                               ;   in Loop: Header=BB0_20 Depth=2
	s_or_b64 exec, exec, s[4:5]
	s_addk_i32 s22, 0x400
	s_cmp_ge_u32 s22, s16
	v_add_u32_e32 v48, 0x800, v48
	s_cbranch_scc1 .LBB0_36
.LBB0_20:                               ;   Parent Loop BB0_16 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_add_u32_e32 v36, s22, v42
	v_cmp_gt_u32_e32 vcc, s16, v36
	v_add_u32_e32 v40, 0x200, v36
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execnz .LBB0_24
; %bb.21:                               ;   in Loop: Header=BB0_20 Depth=2
	s_or_b64 exec, exec, s[6:7]
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execnz .LBB0_27
.LBB0_22:                               ;   in Loop: Header=BB0_20 Depth=2
	s_or_b64 exec, exec, s[6:7]
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execnz .LBB0_30
.LBB0_23:                               ;   in Loop: Header=BB0_20 Depth=2
	s_or_b64 exec, exec, s[6:7]
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_19
	s_branch .LBB0_33
.LBB0_24:                               ;   in Loop: Header=BB0_20 Depth=2
	v_lshl_add_u64 v[50:51], v[36:37], 1, v[38:39]
	v_lshl_add_u64 v[52:53], s[16:17], 1, v[50:51]
	global_load_dwordx4 v[4:7], v[50:51], off nt
	global_load_dwordx4 v[16:19], v[52:53], off nt
	v_cmp_gt_u32_e64 s[4:5], s16, v40
	s_and_saveexec_b64 s[30:31], s[4:5]
	s_cbranch_execz .LBB0_26
; %bb.25:                               ;   in Loop: Header=BB0_20 Depth=2
	v_mov_b32_e32 v41, v37
	v_lshl_add_u64 v[50:51], v[40:41], 1, v[38:39]
	v_lshl_add_u64 v[52:53], s[16:17], 1, v[50:51]
	global_load_dwordx4 v[0:3], v[50:51], off nt
	global_load_dwordx4 v[12:15], v[52:53], off nt
.LBB0_26:                               ;   in Loop: Header=BB0_20 Depth=2
	s_or_b64 exec, exec, s[30:31]
	s_or_b64 exec, exec, s[6:7]
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_22
.LBB0_27:                               ;   in Loop: Header=BB0_20 Depth=2
	v_add_u32_e32 v36, s33, v48
	s_waitcnt lgkmcnt(1)
	ds_read_b128 v[28:31], v48
	s_waitcnt lgkmcnt(1)
	ds_read_b128 v[20:23], v36
	v_cmp_gt_u32_e64 s[4:5], s16, v40
	s_and_saveexec_b64 s[30:31], s[4:5]
	s_cbranch_execz .LBB0_29
; %bb.28:                               ;   in Loop: Header=BB0_20 Depth=2
	ds_read_b128 v[8:11], v48 offset:1024
	ds_read_b128 v[24:27], v36 offset:1024
.LBB0_29:                               ;   in Loop: Header=BB0_20 Depth=2
	s_or_b64 exec, exec, s[30:31]
	s_or_b64 exec, exec, s[6:7]
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_23
.LBB0_30:                               ;   in Loop: Header=BB0_20 Depth=2
	s_waitcnt vmcnt(1) lgkmcnt(1)
	;;#ASMSTART
	v_dot2c_f32_f16 v47, v28, v4
	;;#ASMEND
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	v_dot2c_f32_f16 v46, v28, v16
	;;#ASMEND
	v_cmp_gt_u32_e64 s[4:5], s16, v40
	;;#ASMSTART
	v_dot2c_f32_f16 v47, v29, v5
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v46, v29, v17
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v47, v30, v6
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v46, v30, v18
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v47, v31, v7
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v46, v31, v19
	;;#ASMEND
	s_and_saveexec_b64 s[30:31], s[4:5]
	s_cbranch_execz .LBB0_32
; %bb.31:                               ;   in Loop: Header=BB0_20 Depth=2
	;;#ASMSTART
	v_dot2c_f32_f16 v47, v8, v0
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v46, v8, v12
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v47, v9, v1
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v46, v9, v13
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v47, v10, v2
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v46, v10, v14
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v47, v11, v3
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v46, v11, v15
	;;#ASMEND
.LBB0_32:                               ;   in Loop: Header=BB0_20 Depth=2
	s_or_b64 exec, exec, s[30:31]
	s_or_b64 exec, exec, s[6:7]
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_19
.LBB0_33:                               ;   in Loop: Header=BB0_20 Depth=2
	s_waitcnt vmcnt(1) lgkmcnt(0)
	;;#ASMSTART
	v_dot2c_f32_f16 v45, v20, v4
	;;#ASMEND
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	v_dot2c_f32_f16 v44, v20, v16
	;;#ASMEND
	v_cmp_gt_u32_e32 vcc, s16, v40
	;;#ASMSTART
	v_dot2c_f32_f16 v45, v21, v5
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v44, v21, v17
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v45, v22, v6
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v44, v22, v18
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v45, v23, v7
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v44, v23, v19
	;;#ASMEND
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_18
; %bb.34:                               ;   in Loop: Header=BB0_20 Depth=2
	;;#ASMSTART
	v_dot2c_f32_f16 v45, v24, v0
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v44, v24, v12
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v45, v25, v1
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v44, v25, v13
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v45, v26, v2
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v44, v26, v14
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v45, v27, v3
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v44, v27, v15
	;;#ASMEND
	s_branch .LBB0_18
.LBB0_35:                               ;   in Loop: Header=BB0_16 Depth=1
	v_mov_b32_e32 v47, v37
	v_mov_b32_e32 v46, v37
	v_mov_b32_e32 v45, v37
	v_mov_b32_e32 v44, v37
.LBB0_36:                               ;   in Loop: Header=BB0_16 Depth=1
	;;#ASMSTART
	s_nop 0
	v_add_f32 v47, v47, v47 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v46, v46, v46 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v45, v45, v45 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v44, v44, v44 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v47, v47, v47 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v46, v46, v46 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v45, v45, v45 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v44, v44, v44 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v47, v47, v47 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v46, v46, v46 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v45, v45, v45 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v44, v44, v44 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v47, v47, v47 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v46, v46, v46 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v45, v45, v45 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v44, v44, v44 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v47, v47, v47 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v46, v46, v46 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v45, v45, v45 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v44, v44, v44 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v47, v47, v47 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v46, v46, v46 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v45, v45, v45 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v44, v44, v44 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	s_and_saveexec_b64 s[30:31], s[0:1]
	s_cbranch_execz .LBB0_53
; %bb.37:                               ;   in Loop: Header=BB0_16 Depth=1
	v_cmp_ne_u32_e64 s[6:7], 0, v32
	s_and_saveexec_b64 s[4:5], s[6:7]
	s_cbranch_execz .LBB0_41
; %bb.38:                               ;   in Loop: Header=BB0_16 Depth=1
	v_mul_f32_e32 v36, s12, v47
	s_and_b64 vcc, exec, s[24:25]
	s_cbranch_vccz .LBB0_58
; %bb.39:                               ;   in Loop: Header=BB0_16 Depth=1
	v_lshlrev_b64 v[38:39], 2, v[34:35]
	v_lshl_add_u64 v[40:41], s[8:9], 0, v[38:39]
	global_load_ushort v40, v[40:41], off
	v_lshl_add_u64 v[38:39], s[10:11], 0, v[38:39]
	s_waitcnt vmcnt(0)
	v_fma_mixlo_f16 v40, s13, v40, v36 op_sel_hi:[0,1,0]
	global_store_short v[38:39], v40, off
	s_cbranch_execnz .LBB0_41
.LBB0_40:                               ;   in Loop: Header=BB0_16 Depth=1
	v_cvt_f16_f32_e32 v36, v36
	v_lshl_add_u64 v[38:39], v[34:35], 2, s[10:11]
	global_store_short v[38:39], v36, off
.LBB0_41:                               ;   in Loop: Header=BB0_16 Depth=1
	s_or_b64 exec, exec, s[4:5]
	v_cmp_ne_u32_e64 s[4:5], 0, v33
	s_and_saveexec_b64 s[34:35], s[4:5]
	s_cbranch_execz .LBB0_45
; %bb.42:                               ;   in Loop: Header=BB0_16 Depth=1
	s_andn2_b64 vcc, exec, s[24:25]
	v_mul_f32_e32 v36, s12, v46
	s_cbranch_vccnz .LBB0_59
; %bb.43:                               ;   in Loop: Header=BB0_16 Depth=1
	v_lshlrev_b64 v[38:39], 2, v[34:35]
	v_lshl_add_u64 v[40:41], s[8:9], 0, v[38:39]
	global_load_ushort v40, v[40:41], off offset:4
	v_lshl_add_u64 v[38:39], s[10:11], 0, v[38:39]
	s_waitcnt vmcnt(0)
	v_fma_mixlo_f16 v40, s13, v40, v36 op_sel_hi:[0,1,0]
	global_store_short v[38:39], v40, off offset:4
	s_cbranch_execnz .LBB0_45
.LBB0_44:                               ;   in Loop: Header=BB0_16 Depth=1
	v_cvt_f16_f32_e32 v36, v36
	v_lshl_add_u64 v[38:39], v[34:35], 2, s[10:11]
	global_store_short v[38:39], v36, off offset:4
.LBB0_45:                               ;   in Loop: Header=BB0_16 Depth=1
	s_or_b64 exec, exec, s[34:35]
	s_and_saveexec_b64 s[34:35], s[6:7]
	s_cbranch_execz .LBB0_49
; %bb.46:                               ;   in Loop: Header=BB0_16 Depth=1
	s_andn2_b64 vcc, exec, s[24:25]
	v_mul_f32_e32 v36, s12, v45
	s_cbranch_vccnz .LBB0_60
; %bb.47:                               ;   in Loop: Header=BB0_16 Depth=1
	v_lshlrev_b64 v[38:39], 2, v[34:35]
	v_lshl_add_u64 v[40:41], s[8:9], 0, v[38:39]
	global_load_ushort v40, v[40:41], off offset:2
	v_lshl_add_u64 v[38:39], s[10:11], 0, v[38:39]
	s_waitcnt vmcnt(0)
	v_fma_mixlo_f16 v40, s13, v40, v36 op_sel_hi:[0,1,0]
	global_store_short v[38:39], v40, off offset:2
	s_cbranch_execnz .LBB0_49
.LBB0_48:                               ;   in Loop: Header=BB0_16 Depth=1
	v_cvt_f16_f32_e32 v36, v36
	v_lshl_add_u64 v[38:39], v[34:35], 2, s[10:11]
	global_store_short v[38:39], v36, off offset:2
.LBB0_49:                               ;   in Loop: Header=BB0_16 Depth=1
	s_or_b64 exec, exec, s[34:35]
	s_and_b64 exec, exec, s[4:5]
	s_cbranch_execz .LBB0_53
; %bb.50:                               ;   in Loop: Header=BB0_16 Depth=1
	s_andn2_b64 vcc, exec, s[24:25]
	v_mul_f32_e32 v36, s12, v44
	s_cbranch_vccnz .LBB0_61
; %bb.51:                               ;   in Loop: Header=BB0_16 Depth=1
	v_lshlrev_b64 v[38:39], 2, v[34:35]
	v_lshl_add_u64 v[40:41], s[8:9], 0, v[38:39]
	global_load_ushort v40, v[40:41], off offset:6
	v_lshl_add_u64 v[38:39], s[10:11], 0, v[38:39]
	s_waitcnt vmcnt(0)
	v_fma_mixlo_f16 v40, s13, v40, v36 op_sel_hi:[0,1,0]
	global_store_short v[38:39], v40, off offset:6
	s_cbranch_execnz .LBB0_53
.LBB0_52:                               ;   in Loop: Header=BB0_16 Depth=1
	v_cvt_f16_f32_e32 v36, v36
	v_lshl_add_u64 v[38:39], v[34:35], 2, s[10:11]
	global_store_short v[38:39], v36, off offset:6
.LBB0_53:                               ;   in Loop: Header=BB0_16 Depth=1
	s_or_b64 exec, exec, s[30:31]
	v_lshl_add_u64 v[34:35], v[34:35], 0, s[14:15]
	v_lshl_add_u64 v[38:39], v[34:35], 0, 2
	v_cmp_gt_u64_e32 vcc, s[18:19], v[34:35]
	v_cmp_le_u64_e64 s[4:5], s[18:19], v[38:39]
	s_and_b64 s[4:5], vcc, s[4:5]
	s_and_saveexec_b64 s[6:7], s[4:5]
	s_cbranch_execz .LBB0_15
; %bb.54:                               ;   in Loop: Header=BB0_16 Depth=1
	v_cmp_ne_u64_e32 vcc, s[26:27], v[34:35]
	s_and_saveexec_b64 s[30:31], vcc
	s_cbranch_execz .LBB0_14
; %bb.55:                               ;   in Loop: Header=BB0_16 Depth=1
	v_subrev_co_u32_e32 v34, vcc, s26, v34
	s_mov_b64 s[34:35], 0
	s_nop 0
	v_subbrev_co_u32_e32 v35, vcc, 0, v35, vcc
	s_mov_b64 s[36:37], 0
.LBB0_56:                               ;   Parent Loop BB0_16 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_cmp_lg_u32 s36, 1
	s_cselect_b64 vcc, -1, 0
	s_cmp_lg_u32 s36, 0
	v_cndmask_b32_e32 v33, 0, v33, vcc
	s_cselect_b64 vcc, -1, 0
	s_add_u32 s36, s36, 1
	s_mov_b32 s22, s36
	s_addc_u32 s37, s37, 0
	v_cmp_ge_u64_e64 s[4:5], s[22:23], v[34:35]
	s_or_b64 s[34:35], s[4:5], s[34:35]
	v_cndmask_b32_e32 v32, 0, v32, vcc
	s_andn2_b64 exec, exec, s[34:35]
	s_cbranch_execnz .LBB0_56
; %bb.57:                               ;   in Loop: Header=BB0_16 Depth=1
	s_or_b64 exec, exec, s[34:35]
	s_branch .LBB0_14
.LBB0_58:                               ;   in Loop: Header=BB0_16 Depth=1
	s_branch .LBB0_40
.LBB0_59:                               ;   in Loop: Header=BB0_16 Depth=1
	s_branch .LBB0_44
.LBB0_60:                               ;   in Loop: Header=BB0_16 Depth=1
	s_branch .LBB0_48
.LBB0_61:                               ;   in Loop: Header=BB0_16 Depth=1
	s_branch .LBB0_52
.LBB0_62:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wvSpltK_hf_m2
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
		.amdhsa_accum_offset 56
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
	.size	wvSpltK_hf_m2, .Lfunc_end0-wvSpltK_hf_m2
                                        ; -- End function
	.set wvSpltK_hf_m2.num_vgpr, 54
	.set wvSpltK_hf_m2.num_agpr, 0
	.set wvSpltK_hf_m2.numbered_sgpr, 38
	.set wvSpltK_hf_m2.num_named_barrier, 0
	.set wvSpltK_hf_m2.private_seg_size, 0
	.set wvSpltK_hf_m2.uses_vcc, 1
	.set wvSpltK_hf_m2.uses_flat_scratch, 0
	.set wvSpltK_hf_m2.has_dyn_sized_stack, 0
	.set wvSpltK_hf_m2.has_recursion, 0
	.set wvSpltK_hf_m2.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 2256
; TotalNumSgprs: 44
; NumVgprs: 54
; NumAgprs: 0
; TotalNumVgprs: 54
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 65536 bytes/workgroup (compile time only)
; SGPRBlocks: 12
; VGPRBlocks: 12
; NumSGPRsForWavesPerEU: 102
; NumVGPRsForWavesPerEU: 97
; AccumOffset: 56
; Occupancy: 4
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
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
	.type	__hip_cuid_1278202c0dda20e6,@object ; @__hip_cuid_1278202c0dda20e6
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_1278202c0dda20e6
__hip_cuid_1278202c0dda20e6:
	.byte	0                               ; 0x0
	.size	__hip_cuid_1278202c0dda20e6, 1

	.ident	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.4 26084 f58b06dce1f9c15707c5f808fd002e18c2accf7e)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_1278202c0dda20e6
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
  AssertSizeEqual: { 0: 2, 2: 1 }
  AssertSizeGreaterThan: { 1: 8 }
  AssertSizeLessThan: { 3: 16385 }
  AssertStrideAEqual: { 0: 1, 1: 2 }
  AssertStrideBEqual: { 0: 1 }
  AssertStrideCEqual: { 0: 1, 1: 2 }
  AssertStrideDEqual: { 0: 1, 1: 2 }
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
    .name:           wvSpltK_hf_m2
    .private_segment_fixed_size: 0
    .sgpr_count:     44
    .sgpr_spill_count: 0
    .symbol:         wvSpltK_hf_m2.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     54
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
