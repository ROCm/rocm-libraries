	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.protected	wvSpltK_hf_m4           ; -- Begin function wvSpltK_hf_m4
	.globl	wvSpltK_hf_m4
	.p2align	8
	.type	wvSpltK_hf_m4,@function
wvSpltK_hf_m4:                          ; @wvSpltK_hf_m4
; %bb.156:
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.157:
.LBB0_0:
	v_bfe_u32 v9, v0, 10, 10
	v_lshl_add_u32 v1, s16, 4, v9
	v_lshl_add_u32 v60, v1, 1, v1
	v_mov_b32_e32 v61, 0
	s_mov_b32 s20, s3
	s_ashr_i32 s21, s3, 31
	v_lshl_add_u64 v[2:3], v[60:61], 0, 3
	v_cmp_gt_u64_e32 vcc, s[20:21], v[60:61]
	v_cmp_le_u64_e64 s[16:17], s[20:21], v[2:3]
	v_mov_b32_e32 v56, 1
	s_and_b64 s[16:17], vcc, s[16:17]
	v_mov_b32_e32 v57, v56
	v_mov_b32_e32 v58, v56
	s_and_saveexec_b64 s[22:23], s[16:17]
	s_cbranch_execz .LBB0_6
; %bb.1:
	s_add_i32 s24, s20, -3
	s_mov_b32 s27, 0
	s_mov_b32 s25, s27
	v_cmp_ne_u32_e32 vcc, s24, v60
	s_mov_b32 s16, 1
	v_mov_b32_e32 v57, v56
	v_mov_b32_e32 v58, v56
	s_and_saveexec_b64 s[28:29], vcc
	s_cbranch_execz .LBB0_5
; %bb.2:
	v_subrev_co_u32_e32 v2, vcc, s24, v60
	s_mov_b64 s[30:31], 0
	s_nop 0
	v_subb_co_u32_e64 v3, s[18:19], 0, 0, vcc
	s_mov_b64 s[34:35], 0
	s_mov_b32 s17, s16
	s_mov_b32 s18, s16
.LBB0_3:                                ; =>This Inner Loop Header: Depth=1
	s_cmp_lg_u32 s34, 2
	s_cselect_b32 s18, s18, 0
	s_cmp_lg_u32 s34, 1
	s_cselect_b32 s17, s17, 0
	s_cmp_lg_u32 s34, 0
	s_cselect_b32 s16, s16, 0
	s_add_u32 s34, s34, 1
	s_mov_b32 s26, s34
	s_addc_u32 s35, s35, 0
	v_cmp_ge_u64_e32 vcc, s[26:27], v[2:3]
	v_mov_b32_e32 v58, s18
	s_or_b64 s[30:31], vcc, s[30:31]
	v_mov_b32_e32 v57, s17
	v_mov_b32_e32 v56, s16
	s_andn2_b64 exec, exec, s[30:31]
	s_cbranch_execnz .LBB0_3
; %bb.4:
	s_or_b64 exec, exec, s[30:31]
.LBB0_5:
	s_or_b64 exec, exec, s[28:29]
	v_mov_b64_e32 v[60:61], s[24:25]
.LBB0_6:
	s_or_b64 exec, exec, s[22:23]
	v_lshlrev_b32_e32 v10, 6, v9
	v_and_b32_e32 v8, 0x3ff, v0
	v_add_u32_e32 v0, v10, v8
	s_mov_b64 s[22:23], -1
	s_mov_b64 s[16:17], 0
	s_cmp_lt_i32 s9, 2
	s_mov_b64 s[18:19], 0
	s_cbranch_scc0 .LBB0_10
; %bb.7:
	s_and_b64 vcc, exec, s[22:23]
	s_cbranch_vccnz .LBB0_35
.LBB0_8:
	s_andn2_b64 vcc, exec, s[18:19]
	s_cbranch_vccz .LBB0_36
.LBB0_9:
	s_andn2_b64 vcc, exec, s[16:17]
	s_cbranch_vccz .LBB0_47
	s_branch .LBB0_62
.LBB0_10:
	s_cmp_gt_i32 s9, 2
	s_cbranch_scc0 .LBB0_22
; %bb.11:
	s_cmp_eq_u32 s9, 3
	s_mov_b64 s[18:19], -1
	s_cbranch_scc0 .LBB0_21
; %bb.12:
	s_mul_i32 s3, s2, 3
	v_cmp_gt_u32_e32 vcc, s3, v0
	s_and_saveexec_b64 s[18:19], vcc
	s_cbranch_execz .LBB0_20
; %bb.13:
	v_add_u32_e32 v1, 0x400, v0
	v_max_u32_e32 v2, s3, v1
	v_xad_u32 v2, v8, -1, v2
	v_sub_u32_e32 v3, v2, v10
	s_movk_i32 s22, 0xbff
	s_ashr_i32 s26, s11, 31
	v_cmp_lt_u32_e32 vcc, s22, v3
	s_mov_b64 s[24:25], -1
	v_mov_b32_e32 v2, v0
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_17
; %bb.14:
	v_lshrrev_b32_e32 v2, 10, v3
	v_add_u32_e32 v11, 1, v2
	v_add_u32_e32 v3, 0xc00, v0
	v_add_u32_e32 v2, 0x800, v0
	v_and_b32_e32 v12, 0x7ffffc, v11
	v_mov_b64_e32 v[4:5], v[2:3]
	s_mov_b32 s27, s11
	s_mov_b32 s28, s26
	s_mov_b32 s29, s11
	s_mov_b32 s30, s26
	s_mov_b32 s31, s11
	s_mov_b32 s33, s26
	s_mov_b32 s34, s2
	s_mov_b32 s35, s2
	s_mov_b32 s36, s2
	s_mov_b64 s[24:25], 0
	s_mov_b32 s37, 0xaaaaaaab
	v_mov_b32_e32 v7, 0
	v_mov_b32_e32 v13, v12
	v_mov_b64_e32 v[2:3], v[0:1]
.LBB0_15:                               ; =>This Inner Loop Header: Depth=1
	v_mul_hi_u32 v1, v2, s37
	v_mul_hi_u32 v36, v3, s37
	v_mul_hi_u32 v37, v4, s37
	v_mul_hi_u32 v38, v5, s37
	v_lshrrev_b32_e32 v35, 1, v1
	v_lshrrev_b32_e32 v33, 1, v36
	v_lshrrev_b32_e32 v31, 1, v37
	v_lshrrev_b32_e32 v29, 1, v38
	v_mad_u64_u32 v[26:27], s[38:39], s11, v35, 0
	v_mad_u64_u32 v[20:21], s[38:39], s31, v29, 0
	v_mad_u64_u32 v[22:23], s[38:39], s29, v31, 0
	v_mad_u64_u32 v[24:25], s[38:39], s27, v33, 0
	v_mov_b32_e32 v34, v27
	v_lshl_add_u32 v6, v35, 1, v35
	v_mov_b32_e32 v28, v21
	v_mov_b32_e32 v30, v23
	v_mov_b32_e32 v32, v25
	v_mad_u64_u32 v[34:35], s[38:39], s26, v35, v[34:35]
	v_lshl_add_u32 v14, v33, 1, v33
	v_lshl_add_u32 v16, v31, 1, v31
	v_lshl_add_u32 v18, v29, 1, v29
	v_mad_u64_u32 v[28:29], s[38:39], s33, v29, v[28:29]
	v_mad_u64_u32 v[30:31], s[38:39], s30, v31, v[30:31]
	v_mad_u64_u32 v[32:33], s[38:39], s28, v33, v[32:33]
	v_mov_b32_e32 v27, v34
	v_sub_u32_e32 v6, v2, v6
	v_mov_b32_e32 v21, v28
	v_mov_b32_e32 v23, v30
	v_mov_b32_e32 v25, v32
	v_lshl_add_u64 v[26:27], v[26:27], 1, s[6:7]
	v_mov_b32_e32 v15, v7
	v_mov_b32_e32 v17, v7
	v_mov_b32_e32 v19, v7
	v_sub_u32_e32 v18, v5, v18
	v_sub_u32_e32 v16, v4, v16
	v_sub_u32_e32 v14, v3, v14
	v_lshl_add_u64 v[24:25], v[24:25], 1, s[6:7]
	v_lshl_add_u64 v[22:23], v[22:23], 1, s[6:7]
	v_lshl_add_u64 v[20:21], v[20:21], 1, s[6:7]
	v_lshl_add_u64 v[26:27], v[6:7], 1, v[26:27]
	v_lshl_add_u64 v[24:25], v[14:15], 1, v[24:25]
	v_lshl_add_u64 v[22:23], v[16:17], 1, v[22:23]
	v_lshl_add_u64 v[20:21], v[18:19], 1, v[20:21]
	global_load_ushort v15, v[26:27], off
	global_load_ushort v17, v[24:25], off
	global_load_ushort v19, v[22:23], off
	global_load_ushort v28, v[20:21], off
	v_add_u32_e32 v13, -4, v13
	v_and_b32_e32 v1, -2, v1
	v_cmp_eq_u32_e32 vcc, 0, v13
	v_mul_lo_u32 v6, v6, s2
	v_and_b32_e32 v20, -2, v36
	v_and_b32_e32 v21, -2, v37
	v_and_b32_e32 v22, -2, v38
	s_or_b64 s[24:25], vcc, s[24:25]
	v_add_u32_e32 v5, 0x1000, v5
	v_add_u32_e32 v4, 0x1000, v4
	v_add_u32_e32 v3, 0x1000, v3
	v_add_u32_e32 v2, 0x1000, v2
	v_mul_lo_u32 v18, v18, s36
	v_mul_lo_u32 v16, v16, s35
	v_mul_lo_u32 v14, v14, s34
	v_lshl_add_u32 v1, v6, 1, v1
	v_lshl_add_u32 v6, v14, 1, v20
	v_lshl_add_u32 v14, v16, 1, v21
	v_lshl_add_u32 v16, v18, 1, v22
	s_waitcnt vmcnt(3)
	ds_write_b16 v1, v15
	s_waitcnt vmcnt(2)
	ds_write_b16 v6, v17
	s_waitcnt vmcnt(1)
	ds_write_b16 v14, v19
	s_waitcnt vmcnt(0)
	ds_write_b16 v16, v28
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_15
; %bb.16:
	s_or_b64 exec, exec, s[24:25]
	v_cmp_ne_u32_e32 vcc, v11, v12
	v_lshl_add_u32 v2, v12, 10, v0
	s_orn2_b64 s[24:25], vcc, exec
.LBB0_17:
	s_or_b64 exec, exec, s[22:23]
	s_and_b64 exec, exec, s[24:25]
	s_cbranch_execz .LBB0_20
; %bb.18:
	s_mov_b64 s[22:23], 0
	s_mov_b32 s24, 0xaaaaaaab
	v_mov_b32_e32 v1, 0
.LBB0_19:                               ; =>This Inner Loop Header: Depth=1
	v_mul_hi_u32 v3, v2, s24
	v_lshrrev_b32_e32 v6, 1, v3
	v_mad_u64_u32 v[4:5], s[26:27], v6, -3, v[2:3]
	v_mad_i64_i32 v[6:7], s[26:27], v6, s11, 0
	v_mov_b32_e32 v5, v1
	v_lshl_add_u64 v[6:7], v[6:7], 1, s[6:7]
	v_lshl_add_u64 v[6:7], v[4:5], 1, v[6:7]
	global_load_ushort v5, v[6:7], off
	v_add_u32_e32 v2, 0x400, v2
	v_and_b32_e32 v3, -2, v3
	v_mul_lo_u32 v4, v4, s2
	v_cmp_le_u32_e32 vcc, s3, v2
	v_lshl_add_u32 v3, v4, 1, v3
	s_or_b64 s[22:23], vcc, s[22:23]
	s_waitcnt vmcnt(0)
	ds_write_b16 v3, v5
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB0_19
.LBB0_20:
	s_or_b64 exec, exec, s[18:19]
	s_mov_b64 s[18:19], 0
.LBB0_21:
	s_mov_b64 s[22:23], 0
.LBB0_22:
	s_and_b64 vcc, exec, s[22:23]
	s_cbranch_vccz .LBB0_34
; %bb.23:
	s_lshl_b32 s3, s2, 1
	v_cmp_gt_u32_e32 vcc, s3, v0
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_33
; %bb.24:
	v_and_b32_e32 v1, 1, v8
	v_lshlrev_b32_e32 v2, 1, v1
	v_mul_lo_u32 v1, s2, v1
	v_mov_b32_e32 v3, 0
	v_lshlrev_b32_e32 v11, 1, v1
	v_add_u32_e32 v1, 0x400, v0
	v_lshl_add_u64 v[6:7], s[6:7], 0, v[2:3]
	v_max_u32_e32 v3, s3, v1
	v_xad_u32 v1, v8, -1, v3
	v_sub_u32_e32 v2, v1, v10
	s_movk_i32 s24, 0x2c00
	s_movk_i32 s26, 0x2bff
	s_ashr_i32 s33, s11, 31
	v_cmp_gt_u32_e64 s[24:25], s24, v2
	v_cmp_lt_u32_e32 vcc, s26, v2
	v_mov_b32_e32 v1, v0
	s_and_saveexec_b64 s[26:27], vcc
	s_cbranch_execz .LBB0_30
; %bb.25:
	v_sub_u32_e32 v1, v8, v3
	v_add_u32_e32 v1, v1, v10
	v_or_b32_e32 v1, 0x3ff, v1
	v_cmp_ge_u32_e32 vcc, v1, v0
	s_mov_b64 s[30:31], -1
	v_mov_b32_e32 v1, v0
	s_and_saveexec_b64 s[28:29], vcc
	s_cbranch_execz .LBB0_29
; %bb.26:
	v_lshrrev_b32_e32 v1, 10, v2
	v_add_u32_e32 v12, 1, v1
	v_add_u32_e32 v3, 0xc00, v0
	v_add_u32_e32 v2, 0x800, v0
	v_and_b32_e32 v13, 0x7ffffc, v12
	v_add_u32_e32 v1, 0x400, v0
	v_mov_b64_e32 v[4:5], v[2:3]
	s_mov_b32 s34, s11
	s_mov_b32 s35, s33
	s_mov_b32 s36, s11
	s_mov_b32 s37, s33
	s_mov_b32 s38, s11
	s_mov_b32 s39, s33
	s_mov_b64 s[30:31], 0
	v_mov_b32_e32 v14, v13
	v_mov_b64_e32 v[2:3], v[0:1]
.LBB0_27:                               ; =>This Inner Loop Header: Depth=1
	v_lshrrev_b32_e32 v1, 1, v2
	v_lshrrev_b32_e32 v15, 1, v3
	v_lshrrev_b32_e32 v27, 1, v4
	v_lshrrev_b32_e32 v25, 1, v5
	v_mad_u64_u32 v[22:23], s[40:41], s11, v1, 0
	v_mad_u64_u32 v[16:17], s[40:41], s38, v25, 0
	v_mad_u64_u32 v[18:19], s[40:41], s36, v27, 0
	v_mad_u64_u32 v[20:21], s[40:41], s34, v15, 0
	v_mov_b32_e32 v30, v23
	v_mov_b32_e32 v24, v17
	v_mov_b32_e32 v26, v19
	v_mov_b32_e32 v28, v21
	v_mad_u64_u32 v[30:31], s[40:41], s33, v1, v[30:31]
	v_mad_u64_u32 v[24:25], s[40:41], s39, v25, v[24:25]
	v_mad_u64_u32 v[26:27], s[40:41], s37, v27, v[26:27]
	v_mad_u64_u32 v[28:29], s[40:41], s35, v15, v[28:29]
	v_mov_b32_e32 v23, v30
	v_mov_b32_e32 v17, v24
	v_mov_b32_e32 v19, v26
	v_mov_b32_e32 v21, v28
	v_lshl_add_u64 v[22:23], v[22:23], 1, v[6:7]
	v_lshl_add_u64 v[20:21], v[20:21], 1, v[6:7]
	v_lshl_add_u64 v[18:19], v[18:19], 1, v[6:7]
	v_lshl_add_u64 v[16:17], v[16:17], 1, v[6:7]
	global_load_ushort v1, v[22:23], off
	global_load_ushort v15, v[20:21], off
	global_load_ushort v24, v[18:19], off
	global_load_ushort v25, v[16:17], off
	v_add_u32_e32 v14, -4, v14
	v_and_b32_e32 v16, -2, v2
	v_cmp_eq_u32_e32 vcc, 0, v14
	v_and_b32_e32 v17, -2, v3
	v_and_b32_e32 v18, -2, v4
	v_and_b32_e32 v19, -2, v5
	v_add_u32_e32 v5, 0x1000, v5
	v_add_u32_e32 v4, 0x1000, v4
	v_add_u32_e32 v3, 0x1000, v3
	v_add_u32_e32 v2, 0x1000, v2
	v_add_u32_e32 v16, v11, v16
	s_or_b64 s[30:31], vcc, s[30:31]
	v_add_u32_e32 v17, v11, v17
	v_add_u32_e32 v18, v11, v18
	v_add_u32_e32 v19, v11, v19
	s_waitcnt vmcnt(3)
	ds_write_b16 v16, v1
	s_waitcnt vmcnt(2)
	ds_write_b16 v17, v15
	s_waitcnt vmcnt(1)
	ds_write_b16 v18, v24
	s_waitcnt vmcnt(0)
	ds_write_b16 v19, v25
	s_andn2_b64 exec, exec, s[30:31]
	s_cbranch_execnz .LBB0_27
; %bb.28:
	s_or_b64 exec, exec, s[30:31]
	v_cmp_ne_u32_e32 vcc, v12, v13
	v_lshl_add_u32 v1, v13, 10, v0
	s_orn2_b64 s[30:31], vcc, exec
.LBB0_29:
	s_or_b64 exec, exec, s[28:29]
	s_andn2_b64 s[24:25], s[24:25], exec
	s_and_b64 s[28:29], s[30:31], exec
	s_or_b64 s[24:25], s[24:25], s[28:29]
.LBB0_30:
	s_or_b64 exec, exec, s[26:27]
	s_and_b64 exec, exec, s[24:25]
	s_cbranch_execz .LBB0_33
; %bb.31:
	s_mov_b64 s[24:25], 0
.LBB0_32:                               ; =>This Inner Loop Header: Depth=1
	v_lshrrev_b32_e32 v2, 1, v1
	v_mad_i64_i32 v[2:3], s[26:27], v2, s11, 0
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[6:7]
	global_load_ushort v2, v[2:3], off
	v_and_b32_e32 v3, -2, v1
	v_add_u32_e32 v1, 0x400, v1
	v_cmp_le_u32_e32 vcc, s3, v1
	v_add_u32_e32 v3, v11, v3
	s_or_b64 s[24:25], vcc, s[24:25]
	s_waitcnt vmcnt(0)
	ds_write_b16 v3, v2
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_32
.LBB0_33:
	s_or_b64 exec, exec, s[22:23]
.LBB0_34:
	s_branch .LBB0_8
.LBB0_35:
	s_cmp_lg_u32 s9, 1
	s_mov_b64 s[16:17], -1
	s_cselect_b64 s[18:19], -1, 0
	s_andn2_b64 vcc, exec, s[18:19]
	s_cbranch_vccnz .LBB0_9
.LBB0_36:
	s_lshl_b32 s3, s2, 2
	v_cmp_gt_u32_e32 vcc, s3, v0
	s_and_saveexec_b64 s[16:17], vcc
	s_cbranch_execz .LBB0_46
; %bb.37:
	v_and_b32_e32 v1, 3, v8
	v_lshlrev_b32_e32 v2, 1, v1
	v_mul_lo_u32 v1, s2, v1
	v_mov_b32_e32 v3, 0
	v_lshlrev_b32_e32 v11, 1, v1
	v_add_u32_e32 v1, 0x400, v0
	v_lshl_add_u64 v[6:7], s[6:7], 0, v[2:3]
	v_max_u32_e32 v3, s3, v1
	v_xad_u32 v1, v8, -1, v3
	v_sub_u32_e32 v2, v1, v10
	s_movk_i32 s18, 0x2c00
	s_movk_i32 s22, 0x2bff
	s_ashr_i32 s28, s11, 31
	v_cmp_gt_u32_e64 s[18:19], s18, v2
	v_cmp_lt_u32_e32 vcc, s22, v2
	v_mov_b32_e32 v1, v0
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_43
; %bb.38:
	v_sub_u32_e32 v1, v8, v3
	v_add_u32_e32 v1, v1, v10
	v_or_b32_e32 v1, 0x3ff, v1
	v_cmp_ge_u32_e32 vcc, v1, v0
	s_mov_b64 s[26:27], -1
	v_mov_b32_e32 v1, v0
	s_and_saveexec_b64 s[24:25], vcc
	s_cbranch_execz .LBB0_42
; %bb.39:
	v_lshrrev_b32_e32 v1, 10, v2
	v_add_u32_e32 v12, 1, v1
	v_add_u32_e32 v3, 0xc00, v0
	v_add_u32_e32 v2, 0x800, v0
	v_and_b32_e32 v13, 0x7ffffc, v12
	v_add_u32_e32 v1, 0x400, v0
	v_mov_b64_e32 v[4:5], v[2:3]
	s_mov_b32 s29, s11
	s_mov_b32 s30, s28
	s_mov_b32 s31, s11
	s_mov_b32 s33, s28
	s_mov_b32 s34, s11
	s_mov_b32 s35, s28
	s_mov_b64 s[26:27], 0
	v_mov_b32_e32 v14, v13
	v_mov_b64_e32 v[2:3], v[0:1]
.LBB0_40:                               ; =>This Inner Loop Header: Depth=1
	v_lshrrev_b32_e32 v1, 2, v2
	v_lshrrev_b32_e32 v15, 2, v3
	v_lshrrev_b32_e32 v32, 2, v4
	v_lshrrev_b32_e32 v33, 2, v5
	v_mad_u64_u32 v[22:23], s[36:37], s11, v1, 0
	v_mad_u64_u32 v[16:17], s[36:37], s34, v33, 0
	v_mad_u64_u32 v[18:19], s[36:37], s31, v32, 0
	v_mad_u64_u32 v[20:21], s[36:37], s29, v15, 0
	v_mov_b32_e32 v30, v23
	v_mov_b32_e32 v24, v17
	v_mov_b32_e32 v26, v19
	v_mov_b32_e32 v28, v21
	v_mad_u64_u32 v[30:31], s[36:37], s28, v1, v[30:31]
	v_mad_u64_u32 v[24:25], s[36:37], s35, v33, v[24:25]
	v_mad_u64_u32 v[26:27], s[36:37], s33, v32, v[26:27]
	v_mad_u64_u32 v[28:29], s[36:37], s30, v15, v[28:29]
	v_mov_b32_e32 v23, v30
	v_mov_b32_e32 v17, v24
	v_mov_b32_e32 v19, v26
	v_mov_b32_e32 v21, v28
	v_lshl_add_u64 v[22:23], v[22:23], 1, v[6:7]
	v_lshl_add_u64 v[20:21], v[20:21], 1, v[6:7]
	v_lshl_add_u64 v[18:19], v[18:19], 1, v[6:7]
	v_lshl_add_u64 v[16:17], v[16:17], 1, v[6:7]
	global_load_ushort v24, v[22:23], off
	global_load_ushort v25, v[20:21], off
	global_load_ushort v26, v[18:19], off
	global_load_ushort v27, v[16:17], off
	v_add_u32_e32 v14, -4, v14
	v_cmp_eq_u32_e32 vcc, 0, v14
	v_add_u32_e32 v5, 0x1000, v5
	v_add_u32_e32 v4, 0x1000, v4
	v_add_u32_e32 v3, 0x1000, v3
	v_add_u32_e32 v2, 0x1000, v2
	v_lshl_add_u32 v1, v1, 1, v11
	s_or_b64 s[26:27], vcc, s[26:27]
	v_lshl_add_u32 v15, v15, 1, v11
	v_lshl_add_u32 v16, v32, 1, v11
	v_lshl_add_u32 v17, v33, 1, v11
	s_waitcnt vmcnt(3)
	ds_write_b16 v1, v24
	s_waitcnt vmcnt(2)
	ds_write_b16 v15, v25
	s_waitcnt vmcnt(1)
	ds_write_b16 v16, v26
	s_waitcnt vmcnt(0)
	ds_write_b16 v17, v27
	s_andn2_b64 exec, exec, s[26:27]
	s_cbranch_execnz .LBB0_40
; %bb.41:
	s_or_b64 exec, exec, s[26:27]
	v_cmp_ne_u32_e32 vcc, v12, v13
	v_lshl_add_u32 v1, v13, 10, v0
	s_orn2_b64 s[26:27], vcc, exec
.LBB0_42:
	s_or_b64 exec, exec, s[24:25]
	s_andn2_b64 s[18:19], s[18:19], exec
	s_and_b64 s[24:25], s[26:27], exec
	s_or_b64 s[18:19], s[18:19], s[24:25]
.LBB0_43:
	s_or_b64 exec, exec, s[22:23]
	s_and_b64 exec, exec, s[18:19]
	s_cbranch_execz .LBB0_46
; %bb.44:
	s_mov_b64 s[18:19], 0
.LBB0_45:                               ; =>This Inner Loop Header: Depth=1
	v_lshrrev_b32_e32 v4, 2, v1
	v_mad_i64_i32 v[2:3], s[22:23], v4, s11, 0
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[6:7]
	global_load_ushort v2, v[2:3], off
	v_add_u32_e32 v1, 0x400, v1
	v_cmp_le_u32_e32 vcc, s3, v1
	v_lshl_add_u32 v3, v4, 1, v11
	s_or_b64 s[18:19], vcc, s[18:19]
	s_waitcnt vmcnt(0)
	ds_write_b16 v3, v2
	s_andn2_b64 exec, exec, s[18:19]
	s_cbranch_execnz .LBB0_45
.LBB0_46:
	s_or_b64 exec, exec, s[16:17]
	s_cbranch_execnz .LBB0_62
.LBB0_47:
	v_cmp_gt_u32_e32 vcc, s2, v0
	s_and_saveexec_b64 s[16:17], vcc
	s_cbranch_execz .LBB0_61
; %bb.48:
	v_add_u32_e32 v1, 0x400, v0
	v_max_u32_e32 v2, s2, v1
	v_xad_u32 v1, v8, -1, v2
	v_sub_u32_e32 v1, v1, v10
	s_movk_i32 s18, 0x2c00
	s_movk_i32 s22, 0x2bff
	s_ashr_i32 s3, s11, 31
	v_cmp_gt_u32_e64 s[18:19], s18, v1
	v_cmp_lt_u32_e32 vcc, s22, v1
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_58
; %bb.49:
	v_sub_u32_e32 v2, v8, v2
	v_add_u32_e32 v2, v2, v10
	v_or_b32_e32 v2, 0x3ff, v2
	v_cmp_ge_u32_e32 vcc, v2, v0
	s_mov_b64 s[26:27], -1
	s_and_saveexec_b64 s[24:25], vcc
	s_cbranch_execz .LBB0_57
; %bb.50:
	v_lshrrev_b32_e32 v10, 10, v1
	v_add_u32_e32 v3, 0xc00, v0
	v_add_u32_e32 v2, 0x800, v0
	v_add_u32_e32 v1, 0x400, v0
	v_add_u32_e32 v11, -3, v10
	v_mov_b64_e32 v[6:7], v[2:3]
	v_cmp_lt_u32_e32 vcc, 3, v11
	v_mov_b32_e32 v12, 0
	v_mov_b64_e32 v[4:5], v[0:1]
	s_and_saveexec_b64 s[26:27], vcc
	s_cbranch_execz .LBB0_54
; %bb.51:
	v_lshrrev_b32_e32 v4, 2, v11
	v_add_u32_e32 v4, 1, v4
	v_and_b32_e32 v12, 0x7ffffffe, v4
	v_lshlrev_b32_e32 v4, 1, v8
	v_lshl_add_u32 v9, v9, 7, v4
	v_mov_b64_e32 v[6:7], v[2:3]
	s_mov_b32 s30, 0
	s_mov_b64 s[28:29], 0
	v_mov_b64_e32 v[4:5], v[0:1]
.LBB0_52:                               ; =>This Inner Loop Header: Depth=1
	v_mad_u64_u32 v[2:3], s[34:35], s11, v7, 0
	v_mad_u64_u32 v[14:15], s[34:35], s11, v6, 0
	v_mad_u64_u32 v[16:17], s[34:35], s11, v5, 0
	v_mad_u64_u32 v[18:19], s[34:35], s11, v4, 0
	v_add_u32_e32 v1, 0x1000, v4
	v_add_u32_e32 v13, 0x1000, v5
	v_add_u32_e32 v37, 0x1000, v6
	v_add_u32_e32 v39, 0x1000, v7
	v_mov_b32_e32 v20, v3
	v_mov_b32_e32 v22, v15
	v_mov_b32_e32 v24, v17
	v_mov_b32_e32 v26, v19
	v_mad_u64_u32 v[28:29], s[34:35], s11, v39, 0
	v_mad_u64_u32 v[30:31], s[34:35], s11, v37, 0
	v_mad_u64_u32 v[32:33], s[34:35], s11, v13, 0
	v_mad_u64_u32 v[34:35], s[34:35], s11, v1, 0
	v_mad_u64_u32 v[20:21], s[34:35], s3, v7, v[20:21]
	v_mad_u64_u32 v[22:23], s[34:35], s3, v6, v[22:23]
	v_mad_u64_u32 v[24:25], s[34:35], s3, v5, v[24:25]
	v_mad_u64_u32 v[26:27], s[34:35], s3, v4, v[26:27]
	v_mov_b32_e32 v36, v29
	v_mov_b32_e32 v38, v31
	v_mov_b32_e32 v40, v33
	v_mov_b32_e32 v42, v35
	v_mov_b32_e32 v3, v20
	v_mov_b32_e32 v15, v22
	v_mov_b32_e32 v17, v24
	v_mov_b32_e32 v19, v26
	v_mad_u64_u32 v[20:21], s[34:35], s3, v39, v[36:37]
	v_mad_u64_u32 v[22:23], s[34:35], s3, v37, v[38:39]
	v_mad_u64_u32 v[24:25], s[34:35], s3, v13, v[40:41]
	v_mad_u64_u32 v[26:27], s[34:35], s3, v1, v[42:43]
	v_lshl_add_u64 v[18:19], v[18:19], 1, s[6:7]
	v_mov_b32_e32 v29, v20
	v_mov_b32_e32 v31, v22
	v_mov_b32_e32 v33, v24
	v_mov_b32_e32 v35, v26
	v_lshl_add_u64 v[16:17], v[16:17], 1, s[6:7]
	v_lshl_add_u64 v[14:15], v[14:15], 1, s[6:7]
	v_lshl_add_u64 v[2:3], v[2:3], 1, s[6:7]
	v_lshl_add_u64 v[20:21], v[34:35], 1, s[6:7]
	v_lshl_add_u64 v[22:23], v[32:33], 1, s[6:7]
	v_lshl_add_u64 v[24:25], v[30:31], 1, s[6:7]
	v_lshl_add_u64 v[26:27], v[28:29], 1, s[6:7]
	global_load_ushort v13, v[18:19], off
	global_load_ushort v28, v[16:17], off
	global_load_ushort v29, v[14:15], off
	global_load_ushort v30, v[2:3], off
	global_load_ushort v31, v[20:21], off
	global_load_ushort v32, v[22:23], off
	global_load_ushort v33, v[24:25], off
	global_load_ushort v34, v[26:27], off
	v_add_u32_e32 v12, -2, v12
	s_add_i32 s30, s30, 8
	v_cmp_eq_u32_e32 vcc, 0, v12
	v_mov_b32_e32 v1, s30
	v_add_u32_e32 v7, 0x2000, v7
	v_add_u32_e32 v6, 0x2000, v6
	v_add_u32_e32 v5, 0x2000, v5
	v_add_u32_e32 v4, 0x2000, v4
	s_or_b64 s[28:29], vcc, s[28:29]
	s_waitcnt vmcnt(7)
	ds_write_b16 v9, v13
	s_waitcnt vmcnt(6)
	ds_write_b16 v9, v28 offset:2048
	s_waitcnt vmcnt(5)
	ds_write_b16 v9, v29 offset:4096
	s_waitcnt vmcnt(4)
	ds_write_b16 v9, v30 offset:6144
	s_waitcnt vmcnt(3)
	ds_write_b16 v9, v31 offset:8192
	s_waitcnt vmcnt(2)
	ds_write_b16 v9, v32 offset:10240
	s_waitcnt vmcnt(1)
	ds_write_b16 v9, v33 offset:12288
	s_waitcnt vmcnt(0)
	ds_write_b16 v9, v34 offset:14336
	v_add_u32_e32 v9, 0x4000, v9
	s_andn2_b64 exec, exec, s[28:29]
	s_cbranch_execnz .LBB0_52
; %bb.53:
	s_or_b64 exec, exec, s[28:29]
	v_lshlrev_b32_e32 v12, 10, v1
.LBB0_54:
	s_or_b64 exec, exec, s[26:27]
	v_and_b32_e32 v1, 4, v11
	v_cmp_eq_u32_e32 vcc, 0, v1
	s_and_saveexec_b64 s[26:27], vcc
	s_cbranch_execz .LBB0_56
; %bb.55:
	v_mad_u64_u32 v[2:3], s[28:29], s11, v7, 0
	v_mov_b32_e32 v14, v3
	v_mad_u64_u32 v[14:15], s[28:29], s3, v7, v[14:15]
	v_mov_b32_e32 v3, v14
	v_mad_u64_u32 v[14:15], s[28:29], s11, v6, 0
	v_mov_b32_e32 v16, v15
	v_mad_u64_u32 v[6:7], s[28:29], s3, v6, v[16:17]
	v_mov_b32_e32 v15, v6
	v_mad_u64_u32 v[6:7], s[28:29], s11, v5, 0
	v_mov_b32_e32 v16, v7
	v_mad_u64_u32 v[16:17], s[28:29], s3, v5, v[16:17]
	v_mov_b32_e32 v7, v16
	v_mad_u64_u32 v[16:17], s[28:29], s11, v4, 0
	v_mov_b32_e32 v18, v17
	v_mad_u64_u32 v[4:5], s[28:29], s3, v4, v[18:19]
	v_mov_b32_e32 v17, v4
	v_lshl_add_u64 v[4:5], v[16:17], 1, s[6:7]
	v_lshl_add_u64 v[6:7], v[6:7], 1, s[6:7]
	v_lshl_add_u64 v[14:15], v[14:15], 1, s[6:7]
	v_lshl_add_u64 v[2:3], v[2:3], 1, s[6:7]
	global_load_ushort v1, v[4:5], off
	global_load_ushort v9, v[6:7], off
	global_load_ushort v11, v[14:15], off
	global_load_ushort v13, v[2:3], off
	v_add_lshl_u32 v2, v0, v12, 1
	s_waitcnt vmcnt(3)
	ds_write_b16 v2, v1
	s_waitcnt vmcnt(2)
	ds_write_b16 v2, v9 offset:2048
	s_waitcnt vmcnt(1)
	ds_write_b16 v2, v11 offset:4096
	s_waitcnt vmcnt(0)
	ds_write_b16 v2, v13 offset:6144
.LBB0_56:
	s_or_b64 exec, exec, s[26:27]
	v_add_u32_e32 v1, 1, v10
	v_and_b32_e32 v2, 0x7ffffc, v1
	v_cmp_ne_u32_e32 vcc, v1, v2
	v_lshl_add_u32 v0, v2, 10, v0
	s_orn2_b64 s[26:27], vcc, exec
.LBB0_57:
	s_or_b64 exec, exec, s[24:25]
	s_andn2_b64 s[18:19], s[18:19], exec
	s_and_b64 s[24:25], s[26:27], exec
	s_or_b64 s[18:19], s[18:19], s[24:25]
.LBB0_58:
	s_or_b64 exec, exec, s[22:23]
	s_and_b64 exec, exec, s[18:19]
	s_cbranch_execz .LBB0_61
; %bb.59:
	v_lshlrev_b32_e32 v1, 1, v0
	s_mov_b64 s[18:19], 0
.LBB0_60:                               ; =>This Inner Loop Header: Depth=1
	v_mad_u64_u32 v[2:3], s[22:23], v0, s11, 0
	v_mov_b32_e32 v4, v3
	v_mad_u64_u32 v[4:5], s[22:23], v0, s3, v[4:5]
	v_mov_b32_e32 v3, v4
	v_lshl_add_u64 v[2:3], v[2:3], 1, s[6:7]
	global_load_ushort v2, v[2:3], off
	v_add_u32_e32 v0, 0x400, v0
	v_cmp_le_u32_e32 vcc, s2, v0
	s_or_b64 s[18:19], vcc, s[18:19]
	s_waitcnt vmcnt(0)
	ds_write_b16 v1, v2
	v_add_u32_e32 v1, 0x800, v1
	s_andn2_b64 exec, exec, s[18:19]
	s_cbranch_execnz .LBB0_60
.LBB0_61:
	s_or_b64 exec, exec, s[16:17]
.LBB0_62:
	v_cmp_gt_u64_e32 vcc, s[20:21], v[60:61]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_155
; %bb.63:
	s_load_dwordx4 s[16:19], s[0:1], 0x38
	s_add_i32 s0, s9, -1
	s_min_i32 s3, s0, 3
	s_min_i32 s22, s0, 2
	s_min_i32 s48, s0, 1
	s_min_i32 s49, s0, 0
	s_mul_i32 s24, s8, 48
	s_cmp_lg_u32 s2, 0
	s_cselect_b64 s[6:7], -1, 0
	s_ashr_i32 s11, s10, 31
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s27, s19, 31
	s_mov_b32 s26, s19
	s_ashr_i32 s19, s18, 31
	s_ashr_i32 s25, s24, 31
	s_add_i32 s30, s20, -3
	s_cmp_gt_i32 s9, 0
	s_cselect_b64 s[34:35], -1, 0
	s_lshl_b64 s[36:37], s[18:19], 2
	s_cmp_gt_i32 s9, 1
	s_cselect_b64 s[38:39], -1, 0
	s_cmp_gt_i32 s9, 2
	s_cselect_b64 s[40:41], -1, 0
	s_cmp_gt_i32 s9, 3
	s_mul_i32 s8, s2, s22
	s_cselect_b64 s[42:43], -1, 0
	s_lshl_b32 s33, s8, 1
	s_mul_i32 s8, s2, s48
	s_mov_b32 s23, 0
	s_mul_i32 s3, s2, s3
	s_lshl_b32 s58, s8, 1
	s_mul_i32 s8, s2, s49
	v_cndmask_b32_e64 v0, 0, 1, s[6:7]
	v_lshlrev_b32_e32 v59, 3, v8
	v_cmp_eq_u32_e64 s[0:1], 63, v8
	v_cmp_neq_f32_e64 s[28:29], s17, 0
	s_mov_b32 s31, s23
	s_lshl_b64 s[44:45], s[26:27], 2
	s_lshl_b64 s[46:47], s[10:11], 1
	v_lshlrev_b32_e32 v68, 4, v8
	s_lshl_b32 s3, s3, 1
	s_lshl_b32 s59, s8, 1
	s_mov_b64 s[48:49], 0
	v_cmp_ne_u32_e64 s[8:9], 1, v0
	v_mov_b32_e32 v63, 0
                                        ; implicit-def: $vgpr0_vgpr1_vgpr2_vgpr3
                                        ; implicit-def: $vgpr4_vgpr5_vgpr6_vgpr7
                                        ; implicit-def: $vgpr8_vgpr9_vgpr10_vgpr11
                                        ; implicit-def: $vgpr12_vgpr13_vgpr14_vgpr15
                                        ; implicit-def: $vgpr16_vgpr17_vgpr18_vgpr19
                                        ; implicit-def: $vgpr20_vgpr21_vgpr22_vgpr23
                                        ; implicit-def: $vgpr54_vgpr55
                                        ; implicit-def: $vgpr50_vgpr51
                                        ; implicit-def: $vgpr30_vgpr31
                                        ; implicit-def: $vgpr26_vgpr27
                                        ; implicit-def: $vgpr46_vgpr47
                                        ; implicit-def: $vgpr42_vgpr43
                                        ; implicit-def: $vgpr38_vgpr39
                                        ; implicit-def: $vgpr34_vgpr35
	s_branch .LBB0_66
.LBB0_64:                               ;   in Loop: Header=BB0_66 Depth=1
	s_or_b64 exec, exec, s[52:53]
	v_mov_b64_e32 v[60:61], s[30:31]
.LBB0_65:                               ;   in Loop: Header=BB0_66 Depth=1
	s_or_b64 exec, exec, s[50:51]
	v_cmp_le_u64_e32 vcc, s[20:21], v[60:61]
	s_or_b64 s[48:49], vcc, s[48:49]
	s_andn2_b64 exec, exec, s[48:49]
	s_cbranch_execz .LBB0_155
.LBB0_66:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_70 Depth 2
                                        ;     Child Loop BB0_153 Depth 2
	s_and_b64 vcc, exec, s[8:9]
	s_cbranch_vccnz .LBB0_93
; %bb.67:                               ;   in Loop: Header=BB0_66 Depth=1
	v_mul_lo_u32 v62, v61, s10
	v_mul_lo_u32 v66, v60, s11
	v_mad_u64_u32 v[64:65], s[6:7], v60, s10, 0
	v_add3_u32 v65, v65, v66, v62
	v_lshl_add_u64 v[64:65], v[64:65], 1, s[4:5]
	s_mov_b32 s22, 0
	v_mov_b32_e32 v69, 0
	s_mov_b32 s54, s59
	s_mov_b32 s55, s58
	s_mov_b32 s56, s33
	s_mov_b32 s57, s3
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
	s_branch .LBB0_70
.LBB0_68:                               ;   in Loop: Header=BB0_70 Depth=2
	s_or_b64 exec, exec, s[50:51]
.LBB0_69:                               ;   in Loop: Header=BB0_70 Depth=2
	s_or_b64 exec, exec, s[6:7]
	s_addk_i32 s22, 0x400
	s_addk_i32 s57, 0x800
	s_addk_i32 s56, 0x800
	s_addk_i32 s55, 0x800
	s_addk_i32 s54, 0x800
	s_cmp_ge_u32 s22, s2
	s_cbranch_scc1 .LBB0_94
.LBB0_70:                               ;   Parent Loop BB0_66 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_add_u32_e32 v62, s22, v59
	v_cmp_gt_u32_e32 vcc, s2, v62
	v_add_u32_e32 v66, 0x200, v62
	s_and_saveexec_b64 s[50:51], vcc
	s_cbranch_execnz .LBB0_76
; %bb.71:                               ;   in Loop: Header=BB0_70 Depth=2
	s_or_b64 exec, exec, s[50:51]
	s_and_saveexec_b64 s[50:51], vcc
	s_cbranch_execnz .LBB0_79
.LBB0_72:                               ;   in Loop: Header=BB0_70 Depth=2
	s_or_b64 exec, exec, s[50:51]
	s_and_saveexec_b64 s[50:51], vcc
	s_cbranch_execnz .LBB0_82
.LBB0_73:                               ;   in Loop: Header=BB0_70 Depth=2
	s_or_b64 exec, exec, s[50:51]
	s_and_saveexec_b64 s[50:51], vcc
	s_cbranch_execnz .LBB0_85
.LBB0_74:                               ;   in Loop: Header=BB0_70 Depth=2
	s_or_b64 exec, exec, s[50:51]
	s_and_saveexec_b64 s[50:51], vcc
	s_cbranch_execnz .LBB0_88
.LBB0_75:                               ;   in Loop: Header=BB0_70 Depth=2
	s_or_b64 exec, exec, s[50:51]
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_69
	s_branch .LBB0_91
.LBB0_76:                               ;   in Loop: Header=BB0_70 Depth=2
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[4:5], v[62:63], 1, v[64:65]
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[20:21], s[10:11], 1, v[4:5]
	global_load_dwordx4 v[4:7], v[4:5], off nt
	s_nop 0
	global_load_dwordx4 v[12:15], v[20:21], off nt
	v_lshl_add_u64 v[20:21], v[20:21], 0, s[46:47]
	global_load_dwordx4 v[20:23], v[20:21], off nt
	v_cmp_gt_u32_e64 s[6:7], s2, v66
	s_and_saveexec_b64 s[52:53], s[6:7]
	s_cbranch_execz .LBB0_78
; %bb.77:                               ;   in Loop: Header=BB0_70 Depth=2
	v_mov_b32_e32 v67, v63
	v_lshl_add_u64 v[0:1], v[66:67], 1, v[64:65]
	v_lshl_add_u64 v[16:17], s[10:11], 1, v[0:1]
	global_load_dwordx4 v[0:3], v[0:1], off nt
	s_nop 0
	global_load_dwordx4 v[8:11], v[16:17], off nt
	v_lshl_add_u64 v[16:17], v[16:17], 0, s[46:47]
	global_load_dwordx4 v[16:19], v[16:17], off nt
.LBB0_78:                               ;   in Loop: Header=BB0_70 Depth=2
	s_or_b64 exec, exec, s[52:53]
	s_or_b64 exec, exec, s[50:51]
	s_and_saveexec_b64 s[50:51], vcc
	s_cbranch_execz .LBB0_72
.LBB0_79:                               ;   in Loop: Header=BB0_70 Depth=2
	v_add_u32_e32 v82, s54, v68
	v_add_u32_e32 v81, s55, v68
	v_add_u32_e32 v67, s56, v68
	v_add_u32_e32 v62, s57, v68
	s_waitcnt lgkmcnt(3)
	ds_read_b128 v[32:35], v82
	s_waitcnt lgkmcnt(3)
	ds_read_b128 v[36:39], v81
	s_waitcnt lgkmcnt(3)
	ds_read_b128 v[40:43], v67
	s_waitcnt lgkmcnt(3)
	ds_read_b128 v[44:47], v62
	v_cmp_gt_u32_e64 s[6:7], s2, v66
	s_and_saveexec_b64 s[52:53], s[6:7]
	s_cbranch_execz .LBB0_81
; %bb.80:                               ;   in Loop: Header=BB0_70 Depth=2
	ds_read_b128 v[24:27], v82 offset:1024
	ds_read_b128 v[28:31], v81 offset:1024
	ds_read_b128 v[48:51], v67 offset:1024
	ds_read_b128 v[52:55], v62 offset:1024
.LBB0_81:                               ;   in Loop: Header=BB0_70 Depth=2
	s_or_b64 exec, exec, s[52:53]
	s_or_b64 exec, exec, s[50:51]
	s_and_saveexec_b64 s[50:51], vcc
	s_cbranch_execz .LBB0_73
.LBB0_82:                               ;   in Loop: Header=BB0_70 Depth=2
	s_waitcnt vmcnt(2) lgkmcnt(3)
	;;#ASMSTART
	v_dot2c_f32_f16 v80, v32, v4
	;;#ASMEND
	s_waitcnt vmcnt(1)
	;;#ASMSTART
	v_dot2c_f32_f16 v79, v32, v12
	;;#ASMEND
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	v_dot2c_f32_f16 v78, v32, v20
	;;#ASMEND
	v_cmp_gt_u32_e64 s[6:7], s2, v66
	;;#ASMSTART
	v_dot2c_f32_f16 v80, v33, v5
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v79, v33, v13
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v78, v33, v21
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v80, v34, v6
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v79, v34, v14
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v78, v34, v22
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v80, v35, v7
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v79, v35, v15
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v78, v35, v23
	;;#ASMEND
	s_and_saveexec_b64 s[52:53], s[6:7]
	s_cbranch_execz .LBB0_84
; %bb.83:                               ;   in Loop: Header=BB0_70 Depth=2
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
.LBB0_84:                               ;   in Loop: Header=BB0_70 Depth=2
	s_or_b64 exec, exec, s[52:53]
	s_or_b64 exec, exec, s[50:51]
	s_and_saveexec_b64 s[50:51], vcc
	s_cbranch_execz .LBB0_74
.LBB0_85:                               ;   in Loop: Header=BB0_70 Depth=2
	s_waitcnt vmcnt(2) lgkmcnt(2)
	;;#ASMSTART
	v_dot2c_f32_f16 v77, v36, v4
	;;#ASMEND
	s_waitcnt vmcnt(1)
	;;#ASMSTART
	v_dot2c_f32_f16 v76, v36, v12
	;;#ASMEND
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	v_dot2c_f32_f16 v75, v36, v20
	;;#ASMEND
	v_cmp_gt_u32_e64 s[6:7], s2, v66
	;;#ASMSTART
	v_dot2c_f32_f16 v77, v37, v5
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v76, v37, v13
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v75, v37, v21
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v77, v38, v6
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v76, v38, v14
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v75, v38, v22
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v77, v39, v7
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v76, v39, v15
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v75, v39, v23
	;;#ASMEND
	s_and_saveexec_b64 s[52:53], s[6:7]
	s_cbranch_execz .LBB0_87
; %bb.86:                               ;   in Loop: Header=BB0_70 Depth=2
	;;#ASMSTART
	v_dot2c_f32_f16 v77, v28, v0
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v76, v28, v8
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v75, v28, v16
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v77, v29, v1
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v76, v29, v9
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v75, v29, v17
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v77, v30, v2
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v76, v30, v10
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v75, v30, v18
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v77, v31, v3
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v76, v31, v11
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v75, v31, v19
	;;#ASMEND
.LBB0_87:                               ;   in Loop: Header=BB0_70 Depth=2
	s_or_b64 exec, exec, s[52:53]
	s_or_b64 exec, exec, s[50:51]
	s_and_saveexec_b64 s[50:51], vcc
	s_cbranch_execz .LBB0_75
.LBB0_88:                               ;   in Loop: Header=BB0_70 Depth=2
	s_waitcnt vmcnt(2) lgkmcnt(1)
	;;#ASMSTART
	v_dot2c_f32_f16 v74, v40, v4
	;;#ASMEND
	s_waitcnt vmcnt(1)
	;;#ASMSTART
	v_dot2c_f32_f16 v73, v40, v12
	;;#ASMEND
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	v_dot2c_f32_f16 v72, v40, v20
	;;#ASMEND
	v_cmp_gt_u32_e64 s[6:7], s2, v66
	;;#ASMSTART
	v_dot2c_f32_f16 v74, v41, v5
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v73, v41, v13
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v72, v41, v21
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v74, v42, v6
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v73, v42, v14
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v72, v42, v22
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v74, v43, v7
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v73, v43, v15
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v72, v43, v23
	;;#ASMEND
	s_and_saveexec_b64 s[52:53], s[6:7]
	s_cbranch_execz .LBB0_90
; %bb.89:                               ;   in Loop: Header=BB0_70 Depth=2
	;;#ASMSTART
	v_dot2c_f32_f16 v74, v48, v0
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v73, v48, v8
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v72, v48, v16
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v74, v49, v1
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v73, v49, v9
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v72, v49, v17
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v74, v50, v2
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v73, v50, v10
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v72, v50, v18
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v74, v51, v3
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v73, v51, v11
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v72, v51, v19
	;;#ASMEND
.LBB0_90:                               ;   in Loop: Header=BB0_70 Depth=2
	s_or_b64 exec, exec, s[52:53]
	s_or_b64 exec, exec, s[50:51]
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_69
.LBB0_91:                               ;   in Loop: Header=BB0_70 Depth=2
	s_waitcnt vmcnt(2) lgkmcnt(0)
	;;#ASMSTART
	v_dot2c_f32_f16 v71, v44, v4
	;;#ASMEND
	s_waitcnt vmcnt(1)
	;;#ASMSTART
	v_dot2c_f32_f16 v70, v44, v12
	;;#ASMEND
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	v_dot2c_f32_f16 v69, v44, v20
	;;#ASMEND
	v_cmp_gt_u32_e32 vcc, s2, v66
	;;#ASMSTART
	v_dot2c_f32_f16 v71, v45, v5
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v70, v45, v13
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v69, v45, v21
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v71, v46, v6
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v70, v46, v14
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v69, v46, v22
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v71, v47, v7
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v70, v47, v15
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v69, v47, v23
	;;#ASMEND
	s_and_saveexec_b64 s[50:51], vcc
	s_cbranch_execz .LBB0_68
; %bb.92:                               ;   in Loop: Header=BB0_70 Depth=2
	;;#ASMSTART
	v_dot2c_f32_f16 v71, v52, v0
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v70, v52, v8
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v69, v52, v16
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v71, v53, v1
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v70, v53, v9
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v69, v53, v17
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v71, v54, v2
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v70, v54, v10
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v69, v54, v18
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v71, v55, v3
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v70, v55, v11
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v69, v55, v19
	;;#ASMEND
	s_branch .LBB0_68
.LBB0_93:                               ;   in Loop: Header=BB0_66 Depth=1
	v_mov_b32_e32 v80, v63
	v_mov_b32_e32 v79, v63
	v_mov_b32_e32 v78, v63
	v_mov_b32_e32 v77, v63
	v_mov_b32_e32 v76, v63
	v_mov_b32_e32 v75, v63
	v_mov_b32_e32 v74, v63
	v_mov_b32_e32 v73, v63
	v_mov_b32_e32 v72, v63
	v_mov_b32_e32 v71, v63
	v_mov_b32_e32 v70, v63
	v_mov_b32_e32 v69, v63
.LBB0_94:                               ;   in Loop: Header=BB0_66 Depth=1
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
	s_and_saveexec_b64 s[50:51], s[0:1]
	s_cbranch_execz .LBB0_150
; %bb.95:                               ;   in Loop: Header=BB0_66 Depth=1
	v_mul_lo_u32 v62, v61, s26
	v_mul_lo_u32 v66, v60, s27
	v_mad_u64_u32 v[64:65], s[6:7], v60, s26, 0
	v_add3_u32 v65, v65, v66, v62
	v_mul_lo_u32 v62, v61, s18
	v_mul_lo_u32 v81, v60, s19
	v_mad_u64_u32 v[66:67], s[6:7], v60, s18, 0
	v_add3_u32 v67, v67, v81, v62
	v_lshl_add_u64 v[64:65], v[64:65], 1, s[14:15]
	v_lshl_add_u64 v[66:67], v[66:67], 1, s[12:13]
	s_andn2_b64 vcc, exec, s[34:35]
	v_cmp_ne_u32_e64 s[6:7], 0, v56
	s_cbranch_vccnz .LBB0_109
; %bb.96:                               ;   in Loop: Header=BB0_66 Depth=1
	s_and_saveexec_b64 s[52:53], s[6:7]
	s_cbranch_execnz .LBB0_99
; %bb.97:                               ;   in Loop: Header=BB0_66 Depth=1
	s_or_b64 exec, exec, s[52:53]
	v_cmp_ne_u32_e32 vcc, 0, v57
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execnz .LBB0_102
.LBB0_98:                               ;   in Loop: Header=BB0_66 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_cmp_ne_u32_e32 vcc, 0, v58
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execnz .LBB0_105
	s_branch .LBB0_108
.LBB0_99:                               ;   in Loop: Header=BB0_66 Depth=1
	s_andn2_b64 vcc, exec, s[28:29]
	v_mul_f32_e32 v62, s16, v80
	s_cbranch_vccnz .LBB0_101
; %bb.100:                              ;   in Loop: Header=BB0_66 Depth=1
	global_load_ushort v80, v[66:67], off
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v62, s17, v80, v62 op_sel_hi:[0,1,0]
.LBB0_101:                              ;   in Loop: Header=BB0_66 Depth=1
	v_cvt_f16_f32_e32 v62, v62
	global_store_short v[64:65], v62, off
	s_or_b64 exec, exec, s[52:53]
	v_cmp_ne_u32_e32 vcc, 0, v57
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_98
.LBB0_102:                              ;   in Loop: Header=BB0_66 Depth=1
	s_andn2_b64 vcc, exec, s[28:29]
	v_mul_f32_e32 v62, s16, v79
	s_cbranch_vccnz .LBB0_104
; %bb.103:                              ;   in Loop: Header=BB0_66 Depth=1
	v_lshl_add_u64 v[80:81], s[18:19], 1, v[66:67]
	global_load_ushort v79, v[80:81], off
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v62, s17, v79, v62 op_sel_hi:[0,1,0]
.LBB0_104:                              ;   in Loop: Header=BB0_66 Depth=1
	v_cvt_f16_f32_e32 v62, v62
	v_lshl_add_u64 v[80:81], s[26:27], 1, v[64:65]
	global_store_short v[80:81], v62, off
	s_or_b64 exec, exec, s[6:7]
	v_cmp_ne_u32_e32 vcc, 0, v58
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_108
.LBB0_105:                              ;   in Loop: Header=BB0_66 Depth=1
	s_andn2_b64 vcc, exec, s[28:29]
	v_mul_f32_e32 v62, s16, v78
	s_cbranch_vccnz .LBB0_107
; %bb.106:                              ;   in Loop: Header=BB0_66 Depth=1
	v_lshl_add_u64 v[78:79], v[66:67], 0, s[36:37]
	global_load_ushort v78, v[78:79], off
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v62, s17, v78, v62 op_sel_hi:[0,1,0]
.LBB0_107:                              ;   in Loop: Header=BB0_66 Depth=1
	v_cvt_f16_f32_e32 v62, v62
	v_lshl_add_u64 v[78:79], v[64:65], 0, s[44:45]
	global_store_short v[78:79], v62, off
.LBB0_108:                              ;   in Loop: Header=BB0_66 Depth=1
	s_or_b64 exec, exec, s[6:7]
.LBB0_109:                              ;   in Loop: Header=BB0_66 Depth=1
	s_andn2_b64 vcc, exec, s[38:39]
	s_cbranch_vccz .LBB0_112
; %bb.110:                              ;   in Loop: Header=BB0_66 Depth=1
	s_andn2_b64 vcc, exec, s[40:41]
	s_cbranch_vccz .LBB0_125
.LBB0_111:                              ;   in Loop: Header=BB0_66 Depth=1
	s_andn2_b64 vcc, exec, s[42:43]
	s_cbranch_vccz .LBB0_138
	s_branch .LBB0_150
.LBB0_112:                              ;   in Loop: Header=BB0_66 Depth=1
	v_cmp_ne_u32_e32 vcc, 0, v56
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execnz .LBB0_116
; %bb.113:                              ;   in Loop: Header=BB0_66 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_cmp_ne_u32_e32 vcc, 0, v57
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execnz .LBB0_119
.LBB0_114:                              ;   in Loop: Header=BB0_66 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_cmp_ne_u32_e32 vcc, 0, v58
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execnz .LBB0_122
.LBB0_115:                              ;   in Loop: Header=BB0_66 Depth=1
	s_or_b64 exec, exec, s[6:7]
	s_andn2_b64 vcc, exec, s[40:41]
	s_cbranch_vccnz .LBB0_111
	s_branch .LBB0_125
.LBB0_116:                              ;   in Loop: Header=BB0_66 Depth=1
	s_andn2_b64 vcc, exec, s[28:29]
	v_mul_f32_e32 v62, s16, v77
	s_cbranch_vccnz .LBB0_118
; %bb.117:                              ;   in Loop: Header=BB0_66 Depth=1
	global_load_ushort v77, v[66:67], off offset:2
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v62, s17, v77, v62 op_sel_hi:[0,1,0]
.LBB0_118:                              ;   in Loop: Header=BB0_66 Depth=1
	v_cvt_f16_f32_e32 v62, v62
	global_store_short v[64:65], v62, off offset:2
	s_or_b64 exec, exec, s[6:7]
	v_cmp_ne_u32_e32 vcc, 0, v57
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_114
.LBB0_119:                              ;   in Loop: Header=BB0_66 Depth=1
	s_andn2_b64 vcc, exec, s[28:29]
	v_mul_f32_e32 v62, s16, v76
	s_cbranch_vccnz .LBB0_121
; %bb.120:                              ;   in Loop: Header=BB0_66 Depth=1
	v_lshl_add_u64 v[76:77], s[18:19], 1, v[66:67]
	global_load_ushort v76, v[76:77], off offset:2
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v62, s17, v76, v62 op_sel_hi:[0,1,0]
.LBB0_121:                              ;   in Loop: Header=BB0_66 Depth=1
	v_cvt_f16_f32_e32 v62, v62
	v_lshl_add_u64 v[76:77], s[26:27], 1, v[64:65]
	global_store_short v[76:77], v62, off offset:2
	s_or_b64 exec, exec, s[6:7]
	v_cmp_ne_u32_e32 vcc, 0, v58
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_115
.LBB0_122:                              ;   in Loop: Header=BB0_66 Depth=1
	s_andn2_b64 vcc, exec, s[28:29]
	v_mul_f32_e32 v62, s16, v75
	s_cbranch_vccnz .LBB0_124
; %bb.123:                              ;   in Loop: Header=BB0_66 Depth=1
	v_lshl_add_u64 v[76:77], v[66:67], 0, s[36:37]
	global_load_ushort v75, v[76:77], off offset:2
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v62, s17, v75, v62 op_sel_hi:[0,1,0]
.LBB0_124:                              ;   in Loop: Header=BB0_66 Depth=1
	v_cvt_f16_f32_e32 v62, v62
	v_lshl_add_u64 v[76:77], v[64:65], 0, s[44:45]
	global_store_short v[76:77], v62, off offset:2
	s_or_b64 exec, exec, s[6:7]
	s_andn2_b64 vcc, exec, s[40:41]
	s_cbranch_vccnz .LBB0_111
.LBB0_125:                              ;   in Loop: Header=BB0_66 Depth=1
	v_cmp_ne_u32_e32 vcc, 0, v56
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execnz .LBB0_129
; %bb.126:                              ;   in Loop: Header=BB0_66 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_cmp_ne_u32_e32 vcc, 0, v57
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execnz .LBB0_132
.LBB0_127:                              ;   in Loop: Header=BB0_66 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_cmp_ne_u32_e32 vcc, 0, v58
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execnz .LBB0_135
.LBB0_128:                              ;   in Loop: Header=BB0_66 Depth=1
	s_or_b64 exec, exec, s[6:7]
	s_andn2_b64 vcc, exec, s[42:43]
	s_cbranch_vccz .LBB0_138
	s_branch .LBB0_150
.LBB0_129:                              ;   in Loop: Header=BB0_66 Depth=1
	s_andn2_b64 vcc, exec, s[28:29]
	v_mul_f32_e32 v62, s16, v74
	s_cbranch_vccnz .LBB0_131
; %bb.130:                              ;   in Loop: Header=BB0_66 Depth=1
	global_load_ushort v74, v[66:67], off offset:4
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v62, s17, v74, v62 op_sel_hi:[0,1,0]
.LBB0_131:                              ;   in Loop: Header=BB0_66 Depth=1
	v_cvt_f16_f32_e32 v62, v62
	global_store_short v[64:65], v62, off offset:4
	s_or_b64 exec, exec, s[6:7]
	v_cmp_ne_u32_e32 vcc, 0, v57
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_127
.LBB0_132:                              ;   in Loop: Header=BB0_66 Depth=1
	s_andn2_b64 vcc, exec, s[28:29]
	v_mul_f32_e32 v62, s16, v73
	s_cbranch_vccnz .LBB0_134
; %bb.133:                              ;   in Loop: Header=BB0_66 Depth=1
	v_lshl_add_u64 v[74:75], s[18:19], 1, v[66:67]
	global_load_ushort v73, v[74:75], off offset:4
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v62, s17, v73, v62 op_sel_hi:[0,1,0]
.LBB0_134:                              ;   in Loop: Header=BB0_66 Depth=1
	v_cvt_f16_f32_e32 v62, v62
	v_lshl_add_u64 v[74:75], s[26:27], 1, v[64:65]
	global_store_short v[74:75], v62, off offset:4
	s_or_b64 exec, exec, s[6:7]
	v_cmp_ne_u32_e32 vcc, 0, v58
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_128
.LBB0_135:                              ;   in Loop: Header=BB0_66 Depth=1
	s_andn2_b64 vcc, exec, s[28:29]
	v_mul_f32_e32 v62, s16, v72
	s_cbranch_vccnz .LBB0_137
; %bb.136:                              ;   in Loop: Header=BB0_66 Depth=1
	v_lshl_add_u64 v[72:73], v[66:67], 0, s[36:37]
	global_load_ushort v72, v[72:73], off offset:4
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v62, s17, v72, v62 op_sel_hi:[0,1,0]
.LBB0_137:                              ;   in Loop: Header=BB0_66 Depth=1
	v_cvt_f16_f32_e32 v62, v62
	v_lshl_add_u64 v[72:73], v[64:65], 0, s[44:45]
	global_store_short v[72:73], v62, off offset:4
	s_or_b64 exec, exec, s[6:7]
	s_andn2_b64 vcc, exec, s[42:43]
	s_cbranch_vccnz .LBB0_150
.LBB0_138:                              ;   in Loop: Header=BB0_66 Depth=1
	v_cndmask_b32_e64 v62, 0, 1, s[28:29]
	v_cmp_ne_u32_e32 vcc, 0, v56
	v_cmp_ne_u32_e64 s[6:7], 1, v62
	s_and_saveexec_b64 s[52:53], vcc
	s_cbranch_execnz .LBB0_141
; %bb.139:                              ;   in Loop: Header=BB0_66 Depth=1
	s_or_b64 exec, exec, s[52:53]
	v_cmp_ne_u32_e32 vcc, 0, v57
	s_and_saveexec_b64 s[52:53], vcc
	s_cbranch_execnz .LBB0_144
.LBB0_140:                              ;   in Loop: Header=BB0_66 Depth=1
	s_or_b64 exec, exec, s[52:53]
	v_cmp_ne_u32_e32 vcc, 0, v58
	s_and_b64 exec, exec, vcc
	s_cbranch_execnz .LBB0_147
	s_branch .LBB0_150
.LBB0_141:                              ;   in Loop: Header=BB0_66 Depth=1
	s_and_b64 vcc, exec, s[6:7]
	v_mul_f32_e32 v62, s16, v71
	s_cbranch_vccnz .LBB0_143
; %bb.142:                              ;   in Loop: Header=BB0_66 Depth=1
	global_load_ushort v71, v[66:67], off offset:6
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v62, s17, v71, v62 op_sel_hi:[0,1,0]
.LBB0_143:                              ;   in Loop: Header=BB0_66 Depth=1
	v_cvt_f16_f32_e32 v62, v62
	global_store_short v[64:65], v62, off offset:6
	s_or_b64 exec, exec, s[52:53]
	v_cmp_ne_u32_e32 vcc, 0, v57
	s_and_saveexec_b64 s[52:53], vcc
	s_cbranch_execz .LBB0_140
.LBB0_144:                              ;   in Loop: Header=BB0_66 Depth=1
	s_and_b64 vcc, exec, s[6:7]
	v_mul_f32_e32 v62, s16, v70
	s_cbranch_vccnz .LBB0_146
; %bb.145:                              ;   in Loop: Header=BB0_66 Depth=1
	v_lshl_add_u64 v[70:71], s[18:19], 1, v[66:67]
	global_load_ushort v70, v[70:71], off offset:6
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v62, s17, v70, v62 op_sel_hi:[0,1,0]
.LBB0_146:                              ;   in Loop: Header=BB0_66 Depth=1
	v_cvt_f16_f32_e32 v62, v62
	v_lshl_add_u64 v[70:71], s[26:27], 1, v[64:65]
	global_store_short v[70:71], v62, off offset:6
	s_or_b64 exec, exec, s[52:53]
	v_cmp_ne_u32_e32 vcc, 0, v58
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_150
.LBB0_147:                              ;   in Loop: Header=BB0_66 Depth=1
	s_and_b64 vcc, exec, s[6:7]
	v_mul_f32_e32 v62, s16, v69
	s_cbranch_vccnz .LBB0_149
; %bb.148:                              ;   in Loop: Header=BB0_66 Depth=1
	v_lshl_add_u64 v[66:67], v[66:67], 0, s[36:37]
	global_load_ushort v66, v[66:67], off offset:6
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v62, s17, v66, v62 op_sel_hi:[0,1,0]
.LBB0_149:                              ;   in Loop: Header=BB0_66 Depth=1
	v_cvt_f16_f32_e32 v62, v62
	v_lshl_add_u64 v[64:65], v[64:65], 0, s[44:45]
	global_store_short v[64:65], v62, off offset:6
.LBB0_150:                              ;   in Loop: Header=BB0_66 Depth=1
	s_or_b64 exec, exec, s[50:51]
	v_lshl_add_u64 v[60:61], v[60:61], 0, s[24:25]
	v_lshl_add_u64 v[64:65], v[60:61], 0, 3
	v_cmp_gt_u64_e32 vcc, s[20:21], v[60:61]
	v_cmp_le_u64_e64 s[6:7], s[20:21], v[64:65]
	s_and_b64 s[6:7], vcc, s[6:7]
	s_and_saveexec_b64 s[50:51], s[6:7]
	s_cbranch_execz .LBB0_65
; %bb.151:                              ;   in Loop: Header=BB0_66 Depth=1
	v_cmp_ne_u64_e32 vcc, s[30:31], v[60:61]
	s_and_saveexec_b64 s[52:53], vcc
	s_cbranch_execz .LBB0_64
; %bb.152:                              ;   in Loop: Header=BB0_66 Depth=1
	v_subrev_co_u32_e32 v60, vcc, s30, v60
	s_mov_b64 s[54:55], 0
	s_nop 0
	v_subbrev_co_u32_e32 v61, vcc, 0, v61, vcc
	s_mov_b64 s[56:57], 0
.LBB0_153:                              ;   Parent Loop BB0_66 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_cmp_lg_u32 s56, 2
	s_cselect_b64 vcc, -1, 0
	s_cmp_lg_u32 s56, 1
	v_cndmask_b32_e32 v58, 0, v58, vcc
	s_cselect_b64 vcc, -1, 0
	s_cmp_lg_u32 s56, 0
	v_cndmask_b32_e32 v57, 0, v57, vcc
	s_cselect_b64 vcc, -1, 0
	s_add_u32 s56, s56, 1
	s_mov_b32 s22, s56
	s_addc_u32 s57, s57, 0
	v_cmp_ge_u64_e64 s[6:7], s[22:23], v[60:61]
	s_or_b64 s[54:55], s[6:7], s[54:55]
	v_cndmask_b32_e32 v56, 0, v56, vcc
	s_andn2_b64 exec, exec, s[54:55]
	s_cbranch_execnz .LBB0_153
; %bb.154:                              ;   in Loop: Header=BB0_66 Depth=1
	s_or_b64 exec, exec, s[54:55]
	s_branch .LBB0_64
.LBB0_155:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wvSpltK_hf_m4
		.amdhsa_group_segment_fixed_size 65536
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
	.set wvSpltK_hf_m4.numbered_sgpr, 60
	.set wvSpltK_hf_m4.num_named_barrier, 0
	.set wvSpltK_hf_m4.private_seg_size, 0
	.set wvSpltK_hf_m4.uses_vcc, 1
	.set wvSpltK_hf_m4.uses_flat_scratch, 0
	.set wvSpltK_hf_m4.has_dyn_sized_stack, 0
	.set wvSpltK_hf_m4.has_recursion, 0
	.set wvSpltK_hf_m4.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 8184
; TotalNumSgprs: 66
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
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
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
	.type	__hip_cuid_c2c705e7ac62fcf,@object ; @__hip_cuid_c2c705e7ac62fcf
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_c2c705e7ac62fcf
__hip_cuid_c2c705e7ac62fcf:
	.byte	0                               ; 0x0
	.size	__hip_cuid_c2c705e7ac62fcf, 1

	.ident	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.4 26084 f58b06dce1f9c15707c5f808fd002e18c2accf7e)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_c2c705e7ac62fcf
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
            { type: int32, semantic: ComputeUnits },
            { type: int32, semantic: SizeFree0 },
            { type: int32, semantic: StrideB0 },
            { type: int32, semantic: StrideA0 },
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
  AssertSizeGreaterThan: { 0: 0, 1: 8 }
  AssertSizeLessThan: { 0: 5, 3: 8193 }
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
    .group_segment_fixed_size: 65536
    .kernarg_segment_align: 8
    .kernarg_segment_size: 72
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           wvSpltK_hf_m4
    .private_segment_fixed_size: 0
    .sgpr_count:     66
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
