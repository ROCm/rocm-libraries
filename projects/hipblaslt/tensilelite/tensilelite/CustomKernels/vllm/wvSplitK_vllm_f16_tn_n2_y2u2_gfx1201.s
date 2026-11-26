	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 4
	.text
	.protected	wvSplitK_vllm_f16_tn_n2_y2u2_gfx1201 ; -- Begin function wvSplitK_vllm_f16_tn_n2_y2u2_gfx1201
	.globl	wvSplitK_vllm_f16_tn_n2_y2u2_gfx1201
	.p2align	8
	.type	wvSplitK_vllm_f16_tn_n2_y2u2_gfx1201,@function
wvSplitK_vllm_f16_tn_n2_y2u2_gfx1201:   ; @wvSplitK_vllm_f16_tn_n2_y2u2_gfx1201
; %bb.0:
	s_clause 0x1
	s_load_b32 s2, s[0:1], 0x18
	s_load_b64 s[12:13], s[0:1], 0x0
	s_mov_b32 s6, -1
                                        ; implicit-def: $sgpr22
	s_wait_kmcnt 0x0
	s_lshl_b32 s4, s2, 5
	s_add_co_i32 s33, s13, -1
	s_delay_alu instid0(SALU_CYCLE_1)
	v_add_co_u32 v1, s3, s4, s33
	s_and_not1_b32 vcc_lo, exec_lo, s3
                                        ; implicit-def: $sgpr3
	v_readfirstlane_b32 s5, v1
	s_cbranch_vccz .LBB0_2
; %bb.1:
	s_cvt_f32_u32 s3, s4
	s_sub_co_i32 s6, 0, s4
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_iflag_f32_e32 v1, s3
	v_readfirstlane_b32 s3, v1
	s_mul_f32 s3, s3, 0x4f7ffffe
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_cvt_u32_f32 s3, s3
	s_wait_alu 0xfffe
	s_mul_i32 s6, s6, s3
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_hi_u32 s6, s3, s6
	s_add_co_i32 s3, s3, s6
	s_wait_alu 0xfffe
	s_mul_hi_u32 s3, s5, s3
	s_wait_alu 0xfffe
	s_mul_i32 s6, s3, s4
	s_delay_alu instid0(SALU_CYCLE_1)
	s_sub_co_i32 s5, s5, s6
	s_add_co_i32 s6, s3, 1
	s_wait_alu 0xfffe
	s_sub_co_i32 s7, s5, s4
	s_cmp_ge_u32 s5, s4
	s_cselect_b32 s3, s6, s3
	s_cselect_b32 s5, s7, s5
	s_wait_alu 0xfffe
	s_add_co_i32 s6, s3, 1
	s_cmp_ge_u32 s5, s4
	s_cselect_b32 s3, s6, s3
	s_lshl_b32 s22, s2, 1
	s_wait_alu 0xfffe
	s_mul_i32 s3, s3, s22
	s_wait_alu 0xfffe
	s_cvt_f32_u32 s4, s3
	s_sub_co_i32 s5, 0, s3
	s_add_co_i32 s6, s3, s33
	s_wait_alu 0xfffe
	v_rcp_iflag_f32_e32 v1, s4
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_2)
	v_readfirstlane_b32 s4, v1
	s_mul_f32 s4, s4, 0x4f7ffffe
	s_wait_alu 0xfffe
	s_cvt_u32_f32 s4, s4
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_2)
	s_mul_i32 s5, s5, s4
	s_wait_alu 0xfffe
	s_mul_hi_u32 s5, s4, s5
	s_wait_alu 0xfffe
	s_add_co_i32 s4, s4, s5
	s_wait_alu 0xfffe
	s_mul_hi_u32 s4, s6, s4
	s_wait_alu 0xfffe
	s_mul_i32 s5, s4, s3
	s_wait_alu 0xfffe
	s_sub_co_i32 s5, s6, s5
	s_add_co_i32 s6, s4, 1
	s_wait_alu 0xfffe
	s_sub_co_i32 s7, s5, s3
	s_cmp_ge_u32 s5, s3
	s_cselect_b32 s4, s6, s4
	s_cselect_b32 s5, s7, s5
	s_wait_alu 0xfffe
	s_add_co_i32 s6, s4, 1
	s_cmp_ge_u32 s5, s3
	s_cselect_b32 s3, s6, s4
	s_mov_b32 s6, 0
	s_wait_alu 0xfffe
	s_max_u32 s3, s3, 4
.LBB0_2:
	s_clause 0x1
	s_load_b64 s[14:15], s[0:1], 0x48
	s_load_b64 s[16:17], s[0:1], 0x8
	s_and_not1_b32 vcc_lo, exec_lo, s6
	s_cbranch_vccnz .LBB0_4
; %bb.3:
	s_lshl_b32 s22, s2, 1
	s_mov_b32 s3, 4
.LBB0_4:
	s_clause 0x2
	s_load_b64 s[20:21], s[0:1], 0x10
	s_load_b64 s[18:19], s[0:1], 0x20
	s_load_b256 s[4:11], s[0:1], 0x28
	v_bfe_u32 v41, v0, 10, 10
	s_mul_i32 s0, s3, ttmp9
	s_mul_i32 s34, s3, s22
	s_cmp_gt_u32 s12, 0x4000
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v33, s0, v41, 1
	s_mov_b32 s0, -1
	v_cmp_gt_u32_e32 vcc_lo, s3, v41
	s_cbranch_scc0 .LBB0_43
; %bb.5:
	s_and_saveexec_b32 s35, vcc_lo
	s_cbranch_execz .LBB0_42
; %bb.6:
	s_lshl_b32 s0, s3, 1
	s_mov_b32 s23, 0
	s_wait_alu 0xfffe
	s_cvt_f32_u32 s1, s0
	s_sub_co_i32 s2, 0, s0
	s_add_co_i32 s22, s0, s33
	s_wait_alu 0xfffe
	v_rcp_iflag_f32_e32 v1, s1
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_2)
	v_readfirstlane_b32 s1, v1
	s_mul_f32 s1, s1, 0x4f7ffffe
	s_wait_alu 0xfffe
	s_cvt_u32_f32 s1, s1
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_2)
	s_mul_i32 s2, s2, s1
	s_wait_alu 0xfffe
	s_mul_hi_u32 s2, s1, s2
	s_wait_alu 0xfffe
	s_add_co_i32 s1, s1, s2
	s_wait_alu 0xfffe
	s_mul_hi_u32 s1, s22, s1
	s_wait_alu 0xfffe
	s_mul_i32 s1, s1, s0
	s_wait_alu 0xfffe
	s_sub_co_i32 s1, s22, s1
	s_wait_alu 0xfffe
	s_sub_co_i32 s2, s1, s0
	s_cmp_ge_u32 s1, s0
	s_wait_alu 0xfffe
	s_cselect_b32 s1, s2, s1
	s_wait_alu 0xfffe
	s_sub_co_i32 s2, s1, s0
	s_cmp_ge_u32 s1, s0
	s_wait_alu 0xfffe
	s_cselect_b32 s0, s2, s1
	s_wait_alu 0xfffe
	s_sub_co_i32 s36, s22, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	v_cmp_gt_u32_e32 vcc_lo, s36, v33
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_42
; %bb.7:
	v_dual_mov_b32 v36, v33 :: v_dual_and_b32 v1, 0x3ff, v0
	v_mbcnt_lo_u32_b32 v45, -1, 0
	s_wait_kmcnt 0x0
	s_cmp_lg_u64 s[14:15], 0
	s_mov_b32 s22, s19
	v_dual_mov_b32 v35, 0 :: v_dual_lshlrev_b32 v42, 3, v1
	v_cmp_eq_u32_e64 s1, 31, v1
	v_lshlrev_b32_e32 v1, 4, v1
	s_cselect_b32 s0, -1, 0
	s_lshl_b32 s37, s3, 8
	s_add_co_i32 s38, s12, -8
	s_cmp_neq_f32 s9, 0
	v_lshl_add_u32 v44, v41, 9, v1
	v_lshl_add_u32 v43, v41, 8, v42
	v_xor_b32_e32 v47, 16, v45
	s_mov_b32 s26, s10
	s_mov_b32 s27, s23
	v_add_nc_u32_e32 v46, 0x8000, v44
	s_mov_b32 s28, s11
	s_mov_b32 s29, s23
	s_cselect_b32 s39, -1, 0
	s_lshl_b32 s40, s3, 9
	s_lshl_b64 s[24:25], s[22:23], 1
	s_lshl_b64 s[26:27], s[26:27], 1
	s_lshl_b64 s[28:29], s[28:29], 1
	s_mov_b32 s41, 0
	s_branch .LBB0_10
.LBB0_8:                                ;   in Loop: Header=BB0_10 Depth=1
	v_cvt_f32_f16_e32 v1, v48
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v5, v1
	v_cndmask_b32_e64 v1, v5, v1, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v1, v1
	global_store_b16 v[3:4], v1, off offset:2
.LBB0_9:                                ;   in Loop: Header=BB0_10 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s22
	v_add_nc_u32_e32 v36, s34, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e32 vcc_lo, s36, v36
	s_or_b32 s41, vcc_lo, s41
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s41
	s_cbranch_execz .LBB0_42
.LBB0_10:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_18 Depth 2
                                        ;       Child Loop BB0_22 Depth 3
                                        ;       Child Loop BB0_24 Depth 3
	v_cmp_gt_u32_e64 s2, s13, v36
	v_or_b32_e32 v49, 1, v36
	s_wait_alu 0xfffe
	s_and_b32 s22, s0, s2
	s_wait_alu 0xfffe
	s_xor_b32 s22, s22, -1
	s_wait_alu 0xfffe
	s_and_saveexec_b32 s30, s22
	s_wait_alu 0xfffe
	s_xor_b32 s22, exec_lo, s30
; %bb.11:                               ;   in Loop: Header=BB0_10 Depth=1
	v_or_b32_e32 v49, 1, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_min_u32_e32 v34, s33, v49
; %bb.12:                               ;   in Loop: Header=BB0_10 Depth=1
	s_wait_alu 0xfffe
	s_or_saveexec_b32 s22, s22
	s_wait_loadcnt 0x1
	v_dual_mov_b32 v51, 0 :: v_dual_mov_b32 v50, 0
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v48, 0
	s_wait_alu 0xfffe
	s_xor_b32 exec_lo, exec_lo, s22
	s_cbranch_execz .LBB0_14
; %bb.13:                               ;   in Loop: Header=BB0_10 Depth=1
	v_mov_b32_e32 v37, v35
	v_min_u32_e32 v34, s33, v49
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[1:2], 1, v[36:37]
	v_lshlrev_b64_e32 v[3:4], 1, v[34:35]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v1, vcc_lo, s14, v1
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v2, null, s15, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v3, vcc_lo, s14, v3
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v4, null, s15, v4, vcc_lo
	s_clause 0x1
	global_load_u16 v50, v[1:2], off
	global_load_u16 v48, v[3:4], off
.LBB0_14:                               ;   in Loop: Header=BB0_10 Depth=1
	s_or_b32 exec_lo, exec_lo, s22
	s_wait_dscnt 0x3
	v_mad_co_u64_u32 v[1:2], null, v34, s18, 0
	v_min_u32_e32 v4, s33, v36
	v_dual_mov_b32 v52, 0 :: v_dual_mov_b32 v53, 0
	v_mov_b32_e32 v54, 0
	s_mov_b32 s22, 0
	s_wait_alu 0xfffe
	s_mov_b32 s42, s22
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mad_co_u64_u32 v[2:3], null, v35, s18, v[2:3]
	v_mad_co_u64_u32 v[3:4], null, v4, s18, 0
	v_lshlrev_b64_e32 v[39:40], 1, v[1:2]
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b64_e32 v[37:38], 1, v[3:4]
	s_branch .LBB0_18
.LBB0_15:                               ;   in Loop: Header=BB0_18 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s43
.LBB0_16:                               ;   in Loop: Header=BB0_18 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s31
	s_wait_loadcnt_dscnt 0x301
	v_dot2_f32_f16 v34, v29, v17, v54
	s_wait_loadcnt 0x2
	v_dot2_f32_f16 v29, v29, v21, v53
	s_wait_dscnt 0x0
	v_dot2_f32_f16 v17, v25, v17, v52
	v_dot2_f32_f16 v21, v25, v21, v51
	v_dot2_f32_f16 v25, v30, v18, v34
	v_dot2_f32_f16 v29, v30, v22, v29
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot2_f32_f16 v17, v26, v18, v17
	v_dot2_f32_f16 v18, v26, v22, v21
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot2_f32_f16 v21, v31, v19, v25
	v_dot2_f32_f16 v22, v31, v23, v29
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot2_f32_f16 v17, v27, v19, v17
	v_dot2_f32_f16 v18, v27, v23, v18
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot2_f32_f16 v19, v32, v20, v21
	v_dot2_f32_f16 v21, v32, v24, v22
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot2_f32_f16 v17, v28, v20, v17
	v_dot2_f32_f16 v18, v28, v24, v18
	s_wait_loadcnt 0x1
	v_dot2_f32_f16 v19, v13, v1, v19
	s_wait_loadcnt 0x0
	v_dot2_f32_f16 v13, v13, v5, v21
	v_dot2_f32_f16 v1, v9, v1, v17
	v_dot2_f32_f16 v5, v9, v5, v18
	v_dot2_f32_f16 v9, v14, v2, v19
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot2_f32_f16 v13, v14, v6, v13
	v_dot2_f32_f16 v1, v10, v2, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot2_f32_f16 v2, v10, v6, v5
	v_dot2_f32_f16 v5, v15, v3, v9
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot2_f32_f16 v6, v15, v7, v13
	v_dot2_f32_f16 v1, v11, v3, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot2_f32_f16 v2, v11, v7, v2
	v_dot2_f32_f16 v54, v16, v4, v5
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot2_f32_f16 v53, v16, v8, v6
	v_dot2_f32_f16 v52, v12, v4, v1
	s_delay_alu instid0(VALU_DEP_4)
	v_dot2_f32_f16 v51, v12, v8, v2
.LBB0_17:                               ;   in Loop: Header=BB0_18 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s30
	s_addk_co_i32 s42, 0x200
	s_wait_alu 0xfffe
	s_cmp_ge_u32 s42, s12
	s_cbranch_scc1 .LBB0_29
.LBB0_18:                               ;   Parent Loop BB0_10 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_22 Depth 3
                                        ;       Child Loop BB0_24 Depth 3
	s_wait_alu 0xfffe
	s_cmp_eq_u32 s42, 0
	s_cselect_b32 s31, -1, 0
	s_add_co_i32 s30, s22, 0x4000
	s_wait_alu 0xfffe
	s_cmp_eq_u32 s42, s30
	s_cselect_b32 s43, -1, 0
	s_wait_alu 0xfffe
	s_or_b32 s43, s31, s43
	s_wait_alu 0xfffe
	s_and_not1_b32 vcc_lo, exec_lo, s43
	s_wait_alu 0xfffe
	s_cbranch_vccz .LBB0_20
; %bb.19:                               ;   in Loop: Header=BB0_18 Depth=2
	s_and_saveexec_b32 s30, s2
	s_cbranch_execz .LBB0_17
	s_branch .LBB0_26
.LBB0_20:                               ;   in Loop: Header=BB0_18 Depth=2
	s_and_b32 s31, s31, exec_lo
	s_cselect_b32 s22, s22, s30
	s_mov_b32 s44, exec_lo
	s_wait_alu 0xfffe
	s_sub_co_i32 s30, s12, s22
	s_wait_loadcnt 0x0
	s_wait_alu 0xfffe
	s_min_u32 s43, s30, 0x4000
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu 0xfffe
	v_cmpx_gt_u32_e64 s43, v43
	s_cbranch_execz .LBB0_25
; %bb.21:                               ;   in Loop: Header=BB0_18 Depth=2
	v_dual_mov_b32 v1, v44 :: v_dual_mov_b32 v34, v43
	s_lshl_b64 s[30:31], s[22:23], 1
	s_mov_b32 s45, 0
	s_wait_alu 0xfffe
	s_add_nc_u64 s[30:31], s[20:21], s[30:31]
.LBB0_22:                               ;   Parent Loop BB0_10 Depth=1
                                        ;     Parent Loop BB0_18 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	v_lshlrev_b64_e32 v[2:3], 1, v[34:35]
	v_add_nc_u32_e32 v34, s37, v34
	s_wait_alu 0xfffe
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v2, vcc_lo, s30, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s31, v3, vcc_lo
	v_cmp_le_u32_e32 vcc_lo, s43, v34
	global_load_b128 v[2:5], v[2:3], off
	s_or_b32 s45, vcc_lo, s45
	s_wait_loadcnt 0x0
	ds_store_b128 v1, v[2:5]
	v_add_nc_u32_e32 v1, s40, v1
	s_and_not1_b32 exec_lo, exec_lo, s45
	s_cbranch_execnz .LBB0_22
; %bb.23:                               ;   in Loop: Header=BB0_18 Depth=2
	s_or_b32 exec_lo, exec_lo, s45
	v_dual_mov_b32 v1, v46 :: v_dual_mov_b32 v34, v43
	s_add_nc_u64 s[30:31], s[30:31], s[24:25]
	s_mov_b32 s45, 0
.LBB0_24:                               ;   Parent Loop BB0_10 Depth=1
                                        ;     Parent Loop BB0_18 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[2:3], 1, v[34:35]
	v_add_nc_u32_e32 v34, s37, v34
	s_wait_alu 0xfffe
	v_add_co_u32 v2, vcc_lo, s30, v2
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_ci_u32_e64 v3, null, s31, v3, vcc_lo
	v_cmp_le_u32_e32 vcc_lo, s43, v34
	global_load_b128 v[2:5], v[2:3], off
	s_or_b32 s45, vcc_lo, s45
	s_wait_loadcnt 0x0
	ds_store_b128 v1, v[2:5]
	v_add_nc_u32_e32 v1, s40, v1
	s_and_not1_b32 exec_lo, exec_lo, s45
	s_cbranch_execnz .LBB0_24
.LBB0_25:                               ;   in Loop: Header=BB0_18 Depth=2
	s_or_b32 exec_lo, exec_lo, s44
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s30, s2
	s_cbranch_execz .LBB0_17
.LBB0_26:                               ;   in Loop: Header=BB0_18 Depth=2
	v_dual_mov_b32 v12, 0 :: v_dual_add_nc_u32 v55, s42, v42
	v_dual_mov_b32 v10, 0 :: v_dual_mov_b32 v9, 0
	v_dual_mov_b32 v28, 0 :: v_dual_mov_b32 v27, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_min_u32_e32 v34, s38, v55
	s_wait_dscnt 0x2
	v_dual_mov_b32 v11, 0 :: v_dual_add_nc_u32 v56, 0x100, v55
	v_dual_mov_b32 v26, 0 :: v_dual_mov_b32 v25, 0
	v_lshlrev_b64_e32 v[1:2], 1, v[34:35]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_min_u32_e32 v34, s38, v56
	v_dual_mov_b32 v16, 0 :: v_dual_mov_b32 v15, 0
	v_dual_mov_b32 v14, 0 :: v_dual_mov_b32 v13, 0
	v_add_co_u32 v5, vcc_lo, s16, v1
	s_wait_dscnt 0x0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v6, null, s17, v2, vcc_lo
	v_lshlrev_b64_e32 v[1:2], 1, v[34:35]
	v_add_co_u32 v3, vcc_lo, v5, v37
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_ci_u32_e64 v4, null, v6, v38, vcc_lo
	v_dual_mov_b32 v32, 0 :: v_dual_mov_b32 v31, 0
	v_add_co_u32 v7, vcc_lo, s16, v1
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v8, null, s17, v2, vcc_lo
	v_add_co_u32 v1, vcc_lo, v5, v39
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v2, null, v6, v40, vcc_lo
	v_add_co_u32 v5, vcc_lo, v7, v37
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v6, null, v8, v38, vcc_lo
	v_add_co_u32 v7, vcc_lo, v7, v39
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v8, null, v8, v40, vcc_lo
	s_clause 0x3
	global_load_b128 v[17:20], v[3:4], off
	global_load_b128 v[21:24], v[1:2], off
	global_load_b128 v[1:4], v[5:6], off
	global_load_b128 v[5:8], v[7:8], off
	v_dual_mov_b32 v30, 0 :: v_dual_mov_b32 v29, 0
	s_mov_b32 s31, exec_lo
	v_cmpx_gt_u32_e64 s12, v55
	s_cbranch_execz .LBB0_16
; %bb.27:                               ;   in Loop: Header=BB0_18 Depth=2
	v_subrev_nc_u32_e32 v9, s22, v55
	v_dual_mov_b32 v14, 0 :: v_dual_mov_b32 v15, 0
	v_mov_b32_e32 v16, 0
	v_dual_mov_b32 v10, 0 :: v_dual_mov_b32 v11, 0
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_mov_b32 v13, 0 :: v_dual_lshlrev_b32 v34, 1, v9
	v_dual_mov_b32 v9, 0 :: v_dual_mov_b32 v12, 0
	ds_load_b128 v[29:32], v34
	ds_load_b128 v[25:28], v34 offset:32768
	s_mov_b32 s43, exec_lo
	v_cmpx_gt_u32_e64 s12, v56
	s_cbranch_execz .LBB0_15
; %bb.28:                               ;   in Loop: Header=BB0_18 Depth=2
	ds_load_b128 v[13:16], v34 offset:512
	ds_load_b128 v[9:12], v34 offset:33280
	s_branch .LBB0_15
.LBB0_29:                               ;   in Loop: Header=BB0_10 Depth=1
	s_and_saveexec_b32 s22, s2
	s_cbranch_execz .LBB0_9
; %bb.30:                               ;   in Loop: Header=BB0_10 Depth=1
	v_add_f32_dpp v1, v54, v54 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v53, v53 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v52, v52 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v51, v51 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cmp_gt_i32_e32 vcc_lo, 32, v47
	v_add_f32_dpp v1, v1, v1 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v3, v3 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v5, v45, v47, vcc_lo
	v_add_f32_dpp v1, v1, v1 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v3, v3 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_dscnt 0x0
	v_lshlrev_b32_e32 v6, 2, v5
	v_add_f32_dpp v1, v1, v1 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v10, v2, v2 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v3, v3 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v4, v4 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_loadcnt 0x1
	;;#ASMSTART
	;;#ASMEND
	ds_bpermute_b32 v2, v6, v1
	ds_bpermute_b32 v11, v6, v10
	ds_bpermute_b32 v8, v6, v7
	ds_bpermute_b32 v6, v6, v5
	s_wait_loadcnt 0x0
	;;#ASMSTART
	;;#ASMEND
	s_and_b32 exec_lo, exec_lo, s1
	s_cbranch_execz .LBB0_9
; %bb.31:                               ;   in Loop: Header=BB0_10 Depth=1
	s_wait_dscnt 0x3
	v_add_f32_e32 v9, v1, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v37, v35 :: v_dual_mul_f32 v12, s8, v9
	v_lshlrev_b64_e32 v[3:4], 1, v[36:37]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, s4, v3
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v2, null, s5, v4, vcc_lo
	s_and_not1_b32 vcc_lo, exec_lo, s39
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_33
; %bb.32:                               ;   in Loop: Header=BB0_10 Depth=1
	global_load_u16 v9, v[1:2], off
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v12, s9, v9, v12 op_sel_hi:[0,1,0]
.LBB0_33:                               ;   in Loop: Header=BB0_10 Depth=1
	v_cvt_f32_f16_e32 v9, v50
	v_add_co_u32 v3, vcc_lo, s6, v3
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v4, null, s7, v4, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v13, v12, v9
	v_cmp_gt_u32_e64 s2, s13, v49
	v_cndmask_b32_e64 v12, v12, v13, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v12, v12
	global_store_b16 v[3:4], v12, off
	s_and_saveexec_b32 s30, s2
	s_cbranch_execz .LBB0_37
; %bb.34:                               ;   in Loop: Header=BB0_10 Depth=1
	s_wait_dscnt 0x2
	v_add_f32_e32 v10, v10, v11
	s_and_not1_b32 vcc_lo, exec_lo, s39
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v10, s8, v10
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_36
; %bb.35:                               ;   in Loop: Header=BB0_10 Depth=1
	global_load_u16 v11, v[1:2], off offset:2
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v10, s9, v11, v10 op_sel_hi:[0,1,0]
.LBB0_36:                               ;   in Loop: Header=BB0_10 Depth=1
	v_cvt_f32_f16_e32 v11, v48
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v11, v10, v11
	v_cndmask_b32_e64 v10, v10, v11, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v10, v10
	global_store_b16 v[3:4], v10, off offset:2
.LBB0_37:                               ;   in Loop: Header=BB0_10 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s30
	s_wait_dscnt 0x1
	v_add_f32_e32 v7, v7, v8
	v_add_co_u32 v1, vcc_lo, v1, s26
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v2, null, s27, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_f32_e32 v7, s8, v7
	s_and_not1_b32 vcc_lo, exec_lo, s39
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_39
; %bb.38:                               ;   in Loop: Header=BB0_10 Depth=1
	global_load_u16 v8, v[1:2], off
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v7, s9, v8, v7 op_sel_hi:[0,1,0]
.LBB0_39:                               ;   in Loop: Header=BB0_10 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_add_f32_e32 v8, v7, v9
	v_add_co_u32 v3, vcc_lo, v3, s28
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v4, null, s29, v4, vcc_lo
	v_cndmask_b32_e64 v7, v7, v8, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v7, v7
	global_store_b16 v[3:4], v7, off
	s_and_b32 exec_lo, exec_lo, s2
	s_cbranch_execz .LBB0_9
; %bb.40:                               ;   in Loop: Header=BB0_10 Depth=1
	s_wait_dscnt 0x0
	v_add_f32_e32 v5, v5, v6
	s_and_not1_b32 vcc_lo, exec_lo, s39
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v5, s8, v5
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_8
; %bb.41:                               ;   in Loop: Header=BB0_10 Depth=1
	global_load_u16 v1, v[1:2], off offset:2
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v5, s9, v1, v5 op_sel_hi:[0,1,0]
	s_branch .LBB0_8
.LBB0_42:
	s_or_b32 exec_lo, exec_lo, s35
	s_mov_b32 s0, 0
.LBB0_43:
	s_wait_alu 0xfffe
	s_and_not1_b32 vcc_lo, exec_lo, s0
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_76
; %bb.44:
	s_wait_dscnt 0x3
	v_and_b32_e32 v2, 0x3ff, v0
	s_mov_b32 s2, exec_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v38, 3, v2
	v_lshl_add_u32 v3, v41, 8, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_u32_e64 s12, v3
	s_cbranch_execz .LBB0_49
; %bb.45:
	v_lshlrev_b32_e32 v0, 9, v41
	s_wait_dscnt 0x1
	v_dual_mov_b32 v8, v3 :: v_dual_lshlrev_b32 v1, 4, v2
	s_mov_b32 s1, 0
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v6, v0, v1
	v_mov_b32_e32 v7, v6
	v_add_co_u32 v4, s0, v0, v1
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v5, null, 0, 0, s0
	s_wait_kmcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, s20, v4
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s21, v5, vcc_lo
.LBB0_46:                               ; =>This Inner Loop Header: Depth=1
	global_load_b128 v[9:12], v[0:1], off
	v_add_nc_u32_e32 v8, 0x1000, v8
	v_add_co_u32 v0, vcc_lo, 0x2000, v0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, 0, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_le_u32_e64 s0, s12, v8
	s_wait_alu 0xfffe
	s_or_b32 s1, s0, s1
	s_wait_loadcnt 0x0
	ds_store_b128 v7, v[9:12]
	v_add_nc_u32_e32 v7, 0x2000, v7
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s1
	s_cbranch_execnz .LBB0_46
; %bb.47:
	s_or_b32 exec_lo, exec_lo, s1
	s_mov_b32 s1, 0
	s_mov_b32 s0, s19
	v_lshl_add_u32 v6, s12, 1, v6
	s_wait_alu 0xfffe
	s_lshl_b64 s[22:23], s[0:1], 1
	s_wait_alu 0xfffe
	s_add_nc_u64 s[20:21], s[20:21], s[22:23]
	s_wait_alu 0xfffe
	v_add_co_u32 v0, vcc_lo, s20, v4
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s21, v5, vcc_lo
.LBB0_48:                               ; =>This Inner Loop Header: Depth=1
	global_load_b128 v[7:10], v[0:1], off
	v_add_nc_u32_e32 v3, 0x1000, v3
	v_add_co_u32 v0, vcc_lo, 0x2000, v0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, 0, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_le_u32_e64 s0, s12, v3
	s_or_b32 s1, s0, s1
	s_wait_loadcnt 0x0
	ds_store_b128 v6, v[7:10]
	v_add_nc_u32_e32 v6, 0x2000, v6
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s1
	s_cbranch_execnz .LBB0_48
.LBB0_49:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s2
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_mov_b32 s0, exec_lo
	v_cmpx_gt_u32_e64 s3, v41
	s_cbranch_execz .LBB0_76
; %bb.50:
	v_cmp_gt_u32_e32 vcc_lo, s13, v33
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_76
; %bb.51:
	s_wait_kmcnt 0x0
	s_cmp_eq_u64 s[14:15], 0
	v_mbcnt_lo_u32_b32 v39, -1, 0
	s_cselect_b32 s0, -1, 0
	s_cmp_lg_u64 s[14:15], 0
	s_mov_b32 s3, 0
	s_cselect_b32 s19, -1, 0
	s_cmp_lg_u32 s12, 0
	v_cmp_eq_u32_e64 s1, 31, v2
	s_cselect_b32 s22, -1, 0
	s_add_co_i32 s23, s12, -8
	s_cmp_neq_f32 s9, 0
	v_lshlrev_b32_e32 v40, 4, v2
	v_mov_b32_e32 v32, 0
	v_xor_b32_e32 v41, 16, v39
	s_mov_b32 s2, s10
	s_mov_b32 s20, s11
	s_wait_alu 0xfffe
	s_mov_b32 s21, s3
	s_cselect_b32 s24, -1, 0
	s_lshl_b32 s25, s12, 1
	s_lshl_b64 s[10:11], s[2:3], 1
	s_wait_alu 0xfffe
	s_lshl_b64 s[20:21], s[20:21], 1
	s_branch .LBB0_54
.LBB0_52:                               ;   in Loop: Header=BB0_54 Depth=1
	v_cvt_f32_f16_e32 v0, v42
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v0, v4, v0
	v_cndmask_b32_e64 v0, v0, v4, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v0, v0
	global_store_b16 v[2:3], v0, off offset:2
.LBB0_53:                               ;   in Loop: Header=BB0_54 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s26
	v_add_nc_u32_e32 v33, s34, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e32 vcc_lo, s13, v33
	s_or_b32 s3, vcc_lo, s3
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s3
	s_cbranch_execz .LBB0_76
.LBB0_54:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_60 Depth 2
	v_or_b32_e32 v43, 1, v33
	v_mov_b32_e32 v44, v32
	v_mov_b32_e32 v42, v32
	s_and_not1_b32 vcc_lo, exec_lo, s19
	s_delay_alu instid0(VALU_DEP_3)
	v_min_u32_e32 v31, s33, v43
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_56
; %bb.55:                               ;   in Loop: Header=BB0_54 Depth=1
	v_mov_b32_e32 v34, v32
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[2:3], 1, v[31:32]
	s_wait_dscnt 0x3
	v_lshlrev_b64_e32 v[0:1], 1, v[33:34]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, s14, v0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s15, v1, vcc_lo
	v_add_co_u32 v2, vcc_lo, s14, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s15, v3, vcc_lo
	s_clause 0x1
	global_load_u16 v44, v[0:1], off
	global_load_u16 v42, v[2:3], off
.LBB0_56:                               ;   in Loop: Header=BB0_54 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s22
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_63
; %bb.57:                               ;   in Loop: Header=BB0_54 Depth=1
	v_min_u32_e32 v2, s33, v33
	s_wait_dscnt 0x3
	v_mad_co_u64_u32 v[0:1], null, v31, s18, 0
	v_dual_mov_b32 v45, 0 :: v_dual_mov_b32 v46, v40
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_mad_co_u64_u32 v[2:3], null, v2, s18, 0
	v_dual_mov_b32 v47, 0 :: v_dual_mov_b32 v48, 0
	v_lshlrev_b64_e32 v[34:35], 1, v[0:1]
	v_mov_b32_e32 v49, 0
	s_mov_b32 s2, 0
	s_delay_alu instid0(VALU_DEP_4)
	v_lshlrev_b64_e32 v[36:37], 1, v[2:3]
	s_branch .LBB0_60
.LBB0_58:                               ;   in Loop: Header=BB0_60 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s27
.LBB0_59:                               ;   in Loop: Header=BB0_60 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s26
	s_wait_loadcnt_dscnt 0x301
	v_dot2_f32_f16 v49, v28, v16, v49
	s_wait_loadcnt 0x2
	v_dot2_f32_f16 v28, v28, v20, v48
	s_wait_dscnt 0x0
	v_dot2_f32_f16 v16, v24, v16, v47
	v_dot2_f32_f16 v20, v24, v20, v45
	v_add_nc_u32_e32 v46, 0x400, v46
	v_dot2_f32_f16 v24, v29, v17, v49
	v_dot2_f32_f16 v28, v29, v21, v28
	v_dot2_f32_f16 v16, v25, v17, v16
	v_dot2_f32_f16 v17, v25, v21, v20
	s_addk_co_i32 s2, 0x200
	v_dot2_f32_f16 v20, v30, v18, v24
	v_dot2_f32_f16 v21, v30, v22, v28
	v_dot2_f32_f16 v16, v26, v18, v16
	v_dot2_f32_f16 v17, v26, v22, v17
	s_wait_alu 0xfffe
	s_cmp_ge_u32 s2, s12
	v_dot2_f32_f16 v18, v31, v19, v20
	v_dot2_f32_f16 v20, v31, v23, v21
	v_dot2_f32_f16 v16, v27, v19, v16
	v_dot2_f32_f16 v17, v27, v23, v17
	s_wait_loadcnt 0x1
	v_dot2_f32_f16 v18, v8, v0, v18
	s_wait_loadcnt 0x0
	v_dot2_f32_f16 v8, v8, v4, v20
	v_dot2_f32_f16 v0, v12, v0, v16
	v_dot2_f32_f16 v4, v12, v4, v17
	v_dot2_f32_f16 v12, v9, v1, v18
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot2_f32_f16 v8, v9, v5, v8
	v_dot2_f32_f16 v0, v13, v1, v0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot2_f32_f16 v1, v13, v5, v4
	v_dot2_f32_f16 v4, v10, v2, v12
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot2_f32_f16 v5, v10, v6, v8
	v_dot2_f32_f16 v0, v14, v2, v0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot2_f32_f16 v1, v14, v6, v1
	v_dot2_f32_f16 v49, v11, v3, v4
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot2_f32_f16 v48, v11, v7, v5
	v_dot2_f32_f16 v47, v15, v3, v0
	s_delay_alu instid0(VALU_DEP_4)
	v_dot2_f32_f16 v45, v15, v7, v1
	s_cbranch_scc1 .LBB0_64
.LBB0_60:                               ;   Parent Loop BB0_54 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu 0xfffe
	v_dual_mov_b32 v15, 0 :: v_dual_add_nc_u32 v8, s2, v38
	v_dual_mov_b32 v14, 0 :: v_dual_mov_b32 v27, 0
	v_dual_mov_b32 v12, 0 :: v_dual_mov_b32 v25, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_min_u32_e32 v31, s23, v8
	v_dual_mov_b32 v13, 0 :: v_dual_add_nc_u32 v50, 0x100, v8
	v_dual_mov_b32 v26, 0 :: v_dual_mov_b32 v11, 0
	v_lshlrev_b64_e32 v[0:1], 1, v[31:32]
	s_delay_alu instid0(VALU_DEP_3)
	v_min_u32_e32 v31, s23, v50
	v_dual_mov_b32 v24, 0 :: v_dual_mov_b32 v9, 0
	s_wait_dscnt 0x2
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v30, 0
	v_add_co_u32 v4, vcc_lo, s16, v0
	s_wait_dscnt 0x0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v5, null, s17, v1, vcc_lo
	v_lshlrev_b64_e32 v[0:1], 1, v[31:32]
	v_add_co_u32 v2, vcc_lo, v4, v36
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_ci_u32_e64 v3, null, v5, v37, vcc_lo
	v_mov_b32_e32 v31, 0
	v_add_co_u32 v6, vcc_lo, s16, v0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v7, null, s17, v1, vcc_lo
	v_add_co_u32 v0, vcc_lo, v4, v34
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, v5, v35, vcc_lo
	v_add_co_u32 v4, vcc_lo, v6, v36
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v5, null, v7, v37, vcc_lo
	v_add_co_u32 v6, vcc_lo, v6, v34
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v7, null, v7, v35, vcc_lo
	s_clause 0x3
	global_load_b128 v[16:19], v[2:3], off
	global_load_b128 v[20:23], v[0:1], off
	global_load_b128 v[0:3], v[4:5], off
	global_load_b128 v[4:7], v[6:7], off
	v_cmp_gt_u32_e32 vcc_lo, s12, v8
	v_dual_mov_b32 v8, 0 :: v_dual_mov_b32 v29, 0
	v_mov_b32_e32 v28, 0
	s_and_saveexec_b32 s26, vcc_lo
	s_cbranch_execz .LBB0_59
; %bb.61:                               ;   in Loop: Header=BB0_60 Depth=2
	v_dual_mov_b32 v8, 0 :: v_dual_add_nc_u32 v51, s25, v46
	v_dual_mov_b32 v9, 0 :: v_dual_mov_b32 v10, 0
	ds_load_b128 v[28:31], v46
	ds_load_b128 v[24:27], v51
	v_dual_mov_b32 v11, 0 :: v_dual_mov_b32 v12, 0
	v_dual_mov_b32 v13, 0 :: v_dual_mov_b32 v14, 0
	v_mov_b32_e32 v15, 0
	s_mov_b32 s27, exec_lo
	v_cmpx_gt_u32_e64 s12, v50
	s_cbranch_execz .LBB0_58
; %bb.62:                               ;   in Loop: Header=BB0_60 Depth=2
	ds_load_b128 v[8:11], v46 offset:512
	ds_load_b128 v[12:15], v51 offset:512
	s_branch .LBB0_58
.LBB0_63:                               ;   in Loop: Header=BB0_54 Depth=1
	v_dual_mov_b32 v49, 0 :: v_dual_mov_b32 v48, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v45, 0
.LBB0_64:                               ;   in Loop: Header=BB0_54 Depth=1
	s_delay_alu instid0(VALU_DEP_3)
	v_add_f32_dpp v0, v49, v49 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_dscnt 0x3
	v_add_f32_dpp v1, v48, v48 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v47, v47 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v45, v45 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cmp_gt_i32_e32 vcc_lo, 32, v41
	v_add_f32_dpp v0, v0, v0 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v1, v1, v1 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v3, v3 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v4, v39, v41, vcc_lo
	v_add_f32_dpp v0, v0, v0 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v1, v1, v1 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v3, v3 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_dscnt 0x0
	v_lshlrev_b32_e32 v5, 2, v4
	v_add_f32_dpp v0, v0, v0 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v9, v1, v1 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v2, v2 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v3, v3 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_loadcnt 0x1
	;;#ASMSTART
	;;#ASMEND
	ds_bpermute_b32 v1, v5, v0
	ds_bpermute_b32 v10, v5, v9
	ds_bpermute_b32 v7, v5, v6
	ds_bpermute_b32 v5, v5, v4
	s_wait_loadcnt 0x0
	;;#ASMSTART
	;;#ASMEND
	s_and_saveexec_b32 s26, s1
	s_cbranch_execz .LBB0_53
; %bb.65:                               ;   in Loop: Header=BB0_54 Depth=1
	s_wait_dscnt 0x3
	v_add_f32_e32 v8, v0, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v34, v32 :: v_dual_mul_f32 v11, s8, v8
	v_lshlrev_b64_e32 v[2:3], 1, v[33:34]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, s4, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s5, v3, vcc_lo
	s_and_not1_b32 vcc_lo, exec_lo, s24
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_67
; %bb.66:                               ;   in Loop: Header=BB0_54 Depth=1
	global_load_u16 v8, v[0:1], off
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v11, s9, v8, v11 op_sel_hi:[0,1,0]
.LBB0_67:                               ;   in Loop: Header=BB0_54 Depth=1
	v_cvt_f32_f16_e32 v8, v44
	v_add_co_u32 v2, vcc_lo, s6, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s7, v3, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v12, v11, v8
	v_cmp_gt_u32_e64 s2, s13, v43
	v_cndmask_b32_e64 v11, v12, v11, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v11, v11
	global_store_b16 v[2:3], v11, off
	s_and_saveexec_b32 s27, s2
	s_cbranch_execz .LBB0_71
; %bb.68:                               ;   in Loop: Header=BB0_54 Depth=1
	s_wait_dscnt 0x2
	v_add_f32_e32 v9, v9, v10
	s_and_not1_b32 vcc_lo, exec_lo, s24
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v9, s8, v9
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_70
; %bb.69:                               ;   in Loop: Header=BB0_54 Depth=1
	global_load_u16 v10, v[0:1], off offset:2
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v9, s9, v10, v9 op_sel_hi:[0,1,0]
.LBB0_70:                               ;   in Loop: Header=BB0_54 Depth=1
	v_cvt_f32_f16_e32 v10, v42
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v10, v9, v10
	v_cndmask_b32_e64 v9, v10, v9, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9, v9
	global_store_b16 v[2:3], v9, off offset:2
.LBB0_71:                               ;   in Loop: Header=BB0_54 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s27
	s_wait_dscnt 0x1
	v_add_f32_e32 v6, v6, v7
	v_add_co_u32 v0, vcc_lo, v0, s10
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s11, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_f32_e32 v6, s8, v6
	s_and_not1_b32 vcc_lo, exec_lo, s24
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_73
; %bb.72:                               ;   in Loop: Header=BB0_54 Depth=1
	global_load_u16 v7, v[0:1], off
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v6, s9, v7, v6 op_sel_hi:[0,1,0]
.LBB0_73:                               ;   in Loop: Header=BB0_54 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_add_f32_e32 v7, v6, v8
	v_add_co_u32 v2, vcc_lo, v2, s20
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s21, v3, vcc_lo
	v_cndmask_b32_e64 v6, v7, v6, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v6, v6
	global_store_b16 v[2:3], v6, off
	s_and_b32 exec_lo, exec_lo, s2
	s_cbranch_execz .LBB0_53
; %bb.74:                               ;   in Loop: Header=BB0_54 Depth=1
	s_wait_dscnt 0x0
	v_add_f32_e32 v4, v4, v5
	s_and_not1_b32 vcc_lo, exec_lo, s24
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v4, s8, v4
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_52
; %bb.75:                               ;   in Loop: Header=BB0_54 Depth=1
	global_load_u16 v0, v[0:1], off offset:2
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v4, s9, v0, v4 op_sel_hi:[0,1,0]
	s_branch .LBB0_52
.LBB0_76:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wvSplitK_vllm_f16_tn_n2_y2u2_gfx1201
		.amdhsa_group_segment_fixed_size 65536
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 80
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 1
		.amdhsa_next_free_vgpr 169
		.amdhsa_next_free_sgpr 46
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size 41
		.amdhsa_round_robin_scheduling 0
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
	.size	wvSplitK_vllm_f16_tn_n2_y2u2_gfx1201, .Lfunc_end0-wvSplitK_vllm_f16_tn_n2_y2u2_gfx1201
                                        ; -- End function
	.set wvSplitK_vllm_f16_tn_n2_y2u2_gfx1201.num_vgpr, 57
	.set wvSplitK_vllm_f16_tn_n2_y2u2_gfx1201.num_agpr, 0
	.set wvSplitK_vllm_f16_tn_n2_y2u2_gfx1201.numbered_sgpr, 46
	.set wvSplitK_vllm_f16_tn_n2_y2u2_gfx1201.num_named_barrier, 0
	.set wvSplitK_vllm_f16_tn_n2_y2u2_gfx1201.private_seg_size, 0
	.set wvSplitK_vllm_f16_tn_n2_y2u2_gfx1201.uses_vcc, 1
	.set wvSplitK_vllm_f16_tn_n2_y2u2_gfx1201.uses_flat_scratch, 0
	.set wvSplitK_vllm_f16_tn_n2_y2u2_gfx1201.has_dyn_sized_stack, 0
	.set wvSplitK_vllm_f16_tn_n2_y2u2_gfx1201.has_recursion, 0
	.set wvSplitK_vllm_f16_tn_n2_y2u2_gfx1201.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 5188
; TotalNumSgprs: 48
; NumVgprs: 57
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 65536 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 21
; NumSGPRsForWavesPerEU: 48
; NumVGPRsForWavesPerEU: 169
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 1
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_21eb1536a252a9a4,@object ; @__hip_cuid_21eb1536a252a9a4
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_21eb1536a252a9a4
__hip_cuid_21eb1536a252a9a4:
	.byte	0                               ; 0x0
	.size	__hip_cuid_21eb1536a252a9a4, 1

	.ident	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.0 26014 7b800a19466229b8479a78de19143dc33c3ab9b5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_21eb1536a252a9a4
	.amdgpu_metadata
---
custom.config:
  Source:
    Origin: vllm
    Repository: "https://github.com/vllm-project/vllm"
  Version: 392ed873c7fed4c41fafaad022f83f7d0fe9f57a
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
    macrotile: [32, 16, 8]
    threads: [32, 16, 1]
    grid: [ComputeUnits, One, One]
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 16, 1]
  EnableMatrixInstruction: True
  MIWaveTile: [1, 1]
  AssertSummationElementMultiple: 8
  AssertSizeEqual: { 1: 2, 2: 1 }
  AssertStrideAEqual: { 0: 1 }
  AssertStrideBEqual: { 0: 1 }
  AssertStrideCEqual: { 0: 1 }
  AssertStrideDEqual: { 0: 1 }
  StaggerU: 0
  WavefrontSize: 32
amdhsa.kernels:
  - .args:
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
      - .actual_access:  read_only
        .address_space:  global
        .offset:         72
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 65536
    .kernarg_segment_align: 8
    .kernarg_segment_size: 80
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 512
    .name:           wvSplitK_vllm_f16_tn_n2_y2u2_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count:     48
    .sgpr_spill_count: 0
    .symbol:         wvSplitK_vllm_f16_tn_n2_y2u2_gfx1201.kd
    .vgpr_count:     57
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 1
...

	.end_amdgpu_metadata
