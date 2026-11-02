	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 4
	.text
	.protected	wvSplitK_vllm_f16_tn_bias_n5_y1u4 ; -- Begin function wvSplitK_vllm_f16_tn_bias_n5_y1u4
	.globl	wvSplitK_vllm_f16_tn_bias_n5_y1u4
	.p2align	8
	.type	wvSplitK_vllm_f16_tn_bias_n5_y1u4,@function
wvSplitK_vllm_f16_tn_bias_n5_y1u4:      ; @wvSplitK_vllm_f16_tn_bias_n5_y1u4
; %bb.94:
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.95:
.LBB0_0:
	s_load_dwordx4 s[24:27], s[0:1], 0x38
	s_load_dwordx2 s[28:29], s[0:1], 0x48
	s_mov_b32 s30, s11
	s_mov_b64 s[0:1], -1
	s_cmpk_gt_i32 s2, 0x4000
	v_bfe_u32 v104, v0, 10, 10
	s_cbranch_scc0 .LBB0_52
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
	v_cmp_gt_u32_e32 vcc, s20, v104
	s_and_saveexec_b64 s[34:35], vcc
	s_cbranch_execz .LBB0_51
; %bb.2:
	v_cvt_f32_u32_e32 v1, s20
	s_mul_i32 s0, s20, s16
	s_sub_i32 s11, 0, s20
	v_add_u32_e32 v98, s0, v104
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
	v_cmp_gt_u32_e32 vcc, s9, v98
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_51
; %bb.3:
	s_and_b32 s0, s30, 7
	s_cmp_lg_u32 s0, 0
	s_mov_b32 s37, 0
	s_cselect_b64 s[0:1], -1, 0
	s_and_b32 s36, s6, 15
	s_cmp_lg_u64 s[36:37], 0
	s_cselect_b64 s[18:19], -1, 0
	s_or_b64 s[40:41], s[18:19], s[0:1]
	s_waitcnt lgkmcnt(0)
	s_cmp_eq_u64 s[28:29], 0
	v_and_b32_e32 v2, 0x3ff, v0
	s_cselect_b64 s[0:1], -1, 0
	v_lshlrev_b32_e32 v1, 3, v2
	s_ashr_i32 s31, s30, 31
	v_cmp_eq_u32_e64 s[18:19], 63, v2
	s_ashr_i32 s45, s26, 31
	s_mov_b32 s44, s26
	s_ashr_i32 s47, s27, 31
	s_mov_b32 s46, s27
	v_lshlrev_b32_e32 v2, 4, v2
	s_mov_b64 s[38:39], 0
	v_lshl_add_u32 v105, v104, 9, v1
	s_lshl_b32 s11, s20, 9
	s_add_i32 s17, s2, -8
	s_ashr_i32 s33, s10, 31
	v_cmp_neq_f32_e64 s[42:43], s25, 0
	s_mul_i32 s66, s20, s8
	s_lshl_b64 s[48:49], s[30:31], 1
	s_lshl_b64 s[50:51], s[44:45], 1
	s_lshl_b64 s[52:53], s[46:47], 1
	v_lshl_add_u32 v106, v104, 10, v2
	s_lshl_b32 s67, s20, 10
	s_xor_b64 s[54:55], s[0:1], -1
	v_mov_b32_e32 v101, 0
	v_cndmask_b32_e64 v107, 0, 1, s[40:41]
                                        ; implicit-def: $vgpr108
	s_branch .LBB0_5
.LBB0_4:                                ;   in Loop: Header=BB0_5 Depth=1
	s_or_b64 exec, exec, s[22:23]
	v_add_u32_e32 v98, s66, v98
	v_cmp_le_u32_e32 vcc, s9, v98
	s_or_b64 s[38:39], vcc, s[38:39]
	s_andn2_b64 exec, exec, s[38:39]
	s_cbranch_execz .LBB0_51
.LBB0_5:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_13 Depth 2
                                        ;       Child Loop BB0_18 Depth 3
	v_cmp_gt_u32_e64 s[20:21], s3, v98
	s_and_b64 s[56:57], s[54:55], s[20:21]
	v_mov_b32_e32 v99, v101
	s_and_saveexec_b64 s[22:23], s[56:57]
	s_cbranch_execz .LBB0_7
; %bb.6:                                ;   in Loop: Header=BB0_5 Depth=1
	v_lshl_add_u64 v[2:3], v[98:99], 1, s[28:29]
	global_load_ushort v108, v[2:3], off
.LBB0_7:                                ;   in Loop: Header=BB0_5 Depth=1
	s_or_b64 exec, exec, s[22:23]
	v_mad_u64_u32 v[2:3], s[22:23], v98, s10, 0
	v_mov_b32_e32 v4, v3
	v_mad_u64_u32 v[4:5], s[22:23], v98, s33, v[4:5]
	v_mov_b32_e32 v3, v4
	s_mov_b32 s36, 0
	v_lshl_add_u64 v[102:103], v[2:3], 1, s[4:5]
	v_mov_b32_e32 v113, 0
	s_mov_b32 s68, s36
	v_mov_b32_e32 v112, 0
	v_mov_b32_e32 v111, 0
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v109, 0
	s_branch .LBB0_13
.LBB0_8:                                ;   in Loop: Header=BB0_13 Depth=2
	s_or_b64 exec, exec, s[62:63]
.LBB0_9:                                ;   in Loop: Header=BB0_13 Depth=2
	s_or_b64 exec, exec, s[60:61]
.LBB0_10:                               ;   in Loop: Header=BB0_13 Depth=2
	s_or_b64 exec, exec, s[58:59]
.LBB0_11:                               ;   in Loop: Header=BB0_13 Depth=2
	s_or_b64 exec, exec, s[56:57]
	s_waitcnt vmcnt(3) lgkmcnt(4)
	;;#ASMSTART
	v_dot2c_f32_f16 v113, v94, v46
	;;#ASMEND
	s_waitcnt lgkmcnt(3)
	;;#ASMSTART
	v_dot2c_f32_f16 v112, v86, v46
	;;#ASMEND
	s_waitcnt lgkmcnt(2)
	;;#ASMSTART
	v_dot2c_f32_f16 v111, v78, v46
	;;#ASMEND
	s_waitcnt lgkmcnt(1)
	;;#ASMSTART
	v_dot2c_f32_f16 v110, v74, v46
	;;#ASMEND
	s_waitcnt lgkmcnt(0)
	;;#ASMSTART
	v_dot2c_f32_f16 v109, v66, v46
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v113, v95, v47
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v112, v87, v47
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v111, v79, v47
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v110, v75, v47
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v109, v67, v47
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v113, v96, v48
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v112, v88, v48
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v111, v80, v48
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v110, v76, v48
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v109, v68, v48
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v113, v97, v49
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v112, v89, v49
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v111, v81, v49
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v110, v77, v49
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v109, v69, v49
	;;#ASMEND
	s_waitcnt vmcnt(2)
	;;#ASMSTART
	v_dot2c_f32_f16 v113, v90, v18
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v112, v82, v18
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v111, v70, v18
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v110, v58, v18
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v109, v54, v18
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v113, v91, v19
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v112, v83, v19
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v111, v71, v19
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v110, v59, v19
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v109, v55, v19
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v113, v92, v20
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v112, v84, v20
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v111, v72, v20
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v110, v60, v20
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v109, v56, v20
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v113, v93, v21
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v112, v85, v21
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v111, v73, v21
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v110, v61, v21
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v109, v57, v21
	;;#ASMEND
	s_waitcnt vmcnt(1)
	;;#ASMSTART
	v_dot2c_f32_f16 v113, v62, v6
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v112, v50, v6
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v111, v42, v6
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v110, v38, v6
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v109, v34, v6
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v113, v63, v7
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v112, v51, v7
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v111, v43, v7
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v110, v39, v7
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v109, v35, v7
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v113, v64, v8
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v112, v52, v8
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v111, v44, v8
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v110, v40, v8
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v109, v36, v8
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v113, v65, v9
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v112, v53, v9
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v111, v45, v9
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v110, v41, v9
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v109, v37, v9
	;;#ASMEND
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	v_dot2c_f32_f16 v113, v30, v2
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v112, v26, v2
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v111, v22, v2
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v110, v14, v2
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v109, v10, v2
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v113, v31, v3
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v112, v27, v3
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v111, v23, v3
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v110, v15, v3
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v109, v11, v3
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v113, v32, v4
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v112, v28, v4
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v111, v24, v4
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v110, v16, v4
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v109, v12, v4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v113, v33, v5
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v112, v29, v5
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v111, v25, v5
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v110, v17, v5
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v109, v13, v5
	;;#ASMEND
.LBB0_12:                               ;   in Loop: Header=BB0_13 Depth=2
	s_or_b64 exec, exec, s[22:23]
	s_addk_i32 s68, 0x800
	s_cmp_ge_u32 s68, s2
	s_cbranch_scc1 .LBB0_45
.LBB0_13:                               ;   Parent Loop BB0_5 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_18 Depth 3
	s_cmp_eq_u32 s68, 0
	s_cselect_b64 s[22:23], -1, 0
	s_add_i32 s56, s36, 0x4000
	s_cmp_eq_u32 s68, s56
	s_cselect_b64 s[58:59], -1, 0
	s_or_b64 s[58:59], s[22:23], s[58:59]
	s_andn2_b64 vcc, exec, s[58:59]
	s_cbranch_vccz .LBB0_15
; %bb.14:                               ;   in Loop: Header=BB0_13 Depth=2
	s_and_saveexec_b64 s[22:23], s[20:21]
	s_cbranch_execz .LBB0_12
	s_branch .LBB0_40
.LBB0_15:                               ;   in Loop: Header=BB0_13 Depth=2
	s_and_b64 s[22:23], s[22:23], exec
	s_cselect_b32 s36, s36, s56
	s_sub_i32 s22, s2, s36
	s_min_u32 s69, s22, 0x4000
	s_lshl_b64 s[22:23], s[36:37], 1
	s_add_u32 s56, s6, s22
	s_addc_u32 s57, s7, s23
	s_mov_b32 s70, 0
	s_mov_b64 s[58:59], 0
	v_mov_b32_e32 v4, v106
	s_barrier
                                        ; implicit-def: $sgpr60_sgpr61
	s_branch .LBB0_18
.LBB0_16:                               ;   in Loop: Header=BB0_18 Depth=3
	s_add_i32 s70, s70, s11
	s_cmpk_gt_u32 s70, 0x3fff
	s_cselect_b64 s[22:23], -1, 0
	s_andn2_b64 s[60:61], s[60:61], exec
	s_and_b64 s[22:23], s[22:23], exec
	v_add_u32_e32 v4, s67, v4
	s_or_b64 s[60:61], s[60:61], s[22:23]
.LBB0_17:                               ;   in Loop: Header=BB0_18 Depth=3
	s_or_b64 exec, exec, s[62:63]
	s_and_b64 s[22:23], exec, s[60:61]
	s_or_b64 s[58:59], s[22:23], s[58:59]
	s_andn2_b64 exec, exec, s[58:59]
	s_cbranch_execz .LBB0_39
.LBB0_18:                               ;   Parent Loop BB0_5 Depth=1
                                        ;     Parent Loop BB0_13 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	v_add_u32_e32 v100, s70, v105
	v_cmp_gt_u32_e32 vcc, s69, v100
	s_or_b64 s[60:61], s[60:61], exec
	s_and_saveexec_b64 s[62:63], vcc
	s_cbranch_execz .LBB0_17
; %bb.19:                               ;   in Loop: Header=BB0_18 Depth=3
	v_lshl_add_u64 v[2:3], v[100:101], 1, s[56:57]
	s_mov_b64 s[22:23], -1
	s_and_b64 vcc, exec, s[40:41]
	s_cbranch_vccz .LBB0_21
; %bb.20:                               ;   in Loop: Header=BB0_18 Depth=3
	global_load_dwordx4 v[6:9], v[2:3], off
	s_mov_b64 s[22:23], 0
	s_waitcnt vmcnt(0)
	ds_write_b128 v4, v[6:9]
.LBB0_21:                               ;   in Loop: Header=BB0_18 Depth=3
	s_andn2_b64 vcc, exec, s[22:23]
	s_cbranch_vccnz .LBB0_23
; %bb.22:                               ;   in Loop: Header=BB0_18 Depth=3
	v_readfirstlane_b32 s22, v4
	s_mov_b32 m0, s22
	s_nop 0
	global_load_lds_dwordx4 v[2:3], off
.LBB0_23:                               ;   in Loop: Header=BB0_18 Depth=3
	v_lshl_add_u64 v[2:3], s[30:31], 1, v[2:3]
	v_add_u32_e32 v5, 0x8000, v4
	v_cmp_ne_u32_e64 s[22:23], 1, v107
	s_andn2_b64 vcc, exec, s[40:41]
	s_mov_b64 s[64:65], -1
	s_cbranch_vccnz .LBB0_25
; %bb.24:                               ;   in Loop: Header=BB0_18 Depth=3
	global_load_dwordx4 v[6:9], v[2:3], off
	s_mov_b64 s[64:65], 0
	s_waitcnt vmcnt(0)
	ds_write_b128 v5, v[6:9]
.LBB0_25:                               ;   in Loop: Header=BB0_18 Depth=3
	s_andn2_b64 vcc, exec, s[64:65]
	s_cbranch_vccnz .LBB0_27
; %bb.26:                               ;   in Loop: Header=BB0_18 Depth=3
	v_readfirstlane_b32 s64, v5
	s_mov_b32 m0, s64
	s_nop 0
	global_load_lds_dwordx4 v[2:3], off
.LBB0_27:                               ;   in Loop: Header=BB0_18 Depth=3
	v_lshl_add_u64 v[2:3], v[2:3], 0, s[48:49]
	v_add_u32_e32 v5, 0x10000, v4
	s_and_b64 vcc, exec, s[22:23]
	s_mov_b64 s[64:65], -1
	s_cbranch_vccnz .LBB0_29
; %bb.28:                               ;   in Loop: Header=BB0_18 Depth=3
	global_load_dwordx4 v[6:9], v[2:3], off
	s_mov_b64 s[64:65], 0
	s_waitcnt vmcnt(0)
	ds_write_b128 v5, v[6:9]
.LBB0_29:                               ;   in Loop: Header=BB0_18 Depth=3
	s_andn2_b64 vcc, exec, s[64:65]
	s_cbranch_vccnz .LBB0_31
; %bb.30:                               ;   in Loop: Header=BB0_18 Depth=3
	v_readfirstlane_b32 s64, v5
	s_mov_b32 m0, s64
	s_nop 0
	global_load_lds_dwordx4 v[2:3], off
.LBB0_31:                               ;   in Loop: Header=BB0_18 Depth=3
	v_lshl_add_u64 v[2:3], v[2:3], 0, s[48:49]
	v_add_u32_e32 v5, 0x18000, v4
	s_and_b64 vcc, exec, s[22:23]
	s_mov_b64 s[64:65], -1
	s_cbranch_vccnz .LBB0_33
; %bb.32:                               ;   in Loop: Header=BB0_18 Depth=3
	global_load_dwordx4 v[6:9], v[2:3], off
	s_mov_b64 s[64:65], 0
	s_waitcnt vmcnt(0)
	ds_write_b128 v5, v[6:9]
.LBB0_33:                               ;   in Loop: Header=BB0_18 Depth=3
	s_andn2_b64 vcc, exec, s[64:65]
	s_cbranch_vccnz .LBB0_35
; %bb.34:                               ;   in Loop: Header=BB0_18 Depth=3
	v_readfirstlane_b32 s64, v5
	s_mov_b32 m0, s64
	s_nop 0
	global_load_lds_dwordx4 v[2:3], off
.LBB0_35:                               ;   in Loop: Header=BB0_18 Depth=3
	v_lshl_add_u64 v[2:3], v[2:3], 0, s[48:49]
	v_add_u32_e32 v5, 0x20000, v4
	s_and_b64 vcc, exec, s[22:23]
	s_mov_b64 s[22:23], -1
	s_cbranch_vccnz .LBB0_37
; %bb.36:                               ;   in Loop: Header=BB0_18 Depth=3
	global_load_dwordx4 v[6:9], v[2:3], off
	s_mov_b64 s[22:23], 0
	s_waitcnt vmcnt(0)
	ds_write_b128 v5, v[6:9]
.LBB0_37:                               ;   in Loop: Header=BB0_18 Depth=3
	s_andn2_b64 vcc, exec, s[22:23]
	s_cbranch_vccnz .LBB0_16
; %bb.38:                               ;   in Loop: Header=BB0_18 Depth=3
	v_readfirstlane_b32 s22, v5
	s_mov_b32 m0, s22
	s_nop 0
	global_load_lds_dwordx4 v[2:3], off
	s_branch .LBB0_16
.LBB0_39:                               ;   in Loop: Header=BB0_13 Depth=2
	s_or_b64 exec, exec, s[58:59]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	s_and_saveexec_b64 s[22:23], s[20:21]
	s_cbranch_execz .LBB0_12
.LBB0_40:                               ;   in Loop: Header=BB0_13 Depth=2
	v_add_u32_e32 v117, s68, v1
	v_min_u32_e32 v100, s17, v117
	v_add_u32_e32 v116, 0x200, v117
	v_lshl_add_u64 v[2:3], v[100:101], 1, v[102:103]
	v_min_u32_e32 v100, s17, v116
	v_add_u32_e32 v115, 0x400, v117
	v_lshl_add_u64 v[4:5], v[100:101], 1, v[102:103]
	v_min_u32_e32 v100, s17, v115
	v_add_u32_e32 v114, 0x600, v117
	global_load_dwordx4 v[46:49], v[2:3], off nt
	global_load_dwordx4 v[18:21], v[4:5], off nt
	v_lshl_add_u64 v[2:3], v[100:101], 1, v[102:103]
	v_min_u32_e32 v100, s17, v114
	v_lshl_add_u64 v[4:5], v[100:101], 1, v[102:103]
	global_load_dwordx4 v[6:9], v[2:3], off nt
	s_nop 0
	global_load_dwordx4 v[2:5], v[4:5], off nt
	v_cmp_gt_u32_e32 vcc, s2, v117
	v_mov_b32_e32 v86, 0
	v_mov_b32_e32 v87, 0
	v_mov_b32_e32 v88, 0
	v_mov_b32_e32 v89, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v79, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v75, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v96, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v90, 0
	v_mov_b32_e32 v91, 0
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v84, 0
	v_mov_b32_e32 v85, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v73, 0
	v_mov_b32_e32 v58, 0
	v_mov_b32_e32 v59, 0
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v56, 0
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v63, 0
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v39, 0
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v31, 0
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
	v_mov_b32_e32 v14, 0
	v_mov_b32_e32 v15, 0
	v_mov_b32_e32 v16, 0
	v_mov_b32_e32 v17, 0
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v11, 0
	v_mov_b32_e32 v12, 0
	v_mov_b32_e32 v13, 0
	s_and_saveexec_b64 s[56:57], vcc
	s_cbranch_execz .LBB0_11
; %bb.41:                               ;   in Loop: Header=BB0_13 Depth=2
	v_subrev_u32_e32 v10, s36, v117
	v_lshlrev_b32_e32 v100, 1, v10
	v_add_u32_e32 v10, 0x10000, v100
	ds_read_b128 v[94:97], v100
	ds_read_b128 v[86:89], v100 offset:32768
	v_add_u32_e32 v11, 0x18000, v100
	ds_read_b128 v[78:81], v10
	ds_read_b128 v[74:77], v11
	v_add_u32_e32 v10, 0x20000, v100
	ds_read_b128 v[66:69], v10
	v_cmp_gt_u32_e32 vcc, s2, v116
	v_mov_b32_e32 v13, 0
	v_mov_b32_e32 v12, 0
	v_mov_b32_e32 v11, 0
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v17, 0
	v_mov_b32_e32 v16, 0
	v_mov_b32_e32 v15, 0
	v_mov_b32_e32 v14, 0
	v_mov_b32_e32 v25, 0
	v_mov_b32_e32 v24, 0
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v39, 0
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v63, 0
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v56, 0
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v59, 0
	v_mov_b32_e32 v58, 0
	v_mov_b32_e32 v73, 0
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v85, 0
	v_mov_b32_e32 v84, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v91, 0
	v_mov_b32_e32 v90, 0
	s_and_saveexec_b64 s[58:59], vcc
	s_cbranch_execz .LBB0_10
; %bb.42:                               ;   in Loop: Header=BB0_13 Depth=2
	v_add_u32_e32 v10, 0x10400, v100
	ds_read_b128 v[90:93], v100 offset:1024
	ds_read_b128 v[82:85], v100 offset:33792
	v_add_u32_e32 v11, 0x18400, v100
	ds_read_b128 v[70:73], v10
	ds_read_b128 v[58:61], v11
	v_add_u32_e32 v10, 0x20400, v100
	ds_read_b128 v[54:57], v10
	v_cmp_gt_u32_e32 vcc, s2, v115
	v_mov_b32_e32 v13, 0
	v_mov_b32_e32 v12, 0
	v_mov_b32_e32 v11, 0
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v17, 0
	v_mov_b32_e32 v16, 0
	v_mov_b32_e32 v15, 0
	v_mov_b32_e32 v14, 0
	v_mov_b32_e32 v25, 0
	v_mov_b32_e32 v24, 0
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v39, 0
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v63, 0
	v_mov_b32_e32 v62, 0
	s_and_saveexec_b64 s[60:61], vcc
	s_cbranch_execz .LBB0_9
; %bb.43:                               ;   in Loop: Header=BB0_13 Depth=2
	v_add_u32_e32 v10, 0x10800, v100
	ds_read_b128 v[62:65], v100 offset:2048
	ds_read_b128 v[50:53], v100 offset:34816
	v_add_u32_e32 v11, 0x18800, v100
	ds_read_b128 v[42:45], v10
	ds_read_b128 v[38:41], v11
	v_add_u32_e32 v10, 0x20800, v100
	ds_read_b128 v[34:37], v10
	v_cmp_gt_u32_e32 vcc, s2, v114
	v_mov_b32_e32 v13, 0
	v_mov_b32_e32 v12, 0
	v_mov_b32_e32 v11, 0
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v17, 0
	v_mov_b32_e32 v16, 0
	v_mov_b32_e32 v15, 0
	v_mov_b32_e32 v14, 0
	v_mov_b32_e32 v25, 0
	v_mov_b32_e32 v24, 0
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v30, 0
	s_and_saveexec_b64 s[62:63], vcc
	s_cbranch_execz .LBB0_8
; %bb.44:                               ;   in Loop: Header=BB0_13 Depth=2
	v_add_u32_e32 v10, 0x10c00, v100
	ds_read_b128 v[30:33], v100 offset:3072
	ds_read_b128 v[26:29], v100 offset:35840
	v_add_u32_e32 v11, 0x18c00, v100
	ds_read_b128 v[22:25], v10
	ds_read_b128 v[14:17], v11
	v_add_u32_e32 v10, 0x20c00, v100
	ds_read_b128 v[10:13], v10
	s_branch .LBB0_8
.LBB0_45:                               ;   in Loop: Header=BB0_5 Depth=1
	s_and_saveexec_b64 s[22:23], s[20:21]
	s_cbranch_execz .LBB0_4
; %bb.46:                               ;   in Loop: Header=BB0_5 Depth=1
	v_add_f32_dpp v6, v111, v111 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v113, v113 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v112, v112 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v6, v6 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v6, v6 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v6, v6 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v6, v6 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v110, v110 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v6, v6 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v3, v2 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v5, v4 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v6, v6 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v8, v7 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_add_f32_dpp v6, v6, v6 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v9, v6, v6 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v109, v109 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v10, v9 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v6, v6 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v6, v6, v6 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v6, v6, v6 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v11, v6, v6 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_mov_b32_dpp v12, v11 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_and_b64 exec, exec, s[18:19]
	s_cbranch_execz .LBB0_4
; %bb.47:                               ;   in Loop: Header=BB0_5 Depth=1
	v_add_f32_e32 v6, v2, v3
	v_add_f32_e32 v5, v4, v5
	v_add_f32_e32 v4, v7, v8
	v_add_f32_e32 v3, v9, v10
	s_andn2_b64 vcc, exec, s[42:43]
	v_add_f32_e32 v2, v11, v12
	s_cbranch_vccnz .LBB0_49
; %bb.48:                               ;   in Loop: Header=BB0_5 Depth=1
	v_lshlrev_b64 v[8:9], 1, v[98:99]
	v_lshl_add_u64 v[10:11], s[12:13], 0, v[8:9]
	global_load_ushort v7, v[10:11], off
	s_waitcnt vmcnt(1)
	v_cvt_f32_f16_e32 v12, v108
	v_lshl_add_u64 v[8:9], s[14:15], 0, v[8:9]
	v_lshl_add_u64 v[10:11], s[44:45], 1, v[10:11]
	s_waitcnt vmcnt(0)
	v_cvt_f32_f16_e32 v7, v7
	v_mul_f32_e32 v7, s25, v7
	v_fmac_f32_e32 v7, s24, v6
	v_add_f32_e32 v13, v7, v12
	v_cndmask_b32_e64 v7, v13, v7, s[0:1]
	v_cvt_f16_f32_e32 v7, v7
	global_store_short v[8:9], v7, off
	global_load_ushort v7, v[10:11], off
	v_lshl_add_u64 v[8:9], s[46:47], 1, v[8:9]
	v_lshl_add_u64 v[10:11], v[10:11], 0, s[50:51]
	s_waitcnt vmcnt(0)
	v_cvt_f32_f16_e32 v7, v7
	v_mul_f32_e32 v7, s25, v7
	v_fmac_f32_e32 v7, s24, v5
	v_add_f32_e32 v13, v7, v12
	v_cndmask_b32_e64 v7, v13, v7, s[0:1]
	v_cvt_f16_f32_e32 v7, v7
	global_store_short v[8:9], v7, off
	global_load_ushort v7, v[10:11], off
	v_lshl_add_u64 v[8:9], v[8:9], 0, s[52:53]
	v_lshl_add_u64 v[10:11], v[10:11], 0, s[50:51]
	s_waitcnt vmcnt(0)
	v_cvt_f32_f16_e32 v7, v7
	v_mul_f32_e32 v7, s25, v7
	v_fmac_f32_e32 v7, s24, v4
	v_add_f32_e32 v13, v7, v12
	v_cndmask_b32_e64 v7, v13, v7, s[0:1]
	v_cvt_f16_f32_e32 v7, v7
	global_store_short v[8:9], v7, off
	global_load_ushort v7, v[10:11], off
	v_lshl_add_u64 v[8:9], v[8:9], 0, s[52:53]
	v_lshl_add_u64 v[10:11], v[10:11], 0, s[50:51]
	s_waitcnt vmcnt(0)
	v_cvt_f32_f16_e32 v7, v7
	v_mul_f32_e32 v7, s25, v7
	v_fmac_f32_e32 v7, s24, v3
	v_add_f32_e32 v13, v7, v12
	v_cndmask_b32_e64 v7, v13, v7, s[0:1]
	v_cvt_f16_f32_e32 v7, v7
	global_store_short v[8:9], v7, off
	global_load_ushort v7, v[10:11], off
	v_lshl_add_u64 v[8:9], v[8:9], 0, s[52:53]
	s_waitcnt vmcnt(0)
	v_cvt_f32_f16_e32 v7, v7
	v_mul_f32_e32 v7, s25, v7
	v_fmac_f32_e32 v7, s24, v2
	v_add_f32_e32 v10, v7, v12
	v_cndmask_b32_e64 v7, v10, v7, s[0:1]
	v_cvt_f16_f32_e32 v7, v7
	global_store_short v[8:9], v7, off
	s_cbranch_execnz .LBB0_4
	s_branch .LBB0_50
.LBB0_49:                               ;   in Loop: Header=BB0_5 Depth=1
.LBB0_50:                               ;   in Loop: Header=BB0_5 Depth=1
	v_mul_f32_e32 v7, s24, v6
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v6, s24, v6, v108 op_sel_hi:[0,0,1]
	v_cndmask_b32_e64 v6, v6, v7, s[0:1]
	v_cvt_f16_f32_e32 v8, v6
	v_mul_f32_e32 v9, s24, v5
	v_fma_mix_f32 v5, s24, v5, v108 op_sel_hi:[0,0,1]
	v_lshl_add_u64 v[6:7], v[98:99], 1, s[14:15]
	v_cndmask_b32_e64 v5, v5, v9, s[0:1]
	global_store_short v[6:7], v8, off
	v_cvt_f16_f32_e32 v8, v5
	v_mul_f32_e32 v5, s24, v4
	v_fma_mix_f32 v4, s24, v4, v108 op_sel_hi:[0,0,1]
	v_cndmask_b32_e64 v4, v4, v5, s[0:1]
	v_cvt_f16_f32_e32 v9, v4
	v_lshl_add_u64 v[4:5], s[46:47], 1, v[6:7]
	v_mul_f32_e32 v6, s24, v3
	v_fma_mix_f32 v3, s24, v3, v108 op_sel_hi:[0,0,1]
	v_cndmask_b32_e64 v3, v3, v6, s[0:1]
	v_cvt_f16_f32_e32 v6, v3
	v_mul_f32_e32 v3, s24, v2
	v_fma_mix_f32 v2, s24, v2, v108 op_sel_hi:[0,0,1]
	v_cndmask_b32_e64 v2, v2, v3, s[0:1]
	v_cvt_f16_f32_e32 v7, v2
	global_store_short v[4:5], v8, off
	v_lshl_add_u64 v[4:5], v[4:5], 0, s[52:53]
	v_lshl_add_u64 v[2:3], v[4:5], 0, s[52:53]
	global_store_short v[4:5], v9, off
	global_store_short v[2:3], v6, off
	v_lshl_add_u64 v[2:3], v[2:3], 0, s[52:53]
	global_store_short v[2:3], v7, off
	s_branch .LBB0_4
.LBB0_51:
	s_or_b64 exec, exec, s[34:35]
	s_mov_b64 s[0:1], 0
.LBB0_52:
	s_andn2_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB0_93
; %bb.53:
	v_and_b32_e32 v8, 0x3ff, v0
	s_mul_i32 s9, s2, 5
	v_lshlrev_b32_e32 v102, 3, v8
	s_cmp_eq_u32 s30, s2
	s_mov_b64 s[18:19], 0
	s_cbranch_scc1 .LBB0_55
; %bb.54:
	s_and_b32 s20, s6, 15
	s_mov_b32 s21, 0
	s_mov_b64 s[0:1], -1
	s_min_u32 s9, s9, 0x14000
	v_lshl_add_u32 v9, v104, 9, v102
	s_cbranch_execz .LBB0_56
	s_branch .LBB0_57
.LBB0_55:
                                        ; implicit-def: $sgpr20_sgpr21
	s_mov_b64 s[0:1], 0
	s_min_u32 s9, s9, 0x14000
	v_lshl_add_u32 v9, v104, 9, v102
.LBB0_56:
	s_and_b32 s0, s2, 7
	s_cmp_lg_u32 s0, 0
	s_mov_b32 s21, 0
	s_cselect_b64 s[0:1], -1, 0
	s_and_b32 s20, s6, 15
	s_cmp_lg_u64 s[20:21], 0
	s_cselect_b64 s[18:19], -1, 0
	s_or_b64 s[0:1], s[0:1], s[18:19]
	s_mov_b64 s[18:19], -1
.LBB0_57:
	s_andn2_b64 vcc, exec, s[0:1]
	v_cmp_gt_u32_e64 s[0:1], s9, v9
	s_cbranch_vccz .LBB0_59
; %bb.58:
	s_andn2_b64 vcc, exec, s[18:19]
	s_cbranch_vccz .LBB0_67
	s_branch .LBB0_71
.LBB0_59:
	s_and_saveexec_b64 s[18:19], s[0:1]
	s_cbranch_execz .LBB0_66
; %bb.60:
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
	v_lshl_add_u32 v10, v104, 10, v0
	v_mov_b32_e32 v0, v9
	s_branch .LBB0_62
.LBB0_61:                               ;   in Loop: Header=BB0_62 Depth=1
	v_add_u32_e32 v0, 0x2000, v0
	v_cmp_le_u32_e32 vcc, s9, v0
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[4:5]
	s_or_b64 s[0:1], vcc, s[0:1]
	v_add_u32_e32 v10, 0x4000, v10
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB0_66
.LBB0_62:                               ; =>This Inner Loop Header: Depth=1
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
	s_cbranch_vccz .LBB0_64
; %bb.63:                               ;   in Loop: Header=BB0_62 Depth=1
	global_load_dwordx4 v[12:15], v[6:7], off
	s_mov_b64 s[22:23], 0
	s_waitcnt vmcnt(0)
	ds_write_b128 v10, v[12:15]
.LBB0_64:                               ;   in Loop: Header=BB0_62 Depth=1
	s_andn2_b64 vcc, exec, s[22:23]
	s_cbranch_vccnz .LBB0_61
; %bb.65:                               ;   in Loop: Header=BB0_62 Depth=1
	v_readfirstlane_b32 s22, v10
	s_mov_b32 m0, s22
	s_nop 0
	global_load_lds_dwordx4 v[6:7], off
	s_branch .LBB0_61
.LBB0_66:
	s_or_b64 exec, exec, s[18:19]
	s_cbranch_execnz .LBB0_71
.LBB0_67:
	v_cmp_gt_u32_e32 vcc, s9, v9
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_70
; %bb.68:
	v_lshlrev_b32_e32 v0, 4, v8
	v_lshl_add_u32 v2, v104, 10, v0
	v_lshlrev_b32_e32 v0, 1, v9
	v_mov_b32_e32 v1, 0
	v_lshl_add_u64 v[0:1], s[6:7], 0, v[0:1]
	s_mov_b64 s[6:7], 0
	s_mov_b64 s[18:19], 0x4000
.LBB0_69:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s11, v2
	s_mov_b32 m0, s11
	v_add_u32_e32 v9, 0x2000, v9
	global_load_lds_dwordx4 v[0:1], off
	v_cmp_le_u32_e32 vcc, s9, v9
	v_add_u32_e32 v2, 0x4000, v2
	s_or_b64 s[6:7], vcc, s[6:7]
	v_lshl_add_u64 v[0:1], v[0:1], 0, s[18:19]
	s_andn2_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB0_69
.LBB0_70:
	s_or_b64 exec, exec, s[0:1]
.LBB0_71:
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
	s_max_i32 s45, s0, 4
	s_mul_i32 s0, s45, s16
	v_add_u32_e32 v96, s0, v104
	v_cmp_gt_u32_e32 vcc, s45, v104
	v_cmp_gt_u32_e64 s[0:1], s3, v96
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_93
; %bb.72:
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
	s_ashr_i32 s44, s10, 31
	v_cmp_eq_u32_e64 s[16:17], 63, v8
	v_cmp_neq_f32_e64 s[20:21], s25, 0
	s_mul_i32 s45, s45, s8
	s_lshl_b32 s46, s2, 1
	s_lshl_b64 s[26:27], s[22:23], 1
	s_lshl_b64 s[34:35], s[30:31], 1
	v_lshlrev_b32_e32 v103, 4, v8
	s_lshl_b32 s47, s2, 2
	s_mul_i32 s48, s2, 6
	s_lshl_b32 s49, s2, 3
	v_mov_b32_e32 v99, 0
	v_cmp_ne_u32_e64 s[8:9], 1, v0
                                        ; implicit-def: $vgpr104
	s_branch .LBB0_74
.LBB0_73:                               ;   in Loop: Header=BB0_74 Depth=1
	s_or_b64 exec, exec, s[36:37]
	v_add_u32_e32 v96, s45, v96
	v_cmp_le_u32_e32 vcc, s3, v96
	s_or_b64 s[18:19], vcc, s[18:19]
	s_andn2_b64 exec, exec, s[18:19]
	s_cbranch_execz .LBB0_93
.LBB0_74:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_82 Depth 2
	s_and_b64 vcc, exec, s[6:7]
	s_cbranch_vccnz .LBB0_76
; %bb.75:                               ;   in Loop: Header=BB0_74 Depth=1
	v_mov_b32_e32 v97, v99
	v_lshl_add_u64 v[0:1], v[96:97], 1, s[28:29]
	global_load_ushort v104, v[0:1], off
.LBB0_76:                               ;   in Loop: Header=BB0_74 Depth=1
	s_and_b64 vcc, exec, s[8:9]
	s_cbranch_vccnz .LBB0_87
; %bb.77:                               ;   in Loop: Header=BB0_74 Depth=1
	v_min_u32_e32 v3, s33, v96
	v_mad_u64_u32 v[0:1], s[36:37], v3, s10, 0
	v_mov_b32_e32 v2, v1
	v_mad_u64_u32 v[2:3], s[36:37], v3, s44, v[2:3]
	v_mov_b32_e32 v1, v2
	v_lshl_add_u64 v[100:101], v[0:1], 1, s[4:5]
	v_mov_b32_e32 v108, 0
	s_mov_b32 s50, 0
	v_mov_b32_e32 v109, v103
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v97, 0
	s_branch .LBB0_82
.LBB0_78:                               ;   in Loop: Header=BB0_82 Depth=2
	s_or_b64 exec, exec, s[42:43]
.LBB0_79:                               ;   in Loop: Header=BB0_82 Depth=2
	s_or_b64 exec, exec, s[40:41]
.LBB0_80:                               ;   in Loop: Header=BB0_82 Depth=2
	s_or_b64 exec, exec, s[38:39]
.LBB0_81:                               ;   in Loop: Header=BB0_82 Depth=2
	s_or_b64 exec, exec, s[36:37]
	s_waitcnt vmcnt(3) lgkmcnt(4)
	;;#ASMSTART
	v_dot2c_f32_f16 v108, v84, v24
	;;#ASMEND
	s_waitcnt lgkmcnt(3)
	;;#ASMSTART
	v_dot2c_f32_f16 v107, v64, v24
	;;#ASMEND
	s_waitcnt lgkmcnt(2)
	;;#ASMSTART
	v_dot2c_f32_f16 v106, v56, v24
	;;#ASMEND
	s_waitcnt lgkmcnt(1)
	;;#ASMSTART
	v_dot2c_f32_f16 v105, v48, v24
	;;#ASMEND
	s_waitcnt lgkmcnt(0)
	;;#ASMSTART
	v_dot2c_f32_f16 v97, v44, v24
	;;#ASMEND
	s_addk_i32 s50, 0x800
	;;#ASMSTART
	v_dot2c_f32_f16 v108, v85, v25
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v107, v65, v25
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v106, v57, v25
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v105, v49, v25
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v97, v45, v25
	;;#ASMEND
	s_cmp_ge_u32 s50, s2
	;;#ASMSTART
	v_dot2c_f32_f16 v108, v86, v26
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v107, v66, v26
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v106, v58, v26
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v105, v50, v26
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v97, v46, v26
	;;#ASMEND
	v_add_u32_e32 v109, 0x1000, v109
	;;#ASMSTART
	v_dot2c_f32_f16 v108, v87, v27
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v107, v67, v27
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v106, v59, v27
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v105, v51, v27
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v97, v47, v27
	;;#ASMEND
	s_waitcnt vmcnt(2)
	;;#ASMSTART
	v_dot2c_f32_f16 v108, v92, v8
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v107, v88, v8
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v106, v76, v8
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v105, v68, v8
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v97, v60, v8
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v108, v93, v9
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v107, v89, v9
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v106, v77, v9
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v105, v69, v9
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v97, v61, v9
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v108, v94, v10
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v107, v90, v10
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v106, v78, v10
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v105, v70, v10
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v97, v62, v10
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v108, v95, v11
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v107, v91, v11
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v106, v79, v11
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v105, v71, v11
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v97, v63, v11
	;;#ASMEND
	s_waitcnt vmcnt(1)
	;;#ASMSTART
	v_dot2c_f32_f16 v108, v80, v4
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v107, v72, v4
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v106, v52, v4
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v105, v40, v4
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v97, v32, v4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v108, v81, v5
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v107, v73, v5
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v106, v53, v5
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v105, v41, v5
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v97, v33, v5
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v108, v82, v6
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v107, v74, v6
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v106, v54, v6
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v105, v42, v6
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v97, v34, v6
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v108, v83, v7
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v107, v75, v7
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v106, v55, v7
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v105, v43, v7
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v97, v35, v7
	;;#ASMEND
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	v_dot2c_f32_f16 v108, v36, v0
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v107, v28, v0
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v106, v20, v0
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v105, v16, v0
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v97, v12, v0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v108, v37, v1
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v107, v29, v1
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v106, v21, v1
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v105, v17, v1
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v97, v13, v1
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v108, v38, v2
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v107, v30, v2
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v106, v22, v2
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v105, v18, v2
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v97, v14, v2
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v108, v39, v3
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v107, v31, v3
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v106, v23, v3
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v105, v19, v3
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v97, v15, v3
	;;#ASMEND
	s_cbranch_scc1 .LBB0_88
.LBB0_82:                               ;   Parent Loop BB0_74 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_add_u32_e32 v16, s50, v102
	v_min_u32_e32 v98, s11, v16
	v_add_u32_e32 v115, 0x200, v16
	v_lshl_add_u64 v[0:1], v[98:99], 1, v[100:101]
	v_min_u32_e32 v98, s11, v115
	v_add_u32_e32 v111, 0x400, v16
	v_lshl_add_u64 v[2:3], v[98:99], 1, v[100:101]
	v_min_u32_e32 v98, s11, v111
	v_add_u32_e32 v110, 0x600, v16
	v_lshl_add_u64 v[12:13], v[98:99], 1, v[100:101]
	v_min_u32_e32 v98, s11, v110
	global_load_dwordx4 v[24:27], v[0:1], off nt
	global_load_dwordx4 v[8:11], v[2:3], off nt
	v_lshl_add_u64 v[14:15], v[98:99], 1, v[100:101]
	global_load_dwordx4 v[4:7], v[12:13], off nt
	global_load_dwordx4 v[0:3], v[14:15], off nt
	v_cmp_gt_u32_e32 vcc, s2, v16
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v56, 0
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v58, 0
	v_mov_b32_e32 v59, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v86, 0
	v_mov_b32_e32 v87, 0
	v_mov_b32_e32 v84, 0
	v_mov_b32_e32 v85, 0
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v88, 0
	v_mov_b32_e32 v89, 0
	v_mov_b32_e32 v90, 0
	v_mov_b32_e32 v91, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v79, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v63, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v73, 0
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v75, 0
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v39, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v21, 0
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v16, 0
	v_mov_b32_e32 v17, 0
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v12, 0
	v_mov_b32_e32 v13, 0
	v_mov_b32_e32 v14, 0
	v_mov_b32_e32 v15, 0
	s_and_saveexec_b64 s[36:37], vcc
	s_cbranch_execz .LBB0_81
; %bb.83:                               ;   in Loop: Header=BB0_82 Depth=2
	v_add_u32_e32 v114, s46, v109
	v_add_u32_e32 v112, s48, v109
	ds_read_b128 v[84:87], v109
	v_add_u32_e32 v113, s47, v109
	ds_read_b128 v[64:67], v114
	ds_read_b128 v[56:59], v113
	v_add_u32_e32 v98, s49, v109
	ds_read_b128 v[48:51], v112
	ds_read_b128 v[44:47], v98
	v_cmp_gt_u32_e32 vcc, s2, v115
	v_mov_b32_e32 v15, 0
	v_mov_b32_e32 v14, 0
	v_mov_b32_e32 v13, 0
	v_mov_b32_e32 v12, 0
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v17, 0
	v_mov_b32_e32 v16, 0
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v21, 0
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v39, 0
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v75, 0
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v73, 0
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v63, 0
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v79, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v91, 0
	v_mov_b32_e32 v90, 0
	v_mov_b32_e32 v89, 0
	v_mov_b32_e32 v88, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v92, 0
	s_and_saveexec_b64 s[38:39], vcc
	s_cbranch_execz .LBB0_80
; %bb.84:                               ;   in Loop: Header=BB0_82 Depth=2
	ds_read_b128 v[92:95], v109 offset:1024
	ds_read_b128 v[88:91], v114 offset:1024
	ds_read_b128 v[76:79], v113 offset:1024
	ds_read_b128 v[68:71], v112 offset:1024
	ds_read_b128 v[60:63], v98 offset:1024
	v_cmp_gt_u32_e32 vcc, s2, v111
	v_mov_b32_e32 v15, 0
	v_mov_b32_e32 v14, 0
	v_mov_b32_e32 v13, 0
	v_mov_b32_e32 v12, 0
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v17, 0
	v_mov_b32_e32 v16, 0
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v21, 0
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v39, 0
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v75, 0
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v73, 0
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v80, 0
	s_and_saveexec_b64 s[40:41], vcc
	s_cbranch_execz .LBB0_79
; %bb.85:                               ;   in Loop: Header=BB0_82 Depth=2
	ds_read_b128 v[80:83], v109 offset:2048
	ds_read_b128 v[72:75], v114 offset:2048
	ds_read_b128 v[52:55], v113 offset:2048
	ds_read_b128 v[40:43], v112 offset:2048
	ds_read_b128 v[32:35], v98 offset:2048
	v_cmp_gt_u32_e32 vcc, s2, v110
	v_mov_b32_e32 v15, 0
	v_mov_b32_e32 v14, 0
	v_mov_b32_e32 v13, 0
	v_mov_b32_e32 v12, 0
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v17, 0
	v_mov_b32_e32 v16, 0
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v21, 0
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v39, 0
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v36, 0
	s_and_saveexec_b64 s[42:43], vcc
	s_cbranch_execz .LBB0_78
; %bb.86:                               ;   in Loop: Header=BB0_82 Depth=2
	ds_read_b128 v[36:39], v109 offset:3072
	ds_read_b128 v[28:31], v114 offset:3072
	ds_read_b128 v[20:23], v113 offset:3072
	ds_read_b128 v[16:19], v112 offset:3072
	ds_read_b128 v[12:15], v98 offset:3072
	s_branch .LBB0_78
.LBB0_87:                               ;   in Loop: Header=BB0_74 Depth=1
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v108, 0
.LBB0_88:                               ;   in Loop: Header=BB0_74 Depth=1
	; sched_barrier mask(0x00000000)
	s_nop 0
	v_add_f32_dpp v4, v106, v106 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v108, v108 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v107, v107 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
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
	v_add_f32_dpp v4, v105, v105 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v0, v0 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v1, v0 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v3, v2 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v6, v5 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_add_f32_dpp v4, v4, v4 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v7, v4, v4 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v97, v97 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v8, v7 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v4, v4, v4 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v4, v4, v4 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v9, v4, v4 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_mov_b32_dpp v10, v9 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_and_saveexec_b64 s[36:37], s[16:17]
	s_cbranch_execz .LBB0_73
; %bb.89:                               ;   in Loop: Header=BB0_74 Depth=1
	v_add_f32_e32 v4, v0, v1
	v_add_f32_e32 v3, v2, v3
	v_add_f32_e32 v2, v5, v6
	v_add_f32_e32 v1, v7, v8
	s_andn2_b64 vcc, exec, s[20:21]
	v_add_f32_e32 v0, v9, v10
	s_cbranch_vccnz .LBB0_91
; %bb.90:                               ;   in Loop: Header=BB0_74 Depth=1
	v_mov_b32_e32 v97, v99
	v_lshlrev_b64 v[6:7], 1, v[96:97]
	v_lshl_add_u64 v[8:9], s[12:13], 0, v[6:7]
	global_load_ushort v5, v[8:9], off
	s_waitcnt vmcnt(1)
	v_cvt_f32_f16_e32 v10, v104
	v_lshl_add_u64 v[6:7], s[14:15], 0, v[6:7]
	v_lshl_add_u64 v[8:9], s[22:23], 1, v[8:9]
	s_waitcnt vmcnt(0)
	v_cvt_f32_f16_e32 v5, v5
	v_mul_f32_e32 v5, s25, v5
	v_fmac_f32_e32 v5, s24, v4
	v_add_f32_e32 v11, v5, v10
	v_cndmask_b32_e64 v5, v11, v5, s[0:1]
	v_cvt_f16_f32_e32 v5, v5
	global_store_short v[6:7], v5, off
	global_load_ushort v5, v[8:9], off
	v_lshl_add_u64 v[6:7], s[30:31], 1, v[6:7]
	v_lshl_add_u64 v[8:9], v[8:9], 0, s[26:27]
	s_waitcnt vmcnt(0)
	v_cvt_f32_f16_e32 v5, v5
	v_mul_f32_e32 v5, s25, v5
	v_fmac_f32_e32 v5, s24, v3
	v_add_f32_e32 v11, v5, v10
	v_cndmask_b32_e64 v5, v11, v5, s[0:1]
	v_cvt_f16_f32_e32 v5, v5
	global_store_short v[6:7], v5, off
	global_load_ushort v5, v[8:9], off
	v_lshl_add_u64 v[6:7], v[6:7], 0, s[34:35]
	v_lshl_add_u64 v[8:9], v[8:9], 0, s[26:27]
	s_waitcnt vmcnt(0)
	v_cvt_f32_f16_e32 v5, v5
	v_mul_f32_e32 v5, s25, v5
	v_fmac_f32_e32 v5, s24, v2
	v_add_f32_e32 v11, v5, v10
	v_cndmask_b32_e64 v5, v11, v5, s[0:1]
	v_cvt_f16_f32_e32 v5, v5
	global_store_short v[6:7], v5, off
	global_load_ushort v5, v[8:9], off
	v_lshl_add_u64 v[6:7], v[6:7], 0, s[34:35]
	v_lshl_add_u64 v[8:9], v[8:9], 0, s[26:27]
	s_waitcnt vmcnt(0)
	v_cvt_f32_f16_e32 v5, v5
	v_mul_f32_e32 v5, s25, v5
	v_fmac_f32_e32 v5, s24, v1
	v_add_f32_e32 v11, v5, v10
	v_cndmask_b32_e64 v5, v11, v5, s[0:1]
	v_cvt_f16_f32_e32 v5, v5
	global_store_short v[6:7], v5, off
	global_load_ushort v5, v[8:9], off
	v_lshl_add_u64 v[6:7], v[6:7], 0, s[34:35]
	s_waitcnt vmcnt(0)
	v_cvt_f32_f16_e32 v5, v5
	v_mul_f32_e32 v5, s25, v5
	v_fmac_f32_e32 v5, s24, v0
	v_add_f32_e32 v8, v5, v10
	v_cndmask_b32_e64 v5, v8, v5, s[0:1]
	v_cvt_f16_f32_e32 v5, v5
	global_store_short v[6:7], v5, off
	s_cbranch_execnz .LBB0_73
	s_branch .LBB0_92
.LBB0_91:                               ;   in Loop: Header=BB0_74 Depth=1
.LBB0_92:                               ;   in Loop: Header=BB0_74 Depth=1
	v_mul_f32_e32 v5, s24, v4
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v4, s24, v4, v104 op_sel_hi:[0,0,1]
	v_cndmask_b32_e64 v4, v4, v5, s[0:1]
	v_cvt_f16_f32_e32 v6, v4
	v_mov_b32_e32 v97, v99
	v_lshl_add_u64 v[4:5], v[96:97], 1, s[14:15]
	global_store_short v[4:5], v6, off
	v_mul_f32_e32 v6, s24, v3
	v_fma_mix_f32 v3, s24, v3, v104 op_sel_hi:[0,0,1]
	v_cndmask_b32_e64 v3, v3, v6, s[0:1]
	v_cvt_f16_f32_e32 v6, v3
	v_mul_f32_e32 v3, s24, v2
	v_fma_mix_f32 v2, s24, v2, v104 op_sel_hi:[0,0,1]
	v_cndmask_b32_e64 v2, v2, v3, s[0:1]
	v_cvt_f16_f32_e32 v7, v2
	v_lshl_add_u64 v[2:3], s[30:31], 1, v[4:5]
	v_mul_f32_e32 v4, s24, v1
	v_fma_mix_f32 v1, s24, v1, v104 op_sel_hi:[0,0,1]
	v_cndmask_b32_e64 v1, v1, v4, s[0:1]
	v_cvt_f16_f32_e32 v4, v1
	v_mul_f32_e32 v1, s24, v0
	v_fma_mix_f32 v0, s24, v0, v104 op_sel_hi:[0,0,1]
	v_cndmask_b32_e64 v0, v0, v1, s[0:1]
	v_cvt_f16_f32_e32 v5, v0
	global_store_short v[2:3], v6, off
	v_lshl_add_u64 v[2:3], v[2:3], 0, s[34:35]
	v_lshl_add_u64 v[0:1], v[2:3], 0, s[34:35]
	global_store_short v[2:3], v7, off
	global_store_short v[0:1], v4, off
	v_lshl_add_u64 v[0:1], v[0:1], 0, s[34:35]
	global_store_short v[0:1], v5, off
	s_branch .LBB0_73
.LBB0_93:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wvSplitK_vllm_f16_tn_bias_n5_y1u4
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
		.amdhsa_next_free_vgpr 118
		.amdhsa_next_free_sgpr 96
		.amdhsa_accum_offset 120
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
	.size	wvSplitK_vllm_f16_tn_bias_n5_y1u4, .Lfunc_end0-wvSplitK_vllm_f16_tn_bias_n5_y1u4
                                        ; -- End function
	.set wvSplitK_vllm_f16_tn_bias_n5_y1u4.num_vgpr, 118
	.set wvSplitK_vllm_f16_tn_bias_n5_y1u4.num_agpr, 0
	.set wvSplitK_vllm_f16_tn_bias_n5_y1u4.numbered_sgpr, 71
	.set wvSplitK_vllm_f16_tn_bias_n5_y1u4.num_named_barrier, 0
	.set wvSplitK_vllm_f16_tn_bias_n5_y1u4.private_seg_size, 0
	.set wvSplitK_vllm_f16_tn_bias_n5_y1u4.uses_vcc, 1
	.set wvSplitK_vllm_f16_tn_bias_n5_y1u4.uses_flat_scratch, 0
	.set wvSplitK_vllm_f16_tn_bias_n5_y1u4.has_dyn_sized_stack, 0
	.set wvSplitK_vllm_f16_tn_bias_n5_y1u4.has_recursion, 0
	.set wvSplitK_vllm_f16_tn_bias_n5_y1u4.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 9660
; TotalNumSgprs: 77
; NumVgprs: 118
; NumAgprs: 0
; TotalNumVgprs: 118
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 163840 bytes/workgroup (compile time only)
; SGPRBlocks: 12
; VGPRBlocks: 14
; NumSGPRsForWavesPerEU: 102
; NumVGPRsForWavesPerEU: 118
; AccumOffset: 120
; Occupancy: 4
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 1
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 29
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_bab05840cca5d45,@object ; @__hip_cuid_bab05840cca5d45
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_bab05840cca5d45
__hip_cuid_bab05840cca5d45:
	.byte	0                               ; 0x0
	.size	__hip_cuid_bab05840cca5d45, 1

	.ident	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.4 26084 f58b06dce1f9c15707c5f808fd002e18c2accf7e)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_bab05840cca5d45
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
  AssertSizeEqual: { 1: 5, 2: 1 }
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
    .name:           wvSplitK_vllm_f16_tn_bias_n5_y1u4
    .private_segment_fixed_size: 0
    .sgpr_count:     77
    .sgpr_spill_count: 0
    .symbol:         wvSplitK_vllm_f16_tn_bias_n5_y1u4.kd
    .vgpr_count:     118
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 1
...

	.end_amdgpu_metadata
