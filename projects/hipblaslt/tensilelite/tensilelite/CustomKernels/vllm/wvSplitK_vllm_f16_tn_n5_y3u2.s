	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 4
	.text
	.protected	wvSplitK_vllm_f16_tn_n5_y3u2 ; -- Begin function wvSplitK_vllm_f16_tn_n5_y3u2
	.globl	wvSplitK_vllm_f16_tn_n5_y3u2
	.p2align	8
	.type	wvSplitK_vllm_f16_tn_n5_y3u2,@function
wvSplitK_vllm_f16_tn_n5_y3u2:           ; @wvSplitK_vllm_f16_tn_n5_y3u2
; %bb.218:
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.219:
.LBB0_0:
	s_load_dwordx4 s[28:31], s[0:1], 0x38
	s_mov_b32 s26, s11
	s_mov_b64 s[0:1], -1
	s_cmpk_gt_i32 s2, 0x4000
	v_bfe_u32 v78, v0, 10, 10
	s_mul_i32 s11, s8, 3
	s_mul_i32 s17, s8, 48
	s_cbranch_scc0 .LBB0_121
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
	v_cmp_gt_u32_e32 vcc, s8, v78
	s_and_saveexec_b64 s[34:35], vcc
	s_cbranch_execz .LBB0_120
; %bb.2:
	s_mul_i32 s9, s8, 3
	v_cvt_f32_u32_e32 v3, s9
	s_mul_i32 s0, s8, s16
	v_add_u32_e32 v1, s0, v78
	v_lshl_add_u32 v2, v1, 1, v1
	v_rcp_iflag_f32_e32 v3, v3
	v_add_u32_e32 v1, 3, v2
	v_cmp_gt_u32_e32 vcc, s3, v2
	v_cmp_lt_u32_e64 s[0:1], s3, v1
	s_add_i32 s74, s3, -3
	v_mov_b32_e32 v4, s74
	s_and_b64 s[0:1], vcc, s[0:1]
	v_cndmask_b32_e64 v68, v2, v4, s[0:1]
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
	s_cselect_b32 s75, s3, s1
	v_cmp_gt_u32_e64 s[0:1], s75, v68
	s_and_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB0_120
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
	s_lshl_b32 s76, s8, 9
	s_add_i32 s77, s2, -8
	s_ashr_i32 s78, s10, 31
	s_ashr_i32 s21, s31, 31
	s_lshl_b64 s[18:19], s[18:19], 1
	s_mov_b32 s20, s31
	s_add_u32 s46, s12, s18
	s_addc_u32 s47, s13, s19
	s_lshl_b64 s[20:21], s[20:21], 1
	s_add_u32 s48, s14, s20
	s_addc_u32 s49, s15, s21
	s_add_u32 s50, s46, s18
	s_addc_u32 s51, s47, s19
	s_add_u32 s52, s48, s20
	s_addc_u32 s53, s49, s21
	s_add_u32 s54, s50, s18
	s_addc_u32 s55, s51, s19
	s_add_u32 s56, s52, s20
	s_addc_u32 s57, s53, s21
	s_add_u32 s58, s54, s18
	v_and_b32_e32 v2, 0x3ff, v0
	s_addc_u32 s59, s55, s19
	v_sub_u32_e64 v1, v1, s3 clamp
	v_lshlrev_b32_e32 v79, 3, v2
	v_cmp_eq_u32_e64 s[0:1], 63, v2
	s_add_u32 s60, s56, s20
	v_lshlrev_b32_e32 v2, 4, v2
	s_mov_b64 s[38:39], 0
	v_cndmask_b32_e32 v1, 0, v1, vcc
	v_lshl_add_u32 v80, v78, 9, v79
	s_mul_i32 s79, s8, s11
	v_cmp_eq_f32_e64 s[42:43], s29, 0
	v_cmp_neq_f32_e64 s[44:45], s29, 0
	s_addc_u32 s61, s57, s21
	s_lshl_b64 s[62:63], s[26:27], 1
	v_lshl_add_u32 v81, v78, 10, v2
	s_lshl_b32 s80, s8, 10
	v_mov_b32_e32 v67, 0
	v_cndmask_b32_e64 v82, 0, 1, s[40:41]
	s_branch .LBB0_6
.LBB0_4:                                ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[66:67]
	v_add_u32_e32 v3, 3, v2
	v_cmp_gt_u32_e32 vcc, s3, v2
	v_cmp_lt_u32_e64 s[8:9], s3, v3
	v_subrev_u32_e32 v3, s3, v3
	v_mov_b32_e32 v4, s74
	s_and_b64 vcc, vcc, s[8:9]
	v_cndmask_b32_e32 v2, v2, v4, vcc
	v_cndmask_b32_e32 v1, v1, v3, vcc
.LBB0_5:                                ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[64:65]
	v_cmp_le_u32_e32 vcc, s75, v2
	s_or_b64 s[38:39], vcc, s[38:39]
	v_mov_b32_e32 v68, v2
	s_andn2_b64 exec, exec, s[38:39]
	s_cbranch_execz .LBB0_120
.LBB0_6:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_10 Depth 2
                                        ;       Child Loop BB0_15 Depth 3
	v_min_u32_e32 v5, s33, v68
	v_mad_u64_u32 v[2:3], s[20:21], v5, s10, 0
	v_mov_b32_e32 v4, v3
	v_mad_u64_u32 v[4:5], s[20:21], v5, s78, v[4:5]
	v_mov_b32_e32 v3, v4
	v_add_u32_e32 v4, 1, v68
	v_min_u32_e32 v7, s33, v4
	v_mad_u64_u32 v[4:5], s[20:21], v7, s10, 0
	v_mov_b32_e32 v6, v5
	v_add_u32_e32 v70, 2, v68
	v_mad_u64_u32 v[6:7], s[20:21], v7, s78, v[6:7]
	v_min_u32_e32 v9, s33, v70
	v_mov_b32_e32 v5, v6
	v_mad_u64_u32 v[6:7], s[20:21], v9, s10, 0
	v_mov_b32_e32 v8, v7
	v_mad_u64_u32 v[8:9], s[20:21], v9, s78, v[8:9]
	v_mov_b32_e32 v7, v8
	v_cmp_gt_u32_e64 s[8:9], s3, v68
	v_cmp_le_u32_e64 s[18:19], s3, v68
	v_lshlrev_b64 v[72:73], 1, v[2:3]
	v_lshlrev_b64 v[74:75], 1, v[4:5]
	v_lshlrev_b64 v[76:77], 1, v[6:7]
	v_mov_b32_e32 v89, v67
	v_mov_b32_e32 v88, v67
	v_mov_b32_e32 v87, v67
	v_mov_b32_e32 v86, v67
	v_mov_b32_e32 v85, v67
	v_mov_b32_e32 v84, v67
	v_mov_b32_e32 v83, v67
	v_mov_b32_e32 v71, v67
	v_mov_b32_e32 v69, v67
	v_mov_b32_e32 v90, v67
	v_mov_b32_e32 v91, v67
	v_mov_b32_e32 v92, v67
	v_mov_b32_e32 v93, v67
	v_mov_b32_e32 v94, v67
	v_mov_b32_e32 v95, v67
	s_mov_b32 s36, s37
	s_mov_b32 s70, s37
	s_branch .LBB0_10
.LBB0_7:                                ;   in Loop: Header=BB0_10 Depth=2
	s_or_b64 exec, exec, s[24:25]
.LBB0_8:                                ;   in Loop: Header=BB0_10 Depth=2
	s_or_b64 exec, exec, s[22:23]
	s_waitcnt vmcnt(5) lgkmcnt(4)
	;;#ASMSTART
	v_dot2c_f32_f16 v95, v62, v42
	;;#ASMEND
	s_waitcnt vmcnt(4)
	;;#ASMSTART
	v_dot2c_f32_f16 v94, v62, v38
	;;#ASMEND
	s_waitcnt vmcnt(3)
	;;#ASMSTART
	v_dot2c_f32_f16 v93, v62, v30
	;;#ASMEND
	s_waitcnt lgkmcnt(3)
	;;#ASMSTART
	v_dot2c_f32_f16 v92, v58, v42
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v91, v58, v38
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v90, v58, v30
	;;#ASMEND
	s_waitcnt lgkmcnt(2)
	;;#ASMSTART
	v_dot2c_f32_f16 v89, v54, v42
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v88, v54, v38
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v87, v54, v30
	;;#ASMEND
	s_waitcnt lgkmcnt(1)
	;;#ASMSTART
	v_dot2c_f32_f16 v86, v50, v42
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v85, v50, v38
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v84, v50, v30
	;;#ASMEND
	s_waitcnt lgkmcnt(0)
	;;#ASMSTART
	v_dot2c_f32_f16 v83, v46, v42
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v71, v46, v38
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v69, v46, v30
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v95, v63, v43
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v94, v63, v39
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v93, v63, v31
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v92, v59, v43
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v91, v59, v39
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v90, v59, v31
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v89, v55, v43
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v88, v55, v39
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v87, v55, v31
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v86, v51, v43
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v85, v51, v39
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v84, v51, v31
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v83, v47, v43
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v71, v47, v39
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v69, v47, v31
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v95, v64, v44
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v94, v64, v40
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v93, v64, v32
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v92, v60, v44
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v91, v60, v40
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v90, v60, v32
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v89, v56, v44
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v88, v56, v40
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v87, v56, v32
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v86, v52, v44
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v85, v52, v40
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v84, v52, v32
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v83, v48, v44
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v71, v48, v40
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v69, v48, v32
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v95, v65, v45
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v94, v65, v41
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v93, v65, v33
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v92, v61, v45
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v91, v61, v41
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v90, v61, v33
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v89, v57, v45
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v88, v57, v41
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v87, v57, v33
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v86, v53, v45
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v85, v53, v41
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v84, v53, v33
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v83, v49, v45
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v71, v49, v41
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v69, v49, v33
	;;#ASMEND
	s_waitcnt vmcnt(2)
	;;#ASMSTART
	v_dot2c_f32_f16 v95, v34, v10
	;;#ASMEND
	s_waitcnt vmcnt(1)
	;;#ASMSTART
	v_dot2c_f32_f16 v94, v34, v6
	;;#ASMEND
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	v_dot2c_f32_f16 v93, v34, v2
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v92, v26, v10
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v91, v26, v6
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v90, v26, v2
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v89, v22, v10
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v88, v22, v6
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v87, v22, v2
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v86, v18, v10
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v85, v18, v6
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v84, v18, v2
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v83, v14, v10
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v71, v14, v6
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v69, v14, v2
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v95, v35, v11
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v94, v35, v7
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v93, v35, v3
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v92, v27, v11
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v91, v27, v7
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v90, v27, v3
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v89, v23, v11
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v88, v23, v7
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v87, v23, v3
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v86, v19, v11
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v85, v19, v7
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v84, v19, v3
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v83, v15, v11
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v71, v15, v7
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v69, v15, v3
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v95, v36, v12
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v94, v36, v8
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v93, v36, v4
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v92, v28, v12
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v91, v28, v8
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v90, v28, v4
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v89, v24, v12
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v88, v24, v8
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v87, v24, v4
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v86, v20, v12
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v85, v20, v8
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v84, v20, v4
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v83, v16, v12
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v71, v16, v8
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v69, v16, v4
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v95, v37, v13
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v94, v37, v9
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v93, v37, v5
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v92, v29, v13
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v91, v29, v9
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v90, v29, v5
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v89, v25, v13
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v88, v25, v9
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v87, v25, v5
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v86, v21, v13
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v85, v21, v9
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v84, v21, v5
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v83, v17, v13
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v71, v17, v9
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v69, v17, v5
	;;#ASMEND
.LBB0_9:                                ;   in Loop: Header=BB0_10 Depth=2
	s_or_b64 exec, exec, s[20:21]
	s_addk_i32 s70, 0x400
	s_cmp_ge_u32 s70, s2
	s_cbranch_scc1 .LBB0_40
.LBB0_10:                               ;   Parent Loop BB0_6 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_15 Depth 3
	s_cmp_eq_u32 s70, 0
	s_cselect_b64 s[20:21], -1, 0
	s_add_i32 s22, s36, 0x4000
	s_cmp_eq_u32 s70, s22
	s_cselect_b64 s[24:25], -1, 0
	s_or_b64 s[24:25], s[20:21], s[24:25]
	s_andn2_b64 vcc, exec, s[24:25]
	s_cbranch_vccz .LBB0_12
; %bb.11:                               ;   in Loop: Header=BB0_10 Depth=2
	s_and_saveexec_b64 s[20:21], s[8:9]
	s_cbranch_execz .LBB0_9
	s_branch .LBB0_37
.LBB0_12:                               ;   in Loop: Header=BB0_10 Depth=2
	s_and_b64 s[20:21], s[20:21], exec
	s_cselect_b32 s36, s36, s22
	s_sub_i32 s20, s2, s36
	s_min_u32 s71, s20, 0x4000
	s_lshl_b64 s[20:21], s[36:37], 1
	s_add_u32 s22, s6, s20
	s_addc_u32 s23, s7, s21
	s_mov_b32 s72, 0
	s_mov_b64 s[24:25], 0
	v_mov_b32_e32 v4, v81
	s_barrier
                                        ; implicit-def: $sgpr64_sgpr65
	s_branch .LBB0_15
.LBB0_13:                               ;   in Loop: Header=BB0_15 Depth=3
	s_add_i32 s72, s72, s76
	s_cmpk_gt_u32 s72, 0x3fff
	s_cselect_b64 s[20:21], -1, 0
	s_andn2_b64 s[64:65], s[64:65], exec
	s_and_b64 s[20:21], s[20:21], exec
	v_add_u32_e32 v4, s80, v4
	s_or_b64 s[64:65], s[64:65], s[20:21]
.LBB0_14:                               ;   in Loop: Header=BB0_15 Depth=3
	s_or_b64 exec, exec, s[66:67]
	s_and_b64 s[20:21], exec, s[64:65]
	s_or_b64 s[24:25], s[20:21], s[24:25]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execz .LBB0_36
.LBB0_15:                               ;   Parent Loop BB0_6 Depth=1
                                        ;     Parent Loop BB0_10 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	v_add_u32_e32 v66, s72, v80
	v_cmp_gt_u32_e32 vcc, s71, v66
	s_or_b64 s[64:65], s[64:65], exec
	s_and_saveexec_b64 s[66:67], vcc
	s_cbranch_execz .LBB0_14
; %bb.16:                               ;   in Loop: Header=BB0_15 Depth=3
	v_lshl_add_u64 v[2:3], v[66:67], 1, s[22:23]
	s_mov_b64 s[20:21], -1
	s_and_b64 vcc, exec, s[40:41]
	s_cbranch_vccz .LBB0_18
; %bb.17:                               ;   in Loop: Header=BB0_15 Depth=3
	global_load_dwordx4 v[6:9], v[2:3], off
	s_mov_b64 s[20:21], 0
	s_waitcnt vmcnt(0)
	ds_write_b128 v4, v[6:9]
.LBB0_18:                               ;   in Loop: Header=BB0_15 Depth=3
	s_andn2_b64 vcc, exec, s[20:21]
	s_cbranch_vccnz .LBB0_20
; %bb.19:                               ;   in Loop: Header=BB0_15 Depth=3
	v_readfirstlane_b32 s20, v4
	s_mov_b32 m0, s20
	s_nop 0
	global_load_lds_dwordx4 v[2:3], off
.LBB0_20:                               ;   in Loop: Header=BB0_15 Depth=3
	v_lshl_add_u64 v[2:3], s[26:27], 1, v[2:3]
	v_add_u32_e32 v5, 0x8000, v4
	v_cmp_ne_u32_e64 s[20:21], 1, v82
	s_andn2_b64 vcc, exec, s[40:41]
	s_mov_b64 s[68:69], -1
	s_cbranch_vccnz .LBB0_22
; %bb.21:                               ;   in Loop: Header=BB0_15 Depth=3
	global_load_dwordx4 v[6:9], v[2:3], off
	s_mov_b64 s[68:69], 0
	s_waitcnt vmcnt(0)
	ds_write_b128 v5, v[6:9]
.LBB0_22:                               ;   in Loop: Header=BB0_15 Depth=3
	s_andn2_b64 vcc, exec, s[68:69]
	s_cbranch_vccnz .LBB0_24
; %bb.23:                               ;   in Loop: Header=BB0_15 Depth=3
	v_readfirstlane_b32 s68, v5
	s_mov_b32 m0, s68
	s_nop 0
	global_load_lds_dwordx4 v[2:3], off
.LBB0_24:                               ;   in Loop: Header=BB0_15 Depth=3
	v_lshl_add_u64 v[2:3], v[2:3], 0, s[62:63]
	v_add_u32_e32 v5, 0x10000, v4
	s_and_b64 vcc, exec, s[20:21]
	s_mov_b64 s[68:69], -1
	s_cbranch_vccnz .LBB0_26
; %bb.25:                               ;   in Loop: Header=BB0_15 Depth=3
	global_load_dwordx4 v[6:9], v[2:3], off
	s_mov_b64 s[68:69], 0
	s_waitcnt vmcnt(0)
	ds_write_b128 v5, v[6:9]
.LBB0_26:                               ;   in Loop: Header=BB0_15 Depth=3
	s_andn2_b64 vcc, exec, s[68:69]
	s_cbranch_vccnz .LBB0_28
; %bb.27:                               ;   in Loop: Header=BB0_15 Depth=3
	v_readfirstlane_b32 s68, v5
	s_mov_b32 m0, s68
	s_nop 0
	global_load_lds_dwordx4 v[2:3], off
.LBB0_28:                               ;   in Loop: Header=BB0_15 Depth=3
	v_lshl_add_u64 v[2:3], v[2:3], 0, s[62:63]
	v_add_u32_e32 v5, 0x18000, v4
	s_and_b64 vcc, exec, s[20:21]
	s_mov_b64 s[68:69], -1
	s_cbranch_vccnz .LBB0_30
; %bb.29:                               ;   in Loop: Header=BB0_15 Depth=3
	global_load_dwordx4 v[6:9], v[2:3], off
	s_mov_b64 s[68:69], 0
	s_waitcnt vmcnt(0)
	ds_write_b128 v5, v[6:9]
.LBB0_30:                               ;   in Loop: Header=BB0_15 Depth=3
	s_andn2_b64 vcc, exec, s[68:69]
	s_cbranch_vccnz .LBB0_32
; %bb.31:                               ;   in Loop: Header=BB0_15 Depth=3
	v_readfirstlane_b32 s68, v5
	s_mov_b32 m0, s68
	s_nop 0
	global_load_lds_dwordx4 v[2:3], off
.LBB0_32:                               ;   in Loop: Header=BB0_15 Depth=3
	v_lshl_add_u64 v[2:3], v[2:3], 0, s[62:63]
	v_add_u32_e32 v5, 0x20000, v4
	s_and_b64 vcc, exec, s[20:21]
	s_mov_b64 s[20:21], -1
	s_cbranch_vccnz .LBB0_34
; %bb.33:                               ;   in Loop: Header=BB0_15 Depth=3
	global_load_dwordx4 v[6:9], v[2:3], off
	s_mov_b64 s[20:21], 0
	s_waitcnt vmcnt(0)
	ds_write_b128 v5, v[6:9]
.LBB0_34:                               ;   in Loop: Header=BB0_15 Depth=3
	s_andn2_b64 vcc, exec, s[20:21]
	s_cbranch_vccnz .LBB0_13
; %bb.35:                               ;   in Loop: Header=BB0_15 Depth=3
	v_readfirstlane_b32 s20, v5
	s_mov_b32 m0, s20
	s_nop 0
	global_load_lds_dwordx4 v[2:3], off
	s_branch .LBB0_13
.LBB0_36:                               ;   in Loop: Header=BB0_10 Depth=2
	s_or_b64 exec, exec, s[24:25]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	s_and_saveexec_b64 s[20:21], s[8:9]
	s_cbranch_execz .LBB0_9
.LBB0_37:                               ;   in Loop: Header=BB0_10 Depth=2
	v_add_u32_e32 v96, s70, v79
	v_min_u32_e32 v66, s77, v96
	v_lshl_add_u64 v[2:3], v[66:67], 1, s[4:5]
	v_add_u32_e32 v97, 0x200, v96
	v_lshl_add_u64 v[4:5], v[2:3], 0, v[72:73]
	v_min_u32_e32 v66, s77, v97
	v_lshl_add_u64 v[6:7], v[2:3], 0, v[74:75]
	global_load_dwordx4 v[42:45], v[4:5], off nt
	global_load_dwordx4 v[38:41], v[6:7], off nt
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[76:77]
	v_lshl_add_u64 v[4:5], v[66:67], 1, s[4:5]
	v_lshl_add_u64 v[6:7], v[4:5], 0, v[72:73]
	global_load_dwordx4 v[30:33], v[2:3], off nt
	global_load_dwordx4 v[10:13], v[6:7], off nt
	v_lshl_add_u64 v[2:3], v[4:5], 0, v[74:75]
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[76:77]
	global_load_dwordx4 v[6:9], v[2:3], off nt
	s_nop 0
	global_load_dwordx4 v[2:5], v[4:5], off nt
	v_cmp_gt_u32_e32 vcc, s2, v96
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v63, 0
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v58, 0
	v_mov_b32_e32 v59, 0
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v56, 0
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v24, 0
	v_mov_b32_e32 v25, 0
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
; %bb.38:                               ;   in Loop: Header=BB0_10 Depth=2
	v_subrev_u32_e32 v14, s36, v96
	v_lshlrev_b32_e32 v66, 1, v14
	v_add_u32_e32 v14, 0x10000, v66
	ds_read_b128 v[62:65], v66
	ds_read_b128 v[58:61], v66 offset:32768
	v_add_u32_e32 v15, 0x18000, v66
	ds_read_b128 v[54:57], v14
	ds_read_b128 v[50:53], v15
	v_add_u32_e32 v14, 0x20000, v66
	ds_read_b128 v[46:49], v14
	v_cmp_gt_u32_e32 vcc, s2, v97
	v_mov_b32_e32 v17, 0
	v_mov_b32_e32 v16, 0
	v_mov_b32_e32 v15, 0
	v_mov_b32_e32 v14, 0
	v_mov_b32_e32 v21, 0
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v25, 0
	v_mov_b32_e32 v24, 0
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v34, 0
	s_and_saveexec_b64 s[24:25], vcc
	s_cbranch_execz .LBB0_7
; %bb.39:                               ;   in Loop: Header=BB0_10 Depth=2
	v_add_u32_e32 v14, 0x10400, v66
	ds_read_b128 v[34:37], v66 offset:1024
	ds_read_b128 v[26:29], v66 offset:33792
	v_add_u32_e32 v15, 0x18400, v66
	ds_read_b128 v[22:25], v14
	ds_read_b128 v[18:21], v15
	v_add_u32_e32 v14, 0x20400, v66
	ds_read_b128 v[14:17], v14
	s_branch .LBB0_7
.LBB0_40:                               ;   in Loop: Header=BB0_6 Depth=1
	v_add_u32_e32 v2, s79, v68
	s_and_saveexec_b64 s[8:9], s[18:19]
	s_xor_b64 s[8:9], exec, s[8:9]
; %bb.41:                               ;   in Loop: Header=BB0_6 Depth=1
                                        ; implicit-def: $vgpr68
                                        ; implicit-def: $vgpr70
; %bb.42:                               ;   in Loop: Header=BB0_6 Depth=1
	s_andn2_saveexec_b64 s[64:65], s[8:9]
	s_cbranch_execz .LBB0_5
; %bb.43:                               ;   in Loop: Header=BB0_6 Depth=1
	v_add_f32_dpp v13, v90, v90 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v95, v95 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v94, v94 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v13, v13, v13 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v93, v93 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v9, v92, v92 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v13, v13, v13 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v11, v91, v91 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v3, v3 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v13, v13, v13 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v5, v5 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v7, v7 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v17, v13, v13 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v13, v89, v89 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v9, v9, v9 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v11, v11, v11 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v13, v13, v13 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v3, v3 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v5, v5 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v13, v13, v13 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v7, v7 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v9, v9, v9 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v13, v13, v13 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v11, v11, v11 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v3, v3 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v20, v13, v13 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v13, v88, v88 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v5, v5 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v7, v7 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v13, v13, v13 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v9, v9, v9 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v11, v11, v11 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v13, v13, v13 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v3, v3 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v5, v5 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v13, v13, v13 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v7, v7 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v9, v9, v9 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v22, v13, v13 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v13, v87, v87 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v11, v11, v11 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v4, v3 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v13, v13, v13 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v6, v5 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v8, v7 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v13, v13, v13 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v10, v9 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v12, v11 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v13, v13, v13 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v19, v17 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v21, v20 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v24, v13, v13 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v13, v86, v86 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v23, v22 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v25, v24 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v13, v13, v13 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v13, v13, v13 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v13, v13, v13 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v26, v13, v13 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v13, v85, v85 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v27, v26 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v13, v13, v13 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v13, v13, v13 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v13, v13, v13 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v28, v13, v13 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v13, v84, v84 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v29, v28 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v13, v13, v13 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v13, v13, v13 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v13, v13, v13 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v30, v13, v13 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v13, v83, v83 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v31, v30 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v13, v13, v13 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v13, v13, v13 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v13, v13, v13 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v32, v13, v13 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v13, v71, v71 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v33, v32 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v13, v13, v13 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v13, v13, v13 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v13, v13, v13 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v34, v13, v13 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v13, v69, v69 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v35, v34 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v13, v13, v13 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v13, v13, v13 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v13, v13, v13 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v36, v13, v13 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_mov_b32_dpp v37, v36 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_and_saveexec_b64 s[66:67], s[0:1]
	s_cbranch_execz .LBB0_4
; %bb.44:                               ;   in Loop: Header=BB0_6 Depth=1
	v_cmp_eq_u32_e64 s[8:9], 0, v1
	s_and_b64 s[18:19], s[42:43], s[8:9]
	v_add_f32_e32 v18, v3, v4
	v_add_f32_e32 v16, v5, v6
	v_add_f32_e32 v15, v7, v8
	v_add_f32_e32 v14, v9, v10
	v_add_f32_e32 v13, v11, v12
	v_add_f32_e32 v12, v17, v19
	v_add_f32_e32 v11, v20, v21
	v_add_f32_e32 v10, v22, v23
	v_add_f32_e32 v9, v24, v25
	v_add_f32_e32 v8, v26, v27
	v_add_f32_e32 v7, v28, v29
	v_add_f32_e32 v6, v30, v31
	v_add_f32_e32 v5, v32, v33
	v_add_f32_e32 v4, v34, v35
	v_add_f32_e32 v3, v36, v37
	s_xor_b64 s[20:21], s[18:19], -1
	v_mov_b32_e32 v71, v67
	s_mov_b64 s[18:19], 0
                                        ; implicit-def: $vgpr17
	s_and_saveexec_b64 s[22:23], s[20:21]
	s_xor_b64 s[68:69], exec, s[22:23]
	s_cbranch_execnz .LBB0_47
; %bb.45:                               ;   in Loop: Header=BB0_6 Depth=1
	s_andn2_saveexec_b64 s[8:9], s[68:69]
	s_cbranch_execnz .LBB0_118
.LBB0_46:                               ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[8:9]
	s_and_b64 exec, exec, s[18:19]
	s_cbranch_execz .LBB0_4
	s_branch .LBB0_119
.LBB0_47:                               ;   in Loop: Header=BB0_6 Depth=1
	v_cndmask_b32_e64 v17, 0, 1, s[44:45]
	v_cmp_ne_u32_e64 s[22:23], 1, v17
	s_and_saveexec_b64 s[18:19], s[8:9]
	s_cbranch_execz .LBB0_51
; %bb.48:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v17, s28, v18
	s_and_b64 vcc, exec, s[22:23]
	v_mov_b32_e32 v69, v67
	s_cbranch_vccnz .LBB0_50
; %bb.49:                               ;   in Loop: Header=BB0_6 Depth=1
	v_lshl_add_u64 v[18:19], v[68:69], 1, s[12:13]
	global_load_ushort v18, v[18:19], off
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v17, s29, v18, v17 op_sel_hi:[0,1,0]
.LBB0_50:                               ;   in Loop: Header=BB0_6 Depth=1
	v_cvt_f16_f32_e32 v17, v17
	v_lshl_add_u64 v[18:19], v[68:69], 1, s[14:15]
	global_store_short v[18:19], v17, off
.LBB0_51:                               ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[18:19]
	v_cndmask_b32_e64 v17, 0, 1, s[42:43]
	v_cmp_gt_u32_e64 s[18:19], 2, v1
	v_cmp_ne_u32_e64 s[20:21], 1, v17
	s_and_saveexec_b64 s[24:25], s[18:19]
	s_cbranch_execnz .LBB0_64
; %bb.52:                               ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[24:25]
	v_cmp_gt_u32_e64 s[24:25], 3, v1
	s_and_saveexec_b64 s[70:71], s[24:25]
	s_cbranch_execnz .LBB0_69
.LBB0_53:                               ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[8:9]
	s_cbranch_execnz .LBB0_72
.LBB0_54:                               ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[18:19]
	s_cbranch_execnz .LBB0_75
.LBB0_55:                               ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[24:25]
	s_cbranch_execnz .LBB0_80
.LBB0_56:                               ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[8:9]
	s_cbranch_execnz .LBB0_83
.LBB0_57:                               ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[18:19]
	s_cbranch_execnz .LBB0_86
.LBB0_58:                               ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[24:25]
	s_cbranch_execnz .LBB0_91
.LBB0_59:                               ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[8:9]
	s_cbranch_execnz .LBB0_94
.LBB0_60:                               ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[18:19]
	s_cbranch_execnz .LBB0_97
.LBB0_61:                               ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[24:25]
	s_cbranch_execnz .LBB0_102
.LBB0_62:                               ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[8:9]
	s_cbranch_execnz .LBB0_105
.LBB0_63:                               ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[8:9], s[18:19]
	s_cbranch_execnz .LBB0_108
	s_branch .LBB0_113
.LBB0_64:                               ;   in Loop: Header=BB0_6 Depth=1
	s_and_b64 vcc, exec, s[20:21]
	v_mov_b32_e32 v69, v67
	s_cbranch_vccnz .LBB0_66
; %bb.65:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v16, s28, v16
	s_cbranch_execz .LBB0_67
	s_branch .LBB0_68
.LBB0_66:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v16, s28, v16
.LBB0_67:                               ;   in Loop: Header=BB0_6 Depth=1
	v_lshl_add_u64 v[18:19], v[68:69], 1, s[12:13]
	global_load_ushort v17, v[18:19], off offset:2
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v16, s29, v17, v16 op_sel_hi:[0,1,0]
.LBB0_68:                               ;   in Loop: Header=BB0_6 Depth=1
	v_cvt_f16_f32_e32 v18, v16
	v_lshl_add_u64 v[16:17], v[68:69], 1, s[14:15]
	global_store_short v[16:17], v18, off offset:2
	s_or_b64 exec, exec, s[24:25]
	v_cmp_gt_u32_e64 s[24:25], 3, v1
	s_and_saveexec_b64 s[70:71], s[24:25]
	s_cbranch_execz .LBB0_53
.LBB0_69:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v15, s28, v15
	s_and_b64 vcc, exec, s[22:23]
	v_mov_b32_e32 v71, v67
	s_cbranch_vccnz .LBB0_71
; %bb.70:                               ;   in Loop: Header=BB0_6 Depth=1
	v_lshl_add_u64 v[16:17], v[70:71], 1, s[12:13]
	global_load_ushort v16, v[16:17], off
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v15, s29, v16, v15 op_sel_hi:[0,1,0]
.LBB0_71:                               ;   in Loop: Header=BB0_6 Depth=1
	v_cvt_f16_f32_e32 v15, v15
	v_lshl_add_u64 v[16:17], v[70:71], 1, s[14:15]
	global_store_short v[16:17], v15, off
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[8:9]
	s_cbranch_execz .LBB0_54
.LBB0_72:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v14, s28, v14
	s_and_b64 vcc, exec, s[22:23]
	v_mov_b32_e32 v69, v67
	s_cbranch_vccnz .LBB0_74
; %bb.73:                               ;   in Loop: Header=BB0_6 Depth=1
	v_lshl_add_u64 v[16:17], v[68:69], 1, s[46:47]
	global_load_ushort v15, v[16:17], off
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v14, s29, v15, v14 op_sel_hi:[0,1,0]
.LBB0_74:                               ;   in Loop: Header=BB0_6 Depth=1
	v_cvt_f16_f32_e32 v16, v14
	v_lshl_add_u64 v[14:15], v[68:69], 1, s[48:49]
	global_store_short v[14:15], v16, off
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[18:19]
	s_cbranch_execz .LBB0_55
.LBB0_75:                               ;   in Loop: Header=BB0_6 Depth=1
	s_and_b64 vcc, exec, s[20:21]
	v_mov_b32_e32 v69, v67
	s_cbranch_vccnz .LBB0_77
; %bb.76:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v13, s28, v13
	s_cbranch_execz .LBB0_78
	s_branch .LBB0_79
.LBB0_77:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v13, s28, v13
.LBB0_78:                               ;   in Loop: Header=BB0_6 Depth=1
	v_lshl_add_u64 v[14:15], v[68:69], 1, s[46:47]
	global_load_ushort v14, v[14:15], off offset:2
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v13, s29, v14, v13 op_sel_hi:[0,1,0]
.LBB0_79:                               ;   in Loop: Header=BB0_6 Depth=1
	v_cvt_f16_f32_e32 v13, v13
	v_lshl_add_u64 v[14:15], v[68:69], 1, s[48:49]
	global_store_short v[14:15], v13, off offset:2
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[24:25]
	s_cbranch_execz .LBB0_56
.LBB0_80:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v12, s28, v12
	s_and_b64 vcc, exec, s[22:23]
	v_mov_b32_e32 v71, v67
	s_cbranch_vccnz .LBB0_82
; %bb.81:                               ;   in Loop: Header=BB0_6 Depth=1
	v_lshl_add_u64 v[14:15], v[70:71], 1, s[46:47]
	global_load_ushort v13, v[14:15], off
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v12, s29, v13, v12 op_sel_hi:[0,1,0]
.LBB0_82:                               ;   in Loop: Header=BB0_6 Depth=1
	v_cvt_f16_f32_e32 v14, v12
	v_lshl_add_u64 v[12:13], v[70:71], 1, s[48:49]
	global_store_short v[12:13], v14, off
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[8:9]
	s_cbranch_execz .LBB0_57
.LBB0_83:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v11, s28, v11
	s_and_b64 vcc, exec, s[22:23]
	v_mov_b32_e32 v69, v67
	s_cbranch_vccnz .LBB0_85
; %bb.84:                               ;   in Loop: Header=BB0_6 Depth=1
	v_lshl_add_u64 v[12:13], v[68:69], 1, s[50:51]
	global_load_ushort v12, v[12:13], off
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v11, s29, v12, v11 op_sel_hi:[0,1,0]
.LBB0_85:                               ;   in Loop: Header=BB0_6 Depth=1
	v_cvt_f16_f32_e32 v11, v11
	v_lshl_add_u64 v[12:13], v[68:69], 1, s[52:53]
	global_store_short v[12:13], v11, off
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[18:19]
	s_cbranch_execz .LBB0_58
.LBB0_86:                               ;   in Loop: Header=BB0_6 Depth=1
	s_and_b64 vcc, exec, s[20:21]
	v_mov_b32_e32 v69, v67
	s_cbranch_vccnz .LBB0_88
; %bb.87:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v10, s28, v10
	s_cbranch_execz .LBB0_89
	s_branch .LBB0_90
.LBB0_88:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v10, s28, v10
.LBB0_89:                               ;   in Loop: Header=BB0_6 Depth=1
	v_lshl_add_u64 v[12:13], v[68:69], 1, s[50:51]
	global_load_ushort v11, v[12:13], off offset:2
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v10, s29, v11, v10 op_sel_hi:[0,1,0]
.LBB0_90:                               ;   in Loop: Header=BB0_6 Depth=1
	v_cvt_f16_f32_e32 v12, v10
	v_lshl_add_u64 v[10:11], v[68:69], 1, s[52:53]
	global_store_short v[10:11], v12, off offset:2
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[24:25]
	s_cbranch_execz .LBB0_59
.LBB0_91:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v9, s28, v9
	s_and_b64 vcc, exec, s[22:23]
	v_mov_b32_e32 v71, v67
	s_cbranch_vccnz .LBB0_93
; %bb.92:                               ;   in Loop: Header=BB0_6 Depth=1
	v_lshl_add_u64 v[10:11], v[70:71], 1, s[50:51]
	global_load_ushort v10, v[10:11], off
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v9, s29, v10, v9 op_sel_hi:[0,1,0]
.LBB0_93:                               ;   in Loop: Header=BB0_6 Depth=1
	v_cvt_f16_f32_e32 v9, v9
	v_lshl_add_u64 v[10:11], v[70:71], 1, s[52:53]
	global_store_short v[10:11], v9, off
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[8:9]
	s_cbranch_execz .LBB0_60
.LBB0_94:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v8, s28, v8
	s_and_b64 vcc, exec, s[22:23]
	v_mov_b32_e32 v69, v67
	s_cbranch_vccnz .LBB0_96
; %bb.95:                               ;   in Loop: Header=BB0_6 Depth=1
	v_lshl_add_u64 v[10:11], v[68:69], 1, s[54:55]
	global_load_ushort v9, v[10:11], off
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v8, s29, v9, v8 op_sel_hi:[0,1,0]
.LBB0_96:                               ;   in Loop: Header=BB0_6 Depth=1
	v_cvt_f16_f32_e32 v10, v8
	v_lshl_add_u64 v[8:9], v[68:69], 1, s[56:57]
	global_store_short v[8:9], v10, off
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[18:19]
	s_cbranch_execz .LBB0_61
.LBB0_97:                               ;   in Loop: Header=BB0_6 Depth=1
	s_and_b64 vcc, exec, s[20:21]
	v_mov_b32_e32 v69, v67
	s_cbranch_vccnz .LBB0_99
; %bb.98:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v7, s28, v7
	s_cbranch_execz .LBB0_100
	s_branch .LBB0_101
.LBB0_99:                               ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v7, s28, v7
.LBB0_100:                              ;   in Loop: Header=BB0_6 Depth=1
	v_lshl_add_u64 v[8:9], v[68:69], 1, s[54:55]
	global_load_ushort v8, v[8:9], off offset:2
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v7, s29, v8, v7 op_sel_hi:[0,1,0]
.LBB0_101:                              ;   in Loop: Header=BB0_6 Depth=1
	v_cvt_f16_f32_e32 v7, v7
	v_lshl_add_u64 v[8:9], v[68:69], 1, s[56:57]
	global_store_short v[8:9], v7, off offset:2
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[24:25]
	s_cbranch_execz .LBB0_62
.LBB0_102:                              ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v6, s28, v6
	s_and_b64 vcc, exec, s[22:23]
	v_mov_b32_e32 v71, v67
	s_cbranch_vccnz .LBB0_104
; %bb.103:                              ;   in Loop: Header=BB0_6 Depth=1
	v_lshl_add_u64 v[8:9], v[70:71], 1, s[54:55]
	global_load_ushort v7, v[8:9], off
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v6, s29, v7, v6 op_sel_hi:[0,1,0]
.LBB0_104:                              ;   in Loop: Header=BB0_6 Depth=1
	v_cvt_f16_f32_e32 v8, v6
	v_lshl_add_u64 v[6:7], v[70:71], 1, s[56:57]
	global_store_short v[6:7], v8, off
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[70:71], s[8:9]
	s_cbranch_execz .LBB0_63
.LBB0_105:                              ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v5, s28, v5
	s_and_b64 vcc, exec, s[22:23]
	v_mov_b32_e32 v69, v67
	s_cbranch_vccnz .LBB0_107
; %bb.106:                              ;   in Loop: Header=BB0_6 Depth=1
	v_lshl_add_u64 v[6:7], v[68:69], 1, s[58:59]
	global_load_ushort v6, v[6:7], off
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v5, s29, v6, v5 op_sel_hi:[0,1,0]
.LBB0_107:                              ;   in Loop: Header=BB0_6 Depth=1
	v_cvt_f16_f32_e32 v5, v5
	v_lshl_add_u64 v[6:7], v[68:69], 1, s[60:61]
	global_store_short v[6:7], v5, off
	s_or_b64 exec, exec, s[70:71]
	s_and_saveexec_b64 s[8:9], s[18:19]
	s_cbranch_execz .LBB0_113
.LBB0_108:                              ;   in Loop: Header=BB0_6 Depth=1
	s_and_b64 vcc, exec, s[20:21]
	v_mov_b32_e32 v69, v67
	s_cbranch_vccnz .LBB0_110
; %bb.109:                              ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v4, s28, v4
	s_cbranch_execz .LBB0_111
	s_branch .LBB0_112
.LBB0_110:                              ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v4, s28, v4
.LBB0_111:                              ;   in Loop: Header=BB0_6 Depth=1
	v_lshl_add_u64 v[6:7], v[68:69], 1, s[58:59]
	global_load_ushort v5, v[6:7], off offset:2
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v4, s29, v5, v4 op_sel_hi:[0,1,0]
.LBB0_112:                              ;   in Loop: Header=BB0_6 Depth=1
	v_cvt_f16_f32_e32 v6, v4
	v_lshl_add_u64 v[4:5], v[68:69], 1, s[60:61]
	global_store_short v[4:5], v6, off offset:2
.LBB0_113:                              ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[8:9]
	s_mov_b64 s[18:19], 0
                                        ; implicit-def: $vgpr17
	s_and_saveexec_b64 s[8:9], s[24:25]
	s_xor_b64 s[8:9], exec, s[8:9]
	s_cbranch_execz .LBB0_117
; %bb.114:                              ;   in Loop: Header=BB0_6 Depth=1
	v_mul_f32_e32 v17, s28, v3
	s_and_b64 vcc, exec, s[22:23]
	v_mov_b32_e32 v71, v67
	s_cbranch_vccnz .LBB0_116
; %bb.115:                              ;   in Loop: Header=BB0_6 Depth=1
	v_lshl_add_u64 v[4:5], v[70:71], 1, s[58:59]
	global_load_ushort v3, v[4:5], off
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v17, s29, v3, v17 op_sel_hi:[0,1,0]
.LBB0_116:                              ;   in Loop: Header=BB0_6 Depth=1
	s_mov_b64 s[18:19], exec
.LBB0_117:                              ;   in Loop: Header=BB0_6 Depth=1
	s_or_b64 exec, exec, s[8:9]
	s_and_b64 s[18:19], s[18:19], exec
                                        ; implicit-def: $vgpr18
                                        ; implicit-def: $vgpr16
                                        ; implicit-def: $vgpr15
                                        ; implicit-def: $vgpr14
                                        ; implicit-def: $vgpr13
                                        ; implicit-def: $vgpr12
                                        ; implicit-def: $vgpr11
                                        ; implicit-def: $vgpr10
                                        ; implicit-def: $vgpr9
                                        ; implicit-def: $vgpr8
                                        ; implicit-def: $vgpr7
                                        ; implicit-def: $vgpr6
                                        ; implicit-def: $vgpr5
                                        ; implicit-def: $vgpr4
                                        ; implicit-def: $vgpr3
                                        ; implicit-def: $vgpr68
	s_andn2_saveexec_b64 s[8:9], s[68:69]
	s_cbranch_execz .LBB0_46
.LBB0_118:                              ;   in Loop: Header=BB0_6 Depth=1
	v_mov_b32_e32 v69, v67
	v_mul_f32_e32 v17, s28, v18
	v_lshlrev_b64 v[18:19], 1, v[68:69]
	v_mul_f32_e32 v16, s28, v16
	v_lshl_add_u64 v[20:21], s[14:15], 0, v[18:19]
	v_cvt_pk_f16_f32 v16, v17, v16
	global_store_dword v[20:21], v16, off
	v_lshlrev_b64 v[16:17], 1, v[70:71]
	v_fma_mixlo_f16 v15, s28, v15, 0
	v_lshl_add_u64 v[20:21], s[14:15], 0, v[16:17]
	global_store_short v[20:21], v15, off
	v_mul_f32_e32 v20, s28, v14
	v_mul_f32_e32 v13, s28, v13
	v_lshl_add_u64 v[14:15], s[48:49], 0, v[18:19]
	v_cvt_pk_f16_f32 v13, v20, v13
	global_store_dword v[14:15], v13, off
	v_fma_mixlo_f16 v14, s28, v12, 0
	v_lshl_add_u64 v[12:13], s[48:49], 0, v[16:17]
	v_mul_f32_e32 v11, s28, v11
	v_mul_f32_e32 v10, s28, v10
	global_store_short v[12:13], v14, off
	v_lshl_add_u64 v[12:13], s[52:53], 0, v[18:19]
	v_cvt_pk_f16_f32 v10, v11, v10
	global_store_dword v[12:13], v10, off
	v_fma_mixlo_f16 v9, s28, v9, 0
	v_lshl_add_u64 v[10:11], s[52:53], 0, v[16:17]
	global_store_short v[10:11], v9, off
	v_mul_f32_e32 v10, s28, v8
	v_mul_f32_e32 v7, s28, v7
	v_lshl_add_u64 v[8:9], s[56:57], 0, v[18:19]
	v_cvt_pk_f16_f32 v7, v10, v7
	global_store_dword v[8:9], v7, off
	v_fma_mixlo_f16 v8, s28, v6, 0
	v_lshl_add_u64 v[6:7], s[56:57], 0, v[16:17]
	v_mul_f32_e32 v5, s28, v5
	v_mul_f32_e32 v4, s28, v4
	global_store_short v[6:7], v8, off
	v_lshl_add_u64 v[6:7], s[60:61], 0, v[18:19]
	v_cvt_pk_f16_f32 v4, v5, v4
	global_store_dword v[6:7], v4, off
	v_mul_f32_e32 v17, s28, v3
	s_or_b64 s[18:19], s[18:19], exec
	s_or_b64 exec, exec, s[8:9]
	s_and_b64 exec, exec, s[18:19]
	s_cbranch_execz .LBB0_4
.LBB0_119:                              ;   in Loop: Header=BB0_6 Depth=1
	v_cvt_f16_f32_e32 v3, v17
	v_lshl_add_u64 v[4:5], v[70:71], 1, s[60:61]
	global_store_short v[4:5], v3, off
	s_branch .LBB0_4
.LBB0_120:
	s_or_b64 exec, exec, s[34:35]
	s_mov_b64 s[0:1], 0
.LBB0_121:
	s_andn2_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB0_217
; %bb.122:
	v_and_b32_e32 v8, 0x3ff, v0
	s_mul_i32 s22, s2, 5
	v_lshlrev_b32_e32 v74, 3, v8
	s_cmp_eq_u32 s26, s2
	s_mov_b64 s[8:9], 0
	s_cbranch_scc1 .LBB0_124
; %bb.123:
	s_and_b32 s18, s6, 15
	s_mov_b32 s19, 0
	s_mov_b64 s[0:1], -1
	s_min_u32 s22, s22, 0x14000
	v_lshl_add_u32 v9, v78, 9, v74
	s_cbranch_execz .LBB0_125
	s_branch .LBB0_126
.LBB0_124:
                                        ; implicit-def: $sgpr18_sgpr19
	s_mov_b64 s[0:1], 0
	s_min_u32 s22, s22, 0x14000
	v_lshl_add_u32 v9, v78, 9, v74
.LBB0_125:
	s_and_b32 s0, s2, 7
	s_cmp_lg_u32 s0, 0
	s_mov_b32 s19, 0
	s_cselect_b64 s[0:1], -1, 0
	s_and_b32 s18, s6, 15
	s_cmp_lg_u64 s[18:19], 0
	s_cselect_b64 s[8:9], -1, 0
	s_or_b64 s[0:1], s[0:1], s[8:9]
	s_mov_b64 s[8:9], -1
.LBB0_126:
	s_andn2_b64 vcc, exec, s[0:1]
	v_cmp_gt_u32_e64 s[0:1], s22, v9
	s_cbranch_vccz .LBB0_128
; %bb.127:
	s_andn2_b64 vcc, exec, s[8:9]
	s_cbranch_vccz .LBB0_136
	s_branch .LBB0_140
.LBB0_128:
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_135
; %bb.129:
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
	v_lshl_add_u32 v10, v78, 10, v0
	v_mov_b32_e32 v0, v9
	s_branch .LBB0_131
.LBB0_130:                              ;   in Loop: Header=BB0_131 Depth=1
	v_add_u32_e32 v0, 0x2000, v0
	v_cmp_le_u32_e32 vcc, s22, v0
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[4:5]
	s_or_b64 s[0:1], vcc, s[0:1]
	v_add_u32_e32 v10, 0x4000, v10
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB0_135
.LBB0_131:                              ; =>This Inner Loop Header: Depth=1
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
	s_cbranch_vccz .LBB0_133
; %bb.132:                              ;   in Loop: Header=BB0_131 Depth=1
	global_load_dwordx4 v[12:15], v[6:7], off
	s_mov_b64 s[20:21], 0
	s_waitcnt vmcnt(0)
	ds_write_b128 v10, v[12:15]
.LBB0_133:                              ;   in Loop: Header=BB0_131 Depth=1
	s_andn2_b64 vcc, exec, s[20:21]
	s_cbranch_vccnz .LBB0_130
; %bb.134:                              ;   in Loop: Header=BB0_131 Depth=1
	v_readfirstlane_b32 s20, v10
	s_mov_b32 m0, s20
	s_nop 0
	global_load_lds_dwordx4 v[6:7], off
	s_branch .LBB0_130
.LBB0_135:
	s_or_b64 exec, exec, s[8:9]
	s_cbranch_execnz .LBB0_140
.LBB0_136:
	v_cmp_gt_u32_e32 vcc, s22, v9
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_139
; %bb.137:
	v_lshlrev_b32_e32 v0, 4, v8
	v_lshl_add_u32 v2, v78, 10, v0
	v_lshlrev_b32_e32 v0, 1, v9
	v_mov_b32_e32 v1, 0
	v_lshl_add_u64 v[0:1], s[6:7], 0, v[0:1]
	s_mov_b64 s[6:7], 0
	s_mov_b64 s[8:9], 0x4000
.LBB0_138:                              ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s18, v2
	s_mov_b32 m0, s18
	v_add_u32_e32 v9, 0x2000, v9
	global_load_lds_dwordx4 v[0:1], off
	v_cmp_le_u32_e32 vcc, s22, v9
	v_add_u32_e32 v2, 0x4000, v2
	s_or_b64 s[6:7], vcc, s[6:7]
	v_lshl_add_u64 v[0:1], v[0:1], 0, s[8:9]
	s_andn2_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB0_138
.LBB0_139:
	s_or_b64 exec, exec, s[0:1]
.LBB0_140:
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
	v_add_u32_e32 v0, s0, v78
	v_lshl_add_u32 v1, v0, 1, v0
	v_add_u32_e32 v0, 3, v1
	v_cmp_gt_u32_e32 vcc, s3, v1
	v_cmp_lt_u32_e64 s[0:1], s3, v0
	s_add_i32 s33, s3, -3
	v_mov_b32_e32 v2, s33
	s_and_b64 s[0:1], vcc, s[0:1]
	v_cndmask_b32_e64 v64, v1, v2, s[0:1]
	v_cmp_gt_u32_e64 s[0:1], s8, v78
	v_cmp_gt_u32_e64 s[6:7], s3, v64
	s_and_b64 s[0:1], s[0:1], s[6:7]
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_217
; %bb.141:
	s_cmp_lg_u32 s2, 0
	s_cselect_b64 s[6:7], -1, 0
	s_ashr_i32 s17, s30, 31
	s_mov_b32 s16, s30
	s_add_i32 s54, s2, -8
	s_add_i32 s55, s3, -1
	s_ashr_i32 s56, s10, 31
	s_ashr_i32 s19, s31, 31
	s_lshl_b64 s[16:17], s[16:17], 1
	s_mov_b32 s18, s31
	s_add_u32 s26, s12, s16
	s_addc_u32 s27, s13, s17
	s_lshl_b64 s[18:19], s[18:19], 1
	s_add_u32 s30, s14, s18
	s_addc_u32 s31, s15, s19
	s_add_u32 s34, s26, s16
	s_addc_u32 s35, s27, s17
	s_add_u32 s36, s30, s18
	s_addc_u32 s37, s31, s19
	s_add_u32 s38, s34, s16
	s_addc_u32 s39, s35, s17
	s_add_u32 s40, s36, s18
	s_addc_u32 s41, s37, s19
	s_add_u32 s42, s38, s16
	v_sub_u32_e64 v0, v0, s3 clamp
	s_addc_u32 s43, s39, s17
	v_cndmask_b32_e32 v75, 0, v0, vcc
	s_add_u32 s44, s40, s18
	v_cndmask_b32_e64 v0, 0, 1, s[6:7]
	v_cmp_eq_u32_e64 s[0:1], 63, v8
	v_cmp_eq_f32_e64 s[22:23], s29, 0
	v_cmp_neq_f32_e64 s[24:25], s29, 0
	s_addc_u32 s45, s41, s19
	s_mul_i32 s11, s8, s11
	s_lshl_b32 s57, s2, 1
	v_lshlrev_b32_e32 v76, 4, v8
	s_lshl_b32 s58, s2, 2
	s_mul_i32 s59, s2, 6
	s_lshl_b32 s60, s2, 3
	s_mov_b64 s[46:47], 0
	v_cmp_ne_u32_e64 s[20:21], 1, v0
	v_mov_b32_e32 v67, 0
	s_branch .LBB0_143
.LBB0_142:                              ;   in Loop: Header=BB0_143 Depth=1
	s_or_b64 exec, exec, s[48:49]
	v_add_u32_e32 v0, s11, v64
	v_add_u32_e32 v1, 3, v0
	v_cmp_gt_u32_e32 vcc, s3, v0
	v_cmp_lt_u32_e64 s[6:7], s3, v1
	v_mov_b32_e32 v2, s33
	s_and_b64 vcc, vcc, s[6:7]
	v_cndmask_b32_e32 v64, v0, v2, vcc
	v_subrev_u32_e32 v1, s3, v1
	v_cmp_le_u32_e64 s[6:7], s3, v64
	s_or_b64 s[46:47], s[6:7], s[46:47]
	v_cndmask_b32_e32 v75, v75, v1, vcc
	s_andn2_b64 exec, exec, s[46:47]
	s_cbranch_execz .LBB0_217
.LBB0_143:                              ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_147 Depth 2
	s_and_b64 vcc, exec, s[20:21]
	v_mov_b32_e32 v90, v67
	v_mov_b32_e32 v89, v67
	v_mov_b32_e32 v88, v67
	v_mov_b32_e32 v87, v67
	v_mov_b32_e32 v86, v67
	v_mov_b32_e32 v85, v67
	v_mov_b32_e32 v65, v67
	v_mov_b32_e32 v77, v67
	v_mov_b32_e32 v78, v67
	v_mov_b32_e32 v79, v67
	v_mov_b32_e32 v80, v67
	v_mov_b32_e32 v81, v67
	v_mov_b32_e32 v82, v67
	v_mov_b32_e32 v83, v67
	v_mov_b32_e32 v84, v67
	s_cbranch_vccnz .LBB0_150
; %bb.144:                              ;   in Loop: Header=BB0_143 Depth=1
	v_min_u32_e32 v3, s55, v64
	v_mad_u64_u32 v[0:1], s[6:7], v3, s10, 0
	v_mov_b32_e32 v2, v1
	v_mad_u64_u32 v[2:3], s[6:7], v3, s56, v[2:3]
	v_mov_b32_e32 v1, v2
	v_add_u32_e32 v2, 1, v64
	v_min_u32_e32 v5, s55, v2
	v_mad_u64_u32 v[2:3], s[6:7], v5, s10, 0
	v_mov_b32_e32 v4, v3
	v_mad_u64_u32 v[4:5], s[6:7], v5, s56, v[4:5]
	v_mov_b32_e32 v3, v4
	v_add_u32_e32 v4, 2, v64
	v_min_u32_e32 v7, s55, v4
	v_mad_u64_u32 v[4:5], s[6:7], v7, s10, 0
	v_mov_b32_e32 v6, v5
	v_mad_u64_u32 v[6:7], s[6:7], v7, s56, v[6:7]
	v_mov_b32_e32 v5, v6
	s_mov_b32 s16, 0
	v_mov_b32_e32 v84, 0
	v_lshlrev_b64 v[68:69], 1, v[0:1]
	v_lshlrev_b64 v[70:71], 1, v[2:3]
	v_lshlrev_b64 v[72:73], 1, v[4:5]
	v_mov_b32_e32 v91, v76
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v79, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v85, 0
	v_mov_b32_e32 v86, 0
	v_mov_b32_e32 v87, 0
	v_mov_b32_e32 v88, 0
	v_mov_b32_e32 v89, 0
	v_mov_b32_e32 v90, 0
	s_branch .LBB0_147
.LBB0_145:                              ;   in Loop: Header=BB0_147 Depth=2
	s_or_b64 exec, exec, s[8:9]
.LBB0_146:                              ;   in Loop: Header=BB0_147 Depth=2
	s_or_b64 exec, exec, s[6:7]
	s_waitcnt vmcnt(5) lgkmcnt(4)
	;;#ASMSTART
	v_dot2c_f32_f16 v90, v60, v36
	;;#ASMEND
	s_waitcnt vmcnt(4)
	;;#ASMSTART
	v_dot2c_f32_f16 v89, v60, v32
	;;#ASMEND
	s_waitcnt vmcnt(3)
	;;#ASMSTART
	v_dot2c_f32_f16 v88, v60, v28
	;;#ASMEND
	s_waitcnt lgkmcnt(3)
	;;#ASMSTART
	v_dot2c_f32_f16 v87, v56, v36
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v86, v56, v32
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v85, v56, v28
	;;#ASMEND
	s_waitcnt lgkmcnt(2)
	;;#ASMSTART
	v_dot2c_f32_f16 v84, v52, v36
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v83, v52, v32
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v82, v52, v28
	;;#ASMEND
	s_waitcnt lgkmcnt(1)
	;;#ASMSTART
	v_dot2c_f32_f16 v81, v48, v36
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v80, v48, v32
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v79, v48, v28
	;;#ASMEND
	s_waitcnt lgkmcnt(0)
	;;#ASMSTART
	v_dot2c_f32_f16 v78, v44, v36
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v77, v44, v32
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v65, v44, v28
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v90, v61, v37
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v89, v61, v33
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v88, v61, v29
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v87, v57, v37
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v86, v57, v33
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v85, v57, v29
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v84, v53, v37
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v83, v53, v33
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v82, v53, v29
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v81, v49, v37
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v80, v49, v33
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v79, v49, v29
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v78, v45, v37
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v77, v45, v33
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v65, v45, v29
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v90, v62, v38
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v89, v62, v34
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v88, v62, v30
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v87, v58, v38
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v86, v58, v34
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v85, v58, v30
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v84, v54, v38
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v83, v54, v34
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v82, v54, v30
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v81, v50, v38
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v80, v50, v34
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v79, v50, v30
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v78, v46, v38
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v77, v46, v34
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v65, v46, v30
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v90, v63, v39
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v89, v63, v35
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v88, v63, v31
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v87, v59, v39
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v86, v59, v35
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v85, v59, v31
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v84, v55, v39
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v83, v55, v35
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v82, v55, v31
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v81, v51, v39
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v80, v51, v35
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v79, v51, v31
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v78, v47, v39
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v77, v47, v35
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v65, v47, v31
	;;#ASMEND
	s_waitcnt vmcnt(2)
	;;#ASMSTART
	v_dot2c_f32_f16 v90, v40, v8
	;;#ASMEND
	s_waitcnt vmcnt(1)
	;;#ASMSTART
	v_dot2c_f32_f16 v89, v40, v4
	;;#ASMEND
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	v_dot2c_f32_f16 v88, v40, v0
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v87, v24, v8
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v86, v24, v4
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v85, v24, v0
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v84, v20, v8
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v83, v20, v4
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v82, v20, v0
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v81, v16, v8
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v80, v16, v4
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v79, v16, v0
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v78, v12, v8
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v77, v12, v4
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v65, v12, v0
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v90, v41, v9
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v89, v41, v5
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v88, v41, v1
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v87, v25, v9
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v86, v25, v5
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v85, v25, v1
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v84, v21, v9
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v83, v21, v5
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v82, v21, v1
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v81, v17, v9
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v80, v17, v5
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v79, v17, v1
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v78, v13, v9
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v77, v13, v5
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v65, v13, v1
	;;#ASMEND
	s_addk_i32 s16, 0x400
	;;#ASMSTART
	v_dot2c_f32_f16 v90, v42, v10
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v89, v42, v6
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v88, v42, v2
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v87, v26, v10
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v86, v26, v6
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v85, v26, v2
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v84, v22, v10
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v83, v22, v6
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v82, v22, v2
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v81, v18, v10
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v80, v18, v6
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v79, v18, v2
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v78, v14, v10
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v77, v14, v6
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v65, v14, v2
	;;#ASMEND
	s_cmp_ge_u32 s16, s2
	v_add_u32_e32 v91, 0x800, v91
	;;#ASMSTART
	v_dot2c_f32_f16 v90, v43, v11
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v89, v43, v7
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v88, v43, v3
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v87, v27, v11
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v86, v27, v7
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v85, v27, v3
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v84, v23, v11
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v83, v23, v7
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v82, v23, v3
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v81, v19, v11
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v80, v19, v7
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v79, v19, v3
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v78, v15, v11
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v77, v15, v7
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v65, v15, v3
	;;#ASMEND
	s_cbranch_scc1 .LBB0_150
.LBB0_147:                              ;   Parent Loop BB0_143 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_add_u32_e32 v16, s16, v74
	v_min_u32_e32 v66, s54, v16
	v_lshl_add_u64 v[0:1], v[66:67], 1, s[4:5]
	v_add_u32_e32 v92, 0x200, v16
	v_lshl_add_u64 v[2:3], v[0:1], 0, v[68:69]
	v_min_u32_e32 v66, s54, v92
	v_lshl_add_u64 v[4:5], v[0:1], 0, v[70:71]
	global_load_dwordx4 v[36:39], v[2:3], off nt
	global_load_dwordx4 v[32:35], v[4:5], off nt
	v_lshl_add_u64 v[2:3], v[66:67], 1, s[4:5]
	v_lshl_add_u64 v[0:1], v[0:1], 0, v[72:73]
	v_lshl_add_u64 v[4:5], v[2:3], 0, v[68:69]
	v_lshl_add_u64 v[12:13], v[2:3], 0, v[70:71]
	global_load_dwordx4 v[28:31], v[0:1], off nt
	global_load_dwordx4 v[8:11], v[4:5], off nt
	v_lshl_add_u64 v[14:15], v[2:3], 0, v[72:73]
	global_load_dwordx4 v[4:7], v[12:13], off nt
	global_load_dwordx4 v[0:3], v[14:15], off nt
	v_cmp_gt_u32_e32 vcc, s2, v16
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v63, 0
	v_mov_b32_e32 v56, 0
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v58, 0
	v_mov_b32_e32 v59, 0
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v24, 0
	v_mov_b32_e32 v25, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v27, 0
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
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_146
; %bb.148:                              ;   in Loop: Header=BB0_147 Depth=2
	v_add_u32_e32 v95, s57, v91
	v_add_u32_e32 v93, s59, v91
	ds_read_b128 v[60:63], v91
	v_add_u32_e32 v94, s58, v91
	ds_read_b128 v[56:59], v95
	ds_read_b128 v[52:55], v94
	v_add_u32_e32 v66, s60, v91
	ds_read_b128 v[48:51], v93
	ds_read_b128 v[44:47], v66
	v_cmp_gt_u32_e32 vcc, s2, v92
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
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v25, 0
	v_mov_b32_e32 v24, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v40, 0
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_145
; %bb.149:                              ;   in Loop: Header=BB0_147 Depth=2
	ds_read_b128 v[40:43], v91 offset:1024
	ds_read_b128 v[24:27], v95 offset:1024
	ds_read_b128 v[20:23], v94 offset:1024
	ds_read_b128 v[16:19], v93 offset:1024
	ds_read_b128 v[12:15], v66 offset:1024
	s_branch .LBB0_145
.LBB0_150:                              ;   in Loop: Header=BB0_143 Depth=1
	; sched_barrier mask(0x00000000)
	s_nop 0
	v_add_f32_dpp v10, v85, v85 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v90, v90 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v89, v89 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v10, v10, v10 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v88, v88 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v87, v87 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v10, v10, v10 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v86, v86 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v0, v0 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v10, v10, v10 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v14, v10, v10 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v10, v84, v84 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v6, v6 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v8, v8 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v10, v10, v10 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v0, v0 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v10, v10, v10 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v6, v6 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v10, v10, v10 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v8, v8 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v0, v0 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v17, v10, v10 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v10, v83, v83 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v10, v10, v10 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v6, v6 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v8, v8 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v10, v10, v10 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v0, v0 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v10, v10, v10 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v6, v6 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v19, v10, v10 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v10, v82, v82 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v8, v8 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v1, v0 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v10, v10, v10 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v3, v2 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v5, v4 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v10, v10, v10 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v7, v6 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v9, v8 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v10, v10, v10 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v16, v14 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v18, v17 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v21, v10, v10 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v10, v81, v81 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v20, v19 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v22, v21 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v10, v10, v10 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v10, v10, v10 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v10, v10, v10 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v23, v10, v10 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v10, v80, v80 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v24, v23 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v10, v10, v10 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v10, v10, v10 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v10, v10, v10 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v25, v10, v10 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v10, v79, v79 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v26, v25 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v10, v10, v10 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v10, v10, v10 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v10, v10, v10 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v27, v10, v10 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v10, v78, v78 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v28, v27 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v10, v10, v10 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v10, v10, v10 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v10, v10, v10 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v29, v10, v10 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v10, v77, v77 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v30, v29 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v10, v10, v10 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v10, v10, v10 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v10, v10, v10 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v31, v10, v10 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v10, v65, v65 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v32, v31 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v10, v10, v10 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v10, v10, v10 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v10, v10, v10 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_add_f32_dpp v33, v10, v10 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_mov_b32_dpp v34, v33 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_and_saveexec_b64 s[48:49], s[0:1]
	s_cbranch_execz .LBB0_142
; %bb.151:                              ;   in Loop: Header=BB0_143 Depth=1
	v_cmp_eq_u32_e64 s[8:9], 0, v75
	s_and_b64 s[6:7], s[22:23], s[8:9]
	v_add_f32_e32 v15, v0, v1
	v_add_f32_e32 v13, v2, v3
	v_add_f32_e32 v12, v4, v5
	v_add_f32_e32 v11, v6, v7
	v_add_f32_e32 v10, v8, v9
	v_add_f32_e32 v9, v14, v16
	v_add_f32_e32 v8, v17, v18
	v_add_f32_e32 v7, v19, v20
	v_add_f32_e32 v6, v21, v22
	v_add_f32_e32 v5, v23, v24
	v_add_f32_e32 v4, v25, v26
	v_add_f32_e32 v3, v27, v28
	v_add_f32_e32 v2, v29, v30
	v_add_f32_e32 v1, v31, v32
	v_add_f32_e32 v0, v33, v34
	s_xor_b64 s[16:17], s[6:7], -1
	s_mov_b64 s[6:7], 0
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b64 s[18:19], s[16:17]
	s_xor_b64 s[50:51], exec, s[18:19]
	s_cbranch_execnz .LBB0_154
; %bb.152:                              ;   in Loop: Header=BB0_143 Depth=1
	s_andn2_saveexec_b64 s[8:9], s[50:51]
	s_cbranch_execnz .LBB0_215
.LBB0_153:                              ;   in Loop: Header=BB0_143 Depth=1
	s_or_b64 exec, exec, s[8:9]
	s_and_b64 exec, exec, s[6:7]
	s_cbranch_execz .LBB0_142
	s_branch .LBB0_216
.LBB0_154:                              ;   in Loop: Header=BB0_143 Depth=1
	v_cndmask_b32_e64 v14, 0, 1, s[24:25]
	v_cmp_ne_u32_e64 s[6:7], 1, v14
	s_and_saveexec_b64 s[16:17], s[8:9]
	s_cbranch_execnz .LBB0_168
; %bb.155:                              ;   in Loop: Header=BB0_143 Depth=1
	s_or_b64 exec, exec, s[16:17]
	v_cmp_gt_u32_e64 s[16:17], 2, v75
	s_and_saveexec_b64 s[18:19], s[16:17]
	s_cbranch_execnz .LBB0_171
.LBB0_156:                              ;   in Loop: Header=BB0_143 Depth=1
	s_or_b64 exec, exec, s[18:19]
	v_cmp_gt_u32_e64 s[18:19], 3, v75
	s_and_saveexec_b64 s[52:53], s[18:19]
	s_cbranch_execnz .LBB0_174
.LBB0_157:                              ;   in Loop: Header=BB0_143 Depth=1
	s_or_b64 exec, exec, s[52:53]
	s_and_saveexec_b64 s[52:53], s[8:9]
	s_cbranch_execnz .LBB0_177
.LBB0_158:                              ;   in Loop: Header=BB0_143 Depth=1
	s_or_b64 exec, exec, s[52:53]
	s_and_saveexec_b64 s[52:53], s[16:17]
	s_cbranch_execnz .LBB0_180
.LBB0_159:                              ;   in Loop: Header=BB0_143 Depth=1
	s_or_b64 exec, exec, s[52:53]
	s_and_saveexec_b64 s[52:53], s[18:19]
	s_cbranch_execnz .LBB0_183
.LBB0_160:                              ;   in Loop: Header=BB0_143 Depth=1
	s_or_b64 exec, exec, s[52:53]
	s_and_saveexec_b64 s[52:53], s[8:9]
	s_cbranch_execnz .LBB0_186
.LBB0_161:                              ;   in Loop: Header=BB0_143 Depth=1
	s_or_b64 exec, exec, s[52:53]
	s_and_saveexec_b64 s[52:53], s[16:17]
	s_cbranch_execnz .LBB0_189
.LBB0_162:                              ;   in Loop: Header=BB0_143 Depth=1
	s_or_b64 exec, exec, s[52:53]
	s_and_saveexec_b64 s[52:53], s[18:19]
	s_cbranch_execnz .LBB0_192
.LBB0_163:                              ;   in Loop: Header=BB0_143 Depth=1
	s_or_b64 exec, exec, s[52:53]
	s_and_saveexec_b64 s[52:53], s[8:9]
	s_cbranch_execnz .LBB0_195
.LBB0_164:                              ;   in Loop: Header=BB0_143 Depth=1
	s_or_b64 exec, exec, s[52:53]
	s_and_saveexec_b64 s[52:53], s[16:17]
	s_cbranch_execnz .LBB0_198
.LBB0_165:                              ;   in Loop: Header=BB0_143 Depth=1
	s_or_b64 exec, exec, s[52:53]
	s_and_saveexec_b64 s[52:53], s[18:19]
	s_cbranch_execnz .LBB0_201
.LBB0_166:                              ;   in Loop: Header=BB0_143 Depth=1
	s_or_b64 exec, exec, s[52:53]
	s_and_saveexec_b64 s[52:53], s[8:9]
	s_cbranch_execnz .LBB0_204
.LBB0_167:                              ;   in Loop: Header=BB0_143 Depth=1
	s_or_b64 exec, exec, s[52:53]
	s_and_saveexec_b64 s[8:9], s[16:17]
	s_cbranch_execnz .LBB0_207
	s_branch .LBB0_210
.LBB0_168:                              ;   in Loop: Header=BB0_143 Depth=1
	v_mul_f32_e32 v14, s28, v15
	s_and_b64 vcc, exec, s[6:7]
	v_mov_b32_e32 v65, v67
	s_cbranch_vccnz .LBB0_170
; %bb.169:                              ;   in Loop: Header=BB0_143 Depth=1
	v_lshl_add_u64 v[16:17], v[64:65], 1, s[12:13]
	global_load_ushort v15, v[16:17], off
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v14, s29, v15, v14 op_sel_hi:[0,1,0]
.LBB0_170:                              ;   in Loop: Header=BB0_143 Depth=1
	v_cvt_f16_f32_e32 v16, v14
	v_lshl_add_u64 v[14:15], v[64:65], 1, s[14:15]
	global_store_short v[14:15], v16, off
	s_or_b64 exec, exec, s[16:17]
	v_cmp_gt_u32_e64 s[16:17], 2, v75
	s_and_saveexec_b64 s[18:19], s[16:17]
	s_cbranch_execz .LBB0_156
.LBB0_171:                              ;   in Loop: Header=BB0_143 Depth=1
	v_mul_f32_e32 v13, s28, v13
	s_and_b64 vcc, exec, s[6:7]
	v_add_u32_e32 v66, 1, v64
	s_cbranch_vccnz .LBB0_173
; %bb.172:                              ;   in Loop: Header=BB0_143 Depth=1
	v_lshl_add_u64 v[14:15], v[66:67], 1, s[12:13]
	global_load_ushort v14, v[14:15], off
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v13, s29, v14, v13 op_sel_hi:[0,1,0]
.LBB0_173:                              ;   in Loop: Header=BB0_143 Depth=1
	v_cvt_f16_f32_e32 v13, v13
	v_lshl_add_u64 v[14:15], v[66:67], 1, s[14:15]
	global_store_short v[14:15], v13, off
	s_or_b64 exec, exec, s[18:19]
	v_cmp_gt_u32_e64 s[18:19], 3, v75
	s_and_saveexec_b64 s[52:53], s[18:19]
	s_cbranch_execz .LBB0_157
.LBB0_174:                              ;   in Loop: Header=BB0_143 Depth=1
	v_mul_f32_e32 v12, s28, v12
	s_and_b64 vcc, exec, s[6:7]
	v_add_u32_e32 v66, 2, v64
	s_cbranch_vccnz .LBB0_176
; %bb.175:                              ;   in Loop: Header=BB0_143 Depth=1
	v_lshl_add_u64 v[14:15], v[66:67], 1, s[12:13]
	global_load_ushort v13, v[14:15], off
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v12, s29, v13, v12 op_sel_hi:[0,1,0]
.LBB0_176:                              ;   in Loop: Header=BB0_143 Depth=1
	v_cvt_f16_f32_e32 v14, v12
	v_lshl_add_u64 v[12:13], v[66:67], 1, s[14:15]
	global_store_short v[12:13], v14, off
	s_or_b64 exec, exec, s[52:53]
	s_and_saveexec_b64 s[52:53], s[8:9]
	s_cbranch_execz .LBB0_158
.LBB0_177:                              ;   in Loop: Header=BB0_143 Depth=1
	v_mul_f32_e32 v11, s28, v11
	s_and_b64 vcc, exec, s[6:7]
	v_mov_b32_e32 v65, v67
	s_cbranch_vccnz .LBB0_179
; %bb.178:                              ;   in Loop: Header=BB0_143 Depth=1
	v_lshl_add_u64 v[12:13], v[64:65], 1, s[26:27]
	global_load_ushort v12, v[12:13], off
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v11, s29, v12, v11 op_sel_hi:[0,1,0]
.LBB0_179:                              ;   in Loop: Header=BB0_143 Depth=1
	v_cvt_f16_f32_e32 v11, v11
	v_lshl_add_u64 v[12:13], v[64:65], 1, s[30:31]
	global_store_short v[12:13], v11, off
	s_or_b64 exec, exec, s[52:53]
	s_and_saveexec_b64 s[52:53], s[16:17]
	s_cbranch_execz .LBB0_159
.LBB0_180:                              ;   in Loop: Header=BB0_143 Depth=1
	v_mul_f32_e32 v10, s28, v10
	s_and_b64 vcc, exec, s[6:7]
	v_add_u32_e32 v66, 1, v64
	s_cbranch_vccnz .LBB0_182
; %bb.181:                              ;   in Loop: Header=BB0_143 Depth=1
	v_lshl_add_u64 v[12:13], v[66:67], 1, s[26:27]
	global_load_ushort v11, v[12:13], off
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v10, s29, v11, v10 op_sel_hi:[0,1,0]
.LBB0_182:                              ;   in Loop: Header=BB0_143 Depth=1
	v_cvt_f16_f32_e32 v12, v10
	v_lshl_add_u64 v[10:11], v[66:67], 1, s[30:31]
	global_store_short v[10:11], v12, off
	s_or_b64 exec, exec, s[52:53]
	s_and_saveexec_b64 s[52:53], s[18:19]
	s_cbranch_execz .LBB0_160
.LBB0_183:                              ;   in Loop: Header=BB0_143 Depth=1
	v_mul_f32_e32 v9, s28, v9
	s_and_b64 vcc, exec, s[6:7]
	v_add_u32_e32 v66, 2, v64
	s_cbranch_vccnz .LBB0_185
; %bb.184:                              ;   in Loop: Header=BB0_143 Depth=1
	v_lshl_add_u64 v[10:11], v[66:67], 1, s[26:27]
	global_load_ushort v10, v[10:11], off
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v9, s29, v10, v9 op_sel_hi:[0,1,0]
.LBB0_185:                              ;   in Loop: Header=BB0_143 Depth=1
	v_cvt_f16_f32_e32 v9, v9
	v_lshl_add_u64 v[10:11], v[66:67], 1, s[30:31]
	global_store_short v[10:11], v9, off
	s_or_b64 exec, exec, s[52:53]
	s_and_saveexec_b64 s[52:53], s[8:9]
	s_cbranch_execz .LBB0_161
.LBB0_186:                              ;   in Loop: Header=BB0_143 Depth=1
	v_mul_f32_e32 v8, s28, v8
	s_and_b64 vcc, exec, s[6:7]
	v_mov_b32_e32 v65, v67
	s_cbranch_vccnz .LBB0_188
; %bb.187:                              ;   in Loop: Header=BB0_143 Depth=1
	v_lshl_add_u64 v[10:11], v[64:65], 1, s[34:35]
	global_load_ushort v9, v[10:11], off
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v8, s29, v9, v8 op_sel_hi:[0,1,0]
.LBB0_188:                              ;   in Loop: Header=BB0_143 Depth=1
	v_cvt_f16_f32_e32 v10, v8
	v_lshl_add_u64 v[8:9], v[64:65], 1, s[36:37]
	global_store_short v[8:9], v10, off
	s_or_b64 exec, exec, s[52:53]
	s_and_saveexec_b64 s[52:53], s[16:17]
	s_cbranch_execz .LBB0_162
.LBB0_189:                              ;   in Loop: Header=BB0_143 Depth=1
	v_mul_f32_e32 v7, s28, v7
	s_and_b64 vcc, exec, s[6:7]
	v_add_u32_e32 v66, 1, v64
	s_cbranch_vccnz .LBB0_191
; %bb.190:                              ;   in Loop: Header=BB0_143 Depth=1
	v_lshl_add_u64 v[8:9], v[66:67], 1, s[34:35]
	global_load_ushort v8, v[8:9], off
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v7, s29, v8, v7 op_sel_hi:[0,1,0]
.LBB0_191:                              ;   in Loop: Header=BB0_143 Depth=1
	v_cvt_f16_f32_e32 v7, v7
	v_lshl_add_u64 v[8:9], v[66:67], 1, s[36:37]
	global_store_short v[8:9], v7, off
	s_or_b64 exec, exec, s[52:53]
	s_and_saveexec_b64 s[52:53], s[18:19]
	s_cbranch_execz .LBB0_163
.LBB0_192:                              ;   in Loop: Header=BB0_143 Depth=1
	v_mul_f32_e32 v6, s28, v6
	s_and_b64 vcc, exec, s[6:7]
	v_add_u32_e32 v66, 2, v64
	s_cbranch_vccnz .LBB0_194
; %bb.193:                              ;   in Loop: Header=BB0_143 Depth=1
	v_lshl_add_u64 v[8:9], v[66:67], 1, s[34:35]
	global_load_ushort v7, v[8:9], off
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v6, s29, v7, v6 op_sel_hi:[0,1,0]
.LBB0_194:                              ;   in Loop: Header=BB0_143 Depth=1
	v_cvt_f16_f32_e32 v8, v6
	v_lshl_add_u64 v[6:7], v[66:67], 1, s[36:37]
	global_store_short v[6:7], v8, off
	s_or_b64 exec, exec, s[52:53]
	s_and_saveexec_b64 s[52:53], s[8:9]
	s_cbranch_execz .LBB0_164
.LBB0_195:                              ;   in Loop: Header=BB0_143 Depth=1
	v_mul_f32_e32 v5, s28, v5
	s_and_b64 vcc, exec, s[6:7]
	v_mov_b32_e32 v65, v67
	s_cbranch_vccnz .LBB0_197
; %bb.196:                              ;   in Loop: Header=BB0_143 Depth=1
	v_lshl_add_u64 v[6:7], v[64:65], 1, s[38:39]
	global_load_ushort v6, v[6:7], off
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v5, s29, v6, v5 op_sel_hi:[0,1,0]
.LBB0_197:                              ;   in Loop: Header=BB0_143 Depth=1
	v_cvt_f16_f32_e32 v5, v5
	v_lshl_add_u64 v[6:7], v[64:65], 1, s[40:41]
	global_store_short v[6:7], v5, off
	s_or_b64 exec, exec, s[52:53]
	s_and_saveexec_b64 s[52:53], s[16:17]
	s_cbranch_execz .LBB0_165
.LBB0_198:                              ;   in Loop: Header=BB0_143 Depth=1
	v_mul_f32_e32 v4, s28, v4
	s_and_b64 vcc, exec, s[6:7]
	v_add_u32_e32 v66, 1, v64
	s_cbranch_vccnz .LBB0_200
; %bb.199:                              ;   in Loop: Header=BB0_143 Depth=1
	v_lshl_add_u64 v[6:7], v[66:67], 1, s[38:39]
	global_load_ushort v5, v[6:7], off
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v4, s29, v5, v4 op_sel_hi:[0,1,0]
.LBB0_200:                              ;   in Loop: Header=BB0_143 Depth=1
	v_cvt_f16_f32_e32 v6, v4
	v_lshl_add_u64 v[4:5], v[66:67], 1, s[40:41]
	global_store_short v[4:5], v6, off
	s_or_b64 exec, exec, s[52:53]
	s_and_saveexec_b64 s[52:53], s[18:19]
	s_cbranch_execz .LBB0_166
.LBB0_201:                              ;   in Loop: Header=BB0_143 Depth=1
	v_mul_f32_e32 v3, s28, v3
	s_and_b64 vcc, exec, s[6:7]
	v_add_u32_e32 v66, 2, v64
	s_cbranch_vccnz .LBB0_203
; %bb.202:                              ;   in Loop: Header=BB0_143 Depth=1
	v_lshl_add_u64 v[4:5], v[66:67], 1, s[38:39]
	global_load_ushort v4, v[4:5], off
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v3, s29, v4, v3 op_sel_hi:[0,1,0]
.LBB0_203:                              ;   in Loop: Header=BB0_143 Depth=1
	v_cvt_f16_f32_e32 v3, v3
	v_lshl_add_u64 v[4:5], v[66:67], 1, s[40:41]
	global_store_short v[4:5], v3, off
	s_or_b64 exec, exec, s[52:53]
	s_and_saveexec_b64 s[52:53], s[8:9]
	s_cbranch_execz .LBB0_167
.LBB0_204:                              ;   in Loop: Header=BB0_143 Depth=1
	v_mul_f32_e32 v2, s28, v2
	s_and_b64 vcc, exec, s[6:7]
	v_mov_b32_e32 v65, v67
	s_cbranch_vccnz .LBB0_206
; %bb.205:                              ;   in Loop: Header=BB0_143 Depth=1
	v_lshl_add_u64 v[4:5], v[64:65], 1, s[42:43]
	global_load_ushort v3, v[4:5], off
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v2, s29, v3, v2 op_sel_hi:[0,1,0]
.LBB0_206:                              ;   in Loop: Header=BB0_143 Depth=1
	v_cvt_f16_f32_e32 v4, v2
	v_lshl_add_u64 v[2:3], v[64:65], 1, s[44:45]
	global_store_short v[2:3], v4, off
	s_or_b64 exec, exec, s[52:53]
	s_and_saveexec_b64 s[8:9], s[16:17]
	s_cbranch_execz .LBB0_210
.LBB0_207:                              ;   in Loop: Header=BB0_143 Depth=1
	v_mul_f32_e32 v1, s28, v1
	s_and_b64 vcc, exec, s[6:7]
	v_add_u32_e32 v66, 1, v64
	s_cbranch_vccnz .LBB0_209
; %bb.208:                              ;   in Loop: Header=BB0_143 Depth=1
	v_lshl_add_u64 v[2:3], v[66:67], 1, s[42:43]
	global_load_ushort v2, v[2:3], off
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v1, s29, v2, v1 op_sel_hi:[0,1,0]
.LBB0_209:                              ;   in Loop: Header=BB0_143 Depth=1
	v_cvt_f16_f32_e32 v1, v1
	v_lshl_add_u64 v[2:3], v[66:67], 1, s[44:45]
	global_store_short v[2:3], v1, off
.LBB0_210:                              ;   in Loop: Header=BB0_143 Depth=1
	s_or_b64 exec, exec, s[8:9]
	s_mov_b64 s[16:17], 0
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b64 s[8:9], s[18:19]
	s_xor_b64 s[8:9], exec, s[8:9]
	s_cbranch_execz .LBB0_214
; %bb.211:                              ;   in Loop: Header=BB0_143 Depth=1
	v_mul_f32_e32 v14, s28, v0
	s_and_b64 vcc, exec, s[6:7]
	v_add_u32_e32 v66, 2, v64
	s_cbranch_vccnz .LBB0_213
; %bb.212:                              ;   in Loop: Header=BB0_143 Depth=1
	v_lshl_add_u64 v[0:1], v[66:67], 1, s[42:43]
	global_load_ushort v0, v[0:1], off
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v14, s29, v0, v14 op_sel_hi:[0,1,0]
.LBB0_213:                              ;   in Loop: Header=BB0_143 Depth=1
	s_mov_b64 s[16:17], exec
.LBB0_214:                              ;   in Loop: Header=BB0_143 Depth=1
	s_or_b64 exec, exec, s[8:9]
	s_and_b64 s[6:7], s[16:17], exec
                                        ; implicit-def: $vgpr15
                                        ; implicit-def: $vgpr13
                                        ; implicit-def: $vgpr12
                                        ; implicit-def: $vgpr11
                                        ; implicit-def: $vgpr10
                                        ; implicit-def: $vgpr9
                                        ; implicit-def: $vgpr8
                                        ; implicit-def: $vgpr7
                                        ; implicit-def: $vgpr6
                                        ; implicit-def: $vgpr5
                                        ; implicit-def: $vgpr4
                                        ; implicit-def: $vgpr3
                                        ; implicit-def: $vgpr2
                                        ; implicit-def: $vgpr1
                                        ; implicit-def: $vgpr0
	s_andn2_saveexec_b64 s[8:9], s[50:51]
	s_cbranch_execz .LBB0_153
.LBB0_215:                              ;   in Loop: Header=BB0_143 Depth=1
	v_mov_b32_e32 v65, v67
	v_fma_mixlo_f16 v18, s28, v15, 0
	v_lshlrev_b64 v[14:15], 1, v[64:65]
	v_lshl_add_u64 v[16:17], s[14:15], 0, v[14:15]
	v_add_u32_e32 v66, 1, v64
	global_store_short v[16:17], v18, off
	v_lshlrev_b64 v[16:17], 1, v[66:67]
	v_fma_mixlo_f16 v13, s28, v13, 0
	v_lshl_add_u64 v[18:19], s[14:15], 0, v[16:17]
	v_add_u32_e32 v66, 2, v64
	global_store_short v[18:19], v13, off
	v_fma_mixlo_f16 v20, s28, v12, 0
	v_lshlrev_b64 v[12:13], 1, v[66:67]
	v_lshl_add_u64 v[18:19], s[14:15], 0, v[12:13]
	global_store_short v[18:19], v20, off
	v_fma_mixlo_f16 v11, s28, v11, 0
	v_lshl_add_u64 v[18:19], s[30:31], 0, v[14:15]
	global_store_short v[18:19], v11, off
	v_fma_mixlo_f16 v18, s28, v10, 0
	v_lshl_add_u64 v[10:11], s[30:31], 0, v[16:17]
	global_store_short v[10:11], v18, off
	v_fma_mixlo_f16 v9, s28, v9, 0
	v_lshl_add_u64 v[10:11], s[30:31], 0, v[12:13]
	global_store_short v[10:11], v9, off
	v_fma_mixlo_f16 v10, s28, v8, 0
	v_lshl_add_u64 v[8:9], s[36:37], 0, v[14:15]
	global_store_short v[8:9], v10, off
	v_fma_mixlo_f16 v7, s28, v7, 0
	v_lshl_add_u64 v[8:9], s[36:37], 0, v[16:17]
	global_store_short v[8:9], v7, off
	v_fma_mixlo_f16 v8, s28, v6, 0
	v_lshl_add_u64 v[6:7], s[36:37], 0, v[12:13]
	global_store_short v[6:7], v8, off
	v_fma_mixlo_f16 v5, s28, v5, 0
	v_lshl_add_u64 v[6:7], s[40:41], 0, v[14:15]
	global_store_short v[6:7], v5, off
	v_fma_mixlo_f16 v6, s28, v4, 0
	v_lshl_add_u64 v[4:5], s[40:41], 0, v[16:17]
	global_store_short v[4:5], v6, off
	v_fma_mixlo_f16 v3, s28, v3, 0
	v_lshl_add_u64 v[4:5], s[40:41], 0, v[12:13]
	global_store_short v[4:5], v3, off
	v_fma_mixlo_f16 v4, s28, v2, 0
	v_lshl_add_u64 v[2:3], s[44:45], 0, v[14:15]
	global_store_short v[2:3], v4, off
	v_fma_mixlo_f16 v1, s28, v1, 0
	v_lshl_add_u64 v[2:3], s[44:45], 0, v[16:17]
	global_store_short v[2:3], v1, off
	v_mul_f32_e32 v14, s28, v0
	s_or_b64 s[6:7], s[6:7], exec
	s_or_b64 exec, exec, s[8:9]
	s_and_b64 exec, exec, s[6:7]
	s_cbranch_execz .LBB0_142
.LBB0_216:                              ;   in Loop: Header=BB0_143 Depth=1
	v_cvt_f16_f32_e32 v2, v14
	v_lshl_add_u64 v[0:1], v[66:67], 1, s[44:45]
	global_store_short v[0:1], v2, off
	s_branch .LBB0_142
.LBB0_217:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wvSplitK_vllm_f16_tn_n5_y3u2
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
		.amdhsa_next_free_vgpr 98
		.amdhsa_next_free_sgpr 96
		.amdhsa_accum_offset 100
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
	.size	wvSplitK_vllm_f16_tn_n5_y3u2, .Lfunc_end0-wvSplitK_vllm_f16_tn_n5_y3u2
                                        ; -- End function
	.set wvSplitK_vllm_f16_tn_n5_y3u2.num_vgpr, 98
	.set wvSplitK_vllm_f16_tn_n5_y3u2.num_agpr, 0
	.set wvSplitK_vllm_f16_tn_n5_y3u2.numbered_sgpr, 81
	.set wvSplitK_vllm_f16_tn_n5_y3u2.num_named_barrier, 0
	.set wvSplitK_vllm_f16_tn_n5_y3u2.private_seg_size, 0
	.set wvSplitK_vllm_f16_tn_n5_y3u2.uses_vcc, 1
	.set wvSplitK_vllm_f16_tn_n5_y3u2.uses_flat_scratch, 0
	.set wvSplitK_vllm_f16_tn_n5_y3u2.has_dyn_sized_stack, 0
	.set wvSplitK_vllm_f16_tn_n5_y3u2.has_recursion, 0
	.set wvSplitK_vllm_f16_tn_n5_y3u2.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 13760
; TotalNumSgprs: 87
; NumVgprs: 98
; NumAgprs: 0
; TotalNumVgprs: 98
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 163840 bytes/workgroup (compile time only)
; SGPRBlocks: 12
; VGPRBlocks: 12
; NumSGPRsForWavesPerEU: 102
; NumVGPRsForWavesPerEU: 98
; AccumOffset: 100
; Occupancy: 4
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 1
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 24
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_a88a324983b4629d,@object ; @__hip_cuid_a88a324983b4629d
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_a88a324983b4629d
__hip_cuid_a88a324983b4629d:
	.byte	0                               ; 0x0
	.size	__hip_cuid_a88a324983b4629d, 1

	.ident	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.4 26084 f58b06dce1f9c15707c5f808fd002e18c2accf7e)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_a88a324983b4629d
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
    .group_segment_fixed_size: 163840
    .kernarg_segment_align: 8
    .kernarg_segment_size: 72
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           wvSplitK_vllm_f16_tn_n5_y3u2
    .private_segment_fixed_size: 0
    .sgpr_count:     87
    .sgpr_spill_count: 0
    .symbol:         wvSplitK_vllm_f16_tn_n5_y3u2.kd
    .vgpr_count:     98
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 1
...

	.end_amdgpu_metadata
