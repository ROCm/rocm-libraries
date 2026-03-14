; To reproduce the .rocmasm from .optimized.ll, run:
; llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -mattr='-fma-mix-insts' -O3 <.optimized.ll> -o <out.rocmasm>

	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.text
	.globl	wave_mxfp4_static_gemm_256x192x256_14976x51712x7680
	.p2align	8
	.type	wave_mxfp4_static_gemm_256x192x256_14976x51712x7680,@function
wave_mxfp4_static_gemm_256x192x256_14976x51712x7680:
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
.LBB0_0:
	v_and_b32_e32 v96, 0x3ff, v0
	v_bfe_u32 v0, v0, 10, 10
	v_lshrrev_b32_e32 v4, 6, v96
	v_lshlrev_b32_e32 v1, 5, v0
	v_lshl_or_b32 v2, v4, 3, v1
	s_mov_b64 s[24:25], s[2:3]
	v_readfirstlane_b32 s2, v2
	v_lshrrev_b32_e32 v2, 3, v96
	v_or_b32_e32 v6, v2, v1
	s_lshl_b32 s42, s16, 8
	v_or_b32_e32 v5, s42, v6
	v_bitop3_b32 v7, v2, 7, v96 bitop3:0x48
	v_lshlrev_b32_e32 v101, 4, v7
	v_mul_u32_u24_e32 v102, 0xf00, v5
	s_and_b32 s3, s25, 0xffff
	s_lshl_b32 s16, s2, 7
	s_or_b32 s25, s3, 0x4f000000
	s_mov_b32 s27, 0x27000
	s_mov_b32 s26, 0x7ffffffe
	v_or_b32_e32 v3, v102, v101
	s_mov_b32 m0, s16
	s_or_b32 s44, s16, 0x2000
	buffer_load_dwordx4 v3, s[24:27], 0 offen lds
	v_add_u32_e32 v3, 0x3c000, v3
	s_mov_b32 m0, s44
	v_or_b32_e32 v8, 0x80, v5
	buffer_load_dwordx4 v3, s[24:27], 0 offen lds
	v_mul_u32_u24_e32 v3, 0xf00, v8
	s_movk_i32 s45, 0x3a80
	v_or_b32_e32 v9, v3, v101
	v_bfrev_b32_e32 v103, -2
	v_cmp_gt_u32_e64 s[12:13], s45, v8
	s_or_b32 s46, s16, 0x4000
	s_mov_b32 m0, s46
	v_cndmask_b32_e64 v8, v103, v9, s[12:13]
	buffer_load_dwordx4 v8, s[24:27], 0 offen lds
	v_or_b32_e32 v8, 0xc0, v5
	v_mul_u32_u24_e32 v5, 0xf00, v8
	v_or_b32_e32 v9, v5, v101
	v_cmp_gt_u32_e64 s[2:3], s45, v8
	s_or_b32 s47, s16, 0x6000
	s_mul_i32 s43, s17, 0xc0
	v_cndmask_b32_e64 v8, v103, v9, s[2:3]
	s_mov_b32 m0, s47
	v_and_or_b32 v9, v6, 48, s43
	v_lshrrev_b32_e32 v6, 7, v96
	buffer_load_dwordx4 v8, s[24:27], 0 offen lds
	v_lshlrev_b32_e32 v104, 4, v2
	v_lshlrev_b32_e32 v8, 8, v6
	s_mov_b64 s[20:21], s[6:7]
	v_sub_u32_e32 v10, v104, v8
	v_lshlrev_b32_e32 v7, 8, v7
	s_movk_i32 s6, 0xf00
	v_add_u32_e32 v10, v7, v10
	s_and_b32 s7, s21, 0xffff
	s_or_b32 s21, s7, 0x4f000000
	s_mov_b32 s22, s26
	s_mov_b32 s23, s27
	v_mad_u32_u24 v9, v9, s6, v10
	s_add_i32 m0, s16, 0x10000
	s_cmpk_lt_u32 s17, 0x10d
	buffer_load_dwordx4 v9, s[20:23], 0 offen lds
	v_lshlrev_b32_e32 v9, 4, v6
	v_or3_b32 v9, v9, s43, v1
	v_mad_u32_u24 v9, v9, s6, v10
	v_add_u32_e32 v10, 0x3c000, v9
	s_cselect_b64 s[18:19], -1, 0
	v_cndmask_b32_e64 v10, v103, v10, s[18:19]
	s_add_i32 m0, s44, 0x10000
	v_add_u32_e32 v9, 0x78000, v9
	buffer_load_dwordx4 v10, s[20:23], 0 offen lds
	v_cndmask_b32_e64 v9, v103, v9, s[18:19]
	s_add_i32 m0, s46, 0x10000
	v_bfe_u32 v97, v96, 4, 2
	buffer_load_dwordx4 v9, s[20:23], 0 offen lds
	v_lshrrev_b32_e32 v9, 4, v96
	v_lshlrev_b32_e32 v10, 4, v97
	v_mul_i32_i24_e32 v98, -16, v9
	v_add3_u32 v122, v10, v98, v96
	v_ashrrev_i32_e32 v10, 31, v122
	v_xor_b32_e32 v11, v10, v122
	s_mov_b32 s48, 0x88888889
	v_mul_hi_i32 v12, v11, s48
	v_add_u32_e32 v11, v12, v11
	v_lshrrev_b32_e32 v12, 31, v11
	v_ashrrev_i32_e32 v11, 5, v11
	s_mul_i32 s15, s15, s42
	s_mul_hi_u32 s6, s14, s42
	v_add_u32_e32 v11, v11, v12
	s_add_i32 s15, s6, s15
	s_mul_i32 s22, s14, s42
	v_xor_b32_e32 v15, v11, v10
	v_and_b32_e32 v10, 60, v96
	s_add_u32 s28, s4, s22
	s_load_dwordx2 s[40:41], s[0:1], 0x40
	v_mov_b32_e32 v11, 0xffffff10
	v_cmp_eq_u32_e32 vcc, 60, v10
	s_addc_u32 s4, s5, s15
	s_and_b32 s5, s14, 0x3fff
	v_cndmask_b32_e32 v14, 0, v11, vcc
	v_lshlrev_b32_e32 v11, 2, v96
	v_lshlrev_b32_e32 v12, 6, v9
	s_bitset1_b32 s5, 14
	v_and_b32_e32 v18, 0xc0, v96
	v_lshlrev_b32_e32 v10, 6, v97
	v_sub_u32_e32 v17, v11, v12
	s_and_b32 s4, s4, 0xffff
	s_lshl_b32 s5, s5, 16
	v_mov_b32_e32 v13, s43
	v_add_u32_e32 v16, v15, v18
	v_add3_u32 v14, v17, v10, v14
	s_or_b32 s29, s4, s5
	s_movk_i32 s4, 0x60
	v_add_u32_e32 v19, s42, v16
	v_mad_u64_u32 v[16:17], s[6:7], v16, s14, v[14:15]
	v_mad_u32_u24 v99, v0, s4, v13
	v_add_u32_e32 v13, v99, v15
	v_mad_u32_u24 v15, v0, s4, v15
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s4, s41, s43
	s_mul_hi_u32 s6, s40, s43
	s_add_i32 s4, s6, s4
	s_mul_i32 s22, s40, s43
	v_mad_u64_u32 v[14:15], s[6:7], s40, v15, v[14:15]
	s_add_u32 s36, s8, s22
	s_addc_u32 s4, s9, s4
	s_and_b32 s6, s40, 0x3fff
	v_cmp_gt_i32_e32 vcc, s45, v19
	s_movk_i32 s5, 0x3a60
	s_bitset1_b32 s6, 14
	v_cndmask_b32_e32 v17, v103, v16, vcc
	v_lshl_add_u32 v16, s14, 5, v16
	v_cmp_gt_i32_e32 vcc, s5, v19
	s_mov_b32 s15, 0xca00
	s_and_b32 s4, s4, 0xffff
	s_lshl_b32 s6, s6, 16
	v_cndmask_b32_e32 v16, v103, v16, vcc
	s_or_b32 s37, s4, s6
	v_cmp_gt_i32_e32 vcc, s15, v13
	s_mov_b32 s9, 0xc9e0
	s_lshl_b32 s4, s40, 5
	s_mov_b32 s30, s26
	s_mov_b32 s31, s27
	v_cndmask_b32_e32 v15, v103, v14, vcc
	v_add_u32_e32 v14, s4, v14
	v_cmp_gt_i32_e32 vcc, s9, v13
	s_mov_b32 s49, 0xc9c0
	buffer_load_dword v118, v17, s[28:31], 0 offen
	buffer_load_dword v117, v16, s[28:31], 0 offen
	s_mov_b32 s38, s26
	s_mov_b32 s39, s27
	v_cndmask_b32_e32 v16, v103, v14, vcc
	v_add_u32_e32 v14, s4, v14
	v_cmp_gt_i32_e32 vcc, s49, v13
	v_cmp_eq_u32_e64 s[6:7], 0, v0
	s_mov_b32 s8, 0
	v_cndmask_b32_e32 v13, v103, v14, vcc
	buffer_load_dword v121, v15, s[36:39], 0 offen
	buffer_load_dword v120, v16, s[36:39], 0 offen
	buffer_load_dword v119, v13, s[36:39], 0 offen
	s_mov_b32 s41, 0x10000
	v_readfirstlane_b32 s4, v18
	v_mul_u32_u24_e32 v100, 0x60, v0
	s_and_b64 vcc, exec, s[6:7]
	s_barrier
	s_waitcnt vmcnt(0)
	s_cbranch_vccnz .LBB0_2
	s_barrier
.LBB0_2:
	s_load_dwordx2 s[34:35], s[0:1], 0x48
	s_mul_i32 s1, s17, 0xb4000
	v_add_u32_e32 v1, v2, v1
	v_lshrrev_b32_e32 v1, 4, v1
	s_mov_b32 s17, 0xf000
	v_mov_b32_e32 v2, s1
	v_and_b32_e32 v13, 7, v96
	v_lshlrev_b32_e32 v15, 7, v96
	v_lshlrev_b32_e32 v9, 11, v9
	s_movk_i32 s0, 0x3000
	v_mad_u32_u24 v1, v1, s17, v2
	s_mov_b32 s17, 0x1e000
	v_bitop3_b32 v14, v97, v96, 7 bitop3:0x78
	v_sub_u32_e32 v9, v15, v9
	v_mul_lo_u32 v15, v0, s0
	v_bitop3_b32 v13, v97, v13, 4 bitop3:0x36
	v_or_b32_e32 v1, v1, v7
	v_mul_lo_u32 v0, v0, s17
	v_lshl_add_u32 v4, v4, 13, v9
	v_lshlrev_b32_e32 v14, 4, v14
	v_lshlrev_b32_e32 v13, 4, v13
	v_sub_u32_e32 v1, v1, v8
	v_add_u32_e32 v0, s1, v0
	s_mov_b32 s1, 0xef00
	v_or_b32_e32 v105, v4, v14
	v_or_b32_e32 v106, v13, v4
	v_add_u32_e32 v4, v10, v11
	v_add_u32_e32 v113, 0x800, v1
	v_mul_lo_u32 v1, v6, s1
	v_add_u32_e32 v9, v9, v15
	v_sub_u32_e32 v4, v4, v12
	v_add_u32_e32 v0, v0, v1
	v_mov_b32_e32 v52, 0
	v_or_b32_e32 v107, v9, v14
	v_or_b32_e32 v108, v13, v9
	s_or_b32 s33, s4, s42
	v_add_u32_e32 v109, 32, v100
	v_add_u32_e32 v110, 16, v4
	v_add_u32_e32 v111, 64, v100
	s_add_i32 s0, s4, 32
	v_sub_u32_e32 v112, -5, v122
	v_add_u32_e32 v114, v0, v7
	v_add_u32_e32 v115, 0x80, v5
	v_add_u32_e32 v116, 0x80, v3
	s_mov_b32 s52, 0x8000
	s_mov_b32 s53, 0x16000
	s_mov_b32 s1, 0
	s_mov_b32 s22, s26
	s_mov_b32 s23, s27
	s_movk_i32 s17, 0xffc0
	s_movk_i32 s50, 0xf0
	s_mov_b32 s30, s26
	s_mov_b32 s31, s27
	s_mov_b32 s38, s26
	s_mov_b32 s39, s27
	v_mov_b32_e32 v53, v52
	v_mov_b32_e32 v54, v52
	v_mov_b32_e32 v55, v52
	v_mov_b32_e32 v92, v52
	v_mov_b32_e32 v93, v52
	v_mov_b32_e32 v94, v52
	v_mov_b32_e32 v95, v52
	v_mov_b32_e32 v88, v52
	v_mov_b32_e32 v89, v52
	v_mov_b32_e32 v90, v52
	v_mov_b32_e32 v91, v52
	v_mov_b32_e32 v84, v52
	v_mov_b32_e32 v85, v52
	v_mov_b32_e32 v86, v52
	v_mov_b32_e32 v87, v52
	v_mov_b32_e32 v80, v52
	v_mov_b32_e32 v81, v52
	v_mov_b32_e32 v82, v52
	v_mov_b32_e32 v83, v52
	v_mov_b32_e32 v76, v52
	v_mov_b32_e32 v77, v52
	v_mov_b32_e32 v78, v52
	v_mov_b32_e32 v79, v52
	v_mov_b32_e32 v72, v52
	v_mov_b32_e32 v73, v52
	v_mov_b32_e32 v74, v52
	v_mov_b32_e32 v75, v52
	v_mov_b32_e32 v68, v52
	v_mov_b32_e32 v69, v52
	v_mov_b32_e32 v70, v52
	v_mov_b32_e32 v71, v52
	v_mov_b32_e32 v64, v52
	v_mov_b32_e32 v65, v52
	v_mov_b32_e32 v66, v52
	v_mov_b32_e32 v67, v52
	v_mov_b32_e32 v60, v52
	v_mov_b32_e32 v61, v52
	v_mov_b32_e32 v62, v52
	v_mov_b32_e32 v63, v52
	v_mov_b32_e32 v56, v52
	v_mov_b32_e32 v57, v52
	v_mov_b32_e32 v58, v52
	v_mov_b32_e32 v59, v52
	v_mov_b32_e32 v48, v52
	v_mov_b32_e32 v49, v52
	v_mov_b32_e32 v50, v52
	v_mov_b32_e32 v51, v52
	v_mov_b32_e32 v44, v52
	v_mov_b32_e32 v45, v52
	v_mov_b32_e32 v46, v52
	v_mov_b32_e32 v47, v52
	v_mov_b32_e32 v40, v52
	v_mov_b32_e32 v41, v52
	v_mov_b32_e32 v42, v52
	v_mov_b32_e32 v43, v52
	v_mov_b32_e32 v28, v52
	v_mov_b32_e32 v29, v52
	v_mov_b32_e32 v30, v52
	v_mov_b32_e32 v31, v52
	v_mov_b32_e32 v16, v52
	v_mov_b32_e32 v17, v52
	v_mov_b32_e32 v18, v52
	v_mov_b32_e32 v19, v52
	v_mov_b32_e32 v12, v52
	v_mov_b32_e32 v13, v52
	v_mov_b32_e32 v14, v52
	v_mov_b32_e32 v15, v52
	v_mov_b32_e32 v20, v52
	v_mov_b32_e32 v21, v52
	v_mov_b32_e32 v22, v52
	v_mov_b32_e32 v23, v52
	v_mov_b32_e32 v24, v52
	v_mov_b32_e32 v25, v52
	v_mov_b32_e32 v26, v52
	v_mov_b32_e32 v27, v52
	v_mov_b32_e32 v32, v52
	v_mov_b32_e32 v33, v52
	v_mov_b32_e32 v34, v52
	v_mov_b32_e32 v35, v52
	v_mov_b32_e32 v36, v52
	v_mov_b32_e32 v37, v52
	v_mov_b32_e32 v38, v52
	v_mov_b32_e32 v39, v52
	v_mov_b32_e32 v8, v52
	v_mov_b32_e32 v9, v52
	v_mov_b32_e32 v10, v52
	v_mov_b32_e32 v11, v52
	v_mov_b32_e32 v4, v52
	v_mov_b32_e32 v5, v52
	v_mov_b32_e32 v6, v52
	v_mov_b32_e32 v7, v52
	v_mov_b32_e32 v0, v52
	v_mov_b32_e32 v1, v52
	v_mov_b32_e32 v2, v52
	v_mov_b32_e32 v3, v52
.LBB0_3:
	s_mov_b32 s51, s8
	s_mov_b32 s8, s52
	s_mov_b32 s52, s41
	s_mov_b32 s41, s53
	v_add_u32_e32 v123, v102, v101
	v_add_u32_e32 v124, 0x80, v123
	s_add_i32 m0, s8, s16
	s_barrier
	buffer_load_dwordx4 v124, s[24:27], 0 offen lds
	v_add_u32_e32 v123, 0x3c080, v123
	s_add_i32 m0, s8, s44
	s_nop 0
	buffer_load_dwordx4 v123, s[24:27], 0 offen lds
	v_add_u32_e32 v123, v101, v116
	v_cndmask_b32_e64 v123, v103, v123, s[12:13]
	s_add_i32 m0, s8, s46
	s_nop 0
	buffer_load_dwordx4 v123, s[24:27], 0 offen lds
	v_add_u32_e32 v123, v101, v115
	v_cndmask_b32_e64 v123, v103, v123, s[2:3]
	s_add_i32 m0, s8, s47
	s_nop 0
	buffer_load_dwordx4 v123, s[24:27], 0 offen lds
	v_add_u32_e32 v123, v113, v104
	s_add_i32 m0, s53, s16
	s_nop 0
	buffer_load_dwordx4 v123, s[20:23], 0 offen lds
	v_add_u32_e32 v123, v104, v114
	v_add_u32_e32 v124, 0x3c800, v123
	v_cndmask_b32_e64 v124, v103, v124, s[18:19]
	s_add_i32 m0, s53, s44
	v_add_u32_e32 v123, 0x78800, v123
	buffer_load_dwordx4 v124, s[20:23], 0 offen lds
	v_cndmask_b32_e64 v123, v103, v123, s[18:19]
	s_add_i32 m0, s53, s46
	s_nop 0
	buffer_load_dwordx4 v123, s[20:23], 0 offen lds
	v_add_u32_e32 v128, 64, v122
	v_subrev_u32_e32 v123, 60, v112
	v_cmp_gt_i32_e32 vcc, s17, v122
	v_add_u32_e32 v126, 4, v122
	s_nop 0
	v_cndmask_b32_e32 v123, v128, v123, vcc
	v_mul_hi_i32 v124, v123, s48
	v_add_u32_e32 v123, v124, v123
	v_lshrrev_b32_e32 v124, 31, v123
	v_ashrrev_i32_e32 v123, 5, v123
	v_add_u32_e32 v123, v123, v124
	v_cndmask_b32_e64 v124, 0, -1, vcc
	v_cmp_gt_i32_e32 vcc, -4, v122
	v_xor_b32_e32 v124, v123, v124
	v_add_u32_e32 v125, s4, v124
	v_cndmask_b32_e32 v122, v126, v112, vcc
	v_mul_hi_i32 v126, v122, s48
	v_add_u32_e32 v122, v126, v122
	v_lshrrev_b32_e32 v126, 31, v122
	v_ashrrev_i32_e32 v122, 5, v122
	v_add_u32_e32 v122, v122, v126
	v_cndmask_b32_e64 v126, 0, -1, vcc
	v_xor_b32_e32 v122, v122, v126
	v_mul_lo_u32 v125, v125, s14
	v_mul_lo_u32 v122, v122, s50
	v_add_u32_e32 v127, s0, v124
	v_add_u32_e32 v123, s33, v124
	v_sub_u32_e32 v125, v125, v122
	v_add_u32_e32 v126, s1, v110
	v_mul_lo_u32 v127, s14, v127
	v_add_u32_e32 v125, v126, v125
	v_cmp_gt_i32_e32 vcc, s45, v123
	v_sub_u32_e32 v127, v127, v122
	v_add_u32_e32 v127, v126, v127
	v_cndmask_b32_e32 v125, v103, v125, vcc
	v_cmp_gt_i32_e32 vcc, s5, v123
	v_add_u32_e32 v129, v124, v100
	v_mul_lo_u32 v129, v129, s40
	v_cndmask_b32_e32 v123, v103, v127, vcc
	v_add_u32_e32 v130, v109, v124
	buffer_load_dword v127, v125, s[28:31], 0 offen
	s_nop 0
	buffer_load_dword v123, v123, s[28:31], 0 offen
	v_add_u32_e32 v125, v99, v124
	v_sub_u32_e32 v129, v129, v122
	v_mul_lo_u32 v130, s40, v130
	v_add_u32_e32 v124, v111, v124
	v_add_u32_e32 v129, v126, v129
	v_cmp_gt_i32_e32 vcc, s15, v125
	v_sub_u32_e32 v130, v130, v122
	v_mul_lo_u32 v124, s40, v124
	v_cndmask_b32_e32 v129, v103, v129, vcc
	v_add_u32_e32 v130, v126, v130
	v_cmp_gt_i32_e32 vcc, s9, v125
	v_sub_u32_e32 v122, v124, v122
	v_add_u32_e32 v122, v126, v122
	v_cndmask_b32_e32 v130, v103, v130, vcc
	v_cmp_gt_i32_e32 vcc, s49, v125
	s_nop 1
	v_cndmask_b32_e32 v122, v103, v122, vcc
	buffer_load_dword v126, v129, s[36:39], 0 offen
	buffer_load_dword v125, v130, s[36:39], 0 offen
	buffer_load_dword v124, v122, s[36:39], 0 offen
	v_add_u32_e32 v122, s51, v105
	ds_read_b128 v[130:133], v122
	ds_read_b128 v[134:137], v122 offset:2048
	ds_read_b128 v[138:141], v122 offset:4096
	ds_read_b128 v[142:145], v122 offset:6144
	v_add_u32_e32 v122, s52, v107
	ds_read_b128 v[146:149], v122
	ds_read_b128 v[150:153], v122 offset:2048
	ds_read_b128 v[154:157], v122 offset:4096
	ds_read_b128 v[158:161], v122 offset:6144
	ds_read_b128 v[162:165], v122 offset:8192
	ds_read_b128 v[166:169], v122 offset:10240
	s_barrier
	s_setprio 1
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[130:133], v[146:149], v[52:55], v118, v121 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[130:133], v[150:153], v[92:95], v118, v121 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[130:133], v[154:157], v[88:91], v118, v120 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[84:87], v[130:133], v[158:161], v[84:87], v118, v120 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[80:83], v[130:133], v[162:165], v[80:83], v118, v119 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[130:133], v[166:169], v[76:79], v118, v119 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[134:137], v[146:149], v[72:75], v118, v121 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[134:137], v[150:153], v[68:71], v118, v121 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[134:137], v[154:157], v[64:67], v118, v120 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[134:137], v[158:161], v[60:63], v118, v120 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[134:137], v[162:165], v[56:59], v118, v119 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[134:137], v[166:169], v[48:51], v118, v119 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[138:141], v[146:149], v[44:47], v117, v121 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[138:141], v[150:153], v[40:43], v117, v121 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[138:141], v[154:157], v[28:31], v117, v120 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[138:141], v[158:161], v[16:19], v117, v120 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[138:141], v[162:165], v[12:15], v117, v119 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[138:141], v[166:169], v[20:23], v117, v119 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[142:145], v[146:149], v[24:27], v117, v121 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[142:145], v[150:153], v[32:35], v117, v121 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[142:145], v[154:157], v[36:39], v117, v120 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[142:145], v[158:161], v[8:11], v117, v120 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[142:145], v[162:165], v[4:7], v117, v119 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[142:145], v[166:169], v[0:3], v117, v119 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_barrier
	v_add_u32_e32 v122, s51, v106
	ds_read_b128 v[130:133], v122
	ds_read_b128 v[134:137], v122 offset:2048
	ds_read_b128 v[138:141], v122 offset:4096
	ds_read_b128 v[142:145], v122 offset:6144
	v_add_u32_e32 v122, s52, v108
	ds_read_b128 v[146:149], v122
	ds_read_b128 v[150:153], v122 offset:2048
	ds_read_b128 v[154:157], v122 offset:4096
	ds_read_b128 v[158:161], v122 offset:6144
	ds_read_b128 v[162:165], v122 offset:8192
	ds_read_b128 v[166:169], v122 offset:10240
	s_waitcnt vmcnt(5)
	s_barrier
	s_setprio 1
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[130:133], v[146:149], v[52:55], v118, v121 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[130:133], v[150:153], v[92:95], v118, v121 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[130:133], v[154:157], v[88:91], v118, v120 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[84:87], v[130:133], v[158:161], v[84:87], v118, v120 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[80:83], v[130:133], v[162:165], v[80:83], v118, v119 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[130:133], v[166:169], v[76:79], v118, v119 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[134:137], v[146:149], v[72:75], v118, v121 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[134:137], v[150:153], v[68:71], v118, v121 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[134:137], v[154:157], v[64:67], v118, v120 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[134:137], v[158:161], v[60:63], v118, v120 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[134:137], v[162:165], v[56:59], v118, v119 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[134:137], v[166:169], v[48:51], v118, v119 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[138:141], v[146:149], v[44:47], v117, v121 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[138:141], v[150:153], v[40:43], v117, v121 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[138:141], v[154:157], v[28:31], v117, v120 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[138:141], v[158:161], v[16:19], v117, v120 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[138:141], v[162:165], v[12:15], v117, v119 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[138:141], v[166:169], v[20:23], v117, v119 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[142:145], v[146:149], v[24:27], v117, v121 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[142:145], v[150:153], v[32:35], v117, v121 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[142:145], v[154:157], v[36:39], v117, v120 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[142:145], v[158:161], v[8:11], v117, v120 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[142:145], v[162:165], v[4:7], v117, v119 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[142:145], v[166:169], v[0:3], v117, v119 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	s_addk_i32 s1, 0x100
	v_subrev_u32_e32 v112, 64, v112
	v_add_u32_e32 v113, 0x800, v113
	v_add_u32_e32 v114, 0x800, v114
	v_add_u32_e32 v102, 0x80, v102
	v_add_u32_e32 v115, 0x80, v115
	v_add_u32_e32 v116, 0x80, v116
	s_cmpk_lg_i32 s1, 0x1d00
	v_mov_b32_e32 v122, v128
	s_mov_b32 s53, s52
	s_mov_b32 s52, s51
	s_waitcnt vmcnt(0)
	v_mov_b32_e32 v119, v124
	v_mov_b32_e32 v120, v125
	v_mov_b32_e32 v121, v126
	v_mov_b32_e32 v117, v123
	v_mov_b32_e32 v118, v127
	s_cbranch_scc1 .LBB0_3
	s_andn2_b64 vcc, exec, s[6:7]
	s_cbranch_vccnz .LBB0_6
	s_barrier
.LBB0_6:
	v_add_u32_e32 v101, s41, v107
	v_add_u32_e32 v102, s41, v108
	s_barrier
	ds_read_b128 v[168:171], v101
	ds_read_b128 v[172:175], v101 offset:2048
	ds_read_b128 v[176:179], v102
	ds_read_b128 v[180:183], v102 offset:2048
	ds_read_b128 v[184:187], v101 offset:4096
	ds_read_b128 v[140:143], v101 offset:6144
	ds_read_b128 v[188:191], v102 offset:4096
	ds_read_b128 v[136:139], v102 offset:6144
	ds_read_b128 v[132:135], v101 offset:8192
	ds_read_b128 v[118:121], v101 offset:10240
	ds_read_b128 v[128:131], v102 offset:8192
	ds_read_b128 v[114:117], v102 offset:10240
	v_add_u32_e32 v101, s8, v105
	ds_read_b128 v[156:159], v101
	ds_read_b128 v[192:195], v101 offset:2048
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[156:159], v[168:171], v[52:55], v127, v126 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add_u32_e32 v106, s8, v106
	ds_read_b128 v[160:163], v106
	ds_read_b128 v[196:199], v106 offset:2048
	ds_read_b128 v[200:203], v101 offset:4096
	ds_read_b128 v[102:105], v101 offset:6144
	ds_read_b128 v[204:207], v106 offset:4096
	ds_read_b128 v[106:109], v106 offset:6144
	s_movk_i32 s0, 0x7fff
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[160:163], v[176:179], v[52:55], v127, v126 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mov_b32_e32 v101, 0x7fc0
	s_cmpk_lt_u32 s33, 0x3a80
	s_mul_i32 s2, s35, s42
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[156:159], v[172:175], v[92:95], v127, v126 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_mul_hi_u32 s3, s34, s42
	s_nop 2
	v_bfe_u32 v122, v55, 16, 1
	v_bfe_u32 v144, v54, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[156:159], v[184:187], v[88:91], v127, v125 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v145, v53, 16, 1
	v_bfe_u32 v146, v52, 16, 1
	v_cmp_o_f32_e32 vcc, v55, v55
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], v[160:163], v[180:183], v[92:95], v127, v126 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cselect_b64 s[16:17], -1, 0
	s_add_i32 s3, s3, s2
	s_mul_i32 s2, s34, s42
	v_add3_u32 v95, v55, v122, s0
	v_add3_u32 v94, v54, v144, s0
	v_lshrrev_b32_e32 v95, 16, v95
	v_add3_u32 v92, v52, v146, s0
	v_add3_u32 v93, v53, v145, s0
	v_lshrrev_b32_e32 v122, 16, v94
	v_mfma_scale_f32_16x16x128_f8f6f4 v[144:147], v[160:163], v[188:191], v[88:91], v127, v125 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_lshl_b64 s[2:3], s[2:3], 1
	s_mov_b32 s23, 0x27000
	s_mov_b32 s22, 0x7ffffffd
	v_cndmask_b32_e32 v88, v101, v95, vcc
	v_cmp_o_f32_e32 vcc, v54, v54
	v_lshrrev_b32_e32 v89, 16, v93
	v_lshrrev_b32_e32 v90, 16, v92
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[156:159], v[140:143], v[84:87], v127, v125 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 2
	v_cndmask_b32_e32 v84, v101, v122, vcc
	v_cmp_o_f32_e32 vcc, v53, v53
	v_mfma_scale_f32_16x16x128_f8f6f4 v[148:151], v[160:163], v[136:139], v[92:95], v127, v125 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_bfe_u32 v85, v113, 16, 1
	v_cndmask_b32_e32 v91, v101, v89, vcc
	v_cmp_o_f32_e32 vcc, v52, v52
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[156:159], v[132:135], v[80:83], v127, v124 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e32 v93, v101, v90, vcc
	v_cmp_o_f32_e32 vcc, v113, v113
	v_mfma_scale_f32_16x16x128_f8f6f4 v[152:155], v[160:163], v[128:131], v[52:55], v127, v124 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_bfe_u32 v80, v112, 16, 1
	v_bfe_u32 v81, v111, 16, 1
	v_add3_u32 v80, v112, v80, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[156:159], v[118:121], v[76:79], v127, v124 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v82, v110, 16, 1
	v_add3_u32 v81, v111, v81, s0
	v_add3_u32 v82, v110, v82, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[156:159], v[160:163], v[114:117], v[52:55], v127, v124 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_add3_u32 v76, v113, v85, s0
	v_lshrrev_b32_e32 v76, 16, v76
	v_lshrrev_b32_e32 v77, 16, v80
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[192:195], v[168:171], v[72:75], v127, v126 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v79, v101, v76, vcc
	v_cmp_o_f32_e32 vcc, v112, v112
	v_lshrrev_b32_e32 v78, 16, v81
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[160:163], v[196:199], v[176:179], v[52:55], v127, v126 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_bfe_u32 v72, v147, 16, 1
	v_cndmask_b32_e32 v83, v101, v77, vcc
	v_cmp_o_f32_e32 vcc, v111, v111
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[192:195], v[172:175], v[68:71], v127, v126 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v80, 16, v82
	v_cndmask_b32_e32 v87, v101, v78, vcc
	v_cmp_o_f32_e32 vcc, v110, v110
	v_mfma_scale_f32_16x16x128_f8f6f4 v[164:167], v[196:199], v[180:183], v[52:55], v127, v126 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_bfe_u32 v68, v146, 16, 1
	v_bfe_u32 v69, v145, 16, 1
	v_bfe_u32 v70, v144, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[192:195], v[184:187], v[64:67], v127, v125 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v70, v144, v70, s0
	v_add3_u32 v69, v145, v69, s0
	v_add3_u32 v68, v146, v68, s0
	v_add3_u32 v64, v147, v72, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[196:199], v[188:191], v[52:55], v127, v125 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v92, v101, v80, vcc
	v_lshrrev_b32_e32 v64, 16, v64
	v_lshrrev_b32_e32 v65, 16, v68
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[192:195], v[140:143], v[60:63], v127, v125 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v66, 16, v69
	v_lshrrev_b32_e32 v67, 16, v70
	v_cmp_o_f32_e32 vcc, v147, v147
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[196:199], v[136:139], v[52:55], v127, v125 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_bfe_u32 v60, v151, 16, 1
	v_cndmask_b32_e32 v78, v101, v64, vcc
	v_cmp_o_f32_e32 vcc, v146, v146
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[192:195], v[132:135], v[56:59], v127, v124 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e32 v82, v101, v65, vcc
	v_cmp_o_f32_e32 vcc, v145, v145
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[192:195], v[118:121], v[48:51], v127, v124 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v56, v150, 16, 1
	v_cndmask_b32_e32 v86, v101, v66, vcc
	v_cmp_o_f32_e32 vcc, v144, v144
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[200:203], v[168:171], v[44:47], v123, v126 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v57, v149, 16, 1
	v_cndmask_b32_e32 v90, v101, v67, vcc
	v_cmp_o_f32_e32 vcc, v151, v151
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[196:199], v[128:131], v[52:55], v127, v124 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_bfe_u32 v58, v148, 16, 1
	s_nop 1
	v_add3_u32 v55, v151, v60, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[200:203], v[172:175], v[40:43], v123, v126 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v54, v150, v56, s0
	v_lshrrev_b32_e32 v55, 16, v55
	v_add3_u32 v53, v149, v57, s0
	v_lshrrev_b32_e32 v54, 16, v54
	v_cndmask_b32_e32 v77, v101, v55, vcc
	v_cmp_o_f32_e32 vcc, v150, v150
	v_add3_u32 v52, v148, v58, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[196:199], v[114:117], v[48:51], v127, v124 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v81, v101, v54, vcc
	v_cmp_o_f32_e32 vcc, v149, v149
	s_nop 0
	v_lshrrev_b32_e32 v48, 16, v53
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[204:207], v[176:179], v[44:47], v123, v126 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v49, 16, v52
	v_cndmask_b32_e32 v85, v101, v48, vcc
	v_cmp_o_f32_e32 vcc, v148, v148
	v_bfe_u32 v44, v155, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[200:203], v[132:135], v[12:15], v123, v124 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v45, v154, 16, 1
	v_cndmask_b32_e32 v89, v101, v49, vcc
	v_bfe_u32 v46, v153, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[204:207], v[180:183], v[40:43], v123, v126 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_bfe_u32 v47, v152, 16, 1
	v_cmp_o_f32_e32 vcc, v155, v155
	s_nop 0
	v_add3_u32 v43, v155, v44, s0
	v_add3_u32 v42, v154, v45, s0
	v_lshrrev_b32_e32 v43, 16, v43
	v_add3_u32 v40, v152, v47, s0
	v_add3_u32 v41, v153, v46, s0
	v_lshrrev_b32_e32 v42, 16, v42
	v_cndmask_b32_e32 v76, v101, v43, vcc
	v_cmp_o_f32_e32 vcc, v154, v154
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[200:203], v[184:187], v[28:31], v123, v125 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v94, 16, v41
	v_lshrrev_b32_e32 v95, 16, v40
	v_cndmask_b32_e32 v80, v101, v42, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[204:207], v[128:131], v[12:15], v123, v124 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v153, v153
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[200:203], v[118:121], v[20:23], v123, v124 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e32 v111, v101, v94, vcc
	v_cmp_o_f32_e32 vcc, v152, v152
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[204:207], v[188:191], v[28:31], v123, v125 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e32 v113, v101, v95, vcc
	v_cmp_o_f32_e32 vcc, v159, v159
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[204:207], v[114:117], v[12:15], v123, v124 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[102:105], v[168:171], v[24:27], v123, v126 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[200:203], v[140:143], v[16:19], v123, v125 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 1
	v_bfe_u32 v24, v159, 16, 1
	v_bfe_u32 v25, v158, 16, 1
	v_add3_u32 v24, v159, v24, s0
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[106:109], v[176:179], v[12:15], v123, v126 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_bfe_u32 v26, v157, 16, 1
	v_add3_u32 v25, v158, v25, s0
	v_lshrrev_b32_e32 v24, 16, v24
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[102:105], v[172:175], v[32:35], v123, v126 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v27, v156, 16, 1
	v_add3_u32 v26, v157, v26, s0
	v_lshrrev_b32_e32 v25, 16, v25
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[204:207], v[136:139], v[16:19], v123, v125 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_add3_u32 v27, v156, v27, s0
	v_lshrrev_b32_e32 v26, 16, v26
	v_lshrrev_b32_e32 v27, 16, v27
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[106:109], v[180:183], v[12:15], v123, v126 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[102:105], v[184:187], v[36:39], v123, v125 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[102:105], v[140:143], v[8:11], v123, v125 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[102:105], v[132:135], v[4:7], v123, v124 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[102:105], v[118:121], v[0:3], v123, v124 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v105, v101, v24, vcc
	v_cmp_o_f32_e32 vcc, v158, v158
	v_bfe_u32 v24, v163, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[106:109], v[188:191], v[12:15], v123, v125 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_add3_u32 v24, v163, v24, s0
	v_lshrrev_b32_e32 v24, 16, v24
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[106:109], v[136:139], v[8:11], v123, v125 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[106:109], v[128:131], v[4:7], v123, v124 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[106:109], v[114:117], v[0:3], v123, v124 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v108, v101, v25, vcc
	v_cmp_o_f32_e32 vcc, v157, v157
	v_bfe_u32 v25, v162, 16, 1
	v_add3_u32 v25, v162, v25, s0
	v_cndmask_b32_e32 v109, v101, v26, vcc
	v_cmp_o_f32_e32 vcc, v156, v156
	v_bfe_u32 v26, v161, 16, 1
	v_add3_u32 v26, v161, v26, s0
	v_cndmask_b32_e32 v112, v101, v27, vcc
	v_cmp_o_f32_e32 vcc, v163, v163
	v_bfe_u32 v27, v160, 16, 1
	v_lshrrev_b32_e32 v25, 16, v25
	v_cndmask_b32_e32 v95, v101, v24, vcc
	v_cmp_o_f32_e32 vcc, v162, v162
	v_add3_u32 v27, v160, v27, s0
	v_lshrrev_b32_e32 v26, 16, v26
	v_cndmask_b32_e32 v103, v101, v25, vcc
	v_cmp_o_f32_e32 vcc, v161, v161
	v_bfe_u32 v24, v167, 16, 1
	v_lshrrev_b32_e32 v27, 16, v27
	v_cndmask_b32_e32 v107, v101, v26, vcc
	v_cmp_o_f32_e32 vcc, v160, v160
	v_bfe_u32 v25, v166, 16, 1
	v_add3_u32 v24, v167, v24, s0
	v_cndmask_b32_e32 v110, v101, v27, vcc
	v_bfe_u32 v26, v165, 16, 1
	v_add3_u32 v25, v166, v25, s0
	v_lshrrev_b32_e32 v24, 16, v24
	v_cmp_o_f32_e32 vcc, v167, v167
	v_bfe_u32 v27, v164, 16, 1
	v_add3_u32 v26, v165, v26, s0
	v_lshrrev_b32_e32 v25, 16, v25
	v_cndmask_b32_e32 v94, v101, v24, vcc
	v_cmp_o_f32_e32 vcc, v166, v166
	v_add3_u32 v27, v164, v27, s0
	v_lshrrev_b32_e32 v26, 16, v26
	v_cndmask_b32_e32 v102, v101, v25, vcc
	v_cmp_o_f32_e32 vcc, v165, v165
	v_bfe_u32 v24, v75, 16, 1
	v_lshrrev_b32_e32 v27, 16, v27
	v_cndmask_b32_e32 v104, v101, v26, vcc
	v_cmp_o_f32_e32 vcc, v164, v164
	v_bfe_u32 v25, v74, 16, 1
	v_add3_u32 v24, v75, v24, s0
	v_cndmask_b32_e32 v106, v101, v27, vcc
	v_bfe_u32 v26, v73, 16, 1
	v_add3_u32 v25, v74, v25, s0
	v_lshrrev_b32_e32 v24, 16, v24
	v_cmp_o_f32_e32 vcc, v75, v75
	v_bfe_u32 v27, v72, 16, 1
	v_add3_u32 v26, v73, v26, s0
	v_lshrrev_b32_e32 v25, 16, v25
	v_cndmask_b32_e32 v75, v101, v24, vcc
	v_cmp_o_f32_e32 vcc, v74, v74
	v_add3_u32 v27, v72, v27, s0
	v_lshrrev_b32_e32 v26, 16, v26
	v_cndmask_b32_e32 v74, v101, v25, vcc
	v_cmp_o_f32_e32 vcc, v73, v73
	v_bfe_u32 v24, v71, 16, 1
	v_lshrrev_b32_e32 v27, 16, v27
	v_cndmask_b32_e32 v73, v101, v26, vcc
	v_cmp_o_f32_e32 vcc, v72, v72
	v_bfe_u32 v25, v70, 16, 1
	v_add3_u32 v24, v71, v24, s0
	v_cndmask_b32_e32 v72, v101, v27, vcc
	v_bfe_u32 v26, v69, 16, 1
	v_add3_u32 v25, v70, v25, s0
	v_lshrrev_b32_e32 v24, 16, v24
	v_cmp_o_f32_e32 vcc, v71, v71
	v_bfe_u32 v27, v68, 16, 1
	v_add3_u32 v26, v69, v26, s0
	v_lshrrev_b32_e32 v25, 16, v25
	v_cndmask_b32_e32 v71, v101, v24, vcc
	v_cmp_o_f32_e32 vcc, v70, v70
	v_add3_u32 v27, v68, v27, s0
	v_lshrrev_b32_e32 v26, 16, v26
	v_cndmask_b32_e32 v70, v101, v25, vcc
	v_cmp_o_f32_e32 vcc, v69, v69
	v_bfe_u32 v24, v67, 16, 1
	v_lshrrev_b32_e32 v27, 16, v27
	v_cndmask_b32_e32 v69, v101, v26, vcc
	v_cmp_o_f32_e32 vcc, v68, v68
	v_bfe_u32 v25, v66, 16, 1
	v_add3_u32 v24, v67, v24, s0
	v_cndmask_b32_e32 v68, v101, v27, vcc
	v_bfe_u32 v26, v65, 16, 1
	v_add3_u32 v25, v66, v25, s0
	v_lshrrev_b32_e32 v24, 16, v24
	v_cmp_o_f32_e32 vcc, v67, v67
	v_bfe_u32 v27, v64, 16, 1
	v_add3_u32 v26, v65, v26, s0
	v_lshrrev_b32_e32 v25, 16, v25
	v_cndmask_b32_e32 v67, v101, v24, vcc
	v_cmp_o_f32_e32 vcc, v66, v66
	v_add3_u32 v27, v64, v27, s0
	v_lshrrev_b32_e32 v26, 16, v26
	v_cndmask_b32_e32 v66, v101, v25, vcc
	v_cmp_o_f32_e32 vcc, v65, v65
	v_bfe_u32 v24, v63, 16, 1
	v_lshrrev_b32_e32 v27, 16, v27
	v_cndmask_b32_e32 v65, v101, v26, vcc
	v_cmp_o_f32_e32 vcc, v64, v64
	v_bfe_u32 v25, v62, 16, 1
	v_add3_u32 v24, v63, v24, s0
	v_cndmask_b32_e32 v64, v101, v27, vcc
	v_bfe_u32 v26, v61, 16, 1
	v_add3_u32 v25, v62, v25, s0
	v_lshrrev_b32_e32 v24, 16, v24
	v_cmp_o_f32_e32 vcc, v63, v63
	v_bfe_u32 v27, v60, 16, 1
	v_add3_u32 v26, v61, v26, s0
	v_lshrrev_b32_e32 v25, 16, v25
	v_cndmask_b32_e32 v63, v101, v24, vcc
	v_cmp_o_f32_e32 vcc, v62, v62
	v_add3_u32 v27, v60, v27, s0
	v_lshrrev_b32_e32 v26, 16, v26
	v_cndmask_b32_e32 v62, v101, v25, vcc
	v_cmp_o_f32_e32 vcc, v61, v61
	v_bfe_u32 v24, v59, 16, 1
	v_lshrrev_b32_e32 v27, 16, v27
	v_cndmask_b32_e32 v61, v101, v26, vcc
	v_cmp_o_f32_e32 vcc, v60, v60
	v_bfe_u32 v25, v58, 16, 1
	v_add3_u32 v24, v59, v24, s0
	v_cndmask_b32_e32 v60, v101, v27, vcc
	v_bfe_u32 v26, v57, 16, 1
	v_add3_u32 v25, v58, v25, s0
	v_lshrrev_b32_e32 v24, 16, v24
	v_cmp_o_f32_e32 vcc, v59, v59
	v_bfe_u32 v27, v56, 16, 1
	v_add3_u32 v26, v57, v26, s0
	v_lshrrev_b32_e32 v25, 16, v25
	v_cndmask_b32_e32 v36, v101, v24, vcc
	v_cmp_o_f32_e32 vcc, v58, v58
	v_add3_u32 v27, v56, v27, s0
	v_lshrrev_b32_e32 v26, 16, v26
	v_cndmask_b32_e32 v58, v101, v25, vcc
	v_cmp_o_f32_e32 vcc, v57, v57
	v_bfe_u32 v24, v55, 16, 1
	v_lshrrev_b32_e32 v27, 16, v27
	v_cndmask_b32_e32 v57, v101, v26, vcc
	v_cmp_o_f32_e32 vcc, v56, v56
	v_bfe_u32 v25, v54, 16, 1
	v_add3_u32 v24, v55, v24, s0
	v_cndmask_b32_e32 v56, v101, v27, vcc
	v_bfe_u32 v26, v53, 16, 1
	v_add3_u32 v25, v54, v25, s0
	v_lshrrev_b32_e32 v24, 16, v24
	v_cmp_o_f32_e32 vcc, v55, v55
	v_bfe_u32 v27, v52, 16, 1
	v_add3_u32 v26, v53, v26, s0
	v_lshrrev_b32_e32 v25, 16, v25
	v_cndmask_b32_e32 v34, v101, v24, vcc
	v_cmp_o_f32_e32 vcc, v54, v54
	v_add3_u32 v27, v52, v27, s0
	v_lshrrev_b32_e32 v26, 16, v26
	v_cndmask_b32_e32 v39, v101, v25, vcc
	v_cmp_o_f32_e32 vcc, v53, v53
	v_bfe_u32 v24, v51, 16, 1
	v_lshrrev_b32_e32 v27, 16, v27
	v_cndmask_b32_e32 v53, v101, v26, vcc
	v_cmp_o_f32_e32 vcc, v52, v52
	v_bfe_u32 v25, v50, 16, 1
	v_add3_u32 v24, v51, v24, s0
	v_cndmask_b32_e32 v52, v101, v27, vcc
	v_bfe_u32 v26, v49, 16, 1
	v_add3_u32 v25, v50, v25, s0
	v_lshrrev_b32_e32 v24, 16, v24
	v_cmp_o_f32_e32 vcc, v51, v51
	v_bfe_u32 v27, v48, 16, 1
	v_add3_u32 v26, v49, v26, s0
	v_lshrrev_b32_e32 v25, 16, v25
	v_cndmask_b32_e32 v32, v101, v24, vcc
	v_cmp_o_f32_e32 vcc, v50, v50
	v_add3_u32 v27, v48, v27, s0
	v_lshrrev_b32_e32 v26, 16, v26
	v_cndmask_b32_e32 v37, v101, v25, vcc
	v_cmp_o_f32_e32 vcc, v49, v49
	v_bfe_u32 v24, v47, 16, 1
	v_lshrrev_b32_e32 v27, 16, v27
	v_cndmask_b32_e32 v49, v101, v26, vcc
	v_cmp_o_f32_e32 vcc, v48, v48
	v_bfe_u32 v25, v46, 16, 1
	v_bfe_u32 v26, v45, 16, 1
	v_add3_u32 v24, v47, v24, s0
	v_cndmask_b32_e32 v48, v101, v27, vcc
	v_add3_u32 v26, v45, v26, s0
	v_add3_u32 v25, v46, v25, s0
	v_lshrrev_b32_e32 v24, 16, v24
	v_cmp_o_f32_e32 vcc, v47, v47
	v_bfe_u32 v27, v44, 16, 1
	v_lshrrev_b32_e32 v25, 16, v25
	v_lshrrev_b32_e32 v33, 16, v26
	v_cndmask_b32_e32 v26, v101, v24, vcc
	v_cmp_o_f32_e32 vcc, v46, v46
	v_add3_u32 v27, v44, v27, s0
	v_bfe_u32 v24, v43, 16, 1
	v_cndmask_b32_e32 v35, v101, v25, vcc
	v_cmp_o_f32_e32 vcc, v45, v45
	v_lshrrev_b32_e32 v27, 16, v27
	v_bfe_u32 v25, v42, 16, 1
	v_cndmask_b32_e32 v45, v101, v33, vcc
	v_cmp_o_f32_e32 vcc, v44, v44
	v_add3_u32 v24, v43, v24, s0
	v_bfe_u32 v33, v40, 16, 1
	v_cndmask_b32_e32 v44, v101, v27, vcc
	v_bfe_u32 v27, v41, 16, 1
	v_add3_u32 v25, v42, v25, s0
	v_lshrrev_b32_e32 v24, 16, v24
	v_cmp_o_f32_e32 vcc, v43, v43
	v_add3_u32 v33, v40, v33, s0
	v_add3_u32 v27, v41, v27, s0
	v_lshrrev_b32_e32 v38, 16, v25
	v_cndmask_b32_e32 v25, v101, v24, vcc
	v_cmp_o_f32_e32 vcc, v42, v42
	v_lshrrev_b32_e32 v27, 16, v27
	v_lshrrev_b32_e32 v46, 16, v33
	v_cndmask_b32_e32 v33, v101, v38, vcc
	v_cmp_o_f32_e32 vcc, v41, v41
	v_bfe_u32 v24, v31, 16, 1
	v_add3_u32 v24, v31, v24, s0
	v_cndmask_b32_e32 v38, v101, v27, vcc
	v_cmp_o_f32_e32 vcc, v40, v40
	v_bfe_u32 v27, v30, 16, 1
	v_bfe_u32 v41, v29, 16, 1
	v_cndmask_b32_e32 v40, v101, v46, vcc
	v_add3_u32 v27, v30, v27, s0
	v_lshrrev_b32_e32 v24, 16, v24
	v_cmp_o_f32_e32 vcc, v31, v31
	v_add3_u32 v41, v29, v41, s0
	v_lshrrev_b32_e32 v27, 16, v27
	v_cndmask_b32_e32 v24, v101, v24, vcc
	v_cmp_o_f32_e32 vcc, v30, v30
	v_bfe_u32 v42, v28, 16, 1
	v_lshrrev_b32_e32 v41, 16, v41
	v_cndmask_b32_e32 v27, v101, v27, vcc
	v_cmp_o_f32_e32 vcc, v29, v29
	v_add3_u32 v42, v28, v42, s0
	v_lshrrev_b32_e32 v42, 16, v42
	v_cndmask_b32_e32 v29, v101, v41, vcc
	v_cmp_o_f32_e32 vcc, v28, v28
	v_bfe_u32 v28, v23, 16, 1
	v_bfe_u32 v31, v22, 16, 1
	v_add3_u32 v28, v23, v28, s0
	v_cndmask_b32_e32 v30, v101, v42, vcc
	v_bfe_u32 v41, v21, 16, 1
	v_add3_u32 v31, v22, v31, s0
	v_lshrrev_b32_e32 v28, 16, v28
	v_cmp_o_f32_e32 vcc, v23, v23
	v_add3_u32 v41, v21, v41, s0
	v_lshrrev_b32_e32 v31, 16, v31
	v_cndmask_b32_e32 v23, v101, v28, vcc
	v_cmp_o_f32_e32 vcc, v22, v22
	v_bfe_u32 v42, v20, 16, 1
	v_lshrrev_b32_e32 v41, 16, v41
	v_cndmask_b32_e32 v22, v101, v31, vcc
	v_cmp_o_f32_e32 vcc, v21, v21
	v_add3_u32 v42, v20, v42, s0
	v_lshrrev_b32_e32 v42, 16, v42
	v_cndmask_b32_e32 v21, v101, v41, vcc
	v_cmp_o_f32_e32 vcc, v20, v20
	v_bfe_u32 v20, v19, 16, 1
	v_bfe_u32 v31, v18, 16, 1
	v_add3_u32 v20, v19, v20, s0
	v_cndmask_b32_e32 v28, v101, v42, vcc
	v_bfe_u32 v41, v17, 16, 1
	v_add3_u32 v31, v18, v31, s0
	v_lshrrev_b32_e32 v20, 16, v20
	v_cmp_o_f32_e32 vcc, v19, v19
	v_bfe_u32 v42, v16, 16, 1
	v_add3_u32 v41, v17, v41, s0
	v_lshrrev_b32_e32 v31, 16, v31
	v_cndmask_b32_e32 v19, v101, v20, vcc
	v_cmp_o_f32_e32 vcc, v18, v18
	v_add3_u32 v42, v16, v42, s0
	v_lshrrev_b32_e32 v41, 16, v41
	v_cndmask_b32_e32 v18, v101, v31, vcc
	v_cmp_o_f32_e32 vcc, v17, v17
	v_bfe_u32 v20, v15, 16, 1
	v_lshrrev_b32_e32 v42, 16, v42
	v_cndmask_b32_e32 v17, v101, v41, vcc
	v_cmp_o_f32_e32 vcc, v16, v16
	v_bfe_u32 v31, v14, 16, 1
	v_add3_u32 v20, v15, v20, s0
	v_cndmask_b32_e32 v16, v101, v42, vcc
	v_bfe_u32 v41, v13, 16, 1
	v_add3_u32 v31, v14, v31, s0
	v_lshrrev_b32_e32 v20, 16, v20
	v_cmp_o_f32_e32 vcc, v15, v15
	v_bfe_u32 v42, v12, 16, 1
	v_add3_u32 v41, v13, v41, s0
	v_lshrrev_b32_e32 v31, 16, v31
	v_cndmask_b32_e32 v15, v101, v20, vcc
	v_cmp_o_f32_e32 vcc, v14, v14
	v_add3_u32 v42, v12, v42, s0
	v_lshrrev_b32_e32 v41, 16, v41
	v_cndmask_b32_e32 v14, v101, v31, vcc
	v_cmp_o_f32_e32 vcc, v13, v13
	v_bfe_u32 v20, v11, 16, 1
	v_lshrrev_b32_e32 v42, 16, v42
	v_cndmask_b32_e32 v13, v101, v41, vcc
	v_cmp_o_f32_e32 vcc, v12, v12
	v_bfe_u32 v31, v10, 16, 1
	v_add3_u32 v20, v11, v20, s0
	v_cndmask_b32_e32 v12, v101, v42, vcc
	v_bfe_u32 v41, v9, 16, 1
	v_add3_u32 v31, v10, v31, s0
	v_lshrrev_b32_e32 v20, 16, v20
	v_cmp_o_f32_e32 vcc, v11, v11
	v_bfe_u32 v42, v8, 16, 1
	v_add3_u32 v41, v9, v41, s0
	v_lshrrev_b32_e32 v31, 16, v31
	v_cndmask_b32_e32 v11, v101, v20, vcc
	v_cmp_o_f32_e32 vcc, v10, v10
	v_add3_u32 v42, v8, v42, s0
	v_lshrrev_b32_e32 v41, 16, v41
	v_cndmask_b32_e32 v10, v101, v31, vcc
	v_cmp_o_f32_e32 vcc, v9, v9
	v_bfe_u32 v20, v7, 16, 1
	v_lshrrev_b32_e32 v42, 16, v42
	v_cndmask_b32_e32 v9, v101, v41, vcc
	v_cmp_o_f32_e32 vcc, v8, v8
	v_bfe_u32 v31, v6, 16, 1
	v_add3_u32 v20, v7, v20, s0
	v_cndmask_b32_e32 v8, v101, v42, vcc
	v_bfe_u32 v41, v5, 16, 1
	v_add3_u32 v31, v6, v31, s0
	v_lshrrev_b32_e32 v20, 16, v20
	v_cmp_o_f32_e32 vcc, v7, v7
	v_bfe_u32 v42, v4, 16, 1
	v_add3_u32 v41, v5, v41, s0
	v_lshrrev_b32_e32 v31, 16, v31
	v_cndmask_b32_e32 v7, v101, v20, vcc
	v_cmp_o_f32_e32 vcc, v6, v6
	v_add3_u32 v42, v4, v42, s0
	v_lshrrev_b32_e32 v41, 16, v41
	v_cndmask_b32_e32 v6, v101, v31, vcc
	v_cmp_o_f32_e32 vcc, v5, v5
	v_bfe_u32 v20, v3, 16, 1
	v_lshrrev_b32_e32 v42, 16, v42
	v_cndmask_b32_e32 v5, v101, v41, vcc
	v_cmp_o_f32_e32 vcc, v4, v4
	v_bfe_u32 v31, v2, 16, 1
	v_add3_u32 v20, v3, v20, s0
	v_cndmask_b32_e32 v4, v101, v42, vcc
	v_bfe_u32 v41, v1, 16, 1
	v_add3_u32 v31, v2, v31, s0
	v_lshrrev_b32_e32 v20, 16, v20
	v_cmp_o_f32_e32 vcc, v3, v3
	v_bfe_u32 v42, v0, 16, 1
	v_add3_u32 v41, v1, v41, s0
	v_lshrrev_b32_e32 v31, 16, v31
	v_cndmask_b32_e32 v3, v101, v20, vcc
	v_cmp_o_f32_e32 vcc, v2, v2
	v_add3_u32 v42, v0, v42, s0
	v_lshrrev_b32_e32 v41, 16, v41
	v_cndmask_b32_e32 v2, v101, v31, vcc
	v_cmp_o_f32_e32 vcc, v1, v1
	v_add_u32_e32 v20, v98, v96
	v_lshrrev_b32_e32 v42, 16, v42
	v_cndmask_b32_e32 v1, v101, v41, vcc
	v_cmp_o_f32_e32 vcc, v0, v0
	v_add_u32_e32 v43, v99, v20
	s_mov_b32 s0, 0xca00
	v_cndmask_b32_e32 v0, v101, v42, vcc
	v_cmp_gt_i32_e32 vcc, s0, v43
	s_and_b64 s[0:1], vcc, s[16:17]
	v_lshlrev_b32_e32 v41, 2, v97
	s_add_u32 s2, s10, s2
	v_or_b32_e32 v42, s4, v41
	s_addc_u32 s3, s11, s3
	s_lshl_b32 s4, s43, 1
	s_add_u32 s20, s2, s4
	v_or_b32_e32 v31, s33, v41
	v_add_u32_e32 v41, v20, v100
	v_mul_lo_u32 v46, s34, v42
	s_addc_u32 s2, s3, 0
	s_and_b32 s3, s34, 0x3fff
	v_add_u32_e32 v47, v46, v41
	v_not_b32_e32 v20, -2.0
	s_lshl_b32 s3, s3, 16
	s_and_b32 s2, s2, 0xffff
	s_or_b32 s2, s3, s2
	v_cndmask_b32_e64 v47, v20, v47, s[0:1]
	s_or_b32 s21, s2, 2.0
	v_lshlrev_b32_e32 v47, 1, v47
	buffer_store_short v93, v47, s[20:23], 0 offen
	s_movk_i32 s0, 0x3a7f
	v_or_b32_e32 v47, 1, v42
	v_cmp_gt_u32_e64 s[10:11], s0, v31
	v_mul_lo_u32 v47, s34, v47
	v_add_u32_e32 v50, v47, v41
	s_and_b64 s[0:1], vcc, s[10:11]
	v_cndmask_b32_e64 v50, v20, v50, s[0:1]
	v_lshlrev_b32_e32 v50, 1, v50
	buffer_store_short v91, v50, s[20:23], 0 offen
	s_movk_i32 s0, 0x3a7e
	v_or_b32_e32 v50, 2, v42
	v_cmp_gt_u32_e64 s[12:13], s0, v31
	v_mul_lo_u32 v50, s34, v50
	v_add_u32_e32 v51, v50, v41
	s_and_b64 s[0:1], vcc, s[12:13]
	v_cndmask_b32_e64 v51, v20, v51, s[0:1]
	v_lshlrev_b32_e32 v51, 1, v51
	buffer_store_short v84, v51, s[20:23], 0 offen
	s_movk_i32 s0, 0x3a7d
	v_or_b32_e32 v51, 3, v42
	v_cmp_gt_u32_e64 s[14:15], s0, v31
	v_mul_lo_u32 v51, s34, v51
	v_add_u32_e32 v54, v51, v41
	s_and_b64 s[0:1], vcc, s[14:15]
	v_cndmask_b32_e64 v54, v20, v54, s[0:1]
	v_lshlrev_b32_e32 v54, 1, v54
	s_mov_b32 s0, 0xc9f0
	buffer_store_short v88, v54, s[20:23], 0 offen
	v_cmp_gt_i32_e64 s[0:1], s0, v43
	v_add_u32_e32 v54, 16, v41
	v_add_u32_e32 v55, v46, v54
	s_and_b64 s[2:3], s[0:1], s[16:17]
	v_cndmask_b32_e64 v55, v20, v55, s[2:3]
	v_lshlrev_b32_e32 v55, 1, v55
	buffer_store_short v92, v55, s[20:23], 0 offen
	v_add_u32_e32 v55, v47, v54
	s_and_b64 s[2:3], s[0:1], s[10:11]
	v_cndmask_b32_e64 v55, v20, v55, s[2:3]
	v_lshlrev_b32_e32 v55, 1, v55
	buffer_store_short v87, v55, s[20:23], 0 offen
	v_add_u32_e32 v55, v50, v54
	s_and_b64 s[2:3], s[0:1], s[12:13]
	v_cndmask_b32_e64 v55, v20, v55, s[2:3]
	v_lshlrev_b32_e32 v55, 1, v55
	buffer_store_short v83, v55, s[20:23], 0 offen
	v_add_u32_e32 v55, v51, v54
	s_and_b64 s[2:3], s[0:1], s[14:15]
	v_cndmask_b32_e64 v55, v20, v55, s[2:3]
	v_lshlrev_b32_e32 v55, 1, v55
	s_mov_b32 s2, 0xc9e0
	buffer_store_short v79, v55, s[20:23], 0 offen
	v_cmp_gt_i32_e64 s[2:3], s2, v43
	v_add_u32_e32 v55, 32, v41
	v_add_u32_e32 v59, v46, v55
	s_and_b64 s[4:5], s[2:3], s[16:17]
	v_cndmask_b32_e64 v59, v20, v59, s[4:5]
	v_lshlrev_b32_e32 v59, 1, v59
	buffer_store_short v90, v59, s[20:23], 0 offen
	v_add_u32_e32 v59, v47, v55
	s_and_b64 s[4:5], s[2:3], s[10:11]
	v_cndmask_b32_e64 v59, v20, v59, s[4:5]
	v_lshlrev_b32_e32 v59, 1, v59
	buffer_store_short v86, v59, s[20:23], 0 offen
	v_add_u32_e32 v59, v50, v55
	s_and_b64 s[4:5], s[2:3], s[12:13]
	v_cndmask_b32_e64 v59, v20, v59, s[4:5]
	v_lshlrev_b32_e32 v59, 1, v59
	buffer_store_short v82, v59, s[20:23], 0 offen
	v_add_u32_e32 v59, v51, v55
	s_and_b64 s[4:5], s[2:3], s[14:15]
	v_cndmask_b32_e64 v59, v20, v59, s[4:5]
	v_lshlrev_b32_e32 v59, 1, v59
	s_mov_b32 s4, 0xc9d0
	buffer_store_short v78, v59, s[20:23], 0 offen
	v_cmp_gt_i32_e64 s[4:5], s4, v43
	v_add_u32_e32 v59, 48, v41
	v_add_u32_e32 v78, v46, v59
	s_and_b64 s[6:7], s[4:5], s[16:17]
	v_cndmask_b32_e64 v78, v20, v78, s[6:7]
	v_lshlrev_b32_e32 v78, 1, v78
	buffer_store_short v89, v78, s[20:23], 0 offen
	v_add_u32_e32 v78, v47, v59
	s_and_b64 s[6:7], s[4:5], s[10:11]
	v_cndmask_b32_e64 v78, v20, v78, s[6:7]
	v_lshlrev_b32_e32 v78, 1, v78
	buffer_store_short v85, v78, s[20:23], 0 offen
	v_add_u32_e32 v78, v50, v59
	s_and_b64 s[6:7], s[4:5], s[12:13]
	v_cndmask_b32_e64 v78, v20, v78, s[6:7]
	v_lshlrev_b32_e32 v78, 1, v78
	buffer_store_short v81, v78, s[20:23], 0 offen
	v_add_u32_e32 v78, v51, v59
	s_and_b64 s[6:7], s[4:5], s[14:15]
	v_cndmask_b32_e64 v78, v20, v78, s[6:7]
	v_lshlrev_b32_e32 v78, 1, v78
	s_mov_b32 s6, 0xc9c0
	buffer_store_short v77, v78, s[20:23], 0 offen
	v_cmp_gt_i32_e64 s[6:7], s6, v43
	v_add_u32_e32 v77, 64, v41
	v_add_u32_e32 v78, v46, v77
	s_and_b64 s[8:9], s[6:7], s[16:17]
	v_cndmask_b32_e64 v78, v20, v78, s[8:9]
	v_lshlrev_b32_e32 v78, 1, v78
	buffer_store_short v113, v78, s[20:23], 0 offen
	v_add_u32_e32 v78, v47, v77
	s_and_b64 s[8:9], s[6:7], s[10:11]
	v_cndmask_b32_e64 v78, v20, v78, s[8:9]
	v_lshlrev_b32_e32 v78, 1, v78
	buffer_store_short v111, v78, s[20:23], 0 offen
	v_add_u32_e32 v78, v50, v77
	s_and_b64 s[8:9], s[6:7], s[12:13]
	v_cndmask_b32_e64 v78, v20, v78, s[8:9]
	v_lshlrev_b32_e32 v78, 1, v78
	buffer_store_short v80, v78, s[20:23], 0 offen
	v_add_u32_e32 v78, v51, v77
	s_and_b64 s[8:9], s[6:7], s[14:15]
	v_cndmask_b32_e64 v78, v20, v78, s[8:9]
	s_mov_b32 s8, 0xc9b0
	v_cmp_gt_i32_e64 s[8:9], s8, v43
	v_add_u32_e32 v43, 0x50, v41
	v_add_u32_e32 v46, v46, v43
	s_and_b64 s[16:17], s[8:9], s[16:17]
	v_cndmask_b32_e64 v46, v20, v46, s[16:17]
	v_lshlrev_b32_e32 v78, 1, v78
	v_lshlrev_b32_e32 v46, 1, v46
	buffer_store_short v76, v78, s[20:23], 0 offen
	buffer_store_short v112, v46, s[20:23], 0 offen
	v_add_u32_e32 v46, v47, v43
	s_and_b64 s[10:11], s[8:9], s[10:11]
	v_cndmask_b32_e64 v46, v20, v46, s[10:11]
	v_lshlrev_b32_e32 v46, 1, v46
	buffer_store_short v109, v46, s[20:23], 0 offen
	v_add_u32_e32 v46, v50, v43
	s_and_b64 s[10:11], s[8:9], s[12:13]
	v_cndmask_b32_e64 v46, v20, v46, s[10:11]
	v_lshlrev_b32_e32 v46, 1, v46
	buffer_store_short v108, v46, s[20:23], 0 offen
	v_add_u32_e32 v46, v51, v43
	s_and_b64 s[10:11], s[8:9], s[14:15]
	v_cndmask_b32_e64 v46, v20, v46, s[10:11]
	v_lshlrev_b32_e32 v46, 1, v46
	buffer_store_short v105, v46, s[20:23], 0 offen
	s_cmpk_lt_u32 s33, 0x3a70
	v_or_b32_e32 v46, 16, v42
	s_cselect_b64 s[18:19], -1, 0
	v_mul_lo_u32 v46, s34, v46
	v_add_u32_e32 v47, v46, v41
	s_and_b64 s[10:11], vcc, s[18:19]
	v_cndmask_b32_e64 v47, v20, v47, s[10:11]
	v_lshlrev_b32_e32 v47, 1, v47
	buffer_store_short v110, v47, s[20:23], 0 offen
	s_movk_i32 s10, 0x3a6f
	v_or_b32_e32 v47, 17, v42
	v_cmp_gt_u32_e64 s[10:11], s10, v31
	v_mul_lo_u32 v47, s34, v47
	v_add_u32_e32 v50, v47, v41
	s_and_b64 s[12:13], vcc, s[10:11]
	v_cndmask_b32_e64 v50, v20, v50, s[12:13]
	v_lshlrev_b32_e32 v50, 1, v50
	buffer_store_short v107, v50, s[20:23], 0 offen
	s_movk_i32 s12, 0x3a6e
	v_or_b32_e32 v50, 18, v42
	v_cmp_gt_u32_e64 s[12:13], s12, v31
	v_mul_lo_u32 v50, s34, v50
	v_add_u32_e32 v51, v50, v41
	s_and_b64 s[14:15], vcc, s[12:13]
	v_cndmask_b32_e64 v51, v20, v51, s[14:15]
	v_lshlrev_b32_e32 v51, 1, v51
	buffer_store_short v103, v51, s[20:23], 0 offen
	s_movk_i32 s14, 0x3a6d
	v_or_b32_e32 v51, 19, v42
	v_cmp_gt_u32_e64 s[14:15], s14, v31
	v_mul_lo_u32 v51, s34, v51
	v_add_u32_e32 v76, v51, v41
	s_and_b64 s[16:17], vcc, s[14:15]
	v_cndmask_b32_e64 v76, v20, v76, s[16:17]
	v_lshlrev_b32_e32 v76, 1, v76
	buffer_store_short v95, v76, s[20:23], 0 offen
	v_add_u32_e32 v76, v46, v54
	s_and_b64 s[16:17], s[0:1], s[18:19]
	v_cndmask_b32_e64 v76, v20, v76, s[16:17]
	v_lshlrev_b32_e32 v76, 1, v76
	buffer_store_short v106, v76, s[20:23], 0 offen
	v_add_u32_e32 v76, v47, v54
	s_and_b64 s[16:17], s[0:1], s[10:11]
	v_cndmask_b32_e64 v76, v20, v76, s[16:17]
	v_lshlrev_b32_e32 v76, 1, v76
	buffer_store_short v104, v76, s[20:23], 0 offen
	v_add_u32_e32 v76, v50, v54
	s_and_b64 s[16:17], s[0:1], s[12:13]
	v_cndmask_b32_e64 v76, v20, v76, s[16:17]
	v_lshlrev_b32_e32 v76, 1, v76
	buffer_store_short v102, v76, s[20:23], 0 offen
	v_add_u32_e32 v76, v51, v54
	s_and_b64 s[16:17], s[0:1], s[14:15]
	v_cndmask_b32_e64 v76, v20, v76, s[16:17]
	v_lshlrev_b32_e32 v76, 1, v76
	buffer_store_short v94, v76, s[20:23], 0 offen
	v_add_u32_e32 v76, v46, v55
	s_and_b64 s[16:17], s[2:3], s[18:19]
	v_cndmask_b32_e64 v76, v20, v76, s[16:17]
	v_lshlrev_b32_e32 v76, 1, v76
	buffer_store_short v72, v76, s[20:23], 0 offen
	v_add_u32_e32 v72, v47, v55
	s_and_b64 s[16:17], s[2:3], s[10:11]
	v_cndmask_b32_e64 v72, v20, v72, s[16:17]
	v_lshlrev_b32_e32 v72, 1, v72
	buffer_store_short v73, v72, s[20:23], 0 offen
	v_add_u32_e32 v72, v50, v55
	s_and_b64 s[16:17], s[2:3], s[12:13]
	v_cndmask_b32_e64 v72, v20, v72, s[16:17]
	v_lshlrev_b32_e32 v72, 1, v72
	buffer_store_short v74, v72, s[20:23], 0 offen
	v_add_u32_e32 v72, v51, v55
	s_and_b64 s[16:17], s[2:3], s[14:15]
	v_cndmask_b32_e64 v72, v20, v72, s[16:17]
	v_lshlrev_b32_e32 v72, 1, v72
	buffer_store_short v75, v72, s[20:23], 0 offen
	v_add_u32_e32 v72, v46, v59
	s_and_b64 s[16:17], s[4:5], s[18:19]
	v_cndmask_b32_e64 v72, v20, v72, s[16:17]
	v_lshlrev_b32_e32 v72, 1, v72
	buffer_store_short v68, v72, s[20:23], 0 offen
	v_add_u32_e32 v68, v47, v59
	s_and_b64 s[16:17], s[4:5], s[10:11]
	v_cndmask_b32_e64 v68, v20, v68, s[16:17]
	v_lshlrev_b32_e32 v68, 1, v68
	buffer_store_short v69, v68, s[20:23], 0 offen
	v_add_u32_e32 v68, v50, v59
	s_and_b64 s[16:17], s[4:5], s[12:13]
	v_cndmask_b32_e64 v68, v20, v68, s[16:17]
	v_lshlrev_b32_e32 v68, 1, v68
	buffer_store_short v70, v68, s[20:23], 0 offen
	v_add_u32_e32 v68, v51, v59
	s_and_b64 s[16:17], s[4:5], s[14:15]
	v_cndmask_b32_e64 v68, v20, v68, s[16:17]
	v_lshlrev_b32_e32 v68, 1, v68
	buffer_store_short v71, v68, s[20:23], 0 offen
	v_add_u32_e32 v68, v46, v77
	s_and_b64 s[16:17], s[6:7], s[18:19]
	v_cndmask_b32_e64 v68, v20, v68, s[16:17]
	v_lshlrev_b32_e32 v68, 1, v68
	buffer_store_short v64, v68, s[20:23], 0 offen
	v_add_u32_e32 v64, v47, v77
	s_and_b64 s[16:17], s[6:7], s[10:11]
	v_cndmask_b32_e64 v64, v20, v64, s[16:17]
	v_lshlrev_b32_e32 v64, 1, v64
	buffer_store_short v65, v64, s[20:23], 0 offen
	v_add_u32_e32 v64, v50, v77
	s_and_b64 s[16:17], s[6:7], s[12:13]
	v_cndmask_b32_e64 v64, v20, v64, s[16:17]
	v_lshlrev_b32_e32 v64, 1, v64
	buffer_store_short v66, v64, s[20:23], 0 offen
	v_add_u32_e32 v64, v51, v77
	s_and_b64 s[16:17], s[6:7], s[14:15]
	v_cndmask_b32_e64 v64, v20, v64, s[16:17]
	v_add_u32_e32 v46, v46, v43
	s_and_b64 s[16:17], s[8:9], s[18:19]
	v_cndmask_b32_e64 v46, v20, v46, s[16:17]
	v_lshlrev_b32_e32 v64, 1, v64
	v_lshlrev_b32_e32 v46, 1, v46
	buffer_store_short v67, v64, s[20:23], 0 offen
	buffer_store_short v60, v46, s[20:23], 0 offen
	v_add_u32_e32 v46, v47, v43
	s_and_b64 s[10:11], s[8:9], s[10:11]
	v_cndmask_b32_e64 v46, v20, v46, s[10:11]
	v_lshlrev_b32_e32 v46, 1, v46
	buffer_store_short v61, v46, s[20:23], 0 offen
	v_add_u32_e32 v46, v50, v43
	s_and_b64 s[10:11], s[8:9], s[12:13]
	v_cndmask_b32_e64 v46, v20, v46, s[10:11]
	v_lshlrev_b32_e32 v46, 1, v46
	buffer_store_short v62, v46, s[20:23], 0 offen
	v_add_u32_e32 v46, v51, v43
	s_and_b64 s[10:11], s[8:9], s[14:15]
	v_cndmask_b32_e64 v46, v20, v46, s[10:11]
	v_lshlrev_b32_e32 v46, 1, v46
	buffer_store_short v63, v46, s[20:23], 0 offen
	s_cmpk_lt_u32 s33, 0x3a60
	v_or_b32_e32 v46, 32, v42
	s_cselect_b64 s[18:19], -1, 0
	v_mul_lo_u32 v46, s34, v46
	v_add_u32_e32 v47, v46, v41
	s_and_b64 s[10:11], vcc, s[18:19]
	v_cndmask_b32_e64 v47, v20, v47, s[10:11]
	v_lshlrev_b32_e32 v47, 1, v47
	buffer_store_short v56, v47, s[20:23], 0 offen
	s_movk_i32 s10, 0x3a5f
	v_or_b32_e32 v47, 33, v42
	v_cmp_gt_u32_e64 s[10:11], s10, v31
	v_mul_lo_u32 v47, s34, v47
	v_add_u32_e32 v50, v47, v41
	s_and_b64 s[12:13], vcc, s[10:11]
	v_cndmask_b32_e64 v50, v20, v50, s[12:13]
	v_lshlrev_b32_e32 v50, 1, v50
	buffer_store_short v57, v50, s[20:23], 0 offen
	s_movk_i32 s12, 0x3a5e
	v_or_b32_e32 v50, 34, v42
	v_cmp_gt_u32_e64 s[12:13], s12, v31
	v_mul_lo_u32 v50, s34, v50
	v_add_u32_e32 v51, v50, v41
	s_and_b64 s[14:15], vcc, s[12:13]
	v_cndmask_b32_e64 v51, v20, v51, s[14:15]
	v_lshlrev_b32_e32 v51, 1, v51
	buffer_store_short v58, v51, s[20:23], 0 offen
	s_movk_i32 s14, 0x3a5d
	v_or_b32_e32 v51, 35, v42
	v_cmp_gt_u32_e64 s[14:15], s14, v31
	v_mul_lo_u32 v51, s34, v51
	v_add_u32_e32 v56, v51, v41
	s_and_b64 s[16:17], vcc, s[14:15]
	v_cndmask_b32_e64 v56, v20, v56, s[16:17]
	v_lshlrev_b32_e32 v56, 1, v56
	buffer_store_short v36, v56, s[20:23], 0 offen
	v_add_u32_e32 v36, v46, v54
	s_and_b64 s[16:17], s[0:1], s[18:19]
	v_cndmask_b32_e64 v36, v20, v36, s[16:17]
	v_lshlrev_b32_e32 v36, 1, v36
	buffer_store_short v52, v36, s[20:23], 0 offen
	v_add_u32_e32 v36, v47, v54
	s_and_b64 s[16:17], s[0:1], s[10:11]
	v_cndmask_b32_e64 v36, v20, v36, s[16:17]
	v_lshlrev_b32_e32 v36, 1, v36
	buffer_store_short v53, v36, s[20:23], 0 offen
	v_add_u32_e32 v36, v50, v54
	s_and_b64 s[16:17], s[0:1], s[12:13]
	v_cndmask_b32_e64 v36, v20, v36, s[16:17]
	v_lshlrev_b32_e32 v36, 1, v36
	buffer_store_short v39, v36, s[20:23], 0 offen
	v_add_u32_e32 v36, v51, v54
	s_and_b64 s[16:17], s[0:1], s[14:15]
	v_cndmask_b32_e64 v36, v20, v36, s[16:17]
	v_lshlrev_b32_e32 v36, 1, v36
	buffer_store_short v34, v36, s[20:23], 0 offen
	v_add_u32_e32 v34, v46, v55
	s_and_b64 s[16:17], s[2:3], s[18:19]
	v_cndmask_b32_e64 v34, v20, v34, s[16:17]
	v_lshlrev_b32_e32 v34, 1, v34
	buffer_store_short v48, v34, s[20:23], 0 offen
	v_add_u32_e32 v34, v47, v55
	s_and_b64 s[16:17], s[2:3], s[10:11]
	v_cndmask_b32_e64 v34, v20, v34, s[16:17]
	v_lshlrev_b32_e32 v34, 1, v34
	buffer_store_short v49, v34, s[20:23], 0 offen
	v_add_u32_e32 v34, v50, v55
	s_and_b64 s[16:17], s[2:3], s[12:13]
	v_cndmask_b32_e64 v34, v20, v34, s[16:17]
	v_lshlrev_b32_e32 v34, 1, v34
	buffer_store_short v37, v34, s[20:23], 0 offen
	v_add_u32_e32 v34, v51, v55
	s_and_b64 s[16:17], s[2:3], s[14:15]
	v_cndmask_b32_e64 v34, v20, v34, s[16:17]
	v_lshlrev_b32_e32 v34, 1, v34
	buffer_store_short v32, v34, s[20:23], 0 offen
	v_add_u32_e32 v32, v46, v59
	s_and_b64 s[16:17], s[4:5], s[18:19]
	v_cndmask_b32_e64 v32, v20, v32, s[16:17]
	v_lshlrev_b32_e32 v32, 1, v32
	buffer_store_short v44, v32, s[20:23], 0 offen
	v_add_u32_e32 v32, v47, v59
	s_and_b64 s[16:17], s[4:5], s[10:11]
	v_cndmask_b32_e64 v32, v20, v32, s[16:17]
	v_lshlrev_b32_e32 v32, 1, v32
	buffer_store_short v45, v32, s[20:23], 0 offen
	v_add_u32_e32 v32, v50, v59
	s_and_b64 s[16:17], s[4:5], s[12:13]
	v_cndmask_b32_e64 v32, v20, v32, s[16:17]
	v_lshlrev_b32_e32 v32, 1, v32
	buffer_store_short v35, v32, s[20:23], 0 offen
	v_add_u32_e32 v32, v51, v59
	s_and_b64 s[16:17], s[4:5], s[14:15]
	v_cndmask_b32_e64 v32, v20, v32, s[16:17]
	v_lshlrev_b32_e32 v32, 1, v32
	buffer_store_short v26, v32, s[20:23], 0 offen
	v_add_u32_e32 v26, v46, v77
	s_and_b64 s[16:17], s[6:7], s[18:19]
	v_cndmask_b32_e64 v26, v20, v26, s[16:17]
	v_lshlrev_b32_e32 v26, 1, v26
	buffer_store_short v40, v26, s[20:23], 0 offen
	v_add_u32_e32 v26, v47, v77
	s_and_b64 s[16:17], s[6:7], s[10:11]
	v_cndmask_b32_e64 v26, v20, v26, s[16:17]
	v_lshlrev_b32_e32 v26, 1, v26
	buffer_store_short v38, v26, s[20:23], 0 offen
	v_add_u32_e32 v26, v50, v77
	s_and_b64 s[16:17], s[6:7], s[12:13]
	v_cndmask_b32_e64 v26, v20, v26, s[16:17]
	v_lshlrev_b32_e32 v26, 1, v26
	buffer_store_short v33, v26, s[20:23], 0 offen
	v_add_u32_e32 v26, v51, v77
	s_and_b64 s[16:17], s[6:7], s[14:15]
	v_cndmask_b32_e64 v26, v20, v26, s[16:17]
	v_lshlrev_b32_e32 v26, 1, v26
	buffer_store_short v25, v26, s[20:23], 0 offen
	v_add_u32_e32 v25, v46, v43
	s_and_b64 s[16:17], s[8:9], s[18:19]
	v_cndmask_b32_e64 v25, v20, v25, s[16:17]
	v_lshlrev_b32_e32 v25, 1, v25
	buffer_store_short v30, v25, s[20:23], 0 offen
	v_add_u32_e32 v25, v47, v43
	s_and_b64 s[10:11], s[8:9], s[10:11]
	v_cndmask_b32_e64 v25, v20, v25, s[10:11]
	v_lshlrev_b32_e32 v25, 1, v25
	buffer_store_short v29, v25, s[20:23], 0 offen
	v_add_u32_e32 v25, v50, v43
	s_and_b64 s[10:11], s[8:9], s[12:13]
	v_cndmask_b32_e64 v25, v20, v25, s[10:11]
	v_lshlrev_b32_e32 v25, 1, v25
	buffer_store_short v27, v25, s[20:23], 0 offen
	v_add_u32_e32 v25, v51, v43
	s_and_b64 s[10:11], s[8:9], s[14:15]
	v_cndmask_b32_e64 v25, v20, v25, s[10:11]
	v_lshlrev_b32_e32 v25, 1, v25
	buffer_store_short v24, v25, s[20:23], 0 offen
	s_cmpk_lt_u32 s33, 0x3a50
	v_or_b32_e32 v24, 48, v42
	s_cselect_b64 s[16:17], -1, 0
	v_mul_lo_u32 v24, s34, v24
	v_add_u32_e32 v25, v24, v41
	s_and_b64 s[10:11], vcc, s[16:17]
	v_cndmask_b32_e64 v25, v20, v25, s[10:11]
	v_lshlrev_b32_e32 v25, 1, v25
	buffer_store_short v28, v25, s[20:23], 0 offen
	s_movk_i32 s10, 0x3a4f
	v_or_b32_e32 v25, 49, v42
	v_cmp_gt_u32_e64 s[10:11], s10, v31
	v_mul_lo_u32 v25, s34, v25
	v_add_u32_e32 v26, v25, v41
	s_and_b64 s[12:13], vcc, s[10:11]
	v_cndmask_b32_e64 v26, v20, v26, s[12:13]
	v_lshlrev_b32_e32 v26, 1, v26
	buffer_store_short v21, v26, s[20:23], 0 offen
	s_movk_i32 s12, 0x3a4e
	v_or_b32_e32 v21, 50, v42
	v_cmp_gt_u32_e64 s[12:13], s12, v31
	v_mul_lo_u32 v21, s34, v21
	v_add_u32_e32 v26, v21, v41
	s_and_b64 s[14:15], vcc, s[12:13]
	v_cndmask_b32_e64 v26, v20, v26, s[14:15]
	v_lshlrev_b32_e32 v26, 1, v26
	buffer_store_short v22, v26, s[20:23], 0 offen
	s_movk_i32 s14, 0x3a4d
	v_or_b32_e32 v22, 51, v42
	v_cmp_gt_u32_e64 s[14:15], s14, v31
	v_mul_lo_u32 v22, s34, v22
	v_add_u32_e32 v26, v22, v41
	s_and_b64 vcc, vcc, s[14:15]
	v_cndmask_b32_e32 v26, v20, v26, vcc
	v_lshlrev_b32_e32 v26, 1, v26
	buffer_store_short v23, v26, s[20:23], 0 offen
	v_add_u32_e32 v23, v24, v54
	s_and_b64 vcc, s[0:1], s[16:17]
	v_cndmask_b32_e32 v23, v20, v23, vcc
	v_lshlrev_b32_e32 v23, 1, v23
	buffer_store_short v16, v23, s[20:23], 0 offen
	v_add_u32_e32 v16, v25, v54
	s_and_b64 vcc, s[0:1], s[10:11]
	v_cndmask_b32_e32 v16, v20, v16, vcc
	v_lshlrev_b32_e32 v16, 1, v16
	buffer_store_short v17, v16, s[20:23], 0 offen
	v_add_u32_e32 v16, v21, v54
	s_and_b64 vcc, s[0:1], s[12:13]
	v_cndmask_b32_e32 v16, v20, v16, vcc
	v_lshlrev_b32_e32 v16, 1, v16
	buffer_store_short v18, v16, s[20:23], 0 offen
	v_add_u32_e32 v16, v22, v54
	s_and_b64 vcc, s[0:1], s[14:15]
	v_cndmask_b32_e32 v16, v20, v16, vcc
	v_lshlrev_b32_e32 v16, 1, v16
	buffer_store_short v19, v16, s[20:23], 0 offen
	v_add_u32_e32 v16, v24, v55
	s_and_b64 vcc, s[2:3], s[16:17]
	v_cndmask_b32_e32 v16, v20, v16, vcc
	v_lshlrev_b32_e32 v16, 1, v16
	buffer_store_short v12, v16, s[20:23], 0 offen
	v_add_u32_e32 v12, v25, v55
	s_and_b64 vcc, s[2:3], s[10:11]
	v_cndmask_b32_e32 v12, v20, v12, vcc
	v_lshlrev_b32_e32 v12, 1, v12
	buffer_store_short v13, v12, s[20:23], 0 offen
	v_add_u32_e32 v12, v21, v55
	s_and_b64 vcc, s[2:3], s[12:13]
	v_cndmask_b32_e32 v12, v20, v12, vcc
	v_lshlrev_b32_e32 v12, 1, v12
	buffer_store_short v14, v12, s[20:23], 0 offen
	v_add_u32_e32 v12, v22, v55
	s_and_b64 vcc, s[2:3], s[14:15]
	v_cndmask_b32_e32 v12, v20, v12, vcc
	v_lshlrev_b32_e32 v12, 1, v12
	buffer_store_short v15, v12, s[20:23], 0 offen
	v_add_u32_e32 v12, v24, v59
	s_and_b64 vcc, s[4:5], s[16:17]
	v_cndmask_b32_e32 v12, v20, v12, vcc
	v_lshlrev_b32_e32 v12, 1, v12
	buffer_store_short v8, v12, s[20:23], 0 offen
	v_add_u32_e32 v8, v25, v59
	s_and_b64 vcc, s[4:5], s[10:11]
	v_cndmask_b32_e32 v8, v20, v8, vcc
	v_lshlrev_b32_e32 v8, 1, v8
	buffer_store_short v9, v8, s[20:23], 0 offen
	v_add_u32_e32 v8, v21, v59
	s_and_b64 vcc, s[4:5], s[12:13]
	v_cndmask_b32_e32 v8, v20, v8, vcc
	v_lshlrev_b32_e32 v8, 1, v8
	buffer_store_short v10, v8, s[20:23], 0 offen
	v_add_u32_e32 v8, v22, v59
	s_and_b64 vcc, s[4:5], s[14:15]
	v_cndmask_b32_e32 v8, v20, v8, vcc
	v_lshlrev_b32_e32 v8, 1, v8
	buffer_store_short v11, v8, s[20:23], 0 offen
	v_add_u32_e32 v8, v24, v77
	s_and_b64 vcc, s[6:7], s[16:17]
	v_cndmask_b32_e32 v8, v20, v8, vcc
	v_lshlrev_b32_e32 v8, 1, v8
	buffer_store_short v4, v8, s[20:23], 0 offen
	v_add_u32_e32 v4, v25, v77
	s_and_b64 vcc, s[6:7], s[10:11]
	v_cndmask_b32_e32 v4, v20, v4, vcc
	v_lshlrev_b32_e32 v4, 1, v4
	buffer_store_short v5, v4, s[20:23], 0 offen
	v_add_u32_e32 v4, v21, v77
	s_and_b64 vcc, s[6:7], s[12:13]
	v_cndmask_b32_e32 v4, v20, v4, vcc
	v_lshlrev_b32_e32 v4, 1, v4
	buffer_store_short v6, v4, s[20:23], 0 offen
	v_add_u32_e32 v4, v22, v77
	s_and_b64 vcc, s[6:7], s[14:15]
	v_cndmask_b32_e32 v4, v20, v4, vcc
	v_lshlrev_b32_e32 v4, 1, v4
	buffer_store_short v7, v4, s[20:23], 0 offen
	v_add_u32_e32 v4, v24, v43
	s_and_b64 vcc, s[8:9], s[16:17]
	v_cndmask_b32_e32 v4, v20, v4, vcc
	v_lshlrev_b32_e32 v4, 1, v4
	buffer_store_short v0, v4, s[20:23], 0 offen
	v_add_u32_e32 v0, v25, v43
	s_and_b64 vcc, s[8:9], s[10:11]
	v_cndmask_b32_e32 v0, v20, v0, vcc
	v_lshlrev_b32_e32 v0, 1, v0
	buffer_store_short v1, v0, s[20:23], 0 offen
	v_add_u32_e32 v0, v21, v43
	s_and_b64 vcc, s[8:9], s[12:13]
	v_cndmask_b32_e32 v0, v20, v0, vcc
	v_lshlrev_b32_e32 v0, 1, v0
	buffer_store_short v2, v0, s[20:23], 0 offen
	v_add_u32_e32 v0, v22, v43
	s_and_b64 vcc, s[8:9], s[14:15]
	v_cndmask_b32_e32 v0, v20, v0, vcc
	v_lshlrev_b32_e32 v0, 1, v0
	buffer_store_short v3, v0, s[20:23], 0 offen
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wave_mxfp4_static_gemm_256x192x256_14976x51712x7680
		.amdhsa_group_segment_fixed_size 114688
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
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 1
		.amdhsa_next_free_vgpr 208
		.amdhsa_next_free_sgpr 96
		.amdhsa_accum_offset 208
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
	.size	wave_mxfp4_static_gemm_256x192x256_14976x51712x7680, .Lfunc_end0-wave_mxfp4_static_gemm_256x192x256_14976x51712x7680

	.set wave_mxfp4_static_gemm_256x192x256_14976x51712x7680.num_vgpr, 208
	.set wave_mxfp4_static_gemm_256x192x256_14976x51712x7680.num_agpr, 0
	.set wave_mxfp4_static_gemm_256x192x256_14976x51712x7680.numbered_sgpr, 54
	.set wave_mxfp4_static_gemm_256x192x256_14976x51712x7680.num_named_barrier, 0
	.set wave_mxfp4_static_gemm_256x192x256_14976x51712x7680.private_seg_size, 0
	.set wave_mxfp4_static_gemm_256x192x256_14976x51712x7680.uses_vcc, 1
	.set wave_mxfp4_static_gemm_256x192x256_14976x51712x7680.uses_flat_scratch, 0
	.set wave_mxfp4_static_gemm_256x192x256_14976x51712x7680.has_dyn_sized_stack, 0
	.set wave_mxfp4_static_gemm_256x192x256_14976x51712x7680.has_recursion, 0
	.set wave_mxfp4_static_gemm_256x192x256_14976x51712x7680.has_indirect_call, 0
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.text
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .actual_access:  read_only
        .address_space:  generic
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  generic
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  generic
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  generic
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  generic
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
    .group_segment_fixed_size: 114688
    .kernarg_segment_align: 8
    .kernarg_segment_size: 80
    .max_flat_workgroup_size: 512
    .name:           wave_mxfp4_static_gemm_256x192x256_14976x51712x7680
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 256
      - 2
      - 1
    .sgpr_count:     60
    .sgpr_spill_count: 0
    .symbol:         wave_mxfp4_static_gemm_256x192x256_14976x51712x7680.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     208
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 0
...

	.end_amdgpu_metadata
