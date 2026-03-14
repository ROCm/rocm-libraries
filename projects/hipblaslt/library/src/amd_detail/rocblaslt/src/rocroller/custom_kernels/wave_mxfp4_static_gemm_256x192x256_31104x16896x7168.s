; To reproduce the .rocmasm from .optimized.ll, run:
; llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -mattr='-fma-mix-insts' -O3 <.optimized.ll> -o <out.rocmasm>

	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.text
	.globl	wave_mxfp4_static_gemm_256x192x256_31104x16896x7168
	.p2align	8
	.type	wave_mxfp4_static_gemm_256x192x256_31104x16896x7168,@function
wave_mxfp4_static_gemm_256x192x256_31104x16896x7168:
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
.LBB0_0:
	v_and_b32_e32 v96, 0x3ff, v0
	v_bfe_u32 v2, v0, 10, 10
	v_lshrrev_b32_e32 v4, 6, v96
	v_lshlrev_b32_e32 v0, 5, v2
	v_lshl_or_b32 v1, v4, 3, v0
	s_mov_b64 s[24:25], s[2:3]
	v_readfirstlane_b32 s2, v1
	v_lshrrev_b32_e32 v1, 3, v96
	v_or_b32_e32 v6, v1, v0
	s_lshl_b32 s16, s16, 8
	v_or_b32_e32 v5, s16, v6
	v_bitop3_b32 v7, v1, 7, v96 bitop3:0x48
	v_lshlrev_b32_e32 v100, 4, v7
	v_mul_u32_u24_e32 v101, 0xe00, v5
	s_and_b32 s3, s25, 0xffff
	s_lshl_b32 s34, s2, 7
	s_or_b32 s25, s3, 0x4e000000
	s_mov_b32 s27, 0x27000
	s_mov_b32 s26, 0x7ffffffe
	v_or_b32_e32 v3, v101, v100
	s_mov_b32 m0, s34
	s_or_b32 s35, s34, 0x2000
	buffer_load_dwordx4 v3, s[24:27], 0 offen lds
	v_add_u32_e32 v3, 0x38000, v3
	s_mov_b32 m0, s35
	v_or_b32_e32 v8, 0x80, v5
	buffer_load_dwordx4 v3, s[24:27], 0 offen lds
	v_mul_u32_u24_e32 v3, 0xe00, v8
	s_movk_i32 s40, 0x7980
	s_mov_b64 s[20:21], s[6:7]
	v_or_b32_e32 v9, v3, v100
	v_bfrev_b32_e32 v102, -2
	v_cmp_gt_u32_e64 s[6:7], s40, v8
	s_or_b32 s41, s34, 0x4000
	s_mov_b32 m0, s41
	v_cndmask_b32_e64 v8, v102, v9, s[6:7]
	buffer_load_dwordx4 v8, s[24:27], 0 offen lds
	v_or_b32_e32 v8, 0xc0, v5
	v_mul_u32_u24_e32 v5, 0xe00, v8
	v_or_b32_e32 v9, v5, v100
	v_cmp_gt_u32_e64 s[2:3], s40, v8
	s_or_b32 s42, s34, 0x6000
	s_mov_b32 m0, s42
	v_cndmask_b32_e64 v8, v102, v9, s[2:3]
	buffer_load_dwordx4 v8, s[24:27], 0 offen lds
	s_mul_i32 s33, s17, 0xc0
	v_lshrrev_b32_e32 v8, 7, v96
	v_and_or_b32 v9, v6, 48, s33
	v_lshlrev_b32_e32 v103, 4, v1
	v_lshlrev_b32_e32 v6, 8, v8
	v_sub_u32_e32 v10, v103, v6
	v_lshlrev_b32_e32 v7, 8, v7
	s_movk_i32 s12, 0xe00
	v_add_u32_e32 v10, v7, v10
	s_and_b32 s13, s21, 0xffff
	s_or_b32 s21, s13, 0x4e000000
	s_mov_b32 s22, s26
	s_mov_b32 s23, s27
	v_mad_u32_u24 v9, v9, s12, v10
	s_add_i32 m0, s34, 0x10000
	v_bfe_u32 v98, v96, 4, 2
	buffer_load_dwordx4 v9, s[20:23], 0 offen lds
	v_lshlrev_b32_e32 v9, 4, v8
	v_or3_b32 v9, v9, s33, v0
	v_mad_u32_u24 v9, v9, s12, v10
	v_add_u32_e32 v10, 0x38000, v9
	s_add_i32 m0, s35, 0x10000
	v_add_u32_e32 v9, 0x70000, v9
	buffer_load_dwordx4 v10, s[20:23], 0 offen lds
	s_add_i32 m0, s41, 0x10000
	v_lshlrev_b32_e32 v10, 4, v98
	buffer_load_dwordx4 v9, s[20:23], 0 offen lds
	v_lshrrev_b32_e32 v9, 4, v96
	v_mul_i32_i24_e32 v97, -16, v9
	v_add3_u32 v122, v10, v97, v96
	v_ashrrev_i32_e32 v10, 31, v122
	s_mul_i32 s15, s15, s16
	s_mul_hi_u32 s12, s14, s16
	v_xor_b32_e32 v11, v10, v122
	s_mov_b32 s43, 0x92492493
	s_add_i32 s15, s12, s15
	s_mul_i32 s22, s14, s16
	s_load_dwordx2 s[18:19], s[0:1], 0x40
	v_mul_hi_i32 v12, v11, s43
	s_add_u32 s28, s4, s22
	v_add_u32_e32 v11, v12, v11
	s_addc_u32 s4, s5, s15
	s_and_b32 s5, s14, 0x3fff
	v_lshrrev_b32_e32 v12, 31, v11
	v_ashrrev_i32_e32 v11, 5, v11
	s_bitset1_b32 s5, 14
	v_add_u32_e32 v11, v11, v12
	s_and_b32 s4, s4, 0xffff
	s_lshl_b32 s5, s5, 16
	v_xor_b32_e32 v13, v11, v10
	v_and_b32_e32 v18, 0xc0, v96
	v_and_b32_e32 v10, 56, v96
	s_or_b32 s29, s4, s5
	s_movk_i32 s4, 0x60
	v_add_u32_e32 v15, v13, v18
	v_mov_b32_e32 v11, 0xffffff20
	v_cmp_eq_u32_e32 vcc, 56, v10
	v_mad_u32_u24 v13, v2, s4, v13
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s4, s19, s33
	s_mul_hi_u32 s5, s18, s33
	v_cndmask_b32_e32 v14, 0, v11, vcc
	v_lshlrev_b32_e32 v11, 2, v96
	v_lshlrev_b32_e32 v12, 6, v9
	s_add_i32 s5, s5, s4
	s_mul_i32 s4, s18, s33
	v_lshlrev_b32_e32 v10, 6, v98
	v_sub_u32_e32 v16, v11, v12
	s_add_u32 s36, s8, s4
	v_add_u32_e32 v19, s16, v15
	v_add3_u32 v14, v16, v10, v14
	s_addc_u32 s4, s9, s5
	s_and_b32 s5, s18, 0x3fff
	v_mad_u64_u32 v[16:17], s[12:13], v15, s14, v[14:15]
	v_cmp_gt_i32_e32 vcc, s40, v19
	s_movk_i32 s44, 0x7960
	s_bitset1_b32 s5, 14
	s_mov_b32 s30, s26
	s_mov_b32 s31, s27
	v_cndmask_b32_e32 v15, v102, v16, vcc
	v_lshl_add_u32 v16, s14, 5, v16
	v_cmp_gt_i32_e32 vcc, s44, v19
	s_and_b32 s4, s4, 0xffff
	s_lshl_b32 s5, s5, 16
	v_cndmask_b32_e32 v16, v102, v16, vcc
	buffer_load_dword v118, v15, s[28:31], 0 offen
	buffer_load_dword v117, v16, s[28:31], 0 offen
	s_or_b32 s37, s4, s5
	v_mad_u64_u32 v[14:15], s[4:5], s18, v13, v[14:15]
	s_lshl_b32 s4, s18, 5
	s_mov_b32 s38, s26
	s_mov_b32 s39, s27
	v_add_u32_e32 v13, s4, v14
	v_add_u32_e32 v15, s4, v13
	buffer_load_dword v121, v14, s[36:39], 0 offen
	buffer_load_dword v120, v13, s[36:39], 0 offen
	buffer_load_dword v119, v15, s[36:39], 0 offen
	v_cmp_eq_u32_e64 s[4:5], 0, v2
	s_mov_b32 s9, 0
	s_mov_b32 s19, 0x10000
	v_readfirstlane_b32 s8, v18
	v_mul_u32_u24_e32 v99, 0x60, v2
	s_and_b64 vcc, exec, s[4:5]
	s_barrier
	s_waitcnt vmcnt(0)
	s_cbranch_vccnz .LBB0_2
	s_barrier
.LBB0_2:
	v_and_b32_e32 v13, 7, v96
	v_lshlrev_b32_e32 v15, 7, v96
	v_lshlrev_b32_e32 v9, 11, v9
	v_bitop3_b32 v14, v98, v96, 7 bitop3:0x78
	v_sub_u32_e32 v9, v15, v9
	v_bitop3_b32 v13, v98, v13, 4 bitop3:0x36
	s_load_dwordx2 s[12:13], s[0:1], 0x48
	v_lshl_add_u32 v4, v4, 13, v9
	v_lshlrev_b32_e32 v14, 4, v14
	s_movk_i32 s0, 0x3000
	v_lshlrev_b32_e32 v13, 4, v13
	s_mul_i32 s1, s17, 0xa8000
	s_mov_b32 s17, 0x1c000
	v_or_b32_e32 v104, v4, v14
	v_mul_lo_u32 v15, v2, s0
	v_or_b32_e32 v105, v13, v4
	v_add_u32_e32 v4, v10, v11
	v_mul_lo_u32 v2, v2, s17
	s_mov_b32 s17, 0xdf00
	v_add_u32_e32 v0, v1, v0
	v_sub_u32_e32 v109, v4, v12
	v_mul_lo_u32 v4, v8, s17
	v_lshrrev_b32_e32 v0, 4, v0
	s_mov_b32 s17, 0xe000
	v_mov_b32_e32 v1, s1
	v_mad_u32_u24 v0, v0, s17, v1
	v_add_u32_e32 v2, s1, v2
	v_or_b32_e32 v0, v0, v7
	v_add_u32_e32 v9, v9, v15
	v_add_u32_e32 v2, v2, v4
	v_sub_u32_e32 v0, v0, v6
	v_mov_b32_e32 v52, 0
	v_or_b32_e32 v106, v9, v14
	v_or_b32_e32 v107, v13, v9
	s_or_b32 s15, s8, s16
	v_add_u32_e32 v108, 64, v99
	v_add_u32_e32 v110, 32, v109
	v_add_u32_e32 v111, 32, v99
	s_add_i32 s0, s8, 32
	v_sub_u32_e32 v112, -9, v122
	v_add_u32_e32 v113, v2, v7
	v_add_u32_e32 v114, 0x800, v0
	v_add_u32_e32 v115, 0x80, v5
	v_add_u32_e32 v116, 0x80, v3
	s_mov_b32 s47, 0x8000
	s_mov_b32 s48, 0x16000
	s_mov_b32 s1, 0
	s_mov_b32 s22, s26
	s_mov_b32 s23, s27
	s_movk_i32 s17, 0xffc0
	s_movk_i32 s45, 0xe0
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
	s_mov_b32 s46, s9
	s_mov_b32 s9, s47
	s_mov_b32 s47, s19
	s_mov_b32 s19, s48
	v_add_u32_e32 v123, v101, v100
	v_add_u32_e32 v124, 0x80, v123
	s_add_i32 m0, s9, s34
	s_barrier
	buffer_load_dwordx4 v124, s[24:27], 0 offen lds
	v_add_u32_e32 v123, 0x38080, v123
	s_add_i32 m0, s9, s35
	s_nop 0
	buffer_load_dwordx4 v123, s[24:27], 0 offen lds
	v_add_u32_e32 v123, v100, v116
	v_cndmask_b32_e64 v123, v102, v123, s[6:7]
	s_add_i32 m0, s9, s41
	s_nop 0
	buffer_load_dwordx4 v123, s[24:27], 0 offen lds
	v_add_u32_e32 v123, v100, v115
	v_cndmask_b32_e64 v123, v102, v123, s[2:3]
	s_add_i32 m0, s9, s42
	s_nop 0
	buffer_load_dwordx4 v123, s[24:27], 0 offen lds
	v_add_u32_e32 v123, v114, v103
	s_add_i32 m0, s48, s34
	s_nop 0
	buffer_load_dwordx4 v123, s[20:23], 0 offen lds
	v_add_u32_e32 v123, v113, v103
	v_add_u32_e32 v124, 0x38800, v123
	s_add_i32 m0, s48, s35
	v_add_u32_e32 v123, 0x70800, v123
	buffer_load_dwordx4 v124, s[20:23], 0 offen lds
	s_add_i32 m0, s48, s41
	s_nop 0
	buffer_load_dwordx4 v123, s[20:23], 0 offen lds
	v_add_u32_e32 v128, 64, v122
	v_subrev_u32_e32 v123, 56, v112
	v_cmp_gt_i32_e32 vcc, s17, v122
	v_add_u32_e32 v126, 8, v122
	s_nop 0
	v_cndmask_b32_e32 v123, v128, v123, vcc
	v_mul_hi_i32 v124, v123, s43
	v_add_u32_e32 v123, v124, v123
	v_lshrrev_b32_e32 v124, 31, v123
	v_ashrrev_i32_e32 v123, 5, v123
	v_add_u32_e32 v123, v123, v124
	v_cndmask_b32_e64 v124, 0, -1, vcc
	v_cmp_gt_i32_e32 vcc, -8, v122
	v_xor_b32_e32 v124, v123, v124
	v_add_u32_e32 v125, s8, v124
	v_cndmask_b32_e32 v122, v126, v112, vcc
	v_mul_hi_i32 v126, v122, s43
	v_add_u32_e32 v122, v126, v122
	v_lshrrev_b32_e32 v126, 31, v122
	v_lshrrev_b32_e32 v122, 5, v122
	v_add_u32_e32 v122, v122, v126
	v_cndmask_b32_e64 v126, 0, -1, vcc
	v_xor_b32_e32 v122, v122, v126
	v_mul_lo_u32 v125, v125, s14
	v_mul_lo_u32 v122, v122, s45
	v_add_u32_e32 v127, s0, v124
	v_add_u32_e32 v123, s15, v124
	v_sub_u32_e32 v125, v125, v122
	v_add_u32_e32 v126, s1, v109
	v_mul_lo_u32 v127, s14, v127
	v_add3_u32 v125, v126, v125, 32
	v_cmp_gt_i32_e32 vcc, s40, v123
	v_sub_u32_e32 v127, v127, v122
	v_add3_u32 v126, v126, v127, 32
	v_cndmask_b32_e32 v125, v102, v125, vcc
	v_cmp_gt_i32_e32 vcc, s44, v123
	s_nop 1
	v_cndmask_b32_e32 v123, v102, v126, vcc
	buffer_load_dword v127, v125, s[28:31], 0 offen
	s_nop 0
	buffer_load_dword v123, v123, s[28:31], 0 offen
	v_add_u32_e32 v125, v124, v99
	v_mul_lo_u32 v125, v125, s18
	v_add_u32_e32 v126, v111, v124
	v_add_u32_e32 v124, v108, v124
	v_sub_u32_e32 v125, v125, v122
	v_mul_lo_u32 v126, s18, v126
	v_mul_lo_u32 v124, s18, v124
	v_add3_u32 v125, v110, v125, s1
	v_sub_u32_e32 v126, v126, v122
	v_sub_u32_e32 v122, v124, v122
	v_add3_u32 v129, v110, v126, s1
	v_add3_u32 v122, v110, v122, s1
	buffer_load_dword v126, v125, s[36:39], 0 offen
	s_nop 0
	buffer_load_dword v125, v129, s[36:39], 0 offen
	buffer_load_dword v124, v122, s[36:39], 0 offen
	v_add_u32_e32 v122, s46, v104
	ds_read_b128 v[130:133], v122
	ds_read_b128 v[134:137], v122 offset:2048
	ds_read_b128 v[138:141], v122 offset:4096
	ds_read_b128 v[142:145], v122 offset:6144
	v_add_u32_e32 v122, s47, v106
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
	v_add_u32_e32 v122, s46, v105
	ds_read_b128 v[130:133], v122
	ds_read_b128 v[134:137], v122 offset:2048
	ds_read_b128 v[138:141], v122 offset:4096
	ds_read_b128 v[142:145], v122 offset:6144
	v_add_u32_e32 v122, s47, v107
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
	v_add_u32_e32 v101, 0x80, v101
	v_add_u32_e32 v115, 0x80, v115
	v_add_u32_e32 v116, 0x80, v116
	s_cmpk_lg_i32 s1, 0x1b00
	v_mov_b32_e32 v122, v128
	s_mov_b32 s48, s47
	s_mov_b32 s47, s46
	s_waitcnt vmcnt(0)
	v_mov_b32_e32 v119, v124
	v_mov_b32_e32 v120, v125
	v_mov_b32_e32 v121, v126
	v_mov_b32_e32 v117, v123
	v_mov_b32_e32 v118, v127
	s_cbranch_scc1 .LBB0_3
	s_andn2_b64 vcc, exec, s[4:5]
	s_cbranch_vccnz .LBB0_6
	s_barrier
.LBB0_6:
	v_add_u32_e32 v100, s19, v106
	v_add_u32_e32 v101, s19, v107
	s_barrier
	ds_read_b128 v[164:167], v100
	ds_read_b128 v[168:171], v100 offset:2048
	ds_read_b128 v[172:175], v101
	ds_read_b128 v[176:179], v101 offset:2048
	ds_read_b128 v[180:183], v100 offset:4096
	ds_read_b128 v[136:139], v100 offset:6144
	ds_read_b128 v[184:187], v101 offset:4096
	ds_read_b128 v[128:131], v101 offset:6144
	ds_read_b128 v[118:121], v100 offset:8192
	ds_read_b128 v[110:113], v100 offset:10240
	ds_read_b128 v[114:117], v101 offset:8192
	ds_read_b128 v[106:109], v101 offset:10240
	v_add_u32_e32 v100, s9, v104
	ds_read_b128 v[152:155], v100
	ds_read_b128 v[188:191], v100 offset:2048
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[152:155], v[164:167], v[52:55], v127, v126 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add_u32_e32 v101, s9, v105
	ds_read_b128 v[156:159], v101
	ds_read_b128 v[192:195], v101 offset:2048
	ds_read_b128 v[196:199], v100 offset:4096
	ds_read_b128 v[140:143], v100 offset:6144
	ds_read_b128 v[200:203], v101 offset:4096
	ds_read_b128 v[132:135], v101 offset:6144
	s_movk_i32 s0, 0x7fff
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[156:159], v[172:175], v[52:55], v127, v126 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mov_b32_e32 v100, 0x7fc0
	s_cmpk_lt_u32 s15, 0x7980
	s_mul_hi_u32 s1, s12, s16
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[152:155], v[180:183], v[88:91], v127, v125 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_movk_i32 s4, 0x797d
	s_nop 2
	v_bfe_u32 v101, v55, 16, 1
	v_bfe_u32 v102, v54, 16, 1
	v_add3_u32 v101, v55, v101, s0
	v_bfe_u32 v103, v53, 16, 1
	v_bfe_u32 v104, v52, 16, 1
	v_add3_u32 v102, v54, v102, s0
	v_lshrrev_b32_e32 v101, 16, v101
	v_cmp_o_f32_e32 vcc, v55, v55
	v_add3_u32 v122, v52, v104, s0
	v_add3_u32 v144, v53, v103, s0
	v_lshrrev_b32_e32 v148, 16, v102
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[156:159], v[184:187], v[88:91], v127, v125 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_nop 2
	v_cndmask_b32_e32 v88, v100, v101, vcc
	v_cmp_o_f32_e32 vcc, v54, v54
	v_lshrrev_b32_e32 v89, 16, v144
	v_mfma_scale_f32_16x16x128_f8f6f4 v[144:147], v[152:155], v[136:139], v[84:87], v127, v125 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v90, 16, v122
	s_nop 1
	v_cndmask_b32_e32 v84, v100, v148, vcc
	v_cmp_o_f32_e32 vcc, v53, v53
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[152:155], v[168:171], v[92:95], v127, v126 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e32 v85, v100, v89, vcc
	v_cmp_o_f32_e32 vcc, v52, v52
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[152:155], v[118:121], v[80:83], v127, v124 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e32 v87, v100, v90, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[148:151], v[156:159], v[114:117], v[52:55], v127, v124 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[152:155], v[110:113], v[76:79], v127, v124 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[152:155], v[156:159], v[106:109], v[52:55], v127, v124 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[188:191], v[164:167], v[72:75], v127, v126 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[156:159], v[176:179], v[92:95], v127, v126 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_nop 1
	v_bfe_u32 v72, v105, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[144:147], v[156:159], v[128:131], v[144:147], v127, v125 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[156:159], v[192:195], v[172:175], v[52:55], v127, v126 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_nop 1
	v_bfe_u32 v86, v95, 16, 1
	v_bfe_u32 v80, v94, 16, 1
	v_add3_u32 v76, v95, v86, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[188:191], v[168:171], v[68:71], v127, v126 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v81, v93, 16, 1
	v_add3_u32 v80, v94, v80, s0
	v_lshrrev_b32_e32 v76, 16, v76
	v_mfma_scale_f32_16x16x128_f8f6f4 v[160:163], v[192:195], v[176:179], v[52:55], v127, v126 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v95, v95
	v_bfe_u32 v82, v92, 16, 1
	v_add3_u32 v81, v93, v81, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[188:191], v[180:183], v[64:67], v127, v125 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v77, 16, v80
	v_cndmask_b32_e32 v76, v100, v76, vcc
	v_cmp_o_f32_e32 vcc, v94, v94
	v_add3_u32 v64, v105, v72, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[192:195], v[184:187], v[52:55], v127, v125 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_add3_u32 v82, v92, v82, s0
	v_lshrrev_b32_e32 v78, 16, v81
	v_cndmask_b32_e32 v77, v100, v77, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[188:191], v[136:139], v[60:63], v127, v125 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v93, v93
	v_bfe_u32 v68, v104, 16, 1
	v_bfe_u32 v69, v103, 16, 1
	v_bfe_u32 v70, v102, 16, 1
	v_lshrrev_b32_e32 v80, 16, v82
	v_cndmask_b32_e32 v79, v100, v78, vcc
	v_cmp_o_f32_e32 vcc, v92, v92
	v_add3_u32 v70, v102, v70, s0
	v_add3_u32 v69, v103, v69, s0
	v_add3_u32 v68, v104, v68, s0
	v_cndmask_b32_e32 v82, v100, v80, vcc
	v_lshrrev_b32_e32 v64, 16, v64
	v_lshrrev_b32_e32 v65, 16, v68
	v_lshrrev_b32_e32 v66, 16, v69
	v_lshrrev_b32_e32 v67, 16, v70
	v_cmp_o_f32_e32 vcc, v105, v105
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[192:195], v[128:131], v[52:55], v127, v125 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_bfe_u32 v60, v147, 16, 1
	v_cndmask_b32_e32 v78, v100, v64, vcc
	v_cmp_o_f32_e32 vcc, v104, v104
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[188:191], v[118:121], v[56:59], v127, v124 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e32 v80, v100, v65, vcc
	v_cmp_o_f32_e32 vcc, v103, v103
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[188:191], v[110:113], v[48:51], v127, v124 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v56, v146, 16, 1
	v_cndmask_b32_e32 v83, v100, v66, vcc
	v_cmp_o_f32_e32 vcc, v102, v102
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[196:199], v[164:167], v[44:47], v123, v126 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v57, v145, 16, 1
	v_cndmask_b32_e32 v90, v100, v67, vcc
	v_cmp_o_f32_e32 vcc, v147, v147
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[192:195], v[114:117], v[52:55], v127, v124 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_bfe_u32 v58, v144, 16, 1
	s_nop 1
	v_add3_u32 v55, v147, v60, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[196:199], v[168:171], v[40:43], v123, v126 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v54, v146, v56, s0
	v_lshrrev_b32_e32 v55, 16, v55
	v_add3_u32 v53, v145, v57, s0
	v_lshrrev_b32_e32 v54, 16, v54
	v_cndmask_b32_e32 v81, v100, v55, vcc
	v_cmp_o_f32_e32 vcc, v146, v146
	v_add3_u32 v52, v144, v58, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[192:195], v[106:109], v[48:51], v127, v124 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v86, v100, v54, vcc
	v_cmp_o_f32_e32 vcc, v145, v145
	s_nop 0
	v_lshrrev_b32_e32 v48, 16, v53
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[200:203], v[172:175], v[44:47], v123, v126 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v49, 16, v52
	v_cndmask_b32_e32 v91, v100, v48, vcc
	v_cmp_o_f32_e32 vcc, v144, v144
	v_bfe_u32 v44, v151, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[196:199], v[118:121], v[12:15], v123, v124 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v45, v150, 16, 1
	v_cndmask_b32_e32 v93, v100, v49, vcc
	v_bfe_u32 v46, v149, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[200:203], v[176:179], v[40:43], v123, v126 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_bfe_u32 v47, v148, 16, 1
	v_cmp_o_f32_e32 vcc, v151, v151
	s_nop 0
	v_add3_u32 v43, v151, v44, s0
	v_add3_u32 v42, v150, v45, s0
	v_lshrrev_b32_e32 v43, 16, v43
	v_add3_u32 v40, v148, v47, s0
	v_add3_u32 v41, v149, v46, s0
	v_lshrrev_b32_e32 v42, 16, v42
	v_cndmask_b32_e32 v89, v100, v43, vcc
	v_cmp_o_f32_e32 vcc, v150, v150
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[196:199], v[180:183], v[28:31], v123, v125 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v94, 16, v41
	v_lshrrev_b32_e32 v101, 16, v40
	v_cndmask_b32_e32 v92, v100, v42, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[200:203], v[114:117], v[12:15], v123, v124 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v149, v149
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[196:199], v[110:113], v[20:23], v123, v124 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e32 v95, v100, v94, vcc
	v_cmp_o_f32_e32 vcc, v148, v148
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[200:203], v[184:187], v[28:31], v123, v125 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e32 v104, v100, v101, vcc
	v_cmp_o_f32_e32 vcc, v155, v155
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[200:203], v[106:109], v[12:15], v123, v124 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[140:143], v[164:167], v[24:27], v123, v126 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[196:199], v[136:139], v[16:19], v123, v125 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 1
	v_bfe_u32 v24, v155, 16, 1
	v_bfe_u32 v25, v154, 16, 1
	v_add3_u32 v24, v155, v24, s0
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[132:135], v[172:175], v[12:15], v123, v126 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_bfe_u32 v26, v153, 16, 1
	v_add3_u32 v25, v154, v25, s0
	v_lshrrev_b32_e32 v24, 16, v24
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[140:143], v[168:171], v[32:35], v123, v126 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v27, v152, 16, 1
	v_add3_u32 v26, v153, v26, s0
	v_lshrrev_b32_e32 v25, 16, v25
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[140:143], v[110:113], v[0:3], v123, v124 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v27, v152, v27, s0
	v_lshrrev_b32_e32 v26, 16, v26
	v_lshrrev_b32_e32 v27, 16, v27
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[200:203], v[128:131], v[16:19], v123, v125 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_bfe_u32 v32, v160, 16, 1
	v_add3_u32 v32, v160, v32, s0
	v_lshrrev_b32_e32 v32, 16, v32
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[132:135], v[176:179], v[12:15], v123, v126 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_bfe_u32 v35, v72, 16, 1
	v_add3_u32 v35, v72, v35, s0
	v_lshrrev_b32_e32 v35, 16, v35
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[140:143], v[180:183], v[36:39], v123, v125 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 2
	v_cndmask_b32_e32 v39, v100, v24, vcc
	v_cmp_o_f32_e32 vcc, v154, v154
	v_bfe_u32 v24, v159, 16, 1
	v_add3_u32 v24, v159, v24, s0
	v_cndmask_b32_e32 v101, v100, v25, vcc
	v_cmp_o_f32_e32 vcc, v153, v153
	v_bfe_u32 v25, v158, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[132:135], v[106:109], v[0:3], v123, v124 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v105, v100, v26, vcc
	v_cmp_o_f32_e32 vcc, v152, v152
	v_bfe_u32 v26, v157, 16, 1
	v_add3_u32 v25, v158, v25, s0
	v_cndmask_b32_e32 v107, v100, v27, vcc
	v_lshrrev_b32_e32 v24, 16, v24
	v_cmp_o_f32_e32 vcc, v159, v159
	v_bfe_u32 v27, v156, 16, 1
	v_add3_u32 v26, v157, v26, s0
	v_lshrrev_b32_e32 v25, 16, v25
	v_cndmask_b32_e32 v24, v100, v24, vcc
	v_cmp_o_f32_e32 vcc, v158, v158
	v_add3_u32 v27, v156, v27, s0
	v_lshrrev_b32_e32 v26, 16, v26
	v_cndmask_b32_e32 v34, v100, v25, vcc
	v_cmp_o_f32_e32 vcc, v157, v157
	v_bfe_u32 v25, v163, 16, 1
	v_lshrrev_b32_e32 v27, 16, v27
	v_cndmask_b32_e32 v102, v100, v26, vcc
	v_cmp_o_f32_e32 vcc, v156, v156
	v_bfe_u32 v26, v162, 16, 1
	v_add3_u32 v25, v163, v25, s0
	v_cndmask_b32_e32 v106, v100, v27, vcc
	v_bfe_u32 v27, v161, 16, 1
	v_add3_u32 v26, v162, v26, s0
	v_lshrrev_b32_e32 v25, 16, v25
	v_cmp_o_f32_e32 vcc, v163, v163
	v_add3_u32 v27, v161, v27, s0
	v_lshrrev_b32_e32 v26, 16, v26
	v_cndmask_b32_e32 v25, v100, v25, vcc
	v_cmp_o_f32_e32 vcc, v162, v162
	v_lshrrev_b32_e32 v27, 16, v27
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[132:135], v[184:187], v[12:15], v123, v125 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v33, v100, v26, vcc
	v_cmp_o_f32_e32 vcc, v161, v161
	v_bfe_u32 v26, v75, 16, 1
	v_add3_u32 v26, v75, v26, s0
	v_cndmask_b32_e32 v94, v100, v27, vcc
	v_cmp_o_f32_e32 vcc, v160, v160
	v_bfe_u32 v27, v74, 16, 1
	v_add3_u32 v27, v74, v27, s0
	v_cndmask_b32_e32 v103, v100, v32, vcc
	v_bfe_u32 v32, v73, 16, 1
	v_lshrrev_b32_e32 v26, 16, v26
	v_cmp_o_f32_e32 vcc, v75, v75
	v_add3_u32 v32, v73, v32, s0
	v_lshrrev_b32_e32 v36, 16, v27
	v_cndmask_b32_e32 v27, v100, v26, vcc
	v_cmp_o_f32_e32 vcc, v74, v74
	v_lshrrev_b32_e32 v32, 16, v32
	v_bfe_u32 v26, v71, 16, 1
	v_cndmask_b32_e32 v37, v100, v36, vcc
	v_cmp_o_f32_e32 vcc, v73, v73
	v_add3_u32 v26, v71, v26, s0
	v_lshrrev_b32_e32 v26, 16, v26
	v_cndmask_b32_e32 v73, v100, v32, vcc
	v_cmp_o_f32_e32 vcc, v72, v72
	v_bfe_u32 v32, v70, 16, 1
	v_add3_u32 v32, v70, v32, s0
	v_cndmask_b32_e32 v72, v100, v35, vcc
	v_bfe_u32 v35, v69, 16, 1
	v_add3_u32 v35, v69, v35, s0
	v_cmp_o_f32_e32 vcc, v71, v71
	v_bfe_u32 v36, v68, 16, 1
	v_lshrrev_b32_e32 v32, 16, v32
	v_lshrrev_b32_e32 v38, 16, v35
	v_cndmask_b32_e32 v35, v100, v26, vcc
	v_cmp_o_f32_e32 vcc, v70, v70
	v_add3_u32 v36, v68, v36, s0
	v_bfe_u32 v26, v67, 16, 1
	v_cndmask_b32_e32 v70, v100, v32, vcc
	v_cmp_o_f32_e32 vcc, v69, v69
	v_lshrrev_b32_e32 v36, 16, v36
	v_bfe_u32 v32, v66, 16, 1
	v_cndmask_b32_e32 v69, v100, v38, vcc
	v_cmp_o_f32_e32 vcc, v68, v68
	v_add3_u32 v26, v67, v26, s0
	v_add3_u32 v32, v66, v32, s0
	v_cndmask_b32_e32 v68, v100, v36, vcc
	v_bfe_u32 v36, v65, 16, 1
	v_lshrrev_b32_e32 v26, 16, v26
	v_cmp_o_f32_e32 vcc, v67, v67
	v_bfe_u32 v38, v64, 16, 1
	v_add3_u32 v36, v65, v36, s0
	v_lshrrev_b32_e32 v32, 16, v32
	v_cndmask_b32_e32 v67, v100, v26, vcc
	v_cmp_o_f32_e32 vcc, v66, v66
	v_add3_u32 v38, v64, v38, s0
	v_lshrrev_b32_e32 v36, 16, v36
	v_cndmask_b32_e32 v66, v100, v32, vcc
	v_cmp_o_f32_e32 vcc, v65, v65
	v_bfe_u32 v26, v63, 16, 1
	v_lshrrev_b32_e32 v38, 16, v38
	v_cndmask_b32_e32 v65, v100, v36, vcc
	v_cmp_o_f32_e32 vcc, v64, v64
	v_bfe_u32 v32, v62, 16, 1
	v_add3_u32 v26, v63, v26, s0
	v_cndmask_b32_e32 v64, v100, v38, vcc
	v_bfe_u32 v36, v61, 16, 1
	v_add3_u32 v32, v62, v32, s0
	v_lshrrev_b32_e32 v26, 16, v26
	v_cmp_o_f32_e32 vcc, v63, v63
	v_bfe_u32 v38, v60, 16, 1
	v_add3_u32 v36, v61, v36, s0
	v_lshrrev_b32_e32 v32, 16, v32
	v_cndmask_b32_e32 v63, v100, v26, vcc
	v_cmp_o_f32_e32 vcc, v62, v62
	v_add3_u32 v38, v60, v38, s0
	v_lshrrev_b32_e32 v36, 16, v36
	v_cndmask_b32_e32 v62, v100, v32, vcc
	v_cmp_o_f32_e32 vcc, v61, v61
	v_bfe_u32 v26, v59, 16, 1
	v_lshrrev_b32_e32 v38, 16, v38
	v_cndmask_b32_e32 v61, v100, v36, vcc
	v_cmp_o_f32_e32 vcc, v60, v60
	v_bfe_u32 v32, v58, 16, 1
	v_add3_u32 v26, v59, v26, s0
	v_cndmask_b32_e32 v60, v100, v38, vcc
	v_bfe_u32 v36, v57, 16, 1
	v_add3_u32 v32, v58, v32, s0
	v_lshrrev_b32_e32 v26, 16, v26
	v_cmp_o_f32_e32 vcc, v59, v59
	v_bfe_u32 v38, v56, 16, 1
	v_add3_u32 v36, v57, v36, s0
	v_lshrrev_b32_e32 v32, 16, v32
	v_cndmask_b32_e32 v26, v100, v26, vcc
	v_cmp_o_f32_e32 vcc, v58, v58
	v_add3_u32 v38, v56, v38, s0
	v_lshrrev_b32_e32 v36, 16, v36
	v_cndmask_b32_e32 v58, v100, v32, vcc
	v_cmp_o_f32_e32 vcc, v57, v57
	v_bfe_u32 v32, v55, 16, 1
	v_lshrrev_b32_e32 v38, 16, v38
	v_cndmask_b32_e32 v57, v100, v36, vcc
	v_cmp_o_f32_e32 vcc, v56, v56
	v_bfe_u32 v36, v54, 16, 1
	v_add3_u32 v32, v55, v32, s0
	v_cndmask_b32_e32 v56, v100, v38, vcc
	v_bfe_u32 v38, v53, 16, 1
	v_add3_u32 v36, v54, v36, s0
	v_lshrrev_b32_e32 v32, 16, v32
	v_cmp_o_f32_e32 vcc, v55, v55
	v_bfe_u32 v59, v52, 16, 1
	v_add3_u32 v38, v53, v38, s0
	v_lshrrev_b32_e32 v36, 16, v36
	v_cndmask_b32_e32 v32, v100, v32, vcc
	v_cmp_o_f32_e32 vcc, v54, v54
	v_add3_u32 v59, v52, v59, s0
	v_lshrrev_b32_e32 v71, 16, v38
	v_cndmask_b32_e32 v38, v100, v36, vcc
	v_cmp_o_f32_e32 vcc, v53, v53
	v_bfe_u32 v36, v51, 16, 1
	v_lshrrev_b32_e32 v59, 16, v59
	v_cndmask_b32_e32 v53, v100, v71, vcc
	v_cmp_o_f32_e32 vcc, v52, v52
	v_bfe_u32 v54, v50, 16, 1
	v_add3_u32 v36, v51, v36, s0
	v_cndmask_b32_e32 v52, v100, v59, vcc
	v_bfe_u32 v55, v49, 16, 1
	v_add3_u32 v54, v50, v54, s0
	v_lshrrev_b32_e32 v36, 16, v36
	v_cmp_o_f32_e32 vcc, v51, v51
	v_bfe_u32 v59, v48, 16, 1
	v_add3_u32 v55, v49, v55, s0
	v_lshrrev_b32_e32 v54, 16, v54
	v_cndmask_b32_e32 v36, v100, v36, vcc
	v_cmp_o_f32_e32 vcc, v50, v50
	v_add3_u32 v59, v48, v59, s0
	v_lshrrev_b32_e32 v55, 16, v55
	v_cndmask_b32_e32 v50, v100, v54, vcc
	v_cmp_o_f32_e32 vcc, v49, v49
	v_bfe_u32 v51, v47, 16, 1
	v_lshrrev_b32_e32 v59, 16, v59
	v_cndmask_b32_e32 v49, v100, v55, vcc
	v_cmp_o_f32_e32 vcc, v48, v48
	v_bfe_u32 v54, v46, 16, 1
	v_add3_u32 v51, v47, v51, s0
	v_cndmask_b32_e32 v48, v100, v59, vcc
	v_bfe_u32 v55, v45, 16, 1
	v_add3_u32 v54, v46, v54, s0
	v_lshrrev_b32_e32 v51, 16, v51
	v_cmp_o_f32_e32 vcc, v47, v47
	v_bfe_u32 v59, v44, 16, 1
	v_add3_u32 v55, v45, v55, s0
	v_lshrrev_b32_e32 v54, 16, v54
	v_cndmask_b32_e32 v47, v100, v51, vcc
	v_cmp_o_f32_e32 vcc, v46, v46
	v_add3_u32 v59, v44, v59, s0
	v_lshrrev_b32_e32 v55, 16, v55
	v_cndmask_b32_e32 v46, v100, v54, vcc
	v_cmp_o_f32_e32 vcc, v45, v45
	v_bfe_u32 v51, v43, 16, 1
	v_lshrrev_b32_e32 v59, 16, v59
	v_cndmask_b32_e32 v45, v100, v55, vcc
	v_cmp_o_f32_e32 vcc, v44, v44
	v_bfe_u32 v54, v42, 16, 1
	v_add3_u32 v51, v43, v51, s0
	v_cndmask_b32_e32 v44, v100, v59, vcc
	v_bfe_u32 v55, v41, 16, 1
	v_add3_u32 v54, v42, v54, s0
	v_lshrrev_b32_e32 v51, 16, v51
	v_cmp_o_f32_e32 vcc, v43, v43
	v_bfe_u32 v59, v40, 16, 1
	v_add3_u32 v55, v41, v55, s0
	v_lshrrev_b32_e32 v54, 16, v54
	v_cndmask_b32_e32 v43, v100, v51, vcc
	v_cmp_o_f32_e32 vcc, v42, v42
	v_add3_u32 v59, v40, v59, s0
	v_lshrrev_b32_e32 v55, 16, v55
	v_cndmask_b32_e32 v42, v100, v54, vcc
	v_cmp_o_f32_e32 vcc, v41, v41
	v_bfe_u32 v51, v31, 16, 1
	v_lshrrev_b32_e32 v59, 16, v59
	v_cndmask_b32_e32 v41, v100, v55, vcc
	v_cmp_o_f32_e32 vcc, v40, v40
	v_bfe_u32 v54, v30, 16, 1
	v_add3_u32 v51, v31, v51, s0
	v_cndmask_b32_e32 v40, v100, v59, vcc
	v_bfe_u32 v55, v29, 16, 1
	v_add3_u32 v54, v30, v54, s0
	v_lshrrev_b32_e32 v51, 16, v51
	v_cmp_o_f32_e32 vcc, v31, v31
	v_bfe_u32 v59, v28, 16, 1
	v_add3_u32 v55, v29, v55, s0
	v_lshrrev_b32_e32 v54, 16, v54
	v_cndmask_b32_e32 v31, v100, v51, vcc
	v_cmp_o_f32_e32 vcc, v30, v30
	v_add3_u32 v59, v28, v59, s0
	v_lshrrev_b32_e32 v55, 16, v55
	v_cndmask_b32_e32 v30, v100, v54, vcc
	v_cmp_o_f32_e32 vcc, v29, v29
	v_bfe_u32 v51, v23, 16, 1
	v_lshrrev_b32_e32 v59, 16, v59
	v_cndmask_b32_e32 v29, v100, v55, vcc
	v_cmp_o_f32_e32 vcc, v28, v28
	v_bfe_u32 v54, v22, 16, 1
	v_add3_u32 v51, v23, v51, s0
	v_cndmask_b32_e32 v28, v100, v59, vcc
	v_bfe_u32 v55, v21, 16, 1
	v_add3_u32 v54, v22, v54, s0
	v_lshrrev_b32_e32 v51, 16, v51
	v_cmp_o_f32_e32 vcc, v23, v23
	v_bfe_u32 v59, v20, 16, 1
	v_add3_u32 v55, v21, v55, s0
	v_lshrrev_b32_e32 v54, 16, v54
	v_cndmask_b32_e32 v23, v100, v51, vcc
	v_cmp_o_f32_e32 vcc, v22, v22
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[140:143], v[136:139], v[8:11], v123, v125 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v59, v20, v59, s0
	v_lshrrev_b32_e32 v55, 16, v55
	v_cndmask_b32_e32 v22, v100, v54, vcc
	v_cmp_o_f32_e32 vcc, v21, v21
	v_bfe_u32 v51, v19, 16, 1
	v_lshrrev_b32_e32 v59, 16, v59
	v_cndmask_b32_e32 v21, v100, v55, vcc
	v_cmp_o_f32_e32 vcc, v20, v20
	v_bfe_u32 v54, v18, 16, 1
	v_add3_u32 v51, v19, v51, s0
	v_cndmask_b32_e32 v20, v100, v59, vcc
	v_bfe_u32 v55, v17, 16, 1
	v_add3_u32 v54, v18, v54, s0
	v_lshrrev_b32_e32 v51, 16, v51
	v_cmp_o_f32_e32 vcc, v19, v19
	v_bfe_u32 v59, v16, 16, 1
	v_add3_u32 v55, v17, v55, s0
	v_lshrrev_b32_e32 v54, 16, v54
	v_cndmask_b32_e32 v19, v100, v51, vcc
	v_cmp_o_f32_e32 vcc, v18, v18
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[132:135], v[128:131], v[8:11], v123, v125 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_add3_u32 v59, v16, v59, s0
	v_lshrrev_b32_e32 v55, 16, v55
	v_cndmask_b32_e32 v18, v100, v54, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[140:143], v[118:121], v[4:7], v123, v124 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v17, v17
	v_bfe_u32 v51, v15, 16, 1
	v_lshrrev_b32_e32 v59, 16, v59
	v_cndmask_b32_e32 v17, v100, v55, vcc
	v_cmp_o_f32_e32 vcc, v16, v16
	v_bfe_u32 v54, v14, 16, 1
	v_add3_u32 v51, v15, v51, s0
	v_cndmask_b32_e32 v16, v100, v59, vcc
	v_bfe_u32 v55, v13, 16, 1
	v_add3_u32 v54, v14, v54, s0
	v_lshrrev_b32_e32 v51, 16, v51
	v_cmp_o_f32_e32 vcc, v15, v15
	v_bfe_u32 v59, v12, 16, 1
	v_add3_u32 v55, v13, v55, s0
	v_lshrrev_b32_e32 v54, 16, v54
	v_cndmask_b32_e32 v15, v100, v51, vcc
	v_cmp_o_f32_e32 vcc, v14, v14
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[132:135], v[114:117], v[4:7], v123, v124 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_add3_u32 v59, v12, v59, s0
	v_lshrrev_b32_e32 v55, 16, v55
	v_cndmask_b32_e32 v14, v100, v54, vcc
	v_cmp_o_f32_e32 vcc, v13, v13
	v_bfe_u32 v51, v11, 16, 1
	v_lshrrev_b32_e32 v59, 16, v59
	v_cndmask_b32_e32 v13, v100, v55, vcc
	v_cmp_o_f32_e32 vcc, v12, v12
	v_bfe_u32 v54, v10, 16, 1
	v_add3_u32 v51, v11, v51, s0
	v_cndmask_b32_e32 v12, v100, v59, vcc
	v_bfe_u32 v55, v9, 16, 1
	v_add3_u32 v54, v10, v54, s0
	v_lshrrev_b32_e32 v51, 16, v51
	v_cmp_o_f32_e32 vcc, v11, v11
	v_bfe_u32 v59, v8, 16, 1
	v_add3_u32 v55, v9, v55, s0
	v_lshrrev_b32_e32 v54, 16, v54
	v_cndmask_b32_e32 v11, v100, v51, vcc
	v_cmp_o_f32_e32 vcc, v10, v10
	v_add3_u32 v59, v8, v59, s0
	v_lshrrev_b32_e32 v55, 16, v55
	v_cndmask_b32_e32 v10, v100, v54, vcc
	v_cmp_o_f32_e32 vcc, v9, v9
	v_bfe_u32 v51, v7, 16, 1
	v_lshrrev_b32_e32 v59, 16, v59
	v_cndmask_b32_e32 v9, v100, v55, vcc
	v_cmp_o_f32_e32 vcc, v8, v8
	v_bfe_u32 v54, v6, 16, 1
	v_add3_u32 v51, v7, v51, s0
	v_cndmask_b32_e32 v8, v100, v59, vcc
	v_bfe_u32 v55, v5, 16, 1
	v_add3_u32 v54, v6, v54, s0
	v_lshrrev_b32_e32 v51, 16, v51
	v_cmp_o_f32_e32 vcc, v7, v7
	v_bfe_u32 v59, v4, 16, 1
	v_add3_u32 v55, v5, v55, s0
	v_lshrrev_b32_e32 v54, 16, v54
	v_cndmask_b32_e32 v7, v100, v51, vcc
	v_cmp_o_f32_e32 vcc, v6, v6
	v_add3_u32 v59, v4, v59, s0
	v_lshrrev_b32_e32 v55, 16, v55
	v_cndmask_b32_e32 v6, v100, v54, vcc
	v_cmp_o_f32_e32 vcc, v5, v5
	v_bfe_u32 v51, v3, 16, 1
	v_lshrrev_b32_e32 v59, 16, v59
	v_cndmask_b32_e32 v5, v100, v55, vcc
	v_cmp_o_f32_e32 vcc, v4, v4
	v_bfe_u32 v54, v2, 16, 1
	v_add3_u32 v51, v3, v51, s0
	v_cndmask_b32_e32 v4, v100, v59, vcc
	v_bfe_u32 v55, v1, 16, 1
	v_add3_u32 v54, v2, v54, s0
	v_lshrrev_b32_e32 v51, 16, v51
	v_cmp_o_f32_e32 vcc, v3, v3
	v_bfe_u32 v59, v0, 16, 1
	v_add3_u32 v55, v1, v55, s0
	v_lshrrev_b32_e32 v54, 16, v54
	v_cndmask_b32_e32 v3, v100, v51, vcc
	v_cmp_o_f32_e32 vcc, v2, v2
	v_add3_u32 v59, v0, v59, s0
	v_lshrrev_b32_e32 v55, 16, v55
	v_cndmask_b32_e32 v2, v100, v54, vcc
	v_cmp_o_f32_e32 vcc, v1, v1
	v_lshrrev_b32_e32 v59, 16, v59
	s_mul_i32 s0, s13, s16
	v_cndmask_b32_e32 v1, v100, v55, vcc
	v_cmp_o_f32_e32 vcc, v0, v0
	v_lshlrev_b32_e32 v51, 2, v98
	v_or_b32_e32 v54, s15, v51
	v_cndmask_b32_e32 v0, v100, v59, vcc
	s_cselect_b64 vcc, -1, 0
	s_add_i32 s1, s1, s0
	s_mul_i32 s0, s12, s16
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s0, s10, s0
	s_addc_u32 s1, s11, s1
	s_lshl_b32 s2, s33, 1
	v_or_b32_e32 v51, s8, v51
	s_add_u32 s8, s0, s2
	v_add3_u32 v55, v97, v96, v99
	v_mul_lo_u32 v59, s12, v51
	s_addc_u32 s0, s1, 0
	s_and_b32 s1, s12, 0x3fff
	v_add_u32_e32 v71, v59, v55
	v_not_b32_e32 v74, -2.0
	s_lshl_b32 s1, s1, 16
	s_and_b32 s0, s0, 0xffff
	s_or_b32 s0, s1, s0
	v_cndmask_b32_e32 v71, v74, v71, vcc
	s_or_b32 s9, s0, 2.0
	s_mov_b32 s11, 0x27000
	s_mov_b32 s10, 0x7ffffffd
	v_lshlrev_b32_e32 v71, 1, v71
	buffer_store_short v87, v71, s[8:11], 0 offen
	v_or_b32_e32 v71, 1, v51
	s_movk_i32 s0, 0x797f
	v_mul_lo_u32 v71, s12, v71
	v_add_u32_e32 v75, v71, v55
	v_cmp_gt_u32_e64 s[0:1], s0, v54
	s_movk_i32 s2, 0x797e
	v_cmp_gt_u32_e64 s[2:3], s2, v54
	v_cndmask_b32_e64 v75, v74, v75, s[0:1]
	v_lshlrev_b32_e32 v75, 1, v75
	buffer_store_short v85, v75, s[8:11], 0 offen
	v_or_b32_e32 v75, 2, v51
	v_mul_lo_u32 v75, s12, v75
	v_add_u32_e32 v85, v75, v55
	v_cndmask_b32_e64 v85, v74, v85, s[2:3]
	v_lshlrev_b32_e32 v85, 1, v85
	buffer_store_short v84, v85, s[8:11], 0 offen
	v_or_b32_e32 v84, 3, v51
	v_mul_lo_u32 v84, s12, v84
	v_add_u32_e32 v85, v84, v55
	v_cmp_gt_u32_e64 s[4:5], s4, v54
	s_cmpk_lt_u32 s15, 0x7970
	s_nop 0
	v_cndmask_b32_e64 v85, v74, v85, s[4:5]
	v_lshlrev_b32_e32 v85, 1, v85
	buffer_store_short v88, v85, s[8:11], 0 offen
	v_add_u32_e32 v85, 16, v55
	v_add_u32_e32 v87, v59, v85
	v_cndmask_b32_e32 v87, v74, v87, vcc
	v_lshlrev_b32_e32 v87, 1, v87
	buffer_store_short v82, v87, s[8:11], 0 offen
	v_add_u32_e32 v82, v71, v85
	v_cndmask_b32_e64 v82, v74, v82, s[0:1]
	v_lshlrev_b32_e32 v82, 1, v82
	buffer_store_short v79, v82, s[8:11], 0 offen
	v_add_u32_e32 v79, v75, v85
	v_cndmask_b32_e64 v79, v74, v79, s[2:3]
	v_lshlrev_b32_e32 v79, 1, v79
	buffer_store_short v77, v79, s[8:11], 0 offen
	v_add_u32_e32 v77, v84, v85
	v_cndmask_b32_e64 v77, v74, v77, s[4:5]
	v_lshlrev_b32_e32 v77, 1, v77
	buffer_store_short v76, v77, s[8:11], 0 offen
	v_add_u32_e32 v76, 32, v55
	v_add_u32_e32 v77, v59, v76
	v_cndmask_b32_e32 v77, v74, v77, vcc
	v_lshlrev_b32_e32 v77, 1, v77
	buffer_store_short v90, v77, s[8:11], 0 offen
	v_add_u32_e32 v77, v71, v76
	v_cndmask_b32_e64 v77, v74, v77, s[0:1]
	v_lshlrev_b32_e32 v77, 1, v77
	buffer_store_short v83, v77, s[8:11], 0 offen
	v_add_u32_e32 v77, v75, v76
	v_cndmask_b32_e64 v77, v74, v77, s[2:3]
	v_lshlrev_b32_e32 v77, 1, v77
	buffer_store_short v80, v77, s[8:11], 0 offen
	v_add_u32_e32 v77, v84, v76
	v_cndmask_b32_e64 v77, v74, v77, s[4:5]
	v_lshlrev_b32_e32 v77, 1, v77
	buffer_store_short v78, v77, s[8:11], 0 offen
	v_add_u32_e32 v77, 48, v55
	v_add_u32_e32 v78, v59, v77
	v_cndmask_b32_e32 v78, v74, v78, vcc
	v_lshlrev_b32_e32 v78, 1, v78
	buffer_store_short v93, v78, s[8:11], 0 offen
	v_add_u32_e32 v78, v71, v77
	v_cndmask_b32_e64 v78, v74, v78, s[0:1]
	v_lshlrev_b32_e32 v78, 1, v78
	buffer_store_short v91, v78, s[8:11], 0 offen
	v_add_u32_e32 v78, v75, v77
	v_cndmask_b32_e64 v78, v74, v78, s[2:3]
	v_lshlrev_b32_e32 v78, 1, v78
	buffer_store_short v86, v78, s[8:11], 0 offen
	v_add_u32_e32 v78, v84, v77
	v_cndmask_b32_e64 v78, v74, v78, s[4:5]
	v_lshlrev_b32_e32 v78, 1, v78
	buffer_store_short v81, v78, s[8:11], 0 offen
	v_add_u32_e32 v78, 64, v55
	v_add_u32_e32 v79, v59, v78
	v_cndmask_b32_e32 v79, v74, v79, vcc
	v_lshlrev_b32_e32 v79, 1, v79
	buffer_store_short v104, v79, s[8:11], 0 offen
	v_add_u32_e32 v79, v71, v78
	v_cndmask_b32_e64 v79, v74, v79, s[0:1]
	v_lshlrev_b32_e32 v79, 1, v79
	buffer_store_short v95, v79, s[8:11], 0 offen
	v_add_u32_e32 v79, v75, v78
	v_cndmask_b32_e64 v79, v74, v79, s[2:3]
	v_lshlrev_b32_e32 v79, 1, v79
	buffer_store_short v92, v79, s[8:11], 0 offen
	v_add_u32_e32 v79, v84, v78
	v_cndmask_b32_e64 v79, v74, v79, s[4:5]
	v_lshlrev_b32_e32 v79, 1, v79
	buffer_store_short v89, v79, s[8:11], 0 offen
	v_add_u32_e32 v79, 0x50, v55
	v_add_u32_e32 v59, v59, v79
	v_cndmask_b32_e32 v59, v74, v59, vcc
	v_lshlrev_b32_e32 v59, 1, v59
	buffer_store_short v107, v59, s[8:11], 0 offen
	v_add_u32_e32 v59, v71, v79
	v_cndmask_b32_e64 v59, v74, v59, s[0:1]
	v_lshlrev_b32_e32 v59, 1, v59
	buffer_store_short v105, v59, s[8:11], 0 offen
	v_add_u32_e32 v59, v75, v79
	v_cndmask_b32_e64 v59, v74, v59, s[2:3]
	v_lshlrev_b32_e32 v59, 1, v59
	buffer_store_short v101, v59, s[8:11], 0 offen
	v_add_u32_e32 v59, v84, v79
	v_cndmask_b32_e64 v59, v74, v59, s[4:5]
	v_lshlrev_b32_e32 v59, 1, v59
	buffer_store_short v39, v59, s[8:11], 0 offen
	v_or_b32_e32 v39, 16, v51
	v_mul_lo_u32 v39, s12, v39
	v_add_u32_e32 v59, v39, v55
	s_cselect_b64 vcc, -1, 0
	v_cndmask_b32_e32 v59, v74, v59, vcc
	v_lshlrev_b32_e32 v59, 1, v59
	buffer_store_short v106, v59, s[8:11], 0 offen
	v_or_b32_e32 v59, 17, v51
	s_movk_i32 s0, 0x796f
	v_mul_lo_u32 v59, s12, v59
	v_add_u32_e32 v71, v59, v55
	v_cmp_gt_u32_e64 s[0:1], s0, v54
	s_movk_i32 s2, 0x796e
	v_cmp_gt_u32_e64 s[2:3], s2, v54
	v_cndmask_b32_e64 v71, v74, v71, s[0:1]
	v_lshlrev_b32_e32 v71, 1, v71
	buffer_store_short v102, v71, s[8:11], 0 offen
	v_or_b32_e32 v71, 18, v51
	v_mul_lo_u32 v71, s12, v71
	v_add_u32_e32 v75, v71, v55
	v_cndmask_b32_e64 v75, v74, v75, s[2:3]
	v_lshlrev_b32_e32 v75, 1, v75
	buffer_store_short v34, v75, s[8:11], 0 offen
	v_or_b32_e32 v34, 19, v51
	s_movk_i32 s4, 0x796d
	v_mul_lo_u32 v34, s12, v34
	v_add_u32_e32 v75, v34, v55
	v_cmp_gt_u32_e64 s[4:5], s4, v54
	s_cmpk_lt_u32 s15, 0x7960
	s_nop 0
	v_cndmask_b32_e64 v75, v74, v75, s[4:5]
	v_lshlrev_b32_e32 v75, 1, v75
	buffer_store_short v24, v75, s[8:11], 0 offen
	v_add_u32_e32 v24, v39, v85
	v_cndmask_b32_e32 v24, v74, v24, vcc
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v103, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v59, v85
	v_cndmask_b32_e64 v24, v74, v24, s[0:1]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v94, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v71, v85
	v_cndmask_b32_e64 v24, v74, v24, s[2:3]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v33, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v34, v85
	v_cndmask_b32_e64 v24, v74, v24, s[4:5]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v25, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v39, v76
	v_cndmask_b32_e32 v24, v74, v24, vcc
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v72, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v59, v76
	v_cndmask_b32_e64 v24, v74, v24, s[0:1]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v73, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v71, v76
	v_cndmask_b32_e64 v24, v74, v24, s[2:3]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v37, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v34, v76
	v_cndmask_b32_e64 v24, v74, v24, s[4:5]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v27, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v39, v77
	v_cndmask_b32_e32 v24, v74, v24, vcc
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v68, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v59, v77
	v_cndmask_b32_e64 v24, v74, v24, s[0:1]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v69, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v71, v77
	v_cndmask_b32_e64 v24, v74, v24, s[2:3]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v70, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v34, v77
	v_cndmask_b32_e64 v24, v74, v24, s[4:5]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v35, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v39, v78
	v_cndmask_b32_e32 v24, v74, v24, vcc
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v64, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v59, v78
	v_cndmask_b32_e64 v24, v74, v24, s[0:1]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v65, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v71, v78
	v_cndmask_b32_e64 v24, v74, v24, s[2:3]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v66, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v34, v78
	v_cndmask_b32_e64 v24, v74, v24, s[4:5]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v67, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v39, v79
	v_cndmask_b32_e32 v24, v74, v24, vcc
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v60, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v59, v79
	v_cndmask_b32_e64 v24, v74, v24, s[0:1]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v61, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v71, v79
	v_cndmask_b32_e64 v24, v74, v24, s[2:3]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v62, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v34, v79
	v_cndmask_b32_e64 v24, v74, v24, s[4:5]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v63, v24, s[8:11], 0 offen
	v_or_b32_e32 v24, 32, v51
	v_mul_lo_u32 v24, s12, v24
	v_add_u32_e32 v25, v24, v55
	s_cselect_b64 vcc, -1, 0
	v_cndmask_b32_e32 v25, v74, v25, vcc
	v_lshlrev_b32_e32 v25, 1, v25
	buffer_store_short v56, v25, s[8:11], 0 offen
	v_or_b32_e32 v25, 33, v51
	s_movk_i32 s0, 0x795f
	v_mul_lo_u32 v25, s12, v25
	v_add_u32_e32 v27, v25, v55
	v_cmp_gt_u32_e64 s[0:1], s0, v54
	s_movk_i32 s2, 0x795e
	v_cmp_gt_u32_e64 s[2:3], s2, v54
	v_cndmask_b32_e64 v27, v74, v27, s[0:1]
	v_lshlrev_b32_e32 v27, 1, v27
	buffer_store_short v57, v27, s[8:11], 0 offen
	v_or_b32_e32 v27, 34, v51
	v_mul_lo_u32 v27, s12, v27
	v_add_u32_e32 v33, v27, v55
	v_cndmask_b32_e64 v33, v74, v33, s[2:3]
	v_lshlrev_b32_e32 v33, 1, v33
	buffer_store_short v58, v33, s[8:11], 0 offen
	v_or_b32_e32 v33, 35, v51
	s_movk_i32 s4, 0x795d
	v_mul_lo_u32 v33, s12, v33
	v_add_u32_e32 v34, v33, v55
	v_cmp_gt_u32_e64 s[4:5], s4, v54
	s_cmpk_lt_u32 s15, 0x7950
	s_nop 0
	v_cndmask_b32_e64 v34, v74, v34, s[4:5]
	v_lshlrev_b32_e32 v34, 1, v34
	buffer_store_short v26, v34, s[8:11], 0 offen
	v_add_u32_e32 v26, v24, v85
	v_cndmask_b32_e32 v26, v74, v26, vcc
	v_lshlrev_b32_e32 v26, 1, v26
	buffer_store_short v52, v26, s[8:11], 0 offen
	v_add_u32_e32 v26, v25, v85
	v_cndmask_b32_e64 v26, v74, v26, s[0:1]
	v_lshlrev_b32_e32 v26, 1, v26
	buffer_store_short v53, v26, s[8:11], 0 offen
	v_add_u32_e32 v26, v27, v85
	v_cndmask_b32_e64 v26, v74, v26, s[2:3]
	v_lshlrev_b32_e32 v26, 1, v26
	buffer_store_short v38, v26, s[8:11], 0 offen
	v_add_u32_e32 v26, v33, v85
	v_cndmask_b32_e64 v26, v74, v26, s[4:5]
	v_lshlrev_b32_e32 v26, 1, v26
	buffer_store_short v32, v26, s[8:11], 0 offen
	v_add_u32_e32 v26, v24, v76
	v_cndmask_b32_e32 v26, v74, v26, vcc
	v_lshlrev_b32_e32 v26, 1, v26
	buffer_store_short v48, v26, s[8:11], 0 offen
	v_add_u32_e32 v26, v25, v76
	v_cndmask_b32_e64 v26, v74, v26, s[0:1]
	v_lshlrev_b32_e32 v26, 1, v26
	buffer_store_short v49, v26, s[8:11], 0 offen
	v_add_u32_e32 v26, v27, v76
	v_cndmask_b32_e64 v26, v74, v26, s[2:3]
	v_lshlrev_b32_e32 v26, 1, v26
	buffer_store_short v50, v26, s[8:11], 0 offen
	v_add_u32_e32 v26, v33, v76
	v_cndmask_b32_e64 v26, v74, v26, s[4:5]
	v_lshlrev_b32_e32 v26, 1, v26
	buffer_store_short v36, v26, s[8:11], 0 offen
	v_add_u32_e32 v26, v24, v77
	v_cndmask_b32_e32 v26, v74, v26, vcc
	v_lshlrev_b32_e32 v26, 1, v26
	buffer_store_short v44, v26, s[8:11], 0 offen
	v_add_u32_e32 v26, v25, v77
	v_cndmask_b32_e64 v26, v74, v26, s[0:1]
	v_lshlrev_b32_e32 v26, 1, v26
	buffer_store_short v45, v26, s[8:11], 0 offen
	v_add_u32_e32 v26, v27, v77
	v_cndmask_b32_e64 v26, v74, v26, s[2:3]
	v_lshlrev_b32_e32 v26, 1, v26
	buffer_store_short v46, v26, s[8:11], 0 offen
	v_add_u32_e32 v26, v33, v77
	v_cndmask_b32_e64 v26, v74, v26, s[4:5]
	v_lshlrev_b32_e32 v26, 1, v26
	buffer_store_short v47, v26, s[8:11], 0 offen
	v_add_u32_e32 v26, v24, v78
	v_cndmask_b32_e32 v26, v74, v26, vcc
	v_lshlrev_b32_e32 v26, 1, v26
	buffer_store_short v40, v26, s[8:11], 0 offen
	v_add_u32_e32 v26, v25, v78
	v_cndmask_b32_e64 v26, v74, v26, s[0:1]
	v_lshlrev_b32_e32 v26, 1, v26
	buffer_store_short v41, v26, s[8:11], 0 offen
	v_add_u32_e32 v26, v27, v78
	v_cndmask_b32_e64 v26, v74, v26, s[2:3]
	v_lshlrev_b32_e32 v26, 1, v26
	buffer_store_short v42, v26, s[8:11], 0 offen
	v_add_u32_e32 v26, v33, v78
	v_add_u32_e32 v24, v24, v79
	v_cndmask_b32_e64 v26, v74, v26, s[4:5]
	v_cndmask_b32_e32 v24, v74, v24, vcc
	v_lshlrev_b32_e32 v26, 1, v26
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v43, v26, s[8:11], 0 offen
	buffer_store_short v28, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v25, v79
	v_cndmask_b32_e64 v24, v74, v24, s[0:1]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v29, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v27, v79
	v_cndmask_b32_e64 v24, v74, v24, s[2:3]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v30, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v33, v79
	v_cndmask_b32_e64 v24, v74, v24, s[4:5]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v31, v24, s[8:11], 0 offen
	v_or_b32_e32 v24, 48, v51
	v_mul_lo_u32 v24, s12, v24
	v_add_u32_e32 v25, v24, v55
	s_cselect_b64 vcc, -1, 0
	v_cndmask_b32_e32 v25, v74, v25, vcc
	v_lshlrev_b32_e32 v25, 1, v25
	buffer_store_short v20, v25, s[8:11], 0 offen
	v_or_b32_e32 v20, 49, v51
	s_movk_i32 s0, 0x794f
	v_mul_lo_u32 v20, s12, v20
	v_add_u32_e32 v25, v20, v55
	v_cmp_gt_u32_e64 s[0:1], s0, v54
	s_movk_i32 s2, 0x794e
	v_cmp_gt_u32_e64 s[2:3], s2, v54
	v_cndmask_b32_e64 v25, v74, v25, s[0:1]
	v_lshlrev_b32_e32 v25, 1, v25
	buffer_store_short v21, v25, s[8:11], 0 offen
	v_or_b32_e32 v21, 50, v51
	v_mul_lo_u32 v21, s12, v21
	v_add_u32_e32 v25, v21, v55
	v_cndmask_b32_e64 v25, v74, v25, s[2:3]
	v_lshlrev_b32_e32 v25, 1, v25
	buffer_store_short v22, v25, s[8:11], 0 offen
	v_or_b32_e32 v22, 51, v51
	s_movk_i32 s4, 0x794d
	v_mul_lo_u32 v22, s12, v22
	v_add_u32_e32 v25, v22, v55
	v_cmp_gt_u32_e64 s[4:5], s4, v54
	s_nop 1
	v_cndmask_b32_e64 v25, v74, v25, s[4:5]
	v_lshlrev_b32_e32 v25, 1, v25
	buffer_store_short v23, v25, s[8:11], 0 offen
	v_add_u32_e32 v23, v24, v85
	v_cndmask_b32_e32 v23, v74, v23, vcc
	v_lshlrev_b32_e32 v23, 1, v23
	buffer_store_short v16, v23, s[8:11], 0 offen
	v_add_u32_e32 v16, v20, v85
	v_cndmask_b32_e64 v16, v74, v16, s[0:1]
	v_lshlrev_b32_e32 v16, 1, v16
	buffer_store_short v17, v16, s[8:11], 0 offen
	v_add_u32_e32 v16, v21, v85
	v_cndmask_b32_e64 v16, v74, v16, s[2:3]
	v_lshlrev_b32_e32 v16, 1, v16
	buffer_store_short v18, v16, s[8:11], 0 offen
	v_add_u32_e32 v16, v22, v85
	v_cndmask_b32_e64 v16, v74, v16, s[4:5]
	v_lshlrev_b32_e32 v16, 1, v16
	buffer_store_short v19, v16, s[8:11], 0 offen
	v_add_u32_e32 v16, v24, v76
	v_cndmask_b32_e32 v16, v74, v16, vcc
	v_lshlrev_b32_e32 v16, 1, v16
	buffer_store_short v12, v16, s[8:11], 0 offen
	v_add_u32_e32 v12, v20, v76
	v_cndmask_b32_e64 v12, v74, v12, s[0:1]
	v_lshlrev_b32_e32 v12, 1, v12
	buffer_store_short v13, v12, s[8:11], 0 offen
	v_add_u32_e32 v12, v21, v76
	v_cndmask_b32_e64 v12, v74, v12, s[2:3]
	v_lshlrev_b32_e32 v12, 1, v12
	buffer_store_short v14, v12, s[8:11], 0 offen
	v_add_u32_e32 v12, v22, v76
	v_cndmask_b32_e64 v12, v74, v12, s[4:5]
	v_lshlrev_b32_e32 v12, 1, v12
	buffer_store_short v15, v12, s[8:11], 0 offen
	v_add_u32_e32 v12, v24, v77
	v_cndmask_b32_e32 v12, v74, v12, vcc
	v_lshlrev_b32_e32 v12, 1, v12
	buffer_store_short v8, v12, s[8:11], 0 offen
	v_add_u32_e32 v8, v20, v77
	v_cndmask_b32_e64 v8, v74, v8, s[0:1]
	v_lshlrev_b32_e32 v8, 1, v8
	buffer_store_short v9, v8, s[8:11], 0 offen
	v_add_u32_e32 v8, v21, v77
	v_cndmask_b32_e64 v8, v74, v8, s[2:3]
	v_lshlrev_b32_e32 v8, 1, v8
	buffer_store_short v10, v8, s[8:11], 0 offen
	v_add_u32_e32 v8, v22, v77
	v_cndmask_b32_e64 v8, v74, v8, s[4:5]
	v_lshlrev_b32_e32 v8, 1, v8
	buffer_store_short v11, v8, s[8:11], 0 offen
	v_add_u32_e32 v8, v24, v78
	v_cndmask_b32_e32 v8, v74, v8, vcc
	v_lshlrev_b32_e32 v8, 1, v8
	buffer_store_short v4, v8, s[8:11], 0 offen
	v_add_u32_e32 v4, v20, v78
	v_cndmask_b32_e64 v4, v74, v4, s[0:1]
	v_lshlrev_b32_e32 v4, 1, v4
	buffer_store_short v5, v4, s[8:11], 0 offen
	v_add_u32_e32 v4, v21, v78
	v_cndmask_b32_e64 v4, v74, v4, s[2:3]
	v_lshlrev_b32_e32 v4, 1, v4
	buffer_store_short v6, v4, s[8:11], 0 offen
	v_add_u32_e32 v4, v22, v78
	v_cndmask_b32_e64 v4, v74, v4, s[4:5]
	v_lshlrev_b32_e32 v4, 1, v4
	buffer_store_short v7, v4, s[8:11], 0 offen
	v_add_u32_e32 v4, v24, v79
	v_cndmask_b32_e32 v4, v74, v4, vcc
	v_lshlrev_b32_e32 v4, 1, v4
	buffer_store_short v0, v4, s[8:11], 0 offen
	v_add_u32_e32 v0, v20, v79
	v_cndmask_b32_e64 v0, v74, v0, s[0:1]
	v_lshlrev_b32_e32 v0, 1, v0
	buffer_store_short v1, v0, s[8:11], 0 offen
	v_add_u32_e32 v0, v21, v79
	v_cndmask_b32_e64 v0, v74, v0, s[2:3]
	v_lshlrev_b32_e32 v0, 1, v0
	buffer_store_short v2, v0, s[8:11], 0 offen
	v_add_u32_e32 v0, v22, v79
	v_cndmask_b32_e64 v0, v74, v0, s[4:5]
	v_lshlrev_b32_e32 v0, 1, v0
	buffer_store_short v3, v0, s[8:11], 0 offen
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wave_mxfp4_static_gemm_256x192x256_31104x16896x7168
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
		.amdhsa_next_free_vgpr 204
		.amdhsa_next_free_sgpr 96
		.amdhsa_accum_offset 204
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
	.size	wave_mxfp4_static_gemm_256x192x256_31104x16896x7168, .Lfunc_end0-wave_mxfp4_static_gemm_256x192x256_31104x16896x7168

	.set wave_mxfp4_static_gemm_256x192x256_31104x16896x7168.num_vgpr, 204
	.set wave_mxfp4_static_gemm_256x192x256_31104x16896x7168.num_agpr, 0
	.set wave_mxfp4_static_gemm_256x192x256_31104x16896x7168.numbered_sgpr, 49
	.set wave_mxfp4_static_gemm_256x192x256_31104x16896x7168.num_named_barrier, 0
	.set wave_mxfp4_static_gemm_256x192x256_31104x16896x7168.private_seg_size, 0
	.set wave_mxfp4_static_gemm_256x192x256_31104x16896x7168.uses_vcc, 1
	.set wave_mxfp4_static_gemm_256x192x256_31104x16896x7168.uses_flat_scratch, 0
	.set wave_mxfp4_static_gemm_256x192x256_31104x16896x7168.has_dyn_sized_stack, 0
	.set wave_mxfp4_static_gemm_256x192x256_31104x16896x7168.has_recursion, 0
	.set wave_mxfp4_static_gemm_256x192x256_31104x16896x7168.has_indirect_call, 0
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
    .name:           wave_mxfp4_static_gemm_256x192x256_31104x16896x7168
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 256
      - 2
      - 1
    .sgpr_count:     55
    .sgpr_spill_count: 0
    .symbol:         wave_mxfp4_static_gemm_256x192x256_31104x16896x7168.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     204
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 0
...

	.end_amdgpu_metadata
