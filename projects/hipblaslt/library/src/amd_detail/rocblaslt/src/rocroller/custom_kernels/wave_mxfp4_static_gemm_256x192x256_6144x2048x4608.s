; To reproduce the .rocmasm from .optimized.ll, run:
; llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -mattr='-fma-mix-insts' -O3 <.optimized.ll> -o <out.rocmasm>

	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.text
	.globl	wave_mxfp4_static_gemm_256x192x256_6144x2048x4608
	.p2align	8
	.type	wave_mxfp4_static_gemm_256x192x256_6144x2048x4608,@function
wave_mxfp4_static_gemm_256x192x256_6144x2048x4608:
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
.LBB0_0:
	v_and_b32_e32 v96, 0x3ff, v0
	v_bfe_u32 v2, v0, 10, 10
	v_lshrrev_b32_e32 v3, 6, v96
	v_lshlrev_b32_e32 v0, 5, v2
	v_lshl_or_b32 v1, v3, 3, v0
	s_mov_b64 s[24:25], s[2:3]
	v_readfirstlane_b32 s2, v1
	v_lshrrev_b32_e32 v1, 3, v96
	v_or_b32_e32 v4, v1, v0
	s_lshl_b32 s16, s16, 8
	v_or_b32_e32 v5, s16, v4
	v_bitop3_b32 v7, v1, 7, v96 bitop3:0x48
	s_mov_b64 s[20:21], s[6:7]
	v_lshlrev_b32_e32 v102, 4, v7
	v_mul_u32_u24_e32 v103, 0x900, v5
	s_and_b32 s6, s25, 0xffff
	s_lshl_b32 s34, s2, 7
	s_or_b32 s25, s6, 0x49000000
	s_mov_b32 s27, 0x27000
	s_mov_b32 s26, 0x7ffffffe
	v_or_b32_e32 v5, v103, v102
	s_mov_b32 m0, s34
	s_or_b32 s35, s34, 0x2000
	buffer_load_dwordx4 v5, s[24:27], 0 offen lds
	v_add_u32_e32 v6, 0x24000, v5
	s_mov_b32 m0, s35
	s_or_b32 s40, s34, 0x4000
	buffer_load_dwordx4 v6, s[24:27], 0 offen lds
	v_add_u32_e32 v6, 0x48000, v5
	s_mov_b32 m0, s40
	s_or_b32 s41, s34, 0x6000
	buffer_load_dwordx4 v6, s[24:27], 0 offen lds
	s_mul_i32 s33, s17, 0xc0
	v_lshrrev_b32_e32 v6, 7, v96
	v_add_u32_e32 v5, 0x6c000, v5
	s_mov_b32 m0, s41
	v_and_or_b32 v8, v4, 48, s33
	v_lshlrev_b32_e32 v104, 4, v1
	v_lshlrev_b32_e32 v4, 8, v6
	buffer_load_dwordx4 v5, s[24:27], 0 offen lds
	v_sub_u32_e32 v9, v104, v4
	v_lshlrev_b32_e32 v5, 8, v7
	s_movk_i32 s3, 0x900
	v_add_u32_e32 v7, v5, v9
	s_and_b32 s2, s21, 0xffff
	s_or_b32 s21, s2, 0x49000000
	s_mov_b32 s22, s26
	s_mov_b32 s23, s27
	v_mad_u32_u24 v8, v8, s3, v7
	s_add_i32 m0, s34, 0x10000
	v_bfrev_b32_e32 v105, -2
	buffer_load_dwordx4 v8, s[20:23], 0 offen lds
	v_lshlrev_b32_e32 v8, 4, v6
	v_or3_b32 v8, v8, s33, v0
	s_add_i32 m0, s35, 0x10000
	v_mad_u32_u24 v7, v8, s3, v7
	s_cmp_lt_u32 s17, 10
	v_add_u32_e32 v8, 0x24000, v7
	v_add_u32_e32 v7, 0x48000, v7
	s_cselect_b64 s[6:7], -1, 0
	buffer_load_dwordx4 v8, s[20:23], 0 offen lds
	v_cndmask_b32_e64 v7, v105, v7, s[6:7]
	s_add_i32 m0, s40, 0x10000
	v_bfe_u32 v97, v96, 4, 2
	buffer_load_dwordx4 v7, s[20:23], 0 offen lds
	v_lshrrev_b32_e32 v7, 4, v96
	v_lshlrev_b32_e32 v8, 4, v97
	v_mul_i32_i24_e32 v98, -16, v7
	v_add3_u32 v124, v8, v98, v96
	v_ashrrev_i32_e32 v8, 31, v124
	v_xor_b32_e32 v9, v8, v124
	s_mov_b32 s42, 0x38e38e39
	v_mul_hi_i32 v9, v9, s42
	v_lshrrev_b32_e32 v10, 31, v9
	v_ashrrev_i32_e32 v9, 3, v9
	v_add_u32_e32 v9, v9, v10
	s_mul_i32 s15, s15, s16
	s_mul_hi_u32 s2, s14, s16
	v_xor_b32_e32 v13, v9, v8
	v_and_b32_e32 v8, 60, v96
	s_add_i32 s2, s2, s15
	s_mul_i32 s3, s14, s16
	v_mov_b32_e32 v9, 0xffffff70
	v_cmp_lt_u32_e32 vcc, 35, v8
	s_add_u32 s28, s4, s3
	v_lshlrev_b32_e32 v10, 6, v7
	v_cndmask_b32_e32 v12, 0, v9, vcc
	v_lshlrev_b32_e32 v9, 2, v96
	s_addc_u32 s2, s5, s2
	s_and_b32 s3, s14, 0x3fff
	s_load_dwordx2 s[12:13], s[0:1], 0x40
	v_and_b32_e32 v99, 0xc0, v96
	v_lshlrev_b32_e32 v8, 6, v97
	v_sub_u32_e32 v15, v9, v10
	s_bitset1_b32 s3, 14
	v_add_u32_e32 v14, v13, v99
	v_add3_u32 v12, v15, v8, v12
	s_and_b32 s2, s2, 0xffff
	s_lshl_b32 s3, s3, 16
	s_or_b32 s29, s2, s3
	v_mad_u64_u32 v[14:15], s[2:3], v14, s14, v[12:13]
	v_mov_b32_e32 v11, s33
	s_movk_i32 s2, 0x60
	v_mad_u32_u24 v100, v2, s2, v11
	v_add_u32_e32 v11, v100, v13
	v_mad_u32_u24 v13, v2, s2, v13
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s2, s13, s33
	s_mul_hi_u32 s3, s12, s33
	s_add_i32 s3, s3, s2
	s_mul_i32 s2, s12, s33
	s_add_u32 s36, s8, s2
	s_addc_u32 s2, s9, s3
	s_and_b32 s3, s12, 0x3fff
	s_bitset1_b32 s3, 14
	s_and_b32 s2, s2, 0xffff
	s_lshl_b32 s3, s3, 16
	s_or_b32 s37, s2, s3
	v_mad_u64_u32 v[12:13], s[2:3], s12, v13, v[12:13]
	s_movk_i32 s5, 0x7e0
	s_lshl_b32 s2, s12, 5
	s_mov_b32 s30, s26
	s_mov_b32 s31, s27
	v_add_u32_e32 v13, s2, v12
	v_cmp_gt_i32_e32 vcc, s5, v11
	s_movk_i32 s9, 0x7c0
	v_lshl_add_u32 v15, s14, 5, v14
	buffer_load_dword v120, v14, s[28:31], 0 offen
	buffer_load_dword v119, v15, s[28:31], 0 offen
	s_mov_b32 s38, s26
	s_mov_b32 s39, s27
	v_cndmask_b32_e32 v14, v105, v13, vcc
	v_add_u32_e32 v13, s2, v13
	v_cmp_gt_i32_e32 vcc, s9, v11
	v_cmp_eq_u32_e64 s[2:3], 0, v2
	s_mov_b32 s4, 0
	v_cndmask_b32_e32 v11, v105, v13, vcc
	buffer_load_dword v123, v12, s[36:39], 0 offen
	buffer_load_dword v122, v14, s[36:39], 0 offen
	buffer_load_dword v121, v11, s[36:39], 0 offen
	s_mov_b32 s8, 0x10000
	v_mul_u32_u24_e32 v101, 0x60, v2
	s_and_b64 vcc, exec, s[2:3]
	s_barrier
	s_waitcnt vmcnt(0)
	s_cbranch_vccnz .LBB0_2
	s_barrier
.LBB0_2:
	v_and_b32_e32 v11, 7, v96
	v_lshlrev_b32_e32 v13, 7, v96
	v_lshlrev_b32_e32 v7, 11, v7
	v_bitop3_b32 v12, v97, v96, 7 bitop3:0x78
	v_sub_u32_e32 v7, v13, v7
	v_bitop3_b32 v11, v97, v11, 4 bitop3:0x36
	s_load_dwordx2 s[18:19], s[0:1], 0x48
	v_lshl_add_u32 v3, v3, 13, v7
	v_lshlrev_b32_e32 v12, 4, v12
	s_movk_i32 s0, 0x3000
	v_lshlrev_b32_e32 v11, 4, v11
	s_mov_b32 s13, 0x12000
	v_or_b32_e32 v106, v3, v12
	v_mul_lo_u32 v13, v2, s0
	v_or_b32_e32 v107, v11, v3
	v_add_u32_e32 v3, v8, v9
	s_mul_i32 s1, s17, 0x6c000
	v_mul_lo_u32 v2, v2, s13
	s_mov_b32 s13, 0x8f00
	v_add_u32_e32 v0, v1, v0
	v_sub_u32_e32 v111, v3, v10
	v_mul_lo_u32 v3, v6, s13
	v_lshrrev_b32_e32 v0, 4, v0
	s_mov_b32 s13, 0x9000
	v_mov_b32_e32 v1, s1
	v_add_u32_e32 v2, s1, v2
	v_mad_u32_u24 v0, v0, s13, v1
	v_add_u32_e32 v2, v2, v3
	v_or_b32_e32 v0, v0, v5
	v_add_u32_e32 v7, v7, v13
	v_add_u32_e32 v2, v2, v5
	v_sub_u32_e32 v0, v0, v4
	v_mov_b32_e32 v52, 0
	v_or_b32_e32 v108, v7, v12
	v_or_b32_e32 v109, v11, v7
	v_add_u32_e32 v110, 32, v99
	s_movk_i32 s0, 0x70
	v_add_u32_e32 v112, 0x70, v111
	v_add_u32_e32 v113, 32, v101
	v_add_u32_e32 v114, 64, v101
	v_sub_u32_e32 v115, 0xffffffe3, v124
	v_add_u32_e32 v116, 0x24800, v2
	s_movk_i32 s1, 0x800
	v_add_u32_e32 v117, 0x800, v0
	v_add_u32_e32 v118, 0x48800, v2
	s_mov_b32 s45, 0x8000
	s_mov_b32 s46, 0x16000
	s_mov_b32 s13, 0
	s_mov_b32 s22, s26
	s_mov_b32 s23, s27
	s_movk_i32 s15, 0xffc0
	s_movk_i32 s17, 0xffe4
	s_movk_i32 s43, 0x90
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
	s_mov_b32 s44, s4
	s_mov_b32 s4, s45
	s_mov_b32 s45, s8
	s_mov_b32 s8, s46
	v_add_u32_e32 v125, v103, v102
	v_add_u32_e32 v126, 0x80, v125
	s_add_i32 m0, s4, s34
	s_barrier
	buffer_load_dwordx4 v126, s[24:27], 0 offen lds
	v_add_u32_e32 v126, 0x24080, v125
	s_add_i32 m0, s4, s35
	s_nop 0
	buffer_load_dwordx4 v126, s[24:27], 0 offen lds
	v_add_u32_e32 v126, 0x48080, v125
	s_add_i32 m0, s4, s40
	v_add_u32_e32 v125, 0x6c080, v125
	buffer_load_dwordx4 v126, s[24:27], 0 offen lds
	s_add_i32 m0, s4, s41
	s_nop 0
	buffer_load_dwordx4 v125, s[24:27], 0 offen lds
	v_add_u32_e32 v125, v117, v104
	s_add_i32 m0, s46, s34
	s_nop 0
	buffer_load_dwordx4 v125, s[20:23], 0 offen lds
	v_add_u32_e32 v125, v116, v104
	s_add_i32 m0, s46, s35
	s_nop 0
	buffer_load_dwordx4 v125, s[20:23], 0 offen lds
	v_add_u32_e32 v125, v104, v118
	v_cndmask_b32_e64 v125, v105, v125, s[6:7]
	s_add_i32 m0, s46, s40
	s_nop 0
	buffer_load_dwordx4 v125, s[20:23], 0 offen lds
	v_add_u32_e32 v130, 64, v124
	v_subrev_u32_e32 v125, 36, v115
	v_cmp_gt_i32_e32 vcc, s15, v124
	v_add_u32_e32 v127, 28, v124
	v_add_u32_e32 v131, s13, v111
	v_cndmask_b32_e32 v125, v130, v125, vcc
	v_mul_hi_i32 v125, v125, s42
	v_lshrrev_b32_e32 v126, 31, v125
	v_ashrrev_i32_e32 v125, 3, v125
	v_add_u32_e32 v125, v125, v126
	v_cndmask_b32_e64 v126, 0, -1, vcc
	v_cmp_gt_i32_e32 vcc, s17, v124
	v_xor_b32_e32 v126, v125, v126
	v_add_u32_e32 v125, v126, v99
	v_cndmask_b32_e32 v124, v127, v115, vcc
	v_mul_hi_i32 v124, v124, s42
	v_lshrrev_b32_e32 v127, 31, v124
	v_lshrrev_b32_e32 v124, 3, v124
	v_add_u32_e32 v124, v124, v127
	v_cndmask_b32_e64 v127, 0, -1, vcc
	v_xor_b32_e32 v124, v124, v127
	v_add_u32_e32 v127, v110, v126
	v_mul_lo_u32 v125, v125, s14
	v_mul_lo_u32 v124, v124, s43
	v_mul_lo_u32 v127, s14, v127
	v_sub_u32_e32 v125, v125, v124
	v_sub_u32_e32 v127, v127, v124
	v_add_u32_e32 v128, v126, v101
	v_add3_u32 v125, v112, v125, s13
	v_add3_u32 v127, v112, v127, s13
	v_mul_lo_u32 v128, v128, s12
	v_add_u32_e32 v132, v113, v126
	buffer_load_dword v129, v125, s[28:31], 0 offen
	s_nop 0
	buffer_load_dword v125, v127, s[28:31], 0 offen
	v_add_u32_e32 v127, v100, v126
	v_sub_u32_e32 v128, v128, v124
	v_mul_lo_u32 v132, s12, v132
	v_add_u32_e32 v126, v114, v126
	v_add3_u32 v128, v131, v128, s0
	v_cmp_gt_i32_e32 vcc, s1, v127
	v_sub_u32_e32 v132, v132, v124
	v_mul_lo_u32 v126, s12, v126
	v_cndmask_b32_e32 v128, v105, v128, vcc
	v_add3_u32 v132, v131, v132, s0
	v_cmp_gt_i32_e32 vcc, s5, v127
	v_sub_u32_e32 v124, v126, v124
	v_add3_u32 v124, v131, v124, s0
	v_cndmask_b32_e32 v132, v105, v132, vcc
	v_cmp_gt_i32_e32 vcc, s9, v127
	s_nop 1
	v_cndmask_b32_e32 v124, v105, v124, vcc
	buffer_load_dword v128, v128, s[36:39], 0 offen
	s_nop 0
	buffer_load_dword v127, v132, s[36:39], 0 offen
	buffer_load_dword v126, v124, s[36:39], 0 offen
	v_add_u32_e32 v124, s44, v106
	ds_read_b128 v[132:135], v124
	ds_read_b128 v[136:139], v124 offset:2048
	ds_read_b128 v[140:143], v124 offset:4096
	ds_read_b128 v[144:147], v124 offset:6144
	v_add_u32_e32 v124, s45, v108
	ds_read_b128 v[148:151], v124
	ds_read_b128 v[152:155], v124 offset:2048
	ds_read_b128 v[156:159], v124 offset:4096
	ds_read_b128 v[160:163], v124 offset:6144
	ds_read_b128 v[164:167], v124 offset:8192
	ds_read_b128 v[168:171], v124 offset:10240
	s_barrier
	s_setprio 1
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[132:135], v[148:151], v[52:55], v120, v123 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[132:135], v[152:155], v[92:95], v120, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[132:135], v[156:159], v[88:91], v120, v122 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[84:87], v[132:135], v[160:163], v[84:87], v120, v122 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[80:83], v[132:135], v[164:167], v[80:83], v120, v121 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[132:135], v[168:171], v[76:79], v120, v121 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[136:139], v[148:151], v[72:75], v120, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[136:139], v[152:155], v[68:71], v120, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[136:139], v[156:159], v[64:67], v120, v122 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[136:139], v[160:163], v[60:63], v120, v122 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[136:139], v[164:167], v[56:59], v120, v121 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[136:139], v[168:171], v[48:51], v120, v121 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[140:143], v[148:151], v[44:47], v119, v123 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[140:143], v[152:155], v[40:43], v119, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[140:143], v[156:159], v[28:31], v119, v122 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[140:143], v[160:163], v[16:19], v119, v122 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[140:143], v[164:167], v[12:15], v119, v121 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[140:143], v[168:171], v[20:23], v119, v121 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[144:147], v[148:151], v[24:27], v119, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[144:147], v[152:155], v[32:35], v119, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[144:147], v[156:159], v[36:39], v119, v122 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[144:147], v[160:163], v[8:11], v119, v122 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[144:147], v[164:167], v[4:7], v119, v121 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[144:147], v[168:171], v[0:3], v119, v121 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_barrier
	v_add_u32_e32 v124, s44, v107
	ds_read_b128 v[132:135], v124
	ds_read_b128 v[136:139], v124 offset:2048
	ds_read_b128 v[140:143], v124 offset:4096
	ds_read_b128 v[144:147], v124 offset:6144
	v_add_u32_e32 v124, s45, v109
	ds_read_b128 v[148:151], v124
	ds_read_b128 v[152:155], v124 offset:2048
	ds_read_b128 v[156:159], v124 offset:4096
	ds_read_b128 v[160:163], v124 offset:6144
	ds_read_b128 v[164:167], v124 offset:8192
	ds_read_b128 v[168:171], v124 offset:10240
	s_waitcnt vmcnt(5)
	s_barrier
	s_setprio 1
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[132:135], v[148:151], v[52:55], v120, v123 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[132:135], v[152:155], v[92:95], v120, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[132:135], v[156:159], v[88:91], v120, v122 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[84:87], v[132:135], v[160:163], v[84:87], v120, v122 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[80:83], v[132:135], v[164:167], v[80:83], v120, v121 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[132:135], v[168:171], v[76:79], v120, v121 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[136:139], v[148:151], v[72:75], v120, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[136:139], v[152:155], v[68:71], v120, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[136:139], v[156:159], v[64:67], v120, v122 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[136:139], v[160:163], v[60:63], v120, v122 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[136:139], v[164:167], v[56:59], v120, v121 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[136:139], v[168:171], v[48:51], v120, v121 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[140:143], v[148:151], v[44:47], v119, v123 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[140:143], v[152:155], v[40:43], v119, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[140:143], v[156:159], v[28:31], v119, v122 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[140:143], v[160:163], v[16:19], v119, v122 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[140:143], v[164:167], v[12:15], v119, v121 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[140:143], v[168:171], v[20:23], v119, v121 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[144:147], v[148:151], v[24:27], v119, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[144:147], v[152:155], v[32:35], v119, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[144:147], v[156:159], v[36:39], v119, v122 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[144:147], v[160:163], v[8:11], v119, v122 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[144:147], v[164:167], v[4:7], v119, v121 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[144:147], v[168:171], v[0:3], v119, v121 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	s_addk_i32 s13, 0x100
	v_subrev_u32_e32 v115, 64, v115
	v_add_u32_e32 v116, 0x800, v116
	v_add_u32_e32 v117, 0x800, v117
	v_add_u32_e32 v118, 0x800, v118
	v_add_u32_e32 v103, 0x80, v103
	s_cmpk_lg_i32 s13, 0x1100
	v_mov_b32_e32 v124, v130
	s_mov_b32 s46, s45
	s_mov_b32 s45, s44
	s_waitcnt vmcnt(0)
	v_mov_b32_e32 v121, v126
	v_mov_b32_e32 v122, v127
	v_mov_b32_e32 v123, v128
	v_mov_b32_e32 v119, v125
	v_mov_b32_e32 v120, v129
	s_cbranch_scc1 .LBB0_3
	s_andn2_b64 vcc, exec, s[2:3]
	s_cbranch_vccnz .LBB0_6
	s_barrier
.LBB0_6:
	v_add_u32_e32 v102, s8, v108
	v_add_u32_e32 v103, s8, v109
	s_barrier
	ds_read_b128 v[166:169], v102
	ds_read_b128 v[170:173], v102 offset:2048
	ds_read_b128 v[174:177], v103
	ds_read_b128 v[178:181], v103 offset:2048
	ds_read_b128 v[182:185], v102 offset:4096
	ds_read_b128 v[134:137], v102 offset:6144
	ds_read_b128 v[186:189], v103 offset:4096
	ds_read_b128 v[130:133], v103 offset:6144
	ds_read_b128 v[120:123], v102 offset:8192
	ds_read_b128 v[112:115], v102 offset:10240
	ds_read_b128 v[116:119], v103 offset:8192
	ds_read_b128 v[108:111], v103 offset:10240
	v_add_u32_e32 v102, s4, v106
	ds_read_b128 v[154:157], v102
	ds_read_b128 v[190:193], v102 offset:2048
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[154:157], v[166:169], v[52:55], v129, v128 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add_u32_e32 v103, s4, v107
	ds_read_b128 v[158:161], v103
	ds_read_b128 v[194:197], v103 offset:2048
	ds_read_b128 v[198:201], v102 offset:4096
	ds_read_b128 v[138:141], v102 offset:6144
	ds_read_b128 v[202:205], v103 offset:4096
	ds_read_b128 v[104:107], v103 offset:6144
	s_movk_i32 s0, 0x7fff
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[158:161], v[174:177], v[52:55], v129, v128 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mov_b32_e32 v102, 0x7fc0
	s_mul_hi_u32 s1, s18, s16
	v_lshl_or_b32 v97, v97, 2, v99
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[154:157], v[182:185], v[88:91], v129, v127 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_movk_i32 s2, 0x800
	s_nop 2
	v_bfe_u32 v103, v55, 16, 1
	v_bfe_u32 v124, v54, 16, 1
	v_add3_u32 v103, v55, v103, s0
	v_bfe_u32 v142, v53, 16, 1
	v_bfe_u32 v143, v52, 16, 1
	v_add3_u32 v124, v54, v124, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cmp_o_f32_e32 vcc, v55, v55
	v_add3_u32 v146, v52, v143, s0
	v_add3_u32 v147, v53, v142, s0
	v_lshrrev_b32_e32 v124, 16, v124
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[158:161], v[186:189], v[88:91], v129, v127 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_mov_b32 s15, 0x27000
	s_mov_b32 s14, 0x7ffffffd
	s_movk_i32 s4, 0x7d0
	v_cndmask_b32_e32 v88, v102, v103, vcc
	v_cmp_o_f32_e32 vcc, v54, v54
	v_lshrrev_b32_e32 v89, 16, v147
	v_lshrrev_b32_e32 v90, 16, v146
	v_mfma_scale_f32_16x16x128_f8f6f4 v[146:149], v[154:157], v[134:137], v[84:87], v129, v127 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_movk_i32 s6, 0x7c0
	s_movk_i32 s8, 0x7b0
	s_nop 0
	v_cndmask_b32_e32 v84, v102, v124, vcc
	v_cmp_o_f32_e32 vcc, v53, v53
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[154:157], v[170:173], v[92:95], v129, v128 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e32 v85, v102, v89, vcc
	v_cmp_o_f32_e32 vcc, v52, v52
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[154:157], v[120:123], v[80:83], v129, v126 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e32 v87, v102, v90, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[150:153], v[158:161], v[116:119], v[52:55], v129, v126 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[154:157], v[112:115], v[76:79], v129, v126 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[154:157], v[158:161], v[108:111], v[52:55], v129, v126 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[190:193], v[166:169], v[72:75], v129, v128 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[158:161], v[178:181], v[92:95], v129, v128 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_nop 1
	v_bfe_u32 v72, v145, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[146:149], v[158:161], v[130:133], v[146:149], v129, v127 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[158:161], v[194:197], v[174:177], v[52:55], v129, v128 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_nop 1
	v_bfe_u32 v86, v95, 16, 1
	v_bfe_u32 v80, v94, 16, 1
	v_add3_u32 v76, v95, v86, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[190:193], v[170:173], v[68:71], v129, v128 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v81, v93, 16, 1
	v_add3_u32 v80, v94, v80, s0
	v_lshrrev_b32_e32 v76, 16, v76
	v_mfma_scale_f32_16x16x128_f8f6f4 v[162:165], v[194:197], v[178:181], v[52:55], v129, v128 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v95, v95
	v_bfe_u32 v82, v92, 16, 1
	v_add3_u32 v81, v93, v81, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[190:193], v[182:185], v[64:67], v129, v127 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v77, 16, v80
	v_cndmask_b32_e32 v76, v102, v76, vcc
	v_cmp_o_f32_e32 vcc, v94, v94
	v_add3_u32 v64, v145, v72, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[194:197], v[186:189], v[52:55], v129, v127 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_add3_u32 v82, v92, v82, s0
	v_lshrrev_b32_e32 v79, 16, v81
	v_cndmask_b32_e32 v78, v102, v77, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[190:193], v[134:137], v[60:63], v129, v127 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v93, v93
	v_bfe_u32 v68, v144, 16, 1
	v_bfe_u32 v69, v143, 16, 1
	v_bfe_u32 v70, v142, 16, 1
	v_lshrrev_b32_e32 v81, 16, v82
	v_cndmask_b32_e32 v80, v102, v79, vcc
	v_cmp_o_f32_e32 vcc, v92, v92
	v_add3_u32 v70, v142, v70, s0
	v_add3_u32 v69, v143, v69, s0
	v_add3_u32 v68, v144, v68, s0
	v_cndmask_b32_e32 v89, v102, v81, vcc
	v_lshrrev_b32_e32 v64, 16, v64
	v_lshrrev_b32_e32 v65, 16, v68
	v_lshrrev_b32_e32 v66, 16, v69
	v_lshrrev_b32_e32 v67, 16, v70
	v_cmp_o_f32_e32 vcc, v145, v145
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[194:197], v[130:133], v[52:55], v129, v127 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_bfe_u32 v60, v149, 16, 1
	v_cndmask_b32_e32 v77, v102, v64, vcc
	v_cmp_o_f32_e32 vcc, v144, v144
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[190:193], v[120:123], v[56:59], v129, v126 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e32 v81, v102, v65, vcc
	v_cmp_o_f32_e32 vcc, v143, v143
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[190:193], v[112:115], v[48:51], v129, v126 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v56, v148, 16, 1
	v_cndmask_b32_e32 v83, v102, v66, vcc
	v_cmp_o_f32_e32 vcc, v142, v142
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[198:201], v[166:169], v[44:47], v125, v128 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v57, v147, 16, 1
	v_cndmask_b32_e32 v92, v102, v67, vcc
	v_cmp_o_f32_e32 vcc, v149, v149
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[194:197], v[116:119], v[52:55], v129, v126 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_bfe_u32 v58, v146, 16, 1
	s_nop 1
	v_add3_u32 v55, v149, v60, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[198:201], v[170:173], v[40:43], v125, v128 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v54, v148, v56, s0
	v_lshrrev_b32_e32 v55, 16, v55
	v_add3_u32 v53, v147, v57, s0
	v_lshrrev_b32_e32 v54, 16, v54
	v_cndmask_b32_e32 v79, v102, v55, vcc
	v_cmp_o_f32_e32 vcc, v148, v148
	v_add3_u32 v52, v146, v58, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[194:197], v[108:111], v[48:51], v129, v126 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v86, v102, v54, vcc
	v_cmp_o_f32_e32 vcc, v147, v147
	s_nop 0
	v_lshrrev_b32_e32 v48, 16, v53
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[202:205], v[174:177], v[44:47], v125, v128 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v49, 16, v52
	v_cndmask_b32_e32 v90, v102, v48, vcc
	v_cmp_o_f32_e32 vcc, v146, v146
	v_bfe_u32 v44, v153, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[198:201], v[120:123], v[12:15], v125, v126 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v45, v152, 16, 1
	v_cndmask_b32_e32 v93, v102, v49, vcc
	v_bfe_u32 v46, v151, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[202:205], v[178:181], v[40:43], v125, v128 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_bfe_u32 v47, v150, 16, 1
	v_cmp_o_f32_e32 vcc, v153, v153
	s_nop 0
	v_add3_u32 v43, v153, v44, s0
	v_add3_u32 v42, v152, v45, s0
	v_lshrrev_b32_e32 v43, 16, v43
	v_add3_u32 v40, v150, v47, s0
	v_add3_u32 v41, v151, v46, s0
	v_lshrrev_b32_e32 v42, 16, v42
	v_cndmask_b32_e32 v82, v102, v43, vcc
	v_cmp_o_f32_e32 vcc, v152, v152
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[198:201], v[182:185], v[28:31], v125, v127 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v94, 16, v41
	v_lshrrev_b32_e32 v95, 16, v40
	v_cndmask_b32_e32 v91, v102, v42, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[202:205], v[116:119], v[12:15], v125, v126 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v151, v151
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[198:201], v[112:115], v[20:23], v125, v126 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[202:205], v[186:189], v[28:31], v125, v127 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[202:205], v[108:111], v[12:15], v125, v126 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[138:141], v[166:169], v[24:27], v125, v128 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[198:201], v[134:137], v[16:19], v125, v127 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 1
	v_bfe_u32 v24, v157, 16, 1
	v_bfe_u32 v25, v156, 16, 1
	v_add3_u32 v24, v157, v24, s0
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[104:107], v[174:177], v[12:15], v125, v128 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_bfe_u32 v26, v155, 16, 1
	v_add3_u32 v25, v156, v25, s0
	v_lshrrev_b32_e32 v24, 16, v24
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[138:141], v[170:173], v[32:35], v125, v128 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v27, v154, 16, 1
	v_add3_u32 v26, v155, v26, s0
	v_add3_u32 v27, v154, v27, s0
	v_cndmask_b32_e32 v33, v102, v94, vcc
	v_cmp_o_f32_e32 vcc, v150, v150
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[202:205], v[130:133], v[16:19], v125, v127 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v32, 16, v25
	v_lshrrev_b32_e32 v26, 16, v26
	v_lshrrev_b32_e32 v27, 16, v27
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[104:107], v[178:181], v[12:15], v125, v128 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[138:141], v[182:185], v[36:39], v125, v127 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 2
	v_cndmask_b32_e32 v36, v102, v95, vcc
	v_cmp_o_f32_e32 vcc, v157, v157
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[138:141], v[134:137], v[8:11], v125, v127 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v95, v162, 16, 1
	v_cndmask_b32_e32 v25, v102, v24, vcc
	v_cmp_o_f32_e32 vcc, v156, v156
	v_bfe_u32 v24, v161, 16, 1
	v_add3_u32 v24, v161, v24, s0
	v_cndmask_b32_e32 v34, v102, v32, vcc
	v_cmp_o_f32_e32 vcc, v155, v155
	v_lshrrev_b32_e32 v24, 16, v24
	v_bfe_u32 v32, v158, 16, 1
	v_cndmask_b32_e32 v37, v102, v26, vcc
	v_cmp_o_f32_e32 vcc, v154, v154
	v_bfe_u32 v26, v160, 16, 1
	v_add3_u32 v26, v160, v26, s0
	v_cndmask_b32_e32 v38, v102, v27, vcc
	v_bfe_u32 v27, v159, 16, 1
	v_cmp_o_f32_e32 vcc, v161, v161
	v_add3_u32 v27, v159, v27, s0
	v_lshrrev_b32_e32 v26, 16, v26
	v_cndmask_b32_e32 v24, v102, v24, vcc
	v_cmp_o_f32_e32 vcc, v160, v160
	v_add3_u32 v32, v158, v32, s0
	v_lshrrev_b32_e32 v35, 16, v27
	v_cndmask_b32_e32 v27, v102, v26, vcc
	v_cmp_o_f32_e32 vcc, v159, v159
	v_lshrrev_b32_e32 v32, 16, v32
	v_bfe_u32 v26, v165, 16, 1
	v_cndmask_b32_e32 v39, v102, v35, vcc
	v_cmp_o_f32_e32 vcc, v158, v158
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[138:141], v[120:123], v[4:7], v125, v126 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v26, v165, v26, s0
	v_cndmask_b32_e32 v94, v102, v32, vcc
	v_bfe_u32 v32, v164, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[138:141], v[112:115], v[0:3], v125, v126 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v35, v163, 16, 1
	v_add3_u32 v32, v164, v32, s0
	v_lshrrev_b32_e32 v26, 16, v26
	v_cmp_o_f32_e32 vcc, v165, v165
	v_add3_u32 v35, v163, v35, s0
	v_lshrrev_b32_e32 v32, 16, v32
	v_cndmask_b32_e32 v26, v102, v26, vcc
	v_cmp_o_f32_e32 vcc, v164, v164
	v_add3_u32 v95, v162, v95, s0
	v_lshrrev_b32_e32 v103, 16, v35
	v_cndmask_b32_e32 v35, v102, v32, vcc
	v_cmp_o_f32_e32 vcc, v163, v163
	v_bfe_u32 v32, v75, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[104:107], v[186:189], v[12:15], v125, v127 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v95, 16, v95
	v_cndmask_b32_e32 v103, v102, v103, vcc
	v_cmp_o_f32_e32 vcc, v162, v162
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[104:107], v[130:133], v[8:11], v125, v127 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_add3_u32 v32, v75, v32, s0
	v_cndmask_b32_e32 v95, v102, v95, vcc
	v_lshrrev_b32_e32 v32, 16, v32
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[104:107], v[116:119], v[4:7], v125, v126 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v75, v75
	v_bfe_u32 v75, v71, 16, 1
	v_add3_u32 v75, v71, v75, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[104:107], v[108:111], v[0:3], v125, v126 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_bfe_u32 v104, v74, 16, 1
	v_bfe_u32 v105, v73, 16, 1
	v_add3_u32 v104, v74, v104, s0
	v_bfe_u32 v106, v72, 16, 1
	v_add3_u32 v105, v73, v105, s0
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v32, v102, v32, vcc
	v_cmp_o_f32_e32 vcc, v74, v74
	v_add3_u32 v106, v72, v106, s0
	v_lshrrev_b32_e32 v105, 16, v105
	v_cndmask_b32_e32 v74, v102, v104, vcc
	v_cmp_o_f32_e32 vcc, v73, v73
	v_lshrrev_b32_e32 v106, 16, v106
	v_bfe_u32 v104, v70, 16, 1
	v_cndmask_b32_e32 v73, v102, v105, vcc
	v_cmp_o_f32_e32 vcc, v72, v72
	v_bfe_u32 v105, v69, 16, 1
	v_add3_u32 v104, v70, v104, s0
	v_cndmask_b32_e32 v72, v102, v106, vcc
	v_lshrrev_b32_e32 v75, 16, v75
	v_cmp_o_f32_e32 vcc, v71, v71
	v_bfe_u32 v106, v68, 16, 1
	v_add3_u32 v105, v69, v105, s0
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v71, v102, v75, vcc
	v_cmp_o_f32_e32 vcc, v70, v70
	v_add3_u32 v106, v68, v106, s0
	v_lshrrev_b32_e32 v105, 16, v105
	v_cndmask_b32_e32 v70, v102, v104, vcc
	v_cmp_o_f32_e32 vcc, v69, v69
	v_bfe_u32 v75, v67, 16, 1
	v_lshrrev_b32_e32 v106, 16, v106
	v_cndmask_b32_e32 v69, v102, v105, vcc
	v_cmp_o_f32_e32 vcc, v68, v68
	v_bfe_u32 v104, v66, 16, 1
	v_add3_u32 v75, v67, v75, s0
	v_cndmask_b32_e32 v68, v102, v106, vcc
	v_bfe_u32 v105, v65, 16, 1
	v_add3_u32 v104, v66, v104, s0
	v_lshrrev_b32_e32 v75, 16, v75
	v_cmp_o_f32_e32 vcc, v67, v67
	v_bfe_u32 v106, v64, 16, 1
	v_add3_u32 v105, v65, v105, s0
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v67, v102, v75, vcc
	v_cmp_o_f32_e32 vcc, v66, v66
	v_add3_u32 v106, v64, v106, s0
	v_lshrrev_b32_e32 v105, 16, v105
	v_cndmask_b32_e32 v66, v102, v104, vcc
	v_cmp_o_f32_e32 vcc, v65, v65
	v_bfe_u32 v75, v63, 16, 1
	v_lshrrev_b32_e32 v106, 16, v106
	v_cndmask_b32_e32 v65, v102, v105, vcc
	v_cmp_o_f32_e32 vcc, v64, v64
	v_bfe_u32 v104, v62, 16, 1
	v_add3_u32 v75, v63, v75, s0
	v_cndmask_b32_e32 v64, v102, v106, vcc
	v_bfe_u32 v105, v61, 16, 1
	v_add3_u32 v104, v62, v104, s0
	v_lshrrev_b32_e32 v75, 16, v75
	v_cmp_o_f32_e32 vcc, v63, v63
	v_bfe_u32 v106, v60, 16, 1
	v_add3_u32 v105, v61, v105, s0
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v63, v102, v75, vcc
	v_cmp_o_f32_e32 vcc, v62, v62
	v_add3_u32 v106, v60, v106, s0
	v_lshrrev_b32_e32 v105, 16, v105
	v_cndmask_b32_e32 v62, v102, v104, vcc
	v_cmp_o_f32_e32 vcc, v61, v61
	v_bfe_u32 v75, v59, 16, 1
	v_lshrrev_b32_e32 v106, 16, v106
	v_cndmask_b32_e32 v61, v102, v105, vcc
	v_cmp_o_f32_e32 vcc, v60, v60
	v_bfe_u32 v104, v58, 16, 1
	v_add3_u32 v75, v59, v75, s0
	v_cndmask_b32_e32 v60, v102, v106, vcc
	v_bfe_u32 v105, v57, 16, 1
	v_add3_u32 v104, v58, v104, s0
	v_lshrrev_b32_e32 v75, 16, v75
	v_cmp_o_f32_e32 vcc, v59, v59
	v_bfe_u32 v106, v56, 16, 1
	v_add3_u32 v105, v57, v105, s0
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v59, v102, v75, vcc
	v_cmp_o_f32_e32 vcc, v58, v58
	v_add3_u32 v106, v56, v106, s0
	v_lshrrev_b32_e32 v105, 16, v105
	v_cndmask_b32_e32 v58, v102, v104, vcc
	v_cmp_o_f32_e32 vcc, v57, v57
	v_bfe_u32 v75, v55, 16, 1
	v_lshrrev_b32_e32 v106, 16, v106
	v_cndmask_b32_e32 v57, v102, v105, vcc
	v_cmp_o_f32_e32 vcc, v56, v56
	v_bfe_u32 v104, v54, 16, 1
	v_add3_u32 v75, v55, v75, s0
	v_cndmask_b32_e32 v56, v102, v106, vcc
	v_bfe_u32 v105, v53, 16, 1
	v_add3_u32 v104, v54, v104, s0
	v_lshrrev_b32_e32 v75, 16, v75
	v_cmp_o_f32_e32 vcc, v55, v55
	v_bfe_u32 v106, v52, 16, 1
	v_add3_u32 v105, v53, v105, s0
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v55, v102, v75, vcc
	v_cmp_o_f32_e32 vcc, v54, v54
	v_add3_u32 v106, v52, v106, s0
	v_lshrrev_b32_e32 v105, 16, v105
	v_cndmask_b32_e32 v54, v102, v104, vcc
	v_cmp_o_f32_e32 vcc, v53, v53
	v_bfe_u32 v75, v51, 16, 1
	v_lshrrev_b32_e32 v106, 16, v106
	v_cndmask_b32_e32 v53, v102, v105, vcc
	v_cmp_o_f32_e32 vcc, v52, v52
	v_bfe_u32 v104, v50, 16, 1
	v_add3_u32 v75, v51, v75, s0
	v_cndmask_b32_e32 v52, v102, v106, vcc
	v_bfe_u32 v105, v49, 16, 1
	v_add3_u32 v104, v50, v104, s0
	v_lshrrev_b32_e32 v75, 16, v75
	v_cmp_o_f32_e32 vcc, v51, v51
	v_bfe_u32 v106, v48, 16, 1
	v_add3_u32 v105, v49, v105, s0
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v51, v102, v75, vcc
	v_cmp_o_f32_e32 vcc, v50, v50
	v_add3_u32 v106, v48, v106, s0
	v_lshrrev_b32_e32 v105, 16, v105
	v_cndmask_b32_e32 v50, v102, v104, vcc
	v_cmp_o_f32_e32 vcc, v49, v49
	v_bfe_u32 v75, v47, 16, 1
	v_lshrrev_b32_e32 v106, 16, v106
	v_cndmask_b32_e32 v49, v102, v105, vcc
	v_cmp_o_f32_e32 vcc, v48, v48
	v_bfe_u32 v104, v46, 16, 1
	v_add3_u32 v75, v47, v75, s0
	v_cndmask_b32_e32 v48, v102, v106, vcc
	v_bfe_u32 v105, v45, 16, 1
	v_add3_u32 v104, v46, v104, s0
	v_lshrrev_b32_e32 v75, 16, v75
	v_cmp_o_f32_e32 vcc, v47, v47
	v_bfe_u32 v106, v44, 16, 1
	v_add3_u32 v105, v45, v105, s0
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v47, v102, v75, vcc
	v_cmp_o_f32_e32 vcc, v46, v46
	v_add3_u32 v106, v44, v106, s0
	v_lshrrev_b32_e32 v105, 16, v105
	v_cndmask_b32_e32 v46, v102, v104, vcc
	v_cmp_o_f32_e32 vcc, v45, v45
	v_bfe_u32 v75, v43, 16, 1
	v_lshrrev_b32_e32 v106, 16, v106
	v_cndmask_b32_e32 v45, v102, v105, vcc
	v_cmp_o_f32_e32 vcc, v44, v44
	v_bfe_u32 v104, v42, 16, 1
	v_add3_u32 v75, v43, v75, s0
	v_cndmask_b32_e32 v44, v102, v106, vcc
	v_bfe_u32 v105, v41, 16, 1
	v_add3_u32 v104, v42, v104, s0
	v_lshrrev_b32_e32 v75, 16, v75
	v_cmp_o_f32_e32 vcc, v43, v43
	v_bfe_u32 v106, v40, 16, 1
	v_add3_u32 v105, v41, v105, s0
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v43, v102, v75, vcc
	v_cmp_o_f32_e32 vcc, v42, v42
	v_add3_u32 v106, v40, v106, s0
	v_lshrrev_b32_e32 v105, 16, v105
	v_cndmask_b32_e32 v42, v102, v104, vcc
	v_cmp_o_f32_e32 vcc, v41, v41
	v_bfe_u32 v75, v31, 16, 1
	v_lshrrev_b32_e32 v106, 16, v106
	v_cndmask_b32_e32 v41, v102, v105, vcc
	v_cmp_o_f32_e32 vcc, v40, v40
	v_bfe_u32 v104, v30, 16, 1
	v_add3_u32 v75, v31, v75, s0
	v_cndmask_b32_e32 v40, v102, v106, vcc
	v_bfe_u32 v105, v29, 16, 1
	v_add3_u32 v104, v30, v104, s0
	v_lshrrev_b32_e32 v75, 16, v75
	v_cmp_o_f32_e32 vcc, v31, v31
	v_bfe_u32 v106, v28, 16, 1
	v_add3_u32 v105, v29, v105, s0
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v31, v102, v75, vcc
	v_cmp_o_f32_e32 vcc, v30, v30
	v_add3_u32 v106, v28, v106, s0
	v_lshrrev_b32_e32 v105, 16, v105
	v_cndmask_b32_e32 v30, v102, v104, vcc
	v_cmp_o_f32_e32 vcc, v29, v29
	v_bfe_u32 v75, v23, 16, 1
	v_lshrrev_b32_e32 v106, 16, v106
	v_cndmask_b32_e32 v29, v102, v105, vcc
	v_cmp_o_f32_e32 vcc, v28, v28
	v_bfe_u32 v104, v22, 16, 1
	v_add3_u32 v75, v23, v75, s0
	v_cndmask_b32_e32 v28, v102, v106, vcc
	v_bfe_u32 v105, v21, 16, 1
	v_add3_u32 v104, v22, v104, s0
	v_lshrrev_b32_e32 v75, 16, v75
	v_cmp_o_f32_e32 vcc, v23, v23
	v_bfe_u32 v106, v20, 16, 1
	v_add3_u32 v105, v21, v105, s0
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v23, v102, v75, vcc
	v_cmp_o_f32_e32 vcc, v22, v22
	v_add3_u32 v106, v20, v106, s0
	v_lshrrev_b32_e32 v105, 16, v105
	v_cndmask_b32_e32 v22, v102, v104, vcc
	v_cmp_o_f32_e32 vcc, v21, v21
	v_bfe_u32 v75, v19, 16, 1
	v_lshrrev_b32_e32 v106, 16, v106
	v_cndmask_b32_e32 v21, v102, v105, vcc
	v_cmp_o_f32_e32 vcc, v20, v20
	v_bfe_u32 v104, v18, 16, 1
	v_add3_u32 v75, v19, v75, s0
	v_cndmask_b32_e32 v20, v102, v106, vcc
	v_bfe_u32 v105, v17, 16, 1
	v_add3_u32 v104, v18, v104, s0
	v_lshrrev_b32_e32 v75, 16, v75
	v_cmp_o_f32_e32 vcc, v19, v19
	v_bfe_u32 v106, v16, 16, 1
	v_add3_u32 v105, v17, v105, s0
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v19, v102, v75, vcc
	v_cmp_o_f32_e32 vcc, v18, v18
	v_add3_u32 v106, v16, v106, s0
	v_lshrrev_b32_e32 v105, 16, v105
	v_cndmask_b32_e32 v18, v102, v104, vcc
	v_cmp_o_f32_e32 vcc, v17, v17
	v_bfe_u32 v75, v15, 16, 1
	v_lshrrev_b32_e32 v106, 16, v106
	v_cndmask_b32_e32 v17, v102, v105, vcc
	v_cmp_o_f32_e32 vcc, v16, v16
	v_bfe_u32 v104, v14, 16, 1
	v_add3_u32 v75, v15, v75, s0
	v_cndmask_b32_e32 v16, v102, v106, vcc
	v_bfe_u32 v105, v13, 16, 1
	v_add3_u32 v104, v14, v104, s0
	v_lshrrev_b32_e32 v75, 16, v75
	v_cmp_o_f32_e32 vcc, v15, v15
	v_bfe_u32 v106, v12, 16, 1
	v_add3_u32 v105, v13, v105, s0
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v15, v102, v75, vcc
	v_cmp_o_f32_e32 vcc, v14, v14
	v_add3_u32 v106, v12, v106, s0
	v_lshrrev_b32_e32 v105, 16, v105
	v_cndmask_b32_e32 v14, v102, v104, vcc
	v_cmp_o_f32_e32 vcc, v13, v13
	v_bfe_u32 v75, v11, 16, 1
	v_lshrrev_b32_e32 v106, 16, v106
	v_cndmask_b32_e32 v13, v102, v105, vcc
	v_cmp_o_f32_e32 vcc, v12, v12
	v_bfe_u32 v104, v10, 16, 1
	v_add3_u32 v75, v11, v75, s0
	v_cndmask_b32_e32 v12, v102, v106, vcc
	v_bfe_u32 v105, v9, 16, 1
	v_add3_u32 v104, v10, v104, s0
	v_lshrrev_b32_e32 v75, 16, v75
	v_cmp_o_f32_e32 vcc, v11, v11
	v_bfe_u32 v106, v8, 16, 1
	v_add3_u32 v105, v9, v105, s0
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v11, v102, v75, vcc
	v_cmp_o_f32_e32 vcc, v10, v10
	v_add3_u32 v106, v8, v106, s0
	v_lshrrev_b32_e32 v105, 16, v105
	v_cndmask_b32_e32 v10, v102, v104, vcc
	v_cmp_o_f32_e32 vcc, v9, v9
	v_bfe_u32 v75, v7, 16, 1
	v_lshrrev_b32_e32 v106, 16, v106
	v_cndmask_b32_e32 v9, v102, v105, vcc
	v_cmp_o_f32_e32 vcc, v8, v8
	v_bfe_u32 v104, v6, 16, 1
	v_add3_u32 v75, v7, v75, s0
	v_cndmask_b32_e32 v8, v102, v106, vcc
	v_bfe_u32 v105, v5, 16, 1
	v_add3_u32 v104, v6, v104, s0
	v_lshrrev_b32_e32 v75, 16, v75
	v_cmp_o_f32_e32 vcc, v7, v7
	v_bfe_u32 v106, v4, 16, 1
	v_add3_u32 v105, v5, v105, s0
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v7, v102, v75, vcc
	v_cmp_o_f32_e32 vcc, v6, v6
	v_add3_u32 v106, v4, v106, s0
	v_lshrrev_b32_e32 v105, 16, v105
	v_cndmask_b32_e32 v6, v102, v104, vcc
	v_cmp_o_f32_e32 vcc, v5, v5
	v_lshrrev_b32_e32 v106, 16, v106
	v_bfe_u32 v75, v3, 16, 1
	v_cndmask_b32_e32 v5, v102, v105, vcc
	v_cmp_o_f32_e32 vcc, v4, v4
	v_bfe_u32 v104, v2, 16, 1
	v_bfe_u32 v105, v1, 16, 1
	v_cndmask_b32_e32 v4, v102, v106, vcc
	v_bfe_u32 v106, v0, 16, 1
	v_add3_u32 v106, v0, v106, s0
	v_add3_u32 v105, v1, v105, s0
	v_add3_u32 v104, v2, v104, s0
	v_add3_u32 v75, v3, v75, s0
	s_mul_i32 s0, s19, s16
	s_add_i32 s1, s1, s0
	s_mul_i32 s0, s18, s16
	s_lshl_b64 s[0:1], s[0:1], 1
	v_lshrrev_b32_e32 v75, 16, v75
	v_cmp_o_f32_e32 vcc, v3, v3
	s_add_u32 s0, s10, s0
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v3, v102, v75, vcc
	v_cmp_o_f32_e32 vcc, v2, v2
	s_addc_u32 s1, s11, s1
	s_lshl_b32 s3, s33, 1
	v_lshrrev_b32_e32 v105, 16, v105
	v_cndmask_b32_e32 v2, v102, v104, vcc
	v_cmp_o_f32_e32 vcc, v1, v1
	v_add_u32_e32 v75, v98, v96
	s_add_u32 s12, s0, s3
	v_lshrrev_b32_e32 v106, 16, v106
	v_cndmask_b32_e32 v1, v102, v105, vcc
	v_cmp_o_f32_e32 vcc, v0, v0
	v_add_u32_e32 v96, v100, v75
	v_add_u32_e32 v75, v75, v101
	v_mul_lo_u32 v98, s18, v97
	s_addc_u32 s0, s1, 0
	s_and_b32 s1, s18, 0x3fff
	v_cndmask_b32_e32 v0, v102, v106, vcc
	s_lshl_b32 s1, s1, 16
	s_and_b32 s0, s0, 0xffff
	v_add_u32_e32 v99, v98, v75
	v_not_b32_e32 v100, -2.0
	v_cmp_gt_i32_e32 vcc, s2, v96
	s_or_b32 s0, s1, s0
	s_or_b32 s13, s0, 2.0
	v_cndmask_b32_e32 v99, v100, v99, vcc
	v_lshlrev_b32_e32 v99, 1, v99
	buffer_store_short v87, v99, s[12:15], 0 offen
	v_or_b32_e32 v87, 1, v97
	v_mul_lo_u32 v87, s18, v87
	v_add_u32_e32 v99, v87, v75
	v_cndmask_b32_e32 v99, v100, v99, vcc
	v_lshlrev_b32_e32 v99, 1, v99
	buffer_store_short v85, v99, s[12:15], 0 offen
	v_or_b32_e32 v85, 2, v97
	v_mul_lo_u32 v85, s18, v85
	v_add_u32_e32 v99, v85, v75
	v_cndmask_b32_e32 v99, v100, v99, vcc
	v_lshlrev_b32_e32 v99, 1, v99
	buffer_store_short v84, v99, s[12:15], 0 offen
	v_or_b32_e32 v84, 3, v97
	v_mul_lo_u32 v84, s18, v84
	v_add_u32_e32 v99, v84, v75
	v_cndmask_b32_e32 v99, v100, v99, vcc
	v_lshlrev_b32_e32 v99, 1, v99
	buffer_store_short v88, v99, s[12:15], 0 offen
	s_movk_i32 s0, 0x7f0
	v_add_u32_e32 v88, 16, v75
	v_add_u32_e32 v99, v98, v88
	v_cmp_gt_i32_e64 s[0:1], s0, v96
	s_movk_i32 s2, 0x7e0
	v_cmp_gt_i32_e64 s[2:3], s2, v96
	v_cndmask_b32_e64 v99, v100, v99, s[0:1]
	v_lshlrev_b32_e32 v99, 1, v99
	buffer_store_short v89, v99, s[12:15], 0 offen
	v_add_u32_e32 v89, v87, v88
	v_cndmask_b32_e64 v89, v100, v89, s[0:1]
	v_lshlrev_b32_e32 v89, 1, v89
	buffer_store_short v80, v89, s[12:15], 0 offen
	v_add_u32_e32 v80, v85, v88
	v_cndmask_b32_e64 v80, v100, v80, s[0:1]
	v_lshlrev_b32_e32 v80, 1, v80
	buffer_store_short v78, v80, s[12:15], 0 offen
	v_add_u32_e32 v78, v84, v88
	v_cndmask_b32_e64 v78, v100, v78, s[0:1]
	v_lshlrev_b32_e32 v78, 1, v78
	buffer_store_short v76, v78, s[12:15], 0 offen
	v_add_u32_e32 v76, 32, v75
	v_add_u32_e32 v78, v98, v76
	v_cndmask_b32_e64 v78, v100, v78, s[2:3]
	v_lshlrev_b32_e32 v78, 1, v78
	buffer_store_short v92, v78, s[12:15], 0 offen
	v_add_u32_e32 v78, v87, v76
	v_cndmask_b32_e64 v78, v100, v78, s[2:3]
	v_lshlrev_b32_e32 v78, 1, v78
	buffer_store_short v83, v78, s[12:15], 0 offen
	v_add_u32_e32 v78, v85, v76
	v_cndmask_b32_e64 v78, v100, v78, s[2:3]
	v_lshlrev_b32_e32 v78, 1, v78
	buffer_store_short v81, v78, s[12:15], 0 offen
	v_add_u32_e32 v78, v84, v76
	v_cndmask_b32_e64 v78, v100, v78, s[2:3]
	v_lshlrev_b32_e32 v78, 1, v78
	buffer_store_short v77, v78, s[12:15], 0 offen
	v_add_u32_e32 v77, 48, v75
	v_add_u32_e32 v78, v98, v77
	v_cmp_gt_i32_e64 s[4:5], s4, v96
	v_cmp_gt_i32_e64 s[6:7], s6, v96
	v_cmp_gt_i32_e64 s[8:9], s8, v96
	v_cndmask_b32_e64 v78, v100, v78, s[4:5]
	v_lshlrev_b32_e32 v78, 1, v78
	buffer_store_short v93, v78, s[12:15], 0 offen
	v_add_u32_e32 v78, v87, v77
	v_cndmask_b32_e64 v78, v100, v78, s[4:5]
	v_lshlrev_b32_e32 v78, 1, v78
	buffer_store_short v90, v78, s[12:15], 0 offen
	v_add_u32_e32 v78, v85, v77
	v_cndmask_b32_e64 v78, v100, v78, s[4:5]
	v_lshlrev_b32_e32 v78, 1, v78
	buffer_store_short v86, v78, s[12:15], 0 offen
	v_add_u32_e32 v78, v84, v77
	v_cndmask_b32_e64 v78, v100, v78, s[4:5]
	v_lshlrev_b32_e32 v78, 1, v78
	buffer_store_short v79, v78, s[12:15], 0 offen
	v_add_u32_e32 v78, 64, v75
	v_add_u32_e32 v79, v98, v78
	v_cndmask_b32_e64 v79, v100, v79, s[6:7]
	v_lshlrev_b32_e32 v79, 1, v79
	buffer_store_short v36, v79, s[12:15], 0 offen
	v_add_u32_e32 v36, v87, v78
	v_cndmask_b32_e64 v36, v100, v36, s[6:7]
	v_lshlrev_b32_e32 v36, 1, v36
	buffer_store_short v33, v36, s[12:15], 0 offen
	v_add_u32_e32 v33, v85, v78
	v_cndmask_b32_e64 v33, v100, v33, s[6:7]
	v_lshlrev_b32_e32 v33, 1, v33
	buffer_store_short v91, v33, s[12:15], 0 offen
	v_add_u32_e32 v33, v84, v78
	v_cndmask_b32_e64 v33, v100, v33, s[6:7]
	v_lshlrev_b32_e32 v33, 1, v33
	buffer_store_short v82, v33, s[12:15], 0 offen
	v_add_u32_e32 v33, 0x50, v75
	v_add_u32_e32 v36, v98, v33
	v_cndmask_b32_e64 v36, v100, v36, s[8:9]
	v_lshlrev_b32_e32 v36, 1, v36
	buffer_store_short v38, v36, s[12:15], 0 offen
	v_add_u32_e32 v36, v87, v33
	v_cndmask_b32_e64 v36, v100, v36, s[8:9]
	v_lshlrev_b32_e32 v36, 1, v36
	buffer_store_short v37, v36, s[12:15], 0 offen
	v_add_u32_e32 v36, v85, v33
	v_cndmask_b32_e64 v36, v100, v36, s[8:9]
	v_lshlrev_b32_e32 v36, 1, v36
	buffer_store_short v34, v36, s[12:15], 0 offen
	v_add_u32_e32 v34, v84, v33
	v_cndmask_b32_e64 v34, v100, v34, s[8:9]
	v_lshlrev_b32_e32 v34, 1, v34
	buffer_store_short v25, v34, s[12:15], 0 offen
	v_or_b32_e32 v25, 16, v97
	v_mul_lo_u32 v25, s18, v25
	v_add_u32_e32 v34, v25, v75
	v_cndmask_b32_e32 v34, v100, v34, vcc
	v_lshlrev_b32_e32 v34, 1, v34
	buffer_store_short v94, v34, s[12:15], 0 offen
	v_or_b32_e32 v34, 17, v97
	v_mul_lo_u32 v34, s18, v34
	v_add_u32_e32 v36, v34, v75
	v_cndmask_b32_e32 v36, v100, v36, vcc
	v_lshlrev_b32_e32 v36, 1, v36
	buffer_store_short v39, v36, s[12:15], 0 offen
	v_or_b32_e32 v36, 18, v97
	v_mul_lo_u32 v36, s18, v36
	v_add_u32_e32 v37, v36, v75
	v_cndmask_b32_e32 v37, v100, v37, vcc
	v_lshlrev_b32_e32 v37, 1, v37
	buffer_store_short v27, v37, s[12:15], 0 offen
	v_or_b32_e32 v27, 19, v97
	v_mul_lo_u32 v27, s18, v27
	v_add_u32_e32 v37, v27, v75
	v_cndmask_b32_e32 v37, v100, v37, vcc
	v_lshlrev_b32_e32 v37, 1, v37
	buffer_store_short v24, v37, s[12:15], 0 offen
	v_add_u32_e32 v24, v25, v88
	v_cndmask_b32_e64 v24, v100, v24, s[0:1]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v95, v24, s[12:15], 0 offen
	v_add_u32_e32 v24, v34, v88
	v_cndmask_b32_e64 v24, v100, v24, s[0:1]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v103, v24, s[12:15], 0 offen
	v_add_u32_e32 v24, v36, v88
	v_cndmask_b32_e64 v24, v100, v24, s[0:1]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v35, v24, s[12:15], 0 offen
	v_add_u32_e32 v24, v27, v88
	v_cndmask_b32_e64 v24, v100, v24, s[0:1]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v26, v24, s[12:15], 0 offen
	v_add_u32_e32 v24, v25, v76
	v_cndmask_b32_e64 v24, v100, v24, s[2:3]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v72, v24, s[12:15], 0 offen
	v_add_u32_e32 v24, v34, v76
	v_cndmask_b32_e64 v24, v100, v24, s[2:3]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v73, v24, s[12:15], 0 offen
	v_add_u32_e32 v24, v36, v76
	v_cndmask_b32_e64 v24, v100, v24, s[2:3]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v74, v24, s[12:15], 0 offen
	v_add_u32_e32 v24, v27, v76
	v_cndmask_b32_e64 v24, v100, v24, s[2:3]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v32, v24, s[12:15], 0 offen
	v_add_u32_e32 v24, v25, v77
	v_cndmask_b32_e64 v24, v100, v24, s[4:5]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v68, v24, s[12:15], 0 offen
	v_add_u32_e32 v24, v34, v77
	v_cndmask_b32_e64 v24, v100, v24, s[4:5]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v69, v24, s[12:15], 0 offen
	v_add_u32_e32 v24, v36, v77
	v_cndmask_b32_e64 v24, v100, v24, s[4:5]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v70, v24, s[12:15], 0 offen
	v_add_u32_e32 v24, v27, v77
	v_cndmask_b32_e64 v24, v100, v24, s[4:5]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v71, v24, s[12:15], 0 offen
	v_add_u32_e32 v24, v25, v78
	v_cndmask_b32_e64 v24, v100, v24, s[6:7]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v64, v24, s[12:15], 0 offen
	v_add_u32_e32 v24, v34, v78
	v_cndmask_b32_e64 v24, v100, v24, s[6:7]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v65, v24, s[12:15], 0 offen
	v_add_u32_e32 v24, v36, v78
	v_cndmask_b32_e64 v24, v100, v24, s[6:7]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v66, v24, s[12:15], 0 offen
	v_add_u32_e32 v24, v27, v78
	v_cndmask_b32_e64 v24, v100, v24, s[6:7]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v67, v24, s[12:15], 0 offen
	v_add_u32_e32 v24, v25, v33
	v_cndmask_b32_e64 v24, v100, v24, s[8:9]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v60, v24, s[12:15], 0 offen
	v_add_u32_e32 v24, v34, v33
	v_cndmask_b32_e64 v24, v100, v24, s[8:9]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v61, v24, s[12:15], 0 offen
	v_add_u32_e32 v24, v36, v33
	v_cndmask_b32_e64 v24, v100, v24, s[8:9]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v62, v24, s[12:15], 0 offen
	v_add_u32_e32 v24, v27, v33
	v_cndmask_b32_e64 v24, v100, v24, s[8:9]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v63, v24, s[12:15], 0 offen
	v_or_b32_e32 v24, 32, v97
	v_mul_lo_u32 v24, s18, v24
	v_add_u32_e32 v25, v24, v75
	v_cndmask_b32_e32 v25, v100, v25, vcc
	v_lshlrev_b32_e32 v25, 1, v25
	buffer_store_short v56, v25, s[12:15], 0 offen
	v_or_b32_e32 v25, 33, v97
	v_mul_lo_u32 v25, s18, v25
	v_add_u32_e32 v26, v25, v75
	v_cndmask_b32_e32 v26, v100, v26, vcc
	v_lshlrev_b32_e32 v26, 1, v26
	buffer_store_short v57, v26, s[12:15], 0 offen
	v_or_b32_e32 v26, 34, v97
	v_mul_lo_u32 v26, s18, v26
	v_add_u32_e32 v27, v26, v75
	v_cndmask_b32_e32 v27, v100, v27, vcc
	v_lshlrev_b32_e32 v27, 1, v27
	buffer_store_short v58, v27, s[12:15], 0 offen
	v_or_b32_e32 v27, 35, v97
	v_mul_lo_u32 v27, s18, v27
	v_add_u32_e32 v32, v27, v75
	v_cndmask_b32_e32 v32, v100, v32, vcc
	v_lshlrev_b32_e32 v32, 1, v32
	buffer_store_short v59, v32, s[12:15], 0 offen
	v_add_u32_e32 v32, v24, v88
	v_cndmask_b32_e64 v32, v100, v32, s[0:1]
	v_lshlrev_b32_e32 v32, 1, v32
	buffer_store_short v52, v32, s[12:15], 0 offen
	v_add_u32_e32 v32, v25, v88
	v_cndmask_b32_e64 v32, v100, v32, s[0:1]
	v_lshlrev_b32_e32 v32, 1, v32
	buffer_store_short v53, v32, s[12:15], 0 offen
	v_add_u32_e32 v32, v26, v88
	v_cndmask_b32_e64 v32, v100, v32, s[0:1]
	v_lshlrev_b32_e32 v32, 1, v32
	buffer_store_short v54, v32, s[12:15], 0 offen
	v_add_u32_e32 v32, v27, v88
	v_cndmask_b32_e64 v32, v100, v32, s[0:1]
	v_lshlrev_b32_e32 v32, 1, v32
	buffer_store_short v55, v32, s[12:15], 0 offen
	v_add_u32_e32 v32, v24, v76
	v_cndmask_b32_e64 v32, v100, v32, s[2:3]
	v_lshlrev_b32_e32 v32, 1, v32
	buffer_store_short v48, v32, s[12:15], 0 offen
	v_add_u32_e32 v32, v25, v76
	v_cndmask_b32_e64 v32, v100, v32, s[2:3]
	v_lshlrev_b32_e32 v32, 1, v32
	buffer_store_short v49, v32, s[12:15], 0 offen
	v_add_u32_e32 v32, v26, v76
	v_cndmask_b32_e64 v32, v100, v32, s[2:3]
	v_lshlrev_b32_e32 v32, 1, v32
	buffer_store_short v50, v32, s[12:15], 0 offen
	v_add_u32_e32 v32, v27, v76
	v_cndmask_b32_e64 v32, v100, v32, s[2:3]
	v_lshlrev_b32_e32 v32, 1, v32
	buffer_store_short v51, v32, s[12:15], 0 offen
	v_add_u32_e32 v32, v24, v77
	v_cndmask_b32_e64 v32, v100, v32, s[4:5]
	v_lshlrev_b32_e32 v32, 1, v32
	buffer_store_short v44, v32, s[12:15], 0 offen
	v_add_u32_e32 v32, v25, v77
	v_cndmask_b32_e64 v32, v100, v32, s[4:5]
	v_lshlrev_b32_e32 v32, 1, v32
	buffer_store_short v45, v32, s[12:15], 0 offen
	v_add_u32_e32 v32, v26, v77
	v_cndmask_b32_e64 v32, v100, v32, s[4:5]
	v_lshlrev_b32_e32 v32, 1, v32
	buffer_store_short v46, v32, s[12:15], 0 offen
	v_add_u32_e32 v32, v27, v77
	v_cndmask_b32_e64 v32, v100, v32, s[4:5]
	v_lshlrev_b32_e32 v32, 1, v32
	buffer_store_short v47, v32, s[12:15], 0 offen
	v_add_u32_e32 v32, v24, v78
	v_cndmask_b32_e64 v32, v100, v32, s[6:7]
	v_lshlrev_b32_e32 v32, 1, v32
	buffer_store_short v40, v32, s[12:15], 0 offen
	v_add_u32_e32 v32, v25, v78
	v_cndmask_b32_e64 v32, v100, v32, s[6:7]
	v_lshlrev_b32_e32 v32, 1, v32
	buffer_store_short v41, v32, s[12:15], 0 offen
	v_add_u32_e32 v32, v26, v78
	v_cndmask_b32_e64 v32, v100, v32, s[6:7]
	v_lshlrev_b32_e32 v32, 1, v32
	buffer_store_short v42, v32, s[12:15], 0 offen
	v_add_u32_e32 v32, v27, v78
	v_add_u32_e32 v24, v24, v33
	v_cndmask_b32_e64 v32, v100, v32, s[6:7]
	v_cndmask_b32_e64 v24, v100, v24, s[8:9]
	v_lshlrev_b32_e32 v32, 1, v32
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v43, v32, s[12:15], 0 offen
	buffer_store_short v28, v24, s[12:15], 0 offen
	v_add_u32_e32 v24, v25, v33
	v_cndmask_b32_e64 v24, v100, v24, s[8:9]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v29, v24, s[12:15], 0 offen
	v_add_u32_e32 v24, v26, v33
	v_cndmask_b32_e64 v24, v100, v24, s[8:9]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v30, v24, s[12:15], 0 offen
	v_add_u32_e32 v24, v27, v33
	v_cndmask_b32_e64 v24, v100, v24, s[8:9]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v31, v24, s[12:15], 0 offen
	v_or_b32_e32 v24, 48, v97
	v_mul_lo_u32 v24, s18, v24
	v_add_u32_e32 v25, v24, v75
	v_cndmask_b32_e32 v25, v100, v25, vcc
	v_lshlrev_b32_e32 v25, 1, v25
	buffer_store_short v20, v25, s[12:15], 0 offen
	v_or_b32_e32 v20, 49, v97
	v_mul_lo_u32 v20, s18, v20
	v_add_u32_e32 v25, v20, v75
	v_cndmask_b32_e32 v25, v100, v25, vcc
	v_lshlrev_b32_e32 v25, 1, v25
	buffer_store_short v21, v25, s[12:15], 0 offen
	v_or_b32_e32 v21, 50, v97
	v_mul_lo_u32 v21, s18, v21
	v_add_u32_e32 v25, v21, v75
	v_cndmask_b32_e32 v25, v100, v25, vcc
	v_lshlrev_b32_e32 v25, 1, v25
	buffer_store_short v22, v25, s[12:15], 0 offen
	v_or_b32_e32 v22, 51, v97
	v_mul_lo_u32 v22, s18, v22
	v_add_u32_e32 v25, v22, v75
	v_cndmask_b32_e32 v25, v100, v25, vcc
	v_lshlrev_b32_e32 v25, 1, v25
	buffer_store_short v23, v25, s[12:15], 0 offen
	v_add_u32_e32 v23, v24, v88
	v_cndmask_b32_e64 v23, v100, v23, s[0:1]
	v_lshlrev_b32_e32 v23, 1, v23
	buffer_store_short v16, v23, s[12:15], 0 offen
	v_add_u32_e32 v16, v20, v88
	v_cndmask_b32_e64 v16, v100, v16, s[0:1]
	v_lshlrev_b32_e32 v16, 1, v16
	buffer_store_short v17, v16, s[12:15], 0 offen
	v_add_u32_e32 v16, v21, v88
	v_cndmask_b32_e64 v16, v100, v16, s[0:1]
	v_lshlrev_b32_e32 v16, 1, v16
	buffer_store_short v18, v16, s[12:15], 0 offen
	v_add_u32_e32 v16, v22, v88
	v_cndmask_b32_e64 v16, v100, v16, s[0:1]
	v_lshlrev_b32_e32 v16, 1, v16
	buffer_store_short v19, v16, s[12:15], 0 offen
	v_add_u32_e32 v16, v24, v76
	v_cndmask_b32_e64 v16, v100, v16, s[2:3]
	v_lshlrev_b32_e32 v16, 1, v16
	buffer_store_short v12, v16, s[12:15], 0 offen
	v_add_u32_e32 v12, v20, v76
	v_cndmask_b32_e64 v12, v100, v12, s[2:3]
	v_lshlrev_b32_e32 v12, 1, v12
	buffer_store_short v13, v12, s[12:15], 0 offen
	v_add_u32_e32 v12, v21, v76
	v_cndmask_b32_e64 v12, v100, v12, s[2:3]
	v_lshlrev_b32_e32 v12, 1, v12
	buffer_store_short v14, v12, s[12:15], 0 offen
	v_add_u32_e32 v12, v22, v76
	v_cndmask_b32_e64 v12, v100, v12, s[2:3]
	v_lshlrev_b32_e32 v12, 1, v12
	buffer_store_short v15, v12, s[12:15], 0 offen
	v_add_u32_e32 v12, v24, v77
	v_cndmask_b32_e64 v12, v100, v12, s[4:5]
	v_lshlrev_b32_e32 v12, 1, v12
	buffer_store_short v8, v12, s[12:15], 0 offen
	v_add_u32_e32 v8, v20, v77
	v_cndmask_b32_e64 v8, v100, v8, s[4:5]
	v_lshlrev_b32_e32 v8, 1, v8
	buffer_store_short v9, v8, s[12:15], 0 offen
	v_add_u32_e32 v8, v21, v77
	v_cndmask_b32_e64 v8, v100, v8, s[4:5]
	v_lshlrev_b32_e32 v8, 1, v8
	buffer_store_short v10, v8, s[12:15], 0 offen
	v_add_u32_e32 v8, v22, v77
	v_cndmask_b32_e64 v8, v100, v8, s[4:5]
	v_lshlrev_b32_e32 v8, 1, v8
	buffer_store_short v11, v8, s[12:15], 0 offen
	v_add_u32_e32 v8, v24, v78
	v_cndmask_b32_e64 v8, v100, v8, s[6:7]
	v_lshlrev_b32_e32 v8, 1, v8
	buffer_store_short v4, v8, s[12:15], 0 offen
	v_add_u32_e32 v4, v20, v78
	v_cndmask_b32_e64 v4, v100, v4, s[6:7]
	v_lshlrev_b32_e32 v4, 1, v4
	buffer_store_short v5, v4, s[12:15], 0 offen
	v_add_u32_e32 v4, v21, v78
	v_cndmask_b32_e64 v4, v100, v4, s[6:7]
	v_lshlrev_b32_e32 v4, 1, v4
	buffer_store_short v6, v4, s[12:15], 0 offen
	v_add_u32_e32 v4, v22, v78
	v_cndmask_b32_e64 v4, v100, v4, s[6:7]
	v_lshlrev_b32_e32 v4, 1, v4
	buffer_store_short v7, v4, s[12:15], 0 offen
	v_add_u32_e32 v4, v24, v33
	v_cndmask_b32_e64 v4, v100, v4, s[8:9]
	v_lshlrev_b32_e32 v4, 1, v4
	buffer_store_short v0, v4, s[12:15], 0 offen
	v_add_u32_e32 v0, v20, v33
	v_cndmask_b32_e64 v0, v100, v0, s[8:9]
	v_lshlrev_b32_e32 v0, 1, v0
	buffer_store_short v1, v0, s[12:15], 0 offen
	v_add_u32_e32 v0, v21, v33
	v_cndmask_b32_e64 v0, v100, v0, s[8:9]
	v_lshlrev_b32_e32 v0, 1, v0
	buffer_store_short v2, v0, s[12:15], 0 offen
	v_add_u32_e32 v0, v22, v33
	v_cndmask_b32_e64 v0, v100, v0, s[8:9]
	v_lshlrev_b32_e32 v0, 1, v0
	buffer_store_short v3, v0, s[12:15], 0 offen
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wave_mxfp4_static_gemm_256x192x256_6144x2048x4608
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
		.amdhsa_next_free_vgpr 206
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
	.size	wave_mxfp4_static_gemm_256x192x256_6144x2048x4608, .Lfunc_end0-wave_mxfp4_static_gemm_256x192x256_6144x2048x4608

	.set wave_mxfp4_static_gemm_256x192x256_6144x2048x4608.num_vgpr, 206
	.set wave_mxfp4_static_gemm_256x192x256_6144x2048x4608.num_agpr, 0
	.set wave_mxfp4_static_gemm_256x192x256_6144x2048x4608.numbered_sgpr, 47
	.set wave_mxfp4_static_gemm_256x192x256_6144x2048x4608.num_named_barrier, 0
	.set wave_mxfp4_static_gemm_256x192x256_6144x2048x4608.private_seg_size, 0
	.set wave_mxfp4_static_gemm_256x192x256_6144x2048x4608.uses_vcc, 1
	.set wave_mxfp4_static_gemm_256x192x256_6144x2048x4608.uses_flat_scratch, 0
	.set wave_mxfp4_static_gemm_256x192x256_6144x2048x4608.has_dyn_sized_stack, 0
	.set wave_mxfp4_static_gemm_256x192x256_6144x2048x4608.has_recursion, 0
	.set wave_mxfp4_static_gemm_256x192x256_6144x2048x4608.has_indirect_call, 0
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
    .name:           wave_mxfp4_static_gemm_256x192x256_6144x2048x4608
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 256
      - 2
      - 1
    .sgpr_count:     53
    .sgpr_spill_count: 0
    .symbol:         wave_mxfp4_static_gemm_256x192x256_6144x2048x4608.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     206
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 0
...

	.end_amdgpu_metadata
