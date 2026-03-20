; To reproduce the .rocmasm from .optimized.ll, run:
; llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -mattr='-fma-mix-insts' -O3 <.optimized.ll> -o <out.rocmasm>

	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.text
	.globl	wave_mxfp4_static_gemm_256x192x256_1792x5376x4096
	.p2align	8
	.type	wave_mxfp4_static_gemm_256x192x256_1792x5376x4096,@function
wave_mxfp4_static_gemm_256x192x256_1792x5376x4096:
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
.LBB0_0:
	v_and_b32_e32 v97, 0x3ff, v0
	v_bfe_u32 v1, v0, 10, 10
	v_lshrrev_b32_e32 v6, 6, v97
	v_lshlrev_b32_e32 v2, 5, v1
	v_lshl_or_b32 v0, v6, 3, v2
	v_lshrrev_b32_e32 v3, 3, v97
	s_mov_b64 s[20:21], s[2:3]
	v_readfirstlane_b32 s2, v0
	v_or_b32_e32 v0, v3, v2
	s_lshl_b32 s16, s16, 8
	v_or_b32_e32 v4, s16, v0
	v_bitop3_b32 v5, v3, 7, v97 bitop3:0x48
	v_lshlrev_b32_e32 v103, 4, v5
	v_lshlrev_b32_e32 v106, 11, v4
	s_and_b32 s3, s21, 0xffff
	s_lshl_b32 s19, s2, 7
	s_or_b32 s21, s3, 0x48000000
	s_mov_b32 s23, 0x27000
	s_mov_b32 s22, 0x7ffffffe
	v_or_b32_e32 v4, v106, v103
	s_mov_b32 m0, s19
	v_or_b32_e32 v107, 0x20000, v106
	s_or_b32 s33, s19, 0x2000
	buffer_load_dwordx4 v4, s[20:23], 0 offen lds
	v_or_b32_e32 v4, v107, v103
	s_mov_b32 m0, s33
	v_or_b32_e32 v108, 0x40000, v106
	s_or_b32 s34, s19, 0x4000
	buffer_load_dwordx4 v4, s[20:23], 0 offen lds
	v_or_b32_e32 v4, v108, v103
	s_mov_b32 m0, s34
	v_or_b32_e32 v109, 0x60000, v106
	s_or_b32 s35, s19, 0x6000
	buffer_load_dwordx4 v4, s[20:23], 0 offen lds
	v_or_b32_e32 v4, v109, v103
	s_mov_b32 m0, s35
	s_mov_b64 s[24:25], s[6:7]
	buffer_load_dwordx4 v4, s[20:23], 0 offen lds
	v_lshrrev_b32_e32 v4, 7, v97
	v_lshlrev_b32_e32 v114, 4, v3
	v_lshlrev_b32_e32 v7, 8, v4
	s_and_b32 s2, s25, 0xffff
	s_mul_i32 s18, s17, 0xc0
	v_sub_u32_e32 v8, v114, v7
	v_lshlrev_b32_e32 v5, 8, v5
	s_or_b32 s25, s2, 0x48000000
	s_mul_i32 s15, s15, s16
	s_mul_hi_u32 s2, s14, s16
	s_load_dwordx2 s[12:13], s[0:1], 0x40
	v_and_or_b32 v0, v0, 48, s18
	v_add_u32_e32 v8, v5, v8
	s_add_i32 s36, s19, 0x10000
	s_add_i32 s37, s33, 0x10000
	s_add_i32 s38, s34, 0x10000
	s_add_i32 s2, s2, s15
	s_mul_i32 s3, s14, s16
	v_lshl_add_u32 v112, v0, 11, v8
	v_lshlrev_b32_e32 v0, 4, v4
	s_add_u32 s4, s4, s3
	v_or3_b32 v0, v0, s18, v2
	s_addc_u32 s2, s5, s2
	s_and_b32 s3, s14, 0x3fff
	v_lshlrev_b32_e32 v0, 11, v0
	s_bitset1_b32 s3, 14
	s_mov_b32 s26, s22
	s_mov_b32 s27, s23
	s_mov_b32 m0, s36
	v_add_u32_e32 v111, v0, v8
	v_lshrrev_b32_e32 v11, 4, v97
	v_bfe_u32 v10, v97, 5, 1
	s_and_b32 s2, s2, 0xffff
	s_lshl_b32 s3, s3, 16
	buffer_load_dwordx4 v112, s[24:27], 0 offen lds
	v_add_u32_e32 v0, 0x20000, v111
	s_mov_b32 m0, s37
	v_lshlrev_b32_e32 v8, 6, v11
	v_lshlrev_b32_e32 v9, 7, v10
	s_or_b32 s5, s2, s3
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s2, s13, s18
	s_mul_hi_u32 s3, s12, s18
	buffer_load_dwordx4 v0, s[24:27], 0 offen lds
	v_add_u32_e32 v0, 0x40000, v111
	s_mov_b32 m0, s38
	v_bfe_u32 v98, v97, 4, 2
	v_and_b32_e32 v99, 0xc0, v97
	v_lshlrev_b32_e32 v119, 2, v97
	v_add_u32_e32 v13, v8, v9
	s_lshl_b32 s15, s14, 5
	s_add_i32 s3, s3, s2
	s_mul_i32 s2, s12, s18
	buffer_load_dwordx4 v0, s[24:27], 0 offen lds
	v_or_b32_e32 v12, v10, v99
	v_lshlrev_b32_e32 v0, 6, v98
	v_sub_u32_e32 v13, v119, v13
	s_add_u32 s28, s8, s2
	v_add_u32_e32 v96, v13, v0
	v_mul_lo_u32 v120, s14, v12
	s_addc_u32 s2, s9, s3
	s_and_b32 s3, s12, 0x3fff
	s_mov_b32 s6, s22
	s_mov_b32 s7, s23
	v_add_u32_e32 v12, v120, v96
	v_mul_u32_u24_e32 v100, 0x60, v1
	s_bitset1_b32 s3, 14
	v_add_u32_e32 v13, s15, v12
	buffer_load_dword v115, v12, s[4:7], 0 offen
	buffer_load_dword v110, v13, s[4:7], 0 offen
	v_or_b32_e32 v12, v10, v100
	s_and_b32 s2, s2, 0xffff
	s_lshl_b32 s3, s3, 16
	s_or_b32 s29, s2, s3
	v_mad_u64_u32 v[12:13], s[2:3], s12, v12, v[96:97]
	s_lshl_b32 s13, s12, 5
	s_mov_b32 s30, s22
	s_mov_b32 s31, s23
	v_add_u32_e32 v13, s13, v12
	v_add_u32_e32 v113, s13, v13
	buffer_load_dword v118, v12, s[28:31], 0 offen
	buffer_load_dword v117, v13, s[28:31], 0 offen
	buffer_load_dword v116, v113, s[28:31], 0 offen
	v_cmp_eq_u32_e64 s[2:3], 0, v1
	s_and_b64 vcc, exec, s[2:3]
	s_barrier
	s_waitcnt vmcnt(0)
	s_cbranch_vccnz .LBB0_2
	s_barrier
.LBB0_2:
	v_and_b32_e32 v12, 7, v97
	v_lshlrev_b32_e32 v14, 7, v97
	v_lshlrev_b32_e32 v11, 11, v11
	s_load_dwordx2 s[8:9], s[0:1], 0x48
	v_bitop3_b32 v13, v98, v97, 7 bitop3:0x78
	v_sub_u32_e32 v11, v14, v11
	s_movk_i32 s0, 0x3000
	v_bitop3_b32 v12, v98, v12, 4 bitop3:0x36
	v_lshl_add_u32 v6, v6, 13, v11
	v_lshlrev_b32_e32 v13, 4, v13
	v_mul_lo_u32 v14, v1, s0
	v_lshlrev_b32_e32 v12, 4, v12
	v_or_b32_e32 v101, v6, v13
	v_add_u32_e32 v11, v11, v14
	v_or_b32_e32 v102, v12, v6
	v_add_u32_e32 v6, v100, v10
	v_or_b32_e32 v104, v11, v13
	v_or_b32_e32 v105, v12, v11
	v_add_u32_e32 v11, 0x44, v6
	v_mad_u64_u32 v[12:13], s[0:1], s12, v11, v[0:1]
	v_sub_u32_e32 v11, v12, v8
	v_sub_u32_e32 v121, v11, v9
	v_add_u32_e32 v11, 36, v6
	v_mad_u64_u32 v[12:13], s[6:7], s12, v11, v[0:1]
	v_sub_u32_e32 v11, v12, v8
	v_sub_u32_e32 v122, v11, v9
	v_add_u32_e32 v11, 4, v6
	v_mad_u64_u32 v[12:13], s[6:7], s12, v11, v[0:1]
	v_sub_u32_e32 v11, v12, v8
	v_add_u32_e32 v12, v99, v10
	v_add_u32_e32 v10, 36, v12
	v_sub_u32_e32 v123, v11, v9
	v_mad_u64_u32 v[10:11], s[6:7], s14, v10, v[0:1]
	v_add_u32_e32 v2, v3, v2
	v_sub_u32_e32 v10, v10, v8
	v_lshlrev_b32_e32 v3, 11, v2
	v_sub_u32_e32 v124, v10, v9
	v_add_u32_e32 v10, 4, v12
	s_mul_i32 s17, s17, 0x60000
	v_and_b32_e32 v3, 0x38000, v3
	v_mad_u64_u32 v[10:11], s[6:7], s14, v10, v[0:1]
	v_add_u32_e32 v3, s17, v3
	v_or_b32_e32 v3, v3, v5
	v_lshlrev_b32_e32 v1, 16, v1
	s_movk_i32 s6, 0x7f00
	v_sub_u32_e32 v126, v3, v7
	v_or_b32_e32 v1, s17, v1
	v_mul_lo_u32 v3, v4, s6
	v_or_b32_e32 v1, v1, v3
	v_add_u32_e32 v127, v1, v5
	v_add_u32_e32 v1, s16, v2
	v_lshl_or_b32 v1, v1, 11, v103
	v_add_u32_e32 v128, 0x60100, v1
	v_add_u32_e32 v1, 0x42, v6
	v_mad_u64_u32 v[2:3], s[6:7], s12, v1, v[0:1]
	v_sub_u32_e32 v1, v2, v8
	v_sub_u32_e32 v129, v1, v9
	v_add_u32_e32 v1, 34, v6
	v_mad_u64_u32 v[2:3], s[6:7], s12, v1, v[0:1]
	v_sub_u32_e32 v1, v2, v8
	v_sub_u32_e32 v130, v1, v9
	v_add_u32_e32 v1, 2, v6
	v_mad_u64_u32 v[2:3], s[6:7], s12, v1, v[0:1]
	v_sub_u32_e32 v1, v2, v8
	v_sub_u32_e32 v131, v1, v9
	v_add_u32_e32 v1, 34, v12
	v_mad_u64_u32 v[2:3], s[6:7], s14, v1, v[0:1]
	v_sub_u32_e32 v1, v2, v8
	v_sub_u32_e32 v132, v1, v9
	v_add_u32_e32 v1, 2, v12
	v_mad_u64_u32 v[0:1], s[6:7], s14, v1, v[0:1]
	v_sub_u32_e32 v0, v0, v8
	v_sub_u32_e32 v10, v10, v8
	v_sub_u32_e32 v133, v0, v9
	v_mov_b32_e32 v0, 0
	s_lshl_b32 s0, s12, 2
	s_lshl_b32 s1, s14, 2
	v_sub_u32_e32 v125, v10, v9
	s_mov_b32 s40, -2
	s_add_i32 s45, s19, 0x8000
	s_add_i32 s44, s33, 0x8000
	s_add_i32 s43, s34, 0x8000
	s_add_i32 s42, s35, 0x8000
	s_add_i32 s17, s19, 0x16000
	s_mov_b32 s26, s22
	s_mov_b32 s27, s23
	s_add_i32 s41, s33, 0x16000
	s_add_i32 s39, s34, 0x16000
	v_add_u32_e32 v134, 0x10000, v104
	s_mov_b32 s6, s22
	s_mov_b32 s7, s23
	s_mov_b32 s30, s22
	s_mov_b32 s31, s23
	v_add_u32_e32 v135, 0x10000, v105
	v_add_u32_e32 v136, 0x16000, v104
	v_add_u32_e32 v137, 0x16000, v105
	v_mov_b32_e32 v1, v0
	v_mov_b32_e32 v2, v0
	v_mov_b32_e32 v3, v0
	v_mov_b32_e32 v4, v0
	v_mov_b32_e32 v5, v0
	v_mov_b32_e32 v6, v0
	v_mov_b32_e32 v7, v0
	v_mov_b32_e32 v8, v0
	v_mov_b32_e32 v9, v0
	v_mov_b32_e32 v10, v0
	v_mov_b32_e32 v11, v0
	v_mov_b32_e32 v12, v0
	v_mov_b32_e32 v13, v0
	v_mov_b32_e32 v14, v0
	v_mov_b32_e32 v15, v0
	v_mov_b32_e32 v16, v0
	v_mov_b32_e32 v17, v0
	v_mov_b32_e32 v18, v0
	v_mov_b32_e32 v19, v0
	v_mov_b32_e32 v20, v0
	v_mov_b32_e32 v21, v0
	v_mov_b32_e32 v22, v0
	v_mov_b32_e32 v23, v0
	v_mov_b32_e32 v24, v0
	v_mov_b32_e32 v25, v0
	v_mov_b32_e32 v26, v0
	v_mov_b32_e32 v27, v0
	v_mov_b32_e32 v28, v0
	v_mov_b32_e32 v29, v0
	v_mov_b32_e32 v30, v0
	v_mov_b32_e32 v31, v0
	v_mov_b32_e32 v32, v0
	v_mov_b32_e32 v33, v0
	v_mov_b32_e32 v34, v0
	v_mov_b32_e32 v35, v0
	v_mov_b32_e32 v36, v0
	v_mov_b32_e32 v37, v0
	v_mov_b32_e32 v38, v0
	v_mov_b32_e32 v39, v0
	v_mov_b32_e32 v40, v0
	v_mov_b32_e32 v41, v0
	v_mov_b32_e32 v42, v0
	v_mov_b32_e32 v43, v0
	v_mov_b32_e32 v44, v0
	v_mov_b32_e32 v45, v0
	v_mov_b32_e32 v46, v0
	v_mov_b32_e32 v47, v0
	v_mov_b32_e32 v48, v0
	v_mov_b32_e32 v49, v0
	v_mov_b32_e32 v50, v0
	v_mov_b32_e32 v51, v0
	v_mov_b32_e32 v52, v0
	v_mov_b32_e32 v53, v0
	v_mov_b32_e32 v54, v0
	v_mov_b32_e32 v55, v0
	v_mov_b32_e32 v56, v0
	v_mov_b32_e32 v57, v0
	v_mov_b32_e32 v58, v0
	v_mov_b32_e32 v59, v0
	v_mov_b32_e32 v60, v0
	v_mov_b32_e32 v61, v0
	v_mov_b32_e32 v62, v0
	v_mov_b32_e32 v63, v0
	v_mov_b32_e32 v64, v0
	v_mov_b32_e32 v65, v0
	v_mov_b32_e32 v66, v0
	v_mov_b32_e32 v67, v0
	v_mov_b32_e32 v68, v0
	v_mov_b32_e32 v69, v0
	v_mov_b32_e32 v70, v0
	v_mov_b32_e32 v71, v0
	v_mov_b32_e32 v72, v0
	v_mov_b32_e32 v73, v0
	v_mov_b32_e32 v74, v0
	v_mov_b32_e32 v75, v0
	v_mov_b32_e32 v76, v0
	v_mov_b32_e32 v77, v0
	v_mov_b32_e32 v78, v0
	v_mov_b32_e32 v79, v0
	v_mov_b32_e32 v80, v0
	v_mov_b32_e32 v81, v0
	v_mov_b32_e32 v82, v0
	v_mov_b32_e32 v83, v0
	v_mov_b32_e32 v84, v0
	v_mov_b32_e32 v85, v0
	v_mov_b32_e32 v86, v0
	v_mov_b32_e32 v87, v0
	v_mov_b32_e32 v88, v0
	v_mov_b32_e32 v89, v0
	v_mov_b32_e32 v90, v0
	v_mov_b32_e32 v91, v0
	v_mov_b32_e32 v92, v0
	v_mov_b32_e32 v93, v0
	v_mov_b32_e32 v94, v0
	v_mov_b32_e32 v95, v0
.LBB0_3:
	s_mov_b32 m0, s45
	v_add_u32_e32 v138, 0xfff9ff80, v128
	s_waitcnt vmcnt(5)
	s_barrier
	buffer_load_dwordx4 v138, s[20:23], 0 offen lds
	v_add_u32_e32 v138, 0xfffbff80, v128
	s_mov_b32 m0, s44
	s_nop 0
	buffer_load_dwordx4 v138, s[20:23], 0 offen lds
	v_add_u32_e32 v138, 0xfffdff80, v128
	s_mov_b32 m0, s43
	s_nop 0
	buffer_load_dwordx4 v138, s[20:23], 0 offen lds
	v_add_u32_e32 v138, 0xffffff80, v128
	s_mov_b32 m0, s42
	s_nop 0
	buffer_load_dwordx4 v138, s[20:23], 0 offen lds
	v_add_u32_e32 v138, v126, v114
	v_add_u32_e32 v139, 0x800, v138
	s_mov_b32 m0, s17
	s_nop 0
	buffer_load_dwordx4 v139, s[24:27], 0 offen lds
	v_add_u32_e32 v139, v127, v114
	v_add_u32_e32 v140, 0x20800, v139
	s_mov_b32 m0, s41
	s_nop 0
	buffer_load_dwordx4 v140, s[24:27], 0 offen lds
	v_add_u32_e32 v140, 0x40800, v139
	s_mov_b32 m0, s39
	s_nop 0
	buffer_load_dwordx4 v140, s[24:27], 0 offen lds
	ds_read_b128 v[140:143], v101
	ds_read_b128 v[144:147], v101 offset:2048
	ds_read_b128 v[148:151], v101 offset:4096
	ds_read_b128 v[152:155], v101 offset:6144
	ds_read_b128 v[156:159], v134
	ds_read_b128 v[160:163], v134 offset:2048
	ds_read_b128 v[164:167], v134 offset:4096
	ds_read_b128 v[168:171], v134 offset:6144
	ds_read_b128 v[172:175], v134 offset:8192
	ds_read_b128 v[176:179], v134 offset:10240
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(9) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[140:143], v[156:159], v[0:3], v115, v118 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[140:143], v[160:163], v[4:7], v115, v118 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(8)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[140:143], v[164:167], v[8:11], v115, v117 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[140:143], v[168:171], v[12:15], v115, v117 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[140:143], v[172:175], v[16:19], v115, v116 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[140:143], v[176:179], v[20:23], v115, v116 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[144:147], v[156:159], v[24:27], v115, v118 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[144:147], v[160:163], v[28:31], v115, v118 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[144:147], v[164:167], v[32:35], v115, v117 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[144:147], v[168:171], v[36:39], v115, v117 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[144:147], v[172:175], v[40:43], v115, v116 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[144:147], v[176:179], v[44:47], v115, v116 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[148:151], v[156:159], v[48:51], v110, v118 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[148:151], v[160:163], v[52:55], v110, v118 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[148:151], v[164:167], v[56:59], v110, v117 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[148:151], v[168:171], v[60:63], v110, v117 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[148:151], v[172:175], v[64:67], v110, v116 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[148:151], v[176:179], v[68:71], v110, v116 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[152:155], v[156:159], v[72:75], v110, v118 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[152:155], v[160:163], v[76:79], v110, v118 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[80:83], v[152:155], v[164:167], v[80:83], v110, v117 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[84:87], v[152:155], v[168:171], v[84:87], v110, v117 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[152:155], v[172:175], v[88:91], v110, v116 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[152:155], v[176:179], v[92:95], v110, v116 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_barrier
	v_add_u32_e32 v142, v131, v119
	v_add_u32_e32 v143, v130, v119
	v_add_u32_e32 v144, v129, v119
	v_add_u32_e32 v140, v133, v119
	v_add_u32_e32 v141, v132, v119
	buffer_load_dword v142, v142, s[28:31], 0 offen
	s_nop 0
	buffer_load_dword v143, v143, s[28:31], 0 offen
	s_nop 0
	buffer_load_dword v144, v144, s[28:31], 0 offen
	s_nop 0
	buffer_load_dword v140, v140, s[4:7], 0 offen
	s_nop 0
	buffer_load_dword v141, v141, s[4:7], 0 offen
	ds_read_b128 v[146:149], v102
	ds_read_b128 v[150:153], v102 offset:2048
	ds_read_b128 v[154:157], v102 offset:4096
	ds_read_b128 v[158:161], v102 offset:6144
	ds_read_b128 v[162:165], v135
	ds_read_b128 v[166:169], v135 offset:2048
	ds_read_b128 v[170:173], v135 offset:4096
	ds_read_b128 v[174:177], v135 offset:6144
	ds_read_b128 v[178:181], v135 offset:8192
	ds_read_b128 v[182:185], v135 offset:10240
	s_waitcnt vmcnt(7)
	s_barrier
	s_setprio 1
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[146:149], v[162:165], v[0:3], v115, v118 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[146:149], v[166:169], v[4:7], v115, v118 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[146:149], v[170:173], v[8:11], v115, v117 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[146:149], v[174:177], v[12:15], v115, v117 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[146:149], v[178:181], v[16:19], v115, v116 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[146:149], v[182:185], v[20:23], v115, v116 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[150:153], v[162:165], v[24:27], v115, v118 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[150:153], v[166:169], v[28:31], v115, v118 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[150:153], v[170:173], v[32:35], v115, v117 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[150:153], v[174:177], v[36:39], v115, v117 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[150:153], v[178:181], v[40:43], v115, v116 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[150:153], v[182:185], v[44:47], v115, v116 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[154:157], v[162:165], v[48:51], v110, v118 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[154:157], v[166:169], v[52:55], v110, v118 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[154:157], v[170:173], v[56:59], v110, v117 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[154:157], v[174:177], v[60:63], v110, v117 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[154:157], v[178:181], v[64:67], v110, v116 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[154:157], v[182:185], v[68:71], v110, v116 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[158:161], v[162:165], v[72:75], v110, v118 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[158:161], v[166:169], v[76:79], v110, v118 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[80:83], v[158:161], v[170:173], v[80:83], v110, v117 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[84:87], v[158:161], v[174:177], v[84:87], v110, v117 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[158:161], v[178:181], v[88:91], v110, v116 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[158:161], v[182:185], v[92:95], v110, v116 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	s_mov_b32 m0, s19
	v_add_u32_e32 v110, 0xfffa0000, v128
	s_waitcnt vmcnt(5)
	s_barrier
	buffer_load_dwordx4 v110, s[20:23], 0 offen lds
	v_add_u32_e32 v110, 0xfffc0000, v128
	s_mov_b32 m0, s33
	s_nop 0
	buffer_load_dwordx4 v110, s[20:23], 0 offen lds
	v_add_u32_e32 v110, 0xfffe0000, v128
	s_mov_b32 m0, s34
	s_nop 0
	buffer_load_dwordx4 v110, s[20:23], 0 offen lds
	s_mov_b32 m0, s35
	v_add_u32_e32 v110, 0x1000, v138
	buffer_load_dwordx4 v128, s[20:23], 0 offen lds
	s_mov_b32 m0, s36
	s_nop 0
	buffer_load_dwordx4 v110, s[24:27], 0 offen lds
	v_add_u32_e32 v110, 0x21000, v139
	s_mov_b32 m0, s37
	s_nop 0
	buffer_load_dwordx4 v110, s[24:27], 0 offen lds
	v_add_u32_e32 v110, 0x41000, v139
	s_mov_b32 m0, s38
	s_nop 0
	buffer_load_dwordx4 v110, s[24:27], 0 offen lds
	ds_read_b128 v[146:149], v101 offset:32768
	ds_read_b128 v[150:153], v101 offset:34816
	ds_read_b128 v[154:157], v101 offset:36864
	ds_read_b128 v[158:161], v101 offset:38912
	ds_read_b128 v[162:165], v136
	ds_read_b128 v[166:169], v136 offset:2048
	ds_read_b128 v[170:173], v136 offset:4096
	ds_read_b128 v[174:177], v136 offset:6144
	ds_read_b128 v[178:181], v136 offset:8192
	ds_read_b128 v[182:185], v136 offset:10240
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(8) lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[146:149], v[162:165], v[0:3], v140, v142 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[146:149], v[166:169], v[4:7], v140, v142 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[146:149], v[170:173], v[8:11], v140, v143 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[146:149], v[174:177], v[12:15], v140, v143 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[146:149], v[178:181], v[16:19], v140, v144 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[146:149], v[182:185], v[20:23], v140, v144 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[150:153], v[162:165], v[24:27], v140, v142 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[150:153], v[166:169], v[28:31], v140, v142 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[150:153], v[170:173], v[32:35], v140, v143 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[150:153], v[174:177], v[36:39], v140, v143 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[150:153], v[178:181], v[40:43], v140, v144 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[150:153], v[182:185], v[44:47], v140, v144 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[154:157], v[162:165], v[48:51], v141, v142 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[154:157], v[166:169], v[52:55], v141, v142 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[154:157], v[170:173], v[56:59], v141, v143 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[154:157], v[174:177], v[60:63], v141, v143 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[154:157], v[178:181], v[64:67], v141, v144 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[154:157], v[182:185], v[68:71], v141, v144 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[158:161], v[162:165], v[72:75], v141, v142 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[158:161], v[166:169], v[76:79], v141, v142 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[80:83], v[158:161], v[170:173], v[80:83], v141, v143 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[84:87], v[158:161], v[174:177], v[84:87], v141, v143 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[158:161], v[178:181], v[88:91], v141, v144 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[158:161], v[182:185], v[92:95], v141, v144 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_barrier
	v_add_u32_e32 v110, v125, v119
	v_add_u32_e32 v116, v124, v119
	buffer_load_dword v115, v110, s[4:7], 0 offen
	s_nop 0
	buffer_load_dword v110, v116, s[4:7], 0 offen
	v_add_u32_e32 v116, v123, v119
	v_add_u32_e32 v117, v122, v119
	v_add_u32_e32 v138, v121, v119
	buffer_load_dword v118, v116, s[28:31], 0 offen
	s_nop 0
	buffer_load_dword v117, v117, s[28:31], 0 offen
	s_nop 0
	buffer_load_dword v116, v138, s[28:31], 0 offen
	ds_read_b128 v[146:149], v102 offset:32768
	ds_read_b128 v[150:153], v102 offset:34816
	ds_read_b128 v[154:157], v102 offset:36864
	ds_read_b128 v[158:161], v102 offset:38912
	ds_read_b128 v[162:165], v137
	ds_read_b128 v[166:169], v137 offset:2048
	ds_read_b128 v[170:173], v137 offset:4096
	ds_read_b128 v[174:177], v137 offset:6144
	ds_read_b128 v[178:181], v137 offset:8192
	ds_read_b128 v[182:185], v137 offset:10240
	s_waitcnt vmcnt(7)
	s_barrier
	s_setprio 1
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[146:149], v[162:165], v[0:3], v140, v142 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[146:149], v[166:169], v[4:7], v140, v142 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[146:149], v[170:173], v[8:11], v140, v143 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[146:149], v[174:177], v[12:15], v140, v143 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[146:149], v[178:181], v[16:19], v140, v144 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[146:149], v[182:185], v[20:23], v140, v144 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[150:153], v[162:165], v[24:27], v140, v142 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[150:153], v[166:169], v[28:31], v140, v142 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[150:153], v[170:173], v[32:35], v140, v143 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[150:153], v[174:177], v[36:39], v140, v143 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[150:153], v[178:181], v[40:43], v140, v144 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[150:153], v[182:185], v[44:47], v140, v144 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[154:157], v[162:165], v[48:51], v141, v142 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[154:157], v[166:169], v[52:55], v141, v142 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[154:157], v[170:173], v[56:59], v141, v143 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[154:157], v[174:177], v[60:63], v141, v143 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[154:157], v[178:181], v[64:67], v141, v144 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[154:157], v[182:185], v[68:71], v141, v144 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[158:161], v[162:165], v[72:75], v141, v142 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[158:161], v[166:169], v[76:79], v141, v142 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[80:83], v[158:161], v[170:173], v[80:83], v141, v143 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[84:87], v[158:161], v[174:177], v[84:87], v141, v143 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[158:161], v[178:181], v[88:91], v141, v144 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[158:161], v[182:185], v[92:95], v141, v144 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	s_add_i32 s40, s40, 2
	v_add_u32_e32 v121, s0, v121
	v_add_u32_e32 v122, s0, v122
	v_add_u32_e32 v123, s0, v123
	v_add_u32_e32 v124, s1, v124
	v_add_u32_e32 v125, s1, v125
	v_add_u32_e32 v126, 0x1000, v126
	v_add_u32_e32 v127, 0x1000, v127
	v_add_u32_e32 v128, 0x100, v128
	v_add_u32_e32 v129, s0, v129
	v_add_u32_e32 v130, s0, v130
	v_add_u32_e32 v131, s0, v131
	v_add_u32_e32 v132, s1, v132
	s_cmp_lt_u32 s40, 12
	v_add_u32_e32 v133, s1, v133
	s_cbranch_scc1 .LBB0_3
	v_add_u32_e32 v114, s15, v120
	s_movk_i32 s0, 0x780
	s_mov_b32 m0, s45
	v_add3_u32 v106, v103, v106, s0
	s_waitcnt vmcnt(5)
	s_barrier
	buffer_load_dwordx4 v106, s[20:23], 0 offen lds
	v_add3_u32 v106, v103, v107, s0
	s_mov_b32 m0, s44
	s_mov_b32 s26, s22
	buffer_load_dwordx4 v106, s[20:23], 0 offen lds
	v_add3_u32 v106, v103, v108, s0
	s_mov_b32 m0, s43
	v_add3_u32 v103, v103, v109, s0
	buffer_load_dwordx4 v106, s[20:23], 0 offen lds
	s_mov_b32 m0, s42
	s_mov_b32 s27, s23
	buffer_load_dwordx4 v103, s[20:23], 0 offen lds
	v_add_u32_e32 v103, 0x7800, v112
	s_mov_b32 m0, s17
	s_nop 0
	buffer_load_dwordx4 v103, s[24:27], 0 offen lds
	v_add_u32_e32 v103, 0x27800, v111
	s_mov_b32 m0, s41
	s_nop 0
	buffer_load_dwordx4 v103, s[24:27], 0 offen lds
	v_add_u32_e32 v103, 0x47800, v111
	s_mov_b32 m0, s39
	s_nop 0
	buffer_load_dwordx4 v103, s[24:27], 0 offen lds
	v_add_u32_e32 v103, 0x10000, v104
	ds_read_b128 v[144:147], v103
	ds_read_b128 v[148:151], v103 offset:2048
	ds_read_b128 v[152:155], v103 offset:4096
	ds_read_b128 v[156:159], v103 offset:6144
	ds_read_b128 v[160:163], v103 offset:8192
	ds_read_b128 v[168:171], v103 offset:10240
	ds_read_b128 v[106:109], v101
	ds_read_b128 v[120:123], v101 offset:2048
	ds_read_b128 v[140:143], v101 offset:4096
	ds_read_b128 v[164:167], v101 offset:6144
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(9) lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[106:109], v[144:147], v[0:3], v115, v118 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[106:109], v[148:151], v[4:7], v115, v118 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(8)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[106:109], v[152:155], v[8:11], v115, v117 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[106:109], v[156:159], v[12:15], v115, v117 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[106:109], v[160:163], v[16:19], v115, v116 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[106:109], v[168:171], v[20:23], v115, v116 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[120:123], v[144:147], v[24:27], v115, v118 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[120:123], v[148:151], v[28:31], v115, v118 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[120:123], v[152:155], v[32:35], v115, v117 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[120:123], v[156:159], v[36:39], v115, v117 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[120:123], v[160:163], v[40:43], v115, v116 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[120:123], v[168:171], v[44:47], v115, v116 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[120:123], v[140:143], v[144:147], v[48:51], v110, v118 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[124:127], v[140:143], v[148:151], v[52:55], v110, v118 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[128:131], v[140:143], v[152:155], v[56:59], v110, v117 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[132:135], v[140:143], v[156:159], v[60:63], v110, v117 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[136:139], v[140:143], v[160:163], v[64:67], v110, v116 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[140:143], v[140:143], v[168:171], v[68:71], v110, v116 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[144:147], v[164:167], v[144:147], v[72:75], v110, v118 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[148:151], v[164:167], v[148:151], v[76:79], v110, v118 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[152:155], v[164:167], v[152:155], v[80:83], v110, v117 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[156:159], v[164:167], v[156:159], v[84:87], v110, v117 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[160:163], v[164:167], v[160:163], v[88:91], v110, v116 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[164:167], v[164:167], v[168:171], v[92:95], v110, v116 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_barrier
	s_lshl_b32 s0, s14, 1
	v_subrev_u32_e32 v48, s0, v114
	v_add_u32_e32 v48, v96, v48
	s_mov_b32 s6, s22
	s_mov_b32 s7, s23
	s_mul_i32 s0, s12, 0xffffffde
	v_add_u32_e32 v49, s15, v48
	buffer_load_dword v108, v48, s[4:7], 0 offen
	buffer_load_dword v96, v49, s[4:7], 0 offen
	v_add_u32_e32 v48, s0, v113
	s_mov_b32 s30, s22
	s_mov_b32 s31, s23
	v_add_u32_e32 v49, s13, v48
	v_add_u32_e32 v50, s13, v49
	buffer_load_dword v107, v48, s[28:31], 0 offen
	buffer_load_dword v106, v49, s[28:31], 0 offen
	buffer_load_dword v103, v50, s[28:31], 0 offen
	v_add_u32_e32 v48, 0x10000, v105
	ds_read_b128 v[180:183], v48
	ds_read_b128 v[184:187], v48 offset:2048
	ds_read_b128 v[188:191], v48 offset:4096
	ds_read_b128 v[192:195], v48 offset:6144
	ds_read_b128 v[196:199], v48 offset:8192
	ds_read_b128 v[200:203], v48 offset:10240
	ds_read_b128 v[50:53], v102
	ds_read_b128 v[168:171], v102 offset:2048
	ds_read_b128 v[172:175], v102 offset:4096
	ds_read_b128 v[176:179], v102 offset:6144
	s_waitcnt vmcnt(7)
	s_barrier
	s_setprio 1
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[50:53], v[180:183], v[0:3], v115, v118 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[50:53], v[184:187], v[4:7], v115, v118 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[84:87], v[50:53], v[188:191], v[8:11], v115, v117 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[80:83], v[50:53], v[192:195], v[12:15], v115, v117 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[50:53], v[196:199], v[16:19], v115, v116 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[50:53], v[200:203], v[20:23], v115, v116 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[168:171], v[180:183], v[24:27], v115, v118 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[168:171], v[184:187], v[28:31], v115, v118 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[168:171], v[188:191], v[32:35], v115, v117 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[168:171], v[192:195], v[36:39], v115, v117 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[168:171], v[196:199], v[40:43], v115, v116 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[168:171], v[200:203], v[44:47], v115, v116 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[172:175], v[180:183], v[120:123], v110, v118 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[172:175], v[184:187], v[124:127], v110, v118 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[172:175], v[188:191], v[128:131], v110, v117 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[172:175], v[192:195], v[132:135], v110, v117 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[172:175], v[196:199], v[136:139], v110, v116 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[172:175], v[200:203], v[140:143], v110, v116 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[176:179], v[180:183], v[144:147], v110, v118 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[176:179], v[184:187], v[148:151], v110, v118 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[176:179], v[188:191], v[152:155], v110, v117 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[176:179], v[192:195], v[156:159], v110, v117 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[176:179], v[196:199], v[160:163], v110, v116 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[176:179], v[200:203], v[164:167], v110, v116 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	s_andn2_b64 vcc, exec, s[2:3]
	s_cbranch_vccnz .LBB0_6
	s_barrier
.LBB0_6:
	v_add_u32_e32 v104, 0x16000, v104
	s_waitcnt vmcnt(5)
	s_barrier
	ds_read_b128 v[146:149], v104
	ds_read_b128 v[150:153], v104 offset:2048
	v_add_u32_e32 v105, 0x16000, v105
	ds_read_b128 v[154:157], v105
	ds_read_b128 v[158:161], v105 offset:2048
	ds_read_b128 v[162:165], v104 offset:4096
	ds_read_b128 v[166:169], v104 offset:6144
	ds_read_b128 v[170:173], v105 offset:4096
	ds_read_b128 v[174:177], v105 offset:6144
	ds_read_b128 v[178:181], v104 offset:8192
	ds_read_b128 v[114:117], v104 offset:10240
	ds_read_b128 v[182:185], v105 offset:8192
	ds_read_b128 v[110:113], v105 offset:10240
	ds_read_b128 v[138:141], v101 offset:32768
	ds_read_b128 v[186:189], v101 offset:34816
	ds_read_b128 v[142:145], v102 offset:32768
	ds_read_b128 v[190:193], v102 offset:34816
	ds_read_b128 v[194:197], v101 offset:36864
	ds_read_b128 v[122:125], v101 offset:38912
	ds_read_b128 v[198:201], v102 offset:36864
	ds_read_b128 v[118:121], v102 offset:38912
	s_waitcnt vmcnt(2) lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[138:141], v[146:149], v[92:95], v108, v107 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_movk_i32 s0, 0x7fff
	s_mul_hi_u32 s1, s8, s16
	s_nop 0
	v_mov_b32_e32 v92, 0x7fc0
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[142:145], v[154:157], v[126:129], v108, v107 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_and_b32_e32 v97, 15, v97
	s_mov_b32 s3, 0x27000
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[138:141], v[150:153], v[88:91], v108, v107 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[84:87], v[138:141], v[162:165], v[84:87], v108, v106 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 2
	v_bfe_u32 v93, v129, 16, 1
	v_bfe_u32 v94, v128, 16, 1
	v_add3_u32 v93, v129, v93, s0
	v_bfe_u32 v95, v127, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[142:145], v[158:161], v[88:91], v108, v107 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_add3_u32 v94, v128, v94, s0
	v_lshrrev_b32_e32 v93, 16, v93
	v_cmp_o_f32_e32 vcc, v129, v129
	v_bfe_u32 v101, v126, 16, 1
	v_add3_u32 v95, v127, v95, s0
	v_lshrrev_b32_e32 v94, 16, v94
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], v[142:145], v[170:173], v[84:87], v108, v106 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_add3_u32 v101, v126, v101, s0
	s_nop 1
	v_cndmask_b32_e32 v84, v92, v93, vcc
	v_cmp_o_f32_e32 vcc, v128, v128
	s_waitcnt vmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[138:141], v[178:181], v[76:79], v108, v103 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v85, 16, v95
	v_lshrrev_b32_e32 v86, 16, v101
	v_bfe_u32 v87, v88, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], v[138:141], v[166:169], v[80:83], v108, v106 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 2
	v_cndmask_b32_e32 v80, v92, v94, vcc
	v_cmp_o_f32_e32 vcc, v127, v127
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[138:141], v[114:117], v[72:75], v108, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v83, v91, 16, 1
	v_cndmask_b32_e32 v81, v92, v85, vcc
	v_cmp_o_f32_e32 vcc, v126, v126
	v_bfe_u32 v85, v90, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[142:145], v[182:185], v[76:79], v108, v103 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v82, v92, v86, vcc
	v_bfe_u32 v86, v89, 16, 1
	v_cmp_o_f32_e32 vcc, v91, v91
	v_add3_u32 v76, v88, v87, s0
	v_add3_u32 v77, v89, v86, s0
	v_add3_u32 v79, v91, v83, s0
	v_add3_u32 v78, v90, v85, s0
	v_lshrrev_b32_e32 v79, 16, v79
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], v[142:145], v[110:113], v[72:75], v108, v103 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v83, 16, v76
	v_lshrrev_b32_e32 v78, 16, v78
	v_bfe_u32 v85, v128, 16, 1
	v_lshrrev_b32_e32 v73, 16, v77
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[186:189], v[146:149], v[68:71], v108, v107 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v72, v92, v79, vcc
	v_cmp_o_f32_e32 vcc, v90, v90
	v_bfe_u32 v86, v127, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], v[142:145], v[174:177], v[134:137], v108, v106 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v68, v92, v78, vcc
	v_cmp_o_f32_e32 vcc, v89, v89
	v_bfe_u32 v71, v133, 16, 1
	v_add3_u32 v71, v133, v71, s0
	v_cndmask_b32_e32 v69, v92, v73, vcc
	v_cmp_o_f32_e32 vcc, v88, v88
	v_bfe_u32 v73, v132, 16, 1
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[190:193], v[154:157], v[74:77], v108, v107 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v70, v92, v83, vcc
	v_add3_u32 v73, v132, v73, s0
	v_lshrrev_b32_e32 v71, 16, v71
	v_bfe_u32 v74, v131, 16, 1
	v_cmp_o_f32_e32 vcc, v133, v133
	v_bfe_u32 v75, v130, 16, 1
	v_add3_u32 v74, v131, v74, s0
	v_lshrrev_b32_e32 v73, 16, v73
	v_cndmask_b32_e32 v71, v92, v71, vcc
	v_cmp_o_f32_e32 vcc, v132, v132
	v_add3_u32 v75, v130, v75, s0
	v_lshrrev_b32_e32 v74, 16, v74
	v_cndmask_b32_e32 v73, v92, v73, vcc
	v_cmp_o_f32_e32 vcc, v131, v131
	v_bfe_u32 v76, v137, 16, 1
	v_lshrrev_b32_e32 v75, 16, v75
	v_cndmask_b32_e32 v74, v92, v74, vcc
	v_cmp_o_f32_e32 vcc, v130, v130
	v_bfe_u32 v77, v136, 16, 1
	v_add3_u32 v76, v137, v76, s0
	v_cndmask_b32_e32 v75, v92, v75, vcc
	v_bfe_u32 v78, v135, 16, 1
	v_add3_u32 v77, v136, v77, s0
	v_lshrrev_b32_e32 v76, 16, v76
	v_cmp_o_f32_e32 vcc, v137, v137
	v_bfe_u32 v79, v134, 16, 1
	v_add3_u32 v78, v135, v78, s0
	v_lshrrev_b32_e32 v77, 16, v77
	v_cndmask_b32_e32 v76, v92, v76, vcc
	v_cmp_o_f32_e32 vcc, v136, v136
	v_add3_u32 v79, v134, v79, s0
	v_lshrrev_b32_e32 v78, 16, v78
	v_cndmask_b32_e32 v77, v92, v77, vcc
	v_cmp_o_f32_e32 vcc, v135, v135
	v_bfe_u32 v83, v129, 16, 1
	v_lshrrev_b32_e32 v79, 16, v79
	v_cndmask_b32_e32 v78, v92, v78, vcc
	v_cmp_o_f32_e32 vcc, v134, v134
	v_add3_u32 v83, v129, v83, s0
	v_add3_u32 v85, v128, v85, s0
	v_cndmask_b32_e32 v79, v92, v79, vcc
	v_lshrrev_b32_e32 v83, 16, v83
	v_cmp_o_f32_e32 vcc, v129, v129
	v_bfe_u32 v87, v126, 16, 1
	v_add3_u32 v86, v127, v86, s0
	v_lshrrev_b32_e32 v85, 16, v85
	v_cndmask_b32_e32 v83, v92, v83, vcc
	v_cmp_o_f32_e32 vcc, v128, v128
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[186:189], v[150:153], v[64:67], v108, v107 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v87, v126, v87, s0
	v_lshrrev_b32_e32 v86, 16, v86
	v_cndmask_b32_e32 v85, v92, v85, vcc
	v_cmp_o_f32_e32 vcc, v127, v127
	v_bfe_u32 v88, v141, 16, 1
	v_lshrrev_b32_e32 v87, 16, v87
	v_cndmask_b32_e32 v86, v92, v86, vcc
	v_cmp_o_f32_e32 vcc, v126, v126
	v_bfe_u32 v89, v140, 16, 1
	v_add3_u32 v88, v141, v88, s0
	v_cndmask_b32_e32 v87, v92, v87, vcc
	v_bfe_u32 v90, v139, 16, 1
	v_add3_u32 v89, v140, v89, s0
	v_lshrrev_b32_e32 v88, 16, v88
	v_cmp_o_f32_e32 vcc, v141, v141
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[194:197], v[146:149], v[44:47], v96, v107 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v91, v138, 16, 1
	v_add3_u32 v90, v139, v90, s0
	v_lshrrev_b32_e32 v89, 16, v89
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[194:197], v[150:153], v[40:43], v96, v107 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v88, v92, v88, vcc
	v_cmp_o_f32_e32 vcc, v140, v140
	v_add3_u32 v91, v138, v91, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[194:197], v[162:165], v[36:39], v96, v106 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v90, 16, v90
	v_cndmask_b32_e32 v89, v92, v89, vcc
	v_cmp_o_f32_e32 vcc, v139, v139
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[194:197], v[166:169], v[32:35], v96, v106 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v93, v145, 16, 1
	v_lshrrev_b32_e32 v91, 16, v91
	v_cndmask_b32_e32 v90, v92, v90, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[194:197], v[178:181], v[24:27], v96, v103 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v138, v138
	v_bfe_u32 v94, v144, 16, 1
	v_add3_u32 v93, v145, v93, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[194:197], v[114:117], v[20:23], v96, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v91, v92, v91, vcc
	v_bfe_u32 v95, v143, 16, 1
	v_add3_u32 v94, v144, v94, s0
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[122:125], v[146:149], v[16:19], v96, v107 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v93, 16, v93
	v_cmp_o_f32_e32 vcc, v145, v145
	v_add3_u32 v95, v143, v95, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[122:125], v[150:153], v[12:15], v96, v107 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v94, 16, v94
	v_cndmask_b32_e32 v93, v92, v93, vcc
	v_cmp_o_f32_e32 vcc, v144, v144
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[122:125], v[162:165], v[8:11], v96, v106 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v95, 16, v95
	v_cndmask_b32_e32 v94, v92, v94, vcc
	v_cmp_o_f32_e32 vcc, v143, v143
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[122:125], v[166:169], v[4:7], v96, v106 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e32 v95, v92, v95, vcc
	v_cmp_o_f32_e32 vcc, v142, v142
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[122:125], v[178:181], v[0:3], v96, v103 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[122:125], v[114:117], v[28:31], v96, v103 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[190:193], v[158:161], v[64:67], v108, v107 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[186:189], v[162:165], v[60:63], v108, v106 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[186:189], v[178:181], v[52:55], v108, v103 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 5
	v_bfe_u32 v101, v67, 16, 1
	v_bfe_u32 v102, v66, 16, 1
	v_add3_u32 v101, v67, v101, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[186:189], v[114:117], v[48:51], v108, v103 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v102, v66, v102, s0
	v_lshrrev_b32_e32 v101, 16, v101
	v_bfe_u32 v104, v64, 16, 1
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[198:201], v[154:157], v[44:47], v96, v107 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v102, 16, v102
	v_add3_u32 v104, v64, v104, s0
	v_lshrrev_b32_e32 v104, 16, v104
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[198:201], v[158:161], v[40:43], v96, v107 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[198:201], v[170:173], v[36:39], v96, v106 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[198:201], v[174:177], v[32:35], v96, v106 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[198:201], v[182:185], v[24:27], v96, v103 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[198:201], v[110:113], v[20:23], v96, v103 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[118:121], v[154:157], v[16:19], v96, v107 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[118:121], v[158:161], v[12:15], v96, v107 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[118:121], v[170:173], v[8:11], v96, v106 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[118:121], v[174:177], v[4:7], v96, v106 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[118:121], v[182:185], v[0:3], v96, v103 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[118:121], v[110:113], v[28:31], v96, v103 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_bfe_u32 v96, v142, 16, 1
	v_add3_u32 v96, v142, v96, s0
	v_lshrrev_b32_e32 v96, 16, v96
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[190:193], v[170:173], v[60:63], v108, v106 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v96, v92, v96, vcc
	v_cmp_o_f32_e32 vcc, v67, v67
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[186:189], v[166:169], v[56:59], v108, v106 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e32 v67, v92, v101, vcc
	v_cmp_o_f32_e32 vcc, v66, v66
	s_nop 1
	v_bfe_u32 v101, v63, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[190:193], v[182:185], v[52:55], v108, v103 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v66, v92, v102, vcc
	v_cmp_o_f32_e32 vcc, v65, v65
	v_bfe_u32 v102, v62, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[190:193], v[110:113], v[48:51], v108, v103 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_bfe_u32 v103, v65, 16, 1
	v_add3_u32 v103, v65, v103, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[190:193], v[174:177], v[56:59], v108, v106 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v65, v92, v103, vcc
	v_cmp_o_f32_e32 vcc, v64, v64
	v_add3_u32 v101, v63, v101, s0
	v_bfe_u32 v103, v61, 16, 1
	v_cndmask_b32_e32 v64, v92, v104, vcc
	v_add3_u32 v102, v62, v102, s0
	v_lshrrev_b32_e32 v101, 16, v101
	v_cmp_o_f32_e32 vcc, v63, v63
	v_bfe_u32 v104, v60, 16, 1
	v_add3_u32 v103, v61, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cndmask_b32_e32 v63, v92, v101, vcc
	v_cmp_o_f32_e32 vcc, v62, v62
	v_add3_u32 v104, v60, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v62, v92, v102, vcc
	v_cmp_o_f32_e32 vcc, v61, v61
	v_bfe_u32 v101, v59, 16, 1
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v61, v92, v103, vcc
	v_cmp_o_f32_e32 vcc, v60, v60
	v_bfe_u32 v102, v58, 16, 1
	v_add3_u32 v101, v59, v101, s0
	v_cndmask_b32_e32 v60, v92, v104, vcc
	v_bfe_u32 v103, v57, 16, 1
	v_add3_u32 v102, v58, v102, s0
	v_lshrrev_b32_e32 v101, 16, v101
	v_cmp_o_f32_e32 vcc, v59, v59
	v_bfe_u32 v104, v56, 16, 1
	v_add3_u32 v103, v57, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cndmask_b32_e32 v59, v92, v101, vcc
	v_cmp_o_f32_e32 vcc, v58, v58
	v_add3_u32 v104, v56, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v58, v92, v102, vcc
	v_cmp_o_f32_e32 vcc, v57, v57
	v_bfe_u32 v101, v55, 16, 1
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v57, v92, v103, vcc
	v_cmp_o_f32_e32 vcc, v56, v56
	v_bfe_u32 v102, v54, 16, 1
	v_add3_u32 v101, v55, v101, s0
	v_cndmask_b32_e32 v56, v92, v104, vcc
	v_bfe_u32 v103, v53, 16, 1
	v_add3_u32 v102, v54, v102, s0
	v_lshrrev_b32_e32 v101, 16, v101
	v_cmp_o_f32_e32 vcc, v55, v55
	v_bfe_u32 v104, v52, 16, 1
	v_add3_u32 v103, v53, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cndmask_b32_e32 v55, v92, v101, vcc
	v_cmp_o_f32_e32 vcc, v54, v54
	v_add3_u32 v104, v52, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v54, v92, v102, vcc
	v_cmp_o_f32_e32 vcc, v53, v53
	v_bfe_u32 v101, v51, 16, 1
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v53, v92, v103, vcc
	v_cmp_o_f32_e32 vcc, v52, v52
	v_bfe_u32 v102, v50, 16, 1
	v_add3_u32 v101, v51, v101, s0
	v_cndmask_b32_e32 v52, v92, v104, vcc
	v_bfe_u32 v103, v49, 16, 1
	v_add3_u32 v102, v50, v102, s0
	v_lshrrev_b32_e32 v101, 16, v101
	v_cmp_o_f32_e32 vcc, v51, v51
	v_bfe_u32 v104, v48, 16, 1
	v_add3_u32 v103, v49, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cndmask_b32_e32 v51, v92, v101, vcc
	v_cmp_o_f32_e32 vcc, v50, v50
	v_add3_u32 v104, v48, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v50, v92, v102, vcc
	v_cmp_o_f32_e32 vcc, v49, v49
	v_bfe_u32 v101, v47, 16, 1
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v49, v92, v103, vcc
	v_cmp_o_f32_e32 vcc, v48, v48
	v_bfe_u32 v102, v46, 16, 1
	v_add3_u32 v101, v47, v101, s0
	v_cndmask_b32_e32 v48, v92, v104, vcc
	v_bfe_u32 v103, v45, 16, 1
	v_add3_u32 v102, v46, v102, s0
	v_lshrrev_b32_e32 v101, 16, v101
	v_cmp_o_f32_e32 vcc, v47, v47
	v_bfe_u32 v104, v44, 16, 1
	v_add3_u32 v103, v45, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cndmask_b32_e32 v47, v92, v101, vcc
	v_cmp_o_f32_e32 vcc, v46, v46
	v_add3_u32 v104, v44, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v46, v92, v102, vcc
	v_cmp_o_f32_e32 vcc, v45, v45
	v_bfe_u32 v101, v43, 16, 1
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v45, v92, v103, vcc
	v_cmp_o_f32_e32 vcc, v44, v44
	v_bfe_u32 v102, v42, 16, 1
	v_add3_u32 v101, v43, v101, s0
	v_cndmask_b32_e32 v44, v92, v104, vcc
	v_bfe_u32 v103, v41, 16, 1
	v_add3_u32 v102, v42, v102, s0
	v_lshrrev_b32_e32 v101, 16, v101
	v_cmp_o_f32_e32 vcc, v43, v43
	v_bfe_u32 v104, v40, 16, 1
	v_add3_u32 v103, v41, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cndmask_b32_e32 v43, v92, v101, vcc
	v_cmp_o_f32_e32 vcc, v42, v42
	v_add3_u32 v104, v40, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v42, v92, v102, vcc
	v_cmp_o_f32_e32 vcc, v41, v41
	v_bfe_u32 v101, v39, 16, 1
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v41, v92, v103, vcc
	v_cmp_o_f32_e32 vcc, v40, v40
	v_bfe_u32 v102, v38, 16, 1
	v_add3_u32 v101, v39, v101, s0
	v_cndmask_b32_e32 v40, v92, v104, vcc
	v_bfe_u32 v103, v37, 16, 1
	v_add3_u32 v102, v38, v102, s0
	v_lshrrev_b32_e32 v101, 16, v101
	v_cmp_o_f32_e32 vcc, v39, v39
	v_bfe_u32 v104, v36, 16, 1
	v_add3_u32 v103, v37, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cndmask_b32_e32 v39, v92, v101, vcc
	v_cmp_o_f32_e32 vcc, v38, v38
	v_add3_u32 v104, v36, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v38, v92, v102, vcc
	v_cmp_o_f32_e32 vcc, v37, v37
	v_bfe_u32 v101, v35, 16, 1
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v37, v92, v103, vcc
	v_cmp_o_f32_e32 vcc, v36, v36
	v_bfe_u32 v102, v34, 16, 1
	v_add3_u32 v101, v35, v101, s0
	v_cndmask_b32_e32 v36, v92, v104, vcc
	v_bfe_u32 v103, v33, 16, 1
	v_add3_u32 v102, v34, v102, s0
	v_lshrrev_b32_e32 v101, 16, v101
	v_cmp_o_f32_e32 vcc, v35, v35
	v_bfe_u32 v104, v32, 16, 1
	v_add3_u32 v103, v33, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cndmask_b32_e32 v35, v92, v101, vcc
	v_cmp_o_f32_e32 vcc, v34, v34
	v_add3_u32 v104, v32, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v34, v92, v102, vcc
	v_cmp_o_f32_e32 vcc, v33, v33
	v_bfe_u32 v101, v27, 16, 1
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v33, v92, v103, vcc
	v_cmp_o_f32_e32 vcc, v32, v32
	v_bfe_u32 v102, v26, 16, 1
	v_add3_u32 v101, v27, v101, s0
	v_cndmask_b32_e32 v32, v92, v104, vcc
	v_bfe_u32 v103, v25, 16, 1
	v_add3_u32 v102, v26, v102, s0
	v_lshrrev_b32_e32 v101, 16, v101
	v_cmp_o_f32_e32 vcc, v27, v27
	v_bfe_u32 v104, v24, 16, 1
	v_add3_u32 v103, v25, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cndmask_b32_e32 v27, v92, v101, vcc
	v_cmp_o_f32_e32 vcc, v26, v26
	v_add3_u32 v104, v24, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v26, v92, v102, vcc
	v_cmp_o_f32_e32 vcc, v25, v25
	v_bfe_u32 v101, v23, 16, 1
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v25, v92, v103, vcc
	v_cmp_o_f32_e32 vcc, v24, v24
	v_bfe_u32 v102, v22, 16, 1
	v_add3_u32 v101, v23, v101, s0
	v_cndmask_b32_e32 v24, v92, v104, vcc
	v_bfe_u32 v103, v21, 16, 1
	v_add3_u32 v102, v22, v102, s0
	v_lshrrev_b32_e32 v101, 16, v101
	v_cmp_o_f32_e32 vcc, v23, v23
	v_bfe_u32 v104, v20, 16, 1
	v_add3_u32 v103, v21, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cndmask_b32_e32 v23, v92, v101, vcc
	v_cmp_o_f32_e32 vcc, v22, v22
	v_add3_u32 v104, v20, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v22, v92, v102, vcc
	v_cmp_o_f32_e32 vcc, v21, v21
	v_bfe_u32 v101, v19, 16, 1
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v21, v92, v103, vcc
	v_cmp_o_f32_e32 vcc, v20, v20
	v_bfe_u32 v102, v18, 16, 1
	v_add3_u32 v101, v19, v101, s0
	v_cndmask_b32_e32 v20, v92, v104, vcc
	v_bfe_u32 v103, v17, 16, 1
	v_add3_u32 v102, v18, v102, s0
	v_lshrrev_b32_e32 v101, 16, v101
	v_cmp_o_f32_e32 vcc, v19, v19
	v_bfe_u32 v104, v16, 16, 1
	v_add3_u32 v103, v17, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cndmask_b32_e32 v19, v92, v101, vcc
	v_cmp_o_f32_e32 vcc, v18, v18
	v_add3_u32 v104, v16, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v18, v92, v102, vcc
	v_cmp_o_f32_e32 vcc, v17, v17
	v_bfe_u32 v101, v15, 16, 1
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v17, v92, v103, vcc
	v_cmp_o_f32_e32 vcc, v16, v16
	v_bfe_u32 v102, v14, 16, 1
	v_add3_u32 v101, v15, v101, s0
	v_cndmask_b32_e32 v16, v92, v104, vcc
	v_bfe_u32 v103, v13, 16, 1
	v_add3_u32 v102, v14, v102, s0
	v_lshrrev_b32_e32 v101, 16, v101
	v_cmp_o_f32_e32 vcc, v15, v15
	v_bfe_u32 v104, v12, 16, 1
	v_add3_u32 v103, v13, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cndmask_b32_e32 v15, v92, v101, vcc
	v_cmp_o_f32_e32 vcc, v14, v14
	v_add3_u32 v104, v12, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v14, v92, v102, vcc
	v_cmp_o_f32_e32 vcc, v13, v13
	v_bfe_u32 v101, v11, 16, 1
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v13, v92, v103, vcc
	v_cmp_o_f32_e32 vcc, v12, v12
	v_bfe_u32 v102, v10, 16, 1
	v_add3_u32 v101, v11, v101, s0
	v_cndmask_b32_e32 v12, v92, v104, vcc
	v_bfe_u32 v103, v9, 16, 1
	v_add3_u32 v102, v10, v102, s0
	v_lshrrev_b32_e32 v101, 16, v101
	v_cmp_o_f32_e32 vcc, v11, v11
	v_bfe_u32 v104, v8, 16, 1
	v_add3_u32 v103, v9, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cndmask_b32_e32 v11, v92, v101, vcc
	v_cmp_o_f32_e32 vcc, v10, v10
	v_add3_u32 v104, v8, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v10, v92, v102, vcc
	v_cmp_o_f32_e32 vcc, v9, v9
	v_bfe_u32 v101, v7, 16, 1
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v9, v92, v103, vcc
	v_cmp_o_f32_e32 vcc, v8, v8
	v_bfe_u32 v102, v6, 16, 1
	v_add3_u32 v101, v7, v101, s0
	v_cndmask_b32_e32 v8, v92, v104, vcc
	v_bfe_u32 v103, v5, 16, 1
	v_add3_u32 v102, v6, v102, s0
	v_lshrrev_b32_e32 v101, 16, v101
	v_cmp_o_f32_e32 vcc, v7, v7
	v_bfe_u32 v104, v4, 16, 1
	v_add3_u32 v103, v5, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cndmask_b32_e32 v7, v92, v101, vcc
	v_cmp_o_f32_e32 vcc, v6, v6
	v_add3_u32 v104, v4, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v6, v92, v102, vcc
	v_cmp_o_f32_e32 vcc, v5, v5
	v_bfe_u32 v101, v3, 16, 1
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v5, v92, v103, vcc
	v_cmp_o_f32_e32 vcc, v4, v4
	v_bfe_u32 v102, v2, 16, 1
	v_add3_u32 v101, v3, v101, s0
	v_cndmask_b32_e32 v4, v92, v104, vcc
	v_bfe_u32 v103, v1, 16, 1
	v_add3_u32 v102, v2, v102, s0
	v_lshrrev_b32_e32 v101, 16, v101
	v_cmp_o_f32_e32 vcc, v3, v3
	v_bfe_u32 v104, v0, 16, 1
	v_add3_u32 v103, v1, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cndmask_b32_e32 v3, v92, v101, vcc
	v_cmp_o_f32_e32 vcc, v2, v2
	v_add3_u32 v104, v0, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v2, v92, v102, vcc
	v_cmp_o_f32_e32 vcc, v1, v1
	v_lshrrev_b32_e32 v104, 16, v104
	v_bfe_u32 v101, v31, 16, 1
	v_cndmask_b32_e32 v1, v92, v103, vcc
	v_cmp_o_f32_e32 vcc, v0, v0
	v_bfe_u32 v102, v30, 16, 1
	v_bfe_u32 v103, v29, 16, 1
	v_cndmask_b32_e32 v0, v92, v104, vcc
	v_bfe_u32 v104, v28, 16, 1
	v_add3_u32 v104, v28, v104, s0
	v_add3_u32 v103, v29, v103, s0
	v_add3_u32 v102, v30, v102, s0
	v_add3_u32 v101, v31, v101, s0
	s_mul_i32 s0, s9, s16
	s_add_i32 s1, s1, s0
	s_mul_i32 s0, s8, s16
	s_lshl_b64 s[0:1], s[0:1], 1
	v_lshrrev_b32_e32 v101, 16, v101
	v_cmp_o_f32_e32 vcc, v31, v31
	s_add_u32 s0, s10, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cndmask_b32_e32 v31, v92, v101, vcc
	v_cmp_o_f32_e32 vcc, v30, v30
	s_addc_u32 s1, s11, s1
	s_lshl_b32 s2, s18, 1
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v30, v92, v102, vcc
	v_cmp_o_f32_e32 vcc, v29, v29
	s_add_u32 s0, s0, s2
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v29, v92, v103, vcc
	v_cmp_o_f32_e32 vcc, v28, v28
	s_addc_u32 s1, s1, 0
	s_and_b32 s2, s8, 0x3fff
	v_cndmask_b32_e32 v28, v92, v104, vcc
	v_lshl_or_b32 v92, v98, 2, v99
	s_lshl_b32 s2, s2, 16
	s_and_b32 s1, s1, 0xffff
	v_lshlrev_b32_e32 v98, 1, v100
	v_mul_lo_u32 v92, s8, v92
	s_or_b32 s1, s2, s1
	v_lshl_add_u32 v97, v97, 1, v98
	s_or_b32 s1, s1, 2.0
	s_mov_b32 s2, 0x7ffffffd
	v_lshl_add_u32 v98, v92, 1, v97
	s_lshl_b32 s4, s8, 1
	buffer_store_short v82, v98, s[0:3], 0 offen
	v_add_u32_e32 v82, s4, v98
	buffer_store_short v81, v82, s[0:3], 0 offen
	v_add_u32_e32 v81, s4, v82
	buffer_store_short v80, v81, s[0:3], 0 offen
	v_add_u32_e32 v80, s4, v81
	s_lshl_b32 s5, s8, 4
	buffer_store_short v84, v80, s[0:3], 0 offen
	buffer_store_short v70, v98, s[0:3], 0 offen offset:32
	buffer_store_short v69, v82, s[0:3], 0 offen offset:32
	buffer_store_short v68, v81, s[0:3], 0 offen offset:32
	buffer_store_short v72, v80, s[0:3], 0 offen offset:32
	buffer_store_short v75, v98, s[0:3], 0 offen offset:64
	buffer_store_short v74, v82, s[0:3], 0 offen offset:64
	buffer_store_short v73, v81, s[0:3], 0 offen offset:64
	buffer_store_short v71, v80, s[0:3], 0 offen offset:64
	buffer_store_short v79, v98, s[0:3], 0 offen offset:96
	buffer_store_short v78, v82, s[0:3], 0 offen offset:96
	buffer_store_short v77, v81, s[0:3], 0 offen offset:96
	buffer_store_short v76, v80, s[0:3], 0 offen offset:96
	buffer_store_short v87, v98, s[0:3], 0 offen offset:128
	buffer_store_short v86, v82, s[0:3], 0 offen offset:128
	buffer_store_short v85, v81, s[0:3], 0 offen offset:128
	buffer_store_short v83, v80, s[0:3], 0 offen offset:128
	buffer_store_short v91, v98, s[0:3], 0 offen offset:160
	buffer_store_short v90, v82, s[0:3], 0 offen offset:160
	buffer_store_short v89, v81, s[0:3], 0 offen offset:160
	buffer_store_short v88, v80, s[0:3], 0 offen offset:160
	v_add_u32_e32 v68, s5, v92
	v_lshl_add_u32 v69, v68, 1, v97
	v_add_u32_e32 v70, s4, v69
	v_add_u32_e32 v71, s4, v70
	v_add_u32_e32 v72, s4, v71
	buffer_store_short v96, v69, s[0:3], 0 offen
	buffer_store_short v95, v70, s[0:3], 0 offen
	buffer_store_short v94, v71, s[0:3], 0 offen
	buffer_store_short v93, v72, s[0:3], 0 offen
	buffer_store_short v64, v69, s[0:3], 0 offen offset:32
	buffer_store_short v65, v70, s[0:3], 0 offen offset:32
	buffer_store_short v66, v71, s[0:3], 0 offen offset:32
	buffer_store_short v67, v72, s[0:3], 0 offen offset:32
	buffer_store_short v60, v69, s[0:3], 0 offen offset:64
	buffer_store_short v61, v70, s[0:3], 0 offen offset:64
	buffer_store_short v62, v71, s[0:3], 0 offen offset:64
	buffer_store_short v63, v72, s[0:3], 0 offen offset:64
	buffer_store_short v56, v69, s[0:3], 0 offen offset:96
	buffer_store_short v57, v70, s[0:3], 0 offen offset:96
	buffer_store_short v58, v71, s[0:3], 0 offen offset:96
	buffer_store_short v59, v72, s[0:3], 0 offen offset:96
	buffer_store_short v52, v69, s[0:3], 0 offen offset:128
	buffer_store_short v53, v70, s[0:3], 0 offen offset:128
	buffer_store_short v54, v71, s[0:3], 0 offen offset:128
	buffer_store_short v55, v72, s[0:3], 0 offen offset:128
	buffer_store_short v48, v69, s[0:3], 0 offen offset:160
	buffer_store_short v49, v70, s[0:3], 0 offen offset:160
	buffer_store_short v50, v71, s[0:3], 0 offen offset:160
	buffer_store_short v51, v72, s[0:3], 0 offen offset:160
	v_add_u32_e32 v48, s5, v68
	v_lshl_add_u32 v49, v48, 1, v97
	buffer_store_short v44, v49, s[0:3], 0 offen
	v_add_u32_e32 v44, s4, v49
	buffer_store_short v45, v44, s[0:3], 0 offen
	v_add_u32_e32 v45, s4, v44
	buffer_store_short v46, v45, s[0:3], 0 offen
	v_add_u32_e32 v46, s4, v45
	buffer_store_short v47, v46, s[0:3], 0 offen
	buffer_store_short v40, v49, s[0:3], 0 offen offset:32
	buffer_store_short v41, v44, s[0:3], 0 offen offset:32
	buffer_store_short v42, v45, s[0:3], 0 offen offset:32
	buffer_store_short v43, v46, s[0:3], 0 offen offset:32
	buffer_store_short v36, v49, s[0:3], 0 offen offset:64
	buffer_store_short v37, v44, s[0:3], 0 offen offset:64
	buffer_store_short v38, v45, s[0:3], 0 offen offset:64
	buffer_store_short v39, v46, s[0:3], 0 offen offset:64
	buffer_store_short v32, v49, s[0:3], 0 offen offset:96
	buffer_store_short v33, v44, s[0:3], 0 offen offset:96
	buffer_store_short v34, v45, s[0:3], 0 offen offset:96
	buffer_store_short v35, v46, s[0:3], 0 offen offset:96
	buffer_store_short v24, v49, s[0:3], 0 offen offset:128
	buffer_store_short v25, v44, s[0:3], 0 offen offset:128
	buffer_store_short v26, v45, s[0:3], 0 offen offset:128
	buffer_store_short v27, v46, s[0:3], 0 offen offset:128
	buffer_store_short v20, v49, s[0:3], 0 offen offset:160
	buffer_store_short v21, v44, s[0:3], 0 offen offset:160
	buffer_store_short v22, v45, s[0:3], 0 offen offset:160
	buffer_store_short v23, v46, s[0:3], 0 offen offset:160
	v_add_u32_e32 v20, s5, v48
	v_lshl_add_u32 v20, v20, 1, v97
	buffer_store_short v16, v20, s[0:3], 0 offen
	v_add_u32_e32 v16, s4, v20
	buffer_store_short v17, v16, s[0:3], 0 offen
	v_add_u32_e32 v17, s4, v16
	buffer_store_short v18, v17, s[0:3], 0 offen
	v_add_u32_e32 v18, s4, v17
	buffer_store_short v19, v18, s[0:3], 0 offen
	buffer_store_short v12, v20, s[0:3], 0 offen offset:32
	buffer_store_short v13, v16, s[0:3], 0 offen offset:32
	buffer_store_short v14, v17, s[0:3], 0 offen offset:32
	buffer_store_short v15, v18, s[0:3], 0 offen offset:32
	buffer_store_short v8, v20, s[0:3], 0 offen offset:64
	buffer_store_short v9, v16, s[0:3], 0 offen offset:64
	buffer_store_short v10, v17, s[0:3], 0 offen offset:64
	buffer_store_short v11, v18, s[0:3], 0 offen offset:64
	buffer_store_short v4, v20, s[0:3], 0 offen offset:96
	buffer_store_short v5, v16, s[0:3], 0 offen offset:96
	buffer_store_short v6, v17, s[0:3], 0 offen offset:96
	buffer_store_short v7, v18, s[0:3], 0 offen offset:96
	buffer_store_short v0, v20, s[0:3], 0 offen offset:128
	buffer_store_short v1, v16, s[0:3], 0 offen offset:128
	buffer_store_short v2, v17, s[0:3], 0 offen offset:128
	buffer_store_short v3, v18, s[0:3], 0 offen offset:128
	buffer_store_short v28, v20, s[0:3], 0 offen offset:160
	buffer_store_short v29, v16, s[0:3], 0 offen offset:160
	buffer_store_short v30, v17, s[0:3], 0 offen offset:160
	buffer_store_short v31, v18, s[0:3], 0 offen offset:160
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wave_mxfp4_static_gemm_256x192x256_1792x5376x4096
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
	.size	wave_mxfp4_static_gemm_256x192x256_1792x5376x4096, .Lfunc_end0-wave_mxfp4_static_gemm_256x192x256_1792x5376x4096

	.set wave_mxfp4_static_gemm_256x192x256_1792x5376x4096.num_vgpr, 204
	.set wave_mxfp4_static_gemm_256x192x256_1792x5376x4096.num_agpr, 0
	.set wave_mxfp4_static_gemm_256x192x256_1792x5376x4096.numbered_sgpr, 46
	.set wave_mxfp4_static_gemm_256x192x256_1792x5376x4096.num_named_barrier, 0
	.set wave_mxfp4_static_gemm_256x192x256_1792x5376x4096.private_seg_size, 0
	.set wave_mxfp4_static_gemm_256x192x256_1792x5376x4096.uses_vcc, 1
	.set wave_mxfp4_static_gemm_256x192x256_1792x5376x4096.uses_flat_scratch, 0
	.set wave_mxfp4_static_gemm_256x192x256_1792x5376x4096.has_dyn_sized_stack, 0
	.set wave_mxfp4_static_gemm_256x192x256_1792x5376x4096.has_recursion, 0
	.set wave_mxfp4_static_gemm_256x192x256_1792x5376x4096.has_indirect_call, 0
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
    .name:           wave_mxfp4_static_gemm_256x192x256_1792x5376x4096
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 256
      - 2
      - 1
    .sgpr_count:     52
    .sgpr_spill_count: 0
    .symbol:         wave_mxfp4_static_gemm_256x192x256_1792x5376x4096.kd
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
