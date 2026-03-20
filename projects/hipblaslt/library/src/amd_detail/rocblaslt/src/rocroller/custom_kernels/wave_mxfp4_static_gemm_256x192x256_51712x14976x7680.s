; To reproduce the .rocmasm from .optimized.ll, run:
; llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -mattr='-fma-mix-insts' -O3 <.optimized.ll> -o <out.rocmasm>

	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.text
	.globl	wave_mxfp4_static_gemm_256x192x256_51712x14976x7680
	.p2align	8
	.type	wave_mxfp4_static_gemm_256x192x256_51712x14976x7680,@function
wave_mxfp4_static_gemm_256x192x256_51712x14976x7680:
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
.LBB0_0:
	v_and_b32_e32 v96, 0x3ff, v0
	v_bfe_u32 v0, v0, 10, 10
	v_lshrrev_b32_e32 v1, 6, v96
	v_lshlrev_b32_e32 v5, 5, v0
	v_lshl_or_b32 v2, v1, 3, v5
	s_mov_b64 s[20:21], s[2:3]
	v_readfirstlane_b32 s2, v2
	v_lshrrev_b32_e32 v2, 3, v96
	v_or_b32_e32 v103, v2, v5
	s_lshl_b32 s28, s16, 8
	v_or_b32_e32 v3, s28, v103
	v_bitop3_b32 v4, v2, 7, v96 bitop3:0x48
	s_mov_b64 s[24:25], s[6:7]
	v_lshlrev_b32_e32 v104, 4, v4
	v_mul_u32_u24_e32 v109, 0xf00, v3
	s_and_b32 s6, s21, 0xffff
	s_lshl_b32 s30, s2, 7
	s_or_b32 s21, s6, 0x4f000000
	s_mov_b32 s23, 0x27000
	s_mov_b32 s22, 0x7ffffffe
	v_or_b32_e32 v6, v109, v104
	s_mov_b32 m0, s30
	s_movk_i32 s3, 0xf00
	buffer_load_dwordx4 v6, s[20:23], 0 offen lds
	v_mov_b32_e32 v6, 0x3c000
	v_mad_u32_u24 v112, v3, s3, v6
	s_or_b32 s31, s30, 0x2000
	v_or_b32_e32 v6, v112, v104
	s_mov_b32 m0, s31
	s_or_b32 s33, s30, 0x4000
	buffer_load_dwordx4 v6, s[20:23], 0 offen lds
	v_mov_b32_e32 v6, 0x78000
	v_mad_u32_u24 v113, v3, s3, v6
	v_or_b32_e32 v6, v113, v104
	s_mov_b32 m0, s33
	s_or_b32 s34, s30, 0x6000
	buffer_load_dwordx4 v6, s[20:23], 0 offen lds
	v_mov_b32_e32 v6, 0xb4000
	v_mad_u32_u24 v114, v3, s3, v6
	v_or_b32_e32 v3, v114, v104
	s_mov_b32 m0, s34
	v_lshrrev_b32_e32 v6, 7, v96
	buffer_load_dwordx4 v3, s[20:23], 0 offen lds
	v_lshlrev_b32_e32 v2, 4, v2
	v_lshlrev_b32_e32 v3, 8, v6
	s_mul_i32 s29, s17, 0xc0
	v_sub_u32_e32 v7, v2, v3
	v_lshlrev_b32_e32 v4, 8, v4
	v_lshlrev_b32_e32 v6, 4, v6
	v_and_or_b32 v115, v103, 48, s29
	v_add_u32_e32 v111, v4, v7
	s_and_b32 s2, s25, 0xffff
	s_add_i32 s35, s30, 0x10000
	v_or3_b32 v116, v6, s29, v5
	s_or_b32 s25, s2, 0x4f000000
	s_mov_b32 s26, s22
	s_mov_b32 s27, s23
	v_mad_u32_u24 v7, v115, s3, v111
	s_mov_b32 m0, s35
	v_mad_u32_u24 v5, v116, s3, v111
	s_add_i32 s36, s31, 0x10000
	buffer_load_dwordx4 v7, s[24:27], 0 offen lds
	v_add_u32_e32 v6, 0x3c000, v5
	s_mov_b32 m0, s36
	s_add_i32 s37, s33, 0x10000
	buffer_load_dwordx4 v6, s[24:27], 0 offen lds
	v_add_u32_e32 v5, 0x78000, v5
	s_mov_b32 m0, s37
	v_lshrrev_b32_e32 v6, 4, v96
	v_bfe_u32 v98, v96, 4, 2
	buffer_load_dwordx4 v5, s[24:27], 0 offen lds
	v_lshlrev_b32_e32 v5, 4, v98
	v_mul_i32_i24_e32 v97, -16, v6
	v_add3_u32 v105, v5, v97, v96
	v_ashrrev_i32_e32 v7, 31, v105
	v_xor_b32_e32 v8, v7, v105
	s_mov_b32 s2, 0x88888889
	v_mul_hi_i32 v9, v8, s2
	v_add_u32_e32 v8, v9, v8
	v_lshrrev_b32_e32 v9, 31, v8
	v_ashrrev_i32_e32 v8, 5, v8
	s_mul_i32 s15, s15, s28
	s_mul_hi_u32 s2, s14, s28
	v_add_u32_e32 v8, v8, v9
	v_and_b32_e32 v106, 63, v96
	s_add_i32 s2, s2, s15
	s_mul_i32 s3, s14, s28
	v_xor_b32_e32 v9, v8, v7
	v_mov_b32_e32 v7, 0xffffff10
	v_cmp_lt_u32_e32 vcc, 59, v106
	v_lshlrev_b32_e32 v119, 2, v96
	v_lshlrev_b32_e32 v8, 6, v6
	s_add_u32 s4, s4, s3
	s_load_dwordx2 s[12:13], s[0:1], 0x40
	v_cndmask_b32_e32 v10, 0, v7, vcc
	v_lshlrev_b32_e32 v7, 6, v98
	v_sub_u32_e32 v12, v119, v8
	s_addc_u32 s2, s5, s2
	s_and_b32 s3, s14, 0x3fff
	v_and_b32_e32 v99, 0xc0, v96
	v_add_u32_e32 v107, v12, v7
	s_bitset1_b32 s3, 14
	v_add_u32_e32 v11, v9, v99
	v_add_u32_e32 v10, v107, v10
	s_and_b32 s2, s2, 0xffff
	s_lshl_b32 s3, s3, 16
	s_or_b32 s5, s2, s3
	v_mad_u64_u32 v[12:13], s[2:3], v11, s14, v[10:11]
	s_movk_i32 s2, 0x60
	s_nop 0
	v_mad_u32_u24 v9, v0, s2, v9
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s2, s13, s29
	s_mul_hi_u32 s3, s12, s29
	s_lshl_b32 s15, s14, 5
	s_add_i32 s3, s3, s2
	s_mul_i32 s2, s12, s29
	s_add_u32 s16, s8, s2
	s_addc_u32 s2, s9, s3
	s_and_b32 s3, s12, 0x3fff
	s_bitset1_b32 s3, 14
	s_mov_b32 s6, s22
	s_mov_b32 s7, s23
	v_add_u32_e32 v11, s15, v12
	s_and_b32 s2, s2, 0xffff
	s_lshl_b32 s3, s3, 16
	buffer_load_dword v118, v12, s[4:7], 0 offen
	buffer_load_dword v117, v11, s[4:7], 0 offen
	s_or_b32 s17, s2, s3
	v_mad_u64_u32 v[10:11], s[2:3], s12, v9, v[10:11]
	s_lshl_b32 s13, s12, 5
	s_mov_b32 s18, s22
	s_mov_b32 s19, s23
	v_add_u32_e32 v9, s13, v10
	v_add_u32_e32 v11, s13, v9
	buffer_load_dword v122, v10, s[16:19], 0 offen
	buffer_load_dword v121, v9, s[16:19], 0 offen
	buffer_load_dword v120, v11, s[16:19], 0 offen
	v_cmp_eq_u32_e64 s[2:3], 0, v0
	s_mov_b32 s38, 0
	s_movk_i32 s39, 0xff10
	v_mul_u32_u24_e32 v100, 0x60, v0
	s_and_b64 vcc, exec, s[2:3]
	s_barrier
	s_waitcnt vmcnt(0)
	s_cbranch_vccnz .LBB0_2
	s_barrier
.LBB0_2:
	s_load_dwordx2 s[8:9], s[0:1], 0x48
	v_lshlrev_b32_e32 v11, 7, v96
	v_lshlrev_b32_e32 v6, 11, v6
	s_movk_i32 s0, 0x3000
	v_and_b32_e32 v9, 7, v96
	v_sub_u32_e32 v6, v11, v6
	v_mul_lo_u32 v0, v0, s0
	v_bitop3_b32 v10, v98, v96, 7 bitop3:0x78
	v_lshl_add_u32 v1, v1, 13, v6
	v_add_u32_e32 v0, v6, v0
	v_bitop3_b32 v6, v98, v9, 4 bitop3:0x36
	v_lshlrev_b32_e32 v10, 4, v10
	v_lshlrev_b32_e32 v6, 4, v6
	v_or_b32_e32 v108, v0, v10
	v_or_b32_e32 v110, v6, v0
	v_add_u32_e32 v0, v4, v2
	v_sub_u32_e32 v0, v0, v3
	v_add_u32_e32 v124, 0x79000, v0
	v_add_u32_e32 v0, v97, v96
	v_sub_u32_e32 v126, v7, v8
	v_sub_u32_e32 v130, 0, v0
	v_mov_b32_e32 v0, 0
	v_mul_u32_u24_e32 v123, 0xf00, v116
	v_or_b32_e32 v101, v1, v10
	v_or_b32_e32 v102, v6, v1
	v_add_u32_e32 v125, 32, v99
	v_add_u32_e32 v127, 32, v100
	v_add_u32_e32 v128, 64, v100
	v_sub_u32_e32 v129, 0, v5
	s_mov_b32 s43, -2
	s_add_i32 s45, s30, 0x8000
	s_add_i32 s44, s31, 0x8000
	s_add_i32 s42, s33, 0x8000
	s_add_i32 s41, s34, 0x8000
	s_mov_b32 s46, 0x8889
	s_movk_i32 s47, 0xf100
	s_mov_b32 s48, 0xfff87800
	s_add_i32 s1, s30, 0x16000
	s_mov_b32 s26, s22
	s_mov_b32 s27, s23
	s_add_i32 s40, s31, 0x16000
	s_add_i32 s0, s33, 0x16000
	v_add_u32_e32 v131, 0x10000, v108
	s_movk_i32 s49, 0xffc0
	s_movk_i32 s50, 0x8889
	s_mov_b32 s6, s22
	s_mov_b32 s7, s23
	s_mov_b32 s18, s22
	s_mov_b32 s19, s23
	v_add_u32_e32 v132, 0x10000, v110
	s_mov_b32 s51, 0xfff88000
	v_add_u32_e32 v133, 0x16000, v108
	v_add_u32_e32 v134, 0x16000, v110
	v_mov_b32_e32 v137, v105
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
	v_mov_b32_e32 v135, v126
	v_mov_b32_e32 v136, v109
.LBB0_3:
	v_add_u32_e32 v138, v136, v104
	s_mov_b32 m0, s45
	v_add_u32_e32 v139, 0x80, v138
	s_waitcnt vmcnt(5)
	s_barrier
	buffer_load_dwordx4 v139, s[20:23], 0 offen lds
	v_add_u32_e32 v139, 0x3c080, v138
	s_mov_b32 m0, s44
	s_nop 0
	buffer_load_dwordx4 v139, s[20:23], 0 offen lds
	v_add_u32_e32 v139, 0x78080, v138
	s_mov_b32 m0, s42
	s_nop 0
	buffer_load_dwordx4 v139, s[20:23], 0 offen lds
	v_add_u32_e32 v139, 0xb4080, v138
	s_mov_b32 m0, s41
	s_nop 0
	buffer_load_dwordx4 v139, s[20:23], 0 offen lds
	v_add_u32_e32 v139, s38, v104
	v_add_u32_e32 v140, 0x80, v139
	v_mul_u32_u24_sdwa v140, v140, s46 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshrrev_b32_e32 v140, 23, v140
	v_add_u32_e32 v141, v115, v140
	v_mul_u32_u24_e32 v141, 0xf00, v141
	v_mul_i32_i24_e32 v142, 0xfffff100, v140
	v_mad_i32_i24 v141, v140, s47, v141
	v_add_u32_e32 v140, v116, v140
	v_mul_u32_u24_e32 v140, 0xf00, v140
	v_add3_u32 v141, v124, v141, s48
	s_mov_b32 m0, s1
	v_add3_u32 v140, v142, v140, v124
	buffer_load_dwordx4 v141, s[24:27], 0 offen lds
	v_add_u32_e32 v141, 0xfffc3800, v140
	s_mov_b32 m0, s40
	v_add_u32_e32 v140, 0xfffff800, v140
	buffer_load_dwordx4 v141, s[24:27], 0 offen lds
	s_mov_b32 m0, s0
	s_nop 0
	buffer_load_dwordx4 v140, s[24:27], 0 offen lds
	ds_read_b128 v[140:143], v101
	ds_read_b128 v[144:147], v101 offset:2048
	ds_read_b128 v[148:151], v101 offset:4096
	ds_read_b128 v[152:155], v101 offset:6144
	ds_read_b128 v[156:159], v131
	ds_read_b128 v[160:163], v131 offset:2048
	ds_read_b128 v[164:167], v131 offset:4096
	ds_read_b128 v[168:171], v131 offset:6144
	ds_read_b128 v[172:175], v131 offset:8192
	ds_read_b128 v[176:179], v131 offset:10240
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(9) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[140:143], v[156:159], v[0:3], v118, v122 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[140:143], v[160:163], v[4:7], v118, v122 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(8)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[140:143], v[164:167], v[8:11], v118, v121 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[140:143], v[168:171], v[12:15], v118, v121 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[140:143], v[172:175], v[16:19], v118, v120 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[140:143], v[176:179], v[20:23], v118, v120 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[144:147], v[156:159], v[24:27], v118, v122 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[144:147], v[160:163], v[28:31], v118, v122 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[144:147], v[164:167], v[32:35], v118, v121 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[144:147], v[168:171], v[36:39], v118, v121 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[144:147], v[172:175], v[40:43], v118, v120 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[144:147], v[176:179], v[44:47], v118, v120 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[148:151], v[156:159], v[48:51], v117, v122 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[148:151], v[160:163], v[52:55], v117, v122 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[148:151], v[164:167], v[56:59], v117, v121 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[148:151], v[168:171], v[60:63], v117, v121 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[148:151], v[172:175], v[64:67], v117, v120 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[148:151], v[176:179], v[68:71], v117, v120 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[152:155], v[156:159], v[72:75], v117, v122 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[152:155], v[160:163], v[76:79], v117, v122 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[80:83], v[152:155], v[164:167], v[80:83], v117, v121 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[84:87], v[152:155], v[168:171], v[84:87], v117, v121 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[152:155], v[172:175], v[88:91], v117, v120 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[152:155], v[176:179], v[92:95], v117, v120 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_barrier
	v_add_u32_e32 v141, v129, v130
	v_add_u32_e32 v140, 64, v137
	v_add_u32_e32 v142, 0xffbf, v141
	v_cmp_gt_i32_e32 vcc, s49, v137
	v_add_u32_e32 v144, 4, v137
	v_add_u32_e32 v145, 0xfffb, v141
	v_cndmask_b32_e32 v142, v140, v142, vcc
	v_mul_i32_i24_sdwa v143, sext(v142), s50 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_add_u16_sdwa v142, v143, v142 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_lshrrev_b16_e32 v143, 15, v142
	v_ashrrev_i16_e32 v142, 5, v142
	v_add_u16_e32 v142, v142, v143
	v_cndmask_b32_e64 v143, 0, -1, vcc
	v_cmp_gt_i32_e32 vcc, -4, v137
	v_xor_b32_e32 v142, v142, v143
	v_add_u32_sdwa v143, v99, sext(v142) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_cndmask_b32_e32 v144, v144, v145, vcc
	v_mul_i32_i24_sdwa v145, sext(v144), s50 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_add_u16_sdwa v144, v145, v144 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_lshrrev_b16_e32 v145, 15, v144
	v_ashrrev_i16_e32 v144, 5, v144
	v_add_u16_e32 v144, v144, v145
	v_cndmask_b32_e64 v145, 0, -1, vcc
	v_xor_b32_e32 v144, v144, v145
	v_add_u32_sdwa v145, v125, sext(v142) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_add_u32_sdwa v146, v100, sext(v142) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_add_u32_sdwa v147, v127, sext(v142) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_add_u32_sdwa v142, v128, sext(v142) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_bfe_i32 v144, v144, 0, 16
	v_mul_lo_u32 v146, v146, s12
	v_mul_lo_u32 v142, s12, v142
	v_mul_lo_u32 v143, v143, s14
	v_mul_lo_u32 v145, s14, v145
	v_mad_i32_i24 v146, v144, s39, v146
	v_mul_lo_u32 v147, s12, v147
	v_mad_i32_i24 v142, v144, s39, v142
	v_mad_i32_i24 v143, v144, s39, v143
	v_mad_i32_i24 v145, v144, s39, v145
	v_add3_u32 v146, v119, v146, v126
	v_mad_i32_i24 v147, v144, s39, v147
	v_add3_u32 v142, v119, v142, v126
	v_add3_u32 v143, v119, v143, v135
	v_add3_u32 v145, v119, v145, v135
	v_add3_u32 v147, v119, v147, v126
	buffer_load_dword v144, v146, s[16:19], 0 offen offset:16
	s_nop 0
	buffer_load_dword v146, v147, s[16:19], 0 offen offset:16
	s_nop 0
	buffer_load_dword v142, v142, s[16:19], 0 offen offset:16
	s_nop 0
	buffer_load_dword v143, v143, s[4:7], 0 offen offset:16
	s_nop 0
	buffer_load_dword v145, v145, s[4:7], 0 offen offset:16
	ds_read_b128 v[148:151], v102
	ds_read_b128 v[152:155], v102 offset:2048
	ds_read_b128 v[156:159], v102 offset:4096
	ds_read_b128 v[160:163], v102 offset:6144
	ds_read_b128 v[164:167], v132
	ds_read_b128 v[168:171], v132 offset:2048
	ds_read_b128 v[172:175], v132 offset:4096
	ds_read_b128 v[176:179], v132 offset:6144
	ds_read_b128 v[180:183], v132 offset:8192
	ds_read_b128 v[184:187], v132 offset:10240
	s_waitcnt vmcnt(7)
	s_barrier
	s_setprio 1
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[148:151], v[164:167], v[0:3], v118, v122 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[148:151], v[168:171], v[4:7], v118, v122 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[148:151], v[172:175], v[8:11], v118, v121 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[148:151], v[176:179], v[12:15], v118, v121 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[148:151], v[180:183], v[16:19], v118, v120 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[148:151], v[184:187], v[20:23], v118, v120 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[152:155], v[164:167], v[24:27], v118, v122 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[152:155], v[168:171], v[28:31], v118, v122 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[152:155], v[172:175], v[32:35], v118, v121 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[152:155], v[176:179], v[36:39], v118, v121 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[152:155], v[180:183], v[40:43], v118, v120 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[152:155], v[184:187], v[44:47], v118, v120 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[156:159], v[164:167], v[48:51], v117, v122 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[156:159], v[168:171], v[52:55], v117, v122 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[156:159], v[172:175], v[56:59], v117, v121 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[156:159], v[176:179], v[60:63], v117, v121 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[156:159], v[180:183], v[64:67], v117, v120 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[156:159], v[184:187], v[68:71], v117, v120 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[160:163], v[164:167], v[72:75], v117, v122 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[160:163], v[168:171], v[76:79], v117, v122 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[80:83], v[160:163], v[172:175], v[80:83], v117, v121 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[84:87], v[160:163], v[176:179], v[84:87], v117, v121 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[160:163], v[180:183], v[88:91], v117, v120 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[160:163], v[184:187], v[92:95], v117, v120 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	s_mov_b32 m0, s30
	v_add_u32_e32 v117, 0x100, v138
	s_waitcnt vmcnt(5)
	s_barrier
	buffer_load_dwordx4 v117, s[20:23], 0 offen lds
	v_add_u32_e32 v117, 0x3c100, v138
	s_mov_b32 m0, s31
	s_nop 0
	buffer_load_dwordx4 v117, s[20:23], 0 offen lds
	v_add_u32_e32 v117, 0x78100, v138
	s_mov_b32 m0, s33
	s_nop 0
	buffer_load_dwordx4 v117, s[20:23], 0 offen lds
	v_add_u32_e32 v117, 0xb4100, v138
	s_mov_b32 m0, s34
	s_nop 0
	buffer_load_dwordx4 v117, s[20:23], 0 offen lds
	v_add_u32_e32 v117, 0x100, v139
	v_mul_u32_u24_sdwa v117, v117, s46 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshrrev_b32_e32 v117, 23, v117
	v_add_u32_e32 v118, v115, v117
	v_mul_u32_u24_e32 v118, 0xf00, v118
	v_mul_i32_i24_e32 v120, 0xfffff100, v117
	v_mad_i32_i24 v118, v117, s47, v118
	v_add_u32_e32 v117, v116, v117
	v_mul_u32_u24_e32 v117, 0xf00, v117
	v_add3_u32 v118, v124, v118, s51
	s_mov_b32 m0, s35
	v_add3_u32 v117, v120, v117, v124
	buffer_load_dwordx4 v118, s[24:27], 0 offen lds
	v_add_u32_e32 v118, 0xfffc4000, v117
	s_mov_b32 m0, s36
	s_nop 0
	buffer_load_dwordx4 v118, s[24:27], 0 offen lds
	s_mov_b32 m0, s37
	s_nop 0
	buffer_load_dwordx4 v117, s[24:27], 0 offen lds
	ds_read_b128 v[148:151], v101 offset:32768
	ds_read_b128 v[152:155], v101 offset:34816
	ds_read_b128 v[156:159], v101 offset:36864
	ds_read_b128 v[160:163], v101 offset:38912
	ds_read_b128 v[164:167], v133
	ds_read_b128 v[168:171], v133 offset:2048
	ds_read_b128 v[172:175], v133 offset:4096
	ds_read_b128 v[176:179], v133 offset:6144
	ds_read_b128 v[180:183], v133 offset:8192
	ds_read_b128 v[184:187], v133 offset:10240
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(8) lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[148:151], v[164:167], v[0:3], v143, v144 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[148:151], v[168:171], v[4:7], v143, v144 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[148:151], v[172:175], v[8:11], v143, v146 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[148:151], v[176:179], v[12:15], v143, v146 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[148:151], v[180:183], v[16:19], v143, v142 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[148:151], v[184:187], v[20:23], v143, v142 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[152:155], v[164:167], v[24:27], v143, v144 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[152:155], v[168:171], v[28:31], v143, v144 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[152:155], v[172:175], v[32:35], v143, v146 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[152:155], v[176:179], v[36:39], v143, v146 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[152:155], v[180:183], v[40:43], v143, v142 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[152:155], v[184:187], v[44:47], v143, v142 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[156:159], v[164:167], v[48:51], v145, v144 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[156:159], v[168:171], v[52:55], v145, v144 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[156:159], v[172:175], v[56:59], v145, v146 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[156:159], v[176:179], v[60:63], v145, v146 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[156:159], v[180:183], v[64:67], v145, v142 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[156:159], v[184:187], v[68:71], v145, v142 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[160:163], v[164:167], v[72:75], v145, v144 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[160:163], v[168:171], v[76:79], v145, v144 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[80:83], v[160:163], v[172:175], v[80:83], v145, v146 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[84:87], v[160:163], v[176:179], v[84:87], v145, v146 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[160:163], v[180:183], v[88:91], v145, v142 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[160:163], v[184:187], v[92:95], v145, v142 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_barrier
	v_add_u32_e32 v138, 0x80, v137
	v_add_u32_e32 v117, 0xff7f, v141
	v_cmp_gt_i32_e32 vcc, s49, v140
	v_add_u32_e32 v121, 0xffbb, v141
	ds_read_b128 v[148:151], v102 offset:32768
	ds_read_b128 v[152:155], v102 offset:34816
	ds_read_b128 v[156:159], v102 offset:36864
	ds_read_b128 v[160:163], v102 offset:38912
	ds_read_b128 v[164:167], v134
	ds_read_b128 v[168:171], v134 offset:2048
	ds_read_b128 v[172:175], v134 offset:4096
	ds_read_b128 v[176:179], v134 offset:6144
	ds_read_b128 v[180:183], v134 offset:8192
	ds_read_b128 v[184:187], v134 offset:10240
	v_cndmask_b32_e32 v117, v138, v117, vcc
	v_mul_i32_i24_sdwa v118, sext(v117), s50 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_add_u16_sdwa v117, v118, v117 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_lshrrev_b16_e32 v118, 15, v117
	v_ashrrev_i16_e32 v117, 5, v117
	v_add_u16_e32 v117, v117, v118
	v_cndmask_b32_e64 v118, 0, -1, vcc
	v_xor_b32_e32 v120, v117, v118
	v_add_u32_e32 v118, 0x44, v137
	v_cmp_gt_i32_e32 vcc, -4, v140
	v_add_u32_sdwa v117, v99, sext(v120) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_mul_lo_u32 v117, v117, s14
	v_cndmask_b32_e32 v118, v118, v121, vcc
	v_mul_i32_i24_sdwa v121, sext(v118), s50 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_add_u16_sdwa v118, v121, v118 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_lshrrev_b16_e32 v121, 15, v118
	v_ashrrev_i16_e32 v118, 5, v118
	v_add_u16_e32 v118, v118, v121
	v_cndmask_b32_e64 v121, 0, -1, vcc
	v_xor_b32_e32 v118, v118, v121
	v_bfe_i32 v121, v118, 0, 16
	v_add_u32_sdwa v118, v125, sext(v120) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_mul_lo_u32 v118, s14, v118
	v_mad_i32_i24 v117, v121, s39, v117
	v_mad_i32_i24 v118, v121, s39, v118
	v_add3_u32 v117, v119, v117, v135
	v_add3_u32 v122, v119, v118, v135
	buffer_load_dword v118, v117, s[4:7], 0 offen offset:272
	s_nop 0
	buffer_load_dword v117, v122, s[4:7], 0 offen offset:272
	v_add_u32_sdwa v122, v100, sext(v120) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_add_u32_sdwa v137, v127, sext(v120) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_add_u32_sdwa v120, v128, sext(v120) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_mul_lo_u32 v122, v122, s12
	v_mul_lo_u32 v120, s12, v120
	v_mad_i32_i24 v122, v121, s39, v122
	v_mul_lo_u32 v137, s12, v137
	v_mad_i32_i24 v120, v121, s39, v120
	v_add3_u32 v122, v119, v122, v126
	v_mad_i32_i24 v137, v121, s39, v137
	v_add3_u32 v120, v119, v120, v126
	v_add3_u32 v137, v119, v137, v126
	buffer_load_dword v122, v122, s[16:19], 0 offen offset:272
	s_nop 0
	buffer_load_dword v121, v137, s[16:19], 0 offen offset:272
	s_nop 0
	buffer_load_dword v120, v120, s[16:19], 0 offen offset:272
	s_waitcnt vmcnt(7)
	s_barrier
	s_setprio 1
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[148:151], v[164:167], v[0:3], v143, v144 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[148:151], v[168:171], v[4:7], v143, v144 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[148:151], v[172:175], v[8:11], v143, v146 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[148:151], v[176:179], v[12:15], v143, v146 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[148:151], v[180:183], v[16:19], v143, v142 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[148:151], v[184:187], v[20:23], v143, v142 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[152:155], v[164:167], v[24:27], v143, v144 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[152:155], v[168:171], v[28:31], v143, v144 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[152:155], v[172:175], v[32:35], v143, v146 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[152:155], v[176:179], v[36:39], v143, v146 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[152:155], v[180:183], v[40:43], v143, v142 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[152:155], v[184:187], v[44:47], v143, v142 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[156:159], v[164:167], v[48:51], v145, v144 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[156:159], v[168:171], v[52:55], v145, v144 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[156:159], v[172:175], v[56:59], v145, v146 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[156:159], v[176:179], v[60:63], v145, v146 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[156:159], v[180:183], v[64:67], v145, v142 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[156:159], v[184:187], v[68:71], v145, v142 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[160:163], v[164:167], v[72:75], v145, v144 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[160:163], v[168:171], v[76:79], v145, v144 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[80:83], v[160:163], v[172:175], v[80:83], v145, v146 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[84:87], v[160:163], v[176:179], v[84:87], v145, v146 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[160:163], v[180:183], v[88:91], v145, v142 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[160:163], v[184:187], v[92:95], v145, v142 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	s_add_i32 s43, s43, 2
	s_addk_i32 s38, 0x100
	v_add_u32_e32 v124, 0x1000, v124
	v_add_u32_e32 v136, 0x100, v136
	v_add_u32_e32 v135, 0x200, v135
	v_add_u32_e32 v126, 0x200, v126
	v_add_u32_e32 v130, 0xffffff80, v130
	s_cmp_lt_u32 s43, 26
	v_mov_b32_e32 v137, v138
	s_cbranch_scc1 .LBB0_3
	s_movk_i32 s6, 0xe80
	s_mov_b32 m0, s45
	v_add3_u32 v109, v104, v109, s6
	s_waitcnt vmcnt(5)
	s_barrier
	buffer_load_dwordx4 v109, s[20:23], 0 offen lds
	v_add3_u32 v109, v104, v112, s6
	s_mov_b32 m0, s44
	v_or3_b32 v103, s29, v103, 15
	buffer_load_dwordx4 v109, s[20:23], 0 offen lds
	v_add3_u32 v109, v104, v113, s6
	s_mov_b32 m0, s42
	v_add3_u32 v104, v104, v114, s6
	buffer_load_dwordx4 v109, s[20:23], 0 offen lds
	s_mov_b32 m0, s41
	v_mul_u32_u24_e32 v103, 0xf00, v103
	s_movk_i32 s6, 0x700
	buffer_load_dwordx4 v104, s[20:23], 0 offen lds
	v_add3_u32 v103, v103, v111, s6
	s_mov_b32 s26, s22
	s_mov_b32 s27, s23
	s_mov_b32 m0, s1
	s_nop 0
	buffer_load_dwordx4 v103, s[24:27], 0 offen lds
	v_add_u32_e32 v103, 0x4a100, v123
	v_add_u32_e32 v103, v103, v111
	v_add_u32_e32 v103, 0x700, v103
	s_mov_b32 m0, s40
	s_nop 0
	buffer_load_dwordx4 v103, s[24:27], 0 offen lds
	v_add_u32_e32 v103, 0x86100, v123
	v_add_u32_e32 v103, v103, v111
	v_add_u32_e32 v103, 0x700, v103
	s_mov_b32 m0, s0
	s_nop 0
	buffer_load_dwordx4 v103, s[24:27], 0 offen lds
	v_add_u32_e32 v103, 0x10000, v108
	ds_read_b128 v[144:147], v103
	ds_read_b128 v[148:151], v103 offset:2048
	ds_read_b128 v[152:155], v103 offset:4096
	ds_read_b128 v[156:159], v103 offset:6144
	ds_read_b128 v[160:163], v103 offset:8192
	ds_read_b128 v[168:171], v103 offset:10240
	ds_read_b128 v[112:115], v101
	ds_read_b128 v[124:127], v101 offset:2048
	ds_read_b128 v[140:143], v101 offset:4096
	ds_read_b128 v[164:167], v101 offset:6144
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(9) lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[112:115], v[144:147], v[0:3], v118, v122 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[112:115], v[148:151], v[4:7], v118, v122 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(8)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[112:115], v[152:155], v[8:11], v118, v121 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[112:115], v[156:159], v[12:15], v118, v121 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[112:115], v[160:163], v[16:19], v118, v120 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[112:115], v[168:171], v[20:23], v118, v120 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[124:127], v[144:147], v[24:27], v118, v122 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[124:127], v[148:151], v[28:31], v118, v122 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[124:127], v[152:155], v[32:35], v118, v121 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[124:127], v[156:159], v[36:39], v118, v121 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[124:127], v[160:163], v[40:43], v118, v120 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[124:127], v[168:171], v[44:47], v118, v120 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[112:115], v[140:143], v[144:147], v[48:51], v117, v122 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[124:127], v[140:143], v[148:151], v[52:55], v117, v122 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[128:131], v[140:143], v[152:155], v[56:59], v117, v121 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[132:135], v[140:143], v[156:159], v[60:63], v117, v121 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[136:139], v[140:143], v[160:163], v[64:67], v117, v120 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[140:143], v[140:143], v[168:171], v[68:71], v117, v120 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[144:147], v[164:167], v[144:147], v[72:75], v117, v122 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[148:151], v[164:167], v[148:151], v[76:79], v117, v122 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[152:155], v[164:167], v[152:155], v[80:83], v117, v121 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[156:159], v[164:167], v[156:159], v[84:87], v117, v121 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[160:163], v[164:167], v[160:163], v[88:91], v117, v120 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[164:167], v[164:167], v[168:171], v[92:95], v117, v120 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_barrier
	v_add_u16_e32 v48, 0x740, v105
	v_mul_u32_u24_e32 v48, 0x8889, v48
	v_lshrrev_b32_e32 v49, 21, v48
	v_add_u16_e32 v48, 0x704, v106
	v_mul_u32_u24_e32 v48, 0x889, v48
	v_lshrrev_b32_e32 v48, 17, v48
	v_mul_i32_i24_e32 v48, 0xffffff10, v48
	s_movk_i32 s0, 0x1c10
	v_add_u32_e32 v50, v99, v49
	v_add3_u32 v48, v107, v48, s0
	v_mad_u64_u32 v[50:51], s[0:1], s14, v50, v[48:49]
	v_add_u32_e32 v49, v100, v49
	v_mad_u64_u32 v[48:49], s[0:1], s12, v49, v[48:49]
	s_mov_b32 s6, s22
	s_mov_b32 s7, s23
	s_mov_b32 s18, s22
	s_mov_b32 s19, s23
	v_add_u32_e32 v49, s13, v48
	v_add_u32_e32 v51, s15, v50
	buffer_load_dword v107, v50, s[4:7], 0 offen
	buffer_load_dword v103, v51, s[4:7], 0 offen
	v_add_u32_e32 v50, s13, v49
	buffer_load_dword v106, v48, s[16:19], 0 offen
	buffer_load_dword v105, v49, s[16:19], 0 offen
	buffer_load_dword v104, v50, s[16:19], 0 offen
	v_add_u32_e32 v48, 0x10000, v110
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
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[50:53], v[180:183], v[0:3], v118, v122 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[50:53], v[184:187], v[4:7], v118, v122 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[84:87], v[50:53], v[188:191], v[8:11], v118, v121 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[80:83], v[50:53], v[192:195], v[12:15], v118, v121 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[50:53], v[196:199], v[16:19], v118, v120 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[50:53], v[200:203], v[20:23], v118, v120 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[168:171], v[180:183], v[24:27], v118, v122 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[168:171], v[184:187], v[28:31], v118, v122 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[168:171], v[188:191], v[32:35], v118, v121 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[168:171], v[192:195], v[36:39], v118, v121 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[168:171], v[196:199], v[40:43], v118, v120 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[168:171], v[200:203], v[44:47], v118, v120 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[172:175], v[180:183], v[112:115], v117, v122 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[172:175], v[184:187], v[124:127], v117, v122 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[172:175], v[188:191], v[128:131], v117, v121 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[172:175], v[192:195], v[132:135], v117, v121 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[172:175], v[196:199], v[136:139], v117, v120 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[172:175], v[200:203], v[140:143], v117, v120 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[176:179], v[180:183], v[144:147], v117, v122 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[176:179], v[184:187], v[148:151], v117, v122 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[176:179], v[188:191], v[152:155], v117, v121 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[176:179], v[192:195], v[156:159], v117, v121 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[176:179], v[196:199], v[160:163], v117, v120 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[176:179], v[200:203], v[164:167], v117, v120 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	s_andn2_b64 vcc, exec, s[2:3]
	s_cbranch_vccnz .LBB0_6
	s_barrier
.LBB0_6:
	v_add_u32_e32 v108, 0x16000, v108
	s_waitcnt vmcnt(5)
	s_barrier
	ds_read_b128 v[144:147], v108
	ds_read_b128 v[148:151], v108 offset:2048
	v_add_u32_e32 v109, 0x16000, v110
	ds_read_b128 v[152:155], v109
	ds_read_b128 v[156:159], v109 offset:2048
	ds_read_b128 v[160:163], v108 offset:4096
	ds_read_b128 v[164:167], v108 offset:6144
	ds_read_b128 v[168:171], v109 offset:4096
	ds_read_b128 v[172:175], v109 offset:6144
	ds_read_b128 v[176:179], v108 offset:8192
	ds_read_b128 v[112:115], v108 offset:10240
	ds_read_b128 v[180:183], v109 offset:8192
	ds_read_b128 v[108:111], v109 offset:10240
	ds_read_b128 v[136:139], v101 offset:32768
	ds_read_b128 v[184:187], v101 offset:34816
	ds_read_b128 v[140:143], v102 offset:32768
	ds_read_b128 v[188:191], v102 offset:34816
	ds_read_b128 v[192:195], v101 offset:36864
	ds_read_b128 v[120:123], v101 offset:38912
	ds_read_b128 v[196:199], v102 offset:36864
	ds_read_b128 v[116:119], v102 offset:38912
	s_waitcnt vmcnt(2) lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[124:127], v[136:139], v[144:147], v[92:95], v107, v106 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_movk_i32 s0, 0x7fff
	s_mul_hi_u32 s1, s8, s28
	s_nop 0
	v_mov_b32_e32 v92, 0x7fc0
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[124:127], v[140:143], v[152:155], v[124:127], v107, v106 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_lshlrev_b32_e32 v96, 1, v96
	v_lshl_add_u32 v96, v97, 1, v96
	v_lshl_add_u32 v96, v100, 1, v96
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[136:139], v[148:151], v[88:91], v107, v106 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_mov_b32 s3, 0x27000
	s_nop 2
	v_bfe_u32 v93, v127, 16, 1
	v_bfe_u32 v94, v126, 16, 1
	s_waitcnt vmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[84:87], v[136:139], v[160:163], v[84:87], v107, v105 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v93, v127, v93, s0
	v_bfe_u32 v95, v125, 16, 1
	v_add3_u32 v94, v126, v94, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[140:143], v[156:159], v[88:91], v107, v106 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v93, 16, v93
	v_cmp_o_f32_e32 vcc, v127, v127
	v_bfe_u32 v101, v124, 16, 1
	v_add3_u32 v95, v125, v95, s0
	v_lshrrev_b32_e32 v94, 16, v94
	v_mfma_scale_f32_16x16x128_f8f6f4 v[128:131], v[140:143], v[168:171], v[84:87], v107, v105 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_add3_u32 v101, v124, v101, s0
	s_nop 1
	v_cndmask_b32_e32 v84, v92, v93, vcc
	v_cmp_o_f32_e32 vcc, v126, v126
	s_waitcnt vmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[136:139], v[176:179], v[76:79], v107, v104 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v85, 16, v95
	v_lshrrev_b32_e32 v86, 16, v101
	v_bfe_u32 v87, v88, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[132:135], v[136:139], v[164:167], v[80:83], v107, v105 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 2
	v_cndmask_b32_e32 v80, v92, v94, vcc
	v_cmp_o_f32_e32 vcc, v125, v125
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[136:139], v[112:115], v[72:75], v107, v104 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v83, v91, 16, 1
	v_cndmask_b32_e32 v81, v92, v85, vcc
	v_cmp_o_f32_e32 vcc, v124, v124
	v_bfe_u32 v85, v90, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[124:127], v[140:143], v[180:183], v[76:79], v107, v104 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v82, v92, v86, vcc
	v_bfe_u32 v86, v89, 16, 1
	v_cmp_o_f32_e32 vcc, v91, v91
	v_add3_u32 v76, v88, v87, s0
	v_add3_u32 v77, v89, v86, s0
	v_add3_u32 v79, v91, v83, s0
	v_add3_u32 v78, v90, v85, s0
	v_lshrrev_b32_e32 v79, 16, v79
	v_mfma_scale_f32_16x16x128_f8f6f4 v[136:139], v[140:143], v[108:111], v[72:75], v107, v104 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v83, 16, v76
	v_lshrrev_b32_e32 v78, 16, v78
	v_bfe_u32 v85, v126, 16, 1
	v_lshrrev_b32_e32 v73, 16, v77
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[184:187], v[144:147], v[68:71], v107, v106 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v72, v92, v79, vcc
	v_cmp_o_f32_e32 vcc, v90, v90
	v_bfe_u32 v86, v125, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[132:135], v[140:143], v[172:175], v[132:135], v107, v105 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v68, v92, v78, vcc
	v_cmp_o_f32_e32 vcc, v89, v89
	v_bfe_u32 v71, v131, 16, 1
	v_add3_u32 v71, v131, v71, s0
	v_cndmask_b32_e32 v69, v92, v73, vcc
	v_cmp_o_f32_e32 vcc, v88, v88
	v_bfe_u32 v73, v130, 16, 1
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[140:143], v[188:191], v[152:155], v[74:77], v107, v106 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v70, v92, v83, vcc
	v_add3_u32 v73, v130, v73, s0
	v_lshrrev_b32_e32 v71, 16, v71
	v_bfe_u32 v74, v129, 16, 1
	v_cmp_o_f32_e32 vcc, v131, v131
	v_bfe_u32 v75, v128, 16, 1
	v_add3_u32 v74, v129, v74, s0
	v_lshrrev_b32_e32 v73, 16, v73
	v_cndmask_b32_e32 v71, v92, v71, vcc
	v_cmp_o_f32_e32 vcc, v130, v130
	v_add3_u32 v75, v128, v75, s0
	v_lshrrev_b32_e32 v74, 16, v74
	v_cndmask_b32_e32 v73, v92, v73, vcc
	v_cmp_o_f32_e32 vcc, v129, v129
	v_bfe_u32 v76, v135, 16, 1
	v_lshrrev_b32_e32 v75, 16, v75
	v_cndmask_b32_e32 v74, v92, v74, vcc
	v_cmp_o_f32_e32 vcc, v128, v128
	v_bfe_u32 v77, v134, 16, 1
	v_add3_u32 v76, v135, v76, s0
	v_cndmask_b32_e32 v75, v92, v75, vcc
	v_bfe_u32 v78, v133, 16, 1
	v_add3_u32 v77, v134, v77, s0
	v_lshrrev_b32_e32 v76, 16, v76
	v_cmp_o_f32_e32 vcc, v135, v135
	v_bfe_u32 v79, v132, 16, 1
	v_add3_u32 v78, v133, v78, s0
	v_lshrrev_b32_e32 v77, 16, v77
	v_cndmask_b32_e32 v76, v92, v76, vcc
	v_cmp_o_f32_e32 vcc, v134, v134
	v_add3_u32 v79, v132, v79, s0
	v_lshrrev_b32_e32 v78, 16, v78
	v_cndmask_b32_e32 v77, v92, v77, vcc
	v_cmp_o_f32_e32 vcc, v133, v133
	v_bfe_u32 v83, v127, 16, 1
	v_lshrrev_b32_e32 v79, 16, v79
	v_cndmask_b32_e32 v78, v92, v78, vcc
	v_cmp_o_f32_e32 vcc, v132, v132
	v_add3_u32 v83, v127, v83, s0
	v_add3_u32 v85, v126, v85, s0
	v_cndmask_b32_e32 v79, v92, v79, vcc
	v_lshrrev_b32_e32 v83, 16, v83
	v_cmp_o_f32_e32 vcc, v127, v127
	v_bfe_u32 v87, v124, 16, 1
	v_add3_u32 v86, v125, v86, s0
	v_lshrrev_b32_e32 v85, 16, v85
	v_cndmask_b32_e32 v83, v92, v83, vcc
	v_cmp_o_f32_e32 vcc, v126, v126
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[184:187], v[148:151], v[64:67], v107, v106 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v87, v124, v87, s0
	v_lshrrev_b32_e32 v86, 16, v86
	v_cndmask_b32_e32 v85, v92, v85, vcc
	v_cmp_o_f32_e32 vcc, v125, v125
	v_bfe_u32 v88, v139, 16, 1
	v_lshrrev_b32_e32 v87, 16, v87
	v_cndmask_b32_e32 v86, v92, v86, vcc
	v_cmp_o_f32_e32 vcc, v124, v124
	v_bfe_u32 v89, v138, 16, 1
	v_add3_u32 v88, v139, v88, s0
	v_cndmask_b32_e32 v87, v92, v87, vcc
	v_bfe_u32 v90, v137, 16, 1
	v_add3_u32 v89, v138, v89, s0
	v_lshrrev_b32_e32 v88, 16, v88
	v_cmp_o_f32_e32 vcc, v139, v139
	v_bfe_u32 v91, v136, 16, 1
	v_add3_u32 v90, v137, v90, s0
	v_lshrrev_b32_e32 v89, 16, v89
	v_cndmask_b32_e32 v88, v92, v88, vcc
	v_cmp_o_f32_e32 vcc, v138, v138
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[188:191], v[156:159], v[64:67], v107, v106 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_add3_u32 v91, v136, v91, s0
	v_lshrrev_b32_e32 v90, 16, v90
	v_cndmask_b32_e32 v89, v92, v89, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[184:187], v[160:163], v[60:63], v107, v105 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v137, v137
	v_bfe_u32 v93, v143, 16, 1
	v_lshrrev_b32_e32 v91, 16, v91
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[192:195], v[144:147], v[44:47], v103, v106 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v90, v92, v90, vcc
	v_cmp_o_f32_e32 vcc, v136, v136
	v_bfe_u32 v94, v142, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[192:195], v[148:151], v[40:43], v103, v106 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v93, v143, v93, s0
	v_cndmask_b32_e32 v91, v92, v91, vcc
	v_bfe_u32 v95, v141, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[192:195], v[160:163], v[36:39], v103, v105 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v94, v142, v94, s0
	v_lshrrev_b32_e32 v93, 16, v93
	v_cmp_o_f32_e32 vcc, v143, v143
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[192:195], v[164:167], v[32:35], v103, v105 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v101, v140, 16, 1
	v_add3_u32 v95, v141, v95, s0
	v_lshrrev_b32_e32 v94, 16, v94
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[192:195], v[176:179], v[24:27], v103, v104 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v93, v92, v93, vcc
	v_cmp_o_f32_e32 vcc, v142, v142
	v_add3_u32 v101, v140, v101, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[192:195], v[112:115], v[20:23], v103, v104 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v95, 16, v95
	v_cndmask_b32_e32 v94, v92, v94, vcc
	v_cmp_o_f32_e32 vcc, v141, v141
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[120:123], v[144:147], v[16:19], v103, v106 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v102, v67, 16, 1
	v_lshrrev_b32_e32 v101, 16, v101
	v_cndmask_b32_e32 v95, v92, v95, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[120:123], v[148:151], v[12:15], v103, v106 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v140, v140
	v_add3_u32 v102, v67, v102, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[120:123], v[160:163], v[8:11], v103, v105 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v101, v92, v101, vcc
	v_cmp_o_f32_e32 vcc, v67, v67
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[120:123], v[164:167], v[4:7], v103, v105 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e32 v67, v92, v102, vcc
	v_cmp_o_f32_e32 vcc, v66, v66
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[120:123], v[176:179], v[0:3], v103, v104 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[120:123], v[112:115], v[28:31], v103, v104 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[184:187], v[176:179], v[52:55], v107, v104 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[184:187], v[112:115], v[48:51], v107, v104 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[184:187], v[164:167], v[56:59], v107, v105 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[188:191], v[168:171], v[60:63], v107, v105 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[196:199], v[152:155], v[44:47], v103, v106 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[196:199], v[156:159], v[40:43], v103, v106 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_nop 4
	v_bfe_u32 v102, v63, 16, 1
	v_add3_u32 v102, v63, v102, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[196:199], v[168:171], v[36:39], v103, v105 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[196:199], v[172:175], v[32:35], v103, v105 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[196:199], v[180:183], v[24:27], v103, v104 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[196:199], v[108:111], v[20:23], v103, v104 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[116:119], v[152:155], v[16:19], v103, v106 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[116:119], v[156:159], v[12:15], v103, v106 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[116:119], v[168:171], v[8:11], v103, v105 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[116:119], v[172:175], v[4:7], v103, v105 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[116:119], v[180:183], v[0:3], v103, v104 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[116:119], v[108:111], v[28:31], v103, v104 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_bfe_u32 v103, v66, 16, 1
	v_add3_u32 v103, v66, v103, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[188:191], v[180:183], v[52:55], v107, v104 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v66, v92, v103, vcc
	v_cmp_o_f32_e32 vcc, v65, v65
	v_bfe_u32 v103, v62, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[188:191], v[108:111], v[48:51], v107, v104 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_bfe_u32 v104, v65, 16, 1
	v_add3_u32 v104, v65, v104, s0
	v_lshrrev_b32_e32 v104, 16, v104
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[188:191], v[172:175], v[56:59], v107, v105 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_bfe_u32 v105, v64, 16, 1
	v_add3_u32 v105, v64, v105, s0
	v_lshrrev_b32_e32 v105, 16, v105
	v_cndmask_b32_e32 v65, v92, v104, vcc
	v_cmp_o_f32_e32 vcc, v64, v64
	v_bfe_u32 v104, v61, 16, 1
	v_add3_u32 v103, v62, v103, s0
	v_cndmask_b32_e32 v64, v92, v105, vcc
	v_cmp_o_f32_e32 vcc, v63, v63
	v_bfe_u32 v105, v60, 16, 1
	v_add3_u32 v104, v61, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v63, v92, v102, vcc
	v_cmp_o_f32_e32 vcc, v62, v62
	v_add3_u32 v105, v60, v105, s0
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v62, v92, v103, vcc
	v_cmp_o_f32_e32 vcc, v61, v61
	v_bfe_u32 v102, v59, 16, 1
	v_lshrrev_b32_e32 v105, 16, v105
	v_cndmask_b32_e32 v61, v92, v104, vcc
	v_cmp_o_f32_e32 vcc, v60, v60
	v_bfe_u32 v103, v58, 16, 1
	v_add3_u32 v102, v59, v102, s0
	v_cndmask_b32_e32 v60, v92, v105, vcc
	v_bfe_u32 v104, v57, 16, 1
	v_add3_u32 v103, v58, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cmp_o_f32_e32 vcc, v59, v59
	v_bfe_u32 v105, v56, 16, 1
	v_add3_u32 v104, v57, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v59, v92, v102, vcc
	v_cmp_o_f32_e32 vcc, v58, v58
	v_add3_u32 v105, v56, v105, s0
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v58, v92, v103, vcc
	v_cmp_o_f32_e32 vcc, v57, v57
	v_bfe_u32 v102, v55, 16, 1
	v_lshrrev_b32_e32 v105, 16, v105
	v_cndmask_b32_e32 v57, v92, v104, vcc
	v_cmp_o_f32_e32 vcc, v56, v56
	v_bfe_u32 v103, v54, 16, 1
	v_add3_u32 v102, v55, v102, s0
	v_cndmask_b32_e32 v56, v92, v105, vcc
	v_bfe_u32 v104, v53, 16, 1
	v_add3_u32 v103, v54, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cmp_o_f32_e32 vcc, v55, v55
	v_bfe_u32 v105, v52, 16, 1
	v_add3_u32 v104, v53, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v55, v92, v102, vcc
	v_cmp_o_f32_e32 vcc, v54, v54
	v_add3_u32 v105, v52, v105, s0
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v54, v92, v103, vcc
	v_cmp_o_f32_e32 vcc, v53, v53
	v_bfe_u32 v102, v51, 16, 1
	v_lshrrev_b32_e32 v105, 16, v105
	v_cndmask_b32_e32 v53, v92, v104, vcc
	v_cmp_o_f32_e32 vcc, v52, v52
	v_bfe_u32 v103, v50, 16, 1
	v_add3_u32 v102, v51, v102, s0
	v_cndmask_b32_e32 v52, v92, v105, vcc
	v_bfe_u32 v104, v49, 16, 1
	v_add3_u32 v103, v50, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cmp_o_f32_e32 vcc, v51, v51
	v_bfe_u32 v105, v48, 16, 1
	v_add3_u32 v104, v49, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v51, v92, v102, vcc
	v_cmp_o_f32_e32 vcc, v50, v50
	v_add3_u32 v105, v48, v105, s0
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v50, v92, v103, vcc
	v_cmp_o_f32_e32 vcc, v49, v49
	v_bfe_u32 v102, v47, 16, 1
	v_lshrrev_b32_e32 v105, 16, v105
	v_cndmask_b32_e32 v49, v92, v104, vcc
	v_cmp_o_f32_e32 vcc, v48, v48
	v_bfe_u32 v103, v46, 16, 1
	v_add3_u32 v102, v47, v102, s0
	v_cndmask_b32_e32 v48, v92, v105, vcc
	v_bfe_u32 v104, v45, 16, 1
	v_add3_u32 v103, v46, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cmp_o_f32_e32 vcc, v47, v47
	v_bfe_u32 v105, v44, 16, 1
	v_add3_u32 v104, v45, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v47, v92, v102, vcc
	v_cmp_o_f32_e32 vcc, v46, v46
	v_add3_u32 v105, v44, v105, s0
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v46, v92, v103, vcc
	v_cmp_o_f32_e32 vcc, v45, v45
	v_bfe_u32 v102, v43, 16, 1
	v_lshrrev_b32_e32 v105, 16, v105
	v_cndmask_b32_e32 v45, v92, v104, vcc
	v_cmp_o_f32_e32 vcc, v44, v44
	v_bfe_u32 v103, v42, 16, 1
	v_add3_u32 v102, v43, v102, s0
	v_cndmask_b32_e32 v44, v92, v105, vcc
	v_bfe_u32 v104, v41, 16, 1
	v_add3_u32 v103, v42, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cmp_o_f32_e32 vcc, v43, v43
	v_bfe_u32 v105, v40, 16, 1
	v_add3_u32 v104, v41, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v43, v92, v102, vcc
	v_cmp_o_f32_e32 vcc, v42, v42
	v_add3_u32 v105, v40, v105, s0
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v42, v92, v103, vcc
	v_cmp_o_f32_e32 vcc, v41, v41
	v_bfe_u32 v102, v39, 16, 1
	v_lshrrev_b32_e32 v105, 16, v105
	v_cndmask_b32_e32 v41, v92, v104, vcc
	v_cmp_o_f32_e32 vcc, v40, v40
	v_bfe_u32 v103, v38, 16, 1
	v_add3_u32 v102, v39, v102, s0
	v_cndmask_b32_e32 v40, v92, v105, vcc
	v_bfe_u32 v104, v37, 16, 1
	v_add3_u32 v103, v38, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cmp_o_f32_e32 vcc, v39, v39
	v_bfe_u32 v105, v36, 16, 1
	v_add3_u32 v104, v37, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v39, v92, v102, vcc
	v_cmp_o_f32_e32 vcc, v38, v38
	v_add3_u32 v105, v36, v105, s0
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v38, v92, v103, vcc
	v_cmp_o_f32_e32 vcc, v37, v37
	v_bfe_u32 v102, v35, 16, 1
	v_lshrrev_b32_e32 v105, 16, v105
	v_cndmask_b32_e32 v37, v92, v104, vcc
	v_cmp_o_f32_e32 vcc, v36, v36
	v_bfe_u32 v103, v34, 16, 1
	v_add3_u32 v102, v35, v102, s0
	v_cndmask_b32_e32 v36, v92, v105, vcc
	v_bfe_u32 v104, v33, 16, 1
	v_add3_u32 v103, v34, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cmp_o_f32_e32 vcc, v35, v35
	v_bfe_u32 v105, v32, 16, 1
	v_add3_u32 v104, v33, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v35, v92, v102, vcc
	v_cmp_o_f32_e32 vcc, v34, v34
	v_add3_u32 v105, v32, v105, s0
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v34, v92, v103, vcc
	v_cmp_o_f32_e32 vcc, v33, v33
	v_bfe_u32 v102, v27, 16, 1
	v_lshrrev_b32_e32 v105, 16, v105
	v_cndmask_b32_e32 v33, v92, v104, vcc
	v_cmp_o_f32_e32 vcc, v32, v32
	v_bfe_u32 v103, v26, 16, 1
	v_add3_u32 v102, v27, v102, s0
	v_cndmask_b32_e32 v32, v92, v105, vcc
	v_bfe_u32 v104, v25, 16, 1
	v_add3_u32 v103, v26, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cmp_o_f32_e32 vcc, v27, v27
	v_bfe_u32 v105, v24, 16, 1
	v_add3_u32 v104, v25, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v27, v92, v102, vcc
	v_cmp_o_f32_e32 vcc, v26, v26
	v_add3_u32 v105, v24, v105, s0
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v26, v92, v103, vcc
	v_cmp_o_f32_e32 vcc, v25, v25
	v_bfe_u32 v102, v23, 16, 1
	v_lshrrev_b32_e32 v105, 16, v105
	v_cndmask_b32_e32 v25, v92, v104, vcc
	v_cmp_o_f32_e32 vcc, v24, v24
	v_bfe_u32 v103, v22, 16, 1
	v_add3_u32 v102, v23, v102, s0
	v_cndmask_b32_e32 v24, v92, v105, vcc
	v_bfe_u32 v104, v21, 16, 1
	v_add3_u32 v103, v22, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cmp_o_f32_e32 vcc, v23, v23
	v_bfe_u32 v105, v20, 16, 1
	v_add3_u32 v104, v21, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v23, v92, v102, vcc
	v_cmp_o_f32_e32 vcc, v22, v22
	v_add3_u32 v105, v20, v105, s0
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v22, v92, v103, vcc
	v_cmp_o_f32_e32 vcc, v21, v21
	v_bfe_u32 v102, v19, 16, 1
	v_lshrrev_b32_e32 v105, 16, v105
	v_cndmask_b32_e32 v21, v92, v104, vcc
	v_cmp_o_f32_e32 vcc, v20, v20
	v_bfe_u32 v103, v18, 16, 1
	v_add3_u32 v102, v19, v102, s0
	v_cndmask_b32_e32 v20, v92, v105, vcc
	v_bfe_u32 v104, v17, 16, 1
	v_add3_u32 v103, v18, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cmp_o_f32_e32 vcc, v19, v19
	v_bfe_u32 v105, v16, 16, 1
	v_add3_u32 v104, v17, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v19, v92, v102, vcc
	v_cmp_o_f32_e32 vcc, v18, v18
	v_add3_u32 v105, v16, v105, s0
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v18, v92, v103, vcc
	v_cmp_o_f32_e32 vcc, v17, v17
	v_bfe_u32 v102, v15, 16, 1
	v_lshrrev_b32_e32 v105, 16, v105
	v_cndmask_b32_e32 v17, v92, v104, vcc
	v_cmp_o_f32_e32 vcc, v16, v16
	v_bfe_u32 v103, v14, 16, 1
	v_add3_u32 v102, v15, v102, s0
	v_cndmask_b32_e32 v16, v92, v105, vcc
	v_bfe_u32 v104, v13, 16, 1
	v_add3_u32 v103, v14, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cmp_o_f32_e32 vcc, v15, v15
	v_bfe_u32 v105, v12, 16, 1
	v_add3_u32 v104, v13, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v15, v92, v102, vcc
	v_cmp_o_f32_e32 vcc, v14, v14
	v_add3_u32 v105, v12, v105, s0
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v14, v92, v103, vcc
	v_cmp_o_f32_e32 vcc, v13, v13
	v_bfe_u32 v102, v11, 16, 1
	v_lshrrev_b32_e32 v105, 16, v105
	v_cndmask_b32_e32 v13, v92, v104, vcc
	v_cmp_o_f32_e32 vcc, v12, v12
	v_bfe_u32 v103, v10, 16, 1
	v_add3_u32 v102, v11, v102, s0
	v_cndmask_b32_e32 v12, v92, v105, vcc
	v_bfe_u32 v104, v9, 16, 1
	v_add3_u32 v103, v10, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cmp_o_f32_e32 vcc, v11, v11
	v_bfe_u32 v105, v8, 16, 1
	v_add3_u32 v104, v9, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v11, v92, v102, vcc
	v_cmp_o_f32_e32 vcc, v10, v10
	v_add3_u32 v105, v8, v105, s0
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v10, v92, v103, vcc
	v_cmp_o_f32_e32 vcc, v9, v9
	v_bfe_u32 v102, v7, 16, 1
	v_lshrrev_b32_e32 v105, 16, v105
	v_cndmask_b32_e32 v9, v92, v104, vcc
	v_cmp_o_f32_e32 vcc, v8, v8
	v_bfe_u32 v103, v6, 16, 1
	v_add3_u32 v102, v7, v102, s0
	v_cndmask_b32_e32 v8, v92, v105, vcc
	v_bfe_u32 v104, v5, 16, 1
	v_add3_u32 v103, v6, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cmp_o_f32_e32 vcc, v7, v7
	v_bfe_u32 v105, v4, 16, 1
	v_add3_u32 v104, v5, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v7, v92, v102, vcc
	v_cmp_o_f32_e32 vcc, v6, v6
	v_add3_u32 v105, v4, v105, s0
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v6, v92, v103, vcc
	v_cmp_o_f32_e32 vcc, v5, v5
	v_bfe_u32 v102, v3, 16, 1
	v_lshrrev_b32_e32 v105, 16, v105
	v_cndmask_b32_e32 v5, v92, v104, vcc
	v_cmp_o_f32_e32 vcc, v4, v4
	v_bfe_u32 v103, v2, 16, 1
	v_add3_u32 v102, v3, v102, s0
	v_cndmask_b32_e32 v4, v92, v105, vcc
	v_bfe_u32 v104, v1, 16, 1
	v_add3_u32 v103, v2, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cmp_o_f32_e32 vcc, v3, v3
	v_bfe_u32 v105, v0, 16, 1
	v_add3_u32 v104, v1, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v3, v92, v102, vcc
	v_cmp_o_f32_e32 vcc, v2, v2
	v_add3_u32 v105, v0, v105, s0
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v2, v92, v103, vcc
	v_cmp_o_f32_e32 vcc, v1, v1
	v_lshrrev_b32_e32 v105, 16, v105
	v_bfe_u32 v102, v31, 16, 1
	v_cndmask_b32_e32 v1, v92, v104, vcc
	v_cmp_o_f32_e32 vcc, v0, v0
	v_bfe_u32 v103, v30, 16, 1
	v_bfe_u32 v104, v29, 16, 1
	v_cndmask_b32_e32 v0, v92, v105, vcc
	v_bfe_u32 v105, v28, 16, 1
	v_add3_u32 v105, v28, v105, s0
	v_add3_u32 v104, v29, v104, s0
	v_add3_u32 v103, v30, v103, s0
	v_add3_u32 v102, v31, v102, s0
	s_mul_i32 s0, s9, s28
	s_add_i32 s1, s1, s0
	s_mul_i32 s0, s8, s28
	s_lshl_b64 s[0:1], s[0:1], 1
	v_lshrrev_b32_e32 v102, 16, v102
	v_cmp_o_f32_e32 vcc, v31, v31
	s_add_u32 s0, s10, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v31, v92, v102, vcc
	v_cmp_o_f32_e32 vcc, v30, v30
	s_addc_u32 s1, s11, s1
	s_lshl_b32 s2, s29, 1
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v30, v92, v103, vcc
	v_cmp_o_f32_e32 vcc, v29, v29
	s_add_u32 s0, s0, s2
	v_lshrrev_b32_e32 v105, 16, v105
	v_cndmask_b32_e32 v29, v92, v104, vcc
	v_cmp_o_f32_e32 vcc, v28, v28
	s_addc_u32 s1, s1, 0
	s_and_b32 s2, s8, 0x3fff
	v_cndmask_b32_e32 v28, v92, v105, vcc
	v_lshl_or_b32 v92, v98, 2, v99
	s_lshl_b32 s2, s2, 16
	s_and_b32 s1, s1, 0xffff
	v_mul_lo_u32 v92, s8, v92
	s_or_b32 s1, s2, s1
	s_or_b32 s1, s1, 2.0
	s_mov_b32 s2, 0x7ffffffd
	v_lshl_add_u32 v97, v92, 1, v96
	s_lshl_b32 s4, s8, 1
	buffer_store_short v82, v97, s[0:3], 0 offen
	v_add_u32_e32 v82, s4, v97
	buffer_store_short v81, v82, s[0:3], 0 offen
	v_add_u32_e32 v81, s4, v82
	buffer_store_short v80, v81, s[0:3], 0 offen
	v_add_u32_e32 v80, s4, v81
	s_lshl_b32 s5, s8, 4
	buffer_store_short v84, v80, s[0:3], 0 offen
	buffer_store_short v70, v97, s[0:3], 0 offen offset:32
	buffer_store_short v69, v82, s[0:3], 0 offen offset:32
	buffer_store_short v68, v81, s[0:3], 0 offen offset:32
	buffer_store_short v72, v80, s[0:3], 0 offen offset:32
	buffer_store_short v75, v97, s[0:3], 0 offen offset:64
	buffer_store_short v74, v82, s[0:3], 0 offen offset:64
	buffer_store_short v73, v81, s[0:3], 0 offen offset:64
	buffer_store_short v71, v80, s[0:3], 0 offen offset:64
	buffer_store_short v79, v97, s[0:3], 0 offen offset:96
	buffer_store_short v78, v82, s[0:3], 0 offen offset:96
	buffer_store_short v77, v81, s[0:3], 0 offen offset:96
	buffer_store_short v76, v80, s[0:3], 0 offen offset:96
	buffer_store_short v87, v97, s[0:3], 0 offen offset:128
	buffer_store_short v86, v82, s[0:3], 0 offen offset:128
	buffer_store_short v85, v81, s[0:3], 0 offen offset:128
	buffer_store_short v83, v80, s[0:3], 0 offen offset:128
	buffer_store_short v91, v97, s[0:3], 0 offen offset:160
	buffer_store_short v90, v82, s[0:3], 0 offen offset:160
	buffer_store_short v89, v81, s[0:3], 0 offen offset:160
	buffer_store_short v88, v80, s[0:3], 0 offen offset:160
	v_add_u32_e32 v68, s5, v92
	v_lshl_add_u32 v69, v68, 1, v96
	v_add_u32_e32 v70, s4, v69
	v_add_u32_e32 v71, s4, v70
	v_add_u32_e32 v72, s4, v71
	buffer_store_short v101, v69, s[0:3], 0 offen
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
	v_lshl_add_u32 v49, v48, 1, v96
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
	v_lshl_add_u32 v20, v20, 1, v96
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
	.amdhsa_kernel wave_mxfp4_static_gemm_256x192x256_51712x14976x7680
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
	.size	wave_mxfp4_static_gemm_256x192x256_51712x14976x7680, .Lfunc_end0-wave_mxfp4_static_gemm_256x192x256_51712x14976x7680

	.set wave_mxfp4_static_gemm_256x192x256_51712x14976x7680.num_vgpr, 204
	.set wave_mxfp4_static_gemm_256x192x256_51712x14976x7680.num_agpr, 0
	.set wave_mxfp4_static_gemm_256x192x256_51712x14976x7680.numbered_sgpr, 52
	.set wave_mxfp4_static_gemm_256x192x256_51712x14976x7680.num_named_barrier, 0
	.set wave_mxfp4_static_gemm_256x192x256_51712x14976x7680.private_seg_size, 0
	.set wave_mxfp4_static_gemm_256x192x256_51712x14976x7680.uses_vcc, 1
	.set wave_mxfp4_static_gemm_256x192x256_51712x14976x7680.uses_flat_scratch, 0
	.set wave_mxfp4_static_gemm_256x192x256_51712x14976x7680.has_dyn_sized_stack, 0
	.set wave_mxfp4_static_gemm_256x192x256_51712x14976x7680.has_recursion, 0
	.set wave_mxfp4_static_gemm_256x192x256_51712x14976x7680.has_indirect_call, 0
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
    .name:           wave_mxfp4_static_gemm_256x192x256_51712x14976x7680
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 256
      - 2
      - 1
    .sgpr_count:     58
    .sgpr_spill_count: 0
    .symbol:         wave_mxfp4_static_gemm_256x192x256_51712x14976x7680.kd
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
