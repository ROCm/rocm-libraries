; To reproduce the .rocmasm from .optimized.ll, run:
; llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -mattr='-fma-mix-insts' -O3 <.optimized.ll> -o <out.rocmasm>

	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.text
	.globl	wave_mxfp4_static_gemm_256x192x256_13056x43392x1792
	.p2align	8
	.type	wave_mxfp4_static_gemm_256x192x256_13056x43392x1792,@function
wave_mxfp4_static_gemm_256x192x256_13056x43392x1792:
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
	v_lshlrev_b32_e32 v6, 5, v0
	v_lshl_or_b32 v2, v1, 3, v6
	v_lshrrev_b32_e32 v3, 3, v96
	s_mov_b64 s[24:25], s[2:3]
	v_readfirstlane_b32 s2, v2
	v_or_b32_e32 v2, v3, v6
	s_lshl_b32 s28, s16, 8
	v_or_b32_e32 v2, s28, v2
	v_bitop3_b32 v5, v3, 7, v96 bitop3:0x48
	s_mov_b64 s[20:21], s[6:7]
	v_lshlrev_b32_e32 v103, 4, v5
	v_mul_u32_u24_e32 v104, 0x380, v2
	s_and_b32 s6, s25, 0xffff
	s_lshl_b32 s30, s2, 7
	s_or_b32 s25, s6, 0x43800000
	s_mov_b32 s27, 0x27000
	s_mov_b32 s26, 0x7ffffffe
	v_or_b32_e32 v2, v104, v103
	s_mov_b32 m0, s30
	s_or_b32 s31, s30, 0x2000
	buffer_load_dwordx4 v2, s[24:27], 0 offen lds
	v_add_u32_e32 v4, 0xe000, v2
	s_mov_b32 m0, s31
	s_or_b32 s33, s30, 0x4000
	buffer_load_dwordx4 v4, s[24:27], 0 offen lds
	v_add_u32_e32 v4, 0x1c000, v2
	s_mov_b32 m0, s33
	s_or_b32 s34, s30, 0x6000
	buffer_load_dwordx4 v4, s[24:27], 0 offen lds
	v_add_u32_e32 v2, 0x2a000, v2
	s_mov_b32 m0, s34
	v_bfe_u32 v105, v96, 3, 4
	buffer_load_dwordx4 v2, s[24:27], 0 offen lds
	v_or_b32_e32 v2, v103, v105
	v_mul_lo_u16_e32 v2, 0x93, v2
	v_lshrrev_b32_e32 v9, 7, v96
	v_lshrrev_b16_e32 v7, 13, v2
	v_bitop3_b32 v2, v3, 48, v6 bitop3:0xc8
	v_lshlrev_b32_e32 v3, 4, v3
	v_lshlrev_b32_e32 v4, 8, v9
	s_mul_i32 s29, s17, 0xc0
	v_sub_u32_e32 v10, v3, v4
	v_mul_i32_i24_e32 v11, 0xfffffc80, v7
	v_lshlrev_b32_e32 v5, 8, v5
	s_movk_i32 s3, 0x380
	v_or3_b32 v8, s29, v7, v2
	v_add3_u32 v10, v5, v10, v11
	s_and_b32 s2, s21, 0xffff
	s_add_i32 s35, s30, 0x10000
	s_or_b32 s21, s2, 0x43800000
	s_mov_b32 s22, s26
	s_mov_b32 s23, s27
	v_mad_u32_u24 v8, v8, s3, v10
	s_mov_b32 m0, s35
	s_add_i32 s36, s31, 0x10000
	buffer_load_dwordx4 v8, s[20:23], 0 offen lds
	v_lshlrev_b32_e32 v8, 4, v9
	v_or3_b32 v106, s29, v8, v6
	v_or_b32_e32 v6, v106, v7
	v_mad_u32_u24 v6, v6, s3, v10
	v_add_u32_e32 v7, 0xe000, v6
	s_mov_b32 m0, s36
	s_add_i32 s37, s33, 0x10000
	buffer_load_dwordx4 v7, s[20:23], 0 offen lds
	v_add_u32_e32 v6, 0x1c000, v6
	s_mov_b32 m0, s37
	v_lshrrev_b32_e32 v7, 4, v96
	v_bfe_u32 v98, v96, 4, 2
	buffer_load_dwordx4 v6, s[20:23], 0 offen lds
	v_lshlrev_b32_e32 v6, 4, v98
	v_mul_i32_i24_e32 v97, -16, v7
	v_add3_u32 v128, v6, v97, v96
	v_ashrrev_i32_e32 v8, 31, v128
	v_xor_b32_e32 v9, v8, v128
	s_mov_b32 s2, 0x92492493
	v_mul_hi_i32 v10, v9, s2
	v_add_u32_e32 v9, v10, v9
	v_lshrrev_b32_e32 v10, 31, v9
	v_ashrrev_i32_e32 v9, 3, v9
	v_add_u32_e32 v9, v9, v10
	s_mul_i32 s15, s15, s28
	s_mul_hi_u32 s2, s14, s28
	v_xor_b32_e32 v11, v9, v8
	v_and_b32_e32 v8, 62, v96
	s_add_i32 s2, s2, s15
	s_mul_i32 s3, s14, s28
	v_mul_lo_u16_e32 v8, 37, v8
	s_add_u32 s4, s4, s3
	s_load_dwordx2 s[12:13], s[0:1], 0x40
	v_lshrrev_b16_e32 v8, 9, v8
	v_lshlrev_b32_e32 v109, 2, v96
	v_lshlrev_b32_e32 v9, 6, v7
	s_addc_u32 s2, s5, s2
	s_and_b32 s3, s14, 0x3fff
	v_and_b32_e32 v99, 0xc0, v96
	v_mul_i32_i24_e32 v10, 0xffffffc8, v8
	v_lshlrev_b32_e32 v8, 6, v98
	v_sub_u32_e32 v13, v109, v9
	s_bitset1_b32 s3, 14
	v_add_u32_e32 v12, v11, v99
	v_add3_u32 v10, v13, v8, v10
	s_and_b32 s2, s2, 0xffff
	s_lshl_b32 s3, s3, 16
	s_or_b32 s5, s2, s3
	v_mad_u64_u32 v[12:13], s[2:3], v12, s14, v[10:11]
	s_movk_i32 s2, 0x60
	s_nop 0
	v_mad_u32_u24 v11, v0, s2, v11
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s2, s13, s29
	s_mul_hi_u32 s3, s12, s29
	s_add_i32 s3, s3, s2
	s_mul_i32 s2, s12, s29
	s_add_u32 s16, s8, s2
	s_addc_u32 s2, s9, s3
	s_and_b32 s3, s12, 0x3fff
	s_bitset1_b32 s3, 14
	s_and_b32 s2, s2, 0xffff
	s_lshl_b32 s3, s3, 16
	s_or_b32 s17, s2, s3
	v_mad_u64_u32 v[10:11], s[2:3], s12, v11, v[10:11]
	s_lshl_b32 s2, s12, 5
	s_mov_b32 s6, s26
	s_mov_b32 s7, s27
	s_mov_b32 s18, s26
	s_mov_b32 s19, s27
	v_add_u32_e32 v11, s2, v10
	v_lshl_add_u32 v13, s14, 5, v12
	buffer_load_dword v110, v12, s[4:7], 0 offen
	buffer_load_dword v101, v13, s[4:7], 0 offen
	v_add_u32_e32 v12, s2, v11
	buffer_load_dword v108, v10, s[16:19], 0 offen
	buffer_load_dword v107, v11, s[16:19], 0 offen
	buffer_load_dword v102, v12, s[16:19], 0 offen
	v_cmp_eq_u32_e64 s[2:3], 0, v0
	s_movk_i32 s13, 0xfc80
	s_movk_i32 s15, 0xffc8
	v_mul_u32_u24_e32 v100, 0x60, v0
	s_and_b64 vcc, exec, s[2:3]
	s_barrier
	s_waitcnt vmcnt(0)
	s_cbranch_vccnz .LBB0_2
	s_barrier
.LBB0_2:
	s_load_dwordx2 s[8:9], s[0:1], 0x48
	v_lshlrev_b32_e32 v12, 7, v96
	v_lshlrev_b32_e32 v7, 11, v7
	s_movk_i32 s0, 0x3000
	v_and_b32_e32 v10, 7, v96
	v_sub_u32_e32 v7, v12, v7
	v_mul_lo_u32 v0, v0, s0
	v_bitop3_b32 v11, v98, v96, 7 bitop3:0x78
	v_lshl_add_u32 v1, v1, 13, v7
	v_add_u32_e32 v0, v7, v0
	v_bitop3_b32 v7, v98, v10, 4 bitop3:0x36
	v_lshlrev_b32_e32 v11, 4, v11
	v_lshlrev_b32_e32 v7, 4, v7
	v_or_b32_e32 v113, v0, v11
	v_or_b32_e32 v114, v7, v0
	v_add_u32_e32 v0, v5, v3
	v_sub_u32_e32 v0, v0, v4
	v_add_u32_e32 v116, 0x1c900, v0
	v_sub_u32_e32 v118, v8, v9
	v_add_u32_e32 v0, v97, v96
	v_mov_b32_e32 v44, 0
	v_or_b32_e32 v111, v1, v11
	v_or_b32_e32 v112, v7, v1
	v_or_b32_e32 v115, s29, v2
	v_add_u32_e32 v117, 32, v99
	v_add_u32_e32 v119, 32, v100
	v_add_u32_e32 v120, 64, v100
	v_sub_u32_e32 v121, 0, v6
	v_sub_u32_e32 v122, 0, v0
	s_mov_b32 s0, -2
	s_add_i32 s1, s30, 0x8000
	s_add_i32 s38, s31, 0x8000
	s_add_i32 s39, s33, 0x8000
	s_add_i32 s40, s34, 0x8000
	s_mov_b32 s41, 0xfffe3800
	s_add_i32 s42, s30, 0x16000
	s_mov_b32 s22, s26
	s_mov_b32 s23, s27
	s_add_i32 s43, s31, 0x16000
	s_add_i32 s44, s33, 0x16000
	v_add_u32_e32 v123, 0x10000, v113
	s_movk_i32 s45, 0xffc0
	s_movk_i32 s46, 0x4925
	s_mov_b32 s6, s26
	s_mov_b32 s7, s27
	s_mov_b32 s18, s26
	s_mov_b32 s19, s27
	v_add_u32_e32 v124, 0x10000, v114
	s_mov_b32 s47, 0xfffe4000
	v_add_u32_e32 v125, 0x16000, v113
	v_add_u32_e32 v126, 0x16000, v114
	v_mov_b32_e32 v127, 2
	v_mov_b32_e32 v45, v44
	v_mov_b32_e32 v46, v44
	v_mov_b32_e32 v47, v44
	v_mov_b32_e32 v92, v44
	v_mov_b32_e32 v93, v44
	v_mov_b32_e32 v94, v44
	v_mov_b32_e32 v95, v44
	v_mov_b32_e32 v88, v44
	v_mov_b32_e32 v89, v44
	v_mov_b32_e32 v90, v44
	v_mov_b32_e32 v91, v44
	v_mov_b32_e32 v84, v44
	v_mov_b32_e32 v85, v44
	v_mov_b32_e32 v86, v44
	v_mov_b32_e32 v87, v44
	v_mov_b32_e32 v80, v44
	v_mov_b32_e32 v81, v44
	v_mov_b32_e32 v82, v44
	v_mov_b32_e32 v83, v44
	v_mov_b32_e32 v76, v44
	v_mov_b32_e32 v77, v44
	v_mov_b32_e32 v78, v44
	v_mov_b32_e32 v79, v44
	v_mov_b32_e32 v72, v44
	v_mov_b32_e32 v73, v44
	v_mov_b32_e32 v74, v44
	v_mov_b32_e32 v75, v44
	v_mov_b32_e32 v68, v44
	v_mov_b32_e32 v69, v44
	v_mov_b32_e32 v70, v44
	v_mov_b32_e32 v71, v44
	v_mov_b32_e32 v64, v44
	v_mov_b32_e32 v65, v44
	v_mov_b32_e32 v66, v44
	v_mov_b32_e32 v67, v44
	v_mov_b32_e32 v60, v44
	v_mov_b32_e32 v61, v44
	v_mov_b32_e32 v62, v44
	v_mov_b32_e32 v63, v44
	v_mov_b32_e32 v56, v44
	v_mov_b32_e32 v57, v44
	v_mov_b32_e32 v58, v44
	v_mov_b32_e32 v59, v44
	v_mov_b32_e32 v52, v44
	v_mov_b32_e32 v53, v44
	v_mov_b32_e32 v54, v44
	v_mov_b32_e32 v55, v44
	v_mov_b32_e32 v48, v44
	v_mov_b32_e32 v49, v44
	v_mov_b32_e32 v50, v44
	v_mov_b32_e32 v51, v44
	v_mov_b32_e32 v40, v44
	v_mov_b32_e32 v41, v44
	v_mov_b32_e32 v42, v44
	v_mov_b32_e32 v43, v44
	v_mov_b32_e32 v36, v44
	v_mov_b32_e32 v37, v44
	v_mov_b32_e32 v38, v44
	v_mov_b32_e32 v39, v44
	v_mov_b32_e32 v32, v44
	v_mov_b32_e32 v33, v44
	v_mov_b32_e32 v34, v44
	v_mov_b32_e32 v35, v44
	v_mov_b32_e32 v20, v44
	v_mov_b32_e32 v21, v44
	v_mov_b32_e32 v22, v44
	v_mov_b32_e32 v23, v44
	v_mov_b32_e32 v8, v44
	v_mov_b32_e32 v9, v44
	v_mov_b32_e32 v10, v44
	v_mov_b32_e32 v11, v44
	v_mov_b32_e32 v4, v44
	v_mov_b32_e32 v5, v44
	v_mov_b32_e32 v6, v44
	v_mov_b32_e32 v7, v44
	v_mov_b32_e32 v12, v44
	v_mov_b32_e32 v13, v44
	v_mov_b32_e32 v14, v44
	v_mov_b32_e32 v15, v44
	v_mov_b32_e32 v16, v44
	v_mov_b32_e32 v17, v44
	v_mov_b32_e32 v18, v44
	v_mov_b32_e32 v19, v44
	v_mov_b32_e32 v24, v44
	v_mov_b32_e32 v25, v44
	v_mov_b32_e32 v26, v44
	v_mov_b32_e32 v27, v44
	v_mov_b32_e32 v28, v44
	v_mov_b32_e32 v29, v44
	v_mov_b32_e32 v30, v44
	v_mov_b32_e32 v31, v44
	v_mov_b32_e32 v0, v44
	v_mov_b32_e32 v1, v44
	v_mov_b32_e32 v2, v44
	v_mov_b32_e32 v3, v44
	v_mov_b32_e32 v129, v118
.LBB0_3:
	v_add_u32_e32 v130, v104, v103
	s_mov_b32 m0, s1
	v_add_u32_e32 v131, 0x80, v130
	s_waitcnt vmcnt(5)
	s_barrier
	buffer_load_dwordx4 v131, s[24:27], 0 offen lds
	v_add_u32_e32 v131, 0xe080, v130
	s_mov_b32 m0, s38
	s_nop 0
	buffer_load_dwordx4 v131, s[24:27], 0 offen lds
	v_add_u32_e32 v131, 0x1c080, v130
	s_mov_b32 m0, s39
	s_nop 0
	buffer_load_dwordx4 v131, s[24:27], 0 offen lds
	v_add_u32_e32 v131, 0x2a080, v130
	s_mov_b32 m0, s40
	s_nop 0
	buffer_load_dwordx4 v131, s[24:27], 0 offen lds
	v_add_u32_e32 v131, v103, v105
	v_or_b32_e32 v132, 0x80, v131
	v_lshrrev_b16_e32 v132, 3, v132
	v_add_u16_e32 v134, 16, v131
	v_mul_u32_u24_e32 v132, 0x2493, v132
	v_lshrrev_b16_e32 v134, 3, v134
	v_add_u32_sdwa v133, v115, v132 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_mul_u32_u24_e32 v134, 0x2493, v134
	v_lshrrev_b32_e32 v134, 16, v134
	v_mul_u32_u24_e32 v133, 0x380, v133
	v_add_u32_sdwa v132, v106, v132 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_mul_i32_i24_e32 v135, 0xfffffc80, v134
	v_mad_i32_i24 v133, v134, s13, v133
	v_mul_u32_u24_e32 v132, 0x380, v132
	v_add3_u32 v133, v116, v133, s41
	s_mov_b32 m0, s42
	v_add3_u32 v132, v135, v132, v116
	buffer_load_dwordx4 v133, s[20:23], 0 offen lds
	v_add_u32_e32 v133, 0xffff1800, v132
	s_mov_b32 m0, s43
	v_add_u32_e32 v132, 0xfffff800, v132
	buffer_load_dwordx4 v133, s[20:23], 0 offen lds
	s_mov_b32 m0, s44
	s_nop 0
	buffer_load_dwordx4 v132, s[20:23], 0 offen lds
	ds_read_b128 v[132:135], v111
	ds_read_b128 v[136:139], v111 offset:2048
	ds_read_b128 v[140:143], v111 offset:4096
	ds_read_b128 v[144:147], v111 offset:6144
	ds_read_b128 v[148:151], v123
	ds_read_b128 v[152:155], v123 offset:2048
	ds_read_b128 v[156:159], v123 offset:4096
	ds_read_b128 v[160:163], v123 offset:6144
	ds_read_b128 v[164:167], v123 offset:8192
	ds_read_b128 v[168:171], v123 offset:10240
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(9) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[132:135], v[148:151], v[44:47], v110, v108 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[132:135], v[152:155], v[92:95], v110, v108 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(8)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[132:135], v[156:159], v[88:91], v110, v107 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[84:87], v[132:135], v[160:163], v[84:87], v110, v107 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[80:83], v[132:135], v[164:167], v[80:83], v110, v102 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[132:135], v[168:171], v[76:79], v110, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[136:139], v[148:151], v[72:75], v110, v108 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[136:139], v[152:155], v[68:71], v110, v108 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[136:139], v[156:159], v[64:67], v110, v107 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[136:139], v[160:163], v[60:63], v110, v107 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[136:139], v[164:167], v[56:59], v110, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[136:139], v[168:171], v[52:55], v110, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[140:143], v[148:151], v[48:51], v101, v108 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[140:143], v[152:155], v[40:43], v101, v108 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[140:143], v[156:159], v[36:39], v101, v107 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[140:143], v[160:163], v[32:35], v101, v107 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[140:143], v[164:167], v[20:23], v101, v102 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[140:143], v[168:171], v[8:11], v101, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[144:147], v[148:151], v[4:7], v101, v108 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[144:147], v[152:155], v[12:15], v101, v108 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[144:147], v[156:159], v[16:19], v101, v107 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[144:147], v[160:163], v[24:27], v101, v107 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[144:147], v[164:167], v[28:31], v101, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[144:147], v[168:171], v[0:3], v101, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_barrier
	v_add_u32_e32 v133, v121, v122
	v_add_u32_e32 v132, 64, v128
	v_add_u32_e32 v134, 0xffbf, v133
	v_cmp_gt_i32_e32 vcc, s45, v128
	v_add_u32_e32 v136, 8, v128
	v_add_u32_e32 v137, 0xfff7, v133
	v_cndmask_b32_e32 v134, v132, v134, vcc
	v_mul_i32_i24_sdwa v134, sext(v134), s46 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_ashrrev_i16_sdwa v135, v127, v134 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v134, 31, v134
	v_add_u16_e32 v134, v135, v134
	v_cndmask_b32_e64 v135, 0, -1, vcc
	v_cmp_gt_i32_e32 vcc, -8, v128
	v_xor_b32_e32 v134, v134, v135
	v_add_u32_sdwa v135, v99, sext(v134) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_cndmask_b32_e32 v136, v136, v137, vcc
	v_mul_i32_i24_sdwa v136, sext(v136), s46 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_ashrrev_i16_sdwa v137, v127, v136 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v136, 31, v136
	v_add_u16_e32 v136, v137, v136
	v_cndmask_b32_e64 v137, 0, -1, vcc
	v_xor_b32_e32 v136, v136, v137
	v_add_u32_sdwa v137, v117, sext(v134) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_add_u32_sdwa v138, v100, sext(v134) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_add_u32_sdwa v139, v119, sext(v134) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_add_u32_sdwa v134, v120, sext(v134) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_bfe_i32 v136, v136, 0, 16
	v_mul_lo_u32 v138, v138, s12
	v_mul_lo_u32 v134, s12, v134
	v_mul_lo_u32 v135, v135, s14
	v_mul_lo_u32 v137, s14, v137
	v_mad_i32_i24 v138, v136, s15, v138
	v_mul_lo_u32 v139, s12, v139
	v_mad_i32_i24 v134, v136, s15, v134
	v_mad_i32_i24 v135, v136, s15, v135
	v_mad_i32_i24 v137, v136, s15, v137
	v_add3_u32 v138, v109, v138, v118
	v_mad_i32_i24 v139, v136, s15, v139
	v_add3_u32 v134, v109, v134, v118
	v_add3_u32 v135, v109, v135, v129
	v_add3_u32 v137, v109, v137, v129
	v_add3_u32 v139, v109, v139, v118
	buffer_load_dword v136, v138, s[16:19], 0 offen offset:32
	s_nop 0
	buffer_load_dword v138, v139, s[16:19], 0 offen offset:32
	s_nop 0
	buffer_load_dword v134, v134, s[16:19], 0 offen offset:32
	s_nop 0
	buffer_load_dword v135, v135, s[4:7], 0 offen offset:32
	s_nop 0
	buffer_load_dword v137, v137, s[4:7], 0 offen offset:32
	ds_read_b128 v[140:143], v112
	ds_read_b128 v[144:147], v112 offset:2048
	ds_read_b128 v[148:151], v112 offset:4096
	ds_read_b128 v[152:155], v112 offset:6144
	ds_read_b128 v[156:159], v124
	ds_read_b128 v[160:163], v124 offset:2048
	ds_read_b128 v[164:167], v124 offset:4096
	ds_read_b128 v[168:171], v124 offset:6144
	ds_read_b128 v[172:175], v124 offset:8192
	ds_read_b128 v[176:179], v124 offset:10240
	s_waitcnt vmcnt(7)
	s_barrier
	s_setprio 1
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[140:143], v[156:159], v[44:47], v110, v108 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[140:143], v[160:163], v[92:95], v110, v108 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[140:143], v[164:167], v[88:91], v110, v107 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[84:87], v[140:143], v[168:171], v[84:87], v110, v107 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[80:83], v[140:143], v[172:175], v[80:83], v110, v102 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[140:143], v[176:179], v[76:79], v110, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[144:147], v[156:159], v[72:75], v110, v108 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[144:147], v[160:163], v[68:71], v110, v108 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[144:147], v[164:167], v[64:67], v110, v107 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[144:147], v[168:171], v[60:63], v110, v107 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[144:147], v[172:175], v[56:59], v110, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[144:147], v[176:179], v[52:55], v110, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[148:151], v[156:159], v[48:51], v101, v108 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[148:151], v[160:163], v[40:43], v101, v108 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[148:151], v[164:167], v[36:39], v101, v107 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[148:151], v[168:171], v[32:35], v101, v107 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[148:151], v[172:175], v[20:23], v101, v102 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[148:151], v[176:179], v[8:11], v101, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[152:155], v[156:159], v[4:7], v101, v108 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[152:155], v[160:163], v[12:15], v101, v108 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[152:155], v[164:167], v[16:19], v101, v107 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[152:155], v[168:171], v[24:27], v101, v107 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[152:155], v[172:175], v[28:31], v101, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[152:155], v[176:179], v[0:3], v101, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	s_mov_b32 m0, s30
	v_add_u32_e32 v101, 0x100, v130
	s_waitcnt vmcnt(5)
	s_barrier
	buffer_load_dwordx4 v101, s[24:27], 0 offen lds
	v_add_u32_e32 v101, 0xe100, v130
	s_mov_b32 m0, s31
	v_add_u16_e32 v107, 0x90, v131
	buffer_load_dwordx4 v101, s[24:27], 0 offen lds
	v_add_u32_e32 v101, 0x1c100, v130
	s_mov_b32 m0, s33
	v_lshrrev_b16_e32 v107, 3, v107
	buffer_load_dwordx4 v101, s[24:27], 0 offen lds
	v_add_u32_e32 v101, 0x2a100, v130
	s_mov_b32 m0, s34
	v_mul_u32_u24_e32 v107, 0x2493, v107
	buffer_load_dwordx4 v101, s[24:27], 0 offen lds
	v_add_u16_e32 v101, 0x100, v131
	v_lshrrev_b16_e32 v101, 3, v101
	v_mul_u32_u24_e32 v101, 0x2493, v101
	v_add_u32_sdwa v102, v115, v101 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v107, 16, v107
	v_mul_u32_u24_e32 v102, 0x380, v102
	v_add_u32_sdwa v101, v106, v101 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_mul_i32_i24_e32 v108, 0xfffffc80, v107
	v_mad_i32_i24 v102, v107, s13, v102
	v_mul_u32_u24_e32 v101, 0x380, v101
	v_add3_u32 v102, v116, v102, s47
	s_mov_b32 m0, s35
	v_add3_u32 v101, v108, v101, v116
	buffer_load_dwordx4 v102, s[20:23], 0 offen lds
	v_add_u32_e32 v102, 0xffff2000, v101
	s_mov_b32 m0, s36
	s_nop 0
	buffer_load_dwordx4 v102, s[20:23], 0 offen lds
	s_mov_b32 m0, s37
	s_nop 0
	buffer_load_dwordx4 v101, s[20:23], 0 offen lds
	ds_read_b128 v[140:143], v111 offset:32768
	ds_read_b128 v[144:147], v111 offset:34816
	ds_read_b128 v[148:151], v111 offset:36864
	ds_read_b128 v[152:155], v111 offset:38912
	ds_read_b128 v[156:159], v125
	ds_read_b128 v[160:163], v125 offset:2048
	ds_read_b128 v[164:167], v125 offset:4096
	ds_read_b128 v[168:171], v125 offset:6144
	ds_read_b128 v[172:175], v125 offset:8192
	ds_read_b128 v[176:179], v125 offset:10240
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(8) lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[140:143], v[156:159], v[44:47], v135, v136 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[140:143], v[160:163], v[92:95], v135, v136 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[140:143], v[164:167], v[88:91], v135, v138 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[84:87], v[140:143], v[168:171], v[84:87], v135, v138 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[80:83], v[140:143], v[172:175], v[80:83], v135, v134 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[140:143], v[176:179], v[76:79], v135, v134 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[144:147], v[156:159], v[72:75], v135, v136 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[144:147], v[160:163], v[68:71], v135, v136 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[144:147], v[164:167], v[64:67], v135, v138 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[144:147], v[168:171], v[60:63], v135, v138 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[144:147], v[172:175], v[56:59], v135, v134 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[144:147], v[176:179], v[52:55], v135, v134 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[148:151], v[156:159], v[48:51], v137, v136 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[148:151], v[160:163], v[40:43], v137, v136 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[148:151], v[164:167], v[36:39], v137, v138 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[148:151], v[168:171], v[32:35], v137, v138 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[148:151], v[172:175], v[20:23], v137, v134 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[148:151], v[176:179], v[8:11], v137, v134 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[152:155], v[156:159], v[4:7], v137, v136 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[152:155], v[160:163], v[12:15], v137, v136 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[152:155], v[164:167], v[16:19], v137, v138 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[152:155], v[168:171], v[24:27], v137, v138 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[152:155], v[172:175], v[28:31], v137, v134 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[152:155], v[176:179], v[0:3], v137, v134 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_barrier
	v_add_u32_e32 v130, 0x80, v128
	v_add_u32_e32 v101, 0xff7f, v133
	v_cmp_gt_i32_e32 vcc, s45, v132
	v_add_u32_e32 v107, 0x48, v128
	v_add_u32_e32 v108, 0xffb7, v133
	v_cndmask_b32_e32 v101, v130, v101, vcc
	v_mul_i32_i24_sdwa v101, sext(v101), s46 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshrrev_b32_e32 v102, 31, v101
	v_ashrrev_i32_e32 v101, 18, v101
	v_add_u16_e32 v101, v101, v102
	v_cndmask_b32_e64 v102, 0, -1, vcc
	v_cmp_gt_i32_e32 vcc, -8, v132
	v_xor_b32_e32 v102, v101, v102
	v_add_u32_sdwa v101, v99, sext(v102) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_cndmask_b32_e32 v107, v107, v108, vcc
	v_mul_i32_i24_sdwa v107, sext(v107), s46 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshrrev_b32_e32 v108, 31, v107
	v_ashrrev_i32_e32 v107, 18, v107
	v_add_u16_e32 v107, v107, v108
	v_cndmask_b32_e64 v108, 0, -1, vcc
	v_xor_b32_e32 v107, v107, v108
	v_add_u32_sdwa v108, v117, sext(v102) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_bfe_i32 v107, v107, 0, 16
	v_mul_lo_u32 v101, v101, s14
	v_mul_lo_u32 v108, s14, v108
	v_mad_i32_i24 v101, v107, s15, v101
	v_mad_i32_i24 v108, v107, s15, v108
	v_add3_u32 v101, v109, v101, v129
	v_add3_u32 v108, v109, v108, v129
	buffer_load_dword v110, v101, s[4:7], 0 offen offset:288
	s_nop 0
	buffer_load_dword v101, v108, s[4:7], 0 offen offset:288
	v_add_u32_sdwa v108, v100, sext(v102) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_add_u32_sdwa v128, v119, sext(v102) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_add_u32_sdwa v102, v120, sext(v102) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_mul_lo_u32 v108, v108, s12
	v_mul_lo_u32 v102, s12, v102
	v_mad_i32_i24 v108, v107, s15, v108
	v_mul_lo_u32 v128, s12, v128
	v_mad_i32_i24 v102, v107, s15, v102
	v_add3_u32 v108, v109, v108, v118
	v_mad_i32_i24 v128, v107, s15, v128
	v_add3_u32 v102, v109, v102, v118
	v_add3_u32 v128, v109, v128, v118
	buffer_load_dword v108, v108, s[16:19], 0 offen offset:288
	s_nop 0
	buffer_load_dword v107, v128, s[16:19], 0 offen offset:288
	s_nop 0
	buffer_load_dword v102, v102, s[16:19], 0 offen offset:288
	ds_read_b128 v[140:143], v112 offset:32768
	ds_read_b128 v[144:147], v112 offset:34816
	ds_read_b128 v[148:151], v112 offset:36864
	ds_read_b128 v[152:155], v112 offset:38912
	ds_read_b128 v[156:159], v126
	ds_read_b128 v[160:163], v126 offset:2048
	ds_read_b128 v[164:167], v126 offset:4096
	ds_read_b128 v[168:171], v126 offset:6144
	ds_read_b128 v[172:175], v126 offset:8192
	ds_read_b128 v[176:179], v126 offset:10240
	s_waitcnt vmcnt(7)
	s_barrier
	s_setprio 1
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[140:143], v[156:159], v[44:47], v135, v136 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[140:143], v[160:163], v[92:95], v135, v136 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[140:143], v[164:167], v[88:91], v135, v138 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[84:87], v[140:143], v[168:171], v[84:87], v135, v138 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[80:83], v[140:143], v[172:175], v[80:83], v135, v134 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[140:143], v[176:179], v[76:79], v135, v134 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[144:147], v[156:159], v[72:75], v135, v136 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[144:147], v[160:163], v[68:71], v135, v136 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[144:147], v[164:167], v[64:67], v135, v138 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[144:147], v[168:171], v[60:63], v135, v138 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[144:147], v[172:175], v[56:59], v135, v134 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[144:147], v[176:179], v[52:55], v135, v134 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[148:151], v[156:159], v[48:51], v137, v136 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[148:151], v[160:163], v[40:43], v137, v136 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[148:151], v[164:167], v[36:39], v137, v138 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[148:151], v[168:171], v[32:35], v137, v138 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[148:151], v[172:175], v[20:23], v137, v134 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[148:151], v[176:179], v[8:11], v137, v134 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[152:155], v[156:159], v[4:7], v137, v136 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[152:155], v[160:163], v[12:15], v137, v136 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[152:155], v[164:167], v[16:19], v137, v138 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[152:155], v[168:171], v[24:27], v137, v138 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[152:155], v[172:175], v[28:31], v137, v134 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[152:155], v[176:179], v[0:3], v137, v134 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	s_add_i32 s0, s0, 2
	v_add_u32_e32 v116, 0x1000, v116
	v_add_u32_e32 v104, 0x100, v104
	v_add_u32_e32 v105, 0x100, v105
	v_add_u32_e32 v129, 0x200, v129
	v_add_u32_e32 v118, 0x200, v118
	v_add_u32_e32 v122, 0xffffff80, v122
	s_cmp_lt_u32 s0, 4
	v_mov_b32_e32 v128, v130
	s_cbranch_scc1 .LBB0_3
	s_andn2_b64 vcc, exec, s[2:3]
	s_cbranch_vccnz .LBB0_6
	s_barrier
.LBB0_6:
	v_add_u32_e32 v103, 0x10000, v113
	s_waitcnt vmcnt(5)
	s_barrier
	ds_read_b128 v[150:153], v103
	ds_read_b128 v[154:157], v103 offset:2048
	v_add_u32_e32 v104, 0x10000, v114
	ds_read_b128 v[158:161], v104
	ds_read_b128 v[162:165], v104 offset:2048
	ds_read_b128 v[166:169], v103 offset:4096
	ds_read_b128 v[170:173], v103 offset:6144
	ds_read_b128 v[174:177], v104 offset:4096
	ds_read_b128 v[178:181], v104 offset:6144
	ds_read_b128 v[182:185], v103 offset:8192
	ds_read_b128 v[118:121], v103 offset:10240
	ds_read_b128 v[186:189], v104 offset:8192
	ds_read_b128 v[114:117], v104 offset:10240
	ds_read_b128 v[142:145], v111
	ds_read_b128 v[190:193], v111 offset:2048
	ds_read_b128 v[146:149], v112
	ds_read_b128 v[194:197], v112 offset:2048
	ds_read_b128 v[198:201], v111 offset:4096
	ds_read_b128 v[126:129], v111 offset:6144
	ds_read_b128 v[202:205], v112 offset:4096
	ds_read_b128 v[122:125], v112 offset:6144
	s_waitcnt vmcnt(2) lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[142:145], v[150:153], v[44:47], v110, v108 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_movk_i32 s0, 0x7fff
	v_mov_b32_e32 v103, 0x7fc0
	s_mul_hi_u32 s1, s8, s28
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[146:149], v[158:161], v[44:47], v110, v108 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_mov_b32 s3, 0x27000
	s_waitcnt vmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[142:145], v[166:169], v[88:91], v110, v107 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], v[146:149], v[174:177], v[88:91], v110, v107 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_nop 3
	v_bfe_u32 v104, v47, 16, 1
	v_bfe_u32 v105, v46, 16, 1
	v_add3_u32 v104, v47, v104, s0
	v_bfe_u32 v106, v45, 16, 1
	v_add3_u32 v105, v46, v105, s0
	v_lshrrev_b32_e32 v104, 16, v104
	v_cmp_o_f32_e32 vcc, v47, v47
	v_add3_u32 v106, v45, v106, s0
	v_lshrrev_b32_e32 v105, 16, v105
	v_cndmask_b32_e32 v88, v103, v104, vcc
	v_cmp_o_f32_e32 vcc, v46, v46
	v_bfe_u32 v109, v44, 16, 1
	v_lshrrev_b32_e32 v89, 16, v106
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], v[142:145], v[170:173], v[84:87], v110, v107 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v109, v44, v109, s0
	v_lshrrev_b32_e32 v90, 16, v109
	s_nop 0
	v_cndmask_b32_e32 v84, v103, v105, vcc
	v_cmp_o_f32_e32 vcc, v45, v45
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[142:145], v[154:157], v[92:95], v110, v108 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e32 v85, v103, v89, vcc
	v_cmp_o_f32_e32 vcc, v44, v44
	s_waitcnt vmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[142:145], v[182:185], v[80:83], v110, v102 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v86, v103, v90, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], v[146:149], v[186:189], v[44:47], v110, v102 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[142:145], v[118:121], v[76:79], v110, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[146:149], v[162:165], v[92:95], v110, v108 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[146:149], v[114:117], v[44:47], v110, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[190:193], v[150:153], v[72:75], v110, v108 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 5
	v_bfe_u32 v87, v95, 16, 1
	v_bfe_u32 v80, v94, 16, 1
	v_add3_u32 v76, v95, v87, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], v[146:149], v[178:181], v[134:137], v110, v107 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_bfe_u32 v81, v93, 16, 1
	v_add3_u32 v80, v94, v80, s0
	v_lshrrev_b32_e32 v76, 16, v76
	v_cmp_o_f32_e32 vcc, v95, v95
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[146:149], v[194:197], v[158:161], v[44:47], v110, v108 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_bfe_u32 v82, v92, 16, 1
	v_add3_u32 v81, v93, v81, s0
	v_lshrrev_b32_e32 v77, 16, v80
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[190:193], v[154:157], v[68:71], v110, v108 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v80, v103, v76, vcc
	v_cmp_o_f32_e32 vcc, v94, v94
	v_add3_u32 v82, v92, v82, s0
	v_lshrrev_b32_e32 v78, 16, v81
	v_cndmask_b32_e32 v81, v103, v77, vcc
	v_cmp_o_f32_e32 vcc, v93, v93
	v_lshrrev_b32_e32 v79, 16, v82
	v_bfe_u32 v72, v133, 16, 1
	v_cndmask_b32_e32 v82, v103, v78, vcc
	v_cmp_o_f32_e32 vcc, v92, v92
	v_bfe_u32 v68, v132, 16, 1
	v_bfe_u32 v69, v131, 16, 1
	v_cndmask_b32_e32 v83, v103, v79, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[194:197], v[162:165], v[44:47], v110, v108 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_bfe_u32 v70, v130, 16, 1
	v_add3_u32 v70, v130, v70, s0
	v_add3_u32 v69, v131, v69, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[190:193], v[166:169], v[64:67], v110, v107 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v68, v132, v68, s0
	v_cmp_o_f32_e32 vcc, v133, v133
	s_nop 0
	v_add3_u32 v64, v133, v72, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[194:197], v[174:177], v[44:47], v110, v107 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v64, 16, v64
	v_lshrrev_b32_e32 v65, 16, v68
	v_lshrrev_b32_e32 v66, 16, v69
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[190:193], v[170:173], v[60:63], v110, v107 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v67, 16, v70
	v_cndmask_b32_e32 v87, v103, v64, vcc
	v_cmp_o_f32_e32 vcc, v132, v132
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[194:197], v[178:181], v[44:47], v110, v107 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_bfe_u32 v60, v137, 16, 1
	v_cndmask_b32_e32 v89, v103, v65, vcc
	v_cmp_o_f32_e32 vcc, v131, v131
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[190:193], v[182:185], v[56:59], v110, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e32 v90, v103, v66, vcc
	v_cmp_o_f32_e32 vcc, v130, v130
	v_bfe_u32 v56, v136, 16, 1
	v_bfe_u32 v57, v135, 16, 1
	v_cndmask_b32_e32 v91, v103, v67, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[194:197], v[186:189], v[44:47], v110, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_add3_u32 v56, v136, v56, s0
	v_cmp_o_f32_e32 vcc, v137, v137
	v_bfe_u32 v58, v134, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[190:193], v[118:121], v[52:55], v110, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v57, v135, v57, s0
	v_add3_u32 v58, v134, v58, s0
	s_nop 0
	v_add3_u32 v52, v137, v60, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[194:197], v[114:117], v[44:47], v110, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v52, 16, v52
	v_lshrrev_b32_e32 v53, 16, v56
	v_cndmask_b32_e32 v92, v103, v52, vcc
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[198:201], v[150:153], v[48:51], v101, v108 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v136, v136
	v_lshrrev_b32_e32 v54, 16, v57
	v_lshrrev_b32_e32 v55, 16, v58
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[198:201], v[154:157], v[40:43], v101, v108 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v93, v103, v53, vcc
	v_cmp_o_f32_e32 vcc, v135, v135
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[198:201], v[170:173], v[32:35], v101, v107 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e32 v94, v103, v54, vcc
	v_cmp_o_f32_e32 vcc, v134, v134
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[202:205], v[158:161], v[44:47], v101, v108 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v95, v103, v55, vcc
	v_cmp_o_f32_e32 vcc, v141, v141
	s_nop 0
	v_bfe_u32 v44, v141, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[198:201], v[166:169], v[36:39], v101, v107 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v45, v140, 16, 1
	v_bfe_u32 v46, v139, 16, 1
	v_bfe_u32 v47, v138, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[198:201], v[182:185], v[20:23], v101, v102 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[126:129], v[150:153], v[4:7], v101, v108 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[202:205], v[162:165], v[40:43], v101, v108 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_nop 2
	v_add3_u32 v43, v141, v44, s0
	v_add3_u32 v42, v140, v45, s0
	v_lshrrev_b32_e32 v43, 16, v43
	v_add3_u32 v40, v138, v47, s0
	v_add3_u32 v41, v139, v46, s0
	v_lshrrev_b32_e32 v42, 16, v42
	v_cndmask_b32_e32 v104, v103, v43, vcc
	v_cmp_o_f32_e32 vcc, v140, v140
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[202:205], v[178:181], v[32:35], v101, v107 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e32 v105, v103, v42, vcc
	v_cmp_o_f32_e32 vcc, v139, v139
	v_bfe_u32 v32, v145, 16, 1
	v_bfe_u32 v33, v144, 16, 1
	v_bfe_u32 v34, v143, 16, 1
	v_bfe_u32 v35, v142, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[202:205], v[174:177], v[36:39], v101, v107 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_add3_u32 v110, v142, v35, s0
	v_add3_u32 v111, v143, v34, s0
	s_nop 0
	v_lshrrev_b32_e32 v36, 16, v41
	v_lshrrev_b32_e32 v37, 16, v40
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[202:205], v[186:189], v[20:23], v101, v102 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v106, v103, v36, vcc
	v_cmp_o_f32_e32 vcc, v138, v138
	s_nop 0
	v_add3_u32 v20, v144, v33, s0
	v_add3_u32 v21, v145, v32, s0
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[122:125], v[158:161], v[4:7], v101, v108 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v112, 16, v21
	v_lshrrev_b32_e32 v113, 16, v20
	v_cndmask_b32_e32 v109, v103, v37, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[126:129], v[154:157], v[12:15], v101, v108 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v145, v145
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[122:125], v[162:165], v[4:7], v101, v108 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[126:129], v[166:169], v[16:19], v101, v107 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 2
	v_cndmask_b32_e32 v18, v103, v112, vcc
	v_cmp_o_f32_e32 vcc, v144, v144
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[198:201], v[118:121], v[8:11], v101, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v16, 16, v111
	v_cndmask_b32_e32 v19, v103, v113, vcc
	v_cmp_o_f32_e32 vcc, v143, v143
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[122:125], v[174:177], v[4:7], v101, v107 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v17, 16, v110
	v_cndmask_b32_e32 v16, v103, v16, vcc
	v_cmp_o_f32_e32 vcc, v142, v142
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[126:129], v[170:173], v[24:27], v101, v107 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e32 v17, v103, v17, vcc
	v_cmp_o_f32_e32 vcc, v149, v149
	v_bfe_u32 v24, v149, 16, 1
	v_bfe_u32 v25, v148, 16, 1
	v_add3_u32 v24, v149, v24, s0
	v_bfe_u32 v26, v147, 16, 1
	v_add3_u32 v25, v148, v25, s0
	v_lshrrev_b32_e32 v24, 16, v24
	v_bfe_u32 v27, v146, 16, 1
	v_add3_u32 v26, v147, v26, s0
	v_lshrrev_b32_e32 v25, 16, v25
	v_cndmask_b32_e32 v24, v103, v24, vcc
	v_cmp_o_f32_e32 vcc, v148, v148
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[202:205], v[114:117], v[8:11], v101, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_add3_u32 v27, v146, v27, s0
	v_lshrrev_b32_e32 v26, 16, v26
	v_cndmask_b32_e32 v25, v103, v25, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[122:125], v[178:181], v[4:7], v101, v107 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v147, v147
	v_lshrrev_b32_e32 v27, 16, v27
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[126:129], v[182:185], v[28:31], v101, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v26, v103, v26, vcc
	v_cmp_o_f32_e32 vcc, v146, v146
	s_nop 0
	v_bfe_u32 v28, v79, 16, 1
	v_bfe_u32 v29, v78, 16, 1
	v_add3_u32 v28, v79, v28, s0
	v_cndmask_b32_e32 v27, v103, v27, vcc
	v_bfe_u32 v30, v77, 16, 1
	v_add3_u32 v29, v78, v29, s0
	v_lshrrev_b32_e32 v28, 16, v28
	v_cmp_o_f32_e32 vcc, v79, v79
	v_add3_u32 v30, v77, v30, s0
	v_lshrrev_b32_e32 v29, 16, v29
	v_cndmask_b32_e32 v28, v103, v28, vcc
	v_cmp_o_f32_e32 vcc, v78, v78
	v_bfe_u32 v31, v76, 16, 1
	v_lshrrev_b32_e32 v30, 16, v30
	v_cndmask_b32_e32 v29, v103, v29, vcc
	v_cmp_o_f32_e32 vcc, v77, v77
	v_add3_u32 v31, v76, v31, s0
	v_lshrrev_b32_e32 v31, 16, v31
	v_cndmask_b32_e32 v30, v103, v30, vcc
	v_cmp_o_f32_e32 vcc, v76, v76
	v_bfe_u32 v76, v75, 16, 1
	v_bfe_u32 v77, v74, 16, 1
	v_add3_u32 v76, v75, v76, s0
	v_cndmask_b32_e32 v31, v103, v31, vcc
	v_bfe_u32 v78, v73, 16, 1
	v_add3_u32 v77, v74, v77, s0
	v_lshrrev_b32_e32 v76, 16, v76
	v_cmp_o_f32_e32 vcc, v75, v75
	v_bfe_u32 v79, v72, 16, 1
	v_add3_u32 v78, v73, v78, s0
	v_lshrrev_b32_e32 v77, 16, v77
	v_cndmask_b32_e32 v75, v103, v76, vcc
	v_cmp_o_f32_e32 vcc, v74, v74
	v_add3_u32 v79, v72, v79, s0
	v_lshrrev_b32_e32 v78, 16, v78
	v_cndmask_b32_e32 v74, v103, v77, vcc
	v_cmp_o_f32_e32 vcc, v73, v73
	v_bfe_u32 v76, v71, 16, 1
	v_lshrrev_b32_e32 v79, 16, v79
	v_cndmask_b32_e32 v73, v103, v78, vcc
	v_cmp_o_f32_e32 vcc, v72, v72
	v_bfe_u32 v77, v70, 16, 1
	v_add3_u32 v76, v71, v76, s0
	v_cndmask_b32_e32 v72, v103, v79, vcc
	v_bfe_u32 v78, v69, 16, 1
	v_add3_u32 v77, v70, v77, s0
	v_lshrrev_b32_e32 v76, 16, v76
	v_cmp_o_f32_e32 vcc, v71, v71
	v_bfe_u32 v79, v68, 16, 1
	v_add3_u32 v78, v69, v78, s0
	v_lshrrev_b32_e32 v77, 16, v77
	v_cndmask_b32_e32 v71, v103, v76, vcc
	v_cmp_o_f32_e32 vcc, v70, v70
	v_add3_u32 v79, v68, v79, s0
	v_lshrrev_b32_e32 v78, 16, v78
	v_cndmask_b32_e32 v70, v103, v77, vcc
	v_cmp_o_f32_e32 vcc, v69, v69
	v_bfe_u32 v76, v67, 16, 1
	v_lshrrev_b32_e32 v79, 16, v79
	v_cndmask_b32_e32 v69, v103, v78, vcc
	v_cmp_o_f32_e32 vcc, v68, v68
	v_bfe_u32 v77, v66, 16, 1
	v_add3_u32 v76, v67, v76, s0
	v_cndmask_b32_e32 v68, v103, v79, vcc
	v_bfe_u32 v78, v65, 16, 1
	v_add3_u32 v77, v66, v77, s0
	v_lshrrev_b32_e32 v76, 16, v76
	v_cmp_o_f32_e32 vcc, v67, v67
	v_bfe_u32 v79, v64, 16, 1
	v_add3_u32 v78, v65, v78, s0
	v_lshrrev_b32_e32 v77, 16, v77
	v_cndmask_b32_e32 v67, v103, v76, vcc
	v_cmp_o_f32_e32 vcc, v66, v66
	v_add3_u32 v79, v64, v79, s0
	v_lshrrev_b32_e32 v78, 16, v78
	v_cndmask_b32_e32 v66, v103, v77, vcc
	v_cmp_o_f32_e32 vcc, v65, v65
	v_bfe_u32 v76, v63, 16, 1
	v_lshrrev_b32_e32 v79, 16, v79
	v_cndmask_b32_e32 v65, v103, v78, vcc
	v_cmp_o_f32_e32 vcc, v64, v64
	v_bfe_u32 v77, v62, 16, 1
	v_add3_u32 v76, v63, v76, s0
	v_cndmask_b32_e32 v64, v103, v79, vcc
	v_bfe_u32 v78, v61, 16, 1
	v_add3_u32 v77, v62, v77, s0
	v_lshrrev_b32_e32 v76, 16, v76
	v_cmp_o_f32_e32 vcc, v63, v63
	v_bfe_u32 v79, v60, 16, 1
	v_add3_u32 v78, v61, v78, s0
	v_lshrrev_b32_e32 v77, 16, v77
	v_cndmask_b32_e32 v63, v103, v76, vcc
	v_cmp_o_f32_e32 vcc, v62, v62
	v_add3_u32 v79, v60, v79, s0
	v_lshrrev_b32_e32 v78, 16, v78
	v_cndmask_b32_e32 v62, v103, v77, vcc
	v_cmp_o_f32_e32 vcc, v61, v61
	v_bfe_u32 v76, v59, 16, 1
	v_lshrrev_b32_e32 v79, 16, v79
	v_cndmask_b32_e32 v61, v103, v78, vcc
	v_cmp_o_f32_e32 vcc, v60, v60
	v_bfe_u32 v77, v58, 16, 1
	v_add3_u32 v76, v59, v76, s0
	v_cndmask_b32_e32 v60, v103, v79, vcc
	v_bfe_u32 v78, v57, 16, 1
	v_add3_u32 v77, v58, v77, s0
	v_lshrrev_b32_e32 v76, 16, v76
	v_cmp_o_f32_e32 vcc, v59, v59
	v_bfe_u32 v79, v56, 16, 1
	v_add3_u32 v78, v57, v78, s0
	v_lshrrev_b32_e32 v77, 16, v77
	v_cndmask_b32_e32 v59, v103, v76, vcc
	v_cmp_o_f32_e32 vcc, v58, v58
	v_add3_u32 v79, v56, v79, s0
	v_lshrrev_b32_e32 v78, 16, v78
	v_cndmask_b32_e32 v58, v103, v77, vcc
	v_cmp_o_f32_e32 vcc, v57, v57
	v_bfe_u32 v76, v55, 16, 1
	v_lshrrev_b32_e32 v79, 16, v79
	v_cndmask_b32_e32 v57, v103, v78, vcc
	v_cmp_o_f32_e32 vcc, v56, v56
	v_bfe_u32 v77, v54, 16, 1
	v_add3_u32 v76, v55, v76, s0
	v_cndmask_b32_e32 v56, v103, v79, vcc
	v_bfe_u32 v78, v53, 16, 1
	v_add3_u32 v77, v54, v77, s0
	v_lshrrev_b32_e32 v76, 16, v76
	v_cmp_o_f32_e32 vcc, v55, v55
	v_bfe_u32 v79, v52, 16, 1
	v_add3_u32 v78, v53, v78, s0
	v_lshrrev_b32_e32 v77, 16, v77
	v_cndmask_b32_e32 v55, v103, v76, vcc
	v_cmp_o_f32_e32 vcc, v54, v54
	v_add3_u32 v79, v52, v79, s0
	v_lshrrev_b32_e32 v78, 16, v78
	v_cndmask_b32_e32 v54, v103, v77, vcc
	v_cmp_o_f32_e32 vcc, v53, v53
	v_bfe_u32 v76, v51, 16, 1
	v_lshrrev_b32_e32 v79, 16, v79
	v_cndmask_b32_e32 v53, v103, v78, vcc
	v_cmp_o_f32_e32 vcc, v52, v52
	v_bfe_u32 v77, v50, 16, 1
	v_add3_u32 v76, v51, v76, s0
	v_cndmask_b32_e32 v52, v103, v79, vcc
	v_bfe_u32 v78, v49, 16, 1
	v_add3_u32 v77, v50, v77, s0
	v_lshrrev_b32_e32 v76, 16, v76
	v_cmp_o_f32_e32 vcc, v51, v51
	v_bfe_u32 v79, v48, 16, 1
	v_add3_u32 v78, v49, v78, s0
	v_lshrrev_b32_e32 v77, 16, v77
	v_cndmask_b32_e32 v51, v103, v76, vcc
	v_cmp_o_f32_e32 vcc, v50, v50
	v_add3_u32 v79, v48, v79, s0
	v_lshrrev_b32_e32 v78, 16, v78
	v_cndmask_b32_e32 v50, v103, v77, vcc
	v_cmp_o_f32_e32 vcc, v49, v49
	v_bfe_u32 v76, v47, 16, 1
	v_lshrrev_b32_e32 v79, 16, v79
	v_cndmask_b32_e32 v49, v103, v78, vcc
	v_cmp_o_f32_e32 vcc, v48, v48
	v_bfe_u32 v77, v46, 16, 1
	v_add3_u32 v76, v47, v76, s0
	v_cndmask_b32_e32 v48, v103, v79, vcc
	v_bfe_u32 v78, v45, 16, 1
	v_add3_u32 v77, v46, v77, s0
	v_lshrrev_b32_e32 v76, 16, v76
	v_cmp_o_f32_e32 vcc, v47, v47
	v_bfe_u32 v79, v44, 16, 1
	v_add3_u32 v78, v45, v78, s0
	v_lshrrev_b32_e32 v77, 16, v77
	v_cndmask_b32_e32 v47, v103, v76, vcc
	v_cmp_o_f32_e32 vcc, v46, v46
	v_add3_u32 v79, v44, v79, s0
	v_lshrrev_b32_e32 v78, 16, v78
	v_cndmask_b32_e32 v46, v103, v77, vcc
	v_cmp_o_f32_e32 vcc, v45, v45
	v_bfe_u32 v76, v43, 16, 1
	v_lshrrev_b32_e32 v79, 16, v79
	v_cndmask_b32_e32 v45, v103, v78, vcc
	v_cmp_o_f32_e32 vcc, v44, v44
	v_bfe_u32 v77, v42, 16, 1
	v_add3_u32 v76, v43, v76, s0
	v_cndmask_b32_e32 v44, v103, v79, vcc
	v_bfe_u32 v78, v41, 16, 1
	v_add3_u32 v77, v42, v77, s0
	v_lshrrev_b32_e32 v76, 16, v76
	v_cmp_o_f32_e32 vcc, v43, v43
	v_bfe_u32 v79, v40, 16, 1
	v_add3_u32 v78, v41, v78, s0
	v_lshrrev_b32_e32 v77, 16, v77
	v_cndmask_b32_e32 v43, v103, v76, vcc
	v_cmp_o_f32_e32 vcc, v42, v42
	v_add3_u32 v79, v40, v79, s0
	v_lshrrev_b32_e32 v78, 16, v78
	v_cndmask_b32_e32 v42, v103, v77, vcc
	v_cmp_o_f32_e32 vcc, v41, v41
	v_bfe_u32 v76, v39, 16, 1
	v_lshrrev_b32_e32 v79, 16, v79
	v_cndmask_b32_e32 v41, v103, v78, vcc
	v_cmp_o_f32_e32 vcc, v40, v40
	v_bfe_u32 v77, v38, 16, 1
	v_add3_u32 v76, v39, v76, s0
	v_cndmask_b32_e32 v40, v103, v79, vcc
	v_bfe_u32 v78, v37, 16, 1
	v_add3_u32 v77, v38, v77, s0
	v_lshrrev_b32_e32 v76, 16, v76
	v_cmp_o_f32_e32 vcc, v39, v39
	v_bfe_u32 v79, v36, 16, 1
	v_add3_u32 v78, v37, v78, s0
	v_lshrrev_b32_e32 v77, 16, v77
	v_cndmask_b32_e32 v39, v103, v76, vcc
	v_cmp_o_f32_e32 vcc, v38, v38
	v_add3_u32 v79, v36, v79, s0
	v_lshrrev_b32_e32 v78, 16, v78
	v_cndmask_b32_e32 v38, v103, v77, vcc
	v_cmp_o_f32_e32 vcc, v37, v37
	v_bfe_u32 v76, v35, 16, 1
	v_lshrrev_b32_e32 v79, 16, v79
	v_cndmask_b32_e32 v37, v103, v78, vcc
	v_cmp_o_f32_e32 vcc, v36, v36
	v_bfe_u32 v77, v34, 16, 1
	v_add3_u32 v76, v35, v76, s0
	v_cndmask_b32_e32 v36, v103, v79, vcc
	v_bfe_u32 v78, v33, 16, 1
	v_add3_u32 v77, v34, v77, s0
	v_lshrrev_b32_e32 v76, 16, v76
	v_cmp_o_f32_e32 vcc, v35, v35
	v_bfe_u32 v79, v32, 16, 1
	v_add3_u32 v78, v33, v78, s0
	v_lshrrev_b32_e32 v77, 16, v77
	v_cndmask_b32_e32 v35, v103, v76, vcc
	v_cmp_o_f32_e32 vcc, v34, v34
	v_add3_u32 v79, v32, v79, s0
	v_lshrrev_b32_e32 v78, 16, v78
	v_cndmask_b32_e32 v34, v103, v77, vcc
	v_cmp_o_f32_e32 vcc, v33, v33
	v_bfe_u32 v76, v23, 16, 1
	v_lshrrev_b32_e32 v79, 16, v79
	v_cndmask_b32_e32 v33, v103, v78, vcc
	v_cmp_o_f32_e32 vcc, v32, v32
	v_bfe_u32 v77, v22, 16, 1
	v_add3_u32 v76, v23, v76, s0
	v_cndmask_b32_e32 v32, v103, v79, vcc
	v_bfe_u32 v78, v21, 16, 1
	v_add3_u32 v77, v22, v77, s0
	v_lshrrev_b32_e32 v76, 16, v76
	v_cmp_o_f32_e32 vcc, v23, v23
	v_bfe_u32 v79, v20, 16, 1
	v_add3_u32 v78, v21, v78, s0
	v_lshrrev_b32_e32 v77, 16, v77
	v_cndmask_b32_e32 v23, v103, v76, vcc
	v_cmp_o_f32_e32 vcc, v22, v22
	v_add3_u32 v79, v20, v79, s0
	v_lshrrev_b32_e32 v78, 16, v78
	v_cndmask_b32_e32 v22, v103, v77, vcc
	v_cmp_o_f32_e32 vcc, v21, v21
	v_bfe_u32 v76, v15, 16, 1
	v_lshrrev_b32_e32 v79, 16, v79
	v_cndmask_b32_e32 v21, v103, v78, vcc
	v_cmp_o_f32_e32 vcc, v20, v20
	v_bfe_u32 v77, v14, 16, 1
	v_add3_u32 v76, v15, v76, s0
	v_cndmask_b32_e32 v20, v103, v79, vcc
	v_bfe_u32 v78, v13, 16, 1
	v_add3_u32 v77, v14, v77, s0
	v_lshrrev_b32_e32 v76, 16, v76
	v_cmp_o_f32_e32 vcc, v15, v15
	v_bfe_u32 v79, v12, 16, 1
	v_add3_u32 v78, v13, v78, s0
	v_lshrrev_b32_e32 v77, 16, v77
	v_cndmask_b32_e32 v15, v103, v76, vcc
	v_cmp_o_f32_e32 vcc, v14, v14
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[122:125], v[186:189], v[4:7], v101, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_add3_u32 v79, v12, v79, s0
	v_lshrrev_b32_e32 v78, 16, v78
	v_cndmask_b32_e32 v14, v103, v77, vcc
	v_cmp_o_f32_e32 vcc, v13, v13
	v_bfe_u32 v76, v11, 16, 1
	v_lshrrev_b32_e32 v79, 16, v79
	v_cndmask_b32_e32 v13, v103, v78, vcc
	v_cmp_o_f32_e32 vcc, v12, v12
	v_bfe_u32 v77, v10, 16, 1
	v_add3_u32 v76, v11, v76, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[126:129], v[118:121], v[0:3], v101, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v12, v103, v79, vcc
	v_bfe_u32 v78, v9, 16, 1
	v_add3_u32 v77, v10, v77, s0
	v_lshrrev_b32_e32 v76, 16, v76
	v_cmp_o_f32_e32 vcc, v11, v11
	v_bfe_u32 v79, v8, 16, 1
	v_add3_u32 v78, v9, v78, s0
	v_lshrrev_b32_e32 v77, 16, v77
	v_cndmask_b32_e32 v11, v103, v76, vcc
	v_cmp_o_f32_e32 vcc, v10, v10
	v_add3_u32 v79, v8, v79, s0
	v_lshrrev_b32_e32 v78, 16, v78
	v_cndmask_b32_e32 v10, v103, v77, vcc
	v_cmp_o_f32_e32 vcc, v9, v9
	v_bfe_u32 v76, v7, 16, 1
	v_lshrrev_b32_e32 v79, 16, v79
	v_cndmask_b32_e32 v9, v103, v78, vcc
	v_cmp_o_f32_e32 vcc, v8, v8
	v_bfe_u32 v77, v6, 16, 1
	v_add3_u32 v76, v7, v76, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[122:125], v[114:117], v[0:3], v101, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v8, v103, v79, vcc
	v_bfe_u32 v78, v5, 16, 1
	v_add3_u32 v77, v6, v77, s0
	v_lshrrev_b32_e32 v76, 16, v76
	v_cmp_o_f32_e32 vcc, v7, v7
	v_bfe_u32 v79, v4, 16, 1
	v_add3_u32 v78, v5, v78, s0
	v_lshrrev_b32_e32 v77, 16, v77
	v_cndmask_b32_e32 v7, v103, v76, vcc
	v_cmp_o_f32_e32 vcc, v6, v6
	v_add3_u32 v79, v4, v79, s0
	v_lshrrev_b32_e32 v78, 16, v78
	v_cndmask_b32_e32 v6, v103, v77, vcc
	v_cmp_o_f32_e32 vcc, v5, v5
	v_lshrrev_b32_e32 v79, 16, v79
	v_bfe_u32 v76, v3, 16, 1
	v_cndmask_b32_e32 v5, v103, v78, vcc
	v_cmp_o_f32_e32 vcc, v4, v4
	v_bfe_u32 v77, v2, 16, 1
	v_bfe_u32 v78, v1, 16, 1
	v_cndmask_b32_e32 v4, v103, v79, vcc
	v_bfe_u32 v79, v0, 16, 1
	v_add3_u32 v79, v0, v79, s0
	v_add3_u32 v78, v1, v78, s0
	v_add3_u32 v77, v2, v77, s0
	v_add3_u32 v76, v3, v76, s0
	s_mul_i32 s0, s9, s28
	s_add_i32 s1, s1, s0
	s_mul_i32 s0, s8, s28
	s_lshl_b64 s[0:1], s[0:1], 1
	v_lshrrev_b32_e32 v76, 16, v76
	v_cmp_o_f32_e32 vcc, v3, v3
	s_add_u32 s0, s10, s0
	v_lshrrev_b32_e32 v77, 16, v77
	v_cndmask_b32_e32 v3, v103, v76, vcc
	v_cmp_o_f32_e32 vcc, v2, v2
	s_addc_u32 s1, s11, s1
	s_lshl_b32 s2, s29, 1
	v_cndmask_b32_e32 v2, v103, v77, vcc
	s_add_u32 s0, s0, s2
	v_lshlrev_b32_e32 v77, 1, v96
	v_lshl_or_b32 v76, v98, 2, v99
	s_addc_u32 s1, s1, 0
	s_and_b32 s2, s8, 0x3fff
	v_lshl_add_u32 v77, v97, 1, v77
	v_lshrrev_b32_e32 v78, 16, v78
	v_cmp_o_f32_e32 vcc, v1, v1
	v_mul_lo_u32 v76, s8, v76
	s_lshl_b32 s2, s2, 16
	s_and_b32 s1, s1, 0xffff
	v_lshl_add_u32 v77, v100, 1, v77
	v_lshrrev_b32_e32 v79, 16, v79
	v_cndmask_b32_e32 v1, v103, v78, vcc
	v_cmp_o_f32_e32 vcc, v0, v0
	s_or_b32 s1, s2, s1
	v_lshl_add_u32 v78, v76, 1, v77
	s_lshl_b32 s4, s8, 1
	v_cndmask_b32_e32 v0, v103, v79, vcc
	s_or_b32 s1, s1, 2.0
	s_mov_b32 s2, 0x7ffffffd
	v_add_u32_e32 v79, s4, v78
	buffer_store_short v86, v78, s[0:3], 0 offen
	buffer_store_short v85, v79, s[0:3], 0 offen
	v_add_u32_e32 v85, s4, v79
	buffer_store_short v84, v85, s[0:3], 0 offen
	v_add_u32_e32 v84, s4, v85
	s_lshl_b32 s5, s8, 4
	buffer_store_short v88, v84, s[0:3], 0 offen
	buffer_store_short v83, v78, s[0:3], 0 offen offset:32
	buffer_store_short v82, v79, s[0:3], 0 offen offset:32
	buffer_store_short v81, v85, s[0:3], 0 offen offset:32
	buffer_store_short v80, v84, s[0:3], 0 offen offset:32
	buffer_store_short v91, v78, s[0:3], 0 offen offset:64
	buffer_store_short v90, v79, s[0:3], 0 offen offset:64
	buffer_store_short v89, v85, s[0:3], 0 offen offset:64
	buffer_store_short v87, v84, s[0:3], 0 offen offset:64
	buffer_store_short v95, v78, s[0:3], 0 offen offset:96
	buffer_store_short v94, v79, s[0:3], 0 offen offset:96
	buffer_store_short v93, v85, s[0:3], 0 offen offset:96
	buffer_store_short v92, v84, s[0:3], 0 offen offset:96
	buffer_store_short v109, v78, s[0:3], 0 offen offset:128
	buffer_store_short v106, v79, s[0:3], 0 offen offset:128
	buffer_store_short v105, v85, s[0:3], 0 offen offset:128
	buffer_store_short v104, v84, s[0:3], 0 offen offset:128
	buffer_store_short v17, v78, s[0:3], 0 offen offset:160
	buffer_store_short v16, v79, s[0:3], 0 offen offset:160
	buffer_store_short v19, v85, s[0:3], 0 offen offset:160
	buffer_store_short v18, v84, s[0:3], 0 offen offset:160
	v_add_u32_e32 v16, s5, v76
	v_lshl_add_u32 v17, v16, 1, v77
	v_add_u32_e32 v18, s4, v17
	v_add_u32_e32 v19, s4, v18
	buffer_store_short v27, v17, s[0:3], 0 offen
	buffer_store_short v26, v18, s[0:3], 0 offen
	buffer_store_short v25, v19, s[0:3], 0 offen
	v_add_u32_e32 v25, s4, v19
	v_add_u32_e32 v16, s5, v16
	buffer_store_short v24, v25, s[0:3], 0 offen
	buffer_store_short v31, v17, s[0:3], 0 offen offset:32
	buffer_store_short v30, v18, s[0:3], 0 offen offset:32
	buffer_store_short v29, v19, s[0:3], 0 offen offset:32
	buffer_store_short v28, v25, s[0:3], 0 offen offset:32
	buffer_store_short v72, v17, s[0:3], 0 offen offset:64
	buffer_store_short v73, v18, s[0:3], 0 offen offset:64
	buffer_store_short v74, v19, s[0:3], 0 offen offset:64
	buffer_store_short v75, v25, s[0:3], 0 offen offset:64
	buffer_store_short v68, v17, s[0:3], 0 offen offset:96
	buffer_store_short v69, v18, s[0:3], 0 offen offset:96
	buffer_store_short v70, v19, s[0:3], 0 offen offset:96
	buffer_store_short v71, v25, s[0:3], 0 offen offset:96
	buffer_store_short v64, v17, s[0:3], 0 offen offset:128
	buffer_store_short v65, v18, s[0:3], 0 offen offset:128
	buffer_store_short v66, v19, s[0:3], 0 offen offset:128
	buffer_store_short v67, v25, s[0:3], 0 offen offset:128
	buffer_store_short v60, v17, s[0:3], 0 offen offset:160
	buffer_store_short v61, v18, s[0:3], 0 offen offset:160
	buffer_store_short v62, v19, s[0:3], 0 offen offset:160
	buffer_store_short v63, v25, s[0:3], 0 offen offset:160
	v_lshl_add_u32 v17, v16, 1, v77
	v_add_u32_e32 v18, s4, v17
	v_add_u32_e32 v19, s4, v18
	v_add_u32_e32 v16, s5, v16
	v_add_u32_e32 v24, s4, v19
	v_lshl_add_u32 v16, v16, 1, v77
	buffer_store_short v56, v17, s[0:3], 0 offen
	buffer_store_short v57, v18, s[0:3], 0 offen
	buffer_store_short v58, v19, s[0:3], 0 offen
	buffer_store_short v59, v24, s[0:3], 0 offen
	buffer_store_short v52, v17, s[0:3], 0 offen offset:32
	buffer_store_short v53, v18, s[0:3], 0 offen offset:32
	buffer_store_short v54, v19, s[0:3], 0 offen offset:32
	buffer_store_short v55, v24, s[0:3], 0 offen offset:32
	buffer_store_short v48, v17, s[0:3], 0 offen offset:64
	buffer_store_short v49, v18, s[0:3], 0 offen offset:64
	buffer_store_short v50, v19, s[0:3], 0 offen offset:64
	buffer_store_short v51, v24, s[0:3], 0 offen offset:64
	buffer_store_short v44, v17, s[0:3], 0 offen offset:96
	buffer_store_short v45, v18, s[0:3], 0 offen offset:96
	buffer_store_short v46, v19, s[0:3], 0 offen offset:96
	buffer_store_short v47, v24, s[0:3], 0 offen offset:96
	buffer_store_short v40, v17, s[0:3], 0 offen offset:128
	buffer_store_short v41, v18, s[0:3], 0 offen offset:128
	buffer_store_short v42, v19, s[0:3], 0 offen offset:128
	buffer_store_short v43, v24, s[0:3], 0 offen offset:128
	buffer_store_short v36, v17, s[0:3], 0 offen offset:160
	buffer_store_short v37, v18, s[0:3], 0 offen offset:160
	buffer_store_short v38, v19, s[0:3], 0 offen offset:160
	buffer_store_short v39, v24, s[0:3], 0 offen offset:160
	v_add_u32_e32 v17, s4, v16
	v_add_u32_e32 v18, s4, v17
	v_add_u32_e32 v19, s4, v18
	buffer_store_short v32, v16, s[0:3], 0 offen
	buffer_store_short v33, v17, s[0:3], 0 offen
	buffer_store_short v34, v18, s[0:3], 0 offen
	buffer_store_short v35, v19, s[0:3], 0 offen
	buffer_store_short v20, v16, s[0:3], 0 offen offset:32
	buffer_store_short v21, v17, s[0:3], 0 offen offset:32
	buffer_store_short v22, v18, s[0:3], 0 offen offset:32
	buffer_store_short v23, v19, s[0:3], 0 offen offset:32
	buffer_store_short v12, v16, s[0:3], 0 offen offset:64
	buffer_store_short v13, v17, s[0:3], 0 offen offset:64
	buffer_store_short v14, v18, s[0:3], 0 offen offset:64
	buffer_store_short v15, v19, s[0:3], 0 offen offset:64
	buffer_store_short v8, v16, s[0:3], 0 offen offset:96
	buffer_store_short v9, v17, s[0:3], 0 offen offset:96
	buffer_store_short v10, v18, s[0:3], 0 offen offset:96
	buffer_store_short v11, v19, s[0:3], 0 offen offset:96
	buffer_store_short v4, v16, s[0:3], 0 offen offset:128
	buffer_store_short v5, v17, s[0:3], 0 offen offset:128
	buffer_store_short v6, v18, s[0:3], 0 offen offset:128
	buffer_store_short v7, v19, s[0:3], 0 offen offset:128
	buffer_store_short v0, v16, s[0:3], 0 offen offset:160
	buffer_store_short v1, v17, s[0:3], 0 offen offset:160
	buffer_store_short v2, v18, s[0:3], 0 offen offset:160
	buffer_store_short v3, v19, s[0:3], 0 offen offset:160
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wave_mxfp4_static_gemm_256x192x256_13056x43392x1792
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
	.size	wave_mxfp4_static_gemm_256x192x256_13056x43392x1792, .Lfunc_end0-wave_mxfp4_static_gemm_256x192x256_13056x43392x1792

	.set wave_mxfp4_static_gemm_256x192x256_13056x43392x1792.num_vgpr, 206
	.set wave_mxfp4_static_gemm_256x192x256_13056x43392x1792.num_agpr, 0
	.set wave_mxfp4_static_gemm_256x192x256_13056x43392x1792.numbered_sgpr, 48
	.set wave_mxfp4_static_gemm_256x192x256_13056x43392x1792.num_named_barrier, 0
	.set wave_mxfp4_static_gemm_256x192x256_13056x43392x1792.private_seg_size, 0
	.set wave_mxfp4_static_gemm_256x192x256_13056x43392x1792.uses_vcc, 1
	.set wave_mxfp4_static_gemm_256x192x256_13056x43392x1792.uses_flat_scratch, 0
	.set wave_mxfp4_static_gemm_256x192x256_13056x43392x1792.has_dyn_sized_stack, 0
	.set wave_mxfp4_static_gemm_256x192x256_13056x43392x1792.has_recursion, 0
	.set wave_mxfp4_static_gemm_256x192x256_13056x43392x1792.has_indirect_call, 0
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
    .name:           wave_mxfp4_static_gemm_256x192x256_13056x43392x1792
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 256
      - 2
      - 1
    .sgpr_count:     54
    .sgpr_spill_count: 0
    .symbol:         wave_mxfp4_static_gemm_256x192x256_13056x43392x1792.kd
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
