; To reproduce the .rocmasm from .optimized.ll, run:
; llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -mattr='-fma-mix-insts' -O3 <.optimized.ll> -o <out.rocmasm>

	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.text
	.globl	wave_mxfp4_static_gemm_256x192x256_15360x9984x13824
	.p2align	8
	.type	wave_mxfp4_static_gemm_256x192x256_15360x9984x13824,@function
wave_mxfp4_static_gemm_256x192x256_15360x9984x13824:
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
	s_mov_b64 s[24:25], s[2:3]
	v_readfirstlane_b32 s2, v2
	v_lshrrev_b32_e32 v2, 3, v96
	v_or_b32_e32 v3, v2, v5
	s_lshl_b32 s28, s16, 8
	v_or_b32_e32 v4, s28, v3
	v_bitop3_b32 v6, v2, 7, v96 bitop3:0x48
	v_lshlrev_b32_e32 v101, 4, v6
	v_mul_u32_u24_e32 v102, 0x1b00, v4
	s_and_b32 s3, s25, 0xffff
	s_lshl_b32 s31, s2, 7
	s_or_b32 s25, s3, 0x5b000000
	s_mov_b32 s27, 0x27000
	s_mov_b32 s26, 0x7ffffffe
	v_or_b32_e32 v4, v102, v101
	s_mov_b32 m0, s31
	s_or_b32 s33, s31, 0x2000
	buffer_load_dwordx4 v4, s[24:27], 0 offen lds
	v_add_u32_e32 v7, 0x6c000, v4
	s_mov_b32 m0, s33
	s_or_b32 s34, s31, 0x4000
	buffer_load_dwordx4 v7, s[24:27], 0 offen lds
	v_add_u32_e32 v7, 0xd8000, v4
	s_mov_b32 m0, s34
	s_or_b32 s35, s31, 0x6000
	buffer_load_dwordx4 v7, s[24:27], 0 offen lds
	s_mul_i32 s29, s17, 0xc0
	v_lshrrev_b32_e32 v7, 7, v96
	v_add_u32_e32 v4, 0x144000, v4
	s_mov_b32 m0, s35
	v_and_or_b32 v103, v3, 48, s29
	v_lshlrev_b32_e32 v2, 4, v2
	v_lshlrev_b32_e32 v3, 8, v7
	s_mov_b64 s[20:21], s[6:7]
	buffer_load_dwordx4 v4, s[24:27], 0 offen lds
	v_sub_u32_e32 v8, v2, v3
	v_lshlrev_b32_e32 v4, 8, v6
	v_lshlrev_b32_e32 v7, 4, v7
	s_movk_i32 s30, 0x1b00
	v_add_u32_e32 v6, v4, v8
	s_and_b32 s2, s21, 0xffff
	v_or3_b32 v104, v7, s29, v5
	s_or_b32 s21, s2, 0x5b000000
	s_mov_b32 s22, s26
	s_mov_b32 s23, s27
	v_mad_u32_u24 v8, v103, s30, v6
	s_add_i32 m0, s31, 0x10000
	v_mad_u32_u24 v5, v104, s30, v6
	buffer_load_dwordx4 v8, s[20:23], 0 offen lds
	v_add_u32_e32 v6, 0x6c000, v5
	s_add_i32 m0, s33, 0x10000
	s_mul_i32 s15, s15, s28
	s_mul_hi_u32 s2, s14, s28
	buffer_load_dwordx4 v6, s[20:23], 0 offen lds
	s_add_i32 m0, s34, 0x10000
	s_add_i32 s2, s2, s15
	s_mul_i32 s3, s14, s28
	s_load_dwordx2 s[12:13], s[0:1], 0x40
	v_add_u32_e32 v5, 0xd8000, v5
	s_add_u32 s4, s4, s3
	buffer_load_dwordx4 v5, s[20:23], 0 offen lds
	v_lshrrev_b32_e32 v5, 4, v96
	v_bfe_u32 v98, v96, 4, 2
	s_addc_u32 s2, s5, s2
	s_and_b32 s3, s14, 0x3fff
	v_sub_u32_e32 v7, v98, v5
	v_lshlrev_b32_e32 v6, 2, v96
	s_bitset1_b32 s3, 14
	v_and_b32_e32 v97, 0xc0, v96
	v_lshl_add_u32 v8, v7, 6, v6
	s_and_b32 s2, s2, 0xffff
	s_lshl_b32 s3, s3, 16
	s_or_b32 s5, s2, s3
	v_mad_u64_u32 v[10:11], s[2:3], s14, v97, v[8:9]
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s2, s13, s29
	s_mul_hi_u32 s3, s12, s29
	s_add_i32 s3, s3, s2
	s_mul_i32 s2, s12, s29
	s_add_u32 s16, s8, s2
	s_addc_u32 s2, s9, s3
	s_and_b32 s3, s12, 0x3fff
	s_bitset1_b32 s3, 14
	v_mul_u32_u24_e32 v99, 0x60, v0
	s_and_b32 s2, s2, 0xffff
	s_lshl_b32 s3, s3, 16
	s_or_b32 s17, s2, s3
	v_mad_u64_u32 v[8:9], s[2:3], s12, v99, v[8:9]
	s_mov_b32 s6, s26
	s_mov_b32 s7, s27
	v_lshl_add_u32 v7, s14, 5, v10
	s_lshl_b32 s2, s12, 5
	buffer_load_dword v117, v10, s[4:7], 0 offen
	buffer_load_dword v116, v7, s[4:7], 0 offen
	s_mov_b32 s18, s26
	s_mov_b32 s19, s27
	v_add_u32_e32 v7, s2, v8
	v_add_u32_e32 v9, s2, v7
	buffer_load_dword v120, v8, s[16:19], 0 offen
	buffer_load_dword v119, v7, s[16:19], 0 offen
	buffer_load_dword v118, v9, s[16:19], 0 offen
	v_cmp_eq_u32_e64 s[2:3], 0, v0
	s_mov_b32 s13, 0
	s_mov_b32 s15, 0x10000
	s_and_b64 vcc, exec, s[2:3]
	s_barrier
	s_waitcnt vmcnt(0)
	s_cbranch_vccnz .LBB0_2
	s_barrier
.LBB0_2:
	s_load_dwordx2 s[8:9], s[0:1], 0x48
	v_and_b32_e32 v7, 7, v96
	v_lshlrev_b32_e32 v9, 7, v96
	v_lshlrev_b32_e32 v10, 11, v5
	s_movk_i32 s0, 0x3000
	v_bitop3_b32 v8, v98, v96, 7 bitop3:0x78
	v_sub_u32_e32 v9, v9, v10
	v_mul_lo_u32 v0, v0, s0
	v_bitop3_b32 v7, v98, v7, 4 bitop3:0x36
	v_lshlrev_b32_e32 v8, 4, v8
	v_add_u32_e32 v0, v9, v0
	v_lshlrev_b32_e32 v7, 4, v7
	v_lshl_add_u32 v1, v1, 13, v9
	v_or_b32_e32 v107, v0, v8
	v_or_b32_e32 v108, v7, v0
	v_lshlrev_b32_e32 v0, 4, v98
	v_mul_i32_i24_e32 v100, -16, v5
	v_or_b32_e32 v105, v1, v8
	v_or_b32_e32 v106, v7, v1
	v_add3_u32 v124, v0, v100, v96
	v_lshl_add_u32 v0, v98, 6, v6
	v_lshlrev_b32_e32 v1, 6, v5
	v_sub_u32_e32 v0, v0, v1
	v_add_u32_e32 v110, 0x100, v0
	v_add_u32_e32 v0, v4, v2
	v_sub_u32_e32 v0, v0, v3
	v_mov_b32_e32 v48, 0
	v_add_u32_e32 v109, 32, v97
	v_add_u32_e32 v111, 32, v99
	v_add_u32_e32 v112, 64, v99
	v_sub_u32_e32 v113, 0xffffffbf, v124
	s_movk_i32 s0, 0x80
	v_or_b32_e32 v114, 0x80, v101
	v_add_u32_e32 v115, 0xd8800, v0
	s_mov_b32 s43, 0x8000
	s_mov_b32 s44, 0x16000
	s_mov_b32 s1, 0
	s_mov_b32 s36, 0x97b425f
	s_movk_i32 s37, 0xe500
	s_mov_b32 s38, 0xfff28000
	s_mov_b32 s22, s26
	s_mov_b32 s23, s27
	s_movk_i32 s39, 0xffc0
	s_mov_b32 s40, 0x4bda12f7
	s_movk_i32 s41, 0x1b0
	s_mov_b32 s6, s26
	s_mov_b32 s7, s27
	s_mov_b32 s18, s26
	s_mov_b32 s19, s27
	v_mov_b32_e32 v49, v48
	v_mov_b32_e32 v50, v48
	v_mov_b32_e32 v51, v48
	v_mov_b32_e32 v92, v48
	v_mov_b32_e32 v93, v48
	v_mov_b32_e32 v94, v48
	v_mov_b32_e32 v95, v48
	v_mov_b32_e32 v88, v48
	v_mov_b32_e32 v89, v48
	v_mov_b32_e32 v90, v48
	v_mov_b32_e32 v91, v48
	v_mov_b32_e32 v84, v48
	v_mov_b32_e32 v85, v48
	v_mov_b32_e32 v86, v48
	v_mov_b32_e32 v87, v48
	v_mov_b32_e32 v80, v48
	v_mov_b32_e32 v81, v48
	v_mov_b32_e32 v82, v48
	v_mov_b32_e32 v83, v48
	v_mov_b32_e32 v76, v48
	v_mov_b32_e32 v77, v48
	v_mov_b32_e32 v78, v48
	v_mov_b32_e32 v79, v48
	v_mov_b32_e32 v72, v48
	v_mov_b32_e32 v73, v48
	v_mov_b32_e32 v74, v48
	v_mov_b32_e32 v75, v48
	v_mov_b32_e32 v68, v48
	v_mov_b32_e32 v69, v48
	v_mov_b32_e32 v70, v48
	v_mov_b32_e32 v71, v48
	v_mov_b32_e32 v64, v48
	v_mov_b32_e32 v65, v48
	v_mov_b32_e32 v66, v48
	v_mov_b32_e32 v67, v48
	v_mov_b32_e32 v60, v48
	v_mov_b32_e32 v61, v48
	v_mov_b32_e32 v62, v48
	v_mov_b32_e32 v63, v48
	v_mov_b32_e32 v56, v48
	v_mov_b32_e32 v57, v48
	v_mov_b32_e32 v58, v48
	v_mov_b32_e32 v59, v48
	v_mov_b32_e32 v52, v48
	v_mov_b32_e32 v53, v48
	v_mov_b32_e32 v54, v48
	v_mov_b32_e32 v55, v48
	v_mov_b32_e32 v44, v48
	v_mov_b32_e32 v45, v48
	v_mov_b32_e32 v46, v48
	v_mov_b32_e32 v47, v48
	v_mov_b32_e32 v40, v48
	v_mov_b32_e32 v41, v48
	v_mov_b32_e32 v42, v48
	v_mov_b32_e32 v43, v48
	v_mov_b32_e32 v28, v48
	v_mov_b32_e32 v29, v48
	v_mov_b32_e32 v30, v48
	v_mov_b32_e32 v31, v48
	v_mov_b32_e32 v16, v48
	v_mov_b32_e32 v17, v48
	v_mov_b32_e32 v18, v48
	v_mov_b32_e32 v19, v48
	v_mov_b32_e32 v12, v48
	v_mov_b32_e32 v13, v48
	v_mov_b32_e32 v14, v48
	v_mov_b32_e32 v15, v48
	v_mov_b32_e32 v20, v48
	v_mov_b32_e32 v21, v48
	v_mov_b32_e32 v22, v48
	v_mov_b32_e32 v23, v48
	v_mov_b32_e32 v24, v48
	v_mov_b32_e32 v25, v48
	v_mov_b32_e32 v26, v48
	v_mov_b32_e32 v27, v48
	v_mov_b32_e32 v32, v48
	v_mov_b32_e32 v33, v48
	v_mov_b32_e32 v34, v48
	v_mov_b32_e32 v35, v48
	v_mov_b32_e32 v36, v48
	v_mov_b32_e32 v37, v48
	v_mov_b32_e32 v38, v48
	v_mov_b32_e32 v39, v48
	v_mov_b32_e32 v8, v48
	v_mov_b32_e32 v9, v48
	v_mov_b32_e32 v10, v48
	v_mov_b32_e32 v11, v48
	v_mov_b32_e32 v4, v48
	v_mov_b32_e32 v5, v48
	v_mov_b32_e32 v6, v48
	v_mov_b32_e32 v7, v48
	v_mov_b32_e32 v0, v48
	v_mov_b32_e32 v1, v48
	v_mov_b32_e32 v2, v48
	v_mov_b32_e32 v3, v48
.LBB0_3:
	v_lshrrev_b32_e32 v121, 4, v114
	v_mul_hi_u32 v121, v121, s36
	s_mov_b32 s42, s13
	s_mov_b32 s13, s43
	s_mov_b32 s43, s15
	s_mov_b32 s15, s44
	v_mul_lo_u32 v121, v121, s37
	v_add_u32_e32 v122, v102, v101
	v_add_u32_e32 v123, 0x80, v122
	s_add_i32 m0, s13, s31
	s_barrier
	buffer_load_dwordx4 v123, s[24:27], 0 offen lds
	v_add_u32_e32 v123, 0x6c080, v122
	s_add_i32 m0, s13, s33
	s_nop 0
	buffer_load_dwordx4 v123, s[24:27], 0 offen lds
	v_add_u32_e32 v123, 0xd8080, v122
	s_add_i32 m0, s13, s34
	v_add_u32_e32 v122, 0x144080, v122
	buffer_load_dwordx4 v123, s[24:27], 0 offen lds
	s_add_i32 m0, s13, s35
	s_nop 0
	buffer_load_dwordx4 v122, s[24:27], 0 offen lds
	v_add_u32_e32 v122, s0, v101
	v_lshrrev_b32_e32 v122, 4, v122
	v_mul_hi_u32 v122, v122, s36
	v_add_u32_e32 v123, v122, v103
	v_add_u32_e32 v122, v122, v104
	v_mad_u32_u24 v123, v123, s30, v121
	v_mul_u32_u24_e32 v122, 0x1b00, v122
	v_add3_u32 v123, v115, v123, s38
	s_add_i32 m0, s44, s31
	v_add3_u32 v121, v122, v121, v115
	buffer_load_dwordx4 v123, s[20:23], 0 offen lds
	v_add_u32_e32 v122, 0xfff94000, v121
	s_add_i32 m0, s44, s33
	s_nop 0
	buffer_load_dwordx4 v122, s[20:23], 0 offen lds
	s_add_i32 m0, s44, s34
	s_nop 0
	buffer_load_dwordx4 v121, s[20:23], 0 offen lds
	v_add_u32_e32 v127, 64, v124
	v_cmp_gt_i32_e32 vcc, s39, v124
	s_nop 1
	v_cndmask_b32_e32 v121, v127, v113, vcc
	v_mul_hi_i32 v121, v121, s40
	v_lshrrev_b32_e32 v122, 31, v121
	v_ashrrev_i32_e32 v121, 5, v121
	v_add_u32_e32 v121, v121, v122
	v_cndmask_b32_e64 v122, 0, -1, vcc
	v_xor_b32_e32 v122, v121, v122
	v_add_u32_e32 v121, v122, v97
	v_mul_lo_u32 v121, v121, s14
	v_mul_lo_u32 v123, v122, s41
	v_sub_u32_e32 v121, v121, v123
	v_add3_u32 v124, v110, v121, s1
	v_add_u32_e32 v121, v109, v122
	v_mul_lo_u32 v121, s14, v121
	v_sub_u32_e32 v121, v121, v123
	v_add3_u32 v125, v110, v121, s1
	buffer_load_dword v126, v124, s[4:7], 0 offen
	buffer_load_dword v121, v125, s[4:7], 0 offen
	v_add_u32_e32 v124, v122, v99
	v_add_u32_e32 v125, v111, v122
	v_add_u32_e32 v122, v112, v122
	v_mul_lo_u32 v124, v124, s12
	v_mul_lo_u32 v122, s12, v122
	v_sub_u32_e32 v124, v124, v123
	v_mul_lo_u32 v125, s12, v125
	v_sub_u32_e32 v122, v122, v123
	v_add3_u32 v124, v110, v124, s1
	v_sub_u32_e32 v125, v125, v123
	v_add3_u32 v122, v110, v122, s1
	v_add3_u32 v128, v110, v125, s1
	buffer_load_dword v125, v124, s[16:19], 0 offen
	buffer_load_dword v123, v128, s[16:19], 0 offen
	s_nop 0
	buffer_load_dword v122, v122, s[16:19], 0 offen
	v_add_u32_e32 v124, s42, v105
	ds_read_b128 v[128:131], v124
	ds_read_b128 v[132:135], v124 offset:2048
	ds_read_b128 v[136:139], v124 offset:4096
	ds_read_b128 v[140:143], v124 offset:6144
	v_add_u32_e32 v124, s43, v107
	ds_read_b128 v[144:147], v124
	ds_read_b128 v[148:151], v124 offset:2048
	ds_read_b128 v[152:155], v124 offset:4096
	ds_read_b128 v[156:159], v124 offset:6144
	ds_read_b128 v[160:163], v124 offset:8192
	ds_read_b128 v[164:167], v124 offset:10240
	s_barrier
	s_setprio 1
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[128:131], v[144:147], v[48:51], v117, v120 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[128:131], v[148:151], v[92:95], v117, v120 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[128:131], v[152:155], v[88:91], v117, v119 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[84:87], v[128:131], v[156:159], v[84:87], v117, v119 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[80:83], v[128:131], v[160:163], v[80:83], v117, v118 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[128:131], v[164:167], v[76:79], v117, v118 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[132:135], v[144:147], v[72:75], v117, v120 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[132:135], v[148:151], v[68:71], v117, v120 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[132:135], v[152:155], v[64:67], v117, v119 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[132:135], v[156:159], v[60:63], v117, v119 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[132:135], v[160:163], v[56:59], v117, v118 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[132:135], v[164:167], v[52:55], v117, v118 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[136:139], v[144:147], v[44:47], v116, v120 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[136:139], v[148:151], v[40:43], v116, v120 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[136:139], v[152:155], v[28:31], v116, v119 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[136:139], v[156:159], v[16:19], v116, v119 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[136:139], v[160:163], v[12:15], v116, v118 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[136:139], v[164:167], v[20:23], v116, v118 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[140:143], v[144:147], v[24:27], v116, v120 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[140:143], v[148:151], v[32:35], v116, v120 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[140:143], v[152:155], v[36:39], v116, v119 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[140:143], v[156:159], v[8:11], v116, v119 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[140:143], v[160:163], v[4:7], v116, v118 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[140:143], v[164:167], v[0:3], v116, v118 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_barrier
	v_add_u32_e32 v124, s42, v106
	ds_read_b128 v[128:131], v124
	ds_read_b128 v[132:135], v124 offset:2048
	ds_read_b128 v[136:139], v124 offset:4096
	ds_read_b128 v[140:143], v124 offset:6144
	v_add_u32_e32 v124, s43, v108
	ds_read_b128 v[144:147], v124
	ds_read_b128 v[148:151], v124 offset:2048
	ds_read_b128 v[152:155], v124 offset:4096
	ds_read_b128 v[156:159], v124 offset:6144
	ds_read_b128 v[160:163], v124 offset:8192
	ds_read_b128 v[164:167], v124 offset:10240
	s_waitcnt vmcnt(5)
	s_barrier
	s_setprio 1
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[128:131], v[144:147], v[48:51], v117, v120 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[128:131], v[148:151], v[92:95], v117, v120 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[128:131], v[152:155], v[88:91], v117, v119 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[84:87], v[128:131], v[156:159], v[84:87], v117, v119 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[80:83], v[128:131], v[160:163], v[80:83], v117, v118 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[128:131], v[164:167], v[76:79], v117, v118 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[132:135], v[144:147], v[72:75], v117, v120 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[132:135], v[148:151], v[68:71], v117, v120 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[132:135], v[152:155], v[64:67], v117, v119 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[132:135], v[156:159], v[60:63], v117, v119 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[132:135], v[160:163], v[56:59], v117, v118 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[132:135], v[164:167], v[52:55], v117, v118 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[136:139], v[144:147], v[44:47], v116, v120 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[136:139], v[148:151], v[40:43], v116, v120 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[136:139], v[152:155], v[28:31], v116, v119 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[136:139], v[156:159], v[16:19], v116, v119 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[136:139], v[160:163], v[12:15], v116, v118 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[136:139], v[164:167], v[20:23], v116, v118 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[140:143], v[144:147], v[24:27], v116, v120 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[140:143], v[148:151], v[32:35], v116, v120 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[140:143], v[152:155], v[36:39], v116, v119 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[140:143], v[156:159], v[8:11], v116, v119 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[140:143], v[160:163], v[4:7], v116, v118 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[140:143], v[164:167], v[0:3], v116, v118 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	s_addk_i32 s1, 0x100
	s_addk_i32 s0, 0x80
	v_subrev_u32_e32 v113, 64, v113
	v_add_u32_e32 v114, 0x80, v114
	v_add_u32_e32 v115, 0x800, v115
	v_add_u32_e32 v102, 0x80, v102
	s_cmpk_lg_i32 s1, 0x3500
	v_mov_b32_e32 v124, v127
	s_mov_b32 s44, s43
	s_mov_b32 s43, s42
	s_waitcnt vmcnt(0)
	v_mov_b32_e32 v118, v122
	v_mov_b32_e32 v119, v123
	v_mov_b32_e32 v120, v125
	v_mov_b32_e32 v116, v121
	v_mov_b32_e32 v117, v126
	s_cbranch_scc1 .LBB0_3
	s_andn2_b64 vcc, exec, s[2:3]
	s_cbranch_vccnz .LBB0_6
	s_barrier
.LBB0_6:
	v_add_u32_e32 v101, s15, v107
	v_add_u32_e32 v102, s15, v108
	s_barrier
	ds_read_b128 v[168:171], v101
	ds_read_b128 v[172:175], v101 offset:2048
	ds_read_b128 v[176:179], v102
	ds_read_b128 v[180:183], v102 offset:2048
	ds_read_b128 v[184:187], v101 offset:4096
	ds_read_b128 v[136:139], v101 offset:6144
	ds_read_b128 v[188:191], v102 offset:4096
	ds_read_b128 v[132:135], v102 offset:6144
	ds_read_b128 v[128:131], v101 offset:8192
	ds_read_b128 v[112:115], v101 offset:10240
	ds_read_b128 v[116:119], v102 offset:8192
	ds_read_b128 v[108:111], v102 offset:10240
	v_add_u32_e32 v101, s13, v105
	ds_read_b128 v[156:159], v101
	ds_read_b128 v[192:195], v101 offset:2048
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[156:159], v[168:171], v[48:51], v126, v125 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add_u32_e32 v102, s13, v106
	ds_read_b128 v[160:163], v102
	ds_read_b128 v[196:199], v102 offset:2048
	ds_read_b128 v[200:203], v101 offset:4096
	ds_read_b128 v[140:143], v101 offset:6144
	ds_read_b128 v[204:207], v102 offset:4096
	ds_read_b128 v[102:105], v102 offset:6144
	s_movk_i32 s0, 0x7fff
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[160:163], v[176:179], v[48:51], v126, v125 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mov_b32_e32 v101, 0x7fc0
	s_mul_hi_u32 s1, s8, s28
	v_lshlrev_b32_e32 v96, 1, v96
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[156:159], v[184:187], v[88:91], v126, v123 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshl_or_b32 v97, v98, 2, v97
	s_nop 2
	v_bfe_u32 v106, v51, 16, 1
	v_bfe_u32 v107, v50, 16, 1
	v_add3_u32 v106, v51, v106, s0
	v_bfe_u32 v120, v49, 16, 1
	v_add3_u32 v107, v50, v107, s0
	v_lshrrev_b32_e32 v106, 16, v106
	v_cmp_o_f32_e32 vcc, v51, v51
	v_add3_u32 v120, v49, v120, s0
	v_lshrrev_b32_e32 v107, 16, v107
	v_mfma_scale_f32_16x16x128_f8f6f4 v[144:147], v[160:163], v[188:191], v[88:91], v126, v123 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_bfe_u32 v124, v48, 16, 1
	v_add3_u32 v124, v48, v124, s0
	v_lshl_add_u32 v96, v99, 1, v96
	v_cndmask_b32_e32 v88, v101, v106, vcc
	v_cmp_o_f32_e32 vcc, v50, v50
	v_lshrrev_b32_e32 v89, 16, v120
	v_mfma_scale_f32_16x16x128_f8f6f4 v[148:151], v[156:159], v[136:139], v[84:87], v126, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v90, 16, v124
	v_mul_lo_u32 v97, s8, v97
	v_lshl_add_u32 v96, v100, 1, v96
	v_cndmask_b32_e32 v84, v101, v107, vcc
	v_cmp_o_f32_e32 vcc, v49, v49
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[156:159], v[172:175], v[92:95], v126, v125 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_mov_b32 s3, 0x27000
	v_cndmask_b32_e32 v85, v101, v89, vcc
	v_cmp_o_f32_e32 vcc, v48, v48
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[156:159], v[128:131], v[80:83], v126, v122 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshl_add_u32 v98, v97, 1, v96
	v_cndmask_b32_e32 v86, v101, v90, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[152:155], v[160:163], v[116:119], v[48:51], v126, v122 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[156:159], v[112:115], v[76:79], v126, v122 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[156:159], v[160:163], v[108:111], v[48:51], v126, v122 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[192:195], v[168:171], v[72:75], v126, v125 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[160:163], v[180:183], v[92:95], v126, v125 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_nop 1
	v_bfe_u32 v72, v147, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[148:151], v[160:163], v[132:135], v[148:151], v126, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[160:163], v[196:199], v[176:179], v[48:51], v126, v125 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_nop 1
	v_bfe_u32 v87, v95, 16, 1
	v_bfe_u32 v80, v94, 16, 1
	v_add3_u32 v76, v95, v87, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[192:195], v[172:175], v[68:71], v126, v125 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v81, v93, 16, 1
	v_add3_u32 v80, v94, v80, s0
	v_lshrrev_b32_e32 v76, 16, v76
	v_mfma_scale_f32_16x16x128_f8f6f4 v[164:167], v[196:199], v[180:183], v[48:51], v126, v125 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v95, v95
	v_bfe_u32 v82, v92, 16, 1
	v_add3_u32 v81, v93, v81, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[192:195], v[184:187], v[64:67], v126, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v77, 16, v80
	v_cndmask_b32_e32 v76, v101, v76, vcc
	v_cmp_o_f32_e32 vcc, v94, v94
	v_add3_u32 v64, v147, v72, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[196:199], v[188:191], v[48:51], v126, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_add3_u32 v82, v92, v82, s0
	v_lshrrev_b32_e32 v78, 16, v81
	v_cndmask_b32_e32 v77, v101, v77, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[192:195], v[136:139], v[60:63], v126, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v93, v93
	v_bfe_u32 v68, v146, 16, 1
	v_bfe_u32 v69, v145, 16, 1
	v_bfe_u32 v70, v144, 16, 1
	v_lshrrev_b32_e32 v79, 16, v82
	v_cndmask_b32_e32 v78, v101, v78, vcc
	v_cmp_o_f32_e32 vcc, v92, v92
	v_add3_u32 v70, v144, v70, s0
	v_add3_u32 v69, v145, v69, s0
	v_add3_u32 v68, v146, v68, s0
	v_cndmask_b32_e32 v79, v101, v79, vcc
	v_lshrrev_b32_e32 v64, 16, v64
	v_lshrrev_b32_e32 v65, 16, v68
	v_lshrrev_b32_e32 v66, 16, v69
	v_lshrrev_b32_e32 v67, 16, v70
	v_cmp_o_f32_e32 vcc, v147, v147
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[196:199], v[132:135], v[48:51], v126, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_bfe_u32 v60, v151, 16, 1
	v_cndmask_b32_e32 v80, v101, v64, vcc
	v_cmp_o_f32_e32 vcc, v146, v146
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[192:195], v[128:131], v[56:59], v126, v122 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e32 v81, v101, v65, vcc
	v_cmp_o_f32_e32 vcc, v145, v145
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[200:203], v[168:171], v[44:47], v121, v125 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v56, v150, 16, 1
	v_cndmask_b32_e32 v82, v101, v66, vcc
	v_cmp_o_f32_e32 vcc, v144, v144
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[200:203], v[172:175], v[40:43], v121, v125 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v57, v149, 16, 1
	v_cndmask_b32_e32 v83, v101, v67, vcc
	v_bfe_u32 v58, v148, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[196:199], v[116:119], v[48:51], v126, v122 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_add3_u32 v56, v150, v56, s0
	v_cmp_o_f32_e32 vcc, v151, v151
	v_add3_u32 v58, v148, v58, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[192:195], v[112:115], v[52:55], v126, v122 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v57, v149, v57, s0
	s_nop 1
	v_add3_u32 v52, v151, v60, s0
	v_lshrrev_b32_e32 v52, 16, v52
	v_lshrrev_b32_e32 v53, 16, v56
	v_cndmask_b32_e32 v87, v101, v52, vcc
	v_cmp_o_f32_e32 vcc, v150, v150
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[196:199], v[108:111], v[48:51], v126, v122 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e32 v89, v101, v53, vcc
	v_cmp_o_f32_e32 vcc, v149, v149
	v_lshrrev_b32_e32 v48, 16, v57
	v_lshrrev_b32_e32 v49, 16, v58
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[204:207], v[176:179], v[44:47], v121, v125 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v90, v101, v48, vcc
	v_cmp_o_f32_e32 vcc, v148, v148
	s_nop 0
	v_bfe_u32 v44, v155, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[200:203], v[128:131], v[12:15], v121, v122 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v45, v154, 16, 1
	v_cndmask_b32_e32 v91, v101, v49, vcc
	v_bfe_u32 v46, v153, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[204:207], v[180:183], v[40:43], v121, v125 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_bfe_u32 v47, v152, 16, 1
	v_cmp_o_f32_e32 vcc, v155, v155
	s_nop 0
	v_add3_u32 v43, v155, v44, s0
	v_add3_u32 v42, v154, v45, s0
	v_lshrrev_b32_e32 v43, 16, v43
	v_add3_u32 v40, v152, v47, s0
	v_add3_u32 v41, v153, v46, s0
	v_lshrrev_b32_e32 v42, 16, v42
	v_cndmask_b32_e32 v92, v101, v43, vcc
	v_cmp_o_f32_e32 vcc, v154, v154
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[200:203], v[184:187], v[28:31], v121, v123 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v94, 16, v41
	v_lshrrev_b32_e32 v95, 16, v40
	v_cndmask_b32_e32 v93, v101, v42, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[204:207], v[116:119], v[12:15], v121, v122 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v153, v153
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[200:203], v[112:115], v[20:23], v121, v122 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[204:207], v[188:191], v[28:31], v121, v123 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[204:207], v[108:111], v[12:15], v121, v122 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[140:143], v[168:171], v[24:27], v121, v125 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 2
	v_bfe_u32 v26, v159, 16, 1
	v_cndmask_b32_e32 v24, v101, v94, vcc
	v_cmp_o_f32_e32 vcc, v152, v152
	v_bfe_u32 v27, v158, 16, 1
	v_add3_u32 v26, v159, v26, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[200:203], v[136:139], v[16:19], v121, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v25, v101, v95, vcc
	v_add3_u32 v27, v158, v27, s0
	v_lshrrev_b32_e32 v26, 16, v26
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[102:105], v[176:179], v[12:15], v121, v125 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v159, v159
	v_lshrrev_b32_e32 v27, 16, v27
	v_bfe_u32 v94, v165, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[140:143], v[172:175], v[32:35], v121, v125 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v26, v101, v26, vcc
	v_cmp_o_f32_e32 vcc, v158, v158
	v_bfe_u32 v95, v164, 16, 1
	v_bfe_u32 v32, v157, 16, 1
	v_bfe_u32 v33, v156, 16, 1
	v_add3_u32 v32, v157, v32, s0
	v_add3_u32 v33, v156, v33, s0
	v_lshrrev_b32_e32 v32, 16, v32
	v_cndmask_b32_e32 v27, v101, v27, vcc
	v_cmp_o_f32_e32 vcc, v157, v157
	v_bfe_u32 v34, v163, 16, 1
	v_lshrrev_b32_e32 v33, 16, v33
	v_cndmask_b32_e32 v32, v101, v32, vcc
	v_cmp_o_f32_e32 vcc, v156, v156
	v_bfe_u32 v35, v162, 16, 1
	v_add3_u32 v34, v163, v34, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[204:207], v[132:135], v[16:19], v121, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v33, v101, v33, vcc
	v_add3_u32 v35, v162, v35, s0
	v_lshrrev_b32_e32 v34, 16, v34
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[102:105], v[180:183], v[12:15], v121, v125 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v163, v163
	v_lshrrev_b32_e32 v35, 16, v35
	v_add3_u32 v94, v165, v94, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[140:143], v[184:187], v[36:39], v121, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v34, v101, v34, vcc
	v_cmp_o_f32_e32 vcc, v162, v162
	v_add3_u32 v95, v164, v95, s0
	v_bfe_u32 v36, v161, 16, 1
	v_bfe_u32 v37, v160, 16, 1
	v_add3_u32 v36, v161, v36, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[140:143], v[136:139], v[8:11], v121, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v37, v160, v37, s0
	v_lshrrev_b32_e32 v36, 16, v36
	v_cndmask_b32_e32 v35, v101, v35, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[140:143], v[128:131], v[4:7], v121, v122 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v161, v161
	v_bfe_u32 v38, v167, 16, 1
	v_lshrrev_b32_e32 v37, 16, v37
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[140:143], v[112:115], v[0:3], v121, v122 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v36, v101, v36, vcc
	v_cmp_o_f32_e32 vcc, v160, v160
	v_bfe_u32 v39, v166, 16, 1
	v_add3_u32 v38, v167, v38, s0
	v_cndmask_b32_e32 v37, v101, v37, vcc
	v_add3_u32 v39, v166, v39, s0
	v_lshrrev_b32_e32 v38, 16, v38
	v_cmp_o_f32_e32 vcc, v167, v167
	v_lshrrev_b32_e32 v39, 16, v39
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[102:105], v[188:191], v[12:15], v121, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v38, v101, v38, vcc
	v_cmp_o_f32_e32 vcc, v166, v166
	v_lshrrev_b32_e32 v94, 16, v94
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[102:105], v[132:135], v[8:11], v121, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v39, v101, v39, vcc
	v_cmp_o_f32_e32 vcc, v165, v165
	v_lshrrev_b32_e32 v95, 16, v95
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[102:105], v[116:119], v[4:7], v121, v122 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v94, v101, v94, vcc
	v_cmp_o_f32_e32 vcc, v164, v164
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[102:105], v[108:111], v[0:3], v121, v122 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_bfe_u32 v102, v75, 16, 1
	v_bfe_u32 v103, v74, 16, 1
	v_add3_u32 v102, v75, v102, s0
	v_cndmask_b32_e32 v95, v101, v95, vcc
	v_bfe_u32 v104, v73, 16, 1
	v_add3_u32 v103, v74, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cmp_o_f32_e32 vcc, v75, v75
	v_bfe_u32 v105, v72, 16, 1
	v_add3_u32 v104, v73, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v75, v101, v102, vcc
	v_cmp_o_f32_e32 vcc, v74, v74
	v_add3_u32 v105, v72, v105, s0
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v74, v101, v103, vcc
	v_cmp_o_f32_e32 vcc, v73, v73
	v_bfe_u32 v102, v71, 16, 1
	v_lshrrev_b32_e32 v105, 16, v105
	v_cndmask_b32_e32 v73, v101, v104, vcc
	v_cmp_o_f32_e32 vcc, v72, v72
	v_bfe_u32 v103, v70, 16, 1
	v_add3_u32 v102, v71, v102, s0
	v_cndmask_b32_e32 v72, v101, v105, vcc
	v_bfe_u32 v104, v69, 16, 1
	v_add3_u32 v103, v70, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cmp_o_f32_e32 vcc, v71, v71
	v_bfe_u32 v105, v68, 16, 1
	v_add3_u32 v104, v69, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v71, v101, v102, vcc
	v_cmp_o_f32_e32 vcc, v70, v70
	v_add3_u32 v105, v68, v105, s0
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v70, v101, v103, vcc
	v_cmp_o_f32_e32 vcc, v69, v69
	v_bfe_u32 v102, v67, 16, 1
	v_lshrrev_b32_e32 v105, 16, v105
	v_cndmask_b32_e32 v69, v101, v104, vcc
	v_cmp_o_f32_e32 vcc, v68, v68
	v_bfe_u32 v103, v66, 16, 1
	v_add3_u32 v102, v67, v102, s0
	v_cndmask_b32_e32 v68, v101, v105, vcc
	v_bfe_u32 v104, v65, 16, 1
	v_add3_u32 v103, v66, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cmp_o_f32_e32 vcc, v67, v67
	v_bfe_u32 v105, v64, 16, 1
	v_add3_u32 v104, v65, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v67, v101, v102, vcc
	v_cmp_o_f32_e32 vcc, v66, v66
	v_add3_u32 v105, v64, v105, s0
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v66, v101, v103, vcc
	v_cmp_o_f32_e32 vcc, v65, v65
	v_bfe_u32 v102, v63, 16, 1
	v_lshrrev_b32_e32 v105, 16, v105
	v_cndmask_b32_e32 v65, v101, v104, vcc
	v_cmp_o_f32_e32 vcc, v64, v64
	v_bfe_u32 v103, v62, 16, 1
	v_add3_u32 v102, v63, v102, s0
	v_cndmask_b32_e32 v64, v101, v105, vcc
	v_bfe_u32 v104, v61, 16, 1
	v_add3_u32 v103, v62, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cmp_o_f32_e32 vcc, v63, v63
	v_bfe_u32 v105, v60, 16, 1
	v_add3_u32 v104, v61, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v63, v101, v102, vcc
	v_cmp_o_f32_e32 vcc, v62, v62
	v_add3_u32 v105, v60, v105, s0
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v62, v101, v103, vcc
	v_cmp_o_f32_e32 vcc, v61, v61
	v_bfe_u32 v102, v59, 16, 1
	v_lshrrev_b32_e32 v105, 16, v105
	v_cndmask_b32_e32 v61, v101, v104, vcc
	v_cmp_o_f32_e32 vcc, v60, v60
	v_bfe_u32 v103, v58, 16, 1
	v_add3_u32 v102, v59, v102, s0
	v_cndmask_b32_e32 v60, v101, v105, vcc
	v_bfe_u32 v104, v57, 16, 1
	v_add3_u32 v103, v58, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cmp_o_f32_e32 vcc, v59, v59
	v_bfe_u32 v105, v56, 16, 1
	v_add3_u32 v104, v57, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v59, v101, v102, vcc
	v_cmp_o_f32_e32 vcc, v58, v58
	v_add3_u32 v105, v56, v105, s0
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v58, v101, v103, vcc
	v_cmp_o_f32_e32 vcc, v57, v57
	v_bfe_u32 v102, v55, 16, 1
	v_lshrrev_b32_e32 v105, 16, v105
	v_cndmask_b32_e32 v57, v101, v104, vcc
	v_cmp_o_f32_e32 vcc, v56, v56
	v_bfe_u32 v103, v54, 16, 1
	v_add3_u32 v102, v55, v102, s0
	v_cndmask_b32_e32 v56, v101, v105, vcc
	v_bfe_u32 v104, v53, 16, 1
	v_add3_u32 v103, v54, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cmp_o_f32_e32 vcc, v55, v55
	v_bfe_u32 v105, v52, 16, 1
	v_add3_u32 v104, v53, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v55, v101, v102, vcc
	v_cmp_o_f32_e32 vcc, v54, v54
	v_add3_u32 v105, v52, v105, s0
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v54, v101, v103, vcc
	v_cmp_o_f32_e32 vcc, v53, v53
	v_bfe_u32 v102, v51, 16, 1
	v_lshrrev_b32_e32 v105, 16, v105
	v_cndmask_b32_e32 v53, v101, v104, vcc
	v_cmp_o_f32_e32 vcc, v52, v52
	v_bfe_u32 v103, v50, 16, 1
	v_add3_u32 v102, v51, v102, s0
	v_cndmask_b32_e32 v52, v101, v105, vcc
	v_bfe_u32 v104, v49, 16, 1
	v_add3_u32 v103, v50, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cmp_o_f32_e32 vcc, v51, v51
	v_bfe_u32 v105, v48, 16, 1
	v_add3_u32 v104, v49, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v51, v101, v102, vcc
	v_cmp_o_f32_e32 vcc, v50, v50
	v_add3_u32 v105, v48, v105, s0
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v50, v101, v103, vcc
	v_cmp_o_f32_e32 vcc, v49, v49
	v_bfe_u32 v102, v47, 16, 1
	v_lshrrev_b32_e32 v105, 16, v105
	v_cndmask_b32_e32 v49, v101, v104, vcc
	v_cmp_o_f32_e32 vcc, v48, v48
	v_bfe_u32 v103, v46, 16, 1
	v_add3_u32 v102, v47, v102, s0
	v_cndmask_b32_e32 v48, v101, v105, vcc
	v_bfe_u32 v104, v45, 16, 1
	v_add3_u32 v103, v46, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cmp_o_f32_e32 vcc, v47, v47
	v_bfe_u32 v105, v44, 16, 1
	v_add3_u32 v104, v45, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v47, v101, v102, vcc
	v_cmp_o_f32_e32 vcc, v46, v46
	v_add3_u32 v105, v44, v105, s0
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v46, v101, v103, vcc
	v_cmp_o_f32_e32 vcc, v45, v45
	v_bfe_u32 v102, v43, 16, 1
	v_lshrrev_b32_e32 v105, 16, v105
	v_cndmask_b32_e32 v45, v101, v104, vcc
	v_cmp_o_f32_e32 vcc, v44, v44
	v_bfe_u32 v103, v42, 16, 1
	v_add3_u32 v102, v43, v102, s0
	v_cndmask_b32_e32 v44, v101, v105, vcc
	v_bfe_u32 v104, v41, 16, 1
	v_add3_u32 v103, v42, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cmp_o_f32_e32 vcc, v43, v43
	v_bfe_u32 v105, v40, 16, 1
	v_add3_u32 v104, v41, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v43, v101, v102, vcc
	v_cmp_o_f32_e32 vcc, v42, v42
	v_add3_u32 v105, v40, v105, s0
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v42, v101, v103, vcc
	v_cmp_o_f32_e32 vcc, v41, v41
	v_bfe_u32 v102, v31, 16, 1
	v_lshrrev_b32_e32 v105, 16, v105
	v_cndmask_b32_e32 v41, v101, v104, vcc
	v_cmp_o_f32_e32 vcc, v40, v40
	v_bfe_u32 v103, v30, 16, 1
	v_add3_u32 v102, v31, v102, s0
	v_cndmask_b32_e32 v40, v101, v105, vcc
	v_bfe_u32 v104, v29, 16, 1
	v_add3_u32 v103, v30, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cmp_o_f32_e32 vcc, v31, v31
	v_bfe_u32 v105, v28, 16, 1
	v_add3_u32 v104, v29, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v31, v101, v102, vcc
	v_cmp_o_f32_e32 vcc, v30, v30
	v_add3_u32 v105, v28, v105, s0
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v30, v101, v103, vcc
	v_cmp_o_f32_e32 vcc, v29, v29
	v_bfe_u32 v102, v23, 16, 1
	v_lshrrev_b32_e32 v105, 16, v105
	v_cndmask_b32_e32 v29, v101, v104, vcc
	v_cmp_o_f32_e32 vcc, v28, v28
	v_bfe_u32 v103, v22, 16, 1
	v_add3_u32 v102, v23, v102, s0
	v_cndmask_b32_e32 v28, v101, v105, vcc
	v_bfe_u32 v104, v21, 16, 1
	v_add3_u32 v103, v22, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cmp_o_f32_e32 vcc, v23, v23
	v_bfe_u32 v105, v20, 16, 1
	v_add3_u32 v104, v21, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v23, v101, v102, vcc
	v_cmp_o_f32_e32 vcc, v22, v22
	v_add3_u32 v105, v20, v105, s0
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v22, v101, v103, vcc
	v_cmp_o_f32_e32 vcc, v21, v21
	v_bfe_u32 v102, v19, 16, 1
	v_lshrrev_b32_e32 v105, 16, v105
	v_cndmask_b32_e32 v21, v101, v104, vcc
	v_cmp_o_f32_e32 vcc, v20, v20
	v_bfe_u32 v103, v18, 16, 1
	v_add3_u32 v102, v19, v102, s0
	v_cndmask_b32_e32 v20, v101, v105, vcc
	v_bfe_u32 v104, v17, 16, 1
	v_add3_u32 v103, v18, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cmp_o_f32_e32 vcc, v19, v19
	v_bfe_u32 v105, v16, 16, 1
	v_add3_u32 v104, v17, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v19, v101, v102, vcc
	v_cmp_o_f32_e32 vcc, v18, v18
	v_add3_u32 v105, v16, v105, s0
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v18, v101, v103, vcc
	v_cmp_o_f32_e32 vcc, v17, v17
	v_bfe_u32 v102, v15, 16, 1
	v_lshrrev_b32_e32 v105, 16, v105
	v_cndmask_b32_e32 v17, v101, v104, vcc
	v_cmp_o_f32_e32 vcc, v16, v16
	v_bfe_u32 v103, v14, 16, 1
	v_add3_u32 v102, v15, v102, s0
	v_cndmask_b32_e32 v16, v101, v105, vcc
	v_bfe_u32 v104, v13, 16, 1
	v_add3_u32 v103, v14, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cmp_o_f32_e32 vcc, v15, v15
	v_bfe_u32 v105, v12, 16, 1
	v_add3_u32 v104, v13, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v15, v101, v102, vcc
	v_cmp_o_f32_e32 vcc, v14, v14
	v_add3_u32 v105, v12, v105, s0
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v14, v101, v103, vcc
	v_cmp_o_f32_e32 vcc, v13, v13
	v_bfe_u32 v102, v11, 16, 1
	v_lshrrev_b32_e32 v105, 16, v105
	v_cndmask_b32_e32 v13, v101, v104, vcc
	v_cmp_o_f32_e32 vcc, v12, v12
	v_bfe_u32 v103, v10, 16, 1
	v_add3_u32 v102, v11, v102, s0
	v_cndmask_b32_e32 v12, v101, v105, vcc
	v_bfe_u32 v104, v9, 16, 1
	v_add3_u32 v103, v10, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cmp_o_f32_e32 vcc, v11, v11
	v_bfe_u32 v105, v8, 16, 1
	v_add3_u32 v104, v9, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v11, v101, v102, vcc
	v_cmp_o_f32_e32 vcc, v10, v10
	v_add3_u32 v105, v8, v105, s0
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v10, v101, v103, vcc
	v_cmp_o_f32_e32 vcc, v9, v9
	v_bfe_u32 v102, v7, 16, 1
	v_lshrrev_b32_e32 v105, 16, v105
	v_cndmask_b32_e32 v9, v101, v104, vcc
	v_cmp_o_f32_e32 vcc, v8, v8
	v_bfe_u32 v103, v6, 16, 1
	v_add3_u32 v102, v7, v102, s0
	v_cndmask_b32_e32 v8, v101, v105, vcc
	v_bfe_u32 v104, v5, 16, 1
	v_add3_u32 v103, v6, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cmp_o_f32_e32 vcc, v7, v7
	v_bfe_u32 v105, v4, 16, 1
	v_add3_u32 v104, v5, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v7, v101, v102, vcc
	v_cmp_o_f32_e32 vcc, v6, v6
	v_add3_u32 v105, v4, v105, s0
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v6, v101, v103, vcc
	v_cmp_o_f32_e32 vcc, v5, v5
	v_lshrrev_b32_e32 v105, 16, v105
	v_bfe_u32 v102, v3, 16, 1
	v_cndmask_b32_e32 v5, v101, v104, vcc
	v_cmp_o_f32_e32 vcc, v4, v4
	v_bfe_u32 v103, v2, 16, 1
	v_bfe_u32 v104, v1, 16, 1
	v_cndmask_b32_e32 v4, v101, v105, vcc
	v_bfe_u32 v105, v0, 16, 1
	v_add3_u32 v105, v0, v105, s0
	v_add3_u32 v104, v1, v104, s0
	v_add3_u32 v103, v2, v103, s0
	v_add3_u32 v102, v3, v102, s0
	s_mul_i32 s0, s9, s28
	s_add_i32 s1, s1, s0
	s_mul_i32 s0, s8, s28
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s0, s10, s0
	s_addc_u32 s1, s11, s1
	s_lshl_b32 s2, s29, 1
	s_add_u32 s0, s0, s2
	s_addc_u32 s1, s1, 0
	s_and_b32 s2, s8, 0x3fff
	s_lshl_b32 s2, s2, 16
	s_and_b32 s1, s1, 0xffff
	s_or_b32 s1, s2, s1
	s_or_b32 s1, s1, 2.0
	s_mov_b32 s2, 0x7ffffffd
	s_lshl_b32 s4, s8, 1
	buffer_store_short v86, v98, s[0:3], 0 offen
	v_add_u32_e32 v86, s4, v98
	buffer_store_short v85, v86, s[0:3], 0 offen
	v_add_u32_e32 v85, s4, v86
	buffer_store_short v84, v85, s[0:3], 0 offen
	v_add_u32_e32 v84, s4, v85
	s_lshl_b32 s5, s8, 4
	buffer_store_short v88, v84, s[0:3], 0 offen
	buffer_store_short v79, v98, s[0:3], 0 offen offset:32
	buffer_store_short v78, v86, s[0:3], 0 offen offset:32
	buffer_store_short v77, v85, s[0:3], 0 offen offset:32
	buffer_store_short v76, v84, s[0:3], 0 offen offset:32
	buffer_store_short v83, v98, s[0:3], 0 offen offset:64
	buffer_store_short v82, v86, s[0:3], 0 offen offset:64
	buffer_store_short v81, v85, s[0:3], 0 offen offset:64
	buffer_store_short v80, v84, s[0:3], 0 offen offset:64
	buffer_store_short v91, v98, s[0:3], 0 offen offset:96
	buffer_store_short v90, v86, s[0:3], 0 offen offset:96
	buffer_store_short v89, v85, s[0:3], 0 offen offset:96
	buffer_store_short v87, v84, s[0:3], 0 offen offset:96
	buffer_store_short v25, v98, s[0:3], 0 offen offset:128
	buffer_store_short v24, v86, s[0:3], 0 offen offset:128
	buffer_store_short v93, v85, s[0:3], 0 offen offset:128
	buffer_store_short v92, v84, s[0:3], 0 offen offset:128
	buffer_store_short v33, v98, s[0:3], 0 offen offset:160
	buffer_store_short v32, v86, s[0:3], 0 offen offset:160
	buffer_store_short v27, v85, s[0:3], 0 offen offset:160
	buffer_store_short v26, v84, s[0:3], 0 offen offset:160
	v_add_u32_e32 v24, s5, v97
	v_lshl_add_u32 v25, v24, 1, v96
	v_add_u32_e32 v26, s4, v25
	v_add_u32_e32 v27, s4, v26
	v_add_u32_e32 v32, s4, v27
	v_add_u32_e32 v24, s5, v24
	buffer_store_short v37, v25, s[0:3], 0 offen
	buffer_store_short v36, v26, s[0:3], 0 offen
	buffer_store_short v35, v27, s[0:3], 0 offen
	buffer_store_short v34, v32, s[0:3], 0 offen
	buffer_store_short v95, v25, s[0:3], 0 offen offset:32
	buffer_store_short v94, v26, s[0:3], 0 offen offset:32
	buffer_store_short v39, v27, s[0:3], 0 offen offset:32
	buffer_store_short v38, v32, s[0:3], 0 offen offset:32
	buffer_store_short v72, v25, s[0:3], 0 offen offset:64
	buffer_store_short v73, v26, s[0:3], 0 offen offset:64
	buffer_store_short v74, v27, s[0:3], 0 offen offset:64
	buffer_store_short v75, v32, s[0:3], 0 offen offset:64
	buffer_store_short v68, v25, s[0:3], 0 offen offset:96
	buffer_store_short v69, v26, s[0:3], 0 offen offset:96
	buffer_store_short v70, v27, s[0:3], 0 offen offset:96
	buffer_store_short v71, v32, s[0:3], 0 offen offset:96
	buffer_store_short v64, v25, s[0:3], 0 offen offset:128
	buffer_store_short v65, v26, s[0:3], 0 offen offset:128
	buffer_store_short v66, v27, s[0:3], 0 offen offset:128
	buffer_store_short v67, v32, s[0:3], 0 offen offset:128
	buffer_store_short v60, v25, s[0:3], 0 offen offset:160
	buffer_store_short v61, v26, s[0:3], 0 offen offset:160
	buffer_store_short v62, v27, s[0:3], 0 offen offset:160
	buffer_store_short v63, v32, s[0:3], 0 offen offset:160
	v_lshl_add_u32 v25, v24, 1, v96
	v_add_u32_e32 v26, s4, v25
	v_add_u32_e32 v27, s4, v26
	v_add_u32_e32 v24, s5, v24
	v_lshrrev_b32_e32 v102, 16, v102
	v_cmp_o_f32_e32 vcc, v3, v3
	v_add_u32_e32 v32, s4, v27
	v_lshl_add_u32 v24, v24, 1, v96
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v3, v101, v102, vcc
	v_cmp_o_f32_e32 vcc, v2, v2
	buffer_store_short v56, v25, s[0:3], 0 offen
	buffer_store_short v57, v26, s[0:3], 0 offen
	buffer_store_short v58, v27, s[0:3], 0 offen
	buffer_store_short v59, v32, s[0:3], 0 offen
	buffer_store_short v52, v25, s[0:3], 0 offen offset:32
	buffer_store_short v53, v26, s[0:3], 0 offen offset:32
	buffer_store_short v54, v27, s[0:3], 0 offen offset:32
	buffer_store_short v55, v32, s[0:3], 0 offen offset:32
	buffer_store_short v48, v25, s[0:3], 0 offen offset:64
	buffer_store_short v49, v26, s[0:3], 0 offen offset:64
	buffer_store_short v50, v27, s[0:3], 0 offen offset:64
	buffer_store_short v51, v32, s[0:3], 0 offen offset:64
	buffer_store_short v44, v25, s[0:3], 0 offen offset:96
	buffer_store_short v45, v26, s[0:3], 0 offen offset:96
	buffer_store_short v46, v27, s[0:3], 0 offen offset:96
	buffer_store_short v47, v32, s[0:3], 0 offen offset:96
	buffer_store_short v40, v25, s[0:3], 0 offen offset:128
	buffer_store_short v41, v26, s[0:3], 0 offen offset:128
	buffer_store_short v42, v27, s[0:3], 0 offen offset:128
	buffer_store_short v43, v32, s[0:3], 0 offen offset:128
	buffer_store_short v28, v25, s[0:3], 0 offen offset:160
	buffer_store_short v29, v26, s[0:3], 0 offen offset:160
	buffer_store_short v30, v27, s[0:3], 0 offen offset:160
	buffer_store_short v31, v32, s[0:3], 0 offen offset:160
	buffer_store_short v20, v24, s[0:3], 0 offen
	v_add_u32_e32 v20, s4, v24
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v2, v101, v103, vcc
	v_cmp_o_f32_e32 vcc, v1, v1
	buffer_store_short v21, v20, s[0:3], 0 offen
	v_add_u32_e32 v21, s4, v20
	v_lshrrev_b32_e32 v105, 16, v105
	v_cndmask_b32_e32 v1, v101, v104, vcc
	v_cmp_o_f32_e32 vcc, v0, v0
	buffer_store_short v22, v21, s[0:3], 0 offen
	v_add_u32_e32 v22, s4, v21
	v_cndmask_b32_e32 v0, v101, v105, vcc
	buffer_store_short v23, v22, s[0:3], 0 offen
	buffer_store_short v16, v24, s[0:3], 0 offen offset:32
	buffer_store_short v17, v20, s[0:3], 0 offen offset:32
	buffer_store_short v18, v21, s[0:3], 0 offen offset:32
	buffer_store_short v19, v22, s[0:3], 0 offen offset:32
	buffer_store_short v12, v24, s[0:3], 0 offen offset:64
	buffer_store_short v13, v20, s[0:3], 0 offen offset:64
	buffer_store_short v14, v21, s[0:3], 0 offen offset:64
	buffer_store_short v15, v22, s[0:3], 0 offen offset:64
	buffer_store_short v8, v24, s[0:3], 0 offen offset:96
	buffer_store_short v9, v20, s[0:3], 0 offen offset:96
	buffer_store_short v10, v21, s[0:3], 0 offen offset:96
	buffer_store_short v11, v22, s[0:3], 0 offen offset:96
	buffer_store_short v4, v24, s[0:3], 0 offen offset:128
	buffer_store_short v5, v20, s[0:3], 0 offen offset:128
	buffer_store_short v6, v21, s[0:3], 0 offen offset:128
	buffer_store_short v7, v22, s[0:3], 0 offen offset:128
	buffer_store_short v0, v24, s[0:3], 0 offen offset:160
	buffer_store_short v1, v20, s[0:3], 0 offen offset:160
	buffer_store_short v2, v21, s[0:3], 0 offen offset:160
	buffer_store_short v3, v22, s[0:3], 0 offen offset:160
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wave_mxfp4_static_gemm_256x192x256_15360x9984x13824
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
	.size	wave_mxfp4_static_gemm_256x192x256_15360x9984x13824, .Lfunc_end0-wave_mxfp4_static_gemm_256x192x256_15360x9984x13824

	.set wave_mxfp4_static_gemm_256x192x256_15360x9984x13824.num_vgpr, 208
	.set wave_mxfp4_static_gemm_256x192x256_15360x9984x13824.num_agpr, 0
	.set wave_mxfp4_static_gemm_256x192x256_15360x9984x13824.numbered_sgpr, 45
	.set wave_mxfp4_static_gemm_256x192x256_15360x9984x13824.num_named_barrier, 0
	.set wave_mxfp4_static_gemm_256x192x256_15360x9984x13824.private_seg_size, 0
	.set wave_mxfp4_static_gemm_256x192x256_15360x9984x13824.uses_vcc, 1
	.set wave_mxfp4_static_gemm_256x192x256_15360x9984x13824.uses_flat_scratch, 0
	.set wave_mxfp4_static_gemm_256x192x256_15360x9984x13824.has_dyn_sized_stack, 0
	.set wave_mxfp4_static_gemm_256x192x256_15360x9984x13824.has_recursion, 0
	.set wave_mxfp4_static_gemm_256x192x256_15360x9984x13824.has_indirect_call, 0
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
    .name:           wave_mxfp4_static_gemm_256x192x256_15360x9984x13824
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 256
      - 2
      - 1
    .sgpr_count:     51
    .sgpr_spill_count: 0
    .symbol:         wave_mxfp4_static_gemm_256x192x256_15360x9984x13824.kd
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
