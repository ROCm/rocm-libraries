; To reproduce the .rocmasm from .optimized.ll, run:
; llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -mattr='-fma-mix-insts' -O3 <.optimized.ll> -o <out.rocmasm>

	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.text
	.globl	wave_mxfp4_static_gemm_256x192x256_5376x1792x4096
	.p2align	8
	.type	wave_mxfp4_static_gemm_256x192x256_5376x1792x4096,@function
wave_mxfp4_static_gemm_256x192x256_5376x1792x4096:
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
.LBB0_0:
	v_and_b32_e32 v96, 0x3ff, v0
	v_bfe_u32 v1, v0, 10, 10
	v_lshrrev_b32_e32 v8, 6, v96
	v_lshlrev_b32_e32 v3, 5, v1
	v_lshl_or_b32 v0, v8, 3, v3
	v_lshrrev_b32_e32 v4, 3, v96
	s_mov_b64 s[24:25], s[2:3]
	v_readfirstlane_b32 s2, v0
	v_or_b32_e32 v0, v4, v3
	s_lshl_b32 s16, s16, 8
	v_bitop3_b32 v6, v4, 7, v96 bitop3:0x48
	v_or_b32_e32 v5, s16, v0
	v_lshlrev_b32_e32 v2, 4, v6
	s_and_b32 s3, s25, 0xffff
	s_lshl_b32 s34, s2, 7
	s_or_b32 s25, s3, 0x48000000
	s_mov_b32 s27, 0x27000
	s_mov_b32 s26, 0x7ffffffe
	v_lshl_or_b32 v5, v5, 11, v2
	s_mov_b32 m0, s34
	s_or_b32 s35, s34, 0x2000
	buffer_load_dwordx4 v5, s[24:27], 0 offen lds
	v_or_b32_e32 v7, 0x20000, v5
	s_mov_b32 m0, s35
	s_or_b32 s40, s34, 0x4000
	buffer_load_dwordx4 v7, s[24:27], 0 offen lds
	v_or_b32_e32 v7, 0x40000, v5
	s_mov_b32 m0, s40
	s_or_b32 s41, s34, 0x6000
	buffer_load_dwordx4 v7, s[24:27], 0 offen lds
	v_or_b32_e32 v5, 0x60000, v5
	s_mov_b32 m0, s41
	v_lshlrev_b32_e32 v100, 4, v4
	buffer_load_dwordx4 v5, s[24:27], 0 offen lds
	v_lshrrev_b32_e32 v5, 7, v96
	v_lshlrev_b32_e32 v7, 8, v5
	s_mov_b64 s[20:21], s[6:7]
	s_mul_i32 s33, s17, 0xc0
	v_sub_u32_e32 v9, v100, v7
	v_lshlrev_b32_e32 v6, 8, v6
	v_and_or_b32 v0, v0, 48, s33
	v_add_u32_e32 v9, v6, v9
	s_and_b32 s2, s21, 0xffff
	s_or_b32 s21, s2, 0x48000000
	s_mov_b32 s22, s26
	s_mov_b32 s23, s27
	v_lshl_add_u32 v0, v0, 11, v9
	s_add_i32 m0, s34, 0x10000
	s_cmp_lt_u32 s17, 9
	buffer_load_dwordx4 v0, s[20:23], 0 offen lds
	v_lshlrev_b32_e32 v0, 4, v5
	v_or3_b32 v0, v0, s33, v3
	v_lshl_add_u32 v0, v0, 11, v9
	v_add_u32_e32 v9, 0x20000, v0
	v_bfrev_b32_e32 v101, -2
	s_cselect_b64 s[6:7], -1, 0
	v_cndmask_b32_e64 v9, v101, v9, s[6:7]
	s_add_i32 m0, s35, 0x10000
	s_mul_i32 s15, s15, s16
	s_mul_hi_u32 s2, s14, s16
	buffer_load_dwordx4 v9, s[20:23], 0 offen lds
	s_add_i32 m0, s40, 0x10000
	v_lshrrev_b32_e32 v12, 4, v96
	v_bfe_u32 v11, v96, 5, 1
	s_add_i32 s2, s2, s15
	s_mul_i32 s3, s14, s16
	v_add_u32_e32 v0, 0x40000, v0
	v_lshlrev_b32_e32 v9, 6, v12
	v_lshlrev_b32_e32 v10, 7, v11
	s_add_u32 s28, s4, s3
	v_cndmask_b32_e64 v0, v101, v0, s[6:7]
	v_bfe_u32 v97, v96, 4, 2
	v_lshlrev_b32_e32 v102, 2, v96
	v_add_u32_e32 v14, v9, v10
	s_addc_u32 s2, s5, s2
	s_and_b32 s3, s14, 0x3fff
	s_load_dwordx2 s[18:19], s[0:1], 0x40
	buffer_load_dwordx4 v0, s[20:23], 0 offen lds
	v_and_b32_e32 v98, 0xc0, v96
	v_lshlrev_b32_e32 v0, 6, v97
	v_sub_u32_e32 v14, v102, v14
	s_bitset1_b32 s3, 14
	v_or_b32_e32 v13, v11, v98
	v_add_u32_e32 v14, v14, v0
	s_and_b32 s2, s2, 0xffff
	s_lshl_b32 s3, s3, 16
	s_or_b32 s29, s2, s3
	v_mad_u64_u32 v[16:17], s[2:3], s14, v13, v[14:15]
	s_mov_b32 s30, s26
	s_mov_b32 s31, s27
	v_lshl_add_u32 v13, s14, 5, v16
	buffer_load_dword v117, v16, s[28:31], 0 offen
	buffer_load_dword v116, v13, s[28:31], 0 offen
	v_or_b32_e32 v13, s33, v11
	s_movk_i32 s2, 0x60
	v_mul_u32_u24_e32 v99, 0x60, v1
	v_mad_u32_u24 v103, v1, s2, v13
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s2, s19, s33
	s_mul_hi_u32 s3, s18, s33
	v_or_b32_e32 v13, v11, v99
	s_add_i32 s5, s3, s2
	s_mul_i32 s12, s18, s33
	v_mad_u64_u32 v[16:17], s[2:3], s18, v13, v[14:15]
	s_add_u32 s36, s8, s12
	s_addc_u32 s2, s9, s5
	s_and_b32 s3, s18, 0x3fff
	s_movk_i32 s4, 0x700
	s_bitset1_b32 s3, 14
	s_and_b32 s2, s2, 0xffff
	s_lshl_b32 s3, s3, 16
	v_cmp_gt_u32_e32 vcc, s4, v103
	s_movk_i32 s4, 0x6e0
	v_add_u32_e32 v15, 32, v13
	s_or_b32 s37, s2, s3
	v_cndmask_b32_e32 v18, v101, v16, vcc
	v_mad_u64_u32 v[16:17], s[2:3], s18, v15, v[14:15]
	v_cmp_gt_u32_e32 vcc, s4, v103
	s_movk_i32 s4, 0x6c0
	v_add_u32_e32 v13, 64, v13
	s_mov_b32 s38, s26
	s_mov_b32 s39, s27
	v_cndmask_b32_e32 v16, v101, v16, vcc
	v_mad_u64_u32 v[14:15], s[2:3], s18, v13, v[14:15]
	v_cmp_gt_u32_e32 vcc, s4, v103
	v_cmp_eq_u32_e64 s[2:3], 0, v1
	s_mov_b32 s4, 0
	v_cndmask_b32_e32 v13, v101, v14, vcc
	buffer_load_dword v120, v18, s[36:39], 0 offen
	buffer_load_dword v119, v16, s[36:39], 0 offen
	buffer_load_dword v118, v13, s[36:39], 0 offen
	s_mov_b32 s5, 0x10000
	s_and_b64 vcc, exec, s[2:3]
	s_barrier
	s_waitcnt vmcnt(0)
	s_cbranch_vccnz .LBB0_2
	s_barrier
.LBB0_2:
	v_and_b32_e32 v13, 7, v96
	v_lshlrev_b32_e32 v15, 7, v96
	v_lshlrev_b32_e32 v12, 11, v12
	s_load_dwordx2 s[12:13], s[0:1], 0x48
	v_bitop3_b32 v14, v97, v96, 7 bitop3:0x78
	v_sub_u32_e32 v12, v15, v12
	s_movk_i32 s0, 0x3000
	v_bitop3_b32 v13, v97, v13, 4 bitop3:0x36
	v_lshl_add_u32 v8, v8, 13, v12
	v_lshlrev_b32_e32 v14, 4, v14
	v_mul_lo_u32 v15, v1, s0
	v_lshlrev_b32_e32 v13, 4, v13
	v_or_b32_e32 v104, v8, v14
	v_add_u32_e32 v12, v12, v15
	v_or_b32_e32 v105, v13, v8
	v_add_u32_e32 v8, v99, v11
	v_or_b32_e32 v106, v12, v14
	v_or_b32_e32 v107, v13, v12
	v_add_u32_e32 v12, 0x42, v8
	v_mad_u64_u32 v[12:13], s[0:1], s18, v12, v[0:1]
	v_sub_u32_e32 v12, v12, v9
	v_sub_u32_e32 v108, v12, v10
	v_add_u32_e32 v12, 34, v8
	v_mad_u64_u32 v[12:13], s[8:9], s18, v12, v[0:1]
	v_sub_u32_e32 v12, v12, v9
	v_add_u32_e32 v8, 2, v8
	v_sub_u32_e32 v109, v12, v10
	v_mad_u64_u32 v[12:13], s[8:9], s18, v8, v[0:1]
	v_sub_u32_e32 v8, v12, v9
	v_sub_u32_e32 v110, v8, v10
	v_add_u32_e32 v8, v98, v11
	v_add_u32_e32 v11, 34, v8
	v_mad_u64_u32 v[12:13], s[8:9], s14, v11, v[0:1]
	v_add_u32_e32 v8, 2, v8
	v_sub_u32_e32 v11, v12, v9
	v_mad_u64_u32 v[12:13], s[8:9], s14, v8, v[0:1]
	v_sub_u32_e32 v0, v12, v9
	v_sub_u32_e32 v112, v0, v10
	v_add_u32_e32 v0, v4, v3
	v_lshlrev_b32_e32 v3, 11, v0
	s_mul_i32 s17, s17, 0x60000
	v_and_b32_e32 v3, 0x38000, v3
	v_add_u32_e32 v3, s17, v3
	v_or_b32_e32 v3, v3, v6
	v_sub_u32_e32 v3, v3, v7
	v_lshlrev_b32_e32 v1, 16, v1
	s_movk_i32 s8, 0x7f00
	v_add_u32_e32 v113, 0x800, v3
	v_or_b32_e32 v1, s17, v1
	v_mul_lo_u32 v3, v5, s8
	v_or_b32_e32 v1, v1, v3
	v_add_u32_e32 v0, s16, v0
	v_mov_b32_e32 v48, 0
	s_lshl_b32 s0, s18, 1
	v_sub_u32_e32 v111, v11, v10
	s_lshl_b32 s1, s14, 1
	v_add_u32_e32 v114, v1, v6
	v_lshl_or_b32 v115, v0, 11, v2
	s_mov_b32 s18, 0x8000
	s_mov_b32 s19, 0x16000
	s_mov_b32 s8, 0
	s_mov_b32 s22, s26
	s_mov_b32 s23, s27
	s_mov_b32 s30, s26
	s_mov_b32 s31, s27
	s_movk_i32 s9, 0x6fe
	s_mov_b32 s38, s26
	s_mov_b32 s39, s27
	s_movk_i32 s14, 0x6de
	s_movk_i32 s15, 0x6be
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
	s_mov_b32 s17, s4
	s_mov_b32 s4, s18
	s_mov_b32 s18, s5
	s_mov_b32 s5, s19
	v_add_u32_e32 v121, s8, v115
	v_add_u32_e32 v122, 0x80, v121
	s_add_i32 m0, s4, s34
	s_barrier
	buffer_load_dwordx4 v122, s[24:27], 0 offen lds
	v_add_u32_e32 v122, 0x20080, v121
	s_add_i32 m0, s4, s35
	s_nop 0
	buffer_load_dwordx4 v122, s[24:27], 0 offen lds
	v_add_u32_e32 v122, 0x40080, v121
	s_add_i32 m0, s4, s40
	v_add_u32_e32 v121, 0x60080, v121
	buffer_load_dwordx4 v122, s[24:27], 0 offen lds
	s_add_i32 m0, s4, s41
	s_nop 0
	buffer_load_dwordx4 v121, s[24:27], 0 offen lds
	v_add_u32_e32 v121, v113, v100
	s_add_i32 m0, s19, s34
	s_nop 0
	buffer_load_dwordx4 v121, s[20:23], 0 offen lds
	v_add_u32_e32 v121, v100, v114
	v_add_u32_e32 v122, 0x20800, v121
	v_cndmask_b32_e64 v122, v101, v122, s[6:7]
	s_add_i32 m0, s19, s35
	v_add_u32_e32 v121, 0x40800, v121
	buffer_load_dwordx4 v122, s[20:23], 0 offen lds
	v_cndmask_b32_e64 v121, v101, v121, s[6:7]
	s_add_i32 m0, s19, s40
	s_nop 0
	buffer_load_dwordx4 v121, s[20:23], 0 offen lds
	v_add_u32_e32 v122, v112, v102
	v_add_u32_e32 v123, v111, v102
	buffer_load_dword v125, v122, s[28:31], 0 offen
	buffer_load_dword v121, v123, s[28:31], 0 offen
	v_add_u32_e32 v122, v102, v110
	v_cmp_gt_u32_e32 vcc, s9, v103
	v_add_u32_e32 v123, v102, v109
	s_nop 0
	v_cndmask_b32_e32 v122, v101, v122, vcc
	v_cmp_gt_u32_e32 vcc, s14, v103
	s_nop 1
	v_cndmask_b32_e32 v126, v101, v123, vcc
	v_add_u32_e32 v123, v102, v108
	v_cmp_gt_u32_e32 vcc, s15, v103
	s_nop 1
	v_cndmask_b32_e32 v127, v101, v123, vcc
	buffer_load_dword v124, v122, s[36:39], 0 offen
	buffer_load_dword v123, v126, s[36:39], 0 offen
	s_nop 0
	buffer_load_dword v122, v127, s[36:39], 0 offen
	v_add_u32_e32 v126, s17, v104
	ds_read_b128 v[128:131], v126
	ds_read_b128 v[132:135], v126 offset:2048
	ds_read_b128 v[136:139], v126 offset:4096
	ds_read_b128 v[140:143], v126 offset:6144
	v_add_u32_e32 v126, s18, v106
	ds_read_b128 v[144:147], v126
	ds_read_b128 v[148:151], v126 offset:2048
	ds_read_b128 v[152:155], v126 offset:4096
	ds_read_b128 v[156:159], v126 offset:6144
	ds_read_b128 v[160:163], v126 offset:8192
	ds_read_b128 v[164:167], v126 offset:10240
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
	v_add_u32_e32 v126, s17, v105
	ds_read_b128 v[128:131], v126
	ds_read_b128 v[132:135], v126 offset:2048
	ds_read_b128 v[136:139], v126 offset:4096
	ds_read_b128 v[140:143], v126 offset:6144
	v_add_u32_e32 v126, s18, v107
	ds_read_b128 v[144:147], v126
	ds_read_b128 v[148:151], v126 offset:2048
	ds_read_b128 v[152:155], v126 offset:4096
	ds_read_b128 v[156:159], v126 offset:6144
	ds_read_b128 v[160:163], v126 offset:8192
	ds_read_b128 v[164:167], v126 offset:10240
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
	s_addk_i32 s8, 0x80
	v_add_u32_e32 v108, s0, v108
	v_add_u32_e32 v109, s0, v109
	v_add_u32_e32 v110, s0, v110
	v_add_u32_e32 v103, 2, v103
	v_add_u32_e32 v111, s1, v111
	v_add_u32_e32 v112, s1, v112
	v_add_u32_e32 v113, 0x800, v113
	v_add_u32_e32 v114, 0x800, v114
	s_cmpk_lg_i32 s8, 0x780
	s_mov_b32 s19, s18
	s_mov_b32 s18, s17
	s_waitcnt vmcnt(0)
	v_mov_b32_e32 v118, v122
	v_mov_b32_e32 v119, v123
	v_mov_b32_e32 v120, v124
	v_mov_b32_e32 v116, v121
	v_mov_b32_e32 v117, v125
	s_cbranch_scc1 .LBB0_3
	s_andn2_b64 vcc, exec, s[2:3]
	s_cbranch_vccnz .LBB0_6
	s_barrier
.LBB0_6:
	v_add_u32_e32 v100, s5, v106
	v_add_u32_e32 v101, s5, v107
	s_barrier
	ds_read_b128 v[166:169], v100
	ds_read_b128 v[170:173], v100 offset:2048
	ds_read_b128 v[174:177], v101
	ds_read_b128 v[178:181], v101 offset:2048
	ds_read_b128 v[182:185], v100 offset:4096
	ds_read_b128 v[134:137], v100 offset:6144
	ds_read_b128 v[186:189], v101 offset:4096
	ds_read_b128 v[130:133], v101 offset:6144
	ds_read_b128 v[126:129], v100 offset:8192
	ds_read_b128 v[110:113], v100 offset:10240
	ds_read_b128 v[114:117], v101 offset:8192
	ds_read_b128 v[106:109], v101 offset:10240
	v_add_u32_e32 v100, s4, v104
	ds_read_b128 v[154:157], v100
	ds_read_b128 v[190:193], v100 offset:2048
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[154:157], v[166:169], v[48:51], v125, v124 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add_u32_e32 v101, s4, v105
	ds_read_b128 v[158:161], v101
	ds_read_b128 v[194:197], v101 offset:2048
	ds_read_b128 v[198:201], v100 offset:4096
	ds_read_b128 v[138:141], v100 offset:6144
	ds_read_b128 v[202:205], v101 offset:4096
	ds_read_b128 v[102:105], v101 offset:6144
	s_movk_i32 s0, 0x7fff
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[158:161], v[174:177], v[48:51], v125, v124 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mov_b32_e32 v100, 0x7fc0
	s_mul_hi_u32 s1, s12, s16
	v_lshl_or_b32 v97, v97, 2, v98
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[154:157], v[182:185], v[88:91], v125, v123 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_movk_i32 s2, 0x700
	s_nop 2
	v_bfe_u32 v101, v51, 16, 1
	v_bfe_u32 v118, v50, 16, 1
	v_add3_u32 v101, v51, v101, s0
	v_bfe_u32 v119, v49, 16, 1
	v_add3_u32 v118, v50, v118, s0
	v_lshrrev_b32_e32 v101, 16, v101
	v_cmp_o_f32_e32 vcc, v51, v51
	v_add3_u32 v119, v49, v119, s0
	v_lshrrev_b32_e32 v118, 16, v118
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[158:161], v[186:189], v[88:91], v125, v123 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_bfe_u32 v120, v48, 16, 1
	v_add3_u32 v120, v48, v120, s0
	v_mul_lo_u32 v98, s12, v97
	v_cndmask_b32_e32 v88, v100, v101, vcc
	v_cmp_o_f32_e32 vcc, v50, v50
	v_lshrrev_b32_e32 v89, 16, v119
	v_mfma_scale_f32_16x16x128_f8f6f4 v[146:149], v[154:157], v[134:137], v[84:87], v125, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v90, 16, v120
	s_movk_i32 s4, 0x6c0
	s_movk_i32 s6, 0x6b0
	v_cndmask_b32_e32 v84, v100, v118, vcc
	v_cmp_o_f32_e32 vcc, v49, v49
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[154:157], v[170:173], v[92:95], v125, v124 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e32 v85, v100, v89, vcc
	v_cmp_o_f32_e32 vcc, v48, v48
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[154:157], v[126:129], v[80:83], v125, v122 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e32 v86, v100, v90, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[150:153], v[158:161], v[114:117], v[48:51], v125, v122 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[154:157], v[110:113], v[76:79], v125, v122 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[154:157], v[158:161], v[106:109], v[48:51], v125, v122 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[190:193], v[166:169], v[72:75], v125, v124 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[158:161], v[178:181], v[92:95], v125, v124 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_nop 1
	v_bfe_u32 v72, v145, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[146:149], v[158:161], v[130:133], v[146:149], v125, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[158:161], v[194:197], v[174:177], v[48:51], v125, v124 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_nop 1
	v_bfe_u32 v87, v95, 16, 1
	v_bfe_u32 v80, v94, 16, 1
	v_add3_u32 v76, v95, v87, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[190:193], v[170:173], v[68:71], v125, v124 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v81, v93, 16, 1
	v_add3_u32 v80, v94, v80, s0
	v_lshrrev_b32_e32 v76, 16, v76
	v_mfma_scale_f32_16x16x128_f8f6f4 v[162:165], v[194:197], v[178:181], v[48:51], v125, v124 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v95, v95
	v_bfe_u32 v82, v92, 16, 1
	v_add3_u32 v81, v93, v81, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[190:193], v[182:185], v[64:67], v125, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v77, 16, v80
	v_cndmask_b32_e32 v76, v100, v76, vcc
	v_cmp_o_f32_e32 vcc, v94, v94
	v_add3_u32 v64, v145, v72, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[194:197], v[186:189], v[48:51], v125, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_add3_u32 v82, v92, v82, s0
	v_lshrrev_b32_e32 v79, 16, v81
	v_cndmask_b32_e32 v78, v100, v77, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[190:193], v[134:137], v[60:63], v125, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v93, v93
	v_bfe_u32 v68, v144, 16, 1
	v_bfe_u32 v69, v143, 16, 1
	v_bfe_u32 v70, v142, 16, 1
	v_lshrrev_b32_e32 v81, 16, v82
	v_cndmask_b32_e32 v80, v100, v79, vcc
	v_cmp_o_f32_e32 vcc, v92, v92
	v_add3_u32 v70, v142, v70, s0
	v_add3_u32 v69, v143, v69, s0
	v_add3_u32 v68, v144, v68, s0
	v_cndmask_b32_e32 v89, v100, v81, vcc
	v_lshrrev_b32_e32 v64, 16, v64
	v_lshrrev_b32_e32 v65, 16, v68
	v_lshrrev_b32_e32 v66, 16, v69
	v_lshrrev_b32_e32 v67, 16, v70
	v_cmp_o_f32_e32 vcc, v145, v145
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[194:197], v[130:133], v[48:51], v125, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_bfe_u32 v60, v149, 16, 1
	v_cndmask_b32_e32 v77, v100, v64, vcc
	v_cmp_o_f32_e32 vcc, v144, v144
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[190:193], v[126:129], v[56:59], v125, v122 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e32 v81, v100, v65, vcc
	v_cmp_o_f32_e32 vcc, v143, v143
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[198:201], v[166:169], v[44:47], v121, v124 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v56, v148, 16, 1
	v_cndmask_b32_e32 v83, v100, v66, vcc
	v_cmp_o_f32_e32 vcc, v142, v142
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[198:201], v[170:173], v[40:43], v121, v124 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v57, v147, 16, 1
	v_cndmask_b32_e32 v92, v100, v67, vcc
	v_bfe_u32 v58, v146, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[194:197], v[114:117], v[48:51], v125, v122 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_add3_u32 v56, v148, v56, s0
	v_cmp_o_f32_e32 vcc, v149, v149
	v_add3_u32 v58, v146, v58, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[190:193], v[110:113], v[52:55], v125, v122 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v57, v147, v57, s0
	s_nop 1
	v_add3_u32 v52, v149, v60, s0
	v_lshrrev_b32_e32 v52, 16, v52
	v_lshrrev_b32_e32 v53, 16, v56
	v_cndmask_b32_e32 v79, v100, v52, vcc
	v_cmp_o_f32_e32 vcc, v148, v148
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[194:197], v[106:109], v[48:51], v125, v122 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e32 v87, v100, v53, vcc
	v_cmp_o_f32_e32 vcc, v147, v147
	v_lshrrev_b32_e32 v48, 16, v57
	v_lshrrev_b32_e32 v49, 16, v58
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[202:205], v[174:177], v[44:47], v121, v124 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v90, v100, v48, vcc
	v_cmp_o_f32_e32 vcc, v146, v146
	s_nop 0
	v_bfe_u32 v44, v153, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[198:201], v[126:129], v[12:15], v121, v122 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v45, v152, 16, 1
	v_cndmask_b32_e32 v93, v100, v49, vcc
	v_bfe_u32 v46, v151, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[202:205], v[178:181], v[40:43], v121, v124 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_bfe_u32 v47, v150, 16, 1
	v_cmp_o_f32_e32 vcc, v153, v153
	s_nop 0
	v_add3_u32 v43, v153, v44, s0
	v_add3_u32 v42, v152, v45, s0
	v_lshrrev_b32_e32 v43, 16, v43
	v_add3_u32 v40, v150, v47, s0
	v_add3_u32 v41, v151, v46, s0
	v_lshrrev_b32_e32 v42, 16, v42
	v_cndmask_b32_e32 v82, v100, v43, vcc
	v_cmp_o_f32_e32 vcc, v152, v152
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[198:201], v[182:185], v[28:31], v121, v123 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v94, 16, v41
	v_lshrrev_b32_e32 v95, 16, v40
	v_cndmask_b32_e32 v91, v100, v42, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[202:205], v[114:117], v[12:15], v121, v122 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v151, v151
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[198:201], v[110:113], v[20:23], v121, v122 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[202:205], v[186:189], v[28:31], v121, v123 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[202:205], v[106:109], v[12:15], v121, v122 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[138:141], v[166:169], v[24:27], v121, v124 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 2
	v_bfe_u32 v24, v157, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[198:201], v[134:137], v[16:19], v121, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v25, v156, 16, 1
	v_add3_u32 v24, v157, v24, s0
	v_bfe_u32 v26, v155, 16, 1
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[102:105], v[174:177], v[12:15], v121, v124 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_add3_u32 v25, v156, v25, s0
	v_lshrrev_b32_e32 v24, 16, v24
	v_bfe_u32 v27, v154, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[138:141], v[170:173], v[32:35], v121, v124 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v26, v155, v26, s0
	v_add3_u32 v27, v154, v27, s0
	v_lshrrev_b32_e32 v26, 16, v26
	v_cndmask_b32_e32 v33, v100, v94, vcc
	v_cmp_o_f32_e32 vcc, v150, v150
	v_lshrrev_b32_e32 v32, 16, v25
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[202:205], v[130:133], v[16:19], v121, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v34, v100, v95, vcc
	v_cmp_o_f32_e32 vcc, v157, v157
	v_lshrrev_b32_e32 v27, 16, v27
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[102:105], v[178:181], v[12:15], v121, v124 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v25, v100, v24, vcc
	v_cmp_o_f32_e32 vcc, v156, v156
	v_bfe_u32 v24, v161, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[138:141], v[182:185], v[36:39], v121, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v35, v100, v32, vcc
	v_cmp_o_f32_e32 vcc, v155, v155
	v_add3_u32 v24, v161, v24, s0
	v_lshrrev_b32_e32 v24, 16, v24
	v_cndmask_b32_e32 v36, v100, v26, vcc
	v_cmp_o_f32_e32 vcc, v154, v154
	v_bfe_u32 v26, v160, 16, 1
	v_add3_u32 v26, v160, v26, s0
	v_cndmask_b32_e32 v37, v100, v27, vcc
	v_bfe_u32 v27, v159, 16, 1
	v_cmp_o_f32_e32 vcc, v161, v161
	v_bfe_u32 v32, v158, 16, 1
	v_add3_u32 v27, v159, v27, s0
	v_lshrrev_b32_e32 v26, 16, v26
	v_cndmask_b32_e32 v24, v100, v24, vcc
	v_cmp_o_f32_e32 vcc, v160, v160
	v_add3_u32 v32, v158, v32, s0
	v_lshrrev_b32_e32 v38, 16, v27
	v_cndmask_b32_e32 v27, v100, v26, vcc
	v_cmp_o_f32_e32 vcc, v159, v159
	v_lshrrev_b32_e32 v32, 16, v32
	v_bfe_u32 v26, v165, 16, 1
	v_cndmask_b32_e32 v38, v100, v38, vcc
	v_cmp_o_f32_e32 vcc, v158, v158
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[138:141], v[134:137], v[8:11], v121, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v26, v165, v26, s0
	v_cndmask_b32_e32 v39, v100, v32, vcc
	v_bfe_u32 v32, v164, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[138:141], v[126:129], v[4:7], v121, v122 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v94, v163, 16, 1
	v_add3_u32 v32, v164, v32, s0
	v_lshrrev_b32_e32 v26, 16, v26
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[138:141], v[110:113], v[0:3], v121, v122 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v165, v165
	v_bfe_u32 v95, v162, 16, 1
	v_add3_u32 v94, v163, v94, s0
	v_lshrrev_b32_e32 v32, 16, v32
	v_cndmask_b32_e32 v26, v100, v26, vcc
	v_cmp_o_f32_e32 vcc, v164, v164
	v_add3_u32 v95, v162, v95, s0
	v_lshrrev_b32_e32 v94, 16, v94
	v_cndmask_b32_e32 v101, v100, v32, vcc
	v_cmp_o_f32_e32 vcc, v163, v163
	v_bfe_u32 v32, v75, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[102:105], v[186:189], v[12:15], v121, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v95, 16, v95
	v_cndmask_b32_e32 v94, v100, v94, vcc
	v_cmp_o_f32_e32 vcc, v162, v162
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[102:105], v[130:133], v[8:11], v121, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_add3_u32 v32, v75, v32, s0
	v_cndmask_b32_e32 v95, v100, v95, vcc
	v_lshrrev_b32_e32 v32, 16, v32
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[102:105], v[114:117], v[4:7], v121, v122 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v75, v75
	v_bfe_u32 v75, v71, 16, 1
	v_add3_u32 v75, v71, v75, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[102:105], v[106:109], v[0:3], v121, v122 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_bfe_u32 v102, v74, 16, 1
	v_bfe_u32 v103, v73, 16, 1
	v_add3_u32 v102, v74, v102, s0
	v_bfe_u32 v104, v72, 16, 1
	v_add3_u32 v103, v73, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cndmask_b32_e32 v32, v100, v32, vcc
	v_cmp_o_f32_e32 vcc, v74, v74
	v_add3_u32 v104, v72, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v74, v100, v102, vcc
	v_cmp_o_f32_e32 vcc, v73, v73
	v_lshrrev_b32_e32 v104, 16, v104
	v_bfe_u32 v102, v70, 16, 1
	v_cndmask_b32_e32 v73, v100, v103, vcc
	v_cmp_o_f32_e32 vcc, v72, v72
	v_bfe_u32 v103, v69, 16, 1
	v_add3_u32 v102, v70, v102, s0
	v_cndmask_b32_e32 v72, v100, v104, vcc
	v_lshrrev_b32_e32 v75, 16, v75
	v_cmp_o_f32_e32 vcc, v71, v71
	v_bfe_u32 v104, v68, 16, 1
	v_add3_u32 v103, v69, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cndmask_b32_e32 v71, v100, v75, vcc
	v_cmp_o_f32_e32 vcc, v70, v70
	v_add3_u32 v104, v68, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v70, v100, v102, vcc
	v_cmp_o_f32_e32 vcc, v69, v69
	v_bfe_u32 v75, v67, 16, 1
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v69, v100, v103, vcc
	v_cmp_o_f32_e32 vcc, v68, v68
	v_bfe_u32 v102, v66, 16, 1
	v_add3_u32 v75, v67, v75, s0
	v_cndmask_b32_e32 v68, v100, v104, vcc
	v_bfe_u32 v103, v65, 16, 1
	v_add3_u32 v102, v66, v102, s0
	v_lshrrev_b32_e32 v75, 16, v75
	v_cmp_o_f32_e32 vcc, v67, v67
	v_bfe_u32 v104, v64, 16, 1
	v_add3_u32 v103, v65, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cndmask_b32_e32 v67, v100, v75, vcc
	v_cmp_o_f32_e32 vcc, v66, v66
	v_add3_u32 v104, v64, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v66, v100, v102, vcc
	v_cmp_o_f32_e32 vcc, v65, v65
	v_bfe_u32 v75, v63, 16, 1
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v65, v100, v103, vcc
	v_cmp_o_f32_e32 vcc, v64, v64
	v_bfe_u32 v102, v62, 16, 1
	v_add3_u32 v75, v63, v75, s0
	v_cndmask_b32_e32 v64, v100, v104, vcc
	v_bfe_u32 v103, v61, 16, 1
	v_add3_u32 v102, v62, v102, s0
	v_lshrrev_b32_e32 v75, 16, v75
	v_cmp_o_f32_e32 vcc, v63, v63
	v_bfe_u32 v104, v60, 16, 1
	v_add3_u32 v103, v61, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cndmask_b32_e32 v63, v100, v75, vcc
	v_cmp_o_f32_e32 vcc, v62, v62
	v_add3_u32 v104, v60, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v62, v100, v102, vcc
	v_cmp_o_f32_e32 vcc, v61, v61
	v_bfe_u32 v75, v59, 16, 1
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v61, v100, v103, vcc
	v_cmp_o_f32_e32 vcc, v60, v60
	v_bfe_u32 v102, v58, 16, 1
	v_add3_u32 v75, v59, v75, s0
	v_cndmask_b32_e32 v60, v100, v104, vcc
	v_bfe_u32 v103, v57, 16, 1
	v_add3_u32 v102, v58, v102, s0
	v_lshrrev_b32_e32 v75, 16, v75
	v_cmp_o_f32_e32 vcc, v59, v59
	v_bfe_u32 v104, v56, 16, 1
	v_add3_u32 v103, v57, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cndmask_b32_e32 v59, v100, v75, vcc
	v_cmp_o_f32_e32 vcc, v58, v58
	v_add3_u32 v104, v56, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v58, v100, v102, vcc
	v_cmp_o_f32_e32 vcc, v57, v57
	v_bfe_u32 v75, v55, 16, 1
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v57, v100, v103, vcc
	v_cmp_o_f32_e32 vcc, v56, v56
	v_bfe_u32 v102, v54, 16, 1
	v_add3_u32 v75, v55, v75, s0
	v_cndmask_b32_e32 v56, v100, v104, vcc
	v_bfe_u32 v103, v53, 16, 1
	v_add3_u32 v102, v54, v102, s0
	v_lshrrev_b32_e32 v75, 16, v75
	v_cmp_o_f32_e32 vcc, v55, v55
	v_bfe_u32 v104, v52, 16, 1
	v_add3_u32 v103, v53, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cndmask_b32_e32 v55, v100, v75, vcc
	v_cmp_o_f32_e32 vcc, v54, v54
	v_add3_u32 v104, v52, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v54, v100, v102, vcc
	v_cmp_o_f32_e32 vcc, v53, v53
	v_bfe_u32 v75, v51, 16, 1
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v53, v100, v103, vcc
	v_cmp_o_f32_e32 vcc, v52, v52
	v_bfe_u32 v102, v50, 16, 1
	v_add3_u32 v75, v51, v75, s0
	v_cndmask_b32_e32 v52, v100, v104, vcc
	v_bfe_u32 v103, v49, 16, 1
	v_add3_u32 v102, v50, v102, s0
	v_lshrrev_b32_e32 v75, 16, v75
	v_cmp_o_f32_e32 vcc, v51, v51
	v_bfe_u32 v104, v48, 16, 1
	v_add3_u32 v103, v49, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cndmask_b32_e32 v51, v100, v75, vcc
	v_cmp_o_f32_e32 vcc, v50, v50
	v_add3_u32 v104, v48, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v50, v100, v102, vcc
	v_cmp_o_f32_e32 vcc, v49, v49
	v_bfe_u32 v75, v47, 16, 1
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v49, v100, v103, vcc
	v_cmp_o_f32_e32 vcc, v48, v48
	v_bfe_u32 v102, v46, 16, 1
	v_add3_u32 v75, v47, v75, s0
	v_cndmask_b32_e32 v48, v100, v104, vcc
	v_bfe_u32 v103, v45, 16, 1
	v_add3_u32 v102, v46, v102, s0
	v_lshrrev_b32_e32 v75, 16, v75
	v_cmp_o_f32_e32 vcc, v47, v47
	v_bfe_u32 v104, v44, 16, 1
	v_add3_u32 v103, v45, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cndmask_b32_e32 v47, v100, v75, vcc
	v_cmp_o_f32_e32 vcc, v46, v46
	v_add3_u32 v104, v44, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v46, v100, v102, vcc
	v_cmp_o_f32_e32 vcc, v45, v45
	v_bfe_u32 v75, v43, 16, 1
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v45, v100, v103, vcc
	v_cmp_o_f32_e32 vcc, v44, v44
	v_bfe_u32 v102, v42, 16, 1
	v_add3_u32 v75, v43, v75, s0
	v_cndmask_b32_e32 v44, v100, v104, vcc
	v_bfe_u32 v103, v41, 16, 1
	v_add3_u32 v102, v42, v102, s0
	v_lshrrev_b32_e32 v75, 16, v75
	v_cmp_o_f32_e32 vcc, v43, v43
	v_bfe_u32 v104, v40, 16, 1
	v_add3_u32 v103, v41, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cndmask_b32_e32 v43, v100, v75, vcc
	v_cmp_o_f32_e32 vcc, v42, v42
	v_add3_u32 v104, v40, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v42, v100, v102, vcc
	v_cmp_o_f32_e32 vcc, v41, v41
	v_bfe_u32 v75, v31, 16, 1
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v41, v100, v103, vcc
	v_cmp_o_f32_e32 vcc, v40, v40
	v_bfe_u32 v102, v30, 16, 1
	v_add3_u32 v75, v31, v75, s0
	v_cndmask_b32_e32 v40, v100, v104, vcc
	v_bfe_u32 v103, v29, 16, 1
	v_add3_u32 v102, v30, v102, s0
	v_lshrrev_b32_e32 v75, 16, v75
	v_cmp_o_f32_e32 vcc, v31, v31
	v_bfe_u32 v104, v28, 16, 1
	v_add3_u32 v103, v29, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cndmask_b32_e32 v31, v100, v75, vcc
	v_cmp_o_f32_e32 vcc, v30, v30
	v_add3_u32 v104, v28, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v30, v100, v102, vcc
	v_cmp_o_f32_e32 vcc, v29, v29
	v_bfe_u32 v75, v23, 16, 1
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v29, v100, v103, vcc
	v_cmp_o_f32_e32 vcc, v28, v28
	v_bfe_u32 v102, v22, 16, 1
	v_add3_u32 v75, v23, v75, s0
	v_cndmask_b32_e32 v28, v100, v104, vcc
	v_bfe_u32 v103, v21, 16, 1
	v_add3_u32 v102, v22, v102, s0
	v_lshrrev_b32_e32 v75, 16, v75
	v_cmp_o_f32_e32 vcc, v23, v23
	v_bfe_u32 v104, v20, 16, 1
	v_add3_u32 v103, v21, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cndmask_b32_e32 v23, v100, v75, vcc
	v_cmp_o_f32_e32 vcc, v22, v22
	v_add3_u32 v104, v20, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v22, v100, v102, vcc
	v_cmp_o_f32_e32 vcc, v21, v21
	v_bfe_u32 v75, v19, 16, 1
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v21, v100, v103, vcc
	v_cmp_o_f32_e32 vcc, v20, v20
	v_bfe_u32 v102, v18, 16, 1
	v_add3_u32 v75, v19, v75, s0
	v_cndmask_b32_e32 v20, v100, v104, vcc
	v_bfe_u32 v103, v17, 16, 1
	v_add3_u32 v102, v18, v102, s0
	v_lshrrev_b32_e32 v75, 16, v75
	v_cmp_o_f32_e32 vcc, v19, v19
	v_bfe_u32 v104, v16, 16, 1
	v_add3_u32 v103, v17, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cndmask_b32_e32 v19, v100, v75, vcc
	v_cmp_o_f32_e32 vcc, v18, v18
	v_add3_u32 v104, v16, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v18, v100, v102, vcc
	v_cmp_o_f32_e32 vcc, v17, v17
	v_bfe_u32 v75, v15, 16, 1
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v17, v100, v103, vcc
	v_cmp_o_f32_e32 vcc, v16, v16
	v_bfe_u32 v102, v14, 16, 1
	v_add3_u32 v75, v15, v75, s0
	v_cndmask_b32_e32 v16, v100, v104, vcc
	v_bfe_u32 v103, v13, 16, 1
	v_add3_u32 v102, v14, v102, s0
	v_lshrrev_b32_e32 v75, 16, v75
	v_cmp_o_f32_e32 vcc, v15, v15
	v_bfe_u32 v104, v12, 16, 1
	v_add3_u32 v103, v13, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cndmask_b32_e32 v15, v100, v75, vcc
	v_cmp_o_f32_e32 vcc, v14, v14
	v_add3_u32 v104, v12, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v14, v100, v102, vcc
	v_cmp_o_f32_e32 vcc, v13, v13
	v_bfe_u32 v75, v11, 16, 1
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v13, v100, v103, vcc
	v_cmp_o_f32_e32 vcc, v12, v12
	v_bfe_u32 v102, v10, 16, 1
	v_add3_u32 v75, v11, v75, s0
	v_cndmask_b32_e32 v12, v100, v104, vcc
	v_bfe_u32 v103, v9, 16, 1
	v_add3_u32 v102, v10, v102, s0
	v_lshrrev_b32_e32 v75, 16, v75
	v_cmp_o_f32_e32 vcc, v11, v11
	v_bfe_u32 v104, v8, 16, 1
	v_add3_u32 v103, v9, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cndmask_b32_e32 v11, v100, v75, vcc
	v_cmp_o_f32_e32 vcc, v10, v10
	v_add3_u32 v104, v8, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v10, v100, v102, vcc
	v_cmp_o_f32_e32 vcc, v9, v9
	v_bfe_u32 v75, v7, 16, 1
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v9, v100, v103, vcc
	v_cmp_o_f32_e32 vcc, v8, v8
	v_bfe_u32 v102, v6, 16, 1
	v_add3_u32 v75, v7, v75, s0
	v_cndmask_b32_e32 v8, v100, v104, vcc
	v_bfe_u32 v103, v5, 16, 1
	v_add3_u32 v102, v6, v102, s0
	v_lshrrev_b32_e32 v75, 16, v75
	v_cmp_o_f32_e32 vcc, v7, v7
	v_bfe_u32 v104, v4, 16, 1
	v_add3_u32 v103, v5, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cndmask_b32_e32 v7, v100, v75, vcc
	v_cmp_o_f32_e32 vcc, v6, v6
	v_add3_u32 v104, v4, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v6, v100, v102, vcc
	v_cmp_o_f32_e32 vcc, v5, v5
	v_lshrrev_b32_e32 v104, 16, v104
	v_bfe_u32 v75, v3, 16, 1
	v_cndmask_b32_e32 v5, v100, v103, vcc
	v_cmp_o_f32_e32 vcc, v4, v4
	v_bfe_u32 v102, v2, 16, 1
	v_bfe_u32 v103, v1, 16, 1
	v_cndmask_b32_e32 v4, v100, v104, vcc
	v_bfe_u32 v104, v0, 16, 1
	v_add3_u32 v104, v0, v104, s0
	v_add3_u32 v103, v1, v103, s0
	v_add3_u32 v102, v2, v102, s0
	v_add3_u32 v75, v3, v75, s0
	s_mul_i32 s0, s13, s16
	s_add_i32 s1, s1, s0
	s_mul_i32 s0, s12, s16
	s_lshl_b64 s[0:1], s[0:1], 1
	v_lshrrev_b32_e32 v75, 16, v75
	v_cmp_o_f32_e32 vcc, v3, v3
	s_add_u32 s0, s10, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cndmask_b32_e32 v3, v100, v75, vcc
	v_cmp_o_f32_e32 vcc, v2, v2
	v_and_b32_e32 v75, 15, v96
	s_addc_u32 s1, s11, s1
	s_lshl_b32 s3, s33, 1
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v2, v100, v102, vcc
	v_cmp_o_f32_e32 vcc, v1, v1
	v_or_b32_e32 v96, s33, v75
	s_add_u32 s8, s0, s3
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v1, v100, v103, vcc
	v_cmp_o_f32_e32 vcc, v0, v0
	v_add_u32_e32 v96, v96, v99
	v_or_b32_e32 v75, v75, v99
	s_addc_u32 s0, s1, 0
	s_and_b32 s1, s12, 0x3fff
	v_cndmask_b32_e32 v0, v100, v104, vcc
	s_lshl_b32 s1, s1, 16
	s_and_b32 s0, s0, 0xffff
	v_add_u32_e32 v99, v98, v75
	v_not_b32_e32 v100, -2.0
	v_cmp_gt_u32_e32 vcc, s2, v96
	s_or_b32 s0, s1, s0
	s_or_b32 s9, s0, 2.0
	v_cndmask_b32_e32 v99, v100, v99, vcc
	s_mov_b32 s11, 0x27000
	s_mov_b32 s10, 0x7ffffffd
	v_lshlrev_b32_e32 v99, 1, v99
	buffer_store_short v86, v99, s[8:11], 0 offen
	v_or_b32_e32 v86, 1, v97
	v_mul_lo_u32 v86, s12, v86
	v_add_u32_e32 v99, v86, v75
	v_cndmask_b32_e32 v99, v100, v99, vcc
	v_lshlrev_b32_e32 v99, 1, v99
	buffer_store_short v85, v99, s[8:11], 0 offen
	v_or_b32_e32 v85, 2, v97
	v_mul_lo_u32 v85, s12, v85
	v_add_u32_e32 v99, v85, v75
	v_cndmask_b32_e32 v99, v100, v99, vcc
	v_lshlrev_b32_e32 v99, 1, v99
	buffer_store_short v84, v99, s[8:11], 0 offen
	v_or_b32_e32 v84, 3, v97
	v_mul_lo_u32 v84, s12, v84
	v_add_u32_e32 v99, v84, v75
	v_cndmask_b32_e32 v99, v100, v99, vcc
	v_lshlrev_b32_e32 v99, 1, v99
	buffer_store_short v88, v99, s[8:11], 0 offen
	v_or_b32_e32 v88, 16, v75
	v_add_u32_e32 v99, v98, v88
	v_cndmask_b32_e32 v99, v100, v99, vcc
	v_lshlrev_b32_e32 v99, 1, v99
	buffer_store_short v89, v99, s[8:11], 0 offen
	v_add_u32_e32 v89, v86, v88
	v_cndmask_b32_e32 v89, v100, v89, vcc
	v_lshlrev_b32_e32 v89, 1, v89
	buffer_store_short v80, v89, s[8:11], 0 offen
	v_add_u32_e32 v80, v85, v88
	v_cndmask_b32_e32 v80, v100, v80, vcc
	v_lshlrev_b32_e32 v80, 1, v80
	buffer_store_short v78, v80, s[8:11], 0 offen
	v_add_u32_e32 v78, v84, v88
	v_cndmask_b32_e32 v78, v100, v78, vcc
	v_lshlrev_b32_e32 v78, 1, v78
	buffer_store_short v76, v78, s[8:11], 0 offen
	s_movk_i32 s0, 0x6e0
	v_add_u32_e32 v76, 32, v75
	v_add_u32_e32 v78, v98, v76
	v_cmp_gt_u32_e64 s[0:1], s0, v96
	s_movk_i32 s2, 0x6d0
	v_cmp_gt_u32_e64 s[2:3], s2, v96
	v_cndmask_b32_e64 v78, v100, v78, s[0:1]
	v_lshlrev_b32_e32 v78, 1, v78
	buffer_store_short v92, v78, s[8:11], 0 offen
	v_add_u32_e32 v78, v86, v76
	v_cndmask_b32_e64 v78, v100, v78, s[0:1]
	v_lshlrev_b32_e32 v78, 1, v78
	buffer_store_short v83, v78, s[8:11], 0 offen
	v_add_u32_e32 v78, v85, v76
	v_cndmask_b32_e64 v78, v100, v78, s[0:1]
	v_lshlrev_b32_e32 v78, 1, v78
	buffer_store_short v81, v78, s[8:11], 0 offen
	v_add_u32_e32 v78, v84, v76
	v_cndmask_b32_e64 v78, v100, v78, s[0:1]
	v_lshlrev_b32_e32 v78, 1, v78
	buffer_store_short v77, v78, s[8:11], 0 offen
	v_add_u32_e32 v77, 48, v75
	v_add_u32_e32 v78, v98, v77
	v_cndmask_b32_e64 v78, v100, v78, s[2:3]
	v_lshlrev_b32_e32 v78, 1, v78
	buffer_store_short v93, v78, s[8:11], 0 offen
	v_add_u32_e32 v78, v86, v77
	v_cndmask_b32_e64 v78, v100, v78, s[2:3]
	v_lshlrev_b32_e32 v78, 1, v78
	buffer_store_short v90, v78, s[8:11], 0 offen
	v_add_u32_e32 v78, v85, v77
	v_cndmask_b32_e64 v78, v100, v78, s[2:3]
	v_lshlrev_b32_e32 v78, 1, v78
	buffer_store_short v87, v78, s[8:11], 0 offen
	v_add_u32_e32 v78, v84, v77
	v_cndmask_b32_e64 v78, v100, v78, s[2:3]
	v_lshlrev_b32_e32 v78, 1, v78
	buffer_store_short v79, v78, s[8:11], 0 offen
	v_add_u32_e32 v78, 64, v75
	v_add_u32_e32 v79, v98, v78
	v_cmp_gt_u32_e64 s[4:5], s4, v96
	v_cmp_gt_u32_e64 s[6:7], s6, v96
	s_nop 0
	v_cndmask_b32_e64 v79, v100, v79, s[4:5]
	v_lshlrev_b32_e32 v79, 1, v79
	buffer_store_short v34, v79, s[8:11], 0 offen
	v_add_u32_e32 v34, v86, v78
	v_cndmask_b32_e64 v34, v100, v34, s[4:5]
	v_lshlrev_b32_e32 v34, 1, v34
	buffer_store_short v33, v34, s[8:11], 0 offen
	v_add_u32_e32 v33, v85, v78
	v_cndmask_b32_e64 v33, v100, v33, s[4:5]
	v_lshlrev_b32_e32 v33, 1, v33
	buffer_store_short v91, v33, s[8:11], 0 offen
	v_add_u32_e32 v33, v84, v78
	v_cndmask_b32_e64 v33, v100, v33, s[4:5]
	v_lshlrev_b32_e32 v33, 1, v33
	buffer_store_short v82, v33, s[8:11], 0 offen
	v_add_u32_e32 v33, 0x50, v75
	v_add_u32_e32 v34, v98, v33
	v_cndmask_b32_e64 v34, v100, v34, s[6:7]
	v_lshlrev_b32_e32 v34, 1, v34
	buffer_store_short v37, v34, s[8:11], 0 offen
	v_add_u32_e32 v34, v86, v33
	v_cndmask_b32_e64 v34, v100, v34, s[6:7]
	v_lshlrev_b32_e32 v34, 1, v34
	buffer_store_short v36, v34, s[8:11], 0 offen
	v_add_u32_e32 v34, v85, v33
	v_cndmask_b32_e64 v34, v100, v34, s[6:7]
	v_lshlrev_b32_e32 v34, 1, v34
	buffer_store_short v35, v34, s[8:11], 0 offen
	v_add_u32_e32 v34, v84, v33
	v_cndmask_b32_e64 v34, v100, v34, s[6:7]
	v_lshlrev_b32_e32 v34, 1, v34
	buffer_store_short v25, v34, s[8:11], 0 offen
	v_or_b32_e32 v25, 16, v97
	v_mul_lo_u32 v25, s12, v25
	v_add_u32_e32 v34, v25, v75
	v_cndmask_b32_e32 v34, v100, v34, vcc
	v_lshlrev_b32_e32 v34, 1, v34
	buffer_store_short v39, v34, s[8:11], 0 offen
	v_or_b32_e32 v34, 17, v97
	v_mul_lo_u32 v34, s12, v34
	v_add_u32_e32 v35, v34, v75
	v_cndmask_b32_e32 v35, v100, v35, vcc
	v_lshlrev_b32_e32 v35, 1, v35
	buffer_store_short v38, v35, s[8:11], 0 offen
	v_or_b32_e32 v35, 18, v97
	v_mul_lo_u32 v35, s12, v35
	v_add_u32_e32 v36, v35, v75
	v_cndmask_b32_e32 v36, v100, v36, vcc
	v_lshlrev_b32_e32 v36, 1, v36
	buffer_store_short v27, v36, s[8:11], 0 offen
	v_or_b32_e32 v27, 19, v97
	v_mul_lo_u32 v27, s12, v27
	v_add_u32_e32 v36, v27, v75
	v_cndmask_b32_e32 v36, v100, v36, vcc
	v_lshlrev_b32_e32 v36, 1, v36
	buffer_store_short v24, v36, s[8:11], 0 offen
	v_add_u32_e32 v24, v25, v88
	v_cndmask_b32_e32 v24, v100, v24, vcc
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v95, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v34, v88
	v_cndmask_b32_e32 v24, v100, v24, vcc
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v94, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v35, v88
	v_cndmask_b32_e32 v24, v100, v24, vcc
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v101, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v27, v88
	v_cndmask_b32_e32 v24, v100, v24, vcc
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v26, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v25, v76
	v_cndmask_b32_e64 v24, v100, v24, s[0:1]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v72, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v34, v76
	v_cndmask_b32_e64 v24, v100, v24, s[0:1]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v73, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v35, v76
	v_cndmask_b32_e64 v24, v100, v24, s[0:1]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v74, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v27, v76
	v_cndmask_b32_e64 v24, v100, v24, s[0:1]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v32, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v25, v77
	v_cndmask_b32_e64 v24, v100, v24, s[2:3]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v68, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v34, v77
	v_cndmask_b32_e64 v24, v100, v24, s[2:3]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v69, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v35, v77
	v_cndmask_b32_e64 v24, v100, v24, s[2:3]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v70, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v27, v77
	v_cndmask_b32_e64 v24, v100, v24, s[2:3]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v71, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v25, v78
	v_cndmask_b32_e64 v24, v100, v24, s[4:5]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v64, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v34, v78
	v_cndmask_b32_e64 v24, v100, v24, s[4:5]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v65, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v35, v78
	v_cndmask_b32_e64 v24, v100, v24, s[4:5]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v66, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v27, v78
	v_cndmask_b32_e64 v24, v100, v24, s[4:5]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v67, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v25, v33
	v_cndmask_b32_e64 v24, v100, v24, s[6:7]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v60, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v34, v33
	v_cndmask_b32_e64 v24, v100, v24, s[6:7]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v61, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v35, v33
	v_cndmask_b32_e64 v24, v100, v24, s[6:7]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v62, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v27, v33
	v_cndmask_b32_e64 v24, v100, v24, s[6:7]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v63, v24, s[8:11], 0 offen
	v_or_b32_e32 v24, 32, v97
	v_mul_lo_u32 v24, s12, v24
	v_add_u32_e32 v25, v24, v75
	v_cndmask_b32_e32 v25, v100, v25, vcc
	v_lshlrev_b32_e32 v25, 1, v25
	buffer_store_short v56, v25, s[8:11], 0 offen
	v_or_b32_e32 v25, 33, v97
	v_mul_lo_u32 v25, s12, v25
	v_add_u32_e32 v26, v25, v75
	v_cndmask_b32_e32 v26, v100, v26, vcc
	v_lshlrev_b32_e32 v26, 1, v26
	buffer_store_short v57, v26, s[8:11], 0 offen
	v_or_b32_e32 v26, 34, v97
	v_mul_lo_u32 v26, s12, v26
	v_add_u32_e32 v27, v26, v75
	v_cndmask_b32_e32 v27, v100, v27, vcc
	v_lshlrev_b32_e32 v27, 1, v27
	buffer_store_short v58, v27, s[8:11], 0 offen
	v_or_b32_e32 v27, 35, v97
	v_mul_lo_u32 v27, s12, v27
	v_add_u32_e32 v32, v27, v75
	v_cndmask_b32_e32 v32, v100, v32, vcc
	v_lshlrev_b32_e32 v32, 1, v32
	buffer_store_short v59, v32, s[8:11], 0 offen
	v_add_u32_e32 v32, v24, v88
	v_cndmask_b32_e32 v32, v100, v32, vcc
	v_lshlrev_b32_e32 v32, 1, v32
	buffer_store_short v52, v32, s[8:11], 0 offen
	v_add_u32_e32 v32, v25, v88
	v_cndmask_b32_e32 v32, v100, v32, vcc
	v_lshlrev_b32_e32 v32, 1, v32
	buffer_store_short v53, v32, s[8:11], 0 offen
	v_add_u32_e32 v32, v26, v88
	v_cndmask_b32_e32 v32, v100, v32, vcc
	v_lshlrev_b32_e32 v32, 1, v32
	buffer_store_short v54, v32, s[8:11], 0 offen
	v_add_u32_e32 v32, v27, v88
	v_cndmask_b32_e32 v32, v100, v32, vcc
	v_lshlrev_b32_e32 v32, 1, v32
	buffer_store_short v55, v32, s[8:11], 0 offen
	v_add_u32_e32 v32, v24, v76
	v_cndmask_b32_e64 v32, v100, v32, s[0:1]
	v_lshlrev_b32_e32 v32, 1, v32
	buffer_store_short v48, v32, s[8:11], 0 offen
	v_add_u32_e32 v32, v25, v76
	v_cndmask_b32_e64 v32, v100, v32, s[0:1]
	v_lshlrev_b32_e32 v32, 1, v32
	buffer_store_short v49, v32, s[8:11], 0 offen
	v_add_u32_e32 v32, v26, v76
	v_cndmask_b32_e64 v32, v100, v32, s[0:1]
	v_lshlrev_b32_e32 v32, 1, v32
	buffer_store_short v50, v32, s[8:11], 0 offen
	v_add_u32_e32 v32, v27, v76
	v_cndmask_b32_e64 v32, v100, v32, s[0:1]
	v_lshlrev_b32_e32 v32, 1, v32
	buffer_store_short v51, v32, s[8:11], 0 offen
	v_add_u32_e32 v32, v24, v77
	v_cndmask_b32_e64 v32, v100, v32, s[2:3]
	v_lshlrev_b32_e32 v32, 1, v32
	buffer_store_short v44, v32, s[8:11], 0 offen
	v_add_u32_e32 v32, v25, v77
	v_cndmask_b32_e64 v32, v100, v32, s[2:3]
	v_lshlrev_b32_e32 v32, 1, v32
	buffer_store_short v45, v32, s[8:11], 0 offen
	v_add_u32_e32 v32, v26, v77
	v_cndmask_b32_e64 v32, v100, v32, s[2:3]
	v_lshlrev_b32_e32 v32, 1, v32
	buffer_store_short v46, v32, s[8:11], 0 offen
	v_add_u32_e32 v32, v27, v77
	v_cndmask_b32_e64 v32, v100, v32, s[2:3]
	v_lshlrev_b32_e32 v32, 1, v32
	buffer_store_short v47, v32, s[8:11], 0 offen
	v_add_u32_e32 v32, v24, v78
	v_cndmask_b32_e64 v32, v100, v32, s[4:5]
	v_lshlrev_b32_e32 v32, 1, v32
	buffer_store_short v40, v32, s[8:11], 0 offen
	v_add_u32_e32 v32, v25, v78
	v_cndmask_b32_e64 v32, v100, v32, s[4:5]
	v_lshlrev_b32_e32 v32, 1, v32
	buffer_store_short v41, v32, s[8:11], 0 offen
	v_add_u32_e32 v32, v26, v78
	v_cndmask_b32_e64 v32, v100, v32, s[4:5]
	v_lshlrev_b32_e32 v32, 1, v32
	buffer_store_short v42, v32, s[8:11], 0 offen
	v_add_u32_e32 v32, v27, v78
	v_add_u32_e32 v24, v24, v33
	v_cndmask_b32_e64 v32, v100, v32, s[4:5]
	v_cndmask_b32_e64 v24, v100, v24, s[6:7]
	v_lshlrev_b32_e32 v32, 1, v32
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v43, v32, s[8:11], 0 offen
	buffer_store_short v28, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v25, v33
	v_cndmask_b32_e64 v24, v100, v24, s[6:7]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v29, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v26, v33
	v_cndmask_b32_e64 v24, v100, v24, s[6:7]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v30, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v27, v33
	v_cndmask_b32_e64 v24, v100, v24, s[6:7]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v31, v24, s[8:11], 0 offen
	v_or_b32_e32 v24, 48, v97
	v_mul_lo_u32 v24, s12, v24
	v_add_u32_e32 v25, v24, v75
	v_cndmask_b32_e32 v25, v100, v25, vcc
	v_lshlrev_b32_e32 v25, 1, v25
	buffer_store_short v20, v25, s[8:11], 0 offen
	v_or_b32_e32 v20, 49, v97
	v_mul_lo_u32 v20, s12, v20
	v_add_u32_e32 v25, v20, v75
	v_cndmask_b32_e32 v25, v100, v25, vcc
	v_lshlrev_b32_e32 v25, 1, v25
	buffer_store_short v21, v25, s[8:11], 0 offen
	v_or_b32_e32 v21, 50, v97
	v_mul_lo_u32 v21, s12, v21
	v_add_u32_e32 v25, v21, v75
	v_cndmask_b32_e32 v25, v100, v25, vcc
	v_lshlrev_b32_e32 v25, 1, v25
	buffer_store_short v22, v25, s[8:11], 0 offen
	v_or_b32_e32 v22, 51, v97
	v_mul_lo_u32 v22, s12, v22
	v_add_u32_e32 v25, v22, v75
	v_cndmask_b32_e32 v25, v100, v25, vcc
	v_lshlrev_b32_e32 v25, 1, v25
	buffer_store_short v23, v25, s[8:11], 0 offen
	v_add_u32_e32 v23, v24, v88
	v_cndmask_b32_e32 v23, v100, v23, vcc
	v_lshlrev_b32_e32 v23, 1, v23
	buffer_store_short v16, v23, s[8:11], 0 offen
	v_add_u32_e32 v16, v20, v88
	v_cndmask_b32_e32 v16, v100, v16, vcc
	v_lshlrev_b32_e32 v16, 1, v16
	buffer_store_short v17, v16, s[8:11], 0 offen
	v_add_u32_e32 v16, v21, v88
	v_cndmask_b32_e32 v16, v100, v16, vcc
	v_lshlrev_b32_e32 v16, 1, v16
	buffer_store_short v18, v16, s[8:11], 0 offen
	v_add_u32_e32 v16, v22, v88
	v_cndmask_b32_e32 v16, v100, v16, vcc
	v_lshlrev_b32_e32 v16, 1, v16
	buffer_store_short v19, v16, s[8:11], 0 offen
	v_add_u32_e32 v16, v24, v76
	v_cndmask_b32_e64 v16, v100, v16, s[0:1]
	v_lshlrev_b32_e32 v16, 1, v16
	buffer_store_short v12, v16, s[8:11], 0 offen
	v_add_u32_e32 v12, v20, v76
	v_cndmask_b32_e64 v12, v100, v12, s[0:1]
	v_lshlrev_b32_e32 v12, 1, v12
	buffer_store_short v13, v12, s[8:11], 0 offen
	v_add_u32_e32 v12, v21, v76
	v_cndmask_b32_e64 v12, v100, v12, s[0:1]
	v_lshlrev_b32_e32 v12, 1, v12
	buffer_store_short v14, v12, s[8:11], 0 offen
	v_add_u32_e32 v12, v22, v76
	v_cndmask_b32_e64 v12, v100, v12, s[0:1]
	v_lshlrev_b32_e32 v12, 1, v12
	buffer_store_short v15, v12, s[8:11], 0 offen
	v_add_u32_e32 v12, v24, v77
	v_cndmask_b32_e64 v12, v100, v12, s[2:3]
	v_lshlrev_b32_e32 v12, 1, v12
	buffer_store_short v8, v12, s[8:11], 0 offen
	v_add_u32_e32 v8, v20, v77
	v_cndmask_b32_e64 v8, v100, v8, s[2:3]
	v_lshlrev_b32_e32 v8, 1, v8
	buffer_store_short v9, v8, s[8:11], 0 offen
	v_add_u32_e32 v8, v21, v77
	v_cndmask_b32_e64 v8, v100, v8, s[2:3]
	v_lshlrev_b32_e32 v8, 1, v8
	buffer_store_short v10, v8, s[8:11], 0 offen
	v_add_u32_e32 v8, v22, v77
	v_cndmask_b32_e64 v8, v100, v8, s[2:3]
	v_lshlrev_b32_e32 v8, 1, v8
	buffer_store_short v11, v8, s[8:11], 0 offen
	v_add_u32_e32 v8, v24, v78
	v_cndmask_b32_e64 v8, v100, v8, s[4:5]
	v_lshlrev_b32_e32 v8, 1, v8
	buffer_store_short v4, v8, s[8:11], 0 offen
	v_add_u32_e32 v4, v20, v78
	v_cndmask_b32_e64 v4, v100, v4, s[4:5]
	v_lshlrev_b32_e32 v4, 1, v4
	buffer_store_short v5, v4, s[8:11], 0 offen
	v_add_u32_e32 v4, v21, v78
	v_cndmask_b32_e64 v4, v100, v4, s[4:5]
	v_lshlrev_b32_e32 v4, 1, v4
	buffer_store_short v6, v4, s[8:11], 0 offen
	v_add_u32_e32 v4, v22, v78
	v_cndmask_b32_e64 v4, v100, v4, s[4:5]
	v_lshlrev_b32_e32 v4, 1, v4
	buffer_store_short v7, v4, s[8:11], 0 offen
	v_add_u32_e32 v4, v24, v33
	v_cndmask_b32_e64 v4, v100, v4, s[6:7]
	v_lshlrev_b32_e32 v4, 1, v4
	buffer_store_short v0, v4, s[8:11], 0 offen
	v_add_u32_e32 v0, v20, v33
	v_cndmask_b32_e64 v0, v100, v0, s[6:7]
	v_lshlrev_b32_e32 v0, 1, v0
	buffer_store_short v1, v0, s[8:11], 0 offen
	v_add_u32_e32 v0, v21, v33
	v_cndmask_b32_e64 v0, v100, v0, s[6:7]
	v_lshlrev_b32_e32 v0, 1, v0
	buffer_store_short v2, v0, s[8:11], 0 offen
	v_add_u32_e32 v0, v22, v33
	v_cndmask_b32_e64 v0, v100, v0, s[6:7]
	v_lshlrev_b32_e32 v0, 1, v0
	buffer_store_short v3, v0, s[8:11], 0 offen
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wave_mxfp4_static_gemm_256x192x256_5376x1792x4096
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
	.size	wave_mxfp4_static_gemm_256x192x256_5376x1792x4096, .Lfunc_end0-wave_mxfp4_static_gemm_256x192x256_5376x1792x4096

	.set wave_mxfp4_static_gemm_256x192x256_5376x1792x4096.num_vgpr, 206
	.set wave_mxfp4_static_gemm_256x192x256_5376x1792x4096.num_agpr, 0
	.set wave_mxfp4_static_gemm_256x192x256_5376x1792x4096.numbered_sgpr, 42
	.set wave_mxfp4_static_gemm_256x192x256_5376x1792x4096.num_named_barrier, 0
	.set wave_mxfp4_static_gemm_256x192x256_5376x1792x4096.private_seg_size, 0
	.set wave_mxfp4_static_gemm_256x192x256_5376x1792x4096.uses_vcc, 1
	.set wave_mxfp4_static_gemm_256x192x256_5376x1792x4096.uses_flat_scratch, 0
	.set wave_mxfp4_static_gemm_256x192x256_5376x1792x4096.has_dyn_sized_stack, 0
	.set wave_mxfp4_static_gemm_256x192x256_5376x1792x4096.has_recursion, 0
	.set wave_mxfp4_static_gemm_256x192x256_5376x1792x4096.has_indirect_call, 0
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
    .name:           wave_mxfp4_static_gemm_256x192x256_5376x1792x4096
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 256
      - 2
      - 1
    .sgpr_count:     48
    .sgpr_spill_count: 0
    .symbol:         wave_mxfp4_static_gemm_256x192x256_5376x1792x4096.kd
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
