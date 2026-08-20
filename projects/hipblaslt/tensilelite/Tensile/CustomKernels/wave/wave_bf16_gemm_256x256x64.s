; To reproduce the .rocmasm from .optimized.ll, run:
; llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -mattr='-fma-mix-insts' -O3 <.optimized.ll> -o <out.rocmasm>

	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 5
	.text
	.globl	wave_bf16_gemm_256x256x64
	.p2align	8
	.type	wave_bf16_gemm_256x256x64,@function
wave_bf16_gemm_256x256x64:
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx2 s[12:13], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
.LBB0_0:
	s_mov_b32 s0, s15
	s_mov_b32 s15, 0
	s_mov_b32 s1, s15
	v_and_b32_e32 v90, 0x3ff, v0
	v_bfe_u32 v0, v0, 10, 10
	s_lshl_b64 s[36:37], s[14:15], 8
	s_lshl_b64 s[38:39], s[0:1], 8
	v_lshlrev_b32_e32 v1, 6, v0
	s_cmp_eq_u64 s[12:13], 0
	v_and_or_b32 v71, v90, 15, v1
	v_bfe_u32 v70, v90, 4, 2
	v_mov_b32_e32 v2, 0
	s_cbranch_scc1 .LBB0_68
	s_add_u32 s40, s12, -7
	s_addc_u32 s41, s13, -1
	s_add_u32 s42, s12, -6
	s_addc_u32 s43, s13, -1
	s_add_u32 s44, s12, -5
	s_addc_u32 s45, s13, -1
	s_add_u32 s46, s12, -4
	v_lshrrev_b32_e32 v3, 3, v90
	s_addc_u32 s47, s13, -1
	v_lshl_or_b32 v0, v0, 5, v3
	s_add_u32 s48, s12, -3
	v_or_b32_e32 v4, s36, v0
	v_mov_b32_e32 v5, s37
	s_addc_u32 s49, s13, -1
	v_cmp_gt_u64_e32 vcc, s[8:9], v[4:5]
	v_lshlrev_b32_e32 v5, 3, v90
	v_lshlrev_b32_e32 v3, 6, v3
	s_add_u32 s50, s12, -2
	v_sub_u32_e32 v74, v5, v3
	v_or_b32_e32 v4, 0x80, v4
	v_mov_b32_e32 v5, s37
	s_addc_u32 s51, s13, -1
	v_cmp_gt_u64_e64 s[0:1], s[8:9], v[4:5]
	v_or_b32_e32 v4, s38, v0
	v_mov_b32_e32 v5, s39
	s_add_u32 s52, s12, -1
	v_cmp_gt_u64_e64 s[30:31], s[10:11], v[4:5]
	v_or_b32_e32 v4, 0x80, v4
	s_addc_u32 s53, s13, -1
	v_cmp_gt_u64_e64 s[34:35], s[10:11], v[4:5]
	v_mul_u32_u24_e32 v5, 0x44, v0
	s_add_u32 s16, 0, s12
	v_or_b32_e32 v3, 48, v90
	v_and_b32_e32 v4, 0xcf, v90
	v_add_lshl_u32 v91, v5, v74, 1
	v_lshlrev_b32_e32 v5, 4, v70
	s_movk_i32 s14, 0x88
	s_addc_u32 s17, s13, 0
	v_mad_u32_u24 v92, v71, s14, v5
	v_mad_u32_u24 v93, v4, s14, v5
	v_mad_u32_u24 v94, v3, s14, v5
	s_add_u32 s14, s16, -1
	v_mov_b32_e32 v1, v2
	s_addc_u32 s15, s17, -1
	s_lshr_b64 s[14:15], s[14:15], 6
	v_lshl_add_u64 v[4:5], s[38:39], 0, v[0:1]
	s_mov_b64 s[56:57], 0x80
	s_add_u32 s54, s14, 1
	v_lshl_add_u64 v[6:7], v[4:5], 0, s[56:57]
	s_addc_u32 s55, s15, 0
	v_mul_lo_u32 v3, v7, s16
	v_mul_lo_u32 v8, v6, s17
	v_mad_u64_u32 v[6:7], s[14:15], v6, s16, 0
	v_lshlrev_b32_e32 v76, 4, v90
	v_add3_u32 v7, v7, v8, v3
	v_lshlrev_b64 v[6:7], 1, v[6:7]
	v_and_b32_e32 v3, 0xf80, v76
	v_sub_co_u32_e64 v6, s[14:15], v6, v3
	v_lshl_add_u64 v[0:1], s[36:37], 0, v[0:1]
	s_nop 0
	v_subbrev_co_u32_e64 v7, s[14:15], 0, v7, s[14:15]
	v_lshl_add_u64 v[78:79], s[4:5], 0, v[6:7]
	v_mul_lo_u32 v6, s17, v4
	v_mul_lo_u32 v7, s16, v5
	v_mad_u64_u32 v[4:5], s[14:15], s16, v4, 0
	v_add3_u32 v5, v5, v7, v6
	v_lshlrev_b64 v[4:5], 1, v[4:5]
	v_sub_co_u32_e64 v4, s[14:15], v4, v3
	v_ashrrev_i32_e32 v75, 31, v74
	s_nop 0
	v_subbrev_co_u32_e64 v5, s[14:15], 0, v5, s[14:15]
	v_lshl_add_u64 v[80:81], s[4:5], 0, v[4:5]
	v_lshl_add_u64 v[4:5], v[0:1], 0, s[56:57]
	v_mul_lo_u32 v6, v5, s16
	v_mul_lo_u32 v7, v4, s17
	v_mad_u64_u32 v[4:5], s[4:5], v4, s16, 0
	v_add3_u32 v5, v5, v7, v6
	v_lshlrev_b64 v[4:5], 1, v[4:5]
	v_sub_co_u32_e64 v4, s[4:5], v4, v3
	v_mov_b32_e32 v77, v2
	s_nop 0
	v_subbrev_co_u32_e64 v5, s[4:5], 0, v5, s[4:5]
	v_lshl_add_u64 v[82:83], s[2:3], 0, v[4:5]
	v_mul_lo_u32 v4, s17, v0
	v_mul_lo_u32 v5, s16, v1
	v_mad_u64_u32 v[0:1], s[4:5], s16, v0, 0
	v_add3_u32 v1, v1, v5, v4
	v_lshlrev_b64 v[0:1], 1, v[0:1]
	v_sub_co_u32_e64 v0, s[4:5], v0, v3
	v_mov_b32_e32 v66, 0
	s_nop 0
	v_subbrev_co_u32_e64 v1, s[4:5], 0, v1, s[4:5]
	v_lshl_add_u64 v[84:85], s[2:3], 0, v[0:1]
	s_mov_b32 s4, 0x5040100
	s_mov_b32 s5, 0xffff
	v_mov_b32_e32 v67, v2
	v_mov_b32_e32 v68, v2
	v_mov_b32_e32 v69, v2
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v63, v2
	v_mov_b32_e32 v64, v2
	v_mov_b32_e32 v65, v2
	v_mov_b32_e32 v58, 0
	v_mov_b32_e32 v59, v2
	v_mov_b32_e32 v60, v2
	v_mov_b32_e32 v61, v2
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v55, v2
	v_mov_b32_e32 v56, v2
	v_mov_b32_e32 v57, v2
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v51, v2
	v_mov_b32_e32 v52, v2
	v_mov_b32_e32 v53, v2
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v47, v2
	v_mov_b32_e32 v48, v2
	v_mov_b32_e32 v49, v2
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v43, v2
	v_mov_b32_e32 v44, v2
	v_mov_b32_e32 v45, v2
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v39, v2
	v_mov_b32_e32 v40, v2
	v_mov_b32_e32 v41, v2
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v35, v2
	v_mov_b32_e32 v36, v2
	v_mov_b32_e32 v37, v2
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v31, v2
	v_mov_b32_e32 v32, v2
	v_mov_b32_e32 v33, v2
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v27, v2
	v_mov_b32_e32 v28, v2
	v_mov_b32_e32 v29, v2
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v23, v2
	v_mov_b32_e32 v24, v2
	v_mov_b32_e32 v25, v2
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v19, v2
	v_mov_b32_e32 v20, v2
	v_mov_b32_e32 v21, v2
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v11, v2
	v_mov_b32_e32 v12, v2
	v_mov_b32_e32 v13, v2
	v_mov_b32_e32 v14, 0
	v_mov_b32_e32 v15, v2
	v_mov_b32_e32 v16, v2
	v_mov_b32_e32 v17, v2
	v_mov_b32_e32 v6, 0
	v_mov_b32_e32 v7, v2
	v_mov_b32_e32 v8, v2
	v_mov_b32_e32 v9, v2
	v_mov_b32_e32 v95, 0x5040100
	scratch_store_dword off, v71, off
	s_branch .LBB0_3
.LBB0_2:
	s_or_b64 exec, exec, s[2:3]
	v_add_u32_e32 v0, 0x4400, v91
	ds_write2_b64 v0, v[70:71], v[72:73] offset1:1
	v_add_u32_e32 v0, 0x8800, v93
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read2_b64 v[104:107], v0 offset1:1
	v_add_u32_e32 v0, 0x8840, v93
	ds_read2_b64 v[108:111], v0 offset1:1
	ds_read2_b64 v[112:115], v92 offset1:1
	ds_read2_b64 v[116:119], v92 offset0:8 offset1:9
	v_add_u32_e32 v0, 0x880, v92
	ds_read2_b64 v[120:123], v0 offset1:1
	v_add_u32_e32 v0, 0x8c0, v92
	ds_read2_b64 v[124:127], v0 offset1:1
	v_add_u32_e32 v0, 0x1100, v92
	ds_read2_b64 v[86:89], v0 offset1:1
	v_add_u32_e32 v0, 0x1140, v92
	ds_read2_b64 v[100:103], v0 offset1:1
	v_add_u32_e32 v0, 0x1980, v92
	ds_read2_b64 v[70:73], v0 offset1:1
	v_add_u32_e32 v0, 0x19c0, v92
	ds_read2_b64 v[96:99], v0 offset1:1
	v_add_u32_e32 v0, 0x9080, v93
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_16x16x32_bf16 v[66:69], v[104:107], v[112:115], v[66:69]
	s_add_u32 s54, s54, -1
	s_addc_u32 s55, s55, -1
	v_lshl_add_u64 v[78:79], v[78:79], 0, s[56:57]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x32_bf16 v[62:65], v[104:107], v[120:123], v[62:65]
	v_lshl_add_u64 v[80:81], v[80:81], 0, s[56:57]
	v_lshl_add_u64 v[82:83], v[82:83], 0, s[56:57]
	v_lshl_add_u64 v[84:85], v[84:85], 0, s[56:57]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_bf16 v[58:61], v[104:107], v[86:89], v[58:61]
	v_lshl_add_u64 v[74:75], v[74:75], 0, 64
	s_cmp_lg_u64 s[54:55], 0
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_bf16 v[54:57], v[104:107], v[70:73], v[54:57]
	ds_read2_b64 v[104:107], v0 offset1:1
	v_add_u32_e32 v0, 0x90c0, v93
	v_mfma_f32_16x16x32_bf16 v[66:69], v[108:111], v[116:119], v[66:69]
	v_mfma_f32_16x16x32_bf16 v[62:65], v[108:111], v[124:127], v[62:65]
	v_mfma_f32_16x16x32_bf16 v[58:61], v[108:111], v[100:103], v[58:61]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_bf16 v[54:57], v[108:111], v[96:99], v[54:57]
	ds_read2_b64 v[108:111], v0 offset1:1
	v_add_u32_e32 v0, 0x9900, v93
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_bf16 v[50:53], v[104:107], v[112:115], v[50:53]
	v_mfma_f32_16x16x32_bf16 v[46:49], v[104:107], v[120:123], v[46:49]
	v_mfma_f32_16x16x32_bf16 v[42:45], v[104:107], v[86:89], v[42:45]
	v_mfma_f32_16x16x32_bf16 v[38:41], v[104:107], v[70:73], v[38:41]
	ds_read2_b64 v[104:107], v0 offset1:1
	v_add_u32_e32 v0, 0x9940, v93
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_bf16 v[50:53], v[108:111], v[116:119], v[50:53]
	v_mfma_f32_16x16x32_bf16 v[46:49], v[108:111], v[124:127], v[46:49]
	v_mfma_f32_16x16x32_bf16 v[42:45], v[108:111], v[100:103], v[42:45]
	v_mfma_f32_16x16x32_bf16 v[38:41], v[108:111], v[96:99], v[38:41]
	ds_read2_b64 v[108:111], v0 offset1:1
	v_add_u32_e32 v0, 0x8800, v94
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_bf16 v[34:37], v[104:107], v[112:115], v[34:37]
	v_mfma_f32_16x16x32_bf16 v[30:33], v[104:107], v[120:123], v[30:33]
	v_mfma_f32_16x16x32_bf16 v[26:29], v[104:107], v[86:89], v[26:29]
	v_mfma_f32_16x16x32_bf16 v[22:25], v[104:107], v[70:73], v[22:25]
	ds_read2_b64 v[104:107], v0 offset1:1
	v_add_u32_e32 v0, 0x8840, v94
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_bf16 v[34:37], v[108:111], v[116:119], v[34:37]
	v_mfma_f32_16x16x32_bf16 v[30:33], v[108:111], v[124:127], v[30:33]
	v_mfma_f32_16x16x32_bf16 v[26:29], v[108:111], v[100:103], v[26:29]
	v_mfma_f32_16x16x32_bf16 v[22:25], v[108:111], v[96:99], v[22:25]
	ds_read2_b64 v[108:111], v0 offset1:1
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_bf16 v[18:21], v[104:107], v[112:115], v[18:21]
	v_mfma_f32_16x16x32_bf16 v[10:13], v[104:107], v[120:123], v[10:13]
	v_mfma_f32_16x16x32_bf16 v[14:17], v[104:107], v[86:89], v[14:17]
	v_mfma_f32_16x16x32_bf16 v[4:7], v[104:107], v[70:73], v[6:9]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_bf16 v[18:21], v[108:111], v[116:119], v[18:21]
	v_mfma_f32_16x16x32_bf16 v[10:13], v[108:111], v[124:127], v[10:13]
	v_mfma_f32_16x16x32_bf16 v[14:17], v[108:111], v[100:103], v[14:17]
	v_mfma_f32_16x16x32_bf16 v[6:9], v[108:111], v[96:99], v[4:7]
	s_cbranch_scc0 .LBB0_67
.LBB0_3:
	s_nop 1
	v_mov_b32_e32 v4, v2
	v_mov_b32_e32 v5, v2
	v_cmp_gt_i64_e64 s[28:29], s[12:13], v[74:75]
	v_mov_b32_e32 v3, v2
	v_mov_b64_e32 v[72:73], v[4:5]
	v_cmp_gt_i64_e64 s[14:15], s[40:41], v[74:75]
	v_cmp_gt_i64_e64 s[16:17], s[42:43], v[74:75]
	v_cmp_gt_i64_e64 s[18:19], s[44:45], v[74:75]
	v_cmp_gt_i64_e64 s[20:21], s[46:47], v[74:75]
	v_cmp_gt_i64_e64 s[26:27], s[52:53], v[74:75]
	v_cmp_gt_i64_e64 s[24:25], s[50:51], v[74:75]
	v_cmp_gt_i64_e64 s[22:23], s[48:49], v[74:75]
	s_and_b64 s[58:59], vcc, s[28:29]
	v_lshl_add_u64 v[86:87], v[84:85], 0, v[76:77]
	v_mov_b64_e32 v[70:71], v[2:3]
	s_and_saveexec_b64 s[2:3], s[58:59]
	s_cbranch_execnz .LBB0_38
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[58:59], vcc, s[26:27]
	s_and_saveexec_b64 s[2:3], s[58:59]
	s_cbranch_execnz .LBB0_39
.LBB0_5:
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[58:59], vcc, s[24:25]
	s_and_saveexec_b64 s[2:3], s[58:59]
	s_cbranch_execnz .LBB0_40
.LBB0_6:
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[58:59], vcc, s[22:23]
	s_and_saveexec_b64 s[2:3], s[58:59]
	s_cbranch_execnz .LBB0_41
.LBB0_7:
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[58:59], vcc, s[20:21]
	s_and_saveexec_b64 s[2:3], s[58:59]
	s_cbranch_execnz .LBB0_42
.LBB0_8:
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[58:59], vcc, s[18:19]
	s_and_saveexec_b64 s[2:3], s[58:59]
	s_cbranch_execnz .LBB0_43
.LBB0_9:
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[58:59], vcc, s[16:17]
	s_and_saveexec_b64 s[2:3], s[58:59]
	s_cbranch_execnz .LBB0_44
.LBB0_10:
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[58:59], vcc, s[14:15]
	s_and_saveexec_b64 s[2:3], s[58:59]
	s_cbranch_execz .LBB0_12
.LBB0_11:
	global_load_ushort v0, v[86:87], off offset:14
	s_waitcnt vmcnt(0)
	v_perm_b32 v73, v0, v73, s4
.LBB0_12:
	s_or_b64 exec, exec, s[2:3]
	v_add_u32_e32 v0, 0x8800, v91
	v_mov_b32_e32 v4, v2
	v_mov_b32_e32 v5, v2
	s_barrier
	ds_write2_b64 v0, v[70:71], v[72:73] offset1:1
	v_mov_b32_e32 v3, v2
	v_mov_b64_e32 v[72:73], v[4:5]
	s_and_b64 s[58:59], s[0:1], s[28:29]
	v_lshl_add_u64 v[86:87], v[82:83], 0, v[76:77]
	v_mov_b64_e32 v[70:71], v[2:3]
	s_and_saveexec_b64 s[2:3], s[58:59]
	s_cbranch_execnz .LBB0_45
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[58:59], s[0:1], s[26:27]
	s_and_saveexec_b64 s[2:3], s[58:59]
	s_cbranch_execnz .LBB0_46
.LBB0_14:
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[58:59], s[0:1], s[24:25]
	s_and_saveexec_b64 s[2:3], s[58:59]
	s_cbranch_execnz .LBB0_47
.LBB0_15:
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[58:59], s[0:1], s[22:23]
	s_and_saveexec_b64 s[2:3], s[58:59]
	s_cbranch_execnz .LBB0_48
.LBB0_16:
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[58:59], s[0:1], s[20:21]
	s_and_saveexec_b64 s[2:3], s[58:59]
	s_cbranch_execnz .LBB0_49
.LBB0_17:
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[58:59], s[0:1], s[18:19]
	s_and_saveexec_b64 s[2:3], s[58:59]
	s_cbranch_execnz .LBB0_50
.LBB0_18:
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[58:59], s[0:1], s[16:17]
	s_and_saveexec_b64 s[2:3], s[58:59]
	s_cbranch_execnz .LBB0_51
.LBB0_19:
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[58:59], s[0:1], s[14:15]
	s_and_saveexec_b64 s[2:3], s[58:59]
	s_cbranch_execz .LBB0_21
.LBB0_20:
	global_load_ushort v0, v[86:87], off offset:14
	s_waitcnt vmcnt(0)
	v_perm_b32 v73, v0, v73, s4
.LBB0_21:
	s_or_b64 exec, exec, s[2:3]
	v_add_u32_e32 v0, 0xcc00, v91
	v_mov_b32_e32 v4, v2
	v_mov_b32_e32 v5, v2
	ds_write2_b64 v0, v[70:71], v[72:73] offset1:1
	v_mov_b32_e32 v3, v2
	v_mov_b64_e32 v[72:73], v[4:5]
	s_and_b64 s[58:59], s[30:31], s[28:29]
	v_lshl_add_u64 v[86:87], v[80:81], 0, v[76:77]
	v_mov_b64_e32 v[70:71], v[2:3]
	s_and_saveexec_b64 s[2:3], s[58:59]
	s_cbranch_execnz .LBB0_52
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[58:59], s[30:31], s[26:27]
	s_and_saveexec_b64 s[2:3], s[58:59]
	s_cbranch_execnz .LBB0_53
.LBB0_23:
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[58:59], s[30:31], s[24:25]
	s_and_saveexec_b64 s[2:3], s[58:59]
	s_cbranch_execnz .LBB0_54
.LBB0_24:
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[58:59], s[30:31], s[22:23]
	s_and_saveexec_b64 s[2:3], s[58:59]
	s_cbranch_execnz .LBB0_55
.LBB0_25:
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[58:59], s[30:31], s[20:21]
	s_and_saveexec_b64 s[2:3], s[58:59]
	s_cbranch_execnz .LBB0_56
.LBB0_26:
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[58:59], s[30:31], s[18:19]
	s_and_saveexec_b64 s[2:3], s[58:59]
	s_cbranch_execnz .LBB0_57
.LBB0_27:
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[58:59], s[30:31], s[16:17]
	s_and_saveexec_b64 s[2:3], s[58:59]
	s_cbranch_execnz .LBB0_58
.LBB0_28:
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[58:59], s[30:31], s[14:15]
	s_and_saveexec_b64 s[2:3], s[58:59]
	s_cbranch_execz .LBB0_30
.LBB0_29:
	global_load_ushort v0, v[86:87], off offset:14
	s_waitcnt vmcnt(0)
	v_perm_b32 v73, v0, v73, s4
.LBB0_30:
	s_or_b64 exec, exec, s[2:3]
	v_mov_b32_e32 v4, v2
	v_mov_b32_e32 v5, v2
	ds_write2_b64 v91, v[70:71], v[72:73] offset1:1
	v_mov_b32_e32 v3, v2
	v_mov_b64_e32 v[72:73], v[4:5]
	s_and_b64 s[28:29], s[34:35], s[28:29]
	v_lshl_add_u64 v[86:87], v[78:79], 0, v[76:77]
	v_mov_b64_e32 v[70:71], v[2:3]
	s_and_saveexec_b64 s[2:3], s[28:29]
	s_cbranch_execnz .LBB0_59
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[26:27], s[34:35], s[26:27]
	s_and_saveexec_b64 s[2:3], s[26:27]
	s_cbranch_execnz .LBB0_60
.LBB0_32:
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[24:25], s[34:35], s[24:25]
	s_and_saveexec_b64 s[2:3], s[24:25]
	s_cbranch_execnz .LBB0_61
.LBB0_33:
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[22:23], s[34:35], s[22:23]
	s_and_saveexec_b64 s[2:3], s[22:23]
	s_cbranch_execnz .LBB0_62
.LBB0_34:
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[20:21], s[34:35], s[20:21]
	s_and_saveexec_b64 s[2:3], s[20:21]
	s_cbranch_execnz .LBB0_63
.LBB0_35:
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[18:19], s[34:35], s[18:19]
	s_and_saveexec_b64 s[2:3], s[18:19]
	s_cbranch_execnz .LBB0_64
.LBB0_36:
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[16:17], s[34:35], s[16:17]
	s_and_saveexec_b64 s[2:3], s[16:17]
	s_cbranch_execnz .LBB0_65
.LBB0_37:
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[14:15], s[34:35], s[14:15]
	s_and_saveexec_b64 s[2:3], s[14:15]
	s_cbranch_execz .LBB0_2
	s_branch .LBB0_66
.LBB0_38:
	global_load_ushort v0, v[86:87], off
	v_mov_b32_e32 v3, v2
	v_mov_b32_e32 v1, v2
	s_waitcnt vmcnt(0)
	v_perm_b32 v0, 0, v0, v95
	v_mov_b64_e32 v[72:73], v[2:3]
	v_mov_b64_e32 v[70:71], v[0:1]
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[58:59], vcc, s[26:27]
	s_and_saveexec_b64 s[2:3], s[58:59]
	s_cbranch_execz .LBB0_5
.LBB0_39:
	global_load_ushort v0, v[86:87], off offset:2
	s_waitcnt vmcnt(0)
	v_perm_b32 v70, v0, v70, s4
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[58:59], vcc, s[24:25]
	s_and_saveexec_b64 s[2:3], s[58:59]
	s_cbranch_execz .LBB0_6
.LBB0_40:
	global_load_ushort v0, v[86:87], off offset:4
	s_waitcnt vmcnt(0)
	v_bfi_b32 v71, s5, v0, v71
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[58:59], vcc, s[22:23]
	s_and_saveexec_b64 s[2:3], s[58:59]
	s_cbranch_execz .LBB0_7
.LBB0_41:
	global_load_ushort v0, v[86:87], off offset:6
	s_waitcnt vmcnt(0)
	v_perm_b32 v71, v0, v71, s4
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[58:59], vcc, s[20:21]
	s_and_saveexec_b64 s[2:3], s[58:59]
	s_cbranch_execz .LBB0_8
.LBB0_42:
	global_load_ushort v0, v[86:87], off offset:8
	s_waitcnt vmcnt(0)
	v_bfi_b32 v72, s5, v0, v72
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[58:59], vcc, s[18:19]
	s_and_saveexec_b64 s[2:3], s[58:59]
	s_cbranch_execz .LBB0_9
.LBB0_43:
	global_load_ushort v0, v[86:87], off offset:10
	s_waitcnt vmcnt(0)
	v_perm_b32 v72, v0, v72, s4
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[58:59], vcc, s[16:17]
	s_and_saveexec_b64 s[2:3], s[58:59]
	s_cbranch_execz .LBB0_10
.LBB0_44:
	global_load_ushort v0, v[86:87], off offset:12
	s_waitcnt vmcnt(0)
	v_bfi_b32 v73, s5, v0, v73
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[58:59], vcc, s[14:15]
	s_and_saveexec_b64 s[2:3], s[58:59]
	s_cbranch_execnz .LBB0_11
	s_branch .LBB0_12
.LBB0_45:
	global_load_ushort v0, v[86:87], off
	v_mov_b32_e32 v3, v2
	v_mov_b32_e32 v1, v2
	s_waitcnt vmcnt(0)
	v_perm_b32 v0, 0, v0, v95
	v_mov_b64_e32 v[72:73], v[2:3]
	v_mov_b64_e32 v[70:71], v[0:1]
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[58:59], s[0:1], s[26:27]
	s_and_saveexec_b64 s[2:3], s[58:59]
	s_cbranch_execz .LBB0_14
.LBB0_46:
	global_load_ushort v0, v[86:87], off offset:2
	s_waitcnt vmcnt(0)
	v_perm_b32 v70, v0, v70, s4
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[58:59], s[0:1], s[24:25]
	s_and_saveexec_b64 s[2:3], s[58:59]
	s_cbranch_execz .LBB0_15
.LBB0_47:
	global_load_ushort v0, v[86:87], off offset:4
	s_waitcnt vmcnt(0)
	v_bfi_b32 v71, s5, v0, v71
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[58:59], s[0:1], s[22:23]
	s_and_saveexec_b64 s[2:3], s[58:59]
	s_cbranch_execz .LBB0_16
.LBB0_48:
	global_load_ushort v0, v[86:87], off offset:6
	s_waitcnt vmcnt(0)
	v_perm_b32 v71, v0, v71, s4
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[58:59], s[0:1], s[20:21]
	s_and_saveexec_b64 s[2:3], s[58:59]
	s_cbranch_execz .LBB0_17
.LBB0_49:
	global_load_ushort v0, v[86:87], off offset:8
	s_waitcnt vmcnt(0)
	v_bfi_b32 v72, s5, v0, v72
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[58:59], s[0:1], s[18:19]
	s_and_saveexec_b64 s[2:3], s[58:59]
	s_cbranch_execz .LBB0_18
.LBB0_50:
	global_load_ushort v0, v[86:87], off offset:10
	s_waitcnt vmcnt(0)
	v_perm_b32 v72, v0, v72, s4
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[58:59], s[0:1], s[16:17]
	s_and_saveexec_b64 s[2:3], s[58:59]
	s_cbranch_execz .LBB0_19
.LBB0_51:
	global_load_ushort v0, v[86:87], off offset:12
	s_waitcnt vmcnt(0)
	v_bfi_b32 v73, s5, v0, v73
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[58:59], s[0:1], s[14:15]
	s_and_saveexec_b64 s[2:3], s[58:59]
	s_cbranch_execnz .LBB0_20
	s_branch .LBB0_21
.LBB0_52:
	global_load_ushort v0, v[86:87], off
	v_mov_b32_e32 v3, v2
	v_mov_b32_e32 v1, v2
	s_waitcnt vmcnt(0)
	v_perm_b32 v0, 0, v0, v95
	v_mov_b64_e32 v[72:73], v[2:3]
	v_mov_b64_e32 v[70:71], v[0:1]
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[58:59], s[30:31], s[26:27]
	s_and_saveexec_b64 s[2:3], s[58:59]
	s_cbranch_execz .LBB0_23
.LBB0_53:
	global_load_ushort v0, v[86:87], off offset:2
	s_waitcnt vmcnt(0)
	v_perm_b32 v70, v0, v70, s4
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[58:59], s[30:31], s[24:25]
	s_and_saveexec_b64 s[2:3], s[58:59]
	s_cbranch_execz .LBB0_24
.LBB0_54:
	global_load_ushort v0, v[86:87], off offset:4
	s_waitcnt vmcnt(0)
	v_bfi_b32 v71, s5, v0, v71
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[58:59], s[30:31], s[22:23]
	s_and_saveexec_b64 s[2:3], s[58:59]
	s_cbranch_execz .LBB0_25
.LBB0_55:
	global_load_ushort v0, v[86:87], off offset:6
	s_waitcnt vmcnt(0)
	v_perm_b32 v71, v0, v71, s4
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[58:59], s[30:31], s[20:21]
	s_and_saveexec_b64 s[2:3], s[58:59]
	s_cbranch_execz .LBB0_26
.LBB0_56:
	global_load_ushort v0, v[86:87], off offset:8
	s_waitcnt vmcnt(0)
	v_bfi_b32 v72, s5, v0, v72
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[58:59], s[30:31], s[18:19]
	s_and_saveexec_b64 s[2:3], s[58:59]
	s_cbranch_execz .LBB0_27
.LBB0_57:
	global_load_ushort v0, v[86:87], off offset:10
	s_waitcnt vmcnt(0)
	v_perm_b32 v72, v0, v72, s4
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[58:59], s[30:31], s[16:17]
	s_and_saveexec_b64 s[2:3], s[58:59]
	s_cbranch_execz .LBB0_28
.LBB0_58:
	global_load_ushort v0, v[86:87], off offset:12
	s_waitcnt vmcnt(0)
	v_bfi_b32 v73, s5, v0, v73
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[58:59], s[30:31], s[14:15]
	s_and_saveexec_b64 s[2:3], s[58:59]
	s_cbranch_execnz .LBB0_29
	s_branch .LBB0_30
.LBB0_59:
	global_load_ushort v0, v[86:87], off
	v_mov_b32_e32 v3, v2
	v_mov_b32_e32 v1, v2
	s_waitcnt vmcnt(0)
	v_perm_b32 v0, 0, v0, v95
	v_mov_b64_e32 v[72:73], v[2:3]
	v_mov_b64_e32 v[70:71], v[0:1]
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[26:27], s[34:35], s[26:27]
	s_and_saveexec_b64 s[2:3], s[26:27]
	s_cbranch_execz .LBB0_32
.LBB0_60:
	global_load_ushort v0, v[86:87], off offset:2
	s_waitcnt vmcnt(0)
	v_perm_b32 v70, v0, v70, s4
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[24:25], s[34:35], s[24:25]
	s_and_saveexec_b64 s[2:3], s[24:25]
	s_cbranch_execz .LBB0_33
.LBB0_61:
	global_load_ushort v0, v[86:87], off offset:4
	s_waitcnt vmcnt(0)
	v_bfi_b32 v71, s5, v0, v71
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[22:23], s[34:35], s[22:23]
	s_and_saveexec_b64 s[2:3], s[22:23]
	s_cbranch_execz .LBB0_34
.LBB0_62:
	global_load_ushort v0, v[86:87], off offset:6
	s_waitcnt vmcnt(0)
	v_perm_b32 v71, v0, v71, s4
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[20:21], s[34:35], s[20:21]
	s_and_saveexec_b64 s[2:3], s[20:21]
	s_cbranch_execz .LBB0_35
.LBB0_63:
	global_load_ushort v0, v[86:87], off offset:8
	s_waitcnt vmcnt(0)
	v_bfi_b32 v72, s5, v0, v72
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[18:19], s[34:35], s[18:19]
	s_and_saveexec_b64 s[2:3], s[18:19]
	s_cbranch_execz .LBB0_36
.LBB0_64:
	global_load_ushort v0, v[86:87], off offset:10
	s_waitcnt vmcnt(0)
	v_perm_b32 v72, v0, v72, s4
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[16:17], s[34:35], s[16:17]
	s_and_saveexec_b64 s[2:3], s[16:17]
	s_cbranch_execz .LBB0_37
.LBB0_65:
	global_load_ushort v0, v[86:87], off offset:12
	s_waitcnt vmcnt(0)
	v_bfi_b32 v73, s5, v0, v73
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[14:15], s[34:35], s[14:15]
	s_and_saveexec_b64 s[2:3], s[14:15]
	s_cbranch_execz .LBB0_2
.LBB0_66:
	global_load_ushort v0, v[86:87], off offset:14
	s_waitcnt vmcnt(0)
	v_perm_b32 v73, v0, v73, s4
	s_branch .LBB0_2
.LBB0_67:
	scratch_load_dword v71, off, off
	v_bfe_u32 v70, v90, 4, 2
	s_branch .LBB0_69
.LBB0_68:
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v68, v69
	v_mov_b32_e32 v67, v69
	v_mov_b32_e32 v66, v69
	v_mov_b32_e32 v65, v69
	v_mov_b32_e32 v64, v69
	v_mov_b32_e32 v63, v69
	v_mov_b32_e32 v62, v69
	v_mov_b32_e32 v61, v69
	v_mov_b32_e32 v60, v69
	v_mov_b32_e32 v59, v69
	v_mov_b32_e32 v58, v69
	v_mov_b32_e32 v57, v69
	v_mov_b32_e32 v56, v69
	v_mov_b32_e32 v55, v69
	v_mov_b32_e32 v54, v69
	v_mov_b32_e32 v53, v69
	v_mov_b32_e32 v52, v69
	v_mov_b32_e32 v51, v69
	v_mov_b32_e32 v50, v69
	v_mov_b32_e32 v49, v69
	v_mov_b32_e32 v48, v69
	v_mov_b32_e32 v47, v69
	v_mov_b32_e32 v46, v69
	v_mov_b32_e32 v45, v69
	v_mov_b32_e32 v44, v69
	v_mov_b32_e32 v43, v69
	v_mov_b32_e32 v42, v69
	v_mov_b32_e32 v41, v69
	v_mov_b32_e32 v40, v69
	v_mov_b32_e32 v39, v69
	v_mov_b32_e32 v38, v69
	v_mov_b32_e32 v37, v69
	v_mov_b32_e32 v36, v69
	v_mov_b32_e32 v35, v69
	v_mov_b32_e32 v34, v69
	v_mov_b32_e32 v33, v69
	v_mov_b32_e32 v32, v69
	v_mov_b32_e32 v31, v69
	v_mov_b32_e32 v30, v69
	v_mov_b32_e32 v29, v69
	v_mov_b32_e32 v28, v69
	v_mov_b32_e32 v27, v69
	v_mov_b32_e32 v26, v69
	v_mov_b32_e32 v25, v69
	v_mov_b32_e32 v24, v69
	v_mov_b32_e32 v23, v69
	v_mov_b32_e32 v22, v69
	v_mov_b32_e32 v21, v69
	v_mov_b32_e32 v20, v69
	v_mov_b32_e32 v19, v69
	v_mov_b32_e32 v18, v69
	v_mov_b32_e32 v13, v69
	v_mov_b32_e32 v12, v69
	v_mov_b32_e32 v11, v69
	v_mov_b32_e32 v10, v69
	v_mov_b32_e32 v17, v69
	v_mov_b32_e32 v16, v69
	v_mov_b32_e32 v15, v69
	v_mov_b32_e32 v14, v69
	v_mov_b32_e32 v9, v69
	v_mov_b32_e32 v8, v69
	v_mov_b32_e32 v7, v69
	v_mov_b32_e32 v6, v69
.LBB0_69:
	v_and_b32_e32 v2, 0xc0, v90
	v_lshlrev_b32_e32 v3, 2, v70
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v0, s38, v71
	v_mov_b32_e32 v1, s39
	v_or3_b32 v2, v3, v2, s36
	v_mov_b32_e32 v3, s37
	v_cmp_gt_u64_e32 vcc, s[10:11], v[0:1]
	v_cmp_gt_u64_e64 s[12:13], s[8:9], v[2:3]
	s_and_b64 s[2:3], s[12:13], vcc
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_71
	v_mul_lo_u32 v70, s11, v2
	v_mul_lo_u32 v71, s10, v3
	v_mad_u64_u32 v[4:5], s[2:3], s10, v2, 0
	v_add3_u32 v5, v5, v71, v70
	v_lshl_add_u64 v[4:5], v[4:5], 2, s[6:7]
	v_lshl_add_u64 v[4:5], v[0:1], 2, v[4:5]
	global_store_dword v[4:5], v66, off
.LBB0_71:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v4, 1, v2
	v_mov_b32_e32 v5, v3
	v_cmp_gt_u64_e64 s[14:15], s[8:9], v[4:5]
	s_and_b64 s[2:3], s[14:15], vcc
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_73
	v_mul_lo_u32 v66, s11, v4
	v_mul_lo_u32 v72, s10, v5
	v_mad_u64_u32 v[70:71], s[2:3], s10, v4, 0
	v_add3_u32 v71, v71, v72, v66
	v_lshl_add_u64 v[70:71], v[70:71], 2, s[6:7]
	v_lshl_add_u64 v[70:71], v[0:1], 2, v[70:71]
	global_store_dword v[70:71], v67, off
.LBB0_73:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v66, 2, v2
	v_mov_b32_e32 v67, v3
	v_cmp_gt_u64_e64 s[16:17], s[8:9], v[66:67]
	s_and_b64 s[2:3], s[16:17], vcc
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_75
	v_mul_lo_u32 v72, s11, v66
	v_mul_lo_u32 v73, s10, v67
	v_mad_u64_u32 v[70:71], s[2:3], s10, v66, 0
	v_add3_u32 v71, v71, v73, v72
	v_lshl_add_u64 v[70:71], v[70:71], 2, s[6:7]
	v_lshl_add_u64 v[70:71], v[0:1], 2, v[70:71]
	global_store_dword v[70:71], v68, off
.LBB0_75:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v70, 3, v2
	v_mov_b32_e32 v71, v3
	v_cmp_gt_u64_e64 s[18:19], s[8:9], v[70:71]
	s_and_b64 s[2:3], s[18:19], vcc
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_77
	v_mul_lo_u32 v68, s11, v70
	v_mul_lo_u32 v74, s10, v71
	v_mad_u64_u32 v[72:73], s[2:3], s10, v70, 0
	v_add3_u32 v73, v73, v74, v68
	v_lshl_add_u64 v[72:73], v[72:73], 2, s[6:7]
	v_lshl_add_u64 v[72:73], v[0:1], 2, v[72:73]
	global_store_dword v[72:73], v69, off
.LBB0_77:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v68, 16, v0
	v_mov_b32_e32 v69, v1
	v_cmp_gt_u64_e64 s[0:1], s[10:11], v[68:69]
	s_and_b64 s[4:5], s[12:13], s[0:1]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execnz .LBB0_152
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[4:5], s[14:15], s[0:1]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execnz .LBB0_153
.LBB0_79:
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[4:5], s[16:17], s[0:1]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execnz .LBB0_154
.LBB0_80:
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[4:5], s[18:19], s[0:1]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_82
.LBB0_81:
	v_mul_lo_u32 v64, s11, v70
	v_mul_lo_u32 v68, s10, v71
	v_mad_u64_u32 v[62:63], s[4:5], s10, v70, 0
	v_add3_u32 v63, v63, v68, v64
	v_lshl_add_u64 v[62:63], v[62:63], 2, s[6:7]
	v_lshl_add_u64 v[62:63], v[0:1], 2, v[62:63]
	global_store_dword v[62:63], v65, off offset:64
.LBB0_82:
	s_or_b64 exec, exec, s[2:3]
	v_or_b32_e32 v62, 32, v0
	v_mov_b32_e32 v63, v1
	v_cmp_gt_u64_e64 s[2:3], s[10:11], v[62:63]
	s_and_b64 s[20:21], s[12:13], s[2:3]
	s_and_saveexec_b64 s[4:5], s[20:21]
	s_cbranch_execnz .LBB0_155
	s_or_b64 exec, exec, s[4:5]
	s_and_b64 s[20:21], s[14:15], s[2:3]
	s_and_saveexec_b64 s[4:5], s[20:21]
	s_cbranch_execnz .LBB0_156
.LBB0_84:
	s_or_b64 exec, exec, s[4:5]
	s_and_b64 s[20:21], s[16:17], s[2:3]
	s_and_saveexec_b64 s[4:5], s[20:21]
	s_cbranch_execnz .LBB0_157
.LBB0_85:
	s_or_b64 exec, exec, s[4:5]
	s_and_b64 s[20:21], s[18:19], s[2:3]
	s_and_saveexec_b64 s[4:5], s[20:21]
	s_cbranch_execz .LBB0_87
.LBB0_86:
	v_mul_lo_u32 v60, s11, v70
	v_mul_lo_u32 v62, s10, v71
	v_mad_u64_u32 v[58:59], s[20:21], s10, v70, 0
	v_add3_u32 v59, v59, v62, v60
	v_lshl_add_u64 v[58:59], v[58:59], 2, s[6:7]
	v_lshl_add_u64 v[58:59], v[0:1], 2, v[58:59]
	global_store_dword v[58:59], v61, off offset:128
.LBB0_87:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v58, 48, v0
	v_mov_b32_e32 v59, v1
	v_cmp_gt_u64_e64 s[4:5], s[10:11], v[58:59]
	s_and_b64 s[20:21], s[12:13], s[4:5]
	s_and_saveexec_b64 s[12:13], s[20:21]
	s_cbranch_execnz .LBB0_158
	s_or_b64 exec, exec, s[12:13]
	s_and_b64 s[14:15], s[14:15], s[4:5]
	s_and_saveexec_b64 s[12:13], s[14:15]
	s_cbranch_execnz .LBB0_159
.LBB0_89:
	s_or_b64 exec, exec, s[12:13]
	s_and_b64 s[14:15], s[16:17], s[4:5]
	s_and_saveexec_b64 s[12:13], s[14:15]
	s_cbranch_execnz .LBB0_160
.LBB0_90:
	s_or_b64 exec, exec, s[12:13]
	s_and_b64 s[14:15], s[18:19], s[4:5]
	s_and_saveexec_b64 s[12:13], s[14:15]
	s_cbranch_execz .LBB0_92
.LBB0_91:
	v_mul_lo_u32 v54, s11, v70
	v_mul_lo_u32 v55, s10, v71
	v_mad_u64_u32 v[4:5], s[14:15], s10, v70, 0
	v_add3_u32 v5, v5, v55, v54
	v_lshl_add_u64 v[4:5], v[4:5], 2, s[6:7]
	v_lshl_add_u64 v[4:5], v[0:1], 2, v[4:5]
	global_store_dword v[4:5], v57, off offset:192
.LBB0_92:
	s_or_b64 exec, exec, s[12:13]
	v_or_b32_e32 v4, 16, v2
	v_mov_b32_e32 v5, v3
	v_cmp_gt_u64_e64 s[12:13], s[8:9], v[4:5]
	s_and_b64 s[16:17], s[12:13], vcc
	s_and_saveexec_b64 s[14:15], s[16:17]
	s_cbranch_execz .LBB0_94
	v_mul_lo_u32 v56, s11, v4
	v_mul_lo_u32 v57, s10, v5
	v_mad_u64_u32 v[54:55], s[16:17], s10, v4, 0
	v_add3_u32 v55, v55, v57, v56
	v_lshl_add_u64 v[54:55], v[54:55], 2, s[6:7]
	v_lshl_add_u64 v[54:55], v[0:1], 2, v[54:55]
	global_store_dword v[54:55], v50, off
.LBB0_94:
	s_or_b64 exec, exec, s[14:15]
	v_or_b32_e32 v54, 17, v2
	v_mov_b32_e32 v55, v3
	v_cmp_gt_u64_e64 s[14:15], s[8:9], v[54:55]
	s_and_b64 s[18:19], s[14:15], vcc
	s_and_saveexec_b64 s[16:17], s[18:19]
	s_cbranch_execz .LBB0_96
	v_mul_lo_u32 v50, s11, v54
	v_mul_lo_u32 v58, s10, v55
	v_mad_u64_u32 v[56:57], s[18:19], s10, v54, 0
	v_add3_u32 v57, v57, v58, v50
	v_lshl_add_u64 v[56:57], v[56:57], 2, s[6:7]
	v_lshl_add_u64 v[56:57], v[0:1], 2, v[56:57]
	global_store_dword v[56:57], v51, off
.LBB0_96:
	s_or_b64 exec, exec, s[16:17]
	v_or_b32_e32 v50, 18, v2
	v_mov_b32_e32 v51, v3
	v_cmp_gt_u64_e64 s[16:17], s[8:9], v[50:51]
	s_and_b64 s[20:21], s[16:17], vcc
	s_and_saveexec_b64 s[18:19], s[20:21]
	s_cbranch_execz .LBB0_98
	v_mul_lo_u32 v58, s11, v50
	v_mul_lo_u32 v59, s10, v51
	v_mad_u64_u32 v[56:57], s[20:21], s10, v50, 0
	v_add3_u32 v57, v57, v59, v58
	v_lshl_add_u64 v[56:57], v[56:57], 2, s[6:7]
	v_lshl_add_u64 v[56:57], v[0:1], 2, v[56:57]
	global_store_dword v[56:57], v52, off
.LBB0_98:
	s_or_b64 exec, exec, s[18:19]
	v_or_b32_e32 v56, 19, v2
	v_mov_b32_e32 v57, v3
	v_cmp_gt_u64_e64 s[18:19], s[8:9], v[56:57]
	s_and_b64 s[22:23], s[18:19], vcc
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_161
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[12:13], s[0:1]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_162
.LBB0_100:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[0:1]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_163
.LBB0_101:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[16:17], s[0:1]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_164
.LBB0_102:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[18:19], s[0:1]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_165
.LBB0_103:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[12:13], s[2:3]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_166
.LBB0_104:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[2:3]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_167
.LBB0_105:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[16:17], s[2:3]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_168
.LBB0_106:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[18:19], s[2:3]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_169
.LBB0_107:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[20:21], s[12:13], s[4:5]
	s_and_saveexec_b64 s[12:13], s[20:21]
	s_cbranch_execnz .LBB0_170
.LBB0_108:
	s_or_b64 exec, exec, s[12:13]
	s_and_b64 s[14:15], s[14:15], s[4:5]
	s_and_saveexec_b64 s[12:13], s[14:15]
	s_cbranch_execnz .LBB0_171
.LBB0_109:
	s_or_b64 exec, exec, s[12:13]
	s_and_b64 s[14:15], s[16:17], s[4:5]
	s_and_saveexec_b64 s[12:13], s[14:15]
	s_cbranch_execnz .LBB0_172
.LBB0_110:
	s_or_b64 exec, exec, s[12:13]
	s_and_b64 s[14:15], s[18:19], s[4:5]
	s_and_saveexec_b64 s[12:13], s[14:15]
	s_cbranch_execz .LBB0_112
.LBB0_111:
	v_mul_lo_u32 v38, s11, v56
	v_mul_lo_u32 v39, s10, v57
	v_mad_u64_u32 v[4:5], s[14:15], s10, v56, 0
	v_add3_u32 v5, v5, v39, v38
	v_lshl_add_u64 v[4:5], v[4:5], 2, s[6:7]
	v_lshl_add_u64 v[4:5], v[0:1], 2, v[4:5]
	global_store_dword v[4:5], v41, off offset:192
.LBB0_112:
	s_or_b64 exec, exec, s[12:13]
	v_or_b32_e32 v4, 32, v2
	v_mov_b32_e32 v5, v3
	v_cmp_gt_u64_e64 s[12:13], s[8:9], v[4:5]
	s_and_b64 s[16:17], s[12:13], vcc
	s_and_saveexec_b64 s[14:15], s[16:17]
	s_cbranch_execz .LBB0_114
	v_mul_lo_u32 v40, s11, v4
	v_mul_lo_u32 v41, s10, v5
	v_mad_u64_u32 v[38:39], s[16:17], s10, v4, 0
	v_add3_u32 v39, v39, v41, v40
	v_lshl_add_u64 v[38:39], v[38:39], 2, s[6:7]
	v_lshl_add_u64 v[38:39], v[0:1], 2, v[38:39]
	global_store_dword v[38:39], v34, off
.LBB0_114:
	s_or_b64 exec, exec, s[14:15]
	v_or_b32_e32 v38, 33, v2
	v_mov_b32_e32 v39, v3
	v_cmp_gt_u64_e64 s[14:15], s[8:9], v[38:39]
	s_and_b64 s[18:19], s[14:15], vcc
	s_and_saveexec_b64 s[16:17], s[18:19]
	s_cbranch_execz .LBB0_116
	v_mul_lo_u32 v34, s11, v38
	v_mul_lo_u32 v42, s10, v39
	v_mad_u64_u32 v[40:41], s[18:19], s10, v38, 0
	v_add3_u32 v41, v41, v42, v34
	v_lshl_add_u64 v[40:41], v[40:41], 2, s[6:7]
	v_lshl_add_u64 v[40:41], v[0:1], 2, v[40:41]
	global_store_dword v[40:41], v35, off
.LBB0_116:
	s_or_b64 exec, exec, s[16:17]
	v_or_b32_e32 v34, 34, v2
	v_mov_b32_e32 v35, v3
	v_cmp_gt_u64_e64 s[16:17], s[8:9], v[34:35]
	s_and_b64 s[20:21], s[16:17], vcc
	s_and_saveexec_b64 s[18:19], s[20:21]
	s_cbranch_execz .LBB0_118
	v_mul_lo_u32 v42, s11, v34
	v_mul_lo_u32 v43, s10, v35
	v_mad_u64_u32 v[40:41], s[20:21], s10, v34, 0
	v_add3_u32 v41, v41, v43, v42
	v_lshl_add_u64 v[40:41], v[40:41], 2, s[6:7]
	v_lshl_add_u64 v[40:41], v[0:1], 2, v[40:41]
	global_store_dword v[40:41], v36, off
.LBB0_118:
	s_or_b64 exec, exec, s[18:19]
	v_or_b32_e32 v40, 35, v2
	v_mov_b32_e32 v41, v3
	v_cmp_gt_u64_e64 s[18:19], s[8:9], v[40:41]
	s_and_b64 s[22:23], s[18:19], vcc
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_173
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[12:13], s[0:1]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_174
.LBB0_120:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[0:1]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_175
.LBB0_121:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[16:17], s[0:1]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_176
.LBB0_122:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[18:19], s[0:1]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_177
.LBB0_123:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[12:13], s[2:3]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_178
.LBB0_124:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[2:3]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_179
.LBB0_125:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[16:17], s[2:3]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_180
.LBB0_126:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[18:19], s[2:3]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_181
.LBB0_127:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[20:21], s[12:13], s[4:5]
	s_and_saveexec_b64 s[12:13], s[20:21]
	s_cbranch_execnz .LBB0_182
.LBB0_128:
	s_or_b64 exec, exec, s[12:13]
	s_and_b64 s[14:15], s[14:15], s[4:5]
	s_and_saveexec_b64 s[12:13], s[14:15]
	s_cbranch_execnz .LBB0_183
.LBB0_129:
	s_or_b64 exec, exec, s[12:13]
	s_and_b64 s[14:15], s[16:17], s[4:5]
	s_and_saveexec_b64 s[12:13], s[14:15]
	s_cbranch_execnz .LBB0_184
.LBB0_130:
	s_or_b64 exec, exec, s[12:13]
	s_and_b64 s[14:15], s[18:19], s[4:5]
	s_and_saveexec_b64 s[12:13], s[14:15]
	s_cbranch_execz .LBB0_132
.LBB0_131:
	v_mul_lo_u32 v22, s11, v40
	v_mul_lo_u32 v23, s10, v41
	v_mad_u64_u32 v[4:5], s[14:15], s10, v40, 0
	v_add3_u32 v5, v5, v23, v22
	v_lshl_add_u64 v[4:5], v[4:5], 2, s[6:7]
	v_lshl_add_u64 v[4:5], v[0:1], 2, v[4:5]
	global_store_dword v[4:5], v25, off offset:192
.LBB0_132:
	s_or_b64 exec, exec, s[12:13]
	v_or_b32_e32 v4, 48, v2
	v_mov_b32_e32 v5, v3
	v_cmp_gt_u64_e64 s[12:13], s[8:9], v[4:5]
	s_and_b64 s[16:17], s[12:13], vcc
	s_and_saveexec_b64 s[14:15], s[16:17]
	s_cbranch_execz .LBB0_134
	v_mul_lo_u32 v24, s11, v4
	v_mul_lo_u32 v25, s10, v5
	v_mad_u64_u32 v[22:23], s[16:17], s10, v4, 0
	v_add3_u32 v23, v23, v25, v24
	v_lshl_add_u64 v[22:23], v[22:23], 2, s[6:7]
	v_lshl_add_u64 v[22:23], v[0:1], 2, v[22:23]
	global_store_dword v[22:23], v18, off
.LBB0_134:
	s_or_b64 exec, exec, s[14:15]
	v_or_b32_e32 v22, 49, v2
	v_mov_b32_e32 v23, v3
	v_cmp_gt_u64_e64 s[14:15], s[8:9], v[22:23]
	s_and_b64 s[18:19], s[14:15], vcc
	s_and_saveexec_b64 s[16:17], s[18:19]
	s_cbranch_execz .LBB0_136
	v_mul_lo_u32 v18, s11, v22
	v_mul_lo_u32 v26, s10, v23
	v_mad_u64_u32 v[24:25], s[18:19], s10, v22, 0
	v_add3_u32 v25, v25, v26, v18
	v_lshl_add_u64 v[24:25], v[24:25], 2, s[6:7]
	v_lshl_add_u64 v[24:25], v[0:1], 2, v[24:25]
	global_store_dword v[24:25], v19, off
.LBB0_136:
	s_or_b64 exec, exec, s[16:17]
	v_or_b32_e32 v18, 50, v2
	v_mov_b32_e32 v19, v3
	v_cmp_gt_u64_e64 s[16:17], s[8:9], v[18:19]
	s_and_b64 s[20:21], s[16:17], vcc
	s_and_saveexec_b64 s[18:19], s[20:21]
	s_cbranch_execz .LBB0_138
	v_mul_lo_u32 v26, s11, v18
	v_mul_lo_u32 v27, s10, v19
	v_mad_u64_u32 v[24:25], s[20:21], s10, v18, 0
	v_add3_u32 v25, v25, v27, v26
	v_lshl_add_u64 v[24:25], v[24:25], 2, s[6:7]
	v_lshl_add_u64 v[24:25], v[0:1], 2, v[24:25]
	global_store_dword v[24:25], v20, off
.LBB0_138:
	s_or_b64 exec, exec, s[18:19]
	v_or_b32_e32 v2, 51, v2
	v_cmp_gt_u64_e64 s[8:9], s[8:9], v[2:3]
	s_and_b64 s[20:21], s[8:9], vcc
	s_and_saveexec_b64 s[18:19], s[20:21]
	s_cbranch_execnz .LBB0_185
	s_or_b64 exec, exec, s[18:19]
	s_and_b64 s[20:21], s[12:13], s[0:1]
	s_and_saveexec_b64 s[18:19], s[20:21]
	s_cbranch_execnz .LBB0_186
.LBB0_140:
	s_or_b64 exec, exec, s[18:19]
	s_and_b64 s[20:21], s[14:15], s[0:1]
	s_and_saveexec_b64 s[18:19], s[20:21]
	s_cbranch_execnz .LBB0_187
.LBB0_141:
	s_or_b64 exec, exec, s[18:19]
	s_and_b64 s[20:21], s[16:17], s[0:1]
	s_and_saveexec_b64 s[18:19], s[20:21]
	s_cbranch_execnz .LBB0_188
.LBB0_142:
	s_or_b64 exec, exec, s[18:19]
	s_and_b64 s[18:19], s[8:9], s[0:1]
	s_and_saveexec_b64 s[0:1], s[18:19]
	s_cbranch_execnz .LBB0_189
.LBB0_143:
	s_or_b64 exec, exec, s[0:1]
	s_and_b64 s[18:19], s[12:13], s[2:3]
	s_and_saveexec_b64 s[0:1], s[18:19]
	s_cbranch_execnz .LBB0_190
.LBB0_144:
	s_or_b64 exec, exec, s[0:1]
	s_and_b64 s[18:19], s[14:15], s[2:3]
	s_and_saveexec_b64 s[0:1], s[18:19]
	s_cbranch_execnz .LBB0_191
.LBB0_145:
	s_or_b64 exec, exec, s[0:1]
	s_and_b64 s[18:19], s[16:17], s[2:3]
	s_and_saveexec_b64 s[0:1], s[18:19]
	s_cbranch_execnz .LBB0_192
.LBB0_146:
	s_or_b64 exec, exec, s[0:1]
	s_and_b64 s[2:3], s[8:9], s[2:3]
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execnz .LBB0_193
.LBB0_147:
	s_or_b64 exec, exec, s[0:1]
	s_and_b64 s[2:3], s[12:13], s[4:5]
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execnz .LBB0_194
.LBB0_148:
	s_or_b64 exec, exec, s[0:1]
	s_and_b64 s[2:3], s[14:15], s[4:5]
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execnz .LBB0_195
.LBB0_149:
	s_or_b64 exec, exec, s[0:1]
	s_and_b64 s[2:3], s[16:17], s[4:5]
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execnz .LBB0_196
.LBB0_150:
	s_or_b64 exec, exec, s[0:1]
	s_and_b64 s[0:1], s[8:9], s[4:5]
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execnz .LBB0_197
.LBB0_151:
	s_endpgm
.LBB0_152:
	v_mul_lo_u32 v72, s11, v2
	v_mul_lo_u32 v73, s10, v3
	v_mad_u64_u32 v[68:69], s[4:5], s10, v2, 0
	v_add3_u32 v69, v69, v73, v72
	v_lshl_add_u64 v[68:69], v[68:69], 2, s[6:7]
	v_lshl_add_u64 v[68:69], v[0:1], 2, v[68:69]
	global_store_dword v[68:69], v62, off offset:64
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[4:5], s[14:15], s[0:1]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_79
.LBB0_153:
	v_mul_lo_u32 v62, s11, v4
	v_mul_lo_u32 v72, s10, v5
	v_mad_u64_u32 v[68:69], s[4:5], s10, v4, 0
	v_add3_u32 v69, v69, v72, v62
	v_lshl_add_u64 v[68:69], v[68:69], 2, s[6:7]
	v_lshl_add_u64 v[68:69], v[0:1], 2, v[68:69]
	global_store_dword v[68:69], v63, off offset:64
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[4:5], s[16:17], s[0:1]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_80
.LBB0_154:
	v_mul_lo_u32 v68, s11, v66
	v_mul_lo_u32 v69, s10, v67
	v_mad_u64_u32 v[62:63], s[4:5], s10, v66, 0
	v_add3_u32 v63, v63, v69, v68
	v_lshl_add_u64 v[62:63], v[62:63], 2, s[6:7]
	v_lshl_add_u64 v[62:63], v[0:1], 2, v[62:63]
	global_store_dword v[62:63], v64, off offset:64
	s_or_b64 exec, exec, s[2:3]
	s_and_b64 s[4:5], s[18:19], s[0:1]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execnz .LBB0_81
	s_branch .LBB0_82
.LBB0_155:
	v_mul_lo_u32 v64, s11, v2
	v_mul_lo_u32 v65, s10, v3
	v_mad_u64_u32 v[62:63], s[20:21], s10, v2, 0
	v_add3_u32 v63, v63, v65, v64
	v_lshl_add_u64 v[62:63], v[62:63], 2, s[6:7]
	v_lshl_add_u64 v[62:63], v[0:1], 2, v[62:63]
	global_store_dword v[62:63], v58, off offset:128
	s_or_b64 exec, exec, s[4:5]
	s_and_b64 s[20:21], s[14:15], s[2:3]
	s_and_saveexec_b64 s[4:5], s[20:21]
	s_cbranch_execz .LBB0_84
.LBB0_156:
	v_mul_lo_u32 v58, s11, v4
	v_mul_lo_u32 v64, s10, v5
	v_mad_u64_u32 v[62:63], s[20:21], s10, v4, 0
	v_add3_u32 v63, v63, v64, v58
	v_lshl_add_u64 v[62:63], v[62:63], 2, s[6:7]
	v_lshl_add_u64 v[62:63], v[0:1], 2, v[62:63]
	global_store_dword v[62:63], v59, off offset:128
	s_or_b64 exec, exec, s[4:5]
	s_and_b64 s[20:21], s[16:17], s[2:3]
	s_and_saveexec_b64 s[4:5], s[20:21]
	s_cbranch_execz .LBB0_85
.LBB0_157:
	v_mul_lo_u32 v62, s11, v66
	v_mul_lo_u32 v63, s10, v67
	v_mad_u64_u32 v[58:59], s[20:21], s10, v66, 0
	v_add3_u32 v59, v59, v63, v62
	v_lshl_add_u64 v[58:59], v[58:59], 2, s[6:7]
	v_lshl_add_u64 v[58:59], v[0:1], 2, v[58:59]
	global_store_dword v[58:59], v60, off offset:128
	s_or_b64 exec, exec, s[4:5]
	s_and_b64 s[20:21], s[18:19], s[2:3]
	s_and_saveexec_b64 s[4:5], s[20:21]
	s_cbranch_execnz .LBB0_86
	s_branch .LBB0_87
.LBB0_158:
	v_mul_lo_u32 v60, s11, v2
	v_mul_lo_u32 v61, s10, v3
	v_mad_u64_u32 v[58:59], s[20:21], s10, v2, 0
	v_add3_u32 v59, v59, v61, v60
	v_lshl_add_u64 v[58:59], v[58:59], 2, s[6:7]
	v_lshl_add_u64 v[58:59], v[0:1], 2, v[58:59]
	global_store_dword v[58:59], v54, off offset:192
	s_or_b64 exec, exec, s[12:13]
	s_and_b64 s[14:15], s[14:15], s[4:5]
	s_and_saveexec_b64 s[12:13], s[14:15]
	s_cbranch_execz .LBB0_89
.LBB0_159:
	v_mul_lo_u32 v54, s11, v4
	v_mul_lo_u32 v58, s10, v5
	v_mad_u64_u32 v[4:5], s[14:15], s10, v4, 0
	v_add3_u32 v5, v5, v58, v54
	v_lshl_add_u64 v[4:5], v[4:5], 2, s[6:7]
	v_lshl_add_u64 v[4:5], v[0:1], 2, v[4:5]
	global_store_dword v[4:5], v55, off offset:192
	s_or_b64 exec, exec, s[12:13]
	s_and_b64 s[14:15], s[16:17], s[4:5]
	s_and_saveexec_b64 s[12:13], s[14:15]
	s_cbranch_execz .LBB0_90
.LBB0_160:
	v_mul_lo_u32 v54, s11, v66
	v_mul_lo_u32 v55, s10, v67
	v_mad_u64_u32 v[4:5], s[14:15], s10, v66, 0
	v_add3_u32 v5, v5, v55, v54
	v_lshl_add_u64 v[4:5], v[4:5], 2, s[6:7]
	v_lshl_add_u64 v[4:5], v[0:1], 2, v[4:5]
	global_store_dword v[4:5], v56, off offset:192
	s_or_b64 exec, exec, s[12:13]
	s_and_b64 s[14:15], s[18:19], s[4:5]
	s_and_saveexec_b64 s[12:13], s[14:15]
	s_cbranch_execnz .LBB0_91
	s_branch .LBB0_92
.LBB0_161:
	v_mul_lo_u32 v52, s11, v56
	v_mul_lo_u32 v60, s10, v57
	v_mad_u64_u32 v[58:59], s[22:23], s10, v56, 0
	v_add3_u32 v59, v59, v60, v52
	v_lshl_add_u64 v[58:59], v[58:59], 2, s[6:7]
	v_lshl_add_u64 v[58:59], v[0:1], 2, v[58:59]
	global_store_dword v[58:59], v53, off
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[12:13], s[0:1]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_100
.LBB0_162:
	v_mul_lo_u32 v58, s11, v4
	v_mul_lo_u32 v59, s10, v5
	v_mad_u64_u32 v[52:53], s[22:23], s10, v4, 0
	v_add3_u32 v53, v53, v59, v58
	v_lshl_add_u64 v[52:53], v[52:53], 2, s[6:7]
	v_lshl_add_u64 v[52:53], v[0:1], 2, v[52:53]
	global_store_dword v[52:53], v46, off offset:64
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[0:1]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_101
.LBB0_163:
	v_mul_lo_u32 v46, s11, v54
	v_mul_lo_u32 v58, s10, v55
	v_mad_u64_u32 v[52:53], s[22:23], s10, v54, 0
	v_add3_u32 v53, v53, v58, v46
	v_lshl_add_u64 v[52:53], v[52:53], 2, s[6:7]
	v_lshl_add_u64 v[52:53], v[0:1], 2, v[52:53]
	global_store_dword v[52:53], v47, off offset:64
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[16:17], s[0:1]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_102
.LBB0_164:
	v_mul_lo_u32 v52, s11, v50
	v_mul_lo_u32 v53, s10, v51
	v_mad_u64_u32 v[46:47], s[22:23], s10, v50, 0
	v_add3_u32 v47, v47, v53, v52
	v_lshl_add_u64 v[46:47], v[46:47], 2, s[6:7]
	v_lshl_add_u64 v[46:47], v[0:1], 2, v[46:47]
	global_store_dword v[46:47], v48, off offset:64
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[18:19], s[0:1]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_103
.LBB0_165:
	v_mul_lo_u32 v48, s11, v56
	v_mul_lo_u32 v52, s10, v57
	v_mad_u64_u32 v[46:47], s[22:23], s10, v56, 0
	v_add3_u32 v47, v47, v52, v48
	v_lshl_add_u64 v[46:47], v[46:47], 2, s[6:7]
	v_lshl_add_u64 v[46:47], v[0:1], 2, v[46:47]
	global_store_dword v[46:47], v49, off offset:64
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[12:13], s[2:3]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_104
.LBB0_166:
	v_mul_lo_u32 v48, s11, v4
	v_mul_lo_u32 v49, s10, v5
	v_mad_u64_u32 v[46:47], s[22:23], s10, v4, 0
	v_add3_u32 v47, v47, v49, v48
	v_lshl_add_u64 v[46:47], v[46:47], 2, s[6:7]
	v_lshl_add_u64 v[46:47], v[0:1], 2, v[46:47]
	global_store_dword v[46:47], v42, off offset:128
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[2:3]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_105
.LBB0_167:
	v_mul_lo_u32 v42, s11, v54
	v_mul_lo_u32 v48, s10, v55
	v_mad_u64_u32 v[46:47], s[22:23], s10, v54, 0
	v_add3_u32 v47, v47, v48, v42
	v_lshl_add_u64 v[46:47], v[46:47], 2, s[6:7]
	v_lshl_add_u64 v[46:47], v[0:1], 2, v[46:47]
	global_store_dword v[46:47], v43, off offset:128
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[16:17], s[2:3]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_106
.LBB0_168:
	v_mul_lo_u32 v46, s11, v50
	v_mul_lo_u32 v47, s10, v51
	v_mad_u64_u32 v[42:43], s[22:23], s10, v50, 0
	v_add3_u32 v43, v43, v47, v46
	v_lshl_add_u64 v[42:43], v[42:43], 2, s[6:7]
	v_lshl_add_u64 v[42:43], v[0:1], 2, v[42:43]
	global_store_dword v[42:43], v44, off offset:128
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[18:19], s[2:3]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_107
.LBB0_169:
	v_mul_lo_u32 v44, s11, v56
	v_mul_lo_u32 v46, s10, v57
	v_mad_u64_u32 v[42:43], s[22:23], s10, v56, 0
	v_add3_u32 v43, v43, v46, v44
	v_lshl_add_u64 v[42:43], v[42:43], 2, s[6:7]
	v_lshl_add_u64 v[42:43], v[0:1], 2, v[42:43]
	global_store_dword v[42:43], v45, off offset:128
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[20:21], s[12:13], s[4:5]
	s_and_saveexec_b64 s[12:13], s[20:21]
	s_cbranch_execz .LBB0_108
.LBB0_170:
	v_mul_lo_u32 v42, s11, v4
	v_mul_lo_u32 v43, s10, v5
	v_mad_u64_u32 v[4:5], s[20:21], s10, v4, 0
	v_add3_u32 v5, v5, v43, v42
	v_lshl_add_u64 v[4:5], v[4:5], 2, s[6:7]
	v_lshl_add_u64 v[4:5], v[0:1], 2, v[4:5]
	global_store_dword v[4:5], v38, off offset:192
	s_or_b64 exec, exec, s[12:13]
	s_and_b64 s[14:15], s[14:15], s[4:5]
	s_and_saveexec_b64 s[12:13], s[14:15]
	s_cbranch_execz .LBB0_109
.LBB0_171:
	v_mul_lo_u32 v38, s11, v54
	v_mul_lo_u32 v42, s10, v55
	v_mad_u64_u32 v[4:5], s[14:15], s10, v54, 0
	v_add3_u32 v5, v5, v42, v38
	v_lshl_add_u64 v[4:5], v[4:5], 2, s[6:7]
	v_lshl_add_u64 v[4:5], v[0:1], 2, v[4:5]
	global_store_dword v[4:5], v39, off offset:192
	s_or_b64 exec, exec, s[12:13]
	s_and_b64 s[14:15], s[16:17], s[4:5]
	s_and_saveexec_b64 s[12:13], s[14:15]
	s_cbranch_execz .LBB0_110
.LBB0_172:
	v_mul_lo_u32 v38, s11, v50
	v_mul_lo_u32 v39, s10, v51
	v_mad_u64_u32 v[4:5], s[14:15], s10, v50, 0
	v_add3_u32 v5, v5, v39, v38
	v_lshl_add_u64 v[4:5], v[4:5], 2, s[6:7]
	v_lshl_add_u64 v[4:5], v[0:1], 2, v[4:5]
	global_store_dword v[4:5], v40, off offset:192
	s_or_b64 exec, exec, s[12:13]
	s_and_b64 s[14:15], s[18:19], s[4:5]
	s_and_saveexec_b64 s[12:13], s[14:15]
	s_cbranch_execnz .LBB0_111
	s_branch .LBB0_112
.LBB0_173:
	v_mul_lo_u32 v36, s11, v40
	v_mul_lo_u32 v44, s10, v41
	v_mad_u64_u32 v[42:43], s[22:23], s10, v40, 0
	v_add3_u32 v43, v43, v44, v36
	v_lshl_add_u64 v[42:43], v[42:43], 2, s[6:7]
	v_lshl_add_u64 v[42:43], v[0:1], 2, v[42:43]
	global_store_dword v[42:43], v37, off
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[12:13], s[0:1]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_120
.LBB0_174:
	v_mul_lo_u32 v42, s11, v4
	v_mul_lo_u32 v43, s10, v5
	v_mad_u64_u32 v[36:37], s[22:23], s10, v4, 0
	v_add3_u32 v37, v37, v43, v42
	v_lshl_add_u64 v[36:37], v[36:37], 2, s[6:7]
	v_lshl_add_u64 v[36:37], v[0:1], 2, v[36:37]
	global_store_dword v[36:37], v30, off offset:64
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[0:1]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_121
.LBB0_175:
	v_mul_lo_u32 v30, s11, v38
	v_mul_lo_u32 v42, s10, v39
	v_mad_u64_u32 v[36:37], s[22:23], s10, v38, 0
	v_add3_u32 v37, v37, v42, v30
	v_lshl_add_u64 v[36:37], v[36:37], 2, s[6:7]
	v_lshl_add_u64 v[36:37], v[0:1], 2, v[36:37]
	global_store_dword v[36:37], v31, off offset:64
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[16:17], s[0:1]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_122
.LBB0_176:
	v_mul_lo_u32 v36, s11, v34
	v_mul_lo_u32 v37, s10, v35
	v_mad_u64_u32 v[30:31], s[22:23], s10, v34, 0
	v_add3_u32 v31, v31, v37, v36
	v_lshl_add_u64 v[30:31], v[30:31], 2, s[6:7]
	v_lshl_add_u64 v[30:31], v[0:1], 2, v[30:31]
	global_store_dword v[30:31], v32, off offset:64
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[18:19], s[0:1]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_123
.LBB0_177:
	v_mul_lo_u32 v32, s11, v40
	v_mul_lo_u32 v36, s10, v41
	v_mad_u64_u32 v[30:31], s[22:23], s10, v40, 0
	v_add3_u32 v31, v31, v36, v32
	v_lshl_add_u64 v[30:31], v[30:31], 2, s[6:7]
	v_lshl_add_u64 v[30:31], v[0:1], 2, v[30:31]
	global_store_dword v[30:31], v33, off offset:64
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[12:13], s[2:3]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_124
.LBB0_178:
	v_mul_lo_u32 v32, s11, v4
	v_mul_lo_u32 v33, s10, v5
	v_mad_u64_u32 v[30:31], s[22:23], s10, v4, 0
	v_add3_u32 v31, v31, v33, v32
	v_lshl_add_u64 v[30:31], v[30:31], 2, s[6:7]
	v_lshl_add_u64 v[30:31], v[0:1], 2, v[30:31]
	global_store_dword v[30:31], v26, off offset:128
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[2:3]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_125
.LBB0_179:
	v_mul_lo_u32 v26, s11, v38
	v_mul_lo_u32 v32, s10, v39
	v_mad_u64_u32 v[30:31], s[22:23], s10, v38, 0
	v_add3_u32 v31, v31, v32, v26
	v_lshl_add_u64 v[30:31], v[30:31], 2, s[6:7]
	v_lshl_add_u64 v[30:31], v[0:1], 2, v[30:31]
	global_store_dword v[30:31], v27, off offset:128
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[16:17], s[2:3]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_126
.LBB0_180:
	v_mul_lo_u32 v30, s11, v34
	v_mul_lo_u32 v31, s10, v35
	v_mad_u64_u32 v[26:27], s[22:23], s10, v34, 0
	v_add3_u32 v27, v27, v31, v30
	v_lshl_add_u64 v[26:27], v[26:27], 2, s[6:7]
	v_lshl_add_u64 v[26:27], v[0:1], 2, v[26:27]
	global_store_dword v[26:27], v28, off offset:128
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[18:19], s[2:3]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_127
.LBB0_181:
	v_mul_lo_u32 v28, s11, v40
	v_mul_lo_u32 v30, s10, v41
	v_mad_u64_u32 v[26:27], s[22:23], s10, v40, 0
	v_add3_u32 v27, v27, v30, v28
	v_lshl_add_u64 v[26:27], v[26:27], 2, s[6:7]
	v_lshl_add_u64 v[26:27], v[0:1], 2, v[26:27]
	global_store_dword v[26:27], v29, off offset:128
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[20:21], s[12:13], s[4:5]
	s_and_saveexec_b64 s[12:13], s[20:21]
	s_cbranch_execz .LBB0_128
.LBB0_182:
	v_mul_lo_u32 v26, s11, v4
	v_mul_lo_u32 v27, s10, v5
	v_mad_u64_u32 v[4:5], s[20:21], s10, v4, 0
	v_add3_u32 v5, v5, v27, v26
	v_lshl_add_u64 v[4:5], v[4:5], 2, s[6:7]
	v_lshl_add_u64 v[4:5], v[0:1], 2, v[4:5]
	global_store_dword v[4:5], v22, off offset:192
	s_or_b64 exec, exec, s[12:13]
	s_and_b64 s[14:15], s[14:15], s[4:5]
	s_and_saveexec_b64 s[12:13], s[14:15]
	s_cbranch_execz .LBB0_129
.LBB0_183:
	v_mul_lo_u32 v22, s11, v38
	v_mul_lo_u32 v26, s10, v39
	v_mad_u64_u32 v[4:5], s[14:15], s10, v38, 0
	v_add3_u32 v5, v5, v26, v22
	v_lshl_add_u64 v[4:5], v[4:5], 2, s[6:7]
	v_lshl_add_u64 v[4:5], v[0:1], 2, v[4:5]
	global_store_dword v[4:5], v23, off offset:192
	s_or_b64 exec, exec, s[12:13]
	s_and_b64 s[14:15], s[16:17], s[4:5]
	s_and_saveexec_b64 s[12:13], s[14:15]
	s_cbranch_execz .LBB0_130
.LBB0_184:
	v_mul_lo_u32 v22, s11, v34
	v_mul_lo_u32 v23, s10, v35
	v_mad_u64_u32 v[4:5], s[14:15], s10, v34, 0
	v_add3_u32 v5, v5, v23, v22
	v_lshl_add_u64 v[4:5], v[4:5], 2, s[6:7]
	v_lshl_add_u64 v[4:5], v[0:1], 2, v[4:5]
	global_store_dword v[4:5], v24, off offset:192
	s_or_b64 exec, exec, s[12:13]
	s_and_b64 s[14:15], s[18:19], s[4:5]
	s_and_saveexec_b64 s[12:13], s[14:15]
	s_cbranch_execnz .LBB0_131
	s_branch .LBB0_132
.LBB0_185:
	v_mul_lo_u32 v20, s11, v2
	v_mul_lo_u32 v26, s10, v3
	v_mad_u64_u32 v[24:25], s[20:21], s10, v2, 0
	v_add3_u32 v25, v25, v26, v20
	v_lshl_add_u64 v[24:25], v[24:25], 2, s[6:7]
	v_lshl_add_u64 v[24:25], v[0:1], 2, v[24:25]
	global_store_dword v[24:25], v21, off
	s_or_b64 exec, exec, s[18:19]
	s_and_b64 s[20:21], s[12:13], s[0:1]
	s_and_saveexec_b64 s[18:19], s[20:21]
	s_cbranch_execz .LBB0_140
.LBB0_186:
	v_mul_lo_u32 v24, s11, v4
	v_mul_lo_u32 v25, s10, v5
	v_mad_u64_u32 v[20:21], s[20:21], s10, v4, 0
	v_add3_u32 v21, v21, v25, v24
	v_lshl_add_u64 v[20:21], v[20:21], 2, s[6:7]
	v_lshl_add_u64 v[20:21], v[0:1], 2, v[20:21]
	global_store_dword v[20:21], v10, off offset:64
	s_or_b64 exec, exec, s[18:19]
	s_and_b64 s[20:21], s[14:15], s[0:1]
	s_and_saveexec_b64 s[18:19], s[20:21]
	s_cbranch_execz .LBB0_141
.LBB0_187:
	v_mul_lo_u32 v10, s11, v22
	v_mul_lo_u32 v24, s10, v23
	v_mad_u64_u32 v[20:21], s[20:21], s10, v22, 0
	v_add3_u32 v21, v21, v24, v10
	v_lshl_add_u64 v[20:21], v[20:21], 2, s[6:7]
	v_lshl_add_u64 v[20:21], v[0:1], 2, v[20:21]
	global_store_dword v[20:21], v11, off offset:64
	s_or_b64 exec, exec, s[18:19]
	s_and_b64 s[20:21], s[16:17], s[0:1]
	s_and_saveexec_b64 s[18:19], s[20:21]
	s_cbranch_execz .LBB0_142
.LBB0_188:
	v_mul_lo_u32 v20, s11, v18
	v_mul_lo_u32 v21, s10, v19
	v_mad_u64_u32 v[10:11], s[20:21], s10, v18, 0
	v_add3_u32 v11, v11, v21, v20
	v_lshl_add_u64 v[10:11], v[10:11], 2, s[6:7]
	v_lshl_add_u64 v[10:11], v[0:1], 2, v[10:11]
	global_store_dword v[10:11], v12, off offset:64
	s_or_b64 exec, exec, s[18:19]
	s_and_b64 s[18:19], s[8:9], s[0:1]
	s_and_saveexec_b64 s[0:1], s[18:19]
	s_cbranch_execz .LBB0_143
.LBB0_189:
	v_mul_lo_u32 v12, s11, v2
	v_mul_lo_u32 v20, s10, v3
	v_mad_u64_u32 v[10:11], s[18:19], s10, v2, 0
	v_add3_u32 v11, v11, v20, v12
	v_lshl_add_u64 v[10:11], v[10:11], 2, s[6:7]
	v_lshl_add_u64 v[10:11], v[0:1], 2, v[10:11]
	global_store_dword v[10:11], v13, off offset:64
	s_or_b64 exec, exec, s[0:1]
	s_and_b64 s[18:19], s[12:13], s[2:3]
	s_and_saveexec_b64 s[0:1], s[18:19]
	s_cbranch_execz .LBB0_144
.LBB0_190:
	v_mul_lo_u32 v12, s11, v4
	v_mul_lo_u32 v13, s10, v5
	v_mad_u64_u32 v[10:11], s[18:19], s10, v4, 0
	v_add3_u32 v11, v11, v13, v12
	v_lshl_add_u64 v[10:11], v[10:11], 2, s[6:7]
	v_lshl_add_u64 v[10:11], v[0:1], 2, v[10:11]
	global_store_dword v[10:11], v14, off offset:128
	s_or_b64 exec, exec, s[0:1]
	s_and_b64 s[18:19], s[14:15], s[2:3]
	s_and_saveexec_b64 s[0:1], s[18:19]
	s_cbranch_execz .LBB0_145
.LBB0_191:
	v_mul_lo_u32 v12, s11, v22
	v_mul_lo_u32 v13, s10, v23
	v_mad_u64_u32 v[10:11], s[18:19], s10, v22, 0
	v_add3_u32 v11, v11, v13, v12
	v_lshl_add_u64 v[10:11], v[10:11], 2, s[6:7]
	v_lshl_add_u64 v[10:11], v[0:1], 2, v[10:11]
	global_store_dword v[10:11], v15, off offset:128
	s_or_b64 exec, exec, s[0:1]
	s_and_b64 s[18:19], s[16:17], s[2:3]
	s_and_saveexec_b64 s[0:1], s[18:19]
	s_cbranch_execz .LBB0_146
.LBB0_192:
	v_mul_lo_u32 v12, s11, v18
	v_mul_lo_u32 v13, s10, v19
	v_mad_u64_u32 v[10:11], s[18:19], s10, v18, 0
	v_add3_u32 v11, v11, v13, v12
	v_lshl_add_u64 v[10:11], v[10:11], 2, s[6:7]
	v_lshl_add_u64 v[10:11], v[0:1], 2, v[10:11]
	global_store_dword v[10:11], v16, off offset:128
	s_or_b64 exec, exec, s[0:1]
	s_and_b64 s[2:3], s[8:9], s[2:3]
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_147
.LBB0_193:
	v_mul_lo_u32 v12, s11, v2
	v_mul_lo_u32 v13, s10, v3
	v_mad_u64_u32 v[10:11], s[2:3], s10, v2, 0
	v_add3_u32 v11, v11, v13, v12
	v_lshl_add_u64 v[10:11], v[10:11], 2, s[6:7]
	v_lshl_add_u64 v[10:11], v[0:1], 2, v[10:11]
	global_store_dword v[10:11], v17, off offset:128
	s_or_b64 exec, exec, s[0:1]
	s_and_b64 s[2:3], s[12:13], s[4:5]
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_148
.LBB0_194:
	v_mul_lo_u32 v10, s11, v4
	v_mul_lo_u32 v11, s10, v5
	v_mad_u64_u32 v[4:5], s[2:3], s10, v4, 0
	v_add3_u32 v5, v5, v11, v10
	v_lshl_add_u64 v[4:5], v[4:5], 2, s[6:7]
	v_lshl_add_u64 v[4:5], v[0:1], 2, v[4:5]
	global_store_dword v[4:5], v6, off offset:192
	s_or_b64 exec, exec, s[0:1]
	s_and_b64 s[2:3], s[14:15], s[4:5]
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_149
.LBB0_195:
	v_mul_lo_u32 v6, s11, v22
	v_mul_lo_u32 v10, s10, v23
	v_mad_u64_u32 v[4:5], s[2:3], s10, v22, 0
	v_add3_u32 v5, v5, v10, v6
	v_lshl_add_u64 v[4:5], v[4:5], 2, s[6:7]
	v_lshl_add_u64 v[4:5], v[0:1], 2, v[4:5]
	global_store_dword v[4:5], v7, off offset:192
	s_or_b64 exec, exec, s[0:1]
	s_and_b64 s[2:3], s[16:17], s[4:5]
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_150
.LBB0_196:
	v_mul_lo_u32 v6, s11, v18
	v_mul_lo_u32 v7, s10, v19
	v_mad_u64_u32 v[4:5], s[2:3], s10, v18, 0
	v_add3_u32 v5, v5, v7, v6
	v_lshl_add_u64 v[4:5], v[4:5], 2, s[6:7]
	v_lshl_add_u64 v[4:5], v[0:1], 2, v[4:5]
	global_store_dword v[4:5], v8, off offset:192
	s_or_b64 exec, exec, s[0:1]
	s_and_b64 s[0:1], s[8:9], s[4:5]
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_151
.LBB0_197:
	v_mul_lo_u32 v4, s11, v2
	v_mul_lo_u32 v5, s10, v3
	v_mad_u64_u32 v[2:3], s[0:1], s10, v2, 0
	v_add3_u32 v3, v3, v5, v4
	v_lshl_add_u64 v[2:3], v[2:3], 2, s[6:7]
	v_lshl_add_u64 v[0:1], v[0:1], 2, v[2:3]
	global_store_dword v[0:1], v9, off offset:192
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wave_bf16_gemm_256x256x64
		.amdhsa_group_segment_fixed_size 69632
		.amdhsa_private_segment_fixed_size 8
		.amdhsa_kernarg_size 48
		.amdhsa_user_sgpr_count 14
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 12
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 1
		.amdhsa_next_free_vgpr 128
		.amdhsa_next_free_sgpr 60
		.amdhsa_accum_offset 128
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
	.size	wave_bf16_gemm_256x256x64, .Lfunc_end0-wave_bf16_gemm_256x256x64

	.set wave_bf16_gemm_256x256x64.num_vgpr, 128
	.set wave_bf16_gemm_256x256x64.num_agpr, 0
	.set wave_bf16_gemm_256x256x64.numbered_sgpr, 60
	.set wave_bf16_gemm_256x256x64.num_named_barrier, 0
	.set wave_bf16_gemm_256x256x64.private_seg_size, 8
	.set wave_bf16_gemm_256x256x64.uses_vcc, 1
	.set wave_bf16_gemm_256x256x64.uses_flat_scratch, 0
	.set wave_bf16_gemm_256x256x64.has_dyn_sized_stack, 0
	.set wave_bf16_gemm_256x256x64.has_recursion, 0
	.set wave_bf16_gemm_256x256x64.has_indirect_call, 0
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
custom.config:
  Source:
    Origin: wave
  Version: 1.0.0
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
    DataType: b
    DestDataType: s
    ComputeDataType: s
    HighPrecisionAccumulate: True
    TransposeA: 1
    TransposeB: 0
    UseBeta: True
    Batched: True
    UseBias: 0
    Activation: False
    UseScaleAlphaVec: 0
  CustomKernel:
    args: [ { type: address, semantic: AddressB },
            { type: address, semantic: AddressA },
            { type: address, semantic: AddressD },
            { type: uint32, semantic: SizeFree1 },
            { type: uint32, semantic: Padding },
            { type: uint32, semantic: SizeFree0 },
            { type: uint32, semantic: Padding },
            { type: uint32, semantic: SizeSum },
            { type: uint32, semantic: Padding } ]
    macrotile: [256, 256, 64]
    threads: [256, 4, 1]
    grid: [TilesY, TilesX, One]
  MatrixInstruction: [16, 16, 32, 1]
  EnableMatrixInstruction: True
  MIWaveTile: [4, 4]
  WavefrontSize: 64
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
      - .actual_access:  write_only
        .address_space:  generic
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
      - .offset:         40
        .size:           4
        .value_kind:     by_value
      - .offset:         44
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 69632
    .kernarg_segment_align: 8
    .kernarg_segment_size: 48
    .max_flat_workgroup_size: 1024
    .name:           wave_bf16_gemm_256x256x64
    .private_segment_fixed_size: 8
    .reqd_workgroup_size:
      - 256
      - 4
      - 1
    .sgpr_count:     66
    .sgpr_spill_count: 0
    .symbol:         wave_bf16_gemm_256x256x64.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     128
    .vgpr_spill_count: 1
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
