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
	s_load_dwordx4 s[4:7], s[0:1], 0x8
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
.LBB0_0:
	v_and_b32_e32 v76, 0x3ff, v0
	v_lshrrev_b32_e32 v1, 3, v76
	v_bfe_u32 v0, v0, 10, 10
	v_lshl_or_b32 v2, v0, 5, v1
	v_lshlrev_b32_e32 v3, 3, v76
	v_lshlrev_b32_e32 v4, 6, v1
	v_lshlrev_b32_e32 v5, 6, v0
	v_sub_u32_e32 v4, v3, v4
	v_and_or_b32 v75, v76, 15, v5
	v_bfe_u32 v77, v76, 4, 2
	v_mul_u32_u24_e32 v2, 0x44, v2
	v_or_b32_e32 v74, 16, v75
	v_or_b32_e32 v73, 32, v75
	v_or_b32_e32 v72, 48, v75
	v_and_b32_e32 v5, 0xcf, v76
	v_or_b32_e32 v6, 48, v76
	v_add_lshl_u32 v78, v2, v4, 1
	v_lshlrev_b32_e32 v2, 4, v77
	s_movk_i32 s0, 0x88
	v_mad_u32_u24 v79, v75, s0, v2
	v_mad_u32_u24 v80, v74, s0, v2
	v_mad_u32_u24 v81, v73, s0, v2
	v_mad_u32_u24 v82, v72, s0, v2
	v_mad_u32_u24 v4, v5, s0, v2
	v_mad_u32_u24 v2, v6, s0, v2
	s_lshl_b32 s0, s9, 20
	v_lshlrev_b32_e32 v5, 17, v0
	v_mul_u32_u24_e32 v6, 0xfc0, v1
	v_or3_b32 v0, s0, v5, v6
	v_mov_b32_e32 v21, 0
	s_lshl_b32 s8, s8, 20
	v_add_lshl_u32 v0, v0, v3, 1
	v_mov_b32_e32 v1, v21
	v_add_u32_e32 v20, 0x100000, v0
	v_lshl_add_u64 v[66:67], s[4:5], 0, v[0:1]
	v_or3_b32 v0, s8, v5, v6
	v_add_lshl_u32 v0, v0, v3, 1
	v_lshl_add_u64 v[64:65], s[4:5], 0, v[20:21]
	v_add_u32_e32 v20, 0x100000, v0
	v_lshl_add_u64 v[68:69], s[2:3], 0, v[20:21]
	v_lshl_add_u64 v[70:71], s[2:3], 0, v[0:1]
	s_mov_b64 s[0:1], 0
	v_add_u32_e32 v83, 0x8800, v4
	v_add_u32_e32 v84, 0x8840, v4
	v_add_u32_e32 v85, 0x9080, v4
	v_add_u32_e32 v86, 0x90c0, v4
	v_add_u32_e32 v87, 0x9900, v4
	v_add_u32_e32 v88, 0x9940, v4
	v_add_u32_e32 v89, 0x8800, v2
	v_add_u32_e32 v90, 0x8840, v2
	v_mov_b32_e32 v20, v21
	v_mov_b32_e32 v22, v21
	v_mov_b32_e32 v23, v21
	v_mov_b32_e32 v60, v21
	v_mov_b32_e32 v61, v21
	v_mov_b32_e32 v62, v21
	v_mov_b32_e32 v63, v21
	v_mov_b32_e32 v56, v21
	v_mov_b32_e32 v57, v21
	v_mov_b32_e32 v58, v21
	v_mov_b32_e32 v59, v21
	v_mov_b32_e32 v52, v21
	v_mov_b32_e32 v53, v21
	v_mov_b32_e32 v54, v21
	v_mov_b32_e32 v55, v21
	v_mov_b32_e32 v48, v21
	v_mov_b32_e32 v49, v21
	v_mov_b32_e32 v50, v21
	v_mov_b32_e32 v51, v21
	v_mov_b32_e32 v40, v21
	v_mov_b32_e32 v41, v21
	v_mov_b32_e32 v42, v21
	v_mov_b32_e32 v43, v21
	v_mov_b32_e32 v32, v21
	v_mov_b32_e32 v33, v21
	v_mov_b32_e32 v34, v21
	v_mov_b32_e32 v35, v21
	v_mov_b32_e32 v16, v21
	v_mov_b32_e32 v17, v21
	v_mov_b32_e32 v18, v21
	v_mov_b32_e32 v19, v21
	v_mov_b32_e32 v28, v21
	v_mov_b32_e32 v29, v21
	v_mov_b32_e32 v30, v21
	v_mov_b32_e32 v31, v21
	v_mov_b32_e32 v12, v21
	v_mov_b32_e32 v13, v21
	v_mov_b32_e32 v14, v21
	v_mov_b32_e32 v15, v21
	v_mov_b32_e32 v4, v21
	v_mov_b32_e32 v5, v21
	v_mov_b32_e32 v6, v21
	v_mov_b32_e32 v7, v21
	v_mov_b32_e32 v0, v21
	v_mov_b32_e32 v2, v21
	v_mov_b32_e32 v3, v21
	v_mov_b32_e32 v44, v21
	v_mov_b32_e32 v45, v21
	v_mov_b32_e32 v46, v21
	v_mov_b32_e32 v47, v21
	v_mov_b32_e32 v36, v21
	v_mov_b32_e32 v37, v21
	v_mov_b32_e32 v38, v21
	v_mov_b32_e32 v39, v21
	v_mov_b32_e32 v24, v21
	v_mov_b32_e32 v25, v21
	v_mov_b32_e32 v26, v21
	v_mov_b32_e32 v27, v21
	v_mov_b32_e32 v8, v21
	v_mov_b32_e32 v9, v21
	v_mov_b32_e32 v10, v21
	v_mov_b32_e32 v11, v21
	.p2align	5, , 4
.LBB0_1:
	v_lshl_add_u64 v[92:93], v[70:71], 0, s[0:1]
	global_load_dwordx4 v[92:95], v[92:93], off
	v_lshl_add_u64 v[96:97], v[68:69], 0, s[0:1]
	v_lshl_add_u64 v[100:101], v[66:67], 0, s[0:1]
	v_lshl_add_u64 v[104:105], v[64:65], 0, s[0:1]
	s_barrier
	global_load_dwordx4 v[96:99], v[96:97], off
	s_nop 0
	global_load_dwordx4 v[100:103], v[100:101], off
	s_nop 0
	global_load_dwordx4 v[104:107], v[104:105], off
	v_add_u32_e32 v91, 0x8800, v78
	s_add_u32 s0, s0, 0x80
	s_addc_u32 s1, s1, 0
	s_cmpk_lg_i32 s0, 0x2000
	s_waitcnt vmcnt(3)
	ds_write2_b64 v91, v[92:93], v[94:95] offset1:1
	v_add_u32_e32 v91, 0xcc00, v78
	s_waitcnt vmcnt(2)
	ds_write2_b64 v91, v[96:97], v[98:99] offset1:1
	s_waitcnt vmcnt(1)
	ds_write2_b64 v78, v[100:101], v[102:103] offset1:1
	v_add_u32_e32 v91, 0x4400, v78
	s_waitcnt vmcnt(0)
	ds_write2_b64 v91, v[104:105], v[106:107] offset1:1
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read2_b64 v[92:95], v83 offset1:1
	ds_read2_b64 v[96:99], v79 offset1:1
	ds_read2_b64 v[100:103], v80 offset1:1
	ds_read2_b64 v[104:107], v81 offset1:1
	ds_read2_b64 v[108:111], v82 offset1:1
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_bf16 v[20:23], v[92:95], v[96:99], v[20:23]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_bf16 v[60:63], v[92:95], v[100:103], v[60:63]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_bf16 v[56:59], v[92:95], v[104:107], v[56:59]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_bf16 v[52:55], v[92:95], v[108:111], v[52:55]
	ds_read2_b64 v[92:95], v85 offset1:1
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_bf16 v[48:51], v[92:95], v[96:99], v[48:51]
	v_mfma_f32_16x16x32_bf16 v[40:43], v[92:95], v[100:103], v[40:43]
	v_mfma_f32_16x16x32_bf16 v[32:35], v[92:95], v[104:107], v[32:35]
	v_mfma_f32_16x16x32_bf16 v[16:19], v[92:95], v[108:111], v[16:19]
	ds_read2_b64 v[92:95], v87 offset1:1
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_bf16 v[28:31], v[92:95], v[96:99], v[28:31]
	v_mfma_f32_16x16x32_bf16 v[12:15], v[92:95], v[100:103], v[12:15]
	v_mfma_f32_16x16x32_bf16 v[4:7], v[92:95], v[104:107], v[4:7]
	v_mfma_f32_16x16x32_bf16 v[0:3], v[92:95], v[108:111], v[0:3]
	ds_read2_b64 v[92:95], v89 offset1:1
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_bf16 v[44:47], v[92:95], v[96:99], v[44:47]
	ds_read2_b64 v[96:99], v84 offset1:1
	v_mfma_f32_16x16x32_bf16 v[36:39], v[92:95], v[100:103], v[36:39]
	ds_read2_b64 v[100:103], v79 offset0:8 offset1:9
	v_mfma_f32_16x16x32_bf16 v[24:27], v[92:95], v[104:107], v[24:27]
	ds_read2_b64 v[104:107], v80 offset0:8 offset1:9
	v_mfma_f32_16x16x32_bf16 v[8:11], v[92:95], v[108:111], v[8:11]
	ds_read2_b64 v[108:111], v81 offset0:8 offset1:9
	ds_read2_b64 v[92:95], v82 offset0:8 offset1:9
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_bf16 v[20:23], v[96:99], v[100:103], v[20:23]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_bf16 v[60:63], v[96:99], v[104:107], v[60:63]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_bf16 v[56:59], v[96:99], v[108:111], v[56:59]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_bf16 v[52:55], v[96:99], v[92:95], v[52:55]
	ds_read2_b64 v[96:99], v86 offset1:1
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_bf16 v[48:51], v[96:99], v[100:103], v[48:51]
	v_mfma_f32_16x16x32_bf16 v[40:43], v[96:99], v[104:107], v[40:43]
	v_mfma_f32_16x16x32_bf16 v[32:35], v[96:99], v[108:111], v[32:35]
	v_mfma_f32_16x16x32_bf16 v[16:19], v[96:99], v[92:95], v[16:19]
	ds_read2_b64 v[96:99], v88 offset1:1
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_bf16 v[28:31], v[96:99], v[100:103], v[28:31]
	v_mfma_f32_16x16x32_bf16 v[12:15], v[96:99], v[104:107], v[12:15]
	v_mfma_f32_16x16x32_bf16 v[4:7], v[96:99], v[108:111], v[4:7]
	v_mfma_f32_16x16x32_bf16 v[0:3], v[96:99], v[92:95], v[0:3]
	ds_read2_b64 v[96:99], v90 offset1:1
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_bf16 v[44:47], v[96:99], v[100:103], v[44:47]
	v_mfma_f32_16x16x32_bf16 v[36:39], v[96:99], v[104:107], v[36:39]
	v_mfma_f32_16x16x32_bf16 v[24:27], v[96:99], v[108:111], v[24:27]
	v_mfma_f32_16x16x32_bf16 v[8:11], v[96:99], v[92:95], v[8:11]
	s_cbranch_scc1 .LBB0_1
	v_lshlrev_b32_e32 v64, 12, v76
	v_and_b32_e32 v64, 0xc0000, v64
	s_lshl_b32 s0, s8, 2
	s_lshl_b32 s1, s9, 10
	v_lshl_or_b32 v64, v77, 14, v64
	s_or_b32 s1, s1, s0
	v_or_b32_e32 v65, v64, v75
	s_add_u32 s0, s6, s1
	s_addc_u32 s1, s7, 0
	v_lshlrev_b32_e32 v65, 2, v65
	global_store_dword v65, v20, s[0:1]
	v_or_b32_e32 v20, 0x1000, v64
	v_or_b32_e32 v66, v20, v75
	v_lshlrev_b32_e32 v66, 2, v66
	global_store_dword v66, v21, s[0:1]
	v_or_b32_e32 v21, 0x2000, v64
	v_or_b32_e32 v66, v21, v75
	v_lshlrev_b32_e32 v66, 2, v66
	global_store_dword v66, v22, s[0:1]
	v_or_b32_e32 v22, 0x3000, v64
	v_or_b32_e32 v66, v22, v75
	v_lshlrev_b32_e32 v66, 2, v66
	global_store_dword v66, v23, s[0:1]
	global_store_dword v65, v60, s[0:1] offset:64
	v_or_b32_e32 v23, v20, v74
	v_lshlrev_b32_e32 v23, 2, v23
	global_store_dword v23, v61, s[0:1]
	v_or_b32_e32 v23, v21, v74
	v_lshlrev_b32_e32 v23, 2, v23
	global_store_dword v23, v62, s[0:1]
	v_or_b32_e32 v23, v22, v74
	v_lshlrev_b32_e32 v23, 2, v23
	global_store_dword v23, v63, s[0:1]
	global_store_dword v65, v56, s[0:1] offset:128
	v_or_b32_e32 v23, v20, v73
	v_lshlrev_b32_e32 v23, 2, v23
	global_store_dword v23, v57, s[0:1]
	v_or_b32_e32 v23, v21, v73
	v_lshlrev_b32_e32 v23, 2, v23
	global_store_dword v23, v58, s[0:1]
	v_or_b32_e32 v23, v22, v73
	v_or_b32_e32 v20, v20, v72
	v_lshlrev_b32_e32 v23, 2, v23
	v_lshlrev_b32_e32 v20, 2, v20
	global_store_dword v23, v59, s[0:1]
	global_store_dword v65, v52, s[0:1] offset:192
	global_store_dword v20, v53, s[0:1]
	v_or_b32_e32 v20, v21, v72
	v_lshlrev_b32_e32 v20, 2, v20
	global_store_dword v20, v54, s[0:1]
	v_or_b32_e32 v20, v22, v72
	v_lshlrev_b32_e32 v20, 2, v20
	global_store_dword v20, v55, s[0:1]
	v_or_b32_e32 v20, 0x10000, v64
	v_or_b32_e32 v21, v20, v75
	v_lshlrev_b32_e32 v21, 2, v21
	global_store_dword v21, v48, s[0:1]
	v_or_b32_e32 v21, 0x11000, v64
	v_or_b32_e32 v22, v21, v75
	v_lshlrev_b32_e32 v22, 2, v22
	global_store_dword v22, v49, s[0:1]
	v_or_b32_e32 v22, 0x12000, v64
	v_or_b32_e32 v23, v22, v75
	v_lshlrev_b32_e32 v23, 2, v23
	global_store_dword v23, v50, s[0:1]
	v_or_b32_e32 v23, 0x13000, v64
	v_or_b32_e32 v48, v23, v75
	v_lshlrev_b32_e32 v48, 2, v48
	global_store_dword v48, v51, s[0:1]
	v_or_b32_e32 v48, v20, v74
	v_lshlrev_b32_e32 v48, 2, v48
	global_store_dword v48, v40, s[0:1]
	v_or_b32_e32 v40, v21, v74
	v_lshlrev_b32_e32 v40, 2, v40
	global_store_dword v40, v41, s[0:1]
	v_or_b32_e32 v40, v22, v74
	v_lshlrev_b32_e32 v40, 2, v40
	global_store_dword v40, v42, s[0:1]
	v_or_b32_e32 v40, v23, v74
	v_lshlrev_b32_e32 v40, 2, v40
	global_store_dword v40, v43, s[0:1]
	v_or_b32_e32 v40, v20, v73
	v_or_b32_e32 v20, v20, v72
	v_lshlrev_b32_e32 v20, 2, v20
	global_store_dword v20, v16, s[0:1]
	v_or_b32_e32 v16, v21, v72
	v_lshlrev_b32_e32 v16, 2, v16
	global_store_dword v16, v17, s[0:1]
	v_or_b32_e32 v16, v22, v72
	v_lshlrev_b32_e32 v16, 2, v16
	global_store_dword v16, v18, s[0:1]
	v_or_b32_e32 v16, v23, v72
	v_lshlrev_b32_e32 v16, 2, v16
	global_store_dword v16, v19, s[0:1]
	v_or_b32_e32 v16, 0x20000, v64
	v_or_b32_e32 v17, v16, v75
	v_lshlrev_b32_e32 v17, 2, v17
	global_store_dword v17, v28, s[0:1]
	v_or_b32_e32 v17, 0x21000, v64
	v_or_b32_e32 v18, v17, v75
	v_lshlrev_b32_e32 v18, 2, v18
	global_store_dword v18, v29, s[0:1]
	v_or_b32_e32 v18, 0x22000, v64
	v_or_b32_e32 v19, v18, v75
	v_lshlrev_b32_e32 v19, 2, v19
	global_store_dword v19, v30, s[0:1]
	v_or_b32_e32 v19, 0x23000, v64
	v_or_b32_e32 v20, v19, v75
	v_lshlrev_b32_e32 v20, 2, v20
	global_store_dword v20, v31, s[0:1]
	v_or_b32_e32 v20, v16, v74
	v_lshlrev_b32_e32 v20, 2, v20
	global_store_dword v20, v12, s[0:1]
	v_or_b32_e32 v12, v17, v74
	v_lshlrev_b32_e32 v12, 2, v12
	global_store_dword v12, v13, s[0:1]
	v_or_b32_e32 v12, v18, v74
	v_lshlrev_b32_e32 v12, 2, v12
	global_store_dword v12, v14, s[0:1]
	v_or_b32_e32 v12, v19, v74
	v_lshlrev_b32_e32 v12, 2, v12
	global_store_dword v12, v15, s[0:1]
	v_or_b32_e32 v12, v16, v73
	v_lshlrev_b32_e32 v12, 2, v12
	global_store_dword v12, v4, s[0:1]
	v_or_b32_e32 v4, v17, v73
	v_lshlrev_b32_e32 v4, 2, v4
	global_store_dword v4, v5, s[0:1]
	v_or_b32_e32 v4, v18, v73
	v_lshlrev_b32_e32 v4, 2, v4
	global_store_dword v4, v6, s[0:1]
	v_or_b32_e32 v4, v19, v73
	v_lshlrev_b32_e32 v4, 2, v4
	global_store_dword v4, v7, s[0:1]
	v_or_b32_e32 v4, v16, v72
	v_lshlrev_b32_e32 v4, 2, v4
	global_store_dword v4, v0, s[0:1]
	v_or_b32_e32 v0, v17, v72
	v_lshlrev_b32_e32 v0, 2, v0
	global_store_dword v0, v1, s[0:1]
	v_or_b32_e32 v0, v18, v72
	v_lshlrev_b32_e32 v0, 2, v0
	global_store_dword v0, v2, s[0:1]
	v_or_b32_e32 v0, v19, v72
	v_lshlrev_b32_e32 v0, 2, v0
	global_store_dword v0, v3, s[0:1]
	v_or_b32_e32 v0, 0x30000, v64
	v_or_b32_e32 v1, v0, v75
	v_lshlrev_b32_e32 v1, 2, v1
	global_store_dword v1, v44, s[0:1]
	v_or_b32_e32 v1, 0x31000, v64
	v_or_b32_e32 v2, v1, v75
	v_lshlrev_b32_e32 v2, 2, v2
	global_store_dword v2, v45, s[0:1]
	v_or_b32_e32 v2, 0x32000, v64
	v_or_b32_e32 v3, v2, v75
	v_lshlrev_b32_e32 v3, 2, v3
	global_store_dword v3, v46, s[0:1]
	v_or_b32_e32 v3, 0x33000, v64
	v_or_b32_e32 v4, v3, v75
	v_lshlrev_b32_e32 v4, 2, v4
	global_store_dword v4, v47, s[0:1]
	v_or_b32_e32 v4, v0, v74
	v_lshlrev_b32_e32 v4, 2, v4
	global_store_dword v4, v36, s[0:1]
	v_or_b32_e32 v4, v1, v74
	v_lshlrev_b32_e32 v4, 2, v4
	global_store_dword v4, v37, s[0:1]
	v_or_b32_e32 v4, v2, v74
	v_lshlrev_b32_e32 v4, 2, v4
	global_store_dword v4, v38, s[0:1]
	v_or_b32_e32 v4, v3, v74
	v_lshlrev_b32_e32 v4, 2, v4
	global_store_dword v4, v39, s[0:1]
	v_or_b32_e32 v4, v0, v73
	v_or_b32_e32 v0, v0, v72
	v_lshlrev_b32_e32 v40, 2, v40
	v_lshlrev_b32_e32 v4, 2, v4
	v_lshlrev_b32_e32 v0, 2, v0
	global_store_dword v40, v32, s[0:1]
	v_or_b32_e32 v32, v21, v73
	global_store_dword v4, v24, s[0:1]
	v_or_b32_e32 v4, v1, v73
	global_store_dword v0, v8, s[0:1]
	v_or_b32_e32 v0, v1, v72
	v_lshlrev_b32_e32 v32, 2, v32
	v_lshlrev_b32_e32 v4, 2, v4
	v_lshlrev_b32_e32 v0, 2, v0
	global_store_dword v32, v33, s[0:1]
	v_or_b32_e32 v32, v22, v73
	global_store_dword v4, v25, s[0:1]
	v_or_b32_e32 v4, v2, v73
	global_store_dword v0, v9, s[0:1]
	v_or_b32_e32 v0, v2, v72
	v_lshlrev_b32_e32 v32, 2, v32
	v_lshlrev_b32_e32 v4, 2, v4
	v_lshlrev_b32_e32 v0, 2, v0
	global_store_dword v32, v34, s[0:1]
	v_or_b32_e32 v32, v23, v73
	global_store_dword v4, v26, s[0:1]
	v_or_b32_e32 v4, v3, v73
	global_store_dword v0, v10, s[0:1]
	v_or_b32_e32 v0, v3, v72
	v_lshlrev_b32_e32 v32, 2, v32
	v_lshlrev_b32_e32 v4, 2, v4
	v_lshlrev_b32_e32 v0, 2, v0
	global_store_dword v32, v35, s[0:1]
	global_store_dword v4, v27, s[0:1]
	global_store_dword v0, v11, s[0:1]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wave_bf16_gemm_256x256x64
		.amdhsa_group_segment_fixed_size 69632
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 24
		.amdhsa_user_sgpr_count 8
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 6
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 1
		.amdhsa_next_free_vgpr 112
		.amdhsa_next_free_sgpr 10
		.amdhsa_accum_offset 112
		.amdhsa_reserve_vcc 0
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

	.set wave_bf16_gemm_256x256x64.num_vgpr, 112
	.set wave_bf16_gemm_256x256x64.num_agpr, 0
	.set wave_bf16_gemm_256x256x64.numbered_sgpr, 10
	.set wave_bf16_gemm_256x256x64.num_named_barrier, 0
	.set wave_bf16_gemm_256x256x64.private_seg_size, 0
	.set wave_bf16_gemm_256x256x64.uses_vcc, 0
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
    .group_segment_fixed_size: 69632
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .max_flat_workgroup_size: 1024
    .name:           wave_bf16_gemm_256x256x64
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 256
      - 4
      - 1
    .sgpr_count:     16
    .sgpr_spill_count: 0
    .symbol:         wave_bf16_gemm_256x256x64.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     112
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
