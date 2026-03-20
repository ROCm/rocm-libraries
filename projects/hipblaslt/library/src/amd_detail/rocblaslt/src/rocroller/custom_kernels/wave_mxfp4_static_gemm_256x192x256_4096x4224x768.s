; To reproduce the .rocmasm from .optimized.ll, run:
; llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -mattr='-fma-mix-insts' -O3 <.optimized.ll> -o <out.rocmasm>

	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.text
	.globl	wave_mxfp4_static_gemm_256x192x256_4096x4224x768
	.p2align	8
	.type	wave_mxfp4_static_gemm_256x192x256_4096x4224x768,@function
wave_mxfp4_static_gemm_256x192x256_4096x4224x768:
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
.LBB0_0:
	v_and_b32_e32 v99, 0x3ff, v0
	v_bfe_u32 v12, v0, 10, 10
	v_lshrrev_b32_e32 v14, 6, v99
	v_lshlrev_b32_e32 v0, 5, v12
	v_lshl_or_b32 v1, v14, 3, v0
	s_mov_b64 s[24:25], s[2:3]
	v_readfirstlane_b32 s2, v1
	v_lshrrev_b32_e32 v1, 3, v99
	v_or_b32_e32 v2, v1, v0
	s_lshl_b32 s28, s16, 8
	v_or_b32_e32 v2, s28, v2
	v_bitop3_b32 v8, v1, 7, v99 bitop3:0x48
	v_lshlrev_b32_e32 v7, 4, v8
	v_mul_u32_u24_e32 v2, 0x180, v2
	v_or_b32_e32 v3, v2, v7
	v_and_or_b32 v2, v1, 15, v7
	s_and_b32 s3, s25, 0xffff
	s_lshl_b32 s36, s2, 7
	v_mul_lo_u16_e32 v7, 43, v2
	v_lshrrev_b32_e32 v10, 7, v99
	s_or_b32 s25, s3, 0x41800000
	s_mov_b32 s27, 0x27000
	s_mov_b32 s26, 0x7ffffffe
	s_mov_b32 m0, s36
	s_or_b32 s33, s36, 0x2000
	v_lshrrev_b16_e32 v11, 10, v7
	v_bitop3_b32 v7, v1, 48, v0 bitop3:0xc8
	v_lshlrev_b32_e32 v1, 4, v1
	v_lshlrev_b32_e32 v13, 8, v10
	buffer_load_dwordx4 v3, s[24:27], 0 offen lds
	v_add_u32_e32 v4, 0x6000, v3
	s_mov_b32 m0, s33
	s_or_b32 s34, s36, 0x4000
	v_sub_u32_e32 v1, v1, v13
	s_mov_b64 s[20:21], s[6:7]
	buffer_load_dwordx4 v4, s[24:27], 0 offen lds
	v_add_u32_e32 v5, 0xc000, v3
	s_mov_b32 m0, s34
	s_or_b32 s35, s36, 0x6000
	s_mul_i32 s29, s17, 0xc0
	v_lshl_add_u32 v8, v8, 8, v1
	s_movk_i32 s37, 0xfe80
	s_movk_i32 s30, 0x180
	buffer_load_dwordx4 v5, s[24:27], 0 offen lds
	v_add_u32_e32 v6, 0x12000, v3
	s_mov_b32 m0, s35
	v_or3_b32 v9, s29, v11, v7
	v_mad_i32_i24 v1, v11, s37, v8
	s_and_b32 s2, s21, 0xffff
	s_add_i32 s38, s36, 0x10000
	buffer_load_dwordx4 v6, s[24:27], 0 offen lds
	s_or_b32 s21, s2, 0x41800000
	s_mov_b32 s22, s26
	s_mov_b32 s23, s27
	v_mad_u32_u24 v9, v9, s30, v1
	s_mov_b32 m0, s38
	s_add_i32 s39, s33, 0x10000
	buffer_load_dwordx4 v9, s[20:23], 0 offen lds
	v_lshlrev_b32_e32 v9, 4, v10
	v_or_b32_e32 v10, s29, v0
	v_or_b32_e32 v19, v10, v9
	v_or_b32_e32 v0, v19, v11
	v_mad_u32_u24 v0, v0, s30, v1
	v_add_u32_e32 v1, 0x6000, v0
	s_mov_b32 m0, s39
	s_add_i32 s40, s34, 0x10000
	buffer_load_dwordx4 v1, s[20:23], 0 offen lds
	v_add_u32_e32 v0, 0xc000, v0
	s_mov_b32 m0, s40
	s_mul_i32 s15, s15, s28
	buffer_load_dwordx4 v0, s[20:23], 0 offen lds
	v_and_b32_e32 v0, 63, v99
	s_mul_hi_u32 s2, s14, s28
	v_lshrrev_b32_e32 v18, 4, v99
	v_mul_lo_u16_e32 v1, 43, v0
	s_add_i32 s2, s2, s15
	s_mul_i32 s3, s14, s28
	s_load_dwordx2 s[12:13], s[0:1], 0x40
	v_lshrrev_b16_e32 v15, 8, v1
	v_lshlrev_b32_e32 v1, 2, v99
	v_lshlrev_b32_e32 v13, 6, v18
	s_add_u32 s4, s4, s3
	v_bfe_u32 v96, v99, 4, 2
	v_sub_u32_e32 v1, v1, v13
	s_addc_u32 s2, s5, s2
	s_and_b32 s3, s14, 0x3fff
	v_and_b32_e32 v97, 0xc0, v99
	v_lshl_add_u32 v1, v96, 6, v1
	s_movk_i32 s31, 0xffe8
	s_bitset1_b32 s3, 14
	v_or_b32_e32 v11, v97, v15
	v_mad_i32_i24 v16, v15, s31, v1
	s_and_b32 s2, s2, 0xffff
	s_lshl_b32 s3, s3, 16
	s_or_b32 s5, s2, s3
	v_mad_u64_u32 v[20:21], s[2:3], v11, s14, v[16:17]
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s2, s13, s29
	s_mul_hi_u32 s3, s12, s29
	s_lshl_b32 s15, s14, 5
	s_add_i32 s3, s3, s2
	s_mul_i32 s2, s12, s29
	s_add_u32 s16, s8, s2
	s_addc_u32 s2, s9, s3
	s_and_b32 s3, s12, 0x3fff
	v_mul_u32_u24_e32 v98, 0x60, v12
	s_bitset1_b32 s3, 14
	s_mov_b32 s6, s26
	s_mov_b32 s7, s27
	v_add_u32_e32 v17, s15, v20
	v_or_b32_e32 v15, v98, v15
	s_and_b32 s2, s2, 0xffff
	s_lshl_b32 s3, s3, 16
	buffer_load_dword v13, v20, s[4:7], 0 offen
	buffer_load_dword v11, v17, s[4:7], 0 offen
	s_or_b32 s17, s2, s3
	v_mad_u64_u32 v[20:21], s[2:3], s12, v15, v[16:17]
	s_lshl_b32 s13, s12, 5
	s_mov_b32 s18, s26
	s_mov_b32 s19, s27
	v_add_u32_e32 v21, s13, v20
	v_add_u32_e32 v22, s13, v21
	buffer_load_dword v17, v20, s[16:19], 0 offen
	buffer_load_dword v16, v21, s[16:19], 0 offen
	buffer_load_dword v15, v22, s[16:19], 0 offen
	v_cmp_eq_u32_e64 s[2:3], 0, v12
	s_and_b64 vcc, exec, s[2:3]
	s_waitcnt vmcnt(0)
	s_barrier
	s_cbranch_vccnz .LBB0_2
	s_barrier
.LBB0_2:
	s_load_dwordx2 s[8:9], s[0:1], 0x48
	v_and_b32_e32 v100, 7, v99
	v_add_u32_e32 v20, 0x80, v3
	s_add_i32 m0, s36, 0x8000
	s_waitcnt vmcnt(5)
	s_barrier
	buffer_load_dwordx4 v20, s[24:27], 0 offen lds
	v_add_u32_e32 v20, 0x80, v4
	s_add_i32 m0, s33, 0x8000
	s_movk_i32 s0, 0xab
	buffer_load_dwordx4 v20, s[24:27], 0 offen lds
	v_add_u32_e32 v20, 0x80, v5
	s_add_i32 m0, s34, 0x8000
	v_mov_b32_e32 v22, 0x558
	buffer_load_dwordx4 v20, s[24:27], 0 offen lds
	v_add_u32_e32 v20, 0x80, v6
	s_add_i32 m0, s35, 0x8000
	v_mad_legacy_u16 v22, v2, s0, v22
	buffer_load_dwordx4 v20, s[24:27], 0 offen lds
	v_or_b32_e32 v20, 0x80, v2
	v_mul_lo_u16_e32 v20, 0xab, v20
	v_lshrrev_b16_e32 v20, 12, v20
	v_or3_b32 v21, s29, v20, v7
	v_lshrrev_b16_e32 v22, 12, v22
	s_movk_i32 s1, 0x80
	v_mul_u32_u24_e32 v21, 0x180, v21
	v_mad_i32_i24 v22, v22, s37, v8
	v_or_b32_e32 v19, v19, v20
	v_add3_u32 v21, v21, v22, s1
	s_add_i32 m0, s36, 0x16000
	s_mov_b32 s22, s26
	s_mov_b32 s23, s27
	v_mad_u32_u24 v19, v19, s30, v22
	buffer_load_dwordx4 v21, s[20:23], 0 offen lds
	v_add_u32_e32 v20, 0x6080, v19
	s_add_i32 m0, s33, 0x16000
	v_add_u32_e32 v19, 0xc080, v19
	buffer_load_dwordx4 v20, s[20:23], 0 offen lds
	s_add_i32 m0, s34, 0x16000
	s_nop 0
	buffer_load_dwordx4 v19, s[20:23], 0 offen lds
	v_lshlrev_b32_e32 v20, 7, v99
	v_lshlrev_b32_e32 v18, 11, v18
	s_movk_i32 s1, 0x3000
	v_bitop3_b32 v19, v96, v99, 7 bitop3:0x78
	v_sub_u32_e32 v18, v20, v18
	v_mul_lo_u32 v12, v12, s1
	v_lshlrev_b32_e32 v19, 4, v19
	v_add_u32_e32 v101, v18, v12
	v_or_b32_e32 v103, v101, v19
	v_add_u32_e32 v12, 0x10000, v103
	v_lshl_add_u32 v14, v14, 13, v18
	ds_read_b128 v[90:93], v12
	ds_read_b128 v[108:111], v12 offset:2048
	ds_read_b128 v[112:115], v12 offset:4096
	ds_read_b128 v[116:119], v12 offset:6144
	ds_read_b128 v[120:123], v12 offset:8192
	ds_read_b128 v[124:127], v12 offset:10240
	v_or_b32_e32 v102, v14, v19
	ds_read_b128 v[38:41], v102
	ds_read_b128 v[62:65], v102 offset:2048
	ds_read_b128 v[86:89], v102 offset:4096
	ds_read_b128 v[104:107], v102 offset:6144
	s_barrier
	s_setprio 1
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[38:41], v[90:93], 0, v13, v17 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[38:41], v[108:111], 0, v13, v17 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[38:41], v[112:115], 0, v13, v16 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[38:41], v[116:119], 0, v13, v16 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[38:41], v[120:123], 0, v13, v15 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[38:41], v[124:127], 0, v13, v15 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[62:65], v[90:93], 0, v13, v17 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[62:65], v[108:111], 0, v13, v17 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[62:65], v[112:115], 0, v13, v16 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[62:65], v[116:119], 0, v13, v16 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[62:65], v[120:123], 0, v13, v15 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[62:65], v[124:127], 0, v13, v15 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[86:89], v[90:93], 0, v11, v17 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[86:89], v[108:111], 0, v11, v17 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[86:89], v[112:115], 0, v11, v16 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[86:89], v[116:119], 0, v11, v16 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[86:89], v[120:123], 0, v11, v15 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[86:89], v[124:127], 0, v11, v15 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[104:107], v[90:93], 0, v11, v17 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[108:111], v[104:107], v[108:111], 0, v11, v17 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[112:115], v[104:107], v[112:115], 0, v11, v16 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[116:119], v[104:107], v[116:119], 0, v11, v16 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[120:123], v[104:107], v[120:123], 0, v11, v15 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[124:127], v[104:107], v[124:127], 0, v11, v15 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_barrier
	v_or_b32_e32 v12, 64, v0
	v_mul_lo_u16_e32 v104, 43, v12
	v_mov_b32_e32 v12, 0xac
	v_mad_legacy_u16 v12, v0, 43, v12
	v_lshrrev_b16_e32 v12, 8, v12
	v_or_b32_sdwa v94, v97, v104 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_1
	v_mad_i32_i24 v12, v12, s31, v1
	v_mad_u64_u32 v[94:95], s[6:7], v94, s14, v[12:13]
	s_mov_b32 s6, s26
	s_mov_b32 s7, s27
	v_add_u32_e32 v95, s15, v94
	s_nop 1
	buffer_load_dword v164, v94, s[4:7], 0 offen offset:16
	buffer_load_dword v165, v95, s[4:7], 0 offen offset:16
	v_or_b32_sdwa v94, v98, v104 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_1
	v_mad_u64_u32 v[94:95], s[18:19], s12, v94, v[12:13]
	s_mov_b32 s18, s26
	s_mov_b32 s19, s27
	v_add_u32_e32 v12, s13, v94
	v_add_u32_e32 v95, s13, v12
	s_nop 0
	buffer_load_dword v166, v94, s[16:19], 0 offen offset:16
	buffer_load_dword v167, v12, s[16:19], 0 offen offset:16
	buffer_load_dword v168, v95, s[16:19], 0 offen offset:16
	v_bitop3_b32 v12, v96, v100, 4 bitop3:0x36
	v_lshlrev_b32_e32 v12, 4, v12
	v_or_b32_e32 v107, v12, v101
	v_or_b32_e32 v106, v12, v14
	v_add_u32_e32 v12, 0x10000, v107
	ds_read_b128 v[144:147], v12
	ds_read_b128 v[148:151], v12 offset:2048
	ds_read_b128 v[152:155], v12 offset:4096
	ds_read_b128 v[156:159], v12 offset:6144
	ds_read_b128 v[160:163], v12 offset:8192
	ds_read_b128 v[170:173], v12 offset:10240
	ds_read_b128 v[128:131], v106
	ds_read_b128 v[132:135], v106 offset:2048
	ds_read_b128 v[136:139], v106 offset:4096
	ds_read_b128 v[140:143], v106 offset:6144
	s_waitcnt vmcnt(7)
	s_barrier
	s_setprio 1
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[128:131], v[144:147], v[18:21], v13, v17 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[128:131], v[148:151], v[22:25], v13, v17 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[128:131], v[152:155], v[26:29], v13, v16 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[128:131], v[156:159], v[30:33], v13, v16 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[128:131], v[160:163], v[34:37], v13, v15 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[128:131], v[170:173], v[38:41], v13, v15 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[132:135], v[144:147], v[42:45], v13, v17 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[132:135], v[148:151], v[46:49], v13, v17 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[132:135], v[152:155], v[50:53], v13, v16 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[132:135], v[156:159], v[54:57], v13, v16 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[132:135], v[160:163], v[58:61], v13, v15 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[132:135], v[170:173], v[62:65], v13, v15 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[136:139], v[144:147], v[66:69], v11, v17 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[136:139], v[148:151], v[70:73], v11, v17 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[136:139], v[152:155], v[74:77], v11, v16 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[136:139], v[156:159], v[78:81], v11, v16 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[136:139], v[160:163], v[82:85], v11, v15 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[136:139], v[170:173], v[86:89], v11, v15 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[140:143], v[144:147], v[90:93], v11, v17 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[108:111], v[140:143], v[148:151], v[108:111], v11, v17 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[120:123], v[140:143], v[160:163], v[120:123], v11, v15 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[140:143], v[170:173], v[124:127], v11, v15 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[112:115], v[140:143], v[152:155], v[112:115], v11, v16 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[116:119], v[140:143], v[156:159], v[116:119], v11, v16 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	s_mov_b32 m0, s36
	v_add_u32_e32 v3, 0x100, v3
	s_waitcnt vmcnt(5)
	s_barrier
	buffer_load_dwordx4 v3, s[24:27], 0 offen lds
	v_add_u32_e32 v3, 0x100, v4
	s_mov_b32 m0, s33
	s_movk_i32 s1, 0xaab
	buffer_load_dwordx4 v3, s[24:27], 0 offen lds
	v_add_u32_e32 v3, 0x100, v5
	s_mov_b32 m0, s34
	s_nop 0
	buffer_load_dwordx4 v3, s[24:27], 0 offen lds
	v_add_u32_e32 v3, 0x100, v6
	s_mov_b32 m0, s35
	s_nop 0
	buffer_load_dwordx4 v3, s[24:27], 0 offen lds
	v_or_b32_e32 v3, 0x100, v2
	v_mul_u32_u24_sdwa v3, v3, s1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_add_u16_e32 v2, 0x88, v2
	v_or_b32_sdwa v4, s29, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_mul_u32_u24_e32 v2, 0xaab, v2
	v_add_u32_e32 v4, v4, v7
	v_lshrrev_b32_e32 v2, 16, v2
	v_or_b32_sdwa v3, v9, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_mul_u32_u24_e32 v4, 0x180, v4
	v_mad_i32_i24 v2, v2, s37, v8
	s_movk_i32 s1, 0x880
	v_add_u32_e32 v3, v10, v3
	v_add3_u32 v4, v4, v2, s1
	s_mov_b32 m0, s38
	v_mad_u32_u24 v2, v3, s30, v2
	buffer_load_dwordx4 v4, s[20:23], 0 offen lds
	v_add_u32_e32 v3, 0x6880, v2
	s_mov_b32 m0, s39
	v_add_u32_e32 v2, 0xc880, v2
	buffer_load_dwordx4 v3, s[20:23], 0 offen lds
	s_mov_b32 m0, s40
	s_nop 0
	buffer_load_dwordx4 v2, s[20:23], 0 offen lds
	v_add_u32_e32 v2, 0x16000, v103
	ds_read_b128 v[152:155], v2
	ds_read_b128 v[156:159], v2 offset:2048
	ds_read_b128 v[170:173], v2 offset:4096
	ds_read_b128 v[174:177], v2 offset:6144
	ds_read_b128 v[178:181], v2 offset:8192
	ds_read_b128 v[182:185], v2 offset:10240
	ds_read_b128 v[124:127], v102 offset:32768
	ds_read_b128 v[128:131], v102 offset:34816
	ds_read_b128 v[148:151], v102 offset:36864
	ds_read_b128 v[160:163], v102 offset:38912
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(9) lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], v[124:127], v[152:155], v[18:21], v164, v166 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[6:9], v[124:127], v[156:159], v[22:25], v164, v166 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(8)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[124:127], v[170:173], v[26:29], v164, v167 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[124:127], v[174:177], v[30:33], v164, v167 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[124:127], v[178:181], v[34:37], v164, v168 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[124:127], v[182:185], v[38:41], v164, v168 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[128:131], v[152:155], v[42:45], v164, v166 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[128:131], v[156:159], v[46:49], v164, v166 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[128:131], v[170:173], v[50:53], v164, v167 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[128:131], v[174:177], v[54:57], v164, v167 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[128:131], v[178:181], v[58:61], v164, v168 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[124:127], v[128:131], v[182:185], v[62:65], v164, v168 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[128:131], v[148:151], v[152:155], v[66:69], v165, v166 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[132:135], v[148:151], v[156:159], v[70:73], v165, v166 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[136:139], v[148:151], v[170:173], v[74:77], v165, v167 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[140:143], v[148:151], v[174:177], v[78:81], v165, v167 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[144:147], v[148:151], v[178:181], v[82:85], v165, v168 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[148:151], v[148:151], v[182:185], v[86:89], v165, v168 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[152:155], v[160:163], v[152:155], v[90:93], v165, v166 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[156:159], v[160:163], v[156:159], v[108:111], v165, v166 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], v[160:163], v[170:173], v[112:115], v165, v167 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[114:117], v[160:163], v[174:177], v[116:119], v165, v167 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[160:163], v[178:181], v[120:123], v165, v168 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[160:163], v[160:163], v[182:185], v[12:15], v165, v168 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_barrier
	v_or_b32_e32 v10, 0x80, v0
	v_mov_b32_e32 v11, 0x2d6c
	v_mul_lo_u16_e32 v10, 0xab, v10
	v_mad_legacy_u16 v0, v0, s0, v11
	v_lshrrev_b16_e32 v12, 10, v10
	v_lshrrev_b16_e32 v0, 10, v0
	v_or_b32_e32 v10, v97, v12
	v_mad_i32_i24 v0, v0, s31, v1
	v_mad_u64_u32 v[10:11], s[0:1], v10, s14, v[0:1]
	v_add_u32_e32 v1, s15, v10
	buffer_load_dword v108, v10, s[4:7], 0 offen offset:272
	buffer_load_dword v100, v1, s[4:7], 0 offen offset:272
	v_or_b32_e32 v1, v98, v12
	v_mad_u64_u32 v[0:1], s[0:1], s12, v1, v[0:1]
	v_add_u32_e32 v1, s13, v0
	v_add_u32_e32 v10, s13, v1
	buffer_load_dword v105, v0, s[16:19], 0 offen offset:272
	buffer_load_dword v104, v1, s[16:19], 0 offen offset:272
	buffer_load_dword v101, v10, s[16:19], 0 offen offset:272
	v_add_u32_e32 v0, 0x16000, v107
	ds_read_b128 v[182:185], v0
	ds_read_b128 v[186:189], v0 offset:2048
	ds_read_b128 v[190:193], v0 offset:4096
	ds_read_b128 v[194:197], v0 offset:6144
	ds_read_b128 v[198:201], v0 offset:8192
	ds_read_b128 v[202:205], v0 offset:10240
	ds_read_b128 v[10:13], v106 offset:32768
	ds_read_b128 v[170:173], v106 offset:34816
	ds_read_b128 v[174:177], v106 offset:36864
	ds_read_b128 v[178:181], v106 offset:38912
	s_waitcnt vmcnt(7)
	s_barrier
	s_setprio 1
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[10:13], v[182:185], v[2:5], v164, v166 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[10:13], v[186:189], v[6:9], v164, v166 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[84:87], v[10:13], v[190:193], v[16:19], v164, v167 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[80:83], v[10:13], v[194:197], v[20:23], v164, v167 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[10:13], v[198:201], v[24:27], v164, v168 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[10:13], v[202:205], v[28:31], v164, v168 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[170:173], v[182:185], v[32:35], v164, v166 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[170:173], v[186:189], v[36:39], v164, v166 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[170:173], v[190:193], v[40:43], v164, v167 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[170:173], v[194:197], v[44:47], v164, v167 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[170:173], v[198:201], v[48:51], v164, v168 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[170:173], v[202:205], v[124:127], v164, v168 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[174:177], v[182:185], v[128:131], v165, v166 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[174:177], v[186:189], v[132:135], v165, v166 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[174:177], v[190:193], v[136:139], v165, v167 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[174:177], v[194:197], v[140:143], v165, v167 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[174:177], v[198:201], v[144:147], v165, v168 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[174:177], v[202:205], v[148:151], v165, v168 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[178:181], v[182:185], v[152:155], v165, v166 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[178:181], v[186:189], v[156:159], v165, v166 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[178:181], v[190:193], v[110:113], v165, v167 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[178:181], v[194:197], v[114:117], v165, v167 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[178:181], v[198:201], v[118:121], v165, v168 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[178:181], v[202:205], v[160:163], v165, v168 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	s_andn2_b64 vcc, exec, s[2:3]
	s_cbranch_vccnz .LBB0_4
	s_barrier
.LBB0_4:
	v_add_u32_e32 v103, 0x10000, v103
	s_waitcnt vmcnt(5)
	s_barrier
	ds_read_b128 v[138:141], v103
	ds_read_b128 v[142:145], v103 offset:2048
	v_add_u32_e32 v107, 0x10000, v107
	ds_read_b128 v[146:149], v107
	ds_read_b128 v[150:153], v107 offset:2048
	ds_read_b128 v[154:157], v103 offset:4096
	ds_read_b128 v[158:161], v103 offset:6144
	ds_read_b128 v[162:165], v107 offset:4096
	ds_read_b128 v[166:169], v107 offset:6144
	ds_read_b128 v[170:173], v103 offset:8192
	ds_read_b128 v[174:177], v103 offset:10240
	ds_read_b128 v[178:181], v107 offset:8192
	ds_read_b128 v[110:113], v107 offset:10240
	ds_read_b128 v[130:133], v102
	ds_read_b128 v[182:185], v102 offset:2048
	ds_read_b128 v[134:137], v106
	ds_read_b128 v[186:189], v106 offset:2048
	ds_read_b128 v[190:193], v102 offset:4096
	ds_read_b128 v[194:197], v102 offset:6144
	ds_read_b128 v[198:201], v106 offset:4096
	ds_read_b128 v[114:117], v106 offset:6144
	s_waitcnt vmcnt(2) lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[130:133], v[138:141], v[92:95], v108, v105 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_movk_i32 s0, 0x7fff
	s_mul_hi_u32 s1, s8, s28
	s_nop 0
	v_mov_b32_e32 v92, 0x7fc0
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[134:137], v[146:149], v[118:121], v108, v105 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_and_b32_e32 v99, 15, v99
	s_mov_b32 s3, 0x27000
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[130:133], v[142:145], v[88:91], v108, v105 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[84:87], v[130:133], v[154:157], v[84:87], v108, v104 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 2
	v_bfe_u32 v93, v121, 16, 1
	v_bfe_u32 v94, v120, 16, 1
	v_add3_u32 v93, v121, v93, s0
	v_bfe_u32 v95, v119, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[134:137], v[150:153], v[88:91], v108, v105 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_add3_u32 v94, v120, v94, s0
	v_lshrrev_b32_e32 v93, 16, v93
	v_cmp_o_f32_e32 vcc, v121, v121
	v_bfe_u32 v102, v118, 16, 1
	v_add3_u32 v95, v119, v95, s0
	v_lshrrev_b32_e32 v94, 16, v94
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[134:137], v[162:165], v[84:87], v108, v104 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_add3_u32 v102, v118, v102, s0
	s_nop 1
	v_cndmask_b32_e32 v84, v92, v93, vcc
	v_cmp_o_f32_e32 vcc, v120, v120
	s_waitcnt vmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[130:133], v[170:173], v[76:79], v108, v101 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v85, 16, v95
	v_lshrrev_b32_e32 v86, 16, v102
	v_bfe_u32 v87, v88, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[130:133], v[158:161], v[80:83], v108, v104 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 2
	v_cndmask_b32_e32 v80, v92, v94, vcc
	v_cmp_o_f32_e32 vcc, v119, v119
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[130:133], v[174:177], v[72:75], v108, v101 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v83, v91, 16, 1
	v_cndmask_b32_e32 v81, v92, v85, vcc
	v_cmp_o_f32_e32 vcc, v118, v118
	v_bfe_u32 v85, v90, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[134:137], v[178:181], v[76:79], v108, v101 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v82, v92, v86, vcc
	v_bfe_u32 v86, v89, 16, 1
	v_cmp_o_f32_e32 vcc, v91, v91
	v_add3_u32 v76, v88, v87, s0
	v_add3_u32 v77, v89, v86, s0
	v_add3_u32 v79, v91, v83, s0
	v_add3_u32 v78, v90, v85, s0
	v_lshrrev_b32_e32 v79, 16, v79
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], v[134:137], v[110:113], v[72:75], v108, v101 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v83, 16, v76
	v_lshrrev_b32_e32 v78, 16, v78
	v_bfe_u32 v85, v120, 16, 1
	v_lshrrev_b32_e32 v73, 16, v77
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[182:185], v[138:141], v[68:71], v108, v105 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v72, v92, v79, vcc
	v_cmp_o_f32_e32 vcc, v90, v90
	v_bfe_u32 v86, v119, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[134:137], v[166:169], v[126:129], v108, v104 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v68, v92, v78, vcc
	v_cmp_o_f32_e32 vcc, v89, v89
	v_bfe_u32 v71, v125, 16, 1
	v_add3_u32 v71, v125, v71, s0
	v_cndmask_b32_e32 v69, v92, v73, vcc
	v_cmp_o_f32_e32 vcc, v88, v88
	v_bfe_u32 v73, v124, 16, 1
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], v[186:189], v[146:149], v[74:77], v108, v105 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v70, v92, v83, vcc
	v_add3_u32 v73, v124, v73, s0
	v_lshrrev_b32_e32 v71, 16, v71
	v_bfe_u32 v74, v123, 16, 1
	v_cmp_o_f32_e32 vcc, v125, v125
	v_bfe_u32 v75, v122, 16, 1
	v_add3_u32 v74, v123, v74, s0
	v_lshrrev_b32_e32 v73, 16, v73
	v_cndmask_b32_e32 v71, v92, v71, vcc
	v_cmp_o_f32_e32 vcc, v124, v124
	v_add3_u32 v75, v122, v75, s0
	v_lshrrev_b32_e32 v74, 16, v74
	v_cndmask_b32_e32 v73, v92, v73, vcc
	v_cmp_o_f32_e32 vcc, v123, v123
	v_bfe_u32 v76, v129, 16, 1
	v_lshrrev_b32_e32 v75, 16, v75
	v_cndmask_b32_e32 v74, v92, v74, vcc
	v_cmp_o_f32_e32 vcc, v122, v122
	v_bfe_u32 v77, v128, 16, 1
	v_add3_u32 v76, v129, v76, s0
	v_cndmask_b32_e32 v75, v92, v75, vcc
	v_bfe_u32 v78, v127, 16, 1
	v_add3_u32 v77, v128, v77, s0
	v_lshrrev_b32_e32 v76, 16, v76
	v_cmp_o_f32_e32 vcc, v129, v129
	v_bfe_u32 v79, v126, 16, 1
	v_add3_u32 v78, v127, v78, s0
	v_lshrrev_b32_e32 v77, 16, v77
	v_cndmask_b32_e32 v76, v92, v76, vcc
	v_cmp_o_f32_e32 vcc, v128, v128
	v_add3_u32 v79, v126, v79, s0
	v_lshrrev_b32_e32 v78, 16, v78
	v_cndmask_b32_e32 v77, v92, v77, vcc
	v_cmp_o_f32_e32 vcc, v127, v127
	v_bfe_u32 v83, v121, 16, 1
	v_lshrrev_b32_e32 v79, 16, v79
	v_cndmask_b32_e32 v78, v92, v78, vcc
	v_cmp_o_f32_e32 vcc, v126, v126
	v_add3_u32 v83, v121, v83, s0
	v_add3_u32 v85, v120, v85, s0
	v_cndmask_b32_e32 v79, v92, v79, vcc
	v_lshrrev_b32_e32 v83, 16, v83
	v_cmp_o_f32_e32 vcc, v121, v121
	v_bfe_u32 v87, v118, 16, 1
	v_add3_u32 v86, v119, v86, s0
	v_lshrrev_b32_e32 v85, 16, v85
	v_cndmask_b32_e32 v83, v92, v83, vcc
	v_cmp_o_f32_e32 vcc, v120, v120
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[182:185], v[142:145], v[64:67], v108, v105 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v87, v118, v87, s0
	v_lshrrev_b32_e32 v86, 16, v86
	v_cndmask_b32_e32 v85, v92, v85, vcc
	v_cmp_o_f32_e32 vcc, v119, v119
	v_bfe_u32 v88, v133, 16, 1
	v_lshrrev_b32_e32 v87, 16, v87
	v_cndmask_b32_e32 v86, v92, v86, vcc
	v_cmp_o_f32_e32 vcc, v118, v118
	v_bfe_u32 v89, v132, 16, 1
	v_add3_u32 v88, v133, v88, s0
	v_cndmask_b32_e32 v87, v92, v87, vcc
	v_bfe_u32 v90, v131, 16, 1
	v_add3_u32 v89, v132, v89, s0
	v_lshrrev_b32_e32 v88, 16, v88
	v_cmp_o_f32_e32 vcc, v133, v133
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[190:193], v[138:141], v[44:47], v100, v105 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v91, v130, 16, 1
	v_add3_u32 v90, v131, v90, s0
	v_lshrrev_b32_e32 v89, 16, v89
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[190:193], v[142:145], v[40:43], v100, v105 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v88, v92, v88, vcc
	v_cmp_o_f32_e32 vcc, v132, v132
	v_add3_u32 v91, v130, v91, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[190:193], v[154:157], v[36:39], v100, v104 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v90, 16, v90
	v_cndmask_b32_e32 v89, v92, v89, vcc
	v_cmp_o_f32_e32 vcc, v131, v131
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[190:193], v[158:161], v[32:35], v100, v104 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v93, v137, 16, 1
	v_lshrrev_b32_e32 v91, 16, v91
	v_cndmask_b32_e32 v90, v92, v90, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[190:193], v[170:173], v[28:31], v100, v101 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v130, v130
	v_bfe_u32 v94, v136, 16, 1
	v_add3_u32 v93, v137, v93, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[190:193], v[174:177], v[20:23], v100, v101 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v91, v92, v91, vcc
	v_bfe_u32 v95, v135, 16, 1
	v_add3_u32 v94, v136, v94, s0
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[194:197], v[138:141], v[16:19], v100, v105 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v93, 16, v93
	v_cmp_o_f32_e32 vcc, v137, v137
	v_add3_u32 v95, v135, v95, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[194:197], v[142:145], v[12:15], v100, v105 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v94, 16, v94
	v_cndmask_b32_e32 v93, v92, v93, vcc
	v_cmp_o_f32_e32 vcc, v136, v136
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[194:197], v[154:157], v[8:11], v100, v104 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v95, 16, v95
	v_cndmask_b32_e32 v94, v92, v94, vcc
	v_cmp_o_f32_e32 vcc, v135, v135
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[194:197], v[158:161], v[4:7], v100, v104 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e32 v95, v92, v95, vcc
	v_cmp_o_f32_e32 vcc, v134, v134
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[194:197], v[170:173], v[0:3], v100, v101 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[194:197], v[174:177], v[24:27], v100, v101 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[186:189], v[150:153], v[64:67], v108, v105 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[182:185], v[154:157], v[60:63], v108, v104 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[182:185], v[170:173], v[52:55], v108, v101 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 5
	v_bfe_u32 v102, v66, 16, 1
	v_bfe_u32 v103, v65, 16, 1
	v_add3_u32 v102, v66, v102, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[182:185], v[174:177], v[48:51], v108, v101 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v103, v65, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_lshrrev_b32_e32 v103, 16, v103
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[182:185], v[158:161], v[56:59], v108, v104 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[198:201], v[146:149], v[44:47], v100, v105 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[198:201], v[150:153], v[40:43], v100, v105 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[198:201], v[162:165], v[36:39], v100, v104 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[198:201], v[166:169], v[32:35], v100, v104 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[198:201], v[178:181], v[28:31], v100, v101 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[198:201], v[110:113], v[20:23], v100, v101 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[114:117], v[146:149], v[16:19], v100, v105 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[114:117], v[150:153], v[12:15], v100, v105 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[114:117], v[162:165], v[8:11], v100, v104 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[114:117], v[166:169], v[4:7], v100, v104 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[114:117], v[178:181], v[0:3], v100, v101 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[114:117], v[110:113], v[24:27], v100, v101 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_bfe_u32 v100, v134, 16, 1
	v_add3_u32 v100, v134, v100, s0
	v_lshrrev_b32_e32 v100, 16, v100
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[186:189], v[162:165], v[60:63], v108, v104 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v100, v92, v100, vcc
	v_cmp_o_f32_e32 vcc, v67, v67
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[186:189], v[178:181], v[52:55], v108, v101 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[186:189], v[110:113], v[48:51], v108, v101 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_bfe_u32 v101, v67, 16, 1
	v_add3_u32 v101, v67, v101, s0
	v_lshrrev_b32_e32 v101, 16, v101
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[186:189], v[166:169], v[56:59], v108, v104 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_bfe_u32 v104, v64, 16, 1
	v_cndmask_b32_e32 v67, v92, v101, vcc
	v_cmp_o_f32_e32 vcc, v66, v66
	v_add3_u32 v104, v64, v104, s0
	v_bfe_u32 v101, v63, 16, 1
	v_cndmask_b32_e32 v66, v92, v102, vcc
	v_cmp_o_f32_e32 vcc, v65, v65
	v_lshrrev_b32_e32 v104, 16, v104
	v_bfe_u32 v102, v62, 16, 1
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
	v_bfe_u32 v101, v31, 16, 1
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v33, v92, v103, vcc
	v_cmp_o_f32_e32 vcc, v32, v32
	v_bfe_u32 v102, v30, 16, 1
	v_add3_u32 v101, v31, v101, s0
	v_cndmask_b32_e32 v32, v92, v104, vcc
	v_bfe_u32 v103, v29, 16, 1
	v_add3_u32 v102, v30, v102, s0
	v_lshrrev_b32_e32 v101, 16, v101
	v_cmp_o_f32_e32 vcc, v31, v31
	v_bfe_u32 v104, v28, 16, 1
	v_add3_u32 v103, v29, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cndmask_b32_e32 v31, v92, v101, vcc
	v_cmp_o_f32_e32 vcc, v30, v30
	v_add3_u32 v104, v28, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v30, v92, v102, vcc
	v_cmp_o_f32_e32 vcc, v29, v29
	v_bfe_u32 v101, v23, 16, 1
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v29, v92, v103, vcc
	v_cmp_o_f32_e32 vcc, v28, v28
	v_bfe_u32 v102, v22, 16, 1
	v_add3_u32 v101, v23, v101, s0
	v_cndmask_b32_e32 v28, v92, v104, vcc
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
	v_bfe_u32 v101, v27, 16, 1
	v_cndmask_b32_e32 v1, v92, v103, vcc
	v_cmp_o_f32_e32 vcc, v0, v0
	v_bfe_u32 v102, v26, 16, 1
	v_bfe_u32 v103, v25, 16, 1
	v_cndmask_b32_e32 v0, v92, v104, vcc
	v_bfe_u32 v104, v24, 16, 1
	v_add3_u32 v104, v24, v104, s0
	v_add3_u32 v103, v25, v103, s0
	v_add3_u32 v102, v26, v102, s0
	v_add3_u32 v101, v27, v101, s0
	s_mul_i32 s0, s9, s28
	s_add_i32 s1, s1, s0
	s_mul_i32 s0, s8, s28
	s_lshl_b64 s[0:1], s[0:1], 1
	v_lshrrev_b32_e32 v101, 16, v101
	v_cmp_o_f32_e32 vcc, v27, v27
	s_add_u32 s0, s10, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cndmask_b32_e32 v27, v92, v101, vcc
	v_cmp_o_f32_e32 vcc, v26, v26
	s_addc_u32 s1, s11, s1
	s_lshl_b32 s2, s29, 1
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v26, v92, v102, vcc
	v_cmp_o_f32_e32 vcc, v25, v25
	s_add_u32 s0, s0, s2
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v25, v92, v103, vcc
	v_cmp_o_f32_e32 vcc, v24, v24
	s_addc_u32 s1, s1, 0
	s_and_b32 s2, s8, 0x3fff
	v_cndmask_b32_e32 v24, v92, v104, vcc
	v_lshl_or_b32 v92, v96, 2, v97
	s_lshl_b32 s2, s2, 16
	s_and_b32 s1, s1, 0xffff
	v_lshlrev_b32_e32 v96, 1, v98
	v_mul_lo_u32 v92, s8, v92
	s_or_b32 s1, s2, s1
	v_lshl_add_u32 v96, v99, 1, v96
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
	buffer_store_short v100, v69, s[0:3], 0 offen
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
	buffer_store_short v28, v49, s[0:3], 0 offen offset:128
	buffer_store_short v29, v44, s[0:3], 0 offen offset:128
	buffer_store_short v30, v45, s[0:3], 0 offen offset:128
	buffer_store_short v31, v46, s[0:3], 0 offen offset:128
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
	buffer_store_short v24, v20, s[0:3], 0 offen offset:160
	buffer_store_short v25, v16, s[0:3], 0 offen offset:160
	buffer_store_short v26, v17, s[0:3], 0 offen offset:160
	buffer_store_short v27, v18, s[0:3], 0 offen offset:160
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wave_mxfp4_static_gemm_256x192x256_4096x4224x768
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
	.size	wave_mxfp4_static_gemm_256x192x256_4096x4224x768, .Lfunc_end0-wave_mxfp4_static_gemm_256x192x256_4096x4224x768

	.set wave_mxfp4_static_gemm_256x192x256_4096x4224x768.num_vgpr, 206
	.set wave_mxfp4_static_gemm_256x192x256_4096x4224x768.num_agpr, 0
	.set wave_mxfp4_static_gemm_256x192x256_4096x4224x768.numbered_sgpr, 41
	.set wave_mxfp4_static_gemm_256x192x256_4096x4224x768.num_named_barrier, 0
	.set wave_mxfp4_static_gemm_256x192x256_4096x4224x768.private_seg_size, 0
	.set wave_mxfp4_static_gemm_256x192x256_4096x4224x768.uses_vcc, 1
	.set wave_mxfp4_static_gemm_256x192x256_4096x4224x768.uses_flat_scratch, 0
	.set wave_mxfp4_static_gemm_256x192x256_4096x4224x768.has_dyn_sized_stack, 0
	.set wave_mxfp4_static_gemm_256x192x256_4096x4224x768.has_recursion, 0
	.set wave_mxfp4_static_gemm_256x192x256_4096x4224x768.has_indirect_call, 0
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
    .name:           wave_mxfp4_static_gemm_256x192x256_4096x4224x768
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 256
      - 2
      - 1
    .sgpr_count:     47
    .sgpr_spill_count: 0
    .symbol:         wave_mxfp4_static_gemm_256x192x256_4096x4224x768.kd
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
