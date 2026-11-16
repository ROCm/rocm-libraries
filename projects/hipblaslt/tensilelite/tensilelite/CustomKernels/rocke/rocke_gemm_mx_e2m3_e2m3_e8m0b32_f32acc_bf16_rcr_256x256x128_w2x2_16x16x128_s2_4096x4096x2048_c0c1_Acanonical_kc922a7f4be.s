// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// MX E2M3 x E2M3 GEMM for gfx950: E8M0 scales per 32 K elements, fp32 accumulation,
// bf16 D rounded to nearest even. 256 threads (wave64, 2x2 waves), macrotile 256x256x128,
// v_mfma_scale_f32_16x16x128_f8f6f4 cbsz:2 blgp:2, 127520 bytes of static LDS.
//
// Fixed problem, enforced by the predicates in custom.config:
//   M = 4096, N = 4096, K = 2048, batch 1, TransA = T, TransB = N, lda = ldb = K, D column-major
//   with ldd = 4096. alpha must be 1 (AssertAlphaValue) and beta 0 (UseBeta false): the
//   kernel takes no scalar arguments and never reads C. The grid must be exactly 256
//   workgroups: every buffer descriptor has num_records 0x7ffffff0 and no load or store is
//   masked, so any other size reads and writes out of range. The K loop and the
//   workgroup-to-tile map are compiled for this shape. Its only arguments are the five
//   pointers in the metadata below, so it cannot take the UserArgs launch path. 10 kernarg dwords are preloaded, which keeps the workgroup id in s12.
//
// Inputs, in hipBLASLt naming. For a matrix X of R rows and K columns in this GEMM (A: R = M,
// B: R = N), with packed row r holding K fp6 codes as a 6-bit LSB-first bitstream:
//   A uses HIPBLASLT_ORDER_COL256_R128_EXT (the kernel's b). B is canonical
//   HIPBLASLT_ORDER_COL with ldb = K (the kernel's a): a dense 6-bit LSB-first
//   bitstream, four values in three bytes, K contiguous.
//   The scales of A and B use HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_128_4_EXT: the scale
//   of row r and K block kb (32 elements) is at ((s*R/128 + r/128)*64 + 16*g + r%16)*8
//   + (r%128)/16, with s = kb/4 and g = kb%4.
//   The kernel's a/sa operands are hipBLASLt B and its scale, b/sb are A and its scale.
//   tiled A and canonical B.
//
// Edits to the provider's instruction listing, none of which changes an instruction up to
// s_endpgm:
//   - the address and encoding comments on each instruction are removed;
//   - each PC-relative branch immediate is replaced by a label at the same instruction;
//   - the provider's s_nop padding instructions after s_endpgm are dropped;
//   - the directives, kernel descriptor and metadata are written for code object v4
//     (amdhsa.version [1, 1], no hidden arguments) from the provider's kernel descriptor.
//     .amdhsa_kernarg_preload_length stays 10 so the workgroup id stays in s12.
.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.amdhsa_code_object_version 4
.text
.globl rocke_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s2_4096x4096x2048_c0c1_Acanonical_kc922a7f4be
.p2align 8
.type rocke_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s2_4096x4096x2048_c0c1_Acanonical_kc922a7f4be,@function
rocke_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s2_4096x4096x2048_c0c1_Acanonical_kc922a7f4be:
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_waitcnt lgkmcnt(0)
	s_branch .Lbranch_1900
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
.Lbranch_1900:
	v_lshrrev_b32_e32 v0, 6, v0
	s_nop 0
	v_readfirstlane_b32 s0, v0
	v_mbcnt_lo_u32_b32 v0, -1, 0
	v_mbcnt_hi_u32_b32 v0, -1, v0
	s_lshr_b32 s1, s0, 1
	s_and_b32 s13, s0, 1
	s_and_b32 s14, s12, 7
	s_lshr_b32 s12, s12, 3
	s_lshr_b32 s15, s14, 1
	s_lshl_b32 s15, s15, 2
	s_lshr_b32 s16, s12, 3
	s_add_i32 s15, s15, s16
	s_and_b32 s14, s14, 1
	s_lshl_b32 s14, s14, 3
	s_and_b32 s12, s12, 7
	s_add_i32 s12, s14, s12
	s_lshl_b32 s14, s15, 8
	s_mul_i32 s15, s14, 0x600
	s_lshl_b32 s16, s12, 8
	s_lshl_b32 s1, s1, 7
	s_lshl_b32 s13, s13, 7
	s_add_u32 s20, s2, s15
	s_mul_i32 s2, s0, 0x600
	s_mul_i32 s15, s0, 0x810
	s_addc_u32 s21, s3, 0
	s_lshl_b32 s3, s1, 3
	s_lshl_b32 s17, s0, 6
	v_and_b32_e32 v1, 15, v0
	v_lshrrev_b32_e32 v2, 4, v0
	v_and_b32_e32 v3, 3, v1
	v_add_u32_e32 v4, s17, v0
	v_lshrrev_b32_e32 v5, 2, v1
	v_lshlrev_b32_e32 v3, 5, v3
	v_mul_u32_u24_e32 v6, 24, v2
	v_and_b32_e32 v7, 15, v4
	v_lshrrev_b32_e32 v4, 4, v4
	v_cmp_ge_i32_e64 s[18:19], v2, 1
	v_cmp_ge_i32_e64 s[22:23], v2, 2
	v_cmp_ge_i32_e64 s[24:25], v2, 3
	s_mov_b32 s17, 0x808
	v_mad_u32_u24 v3, s17, v5, v3
	v_lshlrev_b32_e32 v4, 7, v4
	v_lshrrev_b32_e32 v5, 2, v7
	s_add_i32 s26, s2, 0x1800
	v_and_b32_e32 v8, 7, v0
	v_add3_u32 v3, s3, v6, v3
	v_mad_u32_u24 v5, s17, v5, v4
	v_and_b32_e32 v9, 3, v7
	v_lshrrev_b32_e32 v10, 3, v0
	v_lshlrev_b32_e32 v8, 4, v8
	v_add_u32_e32 v11, 0x101a0, v3
	v_add_u32_e32 v3, 0x80a0, v3
	s_mov_b32 s17, 0x810
	v_mad_u32_u24 v4, s17, v7, v4
	v_lshl_add_u32 v5, v9, 5, v5
	s_add_i32 s17, s15, 0x2040
	s_add_i32 s27, s2, 0x3000
	s_mov_b32 s28, 0x6000
	v_mad_u32_u24 v7, s28, v10, v8
	v_add_u32_e32 v4, 0x60, v4
	v_add_u32_e32 v5, 0x10200, v5
	s_add_i32 s28, s15, 0x4080
	s_add_i32 s29, s2, 0x4800
	s_add_i32 s30, s15, 0x60c0
	s_add_i32 s31, s2, 0x30000
	s_add_i32 s32, s15, 0x400
	s_add_i32 s33, s0, 20
	s_add_i32 s34, s0, 4
	s_lshr_b32 s33, s33, 4
	s_lshl_b32 s33, s33, 3
	s_lshl_b32 s35, s33, 4
	s_add_i32 s35, s34, s35
	s_mul_i32 s34, s34, 0x810
	s_lshl_b32 s33, s33, 7
	s_add_i32 s33, s34, s33
	s_add_i32 s34, s0, 24
	s_add_i32 s36, s0, 8
	s_lshr_b32 s34, s34, 4
	s_lshl_b32 s34, s34, 3
	s_lshl_b32 s37, s34, 4
	s_add_i32 s37, s36, s37
	s_mul_i32 s36, s36, 0x810
	s_lshl_b32 s34, s34, 7
	s_add_i32 s34, s36, s34
	s_add_i32 s36, s0, 28
	s_add_i32 s38, s0, 12
	s_lshr_b32 s36, s36, 4
	s_lshl_b32 s36, s36, 3
	s_lshl_b32 s39, s36, 4
	s_add_i32 s39, s38, s39
	s_mul_i32 s38, s38, 0x810
	s_lshl_b32 s36, s36, 7
	s_mov_b32 s40, 0x810
	v_mad_u32_u24 v6, s40, v1, v6
	s_add_i32 s36, s38, s36
	v_add_u32_e32 v6, s3, v6
	v_add_u32_e32 v8, 0x8080, v6
	v_add_u32_e32 v9, 0xffff7e80, v6
	v_cndmask_b32_e64 v10, v6, v8, s[22:23]
	v_cndmask_b32_e64 v12, v6, v9, s[22:23]
	v_cndmask_b32_e64 v13, v6, v8, s[24:25]
	v_cndmask_b32_e64 v14, v6, v9, s[24:25]
	v_cndmask_b32_e64 v15, v11, v8, s[18:19]
	v_cndmask_b32_e64 v16, v3, v9, s[18:19]
	v_cndmask_b32_e64 v8, v11, v8, s[22:23]
	v_cndmask_b32_e64 v3, v3, v9, s[22:23]
	v_lshrrev_b32_e32 v4, 2, v4
	v_lshrrev_b32_e32 v5, 2, v5
	s_mul_i32 s3, s12, 0x60000
	s_add_u32 s40, s4, s3
	s_addc_u32 s41, s5, 0
	s_lshr_b32 s3, s13, 4
	s_and_b32 s41, s41, 0xffff
	s_add_i32 s4, s14, s1
	s_add_i32 s5, s16, s13
	s_and_b32 s45, s7, 0xffff
	s_and_b32 s49, s9, 0xffff
	s_mul_i32 s7, s35, 0x600
	s_lshl_b32 s9, s0, 8
	s_lshr_b32 s12, s14, 7
	s_lshr_b32 s14, s16, 7
	s_lshr_b32 s1, s1, 7
	s_lshr_b32 s13, s13, 7
	s_add_i32 s16, 0xd000, s15
	s_mul_i32 s18, s37, 0x600
	s_and_b32 s21, s21, 0xffff
	s_mul_i32 s19, s39, 0x600
	v_lshrrev_b32_e32 v6, 2, v6
	v_lshrrev_b32_e32 v9, 2, v10
	v_lshrrev_b32_e32 v10, 2, v12
	v_lshrrev_b32_e32 v11, 2, v13
	v_lshrrev_b32_e32 v12, 2, v14
	v_lshrrev_b32_e32 v13, 2, v15
	v_lshrrev_b32_e32 v14, 2, v16
	v_lshrrev_b32_e32 v8, 2, v8
	v_lshrrev_b32_e32 v3, 2, v3
	v_lshlrev_b32_e32 v15, 4, v0
	s_mul_i32 s24, s0, 0x1800
	s_mov_b32 s42, 0x7ffffff0
	s_mov_b32 s43, 0x31014000
	s_mov_b32 s44, s6
	s_mov_b32 s46, 0x7ffffff0
	s_mov_b32 s47, 0x31014000
	s_mov_b32 s48, s8
	s_mov_b32 s50, 0x7ffffff0
	s_mov_b32 s51, 0x31014000
	v_lshlrev_b32_e32 v16, 2, v0
	s_mov_b32 s22, 0x7ffffff0
	s_mov_b32 m0, s16
	s_mov_b32 s23, 0x31014000
	buffer_load_dwordx4 v7, s[20:23], s2 offen lds
	s_add_i32 s6, 0xd000, s17
	s_mov_b32 m0, s6
	s_nop 0
	buffer_load_dwordx4 v7, s[20:23], s26 offen lds
	s_add_i32 s8, 0xd000, s28
	s_mov_b32 m0, s8
	s_nop 0
	buffer_load_dwordx4 v7, s[20:23], s27 offen lds
	s_add_i32 s25, 0xd000, s30
	s_mov_b32 m0, s25
	s_nop 0
	buffer_load_dwordx4 v7, s[20:23], s29 offen lds
	s_add_i32 s35, 0xd000, s32
	s_mov_b32 m0, s35
	s_nop 0
	buffer_load_dwordx4 v7, s[20:23], s31 offen lds
	s_add_i32 s37, 0xd000, s33
	s_mov_b32 m0, s37
	s_nop 0
	buffer_load_dwordx4 v7, s[20:23], s7 offen lds
	s_add_i32 s38, 0xd000, s34
	s_mov_b32 m0, s38
	s_nop 0
	buffer_load_dwordx4 v7, s[20:23], s18 offen lds
	s_add_i32 s39, 0xd000, s36
	s_mov_b32 m0, s39
	s_nop 0
	buffer_load_dwordx4 v7, s[20:23], s19 offen lds
	s_lshl_b32 s52, s12, 9
	s_lshl_b32 s14, s14, 9
	s_add_i32 s53, 0x1000, s24
	s_mov_b32 m0, s53
	s_nop 0
	buffer_load_dwordx4 v15, s[40:43], s24 offen lds
	buffer_load_dwordx4 v15, s[40:43], s24 offen offset:1024 lds
	buffer_load_dwordx4 v15, s[40:43], s24 offen offset:2048 lds
	buffer_load_dwordx4 v15, s[40:43], s24 offen offset:3072 lds
	s_add_i32 s54, s24, 0x1000
	s_add_i32 s55, 0x1000, s54
	s_mov_b32 m0, s55
	s_nop 0
	buffer_load_dwordx4 v15, s[40:43], s54 offen lds
	buffer_load_dwordx4 v15, s[40:43], s54 offen offset:1024 lds
	s_mov_b32 m0, s9
	s_add_i32 s52, s9, s52
	buffer_load_dword v16, s[44:47], s52 offen lds
	s_add_i32 s52, 0x400, s9
	s_mov_b32 m0, s52
	s_add_i32 s14, s9, s14
	buffer_load_dword v16, s[48:51], s14 offen lds
	s_add_i32 s52, s2, 0x80
	s_add_i32 s15, s15, 0x8100
	s_add_i32 s15, 0xd000, s15
	s_mov_b32 m0, s15
	s_nop 0
	buffer_load_dwordx4 v7, s[20:23], s52 offen lds
	s_add_i32 s52, s26, 0x80
	s_add_i32 s17, s17, 0x8100
	s_add_i32 s17, 0xd000, s17
	s_mov_b32 m0, s17
	s_nop 0
	buffer_load_dwordx4 v7, s[20:23], s52 offen lds
	s_add_i32 s52, s27, 0x80
	s_add_i32 s28, s28, 0x8100
	s_add_i32 s28, 0xd000, s28
	s_mov_b32 m0, s28
	s_nop 0
	buffer_load_dwordx4 v7, s[20:23], s52 offen lds
	s_add_i32 s52, s29, 0x80
	s_add_i32 s30, s30, 0x8100
	s_add_i32 s30, 0xd000, s30
	s_mov_b32 m0, s30
	s_nop 0
	buffer_load_dwordx4 v7, s[20:23], s52 offen lds
	s_add_i32 s52, s31, 0x80
	s_add_i32 s32, s32, 0x8100
	s_add_i32 s32, 0xd000, s32
	s_mov_b32 m0, s32
	s_nop 0
	buffer_load_dwordx4 v7, s[20:23], s52 offen lds
	s_add_i32 s52, s7, 0x80
	s_add_i32 s33, s33, 0x8100
	s_add_i32 s33, 0xd000, s33
	s_mov_b32 m0, s33
	s_nop 0
	buffer_load_dwordx4 v7, s[20:23], s52 offen lds
	s_add_i32 s52, s18, 0x80
	s_add_i32 s34, s34, 0x8100
	s_add_i32 s34, 0xd000, s34
	s_mov_b32 m0, s34
	s_nop 0
	buffer_load_dwordx4 v7, s[20:23], s52 offen lds
	s_add_i32 s52, s19, 0x80
	s_add_i32 s36, s36, 0x8100
	s_add_i32 s36, 0xd000, s36
	s_mov_b32 m0, s36
	s_nop 0
	buffer_load_dwordx4 v7, s[20:23], s52 offen lds
	s_add_i32 s52, s12, 32
	s_lshl_b32 s52, s52, 9
	s_add_i32 s56, s24, 0x6000
	s_add_i32 s57, 0x7000, s24
	s_mov_b32 m0, s57
	s_nop 0
	buffer_load_dwordx4 v15, s[40:43], s56 offen lds
	buffer_load_dwordx4 v15, s[40:43], s56 offen offset:1024 lds
	buffer_load_dwordx4 v15, s[40:43], s56 offen offset:2048 lds
	buffer_load_dwordx4 v15, s[40:43], s56 offen offset:3072 lds
	s_add_i32 s54, 0x7000, s54
	s_mov_b32 m0, s54
	s_add_i32 s56, s24, 0x7000
	buffer_load_dwordx4 v15, s[40:43], s56 offen lds
	buffer_load_dwordx4 v15, s[40:43], s56 offen offset:1024 lds
	s_add_i32 s56, 0x800, s9
	s_mov_b32 m0, s56
	s_add_i32 s52, s9, s52
	buffer_load_dword v16, s[44:47], s52 offen lds
	s_add_i32 s52, s14, 0x3c00
	buffer_load_dword v16, s[48:51], s52 offen offset:1024 lds
	s_waitcnt vmcnt(16) lgkmcnt(0)
	s_barrier
	s_lshl_b32 s1, s1, 7
	v_lshlrev_b32_e32 v17, 1, v0
	s_lshl_b32 s13, s13, 7
	v_add_lshl_u32 v18, s1, v17, 2
	v_add_lshl_u32 v17, s13, v17, 2
	v_lshlrev_b32_e32 v6, 2, v6
	ds_read_b64 v[20:21], v18
	ds_read_b64 v[22:23], v17 offset:1024
	ds_read_b64 v[24:25], v6 offset:53248
	ds_read_b64 v[26:27], v6 offset:53256
	ds_read_b64 v[28:29], v6 offset:53264
	s_lshl_b32 s1, s3, 6
	v_add_u32_e32 v0, s1, v0
	v_lshlrev_b32_e32 v19, 4, v0
	ds_read_b128 v[30:33], v19 offset:4096
	v_lshlrev_b32_e32 v0, 3, v0
	ds_read_b64 v[34:35], v0 offset:20480
	ds_read_b64 v[36:37], v6 offset:53376
	ds_read_b64 v[38:39], v6 offset:53384
	ds_read_b64 v[40:41], v6 offset:53392
	ds_read_b64 v[42:43], v6 offset:53504
	ds_read_b64 v[44:45], v6 offset:53512
	ds_read_b64 v[46:47], v6 offset:53520
	ds_read_b64 v[48:49], v6 offset:53632
	ds_read_b64 v[50:51], v6 offset:53640
	ds_read_b64 v[52:53], v6 offset:53648
	ds_read_b64 v[54:55], v6 offset:53760
	ds_read_b64 v[56:57], v6 offset:53768
	ds_read_b64 v[58:59], v6 offset:53776
	ds_read_b64 v[60:61], v6 offset:53888
	ds_read_b64 v[62:63], v6 offset:53896
	ds_read_b64 v[64:65], v6 offset:53904
	ds_read_b64 v[66:67], v6 offset:54016
	ds_read_b64 v[68:69], v6 offset:54024
	ds_read_b64 v[70:71], v6 offset:54032
	ds_read_b64 v[72:73], v6 offset:54144
	ds_read_b64 v[74:75], v6 offset:54152
	ds_read_b64 v[76:77], v6 offset:54160
	ds_read_b128 v[78:81], v19 offset:5120
	ds_read_b64 v[82:83], v0 offset:20992
	ds_read_b128 v[84:87], v19 offset:6144
	ds_read_b64 v[88:89], v0 offset:21504
	ds_read_b128 v[90:93], v19 offset:7168
	ds_read_b64 v[94:95], v0 offset:22016
	ds_read_b128 v[96:99], v19 offset:8192
	ds_read_b64 v[100:101], v0 offset:22528
	ds_read_b128 v[102:105], v19 offset:9216
	ds_read_b64 v[106:107], v0 offset:23040
	ds_read_b128 v[108:111], v19 offset:10240
	ds_read_b64 v[112:113], v0 offset:23552
	ds_read_b128 v[114:117], v19 offset:11264
	v_lshlrev_b32_e32 v4, 2, v4
	ds_read_b64 v[118:119], v0 offset:24064
	ds_read_b128 v[120:123], v4 offset:53248
	ds_read_b128 v[124:127], v4 offset:53264
	v_lshlrev_b32_e32 v5, 2, v5
	s_waitcnt lgkmcnt(1)
	ds_write_b64 v5, v[120:121] offset:53248
	ds_write_b64 v5, v[122:123] offset:53256
	s_waitcnt lgkmcnt(2)
	ds_write_b64 v5, v[124:125] offset:53264
	ds_write_b64 v5, v[126:127] offset:53272
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_mov_b32 m0, s16
	s_add_i32 s1, s2, 0x100
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	s_mov_b32 m0, s6
	s_add_i32 s1, s26, 0x100
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	s_mov_b32 m0, s8
	s_add_i32 s1, s27, 0x100
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	s_mov_b32 m0, s25
	s_add_i32 s1, s29, 0x100
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	s_mov_b32 m0, s35
	s_add_i32 s1, s31, 0x100
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	s_mov_b32 m0, s37
	s_add_i32 s1, s7, 0x100
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	s_mov_b32 m0, s38
	s_add_i32 s1, s18, 0x100
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	s_mov_b32 m0, s39
	s_add_i32 s1, s19, 0x100
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	s_add_i32 s1, s12, 64
	s_lshl_b32 s1, s1, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[24:29], v[30:35], 0, v20, v22 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(8) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[120:121], v18 offset:2048
	ds_read_b64 v[122:123], v17 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[36:41], v[30:35], 0, v20, v22 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_i32 s3, s24, 0xc000
	s_mov_b32 m0, s53
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[42:47], v[30:35], 0, v20, v22 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen lds
	v_lshlrev_b32_e32 v8, 2, v8
	ds_read_b64 v[124:125], v8 offset:53344
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[48:53], v[30:35], 0, v20, v22 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_lshlrev_b32_e32 v13, 2, v13
	ds_read_b64 v[126:127], v13 offset:53352
	ds_read_b64 v[128:129], v13 offset:53360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[54:59], v[30:35], 0, v21, v22 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[130:131], v8 offset:53472
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[60:65], v[30:35], 0, v21, v22 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[132:133], v13 offset:53480
	ds_read_b64 v[134:135], v13 offset:53488
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[66:71], v[30:35], 0, v21, v22 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen offset:1024 lds
	ds_read_b64 v[136:137], v8 offset:53600
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[72:77], v[30:35], 0, v21, v22 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[138:139], v13 offset:53608
	ds_read_b64 v[140:141], v13 offset:53616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[72:77], v[78:83], 0, v21, v22 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[30:31], v8 offset:53728
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[66:71], v[78:83], 0, v21, v22 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[32:33], v13 offset:53736
	ds_read_b64 v[34:35], v13 offset:53744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[60:65], v[78:83], 0, v21, v22 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen offset:2048 lds
	ds_read_b64 v[142:143], v8 offset:53856
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[54:59], v[78:83], 0, v21, v22 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[144:145], v13 offset:53864
	ds_read_b64 v[146:147], v13 offset:53872
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[48:53], v[78:83], 0, v20, v22 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[148:149], v8 offset:53984
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[42:47], v[78:83], 0, v20, v22 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[150:151], v13 offset:53992
	ds_read_b64 v[152:153], v13 offset:54000
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[36:41], v[78:83], 0, v20, v22 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen offset:3072 lds
	ds_read_b64 v[154:155], v8 offset:54112
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[24:29], v[78:83], 0, v20, v22 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[156:157], v13 offset:54120
	ds_read_b64 v[158:159], v13 offset:54128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[24:29], v[84:89], 0, v20, v22 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v8 offset:54240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[36:41], v[84:89], 0, v20, v22 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[80:81], v13 offset:54248
	ds_read_b64 v[82:83], v13 offset:54256
	s_add_i32 s3, s24, 0xd000
	s_mov_b32 m0, s55
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[42:47], v[84:89], 0, v20, v22 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen lds
	ds_read_b128 v[160:163], v19 offset:28672
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[48:53], v[84:89], 0, v20, v22 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[164:165], v0 offset:45056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[54:59], v[84:89], 0, v21, v22 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[166:169], v19 offset:29696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[60:65], v[84:89], 0, v21, v22 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[170:171], v0 offset:45568
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[66:71], v[84:89], 0, v21, v22 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen offset:1024 lds
	ds_read_b128 v[172:175], v19 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[72:77], v[84:89], 0, v21, v22 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[176:177], v0 offset:46080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[72:77], v[90:95], 0, v21, v22 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[84:87], v19 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[66:71], v[90:95], 0, v21, v22 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[88:89], v0 offset:46592
	s_add_i32 s1, s9, s1
	s_mov_b32 m0, s9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[60:65], v[90:95], 0, v21, v22 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v16, s[44:47], s1 offen lds
	ds_read_b128 v[178:181], v19 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[54:59], v[90:95], 0, v21, v22 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[182:183], v0 offset:47104
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[48:53], v[90:95], 0, v20, v22 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[184:187], v19 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[42:47], v[90:95], 0, v20, v22 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[188:189], v0 offset:47616
	s_add_i32 s1, s14, 0x7c00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[36:41], v[90:95], 0, v20, v22 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v16, s[48:51], s1 offen offset:1024 lds
	ds_read_b128 v[190:193], v19 offset:34816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[24:29], v[90:95], 0, v20, v22 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[194:195], v0 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[24:29], v[96:101], 0, v20, v23 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[90:93], v19 offset:35840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[36:41], v[96:101], 0, v20, v23 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[94:95], v0 offset:48640
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[42:47], v[96:101], 0, v20, v23 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[48:53], v[96:101], 0, v20, v23 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[54:59], v[96:101], 0, v21, v23 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[60:65], v[96:101], 0, v21, v23 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[66:71], v[96:101], 0, v21, v23 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[72:77], v[96:101], 0, v21, v23 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[72:77], v[102:107], 0, v21, v23 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[66:71], v[102:107], 0, v21, v23 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[60:65], v[102:107], 0, v21, v23 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[54:59], v[102:107], 0, v21, v23 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[48:53], v[102:107], 0, v20, v23 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[42:47], v[102:107], 0, v20, v23 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[36:41], v[102:107], 0, v20, v23 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[24:29], v[102:107], 0, v20, v23 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[24:29], v[108:113], 0, v20, v23 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[36:41], v[108:113], 0, v20, v23 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[42:47], v[108:113], 0, v20, v23 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[48:53], v[108:113], 0, v20, v23 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[54:59], v[108:113], 0, v21, v23 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[60:65], v[108:113], 0, v21, v23 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[66:71], v[108:113], 0, v21, v23 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[72:77], v[108:113], 0, v21, v23 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[72:77], v[114:119], 0, v21, v23 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[66:71], v[114:119], 0, v21, v23 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[60:65], v[114:119], 0, v21, v23 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[54:59], v[114:119], 0, v21, v23 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[48:53], v[114:119], 0, v20, v23 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[42:47], v[114:119], 0, v20, v23 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[36:41], v[114:119], 0, v20, v23 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[24:29], v[114:119], 0, v20, v23 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s1, s12, 0x60
	s_lshl_b32 s1, s1, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[124:129], v[160:165], a[0:3], v120, v122 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[20:21], v18
	ds_read_b64 v[22:23], v17 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[130:135], v[160:165], a[4:7], v120, v122 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_i32 s3, s24, 0x12000
	s_mov_b32 m0, s57
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[136:141], v[160:165], a[8:11], v120, v122 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen lds
	s_mov_b32 s13, 0x10000
	v_lshl_add_u32 v12, v12, 2, s13
	ds_read_b64 v[24:25], v12 offset:20800
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[30:35], v[160:165], a[12:15], v120, v122 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[26:27], v12 offset:20808
	v_lshl_add_u32 v10, v10, 2, s13
	ds_read_b64 v[28:29], v10 offset:20816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[142:147], v[160:165], a[16:19], v121, v122 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[36:37], v12 offset:20928
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[148:153], v[160:165], a[20:23], v121, v122 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[38:39], v12 offset:20936
	ds_read_b64 v[40:41], v10 offset:20944
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[154:159], v[160:165], a[24:27], v121, v122 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen offset:1024 lds
	ds_read_b64 v[42:43], v12 offset:21056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[78:83], v[160:165], a[28:31], v121, v122 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[44:45], v12 offset:21064
	ds_read_b64 v[46:47], v10 offset:21072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[78:83], v[166:171], a[32:35], v121, v122 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[48:49], v12 offset:21184
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[154:159], v[166:171], a[36:39], v121, v122 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[50:51], v12 offset:21192
	ds_read_b64 v[52:53], v10 offset:21200
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[148:153], v[166:171], a[40:43], v121, v122 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen offset:2048 lds
	ds_read_b64 v[54:55], v12 offset:21312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[142:147], v[166:171], a[44:47], v121, v122 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[56:57], v12 offset:21320
	ds_read_b64 v[58:59], v10 offset:21328
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[30:35], v[166:171], a[48:51], v120, v122 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[60:61], v12 offset:21440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[136:141], v[166:171], a[52:55], v120, v122 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[62:63], v12 offset:21448
	ds_read_b64 v[64:65], v10 offset:21456
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[130:135], v[166:171], a[56:59], v120, v122 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen offset:3072 lds
	ds_read_b64 v[66:67], v12 offset:21568
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[124:129], v[166:171], a[60:63], v120, v122 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[68:69], v12 offset:21576
	ds_read_b64 v[70:71], v10 offset:21584
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[124:129], v[172:177], a[64:67], v120, v122 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[72:73], v12 offset:21696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[130:135], v[172:177], a[68:71], v120, v122 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[74:75], v12 offset:21704
	ds_read_b64 v[76:77], v10 offset:21712
	s_add_i32 s3, s24, 0x13000
	s_mov_b32 m0, s54
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[136:141], v[172:177], a[72:75], v120, v122 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen lds
	ds_read_b128 v[96:99], v19 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[30:35], v[172:177], a[76:79], v120, v122 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[100:101], v0 offset:20480
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[142:147], v[172:177], a[80:83], v121, v122 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[102:105], v19 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[148:153], v[172:177], a[84:87], v121, v122 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[106:107], v0 offset:20992
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[154:159], v[172:177], a[88:91], v121, v122 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen offset:1024 lds
	ds_read_b128 v[108:111], v19 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[78:83], v[172:177], a[92:95], v121, v122 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[112:113], v0 offset:21504
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[78:83], v[84:89], a[96:99], v121, v122 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[114:117], v19 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[154:159], v[84:89], a[100:103], v121, v122 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[118:119], v0 offset:22016
	s_add_i32 s1, s9, s1
	s_mov_b32 m0, s56
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[148:153], v[84:89], a[104:107], v121, v122 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v16, s[44:47], s1 offen lds
	ds_read_b128 v[160:163], v19 offset:8192
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[142:147], v[84:89], a[108:111], v121, v122 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[164:165], v0 offset:22528
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[30:35], v[84:89], a[112:115], v120, v122 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[166:169], v19 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[136:141], v[84:89], a[116:119], v120, v122 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[170:171], v0 offset:23040
	s_add_i32 s1, s14, 0xbc00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[130:135], v[84:89], a[120:123], v120, v122 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v16, s[48:51], s1 offen offset:1024 lds
	ds_read_b128 v[172:175], v19 offset:10240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[124:129], v[84:89], a[124:127], v120, v122 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[176:177], v0 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[124:129], v[178:183], a[128:131], v120, v123 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[84:87], v19 offset:11264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[130:135], v[178:183], a[132:135], v120, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[88:89], v0 offset:24064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[136:141], v[178:183], a[136:139], v120, v123 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[30:35], v[178:183], a[140:143], v120, v123 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s1, s2, 0x180
	s_mov_b32 m0, s15
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[142:147], v[178:183], a[144:147], v121, v123 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[148:153], v[178:183], a[148:151], v121, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[154:159], v[178:183], a[152:155], v121, v123 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s1, s26, 0x180
	s_mov_b32 m0, s17
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[78:83], v[178:183], a[156:159], v121, v123 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[78:83], v[184:189], a[160:163], v121, v123 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[154:159], v[184:189], a[164:167], v121, v123 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[148:153], v[184:189], a[168:171], v121, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_i32 s1, s27, 0x180
	s_mov_b32 m0, s28
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[142:147], v[184:189], a[172:175], v121, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[30:35], v[184:189], a[176:179], v120, v123 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[136:141], v[184:189], a[180:183], v120, v123 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s1, s29, 0x180
	s_mov_b32 m0, s30
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[130:135], v[184:189], a[184:187], v120, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[124:129], v[184:189], a[188:191], v120, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[124:129], v[190:195], a[192:195], v120, v123 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[130:135], v[190:195], a[196:199], v120, v123 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s1, s31, 0x180
	s_mov_b32 m0, s32
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[136:141], v[190:195], a[200:203], v120, v123 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[30:35], v[190:195], a[204:207], v120, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[142:147], v[190:195], a[208:211], v121, v123 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s1, s7, 0x180
	s_mov_b32 m0, s33
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[148:153], v[190:195], a[212:215], v121, v123 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[154:159], v[190:195], a[216:219], v121, v123 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[78:83], v[190:195], a[220:223], v121, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[78:83], v[90:95], a[224:227], v121, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s1, s18, 0x180
	s_mov_b32 m0, s34
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[154:159], v[90:95], a[228:231], v121, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[148:153], v[90:95], a[232:235], v121, v123 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[142:147], v[90:95], a[236:239], v121, v123 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s1, s19, 0x180
	s_mov_b32 m0, s36
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[30:35], v[90:95], a[240:243], v120, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[136:141], v[90:95], a[244:247], v120, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[130:135], v[90:95], a[248:251], v120, v123 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[124:129], v[90:95], a[252:255], v120, v123 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s1, s12, 0x80
	s_lshl_b32 s1, s1, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[24:29], v[96:101], a[0:3], v20, v22 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(8) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[30:31], v18 offset:2048
	ds_read_b64 v[32:33], v17 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[36:41], v[96:101], a[4:7], v20, v22 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_i32 s3, s24, 0x18000
	s_mov_b32 m0, s53
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[42:47], v[96:101], a[8:11], v20, v22 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen lds
	ds_read_b64 v[78:79], v6 offset:53280
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[48:53], v[96:101], a[12:15], v20, v22 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[80:81], v6 offset:53288
	ds_read_b64 v[82:83], v6 offset:53296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[54:59], v[96:101], a[16:19], v21, v22 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[90:91], v6 offset:53408
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[60:65], v[96:101], a[20:23], v21, v22 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[92:93], v6 offset:53416
	ds_read_b64 v[94:95], v6 offset:53424
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[66:71], v[96:101], a[24:27], v21, v22 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen offset:1024 lds
	ds_read_b64 v[120:121], v6 offset:53536
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[72:77], v[96:101], a[28:31], v21, v22 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[122:123], v6 offset:53544
	ds_read_b64 v[124:125], v6 offset:53552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[72:77], v[102:107], a[32:35], v21, v22 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[96:97], v6 offset:53664
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[66:71], v[102:107], a[36:39], v21, v22 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[98:99], v6 offset:53672
	ds_read_b64 v[100:101], v6 offset:53680
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[60:65], v[102:107], a[40:43], v21, v22 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen offset:2048 lds
	ds_read_b64 v[126:127], v6 offset:53792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[54:59], v[102:107], a[44:47], v21, v22 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[128:129], v6 offset:53800
	ds_read_b64 v[130:131], v6 offset:53808
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[48:53], v[102:107], a[48:51], v20, v22 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[132:133], v6 offset:53920
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[42:47], v[102:107], a[52:55], v20, v22 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[134:135], v6 offset:53928
	ds_read_b64 v[136:137], v6 offset:53936
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[36:41], v[102:107], a[56:59], v20, v22 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen offset:3072 lds
	ds_read_b64 v[138:139], v6 offset:54048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[24:29], v[102:107], a[60:63], v20, v22 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[140:141], v6 offset:54056
	ds_read_b64 v[142:143], v6 offset:54064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[24:29], v[108:113], a[64:67], v20, v22 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[102:103], v6 offset:54176
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[36:41], v[108:113], a[68:71], v20, v22 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[104:105], v6 offset:54184
	ds_read_b64 v[106:107], v6 offset:54192
	s_add_i32 s3, s24, 0x19000
	s_mov_b32 m0, s55
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[42:47], v[108:113], a[72:75], v20, v22 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen lds
	ds_read_b128 v[144:147], v19 offset:28672
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[48:53], v[108:113], a[76:79], v20, v22 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[148:149], v0 offset:45056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[54:59], v[108:113], a[80:83], v21, v22 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[150:153], v19 offset:29696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[60:65], v[108:113], a[84:87], v21, v22 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[154:155], v0 offset:45568
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[66:71], v[108:113], a[88:91], v21, v22 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen offset:1024 lds
	ds_read_b128 v[178:181], v19 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[72:77], v[108:113], a[92:95], v21, v22 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[182:183], v0 offset:46080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[72:77], v[114:119], a[96:99], v21, v22 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[108:111], v19 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[66:71], v[114:119], a[100:103], v21, v22 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[112:113], v0 offset:46592
	s_add_i32 s1, s9, s1
	s_mov_b32 m0, s9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[60:65], v[114:119], a[104:107], v21, v22 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v16, s[44:47], s1 offen lds
	ds_read_b128 v[184:187], v19 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[54:59], v[114:119], a[108:111], v21, v22 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[188:189], v0 offset:47104
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[48:53], v[114:119], a[112:115], v20, v22 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[190:193], v19 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[42:47], v[114:119], a[116:119], v20, v22 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[194:195], v0 offset:47616
	s_add_i32 s1, s14, 0xfc00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[36:41], v[114:119], a[120:123], v20, v22 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v16, s[48:51], s1 offen offset:1024 lds
	ds_read_b128 v[196:199], v19 offset:34816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[24:29], v[114:119], a[124:127], v20, v22 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[200:201], v0 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[24:29], v[160:165], a[128:131], v20, v23 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[114:117], v19 offset:35840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[36:41], v[160:165], a[132:135], v20, v23 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[118:119], v0 offset:48640
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[42:47], v[160:165], a[136:139], v20, v23 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[48:53], v[160:165], a[140:143], v20, v23 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s1, s2, 0x200
	s_mov_b32 m0, s16
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[54:59], v[160:165], a[144:147], v21, v23 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[60:65], v[160:165], a[148:151], v21, v23 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[66:71], v[160:165], a[152:155], v21, v23 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s1, s26, 0x200
	s_mov_b32 m0, s6
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[72:77], v[160:165], a[156:159], v21, v23 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[72:77], v[166:171], a[160:163], v21, v23 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[66:71], v[166:171], a[164:167], v21, v23 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[60:65], v[166:171], a[168:171], v21, v23 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_i32 s1, s27, 0x200
	s_mov_b32 m0, s8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[54:59], v[166:171], a[172:175], v21, v23 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[48:53], v[166:171], a[176:179], v20, v23 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[42:47], v[166:171], a[180:183], v20, v23 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s1, s29, 0x200
	s_mov_b32 m0, s25
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[36:41], v[166:171], a[184:187], v20, v23 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[24:29], v[166:171], a[188:191], v20, v23 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[24:29], v[172:177], a[192:195], v20, v23 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[36:41], v[172:177], a[196:199], v20, v23 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s1, s31, 0x200
	s_mov_b32 m0, s35
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[42:47], v[172:177], a[200:203], v20, v23 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[48:53], v[172:177], a[204:207], v20, v23 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[54:59], v[172:177], a[208:211], v21, v23 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s1, s7, 0x200
	s_mov_b32 m0, s37
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[60:65], v[172:177], a[212:215], v21, v23 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[66:71], v[172:177], a[216:219], v21, v23 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[72:77], v[172:177], a[220:223], v21, v23 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[72:77], v[84:89], a[224:227], v21, v23 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s1, s18, 0x200
	s_mov_b32 m0, s38
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[66:71], v[84:89], a[228:231], v21, v23 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[60:65], v[84:89], a[232:235], v21, v23 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[54:59], v[84:89], a[236:239], v21, v23 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s1, s19, 0x200
	s_mov_b32 m0, s39
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[48:53], v[84:89], a[240:243], v20, v23 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[42:47], v[84:89], a[244:247], v20, v23 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[36:41], v[84:89], a[248:251], v20, v23 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[24:29], v[84:89], a[252:255], v20, v23 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s1, s12, 0xa0
	s_lshl_b32 s1, s1, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[78:83], v[144:149], a[0:3], v30, v32 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(8) lgkmcnt(0)
	s_barrier
	v_add_u32_e32 v20, 0x10000, v4
	ds_read_b128 v[22:25], v20 offset:20736
	ds_read_b128 v[26:29], v20 offset:20752
	ds_read_b64 v[34:35], v18
	ds_read_b64 v[36:37], v17 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[90:95], v[144:149], a[4:7], v30, v32 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_i32 s3, s24, 0x1e000
	s_mov_b32 m0, s57
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[120:125], v[144:149], a[8:11], v30, v32 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen lds
	v_add_u32_e32 v21, 0x10000, v6
	ds_read_b64 v[38:39], v21 offset:20736
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[96:101], v[144:149], a[12:15], v30, v32 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[40:41], v21 offset:20744
	ds_read_b64 v[42:43], v21 offset:20752
	s_waitcnt lgkmcnt(6)
	ds_write_b64 v5, v[22:23] offset:53248
	ds_write_b64 v5, v[24:25] offset:53256
	s_waitcnt lgkmcnt(7)
	ds_write_b64 v5, v[26:27] offset:53264
	ds_write_b64 v5, v[28:29] offset:53272
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[126:131], v[144:149], a[16:19], v31, v32 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[22:23], v21 offset:20864
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[132:137], v[144:149], a[20:23], v31, v32 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[24:25], v21 offset:20872
	ds_read_b64 v[26:27], v21 offset:20880
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[138:143], v[144:149], a[24:27], v31, v32 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen offset:1024 lds
	ds_read_b64 v[44:45], v21 offset:20992
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[102:107], v[144:149], a[28:31], v31, v32 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[46:47], v21 offset:21000
	ds_read_b64 v[48:49], v21 offset:21008
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[102:107], v[150:155], a[32:35], v31, v32 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[50:51], v21 offset:21120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[138:143], v[150:155], a[36:39], v31, v32 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[52:53], v21 offset:21128
	ds_read_b64 v[54:55], v21 offset:21136
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[132:137], v[150:155], a[40:43], v31, v32 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen offset:2048 lds
	ds_read_b64 v[56:57], v21 offset:21248
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[126:131], v[150:155], a[44:47], v31, v32 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[58:59], v21 offset:21256
	ds_read_b64 v[60:61], v21 offset:21264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[96:101], v[150:155], a[48:51], v30, v32 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[62:63], v21 offset:21376
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[120:125], v[150:155], a[52:55], v30, v32 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[64:65], v21 offset:21384
	ds_read_b64 v[66:67], v21 offset:21392
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[90:95], v[150:155], a[56:59], v30, v32 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen offset:3072 lds
	ds_read_b64 v[68:69], v21 offset:21504
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[78:83], v[150:155], a[60:63], v30, v32 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[70:71], v21 offset:21512
	ds_read_b64 v[72:73], v21 offset:21520
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[78:83], v[178:183], a[64:67], v30, v32 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[84:85], v21 offset:21632
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[90:95], v[178:183], a[68:71], v30, v32 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[86:87], v21 offset:21640
	ds_read_b64 v[88:89], v21 offset:21648
	s_add_i32 s3, s24, 0x1f000
	s_mov_b32 m0, s54
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[120:125], v[178:183], a[72:75], v30, v32 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen lds
	ds_read_b128 v[144:147], v19 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[96:101], v[178:183], a[76:79], v30, v32 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[148:149], v0 offset:20480
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[126:131], v[178:183], a[80:83], v31, v32 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[150:153], v19 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[132:137], v[178:183], a[84:87], v31, v32 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[154:155], v0 offset:20992
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[138:143], v[178:183], a[88:91], v31, v32 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen offset:1024 lds
	ds_read_b128 v[156:159], v19 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[102:107], v[178:183], a[92:95], v31, v32 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[160:161], v0 offset:21504
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[102:107], v[108:113], a[96:99], v31, v32 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[162:165], v19 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[138:143], v[108:113], a[100:103], v31, v32 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[166:167], v0 offset:22016
	s_add_i32 s1, s9, s1
	s_mov_b32 m0, s56
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[132:137], v[108:113], a[104:107], v31, v32 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v16, s[44:47], s1 offen lds
	ds_read_b128 v[168:171], v19 offset:8192
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[126:131], v[108:113], a[108:111], v31, v32 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[172:173], v0 offset:22528
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[96:101], v[108:113], a[112:115], v30, v32 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[174:177], v19 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[120:125], v[108:113], a[116:119], v30, v32 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[178:179], v0 offset:23040
	s_add_i32 s1, s14, 0x13c00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[90:95], v[108:113], a[120:123], v30, v32 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v16, s[48:51], s1 offen offset:1024 lds
	ds_read_b128 v[202:205], v19 offset:10240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[78:83], v[108:113], a[124:127], v30, v32 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[206:207], v0 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[78:83], v[184:189], a[128:131], v30, v33 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[108:111], v19 offset:11264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[90:95], v[184:189], a[132:135], v30, v33 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[112:113], v0 offset:24064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[120:125], v[184:189], a[136:139], v30, v33 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[96:101], v[184:189], a[140:143], v30, v33 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s1, s2, 0x280
	s_mov_b32 m0, s15
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[126:131], v[184:189], a[144:147], v31, v33 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[132:137], v[184:189], a[148:151], v31, v33 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[138:143], v[184:189], a[152:155], v31, v33 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s1, s26, 0x280
	s_mov_b32 m0, s17
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[102:107], v[184:189], a[156:159], v31, v33 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[102:107], v[190:195], a[160:163], v31, v33 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[138:143], v[190:195], a[164:167], v31, v33 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[132:137], v[190:195], a[168:171], v31, v33 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_i32 s1, s27, 0x280
	s_mov_b32 m0, s28
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[126:131], v[190:195], a[172:175], v31, v33 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[96:101], v[190:195], a[176:179], v30, v33 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[120:125], v[190:195], a[180:183], v30, v33 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s1, s29, 0x280
	s_mov_b32 m0, s30
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[90:95], v[190:195], a[184:187], v30, v33 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[78:83], v[190:195], a[188:191], v30, v33 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[78:83], v[196:201], a[192:195], v30, v33 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[90:95], v[196:201], a[196:199], v30, v33 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s1, s31, 0x280
	s_mov_b32 m0, s32
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[120:125], v[196:201], a[200:203], v30, v33 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[96:101], v[196:201], a[204:207], v30, v33 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[126:131], v[196:201], a[208:211], v31, v33 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s1, s7, 0x280
	s_mov_b32 m0, s33
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[132:137], v[196:201], a[212:215], v31, v33 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[138:143], v[196:201], a[216:219], v31, v33 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[102:107], v[196:201], a[220:223], v31, v33 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[102:107], v[114:119], a[224:227], v31, v33 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s1, s18, 0x280
	s_mov_b32 m0, s34
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[138:143], v[114:119], a[228:231], v31, v33 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[132:137], v[114:119], a[232:235], v31, v33 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[126:131], v[114:119], a[236:239], v31, v33 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s1, s19, 0x280
	s_mov_b32 m0, s36
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[96:101], v[114:119], a[240:243], v30, v33 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[120:125], v[114:119], a[244:247], v30, v33 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[90:95], v[114:119], a[248:251], v30, v33 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[78:83], v[114:119], a[252:255], v30, v33 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s1, s12, 0xc0
	s_lshl_b32 s1, s1, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[38:43], v[144:149], a[0:3], v34, v36 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(8) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[28:29], v18 offset:2048
	ds_read_b64 v[30:31], v17 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[22:27], v[144:149], a[4:7], v34, v36 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_i32 s3, s24, 0x24000
	s_mov_b32 m0, s53
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[44:49], v[144:149], a[8:11], v34, v36 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen lds
	s_mov_b32 s13, 0x10000
	v_lshl_add_u32 v3, v3, 2, s13
	ds_read_b64 v[74:75], v3 offset:20832
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[50:55], v[144:149], a[12:15], v34, v36 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_lshl_add_u32 v14, v14, 2, s13
	ds_read_b64 v[76:77], v14 offset:20840
	ds_read_b64 v[78:79], v14 offset:20848
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[56:61], v[144:149], a[16:19], v35, v36 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[90:91], v3 offset:20960
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[62:67], v[144:149], a[20:23], v35, v36 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[92:93], v14 offset:20968
	ds_read_b64 v[94:95], v14 offset:20976
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[68:73], v[144:149], a[24:27], v35, v36 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen offset:1024 lds
	ds_read_b64 v[96:97], v3 offset:21088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[84:89], v[144:149], a[28:31], v35, v36 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[98:99], v14 offset:21096
	ds_read_b64 v[100:101], v14 offset:21104
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[84:89], v[150:155], a[32:35], v35, v36 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[102:103], v3 offset:21216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[68:73], v[150:155], a[36:39], v35, v36 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[104:105], v14 offset:21224
	ds_read_b64 v[106:107], v14 offset:21232
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[62:67], v[150:155], a[40:43], v35, v36 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen offset:2048 lds
	ds_read_b64 v[114:115], v3 offset:21344
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[56:61], v[150:155], a[44:47], v35, v36 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[116:117], v14 offset:21352
	ds_read_b64 v[118:119], v14 offset:21360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[50:55], v[150:155], a[48:51], v34, v36 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[120:121], v3 offset:21472
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[44:49], v[150:155], a[52:55], v34, v36 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[122:123], v14 offset:21480
	ds_read_b64 v[124:125], v14 offset:21488
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[22:27], v[150:155], a[56:59], v34, v36 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen offset:3072 lds
	ds_read_b64 v[126:127], v3 offset:21600
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[38:43], v[150:155], a[60:63], v34, v36 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[128:129], v14 offset:21608
	ds_read_b64 v[130:131], v14 offset:21616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[38:43], v[156:161], a[64:67], v34, v36 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[132:133], v3 offset:21728
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[22:27], v[156:161], a[68:71], v34, v36 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[134:135], v14 offset:21736
	ds_read_b64 v[136:137], v14 offset:21744
	s_add_i32 s3, s24, 0x25000
	s_mov_b32 m0, s55
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[44:49], v[156:161], a[72:75], v34, v36 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen lds
	ds_read_b128 v[138:141], v19 offset:28672
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[50:55], v[156:161], a[76:79], v34, v36 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[142:143], v0 offset:45056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[56:61], v[156:161], a[80:83], v35, v36 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[144:147], v19 offset:29696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[62:67], v[156:161], a[84:87], v35, v36 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[148:149], v0 offset:45568
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[68:73], v[156:161], a[88:91], v35, v36 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen offset:1024 lds
	ds_read_b128 v[150:153], v19 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[84:89], v[156:161], a[92:95], v35, v36 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[154:155], v0 offset:46080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[84:89], v[162:167], a[96:99], v35, v36 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[156:159], v19 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[68:73], v[162:167], a[100:103], v35, v36 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[160:161], v0 offset:46592
	s_add_i32 s1, s9, s1
	s_mov_b32 m0, s9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[62:67], v[162:167], a[104:107], v35, v36 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v16, s[44:47], s1 offen lds
	ds_read_b128 v[180:183], v19 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[56:61], v[162:167], a[108:111], v35, v36 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[184:185], v0 offset:47104
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[50:55], v[162:167], a[112:115], v34, v36 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[186:189], v19 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[44:49], v[162:167], a[116:119], v34, v36 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[190:191], v0 offset:47616
	s_add_i32 s1, s14, 0x17c00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[22:27], v[162:167], a[120:123], v34, v36 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v16, s[48:51], s1 offen offset:1024 lds
	ds_read_b128 v[192:195], v19 offset:34816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[38:43], v[162:167], a[124:127], v34, v36 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[196:197], v0 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[38:43], v[168:173], a[128:131], v34, v37 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[162:165], v19 offset:35840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[22:27], v[168:173], a[132:135], v34, v37 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[166:167], v0 offset:48640
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[44:49], v[168:173], a[136:139], v34, v37 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[50:55], v[168:173], a[140:143], v34, v37 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[56:61], v[168:173], a[144:147], v35, v37 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[62:67], v[168:173], a[148:151], v35, v37 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[68:73], v[168:173], a[152:155], v35, v37 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[84:89], v[168:173], a[156:159], v35, v37 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[84:89], v[174:179], a[160:163], v35, v37 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[68:73], v[174:179], a[164:167], v35, v37 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[62:67], v[174:179], a[168:171], v35, v37 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[56:61], v[174:179], a[172:175], v35, v37 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[50:55], v[174:179], a[176:179], v34, v37 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[44:49], v[174:179], a[180:183], v34, v37 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[22:27], v[174:179], a[184:187], v34, v37 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[38:43], v[174:179], a[188:191], v34, v37 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[38:43], v[202:207], a[192:195], v34, v37 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[22:27], v[202:207], a[196:199], v34, v37 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[44:49], v[202:207], a[200:203], v34, v37 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[50:55], v[202:207], a[204:207], v34, v37 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[56:61], v[202:207], a[208:211], v35, v37 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[62:67], v[202:207], a[212:215], v35, v37 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[68:73], v[202:207], a[216:219], v35, v37 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[84:89], v[202:207], a[220:223], v35, v37 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[84:89], v[108:113], a[224:227], v35, v37 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[68:73], v[108:113], a[228:231], v35, v37 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[62:67], v[108:113], a[232:235], v35, v37 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[56:61], v[108:113], a[236:239], v35, v37 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[50:55], v[108:113], a[240:243], v34, v37 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[44:49], v[108:113], a[244:247], v34, v37 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[22:27], v[108:113], a[248:251], v34, v37 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[38:43], v[108:113], a[252:255], v34, v37 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s1, s12, 0xe0
	s_lshl_b32 s1, s1, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[74:79], v[138:143], a[0:3], v28, v30 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[22:23], v18
	ds_read_b64 v[24:25], v17 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[90:95], v[138:143], a[4:7], v28, v30 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_i32 s3, s24, 0x2a000
	s_mov_b32 m0, s57
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[96:101], v[138:143], a[8:11], v28, v30 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen lds
	v_lshlrev_b32_e32 v11, 2, v11
	ds_read_b64 v[32:33], v11 offset:53312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[102:107], v[138:143], a[12:15], v28, v30 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[34:35], v11 offset:53320
	v_lshlrev_b32_e32 v9, 2, v9
	ds_read_b64 v[36:37], v9 offset:53328
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[114:119], v[138:143], a[16:19], v29, v30 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[38:39], v11 offset:53440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[120:125], v[138:143], a[20:23], v29, v30 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[40:41], v11 offset:53448
	ds_read_b64 v[42:43], v9 offset:53456
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[126:131], v[138:143], a[24:27], v29, v30 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen offset:1024 lds
	ds_read_b64 v[44:45], v11 offset:53568
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[132:137], v[138:143], a[28:31], v29, v30 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[46:47], v11 offset:53576
	ds_read_b64 v[48:49], v9 offset:53584
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[132:137], v[144:149], a[32:35], v29, v30 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[50:51], v11 offset:53696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[126:131], v[144:149], a[36:39], v29, v30 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[52:53], v11 offset:53704
	ds_read_b64 v[54:55], v9 offset:53712
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[120:125], v[144:149], a[40:43], v29, v30 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen offset:2048 lds
	ds_read_b64 v[56:57], v11 offset:53824
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[114:119], v[144:149], a[44:47], v29, v30 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[58:59], v11 offset:53832
	ds_read_b64 v[60:61], v9 offset:53840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[102:107], v[144:149], a[48:51], v28, v30 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[62:63], v11 offset:53952
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[96:101], v[144:149], a[52:55], v28, v30 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[64:65], v11 offset:53960
	ds_read_b64 v[66:67], v9 offset:53968
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[90:95], v[144:149], a[56:59], v28, v30 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen offset:3072 lds
	ds_read_b64 v[68:69], v11 offset:54080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[74:79], v[144:149], a[60:63], v28, v30 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[70:71], v11 offset:54088
	ds_read_b64 v[72:73], v9 offset:54096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[74:79], v[150:155], a[64:67], v28, v30 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[80:81], v11 offset:54208
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[90:95], v[150:155], a[68:71], v28, v30 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[82:83], v11 offset:54216
	ds_read_b64 v[84:85], v9 offset:54224
	s_add_i32 s3, s24, 0x2b000
	s_mov_b32 m0, s54
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[96:101], v[150:155], a[72:75], v28, v30 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen lds
	ds_read_b128 v[108:111], v19 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[102:107], v[150:155], a[76:79], v28, v30 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[112:113], v0 offset:20480
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[114:119], v[150:155], a[80:83], v29, v30 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[138:141], v19 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[120:125], v[150:155], a[84:87], v29, v30 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[142:143], v0 offset:20992
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[126:131], v[150:155], a[88:91], v29, v30 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen offset:1024 lds
	ds_read_b128 v[144:147], v19 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[132:137], v[150:155], a[92:95], v29, v30 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[148:149], v0 offset:21504
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[132:137], v[156:161], a[96:99], v29, v30 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[150:153], v19 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[126:131], v[156:161], a[100:103], v29, v30 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[154:155], v0 offset:22016
	s_add_i32 s1, s9, s1
	s_mov_b32 m0, s56
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[120:125], v[156:161], a[104:107], v29, v30 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v16, s[44:47], s1 offen lds
	ds_read_b128 v[168:171], v19 offset:8192
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[114:119], v[156:161], a[108:111], v29, v30 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[172:173], v0 offset:22528
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[102:107], v[156:161], a[112:115], v28, v30 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[174:177], v19 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[96:101], v[156:161], a[116:119], v28, v30 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[178:179], v0 offset:23040
	s_add_i32 s1, s14, 0x1bc00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[90:95], v[156:161], a[120:123], v28, v30 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v16, s[48:51], s1 offen offset:1024 lds
	ds_read_b128 v[198:201], v19 offset:10240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[74:79], v[156:161], a[124:127], v28, v30 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[202:203], v0 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[74:79], v[180:185], a[128:131], v28, v31 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[156:159], v19 offset:11264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[90:95], v[180:185], a[132:135], v28, v31 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[160:161], v0 offset:24064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[96:101], v[180:185], a[136:139], v28, v31 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[102:107], v[180:185], a[140:143], v28, v31 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s1, s2, 0x300
	s_mov_b32 m0, s16
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[114:119], v[180:185], a[144:147], v29, v31 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[120:125], v[180:185], a[148:151], v29, v31 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[126:131], v[180:185], a[152:155], v29, v31 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s1, s26, 0x300
	s_mov_b32 m0, s6
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[132:137], v[180:185], a[156:159], v29, v31 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[132:137], v[186:191], a[160:163], v29, v31 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[126:131], v[186:191], a[164:167], v29, v31 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[120:125], v[186:191], a[168:171], v29, v31 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_i32 s1, s27, 0x300
	s_mov_b32 m0, s8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[114:119], v[186:191], a[172:175], v29, v31 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[102:107], v[186:191], a[176:179], v28, v31 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[96:101], v[186:191], a[180:183], v28, v31 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s1, s29, 0x300
	s_mov_b32 m0, s25
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[90:95], v[186:191], a[184:187], v28, v31 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[74:79], v[186:191], a[188:191], v28, v31 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[74:79], v[192:197], a[192:195], v28, v31 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[90:95], v[192:197], a[196:199], v28, v31 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s1, s31, 0x300
	s_mov_b32 m0, s35
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[96:101], v[192:197], a[200:203], v28, v31 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[102:107], v[192:197], a[204:207], v28, v31 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[114:119], v[192:197], a[208:211], v29, v31 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s1, s7, 0x300
	s_mov_b32 m0, s37
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[120:125], v[192:197], a[212:215], v29, v31 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[126:131], v[192:197], a[216:219], v29, v31 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[132:137], v[192:197], a[220:223], v29, v31 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[132:137], v[162:167], a[224:227], v29, v31 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s1, s18, 0x300
	s_mov_b32 m0, s38
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[126:131], v[162:167], a[228:231], v29, v31 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[120:125], v[162:167], a[232:235], v29, v31 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[114:119], v[162:167], a[236:239], v29, v31 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s1, s19, 0x300
	s_mov_b32 m0, s39
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[102:107], v[162:167], a[240:243], v28, v31 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[96:101], v[162:167], a[244:247], v28, v31 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[90:95], v[162:167], a[248:251], v28, v31 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[74:79], v[162:167], a[252:255], v28, v31 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s1, s12, 0x100
	s_lshl_b32 s1, s1, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[32:37], v[108:113], a[0:3], v22, v24 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(8) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[26:27], v18 offset:2048
	ds_read_b64 v[28:29], v17 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[38:43], v[108:113], a[4:7], v22, v24 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_i32 s3, s24, 0x30000
	s_mov_b32 m0, s53
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[44:49], v[108:113], a[8:11], v22, v24 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen lds
	ds_read_b64 v[74:75], v21 offset:20768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[50:55], v[108:113], a[12:15], v22, v24 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[76:77], v21 offset:20776
	ds_read_b64 v[78:79], v21 offset:20784
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[56:61], v[108:113], a[16:19], v23, v24 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[86:87], v21 offset:20896
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[62:67], v[108:113], a[20:23], v23, v24 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[88:89], v21 offset:20904
	ds_read_b64 v[90:91], v21 offset:20912
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[68:73], v[108:113], a[24:27], v23, v24 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen offset:1024 lds
	ds_read_b64 v[92:93], v21 offset:21024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[80:85], v[108:113], a[28:31], v23, v24 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[94:95], v21 offset:21032
	ds_read_b64 v[96:97], v21 offset:21040
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[80:85], v[138:143], a[32:35], v23, v24 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[98:99], v21 offset:21152
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[68:73], v[138:143], a[36:39], v23, v24 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[100:101], v21 offset:21160
	ds_read_b64 v[102:103], v21 offset:21168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[62:67], v[138:143], a[40:43], v23, v24 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen offset:2048 lds
	ds_read_b64 v[104:105], v21 offset:21280
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[56:61], v[138:143], a[44:47], v23, v24 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[106:107], v21 offset:21288
	ds_read_b64 v[108:109], v21 offset:21296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[50:55], v[138:143], a[48:51], v22, v24 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[110:111], v21 offset:21408
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[44:49], v[138:143], a[52:55], v22, v24 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[112:113], v21 offset:21416
	ds_read_b64 v[114:115], v21 offset:21424
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[38:43], v[138:143], a[56:59], v22, v24 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen offset:3072 lds
	ds_read_b64 v[116:117], v21 offset:21536
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[32:37], v[138:143], a[60:63], v22, v24 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[118:119], v21 offset:21544
	ds_read_b64 v[120:121], v21 offset:21552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[32:37], v[144:149], a[64:67], v22, v24 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[122:123], v21 offset:21664
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[38:43], v[144:149], a[68:71], v22, v24 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[124:125], v21 offset:21672
	ds_read_b64 v[126:127], v21 offset:21680
	s_add_i32 s3, s24, 0x31000
	s_mov_b32 m0, s55
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[44:49], v[144:149], a[72:75], v22, v24 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen lds
	ds_read_b128 v[128:131], v19 offset:28672
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[50:55], v[144:149], a[76:79], v22, v24 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[132:133], v0 offset:45056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[56:61], v[144:149], a[80:83], v23, v24 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[134:137], v19 offset:29696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[62:67], v[144:149], a[84:87], v23, v24 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[138:139], v0 offset:45568
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[68:73], v[144:149], a[88:91], v23, v24 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen offset:1024 lds
	ds_read_b128 v[140:143], v19 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[80:85], v[144:149], a[92:95], v23, v24 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[144:145], v0 offset:46080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[80:85], v[150:155], a[96:99], v23, v24 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[162:165], v19 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[68:73], v[150:155], a[100:103], v23, v24 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[166:167], v0 offset:46592
	s_add_i32 s1, s9, s1
	s_mov_b32 m0, s9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[62:67], v[150:155], a[104:107], v23, v24 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v16, s[44:47], s1 offen lds
	ds_read_b128 v[180:183], v19 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[56:61], v[150:155], a[108:111], v23, v24 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[184:185], v0 offset:47104
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[50:55], v[150:155], a[112:115], v22, v24 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[186:189], v19 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[44:49], v[150:155], a[116:119], v22, v24 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[190:191], v0 offset:47616
	s_add_i32 s1, s14, 0x1fc00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[38:43], v[150:155], a[120:123], v22, v24 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v16, s[48:51], s1 offen offset:1024 lds
	ds_read_b128 v[146:149], v19 offset:34816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[32:37], v[150:155], a[124:127], v22, v24 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[150:151], v0 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[32:37], v[168:173], a[128:131], v22, v25 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[192:195], v19 offset:35840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[38:43], v[168:173], a[132:135], v22, v25 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[196:197], v0 offset:48640
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[44:49], v[168:173], a[136:139], v22, v25 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[50:55], v[168:173], a[140:143], v22, v25 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s1, s2, 0x380
	s_mov_b32 m0, s15
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[56:61], v[168:173], a[144:147], v23, v25 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[62:67], v[168:173], a[148:151], v23, v25 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[68:73], v[168:173], a[152:155], v23, v25 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s1, s26, 0x380
	s_mov_b32 m0, s17
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[80:85], v[168:173], a[156:159], v23, v25 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[80:85], v[174:179], a[160:163], v23, v25 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[68:73], v[174:179], a[164:167], v23, v25 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[62:67], v[174:179], a[168:171], v23, v25 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_i32 s1, s27, 0x380
	s_mov_b32 m0, s28
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[56:61], v[174:179], a[172:175], v23, v25 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[50:55], v[174:179], a[176:179], v22, v25 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[44:49], v[174:179], a[180:183], v22, v25 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s1, s29, 0x380
	s_mov_b32 m0, s30
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[38:43], v[174:179], a[184:187], v22, v25 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[32:37], v[174:179], a[188:191], v22, v25 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[32:37], v[198:203], a[192:195], v22, v25 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[38:43], v[198:203], a[196:199], v22, v25 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s1, s31, 0x380
	s_mov_b32 m0, s32
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[44:49], v[198:203], a[200:203], v22, v25 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[50:55], v[198:203], a[204:207], v22, v25 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[56:61], v[198:203], a[208:211], v23, v25 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s1, s7, 0x380
	s_mov_b32 m0, s33
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[62:67], v[198:203], a[212:215], v23, v25 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[68:73], v[198:203], a[216:219], v23, v25 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[80:85], v[198:203], a[220:223], v23, v25 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[80:85], v[156:161], a[224:227], v23, v25 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s1, s18, 0x380
	s_mov_b32 m0, s34
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[68:73], v[156:161], a[228:231], v23, v25 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[62:67], v[156:161], a[232:235], v23, v25 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[56:61], v[156:161], a[236:239], v23, v25 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s1, s19, 0x380
	s_mov_b32 m0, s36
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[50:55], v[156:161], a[240:243], v22, v25 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[44:49], v[156:161], a[244:247], v22, v25 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[38:43], v[156:161], a[248:251], v22, v25 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[32:37], v[156:161], a[252:255], v22, v25 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s1, s12, 0x120
	s_lshl_b32 s1, s1, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[74:79], v[128:133], a[0:3], v26, v28 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(8) lgkmcnt(0)
	s_barrier
	ds_read_b128 v[22:25], v4 offset:53248
	ds_read_b128 v[30:33], v4 offset:53264
	ds_read_b64 v[34:35], v18
	ds_read_b64 v[36:37], v17 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[86:91], v[128:133], a[4:7], v26, v28 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_i32 s3, s24, 0x36000
	s_mov_b32 m0, s57
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[92:97], v[128:133], a[8:11], v26, v28 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen lds
	ds_read_b64 v[38:39], v6 offset:53248
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[98:103], v[128:133], a[12:15], v26, v28 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[40:41], v6 offset:53256
	ds_read_b64 v[42:43], v6 offset:53264
	s_waitcnt lgkmcnt(6)
	ds_write_b64 v5, v[22:23] offset:53248
	ds_write_b64 v5, v[24:25] offset:53256
	s_waitcnt lgkmcnt(7)
	ds_write_b64 v5, v[30:31] offset:53264
	ds_write_b64 v5, v[32:33] offset:53272
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[104:109], v[128:133], a[16:19], v27, v28 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[44:45], v6 offset:53376
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[110:115], v[128:133], a[20:23], v27, v28 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[46:47], v6 offset:53384
	ds_read_b64 v[48:49], v6 offset:53392
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[116:121], v[128:133], a[24:27], v27, v28 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen offset:1024 lds
	ds_read_b64 v[50:51], v6 offset:53504
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[122:127], v[128:133], a[28:31], v27, v28 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[52:53], v6 offset:53512
	ds_read_b64 v[54:55], v6 offset:53520
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[122:127], v[134:139], a[32:35], v27, v28 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[56:57], v6 offset:53632
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[116:121], v[134:139], a[36:39], v27, v28 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[58:59], v6 offset:53640
	ds_read_b64 v[60:61], v6 offset:53648
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[110:115], v[134:139], a[40:43], v27, v28 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen offset:2048 lds
	ds_read_b64 v[62:63], v6 offset:53760
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[104:109], v[134:139], a[44:47], v27, v28 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[64:65], v6 offset:53768
	ds_read_b64 v[66:67], v6 offset:53776
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[98:103], v[134:139], a[48:51], v26, v28 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[68:69], v6 offset:53888
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[92:97], v[134:139], a[52:55], v26, v28 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[70:71], v6 offset:53896
	ds_read_b64 v[72:73], v6 offset:53904
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[86:91], v[134:139], a[56:59], v26, v28 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen offset:3072 lds
	ds_read_b64 v[80:81], v6 offset:54016
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[74:79], v[134:139], a[60:63], v26, v28 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[82:83], v6 offset:54024
	ds_read_b64 v[84:85], v6 offset:54032
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[74:79], v[140:145], a[64:67], v26, v28 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[128:129], v6 offset:54144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[86:91], v[140:145], a[68:71], v26, v28 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[130:131], v6 offset:54152
	ds_read_b64 v[132:133], v6 offset:54160
	s_add_i32 s3, s24, 0x37000
	s_mov_b32 m0, s54
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[92:97], v[140:145], a[72:75], v26, v28 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen lds
	ds_read_b128 v[134:137], v19 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[98:103], v[140:145], a[76:79], v26, v28 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[138:139], v0 offset:20480
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[104:109], v[140:145], a[80:83], v27, v28 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[152:155], v19 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[110:115], v[140:145], a[84:87], v27, v28 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[156:157], v0 offset:20992
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[116:121], v[140:145], a[88:91], v27, v28 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen offset:1024 lds
	ds_read_b128 v[168:171], v19 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[122:127], v[140:145], a[92:95], v27, v28 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[172:173], v0 offset:21504
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[122:127], v[162:167], a[96:99], v27, v28 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[140:143], v19 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[116:121], v[162:167], a[100:103], v27, v28 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[144:145], v0 offset:22016
	s_add_i32 s1, s9, s1
	s_mov_b32 m0, s56
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[110:115], v[162:167], a[104:107], v27, v28 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v16, s[44:47], s1 offen lds
	ds_read_b128 v[174:177], v19 offset:8192
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[104:109], v[162:167], a[108:111], v27, v28 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[178:179], v0 offset:22528
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[98:103], v[162:167], a[112:115], v26, v28 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[198:201], v19 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[92:97], v[162:167], a[116:119], v26, v28 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[202:203], v0 offset:23040
	s_add_i32 s1, s14, 0x23c00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[86:91], v[162:167], a[120:123], v26, v28 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v16, s[48:51], s1 offen offset:1024 lds
	ds_read_b128 v[158:161], v19 offset:10240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[74:79], v[162:167], a[124:127], v26, v28 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[162:163], v0 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[74:79], v[180:185], a[128:131], v26, v29 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[204:207], v19 offset:11264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[86:91], v[180:185], a[132:135], v26, v29 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[208:209], v0 offset:24064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[92:97], v[180:185], a[136:139], v26, v29 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[98:103], v[180:185], a[140:143], v26, v29 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s1, s2, 0x400
	s_mov_b32 m0, s16
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[104:109], v[180:185], a[144:147], v27, v29 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[110:115], v[180:185], a[148:151], v27, v29 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[116:121], v[180:185], a[152:155], v27, v29 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s1, s26, 0x400
	s_mov_b32 m0, s6
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[122:127], v[180:185], a[156:159], v27, v29 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[122:127], v[186:191], a[160:163], v27, v29 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[116:121], v[186:191], a[164:167], v27, v29 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[110:115], v[186:191], a[168:171], v27, v29 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_i32 s1, s27, 0x400
	s_mov_b32 m0, s8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[104:109], v[186:191], a[172:175], v27, v29 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[98:103], v[186:191], a[176:179], v26, v29 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[92:97], v[186:191], a[180:183], v26, v29 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s1, s29, 0x400
	s_mov_b32 m0, s25
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[86:91], v[186:191], a[184:187], v26, v29 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[74:79], v[186:191], a[188:191], v26, v29 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[74:79], v[146:151], a[192:195], v26, v29 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[86:91], v[146:151], a[196:199], v26, v29 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s1, s31, 0x400
	s_mov_b32 m0, s35
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[92:97], v[146:151], a[200:203], v26, v29 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[98:103], v[146:151], a[204:207], v26, v29 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[104:109], v[146:151], a[208:211], v27, v29 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s1, s7, 0x400
	s_mov_b32 m0, s37
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[110:115], v[146:151], a[212:215], v27, v29 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[116:121], v[146:151], a[216:219], v27, v29 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[122:127], v[146:151], a[220:223], v27, v29 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[122:127], v[192:197], a[224:227], v27, v29 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s1, s18, 0x400
	s_mov_b32 m0, s38
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[116:121], v[192:197], a[228:231], v27, v29 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[110:115], v[192:197], a[232:235], v27, v29 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[104:109], v[192:197], a[236:239], v27, v29 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s1, s19, 0x400
	s_mov_b32 m0, s39
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[98:103], v[192:197], a[240:243], v26, v29 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[92:97], v[192:197], a[244:247], v26, v29 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[86:91], v[192:197], a[248:251], v26, v29 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[74:79], v[192:197], a[252:255], v26, v29 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s1, s12, 0x140
	s_lshl_b32 s1, s1, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[38:43], v[134:139], a[0:3], v34, v36 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(8) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[22:23], v18 offset:2048
	ds_read_b64 v[24:25], v17 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[44:49], v[134:139], a[4:7], v34, v36 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_i32 s3, s24, 0x3c000
	s_mov_b32 m0, s53
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[50:55], v[134:139], a[8:11], v34, v36 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen lds
	ds_read_b64 v[26:27], v8 offset:53344
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[56:61], v[134:139], a[12:15], v34, v36 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[28:29], v13 offset:53352
	ds_read_b64 v[30:31], v13 offset:53360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[62:67], v[134:139], a[16:19], v35, v36 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[74:75], v8 offset:53472
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[68:73], v[134:139], a[20:23], v35, v36 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[76:77], v13 offset:53480
	ds_read_b64 v[78:79], v13 offset:53488
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[80:85], v[134:139], a[24:27], v35, v36 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen offset:1024 lds
	ds_read_b64 v[86:87], v8 offset:53600
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[128:133], v[134:139], a[28:31], v35, v36 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[88:89], v13 offset:53608
	ds_read_b64 v[90:91], v13 offset:53616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[128:133], v[152:157], a[32:35], v35, v36 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[92:93], v8 offset:53728
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[80:85], v[152:157], a[36:39], v35, v36 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[94:95], v13 offset:53736
	ds_read_b64 v[96:97], v13 offset:53744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[68:73], v[152:157], a[40:43], v35, v36 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen offset:2048 lds
	ds_read_b64 v[98:99], v8 offset:53856
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[62:67], v[152:157], a[44:47], v35, v36 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[100:101], v13 offset:53864
	ds_read_b64 v[102:103], v13 offset:53872
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[56:61], v[152:157], a[48:51], v34, v36 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[104:105], v8 offset:53984
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[50:55], v[152:157], a[52:55], v34, v36 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[106:107], v13 offset:53992
	ds_read_b64 v[108:109], v13 offset:54000
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[44:49], v[152:157], a[56:59], v34, v36 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen offset:3072 lds
	ds_read_b64 v[110:111], v8 offset:54112
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[38:43], v[152:157], a[60:63], v34, v36 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[112:113], v13 offset:54120
	ds_read_b64 v[114:115], v13 offset:54128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[38:43], v[168:173], a[64:67], v34, v36 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[116:117], v8 offset:54240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[44:49], v[168:173], a[68:71], v34, v36 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[118:119], v13 offset:54248
	ds_read_b64 v[120:121], v13 offset:54256
	s_add_i32 s3, s24, 0x3d000
	s_mov_b32 m0, s55
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[50:55], v[168:173], a[72:75], v34, v36 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen lds
	ds_read_b128 v[122:125], v19 offset:28672
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[56:61], v[168:173], a[76:79], v34, v36 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[126:127], v0 offset:45056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[62:67], v[168:173], a[80:83], v35, v36 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[134:137], v19 offset:29696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[68:73], v[168:173], a[84:87], v35, v36 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[138:139], v0 offset:45568
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[80:85], v[168:173], a[88:91], v35, v36 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen offset:1024 lds
	ds_read_b128 v[146:149], v19 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[128:133], v[168:173], a[92:95], v35, v36 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[150:151], v0 offset:46080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[128:133], v[140:145], a[96:99], v35, v36 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[152:155], v19 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[80:85], v[140:145], a[100:103], v35, v36 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[156:157], v0 offset:46592
	s_add_i32 s1, s9, s1
	s_mov_b32 m0, s9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[68:73], v[140:145], a[104:107], v35, v36 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v16, s[44:47], s1 offen lds
	ds_read_b128 v[164:167], v19 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[62:67], v[140:145], a[108:111], v35, v36 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[168:169], v0 offset:47104
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[56:61], v[140:145], a[112:115], v34, v36 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[180:183], v19 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[50:55], v[140:145], a[116:119], v34, v36 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[184:185], v0 offset:47616
	s_add_i32 s1, s14, 0x27c00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[44:49], v[140:145], a[120:123], v34, v36 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v16, s[48:51], s1 offen offset:1024 lds
	ds_read_b128 v[186:189], v19 offset:34816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[38:43], v[140:145], a[124:127], v34, v36 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[190:191], v0 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[38:43], v[174:179], a[128:131], v34, v37 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[140:143], v19 offset:35840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[44:49], v[174:179], a[132:135], v34, v37 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[144:145], v0 offset:48640
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[50:55], v[174:179], a[136:139], v34, v37 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[56:61], v[174:179], a[140:143], v34, v37 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[62:67], v[174:179], a[144:147], v35, v37 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[68:73], v[174:179], a[148:151], v35, v37 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[80:85], v[174:179], a[152:155], v35, v37 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[128:133], v[174:179], a[156:159], v35, v37 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[128:133], v[198:203], a[160:163], v35, v37 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[80:85], v[198:203], a[164:167], v35, v37 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[68:73], v[198:203], a[168:171], v35, v37 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[62:67], v[198:203], a[172:175], v35, v37 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[56:61], v[198:203], a[176:179], v34, v37 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[50:55], v[198:203], a[180:183], v34, v37 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[44:49], v[198:203], a[184:187], v34, v37 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[38:43], v[198:203], a[188:191], v34, v37 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[38:43], v[158:163], a[192:195], v34, v37 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[44:49], v[158:163], a[196:199], v34, v37 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[50:55], v[158:163], a[200:203], v34, v37 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[56:61], v[158:163], a[204:207], v34, v37 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[62:67], v[158:163], a[208:211], v35, v37 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[68:73], v[158:163], a[212:215], v35, v37 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[80:85], v[158:163], a[216:219], v35, v37 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[128:133], v[158:163], a[220:223], v35, v37 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[128:133], v[204:209], a[224:227], v35, v37 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[80:85], v[204:209], a[228:231], v35, v37 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[68:73], v[204:209], a[232:235], v35, v37 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[62:67], v[204:209], a[236:239], v35, v37 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[56:61], v[204:209], a[240:243], v34, v37 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[50:55], v[204:209], a[244:247], v34, v37 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[44:49], v[204:209], a[248:251], v34, v37 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[38:43], v[204:209], a[252:255], v34, v37 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s1, s12, 0x160
	s_lshl_b32 s1, s1, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:31], v[122:127], a[0:3], v22, v24 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[32:33], v18
	ds_read_b64 v[34:35], v17 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[74:79], v[122:127], a[4:7], v22, v24 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_i32 s3, s24, 0x42000
	s_mov_b32 m0, s57
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[86:91], v[122:127], a[8:11], v22, v24 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen lds
	ds_read_b64 v[36:37], v12 offset:20800
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[92:97], v[122:127], a[12:15], v22, v24 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[38:39], v12 offset:20808
	ds_read_b64 v[40:41], v10 offset:20816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[98:103], v[122:127], a[16:19], v23, v24 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[42:43], v12 offset:20928
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[104:109], v[122:127], a[20:23], v23, v24 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[44:45], v12 offset:20936
	ds_read_b64 v[46:47], v10 offset:20944
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[110:115], v[122:127], a[24:27], v23, v24 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen offset:1024 lds
	ds_read_b64 v[48:49], v12 offset:21056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[116:121], v[122:127], a[28:31], v23, v24 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[50:51], v12 offset:21064
	ds_read_b64 v[52:53], v10 offset:21072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[116:121], v[134:139], a[32:35], v23, v24 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[54:55], v12 offset:21184
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[110:115], v[134:139], a[36:39], v23, v24 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[56:57], v12 offset:21192
	ds_read_b64 v[58:59], v10 offset:21200
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[104:109], v[134:139], a[40:43], v23, v24 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen offset:2048 lds
	ds_read_b64 v[60:61], v12 offset:21312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[98:103], v[134:139], a[44:47], v23, v24 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[62:63], v12 offset:21320
	ds_read_b64 v[64:65], v10 offset:21328
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[92:97], v[134:139], a[48:51], v22, v24 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[66:67], v12 offset:21440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[86:91], v[134:139], a[52:55], v22, v24 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[68:69], v12 offset:21448
	ds_read_b64 v[70:71], v10 offset:21456
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[74:79], v[134:139], a[56:59], v22, v24 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen offset:3072 lds
	ds_read_b64 v[80:81], v12 offset:21568
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[26:31], v[134:139], a[60:63], v22, v24 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[82:83], v12 offset:21576
	ds_read_b64 v[84:85], v10 offset:21584
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[26:31], v[146:151], a[64:67], v22, v24 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[122:123], v12 offset:21696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[74:79], v[146:151], a[68:71], v22, v24 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[124:125], v12 offset:21704
	ds_read_b64 v[126:127], v10 offset:21712
	s_add_i32 s3, s24, 0x43000
	s_mov_b32 m0, s54
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[86:91], v[146:151], a[72:75], v22, v24 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen lds
	ds_read_b128 v[128:131], v19 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[92:97], v[146:151], a[76:79], v22, v24 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[132:133], v0 offset:20480
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[98:103], v[146:151], a[80:83], v23, v24 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[134:137], v19 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[104:109], v[146:151], a[84:87], v23, v24 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[138:139], v0 offset:20992
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[110:115], v[146:151], a[88:91], v23, v24 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen offset:1024 lds
	ds_read_b128 v[158:161], v19 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[116:121], v[146:151], a[92:95], v23, v24 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[162:163], v0 offset:21504
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[116:121], v[152:157], a[96:99], v23, v24 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[146:149], v19 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[110:115], v[152:157], a[100:103], v23, v24 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[150:151], v0 offset:22016
	s_add_i32 s1, s9, s1
	s_mov_b32 m0, s56
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[104:109], v[152:157], a[104:107], v23, v24 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v16, s[44:47], s1 offen lds
	ds_read_b128 v[170:173], v19 offset:8192
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[98:103], v[152:157], a[108:111], v23, v24 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[174:175], v0 offset:22528
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[92:97], v[152:157], a[112:115], v22, v24 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[192:195], v19 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[86:91], v[152:157], a[116:119], v22, v24 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[196:197], v0 offset:23040
	s_add_i32 s1, s14, 0x2bc00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[74:79], v[152:157], a[120:123], v22, v24 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v16, s[48:51], s1 offen offset:1024 lds
	ds_read_b128 v[198:201], v19 offset:10240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[26:31], v[152:157], a[124:127], v22, v24 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[202:203], v0 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[26:31], v[164:169], a[128:131], v22, v25 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[152:155], v19 offset:11264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[74:79], v[164:169], a[132:135], v22, v25 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[156:157], v0 offset:24064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[86:91], v[164:169], a[136:139], v22, v25 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[92:97], v[164:169], a[140:143], v22, v25 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s1, s2, 0x480
	s_mov_b32 m0, s15
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[98:103], v[164:169], a[144:147], v23, v25 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[104:109], v[164:169], a[148:151], v23, v25 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[110:115], v[164:169], a[152:155], v23, v25 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s1, s26, 0x480
	s_mov_b32 m0, s17
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[116:121], v[164:169], a[156:159], v23, v25 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[116:121], v[180:185], a[160:163], v23, v25 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[110:115], v[180:185], a[164:167], v23, v25 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[104:109], v[180:185], a[168:171], v23, v25 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_i32 s1, s27, 0x480
	s_mov_b32 m0, s28
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[98:103], v[180:185], a[172:175], v23, v25 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[92:97], v[180:185], a[176:179], v22, v25 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[86:91], v[180:185], a[180:183], v22, v25 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s1, s29, 0x480
	s_mov_b32 m0, s30
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[74:79], v[180:185], a[184:187], v22, v25 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[26:31], v[180:185], a[188:191], v22, v25 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[26:31], v[186:191], a[192:195], v22, v25 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[74:79], v[186:191], a[196:199], v22, v25 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s1, s31, 0x480
	s_mov_b32 m0, s32
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[86:91], v[186:191], a[200:203], v22, v25 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[92:97], v[186:191], a[204:207], v22, v25 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[98:103], v[186:191], a[208:211], v23, v25 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s1, s7, 0x480
	s_mov_b32 m0, s33
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[104:109], v[186:191], a[212:215], v23, v25 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[110:115], v[186:191], a[216:219], v23, v25 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[116:121], v[186:191], a[220:223], v23, v25 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[116:121], v[140:145], a[224:227], v23, v25 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s1, s18, 0x480
	s_mov_b32 m0, s34
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[110:115], v[140:145], a[228:231], v23, v25 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[104:109], v[140:145], a[232:235], v23, v25 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[98:103], v[140:145], a[236:239], v23, v25 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s1, s19, 0x480
	s_mov_b32 m0, s36
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[92:97], v[140:145], a[240:243], v22, v25 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[86:91], v[140:145], a[244:247], v22, v25 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[74:79], v[140:145], a[248:251], v22, v25 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[26:31], v[140:145], a[252:255], v22, v25 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s1, s12, 0x180
	s_lshl_b32 s1, s1, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:41], v[128:133], a[0:3], v32, v34 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(8) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[12:13], v18 offset:2048
	ds_read_b64 v[22:23], v17 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[42:47], v[128:133], a[4:7], v32, v34 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_i32 s3, s24, 0x48000
	s_mov_b32 m0, s53
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[48:53], v[128:133], a[8:11], v32, v34 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen lds
	ds_read_b64 v[24:25], v6 offset:53280
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[54:59], v[128:133], a[12:15], v32, v34 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[26:27], v6 offset:53288
	ds_read_b64 v[28:29], v6 offset:53296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[60:65], v[128:133], a[16:19], v33, v34 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[72:73], v6 offset:53408
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[66:71], v[128:133], a[20:23], v33, v34 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[74:75], v6 offset:53416
	ds_read_b64 v[76:77], v6 offset:53424
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[80:85], v[128:133], a[24:27], v33, v34 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen offset:1024 lds
	ds_read_b64 v[86:87], v6 offset:53536
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[122:127], v[128:133], a[28:31], v33, v34 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[88:89], v6 offset:53544
	ds_read_b64 v[90:91], v6 offset:53552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[122:127], v[134:139], a[32:35], v33, v34 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[92:93], v6 offset:53664
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[80:85], v[134:139], a[36:39], v33, v34 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[94:95], v6 offset:53672
	ds_read_b64 v[96:97], v6 offset:53680
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[66:71], v[134:139], a[40:43], v33, v34 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen offset:2048 lds
	ds_read_b64 v[98:99], v6 offset:53792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[60:65], v[134:139], a[44:47], v33, v34 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[100:101], v6 offset:53800
	ds_read_b64 v[102:103], v6 offset:53808
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[54:59], v[134:139], a[48:51], v32, v34 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[104:105], v6 offset:53920
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[48:53], v[134:139], a[52:55], v32, v34 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[106:107], v6 offset:53928
	ds_read_b64 v[108:109], v6 offset:53936
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[42:47], v[134:139], a[56:59], v32, v34 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen offset:3072 lds
	ds_read_b64 v[110:111], v6 offset:54048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[36:41], v[134:139], a[60:63], v32, v34 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[112:113], v6 offset:54056
	ds_read_b64 v[114:115], v6 offset:54064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[36:41], v[158:163], a[64:67], v32, v34 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[116:117], v6 offset:54176
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[42:47], v[158:163], a[68:71], v32, v34 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[118:119], v6 offset:54184
	ds_read_b64 v[120:121], v6 offset:54192
	s_add_i32 s3, s24, 0x49000
	s_mov_b32 m0, s55
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[48:53], v[158:163], a[72:75], v32, v34 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen lds
	ds_read_b128 v[128:131], v19 offset:28672
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[54:59], v[158:163], a[76:79], v32, v34 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[132:133], v0 offset:45056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[60:65], v[158:163], a[80:83], v33, v34 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[134:137], v19 offset:29696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[66:71], v[158:163], a[84:87], v33, v34 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[138:139], v0 offset:45568
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[80:85], v[158:163], a[88:91], v33, v34 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen offset:1024 lds
	ds_read_b128 v[140:143], v19 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[122:127], v[158:163], a[92:95], v33, v34 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[144:145], v0 offset:46080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[122:127], v[146:151], a[96:99], v33, v34 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[158:161], v19 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[80:85], v[146:151], a[100:103], v33, v34 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[162:163], v0 offset:46592
	s_add_i32 s1, s9, s1
	s_mov_b32 m0, s9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[66:71], v[146:151], a[104:107], v33, v34 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v16, s[44:47], s1 offen lds
	ds_read_b128 v[164:167], v19 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[60:65], v[146:151], a[108:111], v33, v34 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[168:169], v0 offset:47104
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[54:59], v[146:151], a[112:115], v32, v34 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[176:179], v19 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[48:53], v[146:151], a[116:119], v32, v34 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[180:181], v0 offset:47616
	s_add_i32 s1, s14, 0x2fc00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[42:47], v[146:151], a[120:123], v32, v34 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v16, s[48:51], s1 offen offset:1024 lds
	ds_read_b128 v[182:185], v19 offset:34816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[36:41], v[146:151], a[124:127], v32, v34 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[186:187], v0 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[36:41], v[170:175], a[128:131], v32, v35 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[146:149], v19 offset:35840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[42:47], v[170:175], a[132:135], v32, v35 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[150:151], v0 offset:48640
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[48:53], v[170:175], a[136:139], v32, v35 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[54:59], v[170:175], a[140:143], v32, v35 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s1, s2, 0x500
	s_mov_b32 m0, s16
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[60:65], v[170:175], a[144:147], v33, v35 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[66:71], v[170:175], a[148:151], v33, v35 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[80:85], v[170:175], a[152:155], v33, v35 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s1, s26, 0x500
	s_mov_b32 m0, s6
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[122:127], v[170:175], a[156:159], v33, v35 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[122:127], v[192:197], a[160:163], v33, v35 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[80:85], v[192:197], a[164:167], v33, v35 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[66:71], v[192:197], a[168:171], v33, v35 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_i32 s1, s27, 0x500
	s_mov_b32 m0, s8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[60:65], v[192:197], a[172:175], v33, v35 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[54:59], v[192:197], a[176:179], v32, v35 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[48:53], v[192:197], a[180:183], v32, v35 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s1, s29, 0x500
	s_mov_b32 m0, s25
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[42:47], v[192:197], a[184:187], v32, v35 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[36:41], v[192:197], a[188:191], v32, v35 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[36:41], v[198:203], a[192:195], v32, v35 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[42:47], v[198:203], a[196:199], v32, v35 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s1, s31, 0x500
	s_mov_b32 m0, s35
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[48:53], v[198:203], a[200:203], v32, v35 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[54:59], v[198:203], a[204:207], v32, v35 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[60:65], v[198:203], a[208:211], v33, v35 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s1, s7, 0x500
	s_mov_b32 m0, s37
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[66:71], v[198:203], a[212:215], v33, v35 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[80:85], v[198:203], a[216:219], v33, v35 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[122:127], v[198:203], a[220:223], v33, v35 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[122:127], v[152:157], a[224:227], v33, v35 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s1, s18, 0x500
	s_mov_b32 m0, s38
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[80:85], v[152:157], a[228:231], v33, v35 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[66:71], v[152:157], a[232:235], v33, v35 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[60:65], v[152:157], a[236:239], v33, v35 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s1, s19, 0x500
	s_mov_b32 m0, s39
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[54:59], v[152:157], a[240:243], v32, v35 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[48:53], v[152:157], a[244:247], v32, v35 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[42:47], v[152:157], a[248:251], v32, v35 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[36:41], v[152:157], a[252:255], v32, v35 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s1, s12, 0x1a0
	s_lshl_b32 s1, s1, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[24:29], v[128:133], a[0:3], v12, v22 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(8) lgkmcnt(0)
	s_barrier
	ds_read_b128 v[30:33], v20 offset:20736
	ds_read_b128 v[34:37], v20 offset:20752
	ds_read_b64 v[38:39], v18
	ds_read_b64 v[40:41], v17 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[72:77], v[128:133], a[4:7], v12, v22 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_i32 s3, s24, 0x4e000
	s_mov_b32 m0, s57
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[86:91], v[128:133], a[8:11], v12, v22 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen lds
	ds_read_b64 v[42:43], v21 offset:20736
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[92:97], v[128:133], a[12:15], v12, v22 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[44:45], v21 offset:20744
	ds_read_b64 v[46:47], v21 offset:20752
	s_waitcnt lgkmcnt(6)
	ds_write_b64 v5, v[30:31] offset:53248
	ds_write_b64 v5, v[32:33] offset:53256
	s_waitcnt lgkmcnt(7)
	ds_write_b64 v5, v[34:35] offset:53264
	ds_write_b64 v5, v[36:37] offset:53272
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[98:103], v[128:133], a[16:19], v13, v22 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[30:31], v21 offset:20864
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[104:109], v[128:133], a[20:23], v13, v22 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[32:33], v21 offset:20872
	ds_read_b64 v[34:35], v21 offset:20880
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[110:115], v[128:133], a[24:27], v13, v22 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen offset:1024 lds
	ds_read_b64 v[48:49], v21 offset:20992
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[116:121], v[128:133], a[28:31], v13, v22 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[50:51], v21 offset:21000
	ds_read_b64 v[52:53], v21 offset:21008
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[116:121], v[134:139], a[32:35], v13, v22 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[54:55], v21 offset:21120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[110:115], v[134:139], a[36:39], v13, v22 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[56:57], v21 offset:21128
	ds_read_b64 v[58:59], v21 offset:21136
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[104:109], v[134:139], a[40:43], v13, v22 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen offset:2048 lds
	ds_read_b64 v[60:61], v21 offset:21248
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[98:103], v[134:139], a[44:47], v13, v22 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[62:63], v21 offset:21256
	ds_read_b64 v[64:65], v21 offset:21264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[92:97], v[134:139], a[48:51], v12, v22 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[66:67], v21 offset:21376
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[86:91], v[134:139], a[52:55], v12, v22 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[68:69], v21 offset:21384
	ds_read_b64 v[70:71], v21 offset:21392
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[72:77], v[134:139], a[56:59], v12, v22 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen offset:3072 lds
	ds_read_b64 v[78:79], v21 offset:21504
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[24:29], v[134:139], a[60:63], v12, v22 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[80:81], v21 offset:21512
	ds_read_b64 v[82:83], v21 offset:21520
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[24:29], v[140:145], a[64:67], v12, v22 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[122:123], v21 offset:21632
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[72:77], v[140:145], a[68:71], v12, v22 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[124:125], v21 offset:21640
	ds_read_b64 v[126:127], v21 offset:21648
	s_add_i32 s3, s24, 0x4f000
	s_mov_b32 m0, s54
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[86:91], v[140:145], a[72:75], v12, v22 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen lds
	ds_read_b128 v[128:131], v19 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[92:97], v[140:145], a[76:79], v12, v22 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[132:133], v0 offset:20480
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[98:103], v[140:145], a[80:83], v13, v22 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[134:137], v19 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[104:109], v[140:145], a[84:87], v13, v22 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[138:139], v0 offset:20992
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[110:115], v[140:145], a[88:91], v13, v22 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s3 offen offset:1024 lds
	ds_read_b128 v[152:155], v19 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[116:121], v[140:145], a[92:95], v13, v22 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[156:157], v0 offset:21504
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[116:121], v[158:163], a[96:99], v13, v22 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[140:143], v19 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[110:115], v[158:163], a[100:103], v13, v22 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[144:145], v0 offset:22016
	s_add_i32 s1, s9, s1
	s_mov_b32 m0, s56
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[104:109], v[158:163], a[104:107], v13, v22 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v16, s[44:47], s1 offen lds
	ds_read_b128 v[170:173], v19 offset:8192
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[98:103], v[158:163], a[108:111], v13, v22 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[174:175], v0 offset:22528
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[92:97], v[158:163], a[112:115], v12, v22 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[188:191], v19 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[86:91], v[158:163], a[116:119], v12, v22 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[192:193], v0 offset:23040
	s_add_i32 s1, s14, 0x33c00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[72:77], v[158:163], a[120:123], v12, v22 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v16, s[48:51], s1 offen offset:1024 lds
	ds_read_b128 v[194:197], v19 offset:10240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[24:29], v[158:163], a[124:127], v12, v22 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[198:199], v0 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[24:29], v[164:169], a[128:131], v12, v23 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[158:161], v19 offset:11264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[72:77], v[164:169], a[132:135], v12, v23 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[162:163], v0 offset:24064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[86:91], v[164:169], a[136:139], v12, v23 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[92:97], v[164:169], a[140:143], v12, v23 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s1, s2, 0x580
	s_mov_b32 m0, s15
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[98:103], v[164:169], a[144:147], v13, v23 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[104:109], v[164:169], a[148:151], v13, v23 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[110:115], v[164:169], a[152:155], v13, v23 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s1, s26, 0x580
	s_mov_b32 m0, s17
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[116:121], v[164:169], a[156:159], v13, v23 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[116:121], v[176:181], a[160:163], v13, v23 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[110:115], v[176:181], a[164:167], v13, v23 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[104:109], v[176:181], a[168:171], v13, v23 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_i32 s1, s27, 0x580
	s_mov_b32 m0, s28
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[98:103], v[176:181], a[172:175], v13, v23 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[92:97], v[176:181], a[176:179], v12, v23 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[86:91], v[176:181], a[180:183], v12, v23 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s1, s29, 0x580
	s_mov_b32 m0, s30
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[72:77], v[176:181], a[184:187], v12, v23 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[24:29], v[176:181], a[188:191], v12, v23 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[24:29], v[182:187], a[192:195], v12, v23 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[72:77], v[182:187], a[196:199], v12, v23 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s1, s31, 0x580
	s_mov_b32 m0, s32
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[86:91], v[182:187], a[200:203], v12, v23 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[92:97], v[182:187], a[204:207], v12, v23 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[98:103], v[182:187], a[208:211], v13, v23 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s1, s7, 0x580
	s_mov_b32 m0, s33
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[104:109], v[182:187], a[212:215], v13, v23 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[110:115], v[182:187], a[216:219], v13, v23 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[116:121], v[182:187], a[220:223], v13, v23 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[116:121], v[146:151], a[224:227], v13, v23 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s1, s18, 0x580
	s_mov_b32 m0, s34
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[110:115], v[146:151], a[228:231], v13, v23 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[104:109], v[146:151], a[232:235], v13, v23 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[98:103], v[146:151], a[236:239], v13, v23 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s1, s19, 0x580
	s_mov_b32 m0, s36
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[92:97], v[146:151], a[240:243], v12, v23 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[20:23], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[86:91], v[146:151], a[244:247], v12, v23 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[72:77], v[146:151], a[248:251], v12, v23 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[24:29], v[146:151], a[252:255], v12, v23 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s1, s12, 0x1c0
	s_lshl_b32 s1, s1, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[42:47], v[128:133], a[0:3], v38, v40 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(8) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[4:5], v18 offset:2048
	ds_read_b64 v[6:7], v17 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[30:35], v[128:133], a[4:7], v38, v40 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_i32 s2, s24, 0x54000
	s_mov_b32 m0, s53
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[48:53], v[128:133], a[8:11], v38, v40 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s2 offen lds
	ds_read_b64 v[22:23], v3 offset:20832
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[54:59], v[128:133], a[12:15], v38, v40 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[24:25], v14 offset:20840
	ds_read_b64 v[26:27], v14 offset:20848
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[60:65], v[128:133], a[16:19], v39, v40 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[72:73], v3 offset:20960
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[66:71], v[128:133], a[20:23], v39, v40 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[74:75], v14 offset:20968
	ds_read_b64 v[76:77], v14 offset:20976
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[78:83], v[128:133], a[24:27], v39, v40 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s2 offen offset:1024 lds
	ds_read_b64 v[84:85], v3 offset:21088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[122:127], v[128:133], a[28:31], v39, v40 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[86:87], v14 offset:21096
	ds_read_b64 v[88:89], v14 offset:21104
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[122:127], v[134:139], a[32:35], v39, v40 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[90:91], v3 offset:21216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[78:83], v[134:139], a[36:39], v39, v40 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[92:93], v14 offset:21224
	ds_read_b64 v[94:95], v14 offset:21232
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[66:71], v[134:139], a[40:43], v39, v40 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s2 offen offset:2048 lds
	ds_read_b64 v[96:97], v3 offset:21344
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[60:65], v[134:139], a[44:47], v39, v40 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[98:99], v14 offset:21352
	ds_read_b64 v[100:101], v14 offset:21360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[54:59], v[134:139], a[48:51], v38, v40 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[102:103], v3 offset:21472
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[48:53], v[134:139], a[52:55], v38, v40 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[104:105], v14 offset:21480
	ds_read_b64 v[106:107], v14 offset:21488
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[30:35], v[134:139], a[56:59], v38, v40 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s2 offen offset:3072 lds
	ds_read_b64 v[108:109], v3 offset:21600
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[42:47], v[134:139], a[60:63], v38, v40 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[110:111], v14 offset:21608
	ds_read_b64 v[112:113], v14 offset:21616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[42:47], v[152:157], a[64:67], v38, v40 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[114:115], v3 offset:21728
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[30:35], v[152:157], a[68:71], v38, v40 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[116:117], v14 offset:21736
	ds_read_b64 v[118:119], v14 offset:21744
	s_add_i32 s2, s24, 0x55000
	s_mov_b32 m0, s55
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[48:53], v[152:157], a[72:75], v38, v40 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s2 offen lds
	ds_read_b128 v[128:131], v19 offset:28672
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[54:59], v[152:157], a[76:79], v38, v40 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[132:133], v0 offset:45056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[60:65], v[152:157], a[80:83], v39, v40 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[134:137], v19 offset:29696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[66:71], v[152:157], a[84:87], v39, v40 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[138:139], v0 offset:45568
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[78:83], v[152:157], a[88:91], v39, v40 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s2 offen offset:1024 lds
	ds_read_b128 v[146:149], v19 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[122:127], v[152:157], a[92:95], v39, v40 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[150:151], v0 offset:46080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[122:127], v[140:145], a[96:99], v39, v40 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[152:155], v19 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[78:83], v[140:145], a[100:103], v39, v40 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[156:157], v0 offset:46592
	s_add_i32 s1, s9, s1
	s_mov_b32 m0, s9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[66:71], v[140:145], a[104:107], v39, v40 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v16, s[44:47], s1 offen lds
	ds_read_b128 v[164:167], v19 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[60:65], v[140:145], a[108:111], v39, v40 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[168:169], v0 offset:47104
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[54:59], v[140:145], a[112:115], v38, v40 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[176:179], v19 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[48:53], v[140:145], a[116:119], v38, v40 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[180:181], v0 offset:47616
	s_add_i32 s1, s14, 0x37c00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[30:35], v[140:145], a[120:123], v38, v40 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v16, s[48:51], s1 offen offset:1024 lds
	ds_read_b128 v[182:185], v19 offset:34816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[42:47], v[140:145], a[124:127], v38, v40 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[186:187], v0 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[42:47], v[170:175], a[128:131], v38, v41 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[140:143], v19 offset:35840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[30:35], v[170:175], a[132:135], v38, v41 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[144:145], v0 offset:48640
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[48:53], v[170:175], a[136:139], v38, v41 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[54:59], v[170:175], a[140:143], v38, v41 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[60:65], v[170:175], a[144:147], v39, v41 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[66:71], v[170:175], a[148:151], v39, v41 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[78:83], v[170:175], a[152:155], v39, v41 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[122:127], v[170:175], a[156:159], v39, v41 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[122:127], v[188:193], a[160:163], v39, v41 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[78:83], v[188:193], a[164:167], v39, v41 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[66:71], v[188:193], a[168:171], v39, v41 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[60:65], v[188:193], a[172:175], v39, v41 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[54:59], v[188:193], a[176:179], v38, v41 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[48:53], v[188:193], a[180:183], v38, v41 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[30:35], v[188:193], a[184:187], v38, v41 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[42:47], v[188:193], a[188:191], v38, v41 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[42:47], v[194:199], a[192:195], v38, v41 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[30:35], v[194:199], a[196:199], v38, v41 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[48:53], v[194:199], a[200:203], v38, v41 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[54:59], v[194:199], a[204:207], v38, v41 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[60:65], v[194:199], a[208:211], v39, v41 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[66:71], v[194:199], a[212:215], v39, v41 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[78:83], v[194:199], a[216:219], v39, v41 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[122:127], v[194:199], a[220:223], v39, v41 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[122:127], v[158:163], a[224:227], v39, v41 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[78:83], v[158:163], a[228:231], v39, v41 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[66:71], v[158:163], a[232:235], v39, v41 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[60:65], v[158:163], a[236:239], v39, v41 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[54:59], v[158:163], a[240:243], v38, v41 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[48:53], v[158:163], a[244:247], v38, v41 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[30:35], v[158:163], a[248:251], v38, v41 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[42:47], v[158:163], a[252:255], v38, v41 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s1, s12, 0x1e0
	s_lshl_b32 s1, s1, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:27], v[128:133], a[0:3], v4, v6 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[12:13], v18
	ds_read_b64 v[28:29], v17 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[72:77], v[128:133], a[4:7], v4, v6 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_i32 s2, s24, 0x5a000
	s_mov_b32 m0, s57
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[84:89], v[128:133], a[8:11], v4, v6 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s2 offen lds
	ds_read_b64 v[30:31], v11 offset:53312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[90:95], v[128:133], a[12:15], v4, v6 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[32:33], v11 offset:53320
	ds_read_b64 v[34:35], v9 offset:53328
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[96:101], v[128:133], a[16:19], v5, v6 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[36:37], v11 offset:53440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[102:107], v[128:133], a[20:23], v5, v6 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[38:39], v11 offset:53448
	ds_read_b64 v[40:41], v9 offset:53456
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[108:113], v[128:133], a[24:27], v5, v6 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s2 offen offset:1024 lds
	ds_read_b64 v[42:43], v11 offset:53568
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[114:119], v[128:133], a[28:31], v5, v6 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[44:45], v11 offset:53576
	ds_read_b64 v[46:47], v9 offset:53584
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[114:119], v[134:139], a[32:35], v5, v6 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[48:49], v11 offset:53696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[108:113], v[134:139], a[36:39], v5, v6 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[50:51], v11 offset:53704
	ds_read_b64 v[52:53], v9 offset:53712
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[102:107], v[134:139], a[40:43], v5, v6 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s2 offen offset:2048 lds
	ds_read_b64 v[54:55], v11 offset:53824
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[96:101], v[134:139], a[44:47], v5, v6 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[56:57], v11 offset:53832
	ds_read_b64 v[58:59], v9 offset:53840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[90:95], v[134:139], a[48:51], v4, v6 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[60:61], v11 offset:53952
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[84:89], v[134:139], a[52:55], v4, v6 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[62:63], v11 offset:53960
	ds_read_b64 v[64:65], v9 offset:53968
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[72:77], v[134:139], a[56:59], v4, v6 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s2 offen offset:3072 lds
	ds_read_b64 v[66:67], v11 offset:54080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[22:27], v[134:139], a[60:63], v4, v6 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[68:69], v11 offset:54088
	ds_read_b64 v[70:71], v9 offset:54096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[22:27], v[146:151], a[64:67], v4, v6 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v11 offset:54208
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[72:77], v[146:151], a[68:71], v4, v6 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[80:81], v11 offset:54216
	ds_read_b64 v[82:83], v9 offset:54224
	s_add_i32 s2, s24, 0x5b000
	s_mov_b32 m0, s54
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[84:89], v[146:151], a[72:75], v4, v6 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s2 offen lds
	ds_read_b128 v[120:123], v19 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[90:95], v[146:151], a[76:79], v4, v6 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[124:125], v0 offset:20480
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[96:101], v[146:151], a[80:83], v5, v6 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[126:129], v19 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[102:107], v[146:151], a[84:87], v5, v6 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[130:131], v0 offset:20992
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[108:113], v[146:151], a[88:91], v5, v6 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[40:43], s2 offen offset:1024 lds
	ds_read_b128 v[132:135], v19 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[114:119], v[146:151], a[92:95], v5, v6 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[136:137], v0 offset:21504
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[114:119], v[152:157], a[96:99], v5, v6 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[146:149], v19 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[108:113], v[152:157], a[100:103], v5, v6 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[150:151], v0 offset:22016
	s_add_i32 s1, s9, s1
	s_mov_b32 m0, s56
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[102:107], v[152:157], a[104:107], v5, v6 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v16, s[44:47], s1 offen lds
	ds_read_b128 v[158:161], v19 offset:8192
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[96:101], v[152:157], a[108:111], v5, v6 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[162:163], v0 offset:22528
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[90:95], v[152:157], a[112:115], v4, v6 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[170:173], v19 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[84:89], v[152:157], a[116:119], v4, v6 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[174:175], v0 offset:23040
	s_add_i32 s1, s14, 0x3bc00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[72:77], v[152:157], a[120:123], v4, v6 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v16, s[48:51], s1 offen offset:1024 lds
	ds_read_b128 v[188:191], v19 offset:10240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[22:27], v[152:157], a[124:127], v4, v6 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[192:193], v0 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[22:27], v[164:169], a[128:131], v4, v7 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[152:155], v19 offset:11264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[72:77], v[164:169], a[132:135], v4, v7 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[156:157], v0 offset:24064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[84:89], v[164:169], a[136:139], v4, v7 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[90:95], v[164:169], a[140:143], v4, v7 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[96:101], v[164:169], a[144:147], v5, v7 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[102:107], v[164:169], a[148:151], v5, v7 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[108:113], v[164:169], a[152:155], v5, v7 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[114:119], v[164:169], a[156:159], v5, v7 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[114:119], v[176:181], a[160:163], v5, v7 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[108:113], v[176:181], a[164:167], v5, v7 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[102:107], v[176:181], a[168:171], v5, v7 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[96:101], v[176:181], a[172:175], v5, v7 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[90:95], v[176:181], a[176:179], v4, v7 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[84:89], v[176:181], a[180:183], v4, v7 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[72:77], v[176:181], a[184:187], v4, v7 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[22:27], v[176:181], a[188:191], v4, v7 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[22:27], v[182:187], a[192:195], v4, v7 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[72:77], v[182:187], a[196:199], v4, v7 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[84:89], v[182:187], a[200:203], v4, v7 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[90:95], v[182:187], a[204:207], v4, v7 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[96:101], v[182:187], a[208:211], v5, v7 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[102:107], v[182:187], a[212:215], v5, v7 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[108:113], v[182:187], a[216:219], v5, v7 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[114:119], v[182:187], a[220:223], v5, v7 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[114:119], v[140:145], a[224:227], v5, v7 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[108:113], v[140:145], a[228:231], v5, v7 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[102:107], v[140:145], a[232:235], v5, v7 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[96:101], v[140:145], a[236:239], v5, v7 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[90:95], v[140:145], a[240:243], v4, v7 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[84:89], v[140:145], a[244:247], v4, v7 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[72:77], v[140:145], a[248:251], v4, v7 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[22:27], v[140:145], a[252:255], v4, v7 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[4:5], v18 offset:2048
	ds_read_b64 v[6:7], v17 offset:3072
	ds_read_b64 v[22:23], v21 offset:20768
	ds_read_b64 v[24:25], v21 offset:20776
	ds_read_b64 v[26:27], v21 offset:20784
	ds_read_b64 v[72:73], v21 offset:20896
	ds_read_b64 v[74:75], v21 offset:20904
	ds_read_b64 v[76:77], v21 offset:20912
	ds_read_b64 v[84:85], v21 offset:21024
	ds_read_b64 v[86:87], v21 offset:21032
	ds_read_b64 v[88:89], v21 offset:21040
	ds_read_b64 v[90:91], v21 offset:21152
	ds_read_b64 v[92:93], v21 offset:21160
	ds_read_b64 v[94:95], v21 offset:21168
	ds_read_b64 v[96:97], v21 offset:21280
	ds_read_b64 v[98:99], v21 offset:21288
	ds_read_b64 v[100:101], v21 offset:21296
	ds_read_b64 v[102:103], v21 offset:21408
	ds_read_b64 v[104:105], v21 offset:21416
	ds_read_b64 v[106:107], v21 offset:21424
	ds_read_b64 v[108:109], v21 offset:21536
	ds_read_b64 v[110:111], v21 offset:21544
	ds_read_b64 v[112:113], v21 offset:21552
	ds_read_b64 v[114:115], v21 offset:21664
	ds_read_b64 v[116:117], v21 offset:21672
	ds_read_b64 v[118:119], v21 offset:21680
	ds_read_b128 v[138:141], v19 offset:28672
	ds_read_b64 v[142:143], v0 offset:45056
	ds_read_b128 v[164:167], v19 offset:29696
	ds_read_b64 v[168:169], v0 offset:45568
	ds_read_b128 v[176:179], v19 offset:30720
	ds_read_b64 v[180:181], v0 offset:46080
	ds_read_b128 v[182:185], v19 offset:31744
	ds_read_b64 v[186:187], v0 offset:46592
	ds_read_b128 v[194:197], v19 offset:32768
	ds_read_b64 v[198:199], v0 offset:47104
	ds_read_b128 v[200:203], v19 offset:33792
	ds_read_b64 v[204:205], v0 offset:47616
	ds_read_b128 v[206:209], v19 offset:34816
	ds_read_b64 v[210:211], v0 offset:48128
	ds_read_b128 v[14:17], v19 offset:35840
	ds_read_b64 v[18:19], v0 offset:48640
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[30:35], v[120:125], a[0:3], v12, v28 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[30:35], v[126:131], a[60:63], v12, v28 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[30:35], v[132:137], a[64:67], v12, v28 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[30:35], v[146:151], a[124:127], v12, v28 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[30:35], v[158:163], a[128:131], v12, v29 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[30:35], v[170:175], a[188:191], v12, v29 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[30:35], v[188:193], a[192:195], v12, v29 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[30:35], v[152:157], a[252:255], v12, v29 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[22:27], v[14:19], a[252:255], v4, v7 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[22:27], v[206:211], a[192:195], v4, v7 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[22:27], v[200:205], a[188:191], v4, v7 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[22:27], v[194:199], a[128:131], v4, v7 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[22:27], v[182:187], a[124:127], v4, v6 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[22:27], v[176:181], a[64:67], v4, v6 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[22:27], v[164:169], a[60:63], v4, v6 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:27], v[138:143], a[0:3], v4, v6 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[36:41], v[152:157], a[248:251], v12, v29 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_and_b32_e32 v0, 1, v1
	v_sub_u32_e32 v3, v1, v0
	v_ashrrev_i32_e32 v8, 31, v3
	v_cmp_eq_i32_e64 s[2:3], v0, 1
	v_lshrrev_b32_e32 v8, 31, v8
	v_lshlrev_b32_e32 v0, 1, v0
	v_add_u32_e32 v3, v3, v8
	s_nop 3
	v_accvgpr_read_b32 v8, a0
	v_accvgpr_read_b32 v9, a1
	v_cvt_pk_bf16_f32 v8, v8, v9
	v_accvgpr_read_b32 v9, a2
	v_accvgpr_read_b32 v10, a3
	v_cvt_pk_bf16_f32 v9, v9, v10
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[36:41], v[188:193], a[196:199], v12, v29 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_lshl_add_u32 v0, v2, 2, v0
	v_ashrrev_i32_e32 v3, 1, v3
	v_mov_b32_dpp v10, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v11, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_mul_i32 s0, s0, 0x440
	s_mov_b32 s1, 0x44
	v_mad_u32_u24 v0, s1, v0, v3
	v_cndmask_b32_e64 v3, v8, v11, s[2:3]
	v_cndmask_b32_e64 v8, v10, v9, s[2:3]
	v_lshlrev_b32_e32 v9, 16, v8
	v_lshrrev_b32_e32 v10, 16, v3
	v_add_lshl_u32 v0, s0, v0, 2
	v_lshlrev_b32_e32 v11, 2, v1
	v_lshlrev_b32_e32 v1, 4, v1
	s_mov_b32 s6, 0xffff
	v_and_or_b32 v3, s6, v3, v9
	s_mov_b32 s7, 0xffff0000
	v_and_or_b32 v8, s7, v8, v10
	v_add_u32_e32 v0, 0x1000, v0
	s_mov_b32 s8, s10
	s_and_b32 s9, s11, 0xffff
	s_mov_b32 s10, 0x7ffffff0
	s_mov_b32 s11, 0x31014000
	v_mad_u32_u24 v9, s1, v2, v11
	v_lshl_add_u32 v1, v2, 13, v1
	ds_write2_b32 v0, v3, v8 offset1:68
	v_accvgpr_read_b32 v2, a60
	v_accvgpr_read_b32 v3, a61
	v_cvt_pk_bf16_f32 v2, v2, v3
	v_accvgpr_read_b32 v3, a62
	v_accvgpr_read_b32 v8, a63
	v_cvt_pk_bf16_f32 v3, v3, v8
	v_mov_b32_dpp v8, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v10, v3 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v2, v2, v10, s[2:3]
	v_cndmask_b32_e64 v3, v8, v3, s[2:3]
	v_lshlrev_b32_e32 v8, 16, v3
	v_lshrrev_b32_e32 v10, 16, v2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[36:41], v[170:175], a[184:187], v12, v29 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_and_or_b32 v2, s6, v2, v8
	v_and_or_b32 v3, s7, v3, v10
	ds_write2_b32 v0, v2, v3 offset0:8 offset1:76
	v_accvgpr_read_b32 v2, a64
	v_accvgpr_read_b32 v3, a65
	v_cvt_pk_bf16_f32 v2, v2, v3
	v_accvgpr_read_b32 v3, a66
	v_accvgpr_read_b32 v8, a67
	v_cvt_pk_bf16_f32 v3, v3, v8
	v_mov_b32_dpp v8, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v10, v3 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v2, v2, v10, s[2:3]
	v_cndmask_b32_e64 v3, v8, v3, s[2:3]
	v_lshlrev_b32_e32 v8, 16, v3
	v_lshrrev_b32_e32 v10, 16, v2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[36:41], v[158:163], a[132:135], v12, v29 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_mov_b32 s1, 0xffff
	v_and_or_b32 v2, s1, v2, v8
	s_mov_b32 s6, 0xffff0000
	v_and_or_b32 v3, s6, v3, v10
	ds_write2_b32 v0, v2, v3 offset0:16 offset1:84
	v_accvgpr_read_b32 v2, a124
	v_accvgpr_read_b32 v3, a125
	v_cvt_pk_bf16_f32 v2, v2, v3
	v_accvgpr_read_b32 v3, a126
	v_accvgpr_read_b32 v8, a127
	v_cvt_pk_bf16_f32 v3, v3, v8
	v_mov_b32_dpp v8, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v10, v3 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v2, v2, v10, s[2:3]
	v_cndmask_b32_e64 v3, v8, v3, s[2:3]
	v_lshlrev_b32_e32 v8, 16, v3
	v_lshrrev_b32_e32 v10, 16, v2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[36:41], v[146:151], a[120:123], v12, v28 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_and_or_b32 v2, s1, v2, v8
	v_and_or_b32 v3, s6, v3, v10
	ds_write2_b32 v0, v2, v3 offset0:24 offset1:92
	v_accvgpr_read_b32 v2, a128
	v_accvgpr_read_b32 v3, a129
	v_cvt_pk_bf16_f32 v2, v2, v3
	v_accvgpr_read_b32 v3, a130
	v_accvgpr_read_b32 v8, a131
	v_cvt_pk_bf16_f32 v3, v3, v8
	v_mov_b32_dpp v8, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v10, v3 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v2, v2, v10, s[2:3]
	v_cndmask_b32_e64 v3, v8, v3, s[2:3]
	v_lshlrev_b32_e32 v8, 16, v3
	v_lshrrev_b32_e32 v10, 16, v2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[36:41], v[132:137], a[68:71], v12, v28 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_and_or_b32 v2, s1, v2, v8
	v_and_or_b32 v3, s6, v3, v10
	ds_write2_b32 v0, v2, v3 offset0:32 offset1:100
	v_accvgpr_read_b32 v2, a188
	v_accvgpr_read_b32 v3, a189
	v_cvt_pk_bf16_f32 v2, v2, v3
	v_accvgpr_read_b32 v3, a190
	v_accvgpr_read_b32 v8, a191
	v_cvt_pk_bf16_f32 v3, v3, v8
	v_mov_b32_dpp v8, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v10, v3 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v2, v2, v10, s[2:3]
	v_cndmask_b32_e64 v3, v8, v3, s[2:3]
	v_lshlrev_b32_e32 v8, 16, v3
	v_lshrrev_b32_e32 v10, 16, v2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[36:41], v[126:131], a[56:59], v12, v28 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_mov_b32 s1, 0xffff
	v_and_or_b32 v2, s1, v2, v8
	s_mov_b32 s6, 0xffff0000
	v_and_or_b32 v3, s6, v3, v10
	ds_write2_b32 v0, v2, v3 offset0:40 offset1:108
	v_accvgpr_read_b32 v2, a192
	v_accvgpr_read_b32 v3, a193
	v_cvt_pk_bf16_f32 v2, v2, v3
	v_accvgpr_read_b32 v3, a194
	v_accvgpr_read_b32 v8, a195
	v_cvt_pk_bf16_f32 v3, v3, v8
	v_mov_b32_dpp v8, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v10, v3 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v2, v2, v10, s[2:3]
	v_cndmask_b32_e64 v3, v8, v3, s[2:3]
	v_lshlrev_b32_e32 v8, 16, v3
	v_lshrrev_b32_e32 v10, 16, v2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[36:41], v[120:125], a[4:7], v12, v28 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_and_or_b32 v2, s1, v2, v8
	v_and_or_b32 v3, s6, v3, v10
	ds_write2_b32 v0, v2, v3 offset0:48 offset1:116
	v_accvgpr_read_b32 v2, a252
	v_accvgpr_read_b32 v3, a253
	v_cvt_pk_bf16_f32 v2, v2, v3
	v_accvgpr_read_b32 v3, a254
	v_accvgpr_read_b32 v8, a255
	v_cvt_pk_bf16_f32 v3, v3, v8
	v_mov_b32_dpp v8, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v10, v3 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v2, v2, v10, s[2:3]
	v_cndmask_b32_e64 v3, v8, v3, s[2:3]
	v_lshlrev_b32_e32 v8, 16, v3
	v_lshrrev_b32_e32 v10, 16, v2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[72:77], v[138:143], a[4:7], v4, v6 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_and_or_b32 v2, s1, v2, v8
	v_and_or_b32 v3, s6, v3, v10
	ds_write2_b32 v0, v2, v3 offset0:56 offset1:124
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[72:77], v[164:169], a[56:59], v4, v6 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_add_lshl_u32 v2, s0, v9, 2
	ds_read_b128 v[8:11], v2 offset:4096
	ds_read_b128 v[20:23], v2 offset:5184
	ds_read_b128 v[24:27], v2 offset:6272
	ds_read_b128 v[30:33], v2 offset:7360
	s_lshl_b32 s0, s4, 13
	s_lshl_b32 s1, s5, 1
	s_add_i32 s0, s0, s1
	v_add_u32_e32 v3, 0x8000, v1
	v_add_u32_e32 v34, 0x10000, v1
	v_add_u32_e32 v35, 0x18000, v1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[72:77], v[176:181], a[68:71], v4, v6 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(3)
	buffer_store_dwordx4 v[8:11], v1, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(2)
	buffer_store_dwordx4 v[20:23], v3, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[24:27], v34, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[30:33], v35, s[8:11], s0 offen nt
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[72:77], v[182:187], a[120:123], v4, v6 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[72:77], v[194:199], a[132:135], v4, v7 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[72:77], v[200:205], a[184:187], v4, v7 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[72:77], v[206:211], a[196:199], v4, v7 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[72:77], v[14:19], a[248:251], v4, v7 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[42:47], v[120:125], a[8:11], v12, v28 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v8, a4
	v_accvgpr_read_b32 v9, a5
	v_cvt_pk_bf16_f32 v8, v8, v9
	v_accvgpr_read_b32 v9, a6
	v_accvgpr_read_b32 v10, a7
	v_cvt_pk_bf16_f32 v9, v9, v10
	v_mov_b32_dpp v10, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v11, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v11, s[2:3]
	v_cndmask_b32_e64 v9, v10, v9, s[2:3]
	v_lshlrev_b32_e32 v10, 16, v9
	v_lshrrev_b32_e32 v11, 16, v8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[42:47], v[126:131], a[52:55], v12, v28 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 s0, 0xffff
	v_and_or_b32 v8, s0, v8, v10
	s_mov_b32 s5, 0xffff0000
	v_and_or_b32 v9, s5, v9, v11
	ds_write2_b32 v0, v8, v9 offset1:68
	v_accvgpr_read_b32 v8, a56
	v_accvgpr_read_b32 v9, a57
	v_cvt_pk_bf16_f32 v8, v8, v9
	v_accvgpr_read_b32 v9, a58
	v_accvgpr_read_b32 v10, a59
	v_cvt_pk_bf16_f32 v9, v9, v10
	v_mov_b32_dpp v10, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v11, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v11, s[2:3]
	v_cndmask_b32_e64 v9, v10, v9, s[2:3]
	v_lshlrev_b32_e32 v10, 16, v9
	v_lshrrev_b32_e32 v11, 16, v8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[42:47], v[132:137], a[72:75], v12, v28 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_and_or_b32 v8, s0, v8, v10
	v_and_or_b32 v9, s5, v9, v11
	ds_write2_b32 v0, v8, v9 offset0:8 offset1:76
	v_accvgpr_read_b32 v8, a68
	v_accvgpr_read_b32 v9, a69
	v_cvt_pk_bf16_f32 v8, v8, v9
	v_accvgpr_read_b32 v9, a70
	v_accvgpr_read_b32 v10, a71
	v_cvt_pk_bf16_f32 v9, v9, v10
	v_mov_b32_dpp v10, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v11, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v11, s[2:3]
	v_cndmask_b32_e64 v9, v10, v9, s[2:3]
	v_lshlrev_b32_e32 v10, 16, v9
	v_lshrrev_b32_e32 v11, 16, v8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[42:47], v[146:151], a[116:119], v12, v28 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_and_or_b32 v8, s0, v8, v10
	v_and_or_b32 v9, s5, v9, v11
	ds_write2_b32 v0, v8, v9 offset0:16 offset1:84
	v_accvgpr_read_b32 v8, a120
	v_accvgpr_read_b32 v9, a121
	v_cvt_pk_bf16_f32 v8, v8, v9
	v_accvgpr_read_b32 v9, a122
	v_accvgpr_read_b32 v10, a123
	v_cvt_pk_bf16_f32 v9, v9, v10
	v_mov_b32_dpp v10, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v11, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v11, s[2:3]
	v_cndmask_b32_e64 v9, v10, v9, s[2:3]
	v_lshlrev_b32_e32 v10, 16, v9
	v_lshrrev_b32_e32 v11, 16, v8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[42:47], v[158:163], a[136:139], v12, v29 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 s0, 0xffff
	v_and_or_b32 v8, s0, v8, v10
	s_mov_b32 s5, 0xffff0000
	v_and_or_b32 v9, s5, v9, v11
	ds_write2_b32 v0, v8, v9 offset0:24 offset1:92
	v_accvgpr_read_b32 v8, a132
	v_accvgpr_read_b32 v9, a133
	v_cvt_pk_bf16_f32 v8, v8, v9
	v_accvgpr_read_b32 v9, a134
	v_accvgpr_read_b32 v10, a135
	v_cvt_pk_bf16_f32 v9, v9, v10
	v_mov_b32_dpp v10, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v11, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v11, s[2:3]
	v_cndmask_b32_e64 v9, v10, v9, s[2:3]
	v_lshlrev_b32_e32 v10, 16, v9
	v_lshrrev_b32_e32 v11, 16, v8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[42:47], v[170:175], a[180:183], v12, v29 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_and_or_b32 v8, s0, v8, v10
	v_and_or_b32 v9, s5, v9, v11
	ds_write2_b32 v0, v8, v9 offset0:32 offset1:100
	v_accvgpr_read_b32 v8, a184
	v_accvgpr_read_b32 v9, a185
	v_cvt_pk_bf16_f32 v8, v8, v9
	v_accvgpr_read_b32 v9, a186
	v_accvgpr_read_b32 v10, a187
	v_cvt_pk_bf16_f32 v9, v9, v10
	v_mov_b32_dpp v10, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v11, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v11, s[2:3]
	v_cndmask_b32_e64 v9, v10, v9, s[2:3]
	v_lshlrev_b32_e32 v10, 16, v9
	v_lshrrev_b32_e32 v11, 16, v8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[42:47], v[188:193], a[200:203], v12, v29 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_and_or_b32 v8, s0, v8, v10
	v_and_or_b32 v9, s5, v9, v11
	ds_write2_b32 v0, v8, v9 offset0:40 offset1:108
	v_accvgpr_read_b32 v8, a196
	v_accvgpr_read_b32 v9, a197
	v_cvt_pk_bf16_f32 v8, v8, v9
	v_accvgpr_read_b32 v9, a198
	v_accvgpr_read_b32 v10, a199
	v_cvt_pk_bf16_f32 v9, v9, v10
	v_mov_b32_dpp v10, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v11, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v11, s[2:3]
	v_cndmask_b32_e64 v9, v10, v9, s[2:3]
	v_lshlrev_b32_e32 v10, 16, v9
	v_lshrrev_b32_e32 v11, 16, v8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[42:47], v[152:157], a[244:247], v12, v29 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_mov_b32 s0, 0xffff
	v_and_or_b32 v8, s0, v8, v10
	s_mov_b32 s5, 0xffff0000
	v_and_or_b32 v9, s5, v9, v11
	ds_write2_b32 v0, v8, v9 offset0:48 offset1:116
	v_accvgpr_read_b32 v8, a248
	v_accvgpr_read_b32 v9, a249
	v_cvt_pk_bf16_f32 v8, v8, v9
	v_accvgpr_read_b32 v9, a250
	v_accvgpr_read_b32 v10, a251
	v_cvt_pk_bf16_f32 v9, v9, v10
	v_mov_b32_dpp v10, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v11, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v11, s[2:3]
	v_cndmask_b32_e64 v9, v10, v9, s[2:3]
	v_lshlrev_b32_e32 v10, 16, v9
	v_lshrrev_b32_e32 v11, 16, v8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[84:89], v[14:19], a[244:247], v4, v7 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_and_or_b32 v8, s0, v8, v10
	v_and_or_b32 v9, s5, v9, v11
	ds_write2_b32 v0, v8, v9 offset0:56 offset1:124
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[84:89], v[206:211], a[200:203], v4, v7 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	ds_read_b128 v[8:11], v2 offset:4096
	ds_read_b128 v[20:23], v2 offset:5184
	ds_read_b128 v[24:27], v2 offset:6272
	ds_read_b128 v[30:33], v2 offset:7360
	s_add_i32 s0, s4, 16
	s_lshl_b32 s0, s0, 13
	s_add_i32 s0, s1, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[84:89], v[200:205], a[180:183], v4, v7 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(3)
	buffer_store_dwordx4 v[8:11], v1, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(2)
	buffer_store_dwordx4 v[20:23], v3, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[24:27], v34, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[30:33], v35, s[8:11], s0 offen nt
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[84:89], v[194:199], a[136:139], v4, v7 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[84:89], v[182:187], a[116:119], v4, v6 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[84:89], v[176:181], a[72:75], v4, v6 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[84:89], v[164:169], a[52:55], v4, v6 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[84:89], v[138:143], a[8:11], v4, v6 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[48:53], v[152:157], a[240:243], v12, v29 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_mov_b32 s0, 0xffff
	s_mov_b32 s5, 0xffff0000
	s_nop 7
	s_nop 0
	v_accvgpr_read_b32 v8, a8
	v_accvgpr_read_b32 v9, a9
	v_cvt_pk_bf16_f32 v8, v8, v9
	v_accvgpr_read_b32 v9, a10
	v_accvgpr_read_b32 v10, a11
	v_cvt_pk_bf16_f32 v9, v9, v10
	v_mov_b32_dpp v10, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v11, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v11, s[2:3]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[48:53], v[188:193], a[204:207], v12, v29 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_cndmask_b32_e64 v9, v10, v9, s[2:3]
	v_lshlrev_b32_e32 v10, 16, v9
	v_lshrrev_b32_e32 v11, 16, v8
	v_and_or_b32 v8, s0, v8, v10
	v_and_or_b32 v9, s5, v9, v11
	ds_write2_b32 v0, v8, v9 offset1:68
	v_accvgpr_read_b32 v8, a52
	v_accvgpr_read_b32 v9, a53
	v_cvt_pk_bf16_f32 v8, v8, v9
	v_accvgpr_read_b32 v9, a54
	v_accvgpr_read_b32 v10, a55
	v_cvt_pk_bf16_f32 v9, v9, v10
	v_mov_b32_dpp v10, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v11, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v11, s[2:3]
	v_cndmask_b32_e64 v9, v10, v9, s[2:3]
	v_lshlrev_b32_e32 v10, 16, v9
	v_lshrrev_b32_e32 v11, 16, v8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[48:53], v[170:175], a[176:179], v12, v29 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_and_or_b32 v8, s0, v8, v10
	v_and_or_b32 v9, s5, v9, v11
	ds_write2_b32 v0, v8, v9 offset0:8 offset1:76
	v_accvgpr_read_b32 v8, a72
	v_accvgpr_read_b32 v9, a73
	v_cvt_pk_bf16_f32 v8, v8, v9
	v_accvgpr_read_b32 v9, a74
	v_accvgpr_read_b32 v10, a75
	v_cvt_pk_bf16_f32 v9, v9, v10
	v_mov_b32_dpp v10, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v11, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v11, s[2:3]
	v_cndmask_b32_e64 v9, v10, v9, s[2:3]
	v_lshlrev_b32_e32 v10, 16, v9
	v_lshrrev_b32_e32 v11, 16, v8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[48:53], v[158:163], a[140:143], v12, v29 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_and_or_b32 v8, s0, v8, v10
	v_and_or_b32 v9, s5, v9, v11
	ds_write2_b32 v0, v8, v9 offset0:16 offset1:84
	v_accvgpr_read_b32 v8, a116
	v_accvgpr_read_b32 v9, a117
	v_cvt_pk_bf16_f32 v8, v8, v9
	v_accvgpr_read_b32 v9, a118
	v_accvgpr_read_b32 v10, a119
	v_cvt_pk_bf16_f32 v9, v9, v10
	v_mov_b32_dpp v10, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v11, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v11, s[2:3]
	v_cndmask_b32_e64 v9, v10, v9, s[2:3]
	v_lshlrev_b32_e32 v10, 16, v9
	v_lshrrev_b32_e32 v11, 16, v8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[48:53], v[146:151], a[112:115], v12, v28 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_mov_b32 s0, 0xffff
	v_and_or_b32 v8, s0, v8, v10
	s_mov_b32 s5, 0xffff0000
	v_and_or_b32 v9, s5, v9, v11
	ds_write2_b32 v0, v8, v9 offset0:24 offset1:92
	v_accvgpr_read_b32 v8, a136
	v_accvgpr_read_b32 v9, a137
	v_cvt_pk_bf16_f32 v8, v8, v9
	v_accvgpr_read_b32 v9, a138
	v_accvgpr_read_b32 v10, a139
	v_cvt_pk_bf16_f32 v9, v9, v10
	v_mov_b32_dpp v10, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v11, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v11, s[2:3]
	v_cndmask_b32_e64 v9, v10, v9, s[2:3]
	v_lshlrev_b32_e32 v10, 16, v9
	v_lshrrev_b32_e32 v11, 16, v8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[48:53], v[132:137], a[76:79], v12, v28 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_and_or_b32 v8, s0, v8, v10
	v_and_or_b32 v9, s5, v9, v11
	ds_write2_b32 v0, v8, v9 offset0:32 offset1:100
	v_accvgpr_read_b32 v8, a180
	v_accvgpr_read_b32 v9, a181
	v_cvt_pk_bf16_f32 v8, v8, v9
	v_accvgpr_read_b32 v9, a182
	v_accvgpr_read_b32 v10, a183
	v_cvt_pk_bf16_f32 v9, v9, v10
	v_mov_b32_dpp v10, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v11, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v11, s[2:3]
	v_cndmask_b32_e64 v9, v10, v9, s[2:3]
	v_lshlrev_b32_e32 v10, 16, v9
	v_lshrrev_b32_e32 v11, 16, v8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[48:53], v[126:131], a[48:51], v12, v28 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_and_or_b32 v8, s0, v8, v10
	v_and_or_b32 v9, s5, v9, v11
	ds_write2_b32 v0, v8, v9 offset0:40 offset1:108
	v_accvgpr_read_b32 v8, a200
	v_accvgpr_read_b32 v9, a201
	v_cvt_pk_bf16_f32 v8, v8, v9
	v_accvgpr_read_b32 v9, a202
	v_accvgpr_read_b32 v10, a203
	v_cvt_pk_bf16_f32 v9, v9, v10
	v_mov_b32_dpp v10, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v11, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v11, s[2:3]
	v_cndmask_b32_e64 v9, v10, v9, s[2:3]
	v_lshlrev_b32_e32 v10, 16, v9
	v_lshrrev_b32_e32 v11, 16, v8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[48:53], v[120:125], a[12:15], v12, v28 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 s0, 0xffff
	v_and_or_b32 v8, s0, v8, v10
	s_mov_b32 s5, 0xffff0000
	v_and_or_b32 v9, s5, v9, v11
	ds_write2_b32 v0, v8, v9 offset0:48 offset1:116
	v_accvgpr_read_b32 v8, a244
	v_accvgpr_read_b32 v9, a245
	v_cvt_pk_bf16_f32 v8, v8, v9
	v_accvgpr_read_b32 v9, a246
	v_accvgpr_read_b32 v10, a247
	v_cvt_pk_bf16_f32 v9, v9, v10
	v_mov_b32_dpp v10, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v11, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v11, s[2:3]
	v_cndmask_b32_e64 v9, v10, v9, s[2:3]
	v_lshlrev_b32_e32 v10, 16, v9
	v_lshrrev_b32_e32 v11, 16, v8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[90:95], v[138:143], a[12:15], v4, v6 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_and_or_b32 v8, s0, v8, v10
	v_and_or_b32 v9, s5, v9, v11
	ds_write2_b32 v0, v8, v9 offset0:56 offset1:124
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[90:95], v[164:169], a[48:51], v4, v6 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	ds_read_b128 v[8:11], v2 offset:4096
	ds_read_b128 v[20:23], v2 offset:5184
	ds_read_b128 v[24:27], v2 offset:6272
	ds_read_b128 v[30:33], v2 offset:7360
	s_add_i32 s0, s4, 32
	s_lshl_b32 s0, s0, 13
	s_add_i32 s0, s1, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[90:95], v[176:181], a[76:79], v4, v6 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(3)
	buffer_store_dwordx4 v[8:11], v1, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(2)
	buffer_store_dwordx4 v[20:23], v3, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[24:27], v34, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[30:33], v35, s[8:11], s0 offen nt
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[90:95], v[182:187], a[112:115], v4, v6 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[90:95], v[194:199], a[140:143], v4, v7 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[90:95], v[200:205], a[176:179], v4, v7 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[90:95], v[206:211], a[204:207], v4, v7 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[90:95], v[14:19], a[240:243], v4, v7 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[54:59], v[120:125], a[16:19], v13, v28 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a12
	v_accvgpr_read_b32 v8, a13
	v_cvt_pk_bf16_f32 v8, v4, v8
	v_accvgpr_read_b32 v4, a14
	v_accvgpr_read_b32 v9, a15
	v_cvt_pk_bf16_f32 v9, v4, v9
	v_mov_b32_dpp v4, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v10, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v10, s[2:3]
	v_cndmask_b32_e64 v4, v4, v9, s[2:3]
	v_lshlrev_b32_e32 v9, 16, v4
	v_lshrrev_b32_e32 v10, 16, v8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[54:59], v[126:131], a[44:47], v13, v28 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_mov_b32 s0, 0xffff
	v_and_or_b32 v8, s0, v8, v9
	s_mov_b32 s5, 0xffff0000
	v_and_or_b32 v4, s5, v4, v10
	ds_write2_b32 v0, v8, v4 offset1:68
	v_accvgpr_read_b32 v4, a48
	v_accvgpr_read_b32 v8, a49
	v_cvt_pk_bf16_f32 v8, v4, v8
	v_accvgpr_read_b32 v4, a50
	v_accvgpr_read_b32 v9, a51
	v_cvt_pk_bf16_f32 v9, v4, v9
	v_mov_b32_dpp v4, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v10, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v10, s[2:3]
	v_cndmask_b32_e64 v4, v4, v9, s[2:3]
	v_lshlrev_b32_e32 v9, 16, v4
	v_lshrrev_b32_e32 v10, 16, v8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[54:59], v[132:137], a[80:83], v13, v28 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_and_or_b32 v8, s0, v8, v9
	v_and_or_b32 v4, s5, v4, v10
	ds_write2_b32 v0, v8, v4 offset0:8 offset1:76
	v_accvgpr_read_b32 v4, a76
	v_accvgpr_read_b32 v8, a77
	v_cvt_pk_bf16_f32 v8, v4, v8
	v_accvgpr_read_b32 v4, a78
	v_accvgpr_read_b32 v9, a79
	v_cvt_pk_bf16_f32 v9, v4, v9
	v_mov_b32_dpp v4, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v10, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v10, s[2:3]
	v_cndmask_b32_e64 v4, v4, v9, s[2:3]
	v_lshlrev_b32_e32 v9, 16, v4
	v_lshrrev_b32_e32 v10, 16, v8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[54:59], v[146:151], a[108:111], v13, v28 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_and_or_b32 v8, s0, v8, v9
	v_and_or_b32 v4, s5, v4, v10
	ds_write2_b32 v0, v8, v4 offset0:16 offset1:84
	v_accvgpr_read_b32 v4, a112
	v_accvgpr_read_b32 v8, a113
	v_cvt_pk_bf16_f32 v8, v4, v8
	v_accvgpr_read_b32 v4, a114
	v_accvgpr_read_b32 v9, a115
	v_cvt_pk_bf16_f32 v9, v4, v9
	v_mov_b32_dpp v4, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v10, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v10, s[2:3]
	v_cndmask_b32_e64 v4, v4, v9, s[2:3]
	v_lshlrev_b32_e32 v9, 16, v4
	v_lshrrev_b32_e32 v10, 16, v8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[54:59], v[158:163], a[144:147], v13, v29 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_mov_b32 s0, 0xffff
	v_and_or_b32 v8, s0, v8, v9
	s_mov_b32 s5, 0xffff0000
	v_and_or_b32 v4, s5, v4, v10
	ds_write2_b32 v0, v8, v4 offset0:24 offset1:92
	v_accvgpr_read_b32 v4, a140
	v_accvgpr_read_b32 v8, a141
	v_cvt_pk_bf16_f32 v8, v4, v8
	v_accvgpr_read_b32 v4, a142
	v_accvgpr_read_b32 v9, a143
	v_cvt_pk_bf16_f32 v9, v4, v9
	v_mov_b32_dpp v4, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v10, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v10, s[2:3]
	v_cndmask_b32_e64 v4, v4, v9, s[2:3]
	v_lshlrev_b32_e32 v9, 16, v4
	v_lshrrev_b32_e32 v10, 16, v8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[54:59], v[170:175], a[172:175], v13, v29 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_and_or_b32 v8, s0, v8, v9
	v_and_or_b32 v4, s5, v4, v10
	ds_write2_b32 v0, v8, v4 offset0:32 offset1:100
	v_accvgpr_read_b32 v4, a176
	v_accvgpr_read_b32 v8, a177
	v_cvt_pk_bf16_f32 v8, v4, v8
	v_accvgpr_read_b32 v4, a178
	v_accvgpr_read_b32 v9, a179
	v_cvt_pk_bf16_f32 v9, v4, v9
	v_mov_b32_dpp v4, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v10, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v10, s[2:3]
	v_cndmask_b32_e64 v4, v4, v9, s[2:3]
	v_lshlrev_b32_e32 v9, 16, v4
	v_lshrrev_b32_e32 v10, 16, v8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[54:59], v[188:193], a[208:211], v13, v29 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_and_or_b32 v8, s0, v8, v9
	v_and_or_b32 v4, s5, v4, v10
	ds_write2_b32 v0, v8, v4 offset0:40 offset1:108
	v_accvgpr_read_b32 v4, a204
	v_accvgpr_read_b32 v8, a205
	v_cvt_pk_bf16_f32 v8, v4, v8
	v_accvgpr_read_b32 v4, a206
	v_accvgpr_read_b32 v9, a207
	v_cvt_pk_bf16_f32 v9, v4, v9
	v_mov_b32_dpp v4, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v10, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v10, s[2:3]
	v_cndmask_b32_e64 v4, v4, v9, s[2:3]
	v_lshlrev_b32_e32 v9, 16, v4
	v_lshrrev_b32_e32 v10, 16, v8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[54:59], v[152:157], a[236:239], v13, v29 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 s0, 0xffff
	v_and_or_b32 v8, s0, v8, v9
	s_mov_b32 s5, 0xffff0000
	v_and_or_b32 v4, s5, v4, v10
	ds_write2_b32 v0, v8, v4 offset0:48 offset1:116
	v_accvgpr_read_b32 v4, a240
	v_accvgpr_read_b32 v8, a241
	v_cvt_pk_bf16_f32 v8, v4, v8
	v_accvgpr_read_b32 v4, a242
	v_accvgpr_read_b32 v9, a243
	v_cvt_pk_bf16_f32 v9, v4, v9
	v_mov_b32_dpp v4, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v10, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v10, s[2:3]
	v_cndmask_b32_e64 v4, v4, v9, s[2:3]
	v_lshlrev_b32_e32 v9, 16, v4
	v_lshrrev_b32_e32 v10, 16, v8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[96:101], v[14:19], a[236:239], v5, v7 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_and_or_b32 v8, s0, v8, v9
	v_and_or_b32 v4, s5, v4, v10
	ds_write2_b32 v0, v8, v4 offset0:56 offset1:124
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[96:101], v[206:211], a[208:211], v5, v7 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	ds_read_b128 v[8:11], v2 offset:4096
	ds_read_b128 v[20:23], v2 offset:5184
	ds_read_b128 v[24:27], v2 offset:6272
	ds_read_b128 v[30:33], v2 offset:7360
	s_add_i32 s0, s4, 48
	s_lshl_b32 s0, s0, 13
	s_add_i32 s0, s1, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[96:101], v[200:205], a[172:175], v5, v7 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(3)
	buffer_store_dwordx4 v[8:11], v1, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(2)
	buffer_store_dwordx4 v[20:23], v3, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[24:27], v34, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[30:33], v35, s[8:11], s0 offen nt
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[96:101], v[194:199], a[144:147], v5, v7 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[96:101], v[182:187], a[108:111], v5, v6 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[96:101], v[176:181], a[80:83], v5, v6 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[96:101], v[164:169], a[44:47], v5, v6 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[96:101], v[138:143], a[16:19], v5, v6 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[60:65], v[152:157], a[232:235], v13, v29 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 s0, 0xffff
	s_mov_b32 s5, 0xffff0000
	s_nop 7
	s_nop 0
	v_accvgpr_read_b32 v4, a16
	v_accvgpr_read_b32 v8, a17
	v_cvt_pk_bf16_f32 v8, v4, v8
	v_accvgpr_read_b32 v4, a18
	v_accvgpr_read_b32 v9, a19
	v_cvt_pk_bf16_f32 v9, v4, v9
	v_mov_b32_dpp v4, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v10, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v10, s[2:3]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[60:65], v[188:193], a[212:215], v13, v29 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_cndmask_b32_e64 v4, v4, v9, s[2:3]
	v_lshlrev_b32_e32 v9, 16, v4
	v_lshrrev_b32_e32 v10, 16, v8
	v_and_or_b32 v8, s0, v8, v9
	v_and_or_b32 v4, s5, v4, v10
	ds_write2_b32 v0, v8, v4 offset1:68
	v_accvgpr_read_b32 v4, a44
	v_accvgpr_read_b32 v8, a45
	v_cvt_pk_bf16_f32 v8, v4, v8
	v_accvgpr_read_b32 v4, a46
	v_accvgpr_read_b32 v9, a47
	v_cvt_pk_bf16_f32 v9, v4, v9
	v_mov_b32_dpp v4, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v10, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v10, s[2:3]
	v_cndmask_b32_e64 v4, v4, v9, s[2:3]
	v_lshlrev_b32_e32 v9, 16, v4
	v_lshrrev_b32_e32 v10, 16, v8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[60:65], v[170:175], a[168:171], v13, v29 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_and_or_b32 v8, s0, v8, v9
	v_and_or_b32 v4, s5, v4, v10
	ds_write2_b32 v0, v8, v4 offset0:8 offset1:76
	v_accvgpr_read_b32 v4, a80
	v_accvgpr_read_b32 v8, a81
	v_cvt_pk_bf16_f32 v8, v4, v8
	v_accvgpr_read_b32 v4, a82
	v_accvgpr_read_b32 v9, a83
	v_cvt_pk_bf16_f32 v9, v4, v9
	v_mov_b32_dpp v4, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v10, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v10, s[2:3]
	v_cndmask_b32_e64 v4, v4, v9, s[2:3]
	v_lshlrev_b32_e32 v9, 16, v4
	v_lshrrev_b32_e32 v10, 16, v8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[60:65], v[158:163], a[148:151], v13, v29 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_and_or_b32 v8, s0, v8, v9
	v_and_or_b32 v4, s5, v4, v10
	ds_write2_b32 v0, v8, v4 offset0:16 offset1:84
	v_accvgpr_read_b32 v4, a108
	v_accvgpr_read_b32 v8, a109
	v_cvt_pk_bf16_f32 v8, v4, v8
	v_accvgpr_read_b32 v4, a110
	v_accvgpr_read_b32 v9, a111
	v_cvt_pk_bf16_f32 v9, v4, v9
	v_mov_b32_dpp v4, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v10, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v10, s[2:3]
	v_cndmask_b32_e64 v4, v4, v9, s[2:3]
	v_lshlrev_b32_e32 v9, 16, v4
	v_lshrrev_b32_e32 v10, 16, v8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[60:65], v[146:151], a[104:107], v13, v28 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 s0, 0xffff
	v_and_or_b32 v8, s0, v8, v9
	s_mov_b32 s5, 0xffff0000
	v_and_or_b32 v4, s5, v4, v10
	ds_write2_b32 v0, v8, v4 offset0:24 offset1:92
	v_accvgpr_read_b32 v4, a144
	v_accvgpr_read_b32 v8, a145
	v_cvt_pk_bf16_f32 v8, v4, v8
	v_accvgpr_read_b32 v4, a146
	v_accvgpr_read_b32 v9, a147
	v_cvt_pk_bf16_f32 v9, v4, v9
	v_mov_b32_dpp v4, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v10, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v10, s[2:3]
	v_cndmask_b32_e64 v4, v4, v9, s[2:3]
	v_lshlrev_b32_e32 v9, 16, v4
	v_lshrrev_b32_e32 v10, 16, v8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[60:65], v[132:137], a[84:87], v13, v28 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_and_or_b32 v8, s0, v8, v9
	v_and_or_b32 v4, s5, v4, v10
	ds_write2_b32 v0, v8, v4 offset0:32 offset1:100
	v_accvgpr_read_b32 v4, a172
	v_accvgpr_read_b32 v8, a173
	v_cvt_pk_bf16_f32 v8, v4, v8
	v_accvgpr_read_b32 v4, a174
	v_accvgpr_read_b32 v9, a175
	v_cvt_pk_bf16_f32 v9, v4, v9
	v_mov_b32_dpp v4, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v10, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v10, s[2:3]
	v_cndmask_b32_e64 v4, v4, v9, s[2:3]
	v_lshlrev_b32_e32 v9, 16, v4
	v_lshrrev_b32_e32 v10, 16, v8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[60:65], v[126:131], a[40:43], v13, v28 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_and_or_b32 v8, s0, v8, v9
	v_and_or_b32 v4, s5, v4, v10
	ds_write2_b32 v0, v8, v4 offset0:40 offset1:108
	v_accvgpr_read_b32 v4, a208
	v_accvgpr_read_b32 v8, a209
	v_cvt_pk_bf16_f32 v8, v4, v8
	v_accvgpr_read_b32 v4, a210
	v_accvgpr_read_b32 v9, a211
	v_cvt_pk_bf16_f32 v9, v4, v9
	v_mov_b32_dpp v4, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v10, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v10, s[2:3]
	v_cndmask_b32_e64 v4, v4, v9, s[2:3]
	v_lshlrev_b32_e32 v9, 16, v4
	v_lshrrev_b32_e32 v10, 16, v8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[60:65], v[120:125], a[20:23], v13, v28 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_mov_b32 s0, 0xffff
	v_and_or_b32 v8, s0, v8, v9
	s_mov_b32 s5, 0xffff0000
	v_and_or_b32 v4, s5, v4, v10
	ds_write2_b32 v0, v8, v4 offset0:48 offset1:116
	v_accvgpr_read_b32 v4, a236
	v_accvgpr_read_b32 v8, a237
	v_cvt_pk_bf16_f32 v8, v4, v8
	v_accvgpr_read_b32 v4, a238
	v_accvgpr_read_b32 v9, a239
	v_cvt_pk_bf16_f32 v9, v4, v9
	v_mov_b32_dpp v4, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v10, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v10, s[2:3]
	v_cndmask_b32_e64 v4, v4, v9, s[2:3]
	v_lshlrev_b32_e32 v9, 16, v4
	v_lshrrev_b32_e32 v10, 16, v8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[102:107], v[138:143], a[20:23], v5, v6 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_and_or_b32 v8, s0, v8, v9
	v_and_or_b32 v4, s5, v4, v10
	ds_write2_b32 v0, v8, v4 offset0:56 offset1:124
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[102:107], v[164:169], a[40:43], v5, v6 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	ds_read_b128 v[8:11], v2 offset:4096
	ds_read_b128 v[20:23], v2 offset:5184
	ds_read_b128 v[24:27], v2 offset:6272
	ds_read_b128 v[30:33], v2 offset:7360
	s_add_i32 s0, s4, 64
	s_lshl_b32 s0, s0, 13
	s_add_i32 s0, s1, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[102:107], v[176:181], a[84:87], v5, v6 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(3)
	buffer_store_dwordx4 v[8:11], v1, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(2)
	buffer_store_dwordx4 v[20:23], v3, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[24:27], v34, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[30:33], v35, s[8:11], s0 offen nt
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[102:107], v[182:187], a[104:107], v5, v6 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[102:107], v[194:199], a[148:151], v5, v7 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[102:107], v[200:205], a[168:171], v5, v7 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[102:107], v[206:211], a[212:215], v5, v7 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[102:107], v[14:19], a[232:235], v5, v7 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[66:71], v[120:125], a[24:27], v13, v28 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a20
	v_accvgpr_read_b32 v8, a21
	v_cvt_pk_bf16_f32 v8, v4, v8
	v_accvgpr_read_b32 v4, a22
	v_accvgpr_read_b32 v9, a23
	v_cvt_pk_bf16_f32 v9, v4, v9
	v_mov_b32_dpp v4, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v10, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v10, s[2:3]
	v_cndmask_b32_e64 v4, v4, v9, s[2:3]
	v_lshlrev_b32_e32 v9, 16, v4
	v_lshrrev_b32_e32 v10, 16, v8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[66:71], v[126:131], a[36:39], v13, v28 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 s0, 0xffff
	v_and_or_b32 v8, s0, v8, v9
	s_mov_b32 s5, 0xffff0000
	v_and_or_b32 v4, s5, v4, v10
	ds_write2_b32 v0, v8, v4 offset1:68
	v_accvgpr_read_b32 v4, a40
	v_accvgpr_read_b32 v8, a41
	v_cvt_pk_bf16_f32 v8, v4, v8
	v_accvgpr_read_b32 v4, a42
	v_accvgpr_read_b32 v9, a43
	v_cvt_pk_bf16_f32 v9, v4, v9
	v_mov_b32_dpp v4, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v10, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v10, s[2:3]
	v_cndmask_b32_e64 v4, v4, v9, s[2:3]
	v_lshlrev_b32_e32 v9, 16, v4
	v_lshrrev_b32_e32 v10, 16, v8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[66:71], v[132:137], a[88:91], v13, v28 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_and_or_b32 v8, s0, v8, v9
	v_and_or_b32 v4, s5, v4, v10
	ds_write2_b32 v0, v8, v4 offset0:8 offset1:76
	v_accvgpr_read_b32 v4, a84
	v_accvgpr_read_b32 v8, a85
	v_cvt_pk_bf16_f32 v8, v4, v8
	v_accvgpr_read_b32 v4, a86
	v_accvgpr_read_b32 v9, a87
	v_cvt_pk_bf16_f32 v9, v4, v9
	v_mov_b32_dpp v4, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v10, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v10, s[2:3]
	v_cndmask_b32_e64 v4, v4, v9, s[2:3]
	v_lshlrev_b32_e32 v9, 16, v4
	v_lshrrev_b32_e32 v10, 16, v8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[66:71], v[146:151], a[100:103], v13, v28 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_and_or_b32 v8, s0, v8, v9
	v_and_or_b32 v4, s5, v4, v10
	ds_write2_b32 v0, v8, v4 offset0:16 offset1:84
	v_accvgpr_read_b32 v4, a104
	v_accvgpr_read_b32 v8, a105
	v_cvt_pk_bf16_f32 v8, v4, v8
	v_accvgpr_read_b32 v4, a106
	v_accvgpr_read_b32 v9, a107
	v_cvt_pk_bf16_f32 v9, v4, v9
	v_mov_b32_dpp v4, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v10, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v10, s[2:3]
	v_cndmask_b32_e64 v4, v4, v9, s[2:3]
	v_lshlrev_b32_e32 v9, 16, v4
	v_lshrrev_b32_e32 v10, 16, v8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[66:71], v[158:163], a[152:155], v13, v29 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 s0, 0xffff
	v_and_or_b32 v8, s0, v8, v9
	s_mov_b32 s5, 0xffff0000
	v_and_or_b32 v4, s5, v4, v10
	ds_write2_b32 v0, v8, v4 offset0:24 offset1:92
	v_accvgpr_read_b32 v4, a148
	v_accvgpr_read_b32 v8, a149
	v_cvt_pk_bf16_f32 v8, v4, v8
	v_accvgpr_read_b32 v4, a150
	v_accvgpr_read_b32 v9, a151
	v_cvt_pk_bf16_f32 v9, v4, v9
	v_mov_b32_dpp v4, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v10, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v10, s[2:3]
	v_cndmask_b32_e64 v4, v4, v9, s[2:3]
	v_lshlrev_b32_e32 v9, 16, v4
	v_lshrrev_b32_e32 v10, 16, v8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[66:71], v[170:175], a[164:167], v13, v29 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_and_or_b32 v8, s0, v8, v9
	v_and_or_b32 v4, s5, v4, v10
	ds_write2_b32 v0, v8, v4 offset0:32 offset1:100
	v_accvgpr_read_b32 v4, a168
	v_accvgpr_read_b32 v8, a169
	v_cvt_pk_bf16_f32 v8, v4, v8
	v_accvgpr_read_b32 v4, a170
	v_accvgpr_read_b32 v9, a171
	v_cvt_pk_bf16_f32 v9, v4, v9
	v_mov_b32_dpp v4, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v10, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v10, s[2:3]
	v_cndmask_b32_e64 v4, v4, v9, s[2:3]
	v_lshlrev_b32_e32 v9, 16, v4
	v_lshrrev_b32_e32 v10, 16, v8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[66:71], v[188:193], a[216:219], v13, v29 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_and_or_b32 v8, s0, v8, v9
	v_and_or_b32 v4, s5, v4, v10
	ds_write2_b32 v0, v8, v4 offset0:40 offset1:108
	v_accvgpr_read_b32 v4, a212
	v_accvgpr_read_b32 v8, a213
	v_cvt_pk_bf16_f32 v8, v4, v8
	v_accvgpr_read_b32 v4, a214
	v_accvgpr_read_b32 v9, a215
	v_cvt_pk_bf16_f32 v9, v4, v9
	v_mov_b32_dpp v4, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v10, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v10, s[2:3]
	v_cndmask_b32_e64 v4, v4, v9, s[2:3]
	v_lshlrev_b32_e32 v9, 16, v4
	v_lshrrev_b32_e32 v10, 16, v8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[66:71], v[152:157], a[228:231], v13, v29 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_mov_b32 s0, 0xffff
	v_and_or_b32 v8, s0, v8, v9
	s_mov_b32 s5, 0xffff0000
	v_and_or_b32 v4, s5, v4, v10
	ds_write2_b32 v0, v8, v4 offset0:48 offset1:116
	v_accvgpr_read_b32 v4, a232
	v_accvgpr_read_b32 v8, a233
	v_cvt_pk_bf16_f32 v8, v4, v8
	v_accvgpr_read_b32 v4, a234
	v_accvgpr_read_b32 v9, a235
	v_cvt_pk_bf16_f32 v9, v4, v9
	v_mov_b32_dpp v4, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v10, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v10, s[2:3]
	v_cndmask_b32_e64 v4, v4, v9, s[2:3]
	v_lshlrev_b32_e32 v9, 16, v4
	v_lshrrev_b32_e32 v10, 16, v8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[108:113], v[14:19], a[228:231], v5, v7 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_and_or_b32 v8, s0, v8, v9
	v_and_or_b32 v4, s5, v4, v10
	ds_write2_b32 v0, v8, v4 offset0:56 offset1:124
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[108:113], v[206:211], a[216:219], v5, v7 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	ds_read_b128 v[8:11], v2 offset:4096
	ds_read_b128 v[20:23], v2 offset:5184
	ds_read_b128 v[24:27], v2 offset:6272
	ds_read_b128 v[30:33], v2 offset:7360
	s_add_i32 s0, s4, 0x50
	s_lshl_b32 s0, s0, 13
	s_add_i32 s0, s1, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[108:113], v[200:205], a[164:167], v5, v7 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(3)
	buffer_store_dwordx4 v[8:11], v1, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(2)
	buffer_store_dwordx4 v[20:23], v3, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[24:27], v34, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[30:33], v35, s[8:11], s0 offen nt
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[108:113], v[194:199], a[152:155], v5, v7 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[108:113], v[182:187], a[100:103], v5, v6 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[108:113], v[176:181], a[88:91], v5, v6 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[108:113], v[164:169], a[36:39], v5, v6 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[108:113], v[138:143], a[24:27], v5, v6 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[78:83], v[152:157], a[224:227], v13, v29 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_mov_b32 s0, 0xffff
	s_mov_b32 s5, 0xffff0000
	s_nop 7
	s_nop 0
	v_accvgpr_read_b32 v4, a24
	v_accvgpr_read_b32 v8, a25
	v_cvt_pk_bf16_f32 v8, v4, v8
	v_accvgpr_read_b32 v4, a26
	v_accvgpr_read_b32 v9, a27
	v_cvt_pk_bf16_f32 v9, v4, v9
	v_mov_b32_dpp v4, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v10, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v10, s[2:3]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[78:83], v[188:193], a[220:223], v13, v29 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_cndmask_b32_e64 v4, v4, v9, s[2:3]
	v_lshlrev_b32_e32 v9, 16, v4
	v_lshrrev_b32_e32 v10, 16, v8
	v_and_or_b32 v8, s0, v8, v9
	v_and_or_b32 v4, s5, v4, v10
	ds_write2_b32 v0, v8, v4 offset1:68
	v_accvgpr_read_b32 v4, a36
	v_accvgpr_read_b32 v8, a37
	v_cvt_pk_bf16_f32 v8, v4, v8
	v_accvgpr_read_b32 v4, a38
	v_accvgpr_read_b32 v9, a39
	v_cvt_pk_bf16_f32 v9, v4, v9
	v_mov_b32_dpp v4, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v10, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v10, s[2:3]
	v_cndmask_b32_e64 v4, v4, v9, s[2:3]
	v_lshlrev_b32_e32 v9, 16, v4
	v_lshrrev_b32_e32 v10, 16, v8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[78:83], v[170:175], a[160:163], v13, v29 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_and_or_b32 v8, s0, v8, v9
	v_and_or_b32 v4, s5, v4, v10
	ds_write2_b32 v0, v8, v4 offset0:8 offset1:76
	v_accvgpr_read_b32 v4, a88
	v_accvgpr_read_b32 v8, a89
	v_cvt_pk_bf16_f32 v8, v4, v8
	v_accvgpr_read_b32 v4, a90
	v_accvgpr_read_b32 v9, a91
	v_cvt_pk_bf16_f32 v9, v4, v9
	v_mov_b32_dpp v4, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v10, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v10, s[2:3]
	v_cndmask_b32_e64 v4, v4, v9, s[2:3]
	v_lshlrev_b32_e32 v9, 16, v4
	v_lshrrev_b32_e32 v10, 16, v8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[78:83], v[158:163], a[156:159], v13, v29 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_and_or_b32 v8, s0, v8, v9
	v_and_or_b32 v4, s5, v4, v10
	ds_write2_b32 v0, v8, v4 offset0:16 offset1:84
	v_accvgpr_read_b32 v4, a100
	v_accvgpr_read_b32 v8, a101
	v_cvt_pk_bf16_f32 v8, v4, v8
	v_accvgpr_read_b32 v4, a102
	v_accvgpr_read_b32 v9, a103
	v_cvt_pk_bf16_f32 v9, v4, v9
	v_mov_b32_dpp v4, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v10, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v10, s[2:3]
	v_cndmask_b32_e64 v4, v4, v9, s[2:3]
	v_lshlrev_b32_e32 v9, 16, v4
	v_lshrrev_b32_e32 v10, 16, v8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[78:83], v[146:151], a[96:99], v13, v28 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_mov_b32 s0, 0xffff
	v_and_or_b32 v8, s0, v8, v9
	s_mov_b32 s5, 0xffff0000
	v_and_or_b32 v4, s5, v4, v10
	ds_write2_b32 v0, v8, v4 offset0:24 offset1:92
	v_accvgpr_read_b32 v4, a152
	v_accvgpr_read_b32 v8, a153
	v_cvt_pk_bf16_f32 v8, v4, v8
	v_accvgpr_read_b32 v4, a154
	v_accvgpr_read_b32 v9, a155
	v_cvt_pk_bf16_f32 v9, v4, v9
	v_mov_b32_dpp v4, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v10, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v10, s[2:3]
	v_cndmask_b32_e64 v4, v4, v9, s[2:3]
	v_lshlrev_b32_e32 v9, 16, v4
	v_lshrrev_b32_e32 v10, 16, v8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[78:83], v[132:137], a[92:95], v13, v28 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_and_or_b32 v8, s0, v8, v9
	v_and_or_b32 v4, s5, v4, v10
	ds_write2_b32 v0, v8, v4 offset0:32 offset1:100
	v_accvgpr_read_b32 v4, a164
	v_accvgpr_read_b32 v8, a165
	v_cvt_pk_bf16_f32 v8, v4, v8
	v_accvgpr_read_b32 v4, a166
	v_accvgpr_read_b32 v9, a167
	v_cvt_pk_bf16_f32 v9, v4, v9
	v_mov_b32_dpp v4, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v10, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v10, s[2:3]
	v_cndmask_b32_e64 v4, v4, v9, s[2:3]
	v_lshlrev_b32_e32 v9, 16, v4
	v_lshrrev_b32_e32 v10, 16, v8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[78:83], v[126:131], a[32:35], v13, v28 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_and_or_b32 v8, s0, v8, v9
	v_and_or_b32 v4, s5, v4, v10
	ds_write2_b32 v0, v8, v4 offset0:40 offset1:108
	v_accvgpr_read_b32 v4, a216
	v_accvgpr_read_b32 v8, a217
	v_cvt_pk_bf16_f32 v8, v4, v8
	v_accvgpr_read_b32 v4, a218
	v_accvgpr_read_b32 v9, a219
	v_cvt_pk_bf16_f32 v9, v4, v9
	v_mov_b32_dpp v4, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v10, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v10, s[2:3]
	v_cndmask_b32_e64 v4, v4, v9, s[2:3]
	v_lshlrev_b32_e32 v9, 16, v4
	v_lshrrev_b32_e32 v10, 16, v8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[78:83], v[120:125], a[28:31], v13, v28 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 s0, 0xffff
	v_and_or_b32 v8, s0, v8, v9
	s_mov_b32 s5, 0xffff0000
	v_and_or_b32 v4, s5, v4, v10
	ds_write2_b32 v0, v8, v4 offset0:48 offset1:116
	v_accvgpr_read_b32 v4, a228
	v_accvgpr_read_b32 v8, a229
	v_cvt_pk_bf16_f32 v8, v4, v8
	v_accvgpr_read_b32 v4, a230
	v_accvgpr_read_b32 v9, a231
	v_cvt_pk_bf16_f32 v9, v4, v9
	v_mov_b32_dpp v4, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v10, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v10, s[2:3]
	v_cndmask_b32_e64 v4, v4, v9, s[2:3]
	v_lshlrev_b32_e32 v9, 16, v4
	v_lshrrev_b32_e32 v10, 16, v8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[114:119], v[138:143], a[28:31], v5, v6 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_and_or_b32 v8, s0, v8, v9
	v_and_or_b32 v4, s5, v4, v10
	ds_write2_b32 v0, v8, v4 offset0:56 offset1:124
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[114:119], v[164:169], a[32:35], v5, v6 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	ds_read_b128 v[8:11], v2 offset:4096
	ds_read_b128 v[20:23], v2 offset:5184
	ds_read_b128 v[24:27], v2 offset:6272
	ds_read_b128 v[28:31], v2 offset:7360
	s_add_i32 s0, s4, 0x60
	s_lshl_b32 s0, s0, 13
	s_add_i32 s0, s1, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[114:119], v[176:181], a[92:95], v5, v6 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(3)
	buffer_store_dwordx4 v[8:11], v1, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(2)
	buffer_store_dwordx4 v[20:23], v3, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[24:27], v34, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[28:31], v35, s[8:11], s0 offen nt
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[114:119], v[182:187], a[96:99], v5, v6 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[114:119], v[194:199], a[156:159], v5, v7 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[114:119], v[200:205], a[160:163], v5, v7 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[114:119], v[206:211], a[220:223], v5, v7 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[114:119], v[14:19], a[224:227], v5, v7 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a28
	v_accvgpr_read_b32 v5, a29
	v_cvt_pk_bf16_f32 v4, v4, v5
	v_accvgpr_read_b32 v5, a30
	v_accvgpr_read_b32 v6, a31
	v_cvt_pk_bf16_f32 v5, v5, v6
	v_accvgpr_read_b32 v6, a32
	v_accvgpr_read_b32 v7, a33
	v_cvt_pk_bf16_f32 v6, v6, v7
	v_accvgpr_read_b32 v7, a34
	v_accvgpr_read_b32 v8, a35
	v_cvt_pk_bf16_f32 v7, v7, v8
	v_accvgpr_read_b32 v8, a92
	v_accvgpr_read_b32 v9, a93
	v_cvt_pk_bf16_f32 v8, v8, v9
	v_accvgpr_read_b32 v9, a94
	v_accvgpr_read_b32 v10, a95
	v_cvt_pk_bf16_f32 v9, v9, v10
	v_accvgpr_read_b32 v10, a96
	v_accvgpr_read_b32 v11, a97
	v_cvt_pk_bf16_f32 v10, v10, v11
	v_accvgpr_read_b32 v11, a98
	v_accvgpr_read_b32 v12, a99
	v_cvt_pk_bf16_f32 v11, v11, v12
	v_accvgpr_read_b32 v12, a156
	v_accvgpr_read_b32 v13, a157
	v_cvt_pk_bf16_f32 v12, v12, v13
	v_accvgpr_read_b32 v13, a158
	v_accvgpr_read_b32 v14, a159
	v_cvt_pk_bf16_f32 v13, v13, v14
	v_accvgpr_read_b32 v14, a160
	v_accvgpr_read_b32 v15, a161
	v_cvt_pk_bf16_f32 v14, v14, v15
	v_accvgpr_read_b32 v15, a162
	v_accvgpr_read_b32 v16, a163
	v_cvt_pk_bf16_f32 v15, v15, v16
	v_mov_b32_dpp v16, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v17, v5 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v18, v6 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v19, v7 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v20, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v21, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v22, v10 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v23, v11 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v24, v12 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v25, v13 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v26, v14 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v27, v15 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v4, v4, v17, s[2:3]
	v_cndmask_b32_e64 v5, v16, v5, s[2:3]
	v_cndmask_b32_e64 v6, v6, v19, s[2:3]
	v_cndmask_b32_e64 v7, v18, v7, s[2:3]
	v_cndmask_b32_e64 v8, v8, v21, s[2:3]
	v_cndmask_b32_e64 v9, v20, v9, s[2:3]
	v_cndmask_b32_e64 v10, v10, v23, s[2:3]
	v_cndmask_b32_e64 v11, v22, v11, s[2:3]
	v_cndmask_b32_e64 v12, v12, v25, s[2:3]
	v_cndmask_b32_e64 v13, v24, v13, s[2:3]
	v_cndmask_b32_e64 v14, v14, v27, s[2:3]
	v_cndmask_b32_e64 v15, v26, v15, s[2:3]
	v_lshlrev_b32_e32 v16, 16, v5
	v_accvgpr_read_b32 v17, a220
	v_accvgpr_read_b32 v18, a221
	v_cvt_pk_bf16_f32 v18, v17, v18
	v_lshrrev_b32_e32 v17, 16, v4
	v_lshlrev_b32_e32 v20, 16, v7
	v_lshrrev_b32_e32 v21, 16, v6
	v_lshlrev_b32_e32 v22, 16, v9
	v_lshrrev_b32_e32 v23, 16, v8
	v_lshlrev_b32_e32 v24, 16, v11
	v_lshrrev_b32_e32 v25, 16, v10
	v_lshlrev_b32_e32 v26, 16, v13
	v_lshrrev_b32_e32 v27, 16, v12
	v_lshlrev_b32_e32 v28, 16, v15
	v_lshrrev_b32_e32 v29, 16, v14
	s_mov_b32 s0, 0xffff
	v_accvgpr_read_b32 v19, a222
	v_accvgpr_read_b32 v30, a223
	v_cvt_pk_bf16_f32 v19, v19, v30
	v_and_or_b32 v4, s0, v4, v16
	v_mov_b32_dpp v16, v18 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v30, v19 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_mov_b32 s5, 0xffff0000
	v_cndmask_b32_e64 v18, v18, v30, s[2:3]
	v_and_or_b32 v5, s5, v5, v17
	v_cndmask_b32_e64 v16, v16, v19, s[2:3]
	v_and_or_b32 v6, s0, v6, v20
	v_and_or_b32 v7, s5, v7, v21
	v_and_or_b32 v8, s0, v8, v22
	v_and_or_b32 v9, s5, v9, v23
	s_mov_b32 s0, 0xffff
	v_and_or_b32 v10, s0, v10, v24
	s_mov_b32 s5, 0xffff0000
	v_and_or_b32 v11, s5, v11, v25
	v_and_or_b32 v12, s0, v12, v26
	v_and_or_b32 v13, s5, v13, v27
	v_and_or_b32 v14, s0, v14, v28
	v_and_or_b32 v15, s5, v15, v29
	ds_write2_b32 v0, v4, v5 offset1:68
	v_accvgpr_read_b32 v4, a224
	v_accvgpr_read_b32 v5, a225
	v_cvt_pk_bf16_f32 v4, v4, v5
	v_accvgpr_read_b32 v5, a226
	v_accvgpr_read_b32 v17, a227
	v_cvt_pk_bf16_f32 v5, v5, v17
	v_mov_b32_dpp v17, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v19, v5 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshlrev_b32_e32 v20, 16, v16
	v_lshrrev_b32_e32 v21, 16, v18
	s_mov_b32 s0, 0xffff
	v_and_or_b32 v18, s0, v18, v20
	s_mov_b32 s5, 0xffff0000
	v_and_or_b32 v16, s5, v16, v21
	ds_write2_b32 v0, v6, v7 offset0:8 offset1:76
	v_cndmask_b32_e64 v4, v4, v19, s[2:3]
	v_cndmask_b32_e64 v5, v17, v5, s[2:3]
	v_lshlrev_b32_e32 v6, 16, v5
	v_lshrrev_b32_e32 v7, 16, v4
	v_and_or_b32 v4, s0, v4, v6
	v_and_or_b32 v5, s5, v5, v7
	ds_write2_b32 v0, v8, v9 offset0:16 offset1:84
	ds_write2_b32 v0, v10, v11 offset0:24 offset1:92
	ds_write2_b32 v0, v12, v13 offset0:32 offset1:100
	ds_write2_b32 v0, v14, v15 offset0:40 offset1:108
	ds_write2_b32 v0, v18, v16 offset0:48 offset1:116
	ds_write2_b32 v0, v4, v5 offset0:56 offset1:124
	s_waitcnt lgkmcnt(0)
	ds_read_b128 v[4:7], v2 offset:4096
	ds_read_b128 v[8:11], v2 offset:5184
	ds_read_b128 v[12:15], v2 offset:6272
	ds_read_b128 v[16:19], v2 offset:7360
	s_add_i32 s0, s4, 0x70
	s_lshl_b32 s0, s0, 13
	s_add_i32 s0, s1, s0
	s_waitcnt lgkmcnt(3)
	buffer_store_dwordx4 v[4:7], v1, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(2)
	buffer_store_dwordx4 v[8:11], v3, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[12:15], v34, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[16:19], v35, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(0)
	s_endpgm
.Lrocke_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s2_4096x4096x2048_c0c1_Acanonical_kc922a7f4be_end:
.size rocke_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s2_4096x4096x2048_c0c1_Acanonical_kc922a7f4be, .Lrocke_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s2_4096x4096x2048_c0c1_Acanonical_kc922a7f4be_end-rocke_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s2_4096x4096x2048_c0c1_Acanonical_kc922a7f4be

.rodata
.p2align 6
.amdhsa_kernel rocke_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s2_4096x4096x2048_c0c1_Acanonical_kc922a7f4be
  .amdhsa_group_segment_fixed_size 127520
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_kernarg_size 40
  .amdhsa_user_sgpr_count 12
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_user_sgpr_kernarg_preload_length 10
  .amdhsa_user_sgpr_kernarg_preload_offset 0
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_next_free_vgpr 468
  .amdhsa_next_free_sgpr 58
  .amdhsa_reserve_vcc 1
  .amdhsa_accum_offset 212
  .amdhsa_float_denorm_mode_32 3
  .amdhsa_float_denorm_mode_16_64 3
  .amdhsa_ieee_mode 1
  .amdhsa_dx10_clamp 1
.end_amdhsa_kernel

.amdgpu_metadata
---
custom.config:
  Source:
    Origin: rocke
    Description: MX E2M3 GEMM at 4096x4096x2048 on tiled A, canonical B, and shuffled scales
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
    DataType: F6
    DestDataType: b
    ComputeDataType: s
    HighPrecisionAccumulate: True
    TransposeA: True
    TransposeB: False
    UseBeta: False
    Batched: True
    UseBias: 0
    Activation: False
    UseScaleAlphaVec: 0
    SwizzleTensorA: True
    SwizzleTensorB: False
    MXBlockA: 32
    MXBlockB: 32
    DataTypeMXSA: E8
    DataTypeMXSB: E8
  CustomKernel:
    # The kernel's a and sa are the GEMM's B side, b and sb its A side.
    args: [ { type: address, semantic: AddressB },
            { type: address, semantic: AddressA },
            { type: address, semantic: AddressMXScaleB },
            { type: address, semantic: AddressMXScaleA },
            { type: address, semantic: AddressD } ]
    macrotile: [256, 256, 128]
    threads: [256, 1, 1]
    grid: [TilesXY, One, One]
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  EnableMatrixInstruction: True
  MIWaveTile: [8, 8]
  WavefrontSize: 64
  StaggerU: 0
  MXScaleFormat: HostPreSwizzle128x4
  AssertSizeEqual: {0: 4096, 1: 4096, 2: 1, 3: 2048}
  AssertStrideAEqual: {1: 2048}
  AssertStrideBEqual: {1: 2048}
  AssertStrideDEqual: {1: 4096}
  AssertAlphaValue: 1
amdhsa.target: amdgcn-amd-amdhsa--gfx950
amdhsa.version: [ 1, 1 ]
amdhsa.kernels:
  - .name: rocke_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s2_4096x4096x2048_c0c1_Acanonical_kc922a7f4be
    .symbol: rocke_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s2_4096x4096x2048_c0c1_Acanonical_kc922a7f4be.kd
    .args:
      - .address_space:  global
        .name:           a
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .name:           b
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .name:           sa
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .name:           sb
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .name:           c
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 127520
    .kernarg_segment_align: 8
    .kernarg_segment_size: 40
    .max_flat_workgroup_size: 256
    .private_segment_fixed_size: 0
    .sgpr_count: 64
    .vgpr_count: 468
    .agpr_count: 256
    .wavefront_size: 64
...
.end_amdgpu_metadata
