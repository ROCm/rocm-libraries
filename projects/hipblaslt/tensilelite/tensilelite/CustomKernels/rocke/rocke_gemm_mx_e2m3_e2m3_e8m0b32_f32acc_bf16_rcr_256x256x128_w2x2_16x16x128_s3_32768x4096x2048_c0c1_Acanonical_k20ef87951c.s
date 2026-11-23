// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// MX E2M3 x E2M3 GEMM for gfx950: E8M0 scales per 32 K elements, fp32 accumulation,
// bf16 D rounded to nearest even. 256 threads (wave64, 2x2 waves), macrotile 256x256x128,
// v_mfma_scale_f32_16x16x128_f8f6f4 cbsz:2 blgp:2, 154144 bytes of static LDS.
//
// Fixed problem, enforced by the predicates in custom.config:
//   M = 32768, N = 4096, K = 2048, batch 1, TransA = T, TransB = N, lda = ldb = K, D column-major
//   with ldd = 32768. alpha must be 1 (AssertAlphaValue) and beta 0 (UseBeta false): the
//   kernel takes no scalar arguments and never reads C. The grid must be exactly 2048
//   workgroups: every buffer descriptor has num_records 0x7ffffff0 and no load or store is
//   masked, so any other size reads and writes out of range. The K loop and the
//   workgroup-to-tile map are compiled for this shape. Its only arguments are the five
//   pointers in the metadata below, so it cannot take the UserArgs launch path.
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
.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.amdhsa_code_object_version 4
.text
.globl rocke_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_32768x4096x2048_c0c1_Acanonical_k20ef87951c
.p2align 8
.type rocke_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_32768x4096x2048_c0c1_Acanonical_k20ef87951c,@function
rocke_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_32768x4096x2048_c0c1_Acanonical_k20ef87951c:
	s_load_dwordx8 s[4:11], s[0:1], 0x0
	s_nop 0
	s_load_dwordx2 s[0:1], s[0:1], 0x20
	v_lshrrev_b32_e32 v0, 6, v0
	s_nop 0
	v_readfirstlane_b32 s3, v0
	v_mbcnt_lo_u32_b32 v0, -1, 0
	v_mbcnt_hi_u32_b32 v0, -1, v0
	s_lshr_b32 s12, s3, 1
	s_and_b32 s13, s3, 1
	s_lshr_b32 s14, s2, 8
	s_lshl_b32 s14, s14, 8
	s_and_b32 s15, s2, 0xff
	s_lshr_b32 s15, s15, 1
	s_add_i32 s14, s14, s15
	s_and_b32 s2, s2, 1
	s_lshl_b32 s2, s2, 7
	s_add_i32 s2, s14, s2
	s_and_b32 s14, s2, 15
	s_lshr_b32 s2, s2, 4
	s_lshr_b32 s15, s2, 4
	s_lshl_b32 s15, s15, 4
	s_sub_i32 s16, 0x80, s15
	s_min_i32 s16, s16, 16
	v_ashrrev_i32_e64 v1, 31, s16
	v_add_u32_e32 v2, s16, v1
	v_xor_b32_e32 v2, v2, v1
	v_cvt_f32_u32_e32 v3, v2
	v_rcp_iflag_f32_e32 v3, v3
	v_sub_u32_e32 v4, 0, v2
	v_mul_f32_e32 v3, 0x4f7ffffe, v3
	v_cvt_u32_f32_e32 v3, v3
	s_and_b32 s2, s2, 15
	v_mul_lo_u32 v4, v4, v3
	s_lshl_b32 s2, s2, 4
	v_mul_hi_u32 v4, v3, v4
	s_add_i32 s2, s14, s2
	v_add_u32_e32 v3, v3, v4
	v_mul_hi_u32 v3, s2, v3
	v_mul_lo_u32 v4, v3, v2
	v_sub_u32_e32 v4, s2, v4
	v_cmp_ge_u32_e64 s[18:19], v4, v2
	v_sub_u32_e32 v5, v4, v2
	v_add_u32_e32 v6, 1, v3
	v_cndmask_b32_e64 v4, v4, v5, s[18:19]
	v_cmp_ge_u32_e64 s[20:21], v4, v2
	v_cndmask_b32_e64 v2, v3, v6, s[18:19]
	v_add_u32_e32 v3, 1, v2
	v_cndmask_b32_e64 v2, v2, v3, s[20:21]
	v_xor_b32_e32 v2, v2, v1
	v_sub_u32_e32 v1, v2, v1
	s_nop 0
	v_readfirstlane_b32 s14, v1
	s_mul_i32 s17, s3, 0x600
	s_mul_i32 s18, s3, 0x810
	s_mul_i32 s16, s16, s14
	s_sub_i32 s2, s2, s16
	s_add_i32 s2, s15, s2
	; M=32768 is the A tile count and N=4096 the B tile count. The map
	; above emits those counts swapped; exchange them before the offsets.
	s_xor_b32 s2, s2, s14
	s_xor_b32 s14, s2, s14
	s_xor_b32 s2, s2, s14
	s_lshl_b32 s2, s2, 8
	s_mul_i32 s15, s2, 0x600
	s_lshl_b32 s16, s14, 8
	s_lshl_b32 s12, s12, 7
	s_lshl_b32 s13, s13, 7
	s_waitcnt lgkmcnt(0)
	s_add_u32 s4, s4, s15
	s_addc_u32 s5, s5, 0
	s_lshl_b32 s15, s12, 3
	s_lshl_b32 s19, s3, 6
	s_add_i32 s20, s17, 0x1800
	s_add_i32 s21, s18, 0x2040
	s_add_i32 s22, s17, 0x3000
	s_add_i32 s23, s18, 0x4080
	s_add_i32 s24, s17, 0x4800
	s_add_i32 s25, s18, 0x60c0
	s_add_i32 s26, s17, 0x30000
	s_add_i32 s27, s18, 0x400
	s_add_i32 s28, s3, 20
	s_add_i32 s29, s3, 4
	s_lshr_b32 s28, s28, 4
	s_lshl_b32 s28, s28, 3
	s_lshl_b32 s30, s28, 4
	s_add_i32 s30, s29, s30
	s_mul_i32 s29, s29, 0x810
	s_lshl_b32 s28, s28, 7
	s_add_i32 s28, s29, s28
	s_add_i32 s29, s3, 24
	s_add_i32 s31, s3, 8
	s_lshr_b32 s29, s29, 4
	s_lshl_b32 s29, s29, 3
	s_lshl_b32 s32, s29, 4
	s_add_i32 s32, s31, s32
	s_mul_i32 s31, s31, 0x810
	s_lshl_b32 s29, s29, 7
	v_and_b32_e32 v1, 15, v0
	v_lshrrev_b32_e32 v2, 4, v0
	v_and_b32_e32 v3, 3, v1
	v_add_u32_e32 v4, s19, v0
	v_lshrrev_b32_e32 v5, 2, v1
	v_lshlrev_b32_e32 v3, 5, v3
	v_mul_u32_u24_e32 v6, 24, v2
	v_and_b32_e32 v7, 15, v4
	v_lshrrev_b32_e32 v4, 4, v4
	v_cmp_ge_i32_e64 s[34:35], v2, 1
	v_cmp_ge_i32_e64 s[36:37], v2, 2
	v_cmp_ge_i32_e64 s[38:39], v2, 3
	s_mov_b32 s19, 0x808
	v_mad_u32_u24 v3, s19, v5, v3
	v_lshlrev_b32_e32 v4, 7, v4
	v_lshrrev_b32_e32 v5, 2, v7
	v_and_b32_e32 v8, 7, v0
	v_add3_u32 v3, s15, v6, v3
	v_mad_u32_u24 v5, s19, v5, v4
	v_and_b32_e32 v9, 3, v7
	v_lshrrev_b32_e32 v10, 3, v0
	s_add_i32 s19, s31, s29
	s_add_i32 s29, s3, 28
	v_lshlrev_b32_e32 v8, 4, v8
	s_add_i32 s31, s3, 12
	v_add_u32_e32 v11, 0x101a0, v3
	v_add_u32_e32 v3, 0x80a0, v3
	s_mov_b32 s33, 0x810
	v_mad_u32_u24 v4, s33, v7, v4
	v_lshl_add_u32 v5, v9, 5, v5
	s_mov_b32 s33, 0x6000
	s_lshr_b32 s29, s29, 4
	v_mad_u32_u24 v7, s33, v10, v8
	s_lshl_b32 s29, s29, 3
	s_lshl_b32 s33, s29, 4
	s_add_i32 s33, s31, s33
	s_mul_i32 s31, s31, 0x810
	s_lshl_b32 s29, s29, 7
	s_add_i32 s29, s31, s29
	s_mov_b32 s31, 0x810
	v_mad_u32_u24 v6, s31, v1, v6
	v_add_u32_e32 v6, s15, v6
	v_add_u32_e32 v8, 0x8080, v6
	v_add_u32_e32 v9, 0xffff7e80, v6
	v_add_u32_e32 v4, 0x60, v4
	v_add_u32_e32 v5, 0x10200, v5
	v_cndmask_b32_e64 v10, v6, v8, s[36:37]
	v_cndmask_b32_e64 v12, v6, v9, s[36:37]
	v_cndmask_b32_e64 v13, v6, v8, s[38:39]
	v_cndmask_b32_e64 v14, v6, v9, s[38:39]
	v_cndmask_b32_e64 v15, v11, v8, s[34:35]
	v_cndmask_b32_e64 v16, v3, v9, s[34:35]
	v_cndmask_b32_e64 v8, v11, v8, s[36:37]
	v_cndmask_b32_e64 v3, v3, v9, s[36:37]
	v_lshrrev_b32_e32 v4, 2, v4
	v_lshrrev_b32_e32 v5, 2, v5
	s_mul_i32 s14, s14, 0x60000
	s_add_u32 s36, s6, s14
	s_addc_u32 s37, s7, 0
	s_lshr_b32 s14, s13, 4
	s_and_b32 s37, s37, 0xffff
	s_add_i32 s15, s2, s12
	s_add_i32 s31, s16, s13
	s_and_b32 s41, s9, 0xffff
	s_and_b32 s9, s11, 0xffff
	s_mul_i32 s30, s30, 0x600
	s_lshl_b32 s34, s3, 8
	s_ashr_i32 s6, s2, 31
	s_lshr_b32 s6, s6, 25
	s_add_i32 s2, s2, s6
	s_ashr_i32 s2, s2, 7
	s_ashr_i32 s6, s16, 31
	s_lshr_b32 s6, s6, 25
	s_mul_i32 s32, s32, 0x600
	s_add_i32 s6, s16, s6
	s_ashr_i32 s16, s6, 7
	s_lshr_b32 s12, s12, 7
	s_lshr_b32 s13, s13, 7
	s_add_i32 s35, 0x13800, s18
	s_and_b32 s5, s5, 0xffff
	s_mul_i32 s33, s33, 0x600
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
	s_mul_i32 s44, s3, 0x1800
	s_mov_b32 s38, 0x7ffffff0
	s_mov_b32 s39, 0x31014000
	s_mov_b32 s40, s8
	s_mov_b32 s42, 0x7ffffff0
	s_mov_b32 s43, 0x31014000
	s_mov_b32 s8, s10
	s_mov_b32 s10, 0x7ffffff0
	s_mov_b32 s11, 0x31014000
	v_lshlrev_b32_e32 v16, 2, v0
	s_mov_b32 s6, 0x7ffffff0
	s_mov_b32 m0, s35
	s_mov_b32 s7, 0x31014000
	buffer_load_dwordx4 v7, s[4:7], s17 offen lds
	s_add_i32 s45, 0x13800, s21
	s_mov_b32 m0, s45
	s_nop 0
	buffer_load_dwordx4 v7, s[4:7], s20 offen lds
	s_add_i32 s46, 0x13800, s23
	s_mov_b32 m0, s46
	s_nop 0
	buffer_load_dwordx4 v7, s[4:7], s22 offen lds
	s_add_i32 s47, 0x13800, s25
	s_mov_b32 m0, s47
	s_nop 0
	buffer_load_dwordx4 v7, s[4:7], s24 offen lds
	s_add_i32 s48, 0x13800, s27
	s_mov_b32 m0, s48
	s_nop 0
	buffer_load_dwordx4 v7, s[4:7], s26 offen lds
	s_add_i32 s49, 0x13800, s28
	s_mov_b32 m0, s49
	s_nop 0
	buffer_load_dwordx4 v7, s[4:7], s30 offen lds
	s_add_i32 s50, 0x13800, s19
	s_mov_b32 m0, s50
	s_nop 0
	buffer_load_dwordx4 v7, s[4:7], s32 offen lds
	s_add_i32 s51, 0x13800, s29
	s_mov_b32 m0, s51
	s_nop 0
	buffer_load_dwordx4 v7, s[4:7], s33 offen lds
	s_lshl_b32 s52, s2, 9
	s_lshl_b32 s53, s16, 9
	s_add_i32 s54, 0x1800, s44
	s_mov_b32 m0, s54
	s_nop 0
	buffer_load_dwordx4 v15, s[36:39], s44 offen lds
	buffer_load_dwordx4 v15, s[36:39], s44 offen offset:1024 lds
	buffer_load_dwordx4 v15, s[36:39], s44 offen offset:2048 lds
	buffer_load_dwordx4 v15, s[36:39], s44 offen offset:3072 lds
	s_add_i32 s55, s44, 0x1000
	s_add_i32 s56, 0x1800, s55
	s_mov_b32 m0, s56
	s_nop 0
	buffer_load_dwordx4 v15, s[36:39], s55 offen lds
	buffer_load_dwordx4 v15, s[36:39], s55 offen offset:1024 lds
	s_mov_b32 m0, s34
	s_add_i32 s52, s34, s52
	buffer_load_dword v16, s[40:43], s52 offen lds
	s_add_i32 s52, 0x400, s34
	s_mov_b32 m0, s52
	s_add_i32 s53, s34, s53
	buffer_load_dword v16, s[8:11], s53 offen lds
	s_add_i32 s53, s17, 0x80
	s_add_i32 s18, s18, 0x8100
	s_add_i32 s18, 0x13800, s18
	s_mov_b32 m0, s18
	s_nop 0
	buffer_load_dwordx4 v7, s[4:7], s53 offen lds
	s_add_i32 s53, s20, 0x80
	s_add_i32 s21, s21, 0x8100
	s_add_i32 s21, 0x13800, s21
	s_mov_b32 m0, s21
	s_nop 0
	buffer_load_dwordx4 v7, s[4:7], s53 offen lds
	s_add_i32 s53, s22, 0x80
	s_add_i32 s23, s23, 0x8100
	s_add_i32 s23, 0x13800, s23
	s_mov_b32 m0, s23
	s_nop 0
	buffer_load_dwordx4 v7, s[4:7], s53 offen lds
	s_add_i32 s53, s24, 0x80
	s_add_i32 s25, s25, 0x8100
	s_add_i32 s25, 0x13800, s25
	s_mov_b32 m0, s25
	s_nop 0
	buffer_load_dwordx4 v7, s[4:7], s53 offen lds
	s_add_i32 s53, s26, 0x80
	s_add_i32 s27, s27, 0x8100
	s_add_i32 s27, 0x13800, s27
	s_mov_b32 m0, s27
	s_nop 0
	buffer_load_dwordx4 v7, s[4:7], s53 offen lds
	s_add_i32 s53, s30, 0x80
	s_add_i32 s28, s28, 0x8100
	s_add_i32 s28, 0x13800, s28
	s_mov_b32 m0, s28
	s_nop 0
	buffer_load_dwordx4 v7, s[4:7], s53 offen lds
	s_add_i32 s53, s32, 0x80
	s_add_i32 s19, s19, 0x8100
	s_add_i32 s19, 0x13800, s19
	s_mov_b32 m0, s19
	s_nop 0
	buffer_load_dwordx4 v7, s[4:7], s53 offen lds
	s_add_i32 s53, s33, 0x80
	s_add_i32 s29, s29, 0x8100
	s_add_i32 s29, 0x13800, s29
	s_mov_b32 m0, s29
	s_nop 0
	buffer_load_dwordx4 v7, s[4:7], s53 offen lds
	s_add_i32 s53, s2, 0x20
	s_lshl_b32 s53, s53, 9
	s_add_i32 s57, s16, 0x100
	s_lshl_b32 s57, s57, 9
	s_add_i32 s58, s44, 0x6000
	s_add_i32 s59, 0x7800, s44
	s_mov_b32 m0, s59
	s_nop 0
	buffer_load_dwordx4 v15, s[36:39], s58 offen lds
	buffer_load_dwordx4 v15, s[36:39], s58 offen offset:1024 lds
	buffer_load_dwordx4 v15, s[36:39], s58 offen offset:2048 lds
	buffer_load_dwordx4 v15, s[36:39], s58 offen offset:3072 lds
	s_add_i32 s58, 0x7800, s55
	s_mov_b32 m0, s58
	s_add_i32 s60, s44, 0x7000
	buffer_load_dwordx4 v15, s[36:39], s60 offen lds
	buffer_load_dwordx4 v15, s[36:39], s60 offen offset:1024 lds
	s_add_i32 s60, 0x800, s34
	s_mov_b32 m0, s60
	s_add_i32 s53, s34, s53
	buffer_load_dword v16, s[40:43], s53 offen lds
	s_add_i32 s53, 0xc00, s34
	s_mov_b32 m0, s53
	s_add_i32 s57, s34, s57
	buffer_load_dword v16, s[8:11], s57 offen lds
	s_add_i32 s57, s2, 0x40
	s_lshl_b32 s57, s57, 9
	s_add_i32 s61, s16, 0x200
	s_lshl_b32 s61, s61, 9
	s_add_i32 s62, s44, 0xc000
	s_add_i32 s63, 0xd800, s44
	s_mov_b32 m0, s63
	s_nop 0
	buffer_load_dwordx4 v15, s[36:39], s62 offen lds
	buffer_load_dwordx4 v15, s[36:39], s62 offen offset:1024 lds
	buffer_load_dwordx4 v15, s[36:39], s62 offen offset:2048 lds
	buffer_load_dwordx4 v15, s[36:39], s62 offen offset:3072 lds
	s_add_i32 s55, 0xd800, s55
	s_mov_b32 m0, s55
	s_add_i32 s62, s44, 0xd000
	buffer_load_dwordx4 v15, s[36:39], s62 offen lds
	buffer_load_dwordx4 v15, s[36:39], s62 offen offset:1024 lds
	s_add_i32 s62, 0x1000, s34
	s_mov_b32 m0, s62
	s_add_i32 s57, s34, s57
	buffer_load_dword v16, s[40:43], s57 offen lds
	s_add_i32 s57, 0x1400, s34
	s_mov_b32 m0, s57
	s_add_i32 s61, s34, s61
	buffer_load_dword v16, s[8:11], s61 offen lds
	s_waitcnt vmcnt(24) lgkmcnt(0)
	s_barrier
	s_lshl_b32 s12, s12, 7
	v_lshlrev_b32_e32 v17, 1, v0
	s_lshl_b32 s13, s13, 7
	v_add_lshl_u32 v18, s12, v17, 2
	v_add_lshl_u32 v17, s13, v17, 2
	s_mov_b32 s12, 0x10000
	v_lshl_add_u32 v6, v6, 2, s12
	ds_read_b64 v[20:21], v18
	ds_read_b64 v[22:23], v17 offset:1024
	ds_read_b64 v[24:25], v6 offset:14336
	ds_read_b64 v[26:27], v6 offset:14344
	ds_read_b64 v[28:29], v6 offset:14352
	s_lshl_b32 s12, s14, 6
	v_add_u32_e32 v0, s12, v0
	v_lshlrev_b32_e32 v19, 4, v0
	ds_read_b128 v[30:33], v19 offset:6144
	v_lshlrev_b32_e32 v0, 3, v0
	ds_read_b64 v[34:35], v0 offset:22528
	ds_read_b64 v[36:37], v6 offset:14464
	ds_read_b64 v[38:39], v6 offset:14472
	ds_read_b64 v[40:41], v6 offset:14480
	ds_read_b64 v[42:43], v6 offset:14592
	ds_read_b64 v[44:45], v6 offset:14600
	ds_read_b64 v[46:47], v6 offset:14608
	ds_read_b64 v[48:49], v6 offset:14720
	ds_read_b64 v[50:51], v6 offset:14728
	ds_read_b64 v[52:53], v6 offset:14736
	ds_read_b64 v[54:55], v6 offset:14848
	ds_read_b64 v[56:57], v6 offset:14856
	ds_read_b64 v[58:59], v6 offset:14864
	ds_read_b64 v[60:61], v6 offset:14976
	ds_read_b64 v[62:63], v6 offset:14984
	ds_read_b64 v[64:65], v6 offset:14992
	ds_read_b64 v[66:67], v6 offset:15104
	ds_read_b64 v[68:69], v6 offset:15112
	ds_read_b64 v[70:71], v6 offset:15120
	ds_read_b64 v[72:73], v6 offset:15232
	ds_read_b64 v[74:75], v6 offset:15240
	ds_read_b64 v[76:77], v6 offset:15248
	ds_read_b128 v[78:81], v19 offset:7168
	ds_read_b64 v[82:83], v0 offset:23040
	ds_read_b128 v[84:87], v19 offset:8192
	ds_read_b64 v[88:89], v0 offset:23552
	ds_read_b128 v[90:93], v19 offset:9216
	ds_read_b64 v[94:95], v0 offset:24064
	ds_read_b128 v[96:99], v19 offset:10240
	ds_read_b64 v[100:101], v0 offset:24576
	ds_read_b128 v[102:105], v19 offset:11264
	ds_read_b64 v[106:107], v0 offset:25088
	ds_read_b128 v[108:111], v19 offset:12288
	ds_read_b64 v[112:113], v0 offset:25600
	ds_read_b128 v[114:117], v19 offset:13312
	s_mov_b32 s12, 0x10000
	v_lshl_add_u32 v4, v4, 2, s12
	ds_read_b64 v[118:119], v0 offset:26112
	ds_read_b128 v[120:123], v4 offset:14336
	ds_read_b128 v[124:127], v4 offset:14352
	v_lshl_add_u32 v5, v5, 2, s12
	s_waitcnt lgkmcnt(1)
	ds_write_b64 v5, v[120:121] offset:14336
	ds_write_b64 v5, v[122:123] offset:14344
	s_waitcnt lgkmcnt(2)
	ds_write_b64 v5, v[124:125] offset:14352
	ds_write_b64 v5, v[126:127] offset:14360
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_mov_b32 m0, s35
	s_add_i32 s12, s17, 0x100
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	s_mov_b32 m0, s45
	s_add_i32 s12, s20, 0x100
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	s_mov_b32 m0, s46
	s_add_i32 s12, s22, 0x100
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	s_mov_b32 m0, s47
	s_add_i32 s12, s24, 0x100
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	s_mov_b32 m0, s48
	s_add_i32 s12, s26, 0x100
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	s_mov_b32 m0, s49
	s_add_i32 s12, s30, 0x100
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	s_mov_b32 m0, s50
	s_add_i32 s12, s32, 0x100
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	s_mov_b32 m0, s51
	s_add_i32 s12, s33, 0x100
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	s_add_i32 s12, s2, 0x60
	s_lshl_b32 s12, s12, 9
	s_add_i32 s13, s16, 0x300
	s_lshl_b32 s13, s13, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[24:29], v[30:35], 0, v20, v22 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(16) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[120:121], v18 offset:2048
	ds_read_b64 v[122:123], v17 offset:3072
	s_mov_b32 s14, 0x10000
	v_lshl_add_u32 v8, v8, 2, s14
	ds_read_b64 v[124:125], v8 offset:14432
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[36:41], v[30:35], 0, v20, v22 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_lshl_add_u32 v13, v13, 2, s14
	ds_read_b64 v[126:127], v13 offset:14440
	ds_read_b64 v[128:129], v13 offset:14448
	s_add_i32 s14, s44, 0x12000
	s_mov_b32 m0, s54
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[42:47], v[30:35], 0, v20, v22 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen lds
	ds_read_b64 v[130:131], v8 offset:14560
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[48:53], v[30:35], 0, v20, v22 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[132:133], v13 offset:14568
	ds_read_b64 v[134:135], v13 offset:14576
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[54:59], v[30:35], 0, v21, v22 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[136:137], v8 offset:14688
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[60:65], v[30:35], 0, v21, v22 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[138:139], v13 offset:14696
	ds_read_b64 v[140:141], v13 offset:14704
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[66:71], v[30:35], 0, v21, v22 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen offset:1024 lds
	ds_read_b64 v[142:143], v8 offset:14816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[72:77], v[30:35], 0, v21, v22 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[144:145], v13 offset:14824
	ds_read_b64 v[146:147], v13 offset:14832
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[72:77], v[78:83], 0, v21, v22 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[30:31], v8 offset:14944
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[66:71], v[78:83], 0, v21, v22 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[32:33], v13 offset:14952
	ds_read_b64 v[34:35], v13 offset:14960
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[60:65], v[78:83], 0, v21, v22 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen offset:2048 lds
	ds_read_b64 v[148:149], v8 offset:15072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[54:59], v[78:83], 0, v21, v22 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[150:151], v13 offset:15080
	ds_read_b64 v[152:153], v13 offset:15088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[48:53], v[78:83], 0, v20, v22 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[154:155], v8 offset:15200
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[42:47], v[78:83], 0, v20, v22 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[156:157], v13 offset:15208
	ds_read_b64 v[158:159], v13 offset:15216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[36:41], v[78:83], 0, v20, v22 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen offset:3072 lds
	ds_read_b64 v[160:161], v8 offset:15328
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[24:29], v[78:83], 0, v20, v22 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[162:163], v13 offset:15336
	ds_read_b64 v[164:165], v13 offset:15344
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[24:29], v[84:89], 0, v20, v22 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[78:81], v19 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[36:41], v[84:89], 0, v20, v22 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[82:83], v0 offset:47104
	s_add_i32 s14, s44, 0x13000
	s_mov_b32 m0, s56
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[42:47], v[84:89], 0, v20, v22 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen lds
	ds_read_b128 v[166:169], v19 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[48:53], v[84:89], 0, v20, v22 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[170:171], v0 offset:47616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[54:59], v[84:89], 0, v21, v22 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[172:175], v19 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[60:65], v[84:89], 0, v21, v22 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[176:177], v0 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[66:71], v[84:89], 0, v21, v22 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen offset:1024 lds
	ds_read_b128 v[178:181], v19 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[72:77], v[84:89], 0, v21, v22 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[182:183], v0 offset:48640
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[72:77], v[90:95], 0, v21, v22 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[84:87], v19 offset:34816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[66:71], v[90:95], 0, v21, v22 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[88:89], v0 offset:49152
	s_add_i32 s12, s34, s12
	s_mov_b32 m0, s34
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[60:65], v[90:95], 0, v21, v22 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v16, s[40:43], s12 offen lds
	ds_read_b128 v[184:187], v19 offset:35840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[54:59], v[90:95], 0, v21, v22 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[188:189], v0 offset:49664
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[48:53], v[90:95], 0, v20, v22 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[190:193], v19 offset:36864
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[42:47], v[90:95], 0, v20, v22 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[194:195], v0 offset:50176
	s_add_i32 s12, s34, s13
	s_mov_b32 m0, s52
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[36:41], v[90:95], 0, v20, v22 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v16, s[8:11], s12 offen lds
	ds_read_b128 v[196:199], v19 offset:37888
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[24:29], v[90:95], 0, v20, v22 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[200:201], v0 offset:50688
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[24:29], v[96:101], 0, v20, v23 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[36:41], v[96:101], 0, v20, v23 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
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
	s_add_i32 s12, s2, 0x80
	s_lshl_b32 s12, s12, 9
	s_add_i32 s13, s16, 0x400
	s_lshl_b32 s13, s13, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[124:129], v[78:83], a[0:3], v120, v122 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(8) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[20:21], v18 offset:4096
	ds_read_b64 v[22:23], v17 offset:5120
	s_mov_b32 s14, 0x10000
	v_lshl_add_u32 v12, v12, 2, s14
	ds_read_b64 v[24:25], v12 offset:47424
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[130:135], v[78:83], a[4:7], v120, v122 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[26:27], v12 offset:47432
	v_lshl_add_u32 v10, v10, 2, s14
	ds_read_b64 v[28:29], v10 offset:47440
	s_add_i32 s14, s44, 0x18000
	s_mov_b32 m0, s59
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[136:141], v[78:83], a[8:11], v120, v122 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen lds
	ds_read_b64 v[36:37], v12 offset:47552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[142:147], v[78:83], a[12:15], v120, v122 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[38:39], v12 offset:47560
	ds_read_b64 v[40:41], v10 offset:47568
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[30:35], v[78:83], a[16:19], v121, v122 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[42:43], v12 offset:47680
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[148:153], v[78:83], a[20:23], v121, v122 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[44:45], v12 offset:47688
	ds_read_b64 v[46:47], v10 offset:47696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[154:159], v[78:83], a[24:27], v121, v122 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen offset:1024 lds
	ds_read_b64 v[48:49], v12 offset:47808
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[160:165], v[78:83], a[28:31], v121, v122 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[50:51], v12 offset:47816
	ds_read_b64 v[52:53], v10 offset:47824
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[160:165], v[166:171], a[32:35], v121, v122 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[54:55], v12 offset:47936
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[154:159], v[166:171], a[36:39], v121, v122 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[56:57], v12 offset:47944
	ds_read_b64 v[58:59], v10 offset:47952
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[148:153], v[166:171], a[40:43], v121, v122 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen offset:2048 lds
	ds_read_b64 v[60:61], v12 offset:48064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[30:35], v[166:171], a[44:47], v121, v122 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[62:63], v12 offset:48072
	ds_read_b64 v[64:65], v10 offset:48080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[142:147], v[166:171], a[48:51], v120, v122 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[66:67], v12 offset:48192
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[136:141], v[166:171], a[52:55], v120, v122 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[68:69], v12 offset:48200
	ds_read_b64 v[70:71], v10 offset:48208
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[130:135], v[166:171], a[56:59], v120, v122 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen offset:3072 lds
	ds_read_b64 v[72:73], v12 offset:48320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[124:129], v[166:171], a[60:63], v120, v122 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[74:75], v12 offset:48328
	ds_read_b64 v[76:77], v10 offset:48336
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[124:129], v[172:177], a[64:67], v120, v122 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[78:81], v19 offset:55296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[130:135], v[172:177], a[68:71], v120, v122 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_add_u32_e32 v90, 0x10000, v0
	ds_read_b64 v[82:83], v90 offset:6144
	s_add_i32 s14, s44, 0x19000
	s_mov_b32 m0, s58
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[136:141], v[172:177], a[72:75], v120, v122 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen lds
	ds_read_b128 v[92:95], v19 offset:56320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[142:147], v[172:177], a[76:79], v120, v122 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[96:97], v90 offset:6656
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[30:35], v[172:177], a[80:83], v121, v122 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[98:101], v19 offset:57344
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[148:153], v[172:177], a[84:87], v121, v122 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[102:103], v90 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[154:159], v[172:177], a[88:91], v121, v122 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen offset:1024 lds
	ds_read_b128 v[104:107], v19 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[160:165], v[172:177], a[92:95], v121, v122 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[108:109], v90 offset:7680
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[160:165], v[178:183], a[96:99], v121, v122 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[110:113], v19 offset:59392
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[154:159], v[178:183], a[100:103], v121, v122 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[114:115], v90 offset:8192
	s_add_i32 s12, s34, s12
	s_mov_b32 m0, s60
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[148:153], v[178:183], a[104:107], v121, v122 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v16, s[40:43], s12 offen lds
	ds_read_b128 v[166:169], v19 offset:60416
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[30:35], v[178:183], a[108:111], v121, v122 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[170:171], v90 offset:8704
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[142:147], v[178:183], a[112:115], v120, v122 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[172:175], v19 offset:61440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[136:141], v[178:183], a[116:119], v120, v122 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[176:177], v90 offset:9216
	s_add_i32 s12, s34, s13
	s_mov_b32 m0, s53
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[130:135], v[178:183], a[120:123], v120, v122 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v16, s[8:11], s12 offen lds
	ds_read_b128 v[202:205], v19 offset:62464
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[124:129], v[178:183], a[124:127], v120, v122 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[206:207], v90 offset:9728
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[124:129], v[84:89], a[128:131], v120, v123 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[130:135], v[84:89], a[132:135], v120, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[136:141], v[84:89], a[136:139], v120, v123 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[142:147], v[84:89], a[140:143], v120, v123 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s12, s17, 0x180
	s_mov_b32 m0, s18
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[30:35], v[84:89], a[144:147], v121, v123 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[148:153], v[84:89], a[148:151], v121, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[154:159], v[84:89], a[152:155], v121, v123 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s12, s20, 0x180
	s_mov_b32 m0, s21
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[160:165], v[84:89], a[156:159], v121, v123 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[160:165], v[184:189], a[160:163], v121, v123 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[154:159], v[184:189], a[164:167], v121, v123 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[148:153], v[184:189], a[168:171], v121, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_i32 s12, s22, 0x180
	s_mov_b32 m0, s23
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[30:35], v[184:189], a[172:175], v121, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[142:147], v[184:189], a[176:179], v120, v123 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[136:141], v[184:189], a[180:183], v120, v123 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s12, s24, 0x180
	s_mov_b32 m0, s25
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[130:135], v[184:189], a[184:187], v120, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[124:129], v[184:189], a[188:191], v120, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[124:129], v[190:195], a[192:195], v120, v123 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[130:135], v[190:195], a[196:199], v120, v123 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s12, s26, 0x180
	s_mov_b32 m0, s27
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[136:141], v[190:195], a[200:203], v120, v123 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[142:147], v[190:195], a[204:207], v120, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[30:35], v[190:195], a[208:211], v121, v123 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s12, s30, 0x180
	s_mov_b32 m0, s28
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[148:153], v[190:195], a[212:215], v121, v123 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[154:159], v[190:195], a[216:219], v121, v123 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[160:165], v[190:195], a[220:223], v121, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[160:165], v[196:201], a[224:227], v121, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s12, s32, 0x180
	s_mov_b32 m0, s19
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[154:159], v[196:201], a[228:231], v121, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[148:153], v[196:201], a[232:235], v121, v123 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[30:35], v[196:201], a[236:239], v121, v123 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s12, s33, 0x180
	s_mov_b32 m0, s29
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[142:147], v[196:201], a[240:243], v120, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[136:141], v[196:201], a[244:247], v120, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[130:135], v[196:201], a[248:251], v120, v123 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[124:129], v[196:201], a[252:255], v120, v123 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s12, s2, 0xa0
	s_lshl_b32 s12, s12, 9
	s_add_i32 s13, s16, 0x500
	s_lshl_b32 s13, s13, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[24:29], v[78:83], a[0:3], v20, v22 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(16) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[30:31], v18
	ds_read_b64 v[32:33], v17 offset:1024
	ds_read_b64 v[84:85], v6 offset:14368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[36:41], v[78:83], a[4:7], v20, v22 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[86:87], v6 offset:14376
	ds_read_b64 v[88:89], v6 offset:14384
	s_add_i32 s14, s44, 0x1e000
	s_mov_b32 m0, s63
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[42:47], v[78:83], a[8:11], v20, v22 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen lds
	ds_read_b64 v[116:117], v6 offset:14496
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[48:53], v[78:83], a[12:15], v20, v22 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[118:119], v6 offset:14504
	ds_read_b64 v[120:121], v6 offset:14512
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[54:59], v[78:83], a[16:19], v21, v22 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[122:123], v6 offset:14624
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[60:65], v[78:83], a[20:23], v21, v22 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[124:125], v6 offset:14632
	ds_read_b64 v[126:127], v6 offset:14640
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[66:71], v[78:83], a[24:27], v21, v22 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen offset:1024 lds
	ds_read_b64 v[128:129], v6 offset:14752
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[72:77], v[78:83], a[28:31], v21, v22 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[130:131], v6 offset:14760
	ds_read_b64 v[132:133], v6 offset:14768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[72:77], v[92:97], a[32:35], v21, v22 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v6 offset:14880
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[66:71], v[92:97], a[36:39], v21, v22 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[80:81], v6 offset:14888
	ds_read_b64 v[82:83], v6 offset:14896
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[60:65], v[92:97], a[40:43], v21, v22 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen offset:2048 lds
	ds_read_b64 v[134:135], v6 offset:15008
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[54:59], v[92:97], a[44:47], v21, v22 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[136:137], v6 offset:15016
	ds_read_b64 v[138:139], v6 offset:15024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[48:53], v[92:97], a[48:51], v20, v22 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[140:141], v6 offset:15136
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[42:47], v[92:97], a[52:55], v20, v22 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[142:143], v6 offset:15144
	ds_read_b64 v[144:145], v6 offset:15152
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[36:41], v[92:97], a[56:59], v20, v22 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen offset:3072 lds
	ds_read_b64 v[146:147], v6 offset:15264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[24:29], v[92:97], a[60:63], v20, v22 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[148:149], v6 offset:15272
	ds_read_b64 v[150:151], v6 offset:15280
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[24:29], v[98:103], a[64:67], v20, v22 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[92:95], v19 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[36:41], v[98:103], a[68:71], v20, v22 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[96:97], v0 offset:22528
	s_add_i32 s14, s44, 0x1f000
	s_mov_b32 m0, s55
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[42:47], v[98:103], a[72:75], v20, v22 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen lds
	ds_read_b128 v[152:155], v19 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[48:53], v[98:103], a[76:79], v20, v22 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[156:157], v0 offset:23040
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[54:59], v[98:103], a[80:83], v21, v22 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[158:161], v19 offset:8192
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[60:65], v[98:103], a[84:87], v21, v22 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[162:163], v0 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[66:71], v[98:103], a[88:91], v21, v22 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen offset:1024 lds
	ds_read_b128 v[178:181], v19 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[72:77], v[98:103], a[92:95], v21, v22 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[182:183], v0 offset:24064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[72:77], v[104:109], a[96:99], v21, v22 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[98:101], v19 offset:10240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[66:71], v[104:109], a[100:103], v21, v22 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[102:103], v0 offset:24576
	s_add_i32 s12, s34, s12
	s_mov_b32 m0, s62
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[60:65], v[104:109], a[104:107], v21, v22 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v16, s[40:43], s12 offen lds
	ds_read_b128 v[184:187], v19 offset:11264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[54:59], v[104:109], a[108:111], v21, v22 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[188:189], v0 offset:25088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[48:53], v[104:109], a[112:115], v20, v22 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[190:193], v19 offset:12288
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[42:47], v[104:109], a[116:119], v20, v22 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[194:195], v0 offset:25600
	s_add_i32 s12, s34, s13
	s_mov_b32 m0, s57
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[36:41], v[104:109], a[120:123], v20, v22 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v16, s[8:11], s12 offen lds
	ds_read_b128 v[196:199], v19 offset:13312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[24:29], v[104:109], a[124:127], v20, v22 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[200:201], v0 offset:26112
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[24:29], v[110:115], a[128:131], v20, v23 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[36:41], v[110:115], a[132:135], v20, v23 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[42:47], v[110:115], a[136:139], v20, v23 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[48:53], v[110:115], a[140:143], v20, v23 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s12, s17, 0x200
	s_mov_b32 m0, s35
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[54:59], v[110:115], a[144:147], v21, v23 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[60:65], v[110:115], a[148:151], v21, v23 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[66:71], v[110:115], a[152:155], v21, v23 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s12, s20, 0x200
	s_mov_b32 m0, s45
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[72:77], v[110:115], a[156:159], v21, v23 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[72:77], v[166:171], a[160:163], v21, v23 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[66:71], v[166:171], a[164:167], v21, v23 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[60:65], v[166:171], a[168:171], v21, v23 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_i32 s12, s22, 0x200
	s_mov_b32 m0, s46
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[54:59], v[166:171], a[172:175], v21, v23 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[48:53], v[166:171], a[176:179], v20, v23 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[42:47], v[166:171], a[180:183], v20, v23 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s12, s24, 0x200
	s_mov_b32 m0, s47
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[36:41], v[166:171], a[184:187], v20, v23 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[24:29], v[166:171], a[188:191], v20, v23 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[24:29], v[172:177], a[192:195], v20, v23 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[36:41], v[172:177], a[196:199], v20, v23 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s12, s26, 0x200
	s_mov_b32 m0, s48
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[42:47], v[172:177], a[200:203], v20, v23 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[48:53], v[172:177], a[204:207], v20, v23 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[54:59], v[172:177], a[208:211], v21, v23 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s12, s30, 0x200
	s_mov_b32 m0, s49
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[60:65], v[172:177], a[212:215], v21, v23 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[66:71], v[172:177], a[216:219], v21, v23 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[72:77], v[172:177], a[220:223], v21, v23 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[72:77], v[202:207], a[224:227], v21, v23 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s12, s32, 0x200
	s_mov_b32 m0, s50
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[66:71], v[202:207], a[228:231], v21, v23 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[60:65], v[202:207], a[232:235], v21, v23 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[54:59], v[202:207], a[236:239], v21, v23 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s12, s33, 0x200
	s_mov_b32 m0, s51
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[48:53], v[202:207], a[240:243], v20, v23 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[42:47], v[202:207], a[244:247], v20, v23 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[36:41], v[202:207], a[248:251], v20, v23 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[24:29], v[202:207], a[252:255], v20, v23 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s12, s2, 0xc0
	s_lshl_b32 s12, s12, 9
	s_add_i32 s13, s16, 0x600
	s_lshl_b32 s13, s13, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[84:89], v[92:97], a[0:3], v30, v32 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(16) lgkmcnt(0)
	s_barrier
	ds_read_b128 v[20:23], v4 offset:47360
	ds_read_b128 v[24:27], v4 offset:47376
	ds_read_b64 v[28:29], v18 offset:2048
	ds_read_b64 v[34:35], v17 offset:3072
	ds_read_b64 v[36:37], v6 offset:47360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[116:121], v[92:97], a[4:7], v30, v32 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[38:39], v6 offset:47368
	ds_read_b64 v[40:41], v6 offset:47376
	s_add_i32 s14, s44, 0x24000
	s_mov_b32 m0, s54
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[122:127], v[92:97], a[8:11], v30, v32 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen lds
	ds_read_b64 v[42:43], v6 offset:47488
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[128:133], v[92:97], a[12:15], v30, v32 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[44:45], v6 offset:47496
	ds_read_b64 v[46:47], v6 offset:47504
	s_waitcnt lgkmcnt(9)
	ds_write_b64 v5, v[20:21] offset:14336
	ds_write_b64 v5, v[22:23] offset:14344
	s_waitcnt lgkmcnt(10)
	ds_write_b64 v5, v[24:25] offset:14352
	ds_write_b64 v5, v[26:27] offset:14360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[78:83], v[92:97], a[16:19], v31, v32 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[20:21], v6 offset:47616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[134:139], v[92:97], a[20:23], v31, v32 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[22:23], v6 offset:47624
	ds_read_b64 v[24:25], v6 offset:47632
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[140:145], v[92:97], a[24:27], v31, v32 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen offset:1024 lds
	ds_read_b64 v[48:49], v6 offset:47744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[146:151], v[92:97], a[28:31], v31, v32 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[50:51], v6 offset:47752
	ds_read_b64 v[52:53], v6 offset:47760
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[146:151], v[152:157], a[32:35], v31, v32 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[54:55], v6 offset:47872
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[140:145], v[152:157], a[36:39], v31, v32 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[56:57], v6 offset:47880
	ds_read_b64 v[58:59], v6 offset:47888
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[134:139], v[152:157], a[40:43], v31, v32 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen offset:2048 lds
	ds_read_b64 v[60:61], v6 offset:48000
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[78:83], v[152:157], a[44:47], v31, v32 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[62:63], v6 offset:48008
	ds_read_b64 v[64:65], v6 offset:48016
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[128:133], v[152:157], a[48:51], v30, v32 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[66:67], v6 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[122:127], v[152:157], a[52:55], v30, v32 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[68:69], v6 offset:48136
	ds_read_b64 v[70:71], v6 offset:48144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[116:121], v[152:157], a[56:59], v30, v32 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen offset:3072 lds
	ds_read_b64 v[72:73], v6 offset:48256
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[84:89], v[152:157], a[60:63], v30, v32 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[74:75], v6 offset:48264
	ds_read_b64 v[76:77], v6 offset:48272
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[84:89], v[158:163], a[64:67], v30, v32 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[92:95], v19 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[116:121], v[158:163], a[68:71], v30, v32 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[96:97], v0 offset:47104
	s_add_i32 s14, s44, 0x25000
	s_mov_b32 m0, s56
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[122:127], v[158:163], a[72:75], v30, v32 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen lds
	ds_read_b128 v[104:107], v19 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[128:133], v[158:163], a[76:79], v30, v32 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[108:109], v0 offset:47616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[78:83], v[158:163], a[80:83], v31, v32 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[110:113], v19 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[134:139], v[158:163], a[84:87], v31, v32 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[114:115], v0 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[140:145], v[158:163], a[88:91], v31, v32 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen offset:1024 lds
	ds_read_b128 v[152:155], v19 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[146:151], v[158:163], a[92:95], v31, v32 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[156:157], v0 offset:48640
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[146:151], v[178:183], a[96:99], v31, v32 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[158:161], v19 offset:34816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[140:145], v[178:183], a[100:103], v31, v32 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[162:163], v0 offset:49152
	s_add_i32 s12, s34, s12
	s_mov_b32 m0, s34
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[134:139], v[178:183], a[104:107], v31, v32 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v16, s[40:43], s12 offen lds
	ds_read_b128 v[164:167], v19 offset:35840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[78:83], v[178:183], a[108:111], v31, v32 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[168:169], v0 offset:49664
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[128:133], v[178:183], a[112:115], v30, v32 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[170:173], v19 offset:36864
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[122:127], v[178:183], a[116:119], v30, v32 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[174:175], v0 offset:50176
	s_add_i32 s12, s34, s13
	s_mov_b32 m0, s52
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[116:121], v[178:183], a[120:123], v30, v32 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v16, s[8:11], s12 offen lds
	ds_read_b128 v[202:205], v19 offset:37888
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[84:89], v[178:183], a[124:127], v30, v32 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[206:207], v0 offset:50688
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[84:89], v[98:103], a[128:131], v30, v33 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[116:121], v[98:103], a[132:135], v30, v33 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[122:127], v[98:103], a[136:139], v30, v33 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[128:133], v[98:103], a[140:143], v30, v33 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s12, s17, 0x280
	s_mov_b32 m0, s18
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[78:83], v[98:103], a[144:147], v31, v33 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[134:139], v[98:103], a[148:151], v31, v33 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[140:145], v[98:103], a[152:155], v31, v33 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s12, s20, 0x280
	s_mov_b32 m0, s21
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[146:151], v[98:103], a[156:159], v31, v33 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[146:151], v[184:189], a[160:163], v31, v33 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[140:145], v[184:189], a[164:167], v31, v33 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[134:139], v[184:189], a[168:171], v31, v33 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_i32 s12, s22, 0x280
	s_mov_b32 m0, s23
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[78:83], v[184:189], a[172:175], v31, v33 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[128:133], v[184:189], a[176:179], v30, v33 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[122:127], v[184:189], a[180:183], v30, v33 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s12, s24, 0x280
	s_mov_b32 m0, s25
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[116:121], v[184:189], a[184:187], v30, v33 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[84:89], v[184:189], a[188:191], v30, v33 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[84:89], v[190:195], a[192:195], v30, v33 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[116:121], v[190:195], a[196:199], v30, v33 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s12, s26, 0x280
	s_mov_b32 m0, s27
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[122:127], v[190:195], a[200:203], v30, v33 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[128:133], v[190:195], a[204:207], v30, v33 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[78:83], v[190:195], a[208:211], v31, v33 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s12, s30, 0x280
	s_mov_b32 m0, s28
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[134:139], v[190:195], a[212:215], v31, v33 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[140:145], v[190:195], a[216:219], v31, v33 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[146:151], v[190:195], a[220:223], v31, v33 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[146:151], v[196:201], a[224:227], v31, v33 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s12, s32, 0x280
	s_mov_b32 m0, s19
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[140:145], v[196:201], a[228:231], v31, v33 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[134:139], v[196:201], a[232:235], v31, v33 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[78:83], v[196:201], a[236:239], v31, v33 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s12, s33, 0x280
	s_mov_b32 m0, s29
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[128:133], v[196:201], a[240:243], v30, v33 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[122:127], v[196:201], a[244:247], v30, v33 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[116:121], v[196:201], a[248:251], v30, v33 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[84:89], v[196:201], a[252:255], v30, v33 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s12, s2, 0xe0
	s_lshl_b32 s12, s12, 9
	s_add_i32 s13, s16, 0x700
	s_lshl_b32 s13, s13, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:41], v[92:97], a[0:3], v28, v34 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(16) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[26:27], v18 offset:4096
	ds_read_b64 v[30:31], v17 offset:5120
	s_mov_b32 s14, 0x10000
	v_lshl_add_u32 v3, v3, 2, s14
	ds_read_b64 v[78:79], v3 offset:47456
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[42:47], v[92:97], a[4:7], v28, v34 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_lshl_add_u32 v14, v14, 2, s14
	ds_read_b64 v[80:81], v14 offset:47464
	ds_read_b64 v[82:83], v14 offset:47472
	s_add_i32 s14, s44, 0x2a000
	s_mov_b32 m0, s59
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[20:25], v[92:97], a[8:11], v28, v34 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen lds
	ds_read_b64 v[84:85], v3 offset:47584
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[48:53], v[92:97], a[12:15], v28, v34 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[86:87], v14 offset:47592
	ds_read_b64 v[88:89], v14 offset:47600
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[54:59], v[92:97], a[16:19], v29, v34 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[98:99], v3 offset:47712
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[60:65], v[92:97], a[20:23], v29, v34 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[100:101], v14 offset:47720
	ds_read_b64 v[102:103], v14 offset:47728
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[66:71], v[92:97], a[24:27], v29, v34 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen offset:1024 lds
	ds_read_b64 v[116:117], v3 offset:47840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[72:77], v[92:97], a[28:31], v29, v34 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[118:119], v14 offset:47848
	ds_read_b64 v[120:121], v14 offset:47856
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[72:77], v[104:109], a[32:35], v29, v34 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[92:93], v3 offset:47968
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[66:71], v[104:109], a[36:39], v29, v34 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[94:95], v14 offset:47976
	ds_read_b64 v[96:97], v14 offset:47984
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[60:65], v[104:109], a[40:43], v29, v34 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen offset:2048 lds
	ds_read_b64 v[122:123], v3 offset:48096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[54:59], v[104:109], a[44:47], v29, v34 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[124:125], v14 offset:48104
	ds_read_b64 v[126:127], v14 offset:48112
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[48:53], v[104:109], a[48:51], v28, v34 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[128:129], v3 offset:48224
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[20:25], v[104:109], a[52:55], v28, v34 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[130:131], v14 offset:48232
	ds_read_b64 v[132:133], v14 offset:48240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[42:47], v[104:109], a[56:59], v28, v34 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen offset:3072 lds
	ds_read_b64 v[134:135], v3 offset:48352
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[36:41], v[104:109], a[60:63], v28, v34 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[136:137], v14 offset:48360
	ds_read_b64 v[138:139], v14 offset:48368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[36:41], v[110:115], a[64:67], v28, v34 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[104:107], v19 offset:55296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[42:47], v[110:115], a[68:71], v28, v34 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[108:109], v90 offset:6144
	s_add_i32 s14, s44, 0x2b000
	s_mov_b32 m0, s58
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[20:25], v[110:115], a[72:75], v28, v34 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen lds
	ds_read_b128 v[140:143], v19 offset:56320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[48:53], v[110:115], a[76:79], v28, v34 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[144:145], v90 offset:6656
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[54:59], v[110:115], a[80:83], v29, v34 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[146:149], v19 offset:57344
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[60:65], v[110:115], a[84:87], v29, v34 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[150:151], v90 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[66:71], v[110:115], a[88:91], v29, v34 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen offset:1024 lds
	ds_read_b128 v[176:179], v19 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[72:77], v[110:115], a[92:95], v29, v34 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[180:181], v90 offset:7680
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[72:77], v[152:157], a[96:99], v29, v34 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[110:113], v19 offset:59392
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[66:71], v[152:157], a[100:103], v29, v34 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[114:115], v90 offset:8192
	s_add_i32 s12, s34, s12
	s_mov_b32 m0, s60
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[60:65], v[152:157], a[104:107], v29, v34 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v16, s[40:43], s12 offen lds
	ds_read_b128 v[182:185], v19 offset:60416
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[54:59], v[152:157], a[108:111], v29, v34 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[186:187], v90 offset:8704
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[48:53], v[152:157], a[112:115], v28, v34 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[188:191], v19 offset:61440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[20:25], v[152:157], a[116:119], v28, v34 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[192:193], v90 offset:9216
	s_add_i32 s12, s34, s13
	s_mov_b32 m0, s53
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[42:47], v[152:157], a[120:123], v28, v34 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v16, s[8:11], s12 offen lds
	ds_read_b128 v[194:197], v19 offset:62464
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[36:41], v[152:157], a[124:127], v28, v34 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[198:199], v90 offset:9728
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[36:41], v[158:163], a[128:131], v28, v35 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[42:47], v[158:163], a[132:135], v28, v35 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[20:25], v[158:163], a[136:139], v28, v35 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[48:53], v[158:163], a[140:143], v28, v35 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[54:59], v[158:163], a[144:147], v29, v35 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[60:65], v[158:163], a[148:151], v29, v35 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[66:71], v[158:163], a[152:155], v29, v35 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[72:77], v[158:163], a[156:159], v29, v35 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[72:77], v[164:169], a[160:163], v29, v35 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[66:71], v[164:169], a[164:167], v29, v35 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[60:65], v[164:169], a[168:171], v29, v35 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[54:59], v[164:169], a[172:175], v29, v35 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[48:53], v[164:169], a[176:179], v28, v35 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[20:25], v[164:169], a[180:183], v28, v35 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[42:47], v[164:169], a[184:187], v28, v35 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[36:41], v[164:169], a[188:191], v28, v35 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[36:41], v[170:175], a[192:195], v28, v35 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[42:47], v[170:175], a[196:199], v28, v35 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[20:25], v[170:175], a[200:203], v28, v35 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[48:53], v[170:175], a[204:207], v28, v35 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[54:59], v[170:175], a[208:211], v29, v35 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[60:65], v[170:175], a[212:215], v29, v35 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[66:71], v[170:175], a[216:219], v29, v35 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[72:77], v[170:175], a[220:223], v29, v35 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[72:77], v[202:207], a[224:227], v29, v35 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[66:71], v[202:207], a[228:231], v29, v35 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[60:65], v[202:207], a[232:235], v29, v35 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[54:59], v[202:207], a[236:239], v29, v35 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[48:53], v[202:207], a[240:243], v28, v35 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[20:25], v[202:207], a[244:247], v28, v35 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[42:47], v[202:207], a[248:251], v28, v35 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[36:41], v[202:207], a[252:255], v28, v35 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s12, s2, 0x100
	s_lshl_b32 s12, s12, 9
	s_add_i32 s13, s16, 0x800
	s_lshl_b32 s13, s13, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[78:83], v[104:109], a[0:3], v26, v30 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(8) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[20:21], v18
	ds_read_b64 v[22:23], v17 offset:1024
	s_mov_b32 s14, 0x10000
	v_lshl_add_u32 v11, v11, 2, s14
	ds_read_b64 v[32:33], v11 offset:14400
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[84:89], v[104:109], a[4:7], v26, v30 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[34:35], v11 offset:14408
	v_lshl_add_u32 v9, v9, 2, s14
	ds_read_b64 v[36:37], v9 offset:14416
	s_add_i32 s14, s44, 0x30000
	s_mov_b32 m0, s63
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[98:103], v[104:109], a[8:11], v26, v30 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen lds
	ds_read_b64 v[38:39], v11 offset:14528
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[116:121], v[104:109], a[12:15], v26, v30 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[40:41], v11 offset:14536
	ds_read_b64 v[42:43], v9 offset:14544
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[92:97], v[104:109], a[16:19], v27, v30 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[44:45], v11 offset:14656
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[122:127], v[104:109], a[20:23], v27, v30 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[46:47], v11 offset:14664
	ds_read_b64 v[48:49], v9 offset:14672
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[128:133], v[104:109], a[24:27], v27, v30 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen offset:1024 lds
	ds_read_b64 v[50:51], v11 offset:14784
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[134:139], v[104:109], a[28:31], v27, v30 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[52:53], v11 offset:14792
	ds_read_b64 v[54:55], v9 offset:14800
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[134:139], v[140:145], a[32:35], v27, v30 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[56:57], v11 offset:14912
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[128:133], v[140:145], a[36:39], v27, v30 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[58:59], v11 offset:14920
	ds_read_b64 v[60:61], v9 offset:14928
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[122:127], v[140:145], a[40:43], v27, v30 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen offset:2048 lds
	ds_read_b64 v[62:63], v11 offset:15040
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[92:97], v[140:145], a[44:47], v27, v30 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[64:65], v11 offset:15048
	ds_read_b64 v[66:67], v9 offset:15056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[116:121], v[140:145], a[48:51], v26, v30 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[68:69], v11 offset:15168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[98:103], v[140:145], a[52:55], v26, v30 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[70:71], v11 offset:15176
	ds_read_b64 v[72:73], v9 offset:15184
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[84:89], v[140:145], a[56:59], v26, v30 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen offset:3072 lds
	ds_read_b64 v[104:105], v11 offset:15296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[78:83], v[140:145], a[60:63], v26, v30 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[106:107], v11 offset:15304
	ds_read_b64 v[108:109], v9 offset:15312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[78:83], v[146:151], a[64:67], v26, v30 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[140:143], v19 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[84:89], v[146:151], a[68:71], v26, v30 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[144:145], v0 offset:22528
	s_add_i32 s14, s44, 0x31000
	s_mov_b32 m0, s55
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[98:103], v[146:151], a[72:75], v26, v30 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen lds
	ds_read_b128 v[152:155], v19 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[116:121], v[146:151], a[76:79], v26, v30 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[156:157], v0 offset:23040
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[92:97], v[146:151], a[80:83], v27, v30 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[158:161], v19 offset:8192
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[122:127], v[146:151], a[84:87], v27, v30 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[162:163], v0 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[128:133], v[146:151], a[88:91], v27, v30 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen offset:1024 lds
	ds_read_b128 v[164:167], v19 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[134:139], v[146:151], a[92:95], v27, v30 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[168:169], v0 offset:24064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[134:139], v[176:181], a[96:99], v27, v30 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[146:149], v19 offset:10240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[128:133], v[176:181], a[100:103], v27, v30 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[150:151], v0 offset:24576
	s_add_i32 s12, s34, s12
	s_mov_b32 m0, s62
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[122:127], v[176:181], a[104:107], v27, v30 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v16, s[40:43], s12 offen lds
	ds_read_b128 v[170:173], v19 offset:11264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[92:97], v[176:181], a[108:111], v27, v30 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[174:175], v0 offset:25088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[116:121], v[176:181], a[112:115], v26, v30 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[200:203], v19 offset:12288
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[98:103], v[176:181], a[116:119], v26, v30 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[204:205], v0 offset:25600
	s_add_i32 s12, s34, s13
	s_mov_b32 m0, s57
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[84:89], v[176:181], a[120:123], v26, v30 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v16, s[8:11], s12 offen lds
	ds_read_b128 v[206:209], v19 offset:13312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[78:83], v[176:181], a[124:127], v26, v30 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[210:211], v0 offset:26112
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[78:83], v[110:115], a[128:131], v26, v31 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[84:89], v[110:115], a[132:135], v26, v31 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[98:103], v[110:115], a[136:139], v26, v31 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[116:121], v[110:115], a[140:143], v26, v31 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s12, s17, 0x300
	s_mov_b32 m0, s35
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[92:97], v[110:115], a[144:147], v27, v31 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[122:127], v[110:115], a[148:151], v27, v31 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[128:133], v[110:115], a[152:155], v27, v31 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s12, s20, 0x300
	s_mov_b32 m0, s45
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[134:139], v[110:115], a[156:159], v27, v31 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[134:139], v[182:187], a[160:163], v27, v31 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[128:133], v[182:187], a[164:167], v27, v31 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[122:127], v[182:187], a[168:171], v27, v31 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_i32 s12, s22, 0x300
	s_mov_b32 m0, s46
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[92:97], v[182:187], a[172:175], v27, v31 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[116:121], v[182:187], a[176:179], v26, v31 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[98:103], v[182:187], a[180:183], v26, v31 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s12, s24, 0x300
	s_mov_b32 m0, s47
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[84:89], v[182:187], a[184:187], v26, v31 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[78:83], v[182:187], a[188:191], v26, v31 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[78:83], v[188:193], a[192:195], v26, v31 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[84:89], v[188:193], a[196:199], v26, v31 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s12, s26, 0x300
	s_mov_b32 m0, s48
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[98:103], v[188:193], a[200:203], v26, v31 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[116:121], v[188:193], a[204:207], v26, v31 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[92:97], v[188:193], a[208:211], v27, v31 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s12, s30, 0x300
	s_mov_b32 m0, s49
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[122:127], v[188:193], a[212:215], v27, v31 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[128:133], v[188:193], a[216:219], v27, v31 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[134:139], v[188:193], a[220:223], v27, v31 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[134:139], v[194:199], a[224:227], v27, v31 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s12, s32, 0x300
	s_mov_b32 m0, s50
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[128:133], v[194:199], a[228:231], v27, v31 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[122:127], v[194:199], a[232:235], v27, v31 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[92:97], v[194:199], a[236:239], v27, v31 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s12, s33, 0x300
	s_mov_b32 m0, s51
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[116:121], v[194:199], a[240:243], v26, v31 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[98:103], v[194:199], a[244:247], v26, v31 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[84:89], v[194:199], a[248:251], v26, v31 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[78:83], v[194:199], a[252:255], v26, v31 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s12, s2, 0x120
	s_lshl_b32 s12, s12, 9
	s_add_i32 s13, s16, 0x900
	s_lshl_b32 s13, s13, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[32:37], v[140:145], a[0:3], v20, v22 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(16) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[24:25], v18 offset:2048
	ds_read_b64 v[26:27], v17 offset:3072
	ds_read_b64 v[74:75], v6 offset:47392
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[38:43], v[140:145], a[4:7], v20, v22 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[76:77], v6 offset:47400
	ds_read_b64 v[78:79], v6 offset:47408
	s_add_i32 s14, s44, 0x36000
	s_mov_b32 m0, s54
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[44:49], v[140:145], a[8:11], v20, v22 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen lds
	ds_read_b64 v[80:81], v6 offset:47520
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[50:55], v[140:145], a[12:15], v20, v22 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[82:83], v6 offset:47528
	ds_read_b64 v[84:85], v6 offset:47536
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[56:61], v[140:145], a[16:19], v21, v22 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[92:93], v6 offset:47648
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[62:67], v[140:145], a[20:23], v21, v22 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[94:95], v6 offset:47656
	ds_read_b64 v[96:97], v6 offset:47664
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[68:73], v[140:145], a[24:27], v21, v22 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen offset:1024 lds
	ds_read_b64 v[98:99], v6 offset:47776
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[104:109], v[140:145], a[28:31], v21, v22 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[100:101], v6 offset:47784
	ds_read_b64 v[102:103], v6 offset:47792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[104:109], v[152:157], a[32:35], v21, v22 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[110:111], v6 offset:47904
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[68:73], v[152:157], a[36:39], v21, v22 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[112:113], v6 offset:47912
	ds_read_b64 v[114:115], v6 offset:47920
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[62:67], v[152:157], a[40:43], v21, v22 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen offset:2048 lds
	ds_read_b64 v[116:117], v6 offset:48032
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[56:61], v[152:157], a[44:47], v21, v22 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[118:119], v6 offset:48040
	ds_read_b64 v[120:121], v6 offset:48048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[50:55], v[152:157], a[48:51], v20, v22 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[122:123], v6 offset:48160
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[44:49], v[152:157], a[52:55], v20, v22 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[124:125], v6 offset:48168
	ds_read_b64 v[126:127], v6 offset:48176
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[38:43], v[152:157], a[56:59], v20, v22 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen offset:3072 lds
	ds_read_b64 v[128:129], v6 offset:48288
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[32:37], v[152:157], a[60:63], v20, v22 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[130:131], v6 offset:48296
	ds_read_b64 v[132:133], v6 offset:48304
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[32:37], v[158:163], a[64:67], v20, v22 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[134:137], v19 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[38:43], v[158:163], a[68:71], v20, v22 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[138:139], v0 offset:47104
	s_add_i32 s14, s44, 0x37000
	s_mov_b32 m0, s56
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[44:49], v[158:163], a[72:75], v20, v22 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen lds
	ds_read_b128 v[140:143], v19 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[50:55], v[158:163], a[76:79], v20, v22 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[144:145], v0 offset:47616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[56:61], v[158:163], a[80:83], v21, v22 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[152:155], v19 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[62:67], v[158:163], a[84:87], v21, v22 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[156:157], v0 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[68:73], v[158:163], a[88:91], v21, v22 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen offset:1024 lds
	ds_read_b128 v[176:179], v19 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[104:109], v[158:163], a[92:95], v21, v22 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[180:181], v0 offset:48640
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[104:109], v[164:169], a[96:99], v21, v22 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[158:161], v19 offset:34816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[68:73], v[164:169], a[100:103], v21, v22 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[162:163], v0 offset:49152
	s_add_i32 s12, s34, s12
	s_mov_b32 m0, s34
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[62:67], v[164:169], a[104:107], v21, v22 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v16, s[40:43], s12 offen lds
	ds_read_b128 v[182:185], v19 offset:35840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[56:61], v[164:169], a[108:111], v21, v22 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[186:187], v0 offset:49664
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[50:55], v[164:169], a[112:115], v20, v22 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[188:191], v19 offset:36864
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[44:49], v[164:169], a[116:119], v20, v22 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[192:193], v0 offset:50176
	s_add_i32 s12, s34, s13
	s_mov_b32 m0, s52
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[38:43], v[164:169], a[120:123], v20, v22 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v16, s[8:11], s12 offen lds
	ds_read_b128 v[194:197], v19 offset:37888
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[32:37], v[164:169], a[124:127], v20, v22 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[198:199], v0 offset:50688
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[32:37], v[146:151], a[128:131], v20, v23 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[38:43], v[146:151], a[132:135], v20, v23 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[44:49], v[146:151], a[136:139], v20, v23 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[50:55], v[146:151], a[140:143], v20, v23 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s12, s17, 0x380
	s_mov_b32 m0, s18
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[56:61], v[146:151], a[144:147], v21, v23 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[62:67], v[146:151], a[148:151], v21, v23 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[68:73], v[146:151], a[152:155], v21, v23 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s12, s20, 0x380
	s_mov_b32 m0, s21
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[104:109], v[146:151], a[156:159], v21, v23 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[104:109], v[170:175], a[160:163], v21, v23 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[68:73], v[170:175], a[164:167], v21, v23 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[62:67], v[170:175], a[168:171], v21, v23 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_i32 s12, s22, 0x380
	s_mov_b32 m0, s23
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[56:61], v[170:175], a[172:175], v21, v23 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[50:55], v[170:175], a[176:179], v20, v23 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[44:49], v[170:175], a[180:183], v20, v23 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s12, s24, 0x380
	s_mov_b32 m0, s25
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[38:43], v[170:175], a[184:187], v20, v23 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[32:37], v[170:175], a[188:191], v20, v23 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[32:37], v[200:205], a[192:195], v20, v23 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[38:43], v[200:205], a[196:199], v20, v23 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s12, s26, 0x380
	s_mov_b32 m0, s27
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[44:49], v[200:205], a[200:203], v20, v23 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[50:55], v[200:205], a[204:207], v20, v23 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[56:61], v[200:205], a[208:211], v21, v23 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s12, s30, 0x380
	s_mov_b32 m0, s28
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[62:67], v[200:205], a[212:215], v21, v23 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[68:73], v[200:205], a[216:219], v21, v23 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[104:109], v[200:205], a[220:223], v21, v23 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[104:109], v[206:211], a[224:227], v21, v23 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s12, s32, 0x380
	s_mov_b32 m0, s19
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[68:73], v[206:211], a[228:231], v21, v23 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[62:67], v[206:211], a[232:235], v21, v23 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[56:61], v[206:211], a[236:239], v21, v23 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s12, s33, 0x380
	s_mov_b32 m0, s29
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[50:55], v[206:211], a[240:243], v20, v23 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[44:49], v[206:211], a[244:247], v20, v23 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[38:43], v[206:211], a[248:251], v20, v23 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[32:37], v[206:211], a[252:255], v20, v23 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s12, s2, 0x140
	s_lshl_b32 s12, s12, 9
	s_add_i32 s13, s16, 0xa00
	s_lshl_b32 s13, s13, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[74:79], v[134:139], a[0:3], v24, v26 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(16) lgkmcnt(0)
	s_barrier
	ds_read_b128 v[20:23], v4 offset:14336
	ds_read_b128 v[28:31], v4 offset:14352
	ds_read_b64 v[32:33], v18 offset:4096
	ds_read_b64 v[34:35], v17 offset:5120
	ds_read_b64 v[36:37], v6 offset:14336
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[80:85], v[134:139], a[4:7], v24, v26 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[38:39], v6 offset:14344
	ds_read_b64 v[40:41], v6 offset:14352
	s_add_i32 s14, s44, 0x3c000
	s_mov_b32 m0, s59
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[92:97], v[134:139], a[8:11], v24, v26 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen lds
	ds_read_b64 v[42:43], v6 offset:14464
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[98:103], v[134:139], a[12:15], v24, v26 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[44:45], v6 offset:14472
	ds_read_b64 v[46:47], v6 offset:14480
	s_waitcnt lgkmcnt(9)
	ds_write_b64 v5, v[20:21] offset:14336
	ds_write_b64 v5, v[22:23] offset:14344
	s_waitcnt lgkmcnt(10)
	ds_write_b64 v5, v[28:29] offset:14352
	ds_write_b64 v5, v[30:31] offset:14360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[110:115], v[134:139], a[16:19], v25, v26 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[48:49], v6 offset:14592
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[116:121], v[134:139], a[20:23], v25, v26 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[50:51], v6 offset:14600
	ds_read_b64 v[52:53], v6 offset:14608
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[122:127], v[134:139], a[24:27], v25, v26 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen offset:1024 lds
	ds_read_b64 v[54:55], v6 offset:14720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[128:133], v[134:139], a[28:31], v25, v26 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[56:57], v6 offset:14728
	ds_read_b64 v[58:59], v6 offset:14736
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[128:133], v[140:145], a[32:35], v25, v26 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[60:61], v6 offset:14848
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[122:127], v[140:145], a[36:39], v25, v26 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[62:63], v6 offset:14856
	ds_read_b64 v[64:65], v6 offset:14864
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[116:121], v[140:145], a[40:43], v25, v26 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen offset:2048 lds
	ds_read_b64 v[66:67], v6 offset:14976
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[110:115], v[140:145], a[44:47], v25, v26 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[68:69], v6 offset:14984
	ds_read_b64 v[70:71], v6 offset:14992
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[98:103], v[140:145], a[48:51], v24, v26 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[104:105], v6 offset:15104
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[92:97], v[140:145], a[52:55], v24, v26 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[106:107], v6 offset:15112
	ds_read_b64 v[108:109], v6 offset:15120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[80:85], v[140:145], a[56:59], v24, v26 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen offset:3072 lds
	ds_read_b64 v[134:135], v6 offset:15232
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[74:79], v[140:145], a[60:63], v24, v26 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[136:137], v6 offset:15240
	ds_read_b64 v[138:139], v6 offset:15248
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[74:79], v[152:157], a[64:67], v24, v26 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[140:143], v19 offset:55296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[80:85], v[152:157], a[68:71], v24, v26 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[144:145], v90 offset:6144
	s_add_i32 s14, s44, 0x3d000
	s_mov_b32 m0, s58
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[92:97], v[152:157], a[72:75], v24, v26 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen lds
	ds_read_b128 v[146:149], v19 offset:56320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[98:103], v[152:157], a[76:79], v24, v26 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[150:151], v90 offset:6656
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[110:115], v[152:157], a[80:83], v25, v26 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[164:167], v19 offset:57344
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[116:121], v[152:157], a[84:87], v25, v26 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[168:169], v90 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[122:127], v[152:157], a[88:91], v25, v26 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen offset:1024 lds
	ds_read_b128 v[170:173], v19 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[128:133], v[152:157], a[92:95], v25, v26 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[174:175], v90 offset:7680
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[128:133], v[176:181], a[96:99], v25, v26 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[152:155], v19 offset:59392
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[122:127], v[176:181], a[100:103], v25, v26 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[156:157], v90 offset:8192
	s_add_i32 s12, s34, s12
	s_mov_b32 m0, s60
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[116:121], v[176:181], a[104:107], v25, v26 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v16, s[40:43], s12 offen lds
	ds_read_b128 v[200:203], v19 offset:60416
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[110:115], v[176:181], a[108:111], v25, v26 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[204:205], v90 offset:8704
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[98:103], v[176:181], a[112:115], v24, v26 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[206:209], v19 offset:61440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[92:97], v[176:181], a[116:119], v24, v26 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[210:211], v90 offset:9216
	s_add_i32 s12, s34, s13
	s_mov_b32 m0, s53
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[80:85], v[176:181], a[120:123], v24, v26 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v16, s[8:11], s12 offen lds
	ds_read_b128 v[212:215], v19 offset:62464
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[74:79], v[176:181], a[124:127], v24, v26 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[216:217], v90 offset:9728
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[74:79], v[158:163], a[128:131], v24, v27 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[80:85], v[158:163], a[132:135], v24, v27 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[92:97], v[158:163], a[136:139], v24, v27 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[98:103], v[158:163], a[140:143], v24, v27 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s12, s17, 0x400
	s_mov_b32 m0, s35
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[110:115], v[158:163], a[144:147], v25, v27 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[116:121], v[158:163], a[148:151], v25, v27 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[122:127], v[158:163], a[152:155], v25, v27 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s12, s20, 0x400
	s_mov_b32 m0, s45
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[128:133], v[158:163], a[156:159], v25, v27 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[128:133], v[182:187], a[160:163], v25, v27 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[122:127], v[182:187], a[164:167], v25, v27 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[116:121], v[182:187], a[168:171], v25, v27 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_i32 s12, s22, 0x400
	s_mov_b32 m0, s46
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[110:115], v[182:187], a[172:175], v25, v27 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[98:103], v[182:187], a[176:179], v24, v27 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[92:97], v[182:187], a[180:183], v24, v27 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s12, s24, 0x400
	s_mov_b32 m0, s47
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[80:85], v[182:187], a[184:187], v24, v27 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[74:79], v[182:187], a[188:191], v24, v27 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[74:79], v[188:193], a[192:195], v24, v27 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[80:85], v[188:193], a[196:199], v24, v27 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s12, s26, 0x400
	s_mov_b32 m0, s48
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[92:97], v[188:193], a[200:203], v24, v27 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[98:103], v[188:193], a[204:207], v24, v27 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[110:115], v[188:193], a[208:211], v25, v27 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s12, s30, 0x400
	s_mov_b32 m0, s49
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[116:121], v[188:193], a[212:215], v25, v27 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[122:127], v[188:193], a[216:219], v25, v27 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[128:133], v[188:193], a[220:223], v25, v27 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[128:133], v[194:199], a[224:227], v25, v27 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s12, s32, 0x400
	s_mov_b32 m0, s50
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[122:127], v[194:199], a[228:231], v25, v27 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[116:121], v[194:199], a[232:235], v25, v27 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[110:115], v[194:199], a[236:239], v25, v27 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s12, s33, 0x400
	s_mov_b32 m0, s51
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[98:103], v[194:199], a[240:243], v24, v27 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[92:97], v[194:199], a[244:247], v24, v27 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[80:85], v[194:199], a[248:251], v24, v27 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[74:79], v[194:199], a[252:255], v24, v27 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s12, s2, 0x160
	s_lshl_b32 s12, s12, 9
	s_add_i32 s13, s16, 0xb00
	s_lshl_b32 s13, s13, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:41], v[140:145], a[0:3], v32, v34 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(16) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[20:21], v18
	ds_read_b64 v[22:23], v17 offset:1024
	ds_read_b64 v[24:25], v8 offset:14432
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[42:47], v[140:145], a[4:7], v32, v34 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[26:27], v13 offset:14440
	ds_read_b64 v[28:29], v13 offset:14448
	s_add_i32 s14, s44, 0x42000
	s_mov_b32 m0, s63
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[48:53], v[140:145], a[8:11], v32, v34 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen lds
	ds_read_b64 v[72:73], v8 offset:14560
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[54:59], v[140:145], a[12:15], v32, v34 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[74:75], v13 offset:14568
	ds_read_b64 v[76:77], v13 offset:14576
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[60:65], v[140:145], a[16:19], v33, v34 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v8 offset:14688
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[66:71], v[140:145], a[20:23], v33, v34 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[80:81], v13 offset:14696
	ds_read_b64 v[82:83], v13 offset:14704
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[104:109], v[140:145], a[24:27], v33, v34 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen offset:1024 lds
	ds_read_b64 v[84:85], v8 offset:14816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[134:139], v[140:145], a[28:31], v33, v34 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[86:87], v13 offset:14824
	ds_read_b64 v[88:89], v13 offset:14832
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[134:139], v[146:151], a[32:35], v33, v34 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[92:93], v8 offset:14944
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[104:109], v[146:151], a[36:39], v33, v34 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[94:95], v13 offset:14952
	ds_read_b64 v[96:97], v13 offset:14960
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[66:71], v[146:151], a[40:43], v33, v34 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen offset:2048 lds
	ds_read_b64 v[98:99], v8 offset:15072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[60:65], v[146:151], a[44:47], v33, v34 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[100:101], v13 offset:15080
	ds_read_b64 v[102:103], v13 offset:15088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[54:59], v[146:151], a[48:51], v32, v34 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[110:111], v8 offset:15200
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[48:53], v[146:151], a[52:55], v32, v34 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[112:113], v13 offset:15208
	ds_read_b64 v[114:115], v13 offset:15216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[42:47], v[146:151], a[56:59], v32, v34 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen offset:3072 lds
	ds_read_b64 v[116:117], v8 offset:15328
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[36:41], v[146:151], a[60:63], v32, v34 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[118:119], v13 offset:15336
	ds_read_b64 v[120:121], v13 offset:15344
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[36:41], v[164:169], a[64:67], v32, v34 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[122:125], v19 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[42:47], v[164:169], a[68:71], v32, v34 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[126:127], v0 offset:22528
	s_add_i32 s14, s44, 0x43000
	s_mov_b32 m0, s55
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[48:53], v[164:169], a[72:75], v32, v34 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen lds
	ds_read_b128 v[128:131], v19 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[54:59], v[164:169], a[76:79], v32, v34 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[132:133], v0 offset:23040
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[60:65], v[164:169], a[80:83], v33, v34 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[140:143], v19 offset:8192
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[66:71], v[164:169], a[84:87], v33, v34 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[144:145], v0 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[104:109], v[164:169], a[88:91], v33, v34 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen offset:1024 lds
	ds_read_b128 v[146:149], v19 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[134:139], v[164:169], a[92:95], v33, v34 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[150:151], v0 offset:24064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[134:139], v[170:175], a[96:99], v33, v34 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[158:161], v19 offset:10240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[104:109], v[170:175], a[100:103], v33, v34 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[162:163], v0 offset:24576
	s_add_i32 s12, s34, s12
	s_mov_b32 m0, s62
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[66:71], v[170:175], a[104:107], v33, v34 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v16, s[40:43], s12 offen lds
	ds_read_b128 v[164:167], v19 offset:11264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[60:65], v[170:175], a[108:111], v33, v34 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[168:169], v0 offset:25088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[54:59], v[170:175], a[112:115], v32, v34 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[176:179], v19 offset:12288
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[48:53], v[170:175], a[116:119], v32, v34 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[180:181], v0 offset:25600
	s_add_i32 s12, s34, s13
	s_mov_b32 m0, s57
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[42:47], v[170:175], a[120:123], v32, v34 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v16, s[8:11], s12 offen lds
	ds_read_b128 v[182:185], v19 offset:13312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[36:41], v[170:175], a[124:127], v32, v34 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[186:187], v0 offset:26112
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[36:41], v[152:157], a[128:131], v32, v35 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[42:47], v[152:157], a[132:135], v32, v35 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[48:53], v[152:157], a[136:139], v32, v35 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[54:59], v[152:157], a[140:143], v32, v35 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[60:65], v[152:157], a[144:147], v33, v35 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[66:71], v[152:157], a[148:151], v33, v35 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[104:109], v[152:157], a[152:155], v33, v35 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[134:139], v[152:157], a[156:159], v33, v35 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[134:139], v[200:205], a[160:163], v33, v35 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[104:109], v[200:205], a[164:167], v33, v35 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[66:71], v[200:205], a[168:171], v33, v35 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[60:65], v[200:205], a[172:175], v33, v35 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[54:59], v[200:205], a[176:179], v32, v35 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[48:53], v[200:205], a[180:183], v32, v35 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[42:47], v[200:205], a[184:187], v32, v35 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[36:41], v[200:205], a[188:191], v32, v35 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[36:41], v[206:211], a[192:195], v32, v35 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[42:47], v[206:211], a[196:199], v32, v35 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[48:53], v[206:211], a[200:203], v32, v35 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[54:59], v[206:211], a[204:207], v32, v35 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[60:65], v[206:211], a[208:211], v33, v35 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[66:71], v[206:211], a[212:215], v33, v35 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[104:109], v[206:211], a[216:219], v33, v35 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[134:139], v[206:211], a[220:223], v33, v35 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[134:139], v[212:217], a[224:227], v33, v35 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[104:109], v[212:217], a[228:231], v33, v35 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[66:71], v[212:217], a[232:235], v33, v35 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[60:65], v[212:217], a[236:239], v33, v35 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[54:59], v[212:217], a[240:243], v32, v35 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[48:53], v[212:217], a[244:247], v32, v35 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[42:47], v[212:217], a[248:251], v32, v35 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[36:41], v[212:217], a[252:255], v32, v35 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s12, s2, 0x180
	s_lshl_b32 s12, s12, 9
	s_add_i32 s13, s16, 0xc00
	s_lshl_b32 s13, s13, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[24:29], v[122:127], a[0:3], v20, v22 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(8) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[30:31], v18 offset:2048
	ds_read_b64 v[32:33], v17 offset:3072
	ds_read_b64 v[34:35], v12 offset:47424
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[72:77], v[122:127], a[4:7], v20, v22 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[36:37], v12 offset:47432
	ds_read_b64 v[38:39], v10 offset:47440
	s_add_i32 s14, s44, 0x48000
	s_mov_b32 m0, s54
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[78:83], v[122:127], a[8:11], v20, v22 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen lds
	ds_read_b64 v[40:41], v12 offset:47552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[84:89], v[122:127], a[12:15], v20, v22 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[42:43], v12 offset:47560
	ds_read_b64 v[44:45], v10 offset:47568
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[92:97], v[122:127], a[16:19], v21, v22 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[46:47], v12 offset:47680
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[98:103], v[122:127], a[20:23], v21, v22 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[48:49], v12 offset:47688
	ds_read_b64 v[50:51], v10 offset:47696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[110:115], v[122:127], a[24:27], v21, v22 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen offset:1024 lds
	ds_read_b64 v[52:53], v12 offset:47808
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[116:121], v[122:127], a[28:31], v21, v22 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[54:55], v12 offset:47816
	ds_read_b64 v[56:57], v10 offset:47824
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[116:121], v[128:133], a[32:35], v21, v22 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[58:59], v12 offset:47936
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[110:115], v[128:133], a[36:39], v21, v22 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[60:61], v12 offset:47944
	ds_read_b64 v[62:63], v10 offset:47952
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[98:103], v[128:133], a[40:43], v21, v22 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen offset:2048 lds
	ds_read_b64 v[64:65], v12 offset:48064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[92:97], v[128:133], a[44:47], v21, v22 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[66:67], v12 offset:48072
	ds_read_b64 v[68:69], v10 offset:48080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[84:89], v[128:133], a[48:51], v20, v22 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[104:105], v12 offset:48192
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[78:83], v[128:133], a[52:55], v20, v22 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[106:107], v12 offset:48200
	ds_read_b64 v[108:109], v10 offset:48208
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[72:77], v[128:133], a[56:59], v20, v22 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen offset:3072 lds
	ds_read_b64 v[122:123], v12 offset:48320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[24:29], v[128:133], a[60:63], v20, v22 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[124:125], v12 offset:48328
	ds_read_b64 v[126:127], v10 offset:48336
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[24:29], v[140:145], a[64:67], v20, v22 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[128:131], v19 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[72:77], v[140:145], a[68:71], v20, v22 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[132:133], v0 offset:47104
	s_add_i32 s14, s44, 0x49000
	s_mov_b32 m0, s56
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[78:83], v[140:145], a[72:75], v20, v22 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen lds
	ds_read_b128 v[134:137], v19 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[84:89], v[140:145], a[76:79], v20, v22 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[138:139], v0 offset:47616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[92:97], v[140:145], a[80:83], v21, v22 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[152:155], v19 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[98:103], v[140:145], a[84:87], v21, v22 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[156:157], v0 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[110:115], v[140:145], a[88:91], v21, v22 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen offset:1024 lds
	ds_read_b128 v[170:173], v19 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[116:121], v[140:145], a[92:95], v21, v22 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[174:175], v0 offset:48640
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[116:121], v[146:151], a[96:99], v21, v22 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[140:143], v19 offset:34816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[110:115], v[146:151], a[100:103], v21, v22 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[144:145], v0 offset:49152
	s_add_i32 s12, s34, s12
	s_mov_b32 m0, s34
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[98:103], v[146:151], a[104:107], v21, v22 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v16, s[40:43], s12 offen lds
	ds_read_b128 v[188:191], v19 offset:35840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[92:97], v[146:151], a[108:111], v21, v22 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[192:193], v0 offset:49664
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[84:89], v[146:151], a[112:115], v20, v22 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[194:197], v19 offset:36864
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[78:83], v[146:151], a[116:119], v20, v22 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[198:199], v0 offset:50176
	s_add_i32 s12, s34, s13
	s_mov_b32 m0, s52
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[72:77], v[146:151], a[120:123], v20, v22 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v16, s[8:11], s12 offen lds
	ds_read_b128 v[200:203], v19 offset:37888
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[24:29], v[146:151], a[124:127], v20, v22 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[204:205], v0 offset:50688
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[24:29], v[158:163], a[128:131], v20, v23 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[72:77], v[158:163], a[132:135], v20, v23 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[78:83], v[158:163], a[136:139], v20, v23 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[84:89], v[158:163], a[140:143], v20, v23 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s12, s17, 0x480
	s_mov_b32 m0, s18
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[92:97], v[158:163], a[144:147], v21, v23 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[98:103], v[158:163], a[148:151], v21, v23 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[110:115], v[158:163], a[152:155], v21, v23 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s12, s20, 0x480
	s_mov_b32 m0, s21
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[116:121], v[158:163], a[156:159], v21, v23 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[116:121], v[164:169], a[160:163], v21, v23 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[110:115], v[164:169], a[164:167], v21, v23 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[98:103], v[164:169], a[168:171], v21, v23 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_i32 s12, s22, 0x480
	s_mov_b32 m0, s23
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[92:97], v[164:169], a[172:175], v21, v23 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[84:89], v[164:169], a[176:179], v20, v23 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[78:83], v[164:169], a[180:183], v20, v23 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s12, s24, 0x480
	s_mov_b32 m0, s25
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[72:77], v[164:169], a[184:187], v20, v23 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[24:29], v[164:169], a[188:191], v20, v23 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[24:29], v[176:181], a[192:195], v20, v23 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[72:77], v[176:181], a[196:199], v20, v23 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s12, s26, 0x480
	s_mov_b32 m0, s27
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[78:83], v[176:181], a[200:203], v20, v23 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[84:89], v[176:181], a[204:207], v20, v23 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[92:97], v[176:181], a[208:211], v21, v23 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s12, s30, 0x480
	s_mov_b32 m0, s28
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[98:103], v[176:181], a[212:215], v21, v23 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[110:115], v[176:181], a[216:219], v21, v23 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[116:121], v[176:181], a[220:223], v21, v23 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[116:121], v[182:187], a[224:227], v21, v23 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s12, s32, 0x480
	s_mov_b32 m0, s19
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[110:115], v[182:187], a[228:231], v21, v23 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[98:103], v[182:187], a[232:235], v21, v23 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[92:97], v[182:187], a[236:239], v21, v23 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s12, s33, 0x480
	s_mov_b32 m0, s29
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[84:89], v[182:187], a[240:243], v20, v23 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[78:83], v[182:187], a[244:247], v20, v23 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[72:77], v[182:187], a[248:251], v20, v23 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[24:29], v[182:187], a[252:255], v20, v23 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s12, s2, 0x1a0
	s_lshl_b32 s12, s12, 9
	s_add_i32 s13, s16, 0xd00
	s_lshl_b32 s13, s13, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:39], v[128:133], a[0:3], v30, v32 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(16) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[12:13], v18 offset:4096
	ds_read_b64 v[20:21], v17 offset:5120
	ds_read_b64 v[22:23], v6 offset:14368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[40:45], v[128:133], a[4:7], v30, v32 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[24:25], v6 offset:14376
	ds_read_b64 v[26:27], v6 offset:14384
	s_add_i32 s14, s44, 0x4e000
	s_mov_b32 m0, s59
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[46:51], v[128:133], a[8:11], v30, v32 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen lds
	ds_read_b64 v[70:71], v6 offset:14496
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[52:57], v[128:133], a[12:15], v30, v32 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[72:73], v6 offset:14504
	ds_read_b64 v[74:75], v6 offset:14512
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[58:63], v[128:133], a[16:19], v31, v32 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[76:77], v6 offset:14624
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[64:69], v[128:133], a[20:23], v31, v32 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v6 offset:14632
	ds_read_b64 v[80:81], v6 offset:14640
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[104:109], v[128:133], a[24:27], v31, v32 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen offset:1024 lds
	ds_read_b64 v[82:83], v6 offset:14752
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[122:127], v[128:133], a[28:31], v31, v32 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[84:85], v6 offset:14760
	ds_read_b64 v[86:87], v6 offset:14768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[122:127], v[134:139], a[32:35], v31, v32 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[92:93], v6 offset:14880
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[104:109], v[134:139], a[36:39], v31, v32 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[94:95], v6 offset:14888
	ds_read_b64 v[96:97], v6 offset:14896
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[64:69], v[134:139], a[40:43], v31, v32 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen offset:2048 lds
	ds_read_b64 v[98:99], v6 offset:15008
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[58:63], v[134:139], a[44:47], v31, v32 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[100:101], v6 offset:15016
	ds_read_b64 v[102:103], v6 offset:15024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[52:57], v[134:139], a[48:51], v30, v32 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[110:111], v6 offset:15136
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[46:51], v[134:139], a[52:55], v30, v32 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[112:113], v6 offset:15144
	ds_read_b64 v[114:115], v6 offset:15152
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[40:45], v[134:139], a[56:59], v30, v32 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen offset:3072 lds
	ds_read_b64 v[116:117], v6 offset:15264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[34:39], v[134:139], a[60:63], v30, v32 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[118:119], v6 offset:15272
	ds_read_b64 v[120:121], v6 offset:15280
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[34:39], v[152:157], a[64:67], v30, v32 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[128:131], v19 offset:55296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[40:45], v[152:157], a[68:71], v30, v32 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[132:133], v90 offset:6144
	s_add_i32 s14, s44, 0x4f000
	s_mov_b32 m0, s58
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[46:51], v[152:157], a[72:75], v30, v32 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen lds
	ds_read_b128 v[134:137], v19 offset:56320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[52:57], v[152:157], a[76:79], v30, v32 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[138:139], v90 offset:6656
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[58:63], v[152:157], a[80:83], v31, v32 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[146:149], v19 offset:57344
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[64:69], v[152:157], a[84:87], v31, v32 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[150:151], v90 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[104:109], v[152:157], a[88:91], v31, v32 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen offset:1024 lds
	ds_read_b128 v[158:161], v19 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[122:127], v[152:157], a[92:95], v31, v32 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[162:163], v90 offset:7680
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[122:127], v[170:175], a[96:99], v31, v32 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[152:155], v19 offset:59392
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[104:109], v[170:175], a[100:103], v31, v32 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[156:157], v90 offset:8192
	s_add_i32 s12, s34, s12
	s_mov_b32 m0, s60
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[64:69], v[170:175], a[104:107], v31, v32 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v16, s[40:43], s12 offen lds
	ds_read_b128 v[164:167], v19 offset:60416
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[58:63], v[170:175], a[108:111], v31, v32 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[168:169], v90 offset:8704
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[52:57], v[170:175], a[112:115], v30, v32 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[176:179], v19 offset:61440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[46:51], v[170:175], a[116:119], v30, v32 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[180:181], v90 offset:9216
	s_add_i32 s12, s34, s13
	s_mov_b32 m0, s53
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[40:45], v[170:175], a[120:123], v30, v32 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v16, s[8:11], s12 offen lds
	ds_read_b128 v[182:185], v19 offset:62464
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[34:39], v[170:175], a[124:127], v30, v32 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[186:187], v90 offset:9728
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[34:39], v[140:145], a[128:131], v30, v33 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[40:45], v[140:145], a[132:135], v30, v33 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[46:51], v[140:145], a[136:139], v30, v33 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[52:57], v[140:145], a[140:143], v30, v33 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s12, s17, 0x500
	s_mov_b32 m0, s35
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[58:63], v[140:145], a[144:147], v31, v33 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[64:69], v[140:145], a[148:151], v31, v33 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[104:109], v[140:145], a[152:155], v31, v33 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s12, s20, 0x500
	s_mov_b32 m0, s45
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[122:127], v[140:145], a[156:159], v31, v33 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[122:127], v[188:193], a[160:163], v31, v33 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[104:109], v[188:193], a[164:167], v31, v33 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[64:69], v[188:193], a[168:171], v31, v33 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_i32 s12, s22, 0x500
	s_mov_b32 m0, s46
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[58:63], v[188:193], a[172:175], v31, v33 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[52:57], v[188:193], a[176:179], v30, v33 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[46:51], v[188:193], a[180:183], v30, v33 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s12, s24, 0x500
	s_mov_b32 m0, s47
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[40:45], v[188:193], a[184:187], v30, v33 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[34:39], v[188:193], a[188:191], v30, v33 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[34:39], v[194:199], a[192:195], v30, v33 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[40:45], v[194:199], a[196:199], v30, v33 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s12, s26, 0x500
	s_mov_b32 m0, s48
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[46:51], v[194:199], a[200:203], v30, v33 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[52:57], v[194:199], a[204:207], v30, v33 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[58:63], v[194:199], a[208:211], v31, v33 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s12, s30, 0x500
	s_mov_b32 m0, s49
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[64:69], v[194:199], a[212:215], v31, v33 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[104:109], v[194:199], a[216:219], v31, v33 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[122:127], v[194:199], a[220:223], v31, v33 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[122:127], v[200:205], a[224:227], v31, v33 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s12, s32, 0x500
	s_mov_b32 m0, s50
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[104:109], v[200:205], a[228:231], v31, v33 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[64:69], v[200:205], a[232:235], v31, v33 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[58:63], v[200:205], a[236:239], v31, v33 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s12, s33, 0x500
	s_mov_b32 m0, s51
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[52:57], v[200:205], a[240:243], v30, v33 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[46:51], v[200:205], a[244:247], v30, v33 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[40:45], v[200:205], a[248:251], v30, v33 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[34:39], v[200:205], a[252:255], v30, v33 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s12, s2, 0x1c0
	s_lshl_b32 s12, s12, 9
	s_add_i32 s13, s16, 0xe00
	s_lshl_b32 s13, s13, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:27], v[128:133], a[0:3], v12, v20 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(16) lgkmcnt(0)
	s_barrier
	ds_read_b128 v[28:31], v4 offset:47360
	ds_read_b128 v[32:35], v4 offset:47376
	ds_read_b64 v[36:37], v18
	ds_read_b64 v[38:39], v17 offset:1024
	ds_read_b64 v[40:41], v6 offset:47360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[70:75], v[128:133], a[4:7], v12, v20 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[42:43], v6 offset:47368
	ds_read_b64 v[44:45], v6 offset:47376
	s_add_i32 s14, s44, 0x54000
	s_mov_b32 m0, s63
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[76:81], v[128:133], a[8:11], v12, v20 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen lds
	ds_read_b64 v[46:47], v6 offset:47488
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[82:87], v[128:133], a[12:15], v12, v20 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[48:49], v6 offset:47496
	ds_read_b64 v[50:51], v6 offset:47504
	s_waitcnt lgkmcnt(9)
	ds_write_b64 v5, v[28:29] offset:14336
	ds_write_b64 v5, v[30:31] offset:14344
	s_waitcnt lgkmcnt(10)
	ds_write_b64 v5, v[32:33] offset:14352
	ds_write_b64 v5, v[34:35] offset:14360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[92:97], v[128:133], a[16:19], v13, v20 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[28:29], v6 offset:47616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[98:103], v[128:133], a[20:23], v13, v20 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[30:31], v6 offset:47624
	ds_read_b64 v[32:33], v6 offset:47632
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[110:115], v[128:133], a[24:27], v13, v20 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen offset:1024 lds
	ds_read_b64 v[52:53], v6 offset:47744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[116:121], v[128:133], a[28:31], v13, v20 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[54:55], v6 offset:47752
	ds_read_b64 v[56:57], v6 offset:47760
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[116:121], v[134:139], a[32:35], v13, v20 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[58:59], v6 offset:47872
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[110:115], v[134:139], a[36:39], v13, v20 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[60:61], v6 offset:47880
	ds_read_b64 v[62:63], v6 offset:47888
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[98:103], v[134:139], a[40:43], v13, v20 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen offset:2048 lds
	ds_read_b64 v[64:65], v6 offset:48000
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[92:97], v[134:139], a[44:47], v13, v20 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[66:67], v6 offset:48008
	ds_read_b64 v[68:69], v6 offset:48016
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[82:87], v[134:139], a[48:51], v12, v20 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[104:105], v6 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[76:81], v[134:139], a[52:55], v12, v20 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[106:107], v6 offset:48136
	ds_read_b64 v[108:109], v6 offset:48144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[70:75], v[134:139], a[56:59], v12, v20 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen offset:3072 lds
	ds_read_b64 v[122:123], v6 offset:48256
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[22:27], v[134:139], a[60:63], v12, v20 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[124:125], v6 offset:48264
	ds_read_b64 v[126:127], v6 offset:48272
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[22:27], v[146:151], a[64:67], v12, v20 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[128:131], v19 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[70:75], v[146:151], a[68:71], v12, v20 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[132:133], v0 offset:22528
	s_add_i32 s14, s44, 0x55000
	s_mov_b32 m0, s55
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[76:81], v[146:151], a[72:75], v12, v20 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen lds
	ds_read_b128 v[134:137], v19 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[82:87], v[146:151], a[76:79], v12, v20 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[138:139], v0 offset:23040
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[92:97], v[146:151], a[80:83], v13, v20 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[140:143], v19 offset:8192
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[98:103], v[146:151], a[84:87], v13, v20 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[144:145], v0 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[110:115], v[146:151], a[88:91], v13, v20 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s14 offen offset:1024 lds
	ds_read_b128 v[170:173], v19 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[116:121], v[146:151], a[92:95], v13, v20 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[174:175], v0 offset:24064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[116:121], v[158:163], a[96:99], v13, v20 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[146:149], v19 offset:10240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[110:115], v[158:163], a[100:103], v13, v20 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[150:151], v0 offset:24576
	s_add_i32 s12, s34, s12
	s_mov_b32 m0, s62
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[98:103], v[158:163], a[104:107], v13, v20 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v16, s[40:43], s12 offen lds
	ds_read_b128 v[188:191], v19 offset:11264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[92:97], v[158:163], a[108:111], v13, v20 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[192:193], v0 offset:25088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[82:87], v[158:163], a[112:115], v12, v20 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[194:197], v19 offset:12288
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[76:81], v[158:163], a[116:119], v12, v20 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[198:199], v0 offset:25600
	s_add_i32 s12, s34, s13
	s_mov_b32 m0, s57
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[70:75], v[158:163], a[120:123], v12, v20 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v16, s[8:11], s12 offen lds
	ds_read_b128 v[200:203], v19 offset:13312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[22:27], v[158:163], a[124:127], v12, v20 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[204:205], v0 offset:26112
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[22:27], v[152:157], a[128:131], v12, v21 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[70:75], v[152:157], a[132:135], v12, v21 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[76:81], v[152:157], a[136:139], v12, v21 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[82:87], v[152:157], a[140:143], v12, v21 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s12, s17, 0x580
	s_mov_b32 m0, s18
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[92:97], v[152:157], a[144:147], v13, v21 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[98:103], v[152:157], a[148:151], v13, v21 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[110:115], v[152:157], a[152:155], v13, v21 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s12, s20, 0x580
	s_mov_b32 m0, s21
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[116:121], v[152:157], a[156:159], v13, v21 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[116:121], v[164:169], a[160:163], v13, v21 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[110:115], v[164:169], a[164:167], v13, v21 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[98:103], v[164:169], a[168:171], v13, v21 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_i32 s12, s22, 0x580
	s_mov_b32 m0, s23
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[92:97], v[164:169], a[172:175], v13, v21 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[82:87], v[164:169], a[176:179], v12, v21 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[76:81], v[164:169], a[180:183], v12, v21 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s12, s24, 0x580
	s_mov_b32 m0, s25
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[70:75], v[164:169], a[184:187], v12, v21 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[22:27], v[164:169], a[188:191], v12, v21 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[22:27], v[176:181], a[192:195], v12, v21 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[70:75], v[176:181], a[196:199], v12, v21 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s12, s26, 0x580
	s_mov_b32 m0, s27
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[76:81], v[176:181], a[200:203], v12, v21 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[82:87], v[176:181], a[204:207], v12, v21 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[92:97], v[176:181], a[208:211], v13, v21 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s12, s30, 0x580
	s_mov_b32 m0, s28
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[98:103], v[176:181], a[212:215], v13, v21 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[110:115], v[176:181], a[216:219], v13, v21 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[116:121], v[176:181], a[220:223], v13, v21 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[116:121], v[182:187], a[224:227], v13, v21 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s12, s32, 0x580
	s_mov_b32 m0, s19
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[110:115], v[182:187], a[228:231], v13, v21 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[98:103], v[182:187], a[232:235], v13, v21 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[92:97], v[182:187], a[236:239], v13, v21 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s12, s33, 0x580
	s_mov_b32 m0, s29
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[82:87], v[182:187], a[240:243], v12, v21 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v7, s[4:7], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[76:81], v[182:187], a[244:247], v12, v21 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[70:75], v[182:187], a[248:251], v12, v21 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[22:27], v[182:187], a[252:255], v12, v21 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s2, s2, 0x1e0
	s_lshl_b32 s2, s2, 9
	s_add_i32 s4, s16, 0xf00
	s_lshl_b32 s4, s4, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[40:45], v[128:133], a[0:3], v36, v38 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(16) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[4:5], v18 offset:2048
	ds_read_b64 v[12:13], v17 offset:3072
	ds_read_b64 v[20:21], v3 offset:47456
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[46:51], v[128:133], a[4:7], v36, v38 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[22:23], v14 offset:47464
	ds_read_b64 v[24:25], v14 offset:47472
	s_add_i32 s5, s44, 0x5a000
	s_mov_b32 m0, s54
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[28:33], v[128:133], a[8:11], v36, v38 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s5 offen lds
	ds_read_b64 v[70:71], v3 offset:47584
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[52:57], v[128:133], a[12:15], v36, v38 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[72:73], v14 offset:47592
	ds_read_b64 v[74:75], v14 offset:47600
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[58:63], v[128:133], a[16:19], v37, v38 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[76:77], v3 offset:47712
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[64:69], v[128:133], a[20:23], v37, v38 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v14 offset:47720
	ds_read_b64 v[80:81], v14 offset:47728
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[104:109], v[128:133], a[24:27], v37, v38 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s5 offen offset:1024 lds
	ds_read_b64 v[82:83], v3 offset:47840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[122:127], v[128:133], a[28:31], v37, v38 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[84:85], v14 offset:47848
	ds_read_b64 v[86:87], v14 offset:47856
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[122:127], v[134:139], a[32:35], v37, v38 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[92:93], v3 offset:47968
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[104:109], v[134:139], a[36:39], v37, v38 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[94:95], v14 offset:47976
	ds_read_b64 v[96:97], v14 offset:47984
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[64:69], v[134:139], a[40:43], v37, v38 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s5 offen offset:2048 lds
	ds_read_b64 v[98:99], v3 offset:48096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[58:63], v[134:139], a[44:47], v37, v38 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[100:101], v14 offset:48104
	ds_read_b64 v[102:103], v14 offset:48112
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[52:57], v[134:139], a[48:51], v36, v38 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[110:111], v3 offset:48224
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[28:33], v[134:139], a[52:55], v36, v38 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[112:113], v14 offset:48232
	ds_read_b64 v[114:115], v14 offset:48240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[46:51], v[134:139], a[56:59], v36, v38 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s5 offen offset:3072 lds
	ds_read_b64 v[116:117], v3 offset:48352
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[40:45], v[134:139], a[60:63], v36, v38 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[118:119], v14 offset:48360
	ds_read_b64 v[120:121], v14 offset:48368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[40:45], v[140:145], a[64:67], v36, v38 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[128:131], v19 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[46:51], v[140:145], a[68:71], v36, v38 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[132:133], v0 offset:47104
	s_add_i32 s5, s44, 0x5b000
	s_mov_b32 m0, s56
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[28:33], v[140:145], a[72:75], v36, v38 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s5 offen lds
	ds_read_b128 v[134:137], v19 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[52:57], v[140:145], a[76:79], v36, v38 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[138:139], v0 offset:47616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[58:63], v[140:145], a[80:83], v37, v38 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[152:155], v19 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[64:69], v[140:145], a[84:87], v37, v38 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[156:157], v0 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[104:109], v[140:145], a[88:91], v37, v38 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v15, s[36:39], s5 offen offset:1024 lds
	ds_read_b128 v[158:161], v19 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[122:127], v[140:145], a[92:95], v37, v38 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[162:163], v0 offset:48640
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[122:127], v[170:175], a[96:99], v37, v38 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[140:143], v19 offset:34816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[104:109], v[170:175], a[100:103], v37, v38 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[144:145], v0 offset:49152
	s_add_i32 s2, s34, s2
	s_mov_b32 m0, s34
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[64:69], v[170:175], a[104:107], v37, v38 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v16, s[40:43], s2 offen lds
	ds_read_b128 v[164:167], v19 offset:35840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[58:63], v[170:175], a[108:111], v37, v38 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[168:169], v0 offset:49664
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[52:57], v[170:175], a[112:115], v36, v38 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[176:179], v19 offset:36864
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[28:33], v[170:175], a[116:119], v36, v38 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[180:181], v0 offset:50176
	s_add_i32 s2, s34, s4
	s_mov_b32 m0, s52
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[46:51], v[170:175], a[120:123], v36, v38 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v16, s[8:11], s2 offen lds
	ds_read_b128 v[182:185], v19 offset:37888
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[40:45], v[170:175], a[124:127], v36, v38 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[186:187], v0 offset:50688
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[40:45], v[146:151], a[128:131], v36, v39 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[46:51], v[146:151], a[132:135], v36, v39 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[28:33], v[146:151], a[136:139], v36, v39 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[52:57], v[146:151], a[140:143], v36, v39 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[58:63], v[146:151], a[144:147], v37, v39 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[64:69], v[146:151], a[148:151], v37, v39 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[104:109], v[146:151], a[152:155], v37, v39 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[122:127], v[146:151], a[156:159], v37, v39 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[122:127], v[188:193], a[160:163], v37, v39 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[104:109], v[188:193], a[164:167], v37, v39 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[64:69], v[188:193], a[168:171], v37, v39 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[58:63], v[188:193], a[172:175], v37, v39 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[52:57], v[188:193], a[176:179], v36, v39 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[28:33], v[188:193], a[180:183], v36, v39 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[46:51], v[188:193], a[184:187], v36, v39 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[40:45], v[188:193], a[188:191], v36, v39 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[40:45], v[194:199], a[192:195], v36, v39 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[46:51], v[194:199], a[196:199], v36, v39 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[28:33], v[194:199], a[200:203], v36, v39 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[52:57], v[194:199], a[204:207], v36, v39 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[58:63], v[194:199], a[208:211], v37, v39 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[64:69], v[194:199], a[212:215], v37, v39 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[104:109], v[194:199], a[216:219], v37, v39 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[122:127], v[194:199], a[220:223], v37, v39 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[122:127], v[200:205], a[224:227], v37, v39 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[104:109], v[200:205], a[228:231], v37, v39 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[64:69], v[200:205], a[232:235], v37, v39 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[58:63], v[200:205], a[236:239], v37, v39 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[52:57], v[200:205], a[240:243], v36, v39 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[28:33], v[200:205], a[244:247], v36, v39 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[46:51], v[200:205], a[248:251], v36, v39 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[40:45], v[200:205], a[252:255], v36, v39 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:25], v[128:133], a[0:3], v4, v12 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(8) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[14:15], v18 offset:4096
	ds_read_b64 v[26:27], v17 offset:5120
	ds_read_b64 v[28:29], v11 offset:14400
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[70:75], v[128:133], a[4:7], v4, v12 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[30:31], v11 offset:14408
	ds_read_b64 v[32:33], v9 offset:14416
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[76:81], v[128:133], a[8:11], v4, v12 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[34:35], v11 offset:14528
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[82:87], v[128:133], a[12:15], v4, v12 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[36:37], v11 offset:14536
	ds_read_b64 v[38:39], v9 offset:14544
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[92:97], v[128:133], a[16:19], v5, v12 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[40:41], v11 offset:14656
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[98:103], v[128:133], a[20:23], v5, v12 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[42:43], v11 offset:14664
	ds_read_b64 v[44:45], v9 offset:14672
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[110:115], v[128:133], a[24:27], v5, v12 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[46:47], v11 offset:14784
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[116:121], v[128:133], a[28:31], v5, v12 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[48:49], v11 offset:14792
	ds_read_b64 v[50:51], v9 offset:14800
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[116:121], v[134:139], a[32:35], v5, v12 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[52:53], v11 offset:14912
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[110:115], v[134:139], a[36:39], v5, v12 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[54:55], v11 offset:14920
	ds_read_b64 v[56:57], v9 offset:14928
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[98:103], v[134:139], a[40:43], v5, v12 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[58:59], v11 offset:15040
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[92:97], v[134:139], a[44:47], v5, v12 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[60:61], v11 offset:15048
	ds_read_b64 v[62:63], v9 offset:15056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[82:87], v[134:139], a[48:51], v4, v12 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[64:65], v11 offset:15168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[76:81], v[134:139], a[52:55], v4, v12 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[66:67], v11 offset:15176
	ds_read_b64 v[68:69], v9 offset:15184
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[70:75], v[134:139], a[56:59], v4, v12 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[104:105], v11 offset:15296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[20:25], v[134:139], a[60:63], v4, v12 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[106:107], v11 offset:15304
	ds_read_b64 v[108:109], v9 offset:15312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[20:25], v[152:157], a[64:67], v4, v12 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[122:125], v19 offset:55296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[70:75], v[152:157], a[68:71], v4, v12 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[126:127], v90 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[76:81], v[152:157], a[72:75], v4, v12 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[128:131], v19 offset:56320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[82:87], v[152:157], a[76:79], v4, v12 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[132:133], v90 offset:6656
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[92:97], v[152:157], a[80:83], v5, v12 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[134:137], v19 offset:57344
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[98:103], v[152:157], a[84:87], v5, v12 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[138:139], v90 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[110:115], v[152:157], a[88:91], v5, v12 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[146:149], v19 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[116:121], v[152:157], a[92:95], v5, v12 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[150:151], v90 offset:7680
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[116:121], v[158:163], a[96:99], v5, v12 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[152:155], v19 offset:59392
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[110:115], v[158:163], a[100:103], v5, v12 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[156:157], v90 offset:8192
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[98:103], v[158:163], a[104:107], v5, v12 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[170:173], v19 offset:60416
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[92:97], v[158:163], a[108:111], v5, v12 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[174:175], v90 offset:8704
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[82:87], v[158:163], a[112:115], v4, v12 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[188:191], v19 offset:61440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[76:81], v[158:163], a[116:119], v4, v12 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[192:193], v90 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[70:75], v[158:163], a[120:123], v4, v12 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[194:197], v19 offset:62464
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[20:25], v[158:163], a[124:127], v4, v12 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[198:199], v90 offset:9728
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[20:25], v[140:145], a[128:131], v4, v13 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[70:75], v[140:145], a[132:135], v4, v13 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[76:81], v[140:145], a[136:139], v4, v13 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[82:87], v[140:145], a[140:143], v4, v13 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[92:97], v[140:145], a[144:147], v5, v13 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[98:103], v[140:145], a[148:151], v5, v13 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[110:115], v[140:145], a[152:155], v5, v13 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[116:121], v[140:145], a[156:159], v5, v13 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[116:121], v[164:169], a[160:163], v5, v13 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[110:115], v[164:169], a[164:167], v5, v13 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[98:103], v[164:169], a[168:171], v5, v13 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[92:97], v[164:169], a[172:175], v5, v13 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[82:87], v[164:169], a[176:179], v4, v13 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[76:81], v[164:169], a[180:183], v4, v13 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[70:75], v[164:169], a[184:187], v4, v13 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[20:25], v[164:169], a[188:191], v4, v13 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[20:25], v[176:181], a[192:195], v4, v13 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[70:75], v[176:181], a[196:199], v4, v13 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[76:81], v[176:181], a[200:203], v4, v13 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[82:87], v[176:181], a[204:207], v4, v13 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[92:97], v[176:181], a[208:211], v5, v13 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[98:103], v[176:181], a[212:215], v5, v13 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[110:115], v[176:181], a[216:219], v5, v13 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[116:121], v[176:181], a[220:223], v5, v13 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[116:121], v[182:187], a[224:227], v5, v13 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[110:115], v[182:187], a[228:231], v5, v13 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[98:103], v[182:187], a[232:235], v5, v13 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[92:97], v[182:187], a[236:239], v5, v13 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[82:87], v[182:187], a[240:243], v4, v13 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[76:81], v[182:187], a[244:247], v4, v13 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[70:75], v[182:187], a[248:251], v4, v13 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[20:25], v[182:187], a[252:255], v4, v13 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[4:5], v18
	ds_read_b64 v[8:9], v17 offset:1024
	ds_read_b64 v[20:21], v6 offset:47392
	ds_read_b64 v[22:23], v6 offset:47400
	ds_read_b64 v[24:25], v6 offset:47408
	ds_read_b64 v[70:71], v6 offset:47520
	ds_read_b64 v[72:73], v6 offset:47528
	ds_read_b64 v[74:75], v6 offset:47536
	ds_read_b64 v[76:77], v6 offset:47648
	ds_read_b64 v[78:79], v6 offset:47656
	ds_read_b64 v[80:81], v6 offset:47664
	ds_read_b64 v[82:83], v6 offset:47776
	ds_read_b64 v[84:85], v6 offset:47784
	ds_read_b64 v[86:87], v6 offset:47792
	ds_read_b64 v[88:89], v6 offset:47904
	ds_read_b64 v[90:91], v6 offset:47912
	ds_read_b64 v[92:93], v6 offset:47920
	ds_read_b64 v[94:95], v6 offset:48032
	ds_read_b64 v[96:97], v6 offset:48040
	ds_read_b64 v[98:99], v6 offset:48048
	ds_read_b64 v[110:111], v6 offset:48160
	ds_read_b64 v[112:113], v6 offset:48168
	ds_read_b64 v[114:115], v6 offset:48176
	ds_read_b64 v[116:117], v6 offset:48288
	ds_read_b64 v[118:119], v6 offset:48296
	ds_read_b64 v[120:121], v6 offset:48304
	ds_read_b128 v[140:143], v19 offset:6144
	ds_read_b64 v[144:145], v0 offset:22528
	ds_read_b128 v[158:161], v19 offset:7168
	ds_read_b64 v[162:163], v0 offset:23040
	ds_read_b128 v[164:167], v19 offset:8192
	ds_read_b64 v[168:169], v0 offset:23552
	ds_read_b128 v[176:179], v19 offset:9216
	ds_read_b64 v[180:181], v0 offset:24064
	ds_read_b128 v[182:185], v19 offset:10240
	ds_read_b64 v[186:187], v0 offset:24576
	ds_read_b128 v[200:203], v19 offset:11264
	ds_read_b64 v[204:205], v0 offset:25088
	ds_read_b128 v[206:209], v19 offset:12288
	ds_read_b64 v[210:211], v0 offset:25600
	ds_read_b128 v[212:215], v19 offset:13312
	ds_read_b64 v[216:217], v0 offset:26112
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[28:33], v[122:127], a[0:3], v14, v26 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[28:33], v[128:133], a[60:63], v14, v26 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[28:33], v[134:139], a[64:67], v14, v26 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[28:33], v[146:151], a[124:127], v14, v26 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[28:33], v[152:157], a[128:131], v14, v27 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[28:33], v[170:175], a[188:191], v14, v27 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[28:33], v[188:193], a[192:195], v14, v27 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[28:33], v[194:199], a[252:255], v14, v27 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[20:25], v[212:217], a[252:255], v4, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[20:25], v[206:211], a[192:195], v4, v9 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[20:25], v[200:205], a[188:191], v4, v9 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[20:25], v[182:187], a[128:131], v4, v9 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[20:25], v[176:181], a[124:127], v4, v8 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[20:25], v[164:169], a[64:67], v4, v8 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[20:25], v[158:163], a[60:63], v4, v8 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:25], v[140:145], a[0:3], v4, v8 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[34:39], v[194:199], a[248:251], v14, v27 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_and_b32_e32 v0, 1, v1
	v_sub_u32_e32 v3, v1, v0
	v_ashrrev_i32_e32 v6, 31, v3
	v_cmp_eq_i32_e64 s[4:5], v0, 1
	v_lshrrev_b32_e32 v6, 31, v6
	v_lshlrev_b32_e32 v0, 1, v0
	v_add_u32_e32 v3, v3, v6
	v_lshl_add_u32 v0, v2, 2, v0
	v_ashrrev_i32_e32 v3, 1, v3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[34:39], v[188:193], a[196:199], v14, v27 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_nop 0
	v_accvgpr_read_b32 v6, a0
	v_accvgpr_read_b32 v7, a1
	v_cvt_pk_bf16_f32 v6, v6, v7
	v_accvgpr_read_b32 v7, a2
	v_accvgpr_read_b32 v10, a3
	v_cvt_pk_bf16_f32 v7, v7, v10
	s_mul_i32 s6, s3, 0x440
	s_mov_b32 s7, 0x44
	v_mad_u32_u24 v0, s7, v0, v3
	v_cndmask_b32_e64 v3, v7, v6, s[4:5]
	v_mov_b32_e32 v10, 0x1000504
	v_mov_b32_e32 v11, 0x5040100
	v_cndmask_b32_e64 v10, v11, v10, s[4:5]
	v_mov_b32_e32 v11, 0x3020706
	v_mov_b32_e32 v12, 0x7060302
	v_cndmask_b32_e64 v11, v12, v11, s[4:5]
	v_cndmask_b32_e64 v6, v6, v7, s[4:5]
	v_mov_b32_dpp v3, v3 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_lshl_u32 v0, s6, v0, 2
	v_lshlrev_b32_e32 v7, 2, v1
	v_lshlrev_b32_e32 v1, 4, v1
	v_perm_b32 v12, v3, v6, v10
	v_perm_b32 v3, v3, v6, v11
	v_add_u32_e32 v0, 0xd800, v0
	s_and_b32 s1, s1, 0xffff
	s_mov_b32 s2, 0x7ffffff0
	s_mov_b32 s3, 0x31014000
	v_mad_u32_u24 v6, s7, v2, v7
	v_lshl_add_u32 v1, v2, 16, v1
	ds_write2_b32 v0, v12, v3 offset1:68
	v_accvgpr_read_b32 v2, a60
	v_accvgpr_read_b32 v3, a61
	v_cvt_pk_bf16_f32 v2, v2, v3
	v_accvgpr_read_b32 v3, a62
	v_accvgpr_read_b32 v7, a63
	v_cvt_pk_bf16_f32 v3, v3, v7
	v_cndmask_b32_e64 v7, v3, v2, s[4:5]
	v_cndmask_b32_e64 v2, v2, v3, s[4:5]
	s_nop 0
	v_mov_b32_dpp v3, v7 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v7, v3, v2, v10
	v_perm_b32 v2, v3, v2, v11
	ds_write2_b32 v0, v7, v2 offset0:8 offset1:76
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[34:39], v[170:175], a[184:187], v14, v27 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v2, a64
	v_accvgpr_read_b32 v3, a65
	v_cvt_pk_bf16_f32 v2, v2, v3
	v_accvgpr_read_b32 v3, a66
	v_accvgpr_read_b32 v7, a67
	v_cvt_pk_bf16_f32 v3, v3, v7
	v_cndmask_b32_e64 v7, v3, v2, s[4:5]
	v_cndmask_b32_e64 v2, v2, v3, s[4:5]
	s_nop 0
	v_mov_b32_dpp v3, v7 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v7, v3, v2, v10
	v_perm_b32 v2, v3, v2, v11
	ds_write2_b32 v0, v7, v2 offset0:16 offset1:84
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[34:39], v[152:157], a[132:135], v14, v27 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v2, a124
	v_accvgpr_read_b32 v3, a125
	v_cvt_pk_bf16_f32 v2, v2, v3
	v_accvgpr_read_b32 v3, a126
	v_accvgpr_read_b32 v7, a127
	v_cvt_pk_bf16_f32 v3, v3, v7
	v_cndmask_b32_e64 v7, v3, v2, s[4:5]
	v_cndmask_b32_e64 v2, v2, v3, s[4:5]
	s_nop 0
	v_mov_b32_dpp v3, v7 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v7, v3, v2, v10
	v_perm_b32 v2, v3, v2, v11
	ds_write2_b32 v0, v7, v2 offset0:24 offset1:92
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[34:39], v[146:151], a[120:123], v14, v26 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v2, a128
	v_accvgpr_read_b32 v3, a129
	v_cvt_pk_bf16_f32 v2, v2, v3
	v_accvgpr_read_b32 v3, a130
	v_accvgpr_read_b32 v7, a131
	v_cvt_pk_bf16_f32 v3, v3, v7
	v_cndmask_b32_e64 v7, v3, v2, s[4:5]
	v_cndmask_b32_e64 v2, v2, v3, s[4:5]
	s_nop 0
	v_mov_b32_dpp v3, v7 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v7, v3, v2, v10
	v_perm_b32 v2, v3, v2, v11
	ds_write2_b32 v0, v7, v2 offset0:32 offset1:100
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[34:39], v[134:139], a[68:71], v14, v26 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v2, a188
	v_accvgpr_read_b32 v3, a189
	v_cvt_pk_bf16_f32 v2, v2, v3
	v_accvgpr_read_b32 v3, a190
	v_accvgpr_read_b32 v7, a191
	v_cvt_pk_bf16_f32 v3, v3, v7
	v_cndmask_b32_e64 v7, v3, v2, s[4:5]
	v_cndmask_b32_e64 v2, v2, v3, s[4:5]
	s_nop 0
	v_mov_b32_dpp v3, v7 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v7, v3, v2, v10
	v_perm_b32 v2, v3, v2, v11
	ds_write2_b32 v0, v7, v2 offset0:40 offset1:108
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[34:39], v[128:133], a[56:59], v14, v26 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v2, a192
	v_accvgpr_read_b32 v3, a193
	v_cvt_pk_bf16_f32 v2, v2, v3
	v_accvgpr_read_b32 v3, a194
	v_accvgpr_read_b32 v7, a195
	v_cvt_pk_bf16_f32 v3, v3, v7
	v_cndmask_b32_e64 v7, v3, v2, s[4:5]
	v_cndmask_b32_e64 v2, v2, v3, s[4:5]
	s_nop 0
	v_mov_b32_dpp v3, v7 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v7, v3, v2, v10
	v_perm_b32 v2, v3, v2, v11
	ds_write2_b32 v0, v7, v2 offset0:48 offset1:116
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[34:39], v[122:127], a[4:7], v14, v26 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v2, a252
	v_accvgpr_read_b32 v3, a253
	v_cvt_pk_bf16_f32 v2, v2, v3
	v_accvgpr_read_b32 v3, a254
	v_accvgpr_read_b32 v7, a255
	v_cvt_pk_bf16_f32 v3, v3, v7
	v_cndmask_b32_e64 v7, v3, v2, s[4:5]
	v_cndmask_b32_e64 v2, v2, v3, s[4:5]
	s_nop 0
	v_mov_b32_dpp v3, v7 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v7, v3, v2, v10
	v_perm_b32 v2, v3, v2, v11
	ds_write2_b32 v0, v7, v2 offset0:56 offset1:124
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[70:75], v[140:145], a[4:7], v4, v8 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[70:75], v[158:163], a[56:59], v4, v8 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_add_lshl_u32 v2, s6, v6, 2
	ds_read_b128 v[16:19], v2 offset:55296
	ds_read_b128 v[20:23], v2 offset:56384
	ds_read_b128 v[28:31], v2 offset:57472
	ds_read_b128 v[32:35], v2 offset:58560
	s_lshl_b32 s6, s15, 16
	s_lshl_b32 s7, s31, 1
	s_add_i32 s6, s6, s7
	v_add_u32_e32 v3, 0x40000, v1
	v_add_u32_e32 v6, 0x80000, v1
	v_add_u32_e32 v7, 0xc0000, v1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[70:75], v[164:169], a[68:71], v4, v8 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(3)
	buffer_store_dwordx4 v[16:19], v1, s[0:3], s6 offen nt
	s_waitcnt lgkmcnt(2)
	buffer_store_dwordx4 v[20:23], v3, s[0:3], s6 offen nt
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[28:31], v6, s[0:3], s6 offen nt
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[32:35], v7, s[0:3], s6 offen nt
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[70:75], v[176:181], a[120:123], v4, v8 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[70:75], v[182:187], a[132:135], v4, v9 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[70:75], v[200:205], a[184:187], v4, v9 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[70:75], v[206:211], a[196:199], v4, v9 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[70:75], v[212:217], a[248:251], v4, v9 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[40:45], v[122:127], a[8:11], v14, v26 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v12, a4
	v_accvgpr_read_b32 v13, a5
	v_cvt_pk_bf16_f32 v12, v12, v13
	v_accvgpr_read_b32 v13, a6
	v_accvgpr_read_b32 v16, a7
	v_cvt_pk_bf16_f32 v13, v13, v16
	v_cndmask_b32_e64 v16, v13, v12, s[4:5]
	v_cndmask_b32_e64 v12, v12, v13, s[4:5]
	s_nop 0
	v_mov_b32_dpp v13, v16 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v16, v13, v12, v10
	v_perm_b32 v12, v13, v12, v11
	ds_write2_b32 v0, v16, v12 offset1:68
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[40:45], v[128:133], a[52:55], v14, v26 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v12, a56
	v_accvgpr_read_b32 v13, a57
	v_cvt_pk_bf16_f32 v12, v12, v13
	v_accvgpr_read_b32 v13, a58
	v_accvgpr_read_b32 v16, a59
	v_cvt_pk_bf16_f32 v13, v13, v16
	v_cndmask_b32_e64 v16, v13, v12, s[4:5]
	v_cndmask_b32_e64 v12, v12, v13, s[4:5]
	s_nop 0
	v_mov_b32_dpp v13, v16 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v16, v13, v12, v10
	v_perm_b32 v12, v13, v12, v11
	ds_write2_b32 v0, v16, v12 offset0:8 offset1:76
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[40:45], v[134:139], a[72:75], v14, v26 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v12, a68
	v_accvgpr_read_b32 v13, a69
	v_cvt_pk_bf16_f32 v12, v12, v13
	v_accvgpr_read_b32 v13, a70
	v_accvgpr_read_b32 v16, a71
	v_cvt_pk_bf16_f32 v13, v13, v16
	v_cndmask_b32_e64 v16, v13, v12, s[4:5]
	v_cndmask_b32_e64 v12, v12, v13, s[4:5]
	s_nop 0
	v_mov_b32_dpp v13, v16 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v16, v13, v12, v10
	v_perm_b32 v12, v13, v12, v11
	ds_write2_b32 v0, v16, v12 offset0:16 offset1:84
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[40:45], v[146:151], a[116:119], v14, v26 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v12, a120
	v_accvgpr_read_b32 v13, a121
	v_cvt_pk_bf16_f32 v12, v12, v13
	v_accvgpr_read_b32 v13, a122
	v_accvgpr_read_b32 v16, a123
	v_cvt_pk_bf16_f32 v13, v13, v16
	v_cndmask_b32_e64 v16, v13, v12, s[4:5]
	v_cndmask_b32_e64 v12, v12, v13, s[4:5]
	s_nop 0
	v_mov_b32_dpp v13, v16 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v16, v13, v12, v10
	v_perm_b32 v12, v13, v12, v11
	ds_write2_b32 v0, v16, v12 offset0:24 offset1:92
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[40:45], v[152:157], a[136:139], v14, v27 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v12, a132
	v_accvgpr_read_b32 v13, a133
	v_cvt_pk_bf16_f32 v12, v12, v13
	v_accvgpr_read_b32 v13, a134
	v_accvgpr_read_b32 v16, a135
	v_cvt_pk_bf16_f32 v13, v13, v16
	v_cndmask_b32_e64 v16, v13, v12, s[4:5]
	v_cndmask_b32_e64 v12, v12, v13, s[4:5]
	s_nop 0
	v_mov_b32_dpp v13, v16 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v16, v13, v12, v10
	v_perm_b32 v12, v13, v12, v11
	ds_write2_b32 v0, v16, v12 offset0:32 offset1:100
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[40:45], v[170:175], a[180:183], v14, v27 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v12, a184
	v_accvgpr_read_b32 v13, a185
	v_cvt_pk_bf16_f32 v12, v12, v13
	v_accvgpr_read_b32 v13, a186
	v_accvgpr_read_b32 v16, a187
	v_cvt_pk_bf16_f32 v13, v13, v16
	v_cndmask_b32_e64 v16, v13, v12, s[4:5]
	v_cndmask_b32_e64 v12, v12, v13, s[4:5]
	s_nop 0
	v_mov_b32_dpp v13, v16 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v16, v13, v12, v10
	v_perm_b32 v12, v13, v12, v11
	ds_write2_b32 v0, v16, v12 offset0:40 offset1:108
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[40:45], v[188:193], a[200:203], v14, v27 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v12, a196
	v_accvgpr_read_b32 v13, a197
	v_cvt_pk_bf16_f32 v12, v12, v13
	v_accvgpr_read_b32 v13, a198
	v_accvgpr_read_b32 v16, a199
	v_cvt_pk_bf16_f32 v13, v13, v16
	v_cndmask_b32_e64 v16, v13, v12, s[4:5]
	v_cndmask_b32_e64 v12, v12, v13, s[4:5]
	s_nop 0
	v_mov_b32_dpp v13, v16 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v16, v13, v12, v10
	v_perm_b32 v12, v13, v12, v11
	ds_write2_b32 v0, v16, v12 offset0:48 offset1:116
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[40:45], v[194:199], a[244:247], v14, v27 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v12, a248
	v_accvgpr_read_b32 v13, a249
	v_cvt_pk_bf16_f32 v12, v12, v13
	v_accvgpr_read_b32 v13, a250
	v_accvgpr_read_b32 v16, a251
	v_cvt_pk_bf16_f32 v13, v13, v16
	v_cndmask_b32_e64 v16, v13, v12, s[4:5]
	v_cndmask_b32_e64 v12, v12, v13, s[4:5]
	s_nop 0
	v_mov_b32_dpp v13, v16 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v16, v13, v12, v10
	v_perm_b32 v12, v13, v12, v11
	ds_write2_b32 v0, v16, v12 offset0:56 offset1:124
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[76:81], v[212:217], a[244:247], v4, v9 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[76:81], v[206:211], a[200:203], v4, v9 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	ds_read_b128 v[16:19], v2 offset:55296
	ds_read_b128 v[20:23], v2 offset:56384
	ds_read_b128 v[28:31], v2 offset:57472
	ds_read_b128 v[32:35], v2 offset:58560
	s_add_i32 s6, s15, 16
	s_lshl_b32 s6, s6, 16
	s_add_i32 s6, s7, s6
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[76:81], v[200:205], a[180:183], v4, v9 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(3)
	buffer_store_dwordx4 v[16:19], v1, s[0:3], s6 offen nt
	s_waitcnt lgkmcnt(2)
	buffer_store_dwordx4 v[20:23], v3, s[0:3], s6 offen nt
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[28:31], v6, s[0:3], s6 offen nt
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[32:35], v7, s[0:3], s6 offen nt
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[76:81], v[182:187], a[136:139], v4, v9 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[76:81], v[176:181], a[116:119], v4, v8 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[76:81], v[164:169], a[72:75], v4, v8 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[76:81], v[158:163], a[52:55], v4, v8 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[76:81], v[140:145], a[8:11], v4, v8 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[46:51], v[194:199], a[240:243], v14, v27 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_nop 7
	s_nop 2
	v_accvgpr_read_b32 v12, a8
	v_accvgpr_read_b32 v13, a9
	v_cvt_pk_bf16_f32 v12, v12, v13
	v_accvgpr_read_b32 v13, a10
	v_accvgpr_read_b32 v16, a11
	v_cvt_pk_bf16_f32 v13, v13, v16
	v_cndmask_b32_e64 v16, v13, v12, s[4:5]
	v_cndmask_b32_e64 v12, v12, v13, s[4:5]
	s_nop 0
	v_mov_b32_dpp v13, v16 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[46:51], v[188:193], a[204:207], v14, v27 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_perm_b32 v16, v13, v12, v10
	v_perm_b32 v12, v13, v12, v11
	ds_write2_b32 v0, v16, v12 offset1:68
	v_accvgpr_read_b32 v12, a52
	v_accvgpr_read_b32 v13, a53
	v_cvt_pk_bf16_f32 v12, v12, v13
	v_accvgpr_read_b32 v13, a54
	v_accvgpr_read_b32 v16, a55
	v_cvt_pk_bf16_f32 v13, v13, v16
	v_cndmask_b32_e64 v16, v13, v12, s[4:5]
	v_cndmask_b32_e64 v12, v12, v13, s[4:5]
	s_nop 0
	v_mov_b32_dpp v13, v16 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v16, v13, v12, v10
	v_perm_b32 v12, v13, v12, v11
	ds_write2_b32 v0, v16, v12 offset0:8 offset1:76
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[46:51], v[170:175], a[176:179], v14, v27 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v12, a72
	v_accvgpr_read_b32 v13, a73
	v_cvt_pk_bf16_f32 v12, v12, v13
	v_accvgpr_read_b32 v13, a74
	v_accvgpr_read_b32 v16, a75
	v_cvt_pk_bf16_f32 v13, v13, v16
	v_cndmask_b32_e64 v16, v13, v12, s[4:5]
	v_cndmask_b32_e64 v12, v12, v13, s[4:5]
	s_nop 0
	v_mov_b32_dpp v13, v16 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v16, v13, v12, v10
	v_perm_b32 v12, v13, v12, v11
	ds_write2_b32 v0, v16, v12 offset0:16 offset1:84
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[46:51], v[152:157], a[140:143], v14, v27 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v12, a116
	v_accvgpr_read_b32 v13, a117
	v_cvt_pk_bf16_f32 v12, v12, v13
	v_accvgpr_read_b32 v13, a118
	v_accvgpr_read_b32 v16, a119
	v_cvt_pk_bf16_f32 v13, v13, v16
	v_cndmask_b32_e64 v16, v13, v12, s[4:5]
	v_cndmask_b32_e64 v12, v12, v13, s[4:5]
	s_nop 0
	v_mov_b32_dpp v13, v16 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v16, v13, v12, v10
	v_perm_b32 v12, v13, v12, v11
	ds_write2_b32 v0, v16, v12 offset0:24 offset1:92
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[46:51], v[146:151], a[112:115], v14, v26 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v12, a136
	v_accvgpr_read_b32 v13, a137
	v_cvt_pk_bf16_f32 v12, v12, v13
	v_accvgpr_read_b32 v13, a138
	v_accvgpr_read_b32 v16, a139
	v_cvt_pk_bf16_f32 v13, v13, v16
	v_cndmask_b32_e64 v16, v13, v12, s[4:5]
	v_cndmask_b32_e64 v12, v12, v13, s[4:5]
	s_nop 0
	v_mov_b32_dpp v13, v16 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v16, v13, v12, v10
	v_perm_b32 v12, v13, v12, v11
	ds_write2_b32 v0, v16, v12 offset0:32 offset1:100
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[46:51], v[134:139], a[76:79], v14, v26 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v12, a180
	v_accvgpr_read_b32 v13, a181
	v_cvt_pk_bf16_f32 v12, v12, v13
	v_accvgpr_read_b32 v13, a182
	v_accvgpr_read_b32 v16, a183
	v_cvt_pk_bf16_f32 v13, v13, v16
	v_cndmask_b32_e64 v16, v13, v12, s[4:5]
	v_cndmask_b32_e64 v12, v12, v13, s[4:5]
	s_nop 0
	v_mov_b32_dpp v13, v16 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v16, v13, v12, v10
	v_perm_b32 v12, v13, v12, v11
	ds_write2_b32 v0, v16, v12 offset0:40 offset1:108
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[46:51], v[128:133], a[48:51], v14, v26 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v12, a200
	v_accvgpr_read_b32 v13, a201
	v_cvt_pk_bf16_f32 v12, v12, v13
	v_accvgpr_read_b32 v13, a202
	v_accvgpr_read_b32 v16, a203
	v_cvt_pk_bf16_f32 v13, v13, v16
	v_cndmask_b32_e64 v16, v13, v12, s[4:5]
	v_cndmask_b32_e64 v12, v12, v13, s[4:5]
	s_nop 0
	v_mov_b32_dpp v13, v16 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v16, v13, v12, v10
	v_perm_b32 v12, v13, v12, v11
	ds_write2_b32 v0, v16, v12 offset0:48 offset1:116
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[46:51], v[122:127], a[12:15], v14, v26 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v12, a244
	v_accvgpr_read_b32 v13, a245
	v_cvt_pk_bf16_f32 v12, v12, v13
	v_accvgpr_read_b32 v13, a246
	v_accvgpr_read_b32 v14, a247
	v_cvt_pk_bf16_f32 v13, v13, v14
	v_cndmask_b32_e64 v14, v13, v12, s[4:5]
	v_cndmask_b32_e64 v12, v12, v13, s[4:5]
	s_nop 0
	v_mov_b32_dpp v13, v14 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v14, v13, v12, v10
	v_perm_b32 v12, v13, v12, v11
	ds_write2_b32 v0, v14, v12 offset0:56 offset1:124
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[82:87], v[140:145], a[12:15], v4, v8 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[82:87], v[158:163], a[48:51], v4, v8 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	ds_read_b128 v[16:19], v2 offset:55296
	ds_read_b128 v[20:23], v2 offset:56384
	ds_read_b128 v[28:31], v2 offset:57472
	ds_read_b128 v[32:35], v2 offset:58560
	s_add_i32 s6, s15, 32
	s_lshl_b32 s6, s6, 16
	s_add_i32 s6, s7, s6
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[82:87], v[164:169], a[76:79], v4, v8 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(3)
	buffer_store_dwordx4 v[16:19], v1, s[0:3], s6 offen nt
	s_waitcnt lgkmcnt(2)
	buffer_store_dwordx4 v[20:23], v3, s[0:3], s6 offen nt
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[28:31], v6, s[0:3], s6 offen nt
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[32:35], v7, s[0:3], s6 offen nt
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[82:87], v[176:181], a[112:115], v4, v8 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[82:87], v[182:187], a[140:143], v4, v9 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[82:87], v[200:205], a[176:179], v4, v9 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[82:87], v[206:211], a[204:207], v4, v9 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[82:87], v[212:217], a[240:243], v4, v9 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[52:57], v[122:127], a[16:19], v15, v26 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a12
	v_accvgpr_read_b32 v12, a13
	v_cvt_pk_bf16_f32 v12, v4, v12
	v_accvgpr_read_b32 v4, a14
	v_accvgpr_read_b32 v13, a15
	v_cvt_pk_bf16_f32 v13, v4, v13
	v_cndmask_b32_e64 v4, v13, v12, s[4:5]
	v_cndmask_b32_e64 v12, v12, v13, s[4:5]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v13, v4, v12, v10
	v_perm_b32 v4, v4, v12, v11
	ds_write2_b32 v0, v13, v4 offset1:68
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[52:57], v[128:133], a[44:47], v15, v26 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a48
	v_accvgpr_read_b32 v12, a49
	v_cvt_pk_bf16_f32 v12, v4, v12
	v_accvgpr_read_b32 v4, a50
	v_accvgpr_read_b32 v13, a51
	v_cvt_pk_bf16_f32 v13, v4, v13
	v_cndmask_b32_e64 v4, v13, v12, s[4:5]
	v_cndmask_b32_e64 v12, v12, v13, s[4:5]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v13, v4, v12, v10
	v_perm_b32 v4, v4, v12, v11
	ds_write2_b32 v0, v13, v4 offset0:8 offset1:76
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[52:57], v[134:139], a[80:83], v15, v26 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a76
	v_accvgpr_read_b32 v12, a77
	v_cvt_pk_bf16_f32 v12, v4, v12
	v_accvgpr_read_b32 v4, a78
	v_accvgpr_read_b32 v13, a79
	v_cvt_pk_bf16_f32 v13, v4, v13
	v_cndmask_b32_e64 v4, v13, v12, s[4:5]
	v_cndmask_b32_e64 v12, v12, v13, s[4:5]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v13, v4, v12, v10
	v_perm_b32 v4, v4, v12, v11
	ds_write2_b32 v0, v13, v4 offset0:16 offset1:84
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[52:57], v[146:151], a[108:111], v15, v26 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a112
	v_accvgpr_read_b32 v12, a113
	v_cvt_pk_bf16_f32 v12, v4, v12
	v_accvgpr_read_b32 v4, a114
	v_accvgpr_read_b32 v13, a115
	v_cvt_pk_bf16_f32 v13, v4, v13
	v_cndmask_b32_e64 v4, v13, v12, s[4:5]
	v_cndmask_b32_e64 v12, v12, v13, s[4:5]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v13, v4, v12, v10
	v_perm_b32 v4, v4, v12, v11
	ds_write2_b32 v0, v13, v4 offset0:24 offset1:92
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[52:57], v[152:157], a[144:147], v15, v27 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a140
	v_accvgpr_read_b32 v12, a141
	v_cvt_pk_bf16_f32 v12, v4, v12
	v_accvgpr_read_b32 v4, a142
	v_accvgpr_read_b32 v13, a143
	v_cvt_pk_bf16_f32 v13, v4, v13
	v_cndmask_b32_e64 v4, v13, v12, s[4:5]
	v_cndmask_b32_e64 v12, v12, v13, s[4:5]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v13, v4, v12, v10
	v_perm_b32 v4, v4, v12, v11
	ds_write2_b32 v0, v13, v4 offset0:32 offset1:100
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[52:57], v[170:175], a[172:175], v15, v27 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a176
	v_accvgpr_read_b32 v12, a177
	v_cvt_pk_bf16_f32 v12, v4, v12
	v_accvgpr_read_b32 v4, a178
	v_accvgpr_read_b32 v13, a179
	v_cvt_pk_bf16_f32 v13, v4, v13
	v_cndmask_b32_e64 v4, v13, v12, s[4:5]
	v_cndmask_b32_e64 v12, v12, v13, s[4:5]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v13, v4, v12, v10
	v_perm_b32 v4, v4, v12, v11
	ds_write2_b32 v0, v13, v4 offset0:40 offset1:108
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[52:57], v[188:193], a[208:211], v15, v27 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a204
	v_accvgpr_read_b32 v12, a205
	v_cvt_pk_bf16_f32 v12, v4, v12
	v_accvgpr_read_b32 v4, a206
	v_accvgpr_read_b32 v13, a207
	v_cvt_pk_bf16_f32 v13, v4, v13
	v_cndmask_b32_e64 v4, v13, v12, s[4:5]
	v_cndmask_b32_e64 v12, v12, v13, s[4:5]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v13, v4, v12, v10
	v_perm_b32 v4, v4, v12, v11
	ds_write2_b32 v0, v13, v4 offset0:48 offset1:116
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[52:57], v[194:199], a[236:239], v15, v27 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a240
	v_accvgpr_read_b32 v12, a241
	v_cvt_pk_bf16_f32 v12, v4, v12
	v_accvgpr_read_b32 v4, a242
	v_accvgpr_read_b32 v13, a243
	v_cvt_pk_bf16_f32 v13, v4, v13
	v_cndmask_b32_e64 v4, v13, v12, s[4:5]
	v_cndmask_b32_e64 v12, v12, v13, s[4:5]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v13, v4, v12, v10
	v_perm_b32 v4, v4, v12, v11
	ds_write2_b32 v0, v13, v4 offset0:56 offset1:124
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[88:93], v[212:217], a[236:239], v5, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[88:93], v[206:211], a[208:211], v5, v9 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	ds_read_b128 v[16:19], v2 offset:55296
	ds_read_b128 v[20:23], v2 offset:56384
	ds_read_b128 v[28:31], v2 offset:57472
	ds_read_b128 v[32:35], v2 offset:58560
	s_add_i32 s6, s15, 48
	s_lshl_b32 s6, s6, 16
	s_add_i32 s6, s7, s6
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[88:93], v[200:205], a[172:175], v5, v9 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(3)
	buffer_store_dwordx4 v[16:19], v1, s[0:3], s6 offen nt
	s_waitcnt lgkmcnt(2)
	buffer_store_dwordx4 v[20:23], v3, s[0:3], s6 offen nt
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[28:31], v6, s[0:3], s6 offen nt
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[32:35], v7, s[0:3], s6 offen nt
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[88:93], v[182:187], a[144:147], v5, v9 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[88:93], v[176:181], a[108:111], v5, v8 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[88:93], v[164:169], a[80:83], v5, v8 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[88:93], v[158:163], a[44:47], v5, v8 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[88:93], v[140:145], a[16:19], v5, v8 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[58:63], v[194:199], a[232:235], v15, v27 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_nop 7
	s_nop 2
	v_accvgpr_read_b32 v4, a16
	v_accvgpr_read_b32 v12, a17
	v_cvt_pk_bf16_f32 v12, v4, v12
	v_accvgpr_read_b32 v4, a18
	v_accvgpr_read_b32 v13, a19
	v_cvt_pk_bf16_f32 v13, v4, v13
	v_cndmask_b32_e64 v4, v13, v12, s[4:5]
	v_cndmask_b32_e64 v12, v12, v13, s[4:5]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[58:63], v[188:193], a[212:215], v15, v27 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_perm_b32 v13, v4, v12, v10
	v_perm_b32 v4, v4, v12, v11
	ds_write2_b32 v0, v13, v4 offset1:68
	v_accvgpr_read_b32 v4, a44
	v_accvgpr_read_b32 v12, a45
	v_cvt_pk_bf16_f32 v12, v4, v12
	v_accvgpr_read_b32 v4, a46
	v_accvgpr_read_b32 v13, a47
	v_cvt_pk_bf16_f32 v13, v4, v13
	v_cndmask_b32_e64 v4, v13, v12, s[4:5]
	v_cndmask_b32_e64 v12, v12, v13, s[4:5]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v13, v4, v12, v10
	v_perm_b32 v4, v4, v12, v11
	ds_write2_b32 v0, v13, v4 offset0:8 offset1:76
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[58:63], v[170:175], a[168:171], v15, v27 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a80
	v_accvgpr_read_b32 v12, a81
	v_cvt_pk_bf16_f32 v12, v4, v12
	v_accvgpr_read_b32 v4, a82
	v_accvgpr_read_b32 v13, a83
	v_cvt_pk_bf16_f32 v13, v4, v13
	v_cndmask_b32_e64 v4, v13, v12, s[4:5]
	v_cndmask_b32_e64 v12, v12, v13, s[4:5]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v13, v4, v12, v10
	v_perm_b32 v4, v4, v12, v11
	ds_write2_b32 v0, v13, v4 offset0:16 offset1:84
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[58:63], v[152:157], a[148:151], v15, v27 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a108
	v_accvgpr_read_b32 v12, a109
	v_cvt_pk_bf16_f32 v12, v4, v12
	v_accvgpr_read_b32 v4, a110
	v_accvgpr_read_b32 v13, a111
	v_cvt_pk_bf16_f32 v13, v4, v13
	v_cndmask_b32_e64 v4, v13, v12, s[4:5]
	v_cndmask_b32_e64 v12, v12, v13, s[4:5]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v13, v4, v12, v10
	v_perm_b32 v4, v4, v12, v11
	ds_write2_b32 v0, v13, v4 offset0:24 offset1:92
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[58:63], v[146:151], a[104:107], v15, v26 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a144
	v_accvgpr_read_b32 v12, a145
	v_cvt_pk_bf16_f32 v12, v4, v12
	v_accvgpr_read_b32 v4, a146
	v_accvgpr_read_b32 v13, a147
	v_cvt_pk_bf16_f32 v13, v4, v13
	v_cndmask_b32_e64 v4, v13, v12, s[4:5]
	v_cndmask_b32_e64 v12, v12, v13, s[4:5]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v13, v4, v12, v10
	v_perm_b32 v4, v4, v12, v11
	ds_write2_b32 v0, v13, v4 offset0:32 offset1:100
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[58:63], v[134:139], a[84:87], v15, v26 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a172
	v_accvgpr_read_b32 v12, a173
	v_cvt_pk_bf16_f32 v12, v4, v12
	v_accvgpr_read_b32 v4, a174
	v_accvgpr_read_b32 v13, a175
	v_cvt_pk_bf16_f32 v13, v4, v13
	v_cndmask_b32_e64 v4, v13, v12, s[4:5]
	v_cndmask_b32_e64 v12, v12, v13, s[4:5]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v13, v4, v12, v10
	v_perm_b32 v4, v4, v12, v11
	ds_write2_b32 v0, v13, v4 offset0:40 offset1:108
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[58:63], v[128:133], a[40:43], v15, v26 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a208
	v_accvgpr_read_b32 v12, a209
	v_cvt_pk_bf16_f32 v12, v4, v12
	v_accvgpr_read_b32 v4, a210
	v_accvgpr_read_b32 v13, a211
	v_cvt_pk_bf16_f32 v13, v4, v13
	v_cndmask_b32_e64 v4, v13, v12, s[4:5]
	v_cndmask_b32_e64 v12, v12, v13, s[4:5]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v13, v4, v12, v10
	v_perm_b32 v4, v4, v12, v11
	ds_write2_b32 v0, v13, v4 offset0:48 offset1:116
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[58:63], v[122:127], a[20:23], v15, v26 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a236
	v_accvgpr_read_b32 v12, a237
	v_cvt_pk_bf16_f32 v12, v4, v12
	v_accvgpr_read_b32 v4, a238
	v_accvgpr_read_b32 v13, a239
	v_cvt_pk_bf16_f32 v13, v4, v13
	v_cndmask_b32_e64 v4, v13, v12, s[4:5]
	v_cndmask_b32_e64 v12, v12, v13, s[4:5]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v13, v4, v12, v10
	v_perm_b32 v4, v4, v12, v11
	ds_write2_b32 v0, v13, v4 offset0:56 offset1:124
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[94:99], v[140:145], a[20:23], v5, v8 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[94:99], v[158:163], a[40:43], v5, v8 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	ds_read_b128 v[16:19], v2 offset:55296
	ds_read_b128 v[20:23], v2 offset:56384
	ds_read_b128 v[28:31], v2 offset:57472
	ds_read_b128 v[32:35], v2 offset:58560
	s_add_i32 s6, s15, 64
	s_lshl_b32 s6, s6, 16
	s_add_i32 s6, s7, s6
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[94:99], v[164:169], a[84:87], v5, v8 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(3)
	buffer_store_dwordx4 v[16:19], v1, s[0:3], s6 offen nt
	s_waitcnt lgkmcnt(2)
	buffer_store_dwordx4 v[20:23], v3, s[0:3], s6 offen nt
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[28:31], v6, s[0:3], s6 offen nt
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[32:35], v7, s[0:3], s6 offen nt
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[94:99], v[176:181], a[104:107], v5, v8 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[94:99], v[182:187], a[148:151], v5, v9 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[94:99], v[200:205], a[168:171], v5, v9 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[94:99], v[206:211], a[212:215], v5, v9 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[94:99], v[212:217], a[232:235], v5, v9 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[64:69], v[122:127], a[24:27], v15, v26 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a20
	v_accvgpr_read_b32 v12, a21
	v_cvt_pk_bf16_f32 v12, v4, v12
	v_accvgpr_read_b32 v4, a22
	v_accvgpr_read_b32 v13, a23
	v_cvt_pk_bf16_f32 v13, v4, v13
	v_cndmask_b32_e64 v4, v13, v12, s[4:5]
	v_cndmask_b32_e64 v12, v12, v13, s[4:5]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v13, v4, v12, v10
	v_perm_b32 v4, v4, v12, v11
	ds_write2_b32 v0, v13, v4 offset1:68
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[64:69], v[128:133], a[36:39], v15, v26 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a40
	v_accvgpr_read_b32 v12, a41
	v_cvt_pk_bf16_f32 v12, v4, v12
	v_accvgpr_read_b32 v4, a42
	v_accvgpr_read_b32 v13, a43
	v_cvt_pk_bf16_f32 v13, v4, v13
	v_cndmask_b32_e64 v4, v13, v12, s[4:5]
	v_cndmask_b32_e64 v12, v12, v13, s[4:5]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v13, v4, v12, v10
	v_perm_b32 v4, v4, v12, v11
	ds_write2_b32 v0, v13, v4 offset0:8 offset1:76
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[64:69], v[134:139], a[88:91], v15, v26 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a84
	v_accvgpr_read_b32 v12, a85
	v_cvt_pk_bf16_f32 v12, v4, v12
	v_accvgpr_read_b32 v4, a86
	v_accvgpr_read_b32 v13, a87
	v_cvt_pk_bf16_f32 v13, v4, v13
	v_cndmask_b32_e64 v4, v13, v12, s[4:5]
	v_cndmask_b32_e64 v12, v12, v13, s[4:5]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v13, v4, v12, v10
	v_perm_b32 v4, v4, v12, v11
	ds_write2_b32 v0, v13, v4 offset0:16 offset1:84
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[64:69], v[146:151], a[100:103], v15, v26 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a104
	v_accvgpr_read_b32 v12, a105
	v_cvt_pk_bf16_f32 v12, v4, v12
	v_accvgpr_read_b32 v4, a106
	v_accvgpr_read_b32 v13, a107
	v_cvt_pk_bf16_f32 v13, v4, v13
	v_cndmask_b32_e64 v4, v13, v12, s[4:5]
	v_cndmask_b32_e64 v12, v12, v13, s[4:5]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v13, v4, v12, v10
	v_perm_b32 v4, v4, v12, v11
	ds_write2_b32 v0, v13, v4 offset0:24 offset1:92
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[64:69], v[152:157], a[152:155], v15, v27 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a148
	v_accvgpr_read_b32 v12, a149
	v_cvt_pk_bf16_f32 v12, v4, v12
	v_accvgpr_read_b32 v4, a150
	v_accvgpr_read_b32 v13, a151
	v_cvt_pk_bf16_f32 v13, v4, v13
	v_cndmask_b32_e64 v4, v13, v12, s[4:5]
	v_cndmask_b32_e64 v12, v12, v13, s[4:5]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v13, v4, v12, v10
	v_perm_b32 v4, v4, v12, v11
	ds_write2_b32 v0, v13, v4 offset0:32 offset1:100
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[64:69], v[170:175], a[164:167], v15, v27 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a168
	v_accvgpr_read_b32 v12, a169
	v_cvt_pk_bf16_f32 v12, v4, v12
	v_accvgpr_read_b32 v4, a170
	v_accvgpr_read_b32 v13, a171
	v_cvt_pk_bf16_f32 v13, v4, v13
	v_cndmask_b32_e64 v4, v13, v12, s[4:5]
	v_cndmask_b32_e64 v12, v12, v13, s[4:5]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v13, v4, v12, v10
	v_perm_b32 v4, v4, v12, v11
	ds_write2_b32 v0, v13, v4 offset0:40 offset1:108
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[64:69], v[188:193], a[216:219], v15, v27 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a212
	v_accvgpr_read_b32 v12, a213
	v_cvt_pk_bf16_f32 v12, v4, v12
	v_accvgpr_read_b32 v4, a214
	v_accvgpr_read_b32 v13, a215
	v_cvt_pk_bf16_f32 v13, v4, v13
	v_cndmask_b32_e64 v4, v13, v12, s[4:5]
	v_cndmask_b32_e64 v12, v12, v13, s[4:5]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v13, v4, v12, v10
	v_perm_b32 v4, v4, v12, v11
	ds_write2_b32 v0, v13, v4 offset0:48 offset1:116
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[64:69], v[194:199], a[228:231], v15, v27 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a232
	v_accvgpr_read_b32 v12, a233
	v_cvt_pk_bf16_f32 v12, v4, v12
	v_accvgpr_read_b32 v4, a234
	v_accvgpr_read_b32 v13, a235
	v_cvt_pk_bf16_f32 v13, v4, v13
	v_cndmask_b32_e64 v4, v13, v12, s[4:5]
	v_cndmask_b32_e64 v12, v12, v13, s[4:5]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v13, v4, v12, v10
	v_perm_b32 v4, v4, v12, v11
	ds_write2_b32 v0, v13, v4 offset0:56 offset1:124
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[110:115], v[212:217], a[228:231], v5, v9 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[110:115], v[206:211], a[216:219], v5, v9 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	ds_read_b128 v[16:19], v2 offset:55296
	ds_read_b128 v[20:23], v2 offset:56384
	ds_read_b128 v[28:31], v2 offset:57472
	ds_read_b128 v[32:35], v2 offset:58560
	s_add_i32 s6, s15, 0x50
	s_lshl_b32 s6, s6, 16
	s_add_i32 s6, s7, s6
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[110:115], v[200:205], a[164:167], v5, v9 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(3)
	buffer_store_dwordx4 v[16:19], v1, s[0:3], s6 offen nt
	s_waitcnt lgkmcnt(2)
	buffer_store_dwordx4 v[20:23], v3, s[0:3], s6 offen nt
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[28:31], v6, s[0:3], s6 offen nt
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[32:35], v7, s[0:3], s6 offen nt
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[110:115], v[182:187], a[152:155], v5, v9 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[110:115], v[176:181], a[100:103], v5, v8 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[110:115], v[164:169], a[88:91], v5, v8 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[110:115], v[158:163], a[36:39], v5, v8 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[110:115], v[140:145], a[24:27], v5, v8 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[104:109], v[194:199], a[224:227], v15, v27 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_nop 7
	s_nop 2
	v_accvgpr_read_b32 v4, a24
	v_accvgpr_read_b32 v12, a25
	v_cvt_pk_bf16_f32 v12, v4, v12
	v_accvgpr_read_b32 v4, a26
	v_accvgpr_read_b32 v13, a27
	v_cvt_pk_bf16_f32 v13, v4, v13
	v_cndmask_b32_e64 v4, v13, v12, s[4:5]
	v_cndmask_b32_e64 v12, v12, v13, s[4:5]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[104:109], v[188:193], a[220:223], v15, v27 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_perm_b32 v13, v4, v12, v10
	v_perm_b32 v4, v4, v12, v11
	ds_write2_b32 v0, v13, v4 offset1:68
	v_accvgpr_read_b32 v4, a36
	v_accvgpr_read_b32 v12, a37
	v_cvt_pk_bf16_f32 v12, v4, v12
	v_accvgpr_read_b32 v4, a38
	v_accvgpr_read_b32 v13, a39
	v_cvt_pk_bf16_f32 v13, v4, v13
	v_cndmask_b32_e64 v4, v13, v12, s[4:5]
	v_cndmask_b32_e64 v12, v12, v13, s[4:5]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v13, v4, v12, v10
	v_perm_b32 v4, v4, v12, v11
	ds_write2_b32 v0, v13, v4 offset0:8 offset1:76
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[104:109], v[170:175], a[160:163], v15, v27 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a88
	v_accvgpr_read_b32 v12, a89
	v_cvt_pk_bf16_f32 v12, v4, v12
	v_accvgpr_read_b32 v4, a90
	v_accvgpr_read_b32 v13, a91
	v_cvt_pk_bf16_f32 v13, v4, v13
	v_cndmask_b32_e64 v4, v13, v12, s[4:5]
	v_cndmask_b32_e64 v12, v12, v13, s[4:5]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v13, v4, v12, v10
	v_perm_b32 v4, v4, v12, v11
	ds_write2_b32 v0, v13, v4 offset0:16 offset1:84
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[104:109], v[152:157], a[156:159], v15, v27 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a100
	v_accvgpr_read_b32 v12, a101
	v_cvt_pk_bf16_f32 v12, v4, v12
	v_accvgpr_read_b32 v4, a102
	v_accvgpr_read_b32 v13, a103
	v_cvt_pk_bf16_f32 v13, v4, v13
	v_cndmask_b32_e64 v4, v13, v12, s[4:5]
	v_cndmask_b32_e64 v12, v12, v13, s[4:5]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v13, v4, v12, v10
	v_perm_b32 v4, v4, v12, v11
	ds_write2_b32 v0, v13, v4 offset0:24 offset1:92
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[104:109], v[146:151], a[96:99], v15, v26 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a152
	v_accvgpr_read_b32 v12, a153
	v_cvt_pk_bf16_f32 v12, v4, v12
	v_accvgpr_read_b32 v4, a154
	v_accvgpr_read_b32 v13, a155
	v_cvt_pk_bf16_f32 v13, v4, v13
	v_cndmask_b32_e64 v4, v13, v12, s[4:5]
	v_cndmask_b32_e64 v12, v12, v13, s[4:5]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v13, v4, v12, v10
	v_perm_b32 v4, v4, v12, v11
	ds_write2_b32 v0, v13, v4 offset0:32 offset1:100
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[104:109], v[134:139], a[92:95], v15, v26 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a164
	v_accvgpr_read_b32 v12, a165
	v_cvt_pk_bf16_f32 v12, v4, v12
	v_accvgpr_read_b32 v4, a166
	v_accvgpr_read_b32 v13, a167
	v_cvt_pk_bf16_f32 v13, v4, v13
	v_cndmask_b32_e64 v4, v13, v12, s[4:5]
	v_cndmask_b32_e64 v12, v12, v13, s[4:5]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v13, v4, v12, v10
	v_perm_b32 v4, v4, v12, v11
	ds_write2_b32 v0, v13, v4 offset0:40 offset1:108
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[104:109], v[128:133], a[32:35], v15, v26 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a216
	v_accvgpr_read_b32 v12, a217
	v_cvt_pk_bf16_f32 v12, v4, v12
	v_accvgpr_read_b32 v4, a218
	v_accvgpr_read_b32 v13, a219
	v_cvt_pk_bf16_f32 v13, v4, v13
	v_cndmask_b32_e64 v4, v13, v12, s[4:5]
	v_cndmask_b32_e64 v12, v12, v13, s[4:5]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v13, v4, v12, v10
	v_perm_b32 v4, v4, v12, v11
	ds_write2_b32 v0, v13, v4 offset0:48 offset1:116
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[104:109], v[122:127], a[28:31], v15, v26 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a228
	v_accvgpr_read_b32 v12, a229
	v_cvt_pk_bf16_f32 v12, v4, v12
	v_accvgpr_read_b32 v4, a230
	v_accvgpr_read_b32 v13, a231
	v_cvt_pk_bf16_f32 v13, v4, v13
	v_cndmask_b32_e64 v4, v13, v12, s[4:5]
	v_cndmask_b32_e64 v12, v12, v13, s[4:5]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v13, v4, v12, v10
	v_perm_b32 v4, v4, v12, v11
	ds_write2_b32 v0, v13, v4 offset0:56 offset1:124
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[116:121], v[140:145], a[28:31], v5, v8 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[116:121], v[158:163], a[32:35], v5, v8 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	ds_read_b128 v[12:15], v2 offset:55296
	ds_read_b128 v[16:19], v2 offset:56384
	ds_read_b128 v[20:23], v2 offset:57472
	ds_read_b128 v[24:27], v2 offset:58560
	s_add_i32 s6, s15, 0x60
	s_lshl_b32 s6, s6, 16
	s_add_i32 s6, s7, s6
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[116:121], v[164:169], a[92:95], v5, v8 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(3)
	buffer_store_dwordx4 v[12:15], v1, s[0:3], s6 offen nt
	s_waitcnt lgkmcnt(2)
	buffer_store_dwordx4 v[16:19], v3, s[0:3], s6 offen nt
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[20:23], v6, s[0:3], s6 offen nt
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[24:27], v7, s[0:3], s6 offen nt
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[116:121], v[176:181], a[96:99], v5, v8 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[116:121], v[182:187], a[156:159], v5, v9 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[116:121], v[200:205], a[160:163], v5, v9 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[116:121], v[206:211], a[220:223], v5, v9 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[116:121], v[212:217], a[224:227], v5, v9 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a28
	v_accvgpr_read_b32 v5, a29
	v_cvt_pk_bf16_f32 v4, v4, v5
	v_accvgpr_read_b32 v5, a30
	v_accvgpr_read_b32 v8, a31
	v_cvt_pk_bf16_f32 v5, v5, v8
	v_accvgpr_read_b32 v8, a32
	v_accvgpr_read_b32 v9, a33
	v_cvt_pk_bf16_f32 v8, v8, v9
	v_accvgpr_read_b32 v9, a34
	v_accvgpr_read_b32 v12, a35
	v_cvt_pk_bf16_f32 v9, v9, v12
	v_accvgpr_read_b32 v12, a92
	v_accvgpr_read_b32 v13, a93
	v_cvt_pk_bf16_f32 v12, v12, v13
	v_accvgpr_read_b32 v13, a94
	v_accvgpr_read_b32 v14, a95
	v_cvt_pk_bf16_f32 v13, v13, v14
	v_accvgpr_read_b32 v14, a96
	v_accvgpr_read_b32 v15, a97
	v_cvt_pk_bf16_f32 v14, v14, v15
	v_accvgpr_read_b32 v15, a98
	v_accvgpr_read_b32 v16, a99
	v_cvt_pk_bf16_f32 v15, v15, v16
	v_accvgpr_read_b32 v16, a156
	v_accvgpr_read_b32 v17, a157
	v_cvt_pk_bf16_f32 v16, v16, v17
	v_accvgpr_read_b32 v17, a158
	v_accvgpr_read_b32 v18, a159
	v_cvt_pk_bf16_f32 v17, v17, v18
	v_accvgpr_read_b32 v18, a160
	v_accvgpr_read_b32 v19, a161
	v_cvt_pk_bf16_f32 v18, v18, v19
	v_accvgpr_read_b32 v19, a162
	v_accvgpr_read_b32 v20, a163
	v_cvt_pk_bf16_f32 v19, v19, v20
	v_accvgpr_read_b32 v20, a220
	v_accvgpr_read_b32 v21, a221
	v_cvt_pk_bf16_f32 v20, v20, v21
	v_accvgpr_read_b32 v21, a222
	v_accvgpr_read_b32 v22, a223
	v_cvt_pk_bf16_f32 v21, v21, v22
	v_accvgpr_read_b32 v22, a224
	v_accvgpr_read_b32 v23, a225
	v_cvt_pk_bf16_f32 v22, v22, v23
	v_accvgpr_read_b32 v23, a226
	v_accvgpr_read_b32 v24, a227
	v_cvt_pk_bf16_f32 v23, v23, v24
	v_cndmask_b32_e64 v24, v5, v4, s[4:5]
	v_cndmask_b32_e64 v25, v9, v8, s[4:5]
	v_cndmask_b32_e64 v26, v13, v12, s[4:5]
	v_cndmask_b32_e64 v27, v15, v14, s[4:5]
	v_cndmask_b32_e64 v28, v17, v16, s[4:5]
	v_cndmask_b32_e64 v29, v19, v18, s[4:5]
	v_cndmask_b32_e64 v30, v21, v20, s[4:5]
	v_cndmask_b32_e64 v31, v23, v22, s[4:5]
	v_cndmask_b32_e64 v4, v4, v5, s[4:5]
	v_mov_b32_dpp v5, v24 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v9, s[4:5]
	v_mov_b32_dpp v9, v25 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v12, v12, v13, s[4:5]
	v_mov_b32_dpp v13, v26 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v14, v14, v15, s[4:5]
	v_mov_b32_dpp v15, v27 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v16, v16, v17, s[4:5]
	v_mov_b32_dpp v17, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v18, v18, v19, s[4:5]
	v_mov_b32_dpp v19, v29 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v20, v20, v21, s[4:5]
	v_mov_b32_dpp v21, v30 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v22, v22, v23, s[4:5]
	v_mov_b32_dpp v23, v31 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v24, v5, v4, v10
	v_perm_b32 v4, v5, v4, v11
	v_perm_b32 v5, v9, v8, v10
	v_perm_b32 v8, v9, v8, v11
	v_perm_b32 v9, v13, v12, v10
	v_perm_b32 v12, v13, v12, v11
	v_perm_b32 v13, v15, v14, v10
	v_perm_b32 v14, v15, v14, v11
	v_perm_b32 v15, v17, v16, v10
	v_perm_b32 v16, v17, v16, v11
	v_perm_b32 v17, v19, v18, v10
	v_perm_b32 v18, v19, v18, v11
	v_perm_b32 v19, v21, v20, v10
	v_perm_b32 v20, v21, v20, v11
	v_perm_b32 v10, v23, v22, v10
	v_perm_b32 v11, v23, v22, v11
	ds_write2_b32 v0, v24, v4 offset1:68
	ds_write2_b32 v0, v5, v8 offset0:8 offset1:76
	ds_write2_b32 v0, v9, v12 offset0:16 offset1:84
	ds_write2_b32 v0, v13, v14 offset0:24 offset1:92
	ds_write2_b32 v0, v15, v16 offset0:32 offset1:100
	ds_write2_b32 v0, v17, v18 offset0:40 offset1:108
	ds_write2_b32 v0, v19, v20 offset0:48 offset1:116
	ds_write2_b32 v0, v10, v11 offset0:56 offset1:124
	s_waitcnt lgkmcnt(0)
	ds_read_b128 v[8:11], v2 offset:55296
	ds_read_b128 v[12:15], v2 offset:56384
	ds_read_b128 v[16:19], v2 offset:57472
	ds_read_b128 v[20:23], v2 offset:58560
	s_add_i32 s4, s15, 0x70
	s_lshl_b32 s4, s4, 16
	s_add_i32 s4, s7, s4
	s_waitcnt lgkmcnt(3)
	buffer_store_dwordx4 v[8:11], v1, s[0:3], s4 offen nt
	s_waitcnt lgkmcnt(2)
	buffer_store_dwordx4 v[12:15], v3, s[0:3], s4 offen nt
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[16:19], v6, s[0:3], s4 offen nt
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[20:23], v7, s[0:3], s4 offen nt
	s_waitcnt lgkmcnt(0)
	s_endpgm
.Lrocke_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_32768x4096x2048_c0c1_Acanonical_k20ef87951c_end:
.size rocke_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_32768x4096x2048_c0c1_Acanonical_k20ef87951c, .Lrocke_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_32768x4096x2048_c0c1_Acanonical_k20ef87951c_end-rocke_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_32768x4096x2048_c0c1_Acanonical_k20ef87951c

.rodata
.p2align 6
.amdhsa_kernel rocke_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_32768x4096x2048_c0c1_Acanonical_k20ef87951c
  .amdhsa_group_segment_fixed_size 154144
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_kernarg_size 40
  .amdhsa_user_sgpr_count 2
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_next_free_vgpr 476
  .amdhsa_next_free_sgpr 64
  .amdhsa_reserve_vcc 1
  .amdhsa_accum_offset 220
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
    Description: MX E2M3 GEMM at 32768x4096x2048 on tiled A, canonical B, and shuffled scales
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
  AssertSizeEqual: {0: 32768, 1: 4096, 2: 1, 3: 2048}
  AssertStrideAEqual: {1: 2048}
  AssertStrideBEqual: {1: 2048}
  AssertStrideDEqual: {1: 32768}
  AssertAlphaValue: 1
amdhsa.target: amdgcn-amd-amdhsa--gfx950
amdhsa.version: [ 1, 1 ]
amdhsa.kernels:
  - .name: rocke_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_32768x4096x2048_c0c1_Acanonical_k20ef87951c
    .symbol: rocke_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_32768x4096x2048_c0c1_Acanonical_k20ef87951c.kd
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
    .group_segment_fixed_size: 154144
    .kernarg_segment_align: 8
    .kernarg_segment_size: 40
    .max_flat_workgroup_size: 256
    .private_segment_fixed_size: 0
    .sgpr_count: 70
    .vgpr_count: 476
    .agpr_count: 256
    .wavefront_size: 64
...
.end_amdgpu_metadata
