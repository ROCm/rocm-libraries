// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// MX E2M3 x E2M3 GEMM for gfx950: E8M0 scales per 32 K elements, fp32 accumulation,
// bf16 D rounded to nearest even. 256 threads (wave64, 2x2 waves), macrotile 256x256x128,
// v_mfma_scale_f32_16x16x128_f8f6f4 with cbsz:2 blgp:2, 154144 bytes of static LDS.
//
// Fixed problem, enforced by the predicates in custom.config:
//   M = N = 4096, K = 65536, batch 1, TransA = T, TransB = N, lda = ldb = K, D column-major
//   with ldd = 4096. alpha must be 1 (AssertAlphaValue) and beta 0 (UseBeta false): the
//   kernel takes no scalar arguments and never reads C. The grid must be exactly 256
//   workgroups: every buffer descriptor has num_records 0x7ffffff0 and no load or store is
//   masked, so any other size reads and writes out of range. Its only arguments are the
//   five pointers in the metadata below, so it cannot take the UserArgs launch path.
//   The schedule computes 24 * L + 8 K tiles of 128 (a 24-tile main loop and a fixed
//   8-tile tail), here L = 21.
//
// Inputs, in hipBLASLt naming. For a matrix X of R rows and K columns in this GEMM (A: R = M,
// B: R = N), with packed row r holding K fp6 codes as a 6-bit LSB-first bitstream:
//   A uses HIPBLASLT_ORDER_COL256_R128_EXT. With br = r/256, rb = (r%256)/16,
//   r16 = r%16, s = k/128 and kg = (k%128)/32, the 24 bytes of row r covering
//   k in [128s + 32kg, +32) are split: bytes 0..15 go to base + 16*o and bytes 16..23 to
//   base + 16384 + 8*o, where o = 64*rb + 16*kg + r16 and base = (br*K/128 + s)*24576.
//   B uses HIPBLASLT_ORDER_COL with ldb = K: packed row r starts at byte r*K*6/8.
//   The scales of A and B use HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_128_4_EXT: the scale
//   of row r and K block kb (32 elements) is at ((s*R/128 + r/128)*64 + 16*g + r%16)*8
//   + (r%128)/16, with s = kb/4 and g = kb%4.
//   The kernel's a/sa operands are hipBLASLt B and its scale, b/sb are A and its scale.
//
// Edits to the provider's instruction listing, none of which changes an instruction up to
// s_endpgm:
//   - the address and encoding comments on each instruction are removed;
//   - the immediate branch offset of the main-loop s_cbranch_scc1 is replaced by the label
//     .Lmain_loop, which encodes to the same offset;
//   - the provider's 264 s_nop padding instructions after s_endpgm are dropped;
//   - the directives, kernel descriptor and metadata are written for code object v4
//     (amdhsa.version [1, 1], no hidden arguments) from the provider's kernel descriptor.

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.amdhsa_code_object_version 4
.text
.globl rocke_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_4096x4096x65536_c0c1_Acanonical_k21f5fa872a
.p2align 8
.type rocke_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_4096x4096x65536_c0c1_Acanonical_k21f5fa872a,@function
rocke_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_4096x4096x65536_c0c1_Acanonical_k21f5fa872a:
	s_load_dwordx8 s[4:11], s[0:1], 0x0
	s_nop 0
	s_load_dwordx2 s[0:1], s[0:1], 0x20
	v_lshrrev_b32_e32 v0, 6, v0
	s_nop 0
	v_readfirstlane_b32 s16, v0
	v_mbcnt_lo_u32_b32 v0, -1, 0
	v_mbcnt_hi_u32_b32 v6, -1, v0
	s_lshr_b32 s3, s16, 1
	s_and_b32 s12, s16, 1
	s_and_b32 s13, s2, 7
	s_lshr_b32 s2, s2, 3
	s_lshr_b32 s14, s13, 1
	s_lshl_b32 s14, s14, 2
	s_lshr_b32 s15, s2, 3
	s_add_i32 s14, s14, s15
	s_and_b32 s13, s13, 1
	s_lshl_b32 s13, s13, 3
	s_and_b32 s2, s2, 7
	s_add_i32 s17, s13, s2
	s_lshl_b32 s18, s14, 8
	s_mul_i32 s2, s18, 0xc000
	s_lshl_b32 s19, s17, 8
	s_lshl_b32 s20, s3, 7
	s_lshl_b32 s21, s12, 7
	s_waitcnt lgkmcnt(0)
	s_add_u32 s2, s4, s2
	s_mul_i32 s56, s16, 0xc000
	s_mul_i32 s26, s16, 0x810
	s_addc_u32 s3, s5, 0
	s_lshl_b32 s22, s20, 3
	s_lshl_b32 s4, s16, 6
	s_add_i32 s57, s56, 0x30000
	s_add_i32 s27, s26, 0x2040
	s_add_i32 s58, s56, 0x60000
	s_add_i32 s28, s26, 0x4080
	s_add_i32 s59, s56, 0x90000
	s_add_i32 s29, s26, 0x60c0
	s_add_i32 s60, s56, 0x600000
	s_add_i32 s30, s26, 0x400
	s_add_i32 s5, s16, 20
	s_add_i32 s12, s16, 4
	s_lshr_b32 s5, s5, 4
	s_lshl_b32 s5, s5, 3
	s_lshl_b32 s13, s5, 4
	s_add_i32 s23, s12, s13
	s_mul_i32 s12, s12, 0x810
	s_lshl_b32 s5, s5, 7
	s_add_i32 s31, s12, s5
	s_add_i32 s5, s16, 24
	s_add_i32 s12, s16, 8
	s_lshr_b32 s5, s5, 4
	s_lshl_b32 s5, s5, 3
	s_lshl_b32 s13, s5, 4
	s_add_i32 s32, s12, s13
	s_mul_i32 s24, s12, 0x810
	s_lshl_b32 s25, s5, 7
	v_and_b32_e32 v218, 15, v6
	v_lshrrev_b32_e32 v219, 4, v6
	v_and_b32_e32 v0, 3, v218
	v_add_u32_e32 v1, s4, v6
	v_lshrrev_b32_e32 v2, 2, v218
	v_lshlrev_b32_e32 v0, 5, v0
	v_mul_u32_u24_e32 v3, 24, v219
	v_and_b32_e32 v4, 15, v1
	v_lshrrev_b32_e32 v1, 4, v1
	v_cmp_ge_i32_e64 s[4:5], v219, 1
	v_cmp_ge_i32_e64 s[12:13], v219, 2
	v_cmp_ge_i32_e64 s[14:15], v219, 3
	s_mov_b32 s33, 0x808
	v_mad_u32_u24 v0, s33, v2, v0
	v_lshlrev_b32_e32 v1, 7, v1
	v_lshrrev_b32_e32 v2, 2, v4
	v_and_b32_e32 v5, 7, v6
	v_add3_u32 v0, s22, v3, v0
	v_mad_u32_u24 v2, s33, v2, v1
	v_and_b32_e32 v7, 3, v4
	v_lshrrev_b32_e32 v8, 3, v6
	s_add_i32 s33, s24, s25
	s_add_i32 s24, s16, 28
	v_lshlrev_b32_e32 v5, 4, v5
	s_add_i32 s25, s16, 12
	v_add_u32_e32 v9, 0x101a0, v0
	v_add_u32_e32 v0, 0x80a0, v0
	s_mov_b32 s34, 0x810
	v_mad_u32_u24 v1, s34, v4, v1
	v_lshl_add_u32 v2, v7, 5, v2
	s_mov_b32 s34, 0xc0000
	s_lshr_b32 s24, s24, 4
	v_mad_u32_u24 v220, s34, v8, v5
	s_lshl_b32 s24, s24, 3
	s_lshl_b32 s34, s24, 4
	s_add_i32 s34, s25, s34
	s_mul_i32 s25, s25, 0x810
	s_lshl_b32 s24, s24, 7
	s_mov_b32 s35, 0x810
	v_mad_u32_u24 v3, s35, v218, v3
	s_add_i32 s35, s25, s24
	v_add_u32_e32 v3, s22, v3
	v_add_u32_e32 v4, 0x8080, v3
	v_add_u32_e32 v5, 0xffff7e80, v3
	v_add_u32_e32 v1, 0x60, v1
	v_add_u32_e32 v2, 0x10200, v2
	v_cndmask_b32_e64 v7, v3, v4, s[12:13]
	v_cndmask_b32_e64 v8, v3, v5, s[12:13]
	v_cndmask_b32_e64 v10, v3, v4, s[14:15]
	v_cndmask_b32_e64 v11, v3, v5, s[14:15]
	v_cndmask_b32_e64 v12, v9, v4, s[4:5]
	v_cndmask_b32_e64 v13, v0, v5, s[4:5]
	v_cndmask_b32_e64 v4, v9, v4, s[12:13]
	v_cndmask_b32_e64 v0, v0, v5, s[12:13]
	v_lshrrev_b32_e32 v94, 2, v1
	v_lshrrev_b32_e32 v104, 2, v2
	s_mul_i32 s4, s17, 0xc00000
	s_add_u32 s24, s6, s4
	s_addc_u32 s25, s7, 0
	s_lshr_b32 s36, s21, 4
	s_and_b32 s5, s25, 0xffff
	s_add_i32 s61, s18, s20
	s_add_i32 s62, s19, s21
	s_and_b32 s13, s9, 0xffff
	s_and_b32 s17, s11, 0xffff
	s_mul_i32 s63, s23, 0xc000
	s_lshl_b32 s64, s16, 8
	s_lshr_b32 s65, s18, 7
	s_lshr_b32 s37, s19, 7
	s_lshr_b32 s9, s20, 7
	s_lshr_b32 s38, s21, 7
	s_add_i32 s66, 0x13800, s26
	s_mul_i32 s67, s32, 0xc000
	s_and_b32 s21, s3, 0xffff
	s_mul_i32 s68, s34, 0xc000
	v_lshrrev_b32_e32 v1, 2, v3
	v_lshrrev_b32_e32 v178, 2, v7
	v_lshrrev_b32_e32 v179, 2, v8
	v_lshrrev_b32_e32 v180, 2, v10
	v_lshrrev_b32_e32 v181, 2, v11
	v_lshrrev_b32_e32 v105, 2, v12
	v_lshrrev_b32_e32 v182, 2, v13
	v_lshrrev_b32_e32 v106, 2, v4
	v_lshrrev_b32_e32 v183, 2, v0
	v_lshlrev_b32_e32 v221, 4, v6
	s_mul_i32 s69, s16, 0x1800
	s_mov_b32 s4, s24
	s_mov_b32 s6, 0x7ffffff0
	s_mov_b32 s7, 0x31014000
	s_mov_b32 s12, s8
	s_mov_b32 s14, 0x7ffffff0
	s_mov_b32 s15, 0x31014000
	s_mov_b32 s16, s10
	s_mov_b32 s18, 0x7ffffff0
	s_mov_b32 s19, 0x31014000
	v_lshlrev_b32_e32 v222, 2, v6
	s_mov_b32 s20, s2
	s_mov_b32 s22, 0x7ffffff0
	s_mov_b32 m0, s66
	s_mov_b32 s23, 0x31014000
	buffer_load_dwordx4 v220, s[20:23], s56 offen lds
	s_add_i32 s70, 0x13800, s27
	s_mov_b32 m0, s70
	s_nop 0
	buffer_load_dwordx4 v220, s[20:23], s57 offen lds
	s_add_i32 s71, 0x13800, s28
	s_mov_b32 m0, s71
	s_nop 0
	buffer_load_dwordx4 v220, s[20:23], s58 offen lds
	s_add_i32 s72, 0x13800, s29
	s_mov_b32 m0, s72
	s_nop 0
	buffer_load_dwordx4 v220, s[20:23], s59 offen lds
	s_add_i32 s73, 0x13800, s30
	s_mov_b32 m0, s73
	s_nop 0
	buffer_load_dwordx4 v220, s[20:23], s60 offen lds
	s_add_i32 s74, 0x13800, s31
	s_mov_b32 m0, s74
	s_nop 0
	buffer_load_dwordx4 v220, s[20:23], s63 offen lds
	s_add_i32 s75, 0x13800, s33
	s_mov_b32 m0, s75
	s_nop 0
	buffer_load_dwordx4 v220, s[20:23], s67 offen lds
	s_add_i32 s76, 0x13800, s35
	s_mov_b32 m0, s76
	s_nop 0
	buffer_load_dwordx4 v220, s[20:23], s68 offen lds
	s_lshl_b32 s8, s65, 9
	s_lshl_b32 s32, s37, 9
	s_add_i32 s77, 0x1800, s69
	s_mov_b32 m0, s77
	s_nop 0
	buffer_load_dwordx4 v221, s[4:7], s69 offen lds
	buffer_load_dwordx4 v221, s[4:7], s69 offen offset:1024 lds
	buffer_load_dwordx4 v221, s[4:7], s69 offen offset:2048 lds
	buffer_load_dwordx4 v221, s[4:7], s69 offen offset:3072 lds
	s_add_i32 s34, s69, 0x1000
	s_add_i32 s78, 0x1800, s34
	s_mov_b32 m0, s78
	s_nop 0
	buffer_load_dwordx4 v221, s[4:7], s34 offen lds
	buffer_load_dwordx4 v221, s[4:7], s34 offen offset:1024 lds
	s_mov_b32 m0, s64
	s_add_i32 s8, s64, s8
	buffer_load_dword v222, s[12:15], s8 offen lds
	s_add_i32 s8, 0x400, s64
	s_mov_b32 m0, s8
	s_add_i32 s79, s64, s32
	buffer_load_dword v222, s[16:19], s79 offen lds
	s_add_i32 s8, s56, 0x80
	s_add_i32 s26, s26, 0x8100
	s_add_i32 s80, 0x13800, s26
	s_mov_b32 m0, s80
	s_nop 0
	buffer_load_dwordx4 v220, s[20:23], s8 offen lds
	s_add_i32 s8, s57, 0x80
	s_add_i32 s26, s27, 0x8100
	s_add_i32 s81, 0x13800, s26
	s_mov_b32 m0, s81
	s_nop 0
	buffer_load_dwordx4 v220, s[20:23], s8 offen lds
	s_add_i32 s8, s58, 0x80
	s_add_i32 s26, s28, 0x8100
	s_add_i32 s82, 0x13800, s26
	s_mov_b32 m0, s82
	s_nop 0
	buffer_load_dwordx4 v220, s[20:23], s8 offen lds
	s_add_i32 s8, s59, 0x80
	s_add_i32 s26, s29, 0x8100
	s_add_i32 s83, 0x13800, s26
	s_mov_b32 m0, s83
	s_nop 0
	buffer_load_dwordx4 v220, s[20:23], s8 offen lds
	s_add_i32 s8, s60, 0x80
	s_add_i32 s26, s30, 0x8100
	s_add_i32 s84, 0x13800, s26
	s_mov_b32 m0, s84
	s_nop 0
	buffer_load_dwordx4 v220, s[20:23], s8 offen lds
	s_add_i32 s8, s63, 0x80
	s_add_i32 s26, s31, 0x8100
	s_add_i32 s85, 0x13800, s26
	s_mov_b32 m0, s85
	s_nop 0
	buffer_load_dwordx4 v220, s[20:23], s8 offen lds
	s_add_i32 s8, s67, 0x80
	s_add_i32 s26, s33, 0x8100
	s_add_i32 s86, 0x13800, s26
	s_mov_b32 m0, s86
	s_nop 0
	buffer_load_dwordx4 v220, s[20:23], s8 offen lds
	s_add_i32 s8, s68, 0x80
	s_add_i32 s26, s35, 0x8100
	s_add_i32 s87, 0x13800, s26
	s_mov_b32 m0, s87
	s_nop 0
	buffer_load_dwordx4 v220, s[20:23], s8 offen lds
	s_add_i32 s8, s65, 32
	s_lshl_b32 s8, s8, 9
	s_add_i32 s26, s69, 0x6000
	s_add_i32 s88, 0x7800, s69
	s_mov_b32 m0, s88
	s_nop 0
	buffer_load_dwordx4 v221, s[4:7], s26 offen lds
	buffer_load_dwordx4 v221, s[4:7], s26 offen offset:1024 lds
	buffer_load_dwordx4 v221, s[4:7], s26 offen offset:2048 lds
	buffer_load_dwordx4 v221, s[4:7], s26 offen offset:3072 lds
	s_add_i32 s89, 0x7800, s34
	s_mov_b32 m0, s89
	s_add_i32 s26, s69, 0x7000
	buffer_load_dwordx4 v221, s[4:7], s26 offen lds
	buffer_load_dwordx4 v221, s[4:7], s26 offen offset:1024 lds
	s_add_i32 s90, 0x800, s64
	s_mov_b32 m0, s90
	s_add_i32 s8, s64, s8
	buffer_load_dword v222, s[12:15], s8 offen lds
	s_add_i32 s8, s79, 0x3c00
	buffer_load_dword v222, s[16:19], s8 offen offset:1024 lds
	s_add_i32 s8, s65, 64
	s_lshl_b32 s8, s8, 9
	s_add_i32 s26, s69, 0xc000
	s_add_i32 s91, 0xd800, s69
	s_mov_b32 m0, s91
	s_nop 0
	buffer_load_dwordx4 v221, s[4:7], s26 offen lds
	buffer_load_dwordx4 v221, s[4:7], s26 offen offset:1024 lds
	buffer_load_dwordx4 v221, s[4:7], s26 offen offset:2048 lds
	buffer_load_dwordx4 v221, s[4:7], s26 offen offset:3072 lds
	s_add_i32 s92, 0xd800, s34
	s_mov_b32 m0, s92
	s_add_i32 s26, s69, 0xd000
	buffer_load_dwordx4 v221, s[4:7], s26 offen lds
	buffer_load_dwordx4 v221, s[4:7], s26 offen offset:1024 lds
	s_add_i32 s93, 0x1000, s64
	s_mov_b32 m0, s93
	s_add_i32 s8, s64, s8
	buffer_load_dword v222, s[12:15], s8 offen lds
	s_add_i32 s8, s79, 0x7c00
	buffer_load_dword v222, s[16:19], s8 offen offset:1024 lds
	s_waitcnt vmcnt(24) lgkmcnt(0)
	s_barrier
	s_lshl_b32 s8, s9, 7
	v_lshlrev_b32_e32 v0, 1, v6
	s_lshl_b32 s9, s38, 7
	v_lshlrev_b32_e32 v223, 2, v1
	v_add_lshl_u32 v224, s8, v0, 2
	v_add_lshl_u32 v225, s9, v0, 2
	v_add_u32_e32 v52, 0x10000, v223
	ds_read_b64 v[174:175], v224
	ds_read_b64 v[176:177], v225 offset:1024
	ds_read_b64 v[0:1], v52 offset:14336
	ds_read_b64 v[2:3], v52 offset:14344
	ds_read_b64 v[4:5], v52 offset:14352
	s_lshl_b32 s8, s36, 6
	v_add_u32_e32 v10, s8, v6
	v_lshlrev_b32_e32 v226, 4, v10
	ds_read_b128 v[6:9], v226 offset:6144
	v_lshlrev_b32_e32 v227, 3, v10
	ds_read_b64 v[10:11], v227 offset:22528
	ds_read_b64 v[12:13], v52 offset:14464
	ds_read_b64 v[14:15], v52 offset:14472
	ds_read_b64 v[16:17], v52 offset:14480
	ds_read_b64 v[18:19], v52 offset:14592
	ds_read_b64 v[20:21], v52 offset:14600
	ds_read_b64 v[22:23], v52 offset:14608
	ds_read_b64 v[24:25], v52 offset:14720
	ds_read_b64 v[26:27], v52 offset:14728
	ds_read_b64 v[28:29], v52 offset:14736
	ds_read_b64 v[30:31], v52 offset:14848
	ds_read_b64 v[32:33], v52 offset:14856
	ds_read_b64 v[34:35], v52 offset:14864
	ds_read_b64 v[36:37], v52 offset:14976
	ds_read_b64 v[38:39], v52 offset:14984
	ds_read_b64 v[40:41], v52 offset:14992
	ds_read_b64 v[42:43], v52 offset:15104
	ds_read_b64 v[44:45], v52 offset:15112
	ds_read_b64 v[46:47], v52 offset:15120
	ds_read_b64 v[48:49], v52 offset:15232
	ds_read_b64 v[50:51], v52 offset:15240
	ds_read_b64 v[52:53], v52 offset:15248
	ds_read_b128 v[54:57], v226 offset:7168
	ds_read_b64 v[58:59], v227 offset:23040
	ds_read_b128 v[60:63], v226 offset:8192
	ds_read_b64 v[64:65], v227 offset:23552
	ds_read_b128 v[66:69], v226 offset:9216
	ds_read_b64 v[70:71], v227 offset:24064
	ds_read_b128 v[72:75], v226 offset:10240
	ds_read_b64 v[76:77], v227 offset:24576
	ds_read_b128 v[78:81], v226 offset:11264
	ds_read_b64 v[82:83], v227 offset:25088
	ds_read_b128 v[84:87], v226 offset:12288
	ds_read_b64 v[88:89], v227 offset:25600
	ds_read_b128 v[90:93], v226 offset:13312
	v_lshlrev_b32_e32 v228, 2, v94
	v_add_u32_e32 v100, 0x10000, v228
	ds_read_b64 v[94:95], v227 offset:26112
	ds_read_b128 v[96:99], v100 offset:14336
	ds_read_b128 v[100:103], v100 offset:14352
	v_lshlrev_b32_e32 v229, 2, v104
	v_add_u32_e32 v104, 0x10000, v229
	s_waitcnt lgkmcnt(1)
	ds_write_b64 v104, v[96:97] offset:14336
	ds_write_b64 v104, v[98:99] offset:14344
	s_waitcnt lgkmcnt(2)
	ds_write_b64 v104, v[100:101] offset:14352
	ds_write_b64 v104, v[102:103] offset:14360
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_mov_b32 m0, s66
	s_add_i32 s8, s56, 0x100
	buffer_load_dwordx4 v220, s[20:23], s8 offen lds
	s_mov_b32 m0, s70
	s_add_i32 s8, s57, 0x100
	buffer_load_dwordx4 v220, s[20:23], s8 offen lds
	s_mov_b32 m0, s71
	s_add_i32 s8, s58, 0x100
	buffer_load_dwordx4 v220, s[20:23], s8 offen lds
	s_mov_b32 m0, s72
	s_add_i32 s8, s59, 0x100
	buffer_load_dwordx4 v220, s[20:23], s8 offen lds
	s_mov_b32 m0, s73
	s_add_i32 s8, s60, 0x100
	buffer_load_dwordx4 v220, s[20:23], s8 offen lds
	s_mov_b32 m0, s74
	s_add_i32 s8, s63, 0x100
	buffer_load_dwordx4 v220, s[20:23], s8 offen lds
	s_mov_b32 m0, s75
	s_add_i32 s8, s67, 0x100
	buffer_load_dwordx4 v220, s[20:23], s8 offen lds
	s_mov_b32 m0, s76
	s_add_i32 s8, s68, 0x100
	buffer_load_dwordx4 v220, s[20:23], s8 offen lds
	s_add_i32 s8, s65, 0x60
	s_lshl_b32 s8, s8, 9
	v_accvgpr_write_b32 a0, 0
	v_accvgpr_write_b32 a1, 0
	v_accvgpr_write_b32 a2, 0
	v_accvgpr_write_b32 a3, 0
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[0:5], v[6:11], a[0:3], v174, v176 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(16) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[210:211], v224 offset:2048
	ds_read_b64 v[212:213], v225 offset:3072
	v_lshlrev_b32_e32 v184, 2, v106
	v_add_u32_e32 v132, 0x10000, v184
	v_accvgpr_write_b32 a4, 0
	v_accvgpr_write_b32 a5, 0
	v_accvgpr_write_b32 a6, 0
	v_accvgpr_write_b32 a7, 0
	ds_read_b64 v[96:97], v132 offset:14432
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[12:17], v[6:11], a[4:7], v174, v176 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_lshlrev_b32_e32 v185, 2, v105
	v_add_u32_e32 v136, 0x10000, v185
	ds_read_b64 v[98:99], v136 offset:14440
	ds_read_b64 v[100:101], v136 offset:14448
	s_add_i32 s9, s69, 0x12000
	v_accvgpr_write_b32 a8, 0
	v_accvgpr_write_b32 a9, 0
	v_accvgpr_write_b32 a10, 0
	v_accvgpr_write_b32 a11, 0
	s_mov_b32 m0, s77
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:23], v[6:11], a[8:11], v174, v176 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s9 offen lds
	v_accvgpr_write_b32 a12, 0
	v_accvgpr_write_b32 a13, 0
	v_accvgpr_write_b32 a14, 0
	v_accvgpr_write_b32 a15, 0
	ds_read_b64 v[102:103], v132 offset:14560
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[24:29], v[6:11], a[12:15], v174, v176 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[104:105], v136 offset:14568
	ds_read_b64 v[106:107], v136 offset:14576
	v_accvgpr_write_b32 a16, 0
	v_accvgpr_write_b32 a17, 0
	v_accvgpr_write_b32 a18, 0
	v_accvgpr_write_b32 a19, 0
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[30:35], v[6:11], a[16:19], v175, v176 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_accvgpr_write_b32 a20, 0
	v_accvgpr_write_b32 a21, 0
	v_accvgpr_write_b32 a22, 0
	v_accvgpr_write_b32 a23, 0
	ds_read_b64 v[108:109], v132 offset:14688
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[36:41], v[6:11], a[20:23], v175, v176 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[110:111], v136 offset:14696
	ds_read_b64 v[112:113], v136 offset:14704
	v_accvgpr_write_b32 a24, 0
	v_accvgpr_write_b32 a25, 0
	v_accvgpr_write_b32 a26, 0
	v_accvgpr_write_b32 a27, 0
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[42:47], v[6:11], a[24:27], v175, v176 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s9 offen offset:1024 lds
	v_accvgpr_write_b32 a28, 0
	v_accvgpr_write_b32 a29, 0
	v_accvgpr_write_b32 a30, 0
	v_accvgpr_write_b32 a31, 0
	ds_read_b64 v[114:115], v132 offset:14816
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[48:53], v[6:11], a[28:31], v175, v176 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[116:117], v136 offset:14824
	ds_read_b64 v[118:119], v136 offset:14832
	v_accvgpr_write_b32 a32, 0
	v_accvgpr_write_b32 a33, 0
	v_accvgpr_write_b32 a34, 0
	v_accvgpr_write_b32 a35, 0
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[48:53], v[54:59], a[32:35], v175, v176 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_accvgpr_write_b32 a36, 0
	v_accvgpr_write_b32 a37, 0
	v_accvgpr_write_b32 a38, 0
	v_accvgpr_write_b32 a39, 0
	ds_read_b64 v[6:7], v132 offset:14944
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[42:47], v[54:59], a[36:39], v175, v176 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[8:9], v136 offset:14952
	ds_read_b64 v[10:11], v136 offset:14960
	v_accvgpr_write_b32 a40, 0
	v_accvgpr_write_b32 a41, 0
	v_accvgpr_write_b32 a42, 0
	v_accvgpr_write_b32 a43, 0
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[36:41], v[54:59], a[40:43], v175, v176 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s9 offen offset:2048 lds
	v_accvgpr_write_b32 a44, 0
	v_accvgpr_write_b32 a45, 0
	v_accvgpr_write_b32 a46, 0
	v_accvgpr_write_b32 a47, 0
	ds_read_b64 v[120:121], v132 offset:15072
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[30:35], v[54:59], a[44:47], v175, v176 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[122:123], v136 offset:15080
	ds_read_b64 v[124:125], v136 offset:15088
	v_accvgpr_write_b32 a48, 0
	v_accvgpr_write_b32 a49, 0
	v_accvgpr_write_b32 a50, 0
	v_accvgpr_write_b32 a51, 0
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[24:29], v[54:59], a[48:51], v174, v176 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_accvgpr_write_b32 a52, 0
	v_accvgpr_write_b32 a53, 0
	v_accvgpr_write_b32 a54, 0
	v_accvgpr_write_b32 a55, 0
	ds_read_b64 v[126:127], v132 offset:15200
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[18:23], v[54:59], a[52:55], v174, v176 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[128:129], v136 offset:15208
	ds_read_b64 v[130:131], v136 offset:15216
	v_accvgpr_write_b32 a56, 0
	v_accvgpr_write_b32 a57, 0
	v_accvgpr_write_b32 a58, 0
	v_accvgpr_write_b32 a59, 0
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[12:17], v[54:59], a[56:59], v174, v176 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s9 offen offset:3072 lds
	v_accvgpr_write_b32 a60, 0
	v_accvgpr_write_b32 a61, 0
	v_accvgpr_write_b32 a62, 0
	v_accvgpr_write_b32 a63, 0
	ds_read_b64 v[132:133], v132 offset:15328
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[0:5], v[54:59], a[60:63], v174, v176 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[134:135], v136 offset:15336
	ds_read_b64 v[136:137], v136 offset:15344
	v_accvgpr_write_b32 a64, 0
	v_accvgpr_write_b32 a65, 0
	v_accvgpr_write_b32 a66, 0
	v_accvgpr_write_b32 a67, 0
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[0:5], v[60:65], a[64:67], v174, v176 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_accvgpr_write_b32 a68, 0
	v_accvgpr_write_b32 a69, 0
	v_accvgpr_write_b32 a70, 0
	v_accvgpr_write_b32 a71, 0
	ds_read_b128 v[54:57], v226 offset:30720
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[12:17], v[60:65], a[68:71], v174, v176 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[58:59], v227 offset:47104
	s_add_i32 s9, s69, 0x13000
	v_accvgpr_write_b32 a72, 0
	v_accvgpr_write_b32 a73, 0
	v_accvgpr_write_b32 a74, 0
	v_accvgpr_write_b32 a75, 0
	s_mov_b32 m0, s78
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[18:23], v[60:65], a[72:75], v174, v176 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s9 offen lds
	v_accvgpr_write_b32 a76, 0
	v_accvgpr_write_b32 a77, 0
	v_accvgpr_write_b32 a78, 0
	v_accvgpr_write_b32 a79, 0
	ds_read_b128 v[138:141], v226 offset:31744
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[24:29], v[60:65], a[76:79], v174, v176 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[142:143], v227 offset:47616
	v_accvgpr_write_b32 a80, 0
	v_accvgpr_write_b32 a81, 0
	v_accvgpr_write_b32 a82, 0
	v_accvgpr_write_b32 a83, 0
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[30:35], v[60:65], a[80:83], v175, v176 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_accvgpr_write_b32 a84, 0
	v_accvgpr_write_b32 a85, 0
	v_accvgpr_write_b32 a86, 0
	v_accvgpr_write_b32 a87, 0
	ds_read_b128 v[144:147], v226 offset:32768
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[36:41], v[60:65], a[84:87], v175, v176 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[148:149], v227 offset:48128
	v_accvgpr_write_b32 a88, 0
	v_accvgpr_write_b32 a89, 0
	v_accvgpr_write_b32 a90, 0
	v_accvgpr_write_b32 a91, 0
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[42:47], v[60:65], a[88:91], v175, v176 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s9 offen offset:1024 lds
	v_accvgpr_write_b32 a92, 0
	v_accvgpr_write_b32 a93, 0
	v_accvgpr_write_b32 a94, 0
	v_accvgpr_write_b32 a95, 0
	ds_read_b128 v[150:153], v226 offset:33792
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[48:53], v[60:65], a[92:95], v175, v176 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[154:155], v227 offset:48640
	v_accvgpr_write_b32 a96, 0
	v_accvgpr_write_b32 a97, 0
	v_accvgpr_write_b32 a98, 0
	v_accvgpr_write_b32 a99, 0
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[48:53], v[66:71], a[96:99], v175, v176 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_accvgpr_write_b32 a100, 0
	v_accvgpr_write_b32 a101, 0
	v_accvgpr_write_b32 a102, 0
	v_accvgpr_write_b32 a103, 0
	ds_read_b128 v[60:63], v226 offset:34816
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[42:47], v[66:71], a[100:103], v175, v176 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[64:65], v227 offset:49152
	s_add_i32 s8, s64, s8
	v_accvgpr_write_b32 a104, 0
	v_accvgpr_write_b32 a105, 0
	v_accvgpr_write_b32 a106, 0
	v_accvgpr_write_b32 a107, 0
	s_mov_b32 m0, s64
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[36:41], v[66:71], a[104:107], v175, v176 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[12:15], s8 offen lds
	v_accvgpr_write_b32 a108, 0
	v_accvgpr_write_b32 a109, 0
	v_accvgpr_write_b32 a110, 0
	v_accvgpr_write_b32 a111, 0
	ds_read_b128 v[156:159], v226 offset:35840
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[30:35], v[66:71], a[108:111], v175, v176 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[160:161], v227 offset:49664
	v_accvgpr_write_b32 a112, 0
	v_accvgpr_write_b32 a113, 0
	v_accvgpr_write_b32 a114, 0
	v_accvgpr_write_b32 a115, 0
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[24:29], v[66:71], a[112:115], v174, v176 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_accvgpr_write_b32 a116, 0
	v_accvgpr_write_b32 a117, 0
	v_accvgpr_write_b32 a118, 0
	v_accvgpr_write_b32 a119, 0
	ds_read_b128 v[162:165], v226 offset:36864
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[18:23], v[66:71], a[116:119], v174, v176 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[166:167], v227 offset:50176
	s_add_i32 s8, s79, 0xbc00
	v_accvgpr_write_b32 a120, 0
	v_accvgpr_write_b32 a121, 0
	v_accvgpr_write_b32 a122, 0
	v_accvgpr_write_b32 a123, 0
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[12:17], v[66:71], a[120:123], v174, v176 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[16:19], s8 offen offset:1024 lds
	v_accvgpr_write_b32 a124, 0
	v_accvgpr_write_b32 a125, 0
	v_accvgpr_write_b32 a126, 0
	v_accvgpr_write_b32 a127, 0
	ds_read_b128 v[168:171], v226 offset:37888
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[0:5], v[66:71], a[124:127], v174, v176 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[172:173], v227 offset:50688
	v_accvgpr_write_b32 a128, 0
	v_accvgpr_write_b32 a129, 0
	v_accvgpr_write_b32 a130, 0
	v_accvgpr_write_b32 a131, 0
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[0:5], v[72:77], a[128:131], v174, v177 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_accvgpr_write_b32 a132, 0
	v_accvgpr_write_b32 a133, 0
	v_accvgpr_write_b32 a134, 0
	v_accvgpr_write_b32 a135, 0
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[12:17], v[72:77], a[132:135], v174, v177 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_accvgpr_write_b32 a136, 0
	v_accvgpr_write_b32 a137, 0
	v_accvgpr_write_b32 a138, 0
	v_accvgpr_write_b32 a139, 0
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[18:23], v[72:77], a[136:139], v174, v177 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_accvgpr_write_b32 a140, 0
	v_accvgpr_write_b32 a141, 0
	v_accvgpr_write_b32 a142, 0
	v_accvgpr_write_b32 a143, 0
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[24:29], v[72:77], a[140:143], v174, v177 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_accvgpr_write_b32 a144, 0
	v_accvgpr_write_b32 a145, 0
	v_accvgpr_write_b32 a146, 0
	v_accvgpr_write_b32 a147, 0
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[30:35], v[72:77], a[144:147], v175, v177 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_accvgpr_write_b32 a148, 0
	v_accvgpr_write_b32 a149, 0
	v_accvgpr_write_b32 a150, 0
	v_accvgpr_write_b32 a151, 0
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[36:41], v[72:77], a[148:151], v175, v177 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_accvgpr_write_b32 a152, 0
	v_accvgpr_write_b32 a153, 0
	v_accvgpr_write_b32 a154, 0
	v_accvgpr_write_b32 a155, 0
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[42:47], v[72:77], a[152:155], v175, v177 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_accvgpr_write_b32 a156, 0
	v_accvgpr_write_b32 a157, 0
	v_accvgpr_write_b32 a158, 0
	v_accvgpr_write_b32 a159, 0
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[48:53], v[72:77], a[156:159], v175, v177 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_accvgpr_write_b32 a160, 0
	v_accvgpr_write_b32 a161, 0
	v_accvgpr_write_b32 a162, 0
	v_accvgpr_write_b32 a163, 0
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[48:53], v[78:83], a[160:163], v175, v177 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_accvgpr_write_b32 a164, 0
	v_accvgpr_write_b32 a165, 0
	v_accvgpr_write_b32 a166, 0
	v_accvgpr_write_b32 a167, 0
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[42:47], v[78:83], a[164:167], v175, v177 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_accvgpr_write_b32 a168, 0
	v_accvgpr_write_b32 a169, 0
	v_accvgpr_write_b32 a170, 0
	v_accvgpr_write_b32 a171, 0
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[36:41], v[78:83], a[168:171], v175, v177 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_accvgpr_write_b32 a172, 0
	v_accvgpr_write_b32 a173, 0
	v_accvgpr_write_b32 a174, 0
	v_accvgpr_write_b32 a175, 0
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[30:35], v[78:83], a[172:175], v175, v177 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_accvgpr_write_b32 a176, 0
	v_accvgpr_write_b32 a177, 0
	v_accvgpr_write_b32 a178, 0
	v_accvgpr_write_b32 a179, 0
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[24:29], v[78:83], a[176:179], v174, v177 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_accvgpr_write_b32 a180, 0
	v_accvgpr_write_b32 a181, 0
	v_accvgpr_write_b32 a182, 0
	v_accvgpr_write_b32 a183, 0
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[18:23], v[78:83], a[180:183], v174, v177 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_accvgpr_write_b32 a184, 0
	v_accvgpr_write_b32 a185, 0
	v_accvgpr_write_b32 a186, 0
	v_accvgpr_write_b32 a187, 0
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[12:17], v[78:83], a[184:187], v174, v177 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_accvgpr_write_b32 a188, 0
	v_accvgpr_write_b32 a189, 0
	v_accvgpr_write_b32 a190, 0
	v_accvgpr_write_b32 a191, 0
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[0:5], v[78:83], a[188:191], v174, v177 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_accvgpr_write_b32 a192, 0
	v_accvgpr_write_b32 a193, 0
	v_accvgpr_write_b32 a194, 0
	v_accvgpr_write_b32 a195, 0
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[0:5], v[84:89], a[192:195], v174, v177 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_accvgpr_write_b32 a196, 0
	v_accvgpr_write_b32 a197, 0
	v_accvgpr_write_b32 a198, 0
	v_accvgpr_write_b32 a199, 0
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[12:17], v[84:89], a[196:199], v174, v177 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_accvgpr_write_b32 a200, 0
	v_accvgpr_write_b32 a201, 0
	v_accvgpr_write_b32 a202, 0
	v_accvgpr_write_b32 a203, 0
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[18:23], v[84:89], a[200:203], v174, v177 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_accvgpr_write_b32 a204, 0
	v_accvgpr_write_b32 a205, 0
	v_accvgpr_write_b32 a206, 0
	v_accvgpr_write_b32 a207, 0
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[24:29], v[84:89], a[204:207], v174, v177 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_accvgpr_write_b32 a208, 0
	v_accvgpr_write_b32 a209, 0
	v_accvgpr_write_b32 a210, 0
	v_accvgpr_write_b32 a211, 0
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[30:35], v[84:89], a[208:211], v175, v177 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_accvgpr_write_b32 a212, 0
	v_accvgpr_write_b32 a213, 0
	v_accvgpr_write_b32 a214, 0
	v_accvgpr_write_b32 a215, 0
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[36:41], v[84:89], a[212:215], v175, v177 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_accvgpr_write_b32 a216, 0
	v_accvgpr_write_b32 a217, 0
	v_accvgpr_write_b32 a218, 0
	v_accvgpr_write_b32 a219, 0
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[42:47], v[84:89], a[216:219], v175, v177 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_accvgpr_write_b32 a220, 0
	v_accvgpr_write_b32 a221, 0
	v_accvgpr_write_b32 a222, 0
	v_accvgpr_write_b32 a223, 0
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[48:53], v[84:89], a[220:223], v175, v177 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_accvgpr_write_b32 a224, 0
	v_accvgpr_write_b32 a225, 0
	v_accvgpr_write_b32 a226, 0
	v_accvgpr_write_b32 a227, 0
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[48:53], v[90:95], a[224:227], v175, v177 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_accvgpr_write_b32 a228, 0
	v_accvgpr_write_b32 a229, 0
	v_accvgpr_write_b32 a230, 0
	v_accvgpr_write_b32 a231, 0
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[42:47], v[90:95], a[228:231], v175, v177 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_accvgpr_write_b32 a232, 0
	v_accvgpr_write_b32 a233, 0
	v_accvgpr_write_b32 a234, 0
	v_accvgpr_write_b32 a235, 0
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[36:41], v[90:95], a[232:235], v175, v177 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_accvgpr_write_b32 a236, 0
	v_accvgpr_write_b32 a237, 0
	v_accvgpr_write_b32 a238, 0
	v_accvgpr_write_b32 a239, 0
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[30:35], v[90:95], a[236:239], v175, v177 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_accvgpr_write_b32 a240, 0
	v_accvgpr_write_b32 a241, 0
	v_accvgpr_write_b32 a242, 0
	v_accvgpr_write_b32 a243, 0
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[24:29], v[90:95], a[240:243], v174, v177 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_accvgpr_write_b32 a244, 0
	v_accvgpr_write_b32 a245, 0
	v_accvgpr_write_b32 a246, 0
	v_accvgpr_write_b32 a247, 0
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[18:23], v[90:95], a[244:247], v174, v177 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_accvgpr_write_b32 a248, 0
	v_accvgpr_write_b32 a249, 0
	v_accvgpr_write_b32 a250, 0
	v_accvgpr_write_b32 a251, 0
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[12:17], v[90:95], a[248:251], v174, v177 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_accvgpr_write_b32 a252, 0
	v_accvgpr_write_b32 a253, 0
	v_accvgpr_write_b32 a254, 0
	v_accvgpr_write_b32 a255, 0
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[0:5], v[90:95], a[252:255], v174, v177 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_u32 s8, s24, 0x1000
	s_addc_u32 s9, s25, 0
	s_and_b32 s9, s9, 0xffff
	s_mul_i32 s24, s65, 0xfffffe00
	s_lshl_b32 s25, s37, 9
	s_add_i32 s24, s24, s25
	s_add_i32 s24, s24, 0xfffffc00
	s_ashr_i32 s25, s24, 31
	s_add_u32 s24, s10, s24
	s_addc_u32 s25, s11, s25
	s_mul_i32 s10, s56, -1
	s_and_b32 s25, s25, 0xffff
	s_add_i32 s11, s57, s10
	s_add_u32 s28, s2, s11
	s_addc_u32 s29, s3, 0
	s_and_b32 s29, s29, 0xffff
	s_add_i32 s11, s58, s10
	s_add_u32 s32, s2, s11
	s_addc_u32 s33, s3, 0
	s_and_b32 s33, s33, 0xffff
	s_add_i32 s11, s59, s10
	s_add_u32 s36, s2, s11
	s_addc_u32 s37, s3, 0
	s_and_b32 s37, s37, 0xffff
	s_add_i32 s11, s60, s10
	s_add_u32 s40, s2, s11
	s_addc_u32 s41, s3, 0
	s_and_b32 s41, s41, 0xffff
	s_add_i32 s11, s63, s10
	s_add_u32 s44, s2, s11
	s_addc_u32 s45, s3, 0
	s_and_b32 s45, s45, 0xffff
	s_add_i32 s11, s67, s10
	s_add_u32 s48, s2, s11
	s_addc_u32 s49, s3, 0
	s_and_b32 s49, s49, 0xffff
	s_add_i32 s10, s68, s10
	s_add_u32 s52, s2, s10
	s_addc_u32 s53, s3, 0
	s_and_b32 s53, s53, 0xffff
	v_lshlrev_b32_e32 v230, 2, v181
	s_mov_b32 s10, 0x7ffffff0
	v_lshlrev_b32_e32 v231, 2, v179
	s_mov_b32 s11, 0x31014000
	v_lshlrev_b32_e32 v232, 2, v183
	v_lshlrev_b32_e32 v233, 2, v182
	v_lshlrev_b32_e32 v234, 2, v180
	v_lshlrev_b32_e32 v235, 2, v178
	s_mov_b32 s26, 0x7ffffff0
	s_mov_b32 s27, 0x31014000
	s_mov_b32 s30, 0x7ffffff0
	s_mov_b32 s31, 0x31014000
	s_mov_b32 s34, 0x7ffffff0
	s_mov_b32 s35, 0x31014000
	s_mov_b32 s38, 0x7ffffff0
	s_mov_b32 s39, 0x31014000
	s_mov_b32 s42, 0x7ffffff0
	s_mov_b32 s43, 0x31014000
	s_mov_b32 s46, 0x7ffffff0
	s_mov_b32 s47, 0x31014000
	s_mov_b32 s50, 0x7ffffff0
	s_mov_b32 s51, 0x31014000
	s_mov_b32 s54, 0x7ffffff0
	s_mov_b32 s55, 0x31014000
	s_mov_b32 s2, 0
	v_add_u32_e32 v236, 0x10000, v230
	v_add_u32_e32 v237, 0x10000, v231
	v_add_u32_e32 v238, 0x10000, v227
	v_add_u32_e32 v239, 0x10000, v223
	v_add_u32_e32 v240, 0x10000, v228
	v_add_u32_e32 v241, 0x10000, v229
	v_add_u32_e32 v242, 0x10000, v232
	v_add_u32_e32 v243, 0x10000, v233
	v_add_u32_e32 v244, 0x10000, v234
	v_add_u32_e32 v245, 0x10000, v235
	v_add_u32_e32 v246, 0x10000, v184
	v_add_u32_e32 v247, 0x10000, v185
.Lmain_loop:
	s_mul_i32 s3, s2, 24
	s_lshl_b32 s94, s3, 5
	s_add_i32 s94, s65, s94
	s_add_i32 s95, s94, 0x80
	s_lshl_b32 s95, s95, 9
	s_lshr_b32 s96, s3, 2
	s_mul_i32 s96, s96, 3
	s_mul_i32 s97, s3, 0x6000
	s_lshl_b32 s96, s96, 7
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[96:101], v[54:59], a[0:3], v210, v212 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(8) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[174:175], v224 offset:4096
	ds_read_b64 v[176:177], v225 offset:5120
	ds_read_b64 v[0:1], v236 offset:47424
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[102:107], v[54:59], a[4:7], v210, v212 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[2:3], v236 offset:47432
	ds_read_b64 v[4:5], v237 offset:47440
	s_add_i32 s97, s69, s97
	s_add_i32 s98, s97, 0x18000
	s_mov_b32 m0, s88
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[108:113], v[54:59], a[8:11], v210, v212 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s98 offen lds
	ds_read_b64 v[12:13], v236 offset:47552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[114:119], v[54:59], a[12:15], v210, v212 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[14:15], v236 offset:47560
	ds_read_b64 v[16:17], v237 offset:47568
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[6:11], v[54:59], a[16:19], v211, v212 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[18:19], v236 offset:47680
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[120:125], v[54:59], a[20:23], v211, v212 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[20:21], v236 offset:47688
	ds_read_b64 v[22:23], v237 offset:47696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[126:131], v[54:59], a[24:27], v211, v212 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s98 offen offset:1024 lds
	ds_read_b64 v[24:25], v236 offset:47808
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[132:137], v[54:59], a[28:31], v211, v212 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[26:27], v236 offset:47816
	ds_read_b64 v[28:29], v237 offset:47824
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[132:137], v[138:143], a[32:35], v211, v212 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[30:31], v236 offset:47936
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[126:131], v[138:143], a[36:39], v211, v212 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[32:33], v236 offset:47944
	ds_read_b64 v[34:35], v237 offset:47952
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[120:125], v[138:143], a[40:43], v211, v212 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s98 offen offset:2048 lds
	ds_read_b64 v[36:37], v236 offset:48064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[6:11], v[138:143], a[44:47], v211, v212 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[38:39], v236 offset:48072
	ds_read_b64 v[40:41], v237 offset:48080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[114:119], v[138:143], a[48:51], v210, v212 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[42:43], v236 offset:48192
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[108:113], v[138:143], a[52:55], v210, v212 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[44:45], v236 offset:48200
	ds_read_b64 v[46:47], v237 offset:48208
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[102:107], v[138:143], a[56:59], v210, v212 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s98 offen offset:3072 lds
	ds_read_b64 v[48:49], v236 offset:48320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[96:101], v[138:143], a[60:63], v210, v212 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[50:51], v236 offset:48328
	ds_read_b64 v[52:53], v237 offset:48336
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[96:101], v[144:149], a[64:67], v210, v212 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[54:57], v226 offset:55296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[102:107], v[144:149], a[68:71], v210, v212 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[58:59], v238 offset:6144
	s_mov_b32 m0, s89
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[108:113], v[144:149], a[72:75], v210, v212 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[8:11], s98 offen lds
	ds_read_b128 v[66:69], v226 offset:56320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[114:119], v[144:149], a[76:79], v210, v212 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[70:71], v238 offset:6656
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[6:11], v[144:149], a[80:83], v211, v212 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[72:75], v226 offset:57344
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[120:125], v[144:149], a[84:87], v211, v212 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[76:77], v238 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[126:131], v[144:149], a[88:91], v211, v212 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[8:11], s98 offen offset:1024 lds
	ds_read_b128 v[78:81], v226 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[132:137], v[144:149], a[92:95], v211, v212 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[82:83], v238 offset:7680
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[132:137], v[150:155], a[96:99], v211, v212 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[84:87], v226 offset:59392
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[126:131], v[150:155], a[100:103], v211, v212 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[88:89], v238 offset:8192
	s_add_i32 s95, s64, s95
	s_mov_b32 m0, s90
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[120:125], v[150:155], a[104:107], v211, v212 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[12:15], s95 offen lds
	ds_read_b128 v[90:93], v226 offset:60416
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[6:11], v[150:155], a[108:111], v211, v212 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[94:95], v238 offset:8704
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[114:119], v[150:155], a[112:115], v210, v212 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[138:141], v226 offset:61440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[108:113], v[150:155], a[116:119], v210, v212 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[142:143], v238 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[102:107], v[150:155], a[120:123], v210, v212 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[24:27], s95 offen offset:1024 lds
	ds_read_b128 v[144:147], v226 offset:62464
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[96:101], v[150:155], a[124:127], v210, v212 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[148:149], v238 offset:9728
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[96:101], v[60:65], a[128:131], v210, v213 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[102:107], v[60:65], a[132:135], v210, v213 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[108:113], v[60:65], a[136:139], v210, v213 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[114:119], v[60:65], a[140:143], v210, v213 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s95, s56, s96
	s_add_i32 s96, s95, 0x180
	s_mov_b32 m0, s80
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[6:11], v[60:65], a[144:147], v211, v213 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[20:23], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[120:125], v[60:65], a[148:151], v211, v213 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[126:131], v[60:65], a[152:155], v211, v213 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s81
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[132:137], v[60:65], a[156:159], v211, v213 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[28:31], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[132:137], v[156:161], a[160:163], v211, v213 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[126:131], v[156:161], a[164:167], v211, v213 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[120:125], v[156:161], a[168:171], v211, v213 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s82
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[6:11], v[156:161], a[172:175], v211, v213 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[32:35], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[114:119], v[156:161], a[176:179], v210, v213 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[108:113], v[156:161], a[180:183], v210, v213 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s83
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[102:107], v[156:161], a[184:187], v210, v213 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[36:39], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[96:101], v[156:161], a[188:191], v210, v213 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[96:101], v[162:167], a[192:195], v210, v213 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[102:107], v[162:167], a[196:199], v210, v213 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s84
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[108:113], v[162:167], a[200:203], v210, v213 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[40:43], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[114:119], v[162:167], a[204:207], v210, v213 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[6:11], v[162:167], a[208:211], v211, v213 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s85
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[120:125], v[162:167], a[212:215], v211, v213 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[44:47], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[126:131], v[162:167], a[216:219], v211, v213 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[132:137], v[162:167], a[220:223], v211, v213 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[132:137], v[168:173], a[224:227], v211, v213 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s86
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[126:131], v[168:173], a[228:231], v211, v213 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[48:51], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[120:125], v[168:173], a[232:235], v211, v213 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[6:11], v[168:173], a[236:239], v211, v213 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s87
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[114:119], v[168:173], a[240:243], v210, v213 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[52:55], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[108:113], v[168:173], a[244:247], v210, v213 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[102:107], v[168:173], a[248:251], v210, v213 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[96:101], v[168:173], a[252:255], v210, v213 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s96, s3, 5
	s_add_i32 s98, s94, 0xa0
	s_lshl_b32 s98, s98, 9
	s_lshr_b32 s96, s96, 2
	s_mul_i32 s96, s96, 3
	s_lshl_b32 s96, s96, 7
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[0:5], v[54:59], a[0:3], v174, v176 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(16) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[178:179], v224
	ds_read_b64 v[180:181], v225 offset:1024
	ds_read_b64 v[6:7], v239 offset:14368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[12:17], v[54:59], a[4:7], v174, v176 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[8:9], v239 offset:14376
	ds_read_b64 v[10:11], v239 offset:14384
	s_add_i32 s99, s97, 0x1e000
	s_mov_b32 m0, s91
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:23], v[54:59], a[8:11], v174, v176 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s99 offen lds
	ds_read_b64 v[60:61], v239 offset:14496
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[24:29], v[54:59], a[12:15], v174, v176 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[62:63], v239 offset:14504
	ds_read_b64 v[64:65], v239 offset:14512
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[30:35], v[54:59], a[16:19], v175, v176 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[96:97], v239 offset:14624
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[36:41], v[54:59], a[20:23], v175, v176 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[98:99], v239 offset:14632
	ds_read_b64 v[100:101], v239 offset:14640
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[42:47], v[54:59], a[24:27], v175, v176 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s99 offen offset:1024 lds
	ds_read_b64 v[102:103], v239 offset:14752
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[48:53], v[54:59], a[28:31], v175, v176 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[104:105], v239 offset:14760
	ds_read_b64 v[106:107], v239 offset:14768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[48:53], v[66:71], a[32:35], v175, v176 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[54:55], v239 offset:14880
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[42:47], v[66:71], a[36:39], v175, v176 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[56:57], v239 offset:14888
	ds_read_b64 v[58:59], v239 offset:14896
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[36:41], v[66:71], a[40:43], v175, v176 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s99 offen offset:2048 lds
	ds_read_b64 v[108:109], v239 offset:15008
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[30:35], v[66:71], a[44:47], v175, v176 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[110:111], v239 offset:15016
	ds_read_b64 v[112:113], v239 offset:15024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[24:29], v[66:71], a[48:51], v174, v176 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[114:115], v239 offset:15136
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[18:23], v[66:71], a[52:55], v174, v176 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[116:117], v239 offset:15144
	ds_read_b64 v[118:119], v239 offset:15152
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[12:17], v[66:71], a[56:59], v174, v176 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s99 offen offset:3072 lds
	ds_read_b64 v[120:121], v239 offset:15264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[0:5], v[66:71], a[60:63], v174, v176 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[122:123], v239 offset:15272
	ds_read_b64 v[124:125], v239 offset:15280
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[0:5], v[72:77], a[64:67], v174, v176 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[66:69], v226 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[12:17], v[72:77], a[68:71], v174, v176 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[70:71], v227 offset:22528
	s_mov_b32 m0, s92
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[18:23], v[72:77], a[72:75], v174, v176 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[8:11], s99 offen lds
	ds_read_b128 v[126:129], v226 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[24:29], v[72:77], a[76:79], v174, v176 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[130:131], v227 offset:23040
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[30:35], v[72:77], a[80:83], v175, v176 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[132:135], v226 offset:8192
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[36:41], v[72:77], a[84:87], v175, v176 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[136:137], v227 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[42:47], v[72:77], a[88:91], v175, v176 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[8:11], s99 offen offset:1024 lds
	ds_read_b128 v[150:153], v226 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[48:53], v[72:77], a[92:95], v175, v176 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[154:155], v227 offset:24064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[48:53], v[78:83], a[96:99], v175, v176 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[72:75], v226 offset:10240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[42:47], v[78:83], a[100:103], v175, v176 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[76:77], v227 offset:24576
	s_add_i32 s98, s64, s98
	s_mov_b32 m0, s93
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[36:41], v[78:83], a[104:107], v175, v176 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[12:15], s98 offen lds
	ds_read_b128 v[156:159], v226 offset:11264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[30:35], v[78:83], a[108:111], v175, v176 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[160:161], v227 offset:25088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[24:29], v[78:83], a[112:115], v174, v176 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[162:165], v226 offset:12288
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[18:23], v[78:83], a[116:119], v174, v176 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[166:167], v227 offset:25600
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[12:17], v[78:83], a[120:123], v174, v176 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[24:27], s98 offen offset:1024 lds
	ds_read_b128 v[168:171], v226 offset:13312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[0:5], v[78:83], a[124:127], v174, v176 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[172:173], v227 offset:26112
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[0:5], v[84:89], a[128:131], v174, v177 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[12:17], v[84:89], a[132:135], v174, v177 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[18:23], v[84:89], a[136:139], v174, v177 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[24:29], v[84:89], a[140:143], v174, v177 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s96, s56, s96
	s_add_i32 s96, s96, 0x80
	s_mov_b32 m0, s66
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[30:35], v[84:89], a[144:147], v175, v177 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[20:23], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[36:41], v[84:89], a[148:151], v175, v177 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[42:47], v[84:89], a[152:155], v175, v177 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s70
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[48:53], v[84:89], a[156:159], v175, v177 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[28:31], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[48:53], v[90:95], a[160:163], v175, v177 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[42:47], v[90:95], a[164:167], v175, v177 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[36:41], v[90:95], a[168:171], v175, v177 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s71
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[30:35], v[90:95], a[172:175], v175, v177 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[32:35], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[24:29], v[90:95], a[176:179], v174, v177 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[18:23], v[90:95], a[180:183], v174, v177 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s72
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[12:17], v[90:95], a[184:187], v174, v177 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[36:39], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[0:5], v[90:95], a[188:191], v174, v177 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[0:5], v[138:143], a[192:195], v174, v177 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[12:17], v[138:143], a[196:199], v174, v177 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s73
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[18:23], v[138:143], a[200:203], v174, v177 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[40:43], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[24:29], v[138:143], a[204:207], v174, v177 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[30:35], v[138:143], a[208:211], v175, v177 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s74
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[36:41], v[138:143], a[212:215], v175, v177 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[44:47], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[42:47], v[138:143], a[216:219], v175, v177 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[48:53], v[138:143], a[220:223], v175, v177 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[48:53], v[144:149], a[224:227], v175, v177 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s75
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[42:47], v[144:149], a[228:231], v175, v177 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[48:51], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[36:41], v[144:149], a[232:235], v175, v177 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[30:35], v[144:149], a[236:239], v175, v177 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s76
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[24:29], v[144:149], a[240:243], v174, v177 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[52:55], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[18:23], v[144:149], a[244:247], v174, v177 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[12:17], v[144:149], a[248:251], v174, v177 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[0:5], v[144:149], a[252:255], v174, v177 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s96, s3, 6
	s_add_i32 s98, s94, 0xc0
	s_lshl_b32 s98, s98, 9
	s_lshr_b32 s96, s96, 2
	s_mul_i32 s96, s96, 3
	s_lshl_b32 s96, s96, 7
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[6:11], v[66:71], a[0:3], v178, v180 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(16) lgkmcnt(0)
	s_barrier
	ds_read_b128 v[18:21], v240 offset:47360
	ds_read_b128 v[22:25], v240 offset:47376
	ds_read_b64 v[174:175], v224 offset:2048
	ds_read_b64 v[176:177], v225 offset:3072
	ds_read_b64 v[0:1], v239 offset:47360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[60:65], v[66:71], a[4:7], v178, v180 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[2:3], v239 offset:47368
	ds_read_b64 v[4:5], v239 offset:47376
	s_add_i32 s99, s97, 0x24000
	s_mov_b32 m0, s77
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[96:101], v[66:71], a[8:11], v178, v180 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s99 offen lds
	ds_read_b64 v[12:13], v239 offset:47488
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[102:107], v[66:71], a[12:15], v178, v180 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[14:15], v239 offset:47496
	ds_read_b64 v[16:17], v239 offset:47504
	s_waitcnt lgkmcnt(9)
	ds_write_b64 v241, v[18:19] offset:14336
	ds_write_b64 v241, v[20:21] offset:14344
	s_waitcnt lgkmcnt(10)
	ds_write_b64 v241, v[22:23] offset:14352
	ds_write_b64 v241, v[24:25] offset:14360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[54:59], v[66:71], a[16:19], v179, v180 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[18:19], v239 offset:47616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[108:113], v[66:71], a[20:23], v179, v180 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[20:21], v239 offset:47624
	ds_read_b64 v[22:23], v239 offset:47632
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[114:119], v[66:71], a[24:27], v179, v180 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s99 offen offset:1024 lds
	ds_read_b64 v[24:25], v239 offset:47744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[120:125], v[66:71], a[28:31], v179, v180 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[26:27], v239 offset:47752
	ds_read_b64 v[28:29], v239 offset:47760
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[120:125], v[126:131], a[32:35], v179, v180 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[30:31], v239 offset:47872
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[114:119], v[126:131], a[36:39], v179, v180 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[32:33], v239 offset:47880
	ds_read_b64 v[34:35], v239 offset:47888
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[108:113], v[126:131], a[40:43], v179, v180 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s99 offen offset:2048 lds
	ds_read_b64 v[36:37], v239 offset:48000
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[54:59], v[126:131], a[44:47], v179, v180 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[38:39], v239 offset:48008
	ds_read_b64 v[40:41], v239 offset:48016
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[102:107], v[126:131], a[48:51], v178, v180 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[42:43], v239 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[96:101], v[126:131], a[52:55], v178, v180 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[44:45], v239 offset:48136
	ds_read_b64 v[46:47], v239 offset:48144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[60:65], v[126:131], a[56:59], v178, v180 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s99 offen offset:3072 lds
	ds_read_b64 v[48:49], v239 offset:48256
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[6:11], v[126:131], a[60:63], v178, v180 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[50:51], v239 offset:48264
	ds_read_b64 v[52:53], v239 offset:48272
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[6:11], v[132:137], a[64:67], v178, v180 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[66:69], v226 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[60:65], v[132:137], a[68:71], v178, v180 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[70:71], v227 offset:47104
	s_mov_b32 m0, s78
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[96:101], v[132:137], a[72:75], v178, v180 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[8:11], s99 offen lds
	ds_read_b128 v[78:81], v226 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[102:107], v[132:137], a[76:79], v178, v180 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[82:83], v227 offset:47616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[54:59], v[132:137], a[80:83], v179, v180 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[84:87], v226 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[108:113], v[132:137], a[84:87], v179, v180 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[88:89], v227 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[114:119], v[132:137], a[88:91], v179, v180 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[8:11], s99 offen offset:1024 lds
	ds_read_b128 v[90:93], v226 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[120:125], v[132:137], a[92:95], v179, v180 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[94:95], v227 offset:48640
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[120:125], v[150:155], a[96:99], v179, v180 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[126:129], v226 offset:34816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[114:119], v[150:155], a[100:103], v179, v180 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[130:131], v227 offset:49152
	s_add_i32 s98, s64, s98
	s_mov_b32 m0, s64
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[108:113], v[150:155], a[104:107], v179, v180 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[12:15], s98 offen lds
	ds_read_b128 v[132:135], v226 offset:35840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[54:59], v[150:155], a[108:111], v179, v180 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[136:137], v227 offset:49664
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[102:107], v[150:155], a[112:115], v178, v180 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[138:141], v226 offset:36864
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[96:101], v[150:155], a[116:119], v178, v180 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[142:143], v227 offset:50176
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[60:65], v[150:155], a[120:123], v178, v180 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[24:27], s98 offen offset:1024 lds
	ds_read_b128 v[144:147], v226 offset:37888
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[6:11], v[150:155], a[124:127], v178, v180 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[148:149], v227 offset:50688
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[6:11], v[72:77], a[128:131], v178, v181 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[60:65], v[72:77], a[132:135], v178, v181 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[96:101], v[72:77], a[136:139], v178, v181 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[102:107], v[72:77], a[140:143], v178, v181 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s96, s56, s96
	s_add_i32 s96, s96, 0x100
	s_mov_b32 m0, s80
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[54:59], v[72:77], a[144:147], v179, v181 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[20:23], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[108:113], v[72:77], a[148:151], v179, v181 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[114:119], v[72:77], a[152:155], v179, v181 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s81
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[120:125], v[72:77], a[156:159], v179, v181 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[28:31], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[120:125], v[156:161], a[160:163], v179, v181 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[114:119], v[156:161], a[164:167], v179, v181 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[108:113], v[156:161], a[168:171], v179, v181 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s82
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[54:59], v[156:161], a[172:175], v179, v181 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[32:35], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[102:107], v[156:161], a[176:179], v178, v181 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[96:101], v[156:161], a[180:183], v178, v181 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s83
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[60:65], v[156:161], a[184:187], v178, v181 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[36:39], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[6:11], v[156:161], a[188:191], v178, v181 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[6:11], v[162:167], a[192:195], v178, v181 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[60:65], v[162:167], a[196:199], v178, v181 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s84
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[96:101], v[162:167], a[200:203], v178, v181 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[40:43], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[102:107], v[162:167], a[204:207], v178, v181 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[54:59], v[162:167], a[208:211], v179, v181 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s85
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[108:113], v[162:167], a[212:215], v179, v181 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[44:47], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[114:119], v[162:167], a[216:219], v179, v181 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[120:125], v[162:167], a[220:223], v179, v181 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[120:125], v[168:173], a[224:227], v179, v181 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s86
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[114:119], v[168:173], a[228:231], v179, v181 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[48:51], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[108:113], v[168:173], a[232:235], v179, v181 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[54:59], v[168:173], a[236:239], v179, v181 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s87
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[102:107], v[168:173], a[240:243], v178, v181 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[52:55], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[96:101], v[168:173], a[244:247], v178, v181 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[60:65], v[168:173], a[248:251], v178, v181 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[6:11], v[168:173], a[252:255], v178, v181 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s96, s94, 0xe0
	s_lshl_b32 s96, s96, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[0:5], v[66:71], a[0:3], v174, v176 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(16) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[178:179], v224 offset:4096
	ds_read_b64 v[180:181], v225 offset:5120
	ds_read_b64 v[6:7], v242 offset:47456
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[12:17], v[66:71], a[4:7], v174, v176 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[8:9], v243 offset:47464
	ds_read_b64 v[10:11], v243 offset:47472
	s_add_i32 s98, s97, 0x2a000
	s_mov_b32 m0, s88
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:23], v[66:71], a[8:11], v174, v176 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s98 offen lds
	ds_read_b64 v[54:55], v242 offset:47584
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[24:29], v[66:71], a[12:15], v174, v176 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[56:57], v243 offset:47592
	ds_read_b64 v[58:59], v243 offset:47600
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[30:35], v[66:71], a[16:19], v175, v176 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[60:61], v242 offset:47712
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[36:41], v[66:71], a[20:23], v175, v176 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[62:63], v243 offset:47720
	ds_read_b64 v[64:65], v243 offset:47728
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[42:47], v[66:71], a[24:27], v175, v176 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s98 offen offset:1024 lds
	ds_read_b64 v[72:73], v242 offset:47840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[48:53], v[66:71], a[28:31], v175, v176 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[74:75], v243 offset:47848
	ds_read_b64 v[76:77], v243 offset:47856
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[48:53], v[78:83], a[32:35], v175, v176 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[66:67], v242 offset:47968
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[42:47], v[78:83], a[36:39], v175, v176 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[68:69], v243 offset:47976
	ds_read_b64 v[70:71], v243 offset:47984
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[36:41], v[78:83], a[40:43], v175, v176 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s98 offen offset:2048 lds
	ds_read_b64 v[96:97], v242 offset:48096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[30:35], v[78:83], a[44:47], v175, v176 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[98:99], v243 offset:48104
	ds_read_b64 v[100:101], v243 offset:48112
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[24:29], v[78:83], a[48:51], v174, v176 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[102:103], v242 offset:48224
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[18:23], v[78:83], a[52:55], v174, v176 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[104:105], v243 offset:48232
	ds_read_b64 v[106:107], v243 offset:48240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[12:17], v[78:83], a[56:59], v174, v176 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s98 offen offset:3072 lds
	ds_read_b64 v[108:109], v242 offset:48352
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[0:5], v[78:83], a[60:63], v174, v176 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[110:111], v243 offset:48360
	ds_read_b64 v[112:113], v243 offset:48368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[0:5], v[84:89], a[64:67], v174, v176 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[78:81], v226 offset:55296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[12:17], v[84:89], a[68:71], v174, v176 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[82:83], v238 offset:6144
	s_mov_b32 m0, s89
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[18:23], v[84:89], a[72:75], v174, v176 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[8:11], s98 offen lds
	ds_read_b128 v[114:117], v226 offset:56320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[24:29], v[84:89], a[76:79], v174, v176 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[118:119], v238 offset:6656
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[30:35], v[84:89], a[80:83], v175, v176 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[120:123], v226 offset:57344
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[36:41], v[84:89], a[84:87], v175, v176 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[124:125], v238 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[42:47], v[84:89], a[88:91], v175, v176 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[8:11], s98 offen offset:1024 lds
	ds_read_b128 v[150:153], v226 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[48:53], v[84:89], a[92:95], v175, v176 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[154:155], v238 offset:7680
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[48:53], v[90:95], a[96:99], v175, v176 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[84:87], v226 offset:59392
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[42:47], v[90:95], a[100:103], v175, v176 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[88:89], v238 offset:8192
	s_add_i32 s96, s64, s96
	s_mov_b32 m0, s90
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[36:41], v[90:95], a[104:107], v175, v176 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[12:15], s96 offen lds
	ds_read_b128 v[156:159], v226 offset:60416
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[30:35], v[90:95], a[108:111], v175, v176 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[160:161], v238 offset:8704
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[24:29], v[90:95], a[112:115], v174, v176 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[162:165], v226 offset:61440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[18:23], v[90:95], a[116:119], v174, v176 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[166:167], v238 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[12:17], v[90:95], a[120:123], v174, v176 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[24:27], s96 offen offset:1024 lds
	ds_read_b128 v[168:171], v226 offset:62464
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[0:5], v[90:95], a[124:127], v174, v176 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[172:173], v238 offset:9728
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[0:5], v[126:131], a[128:131], v174, v177 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[12:17], v[126:131], a[132:135], v174, v177 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[18:23], v[126:131], a[136:139], v174, v177 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[24:29], v[126:131], a[140:143], v174, v177 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[30:35], v[126:131], a[144:147], v175, v177 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[36:41], v[126:131], a[148:151], v175, v177 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[42:47], v[126:131], a[152:155], v175, v177 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[48:53], v[126:131], a[156:159], v175, v177 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[48:53], v[132:137], a[160:163], v175, v177 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[42:47], v[132:137], a[164:167], v175, v177 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[36:41], v[132:137], a[168:171], v175, v177 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[30:35], v[132:137], a[172:175], v175, v177 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[24:29], v[132:137], a[176:179], v174, v177 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[18:23], v[132:137], a[180:183], v174, v177 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[12:17], v[132:137], a[184:187], v174, v177 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[0:5], v[132:137], a[188:191], v174, v177 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[0:5], v[138:143], a[192:195], v174, v177 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[12:17], v[138:143], a[196:199], v174, v177 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[18:23], v[138:143], a[200:203], v174, v177 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[24:29], v[138:143], a[204:207], v174, v177 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[30:35], v[138:143], a[208:211], v175, v177 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[36:41], v[138:143], a[212:215], v175, v177 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[42:47], v[138:143], a[216:219], v175, v177 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[48:53], v[138:143], a[220:223], v175, v177 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[48:53], v[144:149], a[224:227], v175, v177 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[42:47], v[144:149], a[228:231], v175, v177 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[36:41], v[144:149], a[232:235], v175, v177 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[30:35], v[144:149], a[236:239], v175, v177 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[24:29], v[144:149], a[240:243], v174, v177 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[18:23], v[144:149], a[244:247], v174, v177 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[12:17], v[144:149], a[248:251], v174, v177 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[0:5], v[144:149], a[252:255], v174, v177 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s96, s94, 0x100
	s_lshl_b32 s96, s96, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[6:11], v[78:83], a[0:3], v178, v180 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(8) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[174:175], v224
	ds_read_b64 v[176:177], v225 offset:1024
	ds_read_b64 v[0:1], v244 offset:14400
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[54:59], v[78:83], a[4:7], v178, v180 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[2:3], v244 offset:14408
	ds_read_b64 v[4:5], v245 offset:14416
	s_add_i32 s98, s97, 0x30000
	s_mov_b32 m0, s91
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[60:65], v[78:83], a[8:11], v178, v180 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s98 offen lds
	ds_read_b64 v[12:13], v244 offset:14528
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[72:77], v[78:83], a[12:15], v178, v180 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[14:15], v244 offset:14536
	ds_read_b64 v[16:17], v245 offset:14544
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[66:71], v[78:83], a[16:19], v179, v180 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[18:19], v244 offset:14656
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[96:101], v[78:83], a[20:23], v179, v180 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[20:21], v244 offset:14664
	ds_read_b64 v[22:23], v245 offset:14672
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[102:107], v[78:83], a[24:27], v179, v180 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s98 offen offset:1024 lds
	ds_read_b64 v[24:25], v244 offset:14784
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[108:113], v[78:83], a[28:31], v179, v180 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[26:27], v244 offset:14792
	ds_read_b64 v[28:29], v245 offset:14800
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[108:113], v[114:119], a[32:35], v179, v180 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[30:31], v244 offset:14912
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[102:107], v[114:119], a[36:39], v179, v180 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[32:33], v244 offset:14920
	ds_read_b64 v[34:35], v245 offset:14928
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[96:101], v[114:119], a[40:43], v179, v180 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s98 offen offset:2048 lds
	ds_read_b64 v[36:37], v244 offset:15040
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[66:71], v[114:119], a[44:47], v179, v180 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[38:39], v244 offset:15048
	ds_read_b64 v[40:41], v245 offset:15056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[72:77], v[114:119], a[48:51], v178, v180 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[42:43], v244 offset:15168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[60:65], v[114:119], a[52:55], v178, v180 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[44:45], v244 offset:15176
	ds_read_b64 v[46:47], v245 offset:15184
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[54:59], v[114:119], a[56:59], v178, v180 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s98 offen offset:3072 lds
	ds_read_b64 v[48:49], v244 offset:15296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[6:11], v[114:119], a[60:63], v178, v180 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[50:51], v244 offset:15304
	ds_read_b64 v[52:53], v245 offset:15312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[6:11], v[120:125], a[64:67], v178, v180 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[78:81], v226 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[54:59], v[120:125], a[68:71], v178, v180 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[82:83], v227 offset:22528
	s_mov_b32 m0, s92
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[60:65], v[120:125], a[72:75], v178, v180 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[8:11], s98 offen lds
	ds_read_b128 v[90:93], v226 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[72:77], v[120:125], a[76:79], v178, v180 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[94:95], v227 offset:23040
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[66:71], v[120:125], a[80:83], v179, v180 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[114:117], v226 offset:8192
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[96:101], v[120:125], a[84:87], v179, v180 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[118:119], v227 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[102:107], v[120:125], a[88:91], v179, v180 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[8:11], s98 offen offset:1024 lds
	ds_read_b128 v[126:129], v226 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[108:113], v[120:125], a[92:95], v179, v180 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[130:131], v227 offset:24064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[108:113], v[150:155], a[96:99], v179, v180 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[120:123], v226 offset:10240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[102:107], v[150:155], a[100:103], v179, v180 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[124:125], v227 offset:24576
	s_add_i32 s96, s64, s96
	s_mov_b32 m0, s93
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[96:101], v[150:155], a[104:107], v179, v180 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[12:15], s96 offen lds
	ds_read_b128 v[132:135], v226 offset:11264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[66:71], v[150:155], a[108:111], v179, v180 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[136:137], v227 offset:25088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[72:77], v[150:155], a[112:115], v178, v180 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[138:141], v226 offset:12288
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[60:65], v[150:155], a[116:119], v178, v180 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[142:143], v227 offset:25600
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[54:59], v[150:155], a[120:123], v178, v180 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[24:27], s96 offen offset:1024 lds
	ds_read_b128 v[144:147], v226 offset:13312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[6:11], v[150:155], a[124:127], v178, v180 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[148:149], v227 offset:26112
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[6:11], v[84:89], a[128:131], v178, v181 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[54:59], v[84:89], a[132:135], v178, v181 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[60:65], v[84:89], a[136:139], v178, v181 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[72:77], v[84:89], a[140:143], v178, v181 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s96, s95, 0x300
	s_mov_b32 m0, s66
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[66:71], v[84:89], a[144:147], v179, v181 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[20:23], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[96:101], v[84:89], a[148:151], v179, v181 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[102:107], v[84:89], a[152:155], v179, v181 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s70
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[108:113], v[84:89], a[156:159], v179, v181 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[28:31], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[108:113], v[156:161], a[160:163], v179, v181 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[102:107], v[156:161], a[164:167], v179, v181 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[96:101], v[156:161], a[168:171], v179, v181 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s71
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[66:71], v[156:161], a[172:175], v179, v181 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[32:35], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[72:77], v[156:161], a[176:179], v178, v181 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[60:65], v[156:161], a[180:183], v178, v181 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s72
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[54:59], v[156:161], a[184:187], v178, v181 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[36:39], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[6:11], v[156:161], a[188:191], v178, v181 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[6:11], v[162:167], a[192:195], v178, v181 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[54:59], v[162:167], a[196:199], v178, v181 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s73
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[60:65], v[162:167], a[200:203], v178, v181 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[40:43], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[72:77], v[162:167], a[204:207], v178, v181 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[66:71], v[162:167], a[208:211], v179, v181 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s74
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[96:101], v[162:167], a[212:215], v179, v181 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[44:47], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[102:107], v[162:167], a[216:219], v179, v181 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[108:113], v[162:167], a[220:223], v179, v181 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[108:113], v[168:173], a[224:227], v179, v181 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s75
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[102:107], v[168:173], a[228:231], v179, v181 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[48:51], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[96:101], v[168:173], a[232:235], v179, v181 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[66:71], v[168:173], a[236:239], v179, v181 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s76
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[72:77], v[168:173], a[240:243], v178, v181 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[52:55], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[60:65], v[168:173], a[244:247], v178, v181 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[54:59], v[168:173], a[248:251], v178, v181 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[6:11], v[168:173], a[252:255], v178, v181 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s96, s3, 9
	s_add_i32 s98, s94, 0x120
	s_lshl_b32 s98, s98, 9
	s_lshr_b32 s96, s96, 2
	s_mul_i32 s96, s96, 3
	s_lshl_b32 s96, s96, 7
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[0:5], v[78:83], a[0:3], v174, v176 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(16) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[178:179], v224 offset:2048
	ds_read_b64 v[180:181], v225 offset:3072
	ds_read_b64 v[6:7], v239 offset:47392
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[12:17], v[78:83], a[4:7], v174, v176 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[8:9], v239 offset:47400
	ds_read_b64 v[10:11], v239 offset:47408
	s_add_i32 s99, s97, 0x36000
	s_mov_b32 m0, s77
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:23], v[78:83], a[8:11], v174, v176 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s99 offen lds
	ds_read_b64 v[54:55], v239 offset:47520
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[24:29], v[78:83], a[12:15], v174, v176 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[56:57], v239 offset:47528
	ds_read_b64 v[58:59], v239 offset:47536
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[30:35], v[78:83], a[16:19], v175, v176 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[60:61], v239 offset:47648
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[36:41], v[78:83], a[20:23], v175, v176 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[62:63], v239 offset:47656
	ds_read_b64 v[64:65], v239 offset:47664
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[42:47], v[78:83], a[24:27], v175, v176 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s99 offen offset:1024 lds
	ds_read_b64 v[66:67], v239 offset:47776
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[48:53], v[78:83], a[28:31], v175, v176 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[68:69], v239 offset:47784
	ds_read_b64 v[70:71], v239 offset:47792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[48:53], v[90:95], a[32:35], v175, v176 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[72:73], v239 offset:47904
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[42:47], v[90:95], a[36:39], v175, v176 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[74:75], v239 offset:47912
	ds_read_b64 v[76:77], v239 offset:47920
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[36:41], v[90:95], a[40:43], v175, v176 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s99 offen offset:2048 lds
	ds_read_b64 v[78:79], v239 offset:48032
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[30:35], v[90:95], a[44:47], v175, v176 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[80:81], v239 offset:48040
	ds_read_b64 v[82:83], v239 offset:48048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[24:29], v[90:95], a[48:51], v174, v176 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[84:85], v239 offset:48160
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[18:23], v[90:95], a[52:55], v174, v176 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[86:87], v239 offset:48168
	ds_read_b64 v[88:89], v239 offset:48176
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[12:17], v[90:95], a[56:59], v174, v176 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s99 offen offset:3072 lds
	ds_read_b64 v[96:97], v239 offset:48288
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[0:5], v[90:95], a[60:63], v174, v176 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[98:99], v239 offset:48296
	ds_read_b64 v[100:101], v239 offset:48304
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[0:5], v[114:119], a[64:67], v174, v176 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[90:93], v226 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[12:17], v[114:119], a[68:71], v174, v176 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[94:95], v227 offset:47104
	s_mov_b32 m0, s78
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[18:23], v[114:119], a[72:75], v174, v176 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[8:11], s99 offen lds
	ds_read_b128 v[102:105], v226 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[24:29], v[114:119], a[76:79], v174, v176 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[106:107], v227 offset:47616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[30:35], v[114:119], a[80:83], v175, v176 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[108:111], v226 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[36:41], v[114:119], a[84:87], v175, v176 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[112:113], v227 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[42:47], v[114:119], a[88:91], v175, v176 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[8:11], s99 offen offset:1024 lds
	ds_read_b128 v[150:153], v226 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[48:53], v[114:119], a[92:95], v175, v176 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[154:155], v227 offset:48640
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[48:53], v[126:131], a[96:99], v175, v176 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[114:117], v226 offset:34816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[42:47], v[126:131], a[100:103], v175, v176 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[118:119], v227 offset:49152
	s_add_i32 s98, s64, s98
	s_mov_b32 m0, s64
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[36:41], v[126:131], a[104:107], v175, v176 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[12:15], s98 offen lds
	ds_read_b128 v[156:159], v226 offset:35840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[30:35], v[126:131], a[108:111], v175, v176 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[160:161], v227 offset:49664
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[24:29], v[126:131], a[112:115], v174, v176 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[162:165], v226 offset:36864
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[18:23], v[126:131], a[116:119], v174, v176 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[166:167], v227 offset:50176
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[12:17], v[126:131], a[120:123], v174, v176 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[24:27], s98 offen offset:1024 lds
	ds_read_b128 v[168:171], v226 offset:37888
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[0:5], v[126:131], a[124:127], v174, v176 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[172:173], v227 offset:50688
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[0:5], v[120:125], a[128:131], v174, v177 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[12:17], v[120:125], a[132:135], v174, v177 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[18:23], v[120:125], a[136:139], v174, v177 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[24:29], v[120:125], a[140:143], v174, v177 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s96, s56, s96
	s_add_i32 s96, s96, 0x80
	s_mov_b32 m0, s80
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[30:35], v[120:125], a[144:147], v175, v177 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[20:23], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[36:41], v[120:125], a[148:151], v175, v177 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[42:47], v[120:125], a[152:155], v175, v177 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s81
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[48:53], v[120:125], a[156:159], v175, v177 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[28:31], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[48:53], v[132:137], a[160:163], v175, v177 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[42:47], v[132:137], a[164:167], v175, v177 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[36:41], v[132:137], a[168:171], v175, v177 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s82
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[30:35], v[132:137], a[172:175], v175, v177 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[32:35], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[24:29], v[132:137], a[176:179], v174, v177 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[18:23], v[132:137], a[180:183], v174, v177 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s83
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[12:17], v[132:137], a[184:187], v174, v177 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[36:39], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[0:5], v[132:137], a[188:191], v174, v177 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[0:5], v[138:143], a[192:195], v174, v177 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[12:17], v[138:143], a[196:199], v174, v177 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s84
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[18:23], v[138:143], a[200:203], v174, v177 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[40:43], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[24:29], v[138:143], a[204:207], v174, v177 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[30:35], v[138:143], a[208:211], v175, v177 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s85
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[36:41], v[138:143], a[212:215], v175, v177 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[44:47], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[42:47], v[138:143], a[216:219], v175, v177 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[48:53], v[138:143], a[220:223], v175, v177 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[48:53], v[144:149], a[224:227], v175, v177 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s86
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[42:47], v[144:149], a[228:231], v175, v177 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[48:51], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[36:41], v[144:149], a[232:235], v175, v177 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[30:35], v[144:149], a[236:239], v175, v177 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s87
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[24:29], v[144:149], a[240:243], v174, v177 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[52:55], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[18:23], v[144:149], a[244:247], v174, v177 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[12:17], v[144:149], a[248:251], v174, v177 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[0:5], v[144:149], a[252:255], v174, v177 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s96, s3, 10
	s_add_i32 s98, s94, 0x140
	s_lshl_b32 s98, s98, 9
	s_lshr_b32 s96, s96, 2
	s_mul_i32 s96, s96, 3
	s_lshl_b32 s96, s96, 7
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[6:11], v[90:95], a[0:3], v178, v180 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(16) lgkmcnt(0)
	s_barrier
	ds_read_b128 v[18:21], v240 offset:14336
	ds_read_b128 v[22:25], v240 offset:14352
	ds_read_b64 v[174:175], v224 offset:4096
	ds_read_b64 v[176:177], v225 offset:5120
	ds_read_b64 v[0:1], v239 offset:14336
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[54:59], v[90:95], a[4:7], v178, v180 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[2:3], v239 offset:14344
	ds_read_b64 v[4:5], v239 offset:14352
	s_add_i32 s99, s97, 0x3c000
	s_mov_b32 m0, s88
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[60:65], v[90:95], a[8:11], v178, v180 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s99 offen lds
	ds_read_b64 v[12:13], v239 offset:14464
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:71], v[90:95], a[12:15], v178, v180 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[14:15], v239 offset:14472
	ds_read_b64 v[16:17], v239 offset:14480
	s_waitcnt lgkmcnt(9)
	ds_write_b64 v241, v[18:19] offset:14336
	ds_write_b64 v241, v[20:21] offset:14344
	s_waitcnt lgkmcnt(10)
	ds_write_b64 v241, v[22:23] offset:14352
	ds_write_b64 v241, v[24:25] offset:14360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[72:77], v[90:95], a[16:19], v179, v180 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[18:19], v239 offset:14592
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[78:83], v[90:95], a[20:23], v179, v180 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[20:21], v239 offset:14600
	ds_read_b64 v[22:23], v239 offset:14608
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[84:89], v[90:95], a[24:27], v179, v180 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s99 offen offset:1024 lds
	ds_read_b64 v[24:25], v239 offset:14720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[96:101], v[90:95], a[28:31], v179, v180 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[26:27], v239 offset:14728
	ds_read_b64 v[28:29], v239 offset:14736
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[96:101], v[102:107], a[32:35], v179, v180 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[30:31], v239 offset:14848
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[84:89], v[102:107], a[36:39], v179, v180 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[32:33], v239 offset:14856
	ds_read_b64 v[34:35], v239 offset:14864
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[78:83], v[102:107], a[40:43], v179, v180 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s99 offen offset:2048 lds
	ds_read_b64 v[36:37], v239 offset:14976
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[72:77], v[102:107], a[44:47], v179, v180 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[38:39], v239 offset:14984
	ds_read_b64 v[40:41], v239 offset:14992
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[66:71], v[102:107], a[48:51], v178, v180 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[42:43], v239 offset:15104
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[60:65], v[102:107], a[52:55], v178, v180 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[44:45], v239 offset:15112
	ds_read_b64 v[46:47], v239 offset:15120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[54:59], v[102:107], a[56:59], v178, v180 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s99 offen offset:3072 lds
	ds_read_b64 v[48:49], v239 offset:15232
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[6:11], v[102:107], a[60:63], v178, v180 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[50:51], v239 offset:15240
	ds_read_b64 v[52:53], v239 offset:15248
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[6:11], v[108:113], a[64:67], v178, v180 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[90:93], v226 offset:55296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[54:59], v[108:113], a[68:71], v178, v180 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[94:95], v238 offset:6144
	s_mov_b32 m0, s89
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[60:65], v[108:113], a[72:75], v178, v180 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[8:11], s99 offen lds
	ds_read_b128 v[102:105], v226 offset:56320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[66:71], v[108:113], a[76:79], v178, v180 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[106:107], v238 offset:6656
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[72:77], v[108:113], a[80:83], v179, v180 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[120:123], v226 offset:57344
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[78:83], v[108:113], a[84:87], v179, v180 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[124:125], v238 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[84:89], v[108:113], a[88:91], v179, v180 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[8:11], s99 offen offset:1024 lds
	ds_read_b128 v[126:129], v226 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[96:101], v[108:113], a[92:95], v179, v180 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[130:131], v238 offset:7680
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[96:101], v[150:155], a[96:99], v179, v180 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[108:111], v226 offset:59392
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[84:89], v[150:155], a[100:103], v179, v180 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[112:113], v238 offset:8192
	s_add_i32 s98, s64, s98
	s_mov_b32 m0, s90
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[78:83], v[150:155], a[104:107], v179, v180 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[12:15], s98 offen lds
	ds_read_b128 v[132:135], v226 offset:60416
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[72:77], v[150:155], a[108:111], v179, v180 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[136:137], v238 offset:8704
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[66:71], v[150:155], a[112:115], v178, v180 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[138:141], v226 offset:61440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[60:65], v[150:155], a[116:119], v178, v180 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[142:143], v238 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[54:59], v[150:155], a[120:123], v178, v180 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[24:27], s98 offen offset:1024 lds
	ds_read_b128 v[144:147], v226 offset:62464
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[6:11], v[150:155], a[124:127], v178, v180 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[148:149], v238 offset:9728
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[6:11], v[114:119], a[128:131], v178, v181 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[54:59], v[114:119], a[132:135], v178, v181 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[60:65], v[114:119], a[136:139], v178, v181 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[66:71], v[114:119], a[140:143], v178, v181 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s96, s56, s96
	s_add_i32 s96, s96, 0x100
	s_mov_b32 m0, s66
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[72:77], v[114:119], a[144:147], v179, v181 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[20:23], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[78:83], v[114:119], a[148:151], v179, v181 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[84:89], v[114:119], a[152:155], v179, v181 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s70
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[96:101], v[114:119], a[156:159], v179, v181 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[28:31], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[96:101], v[156:161], a[160:163], v179, v181 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[84:89], v[156:161], a[164:167], v179, v181 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[78:83], v[156:161], a[168:171], v179, v181 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s71
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[72:77], v[156:161], a[172:175], v179, v181 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[32:35], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[66:71], v[156:161], a[176:179], v178, v181 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[60:65], v[156:161], a[180:183], v178, v181 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s72
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[54:59], v[156:161], a[184:187], v178, v181 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[36:39], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[6:11], v[156:161], a[188:191], v178, v181 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[6:11], v[162:167], a[192:195], v178, v181 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[54:59], v[162:167], a[196:199], v178, v181 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s73
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[60:65], v[162:167], a[200:203], v178, v181 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[40:43], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[66:71], v[162:167], a[204:207], v178, v181 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[72:77], v[162:167], a[208:211], v179, v181 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s74
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[78:83], v[162:167], a[212:215], v179, v181 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[44:47], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[84:89], v[162:167], a[216:219], v179, v181 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[96:101], v[162:167], a[220:223], v179, v181 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[96:101], v[168:173], a[224:227], v179, v181 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s75
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[84:89], v[168:173], a[228:231], v179, v181 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[48:51], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[78:83], v[168:173], a[232:235], v179, v181 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[72:77], v[168:173], a[236:239], v179, v181 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s76
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[66:71], v[168:173], a[240:243], v178, v181 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[52:55], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[60:65], v[168:173], a[244:247], v178, v181 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[54:59], v[168:173], a[248:251], v178, v181 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[6:11], v[168:173], a[252:255], v178, v181 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s96, s94, 0x160
	s_lshl_b32 s96, s96, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[0:5], v[90:95], a[0:3], v174, v176 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(16) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[178:179], v224
	ds_read_b64 v[180:181], v225 offset:1024
	ds_read_b64 v[6:7], v246 offset:14432
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[12:17], v[90:95], a[4:7], v174, v176 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[8:9], v247 offset:14440
	ds_read_b64 v[10:11], v247 offset:14448
	s_add_i32 s98, s97, 0x42000
	s_mov_b32 m0, s91
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:23], v[90:95], a[8:11], v174, v176 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s98 offen lds
	ds_read_b64 v[54:55], v246 offset:14560
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[24:29], v[90:95], a[12:15], v174, v176 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[56:57], v247 offset:14568
	ds_read_b64 v[58:59], v247 offset:14576
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[30:35], v[90:95], a[16:19], v175, v176 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[60:61], v246 offset:14688
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[36:41], v[90:95], a[20:23], v175, v176 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[62:63], v247 offset:14696
	ds_read_b64 v[64:65], v247 offset:14704
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[42:47], v[90:95], a[24:27], v175, v176 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s98 offen offset:1024 lds
	ds_read_b64 v[66:67], v246 offset:14816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[48:53], v[90:95], a[28:31], v175, v176 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[68:69], v247 offset:14824
	ds_read_b64 v[70:71], v247 offset:14832
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[48:53], v[102:107], a[32:35], v175, v176 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[72:73], v246 offset:14944
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[42:47], v[102:107], a[36:39], v175, v176 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[74:75], v247 offset:14952
	ds_read_b64 v[76:77], v247 offset:14960
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[36:41], v[102:107], a[40:43], v175, v176 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s98 offen offset:2048 lds
	ds_read_b64 v[78:79], v246 offset:15072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[30:35], v[102:107], a[44:47], v175, v176 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[80:81], v247 offset:15080
	ds_read_b64 v[82:83], v247 offset:15088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[24:29], v[102:107], a[48:51], v174, v176 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[84:85], v246 offset:15200
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[18:23], v[102:107], a[52:55], v174, v176 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[86:87], v247 offset:15208
	ds_read_b64 v[88:89], v247 offset:15216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[12:17], v[102:107], a[56:59], v174, v176 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s98 offen offset:3072 lds
	ds_read_b64 v[90:91], v246 offset:15328
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[0:5], v[102:107], a[60:63], v174, v176 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[92:93], v247 offset:15336
	ds_read_b64 v[94:95], v247 offset:15344
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[0:5], v[120:125], a[64:67], v174, v176 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[96:99], v226 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[12:17], v[120:125], a[68:71], v174, v176 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[100:101], v227 offset:22528
	s_mov_b32 m0, s92
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[18:23], v[120:125], a[72:75], v174, v176 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[8:11], s98 offen lds
	ds_read_b128 v[102:105], v226 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[24:29], v[120:125], a[76:79], v174, v176 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[106:107], v227 offset:23040
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[30:35], v[120:125], a[80:83], v175, v176 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[114:117], v226 offset:8192
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[36:41], v[120:125], a[84:87], v175, v176 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[118:119], v227 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[42:47], v[120:125], a[88:91], v175, v176 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[8:11], s98 offen offset:1024 lds
	ds_read_b128 v[150:153], v226 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[48:53], v[120:125], a[92:95], v175, v176 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[154:155], v227 offset:24064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[48:53], v[126:131], a[96:99], v175, v176 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[120:123], v226 offset:10240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[42:47], v[126:131], a[100:103], v175, v176 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[124:125], v227 offset:24576
	s_add_i32 s96, s64, s96
	s_mov_b32 m0, s93
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[36:41], v[126:131], a[104:107], v175, v176 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[12:15], s96 offen lds
	ds_read_b128 v[156:159], v226 offset:11264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[30:35], v[126:131], a[108:111], v175, v176 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[160:161], v227 offset:25088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[24:29], v[126:131], a[112:115], v174, v176 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[162:165], v226 offset:12288
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[18:23], v[126:131], a[116:119], v174, v176 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[166:167], v227 offset:25600
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[12:17], v[126:131], a[120:123], v174, v176 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[24:27], s96 offen offset:1024 lds
	ds_read_b128 v[168:171], v226 offset:13312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[0:5], v[126:131], a[124:127], v174, v176 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[172:173], v227 offset:26112
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[0:5], v[108:113], a[128:131], v174, v177 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[12:17], v[108:113], a[132:135], v174, v177 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[18:23], v[108:113], a[136:139], v174, v177 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[24:29], v[108:113], a[140:143], v174, v177 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[30:35], v[108:113], a[144:147], v175, v177 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[36:41], v[108:113], a[148:151], v175, v177 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[42:47], v[108:113], a[152:155], v175, v177 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[48:53], v[108:113], a[156:159], v175, v177 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[48:53], v[132:137], a[160:163], v175, v177 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[42:47], v[132:137], a[164:167], v175, v177 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[36:41], v[132:137], a[168:171], v175, v177 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[30:35], v[132:137], a[172:175], v175, v177 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[24:29], v[132:137], a[176:179], v174, v177 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[18:23], v[132:137], a[180:183], v174, v177 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[12:17], v[132:137], a[184:187], v174, v177 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[0:5], v[132:137], a[188:191], v174, v177 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[0:5], v[138:143], a[192:195], v174, v177 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[12:17], v[138:143], a[196:199], v174, v177 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[18:23], v[138:143], a[200:203], v174, v177 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[24:29], v[138:143], a[204:207], v174, v177 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[30:35], v[138:143], a[208:211], v175, v177 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[36:41], v[138:143], a[212:215], v175, v177 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[42:47], v[138:143], a[216:219], v175, v177 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[48:53], v[138:143], a[220:223], v175, v177 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[48:53], v[144:149], a[224:227], v175, v177 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[42:47], v[144:149], a[228:231], v175, v177 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[36:41], v[144:149], a[232:235], v175, v177 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[30:35], v[144:149], a[236:239], v175, v177 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[24:29], v[144:149], a[240:243], v174, v177 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[18:23], v[144:149], a[244:247], v174, v177 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[12:17], v[144:149], a[248:251], v174, v177 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[0:5], v[144:149], a[252:255], v174, v177 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s96, s94, 0x180
	s_lshl_b32 s96, s96, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[6:11], v[96:101], a[0:3], v178, v180 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(8) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[174:175], v224 offset:2048
	ds_read_b64 v[176:177], v225 offset:3072
	ds_read_b64 v[0:1], v236 offset:47424
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[54:59], v[96:101], a[4:7], v178, v180 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[2:3], v236 offset:47432
	ds_read_b64 v[4:5], v237 offset:47440
	s_add_i32 s98, s97, 0x48000
	s_mov_b32 m0, s77
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[60:65], v[96:101], a[8:11], v178, v180 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s98 offen lds
	ds_read_b64 v[12:13], v236 offset:47552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:71], v[96:101], a[12:15], v178, v180 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[14:15], v236 offset:47560
	ds_read_b64 v[16:17], v237 offset:47568
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[72:77], v[96:101], a[16:19], v179, v180 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[18:19], v236 offset:47680
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[78:83], v[96:101], a[20:23], v179, v180 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[20:21], v236 offset:47688
	ds_read_b64 v[22:23], v237 offset:47696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[84:89], v[96:101], a[24:27], v179, v180 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s98 offen offset:1024 lds
	ds_read_b64 v[24:25], v236 offset:47808
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[90:95], v[96:101], a[28:31], v179, v180 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[26:27], v236 offset:47816
	ds_read_b64 v[28:29], v237 offset:47824
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[90:95], v[102:107], a[32:35], v179, v180 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[30:31], v236 offset:47936
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[84:89], v[102:107], a[36:39], v179, v180 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[32:33], v236 offset:47944
	ds_read_b64 v[34:35], v237 offset:47952
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[78:83], v[102:107], a[40:43], v179, v180 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s98 offen offset:2048 lds
	ds_read_b64 v[36:37], v236 offset:48064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[72:77], v[102:107], a[44:47], v179, v180 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[38:39], v236 offset:48072
	ds_read_b64 v[40:41], v237 offset:48080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[66:71], v[102:107], a[48:51], v178, v180 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[42:43], v236 offset:48192
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[60:65], v[102:107], a[52:55], v178, v180 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[44:45], v236 offset:48200
	ds_read_b64 v[46:47], v237 offset:48208
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[54:59], v[102:107], a[56:59], v178, v180 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s98 offen offset:3072 lds
	ds_read_b64 v[48:49], v236 offset:48320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[6:11], v[102:107], a[60:63], v178, v180 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[50:51], v236 offset:48328
	ds_read_b64 v[52:53], v237 offset:48336
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[6:11], v[114:119], a[64:67], v178, v180 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[96:99], v226 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[54:59], v[114:119], a[68:71], v178, v180 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[100:101], v227 offset:47104
	s_mov_b32 m0, s78
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[60:65], v[114:119], a[72:75], v178, v180 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[8:11], s98 offen lds
	ds_read_b128 v[102:105], v226 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[66:71], v[114:119], a[76:79], v178, v180 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[106:107], v227 offset:47616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[72:77], v[114:119], a[80:83], v179, v180 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[108:111], v226 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[78:83], v[114:119], a[84:87], v179, v180 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[112:113], v227 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[84:89], v[114:119], a[88:91], v179, v180 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[8:11], s98 offen offset:1024 lds
	ds_read_b128 v[126:129], v226 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[90:95], v[114:119], a[92:95], v179, v180 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[130:131], v227 offset:48640
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[90:95], v[150:155], a[96:99], v179, v180 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[114:117], v226 offset:34816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[84:89], v[150:155], a[100:103], v179, v180 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[118:119], v227 offset:49152
	s_add_i32 s96, s64, s96
	s_mov_b32 m0, s64
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[78:83], v[150:155], a[104:107], v179, v180 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[12:15], s96 offen lds
	ds_read_b128 v[132:135], v226 offset:35840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[72:77], v[150:155], a[108:111], v179, v180 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[136:137], v227 offset:49664
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[66:71], v[150:155], a[112:115], v178, v180 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[138:141], v226 offset:36864
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[60:65], v[150:155], a[116:119], v178, v180 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[142:143], v227 offset:50176
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[54:59], v[150:155], a[120:123], v178, v180 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[24:27], s96 offen offset:1024 lds
	ds_read_b128 v[144:147], v226 offset:37888
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[6:11], v[150:155], a[124:127], v178, v180 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[148:149], v227 offset:50688
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[6:11], v[120:125], a[128:131], v178, v181 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[54:59], v[120:125], a[132:135], v178, v181 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[60:65], v[120:125], a[136:139], v178, v181 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[66:71], v[120:125], a[140:143], v178, v181 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s96, s95, 0x480
	s_mov_b32 m0, s80
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[72:77], v[120:125], a[144:147], v179, v181 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[20:23], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[78:83], v[120:125], a[148:151], v179, v181 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[84:89], v[120:125], a[152:155], v179, v181 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s81
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[90:95], v[120:125], a[156:159], v179, v181 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[28:31], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[90:95], v[156:161], a[160:163], v179, v181 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[84:89], v[156:161], a[164:167], v179, v181 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[78:83], v[156:161], a[168:171], v179, v181 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s82
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[72:77], v[156:161], a[172:175], v179, v181 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[32:35], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[66:71], v[156:161], a[176:179], v178, v181 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[60:65], v[156:161], a[180:183], v178, v181 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s83
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[54:59], v[156:161], a[184:187], v178, v181 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[36:39], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[6:11], v[156:161], a[188:191], v178, v181 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[6:11], v[162:167], a[192:195], v178, v181 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[54:59], v[162:167], a[196:199], v178, v181 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s84
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[60:65], v[162:167], a[200:203], v178, v181 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[40:43], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[66:71], v[162:167], a[204:207], v178, v181 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[72:77], v[162:167], a[208:211], v179, v181 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s85
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[78:83], v[162:167], a[212:215], v179, v181 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[44:47], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[84:89], v[162:167], a[216:219], v179, v181 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[90:95], v[162:167], a[220:223], v179, v181 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[90:95], v[168:173], a[224:227], v179, v181 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s86
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[84:89], v[168:173], a[228:231], v179, v181 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[48:51], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[78:83], v[168:173], a[232:235], v179, v181 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[72:77], v[168:173], a[236:239], v179, v181 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s87
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[66:71], v[168:173], a[240:243], v178, v181 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[52:55], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[60:65], v[168:173], a[244:247], v178, v181 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[54:59], v[168:173], a[248:251], v178, v181 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[6:11], v[168:173], a[252:255], v178, v181 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s96, s3, 13
	s_add_i32 s98, s94, 0x1a0
	s_lshl_b32 s98, s98, 9
	s_lshr_b32 s96, s96, 2
	s_mul_i32 s96, s96, 3
	s_lshl_b32 s96, s96, 7
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[0:5], v[96:101], a[0:3], v174, v176 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(16) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[178:179], v224 offset:4096
	ds_read_b64 v[180:181], v225 offset:5120
	ds_read_b64 v[6:7], v239 offset:14368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[12:17], v[96:101], a[4:7], v174, v176 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[8:9], v239 offset:14376
	ds_read_b64 v[10:11], v239 offset:14384
	s_add_i32 s99, s97, 0x4e000
	s_mov_b32 m0, s88
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:23], v[96:101], a[8:11], v174, v176 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s99 offen lds
	ds_read_b64 v[54:55], v239 offset:14496
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[24:29], v[96:101], a[12:15], v174, v176 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[56:57], v239 offset:14504
	ds_read_b64 v[58:59], v239 offset:14512
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[30:35], v[96:101], a[16:19], v175, v176 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[60:61], v239 offset:14624
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[36:41], v[96:101], a[20:23], v175, v176 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[62:63], v239 offset:14632
	ds_read_b64 v[64:65], v239 offset:14640
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[42:47], v[96:101], a[24:27], v175, v176 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s99 offen offset:1024 lds
	ds_read_b64 v[66:67], v239 offset:14752
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[48:53], v[96:101], a[28:31], v175, v176 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[68:69], v239 offset:14760
	ds_read_b64 v[70:71], v239 offset:14768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[48:53], v[102:107], a[32:35], v175, v176 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[72:73], v239 offset:14880
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[42:47], v[102:107], a[36:39], v175, v176 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[74:75], v239 offset:14888
	ds_read_b64 v[76:77], v239 offset:14896
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[36:41], v[102:107], a[40:43], v175, v176 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s99 offen offset:2048 lds
	ds_read_b64 v[78:79], v239 offset:15008
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[30:35], v[102:107], a[44:47], v175, v176 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[80:81], v239 offset:15016
	ds_read_b64 v[82:83], v239 offset:15024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[24:29], v[102:107], a[48:51], v174, v176 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[84:85], v239 offset:15136
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[18:23], v[102:107], a[52:55], v174, v176 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[86:87], v239 offset:15144
	ds_read_b64 v[88:89], v239 offset:15152
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[12:17], v[102:107], a[56:59], v174, v176 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s99 offen offset:3072 lds
	ds_read_b64 v[90:91], v239 offset:15264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[0:5], v[102:107], a[60:63], v174, v176 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[92:93], v239 offset:15272
	ds_read_b64 v[94:95], v239 offset:15280
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[0:5], v[108:113], a[64:67], v174, v176 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[96:99], v226 offset:55296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[12:17], v[108:113], a[68:71], v174, v176 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[100:101], v238 offset:6144
	s_mov_b32 m0, s89
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[18:23], v[108:113], a[72:75], v174, v176 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[8:11], s99 offen lds
	ds_read_b128 v[102:105], v226 offset:56320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[24:29], v[108:113], a[76:79], v174, v176 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[106:107], v238 offset:6656
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[30:35], v[108:113], a[80:83], v175, v176 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[120:123], v226 offset:57344
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[36:41], v[108:113], a[84:87], v175, v176 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[124:125], v238 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[42:47], v[108:113], a[88:91], v175, v176 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[8:11], s99 offen offset:1024 lds
	ds_read_b128 v[150:153], v226 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[48:53], v[108:113], a[92:95], v175, v176 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[154:155], v238 offset:7680
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[48:53], v[126:131], a[96:99], v175, v176 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[108:111], v226 offset:59392
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[42:47], v[126:131], a[100:103], v175, v176 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[112:113], v238 offset:8192
	s_add_i32 s98, s64, s98
	s_mov_b32 m0, s90
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[36:41], v[126:131], a[104:107], v175, v176 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[12:15], s98 offen lds
	ds_read_b128 v[156:159], v226 offset:60416
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[30:35], v[126:131], a[108:111], v175, v176 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[160:161], v238 offset:8704
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[24:29], v[126:131], a[112:115], v174, v176 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[162:165], v226 offset:61440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[18:23], v[126:131], a[116:119], v174, v176 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[166:167], v238 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[12:17], v[126:131], a[120:123], v174, v176 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[24:27], s98 offen offset:1024 lds
	ds_read_b128 v[168:171], v226 offset:62464
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[0:5], v[126:131], a[124:127], v174, v176 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[172:173], v238 offset:9728
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[0:5], v[114:119], a[128:131], v174, v177 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[12:17], v[114:119], a[132:135], v174, v177 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[18:23], v[114:119], a[136:139], v174, v177 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[24:29], v[114:119], a[140:143], v174, v177 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s96, s56, s96
	s_add_i32 s96, s96, 0x80
	s_mov_b32 m0, s66
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[30:35], v[114:119], a[144:147], v175, v177 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[20:23], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[36:41], v[114:119], a[148:151], v175, v177 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[42:47], v[114:119], a[152:155], v175, v177 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s70
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[48:53], v[114:119], a[156:159], v175, v177 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[28:31], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[48:53], v[132:137], a[160:163], v175, v177 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[42:47], v[132:137], a[164:167], v175, v177 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[36:41], v[132:137], a[168:171], v175, v177 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s71
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[30:35], v[132:137], a[172:175], v175, v177 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[32:35], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[24:29], v[132:137], a[176:179], v174, v177 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[18:23], v[132:137], a[180:183], v174, v177 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s72
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[12:17], v[132:137], a[184:187], v174, v177 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[36:39], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[0:5], v[132:137], a[188:191], v174, v177 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[0:5], v[138:143], a[192:195], v174, v177 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[12:17], v[138:143], a[196:199], v174, v177 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s73
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[18:23], v[138:143], a[200:203], v174, v177 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[40:43], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[24:29], v[138:143], a[204:207], v174, v177 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[30:35], v[138:143], a[208:211], v175, v177 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s74
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[36:41], v[138:143], a[212:215], v175, v177 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[44:47], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[42:47], v[138:143], a[216:219], v175, v177 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[48:53], v[138:143], a[220:223], v175, v177 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[48:53], v[144:149], a[224:227], v175, v177 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s75
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[42:47], v[144:149], a[228:231], v175, v177 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[48:51], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[36:41], v[144:149], a[232:235], v175, v177 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[30:35], v[144:149], a[236:239], v175, v177 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s76
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[24:29], v[144:149], a[240:243], v174, v177 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[52:55], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[18:23], v[144:149], a[244:247], v174, v177 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[12:17], v[144:149], a[248:251], v174, v177 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[0:5], v[144:149], a[252:255], v174, v177 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s96, s3, 14
	s_add_i32 s98, s94, 0x1c0
	s_lshl_b32 s98, s98, 9
	s_lshr_b32 s96, s96, 2
	s_mul_i32 s96, s96, 3
	s_lshl_b32 s96, s96, 7
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[6:11], v[96:101], a[0:3], v178, v180 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(16) lgkmcnt(0)
	s_barrier
	ds_read_b128 v[18:21], v240 offset:47360
	ds_read_b128 v[22:25], v240 offset:47376
	ds_read_b64 v[174:175], v224
	ds_read_b64 v[176:177], v225 offset:1024
	ds_read_b64 v[0:1], v239 offset:47360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[54:59], v[96:101], a[4:7], v178, v180 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[2:3], v239 offset:47368
	ds_read_b64 v[4:5], v239 offset:47376
	s_add_i32 s99, s97, 0x54000
	s_mov_b32 m0, s91
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[60:65], v[96:101], a[8:11], v178, v180 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s99 offen lds
	ds_read_b64 v[12:13], v239 offset:47488
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:71], v[96:101], a[12:15], v178, v180 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[14:15], v239 offset:47496
	ds_read_b64 v[16:17], v239 offset:47504
	s_waitcnt lgkmcnt(9)
	ds_write_b64 v241, v[18:19] offset:14336
	ds_write_b64 v241, v[20:21] offset:14344
	s_waitcnt lgkmcnt(10)
	ds_write_b64 v241, v[22:23] offset:14352
	ds_write_b64 v241, v[24:25] offset:14360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[72:77], v[96:101], a[16:19], v179, v180 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[18:19], v239 offset:47616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[78:83], v[96:101], a[20:23], v179, v180 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[20:21], v239 offset:47624
	ds_read_b64 v[22:23], v239 offset:47632
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[84:89], v[96:101], a[24:27], v179, v180 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s99 offen offset:1024 lds
	ds_read_b64 v[24:25], v239 offset:47744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[90:95], v[96:101], a[28:31], v179, v180 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[26:27], v239 offset:47752
	ds_read_b64 v[28:29], v239 offset:47760
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[90:95], v[102:107], a[32:35], v179, v180 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[30:31], v239 offset:47872
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[84:89], v[102:107], a[36:39], v179, v180 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[32:33], v239 offset:47880
	ds_read_b64 v[34:35], v239 offset:47888
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[78:83], v[102:107], a[40:43], v179, v180 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s99 offen offset:2048 lds
	ds_read_b64 v[36:37], v239 offset:48000
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[72:77], v[102:107], a[44:47], v179, v180 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[38:39], v239 offset:48008
	ds_read_b64 v[40:41], v239 offset:48016
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[66:71], v[102:107], a[48:51], v178, v180 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[42:43], v239 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[60:65], v[102:107], a[52:55], v178, v180 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[44:45], v239 offset:48136
	ds_read_b64 v[46:47], v239 offset:48144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[54:59], v[102:107], a[56:59], v178, v180 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s99 offen offset:3072 lds
	ds_read_b64 v[48:49], v239 offset:48256
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[6:11], v[102:107], a[60:63], v178, v180 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[50:51], v239 offset:48264
	ds_read_b64 v[52:53], v239 offset:48272
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[6:11], v[120:125], a[64:67], v178, v180 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[96:99], v226 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[54:59], v[120:125], a[68:71], v178, v180 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[100:101], v227 offset:22528
	s_mov_b32 m0, s92
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[60:65], v[120:125], a[72:75], v178, v180 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[8:11], s99 offen lds
	ds_read_b128 v[102:105], v226 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[66:71], v[120:125], a[76:79], v178, v180 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[106:107], v227 offset:23040
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[72:77], v[120:125], a[80:83], v179, v180 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[114:117], v226 offset:8192
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[78:83], v[120:125], a[84:87], v179, v180 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[118:119], v227 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[84:89], v[120:125], a[88:91], v179, v180 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[8:11], s99 offen offset:1024 lds
	ds_read_b128 v[126:129], v226 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[90:95], v[120:125], a[92:95], v179, v180 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[130:131], v227 offset:24064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[90:95], v[150:155], a[96:99], v179, v180 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[120:123], v226 offset:10240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[84:89], v[150:155], a[100:103], v179, v180 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[124:125], v227 offset:24576
	s_add_i32 s98, s64, s98
	s_mov_b32 m0, s93
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[78:83], v[150:155], a[104:107], v179, v180 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[12:15], s98 offen lds
	ds_read_b128 v[132:135], v226 offset:11264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[72:77], v[150:155], a[108:111], v179, v180 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[136:137], v227 offset:25088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[66:71], v[150:155], a[112:115], v178, v180 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[138:141], v226 offset:12288
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[60:65], v[150:155], a[116:119], v178, v180 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[142:143], v227 offset:25600
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[54:59], v[150:155], a[120:123], v178, v180 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[24:27], s98 offen offset:1024 lds
	ds_read_b128 v[144:147], v226 offset:13312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[6:11], v[150:155], a[124:127], v178, v180 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[148:149], v227 offset:26112
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[6:11], v[108:113], a[128:131], v178, v181 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[54:59], v[108:113], a[132:135], v178, v181 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[60:65], v[108:113], a[136:139], v178, v181 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[66:71], v[108:113], a[140:143], v178, v181 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s96, s56, s96
	s_add_i32 s96, s96, 0x100
	s_mov_b32 m0, s80
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[72:77], v[108:113], a[144:147], v179, v181 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[20:23], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[78:83], v[108:113], a[148:151], v179, v181 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[84:89], v[108:113], a[152:155], v179, v181 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s81
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[90:95], v[108:113], a[156:159], v179, v181 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[28:31], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[90:95], v[156:161], a[160:163], v179, v181 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[84:89], v[156:161], a[164:167], v179, v181 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[78:83], v[156:161], a[168:171], v179, v181 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s82
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[72:77], v[156:161], a[172:175], v179, v181 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[32:35], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[66:71], v[156:161], a[176:179], v178, v181 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[60:65], v[156:161], a[180:183], v178, v181 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s83
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[54:59], v[156:161], a[184:187], v178, v181 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[36:39], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[6:11], v[156:161], a[188:191], v178, v181 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[6:11], v[162:167], a[192:195], v178, v181 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[54:59], v[162:167], a[196:199], v178, v181 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s84
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[60:65], v[162:167], a[200:203], v178, v181 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[40:43], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[66:71], v[162:167], a[204:207], v178, v181 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[72:77], v[162:167], a[208:211], v179, v181 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s85
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[78:83], v[162:167], a[212:215], v179, v181 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[44:47], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[84:89], v[162:167], a[216:219], v179, v181 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[90:95], v[162:167], a[220:223], v179, v181 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[90:95], v[168:173], a[224:227], v179, v181 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s86
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[84:89], v[168:173], a[228:231], v179, v181 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[48:51], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[78:83], v[168:173], a[232:235], v179, v181 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[72:77], v[168:173], a[236:239], v179, v181 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s87
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[66:71], v[168:173], a[240:243], v178, v181 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[52:55], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[60:65], v[168:173], a[244:247], v178, v181 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[54:59], v[168:173], a[248:251], v178, v181 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[6:11], v[168:173], a[252:255], v178, v181 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s96, s94, 0x1e0
	s_lshl_b32 s96, s96, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[0:5], v[96:101], a[0:3], v174, v176 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(16) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[178:179], v224 offset:2048
	ds_read_b64 v[180:181], v225 offset:3072
	ds_read_b64 v[6:7], v242 offset:47456
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[12:17], v[96:101], a[4:7], v174, v176 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[8:9], v243 offset:47464
	ds_read_b64 v[10:11], v243 offset:47472
	s_add_i32 s98, s97, 0x5a000
	s_mov_b32 m0, s77
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:23], v[96:101], a[8:11], v174, v176 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s98 offen lds
	ds_read_b64 v[54:55], v242 offset:47584
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[24:29], v[96:101], a[12:15], v174, v176 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[56:57], v243 offset:47592
	ds_read_b64 v[58:59], v243 offset:47600
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[30:35], v[96:101], a[16:19], v175, v176 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[60:61], v242 offset:47712
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[36:41], v[96:101], a[20:23], v175, v176 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[62:63], v243 offset:47720
	ds_read_b64 v[64:65], v243 offset:47728
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[42:47], v[96:101], a[24:27], v175, v176 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s98 offen offset:1024 lds
	ds_read_b64 v[66:67], v242 offset:47840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[48:53], v[96:101], a[28:31], v175, v176 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[68:69], v243 offset:47848
	ds_read_b64 v[70:71], v243 offset:47856
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[48:53], v[102:107], a[32:35], v175, v176 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[72:73], v242 offset:47968
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[42:47], v[102:107], a[36:39], v175, v176 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[74:75], v243 offset:47976
	ds_read_b64 v[76:77], v243 offset:47984
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[36:41], v[102:107], a[40:43], v175, v176 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s98 offen offset:2048 lds
	ds_read_b64 v[78:79], v242 offset:48096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[30:35], v[102:107], a[44:47], v175, v176 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[80:81], v243 offset:48104
	ds_read_b64 v[82:83], v243 offset:48112
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[24:29], v[102:107], a[48:51], v174, v176 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[84:85], v242 offset:48224
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[18:23], v[102:107], a[52:55], v174, v176 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[86:87], v243 offset:48232
	ds_read_b64 v[88:89], v243 offset:48240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[12:17], v[102:107], a[56:59], v174, v176 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s98 offen offset:3072 lds
	ds_read_b64 v[90:91], v242 offset:48352
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[0:5], v[102:107], a[60:63], v174, v176 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[92:93], v243 offset:48360
	ds_read_b64 v[94:95], v243 offset:48368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[0:5], v[114:119], a[64:67], v174, v176 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[96:99], v226 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[12:17], v[114:119], a[68:71], v174, v176 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[100:101], v227 offset:47104
	s_mov_b32 m0, s78
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[18:23], v[114:119], a[72:75], v174, v176 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[8:11], s98 offen lds
	ds_read_b128 v[102:105], v226 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[24:29], v[114:119], a[76:79], v174, v176 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[106:107], v227 offset:47616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[30:35], v[114:119], a[80:83], v175, v176 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[108:111], v226 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[36:41], v[114:119], a[84:87], v175, v176 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[112:113], v227 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[42:47], v[114:119], a[88:91], v175, v176 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[8:11], s98 offen offset:1024 lds
	ds_read_b128 v[150:153], v226 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[48:53], v[114:119], a[92:95], v175, v176 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[154:155], v227 offset:48640
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[48:53], v[126:131], a[96:99], v175, v176 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[114:117], v226 offset:34816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[42:47], v[126:131], a[100:103], v175, v176 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[118:119], v227 offset:49152
	s_add_i32 s96, s64, s96
	s_mov_b32 m0, s64
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[36:41], v[126:131], a[104:107], v175, v176 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[12:15], s96 offen lds
	ds_read_b128 v[156:159], v226 offset:35840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[30:35], v[126:131], a[108:111], v175, v176 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[160:161], v227 offset:49664
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[24:29], v[126:131], a[112:115], v174, v176 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[162:165], v226 offset:36864
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[18:23], v[126:131], a[116:119], v174, v176 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[166:167], v227 offset:50176
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[12:17], v[126:131], a[120:123], v174, v176 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[24:27], s96 offen offset:1024 lds
	ds_read_b128 v[168:171], v226 offset:37888
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[0:5], v[126:131], a[124:127], v174, v176 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[172:173], v227 offset:50688
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[0:5], v[120:125], a[128:131], v174, v177 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[12:17], v[120:125], a[132:135], v174, v177 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[18:23], v[120:125], a[136:139], v174, v177 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[24:29], v[120:125], a[140:143], v174, v177 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[30:35], v[120:125], a[144:147], v175, v177 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[36:41], v[120:125], a[148:151], v175, v177 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[42:47], v[120:125], a[152:155], v175, v177 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[48:53], v[120:125], a[156:159], v175, v177 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[48:53], v[132:137], a[160:163], v175, v177 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[42:47], v[132:137], a[164:167], v175, v177 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[36:41], v[132:137], a[168:171], v175, v177 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[30:35], v[132:137], a[172:175], v175, v177 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[24:29], v[132:137], a[176:179], v174, v177 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[18:23], v[132:137], a[180:183], v174, v177 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[12:17], v[132:137], a[184:187], v174, v177 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[0:5], v[132:137], a[188:191], v174, v177 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[0:5], v[138:143], a[192:195], v174, v177 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[12:17], v[138:143], a[196:199], v174, v177 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[18:23], v[138:143], a[200:203], v174, v177 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[24:29], v[138:143], a[204:207], v174, v177 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[30:35], v[138:143], a[208:211], v175, v177 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[36:41], v[138:143], a[212:215], v175, v177 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[42:47], v[138:143], a[216:219], v175, v177 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[48:53], v[138:143], a[220:223], v175, v177 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[48:53], v[144:149], a[224:227], v175, v177 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[42:47], v[144:149], a[228:231], v175, v177 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[36:41], v[144:149], a[232:235], v175, v177 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[30:35], v[144:149], a[236:239], v175, v177 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[24:29], v[144:149], a[240:243], v174, v177 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[18:23], v[144:149], a[244:247], v174, v177 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[12:17], v[144:149], a[248:251], v174, v177 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[0:5], v[144:149], a[252:255], v174, v177 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s96, s94, 0x200
	s_lshl_b32 s96, s96, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[6:11], v[96:101], a[0:3], v178, v180 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(8) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[174:175], v224 offset:4096
	ds_read_b64 v[176:177], v225 offset:5120
	ds_read_b64 v[0:1], v244 offset:14400
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[54:59], v[96:101], a[4:7], v178, v180 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[2:3], v244 offset:14408
	ds_read_b64 v[4:5], v245 offset:14416
	s_add_i32 s98, s97, 0x60000
	s_mov_b32 m0, s88
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[60:65], v[96:101], a[8:11], v178, v180 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s98 offen lds
	ds_read_b64 v[12:13], v244 offset:14528
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:71], v[96:101], a[12:15], v178, v180 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[14:15], v244 offset:14536
	ds_read_b64 v[16:17], v245 offset:14544
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[72:77], v[96:101], a[16:19], v179, v180 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[18:19], v244 offset:14656
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[78:83], v[96:101], a[20:23], v179, v180 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[20:21], v244 offset:14664
	ds_read_b64 v[22:23], v245 offset:14672
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[84:89], v[96:101], a[24:27], v179, v180 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s98 offen offset:1024 lds
	ds_read_b64 v[24:25], v244 offset:14784
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[90:95], v[96:101], a[28:31], v179, v180 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[26:27], v244 offset:14792
	ds_read_b64 v[28:29], v245 offset:14800
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[90:95], v[102:107], a[32:35], v179, v180 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[30:31], v244 offset:14912
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[84:89], v[102:107], a[36:39], v179, v180 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[32:33], v244 offset:14920
	ds_read_b64 v[34:35], v245 offset:14928
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[78:83], v[102:107], a[40:43], v179, v180 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s98 offen offset:2048 lds
	ds_read_b64 v[36:37], v244 offset:15040
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[72:77], v[102:107], a[44:47], v179, v180 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[38:39], v244 offset:15048
	ds_read_b64 v[40:41], v245 offset:15056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[66:71], v[102:107], a[48:51], v178, v180 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[42:43], v244 offset:15168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[60:65], v[102:107], a[52:55], v178, v180 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[44:45], v244 offset:15176
	ds_read_b64 v[46:47], v245 offset:15184
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[54:59], v[102:107], a[56:59], v178, v180 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s98 offen offset:3072 lds
	ds_read_b64 v[48:49], v244 offset:15296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[6:11], v[102:107], a[60:63], v178, v180 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[50:51], v244 offset:15304
	ds_read_b64 v[52:53], v245 offset:15312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[6:11], v[108:113], a[64:67], v178, v180 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[96:99], v226 offset:55296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[54:59], v[108:113], a[68:71], v178, v180 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[100:101], v238 offset:6144
	s_mov_b32 m0, s89
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[60:65], v[108:113], a[72:75], v178, v180 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[8:11], s98 offen lds
	ds_read_b128 v[102:105], v226 offset:56320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[66:71], v[108:113], a[76:79], v178, v180 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[106:107], v238 offset:6656
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[72:77], v[108:113], a[80:83], v179, v180 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[120:123], v226 offset:57344
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[78:83], v[108:113], a[84:87], v179, v180 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[124:125], v238 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[84:89], v[108:113], a[88:91], v179, v180 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[8:11], s98 offen offset:1024 lds
	ds_read_b128 v[126:129], v226 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[90:95], v[108:113], a[92:95], v179, v180 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[130:131], v238 offset:7680
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[90:95], v[150:155], a[96:99], v179, v180 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[108:111], v226 offset:59392
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[84:89], v[150:155], a[100:103], v179, v180 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[112:113], v238 offset:8192
	s_add_i32 s96, s64, s96
	s_mov_b32 m0, s90
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[78:83], v[150:155], a[104:107], v179, v180 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[12:15], s96 offen lds
	ds_read_b128 v[132:135], v226 offset:60416
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[72:77], v[150:155], a[108:111], v179, v180 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[136:137], v238 offset:8704
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[66:71], v[150:155], a[112:115], v178, v180 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[138:141], v226 offset:61440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[60:65], v[150:155], a[116:119], v178, v180 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[142:143], v238 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[54:59], v[150:155], a[120:123], v178, v180 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[24:27], s96 offen offset:1024 lds
	ds_read_b128 v[144:147], v226 offset:62464
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[6:11], v[150:155], a[124:127], v178, v180 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[148:149], v238 offset:9728
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[6:11], v[114:119], a[128:131], v178, v181 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[54:59], v[114:119], a[132:135], v178, v181 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[60:65], v[114:119], a[136:139], v178, v181 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[66:71], v[114:119], a[140:143], v178, v181 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s96, s95, 0x600
	s_mov_b32 m0, s66
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[72:77], v[114:119], a[144:147], v179, v181 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[20:23], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[78:83], v[114:119], a[148:151], v179, v181 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[84:89], v[114:119], a[152:155], v179, v181 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s70
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[90:95], v[114:119], a[156:159], v179, v181 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[28:31], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[90:95], v[156:161], a[160:163], v179, v181 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[84:89], v[156:161], a[164:167], v179, v181 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[78:83], v[156:161], a[168:171], v179, v181 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s71
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[72:77], v[156:161], a[172:175], v179, v181 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[32:35], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[66:71], v[156:161], a[176:179], v178, v181 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[60:65], v[156:161], a[180:183], v178, v181 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s72
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[54:59], v[156:161], a[184:187], v178, v181 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[36:39], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[6:11], v[156:161], a[188:191], v178, v181 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[6:11], v[162:167], a[192:195], v178, v181 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[54:59], v[162:167], a[196:199], v178, v181 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s73
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[60:65], v[162:167], a[200:203], v178, v181 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[40:43], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[66:71], v[162:167], a[204:207], v178, v181 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[72:77], v[162:167], a[208:211], v179, v181 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s74
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[78:83], v[162:167], a[212:215], v179, v181 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[44:47], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[84:89], v[162:167], a[216:219], v179, v181 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[90:95], v[162:167], a[220:223], v179, v181 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[90:95], v[168:173], a[224:227], v179, v181 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s75
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[84:89], v[168:173], a[228:231], v179, v181 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[48:51], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[78:83], v[168:173], a[232:235], v179, v181 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[72:77], v[168:173], a[236:239], v179, v181 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s76
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[66:71], v[168:173], a[240:243], v178, v181 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[52:55], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[60:65], v[168:173], a[244:247], v178, v181 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[54:59], v[168:173], a[248:251], v178, v181 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[6:11], v[168:173], a[252:255], v178, v181 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s96, s3, 17
	s_add_i32 s98, s94, 0x220
	s_lshl_b32 s98, s98, 9
	s_lshr_b32 s96, s96, 2
	s_mul_i32 s96, s96, 3
	s_lshl_b32 s96, s96, 7
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[0:5], v[96:101], a[0:3], v174, v176 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(16) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[178:179], v224
	ds_read_b64 v[180:181], v225 offset:1024
	ds_read_b64 v[6:7], v239 offset:47392
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[12:17], v[96:101], a[4:7], v174, v176 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[8:9], v239 offset:47400
	ds_read_b64 v[10:11], v239 offset:47408
	s_add_i32 s99, s97, 0x66000
	s_mov_b32 m0, s91
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:23], v[96:101], a[8:11], v174, v176 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s99 offen lds
	ds_read_b64 v[54:55], v239 offset:47520
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[24:29], v[96:101], a[12:15], v174, v176 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[56:57], v239 offset:47528
	ds_read_b64 v[58:59], v239 offset:47536
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[30:35], v[96:101], a[16:19], v175, v176 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[60:61], v239 offset:47648
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[36:41], v[96:101], a[20:23], v175, v176 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[62:63], v239 offset:47656
	ds_read_b64 v[64:65], v239 offset:47664
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[42:47], v[96:101], a[24:27], v175, v176 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s99 offen offset:1024 lds
	ds_read_b64 v[66:67], v239 offset:47776
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[48:53], v[96:101], a[28:31], v175, v176 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[68:69], v239 offset:47784
	ds_read_b64 v[70:71], v239 offset:47792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[48:53], v[102:107], a[32:35], v175, v176 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[72:73], v239 offset:47904
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[42:47], v[102:107], a[36:39], v175, v176 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[74:75], v239 offset:47912
	ds_read_b64 v[76:77], v239 offset:47920
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[36:41], v[102:107], a[40:43], v175, v176 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s99 offen offset:2048 lds
	ds_read_b64 v[78:79], v239 offset:48032
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[30:35], v[102:107], a[44:47], v175, v176 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[80:81], v239 offset:48040
	ds_read_b64 v[82:83], v239 offset:48048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[24:29], v[102:107], a[48:51], v174, v176 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[84:85], v239 offset:48160
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[18:23], v[102:107], a[52:55], v174, v176 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[86:87], v239 offset:48168
	ds_read_b64 v[88:89], v239 offset:48176
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[12:17], v[102:107], a[56:59], v174, v176 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s99 offen offset:3072 lds
	ds_read_b64 v[90:91], v239 offset:48288
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[0:5], v[102:107], a[60:63], v174, v176 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[92:93], v239 offset:48296
	ds_read_b64 v[94:95], v239 offset:48304
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[0:5], v[120:125], a[64:67], v174, v176 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[96:99], v226 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[12:17], v[120:125], a[68:71], v174, v176 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[100:101], v227 offset:22528
	s_mov_b32 m0, s92
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[18:23], v[120:125], a[72:75], v174, v176 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[8:11], s99 offen lds
	ds_read_b128 v[102:105], v226 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[24:29], v[120:125], a[76:79], v174, v176 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[106:107], v227 offset:23040
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[30:35], v[120:125], a[80:83], v175, v176 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[114:117], v226 offset:8192
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[36:41], v[120:125], a[84:87], v175, v176 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[118:119], v227 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[42:47], v[120:125], a[88:91], v175, v176 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[8:11], s99 offen offset:1024 lds
	ds_read_b128 v[150:153], v226 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[48:53], v[120:125], a[92:95], v175, v176 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[154:155], v227 offset:24064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[48:53], v[126:131], a[96:99], v175, v176 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[120:123], v226 offset:10240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[42:47], v[126:131], a[100:103], v175, v176 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[124:125], v227 offset:24576
	s_add_i32 s98, s64, s98
	s_mov_b32 m0, s93
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[36:41], v[126:131], a[104:107], v175, v176 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[12:15], s98 offen lds
	ds_read_b128 v[156:159], v226 offset:11264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[30:35], v[126:131], a[108:111], v175, v176 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[160:161], v227 offset:25088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[24:29], v[126:131], a[112:115], v174, v176 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[162:165], v226 offset:12288
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[18:23], v[126:131], a[116:119], v174, v176 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[166:167], v227 offset:25600
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[12:17], v[126:131], a[120:123], v174, v176 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[24:27], s98 offen offset:1024 lds
	ds_read_b128 v[168:171], v226 offset:13312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[0:5], v[126:131], a[124:127], v174, v176 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[172:173], v227 offset:26112
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[0:5], v[108:113], a[128:131], v174, v177 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[12:17], v[108:113], a[132:135], v174, v177 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[18:23], v[108:113], a[136:139], v174, v177 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[24:29], v[108:113], a[140:143], v174, v177 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s96, s56, s96
	s_add_i32 s96, s96, 0x80
	s_mov_b32 m0, s80
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[30:35], v[108:113], a[144:147], v175, v177 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[20:23], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[36:41], v[108:113], a[148:151], v175, v177 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[42:47], v[108:113], a[152:155], v175, v177 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s81
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[48:53], v[108:113], a[156:159], v175, v177 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[28:31], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[48:53], v[132:137], a[160:163], v175, v177 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[42:47], v[132:137], a[164:167], v175, v177 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[36:41], v[132:137], a[168:171], v175, v177 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s82
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[30:35], v[132:137], a[172:175], v175, v177 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[32:35], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[24:29], v[132:137], a[176:179], v174, v177 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[18:23], v[132:137], a[180:183], v174, v177 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s83
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[12:17], v[132:137], a[184:187], v174, v177 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[36:39], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[0:5], v[132:137], a[188:191], v174, v177 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[0:5], v[138:143], a[192:195], v174, v177 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[12:17], v[138:143], a[196:199], v174, v177 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s84
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[18:23], v[138:143], a[200:203], v174, v177 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[40:43], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[24:29], v[138:143], a[204:207], v174, v177 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[30:35], v[138:143], a[208:211], v175, v177 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s85
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[36:41], v[138:143], a[212:215], v175, v177 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[44:47], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[42:47], v[138:143], a[216:219], v175, v177 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[48:53], v[138:143], a[220:223], v175, v177 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[48:53], v[144:149], a[224:227], v175, v177 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s86
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[42:47], v[144:149], a[228:231], v175, v177 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[48:51], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[36:41], v[144:149], a[232:235], v175, v177 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[30:35], v[144:149], a[236:239], v175, v177 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s87
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[24:29], v[144:149], a[240:243], v174, v177 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[52:55], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[18:23], v[144:149], a[244:247], v174, v177 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[12:17], v[144:149], a[248:251], v174, v177 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[0:5], v[144:149], a[252:255], v174, v177 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s96, s3, 18
	s_add_i32 s98, s94, 0x240
	s_lshl_b32 s98, s98, 9
	s_lshr_b32 s96, s96, 2
	s_mul_i32 s96, s96, 3
	s_lshl_b32 s96, s96, 7
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[6:11], v[96:101], a[0:3], v178, v180 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(16) lgkmcnt(0)
	s_barrier
	ds_read_b128 v[18:21], v240 offset:14336
	ds_read_b128 v[22:25], v240 offset:14352
	ds_read_b64 v[174:175], v224 offset:2048
	ds_read_b64 v[176:177], v225 offset:3072
	ds_read_b64 v[0:1], v239 offset:14336
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[54:59], v[96:101], a[4:7], v178, v180 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[2:3], v239 offset:14344
	ds_read_b64 v[4:5], v239 offset:14352
	s_add_i32 s99, s97, 0x6c000
	s_mov_b32 m0, s77
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[60:65], v[96:101], a[8:11], v178, v180 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s99 offen lds
	ds_read_b64 v[12:13], v239 offset:14464
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:71], v[96:101], a[12:15], v178, v180 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[14:15], v239 offset:14472
	ds_read_b64 v[16:17], v239 offset:14480
	s_waitcnt lgkmcnt(9)
	ds_write_b64 v241, v[18:19] offset:14336
	ds_write_b64 v241, v[20:21] offset:14344
	s_waitcnt lgkmcnt(10)
	ds_write_b64 v241, v[22:23] offset:14352
	ds_write_b64 v241, v[24:25] offset:14360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[72:77], v[96:101], a[16:19], v179, v180 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[18:19], v239 offset:14592
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[78:83], v[96:101], a[20:23], v179, v180 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[20:21], v239 offset:14600
	ds_read_b64 v[22:23], v239 offset:14608
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[84:89], v[96:101], a[24:27], v179, v180 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s99 offen offset:1024 lds
	ds_read_b64 v[24:25], v239 offset:14720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[90:95], v[96:101], a[28:31], v179, v180 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[26:27], v239 offset:14728
	ds_read_b64 v[28:29], v239 offset:14736
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[90:95], v[102:107], a[32:35], v179, v180 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[30:31], v239 offset:14848
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[84:89], v[102:107], a[36:39], v179, v180 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[32:33], v239 offset:14856
	ds_read_b64 v[34:35], v239 offset:14864
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[78:83], v[102:107], a[40:43], v179, v180 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s99 offen offset:2048 lds
	ds_read_b64 v[36:37], v239 offset:14976
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[72:77], v[102:107], a[44:47], v179, v180 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[38:39], v239 offset:14984
	ds_read_b64 v[40:41], v239 offset:14992
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[66:71], v[102:107], a[48:51], v178, v180 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[42:43], v239 offset:15104
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[60:65], v[102:107], a[52:55], v178, v180 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[44:45], v239 offset:15112
	ds_read_b64 v[46:47], v239 offset:15120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[54:59], v[102:107], a[56:59], v178, v180 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s99 offen offset:3072 lds
	ds_read_b64 v[48:49], v239 offset:15232
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[6:11], v[102:107], a[60:63], v178, v180 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[50:51], v239 offset:15240
	ds_read_b64 v[52:53], v239 offset:15248
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[6:11], v[114:119], a[64:67], v178, v180 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[96:99], v226 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[54:59], v[114:119], a[68:71], v178, v180 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[100:101], v227 offset:47104
	s_mov_b32 m0, s78
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[60:65], v[114:119], a[72:75], v178, v180 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[8:11], s99 offen lds
	ds_read_b128 v[102:105], v226 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[66:71], v[114:119], a[76:79], v178, v180 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[106:107], v227 offset:47616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[72:77], v[114:119], a[80:83], v179, v180 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[108:111], v226 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[78:83], v[114:119], a[84:87], v179, v180 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[112:113], v227 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[84:89], v[114:119], a[88:91], v179, v180 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[8:11], s99 offen offset:1024 lds
	ds_read_b128 v[126:129], v226 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[90:95], v[114:119], a[92:95], v179, v180 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[130:131], v227 offset:48640
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[90:95], v[150:155], a[96:99], v179, v180 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[114:117], v226 offset:34816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[84:89], v[150:155], a[100:103], v179, v180 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[118:119], v227 offset:49152
	s_add_i32 s98, s64, s98
	s_mov_b32 m0, s64
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[78:83], v[150:155], a[104:107], v179, v180 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[12:15], s98 offen lds
	ds_read_b128 v[132:135], v226 offset:35840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[72:77], v[150:155], a[108:111], v179, v180 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[136:137], v227 offset:49664
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[66:71], v[150:155], a[112:115], v178, v180 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[138:141], v226 offset:36864
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[60:65], v[150:155], a[116:119], v178, v180 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[142:143], v227 offset:50176
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[54:59], v[150:155], a[120:123], v178, v180 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[24:27], s98 offen offset:1024 lds
	ds_read_b128 v[144:147], v226 offset:37888
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[6:11], v[150:155], a[124:127], v178, v180 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[148:149], v227 offset:50688
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[6:11], v[120:125], a[128:131], v178, v181 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[54:59], v[120:125], a[132:135], v178, v181 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[60:65], v[120:125], a[136:139], v178, v181 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[66:71], v[120:125], a[140:143], v178, v181 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s96, s56, s96
	s_add_i32 s96, s96, 0x100
	s_mov_b32 m0, s66
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[72:77], v[120:125], a[144:147], v179, v181 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[20:23], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[78:83], v[120:125], a[148:151], v179, v181 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[84:89], v[120:125], a[152:155], v179, v181 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s70
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[90:95], v[120:125], a[156:159], v179, v181 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[28:31], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[90:95], v[156:161], a[160:163], v179, v181 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[84:89], v[156:161], a[164:167], v179, v181 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[78:83], v[156:161], a[168:171], v179, v181 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s71
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[72:77], v[156:161], a[172:175], v179, v181 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[32:35], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[66:71], v[156:161], a[176:179], v178, v181 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[60:65], v[156:161], a[180:183], v178, v181 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s72
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[54:59], v[156:161], a[184:187], v178, v181 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[36:39], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[6:11], v[156:161], a[188:191], v178, v181 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[6:11], v[162:167], a[192:195], v178, v181 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[54:59], v[162:167], a[196:199], v178, v181 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s73
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[60:65], v[162:167], a[200:203], v178, v181 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[40:43], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[66:71], v[162:167], a[204:207], v178, v181 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[72:77], v[162:167], a[208:211], v179, v181 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s74
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[78:83], v[162:167], a[212:215], v179, v181 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[44:47], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[84:89], v[162:167], a[216:219], v179, v181 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[90:95], v[162:167], a[220:223], v179, v181 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[90:95], v[168:173], a[224:227], v179, v181 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s75
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[84:89], v[168:173], a[228:231], v179, v181 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[48:51], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[78:83], v[168:173], a[232:235], v179, v181 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[72:77], v[168:173], a[236:239], v179, v181 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s76
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[66:71], v[168:173], a[240:243], v178, v181 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[52:55], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[60:65], v[168:173], a[244:247], v178, v181 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[54:59], v[168:173], a[248:251], v178, v181 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[6:11], v[168:173], a[252:255], v178, v181 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s96, s94, 0x260
	s_lshl_b32 s96, s96, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[0:5], v[96:101], a[0:3], v174, v176 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(16) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[178:179], v224 offset:4096
	ds_read_b64 v[180:181], v225 offset:5120
	ds_read_b64 v[6:7], v246 offset:14432
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[12:17], v[96:101], a[4:7], v174, v176 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[8:9], v247 offset:14440
	ds_read_b64 v[10:11], v247 offset:14448
	s_add_i32 s98, s97, 0x72000
	s_mov_b32 m0, s88
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:23], v[96:101], a[8:11], v174, v176 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s98 offen lds
	ds_read_b64 v[54:55], v246 offset:14560
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[24:29], v[96:101], a[12:15], v174, v176 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[56:57], v247 offset:14568
	ds_read_b64 v[58:59], v247 offset:14576
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[30:35], v[96:101], a[16:19], v175, v176 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[60:61], v246 offset:14688
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[36:41], v[96:101], a[20:23], v175, v176 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[62:63], v247 offset:14696
	ds_read_b64 v[64:65], v247 offset:14704
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[42:47], v[96:101], a[24:27], v175, v176 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s98 offen offset:1024 lds
	ds_read_b64 v[66:67], v246 offset:14816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[48:53], v[96:101], a[28:31], v175, v176 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[68:69], v247 offset:14824
	ds_read_b64 v[70:71], v247 offset:14832
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[48:53], v[102:107], a[32:35], v175, v176 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[72:73], v246 offset:14944
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[42:47], v[102:107], a[36:39], v175, v176 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[74:75], v247 offset:14952
	ds_read_b64 v[76:77], v247 offset:14960
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[36:41], v[102:107], a[40:43], v175, v176 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s98 offen offset:2048 lds
	ds_read_b64 v[78:79], v246 offset:15072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[30:35], v[102:107], a[44:47], v175, v176 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[80:81], v247 offset:15080
	ds_read_b64 v[82:83], v247 offset:15088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[24:29], v[102:107], a[48:51], v174, v176 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[84:85], v246 offset:15200
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[18:23], v[102:107], a[52:55], v174, v176 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[86:87], v247 offset:15208
	ds_read_b64 v[88:89], v247 offset:15216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[12:17], v[102:107], a[56:59], v174, v176 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s98 offen offset:3072 lds
	ds_read_b64 v[90:91], v246 offset:15328
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[0:5], v[102:107], a[60:63], v174, v176 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[92:93], v247 offset:15336
	ds_read_b64 v[94:95], v247 offset:15344
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[0:5], v[108:113], a[64:67], v174, v176 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[96:99], v226 offset:55296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[12:17], v[108:113], a[68:71], v174, v176 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[100:101], v238 offset:6144
	s_mov_b32 m0, s89
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[18:23], v[108:113], a[72:75], v174, v176 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[8:11], s98 offen lds
	ds_read_b128 v[102:105], v226 offset:56320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[24:29], v[108:113], a[76:79], v174, v176 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[106:107], v238 offset:6656
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[30:35], v[108:113], a[80:83], v175, v176 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[120:123], v226 offset:57344
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[36:41], v[108:113], a[84:87], v175, v176 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[124:125], v238 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[42:47], v[108:113], a[88:91], v175, v176 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[8:11], s98 offen offset:1024 lds
	ds_read_b128 v[150:153], v226 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[48:53], v[108:113], a[92:95], v175, v176 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[154:155], v238 offset:7680
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[48:53], v[126:131], a[96:99], v175, v176 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[108:111], v226 offset:59392
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[42:47], v[126:131], a[100:103], v175, v176 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[112:113], v238 offset:8192
	s_add_i32 s96, s64, s96
	s_mov_b32 m0, s90
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[36:41], v[126:131], a[104:107], v175, v176 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[12:15], s96 offen lds
	ds_read_b128 v[156:159], v226 offset:60416
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[30:35], v[126:131], a[108:111], v175, v176 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[160:161], v238 offset:8704
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[24:29], v[126:131], a[112:115], v174, v176 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[162:165], v226 offset:61440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[18:23], v[126:131], a[116:119], v174, v176 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[166:167], v238 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[12:17], v[126:131], a[120:123], v174, v176 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[24:27], s96 offen offset:1024 lds
	ds_read_b128 v[168:171], v226 offset:62464
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[0:5], v[126:131], a[124:127], v174, v176 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[172:173], v238 offset:9728
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[0:5], v[114:119], a[128:131], v174, v177 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[12:17], v[114:119], a[132:135], v174, v177 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[18:23], v[114:119], a[136:139], v174, v177 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[24:29], v[114:119], a[140:143], v174, v177 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[30:35], v[114:119], a[144:147], v175, v177 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[36:41], v[114:119], a[148:151], v175, v177 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[42:47], v[114:119], a[152:155], v175, v177 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[48:53], v[114:119], a[156:159], v175, v177 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[48:53], v[132:137], a[160:163], v175, v177 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[42:47], v[132:137], a[164:167], v175, v177 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[36:41], v[132:137], a[168:171], v175, v177 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[30:35], v[132:137], a[172:175], v175, v177 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[24:29], v[132:137], a[176:179], v174, v177 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[18:23], v[132:137], a[180:183], v174, v177 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[12:17], v[132:137], a[184:187], v174, v177 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[0:5], v[132:137], a[188:191], v174, v177 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[0:5], v[138:143], a[192:195], v174, v177 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[12:17], v[138:143], a[196:199], v174, v177 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[18:23], v[138:143], a[200:203], v174, v177 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[24:29], v[138:143], a[204:207], v174, v177 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[30:35], v[138:143], a[208:211], v175, v177 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[36:41], v[138:143], a[212:215], v175, v177 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[42:47], v[138:143], a[216:219], v175, v177 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[48:53], v[138:143], a[220:223], v175, v177 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[48:53], v[144:149], a[224:227], v175, v177 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[42:47], v[144:149], a[228:231], v175, v177 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[36:41], v[144:149], a[232:235], v175, v177 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[30:35], v[144:149], a[236:239], v175, v177 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[24:29], v[144:149], a[240:243], v174, v177 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[18:23], v[144:149], a[244:247], v174, v177 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[12:17], v[144:149], a[248:251], v174, v177 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[0:5], v[144:149], a[252:255], v174, v177 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s96, s94, 0x280
	s_lshl_b32 s96, s96, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[6:11], v[96:101], a[0:3], v178, v180 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(8) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[174:175], v224
	ds_read_b64 v[176:177], v225 offset:1024
	ds_read_b64 v[0:1], v236 offset:47424
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[54:59], v[96:101], a[4:7], v178, v180 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[2:3], v236 offset:47432
	ds_read_b64 v[4:5], v237 offset:47440
	s_add_i32 s98, s97, 0x78000
	s_mov_b32 m0, s91
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[60:65], v[96:101], a[8:11], v178, v180 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s98 offen lds
	ds_read_b64 v[12:13], v236 offset:47552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:71], v[96:101], a[12:15], v178, v180 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[14:15], v236 offset:47560
	ds_read_b64 v[16:17], v237 offset:47568
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[72:77], v[96:101], a[16:19], v179, v180 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[18:19], v236 offset:47680
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[78:83], v[96:101], a[20:23], v179, v180 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[20:21], v236 offset:47688
	ds_read_b64 v[22:23], v237 offset:47696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[84:89], v[96:101], a[24:27], v179, v180 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s98 offen offset:1024 lds
	ds_read_b64 v[24:25], v236 offset:47808
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[90:95], v[96:101], a[28:31], v179, v180 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[26:27], v236 offset:47816
	ds_read_b64 v[28:29], v237 offset:47824
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[90:95], v[102:107], a[32:35], v179, v180 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[30:31], v236 offset:47936
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[84:89], v[102:107], a[36:39], v179, v180 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[32:33], v236 offset:47944
	ds_read_b64 v[34:35], v237 offset:47952
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[78:83], v[102:107], a[40:43], v179, v180 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s98 offen offset:2048 lds
	ds_read_b64 v[36:37], v236 offset:48064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[72:77], v[102:107], a[44:47], v179, v180 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[38:39], v236 offset:48072
	ds_read_b64 v[40:41], v237 offset:48080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[66:71], v[102:107], a[48:51], v178, v180 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[42:43], v236 offset:48192
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[60:65], v[102:107], a[52:55], v178, v180 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[44:45], v236 offset:48200
	ds_read_b64 v[46:47], v237 offset:48208
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[54:59], v[102:107], a[56:59], v178, v180 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s98 offen offset:3072 lds
	ds_read_b64 v[48:49], v236 offset:48320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[6:11], v[102:107], a[60:63], v178, v180 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[50:51], v236 offset:48328
	ds_read_b64 v[52:53], v237 offset:48336
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[6:11], v[120:125], a[64:67], v178, v180 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[96:99], v226 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[54:59], v[120:125], a[68:71], v178, v180 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[100:101], v227 offset:22528
	s_mov_b32 m0, s92
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[60:65], v[120:125], a[72:75], v178, v180 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[8:11], s98 offen lds
	ds_read_b128 v[102:105], v226 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[66:71], v[120:125], a[76:79], v178, v180 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[106:107], v227 offset:23040
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[72:77], v[120:125], a[80:83], v179, v180 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[114:117], v226 offset:8192
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[78:83], v[120:125], a[84:87], v179, v180 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[118:119], v227 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[84:89], v[120:125], a[88:91], v179, v180 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[8:11], s98 offen offset:1024 lds
	ds_read_b128 v[126:129], v226 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[90:95], v[120:125], a[92:95], v179, v180 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[130:131], v227 offset:24064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[90:95], v[150:155], a[96:99], v179, v180 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[120:123], v226 offset:10240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[84:89], v[150:155], a[100:103], v179, v180 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[124:125], v227 offset:24576
	s_add_i32 s96, s64, s96
	s_mov_b32 m0, s93
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[78:83], v[150:155], a[104:107], v179, v180 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[12:15], s96 offen lds
	ds_read_b128 v[132:135], v226 offset:11264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[72:77], v[150:155], a[108:111], v179, v180 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[136:137], v227 offset:25088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[66:71], v[150:155], a[112:115], v178, v180 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[138:141], v226 offset:12288
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[60:65], v[150:155], a[116:119], v178, v180 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[142:143], v227 offset:25600
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[54:59], v[150:155], a[120:123], v178, v180 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[24:27], s96 offen offset:1024 lds
	ds_read_b128 v[144:147], v226 offset:13312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[6:11], v[150:155], a[124:127], v178, v180 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[148:149], v227 offset:26112
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[6:11], v[108:113], a[128:131], v178, v181 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[54:59], v[108:113], a[132:135], v178, v181 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[60:65], v[108:113], a[136:139], v178, v181 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[66:71], v[108:113], a[140:143], v178, v181 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s96, s95, 0x780
	s_mov_b32 m0, s80
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[72:77], v[108:113], a[144:147], v179, v181 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[20:23], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[78:83], v[108:113], a[148:151], v179, v181 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[84:89], v[108:113], a[152:155], v179, v181 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s81
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[90:95], v[108:113], a[156:159], v179, v181 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[28:31], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[90:95], v[156:161], a[160:163], v179, v181 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[84:89], v[156:161], a[164:167], v179, v181 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[78:83], v[156:161], a[168:171], v179, v181 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s82
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[72:77], v[156:161], a[172:175], v179, v181 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[32:35], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[66:71], v[156:161], a[176:179], v178, v181 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[60:65], v[156:161], a[180:183], v178, v181 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s83
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[54:59], v[156:161], a[184:187], v178, v181 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[36:39], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[6:11], v[156:161], a[188:191], v178, v181 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[6:11], v[162:167], a[192:195], v178, v181 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[54:59], v[162:167], a[196:199], v178, v181 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s84
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[60:65], v[162:167], a[200:203], v178, v181 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[40:43], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[66:71], v[162:167], a[204:207], v178, v181 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[72:77], v[162:167], a[208:211], v179, v181 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s85
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[78:83], v[162:167], a[212:215], v179, v181 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[44:47], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[84:89], v[162:167], a[216:219], v179, v181 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[90:95], v[162:167], a[220:223], v179, v181 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[90:95], v[168:173], a[224:227], v179, v181 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s86
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[84:89], v[168:173], a[228:231], v179, v181 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[48:51], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[78:83], v[168:173], a[232:235], v179, v181 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[72:77], v[168:173], a[236:239], v179, v181 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s87
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[66:71], v[168:173], a[240:243], v178, v181 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[52:55], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[60:65], v[168:173], a[244:247], v178, v181 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[54:59], v[168:173], a[248:251], v178, v181 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[6:11], v[168:173], a[252:255], v178, v181 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s96, s3, 21
	s_add_i32 s98, s94, 0x2a0
	s_lshl_b32 s98, s98, 9
	s_lshr_b32 s96, s96, 2
	s_mul_i32 s96, s96, 3
	s_lshl_b32 s96, s96, 7
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[0:5], v[96:101], a[0:3], v174, v176 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(16) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[178:179], v224 offset:2048
	ds_read_b64 v[180:181], v225 offset:3072
	ds_read_b64 v[6:7], v239 offset:14368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[12:17], v[96:101], a[4:7], v174, v176 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[8:9], v239 offset:14376
	ds_read_b64 v[10:11], v239 offset:14384
	s_add_i32 s99, s97, 0x7e000
	s_mov_b32 m0, s77
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:23], v[96:101], a[8:11], v174, v176 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s99 offen lds
	ds_read_b64 v[54:55], v239 offset:14496
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[24:29], v[96:101], a[12:15], v174, v176 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[56:57], v239 offset:14504
	ds_read_b64 v[58:59], v239 offset:14512
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[30:35], v[96:101], a[16:19], v175, v176 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[60:61], v239 offset:14624
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[36:41], v[96:101], a[20:23], v175, v176 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[62:63], v239 offset:14632
	ds_read_b64 v[64:65], v239 offset:14640
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[42:47], v[96:101], a[24:27], v175, v176 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s99 offen offset:1024 lds
	ds_read_b64 v[66:67], v239 offset:14752
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[48:53], v[96:101], a[28:31], v175, v176 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[68:69], v239 offset:14760
	ds_read_b64 v[70:71], v239 offset:14768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[48:53], v[102:107], a[32:35], v175, v176 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[72:73], v239 offset:14880
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[42:47], v[102:107], a[36:39], v175, v176 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[74:75], v239 offset:14888
	ds_read_b64 v[76:77], v239 offset:14896
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[36:41], v[102:107], a[40:43], v175, v176 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s99 offen offset:2048 lds
	ds_read_b64 v[78:79], v239 offset:15008
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[30:35], v[102:107], a[44:47], v175, v176 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[80:81], v239 offset:15016
	ds_read_b64 v[82:83], v239 offset:15024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[24:29], v[102:107], a[48:51], v174, v176 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[84:85], v239 offset:15136
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[18:23], v[102:107], a[52:55], v174, v176 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[86:87], v239 offset:15144
	ds_read_b64 v[88:89], v239 offset:15152
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[12:17], v[102:107], a[56:59], v174, v176 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s99 offen offset:3072 lds
	ds_read_b64 v[90:91], v239 offset:15264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[0:5], v[102:107], a[60:63], v174, v176 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[92:93], v239 offset:15272
	ds_read_b64 v[94:95], v239 offset:15280
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[0:5], v[114:119], a[64:67], v174, v176 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[96:99], v226 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[12:17], v[114:119], a[68:71], v174, v176 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[100:101], v227 offset:47104
	s_mov_b32 m0, s78
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[18:23], v[114:119], a[72:75], v174, v176 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[8:11], s99 offen lds
	ds_read_b128 v[102:105], v226 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[24:29], v[114:119], a[76:79], v174, v176 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[106:107], v227 offset:47616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[30:35], v[114:119], a[80:83], v175, v176 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[108:111], v226 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[36:41], v[114:119], a[84:87], v175, v176 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[112:113], v227 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[42:47], v[114:119], a[88:91], v175, v176 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[8:11], s99 offen offset:1024 lds
	ds_read_b128 v[150:153], v226 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[48:53], v[114:119], a[92:95], v175, v176 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[154:155], v227 offset:48640
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[48:53], v[126:131], a[96:99], v175, v176 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[114:117], v226 offset:34816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[42:47], v[126:131], a[100:103], v175, v176 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[118:119], v227 offset:49152
	s_add_i32 s98, s64, s98
	s_mov_b32 m0, s64
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[36:41], v[126:131], a[104:107], v175, v176 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[12:15], s98 offen lds
	ds_read_b128 v[156:159], v226 offset:35840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[30:35], v[126:131], a[108:111], v175, v176 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[160:161], v227 offset:49664
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[24:29], v[126:131], a[112:115], v174, v176 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[162:165], v226 offset:36864
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[18:23], v[126:131], a[116:119], v174, v176 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[166:167], v227 offset:50176
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[12:17], v[126:131], a[120:123], v174, v176 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[24:27], s98 offen offset:1024 lds
	ds_read_b128 v[168:171], v226 offset:37888
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[0:5], v[126:131], a[124:127], v174, v176 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[172:173], v227 offset:50688
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[0:5], v[120:125], a[128:131], v174, v177 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[12:17], v[120:125], a[132:135], v174, v177 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[18:23], v[120:125], a[136:139], v174, v177 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[24:29], v[120:125], a[140:143], v174, v177 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s96, s56, s96
	s_add_i32 s96, s96, 0x80
	s_mov_b32 m0, s66
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[30:35], v[120:125], a[144:147], v175, v177 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[20:23], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[36:41], v[120:125], a[148:151], v175, v177 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[42:47], v[120:125], a[152:155], v175, v177 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s70
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[48:53], v[120:125], a[156:159], v175, v177 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[28:31], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[48:53], v[132:137], a[160:163], v175, v177 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[42:47], v[132:137], a[164:167], v175, v177 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[36:41], v[132:137], a[168:171], v175, v177 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s71
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[30:35], v[132:137], a[172:175], v175, v177 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[32:35], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[24:29], v[132:137], a[176:179], v174, v177 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[18:23], v[132:137], a[180:183], v174, v177 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s72
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[12:17], v[132:137], a[184:187], v174, v177 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[36:39], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[0:5], v[132:137], a[188:191], v174, v177 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[0:5], v[138:143], a[192:195], v174, v177 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[12:17], v[138:143], a[196:199], v174, v177 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s73
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[18:23], v[138:143], a[200:203], v174, v177 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[40:43], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[24:29], v[138:143], a[204:207], v174, v177 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[30:35], v[138:143], a[208:211], v175, v177 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s74
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[36:41], v[138:143], a[212:215], v175, v177 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[44:47], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[42:47], v[138:143], a[216:219], v175, v177 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[48:53], v[138:143], a[220:223], v175, v177 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[48:53], v[144:149], a[224:227], v175, v177 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s75
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[42:47], v[144:149], a[228:231], v175, v177 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[48:51], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[36:41], v[144:149], a[232:235], v175, v177 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[30:35], v[144:149], a[236:239], v175, v177 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s76
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[24:29], v[144:149], a[240:243], v174, v177 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[52:55], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[18:23], v[144:149], a[244:247], v174, v177 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[12:17], v[144:149], a[248:251], v174, v177 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[0:5], v[144:149], a[252:255], v174, v177 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s96, s3, 22
	s_add_i32 s98, s94, 0x2c0
	s_lshl_b32 s98, s98, 9
	s_lshr_b32 s96, s96, 2
	s_mul_i32 s96, s96, 3
	s_lshl_b32 s96, s96, 7
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[6:11], v[96:101], a[0:3], v178, v180 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(16) lgkmcnt(0)
	s_barrier
	ds_read_b128 v[18:21], v240 offset:47360
	ds_read_b128 v[22:25], v240 offset:47376
	ds_read_b64 v[174:175], v224 offset:4096
	ds_read_b64 v[176:177], v225 offset:5120
	ds_read_b64 v[0:1], v239 offset:47360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[54:59], v[96:101], a[4:7], v178, v180 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[2:3], v239 offset:47368
	ds_read_b64 v[4:5], v239 offset:47376
	s_add_i32 s99, s97, 0x84000
	s_mov_b32 m0, s88
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[60:65], v[96:101], a[8:11], v178, v180 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s99 offen lds
	ds_read_b64 v[12:13], v239 offset:47488
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:71], v[96:101], a[12:15], v178, v180 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[14:15], v239 offset:47496
	ds_read_b64 v[16:17], v239 offset:47504
	s_waitcnt lgkmcnt(9)
	ds_write_b64 v241, v[18:19] offset:14336
	ds_write_b64 v241, v[20:21] offset:14344
	s_waitcnt lgkmcnt(10)
	ds_write_b64 v241, v[22:23] offset:14352
	ds_write_b64 v241, v[24:25] offset:14360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[72:77], v[96:101], a[16:19], v179, v180 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[18:19], v239 offset:47616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[78:83], v[96:101], a[20:23], v179, v180 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[20:21], v239 offset:47624
	ds_read_b64 v[22:23], v239 offset:47632
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[84:89], v[96:101], a[24:27], v179, v180 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s99 offen offset:1024 lds
	ds_read_b64 v[24:25], v239 offset:47744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[90:95], v[96:101], a[28:31], v179, v180 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[26:27], v239 offset:47752
	ds_read_b64 v[28:29], v239 offset:47760
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[90:95], v[102:107], a[32:35], v179, v180 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[30:31], v239 offset:47872
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[84:89], v[102:107], a[36:39], v179, v180 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[32:33], v239 offset:47880
	ds_read_b64 v[34:35], v239 offset:47888
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[78:83], v[102:107], a[40:43], v179, v180 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s99 offen offset:2048 lds
	ds_read_b64 v[36:37], v239 offset:48000
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[72:77], v[102:107], a[44:47], v179, v180 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[38:39], v239 offset:48008
	ds_read_b64 v[40:41], v239 offset:48016
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[66:71], v[102:107], a[48:51], v178, v180 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[42:43], v239 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[60:65], v[102:107], a[52:55], v178, v180 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[44:45], v239 offset:48136
	ds_read_b64 v[46:47], v239 offset:48144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[54:59], v[102:107], a[56:59], v178, v180 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s99 offen offset:3072 lds
	ds_read_b64 v[48:49], v239 offset:48256
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[6:11], v[102:107], a[60:63], v178, v180 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[50:51], v239 offset:48264
	ds_read_b64 v[52:53], v239 offset:48272
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[6:11], v[108:113], a[64:67], v178, v180 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[96:99], v226 offset:55296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[54:59], v[108:113], a[68:71], v178, v180 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[100:101], v238 offset:6144
	s_mov_b32 m0, s89
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[60:65], v[108:113], a[72:75], v178, v180 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[8:11], s99 offen lds
	ds_read_b128 v[102:105], v226 offset:56320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[66:71], v[108:113], a[76:79], v178, v180 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[106:107], v238 offset:6656
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[72:77], v[108:113], a[80:83], v179, v180 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[120:123], v226 offset:57344
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[78:83], v[108:113], a[84:87], v179, v180 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[124:125], v238 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[84:89], v[108:113], a[88:91], v179, v180 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[8:11], s99 offen offset:1024 lds
	ds_read_b128 v[126:129], v226 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[90:95], v[108:113], a[92:95], v179, v180 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[130:131], v238 offset:7680
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[90:95], v[150:155], a[96:99], v179, v180 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[108:111], v226 offset:59392
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[84:89], v[150:155], a[100:103], v179, v180 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[112:113], v238 offset:8192
	s_add_i32 s98, s64, s98
	s_mov_b32 m0, s90
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[78:83], v[150:155], a[104:107], v179, v180 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[12:15], s98 offen lds
	ds_read_b128 v[132:135], v226 offset:60416
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[72:77], v[150:155], a[108:111], v179, v180 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[136:137], v238 offset:8704
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[66:71], v[150:155], a[112:115], v178, v180 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[138:141], v226 offset:61440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[60:65], v[150:155], a[116:119], v178, v180 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[142:143], v238 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[54:59], v[150:155], a[120:123], v178, v180 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[24:27], s98 offen offset:1024 lds
	ds_read_b128 v[144:147], v226 offset:62464
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[6:11], v[150:155], a[124:127], v178, v180 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[148:149], v238 offset:9728
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[6:11], v[114:119], a[128:131], v178, v181 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[54:59], v[114:119], a[132:135], v178, v181 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[60:65], v[114:119], a[136:139], v178, v181 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[66:71], v[114:119], a[140:143], v178, v181 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s96, s56, s96
	s_add_i32 s96, s96, 0x100
	s_mov_b32 m0, s80
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[72:77], v[114:119], a[144:147], v179, v181 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[20:23], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[78:83], v[114:119], a[148:151], v179, v181 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[84:89], v[114:119], a[152:155], v179, v181 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s81
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[90:95], v[114:119], a[156:159], v179, v181 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[28:31], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[90:95], v[156:161], a[160:163], v179, v181 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[84:89], v[156:161], a[164:167], v179, v181 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[78:83], v[156:161], a[168:171], v179, v181 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s82
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[72:77], v[156:161], a[172:175], v179, v181 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[32:35], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[66:71], v[156:161], a[176:179], v178, v181 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[60:65], v[156:161], a[180:183], v178, v181 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s83
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[54:59], v[156:161], a[184:187], v178, v181 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[36:39], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[6:11], v[156:161], a[188:191], v178, v181 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[6:11], v[162:167], a[192:195], v178, v181 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[54:59], v[162:167], a[196:199], v178, v181 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s84
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[60:65], v[162:167], a[200:203], v178, v181 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[40:43], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[66:71], v[162:167], a[204:207], v178, v181 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[72:77], v[162:167], a[208:211], v179, v181 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s85
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[78:83], v[162:167], a[212:215], v179, v181 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[44:47], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[84:89], v[162:167], a[216:219], v179, v181 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[90:95], v[162:167], a[220:223], v179, v181 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[90:95], v[168:173], a[224:227], v179, v181 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s86
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[84:89], v[168:173], a[228:231], v179, v181 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[48:51], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[78:83], v[168:173], a[232:235], v179, v181 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[72:77], v[168:173], a[236:239], v179, v181 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s87
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[66:71], v[168:173], a[240:243], v178, v181 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[52:55], s96 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[60:65], v[168:173], a[244:247], v178, v181 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[54:59], v[168:173], a[248:251], v178, v181 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[6:11], v[168:173], a[252:255], v178, v181 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s96, s94, 0x2e0
	s_lshl_b32 s96, s96, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[0:5], v[96:101], a[0:3], v174, v176 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(16) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[178:179], v224
	ds_read_b64 v[180:181], v225 offset:1024
	ds_read_b64 v[6:7], v242 offset:47456
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[12:17], v[96:101], a[4:7], v174, v176 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[8:9], v243 offset:47464
	ds_read_b64 v[10:11], v243 offset:47472
	s_add_i32 s98, s97, 0x8a000
	s_mov_b32 m0, s91
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:23], v[96:101], a[8:11], v174, v176 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s98 offen lds
	ds_read_b64 v[54:55], v242 offset:47584
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[24:29], v[96:101], a[12:15], v174, v176 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[56:57], v243 offset:47592
	ds_read_b64 v[58:59], v243 offset:47600
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[30:35], v[96:101], a[16:19], v175, v176 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[60:61], v242 offset:47712
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[36:41], v[96:101], a[20:23], v175, v176 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[62:63], v243 offset:47720
	ds_read_b64 v[64:65], v243 offset:47728
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[42:47], v[96:101], a[24:27], v175, v176 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s98 offen offset:1024 lds
	ds_read_b64 v[66:67], v242 offset:47840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[48:53], v[96:101], a[28:31], v175, v176 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[68:69], v243 offset:47848
	ds_read_b64 v[70:71], v243 offset:47856
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[48:53], v[102:107], a[32:35], v175, v176 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[72:73], v242 offset:47968
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[42:47], v[102:107], a[36:39], v175, v176 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[74:75], v243 offset:47976
	ds_read_b64 v[76:77], v243 offset:47984
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[36:41], v[102:107], a[40:43], v175, v176 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s98 offen offset:2048 lds
	ds_read_b64 v[78:79], v242 offset:48096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[30:35], v[102:107], a[44:47], v175, v176 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[80:81], v243 offset:48104
	ds_read_b64 v[82:83], v243 offset:48112
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[24:29], v[102:107], a[48:51], v174, v176 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[84:85], v242 offset:48224
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[18:23], v[102:107], a[52:55], v174, v176 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[86:87], v243 offset:48232
	ds_read_b64 v[88:89], v243 offset:48240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[12:17], v[102:107], a[56:59], v174, v176 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s98 offen offset:3072 lds
	ds_read_b64 v[90:91], v242 offset:48352
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[0:5], v[102:107], a[60:63], v174, v176 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[92:93], v243 offset:48360
	ds_read_b64 v[94:95], v243 offset:48368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[0:5], v[120:125], a[64:67], v174, v176 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[96:99], v226 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[12:17], v[120:125], a[68:71], v174, v176 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[100:101], v227 offset:22528
	s_mov_b32 m0, s92
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[18:23], v[120:125], a[72:75], v174, v176 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[8:11], s98 offen lds
	ds_read_b128 v[102:105], v226 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[24:29], v[120:125], a[76:79], v174, v176 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[106:107], v227 offset:23040
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[30:35], v[120:125], a[80:83], v175, v176 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[114:117], v226 offset:8192
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[36:41], v[120:125], a[84:87], v175, v176 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[118:119], v227 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[42:47], v[120:125], a[88:91], v175, v176 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[8:11], s98 offen offset:1024 lds
	ds_read_b128 v[150:153], v226 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[48:53], v[120:125], a[92:95], v175, v176 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[154:155], v227 offset:24064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[48:53], v[126:131], a[96:99], v175, v176 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[120:123], v226 offset:10240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[42:47], v[126:131], a[100:103], v175, v176 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[124:125], v227 offset:24576
	s_add_i32 s96, s64, s96
	s_mov_b32 m0, s93
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[36:41], v[126:131], a[104:107], v175, v176 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[12:15], s96 offen lds
	ds_read_b128 v[156:159], v226 offset:11264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[30:35], v[126:131], a[108:111], v175, v176 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[160:161], v227 offset:25088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[24:29], v[126:131], a[112:115], v174, v176 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[162:165], v226 offset:12288
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[18:23], v[126:131], a[116:119], v174, v176 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[166:167], v227 offset:25600
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[12:17], v[126:131], a[120:123], v174, v176 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[24:27], s96 offen offset:1024 lds
	ds_read_b128 v[168:171], v226 offset:13312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[0:5], v[126:131], a[124:127], v174, v176 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[172:173], v227 offset:26112
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[0:5], v[108:113], a[128:131], v174, v177 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[12:17], v[108:113], a[132:135], v174, v177 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[18:23], v[108:113], a[136:139], v174, v177 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[24:29], v[108:113], a[140:143], v174, v177 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[30:35], v[108:113], a[144:147], v175, v177 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[36:41], v[108:113], a[148:151], v175, v177 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[42:47], v[108:113], a[152:155], v175, v177 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[48:53], v[108:113], a[156:159], v175, v177 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[48:53], v[132:137], a[160:163], v175, v177 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[42:47], v[132:137], a[164:167], v175, v177 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[36:41], v[132:137], a[168:171], v175, v177 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[30:35], v[132:137], a[172:175], v175, v177 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[24:29], v[132:137], a[176:179], v174, v177 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[18:23], v[132:137], a[180:183], v174, v177 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[12:17], v[132:137], a[184:187], v174, v177 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[0:5], v[132:137], a[188:191], v174, v177 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[0:5], v[138:143], a[192:195], v174, v177 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[12:17], v[138:143], a[196:199], v174, v177 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[18:23], v[138:143], a[200:203], v174, v177 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[24:29], v[138:143], a[204:207], v174, v177 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[30:35], v[138:143], a[208:211], v175, v177 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[36:41], v[138:143], a[212:215], v175, v177 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[42:47], v[138:143], a[216:219], v175, v177 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[48:53], v[138:143], a[220:223], v175, v177 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[48:53], v[144:149], a[224:227], v175, v177 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[42:47], v[144:149], a[228:231], v175, v177 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[36:41], v[144:149], a[232:235], v175, v177 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[30:35], v[144:149], a[236:239], v175, v177 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[24:29], v[144:149], a[240:243], v174, v177 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[18:23], v[144:149], a[244:247], v174, v177 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[12:17], v[144:149], a[248:251], v174, v177 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[0:5], v[144:149], a[252:255], v174, v177 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s96, s94, 0x300
	s_lshl_b32 s96, s96, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[6:11], v[96:101], a[0:3], v178, v180 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(8) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[174:175], v224 offset:2048
	ds_read_b64 v[176:177], v225 offset:3072
	ds_read_b64 v[0:1], v244 offset:14400
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[54:59], v[96:101], a[4:7], v178, v180 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[2:3], v244 offset:14408
	ds_read_b64 v[4:5], v245 offset:14416
	s_add_i32 s98, s97, 0x90000
	s_mov_b32 m0, s77
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[60:65], v[96:101], a[8:11], v178, v180 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s98 offen lds
	ds_read_b64 v[12:13], v244 offset:14528
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:71], v[96:101], a[12:15], v178, v180 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[14:15], v244 offset:14536
	ds_read_b64 v[16:17], v245 offset:14544
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[72:77], v[96:101], a[16:19], v179, v180 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[18:19], v244 offset:14656
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[78:83], v[96:101], a[20:23], v179, v180 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[20:21], v244 offset:14664
	ds_read_b64 v[22:23], v245 offset:14672
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[84:89], v[96:101], a[24:27], v179, v180 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s98 offen offset:1024 lds
	ds_read_b64 v[24:25], v244 offset:14784
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[90:95], v[96:101], a[28:31], v179, v180 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[26:27], v244 offset:14792
	ds_read_b64 v[28:29], v245 offset:14800
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[90:95], v[102:107], a[32:35], v179, v180 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[30:31], v244 offset:14912
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[84:89], v[102:107], a[36:39], v179, v180 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[32:33], v244 offset:14920
	ds_read_b64 v[34:35], v245 offset:14928
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[78:83], v[102:107], a[40:43], v179, v180 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s98 offen offset:2048 lds
	ds_read_b64 v[36:37], v244 offset:15040
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[72:77], v[102:107], a[44:47], v179, v180 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[38:39], v244 offset:15048
	ds_read_b64 v[40:41], v245 offset:15056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[66:71], v[102:107], a[48:51], v178, v180 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[42:43], v244 offset:15168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[60:65], v[102:107], a[52:55], v178, v180 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[44:45], v244 offset:15176
	ds_read_b64 v[46:47], v245 offset:15184
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[54:59], v[102:107], a[56:59], v178, v180 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s98 offen offset:3072 lds
	ds_read_b64 v[48:49], v244 offset:15296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[6:11], v[102:107], a[60:63], v178, v180 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[50:51], v244 offset:15304
	ds_read_b64 v[52:53], v245 offset:15312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[6:11], v[114:119], a[64:67], v178, v180 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[96:99], v226 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[54:59], v[114:119], a[68:71], v178, v180 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[100:101], v227 offset:47104
	s_mov_b32 m0, s78
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[60:65], v[114:119], a[72:75], v178, v180 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[8:11], s98 offen lds
	ds_read_b128 v[102:105], v226 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[66:71], v[114:119], a[76:79], v178, v180 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[106:107], v227 offset:47616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[72:77], v[114:119], a[80:83], v179, v180 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[108:111], v226 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[78:83], v[114:119], a[84:87], v179, v180 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[112:113], v227 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[84:89], v[114:119], a[88:91], v179, v180 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[8:11], s98 offen offset:1024 lds
	ds_read_b128 v[126:129], v226 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[90:95], v[114:119], a[92:95], v179, v180 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[130:131], v227 offset:48640
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[90:95], v[150:155], a[96:99], v179, v180 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[114:117], v226 offset:34816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[84:89], v[150:155], a[100:103], v179, v180 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[118:119], v227 offset:49152
	s_add_i32 s96, s64, s96
	s_mov_b32 m0, s64
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[78:83], v[150:155], a[104:107], v179, v180 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[12:15], s96 offen lds
	ds_read_b128 v[132:135], v226 offset:35840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[72:77], v[150:155], a[108:111], v179, v180 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[136:137], v227 offset:49664
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[66:71], v[150:155], a[112:115], v178, v180 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[138:141], v226 offset:36864
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[60:65], v[150:155], a[116:119], v178, v180 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[142:143], v227 offset:50176
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[54:59], v[150:155], a[120:123], v178, v180 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[24:27], s96 offen offset:1024 lds
	ds_read_b128 v[144:147], v226 offset:37888
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[6:11], v[150:155], a[124:127], v178, v180 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[148:149], v227 offset:50688
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[6:11], v[120:125], a[128:131], v178, v181 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[54:59], v[120:125], a[132:135], v178, v181 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[60:65], v[120:125], a[136:139], v178, v181 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[66:71], v[120:125], a[140:143], v178, v181 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s95, s95, 0x900
	s_mov_b32 m0, s66
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[72:77], v[120:125], a[144:147], v179, v181 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[20:23], s95 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[78:83], v[120:125], a[148:151], v179, v181 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[84:89], v[120:125], a[152:155], v179, v181 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s70
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[90:95], v[120:125], a[156:159], v179, v181 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[28:31], s95 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[90:95], v[156:161], a[160:163], v179, v181 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[84:89], v[156:161], a[164:167], v179, v181 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[78:83], v[156:161], a[168:171], v179, v181 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s71
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[72:77], v[156:161], a[172:175], v179, v181 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[32:35], s95 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[66:71], v[156:161], a[176:179], v178, v181 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[60:65], v[156:161], a[180:183], v178, v181 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s72
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[54:59], v[156:161], a[184:187], v178, v181 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[36:39], s95 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[6:11], v[156:161], a[188:191], v178, v181 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[6:11], v[162:167], a[192:195], v178, v181 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[54:59], v[162:167], a[196:199], v178, v181 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s73
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[60:65], v[162:167], a[200:203], v178, v181 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[40:43], s95 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[66:71], v[162:167], a[204:207], v178, v181 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[72:77], v[162:167], a[208:211], v179, v181 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s74
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[78:83], v[162:167], a[212:215], v179, v181 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[44:47], s95 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[84:89], v[162:167], a[216:219], v179, v181 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[90:95], v[162:167], a[220:223], v179, v181 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[90:95], v[168:173], a[224:227], v179, v181 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s75
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[84:89], v[168:173], a[228:231], v179, v181 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[48:51], s95 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[78:83], v[168:173], a[232:235], v179, v181 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[72:77], v[168:173], a[236:239], v179, v181 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s76
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[66:71], v[168:173], a[240:243], v178, v181 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[52:55], s95 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[60:65], v[168:173], a[244:247], v178, v181 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[54:59], v[168:173], a[248:251], v178, v181 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[6:11], v[168:173], a[252:255], v178, v181 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s95, s3, 25
	s_add_i32 s96, s94, 0x320
	s_lshl_b32 s96, s96, 9
	s_lshr_b32 s95, s95, 2
	s_mul_i32 s95, s95, 3
	s_lshl_b32 s95, s95, 7
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[0:5], v[96:101], a[0:3], v174, v176 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(16) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[210:211], v224 offset:4096
	ds_read_b64 v[212:213], v225 offset:5120
	ds_read_b64 v[6:7], v239 offset:47392
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[12:17], v[96:101], a[4:7], v174, v176 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[8:9], v239 offset:47400
	ds_read_b64 v[10:11], v239 offset:47408
	s_add_i32 s98, s97, 0x96000
	s_mov_b32 m0, s88
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:23], v[96:101], a[8:11], v174, v176 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s98 offen lds
	ds_read_b64 v[54:55], v239 offset:47520
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[24:29], v[96:101], a[12:15], v174, v176 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[56:57], v239 offset:47528
	ds_read_b64 v[58:59], v239 offset:47536
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[30:35], v[96:101], a[16:19], v175, v176 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[60:61], v239 offset:47648
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[36:41], v[96:101], a[20:23], v175, v176 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[62:63], v239 offset:47656
	ds_read_b64 v[64:65], v239 offset:47664
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[42:47], v[96:101], a[24:27], v175, v176 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s98 offen offset:1024 lds
	ds_read_b64 v[66:67], v239 offset:47776
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[48:53], v[96:101], a[28:31], v175, v176 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[68:69], v239 offset:47784
	ds_read_b64 v[70:71], v239 offset:47792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[48:53], v[102:107], a[32:35], v175, v176 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[72:73], v239 offset:47904
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[42:47], v[102:107], a[36:39], v175, v176 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[74:75], v239 offset:47912
	ds_read_b64 v[76:77], v239 offset:47920
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[36:41], v[102:107], a[40:43], v175, v176 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s98 offen offset:2048 lds
	ds_read_b64 v[78:79], v239 offset:48032
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[30:35], v[102:107], a[44:47], v175, v176 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[80:81], v239 offset:48040
	ds_read_b64 v[82:83], v239 offset:48048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[24:29], v[102:107], a[48:51], v174, v176 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[84:85], v239 offset:48160
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[18:23], v[102:107], a[52:55], v174, v176 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[86:87], v239 offset:48168
	ds_read_b64 v[88:89], v239 offset:48176
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[12:17], v[102:107], a[56:59], v174, v176 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s98 offen offset:3072 lds
	ds_read_b64 v[90:91], v239 offset:48288
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[0:5], v[102:107], a[60:63], v174, v176 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[92:93], v239 offset:48296
	ds_read_b64 v[94:95], v239 offset:48304
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[0:5], v[108:113], a[64:67], v174, v176 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[96:99], v226 offset:55296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[12:17], v[108:113], a[68:71], v174, v176 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[100:101], v238 offset:6144
	s_mov_b32 m0, s89
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[18:23], v[108:113], a[72:75], v174, v176 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[8:11], s98 offen lds
	ds_read_b128 v[102:105], v226 offset:56320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[24:29], v[108:113], a[76:79], v174, v176 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[106:107], v238 offset:6656
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[30:35], v[108:113], a[80:83], v175, v176 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[120:123], v226 offset:57344
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[36:41], v[108:113], a[84:87], v175, v176 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[124:125], v238 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[42:47], v[108:113], a[88:91], v175, v176 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[8:11], s98 offen offset:1024 lds
	ds_read_b128 v[150:153], v226 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[48:53], v[108:113], a[92:95], v175, v176 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[154:155], v238 offset:7680
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[48:53], v[126:131], a[96:99], v175, v176 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[108:111], v226 offset:59392
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[42:47], v[126:131], a[100:103], v175, v176 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[112:113], v238 offset:8192
	s_add_i32 s96, s64, s96
	s_mov_b32 m0, s90
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[36:41], v[126:131], a[104:107], v175, v176 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[12:15], s96 offen lds
	ds_read_b128 v[156:159], v226 offset:60416
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[30:35], v[126:131], a[108:111], v175, v176 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[160:161], v238 offset:8704
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[24:29], v[126:131], a[112:115], v174, v176 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[162:165], v226 offset:61440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[18:23], v[126:131], a[116:119], v174, v176 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[166:167], v238 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[12:17], v[126:131], a[120:123], v174, v176 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[24:27], s96 offen offset:1024 lds
	ds_read_b128 v[168:171], v226 offset:62464
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[0:5], v[126:131], a[124:127], v174, v176 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[172:173], v238 offset:9728
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[0:5], v[114:119], a[128:131], v174, v177 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[12:17], v[114:119], a[132:135], v174, v177 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[18:23], v[114:119], a[136:139], v174, v177 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[24:29], v[114:119], a[140:143], v174, v177 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s95, s56, s95
	s_add_i32 s95, s95, 0x80
	s_mov_b32 m0, s80
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[30:35], v[114:119], a[144:147], v175, v177 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[20:23], s95 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[36:41], v[114:119], a[148:151], v175, v177 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[42:47], v[114:119], a[152:155], v175, v177 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s81
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[48:53], v[114:119], a[156:159], v175, v177 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[28:31], s95 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[48:53], v[132:137], a[160:163], v175, v177 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[42:47], v[132:137], a[164:167], v175, v177 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[36:41], v[132:137], a[168:171], v175, v177 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s82
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[30:35], v[132:137], a[172:175], v175, v177 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[32:35], s95 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[24:29], v[132:137], a[176:179], v174, v177 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[18:23], v[132:137], a[180:183], v174, v177 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s83
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[12:17], v[132:137], a[184:187], v174, v177 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[36:39], s95 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[0:5], v[132:137], a[188:191], v174, v177 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[0:5], v[138:143], a[192:195], v174, v177 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[12:17], v[138:143], a[196:199], v174, v177 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s84
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[18:23], v[138:143], a[200:203], v174, v177 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[40:43], s95 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[24:29], v[138:143], a[204:207], v174, v177 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[30:35], v[138:143], a[208:211], v175, v177 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s85
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[36:41], v[138:143], a[212:215], v175, v177 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[44:47], s95 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[42:47], v[138:143], a[216:219], v175, v177 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[48:53], v[138:143], a[220:223], v175, v177 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[48:53], v[144:149], a[224:227], v175, v177 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s86
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[42:47], v[144:149], a[228:231], v175, v177 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[48:51], s95 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[36:41], v[144:149], a[232:235], v175, v177 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[30:35], v[144:149], a[236:239], v175, v177 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s87
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[24:29], v[144:149], a[240:243], v174, v177 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[52:55], s95 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[18:23], v[144:149], a[244:247], v174, v177 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[12:17], v[144:149], a[248:251], v174, v177 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[0:5], v[144:149], a[252:255], v174, v177 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s3, s3, 26
	s_add_i32 s95, s94, 0x340
	s_lshl_b32 s95, s95, 9
	s_lshr_b32 s3, s3, 2
	s_mul_i32 s3, s3, 3
	s_lshl_b32 s3, s3, 7
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[6:11], v[96:101], a[0:3], v210, v212 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(16) lgkmcnt(0)
	s_barrier
	ds_read_b128 v[18:21], v240 offset:14336
	ds_read_b128 v[22:25], v240 offset:14352
	ds_read_b64 v[214:215], v224
	ds_read_b64 v[216:217], v225 offset:1024
	ds_read_b64 v[0:1], v239 offset:14336
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[54:59], v[96:101], a[4:7], v210, v212 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[2:3], v239 offset:14344
	ds_read_b64 v[4:5], v239 offset:14352
	s_add_i32 s96, s97, 0x9c000
	s_mov_b32 m0, s91
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[60:65], v[96:101], a[8:11], v210, v212 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s96 offen lds
	ds_read_b64 v[12:13], v239 offset:14464
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:71], v[96:101], a[12:15], v210, v212 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[14:15], v239 offset:14472
	ds_read_b64 v[16:17], v239 offset:14480
	s_waitcnt lgkmcnt(9)
	ds_write_b64 v241, v[18:19] offset:14336
	ds_write_b64 v241, v[20:21] offset:14344
	s_waitcnt lgkmcnt(10)
	ds_write_b64 v241, v[22:23] offset:14352
	ds_write_b64 v241, v[24:25] offset:14360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[72:77], v[96:101], a[16:19], v211, v212 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[18:19], v239 offset:14592
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[78:83], v[96:101], a[20:23], v211, v212 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[20:21], v239 offset:14600
	ds_read_b64 v[22:23], v239 offset:14608
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[84:89], v[96:101], a[24:27], v211, v212 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s96 offen offset:1024 lds
	ds_read_b64 v[24:25], v239 offset:14720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[90:95], v[96:101], a[28:31], v211, v212 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[26:27], v239 offset:14728
	ds_read_b64 v[28:29], v239 offset:14736
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[90:95], v[102:107], a[32:35], v211, v212 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[30:31], v239 offset:14848
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[84:89], v[102:107], a[36:39], v211, v212 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[32:33], v239 offset:14856
	ds_read_b64 v[34:35], v239 offset:14864
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[78:83], v[102:107], a[40:43], v211, v212 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s96 offen offset:2048 lds
	ds_read_b64 v[36:37], v239 offset:14976
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[72:77], v[102:107], a[44:47], v211, v212 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[38:39], v239 offset:14984
	ds_read_b64 v[40:41], v239 offset:14992
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[66:71], v[102:107], a[48:51], v210, v212 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[42:43], v239 offset:15104
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[60:65], v[102:107], a[52:55], v210, v212 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[44:45], v239 offset:15112
	ds_read_b64 v[46:47], v239 offset:15120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[54:59], v[102:107], a[56:59], v210, v212 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s96 offen offset:3072 lds
	ds_read_b64 v[48:49], v239 offset:15232
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[6:11], v[102:107], a[60:63], v210, v212 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[50:51], v239 offset:15240
	ds_read_b64 v[52:53], v239 offset:15248
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[6:11], v[120:125], a[64:67], v210, v212 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[126:129], v226 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[54:59], v[120:125], a[68:71], v210, v212 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[130:131], v227 offset:22528
	s_mov_b32 m0, s92
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[60:65], v[120:125], a[72:75], v210, v212 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[8:11], s96 offen lds
	ds_read_b128 v[134:137], v226 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[66:71], v[120:125], a[76:79], v210, v212 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[138:139], v227 offset:23040
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[72:77], v[120:125], a[80:83], v211, v212 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[174:177], v226 offset:8192
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[78:83], v[120:125], a[84:87], v211, v212 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[178:179], v227 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[84:89], v[120:125], a[88:91], v211, v212 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[8:11], s96 offen offset:1024 lds
	ds_read_b128 v[180:183], v226 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[90:95], v[120:125], a[92:95], v211, v212 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[184:185], v227 offset:24064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[90:95], v[150:155], a[96:99], v211, v212 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[186:189], v226 offset:10240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[84:89], v[150:155], a[100:103], v211, v212 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[190:191], v227 offset:24576
	s_add_i32 s95, s64, s95
	s_mov_b32 m0, s93
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[78:83], v[150:155], a[104:107], v211, v212 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[12:15], s95 offen lds
	ds_read_b128 v[192:195], v226 offset:11264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[72:77], v[150:155], a[108:111], v211, v212 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[196:197], v227 offset:25088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[66:71], v[150:155], a[112:115], v210, v212 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[198:201], v226 offset:12288
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[60:65], v[150:155], a[116:119], v210, v212 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[202:203], v227 offset:25600
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[54:59], v[150:155], a[120:123], v210, v212 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[24:27], s95 offen offset:1024 lds
	ds_read_b128 v[204:207], v226 offset:13312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[6:11], v[150:155], a[124:127], v210, v212 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[208:209], v227 offset:26112
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[6:11], v[108:113], a[128:131], v210, v213 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[54:59], v[108:113], a[132:135], v210, v213 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[60:65], v[108:113], a[136:139], v210, v213 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[66:71], v[108:113], a[140:143], v210, v213 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s3, s56, s3
	s_add_i32 s3, s3, 0x100
	s_mov_b32 m0, s66
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[72:77], v[108:113], a[144:147], v211, v213 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[20:23], s3 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[78:83], v[108:113], a[148:151], v211, v213 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[84:89], v[108:113], a[152:155], v211, v213 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s70
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[90:95], v[108:113], a[156:159], v211, v213 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[28:31], s3 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[90:95], v[156:161], a[160:163], v211, v213 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[84:89], v[156:161], a[164:167], v211, v213 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[78:83], v[156:161], a[168:171], v211, v213 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s71
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[72:77], v[156:161], a[172:175], v211, v213 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[32:35], s3 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[66:71], v[156:161], a[176:179], v210, v213 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[60:65], v[156:161], a[180:183], v210, v213 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s72
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[54:59], v[156:161], a[184:187], v210, v213 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[36:39], s3 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[6:11], v[156:161], a[188:191], v210, v213 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[6:11], v[162:167], a[192:195], v210, v213 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[54:59], v[162:167], a[196:199], v210, v213 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s73
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[60:65], v[162:167], a[200:203], v210, v213 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[40:43], s3 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[66:71], v[162:167], a[204:207], v210, v213 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[72:77], v[162:167], a[208:211], v211, v213 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s74
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[78:83], v[162:167], a[212:215], v211, v213 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[44:47], s3 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[84:89], v[162:167], a[216:219], v211, v213 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[90:95], v[162:167], a[220:223], v211, v213 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[90:95], v[168:173], a[224:227], v211, v213 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s75
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[84:89], v[168:173], a[228:231], v211, v213 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[48:51], s3 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[78:83], v[168:173], a[232:235], v211, v213 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[72:77], v[168:173], a[236:239], v211, v213 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s76
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[66:71], v[168:173], a[240:243], v210, v213 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[52:55], s3 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[60:65], v[168:173], a[244:247], v210, v213 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[54:59], v[168:173], a[248:251], v210, v213 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[6:11], v[168:173], a[252:255], v210, v213 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s3, s94, 0x360
	s_lshl_b32 s3, s3, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[0:5], v[126:131], a[0:3], v214, v216 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(16) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[210:211], v224 offset:2048
	ds_read_b64 v[212:213], v225 offset:3072
	ds_read_b64 v[96:97], v246 offset:14432
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[12:17], v[126:131], a[4:7], v214, v216 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[98:99], v247 offset:14440
	ds_read_b64 v[100:101], v247 offset:14448
	s_add_i32 s94, s97, 0xa2000
	s_mov_b32 m0, s77
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:23], v[126:131], a[8:11], v214, v216 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s94 offen lds
	ds_read_b64 v[102:103], v246 offset:14560
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[24:29], v[126:131], a[12:15], v214, v216 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[104:105], v247 offset:14568
	ds_read_b64 v[106:107], v247 offset:14576
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[30:35], v[126:131], a[16:19], v215, v216 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[108:109], v246 offset:14688
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[36:41], v[126:131], a[20:23], v215, v216 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[110:111], v247 offset:14696
	ds_read_b64 v[112:113], v247 offset:14704
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[42:47], v[126:131], a[24:27], v215, v216 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s94 offen offset:1024 lds
	ds_read_b64 v[114:115], v246 offset:14816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[48:53], v[126:131], a[28:31], v215, v216 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[116:117], v247 offset:14824
	ds_read_b64 v[118:119], v247 offset:14832
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[48:53], v[134:139], a[32:35], v215, v216 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[6:7], v246 offset:14944
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[42:47], v[134:139], a[36:39], v215, v216 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[8:9], v247 offset:14952
	ds_read_b64 v[10:11], v247 offset:14960
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[36:41], v[134:139], a[40:43], v215, v216 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s94 offen offset:2048 lds
	ds_read_b64 v[120:121], v246 offset:15072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[30:35], v[134:139], a[44:47], v215, v216 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[122:123], v247 offset:15080
	ds_read_b64 v[124:125], v247 offset:15088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[24:29], v[134:139], a[48:51], v214, v216 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[126:127], v246 offset:15200
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[18:23], v[134:139], a[52:55], v214, v216 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[128:129], v247 offset:15208
	ds_read_b64 v[130:131], v247 offset:15216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[12:17], v[134:139], a[56:59], v214, v216 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s94 offen offset:3072 lds
	ds_read_b64 v[132:133], v246 offset:15328
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[0:5], v[134:139], a[60:63], v214, v216 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[134:135], v247 offset:15336
	ds_read_b64 v[136:137], v247 offset:15344
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[0:5], v[174:179], a[64:67], v214, v216 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[54:57], v226 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[12:17], v[174:179], a[68:71], v214, v216 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[58:59], v227 offset:47104
	s_mov_b32 m0, s78
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[18:23], v[174:179], a[72:75], v214, v216 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[8:11], s94 offen lds
	ds_read_b128 v[138:141], v226 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[24:29], v[174:179], a[76:79], v214, v216 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[142:143], v227 offset:47616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[30:35], v[174:179], a[80:83], v215, v216 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[144:147], v226 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[36:41], v[174:179], a[84:87], v215, v216 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[148:149], v227 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[42:47], v[174:179], a[88:91], v215, v216 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[8:11], s94 offen offset:1024 lds
	ds_read_b128 v[150:153], v226 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[48:53], v[174:179], a[92:95], v215, v216 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[154:155], v227 offset:48640
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[48:53], v[180:185], a[96:99], v215, v216 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[60:63], v226 offset:34816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[42:47], v[180:185], a[100:103], v215, v216 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[64:65], v227 offset:49152
	s_add_i32 s3, s64, s3
	s_mov_b32 m0, s64
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[36:41], v[180:185], a[104:107], v215, v216 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[12:15], s3 offen lds
	ds_read_b128 v[156:159], v226 offset:35840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[30:35], v[180:185], a[108:111], v215, v216 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[160:161], v227 offset:49664
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[24:29], v[180:185], a[112:115], v214, v216 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[162:165], v226 offset:36864
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[18:23], v[180:185], a[116:119], v214, v216 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[166:167], v227 offset:50176
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[12:17], v[180:185], a[120:123], v214, v216 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[24:27], s3 offen offset:1024 lds
	ds_read_b128 v[168:171], v226 offset:37888
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[0:5], v[180:185], a[124:127], v214, v216 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[172:173], v227 offset:50688
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[0:5], v[186:191], a[128:131], v214, v217 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[12:17], v[186:191], a[132:135], v214, v217 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[18:23], v[186:191], a[136:139], v214, v217 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[24:29], v[186:191], a[140:143], v214, v217 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[30:35], v[186:191], a[144:147], v215, v217 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[36:41], v[186:191], a[148:151], v215, v217 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[42:47], v[186:191], a[152:155], v215, v217 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[48:53], v[186:191], a[156:159], v215, v217 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[48:53], v[192:197], a[160:163], v215, v217 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[42:47], v[192:197], a[164:167], v215, v217 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[36:41], v[192:197], a[168:171], v215, v217 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[30:35], v[192:197], a[172:175], v215, v217 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[24:29], v[192:197], a[176:179], v214, v217 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[18:23], v[192:197], a[180:183], v214, v217 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[12:17], v[192:197], a[184:187], v214, v217 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[0:5], v[192:197], a[188:191], v214, v217 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[0:5], v[198:203], a[192:195], v214, v217 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[12:17], v[198:203], a[196:199], v214, v217 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[18:23], v[198:203], a[200:203], v214, v217 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[24:29], v[198:203], a[204:207], v214, v217 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[30:35], v[198:203], a[208:211], v215, v217 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[36:41], v[198:203], a[212:215], v215, v217 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[42:47], v[198:203], a[216:219], v215, v217 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[48:53], v[198:203], a[220:223], v215, v217 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[48:53], v[204:209], a[224:227], v215, v217 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[42:47], v[204:209], a[228:231], v215, v217 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[36:41], v[204:209], a[232:235], v215, v217 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[30:35], v[204:209], a[236:239], v215, v217 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[24:29], v[204:209], a[240:243], v214, v217 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[18:23], v[204:209], a[244:247], v214, v217 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[12:17], v[204:209], a[248:251], v214, v217 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[0:5], v[204:209], a[252:255], v214, v217 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s2, s2, 1
	s_cmp_lt_i32 s2, 21
	s_cbranch_scc1 .Lmain_loop
	s_add_i32 s2, s65, 0x3f80
	s_lshl_b32 s2, s2, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[96:101], v[54:59], a[0:3], v210, v212 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(8) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[174:175], v224 offset:4096
	ds_read_b64 v[176:177], v225 offset:5120
	v_add_u32_e32 v50, 0x10000, v230
	ds_read_b64 v[0:1], v50 offset:47424
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[102:107], v[54:59], a[4:7], v210, v212 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[2:3], v50 offset:47432
	v_add_u32_e32 v52, 0x10000, v231
	ds_read_b64 v[4:5], v52 offset:47440
	s_add_i32 s3, s69, 0xbe8000
	s_mov_b32 m0, s88
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[108:113], v[54:59], a[8:11], v210, v212 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s3 offen lds
	ds_read_b64 v[12:13], v50 offset:47552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[114:119], v[54:59], a[12:15], v210, v212 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[14:15], v50 offset:47560
	ds_read_b64 v[16:17], v52 offset:47568
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[6:11], v[54:59], a[16:19], v211, v212 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[18:19], v50 offset:47680
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[120:125], v[54:59], a[20:23], v211, v212 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[20:21], v50 offset:47688
	ds_read_b64 v[22:23], v52 offset:47696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[126:131], v[54:59], a[24:27], v211, v212 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s3 offen offset:1024 lds
	ds_read_b64 v[24:25], v50 offset:47808
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[132:137], v[54:59], a[28:31], v211, v212 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[26:27], v50 offset:47816
	ds_read_b64 v[28:29], v52 offset:47824
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[132:137], v[138:143], a[32:35], v211, v212 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[30:31], v50 offset:47936
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[126:131], v[138:143], a[36:39], v211, v212 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[32:33], v50 offset:47944
	ds_read_b64 v[34:35], v52 offset:47952
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[120:125], v[138:143], a[40:43], v211, v212 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s3 offen offset:2048 lds
	ds_read_b64 v[36:37], v50 offset:48064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[6:11], v[138:143], a[44:47], v211, v212 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[38:39], v50 offset:48072
	ds_read_b64 v[40:41], v52 offset:48080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[114:119], v[138:143], a[48:51], v210, v212 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[42:43], v50 offset:48192
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[108:113], v[138:143], a[52:55], v210, v212 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[44:45], v50 offset:48200
	ds_read_b64 v[46:47], v52 offset:48208
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[102:107], v[138:143], a[56:59], v210, v212 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s3 offen offset:3072 lds
	ds_read_b64 v[48:49], v50 offset:48320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[96:101], v[138:143], a[60:63], v210, v212 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[50:51], v50 offset:48328
	ds_read_b64 v[52:53], v52 offset:48336
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[96:101], v[144:149], a[64:67], v210, v212 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[54:57], v226 offset:55296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[102:107], v[144:149], a[68:71], v210, v212 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_add_u32_e32 v182, 0x10000, v227
	ds_read_b64 v[58:59], v182 offset:6144
	s_add_i32 s3, s69, 0xbe9000
	s_mov_b32 m0, s89
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[108:113], v[144:149], a[72:75], v210, v212 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s3 offen lds
	ds_read_b128 v[66:69], v226 offset:56320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[114:119], v[144:149], a[76:79], v210, v212 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[70:71], v182 offset:6656
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[6:11], v[144:149], a[80:83], v211, v212 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[72:75], v226 offset:57344
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[120:125], v[144:149], a[84:87], v211, v212 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[76:77], v182 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[126:131], v[144:149], a[88:91], v211, v212 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s3 offen offset:1024 lds
	ds_read_b128 v[78:81], v226 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[132:137], v[144:149], a[92:95], v211, v212 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[82:83], v182 offset:7680
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[132:137], v[150:155], a[96:99], v211, v212 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[84:87], v226 offset:59392
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[126:131], v[150:155], a[100:103], v211, v212 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[88:89], v182 offset:8192
	s_add_i32 s2, s64, s2
	s_mov_b32 m0, s90
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[120:125], v[150:155], a[104:107], v211, v212 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[12:15], s2 offen lds
	ds_read_b128 v[90:93], v226 offset:60416
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[6:11], v[150:155], a[108:111], v211, v212 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[94:95], v182 offset:8704
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[114:119], v[150:155], a[112:115], v210, v212 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[138:141], v226 offset:61440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[108:113], v[150:155], a[116:119], v210, v212 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[142:143], v182 offset:9216
	s_add_i32 s2, s79, 0x7efc00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[102:107], v[150:155], a[120:123], v210, v212 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[16:19], s2 offen offset:1024 lds
	ds_read_b128 v[144:147], v226 offset:62464
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[96:101], v[150:155], a[124:127], v210, v212 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[148:149], v182 offset:9728
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[96:101], v[60:65], a[128:131], v210, v213 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[102:107], v[60:65], a[132:135], v210, v213 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[108:113], v[60:65], a[136:139], v210, v213 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[114:119], v[60:65], a[140:143], v210, v213 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s2, s56, 0xbe80
	s_mov_b32 m0, s80
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[6:11], v[60:65], a[144:147], v211, v213 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[20:23], s2 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[120:125], v[60:65], a[148:151], v211, v213 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[126:131], v[60:65], a[152:155], v211, v213 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s2, s57, 0xbe80
	s_mov_b32 m0, s81
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[132:137], v[60:65], a[156:159], v211, v213 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[20:23], s2 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[132:137], v[156:161], a[160:163], v211, v213 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[126:131], v[156:161], a[164:167], v211, v213 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[120:125], v[156:161], a[168:171], v211, v213 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_i32 s2, s58, 0xbe80
	s_mov_b32 m0, s82
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[6:11], v[156:161], a[172:175], v211, v213 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[20:23], s2 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[114:119], v[156:161], a[176:179], v210, v213 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[108:113], v[156:161], a[180:183], v210, v213 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s2, s59, 0xbe80
	s_mov_b32 m0, s83
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[102:107], v[156:161], a[184:187], v210, v213 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[20:23], s2 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[96:101], v[156:161], a[188:191], v210, v213 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[96:101], v[162:167], a[192:195], v210, v213 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[102:107], v[162:167], a[196:199], v210, v213 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s2, s60, 0xbe80
	s_mov_b32 m0, s84
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[108:113], v[162:167], a[200:203], v210, v213 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[20:23], s2 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[114:119], v[162:167], a[204:207], v210, v213 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[6:11], v[162:167], a[208:211], v211, v213 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s2, s63, 0xbe80
	s_mov_b32 m0, s85
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[120:125], v[162:167], a[212:215], v211, v213 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[20:23], s2 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[126:131], v[162:167], a[216:219], v211, v213 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[132:137], v[162:167], a[220:223], v211, v213 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[132:137], v[168:173], a[224:227], v211, v213 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s2, s67, 0xbe80
	s_mov_b32 m0, s86
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[126:131], v[168:173], a[228:231], v211, v213 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[20:23], s2 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[120:125], v[168:173], a[232:235], v211, v213 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[6:11], v[168:173], a[236:239], v211, v213 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s2, s68, 0xbe80
	s_mov_b32 m0, s87
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[114:119], v[168:173], a[240:243], v210, v213 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[20:23], s2 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[108:113], v[168:173], a[244:247], v210, v213 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[102:107], v[168:173], a[248:251], v210, v213 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[96:101], v[168:173], a[252:255], v210, v213 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s2, s65, 0x3fa0
	s_lshl_b32 s2, s2, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[0:5], v[54:59], a[0:3], v174, v176 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(16) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[178:179], v224
	ds_read_b64 v[180:181], v225 offset:1024
	v_add_u32_e32 v183, 0x10000, v223
	ds_read_b64 v[6:7], v183 offset:14368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[12:17], v[54:59], a[4:7], v174, v176 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[8:9], v183 offset:14376
	ds_read_b64 v[10:11], v183 offset:14384
	s_add_i32 s3, s69, 0xbee000
	s_mov_b32 m0, s91
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:23], v[54:59], a[8:11], v174, v176 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s3 offen lds
	ds_read_b64 v[60:61], v183 offset:14496
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[24:29], v[54:59], a[12:15], v174, v176 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[62:63], v183 offset:14504
	ds_read_b64 v[64:65], v183 offset:14512
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[30:35], v[54:59], a[16:19], v175, v176 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[96:97], v183 offset:14624
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[36:41], v[54:59], a[20:23], v175, v176 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[98:99], v183 offset:14632
	ds_read_b64 v[100:101], v183 offset:14640
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[42:47], v[54:59], a[24:27], v175, v176 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s3 offen offset:1024 lds
	ds_read_b64 v[102:103], v183 offset:14752
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[48:53], v[54:59], a[28:31], v175, v176 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[104:105], v183 offset:14760
	ds_read_b64 v[106:107], v183 offset:14768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[48:53], v[66:71], a[32:35], v175, v176 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[54:55], v183 offset:14880
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[42:47], v[66:71], a[36:39], v175, v176 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[56:57], v183 offset:14888
	ds_read_b64 v[58:59], v183 offset:14896
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[36:41], v[66:71], a[40:43], v175, v176 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s3 offen offset:2048 lds
	ds_read_b64 v[108:109], v183 offset:15008
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[30:35], v[66:71], a[44:47], v175, v176 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[110:111], v183 offset:15016
	ds_read_b64 v[112:113], v183 offset:15024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[24:29], v[66:71], a[48:51], v174, v176 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[114:115], v183 offset:15136
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[18:23], v[66:71], a[52:55], v174, v176 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[116:117], v183 offset:15144
	ds_read_b64 v[118:119], v183 offset:15152
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[12:17], v[66:71], a[56:59], v174, v176 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s3 offen offset:3072 lds
	ds_read_b64 v[120:121], v183 offset:15264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[0:5], v[66:71], a[60:63], v174, v176 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[122:123], v183 offset:15272
	ds_read_b64 v[124:125], v183 offset:15280
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[0:5], v[72:77], a[64:67], v174, v176 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[66:69], v226 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[12:17], v[72:77], a[68:71], v174, v176 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[70:71], v227 offset:22528
	s_add_i32 s3, s69, 0xbef000
	s_mov_b32 m0, s92
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[18:23], v[72:77], a[72:75], v174, v176 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s3 offen lds
	ds_read_b128 v[126:129], v226 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[24:29], v[72:77], a[76:79], v174, v176 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[130:131], v227 offset:23040
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[30:35], v[72:77], a[80:83], v175, v176 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[132:135], v226 offset:8192
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[36:41], v[72:77], a[84:87], v175, v176 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[136:137], v227 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[42:47], v[72:77], a[88:91], v175, v176 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s3 offen offset:1024 lds
	ds_read_b128 v[150:153], v226 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[48:53], v[72:77], a[92:95], v175, v176 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[154:155], v227 offset:24064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[48:53], v[78:83], a[96:99], v175, v176 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[72:75], v226 offset:10240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[42:47], v[78:83], a[100:103], v175, v176 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[76:77], v227 offset:24576
	s_add_i32 s2, s64, s2
	s_mov_b32 m0, s93
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[36:41], v[78:83], a[104:107], v175, v176 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[12:15], s2 offen lds
	ds_read_b128 v[156:159], v226 offset:11264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[30:35], v[78:83], a[108:111], v175, v176 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[160:161], v227 offset:25088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[24:29], v[78:83], a[112:115], v174, v176 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[162:165], v226 offset:12288
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[18:23], v[78:83], a[116:119], v174, v176 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[166:167], v227 offset:25600
	s_add_i32 s2, s79, 0x7f3c00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[12:17], v[78:83], a[120:123], v174, v176 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[16:19], s2 offen offset:1024 lds
	ds_read_b128 v[168:171], v226 offset:13312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[0:5], v[78:83], a[124:127], v174, v176 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[172:173], v227 offset:26112
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[0:5], v[84:89], a[128:131], v174, v177 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[12:17], v[84:89], a[132:135], v174, v177 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[18:23], v[84:89], a[136:139], v174, v177 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[24:29], v[84:89], a[140:143], v174, v177 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s2, s56, 0xbf00
	s_mov_b32 m0, s66
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[30:35], v[84:89], a[144:147], v175, v177 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[20:23], s2 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[36:41], v[84:89], a[148:151], v175, v177 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[42:47], v[84:89], a[152:155], v175, v177 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s2, s57, 0xbf00
	s_mov_b32 m0, s70
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[48:53], v[84:89], a[156:159], v175, v177 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[20:23], s2 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[48:53], v[90:95], a[160:163], v175, v177 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[42:47], v[90:95], a[164:167], v175, v177 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[36:41], v[90:95], a[168:171], v175, v177 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_i32 s2, s58, 0xbf00
	s_mov_b32 m0, s71
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[30:35], v[90:95], a[172:175], v175, v177 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[20:23], s2 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[24:29], v[90:95], a[176:179], v174, v177 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[18:23], v[90:95], a[180:183], v174, v177 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s2, s59, 0xbf00
	s_mov_b32 m0, s72
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[12:17], v[90:95], a[184:187], v174, v177 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[20:23], s2 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[0:5], v[90:95], a[188:191], v174, v177 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[0:5], v[138:143], a[192:195], v174, v177 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[12:17], v[138:143], a[196:199], v174, v177 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s2, s60, 0xbf00
	s_mov_b32 m0, s73
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[18:23], v[138:143], a[200:203], v174, v177 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[20:23], s2 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[24:29], v[138:143], a[204:207], v174, v177 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[30:35], v[138:143], a[208:211], v175, v177 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s2, s63, 0xbf00
	s_mov_b32 m0, s74
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[36:41], v[138:143], a[212:215], v175, v177 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[20:23], s2 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[42:47], v[138:143], a[216:219], v175, v177 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[48:53], v[138:143], a[220:223], v175, v177 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[48:53], v[144:149], a[224:227], v175, v177 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s2, s67, 0xbf00
	s_mov_b32 m0, s75
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[42:47], v[144:149], a[228:231], v175, v177 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[20:23], s2 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[36:41], v[144:149], a[232:235], v175, v177 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[30:35], v[144:149], a[236:239], v175, v177 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s2, s68, 0xbf00
	s_mov_b32 m0, s76
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[24:29], v[144:149], a[240:243], v174, v177 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[20:23], s2 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[18:23], v[144:149], a[244:247], v174, v177 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[12:17], v[144:149], a[248:251], v174, v177 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[0:5], v[144:149], a[252:255], v174, v177 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s2, s65, 0x3fc0
	s_lshl_b32 s2, s2, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[6:11], v[66:71], a[0:3], v178, v180 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(16) lgkmcnt(0)
	s_barrier
	v_add_u32_e32 v0, 0x10000, v228
	ds_read_b128 v[18:21], v0 offset:47360
	ds_read_b128 v[22:25], v0 offset:47376
	ds_read_b64 v[174:175], v224 offset:2048
	ds_read_b64 v[176:177], v225 offset:3072
	ds_read_b64 v[0:1], v183 offset:47360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[60:65], v[66:71], a[4:7], v178, v180 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[2:3], v183 offset:47368
	ds_read_b64 v[4:5], v183 offset:47376
	s_add_i32 s3, s69, 0xbf4000
	s_mov_b32 m0, s77
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[96:101], v[66:71], a[8:11], v178, v180 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s3 offen lds
	ds_read_b64 v[12:13], v183 offset:47488
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[102:107], v[66:71], a[12:15], v178, v180 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[14:15], v183 offset:47496
	ds_read_b64 v[16:17], v183 offset:47504
	v_add_u32_e32 v26, 0x10000, v229
	s_waitcnt lgkmcnt(9)
	ds_write_b64 v26, v[18:19] offset:14336
	ds_write_b64 v26, v[20:21] offset:14344
	s_waitcnt lgkmcnt(10)
	ds_write_b64 v26, v[22:23] offset:14352
	ds_write_b64 v26, v[24:25] offset:14360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[54:59], v[66:71], a[16:19], v179, v180 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[18:19], v183 offset:47616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[108:113], v[66:71], a[20:23], v179, v180 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[20:21], v183 offset:47624
	ds_read_b64 v[22:23], v183 offset:47632
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[114:119], v[66:71], a[24:27], v179, v180 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s3 offen offset:1024 lds
	ds_read_b64 v[24:25], v183 offset:47744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[120:125], v[66:71], a[28:31], v179, v180 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[26:27], v183 offset:47752
	ds_read_b64 v[28:29], v183 offset:47760
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[120:125], v[126:131], a[32:35], v179, v180 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[30:31], v183 offset:47872
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[114:119], v[126:131], a[36:39], v179, v180 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[32:33], v183 offset:47880
	ds_read_b64 v[34:35], v183 offset:47888
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[108:113], v[126:131], a[40:43], v179, v180 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s3 offen offset:2048 lds
	ds_read_b64 v[36:37], v183 offset:48000
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[54:59], v[126:131], a[44:47], v179, v180 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[38:39], v183 offset:48008
	ds_read_b64 v[40:41], v183 offset:48016
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[102:107], v[126:131], a[48:51], v178, v180 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[42:43], v183 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[96:101], v[126:131], a[52:55], v178, v180 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[44:45], v183 offset:48136
	ds_read_b64 v[46:47], v183 offset:48144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[60:65], v[126:131], a[56:59], v178, v180 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s3 offen offset:3072 lds
	ds_read_b64 v[48:49], v183 offset:48256
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[6:11], v[126:131], a[60:63], v178, v180 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[50:51], v183 offset:48264
	ds_read_b64 v[52:53], v183 offset:48272
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[6:11], v[132:137], a[64:67], v178, v180 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[66:69], v226 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[60:65], v[132:137], a[68:71], v178, v180 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[70:71], v227 offset:47104
	s_add_i32 s3, s69, 0xbf5000
	s_mov_b32 m0, s78
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[96:101], v[132:137], a[72:75], v178, v180 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s3 offen lds
	ds_read_b128 v[78:81], v226 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[102:107], v[132:137], a[76:79], v178, v180 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[82:83], v227 offset:47616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[54:59], v[132:137], a[80:83], v179, v180 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[84:87], v226 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[108:113], v[132:137], a[84:87], v179, v180 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[88:89], v227 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[114:119], v[132:137], a[88:91], v179, v180 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s3 offen offset:1024 lds
	ds_read_b128 v[90:93], v226 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[120:125], v[132:137], a[92:95], v179, v180 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[94:95], v227 offset:48640
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[120:125], v[150:155], a[96:99], v179, v180 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[126:129], v226 offset:34816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[114:119], v[150:155], a[100:103], v179, v180 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[130:131], v227 offset:49152
	s_add_i32 s2, s64, s2
	s_mov_b32 m0, s64
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[108:113], v[150:155], a[104:107], v179, v180 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[12:15], s2 offen lds
	ds_read_b128 v[132:135], v226 offset:35840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[54:59], v[150:155], a[108:111], v179, v180 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[136:137], v227 offset:49664
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[102:107], v[150:155], a[112:115], v178, v180 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[138:141], v226 offset:36864
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[96:101], v[150:155], a[116:119], v178, v180 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[142:143], v227 offset:50176
	s_add_i32 s2, s79, 0x7f7c00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[60:65], v[150:155], a[120:123], v178, v180 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[16:19], s2 offen offset:1024 lds
	ds_read_b128 v[144:147], v226 offset:37888
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[6:11], v[150:155], a[124:127], v178, v180 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[148:149], v227 offset:50688
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[6:11], v[72:77], a[128:131], v178, v181 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[60:65], v[72:77], a[132:135], v178, v181 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[96:101], v[72:77], a[136:139], v178, v181 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[102:107], v[72:77], a[140:143], v178, v181 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s2, s56, 0xbf80
	s_mov_b32 m0, s80
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[54:59], v[72:77], a[144:147], v179, v181 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[20:23], s2 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[108:113], v[72:77], a[148:151], v179, v181 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[114:119], v[72:77], a[152:155], v179, v181 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s2, s57, 0xbf80
	s_mov_b32 m0, s81
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[120:125], v[72:77], a[156:159], v179, v181 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[20:23], s2 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[120:125], v[156:161], a[160:163], v179, v181 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[114:119], v[156:161], a[164:167], v179, v181 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[108:113], v[156:161], a[168:171], v179, v181 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_i32 s2, s58, 0xbf80
	s_mov_b32 m0, s82
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[54:59], v[156:161], a[172:175], v179, v181 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[20:23], s2 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[102:107], v[156:161], a[176:179], v178, v181 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[96:101], v[156:161], a[180:183], v178, v181 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s2, s59, 0xbf80
	s_mov_b32 m0, s83
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[60:65], v[156:161], a[184:187], v178, v181 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[20:23], s2 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[6:11], v[156:161], a[188:191], v178, v181 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[6:11], v[162:167], a[192:195], v178, v181 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[60:65], v[162:167], a[196:199], v178, v181 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s2, s60, 0xbf80
	s_mov_b32 m0, s84
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[96:101], v[162:167], a[200:203], v178, v181 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[20:23], s2 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[102:107], v[162:167], a[204:207], v178, v181 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[54:59], v[162:167], a[208:211], v179, v181 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s2, s63, 0xbf80
	s_mov_b32 m0, s85
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[108:113], v[162:167], a[212:215], v179, v181 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[20:23], s2 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[114:119], v[162:167], a[216:219], v179, v181 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[120:125], v[162:167], a[220:223], v179, v181 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[120:125], v[168:173], a[224:227], v179, v181 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s2, s67, 0xbf80
	s_mov_b32 m0, s86
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[114:119], v[168:173], a[228:231], v179, v181 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[20:23], s2 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[108:113], v[168:173], a[232:235], v179, v181 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[54:59], v[168:173], a[236:239], v179, v181 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s2, s68, 0xbf80
	s_mov_b32 m0, s87
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[102:107], v[168:173], a[240:243], v178, v181 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v220, s[20:23], s2 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[96:101], v[168:173], a[244:247], v178, v181 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[60:65], v[168:173], a[248:251], v178, v181 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[6:11], v[168:173], a[252:255], v178, v181 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s2, s65, 0x3fe0
	s_lshl_b32 s2, s2, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[0:5], v[66:71], a[0:3], v174, v176 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(16) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[178:179], v224 offset:4096
	ds_read_b64 v[180:181], v225 offset:5120
	v_add_u32_e32 v108, 0x10000, v232
	ds_read_b64 v[6:7], v108 offset:47456
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[12:17], v[66:71], a[4:7], v174, v176 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_add_u32_e32 v112, 0x10000, v233
	ds_read_b64 v[8:9], v112 offset:47464
	ds_read_b64 v[10:11], v112 offset:47472
	s_add_i32 s3, s69, 0xbfa000
	s_mov_b32 m0, s88
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:23], v[66:71], a[8:11], v174, v176 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s3 offen lds
	ds_read_b64 v[54:55], v108 offset:47584
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[24:29], v[66:71], a[12:15], v174, v176 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[56:57], v112 offset:47592
	ds_read_b64 v[58:59], v112 offset:47600
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[30:35], v[66:71], a[16:19], v175, v176 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[60:61], v108 offset:47712
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[36:41], v[66:71], a[20:23], v175, v176 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[62:63], v112 offset:47720
	ds_read_b64 v[64:65], v112 offset:47728
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[42:47], v[66:71], a[24:27], v175, v176 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s3 offen offset:1024 lds
	ds_read_b64 v[72:73], v108 offset:47840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[48:53], v[66:71], a[28:31], v175, v176 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[74:75], v112 offset:47848
	ds_read_b64 v[76:77], v112 offset:47856
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[48:53], v[78:83], a[32:35], v175, v176 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[66:67], v108 offset:47968
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[42:47], v[78:83], a[36:39], v175, v176 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[68:69], v112 offset:47976
	ds_read_b64 v[70:71], v112 offset:47984
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[36:41], v[78:83], a[40:43], v175, v176 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s3 offen offset:2048 lds
	ds_read_b64 v[96:97], v108 offset:48096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[30:35], v[78:83], a[44:47], v175, v176 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[98:99], v112 offset:48104
	ds_read_b64 v[100:101], v112 offset:48112
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[24:29], v[78:83], a[48:51], v174, v176 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[102:103], v108 offset:48224
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[18:23], v[78:83], a[52:55], v174, v176 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[104:105], v112 offset:48232
	ds_read_b64 v[106:107], v112 offset:48240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[12:17], v[78:83], a[56:59], v174, v176 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s3 offen offset:3072 lds
	ds_read_b64 v[108:109], v108 offset:48352
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[0:5], v[78:83], a[60:63], v174, v176 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[110:111], v112 offset:48360
	ds_read_b64 v[112:113], v112 offset:48368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[0:5], v[84:89], a[64:67], v174, v176 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[78:81], v226 offset:55296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[12:17], v[84:89], a[68:71], v174, v176 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[82:83], v182 offset:6144
	s_add_i32 s3, s69, 0xbfb000
	s_mov_b32 m0, s89
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[18:23], v[84:89], a[72:75], v174, v176 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s3 offen lds
	ds_read_b128 v[114:117], v226 offset:56320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[24:29], v[84:89], a[76:79], v174, v176 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[118:119], v182 offset:6656
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[30:35], v[84:89], a[80:83], v175, v176 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[120:123], v226 offset:57344
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[36:41], v[84:89], a[84:87], v175, v176 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[124:125], v182 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[42:47], v[84:89], a[88:91], v175, v176 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v221, s[4:7], s3 offen offset:1024 lds
	ds_read_b128 v[150:153], v226 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[48:53], v[84:89], a[92:95], v175, v176 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[154:155], v182 offset:7680
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[48:53], v[90:95], a[96:99], v175, v176 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[84:87], v226 offset:59392
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[42:47], v[90:95], a[100:103], v175, v176 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[88:89], v182 offset:8192
	s_add_i32 s2, s64, s2
	s_mov_b32 m0, s90
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[36:41], v[90:95], a[104:107], v175, v176 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[12:15], s2 offen lds
	ds_read_b128 v[156:159], v226 offset:60416
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[30:35], v[90:95], a[108:111], v175, v176 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[160:161], v182 offset:8704
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[24:29], v[90:95], a[112:115], v174, v176 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[162:165], v226 offset:61440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[18:23], v[90:95], a[116:119], v174, v176 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[166:167], v182 offset:9216
	s_add_i32 s2, s79, 0x7fbc00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[12:17], v[90:95], a[120:123], v174, v176 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dword v222, s[16:19], s2 offen offset:1024 lds
	ds_read_b128 v[168:171], v226 offset:62464
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[0:5], v[90:95], a[124:127], v174, v176 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[172:173], v182 offset:9728
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[0:5], v[126:131], a[128:131], v174, v177 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[12:17], v[126:131], a[132:135], v174, v177 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[18:23], v[126:131], a[136:139], v174, v177 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[24:29], v[126:131], a[140:143], v174, v177 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[30:35], v[126:131], a[144:147], v175, v177 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[36:41], v[126:131], a[148:151], v175, v177 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[42:47], v[126:131], a[152:155], v175, v177 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[48:53], v[126:131], a[156:159], v175, v177 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[48:53], v[132:137], a[160:163], v175, v177 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[42:47], v[132:137], a[164:167], v175, v177 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[36:41], v[132:137], a[168:171], v175, v177 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[30:35], v[132:137], a[172:175], v175, v177 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[24:29], v[132:137], a[176:179], v174, v177 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[18:23], v[132:137], a[180:183], v174, v177 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[12:17], v[132:137], a[184:187], v174, v177 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[0:5], v[132:137], a[188:191], v174, v177 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[0:5], v[138:143], a[192:195], v174, v177 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[12:17], v[138:143], a[196:199], v174, v177 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[18:23], v[138:143], a[200:203], v174, v177 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[24:29], v[138:143], a[204:207], v174, v177 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[30:35], v[138:143], a[208:211], v175, v177 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[36:41], v[138:143], a[212:215], v175, v177 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[42:47], v[138:143], a[216:219], v175, v177 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[48:53], v[138:143], a[220:223], v175, v177 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[48:53], v[144:149], a[224:227], v175, v177 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[42:47], v[144:149], a[228:231], v175, v177 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[36:41], v[144:149], a[232:235], v175, v177 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[30:35], v[144:149], a[236:239], v175, v177 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[24:29], v[144:149], a[240:243], v174, v177 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[18:23], v[144:149], a[244:247], v174, v177 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[12:17], v[144:149], a[248:251], v174, v177 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[0:5], v[144:149], a[252:255], v174, v177 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[6:11], v[78:83], a[0:3], v178, v180 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(8) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[174:175], v224
	ds_read_b64 v[176:177], v225 offset:1024
	v_add_u32_e32 v50, 0x10000, v234
	ds_read_b64 v[0:1], v50 offset:14400
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[54:59], v[78:83], a[4:7], v178, v180 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[2:3], v50 offset:14408
	v_add_u32_e32 v52, 0x10000, v235
	ds_read_b64 v[4:5], v52 offset:14416
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[60:65], v[78:83], a[8:11], v178, v180 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[12:13], v50 offset:14528
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[72:77], v[78:83], a[12:15], v178, v180 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[14:15], v50 offset:14536
	ds_read_b64 v[16:17], v52 offset:14544
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[66:71], v[78:83], a[16:19], v179, v180 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[18:19], v50 offset:14656
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[96:101], v[78:83], a[20:23], v179, v180 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[20:21], v50 offset:14664
	ds_read_b64 v[22:23], v52 offset:14672
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[102:107], v[78:83], a[24:27], v179, v180 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[24:25], v50 offset:14784
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[108:113], v[78:83], a[28:31], v179, v180 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[26:27], v50 offset:14792
	ds_read_b64 v[28:29], v52 offset:14800
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[108:113], v[114:119], a[32:35], v179, v180 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[30:31], v50 offset:14912
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[102:107], v[114:119], a[36:39], v179, v180 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[32:33], v50 offset:14920
	ds_read_b64 v[34:35], v52 offset:14928
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[96:101], v[114:119], a[40:43], v179, v180 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[36:37], v50 offset:15040
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[66:71], v[114:119], a[44:47], v179, v180 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[38:39], v50 offset:15048
	ds_read_b64 v[40:41], v52 offset:15056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[72:77], v[114:119], a[48:51], v178, v180 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[42:43], v50 offset:15168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[60:65], v[114:119], a[52:55], v178, v180 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[44:45], v50 offset:15176
	ds_read_b64 v[46:47], v52 offset:15184
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[54:59], v[114:119], a[56:59], v178, v180 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[48:49], v50 offset:15296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[6:11], v[114:119], a[60:63], v178, v180 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[50:51], v50 offset:15304
	ds_read_b64 v[52:53], v52 offset:15312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[6:11], v[120:125], a[64:67], v178, v180 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[78:81], v226 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[54:59], v[120:125], a[68:71], v178, v180 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[82:83], v227 offset:22528
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[60:65], v[120:125], a[72:75], v178, v180 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[90:93], v226 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[72:77], v[120:125], a[76:79], v178, v180 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[94:95], v227 offset:23040
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[66:71], v[120:125], a[80:83], v179, v180 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[114:117], v226 offset:8192
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[96:101], v[120:125], a[84:87], v179, v180 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[118:119], v227 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[102:107], v[120:125], a[88:91], v179, v180 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[126:129], v226 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[108:113], v[120:125], a[92:95], v179, v180 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[130:131], v227 offset:24064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[108:113], v[150:155], a[96:99], v179, v180 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[120:123], v226 offset:10240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[102:107], v[150:155], a[100:103], v179, v180 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[124:125], v227 offset:24576
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[96:101], v[150:155], a[104:107], v179, v180 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[132:135], v226 offset:11264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[66:71], v[150:155], a[108:111], v179, v180 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[136:137], v227 offset:25088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[72:77], v[150:155], a[112:115], v178, v180 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[138:141], v226 offset:12288
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[60:65], v[150:155], a[116:119], v178, v180 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[142:143], v227 offset:25600
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[54:59], v[150:155], a[120:123], v178, v180 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[144:147], v226 offset:13312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[6:11], v[150:155], a[124:127], v178, v180 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[148:149], v227 offset:26112
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[6:11], v[84:89], a[128:131], v178, v181 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[54:59], v[84:89], a[132:135], v178, v181 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[60:65], v[84:89], a[136:139], v178, v181 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[72:77], v[84:89], a[140:143], v178, v181 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[66:71], v[84:89], a[144:147], v179, v181 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[96:101], v[84:89], a[148:151], v179, v181 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[102:107], v[84:89], a[152:155], v179, v181 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[108:113], v[84:89], a[156:159], v179, v181 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[108:113], v[156:161], a[160:163], v179, v181 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[102:107], v[156:161], a[164:167], v179, v181 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[96:101], v[156:161], a[168:171], v179, v181 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[66:71], v[156:161], a[172:175], v179, v181 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[72:77], v[156:161], a[176:179], v178, v181 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[60:65], v[156:161], a[180:183], v178, v181 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[54:59], v[156:161], a[184:187], v178, v181 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[6:11], v[156:161], a[188:191], v178, v181 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[6:11], v[162:167], a[192:195], v178, v181 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[54:59], v[162:167], a[196:199], v178, v181 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[60:65], v[162:167], a[200:203], v178, v181 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[72:77], v[162:167], a[204:207], v178, v181 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[66:71], v[162:167], a[208:211], v179, v181 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[96:101], v[162:167], a[212:215], v179, v181 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[102:107], v[162:167], a[216:219], v179, v181 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[108:113], v[162:167], a[220:223], v179, v181 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[108:113], v[168:173], a[224:227], v179, v181 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[102:107], v[168:173], a[228:231], v179, v181 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[96:101], v[168:173], a[232:235], v179, v181 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[66:71], v[168:173], a[236:239], v179, v181 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[72:77], v[168:173], a[240:243], v178, v181 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[60:65], v[168:173], a[244:247], v178, v181 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[54:59], v[168:173], a[248:251], v178, v181 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[6:11], v[168:173], a[252:255], v178, v181 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[0:5], v[78:83], a[0:3], v174, v176 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[178:179], v224 offset:2048
	ds_read_b64 v[180:181], v225 offset:3072
	ds_read_b64 v[6:7], v183 offset:47392
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[12:17], v[78:83], a[4:7], v174, v176 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[8:9], v183 offset:47400
	ds_read_b64 v[10:11], v183 offset:47408
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:23], v[78:83], a[8:11], v174, v176 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[54:55], v183 offset:47520
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[24:29], v[78:83], a[12:15], v174, v176 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[56:57], v183 offset:47528
	ds_read_b64 v[58:59], v183 offset:47536
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[30:35], v[78:83], a[16:19], v175, v176 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[60:61], v183 offset:47648
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[36:41], v[78:83], a[20:23], v175, v176 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[62:63], v183 offset:47656
	ds_read_b64 v[64:65], v183 offset:47664
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[42:47], v[78:83], a[24:27], v175, v176 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[66:67], v183 offset:47776
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[48:53], v[78:83], a[28:31], v175, v176 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[68:69], v183 offset:47784
	ds_read_b64 v[70:71], v183 offset:47792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[48:53], v[90:95], a[32:35], v175, v176 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[72:73], v183 offset:47904
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[42:47], v[90:95], a[36:39], v175, v176 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[74:75], v183 offset:47912
	ds_read_b64 v[76:77], v183 offset:47920
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[36:41], v[90:95], a[40:43], v175, v176 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v183 offset:48032
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[30:35], v[90:95], a[44:47], v175, v176 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[80:81], v183 offset:48040
	ds_read_b64 v[82:83], v183 offset:48048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[24:29], v[90:95], a[48:51], v174, v176 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[84:85], v183 offset:48160
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[18:23], v[90:95], a[52:55], v174, v176 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[86:87], v183 offset:48168
	ds_read_b64 v[88:89], v183 offset:48176
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[12:17], v[90:95], a[56:59], v174, v176 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[96:97], v183 offset:48288
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[0:5], v[90:95], a[60:63], v174, v176 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[98:99], v183 offset:48296
	ds_read_b64 v[100:101], v183 offset:48304
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[0:5], v[114:119], a[64:67], v174, v176 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[90:93], v226 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[12:17], v[114:119], a[68:71], v174, v176 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[94:95], v227 offset:47104
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[18:23], v[114:119], a[72:75], v174, v176 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[102:105], v226 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[24:29], v[114:119], a[76:79], v174, v176 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[106:107], v227 offset:47616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[30:35], v[114:119], a[80:83], v175, v176 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[108:111], v226 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[36:41], v[114:119], a[84:87], v175, v176 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[112:113], v227 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[42:47], v[114:119], a[88:91], v175, v176 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[150:153], v226 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[48:53], v[114:119], a[92:95], v175, v176 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[154:155], v227 offset:48640
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[48:53], v[126:131], a[96:99], v175, v176 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[114:117], v226 offset:34816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[42:47], v[126:131], a[100:103], v175, v176 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[118:119], v227 offset:49152
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[36:41], v[126:131], a[104:107], v175, v176 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[156:159], v226 offset:35840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[30:35], v[126:131], a[108:111], v175, v176 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[160:161], v227 offset:49664
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[24:29], v[126:131], a[112:115], v174, v176 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[162:165], v226 offset:36864
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[18:23], v[126:131], a[116:119], v174, v176 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[166:167], v227 offset:50176
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[12:17], v[126:131], a[120:123], v174, v176 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[168:171], v226 offset:37888
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[0:5], v[126:131], a[124:127], v174, v176 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[172:173], v227 offset:50688
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[0:5], v[120:125], a[128:131], v174, v177 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[12:17], v[120:125], a[132:135], v174, v177 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[18:23], v[120:125], a[136:139], v174, v177 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[24:29], v[120:125], a[140:143], v174, v177 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[30:35], v[120:125], a[144:147], v175, v177 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[36:41], v[120:125], a[148:151], v175, v177 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[42:47], v[120:125], a[152:155], v175, v177 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[48:53], v[120:125], a[156:159], v175, v177 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[48:53], v[132:137], a[160:163], v175, v177 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[42:47], v[132:137], a[164:167], v175, v177 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[36:41], v[132:137], a[168:171], v175, v177 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[30:35], v[132:137], a[172:175], v175, v177 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[24:29], v[132:137], a[176:179], v174, v177 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[18:23], v[132:137], a[180:183], v174, v177 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[12:17], v[132:137], a[184:187], v174, v177 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[0:5], v[132:137], a[188:191], v174, v177 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[0:5], v[138:143], a[192:195], v174, v177 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[12:17], v[138:143], a[196:199], v174, v177 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[18:23], v[138:143], a[200:203], v174, v177 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[24:29], v[138:143], a[204:207], v174, v177 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[30:35], v[138:143], a[208:211], v175, v177 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[36:41], v[138:143], a[212:215], v175, v177 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[42:47], v[138:143], a[216:219], v175, v177 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[48:53], v[138:143], a[220:223], v175, v177 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[48:53], v[144:149], a[224:227], v175, v177 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[42:47], v[144:149], a[228:231], v175, v177 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[36:41], v[144:149], a[232:235], v175, v177 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[30:35], v[144:149], a[236:239], v175, v177 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[24:29], v[144:149], a[240:243], v174, v177 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[18:23], v[144:149], a[244:247], v174, v177 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[12:17], v[144:149], a[248:251], v174, v177 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[0:5], v[144:149], a[252:255], v174, v177 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[6:11], v[90:95], a[0:3], v178, v180 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[54:59], v[90:95], a[4:7], v178, v180 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[60:65], v[90:95], a[8:11], v178, v180 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:71], v[90:95], a[12:15], v178, v180 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[72:77], v[90:95], a[16:19], v179, v180 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[78:83], v[90:95], a[20:23], v179, v180 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[84:89], v[90:95], a[24:27], v179, v180 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[96:101], v[90:95], a[28:31], v179, v180 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[96:101], v[102:107], a[32:35], v179, v180 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[84:89], v[102:107], a[36:39], v179, v180 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[78:83], v[102:107], a[40:43], v179, v180 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[72:77], v[102:107], a[44:47], v179, v180 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[66:71], v[102:107], a[48:51], v178, v180 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[60:65], v[102:107], a[52:55], v178, v180 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[54:59], v[102:107], a[56:59], v178, v180 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[6:11], v[102:107], a[60:63], v178, v180 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[6:11], v[108:113], a[64:67], v178, v180 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[54:59], v[108:113], a[68:71], v178, v180 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[60:65], v[108:113], a[72:75], v178, v180 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[66:71], v[108:113], a[76:79], v178, v180 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[72:77], v[108:113], a[80:83], v179, v180 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[78:83], v[108:113], a[84:87], v179, v180 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[84:89], v[108:113], a[88:91], v179, v180 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[96:101], v[108:113], a[92:95], v179, v180 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[96:101], v[150:155], a[96:99], v179, v180 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[84:89], v[150:155], a[100:103], v179, v180 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[78:83], v[150:155], a[104:107], v179, v180 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[72:77], v[150:155], a[108:111], v179, v180 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[66:71], v[150:155], a[112:115], v178, v180 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[60:65], v[150:155], a[116:119], v178, v180 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[54:59], v[150:155], a[120:123], v178, v180 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[6:11], v[150:155], a[124:127], v178, v180 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[6:11], v[114:119], a[128:131], v178, v181 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[54:59], v[114:119], a[132:135], v178, v181 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[60:65], v[114:119], a[136:139], v178, v181 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[66:71], v[114:119], a[140:143], v178, v181 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[72:77], v[114:119], a[144:147], v179, v181 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[78:83], v[114:119], a[148:151], v179, v181 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[84:89], v[114:119], a[152:155], v179, v181 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[96:101], v[114:119], a[156:159], v179, v181 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[96:101], v[156:161], a[160:163], v179, v181 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[84:89], v[156:161], a[164:167], v179, v181 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[78:83], v[156:161], a[168:171], v179, v181 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[72:77], v[156:161], a[172:175], v179, v181 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[66:71], v[156:161], a[176:179], v178, v181 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[60:65], v[156:161], a[180:183], v178, v181 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[54:59], v[156:161], a[184:187], v178, v181 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[6:11], v[156:161], a[188:191], v178, v181 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[6:11], v[162:167], a[192:195], v178, v181 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[54:59], v[162:167], a[196:199], v178, v181 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[60:65], v[162:167], a[200:203], v178, v181 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[66:71], v[162:167], a[204:207], v178, v181 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[72:77], v[162:167], a[208:211], v179, v181 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[78:83], v[162:167], a[212:215], v179, v181 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[84:89], v[162:167], a[216:219], v179, v181 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[96:101], v[162:167], a[220:223], v179, v181 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[96:101], v[168:173], a[224:227], v179, v181 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[84:89], v[168:173], a[228:231], v179, v181 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[78:83], v[168:173], a[232:235], v179, v181 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[72:77], v[168:173], a[236:239], v179, v181 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[66:71], v[168:173], a[240:243], v178, v181 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[60:65], v[168:173], a[244:247], v178, v181 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[54:59], v[168:173], a[248:251], v178, v181 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[6:11], v[168:173], a[252:255], v178, v181 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_waitcnt vmcnt(0)
	s_and_b32 s1, s1, 0xffff
	s_lshl_b32 s2, s61, 13
	s_lshl_b32 s6, s62, 1
	s_add_i32 s7, s2, s6
	v_and_b32_e32 v6, 1, v218
	v_cmp_eq_i32_e64 s[4:5], v6, 1
	v_lshlrev_b32_e32 v7, 1, v6
	v_accvgpr_read_b32 v0, a0
	v_accvgpr_read_b32 v1, a1
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a2
	v_accvgpr_read_b32 v2, a3
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_accvgpr_read_b32 v2, a60
	v_accvgpr_read_b32 v3, a61
	v_cvt_pk_bf16_f32 v2, v2, v3
	v_accvgpr_read_b32 v3, a62
	v_accvgpr_read_b32 v4, a63
	v_cvt_pk_bf16_f32 v3, v3, v4
	v_accvgpr_read_b32 v4, a64
	v_accvgpr_read_b32 v5, a65
	v_cvt_pk_bf16_f32 v4, v4, v5
	v_accvgpr_read_b32 v5, a66
	v_accvgpr_read_b32 v8, a67
	v_cvt_pk_bf16_f32 v5, v5, v8
	v_lshl_add_u32 v7, v219, 2, v7
	v_mov_b32_dpp v8, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v9, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v10, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v11, v3 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v12, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v13, v5 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_sub_u32_e32 v6, v218, v6
	v_cndmask_b32_e64 v0, v0, v9, s[4:5]
	v_cndmask_b32_e64 v1, v8, v1, s[4:5]
	v_lshlrev_b32_e32 v7, 13, v7
	v_cndmask_b32_e64 v2, v2, v11, s[4:5]
	v_cndmask_b32_e64 v3, v10, v3, s[4:5]
	v_cndmask_b32_e64 v4, v4, v13, s[4:5]
	v_cndmask_b32_e64 v5, v12, v5, s[4:5]
	v_lshlrev_b32_e32 v8, 16, v1
	v_lshrrev_b32_e32 v9, 16, v0
	v_lshl_add_u32 v6, v6, 1, v7
	v_lshlrev_b32_e32 v7, 16, v3
	v_lshrrev_b32_e32 v10, 16, v2
	v_lshlrev_b32_e32 v11, 16, v5
	v_lshrrev_b32_e32 v12, 16, v4
	s_mov_b32 s2, 0xffff
	v_and_or_b32 v8, s2, v0, v8
	s_mov_b32 s3, 0xffff0000
	v_and_or_b32 v9, s3, v1, v9
	v_add_u32_e32 v13, 0x2000, v6
	v_and_or_b32 v7, s2, v2, v7
	v_and_or_b32 v10, s3, v3, v10
	v_add_u32_e32 v14, 32, v6
	v_add_u32_e32 v15, 0x2020, v6
	s_mov_b32 s8, 0xffff
	v_and_or_b32 v4, s8, v4, v11
	s_mov_b32 s9, 0xffff0000
	v_and_or_b32 v5, s9, v5, v12
	v_add_u32_e32 v11, 64, v6
	v_add_u32_e32 v12, 0x2040, v6
	v_accvgpr_read_b32 v0, a124
	v_accvgpr_read_b32 v1, a125
	v_cvt_pk_bf16_f32 v0, v0, v1
	s_mov_b32 s2, 0x7ffffff0
	v_accvgpr_read_b32 v1, a126
	v_accvgpr_read_b32 v2, a127
	v_cvt_pk_bf16_f32 v1, v1, v2
	s_mov_b32 s3, 0x31014000
	v_accvgpr_read_b32 v2, a128
	v_accvgpr_read_b32 v3, a129
	v_cvt_pk_bf16_f32 v2, v2, v3
	v_accvgpr_read_b32 v3, a130
	v_accvgpr_read_b32 v16, a131
	v_cvt_pk_bf16_f32 v3, v3, v16
	v_mov_b32_dpp v16, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v17, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v18, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v19, v3 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v0, v0, v17, s[4:5]
	v_cndmask_b32_e64 v1, v16, v1, s[4:5]
	v_cndmask_b32_e64 v2, v2, v19, s[4:5]
	v_cndmask_b32_e64 v3, v18, v3, s[4:5]
	v_lshlrev_b32_e32 v16, 16, v1
	v_lshrrev_b32_e32 v17, 16, v0
	v_lshlrev_b32_e32 v18, 16, v3
	v_and_or_b32 v16, s8, v0, v16
	v_and_or_b32 v17, s9, v1, v17
	v_add_u32_e32 v19, 0x60, v6
	v_add_u32_e32 v20, 0x2060, v6
	s_mov_b32 s8, 0xffff
	v_and_or_b32 v18, s8, v2, v18
	v_lshrrev_b32_e32 v0, 16, v2
	buffer_store_dword v8, v6, s[0:3], s7 offen
	s_mov_b32 s9, 0xffff0000
	v_and_or_b32 v2, s9, v3, v0
	v_add_u32_e32 v8, 0x80, v6
	buffer_store_dword v9, v13, s[0:3], s7 offen
	v_accvgpr_read_b32 v0, a188
	v_accvgpr_read_b32 v1, a189
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a190
	v_accvgpr_read_b32 v3, a191
	v_cvt_pk_bf16_f32 v1, v1, v3
	v_mov_b32_dpp v3, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v9, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v0, v0, v9, s[4:5]
	v_cndmask_b32_e64 v1, v3, v1, s[4:5]
	v_lshlrev_b32_e32 v3, 16, v1
	v_add_u32_e32 v9, 0x2080, v6
	v_and_or_b32 v3, s8, v0, v3
	buffer_store_dword v7, v14, s[0:3], s7 offen
	v_lshrrev_b32_e32 v0, 16, v0
	v_and_or_b32 v7, s9, v1, v0
	v_add_u32_e32 v21, 0xa0, v6
	buffer_store_dword v10, v15, s[0:3], s7 offen
	v_accvgpr_read_b32 v0, a192
	v_accvgpr_read_b32 v1, a193
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a194
	v_accvgpr_read_b32 v10, a195
	v_cvt_pk_bf16_f32 v1, v1, v10
	v_mov_b32_dpp v10, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v22, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v0, v0, v22, s[4:5]
	v_cndmask_b32_e64 v1, v10, v1, s[4:5]
	v_lshlrev_b32_e32 v10, 16, v1
	v_add_u32_e32 v22, 0x20a0, v6
	s_mov_b32 s8, 0xffff
	v_and_or_b32 v10, s8, v0, v10
	v_lshrrev_b32_e32 v0, 16, v0
	buffer_store_dword v4, v11, s[0:3], s7 offen
	s_mov_b32 s9, 0xffff0000
	v_and_or_b32 v4, s9, v1, v0
	v_add_u32_e32 v23, 0xc0, v6
	buffer_store_dword v5, v12, s[0:3], s7 offen
	v_accvgpr_read_b32 v0, a252
	v_accvgpr_read_b32 v1, a253
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a254
	v_accvgpr_read_b32 v5, a255
	v_cvt_pk_bf16_f32 v1, v1, v5
	v_mov_b32_dpp v5, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v24, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v0, v0, v24, s[4:5]
	v_cndmask_b32_e64 v1, v5, v1, s[4:5]
	v_lshlrev_b32_e32 v5, 16, v1
	v_add_u32_e32 v24, 0x20c0, v6
	v_and_or_b32 v5, s8, v0, v5
	buffer_store_dword v16, v19, s[0:3], s7 offen
	v_lshrrev_b32_e32 v0, 16, v0
	v_and_or_b32 v16, s9, v1, v0
	v_add_u32_e32 v25, 0xe0, v6
	buffer_store_dword v17, v20, s[0:3], s7 offen
	s_add_i32 s8, s61, 16
	s_lshl_b32 s8, s8, 13
	s_add_i32 s8, s6, s8
	v_accvgpr_read_b32 v0, a4
	v_accvgpr_read_b32 v1, a5
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a6
	v_accvgpr_read_b32 v17, a7
	v_cvt_pk_bf16_f32 v1, v1, v17
	v_mov_b32_dpp v17, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v26, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v0, v0, v26, s[4:5]
	v_cndmask_b32_e64 v17, v17, v1, s[4:5]
	v_add_u32_e32 v26, 0x20e0, v6
	v_lshlrev_b32_e32 v1, 16, v17
	buffer_store_dword v18, v8, s[0:3], s7 offen
	s_mov_b32 s9, 0xffff
	v_and_or_b32 v18, s9, v0, v1
	v_lshrrev_b32_e32 v27, 16, v0
	buffer_store_dword v2, v9, s[0:3], s7 offen
	v_accvgpr_read_b32 v0, a56
	v_accvgpr_read_b32 v1, a57
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a58
	v_accvgpr_read_b32 v2, a59
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_mov_b32_dpp v2, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v28, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v0, v0, v28, s[4:5]
	v_cndmask_b32_e64 v1, v2, v1, s[4:5]
	s_mov_b32 s10, 0xffff0000
	v_and_or_b32 v17, s10, v17, v27
	v_lshlrev_b32_e32 v2, 16, v1
	buffer_store_dword v3, v21, s[0:3], s7 offen
	v_lshrrev_b32_e32 v3, 16, v0
	v_and_or_b32 v27, s9, v0, v2
	v_and_or_b32 v28, s10, v1, v3
	buffer_store_dword v7, v22, s[0:3], s7 offen
	v_accvgpr_read_b32 v0, a68
	v_accvgpr_read_b32 v1, a69
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a70
	v_accvgpr_read_b32 v2, a71
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_mov_b32_dpp v2, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v3, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v0, v0, v3, s[4:5]
	v_cndmask_b32_e64 v1, v2, v1, s[4:5]
	v_lshlrev_b32_e32 v2, 16, v1
	v_lshrrev_b32_e32 v3, 16, v0
	v_and_or_b32 v7, s9, v0, v2
	v_and_or_b32 v29, s10, v1, v3
	buffer_store_dword v10, v23, s[0:3], s7 offen
	buffer_store_dword v4, v24, s[0:3], s7 offen
	v_accvgpr_read_b32 v0, a120
	v_accvgpr_read_b32 v1, a121
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a122
	v_accvgpr_read_b32 v2, a123
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_mov_b32_dpp v2, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v3, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v0, v0, v3, s[4:5]
	v_cndmask_b32_e64 v1, v2, v1, s[4:5]
	v_lshlrev_b32_e32 v2, 16, v1
	v_lshrrev_b32_e32 v3, 16, v0
	s_mov_b32 s9, 0xffff
	v_and_or_b32 v4, s9, v0, v2
	s_mov_b32 s10, 0xffff0000
	v_and_or_b32 v10, s10, v1, v3
	buffer_store_dword v5, v25, s[0:3], s7 offen
	buffer_store_dword v16, v26, s[0:3], s7 offen
	v_accvgpr_read_b32 v0, a132
	v_accvgpr_read_b32 v1, a133
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a134
	v_accvgpr_read_b32 v2, a135
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_accvgpr_read_b32 v2, a184
	v_accvgpr_read_b32 v3, a185
	v_cvt_pk_bf16_f32 v2, v2, v3
	v_accvgpr_read_b32 v3, a186
	v_accvgpr_read_b32 v5, a187
	v_cvt_pk_bf16_f32 v3, v3, v5
	v_mov_b32_dpp v5, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v16, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v0, v0, v16, s[4:5]
	v_cndmask_b32_e64 v1, v5, v1, s[4:5]
	v_lshlrev_b32_e32 v5, 16, v1
	v_lshrrev_b32_e32 v16, 16, v0
	v_and_or_b32 v5, s9, v0, v5
	v_and_or_b32 v16, s10, v1, v16
	v_mov_b32_dpp v30, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	buffer_store_dword v18, v6, s[0:3], s8 offen
	v_mov_b32_dpp v0, v3 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v2, v2, v0, s[4:5]
	buffer_store_dword v17, v13, s[0:3], s8 offen
	v_accvgpr_read_b32 v0, a196
	v_accvgpr_read_b32 v1, a197
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a198
	v_accvgpr_read_b32 v17, a199
	v_cvt_pk_bf16_f32 v1, v1, v17
	v_cndmask_b32_e64 v3, v30, v3, s[4:5]
	v_lshlrev_b32_e32 v17, 16, v3
	v_lshrrev_b32_e32 v18, 16, v2
	v_and_or_b32 v2, s9, v2, v17
	v_and_or_b32 v3, s10, v3, v18
	v_mov_b32_dpp v17, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	buffer_store_dword v27, v14, s[0:3], s8 offen
	v_mov_b32_dpp v18, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v0, v0, v18, s[4:5]
	buffer_store_dword v28, v15, s[0:3], s8 offen
	v_cndmask_b32_e64 v1, v17, v1, s[4:5]
	v_lshlrev_b32_e32 v17, 16, v1
	v_lshrrev_b32_e32 v18, 16, v0
	s_mov_b32 s7, 0xffff
	v_and_or_b32 v17, s7, v0, v17
	s_mov_b32 s9, 0xffff0000
	v_and_or_b32 v18, s9, v1, v18
	v_accvgpr_read_b32 v0, a248
	v_accvgpr_read_b32 v1, a249
	v_cvt_pk_bf16_f32 v0, v0, v1
	buffer_store_dword v7, v11, s[0:3], s8 offen
	v_accvgpr_read_b32 v1, a250
	v_accvgpr_read_b32 v7, a251
	v_cvt_pk_bf16_f32 v1, v1, v7
	v_mov_b32_dpp v7, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	buffer_store_dword v29, v12, s[0:3], s8 offen
	s_add_i32 s10, s61, 32
	s_lshl_b32 s10, s10, 13
	v_mov_b32_dpp v27, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v0, v0, v27, s[4:5]
	v_cndmask_b32_e64 v1, v7, v1, s[4:5]
	v_lshlrev_b32_e32 v7, 16, v1
	v_lshrrev_b32_e32 v27, 16, v0
	v_and_or_b32 v7, s7, v0, v7
	v_and_or_b32 v27, s9, v1, v27
	s_add_i32 s7, s6, s10
	buffer_store_dword v4, v19, s[0:3], s8 offen
	v_accvgpr_read_b32 v0, a8
	v_accvgpr_read_b32 v1, a9
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a10
	v_accvgpr_read_b32 v4, a11
	v_cvt_pk_bf16_f32 v1, v1, v4
	buffer_store_dword v10, v20, s[0:3], s8 offen
	v_mov_b32_dpp v4, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v10, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v0, v0, v10, s[4:5]
	v_cndmask_b32_e64 v1, v4, v1, s[4:5]
	v_lshlrev_b32_e32 v4, 16, v1
	v_lshrrev_b32_e32 v10, 16, v0
	s_mov_b32 s9, 0xffff
	v_and_or_b32 v4, s9, v0, v4
	s_mov_b32 s10, 0xffff0000
	v_and_or_b32 v10, s10, v1, v10
	buffer_store_dword v5, v8, s[0:3], s8 offen
	buffer_store_dword v16, v9, s[0:3], s8 offen
	v_accvgpr_read_b32 v0, a52
	v_accvgpr_read_b32 v1, a53
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a54
	v_accvgpr_read_b32 v5, a55
	v_cvt_pk_bf16_f32 v1, v1, v5
	v_mov_b32_dpp v5, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v16, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v0, v0, v16, s[4:5]
	v_cndmask_b32_e64 v1, v5, v1, s[4:5]
	v_lshlrev_b32_e32 v5, 16, v1
	v_lshrrev_b32_e32 v16, 16, v0
	v_and_or_b32 v5, s9, v0, v5
	v_and_or_b32 v16, s10, v1, v16
	buffer_store_dword v2, v21, s[0:3], s8 offen
	buffer_store_dword v3, v22, s[0:3], s8 offen
	v_accvgpr_read_b32 v0, a72
	v_accvgpr_read_b32 v1, a73
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a74
	v_accvgpr_read_b32 v2, a75
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_mov_b32_dpp v2, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v3, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v0, v0, v3, s[4:5]
	v_cndmask_b32_e64 v1, v2, v1, s[4:5]
	v_lshlrev_b32_e32 v2, 16, v1
	v_lshrrev_b32_e32 v3, 16, v0
	v_and_or_b32 v28, s9, v0, v2
	v_and_or_b32 v29, s10, v1, v3
	buffer_store_dword v17, v23, s[0:3], s8 offen
	v_accvgpr_read_b32 v0, a116
	v_accvgpr_read_b32 v1, a117
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a118
	v_accvgpr_read_b32 v2, a119
	v_cvt_pk_bf16_f32 v1, v1, v2
	buffer_store_dword v18, v24, s[0:3], s8 offen
	v_mov_b32_dpp v2, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v3, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v0, v0, v3, s[4:5]
	v_cndmask_b32_e64 v1, v2, v1, s[4:5]
	v_lshlrev_b32_e32 v2, 16, v1
	v_lshrrev_b32_e32 v3, 16, v0
	s_mov_b32 s9, 0xffff
	v_and_or_b32 v17, s9, v0, v2
	s_mov_b32 s10, 0xffff0000
	v_and_or_b32 v18, s10, v1, v3
	buffer_store_dword v7, v25, s[0:3], s8 offen
	buffer_store_dword v27, v26, s[0:3], s8 offen
	v_accvgpr_read_b32 v0, a136
	v_accvgpr_read_b32 v1, a137
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a138
	v_accvgpr_read_b32 v2, a139
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_accvgpr_read_b32 v2, a180
	v_accvgpr_read_b32 v3, a181
	v_cvt_pk_bf16_f32 v2, v2, v3
	v_accvgpr_read_b32 v3, a182
	v_accvgpr_read_b32 v7, a183
	v_cvt_pk_bf16_f32 v3, v3, v7
	v_mov_b32_dpp v7, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v27, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v0, v0, v27, s[4:5]
	v_cndmask_b32_e64 v1, v7, v1, s[4:5]
	v_lshlrev_b32_e32 v7, 16, v1
	v_lshrrev_b32_e32 v27, 16, v0
	v_and_or_b32 v7, s9, v0, v7
	v_and_or_b32 v27, s10, v1, v27
	v_mov_b32_dpp v30, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	buffer_store_dword v4, v6, s[0:3], s7 offen
	v_mov_b32_dpp v0, v3 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v2, v2, v0, s[4:5]
	buffer_store_dword v10, v13, s[0:3], s7 offen
	v_accvgpr_read_b32 v0, a200
	v_accvgpr_read_b32 v1, a201
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a202
	v_accvgpr_read_b32 v4, a203
	v_cvt_pk_bf16_f32 v1, v1, v4
	v_cndmask_b32_e64 v3, v30, v3, s[4:5]
	v_lshlrev_b32_e32 v4, 16, v3
	v_lshrrev_b32_e32 v10, 16, v2
	v_and_or_b32 v2, s9, v2, v4
	v_and_or_b32 v3, s10, v3, v10
	v_mov_b32_dpp v4, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	buffer_store_dword v5, v14, s[0:3], s7 offen
	v_mov_b32_dpp v5, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v0, v0, v5, s[4:5]
	buffer_store_dword v16, v15, s[0:3], s7 offen
	v_cndmask_b32_e64 v1, v4, v1, s[4:5]
	v_lshlrev_b32_e32 v4, 16, v1
	v_lshrrev_b32_e32 v5, 16, v0
	s_mov_b32 s8, 0xffff
	v_and_or_b32 v4, s8, v0, v4
	s_mov_b32 s9, 0xffff0000
	v_and_or_b32 v5, s9, v1, v5
	v_accvgpr_read_b32 v0, a244
	v_accvgpr_read_b32 v1, a245
	v_cvt_pk_bf16_f32 v0, v0, v1
	buffer_store_dword v28, v11, s[0:3], s7 offen
	v_accvgpr_read_b32 v1, a246
	v_accvgpr_read_b32 v10, a247
	v_cvt_pk_bf16_f32 v1, v1, v10
	v_mov_b32_dpp v10, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	buffer_store_dword v29, v12, s[0:3], s7 offen
	s_add_i32 s10, s61, 48
	s_lshl_b32 s10, s10, 13
	v_mov_b32_dpp v16, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v0, v0, v16, s[4:5]
	v_cndmask_b32_e64 v1, v10, v1, s[4:5]
	v_lshlrev_b32_e32 v10, 16, v1
	v_lshrrev_b32_e32 v16, 16, v0
	v_and_or_b32 v10, s8, v0, v10
	v_and_or_b32 v16, s9, v1, v16
	s_add_i32 s8, s6, s10
	buffer_store_dword v17, v19, s[0:3], s7 offen
	v_accvgpr_read_b32 v0, a12
	v_accvgpr_read_b32 v1, a13
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a14
	v_accvgpr_read_b32 v17, a15
	v_cvt_pk_bf16_f32 v1, v1, v17
	buffer_store_dword v18, v20, s[0:3], s7 offen
	v_mov_b32_dpp v17, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v18, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v0, v0, v18, s[4:5]
	v_cndmask_b32_e64 v1, v17, v1, s[4:5]
	v_lshlrev_b32_e32 v17, 16, v1
	v_lshrrev_b32_e32 v18, 16, v0
	s_mov_b32 s9, 0xffff
	v_and_or_b32 v17, s9, v0, v17
	s_mov_b32 s10, 0xffff0000
	v_and_or_b32 v18, s10, v1, v18
	buffer_store_dword v7, v8, s[0:3], s7 offen
	buffer_store_dword v27, v9, s[0:3], s7 offen
	v_accvgpr_read_b32 v0, a48
	v_accvgpr_read_b32 v1, a49
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a50
	v_accvgpr_read_b32 v7, a51
	v_cvt_pk_bf16_f32 v1, v1, v7
	v_mov_b32_dpp v7, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v27, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v0, v0, v27, s[4:5]
	v_cndmask_b32_e64 v1, v7, v1, s[4:5]
	v_lshlrev_b32_e32 v7, 16, v1
	v_lshrrev_b32_e32 v27, 16, v0
	v_and_or_b32 v7, s9, v0, v7
	v_and_or_b32 v27, s10, v1, v27
	buffer_store_dword v2, v21, s[0:3], s7 offen
	buffer_store_dword v3, v22, s[0:3], s7 offen
	v_accvgpr_read_b32 v0, a76
	v_accvgpr_read_b32 v1, a77
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a78
	v_accvgpr_read_b32 v2, a79
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_mov_b32_dpp v2, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v3, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v0, v0, v3, s[4:5]
	v_cndmask_b32_e64 v1, v2, v1, s[4:5]
	v_lshlrev_b32_e32 v2, 16, v1
	v_lshrrev_b32_e32 v3, 16, v0
	v_and_or_b32 v28, s9, v0, v2
	v_and_or_b32 v29, s10, v1, v3
	buffer_store_dword v4, v23, s[0:3], s7 offen
	v_accvgpr_read_b32 v0, a112
	v_accvgpr_read_b32 v1, a113
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a114
	v_accvgpr_read_b32 v2, a115
	v_cvt_pk_bf16_f32 v1, v1, v2
	buffer_store_dword v5, v24, s[0:3], s7 offen
	v_mov_b32_dpp v2, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v3, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v0, v0, v3, s[4:5]
	v_cndmask_b32_e64 v1, v2, v1, s[4:5]
	v_lshlrev_b32_e32 v2, 16, v1
	v_lshrrev_b32_e32 v3, 16, v0
	s_mov_b32 s9, 0xffff
	v_and_or_b32 v4, s9, v0, v2
	s_mov_b32 s10, 0xffff0000
	v_and_or_b32 v5, s10, v1, v3
	buffer_store_dword v10, v25, s[0:3], s7 offen
	buffer_store_dword v16, v26, s[0:3], s7 offen
	v_accvgpr_read_b32 v0, a140
	v_accvgpr_read_b32 v1, a141
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a142
	v_accvgpr_read_b32 v2, a143
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_accvgpr_read_b32 v2, a176
	v_accvgpr_read_b32 v3, a177
	v_cvt_pk_bf16_f32 v2, v2, v3
	v_accvgpr_read_b32 v3, a178
	v_accvgpr_read_b32 v10, a179
	v_cvt_pk_bf16_f32 v3, v3, v10
	v_mov_b32_dpp v10, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v16, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v0, v0, v16, s[4:5]
	v_cndmask_b32_e64 v1, v10, v1, s[4:5]
	v_lshlrev_b32_e32 v10, 16, v1
	v_lshrrev_b32_e32 v16, 16, v0
	v_and_or_b32 v10, s9, v0, v10
	v_and_or_b32 v16, s10, v1, v16
	v_mov_b32_dpp v30, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	buffer_store_dword v17, v6, s[0:3], s8 offen
	v_mov_b32_dpp v0, v3 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v2, v2, v0, s[4:5]
	buffer_store_dword v18, v13, s[0:3], s8 offen
	v_accvgpr_read_b32 v0, a204
	v_accvgpr_read_b32 v1, a205
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a206
	v_accvgpr_read_b32 v17, a207
	v_cvt_pk_bf16_f32 v1, v1, v17
	v_cndmask_b32_e64 v3, v30, v3, s[4:5]
	v_lshlrev_b32_e32 v17, 16, v3
	v_lshrrev_b32_e32 v18, 16, v2
	v_and_or_b32 v2, s9, v2, v17
	v_and_or_b32 v3, s10, v3, v18
	v_mov_b32_dpp v17, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	buffer_store_dword v7, v14, s[0:3], s8 offen
	v_mov_b32_dpp v7, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v0, v0, v7, s[4:5]
	buffer_store_dword v27, v15, s[0:3], s8 offen
	v_cndmask_b32_e64 v1, v17, v1, s[4:5]
	v_lshlrev_b32_e32 v7, 16, v1
	v_lshrrev_b32_e32 v17, 16, v0
	s_mov_b32 s7, 0xffff
	v_and_or_b32 v7, s7, v0, v7
	s_mov_b32 s9, 0xffff0000
	v_and_or_b32 v17, s9, v1, v17
	v_accvgpr_read_b32 v0, a240
	v_accvgpr_read_b32 v1, a241
	v_cvt_pk_bf16_f32 v0, v0, v1
	buffer_store_dword v28, v11, s[0:3], s8 offen
	v_accvgpr_read_b32 v1, a242
	v_accvgpr_read_b32 v18, a243
	v_cvt_pk_bf16_f32 v1, v1, v18
	v_mov_b32_dpp v18, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	buffer_store_dword v29, v12, s[0:3], s8 offen
	s_add_i32 s10, s61, 64
	s_lshl_b32 s10, s10, 13
	v_mov_b32_dpp v27, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v0, v0, v27, s[4:5]
	v_cndmask_b32_e64 v1, v18, v1, s[4:5]
	v_lshlrev_b32_e32 v18, 16, v1
	v_lshrrev_b32_e32 v27, 16, v0
	v_and_or_b32 v18, s7, v0, v18
	v_and_or_b32 v27, s9, v1, v27
	s_add_i32 s7, s6, s10
	buffer_store_dword v4, v19, s[0:3], s8 offen
	v_accvgpr_read_b32 v0, a16
	v_accvgpr_read_b32 v1, a17
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a18
	v_accvgpr_read_b32 v4, a19
	v_cvt_pk_bf16_f32 v1, v1, v4
	buffer_store_dword v5, v20, s[0:3], s8 offen
	v_mov_b32_dpp v4, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v5, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v0, v0, v5, s[4:5]
	v_cndmask_b32_e64 v1, v4, v1, s[4:5]
	v_lshlrev_b32_e32 v4, 16, v1
	v_lshrrev_b32_e32 v5, 16, v0
	s_mov_b32 s9, 0xffff
	v_and_or_b32 v4, s9, v0, v4
	s_mov_b32 s10, 0xffff0000
	v_and_or_b32 v5, s10, v1, v5
	buffer_store_dword v10, v8, s[0:3], s8 offen
	buffer_store_dword v16, v9, s[0:3], s8 offen
	v_accvgpr_read_b32 v0, a44
	v_accvgpr_read_b32 v1, a45
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a46
	v_accvgpr_read_b32 v10, a47
	v_cvt_pk_bf16_f32 v1, v1, v10
	v_mov_b32_dpp v10, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v16, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v0, v0, v16, s[4:5]
	v_cndmask_b32_e64 v1, v10, v1, s[4:5]
	v_lshlrev_b32_e32 v10, 16, v1
	v_lshrrev_b32_e32 v16, 16, v0
	v_and_or_b32 v10, s9, v0, v10
	v_and_or_b32 v16, s10, v1, v16
	buffer_store_dword v2, v21, s[0:3], s8 offen
	buffer_store_dword v3, v22, s[0:3], s8 offen
	v_accvgpr_read_b32 v0, a80
	v_accvgpr_read_b32 v1, a81
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a82
	v_accvgpr_read_b32 v2, a83
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_mov_b32_dpp v2, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v3, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v0, v0, v3, s[4:5]
	v_cndmask_b32_e64 v1, v2, v1, s[4:5]
	v_lshlrev_b32_e32 v2, 16, v1
	v_lshrrev_b32_e32 v3, 16, v0
	v_and_or_b32 v28, s9, v0, v2
	v_and_or_b32 v29, s10, v1, v3
	buffer_store_dword v7, v23, s[0:3], s8 offen
	v_accvgpr_read_b32 v0, a108
	v_accvgpr_read_b32 v1, a109
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a110
	v_accvgpr_read_b32 v2, a111
	v_cvt_pk_bf16_f32 v1, v1, v2
	buffer_store_dword v17, v24, s[0:3], s8 offen
	v_mov_b32_dpp v2, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v3, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v0, v0, v3, s[4:5]
	v_cndmask_b32_e64 v1, v2, v1, s[4:5]
	v_lshlrev_b32_e32 v2, 16, v1
	v_lshrrev_b32_e32 v3, 16, v0
	s_mov_b32 s9, 0xffff
	v_and_or_b32 v7, s9, v0, v2
	s_mov_b32 s10, 0xffff0000
	v_and_or_b32 v17, s10, v1, v3
	buffer_store_dword v18, v25, s[0:3], s8 offen
	buffer_store_dword v27, v26, s[0:3], s8 offen
	v_accvgpr_read_b32 v0, a144
	v_accvgpr_read_b32 v1, a145
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a146
	v_accvgpr_read_b32 v2, a147
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_accvgpr_read_b32 v2, a172
	v_accvgpr_read_b32 v3, a173
	v_cvt_pk_bf16_f32 v2, v2, v3
	v_accvgpr_read_b32 v3, a174
	v_accvgpr_read_b32 v18, a175
	v_cvt_pk_bf16_f32 v3, v3, v18
	v_mov_b32_dpp v18, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v27, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v0, v0, v27, s[4:5]
	v_cndmask_b32_e64 v1, v18, v1, s[4:5]
	v_lshlrev_b32_e32 v18, 16, v1
	v_lshrrev_b32_e32 v27, 16, v0
	v_and_or_b32 v18, s9, v0, v18
	v_and_or_b32 v27, s10, v1, v27
	v_mov_b32_dpp v30, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	buffer_store_dword v4, v6, s[0:3], s7 offen
	v_mov_b32_dpp v0, v3 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v2, v2, v0, s[4:5]
	buffer_store_dword v5, v13, s[0:3], s7 offen
	v_accvgpr_read_b32 v0, a208
	v_accvgpr_read_b32 v1, a209
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a210
	v_accvgpr_read_b32 v4, a211
	v_cvt_pk_bf16_f32 v1, v1, v4
	v_cndmask_b32_e64 v3, v30, v3, s[4:5]
	v_lshlrev_b32_e32 v4, 16, v3
	v_lshrrev_b32_e32 v5, 16, v2
	v_and_or_b32 v2, s9, v2, v4
	v_and_or_b32 v3, s10, v3, v5
	v_mov_b32_dpp v4, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	buffer_store_dword v10, v14, s[0:3], s7 offen
	v_mov_b32_dpp v5, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v0, v0, v5, s[4:5]
	buffer_store_dword v16, v15, s[0:3], s7 offen
	v_cndmask_b32_e64 v1, v4, v1, s[4:5]
	v_lshlrev_b32_e32 v4, 16, v1
	v_lshrrev_b32_e32 v5, 16, v0
	s_mov_b32 s8, 0xffff
	v_and_or_b32 v4, s8, v0, v4
	s_mov_b32 s9, 0xffff0000
	v_and_or_b32 v5, s9, v1, v5
	v_accvgpr_read_b32 v0, a236
	v_accvgpr_read_b32 v1, a237
	v_cvt_pk_bf16_f32 v0, v0, v1
	buffer_store_dword v28, v11, s[0:3], s7 offen
	v_accvgpr_read_b32 v1, a238
	v_accvgpr_read_b32 v10, a239
	v_cvt_pk_bf16_f32 v1, v1, v10
	v_mov_b32_dpp v10, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	buffer_store_dword v29, v12, s[0:3], s7 offen
	s_add_i32 s10, s61, 0x50
	s_lshl_b32 s10, s10, 13
	v_mov_b32_dpp v16, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v0, v0, v16, s[4:5]
	v_cndmask_b32_e64 v1, v10, v1, s[4:5]
	v_lshlrev_b32_e32 v10, 16, v1
	v_lshrrev_b32_e32 v16, 16, v0
	v_and_or_b32 v10, s8, v0, v10
	v_and_or_b32 v16, s9, v1, v16
	s_add_i32 s8, s6, s10
	buffer_store_dword v7, v19, s[0:3], s7 offen
	v_accvgpr_read_b32 v0, a20
	v_accvgpr_read_b32 v1, a21
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a22
	v_accvgpr_read_b32 v7, a23
	v_cvt_pk_bf16_f32 v1, v1, v7
	buffer_store_dword v17, v20, s[0:3], s7 offen
	v_mov_b32_dpp v7, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v17, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v0, v0, v17, s[4:5]
	v_cndmask_b32_e64 v1, v7, v1, s[4:5]
	v_lshlrev_b32_e32 v7, 16, v1
	v_lshrrev_b32_e32 v17, 16, v0
	s_mov_b32 s9, 0xffff
	v_and_or_b32 v7, s9, v0, v7
	s_mov_b32 s10, 0xffff0000
	v_and_or_b32 v17, s10, v1, v17
	buffer_store_dword v18, v8, s[0:3], s7 offen
	buffer_store_dword v27, v9, s[0:3], s7 offen
	v_accvgpr_read_b32 v0, a40
	v_accvgpr_read_b32 v1, a41
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a42
	v_accvgpr_read_b32 v18, a43
	v_cvt_pk_bf16_f32 v1, v1, v18
	v_mov_b32_dpp v18, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v27, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v0, v0, v27, s[4:5]
	v_cndmask_b32_e64 v1, v18, v1, s[4:5]
	v_lshlrev_b32_e32 v18, 16, v1
	v_lshrrev_b32_e32 v27, 16, v0
	v_and_or_b32 v18, s9, v0, v18
	v_and_or_b32 v27, s10, v1, v27
	buffer_store_dword v2, v21, s[0:3], s7 offen
	buffer_store_dword v3, v22, s[0:3], s7 offen
	v_accvgpr_read_b32 v0, a84
	v_accvgpr_read_b32 v1, a85
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a86
	v_accvgpr_read_b32 v2, a87
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_mov_b32_dpp v2, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v3, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v0, v0, v3, s[4:5]
	v_cndmask_b32_e64 v1, v2, v1, s[4:5]
	v_lshlrev_b32_e32 v2, 16, v1
	v_lshrrev_b32_e32 v3, 16, v0
	v_and_or_b32 v28, s9, v0, v2
	v_and_or_b32 v29, s10, v1, v3
	buffer_store_dword v4, v23, s[0:3], s7 offen
	v_accvgpr_read_b32 v0, a104
	v_accvgpr_read_b32 v1, a105
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a106
	v_accvgpr_read_b32 v2, a107
	v_cvt_pk_bf16_f32 v1, v1, v2
	buffer_store_dword v5, v24, s[0:3], s7 offen
	v_mov_b32_dpp v2, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v3, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v0, v0, v3, s[4:5]
	v_cndmask_b32_e64 v1, v2, v1, s[4:5]
	v_lshlrev_b32_e32 v2, 16, v1
	v_lshrrev_b32_e32 v3, 16, v0
	s_mov_b32 s9, 0xffff
	v_and_or_b32 v4, s9, v0, v2
	s_mov_b32 s10, 0xffff0000
	v_and_or_b32 v5, s10, v1, v3
	buffer_store_dword v10, v25, s[0:3], s7 offen
	buffer_store_dword v16, v26, s[0:3], s7 offen
	v_accvgpr_read_b32 v0, a148
	v_accvgpr_read_b32 v1, a149
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a150
	v_accvgpr_read_b32 v2, a151
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_accvgpr_read_b32 v2, a168
	v_accvgpr_read_b32 v3, a169
	v_cvt_pk_bf16_f32 v2, v2, v3
	v_accvgpr_read_b32 v3, a170
	v_accvgpr_read_b32 v10, a171
	v_cvt_pk_bf16_f32 v3, v3, v10
	v_mov_b32_dpp v10, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v16, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v0, v0, v16, s[4:5]
	v_cndmask_b32_e64 v1, v10, v1, s[4:5]
	v_lshlrev_b32_e32 v10, 16, v1
	v_lshrrev_b32_e32 v16, 16, v0
	v_and_or_b32 v10, s9, v0, v10
	v_and_or_b32 v16, s10, v1, v16
	v_mov_b32_dpp v30, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	buffer_store_dword v7, v6, s[0:3], s8 offen
	v_mov_b32_dpp v0, v3 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v2, v2, v0, s[4:5]
	buffer_store_dword v17, v13, s[0:3], s8 offen
	v_accvgpr_read_b32 v0, a212
	v_accvgpr_read_b32 v1, a213
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a214
	v_accvgpr_read_b32 v7, a215
	v_cvt_pk_bf16_f32 v1, v1, v7
	v_cndmask_b32_e64 v3, v30, v3, s[4:5]
	v_lshlrev_b32_e32 v7, 16, v3
	v_lshrrev_b32_e32 v17, 16, v2
	v_and_or_b32 v2, s9, v2, v7
	v_and_or_b32 v3, s10, v3, v17
	v_mov_b32_dpp v7, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	buffer_store_dword v18, v14, s[0:3], s8 offen
	v_mov_b32_dpp v17, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v0, v0, v17, s[4:5]
	buffer_store_dword v27, v15, s[0:3], s8 offen
	v_cndmask_b32_e64 v1, v7, v1, s[4:5]
	v_lshlrev_b32_e32 v7, 16, v1
	v_lshrrev_b32_e32 v17, 16, v0
	s_mov_b32 s7, 0xffff
	v_and_or_b32 v7, s7, v0, v7
	s_mov_b32 s9, 0xffff0000
	v_and_or_b32 v17, s9, v1, v17
	v_accvgpr_read_b32 v0, a232
	v_accvgpr_read_b32 v1, a233
	v_cvt_pk_bf16_f32 v0, v0, v1
	buffer_store_dword v28, v11, s[0:3], s8 offen
	v_accvgpr_read_b32 v1, a234
	v_accvgpr_read_b32 v18, a235
	v_cvt_pk_bf16_f32 v1, v1, v18
	v_mov_b32_dpp v18, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	buffer_store_dword v29, v12, s[0:3], s8 offen
	s_add_i32 s10, s61, 0x60
	v_mov_b32_dpp v27, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_lshl_b32 s10, s10, 13
	v_cndmask_b32_e64 v0, v0, v27, s[4:5]
	v_cndmask_b32_e64 v1, v18, v1, s[4:5]
	v_lshlrev_b32_e32 v18, 16, v1
	v_lshrrev_b32_e32 v27, 16, v0
	v_and_or_b32 v18, s7, v0, v18
	v_and_or_b32 v27, s9, v1, v27
	s_add_i32 s7, s6, s10
	buffer_store_dword v4, v19, s[0:3], s8 offen
	v_accvgpr_read_b32 v0, a24
	v_accvgpr_read_b32 v1, a25
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a26
	v_accvgpr_read_b32 v4, a27
	v_cvt_pk_bf16_f32 v1, v1, v4
	buffer_store_dword v5, v20, s[0:3], s8 offen
	v_mov_b32_dpp v4, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v5, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v0, v0, v5, s[4:5]
	v_cndmask_b32_e64 v1, v4, v1, s[4:5]
	v_lshlrev_b32_e32 v4, 16, v1
	v_lshrrev_b32_e32 v5, 16, v0
	s_mov_b32 s9, 0xffff
	v_and_or_b32 v4, s9, v0, v4
	s_mov_b32 s10, 0xffff0000
	v_and_or_b32 v5, s10, v1, v5
	buffer_store_dword v10, v8, s[0:3], s8 offen
	buffer_store_dword v16, v9, s[0:3], s8 offen
	v_accvgpr_read_b32 v0, a36
	v_accvgpr_read_b32 v1, a37
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a38
	v_accvgpr_read_b32 v10, a39
	v_cvt_pk_bf16_f32 v1, v1, v10
	v_mov_b32_dpp v10, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v16, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v0, v0, v16, s[4:5]
	v_cndmask_b32_e64 v1, v10, v1, s[4:5]
	v_lshlrev_b32_e32 v10, 16, v1
	v_lshrrev_b32_e32 v16, 16, v0
	v_and_or_b32 v10, s9, v0, v10
	v_and_or_b32 v16, s10, v1, v16
	buffer_store_dword v2, v21, s[0:3], s8 offen
	buffer_store_dword v3, v22, s[0:3], s8 offen
	v_accvgpr_read_b32 v0, a88
	v_accvgpr_read_b32 v1, a89
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a90
	v_accvgpr_read_b32 v2, a91
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_mov_b32_dpp v2, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v3, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v0, v0, v3, s[4:5]
	v_cndmask_b32_e64 v1, v2, v1, s[4:5]
	v_lshlrev_b32_e32 v2, 16, v1
	v_lshrrev_b32_e32 v3, 16, v0
	v_and_or_b32 v28, s9, v0, v2
	v_and_or_b32 v29, s10, v1, v3
	buffer_store_dword v7, v23, s[0:3], s8 offen
	v_accvgpr_read_b32 v0, a100
	v_accvgpr_read_b32 v1, a101
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a102
	v_accvgpr_read_b32 v2, a103
	v_cvt_pk_bf16_f32 v1, v1, v2
	buffer_store_dword v17, v24, s[0:3], s8 offen
	v_mov_b32_dpp v2, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v3, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v0, v0, v3, s[4:5]
	v_cndmask_b32_e64 v1, v2, v1, s[4:5]
	v_lshlrev_b32_e32 v2, 16, v1
	v_lshrrev_b32_e32 v3, 16, v0
	s_mov_b32 s9, 0xffff
	v_and_or_b32 v7, s9, v0, v2
	s_mov_b32 s10, 0xffff0000
	v_and_or_b32 v17, s10, v1, v3
	buffer_store_dword v18, v25, s[0:3], s8 offen
	buffer_store_dword v27, v26, s[0:3], s8 offen
	v_accvgpr_read_b32 v0, a152
	v_accvgpr_read_b32 v1, a153
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a154
	v_accvgpr_read_b32 v2, a155
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_accvgpr_read_b32 v2, a164
	v_accvgpr_read_b32 v3, a165
	v_cvt_pk_bf16_f32 v2, v2, v3
	v_accvgpr_read_b32 v3, a166
	v_accvgpr_read_b32 v18, a167
	v_cvt_pk_bf16_f32 v3, v3, v18
	v_mov_b32_dpp v18, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v27, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v0, v0, v27, s[4:5]
	v_cndmask_b32_e64 v1, v18, v1, s[4:5]
	v_lshlrev_b32_e32 v18, 16, v1
	v_lshrrev_b32_e32 v27, 16, v0
	v_and_or_b32 v18, s9, v0, v18
	v_and_or_b32 v27, s10, v1, v27
	v_mov_b32_dpp v30, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	buffer_store_dword v4, v6, s[0:3], s7 offen
	v_mov_b32_dpp v0, v3 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v2, v2, v0, s[4:5]
	buffer_store_dword v5, v13, s[0:3], s7 offen
	v_accvgpr_read_b32 v0, a216
	v_accvgpr_read_b32 v1, a217
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a218
	v_accvgpr_read_b32 v4, a219
	v_cvt_pk_bf16_f32 v1, v1, v4
	v_cndmask_b32_e64 v3, v30, v3, s[4:5]
	v_lshlrev_b32_e32 v4, 16, v3
	v_lshrrev_b32_e32 v5, 16, v2
	v_and_or_b32 v2, s9, v2, v4
	v_and_or_b32 v3, s10, v3, v5
	v_mov_b32_dpp v4, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	buffer_store_dword v10, v14, s[0:3], s7 offen
	v_mov_b32_dpp v5, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v0, v0, v5, s[4:5]
	buffer_store_dword v16, v15, s[0:3], s7 offen
	v_cndmask_b32_e64 v1, v4, v1, s[4:5]
	v_lshlrev_b32_e32 v4, 16, v1
	v_lshrrev_b32_e32 v5, 16, v0
	s_mov_b32 s8, 0xffff
	v_and_or_b32 v4, s8, v0, v4
	s_mov_b32 s9, 0xffff0000
	v_and_or_b32 v5, s9, v1, v5
	v_accvgpr_read_b32 v0, a228
	v_accvgpr_read_b32 v1, a229
	v_cvt_pk_bf16_f32 v0, v0, v1
	buffer_store_dword v28, v11, s[0:3], s7 offen
	v_accvgpr_read_b32 v1, a230
	v_accvgpr_read_b32 v10, a231
	v_cvt_pk_bf16_f32 v1, v1, v10
	v_mov_b32_dpp v10, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	buffer_store_dword v29, v12, s[0:3], s7 offen
	v_mov_b32_dpp v16, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v0, v0, v16, s[4:5]
	v_cndmask_b32_e64 v1, v10, v1, s[4:5]
	s_add_i32 s10, s61, 0x70
	v_lshlrev_b32_e32 v10, 16, v1
	v_lshrrev_b32_e32 v16, 16, v0
	s_lshl_b32 s10, s10, 13
	v_and_or_b32 v10, s8, v0, v10
	v_and_or_b32 v16, s9, v1, v16
	s_add_i32 s6, s6, s10
	buffer_store_dword v7, v19, s[0:3], s7 offen
	v_accvgpr_read_b32 v0, a28
	v_accvgpr_read_b32 v1, a29
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a30
	v_accvgpr_read_b32 v7, a31
	v_cvt_pk_bf16_f32 v1, v1, v7
	buffer_store_dword v17, v20, s[0:3], s7 offen
	v_mov_b32_dpp v7, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v17, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v0, v0, v17, s[4:5]
	v_cndmask_b32_e64 v1, v7, v1, s[4:5]
	v_lshlrev_b32_e32 v7, 16, v1
	v_lshrrev_b32_e32 v17, 16, v0
	s_mov_b32 s8, 0xffff
	v_and_or_b32 v7, s8, v0, v7
	s_mov_b32 s9, 0xffff0000
	v_and_or_b32 v17, s9, v1, v17
	buffer_store_dword v18, v8, s[0:3], s7 offen
	buffer_store_dword v27, v9, s[0:3], s7 offen
	v_accvgpr_read_b32 v0, a32
	v_accvgpr_read_b32 v1, a33
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a34
	v_accvgpr_read_b32 v18, a35
	v_cvt_pk_bf16_f32 v1, v1, v18
	v_mov_b32_dpp v18, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v27, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v0, v0, v27, s[4:5]
	v_cndmask_b32_e64 v1, v18, v1, s[4:5]
	v_lshlrev_b32_e32 v18, 16, v1
	v_lshrrev_b32_e32 v27, 16, v0
	v_and_or_b32 v18, s8, v0, v18
	v_and_or_b32 v27, s9, v1, v27
	buffer_store_dword v2, v21, s[0:3], s7 offen
	buffer_store_dword v3, v22, s[0:3], s7 offen
	v_accvgpr_read_b32 v0, a92
	v_accvgpr_read_b32 v1, a93
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a94
	v_accvgpr_read_b32 v2, a95
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_mov_b32_dpp v2, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v3, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v0, v0, v3, s[4:5]
	v_cndmask_b32_e64 v1, v2, v1, s[4:5]
	v_lshlrev_b32_e32 v2, 16, v1
	v_lshrrev_b32_e32 v3, 16, v0
	v_and_or_b32 v28, s8, v0, v2
	v_and_or_b32 v29, s9, v1, v3
	buffer_store_dword v4, v23, s[0:3], s7 offen
	v_accvgpr_read_b32 v0, a96
	v_accvgpr_read_b32 v1, a97
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a98
	v_accvgpr_read_b32 v2, a99
	v_cvt_pk_bf16_f32 v1, v1, v2
	buffer_store_dword v5, v24, s[0:3], s7 offen
	v_mov_b32_dpp v2, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v3, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v0, v0, v3, s[4:5]
	v_cndmask_b32_e64 v1, v2, v1, s[4:5]
	v_lshlrev_b32_e32 v2, 16, v1
	v_lshrrev_b32_e32 v3, 16, v0
	s_mov_b32 s8, 0xffff
	v_and_or_b32 v4, s8, v0, v2
	s_mov_b32 s9, 0xffff0000
	v_and_or_b32 v5, s9, v1, v3
	buffer_store_dword v10, v25, s[0:3], s7 offen
	buffer_store_dword v16, v26, s[0:3], s7 offen
	v_accvgpr_read_b32 v0, a156
	v_accvgpr_read_b32 v1, a157
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a158
	v_accvgpr_read_b32 v2, a159
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_accvgpr_read_b32 v2, a160
	v_accvgpr_read_b32 v3, a161
	v_cvt_pk_bf16_f32 v2, v2, v3
	v_accvgpr_read_b32 v3, a162
	v_accvgpr_read_b32 v10, a163
	v_cvt_pk_bf16_f32 v3, v3, v10
	v_mov_b32_dpp v10, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v16, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v0, v0, v16, s[4:5]
	v_cndmask_b32_e64 v1, v10, v1, s[4:5]
	v_lshlrev_b32_e32 v10, 16, v1
	v_lshrrev_b32_e32 v16, 16, v0
	v_and_or_b32 v10, s8, v0, v10
	v_and_or_b32 v16, s9, v1, v16
	v_mov_b32_dpp v30, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	buffer_store_dword v7, v6, s[0:3], s6 offen
	v_mov_b32_dpp v0, v3 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v2, v2, v0, s[4:5]
	buffer_store_dword v17, v13, s[0:3], s6 offen
	v_accvgpr_read_b32 v0, a220
	v_accvgpr_read_b32 v1, a221
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a222
	v_accvgpr_read_b32 v6, a223
	v_cvt_pk_bf16_f32 v1, v1, v6
	v_cndmask_b32_e64 v3, v30, v3, s[4:5]
	v_lshlrev_b32_e32 v6, 16, v3
	v_lshrrev_b32_e32 v7, 16, v2
	v_and_or_b32 v2, s8, v2, v6
	v_and_or_b32 v3, s9, v3, v7
	v_mov_b32_dpp v6, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	buffer_store_dword v18, v14, s[0:3], s6 offen
	v_mov_b32_dpp v7, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v0, v0, v7, s[4:5]
	buffer_store_dword v27, v15, s[0:3], s6 offen
	v_cndmask_b32_e64 v1, v6, v1, s[4:5]
	v_lshlrev_b32_e32 v6, 16, v1
	v_lshrrev_b32_e32 v7, 16, v0
	s_mov_b32 s7, 0xffff
	v_and_or_b32 v6, s7, v0, v6
	s_mov_b32 s8, 0xffff0000
	v_and_or_b32 v7, s8, v1, v7
	v_accvgpr_read_b32 v0, a224
	v_accvgpr_read_b32 v1, a225
	v_cvt_pk_bf16_f32 v0, v0, v1
	buffer_store_dword v28, v11, s[0:3], s6 offen
	v_accvgpr_read_b32 v1, a226
	v_accvgpr_read_b32 v11, a227
	v_cvt_pk_bf16_f32 v1, v1, v11
	v_mov_b32_dpp v11, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	buffer_store_dword v29, v12, s[0:3], s6 offen
	v_mov_b32_dpp v12, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v0, v0, v12, s[4:5]
	v_cndmask_b32_e64 v1, v11, v1, s[4:5]
	v_lshlrev_b32_e32 v11, 16, v1
	v_lshrrev_b32_e32 v12, 16, v0
	v_and_or_b32 v0, s7, v0, v11
	v_and_or_b32 v1, s8, v1, v12
	buffer_store_dword v4, v19, s[0:3], s6 offen
	buffer_store_dword v5, v20, s[0:3], s6 offen
	buffer_store_dword v10, v8, s[0:3], s6 offen
	buffer_store_dword v16, v9, s[0:3], s6 offen
	buffer_store_dword v2, v21, s[0:3], s6 offen
	buffer_store_dword v3, v22, s[0:3], s6 offen
	buffer_store_dword v6, v23, s[0:3], s6 offen
	buffer_store_dword v7, v24, s[0:3], s6 offen
	buffer_store_dword v0, v25, s[0:3], s6 offen
	buffer_store_dword v1, v26, s[0:3], s6 offen
	s_endpgm
.Lrocke_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_4096x4096x65536_c0c1_Acanonical_k21f5fa872a_end:
.size rocke_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_4096x4096x65536_c0c1_Acanonical_k21f5fa872a, .Lrocke_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_4096x4096x65536_c0c1_Acanonical_k21f5fa872a_end-rocke_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_4096x4096x65536_c0c1_Acanonical_k21f5fa872a

.rodata
.p2align 6
.amdhsa_kernel rocke_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_4096x4096x65536_c0c1_Acanonical_k21f5fa872a
  .amdhsa_group_segment_fixed_size 154144
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_kernarg_size 40
  .amdhsa_user_sgpr_count 2
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_next_free_vgpr 504
  .amdhsa_next_free_sgpr 100
  .amdhsa_reserve_vcc 1
  .amdhsa_accum_offset 248
  .amdhsa_float_denorm_mode_32 3
  .amdhsa_float_denorm_mode_16_64 3
.end_amdhsa_kernel

.amdgpu_metadata
---
custom.config:
  Source:
    Origin: rocke
    Description: MX E2M3 GEMM at 4096x4096x65536 on tiled A, canonical B and shuffled scales
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
  AssertSizeEqual: {0: 4096, 1: 4096, 2: 1, 3: 65536}
  AssertStrideAEqual: {1: 65536}
  AssertStrideBEqual: {1: 65536}
  AssertStrideDEqual: {1: 4096}
  AssertAlphaValue: 1
amdhsa.target: amdgcn-amd-amdhsa--gfx950
amdhsa.version: [ 1, 1 ]
amdhsa.kernels:
  - .name: rocke_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_4096x4096x65536_c0c1_Acanonical_k21f5fa872a
    .symbol: rocke_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_4096x4096x65536_c0c1_Acanonical_k21f5fa872a.kd
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
    .sgpr_count: 106
    .vgpr_count: 504
    .agpr_count: 256
    .wavefront_size: 64
...
.end_amdgpu_metadata
