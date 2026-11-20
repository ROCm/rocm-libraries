// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// MX E3M2 x E3M2 GEMM for gfx950: E8M0 scales per 32 K elements, fp32 accumulation,
// bf16 D rounded to nearest even. 256 threads (wave64, 2x2 waves), macrotile 256x256x128,
// v_mfma_scale_f32_16x16x128_f8f6f4 cbsz:3 blgp:3, 127520 bytes of static LDS.
//
// Fixed problem, enforced by the predicates in custom.config:
//   M = 4096, N = 4096, K = 6144, batch 1, TransA = T, TransB = N, lda = ldb = K, D column-major
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
.globl rocke_gemm_mx_e3m2_e3m2_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s2_4096x4096x6144_c0c1_Acanonical_kb952f38bb6
.p2align 8
.type rocke_gemm_mx_e3m2_e3m2_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s2_4096x4096x6144_c0c1_Acanonical_kb952f38bb6,@function
rocke_gemm_mx_e3m2_e3m2_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s2_4096x4096x6144_c0c1_Acanonical_kb952f38bb6:
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
	v_readfirstlane_b32 s56, v0
	v_mbcnt_lo_u32_b32 v0, -1, 0
	v_mbcnt_hi_u32_b32 v6, -1, v0
	s_lshr_b32 s0, s56, 1
	s_and_b32 s1, s56, 1
	s_and_b32 s13, s12, 7
	s_lshr_b32 s12, s12, 3
	s_lshr_b32 s14, s13, 1
	s_lshl_b32 s14, s14, 2
	s_lshr_b32 s15, s12, 3
	s_add_i32 s14, s14, s15
	s_and_b32 s13, s13, 1
	s_lshl_b32 s13, s13, 3
	s_and_b32 s12, s12, 7
	s_add_i32 s15, s13, s12
	s_lshl_b32 s14, s14, 8
	s_mul_i32 s12, s14, 0x1200
	s_lshl_b32 s16, s15, 8
	s_lshl_b32 s17, s0, 7
	s_lshl_b32 s18, s1, 7
	s_add_u32 s22, s2, s12
	s_mul_i32 s57, s56, 0x1200
	s_mul_i32 s24, s56, 0x810
	s_addc_u32 s23, s3, 0
	s_lshl_b32 s19, s17, 3
	s_lshl_b32 s0, s56, 6
	v_and_b32_e32 v212, 15, v6
	v_lshrrev_b32_e32 v213, 4, v6
	v_and_b32_e32 v0, 3, v212
	v_add_u32_e32 v1, s0, v6
	v_lshrrev_b32_e32 v2, 2, v212
	v_lshlrev_b32_e32 v0, 5, v0
	v_mul_u32_u24_e32 v3, 24, v213
	v_and_b32_e32 v4, 15, v1
	v_lshrrev_b32_e32 v1, 4, v1
	v_cmp_ge_i32_e64 s[0:1], v213, 1
	v_cmp_ge_i32_e64 s[2:3], v213, 2
	v_cmp_ge_i32_e64 s[12:13], v213, 3
	s_mov_b32 s20, 0x808
	v_mad_u32_u24 v0, s20, v2, v0
	v_lshlrev_b32_e32 v1, 7, v1
	v_lshrrev_b32_e32 v2, 2, v4
	s_add_i32 s58, s57, 0x4800
	v_and_b32_e32 v5, 7, v6
	v_add3_u32 v0, s19, v3, v0
	v_mad_u32_u24 v2, s20, v2, v1
	v_and_b32_e32 v7, 3, v4
	v_lshrrev_b32_e32 v8, 3, v6
	v_lshlrev_b32_e32 v5, 4, v5
	v_add_u32_e32 v9, 0x101a0, v0
	v_add_u32_e32 v0, 0x80a0, v0
	s_mov_b32 s20, 0x810
	v_mad_u32_u24 v1, s20, v4, v1
	v_lshl_add_u32 v2, v7, 5, v2
	s_add_i32 s25, s24, 0x2040
	s_add_i32 s59, s57, 0x9000
	s_mov_b32 s20, 0x12000
	v_mad_u32_u24 v214, s20, v8, v5
	v_add_u32_e32 v1, 0x60, v1
	v_add_u32_e32 v2, 0x10200, v2
	s_add_i32 s26, s24, 0x4080
	s_add_i32 s60, s57, 0xd800
	s_add_i32 s27, s24, 0x60c0
	s_add_i32 s61, s57, 0x90000
	s_add_i32 s28, s24, 0x400
	s_add_i32 s20, s56, 20
	s_add_i32 s21, s56, 4
	s_lshr_b32 s20, s20, 4
	s_lshl_b32 s20, s20, 3
	s_lshl_b32 s29, s20, 4
	s_add_i32 s29, s21, s29
	s_mul_i32 s21, s21, 0x810
	s_lshl_b32 s20, s20, 7
	s_add_i32 s30, s21, s20
	s_add_i32 s20, s56, 24
	s_add_i32 s21, s56, 8
	s_lshr_b32 s20, s20, 4
	s_lshl_b32 s20, s20, 3
	s_lshl_b32 s31, s20, 4
	s_add_i32 s31, s21, s31
	s_mul_i32 s21, s21, 0x810
	s_lshl_b32 s20, s20, 7
	s_add_i32 s32, s21, s20
	s_add_i32 s20, s56, 28
	s_add_i32 s21, s56, 12
	s_lshr_b32 s20, s20, 4
	s_lshl_b32 s20, s20, 3
	s_lshl_b32 s33, s20, 4
	s_add_i32 s33, s21, s33
	s_mul_i32 s21, s21, 0x810
	s_lshl_b32 s20, s20, 7
	s_mov_b32 s34, 0x810
	v_mad_u32_u24 v3, s34, v212, v3
	s_add_i32 s34, s21, s20
	v_add_u32_e32 v3, s19, v3
	v_add_u32_e32 v4, 0x8080, v3
	v_add_u32_e32 v5, 0xffff7e80, v3
	v_cndmask_b32_e64 v7, v3, v4, s[2:3]
	v_cndmask_b32_e64 v8, v3, v5, s[2:3]
	v_cndmask_b32_e64 v10, v3, v4, s[12:13]
	v_cndmask_b32_e64 v11, v3, v5, s[12:13]
	v_cndmask_b32_e64 v12, v9, v4, s[0:1]
	v_cndmask_b32_e64 v13, v0, v5, s[0:1]
	v_cndmask_b32_e64 v4, v9, v4, s[2:3]
	v_cndmask_b32_e64 v0, v0, v5, s[2:3]
	v_lshrrev_b32_e32 v94, 2, v1
	v_lshrrev_b32_e32 v104, 2, v2
	s_mul_i32 s0, s15, 0x120000
	s_add_u32 s20, s4, s0
	s_addc_u32 s21, s5, 0
	s_lshr_b32 s35, s18, 4
	s_and_b32 s1, s21, 0xffff
	s_add_i32 s62, s14, s17
	s_add_i32 s63, s16, s18
	s_and_b32 s5, s7, 0xffff
	s_and_b32 s13, s9, 0xffff
	s_mul_i32 s64, s29, 0x1200
	s_lshl_b32 s65, s56, 8
	s_lshr_b32 s66, s14, 7
	s_lshr_b32 s29, s16, 7
	s_lshr_b32 s36, s17, 7
	s_lshr_b32 s37, s18, 7
	s_add_i32 s67, 0xd000, s24
	s_mul_i32 s68, s31, 0x1200
	s_and_b32 s17, s23, 0xffff
	s_mul_i32 s69, s33, 0x1200
	v_lshrrev_b32_e32 v1, 2, v3
	v_lshrrev_b32_e32 v172, 2, v7
	v_lshrrev_b32_e32 v173, 2, v8
	v_lshrrev_b32_e32 v174, 2, v10
	v_lshrrev_b32_e32 v175, 2, v11
	v_lshrrev_b32_e32 v105, 2, v12
	v_lshrrev_b32_e32 v176, 2, v13
	v_lshrrev_b32_e32 v106, 2, v4
	v_lshrrev_b32_e32 v177, 2, v0
	v_lshlrev_b32_e32 v215, 4, v6
	s_mul_i32 s70, s56, 0x1800
	s_mov_b32 s0, s20
	s_mov_b32 s2, 0x7ffffff0
	s_mov_b32 s3, 0x31014000
	s_mov_b32 s4, s6
	s_mov_b32 s6, 0x7ffffff0
	s_mov_b32 s7, 0x31014000
	s_mov_b32 s12, s8
	s_mov_b32 s14, 0x7ffffff0
	s_mov_b32 s15, 0x31014000
	v_lshlrev_b32_e32 v216, 2, v6
	s_mov_b32 s16, s22
	s_mov_b32 s18, 0x7ffffff0
	s_mov_b32 m0, s67
	s_mov_b32 s19, 0x31014000
	buffer_load_dwordx4 v214, s[16:19], s57 offen lds
	s_add_i32 s71, 0xd000, s25
	s_mov_b32 m0, s71
	s_nop 0
	buffer_load_dwordx4 v214, s[16:19], s58 offen lds
	s_add_i32 s72, 0xd000, s26
	s_mov_b32 m0, s72
	s_nop 0
	buffer_load_dwordx4 v214, s[16:19], s59 offen lds
	s_add_i32 s73, 0xd000, s27
	s_mov_b32 m0, s73
	s_nop 0
	buffer_load_dwordx4 v214, s[16:19], s60 offen lds
	s_add_i32 s74, 0xd000, s28
	s_mov_b32 m0, s74
	s_nop 0
	buffer_load_dwordx4 v214, s[16:19], s61 offen lds
	s_add_i32 s75, 0xd000, s30
	s_mov_b32 m0, s75
	s_nop 0
	buffer_load_dwordx4 v214, s[16:19], s64 offen lds
	s_add_i32 s76, 0xd000, s32
	s_mov_b32 m0, s76
	s_nop 0
	buffer_load_dwordx4 v214, s[16:19], s68 offen lds
	s_add_i32 s77, 0xd000, s34
	s_mov_b32 m0, s77
	s_nop 0
	buffer_load_dwordx4 v214, s[16:19], s69 offen lds
	s_lshl_b32 s31, s66, 9
	s_lshl_b32 s33, s29, 9
	s_add_i32 s78, 0x1000, s70
	s_mov_b32 m0, s78
	s_nop 0
	buffer_load_dwordx4 v215, s[0:3], s70 offen lds
	buffer_load_dwordx4 v215, s[0:3], s70 offen offset:1024 lds
	buffer_load_dwordx4 v215, s[0:3], s70 offen offset:2048 lds
	buffer_load_dwordx4 v215, s[0:3], s70 offen offset:3072 lds
	s_add_i32 s38, s70, 0x1000
	s_add_i32 s79, 0x1000, s38
	s_mov_b32 m0, s79
	s_nop 0
	buffer_load_dwordx4 v215, s[0:3], s38 offen lds
	buffer_load_dwordx4 v215, s[0:3], s38 offen offset:1024 lds
	s_mov_b32 m0, s65
	s_add_i32 s31, s65, s31
	buffer_load_dword v216, s[4:7], s31 offen lds
	s_add_i32 s31, 0x400, s65
	s_mov_b32 m0, s31
	s_add_i32 s80, s65, s33
	buffer_load_dword v216, s[12:15], s80 offen lds
	s_add_i32 s31, s57, 0x80
	s_add_i32 s24, s24, 0x8100
	s_add_i32 s81, 0xd000, s24
	s_mov_b32 m0, s81
	s_nop 0
	buffer_load_dwordx4 v214, s[16:19], s31 offen lds
	s_add_i32 s24, s58, 0x80
	s_add_i32 s25, s25, 0x8100
	s_add_i32 s82, 0xd000, s25
	s_mov_b32 m0, s82
	s_nop 0
	buffer_load_dwordx4 v214, s[16:19], s24 offen lds
	s_add_i32 s24, s59, 0x80
	s_add_i32 s25, s26, 0x8100
	s_add_i32 s83, 0xd000, s25
	s_mov_b32 m0, s83
	s_nop 0
	buffer_load_dwordx4 v214, s[16:19], s24 offen lds
	s_add_i32 s24, s60, 0x80
	s_add_i32 s25, s27, 0x8100
	s_add_i32 s84, 0xd000, s25
	s_mov_b32 m0, s84
	s_nop 0
	buffer_load_dwordx4 v214, s[16:19], s24 offen lds
	s_add_i32 s24, s61, 0x80
	s_add_i32 s25, s28, 0x8100
	s_add_i32 s85, 0xd000, s25
	s_mov_b32 m0, s85
	s_nop 0
	buffer_load_dwordx4 v214, s[16:19], s24 offen lds
	s_add_i32 s24, s64, 0x80
	s_add_i32 s25, s30, 0x8100
	s_add_i32 s86, 0xd000, s25
	s_mov_b32 m0, s86
	s_nop 0
	buffer_load_dwordx4 v214, s[16:19], s24 offen lds
	s_add_i32 s24, s68, 0x80
	s_add_i32 s25, s32, 0x8100
	s_add_i32 s87, 0xd000, s25
	s_mov_b32 m0, s87
	s_nop 0
	buffer_load_dwordx4 v214, s[16:19], s24 offen lds
	s_add_i32 s24, s69, 0x80
	s_add_i32 s25, s34, 0x8100
	s_add_i32 s88, 0xd000, s25
	s_mov_b32 m0, s88
	s_nop 0
	buffer_load_dwordx4 v214, s[16:19], s24 offen lds
	s_add_i32 s24, s66, 32
	s_lshl_b32 s24, s24, 9
	s_add_i32 s25, s70, 0x6000
	s_add_i32 s89, 0x7000, s70
	s_mov_b32 m0, s89
	s_nop 0
	buffer_load_dwordx4 v215, s[0:3], s25 offen lds
	buffer_load_dwordx4 v215, s[0:3], s25 offen offset:1024 lds
	buffer_load_dwordx4 v215, s[0:3], s25 offen offset:2048 lds
	buffer_load_dwordx4 v215, s[0:3], s25 offen offset:3072 lds
	s_add_i32 s90, 0x7000, s38
	s_mov_b32 m0, s90
	s_add_i32 s25, s70, 0x7000
	buffer_load_dwordx4 v215, s[0:3], s25 offen lds
	buffer_load_dwordx4 v215, s[0:3], s25 offen offset:1024 lds
	s_add_i32 s91, 0x800, s65
	s_mov_b32 m0, s91
	s_add_i32 s24, s65, s24
	buffer_load_dword v216, s[4:7], s24 offen lds
	s_add_i32 s24, s80, 0x3c00
	buffer_load_dword v216, s[12:15], s24 offen offset:1024 lds
	s_waitcnt vmcnt(16) lgkmcnt(0)
	s_barrier
	s_lshl_b32 s24, s36, 7
	v_lshlrev_b32_e32 v0, 1, v6
	s_lshl_b32 s25, s37, 7
	v_add_lshl_u32 v217, s24, v0, 2
	v_add_lshl_u32 v218, s25, v0, 2
	v_lshlrev_b32_e32 v219, 2, v1
	ds_read_b64 v[168:169], v217
	ds_read_b64 v[170:171], v218 offset:1024
	ds_read_b64 v[0:1], v219 offset:53248
	ds_read_b64 v[2:3], v219 offset:53256
	ds_read_b64 v[4:5], v219 offset:53264
	s_lshl_b32 s24, s35, 6
	v_add_u32_e32 v10, s24, v6
	v_lshlrev_b32_e32 v220, 4, v10
	ds_read_b128 v[6:9], v220 offset:4096
	v_lshlrev_b32_e32 v221, 3, v10
	ds_read_b64 v[10:11], v221 offset:20480
	ds_read_b64 v[12:13], v219 offset:53376
	ds_read_b64 v[14:15], v219 offset:53384
	ds_read_b64 v[16:17], v219 offset:53392
	ds_read_b64 v[18:19], v219 offset:53504
	ds_read_b64 v[20:21], v219 offset:53512
	ds_read_b64 v[22:23], v219 offset:53520
	ds_read_b64 v[24:25], v219 offset:53632
	ds_read_b64 v[26:27], v219 offset:53640
	ds_read_b64 v[28:29], v219 offset:53648
	ds_read_b64 v[30:31], v219 offset:53760
	ds_read_b64 v[32:33], v219 offset:53768
	ds_read_b64 v[34:35], v219 offset:53776
	ds_read_b64 v[36:37], v219 offset:53888
	ds_read_b64 v[38:39], v219 offset:53896
	ds_read_b64 v[40:41], v219 offset:53904
	ds_read_b64 v[42:43], v219 offset:54016
	ds_read_b64 v[44:45], v219 offset:54024
	ds_read_b64 v[46:47], v219 offset:54032
	ds_read_b64 v[48:49], v219 offset:54144
	ds_read_b64 v[50:51], v219 offset:54152
	ds_read_b64 v[52:53], v219 offset:54160
	ds_read_b128 v[54:57], v220 offset:5120
	ds_read_b64 v[58:59], v221 offset:20992
	ds_read_b128 v[60:63], v220 offset:6144
	ds_read_b64 v[64:65], v221 offset:21504
	ds_read_b128 v[66:69], v220 offset:7168
	ds_read_b64 v[70:71], v221 offset:22016
	ds_read_b128 v[72:75], v220 offset:8192
	ds_read_b64 v[76:77], v221 offset:22528
	ds_read_b128 v[78:81], v220 offset:9216
	ds_read_b64 v[82:83], v221 offset:23040
	ds_read_b128 v[84:87], v220 offset:10240
	ds_read_b64 v[88:89], v221 offset:23552
	ds_read_b128 v[90:93], v220 offset:11264
	v_lshlrev_b32_e32 v222, 2, v94
	ds_read_b64 v[94:95], v221 offset:24064
	ds_read_b128 v[96:99], v222 offset:53248
	ds_read_b128 v[100:103], v222 offset:53264
	v_lshlrev_b32_e32 v223, 2, v104
	s_waitcnt lgkmcnt(1)
	ds_write_b64 v223, v[96:97] offset:53248
	ds_write_b64 v223, v[98:99] offset:53256
	s_waitcnt lgkmcnt(2)
	ds_write_b64 v223, v[100:101] offset:53264
	ds_write_b64 v223, v[102:103] offset:53272
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_mov_b32 m0, s67
	s_add_i32 s24, s57, 0x100
	buffer_load_dwordx4 v214, s[16:19], s24 offen lds
	s_mov_b32 m0, s71
	s_add_i32 s24, s58, 0x100
	buffer_load_dwordx4 v214, s[16:19], s24 offen lds
	s_mov_b32 m0, s72
	s_add_i32 s24, s59, 0x100
	buffer_load_dwordx4 v214, s[16:19], s24 offen lds
	s_mov_b32 m0, s73
	s_add_i32 s24, s60, 0x100
	buffer_load_dwordx4 v214, s[16:19], s24 offen lds
	s_mov_b32 m0, s74
	s_add_i32 s24, s61, 0x100
	buffer_load_dwordx4 v214, s[16:19], s24 offen lds
	s_mov_b32 m0, s75
	s_add_i32 s24, s64, 0x100
	buffer_load_dwordx4 v214, s[16:19], s24 offen lds
	s_mov_b32 m0, s76
	s_add_i32 s24, s68, 0x100
	buffer_load_dwordx4 v214, s[16:19], s24 offen lds
	s_mov_b32 m0, s77
	s_add_i32 s24, s69, 0x100
	buffer_load_dwordx4 v214, s[16:19], s24 offen lds
	s_add_i32 s24, s66, 64
	s_lshl_b32 s24, s24, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[0:5], v[6:11], 0, v168, v170 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(8) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[204:205], v217 offset:2048
	ds_read_b64 v[206:207], v218 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[12:17], v[6:11], 0, v168, v170 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_i32 s25, s70, 0xc000
	s_mov_b32 m0, s78
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:23], v[6:11], 0, v168, v170 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s25 offen lds
	v_lshlrev_b32_e32 v224, 2, v106
	ds_read_b64 v[96:97], v224 offset:53344
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[24:29], v[6:11], 0, v168, v170 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_lshlrev_b32_e32 v225, 2, v105
	ds_read_b64 v[98:99], v225 offset:53352
	ds_read_b64 v[100:101], v225 offset:53360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[30:35], v[6:11], 0, v169, v170 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[102:103], v224 offset:53472
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[36:41], v[6:11], 0, v169, v170 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[104:105], v225 offset:53480
	ds_read_b64 v[106:107], v225 offset:53488
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[42:47], v[6:11], 0, v169, v170 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s25 offen offset:1024 lds
	ds_read_b64 v[108:109], v224 offset:53600
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[48:53], v[6:11], 0, v169, v170 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[110:111], v225 offset:53608
	ds_read_b64 v[112:113], v225 offset:53616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[48:53], v[54:59], 0, v169, v170 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[6:7], v224 offset:53728
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[42:47], v[54:59], 0, v169, v170 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[8:9], v225 offset:53736
	ds_read_b64 v[10:11], v225 offset:53744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[36:41], v[54:59], 0, v169, v170 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s25 offen offset:2048 lds
	ds_read_b64 v[114:115], v224 offset:53856
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[30:35], v[54:59], 0, v169, v170 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[116:117], v225 offset:53864
	ds_read_b64 v[118:119], v225 offset:53872
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[24:29], v[54:59], 0, v168, v170 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[120:121], v224 offset:53984
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[18:23], v[54:59], 0, v168, v170 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[122:123], v225 offset:53992
	ds_read_b64 v[124:125], v225 offset:54000
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[12:17], v[54:59], 0, v168, v170 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s25 offen offset:3072 lds
	ds_read_b64 v[126:127], v224 offset:54112
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[0:5], v[54:59], 0, v168, v170 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[128:129], v225 offset:54120
	ds_read_b64 v[130:131], v225 offset:54128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[0:5], v[60:65], 0, v168, v170 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[54:55], v224 offset:54240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[12:17], v[60:65], 0, v168, v170 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[56:57], v225 offset:54248
	ds_read_b64 v[58:59], v225 offset:54256
	s_add_i32 s25, s70, 0xd000
	s_mov_b32 m0, s79
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[18:23], v[60:65], 0, v168, v170 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s25 offen lds
	ds_read_b128 v[132:135], v220 offset:28672
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[24:29], v[60:65], 0, v168, v170 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[136:137], v221 offset:45056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[30:35], v[60:65], 0, v169, v170 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[138:141], v220 offset:29696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[36:41], v[60:65], 0, v169, v170 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[142:143], v221 offset:45568
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[42:47], v[60:65], 0, v169, v170 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s25 offen offset:1024 lds
	ds_read_b128 v[144:147], v220 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[48:53], v[60:65], 0, v169, v170 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[148:149], v221 offset:46080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[48:53], v[66:71], 0, v169, v170 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[60:63], v220 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[42:47], v[66:71], 0, v169, v170 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[64:65], v221 offset:46592
	s_add_i32 s24, s65, s24
	s_mov_b32 m0, s65
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[36:41], v[66:71], 0, v169, v170 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[4:7], s24 offen lds
	ds_read_b128 v[150:153], v220 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[30:35], v[66:71], 0, v169, v170 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[154:155], v221 offset:47104
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[24:29], v[66:71], 0, v168, v170 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[156:159], v220 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[18:23], v[66:71], 0, v168, v170 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[160:161], v221 offset:47616
	s_add_i32 s24, s80, 0x7c00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[12:17], v[66:71], 0, v168, v170 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[12:15], s24 offen offset:1024 lds
	ds_read_b128 v[162:165], v220 offset:34816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[0:5], v[66:71], 0, v168, v170 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[166:167], v221 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[0:5], v[72:77], 0, v168, v171 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[66:69], v220 offset:35840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[12:17], v[72:77], 0, v168, v171 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[70:71], v221 offset:48640
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[18:23], v[72:77], 0, v168, v171 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[24:29], v[72:77], 0, v168, v171 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[30:35], v[72:77], 0, v169, v171 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[36:41], v[72:77], 0, v169, v171 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[42:47], v[72:77], 0, v169, v171 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[48:53], v[72:77], 0, v169, v171 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[48:53], v[78:83], 0, v169, v171 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[42:47], v[78:83], 0, v169, v171 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[36:41], v[78:83], 0, v169, v171 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[30:35], v[78:83], 0, v169, v171 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[24:29], v[78:83], 0, v168, v171 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[18:23], v[78:83], 0, v168, v171 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[12:17], v[78:83], 0, v168, v171 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[0:5], v[78:83], 0, v168, v171 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[0:5], v[84:89], 0, v168, v171 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[12:17], v[84:89], 0, v168, v171 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[18:23], v[84:89], 0, v168, v171 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[24:29], v[84:89], 0, v168, v171 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[30:35], v[84:89], 0, v169, v171 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[36:41], v[84:89], 0, v169, v171 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[42:47], v[84:89], 0, v169, v171 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[48:53], v[84:89], 0, v169, v171 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[48:53], v[90:95], 0, v169, v171 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[42:47], v[90:95], 0, v169, v171 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[36:41], v[90:95], 0, v169, v171 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[30:35], v[90:95], 0, v169, v171 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[24:29], v[90:95], 0, v168, v171 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[18:23], v[90:95], 0, v168, v171 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[12:17], v[90:95], 0, v168, v171 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[0:5], v[90:95], 0, v168, v171 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_add_u32 s20, s20, 0x1000
	s_addc_u32 s21, s21, 0
	s_and_b32 s21, s21, 0xffff
	s_mul_i32 s24, s66, 0xfffffe00
	s_lshl_b32 s25, s29, 9
	s_add_i32 s24, s24, s25
	s_add_i32 s24, s24, 0xfffffc00
	s_ashr_i32 s25, s24, 31
	s_add_u32 s24, s8, s24
	s_addc_u32 s25, s9, s25
	s_mul_i32 s8, s57, -1
	s_and_b32 s25, s25, 0xffff
	s_add_i32 s9, s58, s8
	s_add_u32 s28, s22, s9
	s_addc_u32 s29, s23, 0
	s_and_b32 s29, s29, 0xffff
	s_add_i32 s9, s59, s8
	s_add_u32 s32, s22, s9
	s_addc_u32 s33, s23, 0
	s_and_b32 s33, s33, 0xffff
	s_add_i32 s9, s60, s8
	s_add_u32 s36, s22, s9
	s_addc_u32 s37, s23, 0
	s_and_b32 s37, s37, 0xffff
	s_add_i32 s9, s61, s8
	s_add_u32 s40, s22, s9
	s_addc_u32 s41, s23, 0
	s_and_b32 s41, s41, 0xffff
	s_add_i32 s9, s64, s8
	s_add_u32 s44, s22, s9
	s_addc_u32 s45, s23, 0
	s_and_b32 s45, s45, 0xffff
	s_add_i32 s9, s68, s8
	s_add_u32 s48, s22, s9
	s_addc_u32 s49, s23, 0
	s_and_b32 s49, s49, 0xffff
	s_add_i32 s8, s69, s8
	s_add_u32 s52, s22, s8
	s_addc_u32 s53, s23, 0
	s_and_b32 s53, s53, 0xffff
	v_lshlrev_b32_e32 v226, 2, v175
	s_mov_b32 s22, 0x7ffffff0
	v_lshlrev_b32_e32 v227, 2, v173
	s_mov_b32 s23, 0x31014000
	v_lshlrev_b32_e32 v228, 2, v177
	v_lshlrev_b32_e32 v229, 2, v176
	v_lshlrev_b32_e32 v230, 2, v174
	v_lshlrev_b32_e32 v231, 2, v172
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
	s_mov_b32 s8, 0
	v_add_u32_e32 v232, 0x10000, v226
	v_add_u32_e32 v233, 0x10000, v227
	v_add_u32_e32 v234, 0x10000, v222
	v_add_u32_e32 v235, 0x10000, v219
	v_add_u32_e32 v236, 0x10000, v228
	v_add_u32_e32 v237, 0x10000, v229
.Lmain_loop:
	s_lshl_b32 s9, s8, 4
	s_lshl_b32 s92, s9, 5
	s_add_i32 s92, s66, s92
	s_add_i32 s93, s92, 0x60
	s_lshl_b32 s93, s93, 9
	s_lshr_b32 s94, s9, 2
	s_mul_i32 s95, s9, 0x6000
	s_mul_i32 s94, s94, 3
	s_lshl_b32 s94, s94, 7
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[96:101], v[132:137], a[0:3], v204, v206 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[168:169], v217
	ds_read_b64 v[170:171], v218 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[102:107], v[132:137], a[4:7], v204, v206 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_i32 s95, s70, s95
	s_add_i32 s96, s95, 0x12000
	s_mov_b32 m0, s89
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[108:113], v[132:137], a[8:11], v204, v206 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s96 offen lds
	ds_read_b64 v[0:1], v232 offset:20800
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[6:11], v[132:137], a[12:15], v204, v206 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[2:3], v232 offset:20808
	ds_read_b64 v[4:5], v233 offset:20816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[114:119], v[132:137], a[16:19], v205, v206 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[12:13], v232 offset:20928
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[120:125], v[132:137], a[20:23], v205, v206 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[14:15], v232 offset:20936
	ds_read_b64 v[16:17], v233 offset:20944
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[126:131], v[132:137], a[24:27], v205, v206 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s96 offen offset:1024 lds
	ds_read_b64 v[18:19], v232 offset:21056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[54:59], v[132:137], a[28:31], v205, v206 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[20:21], v232 offset:21064
	ds_read_b64 v[22:23], v233 offset:21072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[54:59], v[138:143], a[32:35], v205, v206 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[24:25], v232 offset:21184
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[126:131], v[138:143], a[36:39], v205, v206 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[26:27], v232 offset:21192
	ds_read_b64 v[28:29], v233 offset:21200
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[120:125], v[138:143], a[40:43], v205, v206 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s96 offen offset:2048 lds
	ds_read_b64 v[30:31], v232 offset:21312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[114:119], v[138:143], a[44:47], v205, v206 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[32:33], v232 offset:21320
	ds_read_b64 v[34:35], v233 offset:21328
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[6:11], v[138:143], a[48:51], v204, v206 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[36:37], v232 offset:21440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[108:113], v[138:143], a[52:55], v204, v206 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[38:39], v232 offset:21448
	ds_read_b64 v[40:41], v233 offset:21456
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[102:107], v[138:143], a[56:59], v204, v206 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s96 offen offset:3072 lds
	ds_read_b64 v[42:43], v232 offset:21568
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[96:101], v[138:143], a[60:63], v204, v206 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[44:45], v232 offset:21576
	ds_read_b64 v[46:47], v233 offset:21584
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[96:101], v[144:149], a[64:67], v204, v206 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[48:49], v232 offset:21696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[102:107], v[144:149], a[68:71], v204, v206 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[50:51], v232 offset:21704
	ds_read_b64 v[52:53], v233 offset:21712
	s_mov_b32 m0, s90
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[108:113], v[144:149], a[72:75], v204, v206 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[20:23], s96 offen lds
	ds_read_b128 v[72:75], v220 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[6:11], v[144:149], a[76:79], v204, v206 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[76:77], v221 offset:20480
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[114:119], v[144:149], a[80:83], v205, v206 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[78:81], v220 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[120:125], v[144:149], a[84:87], v205, v206 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[82:83], v221 offset:20992
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[126:131], v[144:149], a[88:91], v205, v206 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[20:23], s96 offen offset:1024 lds
	ds_read_b128 v[84:87], v220 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[54:59], v[144:149], a[92:95], v205, v206 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[88:89], v221 offset:21504
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[54:59], v[60:65], a[96:99], v205, v206 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[90:93], v220 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[126:131], v[60:65], a[100:103], v205, v206 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[94:95], v221 offset:22016
	s_add_i32 s93, s65, s93
	s_mov_b32 m0, s91
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[120:125], v[60:65], a[104:107], v205, v206 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[4:7], s93 offen lds
	ds_read_b128 v[132:135], v220 offset:8192
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[114:119], v[60:65], a[108:111], v205, v206 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[136:137], v221 offset:22528
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[6:11], v[60:65], a[112:115], v204, v206 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[138:141], v220 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[108:113], v[60:65], a[116:119], v204, v206 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[142:143], v221 offset:23040
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[102:107], v[60:65], a[120:123], v204, v206 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[24:27], s93 offen offset:1024 lds
	ds_read_b128 v[144:147], v220 offset:10240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[96:101], v[60:65], a[124:127], v204, v206 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[148:149], v221 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[96:101], v[150:155], a[128:131], v204, v207 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[60:63], v220 offset:11264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[102:107], v[150:155], a[132:135], v204, v207 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[64:65], v221 offset:24064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[108:113], v[150:155], a[136:139], v204, v207 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[6:11], v[150:155], a[140:143], v204, v207 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s93, s57, s94
	s_add_i32 s94, s93, 0x180
	s_mov_b32 m0, s81
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[114:119], v[150:155], a[144:147], v205, v207 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[120:125], v[150:155], a[148:151], v205, v207 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[126:131], v[150:155], a[152:155], v205, v207 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_mov_b32 m0, s82
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[54:59], v[150:155], a[156:159], v205, v207 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[28:31], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[54:59], v[156:161], a[160:163], v205, v207 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[126:131], v[156:161], a[164:167], v205, v207 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[120:125], v[156:161], a[168:171], v205, v207 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_mov_b32 m0, s83
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[114:119], v[156:161], a[172:175], v205, v207 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[32:35], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[6:11], v[156:161], a[176:179], v204, v207 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[108:113], v[156:161], a[180:183], v204, v207 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_mov_b32 m0, s84
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[102:107], v[156:161], a[184:187], v204, v207 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[36:39], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[96:101], v[156:161], a[188:191], v204, v207 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[96:101], v[162:167], a[192:195], v204, v207 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[102:107], v[162:167], a[196:199], v204, v207 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s85
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[108:113], v[162:167], a[200:203], v204, v207 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[40:43], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[6:11], v[162:167], a[204:207], v204, v207 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[114:119], v[162:167], a[208:211], v205, v207 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s86
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[120:125], v[162:167], a[212:215], v205, v207 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[44:47], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[126:131], v[162:167], a[216:219], v205, v207 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[54:59], v[162:167], a[220:223], v205, v207 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[54:59], v[66:71], a[224:227], v205, v207 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s87
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[126:131], v[66:71], a[228:231], v205, v207 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[48:51], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[120:125], v[66:71], a[232:235], v205, v207 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[114:119], v[66:71], a[236:239], v205, v207 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s88
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[6:11], v[66:71], a[240:243], v204, v207 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[52:55], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[108:113], v[66:71], a[244:247], v204, v207 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[102:107], v[66:71], a[248:251], v204, v207 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[96:101], v[66:71], a[252:255], v204, v207 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_add_i32 s94, s92, 0x80
	s_lshl_b32 s94, s94, 9
	s_add_i32 s96, s9, 5
	s_lshr_b32 s96, s96, 2
	s_mul_i32 s96, s96, 3
	s_lshl_b32 s96, s96, 7
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[0:5], v[72:77], a[0:3], v168, v170 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(8) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[172:173], v217 offset:2048
	ds_read_b64 v[174:175], v218 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[12:17], v[72:77], a[4:7], v168, v170 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_i32 s97, s95, 0x18000
	s_mov_b32 m0, s78
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:23], v[72:77], a[8:11], v168, v170 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s97 offen lds
	ds_read_b64 v[6:7], v219 offset:53280
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[24:29], v[72:77], a[12:15], v168, v170 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[8:9], v219 offset:53288
	ds_read_b64 v[10:11], v219 offset:53296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[30:35], v[72:77], a[16:19], v169, v170 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[54:55], v219 offset:53408
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[36:41], v[72:77], a[20:23], v169, v170 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[56:57], v219 offset:53416
	ds_read_b64 v[58:59], v219 offset:53424
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[42:47], v[72:77], a[24:27], v169, v170 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s97 offen offset:1024 lds
	ds_read_b64 v[66:67], v219 offset:53536
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[48:53], v[72:77], a[28:31], v169, v170 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[68:69], v219 offset:53544
	ds_read_b64 v[70:71], v219 offset:53552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[48:53], v[78:83], a[32:35], v169, v170 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[72:73], v219 offset:53664
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[42:47], v[78:83], a[36:39], v169, v170 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[74:75], v219 offset:53672
	ds_read_b64 v[76:77], v219 offset:53680
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[36:41], v[78:83], a[40:43], v169, v170 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s97 offen offset:2048 lds
	ds_read_b64 v[96:97], v219 offset:53792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[30:35], v[78:83], a[44:47], v169, v170 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[98:99], v219 offset:53800
	ds_read_b64 v[100:101], v219 offset:53808
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[24:29], v[78:83], a[48:51], v168, v170 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[102:103], v219 offset:53920
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[18:23], v[78:83], a[52:55], v168, v170 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[104:105], v219 offset:53928
	ds_read_b64 v[106:107], v219 offset:53936
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[12:17], v[78:83], a[56:59], v168, v170 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s97 offen offset:3072 lds
	ds_read_b64 v[108:109], v219 offset:54048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[0:5], v[78:83], a[60:63], v168, v170 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[110:111], v219 offset:54056
	ds_read_b64 v[112:113], v219 offset:54064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[0:5], v[84:89], a[64:67], v168, v170 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[78:79], v219 offset:54176
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[12:17], v[84:89], a[68:71], v168, v170 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[80:81], v219 offset:54184
	ds_read_b64 v[82:83], v219 offset:54192
	s_mov_b32 m0, s79
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[18:23], v[84:89], a[72:75], v168, v170 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[20:23], s97 offen lds
	ds_read_b128 v[114:117], v220 offset:28672
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[24:29], v[84:89], a[76:79], v168, v170 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[118:119], v221 offset:45056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[30:35], v[84:89], a[80:83], v169, v170 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[120:123], v220 offset:29696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[36:41], v[84:89], a[84:87], v169, v170 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[124:125], v221 offset:45568
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[42:47], v[84:89], a[88:91], v169, v170 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[20:23], s97 offen offset:1024 lds
	ds_read_b128 v[126:129], v220 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[48:53], v[84:89], a[92:95], v169, v170 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[130:131], v221 offset:46080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[48:53], v[90:95], a[96:99], v169, v170 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[84:87], v220 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[42:47], v[90:95], a[100:103], v169, v170 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[88:89], v221 offset:46592
	s_add_i32 s94, s65, s94
	s_mov_b32 m0, s65
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[36:41], v[90:95], a[104:107], v169, v170 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[4:7], s94 offen lds
	ds_read_b128 v[150:153], v220 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[30:35], v[90:95], a[108:111], v169, v170 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[154:155], v221 offset:47104
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[24:29], v[90:95], a[112:115], v168, v170 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[156:159], v220 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[18:23], v[90:95], a[116:119], v168, v170 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[160:161], v221 offset:47616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[12:17], v[90:95], a[120:123], v168, v170 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[24:27], s94 offen offset:1024 lds
	ds_read_b128 v[162:165], v220 offset:34816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[0:5], v[90:95], a[124:127], v168, v170 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[166:167], v221 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[0:5], v[132:137], a[128:131], v168, v171 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[90:93], v220 offset:35840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[12:17], v[132:137], a[132:135], v168, v171 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[94:95], v221 offset:48640
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[18:23], v[132:137], a[136:139], v168, v171 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[24:29], v[132:137], a[140:143], v168, v171 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s94, s57, s96
	s_add_i32 s94, s94, 0x80
	s_mov_b32 m0, s67
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[30:35], v[132:137], a[144:147], v169, v171 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[36:41], v[132:137], a[148:151], v169, v171 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[42:47], v[132:137], a[152:155], v169, v171 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_mov_b32 m0, s71
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[48:53], v[132:137], a[156:159], v169, v171 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[28:31], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[48:53], v[138:143], a[160:163], v169, v171 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[42:47], v[138:143], a[164:167], v169, v171 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[36:41], v[138:143], a[168:171], v169, v171 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_mov_b32 m0, s72
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[30:35], v[138:143], a[172:175], v169, v171 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[32:35], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[24:29], v[138:143], a[176:179], v168, v171 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[18:23], v[138:143], a[180:183], v168, v171 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_mov_b32 m0, s73
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[12:17], v[138:143], a[184:187], v168, v171 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[36:39], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[0:5], v[138:143], a[188:191], v168, v171 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[0:5], v[144:149], a[192:195], v168, v171 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[12:17], v[144:149], a[196:199], v168, v171 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s74
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[18:23], v[144:149], a[200:203], v168, v171 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[40:43], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[24:29], v[144:149], a[204:207], v168, v171 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[30:35], v[144:149], a[208:211], v169, v171 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s75
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[36:41], v[144:149], a[212:215], v169, v171 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[44:47], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[42:47], v[144:149], a[216:219], v169, v171 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[48:53], v[144:149], a[220:223], v169, v171 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[48:53], v[60:65], a[224:227], v169, v171 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s76
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[42:47], v[60:65], a[228:231], v169, v171 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[48:51], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[36:41], v[60:65], a[232:235], v169, v171 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[30:35], v[60:65], a[236:239], v169, v171 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s77
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[24:29], v[60:65], a[240:243], v168, v171 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[52:55], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[18:23], v[60:65], a[244:247], v168, v171 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[12:17], v[60:65], a[248:251], v168, v171 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[0:5], v[60:65], a[252:255], v168, v171 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_add_i32 s94, s92, 0xa0
	s_lshl_b32 s94, s94, 9
	s_add_i32 s96, s9, 6
	s_lshr_b32 s96, s96, 2
	s_mul_i32 s96, s96, 3
	s_lshl_b32 s96, s96, 7
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[6:11], v[114:119], a[0:3], v172, v174 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(8) lgkmcnt(0)
	s_barrier
	ds_read_b128 v[12:15], v234 offset:20736
	ds_read_b128 v[16:19], v234 offset:20752
	ds_read_b64 v[168:169], v217
	ds_read_b64 v[170:171], v218 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[54:59], v[114:119], a[4:7], v172, v174 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_i32 s97, s95, 0x1e000
	s_mov_b32 m0, s89
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[66:71], v[114:119], a[8:11], v172, v174 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s97 offen lds
	ds_read_b64 v[0:1], v235 offset:20736
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[72:77], v[114:119], a[12:15], v172, v174 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[2:3], v235 offset:20744
	ds_read_b64 v[4:5], v235 offset:20752
	s_waitcnt lgkmcnt(6)
	ds_write_b64 v223, v[12:13] offset:53248
	ds_write_b64 v223, v[14:15] offset:53256
	s_waitcnt lgkmcnt(7)
	ds_write_b64 v223, v[16:17] offset:53264
	ds_write_b64 v223, v[18:19] offset:53272
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[96:101], v[114:119], a[16:19], v173, v174 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[12:13], v235 offset:20864
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[102:107], v[114:119], a[20:23], v173, v174 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[14:15], v235 offset:20872
	ds_read_b64 v[16:17], v235 offset:20880
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[108:113], v[114:119], a[24:27], v173, v174 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s97 offen offset:1024 lds
	ds_read_b64 v[18:19], v235 offset:20992
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[78:83], v[114:119], a[28:31], v173, v174 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[20:21], v235 offset:21000
	ds_read_b64 v[22:23], v235 offset:21008
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[78:83], v[120:125], a[32:35], v173, v174 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[24:25], v235 offset:21120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[108:113], v[120:125], a[36:39], v173, v174 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[26:27], v235 offset:21128
	ds_read_b64 v[28:29], v235 offset:21136
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[102:107], v[120:125], a[40:43], v173, v174 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s97 offen offset:2048 lds
	ds_read_b64 v[30:31], v235 offset:21248
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[96:101], v[120:125], a[44:47], v173, v174 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[32:33], v235 offset:21256
	ds_read_b64 v[34:35], v235 offset:21264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[72:77], v[120:125], a[48:51], v172, v174 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[36:37], v235 offset:21376
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[66:71], v[120:125], a[52:55], v172, v174 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[38:39], v235 offset:21384
	ds_read_b64 v[40:41], v235 offset:21392
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[54:59], v[120:125], a[56:59], v172, v174 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s97 offen offset:3072 lds
	ds_read_b64 v[42:43], v235 offset:21504
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[6:11], v[120:125], a[60:63], v172, v174 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[44:45], v235 offset:21512
	ds_read_b64 v[46:47], v235 offset:21520
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[6:11], v[126:131], a[64:67], v172, v174 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[48:49], v235 offset:21632
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[54:59], v[126:131], a[68:71], v172, v174 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[50:51], v235 offset:21640
	ds_read_b64 v[52:53], v235 offset:21648
	s_mov_b32 m0, s90
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[66:71], v[126:131], a[72:75], v172, v174 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[20:23], s97 offen lds
	ds_read_b128 v[60:63], v220 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[72:77], v[126:131], a[76:79], v172, v174 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[64:65], v221 offset:20480
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[96:101], v[126:131], a[80:83], v173, v174 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[114:117], v220 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[102:107], v[126:131], a[84:87], v173, v174 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[118:119], v221 offset:20992
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[108:113], v[126:131], a[88:91], v173, v174 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[20:23], s97 offen offset:1024 lds
	ds_read_b128 v[120:123], v220 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[78:83], v[126:131], a[92:95], v173, v174 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[124:125], v221 offset:21504
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[78:83], v[84:89], a[96:99], v173, v174 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[126:129], v220 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[108:113], v[84:89], a[100:103], v173, v174 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[130:131], v221 offset:22016
	s_add_i32 s94, s65, s94
	s_mov_b32 m0, s91
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[102:107], v[84:89], a[104:107], v173, v174 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[4:7], s94 offen lds
	ds_read_b128 v[132:135], v220 offset:8192
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[96:101], v[84:89], a[108:111], v173, v174 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[136:137], v221 offset:22528
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[72:77], v[84:89], a[112:115], v172, v174 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[138:141], v220 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[66:71], v[84:89], a[116:119], v172, v174 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[142:143], v221 offset:23040
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[54:59], v[84:89], a[120:123], v172, v174 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[24:27], s94 offen offset:1024 lds
	ds_read_b128 v[144:147], v220 offset:10240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[6:11], v[84:89], a[124:127], v172, v174 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[148:149], v221 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[6:11], v[150:155], a[128:131], v172, v175 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[84:87], v220 offset:11264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[54:59], v[150:155], a[132:135], v172, v175 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[88:89], v221 offset:24064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[66:71], v[150:155], a[136:139], v172, v175 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[72:77], v[150:155], a[140:143], v172, v175 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s94, s57, s96
	s_add_i32 s94, s94, 0x100
	s_mov_b32 m0, s81
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[96:101], v[150:155], a[144:147], v173, v175 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[102:107], v[150:155], a[148:151], v173, v175 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[108:113], v[150:155], a[152:155], v173, v175 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_mov_b32 m0, s82
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[78:83], v[150:155], a[156:159], v173, v175 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[28:31], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[78:83], v[156:161], a[160:163], v173, v175 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[108:113], v[156:161], a[164:167], v173, v175 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[102:107], v[156:161], a[168:171], v173, v175 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_mov_b32 m0, s83
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[96:101], v[156:161], a[172:175], v173, v175 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[32:35], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[72:77], v[156:161], a[176:179], v172, v175 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[66:71], v[156:161], a[180:183], v172, v175 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_mov_b32 m0, s84
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[54:59], v[156:161], a[184:187], v172, v175 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[36:39], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[6:11], v[156:161], a[188:191], v172, v175 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[6:11], v[162:167], a[192:195], v172, v175 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[54:59], v[162:167], a[196:199], v172, v175 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s85
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[66:71], v[162:167], a[200:203], v172, v175 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[40:43], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[72:77], v[162:167], a[204:207], v172, v175 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[96:101], v[162:167], a[208:211], v173, v175 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s86
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[102:107], v[162:167], a[212:215], v173, v175 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[44:47], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[108:113], v[162:167], a[216:219], v173, v175 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[78:83], v[162:167], a[220:223], v173, v175 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[78:83], v[90:95], a[224:227], v173, v175 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s87
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[108:113], v[90:95], a[228:231], v173, v175 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[48:51], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[102:107], v[90:95], a[232:235], v173, v175 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[96:101], v[90:95], a[236:239], v173, v175 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s88
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[72:77], v[90:95], a[240:243], v172, v175 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[52:55], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[66:71], v[90:95], a[244:247], v172, v175 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[54:59], v[90:95], a[248:251], v172, v175 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[6:11], v[90:95], a[252:255], v172, v175 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_add_i32 s94, s92, 0xc0
	s_lshl_b32 s94, s94, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[0:5], v[60:65], a[0:3], v168, v170 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(8) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[172:173], v217 offset:2048
	ds_read_b64 v[174:175], v218 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[12:17], v[60:65], a[4:7], v168, v170 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_i32 s96, s95, 0x24000
	s_mov_b32 m0, s78
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:23], v[60:65], a[8:11], v168, v170 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s96 offen lds
	ds_read_b64 v[6:7], v236 offset:20832
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[24:29], v[60:65], a[12:15], v168, v170 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[8:9], v237 offset:20840
	ds_read_b64 v[10:11], v237 offset:20848
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[30:35], v[60:65], a[16:19], v169, v170 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[54:55], v236 offset:20960
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[36:41], v[60:65], a[20:23], v169, v170 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[56:57], v237 offset:20968
	ds_read_b64 v[58:59], v237 offset:20976
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[42:47], v[60:65], a[24:27], v169, v170 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s96 offen offset:1024 lds
	ds_read_b64 v[66:67], v236 offset:21088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[48:53], v[60:65], a[28:31], v169, v170 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[68:69], v237 offset:21096
	ds_read_b64 v[70:71], v237 offset:21104
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[48:53], v[114:119], a[32:35], v169, v170 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[60:61], v236 offset:21216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[42:47], v[114:119], a[36:39], v169, v170 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[62:63], v237 offset:21224
	ds_read_b64 v[64:65], v237 offset:21232
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[36:41], v[114:119], a[40:43], v169, v170 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s96 offen offset:2048 lds
	ds_read_b64 v[72:73], v236 offset:21344
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[30:35], v[114:119], a[44:47], v169, v170 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[74:75], v237 offset:21352
	ds_read_b64 v[76:77], v237 offset:21360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[24:29], v[114:119], a[48:51], v168, v170 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[78:79], v236 offset:21472
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[18:23], v[114:119], a[52:55], v168, v170 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[80:81], v237 offset:21480
	ds_read_b64 v[82:83], v237 offset:21488
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[12:17], v[114:119], a[56:59], v168, v170 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s96 offen offset:3072 lds
	ds_read_b64 v[90:91], v236 offset:21600
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[0:5], v[114:119], a[60:63], v168, v170 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[92:93], v237 offset:21608
	ds_read_b64 v[94:95], v237 offset:21616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[0:5], v[120:125], a[64:67], v168, v170 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[96:97], v236 offset:21728
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[12:17], v[120:125], a[68:71], v168, v170 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[98:99], v237 offset:21736
	ds_read_b64 v[100:101], v237 offset:21744
	s_mov_b32 m0, s79
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[18:23], v[120:125], a[72:75], v168, v170 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[20:23], s96 offen lds
	ds_read_b128 v[102:105], v220 offset:28672
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[24:29], v[120:125], a[76:79], v168, v170 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[106:107], v221 offset:45056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[30:35], v[120:125], a[80:83], v169, v170 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[108:111], v220 offset:29696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[36:41], v[120:125], a[84:87], v169, v170 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[112:113], v221 offset:45568
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[42:47], v[120:125], a[88:91], v169, v170 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[20:23], s96 offen offset:1024 lds
	ds_read_b128 v[114:117], v220 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[48:53], v[120:125], a[92:95], v169, v170 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[118:119], v221 offset:46080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[48:53], v[126:131], a[96:99], v169, v170 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[120:123], v220 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[42:47], v[126:131], a[100:103], v169, v170 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[124:125], v221 offset:46592
	s_add_i32 s94, s65, s94
	s_mov_b32 m0, s65
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[36:41], v[126:131], a[104:107], v169, v170 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[4:7], s94 offen lds
	ds_read_b128 v[150:153], v220 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[30:35], v[126:131], a[108:111], v169, v170 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[154:155], v221 offset:47104
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[24:29], v[126:131], a[112:115], v168, v170 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[156:159], v220 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[18:23], v[126:131], a[116:119], v168, v170 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[160:161], v221 offset:47616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[12:17], v[126:131], a[120:123], v168, v170 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[24:27], s94 offen offset:1024 lds
	ds_read_b128 v[162:165], v220 offset:34816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[0:5], v[126:131], a[124:127], v168, v170 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[166:167], v221 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[0:5], v[132:137], a[128:131], v168, v171 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[126:129], v220 offset:35840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[12:17], v[132:137], a[132:135], v168, v171 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[130:131], v221 offset:48640
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[18:23], v[132:137], a[136:139], v168, v171 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[24:29], v[132:137], a[140:143], v168, v171 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[30:35], v[132:137], a[144:147], v169, v171 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[36:41], v[132:137], a[148:151], v169, v171 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[42:47], v[132:137], a[152:155], v169, v171 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[48:53], v[132:137], a[156:159], v169, v171 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[48:53], v[138:143], a[160:163], v169, v171 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[42:47], v[138:143], a[164:167], v169, v171 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[36:41], v[138:143], a[168:171], v169, v171 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[30:35], v[138:143], a[172:175], v169, v171 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[24:29], v[138:143], a[176:179], v168, v171 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[18:23], v[138:143], a[180:183], v168, v171 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[12:17], v[138:143], a[184:187], v168, v171 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[0:5], v[138:143], a[188:191], v168, v171 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[0:5], v[144:149], a[192:195], v168, v171 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[12:17], v[144:149], a[196:199], v168, v171 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[18:23], v[144:149], a[200:203], v168, v171 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[24:29], v[144:149], a[204:207], v168, v171 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[30:35], v[144:149], a[208:211], v169, v171 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[36:41], v[144:149], a[212:215], v169, v171 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[42:47], v[144:149], a[216:219], v169, v171 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[48:53], v[144:149], a[220:223], v169, v171 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[48:53], v[84:89], a[224:227], v169, v171 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[42:47], v[84:89], a[228:231], v169, v171 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[36:41], v[84:89], a[232:235], v169, v171 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[30:35], v[84:89], a[236:239], v169, v171 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[24:29], v[84:89], a[240:243], v168, v171 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[18:23], v[84:89], a[244:247], v168, v171 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[12:17], v[84:89], a[248:251], v168, v171 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[0:5], v[84:89], a[252:255], v168, v171 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_add_i32 s94, s92, 0xe0
	s_lshl_b32 s94, s94, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[6:11], v[102:107], a[0:3], v172, v174 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[168:169], v217
	ds_read_b64 v[170:171], v218 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[54:59], v[102:107], a[4:7], v172, v174 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_i32 s96, s95, 0x2a000
	s_mov_b32 m0, s89
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[66:71], v[102:107], a[8:11], v172, v174 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s96 offen lds
	ds_read_b64 v[0:1], v230 offset:53312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[60:65], v[102:107], a[12:15], v172, v174 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[2:3], v230 offset:53320
	ds_read_b64 v[4:5], v231 offset:53328
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[72:77], v[102:107], a[16:19], v173, v174 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[12:13], v230 offset:53440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[78:83], v[102:107], a[20:23], v173, v174 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[14:15], v230 offset:53448
	ds_read_b64 v[16:17], v231 offset:53456
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[90:95], v[102:107], a[24:27], v173, v174 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s96 offen offset:1024 lds
	ds_read_b64 v[18:19], v230 offset:53568
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[96:101], v[102:107], a[28:31], v173, v174 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[20:21], v230 offset:53576
	ds_read_b64 v[22:23], v231 offset:53584
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[96:101], v[108:113], a[32:35], v173, v174 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[24:25], v230 offset:53696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[90:95], v[108:113], a[36:39], v173, v174 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[26:27], v230 offset:53704
	ds_read_b64 v[28:29], v231 offset:53712
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[78:83], v[108:113], a[40:43], v173, v174 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s96 offen offset:2048 lds
	ds_read_b64 v[30:31], v230 offset:53824
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[72:77], v[108:113], a[44:47], v173, v174 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[32:33], v230 offset:53832
	ds_read_b64 v[34:35], v231 offset:53840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[60:65], v[108:113], a[48:51], v172, v174 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[36:37], v230 offset:53952
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[66:71], v[108:113], a[52:55], v172, v174 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[38:39], v230 offset:53960
	ds_read_b64 v[40:41], v231 offset:53968
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[54:59], v[108:113], a[56:59], v172, v174 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s96 offen offset:3072 lds
	ds_read_b64 v[42:43], v230 offset:54080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[6:11], v[108:113], a[60:63], v172, v174 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[44:45], v230 offset:54088
	ds_read_b64 v[46:47], v231 offset:54096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[6:11], v[114:119], a[64:67], v172, v174 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[48:49], v230 offset:54208
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[54:59], v[114:119], a[68:71], v172, v174 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[50:51], v230 offset:54216
	ds_read_b64 v[52:53], v231 offset:54224
	s_mov_b32 m0, s90
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[66:71], v[114:119], a[72:75], v172, v174 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[20:23], s96 offen lds
	ds_read_b128 v[84:87], v220 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[60:65], v[114:119], a[76:79], v172, v174 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[88:89], v221 offset:20480
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[72:77], v[114:119], a[80:83], v173, v174 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[102:105], v220 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[78:83], v[114:119], a[84:87], v173, v174 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[106:107], v221 offset:20992
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[90:95], v[114:119], a[88:91], v173, v174 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[20:23], s96 offen offset:1024 lds
	ds_read_b128 v[108:111], v220 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[96:101], v[114:119], a[92:95], v173, v174 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[112:113], v221 offset:21504
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[96:101], v[120:125], a[96:99], v173, v174 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[114:117], v220 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[90:95], v[120:125], a[100:103], v173, v174 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[118:119], v221 offset:22016
	s_add_i32 s94, s65, s94
	s_mov_b32 m0, s91
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[78:83], v[120:125], a[104:107], v173, v174 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[4:7], s94 offen lds
	ds_read_b128 v[132:135], v220 offset:8192
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[72:77], v[120:125], a[108:111], v173, v174 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[136:137], v221 offset:22528
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[60:65], v[120:125], a[112:115], v172, v174 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[138:141], v220 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[66:71], v[120:125], a[116:119], v172, v174 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[142:143], v221 offset:23040
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[54:59], v[120:125], a[120:123], v172, v174 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[24:27], s94 offen offset:1024 lds
	ds_read_b128 v[144:147], v220 offset:10240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[6:11], v[120:125], a[124:127], v172, v174 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[148:149], v221 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[6:11], v[150:155], a[128:131], v172, v175 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[120:123], v220 offset:11264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[54:59], v[150:155], a[132:135], v172, v175 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[124:125], v221 offset:24064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[66:71], v[150:155], a[136:139], v172, v175 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[60:65], v[150:155], a[140:143], v172, v175 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s94, s93, 0x300
	s_mov_b32 m0, s67
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[72:77], v[150:155], a[144:147], v173, v175 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[78:83], v[150:155], a[148:151], v173, v175 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[90:95], v[150:155], a[152:155], v173, v175 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_mov_b32 m0, s71
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[96:101], v[150:155], a[156:159], v173, v175 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[28:31], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[96:101], v[156:161], a[160:163], v173, v175 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[90:95], v[156:161], a[164:167], v173, v175 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[78:83], v[156:161], a[168:171], v173, v175 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_mov_b32 m0, s72
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[72:77], v[156:161], a[172:175], v173, v175 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[32:35], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[60:65], v[156:161], a[176:179], v172, v175 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[66:71], v[156:161], a[180:183], v172, v175 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_mov_b32 m0, s73
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[54:59], v[156:161], a[184:187], v172, v175 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[36:39], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[6:11], v[156:161], a[188:191], v172, v175 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[6:11], v[162:167], a[192:195], v172, v175 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[54:59], v[162:167], a[196:199], v172, v175 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s74
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[66:71], v[162:167], a[200:203], v172, v175 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[40:43], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[60:65], v[162:167], a[204:207], v172, v175 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[72:77], v[162:167], a[208:211], v173, v175 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s75
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[78:83], v[162:167], a[212:215], v173, v175 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[44:47], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[90:95], v[162:167], a[216:219], v173, v175 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[96:101], v[162:167], a[220:223], v173, v175 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[96:101], v[126:131], a[224:227], v173, v175 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s76
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[90:95], v[126:131], a[228:231], v173, v175 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[48:51], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[78:83], v[126:131], a[232:235], v173, v175 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[72:77], v[126:131], a[236:239], v173, v175 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s77
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[60:65], v[126:131], a[240:243], v172, v175 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[52:55], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[66:71], v[126:131], a[244:247], v172, v175 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[54:59], v[126:131], a[248:251], v172, v175 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[6:11], v[126:131], a[252:255], v172, v175 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_add_i32 s94, s92, 0x100
	s_lshl_b32 s94, s94, 9
	s_add_i32 s96, s9, 9
	s_lshr_b32 s96, s96, 2
	s_mul_i32 s96, s96, 3
	s_lshl_b32 s96, s96, 7
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[0:5], v[84:89], a[0:3], v168, v170 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(8) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[172:173], v217 offset:2048
	ds_read_b64 v[174:175], v218 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[12:17], v[84:89], a[4:7], v168, v170 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_i32 s97, s95, 0x30000
	s_mov_b32 m0, s78
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:23], v[84:89], a[8:11], v168, v170 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s97 offen lds
	ds_read_b64 v[6:7], v235 offset:20768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[24:29], v[84:89], a[12:15], v168, v170 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[8:9], v235 offset:20776
	ds_read_b64 v[10:11], v235 offset:20784
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[30:35], v[84:89], a[16:19], v169, v170 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[54:55], v235 offset:20896
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[36:41], v[84:89], a[20:23], v169, v170 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[56:57], v235 offset:20904
	ds_read_b64 v[58:59], v235 offset:20912
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[42:47], v[84:89], a[24:27], v169, v170 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s97 offen offset:1024 lds
	ds_read_b64 v[60:61], v235 offset:21024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[48:53], v[84:89], a[28:31], v169, v170 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[62:63], v235 offset:21032
	ds_read_b64 v[64:65], v235 offset:21040
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[48:53], v[102:107], a[32:35], v169, v170 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[66:67], v235 offset:21152
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[42:47], v[102:107], a[36:39], v169, v170 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[68:69], v235 offset:21160
	ds_read_b64 v[70:71], v235 offset:21168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[36:41], v[102:107], a[40:43], v169, v170 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s97 offen offset:2048 lds
	ds_read_b64 v[72:73], v235 offset:21280
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[30:35], v[102:107], a[44:47], v169, v170 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[74:75], v235 offset:21288
	ds_read_b64 v[76:77], v235 offset:21296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[24:29], v[102:107], a[48:51], v168, v170 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[78:79], v235 offset:21408
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[18:23], v[102:107], a[52:55], v168, v170 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[80:81], v235 offset:21416
	ds_read_b64 v[82:83], v235 offset:21424
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[12:17], v[102:107], a[56:59], v168, v170 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s97 offen offset:3072 lds
	ds_read_b64 v[84:85], v235 offset:21536
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[0:5], v[102:107], a[60:63], v168, v170 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[86:87], v235 offset:21544
	ds_read_b64 v[88:89], v235 offset:21552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[0:5], v[108:113], a[64:67], v168, v170 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[90:91], v235 offset:21664
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[12:17], v[108:113], a[68:71], v168, v170 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[92:93], v235 offset:21672
	ds_read_b64 v[94:95], v235 offset:21680
	s_mov_b32 m0, s79
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[18:23], v[108:113], a[72:75], v168, v170 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[20:23], s97 offen lds
	ds_read_b128 v[96:99], v220 offset:28672
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[24:29], v[108:113], a[76:79], v168, v170 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[100:101], v221 offset:45056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[30:35], v[108:113], a[80:83], v169, v170 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[102:105], v220 offset:29696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[36:41], v[108:113], a[84:87], v169, v170 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[106:107], v221 offset:45568
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[42:47], v[108:113], a[88:91], v169, v170 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[20:23], s97 offen offset:1024 lds
	ds_read_b128 v[126:129], v220 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[48:53], v[108:113], a[92:95], v169, v170 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[130:131], v221 offset:46080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[48:53], v[114:119], a[96:99], v169, v170 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[108:111], v220 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[42:47], v[114:119], a[100:103], v169, v170 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[112:113], v221 offset:46592
	s_add_i32 s94, s65, s94
	s_mov_b32 m0, s65
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[36:41], v[114:119], a[104:107], v169, v170 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[4:7], s94 offen lds
	ds_read_b128 v[150:153], v220 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[30:35], v[114:119], a[108:111], v169, v170 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[154:155], v221 offset:47104
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[24:29], v[114:119], a[112:115], v168, v170 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[156:159], v220 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[18:23], v[114:119], a[116:119], v168, v170 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[160:161], v221 offset:47616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[12:17], v[114:119], a[120:123], v168, v170 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[24:27], s94 offen offset:1024 lds
	ds_read_b128 v[162:165], v220 offset:34816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[0:5], v[114:119], a[124:127], v168, v170 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[166:167], v221 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[0:5], v[132:137], a[128:131], v168, v171 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[114:117], v220 offset:35840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[12:17], v[132:137], a[132:135], v168, v171 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[118:119], v221 offset:48640
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[18:23], v[132:137], a[136:139], v168, v171 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[24:29], v[132:137], a[140:143], v168, v171 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s94, s57, s96
	s_add_i32 s94, s94, 0x80
	s_mov_b32 m0, s81
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[30:35], v[132:137], a[144:147], v169, v171 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[36:41], v[132:137], a[148:151], v169, v171 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[42:47], v[132:137], a[152:155], v169, v171 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_mov_b32 m0, s82
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[48:53], v[132:137], a[156:159], v169, v171 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[28:31], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[48:53], v[138:143], a[160:163], v169, v171 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[42:47], v[138:143], a[164:167], v169, v171 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[36:41], v[138:143], a[168:171], v169, v171 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_mov_b32 m0, s83
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[30:35], v[138:143], a[172:175], v169, v171 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[32:35], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[24:29], v[138:143], a[176:179], v168, v171 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[18:23], v[138:143], a[180:183], v168, v171 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_mov_b32 m0, s84
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[12:17], v[138:143], a[184:187], v168, v171 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[36:39], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[0:5], v[138:143], a[188:191], v168, v171 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[0:5], v[144:149], a[192:195], v168, v171 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[12:17], v[144:149], a[196:199], v168, v171 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s85
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[18:23], v[144:149], a[200:203], v168, v171 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[40:43], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[24:29], v[144:149], a[204:207], v168, v171 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[30:35], v[144:149], a[208:211], v169, v171 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s86
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[36:41], v[144:149], a[212:215], v169, v171 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[44:47], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[42:47], v[144:149], a[216:219], v169, v171 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[48:53], v[144:149], a[220:223], v169, v171 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[48:53], v[120:125], a[224:227], v169, v171 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s87
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[42:47], v[120:125], a[228:231], v169, v171 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[48:51], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[36:41], v[120:125], a[232:235], v169, v171 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[30:35], v[120:125], a[236:239], v169, v171 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s88
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[24:29], v[120:125], a[240:243], v168, v171 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[52:55], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[18:23], v[120:125], a[244:247], v168, v171 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[12:17], v[120:125], a[248:251], v168, v171 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[0:5], v[120:125], a[252:255], v168, v171 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_add_i32 s94, s92, 0x120
	s_lshl_b32 s94, s94, 9
	s_add_i32 s96, s9, 10
	s_lshr_b32 s96, s96, 2
	s_mul_i32 s96, s96, 3
	s_lshl_b32 s96, s96, 7
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[6:11], v[96:101], a[0:3], v172, v174 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(8) lgkmcnt(0)
	s_barrier
	ds_read_b128 v[12:15], v222 offset:53248
	ds_read_b128 v[16:19], v222 offset:53264
	ds_read_b64 v[168:169], v217
	ds_read_b64 v[170:171], v218 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[54:59], v[96:101], a[4:7], v172, v174 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_i32 s97, s95, 0x36000
	s_mov_b32 m0, s89
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[60:65], v[96:101], a[8:11], v172, v174 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s97 offen lds
	ds_read_b64 v[0:1], v219 offset:53248
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:71], v[96:101], a[12:15], v172, v174 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[2:3], v219 offset:53256
	ds_read_b64 v[4:5], v219 offset:53264
	s_waitcnt lgkmcnt(6)
	ds_write_b64 v223, v[12:13] offset:53248
	ds_write_b64 v223, v[14:15] offset:53256
	s_waitcnt lgkmcnt(7)
	ds_write_b64 v223, v[16:17] offset:53264
	ds_write_b64 v223, v[18:19] offset:53272
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[72:77], v[96:101], a[16:19], v173, v174 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[12:13], v219 offset:53376
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[78:83], v[96:101], a[20:23], v173, v174 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[14:15], v219 offset:53384
	ds_read_b64 v[16:17], v219 offset:53392
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[84:89], v[96:101], a[24:27], v173, v174 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s97 offen offset:1024 lds
	ds_read_b64 v[18:19], v219 offset:53504
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[90:95], v[96:101], a[28:31], v173, v174 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[20:21], v219 offset:53512
	ds_read_b64 v[22:23], v219 offset:53520
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[90:95], v[102:107], a[32:35], v173, v174 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[24:25], v219 offset:53632
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[84:89], v[102:107], a[36:39], v173, v174 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[26:27], v219 offset:53640
	ds_read_b64 v[28:29], v219 offset:53648
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[78:83], v[102:107], a[40:43], v173, v174 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s97 offen offset:2048 lds
	ds_read_b64 v[30:31], v219 offset:53760
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[72:77], v[102:107], a[44:47], v173, v174 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[32:33], v219 offset:53768
	ds_read_b64 v[34:35], v219 offset:53776
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[66:71], v[102:107], a[48:51], v172, v174 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[36:37], v219 offset:53888
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[60:65], v[102:107], a[52:55], v172, v174 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[38:39], v219 offset:53896
	ds_read_b64 v[40:41], v219 offset:53904
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[54:59], v[102:107], a[56:59], v172, v174 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s97 offen offset:3072 lds
	ds_read_b64 v[42:43], v219 offset:54016
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[6:11], v[102:107], a[60:63], v172, v174 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[44:45], v219 offset:54024
	ds_read_b64 v[46:47], v219 offset:54032
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[6:11], v[126:131], a[64:67], v172, v174 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[48:49], v219 offset:54144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[54:59], v[126:131], a[68:71], v172, v174 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[50:51], v219 offset:54152
	ds_read_b64 v[52:53], v219 offset:54160
	s_mov_b32 m0, s90
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[60:65], v[126:131], a[72:75], v172, v174 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[20:23], s97 offen lds
	ds_read_b128 v[96:99], v220 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[66:71], v[126:131], a[76:79], v172, v174 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[100:101], v221 offset:20480
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[72:77], v[126:131], a[80:83], v173, v174 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[102:105], v220 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[78:83], v[126:131], a[84:87], v173, v174 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[106:107], v221 offset:20992
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[84:89], v[126:131], a[88:91], v173, v174 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[20:23], s97 offen offset:1024 lds
	ds_read_b128 v[120:123], v220 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[90:95], v[126:131], a[92:95], v173, v174 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[124:125], v221 offset:21504
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[90:95], v[108:113], a[96:99], v173, v174 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[126:129], v220 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[84:89], v[108:113], a[100:103], v173, v174 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[130:131], v221 offset:22016
	s_add_i32 s94, s65, s94
	s_mov_b32 m0, s91
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[78:83], v[108:113], a[104:107], v173, v174 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[4:7], s94 offen lds
	ds_read_b128 v[132:135], v220 offset:8192
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[72:77], v[108:113], a[108:111], v173, v174 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[136:137], v221 offset:22528
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[66:71], v[108:113], a[112:115], v172, v174 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[138:141], v220 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[60:65], v[108:113], a[116:119], v172, v174 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[142:143], v221 offset:23040
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[54:59], v[108:113], a[120:123], v172, v174 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[24:27], s94 offen offset:1024 lds
	ds_read_b128 v[144:147], v220 offset:10240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[6:11], v[108:113], a[124:127], v172, v174 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[148:149], v221 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[6:11], v[150:155], a[128:131], v172, v175 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[108:111], v220 offset:11264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[54:59], v[150:155], a[132:135], v172, v175 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[112:113], v221 offset:24064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[60:65], v[150:155], a[136:139], v172, v175 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[66:71], v[150:155], a[140:143], v172, v175 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s94, s57, s96
	s_add_i32 s94, s94, 0x100
	s_mov_b32 m0, s67
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[72:77], v[150:155], a[144:147], v173, v175 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[78:83], v[150:155], a[148:151], v173, v175 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[84:89], v[150:155], a[152:155], v173, v175 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_mov_b32 m0, s71
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[90:95], v[150:155], a[156:159], v173, v175 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[28:31], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[90:95], v[156:161], a[160:163], v173, v175 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[84:89], v[156:161], a[164:167], v173, v175 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[78:83], v[156:161], a[168:171], v173, v175 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_mov_b32 m0, s72
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[72:77], v[156:161], a[172:175], v173, v175 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[32:35], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[66:71], v[156:161], a[176:179], v172, v175 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[60:65], v[156:161], a[180:183], v172, v175 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_mov_b32 m0, s73
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[54:59], v[156:161], a[184:187], v172, v175 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[36:39], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[6:11], v[156:161], a[188:191], v172, v175 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[6:11], v[162:167], a[192:195], v172, v175 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[54:59], v[162:167], a[196:199], v172, v175 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s74
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[60:65], v[162:167], a[200:203], v172, v175 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[40:43], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[66:71], v[162:167], a[204:207], v172, v175 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[72:77], v[162:167], a[208:211], v173, v175 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s75
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[78:83], v[162:167], a[212:215], v173, v175 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[44:47], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[84:89], v[162:167], a[216:219], v173, v175 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[90:95], v[162:167], a[220:223], v173, v175 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[90:95], v[114:119], a[224:227], v173, v175 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s76
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[84:89], v[114:119], a[228:231], v173, v175 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[48:51], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[78:83], v[114:119], a[232:235], v173, v175 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[72:77], v[114:119], a[236:239], v173, v175 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s77
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[66:71], v[114:119], a[240:243], v172, v175 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[52:55], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[60:65], v[114:119], a[244:247], v172, v175 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[54:59], v[114:119], a[248:251], v172, v175 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[6:11], v[114:119], a[252:255], v172, v175 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_add_i32 s94, s92, 0x140
	s_lshl_b32 s94, s94, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[0:5], v[96:101], a[0:3], v168, v170 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(8) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[172:173], v217 offset:2048
	ds_read_b64 v[174:175], v218 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[12:17], v[96:101], a[4:7], v168, v170 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_i32 s96, s95, 0x3c000
	s_mov_b32 m0, s78
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:23], v[96:101], a[8:11], v168, v170 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s96 offen lds
	ds_read_b64 v[6:7], v224 offset:53344
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[24:29], v[96:101], a[12:15], v168, v170 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[8:9], v225 offset:53352
	ds_read_b64 v[10:11], v225 offset:53360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[30:35], v[96:101], a[16:19], v169, v170 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[54:55], v224 offset:53472
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[36:41], v[96:101], a[20:23], v169, v170 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[56:57], v225 offset:53480
	ds_read_b64 v[58:59], v225 offset:53488
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[42:47], v[96:101], a[24:27], v169, v170 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s96 offen offset:1024 lds
	ds_read_b64 v[60:61], v224 offset:53600
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[48:53], v[96:101], a[28:31], v169, v170 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[62:63], v225 offset:53608
	ds_read_b64 v[64:65], v225 offset:53616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[48:53], v[102:107], a[32:35], v169, v170 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[66:67], v224 offset:53728
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[42:47], v[102:107], a[36:39], v169, v170 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[68:69], v225 offset:53736
	ds_read_b64 v[70:71], v225 offset:53744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[36:41], v[102:107], a[40:43], v169, v170 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s96 offen offset:2048 lds
	ds_read_b64 v[72:73], v224 offset:53856
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[30:35], v[102:107], a[44:47], v169, v170 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[74:75], v225 offset:53864
	ds_read_b64 v[76:77], v225 offset:53872
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[24:29], v[102:107], a[48:51], v168, v170 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[78:79], v224 offset:53984
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[18:23], v[102:107], a[52:55], v168, v170 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[80:81], v225 offset:53992
	ds_read_b64 v[82:83], v225 offset:54000
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[12:17], v[102:107], a[56:59], v168, v170 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s96 offen offset:3072 lds
	ds_read_b64 v[84:85], v224 offset:54112
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[0:5], v[102:107], a[60:63], v168, v170 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[86:87], v225 offset:54120
	ds_read_b64 v[88:89], v225 offset:54128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[0:5], v[120:125], a[64:67], v168, v170 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[90:91], v224 offset:54240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[12:17], v[120:125], a[68:71], v168, v170 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[92:93], v225 offset:54248
	ds_read_b64 v[94:95], v225 offset:54256
	s_mov_b32 m0, s79
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[18:23], v[120:125], a[72:75], v168, v170 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[20:23], s96 offen lds
	ds_read_b128 v[96:99], v220 offset:28672
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[24:29], v[120:125], a[76:79], v168, v170 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[100:101], v221 offset:45056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[30:35], v[120:125], a[80:83], v169, v170 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[102:105], v220 offset:29696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[36:41], v[120:125], a[84:87], v169, v170 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[106:107], v221 offset:45568
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[42:47], v[120:125], a[88:91], v169, v170 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[20:23], s96 offen offset:1024 lds
	ds_read_b128 v[114:117], v220 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[48:53], v[120:125], a[92:95], v169, v170 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[118:119], v221 offset:46080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[48:53], v[126:131], a[96:99], v169, v170 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[120:123], v220 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[42:47], v[126:131], a[100:103], v169, v170 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[124:125], v221 offset:46592
	s_add_i32 s94, s65, s94
	s_mov_b32 m0, s65
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[36:41], v[126:131], a[104:107], v169, v170 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[4:7], s94 offen lds
	ds_read_b128 v[150:153], v220 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[30:35], v[126:131], a[108:111], v169, v170 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[154:155], v221 offset:47104
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[24:29], v[126:131], a[112:115], v168, v170 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[156:159], v220 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[18:23], v[126:131], a[116:119], v168, v170 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[160:161], v221 offset:47616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[12:17], v[126:131], a[120:123], v168, v170 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[24:27], s94 offen offset:1024 lds
	ds_read_b128 v[162:165], v220 offset:34816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[0:5], v[126:131], a[124:127], v168, v170 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[166:167], v221 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[0:5], v[132:137], a[128:131], v168, v171 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[126:129], v220 offset:35840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[12:17], v[132:137], a[132:135], v168, v171 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[130:131], v221 offset:48640
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[18:23], v[132:137], a[136:139], v168, v171 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[24:29], v[132:137], a[140:143], v168, v171 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[30:35], v[132:137], a[144:147], v169, v171 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[36:41], v[132:137], a[148:151], v169, v171 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[42:47], v[132:137], a[152:155], v169, v171 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[48:53], v[132:137], a[156:159], v169, v171 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[48:53], v[138:143], a[160:163], v169, v171 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[42:47], v[138:143], a[164:167], v169, v171 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[36:41], v[138:143], a[168:171], v169, v171 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[30:35], v[138:143], a[172:175], v169, v171 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[24:29], v[138:143], a[176:179], v168, v171 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[18:23], v[138:143], a[180:183], v168, v171 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[12:17], v[138:143], a[184:187], v168, v171 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[0:5], v[138:143], a[188:191], v168, v171 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[0:5], v[144:149], a[192:195], v168, v171 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[12:17], v[144:149], a[196:199], v168, v171 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[18:23], v[144:149], a[200:203], v168, v171 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[24:29], v[144:149], a[204:207], v168, v171 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[30:35], v[144:149], a[208:211], v169, v171 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[36:41], v[144:149], a[212:215], v169, v171 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[42:47], v[144:149], a[216:219], v169, v171 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[48:53], v[144:149], a[220:223], v169, v171 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[48:53], v[108:113], a[224:227], v169, v171 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[42:47], v[108:113], a[228:231], v169, v171 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[36:41], v[108:113], a[232:235], v169, v171 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[30:35], v[108:113], a[236:239], v169, v171 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[24:29], v[108:113], a[240:243], v168, v171 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[18:23], v[108:113], a[244:247], v168, v171 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[12:17], v[108:113], a[248:251], v168, v171 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[0:5], v[108:113], a[252:255], v168, v171 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_add_i32 s94, s92, 0x160
	s_lshl_b32 s94, s94, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[6:11], v[96:101], a[0:3], v172, v174 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[168:169], v217
	ds_read_b64 v[170:171], v218 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[54:59], v[96:101], a[4:7], v172, v174 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_i32 s96, s95, 0x42000
	s_mov_b32 m0, s89
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[60:65], v[96:101], a[8:11], v172, v174 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s96 offen lds
	ds_read_b64 v[0:1], v232 offset:20800
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:71], v[96:101], a[12:15], v172, v174 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[2:3], v232 offset:20808
	ds_read_b64 v[4:5], v233 offset:20816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[72:77], v[96:101], a[16:19], v173, v174 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[12:13], v232 offset:20928
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[78:83], v[96:101], a[20:23], v173, v174 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[14:15], v232 offset:20936
	ds_read_b64 v[16:17], v233 offset:20944
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[84:89], v[96:101], a[24:27], v173, v174 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s96 offen offset:1024 lds
	ds_read_b64 v[18:19], v232 offset:21056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[90:95], v[96:101], a[28:31], v173, v174 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[20:21], v232 offset:21064
	ds_read_b64 v[22:23], v233 offset:21072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[90:95], v[102:107], a[32:35], v173, v174 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[24:25], v232 offset:21184
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[84:89], v[102:107], a[36:39], v173, v174 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[26:27], v232 offset:21192
	ds_read_b64 v[28:29], v233 offset:21200
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[78:83], v[102:107], a[40:43], v173, v174 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s96 offen offset:2048 lds
	ds_read_b64 v[30:31], v232 offset:21312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[72:77], v[102:107], a[44:47], v173, v174 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[32:33], v232 offset:21320
	ds_read_b64 v[34:35], v233 offset:21328
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[66:71], v[102:107], a[48:51], v172, v174 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[36:37], v232 offset:21440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[60:65], v[102:107], a[52:55], v172, v174 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[38:39], v232 offset:21448
	ds_read_b64 v[40:41], v233 offset:21456
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[54:59], v[102:107], a[56:59], v172, v174 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s96 offen offset:3072 lds
	ds_read_b64 v[42:43], v232 offset:21568
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[6:11], v[102:107], a[60:63], v172, v174 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[44:45], v232 offset:21576
	ds_read_b64 v[46:47], v233 offset:21584
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[6:11], v[114:119], a[64:67], v172, v174 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[48:49], v232 offset:21696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[54:59], v[114:119], a[68:71], v172, v174 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[50:51], v232 offset:21704
	ds_read_b64 v[52:53], v233 offset:21712
	s_mov_b32 m0, s90
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[60:65], v[114:119], a[72:75], v172, v174 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[20:23], s96 offen lds
	ds_read_b128 v[96:99], v220 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[66:71], v[114:119], a[76:79], v172, v174 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[100:101], v221 offset:20480
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[72:77], v[114:119], a[80:83], v173, v174 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[102:105], v220 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[78:83], v[114:119], a[84:87], v173, v174 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[106:107], v221 offset:20992
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[84:89], v[114:119], a[88:91], v173, v174 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[20:23], s96 offen offset:1024 lds
	ds_read_b128 v[108:111], v220 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[90:95], v[114:119], a[92:95], v173, v174 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[112:113], v221 offset:21504
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[90:95], v[120:125], a[96:99], v173, v174 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[114:117], v220 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[84:89], v[120:125], a[100:103], v173, v174 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[118:119], v221 offset:22016
	s_add_i32 s94, s65, s94
	s_mov_b32 m0, s91
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[78:83], v[120:125], a[104:107], v173, v174 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[4:7], s94 offen lds
	ds_read_b128 v[132:135], v220 offset:8192
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[72:77], v[120:125], a[108:111], v173, v174 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[136:137], v221 offset:22528
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[66:71], v[120:125], a[112:115], v172, v174 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[138:141], v220 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[60:65], v[120:125], a[116:119], v172, v174 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[142:143], v221 offset:23040
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[54:59], v[120:125], a[120:123], v172, v174 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[24:27], s94 offen offset:1024 lds
	ds_read_b128 v[144:147], v220 offset:10240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[6:11], v[120:125], a[124:127], v172, v174 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[148:149], v221 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[6:11], v[150:155], a[128:131], v172, v175 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[120:123], v220 offset:11264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[54:59], v[150:155], a[132:135], v172, v175 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[124:125], v221 offset:24064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[60:65], v[150:155], a[136:139], v172, v175 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[66:71], v[150:155], a[140:143], v172, v175 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s94, s93, 0x480
	s_mov_b32 m0, s81
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[72:77], v[150:155], a[144:147], v173, v175 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[78:83], v[150:155], a[148:151], v173, v175 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[84:89], v[150:155], a[152:155], v173, v175 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_mov_b32 m0, s82
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[90:95], v[150:155], a[156:159], v173, v175 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[28:31], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[90:95], v[156:161], a[160:163], v173, v175 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[84:89], v[156:161], a[164:167], v173, v175 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[78:83], v[156:161], a[168:171], v173, v175 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_mov_b32 m0, s83
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[72:77], v[156:161], a[172:175], v173, v175 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[32:35], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[66:71], v[156:161], a[176:179], v172, v175 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[60:65], v[156:161], a[180:183], v172, v175 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_mov_b32 m0, s84
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[54:59], v[156:161], a[184:187], v172, v175 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[36:39], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[6:11], v[156:161], a[188:191], v172, v175 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[6:11], v[162:167], a[192:195], v172, v175 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[54:59], v[162:167], a[196:199], v172, v175 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s85
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[60:65], v[162:167], a[200:203], v172, v175 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[40:43], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[66:71], v[162:167], a[204:207], v172, v175 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[72:77], v[162:167], a[208:211], v173, v175 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s86
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[78:83], v[162:167], a[212:215], v173, v175 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[44:47], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[84:89], v[162:167], a[216:219], v173, v175 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[90:95], v[162:167], a[220:223], v173, v175 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[90:95], v[126:131], a[224:227], v173, v175 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s87
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[84:89], v[126:131], a[228:231], v173, v175 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[48:51], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[78:83], v[126:131], a[232:235], v173, v175 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[72:77], v[126:131], a[236:239], v173, v175 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s88
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[66:71], v[126:131], a[240:243], v172, v175 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[52:55], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[60:65], v[126:131], a[244:247], v172, v175 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[54:59], v[126:131], a[248:251], v172, v175 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[6:11], v[126:131], a[252:255], v172, v175 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_add_i32 s94, s92, 0x180
	s_lshl_b32 s94, s94, 9
	s_add_i32 s96, s9, 13
	s_lshr_b32 s96, s96, 2
	s_mul_i32 s96, s96, 3
	s_lshl_b32 s96, s96, 7
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[0:5], v[96:101], a[0:3], v168, v170 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(8) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[172:173], v217 offset:2048
	ds_read_b64 v[174:175], v218 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[12:17], v[96:101], a[4:7], v168, v170 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_i32 s97, s95, 0x48000
	s_mov_b32 m0, s78
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:23], v[96:101], a[8:11], v168, v170 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s97 offen lds
	ds_read_b64 v[6:7], v219 offset:53280
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[24:29], v[96:101], a[12:15], v168, v170 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[8:9], v219 offset:53288
	ds_read_b64 v[10:11], v219 offset:53296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[30:35], v[96:101], a[16:19], v169, v170 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[54:55], v219 offset:53408
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[36:41], v[96:101], a[20:23], v169, v170 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[56:57], v219 offset:53416
	ds_read_b64 v[58:59], v219 offset:53424
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[42:47], v[96:101], a[24:27], v169, v170 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s97 offen offset:1024 lds
	ds_read_b64 v[60:61], v219 offset:53536
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[48:53], v[96:101], a[28:31], v169, v170 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[62:63], v219 offset:53544
	ds_read_b64 v[64:65], v219 offset:53552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[48:53], v[102:107], a[32:35], v169, v170 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[66:67], v219 offset:53664
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[42:47], v[102:107], a[36:39], v169, v170 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[68:69], v219 offset:53672
	ds_read_b64 v[70:71], v219 offset:53680
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[36:41], v[102:107], a[40:43], v169, v170 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s97 offen offset:2048 lds
	ds_read_b64 v[72:73], v219 offset:53792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[30:35], v[102:107], a[44:47], v169, v170 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[74:75], v219 offset:53800
	ds_read_b64 v[76:77], v219 offset:53808
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[24:29], v[102:107], a[48:51], v168, v170 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[78:79], v219 offset:53920
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[18:23], v[102:107], a[52:55], v168, v170 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[80:81], v219 offset:53928
	ds_read_b64 v[82:83], v219 offset:53936
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[12:17], v[102:107], a[56:59], v168, v170 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s97 offen offset:3072 lds
	ds_read_b64 v[84:85], v219 offset:54048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[0:5], v[102:107], a[60:63], v168, v170 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[86:87], v219 offset:54056
	ds_read_b64 v[88:89], v219 offset:54064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[0:5], v[108:113], a[64:67], v168, v170 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[90:91], v219 offset:54176
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[12:17], v[108:113], a[68:71], v168, v170 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[92:93], v219 offset:54184
	ds_read_b64 v[94:95], v219 offset:54192
	s_mov_b32 m0, s79
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[18:23], v[108:113], a[72:75], v168, v170 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[20:23], s97 offen lds
	ds_read_b128 v[96:99], v220 offset:28672
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[24:29], v[108:113], a[76:79], v168, v170 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[100:101], v221 offset:45056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[30:35], v[108:113], a[80:83], v169, v170 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[102:105], v220 offset:29696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[36:41], v[108:113], a[84:87], v169, v170 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[106:107], v221 offset:45568
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[42:47], v[108:113], a[88:91], v169, v170 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[20:23], s97 offen offset:1024 lds
	ds_read_b128 v[126:129], v220 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[48:53], v[108:113], a[92:95], v169, v170 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[130:131], v221 offset:46080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[48:53], v[114:119], a[96:99], v169, v170 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[108:111], v220 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[42:47], v[114:119], a[100:103], v169, v170 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[112:113], v221 offset:46592
	s_add_i32 s94, s65, s94
	s_mov_b32 m0, s65
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[36:41], v[114:119], a[104:107], v169, v170 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[4:7], s94 offen lds
	ds_read_b128 v[150:153], v220 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[30:35], v[114:119], a[108:111], v169, v170 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[154:155], v221 offset:47104
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[24:29], v[114:119], a[112:115], v168, v170 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[156:159], v220 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[18:23], v[114:119], a[116:119], v168, v170 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[160:161], v221 offset:47616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[12:17], v[114:119], a[120:123], v168, v170 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[24:27], s94 offen offset:1024 lds
	ds_read_b128 v[162:165], v220 offset:34816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[0:5], v[114:119], a[124:127], v168, v170 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[166:167], v221 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[0:5], v[132:137], a[128:131], v168, v171 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[114:117], v220 offset:35840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[12:17], v[132:137], a[132:135], v168, v171 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[118:119], v221 offset:48640
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[18:23], v[132:137], a[136:139], v168, v171 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[24:29], v[132:137], a[140:143], v168, v171 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s94, s57, s96
	s_add_i32 s94, s94, 0x80
	s_mov_b32 m0, s67
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[30:35], v[132:137], a[144:147], v169, v171 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[36:41], v[132:137], a[148:151], v169, v171 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[42:47], v[132:137], a[152:155], v169, v171 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_mov_b32 m0, s71
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[48:53], v[132:137], a[156:159], v169, v171 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[28:31], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[48:53], v[138:143], a[160:163], v169, v171 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[42:47], v[138:143], a[164:167], v169, v171 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[36:41], v[138:143], a[168:171], v169, v171 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_mov_b32 m0, s72
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[30:35], v[138:143], a[172:175], v169, v171 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[32:35], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[24:29], v[138:143], a[176:179], v168, v171 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[18:23], v[138:143], a[180:183], v168, v171 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_mov_b32 m0, s73
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[12:17], v[138:143], a[184:187], v168, v171 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[36:39], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[0:5], v[138:143], a[188:191], v168, v171 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[0:5], v[144:149], a[192:195], v168, v171 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[12:17], v[144:149], a[196:199], v168, v171 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s74
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[18:23], v[144:149], a[200:203], v168, v171 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[40:43], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[24:29], v[144:149], a[204:207], v168, v171 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[30:35], v[144:149], a[208:211], v169, v171 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s75
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[36:41], v[144:149], a[212:215], v169, v171 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[44:47], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[42:47], v[144:149], a[216:219], v169, v171 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[48:53], v[144:149], a[220:223], v169, v171 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[48:53], v[120:125], a[224:227], v169, v171 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s76
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[42:47], v[120:125], a[228:231], v169, v171 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[48:51], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[36:41], v[120:125], a[232:235], v169, v171 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[30:35], v[120:125], a[236:239], v169, v171 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s77
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[24:29], v[120:125], a[240:243], v168, v171 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[52:55], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[18:23], v[120:125], a[244:247], v168, v171 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[12:17], v[120:125], a[248:251], v168, v171 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[0:5], v[120:125], a[252:255], v168, v171 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_add_i32 s94, s92, 0x1a0
	s_lshl_b32 s94, s94, 9
	s_add_i32 s96, s9, 14
	s_lshr_b32 s96, s96, 2
	s_mul_i32 s96, s96, 3
	s_lshl_b32 s96, s96, 7
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[6:11], v[96:101], a[0:3], v172, v174 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(8) lgkmcnt(0)
	s_barrier
	ds_read_b128 v[12:15], v234 offset:20736
	ds_read_b128 v[16:19], v234 offset:20752
	ds_read_b64 v[168:169], v217
	ds_read_b64 v[170:171], v218 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[54:59], v[96:101], a[4:7], v172, v174 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_i32 s97, s95, 0x4e000
	s_mov_b32 m0, s89
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[60:65], v[96:101], a[8:11], v172, v174 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s97 offen lds
	ds_read_b64 v[0:1], v235 offset:20736
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:71], v[96:101], a[12:15], v172, v174 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[2:3], v235 offset:20744
	ds_read_b64 v[4:5], v235 offset:20752
	s_waitcnt lgkmcnt(6)
	ds_write_b64 v223, v[12:13] offset:53248
	ds_write_b64 v223, v[14:15] offset:53256
	s_waitcnt lgkmcnt(7)
	ds_write_b64 v223, v[16:17] offset:53264
	ds_write_b64 v223, v[18:19] offset:53272
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[72:77], v[96:101], a[16:19], v173, v174 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[12:13], v235 offset:20864
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[78:83], v[96:101], a[20:23], v173, v174 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[14:15], v235 offset:20872
	ds_read_b64 v[16:17], v235 offset:20880
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[84:89], v[96:101], a[24:27], v173, v174 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s97 offen offset:1024 lds
	ds_read_b64 v[18:19], v235 offset:20992
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[90:95], v[96:101], a[28:31], v173, v174 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[20:21], v235 offset:21000
	ds_read_b64 v[22:23], v235 offset:21008
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[90:95], v[102:107], a[32:35], v173, v174 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[24:25], v235 offset:21120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[84:89], v[102:107], a[36:39], v173, v174 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[26:27], v235 offset:21128
	ds_read_b64 v[28:29], v235 offset:21136
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[78:83], v[102:107], a[40:43], v173, v174 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s97 offen offset:2048 lds
	ds_read_b64 v[30:31], v235 offset:21248
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[72:77], v[102:107], a[44:47], v173, v174 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[32:33], v235 offset:21256
	ds_read_b64 v[34:35], v235 offset:21264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[66:71], v[102:107], a[48:51], v172, v174 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[36:37], v235 offset:21376
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[60:65], v[102:107], a[52:55], v172, v174 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[38:39], v235 offset:21384
	ds_read_b64 v[40:41], v235 offset:21392
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[54:59], v[102:107], a[56:59], v172, v174 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s97 offen offset:3072 lds
	ds_read_b64 v[42:43], v235 offset:21504
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[6:11], v[102:107], a[60:63], v172, v174 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[44:45], v235 offset:21512
	ds_read_b64 v[46:47], v235 offset:21520
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[6:11], v[126:131], a[64:67], v172, v174 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[48:49], v235 offset:21632
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[54:59], v[126:131], a[68:71], v172, v174 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[50:51], v235 offset:21640
	ds_read_b64 v[52:53], v235 offset:21648
	s_mov_b32 m0, s90
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[60:65], v[126:131], a[72:75], v172, v174 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[20:23], s97 offen lds
	ds_read_b128 v[96:99], v220 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[66:71], v[126:131], a[76:79], v172, v174 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[100:101], v221 offset:20480
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[72:77], v[126:131], a[80:83], v173, v174 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[102:105], v220 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[78:83], v[126:131], a[84:87], v173, v174 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[106:107], v221 offset:20992
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[84:89], v[126:131], a[88:91], v173, v174 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[20:23], s97 offen offset:1024 lds
	ds_read_b128 v[120:123], v220 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[90:95], v[126:131], a[92:95], v173, v174 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[124:125], v221 offset:21504
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[90:95], v[108:113], a[96:99], v173, v174 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[126:129], v220 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[84:89], v[108:113], a[100:103], v173, v174 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[130:131], v221 offset:22016
	s_add_i32 s94, s65, s94
	s_mov_b32 m0, s91
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[78:83], v[108:113], a[104:107], v173, v174 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[4:7], s94 offen lds
	ds_read_b128 v[132:135], v220 offset:8192
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[72:77], v[108:113], a[108:111], v173, v174 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[136:137], v221 offset:22528
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[66:71], v[108:113], a[112:115], v172, v174 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[138:141], v220 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[60:65], v[108:113], a[116:119], v172, v174 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[142:143], v221 offset:23040
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[54:59], v[108:113], a[120:123], v172, v174 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[24:27], s94 offen offset:1024 lds
	ds_read_b128 v[144:147], v220 offset:10240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[6:11], v[108:113], a[124:127], v172, v174 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[148:149], v221 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[6:11], v[150:155], a[128:131], v172, v175 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[108:111], v220 offset:11264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[54:59], v[150:155], a[132:135], v172, v175 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[112:113], v221 offset:24064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[60:65], v[150:155], a[136:139], v172, v175 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[66:71], v[150:155], a[140:143], v172, v175 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s94, s57, s96
	s_add_i32 s94, s94, 0x100
	s_mov_b32 m0, s81
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[72:77], v[150:155], a[144:147], v173, v175 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[78:83], v[150:155], a[148:151], v173, v175 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[84:89], v[150:155], a[152:155], v173, v175 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_mov_b32 m0, s82
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[90:95], v[150:155], a[156:159], v173, v175 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[28:31], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[90:95], v[156:161], a[160:163], v173, v175 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[84:89], v[156:161], a[164:167], v173, v175 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[78:83], v[156:161], a[168:171], v173, v175 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_mov_b32 m0, s83
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[72:77], v[156:161], a[172:175], v173, v175 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[32:35], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[66:71], v[156:161], a[176:179], v172, v175 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[60:65], v[156:161], a[180:183], v172, v175 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_mov_b32 m0, s84
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[54:59], v[156:161], a[184:187], v172, v175 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[36:39], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[6:11], v[156:161], a[188:191], v172, v175 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[6:11], v[162:167], a[192:195], v172, v175 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[54:59], v[162:167], a[196:199], v172, v175 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s85
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[60:65], v[162:167], a[200:203], v172, v175 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[40:43], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[66:71], v[162:167], a[204:207], v172, v175 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[72:77], v[162:167], a[208:211], v173, v175 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s86
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[78:83], v[162:167], a[212:215], v173, v175 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[44:47], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[84:89], v[162:167], a[216:219], v173, v175 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[90:95], v[162:167], a[220:223], v173, v175 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[90:95], v[114:119], a[224:227], v173, v175 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s87
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[84:89], v[114:119], a[228:231], v173, v175 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[48:51], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[78:83], v[114:119], a[232:235], v173, v175 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[72:77], v[114:119], a[236:239], v173, v175 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s88
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[66:71], v[114:119], a[240:243], v172, v175 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[52:55], s94 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[60:65], v[114:119], a[244:247], v172, v175 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[54:59], v[114:119], a[248:251], v172, v175 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[6:11], v[114:119], a[252:255], v172, v175 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_add_i32 s94, s92, 0x1c0
	s_lshl_b32 s94, s94, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[0:5], v[96:101], a[0:3], v168, v170 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(8) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[172:173], v217 offset:2048
	ds_read_b64 v[174:175], v218 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[12:17], v[96:101], a[4:7], v168, v170 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_i32 s96, s95, 0x54000
	s_mov_b32 m0, s78
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:23], v[96:101], a[8:11], v168, v170 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s96 offen lds
	ds_read_b64 v[6:7], v236 offset:20832
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[24:29], v[96:101], a[12:15], v168, v170 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[8:9], v237 offset:20840
	ds_read_b64 v[10:11], v237 offset:20848
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[30:35], v[96:101], a[16:19], v169, v170 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[54:55], v236 offset:20960
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[36:41], v[96:101], a[20:23], v169, v170 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[56:57], v237 offset:20968
	ds_read_b64 v[58:59], v237 offset:20976
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[42:47], v[96:101], a[24:27], v169, v170 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s96 offen offset:1024 lds
	ds_read_b64 v[60:61], v236 offset:21088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[48:53], v[96:101], a[28:31], v169, v170 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[62:63], v237 offset:21096
	ds_read_b64 v[64:65], v237 offset:21104
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[48:53], v[102:107], a[32:35], v169, v170 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[66:67], v236 offset:21216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[42:47], v[102:107], a[36:39], v169, v170 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[68:69], v237 offset:21224
	ds_read_b64 v[70:71], v237 offset:21232
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[36:41], v[102:107], a[40:43], v169, v170 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s96 offen offset:2048 lds
	ds_read_b64 v[72:73], v236 offset:21344
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[30:35], v[102:107], a[44:47], v169, v170 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[74:75], v237 offset:21352
	ds_read_b64 v[76:77], v237 offset:21360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[24:29], v[102:107], a[48:51], v168, v170 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[78:79], v236 offset:21472
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[18:23], v[102:107], a[52:55], v168, v170 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[80:81], v237 offset:21480
	ds_read_b64 v[82:83], v237 offset:21488
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[12:17], v[102:107], a[56:59], v168, v170 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s96 offen offset:3072 lds
	ds_read_b64 v[84:85], v236 offset:21600
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[0:5], v[102:107], a[60:63], v168, v170 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[86:87], v237 offset:21608
	ds_read_b64 v[88:89], v237 offset:21616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[0:5], v[120:125], a[64:67], v168, v170 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[90:91], v236 offset:21728
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[12:17], v[120:125], a[68:71], v168, v170 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[92:93], v237 offset:21736
	ds_read_b64 v[94:95], v237 offset:21744
	s_mov_b32 m0, s79
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[18:23], v[120:125], a[72:75], v168, v170 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[20:23], s96 offen lds
	ds_read_b128 v[96:99], v220 offset:28672
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[24:29], v[120:125], a[76:79], v168, v170 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[100:101], v221 offset:45056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[30:35], v[120:125], a[80:83], v169, v170 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[102:105], v220 offset:29696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[36:41], v[120:125], a[84:87], v169, v170 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[106:107], v221 offset:45568
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[42:47], v[120:125], a[88:91], v169, v170 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[20:23], s96 offen offset:1024 lds
	ds_read_b128 v[114:117], v220 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[48:53], v[120:125], a[92:95], v169, v170 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[118:119], v221 offset:46080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[48:53], v[126:131], a[96:99], v169, v170 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[120:123], v220 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[42:47], v[126:131], a[100:103], v169, v170 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[124:125], v221 offset:46592
	s_add_i32 s94, s65, s94
	s_mov_b32 m0, s65
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[36:41], v[126:131], a[104:107], v169, v170 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[4:7], s94 offen lds
	ds_read_b128 v[150:153], v220 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[30:35], v[126:131], a[108:111], v169, v170 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[154:155], v221 offset:47104
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[24:29], v[126:131], a[112:115], v168, v170 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[156:159], v220 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[18:23], v[126:131], a[116:119], v168, v170 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[160:161], v221 offset:47616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[12:17], v[126:131], a[120:123], v168, v170 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[24:27], s94 offen offset:1024 lds
	ds_read_b128 v[162:165], v220 offset:34816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[0:5], v[126:131], a[124:127], v168, v170 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[166:167], v221 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[0:5], v[132:137], a[128:131], v168, v171 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[126:129], v220 offset:35840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[12:17], v[132:137], a[132:135], v168, v171 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[130:131], v221 offset:48640
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[18:23], v[132:137], a[136:139], v168, v171 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[24:29], v[132:137], a[140:143], v168, v171 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[30:35], v[132:137], a[144:147], v169, v171 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[36:41], v[132:137], a[148:151], v169, v171 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[42:47], v[132:137], a[152:155], v169, v171 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[48:53], v[132:137], a[156:159], v169, v171 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[48:53], v[138:143], a[160:163], v169, v171 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[42:47], v[138:143], a[164:167], v169, v171 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[36:41], v[138:143], a[168:171], v169, v171 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[30:35], v[138:143], a[172:175], v169, v171 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[24:29], v[138:143], a[176:179], v168, v171 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[18:23], v[138:143], a[180:183], v168, v171 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[12:17], v[138:143], a[184:187], v168, v171 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[0:5], v[138:143], a[188:191], v168, v171 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[0:5], v[144:149], a[192:195], v168, v171 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[12:17], v[144:149], a[196:199], v168, v171 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[18:23], v[144:149], a[200:203], v168, v171 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[24:29], v[144:149], a[204:207], v168, v171 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[30:35], v[144:149], a[208:211], v169, v171 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[36:41], v[144:149], a[212:215], v169, v171 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[42:47], v[144:149], a[216:219], v169, v171 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[48:53], v[144:149], a[220:223], v169, v171 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[48:53], v[108:113], a[224:227], v169, v171 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[42:47], v[108:113], a[228:231], v169, v171 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[36:41], v[108:113], a[232:235], v169, v171 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[30:35], v[108:113], a[236:239], v169, v171 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[24:29], v[108:113], a[240:243], v168, v171 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[18:23], v[108:113], a[244:247], v168, v171 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[12:17], v[108:113], a[248:251], v168, v171 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[0:5], v[108:113], a[252:255], v168, v171 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_add_i32 s94, s92, 0x1e0
	s_lshl_b32 s94, s94, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[6:11], v[96:101], a[0:3], v172, v174 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[168:169], v217
	ds_read_b64 v[170:171], v218 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[54:59], v[96:101], a[4:7], v172, v174 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_i32 s96, s95, 0x5a000
	s_mov_b32 m0, s89
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[60:65], v[96:101], a[8:11], v172, v174 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s96 offen lds
	ds_read_b64 v[0:1], v230 offset:53312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:71], v[96:101], a[12:15], v172, v174 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[2:3], v230 offset:53320
	ds_read_b64 v[4:5], v231 offset:53328
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[72:77], v[96:101], a[16:19], v173, v174 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[12:13], v230 offset:53440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[78:83], v[96:101], a[20:23], v173, v174 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[14:15], v230 offset:53448
	ds_read_b64 v[16:17], v231 offset:53456
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[84:89], v[96:101], a[24:27], v173, v174 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s96 offen offset:1024 lds
	ds_read_b64 v[18:19], v230 offset:53568
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[90:95], v[96:101], a[28:31], v173, v174 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[20:21], v230 offset:53576
	ds_read_b64 v[22:23], v231 offset:53584
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[90:95], v[102:107], a[32:35], v173, v174 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[24:25], v230 offset:53696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[84:89], v[102:107], a[36:39], v173, v174 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[26:27], v230 offset:53704
	ds_read_b64 v[28:29], v231 offset:53712
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[78:83], v[102:107], a[40:43], v173, v174 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s96 offen offset:2048 lds
	ds_read_b64 v[30:31], v230 offset:53824
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[72:77], v[102:107], a[44:47], v173, v174 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[32:33], v230 offset:53832
	ds_read_b64 v[34:35], v231 offset:53840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[66:71], v[102:107], a[48:51], v172, v174 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[36:37], v230 offset:53952
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[60:65], v[102:107], a[52:55], v172, v174 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[38:39], v230 offset:53960
	ds_read_b64 v[40:41], v231 offset:53968
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[54:59], v[102:107], a[56:59], v172, v174 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s96 offen offset:3072 lds
	ds_read_b64 v[42:43], v230 offset:54080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[6:11], v[102:107], a[60:63], v172, v174 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[44:45], v230 offset:54088
	ds_read_b64 v[46:47], v231 offset:54096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[6:11], v[114:119], a[64:67], v172, v174 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[48:49], v230 offset:54208
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[54:59], v[114:119], a[68:71], v172, v174 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[50:51], v230 offset:54216
	ds_read_b64 v[52:53], v231 offset:54224
	s_mov_b32 m0, s90
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[60:65], v[114:119], a[72:75], v172, v174 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[20:23], s96 offen lds
	ds_read_b128 v[96:99], v220 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[66:71], v[114:119], a[76:79], v172, v174 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[100:101], v221 offset:20480
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[72:77], v[114:119], a[80:83], v173, v174 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[102:105], v220 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[78:83], v[114:119], a[84:87], v173, v174 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[106:107], v221 offset:20992
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[84:89], v[114:119], a[88:91], v173, v174 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[20:23], s96 offen offset:1024 lds
	ds_read_b128 v[108:111], v220 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[90:95], v[114:119], a[92:95], v173, v174 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[112:113], v221 offset:21504
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[90:95], v[120:125], a[96:99], v173, v174 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[114:117], v220 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[84:89], v[120:125], a[100:103], v173, v174 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[118:119], v221 offset:22016
	s_add_i32 s94, s65, s94
	s_mov_b32 m0, s91
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[78:83], v[120:125], a[104:107], v173, v174 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[4:7], s94 offen lds
	ds_read_b128 v[132:135], v220 offset:8192
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[72:77], v[120:125], a[108:111], v173, v174 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[136:137], v221 offset:22528
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[66:71], v[120:125], a[112:115], v172, v174 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[138:141], v220 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[60:65], v[120:125], a[116:119], v172, v174 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[142:143], v221 offset:23040
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[54:59], v[120:125], a[120:123], v172, v174 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[24:27], s94 offen offset:1024 lds
	ds_read_b128 v[144:147], v220 offset:10240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[6:11], v[120:125], a[124:127], v172, v174 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[148:149], v221 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[6:11], v[150:155], a[128:131], v172, v175 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[120:123], v220 offset:11264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[54:59], v[150:155], a[132:135], v172, v175 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[124:125], v221 offset:24064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[60:65], v[150:155], a[136:139], v172, v175 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[66:71], v[150:155], a[140:143], v172, v175 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s93, s93, 0x600
	s_mov_b32 m0, s67
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[72:77], v[150:155], a[144:147], v173, v175 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s93 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[78:83], v[150:155], a[148:151], v173, v175 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[84:89], v[150:155], a[152:155], v173, v175 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_mov_b32 m0, s71
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[90:95], v[150:155], a[156:159], v173, v175 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[28:31], s93 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[90:95], v[156:161], a[160:163], v173, v175 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[84:89], v[156:161], a[164:167], v173, v175 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[78:83], v[156:161], a[168:171], v173, v175 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_mov_b32 m0, s72
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[72:77], v[156:161], a[172:175], v173, v175 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[32:35], s93 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[66:71], v[156:161], a[176:179], v172, v175 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[60:65], v[156:161], a[180:183], v172, v175 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_mov_b32 m0, s73
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[54:59], v[156:161], a[184:187], v172, v175 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[36:39], s93 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[6:11], v[156:161], a[188:191], v172, v175 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[6:11], v[162:167], a[192:195], v172, v175 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[54:59], v[162:167], a[196:199], v172, v175 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s74
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[60:65], v[162:167], a[200:203], v172, v175 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[40:43], s93 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[66:71], v[162:167], a[204:207], v172, v175 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[72:77], v[162:167], a[208:211], v173, v175 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s75
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[78:83], v[162:167], a[212:215], v173, v175 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[44:47], s93 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[84:89], v[162:167], a[216:219], v173, v175 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[90:95], v[162:167], a[220:223], v173, v175 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[90:95], v[126:131], a[224:227], v173, v175 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s76
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[84:89], v[126:131], a[228:231], v173, v175 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[48:51], s93 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[78:83], v[126:131], a[232:235], v173, v175 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[72:77], v[126:131], a[236:239], v173, v175 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s77
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[66:71], v[126:131], a[240:243], v172, v175 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[52:55], s93 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[60:65], v[126:131], a[244:247], v172, v175 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[54:59], v[126:131], a[248:251], v172, v175 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[6:11], v[126:131], a[252:255], v172, v175 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_add_i32 s93, s92, 0x200
	s_lshl_b32 s93, s93, 9
	s_add_i32 s94, s9, 17
	s_lshr_b32 s94, s94, 2
	s_mul_i32 s94, s94, 3
	s_lshl_b32 s94, s94, 7
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[0:5], v[96:101], a[0:3], v168, v170 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(8) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[204:205], v217 offset:2048
	ds_read_b64 v[206:207], v218 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[12:17], v[96:101], a[4:7], v168, v170 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_i32 s96, s95, 0x60000
	s_mov_b32 m0, s78
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:23], v[96:101], a[8:11], v168, v170 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s96 offen lds
	ds_read_b64 v[6:7], v235 offset:20768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[24:29], v[96:101], a[12:15], v168, v170 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[8:9], v235 offset:20776
	ds_read_b64 v[10:11], v235 offset:20784
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[30:35], v[96:101], a[16:19], v169, v170 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[54:55], v235 offset:20896
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[36:41], v[96:101], a[20:23], v169, v170 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[56:57], v235 offset:20904
	ds_read_b64 v[58:59], v235 offset:20912
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[42:47], v[96:101], a[24:27], v169, v170 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s96 offen offset:1024 lds
	ds_read_b64 v[60:61], v235 offset:21024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[48:53], v[96:101], a[28:31], v169, v170 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[62:63], v235 offset:21032
	ds_read_b64 v[64:65], v235 offset:21040
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[48:53], v[102:107], a[32:35], v169, v170 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[66:67], v235 offset:21152
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[42:47], v[102:107], a[36:39], v169, v170 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[68:69], v235 offset:21160
	ds_read_b64 v[70:71], v235 offset:21168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[36:41], v[102:107], a[40:43], v169, v170 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s96 offen offset:2048 lds
	ds_read_b64 v[72:73], v235 offset:21280
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[30:35], v[102:107], a[44:47], v169, v170 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[74:75], v235 offset:21288
	ds_read_b64 v[76:77], v235 offset:21296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[24:29], v[102:107], a[48:51], v168, v170 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[78:79], v235 offset:21408
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[18:23], v[102:107], a[52:55], v168, v170 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[80:81], v235 offset:21416
	ds_read_b64 v[82:83], v235 offset:21424
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[12:17], v[102:107], a[56:59], v168, v170 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s96 offen offset:3072 lds
	ds_read_b64 v[84:85], v235 offset:21536
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[0:5], v[102:107], a[60:63], v168, v170 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[86:87], v235 offset:21544
	ds_read_b64 v[88:89], v235 offset:21552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[0:5], v[108:113], a[64:67], v168, v170 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[90:91], v235 offset:21664
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[12:17], v[108:113], a[68:71], v168, v170 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[92:93], v235 offset:21672
	ds_read_b64 v[94:95], v235 offset:21680
	s_mov_b32 m0, s79
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[18:23], v[108:113], a[72:75], v168, v170 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[20:23], s96 offen lds
	ds_read_b128 v[96:99], v220 offset:28672
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[24:29], v[108:113], a[76:79], v168, v170 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[100:101], v221 offset:45056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[30:35], v[108:113], a[80:83], v169, v170 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[102:105], v220 offset:29696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[36:41], v[108:113], a[84:87], v169, v170 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[106:107], v221 offset:45568
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[42:47], v[108:113], a[88:91], v169, v170 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[20:23], s96 offen offset:1024 lds
	ds_read_b128 v[126:129], v220 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[48:53], v[108:113], a[92:95], v169, v170 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[130:131], v221 offset:46080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[48:53], v[114:119], a[96:99], v169, v170 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[108:111], v220 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[42:47], v[114:119], a[100:103], v169, v170 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[112:113], v221 offset:46592
	s_add_i32 s93, s65, s93
	s_mov_b32 m0, s65
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[36:41], v[114:119], a[104:107], v169, v170 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[4:7], s93 offen lds
	ds_read_b128 v[150:153], v220 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[30:35], v[114:119], a[108:111], v169, v170 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[154:155], v221 offset:47104
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[24:29], v[114:119], a[112:115], v168, v170 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[156:159], v220 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[18:23], v[114:119], a[116:119], v168, v170 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[160:161], v221 offset:47616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[12:17], v[114:119], a[120:123], v168, v170 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[24:27], s93 offen offset:1024 lds
	ds_read_b128 v[162:165], v220 offset:34816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[0:5], v[114:119], a[124:127], v168, v170 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[166:167], v221 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[0:5], v[132:137], a[128:131], v168, v171 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[114:117], v220 offset:35840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[12:17], v[132:137], a[132:135], v168, v171 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[118:119], v221 offset:48640
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[18:23], v[132:137], a[136:139], v168, v171 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[24:29], v[132:137], a[140:143], v168, v171 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s93, s57, s94
	s_add_i32 s93, s93, 0x80
	s_mov_b32 m0, s81
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[30:35], v[132:137], a[144:147], v169, v171 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s93 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[36:41], v[132:137], a[148:151], v169, v171 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[42:47], v[132:137], a[152:155], v169, v171 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_mov_b32 m0, s82
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[48:53], v[132:137], a[156:159], v169, v171 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[28:31], s93 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[48:53], v[138:143], a[160:163], v169, v171 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[42:47], v[138:143], a[164:167], v169, v171 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[36:41], v[138:143], a[168:171], v169, v171 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_mov_b32 m0, s83
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[30:35], v[138:143], a[172:175], v169, v171 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[32:35], s93 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[24:29], v[138:143], a[176:179], v168, v171 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[18:23], v[138:143], a[180:183], v168, v171 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_mov_b32 m0, s84
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[12:17], v[138:143], a[184:187], v168, v171 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[36:39], s93 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[0:5], v[138:143], a[188:191], v168, v171 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[0:5], v[144:149], a[192:195], v168, v171 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[12:17], v[144:149], a[196:199], v168, v171 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s85
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[18:23], v[144:149], a[200:203], v168, v171 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[40:43], s93 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[24:29], v[144:149], a[204:207], v168, v171 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[30:35], v[144:149], a[208:211], v169, v171 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s86
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[36:41], v[144:149], a[212:215], v169, v171 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[44:47], s93 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[42:47], v[144:149], a[216:219], v169, v171 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[48:53], v[144:149], a[220:223], v169, v171 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[48:53], v[120:125], a[224:227], v169, v171 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s87
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[42:47], v[120:125], a[228:231], v169, v171 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[48:51], s93 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[36:41], v[120:125], a[232:235], v169, v171 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[30:35], v[120:125], a[236:239], v169, v171 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s88
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[24:29], v[120:125], a[240:243], v168, v171 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[52:55], s93 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[18:23], v[120:125], a[244:247], v168, v171 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[12:17], v[120:125], a[248:251], v168, v171 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[0:5], v[120:125], a[252:255], v168, v171 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_add_i32 s93, s92, 0x220
	s_lshl_b32 s93, s93, 9
	s_add_i32 s9, s9, 18
	s_lshr_b32 s9, s9, 2
	s_mul_i32 s9, s9, 3
	s_lshl_b32 s9, s9, 7
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[6:11], v[96:101], a[0:3], v204, v206 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(8) lgkmcnt(0)
	s_barrier
	ds_read_b128 v[12:15], v222 offset:53248
	ds_read_b128 v[16:19], v222 offset:53264
	ds_read_b64 v[208:209], v217
	ds_read_b64 v[210:211], v218 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[54:59], v[96:101], a[4:7], v204, v206 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_i32 s94, s95, 0x66000
	s_mov_b32 m0, s89
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[60:65], v[96:101], a[8:11], v204, v206 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s94 offen lds
	ds_read_b64 v[0:1], v219 offset:53248
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:71], v[96:101], a[12:15], v204, v206 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[2:3], v219 offset:53256
	ds_read_b64 v[4:5], v219 offset:53264
	s_waitcnt lgkmcnt(6)
	ds_write_b64 v223, v[12:13] offset:53248
	ds_write_b64 v223, v[14:15] offset:53256
	s_waitcnt lgkmcnt(7)
	ds_write_b64 v223, v[16:17] offset:53264
	ds_write_b64 v223, v[18:19] offset:53272
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[72:77], v[96:101], a[16:19], v205, v206 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[12:13], v219 offset:53376
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[78:83], v[96:101], a[20:23], v205, v206 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[14:15], v219 offset:53384
	ds_read_b64 v[16:17], v219 offset:53392
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[84:89], v[96:101], a[24:27], v205, v206 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s94 offen offset:1024 lds
	ds_read_b64 v[18:19], v219 offset:53504
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[90:95], v[96:101], a[28:31], v205, v206 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[20:21], v219 offset:53512
	ds_read_b64 v[22:23], v219 offset:53520
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[90:95], v[102:107], a[32:35], v205, v206 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[24:25], v219 offset:53632
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[84:89], v[102:107], a[36:39], v205, v206 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[26:27], v219 offset:53640
	ds_read_b64 v[28:29], v219 offset:53648
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[78:83], v[102:107], a[40:43], v205, v206 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s94 offen offset:2048 lds
	ds_read_b64 v[30:31], v219 offset:53760
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[72:77], v[102:107], a[44:47], v205, v206 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[32:33], v219 offset:53768
	ds_read_b64 v[34:35], v219 offset:53776
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[66:71], v[102:107], a[48:51], v204, v206 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[36:37], v219 offset:53888
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[60:65], v[102:107], a[52:55], v204, v206 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[38:39], v219 offset:53896
	ds_read_b64 v[40:41], v219 offset:53904
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[54:59], v[102:107], a[56:59], v204, v206 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s94 offen offset:3072 lds
	ds_read_b64 v[42:43], v219 offset:54016
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[6:11], v[102:107], a[60:63], v204, v206 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[44:45], v219 offset:54024
	ds_read_b64 v[46:47], v219 offset:54032
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[6:11], v[126:131], a[64:67], v204, v206 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[48:49], v219 offset:54144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[54:59], v[126:131], a[68:71], v204, v206 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[50:51], v219 offset:54152
	ds_read_b64 v[52:53], v219 offset:54160
	s_mov_b32 m0, s90
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[60:65], v[126:131], a[72:75], v204, v206 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[20:23], s94 offen lds
	ds_read_b128 v[120:123], v220 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[66:71], v[126:131], a[76:79], v204, v206 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[124:125], v221 offset:20480
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[72:77], v[126:131], a[80:83], v205, v206 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[132:135], v220 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[78:83], v[126:131], a[84:87], v205, v206 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[136:137], v221 offset:20992
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[84:89], v[126:131], a[88:91], v205, v206 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[20:23], s94 offen offset:1024 lds
	ds_read_b128 v[168:171], v220 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[90:95], v[126:131], a[92:95], v205, v206 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[172:173], v221 offset:21504
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[90:95], v[108:113], a[96:99], v205, v206 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[174:177], v220 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[84:89], v[108:113], a[100:103], v205, v206 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[178:179], v221 offset:22016
	s_add_i32 s93, s65, s93
	s_mov_b32 m0, s91
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[78:83], v[108:113], a[104:107], v205, v206 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[4:7], s93 offen lds
	ds_read_b128 v[180:183], v220 offset:8192
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[72:77], v[108:113], a[108:111], v205, v206 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[184:185], v221 offset:22528
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[66:71], v[108:113], a[112:115], v204, v206 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[186:189], v220 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[60:65], v[108:113], a[116:119], v204, v206 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[190:191], v221 offset:23040
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[54:59], v[108:113], a[120:123], v204, v206 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[24:27], s93 offen offset:1024 lds
	ds_read_b128 v[192:195], v220 offset:10240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[6:11], v[108:113], a[124:127], v204, v206 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[196:197], v221 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[6:11], v[150:155], a[128:131], v204, v207 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[198:201], v220 offset:11264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[54:59], v[150:155], a[132:135], v204, v207 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[202:203], v221 offset:24064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[60:65], v[150:155], a[136:139], v204, v207 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[66:71], v[150:155], a[140:143], v204, v207 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s9, s57, s9
	s_add_i32 s9, s9, 0x100
	s_mov_b32 m0, s67
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[72:77], v[150:155], a[144:147], v205, v207 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s9 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[78:83], v[150:155], a[148:151], v205, v207 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[84:89], v[150:155], a[152:155], v205, v207 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_mov_b32 m0, s71
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[90:95], v[150:155], a[156:159], v205, v207 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[28:31], s9 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[90:95], v[156:161], a[160:163], v205, v207 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[84:89], v[156:161], a[164:167], v205, v207 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[78:83], v[156:161], a[168:171], v205, v207 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_mov_b32 m0, s72
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[72:77], v[156:161], a[172:175], v205, v207 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[32:35], s9 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[66:71], v[156:161], a[176:179], v204, v207 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[60:65], v[156:161], a[180:183], v204, v207 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_mov_b32 m0, s73
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[54:59], v[156:161], a[184:187], v204, v207 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[36:39], s9 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[6:11], v[156:161], a[188:191], v204, v207 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[6:11], v[162:167], a[192:195], v204, v207 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[54:59], v[162:167], a[196:199], v204, v207 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s74
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[60:65], v[162:167], a[200:203], v204, v207 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[40:43], s9 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[66:71], v[162:167], a[204:207], v204, v207 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[72:77], v[162:167], a[208:211], v205, v207 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s75
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[78:83], v[162:167], a[212:215], v205, v207 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[44:47], s9 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[84:89], v[162:167], a[216:219], v205, v207 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[90:95], v[162:167], a[220:223], v205, v207 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[90:95], v[114:119], a[224:227], v205, v207 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s76
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[84:89], v[114:119], a[228:231], v205, v207 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[48:51], s9 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[78:83], v[114:119], a[232:235], v205, v207 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[72:77], v[114:119], a[236:239], v205, v207 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s77
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[66:71], v[114:119], a[240:243], v204, v207 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[52:55], s9 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[60:65], v[114:119], a[244:247], v204, v207 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[54:59], v[114:119], a[248:251], v204, v207 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[6:11], v[114:119], a[252:255], v204, v207 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_add_i32 s9, s92, 0x240
	s_lshl_b32 s9, s9, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[0:5], v[120:125], a[0:3], v208, v210 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(8) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[204:205], v217 offset:2048
	ds_read_b64 v[206:207], v218 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[12:17], v[120:125], a[4:7], v208, v210 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_i32 s92, s95, 0x6c000
	s_mov_b32 m0, s78
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:23], v[120:125], a[8:11], v208, v210 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s92 offen lds
	ds_read_b64 v[96:97], v224 offset:53344
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[24:29], v[120:125], a[12:15], v208, v210 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[98:99], v225 offset:53352
	ds_read_b64 v[100:101], v225 offset:53360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[30:35], v[120:125], a[16:19], v209, v210 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[102:103], v224 offset:53472
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[36:41], v[120:125], a[20:23], v209, v210 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[104:105], v225 offset:53480
	ds_read_b64 v[106:107], v225 offset:53488
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[42:47], v[120:125], a[24:27], v209, v210 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s92 offen offset:1024 lds
	ds_read_b64 v[108:109], v224 offset:53600
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[48:53], v[120:125], a[28:31], v209, v210 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[110:111], v225 offset:53608
	ds_read_b64 v[112:113], v225 offset:53616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[48:53], v[132:137], a[32:35], v209, v210 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[6:7], v224 offset:53728
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[42:47], v[132:137], a[36:39], v209, v210 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[8:9], v225 offset:53736
	ds_read_b64 v[10:11], v225 offset:53744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[36:41], v[132:137], a[40:43], v209, v210 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s92 offen offset:2048 lds
	ds_read_b64 v[114:115], v224 offset:53856
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[30:35], v[132:137], a[44:47], v209, v210 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[116:117], v225 offset:53864
	ds_read_b64 v[118:119], v225 offset:53872
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[24:29], v[132:137], a[48:51], v208, v210 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[120:121], v224 offset:53984
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[18:23], v[132:137], a[52:55], v208, v210 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[122:123], v225 offset:53992
	ds_read_b64 v[124:125], v225 offset:54000
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[12:17], v[132:137], a[56:59], v208, v210 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s92 offen offset:3072 lds
	ds_read_b64 v[126:127], v224 offset:54112
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[0:5], v[132:137], a[60:63], v208, v210 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[128:129], v225 offset:54120
	ds_read_b64 v[130:131], v225 offset:54128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[0:5], v[168:173], a[64:67], v208, v210 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[54:55], v224 offset:54240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[12:17], v[168:173], a[68:71], v208, v210 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[56:57], v225 offset:54248
	ds_read_b64 v[58:59], v225 offset:54256
	s_mov_b32 m0, s79
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[18:23], v[168:173], a[72:75], v208, v210 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[20:23], s92 offen lds
	ds_read_b128 v[132:135], v220 offset:28672
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[24:29], v[168:173], a[76:79], v208, v210 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[136:137], v221 offset:45056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[30:35], v[168:173], a[80:83], v209, v210 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[138:141], v220 offset:29696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[36:41], v[168:173], a[84:87], v209, v210 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[142:143], v221 offset:45568
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[42:47], v[168:173], a[88:91], v209, v210 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[20:23], s92 offen offset:1024 lds
	ds_read_b128 v[144:147], v220 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[48:53], v[168:173], a[92:95], v209, v210 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[148:149], v221 offset:46080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[48:53], v[174:179], a[96:99], v209, v210 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[60:63], v220 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[42:47], v[174:179], a[100:103], v209, v210 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[64:65], v221 offset:46592
	s_add_i32 s9, s65, s9
	s_mov_b32 m0, s65
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[36:41], v[174:179], a[104:107], v209, v210 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[4:7], s9 offen lds
	ds_read_b128 v[150:153], v220 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[30:35], v[174:179], a[108:111], v209, v210 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[154:155], v221 offset:47104
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[24:29], v[174:179], a[112:115], v208, v210 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[156:159], v220 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[18:23], v[174:179], a[116:119], v208, v210 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[160:161], v221 offset:47616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[12:17], v[174:179], a[120:123], v208, v210 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[24:27], s9 offen offset:1024 lds
	ds_read_b128 v[162:165], v220 offset:34816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[0:5], v[174:179], a[124:127], v208, v210 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[166:167], v221 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[0:5], v[180:185], a[128:131], v208, v211 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[66:69], v220 offset:35840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[12:17], v[180:185], a[132:135], v208, v211 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[70:71], v221 offset:48640
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[18:23], v[180:185], a[136:139], v208, v211 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[24:29], v[180:185], a[140:143], v208, v211 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[30:35], v[180:185], a[144:147], v209, v211 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[36:41], v[180:185], a[148:151], v209, v211 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[42:47], v[180:185], a[152:155], v209, v211 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[48:53], v[180:185], a[156:159], v209, v211 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[48:53], v[186:191], a[160:163], v209, v211 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[42:47], v[186:191], a[164:167], v209, v211 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[36:41], v[186:191], a[168:171], v209, v211 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[30:35], v[186:191], a[172:175], v209, v211 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[24:29], v[186:191], a[176:179], v208, v211 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[18:23], v[186:191], a[180:183], v208, v211 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[12:17], v[186:191], a[184:187], v208, v211 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[0:5], v[186:191], a[188:191], v208, v211 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[0:5], v[192:197], a[192:195], v208, v211 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[12:17], v[192:197], a[196:199], v208, v211 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[18:23], v[192:197], a[200:203], v208, v211 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[24:29], v[192:197], a[204:207], v208, v211 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[30:35], v[192:197], a[208:211], v209, v211 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[36:41], v[192:197], a[212:215], v209, v211 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[42:47], v[192:197], a[216:219], v209, v211 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[48:53], v[192:197], a[220:223], v209, v211 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[48:53], v[198:203], a[224:227], v209, v211 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[42:47], v[198:203], a[228:231], v209, v211 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[36:41], v[198:203], a[232:235], v209, v211 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[30:35], v[198:203], a[236:239], v209, v211 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[24:29], v[198:203], a[240:243], v208, v211 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[18:23], v[198:203], a[244:247], v208, v211 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[12:17], v[198:203], a[248:251], v208, v211 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[0:5], v[198:203], a[252:255], v208, v211 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_add_i32 s8, s8, 1
	s_cmp_lt_i32 s8, 2
	s_cbranch_scc1 .Lmain_loop
	s_add_i32 s8, s66, 0x460
	s_lshl_b32 s8, s8, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[96:101], v[132:137], a[0:3], v204, v206 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[168:169], v217
	ds_read_b64 v[170:171], v218 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[102:107], v[132:137], a[4:7], v204, v206 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_i32 s9, s70, 0xd2000
	s_mov_b32 m0, s89
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[108:113], v[132:137], a[8:11], v204, v206 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen lds
	v_add_u32_e32 v176, 0x10000, v226
	ds_read_b64 v[0:1], v176 offset:20800
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[6:11], v[132:137], a[12:15], v204, v206 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[2:3], v176 offset:20808
	v_add_u32_e32 v177, 0x10000, v227
	ds_read_b64 v[4:5], v177 offset:20816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[114:119], v[132:137], a[16:19], v205, v206 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[12:13], v176 offset:20928
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[120:125], v[132:137], a[20:23], v205, v206 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[14:15], v176 offset:20936
	ds_read_b64 v[16:17], v177 offset:20944
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[126:131], v[132:137], a[24:27], v205, v206 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen offset:1024 lds
	ds_read_b64 v[18:19], v176 offset:21056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[54:59], v[132:137], a[28:31], v205, v206 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[20:21], v176 offset:21064
	ds_read_b64 v[22:23], v177 offset:21072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[54:59], v[138:143], a[32:35], v205, v206 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[24:25], v176 offset:21184
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[126:131], v[138:143], a[36:39], v205, v206 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[26:27], v176 offset:21192
	ds_read_b64 v[28:29], v177 offset:21200
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[120:125], v[138:143], a[40:43], v205, v206 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen offset:2048 lds
	ds_read_b64 v[30:31], v176 offset:21312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[114:119], v[138:143], a[44:47], v205, v206 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[32:33], v176 offset:21320
	ds_read_b64 v[34:35], v177 offset:21328
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[6:11], v[138:143], a[48:51], v204, v206 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[36:37], v176 offset:21440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[108:113], v[138:143], a[52:55], v204, v206 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[38:39], v176 offset:21448
	ds_read_b64 v[40:41], v177 offset:21456
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[102:107], v[138:143], a[56:59], v204, v206 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen offset:3072 lds
	ds_read_b64 v[42:43], v176 offset:21568
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[96:101], v[138:143], a[60:63], v204, v206 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[44:45], v176 offset:21576
	ds_read_b64 v[46:47], v177 offset:21584
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[96:101], v[144:149], a[64:67], v204, v206 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[48:49], v176 offset:21696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[102:107], v[144:149], a[68:71], v204, v206 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[50:51], v176 offset:21704
	ds_read_b64 v[52:53], v177 offset:21712
	s_add_i32 s9, s70, 0xd3000
	s_mov_b32 m0, s90
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[108:113], v[144:149], a[72:75], v204, v206 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen lds
	ds_read_b128 v[72:75], v220 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[6:11], v[144:149], a[76:79], v204, v206 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[76:77], v221 offset:20480
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[114:119], v[144:149], a[80:83], v205, v206 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[78:81], v220 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[120:125], v[144:149], a[84:87], v205, v206 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[82:83], v221 offset:20992
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[126:131], v[144:149], a[88:91], v205, v206 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen offset:1024 lds
	ds_read_b128 v[84:87], v220 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[54:59], v[144:149], a[92:95], v205, v206 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[88:89], v221 offset:21504
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[54:59], v[60:65], a[96:99], v205, v206 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[90:93], v220 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[126:131], v[60:65], a[100:103], v205, v206 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[94:95], v221 offset:22016
	s_add_i32 s8, s65, s8
	s_mov_b32 m0, s91
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[120:125], v[60:65], a[104:107], v205, v206 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[4:7], s8 offen lds
	ds_read_b128 v[132:135], v220 offset:8192
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[114:119], v[60:65], a[108:111], v205, v206 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[136:137], v221 offset:22528
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[6:11], v[60:65], a[112:115], v204, v206 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[138:141], v220 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[108:113], v[60:65], a[116:119], v204, v206 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[142:143], v221 offset:23040
	s_add_i32 s8, s80, 0x8bc00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[102:107], v[60:65], a[120:123], v204, v206 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[12:15], s8 offen offset:1024 lds
	ds_read_b128 v[144:147], v220 offset:10240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[96:101], v[60:65], a[124:127], v204, v206 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[148:149], v221 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[96:101], v[150:155], a[128:131], v204, v207 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[60:63], v220 offset:11264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[102:107], v[150:155], a[132:135], v204, v207 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[64:65], v221 offset:24064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[108:113], v[150:155], a[136:139], v204, v207 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[6:11], v[150:155], a[140:143], v204, v207 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s8, s57, 0xd80
	s_mov_b32 m0, s81
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[114:119], v[150:155], a[144:147], v205, v207 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[120:125], v[150:155], a[148:151], v205, v207 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[126:131], v[150:155], a[152:155], v205, v207 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s8, s58, 0xd80
	s_mov_b32 m0, s82
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[54:59], v[150:155], a[156:159], v205, v207 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[54:59], v[156:161], a[160:163], v205, v207 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[126:131], v[156:161], a[164:167], v205, v207 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[120:125], v[156:161], a[168:171], v205, v207 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_i32 s8, s59, 0xd80
	s_mov_b32 m0, s83
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[114:119], v[156:161], a[172:175], v205, v207 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[6:11], v[156:161], a[176:179], v204, v207 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[108:113], v[156:161], a[180:183], v204, v207 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s8, s60, 0xd80
	s_mov_b32 m0, s84
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[102:107], v[156:161], a[184:187], v204, v207 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[96:101], v[156:161], a[188:191], v204, v207 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[96:101], v[162:167], a[192:195], v204, v207 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[102:107], v[162:167], a[196:199], v204, v207 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_add_i32 s8, s61, 0xd80
	s_mov_b32 m0, s85
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[108:113], v[162:167], a[200:203], v204, v207 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[6:11], v[162:167], a[204:207], v204, v207 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[114:119], v[162:167], a[208:211], v205, v207 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_add_i32 s8, s64, 0xd80
	s_mov_b32 m0, s86
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[120:125], v[162:167], a[212:215], v205, v207 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[126:131], v[162:167], a[216:219], v205, v207 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[54:59], v[162:167], a[220:223], v205, v207 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[54:59], v[66:71], a[224:227], v205, v207 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_i32 s8, s68, 0xd80
	s_mov_b32 m0, s87
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[126:131], v[66:71], a[228:231], v205, v207 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[120:125], v[66:71], a[232:235], v205, v207 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[114:119], v[66:71], a[236:239], v205, v207 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_add_i32 s8, s69, 0xd80
	s_mov_b32 m0, s88
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[6:11], v[66:71], a[240:243], v204, v207 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[108:113], v[66:71], a[244:247], v204, v207 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[102:107], v[66:71], a[248:251], v204, v207 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[96:101], v[66:71], a[252:255], v204, v207 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_add_i32 s8, s66, 0x480
	s_lshl_b32 s8, s8, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[0:5], v[72:77], a[0:3], v168, v170 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(8) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[172:173], v217 offset:2048
	ds_read_b64 v[174:175], v218 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[12:17], v[72:77], a[4:7], v168, v170 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_i32 s9, s70, 0xd8000
	s_mov_b32 m0, s78
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:23], v[72:77], a[8:11], v168, v170 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen lds
	ds_read_b64 v[6:7], v219 offset:53280
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[24:29], v[72:77], a[12:15], v168, v170 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[8:9], v219 offset:53288
	ds_read_b64 v[10:11], v219 offset:53296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[30:35], v[72:77], a[16:19], v169, v170 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[54:55], v219 offset:53408
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[36:41], v[72:77], a[20:23], v169, v170 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[56:57], v219 offset:53416
	ds_read_b64 v[58:59], v219 offset:53424
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[42:47], v[72:77], a[24:27], v169, v170 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen offset:1024 lds
	ds_read_b64 v[66:67], v219 offset:53536
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[48:53], v[72:77], a[28:31], v169, v170 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[68:69], v219 offset:53544
	ds_read_b64 v[70:71], v219 offset:53552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[48:53], v[78:83], a[32:35], v169, v170 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[72:73], v219 offset:53664
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[42:47], v[78:83], a[36:39], v169, v170 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[74:75], v219 offset:53672
	ds_read_b64 v[76:77], v219 offset:53680
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[36:41], v[78:83], a[40:43], v169, v170 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen offset:2048 lds
	ds_read_b64 v[96:97], v219 offset:53792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[30:35], v[78:83], a[44:47], v169, v170 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[98:99], v219 offset:53800
	ds_read_b64 v[100:101], v219 offset:53808
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[24:29], v[78:83], a[48:51], v168, v170 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[102:103], v219 offset:53920
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[18:23], v[78:83], a[52:55], v168, v170 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[104:105], v219 offset:53928
	ds_read_b64 v[106:107], v219 offset:53936
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[12:17], v[78:83], a[56:59], v168, v170 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen offset:3072 lds
	ds_read_b64 v[108:109], v219 offset:54048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[0:5], v[78:83], a[60:63], v168, v170 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[110:111], v219 offset:54056
	ds_read_b64 v[112:113], v219 offset:54064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[0:5], v[84:89], a[64:67], v168, v170 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[78:79], v219 offset:54176
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[12:17], v[84:89], a[68:71], v168, v170 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[80:81], v219 offset:54184
	ds_read_b64 v[82:83], v219 offset:54192
	s_add_i32 s9, s70, 0xd9000
	s_mov_b32 m0, s79
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[18:23], v[84:89], a[72:75], v168, v170 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen lds
	ds_read_b128 v[114:117], v220 offset:28672
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[24:29], v[84:89], a[76:79], v168, v170 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[118:119], v221 offset:45056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[30:35], v[84:89], a[80:83], v169, v170 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[120:123], v220 offset:29696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[36:41], v[84:89], a[84:87], v169, v170 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[124:125], v221 offset:45568
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[42:47], v[84:89], a[88:91], v169, v170 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen offset:1024 lds
	ds_read_b128 v[126:129], v220 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[48:53], v[84:89], a[92:95], v169, v170 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[130:131], v221 offset:46080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[48:53], v[90:95], a[96:99], v169, v170 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[84:87], v220 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[42:47], v[90:95], a[100:103], v169, v170 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[88:89], v221 offset:46592
	s_add_i32 s8, s65, s8
	s_mov_b32 m0, s65
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[36:41], v[90:95], a[104:107], v169, v170 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[4:7], s8 offen lds
	ds_read_b128 v[150:153], v220 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[30:35], v[90:95], a[108:111], v169, v170 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[154:155], v221 offset:47104
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[24:29], v[90:95], a[112:115], v168, v170 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[156:159], v220 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[18:23], v[90:95], a[116:119], v168, v170 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[160:161], v221 offset:47616
	s_add_i32 s8, s80, 0x8fc00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[12:17], v[90:95], a[120:123], v168, v170 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[12:15], s8 offen offset:1024 lds
	ds_read_b128 v[162:165], v220 offset:34816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[0:5], v[90:95], a[124:127], v168, v170 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[166:167], v221 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[0:5], v[132:137], a[128:131], v168, v171 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[90:93], v220 offset:35840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[12:17], v[132:137], a[132:135], v168, v171 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[94:95], v221 offset:48640
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[18:23], v[132:137], a[136:139], v168, v171 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[24:29], v[132:137], a[140:143], v168, v171 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s8, s57, 0xe00
	s_mov_b32 m0, s67
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[30:35], v[132:137], a[144:147], v169, v171 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[36:41], v[132:137], a[148:151], v169, v171 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[42:47], v[132:137], a[152:155], v169, v171 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s8, s58, 0xe00
	s_mov_b32 m0, s71
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[48:53], v[132:137], a[156:159], v169, v171 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[48:53], v[138:143], a[160:163], v169, v171 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[42:47], v[138:143], a[164:167], v169, v171 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[36:41], v[138:143], a[168:171], v169, v171 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_i32 s8, s59, 0xe00
	s_mov_b32 m0, s72
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[30:35], v[138:143], a[172:175], v169, v171 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[24:29], v[138:143], a[176:179], v168, v171 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[18:23], v[138:143], a[180:183], v168, v171 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s8, s60, 0xe00
	s_mov_b32 m0, s73
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[12:17], v[138:143], a[184:187], v168, v171 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[0:5], v[138:143], a[188:191], v168, v171 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[0:5], v[144:149], a[192:195], v168, v171 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[12:17], v[144:149], a[196:199], v168, v171 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_add_i32 s8, s61, 0xe00
	s_mov_b32 m0, s74
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[18:23], v[144:149], a[200:203], v168, v171 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[24:29], v[144:149], a[204:207], v168, v171 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[30:35], v[144:149], a[208:211], v169, v171 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_add_i32 s8, s64, 0xe00
	s_mov_b32 m0, s75
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[36:41], v[144:149], a[212:215], v169, v171 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[42:47], v[144:149], a[216:219], v169, v171 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[48:53], v[144:149], a[220:223], v169, v171 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[48:53], v[60:65], a[224:227], v169, v171 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_i32 s8, s68, 0xe00
	s_mov_b32 m0, s76
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[42:47], v[60:65], a[228:231], v169, v171 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[36:41], v[60:65], a[232:235], v169, v171 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[30:35], v[60:65], a[236:239], v169, v171 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_add_i32 s8, s69, 0xe00
	s_mov_b32 m0, s77
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[24:29], v[60:65], a[240:243], v168, v171 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[18:23], v[60:65], a[244:247], v168, v171 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[12:17], v[60:65], a[248:251], v168, v171 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[0:5], v[60:65], a[252:255], v168, v171 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_add_i32 s8, s66, 0x4a0
	s_lshl_b32 s8, s8, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[6:11], v[114:119], a[0:3], v172, v174 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(8) lgkmcnt(0)
	s_barrier
	v_add_u32_e32 v178, 0x10000, v222
	ds_read_b128 v[12:15], v178 offset:20736
	ds_read_b128 v[16:19], v178 offset:20752
	ds_read_b64 v[168:169], v217
	ds_read_b64 v[170:171], v218 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[54:59], v[114:119], a[4:7], v172, v174 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_i32 s9, s70, 0xde000
	s_mov_b32 m0, s89
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[66:71], v[114:119], a[8:11], v172, v174 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen lds
	v_add_u32_e32 v179, 0x10000, v219
	ds_read_b64 v[0:1], v179 offset:20736
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[72:77], v[114:119], a[12:15], v172, v174 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[2:3], v179 offset:20744
	ds_read_b64 v[4:5], v179 offset:20752
	s_waitcnt lgkmcnt(6)
	ds_write_b64 v223, v[12:13] offset:53248
	ds_write_b64 v223, v[14:15] offset:53256
	s_waitcnt lgkmcnt(7)
	ds_write_b64 v223, v[16:17] offset:53264
	ds_write_b64 v223, v[18:19] offset:53272
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[96:101], v[114:119], a[16:19], v173, v174 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[12:13], v179 offset:20864
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[102:107], v[114:119], a[20:23], v173, v174 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[14:15], v179 offset:20872
	ds_read_b64 v[16:17], v179 offset:20880
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[108:113], v[114:119], a[24:27], v173, v174 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen offset:1024 lds
	ds_read_b64 v[18:19], v179 offset:20992
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[78:83], v[114:119], a[28:31], v173, v174 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[20:21], v179 offset:21000
	ds_read_b64 v[22:23], v179 offset:21008
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[78:83], v[120:125], a[32:35], v173, v174 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[24:25], v179 offset:21120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[108:113], v[120:125], a[36:39], v173, v174 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[26:27], v179 offset:21128
	ds_read_b64 v[28:29], v179 offset:21136
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[102:107], v[120:125], a[40:43], v173, v174 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen offset:2048 lds
	ds_read_b64 v[30:31], v179 offset:21248
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[96:101], v[120:125], a[44:47], v173, v174 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[32:33], v179 offset:21256
	ds_read_b64 v[34:35], v179 offset:21264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[72:77], v[120:125], a[48:51], v172, v174 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[36:37], v179 offset:21376
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[66:71], v[120:125], a[52:55], v172, v174 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[38:39], v179 offset:21384
	ds_read_b64 v[40:41], v179 offset:21392
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[54:59], v[120:125], a[56:59], v172, v174 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen offset:3072 lds
	ds_read_b64 v[42:43], v179 offset:21504
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[6:11], v[120:125], a[60:63], v172, v174 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[44:45], v179 offset:21512
	ds_read_b64 v[46:47], v179 offset:21520
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[6:11], v[126:131], a[64:67], v172, v174 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[48:49], v179 offset:21632
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[54:59], v[126:131], a[68:71], v172, v174 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[50:51], v179 offset:21640
	ds_read_b64 v[52:53], v179 offset:21648
	s_add_i32 s9, s70, 0xdf000
	s_mov_b32 m0, s90
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[66:71], v[126:131], a[72:75], v172, v174 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen lds
	ds_read_b128 v[60:63], v220 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[72:77], v[126:131], a[76:79], v172, v174 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[64:65], v221 offset:20480
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[96:101], v[126:131], a[80:83], v173, v174 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[114:117], v220 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[102:107], v[126:131], a[84:87], v173, v174 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[118:119], v221 offset:20992
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[108:113], v[126:131], a[88:91], v173, v174 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen offset:1024 lds
	ds_read_b128 v[120:123], v220 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[78:83], v[126:131], a[92:95], v173, v174 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[124:125], v221 offset:21504
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[78:83], v[84:89], a[96:99], v173, v174 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[126:129], v220 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[108:113], v[84:89], a[100:103], v173, v174 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[130:131], v221 offset:22016
	s_add_i32 s8, s65, s8
	s_mov_b32 m0, s91
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[102:107], v[84:89], a[104:107], v173, v174 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[4:7], s8 offen lds
	ds_read_b128 v[132:135], v220 offset:8192
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[96:101], v[84:89], a[108:111], v173, v174 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[136:137], v221 offset:22528
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[72:77], v[84:89], a[112:115], v172, v174 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[138:141], v220 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[66:71], v[84:89], a[116:119], v172, v174 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[142:143], v221 offset:23040
	s_add_i32 s8, s80, 0x93c00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[54:59], v[84:89], a[120:123], v172, v174 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[12:15], s8 offen offset:1024 lds
	ds_read_b128 v[144:147], v220 offset:10240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[6:11], v[84:89], a[124:127], v172, v174 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[148:149], v221 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[6:11], v[150:155], a[128:131], v172, v175 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[84:87], v220 offset:11264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[54:59], v[150:155], a[132:135], v172, v175 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[88:89], v221 offset:24064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[66:71], v[150:155], a[136:139], v172, v175 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[72:77], v[150:155], a[140:143], v172, v175 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s8, s57, 0xe80
	s_mov_b32 m0, s81
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[96:101], v[150:155], a[144:147], v173, v175 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[102:107], v[150:155], a[148:151], v173, v175 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[108:113], v[150:155], a[152:155], v173, v175 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s8, s58, 0xe80
	s_mov_b32 m0, s82
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[78:83], v[150:155], a[156:159], v173, v175 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[78:83], v[156:161], a[160:163], v173, v175 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[108:113], v[156:161], a[164:167], v173, v175 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[102:107], v[156:161], a[168:171], v173, v175 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_i32 s8, s59, 0xe80
	s_mov_b32 m0, s83
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[96:101], v[156:161], a[172:175], v173, v175 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[72:77], v[156:161], a[176:179], v172, v175 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[66:71], v[156:161], a[180:183], v172, v175 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s8, s60, 0xe80
	s_mov_b32 m0, s84
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[54:59], v[156:161], a[184:187], v172, v175 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[6:11], v[156:161], a[188:191], v172, v175 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[6:11], v[162:167], a[192:195], v172, v175 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[54:59], v[162:167], a[196:199], v172, v175 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_add_i32 s8, s61, 0xe80
	s_mov_b32 m0, s85
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[66:71], v[162:167], a[200:203], v172, v175 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[72:77], v[162:167], a[204:207], v172, v175 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[96:101], v[162:167], a[208:211], v173, v175 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_add_i32 s8, s64, 0xe80
	s_mov_b32 m0, s86
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[102:107], v[162:167], a[212:215], v173, v175 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[108:113], v[162:167], a[216:219], v173, v175 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[78:83], v[162:167], a[220:223], v173, v175 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[78:83], v[90:95], a[224:227], v173, v175 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_i32 s8, s68, 0xe80
	s_mov_b32 m0, s87
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[108:113], v[90:95], a[228:231], v173, v175 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[102:107], v[90:95], a[232:235], v173, v175 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[96:101], v[90:95], a[236:239], v173, v175 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_add_i32 s8, s69, 0xe80
	s_mov_b32 m0, s88
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[72:77], v[90:95], a[240:243], v172, v175 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[66:71], v[90:95], a[244:247], v172, v175 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[54:59], v[90:95], a[248:251], v172, v175 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[6:11], v[90:95], a[252:255], v172, v175 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_add_i32 s8, s66, 0x4c0
	s_lshl_b32 s8, s8, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[0:5], v[60:65], a[0:3], v168, v170 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(8) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[172:173], v217 offset:2048
	ds_read_b64 v[174:175], v218 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[12:17], v[60:65], a[4:7], v168, v170 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_i32 s9, s70, 0xe4000
	s_mov_b32 m0, s78
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:23], v[60:65], a[8:11], v168, v170 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen lds
	v_add_u32_e32 v180, 0x10000, v228
	ds_read_b64 v[6:7], v180 offset:20832
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[24:29], v[60:65], a[12:15], v168, v170 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_add_u32_e32 v181, 0x10000, v229
	ds_read_b64 v[8:9], v181 offset:20840
	ds_read_b64 v[10:11], v181 offset:20848
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[30:35], v[60:65], a[16:19], v169, v170 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[54:55], v180 offset:20960
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[36:41], v[60:65], a[20:23], v169, v170 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[56:57], v181 offset:20968
	ds_read_b64 v[58:59], v181 offset:20976
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[42:47], v[60:65], a[24:27], v169, v170 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen offset:1024 lds
	ds_read_b64 v[66:67], v180 offset:21088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[48:53], v[60:65], a[28:31], v169, v170 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[68:69], v181 offset:21096
	ds_read_b64 v[70:71], v181 offset:21104
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[48:53], v[114:119], a[32:35], v169, v170 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[60:61], v180 offset:21216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[42:47], v[114:119], a[36:39], v169, v170 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[62:63], v181 offset:21224
	ds_read_b64 v[64:65], v181 offset:21232
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[36:41], v[114:119], a[40:43], v169, v170 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen offset:2048 lds
	ds_read_b64 v[72:73], v180 offset:21344
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[30:35], v[114:119], a[44:47], v169, v170 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[74:75], v181 offset:21352
	ds_read_b64 v[76:77], v181 offset:21360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[24:29], v[114:119], a[48:51], v168, v170 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[78:79], v180 offset:21472
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[18:23], v[114:119], a[52:55], v168, v170 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[80:81], v181 offset:21480
	ds_read_b64 v[82:83], v181 offset:21488
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[12:17], v[114:119], a[56:59], v168, v170 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen offset:3072 lds
	ds_read_b64 v[90:91], v180 offset:21600
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[0:5], v[114:119], a[60:63], v168, v170 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[92:93], v181 offset:21608
	ds_read_b64 v[94:95], v181 offset:21616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[0:5], v[120:125], a[64:67], v168, v170 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[96:97], v180 offset:21728
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[12:17], v[120:125], a[68:71], v168, v170 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[98:99], v181 offset:21736
	ds_read_b64 v[100:101], v181 offset:21744
	s_add_i32 s9, s70, 0xe5000
	s_mov_b32 m0, s79
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[18:23], v[120:125], a[72:75], v168, v170 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen lds
	ds_read_b128 v[102:105], v220 offset:28672
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[24:29], v[120:125], a[76:79], v168, v170 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[106:107], v221 offset:45056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[30:35], v[120:125], a[80:83], v169, v170 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[108:111], v220 offset:29696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[36:41], v[120:125], a[84:87], v169, v170 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[112:113], v221 offset:45568
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[42:47], v[120:125], a[88:91], v169, v170 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen offset:1024 lds
	ds_read_b128 v[114:117], v220 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[48:53], v[120:125], a[92:95], v169, v170 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[118:119], v221 offset:46080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[48:53], v[126:131], a[96:99], v169, v170 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[120:123], v220 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[42:47], v[126:131], a[100:103], v169, v170 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[124:125], v221 offset:46592
	s_add_i32 s8, s65, s8
	s_mov_b32 m0, s65
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[36:41], v[126:131], a[104:107], v169, v170 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[4:7], s8 offen lds
	ds_read_b128 v[150:153], v220 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[30:35], v[126:131], a[108:111], v169, v170 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[154:155], v221 offset:47104
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[24:29], v[126:131], a[112:115], v168, v170 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[156:159], v220 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[18:23], v[126:131], a[116:119], v168, v170 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[160:161], v221 offset:47616
	s_add_i32 s8, s80, 0x97c00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[12:17], v[126:131], a[120:123], v168, v170 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[12:15], s8 offen offset:1024 lds
	ds_read_b128 v[162:165], v220 offset:34816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[0:5], v[126:131], a[124:127], v168, v170 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[166:167], v221 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[0:5], v[132:137], a[128:131], v168, v171 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[126:129], v220 offset:35840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[12:17], v[132:137], a[132:135], v168, v171 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[130:131], v221 offset:48640
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[18:23], v[132:137], a[136:139], v168, v171 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[24:29], v[132:137], a[140:143], v168, v171 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[30:35], v[132:137], a[144:147], v169, v171 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[36:41], v[132:137], a[148:151], v169, v171 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[42:47], v[132:137], a[152:155], v169, v171 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[48:53], v[132:137], a[156:159], v169, v171 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[48:53], v[138:143], a[160:163], v169, v171 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[42:47], v[138:143], a[164:167], v169, v171 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[36:41], v[138:143], a[168:171], v169, v171 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[30:35], v[138:143], a[172:175], v169, v171 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[24:29], v[138:143], a[176:179], v168, v171 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[18:23], v[138:143], a[180:183], v168, v171 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[12:17], v[138:143], a[184:187], v168, v171 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[0:5], v[138:143], a[188:191], v168, v171 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[0:5], v[144:149], a[192:195], v168, v171 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[12:17], v[144:149], a[196:199], v168, v171 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[18:23], v[144:149], a[200:203], v168, v171 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[24:29], v[144:149], a[204:207], v168, v171 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[30:35], v[144:149], a[208:211], v169, v171 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[36:41], v[144:149], a[212:215], v169, v171 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[42:47], v[144:149], a[216:219], v169, v171 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[48:53], v[144:149], a[220:223], v169, v171 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[48:53], v[84:89], a[224:227], v169, v171 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[42:47], v[84:89], a[228:231], v169, v171 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[36:41], v[84:89], a[232:235], v169, v171 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[30:35], v[84:89], a[236:239], v169, v171 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[24:29], v[84:89], a[240:243], v168, v171 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[18:23], v[84:89], a[244:247], v168, v171 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[12:17], v[84:89], a[248:251], v168, v171 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[0:5], v[84:89], a[252:255], v168, v171 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_add_i32 s8, s66, 0x4e0
	s_lshl_b32 s8, s8, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[6:11], v[102:107], a[0:3], v172, v174 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[168:169], v217
	ds_read_b64 v[170:171], v218 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[54:59], v[102:107], a[4:7], v172, v174 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_i32 s9, s70, 0xea000
	s_mov_b32 m0, s89
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[66:71], v[102:107], a[8:11], v172, v174 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen lds
	ds_read_b64 v[0:1], v230 offset:53312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[60:65], v[102:107], a[12:15], v172, v174 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[2:3], v230 offset:53320
	ds_read_b64 v[4:5], v231 offset:53328
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[72:77], v[102:107], a[16:19], v173, v174 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[12:13], v230 offset:53440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[78:83], v[102:107], a[20:23], v173, v174 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[14:15], v230 offset:53448
	ds_read_b64 v[16:17], v231 offset:53456
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[90:95], v[102:107], a[24:27], v173, v174 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen offset:1024 lds
	ds_read_b64 v[18:19], v230 offset:53568
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[96:101], v[102:107], a[28:31], v173, v174 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[20:21], v230 offset:53576
	ds_read_b64 v[22:23], v231 offset:53584
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[96:101], v[108:113], a[32:35], v173, v174 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[24:25], v230 offset:53696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[90:95], v[108:113], a[36:39], v173, v174 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[26:27], v230 offset:53704
	ds_read_b64 v[28:29], v231 offset:53712
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[78:83], v[108:113], a[40:43], v173, v174 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen offset:2048 lds
	ds_read_b64 v[30:31], v230 offset:53824
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[72:77], v[108:113], a[44:47], v173, v174 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[32:33], v230 offset:53832
	ds_read_b64 v[34:35], v231 offset:53840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[60:65], v[108:113], a[48:51], v172, v174 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[36:37], v230 offset:53952
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[66:71], v[108:113], a[52:55], v172, v174 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[38:39], v230 offset:53960
	ds_read_b64 v[40:41], v231 offset:53968
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[54:59], v[108:113], a[56:59], v172, v174 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen offset:3072 lds
	ds_read_b64 v[42:43], v230 offset:54080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[6:11], v[108:113], a[60:63], v172, v174 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[44:45], v230 offset:54088
	ds_read_b64 v[46:47], v231 offset:54096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[6:11], v[114:119], a[64:67], v172, v174 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[48:49], v230 offset:54208
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[54:59], v[114:119], a[68:71], v172, v174 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[50:51], v230 offset:54216
	ds_read_b64 v[52:53], v231 offset:54224
	s_add_i32 s9, s70, 0xeb000
	s_mov_b32 m0, s90
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[66:71], v[114:119], a[72:75], v172, v174 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen lds
	ds_read_b128 v[84:87], v220 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[60:65], v[114:119], a[76:79], v172, v174 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[88:89], v221 offset:20480
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[72:77], v[114:119], a[80:83], v173, v174 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[102:105], v220 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[78:83], v[114:119], a[84:87], v173, v174 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[106:107], v221 offset:20992
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[90:95], v[114:119], a[88:91], v173, v174 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen offset:1024 lds
	ds_read_b128 v[108:111], v220 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[96:101], v[114:119], a[92:95], v173, v174 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[112:113], v221 offset:21504
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[96:101], v[120:125], a[96:99], v173, v174 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[114:117], v220 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[90:95], v[120:125], a[100:103], v173, v174 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[118:119], v221 offset:22016
	s_add_i32 s8, s65, s8
	s_mov_b32 m0, s91
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[78:83], v[120:125], a[104:107], v173, v174 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[4:7], s8 offen lds
	ds_read_b128 v[132:135], v220 offset:8192
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[72:77], v[120:125], a[108:111], v173, v174 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[136:137], v221 offset:22528
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[60:65], v[120:125], a[112:115], v172, v174 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[138:141], v220 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[66:71], v[120:125], a[116:119], v172, v174 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[142:143], v221 offset:23040
	s_add_i32 s8, s80, 0x9bc00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[54:59], v[120:125], a[120:123], v172, v174 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[12:15], s8 offen offset:1024 lds
	ds_read_b128 v[144:147], v220 offset:10240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[6:11], v[120:125], a[124:127], v172, v174 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[148:149], v221 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[6:11], v[150:155], a[128:131], v172, v175 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[120:123], v220 offset:11264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[54:59], v[150:155], a[132:135], v172, v175 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[124:125], v221 offset:24064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[66:71], v[150:155], a[136:139], v172, v175 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[60:65], v[150:155], a[140:143], v172, v175 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s8, s57, 0xf00
	s_mov_b32 m0, s67
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[72:77], v[150:155], a[144:147], v173, v175 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[78:83], v[150:155], a[148:151], v173, v175 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[90:95], v[150:155], a[152:155], v173, v175 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s8, s58, 0xf00
	s_mov_b32 m0, s71
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[96:101], v[150:155], a[156:159], v173, v175 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[96:101], v[156:161], a[160:163], v173, v175 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[90:95], v[156:161], a[164:167], v173, v175 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[78:83], v[156:161], a[168:171], v173, v175 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_i32 s8, s59, 0xf00
	s_mov_b32 m0, s72
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[72:77], v[156:161], a[172:175], v173, v175 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[60:65], v[156:161], a[176:179], v172, v175 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[66:71], v[156:161], a[180:183], v172, v175 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s8, s60, 0xf00
	s_mov_b32 m0, s73
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[54:59], v[156:161], a[184:187], v172, v175 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[6:11], v[156:161], a[188:191], v172, v175 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[6:11], v[162:167], a[192:195], v172, v175 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[54:59], v[162:167], a[196:199], v172, v175 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_add_i32 s8, s61, 0xf00
	s_mov_b32 m0, s74
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[66:71], v[162:167], a[200:203], v172, v175 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[60:65], v[162:167], a[204:207], v172, v175 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[72:77], v[162:167], a[208:211], v173, v175 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_add_i32 s8, s64, 0xf00
	s_mov_b32 m0, s75
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[78:83], v[162:167], a[212:215], v173, v175 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[90:95], v[162:167], a[216:219], v173, v175 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[96:101], v[162:167], a[220:223], v173, v175 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[96:101], v[126:131], a[224:227], v173, v175 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_i32 s8, s68, 0xf00
	s_mov_b32 m0, s76
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[90:95], v[126:131], a[228:231], v173, v175 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[78:83], v[126:131], a[232:235], v173, v175 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[72:77], v[126:131], a[236:239], v173, v175 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_add_i32 s8, s69, 0xf00
	s_mov_b32 m0, s77
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[60:65], v[126:131], a[240:243], v172, v175 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[66:71], v[126:131], a[244:247], v172, v175 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[54:59], v[126:131], a[248:251], v172, v175 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[6:11], v[126:131], a[252:255], v172, v175 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_add_i32 s8, s66, 0x500
	s_lshl_b32 s8, s8, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[0:5], v[84:89], a[0:3], v168, v170 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(8) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[172:173], v217 offset:2048
	ds_read_b64 v[174:175], v218 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[12:17], v[84:89], a[4:7], v168, v170 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_i32 s9, s70, 0xf0000
	s_mov_b32 m0, s78
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:23], v[84:89], a[8:11], v168, v170 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen lds
	ds_read_b64 v[6:7], v179 offset:20768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[24:29], v[84:89], a[12:15], v168, v170 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[8:9], v179 offset:20776
	ds_read_b64 v[10:11], v179 offset:20784
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[30:35], v[84:89], a[16:19], v169, v170 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[54:55], v179 offset:20896
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[36:41], v[84:89], a[20:23], v169, v170 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[56:57], v179 offset:20904
	ds_read_b64 v[58:59], v179 offset:20912
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[42:47], v[84:89], a[24:27], v169, v170 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen offset:1024 lds
	ds_read_b64 v[60:61], v179 offset:21024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[48:53], v[84:89], a[28:31], v169, v170 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[62:63], v179 offset:21032
	ds_read_b64 v[64:65], v179 offset:21040
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[48:53], v[102:107], a[32:35], v169, v170 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[66:67], v179 offset:21152
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[42:47], v[102:107], a[36:39], v169, v170 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[68:69], v179 offset:21160
	ds_read_b64 v[70:71], v179 offset:21168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[36:41], v[102:107], a[40:43], v169, v170 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen offset:2048 lds
	ds_read_b64 v[72:73], v179 offset:21280
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[30:35], v[102:107], a[44:47], v169, v170 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[74:75], v179 offset:21288
	ds_read_b64 v[76:77], v179 offset:21296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[24:29], v[102:107], a[48:51], v168, v170 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[78:79], v179 offset:21408
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[18:23], v[102:107], a[52:55], v168, v170 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[80:81], v179 offset:21416
	ds_read_b64 v[82:83], v179 offset:21424
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[12:17], v[102:107], a[56:59], v168, v170 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen offset:3072 lds
	ds_read_b64 v[84:85], v179 offset:21536
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[0:5], v[102:107], a[60:63], v168, v170 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[86:87], v179 offset:21544
	ds_read_b64 v[88:89], v179 offset:21552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[0:5], v[108:113], a[64:67], v168, v170 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[90:91], v179 offset:21664
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[12:17], v[108:113], a[68:71], v168, v170 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[92:93], v179 offset:21672
	ds_read_b64 v[94:95], v179 offset:21680
	s_add_i32 s9, s70, 0xf1000
	s_mov_b32 m0, s79
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[18:23], v[108:113], a[72:75], v168, v170 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen lds
	ds_read_b128 v[96:99], v220 offset:28672
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[24:29], v[108:113], a[76:79], v168, v170 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[100:101], v221 offset:45056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[30:35], v[108:113], a[80:83], v169, v170 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[102:105], v220 offset:29696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[36:41], v[108:113], a[84:87], v169, v170 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[106:107], v221 offset:45568
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[42:47], v[108:113], a[88:91], v169, v170 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen offset:1024 lds
	ds_read_b128 v[126:129], v220 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[48:53], v[108:113], a[92:95], v169, v170 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[130:131], v221 offset:46080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[48:53], v[114:119], a[96:99], v169, v170 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[108:111], v220 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[42:47], v[114:119], a[100:103], v169, v170 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[112:113], v221 offset:46592
	s_add_i32 s8, s65, s8
	s_mov_b32 m0, s65
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[36:41], v[114:119], a[104:107], v169, v170 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[4:7], s8 offen lds
	ds_read_b128 v[150:153], v220 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[30:35], v[114:119], a[108:111], v169, v170 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[154:155], v221 offset:47104
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[24:29], v[114:119], a[112:115], v168, v170 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[156:159], v220 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[18:23], v[114:119], a[116:119], v168, v170 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[160:161], v221 offset:47616
	s_add_i32 s8, s80, 0x9fc00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[12:17], v[114:119], a[120:123], v168, v170 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[12:15], s8 offen offset:1024 lds
	ds_read_b128 v[162:165], v220 offset:34816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[0:5], v[114:119], a[124:127], v168, v170 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[166:167], v221 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[0:5], v[132:137], a[128:131], v168, v171 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[114:117], v220 offset:35840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[12:17], v[132:137], a[132:135], v168, v171 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[118:119], v221 offset:48640
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[18:23], v[132:137], a[136:139], v168, v171 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[24:29], v[132:137], a[140:143], v168, v171 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s8, s57, 0xf80
	s_mov_b32 m0, s81
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[30:35], v[132:137], a[144:147], v169, v171 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[36:41], v[132:137], a[148:151], v169, v171 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[42:47], v[132:137], a[152:155], v169, v171 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s8, s58, 0xf80
	s_mov_b32 m0, s82
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[48:53], v[132:137], a[156:159], v169, v171 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[48:53], v[138:143], a[160:163], v169, v171 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[42:47], v[138:143], a[164:167], v169, v171 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[36:41], v[138:143], a[168:171], v169, v171 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_i32 s8, s59, 0xf80
	s_mov_b32 m0, s83
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[30:35], v[138:143], a[172:175], v169, v171 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[24:29], v[138:143], a[176:179], v168, v171 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[18:23], v[138:143], a[180:183], v168, v171 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s8, s60, 0xf80
	s_mov_b32 m0, s84
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[12:17], v[138:143], a[184:187], v168, v171 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[0:5], v[138:143], a[188:191], v168, v171 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[0:5], v[144:149], a[192:195], v168, v171 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[12:17], v[144:149], a[196:199], v168, v171 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_add_i32 s8, s61, 0xf80
	s_mov_b32 m0, s85
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[18:23], v[144:149], a[200:203], v168, v171 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[24:29], v[144:149], a[204:207], v168, v171 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[30:35], v[144:149], a[208:211], v169, v171 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_add_i32 s8, s64, 0xf80
	s_mov_b32 m0, s86
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[36:41], v[144:149], a[212:215], v169, v171 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[42:47], v[144:149], a[216:219], v169, v171 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[48:53], v[144:149], a[220:223], v169, v171 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[48:53], v[120:125], a[224:227], v169, v171 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_i32 s8, s68, 0xf80
	s_mov_b32 m0, s87
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[42:47], v[120:125], a[228:231], v169, v171 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[36:41], v[120:125], a[232:235], v169, v171 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[30:35], v[120:125], a[236:239], v169, v171 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_add_i32 s8, s69, 0xf80
	s_mov_b32 m0, s88
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[24:29], v[120:125], a[240:243], v168, v171 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[18:23], v[120:125], a[244:247], v168, v171 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[12:17], v[120:125], a[248:251], v168, v171 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[0:5], v[120:125], a[252:255], v168, v171 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_add_i32 s8, s66, 0x520
	s_lshl_b32 s8, s8, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[6:11], v[96:101], a[0:3], v172, v174 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(8) lgkmcnt(0)
	s_barrier
	ds_read_b128 v[12:15], v222 offset:53248
	ds_read_b128 v[16:19], v222 offset:53264
	ds_read_b64 v[168:169], v217
	ds_read_b64 v[170:171], v218 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[54:59], v[96:101], a[4:7], v172, v174 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_i32 s9, s70, 0xf6000
	s_mov_b32 m0, s89
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[60:65], v[96:101], a[8:11], v172, v174 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen lds
	ds_read_b64 v[0:1], v219 offset:53248
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:71], v[96:101], a[12:15], v172, v174 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[2:3], v219 offset:53256
	ds_read_b64 v[4:5], v219 offset:53264
	s_waitcnt lgkmcnt(6)
	ds_write_b64 v223, v[12:13] offset:53248
	ds_write_b64 v223, v[14:15] offset:53256
	s_waitcnt lgkmcnt(7)
	ds_write_b64 v223, v[16:17] offset:53264
	ds_write_b64 v223, v[18:19] offset:53272
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[72:77], v[96:101], a[16:19], v173, v174 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[12:13], v219 offset:53376
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[78:83], v[96:101], a[20:23], v173, v174 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[14:15], v219 offset:53384
	ds_read_b64 v[16:17], v219 offset:53392
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[84:89], v[96:101], a[24:27], v173, v174 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen offset:1024 lds
	ds_read_b64 v[18:19], v219 offset:53504
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[90:95], v[96:101], a[28:31], v173, v174 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[20:21], v219 offset:53512
	ds_read_b64 v[22:23], v219 offset:53520
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[90:95], v[102:107], a[32:35], v173, v174 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[24:25], v219 offset:53632
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[84:89], v[102:107], a[36:39], v173, v174 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[26:27], v219 offset:53640
	ds_read_b64 v[28:29], v219 offset:53648
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[78:83], v[102:107], a[40:43], v173, v174 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen offset:2048 lds
	ds_read_b64 v[30:31], v219 offset:53760
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[72:77], v[102:107], a[44:47], v173, v174 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[32:33], v219 offset:53768
	ds_read_b64 v[34:35], v219 offset:53776
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[66:71], v[102:107], a[48:51], v172, v174 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[36:37], v219 offset:53888
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[60:65], v[102:107], a[52:55], v172, v174 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[38:39], v219 offset:53896
	ds_read_b64 v[40:41], v219 offset:53904
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[54:59], v[102:107], a[56:59], v172, v174 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen offset:3072 lds
	ds_read_b64 v[42:43], v219 offset:54016
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[6:11], v[102:107], a[60:63], v172, v174 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[44:45], v219 offset:54024
	ds_read_b64 v[46:47], v219 offset:54032
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[6:11], v[126:131], a[64:67], v172, v174 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[48:49], v219 offset:54144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[54:59], v[126:131], a[68:71], v172, v174 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[50:51], v219 offset:54152
	ds_read_b64 v[52:53], v219 offset:54160
	s_add_i32 s9, s70, 0xf7000
	s_mov_b32 m0, s90
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[60:65], v[126:131], a[72:75], v172, v174 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen lds
	ds_read_b128 v[96:99], v220 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[66:71], v[126:131], a[76:79], v172, v174 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[100:101], v221 offset:20480
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[72:77], v[126:131], a[80:83], v173, v174 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[102:105], v220 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[78:83], v[126:131], a[84:87], v173, v174 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[106:107], v221 offset:20992
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[84:89], v[126:131], a[88:91], v173, v174 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen offset:1024 lds
	ds_read_b128 v[120:123], v220 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[90:95], v[126:131], a[92:95], v173, v174 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[124:125], v221 offset:21504
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[90:95], v[108:113], a[96:99], v173, v174 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[126:129], v220 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[84:89], v[108:113], a[100:103], v173, v174 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[130:131], v221 offset:22016
	s_add_i32 s8, s65, s8
	s_mov_b32 m0, s91
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[78:83], v[108:113], a[104:107], v173, v174 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[4:7], s8 offen lds
	ds_read_b128 v[132:135], v220 offset:8192
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[72:77], v[108:113], a[108:111], v173, v174 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[136:137], v221 offset:22528
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[66:71], v[108:113], a[112:115], v172, v174 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[138:141], v220 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[60:65], v[108:113], a[116:119], v172, v174 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[142:143], v221 offset:23040
	s_add_i32 s8, s80, 0xa3c00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[54:59], v[108:113], a[120:123], v172, v174 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[12:15], s8 offen offset:1024 lds
	ds_read_b128 v[144:147], v220 offset:10240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[6:11], v[108:113], a[124:127], v172, v174 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[148:149], v221 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[6:11], v[150:155], a[128:131], v172, v175 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[108:111], v220 offset:11264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[54:59], v[150:155], a[132:135], v172, v175 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[112:113], v221 offset:24064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[60:65], v[150:155], a[136:139], v172, v175 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[66:71], v[150:155], a[140:143], v172, v175 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s8, s57, 0x1000
	s_mov_b32 m0, s67
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[72:77], v[150:155], a[144:147], v173, v175 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[78:83], v[150:155], a[148:151], v173, v175 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[84:89], v[150:155], a[152:155], v173, v175 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s8, s58, 0x1000
	s_mov_b32 m0, s71
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[90:95], v[150:155], a[156:159], v173, v175 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[90:95], v[156:161], a[160:163], v173, v175 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[84:89], v[156:161], a[164:167], v173, v175 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[78:83], v[156:161], a[168:171], v173, v175 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_i32 s8, s59, 0x1000
	s_mov_b32 m0, s72
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[72:77], v[156:161], a[172:175], v173, v175 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[66:71], v[156:161], a[176:179], v172, v175 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[60:65], v[156:161], a[180:183], v172, v175 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s8, s60, 0x1000
	s_mov_b32 m0, s73
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[54:59], v[156:161], a[184:187], v172, v175 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[6:11], v[156:161], a[188:191], v172, v175 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[6:11], v[162:167], a[192:195], v172, v175 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[54:59], v[162:167], a[196:199], v172, v175 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_add_i32 s8, s61, 0x1000
	s_mov_b32 m0, s74
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[60:65], v[162:167], a[200:203], v172, v175 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[66:71], v[162:167], a[204:207], v172, v175 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[72:77], v[162:167], a[208:211], v173, v175 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_add_i32 s8, s64, 0x1000
	s_mov_b32 m0, s75
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[78:83], v[162:167], a[212:215], v173, v175 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[84:89], v[162:167], a[216:219], v173, v175 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[90:95], v[162:167], a[220:223], v173, v175 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[90:95], v[114:119], a[224:227], v173, v175 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_i32 s8, s68, 0x1000
	s_mov_b32 m0, s76
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[84:89], v[114:119], a[228:231], v173, v175 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[78:83], v[114:119], a[232:235], v173, v175 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[72:77], v[114:119], a[236:239], v173, v175 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_add_i32 s8, s69, 0x1000
	s_mov_b32 m0, s77
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[66:71], v[114:119], a[240:243], v172, v175 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[60:65], v[114:119], a[244:247], v172, v175 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[54:59], v[114:119], a[248:251], v172, v175 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[6:11], v[114:119], a[252:255], v172, v175 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_add_i32 s8, s66, 0x540
	s_lshl_b32 s8, s8, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[0:5], v[96:101], a[0:3], v168, v170 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(8) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[172:173], v217 offset:2048
	ds_read_b64 v[174:175], v218 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[12:17], v[96:101], a[4:7], v168, v170 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_i32 s9, s70, 0xfc000
	s_mov_b32 m0, s78
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:23], v[96:101], a[8:11], v168, v170 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen lds
	ds_read_b64 v[6:7], v224 offset:53344
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[24:29], v[96:101], a[12:15], v168, v170 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[8:9], v225 offset:53352
	ds_read_b64 v[10:11], v225 offset:53360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[30:35], v[96:101], a[16:19], v169, v170 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[54:55], v224 offset:53472
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[36:41], v[96:101], a[20:23], v169, v170 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[56:57], v225 offset:53480
	ds_read_b64 v[58:59], v225 offset:53488
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[42:47], v[96:101], a[24:27], v169, v170 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen offset:1024 lds
	ds_read_b64 v[60:61], v224 offset:53600
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[48:53], v[96:101], a[28:31], v169, v170 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[62:63], v225 offset:53608
	ds_read_b64 v[64:65], v225 offset:53616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[48:53], v[102:107], a[32:35], v169, v170 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[66:67], v224 offset:53728
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[42:47], v[102:107], a[36:39], v169, v170 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[68:69], v225 offset:53736
	ds_read_b64 v[70:71], v225 offset:53744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[36:41], v[102:107], a[40:43], v169, v170 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen offset:2048 lds
	ds_read_b64 v[72:73], v224 offset:53856
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[30:35], v[102:107], a[44:47], v169, v170 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[74:75], v225 offset:53864
	ds_read_b64 v[76:77], v225 offset:53872
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[24:29], v[102:107], a[48:51], v168, v170 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[78:79], v224 offset:53984
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[18:23], v[102:107], a[52:55], v168, v170 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[80:81], v225 offset:53992
	ds_read_b64 v[82:83], v225 offset:54000
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[12:17], v[102:107], a[56:59], v168, v170 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen offset:3072 lds
	ds_read_b64 v[84:85], v224 offset:54112
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[0:5], v[102:107], a[60:63], v168, v170 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[86:87], v225 offset:54120
	ds_read_b64 v[88:89], v225 offset:54128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[0:5], v[120:125], a[64:67], v168, v170 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[90:91], v224 offset:54240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[12:17], v[120:125], a[68:71], v168, v170 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[92:93], v225 offset:54248
	ds_read_b64 v[94:95], v225 offset:54256
	s_add_i32 s9, s70, 0xfd000
	s_mov_b32 m0, s79
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[18:23], v[120:125], a[72:75], v168, v170 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen lds
	ds_read_b128 v[96:99], v220 offset:28672
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[24:29], v[120:125], a[76:79], v168, v170 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[100:101], v221 offset:45056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[30:35], v[120:125], a[80:83], v169, v170 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[102:105], v220 offset:29696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[36:41], v[120:125], a[84:87], v169, v170 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[106:107], v221 offset:45568
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[42:47], v[120:125], a[88:91], v169, v170 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen offset:1024 lds
	ds_read_b128 v[114:117], v220 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[48:53], v[120:125], a[92:95], v169, v170 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[118:119], v221 offset:46080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[48:53], v[126:131], a[96:99], v169, v170 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[120:123], v220 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[42:47], v[126:131], a[100:103], v169, v170 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[124:125], v221 offset:46592
	s_add_i32 s8, s65, s8
	s_mov_b32 m0, s65
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[36:41], v[126:131], a[104:107], v169, v170 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[4:7], s8 offen lds
	ds_read_b128 v[150:153], v220 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[30:35], v[126:131], a[108:111], v169, v170 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[154:155], v221 offset:47104
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[24:29], v[126:131], a[112:115], v168, v170 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[156:159], v220 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[18:23], v[126:131], a[116:119], v168, v170 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[160:161], v221 offset:47616
	s_add_i32 s8, s80, 0xa7c00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[12:17], v[126:131], a[120:123], v168, v170 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[12:15], s8 offen offset:1024 lds
	ds_read_b128 v[162:165], v220 offset:34816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[0:5], v[126:131], a[124:127], v168, v170 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[166:167], v221 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[0:5], v[132:137], a[128:131], v168, v171 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[126:129], v220 offset:35840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[12:17], v[132:137], a[132:135], v168, v171 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[130:131], v221 offset:48640
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[18:23], v[132:137], a[136:139], v168, v171 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[24:29], v[132:137], a[140:143], v168, v171 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[30:35], v[132:137], a[144:147], v169, v171 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[36:41], v[132:137], a[148:151], v169, v171 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[42:47], v[132:137], a[152:155], v169, v171 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[48:53], v[132:137], a[156:159], v169, v171 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[48:53], v[138:143], a[160:163], v169, v171 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[42:47], v[138:143], a[164:167], v169, v171 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[36:41], v[138:143], a[168:171], v169, v171 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[30:35], v[138:143], a[172:175], v169, v171 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[24:29], v[138:143], a[176:179], v168, v171 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[18:23], v[138:143], a[180:183], v168, v171 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[12:17], v[138:143], a[184:187], v168, v171 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[0:5], v[138:143], a[188:191], v168, v171 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[0:5], v[144:149], a[192:195], v168, v171 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[12:17], v[144:149], a[196:199], v168, v171 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[18:23], v[144:149], a[200:203], v168, v171 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[24:29], v[144:149], a[204:207], v168, v171 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[30:35], v[144:149], a[208:211], v169, v171 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[36:41], v[144:149], a[212:215], v169, v171 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[42:47], v[144:149], a[216:219], v169, v171 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[48:53], v[144:149], a[220:223], v169, v171 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[48:53], v[108:113], a[224:227], v169, v171 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[42:47], v[108:113], a[228:231], v169, v171 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[36:41], v[108:113], a[232:235], v169, v171 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[30:35], v[108:113], a[236:239], v169, v171 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[24:29], v[108:113], a[240:243], v168, v171 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[18:23], v[108:113], a[244:247], v168, v171 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[12:17], v[108:113], a[248:251], v168, v171 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[0:5], v[108:113], a[252:255], v168, v171 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_add_i32 s8, s66, 0x560
	s_lshl_b32 s8, s8, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[6:11], v[96:101], a[0:3], v172, v174 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[168:169], v217
	ds_read_b64 v[170:171], v218 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[54:59], v[96:101], a[4:7], v172, v174 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_i32 s9, s70, 0x102000
	s_mov_b32 m0, s89
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[60:65], v[96:101], a[8:11], v172, v174 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen lds
	ds_read_b64 v[0:1], v176 offset:20800
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:71], v[96:101], a[12:15], v172, v174 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[2:3], v176 offset:20808
	ds_read_b64 v[4:5], v177 offset:20816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[72:77], v[96:101], a[16:19], v173, v174 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[12:13], v176 offset:20928
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[78:83], v[96:101], a[20:23], v173, v174 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[14:15], v176 offset:20936
	ds_read_b64 v[16:17], v177 offset:20944
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[84:89], v[96:101], a[24:27], v173, v174 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen offset:1024 lds
	ds_read_b64 v[18:19], v176 offset:21056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[90:95], v[96:101], a[28:31], v173, v174 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[20:21], v176 offset:21064
	ds_read_b64 v[22:23], v177 offset:21072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[90:95], v[102:107], a[32:35], v173, v174 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[24:25], v176 offset:21184
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[84:89], v[102:107], a[36:39], v173, v174 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[26:27], v176 offset:21192
	ds_read_b64 v[28:29], v177 offset:21200
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[78:83], v[102:107], a[40:43], v173, v174 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen offset:2048 lds
	ds_read_b64 v[30:31], v176 offset:21312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[72:77], v[102:107], a[44:47], v173, v174 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[32:33], v176 offset:21320
	ds_read_b64 v[34:35], v177 offset:21328
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[66:71], v[102:107], a[48:51], v172, v174 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[36:37], v176 offset:21440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[60:65], v[102:107], a[52:55], v172, v174 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[38:39], v176 offset:21448
	ds_read_b64 v[40:41], v177 offset:21456
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[54:59], v[102:107], a[56:59], v172, v174 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen offset:3072 lds
	ds_read_b64 v[42:43], v176 offset:21568
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[6:11], v[102:107], a[60:63], v172, v174 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[44:45], v176 offset:21576
	ds_read_b64 v[46:47], v177 offset:21584
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[6:11], v[114:119], a[64:67], v172, v174 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[48:49], v176 offset:21696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[54:59], v[114:119], a[68:71], v172, v174 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[50:51], v176 offset:21704
	ds_read_b64 v[52:53], v177 offset:21712
	s_add_i32 s9, s70, 0x103000
	s_mov_b32 m0, s90
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[60:65], v[114:119], a[72:75], v172, v174 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen lds
	ds_read_b128 v[96:99], v220 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[66:71], v[114:119], a[76:79], v172, v174 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[100:101], v221 offset:20480
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[72:77], v[114:119], a[80:83], v173, v174 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[102:105], v220 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[78:83], v[114:119], a[84:87], v173, v174 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[106:107], v221 offset:20992
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[84:89], v[114:119], a[88:91], v173, v174 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen offset:1024 lds
	ds_read_b128 v[108:111], v220 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[90:95], v[114:119], a[92:95], v173, v174 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[112:113], v221 offset:21504
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[90:95], v[120:125], a[96:99], v173, v174 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[114:117], v220 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[84:89], v[120:125], a[100:103], v173, v174 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[118:119], v221 offset:22016
	s_add_i32 s8, s65, s8
	s_mov_b32 m0, s91
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[78:83], v[120:125], a[104:107], v173, v174 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[4:7], s8 offen lds
	ds_read_b128 v[132:135], v220 offset:8192
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[72:77], v[120:125], a[108:111], v173, v174 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[136:137], v221 offset:22528
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[66:71], v[120:125], a[112:115], v172, v174 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[138:141], v220 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[60:65], v[120:125], a[116:119], v172, v174 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[142:143], v221 offset:23040
	s_add_i32 s8, s80, 0xabc00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[54:59], v[120:125], a[120:123], v172, v174 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[12:15], s8 offen offset:1024 lds
	ds_read_b128 v[144:147], v220 offset:10240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[6:11], v[120:125], a[124:127], v172, v174 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[148:149], v221 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[6:11], v[150:155], a[128:131], v172, v175 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[120:123], v220 offset:11264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[54:59], v[150:155], a[132:135], v172, v175 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[124:125], v221 offset:24064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[60:65], v[150:155], a[136:139], v172, v175 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[66:71], v[150:155], a[140:143], v172, v175 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s8, s57, 0x1080
	s_mov_b32 m0, s81
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[72:77], v[150:155], a[144:147], v173, v175 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[78:83], v[150:155], a[148:151], v173, v175 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[84:89], v[150:155], a[152:155], v173, v175 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s8, s58, 0x1080
	s_mov_b32 m0, s82
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[90:95], v[150:155], a[156:159], v173, v175 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[90:95], v[156:161], a[160:163], v173, v175 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[84:89], v[156:161], a[164:167], v173, v175 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[78:83], v[156:161], a[168:171], v173, v175 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_i32 s8, s59, 0x1080
	s_mov_b32 m0, s83
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[72:77], v[156:161], a[172:175], v173, v175 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[66:71], v[156:161], a[176:179], v172, v175 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[60:65], v[156:161], a[180:183], v172, v175 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s8, s60, 0x1080
	s_mov_b32 m0, s84
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[54:59], v[156:161], a[184:187], v172, v175 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[6:11], v[156:161], a[188:191], v172, v175 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[6:11], v[162:167], a[192:195], v172, v175 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[54:59], v[162:167], a[196:199], v172, v175 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_add_i32 s8, s61, 0x1080
	s_mov_b32 m0, s85
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[60:65], v[162:167], a[200:203], v172, v175 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[66:71], v[162:167], a[204:207], v172, v175 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[72:77], v[162:167], a[208:211], v173, v175 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_add_i32 s8, s64, 0x1080
	s_mov_b32 m0, s86
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[78:83], v[162:167], a[212:215], v173, v175 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[84:89], v[162:167], a[216:219], v173, v175 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[90:95], v[162:167], a[220:223], v173, v175 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[90:95], v[126:131], a[224:227], v173, v175 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_i32 s8, s68, 0x1080
	s_mov_b32 m0, s87
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[84:89], v[126:131], a[228:231], v173, v175 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[78:83], v[126:131], a[232:235], v173, v175 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[72:77], v[126:131], a[236:239], v173, v175 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_add_i32 s8, s69, 0x1080
	s_mov_b32 m0, s88
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[66:71], v[126:131], a[240:243], v172, v175 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[60:65], v[126:131], a[244:247], v172, v175 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[54:59], v[126:131], a[248:251], v172, v175 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[6:11], v[126:131], a[252:255], v172, v175 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_add_i32 s8, s66, 0x580
	s_lshl_b32 s8, s8, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[0:5], v[96:101], a[0:3], v168, v170 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(8) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[172:173], v217 offset:2048
	ds_read_b64 v[174:175], v218 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[12:17], v[96:101], a[4:7], v168, v170 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_i32 s9, s70, 0x108000
	s_mov_b32 m0, s78
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:23], v[96:101], a[8:11], v168, v170 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen lds
	ds_read_b64 v[6:7], v219 offset:53280
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[24:29], v[96:101], a[12:15], v168, v170 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[8:9], v219 offset:53288
	ds_read_b64 v[10:11], v219 offset:53296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[30:35], v[96:101], a[16:19], v169, v170 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[54:55], v219 offset:53408
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[36:41], v[96:101], a[20:23], v169, v170 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[56:57], v219 offset:53416
	ds_read_b64 v[58:59], v219 offset:53424
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[42:47], v[96:101], a[24:27], v169, v170 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen offset:1024 lds
	ds_read_b64 v[60:61], v219 offset:53536
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[48:53], v[96:101], a[28:31], v169, v170 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[62:63], v219 offset:53544
	ds_read_b64 v[64:65], v219 offset:53552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[48:53], v[102:107], a[32:35], v169, v170 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[66:67], v219 offset:53664
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[42:47], v[102:107], a[36:39], v169, v170 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[68:69], v219 offset:53672
	ds_read_b64 v[70:71], v219 offset:53680
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[36:41], v[102:107], a[40:43], v169, v170 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen offset:2048 lds
	ds_read_b64 v[72:73], v219 offset:53792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[30:35], v[102:107], a[44:47], v169, v170 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[74:75], v219 offset:53800
	ds_read_b64 v[76:77], v219 offset:53808
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[24:29], v[102:107], a[48:51], v168, v170 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[78:79], v219 offset:53920
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[18:23], v[102:107], a[52:55], v168, v170 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[80:81], v219 offset:53928
	ds_read_b64 v[82:83], v219 offset:53936
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[12:17], v[102:107], a[56:59], v168, v170 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen offset:3072 lds
	ds_read_b64 v[84:85], v219 offset:54048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[0:5], v[102:107], a[60:63], v168, v170 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[86:87], v219 offset:54056
	ds_read_b64 v[88:89], v219 offset:54064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[0:5], v[108:113], a[64:67], v168, v170 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[90:91], v219 offset:54176
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[12:17], v[108:113], a[68:71], v168, v170 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[92:93], v219 offset:54184
	ds_read_b64 v[94:95], v219 offset:54192
	s_add_i32 s9, s70, 0x109000
	s_mov_b32 m0, s79
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[18:23], v[108:113], a[72:75], v168, v170 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen lds
	ds_read_b128 v[96:99], v220 offset:28672
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[24:29], v[108:113], a[76:79], v168, v170 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[100:101], v221 offset:45056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[30:35], v[108:113], a[80:83], v169, v170 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[102:105], v220 offset:29696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[36:41], v[108:113], a[84:87], v169, v170 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[106:107], v221 offset:45568
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[42:47], v[108:113], a[88:91], v169, v170 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen offset:1024 lds
	ds_read_b128 v[126:129], v220 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[48:53], v[108:113], a[92:95], v169, v170 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[130:131], v221 offset:46080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[48:53], v[114:119], a[96:99], v169, v170 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[108:111], v220 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[42:47], v[114:119], a[100:103], v169, v170 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[112:113], v221 offset:46592
	s_add_i32 s8, s65, s8
	s_mov_b32 m0, s65
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[36:41], v[114:119], a[104:107], v169, v170 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[4:7], s8 offen lds
	ds_read_b128 v[150:153], v220 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[30:35], v[114:119], a[108:111], v169, v170 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[154:155], v221 offset:47104
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[24:29], v[114:119], a[112:115], v168, v170 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[156:159], v220 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[18:23], v[114:119], a[116:119], v168, v170 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[160:161], v221 offset:47616
	s_add_i32 s8, s80, 0xafc00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[12:17], v[114:119], a[120:123], v168, v170 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[12:15], s8 offen offset:1024 lds
	ds_read_b128 v[162:165], v220 offset:34816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[0:5], v[114:119], a[124:127], v168, v170 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[166:167], v221 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[0:5], v[132:137], a[128:131], v168, v171 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[114:117], v220 offset:35840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[12:17], v[132:137], a[132:135], v168, v171 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[118:119], v221 offset:48640
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[18:23], v[132:137], a[136:139], v168, v171 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[24:29], v[132:137], a[140:143], v168, v171 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s8, s57, 0x1100
	s_mov_b32 m0, s67
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[30:35], v[132:137], a[144:147], v169, v171 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[36:41], v[132:137], a[148:151], v169, v171 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[42:47], v[132:137], a[152:155], v169, v171 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s8, s58, 0x1100
	s_mov_b32 m0, s71
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[48:53], v[132:137], a[156:159], v169, v171 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[48:53], v[138:143], a[160:163], v169, v171 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[42:47], v[138:143], a[164:167], v169, v171 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[36:41], v[138:143], a[168:171], v169, v171 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_i32 s8, s59, 0x1100
	s_mov_b32 m0, s72
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[30:35], v[138:143], a[172:175], v169, v171 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[24:29], v[138:143], a[176:179], v168, v171 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[18:23], v[138:143], a[180:183], v168, v171 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s8, s60, 0x1100
	s_mov_b32 m0, s73
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[12:17], v[138:143], a[184:187], v168, v171 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[0:5], v[138:143], a[188:191], v168, v171 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[0:5], v[144:149], a[192:195], v168, v171 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[12:17], v[144:149], a[196:199], v168, v171 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_add_i32 s8, s61, 0x1100
	s_mov_b32 m0, s74
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[18:23], v[144:149], a[200:203], v168, v171 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[24:29], v[144:149], a[204:207], v168, v171 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[30:35], v[144:149], a[208:211], v169, v171 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_add_i32 s8, s64, 0x1100
	s_mov_b32 m0, s75
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[36:41], v[144:149], a[212:215], v169, v171 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[42:47], v[144:149], a[216:219], v169, v171 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[48:53], v[144:149], a[220:223], v169, v171 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[48:53], v[120:125], a[224:227], v169, v171 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_i32 s8, s68, 0x1100
	s_mov_b32 m0, s76
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[42:47], v[120:125], a[228:231], v169, v171 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[36:41], v[120:125], a[232:235], v169, v171 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[30:35], v[120:125], a[236:239], v169, v171 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_add_i32 s8, s69, 0x1100
	s_mov_b32 m0, s77
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[24:29], v[120:125], a[240:243], v168, v171 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[18:23], v[120:125], a[244:247], v168, v171 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[12:17], v[120:125], a[248:251], v168, v171 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[0:5], v[120:125], a[252:255], v168, v171 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_add_i32 s8, s66, 0x5a0
	s_lshl_b32 s8, s8, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[6:11], v[96:101], a[0:3], v172, v174 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(8) lgkmcnt(0)
	s_barrier
	ds_read_b128 v[12:15], v178 offset:20736
	ds_read_b128 v[16:19], v178 offset:20752
	ds_read_b64 v[168:169], v217
	ds_read_b64 v[170:171], v218 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[54:59], v[96:101], a[4:7], v172, v174 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_i32 s9, s70, 0x10e000
	s_mov_b32 m0, s89
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[60:65], v[96:101], a[8:11], v172, v174 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen lds
	ds_read_b64 v[0:1], v179 offset:20736
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:71], v[96:101], a[12:15], v172, v174 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[2:3], v179 offset:20744
	ds_read_b64 v[4:5], v179 offset:20752
	s_waitcnt lgkmcnt(6)
	ds_write_b64 v223, v[12:13] offset:53248
	ds_write_b64 v223, v[14:15] offset:53256
	s_waitcnt lgkmcnt(7)
	ds_write_b64 v223, v[16:17] offset:53264
	ds_write_b64 v223, v[18:19] offset:53272
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[72:77], v[96:101], a[16:19], v173, v174 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[12:13], v179 offset:20864
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[78:83], v[96:101], a[20:23], v173, v174 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[14:15], v179 offset:20872
	ds_read_b64 v[16:17], v179 offset:20880
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[84:89], v[96:101], a[24:27], v173, v174 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen offset:1024 lds
	ds_read_b64 v[18:19], v179 offset:20992
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[90:95], v[96:101], a[28:31], v173, v174 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[20:21], v179 offset:21000
	ds_read_b64 v[22:23], v179 offset:21008
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[90:95], v[102:107], a[32:35], v173, v174 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[24:25], v179 offset:21120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[84:89], v[102:107], a[36:39], v173, v174 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[26:27], v179 offset:21128
	ds_read_b64 v[28:29], v179 offset:21136
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[78:83], v[102:107], a[40:43], v173, v174 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen offset:2048 lds
	ds_read_b64 v[30:31], v179 offset:21248
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[72:77], v[102:107], a[44:47], v173, v174 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[32:33], v179 offset:21256
	ds_read_b64 v[34:35], v179 offset:21264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[66:71], v[102:107], a[48:51], v172, v174 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[36:37], v179 offset:21376
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[60:65], v[102:107], a[52:55], v172, v174 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[38:39], v179 offset:21384
	ds_read_b64 v[40:41], v179 offset:21392
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[54:59], v[102:107], a[56:59], v172, v174 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen offset:3072 lds
	ds_read_b64 v[42:43], v179 offset:21504
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[6:11], v[102:107], a[60:63], v172, v174 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[44:45], v179 offset:21512
	ds_read_b64 v[46:47], v179 offset:21520
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[6:11], v[126:131], a[64:67], v172, v174 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[48:49], v179 offset:21632
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[54:59], v[126:131], a[68:71], v172, v174 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[50:51], v179 offset:21640
	ds_read_b64 v[52:53], v179 offset:21648
	s_add_i32 s9, s70, 0x10f000
	s_mov_b32 m0, s90
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[60:65], v[126:131], a[72:75], v172, v174 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen lds
	ds_read_b128 v[96:99], v220 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[66:71], v[126:131], a[76:79], v172, v174 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[100:101], v221 offset:20480
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[72:77], v[126:131], a[80:83], v173, v174 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[102:105], v220 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[78:83], v[126:131], a[84:87], v173, v174 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[106:107], v221 offset:20992
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[84:89], v[126:131], a[88:91], v173, v174 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen offset:1024 lds
	ds_read_b128 v[120:123], v220 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[90:95], v[126:131], a[92:95], v173, v174 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[124:125], v221 offset:21504
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[90:95], v[108:113], a[96:99], v173, v174 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[126:129], v220 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[84:89], v[108:113], a[100:103], v173, v174 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[130:131], v221 offset:22016
	s_add_i32 s8, s65, s8
	s_mov_b32 m0, s91
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[78:83], v[108:113], a[104:107], v173, v174 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[4:7], s8 offen lds
	ds_read_b128 v[132:135], v220 offset:8192
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[72:77], v[108:113], a[108:111], v173, v174 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[136:137], v221 offset:22528
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[66:71], v[108:113], a[112:115], v172, v174 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[138:141], v220 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[60:65], v[108:113], a[116:119], v172, v174 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[142:143], v221 offset:23040
	s_add_i32 s8, s80, 0xb3c00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[54:59], v[108:113], a[120:123], v172, v174 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[12:15], s8 offen offset:1024 lds
	ds_read_b128 v[144:147], v220 offset:10240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[6:11], v[108:113], a[124:127], v172, v174 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[148:149], v221 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[6:11], v[150:155], a[128:131], v172, v175 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[108:111], v220 offset:11264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[54:59], v[150:155], a[132:135], v172, v175 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[112:113], v221 offset:24064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[60:65], v[150:155], a[136:139], v172, v175 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[66:71], v[150:155], a[140:143], v172, v175 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s8, s57, 0x1180
	s_mov_b32 m0, s81
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[72:77], v[150:155], a[144:147], v173, v175 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[78:83], v[150:155], a[148:151], v173, v175 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[84:89], v[150:155], a[152:155], v173, v175 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s8, s58, 0x1180
	s_mov_b32 m0, s82
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[90:95], v[150:155], a[156:159], v173, v175 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[90:95], v[156:161], a[160:163], v173, v175 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[84:89], v[156:161], a[164:167], v173, v175 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[78:83], v[156:161], a[168:171], v173, v175 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_i32 s8, s59, 0x1180
	s_mov_b32 m0, s83
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[72:77], v[156:161], a[172:175], v173, v175 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[66:71], v[156:161], a[176:179], v172, v175 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[60:65], v[156:161], a[180:183], v172, v175 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s8, s60, 0x1180
	s_mov_b32 m0, s84
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[54:59], v[156:161], a[184:187], v172, v175 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[6:11], v[156:161], a[188:191], v172, v175 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[6:11], v[162:167], a[192:195], v172, v175 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[54:59], v[162:167], a[196:199], v172, v175 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_add_i32 s8, s61, 0x1180
	s_mov_b32 m0, s85
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[60:65], v[162:167], a[200:203], v172, v175 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[66:71], v[162:167], a[204:207], v172, v175 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[72:77], v[162:167], a[208:211], v173, v175 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_add_i32 s8, s64, 0x1180
	s_mov_b32 m0, s86
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[78:83], v[162:167], a[212:215], v173, v175 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[84:89], v[162:167], a[216:219], v173, v175 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[90:95], v[162:167], a[220:223], v173, v175 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[90:95], v[114:119], a[224:227], v173, v175 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_i32 s8, s68, 0x1180
	s_mov_b32 m0, s87
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[84:89], v[114:119], a[228:231], v173, v175 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[78:83], v[114:119], a[232:235], v173, v175 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[72:77], v[114:119], a[236:239], v173, v175 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_add_i32 s8, s69, 0x1180
	s_mov_b32 m0, s88
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[66:71], v[114:119], a[240:243], v172, v175 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v214, s[16:19], s8 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[60:65], v[114:119], a[244:247], v172, v175 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[54:59], v[114:119], a[248:251], v172, v175 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[6:11], v[114:119], a[252:255], v172, v175 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_add_i32 s8, s66, 0x5c0
	s_lshl_b32 s8, s8, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[0:5], v[96:101], a[0:3], v168, v170 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(8) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[172:173], v217 offset:2048
	ds_read_b64 v[174:175], v218 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[12:17], v[96:101], a[4:7], v168, v170 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_i32 s9, s70, 0x114000
	s_mov_b32 m0, s78
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:23], v[96:101], a[8:11], v168, v170 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen lds
	ds_read_b64 v[6:7], v180 offset:20832
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[24:29], v[96:101], a[12:15], v168, v170 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[8:9], v181 offset:20840
	ds_read_b64 v[10:11], v181 offset:20848
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[30:35], v[96:101], a[16:19], v169, v170 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[54:55], v180 offset:20960
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[36:41], v[96:101], a[20:23], v169, v170 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[56:57], v181 offset:20968
	ds_read_b64 v[58:59], v181 offset:20976
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[42:47], v[96:101], a[24:27], v169, v170 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen offset:1024 lds
	ds_read_b64 v[60:61], v180 offset:21088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[48:53], v[96:101], a[28:31], v169, v170 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[62:63], v181 offset:21096
	ds_read_b64 v[64:65], v181 offset:21104
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[48:53], v[102:107], a[32:35], v169, v170 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[66:67], v180 offset:21216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[42:47], v[102:107], a[36:39], v169, v170 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[68:69], v181 offset:21224
	ds_read_b64 v[70:71], v181 offset:21232
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[36:41], v[102:107], a[40:43], v169, v170 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen offset:2048 lds
	ds_read_b64 v[72:73], v180 offset:21344
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[30:35], v[102:107], a[44:47], v169, v170 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[74:75], v181 offset:21352
	ds_read_b64 v[76:77], v181 offset:21360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[24:29], v[102:107], a[48:51], v168, v170 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[78:79], v180 offset:21472
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[18:23], v[102:107], a[52:55], v168, v170 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[80:81], v181 offset:21480
	ds_read_b64 v[82:83], v181 offset:21488
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[12:17], v[102:107], a[56:59], v168, v170 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen offset:3072 lds
	ds_read_b64 v[84:85], v180 offset:21600
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[0:5], v[102:107], a[60:63], v168, v170 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[86:87], v181 offset:21608
	ds_read_b64 v[88:89], v181 offset:21616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[0:5], v[120:125], a[64:67], v168, v170 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[90:91], v180 offset:21728
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[12:17], v[120:125], a[68:71], v168, v170 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[92:93], v181 offset:21736
	ds_read_b64 v[94:95], v181 offset:21744
	s_add_i32 s9, s70, 0x115000
	s_mov_b32 m0, s79
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[18:23], v[120:125], a[72:75], v168, v170 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen lds
	ds_read_b128 v[96:99], v220 offset:28672
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[24:29], v[120:125], a[76:79], v168, v170 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[100:101], v221 offset:45056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[30:35], v[120:125], a[80:83], v169, v170 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[102:105], v220 offset:29696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[36:41], v[120:125], a[84:87], v169, v170 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[106:107], v221 offset:45568
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[42:47], v[120:125], a[88:91], v169, v170 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen offset:1024 lds
	ds_read_b128 v[114:117], v220 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[48:53], v[120:125], a[92:95], v169, v170 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[118:119], v221 offset:46080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[48:53], v[126:131], a[96:99], v169, v170 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[120:123], v220 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[42:47], v[126:131], a[100:103], v169, v170 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[124:125], v221 offset:46592
	s_add_i32 s8, s65, s8
	s_mov_b32 m0, s65
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[36:41], v[126:131], a[104:107], v169, v170 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[4:7], s8 offen lds
	ds_read_b128 v[150:153], v220 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[30:35], v[126:131], a[108:111], v169, v170 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[154:155], v221 offset:47104
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[24:29], v[126:131], a[112:115], v168, v170 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[156:159], v220 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[18:23], v[126:131], a[116:119], v168, v170 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[160:161], v221 offset:47616
	s_add_i32 s8, s80, 0xb7c00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[12:17], v[126:131], a[120:123], v168, v170 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[12:15], s8 offen offset:1024 lds
	ds_read_b128 v[162:165], v220 offset:34816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[0:5], v[126:131], a[124:127], v168, v170 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[166:167], v221 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[0:5], v[132:137], a[128:131], v168, v171 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[126:129], v220 offset:35840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[12:17], v[132:137], a[132:135], v168, v171 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[130:131], v221 offset:48640
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[18:23], v[132:137], a[136:139], v168, v171 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[24:29], v[132:137], a[140:143], v168, v171 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[30:35], v[132:137], a[144:147], v169, v171 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[36:41], v[132:137], a[148:151], v169, v171 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[42:47], v[132:137], a[152:155], v169, v171 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[48:53], v[132:137], a[156:159], v169, v171 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[48:53], v[138:143], a[160:163], v169, v171 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[42:47], v[138:143], a[164:167], v169, v171 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[36:41], v[138:143], a[168:171], v169, v171 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[30:35], v[138:143], a[172:175], v169, v171 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[24:29], v[138:143], a[176:179], v168, v171 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[18:23], v[138:143], a[180:183], v168, v171 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[12:17], v[138:143], a[184:187], v168, v171 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[0:5], v[138:143], a[188:191], v168, v171 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[0:5], v[144:149], a[192:195], v168, v171 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[12:17], v[144:149], a[196:199], v168, v171 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[18:23], v[144:149], a[200:203], v168, v171 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[24:29], v[144:149], a[204:207], v168, v171 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[30:35], v[144:149], a[208:211], v169, v171 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[36:41], v[144:149], a[212:215], v169, v171 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[42:47], v[144:149], a[216:219], v169, v171 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[48:53], v[144:149], a[220:223], v169, v171 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[48:53], v[108:113], a[224:227], v169, v171 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[42:47], v[108:113], a[228:231], v169, v171 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[36:41], v[108:113], a[232:235], v169, v171 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[30:35], v[108:113], a[236:239], v169, v171 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[24:29], v[108:113], a[240:243], v168, v171 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[18:23], v[108:113], a[244:247], v168, v171 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[12:17], v[108:113], a[248:251], v168, v171 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[0:5], v[108:113], a[252:255], v168, v171 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_add_i32 s8, s66, 0x5e0
	s_lshl_b32 s8, s8, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[6:11], v[96:101], a[0:3], v172, v174 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[192:193], v217
	ds_read_b64 v[194:195], v218 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[54:59], v[96:101], a[4:7], v172, v174 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_i32 s9, s70, 0x11a000
	s_mov_b32 m0, s89
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[60:65], v[96:101], a[8:11], v172, v174 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen lds
	ds_read_b64 v[0:1], v230 offset:53312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:71], v[96:101], a[12:15], v172, v174 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[2:3], v230 offset:53320
	ds_read_b64 v[4:5], v231 offset:53328
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[72:77], v[96:101], a[16:19], v173, v174 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[12:13], v230 offset:53440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[78:83], v[96:101], a[20:23], v173, v174 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[14:15], v230 offset:53448
	ds_read_b64 v[16:17], v231 offset:53456
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[84:89], v[96:101], a[24:27], v173, v174 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen offset:1024 lds
	ds_read_b64 v[18:19], v230 offset:53568
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[90:95], v[96:101], a[28:31], v173, v174 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[20:21], v230 offset:53576
	ds_read_b64 v[22:23], v231 offset:53584
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[90:95], v[102:107], a[32:35], v173, v174 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[24:25], v230 offset:53696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[84:89], v[102:107], a[36:39], v173, v174 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[26:27], v230 offset:53704
	ds_read_b64 v[28:29], v231 offset:53712
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[78:83], v[102:107], a[40:43], v173, v174 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen offset:2048 lds
	ds_read_b64 v[30:31], v230 offset:53824
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[72:77], v[102:107], a[44:47], v173, v174 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[32:33], v230 offset:53832
	ds_read_b64 v[34:35], v231 offset:53840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[66:71], v[102:107], a[48:51], v172, v174 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[36:37], v230 offset:53952
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[60:65], v[102:107], a[52:55], v172, v174 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[38:39], v230 offset:53960
	ds_read_b64 v[40:41], v231 offset:53968
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[54:59], v[102:107], a[56:59], v172, v174 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen offset:3072 lds
	ds_read_b64 v[42:43], v230 offset:54080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[6:11], v[102:107], a[60:63], v172, v174 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[44:45], v230 offset:54088
	ds_read_b64 v[46:47], v231 offset:54096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[6:11], v[114:119], a[64:67], v172, v174 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[48:49], v230 offset:54208
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[54:59], v[114:119], a[68:71], v172, v174 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[50:51], v230 offset:54216
	ds_read_b64 v[52:53], v231 offset:54224
	s_add_i32 s9, s70, 0x11b000
	s_mov_b32 m0, s90
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[60:65], v[114:119], a[72:75], v172, v174 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen lds
	ds_read_b128 v[96:99], v220 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[66:71], v[114:119], a[76:79], v172, v174 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[100:101], v221 offset:20480
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[72:77], v[114:119], a[80:83], v173, v174 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[102:105], v220 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[78:83], v[114:119], a[84:87], v173, v174 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[106:107], v221 offset:20992
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[84:89], v[114:119], a[88:91], v173, v174 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v215, s[0:3], s9 offen offset:1024 lds
	ds_read_b128 v[108:111], v220 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[90:95], v[114:119], a[92:95], v173, v174 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[112:113], v221 offset:21504
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[90:95], v[120:125], a[96:99], v173, v174 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[114:117], v220 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[84:89], v[120:125], a[100:103], v173, v174 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[118:119], v221 offset:22016
	s_add_i32 s0, s65, s8
	s_mov_b32 m0, s91
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[78:83], v[120:125], a[104:107], v173, v174 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[4:7], s0 offen lds
	ds_read_b128 v[132:135], v220 offset:8192
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[72:77], v[120:125], a[108:111], v173, v174 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[136:137], v221 offset:22528
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[66:71], v[120:125], a[112:115], v172, v174 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[138:141], v220 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[60:65], v[120:125], a[116:119], v172, v174 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[142:143], v221 offset:23040
	s_add_i32 s0, s80, 0xbbc00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[54:59], v[120:125], a[120:123], v172, v174 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v216, s[12:15], s0 offen offset:1024 lds
	ds_read_b128 v[144:147], v220 offset:10240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[6:11], v[120:125], a[124:127], v172, v174 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[148:149], v221 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[6:11], v[150:155], a[128:131], v172, v175 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[120:123], v220 offset:11264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[54:59], v[150:155], a[132:135], v172, v175 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[124:125], v221 offset:24064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[60:65], v[150:155], a[136:139], v172, v175 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[66:71], v[150:155], a[140:143], v172, v175 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[72:77], v[150:155], a[144:147], v173, v175 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[78:83], v[150:155], a[148:151], v173, v175 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[84:89], v[150:155], a[152:155], v173, v175 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[90:95], v[150:155], a[156:159], v173, v175 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[90:95], v[156:161], a[160:163], v173, v175 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[84:89], v[156:161], a[164:167], v173, v175 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[78:83], v[156:161], a[168:171], v173, v175 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[72:77], v[156:161], a[172:175], v173, v175 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[66:71], v[156:161], a[176:179], v172, v175 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[60:65], v[156:161], a[180:183], v172, v175 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[54:59], v[156:161], a[184:187], v172, v175 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[6:11], v[156:161], a[188:191], v172, v175 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[6:11], v[162:167], a[192:195], v172, v175 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[54:59], v[162:167], a[196:199], v172, v175 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[60:65], v[162:167], a[200:203], v172, v175 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[66:71], v[162:167], a[204:207], v172, v175 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[72:77], v[162:167], a[208:211], v173, v175 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[78:83], v[162:167], a[212:215], v173, v175 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[84:89], v[162:167], a[216:219], v173, v175 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[90:95], v[162:167], a[220:223], v173, v175 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[90:95], v[126:131], a[224:227], v173, v175 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[84:89], v[126:131], a[228:231], v173, v175 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[78:83], v[126:131], a[232:235], v173, v175 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[72:77], v[126:131], a[236:239], v173, v175 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[66:71], v[126:131], a[240:243], v172, v175 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[60:65], v[126:131], a[244:247], v172, v175 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[54:59], v[126:131], a[248:251], v172, v175 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[6:11], v[126:131], a[252:255], v172, v175 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[196:197], v217 offset:2048
	ds_read_b64 v[198:199], v218 offset:3072
	ds_read_b64 v[6:7], v179 offset:20768
	ds_read_b64 v[8:9], v179 offset:20776
	ds_read_b64 v[10:11], v179 offset:20784
	ds_read_b64 v[54:55], v179 offset:20896
	ds_read_b64 v[56:57], v179 offset:20904
	ds_read_b64 v[58:59], v179 offset:20912
	ds_read_b64 v[60:61], v179 offset:21024
	ds_read_b64 v[62:63], v179 offset:21032
	ds_read_b64 v[64:65], v179 offset:21040
	ds_read_b64 v[66:67], v179 offset:21152
	ds_read_b64 v[68:69], v179 offset:21160
	ds_read_b64 v[70:71], v179 offset:21168
	ds_read_b64 v[72:73], v179 offset:21280
	ds_read_b64 v[74:75], v179 offset:21288
	ds_read_b64 v[76:77], v179 offset:21296
	ds_read_b64 v[78:79], v179 offset:21408
	ds_read_b64 v[80:81], v179 offset:21416
	ds_read_b64 v[82:83], v179 offset:21424
	ds_read_b64 v[84:85], v179 offset:21536
	ds_read_b64 v[86:87], v179 offset:21544
	ds_read_b64 v[88:89], v179 offset:21552
	ds_read_b64 v[90:91], v179 offset:21664
	ds_read_b64 v[92:93], v179 offset:21672
	ds_read_b64 v[94:95], v179 offset:21680
	ds_read_b128 v[126:129], v220 offset:28672
	ds_read_b64 v[130:131], v221 offset:45056
	ds_read_b128 v[150:153], v220 offset:29696
	ds_read_b64 v[154:155], v221 offset:45568
	ds_read_b128 v[156:159], v220 offset:30720
	ds_read_b64 v[160:161], v221 offset:46080
	ds_read_b128 v[162:165], v220 offset:31744
	ds_read_b64 v[166:167], v221 offset:46592
	ds_read_b128 v[168:171], v220 offset:32768
	ds_read_b64 v[172:173], v221 offset:47104
	ds_read_b128 v[174:177], v220 offset:33792
	ds_read_b64 v[178:179], v221 offset:47616
	ds_read_b128 v[180:183], v220 offset:34816
	ds_read_b64 v[184:185], v221 offset:48128
	ds_read_b128 v[186:189], v220 offset:35840
	ds_read_b64 v[190:191], v221 offset:48640
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[0:5], v[96:101], a[0:3], v192, v194 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[0:5], v[102:107], a[60:63], v192, v194 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[0:5], v[108:113], a[64:67], v192, v194 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[0:5], v[114:119], a[124:127], v192, v194 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[0:5], v[132:137], a[128:131], v192, v195 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[0:5], v[138:143], a[188:191], v192, v195 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[0:5], v[144:149], a[192:195], v192, v195 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[0:5], v[120:125], a[252:255], v192, v195 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[6:11], v[186:191], a[252:255], v196, v199 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[6:11], v[180:185], a[192:195], v196, v199 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[6:11], v[174:179], a[188:191], v196, v199 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[6:11], v[168:173], a[128:131], v196, v199 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[6:11], v[162:167], a[124:127], v196, v198 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[6:11], v[156:161], a[64:67], v196, v198 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[6:11], v[150:155], a[60:63], v196, v198 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[6:11], v[126:131], a[0:3], v196, v198 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[12:17], v[120:125], a[248:251], v192, v195 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_and_b32_e32 v0, 1, v212
	v_sub_u32_e32 v1, v212, v0
	v_ashrrev_i32_e32 v2, 31, v1
	v_cmp_eq_i32_e64 s[4:5], v0, 1
	v_lshrrev_b32_e32 v2, 31, v2
	v_lshlrev_b32_e32 v0, 1, v0
	v_add_u32_e32 v1, v1, v2
	v_lshl_add_u32 v2, v213, 2, v0
	v_ashrrev_i32_e32 v3, 1, v1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[12:17], v[144:149], a[196:199], v192, v195 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_nop 0
	v_accvgpr_read_b32 v0, a0
	v_accvgpr_read_b32 v1, a1
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a2
	v_accvgpr_read_b32 v4, a3
	v_cvt_pk_bf16_f32 v1, v1, v4
	s_mul_i32 s6, s56, 0x440
	s_mov_b32 s7, 0x44
	v_mad_u32_u24 v2, s7, v2, v3
	v_cndmask_b32_e64 v3, v1, v0, s[4:5]
	v_mov_b32_e32 v4, 0x1000504
	v_mov_b32_e32 v5, 0x5040100
	v_cndmask_b32_e64 v200, v5, v4, s[4:5]
	v_mov_b32_e32 v4, 0x3020706
	v_mov_b32_e32 v5, 0x7060302
	v_cndmask_b32_e64 v201, v5, v4, s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]
	v_mov_b32_dpp v1, v3 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_lshl_u32 v2, s6, v2, 2
	v_lshlrev_b32_e32 v3, 2, v212
	v_lshlrev_b32_e32 v4, 4, v212
	v_perm_b32 v5, v1, v0, v200
	v_perm_b32 v0, v1, v0, v201
	v_add_u32_e32 v202, 0x1000, v2
	s_mov_b32 s0, s10
	s_and_b32 s1, s11, 0xffff
	s_mov_b32 s2, 0x7ffffff0
	s_mov_b32 s3, 0x31014000
	v_mad_u32_u24 v2, s7, v213, v3
	v_lshl_add_u32 v203, v213, 13, v4
	ds_write2_b32 v202, v5, v0 offset1:68
	v_accvgpr_read_b32 v0, a60
	v_accvgpr_read_b32 v1, a61
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a62
	v_accvgpr_read_b32 v3, a63
	v_cvt_pk_bf16_f32 v1, v1, v3
	v_cndmask_b32_e64 v3, v1, v0, s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]
	s_nop 0
	v_mov_b32_dpp v1, v3 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v3, v1, v0, v200
	v_perm_b32 v0, v1, v0, v201
	ds_write2_b32 v202, v3, v0 offset0:8 offset1:76
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[12:17], v[138:143], a[184:187], v192, v195 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v0, a64
	v_accvgpr_read_b32 v1, a65
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a66
	v_accvgpr_read_b32 v3, a67
	v_cvt_pk_bf16_f32 v1, v1, v3
	v_cndmask_b32_e64 v3, v1, v0, s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]
	s_nop 0
	v_mov_b32_dpp v1, v3 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v3, v1, v0, v200
	v_perm_b32 v0, v1, v0, v201
	ds_write2_b32 v202, v3, v0 offset0:16 offset1:84
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[12:17], v[132:137], a[132:135], v192, v195 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v0, a124
	v_accvgpr_read_b32 v1, a125
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a126
	v_accvgpr_read_b32 v3, a127
	v_cvt_pk_bf16_f32 v1, v1, v3
	v_cndmask_b32_e64 v3, v1, v0, s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]
	s_nop 0
	v_mov_b32_dpp v1, v3 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v3, v1, v0, v200
	v_perm_b32 v0, v1, v0, v201
	ds_write2_b32 v202, v3, v0 offset0:24 offset1:92
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[12:17], v[114:119], a[120:123], v192, v194 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v0, a128
	v_accvgpr_read_b32 v1, a129
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a130
	v_accvgpr_read_b32 v3, a131
	v_cvt_pk_bf16_f32 v1, v1, v3
	v_cndmask_b32_e64 v3, v1, v0, s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]
	s_nop 0
	v_mov_b32_dpp v1, v3 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v3, v1, v0, v200
	v_perm_b32 v0, v1, v0, v201
	ds_write2_b32 v202, v3, v0 offset0:32 offset1:100
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[12:17], v[108:113], a[68:71], v192, v194 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v0, a188
	v_accvgpr_read_b32 v1, a189
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a190
	v_accvgpr_read_b32 v3, a191
	v_cvt_pk_bf16_f32 v1, v1, v3
	v_cndmask_b32_e64 v3, v1, v0, s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]
	s_nop 0
	v_mov_b32_dpp v1, v3 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v3, v1, v0, v200
	v_perm_b32 v0, v1, v0, v201
	ds_write2_b32 v202, v3, v0 offset0:40 offset1:108
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[12:17], v[102:107], a[56:59], v192, v194 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v0, a192
	v_accvgpr_read_b32 v1, a193
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a194
	v_accvgpr_read_b32 v3, a195
	v_cvt_pk_bf16_f32 v1, v1, v3
	v_cndmask_b32_e64 v3, v1, v0, s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]
	s_nop 0
	v_mov_b32_dpp v1, v3 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v3, v1, v0, v200
	v_perm_b32 v0, v1, v0, v201
	ds_write2_b32 v202, v3, v0 offset0:48 offset1:116
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[12:17], v[96:101], a[4:7], v192, v194 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v0, a252
	v_accvgpr_read_b32 v1, a253
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a254
	v_accvgpr_read_b32 v3, a255
	v_cvt_pk_bf16_f32 v1, v1, v3
	v_cndmask_b32_e64 v3, v1, v0, s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]
	s_nop 0
	v_mov_b32_dpp v1, v3 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v3, v1, v0, v200
	v_perm_b32 v0, v1, v0, v201
	ds_write2_b32 v202, v3, v0 offset0:56 offset1:124
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[54:59], v[126:131], a[4:7], v196, v198 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[54:59], v[150:155], a[56:59], v196, v198 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	v_add_lshl_u32 v16, s6, v2, 2
	ds_read_b128 v[0:3], v16 offset:4096
	ds_read_b128 v[4:7], v16 offset:5184
	ds_read_b128 v[8:11], v16 offset:6272
	ds_read_b128 v[12:15], v16 offset:7360
	s_lshl_b32 s6, s62, 13
	s_lshl_b32 s7, s63, 1
	s_add_i32 s6, s6, s7
	v_add_u32_e32 v17, 0x8000, v203
	v_add_u32_e32 v204, 0x10000, v203
	v_add_u32_e32 v205, 0x18000, v203
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[54:59], v[156:161], a[68:71], v196, v198 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(3)
	buffer_store_dwordx4 v[0:3], v203, s[0:3], s6 offen nt
	s_waitcnt lgkmcnt(2)
	buffer_store_dwordx4 v[4:7], v17, s[0:3], s6 offen nt
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[8:11], v204, s[0:3], s6 offen nt
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[12:15], v205, s[0:3], s6 offen nt
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[54:59], v[162:167], a[120:123], v196, v198 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[54:59], v[168:173], a[132:135], v196, v199 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[54:59], v[174:179], a[184:187], v196, v199 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[54:59], v[180:185], a[196:199], v196, v199 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[54:59], v[186:191], a[248:251], v196, v199 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:23], v[96:101], a[8:11], v192, v194 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v0, a4
	v_accvgpr_read_b32 v1, a5
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a6
	v_accvgpr_read_b32 v2, a7
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_cndmask_b32_e64 v2, v1, v0, s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]
	s_nop 0
	v_mov_b32_dpp v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v2, v1, v0, v200
	v_perm_b32 v0, v1, v0, v201
	ds_write2_b32 v202, v2, v0 offset1:68
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[18:23], v[102:107], a[52:55], v192, v194 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v0, a56
	v_accvgpr_read_b32 v1, a57
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a58
	v_accvgpr_read_b32 v2, a59
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_cndmask_b32_e64 v2, v1, v0, s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]
	s_nop 0
	v_mov_b32_dpp v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v2, v1, v0, v200
	v_perm_b32 v0, v1, v0, v201
	ds_write2_b32 v202, v2, v0 offset0:8 offset1:76
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[18:23], v[108:113], a[72:75], v192, v194 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v0, a68
	v_accvgpr_read_b32 v1, a69
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a70
	v_accvgpr_read_b32 v2, a71
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_cndmask_b32_e64 v2, v1, v0, s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]
	s_nop 0
	v_mov_b32_dpp v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v2, v1, v0, v200
	v_perm_b32 v0, v1, v0, v201
	ds_write2_b32 v202, v2, v0 offset0:16 offset1:84
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[18:23], v[114:119], a[116:119], v192, v194 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v0, a120
	v_accvgpr_read_b32 v1, a121
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a122
	v_accvgpr_read_b32 v2, a123
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_cndmask_b32_e64 v2, v1, v0, s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]
	s_nop 0
	v_mov_b32_dpp v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v2, v1, v0, v200
	v_perm_b32 v0, v1, v0, v201
	ds_write2_b32 v202, v2, v0 offset0:24 offset1:92
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[18:23], v[132:137], a[136:139], v192, v195 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v0, a132
	v_accvgpr_read_b32 v1, a133
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a134
	v_accvgpr_read_b32 v2, a135
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_cndmask_b32_e64 v2, v1, v0, s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]
	s_nop 0
	v_mov_b32_dpp v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v2, v1, v0, v200
	v_perm_b32 v0, v1, v0, v201
	ds_write2_b32 v202, v2, v0 offset0:32 offset1:100
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[18:23], v[138:143], a[180:183], v192, v195 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v0, a184
	v_accvgpr_read_b32 v1, a185
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a186
	v_accvgpr_read_b32 v2, a187
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_cndmask_b32_e64 v2, v1, v0, s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]
	s_nop 0
	v_mov_b32_dpp v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v2, v1, v0, v200
	v_perm_b32 v0, v1, v0, v201
	ds_write2_b32 v202, v2, v0 offset0:40 offset1:108
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[18:23], v[144:149], a[200:203], v192, v195 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v0, a196
	v_accvgpr_read_b32 v1, a197
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a198
	v_accvgpr_read_b32 v2, a199
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_cndmask_b32_e64 v2, v1, v0, s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]
	s_nop 0
	v_mov_b32_dpp v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v2, v1, v0, v200
	v_perm_b32 v0, v1, v0, v201
	ds_write2_b32 v202, v2, v0 offset0:48 offset1:116
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[18:23], v[120:125], a[244:247], v192, v195 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v0, a248
	v_accvgpr_read_b32 v1, a249
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a250
	v_accvgpr_read_b32 v2, a251
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_cndmask_b32_e64 v2, v1, v0, s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]
	s_nop 0
	v_mov_b32_dpp v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v2, v1, v0, v200
	v_perm_b32 v0, v1, v0, v201
	ds_write2_b32 v202, v2, v0 offset0:56 offset1:124
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[60:65], v[186:191], a[244:247], v196, v199 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[60:65], v[180:185], a[200:203], v196, v199 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	ds_read_b128 v[0:3], v16 offset:4096
	ds_read_b128 v[4:7], v16 offset:5184
	ds_read_b128 v[8:11], v16 offset:6272
	ds_read_b128 v[12:15], v16 offset:7360
	s_add_i32 s6, s62, 16
	s_lshl_b32 s6, s6, 13
	s_add_i32 s6, s7, s6
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[60:65], v[174:179], a[180:183], v196, v199 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(3)
	buffer_store_dwordx4 v[0:3], v203, s[0:3], s6 offen nt
	s_waitcnt lgkmcnt(2)
	buffer_store_dwordx4 v[4:7], v17, s[0:3], s6 offen nt
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[8:11], v204, s[0:3], s6 offen nt
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[12:15], v205, s[0:3], s6 offen nt
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[60:65], v[168:173], a[136:139], v196, v199 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[60:65], v[162:167], a[116:119], v196, v198 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[60:65], v[156:161], a[72:75], v196, v198 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[60:65], v[150:155], a[52:55], v196, v198 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[60:65], v[126:131], a[8:11], v196, v198 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[24:29], v[120:125], a[240:243], v192, v195 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_nop 7
	s_nop 2
	v_accvgpr_read_b32 v0, a8
	v_accvgpr_read_b32 v1, a9
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a10
	v_accvgpr_read_b32 v2, a11
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_cndmask_b32_e64 v2, v1, v0, s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]
	s_nop 0
	v_mov_b32_dpp v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[24:29], v[144:149], a[204:207], v192, v195 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_perm_b32 v2, v1, v0, v200
	v_perm_b32 v0, v1, v0, v201
	ds_write2_b32 v202, v2, v0 offset1:68
	v_accvgpr_read_b32 v0, a52
	v_accvgpr_read_b32 v1, a53
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a54
	v_accvgpr_read_b32 v2, a55
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_cndmask_b32_e64 v2, v1, v0, s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]
	s_nop 0
	v_mov_b32_dpp v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v2, v1, v0, v200
	v_perm_b32 v0, v1, v0, v201
	ds_write2_b32 v202, v2, v0 offset0:8 offset1:76
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[24:29], v[138:143], a[176:179], v192, v195 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v0, a72
	v_accvgpr_read_b32 v1, a73
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a74
	v_accvgpr_read_b32 v2, a75
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_cndmask_b32_e64 v2, v1, v0, s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]
	s_nop 0
	v_mov_b32_dpp v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v2, v1, v0, v200
	v_perm_b32 v0, v1, v0, v201
	ds_write2_b32 v202, v2, v0 offset0:16 offset1:84
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[24:29], v[132:137], a[140:143], v192, v195 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v0, a116
	v_accvgpr_read_b32 v1, a117
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a118
	v_accvgpr_read_b32 v2, a119
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_cndmask_b32_e64 v2, v1, v0, s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]
	s_nop 0
	v_mov_b32_dpp v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v2, v1, v0, v200
	v_perm_b32 v0, v1, v0, v201
	ds_write2_b32 v202, v2, v0 offset0:24 offset1:92
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[24:29], v[114:119], a[112:115], v192, v194 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v0, a136
	v_accvgpr_read_b32 v1, a137
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a138
	v_accvgpr_read_b32 v2, a139
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_cndmask_b32_e64 v2, v1, v0, s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]
	s_nop 0
	v_mov_b32_dpp v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v2, v1, v0, v200
	v_perm_b32 v0, v1, v0, v201
	ds_write2_b32 v202, v2, v0 offset0:32 offset1:100
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[24:29], v[108:113], a[76:79], v192, v194 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v0, a180
	v_accvgpr_read_b32 v1, a181
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a182
	v_accvgpr_read_b32 v2, a183
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_cndmask_b32_e64 v2, v1, v0, s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]
	s_nop 0
	v_mov_b32_dpp v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v2, v1, v0, v200
	v_perm_b32 v0, v1, v0, v201
	ds_write2_b32 v202, v2, v0 offset0:40 offset1:108
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[24:29], v[102:107], a[48:51], v192, v194 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v0, a200
	v_accvgpr_read_b32 v1, a201
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a202
	v_accvgpr_read_b32 v2, a203
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_cndmask_b32_e64 v2, v1, v0, s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]
	s_nop 0
	v_mov_b32_dpp v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v2, v1, v0, v200
	v_perm_b32 v0, v1, v0, v201
	ds_write2_b32 v202, v2, v0 offset0:48 offset1:116
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[24:29], v[96:101], a[12:15], v192, v194 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v0, a244
	v_accvgpr_read_b32 v1, a245
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a246
	v_accvgpr_read_b32 v2, a247
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_cndmask_b32_e64 v2, v1, v0, s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]
	s_nop 0
	v_mov_b32_dpp v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v2, v1, v0, v200
	v_perm_b32 v0, v1, v0, v201
	ds_write2_b32 v202, v2, v0 offset0:56 offset1:124
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:71], v[126:131], a[12:15], v196, v198 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[66:71], v[150:155], a[48:51], v196, v198 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	ds_read_b128 v[0:3], v16 offset:4096
	ds_read_b128 v[4:7], v16 offset:5184
	ds_read_b128 v[8:11], v16 offset:6272
	ds_read_b128 v[12:15], v16 offset:7360
	s_add_i32 s6, s62, 32
	s_lshl_b32 s6, s6, 13
	s_add_i32 s6, s7, s6
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[66:71], v[156:161], a[76:79], v196, v198 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(3)
	buffer_store_dwordx4 v[0:3], v203, s[0:3], s6 offen nt
	s_waitcnt lgkmcnt(2)
	buffer_store_dwordx4 v[4:7], v17, s[0:3], s6 offen nt
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[8:11], v204, s[0:3], s6 offen nt
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[12:15], v205, s[0:3], s6 offen nt
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[66:71], v[162:167], a[112:115], v196, v198 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[66:71], v[168:173], a[140:143], v196, v199 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[66:71], v[174:179], a[176:179], v196, v199 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[66:71], v[180:185], a[204:207], v196, v199 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[66:71], v[186:191], a[240:243], v196, v199 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[30:35], v[96:101], a[16:19], v193, v194 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v0, a12
	v_accvgpr_read_b32 v1, a13
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a14
	v_accvgpr_read_b32 v2, a15
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_cndmask_b32_e64 v2, v1, v0, s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]
	s_nop 0
	v_mov_b32_dpp v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v2, v1, v0, v200
	v_perm_b32 v0, v1, v0, v201
	ds_write2_b32 v202, v2, v0 offset1:68
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[30:35], v[102:107], a[44:47], v193, v194 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v0, a48
	v_accvgpr_read_b32 v1, a49
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a50
	v_accvgpr_read_b32 v2, a51
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_cndmask_b32_e64 v2, v1, v0, s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]
	s_nop 0
	v_mov_b32_dpp v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v2, v1, v0, v200
	v_perm_b32 v0, v1, v0, v201
	ds_write2_b32 v202, v2, v0 offset0:8 offset1:76
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[30:35], v[108:113], a[80:83], v193, v194 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v0, a76
	v_accvgpr_read_b32 v1, a77
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a78
	v_accvgpr_read_b32 v2, a79
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_cndmask_b32_e64 v2, v1, v0, s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]
	s_nop 0
	v_mov_b32_dpp v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v2, v1, v0, v200
	v_perm_b32 v0, v1, v0, v201
	ds_write2_b32 v202, v2, v0 offset0:16 offset1:84
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[30:35], v[114:119], a[108:111], v193, v194 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v0, a112
	v_accvgpr_read_b32 v1, a113
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a114
	v_accvgpr_read_b32 v2, a115
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_cndmask_b32_e64 v2, v1, v0, s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]
	s_nop 0
	v_mov_b32_dpp v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v2, v1, v0, v200
	v_perm_b32 v0, v1, v0, v201
	ds_write2_b32 v202, v2, v0 offset0:24 offset1:92
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[30:35], v[132:137], a[144:147], v193, v195 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v0, a140
	v_accvgpr_read_b32 v1, a141
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a142
	v_accvgpr_read_b32 v2, a143
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_cndmask_b32_e64 v2, v1, v0, s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]
	s_nop 0
	v_mov_b32_dpp v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v2, v1, v0, v200
	v_perm_b32 v0, v1, v0, v201
	ds_write2_b32 v202, v2, v0 offset0:32 offset1:100
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[30:35], v[138:143], a[172:175], v193, v195 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v0, a176
	v_accvgpr_read_b32 v1, a177
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a178
	v_accvgpr_read_b32 v2, a179
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_cndmask_b32_e64 v2, v1, v0, s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]
	s_nop 0
	v_mov_b32_dpp v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v2, v1, v0, v200
	v_perm_b32 v0, v1, v0, v201
	ds_write2_b32 v202, v2, v0 offset0:40 offset1:108
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[30:35], v[144:149], a[208:211], v193, v195 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v0, a204
	v_accvgpr_read_b32 v1, a205
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a206
	v_accvgpr_read_b32 v2, a207
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_cndmask_b32_e64 v2, v1, v0, s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]
	s_nop 0
	v_mov_b32_dpp v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v2, v1, v0, v200
	v_perm_b32 v0, v1, v0, v201
	ds_write2_b32 v202, v2, v0 offset0:48 offset1:116
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[30:35], v[120:125], a[236:239], v193, v195 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v0, a240
	v_accvgpr_read_b32 v1, a241
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a242
	v_accvgpr_read_b32 v2, a243
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_cndmask_b32_e64 v2, v1, v0, s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]
	s_nop 0
	v_mov_b32_dpp v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v2, v1, v0, v200
	v_perm_b32 v0, v1, v0, v201
	ds_write2_b32 v202, v2, v0 offset0:56 offset1:124
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[72:77], v[186:191], a[236:239], v197, v199 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[72:77], v[180:185], a[208:211], v197, v199 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	ds_read_b128 v[0:3], v16 offset:4096
	ds_read_b128 v[4:7], v16 offset:5184
	ds_read_b128 v[8:11], v16 offset:6272
	ds_read_b128 v[12:15], v16 offset:7360
	s_add_i32 s6, s62, 48
	s_lshl_b32 s6, s6, 13
	s_add_i32 s6, s7, s6
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[72:77], v[174:179], a[172:175], v197, v199 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(3)
	buffer_store_dwordx4 v[0:3], v203, s[0:3], s6 offen nt
	s_waitcnt lgkmcnt(2)
	buffer_store_dwordx4 v[4:7], v17, s[0:3], s6 offen nt
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[8:11], v204, s[0:3], s6 offen nt
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[12:15], v205, s[0:3], s6 offen nt
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[72:77], v[168:173], a[144:147], v197, v199 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[72:77], v[162:167], a[108:111], v197, v198 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[72:77], v[156:161], a[80:83], v197, v198 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[72:77], v[150:155], a[44:47], v197, v198 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[72:77], v[126:131], a[16:19], v197, v198 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[36:41], v[120:125], a[232:235], v193, v195 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_nop 7
	s_nop 2
	v_accvgpr_read_b32 v0, a16
	v_accvgpr_read_b32 v1, a17
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a18
	v_accvgpr_read_b32 v2, a19
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_cndmask_b32_e64 v2, v1, v0, s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]
	s_nop 0
	v_mov_b32_dpp v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[36:41], v[144:149], a[212:215], v193, v195 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_perm_b32 v2, v1, v0, v200
	v_perm_b32 v0, v1, v0, v201
	ds_write2_b32 v202, v2, v0 offset1:68
	v_accvgpr_read_b32 v0, a44
	v_accvgpr_read_b32 v1, a45
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a46
	v_accvgpr_read_b32 v2, a47
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_cndmask_b32_e64 v2, v1, v0, s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]
	s_nop 0
	v_mov_b32_dpp v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v2, v1, v0, v200
	v_perm_b32 v0, v1, v0, v201
	ds_write2_b32 v202, v2, v0 offset0:8 offset1:76
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[36:41], v[138:143], a[168:171], v193, v195 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v0, a80
	v_accvgpr_read_b32 v1, a81
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a82
	v_accvgpr_read_b32 v2, a83
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_cndmask_b32_e64 v2, v1, v0, s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]
	s_nop 0
	v_mov_b32_dpp v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v2, v1, v0, v200
	v_perm_b32 v0, v1, v0, v201
	ds_write2_b32 v202, v2, v0 offset0:16 offset1:84
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[36:41], v[132:137], a[148:151], v193, v195 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v0, a108
	v_accvgpr_read_b32 v1, a109
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a110
	v_accvgpr_read_b32 v2, a111
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_cndmask_b32_e64 v2, v1, v0, s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]
	s_nop 0
	v_mov_b32_dpp v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v2, v1, v0, v200
	v_perm_b32 v0, v1, v0, v201
	ds_write2_b32 v202, v2, v0 offset0:24 offset1:92
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[36:41], v[114:119], a[104:107], v193, v194 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v0, a144
	v_accvgpr_read_b32 v1, a145
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a146
	v_accvgpr_read_b32 v2, a147
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_cndmask_b32_e64 v2, v1, v0, s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]
	s_nop 0
	v_mov_b32_dpp v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v2, v1, v0, v200
	v_perm_b32 v0, v1, v0, v201
	ds_write2_b32 v202, v2, v0 offset0:32 offset1:100
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[36:41], v[108:113], a[84:87], v193, v194 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v0, a172
	v_accvgpr_read_b32 v1, a173
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a174
	v_accvgpr_read_b32 v2, a175
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_cndmask_b32_e64 v2, v1, v0, s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]
	s_nop 0
	v_mov_b32_dpp v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v2, v1, v0, v200
	v_perm_b32 v0, v1, v0, v201
	ds_write2_b32 v202, v2, v0 offset0:40 offset1:108
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[36:41], v[102:107], a[40:43], v193, v194 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v0, a208
	v_accvgpr_read_b32 v1, a209
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a210
	v_accvgpr_read_b32 v2, a211
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_cndmask_b32_e64 v2, v1, v0, s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]
	s_nop 0
	v_mov_b32_dpp v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v2, v1, v0, v200
	v_perm_b32 v0, v1, v0, v201
	ds_write2_b32 v202, v2, v0 offset0:48 offset1:116
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[36:41], v[96:101], a[20:23], v193, v194 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v0, a236
	v_accvgpr_read_b32 v1, a237
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a238
	v_accvgpr_read_b32 v2, a239
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_cndmask_b32_e64 v2, v1, v0, s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]
	s_nop 0
	v_mov_b32_dpp v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v2, v1, v0, v200
	v_perm_b32 v0, v1, v0, v201
	ds_write2_b32 v202, v2, v0 offset0:56 offset1:124
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[78:83], v[126:131], a[20:23], v197, v198 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[78:83], v[150:155], a[40:43], v197, v198 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	ds_read_b128 v[0:3], v16 offset:4096
	ds_read_b128 v[4:7], v16 offset:5184
	ds_read_b128 v[8:11], v16 offset:6272
	ds_read_b128 v[12:15], v16 offset:7360
	s_add_i32 s6, s62, 64
	s_lshl_b32 s6, s6, 13
	s_add_i32 s6, s7, s6
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[78:83], v[156:161], a[84:87], v197, v198 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(3)
	buffer_store_dwordx4 v[0:3], v203, s[0:3], s6 offen nt
	s_waitcnt lgkmcnt(2)
	buffer_store_dwordx4 v[4:7], v17, s[0:3], s6 offen nt
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[8:11], v204, s[0:3], s6 offen nt
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[12:15], v205, s[0:3], s6 offen nt
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[78:83], v[162:167], a[104:107], v197, v198 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[78:83], v[168:173], a[148:151], v197, v199 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[78:83], v[174:179], a[168:171], v197, v199 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[78:83], v[180:185], a[212:215], v197, v199 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[78:83], v[186:191], a[232:235], v197, v199 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[42:47], v[96:101], a[24:27], v193, v194 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v0, a20
	v_accvgpr_read_b32 v1, a21
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a22
	v_accvgpr_read_b32 v2, a23
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_cndmask_b32_e64 v2, v1, v0, s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]
	s_nop 0
	v_mov_b32_dpp v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v2, v1, v0, v200
	v_perm_b32 v0, v1, v0, v201
	ds_write2_b32 v202, v2, v0 offset1:68
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[42:47], v[102:107], a[36:39], v193, v194 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v0, a40
	v_accvgpr_read_b32 v1, a41
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a42
	v_accvgpr_read_b32 v2, a43
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_cndmask_b32_e64 v2, v1, v0, s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]
	s_nop 0
	v_mov_b32_dpp v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v2, v1, v0, v200
	v_perm_b32 v0, v1, v0, v201
	ds_write2_b32 v202, v2, v0 offset0:8 offset1:76
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[42:47], v[108:113], a[88:91], v193, v194 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v0, a84
	v_accvgpr_read_b32 v1, a85
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a86
	v_accvgpr_read_b32 v2, a87
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_cndmask_b32_e64 v2, v1, v0, s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]
	s_nop 0
	v_mov_b32_dpp v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v2, v1, v0, v200
	v_perm_b32 v0, v1, v0, v201
	ds_write2_b32 v202, v2, v0 offset0:16 offset1:84
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[42:47], v[114:119], a[100:103], v193, v194 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v0, a104
	v_accvgpr_read_b32 v1, a105
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a106
	v_accvgpr_read_b32 v2, a107
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_cndmask_b32_e64 v2, v1, v0, s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]
	s_nop 0
	v_mov_b32_dpp v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v2, v1, v0, v200
	v_perm_b32 v0, v1, v0, v201
	ds_write2_b32 v202, v2, v0 offset0:24 offset1:92
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[42:47], v[132:137], a[152:155], v193, v195 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v0, a148
	v_accvgpr_read_b32 v1, a149
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a150
	v_accvgpr_read_b32 v2, a151
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_cndmask_b32_e64 v2, v1, v0, s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]
	s_nop 0
	v_mov_b32_dpp v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v2, v1, v0, v200
	v_perm_b32 v0, v1, v0, v201
	ds_write2_b32 v202, v2, v0 offset0:32 offset1:100
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[42:47], v[138:143], a[164:167], v193, v195 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v0, a168
	v_accvgpr_read_b32 v1, a169
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a170
	v_accvgpr_read_b32 v2, a171
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_cndmask_b32_e64 v2, v1, v0, s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]
	s_nop 0
	v_mov_b32_dpp v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v2, v1, v0, v200
	v_perm_b32 v0, v1, v0, v201
	ds_write2_b32 v202, v2, v0 offset0:40 offset1:108
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[42:47], v[144:149], a[216:219], v193, v195 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v0, a212
	v_accvgpr_read_b32 v1, a213
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a214
	v_accvgpr_read_b32 v2, a215
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_cndmask_b32_e64 v2, v1, v0, s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]
	s_nop 0
	v_mov_b32_dpp v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v2, v1, v0, v200
	v_perm_b32 v0, v1, v0, v201
	ds_write2_b32 v202, v2, v0 offset0:48 offset1:116
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[42:47], v[120:125], a[228:231], v193, v195 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v0, a232
	v_accvgpr_read_b32 v1, a233
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a234
	v_accvgpr_read_b32 v2, a235
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_cndmask_b32_e64 v2, v1, v0, s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]
	s_nop 0
	v_mov_b32_dpp v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v2, v1, v0, v200
	v_perm_b32 v0, v1, v0, v201
	ds_write2_b32 v202, v2, v0 offset0:56 offset1:124
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[84:89], v[186:191], a[228:231], v197, v199 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[84:89], v[180:185], a[216:219], v197, v199 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	ds_read_b128 v[0:3], v16 offset:4096
	ds_read_b128 v[4:7], v16 offset:5184
	ds_read_b128 v[8:11], v16 offset:6272
	ds_read_b128 v[12:15], v16 offset:7360
	s_add_i32 s6, s62, 0x50
	s_lshl_b32 s6, s6, 13
	s_add_i32 s6, s7, s6
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[84:89], v[174:179], a[164:167], v197, v199 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(3)
	buffer_store_dwordx4 v[0:3], v203, s[0:3], s6 offen nt
	s_waitcnt lgkmcnt(2)
	buffer_store_dwordx4 v[4:7], v17, s[0:3], s6 offen nt
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[8:11], v204, s[0:3], s6 offen nt
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[12:15], v205, s[0:3], s6 offen nt
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[84:89], v[168:173], a[152:155], v197, v199 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[84:89], v[162:167], a[100:103], v197, v198 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[84:89], v[156:161], a[88:91], v197, v198 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[84:89], v[150:155], a[36:39], v197, v198 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[84:89], v[126:131], a[24:27], v197, v198 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[48:53], v[120:125], a[224:227], v193, v195 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_nop 7
	s_nop 2
	v_accvgpr_read_b32 v0, a24
	v_accvgpr_read_b32 v1, a25
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a26
	v_accvgpr_read_b32 v2, a27
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_cndmask_b32_e64 v2, v1, v0, s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]
	s_nop 0
	v_mov_b32_dpp v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[48:53], v[144:149], a[220:223], v193, v195 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_perm_b32 v2, v1, v0, v200
	v_perm_b32 v0, v1, v0, v201
	ds_write2_b32 v202, v2, v0 offset1:68
	v_accvgpr_read_b32 v0, a36
	v_accvgpr_read_b32 v1, a37
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a38
	v_accvgpr_read_b32 v2, a39
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_cndmask_b32_e64 v2, v1, v0, s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]
	s_nop 0
	v_mov_b32_dpp v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v2, v1, v0, v200
	v_perm_b32 v0, v1, v0, v201
	ds_write2_b32 v202, v2, v0 offset0:8 offset1:76
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[48:53], v[138:143], a[160:163], v193, v195 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v0, a88
	v_accvgpr_read_b32 v1, a89
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a90
	v_accvgpr_read_b32 v2, a91
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_cndmask_b32_e64 v2, v1, v0, s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]
	s_nop 0
	v_mov_b32_dpp v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v2, v1, v0, v200
	v_perm_b32 v0, v1, v0, v201
	ds_write2_b32 v202, v2, v0 offset0:16 offset1:84
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[48:53], v[132:137], a[156:159], v193, v195 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v0, a100
	v_accvgpr_read_b32 v1, a101
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a102
	v_accvgpr_read_b32 v2, a103
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_cndmask_b32_e64 v2, v1, v0, s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]
	s_nop 0
	v_mov_b32_dpp v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v2, v1, v0, v200
	v_perm_b32 v0, v1, v0, v201
	ds_write2_b32 v202, v2, v0 offset0:24 offset1:92
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[48:53], v[114:119], a[96:99], v193, v194 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v0, a152
	v_accvgpr_read_b32 v1, a153
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a154
	v_accvgpr_read_b32 v2, a155
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_cndmask_b32_e64 v2, v1, v0, s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]
	s_nop 0
	v_mov_b32_dpp v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v2, v1, v0, v200
	v_perm_b32 v0, v1, v0, v201
	ds_write2_b32 v202, v2, v0 offset0:32 offset1:100
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[48:53], v[108:113], a[92:95], v193, v194 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v0, a164
	v_accvgpr_read_b32 v1, a165
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a166
	v_accvgpr_read_b32 v2, a167
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_cndmask_b32_e64 v2, v1, v0, s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]
	s_nop 0
	v_mov_b32_dpp v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v2, v1, v0, v200
	v_perm_b32 v0, v1, v0, v201
	ds_write2_b32 v202, v2, v0 offset0:40 offset1:108
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[48:53], v[102:107], a[32:35], v193, v194 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v0, a216
	v_accvgpr_read_b32 v1, a217
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a218
	v_accvgpr_read_b32 v2, a219
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_cndmask_b32_e64 v2, v1, v0, s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]
	s_nop 0
	v_mov_b32_dpp v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v2, v1, v0, v200
	v_perm_b32 v0, v1, v0, v201
	ds_write2_b32 v202, v2, v0 offset0:48 offset1:116
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[48:53], v[96:101], a[28:31], v193, v194 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v0, a228
	v_accvgpr_read_b32 v1, a229
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a230
	v_accvgpr_read_b32 v2, a231
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_cndmask_b32_e64 v2, v1, v0, s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]
	s_nop 0
	v_mov_b32_dpp v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v2, v1, v0, v200
	v_perm_b32 v0, v1, v0, v201
	ds_write2_b32 v202, v2, v0 offset0:56 offset1:124
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[90:95], v[126:131], a[28:31], v197, v198 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[90:95], v[150:155], a[32:35], v197, v198 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	ds_read_b128 v[0:3], v16 offset:4096
	ds_read_b128 v[4:7], v16 offset:5184
	ds_read_b128 v[8:11], v16 offset:6272
	ds_read_b128 v[12:15], v16 offset:7360
	s_add_i32 s6, s62, 0x60
	s_lshl_b32 s6, s6, 13
	s_add_i32 s6, s7, s6
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[90:95], v[156:161], a[92:95], v197, v198 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(3)
	buffer_store_dwordx4 v[0:3], v203, s[0:3], s6 offen nt
	s_waitcnt lgkmcnt(2)
	buffer_store_dwordx4 v[4:7], v17, s[0:3], s6 offen nt
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[8:11], v204, s[0:3], s6 offen nt
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[12:15], v205, s[0:3], s6 offen nt
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[90:95], v[162:167], a[96:99], v197, v198 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[90:95], v[168:173], a[156:159], v197, v199 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[90:95], v[174:179], a[160:163], v197, v199 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[90:95], v[180:185], a[220:223], v197, v199 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[90:95], v[186:191], a[224:227], v197, v199 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v0, a28
	v_accvgpr_read_b32 v1, a29
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a30
	v_accvgpr_read_b32 v2, a31
	v_cvt_pk_bf16_f32 v1, v1, v2
	v_accvgpr_read_b32 v2, a32
	v_accvgpr_read_b32 v3, a33
	v_cvt_pk_bf16_f32 v2, v2, v3
	v_accvgpr_read_b32 v3, a34
	v_accvgpr_read_b32 v4, a35
	v_cvt_pk_bf16_f32 v3, v3, v4
	v_accvgpr_read_b32 v4, a92
	v_accvgpr_read_b32 v5, a93
	v_cvt_pk_bf16_f32 v4, v4, v5
	v_accvgpr_read_b32 v5, a94
	v_accvgpr_read_b32 v6, a95
	v_cvt_pk_bf16_f32 v5, v5, v6
	v_accvgpr_read_b32 v6, a96
	v_accvgpr_read_b32 v7, a97
	v_cvt_pk_bf16_f32 v6, v6, v7
	v_accvgpr_read_b32 v7, a98
	v_accvgpr_read_b32 v8, a99
	v_cvt_pk_bf16_f32 v7, v7, v8
	v_accvgpr_read_b32 v8, a156
	v_accvgpr_read_b32 v9, a157
	v_cvt_pk_bf16_f32 v8, v8, v9
	v_accvgpr_read_b32 v9, a158
	v_accvgpr_read_b32 v10, a159
	v_cvt_pk_bf16_f32 v9, v9, v10
	v_accvgpr_read_b32 v10, a160
	v_accvgpr_read_b32 v11, a161
	v_cvt_pk_bf16_f32 v10, v10, v11
	v_accvgpr_read_b32 v11, a162
	v_accvgpr_read_b32 v12, a163
	v_cvt_pk_bf16_f32 v11, v11, v12
	v_accvgpr_read_b32 v12, a220
	v_accvgpr_read_b32 v13, a221
	v_cvt_pk_bf16_f32 v12, v12, v13
	v_accvgpr_read_b32 v13, a222
	v_accvgpr_read_b32 v14, a223
	v_cvt_pk_bf16_f32 v13, v13, v14
	v_accvgpr_read_b32 v14, a224
	v_accvgpr_read_b32 v15, a225
	v_cvt_pk_bf16_f32 v14, v14, v15
	v_accvgpr_read_b32 v15, a226
	v_accvgpr_read_b32 v18, a227
	v_cvt_pk_bf16_f32 v15, v15, v18
	v_cndmask_b32_e64 v18, v1, v0, s[4:5]
	v_cndmask_b32_e64 v19, v3, v2, s[4:5]
	v_cndmask_b32_e64 v20, v5, v4, s[4:5]
	v_cndmask_b32_e64 v21, v7, v6, s[4:5]
	v_cndmask_b32_e64 v22, v9, v8, s[4:5]
	v_cndmask_b32_e64 v23, v11, v10, s[4:5]
	v_cndmask_b32_e64 v24, v13, v12, s[4:5]
	v_cndmask_b32_e64 v25, v15, v14, s[4:5]
	v_cndmask_b32_e64 v0, v0, v1, s[4:5]
	v_mov_b32_dpp v1, v18 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v2, v2, v3, s[4:5]
	v_mov_b32_dpp v3, v19 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v4, v4, v5, s[4:5]
	v_mov_b32_dpp v5, v20 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v6, v6, v7, s[4:5]
	v_mov_b32_dpp v7, v21 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v9, s[4:5]
	v_mov_b32_dpp v9, v22 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v10, v10, v11, s[4:5]
	v_mov_b32_dpp v11, v23 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v12, v12, v13, s[4:5]
	v_mov_b32_dpp v13, v24 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v14, v14, v15, s[4:5]
	v_mov_b32_dpp v15, v25 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v18, v1, v0, v200
	v_perm_b32 v0, v1, v0, v201
	v_perm_b32 v1, v3, v2, v200
	v_perm_b32 v2, v3, v2, v201
	v_perm_b32 v3, v5, v4, v200
	v_perm_b32 v4, v5, v4, v201
	v_perm_b32 v5, v7, v6, v200
	v_perm_b32 v6, v7, v6, v201
	v_perm_b32 v7, v9, v8, v200
	v_perm_b32 v8, v9, v8, v201
	v_perm_b32 v9, v11, v10, v200
	v_perm_b32 v10, v11, v10, v201
	v_perm_b32 v11, v13, v12, v200
	v_perm_b32 v12, v13, v12, v201
	v_perm_b32 v13, v15, v14, v200
	v_perm_b32 v14, v15, v14, v201
	ds_write2_b32 v202, v18, v0 offset1:68
	ds_write2_b32 v202, v1, v2 offset0:8 offset1:76
	ds_write2_b32 v202, v3, v4 offset0:16 offset1:84
	ds_write2_b32 v202, v5, v6 offset0:24 offset1:92
	ds_write2_b32 v202, v7, v8 offset0:32 offset1:100
	ds_write2_b32 v202, v9, v10 offset0:40 offset1:108
	ds_write2_b32 v202, v11, v12 offset0:48 offset1:116
	ds_write2_b32 v202, v13, v14 offset0:56 offset1:124
	s_waitcnt lgkmcnt(0)
	ds_read_b128 v[0:3], v16 offset:4096
	ds_read_b128 v[4:7], v16 offset:5184
	ds_read_b128 v[8:11], v16 offset:6272
	ds_read_b128 v[12:15], v16 offset:7360
	s_add_i32 s4, s62, 0x70
	s_lshl_b32 s4, s4, 13
	s_add_i32 s4, s7, s4
	s_waitcnt lgkmcnt(3)
	buffer_store_dwordx4 v[0:3], v203, s[0:3], s4 offen nt
	s_waitcnt lgkmcnt(2)
	buffer_store_dwordx4 v[4:7], v17, s[0:3], s4 offen nt
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[8:11], v204, s[0:3], s4 offen nt
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[12:15], v205, s[0:3], s4 offen nt
	s_waitcnt lgkmcnt(0)
	s_endpgm
.Lrocke_gemm_mx_e3m2_e3m2_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s2_4096x4096x6144_c0c1_Acanonical_kb952f38bb6_end:
.size rocke_gemm_mx_e3m2_e3m2_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s2_4096x4096x6144_c0c1_Acanonical_kb952f38bb6, .Lrocke_gemm_mx_e3m2_e3m2_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s2_4096x4096x6144_c0c1_Acanonical_kb952f38bb6_end-rocke_gemm_mx_e3m2_e3m2_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s2_4096x4096x6144_c0c1_Acanonical_kb952f38bb6

.rodata
.p2align 6
.amdhsa_kernel rocke_gemm_mx_e3m2_e3m2_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s2_4096x4096x6144_c0c1_Acanonical_kb952f38bb6
  .amdhsa_group_segment_fixed_size 127520
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_kernarg_size 40
  .amdhsa_user_sgpr_count 12
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_user_sgpr_kernarg_preload_length 10
  .amdhsa_user_sgpr_kernarg_preload_offset 0
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_next_free_vgpr 496
  .amdhsa_next_free_sgpr 98
  .amdhsa_reserve_vcc 1
  .amdhsa_accum_offset 240
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
    Description: MX E3M2 GEMM at 4096x4096x6144 on tiled A, canonical B, and shuffled scales
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
    DataType: B6
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
  AssertSizeEqual: {0: 4096, 1: 4096, 2: 1, 3: 6144}
  AssertStrideAEqual: {1: 6144}
  AssertStrideBEqual: {1: 6144}
  AssertStrideDEqual: {1: 4096}
  AssertAlphaValue: 1
amdhsa.target: amdgcn-amd-amdhsa--gfx950
amdhsa.version: [ 1, 1 ]
amdhsa.kernels:
  - .name: rocke_gemm_mx_e3m2_e3m2_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s2_4096x4096x6144_c0c1_Acanonical_kb952f38bb6
    .symbol: rocke_gemm_mx_e3m2_e3m2_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s2_4096x4096x6144_c0c1_Acanonical_kb952f38bb6.kd
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
    .sgpr_count: 104
    .vgpr_count: 496
    .agpr_count: 256
    .wavefront_size: 64
...
.end_amdgpu_metadata
