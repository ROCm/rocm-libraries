// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// MX E2M3 x E2M3 GEMM for gfx950: E8M0 scales per 32 K elements, fp32 accumulation,
// bf16 D rounded to nearest even. 256 threads (wave64, 2x2 waves), macrotile 256x256x128,
// v_mfma_scale_f32_16x16x128_f8f6f4 cbsz:2 blgp:2, 153600 bytes of static LDS.
//
// Fixed problem, enforced by the predicates in custom.config:
//   M = 32768, N = 4096, K = 6144, batch 1, TransA = T, TransB = N, lda = ldb = K, D column-major
//   with ldd = 32768. alpha must be 1 (AssertAlphaValue) and beta 0 (UseBeta false): the
//   kernel takes no scalar arguments and never reads C. The grid must be exactly 2048
//   workgroups: every buffer descriptor has num_records 0x7ffffff0 and no load or store is
//   masked, so any other size reads and writes out of range. The K loop and the
//   workgroup-to-tile map are compiled for this shape. Its only arguments are the five
//   pointers in the metadata below, so it cannot take the UserArgs launch path. 10 kernarg dwords are preloaded, which keeps the workgroup id in s12.
//
// Inputs, in hipBLASLt naming. For a matrix X of R rows and K columns in this GEMM (A: R = M,
// B: R = N), with packed row r holding K fp6 codes as a 6-bit LSB-first bitstream:
//   A and B use HIPBLASLT_ORDER_COL256_R128_EXT. With br = r/256, rb = (r%256)/16,
//   r16 = r%16, s = k/128 and kg = (k%128)/32, the 24 bytes of row r covering
//   k in [128s + 32kg, +32) are split: bytes 0..15 go to base + 16*o and bytes 16..23 to
//   base + 16384 + 8*o, where o = 64*rb + 16*kg + r16 and base = (br*K/128 + s)*24576.
//   The scales of A and B use HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_128_4_EXT: the scale
//   of row r and K block kb (32 elements) is at ((s*R/128 + r/128)*64 + 16*g + r%16)*8
//   + (r%128)/16, with s = kb/4 and g = kb%4.
//   The kernel's a/sa operands are hipBLASLt B and its scale, b/sb are A and its scale.
//   both operands tiled.
//
// Assembly adapted from the provider's instruction listing:
//   - the address and encoding comments on each instruction are removed;
//   - each PC-relative branch immediate is replaced by a label at the same instruction;
//   - the provider's s_nop padding instructions after s_endpgm are dropped;
//   - the directives, kernel descriptor and metadata are written for code object v4
//     (amdhsa.version [1, 1], no hidden arguments) from the provider's kernel descriptor.
//     .amdhsa_kernarg_preload_length stays 10 so the workgroup id stays in s12.
.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.amdhsa_code_object_version 4
.text
.globl rocke_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_32768x4096x6144_c0c1_k3063b64df0
.p2align 8
.type rocke_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_32768x4096x6144_c0c1_k3063b64df0,@function
rocke_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_32768x4096x6144_c0c1_k3063b64df0:
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
	s_lshr_b32 s14, s12, 8
	s_lshl_b32 s14, s14, 8
	s_and_b32 s15, s12, 0xff
	s_lshr_b32 s15, s15, 1
	s_add_i32 s14, s14, s15
	s_and_b32 s12, s12, 1
	s_lshl_b32 s12, s12, 7
	s_add_i32 s12, s14, s12
	s_and_b32 s14, s12, 15
	s_lshr_b32 s12, s12, 4
	s_lshr_b32 s15, s12, 4
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
	s_and_b32 s12, s12, 15
	v_mul_lo_u32 v4, v4, v3
	s_lshl_b32 s12, s12, 4
	v_mul_hi_u32 v4, v3, v4
	s_add_i32 s12, s14, s12
	v_add_u32_e32 v3, v3, v4
	v_mul_hi_u32 v3, s12, v3
	v_mul_lo_u32 v4, v3, v2
	v_sub_u32_e32 v4, s12, v4
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
	s_mul_i32 s17, s0, 0x1800
	v_lshlrev_b32_e32 v1, 4, v0
	s_mul_i32 s16, s16, s14
	s_sub_i32 s12, s12, s16
	s_add_i32 s12, s15, s12
	; M=32768 is the A tile count and N=4096 the B tile count. The map
	; above emits those counts swapped; exchange them before the offsets.
	s_xor_b32 s12, s12, s14
	s_xor_b32 s14, s12, s14
	s_xor_b32 s12, s12, s14
	s_mul_i32 s15, s12, 0x120000
	s_lshl_b32 s12, s12, 8
	s_lshl_b32 s16, s14, 8
	s_lshl_b32 s1, s1, 7
	s_lshl_b32 s13, s13, 7
	s_add_u32 s20, s2, s15
	s_mul_i32 s2, s14, 0x120000
	s_addc_u32 s21, s3, 0
	s_add_u32 s24, s4, s2
	s_addc_u32 s25, s5, 0
	s_lshr_b32 s2, s1, 4
	s_lshr_b32 s3, s13, 4
	s_and_b32 s21, s21, 0xffff
	s_and_b32 s25, s25, 0xffff
	s_add_i32 s4, s12, s1
	s_add_i32 s5, s16, s13
	s_and_b32 s29, s7, 0xffff
	s_and_b32 s33, s9, 0xffff
	s_lshl_b32 s7, s0, 8
	s_ashr_i32 s9, s12, 31
	s_lshr_b32 s9, s9, 25
	s_add_i32 s9, s12, s9
	s_ashr_i32 s9, s9, 7
	s_ashr_i32 s12, s16, 31
	s_lshr_b32 s12, s12, 25
	s_add_i32 s12, s16, s12
	s_ashr_i32 s12, s12, 7
	s_lshr_b32 s1, s1, 7
	s_lshr_b32 s13, s13, 7
	s_lshl_b32 s14, s9, 9
	s_lshl_b32 s15, s12, 9
	s_add_i32 s16, 0x1800, s17
	s_mov_b32 s22, 0x7ffffff0
	s_mov_b32 s23, 0x31014000
	s_mov_b32 s26, 0x7ffffff0
	s_mov_b32 s27, 0x31014000
	s_mov_b32 s28, s6
	s_mov_b32 s30, 0x7ffffff0
	s_mov_b32 s31, 0x31014000
	s_mov_b32 s32, s8
	s_mov_b32 s34, 0x7ffffff0
	s_mov_b32 s35, 0x31014000
	s_mov_b32 m0, s16
	v_lshlrev_b32_e32 v2, 2, v0
	buffer_load_dwordx4 v1, s[20:23], s17 offen lds
	buffer_load_dwordx4 v1, s[20:23], s17 offen offset:1024 lds
	buffer_load_dwordx4 v1, s[20:23], s17 offen offset:2048 lds
	buffer_load_dwordx4 v1, s[20:23], s17 offen offset:3072 lds
	s_add_i32 s6, s17, 0x1000
	s_add_i32 s8, 0x1800, s6
	s_mov_b32 m0, s8
	s_nop 0
	buffer_load_dwordx4 v1, s[20:23], s6 offen lds
	buffer_load_dwordx4 v1, s[20:23], s6 offen offset:1024 lds
	s_add_i32 s18, 0x7800, s17
	s_mov_b32 m0, s18
	s_nop 0
	buffer_load_dwordx4 v1, s[24:27], s17 offen lds
	buffer_load_dwordx4 v1, s[24:27], s17 offen offset:1024 lds
	buffer_load_dwordx4 v1, s[24:27], s17 offen offset:2048 lds
	buffer_load_dwordx4 v1, s[24:27], s17 offen offset:3072 lds
	s_add_i32 s19, 0x7800, s6
	s_mov_b32 m0, s19
	s_nop 0
	buffer_load_dwordx4 v1, s[24:27], s6 offen lds
	buffer_load_dwordx4 v1, s[24:27], s6 offen offset:1024 lds
	s_mov_b32 m0, s7
	s_add_i32 s14, s7, s14
	buffer_load_dword v2, s[28:31], s14 offen lds
	s_add_i32 s14, 0x400, s7
	s_mov_b32 m0, s14
	s_add_i32 s15, s7, s15
	buffer_load_dword v2, s[32:35], s15 offen lds
	s_add_i32 s15, s9, 0x20
	s_lshl_b32 s15, s15, 9
	s_add_i32 s36, s12, 0x100
	s_lshl_b32 s36, s36, 9
	s_add_i32 s37, s17, 0x6000
	s_add_i32 s38, 0xd800, s17
	s_mov_b32 m0, s38
	s_nop 0
	buffer_load_dwordx4 v1, s[20:23], s37 offen lds
	buffer_load_dwordx4 v1, s[20:23], s37 offen offset:1024 lds
	buffer_load_dwordx4 v1, s[20:23], s37 offen offset:2048 lds
	buffer_load_dwordx4 v1, s[20:23], s37 offen offset:3072 lds
	s_add_i32 s39, 0xd800, s6
	s_mov_b32 m0, s39
	s_add_i32 s40, s17, 0x7000
	buffer_load_dwordx4 v1, s[20:23], s40 offen lds
	buffer_load_dwordx4 v1, s[20:23], s40 offen offset:1024 lds
	s_add_i32 s41, 0x13800, s17
	s_mov_b32 m0, s41
	s_nop 0
	buffer_load_dwordx4 v1, s[24:27], s37 offen lds
	buffer_load_dwordx4 v1, s[24:27], s37 offen offset:1024 lds
	buffer_load_dwordx4 v1, s[24:27], s37 offen offset:2048 lds
	buffer_load_dwordx4 v1, s[24:27], s37 offen offset:3072 lds
	s_add_i32 s37, 0x13800, s6
	s_mov_b32 m0, s37
	s_nop 0
	buffer_load_dwordx4 v1, s[24:27], s40 offen lds
	buffer_load_dwordx4 v1, s[24:27], s40 offen offset:1024 lds
	s_add_i32 s40, 0x800, s7
	s_mov_b32 m0, s40
	s_add_i32 s15, s7, s15
	buffer_load_dword v2, s[28:31], s15 offen lds
	s_add_i32 s15, 0xc00, s7
	s_mov_b32 m0, s15
	s_add_i32 s36, s7, s36
	buffer_load_dword v2, s[32:35], s36 offen lds
	s_add_i32 s36, s9, 0x40
	s_lshl_b32 s36, s36, 9
	s_add_i32 s42, s12, 0x200
	s_lshl_b32 s42, s42, 9
	s_add_i32 s43, s17, 0xc000
	s_add_i32 s44, 0x19800, s17
	s_mov_b32 m0, s44
	s_nop 0
	buffer_load_dwordx4 v1, s[20:23], s43 offen lds
	buffer_load_dwordx4 v1, s[20:23], s43 offen offset:1024 lds
	buffer_load_dwordx4 v1, s[20:23], s43 offen offset:2048 lds
	buffer_load_dwordx4 v1, s[20:23], s43 offen offset:3072 lds
	s_add_i32 s45, 0x19800, s6
	s_mov_b32 m0, s45
	s_add_i32 s46, s17, 0xd000
	buffer_load_dwordx4 v1, s[20:23], s46 offen lds
	buffer_load_dwordx4 v1, s[20:23], s46 offen offset:1024 lds
	s_add_i32 s47, 0x1f800, s17
	s_mov_b32 m0, s47
	s_nop 0
	buffer_load_dwordx4 v1, s[24:27], s43 offen lds
	buffer_load_dwordx4 v1, s[24:27], s43 offen offset:1024 lds
	buffer_load_dwordx4 v1, s[24:27], s43 offen offset:2048 lds
	buffer_load_dwordx4 v1, s[24:27], s43 offen offset:3072 lds
	s_add_i32 s6, 0x1f800, s6
	s_mov_b32 m0, s6
	s_nop 0
	buffer_load_dwordx4 v1, s[24:27], s46 offen lds
	buffer_load_dwordx4 v1, s[24:27], s46 offen offset:1024 lds
	s_add_i32 s43, 0x1000, s7
	s_mov_b32 m0, s43
	s_add_i32 s36, s7, s36
	buffer_load_dword v2, s[28:31], s36 offen lds
	s_add_i32 s36, 0x1400, s7
	s_mov_b32 m0, s36
	s_add_i32 s42, s7, s42
	buffer_load_dword v2, s[32:35], s42 offen lds
	s_waitcnt vmcnt(28) lgkmcnt(0)
	s_barrier
	s_lshl_b32 s1, s1, 7
	s_lshl_b32 s13, s13, 7
	s_lshl_b32 s2, s2, 6
	v_lshlrev_b32_e32 v3, 1, v0
	v_add_u32_e32 v4, s2, v0
	v_add_lshl_u32 v5, s1, v3, 2
	v_add_lshl_u32 v3, s13, v3, 2
	v_lshlrev_b32_e32 v6, 4, v4
	ds_read_b64 v[8:9], v5
	ds_read_b64 v[10:11], v3 offset:1024
	ds_read_b128 v[12:15], v6 offset:6144
	v_lshlrev_b32_e32 v4, 3, v4
	ds_read_b64 v[16:17], v4 offset:22528
	s_lshl_b32 s1, s3, 6
	v_add_u32_e32 v7, s1, v0
	v_lshlrev_b32_e32 v18, 4, v7
	ds_read_b128 v[20:23], v18 offset:30720
	v_lshlrev_b32_e32 v7, 3, v7
	ds_read_b64 v[24:25], v7 offset:47104
	ds_read_b128 v[26:29], v18 offset:31744
	ds_read_b64 v[30:31], v7 offset:47616
	ds_read_b128 v[32:35], v18 offset:32768
	ds_read_b64 v[36:37], v7 offset:48128
	ds_read_b128 v[38:41], v18 offset:33792
	ds_read_b64 v[42:43], v7 offset:48640
	ds_read_b128 v[44:47], v18 offset:34816
	ds_read_b64 v[48:49], v7 offset:49152
	ds_read_b128 v[50:53], v18 offset:35840
	ds_read_b64 v[54:55], v7 offset:49664
	ds_read_b128 v[56:59], v18 offset:36864
	ds_read_b64 v[60:61], v7 offset:50176
	ds_read_b128 v[62:65], v18 offset:37888
	ds_read_b64 v[66:67], v7 offset:50688
	ds_read_b128 v[68:71], v6 offset:7168
	ds_read_b64 v[72:73], v4 offset:23040
	ds_read_b128 v[74:77], v6 offset:8192
	ds_read_b64 v[78:79], v4 offset:23552
	ds_read_b128 v[80:83], v6 offset:9216
	ds_read_b64 v[84:85], v4 offset:24064
	ds_read_b128 v[86:89], v6 offset:10240
	ds_read_b64 v[90:91], v4 offset:24576
	ds_read_b128 v[92:95], v6 offset:11264
	ds_read_b64 v[96:97], v4 offset:25088
	ds_read_b128 v[98:101], v6 offset:12288
	ds_read_b64 v[102:103], v4 offset:25600
	ds_read_b128 v[104:107], v6 offset:13312
	ds_read_b64 v[108:109], v4 offset:26112
	s_lshl_b32 s1, s9, 9
	s_add_i32 s1, s7, s1
	s_lshl_b32 s2, s12, 9
	s_add_i32 s2, s7, s2
	s_mov_b32 s3, 0
	v_accvgpr_write_b32 a0, 0
	v_accvgpr_write_b32 a1, 0
	v_accvgpr_write_b32 a2, 0
	v_accvgpr_write_b32 a3, 0
	v_add_u32_e32 v19, 0x10000, v4
	v_accvgpr_write_b32 a4, 0
	v_accvgpr_write_b32 a5, 0
	v_accvgpr_write_b32 a6, 0
	v_accvgpr_write_b32 a7, 0
	v_add_u32_e32 v110, 0x10000, v18
	v_accvgpr_write_b32 a8, 0
	v_accvgpr_write_b32 a9, 0
	v_accvgpr_write_b32 a10, 0
	v_accvgpr_write_b32 a11, 0
	v_add_u32_e32 v111, 0x10000, v7
	v_accvgpr_write_b32 a12, 0
	v_accvgpr_write_b32 a13, 0
	v_accvgpr_write_b32 a14, 0
	v_accvgpr_write_b32 a15, 0
	v_add_u32_e32 v112, 0x10000, v6
	v_accvgpr_write_b32 a16, 0
	v_accvgpr_write_b32 a17, 0
	v_accvgpr_write_b32 a18, 0
	v_accvgpr_write_b32 a19, 0
	v_add_u32_e32 v113, 0x20000, v7
	v_accvgpr_write_b32 a20, 0
	v_accvgpr_write_b32 a21, 0
	v_accvgpr_write_b32 a22, 0
	v_accvgpr_write_b32 a23, 0
	v_add_u32_e32 v114, 0x20000, v18
	v_accvgpr_write_b32 a24, 0
	v_accvgpr_write_b32 a25, 0
	v_accvgpr_write_b32 a26, 0
	v_accvgpr_write_b32 a27, 0
	v_accvgpr_write_b32 a28, 0
	v_accvgpr_write_b32 a29, 0
	v_accvgpr_write_b32 a30, 0
	v_accvgpr_write_b32 a31, 0
	v_accvgpr_write_b32 a32, 0
	v_accvgpr_write_b32 a33, 0
	v_accvgpr_write_b32 a34, 0
	v_accvgpr_write_b32 a35, 0
	v_accvgpr_write_b32 a36, 0
	v_accvgpr_write_b32 a37, 0
	v_accvgpr_write_b32 a38, 0
	v_accvgpr_write_b32 a39, 0
	v_accvgpr_write_b32 a40, 0
	v_accvgpr_write_b32 a41, 0
	v_accvgpr_write_b32 a42, 0
	v_accvgpr_write_b32 a43, 0
	v_accvgpr_write_b32 a44, 0
	v_accvgpr_write_b32 a45, 0
	v_accvgpr_write_b32 a46, 0
	v_accvgpr_write_b32 a47, 0
	v_accvgpr_write_b32 a48, 0
	v_accvgpr_write_b32 a49, 0
	v_accvgpr_write_b32 a50, 0
	v_accvgpr_write_b32 a51, 0
	v_accvgpr_write_b32 a52, 0
	v_accvgpr_write_b32 a53, 0
	v_accvgpr_write_b32 a54, 0
	v_accvgpr_write_b32 a55, 0
	v_accvgpr_write_b32 a56, 0
	v_accvgpr_write_b32 a57, 0
	v_accvgpr_write_b32 a58, 0
	v_accvgpr_write_b32 a59, 0
	v_accvgpr_write_b32 a60, 0
	v_accvgpr_write_b32 a61, 0
	v_accvgpr_write_b32 a62, 0
	v_accvgpr_write_b32 a63, 0
	v_accvgpr_write_b32 a64, 0
	v_accvgpr_write_b32 a65, 0
	v_accvgpr_write_b32 a66, 0
	v_accvgpr_write_b32 a67, 0
	v_accvgpr_write_b32 a68, 0
	v_accvgpr_write_b32 a69, 0
	v_accvgpr_write_b32 a70, 0
	v_accvgpr_write_b32 a71, 0
	v_accvgpr_write_b32 a72, 0
	v_accvgpr_write_b32 a73, 0
	v_accvgpr_write_b32 a74, 0
	v_accvgpr_write_b32 a75, 0
	v_accvgpr_write_b32 a76, 0
	v_accvgpr_write_b32 a77, 0
	v_accvgpr_write_b32 a78, 0
	v_accvgpr_write_b32 a79, 0
	v_accvgpr_write_b32 a80, 0
	v_accvgpr_write_b32 a81, 0
	v_accvgpr_write_b32 a82, 0
	v_accvgpr_write_b32 a83, 0
	v_accvgpr_write_b32 a84, 0
	v_accvgpr_write_b32 a85, 0
	v_accvgpr_write_b32 a86, 0
	v_accvgpr_write_b32 a87, 0
	v_accvgpr_write_b32 a88, 0
	v_accvgpr_write_b32 a89, 0
	v_accvgpr_write_b32 a90, 0
	v_accvgpr_write_b32 a91, 0
	v_accvgpr_write_b32 a92, 0
	v_accvgpr_write_b32 a93, 0
	v_accvgpr_write_b32 a94, 0
	v_accvgpr_write_b32 a95, 0
	v_accvgpr_write_b32 a96, 0
	v_accvgpr_write_b32 a97, 0
	v_accvgpr_write_b32 a98, 0
	v_accvgpr_write_b32 a99, 0
	v_accvgpr_write_b32 a100, 0
	v_accvgpr_write_b32 a101, 0
	v_accvgpr_write_b32 a102, 0
	v_accvgpr_write_b32 a103, 0
	v_accvgpr_write_b32 a104, 0
	v_accvgpr_write_b32 a105, 0
	v_accvgpr_write_b32 a106, 0
	v_accvgpr_write_b32 a107, 0
	v_accvgpr_write_b32 a108, 0
	v_accvgpr_write_b32 a109, 0
	v_accvgpr_write_b32 a110, 0
	v_accvgpr_write_b32 a111, 0
	v_accvgpr_write_b32 a112, 0
	v_accvgpr_write_b32 a113, 0
	v_accvgpr_write_b32 a114, 0
	v_accvgpr_write_b32 a115, 0
	v_accvgpr_write_b32 a116, 0
	v_accvgpr_write_b32 a117, 0
	v_accvgpr_write_b32 a118, 0
	v_accvgpr_write_b32 a119, 0
	v_accvgpr_write_b32 a120, 0
	v_accvgpr_write_b32 a121, 0
	v_accvgpr_write_b32 a122, 0
	v_accvgpr_write_b32 a123, 0
	v_accvgpr_write_b32 a124, 0
	v_accvgpr_write_b32 a125, 0
	v_accvgpr_write_b32 a126, 0
	v_accvgpr_write_b32 a127, 0
	v_accvgpr_write_b32 a128, 0
	v_accvgpr_write_b32 a129, 0
	v_accvgpr_write_b32 a130, 0
	v_accvgpr_write_b32 a131, 0
	v_accvgpr_write_b32 a132, 0
	v_accvgpr_write_b32 a133, 0
	v_accvgpr_write_b32 a134, 0
	v_accvgpr_write_b32 a135, 0
	v_accvgpr_write_b32 a136, 0
	v_accvgpr_write_b32 a137, 0
	v_accvgpr_write_b32 a138, 0
	v_accvgpr_write_b32 a139, 0
	v_accvgpr_write_b32 a140, 0
	v_accvgpr_write_b32 a141, 0
	v_accvgpr_write_b32 a142, 0
	v_accvgpr_write_b32 a143, 0
	v_accvgpr_write_b32 a144, 0
	v_accvgpr_write_b32 a145, 0
	v_accvgpr_write_b32 a146, 0
	v_accvgpr_write_b32 a147, 0
	v_accvgpr_write_b32 a148, 0
	v_accvgpr_write_b32 a149, 0
	v_accvgpr_write_b32 a150, 0
	v_accvgpr_write_b32 a151, 0
	v_accvgpr_write_b32 a152, 0
	v_accvgpr_write_b32 a153, 0
	v_accvgpr_write_b32 a154, 0
	v_accvgpr_write_b32 a155, 0
	v_accvgpr_write_b32 a156, 0
	v_accvgpr_write_b32 a157, 0
	v_accvgpr_write_b32 a158, 0
	v_accvgpr_write_b32 a159, 0
	v_accvgpr_write_b32 a160, 0
	v_accvgpr_write_b32 a161, 0
	v_accvgpr_write_b32 a162, 0
	v_accvgpr_write_b32 a163, 0
	v_accvgpr_write_b32 a164, 0
	v_accvgpr_write_b32 a165, 0
	v_accvgpr_write_b32 a166, 0
	v_accvgpr_write_b32 a167, 0
	v_accvgpr_write_b32 a168, 0
	v_accvgpr_write_b32 a169, 0
	v_accvgpr_write_b32 a170, 0
	v_accvgpr_write_b32 a171, 0
	v_accvgpr_write_b32 a172, 0
	v_accvgpr_write_b32 a173, 0
	v_accvgpr_write_b32 a174, 0
	v_accvgpr_write_b32 a175, 0
	v_accvgpr_write_b32 a176, 0
	v_accvgpr_write_b32 a177, 0
	v_accvgpr_write_b32 a178, 0
	v_accvgpr_write_b32 a179, 0
	v_accvgpr_write_b32 a180, 0
	v_accvgpr_write_b32 a181, 0
	v_accvgpr_write_b32 a182, 0
	v_accvgpr_write_b32 a183, 0
	v_accvgpr_write_b32 a184, 0
	v_accvgpr_write_b32 a185, 0
	v_accvgpr_write_b32 a186, 0
	v_accvgpr_write_b32 a187, 0
	v_accvgpr_write_b32 a188, 0
	v_accvgpr_write_b32 a189, 0
	v_accvgpr_write_b32 a190, 0
	v_accvgpr_write_b32 a191, 0
	v_accvgpr_write_b32 a192, 0
	v_accvgpr_write_b32 a193, 0
	v_accvgpr_write_b32 a194, 0
	v_accvgpr_write_b32 a195, 0
	v_accvgpr_write_b32 a196, 0
	v_accvgpr_write_b32 a197, 0
	v_accvgpr_write_b32 a198, 0
	v_accvgpr_write_b32 a199, 0
	v_accvgpr_write_b32 a200, 0
	v_accvgpr_write_b32 a201, 0
	v_accvgpr_write_b32 a202, 0
	v_accvgpr_write_b32 a203, 0
	v_accvgpr_write_b32 a204, 0
	v_accvgpr_write_b32 a205, 0
	v_accvgpr_write_b32 a206, 0
	v_accvgpr_write_b32 a207, 0
	v_accvgpr_write_b32 a208, 0
	v_accvgpr_write_b32 a209, 0
	v_accvgpr_write_b32 a210, 0
	v_accvgpr_write_b32 a211, 0
	v_accvgpr_write_b32 a212, 0
	v_accvgpr_write_b32 a213, 0
	v_accvgpr_write_b32 a214, 0
	v_accvgpr_write_b32 a215, 0
	v_accvgpr_write_b32 a216, 0
	v_accvgpr_write_b32 a217, 0
	v_accvgpr_write_b32 a218, 0
	v_accvgpr_write_b32 a219, 0
	v_accvgpr_write_b32 a220, 0
	v_accvgpr_write_b32 a221, 0
	v_accvgpr_write_b32 a222, 0
	v_accvgpr_write_b32 a223, 0
	v_accvgpr_write_b32 a224, 0
	v_accvgpr_write_b32 a225, 0
	v_accvgpr_write_b32 a226, 0
	v_accvgpr_write_b32 a227, 0
	v_accvgpr_write_b32 a228, 0
	v_accvgpr_write_b32 a229, 0
	v_accvgpr_write_b32 a230, 0
	v_accvgpr_write_b32 a231, 0
	v_accvgpr_write_b32 a232, 0
	v_accvgpr_write_b32 a233, 0
	v_accvgpr_write_b32 a234, 0
	v_accvgpr_write_b32 a235, 0
	v_accvgpr_write_b32 a236, 0
	v_accvgpr_write_b32 a237, 0
	v_accvgpr_write_b32 a238, 0
	v_accvgpr_write_b32 a239, 0
	v_accvgpr_write_b32 a240, 0
	v_accvgpr_write_b32 a241, 0
	v_accvgpr_write_b32 a242, 0
	v_accvgpr_write_b32 a243, 0
	v_accvgpr_write_b32 a244, 0
	v_accvgpr_write_b32 a245, 0
	v_accvgpr_write_b32 a246, 0
	v_accvgpr_write_b32 a247, 0
	v_accvgpr_write_b32 a248, 0
	v_accvgpr_write_b32 a249, 0
	v_accvgpr_write_b32 a250, 0
	v_accvgpr_write_b32 a251, 0
	v_accvgpr_write_b32 a252, 0
	v_accvgpr_write_b32 a253, 0
	v_accvgpr_write_b32 a254, 0
	v_accvgpr_write_b32 a255, 0
.Lmain_loop:
	s_mul_i32 s13, s3, 12
	s_mul_i32 s13, s13, 0x6000
	s_waitcnt lgkmcnt(14)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[12:17], v[20:25], a[0:3], v8, v10 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[116:117], v5 offset:2048
	ds_read_b64 v[118:119], v3 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[12:17], v[26:31], a[4:7], v8, v10 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_i32 s13, s17, s13
	s_add_i32 s42, s13, 0x12000
	s_mov_b32 m0, s16
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[12:17], v[32:37], a[8:11], v8, v10 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s42 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[12:17], v[38:43], a[12:15], v8, v10 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[12:17], v[44:49], a[16:19], v8, v11 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[120:123], v6 offset:55296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[12:17], v[50:55], a[20:23], v8, v11 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[124:125], v19 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[12:17], v[56:61], a[24:27], v8, v11 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s42 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[12:17], v[62:67], a[28:31], v8, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[12:15], v110 offset:14336
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[68:73], v[62:67], a[60:63], v8, v11 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[16:17], v111 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[68:73], v[56:61], a[56:59], v8, v11 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[68:73], v[50:55], a[52:55], v8, v11 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s42 offen offset:2048 lds
	ds_read_b128 v[126:129], v110 offset:15360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[68:73], v[44:49], a[48:51], v8, v11 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[130:131], v111 offset:31232
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[68:73], v[38:43], a[44:47], v8, v10 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[68:73], v[32:37], a[40:43], v8, v10 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[132:135], v110 offset:16384
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[68:73], v[26:31], a[36:39], v8, v10 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[136:137], v111 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[68:73], v[20:25], a[32:35], v8, v10 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s42 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[74:79], v[20:25], a[64:67], v8, v10 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[68:71], v110 offset:17408
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[74:79], v[26:31], a[68:71], v8, v10 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[72:73], v111 offset:32256
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[74:79], v[32:37], a[72:75], v8, v10 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s46, s13, 0x13000
	s_mov_b32 m0, s8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[74:79], v[38:43], a[76:79], v8, v10 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s46 offen lds
	ds_read_b128 v[138:141], v110 offset:18432
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[74:79], v[44:49], a[80:83], v8, v11 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[142:143], v111 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[74:79], v[50:55], a[84:87], v8, v11 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[74:79], v[56:61], a[88:91], v8, v11 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[144:147], v110 offset:19456
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[74:79], v[62:67], a[92:95], v8, v11 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[148:149], v111 offset:33280
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[80:85], v[62:67], a[124:127], v8, v11 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s46 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[80:85], v[56:61], a[120:123], v8, v11 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[74:77], v110 offset:20480
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[80:85], v[50:55], a[116:119], v8, v11 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v111 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[80:85], v[44:49], a[112:115], v8, v11 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s18
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[80:85], v[38:43], a[108:111], v8, v10 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s42 offen lds
	ds_read_b128 v[150:153], v110 offset:21504
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[80:85], v[32:37], a[104:107], v8, v10 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[154:155], v111 offset:34304
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[80:85], v[26:31], a[100:103], v8, v10 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[80:85], v[20:25], a[96:99], v8, v10 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[80:83], v6 offset:56320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[86:91], v[20:25], a[128:131], v9, v10 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[84:85], v19 offset:6656
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[86:91], v[26:31], a[132:135], v9, v10 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s42 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[86:91], v[32:37], a[136:139], v9, v10 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[156:159], v6 offset:57344
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[86:91], v[38:43], a[140:143], v9, v10 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[160:161], v19 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[86:91], v[44:49], a[144:147], v9, v11 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[86:91], v[50:55], a[148:151], v9, v11 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s42 offen offset:2048 lds
	ds_read_b128 v[162:165], v6 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[86:91], v[56:61], a[152:155], v9, v11 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[166:167], v19 offset:7680
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[86:91], v[62:67], a[156:159], v9, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[92:97], v[62:67], a[188:191], v9, v11 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[86:89], v6 offset:59392
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[92:97], v[56:61], a[184:187], v9, v11 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[90:91], v19 offset:8192
	buffer_load_dwordx4 v1, s[24:27], s42 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[92:97], v[50:55], a[180:183], v9, v11 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[92:97], v[44:49], a[176:179], v9, v11 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[168:171], v6 offset:60416
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[92:97], v[38:43], a[172:175], v9, v10 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[172:173], v19 offset:8704
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[92:97], v[32:37], a[168:171], v9, v10 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s19
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[92:97], v[26:31], a[164:167], v9, v10 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s46 offen lds
	ds_read_b128 v[174:177], v6 offset:61440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[92:97], v[20:25], a[160:163], v9, v10 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[178:179], v19 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[98:103], v[20:25], a[192:195], v9, v10 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[98:103], v[26:31], a[196:199], v9, v10 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[92:95], v6 offset:62464
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[98:103], v[32:37], a[200:203], v9, v10 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[96:97], v19 offset:9728
	buffer_load_dwordx4 v1, s[24:27], s46 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[98:103], v[38:43], a[204:207], v9, v10 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[98:103], v[44:49], a[208:211], v9, v11 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[98:103], v[50:55], a[212:215], v9, v11 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[98:103], v[56:61], a[216:219], v9, v11 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_mul_i32 s42, s3, 0x30000
	s_add_i32 s42, s1, s42
	s_add_i32 s46, s42, 0xc000
	s_mov_b32 m0, s7
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[98:103], v[62:67], a[220:223], v9, v11 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[28:31], s46 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[104:109], v[62:67], a[252:255], v9, v11 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[104:109], v[56:61], a[248:251], v9, v11 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[104:109], v[50:55], a[244:247], v9, v11 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mul_i32 s46, s3, 0x180000
	s_add_i32 s46, s2, s46
	; A-scale planes are M*4 bytes; plane 3 starts at 0x60000 including offset:1024.
	s_add_i32 s48, s46, 0x5fc00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[104:109], v[44:49], a[240:243], v9, v11 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[32:35], s48 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[104:109], v[38:43], a[236:239], v9, v10 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[104:109], v[32:37], a[232:235], v9, v10 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[104:109], v[26:31], a[228:231], v9, v10 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[104:109], v[20:25], a[224:227], v9, v10 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[120:125], v[12:17], a[0:3], v116, v118 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[8:9], v5 offset:4096
	ds_read_b64 v[10:11], v3 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[120:125], v[126:131], a[4:7], v116, v118 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_i32 s48, s13, 0x18000
	s_mov_b32 m0, s38
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[120:125], v[132:137], a[8:11], v116, v118 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s48 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[120:125], v[68:73], a[12:15], v116, v118 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[120:125], v[138:143], a[16:19], v116, v119 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[20:23], v112 offset:38912
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[120:125], v[144:149], a[20:23], v116, v119 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[24:25], v19 offset:55296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[120:125], v[74:79], a[24:27], v116, v119 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s48 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[120:125], v[150:155], a[28:31], v116, v119 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[26:29], v110 offset:63488
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[80:85], v[150:155], a[60:63], v116, v119 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[30:31], v113 offset:14336
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[80:85], v[74:79], a[56:59], v116, v119 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[80:85], v[144:149], a[52:55], v116, v119 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s48 offen offset:2048 lds
	ds_read_b128 v[32:35], v110 offset:64512
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[80:85], v[138:143], a[48:51], v116, v119 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[36:37], v113 offset:14848
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[80:85], v[68:73], a[44:47], v116, v118 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[80:85], v[132:137], a[40:43], v116, v118 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[38:41], v114
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[80:85], v[126:131], a[36:39], v116, v118 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[42:43], v113 offset:15360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[80:85], v[12:17], a[32:35], v116, v118 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s48 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[156:161], v[12:17], a[64:67], v116, v118 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[44:47], v114 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[156:161], v[126:131], a[68:71], v116, v118 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[48:49], v113 offset:15872
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[156:161], v[132:137], a[72:75], v116, v118 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s49, s13, 0x19000
	s_mov_b32 m0, s39
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[156:161], v[68:73], a[76:79], v116, v118 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s49 offen lds
	ds_read_b128 v[50:53], v114 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[156:161], v[138:143], a[80:83], v116, v119 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[54:55], v113 offset:16384
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[156:161], v[144:149], a[84:87], v116, v119 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[156:161], v[74:79], a[88:91], v116, v119 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[56:59], v114 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[156:161], v[150:155], a[92:95], v116, v119 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[60:61], v113 offset:16896
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[162:167], v[150:155], a[124:127], v116, v119 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s49 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[162:167], v[74:79], a[120:123], v116, v119 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[62:65], v114 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[162:167], v[144:149], a[116:119], v116, v119 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[66:67], v113 offset:17408
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[162:167], v[138:143], a[112:115], v116, v119 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s41
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[162:167], v[68:73], a[108:111], v116, v118 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s48 offen lds
	ds_read_b128 v[80:83], v114 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[162:167], v[132:137], a[104:107], v116, v118 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[84:85], v113 offset:17920
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[162:167], v[126:131], a[100:103], v116, v118 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[162:167], v[12:17], a[96:99], v116, v118 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[98:101], v112 offset:39936
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[86:91], v[12:17], a[128:131], v117, v118 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[102:103], v19 offset:55808
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[86:91], v[126:131], a[132:135], v117, v118 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s48 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[86:91], v[132:137], a[136:139], v117, v118 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[104:107], v112 offset:40960
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[86:91], v[68:73], a[140:143], v117, v118 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[108:109], v19 offset:56320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[86:91], v[138:143], a[144:147], v117, v119 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[86:91], v[144:149], a[148:151], v117, v119 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s48 offen offset:2048 lds
	ds_read_b128 v[120:123], v112 offset:41984
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[86:91], v[74:79], a[152:155], v117, v119 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[124:125], v19 offset:56832
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[86:91], v[150:155], a[156:159], v117, v119 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[168:173], v[150:155], a[188:191], v117, v119 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[86:89], v112 offset:43008
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[168:173], v[74:79], a[184:187], v117, v119 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[90:91], v19 offset:57344
	buffer_load_dwordx4 v1, s[24:27], s48 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[168:173], v[144:149], a[180:183], v117, v119 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[168:173], v[138:143], a[176:179], v117, v119 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[156:159], v112 offset:44032
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[168:173], v[68:73], a[172:175], v117, v118 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[160:161], v19 offset:57856
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[168:173], v[132:137], a[168:171], v117, v118 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s37
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[168:173], v[126:131], a[164:167], v117, v118 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s49 offen lds
	ds_read_b128 v[162:165], v112 offset:45056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[168:173], v[12:17], a[160:163], v117, v118 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[166:167], v19 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[174:179], v[12:17], a[192:195], v117, v118 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[174:179], v[126:131], a[196:199], v117, v118 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[168:171], v112 offset:46080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[174:179], v[132:137], a[200:203], v117, v118 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[172:173], v19 offset:58880
	buffer_load_dwordx4 v1, s[24:27], s49 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[174:179], v[68:73], a[204:207], v117, v118 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[174:179], v[138:143], a[208:211], v117, v119 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[174:179], v[144:149], a[212:215], v117, v119 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[174:179], v[74:79], a[216:219], v117, v119 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s48, s42, 0x10000
	s_mov_b32 m0, s40
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[174:179], v[150:155], a[220:223], v117, v119 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[28:31], s48 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[92:97], v[150:155], a[252:255], v117, v119 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[92:97], v[74:79], a[248:251], v117, v119 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[92:97], v[144:149], a[244:247], v117, v119 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s48, s46, 0x7fc00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[92:97], v[138:143], a[240:243], v117, v119 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[32:35], s48 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[92:97], v[68:73], a[236:239], v117, v118 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[92:97], v[132:137], a[232:235], v117, v118 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[92:97], v[126:131], a[228:231], v117, v118 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[92:97], v[12:17], a[224:227], v117, v118 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:25], v[26:31], a[0:3], v8, v10 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[12:13], v5
	ds_read_b64 v[14:15], v3 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[20:25], v[32:37], a[4:7], v8, v10 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_i32 s48, s13, 0x1e000
	s_mov_b32 m0, s44
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[20:25], v[38:43], a[8:11], v8, v10 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s48 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[20:25], v[44:49], a[12:15], v8, v10 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[20:25], v[50:55], a[16:19], v8, v11 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[68:71], v6 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[20:25], v[56:61], a[20:23], v8, v11 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[72:73], v4 offset:22528
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[20:25], v[62:67], a[24:27], v8, v11 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s48 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[20:25], v[80:85], a[28:31], v8, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[20:23], v18 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[98:103], v[80:85], a[60:63], v8, v11 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[24:25], v7 offset:47104
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[98:103], v[62:67], a[56:59], v8, v11 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[98:103], v[56:61], a[52:55], v8, v11 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s48 offen offset:2048 lds
	ds_read_b128 v[74:77], v18 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[98:103], v[50:55], a[48:51], v8, v11 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v7 offset:47616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[98:103], v[44:49], a[44:47], v8, v10 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[98:103], v[38:43], a[40:43], v8, v10 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[92:95], v18 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[98:103], v[32:37], a[36:39], v8, v10 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[96:97], v7 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[98:103], v[26:31], a[32:35], v8, v10 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s48 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[104:109], v[26:31], a[64:67], v8, v10 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[98:101], v18 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[104:109], v[32:37], a[68:71], v8, v10 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[102:103], v7 offset:48640
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[104:109], v[38:43], a[72:75], v8, v10 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s49, s13, 0x1f000
	s_mov_b32 m0, s45
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[104:109], v[44:49], a[76:79], v8, v10 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s49 offen lds
	ds_read_b128 v[126:129], v18 offset:34816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[104:109], v[50:55], a[80:83], v8, v11 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[130:131], v7 offset:49152
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[104:109], v[56:61], a[84:87], v8, v11 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[104:109], v[62:67], a[88:91], v8, v11 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[132:135], v18 offset:35840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[104:109], v[80:85], a[92:95], v8, v11 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[136:137], v7 offset:49664
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[120:125], v[80:85], a[124:127], v8, v11 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s49 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[120:125], v[62:67], a[120:123], v8, v11 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[104:107], v18 offset:36864
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[120:125], v[56:61], a[116:119], v8, v11 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[108:109], v7 offset:50176
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[120:125], v[50:55], a[112:115], v8, v11 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s47
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[120:125], v[44:49], a[108:111], v8, v10 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s48 offen lds
	ds_read_b128 v[138:141], v18 offset:37888
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[120:125], v[38:43], a[104:107], v8, v10 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[142:143], v7 offset:50688
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[120:125], v[32:37], a[100:103], v8, v10 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[120:125], v[26:31], a[96:99], v8, v10 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[116:119], v6 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[86:91], v[26:31], a[128:131], v9, v10 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[120:121], v4 offset:23040
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[86:91], v[32:37], a[132:135], v9, v10 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s48 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[86:91], v[38:43], a[136:139], v9, v10 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[144:147], v6 offset:8192
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[86:91], v[44:49], a[140:143], v9, v10 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[148:149], v4 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[86:91], v[50:55], a[144:147], v9, v11 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[86:91], v[56:61], a[148:151], v9, v11 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s48 offen offset:2048 lds
	ds_read_b128 v[150:153], v6 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[86:91], v[62:67], a[152:155], v9, v11 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[154:155], v4 offset:24064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[86:91], v[80:85], a[156:159], v9, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[156:161], v[80:85], a[188:191], v9, v11 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[86:89], v6 offset:10240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[156:161], v[62:67], a[184:187], v9, v11 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[90:91], v4 offset:24576
	buffer_load_dwordx4 v1, s[24:27], s48 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[156:161], v[56:61], a[180:183], v9, v11 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[156:161], v[50:55], a[176:179], v9, v11 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[174:177], v6 offset:11264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[156:161], v[44:49], a[172:175], v9, v10 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[178:179], v4 offset:25088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[156:161], v[38:43], a[168:171], v9, v10 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s6
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[156:161], v[32:37], a[164:167], v9, v10 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s49 offen lds
	ds_read_b128 v[180:183], v6 offset:12288
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[156:161], v[26:31], a[160:163], v9, v10 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[184:185], v4 offset:25600
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[162:167], v[26:31], a[192:195], v9, v10 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[162:167], v[32:37], a[196:199], v9, v10 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[156:159], v6 offset:13312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[162:167], v[38:43], a[200:203], v9, v10 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[160:161], v4 offset:26112
	buffer_load_dwordx4 v1, s[24:27], s49 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[162:167], v[44:49], a[204:207], v9, v10 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[162:167], v[50:55], a[208:211], v9, v11 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[162:167], v[56:61], a[212:215], v9, v11 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[162:167], v[62:67], a[216:219], v9, v11 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s48, s42, 0x14000
	s_mov_b32 m0, s43
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[162:167], v[80:85], a[220:223], v9, v11 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[28:31], s48 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[168:173], v[80:85], a[252:255], v9, v11 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[168:173], v[62:67], a[248:251], v9, v11 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[168:173], v[56:61], a[244:247], v9, v11 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s48, s46, 0x9fc00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[168:173], v[50:55], a[240:243], v9, v11 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[32:35], s48 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[168:173], v[44:49], a[236:239], v9, v10 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[168:173], v[38:43], a[232:235], v9, v10 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[168:173], v[32:37], a[228:231], v9, v10 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[168:173], v[26:31], a[224:227], v9, v10 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[68:73], v[20:25], a[0:3], v12, v14 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[8:9], v5 offset:2048
	ds_read_b64 v[10:11], v3 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[68:73], v[74:79], a[4:7], v12, v14 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_i32 s48, s13, 0x24000
	s_mov_b32 m0, s16
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[68:73], v[92:97], a[8:11], v12, v14 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s48 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[68:73], v[98:103], a[12:15], v12, v14 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[68:73], v[126:131], a[16:19], v12, v15 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[26:29], v6 offset:55296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[68:73], v[132:137], a[20:23], v12, v15 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[30:31], v19 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[68:73], v[104:109], a[24:27], v12, v15 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s48 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[68:73], v[138:143], a[28:31], v12, v15 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[32:35], v110 offset:14336
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[116:121], v[138:143], a[60:63], v12, v15 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[36:37], v111 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[116:121], v[104:109], a[56:59], v12, v15 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[116:121], v[132:137], a[52:55], v12, v15 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s48 offen offset:2048 lds
	ds_read_b128 v[38:41], v110 offset:15360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[116:121], v[126:131], a[48:51], v12, v15 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[42:43], v111 offset:31232
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[116:121], v[98:103], a[44:47], v12, v14 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[116:121], v[92:97], a[40:43], v12, v14 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[44:47], v110 offset:16384
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[116:121], v[74:79], a[36:39], v12, v14 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[48:49], v111 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[116:121], v[20:25], a[32:35], v12, v14 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s48 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[144:149], v[20:25], a[64:67], v12, v14 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[50:53], v110 offset:17408
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[144:149], v[74:79], a[68:71], v12, v14 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[54:55], v111 offset:32256
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[144:149], v[92:97], a[72:75], v12, v14 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s49, s13, 0x25000
	s_mov_b32 m0, s8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[144:149], v[98:103], a[76:79], v12, v14 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s49 offen lds
	ds_read_b128 v[56:59], v110 offset:18432
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[144:149], v[126:131], a[80:83], v12, v15 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[60:61], v111 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[144:149], v[132:137], a[84:87], v12, v15 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[144:149], v[104:109], a[88:91], v12, v15 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[62:65], v110 offset:19456
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[144:149], v[138:143], a[92:95], v12, v15 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[66:67], v111 offset:33280
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[150:155], v[138:143], a[124:127], v12, v15 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s49 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[150:155], v[104:109], a[120:123], v12, v15 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[68:71], v110 offset:20480
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[150:155], v[132:137], a[116:119], v12, v15 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[72:73], v111 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[150:155], v[126:131], a[112:115], v12, v15 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s18
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[150:155], v[98:103], a[108:111], v12, v14 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s48 offen lds
	ds_read_b128 v[80:83], v110 offset:21504
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[150:155], v[92:97], a[104:107], v12, v14 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[84:85], v111 offset:34304
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[150:155], v[74:79], a[100:103], v12, v14 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[150:155], v[20:25], a[96:99], v12, v14 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[116:119], v6 offset:56320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[86:91], v[20:25], a[128:131], v13, v14 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[120:121], v19 offset:6656
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[86:91], v[74:79], a[132:135], v13, v14 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s48 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[86:91], v[92:97], a[136:139], v13, v14 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[144:147], v6 offset:57344
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[86:91], v[98:103], a[140:143], v13, v14 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[148:149], v19 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[86:91], v[126:131], a[144:147], v13, v15 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[86:91], v[132:137], a[148:151], v13, v15 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s48 offen offset:2048 lds
	ds_read_b128 v[150:153], v6 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[86:91], v[104:109], a[152:155], v13, v15 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[154:155], v19 offset:7680
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[86:91], v[138:143], a[156:159], v13, v15 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[174:179], v[138:143], a[188:191], v13, v15 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[86:89], v6 offset:59392
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[174:179], v[104:109], a[184:187], v13, v15 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[90:91], v19 offset:8192
	buffer_load_dwordx4 v1, s[24:27], s48 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[174:179], v[132:137], a[180:183], v13, v15 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[174:179], v[126:131], a[176:179], v13, v15 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[162:165], v6 offset:60416
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[174:179], v[98:103], a[172:175], v13, v14 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[166:167], v19 offset:8704
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[174:179], v[92:97], a[168:171], v13, v14 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s19
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[174:179], v[74:79], a[164:167], v13, v14 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s49 offen lds
	ds_read_b128 v[168:171], v6 offset:61440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[174:179], v[20:25], a[160:163], v13, v14 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[172:173], v19 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[180:185], v[20:25], a[192:195], v13, v14 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[180:185], v[74:79], a[196:199], v13, v14 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[174:177], v6 offset:62464
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[180:185], v[92:97], a[200:203], v13, v14 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[178:179], v19 offset:9728
	buffer_load_dwordx4 v1, s[24:27], s49 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[180:185], v[98:103], a[204:207], v13, v14 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[180:185], v[126:131], a[208:211], v13, v15 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[180:185], v[132:137], a[212:215], v13, v15 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[180:185], v[104:109], a[216:219], v13, v15 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s48, s42, 0x18000
	s_mov_b32 m0, s7
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[180:185], v[138:143], a[220:223], v13, v15 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[28:31], s48 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[156:161], v[138:143], a[252:255], v13, v15 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[156:161], v[104:109], a[248:251], v13, v15 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[156:161], v[132:137], a[244:247], v13, v15 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s48, s46, 0xbfc00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[156:161], v[126:131], a[240:243], v13, v15 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[32:35], s48 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[156:161], v[98:103], a[236:239], v13, v14 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[156:161], v[92:97], a[232:235], v13, v14 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[156:161], v[74:79], a[228:231], v13, v14 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[156:161], v[20:25], a[224:227], v13, v14 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:31], v[32:37], a[0:3], v8, v10 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[12:13], v5 offset:4096
	ds_read_b64 v[14:15], v3 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[26:31], v[38:43], a[4:7], v8, v10 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_i32 s48, s13, 0x2a000
	s_mov_b32 m0, s38
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[26:31], v[44:49], a[8:11], v8, v10 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s48 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[26:31], v[50:55], a[12:15], v8, v10 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[26:31], v[56:61], a[16:19], v8, v11 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[20:23], v112 offset:38912
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[26:31], v[62:67], a[20:23], v8, v11 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[24:25], v19 offset:55296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[26:31], v[68:73], a[24:27], v8, v11 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s48 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[26:31], v[80:85], a[28:31], v8, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[26:29], v110 offset:63488
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[116:121], v[80:85], a[60:63], v8, v11 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[30:31], v113 offset:14336
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[116:121], v[68:73], a[56:59], v8, v11 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[116:121], v[62:67], a[52:55], v8, v11 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s48 offen offset:2048 lds
	ds_read_b128 v[74:77], v110 offset:64512
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[116:121], v[56:61], a[48:51], v8, v11 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v113 offset:14848
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[116:121], v[50:55], a[44:47], v8, v10 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[116:121], v[44:49], a[40:43], v8, v10 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[92:95], v114
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[116:121], v[38:43], a[36:39], v8, v10 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[96:97], v113 offset:15360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[116:121], v[32:37], a[32:35], v8, v10 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s48 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[144:149], v[32:37], a[64:67], v8, v10 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[98:101], v114 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[144:149], v[38:43], a[68:71], v8, v10 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[102:103], v113 offset:15872
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[144:149], v[44:49], a[72:75], v8, v10 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s49, s13, 0x2b000
	s_mov_b32 m0, s39
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[144:149], v[50:55], a[76:79], v8, v10 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s49 offen lds
	ds_read_b128 v[104:107], v114 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[144:149], v[56:61], a[80:83], v8, v11 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[108:109], v113 offset:16384
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[144:149], v[62:67], a[84:87], v8, v11 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[144:149], v[68:73], a[88:91], v8, v11 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[116:119], v114 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[144:149], v[80:85], a[92:95], v8, v11 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[120:121], v113 offset:16896
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[150:155], v[80:85], a[124:127], v8, v11 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s49 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[150:155], v[68:73], a[120:123], v8, v11 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[122:125], v114 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[150:155], v[62:67], a[116:119], v8, v11 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[126:127], v113 offset:17408
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[150:155], v[56:61], a[112:115], v8, v11 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s41
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[150:155], v[50:55], a[108:111], v8, v10 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s48 offen lds
	ds_read_b128 v[128:131], v114 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[150:155], v[44:49], a[104:107], v8, v10 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[132:133], v113 offset:17920
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[150:155], v[38:43], a[100:103], v8, v10 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[150:155], v[32:37], a[96:99], v8, v10 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[134:137], v112 offset:39936
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[86:91], v[32:37], a[128:131], v9, v10 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[138:139], v19 offset:55808
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[86:91], v[38:43], a[132:135], v9, v10 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s48 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[86:91], v[44:49], a[136:139], v9, v10 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[140:143], v112 offset:40960
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[86:91], v[50:55], a[140:143], v9, v10 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[144:145], v19 offset:56320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[86:91], v[56:61], a[144:147], v9, v11 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[86:91], v[62:67], a[148:151], v9, v11 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s48 offen offset:2048 lds
	ds_read_b128 v[146:149], v112 offset:41984
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[86:91], v[68:73], a[152:155], v9, v11 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[150:151], v19 offset:56832
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[86:91], v[80:85], a[156:159], v9, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[162:167], v[80:85], a[188:191], v9, v11 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[86:89], v112 offset:43008
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[162:167], v[68:73], a[184:187], v9, v11 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[90:91], v19 offset:57344
	buffer_load_dwordx4 v1, s[24:27], s48 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[162:167], v[62:67], a[180:183], v9, v11 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[162:167], v[56:61], a[176:179], v9, v11 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[152:155], v112 offset:44032
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[162:167], v[50:55], a[172:175], v9, v10 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[156:157], v19 offset:57856
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[162:167], v[44:49], a[168:171], v9, v10 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s37
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[162:167], v[38:43], a[164:167], v9, v10 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s49 offen lds
	ds_read_b128 v[158:161], v112 offset:45056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[162:167], v[32:37], a[160:163], v9, v10 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[162:163], v19 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[168:173], v[32:37], a[192:195], v9, v10 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[168:173], v[38:43], a[196:199], v9, v10 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[180:183], v112 offset:46080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[168:173], v[44:49], a[200:203], v9, v10 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[184:185], v19 offset:58880
	buffer_load_dwordx4 v1, s[24:27], s49 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[168:173], v[50:55], a[204:207], v9, v10 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[168:173], v[56:61], a[208:211], v9, v11 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[168:173], v[62:67], a[212:215], v9, v11 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[168:173], v[68:73], a[216:219], v9, v11 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s48, s42, 0x1c000
	s_mov_b32 m0, s40
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[168:173], v[80:85], a[220:223], v9, v11 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[28:31], s48 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[174:179], v[80:85], a[252:255], v9, v11 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[174:179], v[68:73], a[248:251], v9, v11 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[174:179], v[62:67], a[244:247], v9, v11 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s48, s46, 0xdfc00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[174:179], v[56:61], a[240:243], v9, v11 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[32:35], s48 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[174:179], v[50:55], a[236:239], v9, v10 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[174:179], v[44:49], a[232:235], v9, v10 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[174:179], v[38:43], a[228:231], v9, v10 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[174:179], v[32:37], a[224:227], v9, v10 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:25], v[26:31], a[0:3], v12, v14 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[8:9], v5
	ds_read_b64 v[10:11], v3 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[20:25], v[74:79], a[4:7], v12, v14 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_i32 s48, s13, 0x30000
	s_mov_b32 m0, s44
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[20:25], v[92:97], a[8:11], v12, v14 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s48 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[20:25], v[98:103], a[12:15], v12, v14 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[20:25], v[104:109], a[16:19], v12, v15 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[32:35], v6 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[20:25], v[116:121], a[20:23], v12, v15 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[36:37], v4 offset:22528
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[20:25], v[122:127], a[24:27], v12, v15 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s48 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[20:25], v[128:133], a[28:31], v12, v15 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[20:23], v18 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[134:139], v[128:133], a[60:63], v12, v15 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[24:25], v7 offset:47104
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[134:139], v[122:127], a[56:59], v12, v15 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[134:139], v[116:121], a[52:55], v12, v15 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s48 offen offset:2048 lds
	ds_read_b128 v[38:41], v18 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[134:139], v[104:109], a[48:51], v12, v15 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[42:43], v7 offset:47616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[134:139], v[98:103], a[44:47], v12, v14 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[134:139], v[92:97], a[40:43], v12, v14 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[44:47], v18 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[134:139], v[74:79], a[36:39], v12, v14 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[48:49], v7 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[134:139], v[26:31], a[32:35], v12, v14 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s48 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[140:145], v[26:31], a[64:67], v12, v14 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[50:53], v18 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[140:145], v[74:79], a[68:71], v12, v14 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[54:55], v7 offset:48640
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[140:145], v[92:97], a[72:75], v12, v14 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s49, s13, 0x31000
	s_mov_b32 m0, s45
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[140:145], v[98:103], a[76:79], v12, v14 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s49 offen lds
	ds_read_b128 v[56:59], v18 offset:34816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[140:145], v[104:109], a[80:83], v12, v15 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[60:61], v7 offset:49152
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[140:145], v[116:121], a[84:87], v12, v15 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[140:145], v[122:127], a[88:91], v12, v15 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[62:65], v18 offset:35840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[140:145], v[128:133], a[92:95], v12, v15 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[66:67], v7 offset:49664
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[146:151], v[128:133], a[124:127], v12, v15 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s49 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[146:151], v[122:127], a[120:123], v12, v15 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[68:71], v18 offset:36864
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[146:151], v[116:121], a[116:119], v12, v15 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[72:73], v7 offset:50176
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[146:151], v[104:109], a[112:115], v12, v15 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s47
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[146:151], v[98:103], a[108:111], v12, v14 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s48 offen lds
	ds_read_b128 v[80:83], v18 offset:37888
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[146:151], v[92:97], a[104:107], v12, v14 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[84:85], v7 offset:50688
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[146:151], v[74:79], a[100:103], v12, v14 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[146:151], v[26:31], a[96:99], v12, v14 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[134:137], v6 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[86:91], v[26:31], a[128:131], v13, v14 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[138:139], v4 offset:23040
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[86:91], v[74:79], a[132:135], v13, v14 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s48 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[86:91], v[92:97], a[136:139], v13, v14 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[140:143], v6 offset:8192
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[86:91], v[98:103], a[140:143], v13, v14 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[144:145], v4 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[86:91], v[104:109], a[144:147], v13, v15 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[86:91], v[116:121], a[148:151], v13, v15 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s48 offen offset:2048 lds
	ds_read_b128 v[146:149], v6 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[86:91], v[122:127], a[152:155], v13, v15 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[150:151], v4 offset:24064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[86:91], v[128:133], a[156:159], v13, v15 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[152:157], v[128:133], a[188:191], v13, v15 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[86:89], v6 offset:10240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[152:157], v[122:127], a[184:187], v13, v15 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[90:91], v4 offset:24576
	buffer_load_dwordx4 v1, s[24:27], s48 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[152:157], v[116:121], a[180:183], v13, v15 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[152:157], v[104:109], a[176:179], v13, v15 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[164:167], v6 offset:11264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[152:157], v[98:103], a[172:175], v13, v14 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[168:169], v4 offset:25088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[152:157], v[92:97], a[168:171], v13, v14 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s6
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[152:157], v[74:79], a[164:167], v13, v14 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s49 offen lds
	ds_read_b128 v[170:173], v6 offset:12288
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[152:157], v[26:31], a[160:163], v13, v14 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[174:175], v4 offset:25600
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[158:163], v[26:31], a[192:195], v13, v14 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[158:163], v[74:79], a[196:199], v13, v14 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[152:155], v6 offset:13312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[158:163], v[92:97], a[200:203], v13, v14 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[156:157], v4 offset:26112
	buffer_load_dwordx4 v1, s[24:27], s49 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[158:163], v[98:103], a[204:207], v13, v14 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[158:163], v[104:109], a[208:211], v13, v15 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[158:163], v[116:121], a[212:215], v13, v15 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[158:163], v[122:127], a[216:219], v13, v15 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s48, s42, 0x20000
	s_mov_b32 m0, s43
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[158:163], v[128:133], a[220:223], v13, v15 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[28:31], s48 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[180:185], v[128:133], a[252:255], v13, v15 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[180:185], v[122:127], a[248:251], v13, v15 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[180:185], v[116:121], a[244:247], v13, v15 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s48, s46, 0xffc00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[180:185], v[104:109], a[240:243], v13, v15 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[32:35], s48 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[180:185], v[98:103], a[236:239], v13, v14 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[180:185], v[92:97], a[232:235], v13, v14 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[180:185], v[74:79], a[228:231], v13, v14 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[180:185], v[26:31], a[224:227], v13, v14 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[32:37], v[20:25], a[0:3], v8, v10 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[12:13], v5 offset:2048
	ds_read_b64 v[14:15], v3 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[32:37], v[38:43], a[4:7], v8, v10 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_i32 s48, s13, 0x36000
	s_mov_b32 m0, s16
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[32:37], v[44:49], a[8:11], v8, v10 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s48 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[32:37], v[50:55], a[12:15], v8, v10 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[32:37], v[56:61], a[16:19], v8, v11 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[26:29], v6 offset:55296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[32:37], v[62:67], a[20:23], v8, v11 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[30:31], v19 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[32:37], v[68:73], a[24:27], v8, v11 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s48 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[32:37], v[80:85], a[28:31], v8, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[32:35], v110 offset:14336
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[134:139], v[80:85], a[60:63], v8, v11 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[36:37], v111 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[134:139], v[68:73], a[56:59], v8, v11 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[134:139], v[62:67], a[52:55], v8, v11 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s48 offen offset:2048 lds
	ds_read_b128 v[74:77], v110 offset:15360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[134:139], v[56:61], a[48:51], v8, v11 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v111 offset:31232
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[134:139], v[50:55], a[44:47], v8, v10 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[134:139], v[44:49], a[40:43], v8, v10 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[92:95], v110 offset:16384
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[134:139], v[38:43], a[36:39], v8, v10 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[96:97], v111 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[134:139], v[20:25], a[32:35], v8, v10 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s48 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[140:145], v[20:25], a[64:67], v8, v10 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[98:101], v110 offset:17408
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[140:145], v[38:43], a[68:71], v8, v10 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[102:103], v111 offset:32256
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[140:145], v[44:49], a[72:75], v8, v10 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s49, s13, 0x37000
	s_mov_b32 m0, s8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[140:145], v[50:55], a[76:79], v8, v10 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s49 offen lds
	ds_read_b128 v[104:107], v110 offset:18432
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[140:145], v[56:61], a[80:83], v8, v11 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[108:109], v111 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[140:145], v[62:67], a[84:87], v8, v11 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[140:145], v[68:73], a[88:91], v8, v11 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[116:119], v110 offset:19456
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[140:145], v[80:85], a[92:95], v8, v11 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[120:121], v111 offset:33280
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[146:151], v[80:85], a[124:127], v8, v11 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s49 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[146:151], v[68:73], a[120:123], v8, v11 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[122:125], v110 offset:20480
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[146:151], v[62:67], a[116:119], v8, v11 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[126:127], v111 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[146:151], v[56:61], a[112:115], v8, v11 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s18
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[146:151], v[50:55], a[108:111], v8, v10 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s48 offen lds
	ds_read_b128 v[128:131], v110 offset:21504
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[146:151], v[44:49], a[104:107], v8, v10 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[132:133], v111 offset:34304
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[146:151], v[38:43], a[100:103], v8, v10 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[146:151], v[20:25], a[96:99], v8, v10 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[134:137], v6 offset:56320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[86:91], v[20:25], a[128:131], v9, v10 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[138:139], v19 offset:6656
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[86:91], v[38:43], a[132:135], v9, v10 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s48 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[86:91], v[44:49], a[136:139], v9, v10 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[140:143], v6 offset:57344
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[86:91], v[50:55], a[140:143], v9, v10 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[144:145], v19 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[86:91], v[56:61], a[144:147], v9, v11 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[86:91], v[62:67], a[148:151], v9, v11 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s48 offen offset:2048 lds
	ds_read_b128 v[146:149], v6 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[86:91], v[68:73], a[152:155], v9, v11 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[150:151], v19 offset:7680
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[86:91], v[80:85], a[156:159], v9, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[164:169], v[80:85], a[188:191], v9, v11 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[86:89], v6 offset:59392
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[164:169], v[68:73], a[184:187], v9, v11 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[90:91], v19 offset:8192
	buffer_load_dwordx4 v1, s[24:27], s48 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[164:169], v[62:67], a[180:183], v9, v11 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[164:169], v[56:61], a[176:179], v9, v11 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[158:161], v6 offset:60416
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[164:169], v[50:55], a[172:175], v9, v10 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[162:163], v19 offset:8704
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[164:169], v[44:49], a[168:171], v9, v10 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s19
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[164:169], v[38:43], a[164:167], v9, v10 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s49 offen lds
	ds_read_b128 v[176:179], v6 offset:61440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[164:169], v[20:25], a[160:163], v9, v10 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[180:181], v19 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[170:175], v[20:25], a[192:195], v9, v10 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[170:175], v[38:43], a[196:199], v9, v10 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[164:167], v6 offset:62464
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[170:175], v[44:49], a[200:203], v9, v10 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[168:169], v19 offset:9728
	buffer_load_dwordx4 v1, s[24:27], s49 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[170:175], v[50:55], a[204:207], v9, v10 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[170:175], v[56:61], a[208:211], v9, v11 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[170:175], v[62:67], a[212:215], v9, v11 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[170:175], v[68:73], a[216:219], v9, v11 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s48, s42, 0x24000
	s_mov_b32 m0, s7
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[170:175], v[80:85], a[220:223], v9, v11 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[28:31], s48 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[152:157], v[80:85], a[252:255], v9, v11 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[152:157], v[68:73], a[248:251], v9, v11 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[152:157], v[62:67], a[244:247], v9, v11 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s48, s46, 0x11fc00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[152:157], v[56:61], a[240:243], v9, v11 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[32:35], s48 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[152:157], v[50:55], a[236:239], v9, v10 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[152:157], v[44:49], a[232:235], v9, v10 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[152:157], v[38:43], a[228:231], v9, v10 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[152:157], v[20:25], a[224:227], v9, v10 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:31], v[32:37], a[0:3], v12, v14 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[8:9], v5 offset:4096
	ds_read_b64 v[10:11], v3 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[26:31], v[74:79], a[4:7], v12, v14 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_i32 s48, s13, 0x3c000
	s_mov_b32 m0, s38
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[26:31], v[92:97], a[8:11], v12, v14 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s48 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[26:31], v[98:103], a[12:15], v12, v14 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[26:31], v[104:109], a[16:19], v12, v15 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[20:23], v112 offset:38912
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[26:31], v[116:121], a[20:23], v12, v15 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[24:25], v19 offset:55296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[26:31], v[122:127], a[24:27], v12, v15 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s48 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[26:31], v[128:133], a[28:31], v12, v15 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[26:29], v110 offset:63488
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[134:139], v[128:133], a[60:63], v12, v15 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[30:31], v113 offset:14336
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[134:139], v[122:127], a[56:59], v12, v15 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[134:139], v[116:121], a[52:55], v12, v15 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s48 offen offset:2048 lds
	ds_read_b128 v[38:41], v110 offset:64512
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[134:139], v[104:109], a[48:51], v12, v15 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[42:43], v113 offset:14848
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[134:139], v[98:103], a[44:47], v12, v14 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[134:139], v[92:97], a[40:43], v12, v14 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[44:47], v114
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[134:139], v[74:79], a[36:39], v12, v14 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[48:49], v113 offset:15360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[134:139], v[32:37], a[32:35], v12, v14 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s48 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[140:145], v[32:37], a[64:67], v12, v14 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[50:53], v114 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[140:145], v[74:79], a[68:71], v12, v14 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[54:55], v113 offset:15872
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[140:145], v[92:97], a[72:75], v12, v14 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s49, s13, 0x3d000
	s_mov_b32 m0, s39
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[140:145], v[98:103], a[76:79], v12, v14 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s49 offen lds
	ds_read_b128 v[56:59], v114 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[140:145], v[104:109], a[80:83], v12, v15 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[60:61], v113 offset:16384
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[140:145], v[116:121], a[84:87], v12, v15 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[140:145], v[122:127], a[88:91], v12, v15 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[62:65], v114 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[140:145], v[128:133], a[92:95], v12, v15 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[66:67], v113 offset:16896
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[146:151], v[128:133], a[124:127], v12, v15 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s49 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[146:151], v[122:127], a[120:123], v12, v15 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[68:71], v114 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[146:151], v[116:121], a[116:119], v12, v15 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[72:73], v113 offset:17408
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[146:151], v[104:109], a[112:115], v12, v15 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s41
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[146:151], v[98:103], a[108:111], v12, v14 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s48 offen lds
	ds_read_b128 v[80:83], v114 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[146:151], v[92:97], a[104:107], v12, v14 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[84:85], v113 offset:17920
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[146:151], v[74:79], a[100:103], v12, v14 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[146:151], v[32:37], a[96:99], v12, v14 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[134:137], v112 offset:39936
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[86:91], v[32:37], a[128:131], v13, v14 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[138:139], v19 offset:55808
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[86:91], v[74:79], a[132:135], v13, v14 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s48 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[86:91], v[92:97], a[136:139], v13, v14 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[140:143], v112 offset:40960
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[86:91], v[98:103], a[140:143], v13, v14 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[144:145], v19 offset:56320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[86:91], v[104:109], a[144:147], v13, v15 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[86:91], v[116:121], a[148:151], v13, v15 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s48 offen offset:2048 lds
	ds_read_b128 v[146:149], v112 offset:41984
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[86:91], v[122:127], a[152:155], v13, v15 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[150:151], v19 offset:56832
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[86:91], v[128:133], a[156:159], v13, v15 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[158:163], v[128:133], a[188:191], v13, v15 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[86:89], v112 offset:43008
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[158:163], v[122:127], a[184:187], v13, v15 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[90:91], v19 offset:57344
	buffer_load_dwordx4 v1, s[24:27], s48 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[158:163], v[116:121], a[180:183], v13, v15 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[158:163], v[104:109], a[176:179], v13, v15 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[152:155], v112 offset:44032
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[158:163], v[98:103], a[172:175], v13, v14 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[156:157], v19 offset:57856
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[158:163], v[92:97], a[168:171], v13, v14 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s37
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[158:163], v[74:79], a[164:167], v13, v14 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s49 offen lds
	ds_read_b128 v[170:173], v112 offset:45056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[158:163], v[32:37], a[160:163], v13, v14 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[174:175], v19 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[176:181], v[32:37], a[192:195], v13, v14 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[176:181], v[74:79], a[196:199], v13, v14 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[158:161], v112 offset:46080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[176:181], v[92:97], a[200:203], v13, v14 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[162:163], v19 offset:58880
	buffer_load_dwordx4 v1, s[24:27], s49 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[176:181], v[98:103], a[204:207], v13, v14 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[176:181], v[104:109], a[208:211], v13, v15 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[176:181], v[116:121], a[212:215], v13, v15 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[176:181], v[122:127], a[216:219], v13, v15 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s48, s42, 0x28000
	s_mov_b32 m0, s40
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[176:181], v[128:133], a[220:223], v13, v15 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[28:31], s48 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[164:169], v[128:133], a[252:255], v13, v15 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[164:169], v[122:127], a[248:251], v13, v15 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[164:169], v[116:121], a[244:247], v13, v15 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s48, s46, 0x13fc00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[164:169], v[104:109], a[240:243], v13, v15 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[32:35], s48 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[164:169], v[98:103], a[236:239], v13, v14 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[164:169], v[92:97], a[232:235], v13, v14 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[164:169], v[74:79], a[228:231], v13, v14 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[164:169], v[32:37], a[224:227], v13, v14 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:25], v[26:31], a[0:3], v8, v10 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[12:13], v5
	ds_read_b64 v[14:15], v3 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[20:25], v[38:43], a[4:7], v8, v10 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_i32 s48, s13, 0x42000
	s_mov_b32 m0, s44
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[20:25], v[44:49], a[8:11], v8, v10 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s48 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[20:25], v[50:55], a[12:15], v8, v10 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[20:25], v[56:61], a[16:19], v8, v11 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[32:35], v6 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[20:25], v[62:67], a[20:23], v8, v11 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[36:37], v4 offset:22528
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[20:25], v[68:73], a[24:27], v8, v11 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s48 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[20:25], v[80:85], a[28:31], v8, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[20:23], v18 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[134:139], v[80:85], a[60:63], v8, v11 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[24:25], v7 offset:47104
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[134:139], v[68:73], a[56:59], v8, v11 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[134:139], v[62:67], a[52:55], v8, v11 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s48 offen offset:2048 lds
	ds_read_b128 v[74:77], v18 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[134:139], v[56:61], a[48:51], v8, v11 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v7 offset:47616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[134:139], v[50:55], a[44:47], v8, v10 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[134:139], v[44:49], a[40:43], v8, v10 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[92:95], v18 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[134:139], v[38:43], a[36:39], v8, v10 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[96:97], v7 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[134:139], v[26:31], a[32:35], v8, v10 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s48 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[140:145], v[26:31], a[64:67], v8, v10 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[98:101], v18 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[140:145], v[38:43], a[68:71], v8, v10 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[102:103], v7 offset:48640
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[140:145], v[44:49], a[72:75], v8, v10 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s49, s13, 0x43000
	s_mov_b32 m0, s45
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[140:145], v[50:55], a[76:79], v8, v10 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s49 offen lds
	ds_read_b128 v[104:107], v18 offset:34816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[140:145], v[56:61], a[80:83], v8, v11 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[108:109], v7 offset:49152
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[140:145], v[62:67], a[84:87], v8, v11 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[140:145], v[68:73], a[88:91], v8, v11 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[116:119], v18 offset:35840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[140:145], v[80:85], a[92:95], v8, v11 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[120:121], v7 offset:49664
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[146:151], v[80:85], a[124:127], v8, v11 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s49 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[146:151], v[68:73], a[120:123], v8, v11 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[122:125], v18 offset:36864
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[146:151], v[62:67], a[116:119], v8, v11 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[126:127], v7 offset:50176
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[146:151], v[56:61], a[112:115], v8, v11 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s47
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[146:151], v[50:55], a[108:111], v8, v10 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s48 offen lds
	ds_read_b128 v[128:131], v18 offset:37888
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[146:151], v[44:49], a[104:107], v8, v10 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[132:133], v7 offset:50688
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[146:151], v[38:43], a[100:103], v8, v10 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[146:151], v[26:31], a[96:99], v8, v10 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[134:137], v6 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[86:91], v[26:31], a[128:131], v9, v10 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[138:139], v4 offset:23040
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[86:91], v[38:43], a[132:135], v9, v10 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s48 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[86:91], v[44:49], a[136:139], v9, v10 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[140:143], v6 offset:8192
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[86:91], v[50:55], a[140:143], v9, v10 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[144:145], v4 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[86:91], v[56:61], a[144:147], v9, v11 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[86:91], v[62:67], a[148:151], v9, v11 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s48 offen offset:2048 lds
	ds_read_b128 v[146:149], v6 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[86:91], v[68:73], a[152:155], v9, v11 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[150:151], v4 offset:24064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[86:91], v[80:85], a[156:159], v9, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[152:157], v[80:85], a[188:191], v9, v11 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[86:89], v6 offset:10240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[152:157], v[68:73], a[184:187], v9, v11 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[90:91], v4 offset:24576
	buffer_load_dwordx4 v1, s[24:27], s48 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[152:157], v[62:67], a[180:183], v9, v11 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[152:157], v[56:61], a[176:179], v9, v11 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[164:167], v6 offset:11264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[152:157], v[50:55], a[172:175], v9, v10 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[168:169], v4 offset:25088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[152:157], v[44:49], a[168:171], v9, v10 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s6
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[152:157], v[38:43], a[164:167], v9, v10 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s49 offen lds
	ds_read_b128 v[176:179], v6 offset:12288
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[152:157], v[26:31], a[160:163], v9, v10 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[180:181], v4 offset:25600
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[170:175], v[26:31], a[192:195], v9, v10 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[170:175], v[38:43], a[196:199], v9, v10 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[152:155], v6 offset:13312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[170:175], v[44:49], a[200:203], v9, v10 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[156:157], v4 offset:26112
	buffer_load_dwordx4 v1, s[24:27], s49 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[170:175], v[50:55], a[204:207], v9, v10 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[170:175], v[56:61], a[208:211], v9, v11 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[170:175], v[62:67], a[212:215], v9, v11 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[170:175], v[68:73], a[216:219], v9, v11 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s48, s42, 0x2c000
	s_mov_b32 m0, s43
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[170:175], v[80:85], a[220:223], v9, v11 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[28:31], s48 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[158:163], v[80:85], a[252:255], v9, v11 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[158:163], v[68:73], a[248:251], v9, v11 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[158:163], v[62:67], a[244:247], v9, v11 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s48, s46, 0x15fc00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[158:163], v[56:61], a[240:243], v9, v11 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[32:35], s48 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[158:163], v[50:55], a[236:239], v9, v10 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[158:163], v[44:49], a[232:235], v9, v10 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[158:163], v[38:43], a[228:231], v9, v10 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[158:163], v[26:31], a[224:227], v9, v10 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[32:37], v[20:25], a[0:3], v12, v14 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[8:9], v5 offset:2048
	ds_read_b64 v[10:11], v3 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[32:37], v[74:79], a[4:7], v12, v14 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_i32 s48, s13, 0x48000
	s_mov_b32 m0, s16
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[32:37], v[92:97], a[8:11], v12, v14 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s48 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[32:37], v[98:103], a[12:15], v12, v14 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[32:37], v[104:109], a[16:19], v12, v15 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[26:29], v6 offset:55296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[32:37], v[116:121], a[20:23], v12, v15 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[30:31], v19 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[32:37], v[122:127], a[24:27], v12, v15 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s48 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[32:37], v[128:133], a[28:31], v12, v15 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[32:35], v110 offset:14336
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[134:139], v[128:133], a[60:63], v12, v15 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[36:37], v111 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[134:139], v[122:127], a[56:59], v12, v15 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[134:139], v[116:121], a[52:55], v12, v15 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s48 offen offset:2048 lds
	ds_read_b128 v[38:41], v110 offset:15360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[134:139], v[104:109], a[48:51], v12, v15 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[42:43], v111 offset:31232
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[134:139], v[98:103], a[44:47], v12, v14 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[134:139], v[92:97], a[40:43], v12, v14 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[44:47], v110 offset:16384
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[134:139], v[74:79], a[36:39], v12, v14 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[48:49], v111 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[134:139], v[20:25], a[32:35], v12, v14 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s48 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[140:145], v[20:25], a[64:67], v12, v14 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[50:53], v110 offset:17408
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[140:145], v[74:79], a[68:71], v12, v14 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[54:55], v111 offset:32256
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[140:145], v[92:97], a[72:75], v12, v14 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s49, s13, 0x49000
	s_mov_b32 m0, s8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[140:145], v[98:103], a[76:79], v12, v14 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s49 offen lds
	ds_read_b128 v[56:59], v110 offset:18432
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[140:145], v[104:109], a[80:83], v12, v15 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[60:61], v111 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[140:145], v[116:121], a[84:87], v12, v15 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[140:145], v[122:127], a[88:91], v12, v15 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[62:65], v110 offset:19456
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[140:145], v[128:133], a[92:95], v12, v15 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[66:67], v111 offset:33280
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[146:151], v[128:133], a[124:127], v12, v15 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s49 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[146:151], v[122:127], a[120:123], v12, v15 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[68:71], v110 offset:20480
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[146:151], v[116:121], a[116:119], v12, v15 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[72:73], v111 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[146:151], v[104:109], a[112:115], v12, v15 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s18
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[146:151], v[98:103], a[108:111], v12, v14 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s48 offen lds
	ds_read_b128 v[80:83], v110 offset:21504
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[146:151], v[92:97], a[104:107], v12, v14 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[84:85], v111 offset:34304
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[146:151], v[74:79], a[100:103], v12, v14 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[146:151], v[20:25], a[96:99], v12, v14 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[134:137], v6 offset:56320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[86:91], v[20:25], a[128:131], v13, v14 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[138:139], v19 offset:6656
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[86:91], v[74:79], a[132:135], v13, v14 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s48 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[86:91], v[92:97], a[136:139], v13, v14 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[140:143], v6 offset:57344
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[86:91], v[98:103], a[140:143], v13, v14 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[144:145], v19 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[86:91], v[104:109], a[144:147], v13, v15 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[86:91], v[116:121], a[148:151], v13, v15 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s48 offen offset:2048 lds
	ds_read_b128 v[146:149], v6 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[86:91], v[122:127], a[152:155], v13, v15 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[150:151], v19 offset:7680
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[86:91], v[128:133], a[156:159], v13, v15 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[164:169], v[128:133], a[188:191], v13, v15 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[86:89], v6 offset:59392
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[164:169], v[122:127], a[184:187], v13, v15 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[90:91], v19 offset:8192
	buffer_load_dwordx4 v1, s[24:27], s48 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[164:169], v[116:121], a[180:183], v13, v15 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[164:169], v[104:109], a[176:179], v13, v15 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[158:161], v6 offset:60416
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[164:169], v[98:103], a[172:175], v13, v14 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[162:163], v19 offset:8704
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[164:169], v[92:97], a[168:171], v13, v14 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s19
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[164:169], v[74:79], a[164:167], v13, v14 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s49 offen lds
	ds_read_b128 v[170:173], v6 offset:61440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[164:169], v[20:25], a[160:163], v13, v14 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[174:175], v19 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[176:181], v[20:25], a[192:195], v13, v14 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[176:181], v[74:79], a[196:199], v13, v14 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[164:167], v6 offset:62464
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[176:181], v[92:97], a[200:203], v13, v14 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[168:169], v19 offset:9728
	buffer_load_dwordx4 v1, s[24:27], s49 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[176:181], v[98:103], a[204:207], v13, v14 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[176:181], v[104:109], a[208:211], v13, v15 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[176:181], v[116:121], a[212:215], v13, v15 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[176:181], v[122:127], a[216:219], v13, v15 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s48, s42, 0x30000
	s_mov_b32 m0, s7
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[176:181], v[128:133], a[220:223], v13, v15 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[28:31], s48 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[152:157], v[128:133], a[252:255], v13, v15 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[152:157], v[122:127], a[248:251], v13, v15 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[152:157], v[116:121], a[244:247], v13, v15 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s48, s46, 0x17fc00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[152:157], v[104:109], a[240:243], v13, v15 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[32:35], s48 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[152:157], v[98:103], a[236:239], v13, v14 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[152:157], v[92:97], a[232:235], v13, v14 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[152:157], v[74:79], a[228:231], v13, v14 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[152:157], v[20:25], a[224:227], v13, v14 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:31], v[32:37], a[0:3], v8, v10 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[116:117], v5 offset:4096
	ds_read_b64 v[118:119], v3 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[26:31], v[38:43], a[4:7], v8, v10 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_i32 s48, s13, 0x4e000
	s_mov_b32 m0, s38
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[26:31], v[44:49], a[8:11], v8, v10 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s48 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[26:31], v[50:55], a[12:15], v8, v10 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[26:31], v[56:61], a[16:19], v8, v11 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[20:23], v112 offset:38912
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[26:31], v[62:67], a[20:23], v8, v11 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[24:25], v19 offset:55296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[26:31], v[68:73], a[24:27], v8, v11 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s48 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[26:31], v[80:85], a[28:31], v8, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[120:123], v110 offset:63488
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[134:139], v[80:85], a[60:63], v8, v11 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[124:125], v113 offset:14336
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[134:139], v[68:73], a[56:59], v8, v11 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[134:139], v[62:67], a[52:55], v8, v11 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s48 offen offset:2048 lds
	ds_read_b128 v[126:129], v110 offset:64512
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[134:139], v[56:61], a[48:51], v8, v11 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[130:131], v113 offset:14848
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[134:139], v[50:55], a[44:47], v8, v10 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[134:139], v[44:49], a[40:43], v8, v10 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[152:155], v114
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[134:139], v[38:43], a[36:39], v8, v10 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[156:157], v113 offset:15360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[134:139], v[32:37], a[32:35], v8, v10 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s48 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[140:145], v[32:37], a[64:67], v8, v10 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[132:135], v114 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[140:145], v[38:43], a[68:71], v8, v10 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[136:137], v113 offset:15872
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[140:145], v[44:49], a[72:75], v8, v10 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s49, s13, 0x4f000
	s_mov_b32 m0, s39
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[140:145], v[50:55], a[76:79], v8, v10 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s49 offen lds
	ds_read_b128 v[176:179], v114 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[140:145], v[56:61], a[80:83], v8, v11 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[180:181], v113 offset:16384
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[140:145], v[62:67], a[84:87], v8, v11 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[140:145], v[68:73], a[88:91], v8, v11 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[182:185], v114 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[140:145], v[80:85], a[92:95], v8, v11 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[186:187], v113 offset:16896
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[146:151], v[80:85], a[124:127], v8, v11 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s49 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[146:151], v[68:73], a[120:123], v8, v11 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[138:141], v114 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[146:151], v[62:67], a[116:119], v8, v11 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[142:143], v113 offset:17408
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[146:151], v[56:61], a[112:115], v8, v11 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s41
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[146:151], v[50:55], a[108:111], v8, v10 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s48 offen lds
	ds_read_b128 v[188:191], v114 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[146:151], v[44:49], a[104:107], v8, v10 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[192:193], v113 offset:17920
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[146:151], v[38:43], a[100:103], v8, v10 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[146:151], v[32:37], a[96:99], v8, v10 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[74:77], v112 offset:39936
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[86:91], v[32:37], a[128:131], v9, v10 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v19 offset:55808
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[86:91], v[38:43], a[132:135], v9, v10 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s48 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[86:91], v[44:49], a[136:139], v9, v10 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[92:95], v112 offset:40960
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[86:91], v[50:55], a[140:143], v9, v10 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[96:97], v19 offset:56320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[86:91], v[56:61], a[144:147], v9, v11 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[86:91], v[62:67], a[148:151], v9, v11 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s48 offen offset:2048 lds
	ds_read_b128 v[98:101], v112 offset:41984
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[86:91], v[68:73], a[152:155], v9, v11 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[102:103], v19 offset:56832
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[86:91], v[80:85], a[156:159], v9, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[158:163], v[80:85], a[188:191], v9, v11 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[86:89], v112 offset:43008
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[158:163], v[68:73], a[184:187], v9, v11 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[90:91], v19 offset:57344
	buffer_load_dwordx4 v1, s[24:27], s48 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[158:163], v[62:67], a[180:183], v9, v11 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[158:163], v[56:61], a[176:179], v9, v11 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[104:107], v112 offset:44032
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[158:163], v[50:55], a[172:175], v9, v10 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[108:109], v19 offset:57856
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[158:163], v[44:49], a[168:171], v9, v10 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s37
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[158:163], v[38:43], a[164:167], v9, v10 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s49 offen lds
	ds_read_b128 v[144:147], v112 offset:45056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[158:163], v[32:37], a[160:163], v9, v10 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[148:149], v19 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[170:175], v[32:37], a[192:195], v9, v10 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[170:175], v[38:43], a[196:199], v9, v10 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[158:161], v112 offset:46080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[170:175], v[44:49], a[200:203], v9, v10 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[162:163], v19 offset:58880
	buffer_load_dwordx4 v1, s[24:27], s49 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[170:175], v[50:55], a[204:207], v9, v10 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[170:175], v[56:61], a[208:211], v9, v11 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[170:175], v[62:67], a[212:215], v9, v11 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[170:175], v[68:73], a[216:219], v9, v11 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s48, s42, 0x34000
	s_mov_b32 m0, s40
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[170:175], v[80:85], a[220:223], v9, v11 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[28:31], s48 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[164:169], v[80:85], a[252:255], v9, v11 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[164:169], v[68:73], a[248:251], v9, v11 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[164:169], v[62:67], a[244:247], v9, v11 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s48, s46, 0x19fc00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[164:169], v[56:61], a[240:243], v9, v11 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[32:35], s48 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[164:169], v[50:55], a[236:239], v9, v10 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[164:169], v[44:49], a[232:235], v9, v10 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[164:169], v[38:43], a[228:231], v9, v10 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[164:169], v[32:37], a[224:227], v9, v10 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:25], v[120:125], a[0:3], v116, v118 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[8:9], v5
	ds_read_b64 v[10:11], v3 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[20:25], v[126:131], a[4:7], v116, v118 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_i32 s48, s13, 0x54000
	s_mov_b32 m0, s44
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[20:25], v[152:157], a[8:11], v116, v118 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s48 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[20:25], v[132:137], a[12:15], v116, v118 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[20:25], v[176:181], a[16:19], v116, v119 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[12:15], v6 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[20:25], v[182:187], a[20:23], v116, v119 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[16:17], v4 offset:22528
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[20:25], v[138:143], a[24:27], v116, v119 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s48 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[20:25], v[188:193], a[28:31], v116, v119 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[20:23], v18 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[74:79], v[188:193], a[60:63], v116, v119 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[24:25], v7 offset:47104
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[74:79], v[138:143], a[56:59], v116, v119 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[74:79], v[182:187], a[52:55], v116, v119 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s48 offen offset:2048 lds
	ds_read_b128 v[26:29], v18 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[74:79], v[176:181], a[48:51], v116, v119 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[30:31], v7 offset:47616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[74:79], v[132:137], a[44:47], v116, v118 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[74:79], v[152:157], a[40:43], v116, v118 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[32:35], v18 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[74:79], v[126:131], a[36:39], v116, v118 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[36:37], v7 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[74:79], v[120:125], a[32:35], v116, v118 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s48 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[92:97], v[120:125], a[64:67], v116, v118 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[38:41], v18 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[92:97], v[126:131], a[68:71], v116, v118 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[42:43], v7 offset:48640
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[92:97], v[152:157], a[72:75], v116, v118 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s13, s13, 0x55000
	s_mov_b32 m0, s45
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[92:97], v[132:137], a[76:79], v116, v118 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s13 offen lds
	ds_read_b128 v[44:47], v18 offset:34816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[92:97], v[176:181], a[80:83], v116, v119 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[48:49], v7 offset:49152
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[92:97], v[182:187], a[84:87], v116, v119 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[92:97], v[138:143], a[88:91], v116, v119 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[50:53], v18 offset:35840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[92:97], v[188:193], a[92:95], v116, v119 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[54:55], v7 offset:49664
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[98:103], v[188:193], a[124:127], v116, v119 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s13 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[98:103], v[138:143], a[120:123], v116, v119 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[56:59], v18 offset:36864
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[98:103], v[182:187], a[116:119], v116, v119 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[60:61], v7 offset:50176
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[98:103], v[176:181], a[112:115], v116, v119 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s47
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[98:103], v[132:137], a[108:111], v116, v118 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s48 offen lds
	ds_read_b128 v[62:65], v18 offset:37888
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[98:103], v[152:157], a[104:107], v116, v118 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[66:67], v7 offset:50688
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[98:103], v[126:131], a[100:103], v116, v118 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[98:103], v[120:125], a[96:99], v116, v118 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[68:71], v6 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[86:91], v[120:125], a[128:131], v117, v118 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[72:73], v4 offset:23040
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[86:91], v[126:131], a[132:135], v117, v118 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s48 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[86:91], v[152:157], a[136:139], v117, v118 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[74:77], v6 offset:8192
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[86:91], v[132:137], a[140:143], v117, v118 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v4 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[86:91], v[176:181], a[144:147], v117, v119 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[86:91], v[182:187], a[148:151], v117, v119 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s48 offen offset:2048 lds
	ds_read_b128 v[80:83], v6 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[86:91], v[138:143], a[152:155], v117, v119 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[84:85], v4 offset:24064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[86:91], v[188:193], a[156:159], v117, v119 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[104:109], v[188:193], a[188:191], v117, v119 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[86:89], v6 offset:10240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[104:109], v[138:143], a[184:187], v117, v119 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[90:91], v4 offset:24576
	buffer_load_dwordx4 v1, s[24:27], s48 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[104:109], v[182:187], a[180:183], v117, v119 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[104:109], v[176:181], a[176:179], v117, v119 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[92:95], v6 offset:11264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[104:109], v[132:137], a[172:175], v117, v118 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[96:97], v4 offset:25088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[104:109], v[152:157], a[168:171], v117, v118 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s6
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[104:109], v[126:131], a[164:167], v117, v118 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s13 offen lds
	ds_read_b128 v[98:101], v6 offset:12288
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[104:109], v[120:125], a[160:163], v117, v118 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[102:103], v4 offset:25600
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[144:149], v[120:125], a[192:195], v117, v118 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[144:149], v[126:131], a[196:199], v117, v118 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[104:107], v6 offset:13312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[144:149], v[152:157], a[200:203], v117, v118 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[108:109], v4 offset:26112
	buffer_load_dwordx4 v1, s[24:27], s13 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[144:149], v[132:137], a[204:207], v117, v118 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[144:149], v[176:181], a[208:211], v117, v119 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[144:149], v[182:187], a[212:215], v117, v119 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[144:149], v[138:143], a[216:219], v117, v119 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s13, s42, 0x38000
	s_mov_b32 m0, s43
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[144:149], v[188:193], a[220:223], v117, v119 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[28:31], s13 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[158:163], v[188:193], a[252:255], v117, v119 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[158:163], v[138:143], a[248:251], v117, v119 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[158:163], v[182:187], a[244:247], v117, v119 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s13, s46, 0x1bfc00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[158:163], v[176:181], a[240:243], v117, v119 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[32:35], s13 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[158:163], v[132:137], a[236:239], v117, v118 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[158:163], v[152:157], a[232:235], v117, v118 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[158:163], v[126:131], a[228:231], v117, v118 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[158:163], v[120:125], a[224:227], v117, v118 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s3, s3, 1
	s_cmp_lt_i32 s3, 3
	s_cbranch_scc1 .Lmain_loop
	s_add_i32 s1, s9, 0x4e0
	s_lshl_b32 s1, s1, 9
	s_add_i32 s2, s12, 0x2700
	s_lshl_b32 s2, s2, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[12:17], v[20:25], a[0:3], v8, v10 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[110:111], v5 offset:2048
	ds_read_b64 v[112:113], v3 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[12:17], v[26:31], a[4:7], v8, v10 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_i32 s3, s17, 0xea000
	s_mov_b32 m0, s16
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[12:17], v[32:37], a[8:11], v8, v10 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s3 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[12:17], v[38:43], a[12:15], v8, v10 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[12:17], v[44:49], a[16:19], v8, v11 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[114:117], v6 offset:55296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[12:17], v[50:55], a[20:23], v8, v11 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_add_u32_e32 v19, 0x10000, v4
	ds_read_b64 v[118:119], v19 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[12:17], v[56:61], a[24:27], v8, v11 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s3 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[12:17], v[62:67], a[28:31], v8, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_add_u32_e32 v12, 0x10000, v18
	ds_read_b128 v[120:123], v12 offset:14336
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[68:73], v[62:67], a[60:63], v8, v11 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_add_u32_e32 v13, 0x10000, v7
	ds_read_b64 v[124:125], v13 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[68:73], v[56:61], a[56:59], v8, v11 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[68:73], v[50:55], a[52:55], v8, v11 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s3 offen offset:2048 lds
	ds_read_b128 v[126:129], v12 offset:15360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[68:73], v[44:49], a[48:51], v8, v11 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[130:131], v13 offset:31232
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[68:73], v[38:43], a[44:47], v8, v10 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[68:73], v[32:37], a[40:43], v8, v10 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[132:135], v12 offset:16384
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[68:73], v[26:31], a[36:39], v8, v10 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[136:137], v13 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[68:73], v[20:25], a[32:35], v8, v10 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s3 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[74:79], v[20:25], a[64:67], v8, v10 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[68:71], v12 offset:17408
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[74:79], v[26:31], a[68:71], v8, v10 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[72:73], v13 offset:32256
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[74:79], v[32:37], a[72:75], v8, v10 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s13, s17, 0xeb000
	s_mov_b32 m0, s8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[74:79], v[38:43], a[76:79], v8, v10 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s13 offen lds
	ds_read_b128 v[138:141], v12 offset:18432
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[74:79], v[44:49], a[80:83], v8, v11 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[142:143], v13 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[74:79], v[50:55], a[84:87], v8, v11 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[74:79], v[56:61], a[88:91], v8, v11 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[144:147], v12 offset:19456
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[74:79], v[62:67], a[92:95], v8, v11 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[148:149], v13 offset:33280
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[80:85], v[62:67], a[124:127], v8, v11 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s13 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[80:85], v[56:61], a[120:123], v8, v11 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[74:77], v12 offset:20480
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[80:85], v[50:55], a[116:119], v8, v11 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v13 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[80:85], v[44:49], a[112:115], v8, v11 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s18
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[80:85], v[38:43], a[108:111], v8, v10 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s3 offen lds
	ds_read_b128 v[150:153], v12 offset:21504
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[80:85], v[32:37], a[104:107], v8, v10 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[154:155], v13 offset:34304
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[80:85], v[26:31], a[100:103], v8, v10 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[80:85], v[20:25], a[96:99], v8, v10 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[80:83], v6 offset:56320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[86:91], v[20:25], a[128:131], v9, v10 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[84:85], v19 offset:6656
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[86:91], v[26:31], a[132:135], v9, v10 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s3 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[86:91], v[32:37], a[136:139], v9, v10 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[156:159], v6 offset:57344
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[86:91], v[38:43], a[140:143], v9, v10 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[160:161], v19 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[86:91], v[44:49], a[144:147], v9, v11 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[86:91], v[50:55], a[148:151], v9, v11 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s3 offen offset:2048 lds
	ds_read_b128 v[162:165], v6 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[86:91], v[56:61], a[152:155], v9, v11 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[166:167], v19 offset:7680
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[86:91], v[62:67], a[156:159], v9, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[92:97], v[62:67], a[188:191], v9, v11 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[86:89], v6 offset:59392
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[92:97], v[56:61], a[184:187], v9, v11 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[90:91], v19 offset:8192
	buffer_load_dwordx4 v1, s[24:27], s3 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[92:97], v[50:55], a[180:183], v9, v11 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[92:97], v[44:49], a[176:179], v9, v11 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[168:171], v6 offset:60416
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[92:97], v[38:43], a[172:175], v9, v10 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[172:173], v19 offset:8704
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[92:97], v[32:37], a[168:171], v9, v10 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s19
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[92:97], v[26:31], a[164:167], v9, v10 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s13 offen lds
	ds_read_b128 v[174:177], v6 offset:61440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[92:97], v[20:25], a[160:163], v9, v10 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[178:179], v19 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[98:103], v[20:25], a[192:195], v9, v10 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[98:103], v[26:31], a[196:199], v9, v10 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[92:95], v6 offset:62464
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[98:103], v[32:37], a[200:203], v9, v10 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[96:97], v19 offset:9728
	buffer_load_dwordx4 v1, s[24:27], s13 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[98:103], v[38:43], a[204:207], v9, v10 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[98:103], v[44:49], a[208:211], v9, v11 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[98:103], v[50:55], a[212:215], v9, v11 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[98:103], v[56:61], a[216:219], v9, v11 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s1, s7, s1
	s_mov_b32 m0, s7
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[98:103], v[62:67], a[220:223], v9, v11 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[28:31], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[104:109], v[62:67], a[252:255], v9, v11 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[104:109], v[56:61], a[248:251], v9, v11 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[104:109], v[50:55], a[244:247], v9, v11 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s1, s7, s2
	s_mov_b32 m0, s14
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[104:109], v[44:49], a[240:243], v9, v11 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[32:35], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[104:109], v[38:43], a[236:239], v9, v10 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[104:109], v[32:37], a[232:235], v9, v10 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[104:109], v[26:31], a[228:231], v9, v10 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[104:109], v[20:25], a[224:227], v9, v10 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s1, s9, 0x500
	s_lshl_b32 s1, s1, 9
	s_add_i32 s2, s12, 0x2800
	s_lshl_b32 s2, s2, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[114:119], v[120:125], a[0:3], v110, v112 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[8:9], v5 offset:4096
	ds_read_b64 v[10:11], v3 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[114:119], v[126:131], a[4:7], v110, v112 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_i32 s3, s17, 0xf0000
	s_mov_b32 m0, s38
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[114:119], v[132:137], a[8:11], v110, v112 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s3 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[114:119], v[68:73], a[12:15], v110, v112 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[114:119], v[138:143], a[16:19], v110, v113 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_add_u32_e32 v14, 0x10000, v6
	ds_read_b128 v[20:23], v14 offset:38912
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[114:119], v[144:149], a[20:23], v110, v113 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[24:25], v19 offset:55296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[114:119], v[74:79], a[24:27], v110, v113 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s3 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[114:119], v[150:155], a[28:31], v110, v113 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[26:29], v12 offset:63488
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[80:85], v[150:155], a[60:63], v110, v113 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_add_u32_e32 v15, 0x20000, v7
	ds_read_b64 v[30:31], v15 offset:14336
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[80:85], v[74:79], a[56:59], v110, v113 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[80:85], v[144:149], a[52:55], v110, v113 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s3 offen offset:2048 lds
	ds_read_b128 v[32:35], v12 offset:64512
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[80:85], v[138:143], a[48:51], v110, v113 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[36:37], v15 offset:14848
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[80:85], v[68:73], a[44:47], v110, v112 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[80:85], v[132:137], a[40:43], v110, v112 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_add_u32_e32 v16, 0x20000, v18
	ds_read_b128 v[38:41], v16
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[80:85], v[126:131], a[36:39], v110, v112 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[42:43], v15 offset:15360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[80:85], v[120:125], a[32:35], v110, v112 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s3 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[156:161], v[120:125], a[64:67], v110, v112 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[44:47], v16 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[156:161], v[126:131], a[68:71], v110, v112 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[48:49], v15 offset:15872
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[156:161], v[132:137], a[72:75], v110, v112 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s13, s17, 0xf1000
	s_mov_b32 m0, s39
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[156:161], v[68:73], a[76:79], v110, v112 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s13 offen lds
	ds_read_b128 v[50:53], v16 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[156:161], v[138:143], a[80:83], v110, v113 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[54:55], v15 offset:16384
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[156:161], v[144:149], a[84:87], v110, v113 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[156:161], v[74:79], a[88:91], v110, v113 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[56:59], v16 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[156:161], v[150:155], a[92:95], v110, v113 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[60:61], v15 offset:16896
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[162:167], v[150:155], a[124:127], v110, v113 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s13 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[162:167], v[74:79], a[120:123], v110, v113 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[62:65], v16 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[162:167], v[144:149], a[116:119], v110, v113 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[66:67], v15 offset:17408
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[162:167], v[138:143], a[112:115], v110, v113 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s41
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[162:167], v[68:73], a[108:111], v110, v112 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s3 offen lds
	ds_read_b128 v[80:83], v16 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[162:167], v[132:137], a[104:107], v110, v112 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[84:85], v15 offset:17920
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[162:167], v[126:131], a[100:103], v110, v112 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[162:167], v[120:125], a[96:99], v110, v112 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[98:101], v14 offset:39936
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[86:91], v[120:125], a[128:131], v111, v112 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[102:103], v19 offset:55808
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[86:91], v[126:131], a[132:135], v111, v112 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s3 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[86:91], v[132:137], a[136:139], v111, v112 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[104:107], v14 offset:40960
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[86:91], v[68:73], a[140:143], v111, v112 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[108:109], v19 offset:56320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[86:91], v[138:143], a[144:147], v111, v113 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[86:91], v[144:149], a[148:151], v111, v113 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s3 offen offset:2048 lds
	ds_read_b128 v[114:117], v14 offset:41984
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[86:91], v[74:79], a[152:155], v111, v113 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[118:119], v19 offset:56832
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[86:91], v[150:155], a[156:159], v111, v113 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[168:173], v[150:155], a[188:191], v111, v113 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[86:89], v14 offset:43008
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[168:173], v[74:79], a[184:187], v111, v113 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[90:91], v19 offset:57344
	buffer_load_dwordx4 v1, s[24:27], s3 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[168:173], v[144:149], a[180:183], v111, v113 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[168:173], v[138:143], a[176:179], v111, v113 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[156:159], v14 offset:44032
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[168:173], v[68:73], a[172:175], v111, v112 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[160:161], v19 offset:57856
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[168:173], v[132:137], a[168:171], v111, v112 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s37
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[168:173], v[126:131], a[164:167], v111, v112 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s13 offen lds
	ds_read_b128 v[162:165], v14 offset:45056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[168:173], v[120:125], a[160:163], v111, v112 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[166:167], v19 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[174:179], v[120:125], a[192:195], v111, v112 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[174:179], v[126:131], a[196:199], v111, v112 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[168:171], v14 offset:46080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[174:179], v[132:137], a[200:203], v111, v112 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[172:173], v19 offset:58880
	buffer_load_dwordx4 v1, s[24:27], s13 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[174:179], v[68:73], a[204:207], v111, v112 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[174:179], v[138:143], a[208:211], v111, v113 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[174:179], v[144:149], a[212:215], v111, v113 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[174:179], v[74:79], a[216:219], v111, v113 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s1, s7, s1
	s_mov_b32 m0, s40
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[174:179], v[150:155], a[220:223], v111, v113 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[28:31], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[92:97], v[150:155], a[252:255], v111, v113 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[92:97], v[74:79], a[248:251], v111, v113 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[92:97], v[144:149], a[244:247], v111, v113 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s1, s7, s2
	s_mov_b32 m0, s15
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[92:97], v[138:143], a[240:243], v111, v113 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[32:35], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[92:97], v[68:73], a[236:239], v111, v112 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[92:97], v[132:137], a[232:235], v111, v112 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[92:97], v[126:131], a[228:231], v111, v112 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[92:97], v[120:125], a[224:227], v111, v112 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s1, s9, 0x520
	s_lshl_b32 s1, s1, 9
	s_add_i32 s2, s12, 0x2900
	s_lshl_b32 s2, s2, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:25], v[26:31], a[0:3], v8, v10 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[68:69], v5
	ds_read_b64 v[70:71], v3 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[20:25], v[32:37], a[4:7], v8, v10 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_i32 s3, s17, 0xf6000
	s_mov_b32 m0, s44
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[20:25], v[38:43], a[8:11], v8, v10 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s3 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[20:25], v[44:49], a[12:15], v8, v10 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[20:25], v[50:55], a[16:19], v8, v11 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[72:75], v6 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[20:25], v[56:61], a[20:23], v8, v11 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[76:77], v4 offset:22528
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[20:25], v[62:67], a[24:27], v8, v11 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s3 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[20:25], v[80:85], a[28:31], v8, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[20:23], v18 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[98:103], v[80:85], a[60:63], v8, v11 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[24:25], v7 offset:47104
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[98:103], v[62:67], a[56:59], v8, v11 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[98:103], v[56:61], a[52:55], v8, v11 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s3 offen offset:2048 lds
	ds_read_b128 v[92:95], v18 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[98:103], v[50:55], a[48:51], v8, v11 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[96:97], v7 offset:47616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[98:103], v[44:49], a[44:47], v8, v10 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[98:103], v[38:43], a[40:43], v8, v10 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[120:123], v18 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[98:103], v[32:37], a[36:39], v8, v10 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[124:125], v7 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[98:103], v[26:31], a[32:35], v8, v10 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s3 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[104:109], v[26:31], a[64:67], v8, v10 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[98:101], v18 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[104:109], v[32:37], a[68:71], v8, v10 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[102:103], v7 offset:48640
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[104:109], v[38:43], a[72:75], v8, v10 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s13, s17, 0xf7000
	s_mov_b32 m0, s45
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[104:109], v[44:49], a[76:79], v8, v10 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s13 offen lds
	ds_read_b128 v[126:129], v18 offset:34816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[104:109], v[50:55], a[80:83], v8, v11 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[130:131], v7 offset:49152
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[104:109], v[56:61], a[84:87], v8, v11 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[104:109], v[62:67], a[88:91], v8, v11 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[132:135], v18 offset:35840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[104:109], v[80:85], a[92:95], v8, v11 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[136:137], v7 offset:49664
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[114:119], v[80:85], a[124:127], v8, v11 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s13 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[114:119], v[62:67], a[120:123], v8, v11 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[104:107], v18 offset:36864
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[114:119], v[56:61], a[116:119], v8, v11 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[108:109], v7 offset:50176
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[114:119], v[50:55], a[112:115], v8, v11 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s47
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[114:119], v[44:49], a[108:111], v8, v10 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s3 offen lds
	ds_read_b128 v[138:141], v18 offset:37888
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[114:119], v[38:43], a[104:107], v8, v10 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[142:143], v7 offset:50688
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[114:119], v[32:37], a[100:103], v8, v10 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[114:119], v[26:31], a[96:99], v8, v10 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[110:113], v6 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[86:91], v[26:31], a[128:131], v9, v10 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[114:115], v4 offset:23040
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[86:91], v[32:37], a[132:135], v9, v10 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s3 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[86:91], v[38:43], a[136:139], v9, v10 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[144:147], v6 offset:8192
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[86:91], v[44:49], a[140:143], v9, v10 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[148:149], v4 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[86:91], v[50:55], a[144:147], v9, v11 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[86:91], v[56:61], a[148:151], v9, v11 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s3 offen offset:2048 lds
	ds_read_b128 v[150:153], v6 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[86:91], v[62:67], a[152:155], v9, v11 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[154:155], v4 offset:24064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[86:91], v[80:85], a[156:159], v9, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[156:161], v[80:85], a[188:191], v9, v11 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[86:89], v6 offset:10240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[156:161], v[62:67], a[184:187], v9, v11 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[90:91], v4 offset:24576
	buffer_load_dwordx4 v1, s[24:27], s3 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[156:161], v[56:61], a[180:183], v9, v11 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[156:161], v[50:55], a[176:179], v9, v11 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[174:177], v6 offset:11264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[156:161], v[44:49], a[172:175], v9, v10 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[178:179], v4 offset:25088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[156:161], v[38:43], a[168:171], v9, v10 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s6
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[156:161], v[32:37], a[164:167], v9, v10 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s13 offen lds
	ds_read_b128 v[180:183], v6 offset:12288
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[156:161], v[26:31], a[160:163], v9, v10 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[184:185], v4 offset:25600
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[162:167], v[26:31], a[192:195], v9, v10 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[162:167], v[32:37], a[196:199], v9, v10 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[156:159], v6 offset:13312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[162:167], v[38:43], a[200:203], v9, v10 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[160:161], v4 offset:26112
	buffer_load_dwordx4 v1, s[24:27], s13 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[162:167], v[44:49], a[204:207], v9, v10 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[162:167], v[50:55], a[208:211], v9, v11 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[162:167], v[56:61], a[212:215], v9, v11 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[162:167], v[62:67], a[216:219], v9, v11 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s1, s7, s1
	s_mov_b32 m0, s43
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[162:167], v[80:85], a[220:223], v9, v11 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[28:31], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[168:173], v[80:85], a[252:255], v9, v11 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[168:173], v[62:67], a[248:251], v9, v11 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[168:173], v[56:61], a[244:247], v9, v11 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s1, s7, s2
	s_mov_b32 m0, s36
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[168:173], v[50:55], a[240:243], v9, v11 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[32:35], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[168:173], v[44:49], a[236:239], v9, v10 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[168:173], v[38:43], a[232:235], v9, v10 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[168:173], v[32:37], a[228:231], v9, v10 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[168:173], v[26:31], a[224:227], v9, v10 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s1, s9, 0x540
	s_lshl_b32 s1, s1, 9
	s_add_i32 s2, s12, 0x2a00
	s_lshl_b32 s2, s2, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[72:77], v[20:25], a[0:3], v68, v70 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[8:9], v5 offset:2048
	ds_read_b64 v[10:11], v3 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[72:77], v[92:97], a[4:7], v68, v70 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_i32 s3, s17, 0xfc000
	s_mov_b32 m0, s16
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[72:77], v[120:125], a[8:11], v68, v70 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s3 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[72:77], v[98:103], a[12:15], v68, v70 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[72:77], v[126:131], a[16:19], v68, v71 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[26:29], v6 offset:55296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[72:77], v[132:137], a[20:23], v68, v71 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[30:31], v19 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[72:77], v[104:109], a[24:27], v68, v71 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s3 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[72:77], v[138:143], a[28:31], v68, v71 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[32:35], v12 offset:14336
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[110:115], v[138:143], a[60:63], v68, v71 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[36:37], v13 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[110:115], v[104:109], a[56:59], v68, v71 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[110:115], v[132:137], a[52:55], v68, v71 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s3 offen offset:2048 lds
	ds_read_b128 v[38:41], v12 offset:15360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[110:115], v[126:131], a[48:51], v68, v71 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[42:43], v13 offset:31232
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[110:115], v[98:103], a[44:47], v68, v70 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[110:115], v[120:125], a[40:43], v68, v70 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[44:47], v12 offset:16384
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[110:115], v[92:97], a[36:39], v68, v70 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[48:49], v13 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[110:115], v[20:25], a[32:35], v68, v70 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s3 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[144:149], v[20:25], a[64:67], v68, v70 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[50:53], v12 offset:17408
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[144:149], v[92:97], a[68:71], v68, v70 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[54:55], v13 offset:32256
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[144:149], v[120:125], a[72:75], v68, v70 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s13, s17, 0xfd000
	s_mov_b32 m0, s8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[144:149], v[98:103], a[76:79], v68, v70 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s13 offen lds
	ds_read_b128 v[56:59], v12 offset:18432
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[144:149], v[126:131], a[80:83], v68, v71 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[60:61], v13 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[144:149], v[132:137], a[84:87], v68, v71 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[144:149], v[104:109], a[88:91], v68, v71 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[62:65], v12 offset:19456
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[144:149], v[138:143], a[92:95], v68, v71 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[66:67], v13 offset:33280
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[150:155], v[138:143], a[124:127], v68, v71 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s13 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[150:155], v[104:109], a[120:123], v68, v71 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[72:75], v12 offset:20480
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[150:155], v[132:137], a[116:119], v68, v71 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[76:77], v13 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[150:155], v[126:131], a[112:115], v68, v71 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s18
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[150:155], v[98:103], a[108:111], v68, v70 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s3 offen lds
	ds_read_b128 v[78:81], v12 offset:21504
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[150:155], v[120:125], a[104:107], v68, v70 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[82:83], v13 offset:34304
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[150:155], v[92:97], a[100:103], v68, v70 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[150:155], v[20:25], a[96:99], v68, v70 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[110:113], v6 offset:56320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[86:91], v[20:25], a[128:131], v69, v70 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[114:115], v19 offset:6656
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[86:91], v[92:97], a[132:135], v69, v70 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s3 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[86:91], v[120:125], a[136:139], v69, v70 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[144:147], v6 offset:57344
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[86:91], v[98:103], a[140:143], v69, v70 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[148:149], v19 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[86:91], v[126:131], a[144:147], v69, v71 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[86:91], v[132:137], a[148:151], v69, v71 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s3 offen offset:2048 lds
	ds_read_b128 v[150:153], v6 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[86:91], v[104:109], a[152:155], v69, v71 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[154:155], v19 offset:7680
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[86:91], v[138:143], a[156:159], v69, v71 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[174:179], v[138:143], a[188:191], v69, v71 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[84:87], v6 offset:59392
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[174:179], v[104:109], a[184:187], v69, v71 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[88:89], v19 offset:8192
	buffer_load_dwordx4 v1, s[24:27], s3 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[174:179], v[132:137], a[180:183], v69, v71 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[174:179], v[126:131], a[176:179], v69, v71 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[162:165], v6 offset:60416
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[174:179], v[98:103], a[172:175], v69, v70 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[166:167], v19 offset:8704
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[174:179], v[120:125], a[168:171], v69, v70 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s19
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[174:179], v[92:97], a[164:167], v69, v70 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s13 offen lds
	ds_read_b128 v[168:171], v6 offset:61440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[174:179], v[20:25], a[160:163], v69, v70 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[172:173], v19 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[180:185], v[20:25], a[192:195], v69, v70 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[180:185], v[92:97], a[196:199], v69, v70 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[174:177], v6 offset:62464
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[180:185], v[120:125], a[200:203], v69, v70 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[178:179], v19 offset:9728
	buffer_load_dwordx4 v1, s[24:27], s13 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[180:185], v[98:103], a[204:207], v69, v70 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[180:185], v[126:131], a[208:211], v69, v71 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[180:185], v[132:137], a[212:215], v69, v71 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[180:185], v[104:109], a[216:219], v69, v71 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s1, s7, s1
	s_mov_b32 m0, s7
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[180:185], v[138:143], a[220:223], v69, v71 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[28:31], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[156:161], v[138:143], a[252:255], v69, v71 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[156:161], v[104:109], a[248:251], v69, v71 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[156:161], v[132:137], a[244:247], v69, v71 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s1, s7, s2
	s_mov_b32 m0, s14
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[156:161], v[126:131], a[240:243], v69, v71 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[32:35], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[156:161], v[98:103], a[236:239], v69, v70 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[156:161], v[120:125], a[232:235], v69, v70 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[156:161], v[92:97], a[228:231], v69, v70 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[156:161], v[20:25], a[224:227], v69, v70 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s1, s9, 0x560
	s_lshl_b32 s1, s1, 9
	s_add_i32 s2, s12, 0x2b00
	s_lshl_b32 s2, s2, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:31], v[32:37], a[0:3], v8, v10 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[20:21], v5 offset:4096
	ds_read_b64 v[22:23], v3 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[26:31], v[38:43], a[4:7], v8, v10 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_i32 s3, s17, 0x102000
	s_mov_b32 m0, s38
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[26:31], v[44:49], a[8:11], v8, v10 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s3 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[26:31], v[50:55], a[12:15], v8, v10 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[26:31], v[56:61], a[16:19], v8, v11 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[90:93], v14 offset:38912
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[26:31], v[62:67], a[20:23], v8, v11 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[94:95], v19 offset:55296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[26:31], v[72:77], a[24:27], v8, v11 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s3 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[26:31], v[78:83], a[28:31], v8, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[24:27], v12 offset:63488
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[110:115], v[78:83], a[60:63], v8, v11 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[28:29], v15 offset:14336
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[110:115], v[72:77], a[56:59], v8, v11 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[110:115], v[62:67], a[52:55], v8, v11 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s3 offen offset:2048 lds
	ds_read_b128 v[96:99], v12 offset:64512
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[110:115], v[56:61], a[48:51], v8, v11 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[100:101], v15 offset:14848
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[110:115], v[50:55], a[44:47], v8, v10 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[110:115], v[44:49], a[40:43], v8, v10 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[102:105], v16
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[110:115], v[38:43], a[36:39], v8, v10 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[106:107], v15 offset:15360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[110:115], v[32:37], a[32:35], v8, v10 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s3 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[144:149], v[32:37], a[64:67], v8, v10 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[108:111], v16 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[144:149], v[38:43], a[68:71], v8, v10 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[112:113], v15 offset:15872
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[144:149], v[44:49], a[72:75], v8, v10 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s13, s17, 0x103000
	s_mov_b32 m0, s39
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[144:149], v[50:55], a[76:79], v8, v10 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s13 offen lds
	ds_read_b128 v[114:117], v16 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[144:149], v[56:61], a[80:83], v8, v11 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[118:119], v15 offset:16384
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[144:149], v[62:67], a[84:87], v8, v11 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[144:149], v[72:77], a[88:91], v8, v11 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[120:123], v16 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[144:149], v[78:83], a[92:95], v8, v11 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[124:125], v15 offset:16896
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[150:155], v[78:83], a[124:127], v8, v11 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s13 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[150:155], v[72:77], a[120:123], v8, v11 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[126:129], v16 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[150:155], v[62:67], a[116:119], v8, v11 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[130:131], v15 offset:17408
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[150:155], v[56:61], a[112:115], v8, v11 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s41
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[150:155], v[50:55], a[108:111], v8, v10 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s3 offen lds
	ds_read_b128 v[132:135], v16 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[150:155], v[44:49], a[104:107], v8, v10 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[136:137], v15 offset:17920
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[150:155], v[38:43], a[100:103], v8, v10 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[150:155], v[32:37], a[96:99], v8, v10 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[138:141], v14 offset:39936
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[84:89], v[32:37], a[128:131], v9, v10 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[142:143], v19 offset:55808
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[84:89], v[38:43], a[132:135], v9, v10 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s3 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[84:89], v[44:49], a[136:139], v9, v10 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[144:147], v14 offset:40960
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[84:89], v[50:55], a[140:143], v9, v10 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[148:149], v19 offset:56320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[84:89], v[56:61], a[144:147], v9, v11 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[84:89], v[62:67], a[148:151], v9, v11 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s3 offen offset:2048 lds
	ds_read_b128 v[150:153], v14 offset:41984
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[84:89], v[72:77], a[152:155], v9, v11 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[154:155], v19 offset:56832
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[84:89], v[78:83], a[156:159], v9, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[162:167], v[78:83], a[188:191], v9, v11 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[84:87], v14 offset:43008
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[162:167], v[72:77], a[184:187], v9, v11 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[88:89], v19 offset:57344
	buffer_load_dwordx4 v1, s[24:27], s3 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[162:167], v[62:67], a[180:183], v9, v11 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[162:167], v[56:61], a[176:179], v9, v11 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[156:159], v14 offset:44032
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[162:167], v[50:55], a[172:175], v9, v10 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[160:161], v19 offset:57856
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[162:167], v[44:49], a[168:171], v9, v10 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s37
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[162:167], v[38:43], a[164:167], v9, v10 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s13 offen lds
	ds_read_b128 v[180:183], v14 offset:45056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[162:167], v[32:37], a[160:163], v9, v10 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[184:185], v19 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[168:173], v[32:37], a[192:195], v9, v10 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[168:173], v[38:43], a[196:199], v9, v10 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[162:165], v14 offset:46080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[168:173], v[44:49], a[200:203], v9, v10 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[166:167], v19 offset:58880
	buffer_load_dwordx4 v1, s[24:27], s13 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[168:173], v[50:55], a[204:207], v9, v10 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[168:173], v[56:61], a[208:211], v9, v11 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[168:173], v[62:67], a[212:215], v9, v11 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[168:173], v[72:77], a[216:219], v9, v11 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s1, s7, s1
	s_mov_b32 m0, s40
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[168:173], v[78:83], a[220:223], v9, v11 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[28:31], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[174:179], v[78:83], a[252:255], v9, v11 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[174:179], v[72:77], a[248:251], v9, v11 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[174:179], v[62:67], a[244:247], v9, v11 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s1, s7, s2
	s_mov_b32 m0, s15
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[174:179], v[56:61], a[240:243], v9, v11 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[32:35], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[174:179], v[50:55], a[236:239], v9, v10 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[174:179], v[44:49], a[232:235], v9, v10 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[174:179], v[38:43], a[228:231], v9, v10 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[174:179], v[32:37], a[224:227], v9, v10 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s1, s9, 0x580
	s_lshl_b32 s1, s1, 9
	s_add_i32 s2, s12, 0x2c00
	s_lshl_b32 s2, s2, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[90:95], v[24:29], a[0:3], v20, v22 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[8:9], v5
	ds_read_b64 v[10:11], v3 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[90:95], v[96:101], a[4:7], v20, v22 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_i32 s3, s17, 0x108000
	s_mov_b32 m0, s44
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[90:95], v[102:107], a[8:11], v20, v22 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s3 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[90:95], v[108:113], a[12:15], v20, v22 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[90:95], v[114:119], a[16:19], v20, v23 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[30:33], v6 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[90:95], v[120:125], a[20:23], v20, v23 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[34:35], v4 offset:22528
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[90:95], v[126:131], a[24:27], v20, v23 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s3 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[90:95], v[132:137], a[28:31], v20, v23 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[36:39], v18 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[138:143], v[132:137], a[60:63], v20, v23 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[40:41], v7 offset:47104
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[138:143], v[126:131], a[56:59], v20, v23 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[138:143], v[120:125], a[52:55], v20, v23 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s3 offen offset:2048 lds
	ds_read_b128 v[42:45], v18 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[138:143], v[114:119], a[48:51], v20, v23 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[46:47], v7 offset:47616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[138:143], v[108:113], a[44:47], v20, v22 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[138:143], v[102:107], a[40:43], v20, v22 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[48:51], v18 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[138:143], v[96:101], a[36:39], v20, v22 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[52:53], v7 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[138:143], v[24:29], a[32:35], v20, v22 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s3 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[144:149], v[24:29], a[64:67], v20, v22 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[54:57], v18 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[144:149], v[96:101], a[68:71], v20, v22 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[58:59], v7 offset:48640
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[144:149], v[102:107], a[72:75], v20, v22 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s13, s17, 0x109000
	s_mov_b32 m0, s45
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[144:149], v[108:113], a[76:79], v20, v22 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s13 offen lds
	ds_read_b128 v[60:63], v18 offset:34816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[144:149], v[114:119], a[80:83], v20, v23 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[64:65], v7 offset:49152
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[144:149], v[120:125], a[84:87], v20, v23 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[144:149], v[126:131], a[88:91], v20, v23 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[66:69], v18 offset:35840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[144:149], v[132:137], a[92:95], v20, v23 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[70:71], v7 offset:49664
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[150:155], v[132:137], a[124:127], v20, v23 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s13 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[150:155], v[126:131], a[120:123], v20, v23 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[72:75], v18 offset:36864
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[150:155], v[120:125], a[116:119], v20, v23 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[76:77], v7 offset:50176
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[150:155], v[114:119], a[112:115], v20, v23 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s47
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[150:155], v[108:113], a[108:111], v20, v22 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s3 offen lds
	ds_read_b128 v[78:81], v18 offset:37888
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[150:155], v[102:107], a[104:107], v20, v22 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[82:83], v7 offset:50688
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[150:155], v[96:101], a[100:103], v20, v22 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[150:155], v[24:29], a[96:99], v20, v22 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[90:93], v6 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[84:89], v[24:29], a[128:131], v21, v22 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[94:95], v4 offset:23040
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[84:89], v[96:101], a[132:135], v21, v22 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s3 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[84:89], v[102:107], a[136:139], v21, v22 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[138:141], v6 offset:8192
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[84:89], v[108:113], a[140:143], v21, v22 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[142:143], v4 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[84:89], v[114:119], a[144:147], v21, v23 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[84:89], v[120:125], a[148:151], v21, v23 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s3 offen offset:2048 lds
	ds_read_b128 v[144:147], v6 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[84:89], v[126:131], a[152:155], v21, v23 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[148:149], v4 offset:24064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[84:89], v[132:137], a[156:159], v21, v23 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[156:161], v[132:137], a[188:191], v21, v23 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[84:87], v6 offset:10240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[156:161], v[126:131], a[184:187], v21, v23 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[88:89], v4 offset:24576
	buffer_load_dwordx4 v1, s[24:27], s3 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[156:161], v[120:125], a[180:183], v21, v23 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[156:161], v[114:119], a[176:179], v21, v23 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[150:153], v6 offset:11264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[156:161], v[108:113], a[172:175], v21, v22 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[154:155], v4 offset:25088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[156:161], v[102:107], a[168:171], v21, v22 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s6
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[156:161], v[96:101], a[164:167], v21, v22 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s13 offen lds
	ds_read_b128 v[168:171], v6 offset:12288
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[156:161], v[24:29], a[160:163], v21, v22 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[172:173], v4 offset:25600
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[180:185], v[24:29], a[192:195], v21, v22 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[180:185], v[96:101], a[196:199], v21, v22 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[156:159], v6 offset:13312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[180:185], v[102:107], a[200:203], v21, v22 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[160:161], v4 offset:26112
	buffer_load_dwordx4 v1, s[24:27], s13 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[180:185], v[108:113], a[204:207], v21, v22 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[180:185], v[114:119], a[208:211], v21, v23 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[180:185], v[120:125], a[212:215], v21, v23 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[180:185], v[126:131], a[216:219], v21, v23 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s1, s7, s1
	s_mov_b32 m0, s43
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[180:185], v[132:137], a[220:223], v21, v23 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[28:31], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[162:167], v[132:137], a[252:255], v21, v23 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[162:167], v[126:131], a[248:251], v21, v23 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[162:167], v[120:125], a[244:247], v21, v23 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s1, s7, s2
	s_mov_b32 m0, s36
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[162:167], v[114:119], a[240:243], v21, v23 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[32:35], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[162:167], v[108:113], a[236:239], v21, v22 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[162:167], v[102:107], a[232:235], v21, v22 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[162:167], v[96:101], a[228:231], v21, v22 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[162:167], v[24:29], a[224:227], v21, v22 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s1, s9, 0x5a0
	s_lshl_b32 s1, s1, 9
	s_add_i32 s2, s12, 0x2d00
	s_lshl_b32 s2, s2, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[30:35], v[36:41], a[0:3], v8, v10 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[20:21], v5 offset:2048
	ds_read_b64 v[22:23], v3 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[30:35], v[42:47], a[4:7], v8, v10 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_i32 s3, s17, 0x10e000
	s_mov_b32 m0, s16
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[30:35], v[48:53], a[8:11], v8, v10 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s3 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[30:35], v[54:59], a[12:15], v8, v10 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[30:35], v[60:65], a[16:19], v8, v11 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[24:27], v6 offset:55296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[30:35], v[66:71], a[20:23], v8, v11 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[28:29], v19 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[30:35], v[72:77], a[24:27], v8, v11 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s3 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[30:35], v[78:83], a[28:31], v8, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[30:33], v12 offset:14336
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[90:95], v[78:83], a[60:63], v8, v11 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[34:35], v13 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[90:95], v[72:77], a[56:59], v8, v11 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[90:95], v[66:71], a[52:55], v8, v11 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s3 offen offset:2048 lds
	ds_read_b128 v[96:99], v12 offset:15360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[90:95], v[60:65], a[48:51], v8, v11 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[100:101], v13 offset:31232
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[90:95], v[54:59], a[44:47], v8, v10 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[90:95], v[48:53], a[40:43], v8, v10 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[102:105], v12 offset:16384
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[90:95], v[42:47], a[36:39], v8, v10 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[106:107], v13 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[90:95], v[36:41], a[32:35], v8, v10 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s3 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[138:143], v[36:41], a[64:67], v8, v10 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[90:93], v12 offset:17408
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[138:143], v[42:47], a[68:71], v8, v10 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[94:95], v13 offset:32256
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[138:143], v[48:53], a[72:75], v8, v10 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s13, s17, 0x10f000
	s_mov_b32 m0, s8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[138:143], v[54:59], a[76:79], v8, v10 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s13 offen lds
	ds_read_b128 v[108:111], v12 offset:18432
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[138:143], v[60:65], a[80:83], v8, v11 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[112:113], v13 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[138:143], v[66:71], a[84:87], v8, v11 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[138:143], v[72:77], a[88:91], v8, v11 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[114:117], v12 offset:19456
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[138:143], v[78:83], a[92:95], v8, v11 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[118:119], v13 offset:33280
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[144:149], v[78:83], a[124:127], v8, v11 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s13 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[144:149], v[72:77], a[120:123], v8, v11 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[120:123], v12 offset:20480
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[144:149], v[66:71], a[116:119], v8, v11 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[124:125], v13 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[144:149], v[60:65], a[112:115], v8, v11 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s18
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[144:149], v[54:59], a[108:111], v8, v10 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s3 offen lds
	ds_read_b128 v[126:129], v12 offset:21504
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[144:149], v[48:53], a[104:107], v8, v10 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[130:131], v13 offset:34304
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[144:149], v[42:47], a[100:103], v8, v10 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[144:149], v[36:41], a[96:99], v8, v10 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[132:135], v6 offset:56320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[84:89], v[36:41], a[128:131], v9, v10 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[136:137], v19 offset:6656
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[84:89], v[42:47], a[132:135], v9, v10 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s3 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[84:89], v[48:53], a[136:139], v9, v10 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[138:141], v6 offset:57344
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[84:89], v[54:59], a[140:143], v9, v10 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[142:143], v19 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[84:89], v[60:65], a[144:147], v9, v11 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[84:89], v[66:71], a[148:151], v9, v11 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s3 offen offset:2048 lds
	ds_read_b128 v[144:147], v6 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[84:89], v[72:77], a[152:155], v9, v11 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[148:149], v19 offset:7680
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[84:89], v[78:83], a[156:159], v9, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[150:155], v[78:83], a[188:191], v9, v11 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[84:87], v6 offset:59392
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[150:155], v[72:77], a[184:187], v9, v11 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[88:89], v19 offset:8192
	buffer_load_dwordx4 v1, s[24:27], s3 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[150:155], v[66:71], a[180:183], v9, v11 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[150:155], v[60:65], a[176:179], v9, v11 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[162:165], v6 offset:60416
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[150:155], v[54:59], a[172:175], v9, v10 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[166:167], v19 offset:8704
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[150:155], v[48:53], a[168:171], v9, v10 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s19
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[150:155], v[42:47], a[164:167], v9, v10 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s13 offen lds
	ds_read_b128 v[174:177], v6 offset:61440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[150:155], v[36:41], a[160:163], v9, v10 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[178:179], v19 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[168:173], v[36:41], a[192:195], v9, v10 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[168:173], v[42:47], a[196:199], v9, v10 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[150:153], v6 offset:62464
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[168:173], v[48:53], a[200:203], v9, v10 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[154:155], v19 offset:9728
	buffer_load_dwordx4 v1, s[24:27], s13 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[168:173], v[54:59], a[204:207], v9, v10 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[168:173], v[60:65], a[208:211], v9, v11 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[168:173], v[66:71], a[212:215], v9, v11 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[168:173], v[72:77], a[216:219], v9, v11 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s1, s7, s1
	s_mov_b32 m0, s7
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[168:173], v[78:83], a[220:223], v9, v11 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[28:31], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[156:161], v[78:83], a[252:255], v9, v11 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[156:161], v[72:77], a[248:251], v9, v11 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[156:161], v[66:71], a[244:247], v9, v11 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s1, s7, s2
	s_mov_b32 m0, s14
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[156:161], v[60:65], a[240:243], v9, v11 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[32:35], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[156:161], v[54:59], a[236:239], v9, v10 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[156:161], v[48:53], a[232:235], v9, v10 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[156:161], v[42:47], a[228:231], v9, v10 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[156:161], v[36:41], a[224:227], v9, v10 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s1, s9, 0x5c0
	s_lshl_b32 s1, s1, 9
	s_add_i32 s2, s12, 0x2e00
	s_lshl_b32 s2, s2, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[24:29], v[30:35], a[0:3], v20, v22 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[8:9], v5 offset:4096
	ds_read_b64 v[10:11], v3 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[24:29], v[96:101], a[4:7], v20, v22 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_i32 s3, s17, 0x114000
	s_mov_b32 m0, s38
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[24:29], v[102:107], a[8:11], v20, v22 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s3 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[24:29], v[90:95], a[12:15], v20, v22 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[24:29], v[108:113], a[16:19], v20, v23 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[36:39], v14 offset:38912
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[24:29], v[114:119], a[20:23], v20, v23 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[40:41], v19 offset:55296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[24:29], v[120:125], a[24:27], v20, v23 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s3 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[24:29], v[126:131], a[28:31], v20, v23 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[24:27], v12 offset:63488
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[132:137], v[126:131], a[60:63], v20, v23 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[28:29], v15 offset:14336
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[132:137], v[120:125], a[56:59], v20, v23 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[132:137], v[114:119], a[52:55], v20, v23 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s3 offen offset:2048 lds
	ds_read_b128 v[42:45], v12 offset:64512
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[132:137], v[108:113], a[48:51], v20, v23 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[46:47], v15 offset:14848
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[132:137], v[90:95], a[44:47], v20, v22 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[132:137], v[102:107], a[40:43], v20, v22 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[48:51], v16
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[132:137], v[96:101], a[36:39], v20, v22 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[52:53], v15 offset:15360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[132:137], v[30:35], a[32:35], v20, v22 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s3 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[138:143], v[30:35], a[64:67], v20, v22 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[54:57], v16 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[138:143], v[96:101], a[68:71], v20, v22 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[58:59], v15 offset:15872
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[138:143], v[102:107], a[72:75], v20, v22 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s8, s17, 0x115000
	s_mov_b32 m0, s39
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[138:143], v[90:95], a[76:79], v20, v22 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s8 offen lds
	ds_read_b128 v[60:63], v16 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[138:143], v[108:113], a[80:83], v20, v23 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[64:65], v15 offset:16384
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[138:143], v[114:119], a[84:87], v20, v23 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[138:143], v[120:125], a[88:91], v20, v23 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[66:69], v16 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[138:143], v[126:131], a[92:95], v20, v23 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[70:71], v15 offset:16896
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[144:149], v[126:131], a[124:127], v20, v23 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s8 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[144:149], v[120:125], a[120:123], v20, v23 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[72:75], v16 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[144:149], v[114:119], a[116:119], v20, v23 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[76:77], v15 offset:17408
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[144:149], v[108:113], a[112:115], v20, v23 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s41
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[144:149], v[90:95], a[108:111], v20, v22 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s3 offen lds
	ds_read_b128 v[78:81], v16 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[144:149], v[102:107], a[104:107], v20, v22 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[82:83], v15 offset:17920
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[144:149], v[96:101], a[100:103], v20, v22 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[144:149], v[30:35], a[96:99], v20, v22 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[132:135], v14 offset:39936
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[84:89], v[30:35], a[128:131], v21, v22 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[136:137], v19 offset:55808
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[84:89], v[96:101], a[132:135], v21, v22 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s3 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[84:89], v[102:107], a[136:139], v21, v22 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[138:141], v14 offset:40960
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[84:89], v[90:95], a[140:143], v21, v22 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[142:143], v19 offset:56320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[84:89], v[108:113], a[144:147], v21, v23 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[84:89], v[114:119], a[148:151], v21, v23 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s3 offen offset:2048 lds
	ds_read_b128 v[144:147], v14 offset:41984
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[84:89], v[120:125], a[152:155], v21, v23 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[148:149], v19 offset:56832
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[84:89], v[126:131], a[156:159], v21, v23 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[162:167], v[126:131], a[188:191], v21, v23 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[84:87], v14 offset:43008
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[162:167], v[120:125], a[184:187], v21, v23 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[88:89], v19 offset:57344
	buffer_load_dwordx4 v1, s[24:27], s3 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[162:167], v[114:119], a[180:183], v21, v23 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[162:167], v[108:113], a[176:179], v21, v23 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[156:159], v14 offset:44032
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[162:167], v[90:95], a[172:175], v21, v22 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[160:161], v19 offset:57856
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[162:167], v[102:107], a[168:171], v21, v22 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s37
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[162:167], v[96:101], a[164:167], v21, v22 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s8 offen lds
	ds_read_b128 v[168:171], v14 offset:45056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[162:167], v[30:35], a[160:163], v21, v22 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[172:173], v19 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[174:179], v[30:35], a[192:195], v21, v22 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[174:179], v[96:101], a[196:199], v21, v22 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[162:165], v14 offset:46080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[174:179], v[102:107], a[200:203], v21, v22 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[166:167], v19 offset:58880
	buffer_load_dwordx4 v1, s[24:27], s8 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[174:179], v[90:95], a[204:207], v21, v22 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[174:179], v[108:113], a[208:211], v21, v23 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[174:179], v[114:119], a[212:215], v21, v23 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[174:179], v[120:125], a[216:219], v21, v23 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s1, s7, s1
	s_mov_b32 m0, s40
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[174:179], v[126:131], a[220:223], v21, v23 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[28:31], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[150:155], v[126:131], a[252:255], v21, v23 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[150:155], v[120:125], a[248:251], v21, v23 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[150:155], v[114:119], a[244:247], v21, v23 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s1, s7, s2
	s_mov_b32 m0, s15
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[150:155], v[108:113], a[240:243], v21, v23 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[32:35], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[150:155], v[90:95], a[236:239], v21, v22 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[150:155], v[102:107], a[232:235], v21, v22 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[150:155], v[96:101], a[228:231], v21, v22 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[150:155], v[30:35], a[224:227], v21, v22 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s1, s9, 0x5e0
	s_lshl_b32 s1, s1, 9
	s_add_i32 s2, s12, 0x2f00
	s_lshl_b32 s2, s2, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:41], v[24:29], a[0:3], v8, v10 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[20:21], v5
	ds_read_b64 v[22:23], v3 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[36:41], v[42:47], a[4:7], v8, v10 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_i32 s3, s17, 0x11a000
	s_mov_b32 m0, s44
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[36:41], v[48:53], a[8:11], v8, v10 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s3 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[36:41], v[54:59], a[12:15], v8, v10 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[36:41], v[60:65], a[16:19], v8, v11 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[30:33], v6 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[36:41], v[66:71], a[20:23], v8, v11 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[34:35], v4 offset:22528
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[36:41], v[72:77], a[24:27], v8, v11 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s3 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[36:41], v[78:83], a[28:31], v8, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[36:39], v18 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[132:137], v[78:83], a[60:63], v8, v11 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[40:41], v7 offset:47104
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[132:137], v[72:77], a[56:59], v8, v11 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[132:137], v[66:71], a[52:55], v8, v11 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s3 offen offset:2048 lds
	ds_read_b128 v[90:93], v18 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[132:137], v[60:65], a[48:51], v8, v11 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[94:95], v7 offset:47616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[132:137], v[54:59], a[44:47], v8, v10 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[132:137], v[48:53], a[40:43], v8, v10 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[96:99], v18 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[132:137], v[42:47], a[36:39], v8, v10 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[100:101], v7 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[132:137], v[24:29], a[32:35], v8, v10 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s3 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[138:143], v[24:29], a[64:67], v8, v10 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[102:105], v18 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[138:143], v[42:47], a[68:71], v8, v10 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[106:107], v7 offset:48640
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[138:143], v[48:53], a[72:75], v8, v10 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s8, s17, 0x11b000
	s_mov_b32 m0, s45
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[138:143], v[54:59], a[76:79], v8, v10 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s8 offen lds
	ds_read_b128 v[108:111], v18 offset:34816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[138:143], v[60:65], a[80:83], v8, v11 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[112:113], v7 offset:49152
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[138:143], v[66:71], a[84:87], v8, v11 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[138:143], v[72:77], a[88:91], v8, v11 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[114:117], v18 offset:35840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[138:143], v[78:83], a[92:95], v8, v11 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[118:119], v7 offset:49664
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[144:149], v[78:83], a[124:127], v8, v11 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s8 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[144:149], v[72:77], a[120:123], v8, v11 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[120:123], v18 offset:36864
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[144:149], v[66:71], a[116:119], v8, v11 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[124:125], v7 offset:50176
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[144:149], v[60:65], a[112:115], v8, v11 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_mov_b32 m0, s47
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[144:149], v[54:59], a[108:111], v8, v10 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s3 offen lds
	ds_read_b128 v[126:129], v18 offset:37888
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[144:149], v[48:53], a[104:107], v8, v10 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[130:131], v7 offset:50688
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[144:149], v[42:47], a[100:103], v8, v10 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[144:149], v[24:29], a[96:99], v8, v10 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[132:135], v6 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[84:89], v[24:29], a[128:131], v9, v10 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[136:137], v4 offset:23040
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[84:89], v[42:47], a[132:135], v9, v10 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s3 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[84:89], v[48:53], a[136:139], v9, v10 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[138:141], v6 offset:8192
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[84:89], v[54:59], a[140:143], v9, v10 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[142:143], v4 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[84:89], v[60:65], a[144:147], v9, v11 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[84:89], v[66:71], a[148:151], v9, v11 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s3 offen offset:2048 lds
	ds_read_b128 v[144:147], v6 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[84:89], v[72:77], a[152:155], v9, v11 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[148:149], v4 offset:24064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[84:89], v[78:83], a[156:159], v9, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[156:161], v[78:83], a[188:191], v9, v11 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[84:87], v6 offset:10240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[156:161], v[72:77], a[184:187], v9, v11 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[88:89], v4 offset:24576
	buffer_load_dwordx4 v1, s[24:27], s3 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[156:161], v[66:71], a[180:183], v9, v11 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[156:161], v[60:65], a[176:179], v9, v11 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[150:153], v6 offset:11264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[156:161], v[54:59], a[172:175], v9, v10 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[154:155], v4 offset:25088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[156:161], v[48:53], a[168:171], v9, v10 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s6
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[156:161], v[42:47], a[164:167], v9, v10 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s8 offen lds
	ds_read_b128 v[174:177], v6 offset:12288
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[156:161], v[24:29], a[160:163], v9, v10 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[178:179], v4 offset:25600
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[168:173], v[24:29], a[192:195], v9, v10 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[168:173], v[42:47], a[196:199], v9, v10 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[156:159], v6 offset:13312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[168:173], v[48:53], a[200:203], v9, v10 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[160:161], v4 offset:26112
	buffer_load_dwordx4 v1, s[24:27], s8 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[168:173], v[54:59], a[204:207], v9, v10 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[168:173], v[60:65], a[208:211], v9, v11 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[168:173], v[66:71], a[212:215], v9, v11 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[168:173], v[72:77], a[216:219], v9, v11 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s1, s7, s1
	s_mov_b32 m0, s43
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[168:173], v[78:83], a[220:223], v9, v11 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[28:31], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[162:167], v[78:83], a[252:255], v9, v11 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[162:167], v[72:77], a[248:251], v9, v11 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[162:167], v[66:71], a[244:247], v9, v11 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s1, s7, s2
	s_mov_b32 m0, s36
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[162:167], v[60:65], a[240:243], v9, v11 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[32:35], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[162:167], v[54:59], a[236:239], v9, v10 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[162:167], v[48:53], a[232:235], v9, v10 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[162:167], v[42:47], a[228:231], v9, v10 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[162:167], v[24:29], a[224:227], v9, v10 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[30:35], v[36:41], a[0:3], v20, v22 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[8:9], v5 offset:2048
	ds_read_b64 v[10:11], v3 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[30:35], v[90:95], a[4:7], v20, v22 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[30:35], v[96:101], a[8:11], v20, v22 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[30:35], v[102:107], a[12:15], v20, v22 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[30:35], v[108:113], a[16:19], v20, v23 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[24:27], v6 offset:55296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[30:35], v[114:119], a[20:23], v20, v23 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[28:29], v19 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[30:35], v[120:125], a[24:27], v20, v23 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[30:35], v[126:131], a[28:31], v20, v23 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[30:33], v12 offset:14336
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[132:137], v[126:131], a[60:63], v20, v23 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[34:35], v13 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[132:137], v[120:125], a[56:59], v20, v23 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[132:137], v[114:119], a[52:55], v20, v23 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[42:45], v12 offset:15360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[132:137], v[108:113], a[48:51], v20, v23 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[46:47], v13 offset:31232
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[132:137], v[102:107], a[44:47], v20, v22 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[132:137], v[96:101], a[40:43], v20, v22 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[48:51], v12 offset:16384
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[132:137], v[90:95], a[36:39], v20, v22 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[52:53], v13 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[132:137], v[36:41], a[32:35], v20, v22 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[138:143], v[36:41], a[64:67], v20, v22 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[54:57], v12 offset:17408
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[138:143], v[90:95], a[68:71], v20, v22 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[58:59], v13 offset:32256
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[138:143], v[96:101], a[72:75], v20, v22 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[138:143], v[102:107], a[76:79], v20, v22 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[60:63], v12 offset:18432
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[138:143], v[108:113], a[80:83], v20, v23 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[64:65], v13 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[138:143], v[114:119], a[84:87], v20, v23 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[138:143], v[120:125], a[88:91], v20, v23 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[66:69], v12 offset:19456
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[138:143], v[126:131], a[92:95], v20, v23 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[70:71], v13 offset:33280
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[144:149], v[126:131], a[124:127], v20, v23 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[144:149], v[120:125], a[120:123], v20, v23 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[72:75], v12 offset:20480
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[144:149], v[114:119], a[116:119], v20, v23 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[76:77], v13 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[144:149], v[108:113], a[112:115], v20, v23 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[144:149], v[102:107], a[108:111], v20, v22 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[78:81], v12 offset:21504
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[144:149], v[96:101], a[104:107], v20, v22 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[82:83], v13 offset:34304
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[144:149], v[90:95], a[100:103], v20, v22 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[144:149], v[36:41], a[96:99], v20, v22 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[132:135], v6 offset:56320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[84:89], v[36:41], a[128:131], v21, v22 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[136:137], v19 offset:6656
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[84:89], v[90:95], a[132:135], v21, v22 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[84:89], v[96:101], a[136:139], v21, v22 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[138:141], v6 offset:57344
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[84:89], v[102:107], a[140:143], v21, v22 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[142:143], v19 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[84:89], v[108:113], a[144:147], v21, v23 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[84:89], v[114:119], a[148:151], v21, v23 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[144:147], v6 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[84:89], v[120:125], a[152:155], v21, v23 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[148:149], v19 offset:7680
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[84:89], v[126:131], a[156:159], v21, v23 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[150:155], v[126:131], a[188:191], v21, v23 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[84:87], v6 offset:59392
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[150:155], v[120:125], a[184:187], v21, v23 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[88:89], v19 offset:8192
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[150:155], v[114:119], a[180:183], v21, v23 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[150:155], v[108:113], a[176:179], v21, v23 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[162:165], v6 offset:60416
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[150:155], v[102:107], a[172:175], v21, v22 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[166:167], v19 offset:8704
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[150:155], v[96:101], a[168:171], v21, v22 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[150:155], v[90:95], a[164:167], v21, v22 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[168:171], v6 offset:61440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[150:155], v[36:41], a[160:163], v21, v22 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[172:173], v19 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[174:179], v[36:41], a[192:195], v21, v22 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[174:179], v[90:95], a[196:199], v21, v22 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[150:153], v6 offset:62464
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[174:179], v[96:101], a[200:203], v21, v22 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[154:155], v19 offset:9728
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[174:179], v[102:107], a[204:207], v21, v22 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[174:179], v[108:113], a[208:211], v21, v23 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[174:179], v[114:119], a[212:215], v21, v23 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[174:179], v[120:125], a[216:219], v21, v23 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[174:179], v[126:131], a[220:223], v21, v23 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[156:161], v[126:131], a[252:255], v21, v23 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[156:161], v[120:125], a[248:251], v21, v23 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[156:161], v[114:119], a[244:247], v21, v23 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[156:161], v[108:113], a[240:243], v21, v23 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[156:161], v[102:107], a[236:239], v21, v22 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[156:161], v[96:101], a[232:235], v21, v22 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[156:161], v[90:95], a[228:231], v21, v22 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[156:161], v[36:41], a[224:227], v21, v22 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[4:5], v5 offset:4096
	ds_read_b64 v[2:3], v3 offset:5120
	ds_read_b128 v[36:39], v14 offset:38912
	ds_read_b64 v[40:41], v19 offset:55296
	ds_read_b128 v[90:93], v12 offset:63488
	ds_read_b64 v[94:95], v15 offset:14336
	ds_read_b128 v[96:99], v12 offset:64512
	ds_read_b64 v[100:101], v15 offset:14848
	ds_read_b128 v[102:105], v16
	ds_read_b64 v[106:107], v15 offset:15360
	ds_read_b128 v[108:111], v16 offset:1024
	ds_read_b64 v[112:113], v15 offset:15872
	ds_read_b128 v[114:117], v16 offset:2048
	ds_read_b64 v[118:119], v15 offset:16384
	ds_read_b128 v[120:123], v16 offset:3072
	ds_read_b64 v[124:125], v15 offset:16896
	ds_read_b128 v[126:129], v16 offset:4096
	ds_read_b64 v[130:131], v15 offset:17408
	ds_read_b128 v[156:159], v16 offset:5120
	ds_read_b64 v[160:161], v15 offset:17920
	ds_read_b128 v[174:177], v14 offset:39936
	ds_read_b64 v[178:179], v19 offset:55808
	ds_read_b128 v[180:183], v14 offset:40960
	ds_read_b64 v[184:185], v19 offset:56320
	ds_read_b128 v[186:189], v14 offset:41984
	ds_read_b64 v[190:191], v19 offset:56832
	ds_read_b128 v[192:195], v14 offset:43008
	ds_read_b64 v[196:197], v19 offset:57344
	ds_read_b128 v[198:201], v14 offset:44032
	ds_read_b64 v[202:203], v19 offset:57856
	ds_read_b128 v[204:207], v14 offset:45056
	ds_read_b64 v[208:209], v19 offset:58368
	ds_read_b128 v[12:15], v14 offset:46080
	ds_read_b64 v[16:17], v19 offset:58880
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[24:29], v[30:35], a[0:3], v8, v10 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[24:29], v[42:47], a[4:7], v8, v10 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[24:29], v[48:53], a[8:11], v8, v10 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[24:29], v[54:59], a[12:15], v8, v10 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[24:29], v[60:65], a[16:19], v8, v11 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[24:29], v[66:71], a[20:23], v8, v11 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[24:29], v[72:77], a[24:27], v8, v11 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[24:29], v[78:83], a[28:31], v8, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[36:41], v[156:161], a[28:31], v4, v3 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[36:41], v[126:131], a[24:27], v4, v3 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[36:41], v[120:125], a[20:23], v4, v3 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[36:41], v[114:119], a[16:19], v4, v3 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[36:41], v[108:113], a[12:15], v4, v2 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[36:41], v[102:107], a[8:11], v4, v2 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[36:41], v[96:101], a[4:7], v4, v2 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:41], v[90:95], a[0:3], v4, v2 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[132:137], v[78:83], a[60:63], v8, v11 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_and_b32_e32 v1, 15, v0
	v_and_b32_e32 v6, 1, v1
	v_sub_u32_e32 v7, v1, v6
	v_ashrrev_i32_e32 v18, 31, v7
	v_cmp_eq_i32_e64 s[2:3], v6, 1
	v_lshrrev_b32_e32 v18, 31, v18
	v_lshrrev_b32_e32 v0, 4, v0
	v_lshlrev_b32_e32 v6, 1, v6
	v_add_u32_e32 v7, v7, v18
	v_lshl_add_u32 v6, v0, 2, v6
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[132:137], v[72:77], a[56:59], v8, v11 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_ashrrev_i32_e32 v7, 1, v7
	v_accvgpr_read_b32 v18, a0
	v_accvgpr_read_b32 v19, a1
	v_cvt_pk_bf16_f32 v18, v18, v19
	v_accvgpr_read_b32 v19, a2
	v_accvgpr_read_b32 v20, a3
	v_cvt_pk_bf16_f32 v19, v19, v20
	s_mul_i32 s0, s0, 0x440
	s_mov_b32 s1, 0x44
	v_mad_u32_u24 v6, s1, v6, v7
	v_cndmask_b32_e64 v7, v19, v18, s[2:3]
	v_mov_b32_e32 v20, 0x1000504
	v_mov_b32_e32 v21, 0x5040100
	v_cndmask_b32_e64 v20, v21, v20, s[2:3]
	v_mov_b32_e32 v21, 0x3020706
	v_mov_b32_e32 v22, 0x7060302
	v_cndmask_b32_e64 v21, v22, v21, s[2:3]
	v_cndmask_b32_e64 v18, v18, v19, s[2:3]
	v_mov_b32_dpp v7, v7 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_lshl_u32 v6, s0, v6, 2
	v_lshlrev_b32_e32 v19, 2, v1
	v_lshlrev_b32_e32 v1, 4, v1
	v_perm_b32 v22, v7, v18, v20
	v_perm_b32 v7, v7, v18, v21
	v_add_u32_e32 v6, 0xd800, v6
	s_mov_b32 s8, s10
	s_and_b32 s9, s11, 0xffff
	s_mov_b32 s10, 0x7ffffff0
	s_mov_b32 s11, 0x31014000
	s_mov_b32 s1, 0x44
	v_mad_u32_u24 v18, s1, v0, v19
	v_lshl_add_u32 v0, v0, 16, v1
	ds_write2_b32 v6, v22, v7 offset1:68
	v_accvgpr_read_b32 v1, a4
	v_accvgpr_read_b32 v7, a5
	v_cvt_pk_bf16_f32 v22, v1, v7
	v_accvgpr_read_b32 v1, a6
	v_accvgpr_read_b32 v7, a7
	v_cvt_pk_bf16_f32 v23, v1, v7
	v_cndmask_b32_e64 v1, v23, v22, s[2:3]
	v_cndmask_b32_e64 v7, v22, v23, s[2:3]
	s_nop 0
	v_mov_b32_dpp v1, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v19, v1, v7, v20
	v_perm_b32 v1, v1, v7, v21
	ds_write2_b32 v6, v19, v1 offset0:8 offset1:76
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[132:137], v[66:71], a[52:55], v8, v11 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v1, a8
	v_accvgpr_read_b32 v7, a9
	v_cvt_pk_bf16_f32 v22, v1, v7
	v_accvgpr_read_b32 v1, a10
	v_accvgpr_read_b32 v7, a11
	v_cvt_pk_bf16_f32 v23, v1, v7
	v_cndmask_b32_e64 v1, v23, v22, s[2:3]
	v_cndmask_b32_e64 v7, v22, v23, s[2:3]
	s_nop 0
	v_mov_b32_dpp v1, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v19, v1, v7, v20
	v_perm_b32 v1, v1, v7, v21
	ds_write2_b32 v6, v19, v1 offset0:16 offset1:84
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[132:137], v[60:65], a[48:51], v8, v11 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v1, a12
	v_accvgpr_read_b32 v7, a13
	v_cvt_pk_bf16_f32 v22, v1, v7
	v_accvgpr_read_b32 v1, a14
	v_accvgpr_read_b32 v7, a15
	v_cvt_pk_bf16_f32 v23, v1, v7
	v_cndmask_b32_e64 v1, v23, v22, s[2:3]
	v_cndmask_b32_e64 v7, v22, v23, s[2:3]
	s_nop 0
	v_mov_b32_dpp v1, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v19, v1, v7, v20
	v_perm_b32 v1, v1, v7, v21
	ds_write2_b32 v6, v19, v1 offset0:24 offset1:92
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[132:137], v[54:59], a[44:47], v8, v10 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v1, a16
	v_accvgpr_read_b32 v7, a17
	v_cvt_pk_bf16_f32 v22, v1, v7
	v_accvgpr_read_b32 v1, a18
	v_accvgpr_read_b32 v7, a19
	v_cvt_pk_bf16_f32 v23, v1, v7
	v_cndmask_b32_e64 v1, v23, v22, s[2:3]
	v_cndmask_b32_e64 v7, v22, v23, s[2:3]
	s_nop 0
	v_mov_b32_dpp v1, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v19, v1, v7, v20
	v_perm_b32 v1, v1, v7, v21
	ds_write2_b32 v6, v19, v1 offset0:32 offset1:100
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[132:137], v[48:53], a[40:43], v8, v10 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v1, a20
	v_accvgpr_read_b32 v7, a21
	v_cvt_pk_bf16_f32 v22, v1, v7
	v_accvgpr_read_b32 v1, a22
	v_accvgpr_read_b32 v7, a23
	v_cvt_pk_bf16_f32 v23, v1, v7
	v_cndmask_b32_e64 v1, v23, v22, s[2:3]
	v_cndmask_b32_e64 v7, v22, v23, s[2:3]
	s_nop 0
	v_mov_b32_dpp v1, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v19, v1, v7, v20
	v_perm_b32 v1, v1, v7, v21
	ds_write2_b32 v6, v19, v1 offset0:40 offset1:108
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[132:137], v[42:47], a[36:39], v8, v10 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v1, a24
	v_accvgpr_read_b32 v7, a25
	v_cvt_pk_bf16_f32 v22, v1, v7
	v_accvgpr_read_b32 v1, a26
	v_accvgpr_read_b32 v7, a27
	v_cvt_pk_bf16_f32 v23, v1, v7
	v_cndmask_b32_e64 v1, v23, v22, s[2:3]
	v_cndmask_b32_e64 v7, v22, v23, s[2:3]
	s_nop 0
	v_mov_b32_dpp v1, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v19, v1, v7, v20
	v_perm_b32 v1, v1, v7, v21
	ds_write2_b32 v6, v19, v1 offset0:48 offset1:116
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[132:137], v[30:35], a[32:35], v8, v10 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v1, a28
	v_accvgpr_read_b32 v7, a29
	v_cvt_pk_bf16_f32 v22, v1, v7
	v_accvgpr_read_b32 v1, a30
	v_accvgpr_read_b32 v7, a31
	v_cvt_pk_bf16_f32 v23, v1, v7
	v_cndmask_b32_e64 v1, v23, v22, s[2:3]
	v_cndmask_b32_e64 v7, v22, v23, s[2:3]
	s_nop 0
	v_mov_b32_dpp v1, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v19, v1, v7, v20
	v_perm_b32 v1, v1, v7, v21
	ds_write2_b32 v6, v19, v1 offset0:56 offset1:124
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[174:179], v[90:95], a[32:35], v4, v2 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[174:179], v[96:101], a[36:39], v4, v2 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_add_lshl_u32 v1, s0, v18, 2
	ds_read_b128 v[22:25], v1 offset:55296
	ds_read_b128 v[26:29], v1 offset:56384
	ds_read_b128 v[36:39], v1 offset:57472
	ds_read_b128 v[132:135], v1 offset:58560
	s_lshl_b32 s0, s4, 16
	s_lshl_b32 s1, s5, 1
	s_add_i32 s0, s0, s1
	v_add_u32_e32 v7, 0x40000, v0
	v_add_u32_e32 v18, 0x80000, v0
	v_add_u32_e32 v19, 0xc0000, v0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[174:179], v[102:107], a[40:43], v4, v2 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(3)
	buffer_store_dwordx4 v[22:25], v0, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(2)
	buffer_store_dwordx4 v[26:29], v7, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[36:39], v18, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[132:135], v19, s[8:11], s0 offen nt
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[174:179], v[108:113], a[44:47], v4, v2 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[174:179], v[114:119], a[48:51], v4, v3 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[174:179], v[120:125], a[52:55], v4, v3 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[174:179], v[126:131], a[56:59], v4, v3 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[174:179], v[156:161], a[60:63], v4, v3 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[138:143], v[30:35], a[64:67], v8, v10 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v22, a32
	v_accvgpr_read_b32 v23, a33
	v_cvt_pk_bf16_f32 v22, v22, v23
	v_accvgpr_read_b32 v23, a34
	v_accvgpr_read_b32 v24, a35
	v_cvt_pk_bf16_f32 v23, v23, v24
	v_cndmask_b32_e64 v24, v23, v22, s[2:3]
	v_cndmask_b32_e64 v22, v22, v23, s[2:3]
	s_nop 0
	v_mov_b32_dpp v23, v24 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v24, v23, v22, v20
	v_perm_b32 v22, v23, v22, v21
	ds_write2_b32 v6, v24, v22 offset1:68
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[138:143], v[42:47], a[68:71], v8, v10 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v22, a36
	v_accvgpr_read_b32 v23, a37
	v_cvt_pk_bf16_f32 v22, v22, v23
	v_accvgpr_read_b32 v23, a38
	v_accvgpr_read_b32 v24, a39
	v_cvt_pk_bf16_f32 v23, v23, v24
	v_cndmask_b32_e64 v24, v23, v22, s[2:3]
	v_cndmask_b32_e64 v22, v22, v23, s[2:3]
	s_nop 0
	v_mov_b32_dpp v23, v24 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v24, v23, v22, v20
	v_perm_b32 v22, v23, v22, v21
	ds_write2_b32 v6, v24, v22 offset0:8 offset1:76
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[138:143], v[48:53], a[72:75], v8, v10 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v22, a40
	v_accvgpr_read_b32 v23, a41
	v_cvt_pk_bf16_f32 v22, v22, v23
	v_accvgpr_read_b32 v23, a42
	v_accvgpr_read_b32 v24, a43
	v_cvt_pk_bf16_f32 v23, v23, v24
	v_cndmask_b32_e64 v24, v23, v22, s[2:3]
	v_cndmask_b32_e64 v22, v22, v23, s[2:3]
	s_nop 0
	v_mov_b32_dpp v23, v24 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v24, v23, v22, v20
	v_perm_b32 v22, v23, v22, v21
	ds_write2_b32 v6, v24, v22 offset0:16 offset1:84
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[138:143], v[54:59], a[76:79], v8, v10 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v22, a44
	v_accvgpr_read_b32 v23, a45
	v_cvt_pk_bf16_f32 v22, v22, v23
	v_accvgpr_read_b32 v23, a46
	v_accvgpr_read_b32 v24, a47
	v_cvt_pk_bf16_f32 v23, v23, v24
	v_cndmask_b32_e64 v24, v23, v22, s[2:3]
	v_cndmask_b32_e64 v22, v22, v23, s[2:3]
	s_nop 0
	v_mov_b32_dpp v23, v24 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v24, v23, v22, v20
	v_perm_b32 v22, v23, v22, v21
	ds_write2_b32 v6, v24, v22 offset0:24 offset1:92
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[138:143], v[60:65], a[80:83], v8, v11 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v22, a48
	v_accvgpr_read_b32 v23, a49
	v_cvt_pk_bf16_f32 v22, v22, v23
	v_accvgpr_read_b32 v23, a50
	v_accvgpr_read_b32 v24, a51
	v_cvt_pk_bf16_f32 v23, v23, v24
	v_cndmask_b32_e64 v24, v23, v22, s[2:3]
	v_cndmask_b32_e64 v22, v22, v23, s[2:3]
	s_nop 0
	v_mov_b32_dpp v23, v24 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v24, v23, v22, v20
	v_perm_b32 v22, v23, v22, v21
	ds_write2_b32 v6, v24, v22 offset0:32 offset1:100
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[138:143], v[66:71], a[84:87], v8, v11 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v22, a52
	v_accvgpr_read_b32 v23, a53
	v_cvt_pk_bf16_f32 v22, v22, v23
	v_accvgpr_read_b32 v23, a54
	v_accvgpr_read_b32 v24, a55
	v_cvt_pk_bf16_f32 v23, v23, v24
	v_cndmask_b32_e64 v24, v23, v22, s[2:3]
	v_cndmask_b32_e64 v22, v22, v23, s[2:3]
	s_nop 0
	v_mov_b32_dpp v23, v24 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v24, v23, v22, v20
	v_perm_b32 v22, v23, v22, v21
	ds_write2_b32 v6, v24, v22 offset0:40 offset1:108
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[138:143], v[72:77], a[88:91], v8, v11 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v22, a56
	v_accvgpr_read_b32 v23, a57
	v_cvt_pk_bf16_f32 v22, v22, v23
	v_accvgpr_read_b32 v23, a58
	v_accvgpr_read_b32 v24, a59
	v_cvt_pk_bf16_f32 v23, v23, v24
	v_cndmask_b32_e64 v24, v23, v22, s[2:3]
	v_cndmask_b32_e64 v22, v22, v23, s[2:3]
	s_nop 0
	v_mov_b32_dpp v23, v24 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v24, v23, v22, v20
	v_perm_b32 v22, v23, v22, v21
	ds_write2_b32 v6, v24, v22 offset0:48 offset1:116
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[138:143], v[78:83], a[92:95], v8, v11 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v22, a60
	v_accvgpr_read_b32 v23, a61
	v_cvt_pk_bf16_f32 v22, v22, v23
	v_accvgpr_read_b32 v23, a62
	v_accvgpr_read_b32 v24, a63
	v_cvt_pk_bf16_f32 v23, v23, v24
	v_cndmask_b32_e64 v24, v23, v22, s[2:3]
	v_cndmask_b32_e64 v22, v22, v23, s[2:3]
	s_nop 0
	v_mov_b32_dpp v23, v24 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v24, v23, v22, v20
	v_perm_b32 v22, v23, v22, v21
	ds_write2_b32 v6, v24, v22 offset0:56 offset1:124
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[180:185], v[156:161], a[92:95], v4, v3 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[180:185], v[126:131], a[88:91], v4, v3 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	ds_read_b128 v[22:25], v1 offset:55296
	ds_read_b128 v[26:29], v1 offset:56384
	ds_read_b128 v[36:39], v1 offset:57472
	ds_read_b128 v[132:135], v1 offset:58560
	s_add_i32 s0, s4, 16
	s_lshl_b32 s0, s0, 16
	s_add_i32 s0, s1, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[180:185], v[120:125], a[84:87], v4, v3 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(3)
	buffer_store_dwordx4 v[22:25], v0, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(2)
	buffer_store_dwordx4 v[26:29], v7, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[36:39], v18, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[132:135], v19, s[8:11], s0 offen nt
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[180:185], v[114:119], a[80:83], v4, v3 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[180:185], v[108:113], a[76:79], v4, v2 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[180:185], v[102:107], a[72:75], v4, v2 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[180:185], v[96:101], a[68:71], v4, v2 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[180:185], v[90:95], a[64:67], v4, v2 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[144:149], v[78:83], a[124:127], v8, v11 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_nop 7
	s_nop 2
	v_accvgpr_read_b32 v22, a64
	v_accvgpr_read_b32 v23, a65
	v_cvt_pk_bf16_f32 v22, v22, v23
	v_accvgpr_read_b32 v23, a66
	v_accvgpr_read_b32 v24, a67
	v_cvt_pk_bf16_f32 v23, v23, v24
	v_cndmask_b32_e64 v24, v23, v22, s[2:3]
	v_cndmask_b32_e64 v22, v22, v23, s[2:3]
	s_nop 0
	v_mov_b32_dpp v23, v24 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[144:149], v[72:77], a[120:123], v8, v11 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_perm_b32 v24, v23, v22, v20
	v_perm_b32 v22, v23, v22, v21
	ds_write2_b32 v6, v24, v22 offset1:68
	v_accvgpr_read_b32 v22, a68
	v_accvgpr_read_b32 v23, a69
	v_cvt_pk_bf16_f32 v22, v22, v23
	v_accvgpr_read_b32 v23, a70
	v_accvgpr_read_b32 v24, a71
	v_cvt_pk_bf16_f32 v23, v23, v24
	v_cndmask_b32_e64 v24, v23, v22, s[2:3]
	v_cndmask_b32_e64 v22, v22, v23, s[2:3]
	s_nop 0
	v_mov_b32_dpp v23, v24 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v24, v23, v22, v20
	v_perm_b32 v22, v23, v22, v21
	ds_write2_b32 v6, v24, v22 offset0:8 offset1:76
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[144:149], v[66:71], a[116:119], v8, v11 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v22, a72
	v_accvgpr_read_b32 v23, a73
	v_cvt_pk_bf16_f32 v22, v22, v23
	v_accvgpr_read_b32 v23, a74
	v_accvgpr_read_b32 v24, a75
	v_cvt_pk_bf16_f32 v23, v23, v24
	v_cndmask_b32_e64 v24, v23, v22, s[2:3]
	v_cndmask_b32_e64 v22, v22, v23, s[2:3]
	s_nop 0
	v_mov_b32_dpp v23, v24 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v24, v23, v22, v20
	v_perm_b32 v22, v23, v22, v21
	ds_write2_b32 v6, v24, v22 offset0:16 offset1:84
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[144:149], v[60:65], a[112:115], v8, v11 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v22, a76
	v_accvgpr_read_b32 v23, a77
	v_cvt_pk_bf16_f32 v22, v22, v23
	v_accvgpr_read_b32 v23, a78
	v_accvgpr_read_b32 v24, a79
	v_cvt_pk_bf16_f32 v23, v23, v24
	v_cndmask_b32_e64 v24, v23, v22, s[2:3]
	v_cndmask_b32_e64 v22, v22, v23, s[2:3]
	s_nop 0
	v_mov_b32_dpp v23, v24 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v24, v23, v22, v20
	v_perm_b32 v22, v23, v22, v21
	ds_write2_b32 v6, v24, v22 offset0:24 offset1:92
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[144:149], v[54:59], a[108:111], v8, v10 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v22, a80
	v_accvgpr_read_b32 v23, a81
	v_cvt_pk_bf16_f32 v22, v22, v23
	v_accvgpr_read_b32 v23, a82
	v_accvgpr_read_b32 v24, a83
	v_cvt_pk_bf16_f32 v23, v23, v24
	v_cndmask_b32_e64 v24, v23, v22, s[2:3]
	v_cndmask_b32_e64 v22, v22, v23, s[2:3]
	s_nop 0
	v_mov_b32_dpp v23, v24 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v24, v23, v22, v20
	v_perm_b32 v22, v23, v22, v21
	ds_write2_b32 v6, v24, v22 offset0:32 offset1:100
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[144:149], v[48:53], a[104:107], v8, v10 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v22, a84
	v_accvgpr_read_b32 v23, a85
	v_cvt_pk_bf16_f32 v22, v22, v23
	v_accvgpr_read_b32 v23, a86
	v_accvgpr_read_b32 v24, a87
	v_cvt_pk_bf16_f32 v23, v23, v24
	v_cndmask_b32_e64 v24, v23, v22, s[2:3]
	v_cndmask_b32_e64 v22, v22, v23, s[2:3]
	s_nop 0
	v_mov_b32_dpp v23, v24 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v24, v23, v22, v20
	v_perm_b32 v22, v23, v22, v21
	ds_write2_b32 v6, v24, v22 offset0:40 offset1:108
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[144:149], v[42:47], a[100:103], v8, v10 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v22, a88
	v_accvgpr_read_b32 v23, a89
	v_cvt_pk_bf16_f32 v22, v22, v23
	v_accvgpr_read_b32 v23, a90
	v_accvgpr_read_b32 v24, a91
	v_cvt_pk_bf16_f32 v23, v23, v24
	v_cndmask_b32_e64 v24, v23, v22, s[2:3]
	v_cndmask_b32_e64 v22, v22, v23, s[2:3]
	s_nop 0
	v_mov_b32_dpp v23, v24 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v24, v23, v22, v20
	v_perm_b32 v22, v23, v22, v21
	ds_write2_b32 v6, v24, v22 offset0:48 offset1:116
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[144:149], v[30:35], a[96:99], v8, v10 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v8, a92
	v_accvgpr_read_b32 v22, a93
	v_cvt_pk_bf16_f32 v22, v8, v22
	v_accvgpr_read_b32 v8, a94
	v_accvgpr_read_b32 v23, a95
	v_cvt_pk_bf16_f32 v23, v8, v23
	v_cndmask_b32_e64 v8, v23, v22, s[2:3]
	v_cndmask_b32_e64 v22, v22, v23, s[2:3]
	s_nop 0
	v_mov_b32_dpp v8, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v23, v8, v22, v20
	v_perm_b32 v8, v8, v22, v21
	ds_write2_b32 v6, v23, v8 offset0:56 offset1:124
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[186:191], v[90:95], a[96:99], v4, v2 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[186:191], v[96:101], a[100:103], v4, v2 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	ds_read_b128 v[22:25], v1 offset:55296
	ds_read_b128 v[26:29], v1 offset:56384
	ds_read_b128 v[36:39], v1 offset:57472
	ds_read_b128 v[132:135], v1 offset:58560
	s_add_i32 s0, s4, 32
	s_lshl_b32 s0, s0, 16
	s_add_i32 s0, s1, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[186:191], v[102:107], a[104:107], v4, v2 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(3)
	buffer_store_dwordx4 v[22:25], v0, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(2)
	buffer_store_dwordx4 v[26:29], v7, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[36:39], v18, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[132:135], v19, s[8:11], s0 offen nt
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[186:191], v[108:113], a[108:111], v4, v2 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[186:191], v[114:119], a[112:115], v4, v3 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[186:191], v[120:125], a[116:119], v4, v3 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[186:191], v[126:131], a[120:123], v4, v3 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[186:191], v[156:161], a[124:127], v4, v3 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[84:89], v[30:35], a[128:131], v9, v10 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a96
	v_accvgpr_read_b32 v8, a97
	v_cvt_pk_bf16_f32 v22, v4, v8
	v_accvgpr_read_b32 v4, a98
	v_accvgpr_read_b32 v8, a99
	v_cvt_pk_bf16_f32 v23, v4, v8
	v_cndmask_b32_e64 v4, v23, v22, s[2:3]
	v_cndmask_b32_e64 v8, v22, v23, s[2:3]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v22, v4, v8, v20
	v_perm_b32 v4, v4, v8, v21
	ds_write2_b32 v6, v22, v4 offset1:68
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[84:89], v[42:47], a[132:135], v9, v10 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a100
	v_accvgpr_read_b32 v8, a101
	v_cvt_pk_bf16_f32 v22, v4, v8
	v_accvgpr_read_b32 v4, a102
	v_accvgpr_read_b32 v8, a103
	v_cvt_pk_bf16_f32 v23, v4, v8
	v_cndmask_b32_e64 v4, v23, v22, s[2:3]
	v_cndmask_b32_e64 v8, v22, v23, s[2:3]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v22, v4, v8, v20
	v_perm_b32 v4, v4, v8, v21
	ds_write2_b32 v6, v22, v4 offset0:8 offset1:76
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[84:89], v[48:53], a[136:139], v9, v10 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a104
	v_accvgpr_read_b32 v8, a105
	v_cvt_pk_bf16_f32 v22, v4, v8
	v_accvgpr_read_b32 v4, a106
	v_accvgpr_read_b32 v8, a107
	v_cvt_pk_bf16_f32 v23, v4, v8
	v_cndmask_b32_e64 v4, v23, v22, s[2:3]
	v_cndmask_b32_e64 v8, v22, v23, s[2:3]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v22, v4, v8, v20
	v_perm_b32 v4, v4, v8, v21
	ds_write2_b32 v6, v22, v4 offset0:16 offset1:84
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[84:89], v[54:59], a[140:143], v9, v10 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a108
	v_accvgpr_read_b32 v8, a109
	v_cvt_pk_bf16_f32 v22, v4, v8
	v_accvgpr_read_b32 v4, a110
	v_accvgpr_read_b32 v8, a111
	v_cvt_pk_bf16_f32 v23, v4, v8
	v_cndmask_b32_e64 v4, v23, v22, s[2:3]
	v_cndmask_b32_e64 v8, v22, v23, s[2:3]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v22, v4, v8, v20
	v_perm_b32 v4, v4, v8, v21
	ds_write2_b32 v6, v22, v4 offset0:24 offset1:92
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[84:89], v[60:65], a[144:147], v9, v11 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a112
	v_accvgpr_read_b32 v8, a113
	v_cvt_pk_bf16_f32 v22, v4, v8
	v_accvgpr_read_b32 v4, a114
	v_accvgpr_read_b32 v8, a115
	v_cvt_pk_bf16_f32 v23, v4, v8
	v_cndmask_b32_e64 v4, v23, v22, s[2:3]
	v_cndmask_b32_e64 v8, v22, v23, s[2:3]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v22, v4, v8, v20
	v_perm_b32 v4, v4, v8, v21
	ds_write2_b32 v6, v22, v4 offset0:32 offset1:100
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[84:89], v[66:71], a[148:151], v9, v11 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a116
	v_accvgpr_read_b32 v8, a117
	v_cvt_pk_bf16_f32 v22, v4, v8
	v_accvgpr_read_b32 v4, a118
	v_accvgpr_read_b32 v8, a119
	v_cvt_pk_bf16_f32 v23, v4, v8
	v_cndmask_b32_e64 v4, v23, v22, s[2:3]
	v_cndmask_b32_e64 v8, v22, v23, s[2:3]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v22, v4, v8, v20
	v_perm_b32 v4, v4, v8, v21
	ds_write2_b32 v6, v22, v4 offset0:40 offset1:108
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[84:89], v[72:77], a[152:155], v9, v11 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a120
	v_accvgpr_read_b32 v8, a121
	v_cvt_pk_bf16_f32 v22, v4, v8
	v_accvgpr_read_b32 v4, a122
	v_accvgpr_read_b32 v8, a123
	v_cvt_pk_bf16_f32 v23, v4, v8
	v_cndmask_b32_e64 v4, v23, v22, s[2:3]
	v_cndmask_b32_e64 v8, v22, v23, s[2:3]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v22, v4, v8, v20
	v_perm_b32 v4, v4, v8, v21
	ds_write2_b32 v6, v22, v4 offset0:48 offset1:116
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[84:89], v[78:83], a[156:159], v9, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a124
	v_accvgpr_read_b32 v8, a125
	v_cvt_pk_bf16_f32 v22, v4, v8
	v_accvgpr_read_b32 v4, a126
	v_accvgpr_read_b32 v8, a127
	v_cvt_pk_bf16_f32 v23, v4, v8
	v_cndmask_b32_e64 v4, v23, v22, s[2:3]
	v_cndmask_b32_e64 v8, v22, v23, s[2:3]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v22, v4, v8, v20
	v_perm_b32 v4, v4, v8, v21
	ds_write2_b32 v6, v22, v4 offset0:56 offset1:124
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[192:197], v[156:161], a[156:159], v5, v3 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[192:197], v[126:131], a[152:155], v5, v3 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	ds_read_b128 v[22:25], v1 offset:55296
	ds_read_b128 v[26:29], v1 offset:56384
	ds_read_b128 v[36:39], v1 offset:57472
	ds_read_b128 v[84:87], v1 offset:58560
	s_add_i32 s0, s4, 48
	s_lshl_b32 s0, s0, 16
	s_add_i32 s0, s1, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[192:197], v[120:125], a[148:151], v5, v3 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(3)
	buffer_store_dwordx4 v[22:25], v0, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(2)
	buffer_store_dwordx4 v[26:29], v7, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[36:39], v18, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[84:87], v19, s[8:11], s0 offen nt
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[192:197], v[114:119], a[144:147], v5, v3 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[192:197], v[108:113], a[140:143], v5, v2 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[192:197], v[102:107], a[136:139], v5, v2 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[192:197], v[96:101], a[132:135], v5, v2 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[192:197], v[90:95], a[128:131], v5, v2 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[162:167], v[78:83], a[188:191], v9, v11 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_nop 7
	s_nop 2
	v_accvgpr_read_b32 v4, a128
	v_accvgpr_read_b32 v8, a129
	v_cvt_pk_bf16_f32 v22, v4, v8
	v_accvgpr_read_b32 v4, a130
	v_accvgpr_read_b32 v8, a131
	v_cvt_pk_bf16_f32 v23, v4, v8
	v_cndmask_b32_e64 v4, v23, v22, s[2:3]
	v_cndmask_b32_e64 v8, v22, v23, s[2:3]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[162:167], v[72:77], a[184:187], v9, v11 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_perm_b32 v22, v4, v8, v20
	v_perm_b32 v4, v4, v8, v21
	ds_write2_b32 v6, v22, v4 offset1:68
	v_accvgpr_read_b32 v4, a132
	v_accvgpr_read_b32 v8, a133
	v_cvt_pk_bf16_f32 v22, v4, v8
	v_accvgpr_read_b32 v4, a134
	v_accvgpr_read_b32 v8, a135
	v_cvt_pk_bf16_f32 v23, v4, v8
	v_cndmask_b32_e64 v4, v23, v22, s[2:3]
	v_cndmask_b32_e64 v8, v22, v23, s[2:3]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v22, v4, v8, v20
	v_perm_b32 v4, v4, v8, v21
	ds_write2_b32 v6, v22, v4 offset0:8 offset1:76
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[162:167], v[66:71], a[180:183], v9, v11 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a136
	v_accvgpr_read_b32 v8, a137
	v_cvt_pk_bf16_f32 v22, v4, v8
	v_accvgpr_read_b32 v4, a138
	v_accvgpr_read_b32 v8, a139
	v_cvt_pk_bf16_f32 v23, v4, v8
	v_cndmask_b32_e64 v4, v23, v22, s[2:3]
	v_cndmask_b32_e64 v8, v22, v23, s[2:3]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v22, v4, v8, v20
	v_perm_b32 v4, v4, v8, v21
	ds_write2_b32 v6, v22, v4 offset0:16 offset1:84
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[162:167], v[60:65], a[176:179], v9, v11 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a140
	v_accvgpr_read_b32 v8, a141
	v_cvt_pk_bf16_f32 v22, v4, v8
	v_accvgpr_read_b32 v4, a142
	v_accvgpr_read_b32 v8, a143
	v_cvt_pk_bf16_f32 v23, v4, v8
	v_cndmask_b32_e64 v4, v23, v22, s[2:3]
	v_cndmask_b32_e64 v8, v22, v23, s[2:3]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v22, v4, v8, v20
	v_perm_b32 v4, v4, v8, v21
	ds_write2_b32 v6, v22, v4 offset0:24 offset1:92
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[162:167], v[54:59], a[172:175], v9, v10 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a144
	v_accvgpr_read_b32 v8, a145
	v_cvt_pk_bf16_f32 v22, v4, v8
	v_accvgpr_read_b32 v4, a146
	v_accvgpr_read_b32 v8, a147
	v_cvt_pk_bf16_f32 v23, v4, v8
	v_cndmask_b32_e64 v4, v23, v22, s[2:3]
	v_cndmask_b32_e64 v8, v22, v23, s[2:3]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v22, v4, v8, v20
	v_perm_b32 v4, v4, v8, v21
	ds_write2_b32 v6, v22, v4 offset0:32 offset1:100
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[162:167], v[48:53], a[168:171], v9, v10 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a148
	v_accvgpr_read_b32 v8, a149
	v_cvt_pk_bf16_f32 v22, v4, v8
	v_accvgpr_read_b32 v4, a150
	v_accvgpr_read_b32 v8, a151
	v_cvt_pk_bf16_f32 v23, v4, v8
	v_cndmask_b32_e64 v4, v23, v22, s[2:3]
	v_cndmask_b32_e64 v8, v22, v23, s[2:3]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v22, v4, v8, v20
	v_perm_b32 v4, v4, v8, v21
	ds_write2_b32 v6, v22, v4 offset0:40 offset1:108
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[162:167], v[42:47], a[164:167], v9, v10 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a152
	v_accvgpr_read_b32 v8, a153
	v_cvt_pk_bf16_f32 v22, v4, v8
	v_accvgpr_read_b32 v4, a154
	v_accvgpr_read_b32 v8, a155
	v_cvt_pk_bf16_f32 v23, v4, v8
	v_cndmask_b32_e64 v4, v23, v22, s[2:3]
	v_cndmask_b32_e64 v8, v22, v23, s[2:3]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v22, v4, v8, v20
	v_perm_b32 v4, v4, v8, v21
	ds_write2_b32 v6, v22, v4 offset0:48 offset1:116
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[162:167], v[30:35], a[160:163], v9, v10 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a156
	v_accvgpr_read_b32 v8, a157
	v_cvt_pk_bf16_f32 v22, v4, v8
	v_accvgpr_read_b32 v4, a158
	v_accvgpr_read_b32 v8, a159
	v_cvt_pk_bf16_f32 v23, v4, v8
	v_cndmask_b32_e64 v4, v23, v22, s[2:3]
	v_cndmask_b32_e64 v8, v22, v23, s[2:3]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v22, v4, v8, v20
	v_perm_b32 v4, v4, v8, v21
	ds_write2_b32 v6, v22, v4 offset0:56 offset1:124
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[198:203], v[90:95], a[160:163], v5, v2 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[198:203], v[96:101], a[164:167], v5, v2 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	ds_read_b128 v[22:25], v1 offset:55296
	ds_read_b128 v[26:29], v1 offset:56384
	ds_read_b128 v[36:39], v1 offset:57472
	ds_read_b128 v[84:87], v1 offset:58560
	s_add_i32 s0, s4, 64
	s_lshl_b32 s0, s0, 16
	s_add_i32 s0, s1, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[198:203], v[102:107], a[168:171], v5, v2 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(3)
	buffer_store_dwordx4 v[22:25], v0, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(2)
	buffer_store_dwordx4 v[26:29], v7, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[36:39], v18, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[84:87], v19, s[8:11], s0 offen nt
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[198:203], v[108:113], a[172:175], v5, v2 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[198:203], v[114:119], a[176:179], v5, v3 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[198:203], v[120:125], a[180:183], v5, v3 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[198:203], v[126:131], a[184:187], v5, v3 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[198:203], v[156:161], a[188:191], v5, v3 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[168:173], v[30:35], a[192:195], v9, v10 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a160
	v_accvgpr_read_b32 v8, a161
	v_cvt_pk_bf16_f32 v22, v4, v8
	v_accvgpr_read_b32 v4, a162
	v_accvgpr_read_b32 v8, a163
	v_cvt_pk_bf16_f32 v23, v4, v8
	v_cndmask_b32_e64 v4, v23, v22, s[2:3]
	v_cndmask_b32_e64 v8, v22, v23, s[2:3]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v22, v4, v8, v20
	v_perm_b32 v4, v4, v8, v21
	ds_write2_b32 v6, v22, v4 offset1:68
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[168:173], v[42:47], a[196:199], v9, v10 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a164
	v_accvgpr_read_b32 v8, a165
	v_cvt_pk_bf16_f32 v22, v4, v8
	v_accvgpr_read_b32 v4, a166
	v_accvgpr_read_b32 v8, a167
	v_cvt_pk_bf16_f32 v23, v4, v8
	v_cndmask_b32_e64 v4, v23, v22, s[2:3]
	v_cndmask_b32_e64 v8, v22, v23, s[2:3]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v22, v4, v8, v20
	v_perm_b32 v4, v4, v8, v21
	ds_write2_b32 v6, v22, v4 offset0:8 offset1:76
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[168:173], v[48:53], a[200:203], v9, v10 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a168
	v_accvgpr_read_b32 v8, a169
	v_cvt_pk_bf16_f32 v22, v4, v8
	v_accvgpr_read_b32 v4, a170
	v_accvgpr_read_b32 v8, a171
	v_cvt_pk_bf16_f32 v23, v4, v8
	v_cndmask_b32_e64 v4, v23, v22, s[2:3]
	v_cndmask_b32_e64 v8, v22, v23, s[2:3]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v22, v4, v8, v20
	v_perm_b32 v4, v4, v8, v21
	ds_write2_b32 v6, v22, v4 offset0:16 offset1:84
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[168:173], v[54:59], a[204:207], v9, v10 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a172
	v_accvgpr_read_b32 v8, a173
	v_cvt_pk_bf16_f32 v22, v4, v8
	v_accvgpr_read_b32 v4, a174
	v_accvgpr_read_b32 v8, a175
	v_cvt_pk_bf16_f32 v23, v4, v8
	v_cndmask_b32_e64 v4, v23, v22, s[2:3]
	v_cndmask_b32_e64 v8, v22, v23, s[2:3]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v22, v4, v8, v20
	v_perm_b32 v4, v4, v8, v21
	ds_write2_b32 v6, v22, v4 offset0:24 offset1:92
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[168:173], v[60:65], a[208:211], v9, v11 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a176
	v_accvgpr_read_b32 v8, a177
	v_cvt_pk_bf16_f32 v22, v4, v8
	v_accvgpr_read_b32 v4, a178
	v_accvgpr_read_b32 v8, a179
	v_cvt_pk_bf16_f32 v23, v4, v8
	v_cndmask_b32_e64 v4, v23, v22, s[2:3]
	v_cndmask_b32_e64 v8, v22, v23, s[2:3]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v22, v4, v8, v20
	v_perm_b32 v4, v4, v8, v21
	ds_write2_b32 v6, v22, v4 offset0:32 offset1:100
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[168:173], v[66:71], a[212:215], v9, v11 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a180
	v_accvgpr_read_b32 v8, a181
	v_cvt_pk_bf16_f32 v22, v4, v8
	v_accvgpr_read_b32 v4, a182
	v_accvgpr_read_b32 v8, a183
	v_cvt_pk_bf16_f32 v23, v4, v8
	v_cndmask_b32_e64 v4, v23, v22, s[2:3]
	v_cndmask_b32_e64 v8, v22, v23, s[2:3]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v22, v4, v8, v20
	v_perm_b32 v4, v4, v8, v21
	ds_write2_b32 v6, v22, v4 offset0:40 offset1:108
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[168:173], v[72:77], a[216:219], v9, v11 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a184
	v_accvgpr_read_b32 v8, a185
	v_cvt_pk_bf16_f32 v22, v4, v8
	v_accvgpr_read_b32 v4, a186
	v_accvgpr_read_b32 v8, a187
	v_cvt_pk_bf16_f32 v23, v4, v8
	v_cndmask_b32_e64 v4, v23, v22, s[2:3]
	v_cndmask_b32_e64 v8, v22, v23, s[2:3]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v22, v4, v8, v20
	v_perm_b32 v4, v4, v8, v21
	ds_write2_b32 v6, v22, v4 offset0:48 offset1:116
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[168:173], v[78:83], a[220:223], v9, v11 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a188
	v_accvgpr_read_b32 v8, a189
	v_cvt_pk_bf16_f32 v22, v4, v8
	v_accvgpr_read_b32 v4, a190
	v_accvgpr_read_b32 v8, a191
	v_cvt_pk_bf16_f32 v23, v4, v8
	v_cndmask_b32_e64 v4, v23, v22, s[2:3]
	v_cndmask_b32_e64 v8, v22, v23, s[2:3]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v22, v4, v8, v20
	v_perm_b32 v4, v4, v8, v21
	ds_write2_b32 v6, v22, v4 offset0:56 offset1:124
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[204:209], v[156:161], a[220:223], v5, v3 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[204:209], v[126:131], a[216:219], v5, v3 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	ds_read_b128 v[22:25], v1 offset:55296
	ds_read_b128 v[26:29], v1 offset:56384
	ds_read_b128 v[36:39], v1 offset:57472
	ds_read_b128 v[84:87], v1 offset:58560
	s_add_i32 s0, s4, 0x50
	s_lshl_b32 s0, s0, 16
	s_add_i32 s0, s1, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[204:209], v[120:125], a[212:215], v5, v3 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(3)
	buffer_store_dwordx4 v[22:25], v0, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(2)
	buffer_store_dwordx4 v[26:29], v7, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[36:39], v18, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[84:87], v19, s[8:11], s0 offen nt
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[204:209], v[114:119], a[208:211], v5, v3 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[204:209], v[108:113], a[204:207], v5, v2 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[204:209], v[102:107], a[200:203], v5, v2 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[204:209], v[96:101], a[196:199], v5, v2 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[204:209], v[90:95], a[192:195], v5, v2 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[150:155], v[78:83], a[252:255], v9, v11 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_nop 7
	s_nop 2
	v_accvgpr_read_b32 v4, a192
	v_accvgpr_read_b32 v8, a193
	v_cvt_pk_bf16_f32 v22, v4, v8
	v_accvgpr_read_b32 v4, a194
	v_accvgpr_read_b32 v8, a195
	v_cvt_pk_bf16_f32 v23, v4, v8
	v_cndmask_b32_e64 v4, v23, v22, s[2:3]
	v_cndmask_b32_e64 v8, v22, v23, s[2:3]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[150:155], v[72:77], a[248:251], v9, v11 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_perm_b32 v22, v4, v8, v20
	v_perm_b32 v4, v4, v8, v21
	ds_write2_b32 v6, v22, v4 offset1:68
	v_accvgpr_read_b32 v4, a196
	v_accvgpr_read_b32 v8, a197
	v_cvt_pk_bf16_f32 v22, v4, v8
	v_accvgpr_read_b32 v4, a198
	v_accvgpr_read_b32 v8, a199
	v_cvt_pk_bf16_f32 v23, v4, v8
	v_cndmask_b32_e64 v4, v23, v22, s[2:3]
	v_cndmask_b32_e64 v8, v22, v23, s[2:3]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v22, v4, v8, v20
	v_perm_b32 v4, v4, v8, v21
	ds_write2_b32 v6, v22, v4 offset0:8 offset1:76
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[150:155], v[66:71], a[244:247], v9, v11 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a200
	v_accvgpr_read_b32 v8, a201
	v_cvt_pk_bf16_f32 v22, v4, v8
	v_accvgpr_read_b32 v4, a202
	v_accvgpr_read_b32 v8, a203
	v_cvt_pk_bf16_f32 v23, v4, v8
	v_cndmask_b32_e64 v4, v23, v22, s[2:3]
	v_cndmask_b32_e64 v8, v22, v23, s[2:3]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v22, v4, v8, v20
	v_perm_b32 v4, v4, v8, v21
	ds_write2_b32 v6, v22, v4 offset0:16 offset1:84
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[150:155], v[60:65], a[240:243], v9, v11 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a204
	v_accvgpr_read_b32 v8, a205
	v_cvt_pk_bf16_f32 v22, v4, v8
	v_accvgpr_read_b32 v4, a206
	v_accvgpr_read_b32 v8, a207
	v_cvt_pk_bf16_f32 v23, v4, v8
	v_cndmask_b32_e64 v4, v23, v22, s[2:3]
	v_cndmask_b32_e64 v8, v22, v23, s[2:3]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v11, v4, v8, v20
	v_perm_b32 v4, v4, v8, v21
	ds_write2_b32 v6, v11, v4 offset0:24 offset1:92
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[150:155], v[54:59], a[236:239], v9, v10 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a208
	v_accvgpr_read_b32 v8, a209
	v_cvt_pk_bf16_f32 v22, v4, v8
	v_accvgpr_read_b32 v4, a210
	v_accvgpr_read_b32 v8, a211
	v_cvt_pk_bf16_f32 v23, v4, v8
	v_cndmask_b32_e64 v4, v23, v22, s[2:3]
	v_cndmask_b32_e64 v8, v22, v23, s[2:3]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v11, v4, v8, v20
	v_perm_b32 v4, v4, v8, v21
	ds_write2_b32 v6, v11, v4 offset0:32 offset1:100
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[150:155], v[48:53], a[232:235], v9, v10 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a212
	v_accvgpr_read_b32 v8, a213
	v_cvt_pk_bf16_f32 v22, v4, v8
	v_accvgpr_read_b32 v4, a214
	v_accvgpr_read_b32 v8, a215
	v_cvt_pk_bf16_f32 v23, v4, v8
	v_cndmask_b32_e64 v4, v23, v22, s[2:3]
	v_cndmask_b32_e64 v8, v22, v23, s[2:3]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v11, v4, v8, v20
	v_perm_b32 v4, v4, v8, v21
	ds_write2_b32 v6, v11, v4 offset0:40 offset1:108
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[150:155], v[42:47], a[228:231], v9, v10 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a216
	v_accvgpr_read_b32 v8, a217
	v_cvt_pk_bf16_f32 v22, v4, v8
	v_accvgpr_read_b32 v4, a218
	v_accvgpr_read_b32 v8, a219
	v_cvt_pk_bf16_f32 v23, v4, v8
	v_cndmask_b32_e64 v4, v23, v22, s[2:3]
	v_cndmask_b32_e64 v8, v22, v23, s[2:3]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v11, v4, v8, v20
	v_perm_b32 v4, v4, v8, v21
	ds_write2_b32 v6, v11, v4 offset0:48 offset1:116
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[150:155], v[30:35], a[224:227], v9, v10 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v4, a220
	v_accvgpr_read_b32 v8, a221
	v_cvt_pk_bf16_f32 v8, v4, v8
	v_accvgpr_read_b32 v4, a222
	v_accvgpr_read_b32 v9, a223
	v_cvt_pk_bf16_f32 v9, v4, v9
	v_cndmask_b32_e64 v4, v9, v8, s[2:3]
	v_cndmask_b32_e64 v8, v8, v9, s[2:3]
	s_nop 0
	v_mov_b32_dpp v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v9, v4, v8, v20
	v_perm_b32 v4, v4, v8, v21
	ds_write2_b32 v6, v9, v4 offset0:56 offset1:124
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[12:17], v[90:95], a[224:227], v5, v2 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[12:17], v[96:101], a[228:231], v5, v2 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	ds_read_b128 v[8:11], v1 offset:55296
	ds_read_b128 v[22:25], v1 offset:56384
	ds_read_b128 v[26:29], v1 offset:57472
	ds_read_b128 v[30:33], v1 offset:58560
	s_add_i32 s0, s4, 0x60
	s_lshl_b32 s0, s0, 16
	s_add_i32 s0, s1, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[12:17], v[102:107], a[232:235], v5, v2 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(3)
	buffer_store_dwordx4 v[8:11], v0, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(2)
	buffer_store_dwordx4 v[22:25], v7, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[26:29], v18, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[30:33], v19, s[8:11], s0 offen nt
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[12:17], v[108:113], a[236:239], v5, v2 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[12:17], v[114:119], a[240:243], v5, v3 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[12:17], v[120:125], a[244:247], v5, v3 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[12:17], v[126:131], a[248:251], v5, v3 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[12:17], v[156:161], a[252:255], v5, v3 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_accvgpr_read_b32 v2, a224
	v_accvgpr_read_b32 v3, a225
	v_cvt_pk_bf16_f32 v2, v2, v3
	v_accvgpr_read_b32 v3, a226
	v_accvgpr_read_b32 v4, a227
	v_cvt_pk_bf16_f32 v3, v3, v4
	v_accvgpr_read_b32 v4, a228
	v_accvgpr_read_b32 v5, a229
	v_cvt_pk_bf16_f32 v4, v4, v5
	v_accvgpr_read_b32 v5, a230
	v_accvgpr_read_b32 v8, a231
	v_cvt_pk_bf16_f32 v5, v5, v8
	v_accvgpr_read_b32 v8, a232
	v_accvgpr_read_b32 v9, a233
	v_cvt_pk_bf16_f32 v8, v8, v9
	v_accvgpr_read_b32 v9, a234
	v_accvgpr_read_b32 v10, a235
	v_cvt_pk_bf16_f32 v9, v9, v10
	v_accvgpr_read_b32 v10, a236
	v_accvgpr_read_b32 v11, a237
	v_cvt_pk_bf16_f32 v10, v10, v11
	v_accvgpr_read_b32 v11, a238
	v_accvgpr_read_b32 v12, a239
	v_cvt_pk_bf16_f32 v11, v11, v12
	v_accvgpr_read_b32 v12, a240
	v_accvgpr_read_b32 v13, a241
	v_cvt_pk_bf16_f32 v12, v12, v13
	v_accvgpr_read_b32 v13, a242
	v_accvgpr_read_b32 v14, a243
	v_cvt_pk_bf16_f32 v13, v13, v14
	v_accvgpr_read_b32 v14, a244
	v_accvgpr_read_b32 v15, a245
	v_cvt_pk_bf16_f32 v14, v14, v15
	v_accvgpr_read_b32 v15, a246
	v_accvgpr_read_b32 v16, a247
	v_cvt_pk_bf16_f32 v15, v15, v16
	v_accvgpr_read_b32 v16, a248
	v_accvgpr_read_b32 v17, a249
	v_cvt_pk_bf16_f32 v16, v16, v17
	v_accvgpr_read_b32 v17, a250
	v_accvgpr_read_b32 v22, a251
	v_cvt_pk_bf16_f32 v17, v17, v22
	v_accvgpr_read_b32 v22, a252
	v_accvgpr_read_b32 v23, a253
	v_cvt_pk_bf16_f32 v22, v22, v23
	v_accvgpr_read_b32 v23, a254
	v_accvgpr_read_b32 v24, a255
	v_cvt_pk_bf16_f32 v23, v23, v24
	v_cndmask_b32_e64 v24, v3, v2, s[2:3]
	v_cndmask_b32_e64 v25, v5, v4, s[2:3]
	v_cndmask_b32_e64 v26, v9, v8, s[2:3]
	v_cndmask_b32_e64 v27, v11, v10, s[2:3]
	v_cndmask_b32_e64 v28, v13, v12, s[2:3]
	v_cndmask_b32_e64 v29, v15, v14, s[2:3]
	v_cndmask_b32_e64 v30, v17, v16, s[2:3]
	v_cndmask_b32_e64 v31, v23, v22, s[2:3]
	v_cndmask_b32_e64 v2, v2, v3, s[2:3]
	v_mov_b32_dpp v3, v24 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v4, v4, v5, s[2:3]
	v_mov_b32_dpp v5, v25 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v9, s[2:3]
	v_mov_b32_dpp v9, v26 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v10, v10, v11, s[2:3]
	v_mov_b32_dpp v11, v27 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v12, v12, v13, s[2:3]
	v_mov_b32_dpp v13, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v14, v14, v15, s[2:3]
	v_mov_b32_dpp v15, v29 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v16, v16, v17, s[2:3]
	v_mov_b32_dpp v17, v30 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v22, v22, v23, s[2:3]
	v_mov_b32_dpp v23, v31 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v24, v3, v2, v20
	v_perm_b32 v2, v3, v2, v21
	v_perm_b32 v3, v5, v4, v20
	v_perm_b32 v4, v5, v4, v21
	v_perm_b32 v5, v9, v8, v20
	v_perm_b32 v8, v9, v8, v21
	v_perm_b32 v9, v11, v10, v20
	v_perm_b32 v10, v11, v10, v21
	v_perm_b32 v11, v13, v12, v20
	v_perm_b32 v12, v13, v12, v21
	v_perm_b32 v13, v15, v14, v20
	v_perm_b32 v14, v15, v14, v21
	v_perm_b32 v15, v17, v16, v20
	v_perm_b32 v16, v17, v16, v21
	v_perm_b32 v17, v23, v22, v20
	v_perm_b32 v20, v23, v22, v21
	ds_write2_b32 v6, v24, v2 offset1:68
	ds_write2_b32 v6, v3, v4 offset0:8 offset1:76
	ds_write2_b32 v6, v5, v8 offset0:16 offset1:84
	ds_write2_b32 v6, v9, v10 offset0:24 offset1:92
	ds_write2_b32 v6, v11, v12 offset0:32 offset1:100
	ds_write2_b32 v6, v13, v14 offset0:40 offset1:108
	ds_write2_b32 v6, v15, v16 offset0:48 offset1:116
	ds_write2_b32 v6, v17, v20 offset0:56 offset1:124
	s_waitcnt lgkmcnt(0)
	ds_read_b128 v[2:5], v1 offset:55296
	ds_read_b128 v[8:11], v1 offset:56384
	ds_read_b128 v[12:15], v1 offset:57472
	ds_read_b128 v[20:23], v1 offset:58560
	s_add_i32 s0, s4, 0x70
	s_lshl_b32 s0, s0, 16
	s_add_i32 s0, s1, s0
	s_waitcnt lgkmcnt(3)
	buffer_store_dwordx4 v[2:5], v0, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(2)
	buffer_store_dwordx4 v[8:11], v7, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[12:15], v18, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[20:23], v19, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(0)
	s_endpgm
.Lrocke_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_32768x4096x6144_c0c1_k3063b64df0_end:
.size rocke_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_32768x4096x6144_c0c1_k3063b64df0, .Lrocke_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_32768x4096x6144_c0c1_k3063b64df0_end-rocke_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_32768x4096x6144_c0c1_k3063b64df0

.rodata
.p2align 6
.amdhsa_kernel rocke_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_32768x4096x6144_c0c1_k3063b64df0
  .amdhsa_group_segment_fixed_size 153600
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_kernarg_size 40
  .amdhsa_user_sgpr_count 12
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_user_sgpr_kernarg_preload_length 10
  .amdhsa_user_sgpr_kernarg_preload_offset 0
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_next_free_vgpr 468
  .amdhsa_next_free_sgpr 50
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
    Description: MX E2M3 GEMM at 32768x4096x6144 on tiled operands and shuffled scales
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
    SwizzleTensorB: True
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
  AssertSizeEqual: {0: 32768, 1: 4096, 2: 1, 3: 6144}
  AssertStrideAEqual: {1: 6144}
  AssertStrideBEqual: {1: 6144}
  AssertStrideDEqual: {1: 32768}
  AssertAlphaValue: 1
amdhsa.target: amdgcn-amd-amdhsa--gfx950
amdhsa.version: [ 1, 1 ]
amdhsa.kernels:
  - .name: rocke_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_32768x4096x6144_c0c1_k3063b64df0
    .symbol: rocke_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_32768x4096x6144_c0c1_k3063b64df0.kd
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
    .group_segment_fixed_size: 153600
    .kernarg_segment_align: 8
    .kernarg_segment_size: 40
    .max_flat_workgroup_size: 256
    .private_segment_fixed_size: 0
    .sgpr_count: 56
    .vgpr_count: 468
    .agpr_count: 256
    .wavefront_size: 64
...
.end_amdgpu_metadata
