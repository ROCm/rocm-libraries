// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// MX E3M2 x E3M2 GEMM for gfx950: E8M0 scales per 32 K elements, fp32 accumulation,
// bf16 D rounded to nearest even. 256 threads (wave64, 2x2 waves), macrotile 256x256x128,
// v_mfma_scale_f32_16x16x128_f8f6f4 with cbsz:3 blgp:3, 153600 bytes of static LDS.
//
// Fixed problem, enforced by the predicates in custom.config:
//   M = N = 4096, K = 65536, batch 1, TransA = T, TransB = N, lda = ldb = K, D column-major
//   with ldd = 4096. alpha must be 1 (AssertAlphaValue) and beta 0 (UseBeta false): the
//   kernel takes no scalar arguments and never reads C. The grid must be exactly 256
//   workgroups: every buffer descriptor has num_records 0x7ffffff0 and no load or store is
//   masked, so any other size reads and writes out of range. Its only arguments are the
//   five pointers in the metadata below, so it cannot take the UserArgs launch path.
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
//
// Edits to the provider's instruction listing, none of which changes an instruction up to
// s_endpgm:
//   - the address and encoding comments on each instruction are removed;
//   - the immediate branch offset of the main-loop s_cbranch_scc1 is replaced by the label
//     .Lmain_loop, which encodes to the same offset;
//   - the provider's 256 s_nop padding instructions after s_endpgm are dropped;
//   - the directives, kernel descriptor and metadata are written for code object v4
//     (amdhsa.version [1, 1], no hidden arguments) from the provider's kernel descriptor.

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.amdhsa_code_object_version 4
.text
.globl rocke_gemm_mx_e3m2_e3m2_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_4096x4096x65536_c0c1_k4b371438f4
.p2align 8
.type rocke_gemm_mx_e3m2_e3m2_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_4096x4096x65536_c0c1_k4b371438f4,@function
rocke_gemm_mx_e3m2_e3m2_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_4096x4096x65536_c0c1_k4b371438f4:
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
	s_and_b32 s14, s2, 7
	s_lshr_b32 s2, s2, 3
	s_lshr_b32 s15, s14, 1
	s_lshl_b32 s15, s15, 2
	s_lshr_b32 s16, s2, 3
	s_add_i32 s15, s15, s16
	s_and_b32 s14, s14, 1
	s_lshl_b32 s14, s14, 3
	s_and_b32 s2, s2, 7
	s_add_i32 s2, s14, s2
	s_mul_i32 s14, s15, 0xc00000
	s_lshl_b32 s15, s15, 8
	s_lshl_b32 s16, s2, 8
	s_lshl_b32 s12, s12, 7
	s_lshl_b32 s13, s13, 7
	s_waitcnt lgkmcnt(0)
	s_add_u32 s4, s4, s14
	s_mul_i32 s2, s2, 0xc00000
	s_addc_u32 s5, s5, 0
	s_add_u32 s20, s6, s2
	s_addc_u32 s21, s7, 0
	v_lshlrev_b32_e32 v1, 4, v0
	s_mul_i32 s2, s3, 0x1800
	s_lshr_b32 s14, s12, 4
	s_lshr_b32 s17, s13, 4
	s_and_b32 s5, s5, 0xffff
	s_and_b32 s21, s21, 0xffff
	s_add_i32 s18, s15, s12
	s_add_i32 s19, s16, s13
	s_and_b32 s25, s9, 0xffff
	s_and_b32 s9, s11, 0xffff
	s_lshl_b32 s3, s3, 8
	s_lshr_b32 s15, s15, 7
	s_lshr_b32 s16, s16, 7
	s_mov_b32 s6, 0x7ffffff0
	s_mov_b32 s7, 0x31014000
	s_mov_b32 s22, 0x7ffffff0
	s_mov_b32 s23, 0x31014000
	s_mov_b32 s24, s8
	s_mov_b32 s26, 0x7ffffff0
	s_mov_b32 s27, 0x31014000
	s_mov_b32 s8, s10
	s_mov_b32 s10, 0x7ffffff0
	s_mov_b32 s11, 0x31014000
	v_lshlrev_b32_e32 v2, 2, v0
	s_lshr_b32 s12, s12, 7
	s_lshr_b32 s13, s13, 7
	s_lshl_b32 s28, s15, 9
	s_mov_b32 m0, s2
	s_lshl_b32 s29, s16, 9
	buffer_load_dwordx4 v1, s[4:7], s2 offen lds
	buffer_load_dwordx4 v1, s[4:7], s2 offen offset:1024 lds
	buffer_load_dwordx4 v1, s[4:7], s2 offen offset:2048 lds
	buffer_load_dwordx4 v1, s[4:7], s2 offen offset:3072 lds
	s_add_i32 s30, s2, 0x1000
	s_mov_b32 m0, s30
	s_nop 0
	buffer_load_dwordx4 v1, s[4:7], s30 offen lds
	buffer_load_dwordx4 v1, s[4:7], s30 offen offset:1024 lds
	s_add_i32 s31, 0x6400, s2
	s_mov_b32 m0, s31
	s_nop 0
	buffer_load_dwordx4 v1, s[20:23], s2 offen lds
	buffer_load_dwordx4 v1, s[20:23], s2 offen offset:1024 lds
	buffer_load_dwordx4 v1, s[20:23], s2 offen offset:2048 lds
	buffer_load_dwordx4 v1, s[20:23], s2 offen offset:3072 lds
	s_add_i32 s32, 0x6400, s30
	s_mov_b32 m0, s32
	s_nop 0
	buffer_load_dwordx4 v1, s[20:23], s30 offen lds
	buffer_load_dwordx4 v1, s[20:23], s30 offen offset:1024 lds
	s_add_i32 s33, 0x6000, s3
	s_mov_b32 m0, s33
	s_add_i32 s28, s3, s28
	buffer_load_dword v2, s[24:27], s28 offen lds
	s_add_i32 s28, 0xc400, s3
	s_mov_b32 m0, s28
	s_add_i32 s29, s3, s29
	buffer_load_dword v2, s[8:11], s29 offen lds
	s_add_i32 s34, s15, 32
	s_lshl_b32 s34, s34, 9
	s_add_i32 s35, s2, 0x6000
	s_add_i32 s36, 0xc800, s2
	s_mov_b32 m0, s36
	s_nop 0
	buffer_load_dwordx4 v1, s[4:7], s35 offen lds
	buffer_load_dwordx4 v1, s[4:7], s35 offen offset:1024 lds
	buffer_load_dwordx4 v1, s[4:7], s35 offen offset:2048 lds
	buffer_load_dwordx4 v1, s[4:7], s35 offen offset:3072 lds
	s_add_i32 s37, 0xc800, s30
	s_mov_b32 m0, s37
	s_add_i32 s38, s2, 0x7000
	buffer_load_dwordx4 v1, s[4:7], s38 offen lds
	buffer_load_dwordx4 v1, s[4:7], s38 offen offset:1024 lds
	s_add_i32 s39, 0x12c00, s2
	s_mov_b32 m0, s39
	s_nop 0
	buffer_load_dwordx4 v1, s[20:23], s35 offen lds
	buffer_load_dwordx4 v1, s[20:23], s35 offen offset:1024 lds
	buffer_load_dwordx4 v1, s[20:23], s35 offen offset:2048 lds
	buffer_load_dwordx4 v1, s[20:23], s35 offen offset:3072 lds
	s_add_i32 s35, 0x12c00, s30
	s_mov_b32 m0, s35
	s_nop 0
	buffer_load_dwordx4 v1, s[20:23], s38 offen lds
	buffer_load_dwordx4 v1, s[20:23], s38 offen offset:1024 lds
	s_add_i32 s38, 0x12800, s3
	s_mov_b32 m0, s38
	s_add_i32 s34, s3, s34
	buffer_load_dword v2, s[24:27], s34 offen lds
	s_add_i32 s34, 0x18c00, s3
	s_mov_b32 m0, s34
	s_add_i32 s40, s29, 0x4000
	buffer_load_dword v2, s[8:11], s40 offen lds
	s_add_i32 s40, s15, 64
	s_lshl_b32 s40, s40, 9
	s_add_i32 s41, s2, 0xc000
	s_add_i32 s42, 0x19000, s2
	s_mov_b32 m0, s42
	s_nop 0
	buffer_load_dwordx4 v1, s[4:7], s41 offen lds
	buffer_load_dwordx4 v1, s[4:7], s41 offen offset:1024 lds
	buffer_load_dwordx4 v1, s[4:7], s41 offen offset:2048 lds
	buffer_load_dwordx4 v1, s[4:7], s41 offen offset:3072 lds
	s_add_i32 s43, 0x19000, s30
	s_mov_b32 m0, s43
	s_add_i32 s44, s2, 0xd000
	buffer_load_dwordx4 v1, s[4:7], s44 offen lds
	buffer_load_dwordx4 v1, s[4:7], s44 offen offset:1024 lds
	s_add_i32 s45, 0x1f400, s2
	s_mov_b32 m0, s45
	s_nop 0
	buffer_load_dwordx4 v1, s[20:23], s41 offen lds
	buffer_load_dwordx4 v1, s[20:23], s41 offen offset:1024 lds
	buffer_load_dwordx4 v1, s[20:23], s41 offen offset:2048 lds
	buffer_load_dwordx4 v1, s[20:23], s41 offen offset:3072 lds
	s_add_i32 s41, 0x1f400, s30
	s_mov_b32 m0, s41
	s_nop 0
	buffer_load_dwordx4 v1, s[20:23], s44 offen lds
	buffer_load_dwordx4 v1, s[20:23], s44 offen offset:1024 lds
	s_add_i32 s44, 0x1f000, s3
	s_mov_b32 m0, s44
	s_add_i32 s40, s3, s40
	buffer_load_dword v2, s[24:27], s40 offen lds
	s_add_i32 s40, 0x25400, s3
	s_mov_b32 m0, s40
	s_add_i32 s46, s29, 0x8000
	buffer_load_dword v2, s[8:11], s46 offen lds
	s_waitcnt vmcnt(28) lgkmcnt(0)
	s_barrier
	s_lshl_b32 s12, s12, 7
	s_lshl_b32 s13, s13, 7
	s_lshl_b32 s14, s14, 6
	v_lshlrev_b32_e32 v3, 1, v0
	v_add_u32_e32 v4, s14, v0
	v_add_lshl_u32 v5, s12, v3, 2
	v_add_lshl_u32 v3, s13, v3, 2
	v_lshlrev_b32_e32 v6, 4, v4
	ds_read_b64 v[8:9], v5 offset:24576
	ds_read_b64 v[10:11], v3 offset:50176
	ds_read_b128 v[12:15], v6
	v_lshlrev_b32_e32 v4, 3, v4
	ds_read_b64 v[16:17], v4 offset:16384
	s_lshl_b32 s12, s17, 6
	v_add_u32_e32 v7, s12, v0
	v_lshlrev_b32_e32 v18, 4, v7
	ds_read_b128 v[20:23], v18 offset:25600
	v_lshlrev_b32_e32 v7, 3, v7
	ds_read_b64 v[24:25], v7 offset:41984
	ds_read_b128 v[26:29], v18 offset:26624
	ds_read_b64 v[30:31], v7 offset:42496
	ds_read_b128 v[32:35], v18 offset:27648
	ds_read_b64 v[36:37], v7 offset:43008
	ds_read_b128 v[38:41], v18 offset:28672
	ds_read_b64 v[42:43], v7 offset:43520
	ds_read_b128 v[44:47], v18 offset:29696
	ds_read_b64 v[48:49], v7 offset:44032
	ds_read_b128 v[50:53], v18 offset:30720
	ds_read_b64 v[54:55], v7 offset:44544
	ds_read_b128 v[56:59], v18 offset:31744
	ds_read_b64 v[60:61], v7 offset:45056
	ds_read_b128 v[62:65], v18 offset:32768
	ds_read_b64 v[66:67], v7 offset:45568
	ds_read_b128 v[68:71], v6 offset:1024
	ds_read_b64 v[72:73], v4 offset:16896
	ds_read_b128 v[74:77], v6 offset:2048
	ds_read_b64 v[78:79], v4 offset:17408
	ds_read_b128 v[80:83], v6 offset:3072
	ds_read_b64 v[84:85], v4 offset:17920
	ds_read_b128 v[86:89], v6 offset:4096
	ds_read_b64 v[90:91], v4 offset:18432
	ds_read_b128 v[92:95], v6 offset:5120
	ds_read_b64 v[96:97], v4 offset:18944
	ds_read_b128 v[98:101], v6 offset:6144
	ds_read_b64 v[102:103], v4 offset:19456
	ds_read_b128 v[104:107], v6 offset:7168
	ds_read_b64 v[108:109], v4 offset:19968
	s_mov_b32 s12, 0
	v_accvgpr_write_b32 a0, 0
	v_accvgpr_write_b32 a1, 0
	v_accvgpr_write_b32 a2, 0
	v_accvgpr_write_b32 a3, 0
	v_add_u32_e32 v19, 0x10000, v5
	v_accvgpr_write_b32 a4, 0
	v_accvgpr_write_b32 a5, 0
	v_accvgpr_write_b32 a6, 0
	v_accvgpr_write_b32 a7, 0
	v_add_u32_e32 v110, 0x10000, v3
	v_accvgpr_write_b32 a8, 0
	v_accvgpr_write_b32 a9, 0
	v_accvgpr_write_b32 a10, 0
	v_accvgpr_write_b32 a11, 0
	v_add_u32_e32 v111, 0x10000, v4
	v_accvgpr_write_b32 a12, 0
	v_accvgpr_write_b32 a13, 0
	v_accvgpr_write_b32 a14, 0
	v_accvgpr_write_b32 a15, 0
	v_add_u32_e32 v112, 0x10000, v18
	v_accvgpr_write_b32 a16, 0
	v_accvgpr_write_b32 a17, 0
	v_accvgpr_write_b32 a18, 0
	v_accvgpr_write_b32 a19, 0
	v_add_u32_e32 v113, 0x10000, v7
	v_accvgpr_write_b32 a20, 0
	v_accvgpr_write_b32 a21, 0
	v_accvgpr_write_b32 a22, 0
	v_accvgpr_write_b32 a23, 0
	v_add_u32_e32 v114, 0x20000, v3
	v_accvgpr_write_b32 a24, 0
	v_accvgpr_write_b32 a25, 0
	v_accvgpr_write_b32 a26, 0
	v_accvgpr_write_b32 a27, 0
	v_add_u32_e32 v115, 0x10000, v6
	v_accvgpr_write_b32 a28, 0
	v_accvgpr_write_b32 a29, 0
	v_accvgpr_write_b32 a30, 0
	v_accvgpr_write_b32 a31, 0
	v_add_u32_e32 v116, 0x20000, v7
	v_accvgpr_write_b32 a32, 0
	v_accvgpr_write_b32 a33, 0
	v_accvgpr_write_b32 a34, 0
	v_accvgpr_write_b32 a35, 0
	v_add_u32_e32 v117, 0x20000, v18
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
	s_mul_i32 s13, s12, 12
	s_lshl_b32 s14, s13, 5
	s_add_i32 s17, s15, s14
	s_add_i32 s46, s17, 0x60
	s_mul_i32 s13, s13, 0x6000
	s_lshl_b32 s46, s46, 9
	s_add_i32 s14, s16, s14
	s_lshl_b32 s14, s14, 9
	s_waitcnt lgkmcnt(14)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[12:17], v[20:25], a[0:3], v8, v10 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[12:17], v[26:31], a[4:7], v8, v10 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[12:17], v[32:37], a[8:11], v8, v10 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[118:119], v19 offset:10240
	ds_read_b64 v[120:121], v110 offset:35840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[12:17], v[38:43], a[12:15], v8, v10 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_add_i32 s13, s2, s13
	s_add_i32 s47, s13, 0x12000
	s_mov_b32 m0, s2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[12:17], v[44:49], a[16:19], v8, v11 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s47 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[12:17], v[50:55], a[20:23], v8, v11 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[12:17], v[56:61], a[24:27], v8, v11 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[122:125], v6 offset:51200
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[12:17], v[62:67], a[28:31], v8, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[126:127], v111 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[68:73], v[62:67], a[60:63], v8, v11 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s47 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[68:73], v[56:61], a[56:59], v8, v11 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[12:15], v112 offset:11264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[68:73], v[50:55], a[52:55], v8, v11 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[16:17], v113 offset:27648
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[68:73], v[44:49], a[48:51], v8, v11 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[68:73], v[38:43], a[44:47], v8, v10 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s47 offen offset:2048 lds
	ds_read_b128 v[128:131], v112 offset:12288
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[68:73], v[32:37], a[40:43], v8, v10 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[132:133], v113 offset:28160
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[68:73], v[26:31], a[36:39], v8, v10 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[68:73], v[20:25], a[32:35], v8, v10 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[68:71], v112 offset:13312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[74:79], v[20:25], a[64:67], v8, v10 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[72:73], v113 offset:28672
	buffer_load_dwordx4 v1, s[4:7], s47 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[74:79], v[26:31], a[68:71], v8, v10 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[74:79], v[32:37], a[72:75], v8, v10 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[134:137], v112 offset:14336
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[74:79], v[38:43], a[76:79], v8, v10 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[138:139], v113 offset:29184
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[74:79], v[44:49], a[80:83], v8, v11 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s48, s13, 0x13000
	s_mov_b32 m0, s30
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[74:79], v[50:55], a[84:87], v8, v11 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s48 offen lds
	ds_read_b128 v[140:143], v112 offset:15360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[74:79], v[56:61], a[88:91], v8, v11 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[144:145], v113 offset:29696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[74:79], v[62:67], a[92:95], v8, v11 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[80:85], v[62:67], a[124:127], v8, v11 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[74:77], v112 offset:16384
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[80:85], v[56:61], a[120:123], v8, v11 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[78:79], v113 offset:30208
	buffer_load_dwordx4 v1, s[4:7], s48 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[80:85], v[50:55], a[116:119], v8, v11 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[80:85], v[44:49], a[112:115], v8, v11 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[146:149], v112 offset:17408
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[80:85], v[38:43], a[108:111], v8, v10 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[150:151], v113 offset:30720
	s_mov_b32 m0, s31
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[80:85], v[32:37], a[104:107], v8, v10 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s47 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[80:85], v[26:31], a[100:103], v8, v10 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[152:155], v112 offset:18432
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[80:85], v[20:25], a[96:99], v8, v10 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[156:157], v113 offset:31232
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[86:91], v[20:25], a[128:131], v9, v10 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[86:91], v[26:31], a[132:135], v9, v10 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[80:83], v6 offset:52224
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[86:91], v[32:37], a[136:139], v9, v10 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[84:85], v111 offset:2560
	buffer_load_dwordx4 v1, s[20:23], s47 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[86:91], v[38:43], a[140:143], v9, v10 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[86:91], v[44:49], a[144:147], v9, v11 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[158:161], v6 offset:53248
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[86:91], v[50:55], a[148:151], v9, v11 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[162:163], v111 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[86:91], v[56:61], a[152:155], v9, v11 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s47 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[86:91], v[62:67], a[156:159], v9, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[86:89], v6 offset:54272
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[92:97], v[62:67], a[188:191], v9, v11 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[90:91], v111 offset:3584
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[92:97], v[56:61], a[184:187], v9, v11 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[92:97], v[50:55], a[180:183], v9, v11 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s47 offen offset:3072 lds
	ds_read_b128 v[164:167], v6 offset:55296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[92:97], v[44:49], a[176:179], v9, v11 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[168:169], v111 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[92:97], v[38:43], a[172:175], v9, v10 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[92:97], v[32:37], a[168:171], v9, v10 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[170:173], v6 offset:56320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[92:97], v[26:31], a[164:167], v9, v10 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_mov_b32 m0, s32
	ds_read_b64 v[174:175], v111 offset:4608
	buffer_load_dwordx4 v1, s[20:23], s48 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[92:97], v[20:25], a[160:163], v9, v10 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[98:103], v[20:25], a[192:195], v9, v10 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[92:95], v6 offset:57344
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[98:103], v[26:31], a[196:199], v9, v10 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[96:97], v111 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[98:103], v[32:37], a[200:203], v9, v10 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[98:103], v[38:43], a[204:207], v9, v10 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s48 offen offset:1024 lds
	ds_read_b128 v[176:179], v6 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[98:103], v[44:49], a[208:211], v9, v11 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[180:181], v111 offset:5632
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[98:103], v[50:55], a[212:215], v9, v11 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[98:103], v[56:61], a[216:219], v9, v11 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_i32 s46, s3, s46
	s_mov_b32 m0, s33
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[98:103], v[62:67], a[220:223], v9, v11 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dword v2, s[24:27], s46 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[104:109], v[62:67], a[252:255], v9, v11 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[104:109], v[56:61], a[248:251], v9, v11 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[104:109], v[50:55], a[244:247], v9, v11 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s14, s3, s14
	s_add_i32 s46, s14, 0xc000
	s_mov_b32 m0, s28
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[104:109], v[44:49], a[240:243], v9, v11 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dword v2, s[8:11], s46 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[104:109], v[38:43], a[236:239], v9, v10 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[104:109], v[32:37], a[232:235], v9, v10 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[104:109], v[26:31], a[228:231], v9, v10 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[104:109], v[20:25], a[224:227], v9, v10 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_add_i32 s46, s17, 0x80
	s_lshl_b32 s46, s46, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[122:127], v[12:17], a[0:3], v118, v120 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[122:127], v[128:133], a[4:7], v118, v120 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[122:127], v[68:73], a[8:11], v118, v120 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[8:9], v19 offset:61440
	ds_read_b64 v[10:11], v114 offset:21504
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[122:127], v[134:139], a[12:15], v118, v120 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_add_i32 s47, s13, 0x18000
	s_mov_b32 m0, s36
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[122:127], v[140:145], a[16:19], v118, v121 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s47 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[122:127], v[74:79], a[20:23], v118, v121 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[122:127], v[146:151], a[24:27], v118, v121 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[20:23], v115 offset:36864
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[122:127], v[152:157], a[28:31], v118, v121 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[24:25], v111 offset:53248
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[80:85], v[152:157], a[60:63], v118, v121 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s47 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[80:85], v[146:151], a[56:59], v118, v121 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[26:29], v112 offset:62464
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[80:85], v[74:79], a[52:55], v118, v121 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[30:31], v116 offset:13312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[80:85], v[140:145], a[48:51], v118, v121 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[80:85], v[134:139], a[44:47], v118, v120 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s47 offen offset:2048 lds
	ds_read_b128 v[32:35], v112 offset:63488
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[80:85], v[68:73], a[40:43], v118, v120 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[36:37], v116 offset:13824
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[80:85], v[128:133], a[36:39], v118, v120 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[80:85], v[12:17], a[32:35], v118, v120 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[38:41], v112 offset:64512
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[158:163], v[12:17], a[64:67], v118, v120 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[42:43], v116 offset:14336
	buffer_load_dwordx4 v1, s[4:7], s47 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[158:163], v[128:133], a[68:71], v118, v120 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[158:163], v[68:73], a[72:75], v118, v120 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[44:47], v117
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[158:163], v[134:139], a[76:79], v118, v120 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[48:49], v116 offset:14848
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[158:163], v[140:145], a[80:83], v118, v121 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s48, s13, 0x19000
	s_mov_b32 m0, s37
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[158:163], v[74:79], a[84:87], v118, v121 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s48 offen lds
	ds_read_b128 v[50:53], v117 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[158:163], v[146:151], a[88:91], v118, v121 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[54:55], v116 offset:15360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[158:163], v[152:157], a[92:95], v118, v121 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[86:91], v[152:157], a[124:127], v118, v121 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[56:59], v117 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[86:91], v[146:151], a[120:123], v118, v121 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[60:61], v116 offset:15872
	buffer_load_dwordx4 v1, s[4:7], s48 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[86:91], v[74:79], a[116:119], v118, v121 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[86:91], v[140:145], a[112:115], v118, v121 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[62:65], v117 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[86:91], v[134:139], a[108:111], v118, v120 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[66:67], v116 offset:16384
	s_mov_b32 m0, s39
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[86:91], v[68:73], a[104:107], v118, v120 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s47 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[86:91], v[128:133], a[100:103], v118, v120 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[80:83], v117 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[86:91], v[12:17], a[96:99], v118, v120 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[84:85], v116 offset:16896
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[164:169], v[12:17], a[128:131], v119, v120 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[164:169], v[128:133], a[132:135], v119, v120 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[86:89], v115 offset:37888
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[164:169], v[68:73], a[136:139], v119, v120 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[90:91], v111 offset:53760
	buffer_load_dwordx4 v1, s[20:23], s47 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[164:169], v[134:139], a[140:143], v119, v120 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[164:169], v[140:145], a[144:147], v119, v121 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[98:101], v115 offset:38912
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[164:169], v[74:79], a[148:151], v119, v121 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[102:103], v111 offset:54272
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[164:169], v[146:151], a[152:155], v119, v121 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s47 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[164:169], v[152:157], a[156:159], v119, v121 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[104:107], v115 offset:39936
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[170:175], v[152:157], a[188:191], v119, v121 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[108:109], v111 offset:54784
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[170:175], v[146:151], a[184:187], v119, v121 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[170:175], v[74:79], a[180:183], v119, v121 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s47 offen offset:3072 lds
	ds_read_b128 v[122:125], v115 offset:40960
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[170:175], v[140:145], a[176:179], v119, v121 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[126:127], v111 offset:55296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[170:175], v[134:139], a[172:175], v119, v120 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[170:175], v[68:73], a[168:171], v119, v120 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[158:161], v115 offset:41984
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[170:175], v[128:133], a[164:167], v119, v120 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_mov_b32 m0, s35
	ds_read_b64 v[162:163], v111 offset:55808
	buffer_load_dwordx4 v1, s[20:23], s48 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[170:175], v[12:17], a[160:163], v119, v120 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[92:97], v[12:17], a[192:195], v119, v120 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[164:167], v115 offset:43008
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[92:97], v[128:133], a[196:199], v119, v120 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[168:169], v111 offset:56320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[92:97], v[68:73], a[200:203], v119, v120 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[92:97], v[134:139], a[204:207], v119, v120 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s48 offen offset:1024 lds
	ds_read_b128 v[170:173], v115 offset:44032
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[92:97], v[140:145], a[208:211], v119, v121 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[174:175], v111 offset:56832
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[92:97], v[74:79], a[212:215], v119, v121 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[92:97], v[146:151], a[216:219], v119, v121 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_i32 s46, s3, s46
	s_mov_b32 m0, s38
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[92:97], v[152:157], a[220:223], v119, v121 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dword v2, s[24:27], s46 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[176:181], v[152:157], a[252:255], v119, v121 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[176:181], v[146:151], a[248:251], v119, v121 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[176:181], v[74:79], a[244:247], v119, v121 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s46, s14, 0x10000
	s_mov_b32 m0, s34
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[176:181], v[140:145], a[240:243], v119, v121 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dword v2, s[8:11], s46 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[176:181], v[134:139], a[236:239], v119, v120 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[176:181], v[68:73], a[232:235], v119, v120 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[176:181], v[128:133], a[228:231], v119, v120 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[176:181], v[12:17], a[224:227], v119, v120 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_add_i32 s46, s17, 0xa0
	s_lshl_b32 s46, s46, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:25], v[26:31], a[0:3], v8, v10 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[20:25], v[32:37], a[4:7], v8, v10 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[20:25], v[38:43], a[8:11], v8, v10 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[12:13], v5 offset:24576
	ds_read_b64 v[14:15], v3 offset:50176
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[20:25], v[44:49], a[12:15], v8, v10 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_add_i32 s47, s13, 0x1e000
	s_mov_b32 m0, s42
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[20:25], v[50:55], a[16:19], v8, v11 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s47 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[20:25], v[56:61], a[20:23], v8, v11 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[20:25], v[62:67], a[24:27], v8, v11 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[68:71], v6
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[20:25], v[80:85], a[28:31], v8, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[72:73], v4 offset:16384
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[86:91], v[80:85], a[60:63], v8, v11 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s47 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[86:91], v[62:67], a[56:59], v8, v11 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[20:23], v18 offset:25600
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[86:91], v[56:61], a[52:55], v8, v11 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[24:25], v7 offset:41984
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[86:91], v[50:55], a[48:51], v8, v11 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[86:91], v[44:49], a[44:47], v8, v10 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s47 offen offset:2048 lds
	ds_read_b128 v[74:77], v18 offset:26624
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[86:91], v[38:43], a[40:43], v8, v10 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[78:79], v7 offset:42496
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[86:91], v[32:37], a[36:39], v8, v10 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[86:91], v[26:31], a[32:35], v8, v10 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[86:89], v18 offset:27648
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[98:103], v[26:31], a[64:67], v8, v10 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[90:91], v7 offset:43008
	buffer_load_dwordx4 v1, s[4:7], s47 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[98:103], v[32:37], a[68:71], v8, v10 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[98:103], v[38:43], a[72:75], v8, v10 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[92:95], v18 offset:28672
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[98:103], v[44:49], a[76:79], v8, v10 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[96:97], v7 offset:43520
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[98:103], v[50:55], a[80:83], v8, v11 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s48, s13, 0x1f000
	s_mov_b32 m0, s43
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[98:103], v[56:61], a[84:87], v8, v11 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s48 offen lds
	ds_read_b128 v[128:131], v18 offset:29696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[98:103], v[62:67], a[88:91], v8, v11 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[132:133], v7 offset:44032
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[98:103], v[80:85], a[92:95], v8, v11 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[104:109], v[80:85], a[124:127], v8, v11 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[98:101], v18 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[104:109], v[62:67], a[120:123], v8, v11 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[102:103], v7 offset:44544
	buffer_load_dwordx4 v1, s[4:7], s48 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[104:109], v[56:61], a[116:119], v8, v11 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[104:109], v[50:55], a[112:115], v8, v11 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[134:137], v18 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[104:109], v[44:49], a[108:111], v8, v10 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[138:139], v7 offset:45056
	s_mov_b32 m0, s45
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[104:109], v[38:43], a[104:107], v8, v10 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s47 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[104:109], v[32:37], a[100:103], v8, v10 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[140:143], v18 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[104:109], v[26:31], a[96:99], v8, v10 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[144:145], v7 offset:45568
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[122:127], v[26:31], a[128:131], v9, v10 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[122:127], v[32:37], a[132:135], v9, v10 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[104:107], v6 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[122:127], v[38:43], a[136:139], v9, v10 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[108:109], v4 offset:16896
	buffer_load_dwordx4 v1, s[20:23], s47 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[122:127], v[44:49], a[140:143], v9, v10 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[122:127], v[50:55], a[144:147], v9, v11 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[146:149], v6 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[122:127], v[56:61], a[148:151], v9, v11 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[150:151], v4 offset:17408
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[122:127], v[62:67], a[152:155], v9, v11 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s47 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[122:127], v[80:85], a[156:159], v9, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[118:121], v6 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[158:163], v[80:85], a[188:191], v9, v11 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[122:123], v4 offset:17920
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[158:163], v[62:67], a[184:187], v9, v11 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[158:163], v[56:61], a[180:183], v9, v11 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s47 offen offset:3072 lds
	ds_read_b128 v[152:155], v6 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[158:163], v[50:55], a[176:179], v9, v11 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[156:157], v4 offset:18432
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[158:163], v[44:49], a[172:175], v9, v10 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[158:163], v[38:43], a[168:171], v9, v10 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[176:179], v6 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[158:163], v[32:37], a[164:167], v9, v10 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_mov_b32 m0, s41
	ds_read_b64 v[180:181], v4 offset:18944
	buffer_load_dwordx4 v1, s[20:23], s48 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[158:163], v[26:31], a[160:163], v9, v10 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[164:169], v[26:31], a[192:195], v9, v10 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[158:161], v6 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[164:169], v[32:37], a[196:199], v9, v10 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[162:163], v4 offset:19456
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[164:169], v[38:43], a[200:203], v9, v10 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[164:169], v[44:49], a[204:207], v9, v10 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s48 offen offset:1024 lds
	ds_read_b128 v[182:185], v6 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[164:169], v[50:55], a[208:211], v9, v11 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[186:187], v4 offset:19968
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[164:169], v[56:61], a[212:215], v9, v11 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[164:169], v[62:67], a[216:219], v9, v11 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_i32 s46, s3, s46
	s_mov_b32 m0, s44
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[164:169], v[80:85], a[220:223], v9, v11 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dword v2, s[24:27], s46 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[170:175], v[80:85], a[252:255], v9, v11 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[170:175], v[62:67], a[248:251], v9, v11 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[170:175], v[56:61], a[244:247], v9, v11 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s46, s14, 0x14000
	s_mov_b32 m0, s40
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[170:175], v[50:55], a[240:243], v9, v11 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dword v2, s[8:11], s46 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[170:175], v[44:49], a[236:239], v9, v10 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[170:175], v[38:43], a[232:235], v9, v10 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[170:175], v[32:37], a[228:231], v9, v10 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[170:175], v[26:31], a[224:227], v9, v10 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_add_i32 s46, s17, 0xc0
	s_lshl_b32 s46, s46, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[68:73], v[20:25], a[0:3], v12, v14 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[68:73], v[74:79], a[4:7], v12, v14 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[68:73], v[86:91], a[8:11], v12, v14 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[8:9], v19 offset:10240
	ds_read_b64 v[10:11], v110 offset:35840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[68:73], v[92:97], a[12:15], v12, v14 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_add_i32 s47, s13, 0x24000
	s_mov_b32 m0, s2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[68:73], v[128:133], a[16:19], v12, v15 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s47 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[68:73], v[98:103], a[20:23], v12, v15 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[68:73], v[134:139], a[24:27], v12, v15 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[26:29], v6 offset:51200
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[68:73], v[140:145], a[28:31], v12, v15 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[30:31], v111 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[104:109], v[140:145], a[60:63], v12, v15 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s47 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[104:109], v[134:139], a[56:59], v12, v15 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[32:35], v112 offset:11264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[104:109], v[98:103], a[52:55], v12, v15 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[36:37], v113 offset:27648
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[104:109], v[128:133], a[48:51], v12, v15 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[104:109], v[92:97], a[44:47], v12, v14 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s47 offen offset:2048 lds
	ds_read_b128 v[38:41], v112 offset:12288
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[104:109], v[86:91], a[40:43], v12, v14 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[42:43], v113 offset:28160
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[104:109], v[74:79], a[36:39], v12, v14 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[104:109], v[20:25], a[32:35], v12, v14 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[44:47], v112 offset:13312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[146:151], v[20:25], a[64:67], v12, v14 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[48:49], v113 offset:28672
	buffer_load_dwordx4 v1, s[4:7], s47 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[146:151], v[74:79], a[68:71], v12, v14 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[146:151], v[86:91], a[72:75], v12, v14 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[50:53], v112 offset:14336
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[146:151], v[92:97], a[76:79], v12, v14 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[54:55], v113 offset:29184
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[146:151], v[128:133], a[80:83], v12, v15 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s48, s13, 0x25000
	s_mov_b32 m0, s30
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[146:151], v[98:103], a[84:87], v12, v15 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s48 offen lds
	ds_read_b128 v[56:59], v112 offset:15360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[146:151], v[134:139], a[88:91], v12, v15 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[60:61], v113 offset:29696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[146:151], v[140:145], a[92:95], v12, v15 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[118:123], v[140:145], a[124:127], v12, v15 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[62:65], v112 offset:16384
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[118:123], v[134:139], a[120:123], v12, v15 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[66:67], v113 offset:30208
	buffer_load_dwordx4 v1, s[4:7], s48 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[118:123], v[98:103], a[116:119], v12, v15 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[118:123], v[128:133], a[112:115], v12, v15 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[68:71], v112 offset:17408
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[118:123], v[92:97], a[108:111], v12, v14 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[72:73], v113 offset:30720
	s_mov_b32 m0, s31
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[118:123], v[86:91], a[104:107], v12, v14 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s47 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[118:123], v[74:79], a[100:103], v12, v14 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[80:83], v112 offset:18432
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[118:123], v[20:25], a[96:99], v12, v14 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[84:85], v113 offset:31232
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[152:157], v[20:25], a[128:131], v13, v14 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[152:157], v[74:79], a[132:135], v13, v14 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[104:107], v6 offset:52224
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[152:157], v[86:91], a[136:139], v13, v14 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[108:109], v111 offset:2560
	buffer_load_dwordx4 v1, s[20:23], s47 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[152:157], v[92:97], a[140:143], v13, v14 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[152:157], v[128:133], a[144:147], v13, v15 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[118:121], v6 offset:53248
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[152:157], v[98:103], a[148:151], v13, v15 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[122:123], v111 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[152:157], v[134:139], a[152:155], v13, v15 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s47 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[152:157], v[140:145], a[156:159], v13, v15 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[146:149], v6 offset:54272
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[176:181], v[140:145], a[188:191], v13, v15 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[150:151], v111 offset:3584
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[176:181], v[134:139], a[184:187], v13, v15 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[176:181], v[98:103], a[180:183], v13, v15 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s47 offen offset:3072 lds
	ds_read_b128 v[152:155], v6 offset:55296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[176:181], v[128:133], a[176:179], v13, v15 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[156:157], v111 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[176:181], v[92:97], a[172:175], v13, v14 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[176:181], v[86:91], a[168:171], v13, v14 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[164:167], v6 offset:56320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[176:181], v[74:79], a[164:167], v13, v14 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_mov_b32 m0, s32
	ds_read_b64 v[168:169], v111 offset:4608
	buffer_load_dwordx4 v1, s[20:23], s48 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[176:181], v[20:25], a[160:163], v13, v14 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[158:163], v[20:25], a[192:195], v13, v14 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[170:173], v6 offset:57344
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[158:163], v[74:79], a[196:199], v13, v14 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[174:175], v111 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[158:163], v[86:91], a[200:203], v13, v14 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[158:163], v[92:97], a[204:207], v13, v14 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s48 offen offset:1024 lds
	ds_read_b128 v[176:179], v6 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[158:163], v[128:133], a[208:211], v13, v15 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[180:181], v111 offset:5632
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[158:163], v[98:103], a[212:215], v13, v15 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[158:163], v[134:139], a[216:219], v13, v15 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_i32 s46, s3, s46
	s_mov_b32 m0, s33
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[158:163], v[140:145], a[220:223], v13, v15 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dword v2, s[24:27], s46 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[182:187], v[140:145], a[252:255], v13, v15 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[182:187], v[134:139], a[248:251], v13, v15 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[182:187], v[98:103], a[244:247], v13, v15 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s46, s14, 0x18000
	s_mov_b32 m0, s28
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[182:187], v[128:133], a[240:243], v13, v15 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dword v2, s[8:11], s46 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[182:187], v[92:97], a[236:239], v13, v14 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[182:187], v[86:91], a[232:235], v13, v14 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[182:187], v[74:79], a[228:231], v13, v14 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[182:187], v[20:25], a[224:227], v13, v14 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_add_i32 s46, s17, 0xe0
	s_lshl_b32 s46, s46, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:31], v[32:37], a[0:3], v8, v10 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[26:31], v[38:43], a[4:7], v8, v10 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[26:31], v[44:49], a[8:11], v8, v10 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[12:13], v19 offset:61440
	ds_read_b64 v[14:15], v114 offset:21504
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[26:31], v[50:55], a[12:15], v8, v10 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_add_i32 s47, s13, 0x2a000
	s_mov_b32 m0, s36
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[26:31], v[56:61], a[16:19], v8, v11 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s47 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[26:31], v[62:67], a[20:23], v8, v11 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[26:31], v[68:73], a[24:27], v8, v11 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[20:23], v115 offset:36864
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[26:31], v[80:85], a[28:31], v8, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[24:25], v111 offset:53248
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[104:109], v[80:85], a[60:63], v8, v11 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s47 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[104:109], v[68:73], a[56:59], v8, v11 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[26:29], v112 offset:62464
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[104:109], v[62:67], a[52:55], v8, v11 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[30:31], v116 offset:13312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[104:109], v[56:61], a[48:51], v8, v11 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[104:109], v[50:55], a[44:47], v8, v10 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s47 offen offset:2048 lds
	ds_read_b128 v[74:77], v112 offset:63488
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[104:109], v[44:49], a[40:43], v8, v10 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[78:79], v116 offset:13824
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[104:109], v[38:43], a[36:39], v8, v10 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[104:109], v[32:37], a[32:35], v8, v10 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[86:89], v112 offset:64512
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[118:123], v[32:37], a[64:67], v8, v10 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[90:91], v116 offset:14336
	buffer_load_dwordx4 v1, s[4:7], s47 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[118:123], v[38:43], a[68:71], v8, v10 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[118:123], v[44:49], a[72:75], v8, v10 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[92:95], v117
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[118:123], v[50:55], a[76:79], v8, v10 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[96:97], v116 offset:14848
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[118:123], v[56:61], a[80:83], v8, v11 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s48, s13, 0x2b000
	s_mov_b32 m0, s37
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[118:123], v[62:67], a[84:87], v8, v11 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s48 offen lds
	ds_read_b128 v[98:101], v117 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[118:123], v[68:73], a[88:91], v8, v11 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[102:103], v116 offset:15360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[118:123], v[80:85], a[92:95], v8, v11 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[146:151], v[80:85], a[124:127], v8, v11 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[104:107], v117 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[146:151], v[68:73], a[120:123], v8, v11 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[108:109], v116 offset:15872
	buffer_load_dwordx4 v1, s[4:7], s48 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[146:151], v[62:67], a[116:119], v8, v11 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[146:151], v[56:61], a[112:115], v8, v11 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[118:121], v117 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[146:151], v[50:55], a[108:111], v8, v10 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[122:123], v116 offset:16384
	s_mov_b32 m0, s39
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[146:151], v[44:49], a[104:107], v8, v10 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s47 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[146:151], v[38:43], a[100:103], v8, v10 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[124:127], v117 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[146:151], v[32:37], a[96:99], v8, v10 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[128:129], v116 offset:16896
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[152:157], v[32:37], a[128:131], v9, v10 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[152:157], v[38:43], a[132:135], v9, v10 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[130:133], v115 offset:37888
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[152:157], v[44:49], a[136:139], v9, v10 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[134:135], v111 offset:53760
	buffer_load_dwordx4 v1, s[20:23], s47 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[152:157], v[50:55], a[140:143], v9, v10 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[152:157], v[56:61], a[144:147], v9, v11 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[136:139], v115 offset:38912
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[152:157], v[62:67], a[148:151], v9, v11 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[140:141], v111 offset:54272
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[152:157], v[68:73], a[152:155], v9, v11 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s47 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[152:157], v[80:85], a[156:159], v9, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[142:145], v115 offset:39936
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[164:169], v[80:85], a[188:191], v9, v11 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[146:147], v111 offset:54784
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[164:169], v[68:73], a[184:187], v9, v11 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[164:169], v[62:67], a[180:183], v9, v11 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s47 offen offset:3072 lds
	ds_read_b128 v[148:151], v115 offset:40960
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[164:169], v[56:61], a[176:179], v9, v11 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[152:153], v111 offset:55296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[164:169], v[50:55], a[172:175], v9, v10 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[164:169], v[44:49], a[168:171], v9, v10 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[154:157], v115 offset:41984
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[164:169], v[38:43], a[164:167], v9, v10 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_mov_b32 m0, s35
	ds_read_b64 v[158:159], v111 offset:55808
	buffer_load_dwordx4 v1, s[20:23], s48 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[164:169], v[32:37], a[160:163], v9, v10 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[170:175], v[32:37], a[192:195], v9, v10 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[160:163], v115 offset:43008
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[170:175], v[38:43], a[196:199], v9, v10 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[164:165], v111 offset:56320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[170:175], v[44:49], a[200:203], v9, v10 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[170:175], v[50:55], a[204:207], v9, v10 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s48 offen offset:1024 lds
	ds_read_b128 v[182:185], v115 offset:44032
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[170:175], v[56:61], a[208:211], v9, v11 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[186:187], v111 offset:56832
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[170:175], v[62:67], a[212:215], v9, v11 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[170:175], v[68:73], a[216:219], v9, v11 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_i32 s46, s3, s46
	s_mov_b32 m0, s38
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[170:175], v[80:85], a[220:223], v9, v11 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dword v2, s[24:27], s46 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[176:181], v[80:85], a[252:255], v9, v11 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[176:181], v[68:73], a[248:251], v9, v11 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[176:181], v[62:67], a[244:247], v9, v11 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s46, s14, 0x1c000
	s_mov_b32 m0, s34
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[176:181], v[56:61], a[240:243], v9, v11 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dword v2, s[8:11], s46 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[176:181], v[50:55], a[236:239], v9, v10 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[176:181], v[44:49], a[232:235], v9, v10 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[176:181], v[38:43], a[228:231], v9, v10 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[176:181], v[32:37], a[224:227], v9, v10 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_add_i32 s46, s17, 0x100
	s_lshl_b32 s46, s46, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:25], v[26:31], a[0:3], v12, v14 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[20:25], v[74:79], a[4:7], v12, v14 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[20:25], v[86:91], a[8:11], v12, v14 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[8:9], v5 offset:24576
	ds_read_b64 v[10:11], v3 offset:50176
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[20:25], v[92:97], a[12:15], v12, v14 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_add_i32 s47, s13, 0x30000
	s_mov_b32 m0, s42
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[20:25], v[98:103], a[16:19], v12, v15 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s47 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[20:25], v[104:109], a[20:23], v12, v15 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[20:25], v[118:123], a[24:27], v12, v15 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[32:35], v6
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[20:25], v[124:129], a[28:31], v12, v15 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[36:37], v4 offset:16384
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[130:135], v[124:129], a[60:63], v12, v15 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s47 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[130:135], v[118:123], a[56:59], v12, v15 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[20:23], v18 offset:25600
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[130:135], v[104:109], a[52:55], v12, v15 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[24:25], v7 offset:41984
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[130:135], v[98:103], a[48:51], v12, v15 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[130:135], v[92:97], a[44:47], v12, v14 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s47 offen offset:2048 lds
	ds_read_b128 v[38:41], v18 offset:26624
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[130:135], v[86:91], a[40:43], v12, v14 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[42:43], v7 offset:42496
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[130:135], v[74:79], a[36:39], v12, v14 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[130:135], v[26:31], a[32:35], v12, v14 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[44:47], v18 offset:27648
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[136:141], v[26:31], a[64:67], v12, v14 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[48:49], v7 offset:43008
	buffer_load_dwordx4 v1, s[4:7], s47 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[136:141], v[74:79], a[68:71], v12, v14 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[136:141], v[86:91], a[72:75], v12, v14 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[50:53], v18 offset:28672
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[136:141], v[92:97], a[76:79], v12, v14 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[54:55], v7 offset:43520
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[136:141], v[98:103], a[80:83], v12, v15 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s48, s13, 0x31000
	s_mov_b32 m0, s43
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[136:141], v[104:109], a[84:87], v12, v15 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s48 offen lds
	ds_read_b128 v[56:59], v18 offset:29696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[136:141], v[118:123], a[88:91], v12, v15 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[60:61], v7 offset:44032
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[136:141], v[124:129], a[92:95], v12, v15 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[142:147], v[124:129], a[124:127], v12, v15 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[62:65], v18 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[142:147], v[118:123], a[120:123], v12, v15 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[66:67], v7 offset:44544
	buffer_load_dwordx4 v1, s[4:7], s48 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[142:147], v[104:109], a[116:119], v12, v15 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[142:147], v[98:103], a[112:115], v12, v15 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[68:71], v18 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[142:147], v[92:97], a[108:111], v12, v14 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[72:73], v7 offset:45056
	s_mov_b32 m0, s45
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[142:147], v[86:91], a[104:107], v12, v14 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s47 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[142:147], v[74:79], a[100:103], v12, v14 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[80:83], v18 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[142:147], v[26:31], a[96:99], v12, v14 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[84:85], v7 offset:45568
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[148:153], v[26:31], a[128:131], v13, v14 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[148:153], v[74:79], a[132:135], v13, v14 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[130:133], v6 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[148:153], v[86:91], a[136:139], v13, v14 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[134:135], v4 offset:16896
	buffer_load_dwordx4 v1, s[20:23], s47 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[148:153], v[92:97], a[140:143], v13, v14 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[148:153], v[98:103], a[144:147], v13, v15 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[136:139], v6 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[148:153], v[104:109], a[148:151], v13, v15 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[140:141], v4 offset:17408
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[148:153], v[118:123], a[152:155], v13, v15 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s47 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[148:153], v[124:129], a[156:159], v13, v15 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[142:145], v6 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[154:159], v[124:129], a[188:191], v13, v15 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[146:147], v4 offset:17920
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[154:159], v[118:123], a[184:187], v13, v15 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[154:159], v[104:109], a[180:183], v13, v15 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s47 offen offset:3072 lds
	ds_read_b128 v[148:151], v6 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[154:159], v[98:103], a[176:179], v13, v15 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[152:153], v4 offset:18432
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[154:159], v[92:97], a[172:175], v13, v14 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[154:159], v[86:91], a[168:171], v13, v14 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[166:169], v6 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[154:159], v[74:79], a[164:167], v13, v14 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_mov_b32 m0, s41
	ds_read_b64 v[170:171], v4 offset:18944
	buffer_load_dwordx4 v1, s[20:23], s48 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[154:159], v[26:31], a[160:163], v13, v14 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[160:165], v[26:31], a[192:195], v13, v14 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[154:157], v6 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[160:165], v[74:79], a[196:199], v13, v14 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[158:159], v4 offset:19456
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[160:165], v[86:91], a[200:203], v13, v14 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[160:165], v[92:97], a[204:207], v13, v14 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s48 offen offset:1024 lds
	ds_read_b128 v[172:175], v6 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[160:165], v[98:103], a[208:211], v13, v15 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[176:177], v4 offset:19968
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[160:165], v[104:109], a[212:215], v13, v15 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[160:165], v[118:123], a[216:219], v13, v15 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_i32 s46, s3, s46
	s_mov_b32 m0, s44
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[160:165], v[124:129], a[220:223], v13, v15 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dword v2, s[24:27], s46 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[182:187], v[124:129], a[252:255], v13, v15 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[182:187], v[118:123], a[248:251], v13, v15 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[182:187], v[104:109], a[244:247], v13, v15 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s46, s14, 0x20000
	s_mov_b32 m0, s40
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[182:187], v[98:103], a[240:243], v13, v15 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dword v2, s[8:11], s46 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[182:187], v[92:97], a[236:239], v13, v14 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[182:187], v[86:91], a[232:235], v13, v14 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[182:187], v[74:79], a[228:231], v13, v14 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[182:187], v[26:31], a[224:227], v13, v14 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_add_i32 s46, s17, 0x120
	s_lshl_b32 s46, s46, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[32:37], v[20:25], a[0:3], v8, v10 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[32:37], v[38:43], a[4:7], v8, v10 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[32:37], v[44:49], a[8:11], v8, v10 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[12:13], v19 offset:10240
	ds_read_b64 v[14:15], v110 offset:35840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[32:37], v[50:55], a[12:15], v8, v10 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_add_i32 s47, s13, 0x36000
	s_mov_b32 m0, s2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[32:37], v[56:61], a[16:19], v8, v11 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s47 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[32:37], v[62:67], a[20:23], v8, v11 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[32:37], v[68:73], a[24:27], v8, v11 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[26:29], v6 offset:51200
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[32:37], v[80:85], a[28:31], v8, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[30:31], v111 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[130:135], v[80:85], a[60:63], v8, v11 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s47 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[130:135], v[68:73], a[56:59], v8, v11 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[32:35], v112 offset:11264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[130:135], v[62:67], a[52:55], v8, v11 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[36:37], v113 offset:27648
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[130:135], v[56:61], a[48:51], v8, v11 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[130:135], v[50:55], a[44:47], v8, v10 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s47 offen offset:2048 lds
	ds_read_b128 v[74:77], v112 offset:12288
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[130:135], v[44:49], a[40:43], v8, v10 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[78:79], v113 offset:28160
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[130:135], v[38:43], a[36:39], v8, v10 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[130:135], v[20:25], a[32:35], v8, v10 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[86:89], v112 offset:13312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[136:141], v[20:25], a[64:67], v8, v10 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[90:91], v113 offset:28672
	buffer_load_dwordx4 v1, s[4:7], s47 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[136:141], v[38:43], a[68:71], v8, v10 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[136:141], v[44:49], a[72:75], v8, v10 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[92:95], v112 offset:14336
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[136:141], v[50:55], a[76:79], v8, v10 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[96:97], v113 offset:29184
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[136:141], v[56:61], a[80:83], v8, v11 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s48, s13, 0x37000
	s_mov_b32 m0, s30
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[136:141], v[62:67], a[84:87], v8, v11 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s48 offen lds
	ds_read_b128 v[98:101], v112 offset:15360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[136:141], v[68:73], a[88:91], v8, v11 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[102:103], v113 offset:29696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[136:141], v[80:85], a[92:95], v8, v11 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[142:147], v[80:85], a[124:127], v8, v11 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[104:107], v112 offset:16384
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[142:147], v[68:73], a[120:123], v8, v11 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[108:109], v113 offset:30208
	buffer_load_dwordx4 v1, s[4:7], s48 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[142:147], v[62:67], a[116:119], v8, v11 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[142:147], v[56:61], a[112:115], v8, v11 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[118:121], v112 offset:17408
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[142:147], v[50:55], a[108:111], v8, v10 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[122:123], v113 offset:30720
	s_mov_b32 m0, s31
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[142:147], v[44:49], a[104:107], v8, v10 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s47 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[142:147], v[38:43], a[100:103], v8, v10 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[124:127], v112 offset:18432
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[142:147], v[20:25], a[96:99], v8, v10 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[128:129], v113 offset:31232
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[148:153], v[20:25], a[128:131], v9, v10 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[148:153], v[38:43], a[132:135], v9, v10 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[130:133], v6 offset:52224
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[148:153], v[44:49], a[136:139], v9, v10 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[134:135], v111 offset:2560
	buffer_load_dwordx4 v1, s[20:23], s47 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[148:153], v[50:55], a[140:143], v9, v10 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[148:153], v[56:61], a[144:147], v9, v11 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[136:139], v6 offset:53248
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[148:153], v[62:67], a[148:151], v9, v11 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[140:141], v111 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[148:153], v[68:73], a[152:155], v9, v11 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s47 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[148:153], v[80:85], a[156:159], v9, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[142:145], v6 offset:54272
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[166:171], v[80:85], a[188:191], v9, v11 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[146:147], v111 offset:3584
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[166:171], v[68:73], a[184:187], v9, v11 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[166:171], v[62:67], a[180:183], v9, v11 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s47 offen offset:3072 lds
	ds_read_b128 v[148:151], v6 offset:55296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[166:171], v[56:61], a[176:179], v9, v11 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[152:153], v111 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[166:171], v[50:55], a[172:175], v9, v10 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[166:171], v[44:49], a[168:171], v9, v10 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[160:163], v6 offset:56320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[166:171], v[38:43], a[164:167], v9, v10 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_mov_b32 m0, s32
	ds_read_b64 v[164:165], v111 offset:4608
	buffer_load_dwordx4 v1, s[20:23], s48 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[166:171], v[20:25], a[160:163], v9, v10 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[154:159], v[20:25], a[192:195], v9, v10 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[166:169], v6 offset:57344
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[154:159], v[38:43], a[196:199], v9, v10 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[170:171], v111 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[154:159], v[44:49], a[200:203], v9, v10 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[154:159], v[50:55], a[204:207], v9, v10 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s48 offen offset:1024 lds
	ds_read_b128 v[178:181], v6 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[154:159], v[56:61], a[208:211], v9, v11 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[182:183], v111 offset:5632
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[154:159], v[62:67], a[212:215], v9, v11 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[154:159], v[68:73], a[216:219], v9, v11 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_i32 s46, s3, s46
	s_mov_b32 m0, s33
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[154:159], v[80:85], a[220:223], v9, v11 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dword v2, s[24:27], s46 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[172:177], v[80:85], a[252:255], v9, v11 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[172:177], v[68:73], a[248:251], v9, v11 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[172:177], v[62:67], a[244:247], v9, v11 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s46, s14, 0x24000
	s_mov_b32 m0, s28
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[172:177], v[56:61], a[240:243], v9, v11 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dword v2, s[8:11], s46 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[172:177], v[50:55], a[236:239], v9, v10 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[172:177], v[44:49], a[232:235], v9, v10 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[172:177], v[38:43], a[228:231], v9, v10 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[172:177], v[20:25], a[224:227], v9, v10 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_add_i32 s46, s17, 0x140
	s_lshl_b32 s46, s46, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:31], v[32:37], a[0:3], v12, v14 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[26:31], v[74:79], a[4:7], v12, v14 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[26:31], v[86:91], a[8:11], v12, v14 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[8:9], v19 offset:61440
	ds_read_b64 v[10:11], v114 offset:21504
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[26:31], v[92:97], a[12:15], v12, v14 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_add_i32 s47, s13, 0x3c000
	s_mov_b32 m0, s36
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[26:31], v[98:103], a[16:19], v12, v15 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s47 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[26:31], v[104:109], a[20:23], v12, v15 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[26:31], v[118:123], a[24:27], v12, v15 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[20:23], v115 offset:36864
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[26:31], v[124:129], a[28:31], v12, v15 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[24:25], v111 offset:53248
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[130:135], v[124:129], a[60:63], v12, v15 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s47 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[130:135], v[118:123], a[56:59], v12, v15 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[26:29], v112 offset:62464
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[130:135], v[104:109], a[52:55], v12, v15 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[30:31], v116 offset:13312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[130:135], v[98:103], a[48:51], v12, v15 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[130:135], v[92:97], a[44:47], v12, v14 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s47 offen offset:2048 lds
	ds_read_b128 v[38:41], v112 offset:63488
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[130:135], v[86:91], a[40:43], v12, v14 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[42:43], v116 offset:13824
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[130:135], v[74:79], a[36:39], v12, v14 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[130:135], v[32:37], a[32:35], v12, v14 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[44:47], v112 offset:64512
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[136:141], v[32:37], a[64:67], v12, v14 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[48:49], v116 offset:14336
	buffer_load_dwordx4 v1, s[4:7], s47 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[136:141], v[74:79], a[68:71], v12, v14 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[136:141], v[86:91], a[72:75], v12, v14 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[50:53], v117
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[136:141], v[92:97], a[76:79], v12, v14 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[54:55], v116 offset:14848
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[136:141], v[98:103], a[80:83], v12, v15 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s48, s13, 0x3d000
	s_mov_b32 m0, s37
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[136:141], v[104:109], a[84:87], v12, v15 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s48 offen lds
	ds_read_b128 v[56:59], v117 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[136:141], v[118:123], a[88:91], v12, v15 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[60:61], v116 offset:15360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[136:141], v[124:129], a[92:95], v12, v15 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[142:147], v[124:129], a[124:127], v12, v15 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[62:65], v117 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[142:147], v[118:123], a[120:123], v12, v15 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[66:67], v116 offset:15872
	buffer_load_dwordx4 v1, s[4:7], s48 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[142:147], v[104:109], a[116:119], v12, v15 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[142:147], v[98:103], a[112:115], v12, v15 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[68:71], v117 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[142:147], v[92:97], a[108:111], v12, v14 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[72:73], v116 offset:16384
	s_mov_b32 m0, s39
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[142:147], v[86:91], a[104:107], v12, v14 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s47 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[142:147], v[74:79], a[100:103], v12, v14 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[80:83], v117 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[142:147], v[32:37], a[96:99], v12, v14 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[84:85], v116 offset:16896
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[148:153], v[32:37], a[128:131], v13, v14 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[148:153], v[74:79], a[132:135], v13, v14 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[130:133], v115 offset:37888
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[148:153], v[86:91], a[136:139], v13, v14 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[134:135], v111 offset:53760
	buffer_load_dwordx4 v1, s[20:23], s47 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[148:153], v[92:97], a[140:143], v13, v14 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[148:153], v[98:103], a[144:147], v13, v15 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[136:139], v115 offset:38912
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[148:153], v[104:109], a[148:151], v13, v15 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[140:141], v111 offset:54272
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[148:153], v[118:123], a[152:155], v13, v15 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s47 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[148:153], v[124:129], a[156:159], v13, v15 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[142:145], v115 offset:39936
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[160:165], v[124:129], a[188:191], v13, v15 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[146:147], v111 offset:54784
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[160:165], v[118:123], a[184:187], v13, v15 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[160:165], v[104:109], a[180:183], v13, v15 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s47 offen offset:3072 lds
	ds_read_b128 v[148:151], v115 offset:40960
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[160:165], v[98:103], a[176:179], v13, v15 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[152:153], v111 offset:55296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[160:165], v[92:97], a[172:175], v13, v14 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[160:165], v[86:91], a[168:171], v13, v14 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[154:157], v115 offset:41984
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[160:165], v[74:79], a[164:167], v13, v14 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_mov_b32 m0, s35
	ds_read_b64 v[158:159], v111 offset:55808
	buffer_load_dwordx4 v1, s[20:23], s48 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[160:165], v[32:37], a[160:163], v13, v14 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[166:171], v[32:37], a[192:195], v13, v14 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[160:163], v115 offset:43008
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[166:171], v[74:79], a[196:199], v13, v14 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[164:165], v111 offset:56320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[166:171], v[86:91], a[200:203], v13, v14 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[166:171], v[92:97], a[204:207], v13, v14 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s48 offen offset:1024 lds
	ds_read_b128 v[172:175], v115 offset:44032
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[166:171], v[98:103], a[208:211], v13, v15 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[176:177], v111 offset:56832
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[166:171], v[104:109], a[212:215], v13, v15 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[166:171], v[118:123], a[216:219], v13, v15 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_i32 s46, s3, s46
	s_mov_b32 m0, s38
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[166:171], v[124:129], a[220:223], v13, v15 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dword v2, s[24:27], s46 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[178:183], v[124:129], a[252:255], v13, v15 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[178:183], v[118:123], a[248:251], v13, v15 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[178:183], v[104:109], a[244:247], v13, v15 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s46, s14, 0x28000
	s_mov_b32 m0, s34
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[178:183], v[98:103], a[240:243], v13, v15 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dword v2, s[8:11], s46 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[178:183], v[92:97], a[236:239], v13, v14 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[178:183], v[86:91], a[232:235], v13, v14 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[178:183], v[74:79], a[228:231], v13, v14 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[178:183], v[32:37], a[224:227], v13, v14 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_add_i32 s46, s17, 0x160
	s_lshl_b32 s46, s46, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:25], v[26:31], a[0:3], v8, v10 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[20:25], v[38:43], a[4:7], v8, v10 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[20:25], v[44:49], a[8:11], v8, v10 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[12:13], v5 offset:24576
	ds_read_b64 v[14:15], v3 offset:50176
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[20:25], v[50:55], a[12:15], v8, v10 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_add_i32 s47, s13, 0x42000
	s_mov_b32 m0, s42
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[20:25], v[56:61], a[16:19], v8, v11 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s47 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[20:25], v[62:67], a[20:23], v8, v11 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[20:25], v[68:73], a[24:27], v8, v11 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[32:35], v6
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[20:25], v[80:85], a[28:31], v8, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[36:37], v4 offset:16384
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[130:135], v[80:85], a[60:63], v8, v11 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s47 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[130:135], v[68:73], a[56:59], v8, v11 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[20:23], v18 offset:25600
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[130:135], v[62:67], a[52:55], v8, v11 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[24:25], v7 offset:41984
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[130:135], v[56:61], a[48:51], v8, v11 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[130:135], v[50:55], a[44:47], v8, v10 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s47 offen offset:2048 lds
	ds_read_b128 v[74:77], v18 offset:26624
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[130:135], v[44:49], a[40:43], v8, v10 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[78:79], v7 offset:42496
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[130:135], v[38:43], a[36:39], v8, v10 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[130:135], v[26:31], a[32:35], v8, v10 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[86:89], v18 offset:27648
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[136:141], v[26:31], a[64:67], v8, v10 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[90:91], v7 offset:43008
	buffer_load_dwordx4 v1, s[4:7], s47 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[136:141], v[38:43], a[68:71], v8, v10 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[136:141], v[44:49], a[72:75], v8, v10 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[92:95], v18 offset:28672
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[136:141], v[50:55], a[76:79], v8, v10 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[96:97], v7 offset:43520
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[136:141], v[56:61], a[80:83], v8, v11 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s48, s13, 0x43000
	s_mov_b32 m0, s43
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[136:141], v[62:67], a[84:87], v8, v11 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s48 offen lds
	ds_read_b128 v[98:101], v18 offset:29696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[136:141], v[68:73], a[88:91], v8, v11 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[102:103], v7 offset:44032
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[136:141], v[80:85], a[92:95], v8, v11 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[142:147], v[80:85], a[124:127], v8, v11 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[104:107], v18 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[142:147], v[68:73], a[120:123], v8, v11 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[108:109], v7 offset:44544
	buffer_load_dwordx4 v1, s[4:7], s48 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[142:147], v[62:67], a[116:119], v8, v11 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[142:147], v[56:61], a[112:115], v8, v11 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[118:121], v18 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[142:147], v[50:55], a[108:111], v8, v10 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[122:123], v7 offset:45056
	s_mov_b32 m0, s45
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[142:147], v[44:49], a[104:107], v8, v10 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s47 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[142:147], v[38:43], a[100:103], v8, v10 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[124:127], v18 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[142:147], v[26:31], a[96:99], v8, v10 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[128:129], v7 offset:45568
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[148:153], v[26:31], a[128:131], v9, v10 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[148:153], v[38:43], a[132:135], v9, v10 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[130:133], v6 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[148:153], v[44:49], a[136:139], v9, v10 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[134:135], v4 offset:16896
	buffer_load_dwordx4 v1, s[20:23], s47 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[148:153], v[50:55], a[140:143], v9, v10 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[148:153], v[56:61], a[144:147], v9, v11 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[136:139], v6 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[148:153], v[62:67], a[148:151], v9, v11 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[140:141], v4 offset:17408
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[148:153], v[68:73], a[152:155], v9, v11 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s47 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[148:153], v[80:85], a[156:159], v9, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[142:145], v6 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[154:159], v[80:85], a[188:191], v9, v11 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[146:147], v4 offset:17920
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[154:159], v[68:73], a[184:187], v9, v11 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[154:159], v[62:67], a[180:183], v9, v11 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s47 offen offset:3072 lds
	ds_read_b128 v[148:151], v6 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[154:159], v[56:61], a[176:179], v9, v11 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[152:153], v4 offset:18432
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[154:159], v[50:55], a[172:175], v9, v10 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[154:159], v[44:49], a[168:171], v9, v10 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[166:169], v6 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[154:159], v[38:43], a[164:167], v9, v10 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_mov_b32 m0, s41
	ds_read_b64 v[170:171], v4 offset:18944
	buffer_load_dwordx4 v1, s[20:23], s48 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[154:159], v[26:31], a[160:163], v9, v10 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[160:165], v[26:31], a[192:195], v9, v10 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[154:157], v6 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[160:165], v[38:43], a[196:199], v9, v10 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[158:159], v4 offset:19456
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[160:165], v[44:49], a[200:203], v9, v10 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[160:165], v[50:55], a[204:207], v9, v10 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s48 offen offset:1024 lds
	ds_read_b128 v[178:181], v6 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[160:165], v[56:61], a[208:211], v9, v11 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[182:183], v4 offset:19968
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[160:165], v[62:67], a[212:215], v9, v11 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[160:165], v[68:73], a[216:219], v9, v11 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_i32 s46, s3, s46
	s_mov_b32 m0, s44
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[160:165], v[80:85], a[220:223], v9, v11 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dword v2, s[24:27], s46 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[172:177], v[80:85], a[252:255], v9, v11 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[172:177], v[68:73], a[248:251], v9, v11 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[172:177], v[62:67], a[244:247], v9, v11 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s46, s14, 0x2c000
	s_mov_b32 m0, s40
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[172:177], v[56:61], a[240:243], v9, v11 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dword v2, s[8:11], s46 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[172:177], v[50:55], a[236:239], v9, v10 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[172:177], v[44:49], a[232:235], v9, v10 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[172:177], v[38:43], a[228:231], v9, v10 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[172:177], v[26:31], a[224:227], v9, v10 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_add_i32 s46, s17, 0x180
	s_lshl_b32 s46, s46, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[32:37], v[20:25], a[0:3], v12, v14 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[32:37], v[74:79], a[4:7], v12, v14 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[32:37], v[86:91], a[8:11], v12, v14 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[8:9], v19 offset:10240
	ds_read_b64 v[10:11], v110 offset:35840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[32:37], v[92:97], a[12:15], v12, v14 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_add_i32 s47, s13, 0x48000
	s_mov_b32 m0, s2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[32:37], v[98:103], a[16:19], v12, v15 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s47 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[32:37], v[104:109], a[20:23], v12, v15 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[32:37], v[118:123], a[24:27], v12, v15 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[26:29], v6 offset:51200
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[32:37], v[124:129], a[28:31], v12, v15 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[30:31], v111 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[130:135], v[124:129], a[60:63], v12, v15 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s47 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[130:135], v[118:123], a[56:59], v12, v15 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[32:35], v112 offset:11264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[130:135], v[104:109], a[52:55], v12, v15 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[36:37], v113 offset:27648
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[130:135], v[98:103], a[48:51], v12, v15 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[130:135], v[92:97], a[44:47], v12, v14 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s47 offen offset:2048 lds
	ds_read_b128 v[38:41], v112 offset:12288
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[130:135], v[86:91], a[40:43], v12, v14 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[42:43], v113 offset:28160
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[130:135], v[74:79], a[36:39], v12, v14 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[130:135], v[20:25], a[32:35], v12, v14 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[44:47], v112 offset:13312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[136:141], v[20:25], a[64:67], v12, v14 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[48:49], v113 offset:28672
	buffer_load_dwordx4 v1, s[4:7], s47 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[136:141], v[74:79], a[68:71], v12, v14 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[136:141], v[86:91], a[72:75], v12, v14 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[50:53], v112 offset:14336
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[136:141], v[92:97], a[76:79], v12, v14 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[54:55], v113 offset:29184
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[136:141], v[98:103], a[80:83], v12, v15 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s48, s13, 0x49000
	s_mov_b32 m0, s30
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[136:141], v[104:109], a[84:87], v12, v15 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s48 offen lds
	ds_read_b128 v[56:59], v112 offset:15360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[136:141], v[118:123], a[88:91], v12, v15 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[60:61], v113 offset:29696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[136:141], v[124:129], a[92:95], v12, v15 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[142:147], v[124:129], a[124:127], v12, v15 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[62:65], v112 offset:16384
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[142:147], v[118:123], a[120:123], v12, v15 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[66:67], v113 offset:30208
	buffer_load_dwordx4 v1, s[4:7], s48 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[142:147], v[104:109], a[116:119], v12, v15 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[142:147], v[98:103], a[112:115], v12, v15 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[68:71], v112 offset:17408
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[142:147], v[92:97], a[108:111], v12, v14 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[72:73], v113 offset:30720
	s_mov_b32 m0, s31
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[142:147], v[86:91], a[104:107], v12, v14 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s47 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[142:147], v[74:79], a[100:103], v12, v14 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[80:83], v112 offset:18432
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[142:147], v[20:25], a[96:99], v12, v14 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[84:85], v113 offset:31232
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[148:153], v[20:25], a[128:131], v13, v14 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[148:153], v[74:79], a[132:135], v13, v14 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[130:133], v6 offset:52224
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[148:153], v[86:91], a[136:139], v13, v14 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[134:135], v111 offset:2560
	buffer_load_dwordx4 v1, s[20:23], s47 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[148:153], v[92:97], a[140:143], v13, v14 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[148:153], v[98:103], a[144:147], v13, v15 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[136:139], v6 offset:53248
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[148:153], v[104:109], a[148:151], v13, v15 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[140:141], v111 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[148:153], v[118:123], a[152:155], v13, v15 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s47 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[148:153], v[124:129], a[156:159], v13, v15 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[142:145], v6 offset:54272
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[166:171], v[124:129], a[188:191], v13, v15 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[146:147], v111 offset:3584
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[166:171], v[118:123], a[184:187], v13, v15 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[166:171], v[104:109], a[180:183], v13, v15 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s47 offen offset:3072 lds
	ds_read_b128 v[148:151], v6 offset:55296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[166:171], v[98:103], a[176:179], v13, v15 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[152:153], v111 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[166:171], v[92:97], a[172:175], v13, v14 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[166:171], v[86:91], a[168:171], v13, v14 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[160:163], v6 offset:56320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[166:171], v[74:79], a[164:167], v13, v14 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_mov_b32 m0, s32
	ds_read_b64 v[164:165], v111 offset:4608
	buffer_load_dwordx4 v1, s[20:23], s48 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[166:171], v[20:25], a[160:163], v13, v14 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[154:159], v[20:25], a[192:195], v13, v14 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[166:169], v6 offset:57344
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[154:159], v[74:79], a[196:199], v13, v14 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[170:171], v111 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[154:159], v[86:91], a[200:203], v13, v14 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[154:159], v[92:97], a[204:207], v13, v14 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s48 offen offset:1024 lds
	ds_read_b128 v[172:175], v6 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[154:159], v[98:103], a[208:211], v13, v15 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[176:177], v111 offset:5632
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[154:159], v[104:109], a[212:215], v13, v15 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[154:159], v[118:123], a[216:219], v13, v15 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_i32 s46, s3, s46
	s_mov_b32 m0, s33
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[154:159], v[124:129], a[220:223], v13, v15 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dword v2, s[24:27], s46 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[178:183], v[124:129], a[252:255], v13, v15 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[178:183], v[118:123], a[248:251], v13, v15 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[178:183], v[104:109], a[244:247], v13, v15 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s46, s14, 0x30000
	s_mov_b32 m0, s28
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[178:183], v[98:103], a[240:243], v13, v15 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dword v2, s[8:11], s46 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[178:183], v[92:97], a[236:239], v13, v14 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[178:183], v[86:91], a[232:235], v13, v14 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[178:183], v[74:79], a[228:231], v13, v14 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[178:183], v[20:25], a[224:227], v13, v14 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_add_i32 s46, s17, 0x1a0
	s_lshl_b32 s46, s46, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:31], v[32:37], a[0:3], v8, v10 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[26:31], v[38:43], a[4:7], v8, v10 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[26:31], v[44:49], a[8:11], v8, v10 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[118:119], v19 offset:61440
	ds_read_b64 v[120:121], v114 offset:21504
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[26:31], v[50:55], a[12:15], v8, v10 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_add_i32 s47, s13, 0x4e000
	s_mov_b32 m0, s36
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[26:31], v[56:61], a[16:19], v8, v11 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s47 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[26:31], v[62:67], a[20:23], v8, v11 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[26:31], v[68:73], a[24:27], v8, v11 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[20:23], v115 offset:36864
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[26:31], v[80:85], a[28:31], v8, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[24:25], v111 offset:53248
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[130:135], v[80:85], a[60:63], v8, v11 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s47 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[130:135], v[68:73], a[56:59], v8, v11 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[122:125], v112 offset:62464
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[130:135], v[62:67], a[52:55], v8, v11 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[126:127], v116 offset:13312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[130:135], v[56:61], a[48:51], v8, v11 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[130:135], v[50:55], a[44:47], v8, v10 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s47 offen offset:2048 lds
	ds_read_b128 v[154:157], v112 offset:63488
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[130:135], v[44:49], a[40:43], v8, v10 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[158:159], v116 offset:13824
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[130:135], v[38:43], a[36:39], v8, v10 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[130:135], v[32:37], a[32:35], v8, v10 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[128:131], v112 offset:64512
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[136:141], v[32:37], a[64:67], v8, v10 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[132:133], v116 offset:14336
	buffer_load_dwordx4 v1, s[4:7], s47 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[136:141], v[38:43], a[68:71], v8, v10 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[136:141], v[44:49], a[72:75], v8, v10 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[178:181], v117
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[136:141], v[50:55], a[76:79], v8, v10 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[182:183], v116 offset:14848
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[136:141], v[56:61], a[80:83], v8, v11 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s48, s13, 0x4f000
	s_mov_b32 m0, s37
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[136:141], v[62:67], a[84:87], v8, v11 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s48 offen lds
	ds_read_b128 v[184:187], v117 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[136:141], v[68:73], a[88:91], v8, v11 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[188:189], v116 offset:15360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[136:141], v[80:85], a[92:95], v8, v11 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[142:147], v[80:85], a[124:127], v8, v11 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[134:137], v117 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[142:147], v[68:73], a[120:123], v8, v11 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[138:139], v116 offset:15872
	buffer_load_dwordx4 v1, s[4:7], s48 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[142:147], v[62:67], a[116:119], v8, v11 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[142:147], v[56:61], a[112:115], v8, v11 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[190:193], v117 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[142:147], v[50:55], a[108:111], v8, v10 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[194:195], v116 offset:16384
	s_mov_b32 m0, s39
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[142:147], v[44:49], a[104:107], v8, v10 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s47 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[142:147], v[38:43], a[100:103], v8, v10 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[196:199], v117 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[142:147], v[32:37], a[96:99], v8, v10 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[200:201], v116 offset:16896
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[148:153], v[32:37], a[128:131], v9, v10 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[148:153], v[38:43], a[132:135], v9, v10 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[74:77], v115 offset:37888
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[148:153], v[44:49], a[136:139], v9, v10 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[78:79], v111 offset:53760
	buffer_load_dwordx4 v1, s[20:23], s47 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[148:153], v[50:55], a[140:143], v9, v10 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[148:153], v[56:61], a[144:147], v9, v11 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[86:89], v115 offset:38912
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[148:153], v[62:67], a[148:151], v9, v11 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[90:91], v111 offset:54272
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[148:153], v[68:73], a[152:155], v9, v11 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s47 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[148:153], v[80:85], a[156:159], v9, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[92:95], v115 offset:39936
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[160:165], v[80:85], a[188:191], v9, v11 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[96:97], v111 offset:54784
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[160:165], v[68:73], a[184:187], v9, v11 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[160:165], v[62:67], a[180:183], v9, v11 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s47 offen offset:3072 lds
	ds_read_b128 v[98:101], v115 offset:40960
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[160:165], v[56:61], a[176:179], v9, v11 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[102:103], v111 offset:55296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[160:165], v[50:55], a[172:175], v9, v10 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[160:165], v[44:49], a[168:171], v9, v10 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[104:107], v115 offset:41984
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[160:165], v[38:43], a[164:167], v9, v10 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_mov_b32 m0, s35
	ds_read_b64 v[108:109], v111 offset:55808
	buffer_load_dwordx4 v1, s[20:23], s48 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[160:165], v[32:37], a[160:163], v9, v10 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[166:171], v[32:37], a[192:195], v9, v10 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[140:143], v115 offset:43008
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[166:171], v[38:43], a[196:199], v9, v10 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[144:145], v111 offset:56320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[166:171], v[44:49], a[200:203], v9, v10 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[166:171], v[50:55], a[204:207], v9, v10 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s48 offen offset:1024 lds
	ds_read_b128 v[146:149], v115 offset:44032
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[166:171], v[56:61], a[208:211], v9, v11 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[150:151], v111 offset:56832
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[166:171], v[62:67], a[212:215], v9, v11 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[166:171], v[68:73], a[216:219], v9, v11 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_i32 s46, s3, s46
	s_mov_b32 m0, s38
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[166:171], v[80:85], a[220:223], v9, v11 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dword v2, s[24:27], s46 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[172:177], v[80:85], a[252:255], v9, v11 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[172:177], v[68:73], a[248:251], v9, v11 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[172:177], v[62:67], a[244:247], v9, v11 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s46, s14, 0x34000
	s_mov_b32 m0, s34
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[172:177], v[56:61], a[240:243], v9, v11 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dword v2, s[8:11], s46 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[172:177], v[50:55], a[236:239], v9, v10 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[172:177], v[44:49], a[232:235], v9, v10 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[172:177], v[38:43], a[228:231], v9, v10 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[172:177], v[32:37], a[224:227], v9, v10 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_add_i32 s17, s17, 0x1c0
	s_lshl_b32 s17, s17, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:25], v[122:127], a[0:3], v118, v120 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[20:25], v[154:159], a[4:7], v118, v120 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[20:25], v[128:133], a[8:11], v118, v120 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[8:9], v5 offset:24576
	ds_read_b64 v[10:11], v3 offset:50176
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[20:25], v[178:183], a[12:15], v118, v120 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_add_i32 s46, s13, 0x54000
	s_mov_b32 m0, s42
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[20:25], v[184:189], a[16:19], v118, v121 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s46 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[20:25], v[134:139], a[20:23], v118, v121 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[20:25], v[190:195], a[24:27], v118, v121 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[12:15], v6
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[20:25], v[196:201], a[28:31], v118, v121 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[16:17], v4 offset:16384
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[74:79], v[196:201], a[60:63], v118, v121 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s46 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[74:79], v[190:195], a[56:59], v118, v121 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[20:23], v18 offset:25600
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[74:79], v[134:139], a[52:55], v118, v121 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[24:25], v7 offset:41984
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[74:79], v[184:189], a[48:51], v118, v121 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[74:79], v[178:183], a[44:47], v118, v120 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s46 offen offset:2048 lds
	ds_read_b128 v[26:29], v18 offset:26624
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[74:79], v[128:133], a[40:43], v118, v120 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[30:31], v7 offset:42496
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[74:79], v[154:159], a[36:39], v118, v120 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[74:79], v[122:127], a[32:35], v118, v120 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[32:35], v18 offset:27648
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[86:91], v[122:127], a[64:67], v118, v120 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[36:37], v7 offset:43008
	buffer_load_dwordx4 v1, s[4:7], s46 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[86:91], v[154:159], a[68:71], v118, v120 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[86:91], v[128:133], a[72:75], v118, v120 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[38:41], v18 offset:28672
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[86:91], v[178:183], a[76:79], v118, v120 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[42:43], v7 offset:43520
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[86:91], v[184:189], a[80:83], v118, v121 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s13, s13, 0x55000
	s_mov_b32 m0, s43
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[86:91], v[134:139], a[84:87], v118, v121 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s13 offen lds
	ds_read_b128 v[44:47], v18 offset:29696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[86:91], v[190:195], a[88:91], v118, v121 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[48:49], v7 offset:44032
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[86:91], v[196:201], a[92:95], v118, v121 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[92:97], v[196:201], a[124:127], v118, v121 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[50:53], v18 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[92:97], v[190:195], a[120:123], v118, v121 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[54:55], v7 offset:44544
	buffer_load_dwordx4 v1, s[4:7], s13 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[92:97], v[134:139], a[116:119], v118, v121 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[92:97], v[184:189], a[112:115], v118, v121 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[56:59], v18 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[92:97], v[178:183], a[108:111], v118, v120 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[60:61], v7 offset:45056
	s_mov_b32 m0, s45
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[92:97], v[128:133], a[104:107], v118, v120 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s46 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[92:97], v[154:159], a[100:103], v118, v120 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[62:65], v18 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[92:97], v[122:127], a[96:99], v118, v120 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[66:67], v7 offset:45568
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[98:103], v[122:127], a[128:131], v119, v120 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[98:103], v[154:159], a[132:135], v119, v120 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[68:71], v6 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[98:103], v[128:133], a[136:139], v119, v120 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[72:73], v4 offset:16896
	buffer_load_dwordx4 v1, s[20:23], s46 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[98:103], v[178:183], a[140:143], v119, v120 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[98:103], v[184:189], a[144:147], v119, v121 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[74:77], v6 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[98:103], v[134:139], a[148:151], v119, v121 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[78:79], v4 offset:17408
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[98:103], v[190:195], a[152:155], v119, v121 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s46 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[98:103], v[196:201], a[156:159], v119, v121 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[80:83], v6 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[104:109], v[196:201], a[188:191], v119, v121 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[84:85], v4 offset:17920
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[104:109], v[190:195], a[184:187], v119, v121 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[104:109], v[134:139], a[180:183], v119, v121 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s46 offen offset:3072 lds
	ds_read_b128 v[86:89], v6 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[104:109], v[184:189], a[176:179], v119, v121 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[90:91], v4 offset:18432
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[104:109], v[178:183], a[172:175], v119, v120 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[104:109], v[128:133], a[168:171], v119, v120 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[92:95], v6 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[104:109], v[154:159], a[164:167], v119, v120 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_mov_b32 m0, s41
	ds_read_b64 v[96:97], v4 offset:18944
	buffer_load_dwordx4 v1, s[20:23], s13 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[104:109], v[122:127], a[160:163], v119, v120 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[140:145], v[122:127], a[192:195], v119, v120 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[98:101], v6 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[140:145], v[154:159], a[196:199], v119, v120 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[102:103], v4 offset:19456
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[140:145], v[128:133], a[200:203], v119, v120 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[140:145], v[178:183], a[204:207], v119, v120 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s13 offen offset:1024 lds
	ds_read_b128 v[104:107], v6 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[140:145], v[184:189], a[208:211], v119, v121 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[108:109], v4 offset:19968
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[140:145], v[134:139], a[212:215], v119, v121 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[140:145], v[190:195], a[216:219], v119, v121 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_i32 s13, s3, s17
	s_mov_b32 m0, s44
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[140:145], v[196:201], a[220:223], v119, v121 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dword v2, s[24:27], s13 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[146:151], v[196:201], a[252:255], v119, v121 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[146:151], v[190:195], a[248:251], v119, v121 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[146:151], v[134:139], a[244:247], v119, v121 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s13, s14, 0x38000
	s_mov_b32 m0, s40
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[146:151], v[184:189], a[240:243], v119, v121 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dword v2, s[8:11], s13 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[146:151], v[178:183], a[236:239], v119, v120 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[146:151], v[128:133], a[232:235], v119, v120 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[146:151], v[154:159], a[228:231], v119, v120 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[146:151], v[122:127], a[224:227], v119, v120 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_add_i32 s12, s12, 1
	s_cmp_lt_i32 s12, 42
	s_cbranch_scc1 .Lmain_loop
	s_add_i32 s12, s15, 0x3f60
	s_lshl_b32 s12, s12, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[12:17], v[20:25], a[0:3], v8, v10 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[12:17], v[26:31], a[4:7], v8, v10 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[12:17], v[32:37], a[8:11], v8, v10 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_barrier
	v_add_u32_e32 v19, 0x10000, v5
	v_add_u32_e32 v110, 0x10000, v3
	ds_read_b64 v[112:113], v19 offset:10240
	ds_read_b64 v[114:115], v110 offset:35840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[12:17], v[38:43], a[12:15], v8, v10 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_add_i32 s13, s2, 0xbe2000
	s_mov_b32 m0, s2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[12:17], v[44:49], a[16:19], v8, v11 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s13 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[12:17], v[50:55], a[20:23], v8, v11 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[12:17], v[56:61], a[24:27], v8, v11 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[116:119], v6 offset:51200
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[12:17], v[62:67], a[28:31], v8, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_add_u32_e32 v12, 0x10000, v4
	ds_read_b64 v[120:121], v12 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[68:73], v[62:67], a[60:63], v8, v11 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s13 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[68:73], v[56:61], a[56:59], v8, v11 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_add_u32_e32 v13, 0x10000, v18
	ds_read_b128 v[122:125], v13 offset:11264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[68:73], v[50:55], a[52:55], v8, v11 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_add_u32_e32 v14, 0x10000, v7
	ds_read_b64 v[126:127], v14 offset:27648
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[68:73], v[44:49], a[48:51], v8, v11 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[68:73], v[38:43], a[44:47], v8, v10 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s13 offen offset:2048 lds
	ds_read_b128 v[128:131], v13 offset:12288
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[68:73], v[32:37], a[40:43], v8, v10 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[132:133], v14 offset:28160
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[68:73], v[26:31], a[36:39], v8, v10 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[68:73], v[20:25], a[32:35], v8, v10 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[68:71], v13 offset:13312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[74:79], v[20:25], a[64:67], v8, v10 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[72:73], v14 offset:28672
	buffer_load_dwordx4 v1, s[4:7], s13 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[74:79], v[26:31], a[68:71], v8, v10 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[74:79], v[32:37], a[72:75], v8, v10 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[134:137], v13 offset:14336
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[74:79], v[38:43], a[76:79], v8, v10 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[138:139], v14 offset:29184
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[74:79], v[44:49], a[80:83], v8, v11 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s14, s2, 0xbe3000
	s_mov_b32 m0, s30
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[74:79], v[50:55], a[84:87], v8, v11 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s14 offen lds
	ds_read_b128 v[140:143], v13 offset:15360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[74:79], v[56:61], a[88:91], v8, v11 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[144:145], v14 offset:29696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[74:79], v[62:67], a[92:95], v8, v11 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[80:85], v[62:67], a[124:127], v8, v11 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[74:77], v13 offset:16384
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[80:85], v[56:61], a[120:123], v8, v11 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[78:79], v14 offset:30208
	buffer_load_dwordx4 v1, s[4:7], s14 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[80:85], v[50:55], a[116:119], v8, v11 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[80:85], v[44:49], a[112:115], v8, v11 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[146:149], v13 offset:17408
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[80:85], v[38:43], a[108:111], v8, v10 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[150:151], v14 offset:30720
	s_mov_b32 m0, s31
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[80:85], v[32:37], a[104:107], v8, v10 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s13 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[80:85], v[26:31], a[100:103], v8, v10 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[152:155], v13 offset:18432
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[80:85], v[20:25], a[96:99], v8, v10 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[156:157], v14 offset:31232
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[86:91], v[20:25], a[128:131], v9, v10 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[86:91], v[26:31], a[132:135], v9, v10 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[80:83], v6 offset:52224
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[86:91], v[32:37], a[136:139], v9, v10 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[84:85], v12 offset:2560
	buffer_load_dwordx4 v1, s[20:23], s13 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[86:91], v[38:43], a[140:143], v9, v10 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[86:91], v[44:49], a[144:147], v9, v11 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[158:161], v6 offset:53248
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[86:91], v[50:55], a[148:151], v9, v11 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[162:163], v12 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[86:91], v[56:61], a[152:155], v9, v11 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s13 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[86:91], v[62:67], a[156:159], v9, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[86:89], v6 offset:54272
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[92:97], v[62:67], a[188:191], v9, v11 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[90:91], v12 offset:3584
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[92:97], v[56:61], a[184:187], v9, v11 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[92:97], v[50:55], a[180:183], v9, v11 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s13 offen offset:3072 lds
	ds_read_b128 v[164:167], v6 offset:55296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[92:97], v[44:49], a[176:179], v9, v11 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[168:169], v12 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[92:97], v[38:43], a[172:175], v9, v10 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[92:97], v[32:37], a[168:171], v9, v10 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[170:173], v6 offset:56320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[92:97], v[26:31], a[164:167], v9, v10 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_mov_b32 m0, s32
	ds_read_b64 v[174:175], v12 offset:4608
	buffer_load_dwordx4 v1, s[20:23], s14 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[92:97], v[20:25], a[160:163], v9, v10 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[98:103], v[20:25], a[192:195], v9, v10 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[92:95], v6 offset:57344
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[98:103], v[26:31], a[196:199], v9, v10 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[96:97], v12 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[98:103], v[32:37], a[200:203], v9, v10 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[98:103], v[38:43], a[204:207], v9, v10 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s14 offen offset:1024 lds
	ds_read_b128 v[176:179], v6 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[98:103], v[44:49], a[208:211], v9, v11 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[180:181], v12 offset:5632
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[98:103], v[50:55], a[212:215], v9, v11 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[98:103], v[56:61], a[216:219], v9, v11 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_i32 s12, s3, s12
	s_mov_b32 m0, s33
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[98:103], v[62:67], a[220:223], v9, v11 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dword v2, s[24:27], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[104:109], v[62:67], a[252:255], v9, v11 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[104:109], v[56:61], a[248:251], v9, v11 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[104:109], v[50:55], a[244:247], v9, v11 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s12, s29, 0x7ec000
	s_mov_b32 m0, s28
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[104:109], v[44:49], a[240:243], v9, v11 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dword v2, s[8:11], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[104:109], v[38:43], a[236:239], v9, v10 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[104:109], v[32:37], a[232:235], v9, v10 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[104:109], v[26:31], a[228:231], v9, v10 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[104:109], v[20:25], a[224:227], v9, v10 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_add_i32 s12, s15, 0x3f80
	s_lshl_b32 s12, s12, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[116:121], v[122:127], a[0:3], v112, v114 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[116:121], v[128:133], a[4:7], v112, v114 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[116:121], v[68:73], a[8:11], v112, v114 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_barrier
	v_add_u32_e32 v8, 0x20000, v3
	ds_read_b64 v[10:11], v19 offset:61440
	ds_read_b64 v[16:17], v8 offset:21504
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[116:121], v[134:139], a[12:15], v112, v114 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_add_i32 s13, s2, 0xbe8000
	s_mov_b32 m0, s36
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[116:121], v[140:145], a[16:19], v112, v115 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s13 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[116:121], v[74:79], a[20:23], v112, v115 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[116:121], v[146:151], a[24:27], v112, v115 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_add_u32_e32 v9, 0x10000, v6
	ds_read_b128 v[20:23], v9 offset:36864
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[116:121], v[152:157], a[28:31], v112, v115 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[24:25], v12 offset:53248
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[80:85], v[152:157], a[60:63], v112, v115 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s13 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[80:85], v[146:151], a[56:59], v112, v115 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[26:29], v13 offset:62464
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[80:85], v[74:79], a[52:55], v112, v115 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_add_u32_e32 v15, 0x20000, v7
	ds_read_b64 v[30:31], v15 offset:13312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[80:85], v[140:145], a[48:51], v112, v115 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[80:85], v[134:139], a[44:47], v112, v114 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s13 offen offset:2048 lds
	ds_read_b128 v[32:35], v13 offset:63488
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[80:85], v[68:73], a[40:43], v112, v114 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[36:37], v15 offset:13824
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[80:85], v[128:133], a[36:39], v112, v114 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[80:85], v[122:127], a[32:35], v112, v114 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[38:41], v13 offset:64512
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[158:163], v[122:127], a[64:67], v112, v114 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[42:43], v15 offset:14336
	buffer_load_dwordx4 v1, s[4:7], s13 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[158:163], v[128:133], a[68:71], v112, v114 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[158:163], v[68:73], a[72:75], v112, v114 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_add_u32_e32 v44, 0x20000, v18
	ds_read_b128 v[46:49], v44
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[158:163], v[134:139], a[76:79], v112, v114 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[50:51], v15 offset:14848
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[158:163], v[140:145], a[80:83], v112, v115 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s14, s2, 0xbe9000
	s_mov_b32 m0, s37
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[158:163], v[74:79], a[84:87], v112, v115 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s14 offen lds
	ds_read_b128 v[52:55], v44 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[158:163], v[146:151], a[88:91], v112, v115 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[56:57], v15 offset:15360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[158:163], v[152:157], a[92:95], v112, v115 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[86:91], v[152:157], a[124:127], v112, v115 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[58:61], v44 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[86:91], v[146:151], a[120:123], v112, v115 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[62:63], v15 offset:15872
	buffer_load_dwordx4 v1, s[4:7], s14 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[86:91], v[74:79], a[116:119], v112, v115 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[86:91], v[140:145], a[112:115], v112, v115 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[80:83], v44 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[86:91], v[134:139], a[108:111], v112, v114 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[84:85], v15 offset:16384
	s_mov_b32 m0, s39
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[86:91], v[68:73], a[104:107], v112, v114 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s13 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[86:91], v[128:133], a[100:103], v112, v114 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[98:101], v44 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[86:91], v[122:127], a[96:99], v112, v114 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[102:103], v15 offset:16896
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[164:169], v[122:127], a[128:131], v113, v114 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[164:169], v[128:133], a[132:135], v113, v114 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[86:89], v9 offset:37888
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[164:169], v[68:73], a[136:139], v113, v114 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[90:91], v12 offset:53760
	buffer_load_dwordx4 v1, s[20:23], s13 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[164:169], v[134:139], a[140:143], v113, v114 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[164:169], v[140:145], a[144:147], v113, v115 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[104:107], v9 offset:38912
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[164:169], v[74:79], a[148:151], v113, v115 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[108:109], v12 offset:54272
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[164:169], v[146:151], a[152:155], v113, v115 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s13 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[164:169], v[152:157], a[156:159], v113, v115 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[116:119], v9 offset:39936
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[170:175], v[152:157], a[188:191], v113, v115 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[120:121], v12 offset:54784
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[170:175], v[146:151], a[184:187], v113, v115 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[170:175], v[74:79], a[180:183], v113, v115 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s13 offen offset:3072 lds
	ds_read_b128 v[158:161], v9 offset:40960
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[170:175], v[140:145], a[176:179], v113, v115 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[162:163], v12 offset:55296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[170:175], v[134:139], a[172:175], v113, v114 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[170:175], v[68:73], a[168:171], v113, v114 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[164:167], v9 offset:41984
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[170:175], v[128:133], a[164:167], v113, v114 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_mov_b32 m0, s35
	ds_read_b64 v[168:169], v12 offset:55808
	buffer_load_dwordx4 v1, s[20:23], s14 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[170:175], v[122:127], a[160:163], v113, v114 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[92:97], v[122:127], a[192:195], v113, v114 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[170:173], v9 offset:43008
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[92:97], v[128:133], a[196:199], v113, v114 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[174:175], v12 offset:56320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[92:97], v[68:73], a[200:203], v113, v114 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[92:97], v[134:139], a[204:207], v113, v114 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s14 offen offset:1024 lds
	ds_read_b128 v[182:185], v9 offset:44032
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[92:97], v[140:145], a[208:211], v113, v115 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[186:187], v12 offset:56832
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[92:97], v[74:79], a[212:215], v113, v115 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[92:97], v[146:151], a[216:219], v113, v115 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_i32 s12, s3, s12
	s_mov_b32 m0, s38
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[92:97], v[152:157], a[220:223], v113, v115 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dword v2, s[24:27], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[176:181], v[152:157], a[252:255], v113, v115 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[176:181], v[146:151], a[248:251], v113, v115 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[176:181], v[74:79], a[244:247], v113, v115 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s12, s29, 0x7f0000
	s_mov_b32 m0, s34
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[176:181], v[140:145], a[240:243], v113, v115 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dword v2, s[8:11], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[176:181], v[134:139], a[236:239], v113, v114 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[176:181], v[68:73], a[232:235], v113, v114 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[176:181], v[128:133], a[228:231], v113, v114 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[176:181], v[122:127], a[224:227], v113, v114 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_add_i32 s12, s15, 0x3fa0
	s_lshl_b32 s12, s12, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:25], v[26:31], a[0:3], v10, v16 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[20:25], v[32:37], a[4:7], v10, v16 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[20:25], v[38:43], a[8:11], v10, v16 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[64:65], v5 offset:24576
	ds_read_b64 v[66:67], v3 offset:50176
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[20:25], v[46:51], a[12:15], v10, v16 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_add_i32 s13, s2, 0xbee000
	s_mov_b32 m0, s42
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[20:25], v[52:57], a[16:19], v10, v17 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s13 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[20:25], v[58:63], a[20:23], v10, v17 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[20:25], v[80:85], a[24:27], v10, v17 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[68:71], v6
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[20:25], v[98:103], a[28:31], v10, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[72:73], v4 offset:16384
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[86:91], v[98:103], a[60:63], v10, v17 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s13 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[86:91], v[80:85], a[56:59], v10, v17 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[20:23], v18 offset:25600
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[86:91], v[58:63], a[52:55], v10, v17 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[24:25], v7 offset:41984
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[86:91], v[52:57], a[48:51], v10, v17 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[86:91], v[46:51], a[44:47], v10, v16 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s13 offen offset:2048 lds
	ds_read_b128 v[74:77], v18 offset:26624
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[86:91], v[38:43], a[40:43], v10, v16 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[78:79], v7 offset:42496
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[86:91], v[32:37], a[36:39], v10, v16 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[86:91], v[26:31], a[32:35], v10, v16 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[86:89], v18 offset:27648
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[104:109], v[26:31], a[64:67], v10, v16 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[90:91], v7 offset:43008
	buffer_load_dwordx4 v1, s[4:7], s13 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[104:109], v[32:37], a[68:71], v10, v16 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[104:109], v[38:43], a[72:75], v10, v16 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[92:95], v18 offset:28672
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[104:109], v[46:51], a[76:79], v10, v16 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[96:97], v7 offset:43520
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[104:109], v[52:57], a[80:83], v10, v17 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s14, s2, 0xbef000
	s_mov_b32 m0, s43
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[104:109], v[58:63], a[84:87], v10, v17 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s14 offen lds
	ds_read_b128 v[122:125], v18 offset:29696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[104:109], v[80:85], a[88:91], v10, v17 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[126:127], v7 offset:44032
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[104:109], v[98:103], a[92:95], v10, v17 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[116:121], v[98:103], a[124:127], v10, v17 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[104:107], v18 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[116:121], v[80:85], a[120:123], v10, v17 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[108:109], v7 offset:44544
	buffer_load_dwordx4 v1, s[4:7], s14 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[116:121], v[58:63], a[116:119], v10, v17 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[116:121], v[52:57], a[112:115], v10, v17 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[128:131], v18 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[116:121], v[46:51], a[108:111], v10, v16 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[132:133], v7 offset:45056
	s_mov_b32 m0, s45
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[116:121], v[38:43], a[104:107], v10, v16 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s13 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[116:121], v[32:37], a[100:103], v10, v16 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[112:115], v18 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[116:121], v[26:31], a[96:99], v10, v16 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[116:117], v7 offset:45568
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[158:163], v[26:31], a[128:131], v11, v16 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[158:163], v[32:37], a[132:135], v11, v16 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[134:137], v6 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[158:163], v[38:43], a[136:139], v11, v16 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[138:139], v4 offset:16896
	buffer_load_dwordx4 v1, s[20:23], s13 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[158:163], v[46:51], a[140:143], v11, v16 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[158:163], v[52:57], a[144:147], v11, v17 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[140:143], v6 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[158:163], v[58:63], a[148:151], v11, v17 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[144:145], v4 offset:17408
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[158:163], v[80:85], a[152:155], v11, v17 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s13 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[158:163], v[98:103], a[156:159], v11, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[146:149], v6 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[164:169], v[98:103], a[188:191], v11, v17 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[150:151], v4 offset:17920
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[164:169], v[80:85], a[184:187], v11, v17 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[164:169], v[58:63], a[180:183], v11, v17 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s13 offen offset:3072 lds
	ds_read_b128 v[152:155], v6 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[164:169], v[52:57], a[176:179], v11, v17 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[156:157], v4 offset:18432
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[164:169], v[46:51], a[172:175], v11, v16 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[164:169], v[38:43], a[168:171], v11, v16 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[158:161], v6 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[164:169], v[32:37], a[164:167], v11, v16 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_mov_b32 m0, s41
	ds_read_b64 v[162:163], v4 offset:18944
	buffer_load_dwordx4 v1, s[20:23], s14 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[164:169], v[26:31], a[160:163], v11, v16 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[170:175], v[26:31], a[192:195], v11, v16 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[164:167], v6 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[170:175], v[32:37], a[196:199], v11, v16 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[168:169], v4 offset:19456
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[170:175], v[38:43], a[200:203], v11, v16 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[170:175], v[46:51], a[204:207], v11, v16 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s14 offen offset:1024 lds
	ds_read_b128 v[176:179], v6 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[170:175], v[52:57], a[208:211], v11, v17 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[180:181], v4 offset:19968
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[170:175], v[58:63], a[212:215], v11, v17 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[170:175], v[80:85], a[216:219], v11, v17 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_i32 s12, s3, s12
	s_mov_b32 m0, s44
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[170:175], v[98:103], a[220:223], v11, v17 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dword v2, s[24:27], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[182:187], v[98:103], a[252:255], v11, v17 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[182:187], v[80:85], a[248:251], v11, v17 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[182:187], v[58:63], a[244:247], v11, v17 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s12, s29, 0x7f4000
	s_mov_b32 m0, s40
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[182:187], v[52:57], a[240:243], v11, v17 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dword v2, s[8:11], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[182:187], v[46:51], a[236:239], v11, v16 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[182:187], v[38:43], a[232:235], v11, v16 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[182:187], v[32:37], a[228:231], v11, v16 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[182:187], v[26:31], a[224:227], v11, v16 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_add_i32 s12, s15, 0x3fc0
	s_lshl_b32 s12, s12, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[68:73], v[20:25], a[0:3], v64, v66 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[68:73], v[74:79], a[4:7], v64, v66 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[68:73], v[86:91], a[8:11], v64, v66 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[10:11], v19 offset:10240
	ds_read_b64 v[16:17], v110 offset:35840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[68:73], v[92:97], a[12:15], v64, v66 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_add_i32 s13, s2, 0xbf4000
	s_mov_b32 m0, s2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[68:73], v[122:127], a[16:19], v64, v67 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s13 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[68:73], v[104:109], a[20:23], v64, v67 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[68:73], v[128:133], a[24:27], v64, v67 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[26:29], v6 offset:51200
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[68:73], v[112:117], a[28:31], v64, v67 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[30:31], v12 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[134:139], v[112:117], a[60:63], v64, v67 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s13 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[134:139], v[128:133], a[56:59], v64, v67 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[32:35], v13 offset:11264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[134:139], v[104:109], a[52:55], v64, v67 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[36:37], v14 offset:27648
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[134:139], v[122:127], a[48:51], v64, v67 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[134:139], v[92:97], a[44:47], v64, v66 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s13 offen offset:2048 lds
	ds_read_b128 v[38:41], v13 offset:12288
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[134:139], v[86:91], a[40:43], v64, v66 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[42:43], v14 offset:28160
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[134:139], v[74:79], a[36:39], v64, v66 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[134:139], v[20:25], a[32:35], v64, v66 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[46:49], v13 offset:13312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[140:145], v[20:25], a[64:67], v64, v66 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[50:51], v14 offset:28672
	buffer_load_dwordx4 v1, s[4:7], s13 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[140:145], v[74:79], a[68:71], v64, v66 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[140:145], v[86:91], a[72:75], v64, v66 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[52:55], v13 offset:14336
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[140:145], v[92:97], a[76:79], v64, v66 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[56:57], v14 offset:29184
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[140:145], v[122:127], a[80:83], v64, v67 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s14, s2, 0xbf5000
	s_mov_b32 m0, s30
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[140:145], v[104:109], a[84:87], v64, v67 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s14 offen lds
	ds_read_b128 v[58:61], v13 offset:15360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[140:145], v[128:133], a[88:91], v64, v67 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[62:63], v14 offset:29696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[140:145], v[112:117], a[92:95], v64, v67 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[146:151], v[112:117], a[124:127], v64, v67 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[68:71], v13 offset:16384
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[146:151], v[128:133], a[120:123], v64, v67 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[72:73], v14 offset:30208
	buffer_load_dwordx4 v1, s[4:7], s14 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[146:151], v[104:109], a[116:119], v64, v67 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[146:151], v[122:127], a[112:115], v64, v67 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[80:83], v13 offset:17408
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[146:151], v[92:97], a[108:111], v64, v66 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[84:85], v14 offset:30720
	s_mov_b32 m0, s31
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[146:151], v[86:91], a[104:107], v64, v66 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s13 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[146:151], v[74:79], a[100:103], v64, v66 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[98:101], v13 offset:18432
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[146:151], v[20:25], a[96:99], v64, v66 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[102:103], v14 offset:31232
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[152:157], v[20:25], a[128:131], v65, v66 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[152:157], v[74:79], a[132:135], v65, v66 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[134:137], v6 offset:52224
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[152:157], v[86:91], a[136:139], v65, v66 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[138:139], v12 offset:2560
	buffer_load_dwordx4 v1, s[20:23], s13 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[152:157], v[92:97], a[140:143], v65, v66 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[152:157], v[122:127], a[144:147], v65, v67 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[140:143], v6 offset:53248
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[152:157], v[104:109], a[148:151], v65, v67 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[144:145], v12 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[152:157], v[128:133], a[152:155], v65, v67 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s13 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[152:157], v[112:117], a[156:159], v65, v67 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[146:149], v6 offset:54272
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[158:163], v[112:117], a[188:191], v65, v67 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[150:151], v12 offset:3584
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[158:163], v[128:133], a[184:187], v65, v67 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[158:163], v[104:109], a[180:183], v65, v67 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s13 offen offset:3072 lds
	ds_read_b128 v[152:155], v6 offset:55296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[158:163], v[122:127], a[176:179], v65, v67 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[156:157], v12 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[158:163], v[92:97], a[172:175], v65, v66 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[158:163], v[86:91], a[168:171], v65, v66 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[170:173], v6 offset:56320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[158:163], v[74:79], a[164:167], v65, v66 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_mov_b32 m0, s32
	ds_read_b64 v[174:175], v12 offset:4608
	buffer_load_dwordx4 v1, s[20:23], s14 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[158:163], v[20:25], a[160:163], v65, v66 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[164:169], v[20:25], a[192:195], v65, v66 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[158:161], v6 offset:57344
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[164:169], v[74:79], a[196:199], v65, v66 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[162:163], v12 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[164:169], v[86:91], a[200:203], v65, v66 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[164:169], v[92:97], a[204:207], v65, v66 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s14 offen offset:1024 lds
	ds_read_b128 v[182:185], v6 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[164:169], v[122:127], a[208:211], v65, v67 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[186:187], v12 offset:5632
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[164:169], v[104:109], a[212:215], v65, v67 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[164:169], v[128:133], a[216:219], v65, v67 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_i32 s12, s3, s12
	s_mov_b32 m0, s33
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[164:169], v[112:117], a[220:223], v65, v67 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dword v2, s[24:27], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[176:181], v[112:117], a[252:255], v65, v67 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[176:181], v[128:133], a[248:251], v65, v67 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[176:181], v[104:109], a[244:247], v65, v67 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s12, s29, 0x7f8000
	s_mov_b32 m0, s28
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[176:181], v[122:127], a[240:243], v65, v67 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dword v2, s[8:11], s12 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[176:181], v[92:97], a[236:239], v65, v66 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[176:181], v[86:91], a[232:235], v65, v66 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[176:181], v[74:79], a[228:231], v65, v66 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[176:181], v[20:25], a[224:227], v65, v66 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_add_i32 s12, s15, 0x3fe0
	s_lshl_b32 s12, s12, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:31], v[32:37], a[0:3], v10, v16 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[26:31], v[38:43], a[4:7], v10, v16 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[26:31], v[46:51], a[8:11], v10, v16 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[20:21], v19 offset:61440
	ds_read_b64 v[22:23], v8 offset:21504
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[26:31], v[52:57], a[12:15], v10, v16 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_add_i32 s13, s2, 0xbfa000
	s_mov_b32 m0, s36
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[26:31], v[58:63], a[16:19], v10, v17 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s13 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[26:31], v[68:73], a[20:23], v10, v17 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[26:31], v[80:85], a[24:27], v10, v17 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[74:77], v9 offset:36864
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[26:31], v[98:103], a[28:31], v10, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[78:79], v12 offset:53248
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[134:139], v[98:103], a[60:63], v10, v17 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s13 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[134:139], v[80:85], a[56:59], v10, v17 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[24:27], v13 offset:62464
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[134:139], v[68:73], a[52:55], v10, v17 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[28:29], v15 offset:13312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[134:139], v[58:63], a[48:51], v10, v17 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[134:139], v[52:57], a[44:47], v10, v16 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s13 offen offset:2048 lds
	ds_read_b128 v[86:89], v13 offset:63488
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[134:139], v[46:51], a[40:43], v10, v16 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[90:91], v15 offset:13824
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[134:139], v[38:43], a[36:39], v10, v16 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[134:139], v[32:37], a[32:35], v10, v16 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[92:95], v13 offset:64512
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[140:145], v[32:37], a[64:67], v10, v16 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[96:97], v15 offset:14336
	buffer_load_dwordx4 v1, s[4:7], s13 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[140:145], v[38:43], a[68:71], v10, v16 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[140:145], v[46:51], a[72:75], v10, v16 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[104:107], v44
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[140:145], v[52:57], a[76:79], v10, v16 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[108:109], v15 offset:14848
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[140:145], v[58:63], a[80:83], v10, v17 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s2, s2, 0xbfb000
	s_mov_b32 m0, s37
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[140:145], v[68:73], a[84:87], v10, v17 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], s2 offen lds
	ds_read_b128 v[112:115], v44 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[140:145], v[80:85], a[88:91], v10, v17 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[116:117], v15 offset:15360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[140:145], v[98:103], a[92:95], v10, v17 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[146:151], v[98:103], a[124:127], v10, v17 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[118:121], v44 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[146:151], v[80:85], a[120:123], v10, v17 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[122:123], v15 offset:15872
	buffer_load_dwordx4 v1, s[4:7], s2 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[146:151], v[68:73], a[116:119], v10, v17 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[146:151], v[58:63], a[112:115], v10, v17 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[124:127], v44 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[146:151], v[52:57], a[108:111], v10, v16 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[128:129], v15 offset:16384
	s_mov_b32 m0, s39
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[146:151], v[46:51], a[104:107], v10, v16 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s13 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[146:151], v[38:43], a[100:103], v10, v16 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[130:133], v44 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[146:151], v[32:37], a[96:99], v10, v16 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[134:135], v15 offset:16896
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[152:157], v[32:37], a[128:131], v11, v16 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[152:157], v[38:43], a[132:135], v11, v16 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[136:139], v9 offset:37888
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[152:157], v[46:51], a[136:139], v11, v16 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[140:141], v12 offset:53760
	buffer_load_dwordx4 v1, s[20:23], s13 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[152:157], v[52:57], a[140:143], v11, v16 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[152:157], v[58:63], a[144:147], v11, v17 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[142:145], v9 offset:38912
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[152:157], v[68:73], a[148:151], v11, v17 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[146:147], v12 offset:54272
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[152:157], v[80:85], a[152:155], v11, v17 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s13 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[152:157], v[98:103], a[156:159], v11, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[148:151], v9 offset:39936
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[170:175], v[98:103], a[188:191], v11, v17 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[152:153], v12 offset:54784
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[170:175], v[80:85], a[184:187], v11, v17 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[170:175], v[68:73], a[180:183], v11, v17 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s13 offen offset:3072 lds
	ds_read_b128 v[164:167], v9 offset:40960
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[170:175], v[58:63], a[176:179], v11, v17 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[168:169], v12 offset:55296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[170:175], v[52:57], a[172:175], v11, v16 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[170:175], v[46:51], a[168:171], v11, v16 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[176:179], v9 offset:41984
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[170:175], v[38:43], a[164:167], v11, v16 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_mov_b32 m0, s35
	ds_read_b64 v[180:181], v12 offset:55808
	buffer_load_dwordx4 v1, s[20:23], s2 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[170:175], v[32:37], a[160:163], v11, v16 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[158:163], v[32:37], a[192:195], v11, v16 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[170:173], v9 offset:43008
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[158:163], v[38:43], a[196:199], v11, v16 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[174:175], v12 offset:56320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[158:163], v[46:51], a[200:203], v11, v16 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[158:163], v[52:57], a[204:207], v11, v16 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[20:23], s2 offen offset:1024 lds
	ds_read_b128 v[188:191], v9 offset:44032
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[158:163], v[58:63], a[208:211], v11, v17 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[192:193], v12 offset:56832
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[158:163], v[68:73], a[212:215], v11, v17 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[158:163], v[80:85], a[216:219], v11, v17 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_i32 s2, s3, s12
	s_mov_b32 m0, s38
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[158:163], v[98:103], a[220:223], v11, v17 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dword v2, s[24:27], s2 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[182:187], v[98:103], a[252:255], v11, v17 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[182:187], v[80:85], a[248:251], v11, v17 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[182:187], v[68:73], a[244:247], v11, v17 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s2, s29, 0x7fc000
	s_mov_b32 m0, s34
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[182:187], v[58:63], a[240:243], v11, v17 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dword v2, s[8:11], s2 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[182:187], v[52:57], a[236:239], v11, v16 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[182:187], v[46:51], a[232:235], v11, v16 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[182:187], v[38:43], a[228:231], v11, v16 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[182:187], v[32:37], a[224:227], v11, v16 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[74:79], v[24:29], a[0:3], v20, v22 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[74:79], v[86:91], a[4:7], v20, v22 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[74:79], v[92:97], a[8:11], v20, v22 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[8:9], v5 offset:24576
	ds_read_b64 v[2:3], v3 offset:50176
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[74:79], v[104:109], a[12:15], v20, v22 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[74:79], v[112:117], a[16:19], v20, v23 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[74:79], v[118:123], a[20:23], v20, v23 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[74:79], v[124:129], a[24:27], v20, v23 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[30:33], v6
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[74:79], v[130:135], a[28:31], v20, v23 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[34:35], v4 offset:16384
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[136:141], v[130:135], a[60:63], v20, v23 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[136:141], v[124:129], a[56:59], v20, v23 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[36:39], v18 offset:25600
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[136:141], v[118:123], a[52:55], v20, v23 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[40:41], v7 offset:41984
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[136:141], v[112:117], a[48:51], v20, v23 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[136:141], v[104:109], a[44:47], v20, v22 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[42:45], v18 offset:26624
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[136:141], v[92:97], a[40:43], v20, v22 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[46:47], v7 offset:42496
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[136:141], v[86:91], a[36:39], v20, v22 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[136:141], v[24:29], a[32:35], v20, v22 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[48:51], v18 offset:27648
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[142:147], v[24:29], a[64:67], v20, v22 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[52:53], v7 offset:43008
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[142:147], v[86:91], a[68:71], v20, v22 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[142:147], v[92:97], a[72:75], v20, v22 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[54:57], v18 offset:28672
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[142:147], v[104:109], a[76:79], v20, v22 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[58:59], v7 offset:43520
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[142:147], v[112:117], a[80:83], v20, v23 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[142:147], v[118:123], a[84:87], v20, v23 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[60:63], v18 offset:29696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[142:147], v[124:129], a[88:91], v20, v23 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[64:65], v7 offset:44032
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[142:147], v[130:135], a[92:95], v20, v23 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[148:153], v[130:135], a[124:127], v20, v23 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[66:69], v18 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[148:153], v[124:129], a[120:123], v20, v23 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[70:71], v7 offset:44544
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[148:153], v[118:123], a[116:119], v20, v23 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[148:153], v[112:117], a[112:115], v20, v23 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[72:75], v18 offset:31744
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[148:153], v[104:109], a[108:111], v20, v22 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[76:77], v7 offset:45056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[148:153], v[92:97], a[104:107], v20, v22 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[148:153], v[86:91], a[100:103], v20, v22 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[78:81], v18 offset:32768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[148:153], v[24:29], a[96:99], v20, v22 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[82:83], v7 offset:45568
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[164:169], v[24:29], a[128:131], v21, v22 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[164:169], v[86:91], a[132:135], v21, v22 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[98:101], v6 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[164:169], v[92:97], a[136:139], v21, v22 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[102:103], v4 offset:16896
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[164:169], v[104:109], a[140:143], v21, v22 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[164:169], v[112:117], a[144:147], v21, v23 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[136:139], v6 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[164:169], v[118:123], a[148:151], v21, v23 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[140:141], v4 offset:17408
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[164:169], v[124:129], a[152:155], v21, v23 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[164:169], v[130:135], a[156:159], v21, v23 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[142:145], v6 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[176:181], v[130:135], a[188:191], v21, v23 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[146:147], v4 offset:17920
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[176:181], v[124:129], a[184:187], v21, v23 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[176:181], v[118:123], a[180:183], v21, v23 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[148:151], v6 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[176:181], v[112:117], a[176:179], v21, v23 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[152:153], v4 offset:18432
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[176:181], v[104:109], a[172:175], v21, v22 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[176:181], v[92:97], a[168:171], v21, v22 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[154:157], v6 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[176:181], v[86:91], a[164:167], v21, v22 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[158:159], v4 offset:18944
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[176:181], v[24:29], a[160:163], v21, v22 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[170:175], v[24:29], a[192:195], v21, v22 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[160:163], v6 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[170:175], v[86:91], a[196:199], v21, v22 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[164:165], v4 offset:19456
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[170:175], v[92:97], a[200:203], v21, v22 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[170:175], v[104:109], a[204:207], v21, v22 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[176:179], v6 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[170:175], v[112:117], a[208:211], v21, v23 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[180:181], v4 offset:19968
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[170:175], v[118:123], a[212:215], v21, v23 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[170:175], v[124:129], a[216:219], v21, v23 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[170:175], v[130:135], a[220:223], v21, v23 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[188:193], v[130:135], a[252:255], v21, v23 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[188:193], v[124:129], a[248:251], v21, v23 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[188:193], v[118:123], a[244:247], v21, v23 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[188:193], v[112:117], a[240:243], v21, v23 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[188:193], v[104:109], a[236:239], v21, v22 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[188:193], v[92:97], a[232:235], v21, v22 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[188:193], v[86:91], a[228:231], v21, v22 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[188:193], v[24:29], a[224:227], v21, v22 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[30:35], v[36:41], a[0:3], v8, v2 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[30:35], v[42:47], a[4:7], v8, v2 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[30:35], v[48:53], a[8:11], v8, v2 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[4:5], v19 offset:10240
	ds_read_b64 v[10:11], v110 offset:35840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[30:35], v[54:59], a[12:15], v8, v2 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[30:35], v[60:65], a[16:19], v8, v3 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[30:35], v[66:71], a[20:23], v8, v3 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[30:35], v[72:77], a[24:27], v8, v3 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[16:19], v6 offset:51200
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[30:35], v[78:83], a[28:31], v8, v3 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[20:21], v12 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[98:103], v[78:83], a[60:63], v8, v3 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[98:103], v[72:77], a[56:59], v8, v3 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[22:25], v13 offset:11264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[98:103], v[66:71], a[52:55], v8, v3 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[26:27], v14 offset:27648
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[98:103], v[60:65], a[48:51], v8, v3 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[98:103], v[54:59], a[44:47], v8, v2 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[28:31], v13 offset:12288
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[98:103], v[48:53], a[40:43], v8, v2 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[32:33], v14 offset:28160
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[98:103], v[42:47], a[36:39], v8, v2 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[98:103], v[36:41], a[32:35], v8, v2 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[84:87], v13 offset:13312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[136:141], v[36:41], a[64:67], v8, v2 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[88:89], v14 offset:28672
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[136:141], v[42:47], a[68:71], v8, v2 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[136:141], v[48:53], a[72:75], v8, v2 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[90:93], v13 offset:14336
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[136:141], v[54:59], a[76:79], v8, v2 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[94:95], v14 offset:29184
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[136:141], v[60:65], a[80:83], v8, v3 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[136:141], v[66:71], a[84:87], v8, v3 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[96:99], v13 offset:15360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[136:141], v[72:77], a[88:91], v8, v3 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[100:101], v14 offset:29696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[136:141], v[78:83], a[92:95], v8, v3 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[142:147], v[78:83], a[124:127], v8, v3 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[102:105], v13 offset:16384
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[142:147], v[72:77], a[120:123], v8, v3 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[106:107], v14 offset:30208
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[142:147], v[66:71], a[116:119], v8, v3 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[142:147], v[60:65], a[112:115], v8, v3 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[108:111], v13 offset:17408
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[142:147], v[54:59], a[108:111], v8, v2 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b64 v[112:113], v14 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[142:147], v[48:53], a[104:107], v8, v2 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[142:147], v[42:47], a[100:103], v8, v2 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[114:117], v13 offset:18432
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[142:147], v[36:41], a[96:99], v8, v2 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[118:119], v14 offset:31232
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[148:153], v[36:41], a[128:131], v9, v2 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[148:153], v[42:47], a[132:135], v9, v2 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[120:123], v6 offset:52224
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[148:153], v[48:53], a[136:139], v9, v2 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[124:125], v12 offset:2560
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[148:153], v[54:59], a[140:143], v9, v2 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[148:153], v[60:65], a[144:147], v9, v3 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[126:129], v6 offset:53248
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[148:153], v[66:71], a[148:151], v9, v3 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[130:131], v12 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[148:153], v[72:77], a[152:155], v9, v3 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[148:153], v[78:83], a[156:159], v9, v3 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[132:135], v6 offset:54272
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[154:159], v[78:83], a[188:191], v9, v3 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b64 v[136:137], v12 offset:3584
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[154:159], v[72:77], a[184:187], v9, v3 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[154:159], v[66:71], a[180:183], v9, v3 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[138:141], v6 offset:55296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[154:159], v[60:65], a[176:179], v9, v3 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[142:143], v12 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[154:159], v[54:59], a[172:175], v9, v2 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[154:159], v[48:53], a[168:171], v9, v2 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[144:147], v6 offset:56320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[154:159], v[42:47], a[164:167], v9, v2 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[148:149], v12 offset:4608
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[154:159], v[36:41], a[160:163], v9, v2 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[160:165], v[36:41], a[192:195], v9, v2 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[150:153], v6 offset:57344
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[160:165], v[42:47], a[196:199], v9, v2 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[154:155], v12 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[160:165], v[48:53], a[200:203], v9, v2 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[160:165], v[54:59], a[204:207], v9, v2 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[166:169], v6 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[160:165], v[60:65], a[208:211], v9, v3 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b64 v[170:171], v12 offset:5632
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[160:165], v[66:71], a[212:215], v9, v3 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[160:165], v[72:77], a[216:219], v9, v3 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[160:165], v[78:83], a[220:223], v9, v3 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[176:181], v[78:83], a[252:255], v9, v3 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[176:181], v[72:77], a[248:251], v9, v3 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[176:181], v[66:71], a[244:247], v9, v3 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[176:181], v[60:65], a[240:243], v9, v3 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[176:181], v[54:59], a[236:239], v9, v2 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[176:181], v[48:53], a[232:235], v9, v2 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[176:181], v[42:47], a[228:231], v9, v2 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[176:181], v[36:41], a[224:227], v9, v2 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[16:21], v[22:27], a[0:3], v4, v10 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[16:21], v[28:33], a[4:7], v4, v10 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[16:21], v[84:89], a[8:11], v4, v10 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[16:21], v[90:95], a[12:15], v4, v10 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[16:21], v[96:101], a[16:19], v4, v11 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[16:21], v[102:107], a[20:23], v4, v11 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[16:21], v[108:113], a[24:27], v4, v11 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[16:21], v[114:119], a[28:31], v4, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[120:125], v[114:119], a[60:63], v4, v11 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[120:125], v[108:113], a[56:59], v4, v11 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[120:125], v[102:107], a[52:55], v4, v11 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[120:125], v[96:101], a[48:51], v4, v11 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[120:125], v[90:95], a[44:47], v4, v10 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[120:125], v[84:89], a[40:43], v4, v10 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[120:125], v[28:33], a[36:39], v4, v10 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[120:125], v[22:27], a[32:35], v4, v10 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[126:131], v[22:27], a[64:67], v4, v10 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[126:131], v[28:33], a[68:71], v4, v10 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[126:131], v[84:89], a[72:75], v4, v10 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[126:131], v[90:95], a[76:79], v4, v10 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[126:131], v[96:101], a[80:83], v4, v11 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[126:131], v[102:107], a[84:87], v4, v11 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[126:131], v[108:113], a[88:91], v4, v11 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[126:131], v[114:119], a[92:95], v4, v11 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[132:137], v[114:119], a[124:127], v4, v11 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[132:137], v[108:113], a[120:123], v4, v11 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[132:137], v[102:107], a[116:119], v4, v11 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[132:137], v[96:101], a[112:115], v4, v11 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[132:137], v[90:95], a[108:111], v4, v10 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[132:137], v[84:89], a[104:107], v4, v10 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[132:137], v[28:33], a[100:103], v4, v10 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[132:137], v[22:27], a[96:99], v4, v10 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[138:143], v[22:27], a[128:131], v5, v10 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[138:143], v[28:33], a[132:135], v5, v10 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[138:143], v[84:89], a[136:139], v5, v10 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[138:143], v[90:95], a[140:143], v5, v10 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[138:143], v[96:101], a[144:147], v5, v11 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[138:143], v[102:107], a[148:151], v5, v11 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[138:143], v[108:113], a[152:155], v5, v11 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[138:143], v[114:119], a[156:159], v5, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[144:149], v[114:119], a[188:191], v5, v11 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[144:149], v[108:113], a[184:187], v5, v11 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[144:149], v[102:107], a[180:183], v5, v11 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[144:149], v[96:101], a[176:179], v5, v11 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[144:149], v[90:95], a[172:175], v5, v10 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[144:149], v[84:89], a[168:171], v5, v10 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[144:149], v[28:33], a[164:167], v5, v10 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[144:149], v[22:27], a[160:163], v5, v10 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[150:155], v[22:27], a[192:195], v5, v10 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[150:155], v[28:33], a[196:199], v5, v10 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[150:155], v[84:89], a[200:203], v5, v10 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[150:155], v[90:95], a[204:207], v5, v10 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[150:155], v[96:101], a[208:211], v5, v11 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[150:155], v[102:107], a[212:215], v5, v11 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[150:155], v[108:113], a[216:219], v5, v11 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[150:155], v[114:119], a[220:223], v5, v11 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[166:171], v[114:119], a[252:255], v5, v11 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[166:171], v[108:113], a[248:251], v5, v11 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[166:171], v[102:107], a[244:247], v5, v11 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[166:171], v[96:101], a[240:243], v5, v11 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[166:171], v[90:95], a[236:239], v5, v10 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[166:171], v[84:89], a[232:235], v5, v10 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[166:171], v[28:33], a[228:231], v5, v10 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[166:171], v[22:27], a[224:227], v5, v10 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	s_waitcnt vmcnt(0)
	s_and_b32 s1, s1, 0xffff
	s_lshl_b32 s2, s18, 13
	s_lshl_b32 s4, s19, 1
	s_add_i32 s5, s2, s4
	v_and_b32_e32 v1, 15, v0
	v_and_b32_e32 v2, 1, v1
	v_cmp_eq_i32_e64 s[6:7], v2, 1
	v_lshrrev_b32_e32 v0, 4, v0
	v_lshlrev_b32_e32 v3, 1, v2
	v_accvgpr_read_b32 v4, a0
	v_accvgpr_read_b32 v5, a1
	v_cvt_pk_bf16_f32 v4, v4, v5
	v_accvgpr_read_b32 v5, a2
	v_accvgpr_read_b32 v6, a3
	v_cvt_pk_bf16_f32 v5, v5, v6
	v_accvgpr_read_b32 v6, a4
	v_accvgpr_read_b32 v7, a5
	v_cvt_pk_bf16_f32 v6, v6, v7
	v_accvgpr_read_b32 v7, a6
	v_accvgpr_read_b32 v8, a7
	v_cvt_pk_bf16_f32 v7, v7, v8
	v_accvgpr_read_b32 v8, a8
	v_accvgpr_read_b32 v9, a9
	v_cvt_pk_bf16_f32 v8, v8, v9
	v_accvgpr_read_b32 v9, a10
	v_accvgpr_read_b32 v10, a11
	v_cvt_pk_bf16_f32 v9, v9, v10
	v_lshl_add_u32 v0, v0, 2, v3
	v_mov_b32_dpp v3, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v10, v5 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v11, v6 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v12, v7 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v13, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v14, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_sub_u32_e32 v1, v1, v2
	v_cndmask_b32_e64 v2, v4, v10, s[6:7]
	v_cndmask_b32_e64 v3, v3, v5, s[6:7]
	v_lshlrev_b32_e32 v0, 13, v0
	v_cndmask_b32_e64 v4, v6, v12, s[6:7]
	v_cndmask_b32_e64 v5, v11, v7, s[6:7]
	v_cndmask_b32_e64 v6, v8, v14, s[6:7]
	v_cndmask_b32_e64 v7, v13, v9, s[6:7]
	v_lshlrev_b32_e32 v8, 16, v3
	v_lshrrev_b32_e32 v9, 16, v2
	v_lshl_add_u32 v0, v1, 1, v0
	v_lshlrev_b32_e32 v1, 16, v5
	v_lshrrev_b32_e32 v10, 16, v4
	v_lshlrev_b32_e32 v11, 16, v7
	v_lshrrev_b32_e32 v12, 16, v6
	s_mov_b32 s2, 0xffff
	v_and_or_b32 v2, s2, v2, v8
	s_mov_b32 s2, 0xffff0000
	v_and_or_b32 v3, s2, v3, v9
	v_add_u32_e32 v8, 0x2000, v0
	s_mov_b32 s2, 0xffff
	v_and_or_b32 v1, s2, v4, v1
	s_mov_b32 s2, 0xffff0000
	v_and_or_b32 v4, s2, v5, v10
	v_add_u32_e32 v5, 32, v0
	v_add_u32_e32 v9, 0x2020, v0
	s_mov_b32 s2, 0xffff
	v_and_or_b32 v6, s2, v6, v11
	s_mov_b32 s2, 0xffff0000
	v_and_or_b32 v7, s2, v7, v12
	v_add_u32_e32 v10, 64, v0
	v_add_u32_e32 v11, 0x2040, v0
	s_mov_b32 s2, 0x7ffffff0
	s_mov_b32 s3, 0x31014000
	v_accvgpr_read_b32 v12, a12
	v_accvgpr_read_b32 v13, a13
	v_cvt_pk_bf16_f32 v12, v12, v13
	v_accvgpr_read_b32 v13, a14
	v_accvgpr_read_b32 v14, a15
	v_cvt_pk_bf16_f32 v13, v13, v14
	v_accvgpr_read_b32 v14, a16
	v_accvgpr_read_b32 v15, a17
	v_cvt_pk_bf16_f32 v14, v14, v15
	v_accvgpr_read_b32 v15, a18
	v_accvgpr_read_b32 v16, a19
	v_cvt_pk_bf16_f32 v15, v15, v16
	v_mov_b32_dpp v16, v12 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v17, v13 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v18, v14 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v19, v15 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v12, v12, v17, s[6:7]
	v_cndmask_b32_e64 v13, v16, v13, s[6:7]
	v_cndmask_b32_e64 v14, v14, v19, s[6:7]
	v_lshlrev_b32_e32 v16, 16, v13
	v_lshrrev_b32_e32 v17, 16, v12
	s_mov_b32 s8, 0xffff
	v_and_or_b32 v12, s8, v12, v16
	s_mov_b32 s8, 0xffff0000
	v_and_or_b32 v13, s8, v13, v17
	v_add_u32_e32 v16, 0x60, v0
	v_add_u32_e32 v17, 0x2060, v0
	v_cndmask_b32_e64 v15, v18, v15, s[6:7]
	buffer_store_dword v2, v0, s[0:3], s5 offen
	v_lshlrev_b32_e32 v2, 16, v15
	s_mov_b32 s8, 0xffff
	v_and_or_b32 v2, s8, v14, v2
	buffer_store_dword v3, v8, s[0:3], s5 offen
	v_accvgpr_read_b32 v3, a20
	v_accvgpr_read_b32 v18, a21
	v_cvt_pk_bf16_f32 v18, v3, v18
	v_accvgpr_read_b32 v3, a22
	v_accvgpr_read_b32 v19, a23
	v_cvt_pk_bf16_f32 v19, v3, v19
	v_mov_b32_dpp v3, v18 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v20, v19 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v18, v18, v20, s[6:7]
	v_lshrrev_b32_e32 v14, 16, v14
	s_mov_b32 s8, 0xffff0000
	v_and_or_b32 v14, s8, v15, v14
	v_add_u32_e32 v15, 0x80, v0
	v_add_u32_e32 v20, 0x2080, v0
	v_cndmask_b32_e64 v3, v3, v19, s[6:7]
	buffer_store_dword v1, v5, s[0:3], s5 offen
	v_lshlrev_b32_e32 v1, 16, v3
	s_mov_b32 s8, 0xffff
	v_and_or_b32 v1, s8, v18, v1
	buffer_store_dword v4, v9, s[0:3], s5 offen
	v_accvgpr_read_b32 v4, a24
	v_accvgpr_read_b32 v19, a25
	v_cvt_pk_bf16_f32 v22, v4, v19
	v_accvgpr_read_b32 v4, a26
	v_accvgpr_read_b32 v19, a27
	v_cvt_pk_bf16_f32 v23, v4, v19
	v_mov_b32_dpp v4, v22 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v19, v23 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v19, v22, v19, s[6:7]
	v_lshrrev_b32_e32 v18, 16, v18
	s_mov_b32 s8, 0xffff0000
	v_and_or_b32 v3, s8, v3, v18
	v_add_u32_e32 v18, 0xa0, v0
	v_add_u32_e32 v21, 0x20a0, v0
	v_cndmask_b32_e64 v4, v4, v23, s[6:7]
	buffer_store_dword v6, v10, s[0:3], s5 offen
	v_lshlrev_b32_e32 v6, 16, v4
	s_mov_b32 s8, 0xffff
	v_and_or_b32 v6, s8, v19, v6
	buffer_store_dword v7, v11, s[0:3], s5 offen
	v_accvgpr_read_b32 v7, a28
	v_accvgpr_read_b32 v22, a29
	v_cvt_pk_bf16_f32 v22, v7, v22
	v_accvgpr_read_b32 v7, a30
	v_accvgpr_read_b32 v23, a31
	v_cvt_pk_bf16_f32 v23, v7, v23
	v_mov_b32_dpp v7, v22 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v24, v23 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v22, v22, v24, s[6:7]
	v_lshrrev_b32_e32 v19, 16, v19
	s_mov_b32 s8, 0xffff0000
	v_and_or_b32 v4, s8, v4, v19
	v_add_u32_e32 v19, 0xc0, v0
	v_add_u32_e32 v24, 0x20c0, v0
	v_cndmask_b32_e64 v7, v7, v23, s[6:7]
	buffer_store_dword v12, v16, s[0:3], s5 offen
	v_lshlrev_b32_e32 v12, 16, v7
	s_mov_b32 s8, 0xffff
	v_and_or_b32 v12, s8, v22, v12
	buffer_store_dword v13, v17, s[0:3], s5 offen
	s_add_i32 s8, s18, 16
	s_lshl_b32 s8, s8, 13
	s_add_i32 s8, s4, s8
	v_accvgpr_read_b32 v13, a32
	v_accvgpr_read_b32 v23, a33
	v_cvt_pk_bf16_f32 v26, v13, v23
	v_accvgpr_read_b32 v13, a34
	v_accvgpr_read_b32 v23, a35
	v_cvt_pk_bf16_f32 v27, v13, v23
	v_lshrrev_b32_e32 v13, 16, v22
	s_mov_b32 s9, 0xffff0000
	v_and_or_b32 v7, s9, v7, v13
	v_add_u32_e32 v13, 0xe0, v0
	v_add_u32_e32 v22, 0x20e0, v0
	v_mov_b32_dpp v23, v26 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	buffer_store_dword v2, v15, s[0:3], s5 offen
	v_mov_b32_dpp v2, v27 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v2, v26, v2, s[6:7]
	v_cndmask_b32_e64 v23, v23, v27, s[6:7]
	buffer_store_dword v14, v20, s[0:3], s5 offen
	v_accvgpr_read_b32 v14, a36
	v_accvgpr_read_b32 v25, a37
	v_cvt_pk_bf16_f32 v26, v14, v25
	v_accvgpr_read_b32 v14, a38
	v_accvgpr_read_b32 v25, a39
	v_cvt_pk_bf16_f32 v27, v14, v25
	v_mov_b32_dpp v14, v26 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v25, v27 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshlrev_b32_e32 v28, 16, v23
	v_lshrrev_b32_e32 v29, 16, v2
	s_mov_b32 s9, 0xffff
	v_and_or_b32 v2, s9, v2, v28
	s_mov_b32 s9, 0xffff0000
	v_and_or_b32 v23, s9, v23, v29
	v_cndmask_b32_e64 v25, v26, v25, s[6:7]
	buffer_store_dword v1, v18, s[0:3], s5 offen
	v_cndmask_b32_e64 v1, v14, v27, s[6:7]
	v_lshlrev_b32_e32 v14, 16, v1
	s_mov_b32 s9, 0xffff
	buffer_store_dword v3, v21, s[0:3], s5 offen
	v_accvgpr_read_b32 v3, a40
	v_accvgpr_read_b32 v26, a41
	v_cvt_pk_bf16_f32 v26, v3, v26
	v_accvgpr_read_b32 v3, a42
	v_accvgpr_read_b32 v27, a43
	v_cvt_pk_bf16_f32 v27, v3, v27
	v_mov_b32_dpp v3, v26 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v28, v27 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v26, v26, v28, s[6:7]
	v_cndmask_b32_e64 v3, v3, v27, s[6:7]
	v_lshrrev_b32_e32 v27, 16, v25
	v_and_or_b32 v14, s9, v25, v14
	s_mov_b32 s9, 0xffff0000
	v_and_or_b32 v1, s9, v1, v27
	v_lshlrev_b32_e32 v25, 16, v3
	buffer_store_dword v6, v19, s[0:3], s5 offen
	s_mov_b32 s9, 0xffff
	v_and_or_b32 v6, s9, v26, v25
	v_lshrrev_b32_e32 v25, 16, v26
	buffer_store_dword v4, v24, s[0:3], s5 offen
	v_accvgpr_read_b32 v4, a44
	v_accvgpr_read_b32 v26, a45
	v_cvt_pk_bf16_f32 v26, v4, v26
	v_accvgpr_read_b32 v4, a46
	v_accvgpr_read_b32 v27, a47
	v_cvt_pk_bf16_f32 v27, v4, v27
	v_mov_b32_dpp v4, v26 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v28, v27 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v26, v26, v28, s[6:7]
	v_cndmask_b32_e64 v4, v4, v27, s[6:7]
	v_lshlrev_b32_e32 v27, 16, v4
	s_mov_b32 s9, 0xffff0000
	v_and_or_b32 v3, s9, v3, v25
	s_mov_b32 s9, 0xffff
	v_and_or_b32 v25, s9, v26, v27
	buffer_store_dword v12, v13, s[0:3], s5 offen
	v_lshrrev_b32_e32 v12, 16, v26
	s_mov_b32 s9, 0xffff0000
	v_and_or_b32 v4, s9, v4, v12
	buffer_store_dword v7, v22, s[0:3], s5 offen
	v_accvgpr_read_b32 v7, a48
	v_accvgpr_read_b32 v12, a49
	v_cvt_pk_bf16_f32 v26, v7, v12
	v_accvgpr_read_b32 v7, a50
	v_accvgpr_read_b32 v12, a51
	v_cvt_pk_bf16_f32 v27, v7, v12
	v_mov_b32_dpp v7, v26 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v12, v27 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v12, v26, v12, s[6:7]
	v_cndmask_b32_e64 v7, v7, v27, s[6:7]
	v_lshlrev_b32_e32 v26, 16, v7
	v_lshrrev_b32_e32 v27, 16, v12
	s_mov_b32 s5, 0xffff
	v_and_or_b32 v12, s5, v12, v26
	s_mov_b32 s5, 0xffff0000
	v_and_or_b32 v7, s5, v7, v27
	buffer_store_dword v2, v0, s[0:3], s8 offen
	buffer_store_dword v23, v8, s[0:3], s8 offen
	v_accvgpr_read_b32 v2, a52
	v_accvgpr_read_b32 v23, a53
	v_cvt_pk_bf16_f32 v26, v2, v23
	v_accvgpr_read_b32 v2, a54
	v_accvgpr_read_b32 v23, a55
	v_cvt_pk_bf16_f32 v27, v2, v23
	v_mov_b32_dpp v2, v26 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v23, v27 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v23, v26, v23, s[6:7]
	v_cndmask_b32_e64 v2, v2, v27, s[6:7]
	v_lshlrev_b32_e32 v26, 16, v2
	v_lshrrev_b32_e32 v27, 16, v23
	s_mov_b32 s5, 0xffff
	v_and_or_b32 v23, s5, v23, v26
	s_mov_b32 s5, 0xffff0000
	v_and_or_b32 v2, s5, v2, v27
	buffer_store_dword v14, v5, s[0:3], s8 offen
	buffer_store_dword v1, v9, s[0:3], s8 offen
	v_accvgpr_read_b32 v1, a56
	v_accvgpr_read_b32 v14, a57
	v_cvt_pk_bf16_f32 v26, v1, v14
	v_accvgpr_read_b32 v1, a58
	v_accvgpr_read_b32 v14, a59
	v_cvt_pk_bf16_f32 v27, v1, v14
	v_mov_b32_dpp v1, v26 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v14, v27 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v14, v26, v14, s[6:7]
	v_cndmask_b32_e64 v1, v1, v27, s[6:7]
	v_lshlrev_b32_e32 v26, 16, v1
	v_lshrrev_b32_e32 v27, 16, v14
	s_mov_b32 s5, 0xffff
	v_and_or_b32 v14, s5, v14, v26
	s_mov_b32 s5, 0xffff0000
	v_and_or_b32 v1, s5, v1, v27
	buffer_store_dword v6, v10, s[0:3], s8 offen
	buffer_store_dword v3, v11, s[0:3], s8 offen
	v_accvgpr_read_b32 v3, a60
	v_accvgpr_read_b32 v6, a61
	v_cvt_pk_bf16_f32 v26, v3, v6
	v_accvgpr_read_b32 v3, a62
	v_accvgpr_read_b32 v6, a63
	v_cvt_pk_bf16_f32 v27, v3, v6
	v_mov_b32_dpp v3, v26 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v6, v27 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v6, v26, v6, s[6:7]
	v_cndmask_b32_e64 v3, v3, v27, s[6:7]
	v_lshlrev_b32_e32 v26, 16, v3
	v_lshrrev_b32_e32 v27, 16, v6
	s_mov_b32 s5, 0xffff
	v_and_or_b32 v6, s5, v6, v26
	s_mov_b32 s5, 0xffff0000
	v_and_or_b32 v3, s5, v3, v27
	buffer_store_dword v25, v16, s[0:3], s8 offen
	buffer_store_dword v4, v17, s[0:3], s8 offen
	s_add_i32 s5, s18, 32
	s_lshl_b32 s5, s5, 13
	s_add_i32 s5, s4, s5
	v_accvgpr_read_b32 v4, a64
	v_accvgpr_read_b32 v25, a65
	v_cvt_pk_bf16_f32 v26, v4, v25
	v_accvgpr_read_b32 v4, a66
	v_accvgpr_read_b32 v25, a67
	v_cvt_pk_bf16_f32 v27, v4, v25
	v_mov_b32_dpp v4, v26 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v25, v27 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v25, v26, v25, s[6:7]
	v_cndmask_b32_e64 v4, v4, v27, s[6:7]
	v_lshlrev_b32_e32 v26, 16, v4
	s_mov_b32 s9, 0xffff
	v_and_or_b32 v26, s9, v25, v26
	v_lshrrev_b32_e32 v25, 16, v25
	buffer_store_dword v12, v15, s[0:3], s8 offen
	s_mov_b32 s9, 0xffff0000
	v_and_or_b32 v4, s9, v4, v25
	buffer_store_dword v7, v20, s[0:3], s8 offen
	v_accvgpr_read_b32 v7, a68
	v_accvgpr_read_b32 v12, a69
	v_cvt_pk_bf16_f32 v28, v7, v12
	v_accvgpr_read_b32 v7, a70
	v_accvgpr_read_b32 v12, a71
	v_cvt_pk_bf16_f32 v29, v7, v12
	v_mov_b32_dpp v7, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v12, v29 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v12, v28, v12, s[6:7]
	v_cndmask_b32_e64 v7, v7, v29, s[6:7]
	v_lshlrev_b32_e32 v25, 16, v7
	s_mov_b32 s9, 0xffff
	v_and_or_b32 v25, s9, v12, v25
	v_lshrrev_b32_e32 v12, 16, v12
	buffer_store_dword v23, v18, s[0:3], s8 offen
	s_mov_b32 s9, 0xffff0000
	v_and_or_b32 v7, s9, v7, v12
	buffer_store_dword v2, v21, s[0:3], s8 offen
	v_accvgpr_read_b32 v2, a72
	v_accvgpr_read_b32 v12, a73
	v_cvt_pk_bf16_f32 v28, v2, v12
	v_accvgpr_read_b32 v2, a74
	v_accvgpr_read_b32 v12, a75
	v_cvt_pk_bf16_f32 v29, v2, v12
	v_mov_b32_dpp v2, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v12, v29 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v12, v28, v12, s[6:7]
	v_cndmask_b32_e64 v2, v2, v29, s[6:7]
	v_lshlrev_b32_e32 v23, 16, v2
	s_mov_b32 s9, 0xffff
	v_and_or_b32 v23, s9, v12, v23
	v_lshrrev_b32_e32 v12, 16, v12
	buffer_store_dword v14, v19, s[0:3], s8 offen
	s_mov_b32 s9, 0xffff0000
	v_and_or_b32 v2, s9, v2, v12
	buffer_store_dword v1, v24, s[0:3], s8 offen
	v_accvgpr_read_b32 v1, a76
	v_accvgpr_read_b32 v12, a77
	v_cvt_pk_bf16_f32 v28, v1, v12
	v_accvgpr_read_b32 v1, a78
	v_accvgpr_read_b32 v12, a79
	v_cvt_pk_bf16_f32 v29, v1, v12
	v_mov_b32_dpp v1, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v12, v29 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v12, v28, v12, s[6:7]
	v_cndmask_b32_e64 v1, v1, v29, s[6:7]
	v_lshlrev_b32_e32 v14, 16, v1
	s_mov_b32 s9, 0xffff
	v_and_or_b32 v14, s9, v12, v14
	v_lshrrev_b32_e32 v12, 16, v12
	buffer_store_dword v6, v13, s[0:3], s8 offen
	s_mov_b32 s9, 0xffff0000
	v_and_or_b32 v1, s9, v1, v12
	buffer_store_dword v3, v22, s[0:3], s8 offen
	v_accvgpr_read_b32 v3, a80
	v_accvgpr_read_b32 v6, a81
	v_cvt_pk_bf16_f32 v28, v3, v6
	v_accvgpr_read_b32 v3, a82
	v_accvgpr_read_b32 v6, a83
	v_cvt_pk_bf16_f32 v29, v3, v6
	v_mov_b32_dpp v3, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v6, v29 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v6, v28, v6, s[6:7]
	v_cndmask_b32_e64 v3, v3, v29, s[6:7]
	v_lshlrev_b32_e32 v12, 16, v3
	v_lshrrev_b32_e32 v27, 16, v6
	s_mov_b32 s8, 0xffff
	v_and_or_b32 v6, s8, v6, v12
	s_mov_b32 s8, 0xffff0000
	v_and_or_b32 v3, s8, v3, v27
	buffer_store_dword v26, v0, s[0:3], s5 offen
	buffer_store_dword v4, v8, s[0:3], s5 offen
	v_accvgpr_read_b32 v4, a84
	v_accvgpr_read_b32 v12, a85
	v_cvt_pk_bf16_f32 v26, v4, v12
	v_accvgpr_read_b32 v4, a86
	v_accvgpr_read_b32 v12, a87
	v_cvt_pk_bf16_f32 v27, v4, v12
	v_mov_b32_dpp v4, v26 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v12, v27 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v12, v26, v12, s[6:7]
	v_cndmask_b32_e64 v4, v4, v27, s[6:7]
	v_lshlrev_b32_e32 v26, 16, v4
	v_lshrrev_b32_e32 v27, 16, v12
	s_mov_b32 s8, 0xffff
	v_and_or_b32 v12, s8, v12, v26
	s_mov_b32 s8, 0xffff0000
	v_and_or_b32 v4, s8, v4, v27
	buffer_store_dword v25, v5, s[0:3], s5 offen
	buffer_store_dword v7, v9, s[0:3], s5 offen
	v_accvgpr_read_b32 v7, a88
	v_accvgpr_read_b32 v25, a89
	v_cvt_pk_bf16_f32 v26, v7, v25
	v_accvgpr_read_b32 v7, a90
	v_accvgpr_read_b32 v25, a91
	v_cvt_pk_bf16_f32 v27, v7, v25
	v_mov_b32_dpp v7, v26 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v25, v27 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v25, v26, v25, s[6:7]
	v_cndmask_b32_e64 v7, v7, v27, s[6:7]
	v_lshlrev_b32_e32 v26, 16, v7
	v_lshrrev_b32_e32 v27, 16, v25
	s_mov_b32 s8, 0xffff
	v_and_or_b32 v25, s8, v25, v26
	s_mov_b32 s8, 0xffff0000
	v_and_or_b32 v7, s8, v7, v27
	buffer_store_dword v23, v10, s[0:3], s5 offen
	buffer_store_dword v2, v11, s[0:3], s5 offen
	v_accvgpr_read_b32 v2, a92
	v_accvgpr_read_b32 v23, a93
	v_cvt_pk_bf16_f32 v26, v2, v23
	v_accvgpr_read_b32 v2, a94
	v_accvgpr_read_b32 v23, a95
	v_cvt_pk_bf16_f32 v27, v2, v23
	v_mov_b32_dpp v2, v26 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v23, v27 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v23, v26, v23, s[6:7]
	v_cndmask_b32_e64 v2, v2, v27, s[6:7]
	v_lshlrev_b32_e32 v26, 16, v2
	v_lshrrev_b32_e32 v27, 16, v23
	s_mov_b32 s8, 0xffff
	v_and_or_b32 v23, s8, v23, v26
	s_mov_b32 s8, 0xffff0000
	v_and_or_b32 v2, s8, v2, v27
	buffer_store_dword v14, v16, s[0:3], s5 offen
	buffer_store_dword v1, v17, s[0:3], s5 offen
	s_add_i32 s8, s18, 48
	s_lshl_b32 s8, s8, 13
	s_add_i32 s8, s4, s8
	v_accvgpr_read_b32 v1, a96
	v_accvgpr_read_b32 v14, a97
	v_cvt_pk_bf16_f32 v26, v1, v14
	v_accvgpr_read_b32 v1, a98
	v_accvgpr_read_b32 v14, a99
	v_cvt_pk_bf16_f32 v27, v1, v14
	v_mov_b32_dpp v1, v26 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v14, v27 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v14, v26, v14, s[6:7]
	v_cndmask_b32_e64 v1, v1, v27, s[6:7]
	v_lshlrev_b32_e32 v26, 16, v1
	s_mov_b32 s9, 0xffff
	v_and_or_b32 v26, s9, v14, v26
	v_lshrrev_b32_e32 v14, 16, v14
	buffer_store_dword v6, v15, s[0:3], s5 offen
	s_mov_b32 s9, 0xffff0000
	v_and_or_b32 v1, s9, v1, v14
	buffer_store_dword v3, v20, s[0:3], s5 offen
	v_accvgpr_read_b32 v3, a100
	v_accvgpr_read_b32 v6, a101
	v_cvt_pk_bf16_f32 v28, v3, v6
	v_accvgpr_read_b32 v3, a102
	v_accvgpr_read_b32 v6, a103
	v_cvt_pk_bf16_f32 v29, v3, v6
	v_mov_b32_dpp v3, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v6, v29 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v6, v28, v6, s[6:7]
	v_cndmask_b32_e64 v3, v3, v29, s[6:7]
	v_lshlrev_b32_e32 v14, 16, v3
	s_mov_b32 s9, 0xffff
	v_and_or_b32 v14, s9, v6, v14
	v_lshrrev_b32_e32 v6, 16, v6
	buffer_store_dword v12, v18, s[0:3], s5 offen
	s_mov_b32 s9, 0xffff0000
	v_and_or_b32 v3, s9, v3, v6
	buffer_store_dword v4, v21, s[0:3], s5 offen
	v_accvgpr_read_b32 v4, a104
	v_accvgpr_read_b32 v6, a105
	v_cvt_pk_bf16_f32 v28, v4, v6
	v_accvgpr_read_b32 v4, a106
	v_accvgpr_read_b32 v6, a107
	v_cvt_pk_bf16_f32 v29, v4, v6
	v_mov_b32_dpp v4, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v6, v29 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v6, v28, v6, s[6:7]
	v_cndmask_b32_e64 v4, v4, v29, s[6:7]
	v_lshlrev_b32_e32 v12, 16, v4
	s_mov_b32 s9, 0xffff
	v_and_or_b32 v12, s9, v6, v12
	v_lshrrev_b32_e32 v6, 16, v6
	buffer_store_dword v25, v19, s[0:3], s5 offen
	s_mov_b32 s9, 0xffff0000
	v_and_or_b32 v4, s9, v4, v6
	buffer_store_dword v7, v24, s[0:3], s5 offen
	v_accvgpr_read_b32 v6, a108
	v_accvgpr_read_b32 v7, a109
	v_cvt_pk_bf16_f32 v6, v6, v7
	v_accvgpr_read_b32 v7, a110
	v_accvgpr_read_b32 v25, a111
	v_cvt_pk_bf16_f32 v7, v7, v25
	v_mov_b32_dpp v25, v6 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v27, v7 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v6, v6, v27, s[6:7]
	v_cndmask_b32_e64 v7, v25, v7, s[6:7]
	v_lshlrev_b32_e32 v25, 16, v7
	s_mov_b32 s9, 0xffff
	v_and_or_b32 v25, s9, v6, v25
	v_lshrrev_b32_e32 v6, 16, v6
	buffer_store_dword v23, v13, s[0:3], s5 offen
	s_mov_b32 s9, 0xffff0000
	v_and_or_b32 v6, s9, v7, v6
	buffer_store_dword v2, v22, s[0:3], s5 offen
	v_accvgpr_read_b32 v2, a112
	v_accvgpr_read_b32 v7, a113
	v_cvt_pk_bf16_f32 v28, v2, v7
	v_accvgpr_read_b32 v2, a114
	v_accvgpr_read_b32 v7, a115
	v_cvt_pk_bf16_f32 v29, v2, v7
	v_mov_b32_dpp v2, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v7, v29 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v7, v28, v7, s[6:7]
	v_cndmask_b32_e64 v2, v2, v29, s[6:7]
	v_lshlrev_b32_e32 v23, 16, v2
	v_lshrrev_b32_e32 v27, 16, v7
	s_mov_b32 s5, 0xffff
	v_and_or_b32 v7, s5, v7, v23
	s_mov_b32 s5, 0xffff0000
	v_and_or_b32 v2, s5, v2, v27
	buffer_store_dword v26, v0, s[0:3], s8 offen
	buffer_store_dword v1, v8, s[0:3], s8 offen
	v_accvgpr_read_b32 v1, a116
	v_accvgpr_read_b32 v23, a117
	v_cvt_pk_bf16_f32 v26, v1, v23
	v_accvgpr_read_b32 v1, a118
	v_accvgpr_read_b32 v23, a119
	v_cvt_pk_bf16_f32 v27, v1, v23
	v_mov_b32_dpp v1, v26 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v23, v27 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v23, v26, v23, s[6:7]
	v_cndmask_b32_e64 v1, v1, v27, s[6:7]
	v_lshlrev_b32_e32 v26, 16, v1
	v_lshrrev_b32_e32 v27, 16, v23
	s_mov_b32 s5, 0xffff
	v_and_or_b32 v23, s5, v23, v26
	s_mov_b32 s5, 0xffff0000
	v_and_or_b32 v1, s5, v1, v27
	buffer_store_dword v14, v5, s[0:3], s8 offen
	buffer_store_dword v3, v9, s[0:3], s8 offen
	v_accvgpr_read_b32 v3, a120
	v_accvgpr_read_b32 v14, a121
	v_cvt_pk_bf16_f32 v26, v3, v14
	v_accvgpr_read_b32 v3, a122
	v_accvgpr_read_b32 v14, a123
	v_cvt_pk_bf16_f32 v27, v3, v14
	v_mov_b32_dpp v3, v26 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v14, v27 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v14, v26, v14, s[6:7]
	v_cndmask_b32_e64 v3, v3, v27, s[6:7]
	v_lshlrev_b32_e32 v26, 16, v3
	v_lshrrev_b32_e32 v27, 16, v14
	s_mov_b32 s5, 0xffff
	v_and_or_b32 v14, s5, v14, v26
	s_mov_b32 s5, 0xffff0000
	v_and_or_b32 v3, s5, v3, v27
	buffer_store_dword v12, v10, s[0:3], s8 offen
	buffer_store_dword v4, v11, s[0:3], s8 offen
	v_accvgpr_read_b32 v4, a124
	v_accvgpr_read_b32 v12, a125
	v_cvt_pk_bf16_f32 v26, v4, v12
	v_accvgpr_read_b32 v4, a126
	v_accvgpr_read_b32 v12, a127
	v_cvt_pk_bf16_f32 v27, v4, v12
	v_mov_b32_dpp v4, v26 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v12, v27 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v12, v26, v12, s[6:7]
	v_cndmask_b32_e64 v4, v4, v27, s[6:7]
	v_lshlrev_b32_e32 v26, 16, v4
	v_lshrrev_b32_e32 v27, 16, v12
	s_mov_b32 s5, 0xffff
	v_and_or_b32 v12, s5, v12, v26
	s_mov_b32 s5, 0xffff0000
	v_and_or_b32 v4, s5, v4, v27
	buffer_store_dword v25, v16, s[0:3], s8 offen
	buffer_store_dword v6, v17, s[0:3], s8 offen
	s_add_i32 s5, s18, 64
	s_lshl_b32 s5, s5, 13
	s_add_i32 s5, s4, s5
	v_accvgpr_read_b32 v6, a128
	v_accvgpr_read_b32 v25, a129
	v_cvt_pk_bf16_f32 v26, v6, v25
	v_accvgpr_read_b32 v6, a130
	v_accvgpr_read_b32 v25, a131
	v_cvt_pk_bf16_f32 v27, v6, v25
	v_mov_b32_dpp v6, v26 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v25, v27 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v25, v26, v25, s[6:7]
	v_cndmask_b32_e64 v6, v6, v27, s[6:7]
	v_lshlrev_b32_e32 v26, 16, v6
	s_mov_b32 s9, 0xffff
	v_and_or_b32 v26, s9, v25, v26
	v_lshrrev_b32_e32 v25, 16, v25
	buffer_store_dword v7, v15, s[0:3], s8 offen
	s_mov_b32 s9, 0xffff0000
	v_and_or_b32 v6, s9, v6, v25
	buffer_store_dword v2, v20, s[0:3], s8 offen
	v_accvgpr_read_b32 v2, a132
	v_accvgpr_read_b32 v7, a133
	v_cvt_pk_bf16_f32 v28, v2, v7
	v_accvgpr_read_b32 v2, a134
	v_accvgpr_read_b32 v7, a135
	v_cvt_pk_bf16_f32 v29, v2, v7
	v_mov_b32_dpp v2, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v7, v29 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v7, v28, v7, s[6:7]
	v_cndmask_b32_e64 v2, v2, v29, s[6:7]
	v_lshlrev_b32_e32 v25, 16, v2
	s_mov_b32 s9, 0xffff
	v_and_or_b32 v25, s9, v7, v25
	v_lshrrev_b32_e32 v7, 16, v7
	buffer_store_dword v23, v18, s[0:3], s8 offen
	s_mov_b32 s9, 0xffff0000
	v_and_or_b32 v2, s9, v2, v7
	buffer_store_dword v1, v21, s[0:3], s8 offen
	v_accvgpr_read_b32 v1, a136
	v_accvgpr_read_b32 v7, a137
	v_cvt_pk_bf16_f32 v28, v1, v7
	v_accvgpr_read_b32 v1, a138
	v_accvgpr_read_b32 v7, a139
	v_cvt_pk_bf16_f32 v29, v1, v7
	v_mov_b32_dpp v1, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v7, v29 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v7, v28, v7, s[6:7]
	v_cndmask_b32_e64 v1, v1, v29, s[6:7]
	v_lshlrev_b32_e32 v23, 16, v1
	s_mov_b32 s9, 0xffff
	v_and_or_b32 v23, s9, v7, v23
	v_lshrrev_b32_e32 v7, 16, v7
	buffer_store_dword v14, v19, s[0:3], s8 offen
	s_mov_b32 s9, 0xffff0000
	v_and_or_b32 v1, s9, v1, v7
	buffer_store_dword v3, v24, s[0:3], s8 offen
	v_accvgpr_read_b32 v3, a140
	v_accvgpr_read_b32 v7, a141
	v_cvt_pk_bf16_f32 v28, v3, v7
	v_accvgpr_read_b32 v3, a142
	v_accvgpr_read_b32 v7, a143
	v_cvt_pk_bf16_f32 v29, v3, v7
	v_mov_b32_dpp v3, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v7, v29 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v7, v28, v7, s[6:7]
	v_cndmask_b32_e64 v3, v3, v29, s[6:7]
	v_lshlrev_b32_e32 v14, 16, v3
	s_mov_b32 s9, 0xffff
	v_and_or_b32 v14, s9, v7, v14
	v_lshrrev_b32_e32 v7, 16, v7
	buffer_store_dword v12, v13, s[0:3], s8 offen
	s_mov_b32 s9, 0xffff0000
	v_and_or_b32 v3, s9, v3, v7
	buffer_store_dword v4, v22, s[0:3], s8 offen
	v_accvgpr_read_b32 v4, a144
	v_accvgpr_read_b32 v7, a145
	v_cvt_pk_bf16_f32 v28, v4, v7
	v_accvgpr_read_b32 v4, a146
	v_accvgpr_read_b32 v7, a147
	v_cvt_pk_bf16_f32 v29, v4, v7
	v_mov_b32_dpp v4, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v7, v29 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v7, v28, v7, s[6:7]
	v_cndmask_b32_e64 v4, v4, v29, s[6:7]
	v_lshlrev_b32_e32 v12, 16, v4
	v_lshrrev_b32_e32 v27, 16, v7
	s_mov_b32 s8, 0xffff
	v_and_or_b32 v7, s8, v7, v12
	s_mov_b32 s8, 0xffff0000
	v_and_or_b32 v4, s8, v4, v27
	buffer_store_dword v26, v0, s[0:3], s5 offen
	buffer_store_dword v6, v8, s[0:3], s5 offen
	v_accvgpr_read_b32 v6, a148
	v_accvgpr_read_b32 v12, a149
	v_cvt_pk_bf16_f32 v26, v6, v12
	v_accvgpr_read_b32 v6, a150
	v_accvgpr_read_b32 v12, a151
	v_cvt_pk_bf16_f32 v27, v6, v12
	v_mov_b32_dpp v6, v26 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v12, v27 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v12, v26, v12, s[6:7]
	v_cndmask_b32_e64 v6, v6, v27, s[6:7]
	v_lshlrev_b32_e32 v26, 16, v6
	v_lshrrev_b32_e32 v27, 16, v12
	s_mov_b32 s8, 0xffff
	v_and_or_b32 v12, s8, v12, v26
	s_mov_b32 s8, 0xffff0000
	v_and_or_b32 v6, s8, v6, v27
	buffer_store_dword v25, v5, s[0:3], s5 offen
	buffer_store_dword v2, v9, s[0:3], s5 offen
	v_accvgpr_read_b32 v2, a152
	v_accvgpr_read_b32 v25, a153
	v_cvt_pk_bf16_f32 v26, v2, v25
	v_accvgpr_read_b32 v2, a154
	v_accvgpr_read_b32 v25, a155
	v_cvt_pk_bf16_f32 v27, v2, v25
	v_mov_b32_dpp v2, v26 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v25, v27 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v25, v26, v25, s[6:7]
	v_cndmask_b32_e64 v2, v2, v27, s[6:7]
	v_lshlrev_b32_e32 v26, 16, v2
	v_lshrrev_b32_e32 v27, 16, v25
	s_mov_b32 s8, 0xffff
	v_and_or_b32 v25, s8, v25, v26
	s_mov_b32 s8, 0xffff0000
	v_and_or_b32 v2, s8, v2, v27
	buffer_store_dword v23, v10, s[0:3], s5 offen
	buffer_store_dword v1, v11, s[0:3], s5 offen
	v_accvgpr_read_b32 v1, a156
	v_accvgpr_read_b32 v23, a157
	v_cvt_pk_bf16_f32 v26, v1, v23
	v_accvgpr_read_b32 v1, a158
	v_accvgpr_read_b32 v23, a159
	v_cvt_pk_bf16_f32 v27, v1, v23
	v_mov_b32_dpp v1, v26 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v23, v27 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v23, v26, v23, s[6:7]
	v_cndmask_b32_e64 v1, v1, v27, s[6:7]
	v_lshlrev_b32_e32 v26, 16, v1
	v_lshrrev_b32_e32 v27, 16, v23
	s_mov_b32 s8, 0xffff
	v_and_or_b32 v23, s8, v23, v26
	s_mov_b32 s8, 0xffff0000
	v_and_or_b32 v1, s8, v1, v27
	buffer_store_dword v14, v16, s[0:3], s5 offen
	buffer_store_dword v3, v17, s[0:3], s5 offen
	s_add_i32 s8, s18, 0x50
	s_lshl_b32 s8, s8, 13
	s_add_i32 s8, s4, s8
	v_accvgpr_read_b32 v3, a160
	v_accvgpr_read_b32 v14, a161
	v_cvt_pk_bf16_f32 v26, v3, v14
	v_accvgpr_read_b32 v3, a162
	v_accvgpr_read_b32 v14, a163
	v_cvt_pk_bf16_f32 v27, v3, v14
	v_mov_b32_dpp v3, v26 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v14, v27 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v14, v26, v14, s[6:7]
	v_cndmask_b32_e64 v3, v3, v27, s[6:7]
	v_lshlrev_b32_e32 v26, 16, v3
	s_mov_b32 s9, 0xffff
	v_and_or_b32 v26, s9, v14, v26
	v_lshrrev_b32_e32 v14, 16, v14
	buffer_store_dword v7, v15, s[0:3], s5 offen
	s_mov_b32 s9, 0xffff0000
	v_and_or_b32 v3, s9, v3, v14
	buffer_store_dword v4, v20, s[0:3], s5 offen
	v_accvgpr_read_b32 v4, a164
	v_accvgpr_read_b32 v7, a165
	v_cvt_pk_bf16_f32 v28, v4, v7
	v_accvgpr_read_b32 v4, a166
	v_accvgpr_read_b32 v7, a167
	v_cvt_pk_bf16_f32 v29, v4, v7
	v_mov_b32_dpp v4, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v7, v29 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v7, v28, v7, s[6:7]
	v_cndmask_b32_e64 v4, v4, v29, s[6:7]
	v_lshlrev_b32_e32 v14, 16, v4
	s_mov_b32 s9, 0xffff
	v_and_or_b32 v14, s9, v7, v14
	v_lshrrev_b32_e32 v7, 16, v7
	buffer_store_dword v12, v18, s[0:3], s5 offen
	s_mov_b32 s9, 0xffff0000
	v_and_or_b32 v4, s9, v4, v7
	buffer_store_dword v6, v21, s[0:3], s5 offen
	v_accvgpr_read_b32 v6, a168
	v_accvgpr_read_b32 v7, a169
	v_cvt_pk_bf16_f32 v6, v6, v7
	v_accvgpr_read_b32 v7, a170
	v_accvgpr_read_b32 v12, a171
	v_cvt_pk_bf16_f32 v7, v7, v12
	v_mov_b32_dpp v12, v6 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v27, v7 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v6, v6, v27, s[6:7]
	v_cndmask_b32_e64 v7, v12, v7, s[6:7]
	v_lshlrev_b32_e32 v12, 16, v7
	s_mov_b32 s9, 0xffff
	v_and_or_b32 v12, s9, v6, v12
	v_lshrrev_b32_e32 v6, 16, v6
	buffer_store_dword v25, v19, s[0:3], s5 offen
	s_mov_b32 s9, 0xffff0000
	v_and_or_b32 v6, s9, v7, v6
	buffer_store_dword v2, v24, s[0:3], s5 offen
	v_accvgpr_read_b32 v2, a172
	v_accvgpr_read_b32 v7, a173
	v_cvt_pk_bf16_f32 v28, v2, v7
	v_accvgpr_read_b32 v2, a174
	v_accvgpr_read_b32 v7, a175
	v_cvt_pk_bf16_f32 v29, v2, v7
	v_mov_b32_dpp v2, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v7, v29 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v7, v28, v7, s[6:7]
	v_cndmask_b32_e64 v2, v2, v29, s[6:7]
	v_lshlrev_b32_e32 v25, 16, v2
	s_mov_b32 s9, 0xffff
	v_and_or_b32 v25, s9, v7, v25
	v_lshrrev_b32_e32 v7, 16, v7
	buffer_store_dword v23, v13, s[0:3], s5 offen
	s_mov_b32 s9, 0xffff0000
	v_and_or_b32 v2, s9, v2, v7
	buffer_store_dword v1, v22, s[0:3], s5 offen
	v_accvgpr_read_b32 v1, a176
	v_accvgpr_read_b32 v7, a177
	v_cvt_pk_bf16_f32 v28, v1, v7
	v_accvgpr_read_b32 v1, a178
	v_accvgpr_read_b32 v7, a179
	v_cvt_pk_bf16_f32 v29, v1, v7
	v_mov_b32_dpp v1, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v7, v29 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v7, v28, v7, s[6:7]
	v_cndmask_b32_e64 v1, v1, v29, s[6:7]
	v_lshlrev_b32_e32 v23, 16, v1
	v_lshrrev_b32_e32 v27, 16, v7
	s_mov_b32 s5, 0xffff
	v_and_or_b32 v7, s5, v7, v23
	s_mov_b32 s5, 0xffff0000
	v_and_or_b32 v1, s5, v1, v27
	buffer_store_dword v26, v0, s[0:3], s8 offen
	buffer_store_dword v3, v8, s[0:3], s8 offen
	v_accvgpr_read_b32 v3, a180
	v_accvgpr_read_b32 v23, a181
	v_cvt_pk_bf16_f32 v26, v3, v23
	v_accvgpr_read_b32 v3, a182
	v_accvgpr_read_b32 v23, a183
	v_cvt_pk_bf16_f32 v27, v3, v23
	v_mov_b32_dpp v3, v26 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v23, v27 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v23, v26, v23, s[6:7]
	v_cndmask_b32_e64 v3, v3, v27, s[6:7]
	v_lshlrev_b32_e32 v26, 16, v3
	v_lshrrev_b32_e32 v27, 16, v23
	s_mov_b32 s5, 0xffff
	v_and_or_b32 v23, s5, v23, v26
	s_mov_b32 s5, 0xffff0000
	v_and_or_b32 v3, s5, v3, v27
	buffer_store_dword v14, v5, s[0:3], s8 offen
	buffer_store_dword v4, v9, s[0:3], s8 offen
	v_accvgpr_read_b32 v4, a184
	v_accvgpr_read_b32 v14, a185
	v_cvt_pk_bf16_f32 v26, v4, v14
	v_accvgpr_read_b32 v4, a186
	v_accvgpr_read_b32 v14, a187
	v_cvt_pk_bf16_f32 v27, v4, v14
	v_mov_b32_dpp v4, v26 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v14, v27 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v14, v26, v14, s[6:7]
	v_cndmask_b32_e64 v4, v4, v27, s[6:7]
	v_lshlrev_b32_e32 v26, 16, v4
	v_lshrrev_b32_e32 v27, 16, v14
	s_mov_b32 s5, 0xffff
	v_and_or_b32 v14, s5, v14, v26
	s_mov_b32 s5, 0xffff0000
	v_and_or_b32 v4, s5, v4, v27
	buffer_store_dword v12, v10, s[0:3], s8 offen
	buffer_store_dword v6, v11, s[0:3], s8 offen
	v_accvgpr_read_b32 v6, a188
	v_accvgpr_read_b32 v12, a189
	v_cvt_pk_bf16_f32 v26, v6, v12
	v_accvgpr_read_b32 v6, a190
	v_accvgpr_read_b32 v12, a191
	v_cvt_pk_bf16_f32 v27, v6, v12
	v_mov_b32_dpp v6, v26 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v12, v27 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v12, v26, v12, s[6:7]
	v_cndmask_b32_e64 v6, v6, v27, s[6:7]
	v_lshlrev_b32_e32 v26, 16, v6
	v_lshrrev_b32_e32 v27, 16, v12
	s_mov_b32 s5, 0xffff
	v_and_or_b32 v12, s5, v12, v26
	s_mov_b32 s5, 0xffff0000
	v_and_or_b32 v6, s5, v6, v27
	buffer_store_dword v25, v16, s[0:3], s8 offen
	buffer_store_dword v2, v17, s[0:3], s8 offen
	s_add_i32 s5, s18, 0x60
	v_accvgpr_read_b32 v2, a192
	v_accvgpr_read_b32 v25, a193
	v_cvt_pk_bf16_f32 v26, v2, v25
	v_accvgpr_read_b32 v2, a194
	v_accvgpr_read_b32 v25, a195
	v_cvt_pk_bf16_f32 v27, v2, v25
	s_lshl_b32 s5, s5, 13
	v_mov_b32_dpp v2, v26 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v25, v27 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_add_i32 s5, s4, s5
	v_cndmask_b32_e64 v25, v26, v25, s[6:7]
	v_cndmask_b32_e64 v2, v2, v27, s[6:7]
	v_lshlrev_b32_e32 v26, 16, v2
	s_mov_b32 s9, 0xffff
	v_and_or_b32 v26, s9, v25, v26
	v_lshrrev_b32_e32 v25, 16, v25
	buffer_store_dword v7, v15, s[0:3], s8 offen
	s_mov_b32 s9, 0xffff0000
	v_and_or_b32 v2, s9, v2, v25
	buffer_store_dword v1, v20, s[0:3], s8 offen
	v_accvgpr_read_b32 v1, a196
	v_accvgpr_read_b32 v7, a197
	v_cvt_pk_bf16_f32 v28, v1, v7
	v_accvgpr_read_b32 v1, a198
	v_accvgpr_read_b32 v7, a199
	v_cvt_pk_bf16_f32 v29, v1, v7
	v_mov_b32_dpp v1, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v7, v29 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v7, v28, v7, s[6:7]
	v_cndmask_b32_e64 v1, v1, v29, s[6:7]
	v_lshlrev_b32_e32 v25, 16, v1
	s_mov_b32 s9, 0xffff
	v_and_or_b32 v25, s9, v7, v25
	v_lshrrev_b32_e32 v7, 16, v7
	buffer_store_dword v23, v18, s[0:3], s8 offen
	s_mov_b32 s9, 0xffff0000
	v_and_or_b32 v1, s9, v1, v7
	buffer_store_dword v3, v21, s[0:3], s8 offen
	v_accvgpr_read_b32 v3, a200
	v_accvgpr_read_b32 v7, a201
	v_cvt_pk_bf16_f32 v28, v3, v7
	v_accvgpr_read_b32 v3, a202
	v_accvgpr_read_b32 v7, a203
	v_cvt_pk_bf16_f32 v29, v3, v7
	v_mov_b32_dpp v3, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v7, v29 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v7, v28, v7, s[6:7]
	v_cndmask_b32_e64 v3, v3, v29, s[6:7]
	v_lshlrev_b32_e32 v23, 16, v3
	s_mov_b32 s9, 0xffff
	v_and_or_b32 v23, s9, v7, v23
	v_lshrrev_b32_e32 v7, 16, v7
	buffer_store_dword v14, v19, s[0:3], s8 offen
	s_mov_b32 s9, 0xffff0000
	v_and_or_b32 v3, s9, v3, v7
	buffer_store_dword v4, v24, s[0:3], s8 offen
	v_accvgpr_read_b32 v4, a204
	v_accvgpr_read_b32 v7, a205
	v_cvt_pk_bf16_f32 v28, v4, v7
	v_accvgpr_read_b32 v4, a206
	v_accvgpr_read_b32 v7, a207
	v_cvt_pk_bf16_f32 v29, v4, v7
	v_mov_b32_dpp v4, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v7, v29 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v7, v28, v7, s[6:7]
	v_cndmask_b32_e64 v4, v4, v29, s[6:7]
	v_lshlrev_b32_e32 v14, 16, v4
	s_mov_b32 s9, 0xffff
	v_and_or_b32 v14, s9, v7, v14
	v_lshrrev_b32_e32 v7, 16, v7
	buffer_store_dword v12, v13, s[0:3], s8 offen
	s_mov_b32 s9, 0xffff0000
	v_and_or_b32 v4, s9, v4, v7
	buffer_store_dword v6, v22, s[0:3], s8 offen
	v_accvgpr_read_b32 v6, a208
	v_accvgpr_read_b32 v7, a209
	v_cvt_pk_bf16_f32 v6, v6, v7
	v_accvgpr_read_b32 v7, a210
	v_accvgpr_read_b32 v12, a211
	v_cvt_pk_bf16_f32 v7, v7, v12
	v_mov_b32_dpp v12, v6 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v27, v7 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v6, v6, v27, s[6:7]
	v_cndmask_b32_e64 v7, v12, v7, s[6:7]
	v_lshlrev_b32_e32 v12, 16, v7
	v_lshrrev_b32_e32 v27, 16, v6
	s_mov_b32 s8, 0xffff
	v_and_or_b32 v6, s8, v6, v12
	s_mov_b32 s8, 0xffff0000
	v_and_or_b32 v7, s8, v7, v27
	buffer_store_dword v26, v0, s[0:3], s5 offen
	buffer_store_dword v2, v8, s[0:3], s5 offen
	v_accvgpr_read_b32 v2, a212
	v_accvgpr_read_b32 v12, a213
	v_cvt_pk_bf16_f32 v26, v2, v12
	v_accvgpr_read_b32 v2, a214
	v_accvgpr_read_b32 v12, a215
	v_cvt_pk_bf16_f32 v27, v2, v12
	v_mov_b32_dpp v2, v26 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v12, v27 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v12, v26, v12, s[6:7]
	v_cndmask_b32_e64 v2, v2, v27, s[6:7]
	v_lshlrev_b32_e32 v26, 16, v2
	v_lshrrev_b32_e32 v27, 16, v12
	s_mov_b32 s8, 0xffff
	v_and_or_b32 v12, s8, v12, v26
	s_mov_b32 s8, 0xffff0000
	v_and_or_b32 v2, s8, v2, v27
	buffer_store_dword v25, v5, s[0:3], s5 offen
	buffer_store_dword v1, v9, s[0:3], s5 offen
	v_accvgpr_read_b32 v1, a216
	v_accvgpr_read_b32 v25, a217
	v_cvt_pk_bf16_f32 v26, v1, v25
	v_accvgpr_read_b32 v1, a218
	v_accvgpr_read_b32 v25, a219
	v_cvt_pk_bf16_f32 v27, v1, v25
	v_mov_b32_dpp v1, v26 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v25, v27 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v25, v26, v25, s[6:7]
	v_cndmask_b32_e64 v1, v1, v27, s[6:7]
	v_lshlrev_b32_e32 v26, 16, v1
	v_lshrrev_b32_e32 v27, 16, v25
	s_mov_b32 s8, 0xffff
	v_and_or_b32 v25, s8, v25, v26
	s_mov_b32 s8, 0xffff0000
	v_and_or_b32 v1, s8, v1, v27
	buffer_store_dword v23, v10, s[0:3], s5 offen
	buffer_store_dword v3, v11, s[0:3], s5 offen
	v_accvgpr_read_b32 v3, a220
	v_accvgpr_read_b32 v23, a221
	v_cvt_pk_bf16_f32 v26, v3, v23
	v_accvgpr_read_b32 v3, a222
	v_accvgpr_read_b32 v23, a223
	v_cvt_pk_bf16_f32 v27, v3, v23
	v_mov_b32_dpp v3, v26 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v23, v27 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v23, v26, v23, s[6:7]
	v_cndmask_b32_e64 v3, v3, v27, s[6:7]
	v_lshlrev_b32_e32 v26, 16, v3
	v_lshrrev_b32_e32 v27, 16, v23
	s_mov_b32 s8, 0xffff
	v_and_or_b32 v23, s8, v23, v26
	s_mov_b32 s8, 0xffff0000
	v_and_or_b32 v3, s8, v3, v27
	buffer_store_dword v14, v16, s[0:3], s5 offen
	buffer_store_dword v4, v17, s[0:3], s5 offen
	v_accvgpr_read_b32 v4, a224
	v_accvgpr_read_b32 v14, a225
	v_cvt_pk_bf16_f32 v26, v4, v14
	v_accvgpr_read_b32 v4, a226
	v_accvgpr_read_b32 v14, a227
	v_cvt_pk_bf16_f32 v27, v4, v14
	v_mov_b32_dpp v4, v26 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v14, v27 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_add_i32 s8, s18, 0x70
	v_cndmask_b32_e64 v14, v26, v14, s[6:7]
	v_cndmask_b32_e64 v4, v4, v27, s[6:7]
	s_lshl_b32 s8, s8, 13
	v_lshlrev_b32_e32 v26, 16, v4
	s_add_i32 s4, s4, s8
	s_mov_b32 s8, 0xffff
	v_and_or_b32 v26, s8, v14, v26
	v_lshrrev_b32_e32 v14, 16, v14
	buffer_store_dword v6, v15, s[0:3], s5 offen
	s_mov_b32 s8, 0xffff0000
	v_and_or_b32 v4, s8, v4, v14
	buffer_store_dword v7, v20, s[0:3], s5 offen
	v_accvgpr_read_b32 v6, a228
	v_accvgpr_read_b32 v7, a229
	v_cvt_pk_bf16_f32 v6, v6, v7
	v_accvgpr_read_b32 v7, a230
	v_accvgpr_read_b32 v14, a231
	v_cvt_pk_bf16_f32 v7, v7, v14
	v_mov_b32_dpp v14, v6 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v27, v7 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v6, v6, v27, s[6:7]
	v_cndmask_b32_e64 v7, v14, v7, s[6:7]
	v_lshlrev_b32_e32 v14, 16, v7
	s_mov_b32 s8, 0xffff
	v_and_or_b32 v14, s8, v6, v14
	v_lshrrev_b32_e32 v6, 16, v6
	buffer_store_dword v12, v18, s[0:3], s5 offen
	s_mov_b32 s8, 0xffff0000
	v_and_or_b32 v6, s8, v7, v6
	buffer_store_dword v2, v21, s[0:3], s5 offen
	v_accvgpr_read_b32 v2, a232
	v_accvgpr_read_b32 v7, a233
	v_cvt_pk_bf16_f32 v28, v2, v7
	v_accvgpr_read_b32 v2, a234
	v_accvgpr_read_b32 v7, a235
	v_cvt_pk_bf16_f32 v29, v2, v7
	v_mov_b32_dpp v2, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v7, v29 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v7, v28, v7, s[6:7]
	v_cndmask_b32_e64 v2, v2, v29, s[6:7]
	v_lshlrev_b32_e32 v12, 16, v2
	s_mov_b32 s8, 0xffff
	v_and_or_b32 v12, s8, v7, v12
	v_lshrrev_b32_e32 v7, 16, v7
	buffer_store_dword v25, v19, s[0:3], s5 offen
	s_mov_b32 s8, 0xffff0000
	v_and_or_b32 v2, s8, v2, v7
	buffer_store_dword v1, v24, s[0:3], s5 offen
	v_accvgpr_read_b32 v1, a236
	v_accvgpr_read_b32 v7, a237
	v_cvt_pk_bf16_f32 v28, v1, v7
	v_accvgpr_read_b32 v1, a238
	v_accvgpr_read_b32 v7, a239
	v_cvt_pk_bf16_f32 v29, v1, v7
	v_mov_b32_dpp v1, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v7, v29 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v7, v28, v7, s[6:7]
	v_cndmask_b32_e64 v1, v1, v29, s[6:7]
	v_lshlrev_b32_e32 v25, 16, v1
	s_mov_b32 s8, 0xffff
	v_and_or_b32 v25, s8, v7, v25
	v_lshrrev_b32_e32 v7, 16, v7
	buffer_store_dword v23, v13, s[0:3], s5 offen
	s_mov_b32 s8, 0xffff0000
	v_and_or_b32 v1, s8, v1, v7
	buffer_store_dword v3, v22, s[0:3], s5 offen
	v_accvgpr_read_b32 v3, a240
	v_accvgpr_read_b32 v7, a241
	v_cvt_pk_bf16_f32 v28, v3, v7
	v_accvgpr_read_b32 v3, a242
	v_accvgpr_read_b32 v7, a243
	v_cvt_pk_bf16_f32 v29, v3, v7
	v_mov_b32_dpp v3, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v7, v29 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v7, v28, v7, s[6:7]
	v_cndmask_b32_e64 v3, v3, v29, s[6:7]
	v_lshlrev_b32_e32 v23, 16, v3
	v_lshrrev_b32_e32 v27, 16, v7
	s_mov_b32 s5, 0xffff
	v_and_or_b32 v7, s5, v7, v23
	s_mov_b32 s5, 0xffff0000
	v_and_or_b32 v3, s5, v3, v27
	buffer_store_dword v26, v0, s[0:3], s4 offen
	buffer_store_dword v4, v8, s[0:3], s4 offen
	v_accvgpr_read_b32 v0, a244
	v_accvgpr_read_b32 v4, a245
	v_cvt_pk_bf16_f32 v26, v0, v4
	v_accvgpr_read_b32 v0, a246
	v_accvgpr_read_b32 v4, a247
	v_cvt_pk_bf16_f32 v27, v0, v4
	v_mov_b32_dpp v0, v26 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v4, v27 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v4, v26, v4, s[6:7]
	v_cndmask_b32_e64 v0, v0, v27, s[6:7]
	v_lshlrev_b32_e32 v8, 16, v0
	v_lshrrev_b32_e32 v23, 16, v4
	s_mov_b32 s5, 0xffff
	v_and_or_b32 v4, s5, v4, v8
	s_mov_b32 s5, 0xffff0000
	v_and_or_b32 v0, s5, v0, v23
	buffer_store_dword v14, v5, s[0:3], s4 offen
	buffer_store_dword v6, v9, s[0:3], s4 offen
	v_accvgpr_read_b32 v5, a248
	v_accvgpr_read_b32 v6, a249
	v_cvt_pk_bf16_f32 v8, v5, v6
	v_accvgpr_read_b32 v5, a250
	v_accvgpr_read_b32 v6, a251
	v_cvt_pk_bf16_f32 v9, v5, v6
	v_mov_b32_dpp v5, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v6, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v6, v8, v6, s[6:7]
	v_cndmask_b32_e64 v5, v5, v9, s[6:7]
	v_lshlrev_b32_e32 v8, 16, v5
	v_lshrrev_b32_e32 v9, 16, v6
	s_mov_b32 s5, 0xffff
	v_and_or_b32 v6, s5, v6, v8
	s_mov_b32 s5, 0xffff0000
	v_and_or_b32 v5, s5, v5, v9
	buffer_store_dword v12, v10, s[0:3], s4 offen
	buffer_store_dword v2, v11, s[0:3], s4 offen
	v_accvgpr_read_b32 v2, a252
	v_accvgpr_read_b32 v8, a253
	v_cvt_pk_bf16_f32 v8, v2, v8
	v_accvgpr_read_b32 v2, a254
	v_accvgpr_read_b32 v9, a255
	v_cvt_pk_bf16_f32 v9, v2, v9
	v_mov_b32_dpp v2, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 0
	v_mov_b32_dpp v10, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v10, s[6:7]
	v_cndmask_b32_e64 v2, v2, v9, s[6:7]
	v_lshlrev_b32_e32 v9, 16, v2
	v_lshrrev_b32_e32 v10, 16, v8
	s_mov_b32 s5, 0xffff
	v_and_or_b32 v8, s5, v8, v9
	s_mov_b32 s5, 0xffff0000
	v_and_or_b32 v2, s5, v2, v10
	buffer_store_dword v25, v16, s[0:3], s4 offen
	buffer_store_dword v1, v17, s[0:3], s4 offen
	buffer_store_dword v7, v15, s[0:3], s4 offen
	buffer_store_dword v3, v20, s[0:3], s4 offen
	buffer_store_dword v4, v18, s[0:3], s4 offen
	buffer_store_dword v0, v21, s[0:3], s4 offen
	buffer_store_dword v6, v19, s[0:3], s4 offen
	buffer_store_dword v5, v24, s[0:3], s4 offen
	buffer_store_dword v8, v13, s[0:3], s4 offen
	buffer_store_dword v2, v22, s[0:3], s4 offen
	s_endpgm
.Lrocke_gemm_mx_e3m2_e3m2_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_4096x4096x65536_c0c1_k4b371438f4_end:
.size rocke_gemm_mx_e3m2_e3m2_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_4096x4096x65536_c0c1_k4b371438f4, .Lrocke_gemm_mx_e3m2_e3m2_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_4096x4096x65536_c0c1_k4b371438f4_end-rocke_gemm_mx_e3m2_e3m2_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_4096x4096x65536_c0c1_k4b371438f4

.rodata
.p2align 6
.amdhsa_kernel rocke_gemm_mx_e3m2_e3m2_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_4096x4096x65536_c0c1_k4b371438f4
  .amdhsa_group_segment_fixed_size 153600
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_kernarg_size 40
  .amdhsa_user_sgpr_count 2
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_next_free_vgpr 460
  .amdhsa_next_free_sgpr 49
  .amdhsa_reserve_vcc 1
  .amdhsa_accum_offset 204
  .amdhsa_float_denorm_mode_32 3
  .amdhsa_float_denorm_mode_16_64 3
.end_amdhsa_kernel

.amdgpu_metadata
---
custom.config:
  Source:
    Origin: rocke
    Description: MX E3M2 GEMM at 4096x4096x65536 on tiled operands and shuffled scales
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
  AssertSizeEqual: {0: 4096, 1: 4096, 2: 1, 3: 65536}
  AssertStrideAEqual: {1: 65536}
  AssertStrideBEqual: {1: 65536}
  AssertStrideDEqual: {1: 4096}
  AssertAlphaValue: 1
amdhsa.target: amdgcn-amd-amdhsa--gfx950
amdhsa.version: [ 1, 1 ]
amdhsa.kernels:
  - .name: rocke_gemm_mx_e3m2_e3m2_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_4096x4096x65536_c0c1_k4b371438f4
    .symbol: rocke_gemm_mx_e3m2_e3m2_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_4096x4096x65536_c0c1_k4b371438f4.kd
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
    .sgpr_count: 55
    .vgpr_count: 460
    .agpr_count: 256
    .wavefront_size: 64
...
.end_amdgpu_metadata
