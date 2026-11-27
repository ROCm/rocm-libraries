// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// MX E3M2 x E3M2 GEMM for gfx950: E8M0 scales per 32 K elements, fp32 accumulation,
// bf16 D rounded to nearest even. 256 threads (wave64, 2x2 waves), macrotile 256x256x128,
// v_mfma_scale_f32_16x16x128_f8f6f4 with cbsz:3 blgp:3, 153600 bytes of static LDS,
// 232 VGPRs and 256 AGPRs.
//
// Fixed problem, enforced by the predicates in custom.config:
//   M = N = 4096, K = 65536, batch 1, TransA = T, TransB = N, lda = ldb = K, D column-major
//   with ldd = 4096. alpha must be 1 (AssertAlphaValue) and beta 0 (UseBeta false): the
//   kernel never reads C. The grid must be exactly 256 workgroups, one per 256x256 tile
//   of D. Its arguments are the five pointers and the int32 in the metadata below, so it
//   cannot take the UserArgs launch path.
//   Every operand and scale descriptor spans one K step (56320 or 1024 bytes) and is
//   rebased per step, and the D descriptor spans 4096x4096 bf16. No load or store is
//   masked and nothing checks the problem size, so any other size reads and writes out
//   of range.
//   The schedule computes 24 * L + 8 K tiles of 128 (a 24-tile main loop and a fixed
//   8-tile tail), here L = 21.
//
// Inputs, in hipBLASLt naming:
//   A uses HIPBLASLT_ORDER_COL256_R128_P48_PAGE24_EXT and its scales
//   HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_128_4_PAGE24_EXT.
//   B uses HIPBLASLT_ORDER_COL256_R128_P48_PAGE24_SPLIT_EXT and its scales
//   HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_128_4_EXT.
//   The byte layouts are those of hipblaslt.h. The kernel's a/sa operands are hipBLASLt B
//   and its scale, b/sb are A and its scale.
//   aidc (AIDPlacementBit, byte 40): only bit 0 is read. The workgroup that the grid
//   assigns column block n of D computes block n ^ (8 * (aidc & 1)) instead, and reads
//   the 8 KiB halves of A and of A's scales that match it. Both values give the same D;
//   the bit only decides which die reads which half.
//
// The E2M3 kernel is the same instruction stream with cbsz:2 blgp:2 on every MFMA.
//
// Edits to the provider's llvm-objdump listing, none of which changes a byte of .text:
//   - the listing banner and the address and encoding comments on each instruction are
//     removed;
//   - the immediate branch offset of the main-loop s_cbranch_scc1 is replaced by the label
//     .Lmain_loop, which encodes to the same offset;
//   - the provider's 266 s_nop padding instructions after s_endpgm are kept after the
//     symbol's .size, so .text is the provider's;
//   - the directives, kernel descriptor and metadata are written for code object v4
//     (amdhsa.version [1, 1], no hidden arguments) from the provider's kernel descriptor.

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.amdhsa_code_object_version 4
.text
.globl rocke_gemm_mx_e3m2_e3m2_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_4096x4096x65536_c0c1p48_ka941ce3864
.p2align 8
.type rocke_gemm_mx_e3m2_e3m2_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_4096x4096x65536_c0c1p48_ka941ce3864,@function
rocke_gemm_mx_e3m2_e3m2_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_4096x4096x65536_c0c1p48_ka941ce3864:
	s_load_dwordx8 s[4:11], s[0:1], 0x0
	s_load_dwordx2 s[12:13], s[0:1], 0x20
	s_load_dword s3, s[0:1], 0x28
	v_lshrrev_b32_e32 v0, 6, v0
	s_nop 0
	v_readfirstlane_b32 s0, v0
	v_mbcnt_lo_u32_b32 v0, -1, 0
	v_mbcnt_hi_u32_b32 v0, -1, v0
	s_lshr_b32 s1, s0, 1
	s_and_b32 s14, s0, 1
	s_and_b32 s15, s2, 7
	s_lshr_b32 s2, s2, 3
	s_lshr_b32 s16, s15, 1
	s_lshl_b32 s16, s16, 2
	s_lshr_b32 s17, s2, 3
	s_add_i32 s16, s16, s17
	s_and_b32 s15, s15, 1
	s_lshl_b32 s15, s15, 3
	s_and_b32 s2, s2, 7
	s_add_i32 s2, s15, s2
	s_waitcnt lgkmcnt(0)
	s_and_b32 s3, s3, 1
	s_lshl_b32 s3, s3, 3
	s_xor_b32 s3, s16, s3
	s_lshl_b32 s15, s3, 8
	s_lshl_b32 s16, s2, 8
	s_lshl_b32 s1, s1, 7
	s_lshl_b32 s14, s14, 7
	s_lshl_b32 s17, s0, 14
	s_and_b32 s18, s3, 7
	s_lshr_b32 s3, s3, 3
	s_and_b32 s3, s3, 1
	s_lshl_b32 s19, s18, 6
	s_lshr_b32 s20, s18, 1
	s_and_b32 s20, s20, 1
	s_add_i32 s19, s19, s20
	s_lshl_b32 s19, s19, 21
	s_lshl_b32 s3, s3, 13
	s_add_u32 s20, s3, s19
	s_addc_u32 s21, 0, 0
	s_lshl_b32 s18, s18, 16
	s_add_u32 s18, s20, s18
	s_addc_u32 s19, s21, 0
	s_add_u32 s4, s4, s18
	s_addc_u32 s5, s5, s19
	s_lshl_b32 s18, s2, 6
	s_lshr_b32 s19, s2, 1
	s_and_b32 s19, s19, 1
	s_add_i32 s18, s18, s19
	s_lshl_b32 s18, s18, 21
	s_add_u32 s18, s3, s18
	s_addc_u32 s19, 0, 0
	s_and_b32 s20, s2, 7
	s_lshl_b32 s21, s20, 16
	s_add_u32 s18, s18, s21
	s_addc_u32 s19, s19, 0
	s_add_u32 s6, s6, s18
	s_addc_u32 s7, s7, s19
	s_lshr_b32 s2, s2, 3
	s_lshl_b32 s2, s2, 14
	s_lshl_b32 s18, s20, 10
	s_add_i32 s2, s2, s18
	s_add_i32 s2, s3, s2
	s_lshr_b32 s3, s1, 4
	s_lshr_b32 s18, s14, 4
	s_lshr_b32 s3, s3, 1
	s_lshr_b32 s18, s18, 1
	s_add_i32 s19, s15, s1
	s_add_i32 s16, s16, s14
	s_lshl_b32 s20, s0, 8
	s_ashr_i32 s21, s15, 31
	s_lshr_b32 s21, s21, 25
	s_add_i32 s15, s15, s21
	s_ashr_i32 s15, s15, 7
	s_lshr_b32 s1, s1, 7
	s_lshr_b32 s14, s14, 7
	s_lshl_b32 s15, s15, 9
	s_add_u32 s8, s8, s15
	s_mul_i32 s0, s0, 0x1800
	s_addc_u32 s9, s9, 0
	s_add_u32 s10, s10, s2
	v_lshl_add_u32 v1, v0, 4, s17
	s_addc_u32 s11, s11, 0
	s_add_i32 s2, 0x1800, s0
	v_add_u32_e32 v2, 0x1000, v1
	s_mul_i32 s3, s3, 0x300
	v_mul_u32_u24_e32 v3, 12, v0
	s_mul_i32 s15, s18, 0x300
	v_lshl_add_u32 v4, v0, 2, s20
	s_mov_b32 s24, s4
	s_mov_b32 s25, s5
	s_mov_b32 s26, 0xdc00
	s_mov_b32 m0, s2
	s_mov_b32 s27, 0x31014000
	buffer_load_dwordx4 v1, s[24:27], 0 offen sc0 lds
	buffer_load_dwordx4 v1, s[24:27], 0 offen offset:1024 sc0 lds
	buffer_load_dwordx4 v1, s[24:27], 0 offen offset:2048 sc0 lds
	s_add_i32 s17, s0, 0xc00
	s_add_i32 s18, 0x1800, s17
	s_mov_b32 m0, s18
	s_nop 0
	buffer_load_dwordx4 v2, s[24:27], 0 offen sc0 lds
	buffer_load_dwordx4 v2, s[24:27], 0 offen offset:1024 sc0 lds
	buffer_load_dwordx4 v2, s[24:27], 0 offen offset:2048 sc0 lds
	s_add_u32 s4, s4, 0x10000
	s_addc_u32 s5, s5, 0
	s_add_i32 s21, 0x7800, s0
	s_mov_b32 s24, s6
	s_mov_b32 m0, s21
	s_mov_b32 s25, s7
	buffer_load_dwordx4 v1, s[24:27], 0 offen sc0 lds
	buffer_load_dwordx4 v1, s[24:27], 0 offen offset:1024 sc0 lds
	buffer_load_dwordx4 v1, s[24:27], 0 offen offset:2048 sc0 lds
	s_add_i32 s22, 0x7800, s17
	s_mov_b32 m0, s22
	s_nop 0
	buffer_load_dwordx4 v2, s[24:27], 0 offen sc0 lds
	buffer_load_dwordx4 v2, s[24:27], 0 offen offset:1024 sc0 lds
	buffer_load_dwordx4 v2, s[24:27], 0 offen offset:2048 sc0 lds
	s_add_u32 s6, s6, 0x10000
	s_addc_u32 s7, s7, 0
	s_mov_b32 s24, s8
	s_mov_b32 s25, s9
	s_mov_b32 m0, s20
	s_mov_b32 s26, 0x400
	buffer_load_dword v4, s[24:27], 0 offen lds
	s_add_u32 s8, s8, 0x4000
	s_addc_u32 s9, s9, 0
	s_add_i32 s23, 0x400, s20
	s_mov_b32 s24, s10
	s_mov_b32 m0, s23
	s_mov_b32 s25, s11
	buffer_load_dword v4, s[24:27], 0 offen lds
	s_add_u32 s10, s10, 0x8000
	s_addc_u32 s11, s11, 0
	s_add_i32 s24, 0xd800, s0
	s_mov_b32 s28, s4
	s_mov_b32 s29, s5
	s_mov_b32 s30, 0xdc00
	s_mov_b32 m0, s24
	s_mov_b32 s31, 0x31014000
	buffer_load_dwordx4 v1, s[28:31], 0 offen sc0 lds
	buffer_load_dwordx4 v1, s[28:31], 0 offen offset:1024 sc0 lds
	buffer_load_dwordx4 v1, s[28:31], 0 offen offset:2048 sc0 lds
	s_add_i32 s25, 0xd800, s17
	s_mov_b32 m0, s25
	s_nop 0
	buffer_load_dwordx4 v2, s[28:31], 0 offen sc0 lds
	buffer_load_dwordx4 v2, s[28:31], 0 offen offset:1024 sc0 lds
	buffer_load_dwordx4 v2, s[28:31], 0 offen offset:2048 sc0 lds
	s_add_u32 s4, s4, 0x10000
	s_addc_u32 s5, s5, 0
	s_add_i32 s26, 0x13800, s0
	s_mov_b32 s28, s6
	s_mov_b32 m0, s26
	s_mov_b32 s29, s7
	buffer_load_dwordx4 v1, s[28:31], 0 offen sc0 lds
	buffer_load_dwordx4 v1, s[28:31], 0 offen offset:1024 sc0 lds
	buffer_load_dwordx4 v1, s[28:31], 0 offen offset:2048 sc0 lds
	s_add_i32 s27, 0x13800, s17
	s_mov_b32 m0, s27
	s_nop 0
	buffer_load_dwordx4 v2, s[28:31], 0 offen sc0 lds
	buffer_load_dwordx4 v2, s[28:31], 0 offen offset:1024 sc0 lds
	buffer_load_dwordx4 v2, s[28:31], 0 offen offset:2048 sc0 lds
	s_add_u32 s6, s6, 0x10000
	s_addc_u32 s7, s7, 0
	s_add_i32 s28, 0x800, s20
	s_mov_b32 s32, s8
	s_mov_b32 s33, s9
	s_mov_b32 s34, 0x400
	s_mov_b32 m0, s28
	s_mov_b32 s35, 0x31014000
	buffer_load_dword v4, s[32:35], 0 offen lds
	s_add_u32 s8, s8, 0x4000
	s_addc_u32 s9, s9, 0
	s_add_i32 s29, 0xc00, s20
	s_mov_b32 s32, s10
	s_mov_b32 m0, s29
	s_mov_b32 s33, s11
	buffer_load_dword v4, s[32:35], 0 offen lds
	s_add_u32 s10, s10, 0x8000
	s_addc_u32 s11, s11, 0
	s_add_i32 s30, 0x19800, s0
	s_mov_b32 s32, s4
	s_mov_b32 s33, s5
	s_mov_b32 m0, s30
	s_mov_b32 s34, 0xdc00
	buffer_load_dwordx4 v1, s[32:35], 0 offen sc0 lds
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:1024 sc0 lds
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:2048 sc0 lds
	s_add_i32 s31, 0x19800, s17
	s_mov_b32 m0, s31
	s_nop 0
	buffer_load_dwordx4 v2, s[32:35], 0 offen sc0 lds
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:1024 sc0 lds
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:2048 sc0 lds
	s_add_u32 s4, s4, 0x10000
	s_addc_u32 s5, s5, 0
	s_add_i32 s0, 0x1f800, s0
	s_mov_b32 s32, s6
	s_mov_b32 m0, s0
	s_mov_b32 s33, s7
	buffer_load_dwordx4 v1, s[32:35], 0 offen sc0 lds
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:1024 sc0 lds
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:2048 sc0 lds
	s_add_i32 s17, 0x1f800, s17
	s_mov_b32 m0, s17
	s_nop 0
	buffer_load_dwordx4 v2, s[32:35], 0 offen sc0 lds
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:1024 sc0 lds
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:2048 sc0 lds
	s_add_u32 s32, s6, 0x10000
	s_addc_u32 s33, s7, 0
	s_add_i32 s36, 0x1000, s20
	s_mov_b32 s40, s8
	s_mov_b32 s41, s9
	s_mov_b32 s42, 0x400
	s_mov_b32 m0, s36
	s_mov_b32 s43, 0x31014000
	buffer_load_dword v4, s[40:43], 0 offen lds
	s_add_u32 s8, s8, 0x4000
	s_addc_u32 s9, s9, 0
	s_add_i32 s37, 0x1400, s20
	s_mov_b32 s40, s10
	s_mov_b32 m0, s37
	s_mov_b32 s41, s11
	buffer_load_dword v4, s[40:43], 0 offen lds
	s_add_u32 s40, s10, 0x8000
	s_addc_u32 s41, s11, 0
	v_accvgpr_write_b32 a0, 0
	v_accvgpr_write_b32 a1, 0
	v_accvgpr_write_b32 a2, 0
	v_accvgpr_write_b32 a3, 0
	v_accvgpr_write_b32 a4, 0
	v_accvgpr_write_b32 a5, 0
	v_accvgpr_write_b32 a6, 0
	v_accvgpr_write_b32 a7, 0
	v_accvgpr_write_b32 a8, 0
	v_accvgpr_write_b32 a9, 0
	v_accvgpr_write_b32 a10, 0
	v_accvgpr_write_b32 a11, 0
	v_accvgpr_write_b32 a12, 0
	v_accvgpr_write_b32 a13, 0
	v_accvgpr_write_b32 a14, 0
	v_accvgpr_write_b32 a15, 0
	v_accvgpr_write_b32 a16, 0
	v_accvgpr_write_b32 a17, 0
	v_accvgpr_write_b32 a18, 0
	v_accvgpr_write_b32 a19, 0
	v_accvgpr_write_b32 a20, 0
	v_accvgpr_write_b32 a21, 0
	v_accvgpr_write_b32 a22, 0
	v_accvgpr_write_b32 a23, 0
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
	s_waitcnt vmcnt(28) lgkmcnt(0)
	s_barrier
	s_lshl_b32 s1, s1, 7
	v_lshlrev_b32_e32 v5, 1, v0
	s_lshl_b32 s6, s14, 7
	v_add_lshl_u32 v6, s1, v5, 2
	v_add_lshl_u32 v5, s6, v5, 2
	v_add_lshl_u32 v7, s3, v3, 2
	ds_read_b64 v[8:9], v6
	ds_read_b64 v[10:11], v5 offset:1024
	ds_read_b128 v[12:15], v7 offset:6144
	ds_read_b128 v[16:19], v7 offset:6160
	ds_read_b128 v[20:23], v7 offset:6176
	v_add_lshl_u32 v3, s15, v3, 2
	ds_read_b128 v[24:27], v3 offset:30720
	ds_read_b128 v[28:31], v3 offset:30736
	ds_read_b128 v[32:35], v3 offset:30752
	ds_read_b128 v[36:39], v3 offset:33792
	ds_read_b128 v[40:43], v3 offset:33808
	ds_read_b128 v[44:47], v3 offset:33824
	ds_read_b128 v[48:51], v3 offset:36864
	ds_read_b128 v[52:55], v3 offset:36880
	ds_read_b128 v[56:59], v3 offset:36896
	ds_read_b128 v[60:63], v3 offset:39936
	ds_read_b128 v[64:67], v3 offset:39952
	ds_read_b128 v[68:71], v3 offset:39968
	ds_read_b128 v[72:75], v7 offset:9216
	ds_read_b128 v[76:79], v7 offset:9232
	ds_read_b128 v[80:83], v7 offset:9248
	ds_read_b128 v[84:87], v7 offset:12288
	ds_read_b128 v[88:91], v7 offset:12304
	ds_read_b128 v[92:95], v7 offset:12320
	ds_read_b128 v[96:99], v7 offset:15360
	ds_read_b128 v[100:103], v7 offset:15376
	ds_read_b128 v[104:107], v7 offset:15392
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_mov_b32 s1, 0
	v_add_u32_e32 v108, 0x10000, v3
	v_add_u32_e32 v109, 0x10000, v7
	v_add_u32_e32 v110, 0x20000, v3
	s_mov_b32 s6, 0xdc00
	s_mov_b32 s7, 0x31014000
	s_mov_b32 s10, 0x400
	s_mov_b32 s11, 0x31014000
.Lmain_loop:
	s_barrier
	ds_read_b64 v[112:113], v6 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[24:29], v[12:17], a[0:3], v10, v8 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[114:115], v5 offset:3072
	s_mov_b32 m0, s2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[30:35], v[12:17], a[4:7], v10, v8 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[36:41], v[12:17], a[8:11], v10, v8 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[116:119], v7 offset:55296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[42:47], v[12:17], a[12:15], v10, v8 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[128:131], v108 offset:14336
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[48:53], v[12:17], a[16:19], v11, v8 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[54:59], v[12:17], a[20:23], v11, v8 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[60:65], v[12:17], a[24:27], v11, v8 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[120:123], v7 offset:55312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[66:71], v[12:17], a[28:31], v11, v8 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[132:135], v108 offset:14352
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[66:71], v[18:23], a[60:63], v11, v8 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[60:65], v[18:23], a[56:59], v11, v8 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[54:59], v[18:23], a[52:55], v11, v8 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[124:127], v7 offset:55328
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[48:53], v[18:23], a[48:51], v11, v8 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[136:139], v108 offset:14368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[42:47], v[18:23], a[44:47], v10, v8 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s18, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[36:41], v[18:23], a[40:43], v10, v8 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[30:35], v[18:23], a[36:39], v10, v8 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[12:15], v7 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[24:29], v[18:23], a[32:35], v10, v8 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[140:143], v108 offset:17408
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[24:29], v[72:77], a[64:67], v10, v8 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[30:35], v[72:77], a[68:71], v10, v8 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[36:41], v[72:77], a[72:75], v10, v8 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[16:19], v7 offset:58384
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[42:47], v[72:77], a[76:79], v10, v8 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[144:147], v108 offset:17424
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[48:53], v[72:77], a[80:83], v11, v8 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[54:59], v[72:77], a[84:87], v11, v8 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[60:65], v[72:77], a[88:91], v11, v8 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s4, s4, 0x10000
	s_addc_u32 s5, s5, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[66:71], v[72:77], a[92:95], v11, v8 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[20:23], v7 offset:58400
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[66:71], v[78:83], a[124:127], v11, v8 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s21
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[60:65], v[78:83], a[120:123], v11, v8 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[54:59], v[78:83], a[116:119], v11, v8 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[148:151], v108 offset:17440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[48:53], v[78:83], a[112:115], v11, v8 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[42:47], v[78:83], a[108:111], v10, v8 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[72:75], v7 offset:61440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[36:41], v[78:83], a[104:107], v10, v8 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[30:35], v[78:83], a[100:103], v10, v8 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[152:155], v108 offset:20480
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[24:29], v[78:83], a[96:99], v10, v8 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[76:79], v7 offset:61456
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[24:29], v[84:89], a[128:131], v10, v9 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[30:35], v[84:89], a[132:135], v10, v9 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[36:41], v[84:89], a[136:139], v10, v9 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[156:159], v108 offset:20496
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[42:47], v[84:89], a[140:143], v10, v9 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[80:83], v7 offset:61472
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[48:53], v[84:89], a[144:147], v11, v9 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s22, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[54:59], v[84:89], a[148:151], v11, v9 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[60:65], v[84:89], a[152:155], v11, v9 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[160:163], v108 offset:20512
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[66:71], v[84:89], a[156:159], v11, v9 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[66:71], v[90:95], a[188:191], v11, v9 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[164:167], v7 offset:64512
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[60:65], v[90:95], a[184:187], v11, v9 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[54:59], v[90:95], a[180:183], v11, v9 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[84:87], v108 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[48:53], v[90:95], a[176:179], v11, v9 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[168:171], v7 offset:64528
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[42:47], v[90:95], a[172:175], v10, v9 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[36:41], v[90:95], a[168:171], v10, v9 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[30:35], v[90:95], a[164:167], v10, v9 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_u32 s32, s32, 0x10000
	s_addc_u32 s33, s33, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[24:29], v[90:95], a[160:163], v10, v9 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[88:91], v108 offset:23568
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[24:29], v[96:101], a[192:195], v10, v9 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s20
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[30:35], v[96:101], a[196:199], v10, v9 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v4, s[8:11], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[36:41], v[96:101], a[200:203], v10, v9 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s8, s8, 0x4000
	s_addc_u32 s9, s9, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[42:47], v[96:101], a[204:207], v10, v9 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[172:175], v7 offset:64544
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[48:53], v[96:101], a[208:211], v11, v9 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s23
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[54:59], v[96:101], a[212:215], v11, v9 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v4, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[60:65], v[96:101], a[216:219], v11, v9 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s40, s40, 0x8000
	s_addc_u32 s41, s41, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[66:71], v[96:101], a[220:223], v11, v9 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[92:95], v108 offset:23584
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[66:71], v[102:107], a[252:255], v11, v9 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[60:65], v[102:107], a[248:251], v11, v9 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[54:59], v[102:107], a[244:247], v11, v9 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[48:53], v[102:107], a[240:243], v11, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[42:47], v[102:107], a[236:239], v10, v9 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[36:41], v[102:107], a[232:235], v10, v9 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[30:35], v[102:107], a[228:231], v10, v9 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(14) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[24:29], v[102:107], a[224:227], v10, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_barrier
	ds_read_b64 v[8:9], v6 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[128:133], v[116:121], a[0:3], v114, v112 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[10:11], v5 offset:5120
	s_add_u32 m0, s24, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[134:139], v[116:121], a[4:7], v114, v112 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[140:145], v[116:121], a[8:11], v114, v112 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[24:27], v109 offset:38912
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[146:151], v[116:121], a[12:15], v114, v112 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[36:39], v108 offset:63488
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[152:157], v[116:121], a[16:19], v115, v112 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[158:163], v[116:121], a[20:23], v115, v112 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[84:89], v[116:121], a[24:27], v115, v112 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[28:31], v109 offset:38928
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[90:95], v[116:121], a[28:31], v115, v112 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[40:43], v108 offset:63504
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[90:95], v[122:127], a[60:63], v115, v112 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[84:89], v[122:127], a[56:59], v115, v112 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[158:163], v[122:127], a[52:55], v115, v112 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[32:35], v109 offset:38944
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[152:157], v[122:127], a[48:51], v115, v112 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[44:47], v108 offset:63520
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[146:151], v[122:127], a[44:47], v114, v112 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s25, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[140:145], v[122:127], a[40:43], v114, v112 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[134:139], v[122:127], a[36:39], v114, v112 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[48:51], v109 offset:41984
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[128:133], v[122:127], a[32:35], v114, v112 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[60:63], v110 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[128:133], v[12:17], a[64:67], v114, v112 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[134:139], v[12:17], a[68:71], v114, v112 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[140:145], v[12:17], a[72:75], v114, v112 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[52:55], v109 offset:42000
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[146:151], v[12:17], a[76:79], v114, v112 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[64:67], v110 offset:1040
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[152:157], v[12:17], a[80:83], v115, v112 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[158:163], v[12:17], a[84:87], v115, v112 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[84:89], v[12:17], a[88:91], v115, v112 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s4, s4, 0x10000
	s_addc_u32 s5, s5, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[90:95], v[12:17], a[92:95], v115, v112 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[56:59], v109 offset:42016
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[90:95], v[18:23], a[124:127], v115, v112 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s26
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[84:89], v[18:23], a[120:123], v115, v112 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[158:163], v[18:23], a[116:119], v115, v112 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[68:71], v110 offset:1056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[152:157], v[18:23], a[112:115], v115, v112 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[12:15], v109 offset:45056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[146:151], v[18:23], a[108:111], v114, v112 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[140:145], v[18:23], a[104:107], v114, v112 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[134:139], v[18:23], a[100:103], v114, v112 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[96:99], v110 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[128:133], v[18:23], a[96:99], v114, v112 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[16:19], v109 offset:45072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[128:133], v[72:77], a[128:131], v114, v113 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[134:139], v[72:77], a[132:135], v114, v113 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[140:145], v[72:77], a[136:139], v114, v113 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[100:103], v110 offset:4112
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[146:151], v[72:77], a[140:143], v114, v113 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[20:23], v109 offset:45088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[152:157], v[72:77], a[144:147], v115, v113 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s27, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[158:163], v[72:77], a[148:151], v115, v113 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[84:89], v[72:77], a[152:155], v115, v113 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[104:107], v110 offset:4128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[90:95], v[72:77], a[156:159], v115, v113 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[116:119], v109 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[90:95], v[78:83], a[188:191], v115, v113 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[84:89], v[78:83], a[184:187], v115, v113 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[158:163], v[78:83], a[180:183], v115, v113 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[72:75], v110 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[152:157], v[78:83], a[176:179], v115, v113 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[120:123], v109 offset:48144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[146:151], v[78:83], a[172:175], v114, v113 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[140:145], v[78:83], a[168:171], v114, v113 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[134:139], v[78:83], a[164:167], v114, v113 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_u32 s32, s32, 0x10000
	s_addc_u32 s33, s33, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[128:133], v[78:83], a[160:163], v114, v113 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[76:79], v110 offset:7184
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[128:133], v[164:169], a[192:195], v114, v113 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s28
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[134:139], v[164:169], a[196:199], v114, v113 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v4, s[8:11], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[140:145], v[164:169], a[200:203], v114, v113 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s8, s8, 0x4000
	s_addc_u32 s9, s9, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[146:151], v[164:169], a[204:207], v114, v113 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[124:127], v109 offset:48160
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[152:157], v[164:169], a[208:211], v115, v113 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s29
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[158:163], v[164:169], a[212:215], v115, v113 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v4, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[84:89], v[164:169], a[216:219], v115, v113 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s40, s40, 0x8000
	s_addc_u32 s41, s41, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[90:95], v[164:169], a[220:223], v115, v113 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[80:83], v110 offset:7200
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[90:95], v[170:175], a[252:255], v115, v113 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[84:89], v[170:175], a[248:251], v115, v113 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[158:163], v[170:175], a[244:247], v115, v113 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[152:157], v[170:175], a[240:243], v115, v113 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[146:151], v[170:175], a[236:239], v114, v113 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[140:145], v[170:175], a[232:235], v114, v113 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[134:139], v[170:175], a[228:231], v114, v113 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(14) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[128:133], v[170:175], a[224:227], v114, v113 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_barrier
	ds_read_b64 v[84:85], v6
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:41], v[24:29], a[0:3], v10, v8 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[86:87], v5 offset:1024
	s_add_u32 m0, s30, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[42:47], v[24:29], a[4:7], v10, v8 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[60:65], v[24:29], a[8:11], v10, v8 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[128:131], v7 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:71], v[24:29], a[12:15], v10, v8 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[140:143], v3 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[96:101], v[24:29], a[16:19], v11, v8 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[102:107], v[24:29], a[20:23], v11, v8 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[72:77], v[24:29], a[24:27], v11, v8 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[132:135], v7 offset:6160
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[78:83], v[24:29], a[28:31], v11, v8 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[144:147], v3 offset:30736
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[78:83], v[30:35], a[60:63], v11, v8 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[72:77], v[30:35], a[56:59], v11, v8 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[102:107], v[30:35], a[52:55], v11, v8 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[136:139], v7 offset:6176
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[96:101], v[30:35], a[48:51], v11, v8 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[148:151], v3 offset:30752
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[66:71], v[30:35], a[44:47], v10, v8 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s31, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[60:65], v[30:35], a[40:43], v10, v8 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[42:47], v[30:35], a[36:39], v10, v8 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[24:27], v7 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[36:41], v[30:35], a[32:35], v10, v8 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[152:155], v3 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[36:41], v[48:53], a[64:67], v10, v8 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[42:47], v[48:53], a[68:71], v10, v8 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[60:65], v[48:53], a[72:75], v10, v8 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[28:31], v7 offset:9232
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[66:71], v[48:53], a[76:79], v10, v8 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[156:159], v3 offset:33808
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[96:101], v[48:53], a[80:83], v11, v8 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[102:107], v[48:53], a[84:87], v11, v8 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[72:77], v[48:53], a[88:91], v11, v8 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s4, s4, 0x10000
	s_addc_u32 s5, s5, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[78:83], v[48:53], a[92:95], v11, v8 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[32:35], v7 offset:9248
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[78:83], v[54:59], a[124:127], v11, v8 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[72:77], v[54:59], a[120:123], v11, v8 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[102:107], v[54:59], a[116:119], v11, v8 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[160:163], v3 offset:33824
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[96:101], v[54:59], a[112:115], v11, v8 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[48:51], v7 offset:12288
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[66:71], v[54:59], a[108:111], v10, v8 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[60:65], v[54:59], a[104:107], v10, v8 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[42:47], v[54:59], a[100:103], v10, v8 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[164:167], v3 offset:36864
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[36:41], v[54:59], a[96:99], v10, v8 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[52:55], v7 offset:12304
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[36:41], v[12:17], a[128:131], v10, v9 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[42:47], v[12:17], a[132:135], v10, v9 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[60:65], v[12:17], a[136:139], v10, v9 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[168:171], v3 offset:36880
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[66:71], v[12:17], a[140:143], v10, v9 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[56:59], v7 offset:12320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[96:101], v[12:17], a[144:147], v11, v9 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s17, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[102:107], v[12:17], a[148:151], v11, v9 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[72:77], v[12:17], a[152:155], v11, v9 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[172:175], v3 offset:36896
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[78:83], v[12:17], a[156:159], v11, v9 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[176:179], v7 offset:15360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[78:83], v[18:23], a[188:191], v11, v9 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[72:77], v[18:23], a[184:187], v11, v9 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[102:107], v[18:23], a[180:183], v11, v9 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[12:15], v3 offset:39936
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[96:101], v[18:23], a[176:179], v11, v9 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[180:183], v7 offset:15376
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[66:71], v[18:23], a[172:175], v10, v9 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[60:65], v[18:23], a[168:171], v10, v9 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[42:47], v[18:23], a[164:167], v10, v9 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_u32 s32, s32, 0x10000
	s_addc_u32 s33, s33, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[36:41], v[18:23], a[160:163], v10, v9 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[16:19], v3 offset:39952
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[36:41], v[116:121], a[192:195], v10, v9 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s36
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[42:47], v[116:121], a[196:199], v10, v9 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v4, s[8:11], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[60:65], v[116:121], a[200:203], v10, v9 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s8, s8, 0x4000
	s_addc_u32 s9, s9, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[66:71], v[116:121], a[204:207], v10, v9 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[184:187], v7 offset:15392
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[96:101], v[116:121], a[208:211], v11, v9 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s37
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[102:107], v[116:121], a[212:215], v11, v9 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v4, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[72:77], v[116:121], a[216:219], v11, v9 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s40, s40, 0x8000
	s_addc_u32 s41, s41, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[78:83], v[116:121], a[220:223], v11, v9 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[20:23], v3 offset:39968
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[78:83], v[122:127], a[252:255], v11, v9 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[72:77], v[122:127], a[248:251], v11, v9 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[102:107], v[122:127], a[244:247], v11, v9 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[96:101], v[122:127], a[240:243], v11, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[66:71], v[122:127], a[236:239], v10, v9 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[60:65], v[122:127], a[232:235], v10, v9 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[42:47], v[122:127], a[228:231], v10, v9 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(14) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[36:41], v[122:127], a[224:227], v10, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_barrier
	ds_read_b64 v[8:9], v6 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[140:145], v[128:133], a[0:3], v86, v84 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[10:11], v5 offset:3072
	s_add_u32 m0, s2, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[146:151], v[128:133], a[4:7], v86, v84 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[152:157], v[128:133], a[8:11], v86, v84 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[36:39], v7 offset:55296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[158:163], v[128:133], a[12:15], v86, v84 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[60:63], v108 offset:14336
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[164:169], v[128:133], a[16:19], v87, v84 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[170:175], v[128:133], a[20:23], v87, v84 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[12:17], v[128:133], a[24:27], v87, v84 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[40:43], v7 offset:55312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[18:23], v[128:133], a[28:31], v87, v84 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[64:67], v108 offset:14352
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[18:23], v[134:139], a[60:63], v87, v84 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[12:17], v[134:139], a[56:59], v87, v84 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[170:175], v[134:139], a[52:55], v87, v84 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[44:47], v7 offset:55328
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[164:169], v[134:139], a[48:51], v87, v84 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[68:71], v108 offset:14368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[158:163], v[134:139], a[44:47], v86, v84 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s18, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[152:157], v[134:139], a[40:43], v86, v84 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[146:151], v[134:139], a[36:39], v86, v84 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[72:75], v7 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[140:145], v[134:139], a[32:35], v86, v84 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[88:91], v108 offset:17408
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[140:145], v[24:29], a[64:67], v86, v84 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[146:151], v[24:29], a[68:71], v86, v84 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[152:157], v[24:29], a[72:75], v86, v84 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[76:79], v7 offset:58384
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[158:163], v[24:29], a[76:79], v86, v84 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[92:95], v108 offset:17424
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[164:169], v[24:29], a[80:83], v87, v84 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[170:175], v[24:29], a[84:87], v87, v84 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[12:17], v[24:29], a[88:91], v87, v84 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s4, s4, 0x10000
	s_addc_u32 s5, s5, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[18:23], v[24:29], a[92:95], v87, v84 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[80:83], v7 offset:58400
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[18:23], v[30:35], a[124:127], v87, v84 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s21
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[12:17], v[30:35], a[120:123], v87, v84 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[170:175], v[30:35], a[116:119], v87, v84 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[96:99], v108 offset:17440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[164:169], v[30:35], a[112:115], v87, v84 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[24:27], v7 offset:61440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[158:163], v[30:35], a[108:111], v86, v84 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[152:157], v[30:35], a[104:107], v86, v84 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[146:151], v[30:35], a[100:103], v86, v84 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[112:115], v108 offset:20480
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[140:145], v[30:35], a[96:99], v86, v84 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[28:31], v7 offset:61456
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[140:145], v[48:53], a[128:131], v86, v85 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[146:151], v[48:53], a[132:135], v86, v85 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[152:157], v[48:53], a[136:139], v86, v85 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[116:119], v108 offset:20496
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[158:163], v[48:53], a[140:143], v86, v85 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[32:35], v7 offset:61472
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[164:169], v[48:53], a[144:147], v87, v85 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s22, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[170:175], v[48:53], a[148:151], v87, v85 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[12:17], v[48:53], a[152:155], v87, v85 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[120:123], v108 offset:20512
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[18:23], v[48:53], a[156:159], v87, v85 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[124:127], v7 offset:64512
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[18:23], v[54:59], a[188:191], v87, v85 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[12:17], v[54:59], a[184:187], v87, v85 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[170:175], v[54:59], a[180:183], v87, v85 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[48:51], v108 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[164:169], v[54:59], a[176:179], v87, v85 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[128:131], v7 offset:64528
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[158:163], v[54:59], a[172:175], v86, v85 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[152:157], v[54:59], a[168:171], v86, v85 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[146:151], v[54:59], a[164:167], v86, v85 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_u32 s32, s32, 0x10000
	s_addc_u32 s33, s33, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[140:145], v[54:59], a[160:163], v86, v85 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[52:55], v108 offset:23568
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[140:145], v[176:181], a[192:195], v86, v85 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s20
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[146:151], v[176:181], a[196:199], v86, v85 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v4, s[8:11], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[152:157], v[176:181], a[200:203], v86, v85 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s8, s8, 0x4000
	s_addc_u32 s9, s9, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[158:163], v[176:181], a[204:207], v86, v85 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[132:135], v7 offset:64544
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[164:169], v[176:181], a[208:211], v87, v85 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s23
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[170:175], v[176:181], a[212:215], v87, v85 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v4, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[12:17], v[176:181], a[216:219], v87, v85 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s40, s40, 0x8000
	s_addc_u32 s41, s41, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[18:23], v[176:181], a[220:223], v87, v85 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[56:59], v108 offset:23584
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[18:23], v[182:187], a[252:255], v87, v85 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[12:17], v[182:187], a[248:251], v87, v85 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[170:175], v[182:187], a[244:247], v87, v85 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[164:169], v[182:187], a[240:243], v87, v85 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[158:163], v[182:187], a[236:239], v86, v85 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[152:157], v[182:187], a[232:235], v86, v85 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[146:151], v[182:187], a[228:231], v86, v85 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(14) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[140:145], v[182:187], a[224:227], v86, v85 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_barrier
	ds_read_b64 v[12:13], v6 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[60:65], v[36:41], a[0:3], v10, v8 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[14:15], v5 offset:5120
	s_add_u32 m0, s24, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[66:71], v[36:41], a[4:7], v10, v8 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[88:93], v[36:41], a[8:11], v10, v8 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[136:139], v109 offset:38912
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[94:99], v[36:41], a[12:15], v10, v8 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[148:151], v108 offset:63488
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[112:117], v[36:41], a[16:19], v11, v8 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[118:123], v[36:41], a[20:23], v11, v8 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[48:53], v[36:41], a[24:27], v11, v8 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[140:143], v109 offset:38928
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[54:59], v[36:41], a[28:31], v11, v8 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[152:155], v108 offset:63504
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[54:59], v[42:47], a[60:63], v11, v8 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[48:53], v[42:47], a[56:59], v11, v8 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[118:123], v[42:47], a[52:55], v11, v8 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[144:147], v109 offset:38944
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[112:117], v[42:47], a[48:51], v11, v8 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[156:159], v108 offset:63520
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[94:99], v[42:47], a[44:47], v10, v8 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s25, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[88:93], v[42:47], a[40:43], v10, v8 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[66:71], v[42:47], a[36:39], v10, v8 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[36:39], v109 offset:41984
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[60:65], v[42:47], a[32:35], v10, v8 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[160:163], v110 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[60:65], v[72:77], a[64:67], v10, v8 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[66:71], v[72:77], a[68:71], v10, v8 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[88:93], v[72:77], a[72:75], v10, v8 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[40:43], v109 offset:42000
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[94:99], v[72:77], a[76:79], v10, v8 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[164:167], v110 offset:1040
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[112:117], v[72:77], a[80:83], v11, v8 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[118:123], v[72:77], a[84:87], v11, v8 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[48:53], v[72:77], a[88:91], v11, v8 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s4, s4, 0x10000
	s_addc_u32 s5, s5, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[54:59], v[72:77], a[92:95], v11, v8 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[44:47], v109 offset:42016
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[54:59], v[78:83], a[124:127], v11, v8 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s26
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[48:53], v[78:83], a[120:123], v11, v8 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[118:123], v[78:83], a[116:119], v11, v8 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[168:171], v110 offset:1056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[112:117], v[78:83], a[112:115], v11, v8 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[72:75], v109 offset:45056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[94:99], v[78:83], a[108:111], v10, v8 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[88:93], v[78:83], a[104:107], v10, v8 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[66:71], v[78:83], a[100:103], v10, v8 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[172:175], v110 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[60:65], v[78:83], a[96:99], v10, v8 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[76:79], v109 offset:45072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[60:65], v[24:29], a[128:131], v10, v9 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[66:71], v[24:29], a[132:135], v10, v9 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[88:93], v[24:29], a[136:139], v10, v9 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[176:179], v110 offset:4112
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[94:99], v[24:29], a[140:143], v10, v9 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[80:83], v109 offset:45088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[112:117], v[24:29], a[144:147], v11, v9 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s27, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[118:123], v[24:29], a[148:151], v11, v9 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[48:53], v[24:29], a[152:155], v11, v9 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[180:183], v110 offset:4128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[54:59], v[24:29], a[156:159], v11, v9 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[16:19], v109 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[54:59], v[30:35], a[188:191], v11, v9 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[48:53], v[30:35], a[184:187], v11, v9 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[118:123], v[30:35], a[180:183], v11, v9 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[184:187], v110 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[112:117], v[30:35], a[176:179], v11, v9 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[20:23], v109 offset:48144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[94:99], v[30:35], a[172:175], v10, v9 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[88:93], v[30:35], a[168:171], v10, v9 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[66:71], v[30:35], a[164:167], v10, v9 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_u32 s32, s32, 0x10000
	s_addc_u32 s33, s33, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[60:65], v[30:35], a[160:163], v10, v9 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[188:191], v110 offset:7184
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[60:65], v[124:129], a[192:195], v10, v9 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s28
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[66:71], v[124:129], a[196:199], v10, v9 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v4, s[8:11], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[88:93], v[124:129], a[200:203], v10, v9 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s8, s8, 0x4000
	s_addc_u32 s9, s9, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[94:99], v[124:129], a[204:207], v10, v9 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[24:27], v109 offset:48160
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[112:117], v[124:129], a[208:211], v11, v9 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s29
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[118:123], v[124:129], a[212:215], v11, v9 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v4, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[48:53], v[124:129], a[216:219], v11, v9 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s40, s40, 0x8000
	s_addc_u32 s41, s41, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[54:59], v[124:129], a[220:223], v11, v9 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[192:195], v110 offset:7200
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[54:59], v[130:135], a[252:255], v11, v9 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[48:53], v[130:135], a[248:251], v11, v9 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[118:123], v[130:135], a[244:247], v11, v9 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[112:117], v[130:135], a[240:243], v11, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[94:99], v[130:135], a[236:239], v10, v9 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[88:93], v[130:135], a[232:235], v10, v9 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[66:71], v[130:135], a[228:231], v10, v9 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(14) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[60:65], v[130:135], a[224:227], v10, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_barrier
	ds_read_b64 v[8:9], v6
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[148:153], v[136:141], a[0:3], v14, v12 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[10:11], v5 offset:1024
	s_add_u32 m0, s30, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[154:159], v[136:141], a[4:7], v14, v12 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[160:165], v[136:141], a[8:11], v14, v12 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[48:51], v7 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[166:171], v[136:141], a[12:15], v14, v12 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[60:63], v3 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[172:177], v[136:141], a[16:19], v15, v12 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[178:183], v[136:141], a[20:23], v15, v12 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[184:189], v[136:141], a[24:27], v15, v12 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[52:55], v7 offset:6160
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[190:195], v[136:141], a[28:31], v15, v12 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[64:67], v3 offset:30736
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[190:195], v[142:147], a[60:63], v15, v12 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[184:189], v[142:147], a[56:59], v15, v12 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[178:183], v[142:147], a[52:55], v15, v12 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[56:59], v7 offset:6176
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[172:177], v[142:147], a[48:51], v15, v12 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[68:71], v3 offset:30752
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[166:171], v[142:147], a[44:47], v14, v12 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s31, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[160:165], v[142:147], a[40:43], v14, v12 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[154:159], v[142:147], a[36:39], v14, v12 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[28:31], v7 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[148:153], v[142:147], a[32:35], v14, v12 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[84:87], v3 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[148:153], v[36:41], a[64:67], v14, v12 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[154:159], v[36:41], a[68:71], v14, v12 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[160:165], v[36:41], a[72:75], v14, v12 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[32:35], v7 offset:9232
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[166:171], v[36:41], a[76:79], v14, v12 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[88:91], v3 offset:33808
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[172:177], v[36:41], a[80:83], v15, v12 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[178:183], v[36:41], a[84:87], v15, v12 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[184:189], v[36:41], a[88:91], v15, v12 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s4, s4, 0x10000
	s_addc_u32 s5, s5, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[190:195], v[36:41], a[92:95], v15, v12 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[36:39], v7 offset:9248
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[190:195], v[42:47], a[124:127], v15, v12 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[184:189], v[42:47], a[120:123], v15, v12 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[178:183], v[42:47], a[116:119], v15, v12 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[92:95], v3 offset:33824
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[172:177], v[42:47], a[112:115], v15, v12 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[96:99], v7 offset:12288
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[166:171], v[42:47], a[108:111], v14, v12 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[160:165], v[42:47], a[104:107], v14, v12 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[154:159], v[42:47], a[100:103], v14, v12 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[112:115], v3 offset:36864
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[148:153], v[42:47], a[96:99], v14, v12 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[100:103], v7 offset:12304
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[148:153], v[72:77], a[128:131], v14, v13 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[154:159], v[72:77], a[132:135], v14, v13 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[160:165], v[72:77], a[136:139], v14, v13 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[116:119], v3 offset:36880
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[166:171], v[72:77], a[140:143], v14, v13 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[104:107], v7 offset:12320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[172:177], v[72:77], a[144:147], v15, v13 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s17, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[178:183], v[72:77], a[148:151], v15, v13 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[184:189], v[72:77], a[152:155], v15, v13 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[120:123], v3 offset:36896
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[190:195], v[72:77], a[156:159], v15, v13 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[124:127], v7 offset:15360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[190:195], v[78:83], a[188:191], v15, v13 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[184:189], v[78:83], a[184:187], v15, v13 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[178:183], v[78:83], a[180:183], v15, v13 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[72:75], v3 offset:39936
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[172:177], v[78:83], a[176:179], v15, v13 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[128:131], v7 offset:15376
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[166:171], v[78:83], a[172:175], v14, v13 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[160:165], v[78:83], a[168:171], v14, v13 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[154:159], v[78:83], a[164:167], v14, v13 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_u32 s32, s32, 0x10000
	s_addc_u32 s33, s33, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[148:153], v[78:83], a[160:163], v14, v13 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[76:79], v3 offset:39952
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[148:153], v[16:21], a[192:195], v14, v13 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s36
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[154:159], v[16:21], a[196:199], v14, v13 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v4, s[8:11], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[160:165], v[16:21], a[200:203], v14, v13 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s8, s8, 0x4000
	s_addc_u32 s9, s9, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[166:171], v[16:21], a[204:207], v14, v13 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[132:135], v7 offset:15392
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[172:177], v[16:21], a[208:211], v15, v13 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s37
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[178:183], v[16:21], a[212:215], v15, v13 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v4, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[184:189], v[16:21], a[216:219], v15, v13 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s40, s40, 0x8000
	s_addc_u32 s41, s41, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[190:195], v[16:21], a[220:223], v15, v13 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[80:83], v3 offset:39968
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[190:195], v[22:27], a[252:255], v15, v13 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[184:189], v[22:27], a[248:251], v15, v13 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[178:183], v[22:27], a[244:247], v15, v13 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[172:177], v[22:27], a[240:243], v15, v13 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[166:171], v[22:27], a[236:239], v14, v13 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[160:165], v[22:27], a[232:235], v14, v13 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[154:159], v[22:27], a[228:231], v14, v13 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(14) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[148:153], v[22:27], a[224:227], v14, v13 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_barrier
	ds_read_b64 v[12:13], v6 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[60:65], v[48:53], a[0:3], v10, v8 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[14:15], v5 offset:3072
	s_add_u32 m0, s2, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[66:71], v[48:53], a[4:7], v10, v8 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[84:89], v[48:53], a[8:11], v10, v8 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[16:19], v7 offset:55296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[90:95], v[48:53], a[12:15], v10, v8 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[40:43], v108 offset:14336
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[112:117], v[48:53], a[16:19], v11, v8 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[118:123], v[48:53], a[20:23], v11, v8 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[72:77], v[48:53], a[24:27], v11, v8 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[20:23], v7 offset:55312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[78:83], v[48:53], a[28:31], v11, v8 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[44:47], v108 offset:14352
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[78:83], v[54:59], a[60:63], v11, v8 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[72:77], v[54:59], a[56:59], v11, v8 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[118:123], v[54:59], a[52:55], v11, v8 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[24:27], v7 offset:55328
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[112:117], v[54:59], a[48:51], v11, v8 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[48:51], v108 offset:14368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[90:95], v[54:59], a[44:47], v10, v8 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s18, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[84:89], v[54:59], a[40:43], v10, v8 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[66:71], v[54:59], a[36:39], v10, v8 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[136:139], v7 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[60:65], v[54:59], a[32:35], v10, v8 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[148:151], v108 offset:17408
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[60:65], v[28:33], a[64:67], v10, v8 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[66:71], v[28:33], a[68:71], v10, v8 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[84:89], v[28:33], a[72:75], v10, v8 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[140:143], v7 offset:58384
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[90:95], v[28:33], a[76:79], v10, v8 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[152:155], v108 offset:17424
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[112:117], v[28:33], a[80:83], v11, v8 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[118:123], v[28:33], a[84:87], v11, v8 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[72:77], v[28:33], a[88:91], v11, v8 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s4, s4, 0x10000
	s_addc_u32 s5, s5, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[78:83], v[28:33], a[92:95], v11, v8 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[144:147], v7 offset:58400
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[78:83], v[34:39], a[124:127], v11, v8 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s21
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[72:77], v[34:39], a[120:123], v11, v8 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[118:123], v[34:39], a[116:119], v11, v8 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[156:159], v108 offset:17440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[112:117], v[34:39], a[112:115], v11, v8 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[28:31], v7 offset:61440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[90:95], v[34:39], a[108:111], v10, v8 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[84:89], v[34:39], a[104:107], v10, v8 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[66:71], v[34:39], a[100:103], v10, v8 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[160:163], v108 offset:20480
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[60:65], v[34:39], a[96:99], v10, v8 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[32:35], v7 offset:61456
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[60:65], v[96:101], a[128:131], v10, v9 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[66:71], v[96:101], a[132:135], v10, v9 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[84:89], v[96:101], a[136:139], v10, v9 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[164:167], v108 offset:20496
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[90:95], v[96:101], a[140:143], v10, v9 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[36:39], v7 offset:61472
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[112:117], v[96:101], a[144:147], v11, v9 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s22, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[118:123], v[96:101], a[148:151], v11, v9 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[72:77], v[96:101], a[152:155], v11, v9 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[168:171], v108 offset:20512
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[78:83], v[96:101], a[156:159], v11, v9 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[172:175], v7 offset:64512
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[78:83], v[102:107], a[188:191], v11, v9 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[72:77], v[102:107], a[184:187], v11, v9 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[118:123], v[102:107], a[180:183], v11, v9 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[96:99], v108 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[112:117], v[102:107], a[176:179], v11, v9 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[176:179], v7 offset:64528
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[90:95], v[102:107], a[172:175], v10, v9 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[84:89], v[102:107], a[168:171], v10, v9 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[66:71], v[102:107], a[164:167], v10, v9 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_u32 s32, s32, 0x10000
	s_addc_u32 s33, s33, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[60:65], v[102:107], a[160:163], v10, v9 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[100:103], v108 offset:23568
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[60:65], v[124:129], a[192:195], v10, v9 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s20
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[66:71], v[124:129], a[196:199], v10, v9 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v4, s[8:11], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[84:89], v[124:129], a[200:203], v10, v9 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s8, s8, 0x4000
	s_addc_u32 s9, s9, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[90:95], v[124:129], a[204:207], v10, v9 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[180:183], v7 offset:64544
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[112:117], v[124:129], a[208:211], v11, v9 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s23
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[118:123], v[124:129], a[212:215], v11, v9 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v4, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[72:77], v[124:129], a[216:219], v11, v9 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s40, s40, 0x8000
	s_addc_u32 s41, s41, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[78:83], v[124:129], a[220:223], v11, v9 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[104:107], v108 offset:23584
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[78:83], v[130:135], a[252:255], v11, v9 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[72:77], v[130:135], a[248:251], v11, v9 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[118:123], v[130:135], a[244:247], v11, v9 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[112:117], v[130:135], a[240:243], v11, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[90:95], v[130:135], a[236:239], v10, v9 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[84:89], v[130:135], a[232:235], v10, v9 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[66:71], v[130:135], a[228:231], v10, v9 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(14) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[60:65], v[130:135], a[224:227], v10, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_barrier
	ds_read_b64 v[8:9], v6 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[40:45], v[16:21], a[0:3], v14, v12 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[10:11], v5 offset:5120
	s_add_u32 m0, s24, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[46:51], v[16:21], a[4:7], v14, v12 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[148:153], v[16:21], a[8:11], v14, v12 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[52:55], v109 offset:38912
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[154:159], v[16:21], a[12:15], v14, v12 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[64:67], v108 offset:63488
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[160:165], v[16:21], a[16:19], v15, v12 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[166:171], v[16:21], a[20:23], v15, v12 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[96:101], v[16:21], a[24:27], v15, v12 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[56:59], v109 offset:38928
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[102:107], v[16:21], a[28:31], v15, v12 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[68:71], v108 offset:63504
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[102:107], v[22:27], a[60:63], v15, v12 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[96:101], v[22:27], a[56:59], v15, v12 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[166:171], v[22:27], a[52:55], v15, v12 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[60:63], v109 offset:38944
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[160:165], v[22:27], a[48:51], v15, v12 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[72:75], v108 offset:63520
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[154:159], v[22:27], a[44:47], v14, v12 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s25, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[148:153], v[22:27], a[40:43], v14, v12 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[46:51], v[22:27], a[36:39], v14, v12 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[16:19], v109 offset:41984
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[40:45], v[22:27], a[32:35], v14, v12 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[76:79], v110 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[40:45], v[136:141], a[64:67], v14, v12 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[46:51], v[136:141], a[68:71], v14, v12 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[148:153], v[136:141], a[72:75], v14, v12 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[20:23], v109 offset:42000
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[154:159], v[136:141], a[76:79], v14, v12 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[80:83], v110 offset:1040
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[160:165], v[136:141], a[80:83], v15, v12 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[166:171], v[136:141], a[84:87], v15, v12 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[96:101], v[136:141], a[88:91], v15, v12 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s4, s4, 0x10000
	s_addc_u32 s5, s5, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[102:107], v[136:141], a[92:95], v15, v12 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[24:27], v109 offset:42016
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[102:107], v[142:147], a[124:127], v15, v12 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s26
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[96:101], v[142:147], a[120:123], v15, v12 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[166:171], v[142:147], a[116:119], v15, v12 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[84:87], v110 offset:1056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[160:165], v[142:147], a[112:115], v15, v12 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[112:115], v109 offset:45056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[154:159], v[142:147], a[108:111], v14, v12 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[148:153], v[142:147], a[104:107], v14, v12 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[46:51], v[142:147], a[100:103], v14, v12 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[124:127], v110 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[40:45], v[142:147], a[96:99], v14, v12 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[116:119], v109 offset:45072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[40:45], v[28:33], a[128:131], v14, v13 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[46:51], v[28:33], a[132:135], v14, v13 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[148:153], v[28:33], a[136:139], v14, v13 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[128:131], v110 offset:4112
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[154:159], v[28:33], a[140:143], v14, v13 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[120:123], v109 offset:45088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[160:165], v[28:33], a[144:147], v15, v13 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s27, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[166:171], v[28:33], a[148:151], v15, v13 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[96:101], v[28:33], a[152:155], v15, v13 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[132:135], v110 offset:4128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[102:107], v[28:33], a[156:159], v15, v13 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[136:139], v109 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[102:107], v[34:39], a[188:191], v15, v13 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[96:101], v[34:39], a[184:187], v15, v13 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[166:171], v[34:39], a[180:183], v15, v13 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[28:31], v110 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[160:165], v[34:39], a[176:179], v15, v13 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[140:143], v109 offset:48144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[154:159], v[34:39], a[172:175], v14, v13 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[148:153], v[34:39], a[168:171], v14, v13 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[46:51], v[34:39], a[164:167], v14, v13 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_u32 s32, s32, 0x10000
	s_addc_u32 s33, s33, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[40:45], v[34:39], a[160:163], v14, v13 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[32:35], v110 offset:7184
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[40:45], v[172:177], a[192:195], v14, v13 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s28
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[46:51], v[172:177], a[196:199], v14, v13 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v4, s[8:11], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[148:153], v[172:177], a[200:203], v14, v13 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s8, s8, 0x4000
	s_addc_u32 s9, s9, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[154:159], v[172:177], a[204:207], v14, v13 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[144:147], v109 offset:48160
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[160:165], v[172:177], a[208:211], v15, v13 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s29
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[166:171], v[172:177], a[212:215], v15, v13 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v4, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[96:101], v[172:177], a[216:219], v15, v13 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s40, s40, 0x8000
	s_addc_u32 s41, s41, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[102:107], v[172:177], a[220:223], v15, v13 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[36:39], v110 offset:7200
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[102:107], v[178:183], a[252:255], v15, v13 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[96:101], v[178:183], a[248:251], v15, v13 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[166:171], v[178:183], a[244:247], v15, v13 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[160:165], v[178:183], a[240:243], v15, v13 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[154:159], v[178:183], a[236:239], v14, v13 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[148:153], v[178:183], a[232:235], v14, v13 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[46:51], v[178:183], a[228:231], v14, v13 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(14) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[40:45], v[178:183], a[224:227], v14, v13 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_barrier
	ds_read_b64 v[12:13], v6
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[64:69], v[52:57], a[0:3], v10, v8 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[14:15], v5 offset:1024
	s_add_u32 m0, s30, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[70:75], v[52:57], a[4:7], v10, v8 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[76:81], v[52:57], a[8:11], v10, v8 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[40:43], v7 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[82:87], v[52:57], a[12:15], v10, v8 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[88:91], v3 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[124:129], v[52:57], a[16:19], v11, v8 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[130:135], v[52:57], a[20:23], v11, v8 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[28:33], v[52:57], a[24:27], v11, v8 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[44:47], v7 offset:6160
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[34:39], v[52:57], a[28:31], v11, v8 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[92:95], v3 offset:30736
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[34:39], v[58:63], a[60:63], v11, v8 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[28:33], v[58:63], a[56:59], v11, v8 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[130:135], v[58:63], a[52:55], v11, v8 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[48:51], v7 offset:6176
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[124:129], v[58:63], a[48:51], v11, v8 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[96:99], v3 offset:30752
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[82:87], v[58:63], a[44:47], v10, v8 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s31, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[76:81], v[58:63], a[40:43], v10, v8 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[70:75], v[58:63], a[36:39], v10, v8 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[52:55], v7 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[64:69], v[58:63], a[32:35], v10, v8 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[148:151], v3 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[64:69], v[16:21], a[64:67], v10, v8 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[70:75], v[16:21], a[68:71], v10, v8 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[76:81], v[16:21], a[72:75], v10, v8 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[56:59], v7 offset:9232
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[82:87], v[16:21], a[76:79], v10, v8 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[152:155], v3 offset:33808
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[124:129], v[16:21], a[80:83], v11, v8 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[130:135], v[16:21], a[84:87], v11, v8 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[28:33], v[16:21], a[88:91], v11, v8 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s4, s4, 0x10000
	s_addc_u32 s5, s5, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[34:39], v[16:21], a[92:95], v11, v8 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[60:63], v7 offset:9248
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[34:39], v[22:27], a[124:127], v11, v8 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[28:33], v[22:27], a[120:123], v11, v8 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[130:135], v[22:27], a[116:119], v11, v8 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[156:159], v3 offset:33824
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[124:129], v[22:27], a[112:115], v11, v8 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[16:19], v7 offset:12288
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[82:87], v[22:27], a[108:111], v10, v8 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[76:81], v[22:27], a[104:107], v10, v8 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[70:75], v[22:27], a[100:103], v10, v8 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[160:163], v3 offset:36864
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[64:69], v[22:27], a[96:99], v10, v8 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[20:23], v7 offset:12304
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[64:69], v[112:117], a[128:131], v10, v9 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[70:75], v[112:117], a[132:135], v10, v9 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[76:81], v[112:117], a[136:139], v10, v9 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[164:167], v3 offset:36880
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[82:87], v[112:117], a[140:143], v10, v9 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[24:27], v7 offset:12320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[124:129], v[112:117], a[144:147], v11, v9 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s17, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[130:135], v[112:117], a[148:151], v11, v9 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[28:33], v[112:117], a[152:155], v11, v9 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[168:171], v3 offset:36896
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[34:39], v[112:117], a[156:159], v11, v9 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[172:175], v7 offset:15360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[34:39], v[118:123], a[188:191], v11, v9 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[28:33], v[118:123], a[184:187], v11, v9 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[130:135], v[118:123], a[180:183], v11, v9 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[112:115], v3 offset:39936
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[124:129], v[118:123], a[176:179], v11, v9 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[176:179], v7 offset:15376
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[82:87], v[118:123], a[172:175], v10, v9 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[76:81], v[118:123], a[168:171], v10, v9 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[70:75], v[118:123], a[164:167], v10, v9 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_u32 s32, s32, 0x10000
	s_addc_u32 s33, s33, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[64:69], v[118:123], a[160:163], v10, v9 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[116:119], v3 offset:39952
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[64:69], v[136:141], a[192:195], v10, v9 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s36
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[70:75], v[136:141], a[196:199], v10, v9 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v4, s[8:11], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[76:81], v[136:141], a[200:203], v10, v9 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s8, s8, 0x4000
	s_addc_u32 s9, s9, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[82:87], v[136:141], a[204:207], v10, v9 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[180:183], v7 offset:15392
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[124:129], v[136:141], a[208:211], v11, v9 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s37
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[130:135], v[136:141], a[212:215], v11, v9 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v4, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[28:33], v[136:141], a[216:219], v11, v9 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s40, s40, 0x8000
	s_addc_u32 s41, s41, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[34:39], v[136:141], a[220:223], v11, v9 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[120:123], v3 offset:39968
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[34:39], v[142:147], a[252:255], v11, v9 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[28:33], v[142:147], a[248:251], v11, v9 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[130:135], v[142:147], a[244:247], v11, v9 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[124:129], v[142:147], a[240:243], v11, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[82:87], v[142:147], a[236:239], v10, v9 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[76:81], v[142:147], a[232:235], v10, v9 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[70:75], v[142:147], a[228:231], v10, v9 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(14) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[64:69], v[142:147], a[224:227], v10, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_barrier
	ds_read_b64 v[8:9], v6 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[88:93], v[40:45], a[0:3], v14, v12 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[10:11], v5 offset:3072
	s_add_u32 m0, s2, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[94:99], v[40:45], a[4:7], v14, v12 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[148:153], v[40:45], a[8:11], v14, v12 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[28:31], v7 offset:55296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[154:159], v[40:45], a[12:15], v14, v12 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[64:67], v108 offset:14336
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[160:165], v[40:45], a[16:19], v15, v12 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[166:171], v[40:45], a[20:23], v15, v12 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[112:117], v[40:45], a[24:27], v15, v12 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[32:35], v7 offset:55312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[118:123], v[40:45], a[28:31], v15, v12 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[68:71], v108 offset:14352
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[118:123], v[46:51], a[60:63], v15, v12 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[112:117], v[46:51], a[56:59], v15, v12 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[166:171], v[46:51], a[52:55], v15, v12 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[36:39], v7 offset:55328
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[160:165], v[46:51], a[48:51], v15, v12 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[72:75], v108 offset:14368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[154:159], v[46:51], a[44:47], v14, v12 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s18, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[148:153], v[46:51], a[40:43], v14, v12 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[94:99], v[46:51], a[36:39], v14, v12 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[40:43], v7 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[88:93], v[46:51], a[32:35], v14, v12 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[76:79], v108 offset:17408
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[88:93], v[52:57], a[64:67], v14, v12 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[94:99], v[52:57], a[68:71], v14, v12 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[148:153], v[52:57], a[72:75], v14, v12 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[44:47], v7 offset:58384
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[154:159], v[52:57], a[76:79], v14, v12 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[80:83], v108 offset:17424
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[160:165], v[52:57], a[80:83], v15, v12 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[166:171], v[52:57], a[84:87], v15, v12 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[112:117], v[52:57], a[88:91], v15, v12 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s4, s4, 0x10000
	s_addc_u32 s5, s5, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[118:123], v[52:57], a[92:95], v15, v12 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[48:51], v7 offset:58400
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[118:123], v[58:63], a[124:127], v15, v12 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s21
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[112:117], v[58:63], a[120:123], v15, v12 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[166:171], v[58:63], a[116:119], v15, v12 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[84:87], v108 offset:17440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[160:165], v[58:63], a[112:115], v15, v12 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[52:55], v7 offset:61440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[154:159], v[58:63], a[108:111], v14, v12 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[148:153], v[58:63], a[104:107], v14, v12 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[94:99], v[58:63], a[100:103], v14, v12 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[124:127], v108 offset:20480
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[88:93], v[58:63], a[96:99], v14, v12 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[56:59], v7 offset:61456
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[88:93], v[16:21], a[128:131], v14, v13 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[94:99], v[16:21], a[132:135], v14, v13 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[148:153], v[16:21], a[136:139], v14, v13 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[128:131], v108 offset:20496
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[154:159], v[16:21], a[140:143], v14, v13 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[60:63], v7 offset:61472
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[160:165], v[16:21], a[144:147], v15, v13 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s22, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[166:171], v[16:21], a[148:151], v15, v13 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[112:117], v[16:21], a[152:155], v15, v13 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[132:135], v108 offset:20512
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[118:123], v[16:21], a[156:159], v15, v13 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[136:139], v7 offset:64512
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[118:123], v[22:27], a[188:191], v15, v13 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[112:117], v[22:27], a[184:187], v15, v13 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[166:171], v[22:27], a[180:183], v15, v13 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[16:19], v108 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[160:165], v[22:27], a[176:179], v15, v13 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[140:143], v7 offset:64528
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[154:159], v[22:27], a[172:175], v14, v13 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[148:153], v[22:27], a[168:171], v14, v13 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[94:99], v[22:27], a[164:167], v14, v13 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_u32 s32, s32, 0x10000
	s_addc_u32 s33, s33, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[88:93], v[22:27], a[160:163], v14, v13 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[20:23], v108 offset:23568
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[88:93], v[172:177], a[192:195], v14, v13 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s20
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[94:99], v[172:177], a[196:199], v14, v13 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v4, s[8:11], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[148:153], v[172:177], a[200:203], v14, v13 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s8, s8, 0x4000
	s_addc_u32 s9, s9, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[154:159], v[172:177], a[204:207], v14, v13 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[144:147], v7 offset:64544
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[160:165], v[172:177], a[208:211], v15, v13 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s23
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[166:171], v[172:177], a[212:215], v15, v13 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v4, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[112:117], v[172:177], a[216:219], v15, v13 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s40, s40, 0x8000
	s_addc_u32 s41, s41, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[118:123], v[172:177], a[220:223], v15, v13 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[24:27], v108 offset:23584
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[118:123], v[178:183], a[252:255], v15, v13 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[112:117], v[178:183], a[248:251], v15, v13 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[166:171], v[178:183], a[244:247], v15, v13 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[160:165], v[178:183], a[240:243], v15, v13 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[154:159], v[178:183], a[236:239], v14, v13 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[148:153], v[178:183], a[232:235], v14, v13 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[94:99], v[178:183], a[228:231], v14, v13 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(14) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[88:93], v[178:183], a[224:227], v14, v13 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_barrier
	ds_read_b64 v[12:13], v6 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[64:69], v[28:33], a[0:3], v10, v8 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[14:15], v5 offset:5120
	s_add_u32 m0, s24, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[70:75], v[28:33], a[4:7], v10, v8 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[76:81], v[28:33], a[8:11], v10, v8 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[88:91], v109 offset:38912
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[82:87], v[28:33], a[12:15], v10, v8 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[112:115], v108 offset:63488
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[124:129], v[28:33], a[16:19], v11, v8 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[130:135], v[28:33], a[20:23], v11, v8 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[16:21], v[28:33], a[24:27], v11, v8 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[92:95], v109 offset:38928
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[22:27], v[28:33], a[28:31], v11, v8 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[116:119], v108 offset:63504
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[22:27], v[34:39], a[60:63], v11, v8 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[16:21], v[34:39], a[56:59], v11, v8 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[130:135], v[34:39], a[52:55], v11, v8 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[96:99], v109 offset:38944
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[124:129], v[34:39], a[48:51], v11, v8 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[120:123], v108 offset:63520
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[82:87], v[34:39], a[44:47], v10, v8 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s25, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[76:81], v[34:39], a[40:43], v10, v8 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[70:75], v[34:39], a[36:39], v10, v8 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[28:31], v109 offset:41984
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[64:69], v[34:39], a[32:35], v10, v8 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[148:151], v110 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[64:69], v[40:45], a[64:67], v10, v8 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[70:75], v[40:45], a[68:71], v10, v8 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[76:81], v[40:45], a[72:75], v10, v8 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[32:35], v109 offset:42000
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[82:87], v[40:45], a[76:79], v10, v8 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[152:155], v110 offset:1040
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[124:129], v[40:45], a[80:83], v11, v8 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[130:135], v[40:45], a[84:87], v11, v8 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[16:21], v[40:45], a[88:91], v11, v8 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s4, s4, 0x10000
	s_addc_u32 s5, s5, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[22:27], v[40:45], a[92:95], v11, v8 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[36:39], v109 offset:42016
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[22:27], v[46:51], a[124:127], v11, v8 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s26
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[16:21], v[46:51], a[120:123], v11, v8 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[130:135], v[46:51], a[116:119], v11, v8 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[156:159], v110 offset:1056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[124:129], v[46:51], a[112:115], v11, v8 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[40:43], v109 offset:45056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[82:87], v[46:51], a[108:111], v10, v8 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[76:81], v[46:51], a[104:107], v10, v8 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[70:75], v[46:51], a[100:103], v10, v8 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[160:163], v110 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[64:69], v[46:51], a[96:99], v10, v8 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[44:47], v109 offset:45072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[64:69], v[52:57], a[128:131], v10, v9 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[70:75], v[52:57], a[132:135], v10, v9 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[76:81], v[52:57], a[136:139], v10, v9 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[164:167], v110 offset:4112
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[82:87], v[52:57], a[140:143], v10, v9 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[48:51], v109 offset:45088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[124:129], v[52:57], a[144:147], v11, v9 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s27, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[130:135], v[52:57], a[148:151], v11, v9 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[16:21], v[52:57], a[152:155], v11, v9 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[168:171], v110 offset:4128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[22:27], v[52:57], a[156:159], v11, v9 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[172:175], v109 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[22:27], v[58:63], a[188:191], v11, v9 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[16:21], v[58:63], a[184:187], v11, v9 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[130:135], v[58:63], a[180:183], v11, v9 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[52:55], v110 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[124:129], v[58:63], a[176:179], v11, v9 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[176:179], v109 offset:48144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[82:87], v[58:63], a[172:175], v10, v9 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[76:81], v[58:63], a[168:171], v10, v9 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[70:75], v[58:63], a[164:167], v10, v9 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_u32 s32, s32, 0x10000
	s_addc_u32 s33, s33, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[64:69], v[58:63], a[160:163], v10, v9 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[56:59], v110 offset:7184
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[64:69], v[136:141], a[192:195], v10, v9 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s28
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[70:75], v[136:141], a[196:199], v10, v9 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v4, s[8:11], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[76:81], v[136:141], a[200:203], v10, v9 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s8, s8, 0x4000
	s_addc_u32 s9, s9, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[82:87], v[136:141], a[204:207], v10, v9 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[180:183], v109 offset:48160
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[124:129], v[136:141], a[208:211], v11, v9 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s29
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[130:135], v[136:141], a[212:215], v11, v9 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v4, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[16:21], v[136:141], a[216:219], v11, v9 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s40, s40, 0x8000
	s_addc_u32 s41, s41, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[22:27], v[136:141], a[220:223], v11, v9 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[60:63], v110 offset:7200
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[22:27], v[142:147], a[252:255], v11, v9 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[16:21], v[142:147], a[248:251], v11, v9 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[130:135], v[142:147], a[244:247], v11, v9 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[124:129], v[142:147], a[240:243], v11, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[82:87], v[142:147], a[236:239], v10, v9 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[76:81], v[142:147], a[232:235], v10, v9 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[70:75], v[142:147], a[228:231], v10, v9 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(14) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[64:69], v[142:147], a[224:227], v10, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_barrier
	ds_read_b64 v[8:9], v6
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[112:117], v[88:93], a[0:3], v14, v12 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[10:11], v5 offset:1024
	s_add_u32 m0, s30, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[118:123], v[88:93], a[4:7], v14, v12 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[148:153], v[88:93], a[8:11], v14, v12 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[16:19], v7 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[154:159], v[88:93], a[12:15], v14, v12 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[64:67], v3 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[160:165], v[88:93], a[16:19], v15, v12 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[166:171], v[88:93], a[20:23], v15, v12 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[52:57], v[88:93], a[24:27], v15, v12 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[20:23], v7 offset:6160
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[58:63], v[88:93], a[28:31], v15, v12 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[68:71], v3 offset:30736
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[58:63], v[94:99], a[60:63], v15, v12 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[52:57], v[94:99], a[56:59], v15, v12 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[166:171], v[94:99], a[52:55], v15, v12 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[24:27], v7 offset:6176
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[160:165], v[94:99], a[48:51], v15, v12 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[72:75], v3 offset:30752
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[154:159], v[94:99], a[44:47], v14, v12 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s31, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[148:153], v[94:99], a[40:43], v14, v12 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[118:123], v[94:99], a[36:39], v14, v12 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[76:79], v7 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[112:117], v[94:99], a[32:35], v14, v12 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[88:91], v3 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[112:117], v[28:33], a[64:67], v14, v12 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[118:123], v[28:33], a[68:71], v14, v12 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[148:153], v[28:33], a[72:75], v14, v12 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[80:83], v7 offset:9232
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[154:159], v[28:33], a[76:79], v14, v12 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[92:95], v3 offset:33808
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[160:165], v[28:33], a[80:83], v15, v12 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[166:171], v[28:33], a[84:87], v15, v12 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[52:57], v[28:33], a[88:91], v15, v12 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s4, s4, 0x10000
	s_addc_u32 s5, s5, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[58:63], v[28:33], a[92:95], v15, v12 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[84:87], v7 offset:9248
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[58:63], v[34:39], a[124:127], v15, v12 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[52:57], v[34:39], a[120:123], v15, v12 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[166:171], v[34:39], a[116:119], v15, v12 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[96:99], v3 offset:33824
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[160:165], v[34:39], a[112:115], v15, v12 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[28:31], v7 offset:12288
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[154:159], v[34:39], a[108:111], v14, v12 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[148:153], v[34:39], a[104:107], v14, v12 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[118:123], v[34:39], a[100:103], v14, v12 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[124:127], v3 offset:36864
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[112:117], v[34:39], a[96:99], v14, v12 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[32:35], v7 offset:12304
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[112:117], v[40:45], a[128:131], v14, v13 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[118:123], v[40:45], a[132:135], v14, v13 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[148:153], v[40:45], a[136:139], v14, v13 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[128:131], v3 offset:36880
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[154:159], v[40:45], a[140:143], v14, v13 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[36:39], v7 offset:12320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[160:165], v[40:45], a[144:147], v15, v13 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s17, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[166:171], v[40:45], a[148:151], v15, v13 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[52:57], v[40:45], a[152:155], v15, v13 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[132:135], v3 offset:36896
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[58:63], v[40:45], a[156:159], v15, v13 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[136:139], v7 offset:15360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[58:63], v[46:51], a[188:191], v15, v13 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[52:57], v[46:51], a[184:187], v15, v13 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[166:171], v[46:51], a[180:183], v15, v13 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[40:43], v3 offset:39936
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[160:165], v[46:51], a[176:179], v15, v13 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[140:143], v7 offset:15376
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[154:159], v[46:51], a[172:175], v14, v13 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[148:153], v[46:51], a[168:171], v14, v13 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[118:123], v[46:51], a[164:167], v14, v13 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_u32 s32, s32, 0x10000
	s_addc_u32 s33, s33, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[112:117], v[46:51], a[160:163], v14, v13 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[44:47], v3 offset:39952
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[112:117], v[172:177], a[192:195], v14, v13 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s36
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[118:123], v[172:177], a[196:199], v14, v13 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v4, s[8:11], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[148:153], v[172:177], a[200:203], v14, v13 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s8, s8, 0x4000
	s_addc_u32 s9, s9, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[154:159], v[172:177], a[204:207], v14, v13 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[144:147], v7 offset:15392
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[160:165], v[172:177], a[208:211], v15, v13 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s37
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[166:171], v[172:177], a[212:215], v15, v13 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v4, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[52:57], v[172:177], a[216:219], v15, v13 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s40, s40, 0x8000
	s_addc_u32 s41, s41, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[58:63], v[172:177], a[220:223], v15, v13 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[48:51], v3 offset:39968
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[58:63], v[178:183], a[252:255], v15, v13 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[52:57], v[178:183], a[248:251], v15, v13 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[166:171], v[178:183], a[244:247], v15, v13 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[160:165], v[178:183], a[240:243], v15, v13 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[154:159], v[178:183], a[236:239], v14, v13 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[148:153], v[178:183], a[232:235], v14, v13 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[118:123], v[178:183], a[228:231], v14, v13 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(14) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[112:117], v[178:183], a[224:227], v14, v13 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_barrier
	ds_read_b64 v[12:13], v6 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[64:69], v[16:21], a[0:3], v10, v8 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[14:15], v5 offset:3072
	s_add_u32 m0, s2, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[70:75], v[16:21], a[4:7], v10, v8 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[88:93], v[16:21], a[8:11], v10, v8 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[52:55], v7 offset:55296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[94:99], v[16:21], a[12:15], v10, v8 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[112:115], v108 offset:14336
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[124:129], v[16:21], a[16:19], v11, v8 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[130:135], v[16:21], a[20:23], v11, v8 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[40:45], v[16:21], a[24:27], v11, v8 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[56:59], v7 offset:55312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[46:51], v[16:21], a[28:31], v11, v8 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[116:119], v108 offset:14352
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[46:51], v[22:27], a[60:63], v11, v8 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[40:45], v[22:27], a[56:59], v11, v8 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[130:135], v[22:27], a[52:55], v11, v8 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[60:63], v7 offset:55328
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[124:129], v[22:27], a[48:51], v11, v8 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[120:123], v108 offset:14368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[94:99], v[22:27], a[44:47], v10, v8 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s18, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[88:93], v[22:27], a[40:43], v10, v8 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[70:75], v[22:27], a[36:39], v10, v8 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[16:19], v7 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[64:69], v[22:27], a[32:35], v10, v8 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[148:151], v108 offset:17408
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[64:69], v[76:81], a[64:67], v10, v8 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[70:75], v[76:81], a[68:71], v10, v8 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[88:93], v[76:81], a[72:75], v10, v8 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[20:23], v7 offset:58384
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[94:99], v[76:81], a[76:79], v10, v8 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[152:155], v108 offset:17424
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[124:129], v[76:81], a[80:83], v11, v8 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[130:135], v[76:81], a[84:87], v11, v8 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[40:45], v[76:81], a[88:91], v11, v8 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s4, s4, 0x10000
	s_addc_u32 s5, s5, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[46:51], v[76:81], a[92:95], v11, v8 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[24:27], v7 offset:58400
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[46:51], v[82:87], a[124:127], v11, v8 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s21
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[40:45], v[82:87], a[120:123], v11, v8 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[130:135], v[82:87], a[116:119], v11, v8 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[156:159], v108 offset:17440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[124:129], v[82:87], a[112:115], v11, v8 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[76:79], v7 offset:61440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[94:99], v[82:87], a[108:111], v10, v8 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[88:93], v[82:87], a[104:107], v10, v8 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[70:75], v[82:87], a[100:103], v10, v8 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[160:163], v108 offset:20480
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[64:69], v[82:87], a[96:99], v10, v8 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[80:83], v7 offset:61456
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[64:69], v[28:33], a[128:131], v10, v9 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[70:75], v[28:33], a[132:135], v10, v9 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[88:93], v[28:33], a[136:139], v10, v9 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[164:167], v108 offset:20496
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[94:99], v[28:33], a[140:143], v10, v9 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[84:87], v7 offset:61472
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[124:129], v[28:33], a[144:147], v11, v9 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s22, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[130:135], v[28:33], a[148:151], v11, v9 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[40:45], v[28:33], a[152:155], v11, v9 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[168:171], v108 offset:20512
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[46:51], v[28:33], a[156:159], v11, v9 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[172:175], v7 offset:64512
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[46:51], v[34:39], a[188:191], v11, v9 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[40:45], v[34:39], a[184:187], v11, v9 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[130:135], v[34:39], a[180:183], v11, v9 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[28:31], v108 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[124:129], v[34:39], a[176:179], v11, v9 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[176:179], v7 offset:64528
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[94:99], v[34:39], a[172:175], v10, v9 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[88:93], v[34:39], a[168:171], v10, v9 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[70:75], v[34:39], a[164:167], v10, v9 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_u32 s32, s32, 0x10000
	s_addc_u32 s33, s33, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[64:69], v[34:39], a[160:163], v10, v9 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[32:35], v108 offset:23568
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[64:69], v[136:141], a[192:195], v10, v9 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s20
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[70:75], v[136:141], a[196:199], v10, v9 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v4, s[8:11], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[88:93], v[136:141], a[200:203], v10, v9 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s8, s8, 0x4000
	s_addc_u32 s9, s9, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[94:99], v[136:141], a[204:207], v10, v9 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[180:183], v7 offset:64544
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[124:129], v[136:141], a[208:211], v11, v9 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s23
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[130:135], v[136:141], a[212:215], v11, v9 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v4, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[40:45], v[136:141], a[216:219], v11, v9 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s40, s40, 0x8000
	s_addc_u32 s41, s41, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[46:51], v[136:141], a[220:223], v11, v9 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[36:39], v108 offset:23584
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[46:51], v[142:147], a[252:255], v11, v9 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[40:45], v[142:147], a[248:251], v11, v9 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[130:135], v[142:147], a[244:247], v11, v9 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[124:129], v[142:147], a[240:243], v11, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[94:99], v[142:147], a[236:239], v10, v9 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[88:93], v[142:147], a[232:235], v10, v9 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[70:75], v[142:147], a[228:231], v10, v9 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(14) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[64:69], v[142:147], a[224:227], v10, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_barrier
	ds_read_b64 v[8:9], v6 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[112:117], v[52:57], a[0:3], v14, v12 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[10:11], v5 offset:5120
	s_add_u32 m0, s24, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[118:123], v[52:57], a[4:7], v14, v12 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[148:153], v[52:57], a[8:11], v14, v12 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[40:43], v109 offset:38912
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[154:159], v[52:57], a[12:15], v14, v12 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[64:67], v108 offset:63488
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[160:165], v[52:57], a[16:19], v15, v12 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[166:171], v[52:57], a[20:23], v15, v12 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[28:33], v[52:57], a[24:27], v15, v12 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[44:47], v109 offset:38928
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[34:39], v[52:57], a[28:31], v15, v12 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[68:71], v108 offset:63504
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[34:39], v[58:63], a[60:63], v15, v12 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[28:33], v[58:63], a[56:59], v15, v12 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[166:171], v[58:63], a[52:55], v15, v12 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[48:51], v109 offset:38944
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[160:165], v[58:63], a[48:51], v15, v12 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[72:75], v108 offset:63520
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[154:159], v[58:63], a[44:47], v14, v12 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s25, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[148:153], v[58:63], a[40:43], v14, v12 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[118:123], v[58:63], a[36:39], v14, v12 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[52:55], v109 offset:41984
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[112:117], v[58:63], a[32:35], v14, v12 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[88:91], v110 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[112:117], v[16:21], a[64:67], v14, v12 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[118:123], v[16:21], a[68:71], v14, v12 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[148:153], v[16:21], a[72:75], v14, v12 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[56:59], v109 offset:42000
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[154:159], v[16:21], a[76:79], v14, v12 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[92:95], v110 offset:1040
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[160:165], v[16:21], a[80:83], v15, v12 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[166:171], v[16:21], a[84:87], v15, v12 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[28:33], v[16:21], a[88:91], v15, v12 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s4, s4, 0x10000
	s_addc_u32 s5, s5, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[34:39], v[16:21], a[92:95], v15, v12 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[60:63], v109 offset:42016
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[34:39], v[22:27], a[124:127], v15, v12 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s26
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[28:33], v[22:27], a[120:123], v15, v12 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[166:171], v[22:27], a[116:119], v15, v12 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[96:99], v110 offset:1056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[160:165], v[22:27], a[112:115], v15, v12 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[16:19], v109 offset:45056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[154:159], v[22:27], a[108:111], v14, v12 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[148:153], v[22:27], a[104:107], v14, v12 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[118:123], v[22:27], a[100:103], v14, v12 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[124:127], v110 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[112:117], v[22:27], a[96:99], v14, v12 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[20:23], v109 offset:45072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[112:117], v[76:81], a[128:131], v14, v13 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[118:123], v[76:81], a[132:135], v14, v13 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[148:153], v[76:81], a[136:139], v14, v13 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[128:131], v110 offset:4112
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[154:159], v[76:81], a[140:143], v14, v13 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[24:27], v109 offset:45088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[160:165], v[76:81], a[144:147], v15, v13 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s27, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[166:171], v[76:81], a[148:151], v15, v13 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[28:33], v[76:81], a[152:155], v15, v13 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[132:135], v110 offset:4128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[34:39], v[76:81], a[156:159], v15, v13 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[136:139], v109 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[34:39], v[82:87], a[188:191], v15, v13 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[28:33], v[82:87], a[184:187], v15, v13 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[166:171], v[82:87], a[180:183], v15, v13 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[76:79], v110 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[160:165], v[82:87], a[176:179], v15, v13 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[140:143], v109 offset:48144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[154:159], v[82:87], a[172:175], v14, v13 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[148:153], v[82:87], a[168:171], v14, v13 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[118:123], v[82:87], a[164:167], v14, v13 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_u32 s32, s32, 0x10000
	s_addc_u32 s33, s33, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[112:117], v[82:87], a[160:163], v14, v13 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[80:83], v110 offset:7184
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[112:117], v[172:177], a[192:195], v14, v13 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s28
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[118:123], v[172:177], a[196:199], v14, v13 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v4, s[8:11], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[148:153], v[172:177], a[200:203], v14, v13 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s8, s8, 0x4000
	s_addc_u32 s9, s9, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[154:159], v[172:177], a[204:207], v14, v13 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[144:147], v109 offset:48160
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[160:165], v[172:177], a[208:211], v15, v13 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s29
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[166:171], v[172:177], a[212:215], v15, v13 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v4, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[28:33], v[172:177], a[216:219], v15, v13 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s40, s40, 0x8000
	s_addc_u32 s41, s41, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[34:39], v[172:177], a[220:223], v15, v13 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[84:87], v110 offset:7200
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[34:39], v[178:183], a[252:255], v15, v13 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[28:33], v[178:183], a[248:251], v15, v13 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[166:171], v[178:183], a[244:247], v15, v13 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[160:165], v[178:183], a[240:243], v15, v13 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[154:159], v[178:183], a[236:239], v14, v13 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[148:153], v[178:183], a[232:235], v14, v13 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[118:123], v[178:183], a[228:231], v14, v13 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(14) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[112:117], v[178:183], a[224:227], v14, v13 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_barrier
	ds_read_b64 v[12:13], v6
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[64:69], v[40:45], a[0:3], v10, v8 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[14:15], v5 offset:1024
	s_add_u32 m0, s30, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[70:75], v[40:45], a[4:7], v10, v8 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[88:93], v[40:45], a[8:11], v10, v8 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[28:31], v7 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[94:99], v[40:45], a[12:15], v10, v8 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[112:115], v3 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[124:129], v[40:45], a[16:19], v11, v8 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[130:135], v[40:45], a[20:23], v11, v8 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[76:81], v[40:45], a[24:27], v11, v8 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[32:35], v7 offset:6160
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[82:87], v[40:45], a[28:31], v11, v8 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[116:119], v3 offset:30736
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[82:87], v[46:51], a[60:63], v11, v8 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[76:81], v[46:51], a[56:59], v11, v8 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[130:135], v[46:51], a[52:55], v11, v8 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[36:39], v7 offset:6176
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[124:129], v[46:51], a[48:51], v11, v8 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[120:123], v3 offset:30752
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[94:99], v[46:51], a[44:47], v10, v8 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s31, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[88:93], v[46:51], a[40:43], v10, v8 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[70:75], v[46:51], a[36:39], v10, v8 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[40:43], v7 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[64:69], v[46:51], a[32:35], v10, v8 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[148:151], v3 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[64:69], v[52:57], a[64:67], v10, v8 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[70:75], v[52:57], a[68:71], v10, v8 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[88:93], v[52:57], a[72:75], v10, v8 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[44:47], v7 offset:9232
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[94:99], v[52:57], a[76:79], v10, v8 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[152:155], v3 offset:33808
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[124:129], v[52:57], a[80:83], v11, v8 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[130:135], v[52:57], a[84:87], v11, v8 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[76:81], v[52:57], a[88:91], v11, v8 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s4, s4, 0x10000
	s_addc_u32 s5, s5, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[82:87], v[52:57], a[92:95], v11, v8 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[48:51], v7 offset:9248
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[82:87], v[58:63], a[124:127], v11, v8 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[76:81], v[58:63], a[120:123], v11, v8 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[130:135], v[58:63], a[116:119], v11, v8 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[156:159], v3 offset:33824
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[124:129], v[58:63], a[112:115], v11, v8 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[52:55], v7 offset:12288
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[94:99], v[58:63], a[108:111], v10, v8 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[88:93], v[58:63], a[104:107], v10, v8 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[70:75], v[58:63], a[100:103], v10, v8 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[160:163], v3 offset:36864
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[64:69], v[58:63], a[96:99], v10, v8 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[56:59], v7 offset:12304
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[64:69], v[16:21], a[128:131], v10, v9 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[70:75], v[16:21], a[132:135], v10, v9 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[88:93], v[16:21], a[136:139], v10, v9 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[164:167], v3 offset:36880
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[94:99], v[16:21], a[140:143], v10, v9 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[60:63], v7 offset:12320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[124:129], v[16:21], a[144:147], v11, v9 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s17, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[130:135], v[16:21], a[148:151], v11, v9 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[76:81], v[16:21], a[152:155], v11, v9 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[168:171], v3 offset:36896
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[82:87], v[16:21], a[156:159], v11, v9 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[172:175], v7 offset:15360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[82:87], v[22:27], a[188:191], v11, v9 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[76:81], v[22:27], a[184:187], v11, v9 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[130:135], v[22:27], a[180:183], v11, v9 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[16:19], v3 offset:39936
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[124:129], v[22:27], a[176:179], v11, v9 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[176:179], v7 offset:15376
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[94:99], v[22:27], a[172:175], v10, v9 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[88:93], v[22:27], a[168:171], v10, v9 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[70:75], v[22:27], a[164:167], v10, v9 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_u32 s32, s32, 0x10000
	s_addc_u32 s33, s33, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[64:69], v[22:27], a[160:163], v10, v9 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[20:23], v3 offset:39952
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[64:69], v[136:141], a[192:195], v10, v9 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s36
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[70:75], v[136:141], a[196:199], v10, v9 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v4, s[8:11], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[88:93], v[136:141], a[200:203], v10, v9 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s8, s8, 0x4000
	s_addc_u32 s9, s9, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[94:99], v[136:141], a[204:207], v10, v9 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[180:183], v7 offset:15392
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[124:129], v[136:141], a[208:211], v11, v9 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s37
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[130:135], v[136:141], a[212:215], v11, v9 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v4, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[76:81], v[136:141], a[216:219], v11, v9 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s40, s40, 0x8000
	s_addc_u32 s41, s41, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[82:87], v[136:141], a[220:223], v11, v9 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[24:27], v3 offset:39968
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[82:87], v[142:147], a[252:255], v11, v9 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[76:81], v[142:147], a[248:251], v11, v9 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[130:135], v[142:147], a[244:247], v11, v9 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[124:129], v[142:147], a[240:243], v11, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[94:99], v[142:147], a[236:239], v10, v9 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[88:93], v[142:147], a[232:235], v10, v9 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[70:75], v[142:147], a[228:231], v10, v9 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(14) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[64:69], v[142:147], a[224:227], v10, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_barrier
	ds_read_b64 v[8:9], v6 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[112:117], v[28:33], a[0:3], v14, v12 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[10:11], v5 offset:3072
	s_add_u32 m0, s2, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[118:123], v[28:33], a[4:7], v14, v12 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[148:153], v[28:33], a[8:11], v14, v12 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[64:67], v7 offset:55296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[154:159], v[28:33], a[12:15], v14, v12 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[76:79], v108 offset:14336
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[160:165], v[28:33], a[16:19], v15, v12 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[166:171], v[28:33], a[20:23], v15, v12 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[16:21], v[28:33], a[24:27], v15, v12 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[68:71], v7 offset:55312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[22:27], v[28:33], a[28:31], v15, v12 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[80:83], v108 offset:14352
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[22:27], v[34:39], a[60:63], v15, v12 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[16:21], v[34:39], a[56:59], v15, v12 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[166:171], v[34:39], a[52:55], v15, v12 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[72:75], v7 offset:55328
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[160:165], v[34:39], a[48:51], v15, v12 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[84:87], v108 offset:14368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[154:159], v[34:39], a[44:47], v14, v12 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s18, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[148:153], v[34:39], a[40:43], v14, v12 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[118:123], v[34:39], a[36:39], v14, v12 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[28:31], v7 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[112:117], v[34:39], a[32:35], v14, v12 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[88:91], v108 offset:17408
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[112:117], v[40:45], a[64:67], v14, v12 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[118:123], v[40:45], a[68:71], v14, v12 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[148:153], v[40:45], a[72:75], v14, v12 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[32:35], v7 offset:58384
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[154:159], v[40:45], a[76:79], v14, v12 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[92:95], v108 offset:17424
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[160:165], v[40:45], a[80:83], v15, v12 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[166:171], v[40:45], a[84:87], v15, v12 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[16:21], v[40:45], a[88:91], v15, v12 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s4, s4, 0x10000
	s_addc_u32 s5, s5, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[22:27], v[40:45], a[92:95], v15, v12 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[36:39], v7 offset:58400
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[22:27], v[46:51], a[124:127], v15, v12 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s21
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[16:21], v[46:51], a[120:123], v15, v12 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[166:171], v[46:51], a[116:119], v15, v12 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[96:99], v108 offset:17440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[160:165], v[46:51], a[112:115], v15, v12 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[40:43], v7 offset:61440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[154:159], v[46:51], a[108:111], v14, v12 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[148:153], v[46:51], a[104:107], v14, v12 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[118:123], v[46:51], a[100:103], v14, v12 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[124:127], v108 offset:20480
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[112:117], v[46:51], a[96:99], v14, v12 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[44:47], v7 offset:61456
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[112:117], v[52:57], a[128:131], v14, v13 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[118:123], v[52:57], a[132:135], v14, v13 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[148:153], v[52:57], a[136:139], v14, v13 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[128:131], v108 offset:20496
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[154:159], v[52:57], a[140:143], v14, v13 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[48:51], v7 offset:61472
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[160:165], v[52:57], a[144:147], v15, v13 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s22, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[166:171], v[52:57], a[148:151], v15, v13 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[16:21], v[52:57], a[152:155], v15, v13 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[132:135], v108 offset:20512
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[22:27], v[52:57], a[156:159], v15, v13 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[136:139], v7 offset:64512
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[22:27], v[58:63], a[188:191], v15, v13 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[16:21], v[58:63], a[184:187], v15, v13 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[166:171], v[58:63], a[180:183], v15, v13 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[52:55], v108 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[160:165], v[58:63], a[176:179], v15, v13 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[140:143], v7 offset:64528
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[154:159], v[58:63], a[172:175], v14, v13 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[148:153], v[58:63], a[168:171], v14, v13 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[118:123], v[58:63], a[164:167], v14, v13 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_u32 s32, s32, 0x10000
	s_addc_u32 s33, s33, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[112:117], v[58:63], a[160:163], v14, v13 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[56:59], v108 offset:23568
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[112:117], v[172:177], a[192:195], v14, v13 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s20
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[118:123], v[172:177], a[196:199], v14, v13 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v4, s[8:11], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[148:153], v[172:177], a[200:203], v14, v13 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s8, s8, 0x4000
	s_addc_u32 s9, s9, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[154:159], v[172:177], a[204:207], v14, v13 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[144:147], v7 offset:64544
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[160:165], v[172:177], a[208:211], v15, v13 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s23
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[166:171], v[172:177], a[212:215], v15, v13 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v4, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[16:21], v[172:177], a[216:219], v15, v13 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s40, s40, 0x8000
	s_addc_u32 s41, s41, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[22:27], v[172:177], a[220:223], v15, v13 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[60:63], v108 offset:23584
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[22:27], v[178:183], a[252:255], v15, v13 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[16:21], v[178:183], a[248:251], v15, v13 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[166:171], v[178:183], a[244:247], v15, v13 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[160:165], v[178:183], a[240:243], v15, v13 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[154:159], v[178:183], a[236:239], v14, v13 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[148:153], v[178:183], a[232:235], v14, v13 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[118:123], v[178:183], a[228:231], v14, v13 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(14) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[112:117], v[178:183], a[224:227], v14, v13 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_barrier
	ds_read_b64 v[12:13], v6 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[76:81], v[64:69], a[0:3], v10, v8 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[14:15], v5 offset:5120
	s_add_u32 m0, s24, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[82:87], v[64:69], a[4:7], v10, v8 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[88:93], v[64:69], a[8:11], v10, v8 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[16:19], v109 offset:38912
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[94:99], v[64:69], a[12:15], v10, v8 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[112:115], v108 offset:63488
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[124:129], v[64:69], a[16:19], v11, v8 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[130:135], v[64:69], a[20:23], v11, v8 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[52:57], v[64:69], a[24:27], v11, v8 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[20:23], v109 offset:38928
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[58:63], v[64:69], a[28:31], v11, v8 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[116:119], v108 offset:63504
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[58:63], v[70:75], a[60:63], v11, v8 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[52:57], v[70:75], a[56:59], v11, v8 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[130:135], v[70:75], a[52:55], v11, v8 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[24:27], v109 offset:38944
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[124:129], v[70:75], a[48:51], v11, v8 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[120:123], v108 offset:63520
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[94:99], v[70:75], a[44:47], v10, v8 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s25, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[88:93], v[70:75], a[40:43], v10, v8 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[82:87], v[70:75], a[36:39], v10, v8 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[64:67], v109 offset:41984
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[76:81], v[70:75], a[32:35], v10, v8 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[148:151], v110 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[76:81], v[28:33], a[64:67], v10, v8 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[82:87], v[28:33], a[68:71], v10, v8 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[88:93], v[28:33], a[72:75], v10, v8 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[68:71], v109 offset:42000
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[94:99], v[28:33], a[76:79], v10, v8 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[152:155], v110 offset:1040
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[124:129], v[28:33], a[80:83], v11, v8 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[130:135], v[28:33], a[84:87], v11, v8 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[52:57], v[28:33], a[88:91], v11, v8 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s4, s4, 0x10000
	s_addc_u32 s5, s5, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[58:63], v[28:33], a[92:95], v11, v8 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[72:75], v109 offset:42016
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[58:63], v[34:39], a[124:127], v11, v8 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s26
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[52:57], v[34:39], a[120:123], v11, v8 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[130:135], v[34:39], a[116:119], v11, v8 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[156:159], v110 offset:1056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[124:129], v[34:39], a[112:115], v11, v8 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[28:31], v109 offset:45056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[94:99], v[34:39], a[108:111], v10, v8 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[88:93], v[34:39], a[104:107], v10, v8 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[82:87], v[34:39], a[100:103], v10, v8 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[160:163], v110 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[76:81], v[34:39], a[96:99], v10, v8 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[32:35], v109 offset:45072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[76:81], v[40:45], a[128:131], v10, v9 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[82:87], v[40:45], a[132:135], v10, v9 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[88:93], v[40:45], a[136:139], v10, v9 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[164:167], v110 offset:4112
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[94:99], v[40:45], a[140:143], v10, v9 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[36:39], v109 offset:45088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[124:129], v[40:45], a[144:147], v11, v9 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s27, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[130:135], v[40:45], a[148:151], v11, v9 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[52:57], v[40:45], a[152:155], v11, v9 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[168:171], v110 offset:4128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[58:63], v[40:45], a[156:159], v11, v9 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[172:175], v109 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[58:63], v[46:51], a[188:191], v11, v9 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[52:57], v[46:51], a[184:187], v11, v9 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[130:135], v[46:51], a[180:183], v11, v9 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[40:43], v110 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[124:129], v[46:51], a[176:179], v11, v9 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[176:179], v109 offset:48144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[94:99], v[46:51], a[172:175], v10, v9 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[88:93], v[46:51], a[168:171], v10, v9 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[82:87], v[46:51], a[164:167], v10, v9 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_u32 s32, s32, 0x10000
	s_addc_u32 s33, s33, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[76:81], v[46:51], a[160:163], v10, v9 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[44:47], v110 offset:7184
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[76:81], v[136:141], a[192:195], v10, v9 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s28
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[82:87], v[136:141], a[196:199], v10, v9 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v4, s[8:11], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[88:93], v[136:141], a[200:203], v10, v9 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s8, s8, 0x4000
	s_addc_u32 s9, s9, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[94:99], v[136:141], a[204:207], v10, v9 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[180:183], v109 offset:48160
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[124:129], v[136:141], a[208:211], v11, v9 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s29
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[130:135], v[136:141], a[212:215], v11, v9 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v4, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[52:57], v[136:141], a[216:219], v11, v9 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s40, s40, 0x8000
	s_addc_u32 s41, s41, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[58:63], v[136:141], a[220:223], v11, v9 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[48:51], v110 offset:7200
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[58:63], v[142:147], a[252:255], v11, v9 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[52:57], v[142:147], a[248:251], v11, v9 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[130:135], v[142:147], a[244:247], v11, v9 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[124:129], v[142:147], a[240:243], v11, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[94:99], v[142:147], a[236:239], v10, v9 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[88:93], v[142:147], a[232:235], v10, v9 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[82:87], v[142:147], a[228:231], v10, v9 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(14) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[76:81], v[142:147], a[224:227], v10, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_barrier
	ds_read_b64 v[8:9], v6
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[112:117], v[16:21], a[0:3], v14, v12 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[10:11], v5 offset:1024
	s_add_u32 m0, s30, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[118:123], v[16:21], a[4:7], v14, v12 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[148:153], v[16:21], a[8:11], v14, v12 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[52:55], v7 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[154:159], v[16:21], a[12:15], v14, v12 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[76:79], v3 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[160:165], v[16:21], a[16:19], v15, v12 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[166:171], v[16:21], a[20:23], v15, v12 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[40:45], v[16:21], a[24:27], v15, v12 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[56:59], v7 offset:6160
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[46:51], v[16:21], a[28:31], v15, v12 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[80:83], v3 offset:30736
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[46:51], v[22:27], a[60:63], v15, v12 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[40:45], v[22:27], a[56:59], v15, v12 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[166:171], v[22:27], a[52:55], v15, v12 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[60:63], v7 offset:6176
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[160:165], v[22:27], a[48:51], v15, v12 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[84:87], v3 offset:30752
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[154:159], v[22:27], a[44:47], v14, v12 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s31, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[148:153], v[22:27], a[40:43], v14, v12 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[118:123], v[22:27], a[36:39], v14, v12 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[16:19], v7 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[112:117], v[22:27], a[32:35], v14, v12 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[88:91], v3 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[112:117], v[64:69], a[64:67], v14, v12 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[118:123], v[64:69], a[68:71], v14, v12 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[148:153], v[64:69], a[72:75], v14, v12 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[20:23], v7 offset:9232
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[154:159], v[64:69], a[76:79], v14, v12 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[92:95], v3 offset:33808
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[160:165], v[64:69], a[80:83], v15, v12 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[166:171], v[64:69], a[84:87], v15, v12 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[40:45], v[64:69], a[88:91], v15, v12 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s4, s4, 0x10000
	s_addc_u32 s5, s5, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[46:51], v[64:69], a[92:95], v15, v12 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[24:27], v7 offset:9248
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[46:51], v[70:75], a[124:127], v15, v12 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[40:45], v[70:75], a[120:123], v15, v12 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[166:171], v[70:75], a[116:119], v15, v12 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[96:99], v3 offset:33824
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[160:165], v[70:75], a[112:115], v15, v12 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[64:67], v7 offset:12288
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[154:159], v[70:75], a[108:111], v14, v12 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[148:153], v[70:75], a[104:107], v14, v12 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[118:123], v[70:75], a[100:103], v14, v12 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[124:127], v3 offset:36864
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[112:117], v[70:75], a[96:99], v14, v12 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[68:71], v7 offset:12304
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[112:117], v[28:33], a[128:131], v14, v13 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[118:123], v[28:33], a[132:135], v14, v13 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[148:153], v[28:33], a[136:139], v14, v13 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[128:131], v3 offset:36880
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[154:159], v[28:33], a[140:143], v14, v13 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[72:75], v7 offset:12320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[160:165], v[28:33], a[144:147], v15, v13 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s17, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[166:171], v[28:33], a[148:151], v15, v13 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[40:45], v[28:33], a[152:155], v15, v13 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[132:135], v3 offset:36896
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[46:51], v[28:33], a[156:159], v15, v13 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[136:139], v7 offset:15360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[46:51], v[34:39], a[188:191], v15, v13 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[40:45], v[34:39], a[184:187], v15, v13 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[166:171], v[34:39], a[180:183], v15, v13 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[28:31], v3 offset:39936
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[160:165], v[34:39], a[176:179], v15, v13 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[140:143], v7 offset:15376
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[154:159], v[34:39], a[172:175], v14, v13 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[148:153], v[34:39], a[168:171], v14, v13 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[118:123], v[34:39], a[164:167], v14, v13 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_u32 s32, s32, 0x10000
	s_addc_u32 s33, s33, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[112:117], v[34:39], a[160:163], v14, v13 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[32:35], v3 offset:39952
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[112:117], v[172:177], a[192:195], v14, v13 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s36
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[118:123], v[172:177], a[196:199], v14, v13 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v4, s[8:11], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[148:153], v[172:177], a[200:203], v14, v13 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s8, s8, 0x4000
	s_addc_u32 s9, s9, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[154:159], v[172:177], a[204:207], v14, v13 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[144:147], v7 offset:15392
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[160:165], v[172:177], a[208:211], v15, v13 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s37
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[166:171], v[172:177], a[212:215], v15, v13 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v4, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[40:45], v[172:177], a[216:219], v15, v13 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s40, s40, 0x8000
	s_addc_u32 s41, s41, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[46:51], v[172:177], a[220:223], v15, v13 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[36:39], v3 offset:39968
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[46:51], v[178:183], a[252:255], v15, v13 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[40:45], v[178:183], a[248:251], v15, v13 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[166:171], v[178:183], a[244:247], v15, v13 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[160:165], v[178:183], a[240:243], v15, v13 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[154:159], v[178:183], a[236:239], v14, v13 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[148:153], v[178:183], a[232:235], v14, v13 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[118:123], v[178:183], a[228:231], v14, v13 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(14) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[112:117], v[178:183], a[224:227], v14, v13 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_barrier
	ds_read_b64 v[12:13], v6 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[76:81], v[52:57], a[0:3], v10, v8 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[14:15], v5 offset:3072
	s_add_u32 m0, s2, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[82:87], v[52:57], a[4:7], v10, v8 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[88:93], v[52:57], a[8:11], v10, v8 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[40:43], v7 offset:55296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[94:99], v[52:57], a[12:15], v10, v8 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[112:115], v108 offset:14336
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[124:129], v[52:57], a[16:19], v11, v8 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[130:135], v[52:57], a[20:23], v11, v8 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[28:33], v[52:57], a[24:27], v11, v8 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[44:47], v7 offset:55312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[34:39], v[52:57], a[28:31], v11, v8 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[116:119], v108 offset:14352
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[34:39], v[58:63], a[60:63], v11, v8 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[28:33], v[58:63], a[56:59], v11, v8 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[130:135], v[58:63], a[52:55], v11, v8 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[48:51], v7 offset:55328
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[124:129], v[58:63], a[48:51], v11, v8 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[120:123], v108 offset:14368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[94:99], v[58:63], a[44:47], v10, v8 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s18, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[88:93], v[58:63], a[40:43], v10, v8 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[82:87], v[58:63], a[36:39], v10, v8 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[52:55], v7 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[76:81], v[58:63], a[32:35], v10, v8 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[148:151], v108 offset:17408
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[76:81], v[16:21], a[64:67], v10, v8 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[82:87], v[16:21], a[68:71], v10, v8 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[88:93], v[16:21], a[72:75], v10, v8 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[56:59], v7 offset:58384
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[94:99], v[16:21], a[76:79], v10, v8 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[152:155], v108 offset:17424
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[124:129], v[16:21], a[80:83], v11, v8 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[130:135], v[16:21], a[84:87], v11, v8 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[28:33], v[16:21], a[88:91], v11, v8 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s4, s4, 0x10000
	s_addc_u32 s5, s5, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[34:39], v[16:21], a[92:95], v11, v8 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[60:63], v7 offset:58400
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[34:39], v[22:27], a[124:127], v11, v8 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s21
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[28:33], v[22:27], a[120:123], v11, v8 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[130:135], v[22:27], a[116:119], v11, v8 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[156:159], v108 offset:17440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[124:129], v[22:27], a[112:115], v11, v8 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[16:19], v7 offset:61440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[94:99], v[22:27], a[108:111], v10, v8 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[88:93], v[22:27], a[104:107], v10, v8 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[82:87], v[22:27], a[100:103], v10, v8 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[160:163], v108 offset:20480
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[76:81], v[22:27], a[96:99], v10, v8 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[20:23], v7 offset:61456
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[76:81], v[64:69], a[128:131], v10, v9 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[82:87], v[64:69], a[132:135], v10, v9 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[88:93], v[64:69], a[136:139], v10, v9 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[164:167], v108 offset:20496
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[94:99], v[64:69], a[140:143], v10, v9 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[24:27], v7 offset:61472
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[124:129], v[64:69], a[144:147], v11, v9 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s22, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[130:135], v[64:69], a[148:151], v11, v9 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[28:33], v[64:69], a[152:155], v11, v9 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[168:171], v108 offset:20512
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[34:39], v[64:69], a[156:159], v11, v9 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[172:175], v7 offset:64512
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[34:39], v[70:75], a[188:191], v11, v9 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[28:33], v[70:75], a[184:187], v11, v9 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[130:135], v[70:75], a[180:183], v11, v9 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[64:67], v108 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[124:129], v[70:75], a[176:179], v11, v9 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[176:179], v7 offset:64528
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[94:99], v[70:75], a[172:175], v10, v9 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[88:93], v[70:75], a[168:171], v10, v9 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[82:87], v[70:75], a[164:167], v10, v9 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_u32 s32, s32, 0x10000
	s_addc_u32 s33, s33, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[76:81], v[70:75], a[160:163], v10, v9 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[68:71], v108 offset:23568
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[76:81], v[136:141], a[192:195], v10, v9 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s20
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[82:87], v[136:141], a[196:199], v10, v9 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v4, s[8:11], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[88:93], v[136:141], a[200:203], v10, v9 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s8, s8, 0x4000
	s_addc_u32 s9, s9, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[94:99], v[136:141], a[204:207], v10, v9 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[180:183], v7 offset:64544
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[124:129], v[136:141], a[208:211], v11, v9 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s23
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[130:135], v[136:141], a[212:215], v11, v9 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v4, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[28:33], v[136:141], a[216:219], v11, v9 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s40, s40, 0x8000
	s_addc_u32 s41, s41, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[34:39], v[136:141], a[220:223], v11, v9 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[72:75], v108 offset:23584
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[34:39], v[142:147], a[252:255], v11, v9 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[28:33], v[142:147], a[248:251], v11, v9 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[130:135], v[142:147], a[244:247], v11, v9 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[124:129], v[142:147], a[240:243], v11, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[94:99], v[142:147], a[236:239], v10, v9 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[88:93], v[142:147], a[232:235], v10, v9 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[82:87], v[142:147], a[228:231], v10, v9 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(14) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[76:81], v[142:147], a[224:227], v10, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_barrier
	ds_read_b64 v[8:9], v6 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[112:117], v[40:45], a[0:3], v14, v12 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[10:11], v5 offset:5120
	s_add_u32 m0, s24, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[118:123], v[40:45], a[4:7], v14, v12 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[148:153], v[40:45], a[8:11], v14, v12 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[28:31], v109 offset:38912
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[154:159], v[40:45], a[12:15], v14, v12 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[76:79], v108 offset:63488
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[160:165], v[40:45], a[16:19], v15, v12 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[166:171], v[40:45], a[20:23], v15, v12 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[64:69], v[40:45], a[24:27], v15, v12 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[32:35], v109 offset:38928
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[70:75], v[40:45], a[28:31], v15, v12 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[80:83], v108 offset:63504
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[70:75], v[46:51], a[60:63], v15, v12 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[64:69], v[46:51], a[56:59], v15, v12 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[166:171], v[46:51], a[52:55], v15, v12 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[36:39], v109 offset:38944
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[160:165], v[46:51], a[48:51], v15, v12 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[84:87], v108 offset:63520
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[154:159], v[46:51], a[44:47], v14, v12 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s25, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[148:153], v[46:51], a[40:43], v14, v12 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[118:123], v[46:51], a[36:39], v14, v12 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[40:43], v109 offset:41984
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[112:117], v[46:51], a[32:35], v14, v12 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[88:91], v110 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[112:117], v[52:57], a[64:67], v14, v12 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[118:123], v[52:57], a[68:71], v14, v12 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[148:153], v[52:57], a[72:75], v14, v12 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[44:47], v109 offset:42000
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[154:159], v[52:57], a[76:79], v14, v12 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[92:95], v110 offset:1040
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[160:165], v[52:57], a[80:83], v15, v12 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[166:171], v[52:57], a[84:87], v15, v12 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[64:69], v[52:57], a[88:91], v15, v12 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s4, s4, 0x10000
	s_addc_u32 s5, s5, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[70:75], v[52:57], a[92:95], v15, v12 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[48:51], v109 offset:42016
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[70:75], v[58:63], a[124:127], v15, v12 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s26
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[64:69], v[58:63], a[120:123], v15, v12 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[166:171], v[58:63], a[116:119], v15, v12 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[96:99], v110 offset:1056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[160:165], v[58:63], a[112:115], v15, v12 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[52:55], v109 offset:45056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[154:159], v[58:63], a[108:111], v14, v12 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[148:153], v[58:63], a[104:107], v14, v12 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[118:123], v[58:63], a[100:103], v14, v12 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[124:127], v110 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[112:117], v[58:63], a[96:99], v14, v12 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[56:59], v109 offset:45072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[112:117], v[16:21], a[128:131], v14, v13 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[118:123], v[16:21], a[132:135], v14, v13 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[148:153], v[16:21], a[136:139], v14, v13 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[128:131], v110 offset:4112
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[154:159], v[16:21], a[140:143], v14, v13 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[60:63], v109 offset:45088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[160:165], v[16:21], a[144:147], v15, v13 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s27, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[166:171], v[16:21], a[148:151], v15, v13 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[64:69], v[16:21], a[152:155], v15, v13 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[132:135], v110 offset:4128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[70:75], v[16:21], a[156:159], v15, v13 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[136:139], v109 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[70:75], v[22:27], a[188:191], v15, v13 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[64:69], v[22:27], a[184:187], v15, v13 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[166:171], v[22:27], a[180:183], v15, v13 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[16:19], v110 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[160:165], v[22:27], a[176:179], v15, v13 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[140:143], v109 offset:48144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[154:159], v[22:27], a[172:175], v14, v13 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[148:153], v[22:27], a[168:171], v14, v13 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[118:123], v[22:27], a[164:167], v14, v13 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_u32 s32, s32, 0x10000
	s_addc_u32 s33, s33, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[112:117], v[22:27], a[160:163], v14, v13 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[20:23], v110 offset:7184
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[112:117], v[172:177], a[192:195], v14, v13 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s28
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[118:123], v[172:177], a[196:199], v14, v13 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v4, s[8:11], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[148:153], v[172:177], a[200:203], v14, v13 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s8, s8, 0x4000
	s_addc_u32 s9, s9, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[154:159], v[172:177], a[204:207], v14, v13 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[144:147], v109 offset:48160
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[160:165], v[172:177], a[208:211], v15, v13 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s29
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[166:171], v[172:177], a[212:215], v15, v13 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v4, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[64:69], v[172:177], a[216:219], v15, v13 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s40, s40, 0x8000
	s_addc_u32 s41, s41, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[70:75], v[172:177], a[220:223], v15, v13 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[24:27], v110 offset:7200
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[70:75], v[178:183], a[252:255], v15, v13 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[64:69], v[178:183], a[248:251], v15, v13 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[166:171], v[178:183], a[244:247], v15, v13 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[160:165], v[178:183], a[240:243], v15, v13 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[154:159], v[178:183], a[236:239], v14, v13 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[148:153], v[178:183], a[232:235], v14, v13 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[118:123], v[178:183], a[228:231], v14, v13 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(14) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[112:117], v[178:183], a[224:227], v14, v13 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_barrier
	ds_read_b64 v[12:13], v6
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[76:81], v[28:33], a[0:3], v10, v8 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[14:15], v5 offset:1024
	s_add_u32 m0, s30, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[82:87], v[28:33], a[4:7], v10, v8 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[88:93], v[28:33], a[8:11], v10, v8 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[64:67], v7 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[94:99], v[28:33], a[12:15], v10, v8 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[112:115], v3 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[124:129], v[28:33], a[16:19], v11, v8 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[130:135], v[28:33], a[20:23], v11, v8 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[16:21], v[28:33], a[24:27], v11, v8 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[68:71], v7 offset:6160
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[22:27], v[28:33], a[28:31], v11, v8 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[116:119], v3 offset:30736
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[22:27], v[34:39], a[60:63], v11, v8 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[16:21], v[34:39], a[56:59], v11, v8 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[130:135], v[34:39], a[52:55], v11, v8 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[72:75], v7 offset:6176
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[124:129], v[34:39], a[48:51], v11, v8 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[120:123], v3 offset:30752
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[94:99], v[34:39], a[44:47], v10, v8 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s31, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[88:93], v[34:39], a[40:43], v10, v8 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[82:87], v[34:39], a[36:39], v10, v8 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[28:31], v7 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[76:81], v[34:39], a[32:35], v10, v8 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[148:151], v3 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[76:81], v[40:45], a[64:67], v10, v8 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[82:87], v[40:45], a[68:71], v10, v8 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[88:93], v[40:45], a[72:75], v10, v8 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[32:35], v7 offset:9232
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[94:99], v[40:45], a[76:79], v10, v8 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[152:155], v3 offset:33808
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[124:129], v[40:45], a[80:83], v11, v8 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[130:135], v[40:45], a[84:87], v11, v8 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[16:21], v[40:45], a[88:91], v11, v8 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s4, s4, 0x290000
	s_addc_u32 s5, s5, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[22:27], v[40:45], a[92:95], v11, v8 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[36:39], v7 offset:9248
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[22:27], v[46:51], a[124:127], v11, v8 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[16:21], v[46:51], a[120:123], v11, v8 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[130:135], v[46:51], a[116:119], v11, v8 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[156:159], v3 offset:33824
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[124:129], v[46:51], a[112:115], v11, v8 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[40:43], v7 offset:12288
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[94:99], v[46:51], a[108:111], v10, v8 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[88:93], v[46:51], a[104:107], v10, v8 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[82:87], v[46:51], a[100:103], v10, v8 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[160:163], v3 offset:36864
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[76:81], v[46:51], a[96:99], v10, v8 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[44:47], v7 offset:12304
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[76:81], v[52:57], a[128:131], v10, v9 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[82:87], v[52:57], a[132:135], v10, v9 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[88:93], v[52:57], a[136:139], v10, v9 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[164:167], v3 offset:36880
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[94:99], v[52:57], a[140:143], v10, v9 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[48:51], v7 offset:12320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[124:129], v[52:57], a[144:147], v11, v9 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s17, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[130:135], v[52:57], a[148:151], v11, v9 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[16:21], v[52:57], a[152:155], v11, v9 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[168:171], v3 offset:36896
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[22:27], v[52:57], a[156:159], v11, v9 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[172:175], v7 offset:15360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[22:27], v[58:63], a[188:191], v11, v9 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[16:21], v[58:63], a[184:187], v11, v9 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[130:135], v[58:63], a[180:183], v11, v9 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[52:55], v3 offset:39936
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[124:129], v[58:63], a[176:179], v11, v9 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[176:179], v7 offset:15376
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[94:99], v[58:63], a[172:175], v10, v9 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[88:93], v[58:63], a[168:171], v10, v9 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[82:87], v[58:63], a[164:167], v10, v9 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_u32 s32, s32, 0x290000
	s_addc_u32 s33, s33, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[76:81], v[58:63], a[160:163], v10, v9 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[56:59], v3 offset:39952
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[76:81], v[136:141], a[192:195], v10, v9 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s36
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[82:87], v[136:141], a[196:199], v10, v9 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v4, s[8:11], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[88:93], v[136:141], a[200:203], v10, v9 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s8, s8, 0x4000
	s_addc_u32 s9, s9, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[94:99], v[136:141], a[204:207], v10, v9 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[180:183], v7 offset:15392
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[124:129], v[136:141], a[208:211], v11, v9 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s37
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[130:135], v[136:141], a[212:215], v11, v9 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v4, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[16:21], v[136:141], a[216:219], v11, v9 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s40, s40, 0x348000
	s_addc_u32 s41, s41, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[22:27], v[136:141], a[220:223], v11, v9 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[60:63], v3 offset:39968
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[22:27], v[142:147], a[252:255], v11, v9 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[16:21], v[142:147], a[248:251], v11, v9 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[130:135], v[142:147], a[244:247], v11, v9 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[124:129], v[142:147], a[240:243], v11, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[94:99], v[142:147], a[236:239], v10, v9 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[88:93], v[142:147], a[232:235], v10, v9 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[82:87], v[142:147], a[228:231], v10, v9 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(14) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[76:81], v[142:147], a[224:227], v10, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_barrier
	ds_read_b64 v[8:9], v6 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[112:117], v[64:69], a[0:3], v14, v12 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[10:11], v5 offset:3072
	s_add_u32 m0, s2, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[118:123], v[64:69], a[4:7], v14, v12 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[148:153], v[64:69], a[8:11], v14, v12 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[16:19], v7 offset:55296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[154:159], v[64:69], a[12:15], v14, v12 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[76:79], v108 offset:14336
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[160:165], v[64:69], a[16:19], v15, v12 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[166:171], v[64:69], a[20:23], v15, v12 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[52:57], v[64:69], a[24:27], v15, v12 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[20:23], v7 offset:55312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[58:63], v[64:69], a[28:31], v15, v12 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[80:83], v108 offset:14352
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[58:63], v[70:75], a[60:63], v15, v12 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[52:57], v[70:75], a[56:59], v15, v12 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[166:171], v[70:75], a[52:55], v15, v12 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[24:27], v7 offset:55328
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[160:165], v[70:75], a[48:51], v15, v12 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[84:87], v108 offset:14368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[154:159], v[70:75], a[44:47], v14, v12 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s18, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[148:153], v[70:75], a[40:43], v14, v12 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[118:123], v[70:75], a[36:39], v14, v12 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[64:67], v7 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[112:117], v[70:75], a[32:35], v14, v12 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[88:91], v108 offset:17408
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[112:117], v[28:33], a[64:67], v14, v12 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[118:123], v[28:33], a[68:71], v14, v12 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[148:153], v[28:33], a[72:75], v14, v12 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[68:71], v7 offset:58384
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[154:159], v[28:33], a[76:79], v14, v12 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[92:95], v108 offset:17424
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[160:165], v[28:33], a[80:83], v15, v12 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[166:171], v[28:33], a[84:87], v15, v12 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[52:57], v[28:33], a[88:91], v15, v12 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s4, s4, 0x10000
	s_addc_u32 s5, s5, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[58:63], v[28:33], a[92:95], v15, v12 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[72:75], v7 offset:58400
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[58:63], v[34:39], a[124:127], v15, v12 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s21
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[52:57], v[34:39], a[120:123], v15, v12 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[166:171], v[34:39], a[116:119], v15, v12 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[96:99], v108 offset:17440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[160:165], v[34:39], a[112:115], v15, v12 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[28:31], v7 offset:61440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[154:159], v[34:39], a[108:111], v14, v12 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[148:153], v[34:39], a[104:107], v14, v12 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[118:123], v[34:39], a[100:103], v14, v12 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[124:127], v108 offset:20480
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[112:117], v[34:39], a[96:99], v14, v12 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[32:35], v7 offset:61456
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[112:117], v[40:45], a[128:131], v14, v13 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[118:123], v[40:45], a[132:135], v14, v13 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[148:153], v[40:45], a[136:139], v14, v13 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[128:131], v108 offset:20496
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[154:159], v[40:45], a[140:143], v14, v13 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[36:39], v7 offset:61472
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[160:165], v[40:45], a[144:147], v15, v13 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s22, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[166:171], v[40:45], a[148:151], v15, v13 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[52:57], v[40:45], a[152:155], v15, v13 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[132:135], v108 offset:20512
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[58:63], v[40:45], a[156:159], v15, v13 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[136:139], v7 offset:64512
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[58:63], v[46:51], a[188:191], v15, v13 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[52:57], v[46:51], a[184:187], v15, v13 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[166:171], v[46:51], a[180:183], v15, v13 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[40:43], v108 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[160:165], v[46:51], a[176:179], v15, v13 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[140:143], v7 offset:64528
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[154:159], v[46:51], a[172:175], v14, v13 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[148:153], v[46:51], a[168:171], v14, v13 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[118:123], v[46:51], a[164:167], v14, v13 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_u32 s32, s32, 0x10000
	s_addc_u32 s33, s33, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[112:117], v[46:51], a[160:163], v14, v13 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[44:47], v108 offset:23568
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[112:117], v[172:177], a[192:195], v14, v13 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s20
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[118:123], v[172:177], a[196:199], v14, v13 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v4, s[8:11], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[148:153], v[172:177], a[200:203], v14, v13 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s8, s8, 0x4000
	s_addc_u32 s9, s9, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[154:159], v[172:177], a[204:207], v14, v13 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[144:147], v7 offset:64544
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[160:165], v[172:177], a[208:211], v15, v13 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s23
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[166:171], v[172:177], a[212:215], v15, v13 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v4, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[52:57], v[172:177], a[216:219], v15, v13 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s40, s40, 0x8000
	s_addc_u32 s41, s41, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[58:63], v[172:177], a[220:223], v15, v13 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[48:51], v108 offset:23584
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[58:63], v[178:183], a[252:255], v15, v13 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[52:57], v[178:183], a[248:251], v15, v13 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[166:171], v[178:183], a[244:247], v15, v13 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[160:165], v[178:183], a[240:243], v15, v13 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[154:159], v[178:183], a[236:239], v14, v13 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[148:153], v[178:183], a[232:235], v14, v13 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[118:123], v[178:183], a[228:231], v14, v13 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(14) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[112:117], v[178:183], a[224:227], v14, v13 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_barrier
	ds_read_b64 v[112:113], v6 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[76:81], v[16:21], a[0:3], v10, v8 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[114:115], v5 offset:5120
	s_add_u32 m0, s24, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[82:87], v[16:21], a[4:7], v10, v8 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[88:93], v[16:21], a[8:11], v10, v8 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[52:55], v109 offset:38912
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[94:99], v[16:21], a[12:15], v10, v8 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[148:151], v108 offset:63488
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[124:129], v[16:21], a[16:19], v11, v8 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[130:135], v[16:21], a[20:23], v11, v8 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[40:45], v[16:21], a[24:27], v11, v8 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[56:59], v109 offset:38928
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[46:51], v[16:21], a[28:31], v11, v8 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[152:155], v108 offset:63504
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[46:51], v[22:27], a[60:63], v11, v8 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[40:45], v[22:27], a[56:59], v11, v8 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[130:135], v[22:27], a[52:55], v11, v8 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[60:63], v109 offset:38944
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[124:129], v[22:27], a[48:51], v11, v8 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[156:159], v108 offset:63520
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[94:99], v[22:27], a[44:47], v10, v8 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s25, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[88:93], v[22:27], a[40:43], v10, v8 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[82:87], v[22:27], a[36:39], v10, v8 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[160:163], v109 offset:41984
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[76:81], v[22:27], a[32:35], v10, v8 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[172:175], v110 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[76:81], v[64:69], a[64:67], v10, v8 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[82:87], v[64:69], a[68:71], v10, v8 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[88:93], v[64:69], a[72:75], v10, v8 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[164:167], v109 offset:42000
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[94:99], v[64:69], a[76:79], v10, v8 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[176:179], v110 offset:1040
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[124:129], v[64:69], a[80:83], v11, v8 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[130:135], v[64:69], a[84:87], v11, v8 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[40:45], v[64:69], a[88:91], v11, v8 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s4, s4, 0x10000
	s_addc_u32 s5, s5, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[46:51], v[64:69], a[92:95], v11, v8 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[168:171], v109 offset:42016
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[46:51], v[70:75], a[124:127], v11, v8 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s26
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[40:45], v[70:75], a[120:123], v11, v8 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[130:135], v[70:75], a[116:119], v11, v8 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[180:183], v110 offset:1056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[124:129], v[70:75], a[112:115], v11, v8 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[184:187], v109 offset:45056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[94:99], v[70:75], a[108:111], v10, v8 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[88:93], v[70:75], a[104:107], v10, v8 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[82:87], v[70:75], a[100:103], v10, v8 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[196:199], v110 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[76:81], v[70:75], a[96:99], v10, v8 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[188:191], v109 offset:45072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[76:81], v[28:33], a[128:131], v10, v9 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[82:87], v[28:33], a[132:135], v10, v9 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[88:93], v[28:33], a[136:139], v10, v9 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[200:203], v110 offset:4112
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[94:99], v[28:33], a[140:143], v10, v9 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[192:195], v109 offset:45088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[124:129], v[28:33], a[144:147], v11, v9 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s27, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[130:135], v[28:33], a[148:151], v11, v9 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[40:45], v[28:33], a[152:155], v11, v9 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[204:207], v110 offset:4128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[46:51], v[28:33], a[156:159], v11, v9 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[208:211], v109 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[46:51], v[34:39], a[188:191], v11, v9 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[40:45], v[34:39], a[184:187], v11, v9 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[130:135], v[34:39], a[180:183], v11, v9 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[220:223], v110 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[124:129], v[34:39], a[176:179], v11, v9 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[212:215], v109 offset:48144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[94:99], v[34:39], a[172:175], v10, v9 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[88:93], v[34:39], a[168:171], v10, v9 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[82:87], v[34:39], a[164:167], v10, v9 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_u32 s32, s32, 0x10000
	s_addc_u32 s33, s33, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[76:81], v[34:39], a[160:163], v10, v9 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[224:227], v110 offset:7184
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[76:81], v[136:141], a[192:195], v10, v9 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s28
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[82:87], v[136:141], a[196:199], v10, v9 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v4, s[8:11], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[88:93], v[136:141], a[200:203], v10, v9 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s8, s8, 0x4000
	s_addc_u32 s9, s9, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[94:99], v[136:141], a[204:207], v10, v9 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[216:219], v109 offset:48160
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[124:129], v[136:141], a[208:211], v11, v9 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s29
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[130:135], v[136:141], a[212:215], v11, v9 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v4, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[40:45], v[136:141], a[216:219], v11, v9 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s40, s40, 0x8000
	s_addc_u32 s41, s41, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[46:51], v[136:141], a[220:223], v11, v9 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[228:231], v110 offset:7200
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[46:51], v[142:147], a[252:255], v11, v9 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[40:45], v[142:147], a[248:251], v11, v9 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[130:135], v[142:147], a[244:247], v11, v9 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[124:129], v[142:147], a[240:243], v11, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[94:99], v[142:147], a[236:239], v10, v9 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[88:93], v[142:147], a[232:235], v10, v9 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[82:87], v[142:147], a[228:231], v10, v9 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(14) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[76:81], v[142:147], a[224:227], v10, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_barrier
	ds_read_b64 v[8:9], v6
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[148:153], v[52:57], a[0:3], v114, v112 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[10:11], v5 offset:1024
	s_add_u32 m0, s30, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[154:159], v[52:57], a[4:7], v114, v112 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[172:177], v[52:57], a[8:11], v114, v112 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[12:15], v7 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[178:183], v[52:57], a[12:15], v114, v112 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[24:27], v3 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[196:201], v[52:57], a[16:19], v115, v112 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[202:207], v[52:57], a[20:23], v115, v112 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[220:225], v[52:57], a[24:27], v115, v112 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[16:19], v7 offset:6160
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[226:231], v[52:57], a[28:31], v115, v112 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[28:31], v3 offset:30736
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[226:231], v[58:63], a[60:63], v115, v112 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[220:225], v[58:63], a[56:59], v115, v112 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[202:207], v[58:63], a[52:55], v115, v112 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[20:23], v7 offset:6176
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[196:201], v[58:63], a[48:51], v115, v112 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[32:35], v3 offset:30752
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[178:183], v[58:63], a[44:47], v114, v112 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s31, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[172:177], v[58:63], a[40:43], v114, v112 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[154:159], v[58:63], a[36:39], v114, v112 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[72:75], v7 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[148:153], v[58:63], a[32:35], v114, v112 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[36:39], v3 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[148:153], v[160:165], a[64:67], v114, v112 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[154:159], v[160:165], a[68:71], v114, v112 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[172:177], v[160:165], a[72:75], v114, v112 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[76:79], v7 offset:9232
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[178:183], v[160:165], a[76:79], v114, v112 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[40:43], v3 offset:33808
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[196:201], v[160:165], a[80:83], v115, v112 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[202:207], v[160:165], a[84:87], v115, v112 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[220:225], v[160:165], a[88:91], v115, v112 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s4, s4, 0x10000
	s_addc_u32 s5, s5, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[226:231], v[160:165], a[92:95], v115, v112 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[80:83], v7 offset:9248
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[226:231], v[166:171], a[124:127], v115, v112 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[220:225], v[166:171], a[120:123], v115, v112 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[202:207], v[166:171], a[116:119], v115, v112 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[44:47], v3 offset:33824
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[196:201], v[166:171], a[112:115], v115, v112 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[84:87], v7 offset:12288
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[178:183], v[166:171], a[108:111], v114, v112 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[172:177], v[166:171], a[104:107], v114, v112 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[154:159], v[166:171], a[100:103], v114, v112 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[48:51], v3 offset:36864
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[148:153], v[166:171], a[96:99], v114, v112 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[88:91], v7 offset:12304
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[148:153], v[184:189], a[128:131], v114, v113 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[154:159], v[184:189], a[132:135], v114, v113 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[172:177], v[184:189], a[136:139], v114, v113 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[52:55], v3 offset:36880
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[178:183], v[184:189], a[140:143], v114, v113 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[92:95], v7 offset:12320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[196:201], v[184:189], a[144:147], v115, v113 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s17, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[202:207], v[184:189], a[148:151], v115, v113 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[220:225], v[184:189], a[152:155], v115, v113 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[56:59], v3 offset:36896
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[226:231], v[184:189], a[156:159], v115, v113 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[96:99], v7 offset:15360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[226:231], v[190:195], a[188:191], v115, v113 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[220:225], v[190:195], a[184:187], v115, v113 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[202:207], v[190:195], a[180:183], v115, v113 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[60:63], v3 offset:39936
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[196:201], v[190:195], a[176:179], v115, v113 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[100:103], v7 offset:15376
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[178:183], v[190:195], a[172:175], v114, v113 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[172:177], v[190:195], a[168:171], v114, v113 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[154:159], v[190:195], a[164:167], v114, v113 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_u32 s32, s32, 0x10000
	s_addc_u32 s33, s33, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[148:153], v[190:195], a[160:163], v114, v113 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[64:67], v3 offset:39952
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[148:153], v[208:213], a[192:195], v114, v113 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s36
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[154:159], v[208:213], a[196:199], v114, v113 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v4, s[8:11], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[172:177], v[208:213], a[200:203], v114, v113 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s8, s8, 0x4000
	s_addc_u32 s9, s9, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[178:183], v[208:213], a[204:207], v114, v113 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[104:107], v7 offset:15392
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[196:201], v[208:213], a[208:211], v115, v113 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s37
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[202:207], v[208:213], a[212:215], v115, v113 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v4, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[220:225], v[208:213], a[216:219], v115, v113 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s40, s40, 0x8000
	s_addc_u32 s41, s41, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[226:231], v[208:213], a[220:223], v115, v113 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[68:71], v3 offset:39968
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[226:231], v[214:219], a[252:255], v115, v113 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[220:225], v[214:219], a[248:251], v115, v113 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[202:207], v[214:219], a[244:247], v115, v113 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[196:201], v[214:219], a[240:243], v115, v113 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[178:183], v[214:219], a[236:239], v114, v113 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[172:177], v[214:219], a[232:235], v114, v113 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[154:159], v[214:219], a[228:231], v114, v113 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(14) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[148:153], v[214:219], a[224:227], v114, v113 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_add_i32 s1, s1, 1
	s_cmp_lt_i32 s1, 21
	s_cbranch_scc1 .Lmain_loop
	s_barrier
	ds_read_b64 v[108:109], v6 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[24:29], v[12:17], a[0:3], v10, v8 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[110:111], v5 offset:3072
	s_mov_b32 s6, 0xdc00
	s_mov_b32 s7, 0x31014000
	s_mov_b32 m0, s2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[30:35], v[12:17], a[4:7], v10, v8 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[36:41], v[12:17], a[8:11], v10, v8 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[112:115], v7 offset:55296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[42:47], v[12:17], a[12:15], v10, v8 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_add_u32_e32 v124, 0x10000, v3
	ds_read_b128 v[126:129], v124 offset:14336
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[48:53], v[12:17], a[16:19], v11, v8 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[54:59], v[12:17], a[20:23], v11, v8 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[60:65], v[12:17], a[24:27], v11, v8 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[116:119], v7 offset:55312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[66:71], v[12:17], a[28:31], v11, v8 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[130:133], v124 offset:14352
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[66:71], v[18:23], a[60:63], v11, v8 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[60:65], v[18:23], a[56:59], v11, v8 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[54:59], v[18:23], a[52:55], v11, v8 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[120:123], v7 offset:55328
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[48:53], v[18:23], a[48:51], v11, v8 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[134:137], v124 offset:14368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[42:47], v[18:23], a[44:47], v10, v8 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s18, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[36:41], v[18:23], a[40:43], v10, v8 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[30:35], v[18:23], a[36:39], v10, v8 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[12:15], v7 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[24:29], v[18:23], a[32:35], v10, v8 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[138:141], v124 offset:17408
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[24:29], v[72:77], a[64:67], v10, v8 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[30:35], v[72:77], a[68:71], v10, v8 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[36:41], v[72:77], a[72:75], v10, v8 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[16:19], v7 offset:58384
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[42:47], v[72:77], a[76:79], v10, v8 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[142:145], v124 offset:17424
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[48:53], v[72:77], a[80:83], v11, v8 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[54:59], v[72:77], a[84:87], v11, v8 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[60:65], v[72:77], a[88:91], v11, v8 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s4, s4, 0x10000
	s_addc_u32 s5, s5, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[66:71], v[72:77], a[92:95], v11, v8 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[20:23], v7 offset:58400
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[66:71], v[78:83], a[124:127], v11, v8 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[60:65], v[78:83], a[120:123], v11, v8 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_mov_b32 s34, 0xdc00
	s_add_u32 m0, s21, lit(0x0)
	s_mov_b32 s35, 0x31014000
	buffer_load_dwordx4 v1, s[32:35], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[54:59], v[78:83], a[116:119], v11, v8 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[146:149], v124 offset:17440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[48:53], v[78:83], a[112:115], v11, v8 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[42:47], v[78:83], a[108:111], v10, v8 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[72:75], v7 offset:61440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[36:41], v[78:83], a[104:107], v10, v8 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[30:35], v[78:83], a[100:103], v10, v8 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[150:153], v124 offset:20480
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[24:29], v[78:83], a[96:99], v10, v8 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[76:79], v7 offset:61456
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[24:29], v[84:89], a[128:131], v10, v9 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[30:35], v[84:89], a[132:135], v10, v9 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[36:41], v[84:89], a[136:139], v10, v9 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[154:157], v124 offset:20496
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[42:47], v[84:89], a[140:143], v10, v9 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[80:83], v7 offset:61472
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[48:53], v[84:89], a[144:147], v11, v9 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s22, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[54:59], v[84:89], a[148:151], v11, v9 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[60:65], v[84:89], a[152:155], v11, v9 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[158:161], v124 offset:20512
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[66:71], v[84:89], a[156:159], v11, v9 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[66:71], v[90:95], a[188:191], v11, v9 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[162:165], v7 offset:64512
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[60:65], v[90:95], a[184:187], v11, v9 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[54:59], v[90:95], a[180:183], v11, v9 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[84:87], v124 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[48:53], v[90:95], a[176:179], v11, v9 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[166:169], v7 offset:64528
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[42:47], v[90:95], a[172:175], v10, v9 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[36:41], v[90:95], a[168:171], v10, v9 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[30:35], v[90:95], a[164:167], v10, v9 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_u32 s32, s32, 0x10000
	s_addc_u32 s33, s33, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[24:29], v[90:95], a[160:163], v10, v9 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[88:91], v124 offset:23568
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[24:29], v[96:101], a[192:195], v10, v9 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 s10, 0x400
	s_mov_b32 s11, 0x31014000
	s_mov_b32 m0, s20
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[30:35], v[96:101], a[196:199], v10, v9 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v4, s[8:11], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[36:41], v[96:101], a[200:203], v10, v9 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s8, s8, 0x4000
	s_addc_u32 s9, s9, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[42:47], v[96:101], a[204:207], v10, v9 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[170:173], v7 offset:64544
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[48:53], v[96:101], a[208:211], v11, v9 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[54:59], v[96:101], a[212:215], v11, v9 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 s42, 0x400
	s_mov_b32 m0, s23
	s_mov_b32 s43, 0x31014000
	buffer_load_dword v4, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[60:65], v[96:101], a[216:219], v11, v9 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s40, s40, 0x8000
	s_addc_u32 s41, s41, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[66:71], v[96:101], a[220:223], v11, v9 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[92:95], v124 offset:23584
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[66:71], v[102:107], a[252:255], v11, v9 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[60:65], v[102:107], a[248:251], v11, v9 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[54:59], v[102:107], a[244:247], v11, v9 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[48:53], v[102:107], a[240:243], v11, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[42:47], v[102:107], a[236:239], v10, v9 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[36:41], v[102:107], a[232:235], v10, v9 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[30:35], v[102:107], a[228:231], v10, v9 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[24:29], v[102:107], a[224:227], v10, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_barrier
	ds_read_b64 v[8:9], v6 offset:4096
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[126:131], v[112:117], a[0:3], v110, v108 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[10:11], v5 offset:5120
	s_add_u32 m0, s24, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[132:137], v[112:117], a[4:7], v110, v108 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[138:143], v[112:117], a[8:11], v110, v108 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_add_u32_e32 v24, 0x10000, v7
	ds_read_b128 v[26:29], v24 offset:38912
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[144:149], v[112:117], a[12:15], v110, v108 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[38:41], v124 offset:63488
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[150:155], v[112:117], a[16:19], v111, v108 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[156:161], v[112:117], a[20:23], v111, v108 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[84:89], v[112:117], a[24:27], v111, v108 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[30:33], v24 offset:38928
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[90:95], v[112:117], a[28:31], v111, v108 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[42:45], v124 offset:63504
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[90:95], v[118:123], a[60:63], v111, v108 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[84:89], v[118:123], a[56:59], v111, v108 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[156:161], v[118:123], a[52:55], v111, v108 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[34:37], v24 offset:38944
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[150:155], v[118:123], a[48:51], v111, v108 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[46:49], v124 offset:63520
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[144:149], v[118:123], a[44:47], v110, v108 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s25, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[138:143], v[118:123], a[40:43], v110, v108 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[132:137], v[118:123], a[36:39], v110, v108 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[50:53], v24 offset:41984
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[126:131], v[118:123], a[32:35], v110, v108 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_add_u32_e32 v25, 0x20000, v3
	ds_read_b128 v[96:99], v25 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[126:131], v[12:17], a[64:67], v110, v108 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[132:137], v[12:17], a[68:71], v110, v108 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[138:143], v[12:17], a[72:75], v110, v108 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[54:57], v24 offset:42000
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[144:149], v[12:17], a[76:79], v110, v108 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[100:103], v25 offset:1040
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[150:155], v[12:17], a[80:83], v111, v108 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[156:161], v[12:17], a[84:87], v111, v108 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[84:89], v[12:17], a[88:91], v111, v108 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s4, s4, 0x10000
	s_addc_u32 s5, s5, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[90:95], v[12:17], a[92:95], v111, v108 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[58:61], v24 offset:42016
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[90:95], v[18:23], a[124:127], v111, v108 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 m0, s26, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[84:89], v[18:23], a[120:123], v111, v108 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[156:161], v[18:23], a[116:119], v111, v108 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[104:107], v25 offset:1056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[150:155], v[18:23], a[112:115], v111, v108 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[12:15], v24 offset:45056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[144:149], v[18:23], a[108:111], v110, v108 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[138:143], v[18:23], a[104:107], v110, v108 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[132:137], v[18:23], a[100:103], v110, v108 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[112:115], v25 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[126:131], v[18:23], a[96:99], v110, v108 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[16:19], v24 offset:45072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[126:131], v[72:77], a[128:131], v110, v109 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[132:137], v[72:77], a[132:135], v110, v109 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[138:143], v[72:77], a[136:139], v110, v109 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[116:119], v25 offset:4112
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[144:149], v[72:77], a[140:143], v110, v109 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[20:23], v24 offset:45088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[150:155], v[72:77], a[144:147], v111, v109 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s27, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[156:161], v[72:77], a[148:151], v111, v109 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[84:89], v[72:77], a[152:155], v111, v109 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[120:123], v25 offset:4128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[90:95], v[72:77], a[156:159], v111, v109 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[62:65], v24 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[90:95], v[78:83], a[188:191], v111, v109 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[84:89], v[78:83], a[184:187], v111, v109 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[156:161], v[78:83], a[180:183], v111, v109 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[174:177], v25 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[150:155], v[78:83], a[176:179], v111, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[66:69], v24 offset:48144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[144:149], v[78:83], a[172:175], v110, v109 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[138:143], v[78:83], a[168:171], v110, v109 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[132:137], v[78:83], a[164:167], v110, v109 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_u32 s32, s32, 0x10000
	s_addc_u32 s33, s33, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[126:131], v[78:83], a[160:163], v110, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[178:181], v25 offset:7184
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[126:131], v[162:167], a[192:195], v110, v109 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s28
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[132:137], v[162:167], a[196:199], v110, v109 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v4, s[8:11], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[138:143], v[162:167], a[200:203], v110, v109 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s8, s8, 0x4000
	s_addc_u32 s9, s9, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[144:149], v[162:167], a[204:207], v110, v109 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[70:73], v24 offset:48160
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[150:155], v[162:167], a[208:211], v111, v109 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s29
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[156:161], v[162:167], a[212:215], v111, v109 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v4, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[84:89], v[162:167], a[216:219], v111, v109 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s40, s40, 0x8000
	s_addc_u32 s41, s41, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[90:95], v[162:167], a[220:223], v111, v109 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[182:185], v25 offset:7200
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[90:95], v[168:173], a[252:255], v111, v109 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[84:89], v[168:173], a[248:251], v111, v109 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[156:161], v[168:173], a[244:247], v111, v109 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[150:155], v[168:173], a[240:243], v111, v109 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[144:149], v[168:173], a[236:239], v110, v109 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[138:143], v[168:173], a[232:235], v110, v109 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[132:137], v[168:173], a[228:231], v110, v109 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(14) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[126:131], v[168:173], a[224:227], v110, v109 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_barrier
	ds_read_b64 v[74:75], v6
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[38:43], v[26:31], a[0:3], v10, v8 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[76:77], v5 offset:1024
	s_mov_b32 m0, s30
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[44:49], v[26:31], a[4:7], v10, v8 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[96:101], v[26:31], a[8:11], v10, v8 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[78:81], v7 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[102:107], v[26:31], a[12:15], v10, v8 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[126:129], v3 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[112:117], v[26:31], a[16:19], v11, v8 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[118:123], v[26:31], a[20:23], v11, v8 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[174:179], v[26:31], a[24:27], v11, v8 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[82:85], v7 offset:6160
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[180:185], v[26:31], a[28:31], v11, v8 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[130:133], v3 offset:30736
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[180:185], v[32:37], a[60:63], v11, v8 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[174:179], v[32:37], a[56:59], v11, v8 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[118:123], v[32:37], a[52:55], v11, v8 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[86:89], v7 offset:6176
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[112:117], v[32:37], a[48:51], v11, v8 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[134:137], v3 offset:30752
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[102:107], v[32:37], a[44:47], v10, v8 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s31, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[96:101], v[32:37], a[40:43], v10, v8 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[44:49], v[32:37], a[36:39], v10, v8 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[26:29], v7 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[38:43], v[32:37], a[32:35], v10, v8 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[138:141], v3 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[38:43], v[50:55], a[64:67], v10, v8 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[44:49], v[50:55], a[68:71], v10, v8 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[96:101], v[50:55], a[72:75], v10, v8 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[30:33], v7 offset:9232
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[102:107], v[50:55], a[76:79], v10, v8 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[142:145], v3 offset:33808
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[112:117], v[50:55], a[80:83], v11, v8 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[118:123], v[50:55], a[84:87], v11, v8 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[174:179], v[50:55], a[88:91], v11, v8 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s4, s4, 0x10000
	s_addc_u32 s5, s5, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[180:185], v[50:55], a[92:95], v11, v8 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[34:37], v7 offset:9248
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[180:185], v[56:61], a[124:127], v11, v8 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 m0, s0, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[174:179], v[56:61], a[120:123], v11, v8 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[118:123], v[56:61], a[116:119], v11, v8 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[146:149], v3 offset:33824
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[112:117], v[56:61], a[112:115], v11, v8 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[50:53], v7 offset:12288
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[102:107], v[56:61], a[108:111], v10, v8 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[96:101], v[56:61], a[104:107], v10, v8 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[44:49], v[56:61], a[100:103], v10, v8 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[150:153], v3 offset:36864
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[38:43], v[56:61], a[96:99], v10, v8 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[54:57], v7 offset:12304
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[38:43], v[12:17], a[128:131], v10, v9 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[44:49], v[12:17], a[132:135], v10, v9 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[32:35], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[96:101], v[12:17], a[136:139], v10, v9 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[154:157], v3 offset:36880
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[102:107], v[12:17], a[140:143], v10, v9 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[58:61], v7 offset:12320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[112:117], v[12:17], a[144:147], v11, v9 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s17, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[118:123], v[12:17], a[148:151], v11, v9 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[174:179], v[12:17], a[152:155], v11, v9 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[158:161], v3 offset:36896
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[180:185], v[12:17], a[156:159], v11, v9 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[162:165], v7 offset:15360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[180:185], v[18:23], a[188:191], v11, v9 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[174:179], v[18:23], a[184:187], v11, v9 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[118:123], v[18:23], a[180:183], v11, v9 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[12:15], v3 offset:39936
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[112:117], v[18:23], a[176:179], v11, v9 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[166:169], v7 offset:15376
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[102:107], v[18:23], a[172:175], v10, v9 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[96:101], v[18:23], a[168:171], v10, v9 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[32:35], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[44:49], v[18:23], a[164:167], v10, v9 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_u32 s0, s32, 0x10000
	s_addc_u32 s1, s33, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[38:43], v[18:23], a[160:163], v10, v9 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[16:19], v3 offset:39952
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[38:43], v[62:67], a[192:195], v10, v9 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s36
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[44:49], v[62:67], a[196:199], v10, v9 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v4, s[8:11], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[96:101], v[62:67], a[200:203], v10, v9 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s8, s8, 0x4000
	s_addc_u32 s9, s9, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[102:107], v[62:67], a[204:207], v10, v9 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[170:173], v7 offset:15392
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[112:117], v[62:67], a[208:211], v11, v9 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s37
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[118:123], v[62:67], a[212:215], v11, v9 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v4, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[174:179], v[62:67], a[216:219], v11, v9 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s32, s40, 0x8000
	s_addc_u32 s33, s41, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[180:185], v[62:67], a[220:223], v11, v9 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[20:23], v3 offset:39968
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[180:185], v[68:73], a[252:255], v11, v9 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[174:179], v[68:73], a[248:251], v11, v9 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[118:123], v[68:73], a[244:247], v11, v9 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[112:117], v[68:73], a[240:243], v11, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[102:107], v[68:73], a[236:239], v10, v9 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[96:101], v[68:73], a[232:235], v10, v9 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[44:49], v[68:73], a[228:231], v10, v9 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(14) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[38:43], v[68:73], a[224:227], v10, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_barrier
	ds_read_b64 v[8:9], v6 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[126:131], v[78:83], a[0:3], v76, v74 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[10:11], v5 offset:3072
	s_mov_b32 m0, s2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[132:137], v[78:83], a[4:7], v76, v74 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[138:143], v[78:83], a[8:11], v76, v74 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[38:41], v7 offset:55296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[144:149], v[78:83], a[12:15], v76, v74 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[62:65], v124 offset:14336
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[150:155], v[78:83], a[16:19], v77, v74 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[156:161], v[78:83], a[20:23], v77, v74 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[12:17], v[78:83], a[24:27], v77, v74 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[42:45], v7 offset:55312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[18:23], v[78:83], a[28:31], v77, v74 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[66:69], v124 offset:14352
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[18:23], v[84:89], a[60:63], v77, v74 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[12:17], v[84:89], a[56:59], v77, v74 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[156:161], v[84:89], a[52:55], v77, v74 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[46:49], v7 offset:55328
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[150:155], v[84:89], a[48:51], v77, v74 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[70:73], v124 offset:14368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[144:149], v[84:89], a[44:47], v76, v74 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s18, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[138:143], v[84:89], a[40:43], v76, v74 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[132:137], v[84:89], a[36:39], v76, v74 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[78:81], v7 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[126:131], v[84:89], a[32:35], v76, v74 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[90:93], v124 offset:17408
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[126:131], v[26:31], a[64:67], v76, v74 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[132:137], v[26:31], a[68:71], v76, v74 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[138:143], v[26:31], a[72:75], v76, v74 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[82:85], v7 offset:58384
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[144:149], v[26:31], a[76:79], v76, v74 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[94:97], v124 offset:17424
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[150:155], v[26:31], a[80:83], v77, v74 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[156:161], v[26:31], a[84:87], v77, v74 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[12:17], v[26:31], a[88:91], v77, v74 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s4, s4, 0x10000
	s_addc_u32 s5, s5, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[18:23], v[26:31], a[92:95], v77, v74 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[86:89], v7 offset:58400
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[18:23], v[32:37], a[124:127], v77, v74 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_mov_b32 s2, 0xdc00
	s_mov_b32 s3, 0x31014000
	s_add_u32 m0, s21, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[12:17], v[32:37], a[120:123], v77, v74 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[0:3], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[156:161], v[32:37], a[116:119], v77, v74 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[98:101], v124 offset:17440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[150:155], v[32:37], a[112:115], v77, v74 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[26:29], v7 offset:61440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[144:149], v[32:37], a[108:111], v76, v74 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[138:143], v[32:37], a[104:107], v76, v74 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[0:3], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[132:137], v[32:37], a[100:103], v76, v74 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[102:105], v124 offset:20480
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[126:131], v[32:37], a[96:99], v76, v74 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[30:33], v7 offset:61456
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[126:131], v[50:55], a[128:131], v76, v75 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[132:137], v[50:55], a[132:135], v76, v75 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[0:3], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[138:143], v[50:55], a[136:139], v76, v75 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[106:109], v124 offset:20496
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[144:149], v[50:55], a[140:143], v76, v75 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[34:37], v7 offset:61472
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[150:155], v[50:55], a[144:147], v77, v75 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s22, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[156:161], v[50:55], a[148:151], v77, v75 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[0:3], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[12:17], v[50:55], a[152:155], v77, v75 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[110:113], v124 offset:20512
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[18:23], v[50:55], a[156:159], v77, v75 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[174:177], v7 offset:64512
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[18:23], v[56:61], a[188:191], v77, v75 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[12:17], v[56:61], a[184:187], v77, v75 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[0:3], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[156:161], v[56:61], a[180:183], v77, v75 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[50:53], v124 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[150:155], v[56:61], a[176:179], v77, v75 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[178:181], v7 offset:64528
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[144:149], v[56:61], a[172:175], v76, v75 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[138:143], v[56:61], a[168:171], v76, v75 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[0:3], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[132:137], v[56:61], a[164:167], v76, v75 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_u32 s0, s0, 0x10000
	s_addc_u32 s1, s1, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[126:131], v[56:61], a[160:163], v76, v75 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[54:57], v124 offset:23568
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[126:131], v[162:167], a[192:195], v76, v75 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 m0, s20
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[132:137], v[162:167], a[196:199], v76, v75 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v4, s[8:11], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[138:143], v[162:167], a[200:203], v76, v75 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s8, s8, 0x4000
	s_addc_u32 s9, s9, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[144:149], v[162:167], a[204:207], v76, v75 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[182:185], v7 offset:64544
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[150:155], v[162:167], a[208:211], v77, v75 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 s34, 0x400
	s_mov_b32 m0, s23
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[156:161], v[162:167], a[212:215], v77, v75 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v4, s[32:35], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[12:17], v[162:167], a[216:219], v77, v75 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 s20, s32, 0x8000
	s_addc_u32 s21, s33, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[18:23], v[162:167], a[220:223], v77, v75 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[58:61], v124 offset:23584
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[18:23], v[168:173], a[252:255], v77, v75 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[12:17], v[168:173], a[248:251], v77, v75 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[156:161], v[168:173], a[244:247], v77, v75 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[150:155], v[168:173], a[240:243], v77, v75 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[144:149], v[168:173], a[236:239], v76, v75 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[138:143], v[168:173], a[232:235], v76, v75 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[132:137], v[168:173], a[228:231], v76, v75 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(14) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[126:131], v[168:173], a[224:227], v76, v75 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_barrier
	ds_read_b64 v[12:13], v6 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:67], v[38:43], a[0:3], v10, v8 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[14:15], v5 offset:5120
	s_add_u32 m0, s24, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[68:73], v[38:43], a[4:7], v10, v8 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[90:95], v[38:43], a[8:11], v10, v8 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[126:129], v24 offset:38912
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[96:101], v[38:43], a[12:15], v10, v8 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[138:141], v124 offset:63488
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[102:107], v[38:43], a[16:19], v11, v8 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[108:113], v[38:43], a[20:23], v11, v8 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[50:55], v[38:43], a[24:27], v11, v8 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[130:133], v24 offset:38928
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[56:61], v[38:43], a[28:31], v11, v8 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[142:145], v124 offset:63504
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[56:61], v[44:49], a[60:63], v11, v8 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[50:55], v[44:49], a[56:59], v11, v8 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[108:113], v[44:49], a[52:55], v11, v8 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[134:137], v24 offset:38944
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[102:107], v[44:49], a[48:51], v11, v8 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[146:149], v124 offset:63520
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[96:101], v[44:49], a[44:47], v10, v8 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s25, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[90:95], v[44:49], a[40:43], v10, v8 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[68:73], v[44:49], a[36:39], v10, v8 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[38:41], v24 offset:41984
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[62:67], v[44:49], a[32:35], v10, v8 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[150:153], v25 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[62:67], v[78:83], a[64:67], v10, v8 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[68:73], v[78:83], a[68:71], v10, v8 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[90:95], v[78:83], a[72:75], v10, v8 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[42:45], v24 offset:42000
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[96:101], v[78:83], a[76:79], v10, v8 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[154:157], v25 offset:1040
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[102:107], v[78:83], a[80:83], v11, v8 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[108:113], v[78:83], a[84:87], v11, v8 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[4:7], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[50:55], v[78:83], a[88:91], v11, v8 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[56:61], v[78:83], a[92:95], v11, v8 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[46:49], v24 offset:42016
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[56:61], v[84:89], a[124:127], v11, v8 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_u32 m0, s26, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[50:55], v[84:89], a[120:123], v11, v8 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[0:3], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[108:113], v[84:89], a[116:119], v11, v8 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[158:161], v25 offset:1056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[102:107], v[84:89], a[112:115], v11, v8 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[74:77], v24 offset:45056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[96:101], v[84:89], a[108:111], v10, v8 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[90:95], v[84:89], a[104:107], v10, v8 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[0:3], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[68:73], v[84:89], a[100:103], v10, v8 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[162:165], v25 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[62:67], v[84:89], a[96:99], v10, v8 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[78:81], v24 offset:45072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[62:67], v[26:31], a[128:131], v10, v9 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[68:73], v[26:31], a[132:135], v10, v9 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v1, s[0:3], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[90:95], v[26:31], a[136:139], v10, v9 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[166:169], v25 offset:4112
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[96:101], v[26:31], a[140:143], v10, v9 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[82:85], v24 offset:45088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[102:107], v[26:31], a[144:147], v11, v9 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_u32 m0, s27, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[108:113], v[26:31], a[148:151], v11, v9 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[0:3], 0 offen sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[50:55], v[26:31], a[152:155], v11, v9 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[170:173], v25 offset:4128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[56:61], v[26:31], a[156:159], v11, v9 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[186:189], v24 offset:48128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[56:61], v[32:37], a[188:191], v11, v9 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[50:55], v[32:37], a[184:187], v11, v9 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[0:3], 0 offen offset:1024 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[108:113], v[32:37], a[180:183], v11, v9 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[16:19], v25 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[102:107], v[32:37], a[176:179], v11, v9 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[190:193], v24 offset:48144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[96:101], v[32:37], a[172:175], v10, v9 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[90:95], v[32:37], a[168:171], v10, v9 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	buffer_load_dwordx4 v2, s[0:3], 0 offen offset:2048 sc0 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[68:73], v[32:37], a[164:167], v10, v9 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[62:67], v[32:37], a[160:163], v10, v9 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[20:23], v25 offset:7184
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[62:67], v[174:179], a[192:195], v10, v9 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_add_u32 m0, s28, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[68:73], v[174:179], a[196:199], v10, v9 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v4, s[8:11], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[90:95], v[174:179], a[200:203], v10, v9 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[96:101], v[174:179], a[204:207], v10, v9 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[194:197], v24 offset:48160
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[102:107], v[174:179], a[208:211], v11, v9 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_mov_b32 s22, 0x400
	s_mov_b32 s23, 0x31014000
	s_add_u32 m0, s29, lit(0x0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[108:113], v[174:179], a[212:215], v11, v9 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	buffer_load_dword v4, s[20:23], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[50:55], v[174:179], a[216:219], v11, v9 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[56:61], v[174:179], a[220:223], v11, v9 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[24:27], v25 offset:7200
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[56:61], v[180:185], a[252:255], v11, v9 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[50:55], v[180:185], a[248:251], v11, v9 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[108:113], v[180:185], a[244:247], v11, v9 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[102:107], v[180:185], a[240:243], v11, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[96:101], v[180:185], a[236:239], v10, v9 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[90:95], v[180:185], a[232:235], v10, v9 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[68:73], v[180:185], a[228:231], v10, v9 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[62:67], v[180:185], a[224:227], v10, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_barrier
	ds_read_b64 v[8:9], v6
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[138:143], v[126:131], a[0:3], v14, v12 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[10:11], v5 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[144:149], v[126:131], a[4:7], v14, v12 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[150:155], v[126:131], a[8:11], v14, v12 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[50:53], v7 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[156:161], v[126:131], a[12:15], v14, v12 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[62:65], v3 offset:30720
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[162:167], v[126:131], a[16:19], v15, v12 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[168:173], v[126:131], a[20:23], v15, v12 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[16:21], v[126:131], a[24:27], v15, v12 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[54:57], v7 offset:6160
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[22:27], v[126:131], a[28:31], v15, v12 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[66:69], v3 offset:30736
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[22:27], v[132:137], a[60:63], v15, v12 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[16:21], v[132:137], a[56:59], v15, v12 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[168:173], v[132:137], a[52:55], v15, v12 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[58:61], v7 offset:6176
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[162:167], v[132:137], a[48:51], v15, v12 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[70:73], v3 offset:30752
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[156:161], v[132:137], a[44:47], v14, v12 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[150:155], v[132:137], a[40:43], v14, v12 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[144:149], v[132:137], a[36:39], v14, v12 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[28:31], v7 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[138:143], v[132:137], a[32:35], v14, v12 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[86:89], v3 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[138:143], v[38:43], a[64:67], v14, v12 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[144:149], v[38:43], a[68:71], v14, v12 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[150:155], v[38:43], a[72:75], v14, v12 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[32:35], v7 offset:9232
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[156:161], v[38:43], a[76:79], v14, v12 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[90:93], v3 offset:33808
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[162:167], v[38:43], a[80:83], v15, v12 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[168:173], v[38:43], a[84:87], v15, v12 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[16:21], v[38:43], a[88:91], v15, v12 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[22:27], v[38:43], a[92:95], v15, v12 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[36:39], v7 offset:9248
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[22:27], v[44:49], a[124:127], v15, v12 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[16:21], v[44:49], a[120:123], v15, v12 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[168:173], v[44:49], a[116:119], v15, v12 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[94:97], v3 offset:33824
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[162:167], v[44:49], a[112:115], v15, v12 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[98:101], v7 offset:12288
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[156:161], v[44:49], a[108:111], v14, v12 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[150:155], v[44:49], a[104:107], v14, v12 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[144:149], v[44:49], a[100:103], v14, v12 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[110:113], v3 offset:36864
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[138:143], v[44:49], a[96:99], v14, v12 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[102:105], v7 offset:12304
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[138:143], v[74:79], a[128:131], v14, v13 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[144:149], v[74:79], a[132:135], v14, v13 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[150:155], v[74:79], a[136:139], v14, v13 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[114:117], v3 offset:36880
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[156:161], v[74:79], a[140:143], v14, v13 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[106:109], v7 offset:12320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[162:167], v[74:79], a[144:147], v15, v13 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[168:173], v[74:79], a[148:151], v15, v13 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[16:21], v[74:79], a[152:155], v15, v13 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[118:121], v3 offset:36896
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[22:27], v[74:79], a[156:159], v15, v13 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[126:129], v7 offset:15360
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[22:27], v[80:85], a[188:191], v15, v13 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[16:21], v[80:85], a[184:187], v15, v13 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[168:173], v[80:85], a[180:183], v15, v13 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[74:77], v3 offset:39936
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[162:167], v[80:85], a[176:179], v15, v13 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[130:133], v7 offset:15376
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[156:161], v[80:85], a[172:175], v14, v13 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[150:155], v[80:85], a[168:171], v14, v13 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[144:149], v[80:85], a[164:167], v14, v13 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[138:143], v[80:85], a[160:163], v14, v13 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[78:81], v3 offset:39952
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[138:143], v[186:191], a[192:195], v14, v13 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[144:149], v[186:191], a[196:199], v14, v13 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[150:155], v[186:191], a[200:203], v14, v13 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[156:161], v[186:191], a[204:207], v14, v13 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[134:137], v7 offset:15392
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[162:167], v[186:191], a[208:211], v15, v13 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[168:173], v[186:191], a[212:215], v15, v13 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[16:21], v[186:191], a[216:219], v15, v13 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[22:27], v[186:191], a[220:223], v15, v13 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[82:85], v3 offset:39968
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[22:27], v[192:197], a[252:255], v15, v13 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[16:21], v[192:197], a[248:251], v15, v13 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[168:173], v[192:197], a[244:247], v15, v13 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[162:167], v[192:197], a[240:243], v15, v13 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[156:161], v[192:197], a[236:239], v14, v13 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[150:155], v[192:197], a[232:235], v14, v13 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[144:149], v[192:197], a[228:231], v14, v13 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[138:143], v[192:197], a[224:227], v14, v13 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_barrier
	ds_read_b64 v[2:3], v6 offset:2048
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:67], v[50:55], a[0:3], v10, v8 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b64 v[4:5], v5 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[68:73], v[50:55], a[4:7], v10, v8 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[86:91], v[50:55], a[8:11], v10, v8 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[12:15], v7 offset:55296
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[92:97], v[50:55], a[12:15], v10, v8 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[40:43], v124 offset:14336
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[110:115], v[50:55], a[16:19], v11, v8 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[116:121], v[50:55], a[20:23], v11, v8 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[74:79], v[50:55], a[24:27], v11, v8 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[16:19], v7 offset:55312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[80:85], v[50:55], a[28:31], v11, v8 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[44:47], v124 offset:14352
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[80:85], v[56:61], a[60:63], v11, v8 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[74:79], v[56:61], a[56:59], v11, v8 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[116:121], v[56:61], a[52:55], v11, v8 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[20:23], v7 offset:55328
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[110:115], v[56:61], a[48:51], v11, v8 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[48:51], v124 offset:14368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[92:97], v[56:61], a[44:47], v10, v8 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[86:91], v[56:61], a[40:43], v10, v8 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[68:73], v[56:61], a[36:39], v10, v8 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[138:141], v7 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[62:67], v[56:61], a[32:35], v10, v8 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[150:153], v124 offset:17408
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[62:67], v[28:33], a[64:67], v10, v8 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[68:73], v[28:33], a[68:71], v10, v8 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[86:91], v[28:33], a[72:75], v10, v8 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[142:145], v7 offset:58384
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[92:97], v[28:33], a[76:79], v10, v8 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[154:157], v124 offset:17424
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[110:115], v[28:33], a[80:83], v11, v8 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[116:121], v[28:33], a[84:87], v11, v8 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[74:79], v[28:33], a[88:91], v11, v8 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[80:85], v[28:33], a[92:95], v11, v8 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[146:149], v7 offset:58400
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[80:85], v[34:39], a[124:127], v11, v8 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[74:79], v[34:39], a[120:123], v11, v8 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[116:121], v[34:39], a[116:119], v11, v8 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[158:161], v124 offset:17440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[110:115], v[34:39], a[112:115], v11, v8 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[24:27], v7 offset:61440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[92:97], v[34:39], a[108:111], v10, v8 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[86:91], v[34:39], a[104:107], v10, v8 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[68:73], v[34:39], a[100:103], v10, v8 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[162:165], v124 offset:20480
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[62:67], v[34:39], a[96:99], v10, v8 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	ds_read_b128 v[28:31], v7 offset:61456
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[62:67], v[98:103], a[128:131], v10, v9 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[68:73], v[98:103], a[132:135], v10, v9 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[86:91], v[98:103], a[136:139], v10, v9 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[166:169], v124 offset:20496
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[92:97], v[98:103], a[140:143], v10, v9 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[32:35], v7 offset:61472
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[110:115], v[98:103], a[144:147], v11, v9 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[116:121], v[98:103], a[148:151], v11, v9 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[74:79], v[98:103], a[152:155], v11, v9 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[170:173], v124 offset:20512
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[80:85], v[98:103], a[156:159], v11, v9 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	ds_read_b128 v[174:177], v7 offset:64512
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[80:85], v[104:109], a[188:191], v11, v9 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[74:79], v[104:109], a[184:187], v11, v9 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[116:121], v[104:109], a[180:183], v11, v9 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[98:101], v124 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[110:115], v[104:109], a[176:179], v11, v9 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[178:181], v7 offset:64528
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[92:97], v[104:109], a[172:175], v10, v9 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[86:91], v[104:109], a[168:171], v10, v9 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[68:73], v[104:109], a[164:167], v10, v9 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[62:67], v[104:109], a[160:163], v10, v9 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	ds_read_b128 v[102:105], v124 offset:23568
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[62:67], v[126:131], a[192:195], v10, v9 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[68:73], v[126:131], a[196:199], v10, v9 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[86:91], v[126:131], a[200:203], v10, v9 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[92:97], v[126:131], a[204:207], v10, v9 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[182:185], v7 offset:64544
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[110:115], v[126:131], a[208:211], v11, v9 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[116:121], v[126:131], a[212:215], v11, v9 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[74:79], v[126:131], a[216:219], v11, v9 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[80:85], v[126:131], a[220:223], v11, v9 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	ds_read_b128 v[106:109], v124 offset:23584
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[80:85], v[132:137], a[252:255], v11, v9 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[74:79], v[132:137], a[248:251], v11, v9 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[116:121], v[132:137], a[244:247], v11, v9 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[110:115], v[132:137], a[240:243], v11, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[92:97], v[132:137], a[236:239], v10, v9 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[86:91], v[132:137], a[232:235], v10, v9 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[68:73], v[132:137], a[228:231], v10, v9 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[62:67], v[132:137], a[224:227], v10, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(14)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[40:45], v[12:17], a[0:3], v4, v2 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[46:51], v[12:17], a[4:7], v4, v2 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[150:155], v[12:17], a[8:11], v4, v2 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(12)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[156:161], v[12:17], a[12:15], v4, v2 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(8)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[162:167], v[12:17], a[16:19], v5, v2 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[168:173], v[12:17], a[20:23], v5, v2 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[98:103], v[12:17], a[24:27], v5, v2 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[104:109], v[12:17], a[28:31], v5, v2 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[104:109], v[18:23], a[60:63], v5, v2 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_lshrrev_b32_e32 v1, 4, v0
	v_lshrrev_b32_e32 v6, 1, v1
	v_and_b32_e32 v1, 1, v1
	v_lshlrev_b32_e32 v6, 4, v6
	v_and_b32_e32 v0, 15, v0
	v_lshl_add_u32 v1, v1, 5, v6
	v_accvgpr_read_b32 v6, a0
	v_accvgpr_read_b32 v7, a1
	v_cvt_pk_bf16_f32 v6, v6, v7
	v_accvgpr_read_b32 v7, a2
	v_accvgpr_read_b32 v8, a3
	v_cvt_pk_bf16_f32 v7, v7, v8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[98:103], v[18:23], a[56:59], v5, v2 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v8, a4
	v_accvgpr_read_b32 v9, a5
	v_cvt_pk_bf16_f32 v8, v8, v9
	v_accvgpr_read_b32 v9, a6
	v_accvgpr_read_b32 v10, a7
	v_cvt_pk_bf16_f32 v9, v9, v10
	s_lshl_b32 s0, s19, 13
	s_lshl_b32 s1, s16, 1
	v_lshl_add_u32 v0, v0, 13, v1
	v_permlane16_swap_b32_e32 v6, v8
	v_permlane16_swap_b32_e32 v7, v9
	s_add_i32 s0, s0, s1
	s_mov_b32 s14, 0x2000000
	s_mov_b32 s15, 0x31014000
	buffer_store_dwordx4 v[6:9], v0, s[12:15], s0 offen sc1
	v_accvgpr_read_b32 v1, a8
	v_accvgpr_read_b32 v6, a9
	v_cvt_pk_bf16_f32 v6, v1, v6
	v_accvgpr_read_b32 v1, a10
	v_accvgpr_read_b32 v7, a11
	v_cvt_pk_bf16_f32 v7, v1, v7
	v_accvgpr_read_b32 v1, a12
	v_accvgpr_read_b32 v8, a13
	v_cvt_pk_bf16_f32 v8, v1, v8
	v_accvgpr_read_b32 v1, a14
	v_accvgpr_read_b32 v9, a15
	v_cvt_pk_bf16_f32 v9, v1, v9
	v_permlane16_swap_b32_e32 v6, v8
	s_nop 0
	v_permlane16_swap_b32_e32 v7, v9
	buffer_store_dwordx4 v[6:9], v0, s[12:15], s0 offen offset:64 sc1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[168:173], v[18:23], a[52:55], v5, v2 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v1, a16
	v_accvgpr_read_b32 v6, a17
	v_cvt_pk_bf16_f32 v6, v1, v6
	v_accvgpr_read_b32 v1, a18
	v_accvgpr_read_b32 v7, a19
	v_cvt_pk_bf16_f32 v7, v1, v7
	v_accvgpr_read_b32 v1, a20
	v_accvgpr_read_b32 v8, a21
	v_cvt_pk_bf16_f32 v8, v1, v8
	v_accvgpr_read_b32 v1, a22
	v_accvgpr_read_b32 v9, a23
	v_cvt_pk_bf16_f32 v9, v1, v9
	v_permlane16_swap_b32_e32 v6, v8
	s_nop 0
	v_permlane16_swap_b32_e32 v7, v9
	buffer_store_dwordx4 v[6:9], v0, s[12:15], s0 offen offset:128 sc1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[162:167], v[18:23], a[48:51], v5, v2 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v1, a24
	v_accvgpr_read_b32 v6, a25
	v_cvt_pk_bf16_f32 v6, v1, v6
	v_accvgpr_read_b32 v1, a26
	v_accvgpr_read_b32 v7, a27
	v_cvt_pk_bf16_f32 v7, v1, v7
	v_accvgpr_read_b32 v1, a28
	v_accvgpr_read_b32 v8, a29
	v_cvt_pk_bf16_f32 v8, v1, v8
	v_accvgpr_read_b32 v1, a30
	v_accvgpr_read_b32 v9, a31
	v_cvt_pk_bf16_f32 v9, v1, v9
	v_permlane16_swap_b32_e32 v6, v8
	s_nop 0
	v_permlane16_swap_b32_e32 v7, v9
	buffer_store_dwordx4 v[6:9], v0, s[12:15], s0 offen offset:192 sc1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[156:161], v[18:23], a[44:47], v4, v2 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[150:155], v[18:23], a[40:43], v4, v2 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[46:51], v[18:23], a[36:39], v4, v2 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[40:45], v[18:23], a[32:35], v4, v2 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[40:45], v[138:143], a[64:67], v4, v2 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_add_i32 s0, s19, 16
	s_nop 7
	s_nop 0
	v_accvgpr_read_b32 v1, a36
	v_accvgpr_read_b32 v6, a37
	v_cvt_pk_bf16_f32 v8, v1, v6
	v_accvgpr_read_b32 v1, a38
	v_accvgpr_read_b32 v6, a39
	v_cvt_pk_bf16_f32 v9, v1, v6
	v_accvgpr_read_b32 v1, a32
	v_accvgpr_read_b32 v6, a33
	v_cvt_pk_bf16_f32 v6, v1, v6
	v_accvgpr_read_b32 v1, a34
	v_accvgpr_read_b32 v7, a35
	v_cvt_pk_bf16_f32 v7, v1, v7
	s_lshl_b32 s0, s0, 13
	v_permlane16_swap_b32_e32 v6, v8
	v_permlane16_swap_b32_e32 v7, v9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[46:51], v[138:143], a[68:71], v4, v2 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_add_i32 s0, s1, s0
	buffer_store_dwordx4 v[6:9], v0, s[12:15], s0 offen sc1
	v_accvgpr_read_b32 v1, a40
	v_accvgpr_read_b32 v6, a41
	v_cvt_pk_bf16_f32 v6, v1, v6
	v_accvgpr_read_b32 v1, a42
	v_accvgpr_read_b32 v7, a43
	v_cvt_pk_bf16_f32 v7, v1, v7
	v_accvgpr_read_b32 v1, a44
	v_accvgpr_read_b32 v8, a45
	v_cvt_pk_bf16_f32 v8, v1, v8
	v_accvgpr_read_b32 v1, a46
	v_accvgpr_read_b32 v9, a47
	v_cvt_pk_bf16_f32 v9, v1, v9
	v_permlane16_swap_b32_e32 v6, v8
	s_nop 0
	v_permlane16_swap_b32_e32 v7, v9
	buffer_store_dwordx4 v[6:9], v0, s[12:15], s0 offen offset:64 sc1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[150:155], v[138:143], a[72:75], v4, v2 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v1, a48
	v_accvgpr_read_b32 v6, a49
	v_cvt_pk_bf16_f32 v6, v1, v6
	v_accvgpr_read_b32 v1, a50
	v_accvgpr_read_b32 v7, a51
	v_cvt_pk_bf16_f32 v7, v1, v7
	v_accvgpr_read_b32 v1, a52
	v_accvgpr_read_b32 v8, a53
	v_cvt_pk_bf16_f32 v8, v1, v8
	v_accvgpr_read_b32 v1, a54
	v_accvgpr_read_b32 v9, a55
	v_cvt_pk_bf16_f32 v9, v1, v9
	v_permlane16_swap_b32_e32 v6, v8
	s_nop 0
	v_permlane16_swap_b32_e32 v7, v9
	buffer_store_dwordx4 v[6:9], v0, s[12:15], s0 offen offset:128 sc1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[156:161], v[138:143], a[76:79], v4, v2 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v1, a56
	v_accvgpr_read_b32 v6, a57
	v_cvt_pk_bf16_f32 v6, v1, v6
	v_accvgpr_read_b32 v1, a58
	v_accvgpr_read_b32 v7, a59
	v_cvt_pk_bf16_f32 v7, v1, v7
	v_accvgpr_read_b32 v1, a60
	v_accvgpr_read_b32 v8, a61
	v_cvt_pk_bf16_f32 v8, v1, v8
	v_accvgpr_read_b32 v1, a62
	v_accvgpr_read_b32 v9, a63
	v_cvt_pk_bf16_f32 v9, v1, v9
	v_permlane16_swap_b32_e32 v6, v8
	s_nop 0
	v_permlane16_swap_b32_e32 v7, v9
	buffer_store_dwordx4 v[6:9], v0, s[12:15], s0 offen offset:192 sc1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[162:167], v[138:143], a[80:83], v5, v2 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[168:173], v[138:143], a[84:87], v5, v2 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[98:103], v[138:143], a[88:91], v5, v2 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[104:109], v[138:143], a[92:95], v5, v2 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[104:109], v[144:149], a[124:127], v5, v2 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_i32 s0, s19, 32
	v_accvgpr_read_b32 v1, a64
	v_accvgpr_read_b32 v6, a65
	v_cvt_pk_bf16_f32 v6, v1, v6
	v_accvgpr_read_b32 v1, a66
	v_accvgpr_read_b32 v7, a67
	v_cvt_pk_bf16_f32 v7, v1, v7
	v_accvgpr_read_b32 v1, a68
	v_accvgpr_read_b32 v8, a69
	v_cvt_pk_bf16_f32 v8, v1, v8
	v_accvgpr_read_b32 v1, a70
	v_accvgpr_read_b32 v9, a71
	v_cvt_pk_bf16_f32 v9, v1, v9
	s_lshl_b32 s0, s0, 13
	v_permlane16_swap_b32_e32 v6, v8
	v_permlane16_swap_b32_e32 v7, v9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[98:103], v[144:149], a[120:123], v5, v2 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_i32 s0, s1, s0
	buffer_store_dwordx4 v[6:9], v0, s[12:15], s0 offen sc1
	v_accvgpr_read_b32 v1, a72
	v_accvgpr_read_b32 v6, a73
	v_cvt_pk_bf16_f32 v6, v1, v6
	v_accvgpr_read_b32 v1, a74
	v_accvgpr_read_b32 v7, a75
	v_cvt_pk_bf16_f32 v7, v1, v7
	v_accvgpr_read_b32 v1, a76
	v_accvgpr_read_b32 v8, a77
	v_cvt_pk_bf16_f32 v8, v1, v8
	v_accvgpr_read_b32 v1, a78
	v_accvgpr_read_b32 v9, a79
	v_cvt_pk_bf16_f32 v9, v1, v9
	v_permlane16_swap_b32_e32 v6, v8
	s_nop 0
	v_permlane16_swap_b32_e32 v7, v9
	buffer_store_dwordx4 v[6:9], v0, s[12:15], s0 offen offset:64 sc1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[168:173], v[144:149], a[116:119], v5, v2 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v1, a80
	v_accvgpr_read_b32 v6, a81
	v_cvt_pk_bf16_f32 v6, v1, v6
	v_accvgpr_read_b32 v1, a82
	v_accvgpr_read_b32 v7, a83
	v_cvt_pk_bf16_f32 v7, v1, v7
	v_accvgpr_read_b32 v1, a84
	v_accvgpr_read_b32 v8, a85
	v_cvt_pk_bf16_f32 v8, v1, v8
	v_accvgpr_read_b32 v1, a86
	v_accvgpr_read_b32 v9, a87
	v_cvt_pk_bf16_f32 v9, v1, v9
	v_permlane16_swap_b32_e32 v6, v8
	s_nop 0
	v_permlane16_swap_b32_e32 v7, v9
	buffer_store_dwordx4 v[6:9], v0, s[12:15], s0 offen offset:128 sc1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[162:167], v[144:149], a[112:115], v5, v2 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v1, a88
	v_accvgpr_read_b32 v6, a89
	v_cvt_pk_bf16_f32 v6, v1, v6
	v_accvgpr_read_b32 v1, a90
	v_accvgpr_read_b32 v7, a91
	v_cvt_pk_bf16_f32 v7, v1, v7
	v_accvgpr_read_b32 v1, a92
	v_accvgpr_read_b32 v8, a93
	v_cvt_pk_bf16_f32 v8, v1, v8
	v_accvgpr_read_b32 v1, a94
	v_accvgpr_read_b32 v9, a95
	v_cvt_pk_bf16_f32 v9, v1, v9
	v_permlane16_swap_b32_e32 v6, v8
	s_nop 0
	v_permlane16_swap_b32_e32 v7, v9
	buffer_store_dwordx4 v[6:9], v0, s[12:15], s0 offen offset:192 sc1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[156:161], v[144:149], a[108:111], v4, v2 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[150:155], v[144:149], a[104:107], v4, v2 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[46:51], v[144:149], a[100:103], v4, v2 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[40:45], v[144:149], a[96:99], v4, v2 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[40:45], v[24:29], a[128:131], v4, v3 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_i32 s0, s19, 48
	s_nop 7
	s_nop 0
	v_accvgpr_read_b32 v1, a100
	v_accvgpr_read_b32 v2, a101
	v_cvt_pk_bf16_f32 v8, v1, v2
	v_accvgpr_read_b32 v1, a102
	v_accvgpr_read_b32 v2, a103
	v_cvt_pk_bf16_f32 v9, v1, v2
	v_accvgpr_read_b32 v1, a96
	v_accvgpr_read_b32 v2, a97
	v_cvt_pk_bf16_f32 v6, v1, v2
	v_accvgpr_read_b32 v1, a98
	v_accvgpr_read_b32 v2, a99
	v_cvt_pk_bf16_f32 v7, v1, v2
	s_lshl_b32 s0, s0, 13
	v_permlane16_swap_b32_e32 v6, v8
	v_permlane16_swap_b32_e32 v7, v9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[46:51], v[24:29], a[132:135], v4, v3 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	s_add_i32 s0, s1, s0
	buffer_store_dwordx4 v[6:9], v0, s[12:15], s0 offen sc1
	v_accvgpr_read_b32 v1, a104
	v_accvgpr_read_b32 v2, a105
	v_cvt_pk_bf16_f32 v6, v1, v2
	v_accvgpr_read_b32 v1, a106
	v_accvgpr_read_b32 v2, a107
	v_cvt_pk_bf16_f32 v7, v1, v2
	v_accvgpr_read_b32 v1, a108
	v_accvgpr_read_b32 v2, a109
	v_cvt_pk_bf16_f32 v8, v1, v2
	v_accvgpr_read_b32 v1, a110
	v_accvgpr_read_b32 v2, a111
	v_cvt_pk_bf16_f32 v9, v1, v2
	v_permlane16_swap_b32_e32 v6, v8
	s_nop 0
	v_permlane16_swap_b32_e32 v7, v9
	buffer_store_dwordx4 v[6:9], v0, s[12:15], s0 offen offset:64 sc1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[150:155], v[24:29], a[136:139], v4, v3 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v1, a112
	v_accvgpr_read_b32 v2, a113
	v_cvt_pk_bf16_f32 v6, v1, v2
	v_accvgpr_read_b32 v1, a114
	v_accvgpr_read_b32 v2, a115
	v_cvt_pk_bf16_f32 v7, v1, v2
	v_accvgpr_read_b32 v1, a116
	v_accvgpr_read_b32 v2, a117
	v_cvt_pk_bf16_f32 v8, v1, v2
	v_accvgpr_read_b32 v1, a118
	v_accvgpr_read_b32 v2, a119
	v_cvt_pk_bf16_f32 v9, v1, v2
	v_permlane16_swap_b32_e32 v6, v8
	s_nop 0
	v_permlane16_swap_b32_e32 v7, v9
	buffer_store_dwordx4 v[6:9], v0, s[12:15], s0 offen offset:128 sc1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[156:161], v[24:29], a[140:143], v4, v3 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v1, a120
	v_accvgpr_read_b32 v2, a121
	v_cvt_pk_bf16_f32 v6, v1, v2
	v_accvgpr_read_b32 v1, a122
	v_accvgpr_read_b32 v2, a123
	v_cvt_pk_bf16_f32 v7, v1, v2
	v_accvgpr_read_b32 v1, a124
	v_accvgpr_read_b32 v2, a125
	v_cvt_pk_bf16_f32 v8, v1, v2
	v_accvgpr_read_b32 v1, a126
	v_accvgpr_read_b32 v2, a127
	v_cvt_pk_bf16_f32 v9, v1, v2
	v_permlane16_swap_b32_e32 v6, v8
	s_nop 0
	v_permlane16_swap_b32_e32 v7, v9
	buffer_store_dwordx4 v[6:9], v0, s[12:15], s0 offen offset:192 sc1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[162:167], v[24:29], a[144:147], v5, v3 op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[168:173], v[24:29], a[148:151], v5, v3 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[98:103], v[24:29], a[152:155], v5, v3 op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[104:109], v[24:29], a[156:159], v5, v3 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[104:109], v[30:35], a[188:191], v5, v3 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s0, s19, 64
	v_accvgpr_read_b32 v1, a128
	v_accvgpr_read_b32 v2, a129
	v_cvt_pk_bf16_f32 v6, v1, v2
	v_accvgpr_read_b32 v1, a130
	v_accvgpr_read_b32 v2, a131
	v_cvt_pk_bf16_f32 v7, v1, v2
	v_accvgpr_read_b32 v1, a132
	v_accvgpr_read_b32 v2, a133
	v_cvt_pk_bf16_f32 v8, v1, v2
	v_accvgpr_read_b32 v1, a134
	v_accvgpr_read_b32 v2, a135
	v_cvt_pk_bf16_f32 v9, v1, v2
	s_lshl_b32 s0, s0, 13
	v_permlane16_swap_b32_e32 v6, v8
	v_permlane16_swap_b32_e32 v7, v9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[98:103], v[30:35], a[184:187], v5, v3 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	s_add_i32 s0, s1, s0
	buffer_store_dwordx4 v[6:9], v0, s[12:15], s0 offen sc1
	v_accvgpr_read_b32 v1, a136
	v_accvgpr_read_b32 v2, a137
	v_cvt_pk_bf16_f32 v6, v1, v2
	v_accvgpr_read_b32 v1, a138
	v_accvgpr_read_b32 v2, a139
	v_cvt_pk_bf16_f32 v7, v1, v2
	v_accvgpr_read_b32 v1, a140
	v_accvgpr_read_b32 v2, a141
	v_cvt_pk_bf16_f32 v8, v1, v2
	v_accvgpr_read_b32 v1, a142
	v_accvgpr_read_b32 v2, a143
	v_cvt_pk_bf16_f32 v9, v1, v2
	v_permlane16_swap_b32_e32 v6, v8
	s_nop 0
	v_permlane16_swap_b32_e32 v7, v9
	buffer_store_dwordx4 v[6:9], v0, s[12:15], s0 offen offset:64 sc1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[168:173], v[30:35], a[180:183], v5, v3 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v1, a144
	v_accvgpr_read_b32 v2, a145
	v_cvt_pk_bf16_f32 v6, v1, v2
	v_accvgpr_read_b32 v1, a146
	v_accvgpr_read_b32 v2, a147
	v_cvt_pk_bf16_f32 v7, v1, v2
	v_accvgpr_read_b32 v1, a148
	v_accvgpr_read_b32 v2, a149
	v_cvt_pk_bf16_f32 v8, v1, v2
	v_accvgpr_read_b32 v1, a150
	v_accvgpr_read_b32 v2, a151
	v_cvt_pk_bf16_f32 v9, v1, v2
	v_permlane16_swap_b32_e32 v6, v8
	s_nop 0
	v_permlane16_swap_b32_e32 v7, v9
	buffer_store_dwordx4 v[6:9], v0, s[12:15], s0 offen offset:128 sc1
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[162:167], v[30:35], a[176:179], v5, v3 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v1, a152
	v_accvgpr_read_b32 v2, a153
	v_cvt_pk_bf16_f32 v6, v1, v2
	v_accvgpr_read_b32 v1, a154
	v_accvgpr_read_b32 v2, a155
	v_cvt_pk_bf16_f32 v7, v1, v2
	v_accvgpr_read_b32 v1, a156
	v_accvgpr_read_b32 v2, a157
	v_cvt_pk_bf16_f32 v8, v1, v2
	v_accvgpr_read_b32 v1, a158
	v_accvgpr_read_b32 v2, a159
	v_cvt_pk_bf16_f32 v9, v1, v2
	v_permlane16_swap_b32_e32 v6, v8
	s_nop 0
	v_permlane16_swap_b32_e32 v7, v9
	buffer_store_dwordx4 v[6:9], v0, s[12:15], s0 offen offset:192 sc1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[156:161], v[30:35], a[172:175], v4, v3 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[150:155], v[30:35], a[168:171], v4, v3 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[46:51], v[30:35], a[164:167], v4, v3 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[40:45], v[30:35], a[160:163], v4, v3 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[40:45], v[174:179], a[192:195], v4, v3 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_add_i32 s0, s19, 0x50
	s_nop 7
	s_nop 0
	v_accvgpr_read_b32 v1, a164
	v_accvgpr_read_b32 v2, a165
	v_cvt_pk_bf16_f32 v8, v1, v2
	v_accvgpr_read_b32 v1, a166
	v_accvgpr_read_b32 v2, a167
	v_cvt_pk_bf16_f32 v9, v1, v2
	v_accvgpr_read_b32 v1, a160
	v_accvgpr_read_b32 v2, a161
	v_cvt_pk_bf16_f32 v6, v1, v2
	v_accvgpr_read_b32 v1, a162
	v_accvgpr_read_b32 v2, a163
	v_cvt_pk_bf16_f32 v7, v1, v2
	s_lshl_b32 s0, s0, 13
	v_permlane16_swap_b32_e32 v6, v8
	v_permlane16_swap_b32_e32 v7, v9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[46:51], v[174:179], a[196:199], v4, v3 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_add_i32 s0, s1, s0
	buffer_store_dwordx4 v[6:9], v0, s[12:15], s0 offen sc1
	v_accvgpr_read_b32 v1, a168
	v_accvgpr_read_b32 v2, a169
	v_cvt_pk_bf16_f32 v6, v1, v2
	v_accvgpr_read_b32 v1, a170
	v_accvgpr_read_b32 v2, a171
	v_cvt_pk_bf16_f32 v7, v1, v2
	v_accvgpr_read_b32 v1, a172
	v_accvgpr_read_b32 v2, a173
	v_cvt_pk_bf16_f32 v8, v1, v2
	v_accvgpr_read_b32 v1, a174
	v_accvgpr_read_b32 v2, a175
	v_cvt_pk_bf16_f32 v9, v1, v2
	v_permlane16_swap_b32_e32 v6, v8
	s_nop 0
	v_permlane16_swap_b32_e32 v7, v9
	buffer_store_dwordx4 v[6:9], v0, s[12:15], s0 offen offset:64 sc1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[150:155], v[174:179], a[200:203], v4, v3 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v1, a176
	v_accvgpr_read_b32 v2, a177
	v_cvt_pk_bf16_f32 v6, v1, v2
	v_accvgpr_read_b32 v1, a178
	v_accvgpr_read_b32 v2, a179
	v_cvt_pk_bf16_f32 v7, v1, v2
	v_accvgpr_read_b32 v1, a180
	v_accvgpr_read_b32 v2, a181
	v_cvt_pk_bf16_f32 v8, v1, v2
	v_accvgpr_read_b32 v1, a182
	v_accvgpr_read_b32 v2, a183
	v_cvt_pk_bf16_f32 v9, v1, v2
	v_permlane16_swap_b32_e32 v6, v8
	s_nop 0
	v_permlane16_swap_b32_e32 v7, v9
	buffer_store_dwordx4 v[6:9], v0, s[12:15], s0 offen offset:128 sc1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[156:161], v[174:179], a[204:207], v4, v3 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v1, a184
	v_accvgpr_read_b32 v2, a185
	v_cvt_pk_bf16_f32 v6, v1, v2
	v_accvgpr_read_b32 v1, a186
	v_accvgpr_read_b32 v2, a187
	v_cvt_pk_bf16_f32 v7, v1, v2
	v_accvgpr_read_b32 v1, a188
	v_accvgpr_read_b32 v2, a189
	v_cvt_pk_bf16_f32 v8, v1, v2
	v_accvgpr_read_b32 v1, a190
	v_accvgpr_read_b32 v2, a191
	v_cvt_pk_bf16_f32 v9, v1, v2
	v_permlane16_swap_b32_e32 v6, v8
	s_nop 0
	v_permlane16_swap_b32_e32 v7, v9
	buffer_store_dwordx4 v[6:9], v0, s[12:15], s0 offen offset:192 sc1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[162:167], v[174:179], a[208:211], v5, v3 op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[168:173], v[174:179], a[212:215], v5, v3 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[98:103], v[174:179], a[216:219], v5, v3 op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[104:109], v[174:179], a[220:223], v5, v3 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[104:109], v[180:185], a[252:255], v5, v3 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_i32 s0, s19, 0x60
	v_accvgpr_read_b32 v1, a192
	v_accvgpr_read_b32 v2, a193
	v_cvt_pk_bf16_f32 v6, v1, v2
	v_accvgpr_read_b32 v1, a194
	v_accvgpr_read_b32 v2, a195
	v_cvt_pk_bf16_f32 v7, v1, v2
	v_accvgpr_read_b32 v1, a196
	v_accvgpr_read_b32 v2, a197
	v_cvt_pk_bf16_f32 v8, v1, v2
	v_accvgpr_read_b32 v1, a198
	v_accvgpr_read_b32 v2, a199
	v_cvt_pk_bf16_f32 v9, v1, v2
	s_lshl_b32 s0, s0, lit(0xd)
	v_permlane16_swap_b32_e32 v6, v8
	v_permlane16_swap_b32_e32 v7, v9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[98:103], v[180:185], a[248:251], v5, v3 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	s_add_i32 s0, s1, s0
	buffer_store_dwordx4 v[6:9], v0, s[12:15], s0 offen sc1
	v_accvgpr_read_b32 v1, a200
	v_accvgpr_read_b32 v2, a201
	v_cvt_pk_bf16_f32 v6, v1, v2
	v_accvgpr_read_b32 v1, a202
	v_accvgpr_read_b32 v2, a203
	v_cvt_pk_bf16_f32 v7, v1, v2
	v_accvgpr_read_b32 v1, a204
	v_accvgpr_read_b32 v2, a205
	v_cvt_pk_bf16_f32 v8, v1, v2
	v_accvgpr_read_b32 v1, a206
	v_accvgpr_read_b32 v2, a207
	v_cvt_pk_bf16_f32 v9, v1, v2
	v_permlane16_swap_b32_e32 v6, v8
	s_nop 0
	v_permlane16_swap_b32_e32 v7, v9
	buffer_store_dwordx4 v[6:9], v0, s[12:15], s0 offen offset:64 sc1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[168:173], v[180:185], a[244:247], v5, v3 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v1, a208
	v_accvgpr_read_b32 v2, a209
	v_cvt_pk_bf16_f32 v6, v1, v2
	v_accvgpr_read_b32 v1, a210
	v_accvgpr_read_b32 v2, a211
	v_cvt_pk_bf16_f32 v7, v1, v2
	v_accvgpr_read_b32 v1, a212
	v_accvgpr_read_b32 v2, a213
	v_cvt_pk_bf16_f32 v8, v1, v2
	v_accvgpr_read_b32 v1, a214
	v_accvgpr_read_b32 v2, a215
	v_cvt_pk_bf16_f32 v9, v1, v2
	v_permlane16_swap_b32_e32 v6, v8
	s_nop 0
	v_permlane16_swap_b32_e32 v7, v9
	buffer_store_dwordx4 v[6:9], v0, s[12:15], s0 offen offset:128 sc1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[162:167], v[180:185], a[240:243], v5, v3 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_accvgpr_read_b32 v1, a216
	v_accvgpr_read_b32 v2, a217
	v_cvt_pk_bf16_f32 v6, v1, v2
	v_accvgpr_read_b32 v1, a218
	v_accvgpr_read_b32 v2, a219
	v_cvt_pk_bf16_f32 v7, v1, v2
	v_accvgpr_read_b32 v1, a220
	v_accvgpr_read_b32 v2, a221
	v_cvt_pk_bf16_f32 v8, v1, v2
	v_accvgpr_read_b32 v1, a222
	v_accvgpr_read_b32 v2, a223
	v_cvt_pk_bf16_f32 v9, v1, v2
	v_permlane16_swap_b32_e32 v6, v8
	s_nop 0
	v_permlane16_swap_b32_e32 v7, v9
	buffer_store_dwordx4 v[6:9], v0, s[12:15], s0 offen offset:192 sc1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[156:161], v[180:185], a[236:239], v4, v3 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[150:155], v[180:185], a[232:235], v4, v3 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[46:51], v[180:185], a[228:231], v4, v3 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[40:45], v[180:185], a[224:227], v4, v3 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:3
	s_add_i32 s0, s19, 0x70
	s_lshl_b32 s0, s0, 13
	s_nop 7
	v_accvgpr_read_b32 v1, a232
	v_accvgpr_read_b32 v2, a233
	v_cvt_pk_bf16_f32 v2, v1, v2
	v_accvgpr_read_b32 v1, a228
	v_accvgpr_read_b32 v3, a229
	v_cvt_pk_bf16_f32 v8, v1, v3
	v_accvgpr_read_b32 v1, a230
	v_accvgpr_read_b32 v3, a231
	v_cvt_pk_bf16_f32 v9, v1, v3
	v_accvgpr_read_b32 v1, a234
	v_accvgpr_read_b32 v3, a235
	v_cvt_pk_bf16_f32 v3, v1, v3
	v_accvgpr_read_b32 v1, a236
	v_accvgpr_read_b32 v4, a237
	v_cvt_pk_bf16_f32 v4, v1, v4
	v_accvgpr_read_b32 v1, a238
	v_accvgpr_read_b32 v5, a239
	v_cvt_pk_bf16_f32 v5, v1, v5
	v_accvgpr_read_b32 v1, a240
	v_accvgpr_read_b32 v6, a241
	v_cvt_pk_bf16_f32 v10, v1, v6
	v_accvgpr_read_b32 v1, a242
	v_accvgpr_read_b32 v6, a243
	v_cvt_pk_bf16_f32 v11, v1, v6
	v_accvgpr_read_b32 v1, a244
	v_accvgpr_read_b32 v6, a245
	v_cvt_pk_bf16_f32 v12, v1, v6
	v_accvgpr_read_b32 v1, a246
	v_accvgpr_read_b32 v6, a247
	v_cvt_pk_bf16_f32 v13, v1, v6
	v_accvgpr_read_b32 v1, a224
	v_accvgpr_read_b32 v6, a225
	v_cvt_pk_bf16_f32 v6, v1, v6
	v_accvgpr_read_b32 v1, a226
	v_accvgpr_read_b32 v7, a227
	v_cvt_pk_bf16_f32 v7, v1, v7
	v_accvgpr_read_b32 v1, a248
	v_accvgpr_read_b32 v14, a249
	v_cvt_pk_bf16_f32 v14, v1, v14
	v_accvgpr_read_b32 v1, a250
	v_accvgpr_read_b32 v15, a251
	v_cvt_pk_bf16_f32 v15, v1, v15
	v_accvgpr_read_b32 v1, a252
	v_accvgpr_read_b32 v16, a253
	v_cvt_pk_bf16_f32 v16, v1, v16
	v_accvgpr_read_b32 v1, a254
	v_accvgpr_read_b32 v17, a255
	v_cvt_pk_bf16_f32 v17, v1, v17
	v_permlane16_swap_b32_e32 v6, v8
	v_permlane16_swap_b32_e32 v7, v9
	s_add_i32 s0, s1, s0
	v_permlane16_swap_b32_e32 v2, v4
	v_permlane16_swap_b32_e32 v3, v5
	v_permlane16_swap_b32_e32 v10, v12
	v_permlane16_swap_b32_e32 v11, v13
	v_permlane16_swap_b32_e32 v14, v16
	v_permlane16_swap_b32_e32 v15, v17
	buffer_store_dwordx4 v[6:9], v0, s[12:15], s0 offen sc1
	buffer_store_dwordx4 v[2:5], v0, s[12:15], s0 offen offset:64 sc1
	buffer_store_dwordx4 v[10:13], v0, s[12:15], s0 offen offset:128 sc1
	buffer_store_dwordx4 v[14:17], v0, s[12:15], s0 offen offset:192 sc1
	s_endpgm
.Lrocke_gemm_mx_e3m2_e3m2_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_4096x4096x65536_c0c1p48_ka941ce3864_end:
.size rocke_gemm_mx_e3m2_e3m2_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_4096x4096x65536_c0c1p48_ka941ce3864, .Lrocke_gemm_mx_e3m2_e3m2_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_4096x4096x65536_c0c1p48_ka941ce3864_end-rocke_gemm_mx_e3m2_e3m2_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_4096x4096x65536_c0c1p48_ka941ce3864
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

.rodata
.p2align 6
.amdhsa_kernel rocke_gemm_mx_e3m2_e3m2_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_4096x4096x65536_c0c1p48_ka941ce3864
  .amdhsa_group_segment_fixed_size 153600
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_kernarg_size 44
  .amdhsa_accum_offset 232
  .amdhsa_tg_split 0
  .amdhsa_next_free_vgpr 488
  .amdhsa_reserve_vcc 0
  .amdhsa_next_free_sgpr 44
  .amdhsa_float_round_mode_32 0
  .amdhsa_float_round_mode_16_64 0
  .amdhsa_float_denorm_mode_32 3
  .amdhsa_float_denorm_mode_16_64 3
  .amdhsa_dx10_clamp 1
  .amdhsa_ieee_mode 1
  .amdhsa_fp16_overflow 0
  .amdhsa_enable_private_segment 0
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_sgpr_workgroup_id_y 0
  .amdhsa_system_sgpr_workgroup_id_z 0
  .amdhsa_system_sgpr_workgroup_info 0
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_exception_fp_ieee_invalid_op 0
  .amdhsa_exception_fp_denorm_src 0
  .amdhsa_exception_fp_ieee_div_zero 0
  .amdhsa_exception_fp_ieee_overflow 0
  .amdhsa_exception_fp_ieee_underflow 0
  .amdhsa_exception_fp_ieee_inexact 0
  .amdhsa_exception_int_div_zero 0
  .amdhsa_user_sgpr_dispatch_ptr 0
  .amdhsa_user_sgpr_queue_ptr 0
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_user_sgpr_dispatch_id 0
  .amdhsa_user_sgpr_private_segment_size 0
.end_amdhsa_kernel

.amdgpu_metadata
---
custom.config:
  Source:
    Origin: rocke
    Description: MX E3M2 GEMM at 4096x4096x65536 on page24-placed operands and scales
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
    SwizzleLayoutA: P48Page24
    SwizzleLayoutB: P48Page24Split
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
            { type: address, semantic: AddressD },
            { type: int32, semantic: AIDPlacementBit } ]
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
  - .name: rocke_gemm_mx_e3m2_e3m2_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_4096x4096x65536_c0c1p48_ka941ce3864
    .symbol: rocke_gemm_mx_e3m2_e3m2_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_4096x4096x65536_c0c1p48_ka941ce3864.kd
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
      - .name:           aidc
        .offset:         40
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 153600
    .kernarg_segment_align: 8
    .kernarg_segment_size: 44
    .max_flat_workgroup_size: 256
    .private_segment_fixed_size: 0
    .sgpr_count: 50
    .vgpr_count: 488
    .agpr_count: 256
    .wavefront_size: 64
...
.end_amdgpu_metadata
