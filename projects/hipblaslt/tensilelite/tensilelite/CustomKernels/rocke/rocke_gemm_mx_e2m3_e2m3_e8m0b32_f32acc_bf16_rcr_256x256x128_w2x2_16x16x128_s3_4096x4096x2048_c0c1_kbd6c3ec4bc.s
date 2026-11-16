// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// MX E2M3 x E2M3 GEMM for gfx950: E8M0 scales per 32 K elements, fp32 accumulation,
// bf16 D rounded to nearest even. 256 threads (wave64, 2x2 waves), macrotile 256x256x128,
// v_mfma_scale_f32_16x16x128_f8f6f4 cbsz:2 blgp:2, 153600 bytes of static LDS.
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
.globl rocke_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_4096x4096x2048_c0c1_kbd6c3ec4bc
.p2align 8
.type rocke_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_4096x4096x2048_c0c1_kbd6c3ec4bc,@function
rocke_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_4096x4096x2048_c0c1_kbd6c3ec4bc:
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
	s_mul_i32 s14, s15, 0x60000
	s_lshl_b32 s15, s15, 8
	s_lshl_b32 s16, s12, 8
	s_lshl_b32 s1, s1, 7
	s_lshl_b32 s13, s13, 7
	s_add_u32 s20, s2, s14
	s_mul_i32 s2, s12, 0x60000
	s_addc_u32 s21, s3, 0
	s_add_u32 s24, s4, s2
	s_addc_u32 s25, s5, 0
	s_mul_i32 s2, s0, 0x1800
	s_lshr_b32 s3, s1, 4
	s_lshr_b32 s4, s13, 4
	s_and_b32 s21, s21, 0xffff
	s_and_b32 s25, s25, 0xffff
	s_add_i32 s5, s15, s1
	s_add_i32 s12, s16, s13
	s_and_b32 s17, s7, 0xffff
	s_and_b32 s29, s9, 0xffff
	s_lshl_b32 s7, s0, 8
	s_lshr_b32 s9, s15, 7
	s_lshr_b32 s14, s16, 7
	v_lshlrev_b32_e32 v1, 4, v0
	s_lshr_b32 s1, s1, 7
	s_lshr_b32 s13, s13, 7
	s_lshl_b32 s15, s9, 9
	s_lshl_b32 s32, s14, 9
	s_add_i32 s33, 0x1800, s2
	s_mov_b32 s22, 0x7ffffff0
	s_mov_b32 s23, 0x31014000
	s_mov_b32 s26, 0x7ffffff0
	s_mov_b32 s27, 0x31014000
	s_mov_b32 s16, s6
	s_mov_b32 s18, 0x7ffffff0
	s_mov_b32 s19, 0x31014000
	s_mov_b32 s28, s8
	s_mov_b32 s30, 0x7ffffff0
	s_mov_b32 s31, 0x31014000
	s_mov_b32 m0, s33
	v_lshlrev_b32_e32 v2, 2, v0
	buffer_load_dwordx4 v1, s[20:23], s2 offen lds
	buffer_load_dwordx4 v1, s[20:23], s2 offen offset:1024 lds
	buffer_load_dwordx4 v1, s[20:23], s2 offen offset:2048 lds
	buffer_load_dwordx4 v1, s[20:23], s2 offen offset:3072 lds
	s_add_i32 s6, s2, 0x1000
	s_add_i32 s8, 0x1800, s6
	s_mov_b32 m0, s8
	s_nop 0
	buffer_load_dwordx4 v1, s[20:23], s6 offen lds
	buffer_load_dwordx4 v1, s[20:23], s6 offen offset:1024 lds
	s_add_i32 s34, 0x7800, s2
	s_mov_b32 m0, s34
	s_nop 0
	buffer_load_dwordx4 v1, s[24:27], s2 offen lds
	buffer_load_dwordx4 v1, s[24:27], s2 offen offset:1024 lds
	buffer_load_dwordx4 v1, s[24:27], s2 offen offset:2048 lds
	buffer_load_dwordx4 v1, s[24:27], s2 offen offset:3072 lds
	s_add_i32 s35, 0x7800, s6
	s_mov_b32 m0, s35
	s_nop 0
	buffer_load_dwordx4 v1, s[24:27], s6 offen lds
	buffer_load_dwordx4 v1, s[24:27], s6 offen offset:1024 lds
	s_mov_b32 m0, s7
	s_add_i32 s15, s7, s15
	buffer_load_dword v2, s[16:19], s15 offen lds
	s_add_i32 s15, 0x400, s7
	s_mov_b32 m0, s15
	s_add_i32 s32, s7, s32
	buffer_load_dword v2, s[28:31], s32 offen lds
	s_add_i32 s15, s9, 32
	s_lshl_b32 s15, s15, 9
	s_add_i32 s36, s2, 0x6000
	s_add_i32 s37, 0xd800, s2
	s_mov_b32 m0, s37
	s_nop 0
	buffer_load_dwordx4 v1, s[20:23], s36 offen lds
	buffer_load_dwordx4 v1, s[20:23], s36 offen offset:1024 lds
	buffer_load_dwordx4 v1, s[20:23], s36 offen offset:2048 lds
	buffer_load_dwordx4 v1, s[20:23], s36 offen offset:3072 lds
	s_add_i32 s38, 0xd800, s6
	s_mov_b32 m0, s38
	s_add_i32 s39, s2, 0x7000
	buffer_load_dwordx4 v1, s[20:23], s39 offen lds
	buffer_load_dwordx4 v1, s[20:23], s39 offen offset:1024 lds
	s_add_i32 s40, 0x13800, s2
	s_mov_b32 m0, s40
	s_nop 0
	buffer_load_dwordx4 v1, s[24:27], s36 offen lds
	buffer_load_dwordx4 v1, s[24:27], s36 offen offset:1024 lds
	buffer_load_dwordx4 v1, s[24:27], s36 offen offset:2048 lds
	buffer_load_dwordx4 v1, s[24:27], s36 offen offset:3072 lds
	s_add_i32 s36, 0x13800, s6
	s_mov_b32 m0, s36
	s_nop 0
	buffer_load_dwordx4 v1, s[24:27], s39 offen lds
	buffer_load_dwordx4 v1, s[24:27], s39 offen offset:1024 lds
	s_add_i32 s39, 0x800, s7
	s_mov_b32 m0, s39
	s_add_i32 s15, s7, s15
	buffer_load_dword v2, s[16:19], s15 offen lds
	s_add_i32 s15, s32, 0x3c00
	buffer_load_dword v2, s[28:31], s15 offen offset:1024 lds
	s_add_i32 s15, s9, 64
	s_lshl_b32 s15, s15, 9
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_barrier
	s_lshl_b32 s1, s1, 7
	s_lshl_b32 s13, s13, 7
	s_lshl_b32 s3, s3, 6
	v_lshlrev_b32_e32 v3, 1, v0
	v_add_u32_e32 v4, s3, v0
	v_add_lshl_u32 v5, s1, v3, 2
	v_add_lshl_u32 v3, s13, v3, 2
	v_lshlrev_b32_e32 v6, 4, v4
	ds_read_b64 v[8:9], v5
	ds_read_b64 v[10:11], v3 offset:1024
	ds_read_b128 v[12:15], v6 offset:6144
	v_lshlrev_b32_e32 v4, 3, v4
	ds_read_b64 v[16:17], v4 offset:22528
	s_lshl_b32 s1, s4, 6
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
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[12:17], v[20:25], 0, v8, v10 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[44:47], v18 offset:34816
	ds_read_b64 v[48:49], v7 offset:49152
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[12:17], v[26:31], 0, v8, v10 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[50:53], v18 offset:35840
	ds_read_b64 v[54:55], v7 offset:49664
	s_add_i32 s1, s2, 0xc000
	s_add_i32 s3, 0x19800, s2
	s_mov_b32 m0, s3
	s_nop 0
	buffer_load_dwordx4 v1, s[20:23], s1 offen lds
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[12:17], v[32:37], 0, v8, v10 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[56:59], v18 offset:36864
	ds_read_b64 v[60:61], v7 offset:50176
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[12:17], v[38:43], 0, v8, v10 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[62:65], v18 offset:37888
	ds_read_b64 v[66:67], v7 offset:50688
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[12:17], v[44:49], 0, v8, v11 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[68:71], v6 offset:7168
	ds_read_b64 v[72:73], v4 offset:23040
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[12:17], v[50:55], 0, v8, v11 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s1 offen offset:1024 lds
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[12:17], v[56:61], 0, v8, v11 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[12:17], v[62:67], 0, v8, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[68:73], v[62:67], 0, v8, v11 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[68:73], v[56:61], 0, v8, v11 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[68:73], v[50:55], 0, v8, v11 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s1 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[68:73], v[44:49], 0, v8, v11 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[12:15], v6 offset:8192
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[68:73], v[38:43], 0, v8, v10 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[16:17], v4 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[68:73], v[32:37], 0, v8, v10 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[68:73], v[26:31], 0, v8, v10 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s1 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[68:73], v[20:25], 0, v8, v10 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[12:17], v[20:25], 0, v8, v10 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[12:17], v[26:31], 0, v8, v10 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[12:17], v[32:37], 0, v8, v10 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s4, 0x19800, s6
	s_add_i32 s13, s2, 0xd000
	s_mov_b32 m0, s4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[12:17], v[38:43], 0, v8, v10 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s13 offen lds
	ds_read_b128 v[68:71], v6 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[12:17], v[44:49], 0, v8, v11 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[72:73], v4 offset:24064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[12:17], v[50:55], 0, v8, v11 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[12:17], v[56:61], 0, v8, v11 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[12:17], v[62:67], 0, v8, v11 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s13 offen offset:1024 lds
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[68:73], v[62:67], 0, v8, v11 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[68:73], v[56:61], 0, v8, v11 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[68:73], v[50:55], 0, v8, v11 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[68:73], v[44:49], 0, v8, v11 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[12:15], v6 offset:10240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[68:73], v[38:43], 0, v8, v10 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[16:17], v4 offset:24576
	s_add_i32 s41, 0x1f800, s2
	s_mov_b32 m0, s41
	s_nop 0
	buffer_load_dwordx4 v1, s[24:27], s1 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[68:73], v[32:37], 0, v8, v10 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[68:73], v[26:31], 0, v8, v10 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[68:73], v[20:25], 0, v8, v10 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[12:17], v[20:25], 0, v9, v10 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s1 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[12:17], v[26:31], 0, v9, v10 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[12:17], v[32:37], 0, v9, v10 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[12:17], v[38:43], 0, v9, v10 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[68:71], v6 offset:11264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[12:17], v[44:49], 0, v9, v11 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[72:73], v4 offset:25088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[12:17], v[50:55], 0, v9, v11 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s1 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[12:17], v[56:61], 0, v9, v11 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[12:17], v[62:67], 0, v9, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[68:73], v[62:67], 0, v9, v11 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[68:73], v[56:61], 0, v9, v11 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s1 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[68:73], v[50:55], 0, v9, v11 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[68:73], v[44:49], 0, v9, v11 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[12:15], v6 offset:12288
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[68:73], v[38:43], 0, v9, v10 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[16:17], v4 offset:25600
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[68:73], v[32:37], 0, v9, v10 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_i32 s1, 0x1f800, s6
	s_mov_b32 m0, s1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[68:73], v[26:31], 0, v9, v10 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s13 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[68:73], v[20:25], 0, v9, v10 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[12:17], v[20:25], 0, v9, v10 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[12:17], v[26:31], 0, v9, v10 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[12:17], v[32:37], 0, v9, v10 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s13 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[12:17], v[38:43], 0, v9, v10 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[68:71], v6 offset:13312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[12:17], v[44:49], 0, v9, v11 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[72:73], v4 offset:26112
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[12:17], v[50:55], 0, v9, v11 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[12:17], v[56:61], 0, v9, v11 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s6, 0x1000, s7
	s_add_i32 s13, s7, s15
	s_mov_b32 m0, s6
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[12:17], v[62:67], 0, v9, v11 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[16:19], s13 offen lds
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[68:73], v[62:67], 0, v9, v11 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[68:73], v[56:61], 0, v9, v11 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[68:73], v[50:55], 0, v9, v11 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s13, s32, 0x7c00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[68:73], v[44:49], 0, v9, v11 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[28:31], s13 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[68:73], v[38:43], 0, v9, v10 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[68:73], v[32:37], 0, v9, v10 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[68:73], v[26:31], 0, v9, v10 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[68:73], v[20:25], 0, v9, v10 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s13, s9, 9
	s_add_i32 s13, s7, s13
	s_mov_b32 s15, 0
	v_add_u32_e32 v8, 0x10000, v4
	v_add_u32_e32 v9, 0x10000, v18
	v_add_u32_e32 v10, 0x10000, v7
	v_add_u32_e32 v11, 0x10000, v6
	v_add_u32_e32 v12, 0x20000, v7
	v_add_u32_e32 v13, 0x20000, v18
.Lmain_loop:
	s_mul_i32 s42, s15, 12
	s_lshl_b32 s43, s42, 5
	s_add_i32 s43, s14, s43
	s_mul_i32 s42, s42, 0x6000
	s_lshl_b32 s43, s43, 9
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[14:15], v5 offset:2048
	ds_read_b64 v[16:17], v3 offset:3072
	ds_read_b128 v[20:23], v6 offset:55296
	ds_read_b64 v[24:25], v8 offset:6144
	ds_read_b128 v[26:29], v9 offset:14336
	ds_read_b64 v[30:31], v10 offset:30720
	ds_read_b128 v[32:35], v9 offset:15360
	ds_read_b64 v[36:37], v10 offset:31232
	ds_read_b128 v[38:41], v9 offset:16384
	ds_read_b64 v[42:43], v10 offset:31744
	ds_read_b128 v[44:47], v9 offset:17408
	ds_read_b64 v[48:49], v10 offset:32256
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:25], v[26:31], a[0:3], v14, v16 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[50:53], v9 offset:18432
	ds_read_b64 v[54:55], v10 offset:32768
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[20:25], v[32:37], a[4:7], v14, v16 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[56:59], v9 offset:19456
	ds_read_b64 v[60:61], v10 offset:33280
	s_add_i32 s42, s2, s42
	s_mov_b32 m0, s33
	s_add_i32 s44, s42, 0x12000
	buffer_load_dwordx4 v1, s[20:23], s44 offen lds
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[20:25], v[38:43], a[8:11], v14, v16 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[62:65], v9 offset:20480
	ds_read_b64 v[66:67], v10 offset:33792
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[20:25], v[44:49], a[12:15], v14, v16 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[68:71], v9 offset:21504
	ds_read_b64 v[72:73], v10 offset:34304
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[20:25], v[50:55], a[16:19], v14, v17 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[74:77], v6 offset:56320
	ds_read_b64 v[78:79], v8 offset:6656
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[20:25], v[56:61], a[20:23], v14, v17 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s44 offen offset:1024 lds
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[20:25], v[62:67], a[24:27], v14, v17 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[20:25], v[68:73], a[28:31], v14, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[74:79], v[68:73], a[32:35], v14, v17 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[74:79], v[62:67], a[36:39], v14, v17 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[74:79], v[56:61], a[40:43], v14, v17 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s44 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[74:79], v[50:55], a[44:47], v14, v17 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[20:23], v6 offset:57344
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[74:79], v[44:49], a[48:51], v14, v16 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[24:25], v8 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[74:79], v[38:43], a[52:55], v14, v16 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[74:79], v[32:37], a[56:59], v14, v16 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s44 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[74:79], v[26:31], a[60:63], v14, v16 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[20:25], v[26:31], a[64:67], v14, v16 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[20:25], v[32:37], a[68:71], v14, v16 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[20:25], v[38:43], a[72:75], v14, v16 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s45, s42, 0x13000
	s_mov_b32 m0, s8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[20:25], v[44:49], a[76:79], v14, v16 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s45 offen lds
	ds_read_b128 v[74:77], v6 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[20:25], v[50:55], a[80:83], v14, v17 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v8 offset:7680
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[20:25], v[56:61], a[84:87], v14, v17 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[20:25], v[62:67], a[88:91], v14, v17 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[20:25], v[68:73], a[92:95], v14, v17 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s45 offen offset:1024 lds
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[74:79], v[68:73], a[96:99], v14, v17 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[74:79], v[62:67], a[100:103], v14, v17 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[74:79], v[56:61], a[104:107], v14, v17 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[74:79], v[50:55], a[108:111], v14, v17 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[20:23], v6 offset:59392
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[74:79], v[44:49], a[112:115], v14, v16 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s34
	ds_read_b64 v[24:25], v8 offset:8192
	buffer_load_dwordx4 v1, s[24:27], s44 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[74:79], v[38:43], a[116:119], v14, v16 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[74:79], v[32:37], a[120:123], v14, v16 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[74:79], v[26:31], a[124:127], v14, v16 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[20:25], v[26:31], a[128:131], v15, v16 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s44 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[20:25], v[32:37], a[132:135], v15, v16 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[20:25], v[38:43], a[136:139], v15, v16 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[20:25], v[44:49], a[140:143], v15, v16 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[74:77], v6 offset:60416
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[20:25], v[50:55], a[144:147], v15, v17 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v8 offset:8704
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[20:25], v[56:61], a[148:151], v15, v17 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s44 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[20:25], v[62:67], a[152:155], v15, v17 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[20:25], v[68:73], a[156:159], v15, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[74:79], v[68:73], a[160:163], v15, v17 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[74:79], v[62:67], a[164:167], v15, v17 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s44 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[74:79], v[56:61], a[168:171], v15, v17 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[74:79], v[50:55], a[172:175], v15, v17 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[20:23], v6 offset:61440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[74:79], v[44:49], a[176:179], v15, v16 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[24:25], v8 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[74:79], v[38:43], a[180:183], v15, v16 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s35
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[74:79], v[32:37], a[184:187], v15, v16 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s45 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[74:79], v[26:31], a[188:191], v15, v16 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[20:25], v[26:31], a[192:195], v15, v16 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[20:25], v[32:37], a[196:199], v15, v16 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[20:25], v[38:43], a[200:203], v15, v16 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s45 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[20:25], v[44:49], a[204:207], v15, v16 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[74:77], v6 offset:62464
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[20:25], v[50:55], a[208:211], v15, v17 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v8 offset:9728
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[20:25], v[56:61], a[212:215], v15, v17 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[20:25], v[62:67], a[216:219], v15, v17 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_mul_i32 s44, s15, 0x30000
	s_add_i32 s44, s13, s44
	s_add_i32 s45, s44, 0xc000
	s_mov_b32 m0, s7
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[20:25], v[68:73], a[220:223], v15, v17 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[16:19], s45 offen lds
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[74:79], v[68:73], a[224:227], v15, v17 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[74:79], v[62:67], a[228:231], v15, v17 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[74:79], v[56:61], a[232:235], v15, v17 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s43, s7, s43
	s_add_i32 s45, s43, 0xbc00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[74:79], v[50:55], a[236:239], v15, v17 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[28:31], s45 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[74:79], v[44:49], a[240:243], v15, v16 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[74:79], v[38:43], a[244:247], v15, v16 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[74:79], v[32:37], a[248:251], v15, v16 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[74:79], v[26:31], a[252:255], v15, v16 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[14:15], v5 offset:4096
	ds_read_b64 v[16:17], v3 offset:5120
	ds_read_b128 v[20:23], v11 offset:38912
	ds_read_b64 v[24:25], v8 offset:55296
	ds_read_b128 v[26:29], v9 offset:63488
	ds_read_b64 v[30:31], v12 offset:14336
	ds_read_b128 v[32:35], v9 offset:64512
	ds_read_b64 v[36:37], v12 offset:14848
	ds_read_b128 v[38:41], v13
	ds_read_b64 v[42:43], v12 offset:15360
	ds_read_b128 v[44:47], v13 offset:1024
	ds_read_b64 v[48:49], v12 offset:15872
	ds_read_b128 v[50:53], v13 offset:2048
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:25], v[26:31], a[0:3], v14, v16 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[54:55], v12 offset:16384
	ds_read_b128 v[56:59], v13 offset:3072
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[20:25], v[32:37], a[4:7], v14, v16 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[60:61], v12 offset:16896
	s_mov_b32 m0, s37
	s_add_i32 s45, s42, 0x18000
	buffer_load_dwordx4 v1, s[20:23], s45 offen lds
	ds_read_b128 v[62:65], v13 offset:4096
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[20:25], v[38:43], a[8:11], v14, v16 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[66:67], v12 offset:17408
	ds_read_b128 v[68:71], v13 offset:5120
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[20:25], v[44:49], a[12:15], v14, v16 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[72:73], v12 offset:17920
	ds_read_b128 v[74:77], v11 offset:39936
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[20:25], v[50:55], a[16:19], v14, v17 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v8 offset:55808
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[20:25], v[56:61], a[20:23], v14, v17 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s45 offen offset:1024 lds
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[20:25], v[62:67], a[24:27], v14, v17 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[20:25], v[68:73], a[28:31], v14, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[74:79], v[68:73], a[32:35], v14, v17 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[74:79], v[62:67], a[36:39], v14, v17 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[74:79], v[56:61], a[40:43], v14, v17 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s45 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[74:79], v[50:55], a[44:47], v14, v17 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[20:23], v11 offset:40960
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[74:79], v[44:49], a[48:51], v14, v16 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[24:25], v8 offset:56320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[74:79], v[38:43], a[52:55], v14, v16 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[74:79], v[32:37], a[56:59], v14, v16 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s45 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[74:79], v[26:31], a[60:63], v14, v16 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[20:25], v[26:31], a[64:67], v14, v16 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[20:25], v[32:37], a[68:71], v14, v16 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[20:25], v[38:43], a[72:75], v14, v16 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s46, s42, 0x19000
	s_mov_b32 m0, s38
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[20:25], v[44:49], a[76:79], v14, v16 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s46 offen lds
	ds_read_b128 v[74:77], v11 offset:41984
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[20:25], v[50:55], a[80:83], v14, v17 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v8 offset:56832
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[20:25], v[56:61], a[84:87], v14, v17 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[20:25], v[62:67], a[88:91], v14, v17 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[20:25], v[68:73], a[92:95], v14, v17 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s46 offen offset:1024 lds
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[74:79], v[68:73], a[96:99], v14, v17 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[74:79], v[62:67], a[100:103], v14, v17 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[74:79], v[56:61], a[104:107], v14, v17 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[74:79], v[50:55], a[108:111], v14, v17 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[20:23], v11 offset:43008
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[74:79], v[44:49], a[112:115], v14, v16 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s40
	ds_read_b64 v[24:25], v8 offset:57344
	buffer_load_dwordx4 v1, s[24:27], s45 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[74:79], v[38:43], a[116:119], v14, v16 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[74:79], v[32:37], a[120:123], v14, v16 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[74:79], v[26:31], a[124:127], v14, v16 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[20:25], v[26:31], a[128:131], v15, v16 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s45 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[20:25], v[32:37], a[132:135], v15, v16 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[20:25], v[38:43], a[136:139], v15, v16 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[20:25], v[44:49], a[140:143], v15, v16 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[74:77], v11 offset:44032
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[20:25], v[50:55], a[144:147], v15, v17 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v8 offset:57856
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[20:25], v[56:61], a[148:151], v15, v17 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s45 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[20:25], v[62:67], a[152:155], v15, v17 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[20:25], v[68:73], a[156:159], v15, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[74:79], v[68:73], a[160:163], v15, v17 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[74:79], v[62:67], a[164:167], v15, v17 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s45 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[74:79], v[56:61], a[168:171], v15, v17 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[74:79], v[50:55], a[172:175], v15, v17 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[20:23], v11 offset:45056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[74:79], v[44:49], a[176:179], v15, v16 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[24:25], v8 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[74:79], v[38:43], a[180:183], v15, v16 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s36
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[74:79], v[32:37], a[184:187], v15, v16 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s46 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[74:79], v[26:31], a[188:191], v15, v16 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[20:25], v[26:31], a[192:195], v15, v16 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[20:25], v[32:37], a[196:199], v15, v16 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[20:25], v[38:43], a[200:203], v15, v16 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s46 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[20:25], v[44:49], a[204:207], v15, v16 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[74:77], v11 offset:46080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[20:25], v[50:55], a[208:211], v15, v17 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v8 offset:58880
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[20:25], v[56:61], a[212:215], v15, v17 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[20:25], v[62:67], a[216:219], v15, v17 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s45, s44, 0x10000
	s_mov_b32 m0, s39
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[20:25], v[68:73], a[220:223], v15, v17 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[16:19], s45 offen lds
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[74:79], v[68:73], a[224:227], v15, v17 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[74:79], v[62:67], a[228:231], v15, v17 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[74:79], v[56:61], a[232:235], v15, v17 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s45, s43, 0xfc00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[74:79], v[50:55], a[236:239], v15, v17 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[28:31], s45 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[74:79], v[44:49], a[240:243], v15, v16 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[74:79], v[38:43], a[244:247], v15, v16 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[74:79], v[32:37], a[248:251], v15, v16 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[74:79], v[26:31], a[252:255], v15, v16 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[14:15], v5
	ds_read_b64 v[16:17], v3 offset:1024
	ds_read_b128 v[20:23], v6 offset:6144
	ds_read_b64 v[24:25], v4 offset:22528
	ds_read_b128 v[26:29], v18 offset:30720
	ds_read_b64 v[30:31], v7 offset:47104
	ds_read_b128 v[32:35], v18 offset:31744
	ds_read_b64 v[36:37], v7 offset:47616
	ds_read_b128 v[38:41], v18 offset:32768
	ds_read_b64 v[42:43], v7 offset:48128
	ds_read_b128 v[44:47], v18 offset:33792
	ds_read_b64 v[48:49], v7 offset:48640
	ds_read_b128 v[50:53], v18 offset:34816
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:25], v[26:31], a[0:3], v14, v16 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[54:55], v7 offset:49152
	ds_read_b128 v[56:59], v18 offset:35840
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[20:25], v[32:37], a[4:7], v14, v16 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[60:61], v7 offset:49664
	s_mov_b32 m0, s3
	s_add_i32 s45, s42, 0x1e000
	buffer_load_dwordx4 v1, s[20:23], s45 offen lds
	ds_read_b128 v[62:65], v18 offset:36864
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[20:25], v[38:43], a[8:11], v14, v16 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[66:67], v7 offset:50176
	ds_read_b128 v[68:71], v18 offset:37888
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[20:25], v[44:49], a[12:15], v14, v16 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[72:73], v7 offset:50688
	ds_read_b128 v[74:77], v6 offset:7168
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[20:25], v[50:55], a[16:19], v14, v17 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v4 offset:23040
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[20:25], v[56:61], a[20:23], v14, v17 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s45 offen offset:1024 lds
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[20:25], v[62:67], a[24:27], v14, v17 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[20:25], v[68:73], a[28:31], v14, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[74:79], v[68:73], a[32:35], v14, v17 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[74:79], v[62:67], a[36:39], v14, v17 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[74:79], v[56:61], a[40:43], v14, v17 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s45 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[74:79], v[50:55], a[44:47], v14, v17 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[20:23], v6 offset:8192
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[74:79], v[44:49], a[48:51], v14, v16 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[24:25], v4 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[74:79], v[38:43], a[52:55], v14, v16 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[74:79], v[32:37], a[56:59], v14, v16 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s45 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[74:79], v[26:31], a[60:63], v14, v16 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[20:25], v[26:31], a[64:67], v14, v16 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[20:25], v[32:37], a[68:71], v14, v16 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[20:25], v[38:43], a[72:75], v14, v16 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s46, s42, 0x1f000
	s_mov_b32 m0, s4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[20:25], v[44:49], a[76:79], v14, v16 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s46 offen lds
	ds_read_b128 v[74:77], v6 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[20:25], v[50:55], a[80:83], v14, v17 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v4 offset:24064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[20:25], v[56:61], a[84:87], v14, v17 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[20:25], v[62:67], a[88:91], v14, v17 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[20:25], v[68:73], a[92:95], v14, v17 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s46 offen offset:1024 lds
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[74:79], v[68:73], a[96:99], v14, v17 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[74:79], v[62:67], a[100:103], v14, v17 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[74:79], v[56:61], a[104:107], v14, v17 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[74:79], v[50:55], a[108:111], v14, v17 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[20:23], v6 offset:10240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[74:79], v[44:49], a[112:115], v14, v16 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s41
	ds_read_b64 v[24:25], v4 offset:24576
	buffer_load_dwordx4 v1, s[24:27], s45 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[74:79], v[38:43], a[116:119], v14, v16 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[74:79], v[32:37], a[120:123], v14, v16 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[74:79], v[26:31], a[124:127], v14, v16 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[20:25], v[26:31], a[128:131], v15, v16 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s45 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[20:25], v[32:37], a[132:135], v15, v16 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[20:25], v[38:43], a[136:139], v15, v16 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[20:25], v[44:49], a[140:143], v15, v16 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[74:77], v6 offset:11264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[20:25], v[50:55], a[144:147], v15, v17 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v4 offset:25088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[20:25], v[56:61], a[148:151], v15, v17 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s45 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[20:25], v[62:67], a[152:155], v15, v17 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[20:25], v[68:73], a[156:159], v15, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[74:79], v[68:73], a[160:163], v15, v17 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[74:79], v[62:67], a[164:167], v15, v17 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s45 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[74:79], v[56:61], a[168:171], v15, v17 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[74:79], v[50:55], a[172:175], v15, v17 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[20:23], v6 offset:12288
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[74:79], v[44:49], a[176:179], v15, v16 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[24:25], v4 offset:25600
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[74:79], v[38:43], a[180:183], v15, v16 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[74:79], v[32:37], a[184:187], v15, v16 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s46 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[74:79], v[26:31], a[188:191], v15, v16 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[20:25], v[26:31], a[192:195], v15, v16 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[20:25], v[32:37], a[196:199], v15, v16 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[20:25], v[38:43], a[200:203], v15, v16 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s46 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[20:25], v[44:49], a[204:207], v15, v16 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[74:77], v6 offset:13312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[20:25], v[50:55], a[208:211], v15, v17 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v4 offset:26112
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[20:25], v[56:61], a[212:215], v15, v17 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[20:25], v[62:67], a[216:219], v15, v17 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s45, s44, 0x14000
	s_mov_b32 m0, s6
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[20:25], v[68:73], a[220:223], v15, v17 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[16:19], s45 offen lds
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[74:79], v[68:73], a[224:227], v15, v17 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[74:79], v[62:67], a[228:231], v15, v17 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[74:79], v[56:61], a[232:235], v15, v17 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s45, s43, 0x13c00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[74:79], v[50:55], a[236:239], v15, v17 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[28:31], s45 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[74:79], v[44:49], a[240:243], v15, v16 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[74:79], v[38:43], a[244:247], v15, v16 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[74:79], v[32:37], a[248:251], v15, v16 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[74:79], v[26:31], a[252:255], v15, v16 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[14:15], v5 offset:2048
	ds_read_b64 v[16:17], v3 offset:3072
	ds_read_b128 v[20:23], v6 offset:55296
	ds_read_b64 v[24:25], v8 offset:6144
	ds_read_b128 v[26:29], v9 offset:14336
	ds_read_b64 v[30:31], v10 offset:30720
	ds_read_b128 v[32:35], v9 offset:15360
	ds_read_b64 v[36:37], v10 offset:31232
	ds_read_b128 v[38:41], v9 offset:16384
	ds_read_b64 v[42:43], v10 offset:31744
	ds_read_b128 v[44:47], v9 offset:17408
	ds_read_b64 v[48:49], v10 offset:32256
	ds_read_b128 v[50:53], v9 offset:18432
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:25], v[26:31], a[0:3], v14, v16 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[54:55], v10 offset:32768
	ds_read_b128 v[56:59], v9 offset:19456
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[20:25], v[32:37], a[4:7], v14, v16 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[60:61], v10 offset:33280
	s_mov_b32 m0, s33
	s_add_i32 s45, s42, 0x24000
	buffer_load_dwordx4 v1, s[20:23], s45 offen lds
	ds_read_b128 v[62:65], v9 offset:20480
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[20:25], v[38:43], a[8:11], v14, v16 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[66:67], v10 offset:33792
	ds_read_b128 v[68:71], v9 offset:21504
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[20:25], v[44:49], a[12:15], v14, v16 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[72:73], v10 offset:34304
	ds_read_b128 v[74:77], v6 offset:56320
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[20:25], v[50:55], a[16:19], v14, v17 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v8 offset:6656
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[20:25], v[56:61], a[20:23], v14, v17 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s45 offen offset:1024 lds
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[20:25], v[62:67], a[24:27], v14, v17 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[20:25], v[68:73], a[28:31], v14, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[74:79], v[68:73], a[32:35], v14, v17 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[74:79], v[62:67], a[36:39], v14, v17 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[74:79], v[56:61], a[40:43], v14, v17 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s45 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[74:79], v[50:55], a[44:47], v14, v17 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[20:23], v6 offset:57344
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[74:79], v[44:49], a[48:51], v14, v16 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[24:25], v8 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[74:79], v[38:43], a[52:55], v14, v16 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[74:79], v[32:37], a[56:59], v14, v16 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s45 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[74:79], v[26:31], a[60:63], v14, v16 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[20:25], v[26:31], a[64:67], v14, v16 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[20:25], v[32:37], a[68:71], v14, v16 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[20:25], v[38:43], a[72:75], v14, v16 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s46, s42, 0x25000
	s_mov_b32 m0, s8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[20:25], v[44:49], a[76:79], v14, v16 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s46 offen lds
	ds_read_b128 v[74:77], v6 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[20:25], v[50:55], a[80:83], v14, v17 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v8 offset:7680
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[20:25], v[56:61], a[84:87], v14, v17 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[20:25], v[62:67], a[88:91], v14, v17 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[20:25], v[68:73], a[92:95], v14, v17 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s46 offen offset:1024 lds
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[74:79], v[68:73], a[96:99], v14, v17 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[74:79], v[62:67], a[100:103], v14, v17 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[74:79], v[56:61], a[104:107], v14, v17 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[74:79], v[50:55], a[108:111], v14, v17 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[20:23], v6 offset:59392
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[74:79], v[44:49], a[112:115], v14, v16 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s34
	ds_read_b64 v[24:25], v8 offset:8192
	buffer_load_dwordx4 v1, s[24:27], s45 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[74:79], v[38:43], a[116:119], v14, v16 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[74:79], v[32:37], a[120:123], v14, v16 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[74:79], v[26:31], a[124:127], v14, v16 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[20:25], v[26:31], a[128:131], v15, v16 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s45 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[20:25], v[32:37], a[132:135], v15, v16 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[20:25], v[38:43], a[136:139], v15, v16 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[20:25], v[44:49], a[140:143], v15, v16 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[74:77], v6 offset:60416
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[20:25], v[50:55], a[144:147], v15, v17 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v8 offset:8704
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[20:25], v[56:61], a[148:151], v15, v17 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s45 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[20:25], v[62:67], a[152:155], v15, v17 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[20:25], v[68:73], a[156:159], v15, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[74:79], v[68:73], a[160:163], v15, v17 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[74:79], v[62:67], a[164:167], v15, v17 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s45 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[74:79], v[56:61], a[168:171], v15, v17 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[74:79], v[50:55], a[172:175], v15, v17 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[20:23], v6 offset:61440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[74:79], v[44:49], a[176:179], v15, v16 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[24:25], v8 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[74:79], v[38:43], a[180:183], v15, v16 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s35
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[74:79], v[32:37], a[184:187], v15, v16 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s46 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[74:79], v[26:31], a[188:191], v15, v16 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[20:25], v[26:31], a[192:195], v15, v16 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[20:25], v[32:37], a[196:199], v15, v16 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[20:25], v[38:43], a[200:203], v15, v16 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s46 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[20:25], v[44:49], a[204:207], v15, v16 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[74:77], v6 offset:62464
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[20:25], v[50:55], a[208:211], v15, v17 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v8 offset:9728
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[20:25], v[56:61], a[212:215], v15, v17 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[20:25], v[62:67], a[216:219], v15, v17 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s45, s44, 0x18000
	s_mov_b32 m0, s7
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[20:25], v[68:73], a[220:223], v15, v17 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[16:19], s45 offen lds
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[74:79], v[68:73], a[224:227], v15, v17 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[74:79], v[62:67], a[228:231], v15, v17 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[74:79], v[56:61], a[232:235], v15, v17 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s45, s43, 0x17c00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[74:79], v[50:55], a[236:239], v15, v17 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[28:31], s45 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[74:79], v[44:49], a[240:243], v15, v16 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[74:79], v[38:43], a[244:247], v15, v16 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[74:79], v[32:37], a[248:251], v15, v16 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[74:79], v[26:31], a[252:255], v15, v16 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[14:15], v5 offset:4096
	ds_read_b64 v[16:17], v3 offset:5120
	ds_read_b128 v[20:23], v11 offset:38912
	ds_read_b64 v[24:25], v8 offset:55296
	ds_read_b128 v[26:29], v9 offset:63488
	ds_read_b64 v[30:31], v12 offset:14336
	ds_read_b128 v[32:35], v9 offset:64512
	ds_read_b64 v[36:37], v12 offset:14848
	ds_read_b128 v[38:41], v13
	ds_read_b64 v[42:43], v12 offset:15360
	ds_read_b128 v[44:47], v13 offset:1024
	ds_read_b64 v[48:49], v12 offset:15872
	ds_read_b128 v[50:53], v13 offset:2048
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:25], v[26:31], a[0:3], v14, v16 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[54:55], v12 offset:16384
	ds_read_b128 v[56:59], v13 offset:3072
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[20:25], v[32:37], a[4:7], v14, v16 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[60:61], v12 offset:16896
	s_mov_b32 m0, s37
	s_add_i32 s45, s42, 0x2a000
	buffer_load_dwordx4 v1, s[20:23], s45 offen lds
	ds_read_b128 v[62:65], v13 offset:4096
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[20:25], v[38:43], a[8:11], v14, v16 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[66:67], v12 offset:17408
	ds_read_b128 v[68:71], v13 offset:5120
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[20:25], v[44:49], a[12:15], v14, v16 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[72:73], v12 offset:17920
	ds_read_b128 v[74:77], v11 offset:39936
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[20:25], v[50:55], a[16:19], v14, v17 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v8 offset:55808
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[20:25], v[56:61], a[20:23], v14, v17 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s45 offen offset:1024 lds
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[20:25], v[62:67], a[24:27], v14, v17 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[20:25], v[68:73], a[28:31], v14, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[74:79], v[68:73], a[32:35], v14, v17 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[74:79], v[62:67], a[36:39], v14, v17 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[74:79], v[56:61], a[40:43], v14, v17 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s45 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[74:79], v[50:55], a[44:47], v14, v17 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[20:23], v11 offset:40960
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[74:79], v[44:49], a[48:51], v14, v16 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[24:25], v8 offset:56320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[74:79], v[38:43], a[52:55], v14, v16 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[74:79], v[32:37], a[56:59], v14, v16 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s45 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[74:79], v[26:31], a[60:63], v14, v16 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[20:25], v[26:31], a[64:67], v14, v16 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[20:25], v[32:37], a[68:71], v14, v16 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[20:25], v[38:43], a[72:75], v14, v16 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s46, s42, 0x2b000
	s_mov_b32 m0, s38
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[20:25], v[44:49], a[76:79], v14, v16 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s46 offen lds
	ds_read_b128 v[74:77], v11 offset:41984
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[20:25], v[50:55], a[80:83], v14, v17 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v8 offset:56832
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[20:25], v[56:61], a[84:87], v14, v17 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[20:25], v[62:67], a[88:91], v14, v17 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[20:25], v[68:73], a[92:95], v14, v17 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s46 offen offset:1024 lds
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[74:79], v[68:73], a[96:99], v14, v17 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[74:79], v[62:67], a[100:103], v14, v17 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[74:79], v[56:61], a[104:107], v14, v17 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[74:79], v[50:55], a[108:111], v14, v17 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[20:23], v11 offset:43008
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[74:79], v[44:49], a[112:115], v14, v16 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s40
	ds_read_b64 v[24:25], v8 offset:57344
	buffer_load_dwordx4 v1, s[24:27], s45 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[74:79], v[38:43], a[116:119], v14, v16 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[74:79], v[32:37], a[120:123], v14, v16 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[74:79], v[26:31], a[124:127], v14, v16 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[20:25], v[26:31], a[128:131], v15, v16 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s45 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[20:25], v[32:37], a[132:135], v15, v16 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[20:25], v[38:43], a[136:139], v15, v16 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[20:25], v[44:49], a[140:143], v15, v16 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[74:77], v11 offset:44032
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[20:25], v[50:55], a[144:147], v15, v17 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v8 offset:57856
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[20:25], v[56:61], a[148:151], v15, v17 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s45 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[20:25], v[62:67], a[152:155], v15, v17 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[20:25], v[68:73], a[156:159], v15, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[74:79], v[68:73], a[160:163], v15, v17 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[74:79], v[62:67], a[164:167], v15, v17 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s45 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[74:79], v[56:61], a[168:171], v15, v17 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[74:79], v[50:55], a[172:175], v15, v17 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[20:23], v11 offset:45056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[74:79], v[44:49], a[176:179], v15, v16 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[24:25], v8 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[74:79], v[38:43], a[180:183], v15, v16 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s36
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[74:79], v[32:37], a[184:187], v15, v16 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s46 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[74:79], v[26:31], a[188:191], v15, v16 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[20:25], v[26:31], a[192:195], v15, v16 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[20:25], v[32:37], a[196:199], v15, v16 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[20:25], v[38:43], a[200:203], v15, v16 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s46 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[20:25], v[44:49], a[204:207], v15, v16 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[74:77], v11 offset:46080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[20:25], v[50:55], a[208:211], v15, v17 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v8 offset:58880
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[20:25], v[56:61], a[212:215], v15, v17 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[20:25], v[62:67], a[216:219], v15, v17 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s45, s44, 0x1c000
	s_mov_b32 m0, s39
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[20:25], v[68:73], a[220:223], v15, v17 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[16:19], s45 offen lds
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[74:79], v[68:73], a[224:227], v15, v17 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[74:79], v[62:67], a[228:231], v15, v17 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[74:79], v[56:61], a[232:235], v15, v17 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s45, s43, 0x1bc00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[74:79], v[50:55], a[236:239], v15, v17 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[28:31], s45 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[74:79], v[44:49], a[240:243], v15, v16 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[74:79], v[38:43], a[244:247], v15, v16 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[74:79], v[32:37], a[248:251], v15, v16 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[74:79], v[26:31], a[252:255], v15, v16 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[14:15], v5
	ds_read_b64 v[16:17], v3 offset:1024
	ds_read_b128 v[20:23], v6 offset:6144
	ds_read_b64 v[24:25], v4 offset:22528
	ds_read_b128 v[26:29], v18 offset:30720
	ds_read_b64 v[30:31], v7 offset:47104
	ds_read_b128 v[32:35], v18 offset:31744
	ds_read_b64 v[36:37], v7 offset:47616
	ds_read_b128 v[38:41], v18 offset:32768
	ds_read_b64 v[42:43], v7 offset:48128
	ds_read_b128 v[44:47], v18 offset:33792
	ds_read_b64 v[48:49], v7 offset:48640
	ds_read_b128 v[50:53], v18 offset:34816
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:25], v[26:31], a[0:3], v14, v16 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[54:55], v7 offset:49152
	ds_read_b128 v[56:59], v18 offset:35840
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[20:25], v[32:37], a[4:7], v14, v16 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[60:61], v7 offset:49664
	s_mov_b32 m0, s3
	s_add_i32 s45, s42, 0x30000
	buffer_load_dwordx4 v1, s[20:23], s45 offen lds
	ds_read_b128 v[62:65], v18 offset:36864
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[20:25], v[38:43], a[8:11], v14, v16 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[66:67], v7 offset:50176
	ds_read_b128 v[68:71], v18 offset:37888
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[20:25], v[44:49], a[12:15], v14, v16 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[72:73], v7 offset:50688
	ds_read_b128 v[74:77], v6 offset:7168
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[20:25], v[50:55], a[16:19], v14, v17 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v4 offset:23040
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[20:25], v[56:61], a[20:23], v14, v17 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s45 offen offset:1024 lds
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[20:25], v[62:67], a[24:27], v14, v17 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[20:25], v[68:73], a[28:31], v14, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[74:79], v[68:73], a[32:35], v14, v17 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[74:79], v[62:67], a[36:39], v14, v17 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[74:79], v[56:61], a[40:43], v14, v17 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s45 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[74:79], v[50:55], a[44:47], v14, v17 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[20:23], v6 offset:8192
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[74:79], v[44:49], a[48:51], v14, v16 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[24:25], v4 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[74:79], v[38:43], a[52:55], v14, v16 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[74:79], v[32:37], a[56:59], v14, v16 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s45 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[74:79], v[26:31], a[60:63], v14, v16 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[20:25], v[26:31], a[64:67], v14, v16 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[20:25], v[32:37], a[68:71], v14, v16 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[20:25], v[38:43], a[72:75], v14, v16 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s46, s42, 0x31000
	s_mov_b32 m0, s4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[20:25], v[44:49], a[76:79], v14, v16 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s46 offen lds
	ds_read_b128 v[74:77], v6 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[20:25], v[50:55], a[80:83], v14, v17 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v4 offset:24064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[20:25], v[56:61], a[84:87], v14, v17 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[20:25], v[62:67], a[88:91], v14, v17 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[20:25], v[68:73], a[92:95], v14, v17 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s46 offen offset:1024 lds
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[74:79], v[68:73], a[96:99], v14, v17 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[74:79], v[62:67], a[100:103], v14, v17 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[74:79], v[56:61], a[104:107], v14, v17 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[74:79], v[50:55], a[108:111], v14, v17 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[20:23], v6 offset:10240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[74:79], v[44:49], a[112:115], v14, v16 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s41
	ds_read_b64 v[24:25], v4 offset:24576
	buffer_load_dwordx4 v1, s[24:27], s45 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[74:79], v[38:43], a[116:119], v14, v16 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[74:79], v[32:37], a[120:123], v14, v16 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[74:79], v[26:31], a[124:127], v14, v16 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[20:25], v[26:31], a[128:131], v15, v16 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s45 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[20:25], v[32:37], a[132:135], v15, v16 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[20:25], v[38:43], a[136:139], v15, v16 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[20:25], v[44:49], a[140:143], v15, v16 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[74:77], v6 offset:11264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[20:25], v[50:55], a[144:147], v15, v17 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v4 offset:25088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[20:25], v[56:61], a[148:151], v15, v17 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s45 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[20:25], v[62:67], a[152:155], v15, v17 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[20:25], v[68:73], a[156:159], v15, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[74:79], v[68:73], a[160:163], v15, v17 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[74:79], v[62:67], a[164:167], v15, v17 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s45 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[74:79], v[56:61], a[168:171], v15, v17 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[74:79], v[50:55], a[172:175], v15, v17 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[20:23], v6 offset:12288
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[74:79], v[44:49], a[176:179], v15, v16 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[24:25], v4 offset:25600
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[74:79], v[38:43], a[180:183], v15, v16 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[74:79], v[32:37], a[184:187], v15, v16 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s46 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[74:79], v[26:31], a[188:191], v15, v16 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[20:25], v[26:31], a[192:195], v15, v16 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[20:25], v[32:37], a[196:199], v15, v16 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[20:25], v[38:43], a[200:203], v15, v16 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s46 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[20:25], v[44:49], a[204:207], v15, v16 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[74:77], v6 offset:13312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[20:25], v[50:55], a[208:211], v15, v17 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v4 offset:26112
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[20:25], v[56:61], a[212:215], v15, v17 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[20:25], v[62:67], a[216:219], v15, v17 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s45, s44, 0x20000
	s_mov_b32 m0, s6
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[20:25], v[68:73], a[220:223], v15, v17 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[16:19], s45 offen lds
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[74:79], v[68:73], a[224:227], v15, v17 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[74:79], v[62:67], a[228:231], v15, v17 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[74:79], v[56:61], a[232:235], v15, v17 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s45, s43, 0x1fc00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[74:79], v[50:55], a[236:239], v15, v17 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[28:31], s45 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[74:79], v[44:49], a[240:243], v15, v16 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[74:79], v[38:43], a[244:247], v15, v16 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[74:79], v[32:37], a[248:251], v15, v16 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[74:79], v[26:31], a[252:255], v15, v16 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[14:15], v5 offset:2048
	ds_read_b64 v[16:17], v3 offset:3072
	ds_read_b128 v[20:23], v6 offset:55296
	ds_read_b64 v[24:25], v8 offset:6144
	ds_read_b128 v[26:29], v9 offset:14336
	ds_read_b64 v[30:31], v10 offset:30720
	ds_read_b128 v[32:35], v9 offset:15360
	ds_read_b64 v[36:37], v10 offset:31232
	ds_read_b128 v[38:41], v9 offset:16384
	ds_read_b64 v[42:43], v10 offset:31744
	ds_read_b128 v[44:47], v9 offset:17408
	ds_read_b64 v[48:49], v10 offset:32256
	ds_read_b128 v[50:53], v9 offset:18432
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:25], v[26:31], a[0:3], v14, v16 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[54:55], v10 offset:32768
	ds_read_b128 v[56:59], v9 offset:19456
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[20:25], v[32:37], a[4:7], v14, v16 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[60:61], v10 offset:33280
	s_mov_b32 m0, s33
	s_add_i32 s45, s42, 0x36000
	buffer_load_dwordx4 v1, s[20:23], s45 offen lds
	ds_read_b128 v[62:65], v9 offset:20480
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[20:25], v[38:43], a[8:11], v14, v16 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[66:67], v10 offset:33792
	ds_read_b128 v[68:71], v9 offset:21504
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[20:25], v[44:49], a[12:15], v14, v16 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[72:73], v10 offset:34304
	ds_read_b128 v[74:77], v6 offset:56320
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[20:25], v[50:55], a[16:19], v14, v17 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v8 offset:6656
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[20:25], v[56:61], a[20:23], v14, v17 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s45 offen offset:1024 lds
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[20:25], v[62:67], a[24:27], v14, v17 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[20:25], v[68:73], a[28:31], v14, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[74:79], v[68:73], a[32:35], v14, v17 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[74:79], v[62:67], a[36:39], v14, v17 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[74:79], v[56:61], a[40:43], v14, v17 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s45 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[74:79], v[50:55], a[44:47], v14, v17 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[20:23], v6 offset:57344
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[74:79], v[44:49], a[48:51], v14, v16 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[24:25], v8 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[74:79], v[38:43], a[52:55], v14, v16 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[74:79], v[32:37], a[56:59], v14, v16 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s45 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[74:79], v[26:31], a[60:63], v14, v16 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[20:25], v[26:31], a[64:67], v14, v16 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[20:25], v[32:37], a[68:71], v14, v16 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[20:25], v[38:43], a[72:75], v14, v16 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s46, s42, 0x37000
	s_mov_b32 m0, s8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[20:25], v[44:49], a[76:79], v14, v16 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s46 offen lds
	ds_read_b128 v[74:77], v6 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[20:25], v[50:55], a[80:83], v14, v17 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v8 offset:7680
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[20:25], v[56:61], a[84:87], v14, v17 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[20:25], v[62:67], a[88:91], v14, v17 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[20:25], v[68:73], a[92:95], v14, v17 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s46 offen offset:1024 lds
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[74:79], v[68:73], a[96:99], v14, v17 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[74:79], v[62:67], a[100:103], v14, v17 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[74:79], v[56:61], a[104:107], v14, v17 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[74:79], v[50:55], a[108:111], v14, v17 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[20:23], v6 offset:59392
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[74:79], v[44:49], a[112:115], v14, v16 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s34
	ds_read_b64 v[24:25], v8 offset:8192
	buffer_load_dwordx4 v1, s[24:27], s45 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[74:79], v[38:43], a[116:119], v14, v16 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[74:79], v[32:37], a[120:123], v14, v16 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[74:79], v[26:31], a[124:127], v14, v16 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[20:25], v[26:31], a[128:131], v15, v16 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s45 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[20:25], v[32:37], a[132:135], v15, v16 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[20:25], v[38:43], a[136:139], v15, v16 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[20:25], v[44:49], a[140:143], v15, v16 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[74:77], v6 offset:60416
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[20:25], v[50:55], a[144:147], v15, v17 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v8 offset:8704
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[20:25], v[56:61], a[148:151], v15, v17 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s45 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[20:25], v[62:67], a[152:155], v15, v17 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[20:25], v[68:73], a[156:159], v15, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[74:79], v[68:73], a[160:163], v15, v17 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[74:79], v[62:67], a[164:167], v15, v17 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s45 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[74:79], v[56:61], a[168:171], v15, v17 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[74:79], v[50:55], a[172:175], v15, v17 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[20:23], v6 offset:61440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[74:79], v[44:49], a[176:179], v15, v16 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[24:25], v8 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[74:79], v[38:43], a[180:183], v15, v16 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s35
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[74:79], v[32:37], a[184:187], v15, v16 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s46 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[74:79], v[26:31], a[188:191], v15, v16 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[20:25], v[26:31], a[192:195], v15, v16 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[20:25], v[32:37], a[196:199], v15, v16 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[20:25], v[38:43], a[200:203], v15, v16 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s46 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[20:25], v[44:49], a[204:207], v15, v16 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[74:77], v6 offset:62464
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[20:25], v[50:55], a[208:211], v15, v17 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v8 offset:9728
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[20:25], v[56:61], a[212:215], v15, v17 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[20:25], v[62:67], a[216:219], v15, v17 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s45, s44, 0x24000
	s_mov_b32 m0, s7
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[20:25], v[68:73], a[220:223], v15, v17 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[16:19], s45 offen lds
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[74:79], v[68:73], a[224:227], v15, v17 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[74:79], v[62:67], a[228:231], v15, v17 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[74:79], v[56:61], a[232:235], v15, v17 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s45, s43, 0x23c00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[74:79], v[50:55], a[236:239], v15, v17 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[28:31], s45 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[74:79], v[44:49], a[240:243], v15, v16 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[74:79], v[38:43], a[244:247], v15, v16 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[74:79], v[32:37], a[248:251], v15, v16 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[74:79], v[26:31], a[252:255], v15, v16 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[14:15], v5 offset:4096
	ds_read_b64 v[16:17], v3 offset:5120
	ds_read_b128 v[20:23], v11 offset:38912
	ds_read_b64 v[24:25], v8 offset:55296
	ds_read_b128 v[26:29], v9 offset:63488
	ds_read_b64 v[30:31], v12 offset:14336
	ds_read_b128 v[32:35], v9 offset:64512
	ds_read_b64 v[36:37], v12 offset:14848
	ds_read_b128 v[38:41], v13
	ds_read_b64 v[42:43], v12 offset:15360
	ds_read_b128 v[44:47], v13 offset:1024
	ds_read_b64 v[48:49], v12 offset:15872
	ds_read_b128 v[50:53], v13 offset:2048
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:25], v[26:31], a[0:3], v14, v16 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[54:55], v12 offset:16384
	ds_read_b128 v[56:59], v13 offset:3072
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[20:25], v[32:37], a[4:7], v14, v16 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[60:61], v12 offset:16896
	s_mov_b32 m0, s37
	s_add_i32 s45, s42, 0x3c000
	buffer_load_dwordx4 v1, s[20:23], s45 offen lds
	ds_read_b128 v[62:65], v13 offset:4096
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[20:25], v[38:43], a[8:11], v14, v16 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[66:67], v12 offset:17408
	ds_read_b128 v[68:71], v13 offset:5120
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[20:25], v[44:49], a[12:15], v14, v16 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[72:73], v12 offset:17920
	ds_read_b128 v[74:77], v11 offset:39936
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[20:25], v[50:55], a[16:19], v14, v17 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v8 offset:55808
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[20:25], v[56:61], a[20:23], v14, v17 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s45 offen offset:1024 lds
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[20:25], v[62:67], a[24:27], v14, v17 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[20:25], v[68:73], a[28:31], v14, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[74:79], v[68:73], a[32:35], v14, v17 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[74:79], v[62:67], a[36:39], v14, v17 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[74:79], v[56:61], a[40:43], v14, v17 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s45 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[74:79], v[50:55], a[44:47], v14, v17 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[20:23], v11 offset:40960
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[74:79], v[44:49], a[48:51], v14, v16 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[24:25], v8 offset:56320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[74:79], v[38:43], a[52:55], v14, v16 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[74:79], v[32:37], a[56:59], v14, v16 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s45 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[74:79], v[26:31], a[60:63], v14, v16 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[20:25], v[26:31], a[64:67], v14, v16 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[20:25], v[32:37], a[68:71], v14, v16 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[20:25], v[38:43], a[72:75], v14, v16 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s46, s42, 0x3d000
	s_mov_b32 m0, s38
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[20:25], v[44:49], a[76:79], v14, v16 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s46 offen lds
	ds_read_b128 v[74:77], v11 offset:41984
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[20:25], v[50:55], a[80:83], v14, v17 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v8 offset:56832
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[20:25], v[56:61], a[84:87], v14, v17 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[20:25], v[62:67], a[88:91], v14, v17 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[20:25], v[68:73], a[92:95], v14, v17 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s46 offen offset:1024 lds
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[74:79], v[68:73], a[96:99], v14, v17 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[74:79], v[62:67], a[100:103], v14, v17 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[74:79], v[56:61], a[104:107], v14, v17 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[74:79], v[50:55], a[108:111], v14, v17 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[20:23], v11 offset:43008
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[74:79], v[44:49], a[112:115], v14, v16 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s40
	ds_read_b64 v[24:25], v8 offset:57344
	buffer_load_dwordx4 v1, s[24:27], s45 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[74:79], v[38:43], a[116:119], v14, v16 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[74:79], v[32:37], a[120:123], v14, v16 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[74:79], v[26:31], a[124:127], v14, v16 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[20:25], v[26:31], a[128:131], v15, v16 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s45 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[20:25], v[32:37], a[132:135], v15, v16 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[20:25], v[38:43], a[136:139], v15, v16 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[20:25], v[44:49], a[140:143], v15, v16 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[74:77], v11 offset:44032
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[20:25], v[50:55], a[144:147], v15, v17 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v8 offset:57856
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[20:25], v[56:61], a[148:151], v15, v17 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s45 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[20:25], v[62:67], a[152:155], v15, v17 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[20:25], v[68:73], a[156:159], v15, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[74:79], v[68:73], a[160:163], v15, v17 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[74:79], v[62:67], a[164:167], v15, v17 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s45 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[74:79], v[56:61], a[168:171], v15, v17 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[74:79], v[50:55], a[172:175], v15, v17 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[20:23], v11 offset:45056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[74:79], v[44:49], a[176:179], v15, v16 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[24:25], v8 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[74:79], v[38:43], a[180:183], v15, v16 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s36
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[74:79], v[32:37], a[184:187], v15, v16 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s46 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[74:79], v[26:31], a[188:191], v15, v16 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[20:25], v[26:31], a[192:195], v15, v16 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[20:25], v[32:37], a[196:199], v15, v16 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[20:25], v[38:43], a[200:203], v15, v16 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s46 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[20:25], v[44:49], a[204:207], v15, v16 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[74:77], v11 offset:46080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[20:25], v[50:55], a[208:211], v15, v17 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v8 offset:58880
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[20:25], v[56:61], a[212:215], v15, v17 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[20:25], v[62:67], a[216:219], v15, v17 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s45, s44, 0x28000
	s_mov_b32 m0, s39
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[20:25], v[68:73], a[220:223], v15, v17 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[16:19], s45 offen lds
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[74:79], v[68:73], a[224:227], v15, v17 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[74:79], v[62:67], a[228:231], v15, v17 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[74:79], v[56:61], a[232:235], v15, v17 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s45, s43, 0x27c00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[74:79], v[50:55], a[236:239], v15, v17 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[28:31], s45 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[74:79], v[44:49], a[240:243], v15, v16 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[74:79], v[38:43], a[244:247], v15, v16 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[74:79], v[32:37], a[248:251], v15, v16 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[74:79], v[26:31], a[252:255], v15, v16 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[14:15], v5
	ds_read_b64 v[16:17], v3 offset:1024
	ds_read_b128 v[20:23], v6 offset:6144
	ds_read_b64 v[24:25], v4 offset:22528
	ds_read_b128 v[26:29], v18 offset:30720
	ds_read_b64 v[30:31], v7 offset:47104
	ds_read_b128 v[32:35], v18 offset:31744
	ds_read_b64 v[36:37], v7 offset:47616
	ds_read_b128 v[38:41], v18 offset:32768
	ds_read_b64 v[42:43], v7 offset:48128
	ds_read_b128 v[44:47], v18 offset:33792
	ds_read_b64 v[48:49], v7 offset:48640
	ds_read_b128 v[50:53], v18 offset:34816
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:25], v[26:31], a[0:3], v14, v16 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[54:55], v7 offset:49152
	ds_read_b128 v[56:59], v18 offset:35840
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[20:25], v[32:37], a[4:7], v14, v16 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[60:61], v7 offset:49664
	s_mov_b32 m0, s3
	s_add_i32 s45, s42, 0x42000
	buffer_load_dwordx4 v1, s[20:23], s45 offen lds
	ds_read_b128 v[62:65], v18 offset:36864
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[20:25], v[38:43], a[8:11], v14, v16 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[66:67], v7 offset:50176
	ds_read_b128 v[68:71], v18 offset:37888
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[20:25], v[44:49], a[12:15], v14, v16 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[72:73], v7 offset:50688
	ds_read_b128 v[74:77], v6 offset:7168
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[20:25], v[50:55], a[16:19], v14, v17 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v4 offset:23040
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[20:25], v[56:61], a[20:23], v14, v17 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s45 offen offset:1024 lds
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[20:25], v[62:67], a[24:27], v14, v17 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[20:25], v[68:73], a[28:31], v14, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[74:79], v[68:73], a[32:35], v14, v17 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[74:79], v[62:67], a[36:39], v14, v17 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[74:79], v[56:61], a[40:43], v14, v17 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s45 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[74:79], v[50:55], a[44:47], v14, v17 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[20:23], v6 offset:8192
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[74:79], v[44:49], a[48:51], v14, v16 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[24:25], v4 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[74:79], v[38:43], a[52:55], v14, v16 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[74:79], v[32:37], a[56:59], v14, v16 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s45 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[74:79], v[26:31], a[60:63], v14, v16 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[20:25], v[26:31], a[64:67], v14, v16 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[20:25], v[32:37], a[68:71], v14, v16 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[20:25], v[38:43], a[72:75], v14, v16 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s46, s42, 0x43000
	s_mov_b32 m0, s4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[20:25], v[44:49], a[76:79], v14, v16 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s46 offen lds
	ds_read_b128 v[74:77], v6 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[20:25], v[50:55], a[80:83], v14, v17 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v4 offset:24064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[20:25], v[56:61], a[84:87], v14, v17 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[20:25], v[62:67], a[88:91], v14, v17 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[20:25], v[68:73], a[92:95], v14, v17 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s46 offen offset:1024 lds
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[74:79], v[68:73], a[96:99], v14, v17 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[74:79], v[62:67], a[100:103], v14, v17 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[74:79], v[56:61], a[104:107], v14, v17 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[74:79], v[50:55], a[108:111], v14, v17 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[20:23], v6 offset:10240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[74:79], v[44:49], a[112:115], v14, v16 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s41
	ds_read_b64 v[24:25], v4 offset:24576
	buffer_load_dwordx4 v1, s[24:27], s45 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[74:79], v[38:43], a[116:119], v14, v16 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[74:79], v[32:37], a[120:123], v14, v16 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[74:79], v[26:31], a[124:127], v14, v16 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[20:25], v[26:31], a[128:131], v15, v16 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s45 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[20:25], v[32:37], a[132:135], v15, v16 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[20:25], v[38:43], a[136:139], v15, v16 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[20:25], v[44:49], a[140:143], v15, v16 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[74:77], v6 offset:11264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[20:25], v[50:55], a[144:147], v15, v17 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v4 offset:25088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[20:25], v[56:61], a[148:151], v15, v17 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s45 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[20:25], v[62:67], a[152:155], v15, v17 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[20:25], v[68:73], a[156:159], v15, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[74:79], v[68:73], a[160:163], v15, v17 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[74:79], v[62:67], a[164:167], v15, v17 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s45 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[74:79], v[56:61], a[168:171], v15, v17 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[74:79], v[50:55], a[172:175], v15, v17 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[20:23], v6 offset:12288
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[74:79], v[44:49], a[176:179], v15, v16 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[24:25], v4 offset:25600
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[74:79], v[38:43], a[180:183], v15, v16 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[74:79], v[32:37], a[184:187], v15, v16 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s46 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[74:79], v[26:31], a[188:191], v15, v16 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[20:25], v[26:31], a[192:195], v15, v16 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[20:25], v[32:37], a[196:199], v15, v16 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[20:25], v[38:43], a[200:203], v15, v16 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s46 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[20:25], v[44:49], a[204:207], v15, v16 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[74:77], v6 offset:13312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[20:25], v[50:55], a[208:211], v15, v17 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v4 offset:26112
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[20:25], v[56:61], a[212:215], v15, v17 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[20:25], v[62:67], a[216:219], v15, v17 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s45, s44, 0x2c000
	s_mov_b32 m0, s6
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[20:25], v[68:73], a[220:223], v15, v17 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[16:19], s45 offen lds
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[74:79], v[68:73], a[224:227], v15, v17 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[74:79], v[62:67], a[228:231], v15, v17 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[74:79], v[56:61], a[232:235], v15, v17 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s45, s43, 0x2bc00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[74:79], v[50:55], a[236:239], v15, v17 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[28:31], s45 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[74:79], v[44:49], a[240:243], v15, v16 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[74:79], v[38:43], a[244:247], v15, v16 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[74:79], v[32:37], a[248:251], v15, v16 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[74:79], v[26:31], a[252:255], v15, v16 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[14:15], v5 offset:2048
	ds_read_b64 v[16:17], v3 offset:3072
	ds_read_b128 v[20:23], v6 offset:55296
	ds_read_b64 v[24:25], v8 offset:6144
	ds_read_b128 v[26:29], v9 offset:14336
	ds_read_b64 v[30:31], v10 offset:30720
	ds_read_b128 v[32:35], v9 offset:15360
	ds_read_b64 v[36:37], v10 offset:31232
	ds_read_b128 v[38:41], v9 offset:16384
	ds_read_b64 v[42:43], v10 offset:31744
	ds_read_b128 v[44:47], v9 offset:17408
	ds_read_b64 v[48:49], v10 offset:32256
	ds_read_b128 v[50:53], v9 offset:18432
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:25], v[26:31], a[0:3], v14, v16 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[54:55], v10 offset:32768
	ds_read_b128 v[56:59], v9 offset:19456
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[20:25], v[32:37], a[4:7], v14, v16 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[60:61], v10 offset:33280
	s_mov_b32 m0, s33
	s_add_i32 s45, s42, 0x48000
	buffer_load_dwordx4 v1, s[20:23], s45 offen lds
	ds_read_b128 v[62:65], v9 offset:20480
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[20:25], v[38:43], a[8:11], v14, v16 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[66:67], v10 offset:33792
	ds_read_b128 v[68:71], v9 offset:21504
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[20:25], v[44:49], a[12:15], v14, v16 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[72:73], v10 offset:34304
	ds_read_b128 v[74:77], v6 offset:56320
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[20:25], v[50:55], a[16:19], v14, v17 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v8 offset:6656
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[20:25], v[56:61], a[20:23], v14, v17 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s45 offen offset:1024 lds
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[20:25], v[62:67], a[24:27], v14, v17 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[20:25], v[68:73], a[28:31], v14, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[74:79], v[68:73], a[32:35], v14, v17 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[74:79], v[62:67], a[36:39], v14, v17 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[74:79], v[56:61], a[40:43], v14, v17 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s45 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[74:79], v[50:55], a[44:47], v14, v17 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[20:23], v6 offset:57344
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[74:79], v[44:49], a[48:51], v14, v16 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[24:25], v8 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[74:79], v[38:43], a[52:55], v14, v16 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[74:79], v[32:37], a[56:59], v14, v16 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s45 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[74:79], v[26:31], a[60:63], v14, v16 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[20:25], v[26:31], a[64:67], v14, v16 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[20:25], v[32:37], a[68:71], v14, v16 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[20:25], v[38:43], a[72:75], v14, v16 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s46, s42, 0x49000
	s_mov_b32 m0, s8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[20:25], v[44:49], a[76:79], v14, v16 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s46 offen lds
	ds_read_b128 v[74:77], v6 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[20:25], v[50:55], a[80:83], v14, v17 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v8 offset:7680
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[20:25], v[56:61], a[84:87], v14, v17 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[20:25], v[62:67], a[88:91], v14, v17 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[20:25], v[68:73], a[92:95], v14, v17 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s46 offen offset:1024 lds
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[74:79], v[68:73], a[96:99], v14, v17 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[74:79], v[62:67], a[100:103], v14, v17 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[74:79], v[56:61], a[104:107], v14, v17 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[74:79], v[50:55], a[108:111], v14, v17 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[20:23], v6 offset:59392
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[74:79], v[44:49], a[112:115], v14, v16 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s34
	ds_read_b64 v[24:25], v8 offset:8192
	buffer_load_dwordx4 v1, s[24:27], s45 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[74:79], v[38:43], a[116:119], v14, v16 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[74:79], v[32:37], a[120:123], v14, v16 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[74:79], v[26:31], a[124:127], v14, v16 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[20:25], v[26:31], a[128:131], v15, v16 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s45 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[20:25], v[32:37], a[132:135], v15, v16 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[20:25], v[38:43], a[136:139], v15, v16 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[20:25], v[44:49], a[140:143], v15, v16 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[74:77], v6 offset:60416
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[20:25], v[50:55], a[144:147], v15, v17 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v8 offset:8704
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[20:25], v[56:61], a[148:151], v15, v17 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s45 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[20:25], v[62:67], a[152:155], v15, v17 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[20:25], v[68:73], a[156:159], v15, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[74:79], v[68:73], a[160:163], v15, v17 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[74:79], v[62:67], a[164:167], v15, v17 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s45 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[74:79], v[56:61], a[168:171], v15, v17 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[74:79], v[50:55], a[172:175], v15, v17 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[20:23], v6 offset:61440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[74:79], v[44:49], a[176:179], v15, v16 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[24:25], v8 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[74:79], v[38:43], a[180:183], v15, v16 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s35
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[74:79], v[32:37], a[184:187], v15, v16 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s46 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[74:79], v[26:31], a[188:191], v15, v16 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[20:25], v[26:31], a[192:195], v15, v16 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[20:25], v[32:37], a[196:199], v15, v16 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[20:25], v[38:43], a[200:203], v15, v16 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s46 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[20:25], v[44:49], a[204:207], v15, v16 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[74:77], v6 offset:62464
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[20:25], v[50:55], a[208:211], v15, v17 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v8 offset:9728
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[20:25], v[56:61], a[212:215], v15, v17 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[20:25], v[62:67], a[216:219], v15, v17 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s45, s44, 0x30000
	s_mov_b32 m0, s7
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[20:25], v[68:73], a[220:223], v15, v17 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[16:19], s45 offen lds
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[74:79], v[68:73], a[224:227], v15, v17 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[74:79], v[62:67], a[228:231], v15, v17 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[74:79], v[56:61], a[232:235], v15, v17 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s45, s43, 0x2fc00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[74:79], v[50:55], a[236:239], v15, v17 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[28:31], s45 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[74:79], v[44:49], a[240:243], v15, v16 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[74:79], v[38:43], a[244:247], v15, v16 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[74:79], v[32:37], a[248:251], v15, v16 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[74:79], v[26:31], a[252:255], v15, v16 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[14:15], v5 offset:4096
	ds_read_b64 v[16:17], v3 offset:5120
	ds_read_b128 v[20:23], v11 offset:38912
	ds_read_b64 v[24:25], v8 offset:55296
	ds_read_b128 v[26:29], v9 offset:63488
	ds_read_b64 v[30:31], v12 offset:14336
	ds_read_b128 v[32:35], v9 offset:64512
	ds_read_b64 v[36:37], v12 offset:14848
	ds_read_b128 v[38:41], v13
	ds_read_b64 v[42:43], v12 offset:15360
	ds_read_b128 v[44:47], v13 offset:1024
	ds_read_b64 v[48:49], v12 offset:15872
	ds_read_b128 v[50:53], v13 offset:2048
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:25], v[26:31], a[0:3], v14, v16 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[54:55], v12 offset:16384
	ds_read_b128 v[56:59], v13 offset:3072
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[20:25], v[32:37], a[4:7], v14, v16 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[60:61], v12 offset:16896
	s_mov_b32 m0, s37
	s_add_i32 s45, s42, 0x4e000
	buffer_load_dwordx4 v1, s[20:23], s45 offen lds
	ds_read_b128 v[62:65], v13 offset:4096
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[20:25], v[38:43], a[8:11], v14, v16 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[66:67], v12 offset:17408
	ds_read_b128 v[68:71], v13 offset:5120
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[20:25], v[44:49], a[12:15], v14, v16 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[72:73], v12 offset:17920
	ds_read_b128 v[74:77], v11 offset:39936
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[20:25], v[50:55], a[16:19], v14, v17 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v8 offset:55808
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[20:25], v[56:61], a[20:23], v14, v17 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s45 offen offset:1024 lds
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[20:25], v[62:67], a[24:27], v14, v17 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[20:25], v[68:73], a[28:31], v14, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[74:79], v[68:73], a[32:35], v14, v17 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[74:79], v[62:67], a[36:39], v14, v17 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[74:79], v[56:61], a[40:43], v14, v17 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s45 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[74:79], v[50:55], a[44:47], v14, v17 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[20:23], v11 offset:40960
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[74:79], v[44:49], a[48:51], v14, v16 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[24:25], v8 offset:56320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[74:79], v[38:43], a[52:55], v14, v16 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[74:79], v[32:37], a[56:59], v14, v16 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s45 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[74:79], v[26:31], a[60:63], v14, v16 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[20:25], v[26:31], a[64:67], v14, v16 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[20:25], v[32:37], a[68:71], v14, v16 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[20:25], v[38:43], a[72:75], v14, v16 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s46, s42, 0x4f000
	s_mov_b32 m0, s38
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[20:25], v[44:49], a[76:79], v14, v16 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s46 offen lds
	ds_read_b128 v[74:77], v11 offset:41984
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[20:25], v[50:55], a[80:83], v14, v17 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v8 offset:56832
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[20:25], v[56:61], a[84:87], v14, v17 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[20:25], v[62:67], a[88:91], v14, v17 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[20:25], v[68:73], a[92:95], v14, v17 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s46 offen offset:1024 lds
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[74:79], v[68:73], a[96:99], v14, v17 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[74:79], v[62:67], a[100:103], v14, v17 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[74:79], v[56:61], a[104:107], v14, v17 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[74:79], v[50:55], a[108:111], v14, v17 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[20:23], v11 offset:43008
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[74:79], v[44:49], a[112:115], v14, v16 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s40
	ds_read_b64 v[24:25], v8 offset:57344
	buffer_load_dwordx4 v1, s[24:27], s45 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[74:79], v[38:43], a[116:119], v14, v16 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[74:79], v[32:37], a[120:123], v14, v16 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[74:79], v[26:31], a[124:127], v14, v16 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[20:25], v[26:31], a[128:131], v15, v16 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s45 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[20:25], v[32:37], a[132:135], v15, v16 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[20:25], v[38:43], a[136:139], v15, v16 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[20:25], v[44:49], a[140:143], v15, v16 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[74:77], v11 offset:44032
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[20:25], v[50:55], a[144:147], v15, v17 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v8 offset:57856
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[20:25], v[56:61], a[148:151], v15, v17 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s45 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[20:25], v[62:67], a[152:155], v15, v17 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[20:25], v[68:73], a[156:159], v15, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[74:79], v[68:73], a[160:163], v15, v17 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[74:79], v[62:67], a[164:167], v15, v17 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s45 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[74:79], v[56:61], a[168:171], v15, v17 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[74:79], v[50:55], a[172:175], v15, v17 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[20:23], v11 offset:45056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[74:79], v[44:49], a[176:179], v15, v16 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[24:25], v8 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[74:79], v[38:43], a[180:183], v15, v16 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s36
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[74:79], v[32:37], a[184:187], v15, v16 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s46 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[74:79], v[26:31], a[188:191], v15, v16 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[20:25], v[26:31], a[192:195], v15, v16 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[20:25], v[32:37], a[196:199], v15, v16 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[20:25], v[38:43], a[200:203], v15, v16 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s46 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[20:25], v[44:49], a[204:207], v15, v16 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[74:77], v11 offset:46080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[20:25], v[50:55], a[208:211], v15, v17 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v8 offset:58880
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[20:25], v[56:61], a[212:215], v15, v17 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[20:25], v[62:67], a[216:219], v15, v17 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s45, s44, 0x34000
	s_mov_b32 m0, s39
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[20:25], v[68:73], a[220:223], v15, v17 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[16:19], s45 offen lds
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[74:79], v[68:73], a[224:227], v15, v17 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[74:79], v[62:67], a[228:231], v15, v17 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[74:79], v[56:61], a[232:235], v15, v17 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s45, s43, 0x33c00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[74:79], v[50:55], a[236:239], v15, v17 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[28:31], s45 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[74:79], v[44:49], a[240:243], v15, v16 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[74:79], v[38:43], a[244:247], v15, v16 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[74:79], v[32:37], a[248:251], v15, v16 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[74:79], v[26:31], a[252:255], v15, v16 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[14:15], v5
	ds_read_b64 v[16:17], v3 offset:1024
	ds_read_b128 v[20:23], v6 offset:6144
	ds_read_b64 v[24:25], v4 offset:22528
	ds_read_b128 v[26:29], v18 offset:30720
	ds_read_b64 v[30:31], v7 offset:47104
	ds_read_b128 v[32:35], v18 offset:31744
	ds_read_b64 v[36:37], v7 offset:47616
	ds_read_b128 v[38:41], v18 offset:32768
	ds_read_b64 v[42:43], v7 offset:48128
	ds_read_b128 v[44:47], v18 offset:33792
	ds_read_b64 v[48:49], v7 offset:48640
	ds_read_b128 v[50:53], v18 offset:34816
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:25], v[26:31], a[0:3], v14, v16 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[54:55], v7 offset:49152
	ds_read_b128 v[56:59], v18 offset:35840
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[20:25], v[32:37], a[4:7], v14, v16 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[60:61], v7 offset:49664
	s_mov_b32 m0, s3
	s_add_i32 s45, s42, 0x54000
	buffer_load_dwordx4 v1, s[20:23], s45 offen lds
	ds_read_b128 v[62:65], v18 offset:36864
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[20:25], v[38:43], a[8:11], v14, v16 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[66:67], v7 offset:50176
	ds_read_b128 v[68:71], v18 offset:37888
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[20:25], v[44:49], a[12:15], v14, v16 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[72:73], v7 offset:50688
	ds_read_b128 v[74:77], v6 offset:7168
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[20:25], v[50:55], a[16:19], v14, v17 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v4 offset:23040
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[20:25], v[56:61], a[20:23], v14, v17 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s45 offen offset:1024 lds
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[20:25], v[62:67], a[24:27], v14, v17 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[20:25], v[68:73], a[28:31], v14, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[74:79], v[68:73], a[32:35], v14, v17 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[74:79], v[62:67], a[36:39], v14, v17 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[74:79], v[56:61], a[40:43], v14, v17 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s45 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[74:79], v[50:55], a[44:47], v14, v17 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[20:23], v6 offset:8192
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[74:79], v[44:49], a[48:51], v14, v16 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[24:25], v4 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[74:79], v[38:43], a[52:55], v14, v16 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[74:79], v[32:37], a[56:59], v14, v16 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s45 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[74:79], v[26:31], a[60:63], v14, v16 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[20:25], v[26:31], a[64:67], v14, v16 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[20:25], v[32:37], a[68:71], v14, v16 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[20:25], v[38:43], a[72:75], v14, v16 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s42, s42, 0x55000
	s_mov_b32 m0, s4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[20:25], v[44:49], a[76:79], v14, v16 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s42 offen lds
	ds_read_b128 v[74:77], v6 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[20:25], v[50:55], a[80:83], v14, v17 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v4 offset:24064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[20:25], v[56:61], a[84:87], v14, v17 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[20:25], v[62:67], a[88:91], v14, v17 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[20:25], v[68:73], a[92:95], v14, v17 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s42 offen offset:1024 lds
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[74:79], v[68:73], a[96:99], v14, v17 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[74:79], v[62:67], a[100:103], v14, v17 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[74:79], v[56:61], a[104:107], v14, v17 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[74:79], v[50:55], a[108:111], v14, v17 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[20:23], v6 offset:10240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[74:79], v[44:49], a[112:115], v14, v16 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s41
	ds_read_b64 v[24:25], v4 offset:24576
	buffer_load_dwordx4 v1, s[24:27], s45 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[74:79], v[38:43], a[116:119], v14, v16 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[74:79], v[32:37], a[120:123], v14, v16 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[74:79], v[26:31], a[124:127], v14, v16 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[20:25], v[26:31], a[128:131], v15, v16 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s45 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[20:25], v[32:37], a[132:135], v15, v16 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[20:25], v[38:43], a[136:139], v15, v16 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[20:25], v[44:49], a[140:143], v15, v16 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[74:77], v6 offset:11264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[20:25], v[50:55], a[144:147], v15, v17 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v4 offset:25088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[20:25], v[56:61], a[148:151], v15, v17 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s45 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[20:25], v[62:67], a[152:155], v15, v17 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[20:25], v[68:73], a[156:159], v15, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[74:79], v[68:73], a[160:163], v15, v17 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[74:79], v[62:67], a[164:167], v15, v17 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s45 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[74:79], v[56:61], a[168:171], v15, v17 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[74:79], v[50:55], a[172:175], v15, v17 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[20:23], v6 offset:12288
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[74:79], v[44:49], a[176:179], v15, v16 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[24:25], v4 offset:25600
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[74:79], v[38:43], a[180:183], v15, v16 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[74:79], v[32:37], a[184:187], v15, v16 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s42 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[74:79], v[26:31], a[188:191], v15, v16 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[20:25], v[26:31], a[192:195], v15, v16 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[20:25], v[32:37], a[196:199], v15, v16 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[20:25], v[38:43], a[200:203], v15, v16 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s42 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[20:25], v[44:49], a[204:207], v15, v16 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[74:77], v6 offset:13312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[20:25], v[50:55], a[208:211], v15, v17 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v4 offset:26112
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[20:25], v[56:61], a[212:215], v15, v17 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[20:25], v[62:67], a[216:219], v15, v17 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s42, s44, 0x38000
	s_mov_b32 m0, s6
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[20:25], v[68:73], a[220:223], v15, v17 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[16:19], s42 offen lds
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[74:79], v[68:73], a[224:227], v15, v17 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[74:79], v[62:67], a[228:231], v15, v17 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[74:79], v[56:61], a[232:235], v15, v17 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s42, s43, 0x37c00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[74:79], v[50:55], a[236:239], v15, v17 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[28:31], s42 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[74:79], v[44:49], a[240:243], v15, v16 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[74:79], v[38:43], a[244:247], v15, v16 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[74:79], v[32:37], a[248:251], v15, v16 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[74:79], v[26:31], a[252:255], v15, v16 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s15, s15, 1
	s_cmp_lt_i32 s15, 1
	s_cbranch_scc1 .Lmain_loop
	s_add_i32 s1, s9, 0x1e0
	s_lshl_b32 s1, s1, 9
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[8:9], v5 offset:2048
	ds_read_b64 v[10:11], v3 offset:3072
	ds_read_b128 v[12:15], v6 offset:55296
	v_add_u32_e32 v19, 0x10000, v4
	ds_read_b64 v[16:17], v19 offset:6144
	v_add_u32_e32 v20, 0x10000, v18
	ds_read_b128 v[22:25], v20 offset:14336
	v_add_u32_e32 v21, 0x10000, v7
	ds_read_b64 v[26:27], v21 offset:30720
	ds_read_b128 v[28:31], v20 offset:15360
	ds_read_b64 v[32:33], v21 offset:31232
	ds_read_b128 v[34:37], v20 offset:16384
	ds_read_b64 v[38:39], v21 offset:31744
	ds_read_b128 v[40:43], v20 offset:17408
	ds_read_b64 v[44:45], v21 offset:32256
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[12:17], v[22:27], a[0:3], v8, v10 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[46:49], v20 offset:18432
	ds_read_b64 v[50:51], v21 offset:32768
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[12:17], v[28:33], a[4:7], v8, v10 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[52:55], v20 offset:19456
	ds_read_b64 v[56:57], v21 offset:33280
	s_mov_b32 m0, s33
	s_add_i32 s3, s2, 0x5a000
	buffer_load_dwordx4 v1, s[20:23], s3 offen lds
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[12:17], v[34:39], a[8:11], v8, v10 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[58:61], v20 offset:20480
	ds_read_b64 v[62:63], v21 offset:33792
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[12:17], v[40:45], a[12:15], v8, v10 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[64:67], v20 offset:21504
	ds_read_b64 v[68:69], v21 offset:34304
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[12:17], v[46:51], a[16:19], v8, v11 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[70:73], v6 offset:56320
	ds_read_b64 v[74:75], v19 offset:6656
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[12:17], v[52:57], a[20:23], v8, v11 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s3 offen offset:1024 lds
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[12:17], v[58:63], a[24:27], v8, v11 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[12:17], v[64:69], a[28:31], v8, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[70:75], v[64:69], a[32:35], v8, v11 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[70:75], v[58:63], a[36:39], v8, v11 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[70:75], v[52:57], a[40:43], v8, v11 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s3 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[70:75], v[46:51], a[44:47], v8, v11 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[12:15], v6 offset:57344
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[70:75], v[40:45], a[48:51], v8, v10 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[16:17], v19 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[70:75], v[34:39], a[52:55], v8, v10 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[70:75], v[28:33], a[56:59], v8, v10 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s3 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[70:75], v[22:27], a[60:63], v8, v10 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[12:17], v[22:27], a[64:67], v8, v10 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[12:17], v[28:33], a[68:71], v8, v10 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[12:17], v[34:39], a[72:75], v8, v10 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s2, s2, 0x5b000
	s_mov_b32 m0, s8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[12:17], v[40:45], a[76:79], v8, v10 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s2 offen lds
	ds_read_b128 v[70:73], v6 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[12:17], v[46:51], a[80:83], v8, v11 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[74:75], v19 offset:7680
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[12:17], v[52:57], a[84:87], v8, v11 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[12:17], v[58:63], a[88:91], v8, v11 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[12:17], v[64:69], a[92:95], v8, v11 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[20:23], s2 offen offset:1024 lds
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[70:75], v[64:69], a[96:99], v8, v11 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[70:75], v[58:63], a[100:103], v8, v11 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[70:75], v[52:57], a[104:107], v8, v11 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[70:75], v[46:51], a[108:111], v8, v11 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[12:15], v6 offset:59392
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[70:75], v[40:45], a[112:115], v8, v10 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s34
	ds_read_b64 v[16:17], v19 offset:8192
	buffer_load_dwordx4 v1, s[24:27], s3 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[70:75], v[34:39], a[116:119], v8, v10 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[70:75], v[28:33], a[120:123], v8, v10 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[70:75], v[22:27], a[124:127], v8, v10 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[12:17], v[22:27], a[128:131], v9, v10 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s3 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[12:17], v[28:33], a[132:135], v9, v10 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[12:17], v[34:39], a[136:139], v9, v10 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[12:17], v[40:45], a[140:143], v9, v10 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[70:73], v6 offset:60416
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[12:17], v[46:51], a[144:147], v9, v11 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[74:75], v19 offset:8704
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[12:17], v[52:57], a[148:151], v9, v11 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s3 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[12:17], v[58:63], a[152:155], v9, v11 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[12:17], v[64:69], a[156:159], v9, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[70:75], v[64:69], a[160:163], v9, v11 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[70:75], v[58:63], a[164:167], v9, v11 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s3 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[70:75], v[52:57], a[168:171], v9, v11 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[70:75], v[46:51], a[172:175], v9, v11 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[12:15], v6 offset:61440
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[70:75], v[40:45], a[176:179], v9, v10 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[16:17], v19 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[70:75], v[34:39], a[180:183], v9, v10 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_mov_b32 m0, s35
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[70:75], v[28:33], a[184:187], v9, v10 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s2 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[70:75], v[22:27], a[188:191], v9, v10 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[12:17], v[22:27], a[192:195], v9, v10 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[12:17], v[28:33], a[196:199], v9, v10 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[12:17], v[34:39], a[200:203], v9, v10 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v1, s[24:27], s2 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[12:17], v[40:45], a[204:207], v9, v10 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[70:73], v6 offset:62464
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[12:17], v[46:51], a[208:211], v9, v11 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[74:75], v19 offset:9728
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[12:17], v[52:57], a[212:215], v9, v11 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[12:17], v[58:63], a[216:219], v9, v11 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_i32 s1, s7, s1
	s_mov_b32 m0, s7
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[12:17], v[64:69], a[220:223], v9, v11 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[16:19], s1 offen lds
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[70:75], v[64:69], a[224:227], v9, v11 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[70:75], v[58:63], a[228:231], v9, v11 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[70:75], v[52:57], a[232:235], v9, v11 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_i32 s1, s32, 0x3bc00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[70:75], v[46:51], a[236:239], v9, v11 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dword v2, s[28:31], s1 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[70:75], v[40:45], a[240:243], v9, v10 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[70:75], v[34:39], a[244:247], v9, v10 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[70:75], v[28:33], a[248:251], v9, v10 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[70:75], v[22:27], a[252:255], v9, v10 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_waitcnt vmcnt(14) lgkmcnt(0)
	s_barrier
	v_add_u32_e32 v1, 0x10000, v6
	ds_read_b64 v[8:9], v5 offset:4096
	ds_read_b64 v[10:11], v3 offset:5120
	ds_read_b128 v[12:15], v1 offset:38912
	ds_read_b64 v[16:17], v19 offset:55296
	ds_read_b128 v[22:25], v20 offset:63488
	v_add_u32_e32 v2, 0x20000, v7
	ds_read_b64 v[26:27], v2 offset:14336
	ds_read_b128 v[28:31], v20 offset:64512
	ds_read_b64 v[32:33], v2 offset:14848
	v_add_u32_e32 v20, 0x20000, v18
	ds_read_b128 v[34:37], v20
	ds_read_b64 v[38:39], v2 offset:15360
	ds_read_b128 v[40:43], v20 offset:1024
	ds_read_b64 v[44:45], v2 offset:15872
	ds_read_b128 v[46:49], v20 offset:2048
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[12:17], v[22:27], a[0:3], v8, v10 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[50:51], v2 offset:16384
	ds_read_b128 v[52:55], v20 offset:3072
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[12:17], v[28:33], a[4:7], v8, v10 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[56:57], v2 offset:16896
	ds_read_b128 v[58:61], v20 offset:4096
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[12:17], v[34:39], a[8:11], v8, v10 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[62:63], v2 offset:17408
	ds_read_b128 v[64:67], v20 offset:5120
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[12:17], v[40:45], a[12:15], v8, v10 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[68:69], v2 offset:17920
	ds_read_b128 v[70:73], v1 offset:39936
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[12:17], v[46:51], a[16:19], v8, v11 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[74:75], v19 offset:55808
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[12:17], v[52:57], a[20:23], v8, v11 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[12:17], v[58:63], a[24:27], v8, v11 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[12:17], v[64:69], a[28:31], v8, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[70:75], v[64:69], a[32:35], v8, v11 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[70:75], v[58:63], a[36:39], v8, v11 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[70:75], v[52:57], a[40:43], v8, v11 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[70:75], v[46:51], a[44:47], v8, v11 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[12:15], v1 offset:40960
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[70:75], v[40:45], a[48:51], v8, v10 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[16:17], v19 offset:56320
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[70:75], v[34:39], a[52:55], v8, v10 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[70:75], v[28:33], a[56:59], v8, v10 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[70:75], v[22:27], a[60:63], v8, v10 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[12:17], v[22:27], a[64:67], v8, v10 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[12:17], v[28:33], a[68:71], v8, v10 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[12:17], v[34:39], a[72:75], v8, v10 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[12:17], v[40:45], a[76:79], v8, v10 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[70:73], v1 offset:41984
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[12:17], v[46:51], a[80:83], v8, v11 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[74:75], v19 offset:56832
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[12:17], v[52:57], a[84:87], v8, v11 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[12:17], v[58:63], a[88:91], v8, v11 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[12:17], v[64:69], a[92:95], v8, v11 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[70:75], v[64:69], a[96:99], v8, v11 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[70:75], v[58:63], a[100:103], v8, v11 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[70:75], v[52:57], a[104:107], v8, v11 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[70:75], v[46:51], a[108:111], v8, v11 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[12:15], v1 offset:43008
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[70:75], v[40:45], a[112:115], v8, v10 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[16:17], v19 offset:57344
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[70:75], v[34:39], a[116:119], v8, v10 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[70:75], v[28:33], a[120:123], v8, v10 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[70:75], v[22:27], a[124:127], v8, v10 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[12:17], v[22:27], a[128:131], v9, v10 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[12:17], v[28:33], a[132:135], v9, v10 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[12:17], v[34:39], a[136:139], v9, v10 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[12:17], v[40:45], a[140:143], v9, v10 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[70:73], v1 offset:44032
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[12:17], v[46:51], a[144:147], v9, v11 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[74:75], v19 offset:57856
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[12:17], v[52:57], a[148:151], v9, v11 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[12:17], v[58:63], a[152:155], v9, v11 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[12:17], v[64:69], a[156:159], v9, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[70:75], v[64:69], a[160:163], v9, v11 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[70:75], v[58:63], a[164:167], v9, v11 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[70:75], v[52:57], a[168:171], v9, v11 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[70:75], v[46:51], a[172:175], v9, v11 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[12:15], v1 offset:45056
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[70:75], v[40:45], a[176:179], v9, v10 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[16:17], v19 offset:58368
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[70:75], v[34:39], a[180:183], v9, v10 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[70:75], v[28:33], a[184:187], v9, v10 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[70:75], v[22:27], a[188:191], v9, v10 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[12:17], v[22:27], a[192:195], v9, v10 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[12:17], v[28:33], a[196:199], v9, v10 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[12:17], v[34:39], a[200:203], v9, v10 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[12:17], v[40:45], a[204:207], v9, v10 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[70:73], v1 offset:46080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[12:17], v[46:51], a[208:211], v9, v11 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[74:75], v19 offset:58880
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[12:17], v[52:57], a[212:215], v9, v11 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[12:17], v[58:63], a[216:219], v9, v11 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[12:17], v[64:69], a[220:223], v9, v11 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[70:75], v[64:69], a[224:227], v9, v11 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[70:75], v[58:63], a[228:231], v9, v11 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[70:75], v[52:57], a[232:235], v9, v11 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[70:75], v[46:51], a[236:239], v9, v11 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[70:75], v[40:45], a[240:243], v9, v10 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[70:75], v[34:39], a[244:247], v9, v10 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[70:75], v[28:33], a[248:251], v9, v10 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[70:75], v[22:27], a[252:255], v9, v10 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	ds_read_b64 v[8:9], v5
	ds_read_b64 v[2:3], v3 offset:1024
	ds_read_b128 v[10:13], v6 offset:6144
	ds_read_b64 v[14:15], v4 offset:22528
	ds_read_b128 v[20:23], v18 offset:30720
	ds_read_b64 v[24:25], v7 offset:47104
	ds_read_b128 v[26:29], v18 offset:31744
	ds_read_b64 v[30:31], v7 offset:47616
	ds_read_b128 v[32:35], v18 offset:32768
	ds_read_b64 v[36:37], v7 offset:48128
	ds_read_b128 v[38:41], v18 offset:33792
	ds_read_b64 v[42:43], v7 offset:48640
	ds_read_b128 v[44:47], v18 offset:34816
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[10:15], v[20:25], a[0:3], v8, v2 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[48:49], v7 offset:49152
	ds_read_b128 v[50:53], v18 offset:35840
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[10:15], v[26:31], a[4:7], v8, v2 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[54:55], v7 offset:49664
	ds_read_b128 v[56:59], v18 offset:36864
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[10:15], v[32:37], a[8:11], v8, v2 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[60:61], v7 offset:50176
	ds_read_b128 v[62:65], v18 offset:37888
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[10:15], v[38:43], a[12:15], v8, v2 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[66:67], v7 offset:50688
	ds_read_b128 v[68:71], v6 offset:7168
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[10:15], v[44:49], a[16:19], v8, v3 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[72:73], v4 offset:23040
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[10:15], v[50:55], a[20:23], v8, v3 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[10:15], v[56:61], a[24:27], v8, v3 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[10:15], v[62:67], a[28:31], v8, v3 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[68:73], v[62:67], a[32:35], v8, v3 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[68:73], v[56:61], a[36:39], v8, v3 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[68:73], v[50:55], a[40:43], v8, v3 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[68:73], v[44:49], a[44:47], v8, v3 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[10:13], v6 offset:8192
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[68:73], v[38:43], a[48:51], v8, v2 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[14:15], v4 offset:23552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[68:73], v[32:37], a[52:55], v8, v2 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[68:73], v[26:31], a[56:59], v8, v2 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[68:73], v[20:25], a[60:63], v8, v2 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[10:15], v[20:25], a[64:67], v8, v2 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[10:15], v[26:31], a[68:71], v8, v2 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[10:15], v[32:37], a[72:75], v8, v2 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[10:15], v[38:43], a[76:79], v8, v2 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[68:71], v6 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[10:15], v[44:49], a[80:83], v8, v3 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[72:73], v4 offset:24064
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[10:15], v[50:55], a[84:87], v8, v3 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[10:15], v[56:61], a[88:91], v8, v3 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[10:15], v[62:67], a[92:95], v8, v3 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[68:73], v[62:67], a[96:99], v8, v3 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[68:73], v[56:61], a[100:103], v8, v3 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[68:73], v[50:55], a[104:107], v8, v3 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[68:73], v[44:49], a[108:111], v8, v3 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[10:13], v6 offset:10240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[68:73], v[38:43], a[112:115], v8, v2 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[14:15], v4 offset:24576
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[68:73], v[32:37], a[116:119], v8, v2 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[68:73], v[26:31], a[120:123], v8, v2 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[68:73], v[20:25], a[124:127], v8, v2 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[10:15], v[20:25], a[128:131], v9, v2 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[10:15], v[26:31], a[132:135], v9, v2 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[10:15], v[32:37], a[136:139], v9, v2 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[10:15], v[38:43], a[140:143], v9, v2 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[68:71], v6 offset:11264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[10:15], v[44:49], a[144:147], v9, v3 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[72:73], v4 offset:25088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[10:15], v[50:55], a[148:151], v9, v3 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[10:15], v[56:61], a[152:155], v9, v3 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[10:15], v[62:67], a[156:159], v9, v3 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[68:73], v[62:67], a[160:163], v9, v3 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[68:73], v[56:61], a[164:167], v9, v3 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[68:73], v[50:55], a[168:171], v9, v3 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[68:73], v[44:49], a[172:175], v9, v3 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[10:13], v6 offset:12288
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[68:73], v[38:43], a[176:179], v9, v2 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[14:15], v4 offset:25600
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[68:73], v[32:37], a[180:183], v9, v2 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[68:73], v[26:31], a[184:187], v9, v2 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[68:73], v[20:25], a[188:191], v9, v2 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[10:15], v[20:25], a[192:195], v9, v2 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[10:15], v[26:31], a[196:199], v9, v2 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[10:15], v[32:37], a[200:203], v9, v2 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[10:15], v[38:43], a[204:207], v9, v2 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[68:71], v6 offset:13312
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[10:15], v[44:49], a[208:211], v9, v3 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[72:73], v4 offset:26112
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[10:15], v[50:55], a[212:215], v9, v3 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[10:15], v[56:61], a[216:219], v9, v3 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[10:15], v[62:67], a[220:223], v9, v3 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[68:73], v[62:67], a[224:227], v9, v3 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[68:73], v[56:61], a[228:231], v9, v3 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[68:73], v[50:55], a[232:235], v9, v3 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[68:73], v[44:49], a[236:239], v9, v3 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[68:73], v[38:43], a[240:243], v9, v2 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[68:73], v[32:37], a[244:247], v9, v2 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[68:73], v[26:31], a[248:251], v9, v2 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[68:73], v[20:25], a[252:255], v9, v2 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v1, 15, v0
	v_and_b32_e32 v2, 1, v1
	v_sub_u32_e32 v3, v1, v2
	v_ashrrev_i32_e32 v4, 31, v3
	v_cmp_eq_i32_e64 s[2:3], v2, 1
	v_lshrrev_b32_e32 v4, 31, v4
	v_lshrrev_b32_e32 v0, 4, v0
	v_lshlrev_b32_e32 v2, 1, v2
	v_add_u32_e32 v3, v3, v4
	v_lshl_add_u32 v2, v0, 2, v2
	v_ashrrev_i32_e32 v3, 1, v3
	v_lshlrev_b32_e32 v4, 2, v1
	v_lshlrev_b32_e32 v1, 4, v1
	s_mov_b32 s8, s10
	s_and_b32 s9, s11, 0xffff
	s_mov_b32 s10, 0x7ffffff0
	s_mov_b32 s11, 0x31014000
	s_mul_i32 s0, s0, 0x440
	s_mov_b32 s1, 0x44
	v_mad_u32_u24 v2, s1, v2, v3
	s_mov_b32 s1, 0x44
	v_mad_u32_u24 v3, s1, v0, v4
	v_lshl_add_u32 v0, v0, 13, v1
	v_mov_b32_e32 v1, 0x1000504
	v_mov_b32_e32 v4, 0x5040100
	v_cndmask_b32_e64 v1, v4, v1, s[2:3]
	v_mov_b32_e32 v4, 0x3020706
	v_mov_b32_e32 v5, 0x7060302
	v_cndmask_b32_e64 v4, v5, v4, s[2:3]
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_accvgpr_read_b32 v5, a0
	v_accvgpr_read_b32 v6, a1
	v_cvt_pk_bf16_f32 v6, v5, v6
	v_accvgpr_read_b32 v5, a2
	v_accvgpr_read_b32 v7, a3
	v_cvt_pk_bf16_f32 v7, v5, v7
	v_accvgpr_read_b32 v5, a4
	v_accvgpr_read_b32 v8, a5
	v_cvt_pk_bf16_f32 v8, v5, v8
	v_accvgpr_read_b32 v5, a6
	v_accvgpr_read_b32 v9, a7
	v_cvt_pk_bf16_f32 v9, v5, v9
	v_accvgpr_read_b32 v5, a8
	v_accvgpr_read_b32 v10, a9
	v_cvt_pk_bf16_f32 v10, v5, v10
	v_accvgpr_read_b32 v5, a10
	v_accvgpr_read_b32 v11, a11
	v_cvt_pk_bf16_f32 v11, v5, v11
	v_accvgpr_read_b32 v5, a12
	v_accvgpr_read_b32 v12, a13
	v_cvt_pk_bf16_f32 v12, v5, v12
	v_accvgpr_read_b32 v5, a14
	v_accvgpr_read_b32 v13, a15
	v_cvt_pk_bf16_f32 v13, v5, v13
	v_accvgpr_read_b32 v5, a16
	v_accvgpr_read_b32 v14, a17
	v_cvt_pk_bf16_f32 v14, v5, v14
	v_accvgpr_read_b32 v5, a18
	v_accvgpr_read_b32 v15, a19
	v_cvt_pk_bf16_f32 v15, v5, v15
	v_accvgpr_read_b32 v5, a20
	v_accvgpr_read_b32 v16, a21
	v_cvt_pk_bf16_f32 v16, v5, v16
	v_accvgpr_read_b32 v5, a22
	v_accvgpr_read_b32 v17, a23
	v_cvt_pk_bf16_f32 v17, v5, v17
	v_accvgpr_read_b32 v5, a24
	v_accvgpr_read_b32 v18, a25
	v_cvt_pk_bf16_f32 v18, v5, v18
	v_accvgpr_read_b32 v5, a26
	v_accvgpr_read_b32 v19, a27
	v_cvt_pk_bf16_f32 v19, v5, v19
	v_accvgpr_read_b32 v5, a28
	v_accvgpr_read_b32 v20, a29
	v_cvt_pk_bf16_f32 v20, v5, v20
	v_accvgpr_read_b32 v5, a30
	v_accvgpr_read_b32 v21, a31
	v_cvt_pk_bf16_f32 v21, v5, v21
	v_cndmask_b32_e64 v5, v7, v6, s[2:3]
	v_cndmask_b32_e64 v22, v9, v8, s[2:3]
	v_cndmask_b32_e64 v23, v11, v10, s[2:3]
	v_cndmask_b32_e64 v24, v13, v12, s[2:3]
	v_cndmask_b32_e64 v25, v15, v14, s[2:3]
	v_cndmask_b32_e64 v26, v17, v16, s[2:3]
	v_cndmask_b32_e64 v27, v19, v18, s[2:3]
	v_cndmask_b32_e64 v28, v21, v20, s[2:3]
	v_cndmask_b32_e64 v6, v6, v7, s[2:3]
	v_mov_b32_dpp v5, v5 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_lshl_u32 v2, s0, v2, 2
	v_cndmask_b32_e64 v7, v8, v9, s[2:3]
	v_mov_b32_dpp v8, v22 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v9, v10, v11, s[2:3]
	v_mov_b32_dpp v10, v23 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v11, v12, v13, s[2:3]
	v_mov_b32_dpp v12, v24 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v13, v14, v15, s[2:3]
	v_mov_b32_dpp v14, v25 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v15, v16, v17, s[2:3]
	v_mov_b32_dpp v16, v26 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v17, v18, v19, s[2:3]
	v_mov_b32_dpp v18, v27 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_lshl_b32 s1, s5, 13
	v_cndmask_b32_e64 v19, v20, v21, s[2:3]
	s_lshl_b32 s4, s12, 1
	v_mov_b32_dpp v20, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_perm_b32 v21, v5, v6, v1
	v_perm_b32 v5, v5, v6, v4
	v_add_u32_e32 v2, 0x1800, v2
	v_perm_b32 v6, v8, v7, v1
	v_perm_b32 v7, v8, v7, v4
	v_perm_b32 v8, v10, v9, v1
	v_perm_b32 v9, v10, v9, v4
	v_perm_b32 v10, v12, v11, v1
	v_perm_b32 v11, v12, v11, v4
	v_perm_b32 v12, v14, v13, v1
	v_perm_b32 v13, v14, v13, v4
	v_perm_b32 v14, v16, v15, v1
	v_perm_b32 v15, v16, v15, v4
	v_perm_b32 v16, v18, v17, v1
	v_perm_b32 v17, v18, v17, v4
	s_add_i32 s1, s1, s4
	v_perm_b32 v18, v20, v19, v1
	v_perm_b32 v19, v20, v19, v4
	ds_write2_b32 v2, v21, v5 offset1:68
	ds_write2_b32 v2, v6, v7 offset0:8 offset1:76
	ds_write2_b32 v2, v8, v9 offset0:16 offset1:84
	ds_write2_b32 v2, v10, v11 offset0:24 offset1:92
	ds_write2_b32 v2, v12, v13 offset0:32 offset1:100
	ds_write2_b32 v2, v14, v15 offset0:40 offset1:108
	ds_write2_b32 v2, v16, v17 offset0:48 offset1:116
	ds_write2_b32 v2, v18, v19 offset0:56 offset1:124
	s_waitcnt lgkmcnt(0)
	v_add_lshl_u32 v3, s0, v3, 2
	ds_read_b128 v[6:9], v3 offset:6144
	ds_read_b128 v[10:13], v3 offset:7232
	ds_read_b128 v[14:17], v3 offset:8320
	ds_read_b128 v[18:21], v3 offset:9408
	v_add_u32_e32 v5, 0x8000, v0
	v_add_u32_e32 v22, 0x10000, v0
	v_add_u32_e32 v23, 0x18000, v0
	s_waitcnt lgkmcnt(3)
	buffer_store_dwordx4 v[6:9], v0, s[8:11], s1 offen nt
	s_waitcnt lgkmcnt(2)
	buffer_store_dwordx4 v[10:13], v5, s[8:11], s1 offen nt
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[14:17], v22, s[8:11], s1 offen nt
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[18:21], v23, s[8:11], s1 offen nt
	s_waitcnt lgkmcnt(0)
	v_accvgpr_read_b32 v6, a60
	v_accvgpr_read_b32 v7, a61
	v_cvt_pk_bf16_f32 v6, v6, v7
	v_accvgpr_read_b32 v7, a62
	v_accvgpr_read_b32 v8, a63
	v_cvt_pk_bf16_f32 v7, v7, v8
	v_accvgpr_read_b32 v8, a56
	v_accvgpr_read_b32 v9, a57
	v_cvt_pk_bf16_f32 v8, v8, v9
	v_accvgpr_read_b32 v9, a58
	v_accvgpr_read_b32 v10, a59
	v_cvt_pk_bf16_f32 v9, v9, v10
	v_accvgpr_read_b32 v10, a52
	v_accvgpr_read_b32 v11, a53
	v_cvt_pk_bf16_f32 v10, v10, v11
	v_accvgpr_read_b32 v11, a54
	v_accvgpr_read_b32 v12, a55
	v_cvt_pk_bf16_f32 v11, v11, v12
	v_accvgpr_read_b32 v12, a48
	v_accvgpr_read_b32 v13, a49
	v_cvt_pk_bf16_f32 v12, v12, v13
	v_accvgpr_read_b32 v13, a50
	v_accvgpr_read_b32 v14, a51
	v_cvt_pk_bf16_f32 v13, v13, v14
	v_accvgpr_read_b32 v14, a44
	v_accvgpr_read_b32 v15, a45
	v_cvt_pk_bf16_f32 v14, v14, v15
	v_accvgpr_read_b32 v15, a46
	v_accvgpr_read_b32 v16, a47
	v_cvt_pk_bf16_f32 v15, v15, v16
	v_accvgpr_read_b32 v16, a40
	v_accvgpr_read_b32 v17, a41
	v_cvt_pk_bf16_f32 v16, v16, v17
	v_accvgpr_read_b32 v17, a42
	v_accvgpr_read_b32 v18, a43
	v_cvt_pk_bf16_f32 v17, v17, v18
	v_accvgpr_read_b32 v18, a36
	v_accvgpr_read_b32 v19, a37
	v_cvt_pk_bf16_f32 v18, v18, v19
	v_accvgpr_read_b32 v19, a38
	v_accvgpr_read_b32 v20, a39
	v_cvt_pk_bf16_f32 v19, v19, v20
	v_accvgpr_read_b32 v20, a32
	v_accvgpr_read_b32 v21, a33
	v_cvt_pk_bf16_f32 v20, v20, v21
	v_accvgpr_read_b32 v21, a34
	v_accvgpr_read_b32 v24, a35
	v_cvt_pk_bf16_f32 v21, v21, v24
	v_cndmask_b32_e64 v24, v7, v6, s[2:3]
	v_cndmask_b32_e64 v25, v9, v8, s[2:3]
	v_cndmask_b32_e64 v26, v11, v10, s[2:3]
	v_cndmask_b32_e64 v27, v13, v12, s[2:3]
	v_cndmask_b32_e64 v28, v15, v14, s[2:3]
	v_cndmask_b32_e64 v29, v17, v16, s[2:3]
	v_cndmask_b32_e64 v30, v19, v18, s[2:3]
	v_cndmask_b32_e64 v31, v21, v20, s[2:3]
	s_add_i32 s0, s5, 16
	v_cndmask_b32_e64 v6, v6, v7, s[2:3]
	v_mov_b32_dpp v7, v24 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v9, s[2:3]
	v_mov_b32_dpp v9, v25 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v10, v10, v11, s[2:3]
	v_mov_b32_dpp v11, v26 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v12, v12, v13, s[2:3]
	v_mov_b32_dpp v13, v27 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v14, v14, v15, s[2:3]
	v_mov_b32_dpp v15, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v16, v16, v17, s[2:3]
	v_mov_b32_dpp v17, v29 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v18, v18, v19, s[2:3]
	v_mov_b32_dpp v19, v30 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v20, v20, v21, s[2:3]
	v_mov_b32_dpp v21, v31 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_lshl_b32 s0, s0, 13
	v_perm_b32 v24, v7, v6, v1
	v_perm_b32 v6, v7, v6, v4
	v_perm_b32 v7, v9, v8, v1
	v_perm_b32 v8, v9, v8, v4
	v_perm_b32 v9, v11, v10, v1
	v_perm_b32 v10, v11, v10, v4
	v_perm_b32 v11, v13, v12, v1
	v_perm_b32 v12, v13, v12, v4
	v_perm_b32 v13, v15, v14, v1
	v_perm_b32 v14, v15, v14, v4
	v_perm_b32 v15, v17, v16, v1
	v_perm_b32 v16, v17, v16, v4
	v_perm_b32 v17, v19, v18, v1
	v_perm_b32 v18, v19, v18, v4
	v_perm_b32 v19, v21, v20, v1
	v_perm_b32 v20, v21, v20, v4
	s_add_i32 s0, s4, s0
	ds_write2_b32 v2, v24, v6 offset1:68
	ds_write2_b32 v2, v7, v8 offset0:8 offset1:76
	ds_write2_b32 v2, v9, v10 offset0:16 offset1:84
	ds_write2_b32 v2, v11, v12 offset0:24 offset1:92
	ds_write2_b32 v2, v13, v14 offset0:32 offset1:100
	ds_write2_b32 v2, v15, v16 offset0:40 offset1:108
	ds_write2_b32 v2, v17, v18 offset0:48 offset1:116
	ds_write2_b32 v2, v19, v20 offset0:56 offset1:124
	s_waitcnt lgkmcnt(0)
	ds_read_b128 v[6:9], v3 offset:6144
	ds_read_b128 v[10:13], v3 offset:7232
	ds_read_b128 v[14:17], v3 offset:8320
	ds_read_b128 v[18:21], v3 offset:9408
	s_waitcnt lgkmcnt(3)
	buffer_store_dwordx4 v[6:9], v0, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(2)
	buffer_store_dwordx4 v[10:13], v5, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[14:17], v22, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[18:21], v23, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(0)
	v_accvgpr_read_b32 v6, a64
	v_accvgpr_read_b32 v7, a65
	v_cvt_pk_bf16_f32 v6, v6, v7
	v_accvgpr_read_b32 v7, a66
	v_accvgpr_read_b32 v8, a67
	v_cvt_pk_bf16_f32 v7, v7, v8
	v_accvgpr_read_b32 v8, a68
	v_accvgpr_read_b32 v9, a69
	v_cvt_pk_bf16_f32 v8, v8, v9
	v_accvgpr_read_b32 v9, a70
	v_accvgpr_read_b32 v10, a71
	v_cvt_pk_bf16_f32 v9, v9, v10
	v_accvgpr_read_b32 v10, a72
	v_accvgpr_read_b32 v11, a73
	v_cvt_pk_bf16_f32 v10, v10, v11
	v_accvgpr_read_b32 v11, a74
	v_accvgpr_read_b32 v12, a75
	v_cvt_pk_bf16_f32 v11, v11, v12
	v_accvgpr_read_b32 v12, a76
	v_accvgpr_read_b32 v13, a77
	v_cvt_pk_bf16_f32 v12, v12, v13
	v_accvgpr_read_b32 v13, a78
	v_accvgpr_read_b32 v14, a79
	v_cvt_pk_bf16_f32 v13, v13, v14
	v_accvgpr_read_b32 v14, a80
	v_accvgpr_read_b32 v15, a81
	v_cvt_pk_bf16_f32 v14, v14, v15
	v_accvgpr_read_b32 v15, a82
	v_accvgpr_read_b32 v16, a83
	v_cvt_pk_bf16_f32 v15, v15, v16
	v_accvgpr_read_b32 v16, a84
	v_accvgpr_read_b32 v17, a85
	v_cvt_pk_bf16_f32 v16, v16, v17
	v_accvgpr_read_b32 v17, a86
	v_accvgpr_read_b32 v18, a87
	v_cvt_pk_bf16_f32 v17, v17, v18
	v_accvgpr_read_b32 v18, a88
	v_accvgpr_read_b32 v19, a89
	v_cvt_pk_bf16_f32 v18, v18, v19
	v_accvgpr_read_b32 v19, a90
	v_accvgpr_read_b32 v20, a91
	v_cvt_pk_bf16_f32 v19, v19, v20
	v_accvgpr_read_b32 v20, a92
	v_accvgpr_read_b32 v21, a93
	v_cvt_pk_bf16_f32 v20, v20, v21
	v_accvgpr_read_b32 v21, a94
	v_accvgpr_read_b32 v24, a95
	v_cvt_pk_bf16_f32 v21, v21, v24
	v_cndmask_b32_e64 v24, v7, v6, s[2:3]
	v_cndmask_b32_e64 v25, v9, v8, s[2:3]
	v_cndmask_b32_e64 v26, v11, v10, s[2:3]
	v_cndmask_b32_e64 v27, v13, v12, s[2:3]
	v_cndmask_b32_e64 v28, v15, v14, s[2:3]
	v_cndmask_b32_e64 v29, v17, v16, s[2:3]
	v_cndmask_b32_e64 v30, v19, v18, s[2:3]
	v_cndmask_b32_e64 v31, v21, v20, s[2:3]
	s_add_i32 s0, s5, 32
	v_cndmask_b32_e64 v6, v6, v7, s[2:3]
	v_mov_b32_dpp v7, v24 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v9, s[2:3]
	v_mov_b32_dpp v9, v25 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v10, v10, v11, s[2:3]
	v_mov_b32_dpp v11, v26 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v12, v12, v13, s[2:3]
	v_mov_b32_dpp v13, v27 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v14, v14, v15, s[2:3]
	v_mov_b32_dpp v15, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v16, v16, v17, s[2:3]
	v_mov_b32_dpp v17, v29 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v18, v18, v19, s[2:3]
	v_mov_b32_dpp v19, v30 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v20, v20, v21, s[2:3]
	v_mov_b32_dpp v21, v31 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_lshl_b32 s0, s0, 13
	v_perm_b32 v24, v7, v6, v1
	v_perm_b32 v6, v7, v6, v4
	v_perm_b32 v7, v9, v8, v1
	v_perm_b32 v8, v9, v8, v4
	v_perm_b32 v9, v11, v10, v1
	v_perm_b32 v10, v11, v10, v4
	v_perm_b32 v11, v13, v12, v1
	v_perm_b32 v12, v13, v12, v4
	v_perm_b32 v13, v15, v14, v1
	v_perm_b32 v14, v15, v14, v4
	v_perm_b32 v15, v17, v16, v1
	v_perm_b32 v16, v17, v16, v4
	v_perm_b32 v17, v19, v18, v1
	v_perm_b32 v18, v19, v18, v4
	v_perm_b32 v19, v21, v20, v1
	v_perm_b32 v20, v21, v20, v4
	s_add_i32 s0, s4, s0
	ds_write2_b32 v2, v24, v6 offset1:68
	ds_write2_b32 v2, v7, v8 offset0:8 offset1:76
	ds_write2_b32 v2, v9, v10 offset0:16 offset1:84
	ds_write2_b32 v2, v11, v12 offset0:24 offset1:92
	ds_write2_b32 v2, v13, v14 offset0:32 offset1:100
	ds_write2_b32 v2, v15, v16 offset0:40 offset1:108
	ds_write2_b32 v2, v17, v18 offset0:48 offset1:116
	ds_write2_b32 v2, v19, v20 offset0:56 offset1:124
	s_waitcnt lgkmcnt(0)
	ds_read_b128 v[6:9], v3 offset:6144
	ds_read_b128 v[10:13], v3 offset:7232
	ds_read_b128 v[14:17], v3 offset:8320
	ds_read_b128 v[18:21], v3 offset:9408
	s_waitcnt lgkmcnt(3)
	buffer_store_dwordx4 v[6:9], v0, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(2)
	buffer_store_dwordx4 v[10:13], v5, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[14:17], v22, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[18:21], v23, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(0)
	v_accvgpr_read_b32 v6, a124
	v_accvgpr_read_b32 v7, a125
	v_cvt_pk_bf16_f32 v6, v6, v7
	v_accvgpr_read_b32 v7, a126
	v_accvgpr_read_b32 v8, a127
	v_cvt_pk_bf16_f32 v7, v7, v8
	v_accvgpr_read_b32 v8, a120
	v_accvgpr_read_b32 v9, a121
	v_cvt_pk_bf16_f32 v8, v8, v9
	v_accvgpr_read_b32 v9, a122
	v_accvgpr_read_b32 v10, a123
	v_cvt_pk_bf16_f32 v9, v9, v10
	v_accvgpr_read_b32 v10, a116
	v_accvgpr_read_b32 v11, a117
	v_cvt_pk_bf16_f32 v10, v10, v11
	v_accvgpr_read_b32 v11, a118
	v_accvgpr_read_b32 v12, a119
	v_cvt_pk_bf16_f32 v11, v11, v12
	v_accvgpr_read_b32 v12, a112
	v_accvgpr_read_b32 v13, a113
	v_cvt_pk_bf16_f32 v12, v12, v13
	v_accvgpr_read_b32 v13, a114
	v_accvgpr_read_b32 v14, a115
	v_cvt_pk_bf16_f32 v13, v13, v14
	v_accvgpr_read_b32 v14, a108
	v_accvgpr_read_b32 v15, a109
	v_cvt_pk_bf16_f32 v14, v14, v15
	v_accvgpr_read_b32 v15, a110
	v_accvgpr_read_b32 v16, a111
	v_cvt_pk_bf16_f32 v15, v15, v16
	v_accvgpr_read_b32 v16, a104
	v_accvgpr_read_b32 v17, a105
	v_cvt_pk_bf16_f32 v16, v16, v17
	v_accvgpr_read_b32 v17, a106
	v_accvgpr_read_b32 v18, a107
	v_cvt_pk_bf16_f32 v17, v17, v18
	v_accvgpr_read_b32 v18, a100
	v_accvgpr_read_b32 v19, a101
	v_cvt_pk_bf16_f32 v18, v18, v19
	v_accvgpr_read_b32 v19, a102
	v_accvgpr_read_b32 v20, a103
	v_cvt_pk_bf16_f32 v19, v19, v20
	v_accvgpr_read_b32 v20, a96
	v_accvgpr_read_b32 v21, a97
	v_cvt_pk_bf16_f32 v20, v20, v21
	v_accvgpr_read_b32 v21, a98
	v_accvgpr_read_b32 v24, a99
	v_cvt_pk_bf16_f32 v21, v21, v24
	v_cndmask_b32_e64 v24, v7, v6, s[2:3]
	v_cndmask_b32_e64 v25, v9, v8, s[2:3]
	v_cndmask_b32_e64 v26, v11, v10, s[2:3]
	v_cndmask_b32_e64 v27, v13, v12, s[2:3]
	v_cndmask_b32_e64 v28, v15, v14, s[2:3]
	v_cndmask_b32_e64 v29, v17, v16, s[2:3]
	v_cndmask_b32_e64 v30, v19, v18, s[2:3]
	v_cndmask_b32_e64 v31, v21, v20, s[2:3]
	s_add_i32 s0, s5, 48
	v_cndmask_b32_e64 v6, v6, v7, s[2:3]
	v_mov_b32_dpp v7, v24 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v9, s[2:3]
	v_mov_b32_dpp v9, v25 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v10, v10, v11, s[2:3]
	v_mov_b32_dpp v11, v26 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v12, v12, v13, s[2:3]
	v_mov_b32_dpp v13, v27 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v14, v14, v15, s[2:3]
	v_mov_b32_dpp v15, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v16, v16, v17, s[2:3]
	v_mov_b32_dpp v17, v29 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v18, v18, v19, s[2:3]
	v_mov_b32_dpp v19, v30 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v20, v20, v21, s[2:3]
	v_mov_b32_dpp v21, v31 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_lshl_b32 s0, s0, 13
	v_perm_b32 v24, v7, v6, v1
	v_perm_b32 v6, v7, v6, v4
	v_perm_b32 v7, v9, v8, v1
	v_perm_b32 v8, v9, v8, v4
	v_perm_b32 v9, v11, v10, v1
	v_perm_b32 v10, v11, v10, v4
	v_perm_b32 v11, v13, v12, v1
	v_perm_b32 v12, v13, v12, v4
	v_perm_b32 v13, v15, v14, v1
	v_perm_b32 v14, v15, v14, v4
	v_perm_b32 v15, v17, v16, v1
	v_perm_b32 v16, v17, v16, v4
	v_perm_b32 v17, v19, v18, v1
	v_perm_b32 v18, v19, v18, v4
	v_perm_b32 v19, v21, v20, v1
	v_perm_b32 v20, v21, v20, v4
	s_add_i32 s0, s4, s0
	ds_write2_b32 v2, v24, v6 offset1:68
	ds_write2_b32 v2, v7, v8 offset0:8 offset1:76
	ds_write2_b32 v2, v9, v10 offset0:16 offset1:84
	ds_write2_b32 v2, v11, v12 offset0:24 offset1:92
	ds_write2_b32 v2, v13, v14 offset0:32 offset1:100
	ds_write2_b32 v2, v15, v16 offset0:40 offset1:108
	ds_write2_b32 v2, v17, v18 offset0:48 offset1:116
	ds_write2_b32 v2, v19, v20 offset0:56 offset1:124
	s_waitcnt lgkmcnt(0)
	ds_read_b128 v[6:9], v3 offset:6144
	ds_read_b128 v[10:13], v3 offset:7232
	ds_read_b128 v[14:17], v3 offset:8320
	ds_read_b128 v[18:21], v3 offset:9408
	s_waitcnt lgkmcnt(3)
	buffer_store_dwordx4 v[6:9], v0, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(2)
	buffer_store_dwordx4 v[10:13], v5, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[14:17], v22, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[18:21], v23, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(0)
	v_accvgpr_read_b32 v6, a128
	v_accvgpr_read_b32 v7, a129
	v_cvt_pk_bf16_f32 v6, v6, v7
	v_accvgpr_read_b32 v7, a130
	v_accvgpr_read_b32 v8, a131
	v_cvt_pk_bf16_f32 v7, v7, v8
	v_accvgpr_read_b32 v8, a132
	v_accvgpr_read_b32 v9, a133
	v_cvt_pk_bf16_f32 v8, v8, v9
	v_accvgpr_read_b32 v9, a134
	v_accvgpr_read_b32 v10, a135
	v_cvt_pk_bf16_f32 v9, v9, v10
	v_accvgpr_read_b32 v10, a136
	v_accvgpr_read_b32 v11, a137
	v_cvt_pk_bf16_f32 v10, v10, v11
	v_accvgpr_read_b32 v11, a138
	v_accvgpr_read_b32 v12, a139
	v_cvt_pk_bf16_f32 v11, v11, v12
	v_accvgpr_read_b32 v12, a140
	v_accvgpr_read_b32 v13, a141
	v_cvt_pk_bf16_f32 v12, v12, v13
	v_accvgpr_read_b32 v13, a142
	v_accvgpr_read_b32 v14, a143
	v_cvt_pk_bf16_f32 v13, v13, v14
	v_accvgpr_read_b32 v14, a144
	v_accvgpr_read_b32 v15, a145
	v_cvt_pk_bf16_f32 v14, v14, v15
	v_accvgpr_read_b32 v15, a146
	v_accvgpr_read_b32 v16, a147
	v_cvt_pk_bf16_f32 v15, v15, v16
	v_accvgpr_read_b32 v16, a148
	v_accvgpr_read_b32 v17, a149
	v_cvt_pk_bf16_f32 v16, v16, v17
	v_accvgpr_read_b32 v17, a150
	v_accvgpr_read_b32 v18, a151
	v_cvt_pk_bf16_f32 v17, v17, v18
	v_accvgpr_read_b32 v18, a152
	v_accvgpr_read_b32 v19, a153
	v_cvt_pk_bf16_f32 v18, v18, v19
	v_accvgpr_read_b32 v19, a154
	v_accvgpr_read_b32 v20, a155
	v_cvt_pk_bf16_f32 v19, v19, v20
	v_accvgpr_read_b32 v20, a156
	v_accvgpr_read_b32 v21, a157
	v_cvt_pk_bf16_f32 v20, v20, v21
	v_accvgpr_read_b32 v21, a158
	v_accvgpr_read_b32 v24, a159
	v_cvt_pk_bf16_f32 v21, v21, v24
	v_cndmask_b32_e64 v24, v7, v6, s[2:3]
	v_cndmask_b32_e64 v25, v9, v8, s[2:3]
	v_cndmask_b32_e64 v26, v11, v10, s[2:3]
	v_cndmask_b32_e64 v27, v13, v12, s[2:3]
	v_cndmask_b32_e64 v28, v15, v14, s[2:3]
	v_cndmask_b32_e64 v29, v17, v16, s[2:3]
	v_cndmask_b32_e64 v30, v19, v18, s[2:3]
	v_cndmask_b32_e64 v31, v21, v20, s[2:3]
	s_add_i32 s0, s5, 64
	v_cndmask_b32_e64 v6, v6, v7, s[2:3]
	v_mov_b32_dpp v7, v24 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v9, s[2:3]
	v_mov_b32_dpp v9, v25 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v10, v10, v11, s[2:3]
	v_mov_b32_dpp v11, v26 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v12, v12, v13, s[2:3]
	v_mov_b32_dpp v13, v27 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v14, v14, v15, s[2:3]
	v_mov_b32_dpp v15, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v16, v16, v17, s[2:3]
	v_mov_b32_dpp v17, v29 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v18, v18, v19, s[2:3]
	v_mov_b32_dpp v19, v30 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v20, v20, v21, s[2:3]
	v_mov_b32_dpp v21, v31 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_lshl_b32 s0, s0, 13
	v_perm_b32 v24, v7, v6, v1
	v_perm_b32 v6, v7, v6, v4
	v_perm_b32 v7, v9, v8, v1
	v_perm_b32 v8, v9, v8, v4
	v_perm_b32 v9, v11, v10, v1
	v_perm_b32 v10, v11, v10, v4
	v_perm_b32 v11, v13, v12, v1
	v_perm_b32 v12, v13, v12, v4
	v_perm_b32 v13, v15, v14, v1
	v_perm_b32 v14, v15, v14, v4
	v_perm_b32 v15, v17, v16, v1
	v_perm_b32 v16, v17, v16, v4
	v_perm_b32 v17, v19, v18, v1
	v_perm_b32 v18, v19, v18, v4
	v_perm_b32 v19, v21, v20, v1
	v_perm_b32 v20, v21, v20, v4
	s_add_i32 s0, s4, s0
	ds_write2_b32 v2, v24, v6 offset1:68
	ds_write2_b32 v2, v7, v8 offset0:8 offset1:76
	ds_write2_b32 v2, v9, v10 offset0:16 offset1:84
	ds_write2_b32 v2, v11, v12 offset0:24 offset1:92
	ds_write2_b32 v2, v13, v14 offset0:32 offset1:100
	ds_write2_b32 v2, v15, v16 offset0:40 offset1:108
	ds_write2_b32 v2, v17, v18 offset0:48 offset1:116
	ds_write2_b32 v2, v19, v20 offset0:56 offset1:124
	s_waitcnt lgkmcnt(0)
	ds_read_b128 v[6:9], v3 offset:6144
	ds_read_b128 v[10:13], v3 offset:7232
	ds_read_b128 v[14:17], v3 offset:8320
	ds_read_b128 v[18:21], v3 offset:9408
	s_waitcnt lgkmcnt(3)
	buffer_store_dwordx4 v[6:9], v0, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(2)
	buffer_store_dwordx4 v[10:13], v5, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[14:17], v22, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[18:21], v23, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(0)
	v_accvgpr_read_b32 v6, a188
	v_accvgpr_read_b32 v7, a189
	v_cvt_pk_bf16_f32 v6, v6, v7
	v_accvgpr_read_b32 v7, a190
	v_accvgpr_read_b32 v8, a191
	v_cvt_pk_bf16_f32 v7, v7, v8
	v_accvgpr_read_b32 v8, a184
	v_accvgpr_read_b32 v9, a185
	v_cvt_pk_bf16_f32 v8, v8, v9
	v_accvgpr_read_b32 v9, a186
	v_accvgpr_read_b32 v10, a187
	v_cvt_pk_bf16_f32 v9, v9, v10
	v_accvgpr_read_b32 v10, a180
	v_accvgpr_read_b32 v11, a181
	v_cvt_pk_bf16_f32 v10, v10, v11
	v_accvgpr_read_b32 v11, a182
	v_accvgpr_read_b32 v12, a183
	v_cvt_pk_bf16_f32 v11, v11, v12
	v_accvgpr_read_b32 v12, a176
	v_accvgpr_read_b32 v13, a177
	v_cvt_pk_bf16_f32 v12, v12, v13
	v_accvgpr_read_b32 v13, a178
	v_accvgpr_read_b32 v14, a179
	v_cvt_pk_bf16_f32 v13, v13, v14
	v_accvgpr_read_b32 v14, a172
	v_accvgpr_read_b32 v15, a173
	v_cvt_pk_bf16_f32 v14, v14, v15
	v_accvgpr_read_b32 v15, a174
	v_accvgpr_read_b32 v16, a175
	v_cvt_pk_bf16_f32 v15, v15, v16
	v_accvgpr_read_b32 v16, a168
	v_accvgpr_read_b32 v17, a169
	v_cvt_pk_bf16_f32 v16, v16, v17
	v_accvgpr_read_b32 v17, a170
	v_accvgpr_read_b32 v18, a171
	v_cvt_pk_bf16_f32 v17, v17, v18
	v_accvgpr_read_b32 v18, a164
	v_accvgpr_read_b32 v19, a165
	v_cvt_pk_bf16_f32 v18, v18, v19
	v_accvgpr_read_b32 v19, a166
	v_accvgpr_read_b32 v20, a167
	v_cvt_pk_bf16_f32 v19, v19, v20
	v_accvgpr_read_b32 v20, a160
	v_accvgpr_read_b32 v21, a161
	v_cvt_pk_bf16_f32 v20, v20, v21
	v_accvgpr_read_b32 v21, a162
	v_accvgpr_read_b32 v24, a163
	v_cvt_pk_bf16_f32 v21, v21, v24
	v_cndmask_b32_e64 v24, v7, v6, s[2:3]
	v_cndmask_b32_e64 v25, v9, v8, s[2:3]
	v_cndmask_b32_e64 v26, v11, v10, s[2:3]
	v_cndmask_b32_e64 v27, v13, v12, s[2:3]
	v_cndmask_b32_e64 v28, v15, v14, s[2:3]
	v_cndmask_b32_e64 v29, v17, v16, s[2:3]
	v_cndmask_b32_e64 v30, v19, v18, s[2:3]
	v_cndmask_b32_e64 v31, v21, v20, s[2:3]
	s_add_i32 s0, s5, 0x50
	v_cndmask_b32_e64 v6, v6, v7, s[2:3]
	v_mov_b32_dpp v7, v24 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v9, s[2:3]
	v_mov_b32_dpp v9, v25 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v10, v10, v11, s[2:3]
	v_mov_b32_dpp v11, v26 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v12, v12, v13, s[2:3]
	v_mov_b32_dpp v13, v27 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v14, v14, v15, s[2:3]
	v_mov_b32_dpp v15, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v16, v16, v17, s[2:3]
	v_mov_b32_dpp v17, v29 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v18, v18, v19, s[2:3]
	v_mov_b32_dpp v19, v30 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v20, v20, v21, s[2:3]
	v_mov_b32_dpp v21, v31 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_lshl_b32 s0, s0, 13
	v_perm_b32 v24, v7, v6, v1
	v_perm_b32 v6, v7, v6, v4
	v_perm_b32 v7, v9, v8, v1
	v_perm_b32 v8, v9, v8, v4
	v_perm_b32 v9, v11, v10, v1
	v_perm_b32 v10, v11, v10, v4
	v_perm_b32 v11, v13, v12, v1
	v_perm_b32 v12, v13, v12, v4
	v_perm_b32 v13, v15, v14, v1
	v_perm_b32 v14, v15, v14, v4
	v_perm_b32 v15, v17, v16, v1
	v_perm_b32 v16, v17, v16, v4
	v_perm_b32 v17, v19, v18, v1
	v_perm_b32 v18, v19, v18, v4
	v_perm_b32 v19, v21, v20, v1
	v_perm_b32 v20, v21, v20, v4
	s_add_i32 s0, s4, s0
	ds_write2_b32 v2, v24, v6 offset1:68
	ds_write2_b32 v2, v7, v8 offset0:8 offset1:76
	ds_write2_b32 v2, v9, v10 offset0:16 offset1:84
	ds_write2_b32 v2, v11, v12 offset0:24 offset1:92
	ds_write2_b32 v2, v13, v14 offset0:32 offset1:100
	ds_write2_b32 v2, v15, v16 offset0:40 offset1:108
	ds_write2_b32 v2, v17, v18 offset0:48 offset1:116
	ds_write2_b32 v2, v19, v20 offset0:56 offset1:124
	s_waitcnt lgkmcnt(0)
	ds_read_b128 v[6:9], v3 offset:6144
	ds_read_b128 v[10:13], v3 offset:7232
	ds_read_b128 v[14:17], v3 offset:8320
	ds_read_b128 v[18:21], v3 offset:9408
	s_waitcnt lgkmcnt(3)
	buffer_store_dwordx4 v[6:9], v0, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(2)
	buffer_store_dwordx4 v[10:13], v5, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[14:17], v22, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[18:21], v23, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(0)
	v_accvgpr_read_b32 v6, a192
	v_accvgpr_read_b32 v7, a193
	v_cvt_pk_bf16_f32 v6, v6, v7
	v_accvgpr_read_b32 v7, a194
	v_accvgpr_read_b32 v8, a195
	v_cvt_pk_bf16_f32 v7, v7, v8
	v_accvgpr_read_b32 v8, a196
	v_accvgpr_read_b32 v9, a197
	v_cvt_pk_bf16_f32 v8, v8, v9
	v_accvgpr_read_b32 v9, a198
	v_accvgpr_read_b32 v10, a199
	v_cvt_pk_bf16_f32 v9, v9, v10
	v_accvgpr_read_b32 v10, a200
	v_accvgpr_read_b32 v11, a201
	v_cvt_pk_bf16_f32 v10, v10, v11
	v_accvgpr_read_b32 v11, a202
	v_accvgpr_read_b32 v12, a203
	v_cvt_pk_bf16_f32 v11, v11, v12
	v_accvgpr_read_b32 v12, a204
	v_accvgpr_read_b32 v13, a205
	v_cvt_pk_bf16_f32 v12, v12, v13
	v_accvgpr_read_b32 v13, a206
	v_accvgpr_read_b32 v14, a207
	v_cvt_pk_bf16_f32 v13, v13, v14
	v_accvgpr_read_b32 v14, a208
	v_accvgpr_read_b32 v15, a209
	v_cvt_pk_bf16_f32 v14, v14, v15
	v_accvgpr_read_b32 v15, a210
	v_accvgpr_read_b32 v16, a211
	v_cvt_pk_bf16_f32 v15, v15, v16
	v_accvgpr_read_b32 v16, a212
	v_accvgpr_read_b32 v17, a213
	v_cvt_pk_bf16_f32 v16, v16, v17
	v_accvgpr_read_b32 v17, a214
	v_accvgpr_read_b32 v18, a215
	v_cvt_pk_bf16_f32 v17, v17, v18
	v_accvgpr_read_b32 v18, a216
	v_accvgpr_read_b32 v19, a217
	v_cvt_pk_bf16_f32 v18, v18, v19
	v_accvgpr_read_b32 v19, a218
	v_accvgpr_read_b32 v20, a219
	v_cvt_pk_bf16_f32 v19, v19, v20
	v_accvgpr_read_b32 v20, a220
	v_accvgpr_read_b32 v21, a221
	v_cvt_pk_bf16_f32 v20, v20, v21
	v_accvgpr_read_b32 v21, a222
	v_accvgpr_read_b32 v24, a223
	v_cvt_pk_bf16_f32 v21, v21, v24
	v_cndmask_b32_e64 v24, v7, v6, s[2:3]
	v_cndmask_b32_e64 v25, v9, v8, s[2:3]
	v_cndmask_b32_e64 v26, v11, v10, s[2:3]
	v_cndmask_b32_e64 v27, v13, v12, s[2:3]
	v_cndmask_b32_e64 v28, v15, v14, s[2:3]
	v_cndmask_b32_e64 v29, v17, v16, s[2:3]
	v_cndmask_b32_e64 v30, v19, v18, s[2:3]
	v_cndmask_b32_e64 v31, v21, v20, s[2:3]
	s_add_i32 s0, s5, 0x60
	v_cndmask_b32_e64 v6, v6, v7, s[2:3]
	v_mov_b32_dpp v7, v24 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v9, s[2:3]
	v_mov_b32_dpp v9, v25 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v10, v10, v11, s[2:3]
	v_mov_b32_dpp v11, v26 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v12, v12, v13, s[2:3]
	v_mov_b32_dpp v13, v27 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v14, v14, v15, s[2:3]
	v_mov_b32_dpp v15, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v16, v16, v17, s[2:3]
	v_mov_b32_dpp v17, v29 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v18, v18, v19, s[2:3]
	v_mov_b32_dpp v19, v30 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v20, v20, v21, s[2:3]
	v_mov_b32_dpp v21, v31 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_lshl_b32 s0, s0, 13
	v_perm_b32 v24, v7, v6, v1
	v_perm_b32 v6, v7, v6, v4
	v_perm_b32 v7, v9, v8, v1
	v_perm_b32 v8, v9, v8, v4
	v_perm_b32 v9, v11, v10, v1
	v_perm_b32 v10, v11, v10, v4
	v_perm_b32 v11, v13, v12, v1
	v_perm_b32 v12, v13, v12, v4
	v_perm_b32 v13, v15, v14, v1
	v_perm_b32 v14, v15, v14, v4
	v_perm_b32 v15, v17, v16, v1
	v_perm_b32 v16, v17, v16, v4
	v_perm_b32 v17, v19, v18, v1
	v_perm_b32 v18, v19, v18, v4
	v_perm_b32 v19, v21, v20, v1
	v_perm_b32 v20, v21, v20, v4
	s_add_i32 s0, s4, s0
	ds_write2_b32 v2, v24, v6 offset1:68
	ds_write2_b32 v2, v7, v8 offset0:8 offset1:76
	ds_write2_b32 v2, v9, v10 offset0:16 offset1:84
	ds_write2_b32 v2, v11, v12 offset0:24 offset1:92
	ds_write2_b32 v2, v13, v14 offset0:32 offset1:100
	ds_write2_b32 v2, v15, v16 offset0:40 offset1:108
	ds_write2_b32 v2, v17, v18 offset0:48 offset1:116
	ds_write2_b32 v2, v19, v20 offset0:56 offset1:124
	s_waitcnt lgkmcnt(0)
	ds_read_b128 v[6:9], v3 offset:6144
	ds_read_b128 v[10:13], v3 offset:7232
	ds_read_b128 v[14:17], v3 offset:8320
	ds_read_b128 v[18:21], v3 offset:9408
	s_waitcnt lgkmcnt(3)
	buffer_store_dwordx4 v[6:9], v0, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(2)
	buffer_store_dwordx4 v[10:13], v5, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[14:17], v22, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[18:21], v23, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(0)
	v_accvgpr_read_b32 v6, a252
	v_accvgpr_read_b32 v7, a253
	v_cvt_pk_bf16_f32 v6, v6, v7
	v_accvgpr_read_b32 v7, a254
	v_accvgpr_read_b32 v8, a255
	v_cvt_pk_bf16_f32 v7, v7, v8
	v_accvgpr_read_b32 v8, a248
	v_accvgpr_read_b32 v9, a249
	v_cvt_pk_bf16_f32 v8, v8, v9
	v_accvgpr_read_b32 v9, a250
	v_accvgpr_read_b32 v10, a251
	v_cvt_pk_bf16_f32 v9, v9, v10
	v_accvgpr_read_b32 v10, a244
	v_accvgpr_read_b32 v11, a245
	v_cvt_pk_bf16_f32 v10, v10, v11
	v_accvgpr_read_b32 v11, a246
	v_accvgpr_read_b32 v12, a247
	v_cvt_pk_bf16_f32 v11, v11, v12
	v_accvgpr_read_b32 v12, a240
	v_accvgpr_read_b32 v13, a241
	v_cvt_pk_bf16_f32 v12, v12, v13
	v_accvgpr_read_b32 v13, a242
	v_accvgpr_read_b32 v14, a243
	v_cvt_pk_bf16_f32 v13, v13, v14
	v_accvgpr_read_b32 v14, a236
	v_accvgpr_read_b32 v15, a237
	v_cvt_pk_bf16_f32 v14, v14, v15
	v_accvgpr_read_b32 v15, a238
	v_accvgpr_read_b32 v16, a239
	v_cvt_pk_bf16_f32 v15, v15, v16
	v_accvgpr_read_b32 v16, a232
	v_accvgpr_read_b32 v17, a233
	v_cvt_pk_bf16_f32 v16, v16, v17
	v_accvgpr_read_b32 v17, a234
	v_accvgpr_read_b32 v18, a235
	v_cvt_pk_bf16_f32 v17, v17, v18
	v_accvgpr_read_b32 v18, a228
	v_accvgpr_read_b32 v19, a229
	v_cvt_pk_bf16_f32 v18, v18, v19
	v_accvgpr_read_b32 v19, a230
	v_accvgpr_read_b32 v20, a231
	v_cvt_pk_bf16_f32 v19, v19, v20
	v_accvgpr_read_b32 v20, a224
	v_accvgpr_read_b32 v21, a225
	v_cvt_pk_bf16_f32 v20, v20, v21
	v_accvgpr_read_b32 v21, a226
	v_accvgpr_read_b32 v24, a227
	v_cvt_pk_bf16_f32 v21, v21, v24
	v_cndmask_b32_e64 v24, v7, v6, s[2:3]
	v_cndmask_b32_e64 v25, v9, v8, s[2:3]
	v_cndmask_b32_e64 v26, v11, v10, s[2:3]
	v_cndmask_b32_e64 v27, v13, v12, s[2:3]
	v_cndmask_b32_e64 v28, v15, v14, s[2:3]
	v_cndmask_b32_e64 v29, v17, v16, s[2:3]
	v_cndmask_b32_e64 v30, v19, v18, s[2:3]
	v_cndmask_b32_e64 v31, v21, v20, s[2:3]
	s_add_i32 s0, s5, 0x70
	v_cndmask_b32_e64 v6, v6, v7, s[2:3]
	v_mov_b32_dpp v7, v24 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v8, v8, v9, s[2:3]
	v_mov_b32_dpp v9, v25 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v10, v10, v11, s[2:3]
	v_mov_b32_dpp v11, v26 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v12, v12, v13, s[2:3]
	v_mov_b32_dpp v13, v27 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v14, v14, v15, s[2:3]
	v_mov_b32_dpp v15, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v16, v16, v17, s[2:3]
	v_mov_b32_dpp v17, v29 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v18, v18, v19, s[2:3]
	v_mov_b32_dpp v19, v30 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e64 v20, v20, v21, s[2:3]
	v_mov_b32_dpp v21, v31 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_lshl_b32 s0, s0, 13
	v_perm_b32 v24, v7, v6, v1
	v_perm_b32 v6, v7, v6, v4
	v_perm_b32 v7, v9, v8, v1
	v_perm_b32 v8, v9, v8, v4
	v_perm_b32 v9, v11, v10, v1
	v_perm_b32 v10, v11, v10, v4
	v_perm_b32 v11, v13, v12, v1
	v_perm_b32 v12, v13, v12, v4
	v_perm_b32 v13, v15, v14, v1
	v_perm_b32 v14, v15, v14, v4
	v_perm_b32 v15, v17, v16, v1
	v_perm_b32 v16, v17, v16, v4
	v_perm_b32 v17, v19, v18, v1
	v_perm_b32 v18, v19, v18, v4
	v_perm_b32 v1, v21, v20, v1
	v_perm_b32 v4, v21, v20, v4
	s_add_i32 s0, s4, s0
	ds_write2_b32 v2, v24, v6 offset1:68
	ds_write2_b32 v2, v7, v8 offset0:8 offset1:76
	ds_write2_b32 v2, v9, v10 offset0:16 offset1:84
	ds_write2_b32 v2, v11, v12 offset0:24 offset1:92
	ds_write2_b32 v2, v13, v14 offset0:32 offset1:100
	ds_write2_b32 v2, v15, v16 offset0:40 offset1:108
	ds_write2_b32 v2, v17, v18 offset0:48 offset1:116
	ds_write2_b32 v2, v1, v4 offset0:56 offset1:124
	s_waitcnt lgkmcnt(0)
	ds_read_b128 v[6:9], v3 offset:6144
	ds_read_b128 v[10:13], v3 offset:7232
	ds_read_b128 v[14:17], v3 offset:8320
	ds_read_b128 v[18:21], v3 offset:9408
	s_waitcnt lgkmcnt(3)
	buffer_store_dwordx4 v[6:9], v0, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(2)
	buffer_store_dwordx4 v[10:13], v5, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[14:17], v22, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[18:21], v23, s[8:11], s0 offen nt
	s_waitcnt lgkmcnt(0)
	s_endpgm
.Lrocke_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_4096x4096x2048_c0c1_kbd6c3ec4bc_end:
.size rocke_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_4096x4096x2048_c0c1_kbd6c3ec4bc, .Lrocke_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_4096x4096x2048_c0c1_kbd6c3ec4bc_end-rocke_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_4096x4096x2048_c0c1_kbd6c3ec4bc

.rodata
.p2align 6
.amdhsa_kernel rocke_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_4096x4096x2048_c0c1_kbd6c3ec4bc
  .amdhsa_group_segment_fixed_size 153600
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_kernarg_size 40
  .amdhsa_user_sgpr_count 12
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_user_sgpr_kernarg_preload_length 10
  .amdhsa_user_sgpr_kernarg_preload_offset 0
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_next_free_vgpr 336
  .amdhsa_next_free_sgpr 47
  .amdhsa_reserve_vcc 1
  .amdhsa_accum_offset 80
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
    Description: MX E2M3 GEMM at 4096x4096x2048 on tiled operands and shuffled scales
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
  AssertSizeEqual: {0: 4096, 1: 4096, 2: 1, 3: 2048}
  AssertStrideAEqual: {1: 2048}
  AssertStrideBEqual: {1: 2048}
  AssertStrideDEqual: {1: 4096}
  AssertAlphaValue: 1
amdhsa.target: amdgcn-amd-amdhsa--gfx950
amdhsa.version: [ 1, 1 ]
amdhsa.kernels:
  - .name: rocke_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_4096x4096x2048_c0c1_kbd6c3ec4bc
    .symbol: rocke_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_4096x4096x2048_c0c1_kbd6c3ec4bc.kd
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
    .sgpr_count: 53
    .vgpr_count: 336
    .agpr_count: 256
    .wavefront_size: 64
...
.end_amdgpu_metadata
