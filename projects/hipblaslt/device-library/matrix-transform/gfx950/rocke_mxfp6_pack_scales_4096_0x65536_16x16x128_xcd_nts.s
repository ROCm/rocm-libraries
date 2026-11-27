// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// MXFP6 scale packer for gfx950, launched by hipblasLtMatrixTransform. It copies E8M0 scales
// (HIP_R_8F_UE8M0, one per 32 K elements, K block fastest, as in
// HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0) to
// HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_128_4_EXT. 256 threads (wave64), 18432 bytes of
// static LDS, non-temporal stores.
//
// Fixed problem, enforced by the caller (rocblaslt_matrix_transform):
//   4096 rows (M or N indices) of 2048 K blocks (K = 65536), batch 1, so the source and the
//   destination are 8388608 bytes each. The grid must be exactly 512 workgroups: no load or
//   store is bounds-checked, so any other shape reads and writes out of range.
//
// Arguments: sa_src, sb_src, sa_dst, sb_dst, four 8-byte global pointers. The kernel reads
// only sa_src (offset 0) and sa_dst (offset 16); the second operand's pointers are never
// loaded.
//
// Edits to the provider's instruction listing (llvm-objdump output), none of which changes an
// instruction:
//   - the objdump banner is replaced by the directives and the symbol label below;
//   - every instruction line, with its address and encoding comment, and the s_nop padding
//     after s_endpgm are kept as listed, so .text matches the provider's code object byte for
//     byte;
//   - the kernel descriptor and metadata are written for code object v4 (amdhsa.version
//     [1, 1], no hidden arguments) from the provider's kernel descriptor, including its
//     register allocation (next_free_vgpr 32, accum_offset 32, next_free_sgpr 10; the assembler adds the 6 SGPRs that
//     gfx950 reserves for flat scratch).

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.amdhsa_code_object_version 4
.text
.globl rocke_mxfp6_pack_scales_4096_0x65536_16x16x128_xcd_nts
.p2align 8
.type rocke_mxfp6_pack_scales_4096_0x65536_16x16x128_xcd_nts,@function
rocke_mxfp6_pack_scales_4096_0x65536_16x16x128_xcd_nts:
	s_load_dwordx2 s[6:7], s[0:1], 0x0                         // 000000001700: C0060180 00000000
	s_load_dwordx2 s[4:5], s[0:1], 0x10                        // 000000001708: C0060100 00000010
	s_lshl_b32 s0, s2, 6                                       // 000000001710: 8E008602
	s_and_b32 s0, s0, 0x1c0                                    // 000000001714: 8600FF00 000001C0
	s_lshr_b32 s1, s2, 3                                       // 00000000171C: 8F018302
	s_add_i32 s0, s0, s1                                       // 000000001720: 81000100
	s_lshr_b32 s0, s0, 4                                       // 000000001724: 8F008400
	s_lshl_b32 s1, s0, 7                                       // 000000001728: 8E018700
	v_lshrrev_b32_e32 v22, 3, v0                               // 00000000172C: 202C0083
	v_and_b32_e32 v23, 7, v0                                   // 000000001730: 262E0087
	v_or_b32_e32 v1, s1, v22                                   // 000000001734: 28022C01
	s_lshl_b32 s2, s2, 2                                       // 000000001738: 8E028202
	v_lshlrev_b32_e32 v1, 9, v1                                // 00000000173C: 24020289
	s_and_b32 s2, s2, 0x1e0                                    // 000000001740: 8602FF02 000001E0
	v_lshlrev_b32_e32 v2, 2, v23                               // 000000001748: 24042E82
	v_or3_b32 v18, v1, s2, v2                                  // 00000000174C: D2020012 04080501
	v_or_b32_e32 v1, 0x100, v0                                 // 000000001754: 280200FF 00000100
	v_lshrrev_b32_e32 v24, 3, v1                               // 00000000175C: 20300283
	v_or_b32_e32 v1, s1, v24                                   // 000000001760: 28023001
	v_lshlrev_b32_e32 v2, 2, v0                                // 000000001764: 24040082
	v_mov_b32_e32 v19, 0                                       // 000000001768: 7E260280
	v_lshlrev_b32_e32 v1, 9, v1                                // 00000000176C: 24020289
	v_and_b32_e32 v25, 28, v2                                  // 000000001770: 2632049C
	s_waitcnt lgkmcnt(0)                                       // 000000001774: BF8CC07F
	v_lshl_add_u64 v[10:11], v[18:19], 2, s[6:7]               // 000000001778: D208000A 00190512
	v_or3_b32 v18, v1, s2, v25                                 // 000000001780: D2020012 04640501
	v_or_b32_e32 v1, 0x200, v0                                 // 000000001788: 280200FF 00000200
	v_lshrrev_b32_e32 v26, 3, v1                               // 000000001790: 20340283
	v_or_b32_e32 v0, 0x300, v0                                 // 000000001794: 280000FF 00000300
	v_or_b32_e32 v1, s1, v26                                   // 00000000179C: 28023401
	v_lshrrev_b32_e32 v27, 3, v0                               // 0000000017A0: 20360083
	v_lshlrev_b32_e32 v1, 9, v1                                // 0000000017A4: 24020289
	v_or_b32_e32 v0, s1, v27                                   // 0000000017A8: 28003601
	v_lshl_add_u64 v[12:13], v[18:19], 2, s[6:7]               // 0000000017AC: D208000C 00190512
	v_or3_b32 v18, v1, s2, v25                                 // 0000000017B4: D2020012 04640501
	v_lshlrev_b32_e32 v0, 9, v0                                // 0000000017BC: 24000089
	v_lshl_add_u64 v[20:21], v[18:19], 2, s[6:7]               // 0000000017C0: D2080014 00190512
	v_or3_b32 v18, v0, s2, v25                                 // 0000000017C8: D2020012 04640500
	global_load_dwordx4 v[2:5], v[10:11], off                  // 0000000017D0: DC5C8000 027F000A
	global_load_dwordx4 v[6:9], v[12:13], off                  // 0000000017D8: DC5C8000 067F000C
	v_lshl_add_u64 v[0:1], v[18:19], 2, s[6:7]                 // 0000000017E0: D2080000 00190512
	global_load_dwordx4 v[10:13], v[20:21], off                // 0000000017E8: DC5C8000 0A7F0014
	global_load_dwordx4 v[14:17], v[0:1], off                  // 0000000017F0: DC5C8000 0E7F0000
	s_movk_i32 s1, 0x90                                        // 0000000017F8: B0010090
	s_movk_i32 s3, 0x120                                       // 0000000017FC: B0030120
	v_lshlrev_b32_e32 v0, 4, v23                               // 000000001800: 24002E84
	v_lshlrev_b32_e32 v18, 1, v23                              // 000000001804: 24242E81
	v_lshlrev_b32_e32 v1, 2, v22                               // 000000001808: 24022C82
	v_lshlrev_b32_e32 v20, 2, v25                              // 00000000180C: 24283282
	v_mad_u32_u24 v0, v22, s1, v0                              // 000000001810: D1C30000 04000316
	v_mad_u32_u24 v21, v23, s3, v1                             // 000000001818: D1C30015 04040717
	v_or_b32_e32 v23, 16, v18                                  // 000000001820: 282E2490
	v_mad_u32_u24 v24, v24, s1, v20                            // 000000001824: D1C30018 04500318
	v_mad_u32_u24 v25, v26, s1, v20                            // 00000000182C: D1C30019 0450031A
	v_mad_u32_u24 v20, v27, s1, v20                            // 000000001834: D1C30014 0450031B
	v_mad_u32_u24 v26, v23, s1, v1                             // 00000000183C: D1C3001A 04040317
	v_add_u32_e32 v27, 0x90, v21                               // 000000001844: 68362AFF 00000090
	v_add_u32_e32 v28, 0x2400, v21                             // 00000000184C: 68382AFF 00002400
	v_add_u32_e32 v29, 0x2c00, v21                             // 000000001854: 683A2AFF 00002C00
	v_add_u32_e32 v30, 0x3400, v21                             // 00000000185C: 683C2AFF 00003400
	v_add_u32_e32 v31, 0x3c00, v21                             // 000000001864: 683E2AFF 00003C00
	s_mov_b32 s6, 0x5010400                                    // 00000000186C: BE8600FF 05010400
	s_mov_b32 s7, 0x7030602                                    // 000000001874: BE8700FF 07030602
	s_mov_b32 s8, 0x5040100                                    // 00000000187C: BE8800FF 05040100
	s_mov_b32 s9, 0x7060302                                    // 000000001884: BE8900FF 07060302
	s_lshl_b32 s0, s0, 6                                       // 00000000188C: 8E008600
	s_waitcnt vmcnt(3)                                         // 000000001890: BF8C0F73
	ds_write_b128 v0, v[2:5]                                   // 000000001894: D9BE0000 00000200
	s_waitcnt vmcnt(2)                                         // 00000000189C: BF8C0F72
	ds_write_b128 v24, v[6:9]                                  // 0000000018A0: D9BE0000 00000618
	s_waitcnt vmcnt(1)                                         // 0000000018A8: BF8C0F71
	ds_write_b128 v25, v[10:13]                                // 0000000018AC: D9BE0000 00000A19
	s_waitcnt vmcnt(0)                                         // 0000000018B4: BF8C0F70
	ds_write_b128 v20, v[14:17]                                // 0000000018B8: D9BE0000 00000E14
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 0000000018C0: BF8C0070
	s_barrier                                                  // 0000000018C4: BF8A0000
	ds_read2_b32 v[0:1], v21 offset1:36                        // 0000000018C8: D86E2400 00000015
	ds_read2st64_b32 v[2:3], v26 offset1:9                     // 0000000018D0: D8700900 0200001A
	ds_read2st64_b32 v[4:5], v27 offset0:9 offset1:18          // 0000000018D8: D8701209 0400001B
	ds_read_b32 v8, v26 offset:4608                            // 0000000018E0: D86C1200 0800001A
	ds_read_b32 v9, v21 offset:7056                            // 0000000018E8: D86C1B90 09000015
	ds_read2_b32 v[6:7], v28 offset1:36                        // 0000000018F0: D86E2400 0600001C
	ds_read2_b32 v[10:11], v29 offset0:64 offset1:100          // 0000000018F8: D86E6440 0A00001D
	ds_read2_b32 v[16:17], v30 offset0:128 offset1:164         // 000000001900: D86EA480 1000001E
	ds_read2_b32 v[20:21], v31 offset0:192 offset1:228         // 000000001908: D86EE4C0 1400001F
	s_waitcnt lgkmcnt(7)                                       // 000000001910: BF8CC77F
	v_perm_b32 v12, v2, v0, s6                                 // 000000001914: D1ED000C 001A0102
	v_perm_b32 v2, v2, v0, s7                                  // 00000000191C: D1ED0002 001E0102
	s_waitcnt lgkmcnt(5)                                       // 000000001924: BF8CC57F
	v_perm_b32 v13, v8, v3, s6                                 // 000000001928: D1ED000D 001A0708
	v_perm_b32 v3, v8, v3, s7                                  // 000000001930: D1ED0003 001E0708
	v_perm_b32 v14, v4, v1, s6                                 // 000000001938: D1ED000E 001A0304
	s_waitcnt lgkmcnt(4)                                       // 000000001940: BF8CC47F
	v_perm_b32 v15, v9, v5, s6                                 // 000000001944: D1ED000F 001A0B09
	v_perm_b32 v1, v4, v1, s7                                  // 00000000194C: D1ED0001 001E0304
	v_perm_b32 v5, v9, v5, s7                                  // 000000001954: D1ED0005 001E0B09
	s_waitcnt lgkmcnt(2)                                       // 00000000195C: BF8CC27F
	v_perm_b32 v9, v10, v6, s6                                 // 000000001960: D1ED0009 001A0D0A
	v_perm_b32 v24, v10, v6, s7                                // 000000001968: D1ED0018 001E0D0A
	v_perm_b32 v0, v13, v12, s8                                // 000000001970: D1ED0000 0022190D
	v_perm_b32 v4, v13, v12, s9                                // 000000001978: D1ED0004 0026190D
	v_perm_b32 v8, v3, v2, s8                                  // 000000001980: D1ED0008 00220503
	v_perm_b32 v12, v3, v2, s9                                 // 000000001988: D1ED000C 00260503
	v_perm_b32 v2, v15, v14, s8                                // 000000001990: D1ED0002 00221D0F
	v_perm_b32 v6, v15, v14, s9                                // 000000001998: D1ED0006 00261D0F
	s_waitcnt lgkmcnt(0)                                       // 0000000019A0: BF8CC07F
	v_perm_b32 v3, v20, v16, s6                                // 0000000019A4: D1ED0003 001A2114
	v_perm_b32 v13, v20, v16, s7                               // 0000000019AC: D1ED000D 001E2114
	v_perm_b32 v15, v11, v7, s6                                // 0000000019B4: D1ED000F 001A0F0B
	v_perm_b32 v16, v11, v7, s7                                // 0000000019BC: D1ED0010 001E0F0B
	v_perm_b32 v7, v21, v17, s6                                // 0000000019C4: D1ED0007 001A2315
	v_perm_b32 v17, v21, v17, s7                               // 0000000019CC: D1ED0011 001E2315
	v_perm_b32 v10, v5, v1, s8                                 // 0000000019D4: D1ED000A 00220305
	v_perm_b32 v14, v5, v1, s9                                 // 0000000019DC: D1ED000E 00260305
	v_perm_b32 v1, v3, v9, s8                                  // 0000000019E4: D1ED0001 00221303
	v_perm_b32 v5, v3, v9, s9                                  // 0000000019EC: D1ED0005 00261303
	v_perm_b32 v3, v7, v15, s8                                 // 0000000019F4: D1ED0003 00221F07
	v_perm_b32 v7, v7, v15, s9                                 // 0000000019FC: D1ED0007 00261F07
	v_perm_b32 v11, v17, v16, s8                               // 000000001A04: D1ED000B 00222111
	v_perm_b32 v15, v17, v16, s9                               // 000000001A0C: D1ED000F 00262111
	v_or_b32_e32 v16, s2, v22                                  // 000000001A14: 28202C02
	v_lshl_add_u32 v20, v16, 11, s0                            // 000000001A18: D1FD0014 00011710
	v_or_b32_e32 v16, v20, v18                                 // 000000001A20: 28202514
	v_lshlrev_b32_e32 v18, 1, v16                              // 000000001A24: 24242081
	v_lshl_add_u64 v[16:17], v[18:19], 2, s[4:5]               // 000000001A28: D2080010 00110512
	global_store_dwordx4 v[16:17], v[0:3], off nt              // 000000001A30: DC7E8000 007F0010
	v_perm_b32 v9, v13, v24, s8                                // 000000001A38: D1ED0009 0022310D
	v_perm_b32 v13, v13, v24, s9                               // 000000001A40: D1ED000D 0026310D
	v_or_b32_e32 v0, v20, v23                                  // 000000001A48: 28002F14
	v_lshlrev_b32_e32 v0, 1, v0                                // 000000001A4C: 24000081
	v_mov_b32_e32 v1, v19                                      // 000000001A50: 7E020313
	v_lshl_add_u64 v[0:1], v[0:1], 2, s[4:5]                   // 000000001A54: D2080000 00110500
	global_store_dwordx4 v[0:1], v[4:7], off nt                // 000000001A5C: DC7E8000 007F0400
	v_or_b32_e32 v0, 64, v18                                   // 000000001A64: 280024C0
	v_mov_b32_e32 v1, v19                                      // 000000001A68: 7E020313
	v_lshl_add_u64 v[0:1], v[0:1], 2, s[4:5]                   // 000000001A6C: D2080000 00110500
	v_or_b32_e32 v18, 0x60, v18                                // 000000001A74: 282424FF 00000060
	global_store_dwordx4 v[0:1], v[8:11], off nt               // 000000001A7C: DC7E8000 007F0800
	v_lshl_add_u64 v[0:1], v[18:19], 2, s[4:5]                 // 000000001A84: D2080000 00110512
	global_store_dwordx4 v[0:1], v[12:15], off nt              // 000000001A8C: DC7E8000 007F0C00
	s_endpgm                                                   // 000000001A94: BF810000
	s_nop 0                                                    // 000000001A98: BF800000
	s_nop 0                                                    // 000000001A9C: BF800000
	s_nop 0                                                    // 000000001AA0: BF800000
	s_nop 0                                                    // 000000001AA4: BF800000
	s_nop 0                                                    // 000000001AA8: BF800000
	s_nop 0                                                    // 000000001AAC: BF800000
	s_nop 0                                                    // 000000001AB0: BF800000
	s_nop 0                                                    // 000000001AB4: BF800000
	s_nop 0                                                    // 000000001AB8: BF800000
	s_nop 0                                                    // 000000001ABC: BF800000
	s_nop 0                                                    // 000000001AC0: BF800000
	s_nop 0                                                    // 000000001AC4: BF800000
	s_nop 0                                                    // 000000001AC8: BF800000
	s_nop 0                                                    // 000000001ACC: BF800000
	s_nop 0                                                    // 000000001AD0: BF800000
	s_nop 0                                                    // 000000001AD4: BF800000
	s_nop 0                                                    // 000000001AD8: BF800000
	s_nop 0                                                    // 000000001ADC: BF800000
	s_nop 0                                                    // 000000001AE0: BF800000
	s_nop 0                                                    // 000000001AE4: BF800000
	s_nop 0                                                    // 000000001AE8: BF800000
	s_nop 0                                                    // 000000001AEC: BF800000
	s_nop 0                                                    // 000000001AF0: BF800000
	s_nop 0                                                    // 000000001AF4: BF800000
	s_nop 0                                                    // 000000001AF8: BF800000
	s_nop 0                                                    // 000000001AFC: BF800000
	s_nop 0                                                    // 000000001B00: BF800000
	s_nop 0                                                    // 000000001B04: BF800000
	s_nop 0                                                    // 000000001B08: BF800000
	s_nop 0                                                    // 000000001B0C: BF800000
	s_nop 0                                                    // 000000001B10: BF800000
	s_nop 0                                                    // 000000001B14: BF800000
	s_nop 0                                                    // 000000001B18: BF800000
	s_nop 0                                                    // 000000001B1C: BF800000
	s_nop 0                                                    // 000000001B20: BF800000
	s_nop 0                                                    // 000000001B24: BF800000
	s_nop 0                                                    // 000000001B28: BF800000
	s_nop 0                                                    // 000000001B2C: BF800000
	s_nop 0                                                    // 000000001B30: BF800000
	s_nop 0                                                    // 000000001B34: BF800000
	s_nop 0                                                    // 000000001B38: BF800000
	s_nop 0                                                    // 000000001B3C: BF800000
	s_nop 0                                                    // 000000001B40: BF800000
	s_nop 0                                                    // 000000001B44: BF800000
	s_nop 0                                                    // 000000001B48: BF800000
	s_nop 0                                                    // 000000001B4C: BF800000
	s_nop 0                                                    // 000000001B50: BF800000
	s_nop 0                                                    // 000000001B54: BF800000
	s_nop 0                                                    // 000000001B58: BF800000
	s_nop 0                                                    // 000000001B5C: BF800000
	s_nop 0                                                    // 000000001B60: BF800000
	s_nop 0                                                    // 000000001B64: BF800000
	s_nop 0                                                    // 000000001B68: BF800000
	s_nop 0                                                    // 000000001B6C: BF800000
	s_nop 0                                                    // 000000001B70: BF800000
	s_nop 0                                                    // 000000001B74: BF800000
	s_nop 0                                                    // 000000001B78: BF800000
	s_nop 0                                                    // 000000001B7C: BF800000
	s_nop 0                                                    // 000000001B80: BF800000
	s_nop 0                                                    // 000000001B84: BF800000
	s_nop 0                                                    // 000000001B88: BF800000
	s_nop 0                                                    // 000000001B8C: BF800000
	s_nop 0                                                    // 000000001B90: BF800000
	s_nop 0                                                    // 000000001B94: BF800000
	s_nop 0                                                    // 000000001B98: BF800000
	s_nop 0                                                    // 000000001B9C: BF800000
	s_nop 0                                                    // 000000001BA0: BF800000
	s_nop 0                                                    // 000000001BA4: BF800000
	s_nop 0                                                    // 000000001BA8: BF800000
	s_nop 0                                                    // 000000001BAC: BF800000
	s_nop 0                                                    // 000000001BB0: BF800000
	s_nop 0                                                    // 000000001BB4: BF800000
	s_nop 0                                                    // 000000001BB8: BF800000
	s_nop 0                                                    // 000000001BBC: BF800000
	s_nop 0                                                    // 000000001BC0: BF800000
	s_nop 0                                                    // 000000001BC4: BF800000
	s_nop 0                                                    // 000000001BC8: BF800000
	s_nop 0                                                    // 000000001BCC: BF800000
	s_nop 0                                                    // 000000001BD0: BF800000
	s_nop 0                                                    // 000000001BD4: BF800000
	s_nop 0                                                    // 000000001BD8: BF800000
	s_nop 0                                                    // 000000001BDC: BF800000
	s_nop 0                                                    // 000000001BE0: BF800000
	s_nop 0                                                    // 000000001BE4: BF800000
	s_nop 0                                                    // 000000001BE8: BF800000
	s_nop 0                                                    // 000000001BEC: BF800000
	s_nop 0                                                    // 000000001BF0: BF800000
	s_nop 0                                                    // 000000001BF4: BF800000
	s_nop 0                                                    // 000000001BF8: BF800000
	s_nop 0                                                    // 000000001BFC: BF800000
	s_nop 0                                                    // 000000001C00: BF800000
	s_nop 0                                                    // 000000001C04: BF800000
	s_nop 0                                                    // 000000001C08: BF800000
	s_nop 0                                                    // 000000001C0C: BF800000
	s_nop 0                                                    // 000000001C10: BF800000
	s_nop 0                                                    // 000000001C14: BF800000
	s_nop 0                                                    // 000000001C18: BF800000
	s_nop 0                                                    // 000000001C1C: BF800000
	s_nop 0                                                    // 000000001C20: BF800000
	s_nop 0                                                    // 000000001C24: BF800000
	s_nop 0                                                    // 000000001C28: BF800000
	s_nop 0                                                    // 000000001C2C: BF800000
	s_nop 0                                                    // 000000001C30: BF800000
	s_nop 0                                                    // 000000001C34: BF800000
	s_nop 0                                                    // 000000001C38: BF800000
	s_nop 0                                                    // 000000001C3C: BF800000
	s_nop 0                                                    // 000000001C40: BF800000
	s_nop 0                                                    // 000000001C44: BF800000
	s_nop 0                                                    // 000000001C48: BF800000
	s_nop 0                                                    // 000000001C4C: BF800000
	s_nop 0                                                    // 000000001C50: BF800000
	s_nop 0                                                    // 000000001C54: BF800000
	s_nop 0                                                    // 000000001C58: BF800000
	s_nop 0                                                    // 000000001C5C: BF800000
	s_nop 0                                                    // 000000001C60: BF800000
	s_nop 0                                                    // 000000001C64: BF800000
	s_nop 0                                                    // 000000001C68: BF800000
	s_nop 0                                                    // 000000001C6C: BF800000
	s_nop 0                                                    // 000000001C70: BF800000
	s_nop 0                                                    // 000000001C74: BF800000
	s_nop 0                                                    // 000000001C78: BF800000
	s_nop 0                                                    // 000000001C7C: BF800000
	s_nop 0                                                    // 000000001C80: BF800000
	s_nop 0                                                    // 000000001C84: BF800000
	s_nop 0                                                    // 000000001C88: BF800000
	s_nop 0                                                    // 000000001C8C: BF800000
	s_nop 0                                                    // 000000001C90: BF800000
	s_nop 0                                                    // 000000001C94: BF800000
	s_nop 0                                                    // 000000001C98: BF800000
	s_nop 0                                                    // 000000001C9C: BF800000
	s_nop 0                                                    // 000000001CA0: BF800000
	s_nop 0                                                    // 000000001CA4: BF800000
	s_nop 0                                                    // 000000001CA8: BF800000
	s_nop 0                                                    // 000000001CAC: BF800000
	s_nop 0                                                    // 000000001CB0: BF800000
	s_nop 0                                                    // 000000001CB4: BF800000
	s_nop 0                                                    // 000000001CB8: BF800000
	s_nop 0                                                    // 000000001CBC: BF800000
	s_nop 0                                                    // 000000001CC0: BF800000
	s_nop 0                                                    // 000000001CC4: BF800000
	s_nop 0                                                    // 000000001CC8: BF800000
	s_nop 0                                                    // 000000001CCC: BF800000
	s_nop 0                                                    // 000000001CD0: BF800000
	s_nop 0                                                    // 000000001CD4: BF800000
	s_nop 0                                                    // 000000001CD8: BF800000
	s_nop 0                                                    // 000000001CDC: BF800000
	s_nop 0                                                    // 000000001CE0: BF800000
	s_nop 0                                                    // 000000001CE4: BF800000
	s_nop 0                                                    // 000000001CE8: BF800000
	s_nop 0                                                    // 000000001CEC: BF800000
	s_nop 0                                                    // 000000001CF0: BF800000
	s_nop 0                                                    // 000000001CF4: BF800000
	s_nop 0                                                    // 000000001CF8: BF800000
	s_nop 0                                                    // 000000001CFC: BF800000
	s_nop 0                                                    // 000000001D00: BF800000
	s_nop 0                                                    // 000000001D04: BF800000
	s_nop 0                                                    // 000000001D08: BF800000
	s_nop 0                                                    // 000000001D0C: BF800000
	s_nop 0                                                    // 000000001D10: BF800000
	s_nop 0                                                    // 000000001D14: BF800000
	s_nop 0                                                    // 000000001D18: BF800000
	s_nop 0                                                    // 000000001D1C: BF800000
	s_nop 0                                                    // 000000001D20: BF800000
	s_nop 0                                                    // 000000001D24: BF800000
	s_nop 0                                                    // 000000001D28: BF800000
	s_nop 0                                                    // 000000001D2C: BF800000
	s_nop 0                                                    // 000000001D30: BF800000
	s_nop 0                                                    // 000000001D34: BF800000
	s_nop 0                                                    // 000000001D38: BF800000
	s_nop 0                                                    // 000000001D3C: BF800000
	s_nop 0                                                    // 000000001D40: BF800000
	s_nop 0                                                    // 000000001D44: BF800000
	s_nop 0                                                    // 000000001D48: BF800000
	s_nop 0                                                    // 000000001D4C: BF800000
	s_nop 0                                                    // 000000001D50: BF800000
	s_nop 0                                                    // 000000001D54: BF800000
	s_nop 0                                                    // 000000001D58: BF800000
	s_nop 0                                                    // 000000001D5C: BF800000
	s_nop 0                                                    // 000000001D60: BF800000
	s_nop 0                                                    // 000000001D64: BF800000
	s_nop 0                                                    // 000000001D68: BF800000
	s_nop 0                                                    // 000000001D6C: BF800000
	s_nop 0                                                    // 000000001D70: BF800000
	s_nop 0                                                    // 000000001D74: BF800000
	s_nop 0                                                    // 000000001D78: BF800000
	s_nop 0                                                    // 000000001D7C: BF800000
	s_nop 0                                                    // 000000001D80: BF800000
	s_nop 0                                                    // 000000001D84: BF800000
	s_nop 0                                                    // 000000001D88: BF800000
	s_nop 0                                                    // 000000001D8C: BF800000
	s_nop 0                                                    // 000000001D90: BF800000
	s_nop 0                                                    // 000000001D94: BF800000
	s_nop 0                                                    // 000000001D98: BF800000
	s_nop 0                                                    // 000000001D9C: BF800000
	s_nop 0                                                    // 000000001DA0: BF800000
	s_nop 0                                                    // 000000001DA4: BF800000
	s_nop 0                                                    // 000000001DA8: BF800000
	s_nop 0                                                    // 000000001DAC: BF800000
	s_nop 0                                                    // 000000001DB0: BF800000
	s_nop 0                                                    // 000000001DB4: BF800000
	s_nop 0                                                    // 000000001DB8: BF800000
	s_nop 0                                                    // 000000001DBC: BF800000
	s_nop 0                                                    // 000000001DC0: BF800000
	s_nop 0                                                    // 000000001DC4: BF800000
	s_nop 0                                                    // 000000001DC8: BF800000
	s_nop 0                                                    // 000000001DCC: BF800000
	s_nop 0                                                    // 000000001DD0: BF800000
	s_nop 0                                                    // 000000001DD4: BF800000
	s_nop 0                                                    // 000000001DD8: BF800000
	s_nop 0                                                    // 000000001DDC: BF800000
	s_nop 0                                                    // 000000001DE0: BF800000
	s_nop 0                                                    // 000000001DE4: BF800000
	s_nop 0                                                    // 000000001DE8: BF800000
	s_nop 0                                                    // 000000001DEC: BF800000
	s_nop 0                                                    // 000000001DF0: BF800000
	s_nop 0                                                    // 000000001DF4: BF800000
	s_nop 0                                                    // 000000001DF8: BF800000
	s_nop 0                                                    // 000000001DFC: BF800000
	s_nop 0                                                    // 000000001E00: BF800000
	s_nop 0                                                    // 000000001E04: BF800000
	s_nop 0                                                    // 000000001E08: BF800000
	s_nop 0                                                    // 000000001E0C: BF800000
	s_nop 0                                                    // 000000001E10: BF800000
	s_nop 0                                                    // 000000001E14: BF800000
	s_nop 0                                                    // 000000001E18: BF800000
	s_nop 0                                                    // 000000001E1C: BF800000
	s_nop 0                                                    // 000000001E20: BF800000
	s_nop 0                                                    // 000000001E24: BF800000
	s_nop 0                                                    // 000000001E28: BF800000
	s_nop 0                                                    // 000000001E2C: BF800000
	s_nop 0                                                    // 000000001E30: BF800000
	s_nop 0                                                    // 000000001E34: BF800000
	s_nop 0                                                    // 000000001E38: BF800000
	s_nop 0                                                    // 000000001E3C: BF800000
	s_nop 0                                                    // 000000001E40: BF800000
	s_nop 0                                                    // 000000001E44: BF800000
	s_nop 0                                                    // 000000001E48: BF800000
	s_nop 0                                                    // 000000001E4C: BF800000
	s_nop 0                                                    // 000000001E50: BF800000
	s_nop 0                                                    // 000000001E54: BF800000
	s_nop 0                                                    // 000000001E58: BF800000
	s_nop 0                                                    // 000000001E5C: BF800000
	s_nop 0                                                    // 000000001E60: BF800000
	s_nop 0                                                    // 000000001E64: BF800000
	s_nop 0                                                    // 000000001E68: BF800000
	s_nop 0                                                    // 000000001E6C: BF800000
	s_nop 0                                                    // 000000001E70: BF800000
	s_nop 0                                                    // 000000001E74: BF800000
	s_nop 0                                                    // 000000001E78: BF800000
	s_nop 0                                                    // 000000001E7C: BF800000
	s_nop 0                                                    // 000000001E80: BF800000
	s_nop 0                                                    // 000000001E84: BF800000
	s_nop 0                                                    // 000000001E88: BF800000
	s_nop 0                                                    // 000000001E8C: BF800000
	s_nop 0                                                    // 000000001E90: BF800000
	s_nop 0                                                    // 000000001E94: BF800000
	s_nop 0                                                    // 000000001E98: BF800000
	s_nop 0                                                    // 000000001E9C: BF800000
	s_nop 0                                                    // 000000001EA0: BF800000
	s_nop 0                                                    // 000000001EA4: BF800000
	s_nop 0                                                    // 000000001EA8: BF800000
	s_nop 0                                                    // 000000001EAC: BF800000
	s_nop 0                                                    // 000000001EB0: BF800000
	s_nop 0                                                    // 000000001EB4: BF800000
	s_nop 0                                                    // 000000001EB8: BF800000
	s_nop 0                                                    // 000000001EBC: BF800000
.Lrocke_mxfp6_pack_scales_4096_0x65536_16x16x128_xcd_nts_end:
.size rocke_mxfp6_pack_scales_4096_0x65536_16x16x128_xcd_nts, .Lrocke_mxfp6_pack_scales_4096_0x65536_16x16x128_xcd_nts_end-rocke_mxfp6_pack_scales_4096_0x65536_16x16x128_xcd_nts

.rodata
.p2align 6
.amdhsa_kernel rocke_mxfp6_pack_scales_4096_0x65536_16x16x128_xcd_nts
  .amdhsa_group_segment_fixed_size 18432
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_kernarg_size 32
  .amdhsa_user_sgpr_count 2
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_next_free_vgpr 32
  .amdhsa_next_free_sgpr 10
  .amdhsa_reserve_vcc 0
  .amdhsa_accum_offset 32
  .amdhsa_float_denorm_mode_32 3
  .amdhsa_float_denorm_mode_16_64 3
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.target: amdgcn-amd-amdhsa--gfx950
amdhsa.version: [ 1, 1 ]
amdhsa.kernels:
  - .name: rocke_mxfp6_pack_scales_4096_0x65536_16x16x128_xcd_nts
    .symbol: rocke_mxfp6_pack_scales_4096_0x65536_16x16x128_xcd_nts.kd
    .args:
      - .address_space:  global
        .name:           sa_src
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .name:           sb_src
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .name:           sa_dst
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .name:           sb_dst
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 18432
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .max_flat_workgroup_size: 256
    .private_segment_fixed_size: 0
    .sgpr_count: 16
    .sgpr_spill_count: 0
    .uniform_work_group_size: 1
    .vgpr_count: 32
    .vgpr_spill_count: 0
    .agpr_count: 0
    .wavefront_size: 64
...
.end_amdgpu_metadata
