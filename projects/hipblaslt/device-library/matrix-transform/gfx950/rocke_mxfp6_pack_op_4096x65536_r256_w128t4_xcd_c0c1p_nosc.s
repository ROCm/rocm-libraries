// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// MXFP6 operand packer for gfx950, launched by hipblasLtMatrixTransform. It copies an FP6
// operand (HIP_R_6F_E2M3 or HIP_R_6F_E3M2: it moves whole bytes and never reads the format)
// from HIPBLASLT_ORDER_COL to HIPBLASLT_ORDER_COL256_R128_P16_EXT. 256 threads (wave64),
// 51200 bytes of static LDS.
//
// Fixed problem, enforced by the caller (rocblaslt_matrix_transform):
//   4096 columns (M or N indices) of 65536 rows (K), ld = 65536, batch 1, so the source and
//   the destination are 201326592 bytes each. The grid must be exactly 4096 workgroups: no
//   load or store is bounds-checked, so any other shape reads and writes out of range.
//
// Arguments: codes_src, scales_src, codes_dst, scales_dst, four 8-byte global pointers. The
// kernel reads only codes_src (offset 0) and codes_dst (offset 16); the scale pointers are
// never loaded.
//
// Edits to the provider's instruction listing (llvm-objdump output), none of which changes an
// instruction:
//   - the objdump banner is replaced by the directives and the symbol label below;
//   - every instruction line, with its address and encoding comment, and the s_nop padding
//     after s_endpgm are kept as listed, so .text matches the provider's code object byte for
//     byte;
//   - the kernel descriptor and metadata are written for code object v4 (amdhsa.version
//     [1, 1], no hidden arguments) from the provider's kernel descriptor, including its
//     register allocation (next_free_vgpr 136, accum_offset 80, next_free_sgpr 98; the assembler adds the 6 SGPRs that
//     gfx950 reserves for flat scratch).

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.amdhsa_code_object_version 4
.text
.globl rocke_mxfp6_pack_op_4096x65536_r256_w128t4_xcd_c0c1p_nosc
.p2align 8
.type rocke_mxfp6_pack_op_4096x65536_r256_w128t4_xcd_c0c1p_nosc,@function
rocke_mxfp6_pack_op_4096x65536_r256_w128t4_xcd_c0c1p_nosc:
	s_load_dwordx2 s[6:7], s[0:1], 0x0                         // 000000001700: C0060180 00000000
	s_load_dwordx2 s[4:5], s[0:1], 0x10                        // 000000001708: C0060100 00000010
	s_lshl_b32 s0, s2, 9                                       // 000000001710: 8E008902
	s_and_b32 s3, s0, 0xe00                                    // 000000001714: 8603FF00 00000E00
	s_lshr_b32 s0, s2, 3                                       // 00000000171C: 8F008302
	s_add_i32 s3, s3, s0                                       // 000000001720: 81030003
	v_mul_u32_u24_e32 v1, 0xaab, v0                            // 000000001724: 100200FF 00000AAB
	s_and_b32 s0, s3, 0x3fffff80                               // 00000000172C: 8600FF03 3FFFFF80
	v_lshrrev_b32_e32 v1, 16, v1                               // 000000001734: 20020290
	s_movk_i32 s8, 0xffe8                                      // 000000001738: B008FFE8
	s_bfe_u32 s1, s2, 0x70003                                  // 00000000173C: 9281FF02 00070003
	v_mad_i32_i24 v52, v1, s8, v0                              // 000000001744: D1C20034 04001101
	v_or_b32_e32 v2, s0, v1                                    // 00000000174C: 28040200
	s_movk_i32 s9, 0x3000                                      // 000000001750: B0093000
	v_mul_lo_u32 v2, v2, s9                                    // 000000001754: D2850002 00001302
	s_mul_i32 s10, s1, 0x60                                    // 00000000175C: 920AFF01 00000060
	v_lshlrev_b32_e32 v3, 2, v52                               // 000000001764: 24066882
	v_add3_u32 v2, v2, s10, v3                                 // 000000001768: D1FF0002 040C1502
	v_ashrrev_i32_e32 v3, 31, v2                               // 000000001770: 2206049F
	v_or_b32_e32 v53, 0x100, v0                                // 000000001774: 286A00FF 00000100
	s_waitcnt lgkmcnt(0)                                       // 00000000177C: BF8CC07F
	v_lshl_add_u64 v[12:13], v[2:3], 2, s[6:7]                 // 000000001780: D208000C 00190502
	v_mul_u32_u24_e32 v2, 0xaab, v53                           // 000000001788: 10046AFF 00000AAB
	v_lshrrev_b32_e32 v54, 16, v2                              // 000000001790: 206C0490
	v_mad_i32_i24 v55, v54, s8, v53                            // 000000001794: D1C20037 04D41136
	v_or_b32_e32 v2, s0, v54                                   // 00000000179C: 28046C00
	v_mul_lo_u32 v2, v2, s9                                    // 0000000017A0: D2850002 00001302
	v_lshlrev_b32_e32 v3, 2, v55                               // 0000000017A8: 24066E82
	v_add3_u32 v2, v2, s10, v3                                 // 0000000017AC: D1FF0002 040C1502
	v_mov_b32_e32 v3, 0                                        // 0000000017B4: 7E060280
	v_lshl_add_u64 v[14:15], v[2:3], 2, s[6:7]                 // 0000000017B8: D208000E 00190502
	v_or_b32_e32 v2, 0x200, v0                                 // 0000000017C0: 280400FF 00000200
	global_load_dwordx4 v[4:7], v[12:13], off                  // 0000000017C8: DC5C8000 047F000C
	global_load_dwordx4 v[8:11], v[14:15], off                 // 0000000017D0: DC5C8000 087F000E
	v_mul_u32_u24_e32 v12, 0xaab, v2                           // 0000000017D8: 101804FF 00000AAB
	v_lshrrev_b32_e32 v56, 16, v12                             // 0000000017E0: 20701890
	v_mad_i32_i24 v57, v56, s8, v2                             // 0000000017E4: D1C20039 04081138
	v_or_b32_e32 v2, s0, v56                                   // 0000000017EC: 28047000
	v_mul_lo_u32 v2, v2, s9                                    // 0000000017F0: D2850002 00001302
	v_lshlrev_b32_e32 v12, 2, v57                              // 0000000017F8: 24187282
	v_add3_u32 v2, v2, s10, v12                                // 0000000017FC: D1FF0002 04301502
	v_or_b32_e32 v58, 0x300, v0                                // 000000001804: 287400FF 00000300
	v_lshl_add_u64 v[20:21], v[2:3], 2, s[6:7]                 // 00000000180C: D2080014 00190502
	v_mul_u32_u24_e32 v2, 0xaab, v58                           // 000000001814: 100474FF 00000AAB
	v_lshrrev_b32_e32 v59, 16, v2                              // 00000000181C: 20760490
	v_mad_i32_i24 v60, v59, s8, v58                            // 000000001820: D1C2003C 04E8113B
	v_or_b32_e32 v2, s0, v59                                   // 000000001828: 28047600
	v_mul_lo_u32 v2, v2, s9                                    // 00000000182C: D2850002 00001302
	v_lshlrev_b32_e32 v12, 2, v60                              // 000000001834: 24187882
	v_add3_u32 v2, v2, s10, v12                                // 000000001838: D1FF0002 04301502
	v_lshl_add_u64 v[22:23], v[2:3], 2, s[6:7]                 // 000000001840: D2080016 00190502
	v_or_b32_e32 v2, 0x400, v0                                 // 000000001848: 280400FF 00000400
	global_load_dwordx4 v[12:15], v[20:21], off                // 000000001850: DC5C8000 0C7F0014
	global_load_dwordx4 v[16:19], v[22:23], off                // 000000001858: DC5C8000 107F0016
	v_mul_u32_u24_e32 v20, 0xaab, v2                           // 000000001860: 102804FF 00000AAB
	v_lshrrev_b32_e32 v61, 16, v20                             // 000000001868: 207A2890
	v_mad_i32_i24 v62, v61, s8, v2                             // 00000000186C: D1C2003E 0408113D
	v_or_b32_e32 v2, s0, v61                                   // 000000001874: 28047A00
	v_mul_lo_u32 v2, v2, s9                                    // 000000001878: D2850002 00001302
	v_lshlrev_b32_e32 v20, 2, v62                              // 000000001880: 24287C82
	v_add3_u32 v2, v2, s10, v20                                // 000000001884: D1FF0002 04501502
	v_or_b32_e32 v63, 0x500, v0                                // 00000000188C: 287E00FF 00000500
	v_lshl_add_u64 v[28:29], v[2:3], 2, s[6:7]                 // 000000001894: D208001C 00190502
	v_mul_u32_u24_e32 v2, 0xaab, v63                           // 00000000189C: 10047EFF 00000AAB
	v_lshrrev_b32_e32 v64, 16, v2                              // 0000000018A4: 20800490
	v_mad_i32_i24 v65, v64, s8, v63                            // 0000000018A8: D1C20041 04FC1140
	v_or_b32_e32 v2, s0, v64                                   // 0000000018B0: 28048000
	v_mul_lo_u32 v2, v2, s9                                    // 0000000018B4: D2850002 00001302
	v_lshlrev_b32_e32 v20, 2, v65                              // 0000000018BC: 24288282
	v_add3_u32 v2, v2, s10, v20                                // 0000000018C0: D1FF0002 04501502
	v_lshl_add_u64 v[30:31], v[2:3], 2, s[6:7]                 // 0000000018C8: D208001E 00190502
	v_or_b32_e32 v2, 0x600, v0                                 // 0000000018D0: 280400FF 00000600
	global_load_dwordx4 v[20:23], v[28:29], off                // 0000000018D8: DC5C8000 147F001C
	global_load_dwordx4 v[24:27], v[30:31], off                // 0000000018E0: DC5C8000 187F001E
	v_mul_u32_u24_e32 v28, 0xaab, v2                           // 0000000018E8: 103804FF 00000AAB
	v_lshrrev_b32_e32 v66, 16, v28                             // 0000000018F0: 20843890
	v_mad_i32_i24 v67, v66, s8, v2                             // 0000000018F4: D1C20043 04081142
	v_or_b32_e32 v2, s0, v66                                   // 0000000018FC: 28048400
	v_mul_lo_u32 v2, v2, s9                                    // 000000001900: D2850002 00001302
	v_lshlrev_b32_e32 v28, 2, v67                              // 000000001908: 24388682
	v_add3_u32 v2, v2, s10, v28                                // 00000000190C: D1FF0002 04701502
	v_or_b32_e32 v68, 0x700, v0                                // 000000001914: 288800FF 00000700
	v_lshl_add_u64 v[36:37], v[2:3], 2, s[6:7]                 // 00000000191C: D2080024 00190502
	v_mul_u32_u24_e32 v2, 0xaab, v68                           // 000000001924: 100488FF 00000AAB
	v_lshrrev_b32_e32 v69, 16, v2                              // 00000000192C: 208A0490
	v_mad_i32_i24 v70, v69, s8, v68                            // 000000001930: D1C20046 05101145
	v_or_b32_e32 v2, s0, v69                                   // 000000001938: 28048A00
	v_mul_lo_u32 v2, v2, s9                                    // 00000000193C: D2850002 00001302
	v_lshlrev_b32_e32 v28, 2, v70                              // 000000001944: 24388C82
	v_add3_u32 v2, v2, s10, v28                                // 000000001948: D1FF0002 04701502
	v_lshl_add_u64 v[38:39], v[2:3], 2, s[6:7]                 // 000000001950: D2080026 00190502
	v_or_b32_e32 v2, 0x800, v0                                 // 000000001958: 280400FF 00000800
	global_load_dwordx4 v[28:31], v[36:37], off                // 000000001960: DC5C8000 1C7F0024
	global_load_dwordx4 v[32:35], v[38:39], off                // 000000001968: DC5C8000 207F0026
	v_mul_u32_u24_e32 v36, 0xaab, v2                           // 000000001970: 104804FF 00000AAB
	v_lshrrev_b32_e32 v71, 16, v36                             // 000000001978: 208E4890
	v_mad_i32_i24 v72, v71, s8, v2                             // 00000000197C: D1C20048 04081147
	v_or_b32_e32 v2, s0, v71                                   // 000000001984: 28048E00
	v_mul_lo_u32 v2, v2, s9                                    // 000000001988: D2850002 00001302
	v_lshlrev_b32_e32 v36, 2, v72                              // 000000001990: 24489082
	v_add3_u32 v2, v2, s10, v36                                // 000000001994: D1FF0002 04901502
	v_lshl_add_u64 v[44:45], v[2:3], 2, s[6:7]                 // 00000000199C: D208002C 00190502
	v_or_b32_e32 v2, 0x900, v0                                 // 0000000019A4: 280400FF 00000900
	v_mul_u32_u24_e32 v36, 0xaab, v2                           // 0000000019AC: 104804FF 00000AAB
	v_lshrrev_b32_e32 v73, 16, v36                             // 0000000019B4: 20924890
	v_mad_i32_i24 v74, v73, s8, v2                             // 0000000019B8: D1C2004A 04081149
	v_or_b32_e32 v2, s0, v73                                   // 0000000019C0: 28049200
	v_mul_lo_u32 v2, v2, s9                                    // 0000000019C4: D2850002 00001302
	v_lshlrev_b32_e32 v36, 2, v74                              // 0000000019CC: 24489482
	v_add3_u32 v2, v2, s10, v36                                // 0000000019D0: D1FF0002 04901502
	v_lshl_add_u64 v[46:47], v[2:3], 2, s[6:7]                 // 0000000019D8: D208002E 00190502
	v_or_b32_e32 v2, 0xa00, v0                                 // 0000000019E0: 280400FF 00000A00
	global_load_dwordx4 v[36:39], v[44:45], off                // 0000000019E8: DC5C8000 247F002C
	global_load_dwordx4 v[40:43], v[46:47], off                // 0000000019F0: DC5C8000 287F002E
	v_mul_u32_u24_e32 v44, 0xaab, v2                           // 0000000019F8: 105804FF 00000AAB
	v_lshrrev_b32_e32 v75, 16, v44                             // 000000001A00: 20965890
	v_mad_i32_i24 v76, v75, s8, v2                             // 000000001A04: D1C2004C 0408114B
	v_or_b32_e32 v2, s0, v75                                   // 000000001A0C: 28049600
	v_mul_lo_u32 v2, v2, s9                                    // 000000001A10: D2850002 00001302
	v_lshlrev_b32_e32 v44, 2, v76                              // 000000001A18: 24589882
	v_add3_u32 v2, v2, s10, v44                                // 000000001A1C: D1FF0002 04B01502
	v_lshl_add_u64 v[44:45], v[2:3], 2, s[6:7]                 // 000000001A24: D208002C 00190502
	v_or_b32_e32 v2, 0xb00, v0                                 // 000000001A2C: 280400FF 00000B00
	v_mul_u32_u24_e32 v48, 0xaab, v2                           // 000000001A34: 106004FF 00000AAB
	v_lshrrev_b32_e32 v77, 16, v48                             // 000000001A3C: 209A6090
	v_mad_i32_i24 v78, v77, s8, v2                             // 000000001A40: D1C2004E 0408114D
	v_or_b32_e32 v2, s0, v77                                   // 000000001A48: 28049A00
	v_mul_lo_u32 v2, v2, s9                                    // 000000001A4C: D2850002 00001302
	v_lshlrev_b32_e32 v48, 2, v78                              // 000000001A54: 24609C82
	global_load_dwordx4 v[44:47], v[44:45], off                // 000000001A58: DC5C8000 2C7F002C
	v_add3_u32 v2, v2, s10, v48                                // 000000001A60: D1FF0002 04C01502
	v_lshl_add_u64 v[48:49], v[2:3], 2, s[6:7]                 // 000000001A68: D2080030 00190502
	global_load_dwordx4 v[48:51], v[48:49], off                // 000000001A70: DC5C8000 307F0030
	v_lshlrev_b32_e32 v2, 4, v52                               // 000000001A78: 24046884
	s_movk_i32 s6, 0x190                                       // 000000001A7C: B0060190
	v_mad_u32_u24 v1, v1, s6, v2                               // 000000001A80: D1C30001 04080D01
	s_waitcnt vmcnt(11)                                        // 000000001A88: BF8C0F7B
	ds_write_b128 v1, v[4:7]                                   // 000000001A8C: D9BE0000 00000401
	v_lshlrev_b32_e32 v1, 4, v55                               // 000000001A94: 24026E84
	v_mad_u32_u24 v1, v54, s6, v1                              // 000000001A98: D1C30001 04040D36
	s_waitcnt vmcnt(10)                                        // 000000001AA0: BF8C0F7A
	ds_write_b128 v1, v[8:11]                                  // 000000001AA4: D9BE0000 00000801
	v_lshlrev_b32_e32 v1, 4, v57                               // 000000001AAC: 24027284
	v_mad_u32_u24 v1, v56, s6, v1                              // 000000001AB0: D1C30001 04040D38
	s_waitcnt vmcnt(9)                                         // 000000001AB8: BF8C0F79
	ds_write_b128 v1, v[12:15]                                 // 000000001ABC: D9BE0000 00000C01
	v_lshlrev_b32_e32 v1, 4, v60                               // 000000001AC4: 24027884
	v_mad_u32_u24 v1, v59, s6, v1                              // 000000001AC8: D1C30001 04040D3B
	s_waitcnt vmcnt(8)                                         // 000000001AD0: BF8C0F78
	ds_write_b128 v1, v[16:19]                                 // 000000001AD4: D9BE0000 00001001
	v_lshlrev_b32_e32 v1, 4, v62                               // 000000001ADC: 24027C84
	v_mad_u32_u24 v1, v61, s6, v1                              // 000000001AE0: D1C30001 04040D3D
	v_bfe_u32 v2, v0, 4, 2                                     // 000000001AE8: D1C80002 02090900
	v_and_b32_e32 v14, 15, v0                                  // 000000001AF0: 261C008F
	v_lshrrev_b32_e32 v4, 2, v0                                // 000000001AF4: 20080082
	v_mul_u32_u24_e32 v2, 6, v2                                // 000000001AF8: 10040486
	v_and_or_b32 v4, v4, 48, v14                               // 000000001AFC: D2010004 04396104
	v_lshlrev_b32_e32 v15, 2, v2                               // 000000001B04: 241E0482
	s_waitcnt vmcnt(7)                                         // 000000001B08: BF8C0F77
	ds_write_b128 v1, v[20:23]                                 // 000000001B0C: D9BE0000 00001401
	v_lshlrev_b32_e32 v1, 4, v65                               // 000000001B14: 24028284
	v_mad_u32_u24 v1, v64, s6, v1                              // 000000001B18: D1C30001 04040D40
	s_waitcnt vmcnt(6)                                         // 000000001B20: BF8C0F76
	ds_write_b128 v1, v[24:27]                                 // 000000001B24: D9BE0000 00001801
	v_lshlrev_b32_e32 v1, 4, v67                               // 000000001B2C: 24028684
	v_mad_u32_u24 v1, v66, s6, v1                              // 000000001B30: D1C30001 04040D42
	s_lshr_b32 s2, s2, 1                                       // 000000001B38: 8F028102
	v_mad_u32_u24 v16, v4, s6, v15                             // 000000001B3C: D1C30010 043C0D04
	s_and_b32 s2, s2, 0x200                                    // 000000001B44: 8602FF02 00000200
	s_lshl_b32 s3, s3, 1                                       // 000000001B4C: 8E038103
	s_and_b32 s3, s3, 0x7ffffe00                               // 000000001B50: 8603FF03 7FFFFE00
	s_lshl_b32 s1, s1, 2                                       // 000000001B58: 8E018201
	s_or_b32 s1, s3, s1                                        // 000000001B5C: 87010103
	v_bfe_u32 v17, v0, 6, 1                                    // 000000001B60: D1C80011 02050D00
	s_mul_i32 s3, s1, 0x1800                                   // 000000001B68: 9203FF01 00001800
	v_cmp_eq_u32_e32 vcc, 0, v17                               // 000000001B70: 7D942280
	s_movk_i32 s8, 0x70                                        // 000000001B74: B0080070
	s_movk_i32 s7, 0x180                                       // 000000001B78: B0070180
	s_movk_i32 s0, 0x300                                       // 000000001B7C: B0000300
	s_waitcnt vmcnt(5)                                         // 000000001B80: BF8C0F75
	ds_write_b128 v1, v[28:31]                                 // 000000001B84: D9BE0000 00001C01
	v_lshlrev_b32_e32 v1, 4, v70                               // 000000001B8C: 24028C84
	v_mad_u32_u24 v1, v69, s6, v1                              // 000000001B90: D1C30001 04040D45
	s_waitcnt vmcnt(4)                                         // 000000001B98: BF8C0F74
	ds_write_b128 v1, v[32:35]                                 // 000000001B9C: D9BE0000 00002001
	v_lshlrev_b32_e32 v1, 4, v72                               // 000000001BA4: 24029084
	v_mad_u32_u24 v1, v71, s6, v1                              // 000000001BA8: D1C30001 04040D47
	s_waitcnt vmcnt(3)                                         // 000000001BB0: BF8C0F73
	ds_write_b128 v1, v[36:39]                                 // 000000001BB4: D9BE0000 00002401
	v_lshlrev_b32_e32 v1, 4, v74                               // 000000001BBC: 24029484
	v_mad_u32_u24 v1, v73, s6, v1                              // 000000001BC0: D1C30001 04040D49
	s_waitcnt vmcnt(2)                                         // 000000001BC8: BF8C0F72
	ds_write_b128 v1, v[40:43]                                 // 000000001BCC: D9BE0000 00002801
	v_lshlrev_b32_e32 v1, 4, v76                               // 000000001BD4: 24029884
	v_mad_u32_u24 v1, v75, s6, v1                              // 000000001BD8: D1C30001 04040D4B
	s_waitcnt vmcnt(1)                                         // 000000001BE0: BF8C0F71
	ds_write_b128 v1, v[44:47]                                 // 000000001BE4: D9BE0000 00002C01
	v_lshlrev_b32_e32 v1, 4, v78                               // 000000001BEC: 24029C84
	v_mad_u32_u24 v1, v77, s6, v1                              // 000000001BF0: D1C30001 04040D4D
	s_waitcnt vmcnt(0)                                         // 000000001BF8: BF8C0F70
	ds_write_b128 v1, v[48:51]                                 // 000000001BFC: D9BE0000 00003001
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 000000001C04: BF8C0070
	s_barrier                                                  // 000000001C08: BF8A0000
	v_and_b32_e32 v1, 0x80, v0                                 // 000000001C0C: 260200FF 00000080
	ds_read2_b64 v[4:7], v16 offset1:1                         // 000000001C14: D8EE0100 04000010
	ds_read_b64 v[12:13], v16 offset:16                        // 000000001C1C: D8EC0010 0C000010
	v_or_b32_e32 v1, s2, v1                                    // 000000001C24: 28020202
	v_lshrrev_b32_e32 v1, 7, v1                                // 000000001C28: 20020287
	v_mul_u32_u24_e32 v2, 0x300, v1                            // 000000001C2C: 100402FF 00000300
	v_or_b32_e32 v18, s3, v2                                   // 000000001C34: 28240403
	v_lshlrev_b32_e32 v2, 2, v0                                // 000000001C38: 24040082
	s_waitcnt lgkmcnt(1)                                       // 000000001C3C: BF8CC17F
	v_cndmask_b32_e32 v8, v6, v4, vcc                          // 000000001C40: 00100906
	s_waitcnt lgkmcnt(0)                                       // 000000001C44: BF8CC07F
	v_cndmask_b32_e32 v10, v12, v6, vcc                        // 000000001C48: 00140D0C
	v_cndmask_b32_e32 v4, v4, v12, vcc                         // 000000001C4C: 00081904
	v_and_b32_e32 v12, 0xfc, v2                                // 000000001C50: 261804FF 000000FC
	v_lshl_or_b32 v19, v17, 9, v12                             // 000000001C58: D2000013 04311311
	v_add_u32_e32 v2, v18, v19                                 // 000000001C60: 68042712
	v_lshl_or_b32 v17, v17, 1, v12                             // 000000001C64: D2000011 04310311
	v_cndmask_b32_e32 v9, v7, v5, vcc                          // 000000001C6C: 00120B07
	v_cndmask_b32_e32 v11, v13, v7, vcc                        // 000000001C70: 00160F0D
	v_lshl_add_u64 v[6:7], v[2:3], 2, s[4:5]                   // 000000001C74: D2080006 00110502
	v_or_b32_e32 v2, v18, v17                                  // 000000001C7C: 28042312
	v_cndmask_b32_e32 v5, v5, v13, vcc                         // 000000001C80: 000A1B05
	global_store_dwordx4 v[6:7], v[8:11], off                  // 000000001C84: DC7C8000 007F0806
	v_lshl_add_u64 v[6:7], v[2:3], 2, s[4:5]                   // 000000001C8C: D2080006 00110502
	global_store_dwordx2 v[6:7], v[4:5], off offset:1024       // 000000001C94: DC748400 007F0406
	v_lshrrev_b32_e32 v4, 2, v53                               // 000000001C9C: 20086A82
	v_mov_b32_e32 v2, 0x100                                    // 000000001CA0: 7E0402FF 00000100
	v_and_or_b32 v4, v4, s8, v14                               // 000000001CA8: D2010004 04381104
	v_bitop3_b32 v2, v0, s7, v2 bitop3:0xc8                    // 000000001CB0: D2340102 1C080F00
	v_mad_u32_u24 v8, v4, s6, v15                              // 000000001CB8: D1C30008 043C0D04
	ds_read2_b64 v[4:7], v8 offset1:1                          // 000000001CC0: D8EE0100 04000008
	ds_read_b64 v[12:13], v8 offset:16                         // 000000001CC8: D8EC0010 0C000008
	v_or_b32_e32 v2, s2, v2                                    // 000000001CD0: 28040402
	v_lshrrev_b32_e32 v2, 7, v2                                // 000000001CD4: 20040487
	v_mul_u32_u24_e32 v2, 0x300, v2                            // 000000001CD8: 100404FF 00000300
	v_or_b32_e32 v18, s3, v2                                   // 000000001CE0: 28240403
	v_add_u32_e32 v2, v18, v19                                 // 000000001CE4: 68042712
	s_waitcnt lgkmcnt(1)                                       // 000000001CE8: BF8CC17F
	v_cndmask_b32_e32 v8, v6, v4, vcc                          // 000000001CEC: 00100906
	v_cndmask_b32_e32 v9, v7, v5, vcc                          // 000000001CF0: 00120B07
	s_waitcnt lgkmcnt(0)                                       // 000000001CF4: BF8CC07F
	v_cndmask_b32_e32 v10, v12, v6, vcc                        // 000000001CF8: 00140D0C
	v_cndmask_b32_e32 v11, v13, v7, vcc                        // 000000001CFC: 00160F0D
	v_lshl_add_u64 v[6:7], v[2:3], 2, s[4:5]                   // 000000001D00: D2080006 00110502
	v_or_b32_e32 v2, v18, v17                                  // 000000001D08: 28042312
	v_cndmask_b32_e32 v4, v4, v12, vcc                         // 000000001D0C: 00081904
	v_cndmask_b32_e32 v5, v5, v13, vcc                         // 000000001D10: 000A1B05
	global_store_dwordx4 v[6:7], v[8:11], off                  // 000000001D14: DC7C8000 007F0806
	v_lshl_add_u64 v[6:7], v[2:3], 2, s[4:5]                   // 000000001D1C: D2080006 00110502
	v_mov_b32_e32 v18, 0x60                                    // 000000001D24: 7E2402FF 00000060
	global_store_dwordx2 v[6:7], v[4:5], off offset:1024       // 000000001D2C: DC748400 007F0406
	v_mad_u32_u24 v2, 1, v18, v16                              // 000000001D34: D1C30002 04422481
	ds_read2_b64 v[4:7], v2 offset1:1                          // 000000001D3C: D8EE0100 04000002
	ds_read_b64 v[12:13], v2 offset:16                         // 000000001D44: D8EC0010 0C000002
	s_or_b32 s6, s1, 1                                         // 000000001D4C: 87068101
	s_mulk_i32 s6, 0x1800                                      // 000000001D50: B7861800
	v_mov_b32_e32 v20, s6                                      // 000000001D54: 7E280206
	v_mad_u32_u24 v21, v1, s0, v20                             // 000000001D58: D1C30015 04500101
	v_add_u32_e32 v2, v21, v19                                 // 000000001D60: 68042715
	s_waitcnt lgkmcnt(1)                                       // 000000001D64: BF8CC17F
	v_cndmask_b32_e32 v8, v6, v4, vcc                          // 000000001D68: 00100906
	v_cndmask_b32_e32 v9, v7, v5, vcc                          // 000000001D6C: 00120B07
	s_waitcnt lgkmcnt(0)                                       // 000000001D70: BF8CC07F
	v_cndmask_b32_e32 v10, v12, v6, vcc                        // 000000001D74: 00140D0C
	v_cndmask_b32_e32 v11, v13, v7, vcc                        // 000000001D78: 00160F0D
	v_lshl_add_u64 v[6:7], v[2:3], 2, s[4:5]                   // 000000001D7C: D2080006 00110502
	v_or_b32_e32 v2, v21, v17                                  // 000000001D84: 28042315
	v_cndmask_b32_e32 v4, v4, v12, vcc                         // 000000001D88: 00081904
	v_cndmask_b32_e32 v5, v5, v13, vcc                         // 000000001D8C: 000A1B05
	global_store_dwordx4 v[6:7], v[8:11], off                  // 000000001D90: DC7C8000 007F0806
	v_lshl_add_u64 v[6:7], v[2:3], 2, s[4:5]                   // 000000001D98: D2080006 00110502
	global_store_dwordx2 v[6:7], v[4:5], off offset:1024       // 000000001DA0: DC748400 007F0406
	v_lshrrev_b32_e32 v4, 2, v58                               // 000000001DA8: 20087482
	v_and_or_b32 v4, v4, s8, v14                               // 000000001DAC: D2010004 04381104
	v_mul_u32_u24_e32 v4, 0x190, v4                            // 000000001DB4: 100808FF 00000190
	v_mul_u32_u24_e32 v5, 1, v18                               // 000000001DBC: 100A2481
	v_mov_b32_e32 v2, 0x300                                    // 000000001DC0: 7E0402FF 00000300
	v_add3_u32 v8, v5, v4, v15                                 // 000000001DC8: D1FF0008 043E0905
	v_bitop3_b32 v2, v0, s7, v2 bitop3:0xc8                    // 000000001DD0: D2340102 1C080F00
	ds_read2_b64 v[4:7], v8 offset1:1                          // 000000001DD8: D8EE0100 04000008
	ds_read_b64 v[12:13], v8 offset:16                         // 000000001DE0: D8EC0010 0C000008
	v_or_b32_e32 v2, s2, v2                                    // 000000001DE8: 28040402
	v_lshrrev_b32_e32 v2, 7, v2                                // 000000001DEC: 20040487
	v_mad_u32_u24 v20, v2, s0, v20                             // 000000001DF0: D1C30014 04500102
	v_add_u32_e32 v2, v20, v19                                 // 000000001DF8: 68042714
	s_waitcnt lgkmcnt(1)                                       // 000000001DFC: BF8CC17F
	v_cndmask_b32_e32 v8, v6, v4, vcc                          // 000000001E00: 00100906
	v_cndmask_b32_e32 v9, v7, v5, vcc                          // 000000001E04: 00120B07
	s_waitcnt lgkmcnt(0)                                       // 000000001E08: BF8CC07F
	v_cndmask_b32_e32 v10, v12, v6, vcc                        // 000000001E0C: 00140D0C
	v_cndmask_b32_e32 v11, v13, v7, vcc                        // 000000001E10: 00160F0D
	v_lshl_add_u64 v[6:7], v[2:3], 2, s[4:5]                   // 000000001E14: D2080006 00110502
	v_or_b32_e32 v2, v20, v17                                  // 000000001E1C: 28042314
	v_cndmask_b32_e32 v4, v4, v12, vcc                         // 000000001E20: 00081904
	v_cndmask_b32_e32 v5, v5, v13, vcc                         // 000000001E24: 000A1B05
	global_store_dwordx4 v[6:7], v[8:11], off                  // 000000001E28: DC7C8000 007F0806
	v_lshl_add_u64 v[6:7], v[2:3], 2, s[4:5]                   // 000000001E30: D2080006 00110502
	global_store_dwordx2 v[6:7], v[4:5], off offset:1024       // 000000001E38: DC748400 007F0406
	v_mad_u32_u24 v2, 2, v18, v16                              // 000000001E40: D1C30002 04422482
	ds_read2_b64 v[4:7], v2 offset1:1                          // 000000001E48: D8EE0100 04000002
	ds_read_b64 v[12:13], v2 offset:16                         // 000000001E50: D8EC0010 0C000002
	s_or_b32 s1, s1, 2                                         // 000000001E58: 87018201
	s_mulk_i32 s1, 0x1800                                      // 000000001E5C: B7811800
	v_mov_b32_e32 v2, s1                                       // 000000001E60: 7E040201
	v_mad_u32_u24 v20, v1, s0, v2                              // 000000001E64: D1C30014 04080101
	v_add_u32_e32 v2, v20, v19                                 // 000000001E6C: 68042714
	s_waitcnt lgkmcnt(1)                                       // 000000001E70: BF8CC17F
	v_cndmask_b32_e32 v8, v6, v4, vcc                          // 000000001E74: 00100906
	v_cndmask_b32_e32 v9, v7, v5, vcc                          // 000000001E78: 00120B07
	s_waitcnt lgkmcnt(0)                                       // 000000001E7C: BF8CC07F
	v_cndmask_b32_e32 v10, v12, v6, vcc                        // 000000001E80: 00140D0C
	v_cndmask_b32_e32 v11, v13, v7, vcc                        // 000000001E84: 00160F0D
	v_lshl_add_u64 v[6:7], v[2:3], 2, s[4:5]                   // 000000001E88: D2080006 00110502
	v_or_b32_e32 v2, v20, v17                                  // 000000001E90: 28042314
	v_cndmask_b32_e32 v4, v4, v12, vcc                         // 000000001E94: 00081904
	v_cndmask_b32_e32 v5, v5, v13, vcc                         // 000000001E98: 000A1B05
	global_store_dwordx4 v[6:7], v[8:11], off                  // 000000001E9C: DC7C8000 007F0806
	v_lshl_add_u64 v[6:7], v[2:3], 2, s[4:5]                   // 000000001EA4: D2080006 00110502
	global_store_dwordx2 v[6:7], v[4:5], off offset:1024       // 000000001EAC: DC748400 007F0406
	v_lshrrev_b32_e32 v4, 2, v63                               // 000000001EB4: 20087E82
	v_and_or_b32 v4, v4, s8, v14                               // 000000001EB8: D2010004 04381104
	v_mul_u32_u24_e32 v4, 0x190, v4                            // 000000001EC0: 100808FF 00000190
	v_mul_u32_u24_e32 v5, 2, v18                               // 000000001EC8: 100A2482
	v_mov_b32_e32 v2, 0x500                                    // 000000001ECC: 7E0402FF 00000500
	v_add3_u32 v8, v5, v4, v15                                 // 000000001ED4: D1FF0008 043E0905
	v_bitop3_b32 v2, v0, s7, v2 bitop3:0xc8                    // 000000001EDC: D2340102 1C080F00
	ds_read2_b64 v[4:7], v8 offset1:1                          // 000000001EE4: D8EE0100 04000008
	ds_read_b64 v[12:13], v8 offset:16                         // 000000001EEC: D8EC0010 0C000008
	s_add_i32 s1, s3, 0x3000                                   // 000000001EF4: 8101FF03 00003000
	v_or_b32_e32 v2, s2, v2                                    // 000000001EFC: 28040402
	v_lshrrev_b32_e32 v2, 7, v2                                // 000000001F00: 20040487
	v_mov_b32_e32 v8, s1                                       // 000000001F04: 7E100201
	v_mad_u32_u24 v20, v2, s0, v8                              // 000000001F08: D1C30014 04200102
	v_add_u32_e32 v2, v20, v19                                 // 000000001F10: 68042714
	s_waitcnt lgkmcnt(1)                                       // 000000001F14: BF8CC17F
	v_cndmask_b32_e32 v8, v6, v4, vcc                          // 000000001F18: 00100906
	v_cndmask_b32_e32 v9, v7, v5, vcc                          // 000000001F1C: 00120B07
	s_waitcnt lgkmcnt(0)                                       // 000000001F20: BF8CC07F
	v_cndmask_b32_e32 v10, v12, v6, vcc                        // 000000001F24: 00140D0C
	v_cndmask_b32_e32 v11, v13, v7, vcc                        // 000000001F28: 00160F0D
	v_lshl_add_u64 v[6:7], v[2:3], 2, s[4:5]                   // 000000001F2C: D2080006 00110502
	v_or_b32_e32 v2, v20, v17                                  // 000000001F34: 28042314
	v_cndmask_b32_e32 v4, v4, v12, vcc                         // 000000001F38: 00081904
	v_cndmask_b32_e32 v5, v5, v13, vcc                         // 000000001F3C: 000A1B05
	global_store_dwordx4 v[6:7], v[8:11], off                  // 000000001F40: DC7C8000 007F0806
	v_lshl_add_u64 v[6:7], v[2:3], 2, s[4:5]                   // 000000001F48: D2080006 00110502
	global_store_dwordx2 v[6:7], v[4:5], off offset:1024       // 000000001F50: DC748400 007F0406
	v_mad_u32_u24 v2, 3, v18, v16                              // 000000001F58: D1C30002 04422483
	ds_read2_b64 v[4:7], v2 offset1:1                          // 000000001F60: D8EE0100 04000002
	ds_read_b64 v[12:13], v2 offset:16                         // 000000001F68: D8EC0010 0C000002
	s_add_i32 s1, s3, 0x4800                                   // 000000001F70: 8101FF03 00004800
	v_mov_b32_e32 v16, s1                                      // 000000001F78: 7E200201
	v_mad_u32_u24 v1, v1, s0, v16                              // 000000001F7C: D1C30001 04400101
	v_add_u32_e32 v2, v1, v19                                  // 000000001F84: 68042701
	s_waitcnt lgkmcnt(1)                                       // 000000001F88: BF8CC17F
	v_cndmask_b32_e32 v8, v6, v4, vcc                          // 000000001F8C: 00100906
	v_cndmask_b32_e32 v9, v7, v5, vcc                          // 000000001F90: 00120B07
	s_waitcnt lgkmcnt(0)                                       // 000000001F94: BF8CC07F
	v_cndmask_b32_e32 v10, v12, v6, vcc                        // 000000001F98: 00140D0C
	v_cndmask_b32_e32 v11, v13, v7, vcc                        // 000000001F9C: 00160F0D
	v_lshl_add_u64 v[6:7], v[2:3], 2, s[4:5]                   // 000000001FA0: D2080006 00110502
	v_or_b32_e32 v2, v1, v17                                   // 000000001FA8: 28042301
	v_mov_b32_e32 v1, 0x700                                    // 000000001FAC: 7E0202FF 00000700
	global_store_dwordx4 v[6:7], v[8:11], off                  // 000000001FB4: DC7C8000 007F0806
	v_lshl_add_u64 v[6:7], v[2:3], 2, s[4:5]                   // 000000001FBC: D2080006 00110502
	v_bitop3_b32 v2, v0, s7, v1 bitop3:0xc8                    // 000000001FC4: D2340102 1C040F00
	v_lshrrev_b32_e32 v0, 2, v68                               // 000000001FCC: 20008882
	v_and_or_b32 v0, v0, s8, v14                               // 000000001FD0: D2010000 04381100
	v_cndmask_b32_e32 v4, v4, v12, vcc                         // 000000001FD8: 00081904
	v_cndmask_b32_e32 v5, v5, v13, vcc                         // 000000001FDC: 000A1B05
	v_mul_u32_u24_e32 v0, 0x190, v0                            // 000000001FE0: 100000FF 00000190
	v_mul_u32_u24_e32 v1, 3, v18                               // 000000001FE8: 10022483
	global_store_dwordx2 v[6:7], v[4:5], off offset:1024       // 000000001FEC: DC748400 007F0406
	v_add3_u32 v0, v1, v0, v15                                 // 000000001FF4: D1FF0000 043E0101
	ds_read2_b64 v[4:7], v0 offset1:1                          // 000000001FFC: D8EE0100 04000000
	ds_read_b64 v[0:1], v0 offset:16                           // 000000002004: D8EC0010 00000000
	v_or_b32_e32 v2, s2, v2                                    // 00000000200C: 28040402
	v_lshrrev_b32_e32 v2, 7, v2                                // 000000002010: 20040487
	v_mad_u32_u24 v12, v2, s0, v16                             // 000000002014: D1C3000C 04400102
	v_add_u32_e32 v2, v12, v19                                 // 00000000201C: 6804270C
	s_waitcnt lgkmcnt(1)                                       // 000000002020: BF8CC17F
	v_cndmask_b32_e32 v8, v6, v4, vcc                          // 000000002024: 00100906
	v_cndmask_b32_e32 v9, v7, v5, vcc                          // 000000002028: 00120B07
	s_waitcnt lgkmcnt(0)                                       // 00000000202C: BF8CC07F
	v_cndmask_b32_e32 v10, v0, v6, vcc                         // 000000002030: 00140D00
	v_cndmask_b32_e32 v11, v1, v7, vcc                         // 000000002034: 00160F01
	v_cndmask_b32_e32 v0, v4, v0, vcc                          // 000000002038: 00000104
	v_cndmask_b32_e32 v1, v5, v1, vcc                          // 00000000203C: 00020305
	v_lshl_add_u64 v[4:5], v[2:3], 2, s[4:5]                   // 000000002040: D2080004 00110502
	v_or_b32_e32 v2, v12, v17                                  // 000000002048: 2804230C
	v_lshl_add_u64 v[2:3], v[2:3], 2, s[4:5]                   // 00000000204C: D2080002 00110502
	global_store_dwordx4 v[4:5], v[8:11], off                  // 000000002054: DC7C8000 007F0804
	global_store_dwordx2 v[2:3], v[0:1], off offset:1024       // 00000000205C: DC748400 007F0002
	s_endpgm                                                   // 000000002064: BF810000
	s_nop 0                                                    // 000000002068: BF800000
	s_nop 0                                                    // 00000000206C: BF800000
	s_nop 0                                                    // 000000002070: BF800000
	s_nop 0                                                    // 000000002074: BF800000
	s_nop 0                                                    // 000000002078: BF800000
	s_nop 0                                                    // 00000000207C: BF800000
	s_nop 0                                                    // 000000002080: BF800000
	s_nop 0                                                    // 000000002084: BF800000
	s_nop 0                                                    // 000000002088: BF800000
	s_nop 0                                                    // 00000000208C: BF800000
	s_nop 0                                                    // 000000002090: BF800000
	s_nop 0                                                    // 000000002094: BF800000
	s_nop 0                                                    // 000000002098: BF800000
	s_nop 0                                                    // 00000000209C: BF800000
	s_nop 0                                                    // 0000000020A0: BF800000
	s_nop 0                                                    // 0000000020A4: BF800000
	s_nop 0                                                    // 0000000020A8: BF800000
	s_nop 0                                                    // 0000000020AC: BF800000
	s_nop 0                                                    // 0000000020B0: BF800000
	s_nop 0                                                    // 0000000020B4: BF800000
	s_nop 0                                                    // 0000000020B8: BF800000
	s_nop 0                                                    // 0000000020BC: BF800000
	s_nop 0                                                    // 0000000020C0: BF800000
	s_nop 0                                                    // 0000000020C4: BF800000
	s_nop 0                                                    // 0000000020C8: BF800000
	s_nop 0                                                    // 0000000020CC: BF800000
	s_nop 0                                                    // 0000000020D0: BF800000
	s_nop 0                                                    // 0000000020D4: BF800000
	s_nop 0                                                    // 0000000020D8: BF800000
	s_nop 0                                                    // 0000000020DC: BF800000
	s_nop 0                                                    // 0000000020E0: BF800000
	s_nop 0                                                    // 0000000020E4: BF800000
	s_nop 0                                                    // 0000000020E8: BF800000
	s_nop 0                                                    // 0000000020EC: BF800000
	s_nop 0                                                    // 0000000020F0: BF800000
	s_nop 0                                                    // 0000000020F4: BF800000
	s_nop 0                                                    // 0000000020F8: BF800000
	s_nop 0                                                    // 0000000020FC: BF800000
	s_nop 0                                                    // 000000002100: BF800000
	s_nop 0                                                    // 000000002104: BF800000
	s_nop 0                                                    // 000000002108: BF800000
	s_nop 0                                                    // 00000000210C: BF800000
	s_nop 0                                                    // 000000002110: BF800000
	s_nop 0                                                    // 000000002114: BF800000
	s_nop 0                                                    // 000000002118: BF800000
	s_nop 0                                                    // 00000000211C: BF800000
	s_nop 0                                                    // 000000002120: BF800000
	s_nop 0                                                    // 000000002124: BF800000
	s_nop 0                                                    // 000000002128: BF800000
	s_nop 0                                                    // 00000000212C: BF800000
	s_nop 0                                                    // 000000002130: BF800000
	s_nop 0                                                    // 000000002134: BF800000
	s_nop 0                                                    // 000000002138: BF800000
	s_nop 0                                                    // 00000000213C: BF800000
	s_nop 0                                                    // 000000002140: BF800000
	s_nop 0                                                    // 000000002144: BF800000
	s_nop 0                                                    // 000000002148: BF800000
	s_nop 0                                                    // 00000000214C: BF800000
	s_nop 0                                                    // 000000002150: BF800000
	s_nop 0                                                    // 000000002154: BF800000
	s_nop 0                                                    // 000000002158: BF800000
	s_nop 0                                                    // 00000000215C: BF800000
	s_nop 0                                                    // 000000002160: BF800000
	s_nop 0                                                    // 000000002164: BF800000
	s_nop 0                                                    // 000000002168: BF800000
	s_nop 0                                                    // 00000000216C: BF800000
	s_nop 0                                                    // 000000002170: BF800000
	s_nop 0                                                    // 000000002174: BF800000
	s_nop 0                                                    // 000000002178: BF800000
	s_nop 0                                                    // 00000000217C: BF800000
	s_nop 0                                                    // 000000002180: BF800000
	s_nop 0                                                    // 000000002184: BF800000
	s_nop 0                                                    // 000000002188: BF800000
	s_nop 0                                                    // 00000000218C: BF800000
	s_nop 0                                                    // 000000002190: BF800000
	s_nop 0                                                    // 000000002194: BF800000
	s_nop 0                                                    // 000000002198: BF800000
	s_nop 0                                                    // 00000000219C: BF800000
	s_nop 0                                                    // 0000000021A0: BF800000
	s_nop 0                                                    // 0000000021A4: BF800000
	s_nop 0                                                    // 0000000021A8: BF800000
	s_nop 0                                                    // 0000000021AC: BF800000
	s_nop 0                                                    // 0000000021B0: BF800000
	s_nop 0                                                    // 0000000021B4: BF800000
	s_nop 0                                                    // 0000000021B8: BF800000
	s_nop 0                                                    // 0000000021BC: BF800000
	s_nop 0                                                    // 0000000021C0: BF800000
	s_nop 0                                                    // 0000000021C4: BF800000
	s_nop 0                                                    // 0000000021C8: BF800000
	s_nop 0                                                    // 0000000021CC: BF800000
	s_nop 0                                                    // 0000000021D0: BF800000
	s_nop 0                                                    // 0000000021D4: BF800000
	s_nop 0                                                    // 0000000021D8: BF800000
	s_nop 0                                                    // 0000000021DC: BF800000
	s_nop 0                                                    // 0000000021E0: BF800000
	s_nop 0                                                    // 0000000021E4: BF800000
	s_nop 0                                                    // 0000000021E8: BF800000
	s_nop 0                                                    // 0000000021EC: BF800000
	s_nop 0                                                    // 0000000021F0: BF800000
	s_nop 0                                                    // 0000000021F4: BF800000
	s_nop 0                                                    // 0000000021F8: BF800000
	s_nop 0                                                    // 0000000021FC: BF800000
	s_nop 0                                                    // 000000002200: BF800000
	s_nop 0                                                    // 000000002204: BF800000
	s_nop 0                                                    // 000000002208: BF800000
	s_nop 0                                                    // 00000000220C: BF800000
	s_nop 0                                                    // 000000002210: BF800000
	s_nop 0                                                    // 000000002214: BF800000
	s_nop 0                                                    // 000000002218: BF800000
	s_nop 0                                                    // 00000000221C: BF800000
	s_nop 0                                                    // 000000002220: BF800000
	s_nop 0                                                    // 000000002224: BF800000
	s_nop 0                                                    // 000000002228: BF800000
	s_nop 0                                                    // 00000000222C: BF800000
	s_nop 0                                                    // 000000002230: BF800000
	s_nop 0                                                    // 000000002234: BF800000
	s_nop 0                                                    // 000000002238: BF800000
	s_nop 0                                                    // 00000000223C: BF800000
	s_nop 0                                                    // 000000002240: BF800000
	s_nop 0                                                    // 000000002244: BF800000
	s_nop 0                                                    // 000000002248: BF800000
	s_nop 0                                                    // 00000000224C: BF800000
	s_nop 0                                                    // 000000002250: BF800000
	s_nop 0                                                    // 000000002254: BF800000
	s_nop 0                                                    // 000000002258: BF800000
	s_nop 0                                                    // 00000000225C: BF800000
	s_nop 0                                                    // 000000002260: BF800000
	s_nop 0                                                    // 000000002264: BF800000
	s_nop 0                                                    // 000000002268: BF800000
	s_nop 0                                                    // 00000000226C: BF800000
	s_nop 0                                                    // 000000002270: BF800000
	s_nop 0                                                    // 000000002274: BF800000
	s_nop 0                                                    // 000000002278: BF800000
	s_nop 0                                                    // 00000000227C: BF800000
	s_nop 0                                                    // 000000002280: BF800000
	s_nop 0                                                    // 000000002284: BF800000
	s_nop 0                                                    // 000000002288: BF800000
	s_nop 0                                                    // 00000000228C: BF800000
	s_nop 0                                                    // 000000002290: BF800000
	s_nop 0                                                    // 000000002294: BF800000
	s_nop 0                                                    // 000000002298: BF800000
	s_nop 0                                                    // 00000000229C: BF800000
	s_nop 0                                                    // 0000000022A0: BF800000
	s_nop 0                                                    // 0000000022A4: BF800000
	s_nop 0                                                    // 0000000022A8: BF800000
	s_nop 0                                                    // 0000000022AC: BF800000
	s_nop 0                                                    // 0000000022B0: BF800000
	s_nop 0                                                    // 0000000022B4: BF800000
	s_nop 0                                                    // 0000000022B8: BF800000
	s_nop 0                                                    // 0000000022BC: BF800000
	s_nop 0                                                    // 0000000022C0: BF800000
	s_nop 0                                                    // 0000000022C4: BF800000
	s_nop 0                                                    // 0000000022C8: BF800000
	s_nop 0                                                    // 0000000022CC: BF800000
	s_nop 0                                                    // 0000000022D0: BF800000
	s_nop 0                                                    // 0000000022D4: BF800000
	s_nop 0                                                    // 0000000022D8: BF800000
	s_nop 0                                                    // 0000000022DC: BF800000
	s_nop 0                                                    // 0000000022E0: BF800000
	s_nop 0                                                    // 0000000022E4: BF800000
	s_nop 0                                                    // 0000000022E8: BF800000
	s_nop 0                                                    // 0000000022EC: BF800000
	s_nop 0                                                    // 0000000022F0: BF800000
	s_nop 0                                                    // 0000000022F4: BF800000
	s_nop 0                                                    // 0000000022F8: BF800000
	s_nop 0                                                    // 0000000022FC: BF800000
	s_nop 0                                                    // 000000002300: BF800000
	s_nop 0                                                    // 000000002304: BF800000
	s_nop 0                                                    // 000000002308: BF800000
	s_nop 0                                                    // 00000000230C: BF800000
	s_nop 0                                                    // 000000002310: BF800000
	s_nop 0                                                    // 000000002314: BF800000
	s_nop 0                                                    // 000000002318: BF800000
	s_nop 0                                                    // 00000000231C: BF800000
	s_nop 0                                                    // 000000002320: BF800000
	s_nop 0                                                    // 000000002324: BF800000
	s_nop 0                                                    // 000000002328: BF800000
	s_nop 0                                                    // 00000000232C: BF800000
	s_nop 0                                                    // 000000002330: BF800000
	s_nop 0                                                    // 000000002334: BF800000
	s_nop 0                                                    // 000000002338: BF800000
	s_nop 0                                                    // 00000000233C: BF800000
	s_nop 0                                                    // 000000002340: BF800000
	s_nop 0                                                    // 000000002344: BF800000
	s_nop 0                                                    // 000000002348: BF800000
	s_nop 0                                                    // 00000000234C: BF800000
	s_nop 0                                                    // 000000002350: BF800000
	s_nop 0                                                    // 000000002354: BF800000
	s_nop 0                                                    // 000000002358: BF800000
	s_nop 0                                                    // 00000000235C: BF800000
	s_nop 0                                                    // 000000002360: BF800000
	s_nop 0                                                    // 000000002364: BF800000
	s_nop 0                                                    // 000000002368: BF800000
	s_nop 0                                                    // 00000000236C: BF800000
	s_nop 0                                                    // 000000002370: BF800000
	s_nop 0                                                    // 000000002374: BF800000
	s_nop 0                                                    // 000000002378: BF800000
	s_nop 0                                                    // 00000000237C: BF800000
	s_nop 0                                                    // 000000002380: BF800000
	s_nop 0                                                    // 000000002384: BF800000
	s_nop 0                                                    // 000000002388: BF800000
	s_nop 0                                                    // 00000000238C: BF800000
	s_nop 0                                                    // 000000002390: BF800000
	s_nop 0                                                    // 000000002394: BF800000
	s_nop 0                                                    // 000000002398: BF800000
	s_nop 0                                                    // 00000000239C: BF800000
	s_nop 0                                                    // 0000000023A0: BF800000
	s_nop 0                                                    // 0000000023A4: BF800000
	s_nop 0                                                    // 0000000023A8: BF800000
	s_nop 0                                                    // 0000000023AC: BF800000
	s_nop 0                                                    // 0000000023B0: BF800000
	s_nop 0                                                    // 0000000023B4: BF800000
	s_nop 0                                                    // 0000000023B8: BF800000
	s_nop 0                                                    // 0000000023BC: BF800000
	s_nop 0                                                    // 0000000023C0: BF800000
	s_nop 0                                                    // 0000000023C4: BF800000
	s_nop 0                                                    // 0000000023C8: BF800000
	s_nop 0                                                    // 0000000023CC: BF800000
	s_nop 0                                                    // 0000000023D0: BF800000
	s_nop 0                                                    // 0000000023D4: BF800000
	s_nop 0                                                    // 0000000023D8: BF800000
	s_nop 0                                                    // 0000000023DC: BF800000
	s_nop 0                                                    // 0000000023E0: BF800000
	s_nop 0                                                    // 0000000023E4: BF800000
	s_nop 0                                                    // 0000000023E8: BF800000
	s_nop 0                                                    // 0000000023EC: BF800000
	s_nop 0                                                    // 0000000023F0: BF800000
	s_nop 0                                                    // 0000000023F4: BF800000
	s_nop 0                                                    // 0000000023F8: BF800000
	s_nop 0                                                    // 0000000023FC: BF800000
	s_nop 0                                                    // 000000002400: BF800000
	s_nop 0                                                    // 000000002404: BF800000
	s_nop 0                                                    // 000000002408: BF800000
	s_nop 0                                                    // 00000000240C: BF800000
	s_nop 0                                                    // 000000002410: BF800000
	s_nop 0                                                    // 000000002414: BF800000
	s_nop 0                                                    // 000000002418: BF800000
	s_nop 0                                                    // 00000000241C: BF800000
	s_nop 0                                                    // 000000002420: BF800000
	s_nop 0                                                    // 000000002424: BF800000
	s_nop 0                                                    // 000000002428: BF800000
	s_nop 0                                                    // 00000000242C: BF800000
	s_nop 0                                                    // 000000002430: BF800000
	s_nop 0                                                    // 000000002434: BF800000
	s_nop 0                                                    // 000000002438: BF800000
	s_nop 0                                                    // 00000000243C: BF800000
	s_nop 0                                                    // 000000002440: BF800000
	s_nop 0                                                    // 000000002444: BF800000
	s_nop 0                                                    // 000000002448: BF800000
	s_nop 0                                                    // 00000000244C: BF800000
	s_nop 0                                                    // 000000002450: BF800000
	s_nop 0                                                    // 000000002454: BF800000
	s_nop 0                                                    // 000000002458: BF800000
	s_nop 0                                                    // 00000000245C: BF800000
	s_nop 0                                                    // 000000002460: BF800000
	s_nop 0                                                    // 000000002464: BF800000
	s_nop 0                                                    // 000000002468: BF800000
	s_nop 0                                                    // 00000000246C: BF800000
	s_nop 0                                                    // 000000002470: BF800000
	s_nop 0                                                    // 000000002474: BF800000
	s_nop 0                                                    // 000000002478: BF800000
	s_nop 0                                                    // 00000000247C: BF800000
.Lrocke_mxfp6_pack_op_4096x65536_r256_w128t4_xcd_c0c1p_nosc_end:
.size rocke_mxfp6_pack_op_4096x65536_r256_w128t4_xcd_c0c1p_nosc, .Lrocke_mxfp6_pack_op_4096x65536_r256_w128t4_xcd_c0c1p_nosc_end-rocke_mxfp6_pack_op_4096x65536_r256_w128t4_xcd_c0c1p_nosc

.rodata
.p2align 6
.amdhsa_kernel rocke_mxfp6_pack_op_4096x65536_r256_w128t4_xcd_c0c1p_nosc
  .amdhsa_group_segment_fixed_size 51200
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_kernarg_size 32
  .amdhsa_user_sgpr_count 2
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_next_free_vgpr 136
  .amdhsa_next_free_sgpr 98
  .amdhsa_reserve_vcc 0
  .amdhsa_accum_offset 80
  .amdhsa_float_denorm_mode_32 3
  .amdhsa_float_denorm_mode_16_64 3
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.target: amdgcn-amd-amdhsa--gfx950
amdhsa.version: [ 1, 1 ]
amdhsa.kernels:
  - .name: rocke_mxfp6_pack_op_4096x65536_r256_w128t4_xcd_c0c1p_nosc
    .symbol: rocke_mxfp6_pack_op_4096x65536_r256_w128t4_xcd_c0c1p_nosc.kd
    .args:
      - .address_space:  global
        .name:           codes_src
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .name:           scales_src
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .name:           codes_dst
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .name:           scales_dst
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 51200
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .max_flat_workgroup_size: 256
    .private_segment_fixed_size: 0
    .sgpr_count: 17
    .sgpr_spill_count: 0
    .uniform_work_group_size: 1
    .vgpr_count: 79
    .vgpr_spill_count: 0
    .agpr_count: 0
    .wavefront_size: 64
...
.end_amdgpu_metadata
