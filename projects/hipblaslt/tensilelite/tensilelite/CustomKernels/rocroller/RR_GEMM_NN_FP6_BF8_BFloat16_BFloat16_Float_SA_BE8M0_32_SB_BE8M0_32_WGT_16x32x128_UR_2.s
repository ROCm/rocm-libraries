// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2
.p2align 8
.type RR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2,@function
RR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_exec_begin:


s_load_dwordx2 s[20:21], s[0:1], 56 // Load scalar value
s_load_dwordx2 s[22:23], s[0:1], 64 // Load scalar value
s_load_dwordx2 s[24:25], s[0:1], 72 // Load scalar value
s_load_dwordx2 s[26:27], s[0:1], 80 // Load scalar value
s_load_dwordx2 s[28:29], s[0:1], 88 // Load scalar value
s_load_dwordx2 s[30:31], s[0:1], 96 // Load scalar value
s_load_dwordx2 s[32:33], s[0:1], 104 // Load scalar value
s_load_dwordx2 s[34:35], s[0:1], 112 // Load scalar value
s_load_dwordx2 s[36:37], s[0:1], 120 // Load scalar value

v_bfe_u32 v1, v0, 0, 10
v_bfe_u32 v2, v0, 10, 10


v_and_b32 v0, 63, v1 // (op 771) 
v_and_b32 v3, 48, v1 // (op 771) 
v_sub_u32 v4, v0, v3 // (op 771) 

v_lshrrev_b32 v3, 4, v0 // (op 771) 

v_lshl_add_u32 v0, v3, 5, v4 // (op 771) 


v_lshrrev_b32 v3, 7, v1 // (op 771) 
v_lshlrev_b32 v4, 4, v3 // (op 771) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 771) 


v_mul_lo_u32 v0, v3, 6 // (op 771) 
v_lshrrev_b32 v4, 3, v0 // (op 771) 

v_lshrrev_b32 v0, 4, v3 // (op 771) 

v_lshlrev_b32 v3, 2, v0 // (op 771) 

v_add_u32 v0, v4, v3 // (op 771) 


v_lshrrev_b32 v3, 6, v1 // (op 823) 
v_and_b32 v4, 1, v3 // (op 823) 

v_and_b32 v3, 15, v1 // (op 823) 
v_lshl_add_u32 v5, v4, 4, v3 // (op 823) 


v_and_b32 v3, 48, v1 // (op 823) 
v_lshl_add_u32 v4, v5, 7, v3 // (op 823) 


v_lshrrev_b32 v3, 6, v1 // (op 876) 
v_and_b32 v5, 1, v3 // (op 876) 

v_and_b32 v3, 15, v1 // (op 876) 
v_lshl_add_u32 v6, v5, 4, v3 // (op 876) 


v_and_b32 v3, 63, v1 // (op 876) 
v_lshrrev_b32 v5, 4, v3 // (op 876) 

v_lshl_add_u32 v3, v6, 2, v5 // (op 876) 


v_and_b32 v5, 63, v1 // (op 934) 
v_and_b32 v6, 48, v1 // (op 934) 
v_sub_u32 v7, v5, v6 // (op 934) 

v_lshrrev_b32 v6, 4, v5 // (op 934) 

v_lshl_add_u32 v5, v6, 5, v7 // (op 934) 


v_lshrrev_b32 v6, 7, v1 // (op 934) 
v_lshlrev_b32 v7, 4, v6 // (op 934) 

v_lshl_add_u32 v6, v5, 4, v7 // (op 934) 


v_mul_lo_u32 v5, v6, 6 // (op 934) 
v_lshrrev_b32 v7, 3, v5 // (op 934) 

v_lshrrev_b32 v5, 4, v6 // (op 934) 

v_lshlrev_b32 v6, 2, v5 // (op 934) 

v_add_u32 v5, v7, v6 // (op 934) 


v_lshrrev_b32 v6, 6, v1 // (op 973) 
v_and_b32 v7, 1, v6 // (op 973) 

v_and_b32 v6, 15, v1 // (op 973) 
v_lshl_add_u32 v8, v7, 4, v6 // (op 973) 


v_and_b32 v6, 48, v1 // (op 973) 
v_lshl_add_u32 v7, v8, 7, v6 // (op 973) 


v_lshrrev_b32 v6, 6, v1 // (op 997) 
v_and_b32 v8, 1, v6 // (op 997) 

v_and_b32 v6, 15, v1 // (op 997) 
v_lshl_add_u32 v9, v8, 4, v6 // (op 997) 


v_and_b32 v6, 63, v1 // (op 997) 
v_lshrrev_b32 v8, 4, v6 // (op 997) 

v_lshl_add_u32 v6, v9, 2, v8 // (op 997) 


v_mov_b32 v8, s14 // (op 2) Move value
v_mov_b32 v9, s15 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 61) call()
v_accvgpr_write a1, 0.00000 // (op 61) call()
v_accvgpr_write a2, 0.00000 // (op 61) call()
v_accvgpr_write a3, 0.00000 // (op 61) call()
s_lshl_b32 s0, s16, 4 // (op 481) 
v_mul_lo_u32 v11, v1, s3 // (op 481) most significant: low of low * high
s_mov_b32 s15, 0 // (op 481) convert
s_mov_b32 s14, s0 // (op 481) convert

v_mul_hi_u32 v12, v1, s2 // (op 481) most significant: high of low * low
v_mul_lo_u32 v10, v1, s2 // (op 481) least significant: low of low * low
v_add_u32 v11, v11, v12 // (op 481) most significant: sum

v_mov_b32 v12, s14 // (op 481) 
v_mov_b32 v13, s15 // (op 481) 
v_add_co_u32 v14, s[0:1], v12, v10 // (op 481) least significant half
v_addc_co_u32 v15, s[0:1], v13, v11, s[0:1] // (op 481) most significant half





v_mul_lo_u32 v11, v15, 6 // (op 481) most significant: low of high * low
v_mul_hi_u32 v12, v14, 6 // (op 481) most significant: high of low * low
v_mul_lo_u32 v10, v14, 6 // (op 481) least significant: low of low * low
v_add_u32 v11, v11, v12 // (op 481) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 481) 

v_mov_b32 v10, v12 // (op 481) convert

s_mul_i32 s1, s7, 6 // (op 483) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 483) least significant
s_mul_hi_u32 s14, s6, 6 // (op 483) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 483) most significant: sum

s_mul_i32 s15, s2, s1 // (op 483) most significant: low of low * high
s_mul_i32 s18, s3, s0 // (op 483) most significant: low of high * low
s_mul_i32 s14, s2, s0 // (op 483) least significant
s_mul_hi_u32 s19, s2, s0 // (op 483) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 483) most significant: sum
s_add_u32 s15, s15, s19 // (op 483) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 483) 

s_mov_b32 s40, s4 // (op 483) reg expression
s_mov_b32 s41, s5 // (op 483) reg expression
s_mov_b32 s42, s0 // (op 483) reg expression
s_mov_b32 s43, 131072 // (op 483) call()

s_lshl_b32 s0, s16, 4 // (op 500) 
v_mul_lo_u32 v13, v1, s3 // (op 500) most significant: low of low * high
v_mul_hi_u32 v11, v1, s2 // (op 500) most significant: high of low * low
v_mul_lo_u32 v12, v1, s2 // (op 500) least significant: low of low * low
v_add_u32 v13, v13, v11 // (op 500) most significant: sum

s_mov_b32 s15, 0 // (op 500) convert
s_mov_b32 s14, s0 // (op 500) convert

v_mov_b32 v11, s14 // (op 500) 
v_mov_b32 v14, s15 // (op 500) 
v_add_co_u32 v16, s[0:1], v11, v12 // (op 500) least significant half
v_addc_co_u32 v17, s[0:1], v14, v13, s[0:1] // (op 500) most significant half





v_mul_lo_u32 v15, v17, 6 // (op 500) most significant: low of high * low
v_mul_hi_u32 v11, v16, 6 // (op 500) most significant: high of low * low
v_mul_lo_u32 v14, v16, 6 // (op 500) least significant: low of low * low
v_add_u32 v15, v15, v11 // (op 500) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 500) 

v_mov_b32 v11, v12 // (op 500) convert

s_mul_i32 s1, s7, 6 // (op 502) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 502) least significant
s_mul_hi_u32 s14, s6, 6 // (op 502) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 502) most significant: sum

s_mul_i32 s15, s2, s1 // (op 502) most significant: low of low * high
s_mul_i32 s18, s3, s0 // (op 502) most significant: low of high * low
s_mul_i32 s14, s2, s0 // (op 502) least significant
s_mul_hi_u32 s19, s2, s0 // (op 502) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 502) most significant: sum
s_add_u32 s15, s15, s19 // (op 502) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 502) 

s_mov_b32 s40, s4 // (op 502) reg expression
s_mov_b32 s41, s5 // (op 502) reg expression
s_mov_b32 s42, s0 // (op 502) reg expression
s_mov_b32 s43, 131072 // (op 502) call()

v_and_b32 v12, -8, v1 // (op 545) 
v_sub_u32 v13, v1, v12 // (op 545) 

v_lshlrev_b32 v12, 4, v13 // (op 545) 

v_mov_b32 v15, 0 // (op 545) convert
v_mov_b32 v14, v12 // (op 545) convert

v_lshrrev_b32 v12, 3, v1 // (op 545) 
v_lshl_add_u32 v13, s17, 5, v12 // (op 545) 

v_mul_lo_u32 v17, v13, s9 // (op 545) most significant: low of low * high
v_mul_hi_u32 v12, v13, s8 // (op 545) most significant: high of low * low
v_mul_lo_u32 v16, v13, s8 // (op 545) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 545) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v16 // (op 545) least significant half
v_addc_co_u32 v13, s[0:1], v15, v17, s[0:1] // (op 545) most significant half



s_mul_i32 s1, s8, s13 // (op 547) most significant: low of low * high
s_mul_i32 s14, s9, s12 // (op 547) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 547) least significant
s_mul_hi_u32 s15, s8, s12 // (op 547) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 547) most significant: sum
s_add_u32 s1, s1, s15 // (op 547) most significant: sum


s_mov_b32 s44, s10 // (op 547) reg expression
s_mov_b32 s45, s11 // (op 547) reg expression
s_mov_b32 s46, s0 // (op 547) reg expression
s_mov_b32 s47, 131072 // (op 547) call()

v_and_b32 v14, -8, v1 // (op 564) 
v_sub_u32 v15, v1, v14 // (op 564) 

v_lshlrev_b32 v14, 4, v15 // (op 564) 

v_mov_b32 v17, 0 // (op 564) convert
v_mov_b32 v16, v14 // (op 564) convert

v_lshrrev_b32 v14, 3, v1 // (op 564) 
v_lshl_add_u32 v15, s17, 5, v14 // (op 564) 

v_mul_lo_u32 v19, v15, s9 // (op 564) most significant: low of low * high
v_mul_hi_u32 v14, v15, s8 // (op 564) most significant: high of low * low
v_mul_lo_u32 v18, v15, s8 // (op 564) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 564) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v18 // (op 564) least significant half
v_addc_co_u32 v15, s[0:1], v17, v19, s[0:1] // (op 564) most significant half



s_mul_i32 s1, s8, s13 // (op 566) most significant: low of low * high
s_mul_i32 s14, s9, s12 // (op 566) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 566) least significant
s_mul_hi_u32 s15, s8, s12 // (op 566) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 566) most significant: sum
s_add_u32 s1, s1, s15 // (op 566) most significant: sum


s_mov_b32 s44, s10 // (op 566) reg expression
s_mov_b32 s45, s11 // (op 566) reg expression
s_mov_b32 s46, s0 // (op 566) reg expression
s_mov_b32 s47, 131072 // (op 566) call()

v_lshrrev_b32 v16, 7, v1 // (op 609) 
v_and_b32 v17, 15, v1 // (op 609) 
v_lshl_add_u32 v18, v16, 4, v17 // (op 609) 


v_lshl_add_u32 v16, s16, 4, v18 // (op 609) 

v_and_b32 v17, 63, v1 // (op 609) 
v_lshrrev_b32 v18, 4, v17 // (op 609) 


s_waitcnt lgkmcnt(0)
v_mul_lo_u32 v17, v18, s20 // (op 609) 

v_add_u32 v18, v16, v17 // (op 609) 


s_mul_i32 s1, s20, s25 // (op 611) most significant: low of low * high
s_mul_i32 s14, s21, s24 // (op 611) most significant: low of high * low
s_mul_i32 s0, s20, s24 // (op 611) least significant
s_mul_hi_u32 s15, s20, s24 // (op 611) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 611) most significant: sum
s_add_u32 s1, s1, s15 // (op 611) most significant: sum


s_mov_b32 s48, s22 // (op 611) reg expression
s_mov_b32 s49, s23 // (op 611) reg expression
s_mov_b32 s50, s0 // (op 611) reg expression
s_mov_b32 s51, 131072 // (op 611) call()

v_lshrrev_b32 v16, 7, v1 // (op 628) 
v_and_b32 v17, 15, v1 // (op 628) 
v_lshl_add_u32 v19, v16, 4, v17 // (op 628) 


v_lshl_add_u32 v16, s16, 4, v19 // (op 628) 

v_and_b32 v17, 63, v1 // (op 628) 
v_lshrrev_b32 v19, 4, v17 // (op 628) 

v_mul_lo_u32 v17, v19, s20 // (op 628) 

v_add_u32 v19, v16, v17 // (op 628) 


s_mul_i32 s1, s20, s25 // (op 630) most significant: low of low * high
s_mul_i32 s14, s21, s24 // (op 630) most significant: low of high * low
s_mul_i32 s0, s20, s24 // (op 630) least significant
s_mul_hi_u32 s15, s20, s24 // (op 630) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 630) most significant: sum
s_add_u32 s1, s1, s15 // (op 630) most significant: sum


s_mov_b32 s48, s22 // (op 630) reg expression
s_mov_b32 s49, s23 // (op 630) reg expression
s_mov_b32 s50, s0 // (op 630) reg expression
s_mov_b32 s51, 131072 // (op 630) call()

v_and_b32 v16, -4, v1 // (op 673) 
v_sub_u32 v17, v1, v16 // (op 673) 

v_lshrrev_b32 v16, 2, v1 // (op 673) 
v_lshl_add_u32 v20, s17, 5, v16 // (op 673) 

v_mul_lo_u32 v16, v20, s26 // (op 673) 

v_add_u32 v20, v17, v16 // (op 673) 


s_mul_i32 s1, s26, s13 // (op 675) most significant: low of low * high
s_mul_i32 s14, s27, s12 // (op 675) most significant: low of high * low
s_mul_i32 s0, s26, s12 // (op 675) least significant
s_mul_hi_u32 s15, s26, s12 // (op 675) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 675) most significant: sum
s_add_u32 s1, s1, s15 // (op 675) most significant: sum


s_mov_b32 s52, s28 // (op 675) reg expression
s_mov_b32 s53, s29 // (op 675) reg expression
s_mov_b32 s54, s0 // (op 675) reg expression
s_mov_b32 s55, 131072 // (op 675) call()

v_and_b32 v16, -4, v1 // (op 692) 
v_sub_u32 v17, v1, v16 // (op 692) 

v_lshrrev_b32 v16, 2, v1 // (op 692) 
v_lshl_add_u32 v21, s17, 5, v16 // (op 692) 

v_mul_lo_u32 v16, v21, s26 // (op 692) 

v_add_u32 v21, v17, v16 // (op 692) 


s_mul_i32 s1, s26, s13 // (op 694) most significant: low of low * high
s_mul_i32 s14, s27, s12 // (op 694) most significant: low of high * low
s_mul_i32 s0, s26, s12 // (op 694) least significant
s_mul_hi_u32 s15, s26, s12 // (op 694) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 694) most significant: sum
s_add_u32 s1, s1, s15 // (op 694) most significant: sum


s_mov_b32 s52, s28 // (op 694) reg expression
s_mov_b32 s53, s29 // (op 694) reg expression
s_mov_b32 s54, s0 // (op 694) reg expression
s_mov_b32 s55, 131072 // (op 694) call()


v_mov_b64 v[16:17], s[20:21] // (op 341) convert

v_mov_b32 v22, v18 // (op 341) reg expression
v_mov_b32 v23, v22 // (op 341) 
buffer_load_ubyte v24, v23, s[48:51], 0 offen offset: 0 // (op 341) Load value



v_and_b32 v16, -8, v1 // (op 811) 
v_sub_u32 v17, v1, v16 // (op 811) 

v_lshlrev_b32 v16, 4, v17 // (op 811) 

v_lshrrev_b32 v17, 3, v1 // (op 811) 
v_lshl_add_u32 v22, v17, 7, v16 // (op 811) 


s_nop 0
v_readfirstlane_b32 s0, v22 // (op 811) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[8:9], 4 // (op 437) 
v_mov_b64 v[16:17], s[14:15] // (op 437) convert


v_mov_b64 v[22:23], v[12:13] // (op 437) 
s_mov_b32 m0, s0 // (op 437) reg expression
buffer_load_dwordx4 v22, s[44:47], 0 offen offset: 0 lds // (op 437) Load value direct to lds
v_add_co_u32 v22, s[14:15], v22, v16 // (op 437) least significant half
v_addc_co_u32 v23, s[14:15], 0, v23, s[14:15] // (op 437) most significant half

s_add_u32 m0, m0, 2048 // (op 437) 
buffer_load_dwordx4 v22, s[44:47], 0 offen offset: 0 lds // (op 437) Load value direct to lds






buffer_load_ubyte v16, v20, s[52:55], 0 offen offset: 0 // (op 348) Load value


v_mov_b32 v17, v10 // (op 351) reg expression

buffer_load_dwordx3 v[26:28], v17, s[40:43], 0 offen // (op 351) Load value


v_mov_b64 v[22:23], s[20:21] // (op 355) convert

s_lshl_b32 s0, s20, 2 // (op 355) 
v_add_u32 v17, v18, s0 // (op 355) 

v_mov_b32 v25, v17 // (op 355) 
buffer_load_ubyte v29, v25, s[48:51], 0 offen offset: 0 // (op 355) Load value




v_and_b32 v25, -8, v1 // (op 961) 
v_sub_u32 v17, v1, v25 // (op 961) 

v_lshlrev_b32 v18, 4, v17 // (op 961) 

v_lshrrev_b32 v17, 3, v1 // (op 961) 
v_lshl_add_u32 v25, v17, 7, v18 // (op 961) 


s_nop 0
v_readfirstlane_b32 s0, v25 // (op 961) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[8:9], 4 // (op 442) 
v_mov_b64 v[22:23], s[14:15] // (op 442) convert


v_mov_b64 v[30:31], v[12:13] // (op 442) 
s_add_u32 s1, 4096, s0 // (op 442) 
s_mov_b32 m0, s1 // (op 442) reg expression
s_mov_b32 s14, 128 // (op 442) call()
buffer_load_dwordx4 v30, s[44:47], s14 offen offset: 0 lds // (op 442) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v22 // (op 442) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 442) most significant half

s_add_u32 m0, m0, 2048 // (op 442) 
buffer_load_dwordx4 v30, s[44:47], s14 offen offset: 0 lds // (op 442) Load value direct to lds









buffer_load_ubyte v25, v20, s[52:55], 0 offen offset:4 // (op 362) Load value



s_lshl_b64 s[0:1], s[2:3], 7 // (op 365) 
s_mul_i32 s15, s1, 6 // (op 365) most significant: low of high * low
s_mul_i32 s14, s0, 6 // (op 365) least significant
s_mul_hi_u32 s18, s0, 6 // (op 365) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 365) most significant: sum


s_lshr_b64 s[0:1], s[14:15], 3 // (op 365) 

v_add_u32 v20, v10, s0 // (op 365) 


buffer_load_dwordx3 v[30:32], v20, s[40:43], 0 offen // (op 365) Load value


v_and_b32 v10, -4, v1 // (op 864) 
v_sub_u32 v20, v1, v10 // (op 864) 

v_and_b32 v10, 4294967292, v1 // (op 864) 
v_add_u32 v12, v10, v20 // (op 864) 





s_waitcnt vmcnt(6)
ds_write_b8 v12, v16 offset:8192 // (op 368) Store local data 

v_lshlrev_b32 v10, 4, v1 // (op 759) 
v_mul_lo_u32 v20, v10, 6 // (op 759) 

v_lshrrev_b32 v10, 3, v20 // (op 759) 

v_and_b32 v20, 268435455, v1 // (op 759) 
v_lshlrev_b32 v12, 2, v20 // (op 759) 

v_add_u32 v13, v10, v12 // (op 759) 





s_waitcnt vmcnt(5)
ds_write_b96 v13, v[26:28] offset:8320 // (op 369) Store local data 

v_lshlrev_b32 v10, 4, v1 // (op 730) 
v_mul_lo_u32 v20, v10, 6 // (op 730) 

v_lshrrev_b32 v10, 3, v20 // (op 730) 

v_and_b32 v20, 268435455, v1 // (op 730) 
v_lshlrev_b32 v12, 2, v20 // (op 730) 

v_add_u32 v13, v10, v12 // (op 730) 


v_and_b32 v10, -8, v1 // (op 783) 
v_sub_u32 v12, v1, v10 // (op 783) 

v_lshlrev_b32 v10, 4, v12 // (op 783) 

v_lshrrev_b32 v12, 3, v1 // (op 783) 
v_lshl_add_u32 v20, v12, 7, v10 // (op 783) 


s_nop 0
v_readfirstlane_b32 s0, v20 // (op 783) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v10, -4, v1 // (op 835) 
v_sub_u32 v12, v1, v10 // (op 835) 

v_and_b32 v10, 4294967292, v1 // (op 835) 
v_add_u32 v20, v10, v12 // (op 835) 


v_lshlrev_b32 v10, 4, v1 // (op 922) 
v_mul_lo_u32 v12, v10, 6 // (op 922) 

v_lshrrev_b32 v10, 3, v12 // (op 922) 

v_and_b32 v12, 268435455, v1 // (op 922) 
v_lshlrev_b32 v17, 2, v12 // (op 922) 

v_add_u32 v12, v10, v17 // (op 922) 


v_and_b32 v10, -8, v1 // (op 946) 
v_sub_u32 v17, v1, v10 // (op 946) 

v_lshlrev_b32 v10, 4, v17 // (op 946) 

v_lshrrev_b32 v17, 3, v1 // (op 946) 
v_lshl_add_u32 v18, v17, 7, v10 // (op 946) 


s_nop 0
v_readfirstlane_b32 s1, v18 // (op 946) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v10, -4, v1 // (op 985) 
v_sub_u32 v17, v1, v10 // (op 985) 

v_and_b32 v10, 4294967292, v1 // (op 985) 
v_add_u32 v18, v10, v17 // (op 985) 


s_mov_b32 s14, 0 // (op 57) call()
s_ashr_i64 s[18:19], s[6:7], 7 // (op 56) 
s_mov_b32 s15, s18 // (op 56) convert

s_and_b32 s18, s15, 4294967294 // (op 56) 

s_cmp_lt_u32 s14, s18 // (op 56) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoop_56 // (op 56) Condition: Top (jump to RR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoop_56 if false)
RR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoop_56:
s_add_u32 s14, s14, 2 // (op 58) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1030) 



ds_read_b96_tr_b6 v[34:36], v0 offset:8320 // (op 67) Transpose load local data 
ds_read_b96_tr_b6 v[37:39], v0 offset:8576 // (op 67) Transpose load local data 
s_barrier  // (op 1043) 



ds_read_b128 v[40:43], v4 // (op 72) Load local data 
ds_read_b128 v[44:47], v4 offset:64 // (op 72) Load local data 
s_barrier  // (op 1056) 



ds_read_u8 v10, v3 offset:8192 // (op 81) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:39], v[40:47], a[0:3], v24, v10 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 84) 




v_mov_b64 v[22:23], s[20:21] // (op 77) convert

v_mov_b32 v10, v19 // (op 77) 
s_lshl_b32 s15, s20, 3 // (op 77) 
v_add_u32 v17, v10, s15 // (op 77) 

v_mov_b32 v33, v17 // (op 77) 
buffer_load_ubyte v24, v33, s[48:51], 0 offen offset: 0 // (op 77) Load value



s_barrier  // (op 1046) 

s_lshl_b64 s[18:19], s[8:9], 4 // (op 420) 
v_mov_b64 v[22:23], s[18:19] // (op 420) convert


v_mov_b64 v[34:35], v[14:15] // (op 420) 
v_mov_b64 v[36:37], v[34:35] // (op 420) 
s_mov_b32 m0, s0 // (op 420) reg expression
s_mov_b32 s15, 256 // (op 420) call()
buffer_load_dwordx4 v36, s[44:47], s15 offen offset: 0 lds // (op 420) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v22 // (op 420) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 420) most significant half

s_add_u32 m0, m0, 2048 // (op 420) 
buffer_load_dwordx4 v36, s[44:47], s15 offen offset: 0 lds // (op 420) Load value direct to lds





v_mov_b32 v17, v21 // (op 21) 

buffer_load_ubyte v16, v17, s[52:55], 0 offen offset:8 // (op 21) Load value


v_mov_b32 v33, v11 // (op 4) 
s_lshl_b64 s[18:19], s[2:3], 7 // (op 4) 
s_mul_i32 s39, s19, 6 // (op 4) most significant: low of high * low
s_mul_i32 s38, s18, 6 // (op 4) least significant
s_mul_hi_u32 s15, s18, 6 // (op 4) most significant: high of low * low
s_add_u32 s39, s39, s15 // (op 4) most significant: sum


s_lshr_b64 s[18:19], s[38:39], 3 // (op 4) 

s_lshl_b32 s15, s18, 1 // (op 4) 

v_add_u32 v22, v33, s15 // (op 4) 


buffer_load_dwordx3 v[26:28], v22, s[40:43], 0 offen // (op 4) Load value

s_barrier  // (op 1085) 



s_waitcnt vmcnt(6)
ds_write_b8 v18, v25 offset:10368 // (op 240) Store local data 
s_barrier  // (op 1073) 



s_waitcnt vmcnt(5)
ds_write_b96 v12, v[30:32] offset:10496 // (op 229) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1070) 



ds_read_b96_tr_b6 v[36:38], v5 offset:10496 // (op 231) Transpose load local data 
ds_read_b96_tr_b6 v[39:41], v5 offset:10752 // (op 231) Transpose load local data 
s_barrier  // (op 1076) 
s_barrier  // (op 1136) 



ds_read_b128 v[42:45], v7 offset:4096 // (op 233) Load local data 
ds_read_b128 v[46:49], v7 offset:4160 // (op 233) Load local data 
s_barrier  // (op 1082) 



ds_read_u8 v22, v6 offset:10368 // (op 237) Load local data 
s_barrier  // (op 1146) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:41], v[42:49], a[0:3], v29, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 238) 




v_mov_b64 v[22:23], s[20:21] // (op 235) convert

s_lshl_b32 s15, s20, 2 // (op 235) 
s_mul_i32 s18, 3, s15 // (op 235) 

v_add_u32 v36, v10, s18 // (op 235) 

v_mov_b32 v37, v36 // (op 235) 
buffer_load_ubyte v29, v37, s[48:51], 0 offen offset: 0 // (op 235) Load value




s_barrier  // (op 1079) 
s_lshl_b32 s15, s20, 3 // (op 620) 

v_add_u32 v19, v19, s15 // (op 620) 

s_lshl_b64 s[18:19], s[8:9], 4 // (op 432) 
v_mov_b64 v[22:23], s[18:19] // (op 432) convert


v_mov_b64 v[36:37], v[34:35] // (op 432) 
s_add_u32 s15, 4096, s1 // (op 432) 
s_mov_b32 m0, s15 // (op 432) reg expression
s_mov_b32 s18, 384 // (op 432) call()
buffer_load_dwordx4 v36, s[44:47], s18 offen offset: 0 lds // (op 432) Load value direct to lds
v_add_co_u32 v36, s[38:39], v36, v22 // (op 432) least significant half
v_addc_co_u32 v37, s[38:39], 0, v37, s[38:39] // (op 432) most significant half

s_add_u32 m0, m0, 2048 // (op 432) 
buffer_load_dwordx4 v36, s[44:47], s18 offen offset: 0 lds // (op 432) Load value direct to lds








v_add_co_u32 v14, vcc, 256, v14 // (op 556) least significant half
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 556) most significant half
buffer_load_ubyte v25, v17, s[52:55], 0 offen offset:12 // (op 242) Load value



v_add_u32 v21, 8, v21 // (op 684) 
s_lshl_b64 s[18:19], s[2:3], 7 // (op 228) 
s_mul_i32 s39, s19, 6 // (op 228) most significant: low of high * low
s_mul_i32 s38, s18, 6 // (op 228) least significant
s_mul_hi_u32 s15, s18, 6 // (op 228) most significant: high of low * low
s_add_u32 s39, s39, s15 // (op 228) most significant: sum


s_lshr_b64 s[18:19], s[38:39], 3 // (op 228) 

s_mul_i32 s15, 3, s18 // (op 228) 

v_add_u32 v10, v33, s15 // (op 228) 


buffer_load_dwordx3 v[30:32], v10, s[40:43], 0 offen // (op 228) Load value


s_barrier  // (op 1059) 
s_lshl_b64 s[18:19], s[2:3], 7 // (op 492) 



s_mul_i32 s39, s19, 6 // (op 492) most significant: low of high * low
s_mul_i32 s38, s18, 6 // (op 492) least significant
s_mul_hi_u32 s15, s18, 6 // (op 492) most significant: high of low * low
s_add_u32 s39, s39, s15 // (op 492) most significant: sum


s_lshr_b64 s[18:19], s[38:39], 3 // (op 492) 

s_lshl_b32 s15, s18, 1 // (op 492) 

v_add_u32 v11, v11, s15 // (op 492) 

s_waitcnt vmcnt(6)
ds_write_b8 v20, v16 offset:8192 // (op 26) Store local data 
s_barrier  // (op 1033) 



s_waitcnt vmcnt(5)
ds_write_b96 v13, v[26:28] offset:8320 // (op 6) Store local data 
s_ashr_i64 s[18:19], s[6:7], 7 // (op 56) 
s_mov_b32 s15, s18 // (op 56) convert

s_and_b32 s18, s15, 4294967294 // (op 56) 

s_cmp_lt_u32 s14, s18 // (op 56) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoop_56 // (op 56) Condition: Bottom (jump to RR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoop_56 if true)
RR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoop_56:














s_barrier  // (op 1098) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 455) 
s_lshr_b32 s14, s0, 1 // (op 455) 

v_lshl_add_u32 v10, s14, 8, v1 // (op 455) 

s_lshl_b32 s0, s16, 4 // (op 455) 
v_mul_lo_u32 v27, v10, s3 // (op 455) most significant: low of low * high
v_mul_hi_u32 v28, v10, s2 // (op 455) most significant: high of low * low
v_mul_lo_u32 v26, v10, s2 // (op 455) least significant: low of low * low
v_add_u32 v27, v27, v28 // (op 455) most significant: sum


s_mov_b32 s15, 0 // (op 455) convert
s_mov_b32 s14, s0 // (op 455) convert

v_mov_b32 v28, s14 // (op 455) 
v_mov_b32 v10, s15 // (op 455) 
v_add_co_u32 v12, s[0:1], v28, v26 // (op 455) least significant half
v_addc_co_u32 v13, s[0:1], v10, v27, s[0:1] // (op 455) most significant half





v_mul_lo_u32 v11, v13, 6 // (op 455) most significant: low of high * low
v_mul_hi_u32 v14, v12, 6 // (op 455) most significant: high of low * low
v_mul_lo_u32 v10, v12, 6 // (op 455) least significant: low of low * low
v_add_u32 v11, v11, v14 // (op 455) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 455) 

v_mov_b32 v10, v12 // (op 455) convert

s_mul_i32 s1, s7, 6 // (op 457) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 457) least significant
s_mul_hi_u32 s14, s6, 6 // (op 457) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 457) most significant: sum

s_mul_i32 s15, s2, s1 // (op 457) most significant: low of low * high
s_mul_i32 s18, s3, s0 // (op 457) most significant: low of high * low
s_mul_i32 s14, s2, s0 // (op 457) least significant
s_mul_hi_u32 s19, s2, s0 // (op 457) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 457) most significant: sum
s_add_u32 s15, s15, s19 // (op 457) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 457) 

s_mov_b32 s40, s4 // (op 457) reg expression
s_mov_b32 s41, s5 // (op 457) reg expression
s_mov_b32 s42, s0 // (op 457) reg expression
s_mov_b32 s43, 131072 // (op 457) call()

v_and_b32 v11, -8, v1 // (op 522) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 522) 
v_sub_u32 v12, v1, v11 // (op 522) 

s_lshr_b32 s14, s0, 1 // (op 522) 

v_lshlrev_b32 v11, 4, v12 // (op 522) 

v_lshl_add_u32 v12, s14, 8, v11 // (op 522) 


v_mov_b32 v27, 0 // (op 522) convert
v_mov_b32 v26, v12 // (op 522) convert

v_lshrrev_b32 v28, 3, v1 // (op 522) 
v_lshl_add_u32 v11, s17, 5, v28 // (op 522) 

v_mul_lo_u32 v13, v11, s9 // (op 522) most significant: low of low * high
v_mul_hi_u32 v28, v11, s8 // (op 522) most significant: high of low * low
v_mul_lo_u32 v12, v11, s8 // (op 522) least significant: low of low * low
v_add_u32 v13, v13, v28 // (op 522) most significant: sum


v_add_co_u32 v14, s[0:1], v26, v12 // (op 522) least significant half
v_addc_co_u32 v15, s[0:1], v27, v13, s[0:1] // (op 522) most significant half



s_mul_i32 s1, s8, s13 // (op 524) most significant: low of low * high
s_mul_i32 s14, s9, s12 // (op 524) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 524) least significant
s_mul_hi_u32 s15, s8, s12 // (op 524) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 524) most significant: sum
s_add_u32 s1, s1, s15 // (op 524) most significant: sum


s_mov_b32 s44, s10 // (op 524) reg expression
s_mov_b32 s45, s11 // (op 524) reg expression
s_mov_b32 s46, s0 // (op 524) reg expression
s_mov_b32 s47, 131072 // (op 524) call()

v_lshrrev_b32 v11, 7, v1 // (op 586) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 586) 
v_and_b32 v12, 15, v1 // (op 586) 
s_lshr_b32 s14, s0, 1 // (op 586) 

v_lshl_add_u32 v13, v11, 4, v12 // (op 586) 


v_lshl_add_u32 v11, s16, 4, v13 // (op 586) 

v_and_b32 v12, 63, v1 // (op 586) 
v_lshrrev_b32 v13, 4, v12 // (op 586) 

v_lshl_add_u32 v12, s14, 3, v13 // (op 586) 


v_mul_lo_u32 v13, v12, s20 // (op 586) 

v_add_u32 v12, v11, v13 // (op 586) 


s_mul_i32 s1, s20, s25 // (op 588) most significant: low of low * high
s_mul_i32 s14, s21, s24 // (op 588) most significant: low of high * low
s_mul_i32 s0, s20, s24 // (op 588) least significant
s_mul_hi_u32 s15, s20, s24 // (op 588) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 588) most significant: sum
s_add_u32 s1, s1, s15 // (op 588) most significant: sum


s_mov_b32 s48, s22 // (op 588) reg expression
s_mov_b32 s49, s23 // (op 588) reg expression
s_mov_b32 s50, s0 // (op 588) reg expression
s_mov_b32 s51, 131072 // (op 588) call()

v_and_b32 v11, -4, v1 // (op 650) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 650) 
v_sub_u32 v13, v1, v11 // (op 650) 

s_lshr_b32 s14, s0, 1 // (op 650) 

v_lshrrev_b32 v11, 2, v1 // (op 650) 
v_lshl_add_u32 v16, s14, 3, v13 // (op 650) 


v_lshl_add_u32 v13, s17, 5, v11 // (op 650) 

v_mul_lo_u32 v11, v13, s26 // (op 650) 

v_add_u32 v13, v16, v11 // (op 650) 


s_mul_i32 s1, s26, s13 // (op 652) most significant: low of low * high
s_mul_i32 s14, s27, s12 // (op 652) most significant: low of high * low
s_mul_i32 s0, s26, s12 // (op 652) least significant
s_mul_hi_u32 s15, s26, s12 // (op 652) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 652) most significant: sum
s_add_u32 s1, s1, s15 // (op 652) most significant: sum


s_mov_b32 s52, s28 // (op 652) reg expression
s_mov_b32 s53, s29 // (op 652) reg expression
s_mov_b32 s54, s0 // (op 652) reg expression
s_mov_b32 s55, 131072 // (op 652) call()

v_lshlrev_b32 v11, 4, v1 // (op 743) 
v_mul_lo_u32 v16, v11, 6 // (op 743) 

v_lshrrev_b32 v11, 3, v16 // (op 743) 

v_and_b32 v16, 268435455, v1 // (op 743) 
v_lshlrev_b32 v17, 2, v16 // (op 743) 

v_add_u32 v16, v11, v17 // (op 743) 


v_and_b32 v11, -8, v1 // (op 796) 
v_sub_u32 v17, v1, v11 // (op 796) 

v_lshlrev_b32 v11, 4, v17 // (op 796) 

v_lshrrev_b32 v17, 3, v1 // (op 796) 
v_lshl_add_u32 v18, v17, 7, v11 // (op 796) 


s_nop 0
v_readfirstlane_b32 s0, v18 // (op 796) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v11, -4, v1 // (op 848) 
v_sub_u32 v17, v1, v11 // (op 848) 

v_and_b32 v11, 4294967292, v1 // (op 848) 
v_add_u32 v18, v11, v17 // (op 848) 


s_ashr_i64 s[14:15], s[6:7], 7 // (op 166) 
s_and_b32 s1, s14, 4294967294 // (op 166) 

s_ashr_i64 s[14:15], s[6:7], 7 // (op 165) 
s_mov_b32 s18, s14 // (op 165) convert

s_cmp_lt_u32 s1, s18 // (op 165) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoopTail_165 // (op 165) Condition: Top (jump to RR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoopTail_165 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoopTail_165:


s_add_u32 s1, s1, 1 // (op 167) 
v_mov_b32 v11, v10 // (op 170) 
v_mov_b32 v17, v11 // (op 170) reg expression

buffer_load_dwordx3 v[20:22], v17, s[40:43], 0 offen // (op 170) Load value


s_lshl_b64 s[14:15], s[2:3], 7 // (op 445) 
s_mul_i32 s19, s15, 6 // (op 445) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 445) least significant
s_mul_hi_u32 s38, s14, 6 // (op 445) most significant: high of low * low
s_add_u32 s19, s19, s38 // (op 445) most significant: sum


s_lshr_b64 s[14:15], s[18:19], 3 // (op 445) 

v_add_u32 v10, v10, s14 // (op 445) 

s_barrier  // (op 1036) 
s_barrier  // (op 1049) 




s_lshl_b64 s[14:15], s[8:9], 4 // (op 425) 
v_mov_b64 v[24:25], s[14:15] // (op 425) convert


v_mov_b64 v[26:27], v[14:15] // (op 425) 
v_mov_b64 v[28:29], v[26:27] // (op 425) 
s_mov_b32 m0, s0 // (op 425) reg expression
buffer_load_dwordx4 v28, s[44:47], 0 offen offset: 0 lds // (op 425) Load value direct to lds
v_add_co_u32 v28, s[14:15], v28, v24 // (op 425) least significant half
v_addc_co_u32 v29, s[14:15], 0, v29, s[14:15] // (op 425) most significant half

s_add_u32 m0, m0, 2048 // (op 425) 
buffer_load_dwordx4 v28, s[44:47], 0 offen offset: 0 lds // (op 425) Load value direct to lds



v_add_co_u32 v14, vcc, 128, v14 // (op 514) least significant half
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 514) most significant half
s_waitcnt vmcnt(2)
ds_write_b96 v16, v[20:22] offset:8320 // (op 171) Store local data 



v_mov_b32 v11, v13 // (op 184) 

buffer_load_ubyte v17, v11, s[52:55], 0 offen offset: 0 // (op 184) Load value

v_add_u32 v13, 4, v13 // (op 642) 
s_barrier  // (op 1062) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1039) 






ds_read_b96_tr_b6 v[20:22], v0 offset:8320 // (op 173) Transpose load local data 
ds_read_b96_tr_b6 v[23:25], v0 offset:8576 // (op 173) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v18, v17 offset:8192 // (op 182) Store local data 

s_barrier  // (op 1052) 



ds_read_b128 v[26:29], v4 // (op 175) Load local data 
ds_read_b128 v[30:33], v4 offset:64 // (op 175) Load local data 

v_mov_b64 v[34:35], s[20:21] // (op 177) convert

v_mov_b32 v11, v12 // (op 177) 
v_mov_b32 v17, v11 // (op 177) reg expression
v_mov_b32 v19, v17 // (op 177) 
buffer_load_ubyte v36, v19, s[48:51], 0 offen offset: 0 // (op 177) Load value




s_lshl_b32 s14, s20, 2 // (op 578) 
v_add_u32 v12, v12, s14 // (op 578) 

s_waitcnt lgkmcnt(2)
s_barrier  // (op 1066) 



ds_read_u8 v11, v3 offset:8192 // (op 179) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:25], v[26:33], a[0:3], v36, v11 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 180) 




s_ashr_i64 s[14:15], s[6:7], 7 // (op 165) 
s_mov_b32 s18, s14 // (op 165) convert

s_cmp_lt_u32 s1, s18 // (op 165) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoopTail_165 // (op 165) Condition: Bottom (jump to RR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoopTail_165 if true)
RR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoopTail_165:






















s_barrier  // (op 1130) 






v_cmp_eq_f32 vcc, 0.00000, v8 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalFalse_0_beta_125
s_nop 0
v_accvgpr_read v4, a0 // (op 46) Wait state hazard: XDL Write Hazard
v_accvgpr_read v5, a1 // (op 46) 
v_accvgpr_read v6, a2 // (op 46) 
v_accvgpr_read v7, a3 // (op 46) 
v_mul_f32 v10, v9, v4 // (op 46) 
v_mul_f32 v11, v9, v5 // (op 46) 
v_mul_f32 v12, v9, v6 // (op 46) 
v_mul_f32 v13, v9, v7 // (op 46) 

v_lshrrev_b32 v4, 16, v10 // (op 52) 
v_lshrrev_b32 v5, 16, v11 // (op 52) 
v_lshrrev_b32 v6, 16, v12 // (op 52) 
v_lshrrev_b32 v7, 16, v13 // (op 52) 

v_lshrrev_b32 v0, 7, v1 // (op 890) 
v_and_b32 v3, 63, v1 // (op 890) 
v_lshrrev_b32 v10, 4, v3 // (op 890) 
v_lshlrev_b32 v11, 2, v10 // (op 890) 

v_lshl_add_u32 v10, v0, 4, v11 // (op 890) 


v_lshl_add_u32 v0, s16, 4, v10 // (op 890) 

v_lshrrev_b32 v10, 2, v3 // (op 890) 

v_and_b32 v3, 3, v10 // (op 890) 

v_and_b32 v10, 3, v1 // (op 890) 
v_lshl_add_u32 v11, v3, 2, v10 // (op 890) 


v_lshrrev_b32 v3, 6, v1 // (op 890) 
v_and_b32 v10, 1, v3 // (op 890) 

v_lshl_add_u32 v3, v10, 4, v11 // (op 890) 


v_lshl_add_u32 v10, s17, 5, v3 // (op 890) 

v_mul_lo_u32 v3, v10, s30 // (op 890) 

v_add_lshl_u32 v10, v0, v3, 1 // (op 890) 


s_mul_i32 s15, s30, s13 // (op 892) most significant: low of low * high
s_mul_i32 s0, s31, s12 // (op 892) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 892) least significant
s_mul_hi_u32 s1, s30, s12 // (op 892) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 892) most significant: sum
s_add_u32 s15, s15, s1 // (op 892) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 1 // (op 892) 

s_mov_b32 s8, s32 // (op 892) reg expression
s_mov_b32 s9, s33 // (op 892) reg expression
s_mov_b32 s10, s0 // (op 892) reg expression
s_mov_b32 s11, 131072 // (op 892) call()


v_pack_b32_f16 v12, v4, v5 // (op 54) 
v_pack_b32_f16 v13, v6, v7 // (op 54) 


buffer_store_dwordx2 v[12:13], v10, s[8:11], 0 offen // (op 54) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalBottom_0_beta_125
RR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalFalse_0_beta_125:
v_lshrrev_b32 v0, 7, v1 // (op 713) 
v_and_b32 v3, 63, v1 // (op 713) 
v_lshrrev_b32 v4, 4, v3 // (op 713) 
v_lshlrev_b32 v5, 2, v4 // (op 713) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 713) 


v_lshl_add_u32 v0, s16, 4, v4 // (op 713) 

v_lshrrev_b32 v4, 2, v3 // (op 713) 

v_and_b32 v3, 3, v4 // (op 713) 

v_and_b32 v4, 3, v1 // (op 713) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 713) 


v_lshrrev_b32 v3, 6, v1 // (op 713) 
v_and_b32 v4, 1, v3 // (op 713) 

v_lshl_add_u32 v3, v4, 4, v5 // (op 713) 


v_lshl_add_u32 v4, s17, 5, v3 // (op 713) 

v_mul_lo_u32 v3, v4, s34 // (op 713) 

v_add_lshl_u32 v4, v0, v3, 1 // (op 713) 


s_mul_i32 s15, s34, s13 // (op 715) most significant: low of low * high
s_mul_i32 s0, s35, s12 // (op 715) most significant: low of high * low
s_mul_i32 s14, s34, s12 // (op 715) least significant
s_mul_hi_u32 s1, s34, s12 // (op 715) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 715) most significant: sum
s_add_u32 s15, s15, s1 // (op 715) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 1 // (op 715) 

s_mov_b32 s4, s36 // (op 715) reg expression
s_mov_b32 s5, s37 // (op 715) reg expression
s_mov_b32 s6, s0 // (op 715) reg expression
s_mov_b32 s7, 131072 // (op 715) call()





buffer_load_dwordx2 v[6:7], v4, s[4:7], 0 offen // (op 138) Load value



v_accvgpr_read v10, a0 // (op 134) 
v_accvgpr_read v11, a1 // (op 134) 
v_accvgpr_read v12, a2 // (op 134) 
v_accvgpr_read v13, a3 // (op 134) 
v_mul_f32 v14, v9, v10 // (op 134) 
v_mul_f32 v15, v9, v11 // (op 134) 
v_mul_f32 v16, v9, v12 // (op 134) 
v_mul_f32 v17, v9, v13 // (op 134) 

s_waitcnt vmcnt(0)
v_and_b32 v4, 65535, v6 // (op 139) 
v_lshrrev_b32 v5, 16, v6 // (op 139) 
v_lshlrev_b32 v4, 16, v4 // (op 139) 
v_lshlrev_b32 v5, 16, v5 // (op 139) 
v_mul_f32 v10, v8, v4 // (op 139) 
v_mul_f32 v11, v8, v5 // (op 139) 
v_and_b32 v4, 65535, v7 // (op 139) 
v_lshrrev_b32 v5, 16, v7 // (op 139) 
v_lshlrev_b32 v4, 16, v4 // (op 139) 
v_lshlrev_b32 v5, 16, v5 // (op 139) 
v_mul_f32 v12, v8, v4 // (op 139) 
v_mul_f32 v13, v8, v5 // (op 139) 


v_add_f32 v4, v10, v14 // (op 135) 
v_add_f32 v5, v11, v15 // (op 135) 
v_add_f32 v6, v12, v16 // (op 135) 
v_add_f32 v7, v13, v17 // (op 135) 


v_lshrrev_b32 v10, 16, v4 // (op 136) 
v_lshrrev_b32 v11, 16, v5 // (op 136) 
v_lshrrev_b32 v12, 16, v6 // (op 136) 
v_lshrrev_b32 v13, 16, v7 // (op 136) 

v_lshrrev_b32 v0, 7, v1 // (op 907) 
v_and_b32 v3, 63, v1 // (op 907) 
v_lshrrev_b32 v4, 4, v3 // (op 907) 
v_lshlrev_b32 v5, 2, v4 // (op 907) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 907) 


v_lshl_add_u32 v0, s16, 4, v4 // (op 907) 

v_lshrrev_b32 v4, 2, v3 // (op 907) 

v_and_b32 v3, 3, v4 // (op 907) 

v_and_b32 v4, 3, v1 // (op 907) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 907) 


v_lshrrev_b32 v3, 6, v1 // (op 907) 
v_and_b32 v4, 1, v3 // (op 907) 

v_lshl_add_u32 v3, v4, 4, v5 // (op 907) 


v_lshl_add_u32 v4, s17, 5, v3 // (op 907) 

v_mul_lo_u32 v3, v4, s30 // (op 907) 

v_add_lshl_u32 v4, v0, v3, 1 // (op 907) 


s_mul_i32 s15, s30, s13 // (op 909) most significant: low of low * high
s_mul_i32 s0, s31, s12 // (op 909) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 909) least significant
s_mul_hi_u32 s1, s30, s12 // (op 909) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 909) most significant: sum
s_add_u32 s15, s15, s1 // (op 909) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 1 // (op 909) 

s_mov_b32 s8, s32 // (op 909) reg expression
s_mov_b32 s9, s33 // (op 909) reg expression
s_mov_b32 s10, s0 // (op 909) reg expression
s_mov_b32 s11, 131072 // (op 909) call()


v_pack_b32_f16 v6, v10, v11 // (op 137) 
v_pack_b32_f16 v7, v12, v13 // (op 137) 


buffer_store_dwordx2 v[6:7], v4, s[8:11], 0 offen // (op 137) Store value






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalBottom_0_beta_125:




s_endpgm  // End of RR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2
.LRR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_end:
.size RR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2, .LRR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_end-RR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2
  .amdhsa_next_free_vgpr 56
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 12544 // lds bytes
  .amdhsa_accum_offset 52
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_user_sgpr_kernarg_preload_length 14
  .amdhsa_user_sgpr_kernarg_preload_offset 0
.amdhsa_system_sgpr_workgroup_id_x 1
.amdhsa_system_sgpr_workgroup_id_y 1
.amdhsa_system_sgpr_workgroup_id_z 0
.amdhsa_system_sgpr_workgroup_info 0
.amdhsa_system_vgpr_workitem_id 1
.end_amdhsa_kernel

.amdgpu_metadata
---
custom.config:
  Source:
    Origin: rocroller
  Version: 1.0.0
  Features:
    SupportsUserArgs: false
    SupportsBias: false
    SupportsActivation: false
    SupportsScaleAlpha: false
    SupportsGSU: false
  InternalSupportParams:
    KernArgsVersion: 0
  MXScaleFormat: NoSwizzle
  ProblemType:
    OperationType: GEMM
    DataType: F6
    DataTypeA: F6
    DataTypeB: B8
    DestDataType: b
    ComputeDataType: s
    HighPrecisionAccumulate: True
    TransposeA: False
    TransposeB: False
    UseBeta: True
    Batched: True
    UseBias: 0
    Activation: False
    UseScaleAlphaVec: 0
    MXBlockA: 32
    MXBlockB: 32
  CustomKernel:
    args: [ { type: int64, semantic: StrideA0 },
            { type: address, semantic: AddressA },
            { type: int64, semantic: SizeSum },
            { type: int64, semantic: StrideB0 },
            { type: address, semantic: AddressB },
            { type: int64, semantic: SizeFree1 },
            { type: float32, semantic: Beta },
            { type: float32, semantic: Alpha },
            { type: int64, semantic: StrideScaleA0 },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: SizeSumScaleA },
            { type: int64, semantic: StrideScaleB0 },
            { type: address, semantic: AddressMXScaleB },
            { type: int64, semantic: StrideD0 },
            { type: address, semantic: AddressD },
            { type: int64, semantic: StrideC0 },
            { type: address, semantic: AddressC } ]
    macrotile: [16, 32, 128]
    threads: [128, 1, 1]
    grid: [TilesX, TilesY, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2
    .symbol: RR_GEMM_NN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 12544
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 56
    .vgpr_count: 50
    .agpr_count: 4
    .max_flat_workgroup_size: 128
    .wavefront_size: 64
    .args:
      - .name: Tensor_0_stride_1_0
        .size: 8
        .offset: 0
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 8
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_1_1
        .size: 8
        .offset: 16
        .value_kind: by_value
      - .name: Tensor_2_stride_1_3
        .size: 8
        .offset: 24
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 32
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_2_size_1_4
        .size: 8
        .offset: 40
        .value_kind: by_value
      - .name: user_Float_Value_14
        .size: 4
        .offset: 48
        .value_kind: by_value
      - .name: user_Float_Value_12
        .size: 4
        .offset: 52
        .value_kind: by_value
      - .name: Tensor_4_stride_1_6
        .size: 8
        .offset: 56
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 64
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_size_1_7
        .size: 8
        .offset: 72
        .value_kind: by_value
      - .name: Tensor_7_stride_1_9
        .size: 8
        .offset: 80
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 88
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_21_stride_1_16
        .size: 8
        .offset: 96
        .value_kind: by_value
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 104
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_10_stride_1_11
        .size: 8
        .offset: 112
        .value_kind: by_value
      - .name: Tensor_10_pointer
        .size: 8
        .offset: 120
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
...

.end_amdgpu_metadata

