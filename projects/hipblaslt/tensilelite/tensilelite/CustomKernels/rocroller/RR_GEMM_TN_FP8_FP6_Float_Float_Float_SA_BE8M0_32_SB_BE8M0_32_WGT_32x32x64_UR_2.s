// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT



.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2
.p2align 8
.type RR_GEMM_TN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2,@function
RR_GEMM_TN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_exec_begin:


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


v_and_b32 v0, 63, v1 // (op 787) 
v_lshrrev_b32 v3, 4, v0 // (op 787) 
v_and_b32 v4, 1, v3 // (op 787) 

v_and_b32 v3, 15, v1 // (op 787) 
v_lshl_add_u32 v5, v4, 4, v3 // (op 787) 


v_lshrrev_b32 v3, 6, v1 // (op 787) 
v_lshl_add_u32 v4, v3, 5, v5 // (op 787) 


v_lshrrev_b32 v3, 5, v0 // (op 787) 

v_lshlrev_b32 v0, 4, v3 // (op 787) 

v_lshl_add_u32 v3, v4, 6, v0 // (op 787) 


v_and_b32 v0, 63, v1 // (op 840) 
v_lshrrev_b32 v4, 4, v0 // (op 840) 

v_and_b32 v0, 1, v4 // (op 840) 

v_and_b32 v4, 15, v1 // (op 840) 
v_lshl_add_u32 v5, v0, 4, v4 // (op 840) 


v_and_b32 v0, 32, v1 // (op 840) 
v_lshl_add_u32 v4, v5, 6, v0 // (op 840) 


v_mul_lo_u32 v0, v4, 6 // (op 840) 

v_lshrrev_b32 v4, 3, v0 // (op 840) 

v_lshrrev_b32 v0, 6, v1 // (op 893) 
v_and_b32 v5, 31, v1 // (op 893) 
v_lshl_add_u32 v6, v0, 5, v5 // (op 893) 


v_and_b32 v0, 63, v1 // (op 893) 
v_lshrrev_b32 v5, 5, v0 // (op 893) 

v_lshl_add_u32 v0, v6, 1, v5 // (op 893) 


v_and_b32 v5, 31, v1 // (op 946) 
v_and_b32 v6, 63, v1 // (op 946) 
v_lshrrev_b32 v7, 5, v6 // (op 946) 

v_lshl_add_u32 v6, v5, 1, v7 // (op 946) 


v_and_b32 v5, 63, v1 // (op 1019) 
v_lshrrev_b32 v7, 4, v5 // (op 1019) 
v_and_b32 v8, 1, v7 // (op 1019) 

v_and_b32 v7, 15, v1 // (op 1019) 
v_lshl_add_u32 v9, v8, 4, v7 // (op 1019) 


v_lshrrev_b32 v7, 6, v1 // (op 1019) 
v_lshl_add_u32 v8, v7, 5, v9 // (op 1019) 


v_lshrrev_b32 v7, 5, v5 // (op 1019) 

v_lshlrev_b32 v5, 4, v7 // (op 1019) 

v_lshl_add_u32 v7, v8, 6, v5 // (op 1019) 


v_and_b32 v5, 63, v1 // (op 1043) 
v_lshrrev_b32 v8, 4, v5 // (op 1043) 

v_and_b32 v5, 1, v8 // (op 1043) 

v_and_b32 v8, 15, v1 // (op 1043) 
v_lshl_add_u32 v9, v5, 4, v8 // (op 1043) 


v_and_b32 v5, 32, v1 // (op 1043) 
v_lshl_add_u32 v8, v9, 6, v5 // (op 1043) 


v_mul_lo_u32 v5, v8, 6 // (op 1043) 

v_lshrrev_b32 v8, 3, v5 // (op 1043) 

v_lshrrev_b32 v5, 6, v1 // (op 1067) 
v_and_b32 v9, 31, v1 // (op 1067) 
v_lshl_add_u32 v10, v5, 5, v9 // (op 1067) 


v_and_b32 v5, 63, v1 // (op 1067) 
v_lshrrev_b32 v9, 5, v5 // (op 1067) 

v_lshl_add_u32 v5, v10, 1, v9 // (op 1067) 


v_and_b32 v9, 31, v1 // (op 1091) 
v_and_b32 v10, 63, v1 // (op 1091) 
v_lshrrev_b32 v11, 5, v10 // (op 1091) 

v_lshl_add_u32 v10, v9, 1, v11 // (op 1091) 


v_mov_b32 v9, s14 // (op 2) Move value
v_mov_b32 v11, s15 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 63) call()
v_accvgpr_write a1, 0.00000 // (op 63) call()
v_accvgpr_write a2, 0.00000 // (op 63) call()
v_accvgpr_write a3, 0.00000 // (op 63) call()
v_accvgpr_write a4, 0.00000 // (op 63) call()
v_accvgpr_write a5, 0.00000 // (op 63) call()
v_accvgpr_write a6, 0.00000 // (op 63) call()
v_accvgpr_write a7, 0.00000 // (op 63) call()
v_accvgpr_write a8, 0.00000 // (op 63) call()
v_accvgpr_write a9, 0.00000 // (op 63) call()
v_accvgpr_write a10, 0.00000 // (op 63) call()
v_accvgpr_write a11, 0.00000 // (op 63) call()
v_accvgpr_write a12, 0.00000 // (op 63) call()
v_accvgpr_write a13, 0.00000 // (op 63) call()
v_accvgpr_write a14, 0.00000 // (op 63) call()
v_accvgpr_write a15, 0.00000 // (op 63) call()
v_lshrrev_b32 v12, 2, v1 // (op 498) 
v_lshl_add_u32 v13, s16, 5, v12 // (op 498) 

v_and_b32 v12, -4, v1 // (op 498) 
v_sub_u32 v14, v1, v12 // (op 498) 

v_lshlrev_b32 v12, 4, v14 // (op 498) 

v_mov_b32 v15, 0 // (op 498) convert
v_mov_b32 v14, v12 // (op 498) convert

v_mul_lo_u32 v17, v13, s3 // (op 498) most significant: low of low * high
v_mul_hi_u32 v12, v13, s2 // (op 498) most significant: high of low * low
v_mul_lo_u32 v16, v13, s2 // (op 498) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 498) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 498) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 498) most significant half



s_mul_i32 s1, s2, s7 // (op 500) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 500) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 500) least significant
s_mul_hi_u32 s15, s2, s6 // (op 500) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 500) most significant: sum
s_add_u32 s1, s1, s15 // (op 500) most significant: sum


s_mov_b32 s40, s4 // (op 500) reg expression
s_mov_b32 s41, s5 // (op 500) reg expression
s_mov_b32 s42, s0 // (op 500) reg expression
s_mov_b32 s43, 131072 // (op 500) call()

v_lshrrev_b32 v12, 2, v1 // (op 517) 
v_lshl_add_u32 v13, s16, 5, v12 // (op 517) 

v_and_b32 v12, -4, v1 // (op 517) 
v_sub_u32 v14, v1, v12 // (op 517) 

v_lshlrev_b32 v12, 4, v14 // (op 517) 

v_mov_b32 v15, 0 // (op 517) convert
v_mov_b32 v14, v12 // (op 517) convert

v_mul_lo_u32 v19, v13, s3 // (op 517) most significant: low of low * high
v_mul_hi_u32 v12, v13, s2 // (op 517) most significant: high of low * low
v_mul_lo_u32 v18, v13, s2 // (op 517) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 517) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v14 // (op 517) least significant half
v_addc_co_u32 v19, s[0:1], v19, v15, s[0:1] // (op 517) most significant half



s_mul_i32 s1, s2, s7 // (op 519) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 519) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 519) least significant
s_mul_hi_u32 s15, s2, s6 // (op 519) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 519) most significant: sum
s_add_u32 s1, s1, s15 // (op 519) most significant: sum


s_mov_b32 s40, s4 // (op 519) reg expression
s_mov_b32 s41, s5 // (op 519) reg expression
s_mov_b32 s42, s0 // (op 519) reg expression
s_mov_b32 s43, 131072 // (op 519) call()

v_and_b32 v12, -2, v1 // (op 562) 
v_sub_u32 v13, v1, v12 // (op 562) 

v_lshlrev_b32 v12, 5, v13 // (op 562) 

v_mov_b32 v15, 0 // (op 562) convert
v_mov_b32 v14, v12 // (op 562) convert

v_lshrrev_b32 v12, 1, v1 // (op 562) 
v_lshl_add_u32 v13, s17, 5, v12 // (op 562) 

v_mul_lo_u32 v21, v13, s9 // (op 562) most significant: low of low * high
v_mul_hi_u32 v12, v13, s8 // (op 562) most significant: high of low * low
v_mul_lo_u32 v20, v13, s8 // (op 562) least significant: low of low * low
v_add_u32 v21, v21, v12 // (op 562) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v20 // (op 562) least significant half
v_addc_co_u32 v13, s[0:1], v15, v21, s[0:1] // (op 562) most significant half



v_mul_lo_u32 v15, v13, 6 // (op 562) most significant: low of high * low
v_mul_hi_u32 v20, v12, 6 // (op 562) most significant: high of low * low
v_mul_lo_u32 v14, v12, 6 // (op 562) least significant: low of low * low
v_add_u32 v15, v15, v20 // (op 562) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 562) 

v_mov_b32 v14, v12 // (op 562) convert

s_mul_i32 s1, s13, 6 // (op 564) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 564) least significant
s_mul_hi_u32 s14, s12, 6 // (op 564) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 564) most significant: sum

s_mul_i32 s15, s8, s1 // (op 564) most significant: low of low * high
s_mul_i32 s18, s9, s0 // (op 564) most significant: low of high * low
s_mul_i32 s14, s8, s0 // (op 564) least significant
s_mul_hi_u32 s19, s8, s0 // (op 564) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 564) most significant: sum
s_add_u32 s15, s15, s19 // (op 564) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 564) 

s_mov_b32 s44, s10 // (op 564) reg expression
s_mov_b32 s45, s11 // (op 564) reg expression
s_mov_b32 s46, s0 // (op 564) reg expression
s_mov_b32 s47, 131072 // (op 564) call()

v_and_b32 v15, -2, v1 // (op 581) 
v_sub_u32 v12, v1, v15 // (op 581) 

v_lshlrev_b32 v13, 5, v12 // (op 581) 

v_mov_b32 v21, 0 // (op 581) convert
v_mov_b32 v20, v13 // (op 581) convert

v_lshrrev_b32 v15, 1, v1 // (op 581) 
v_lshl_add_u32 v12, s17, 5, v15 // (op 581) 

v_mul_lo_u32 v23, v12, s9 // (op 581) most significant: low of low * high
v_mul_hi_u32 v13, v12, s8 // (op 581) most significant: high of low * low
v_mul_lo_u32 v22, v12, s8 // (op 581) least significant: low of low * low
v_add_u32 v23, v23, v13 // (op 581) most significant: sum


v_add_co_u32 v12, s[0:1], v20, v22 // (op 581) least significant half
v_addc_co_u32 v13, s[0:1], v21, v23, s[0:1] // (op 581) most significant half



v_mul_lo_u32 v21, v13, 6 // (op 581) most significant: low of high * low
v_mul_hi_u32 v15, v12, 6 // (op 581) most significant: high of low * low
v_mul_lo_u32 v20, v12, 6 // (op 581) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 581) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[20:21] // (op 581) 

v_mov_b32 v15, v12 // (op 581) convert

s_mul_i32 s1, s13, 6 // (op 583) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 583) least significant
s_mul_hi_u32 s14, s12, 6 // (op 583) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 583) most significant: sum

s_mul_i32 s15, s8, s1 // (op 583) most significant: low of low * high
s_mul_i32 s18, s9, s0 // (op 583) most significant: low of high * low
s_mul_i32 s14, s8, s0 // (op 583) least significant
s_mul_hi_u32 s19, s8, s0 // (op 583) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 583) most significant: sum
s_add_u32 s15, s15, s19 // (op 583) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 583) 

s_mov_b32 s44, s10 // (op 583) reg expression
s_mov_b32 s45, s11 // (op 583) reg expression
s_mov_b32 s46, s0 // (op 583) reg expression
s_mov_b32 s47, 131072 // (op 583) call()

v_lshrrev_b32 v12, 1, v1 // (op 626) 
v_lshl_add_u32 v13, s16, 5, v12 // (op 626) 

v_and_b32 v12, -2, v1 // (op 626) 
v_sub_u32 v20, v1, v12 // (op 626) 


s_waitcnt lgkmcnt(0)
v_mul_lo_u32 v12, v13, s20 // (op 626) 
v_add_u32 v12, v12, v20 // (op 626) 


s_mul_i32 s1, s20, s7 // (op 628) most significant: low of low * high
s_mul_i32 s14, s21, s6 // (op 628) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 628) least significant
s_mul_hi_u32 s15, s20, s6 // (op 628) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 628) most significant: sum
s_add_u32 s1, s1, s15 // (op 628) most significant: sum


s_mov_b32 s48, s22 // (op 628) reg expression
s_mov_b32 s49, s23 // (op 628) reg expression
s_mov_b32 s50, s0 // (op 628) reg expression
s_mov_b32 s51, 131072 // (op 628) call()

v_lshrrev_b32 v13, 1, v1 // (op 645) 
v_lshl_add_u32 v20, s16, 5, v13 // (op 645) 

v_and_b32 v13, -2, v1 // (op 645) 
v_sub_u32 v21, v1, v13 // (op 645) 

v_mul_lo_u32 v13, v20, s20 // (op 645) 
v_add_u32 v13, v13, v21 // (op 645) 


s_mul_i32 s1, s20, s7 // (op 647) most significant: low of low * high
s_mul_i32 s14, s21, s6 // (op 647) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 647) least significant
s_mul_hi_u32 s15, s20, s6 // (op 647) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 647) most significant: sum
s_add_u32 s1, s1, s15 // (op 647) most significant: sum


s_mov_b32 s48, s22 // (op 647) reg expression
s_mov_b32 s49, s23 // (op 647) reg expression
s_mov_b32 s50, s0 // (op 647) reg expression
s_mov_b32 s51, 131072 // (op 647) call()

v_and_b32 v20, -2, v1 // (op 690) 
v_sub_u32 v21, v1, v20 // (op 690) 

v_lshrrev_b32 v20, 1, v1 // (op 690) 
v_lshl_add_u32 v22, s17, 5, v20 // (op 690) 

v_mul_lo_u32 v20, v22, s24 // (op 690) 

v_add_u32 v22, v21, v20 // (op 690) 


s_mul_i32 s1, s24, s13 // (op 692) most significant: low of low * high
s_mul_i32 s14, s25, s12 // (op 692) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 692) least significant
s_mul_hi_u32 s15, s24, s12 // (op 692) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 692) most significant: sum
s_add_u32 s1, s1, s15 // (op 692) most significant: sum


s_mov_b32 s52, s26 // (op 692) reg expression
s_mov_b32 s53, s27 // (op 692) reg expression
s_mov_b32 s54, s0 // (op 692) reg expression
s_mov_b32 s55, 131072 // (op 692) call()

v_and_b32 v20, -2, v1 // (op 709) 
v_sub_u32 v21, v1, v20 // (op 709) 

v_lshrrev_b32 v20, 1, v1 // (op 709) 
v_lshl_add_u32 v23, s17, 5, v20 // (op 709) 

v_mul_lo_u32 v20, v23, s24 // (op 709) 

v_add_u32 v23, v21, v20 // (op 709) 


s_mul_i32 s1, s24, s13 // (op 711) most significant: low of low * high
s_mul_i32 s14, s25, s12 // (op 711) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 711) least significant
s_mul_hi_u32 s15, s24, s12 // (op 711) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 711) most significant: sum
s_add_u32 s1, s1, s15 // (op 711) most significant: sum


s_mov_b32 s52, s26 // (op 711) reg expression
s_mov_b32 s53, s27 // (op 711) reg expression
s_mov_b32 s54, s0 // (op 711) reg expression
s_mov_b32 s55, 131072 // (op 711) call()




buffer_load_dwordx4 v[24:27], v14, s[44:47], 0 offen // (op 366) Load value
buffer_load_dwordx2 v[28:29], v14, s[44:47], 0 offen offset:16 // (op 366) Load value



buffer_load_ubyte v20, v22, s[52:55], 0 offen offset: 0 // (op 369) Load value
v_and_b32 v21, -4, v1 // (op 775) 
v_sub_u32 v30, v1, v21 // (op 775) 

v_lshlrev_b32 v21, 4, v30 // (op 775) 

v_lshrrev_b32 v30, 2, v1 // (op 775) 
v_lshl_add_u32 v31, v30, 6, v21 // (op 775) 


s_nop 0
v_readfirstlane_b32 s0, v31 // (op 775) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[2:3], 4 // (op 454) 
v_mov_b64 v[30:31], s[14:15] // (op 454) convert


v_mov_b64 v[32:33], v[16:17] // (op 454) 
s_mov_b32 m0, s0 // (op 454) reg expression
buffer_load_dwordx4 v32, s[40:43], 0 offen offset: 0 lds // (op 454) Load value direct to lds
v_add_co_u32 v32, s[14:15], v32, v30 // (op 454) least significant half
v_addc_co_u32 v33, s[14:15], 0, v33, s[14:15] // (op 454) most significant half

s_add_u32 m0, m0, 1024 // (op 454) 
buffer_load_dwordx4 v32, s[40:43], 0 offen offset: 0 lds // (op 454) Load value direct to lds






buffer_load_ubyte v21, v12, s[48:51], 0 offen offset: 0 // (op 375) Load value



buffer_load_dwordx4 v[30:33], v14, s[44:47], 0 offen offset:48 // (op 378) Load value
buffer_load_dwordx2 v[34:35], v14, s[44:47], 0 offen offset:64 // (op 378) Load value




buffer_load_ubyte v14, v22, s[52:55], 0 offen offset:2 // (op 381) Load value

v_and_b32 v22, -4, v1 // (op 1007) 
v_sub_u32 v36, v1, v22 // (op 1007) 

v_lshlrev_b32 v22, 4, v36 // (op 1007) 

v_lshrrev_b32 v36, 2, v1 // (op 1007) 
v_lshl_add_u32 v37, v36, 6, v22 // (op 1007) 


s_nop 0
v_readfirstlane_b32 s0, v37 // (op 1007) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[2:3], 4 // (op 459) 
v_mov_b64 v[36:37], s[14:15] // (op 459) convert


v_mov_b64 v[38:39], v[16:17] // (op 459) 
s_add_u32 s1, 2048, s0 // (op 459) 
s_mov_b32 m0, s1 // (op 459) reg expression
buffer_load_dwordx4 v38, s[40:43], 64 offen offset: 0 lds // (op 459) Load value direct to lds
v_add_co_u32 v38, s[14:15], v38, v36 // (op 459) least significant half
v_addc_co_u32 v39, s[14:15], 0, v39, s[14:15] // (op 459) most significant half

s_add_u32 m0, m0, 1024 // (op 459) 
buffer_load_dwordx4 v38, s[40:43], 64 offen offset: 0 lds // (op 459) Load value direct to lds








buffer_load_ubyte v22, v12, s[48:51], 0 offen offset:2 // (op 387) Load value

v_and_b32 v12, -2, v1 // (op 828) 
v_sub_u32 v16, v1, v12 // (op 828) 

v_lshlrev_b32 v12, 5, v16 // (op 828) 

v_lshrrev_b32 v16, 1, v1 // (op 828) 
v_lshl_add_u32 v17, v16, 6, v12 // (op 828) 


v_mul_lo_u32 v12, v17, 6 // (op 828) 

v_lshrrev_b32 v16, 3, v12 // (op 828) 




s_waitcnt vmcnt(11)
ds_write_b128 v16, v[24:27] offset:4096 // (op 389) Store local data 
s_waitcnt vmcnt(10)
ds_write_b64 v16, v[28:29] offset:4112 // (op 389) Store local data 

v_and_b32 v12, -2, v1 // (op 934) 
v_sub_u32 v16, v1, v12 // (op 934) 

v_and_b32 v12, 4294967294, v1 // (op 934) 
v_add_u32 v17, v12, v16 // (op 934) 





s_waitcnt vmcnt(9)
ds_write_b8 v17, v20 offset:5632 // (op 390) Store local data 

v_and_b32 v12, -2, v1 // (op 881) 
v_sub_u32 v16, v1, v12 // (op 881) 

v_and_b32 v12, 4294967294, v1 // (op 881) 
v_add_u32 v17, v12, v16 // (op 881) 





s_waitcnt vmcnt(6)
ds_write_b8 v17, v21 offset:5696 // (op 392) Store local data 

v_and_b32 v12, -4, v1 // (op 747) 
v_sub_u32 v16, v1, v12 // (op 747) 

v_lshlrev_b32 v12, 4, v16 // (op 747) 

v_lshrrev_b32 v16, 2, v1 // (op 747) 
v_lshl_add_u32 v17, v16, 6, v12 // (op 747) 


s_nop 0
v_readfirstlane_b32 s0, v17 // (op 747) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -2, v1 // (op 799) 
v_sub_u32 v16, v1, v12 // (op 799) 

v_lshlrev_b32 v12, 5, v16 // (op 799) 

v_lshrrev_b32 v16, 1, v1 // (op 799) 
v_lshl_add_u32 v17, v16, 6, v12 // (op 799) 


v_mul_lo_u32 v12, v17, 6 // (op 799) 

v_lshrrev_b32 v16, 3, v12 // (op 799) 

v_and_b32 v12, -2, v1 // (op 852) 
v_sub_u32 v17, v1, v12 // (op 852) 

v_and_b32 v12, 4294967294, v1 // (op 852) 
v_add_u32 v36, v12, v17 // (op 852) 


v_and_b32 v12, -2, v1 // (op 905) 
v_sub_u32 v17, v1, v12 // (op 905) 

v_and_b32 v12, 4294967294, v1 // (op 905) 
v_add_u32 v37, v12, v17 // (op 905) 


v_and_b32 v12, -4, v1 // (op 992) 
v_sub_u32 v17, v1, v12 // (op 992) 

v_lshlrev_b32 v12, 4, v17 // (op 992) 

v_lshrrev_b32 v17, 2, v1 // (op 992) 
v_lshl_add_u32 v38, v17, 6, v12 // (op 992) 


s_nop 0
v_readfirstlane_b32 s1, v38 // (op 992) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -2, v1 // (op 1031) 
v_sub_u32 v17, v1, v12 // (op 1031) 

v_lshlrev_b32 v12, 5, v17 // (op 1031) 

v_lshrrev_b32 v17, 1, v1 // (op 1031) 
v_lshl_add_u32 v38, v17, 6, v12 // (op 1031) 


v_mul_lo_u32 v12, v38, 6 // (op 1031) 

v_lshrrev_b32 v17, 3, v12 // (op 1031) 

v_and_b32 v12, -2, v1 // (op 1055) 
v_sub_u32 v38, v1, v12 // (op 1055) 

v_and_b32 v12, 4294967294, v1 // (op 1055) 
v_add_u32 v39, v12, v38 // (op 1055) 


v_and_b32 v12, -2, v1 // (op 1079) 
v_sub_u32 v38, v1, v12 // (op 1079) 

v_and_b32 v12, 4294967294, v1 // (op 1079) 
v_add_u32 v40, v12, v38 // (op 1079) 


s_mov_b32 s14, 0 // (op 59) call()
s_ashr_i64 s[18:19], s[28:29], 6 // (op 58) 
s_mov_b32 s15, s18 // (op 58) convert

s_and_b32 s18, s15, 4294967294 // (op 58) 

s_cmp_lt_u32 s14, s18 // (op 58) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoop_58 // (op 58) Condition: Top (jump to RR_GEMM_TN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoop_58 if false)
RR_GEMM_TN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoop_58:


s_barrier  // (op 1128) 
s_add_u32 s14, s14, 2 // (op 60) 



v_mov_b32 v12, v15 // (op 10) 

ds_read_b128 v[42:45], v3 // (op 70) Load local data 
ds_read_b128 v[46:49], v3 offset:32 // (op 70) Load local data 
buffer_load_dwordx4 v[24:27], v12, s[44:47], 0 offen offset:96 // (op 10) Load value
buffer_load_dwordx2 v[28:29], v12, s[44:47], 0 offen offset:112 // (op 10) Load value
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1141) 





v_mov_b32 v38, v23 // (op 25) 

buffer_load_ubyte v20, v38, s[52:55], 0 offen offset:4 // (op 25) Load value
ds_read_b128 v[50:53], v4 offset:4096 // (op 75) Load local data 
ds_read_b64 v[54:55], v4 offset:4112 // (op 75) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1131) 

s_barrier  // (op 1155) 
s_lshl_b64 s[18:19], s[2:3], 4 // (op 438) 



ds_read_u8 v41, v0 offset:5696 // (op 80) Load local data 
v_mov_b64 v[56:57], s[18:19] // (op 438) convert


s_barrier  // (op 1169) 
v_mov_b64 v[58:59], v[18:19] // (op 438) 



v_mov_b64 v[60:61], v[58:59] // (op 438) 
s_mov_b32 m0, s0 // (op 438) reg expression
s_mov_b32 s15, 128 // (op 438) call()
ds_read_u8 v62, v6 offset:5632 // (op 85) Load local data 
buffer_load_dwordx4 v60, s[40:43], s15 offen offset: 0 lds // (op 438) Load value direct to lds
v_add_co_u32 v60, s[18:19], v60, v56 // (op 438) least significant half
v_addc_co_u32 v61, s[18:19], 0, v61, s[18:19] // (op 438) most significant half

s_add_u32 m0, m0, 1024 // (op 438) 
buffer_load_dwordx4 v60, s[40:43], s15 offen offset: 0 lds // (op 438) Load value direct to lds



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[42:49], v[50:55], a[0:15], v41, v62 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 88) 






v_mov_b32 v41, v13 // (op 17) 

buffer_load_ubyte v21, v41, s[48:51], 0 offen offset:4 // (op 17) Load value
s_barrier  // (op 1192) 



s_waitcnt vmcnt(11)
ds_write_b128 v17, v[30:33] offset:5760 // (op 253) Store local data 
s_waitcnt vmcnt(10)
ds_write_b64 v17, v[34:35] offset:5776 // (op 253) Store local data 
s_barrier  // (op 1204) 



s_waitcnt vmcnt(9)
ds_write_b8 v40, v14 offset:7296 // (op 255) Store local data 
s_barrier  // (op 1198) 



s_waitcnt vmcnt(6)
ds_write_b8 v39, v22 offset:7360 // (op 254) Store local data 



s_barrier  // (op 1183) 
buffer_load_dwordx4 v[30:33], v12, s[44:47], 0 offen offset:144 // (op 256) Load value



ds_read_b128 v[42:45], v7 offset:2048 // (op 245) Load local data 
ds_read_b128 v[46:49], v7 offset:2080 // (op 245) Load local data 
buffer_load_dwordx2 v[34:35], v12, s[44:47], 0 offen offset:160 // (op 256) Load value

s_waitcnt lgkmcnt(4)
s_barrier  // (op 1189) 






v_add_u32 v15, 96, v15 // (op 573) 
buffer_load_ubyte v14, v38, s[52:55], 0 offen offset:6 // (op 258) Load value
ds_read_b128 v[50:53], v8 offset:5760 // (op 247) Load local data 

ds_read_b64 v[54:55], v8 offset:5776 // (op 247) Load local data 
v_add_u32 v23, 4, v23 // (op 701) 
s_barrier  // (op 1255) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1195) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1186) 




ds_read_u8 v12, v5 offset:7360 // (op 249) Load local data 
s_lshl_b64 s[18:19], s[2:3], 4 // (op 449) 
v_mov_b64 v[56:57], s[18:19] // (op 449) convert


s_barrier  // (op 1201) 
s_barrier  // (op 1261) 



ds_read_u8 v38, v10 offset:7296 // (op 251) Load local data 
v_mov_b64 v[60:61], v[58:59] // (op 449) 
s_add_u32 s15, 2048, s1 // (op 449) 
s_barrier  // (op 1267) 
s_mov_b32 m0, s15 // (op 449) reg expression
s_mov_b32 s18, 192 // (op 449) call()
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[42:49], v[50:55], a[0:15], v12, v38 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 252) 
buffer_load_dwordx4 v60, s[40:43], s18 offen offset: 0 lds // (op 449) Load value direct to lds




v_add_co_u32 v60, s[38:39], v60, v56 // (op 449) least significant half
v_addc_co_u32 v61, s[38:39], 0, v61, s[38:39] // (op 449) most significant half

s_add_u32 m0, m0, 1024 // (op 449) 
buffer_load_dwordx4 v60, s[40:43], s18 offen offset: 0 lds // (op 449) Load value direct to lds








v_add_co_u32 v18, vcc, 128, v18 // (op 509) least significant half
v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 509) most significant half
buffer_load_ubyte v22, v41, s[48:51], 0 offen offset:6 // (op 257) Load value

s_barrier  // (op 1144) 
v_add_u32 v13, 4, v13 // (op 637) 



s_waitcnt vmcnt(11)
ds_write_b128 v16, v[24:27] offset:4096 // (op 13) Store local data 
s_waitcnt vmcnt(10)
ds_write_b64 v16, v[28:29] offset:4112 // (op 13) Store local data 
s_barrier  // (op 1172) 



s_waitcnt vmcnt(9)
ds_write_b8 v37, v20 offset:5632 // (op 30) Store local data 
s_barrier  // (op 1158) 



s_waitcnt vmcnt(6)
ds_write_b8 v36, v21 offset:5696 // (op 21) Store local data 
s_ashr_i64 s[18:19], s[28:29], 6 // (op 58) 
s_mov_b32 s15, s18 // (op 58) convert

s_and_b32 s18, s15, 4294967294 // (op 58) 

s_cmp_lt_u32 s14, s18 // (op 58) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoop_58 // (op 58) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoop_58 if true)
RR_GEMM_TN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoop_58:
















s_barrier  // (op 1217) 
v_lshrrev_b32 v12, 2, v1 // (op 472) 
s_ashr_i64 s[0:1], s[28:29], 6 // (op 472) 
v_lshl_add_u32 v13, s16, 5, v12 // (op 472) 

s_lshr_b32 s14, s0, 1 // (op 472) 

v_and_b32 v12, -4, v1 // (op 472) 
v_sub_u32 v15, v1, v12 // (op 472) 

v_lshlrev_b32 v12, 4, v15 // (op 472) 

v_lshl_add_u32 v15, s14, 7, v12 // (op 472) 


v_mov_b32 v17, 0 // (op 472) convert
v_mov_b32 v16, v15 // (op 472) convert

v_mul_lo_u32 v19, v13, s3 // (op 472) most significant: low of low * high
v_mul_hi_u32 v12, v13, s2 // (op 472) most significant: high of low * low
v_mul_lo_u32 v18, v13, s2 // (op 472) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 472) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 472) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 472) most significant half



s_mul_i32 s1, s2, s7 // (op 474) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 474) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 474) least significant
s_mul_hi_u32 s15, s2, s6 // (op 474) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 474) most significant: sum
s_add_u32 s1, s1, s15 // (op 474) most significant: sum


s_mov_b32 s40, s4 // (op 474) reg expression
s_mov_b32 s41, s5 // (op 474) reg expression
s_mov_b32 s42, s0 // (op 474) reg expression
s_mov_b32 s43, 131072 // (op 474) call()

v_and_b32 v12, -2, v1 // (op 539) 
s_ashr_i64 s[0:1], s[28:29], 6 // (op 539) 
v_sub_u32 v13, v1, v12 // (op 539) 

s_lshr_b32 s14, s0, 1 // (op 539) 

v_lshlrev_b32 v12, 5, v13 // (op 539) 

v_lshl_add_u32 v13, s14, 7, v12 // (op 539) 


v_mov_b32 v17, 0 // (op 539) convert
v_mov_b32 v16, v13 // (op 539) convert

v_lshrrev_b32 v15, 1, v1 // (op 539) 
v_lshl_add_u32 v12, s17, 5, v15 // (op 539) 

v_mul_lo_u32 v21, v12, s9 // (op 539) most significant: low of low * high
v_mul_hi_u32 v13, v12, s8 // (op 539) most significant: high of low * low
v_mul_lo_u32 v20, v12, s8 // (op 539) least significant: low of low * low
v_add_u32 v21, v21, v13 // (op 539) most significant: sum


v_add_co_u32 v12, s[0:1], v16, v20 // (op 539) least significant half
v_addc_co_u32 v13, s[0:1], v17, v21, s[0:1] // (op 539) most significant half



v_mul_lo_u32 v17, v13, 6 // (op 539) most significant: low of high * low
v_mul_hi_u32 v15, v12, 6 // (op 539) most significant: high of low * low
v_mul_lo_u32 v16, v12, 6 // (op 539) least significant: low of low * low
v_add_u32 v17, v17, v15 // (op 539) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[16:17] // (op 539) 

v_mov_b32 v15, v12 // (op 539) convert

s_mul_i32 s1, s13, 6 // (op 541) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 541) least significant
s_mul_hi_u32 s14, s12, 6 // (op 541) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 541) most significant: sum

s_mul_i32 s15, s8, s1 // (op 541) most significant: low of low * high
s_mul_i32 s18, s9, s0 // (op 541) most significant: low of high * low
s_mul_i32 s14, s8, s0 // (op 541) least significant
s_mul_hi_u32 s19, s8, s0 // (op 541) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 541) most significant: sum
s_add_u32 s15, s15, s19 // (op 541) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 541) 

s_mov_b32 s44, s10 // (op 541) reg expression
s_mov_b32 s45, s11 // (op 541) reg expression
s_mov_b32 s46, s0 // (op 541) reg expression
s_mov_b32 s47, 131072 // (op 541) call()

v_lshrrev_b32 v12, 1, v1 // (op 603) 
s_ashr_i64 s[0:1], s[28:29], 6 // (op 603) 
v_lshl_add_u32 v13, s16, 5, v12 // (op 603) 

s_lshr_b32 s14, s0, 1 // (op 603) 

v_and_b32 v12, -2, v1 // (op 603) 
v_sub_u32 v16, v1, v12 // (op 603) 

v_lshl_add_u32 v12, s14, 2, v16 // (op 603) 


v_mul_lo_u32 v16, v13, s20 // (op 603) 
v_add_u32 v16, v16, v12 // (op 603) 


s_mul_i32 s1, s20, s7 // (op 605) most significant: low of low * high
s_mul_i32 s14, s21, s6 // (op 605) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 605) least significant
s_mul_hi_u32 s15, s20, s6 // (op 605) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 605) most significant: sum
s_add_u32 s1, s1, s15 // (op 605) most significant: sum


s_mov_b32 s48, s22 // (op 605) reg expression
s_mov_b32 s49, s23 // (op 605) reg expression
s_mov_b32 s50, s0 // (op 605) reg expression
s_mov_b32 s51, 131072 // (op 605) call()

v_and_b32 v17, -2, v1 // (op 667) 
s_ashr_i64 s[0:1], s[28:29], 6 // (op 667) 
v_sub_u32 v12, v1, v17 // (op 667) 

s_lshr_b32 s14, s0, 1 // (op 667) 

v_lshrrev_b32 v13, 1, v1 // (op 667) 
v_lshl_add_u32 v17, s14, 2, v12 // (op 667) 


v_lshl_add_u32 v12, s17, 5, v13 // (op 667) 

v_mul_lo_u32 v13, v12, s24 // (op 667) 

v_add_u32 v12, v17, v13 // (op 667) 


s_mul_i32 s1, s24, s13 // (op 669) most significant: low of low * high
s_mul_i32 s14, s25, s12 // (op 669) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 669) least significant
s_mul_hi_u32 s15, s24, s12 // (op 669) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 669) most significant: sum
s_add_u32 s1, s1, s15 // (op 669) most significant: sum


s_mov_b32 s52, s26 // (op 669) reg expression
s_mov_b32 s53, s27 // (op 669) reg expression
s_mov_b32 s54, s0 // (op 669) reg expression
s_mov_b32 s55, 131072 // (op 669) call()

v_and_b32 v13, -4, v1 // (op 760) 
v_sub_u32 v17, v1, v13 // (op 760) 

v_lshlrev_b32 v13, 4, v17 // (op 760) 

v_lshrrev_b32 v17, 2, v1 // (op 760) 
v_lshl_add_u32 v20, v17, 6, v13 // (op 760) 


s_nop 0
v_readfirstlane_b32 s0, v20 // (op 760) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v13, -2, v1 // (op 812) 
v_sub_u32 v17, v1, v13 // (op 812) 

v_lshlrev_b32 v13, 5, v17 // (op 812) 

v_lshrrev_b32 v17, 1, v1 // (op 812) 
v_lshl_add_u32 v20, v17, 6, v13 // (op 812) 


v_mul_lo_u32 v13, v20, 6 // (op 812) 

v_lshrrev_b32 v17, 3, v13 // (op 812) 

v_and_b32 v13, -2, v1 // (op 865) 
v_sub_u32 v20, v1, v13 // (op 865) 

v_and_b32 v13, 4294967294, v1 // (op 865) 
v_add_u32 v21, v13, v20 // (op 865) 


v_and_b32 v13, -2, v1 // (op 918) 
v_sub_u32 v20, v1, v13 // (op 918) 

v_and_b32 v13, 4294967294, v1 // (op 918) 
v_add_u32 v23, v13, v20 // (op 918) 


s_ashr_i64 s[14:15], s[28:29], 6 // (op 172) 
s_and_b32 s1, s14, 4294967294 // (op 172) 

s_ashr_i64 s[14:15], s[28:29], 6 // (op 171) 
s_mov_b32 s18, s14 // (op 171) convert

s_cmp_lt_u32 s1, s18 // (op 171) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoopTail_171 // (op 171) Condition: Top (jump to RR_GEMM_TN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoopTail_171 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoopTail_171:
s_barrier  // (op 1134) 
s_add_u32 s1, s1, 1 // (op 173) 

s_lshl_b64 s[14:15], s[2:3], 4 // (op 443) 
v_mov_b64 v[24:25], s[14:15] // (op 443) convert


v_mov_b64 v[26:27], v[18:19] // (op 443) 
v_mov_b64 v[28:29], v[26:27] // (op 443) 
s_mov_b32 m0, s0 // (op 443) reg expression
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 443) Load value direct to lds
v_add_co_u32 v28, s[14:15], v28, v24 // (op 443) least significant half
v_addc_co_u32 v29, s[14:15], 0, v29, s[14:15] // (op 443) most significant half

s_add_u32 m0, m0, 1024 // (op 443) 
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 443) Load value direct to lds



v_add_co_u32 v18, s[14:15], 64, v18 // (op 462) least significant half
v_addc_co_u32 v19, s[14:15], 0, v19, s[14:15] // (op 462) most significant half



v_mov_b32 v20, v15 // (op 190) 

buffer_load_dwordx4 v[24:27], v20, s[44:47], 0 offen // (op 190) Load value
buffer_load_dwordx2 v[28:29], v20, s[44:47], 0 offen offset:16 // (op 190) Load value

v_add_u32 v15, 48, v15 // (op 531) 


s_barrier  // (op 1147) 
v_mov_b32 v20, v16 // (op 191) 




buffer_load_ubyte v22, v20, s[48:51], 0 offen offset: 0 // (op 191) Load value

v_add_u32 v16, 2, v16 // (op 595) 
s_waitcnt vmcnt(2)
ds_write_b128 v17, v[24:27] offset:4096 // (op 187) Store local data 


v_mov_b32 v20, v12 // (op 192) 

buffer_load_ubyte v13, v20, s[52:55], 0 offen offset: 0 // (op 192) Load value

v_add_u32 v12, 2, v12 // (op 659) 
s_waitcnt vmcnt(2)
ds_write_b64 v17, v[28:29] offset:4112 // (op 187) Store local data 

s_barrier  // (op 1137) 



s_barrier  // (op 1161) 
ds_read_b128 v[24:27], v3 // (op 179) Load local data 



ds_read_b128 v[28:31], v3 offset:32 // (op 179) Load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v21, v22 offset:5696 // (op 188) Store local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 1151) 



s_barrier  // (op 1175) 
ds_read_b128 v[32:35], v4 offset:4096 // (op 181) Load local data 



ds_read_b64 v[36:37], v4 offset:4112 // (op 181) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v23, v13 offset:5632 // (op 189) Store local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 1165) 



ds_read_u8 v20, v0 offset:5696 // (op 183) Load local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1179) 



ds_read_u8 v22, v6 offset:5632 // (op 185) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[24:31], v[32:37], a[0:15], v20, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 186) 




s_ashr_i64 s[14:15], s[28:29], 6 // (op 171) 
s_mov_b32 s18, s14 // (op 171) convert

s_cmp_lt_u32 s1, s18 // (op 171) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoopTail_171 // (op 171) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoopTail_171 if true)
RR_GEMM_TN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoopTail_171:























s_barrier  // (op 1249) 








v_cmp_eq_f32 vcc, 0.00000, v9 // (op 133) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalFalse_0_beta_133 // (op 133) Condition: False, jump to RR_GEMM_TN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalFalse_0_beta_133
s_nop 4
v_accvgpr_read v12, a0 // (op 50) Wait state hazard: XDL Write Hazard
v_accvgpr_read v13, a1 // (op 50) 
v_accvgpr_read v14, a2 // (op 50) 
v_accvgpr_read v15, a3 // (op 50) 
v_accvgpr_read v16, a4 // (op 50) 
v_accvgpr_read v17, a5 // (op 50) 
v_accvgpr_read v18, a6 // (op 50) 
v_accvgpr_read v19, a7 // (op 50) 
v_accvgpr_read v20, a8 // (op 50) 
v_accvgpr_read v21, a9 // (op 50) 
v_accvgpr_read v22, a10 // (op 50) 
v_accvgpr_read v23, a11 // (op 50) 
v_accvgpr_read v24, a12 // (op 50) 
v_accvgpr_read v25, a13 // (op 50) 
v_accvgpr_read v26, a14 // (op 50) 
v_accvgpr_read v27, a15 // (op 50) 
v_mul_f32 v28, v11, v12 // (op 50) 
v_mul_f32 v29, v11, v13 // (op 50) 
v_mul_f32 v30, v11, v14 // (op 50) 
v_mul_f32 v31, v11, v15 // (op 50) 
v_mul_f32 v32, v11, v16 // (op 50) 
v_mul_f32 v33, v11, v17 // (op 50) 
v_mul_f32 v34, v11, v18 // (op 50) 
v_mul_f32 v35, v11, v19 // (op 50) 
v_mul_f32 v36, v11, v20 // (op 50) 
v_mul_f32 v37, v11, v21 // (op 50) 
v_mul_f32 v38, v11, v22 // (op 50) 
v_mul_f32 v39, v11, v23 // (op 50) 
v_mul_f32 v40, v11, v24 // (op 50) 
v_mul_f32 v41, v11, v25 // (op 50) 
v_mul_f32 v42, v11, v26 // (op 50) 
v_mul_f32 v43, v11, v27 // (op 50) 

v_lshrrev_b32 v0, 6, v1 // (op 960) 
v_and_b32 v10, 63, v1 // (op 960) 
v_lshrrev_b32 v3, 5, v10 // (op 960) 
v_lshlrev_b32 v4, 2, v3 // (op 960) 

v_lshl_add_u32 v3, v0, 5, v4 // (op 960) 


v_lshl_add_u32 v0, s16, 5, v3 // (op 960) 

v_lshrrev_b32 v3, 2, v10 // (op 960) 

v_and_b32 v10, 7, v3 // (op 960) 

v_and_b32 v3, 3, v1 // (op 960) 
v_lshl_add_u32 v4, v10, 2, v3 // (op 960) 


v_lshl_add_u32 v3, s17, 5, v4 // (op 960) 

v_mul_lo_u32 v10, v3, s30 // (op 960) 

v_add_lshl_u32 v3, v0, v10, 2 // (op 960) 


s_mul_i32 s15, s30, s13 // (op 962) most significant: low of low * high
s_mul_i32 s0, s31, s12 // (op 962) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 962) least significant
s_mul_hi_u32 s1, s30, s12 // (op 962) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 962) most significant: sum
s_add_u32 s15, s15, s1 // (op 962) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 2 // (op 962) 

s_mov_b32 s8, s32 // (op 962) reg expression
s_mov_b32 s9, s33 // (op 962) reg expression
s_mov_b32 s10, s0 // (op 962) reg expression
s_mov_b32 s11, 131072 // (op 962) call()




buffer_store_dwordx4 v[28:31], v3, s[8:11], 0 offen // (op 56) Store value
buffer_store_dwordx4 v[32:35], v3, s[8:11], 0 offen offset:32 // (op 56) Store value
buffer_store_dwordx4 v[36:39], v3, s[8:11], 0 offen offset:64 // (op 56) Store value
buffer_store_dwordx4 v[40:43], v3, s[8:11], 0 offen offset:96 // (op 56) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalBottom_0_beta_133 // (op 133) Condition: Done, jump to RR_GEMM_TN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalBottom_0_beta_133
RR_GEMM_TN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalFalse_0_beta_133:
v_lshrrev_b32 v0, 6, v1 // (op 730) 
v_and_b32 v10, 63, v1 // (op 730) 
v_lshrrev_b32 v3, 5, v10 // (op 730) 
v_lshlrev_b32 v4, 2, v3 // (op 730) 

v_lshl_add_u32 v3, v0, 5, v4 // (op 730) 


v_lshl_add_u32 v0, s16, 5, v3 // (op 730) 

v_lshrrev_b32 v3, 2, v10 // (op 730) 

v_and_b32 v10, 7, v3 // (op 730) 

v_and_b32 v3, 3, v1 // (op 730) 
v_lshl_add_u32 v4, v10, 2, v3 // (op 730) 


v_lshl_add_u32 v3, s17, 5, v4 // (op 730) 

v_mul_lo_u32 v10, v3, s34 // (op 730) 

v_add_lshl_u32 v3, v0, v10, 2 // (op 730) 


s_mul_i32 s15, s34, s13 // (op 732) most significant: low of low * high
s_mul_i32 s0, s35, s12 // (op 732) most significant: low of high * low
s_mul_i32 s14, s34, s12 // (op 732) least significant
s_mul_hi_u32 s1, s34, s12 // (op 732) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 732) most significant: sum
s_add_u32 s15, s15, s1 // (op 732) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 2 // (op 732) 

s_mov_b32 s4, s36 // (op 732) reg expression
s_mov_b32 s5, s37 // (op 732) reg expression
s_mov_b32 s6, s0 // (op 732) reg expression
s_mov_b32 s7, 131072 // (op 732) call()





buffer_load_dwordx4 v[12:15], v3, s[4:7], 0 offen // (op 145) Load value
buffer_load_dwordx4 v[16:19], v3, s[4:7], 0 offen offset:32 // (op 145) Load value
buffer_load_dwordx4 v[20:23], v3, s[4:7], 0 offen offset:64 // (op 145) Load value
buffer_load_dwordx4 v[24:27], v3, s[4:7], 0 offen offset:96 // (op 145) Load value



v_accvgpr_read v28, a0 // (op 142) 
v_accvgpr_read v29, a1 // (op 142) 
v_accvgpr_read v30, a2 // (op 142) 
v_accvgpr_read v31, a3 // (op 142) 
v_accvgpr_read v32, a4 // (op 142) 
v_accvgpr_read v33, a5 // (op 142) 
v_accvgpr_read v34, a6 // (op 142) 
v_accvgpr_read v35, a7 // (op 142) 
v_accvgpr_read v36, a8 // (op 142) 
v_accvgpr_read v37, a9 // (op 142) 
v_accvgpr_read v38, a10 // (op 142) 
v_accvgpr_read v39, a11 // (op 142) 
v_accvgpr_read v40, a12 // (op 142) 
v_accvgpr_read v41, a13 // (op 142) 
v_accvgpr_read v42, a14 // (op 142) 
v_accvgpr_read v43, a15 // (op 142) 
v_mul_f32 v44, v11, v28 // (op 142) 
v_mul_f32 v45, v11, v29 // (op 142) 
v_mul_f32 v46, v11, v30 // (op 142) 
v_mul_f32 v47, v11, v31 // (op 142) 
v_mul_f32 v48, v11, v32 // (op 142) 
v_mul_f32 v49, v11, v33 // (op 142) 
v_mul_f32 v50, v11, v34 // (op 142) 
v_mul_f32 v51, v11, v35 // (op 142) 
v_mul_f32 v52, v11, v36 // (op 142) 
v_mul_f32 v53, v11, v37 // (op 142) 
v_mul_f32 v54, v11, v38 // (op 142) 
v_mul_f32 v55, v11, v39 // (op 142) 
v_mul_f32 v56, v11, v40 // (op 142) 
v_mul_f32 v57, v11, v41 // (op 142) 
v_mul_f32 v58, v11, v42 // (op 142) 
v_mul_f32 v59, v11, v43 // (op 142) 

s_waitcnt vmcnt(3)
v_mul_f32 v28, v9, v12 // (op 146) 
v_mul_f32 v29, v9, v13 // (op 146) 
v_mul_f32 v30, v9, v14 // (op 146) 
v_mul_f32 v31, v9, v15 // (op 146) 
s_waitcnt vmcnt(2)
v_mul_f32 v32, v9, v16 // (op 146) 
v_mul_f32 v33, v9, v17 // (op 146) 
v_mul_f32 v34, v9, v18 // (op 146) 
v_mul_f32 v35, v9, v19 // (op 146) 
s_waitcnt vmcnt(1)
v_mul_f32 v36, v9, v20 // (op 146) 
v_mul_f32 v37, v9, v21 // (op 146) 
v_mul_f32 v38, v9, v22 // (op 146) 
v_mul_f32 v39, v9, v23 // (op 146) 
s_waitcnt vmcnt(0)
v_mul_f32 v40, v9, v24 // (op 146) 
v_mul_f32 v41, v9, v25 // (op 146) 
v_mul_f32 v42, v9, v26 // (op 146) 
v_mul_f32 v43, v9, v27 // (op 146) 

v_add_f32 v12, v28, v44 // (op 143) 
v_add_f32 v13, v29, v45 // (op 143) 
v_add_f32 v14, v30, v46 // (op 143) 
v_add_f32 v15, v31, v47 // (op 143) 
v_add_f32 v16, v32, v48 // (op 143) 
v_add_f32 v17, v33, v49 // (op 143) 
v_add_f32 v18, v34, v50 // (op 143) 
v_add_f32 v19, v35, v51 // (op 143) 
v_add_f32 v20, v36, v52 // (op 143) 
v_add_f32 v21, v37, v53 // (op 143) 
v_add_f32 v22, v38, v54 // (op 143) 
v_add_f32 v23, v39, v55 // (op 143) 
v_add_f32 v24, v40, v56 // (op 143) 
v_add_f32 v25, v41, v57 // (op 143) 
v_add_f32 v26, v42, v58 // (op 143) 
v_add_f32 v27, v43, v59 // (op 143) 


v_lshrrev_b32 v0, 6, v1 // (op 977) 
v_and_b32 v10, 63, v1 // (op 977) 
v_lshrrev_b32 v3, 5, v10 // (op 977) 
v_lshlrev_b32 v4, 2, v3 // (op 977) 

v_lshl_add_u32 v3, v0, 5, v4 // (op 977) 


v_lshl_add_u32 v0, s16, 5, v3 // (op 977) 

v_lshrrev_b32 v3, 2, v10 // (op 977) 

v_and_b32 v10, 7, v3 // (op 977) 

v_and_b32 v3, 3, v1 // (op 977) 
v_lshl_add_u32 v4, v10, 2, v3 // (op 977) 


v_lshl_add_u32 v3, s17, 5, v4 // (op 977) 

v_mul_lo_u32 v10, v3, s30 // (op 977) 

v_add_lshl_u32 v3, v0, v10, 2 // (op 977) 


s_mul_i32 s15, s30, s13 // (op 979) most significant: low of low * high
s_mul_i32 s0, s31, s12 // (op 979) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 979) least significant
s_mul_hi_u32 s1, s30, s12 // (op 979) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 979) most significant: sum
s_add_u32 s15, s15, s1 // (op 979) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 2 // (op 979) 

s_mov_b32 s8, s32 // (op 979) reg expression
s_mov_b32 s9, s33 // (op 979) reg expression
s_mov_b32 s10, s0 // (op 979) reg expression
s_mov_b32 s11, 131072 // (op 979) call()




buffer_store_dwordx4 v[12:15], v3, s[8:11], 0 offen // (op 144) Store value
buffer_store_dwordx4 v[16:19], v3, s[8:11], 0 offen offset:32 // (op 144) Store value
buffer_store_dwordx4 v[20:23], v3, s[8:11], 0 offen offset:64 // (op 144) Store value
buffer_store_dwordx4 v[24:27], v3, s[8:11], 0 offen offset:96 // (op 144) Store value





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalBottom_0_beta_133:




s_endpgm  // End of RR_GEMM_TN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2
.LRR_GEMM_TN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_end:
.size RR_GEMM_TN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2, .LRR_GEMM_TN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_end-RR_GEMM_TN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2
  .amdhsa_next_free_vgpr 80
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 7424 // lds bytes
  .amdhsa_accum_offset 64
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
  ProblemType:
    OperationType: GEMM
    DataType: F8
    DataTypeA: F8
    DataTypeB: F6
    DestDataType: s
    ComputeDataType: s
    HighPrecisionAccumulate: True
    TransposeA: True
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
            { type: int64, semantic: SizeFree0 },
            { type: int64, semantic: StrideB0 },
            { type: address, semantic: AddressB },
            { type: int64, semantic: SizeFree1 },
            { type: float32, semantic: Beta },
            { type: float32, semantic: Alpha },
            { type: int64, semantic: StrideScaleA0 },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: StrideScaleB0 },
            { type: address, semantic: AddressMXScaleB },
            { type: int64, semantic: SizeSum },
            { type: int64, semantic: StrideD0 },
            { type: address, semantic: AddressD },
            { type: int64, semantic: StrideC0 },
            { type: address, semantic: AddressC } ]
    macrotile: [32, 32, 64]
    threads: [64, 1, 1]
    grid: [TilesX, TilesY, Batch]
  MatrixInstruction: [32, 32, 64, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2
    .symbol: RR_GEMM_TN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 7424
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 56
    .vgpr_count: 63
    .agpr_count: 16
    .max_flat_workgroup_size: 64
    .workgroup_size: [64, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, {is-null: true}]
    .dynamic_sharedmemory_bytes: {type: LiteralValue.UInt32, dataType: UInt32, value: 0}
    .args:
      - .name: Tensor_0_stride_0_0
        .size: 8
        .offset: 0
        .expression: {type: CommandArgument, size: 8, offset: 24, name: Tensor_0_stride_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 8
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: FP8, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP8, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_0_1
        .size: 8
        .offset: 16
        .expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_2_stride_1_3
        .size: 8
        .offset: 24
        .expression: {type: CommandArgument, size: 8, offset: 72, name: Tensor_2_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 32
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: FP6, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP6, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_2_size_1_4
        .size: 8
        .offset: 40
        .expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: user_Float_Value_14
        .size: 4
        .offset: 48
        .expression: {type: CommandArgument, size: 4, offset: 204, name: user_Float_Value_14, variableType: {dataType: Float, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Float, pointerType: Value}
        .value_kind: by_value
      - .name: user_Float_Value_12
        .size: 4
        .offset: 52
        .expression: {type: CommandArgument, size: 4, offset: 200, name: user_Float_Value_12, variableType: {dataType: Float, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Float, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_4_stride_0_6
        .size: 8
        .offset: 56
        .expression: {type: CommandArgument, size: 8, offset: 104, name: Tensor_4_stride_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 64
        .expression: {type: CommandArgument, size: 8, offset: 80, name: Tensor_4_pointer, variableType: {dataType: E8M0, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: E8M0, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_7_stride_1_8
        .size: 8
        .offset: 72
        .expression: {type: CommandArgument, size: 8, offset: 152, name: Tensor_7_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 80
        .expression: {type: CommandArgument, size: 8, offset: 120, name: Tensor_7_pointer, variableType: {dataType: E8M0, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: E8M0, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_1_14
        .size: 8
        .offset: 88
        .expression: {type: CommandArgument, size: 8, offset: 16, name: Tensor_0_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_21_stride_1_16
        .size: 8
        .offset: 96
        .expression: {type: CommandArgument, size: 8, offset: 240, name: Tensor_21_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 104
        .expression: {type: CommandArgument, size: 8, offset: 208, name: Tensor_21_pointer, variableType: {dataType: Float, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Float, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_10_stride_1_10
        .size: 8
        .offset: 112
        .expression: {type: CommandArgument, size: 8, offset: 192, name: Tensor_10_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_10_pointer
        .size: 8
        .offset: 120
        .expression: {type: CommandArgument, size: 8, offset: 160, name: Tensor_10_pointer, variableType: {dataType: Float, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Float, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
...

.end_amdgpu_metadata

