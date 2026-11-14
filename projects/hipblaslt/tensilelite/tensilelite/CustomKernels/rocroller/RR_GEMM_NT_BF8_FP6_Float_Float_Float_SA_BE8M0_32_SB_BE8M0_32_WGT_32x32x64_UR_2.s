// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2
.p2align 8
.type RR_GEMM_NT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2,@function
RR_GEMM_NT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_exec_begin:


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
v_and_b32 v4, -2, v3 // (op 787) 
v_sub_u32 v5, v3, v4 // (op 787) 


v_and_b32 v3, 48, v1 // (op 787) 
v_sub_u32 v4, v0, v3 // (op 787) 

v_and_b32 v3, 1, v4 // (op 787) 
v_lshl_add_u32 v6, v5, 1, v3 // (op 787) 


v_lshlrev_b32 v3, 3, v6 // (op 787) 

v_lshrrev_b32 v5, 6, v1 // (op 787) 
v_lshl_add_u32 v6, v5, 5, v3 // (op 787) 


v_lshrrev_b32 v3, 1, v4 // (op 787) 

v_lshrrev_b32 v4, 5, v0 // (op 787) 

v_lshl_add_u32 v0, v4, 4, v3 // (op 787) 


v_lshl_add_u32 v3, v0, 5, v6 // (op 787) 


v_and_b32 v0, 63, v1 // (op 840) 
v_and_b32 v4, 48, v1 // (op 840) 
v_sub_u32 v5, v0, v4 // (op 840) 

v_lshrrev_b32 v4, 5, v0 // (op 840) 
v_lshl_add_u32 v6, v4, 5, v5 // (op 840) 


v_lshrrev_b32 v4, 4, v0 // (op 840) 

v_and_b32 v0, -2, v4 // (op 840) 
v_sub_u32 v5, v4, v0 // (op 840) 


v_lshlrev_b32 v0, 4, v5 // (op 840) 

v_lshl_add_u32 v4, v6, 5, v0 // (op 840) 


v_mul_lo_u32 v0, v4, 6 // (op 840) 
v_lshrrev_b32 v5, 3, v0 // (op 840) 

v_lshrrev_b32 v0, 4, v4 // (op 840) 

v_lshlrev_b32 v4, 2, v0 // (op 840) 

v_add_u32 v0, v5, v4 // (op 840) 


v_lshrrev_b32 v4, 6, v1 // (op 893) 
v_and_b32 v5, 31, v1 // (op 893) 
v_lshl_add_u32 v6, v4, 5, v5 // (op 893) 


v_and_b32 v4, 32, v1 // (op 893) 
v_add_u32 v5, v4, v6 // (op 893) 


v_and_b32 v4, 32, v1 // (op 946) 
v_and_b32 v6, 31, v1 // (op 946) 
v_add_u32 v7, v4, v6 // (op 946) 


v_and_b32 v4, 63, v1 // (op 1019) 
v_lshrrev_b32 v6, 4, v4 // (op 1019) 
v_and_b32 v8, -2, v6 // (op 1019) 
v_sub_u32 v9, v6, v8 // (op 1019) 


v_and_b32 v6, 48, v1 // (op 1019) 
v_sub_u32 v8, v4, v6 // (op 1019) 

v_and_b32 v6, 1, v8 // (op 1019) 
v_lshl_add_u32 v10, v9, 1, v6 // (op 1019) 


v_lshlrev_b32 v6, 3, v10 // (op 1019) 

v_lshrrev_b32 v9, 6, v1 // (op 1019) 
v_lshl_add_u32 v10, v9, 5, v6 // (op 1019) 


v_lshrrev_b32 v6, 1, v8 // (op 1019) 

v_lshrrev_b32 v8, 5, v4 // (op 1019) 

v_lshl_add_u32 v4, v8, 4, v6 // (op 1019) 


v_lshl_add_u32 v6, v4, 5, v10 // (op 1019) 


v_and_b32 v4, 63, v1 // (op 1043) 
v_and_b32 v8, 48, v1 // (op 1043) 
v_sub_u32 v9, v4, v8 // (op 1043) 

v_lshrrev_b32 v8, 5, v4 // (op 1043) 
v_lshl_add_u32 v10, v8, 5, v9 // (op 1043) 


v_lshrrev_b32 v8, 4, v4 // (op 1043) 

v_and_b32 v4, -2, v8 // (op 1043) 
v_sub_u32 v9, v8, v4 // (op 1043) 


v_lshlrev_b32 v4, 4, v9 // (op 1043) 

v_lshl_add_u32 v8, v10, 5, v4 // (op 1043) 


v_mul_lo_u32 v4, v8, 6 // (op 1043) 
v_lshrrev_b32 v9, 3, v4 // (op 1043) 

v_lshrrev_b32 v4, 4, v8 // (op 1043) 

v_lshlrev_b32 v8, 2, v4 // (op 1043) 

v_add_u32 v4, v9, v8 // (op 1043) 


v_lshrrev_b32 v8, 6, v1 // (op 1067) 
v_and_b32 v9, 31, v1 // (op 1067) 
v_lshl_add_u32 v10, v8, 5, v9 // (op 1067) 


v_and_b32 v8, 32, v1 // (op 1067) 
v_add_u32 v9, v8, v10 // (op 1067) 


v_and_b32 v8, 32, v1 // (op 1091) 
v_and_b32 v10, 31, v1 // (op 1091) 
v_add_u32 v11, v8, v10 // (op 1091) 


v_mov_b32 v8, s14 // (op 2) Move value
v_mov_b32 v10, s15 // (op 43) Move value


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
v_and_b32 v12, -2, v1 // (op 498) 
v_sub_u32 v13, v1, v12 // (op 498) 

v_lshlrev_b32 v12, 4, v13 // (op 498) 

v_lshl_add_u32 v13, s16, 5, v12 // (op 498) 

v_mov_b32 v15, 0 // (op 498) convert
v_mov_b32 v14, v13 // (op 498) convert

v_lshrrev_b32 v12, 1, v1 // (op 498) 
v_mul_lo_u32 v17, v12, s3 // (op 498) most significant: low of low * high
v_mul_hi_u32 v13, v12, s2 // (op 498) most significant: high of low * low
v_mul_lo_u32 v16, v12, s2 // (op 498) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 498) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v16 // (op 498) least significant half
v_addc_co_u32 v13, s[0:1], v15, v17, s[0:1] // (op 498) most significant half



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

v_and_b32 v14, -2, v1 // (op 517) 
v_sub_u32 v15, v1, v14 // (op 517) 

v_lshlrev_b32 v14, 4, v15 // (op 517) 

v_lshl_add_u32 v15, s16, 5, v14 // (op 517) 

v_mov_b32 v17, 0 // (op 517) convert
v_mov_b32 v16, v15 // (op 517) convert

v_lshrrev_b32 v14, 1, v1 // (op 517) 
v_mul_lo_u32 v19, v14, s3 // (op 517) most significant: low of low * high
v_mul_hi_u32 v15, v14, s2 // (op 517) most significant: high of low * low
v_mul_lo_u32 v18, v14, s2 // (op 517) least significant: low of low * low
v_add_u32 v19, v19, v15 // (op 517) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v18 // (op 517) least significant half
v_addc_co_u32 v15, s[0:1], v17, v19, s[0:1] // (op 517) most significant half



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

s_lshl_b32 s0, s17, 5 // (op 562) 
s_mov_b32 s15, 0 // (op 562) convert
s_mov_b32 s14, s0 // (op 562) convert

v_mul_lo_u32 v17, v1, s9 // (op 562) most significant: low of low * high
v_mul_hi_u32 v18, v1, s8 // (op 562) most significant: high of low * low
v_mul_lo_u32 v16, v1, s8 // (op 562) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 562) most significant: sum

v_mov_b32 v18, s14 // (op 562) 
v_mov_b32 v19, s15 // (op 562) 
v_add_co_u32 v16, s[0:1], v16, v18 // (op 562) least significant half
v_addc_co_u32 v17, s[0:1], v17, v19, s[0:1] // (op 562) most significant half




v_mul_lo_u32 v19, v17, 6 // (op 562) most significant: low of high * low
v_mul_hi_u32 v20, v16, 6 // (op 562) most significant: high of low * low
v_mul_lo_u32 v18, v16, 6 // (op 562) least significant: low of low * low
v_add_u32 v19, v19, v20 // (op 562) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 562) 

v_mov_b32 v18, v16 // (op 562) convert

s_mul_i32 s1, s7, 6 // (op 564) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 564) least significant
s_mul_hi_u32 s14, s6, 6 // (op 564) most significant: high of low * low
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

s_lshl_b32 s0, s17, 5 // (op 581) 
s_mov_b32 s15, 0 // (op 581) convert
s_mov_b32 s14, s0 // (op 581) convert

v_mul_lo_u32 v17, v1, s9 // (op 581) most significant: low of low * high
v_mul_hi_u32 v19, v1, s8 // (op 581) most significant: high of low * low
v_mul_lo_u32 v16, v1, s8 // (op 581) least significant: low of low * low
v_add_u32 v17, v17, v19 // (op 581) most significant: sum

v_mov_b32 v19, s14 // (op 581) 
v_mov_b32 v20, s15 // (op 581) 
v_add_co_u32 v16, s[0:1], v16, v19 // (op 581) least significant half
v_addc_co_u32 v17, s[0:1], v17, v20, s[0:1] // (op 581) most significant half




v_mul_lo_u32 v21, v17, 6 // (op 581) most significant: low of high * low
v_mul_hi_u32 v19, v16, 6 // (op 581) most significant: high of low * low
v_mul_lo_u32 v20, v16, 6 // (op 581) least significant: low of low * low
v_add_u32 v21, v21, v19 // (op 581) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[20:21] // (op 581) 

v_mov_b32 v19, v16 // (op 581) convert

s_mul_i32 s1, s7, 6 // (op 583) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 583) least significant
s_mul_hi_u32 s14, s6, 6 // (op 583) most significant: high of low * low
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

v_and_b32 v16, -32, v1 // (op 626) 
v_sub_u32 v17, v1, v16 // (op 626) 

v_lshl_add_u32 v16, s16, 5, v17 // (op 626) 

v_lshrrev_b32 v17, 5, v1 // (op 626) 
v_mul_lo_u32 v20, v17, s12 // (op 626) 

v_add_u32 v17, v16, v20 // (op 626) 



s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 628) most significant: low of low * high
s_mul_i32 s14, s13, s22 // (op 628) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 628) least significant
s_mul_hi_u32 s15, s12, s22 // (op 628) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 628) most significant: sum
s_add_u32 s1, s1, s15 // (op 628) most significant: sum


s_mov_b32 s48, s20 // (op 628) reg expression
s_mov_b32 s49, s21 // (op 628) reg expression
s_mov_b32 s50, s0 // (op 628) reg expression
s_mov_b32 s51, 131072 // (op 628) call()

v_and_b32 v16, -32, v1 // (op 645) 
v_sub_u32 v20, v1, v16 // (op 645) 

v_lshl_add_u32 v16, s16, 5, v20 // (op 645) 

v_lshrrev_b32 v20, 5, v1 // (op 645) 
v_mul_lo_u32 v21, v20, s12 // (op 645) 

v_add_u32 v20, v16, v21 // (op 645) 


s_mul_i32 s1, s12, s23 // (op 647) most significant: low of low * high
s_mul_i32 s14, s13, s22 // (op 647) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 647) least significant
s_mul_hi_u32 s15, s12, s22 // (op 647) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 647) most significant: sum
s_add_u32 s1, s1, s15 // (op 647) most significant: sum


s_mov_b32 s48, s20 // (op 647) reg expression
s_mov_b32 s49, s21 // (op 647) reg expression
s_mov_b32 s50, s0 // (op 647) reg expression
s_mov_b32 s51, 131072 // (op 647) call()

v_and_b32 v16, -32, v1 // (op 690) 
v_sub_u32 v21, v1, v16 // (op 690) 

v_lshl_add_u32 v16, s17, 5, v21 // (op 690) 

v_lshrrev_b32 v21, 5, v1 // (op 690) 
v_mul_lo_u32 v22, v21, s24 // (op 690) 
v_add_u32 v22, v22, v16 // (op 690) 


s_mul_i32 s1, s24, s23 // (op 692) most significant: low of low * high
s_mul_i32 s14, s25, s22 // (op 692) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 692) least significant
s_mul_hi_u32 s15, s24, s22 // (op 692) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 692) most significant: sum
s_add_u32 s1, s1, s15 // (op 692) most significant: sum


s_mov_b32 s52, s26 // (op 692) reg expression
s_mov_b32 s53, s27 // (op 692) reg expression
s_mov_b32 s54, s0 // (op 692) reg expression
s_mov_b32 s55, 131072 // (op 692) call()

v_and_b32 v16, -32, v1 // (op 709) 
v_sub_u32 v21, v1, v16 // (op 709) 

v_lshl_add_u32 v16, s17, 5, v21 // (op 709) 

v_lshrrev_b32 v21, 5, v1 // (op 709) 
v_mul_lo_u32 v23, v21, s24 // (op 709) 
v_add_u32 v23, v23, v16 // (op 709) 


s_mul_i32 s1, s24, s23 // (op 711) most significant: low of low * high
s_mul_i32 s14, s25, s22 // (op 711) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 711) least significant
s_mul_hi_u32 s15, s24, s22 // (op 711) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 711) most significant: sum
s_add_u32 s1, s1, s15 // (op 711) most significant: sum


s_mov_b32 s52, s26 // (op 711) reg expression
s_mov_b32 s53, s27 // (op 711) reg expression
s_mov_b32 s54, s0 // (op 711) reg expression
s_mov_b32 s55, 131072 // (op 711) call()



v_mov_b32 v16, v18 // (op 366) reg expression

buffer_load_dwordx3 v[24:26], v16, s[44:47], 0 offen // (op 366) Load value
buffer_load_dwordx3 v[28:30], v16, s[44:47], 0 offen offset:12 // (op 366) Load value



v_mov_b32 v16, v22 // (op 369) reg expression

buffer_load_ubyte v21, v16, s[52:55], 0 offen offset: 0 // (op 369) Load value

v_and_b32 v16, -2, v1 // (op 775) 
v_sub_u32 v27, v1, v16 // (op 775) 

v_lshlrev_b32 v16, 4, v27 // (op 775) 

v_lshrrev_b32 v27, 1, v1 // (op 775) 
v_lshl_add_u32 v31, v27, 5, v16 // (op 775) 


s_nop 0
v_readfirstlane_b32 s0, v31 // (op 775) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[2:3], 5 // (op 454) 
v_mov_b64 v[32:33], s[14:15] // (op 454) convert


v_mov_b64 v[34:35], v[12:13] // (op 454) reg expression
s_mov_b32 m0, s0 // (op 454) reg expression
buffer_load_dwordx4 v34, s[40:43], 0 offen offset: 0 lds // (op 454) Load value direct to lds
v_add_co_u32 v34, s[14:15], v34, v32 // (op 454) least significant half
v_addc_co_u32 v35, s[14:15], 0, v35, s[14:15] // (op 454) most significant half

s_add_u32 m0, m0, 1024 // (op 454) 
buffer_load_dwordx4 v34, s[40:43], 0 offen offset: 0 lds // (op 454) Load value direct to lds





v_mov_b32 v16, v17 // (op 375) reg expression

buffer_load_ubyte v27, v16, s[48:51], 0 offen offset: 0 // (op 375) Load value



s_lshl_b64 s[0:1], s[8:9], 6 // (op 378) 
s_mul_i32 s15, s1, 6 // (op 378) most significant: low of high * low
s_mul_i32 s14, s0, 6 // (op 378) least significant
s_mul_hi_u32 s18, s0, 6 // (op 378) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 378) most significant: sum


s_lshr_b64 s[0:1], s[14:15], 3 // (op 378) 

v_add_u32 v16, v18, s0 // (op 378) 


buffer_load_dwordx3 v[32:34], v16, s[44:47], 0 offen // (op 378) Load value
buffer_load_dwordx3 v[36:38], v16, s[44:47], 0 offen offset:12 // (op 378) Load value




s_lshl_b32 s0, s24, 1 // (op 381) 
v_add_u32 v16, v22, s0 // (op 381) 


buffer_load_ubyte v18, v16, s[52:55], 0 offen offset: 0 // (op 381) Load value


v_and_b32 v16, -2, v1 // (op 1007) 
v_sub_u32 v22, v1, v16 // (op 1007) 

v_lshlrev_b32 v16, 4, v22 // (op 1007) 

v_lshrrev_b32 v22, 1, v1 // (op 1007) 
v_lshl_add_u32 v31, v22, 5, v16 // (op 1007) 


s_nop 0
v_readfirstlane_b32 s0, v31 // (op 1007) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[2:3], 5 // (op 459) 
v_mov_b64 v[40:41], s[14:15] // (op 459) convert


s_lshl_b64 s[14:15], s[2:3], 6 // (op 459) 
v_mov_b32 v16, s14 // (op 459) 
v_mov_b32 v22, s15 // (op 459) 
v_add_co_u32 v42, s[18:19], v12, v16 // (op 459) least significant half
v_addc_co_u32 v43, s[18:19], v13, v22, s[18:19] // (op 459) most significant half




s_add_u32 s1, 2048, s0 // (op 459) 
s_mov_b32 m0, s1 // (op 459) reg expression
buffer_load_dwordx4 v42, s[40:43], 0 offen offset: 0 lds // (op 459) Load value direct to lds
v_add_co_u32 v42, s[14:15], v42, v40 // (op 459) least significant half
v_addc_co_u32 v43, s[14:15], 0, v43, s[14:15] // (op 459) most significant half

s_add_u32 m0, m0, 1024 // (op 459) 
buffer_load_dwordx4 v42, s[40:43], 0 offen offset: 0 lds // (op 459) Load value direct to lds







s_lshl_b32 s0, s12, 1 // (op 387) 
v_add_u32 v16, v17, s0 // (op 387) 


buffer_load_ubyte v22, v16, s[48:51], 0 offen offset: 0 // (op 387) Load value


v_lshlrev_b32 v31, 5, v1 // (op 828) 
v_mul_lo_u32 v35, v31, 6 // (op 828) 
v_lshrrev_b32 v12, 3, v35 // (op 828) 

v_lshrrev_b32 v13, 4, v31 // (op 828) 

v_lshlrev_b32 v31, 2, v13 // (op 828) 

v_add_u32 v13, v12, v31 // (op 828) 





s_waitcnt vmcnt(11)
ds_write_b96 v13, v[24:26] offset:4096 // (op 389) Store local data 
s_waitcnt vmcnt(10)
ds_write_b96 v13, v[28:30] offset:4112 // (op 389) Store local data 

v_and_b32 v31, -32, v1 // (op 934) 
v_sub_u32 v35, v1, v31 // (op 934) 

v_and_b32 v31, 4294967264, v1 // (op 934) 
v_add_u32 v12, v31, v35 // (op 934) 





s_waitcnt vmcnt(9)
ds_write_b8 v12, v21 offset:6144 // (op 390) Store local data 

v_and_b32 v31, -32, v1 // (op 881) 
v_sub_u32 v35, v1, v31 // (op 881) 

v_and_b32 v31, 4294967264, v1 // (op 881) 
v_add_u32 v12, v31, v35 // (op 881) 





s_waitcnt vmcnt(6)
ds_write_b8 v12, v27 offset:6208 // (op 392) Store local data 

v_and_b32 v31, -2, v1 // (op 747) 
v_sub_u32 v35, v1, v31 // (op 747) 

v_lshlrev_b32 v31, 4, v35 // (op 747) 

v_lshrrev_b32 v35, 1, v1 // (op 747) 
v_lshl_add_u32 v12, v35, 5, v31 // (op 747) 


s_nop 0
v_readfirstlane_b32 s0, v12 // (op 747) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v31, 5, v1 // (op 799) 
v_mul_lo_u32 v35, v31, 6 // (op 799) 
v_lshrrev_b32 v12, 3, v35 // (op 799) 

v_lshrrev_b32 v13, 4, v31 // (op 799) 

v_lshlrev_b32 v31, 2, v13 // (op 799) 

v_add_u32 v13, v12, v31 // (op 799) 


v_and_b32 v12, -32, v1 // (op 852) 
v_sub_u32 v31, v1, v12 // (op 852) 

v_and_b32 v12, 4294967264, v1 // (op 852) 
v_add_u32 v35, v12, v31 // (op 852) 


v_and_b32 v12, -32, v1 // (op 905) 
v_sub_u32 v31, v1, v12 // (op 905) 

v_and_b32 v12, 4294967264, v1 // (op 905) 
v_add_u32 v16, v12, v31 // (op 905) 


v_and_b32 v12, -2, v1 // (op 992) 
v_sub_u32 v17, v1, v12 // (op 992) 

v_lshlrev_b32 v12, 4, v17 // (op 992) 

v_lshrrev_b32 v17, 1, v1 // (op 992) 
v_lshl_add_u32 v31, v17, 5, v12 // (op 992) 


s_nop 0
v_readfirstlane_b32 s1, v31 // (op 992) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v12, 5, v1 // (op 1031) 
v_mul_lo_u32 v17, v12, 6 // (op 1031) 
v_lshrrev_b32 v31, 3, v17 // (op 1031) 

v_lshrrev_b32 v17, 4, v12 // (op 1031) 

v_lshlrev_b32 v12, 2, v17 // (op 1031) 

v_add_u32 v17, v31, v12 // (op 1031) 


v_and_b32 v12, -32, v1 // (op 1055) 
v_sub_u32 v31, v1, v12 // (op 1055) 

v_and_b32 v12, 4294967264, v1 // (op 1055) 
v_add_u32 v39, v12, v31 // (op 1055) 


v_and_b32 v12, -32, v1 // (op 1079) 
v_sub_u32 v31, v1, v12 // (op 1079) 

v_and_b32 v12, 4294967264, v1 // (op 1079) 
v_add_u32 v40, v12, v31 // (op 1079) 


s_mov_b32 s14, 0 // (op 59) call()
s_ashr_i64 s[18:19], s[6:7], 6 // (op 58) 
s_mov_b32 s15, s18 // (op 58) convert

s_and_b32 s18, s15, 4294967294 // (op 58) 

s_cmp_lt_u32 s14, s18 // (op 58) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoop_58 // (op 58) Condition: Top (jump to RR_GEMM_NT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoop_58 if false)
RR_GEMM_NT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoop_58:


s_barrier  // (op 1128) 
s_add_u32 s14, s14, 2 // (op 60) 



v_mov_b32 v12, v19 // (op 10) 
s_lshl_b64 s[18:19], s[8:9], 6 // (op 10) 
s_mul_i32 s39, s19, 6 // (op 10) most significant: low of high * low
s_mul_i32 s38, s18, 6 // (op 10) least significant
s_mul_hi_u32 s15, s18, 6 // (op 10) most significant: high of low * low
s_add_u32 s39, s39, s15 // (op 10) most significant: sum


s_lshr_b64 s[18:19], s[38:39], 3 // (op 10) 

s_lshl_b32 s15, s18, 1 // (op 10) 

v_add_u32 v31, v12, s15 // (op 10) 


ds_read_b64_tr_b8 v[42:43], v3 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v3 offset:256 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v3 offset:1024 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v3 offset:1280 // (op 70) Transpose load local data 
buffer_load_dwordx3 v[24:26], v31, s[44:47], 0 offen // (op 10) Load value
buffer_load_dwordx3 v[28:30], v31, s[44:47], 0 offen offset:12 // (op 10) Load value

s_waitcnt lgkmcnt(6)
s_barrier  // (op 1141) 





v_mov_b32 v31, v23 // (op 25) 
s_lshl_b32 s15, s24, 2 // (op 25) 
v_add_u32 v41, v31, s15 // (op 25) 


buffer_load_ubyte v21, v41, s[52:55], 0 offen offset: 0 // (op 25) Load value

ds_read_b96_tr_b6 v[50:52], v0 offset:4096 // (op 75) Transpose load local data 
ds_read_b96_tr_b6 v[53:55], v0 offset:4608 // (op 75) Transpose load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1131) 

s_barrier  // (op 1155) 
s_lshl_b64 s[18:19], s[2:3], 5 // (op 438) 


ds_read_u8 v41, v5 offset:6208 // (op 80) Load local data 
v_mov_b64 v[56:57], s[18:19] // (op 438) convert


s_barrier  // (op 1169) 
v_mov_b64 v[58:59], v[14:15] // (op 438) 


s_lshl_b64 s[18:19], s[2:3], 6 // (op 438) 
s_lshl_b64 s[38:39], s[18:19], 1 // (op 438) 

v_mov_b32 v60, s38 // (op 438) 
v_mov_b32 v61, s39 // (op 438) 
v_add_co_u32 v62, s[18:19], v58, v60 // (op 438) least significant half
v_addc_co_u32 v63, s[18:19], v59, v61, s[18:19] // (op 438) most significant half




s_mov_b32 m0, s0 // (op 438) reg expression
ds_read_u8 v60, v7 offset:6144 // (op 85) Load local data 
buffer_load_dwordx4 v62, s[40:43], 0 offen offset: 0 lds // (op 438) Load value direct to lds
v_add_co_u32 v62, s[18:19], v62, v56 // (op 438) least significant half
v_addc_co_u32 v63, s[18:19], 0, v63, s[18:19] // (op 438) most significant half

s_add_u32 m0, m0, 1024 // (op 438) 
buffer_load_dwordx4 v62, s[40:43], 0 offen offset: 0 lds // (op 438) Load value direct to lds


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[42:49], v[50:55], a[0:15], v41, v60 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 88) 






v_mov_b32 v41, v20 // (op 17) 
s_lshl_b32 s15, s12, 2 // (op 17) 
v_add_u32 v42, v41, s15 // (op 17) 


buffer_load_ubyte v27, v42, s[48:51], 0 offen offset: 0 // (op 17) Load value

s_barrier  // (op 1192) 



s_waitcnt vmcnt(11)
ds_write_b96 v17, v[32:34] offset:6272 // (op 253) Store local data 
s_waitcnt vmcnt(10)
ds_write_b96 v17, v[36:38] offset:6288 // (op 253) Store local data 
s_barrier  // (op 1204) 



s_waitcnt vmcnt(9)
ds_write_b8 v40, v18 offset:8320 // (op 255) Store local data 
s_barrier  // (op 1198) 



s_waitcnt vmcnt(6)
ds_write_b8 v39, v22 offset:8384 // (op 254) Store local data 


s_barrier  // (op 1183) 
s_lshl_b64 s[18:19], s[8:9], 6 // (op 256) 



s_mul_i32 s39, s19, 6 // (op 256) most significant: low of high * low
s_mul_i32 s38, s18, 6 // (op 256) least significant
s_mul_hi_u32 s15, s18, 6 // (op 256) most significant: high of low * low
s_add_u32 s39, s39, s15 // (op 256) most significant: sum


s_lshr_b64 s[18:19], s[38:39], 3 // (op 256) 

s_mul_i32 s15, 3, s18 // (op 256) 

v_add_u32 v42, v12, s15 // (op 256) 


ds_read_b64_tr_b8 v[50:51], v6 offset:2048 // (op 245) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v6 offset:2304 // (op 245) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v6 offset:3072 // (op 245) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v6 offset:3328 // (op 245) Transpose load local data 
buffer_load_dwordx3 v[32:34], v42, s[44:47], 0 offen // (op 256) Load value
buffer_load_dwordx3 v[36:38], v42, s[44:47], 0 offen offset:12 // (op 256) Load value

s_waitcnt lgkmcnt(6)
s_barrier  // (op 1189) 






s_lshl_b32 s15, s24, 1 // (op 258) 
s_mul_i32 s18, 3, s15 // (op 258) 

v_add_u32 v12, v31, s18 // (op 258) 


buffer_load_ubyte v18, v12, s[52:55], 0 offen offset: 0 // (op 258) Load value

s_lshl_b64 s[18:19], s[8:9], 6 // (op 573) 

s_mul_i32 s39, s19, 6 // (op 573) most significant: low of high * low
s_mul_i32 s38, s18, 6 // (op 573) least significant
s_mul_hi_u32 s15, s18, 6 // (op 573) most significant: high of low * low
s_add_u32 s39, s39, s15 // (op 573) most significant: sum


s_lshl_b32 s15, s24, 2 // (op 701) 
v_add_u32 v23, v23, s15 // (op 701) 

s_lshr_b64 s[18:19], s[38:39], 3 // (op 573) 

s_lshl_b32 s15, s18, 1 // (op 573) 

v_add_u32 v19, v19, s15 // (op 573) 

ds_read_b96_tr_b6 v[42:44], v4 offset:6272 // (op 247) Transpose load local data 
ds_read_b96_tr_b6 v[45:47], v4 offset:6784 // (op 247) Transpose load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1186) 

s_barrier  // (op 1195) 
s_barrier  // (op 1255) 


ds_read_u8 v12, v9 offset:8384 // (op 249) Load local data 
s_lshl_b64 s[18:19], s[2:3], 5 // (op 449) 
v_mov_b64 v[48:49], s[18:19] // (op 449) convert


s_barrier  // (op 1201) 
s_barrier  // (op 1261) 


s_lshl_b64 s[18:19], s[2:3], 6 // (op 449) 
s_mul_i32 s39, 3, s19 // (op 449) most significant: low of low * high
s_mul_i32 s38, 3, s18 // (op 449) least significant
s_mul_hi_u32 s15, 3, s18 // (op 449) most significant: high of low * low
s_add_u32 s39, s39, s15 // (op 449) most significant: sum


v_mov_b32 v31, s38 // (op 449) 
v_mov_b32 v60, s39 // (op 449) 
v_add_co_u32 v62, s[18:19], v58, v31 // (op 449) least significant half
v_addc_co_u32 v63, s[18:19], v59, v60, s[18:19] // (op 449) most significant half




s_add_u32 s15, 2048, s1 // (op 449) 
s_mov_b32 m0, s15 // (op 449) reg expression
ds_read_u8 v31, v11 offset:8320 // (op 251) Load local data 
buffer_load_dwordx4 v62, s[40:43], 0 offen offset: 0 lds // (op 449) Load value direct to lds
v_add_co_u32 v62, s[18:19], v62, v48 // (op 449) least significant half
s_barrier  // (op 1267) 
v_addc_co_u32 v63, s[18:19], 0, v63, s[18:19] // (op 449) most significant half

s_add_u32 m0, m0, 1024 // (op 449) 
buffer_load_dwordx4 v62, s[40:43], 0 offen offset: 0 lds // (op 449) Load value direct to lds



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[50:57], v[42:47], a[0:15], v12, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 252) 







s_lshl_b32 s15, s12, 1 // (op 257) 
s_mul_i32 s18, 3, s15 // (op 257) 

v_add_u32 v12, v41, s18 // (op 257) 


buffer_load_ubyte v22, v12, s[48:51], 0 offen offset: 0 // (op 257) Load value

s_lshl_b64 s[18:19], s[2:3], 6 // (op 509) 

s_lshl_b64 s[38:39], s[18:19], 1 // (op 509) 

s_barrier  // (op 1144) 
s_lshl_b32 s15, s12, 2 // (op 637) 



v_add_u32 v20, v20, s15 // (op 637) 

v_mov_b32 v12, s38 // (op 509) 
v_mov_b32 v31, s39 // (op 509) 
v_add_co_u32 v14, s[18:19], v14, v12 // (op 509) least significant half
v_addc_co_u32 v15, s[18:19], v15, v31, s[18:19] // (op 509) most significant half




s_waitcnt vmcnt(11)
ds_write_b96 v13, v[24:26] offset:4096 // (op 13) Store local data 
s_waitcnt vmcnt(10)
ds_write_b96 v13, v[28:30] offset:4112 // (op 13) Store local data 
s_barrier  // (op 1172) 



s_waitcnt vmcnt(9)
ds_write_b8 v16, v21 offset:6144 // (op 30) Store local data 
s_barrier  // (op 1158) 



s_waitcnt vmcnt(6)
ds_write_b8 v35, v27 offset:6208 // (op 21) Store local data 
s_ashr_i64 s[18:19], s[6:7], 6 // (op 58) 
s_mov_b32 s15, s18 // (op 58) convert

s_and_b32 s18, s15, 4294967294 // (op 58) 

s_cmp_lt_u32 s14, s18 // (op 58) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoop_58 // (op 58) Condition: Bottom (jump to RR_GEMM_NT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoop_58 if true)
RR_GEMM_NT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoop_58:
















s_barrier  // (op 1217) 
v_and_b32 v35, -2, v1 // (op 472) 
s_ashr_i64 s[0:1], s[6:7], 6 // (op 472) 
v_sub_u32 v12, v1, v35 // (op 472) 

s_lshr_b32 s14, s0, 1 // (op 472) 

v_lshlrev_b32 v35, 4, v12 // (op 472) 

v_lshl_add_u32 v12, s16, 5, v35 // (op 472) 

v_mov_b32 v21, 0 // (op 472) convert
v_mov_b32 v20, v12 // (op 472) convert

v_lshrrev_b32 v19, 1, v1 // (op 472) 
v_lshl_add_u32 v35, s14, 7, v19 // (op 472) 


v_mul_lo_u32 v13, v35, s3 // (op 472) most significant: low of low * high
v_mul_hi_u32 v19, v35, s2 // (op 472) most significant: high of low * low
v_mul_lo_u32 v12, v35, s2 // (op 472) least significant: low of low * low
v_add_u32 v13, v13, v19 // (op 472) most significant: sum


v_add_co_u32 v14, s[0:1], v20, v12 // (op 472) least significant half
v_addc_co_u32 v15, s[0:1], v21, v13, s[0:1] // (op 472) most significant half



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

s_ashr_i64 s[0:1], s[6:7], 6 // (op 539) 
s_lshr_b32 s14, s0, 1 // (op 539) 

v_lshl_add_u32 v35, s14, 7, v1 // (op 539) 

s_lshl_b32 s0, s17, 5 // (op 539) 
s_mov_b32 s15, 0 // (op 539) convert
s_mov_b32 s14, s0 // (op 539) convert

v_mul_lo_u32 v13, v35, s9 // (op 539) most significant: low of low * high
v_mul_hi_u32 v16, v35, s8 // (op 539) most significant: high of low * low
v_mul_lo_u32 v12, v35, s8 // (op 539) least significant: low of low * low
v_add_u32 v13, v13, v16 // (op 539) most significant: sum

v_mov_b32 v16, s14 // (op 539) 
v_mov_b32 v17, s15 // (op 539) 
v_add_co_u32 v12, s[0:1], v12, v16 // (op 539) least significant half
v_addc_co_u32 v13, s[0:1], v13, v17, s[0:1] // (op 539) most significant half





v_mul_lo_u32 v17, v13, 6 // (op 539) most significant: low of high * low
v_mul_hi_u32 v35, v12, 6 // (op 539) most significant: high of low * low
v_mul_lo_u32 v16, v12, 6 // (op 539) least significant: low of low * low
v_add_u32 v17, v17, v35 // (op 539) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[16:17] // (op 539) 

v_mov_b32 v35, v12 // (op 539) convert

s_mul_i32 s1, s7, 6 // (op 541) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 541) least significant
s_mul_hi_u32 s14, s6, 6 // (op 541) most significant: high of low * low
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

v_and_b32 v12, -32, v1 // (op 603) 
s_ashr_i64 s[0:1], s[6:7], 6 // (op 603) 
v_sub_u32 v13, v1, v12 // (op 603) 

s_lshr_b32 s14, s0, 1 // (op 603) 

v_lshl_add_u32 v12, s16, 5, v13 // (op 603) 

v_lshrrev_b32 v13, 5, v1 // (op 603) 
v_lshl_add_u32 v16, s14, 2, v13 // (op 603) 


v_mul_lo_u32 v13, v16, s12 // (op 603) 

v_add_u32 v16, v12, v13 // (op 603) 


s_mul_i32 s1, s12, s23 // (op 605) most significant: low of low * high
s_mul_i32 s14, s13, s22 // (op 605) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 605) least significant
s_mul_hi_u32 s15, s12, s22 // (op 605) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 605) most significant: sum
s_add_u32 s1, s1, s15 // (op 605) most significant: sum


s_mov_b32 s48, s20 // (op 605) reg expression
s_mov_b32 s49, s21 // (op 605) reg expression
s_mov_b32 s50, s0 // (op 605) reg expression
s_mov_b32 s51, 131072 // (op 605) call()

v_lshrrev_b32 v17, 5, v1 // (op 667) 
s_ashr_i64 s[0:1], s[6:7], 6 // (op 667) 
v_and_b32 v12, -32, v1 // (op 667) 
s_lshr_b32 s14, s0, 1 // (op 667) 

v_sub_u32 v13, v1, v12 // (op 667) 

v_lshl_add_u32 v12, s14, 2, v17 // (op 667) 


v_lshl_add_u32 v17, s17, 5, v13 // (op 667) 

v_mul_lo_u32 v13, v12, s24 // (op 667) 
v_add_u32 v13, v13, v17 // (op 667) 


s_mul_i32 s1, s24, s23 // (op 669) most significant: low of low * high
s_mul_i32 s14, s25, s22 // (op 669) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 669) least significant
s_mul_hi_u32 s15, s24, s22 // (op 669) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 669) most significant: sum
s_add_u32 s1, s1, s15 // (op 669) most significant: sum


s_mov_b32 s52, s26 // (op 669) reg expression
s_mov_b32 s53, s27 // (op 669) reg expression
s_mov_b32 s54, s0 // (op 669) reg expression
s_mov_b32 s55, 131072 // (op 669) call()

v_and_b32 v12, -2, v1 // (op 760) 
v_sub_u32 v17, v1, v12 // (op 760) 

v_lshlrev_b32 v12, 4, v17 // (op 760) 

v_lshrrev_b32 v17, 1, v1 // (op 760) 
v_lshl_add_u32 v19, v17, 5, v12 // (op 760) 


s_nop 0
v_readfirstlane_b32 s0, v19 // (op 760) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v12, 5, v1 // (op 812) 
v_mul_lo_u32 v17, v12, 6 // (op 812) 
v_lshrrev_b32 v19, 3, v17 // (op 812) 

v_lshrrev_b32 v17, 4, v12 // (op 812) 

v_lshlrev_b32 v12, 2, v17 // (op 812) 

v_add_u32 v17, v19, v12 // (op 812) 


v_and_b32 v12, -32, v1 // (op 865) 
v_sub_u32 v19, v1, v12 // (op 865) 

v_and_b32 v12, 4294967264, v1 // (op 865) 
v_add_u32 v20, v12, v19 // (op 865) 


v_and_b32 v12, -32, v1 // (op 918) 
v_sub_u32 v19, v1, v12 // (op 918) 

v_and_b32 v12, 4294967264, v1 // (op 918) 
v_add_u32 v21, v12, v19 // (op 918) 


s_ashr_i64 s[14:15], s[6:7], 6 // (op 172) 
s_and_b32 s1, s14, 4294967294 // (op 172) 

s_ashr_i64 s[14:15], s[6:7], 6 // (op 171) 
s_mov_b32 s18, s14 // (op 171) convert

s_cmp_lt_u32 s1, s18 // (op 171) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoopTail_171 // (op 171) Condition: Top (jump to RR_GEMM_NT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoopTail_171 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoopTail_171:
s_barrier  // (op 1134) 
s_add_u32 s1, s1, 1 // (op 173) 

s_lshl_b64 s[14:15], s[2:3], 5 // (op 443) 
v_mov_b64 v[18:19], s[14:15] // (op 443) convert


v_mov_b64 v[22:23], v[14:15] // (op 443) 
v_mov_b64 v[24:25], v[22:23] // (op 443) reg expression
s_mov_b32 m0, s0 // (op 443) reg expression
buffer_load_dwordx4 v24, s[40:43], 0 offen offset: 0 lds // (op 443) Load value direct to lds
v_add_co_u32 v24, s[14:15], v24, v18 // (op 443) least significant half
v_addc_co_u32 v25, s[14:15], 0, v25, s[14:15] // (op 443) most significant half

s_add_u32 m0, m0, 1024 // (op 443) 
buffer_load_dwordx4 v24, s[40:43], 0 offen offset: 0 lds // (op 443) Load value direct to lds



s_lshl_b64 s[14:15], s[2:3], 6 // (op 462) 
v_mov_b32 v12, s14 // (op 462) 
v_mov_b32 v18, s15 // (op 462) 
v_add_co_u32 v14, s[18:19], v14, v12 // (op 462) least significant half
v_addc_co_u32 v15, s[18:19], v15, v18, s[18:19] // (op 462) most significant half






v_mov_b32 v12, v35 // (op 190) 
v_mov_b32 v18, v12 // (op 190) reg expression

buffer_load_dwordx3 v[22:24], v18, s[44:47], 0 offen // (op 190) Load value
buffer_load_dwordx3 v[26:28], v18, s[44:47], 0 offen offset:12 // (op 190) Load value


s_lshl_b64 s[14:15], s[8:9], 6 // (op 531) 
s_mul_i32 s19, s15, 6 // (op 531) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 531) least significant
s_mul_hi_u32 s38, s14, 6 // (op 531) most significant: high of low * low
s_add_u32 s19, s19, s38 // (op 531) most significant: sum


s_lshr_b64 s[14:15], s[18:19], 3 // (op 531) 

v_add_u32 v35, v35, s14 // (op 531) 



s_barrier  // (op 1147) 
v_mov_b32 v12, v16 // (op 191) 



v_mov_b32 v25, v12 // (op 191) reg expression

buffer_load_ubyte v18, v25, s[48:51], 0 offen offset: 0 // (op 191) Load value


s_lshl_b32 s14, s12, 1 // (op 595) 
v_add_u32 v16, v16, s14 // (op 595) 

s_waitcnt vmcnt(2)
ds_write_b96 v17, v[22:24] offset:4096 // (op 187) Store local data 


v_mov_b32 v12, v13 // (op 192) 
v_mov_b32 v19, v12 // (op 192) reg expression

buffer_load_ubyte v25, v19, s[52:55], 0 offen offset: 0 // (op 192) Load value


s_lshl_b32 s14, s24, 1 // (op 659) 
v_add_u32 v13, v13, s14 // (op 659) 

s_waitcnt vmcnt(2)
ds_write_b96 v17, v[26:28] offset:4112 // (op 187) Store local data 

s_barrier  // (op 1137) 



s_barrier  // (op 1161) 
ds_read_b64_tr_b8 v[26:27], v3 // (op 179) Transpose load local data 



ds_read_b64_tr_b8 v[28:29], v3 offset:256 // (op 179) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v3 offset:1024 // (op 179) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v3 offset:1280 // (op 179) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v20, v18 offset:6208 // (op 188) Store local data 

s_waitcnt lgkmcnt(5)
s_barrier  // (op 1151) 



s_barrier  // (op 1175) 
ds_read_b96_tr_b6 v[36:38], v0 offset:4096 // (op 181) Transpose load local data 



ds_read_b96_tr_b6 v[39:41], v0 offset:4608 // (op 181) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v21, v25 offset:6144 // (op 189) Store local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 1165) 


ds_read_u8 v12, v5 offset:6208 // (op 183) Load local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1179) 


ds_read_u8 v34, v7 offset:6144 // (op 185) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[26:33], v[36:41], a[0:15], v12, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 186) 




s_ashr_i64 s[14:15], s[6:7], 6 // (op 171) 
s_mov_b32 s18, s14 // (op 171) convert

s_cmp_lt_u32 s1, s18 // (op 171) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoopTail_171 // (op 171) Condition: Bottom (jump to RR_GEMM_NT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoopTail_171 if true)
RR_GEMM_NT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoopTail_171:























s_barrier  // (op 1249) 








v_cmp_eq_f32 vcc, 0.00000, v8 // (op 133) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalFalse_0_beta_133 // (op 133) Condition: False, jump to RR_GEMM_NT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalFalse_0_beta_133
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
v_mul_f32 v28, v10, v12 // (op 50) 
v_mul_f32 v29, v10, v13 // (op 50) 
v_mul_f32 v30, v10, v14 // (op 50) 
v_mul_f32 v31, v10, v15 // (op 50) 
v_mul_f32 v32, v10, v16 // (op 50) 
v_mul_f32 v33, v10, v17 // (op 50) 
v_mul_f32 v34, v10, v18 // (op 50) 
v_mul_f32 v35, v10, v19 // (op 50) 
v_mul_f32 v36, v10, v20 // (op 50) 
v_mul_f32 v37, v10, v21 // (op 50) 
v_mul_f32 v38, v10, v22 // (op 50) 
v_mul_f32 v39, v10, v23 // (op 50) 
v_mul_f32 v40, v10, v24 // (op 50) 
v_mul_f32 v41, v10, v25 // (op 50) 
v_mul_f32 v42, v10, v26 // (op 50) 
v_mul_f32 v43, v10, v27 // (op 50) 

v_lshrrev_b32 v0, 6, v1 // (op 960) 
v_and_b32 v9, 63, v1 // (op 960) 
v_lshrrev_b32 v3, 5, v9 // (op 960) 
v_lshlrev_b32 v4, 2, v3 // (op 960) 

v_lshl_add_u32 v3, v0, 5, v4 // (op 960) 


v_lshl_add_u32 v0, s16, 5, v3 // (op 960) 

v_lshrrev_b32 v3, 2, v9 // (op 960) 

v_and_b32 v9, 7, v3 // (op 960) 

v_and_b32 v3, 3, v1 // (op 960) 
v_lshl_add_u32 v4, v9, 2, v3 // (op 960) 


v_lshl_add_u32 v3, s17, 5, v4 // (op 960) 

v_mul_lo_u32 v9, v3, s28 // (op 960) 

v_add_lshl_u32 v3, v0, v9, 2 // (op 960) 


s_mul_i32 s1, s28, s33 // (op 962) most significant: low of low * high
s_mul_i32 s2, s29, s32 // (op 962) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 962) least significant
s_mul_hi_u32 s3, s28, s32 // (op 962) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 962) most significant: sum
s_add_u32 s1, s1, s3 // (op 962) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 2 // (op 962) 

s_mov_b32 s4, s30 // (op 962) reg expression
s_mov_b32 s5, s31 // (op 962) reg expression
s_mov_b32 s6, s2 // (op 962) reg expression
s_mov_b32 s7, 131072 // (op 962) call()




buffer_store_dwordx4 v[28:31], v3, s[4:7], 0 offen // (op 56) Store value
buffer_store_dwordx4 v[32:35], v3, s[4:7], 0 offen offset:32 // (op 56) Store value
buffer_store_dwordx4 v[36:39], v3, s[4:7], 0 offen offset:64 // (op 56) Store value
buffer_store_dwordx4 v[40:43], v3, s[4:7], 0 offen offset:96 // (op 56) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalBottom_0_beta_133 // (op 133) Condition: Done, jump to RR_GEMM_NT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalBottom_0_beta_133
RR_GEMM_NT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalFalse_0_beta_133:
v_lshrrev_b32 v0, 6, v1 // (op 730) 
v_and_b32 v9, 63, v1 // (op 730) 
v_lshrrev_b32 v3, 5, v9 // (op 730) 
v_lshlrev_b32 v4, 2, v3 // (op 730) 

v_lshl_add_u32 v3, v0, 5, v4 // (op 730) 


v_lshl_add_u32 v0, s16, 5, v3 // (op 730) 

v_lshrrev_b32 v3, 2, v9 // (op 730) 

v_and_b32 v9, 7, v3 // (op 730) 

v_and_b32 v3, 3, v1 // (op 730) 
v_lshl_add_u32 v4, v9, 2, v3 // (op 730) 


v_lshl_add_u32 v3, s17, 5, v4 // (op 730) 

v_mul_lo_u32 v9, v3, s34 // (op 730) 

v_add_lshl_u32 v3, v0, v9, 2 // (op 730) 


s_mul_i32 s1, s34, s33 // (op 732) most significant: low of low * high
s_mul_i32 s2, s35, s32 // (op 732) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 732) least significant
s_mul_hi_u32 s3, s34, s32 // (op 732) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 732) most significant: sum
s_add_u32 s1, s1, s3 // (op 732) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 2 // (op 732) 

s_mov_b32 s8, s36 // (op 732) reg expression
s_mov_b32 s9, s37 // (op 732) reg expression
s_mov_b32 s10, s2 // (op 732) reg expression
s_mov_b32 s11, 131072 // (op 732) call()





buffer_load_dwordx4 v[12:15], v3, s[8:11], 0 offen // (op 145) Load value
buffer_load_dwordx4 v[16:19], v3, s[8:11], 0 offen offset:32 // (op 145) Load value
buffer_load_dwordx4 v[20:23], v3, s[8:11], 0 offen offset:64 // (op 145) Load value
buffer_load_dwordx4 v[24:27], v3, s[8:11], 0 offen offset:96 // (op 145) Load value



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
v_mul_f32 v44, v10, v28 // (op 142) 
v_mul_f32 v45, v10, v29 // (op 142) 
v_mul_f32 v46, v10, v30 // (op 142) 
v_mul_f32 v47, v10, v31 // (op 142) 
v_mul_f32 v48, v10, v32 // (op 142) 
v_mul_f32 v49, v10, v33 // (op 142) 
v_mul_f32 v50, v10, v34 // (op 142) 
v_mul_f32 v51, v10, v35 // (op 142) 
v_mul_f32 v52, v10, v36 // (op 142) 
v_mul_f32 v53, v10, v37 // (op 142) 
v_mul_f32 v54, v10, v38 // (op 142) 
v_mul_f32 v55, v10, v39 // (op 142) 
v_mul_f32 v56, v10, v40 // (op 142) 
v_mul_f32 v57, v10, v41 // (op 142) 
v_mul_f32 v58, v10, v42 // (op 142) 
v_mul_f32 v59, v10, v43 // (op 142) 

s_waitcnt vmcnt(3)
v_mul_f32 v28, v8, v12 // (op 146) 
v_mul_f32 v29, v8, v13 // (op 146) 
v_mul_f32 v30, v8, v14 // (op 146) 
v_mul_f32 v31, v8, v15 // (op 146) 
s_waitcnt vmcnt(2)
v_mul_f32 v32, v8, v16 // (op 146) 
v_mul_f32 v33, v8, v17 // (op 146) 
v_mul_f32 v34, v8, v18 // (op 146) 
v_mul_f32 v35, v8, v19 // (op 146) 
s_waitcnt vmcnt(1)
v_mul_f32 v36, v8, v20 // (op 146) 
v_mul_f32 v37, v8, v21 // (op 146) 
v_mul_f32 v38, v8, v22 // (op 146) 
v_mul_f32 v39, v8, v23 // (op 146) 
s_waitcnt vmcnt(0)
v_mul_f32 v40, v8, v24 // (op 146) 
v_mul_f32 v41, v8, v25 // (op 146) 
v_mul_f32 v42, v8, v26 // (op 146) 
v_mul_f32 v43, v8, v27 // (op 146) 

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
v_and_b32 v9, 63, v1 // (op 977) 
v_lshrrev_b32 v11, 5, v9 // (op 977) 
v_lshlrev_b32 v3, 2, v11 // (op 977) 

v_lshl_add_u32 v11, v0, 5, v3 // (op 977) 


v_lshl_add_u32 v0, s16, 5, v11 // (op 977) 

v_lshrrev_b32 v11, 2, v9 // (op 977) 

v_and_b32 v9, 7, v11 // (op 977) 

v_and_b32 v11, 3, v1 // (op 977) 
v_lshl_add_u32 v3, v9, 2, v11 // (op 977) 


v_lshl_add_u32 v9, s17, 5, v3 // (op 977) 

v_mul_lo_u32 v11, v9, s28 // (op 977) 

v_add_lshl_u32 v9, v0, v11, 2 // (op 977) 


s_mul_i32 s1, s28, s33 // (op 979) most significant: low of low * high
s_mul_i32 s2, s29, s32 // (op 979) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 979) least significant
s_mul_hi_u32 s3, s28, s32 // (op 979) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 979) most significant: sum
s_add_u32 s1, s1, s3 // (op 979) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 2 // (op 979) 

s_mov_b32 s4, s30 // (op 979) reg expression
s_mov_b32 s5, s31 // (op 979) reg expression
s_mov_b32 s6, s2 // (op 979) reg expression
s_mov_b32 s7, 131072 // (op 979) call()




buffer_store_dwordx4 v[12:15], v9, s[4:7], 0 offen // (op 144) Store value
buffer_store_dwordx4 v[16:19], v9, s[4:7], 0 offen offset:32 // (op 144) Store value
buffer_store_dwordx4 v[20:23], v9, s[4:7], 0 offen offset:64 // (op 144) Store value
buffer_store_dwordx4 v[24:27], v9, s[4:7], 0 offen offset:96 // (op 144) Store value






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalBottom_0_beta_133:



s_endpgm  // End of RR_GEMM_NT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2
.LRR_GEMM_NT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_end:
.size RR_GEMM_NT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2, .LRR_GEMM_NT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_end-RR_GEMM_NT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2
  .amdhsa_next_free_vgpr 80
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 8448 // lds bytes
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
  MXScaleFormat: NoSwizzle
  ProblemType:
    OperationType: GEMM
    DataType: B8
    DataTypeA: B8
    DataTypeB: F6
    DestDataType: s
    ComputeDataType: s
    HighPrecisionAccumulate: True
    TransposeA: False
    TransposeB: True
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
            { type: int64, semantic: StrideScaleA0 },
            { type: float32, semantic: Beta },
            { type: float32, semantic: Alpha },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: SizeSumScaleA },
            { type: int64, semantic: StrideScaleB0 },
            { type: address, semantic: AddressMXScaleB },
            { type: int64, semantic: StrideD0 },
            { type: address, semantic: AddressD },
            { type: int64, semantic: SizeFree1 },
            { type: int64, semantic: StrideC0 },
            { type: address, semantic: AddressC } ]
    macrotile: [32, 32, 64]
    threads: [64, 1, 1]
    grid: [TilesX, TilesY, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [32, 32, 64, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_NT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2
    .symbol: RR_GEMM_NT_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 8448
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 56
    .vgpr_count: 64
    .agpr_count: 16
    .max_flat_workgroup_size: 64
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
      - .name: Tensor_2_stride_0_3
        .size: 8
        .offset: 24
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 32
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_stride_1_5
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
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 56
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_size_1_6
        .size: 8
        .offset: 64
        .value_kind: by_value
      - .name: Tensor_7_stride_0_8
        .size: 8
        .offset: 72
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 80
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_21_stride_1_16
        .size: 8
        .offset: 88
        .value_kind: by_value
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 96
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_2_size_1_11
        .size: 8
        .offset: 104
        .value_kind: by_value
      - .name: Tensor_10_stride_1_10
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

