// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2
.p2align 8
.type RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2,@function
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 6, v1 // (op 743) 
v_and_b32 v3, 15, v1 // (op 743) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 743) 


v_and_b32 v0, 63, v1 // (op 743) 
v_lshrrev_b32 v3, 4, v0 // (op 743) 

v_lshlrev_b32 v0, 5, v3 // (op 743) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 743) 


v_lshlrev_b32 v0, 2, v3 // (op 743) 

v_lshrrev_b32 v3, 3, v0 // (op 743) 

v_and_b32 v0, 15, v1 // (op 796) 
v_and_b32 v4, 63, v1 // (op 796) 
v_lshrrev_b32 v5, 4, v4 // (op 796) 

v_lshlrev_b32 v4, 5, v5 // (op 796) 

v_lshl_add_u32 v5, v0, 7, v4 // (op 796) 


v_lshlrev_b32 v0, 2, v5 // (op 796) 

v_lshrrev_b32 v4, 3, v0 // (op 796) 

v_lshrrev_b32 v0, 6, v1 // (op 849) 
v_and_b32 v5, 15, v1 // (op 849) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 849) 


v_and_b32 v0, 63, v1 // (op 849) 
v_lshrrev_b32 v5, 4, v0 // (op 849) 

v_lshl_add_u32 v0, v6, 2, v5 // (op 849) 


v_lshrrev_b32 v5, 6, v1 // (op 907) 
v_and_b32 v6, 15, v1 // (op 907) 
v_lshl_add_u32 v7, v5, 4, v6 // (op 907) 


v_and_b32 v5, 63, v1 // (op 907) 
v_lshrrev_b32 v6, 4, v5 // (op 907) 

v_lshlrev_b32 v5, 5, v6 // (op 907) 

v_lshl_add_u32 v6, v7, 7, v5 // (op 907) 


v_lshlrev_b32 v5, 2, v6 // (op 907) 

v_lshrrev_b32 v6, 3, v5 // (op 907) 

v_and_b32 v5, 15, v1 // (op 931) 
v_and_b32 v7, 63, v1 // (op 931) 
v_lshrrev_b32 v8, 4, v7 // (op 931) 

v_lshlrev_b32 v7, 5, v8 // (op 931) 

v_lshl_add_u32 v8, v5, 7, v7 // (op 931) 


v_lshlrev_b32 v5, 2, v8 // (op 931) 

v_lshrrev_b32 v7, 3, v5 // (op 931) 

v_lshrrev_b32 v5, 6, v1 // (op 955) 
v_and_b32 v8, 15, v1 // (op 955) 
v_lshl_add_u32 v9, v5, 4, v8 // (op 955) 


v_and_b32 v5, 63, v1 // (op 955) 
v_lshrrev_b32 v8, 4, v5 // (op 955) 

v_lshl_add_u32 v5, v9, 2, v8 // (op 955) 


v_mov_b32 v8, s14 // (op 2) Move value
v_mov_b32 v9, s15 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 61) call()
v_accvgpr_write a1, 0.00000 // (op 61) call()
v_accvgpr_write a2, 0.00000 // (op 61) call()
v_accvgpr_write a3, 0.00000 // (op 61) call()
v_lshrrev_b32 v10, 2, v1 // (op 453) 
v_lshl_add_u32 v11, s16, 5, v10 // (op 453) 

v_and_b32 v10, -4, v1 // (op 453) 
v_sub_u32 v12, v1, v10 // (op 453) 

v_lshlrev_b32 v10, 5, v12 // (op 453) 

v_mov_b32 v13, 0 // (op 453) convert
v_mov_b32 v12, v10 // (op 453) convert

v_mul_lo_u32 v15, v11, s3 // (op 453) most significant: low of low * high
v_mul_hi_u32 v10, v11, s2 // (op 453) most significant: high of low * low
v_mul_lo_u32 v14, v11, s2 // (op 453) least significant: low of low * low
v_add_u32 v15, v15, v10 // (op 453) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v12 // (op 453) least significant half
v_addc_co_u32 v15, s[0:1], v15, v13, s[0:1] // (op 453) most significant half



v_lshlrev_b64 v[10:11], 2, v[14:15] // (op 453) 

v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 453) 

v_mov_b32 v10, v12 // (op 453) convert

s_mul_i32 s1, s2, s7 // (op 455) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 455) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 455) least significant
s_mul_hi_u32 s15, s2, s6 // (op 455) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 455) most significant: sum
s_add_u32 s1, s1, s15 // (op 455) most significant: sum


s_lshl_b64 s[14:15], s[0:1], 2 // (op 455) 

s_ashr_i64 s[0:1], s[14:15], 3 // (op 455) 

s_mov_b32 s40, s4 // (op 455) reg expression
s_mov_b32 s41, s5 // (op 455) reg expression
s_mov_b32 s42, s0 // (op 455) reg expression
s_mov_b32 s43, 131072 // (op 455) call()

v_lshrrev_b32 v11, 2, v1 // (op 472) 
v_lshl_add_u32 v12, s16, 5, v11 // (op 472) 

v_and_b32 v11, -4, v1 // (op 472) 
v_sub_u32 v13, v1, v11 // (op 472) 

v_lshlrev_b32 v11, 5, v13 // (op 472) 

v_mov_b32 v15, 0 // (op 472) convert
v_mov_b32 v14, v11 // (op 472) convert

v_mul_lo_u32 v17, v12, s3 // (op 472) most significant: low of low * high
v_mul_hi_u32 v11, v12, s2 // (op 472) most significant: high of low * low
v_mul_lo_u32 v16, v12, s2 // (op 472) least significant: low of low * low
v_add_u32 v17, v17, v11 // (op 472) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 472) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 472) most significant half



v_lshlrev_b64 v[14:15], 2, v[16:17] // (op 472) 

v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 472) 

v_mov_b32 v11, v12 // (op 472) convert

s_mul_i32 s1, s2, s7 // (op 474) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 474) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 474) least significant
s_mul_hi_u32 s15, s2, s6 // (op 474) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 474) most significant: sum
s_add_u32 s1, s1, s15 // (op 474) most significant: sum


s_lshl_b64 s[14:15], s[0:1], 2 // (op 474) 

s_ashr_i64 s[0:1], s[14:15], 3 // (op 474) 

s_mov_b32 s40, s4 // (op 474) reg expression
s_mov_b32 s41, s5 // (op 474) reg expression
s_mov_b32 s42, s0 // (op 474) reg expression
s_mov_b32 s43, 131072 // (op 474) call()

v_and_b32 v12, -8, v1 // (op 517) 
v_sub_u32 v13, v1, v12 // (op 517) 

v_lshlrev_b32 v12, 4, v13 // (op 517) 

v_mov_b32 v15, 0 // (op 517) convert
v_mov_b32 v14, v12 // (op 517) convert

v_lshrrev_b32 v12, 3, v1 // (op 517) 
v_lshl_add_u32 v13, s17, 4, v12 // (op 517) 

v_mul_lo_u32 v17, v13, s9 // (op 517) most significant: low of low * high
v_mul_hi_u32 v12, v13, s8 // (op 517) most significant: high of low * low
v_mul_lo_u32 v16, v13, s8 // (op 517) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 517) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v16 // (op 517) least significant half
v_addc_co_u32 v13, s[0:1], v15, v17, s[0:1] // (op 517) most significant half

v_lshlrev_b64 v[12:13], 2, v[12:13] // (op 517) 


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 517) 

v_mov_b32 v12, v14 // (op 517) convert

s_mul_i32 s1, s8, s13 // (op 519) most significant: low of low * high
s_mul_i32 s14, s9, s12 // (op 519) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 519) least significant
s_mul_hi_u32 s15, s8, s12 // (op 519) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 519) most significant: sum
s_add_u32 s1, s1, s15 // (op 519) most significant: sum


s_lshl_b64 s[14:15], s[0:1], 2 // (op 519) 

s_ashr_i64 s[0:1], s[14:15], 3 // (op 519) 

s_mov_b32 s44, s10 // (op 519) reg expression
s_mov_b32 s45, s11 // (op 519) reg expression
s_mov_b32 s46, s0 // (op 519) reg expression
s_mov_b32 s47, 131072 // (op 519) call()

v_and_b32 v13, -8, v1 // (op 536) 
v_sub_u32 v14, v1, v13 // (op 536) 

v_lshlrev_b32 v13, 4, v14 // (op 536) 

v_mov_b32 v15, 0 // (op 536) convert
v_mov_b32 v14, v13 // (op 536) convert

v_lshrrev_b32 v13, 3, v1 // (op 536) 
v_lshl_add_u32 v16, s17, 4, v13 // (op 536) 

v_mul_lo_u32 v19, v16, s9 // (op 536) most significant: low of low * high
v_mul_hi_u32 v13, v16, s8 // (op 536) most significant: high of low * low
v_mul_lo_u32 v18, v16, s8 // (op 536) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 536) most significant: sum


v_add_co_u32 v16, s[0:1], v14, v18 // (op 536) least significant half
v_addc_co_u32 v17, s[0:1], v15, v19, s[0:1] // (op 536) most significant half

v_lshlrev_b64 v[16:17], 2, v[16:17] // (op 536) 


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 536) 

v_mov_b32 v13, v14 // (op 536) convert

s_mul_i32 s1, s8, s13 // (op 538) most significant: low of low * high
s_mul_i32 s14, s9, s12 // (op 538) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 538) least significant
s_mul_hi_u32 s15, s8, s12 // (op 538) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 538) most significant: sum
s_add_u32 s1, s1, s15 // (op 538) most significant: sum


s_lshl_b64 s[14:15], s[0:1], 2 // (op 538) 

s_ashr_i64 s[0:1], s[14:15], 3 // (op 538) 

s_mov_b32 s44, s10 // (op 538) reg expression
s_mov_b32 s45, s11 // (op 538) reg expression
s_mov_b32 s46, s0 // (op 538) reg expression
s_mov_b32 s47, 131072 // (op 538) call()

v_lshrrev_b32 v14, 2, v1 // (op 581) 
v_lshl_add_u32 v15, s16, 5, v14 // (op 581) 

v_and_b32 v14, -4, v1 // (op 581) 
v_sub_u32 v16, v1, v14 // (op 581) 


s_waitcnt lgkmcnt(0)
v_mul_lo_u32 v14, v15, s20 // (op 581) 
v_add_u32 v14, v14, v16 // (op 581) 


s_mul_i32 s1, s20, s7 // (op 583) most significant: low of low * high
s_mul_i32 s14, s21, s6 // (op 583) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 583) least significant
s_mul_hi_u32 s15, s20, s6 // (op 583) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 583) most significant: sum
s_add_u32 s1, s1, s15 // (op 583) most significant: sum


s_mov_b32 s48, s22 // (op 583) reg expression
s_mov_b32 s49, s23 // (op 583) reg expression
s_mov_b32 s50, s0 // (op 583) reg expression
s_mov_b32 s51, 131072 // (op 583) call()

v_lshrrev_b32 v15, 2, v1 // (op 600) 
v_lshl_add_u32 v16, s16, 5, v15 // (op 600) 

v_and_b32 v15, -4, v1 // (op 600) 
v_sub_u32 v17, v1, v15 // (op 600) 

v_mul_lo_u32 v15, v16, s20 // (op 600) 
v_add_u32 v15, v15, v17 // (op 600) 


s_mul_i32 s1, s20, s7 // (op 602) most significant: low of low * high
s_mul_i32 s14, s21, s6 // (op 602) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 602) least significant
s_mul_hi_u32 s15, s20, s6 // (op 602) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 602) most significant: sum
s_add_u32 s1, s1, s15 // (op 602) most significant: sum


s_mov_b32 s48, s22 // (op 602) reg expression
s_mov_b32 s49, s23 // (op 602) reg expression
s_mov_b32 s50, s0 // (op 602) reg expression
s_mov_b32 s51, 131072 // (op 602) call()

v_and_b32 v16, 15, v1 // (op 645) 
v_lshl_add_u32 v17, s17, 4, v16 // (op 645) 

v_mul_lo_u32 v16, v17, s24 // (op 645) 

v_and_b32 v17, 63, v1 // (op 645) 
v_lshrrev_b32 v18, 4, v17 // (op 645) 

v_add_u32 v17, v18, v16 // (op 645) 


s_mul_i32 s1, s24, s13 // (op 647) most significant: low of low * high
s_mul_i32 s14, s25, s12 // (op 647) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 647) least significant
s_mul_hi_u32 s15, s24, s12 // (op 647) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 647) most significant: sum
s_add_u32 s1, s1, s15 // (op 647) most significant: sum


s_mov_b32 s52, s26 // (op 647) reg expression
s_mov_b32 s53, s27 // (op 647) reg expression
s_mov_b32 s54, s0 // (op 647) reg expression
s_mov_b32 s55, 131072 // (op 647) call()

v_and_b32 v16, 15, v1 // (op 664) 
v_lshl_add_u32 v18, s17, 4, v16 // (op 664) 

v_mul_lo_u32 v16, v18, s24 // (op 664) 

v_and_b32 v18, 63, v1 // (op 664) 
v_lshrrev_b32 v19, 4, v18 // (op 664) 

v_add_u32 v18, v19, v16 // (op 664) 


s_mul_i32 s1, s24, s13 // (op 666) most significant: low of low * high
s_mul_i32 s14, s25, s12 // (op 666) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 666) least significant
s_mul_hi_u32 s15, s24, s12 // (op 666) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 666) most significant: sum
s_add_u32 s1, s1, s15 // (op 666) most significant: sum


s_mov_b32 s52, s26 // (op 666) reg expression
s_mov_b32 s53, s27 // (op 666) reg expression
s_mov_b32 s54, s0 // (op 666) reg expression
s_mov_b32 s55, 131072 // (op 666) call()




buffer_load_ubyte v16, v17, s[52:55], 0 offen offset: 0 // (op 341) Load value



buffer_load_dwordx2 v[20:21], v12, s[44:47], 0 offen // (op 345) Load value



buffer_load_dwordx4 v[22:25], v10, s[40:43], 0 offen // (op 348) Load value



buffer_load_ubyte v19, v14, s[48:51], 0 offen offset: 0 // (op 351) Load value



buffer_load_ubyte v26, v17, s[52:55], 0 offen offset:4 // (op 355) Load value




buffer_load_dwordx2 v[28:29], v12, s[44:47], 0 offen offset:64 // (op 359) Load value




buffer_load_dwordx4 v[30:33], v10, s[40:43], 0 offen offset:64 // (op 362) Load value




buffer_load_ubyte v10, v14, s[48:51], 0 offen offset:4 // (op 365) Load value

v_and_b32 v12, -8, v1 // (op 784) 
v_sub_u32 v14, v1, v12 // (op 784) 

v_lshlrev_b32 v12, 4, v14 // (op 784) 

v_lshrrev_b32 v14, 3, v1 // (op 784) 
v_lshl_add_u32 v17, v14, 7, v12 // (op 784) 


v_lshlrev_b32 v12, 2, v17 // (op 784) 

v_lshrrev_b32 v14, 3, v12 // (op 784) 




s_waitcnt vmcnt(6)
ds_write_b64 v14, v[20:21] // (op 367) Store local data 

v_and_b32 v12, -4, v1 // (op 731) 
v_sub_u32 v14, v1, v12 // (op 731) 

v_lshlrev_b32 v12, 5, v14 // (op 731) 

v_lshrrev_b32 v14, 2, v1 // (op 731) 
v_lshl_add_u32 v17, v14, 7, v12 // (op 731) 


v_lshlrev_b32 v12, 2, v17 // (op 731) 

v_lshrrev_b32 v14, 3, v12 // (op 731) 




s_waitcnt vmcnt(5)
ds_write_b128 v14, v[22:25] offset:1024 // (op 368) Store local data 

v_and_b32 v12, -4, v1 // (op 837) 
v_sub_u32 v14, v1, v12 // (op 837) 

v_and_b32 v12, 4294967292, v1 // (op 837) 
v_add_u32 v17, v12, v14 // (op 837) 





s_waitcnt vmcnt(4)
ds_write_b8 v17, v19 offset:3072 // (op 369) Store local data 

v_and_b32 v12, -4, v1 // (op 702) 
v_sub_u32 v14, v1, v12 // (op 702) 

v_lshlrev_b32 v12, 5, v14 // (op 702) 

v_lshrrev_b32 v14, 2, v1 // (op 702) 
v_lshl_add_u32 v17, v14, 7, v12 // (op 702) 


v_lshlrev_b32 v12, 2, v17 // (op 702) 

v_lshrrev_b32 v14, 3, v12 // (op 702) 

v_and_b32 v12, -8, v1 // (op 755) 
v_sub_u32 v17, v1, v12 // (op 755) 

v_lshlrev_b32 v12, 4, v17 // (op 755) 

v_lshrrev_b32 v17, 3, v1 // (op 755) 
v_lshl_add_u32 v27, v17, 7, v12 // (op 755) 


v_lshlrev_b32 v12, 2, v27 // (op 755) 

v_lshrrev_b32 v17, 3, v12 // (op 755) 

v_and_b32 v12, -4, v1 // (op 808) 
v_sub_u32 v27, v1, v12 // (op 808) 

v_and_b32 v12, 4294967292, v1 // (op 808) 
v_add_u32 v34, v12, v27 // (op 808) 


v_and_b32 v12, -4, v1 // (op 895) 
v_sub_u32 v27, v1, v12 // (op 895) 

v_lshlrev_b32 v12, 5, v27 // (op 895) 

v_lshrrev_b32 v27, 2, v1 // (op 895) 
v_lshl_add_u32 v35, v27, 7, v12 // (op 895) 


v_lshlrev_b32 v12, 2, v35 // (op 895) 

v_lshrrev_b32 v27, 3, v12 // (op 895) 

v_and_b32 v12, -8, v1 // (op 919) 
v_sub_u32 v35, v1, v12 // (op 919) 

v_lshlrev_b32 v12, 4, v35 // (op 919) 

v_lshrrev_b32 v35, 3, v1 // (op 919) 
v_lshl_add_u32 v36, v35, 7, v12 // (op 919) 


v_lshlrev_b32 v12, 2, v36 // (op 919) 

v_lshrrev_b32 v35, 3, v12 // (op 919) 

v_and_b32 v12, -4, v1 // (op 943) 
v_sub_u32 v36, v1, v12 // (op 943) 

v_and_b32 v12, 4294967292, v1 // (op 943) 
v_add_u32 v37, v12, v36 // (op 943) 


s_mov_b32 s0, 0 // (op 57) call()
s_ashr_i64 s[14:15], s[28:29], 7 // (op 56) 
s_mov_b32 s1, s14 // (op 56) convert

s_and_b32 s14, s1, 4294967294 // (op 56) 

s_cmp_lt_u32 s0, s14 // (op 56) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoop_56 // (op 56) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoop_56 if false)
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoop_56:
s_add_u32 s0, s0, 2 // (op 58) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 988) 



ds_read_b128 v[38:41], v3 offset:1024 // (op 67) Load local data 
s_barrier  // (op 1001) 



ds_read_b128 v[42:45], v4 // (op 72) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1015) 



ds_read_u8 v12, v0 offset:3072 // (op 77) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[38:41], v[42:45], a[0:3], v12, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 84) 





v_mov_b32 v12, v18 // (op 82) 

buffer_load_ubyte v16, v12, s[52:55], 0 offen offset:8 // (op 82) Load value


v_mov_b32 v36, v13 // (op 10) 

buffer_load_dwordx2 v[20:21], v36, s[44:47], 0 offen offset:128 // (op 10) Load value


v_mov_b32 v38, v11 // (op 4) 

buffer_load_dwordx4 v[22:25], v38, s[40:43], 0 offen offset:128 // (op 4) Load value


v_mov_b32 v39, v15 // (op 17) 

buffer_load_ubyte v19, v39, s[48:51], 0 offen offset:8 // (op 17) Load value
s_barrier  // (op 1038) 



s_waitcnt vmcnt(6)
ds_write_b64 v35, v[28:29] offset:3200 // (op 239) Store local data 
s_barrier  // (op 1032) 



s_waitcnt vmcnt(5)
ds_write_b128 v27, v[30:33] offset:4224 // (op 229) Store local data 
s_barrier  // (op 1044) 



s_waitcnt vmcnt(4)
ds_write_b8 v37, v10 offset:6272 // (op 240) Store local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1029) 



ds_read_b128 v[40:43], v6 offset:4224 // (op 231) Load local data 
s_barrier  // (op 1035) 
s_barrier  // (op 1099) 



ds_read_b128 v[44:47], v7 offset:3200 // (op 233) Load local data 
s_barrier  // (op 1105) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1041) 



ds_read_u8 v48, v5 offset:6272 // (op 235) Load local data 
s_barrier  // (op 1111) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[40:43], v[44:47], a[0:3], v48, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 238) 






buffer_load_ubyte v26, v12, s[52:55], 0 offen offset:12 // (op 237) Load value




v_add_u32 v18, 8, v18 // (op 656) 
buffer_load_dwordx2 v[28:29], v36, s[44:47], 0 offen offset:192 // (op 241) Load value




v_add_u32 v13, 128, v13 // (op 528) 
buffer_load_dwordx4 v[30:33], v38, s[40:43], 0 offen offset:192 // (op 228) Load value




v_add_u32 v11, 128, v11 // (op 464) 
buffer_load_ubyte v10, v39, s[48:51], 0 offen offset:12 // (op 242) Load value

s_barrier  // (op 1004) 
v_add_u32 v15, 8, v15 // (op 592) 



s_waitcnt vmcnt(6)
ds_write_b64 v17, v[20:21] // (op 13) Store local data 
s_barrier  // (op 991) 



s_waitcnt vmcnt(5)
ds_write_b128 v14, v[22:25] offset:1024 // (op 6) Store local data 
s_barrier  // (op 1018) 



s_waitcnt vmcnt(4)
ds_write_b8 v34, v19 offset:3072 // (op 21) Store local data 
s_ashr_i64 s[14:15], s[28:29], 7 // (op 56) 
s_mov_b32 s1, s14 // (op 56) convert

s_and_b32 s14, s1, 4294967294 // (op 56) 

s_cmp_lt_u32 s0, s14 // (op 56) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoop_56 // (op 56) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoop_56 if true)
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoop_56:















v_lshrrev_b32 v27, 2, v1 // (op 427) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 427) 
v_lshl_add_u32 v11, s16, 5, v27 // (op 427) 

s_lshr_b32 s14, s0, 1 // (op 427) 

v_and_b32 v27, -4, v1 // (op 427) 
v_sub_u32 v12, v1, v27 // (op 427) 

v_lshlrev_b32 v27, 5, v12 // (op 427) 

v_lshl_add_u32 v12, s14, 8, v27 // (op 427) 


v_mov_b32 v25, 0 // (op 427) convert
v_mov_b32 v24, v12 // (op 427) convert

v_mul_lo_u32 v13, v11, s3 // (op 427) most significant: low of low * high
v_mul_hi_u32 v27, v11, s2 // (op 427) most significant: high of low * low
v_mul_lo_u32 v12, v11, s2 // (op 427) least significant: low of low * low
v_add_u32 v13, v13, v27 // (op 427) most significant: sum

v_add_co_u32 v12, s[0:1], v12, v24 // (op 427) least significant half
v_addc_co_u32 v13, s[0:1], v13, v25, s[0:1] // (op 427) most significant half



v_lshlrev_b64 v[14:15], 2, v[12:13] // (op 427) 

v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 427) 

v_mov_b32 v11, v12 // (op 427) convert

s_mul_i32 s1, s2, s7 // (op 429) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 429) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 429) least significant
s_mul_hi_u32 s15, s2, s6 // (op 429) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 429) most significant: sum
s_add_u32 s1, s1, s15 // (op 429) most significant: sum


s_lshl_b64 s[14:15], s[0:1], 2 // (op 429) 

s_ashr_i64 s[0:1], s[14:15], 3 // (op 429) 

s_mov_b32 s40, s4 // (op 429) reg expression
s_mov_b32 s41, s5 // (op 429) reg expression
s_mov_b32 s42, s0 // (op 429) reg expression
s_mov_b32 s43, 131072 // (op 429) call()

v_and_b32 v27, -8, v1 // (op 494) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 494) 
v_sub_u32 v12, v1, v27 // (op 494) 

s_lshr_b32 s14, s0, 1 // (op 494) 

v_lshlrev_b32 v27, 4, v12 // (op 494) 

v_lshl_add_u32 v12, s14, 8, v27 // (op 494) 


v_mov_b32 v25, 0 // (op 494) convert
v_mov_b32 v24, v12 // (op 494) convert

v_lshrrev_b32 v27, 3, v1 // (op 494) 
v_lshl_add_u32 v12, s17, 4, v27 // (op 494) 

v_mul_lo_u32 v23, v12, s9 // (op 494) most significant: low of low * high
v_mul_hi_u32 v27, v12, s8 // (op 494) most significant: high of low * low
v_mul_lo_u32 v22, v12, s8 // (op 494) least significant: low of low * low
v_add_u32 v23, v23, v27 // (op 494) most significant: sum


v_add_co_u32 v12, s[0:1], v24, v22 // (op 494) least significant half
v_addc_co_u32 v13, s[0:1], v25, v23, s[0:1] // (op 494) most significant half

v_lshlrev_b64 v[12:13], 2, v[12:13] // (op 494) 


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 494) 

v_mov_b32 v27, v14 // (op 494) convert

s_mul_i32 s1, s8, s13 // (op 496) most significant: low of low * high
s_mul_i32 s14, s9, s12 // (op 496) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 496) least significant
s_mul_hi_u32 s15, s8, s12 // (op 496) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 496) most significant: sum
s_add_u32 s1, s1, s15 // (op 496) most significant: sum


s_lshl_b64 s[14:15], s[0:1], 2 // (op 496) 

s_ashr_i64 s[0:1], s[14:15], 3 // (op 496) 

s_mov_b32 s44, s10 // (op 496) reg expression
s_mov_b32 s45, s11 // (op 496) reg expression
s_mov_b32 s46, s0 // (op 496) reg expression
s_mov_b32 s47, 131072 // (op 496) call()

v_lshrrev_b32 v12, 2, v1 // (op 558) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 558) 
v_lshl_add_u32 v13, s16, 5, v12 // (op 558) 

s_lshr_b32 s14, s0, 1 // (op 558) 

v_and_b32 v12, -4, v1 // (op 558) 
v_sub_u32 v14, v1, v12 // (op 558) 

v_lshl_add_u32 v12, s14, 3, v14 // (op 558) 


v_mul_lo_u32 v14, v13, s20 // (op 558) 
v_add_u32 v14, v14, v12 // (op 558) 


s_mul_i32 s1, s20, s7 // (op 560) most significant: low of low * high
s_mul_i32 s14, s21, s6 // (op 560) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 560) least significant
s_mul_hi_u32 s15, s20, s6 // (op 560) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 560) most significant: sum
s_add_u32 s1, s1, s15 // (op 560) most significant: sum


s_mov_b32 s48, s22 // (op 560) reg expression
s_mov_b32 s49, s23 // (op 560) reg expression
s_mov_b32 s50, s0 // (op 560) reg expression
s_mov_b32 s51, 131072 // (op 560) call()

v_and_b32 v12, 63, v1 // (op 622) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 622) 
v_lshrrev_b32 v13, 4, v12 // (op 622) 

s_lshr_b32 s14, s0, 1 // (op 622) 

v_and_b32 v12, 15, v1 // (op 622) 
v_lshl_add_u32 v15, s14, 3, v13 // (op 622) 


v_lshl_add_u32 v13, s17, 4, v12 // (op 622) 

v_mul_lo_u32 v12, v13, s24 // (op 622) 

v_add_u32 v13, v15, v12 // (op 622) 


s_mul_i32 s1, s24, s13 // (op 624) most significant: low of low * high
s_mul_i32 s14, s25, s12 // (op 624) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 624) least significant
s_mul_hi_u32 s15, s24, s12 // (op 624) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 624) most significant: sum
s_add_u32 s1, s1, s15 // (op 624) most significant: sum


s_mov_b32 s52, s26 // (op 624) reg expression
s_mov_b32 s53, s27 // (op 624) reg expression
s_mov_b32 s54, s0 // (op 624) reg expression
s_mov_b32 s55, 131072 // (op 624) call()

v_and_b32 v12, -4, v1 // (op 715) 
v_sub_u32 v15, v1, v12 // (op 715) 

v_lshlrev_b32 v12, 5, v15 // (op 715) 

v_lshrrev_b32 v15, 2, v1 // (op 715) 
v_lshl_add_u32 v16, v15, 7, v12 // (op 715) 


v_lshlrev_b32 v12, 2, v16 // (op 715) 

v_lshrrev_b32 v15, 3, v12 // (op 715) 

v_and_b32 v12, -8, v1 // (op 768) 
v_sub_u32 v16, v1, v12 // (op 768) 

v_lshlrev_b32 v12, 4, v16 // (op 768) 

v_lshrrev_b32 v16, 3, v1 // (op 768) 
v_lshl_add_u32 v17, v16, 7, v12 // (op 768) 


v_lshlrev_b32 v12, 2, v17 // (op 768) 

v_lshrrev_b32 v16, 3, v12 // (op 768) 

v_and_b32 v12, -4, v1 // (op 821) 
v_sub_u32 v17, v1, v12 // (op 821) 

v_and_b32 v12, 4294967292, v1 // (op 821) 
v_add_u32 v18, v12, v17 // (op 821) 


s_ashr_i64 s[0:1], s[28:29], 7 // (op 166) 
s_and_b32 s14, s0, 4294967294 // (op 166) 

s_ashr_i64 s[0:1], s[28:29], 7 // (op 165) 
s_mov_b32 s15, s0 // (op 165) convert

s_cmp_lt_u32 s14, s15 // (op 165) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoopTail_165 // (op 165) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoopTail_165 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoopTail_165:


s_add_u32 s14, s14, 1 // (op 167) 
v_mov_b32 v10, v11 // (op 170) 

buffer_load_dwordx4 v[20:23], v10, s[40:43], 0 offen // (op 170) Load value

v_add_u32 v11, 64, v11 // (op 417) 


s_barrier  // (op 994) 
v_mov_b32 v10, v27 // (op 183) 




buffer_load_dwordx2 v[24:25], v10, s[44:47], 0 offen // (op 183) Load value

v_add_u32 v27, 64, v27 // (op 486) 
s_waitcnt vmcnt(1)
ds_write_b128 v15, v[20:23] offset:1024 // (op 171) Store local data 



v_mov_b32 v10, v14 // (op 184) 

s_barrier  // (op 1007) 
buffer_load_ubyte v12, v10, s[48:51], 0 offen offset: 0 // (op 184) Load value




v_add_u32 v14, 4, v14 // (op 550) 
s_waitcnt vmcnt(1)
ds_write_b64 v16, v[24:25] // (op 181) Store local data 

s_waitcnt lgkmcnt(1)
s_barrier  // (op 997) 



s_barrier  // (op 1021) 
ds_read_b128 v[20:23], v3 offset:1024 // (op 173) Load local data 



s_waitcnt lgkmcnt(1)
s_barrier  // (op 1011) 
s_waitcnt vmcnt(0)
ds_write_b8 v18, v12 offset:3072 // (op 182) Store local data 




ds_read_b128 v[28:31], v4 // (op 175) Load local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1025) 



ds_read_u8 v10, v0 offset:3072 // (op 177) Load local data 


v_mov_b32 v12, v13 // (op 179) 

buffer_load_ubyte v17, v12, s[52:55], 0 offen offset: 0 // (op 179) Load value

v_add_u32 v13, 4, v13 // (op 614) 
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:23], v[28:31], a[0:3], v10, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 180) 




s_ashr_i64 s[0:1], s[28:29], 7 // (op 165) 
s_mov_b32 s15, s0 // (op 165) convert

s_cmp_lt_u32 s14, s15 // (op 165) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoopTail_165 // (op 165) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoopTail_165 if true)
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoopTail_165:






















s_barrier  // (op 1093) 






v_cmp_eq_f32 vcc, 0.00000, v8 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalFalse_0_beta_125
s_nop 0
v_accvgpr_read v4, a0 // (op 46) Wait state hazard: XDL Write Hazard
v_accvgpr_read v5, a1 // (op 46) 
v_accvgpr_read v6, a2 // (op 46) 
v_accvgpr_read v7, a3 // (op 46) 
v_mul_f32 v10, v9, v4 // (op 46) 
v_mul_f32 v11, v9, v5 // (op 46) 
v_mul_f32 v12, v9, v6 // (op 46) 
v_mul_f32 v13, v9, v7 // (op 46) 

v_cvt_f16_f32 v4, v10 // (op 52) 
v_cvt_f16_f32 v5, v11 // (op 52) 
v_cvt_f16_f32 v6, v12 // (op 52) 
v_cvt_f16_f32 v7, v13 // (op 52) 

v_lshrrev_b32 v0, 6, v1 // (op 863) 
v_and_b32 v3, 63, v1 // (op 863) 
v_lshrrev_b32 v10, 4, v3 // (op 863) 
v_lshlrev_b32 v11, 2, v10 // (op 863) 

v_lshl_add_u32 v10, v0, 4, v11 // (op 863) 


v_lshl_add_u32 v0, s16, 5, v10 // (op 863) 

v_lshrrev_b32 v10, 2, v3 // (op 863) 

v_and_b32 v3, 3, v10 // (op 863) 

v_and_b32 v10, 3, v1 // (op 863) 
v_lshl_add_u32 v11, v3, 2, v10 // (op 863) 


v_lshl_add_u32 v3, s17, 4, v11 // (op 863) 

v_mul_lo_u32 v10, v3, s30 // (op 863) 

v_add_lshl_u32 v3, v0, v10, 1 // (op 863) 


s_mul_i32 s15, s30, s13 // (op 865) most significant: low of low * high
s_mul_i32 s0, s31, s12 // (op 865) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 865) least significant
s_mul_hi_u32 s1, s30, s12 // (op 865) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 865) most significant: sum
s_add_u32 s15, s15, s1 // (op 865) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 1 // (op 865) 

s_mov_b32 s8, s32 // (op 865) reg expression
s_mov_b32 s9, s33 // (op 865) reg expression
s_mov_b32 s10, s0 // (op 865) reg expression
s_mov_b32 s11, 131072 // (op 865) call()


v_pack_b32_f16 v10, v4, v5 // (op 54) 
v_pack_b32_f16 v11, v6, v7 // (op 54) 


buffer_store_dwordx2 v[10:11], v3, s[8:11], 0 offen // (op 54) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalBottom_0_beta_125
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalFalse_0_beta_125:
v_lshrrev_b32 v0, 6, v1 // (op 685) 
v_and_b32 v3, 63, v1 // (op 685) 
v_lshrrev_b32 v4, 4, v3 // (op 685) 
v_lshlrev_b32 v5, 2, v4 // (op 685) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 685) 


v_lshl_add_u32 v0, s16, 5, v4 // (op 685) 

v_lshrrev_b32 v4, 2, v3 // (op 685) 

v_and_b32 v3, 3, v4 // (op 685) 

v_and_b32 v4, 3, v1 // (op 685) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 685) 


v_lshl_add_u32 v3, s17, 4, v5 // (op 685) 

v_mul_lo_u32 v4, v3, s34 // (op 685) 

v_add_lshl_u32 v3, v0, v4, 1 // (op 685) 


s_mul_i32 s15, s34, s13 // (op 687) most significant: low of low * high
s_mul_i32 s0, s35, s12 // (op 687) most significant: low of high * low
s_mul_i32 s14, s34, s12 // (op 687) least significant
s_mul_hi_u32 s1, s34, s12 // (op 687) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 687) most significant: sum
s_add_u32 s15, s15, s1 // (op 687) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 1 // (op 687) 

s_mov_b32 s4, s36 // (op 687) reg expression
s_mov_b32 s5, s37 // (op 687) reg expression
s_mov_b32 s6, s0 // (op 687) reg expression
s_mov_b32 s7, 131072 // (op 687) call()





buffer_load_dwordx2 v[4:5], v3, s[4:7], 0 offen // (op 138) Load value



v_accvgpr_read v10, a0 // (op 134) 
v_accvgpr_read v11, a1 // (op 134) 
v_accvgpr_read v12, a2 // (op 134) 
v_accvgpr_read v13, a3 // (op 134) 
v_mul_f32 v14, v9, v10 // (op 134) 
v_mul_f32 v15, v9, v11 // (op 134) 
v_mul_f32 v16, v9, v12 // (op 134) 
v_mul_f32 v17, v9, v13 // (op 134) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 139) 
v_lshrrev_b32 v7, 16, v4 // (op 139) 
v_cvt_f32_f16 v7, v7 // (op 139) 
v_mul_f32 v10, v8, v6 // (op 139) 
v_mul_f32 v11, v8, v7 // (op 139) 
v_cvt_f32_f16 v6, v5 // (op 139) 
v_lshrrev_b32 v7, 16, v5 // (op 139) 
v_cvt_f32_f16 v7, v7 // (op 139) 
v_mul_f32 v12, v8, v6 // (op 139) 
v_mul_f32 v13, v8, v7 // (op 139) 


v_add_f32 v4, v10, v14 // (op 135) 
v_add_f32 v5, v11, v15 // (op 135) 
v_add_f32 v6, v12, v16 // (op 135) 
v_add_f32 v7, v13, v17 // (op 135) 


v_cvt_f16_f32 v10, v4 // (op 136) 
v_cvt_f16_f32 v11, v5 // (op 136) 
v_cvt_f16_f32 v12, v6 // (op 136) 
v_cvt_f16_f32 v13, v7 // (op 136) 

v_lshrrev_b32 v0, 6, v1 // (op 880) 
v_and_b32 v3, 63, v1 // (op 880) 
v_lshrrev_b32 v4, 4, v3 // (op 880) 
v_lshlrev_b32 v5, 2, v4 // (op 880) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 880) 


v_lshl_add_u32 v0, s16, 5, v4 // (op 880) 

v_lshrrev_b32 v4, 2, v3 // (op 880) 

v_and_b32 v3, 3, v4 // (op 880) 

v_and_b32 v4, 3, v1 // (op 880) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 880) 


v_lshl_add_u32 v3, s17, 4, v5 // (op 880) 

v_mul_lo_u32 v4, v3, s30 // (op 880) 

v_add_lshl_u32 v3, v0, v4, 1 // (op 880) 


s_mul_i32 s15, s30, s13 // (op 882) most significant: low of low * high
s_mul_i32 s0, s31, s12 // (op 882) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 882) least significant
s_mul_hi_u32 s1, s30, s12 // (op 882) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 882) most significant: sum
s_add_u32 s15, s15, s1 // (op 882) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 1 // (op 882) 

s_mov_b32 s8, s32 // (op 882) reg expression
s_mov_b32 s9, s33 // (op 882) reg expression
s_mov_b32 s10, s0 // (op 882) reg expression
s_mov_b32 s11, 131072 // (op 882) call()


v_pack_b32_f16 v4, v10, v11 // (op 137) 
v_pack_b32_f16 v5, v12, v13 // (op 137) 


buffer_store_dwordx2 v[4:5], v3, s[8:11], 0 offen // (op 137) Store value






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalBottom_0_beta_125:




s_endpgm  // End of RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2
.LRR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_end:
.size RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2, .LRR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_end-RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2
  .amdhsa_next_free_vgpr 56
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 6400 // lds bytes
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
  ProblemType:
    OperationType: GEMM
    DataType: F4
    DataTypeA: F4
    DataTypeB: F4
    DestDataType: h
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
    macrotile: [32, 16, 128]
    threads: [128, 1, 1]
    grid: [TilesX, TilesY, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2
    .symbol: RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 6400
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 56
    .vgpr_count: 49
    .agpr_count: 4
    .max_flat_workgroup_size: 128
    .workgroup_size: [128, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, {is-null: true}]
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
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: FP4, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP4, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: FP4, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP4, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 208, name: Tensor_21_pointer, variableType: {dataType: Half, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Half, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 160, name: Tensor_10_pointer, variableType: {dataType: Half, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Half, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
...

.end_amdgpu_metadata

