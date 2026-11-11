// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2
.p2align 8
.type RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2,@function
RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_exec_begin:


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


v_and_b32 v0, 63, v1 // (op 743) 
v_and_b32 v3, 48, v1 // (op 743) 
v_sub_u32 v4, v0, v3 // (op 743) 

v_lshrrev_b32 v3, 4, v0 // (op 743) 

v_lshl_add_u32 v0, v3, 5, v4 // (op 743) 


v_lshrrev_b32 v3, 7, v1 // (op 743) 
v_lshlrev_b32 v4, 4, v3 // (op 743) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 743) 


v_lshlrev_b32 v0, 2, v3 // (op 743) 

v_lshrrev_b32 v3, 3, v0 // (op 743) 

v_and_b32 v0, 48, v1 // (op 796) 
v_and_b32 v4, 63, v1 // (op 796) 
v_sub_u32 v5, v4, v0 // (op 796) 

v_lshrrev_b32 v4, 1, v5 // (op 796) 
v_add_lshl_u32 v6, v0, v4, 5 // (op 796) 


v_and_b32 v0, 1, v5 // (op 796) 

v_lshlrev_b32 v4, 3, v0 // (op 796) 

v_lshrrev_b32 v0, 6, v1 // (op 796) 
v_and_b32 v5, 1, v0 // (op 796) 

v_lshl_add_u32 v0, v5, 4, v4 // (op 796) 


v_add_u32 v4, v6, v0 // (op 796) 


v_lshrrev_b32 v0, 6, v1 // (op 849) 
v_and_b32 v5, 1, v0 // (op 849) 

v_and_b32 v0, 15, v1 // (op 849) 
v_lshl_add_u32 v6, v5, 4, v0 // (op 849) 


v_and_b32 v0, 63, v1 // (op 849) 
v_lshrrev_b32 v5, 4, v0 // (op 849) 

v_lshl_add_u32 v0, v5, 5, v6 // (op 849) 


v_and_b32 v5, 63, v1 // (op 907) 
v_and_b32 v6, 48, v1 // (op 907) 
v_sub_u32 v7, v5, v6 // (op 907) 

v_lshrrev_b32 v6, 4, v5 // (op 907) 

v_lshl_add_u32 v5, v6, 5, v7 // (op 907) 


v_lshrrev_b32 v6, 7, v1 // (op 907) 
v_lshlrev_b32 v7, 4, v6 // (op 907) 

v_lshl_add_u32 v6, v5, 4, v7 // (op 907) 


v_lshlrev_b32 v5, 2, v6 // (op 907) 

v_lshrrev_b32 v6, 3, v5 // (op 907) 

v_and_b32 v5, 48, v1 // (op 931) 
v_and_b32 v7, 63, v1 // (op 931) 
v_sub_u32 v8, v7, v5 // (op 931) 

v_lshrrev_b32 v7, 1, v8 // (op 931) 
v_add_lshl_u32 v9, v5, v7, 5 // (op 931) 


v_and_b32 v5, 1, v8 // (op 931) 

v_lshlrev_b32 v7, 3, v5 // (op 931) 

v_lshrrev_b32 v5, 6, v1 // (op 931) 
v_and_b32 v8, 1, v5 // (op 931) 

v_lshl_add_u32 v5, v8, 4, v7 // (op 931) 


v_add_u32 v7, v9, v5 // (op 931) 


v_lshrrev_b32 v5, 6, v1 // (op 955) 
v_and_b32 v8, 1, v5 // (op 955) 

v_and_b32 v5, 15, v1 // (op 955) 
v_lshl_add_u32 v9, v8, 4, v5 // (op 955) 


v_and_b32 v5, 63, v1 // (op 955) 
v_lshrrev_b32 v8, 4, v5 // (op 955) 

v_lshl_add_u32 v5, v8, 5, v9 // (op 955) 


v_mov_b32 v8, s14 // (op 2) Move value
v_mov_b32 v9, s15 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 59) call()
v_accvgpr_write a1, 0.00000 // (op 59) call()
v_accvgpr_write a2, 0.00000 // (op 59) call()
v_accvgpr_write a3, 0.00000 // (op 59) call()
s_lshl_b32 s0, s16, 4 // (op 453) 
v_mul_lo_u32 v11, v1, s3 // (op 453) most significant: low of low * high
s_mov_b32 s15, 0 // (op 453) convert
s_mov_b32 s14, s0 // (op 453) convert

v_mul_hi_u32 v12, v1, s2 // (op 453) most significant: high of low * low
v_mul_lo_u32 v10, v1, s2 // (op 453) least significant: low of low * low
v_add_u32 v11, v11, v12 // (op 453) most significant: sum

v_mov_b32 v12, s14 // (op 453) 
v_mov_b32 v13, s15 // (op 453) 
v_add_co_u32 v14, s[0:1], v12, v10 // (op 453) least significant half
v_addc_co_u32 v15, s[0:1], v13, v11, s[0:1] // (op 453) most significant half



v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 453) 


v_lshrrev_b64 v[10:11], 3, v[14:15] // (op 453) 

v_mov_b32 v12, v10 // (op 453) convert

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

s_lshl_b32 s0, s16, 4 // (op 472) 
v_mul_lo_u32 v11, v1, s3 // (op 472) most significant: low of low * high
v_mul_hi_u32 v13, v1, s2 // (op 472) most significant: high of low * low
v_mul_lo_u32 v10, v1, s2 // (op 472) least significant: low of low * low
v_add_u32 v11, v11, v13 // (op 472) most significant: sum

s_mov_b32 s15, 0 // (op 472) convert
s_mov_b32 s14, s0 // (op 472) convert

v_mov_b32 v13, s14 // (op 472) 
v_mov_b32 v14, s15 // (op 472) 
v_add_co_u32 v16, s[0:1], v13, v10 // (op 472) least significant half
v_addc_co_u32 v17, s[0:1], v14, v11, s[0:1] // (op 472) most significant half



v_lshlrev_b64 v[16:17], 2, v[16:17] // (op 472) 


v_lshrrev_b64 v[10:11], 3, v[16:17] // (op 472) 

v_mov_b32 v13, v10 // (op 472) convert

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

v_and_b32 v10, -2, v1 // (op 517) 
v_sub_u32 v11, v1, v10 // (op 517) 

v_lshlrev_b32 v10, 4, v11 // (op 517) 

v_lshl_add_u32 v11, s17, 5, v10 // (op 517) 

v_and_b32 v10, 4294967294, v1 // (op 517) 
v_mul_lo_u32 v14, v10, s8 // (op 517) 
v_add_u32 v14, v14, v11 // (op 517) 


s_mul_i32 s1, s8, s7 // (op 519) most significant: low of low * high
s_mul_i32 s14, s9, s6 // (op 519) most significant: low of high * low
s_mul_i32 s0, s8, s6 // (op 519) least significant
s_mul_hi_u32 s15, s8, s6 // (op 519) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 519) most significant: sum
s_add_u32 s1, s1, s15 // (op 519) most significant: sum


s_mov_b32 s44, s10 // (op 519) reg expression
s_mov_b32 s45, s11 // (op 519) reg expression
s_mov_b32 s46, s0 // (op 519) reg expression
s_mov_b32 s47, 131072 // (op 519) call()

v_and_b32 v10, -2, v1 // (op 536) 
v_sub_u32 v11, v1, v10 // (op 536) 

v_lshlrev_b32 v10, 4, v11 // (op 536) 

v_lshl_add_u32 v11, s17, 5, v10 // (op 536) 

v_and_b32 v10, 4294967294, v1 // (op 536) 
v_mul_lo_u32 v15, v10, s8 // (op 536) 
v_add_u32 v15, v15, v11 // (op 536) 


s_mul_i32 s1, s8, s7 // (op 538) most significant: low of low * high
s_mul_i32 s14, s9, s6 // (op 538) most significant: low of high * low
s_mul_i32 s0, s8, s6 // (op 538) least significant
s_mul_hi_u32 s15, s8, s6 // (op 538) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 538) most significant: sum
s_add_u32 s1, s1, s15 // (op 538) most significant: sum


s_mov_b32 s44, s10 // (op 538) reg expression
s_mov_b32 s45, s11 // (op 538) reg expression
s_mov_b32 s46, s0 // (op 538) reg expression
s_mov_b32 s47, 131072 // (op 538) call()

v_lshrrev_b32 v10, 7, v1 // (op 581) 
v_and_b32 v11, 15, v1 // (op 581) 
v_lshl_add_u32 v16, v10, 4, v11 // (op 581) 


v_lshl_add_u32 v10, s16, 4, v16 // (op 581) 

v_and_b32 v11, 63, v1 // (op 581) 
v_lshrrev_b32 v16, 4, v11 // (op 581) 

v_mul_lo_u32 v11, v16, s12 // (op 581) 

v_add_u32 v16, v10, v11 // (op 581) 



s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 583) most significant: low of low * high
s_mul_i32 s14, s13, s22 // (op 583) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 583) least significant
s_mul_hi_u32 s15, s12, s22 // (op 583) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 583) most significant: sum
s_add_u32 s1, s1, s15 // (op 583) most significant: sum


s_mov_b32 s48, s20 // (op 583) reg expression
s_mov_b32 s49, s21 // (op 583) reg expression
s_mov_b32 s50, s0 // (op 583) reg expression
s_mov_b32 s51, 131072 // (op 583) call()

v_lshrrev_b32 v10, 7, v1 // (op 600) 
v_and_b32 v11, 15, v1 // (op 600) 
v_lshl_add_u32 v17, v10, 4, v11 // (op 600) 


v_lshl_add_u32 v10, s16, 4, v17 // (op 600) 

v_and_b32 v11, 63, v1 // (op 600) 
v_lshrrev_b32 v17, 4, v11 // (op 600) 

v_mul_lo_u32 v11, v17, s12 // (op 600) 

v_add_u32 v17, v10, v11 // (op 600) 


s_mul_i32 s1, s12, s23 // (op 602) most significant: low of low * high
s_mul_i32 s14, s13, s22 // (op 602) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 602) least significant
s_mul_hi_u32 s15, s12, s22 // (op 602) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 602) most significant: sum
s_add_u32 s1, s1, s15 // (op 602) most significant: sum


s_mov_b32 s48, s20 // (op 602) reg expression
s_mov_b32 s49, s21 // (op 602) reg expression
s_mov_b32 s50, s0 // (op 602) reg expression
s_mov_b32 s51, 131072 // (op 602) call()

v_and_b32 v10, -32, v1 // (op 645) 
v_sub_u32 v11, v1, v10 // (op 645) 

v_lshl_add_u32 v10, s17, 5, v11 // (op 645) 

v_lshrrev_b32 v11, 5, v1 // (op 645) 
v_mul_lo_u32 v18, v11, s24 // (op 645) 
v_add_u32 v18, v18, v10 // (op 645) 


s_mul_i32 s1, s24, s23 // (op 647) most significant: low of low * high
s_mul_i32 s14, s25, s22 // (op 647) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 647) least significant
s_mul_hi_u32 s15, s24, s22 // (op 647) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 647) most significant: sum
s_add_u32 s1, s1, s15 // (op 647) most significant: sum


s_mov_b32 s52, s26 // (op 647) reg expression
s_mov_b32 s53, s27 // (op 647) reg expression
s_mov_b32 s54, s0 // (op 647) reg expression
s_mov_b32 s55, 131072 // (op 647) call()

v_and_b32 v10, -32, v1 // (op 664) 
v_sub_u32 v11, v1, v10 // (op 664) 

v_lshl_add_u32 v10, s17, 5, v11 // (op 664) 

v_lshrrev_b32 v11, 5, v1 // (op 664) 
v_mul_lo_u32 v19, v11, s24 // (op 664) 
v_add_u32 v19, v19, v10 // (op 664) 


s_mul_i32 s1, s24, s23 // (op 666) most significant: low of low * high
s_mul_i32 s14, s25, s22 // (op 666) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 666) least significant
s_mul_hi_u32 s15, s24, s22 // (op 666) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 666) most significant: sum
s_add_u32 s1, s1, s15 // (op 666) most significant: sum


s_mov_b32 s52, s26 // (op 666) reg expression
s_mov_b32 s53, s27 // (op 666) reg expression
s_mov_b32 s54, s0 // (op 666) reg expression
s_mov_b32 s55, 131072 // (op 666) call()


v_mov_b64 v[10:11], s[12:13] // (op 337) convert

v_mov_b32 v20, v16 // (op 337) reg expression
v_mov_b32 v21, v20 // (op 337) 
buffer_load_ubyte v22, v21, s[48:51], 0 offen offset: 0 // (op 337) Load value




v_mov_b64 v[10:11], s[8:9] // (op 341) convert

v_mov_b32 v20, v14 // (op 341) reg expression
buffer_load_dwordx4 v[24:27], v20, s[44:47], 0 offen // (op 341) Load value
v_add_u32 v20, v20, v10 // (op 341) 
buffer_load_dwordx4 v[28:31], v20, s[44:47], 0 offen // (op 341) Load value




v_mov_b32 v23, v18 // (op 344) reg expression

buffer_load_ubyte v10, v23, s[52:55], 0 offen offset: 0 // (op 344) Load value



v_mov_b32 v11, v12 // (op 347) reg expression

buffer_load_dwordx2 v[20:21], v11, s[40:43], 0 offen // (op 347) Load value


v_mov_b64 v[32:33], s[12:13] // (op 351) convert

s_lshl_b32 s0, s12, 2 // (op 351) 
v_add_u32 v11, v16, s0 // (op 351) 

v_mov_b32 v23, v11 // (op 351) 
buffer_load_ubyte v34, v23, s[48:51], 0 offen offset: 0 // (op 351) Load value





v_mov_b64 v[32:33], s[8:9] // (op 355) convert

s_lshl_b32 s0, s8, 7 // (op 355) 
v_add_u32 v11, v14, s0 // (op 355) 

buffer_load_dwordx4 v[36:39], v11, s[44:47], 0 offen // (op 355) Load value
v_add_u32 v11, v11, v32 // (op 355) 
buffer_load_dwordx4 v[40:43], v11, s[44:47], 0 offen // (op 355) Load value





s_lshl_b32 s0, s24, 2 // (op 358) 
v_add_u32 v11, v18, s0 // (op 358) 


buffer_load_ubyte v14, v11, s[52:55], 0 offen offset: 0 // (op 358) Load value




s_lshl_b64 s[0:1], s[2:3], 7 // (op 361) 
s_lshl_b64 s[14:15], s[0:1], 2 // (op 361) 

s_lshr_b64 s[0:1], s[14:15], 3 // (op 361) 

v_add_u32 v11, v12, s0 // (op 361) 


buffer_load_dwordx2 v[32:33], v11, s[40:43], 0 offen // (op 361) Load value


v_and_b32 v16, -2, v1 // (op 784) 
v_sub_u32 v18, v1, v16 // (op 784) 

v_lshlrev_b32 v16, 4, v18 // (op 784) 

v_lshrrev_b32 v18, 1, v1 // (op 784) 
v_lshl_add_u32 v23, v18, 6, v16 // (op 784) 





s_waitcnt vmcnt(8)
ds_write_b128 v23, v[24:27] // (op 363) Store local data 
s_waitcnt vmcnt(7)
ds_write_b128 v23, v[28:31] offset:32 // (op 363) Store local data 

v_and_b32 v16, -32, v1 // (op 837) 
v_sub_u32 v18, v1, v16 // (op 837) 

v_and_b32 v16, 4294967264, v1 // (op 837) 
v_add_u32 v23, v16, v18 // (op 837) 





s_waitcnt vmcnt(6)
ds_write_b8 v23, v10 offset:4096 // (op 364) Store local data 

v_lshlrev_b32 v16, 6, v1 // (op 731) 
v_lshrrev_b32 v18, 3, v16 // (op 731) 




s_waitcnt vmcnt(5)
ds_write_b64 v18, v[20:21] offset:4224 // (op 365) Store local data 

v_lshlrev_b32 v16, 6, v1 // (op 702) 
v_lshrrev_b32 v18, 3, v16 // (op 702) 

v_and_b32 v16, -2, v1 // (op 755) 
v_sub_u32 v23, v1, v16 // (op 755) 

v_lshlrev_b32 v16, 4, v23 // (op 755) 

v_lshrrev_b32 v23, 1, v1 // (op 755) 
v_lshl_add_u32 v35, v23, 6, v16 // (op 755) 


v_and_b32 v16, -32, v1 // (op 808) 
v_sub_u32 v23, v1, v16 // (op 808) 

v_and_b32 v16, 4294967264, v1 // (op 808) 
v_add_u32 v11, v16, v23 // (op 808) 


v_lshlrev_b32 v12, 6, v1 // (op 895) 
v_lshrrev_b32 v16, 3, v12 // (op 895) 

v_and_b32 v12, -2, v1 // (op 919) 
v_sub_u32 v23, v1, v12 // (op 919) 

v_lshlrev_b32 v12, 4, v23 // (op 919) 

v_lshrrev_b32 v23, 1, v1 // (op 919) 
v_lshl_add_u32 v44, v23, 6, v12 // (op 919) 


v_and_b32 v12, -32, v1 // (op 943) 
v_sub_u32 v23, v1, v12 // (op 943) 

v_and_b32 v12, 4294967264, v1 // (op 943) 
v_add_u32 v45, v12, v23 // (op 943) 


s_mov_b32 s0, 0 // (op 55) call()
s_ashr_i64 s[14:15], s[6:7], 7 // (op 54) 
s_mov_b32 s1, s14 // (op 54) convert

s_and_b32 s14, s1, 4294967294 // (op 54) 

s_cmp_lt_u32 s0, s14 // (op 54) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoop_54 // (op 54) Condition: Top (jump to RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoop_54 if false)
RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoop_54:
s_add_u32 s0, s0, 2 // (op 56) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 988) 



ds_read_b64_tr_b4 v[46:47], v3 offset:4224 // (op 65) Transpose load local data 
ds_read_b64_tr_b4 v[48:49], v3 offset:4352 // (op 65) Transpose load local data 
s_barrier  // (op 1001) 



ds_read_b64_tr_b8 v[50:51], v4 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v4 offset:256 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v4 offset:2048 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v4 offset:2304 // (op 70) Transpose load local data 
s_barrier  // (op 1015) 


ds_read_u8 v12, v0 offset:4096 // (op 79) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[46:49], v[50:57], a[0:3], v22, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 82) 




v_mov_b64 v[46:47], s[12:13] // (op 75) convert

v_mov_b32 v12, v17 // (op 75) 
s_lshl_b32 s1, s12, 3 // (op 75) 
v_add_u32 v23, v12, s1 // (op 75) 

v_mov_b32 v48, v23 // (op 75) 
buffer_load_ubyte v22, v48, s[48:51], 0 offen offset: 0 // (op 75) Load value




v_mov_b64 v[46:47], s[8:9] // (op 10) convert

v_mov_b32 v23, v15 // (op 10) 
s_lshl_b32 s1, s8, 8 // (op 10) 
v_add_u32 v48, v23, s1 // (op 10) 

buffer_load_dwordx4 v[24:27], v48, s[44:47], 0 offen // (op 10) Load value
v_add_u32 v48, v48, v46 // (op 10) 
buffer_load_dwordx4 v[28:31], v48, s[44:47], 0 offen // (op 10) Load value




v_mov_b32 v46, v19 // (op 21) 
s_lshl_b32 s1, s24, 3 // (op 21) 
v_add_u32 v47, v46, s1 // (op 21) 


buffer_load_ubyte v10, v47, s[52:55], 0 offen offset: 0 // (op 21) Load value



v_mov_b32 v47, v13 // (op 4) 
s_lshl_b64 s[14:15], s[2:3], 7 // (op 4) 
s_lshl_b64 s[18:19], s[14:15], 2 // (op 4) 

s_lshr_b64 s[14:15], s[18:19], 3 // (op 4) 

s_lshl_b32 s1, s14, 1 // (op 4) 

v_add_u32 v48, v47, s1 // (op 4) 


buffer_load_dwordx2 v[20:21], v48, s[40:43], 0 offen // (op 4) Load value

s_barrier  // (op 1038) 



s_waitcnt vmcnt(8)
ds_write_b128 v44, v[36:39] offset:5248 // (op 235) Store local data 
s_waitcnt vmcnt(7)
ds_write_b128 v44, v[40:43] offset:5280 // (op 235) Store local data 
s_barrier  // (op 1044) 



s_waitcnt vmcnt(6)
ds_write_b8 v45, v14 offset:9344 // (op 236) Store local data 
s_barrier  // (op 1032) 



s_waitcnt vmcnt(5)
ds_write_b64 v16, v[32:33] offset:9472 // (op 225) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1029) 



ds_read_b64_tr_b4 v[48:49], v6 offset:9472 // (op 227) Transpose load local data 
ds_read_b64_tr_b4 v[50:51], v6 offset:9600 // (op 227) Transpose load local data 
s_barrier  // (op 1035) 
s_barrier  // (op 1095) 



ds_read_b64_tr_b8 v[52:53], v7 offset:5248 // (op 229) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v7 offset:5504 // (op 229) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v7 offset:7296 // (op 229) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v7 offset:7552 // (op 229) Transpose load local data 
s_barrier  // (op 1041) 
s_barrier  // (op 1101) 


ds_read_u8 v60, v5 offset:9344 // (op 233) Load local data 
s_barrier  // (op 1109) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[48:51], v[52:59], a[0:3], v34, v60 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 234) 




v_mov_b64 v[48:49], s[12:13] // (op 231) convert

s_lshl_b32 s1, s12, 2 // (op 231) 
s_mul_i32 s14, 3, s1 // (op 231) 

v_add_u32 v50, v12, s14 // (op 231) 

v_mov_b32 v51, v50 // (op 231) 
buffer_load_ubyte v34, v51, s[48:51], 0 offen offset: 0 // (op 231) Load value





s_lshl_b32 s1, s12, 3 // (op 592) 
v_add_u32 v17, v17, s1 // (op 592) 

v_mov_b64 v[48:49], s[8:9] // (op 237) convert

s_lshl_b32 s1, s8, 7 // (op 237) 
s_mul_i32 s14, 3, s1 // (op 237) 

v_add_u32 v12, v23, s14 // (op 237) 

buffer_load_dwordx4 v[36:39], v12, s[44:47], 0 offen // (op 237) Load value
v_add_u32 v12, v12, v48 // (op 237) 
buffer_load_dwordx4 v[40:43], v12, s[44:47], 0 offen // (op 237) Load value





s_lshl_b32 s1, s24, 2 // (op 238) 
s_mul_i32 s14, 3, s1 // (op 238) 

s_lshl_b32 s1, s8, 8 // (op 528) 
v_add_u32 v15, v15, s1 // (op 528) 

v_add_u32 v12, v46, s14 // (op 238) 


buffer_load_ubyte v14, v12, s[52:55], 0 offen offset: 0 // (op 238) Load value




s_lshl_b32 s1, s24, 3 // (op 656) 
v_add_u32 v19, v19, s1 // (op 656) 

s_lshl_b64 s[14:15], s[2:3], 7 // (op 224) 
s_lshl_b64 s[18:19], s[14:15], 2 // (op 224) 

s_lshr_b64 s[14:15], s[18:19], 3 // (op 224) 

s_mul_i32 s1, 3, s14 // (op 224) 

v_add_u32 v12, v47, s1 // (op 224) 


buffer_load_dwordx2 v[32:33], v12, s[40:43], 0 offen // (op 224) Load value


s_barrier  // (op 1004) 
s_lshl_b64 s[14:15], s[2:3], 7 // (op 464) 



s_lshl_b64 s[18:19], s[14:15], 2 // (op 464) 

s_lshr_b64 s[14:15], s[18:19], 3 // (op 464) 

s_lshl_b32 s1, s14, 1 // (op 464) 

v_add_u32 v13, v13, s1 // (op 464) 

s_waitcnt vmcnt(8)
ds_write_b128 v35, v[24:27] // (op 13) Store local data 
s_waitcnt vmcnt(7)
ds_write_b128 v35, v[28:31] offset:32 // (op 13) Store local data 
s_barrier  // (op 1018) 



s_waitcnt vmcnt(6)
ds_write_b8 v11, v10 offset:4096 // (op 26) Store local data 
s_barrier  // (op 991) 



s_waitcnt vmcnt(5)
ds_write_b64 v18, v[20:21] offset:4224 // (op 6) Store local data 
s_ashr_i64 s[14:15], s[6:7], 7 // (op 54) 
s_mov_b32 s1, s14 // (op 54) convert

s_and_b32 s14, s1, 4294967294 // (op 54) 

s_cmp_lt_u32 s0, s14 // (op 54) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoop_54 // (op 54) Condition: Bottom (jump to RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoop_54 if true)
RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoop_54:















s_ashr_i64 s[0:1], s[6:7], 7 // (op 427) 
s_lshr_b32 s14, s0, 1 // (op 427) 

v_lshl_add_u32 v35, s14, 8, v1 // (op 427) 

s_lshl_b32 s0, s16, 4 // (op 427) 
v_mul_lo_u32 v11, v35, s3 // (op 427) most significant: low of low * high
v_mul_hi_u32 v12, v35, s2 // (op 427) most significant: high of low * low
v_mul_lo_u32 v10, v35, s2 // (op 427) least significant: low of low * low
v_add_u32 v11, v11, v12 // (op 427) most significant: sum


s_mov_b32 s15, 0 // (op 427) convert
s_mov_b32 s14, s0 // (op 427) convert

v_mov_b32 v35, s14 // (op 427) 
v_mov_b32 v12, s15 // (op 427) 
v_add_co_u32 v30, s[0:1], v35, v10 // (op 427) least significant half
v_addc_co_u32 v31, s[0:1], v12, v11, s[0:1] // (op 427) most significant half



v_lshlrev_b64 v[30:31], 2, v[30:31] // (op 427) 


v_lshrrev_b64 v[10:11], 3, v[30:31] // (op 427) 

v_mov_b32 v35, v10 // (op 427) convert

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

v_and_b32 v10, 4294967294, v1 // (op 494) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 494) 
v_and_b32 v11, -2, v1 // (op 494) 
s_lshr_b32 s14, s0, 1 // (op 494) 

v_sub_u32 v12, v1, v11 // (op 494) 

v_lshl_add_u32 v11, s14, 8, v10 // (op 494) 


v_lshlrev_b32 v10, 4, v12 // (op 494) 

v_lshl_add_u32 v12, s17, 5, v10 // (op 494) 

v_mul_lo_u32 v10, v11, s8 // (op 494) 
v_add_u32 v10, v10, v12 // (op 494) 


s_mul_i32 s1, s8, s7 // (op 496) most significant: low of low * high
s_mul_i32 s14, s9, s6 // (op 496) most significant: low of high * low
s_mul_i32 s0, s8, s6 // (op 496) least significant
s_mul_hi_u32 s15, s8, s6 // (op 496) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 496) most significant: sum
s_add_u32 s1, s1, s15 // (op 496) most significant: sum


s_mov_b32 s44, s10 // (op 496) reg expression
s_mov_b32 s45, s11 // (op 496) reg expression
s_mov_b32 s46, s0 // (op 496) reg expression
s_mov_b32 s47, 131072 // (op 496) call()

v_lshrrev_b32 v11, 7, v1 // (op 558) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 558) 
v_and_b32 v12, 15, v1 // (op 558) 
s_lshr_b32 s14, s0, 1 // (op 558) 

v_lshl_add_u32 v13, v11, 4, v12 // (op 558) 


v_lshl_add_u32 v11, s16, 4, v13 // (op 558) 

v_and_b32 v12, 63, v1 // (op 558) 
v_lshrrev_b32 v13, 4, v12 // (op 558) 

v_lshl_add_u32 v12, s14, 3, v13 // (op 558) 


v_mul_lo_u32 v13, v12, s12 // (op 558) 

v_add_u32 v12, v11, v13 // (op 558) 


s_mul_i32 s1, s12, s23 // (op 560) most significant: low of low * high
s_mul_i32 s14, s13, s22 // (op 560) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 560) least significant
s_mul_hi_u32 s15, s12, s22 // (op 560) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 560) most significant: sum
s_add_u32 s1, s1, s15 // (op 560) most significant: sum


s_mov_b32 s48, s20 // (op 560) reg expression
s_mov_b32 s49, s21 // (op 560) reg expression
s_mov_b32 s50, s0 // (op 560) reg expression
s_mov_b32 s51, 131072 // (op 560) call()

v_lshrrev_b32 v11, 5, v1 // (op 622) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 622) 
v_and_b32 v13, -32, v1 // (op 622) 
s_lshr_b32 s14, s0, 1 // (op 622) 

v_sub_u32 v15, v1, v13 // (op 622) 

v_lshl_add_u32 v13, s14, 3, v11 // (op 622) 


v_lshl_add_u32 v11, s17, 5, v15 // (op 622) 

v_mul_lo_u32 v15, v13, s24 // (op 622) 
v_add_u32 v15, v15, v11 // (op 622) 


s_mul_i32 s1, s24, s23 // (op 624) most significant: low of low * high
s_mul_i32 s14, s25, s22 // (op 624) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 624) least significant
s_mul_hi_u32 s15, s24, s22 // (op 624) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 624) most significant: sum
s_add_u32 s1, s1, s15 // (op 624) most significant: sum


s_mov_b32 s52, s26 // (op 624) reg expression
s_mov_b32 s53, s27 // (op 624) reg expression
s_mov_b32 s54, s0 // (op 624) reg expression
s_mov_b32 s55, 131072 // (op 624) call()

v_lshlrev_b32 v11, 6, v1 // (op 715) 
v_lshrrev_b32 v13, 3, v11 // (op 715) 

v_and_b32 v11, -2, v1 // (op 768) 
v_sub_u32 v16, v1, v11 // (op 768) 

v_lshlrev_b32 v11, 4, v16 // (op 768) 

v_lshrrev_b32 v16, 1, v1 // (op 768) 
v_lshl_add_u32 v17, v16, 6, v11 // (op 768) 


v_and_b32 v11, -32, v1 // (op 821) 
v_sub_u32 v16, v1, v11 // (op 821) 

v_and_b32 v11, 4294967264, v1 // (op 821) 
v_add_u32 v18, v11, v16 // (op 821) 


s_ashr_i64 s[0:1], s[6:7], 7 // (op 162) 
s_and_b32 s14, s0, 4294967294 // (op 162) 

s_ashr_i64 s[0:1], s[6:7], 7 // (op 161) 
s_mov_b32 s15, s0 // (op 161) convert

s_cmp_lt_u32 s14, s15 // (op 161) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoopTail_161 // (op 161) Condition: Top (jump to RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoopTail_161 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoopTail_161:


s_add_u32 s14, s14, 1 // (op 163) 
v_mov_b32 v11, v35 // (op 166) 
v_mov_b32 v14, v11 // (op 166) reg expression

buffer_load_dwordx2 v[20:21], v14, s[40:43], 0 offen // (op 166) Load value


s_lshl_b64 s[0:1], s[2:3], 7 // (op 417) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 417) 

s_lshr_b64 s[0:1], s[18:19], 3 // (op 417) 

v_add_u32 v35, v35, s0 // (op 417) 


s_barrier  // (op 994) 
v_mov_b64 v[22:23], s[8:9] // (op 179) convert




v_mov_b32 v11, v10 // (op 179) 
v_mov_b32 v14, v11 // (op 179) reg expression
buffer_load_dwordx4 v[24:27], v14, s[44:47], 0 offen // (op 179) Load value
v_add_u32 v14, v14, v22 // (op 179) 
buffer_load_dwordx4 v[28:31], v14, s[44:47], 0 offen // (op 179) Load value



s_lshl_b32 s15, s8, 7 // (op 486) 
v_add_u32 v10, v10, s15 // (op 486) 

s_waitcnt vmcnt(2)
ds_write_b64 v13, v[20:21] offset:4224 // (op 167) Store local data 



v_mov_b32 v11, v15 // (op 180) 
s_barrier  // (op 1007) 
v_mov_b32 v14, v11 // (op 180) reg expression




buffer_load_ubyte v16, v14, s[52:55], 0 offen offset: 0 // (op 180) Load value


s_lshl_b32 s15, s24, 2 // (op 614) 
v_add_u32 v15, v15, s15 // (op 614) 

s_waitcnt vmcnt(2)
ds_write_b128 v17, v[24:27] // (op 177) Store local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 997) 
s_waitcnt vmcnt(1)
ds_write_b128 v17, v[28:31] offset:32 // (op 177) Store local data 




ds_read_b64_tr_b4 v[20:21], v3 offset:4224 // (op 169) Transpose load local data 
s_barrier  // (op 1021) 
ds_read_b64_tr_b4 v[22:23], v3 offset:4352 // (op 169) Transpose load local data 



s_waitcnt lgkmcnt(2)
s_barrier  // (op 1011) 
s_waitcnt vmcnt(0)
ds_write_b8 v18, v16 offset:4096 // (op 178) Store local data 




ds_read_b64_tr_b8 v[24:25], v4 // (op 171) Transpose load local data 
ds_read_b64_tr_b8 v[26:27], v4 offset:256 // (op 171) Transpose load local data 
ds_read_b64_tr_b8 v[28:29], v4 offset:2048 // (op 171) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v4 offset:2304 // (op 171) Transpose load local data 

v_mov_b64 v[32:33], s[12:13] // (op 173) convert

v_mov_b32 v11, v12 // (op 173) 
v_mov_b32 v14, v11 // (op 173) reg expression
v_mov_b32 v16, v14 // (op 173) 
buffer_load_ubyte v19, v16, s[48:51], 0 offen offset: 0 // (op 173) Load value




s_lshl_b32 s15, s12, 2 // (op 550) 
v_add_u32 v12, v12, s15 // (op 550) 

s_waitcnt lgkmcnt(4)
s_barrier  // (op 1025) 


ds_read_u8 v11, v0 offset:4096 // (op 175) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:23], v[24:31], a[0:3], v19, v11 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 176) 




s_ashr_i64 s[0:1], s[6:7], 7 // (op 161) 
s_mov_b32 s15, s0 // (op 161) convert

s_cmp_lt_u32 s14, s15 // (op 161) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoopTail_161 // (op 161) Condition: Bottom (jump to RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoopTail_161 if true)
RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoopTail_161:






















s_barrier  // (op 1089) 






v_cmp_eq_f32 vcc, 0.00000, v8 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalFalse_0_beta_123
s_nop 0
v_accvgpr_read v4, a0 // (op 46) Wait state hazard: XDL Write Hazard
v_accvgpr_read v5, a1 // (op 46) 
v_accvgpr_read v6, a2 // (op 46) 
v_accvgpr_read v7, a3 // (op 46) 
v_mul_f32 v10, v9, v4 // (op 46) 
v_mul_f32 v11, v9, v5 // (op 46) 
v_mul_f32 v12, v9, v6 // (op 46) 
v_mul_f32 v13, v9, v7 // (op 46) 

v_lshrrev_b32 v0, 7, v1 // (op 863) 
v_and_b32 v3, 63, v1 // (op 863) 
v_lshrrev_b32 v4, 4, v3 // (op 863) 
v_lshlrev_b32 v5, 2, v4 // (op 863) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 863) 


v_lshl_add_u32 v0, s16, 4, v4 // (op 863) 

v_lshrrev_b32 v4, 2, v3 // (op 863) 

v_and_b32 v3, 3, v4 // (op 863) 

v_and_b32 v4, 3, v1 // (op 863) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 863) 


v_lshrrev_b32 v3, 6, v1 // (op 863) 
v_and_b32 v4, 1, v3 // (op 863) 

v_lshl_add_u32 v3, v4, 4, v5 // (op 863) 


v_lshl_add_u32 v4, s17, 5, v3 // (op 863) 

v_mul_lo_u32 v3, v4, s28 // (op 863) 

v_add_lshl_u32 v4, v0, v3, 2 // (op 863) 


s_mul_i32 s1, s28, s33 // (op 865) most significant: low of low * high
s_mul_i32 s2, s29, s32 // (op 865) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 865) least significant
s_mul_hi_u32 s3, s28, s32 // (op 865) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 865) most significant: sum
s_add_u32 s1, s1, s3 // (op 865) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 2 // (op 865) 

s_mov_b32 s4, s30 // (op 865) reg expression
s_mov_b32 s5, s31 // (op 865) reg expression
s_mov_b32 s6, s2 // (op 865) reg expression
s_mov_b32 s7, 131072 // (op 865) call()




buffer_store_dwordx4 v[10:13], v4, s[4:7], 0 offen // (op 52) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalBottom_0_beta_123
RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalFalse_0_beta_123:
v_lshrrev_b32 v0, 7, v1 // (op 685) 
v_and_b32 v3, 63, v1 // (op 685) 
v_lshrrev_b32 v4, 4, v3 // (op 685) 
v_lshlrev_b32 v5, 2, v4 // (op 685) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 685) 


v_lshl_add_u32 v0, s16, 4, v4 // (op 685) 

v_lshrrev_b32 v4, 2, v3 // (op 685) 

v_and_b32 v3, 3, v4 // (op 685) 

v_and_b32 v4, 3, v1 // (op 685) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 685) 


v_lshrrev_b32 v3, 6, v1 // (op 685) 
v_and_b32 v4, 1, v3 // (op 685) 

v_lshl_add_u32 v3, v4, 4, v5 // (op 685) 


v_lshl_add_u32 v4, s17, 5, v3 // (op 685) 

v_mul_lo_u32 v3, v4, s34 // (op 685) 

v_add_lshl_u32 v4, v0, v3, 2 // (op 685) 


s_mul_i32 s1, s34, s33 // (op 687) most significant: low of low * high
s_mul_i32 s2, s35, s32 // (op 687) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 687) least significant
s_mul_hi_u32 s3, s34, s32 // (op 687) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 687) most significant: sum
s_add_u32 s1, s1, s3 // (op 687) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 2 // (op 687) 

s_mov_b32 s8, s36 // (op 687) reg expression
s_mov_b32 s9, s37 // (op 687) reg expression
s_mov_b32 s10, s2 // (op 687) reg expression
s_mov_b32 s11, 131072 // (op 687) call()





buffer_load_dwordx4 v[10:13], v4, s[8:11], 0 offen // (op 135) Load value



v_accvgpr_read v4, a0 // (op 132) 
v_accvgpr_read v5, a1 // (op 132) 
v_accvgpr_read v6, a2 // (op 132) 
v_accvgpr_read v7, a3 // (op 132) 
v_mul_f32 v14, v9, v4 // (op 132) 
v_mul_f32 v15, v9, v5 // (op 132) 
v_mul_f32 v16, v9, v6 // (op 132) 
v_mul_f32 v17, v9, v7 // (op 132) 

s_waitcnt vmcnt(0)
v_mul_f32 v4, v8, v10 // (op 136) 
v_mul_f32 v5, v8, v11 // (op 136) 
v_mul_f32 v6, v8, v12 // (op 136) 
v_mul_f32 v7, v8, v13 // (op 136) 

v_add_f32 v10, v4, v14 // (op 133) 
v_add_f32 v11, v5, v15 // (op 133) 
v_add_f32 v12, v6, v16 // (op 133) 
v_add_f32 v13, v7, v17 // (op 133) 


v_lshrrev_b32 v0, 7, v1 // (op 880) 
v_and_b32 v3, 63, v1 // (op 880) 
v_lshrrev_b32 v4, 4, v3 // (op 880) 
v_lshlrev_b32 v5, 2, v4 // (op 880) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 880) 


v_lshl_add_u32 v0, s16, 4, v4 // (op 880) 

v_lshrrev_b32 v4, 2, v3 // (op 880) 

v_and_b32 v3, 3, v4 // (op 880) 

v_and_b32 v4, 3, v1 // (op 880) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 880) 


v_lshrrev_b32 v3, 6, v1 // (op 880) 
v_and_b32 v4, 1, v3 // (op 880) 

v_lshl_add_u32 v3, v4, 4, v5 // (op 880) 


v_lshl_add_u32 v4, s17, 5, v3 // (op 880) 

v_mul_lo_u32 v3, v4, s28 // (op 880) 

v_add_lshl_u32 v4, v0, v3, 2 // (op 880) 


s_mul_i32 s1, s28, s33 // (op 882) most significant: low of low * high
s_mul_i32 s2, s29, s32 // (op 882) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 882) least significant
s_mul_hi_u32 s3, s28, s32 // (op 882) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 882) most significant: sum
s_add_u32 s1, s1, s3 // (op 882) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 2 // (op 882) 

s_mov_b32 s4, s30 // (op 882) reg expression
s_mov_b32 s5, s31 // (op 882) reg expression
s_mov_b32 s6, s2 // (op 882) reg expression
s_mov_b32 s7, 131072 // (op 882) call()




buffer_store_dwordx4 v[10:13], v4, s[4:7], 0 offen // (op 134) Store value






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalBottom_0_beta_123:



s_endpgm  // End of RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2
.LRR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_end:
.size RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2, .LRR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_end-RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2
  .amdhsa_next_free_vgpr 68
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 10496 // lds bytes
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
    DataType: F4
    DataTypeA: F4
    DataTypeB: B8
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
  - .name: RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2
    .symbol: RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 10496
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 56
    .vgpr_count: 61
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

