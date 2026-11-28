// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2
.p2align 8
.type RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2,@function
RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_exec_begin:


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


v_lshrrev_b32 v3, 6, v1 // (op 743) 
v_lshlrev_b32 v4, 4, v3 // (op 743) 

v_lshl_add_u32 v3, v0, 5, v4 // (op 743) 


v_lshlrev_b32 v0, 2, v3 // (op 743) 

v_lshrrev_b32 v3, 3, v0 // (op 743) 

v_and_b32 v0, 15, v1 // (op 796) 
v_and_b32 v4, 48, v1 // (op 796) 
v_lshl_add_u32 v5, v0, 7, v4 // (op 796) 


v_lshrrev_b32 v0, 6, v1 // (op 849) 
v_and_b32 v4, 15, v1 // (op 849) 
v_lshl_add_u32 v6, v0, 4, v4 // (op 849) 


v_and_b32 v0, 63, v1 // (op 849) 
v_lshrrev_b32 v4, 4, v0 // (op 849) 

v_lshl_add_u32 v0, v4, 5, v6 // (op 849) 


v_and_b32 v4, 63, v1 // (op 907) 
v_and_b32 v6, 48, v1 // (op 907) 
v_sub_u32 v7, v4, v6 // (op 907) 

v_lshrrev_b32 v6, 4, v4 // (op 907) 

v_lshl_add_u32 v4, v6, 5, v7 // (op 907) 


v_lshrrev_b32 v6, 6, v1 // (op 907) 
v_lshlrev_b32 v7, 4, v6 // (op 907) 

v_lshl_add_u32 v6, v4, 5, v7 // (op 907) 


v_lshlrev_b32 v4, 2, v6 // (op 907) 

v_lshrrev_b32 v6, 3, v4 // (op 907) 

v_and_b32 v4, 15, v1 // (op 931) 
v_and_b32 v7, 48, v1 // (op 931) 
v_lshl_add_u32 v8, v4, 7, v7 // (op 931) 


v_lshrrev_b32 v4, 6, v1 // (op 955) 
v_and_b32 v7, 15, v1 // (op 955) 
v_lshl_add_u32 v9, v4, 4, v7 // (op 955) 


v_and_b32 v4, 63, v1 // (op 955) 
v_lshrrev_b32 v7, 4, v4 // (op 955) 

v_lshl_add_u32 v4, v7, 5, v9 // (op 955) 


v_mov_b32 v7, s14 // (op 2) Move value
v_mov_b32 v9, s15 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 61) call()
v_accvgpr_write a1, 0.00000 // (op 61) call()
v_accvgpr_write a2, 0.00000 // (op 61) call()
v_accvgpr_write a3, 0.00000 // (op 61) call()
s_lshl_b32 s0, s16, 5 // (op 453) 
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

s_lshl_b32 s0, s16, 5 // (op 472) 
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

v_and_b32 v10, -8, v1 // (op 517) 
v_sub_u32 v11, v1, v10 // (op 517) 

v_lshrrev_b32 v10, 3, v1 // (op 517) 
v_lshl_add_u32 v14, s17, 4, v10 // (op 517) 

v_mul_lo_u32 v10, v14, s8 // (op 517) 

v_lshl_add_u32 v14, v11, 4, v10 // (op 517) 


s_mul_i32 s1, s8, s13 // (op 519) most significant: low of low * high
s_mul_i32 s14, s9, s12 // (op 519) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 519) least significant
s_mul_hi_u32 s15, s8, s12 // (op 519) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 519) most significant: sum
s_add_u32 s1, s1, s15 // (op 519) most significant: sum


s_mov_b32 s44, s10 // (op 519) reg expression
s_mov_b32 s45, s11 // (op 519) reg expression
s_mov_b32 s46, s0 // (op 519) reg expression
s_mov_b32 s47, 131072 // (op 519) call()

v_and_b32 v10, -8, v1 // (op 536) 
v_sub_u32 v11, v1, v10 // (op 536) 

v_lshrrev_b32 v10, 3, v1 // (op 536) 
v_lshl_add_u32 v15, s17, 4, v10 // (op 536) 

v_mul_lo_u32 v10, v15, s8 // (op 536) 

v_lshl_add_u32 v15, v11, 4, v10 // (op 536) 


s_mul_i32 s1, s8, s13 // (op 538) most significant: low of low * high
s_mul_i32 s14, s9, s12 // (op 538) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 538) least significant
s_mul_hi_u32 s15, s8, s12 // (op 538) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 538) most significant: sum
s_add_u32 s1, s1, s15 // (op 538) most significant: sum


s_mov_b32 s44, s10 // (op 538) reg expression
s_mov_b32 s45, s11 // (op 538) reg expression
s_mov_b32 s46, s0 // (op 538) reg expression
s_mov_b32 s47, 131072 // (op 538) call()

v_and_b32 v10, -32, v1 // (op 581) 
v_sub_u32 v11, v1, v10 // (op 581) 

v_lshl_add_u32 v10, s16, 5, v11 // (op 581) 

v_lshrrev_b32 v11, 5, v1 // (op 581) 

s_waitcnt lgkmcnt(0)
v_mul_lo_u32 v16, v11, s20 // (op 581) 

v_add_u32 v11, v10, v16 // (op 581) 


s_mul_i32 s1, s20, s25 // (op 583) most significant: low of low * high
s_mul_i32 s14, s21, s24 // (op 583) most significant: low of high * low
s_mul_i32 s0, s20, s24 // (op 583) least significant
s_mul_hi_u32 s15, s20, s24 // (op 583) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 583) most significant: sum
s_add_u32 s1, s1, s15 // (op 583) most significant: sum


s_mov_b32 s48, s22 // (op 583) reg expression
s_mov_b32 s49, s23 // (op 583) reg expression
s_mov_b32 s50, s0 // (op 583) reg expression
s_mov_b32 s51, 131072 // (op 583) call()

v_and_b32 v10, -32, v1 // (op 600) 
v_sub_u32 v16, v1, v10 // (op 600) 

v_lshl_add_u32 v10, s16, 5, v16 // (op 600) 

v_lshrrev_b32 v16, 5, v1 // (op 600) 
v_mul_lo_u32 v17, v16, s20 // (op 600) 

v_add_u32 v16, v10, v17 // (op 600) 


s_mul_i32 s1, s20, s25 // (op 602) most significant: low of low * high
s_mul_i32 s14, s21, s24 // (op 602) most significant: low of high * low
s_mul_i32 s0, s20, s24 // (op 602) least significant
s_mul_hi_u32 s15, s20, s24 // (op 602) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 602) most significant: sum
s_add_u32 s1, s1, s15 // (op 602) most significant: sum


s_mov_b32 s48, s22 // (op 602) reg expression
s_mov_b32 s49, s23 // (op 602) reg expression
s_mov_b32 s50, s0 // (op 602) reg expression
s_mov_b32 s51, 131072 // (op 602) call()

v_and_b32 v10, 15, v1 // (op 645) 
v_lshl_add_u32 v17, s17, 4, v10 // (op 645) 

v_mul_lo_u32 v10, v17, s26 // (op 645) 

v_and_b32 v17, 63, v1 // (op 645) 
v_lshrrev_b32 v18, 4, v17 // (op 645) 

v_add_u32 v17, v18, v10 // (op 645) 


s_mul_i32 s1, s26, s13 // (op 647) most significant: low of low * high
s_mul_i32 s14, s27, s12 // (op 647) most significant: low of high * low
s_mul_i32 s0, s26, s12 // (op 647) least significant
s_mul_hi_u32 s15, s26, s12 // (op 647) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 647) most significant: sum
s_add_u32 s1, s1, s15 // (op 647) most significant: sum


s_mov_b32 s52, s28 // (op 647) reg expression
s_mov_b32 s53, s29 // (op 647) reg expression
s_mov_b32 s54, s0 // (op 647) reg expression
s_mov_b32 s55, 131072 // (op 647) call()

v_and_b32 v10, 15, v1 // (op 664) 
v_lshl_add_u32 v18, s17, 4, v10 // (op 664) 

v_mul_lo_u32 v10, v18, s26 // (op 664) 

v_and_b32 v18, 63, v1 // (op 664) 
v_lshrrev_b32 v19, 4, v18 // (op 664) 

v_add_u32 v18, v19, v10 // (op 664) 


s_mul_i32 s1, s26, s13 // (op 666) most significant: low of low * high
s_mul_i32 s14, s27, s12 // (op 666) most significant: low of high * low
s_mul_i32 s0, s26, s12 // (op 666) least significant
s_mul_hi_u32 s15, s26, s12 // (op 666) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 666) most significant: sum
s_add_u32 s1, s1, s15 // (op 666) most significant: sum


s_mov_b32 s52, s28 // (op 666) reg expression
s_mov_b32 s53, s29 // (op 666) reg expression
s_mov_b32 s54, s0 // (op 666) reg expression
s_mov_b32 s55, 131072 // (op 666) call()




buffer_load_ubyte v10, v17, s[52:55], 0 offen offset: 0 // (op 341) Load value



buffer_load_dwordx4 v[20:23], v14, s[44:47], 0 offen // (op 345) Load value


v_mov_b32 v19, v12 // (op 348) reg expression

buffer_load_dwordx4 v[24:27], v19, s[40:43], 0 offen // (op 348) Load value



v_mov_b32 v19, v11 // (op 351) reg expression

buffer_load_ubyte v28, v19, s[48:51], 0 offen offset: 0 // (op 351) Load value




buffer_load_ubyte v19, v17, s[52:55], 0 offen offset:4 // (op 355) Load value




buffer_load_dwordx4 v[30:33], v14, s[44:47], 0 offen offset:128 // (op 359) Load value



s_lshl_b64 s[0:1], s[2:3], 7 // (op 362) 
s_lshl_b64 s[14:15], s[0:1], 2 // (op 362) 

s_lshr_b64 s[0:1], s[14:15], 3 // (op 362) 

v_add_u32 v14, v12, s0 // (op 362) 


buffer_load_dwordx4 v[34:37], v14, s[40:43], 0 offen // (op 362) Load value




s_lshl_b32 s0, s20, 2 // (op 365) 
v_add_u32 v12, v11, s0 // (op 365) 


buffer_load_ubyte v14, v12, s[48:51], 0 offen offset: 0 // (op 365) Load value


v_and_b32 v17, -8, v1 // (op 784) 
v_sub_u32 v29, v1, v17 // (op 784) 

v_lshlrev_b32 v17, 4, v29 // (op 784) 

v_lshrrev_b32 v29, 3, v1 // (op 784) 
v_lshl_add_u32 v11, v29, 7, v17 // (op 784) 





s_waitcnt vmcnt(6)
ds_write_b128 v11, v[20:23] // (op 367) Store local data 

v_lshlrev_b32 v17, 7, v1 // (op 731) 
v_lshrrev_b32 v29, 3, v17 // (op 731) 




s_waitcnt vmcnt(5)
ds_write_b128 v29, v[24:27] offset:2048 // (op 368) Store local data 

v_and_b32 v17, -32, v1 // (op 837) 
v_sub_u32 v29, v1, v17 // (op 837) 

v_and_b32 v17, 4294967264, v1 // (op 837) 
v_add_u32 v11, v17, v29 // (op 837) 





s_waitcnt vmcnt(4)
ds_write_b8 v11, v28 offset:4096 // (op 369) Store local data 

v_lshlrev_b32 v17, 7, v1 // (op 702) 
v_lshrrev_b32 v29, 3, v17 // (op 702) 

v_and_b32 v17, -8, v1 // (op 755) 
v_sub_u32 v11, v1, v17 // (op 755) 

v_lshlrev_b32 v12, 4, v11 // (op 755) 

v_lshrrev_b32 v11, 3, v1 // (op 755) 
v_lshl_add_u32 v17, v11, 7, v12 // (op 755) 


v_and_b32 v11, -32, v1 // (op 808) 
v_sub_u32 v12, v1, v11 // (op 808) 

v_and_b32 v11, 4294967264, v1 // (op 808) 
v_add_u32 v38, v11, v12 // (op 808) 


v_lshlrev_b32 v11, 7, v1 // (op 895) 
v_lshrrev_b32 v12, 3, v11 // (op 895) 

v_and_b32 v11, -8, v1 // (op 919) 
v_sub_u32 v39, v1, v11 // (op 919) 

v_lshlrev_b32 v11, 4, v39 // (op 919) 

v_lshrrev_b32 v39, 3, v1 // (op 919) 
v_lshl_add_u32 v40, v39, 7, v11 // (op 919) 


v_and_b32 v11, -32, v1 // (op 943) 
v_sub_u32 v39, v1, v11 // (op 943) 

v_and_b32 v11, 4294967264, v1 // (op 943) 
v_add_u32 v41, v11, v39 // (op 943) 


s_mov_b32 s0, 0 // (op 57) call()
s_ashr_i64 s[14:15], s[6:7], 7 // (op 56) 
s_mov_b32 s1, s14 // (op 56) convert

s_and_b32 s14, s1, 4294967294 // (op 56) 

s_cmp_lt_u32 s0, s14 // (op 56) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoop_56 // (op 56) Condition: Top (jump to RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoop_56 if false)
RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoop_56:
s_add_u32 s0, s0, 2 // (op 58) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 988) 



ds_read_b64_tr_b4 v[42:43], v3 offset:2048 // (op 67) Transpose load local data 
ds_read_b64_tr_b4 v[44:45], v3 offset:2304 // (op 67) Transpose load local data 
s_barrier  // (op 1001) 



ds_read_b128 v[46:49], v5 // (op 72) Load local data 
ds_read_b128 v[50:53], v5 offset:64 // (op 72) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1015) 


ds_read_u8 v11, v0 offset:4096 // (op 77) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[42:45], v[46:53], a[0:3], v11, v10 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 84) 





v_mov_b32 v11, v18 // (op 82) 

buffer_load_ubyte v10, v11, s[52:55], 0 offen offset:8 // (op 82) Load value


v_mov_b32 v39, v15 // (op 10) 

buffer_load_dwordx4 v[20:23], v39, s[44:47], 0 offen offset:256 // (op 10) Load value


v_mov_b32 v42, v13 // (op 4) 
s_lshl_b64 s[14:15], s[2:3], 7 // (op 4) 
s_lshl_b64 s[18:19], s[14:15], 2 // (op 4) 

s_lshr_b64 s[14:15], s[18:19], 3 // (op 4) 

s_lshl_b32 s1, s14, 1 // (op 4) 

v_add_u32 v43, v42, s1 // (op 4) 


buffer_load_dwordx4 v[24:27], v43, s[40:43], 0 offen // (op 4) Load value



v_mov_b32 v43, v16 // (op 17) 
s_lshl_b32 s1, s20, 3 // (op 17) 
v_add_u32 v44, v43, s1 // (op 17) 


buffer_load_ubyte v28, v44, s[48:51], 0 offen offset: 0 // (op 17) Load value

s_barrier  // (op 1038) 



s_waitcnt vmcnt(6)
ds_write_b128 v40, v[30:33] offset:4224 // (op 239) Store local data 
s_barrier  // (op 1032) 



s_waitcnt vmcnt(5)
ds_write_b128 v12, v[34:37] offset:6272 // (op 229) Store local data 
s_barrier  // (op 1044) 



s_waitcnt vmcnt(4)
ds_write_b8 v41, v14 offset:8320 // (op 240) Store local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1029) 



ds_read_b64_tr_b4 v[44:45], v6 offset:6272 // (op 231) Transpose load local data 
ds_read_b64_tr_b4 v[46:47], v6 offset:6528 // (op 231) Transpose load local data 
s_barrier  // (op 1035) 
s_barrier  // (op 1099) 



ds_read_b128 v[48:51], v8 offset:4224 // (op 233) Load local data 
ds_read_b128 v[52:55], v8 offset:4288 // (op 233) Load local data 
s_barrier  // (op 1105) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1041) 


ds_read_u8 v56, v4 offset:8320 // (op 235) Load local data 
s_barrier  // (op 1111) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[44:47], v[48:55], a[0:3], v56, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 238) 






buffer_load_ubyte v19, v11, s[52:55], 0 offen offset:12 // (op 237) Load value




v_add_u32 v18, 8, v18 // (op 656) 
buffer_load_dwordx4 v[30:33], v39, s[44:47], 0 offen offset:384 // (op 241) Load value



v_add_u32 v15, 256, v15 // (op 528) 
s_lshl_b64 s[14:15], s[2:3], 7 // (op 228) 
s_lshl_b64 s[18:19], s[14:15], 2 // (op 228) 

s_lshr_b64 s[14:15], s[18:19], 3 // (op 228) 

s_mul_i32 s1, 3, s14 // (op 228) 

v_add_u32 v11, v42, s1 // (op 228) 


buffer_load_dwordx4 v[34:37], v11, s[40:43], 0 offen // (op 228) Load value




s_lshl_b32 s1, s20, 2 // (op 242) 
s_mul_i32 s14, 3, s1 // (op 242) 

v_add_u32 v11, v43, s14 // (op 242) 


s_lshl_b64 s[14:15], s[2:3], 7 // (op 464) 
s_lshl_b64 s[18:19], s[14:15], 2 // (op 464) 

s_lshr_b64 s[14:15], s[18:19], 3 // (op 464) 

s_lshl_b32 s1, s14, 1 // (op 464) 

v_add_u32 v13, v13, s1 // (op 464) 

buffer_load_ubyte v14, v11, s[48:51], 0 offen offset: 0 // (op 242) Load value


s_barrier  // (op 1004) 
s_lshl_b32 s1, s20, 3 // (op 592) 



v_add_u32 v16, v16, s1 // (op 592) 

s_waitcnt vmcnt(6)
ds_write_b128 v17, v[20:23] // (op 13) Store local data 
s_barrier  // (op 991) 



s_waitcnt vmcnt(5)
ds_write_b128 v29, v[24:27] offset:2048 // (op 6) Store local data 
s_barrier  // (op 1018) 



s_waitcnt vmcnt(4)
ds_write_b8 v38, v28 offset:4096 // (op 21) Store local data 
s_ashr_i64 s[14:15], s[6:7], 7 // (op 56) 
s_mov_b32 s1, s14 // (op 56) convert

s_and_b32 s14, s1, 4294967294 // (op 56) 

s_cmp_lt_u32 s0, s14 // (op 56) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoop_56 // (op 56) Condition: Bottom (jump to RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoop_56 if true)
RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoop_56:















s_ashr_i64 s[0:1], s[6:7], 7 // (op 427) 
s_lshr_b32 s14, s0, 1 // (op 427) 

v_lshl_add_u32 v10, s14, 8, v1 // (op 427) 

s_lshl_b32 s0, s16, 5 // (op 427) 
v_mul_lo_u32 v13, v10, s3 // (op 427) most significant: low of low * high
v_mul_hi_u32 v11, v10, s2 // (op 427) most significant: high of low * low
v_mul_lo_u32 v12, v10, s2 // (op 427) least significant: low of low * low
v_add_u32 v13, v13, v11 // (op 427) most significant: sum


s_mov_b32 s15, 0 // (op 427) convert
s_mov_b32 s14, s0 // (op 427) convert

v_mov_b32 v10, s14 // (op 427) 
v_mov_b32 v11, s15 // (op 427) 
v_add_co_u32 v20, s[0:1], v10, v12 // (op 427) least significant half
v_addc_co_u32 v21, s[0:1], v11, v13, s[0:1] // (op 427) most significant half



v_lshlrev_b64 v[20:21], 2, v[20:21] // (op 427) 


v_lshrrev_b64 v[10:11], 3, v[20:21] // (op 427) 

v_mov_b32 v12, v10 // (op 427) convert

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

v_and_b32 v13, -8, v1 // (op 494) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 494) 
v_sub_u32 v10, v1, v13 // (op 494) 

s_lshr_b32 s14, s0, 1 // (op 494) 

v_lshlrev_b32 v11, 4, v10 // (op 494) 

v_lshl_add_u32 v10, s14, 8, v11 // (op 494) 


v_lshrrev_b32 v11, 3, v1 // (op 494) 
v_lshl_add_u32 v13, s17, 4, v11 // (op 494) 

v_mul_lo_u32 v11, v13, s8 // (op 494) 

v_add_u32 v13, v10, v11 // (op 494) 


s_mul_i32 s1, s8, s13 // (op 496) most significant: low of low * high
s_mul_i32 s14, s9, s12 // (op 496) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 496) least significant
s_mul_hi_u32 s15, s8, s12 // (op 496) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 496) most significant: sum
s_add_u32 s1, s1, s15 // (op 496) most significant: sum


s_mov_b32 s44, s10 // (op 496) reg expression
s_mov_b32 s45, s11 // (op 496) reg expression
s_mov_b32 s46, s0 // (op 496) reg expression
s_mov_b32 s47, 131072 // (op 496) call()

v_and_b32 v10, -32, v1 // (op 558) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 558) 
v_sub_u32 v11, v1, v10 // (op 558) 

s_lshr_b32 s14, s0, 1 // (op 558) 

v_lshl_add_u32 v10, s16, 5, v11 // (op 558) 

v_lshrrev_b32 v11, 5, v1 // (op 558) 
v_lshl_add_u32 v15, s14, 3, v11 // (op 558) 


v_mul_lo_u32 v11, v15, s20 // (op 558) 

v_add_u32 v15, v10, v11 // (op 558) 


s_mul_i32 s1, s20, s25 // (op 560) most significant: low of low * high
s_mul_i32 s14, s21, s24 // (op 560) most significant: low of high * low
s_mul_i32 s0, s20, s24 // (op 560) least significant
s_mul_hi_u32 s15, s20, s24 // (op 560) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 560) most significant: sum
s_add_u32 s1, s1, s15 // (op 560) most significant: sum


s_mov_b32 s48, s22 // (op 560) reg expression
s_mov_b32 s49, s23 // (op 560) reg expression
s_mov_b32 s50, s0 // (op 560) reg expression
s_mov_b32 s51, 131072 // (op 560) call()

v_and_b32 v10, 63, v1 // (op 622) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 622) 
v_lshrrev_b32 v11, 4, v10 // (op 622) 

s_lshr_b32 s14, s0, 1 // (op 622) 

v_and_b32 v10, 15, v1 // (op 622) 
v_lshl_add_u32 v16, s14, 3, v11 // (op 622) 


v_lshl_add_u32 v11, s17, 4, v10 // (op 622) 

v_mul_lo_u32 v10, v11, s26 // (op 622) 

v_add_u32 v11, v16, v10 // (op 622) 


s_mul_i32 s1, s26, s13 // (op 624) most significant: low of low * high
s_mul_i32 s14, s27, s12 // (op 624) most significant: low of high * low
s_mul_i32 s0, s26, s12 // (op 624) least significant
s_mul_hi_u32 s15, s26, s12 // (op 624) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 624) most significant: sum
s_add_u32 s1, s1, s15 // (op 624) most significant: sum


s_mov_b32 s52, s28 // (op 624) reg expression
s_mov_b32 s53, s29 // (op 624) reg expression
s_mov_b32 s54, s0 // (op 624) reg expression
s_mov_b32 s55, 131072 // (op 624) call()

v_lshlrev_b32 v10, 7, v1 // (op 715) 
v_lshrrev_b32 v16, 3, v10 // (op 715) 

v_and_b32 v10, -8, v1 // (op 768) 
v_sub_u32 v17, v1, v10 // (op 768) 

v_lshlrev_b32 v10, 4, v17 // (op 768) 

v_lshrrev_b32 v17, 3, v1 // (op 768) 
v_lshl_add_u32 v18, v17, 7, v10 // (op 768) 


v_and_b32 v10, -32, v1 // (op 821) 
v_sub_u32 v17, v1, v10 // (op 821) 

v_and_b32 v10, 4294967264, v1 // (op 821) 
v_add_u32 v20, v10, v17 // (op 821) 


s_ashr_i64 s[0:1], s[6:7], 7 // (op 166) 
s_and_b32 s14, s0, 4294967294 // (op 166) 

s_ashr_i64 s[0:1], s[6:7], 7 // (op 165) 
s_mov_b32 s15, s0 // (op 165) convert

s_cmp_lt_u32 s14, s15 // (op 165) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoopTail_165 // (op 165) Condition: Top (jump to RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoopTail_165 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoopTail_165:


s_add_u32 s14, s14, 1 // (op 167) 
v_mov_b32 v10, v12 // (op 170) 
v_mov_b32 v14, v10 // (op 170) reg expression

buffer_load_dwordx4 v[22:25], v14, s[40:43], 0 offen // (op 170) Load value


s_lshl_b64 s[0:1], s[2:3], 7 // (op 417) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 417) 

s_lshr_b64 s[0:1], s[18:19], 3 // (op 417) 

v_add_u32 v12, v12, s0 // (op 417) 



s_barrier  // (op 994) 
v_mov_b32 v10, v13 // (op 183) 




buffer_load_dwordx4 v[26:29], v10, s[44:47], 0 offen // (op 183) Load value

v_add_u32 v13, 128, v13 // (op 486) 
s_waitcnt vmcnt(1)
ds_write_b128 v16, v[22:25] offset:2048 // (op 171) Store local data 



v_mov_b32 v10, v15 // (op 184) 
s_barrier  // (op 1007) 
v_mov_b32 v14, v10 // (op 184) reg expression




buffer_load_ubyte v17, v14, s[48:51], 0 offen offset: 0 // (op 184) Load value


s_lshl_b32 s15, s20, 2 // (op 550) 
v_add_u32 v15, v15, s15 // (op 550) 

s_waitcnt vmcnt(1)
ds_write_b128 v18, v[26:29] // (op 181) Store local data 

s_waitcnt lgkmcnt(1)
s_barrier  // (op 997) 



s_barrier  // (op 1021) 
ds_read_b64_tr_b4 v[22:23], v3 offset:2048 // (op 173) Transpose load local data 



ds_read_b64_tr_b4 v[24:25], v3 offset:2304 // (op 173) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v20, v17 offset:4096 // (op 182) Store local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 1011) 



ds_read_b128 v[26:29], v5 // (op 175) Load local data 
ds_read_b128 v[30:33], v5 offset:64 // (op 175) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1025) 


ds_read_u8 v10, v0 offset:4096 // (op 177) Load local data 


v_mov_b32 v14, v11 // (op 179) 

buffer_load_ubyte v17, v14, s[52:55], 0 offen offset: 0 // (op 179) Load value

v_add_u32 v11, 4, v11 // (op 614) 
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:25], v[26:33], a[0:3], v10, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 180) 




s_ashr_i64 s[0:1], s[6:7], 7 // (op 165) 
s_mov_b32 s15, s0 // (op 165) convert

s_cmp_lt_u32 s14, s15 // (op 165) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoopTail_165 // (op 165) Condition: Bottom (jump to RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoopTail_165 if true)
RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoopTail_165:






















s_barrier  // (op 1093) 






v_cmp_eq_f32 vcc, 0.00000, v7 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalFalse_0_beta_125
s_nop 0
v_accvgpr_read v10, a0 // (op 46) Wait state hazard: XDL Write Hazard
v_accvgpr_read v11, a1 // (op 46) 
v_accvgpr_read v12, a2 // (op 46) 
v_accvgpr_read v13, a3 // (op 46) 
v_mul_f32 v14, v9, v10 // (op 46) 
v_mul_f32 v15, v9, v11 // (op 46) 
v_mul_f32 v16, v9, v12 // (op 46) 
v_mul_f32 v17, v9, v13 // (op 46) 

v_cvt_f16_f32 v10, v14 // (op 52) 
v_cvt_f16_f32 v11, v15 // (op 52) 
v_cvt_f16_f32 v12, v16 // (op 52) 
v_cvt_f16_f32 v13, v17 // (op 52) 

v_lshrrev_b32 v0, 6, v1 // (op 863) 
v_and_b32 v8, 63, v1 // (op 863) 
v_lshrrev_b32 v3, 4, v8 // (op 863) 
v_lshlrev_b32 v4, 2, v3 // (op 863) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 863) 


v_lshl_add_u32 v0, s16, 5, v3 // (op 863) 

v_lshrrev_b32 v3, 2, v8 // (op 863) 

v_and_b32 v8, 3, v3 // (op 863) 

v_and_b32 v3, 3, v1 // (op 863) 
v_lshl_add_u32 v4, v8, 2, v3 // (op 863) 


v_lshl_add_u32 v3, s17, 4, v4 // (op 863) 

v_mul_lo_u32 v8, v3, s30 // (op 863) 

v_add_lshl_u32 v3, v0, v8, 1 // (op 863) 


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


v_pack_b32_f16 v4, v10, v11 // (op 54) 
v_pack_b32_f16 v5, v12, v13 // (op 54) 


buffer_store_dwordx2 v[4:5], v3, s[8:11], 0 offen // (op 54) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalBottom_0_beta_125
RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalFalse_0_beta_125:
v_lshrrev_b32 v0, 6, v1 // (op 685) 
v_and_b32 v8, 63, v1 // (op 685) 
v_lshrrev_b32 v3, 4, v8 // (op 685) 
v_lshlrev_b32 v4, 2, v3 // (op 685) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 685) 


v_lshl_add_u32 v0, s16, 5, v3 // (op 685) 

v_lshrrev_b32 v3, 2, v8 // (op 685) 

v_and_b32 v8, 3, v3 // (op 685) 

v_and_b32 v3, 3, v1 // (op 685) 
v_lshl_add_u32 v4, v8, 2, v3 // (op 685) 


v_lshl_add_u32 v3, s17, 4, v4 // (op 685) 

v_mul_lo_u32 v8, v3, s34 // (op 685) 

v_add_lshl_u32 v3, v0, v8, 1 // (op 685) 


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
v_cvt_f32_f16 v10, v4 // (op 139) 
v_lshrrev_b32 v11, 16, v4 // (op 139) 
v_cvt_f32_f16 v11, v11 // (op 139) 
v_mul_f32 v18, v7, v10 // (op 139) 
v_mul_f32 v19, v7, v11 // (op 139) 
v_cvt_f32_f16 v10, v5 // (op 139) 
v_lshrrev_b32 v11, 16, v5 // (op 139) 
v_cvt_f32_f16 v11, v11 // (op 139) 
v_mul_f32 v20, v7, v10 // (op 139) 
v_mul_f32 v21, v7, v11 // (op 139) 


v_add_f32 v10, v18, v14 // (op 135) 
v_add_f32 v11, v19, v15 // (op 135) 
v_add_f32 v12, v20, v16 // (op 135) 
v_add_f32 v13, v21, v17 // (op 135) 


v_cvt_f16_f32 v14, v10 // (op 136) 
v_cvt_f16_f32 v15, v11 // (op 136) 
v_cvt_f16_f32 v16, v12 // (op 136) 
v_cvt_f16_f32 v17, v13 // (op 136) 

v_lshrrev_b32 v0, 6, v1 // (op 880) 
v_and_b32 v8, 63, v1 // (op 880) 
v_lshrrev_b32 v3, 4, v8 // (op 880) 
v_lshlrev_b32 v4, 2, v3 // (op 880) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 880) 


v_lshl_add_u32 v0, s16, 5, v3 // (op 880) 

v_lshrrev_b32 v3, 2, v8 // (op 880) 

v_and_b32 v8, 3, v3 // (op 880) 

v_and_b32 v3, 3, v1 // (op 880) 
v_lshl_add_u32 v4, v8, 2, v3 // (op 880) 


v_lshl_add_u32 v3, s17, 4, v4 // (op 880) 

v_mul_lo_u32 v8, v3, s30 // (op 880) 

v_add_lshl_u32 v3, v0, v8, 1 // (op 880) 


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


v_pack_b32_f16 v4, v14, v15 // (op 137) 
v_pack_b32_f16 v5, v16, v17 // (op 137) 


buffer_store_dwordx2 v[4:5], v3, s[8:11], 0 offen // (op 137) Store value






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalBottom_0_beta_125:




s_endpgm  // End of RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2
.LRR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_end:
.size RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2, .LRR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_end-RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2
  .amdhsa_next_free_vgpr 64
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 8448 // lds bytes
  .amdhsa_accum_offset 60
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
    DataTypeB: F8
    DestDataType: h
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
    macrotile: [32, 16, 128]
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
  - .name: RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2
    .symbol: RR_GEMM_NN_FP4_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 8448
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 56
    .vgpr_count: 57
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

