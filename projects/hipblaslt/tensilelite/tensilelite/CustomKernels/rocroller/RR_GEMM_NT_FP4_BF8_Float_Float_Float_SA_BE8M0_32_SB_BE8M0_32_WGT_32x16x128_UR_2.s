// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2
.p2align 8
.type RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2,@function
RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_exec_begin:


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


v_and_b32 v0, 63, v1 // (op 739) 
v_and_b32 v3, 48, v1 // (op 739) 
v_sub_u32 v4, v0, v3 // (op 739) 

v_lshrrev_b32 v3, 4, v0 // (op 739) 

v_lshl_add_u32 v0, v3, 5, v4 // (op 739) 


v_lshrrev_b32 v3, 6, v1 // (op 739) 
v_lshlrev_b32 v4, 4, v3 // (op 739) 

v_lshl_add_u32 v3, v0, 5, v4 // (op 739) 


v_lshlrev_b32 v0, 2, v3 // (op 739) 

v_lshrrev_b32 v3, 3, v0 // (op 739) 

v_and_b32 v0, 48, v1 // (op 792) 
v_and_b32 v4, 63, v1 // (op 792) 
v_sub_u32 v5, v4, v0 // (op 792) 

v_lshrrev_b32 v4, 1, v5 // (op 792) 
v_add_lshl_u32 v6, v0, v4, 4 // (op 792) 


v_and_b32 v0, 1, v5 // (op 792) 

v_lshlrev_b32 v4, 3, v0 // (op 792) 

v_add_u32 v0, v6, v4 // (op 792) 


v_lshrrev_b32 v4, 6, v1 // (op 845) 
v_and_b32 v5, 15, v1 // (op 845) 
v_lshl_add_u32 v6, v4, 4, v5 // (op 845) 


v_and_b32 v4, 63, v1 // (op 845) 
v_lshrrev_b32 v5, 4, v4 // (op 845) 

v_lshl_add_u32 v4, v5, 5, v6 // (op 845) 


v_and_b32 v5, 63, v1 // (op 903) 
v_and_b32 v6, 48, v1 // (op 903) 
v_sub_u32 v7, v5, v6 // (op 903) 

v_lshrrev_b32 v6, 4, v5 // (op 903) 

v_lshl_add_u32 v5, v6, 5, v7 // (op 903) 


v_lshrrev_b32 v6, 6, v1 // (op 903) 
v_lshlrev_b32 v7, 4, v6 // (op 903) 

v_lshl_add_u32 v6, v5, 5, v7 // (op 903) 


v_lshlrev_b32 v5, 2, v6 // (op 903) 

v_lshrrev_b32 v6, 3, v5 // (op 903) 

v_and_b32 v5, 48, v1 // (op 927) 
v_and_b32 v7, 63, v1 // (op 927) 
v_sub_u32 v8, v7, v5 // (op 927) 

v_lshrrev_b32 v7, 1, v8 // (op 927) 
v_add_lshl_u32 v9, v5, v7, 4 // (op 927) 


v_and_b32 v5, 1, v8 // (op 927) 

v_lshlrev_b32 v7, 3, v5 // (op 927) 

v_add_u32 v5, v9, v7 // (op 927) 


v_lshrrev_b32 v7, 6, v1 // (op 951) 
v_and_b32 v8, 15, v1 // (op 951) 
v_lshl_add_u32 v9, v7, 4, v8 // (op 951) 


v_and_b32 v7, 63, v1 // (op 951) 
v_lshrrev_b32 v8, 4, v7 // (op 951) 

v_lshl_add_u32 v7, v8, 5, v9 // (op 951) 


v_mov_b32 v8, s14 // (op 2) Move value
v_mov_b32 v9, s15 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 59) call()
v_accvgpr_write a1, 0.00000 // (op 59) call()
v_accvgpr_write a2, 0.00000 // (op 59) call()
v_accvgpr_write a3, 0.00000 // (op 59) call()
s_lshl_b32 s0, s16, 5 // (op 449) 
v_mul_lo_u32 v11, v1, s3 // (op 449) most significant: low of low * high
s_mov_b32 s15, 0 // (op 449) convert
s_mov_b32 s14, s0 // (op 449) convert

v_mul_hi_u32 v12, v1, s2 // (op 449) most significant: high of low * low
v_mul_lo_u32 v10, v1, s2 // (op 449) least significant: low of low * low
v_add_u32 v11, v11, v12 // (op 449) most significant: sum

v_mov_b32 v12, s14 // (op 449) 
v_mov_b32 v13, s15 // (op 449) 
v_add_co_u32 v14, s[0:1], v12, v10 // (op 449) least significant half
v_addc_co_u32 v15, s[0:1], v13, v11, s[0:1] // (op 449) most significant half



v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 449) 


v_lshrrev_b64 v[10:11], 3, v[14:15] // (op 449) 

v_mov_b32 v12, v10 // (op 449) convert

s_mul_i32 s1, s2, s7 // (op 451) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 451) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 451) least significant
s_mul_hi_u32 s15, s2, s6 // (op 451) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 451) most significant: sum
s_add_u32 s1, s1, s15 // (op 451) most significant: sum


s_lshl_b64 s[14:15], s[0:1], 2 // (op 451) 

s_ashr_i64 s[0:1], s[14:15], 3 // (op 451) 

s_mov_b32 s40, s4 // (op 451) reg expression
s_mov_b32 s41, s5 // (op 451) reg expression
s_mov_b32 s42, s0 // (op 451) reg expression
s_mov_b32 s43, 131072 // (op 451) call()

s_lshl_b32 s0, s16, 5 // (op 468) 
v_mul_lo_u32 v11, v1, s3 // (op 468) most significant: low of low * high
v_mul_hi_u32 v13, v1, s2 // (op 468) most significant: high of low * low
v_mul_lo_u32 v10, v1, s2 // (op 468) least significant: low of low * low
v_add_u32 v11, v11, v13 // (op 468) most significant: sum

s_mov_b32 s15, 0 // (op 468) convert
s_mov_b32 s14, s0 // (op 468) convert

v_mov_b32 v13, s14 // (op 468) 
v_mov_b32 v14, s15 // (op 468) 
v_add_co_u32 v16, s[0:1], v13, v10 // (op 468) least significant half
v_addc_co_u32 v17, s[0:1], v14, v11, s[0:1] // (op 468) most significant half



v_lshlrev_b64 v[16:17], 2, v[16:17] // (op 468) 


v_lshrrev_b64 v[10:11], 3, v[16:17] // (op 468) 

v_mov_b32 v13, v10 // (op 468) convert

s_mul_i32 s1, s2, s7 // (op 470) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 470) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 470) least significant
s_mul_hi_u32 s15, s2, s6 // (op 470) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 470) most significant: sum
s_add_u32 s1, s1, s15 // (op 470) most significant: sum


s_lshl_b64 s[14:15], s[0:1], 2 // (op 470) 

s_ashr_i64 s[0:1], s[14:15], 3 // (op 470) 

s_mov_b32 s40, s4 // (op 470) reg expression
s_mov_b32 s41, s5 // (op 470) reg expression
s_mov_b32 s42, s0 // (op 470) reg expression
s_mov_b32 s43, 131072 // (op 470) call()

s_lshl_b32 s0, s17, 4 // (op 513) 
v_mul_lo_u32 v10, v1, s8 // (op 513) 
v_add_u32 v10, v10, s0 // (op 513) 

s_mul_i32 s1, s8, s7 // (op 515) most significant: low of low * high
s_mul_i32 s14, s9, s6 // (op 515) most significant: low of high * low
s_mul_i32 s0, s8, s6 // (op 515) least significant
s_mul_hi_u32 s15, s8, s6 // (op 515) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 515) most significant: sum
s_add_u32 s1, s1, s15 // (op 515) most significant: sum


s_mov_b32 s44, s10 // (op 515) reg expression
s_mov_b32 s45, s11 // (op 515) reg expression
s_mov_b32 s46, s0 // (op 515) reg expression
s_mov_b32 s47, 131072 // (op 515) call()

s_lshl_b32 s0, s17, 4 // (op 532) 
v_mul_lo_u32 v11, v1, s8 // (op 532) 
v_add_u32 v11, v11, s0 // (op 532) 

s_mul_i32 s1, s8, s7 // (op 534) most significant: low of low * high
s_mul_i32 s14, s9, s6 // (op 534) most significant: low of high * low
s_mul_i32 s0, s8, s6 // (op 534) least significant
s_mul_hi_u32 s15, s8, s6 // (op 534) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 534) most significant: sum
s_add_u32 s1, s1, s15 // (op 534) most significant: sum


s_mov_b32 s44, s10 // (op 534) reg expression
s_mov_b32 s45, s11 // (op 534) reg expression
s_mov_b32 s46, s0 // (op 534) reg expression
s_mov_b32 s47, 131072 // (op 534) call()

v_and_b32 v14, -32, v1 // (op 577) 
v_sub_u32 v15, v1, v14 // (op 577) 

v_lshl_add_u32 v14, s16, 5, v15 // (op 577) 

v_lshrrev_b32 v15, 5, v1 // (op 577) 
v_mul_lo_u32 v16, v15, s12 // (op 577) 

v_add_u32 v15, v14, v16 // (op 577) 



s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 579) most significant: low of low * high
s_mul_i32 s14, s13, s22 // (op 579) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 579) least significant
s_mul_hi_u32 s15, s12, s22 // (op 579) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 579) most significant: sum
s_add_u32 s1, s1, s15 // (op 579) most significant: sum


s_mov_b32 s48, s20 // (op 579) reg expression
s_mov_b32 s49, s21 // (op 579) reg expression
s_mov_b32 s50, s0 // (op 579) reg expression
s_mov_b32 s51, 131072 // (op 579) call()

v_and_b32 v14, -32, v1 // (op 596) 
v_sub_u32 v16, v1, v14 // (op 596) 

v_lshl_add_u32 v14, s16, 5, v16 // (op 596) 

v_lshrrev_b32 v16, 5, v1 // (op 596) 
v_mul_lo_u32 v17, v16, s12 // (op 596) 

v_add_u32 v16, v14, v17 // (op 596) 


s_mul_i32 s1, s12, s23 // (op 598) most significant: low of low * high
s_mul_i32 s14, s13, s22 // (op 598) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 598) least significant
s_mul_hi_u32 s15, s12, s22 // (op 598) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 598) most significant: sum
s_add_u32 s1, s1, s15 // (op 598) most significant: sum


s_mov_b32 s48, s20 // (op 598) reg expression
s_mov_b32 s49, s21 // (op 598) reg expression
s_mov_b32 s50, s0 // (op 598) reg expression
s_mov_b32 s51, 131072 // (op 598) call()

v_and_b32 v14, 15, v1 // (op 641) 
v_lshl_add_u32 v17, s17, 4, v14 // (op 641) 

v_and_b32 v14, 63, v1 // (op 641) 
v_lshrrev_b32 v18, 4, v14 // (op 641) 

v_mul_lo_u32 v14, v18, s24 // (op 641) 
v_add_u32 v14, v14, v17 // (op 641) 


s_mul_i32 s1, s24, s23 // (op 643) most significant: low of low * high
s_mul_i32 s14, s25, s22 // (op 643) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 643) least significant
s_mul_hi_u32 s15, s24, s22 // (op 643) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 643) most significant: sum
s_add_u32 s1, s1, s15 // (op 643) most significant: sum


s_mov_b32 s52, s26 // (op 643) reg expression
s_mov_b32 s53, s27 // (op 643) reg expression
s_mov_b32 s54, s0 // (op 643) reg expression
s_mov_b32 s55, 131072 // (op 643) call()

v_and_b32 v17, 15, v1 // (op 660) 
v_lshl_add_u32 v18, s17, 4, v17 // (op 660) 

v_and_b32 v17, 63, v1 // (op 660) 
v_lshrrev_b32 v19, 4, v17 // (op 660) 

v_mul_lo_u32 v17, v19, s24 // (op 660) 
v_add_u32 v17, v17, v18 // (op 660) 


s_mul_i32 s1, s24, s23 // (op 662) most significant: low of low * high
s_mul_i32 s14, s25, s22 // (op 662) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 662) least significant
s_mul_hi_u32 s15, s24, s22 // (op 662) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 662) most significant: sum
s_add_u32 s1, s1, s15 // (op 662) most significant: sum


s_mov_b32 s52, s26 // (op 662) reg expression
s_mov_b32 s53, s27 // (op 662) reg expression
s_mov_b32 s54, s0 // (op 662) reg expression
s_mov_b32 s55, 131072 // (op 662) call()


v_mov_b64 v[18:19], s[24:25] // (op 337) convert

v_mov_b32 v20, v14 // (op 337) reg expression
v_mov_b32 v21, v20 // (op 337) 
buffer_load_ubyte v22, v21, s[52:55], 0 offen offset: 0 // (op 337) Load value





v_mov_b32 v18, v10 // (op 341) reg expression

buffer_load_dwordx4 v[24:27], v18, s[44:47], 0 offen // (op 341) Load value



v_mov_b32 v23, v12 // (op 344) reg expression

buffer_load_dwordx4 v[18:21], v23, s[40:43], 0 offen // (op 344) Load value



v_mov_b32 v23, v15 // (op 347) reg expression

buffer_load_ubyte v28, v23, s[48:51], 0 offen offset: 0 // (op 347) Load value


v_mov_b64 v[30:31], s[24:25] // (op 351) convert

s_lshl_b32 s0, s24, 2 // (op 351) 
v_add_u32 v23, v14, s0 // (op 351) 

v_mov_b32 v29, v23 // (op 351) 
buffer_load_ubyte v32, v29, s[52:55], 0 offen offset: 0 // (op 351) Load value






s_lshl_b32 s0, s8, 7 // (op 355) 
v_add_u32 v14, v10, s0 // (op 355) 


buffer_load_dwordx4 v[34:37], v14, s[44:47], 0 offen // (op 355) Load value




s_lshl_b64 s[0:1], s[2:3], 7 // (op 358) 
s_lshl_b64 s[14:15], s[0:1], 2 // (op 358) 

s_lshr_b64 s[0:1], s[14:15], 3 // (op 358) 

v_add_u32 v10, v12, s0 // (op 358) 


buffer_load_dwordx4 v[38:41], v10, s[40:43], 0 offen // (op 358) Load value




s_lshl_b32 s0, s12, 2 // (op 361) 
v_add_u32 v10, v15, s0 // (op 361) 


buffer_load_ubyte v12, v10, s[48:51], 0 offen offset: 0 // (op 361) Load value


v_lshlrev_b32 v10, 4, v1 // (op 780) 



s_waitcnt vmcnt(6)
ds_write_b128 v10, v[24:27] // (op 363) Store local data 

v_lshlrev_b32 v10, 7, v1 // (op 727) 
v_lshrrev_b32 v23, 3, v10 // (op 727) 




s_waitcnt vmcnt(5)
ds_write_b128 v23, v[18:21] offset:2048 // (op 364) Store local data 

v_and_b32 v10, -32, v1 // (op 833) 
v_sub_u32 v23, v1, v10 // (op 833) 

v_and_b32 v10, 4294967264, v1 // (op 833) 
v_add_u32 v33, v10, v23 // (op 833) 





s_waitcnt vmcnt(4)
ds_write_b8 v33, v28 offset:4096 // (op 365) Store local data 

v_lshlrev_b32 v10, 7, v1 // (op 698) 
v_lshrrev_b32 v23, 3, v10 // (op 698) 

v_lshlrev_b32 v10, 4, v1 // (op 751) 
v_and_b32 v33, -32, v1 // (op 804) 
v_sub_u32 v14, v1, v33 // (op 804) 

v_and_b32 v15, 4294967264, v1 // (op 804) 
v_add_u32 v33, v15, v14 // (op 804) 


v_lshlrev_b32 v14, 7, v1 // (op 891) 
v_lshrrev_b32 v15, 3, v14 // (op 891) 

v_lshlrev_b32 v14, 4, v1 // (op 915) 
v_and_b32 v29, -32, v1 // (op 939) 
v_sub_u32 v30, v1, v29 // (op 939) 

v_and_b32 v29, 4294967264, v1 // (op 939) 
v_add_u32 v31, v29, v30 // (op 939) 


s_mov_b32 s0, 0 // (op 55) call()
s_ashr_i64 s[14:15], s[6:7], 7 // (op 54) 
s_mov_b32 s1, s14 // (op 54) convert

s_and_b32 s14, s1, 4294967294 // (op 54) 

s_cmp_lt_u32 s0, s14 // (op 54) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoop_54 // (op 54) Condition: Top (jump to RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoop_54 if false)
RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoop_54:
s_add_u32 s0, s0, 2 // (op 56) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 984) 



ds_read_b64_tr_b4 v[42:43], v3 offset:2048 // (op 65) Transpose load local data 
ds_read_b64_tr_b4 v[44:45], v3 offset:2304 // (op 65) Transpose load local data 
s_barrier  // (op 997) 



ds_read_b64_tr_b8 v[46:47], v0 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v0 offset:128 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v0 offset:1024 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v0 offset:1152 // (op 70) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1011) 


ds_read_u8 v29, v4 offset:4096 // (op 75) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[42:45], v[46:53], a[0:3], v29, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 82) 




v_mov_b64 v[42:43], s[24:25] // (op 80) convert

v_mov_b32 v29, v17 // (op 80) 
s_lshl_b32 s1, s24, 3 // (op 80) 
v_add_u32 v30, v29, s1 // (op 80) 

v_mov_b32 v44, v30 // (op 80) 
buffer_load_ubyte v22, v44, s[52:55], 0 offen offset: 0 // (op 80) Load value





v_mov_b32 v30, v11 // (op 10) 
s_lshl_b32 s1, s8, 8 // (op 10) 
v_add_u32 v42, v30, s1 // (op 10) 


buffer_load_dwordx4 v[24:27], v42, s[44:47], 0 offen // (op 10) Load value



v_mov_b32 v42, v13 // (op 4) 
s_lshl_b64 s[14:15], s[2:3], 7 // (op 4) 
s_lshl_b64 s[18:19], s[14:15], 2 // (op 4) 

s_lshr_b64 s[14:15], s[18:19], 3 // (op 4) 

s_lshl_b32 s1, s14, 1 // (op 4) 

v_add_u32 v43, v42, s1 // (op 4) 


buffer_load_dwordx4 v[18:21], v43, s[40:43], 0 offen // (op 4) Load value



v_mov_b32 v43, v16 // (op 17) 
s_lshl_b32 s1, s12, 3 // (op 17) 
v_add_u32 v44, v43, s1 // (op 17) 


buffer_load_ubyte v28, v44, s[48:51], 0 offen offset: 0 // (op 17) Load value

s_barrier  // (op 1034) 



s_waitcnt vmcnt(6)
ds_write_b128 v14, v[34:37] offset:4224 // (op 235) Store local data 
s_barrier  // (op 1028) 



s_waitcnt vmcnt(5)
ds_write_b128 v15, v[38:41] offset:6272 // (op 225) Store local data 
s_barrier  // (op 1040) 



s_waitcnt vmcnt(4)
ds_write_b8 v31, v12 offset:8320 // (op 236) Store local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1025) 



ds_read_b64_tr_b4 v[44:45], v6 offset:6272 // (op 227) Transpose load local data 
ds_read_b64_tr_b4 v[46:47], v6 offset:6528 // (op 227) Transpose load local data 
s_barrier  // (op 1031) 
s_barrier  // (op 1091) 



ds_read_b64_tr_b8 v[48:49], v5 offset:4224 // (op 229) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v5 offset:4352 // (op 229) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v5 offset:5248 // (op 229) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v5 offset:5376 // (op 229) Transpose load local data 
s_barrier  // (op 1097) 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1037) 


ds_read_u8 v56, v7 offset:8320 // (op 231) Load local data 
s_barrier  // (op 1103) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[44:47], v[48:55], a[0:3], v56, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 234) 




v_mov_b64 v[44:45], s[24:25] // (op 233) convert

s_lshl_b32 s1, s24, 2 // (op 233) 
s_mul_i32 s14, 3, s1 // (op 233) 

v_add_u32 v46, v29, s14 // (op 233) 

v_mov_b32 v47, v46 // (op 233) 
buffer_load_ubyte v32, v47, s[52:55], 0 offen offset: 0 // (op 233) Load value






s_lshl_b32 s1, s8, 7 // (op 237) 
s_mul_i32 s14, 3, s1 // (op 237) 

s_lshl_b32 s1, s24, 3 // (op 652) 
v_add_u32 v17, v17, s1 // (op 652) 

v_add_u32 v29, v30, s14 // (op 237) 


buffer_load_dwordx4 v[34:37], v29, s[44:47], 0 offen // (op 237) Load value




s_lshl_b32 s1, s8, 8 // (op 524) 
v_add_u32 v11, v11, s1 // (op 524) 

s_lshl_b64 s[14:15], s[2:3], 7 // (op 224) 
s_lshl_b64 s[18:19], s[14:15], 2 // (op 224) 

s_lshr_b64 s[14:15], s[18:19], 3 // (op 224) 

s_mul_i32 s1, 3, s14 // (op 224) 

v_add_u32 v29, v42, s1 // (op 224) 


buffer_load_dwordx4 v[38:41], v29, s[40:43], 0 offen // (op 224) Load value




s_lshl_b32 s1, s12, 2 // (op 238) 
s_mul_i32 s14, 3, s1 // (op 238) 

v_add_u32 v42, v43, s14 // (op 238) 


s_lshl_b64 s[14:15], s[2:3], 7 // (op 460) 
s_lshl_b64 s[18:19], s[14:15], 2 // (op 460) 

s_lshr_b64 s[14:15], s[18:19], 3 // (op 460) 

s_lshl_b32 s1, s14, 1 // (op 460) 

v_add_u32 v13, v13, s1 // (op 460) 

buffer_load_ubyte v12, v42, s[48:51], 0 offen offset: 0 // (op 238) Load value


s_barrier  // (op 1000) 
s_lshl_b32 s1, s12, 3 // (op 588) 



v_add_u32 v16, v16, s1 // (op 588) 

s_waitcnt vmcnt(6)
ds_write_b128 v10, v[24:27] // (op 13) Store local data 
s_barrier  // (op 987) 



s_waitcnt vmcnt(5)
ds_write_b128 v23, v[18:21] offset:2048 // (op 6) Store local data 
s_barrier  // (op 1014) 



s_waitcnt vmcnt(4)
ds_write_b8 v33, v28 offset:4096 // (op 21) Store local data 
s_ashr_i64 s[14:15], s[6:7], 7 // (op 54) 
s_mov_b32 s1, s14 // (op 54) convert

s_and_b32 s14, s1, 4294967294 // (op 54) 

s_cmp_lt_u32 s0, s14 // (op 54) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoop_54 // (op 54) Condition: Bottom (jump to RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoop_54 if true)
RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoop_54:















s_ashr_i64 s[0:1], s[6:7], 7 // (op 423) 
s_lshr_b32 s14, s0, 1 // (op 423) 

v_lshl_add_u32 v33, s14, 8, v1 // (op 423) 

s_lshl_b32 s0, s16, 5 // (op 423) 
v_mul_lo_u32 v11, v33, s3 // (op 423) most significant: low of low * high
v_mul_hi_u32 v13, v33, s2 // (op 423) most significant: high of low * low
v_mul_lo_u32 v10, v33, s2 // (op 423) least significant: low of low * low
v_add_u32 v11, v11, v13 // (op 423) most significant: sum


s_mov_b32 s15, 0 // (op 423) convert
s_mov_b32 s14, s0 // (op 423) convert

v_mov_b32 v33, s14 // (op 423) 
v_mov_b32 v13, s15 // (op 423) 
v_add_co_u32 v14, s[0:1], v33, v10 // (op 423) least significant half
v_addc_co_u32 v15, s[0:1], v13, v11, s[0:1] // (op 423) most significant half



v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 423) 


v_lshrrev_b64 v[10:11], 3, v[14:15] // (op 423) 

v_mov_b32 v33, v10 // (op 423) convert

s_mul_i32 s1, s2, s7 // (op 425) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 425) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 425) least significant
s_mul_hi_u32 s15, s2, s6 // (op 425) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 425) most significant: sum
s_add_u32 s1, s1, s15 // (op 425) most significant: sum


s_lshl_b64 s[14:15], s[0:1], 2 // (op 425) 

s_ashr_i64 s[0:1], s[14:15], 3 // (op 425) 

s_mov_b32 s40, s4 // (op 425) reg expression
s_mov_b32 s41, s5 // (op 425) reg expression
s_mov_b32 s42, s0 // (op 425) reg expression
s_mov_b32 s43, 131072 // (op 425) call()

s_ashr_i64 s[0:1], s[6:7], 7 // (op 490) 
s_lshr_b32 s14, s0, 1 // (op 490) 

v_lshl_add_u32 v10, s14, 8, v1 // (op 490) 

s_lshl_b32 s0, s17, 4 // (op 490) 
v_mul_lo_u32 v11, v10, s8 // (op 490) 
v_add_u32 v11, v11, s0 // (op 490) 


s_mul_i32 s1, s8, s7 // (op 492) most significant: low of low * high
s_mul_i32 s14, s9, s6 // (op 492) most significant: low of high * low
s_mul_i32 s0, s8, s6 // (op 492) least significant
s_mul_hi_u32 s15, s8, s6 // (op 492) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 492) most significant: sum
s_add_u32 s1, s1, s15 // (op 492) most significant: sum


s_mov_b32 s44, s10 // (op 492) reg expression
s_mov_b32 s45, s11 // (op 492) reg expression
s_mov_b32 s46, s0 // (op 492) reg expression
s_mov_b32 s47, 131072 // (op 492) call()

v_and_b32 v10, -32, v1 // (op 554) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 554) 
v_sub_u32 v13, v1, v10 // (op 554) 

s_lshr_b32 s14, s0, 1 // (op 554) 

v_lshl_add_u32 v10, s16, 5, v13 // (op 554) 

v_lshrrev_b32 v13, 5, v1 // (op 554) 
v_lshl_add_u32 v14, s14, 3, v13 // (op 554) 


v_mul_lo_u32 v13, v14, s12 // (op 554) 

v_add_u32 v14, v10, v13 // (op 554) 


s_mul_i32 s1, s12, s23 // (op 556) most significant: low of low * high
s_mul_i32 s14, s13, s22 // (op 556) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 556) least significant
s_mul_hi_u32 s15, s12, s22 // (op 556) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 556) most significant: sum
s_add_u32 s1, s1, s15 // (op 556) most significant: sum


s_mov_b32 s48, s20 // (op 556) reg expression
s_mov_b32 s49, s21 // (op 556) reg expression
s_mov_b32 s50, s0 // (op 556) reg expression
s_mov_b32 s51, 131072 // (op 556) call()

v_and_b32 v10, 63, v1 // (op 618) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 618) 
v_lshrrev_b32 v13, 4, v10 // (op 618) 

s_lshr_b32 s14, s0, 1 // (op 618) 

v_and_b32 v10, 15, v1 // (op 618) 
v_lshl_add_u32 v15, s14, 3, v13 // (op 618) 


v_lshl_add_u32 v13, s17, 4, v10 // (op 618) 

v_mul_lo_u32 v10, v15, s24 // (op 618) 
v_add_u32 v10, v10, v13 // (op 618) 


s_mul_i32 s1, s24, s23 // (op 620) most significant: low of low * high
s_mul_i32 s14, s25, s22 // (op 620) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 620) least significant
s_mul_hi_u32 s15, s24, s22 // (op 620) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 620) most significant: sum
s_add_u32 s1, s1, s15 // (op 620) most significant: sum


s_mov_b32 s52, s26 // (op 620) reg expression
s_mov_b32 s53, s27 // (op 620) reg expression
s_mov_b32 s54, s0 // (op 620) reg expression
s_mov_b32 s55, 131072 // (op 620) call()

v_lshlrev_b32 v13, 7, v1 // (op 711) 
v_lshrrev_b32 v15, 3, v13 // (op 711) 

v_lshlrev_b32 v13, 4, v1 // (op 764) 
v_and_b32 v16, -32, v1 // (op 817) 
v_sub_u32 v17, v1, v16 // (op 817) 

v_and_b32 v16, 4294967264, v1 // (op 817) 
v_add_u32 v18, v16, v17 // (op 817) 


s_ashr_i64 s[0:1], s[6:7], 7 // (op 162) 
s_and_b32 s14, s0, 4294967294 // (op 162) 

s_ashr_i64 s[0:1], s[6:7], 7 // (op 161) 
s_mov_b32 s15, s0 // (op 161) convert

s_cmp_lt_u32 s14, s15 // (op 161) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoopTail_161 // (op 161) Condition: Top (jump to RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoopTail_161 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoopTail_161:


s_add_u32 s14, s14, 1 // (op 163) 
v_mov_b32 v12, v33 // (op 166) 
v_mov_b32 v16, v12 // (op 166) reg expression

buffer_load_dwordx4 v[20:23], v16, s[40:43], 0 offen // (op 166) Load value


s_lshl_b64 s[0:1], s[2:3], 7 // (op 413) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 413) 

s_lshr_b64 s[0:1], s[18:19], 3 // (op 413) 

v_add_u32 v33, v33, s0 // (op 413) 



s_barrier  // (op 990) 
v_mov_b32 v12, v11 // (op 179) 



v_mov_b32 v19, v12 // (op 179) reg expression

buffer_load_dwordx4 v[24:27], v19, s[44:47], 0 offen // (op 179) Load value


s_lshl_b32 s15, s8, 7 // (op 482) 
v_add_u32 v11, v11, s15 // (op 482) 

s_waitcnt vmcnt(1)
ds_write_b128 v15, v[20:23] offset:2048 // (op 167) Store local data 



v_mov_b32 v12, v14 // (op 180) 
s_barrier  // (op 1003) 
v_mov_b32 v16, v12 // (op 180) reg expression




buffer_load_ubyte v17, v16, s[48:51], 0 offen offset: 0 // (op 180) Load value


s_lshl_b32 s15, s12, 2 // (op 546) 
v_add_u32 v14, v14, s15 // (op 546) 

s_waitcnt vmcnt(1)
ds_write_b128 v13, v[24:27] // (op 177) Store local data 

s_waitcnt lgkmcnt(1)
s_barrier  // (op 993) 



s_barrier  // (op 1017) 
ds_read_b64_tr_b4 v[20:21], v3 offset:2048 // (op 169) Transpose load local data 



ds_read_b64_tr_b4 v[22:23], v3 offset:2304 // (op 169) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v18, v17 offset:4096 // (op 178) Store local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 1007) 



ds_read_b64_tr_b8 v[24:25], v0 // (op 171) Transpose load local data 
ds_read_b64_tr_b8 v[26:27], v0 offset:128 // (op 171) Transpose load local data 
ds_read_b64_tr_b8 v[28:29], v0 offset:1024 // (op 171) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v0 offset:1152 // (op 171) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1021) 


ds_read_u8 v12, v4 offset:4096 // (op 173) Load local data 

v_mov_b64 v[16:17], s[24:25] // (op 175) convert

v_mov_b32 v19, v10 // (op 175) 
v_mov_b32 v32, v19 // (op 175) reg expression
v_mov_b32 v34, v32 // (op 175) 
buffer_load_ubyte v35, v34, s[52:55], 0 offen offset: 0 // (op 175) Load value




s_lshl_b32 s15, s24, 2 // (op 610) 
v_add_u32 v10, v10, s15 // (op 610) 

s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:23], v[24:31], a[0:3], v12, v35 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 176) 




s_ashr_i64 s[0:1], s[6:7], 7 // (op 161) 
s_mov_b32 s15, s0 // (op 161) convert

s_cmp_lt_u32 s14, s15 // (op 161) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoopTail_161 // (op 161) Condition: Bottom (jump to RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoopTail_161 if true)
RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoopTail_161:






















s_barrier  // (op 1085) 






v_cmp_eq_f32 vcc, 0.00000, v8 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalFalse_0_beta_123
s_nop 0
v_accvgpr_read v4, a0 // (op 46) Wait state hazard: XDL Write Hazard
v_accvgpr_read v5, a1 // (op 46) 
v_accvgpr_read v6, a2 // (op 46) 
v_accvgpr_read v7, a3 // (op 46) 
v_mul_f32 v10, v9, v4 // (op 46) 
v_mul_f32 v11, v9, v5 // (op 46) 
v_mul_f32 v12, v9, v6 // (op 46) 
v_mul_f32 v13, v9, v7 // (op 46) 

v_lshrrev_b32 v0, 6, v1 // (op 859) 
v_and_b32 v3, 63, v1 // (op 859) 
v_lshrrev_b32 v4, 4, v3 // (op 859) 
v_lshlrev_b32 v5, 2, v4 // (op 859) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 859) 


v_lshl_add_u32 v0, s16, 5, v4 // (op 859) 

v_lshrrev_b32 v4, 2, v3 // (op 859) 

v_and_b32 v3, 3, v4 // (op 859) 

v_and_b32 v4, 3, v1 // (op 859) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 859) 


v_lshl_add_u32 v3, s17, 4, v5 // (op 859) 

v_mul_lo_u32 v4, v3, s28 // (op 859) 

v_add_lshl_u32 v3, v0, v4, 2 // (op 859) 


s_mul_i32 s1, s28, s33 // (op 861) most significant: low of low * high
s_mul_i32 s2, s29, s32 // (op 861) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 861) least significant
s_mul_hi_u32 s3, s28, s32 // (op 861) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 861) most significant: sum
s_add_u32 s1, s1, s3 // (op 861) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 2 // (op 861) 

s_mov_b32 s4, s30 // (op 861) reg expression
s_mov_b32 s5, s31 // (op 861) reg expression
s_mov_b32 s6, s2 // (op 861) reg expression
s_mov_b32 s7, 131072 // (op 861) call()




buffer_store_dwordx4 v[10:13], v3, s[4:7], 0 offen // (op 52) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalBottom_0_beta_123
RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalFalse_0_beta_123:
v_lshrrev_b32 v0, 6, v1 // (op 681) 
v_and_b32 v3, 63, v1 // (op 681) 
v_lshrrev_b32 v4, 4, v3 // (op 681) 
v_lshlrev_b32 v5, 2, v4 // (op 681) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 681) 


v_lshl_add_u32 v0, s16, 5, v4 // (op 681) 

v_lshrrev_b32 v4, 2, v3 // (op 681) 

v_and_b32 v3, 3, v4 // (op 681) 

v_and_b32 v4, 3, v1 // (op 681) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 681) 


v_lshl_add_u32 v3, s17, 4, v5 // (op 681) 

v_mul_lo_u32 v4, v3, s34 // (op 681) 

v_add_lshl_u32 v3, v0, v4, 2 // (op 681) 


s_mul_i32 s1, s34, s33 // (op 683) most significant: low of low * high
s_mul_i32 s2, s35, s32 // (op 683) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 683) least significant
s_mul_hi_u32 s3, s34, s32 // (op 683) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 683) most significant: sum
s_add_u32 s1, s1, s3 // (op 683) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 2 // (op 683) 

s_mov_b32 s8, s36 // (op 683) reg expression
s_mov_b32 s9, s37 // (op 683) reg expression
s_mov_b32 s10, s2 // (op 683) reg expression
s_mov_b32 s11, 131072 // (op 683) call()





buffer_load_dwordx4 v[4:7], v3, s[8:11], 0 offen // (op 135) Load value



v_accvgpr_read v10, a0 // (op 132) 
v_accvgpr_read v11, a1 // (op 132) 
v_accvgpr_read v12, a2 // (op 132) 
v_accvgpr_read v13, a3 // (op 132) 
v_mul_f32 v14, v9, v10 // (op 132) 
v_mul_f32 v15, v9, v11 // (op 132) 
v_mul_f32 v16, v9, v12 // (op 132) 
v_mul_f32 v17, v9, v13 // (op 132) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v8, v4 // (op 136) 
v_mul_f32 v11, v8, v5 // (op 136) 
v_mul_f32 v12, v8, v6 // (op 136) 
v_mul_f32 v13, v8, v7 // (op 136) 

v_add_f32 v4, v10, v14 // (op 133) 
v_add_f32 v5, v11, v15 // (op 133) 
v_add_f32 v6, v12, v16 // (op 133) 
v_add_f32 v7, v13, v17 // (op 133) 


v_lshrrev_b32 v0, 6, v1 // (op 876) 
v_and_b32 v3, 63, v1 // (op 876) 
v_lshrrev_b32 v10, 4, v3 // (op 876) 
v_lshlrev_b32 v11, 2, v10 // (op 876) 

v_lshl_add_u32 v10, v0, 4, v11 // (op 876) 


v_lshl_add_u32 v0, s16, 5, v10 // (op 876) 

v_lshrrev_b32 v10, 2, v3 // (op 876) 

v_and_b32 v3, 3, v10 // (op 876) 

v_and_b32 v10, 3, v1 // (op 876) 
v_lshl_add_u32 v11, v3, 2, v10 // (op 876) 


v_lshl_add_u32 v3, s17, 4, v11 // (op 876) 

v_mul_lo_u32 v10, v3, s28 // (op 876) 

v_add_lshl_u32 v3, v0, v10, 2 // (op 876) 


s_mul_i32 s1, s28, s33 // (op 878) most significant: low of low * high
s_mul_i32 s2, s29, s32 // (op 878) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 878) least significant
s_mul_hi_u32 s3, s28, s32 // (op 878) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 878) most significant: sum
s_add_u32 s1, s1, s3 // (op 878) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 2 // (op 878) 

s_mov_b32 s4, s30 // (op 878) reg expression
s_mov_b32 s5, s31 // (op 878) reg expression
s_mov_b32 s6, s2 // (op 878) reg expression
s_mov_b32 s7, 131072 // (op 878) call()




buffer_store_dwordx4 v[4:7], v3, s[4:7], 0 offen // (op 134) Store value






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalBottom_0_beta_123:



s_endpgm  // End of RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2
.LRR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_end:
.size RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2, .LRR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_end-RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2
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
  - .name: RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2
    .symbol: RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2.kd
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

