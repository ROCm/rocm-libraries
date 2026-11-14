// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2
.p2align 8
.type RR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2,@function
RR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_exec_begin:


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


v_and_b32 v0, 48, v1 // (op 759) 
v_and_b32 v3, 63, v1 // (op 759) 
v_sub_u32 v4, v3, v0 // (op 759) 

v_lshrrev_b32 v3, 1, v4 // (op 759) 
v_add_lshl_u32 v5, v0, v3, 5 // (op 759) 


v_and_b32 v0, 1, v4 // (op 759) 

v_lshlrev_b32 v3, 3, v0 // (op 759) 

v_lshrrev_b32 v0, 7, v1 // (op 759) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 759) 


v_add_u32 v0, v5, v4 // (op 759) 


v_and_b32 v3, 48, v1 // (op 811) 
v_and_b32 v4, 63, v1 // (op 811) 
v_sub_u32 v5, v4, v3 // (op 811) 

v_lshrrev_b32 v4, 1, v5 // (op 811) 
v_add_lshl_u32 v6, v3, v4, 5 // (op 811) 


v_and_b32 v3, 1, v5 // (op 811) 

v_lshlrev_b32 v4, 3, v3 // (op 811) 

v_lshrrev_b32 v3, 6, v1 // (op 811) 
v_and_b32 v5, 1, v3 // (op 811) 

v_lshl_add_u32 v3, v5, 4, v4 // (op 811) 


v_add_u32 v4, v6, v3 // (op 811) 


v_and_b32 v3, 48, v1 // (op 884) 
v_and_b32 v5, 63, v1 // (op 884) 
v_sub_u32 v6, v5, v3 // (op 884) 

v_lshrrev_b32 v5, 1, v6 // (op 884) 
v_add_lshl_u32 v7, v3, v5, 5 // (op 884) 


v_and_b32 v3, 1, v6 // (op 884) 

v_lshlrev_b32 v5, 3, v3 // (op 884) 

v_lshrrev_b32 v3, 7, v1 // (op 884) 
v_lshl_add_u32 v6, v3, 4, v5 // (op 884) 


v_add_u32 v3, v7, v6 // (op 884) 


v_and_b32 v5, 48, v1 // (op 923) 
v_and_b32 v6, 63, v1 // (op 923) 
v_sub_u32 v7, v6, v5 // (op 923) 

v_lshrrev_b32 v6, 1, v7 // (op 923) 
v_add_lshl_u32 v8, v5, v6, 5 // (op 923) 


v_and_b32 v5, 1, v7 // (op 923) 

v_lshlrev_b32 v6, 3, v5 // (op 923) 

v_lshrrev_b32 v5, 6, v1 // (op 923) 
v_and_b32 v7, 1, v5 // (op 923) 

v_lshl_add_u32 v5, v7, 4, v6 // (op 923) 


v_add_u32 v6, v8, v5 // (op 923) 


v_mov_b32 v5, s14 // (op 2) Move value
v_mov_b32 v7, s15 // (op 35) Move value


v_accvgpr_write a0, 0.00000 // (op 55) call()
v_accvgpr_write a1, 0.00000 // (op 55) call()
v_accvgpr_write a2, 0.00000 // (op 55) call()
v_accvgpr_write a3, 0.00000 // (op 55) call()
v_and_b32 v8, -2, v1 // (op 470) 
v_sub_u32 v9, v1, v8 // (op 470) 

v_lshlrev_b32 v8, 4, v9 // (op 470) 

v_lshl_add_u32 v9, s16, 5, v8 // (op 470) 

v_mov_b32 v11, 0 // (op 470) convert
v_mov_b32 v10, v9 // (op 470) convert

v_lshrrev_b32 v8, 1, v1 // (op 470) 
v_mul_lo_u32 v13, v8, s3 // (op 470) most significant: low of low * high
v_mul_hi_u32 v9, v8, s2 // (op 470) most significant: high of low * low
v_mul_lo_u32 v12, v8, s2 // (op 470) least significant: low of low * low
v_add_u32 v13, v13, v9 // (op 470) most significant: sum


v_add_co_u32 v8, s[0:1], v10, v12 // (op 470) least significant half
v_addc_co_u32 v9, s[0:1], v11, v13, s[0:1] // (op 470) most significant half



s_mul_i32 s1, s2, s7 // (op 472) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 472) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 472) least significant
s_mul_hi_u32 s15, s2, s6 // (op 472) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 472) most significant: sum
s_add_u32 s1, s1, s15 // (op 472) most significant: sum


s_mov_b32 s40, s4 // (op 472) reg expression
s_mov_b32 s41, s5 // (op 472) reg expression
s_mov_b32 s42, s0 // (op 472) reg expression
s_mov_b32 s43, 131072 // (op 472) call()

v_and_b32 v10, -2, v1 // (op 489) 
v_sub_u32 v11, v1, v10 // (op 489) 

v_lshlrev_b32 v10, 4, v11 // (op 489) 

v_lshl_add_u32 v11, s16, 5, v10 // (op 489) 

v_mov_b32 v13, 0 // (op 489) convert
v_mov_b32 v12, v11 // (op 489) convert

v_lshrrev_b32 v10, 1, v1 // (op 489) 
v_mul_lo_u32 v15, v10, s3 // (op 489) most significant: low of low * high
v_mul_hi_u32 v11, v10, s2 // (op 489) most significant: high of low * low
v_mul_lo_u32 v14, v10, s2 // (op 489) least significant: low of low * low
v_add_u32 v15, v15, v11 // (op 489) most significant: sum


v_add_co_u32 v10, s[0:1], v12, v14 // (op 489) least significant half
v_addc_co_u32 v11, s[0:1], v13, v15, s[0:1] // (op 489) most significant half



s_mul_i32 s1, s2, s7 // (op 491) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 491) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 491) least significant
s_mul_hi_u32 s15, s2, s6 // (op 491) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 491) most significant: sum
s_add_u32 s1, s1, s15 // (op 491) most significant: sum


s_mov_b32 s40, s4 // (op 491) reg expression
s_mov_b32 s41, s5 // (op 491) reg expression
s_mov_b32 s42, s0 // (op 491) reg expression
s_mov_b32 s43, 131072 // (op 491) call()

v_and_b32 v12, -2, v1 // (op 534) 
v_sub_u32 v13, v1, v12 // (op 534) 

v_lshlrev_b32 v12, 4, v13 // (op 534) 

v_lshl_add_u32 v13, s17, 5, v12 // (op 534) 

v_mov_b32 v15, 0 // (op 534) convert
v_mov_b32 v14, v13 // (op 534) convert

v_lshrrev_b32 v12, 1, v1 // (op 534) 
v_mul_lo_u32 v17, v12, s9 // (op 534) most significant: low of low * high
v_mul_hi_u32 v13, v12, s8 // (op 534) most significant: high of low * low
v_mul_lo_u32 v16, v12, s8 // (op 534) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 534) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 534) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 534) most significant half



s_mul_i32 s1, s8, s7 // (op 536) most significant: low of low * high
s_mul_i32 s14, s9, s6 // (op 536) most significant: low of high * low
s_mul_i32 s0, s8, s6 // (op 536) least significant
s_mul_hi_u32 s15, s8, s6 // (op 536) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 536) most significant: sum
s_add_u32 s1, s1, s15 // (op 536) most significant: sum


s_mov_b32 s44, s10 // (op 536) reg expression
s_mov_b32 s45, s11 // (op 536) reg expression
s_mov_b32 s46, s0 // (op 536) reg expression
s_mov_b32 s47, 131072 // (op 536) call()

v_and_b32 v12, -2, v1 // (op 553) 
v_sub_u32 v13, v1, v12 // (op 553) 

v_lshlrev_b32 v12, 4, v13 // (op 553) 

v_lshl_add_u32 v13, s17, 5, v12 // (op 553) 

v_mov_b32 v15, 0 // (op 553) convert
v_mov_b32 v14, v13 // (op 553) convert

v_lshrrev_b32 v12, 1, v1 // (op 553) 
v_mul_lo_u32 v19, v12, s9 // (op 553) most significant: low of low * high
v_mul_hi_u32 v13, v12, s8 // (op 553) most significant: high of low * low
v_mul_lo_u32 v18, v12, s8 // (op 553) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 553) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v14 // (op 553) least significant half
v_addc_co_u32 v19, s[0:1], v19, v15, s[0:1] // (op 553) most significant half



s_mul_i32 s1, s8, s7 // (op 555) most significant: low of low * high
s_mul_i32 s14, s9, s6 // (op 555) most significant: low of high * low
s_mul_i32 s0, s8, s6 // (op 555) least significant
s_mul_hi_u32 s15, s8, s6 // (op 555) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 555) most significant: sum
s_add_u32 s1, s1, s15 // (op 555) most significant: sum


s_mov_b32 s44, s10 // (op 555) reg expression
s_mov_b32 s45, s11 // (op 555) reg expression
s_mov_b32 s46, s0 // (op 555) reg expression
s_mov_b32 s47, 131072 // (op 555) call()

v_lshrrev_b32 v12, 7, v1 // (op 598) 
v_and_b32 v13, 15, v1 // (op 598) 
v_lshl_add_u32 v14, v12, 4, v13 // (op 598) 


v_lshl_add_u32 v15, s16, 5, v14 // (op 598) 

v_and_b32 v12, 63, v1 // (op 598) 
v_lshrrev_b32 v13, 4, v12 // (op 598) 

v_mul_lo_u32 v12, v13, s12 // (op 598) 

v_add_u32 v13, v15, v12 // (op 598) 



s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 600) most significant: low of low * high
s_mul_i32 s14, s13, s22 // (op 600) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 600) least significant
s_mul_hi_u32 s15, s12, s22 // (op 600) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 600) most significant: sum
s_add_u32 s1, s1, s15 // (op 600) most significant: sum


s_mov_b32 s48, s20 // (op 600) reg expression
s_mov_b32 s49, s21 // (op 600) reg expression
s_mov_b32 s50, s0 // (op 600) reg expression
s_mov_b32 s51, 131072 // (op 600) call()

v_lshrrev_b32 v12, 7, v1 // (op 617) 
v_and_b32 v14, 15, v1 // (op 617) 
v_lshl_add_u32 v15, v12, 4, v14 // (op 617) 


v_lshl_add_u32 v12, s16, 5, v15 // (op 617) 

v_and_b32 v14, 63, v1 // (op 617) 
v_lshrrev_b32 v15, 4, v14 // (op 617) 

v_mul_lo_u32 v14, v15, s12 // (op 617) 

v_add_u32 v15, v12, v14 // (op 617) 


s_mul_i32 s1, s12, s23 // (op 619) most significant: low of low * high
s_mul_i32 s14, s13, s22 // (op 619) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 619) least significant
s_mul_hi_u32 s15, s12, s22 // (op 619) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 619) most significant: sum
s_add_u32 s1, s1, s15 // (op 619) most significant: sum


s_mov_b32 s48, s20 // (op 619) reg expression
s_mov_b32 s49, s21 // (op 619) reg expression
s_mov_b32 s50, s0 // (op 619) reg expression
s_mov_b32 s51, 131072 // (op 619) call()

v_lshrrev_b32 v12, 6, v1 // (op 662) 
v_and_b32 v14, 1, v12 // (op 662) 

v_and_b32 v12, 15, v1 // (op 662) 
v_lshl_add_u32 v20, v14, 4, v12 // (op 662) 


v_lshl_add_u32 v12, s17, 5, v20 // (op 662) 

v_and_b32 v14, 63, v1 // (op 662) 
v_lshrrev_b32 v20, 4, v14 // (op 662) 

v_mul_lo_u32 v14, v20, s24 // (op 662) 
v_add_u32 v14, v14, v12 // (op 662) 


s_mul_i32 s1, s24, s23 // (op 664) most significant: low of low * high
s_mul_i32 s14, s25, s22 // (op 664) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 664) least significant
s_mul_hi_u32 s15, s24, s22 // (op 664) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 664) most significant: sum
s_add_u32 s1, s1, s15 // (op 664) most significant: sum


s_mov_b32 s52, s26 // (op 664) reg expression
s_mov_b32 s53, s27 // (op 664) reg expression
s_mov_b32 s54, s0 // (op 664) reg expression
s_mov_b32 s55, 131072 // (op 664) call()

v_lshrrev_b32 v12, 6, v1 // (op 681) 
v_and_b32 v20, 1, v12 // (op 681) 

v_and_b32 v12, 15, v1 // (op 681) 
v_lshl_add_u32 v21, v20, 4, v12 // (op 681) 


v_lshl_add_u32 v12, s17, 5, v21 // (op 681) 

v_and_b32 v20, 63, v1 // (op 681) 
v_lshrrev_b32 v21, 4, v20 // (op 681) 

v_mul_lo_u32 v20, v21, s24 // (op 681) 
v_add_u32 v20, v20, v12 // (op 681) 


s_mul_i32 s1, s24, s23 // (op 683) most significant: low of low * high
s_mul_i32 s14, s25, s22 // (op 683) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 683) least significant
s_mul_hi_u32 s15, s24, s22 // (op 683) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 683) most significant: sum
s_add_u32 s1, s1, s15 // (op 683) most significant: sum


s_mov_b32 s52, s26 // (op 683) reg expression
s_mov_b32 s53, s27 // (op 683) reg expression
s_mov_b32 s54, s0 // (op 683) reg expression
s_mov_b32 s55, 131072 // (op 683) call()


v_mov_b64 v[22:23], s[12:13] // (op 308) convert

v_mov_b32 v12, v13 // (op 308) reg expression
v_mov_b32 v21, v12 // (op 308) 
buffer_load_ubyte v24, v21, s[48:51], 0 offen offset: 0 // (op 308) Load value




v_mov_b64 v[22:23], s[24:25] // (op 313) convert

v_mov_b32 v12, v14 // (op 313) reg expression
v_mov_b32 v21, v12 // (op 313) 
buffer_load_ubyte v25, v21, s[52:55], 0 offen offset: 0 // (op 313) Load value



v_and_b32 v12, -2, v1 // (op 799) 
v_sub_u32 v21, v1, v12 // (op 799) 

v_lshlrev_b32 v12, 4, v21 // (op 799) 

v_lshrrev_b32 v21, 1, v1 // (op 799) 
v_lshl_add_u32 v22, v21, 5, v12 // (op 799) 


s_nop 0
v_readfirstlane_b32 s0, v22 // (op 799) Wait state hazard: VALU Write Readlane Hazard



v_mov_b64 v[22:23], v[16:17] // (op 419) reg expression

s_nop 0
s_mov_b32 m0, s0 // (op 419) reg expression
buffer_load_dwordx4 v22, s[44:47], 0 offen offset: 0 lds // (op 419) Load value direct to lds


v_and_b32 v12, -2, v1 // (op 747) 
v_sub_u32 v21, v1, v12 // (op 747) 

v_lshlrev_b32 v12, 4, v21 // (op 747) 

v_lshrrev_b32 v21, 1, v1 // (op 747) 
v_lshl_add_u32 v22, v21, 5, v12 // (op 747) 


s_nop 0
v_readfirstlane_b32 s0, v22 // (op 747) Wait state hazard: VALU Write Readlane Hazard



v_mov_b64 v[22:23], v[8:9] // (op 424) reg expression
s_nop 0
s_add_u32 s1, 4096, s0 // (op 424) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 424) reg expression
buffer_load_dwordx4 v22, s[40:43], 0 offen offset: 0 lds // (op 424) Load value direct to lds




v_mov_b64 v[22:23], s[12:13] // (op 324) convert

s_lshl_b32 s0, s12, 2 // (op 324) 
v_add_u32 v12, v13, s0 // (op 324) 

v_mov_b32 v21, v12 // (op 324) 
buffer_load_ubyte v26, v21, s[48:51], 0 offen offset: 0 // (op 324) Load value





v_mov_b64 v[12:13], s[24:25] // (op 329) convert

s_lshl_b32 s0, s24, 2 // (op 329) 
v_add_u32 v21, v14, s0 // (op 329) 

v_mov_b32 v22, v21 // (op 329) 
buffer_load_ubyte v23, v22, s[52:55], 0 offen offset: 0 // (op 329) Load value




v_and_b32 v12, -2, v1 // (op 911) 
v_sub_u32 v13, v1, v12 // (op 911) 

v_lshlrev_b32 v12, 4, v13 // (op 911) 

v_lshrrev_b32 v13, 1, v1 // (op 911) 
v_lshl_add_u32 v14, v13, 5, v12 // (op 911) 


s_nop 0
v_readfirstlane_b32 s0, v14 // (op 911) Wait state hazard: VALU Write Readlane Hazard



s_lshl_b64 s[14:15], s[8:9], 7 // (op 429) 
v_mov_b32 v12, s14 // (op 429) 
v_mov_b32 v13, s15 // (op 429) 
v_add_co_u32 v28, s[18:19], v16, v12 // (op 429) least significant half
v_addc_co_u32 v29, s[18:19], v17, v13, s[18:19] // (op 429) most significant half




s_add_u32 s1, 8192, s0 // (op 429) 

s_mov_b32 m0, s1 // (op 429) reg expression
buffer_load_dwordx4 v28, s[44:47], 0 offen offset: 0 lds // (op 429) Load value direct to lds




v_and_b32 v12, -2, v1 // (op 872) 
v_sub_u32 v13, v1, v12 // (op 872) 

v_lshlrev_b32 v12, 4, v13 // (op 872) 

v_lshrrev_b32 v13, 1, v1 // (op 872) 
v_lshl_add_u32 v14, v13, 5, v12 // (op 872) 


s_nop 0
v_readfirstlane_b32 s0, v14 // (op 872) Wait state hazard: VALU Write Readlane Hazard



s_lshl_b64 s[14:15], s[2:3], 7 // (op 431) 
v_mov_b32 v12, s14 // (op 431) 
v_mov_b32 v13, s15 // (op 431) 
v_add_co_u32 v16, s[18:19], v8, v12 // (op 431) least significant half
v_addc_co_u32 v17, s[18:19], v9, v13, s[18:19] // (op 431) most significant half




s_add_u32 s1, 12288, s0 // (op 431) 

s_mov_b32 m0, s1 // (op 431) reg expression
buffer_load_dwordx4 v16, s[40:43], 0 offen offset: 0 lds // (op 431) Load value direct to lds




s_waitcnt vmcnt(5)
s_barrier  // (op 968) 



ds_read_b64_tr_b8 v[28:29], v4 // (op 354) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v4 offset:256 // (op 354) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v4 offset:2048 // (op 354) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v4 offset:2304 // (op 354) Transpose load local data 
s_waitcnt vmcnt(4)
s_barrier  // (op 952) 



ds_read_b64_tr_b8 v[36:37], v0 offset:4096 // (op 351) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v0 offset:4352 // (op 351) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v0 offset:6144 // (op 351) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v0 offset:6400 // (op 351) Transpose load local data 
v_and_b32 v27, -2, v1 // (op 719) 
v_sub_u32 v8, v1, v27 // (op 719) 

v_lshlrev_b32 v9, 4, v8 // (op 719) 

v_lshrrev_b32 v8, 1, v1 // (op 719) 
v_lshl_add_u32 v27, v8, 5, v9 // (op 719) 


s_nop 0
v_readfirstlane_b32 s0, v27 // (op 719) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v27, -2, v1 // (op 771) 
v_sub_u32 v8, v1, v27 // (op 771) 

v_lshlrev_b32 v9, 4, v8 // (op 771) 

v_lshrrev_b32 v8, 1, v1 // (op 771) 
v_lshl_add_u32 v27, v8, 5, v9 // (op 771) 


s_nop 0
v_readfirstlane_b32 s1, v27 // (op 771) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v27, -2, v1 // (op 857) 
v_sub_u32 v8, v1, v27 // (op 857) 

v_lshlrev_b32 v9, 4, v8 // (op 857) 

v_lshrrev_b32 v8, 1, v1 // (op 857) 
v_lshl_add_u32 v27, v8, 5, v9 // (op 857) 


s_nop 0
v_readfirstlane_b32 s14, v27 // (op 857) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v27, -2, v1 // (op 896) 
v_sub_u32 v8, v1, v27 // (op 896) 

v_lshlrev_b32 v9, 4, v8 // (op 896) 

v_lshrrev_b32 v8, 1, v1 // (op 896) 
v_lshl_add_u32 v27, v8, 5, v9 // (op 896) 


s_nop 0
v_readfirstlane_b32 s15, v27 // (op 896) Wait state hazard: VALU Write Readlane Hazard

s_mov_b32 s18, 0 // (op 51) call()
s_ashr_i64 s[38:39], s[6:7], 7 // (op 50) 
s_mov_b32 s19, s38 // (op 50) convert

s_and_b32 s38, s19, 4294967294 // (op 50) 

s_cmp_lt_u32 s18, s38 // (op 50) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoop_50 // (op 50) Condition: Top (jump to RR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoop_50 if false)
RR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoop_50:
s_add_u32 s18, s18, 2 // (op 52) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:43], v[28:35], a[0:3], v24, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 76) 

v_mov_b64 v[8:9], s[12:13] // (op 70) convert

v_mov_b32 v27, v15 // (op 70) 
s_lshl_b32 s19, s12, 3 // (op 70) 
v_add_u32 v12, v27, s19 // (op 70) 

v_mov_b32 v13, v12 // (op 70) 
buffer_load_ubyte v24, v13, s[48:51], 0 offen offset: 0 // (op 70) Load value




v_mov_b64 v[8:9], s[24:25] // (op 74) convert

v_mov_b32 v12, v20 // (op 74) 
s_lshl_b32 s19, s24, 3 // (op 74) 
v_add_u32 v13, v12, s19 // (op 74) 

v_mov_b32 v14, v13 // (op 74) 
buffer_load_ubyte v25, v14, s[52:55], 0 offen offset: 0 // (op 74) Load value



s_barrier  // (op 971) 


v_mov_b64 v[8:9], v[18:19] // (op 392) 
s_lshl_b64 s[38:39], s[8:9], 7 // (op 392) 
s_lshl_b64 s[56:57], s[38:39], 1 // (op 392) 

v_mov_b32 v13, s56 // (op 392) 
v_mov_b32 v14, s57 // (op 392) 
v_add_co_u32 v16, s[38:39], v8, v13 // (op 392) least significant half
v_addc_co_u32 v17, s[38:39], v9, v14, s[38:39] // (op 392) most significant half





s_mov_b32 m0, s1 // (op 392) reg expression
buffer_load_dwordx4 v16, s[44:47], 0 offen offset: 0 lds // (op 392) Load value direct to lds

s_barrier  // (op 955) 


v_mov_b64 v[16:17], v[10:11] // (op 387) 
s_lshl_b64 s[38:39], s[2:3], 7 // (op 387) 
s_lshl_b64 s[56:57], s[38:39], 1 // (op 387) 

v_mov_b32 v13, s56 // (op 387) 
v_mov_b32 v14, s57 // (op 387) 
v_add_co_u32 v44, s[38:39], v16, v13 // (op 387) least significant half
v_addc_co_u32 v45, s[38:39], v17, v14, s[38:39] // (op 387) most significant half




s_add_u32 s19, 4096, s0 // (op 387) 

s_mov_b32 m0, s19 // (op 387) reg expression
buffer_load_dwordx4 v44, s[40:43], 0 offen offset: 0 lds // (op 387) Load value direct to lds


s_waitcnt vmcnt(5)
s_barrier  // (op 990) 



ds_read_b64_tr_b8 v[28:29], v6 offset:8192 // (op 212) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v6 offset:8448 // (op 212) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v6 offset:10240 // (op 212) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v6 offset:10496 // (op 212) Transpose load local data 
s_waitcnt vmcnt(4)
s_barrier  // (op 984) 



ds_read_b64_tr_b8 v[36:37], v3 offset:12288 // (op 210) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v3 offset:12544 // (op 210) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v3 offset:14336 // (op 210) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v3 offset:14592 // (op 210) Transpose load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:43], v[28:35], a[0:3], v26, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 217) 

v_mov_b64 v[44:45], s[12:13] // (op 214) convert

s_lshl_b32 s19, s12, 2 // (op 214) 
s_mul_i32 s38, 3, s19 // (op 214) 

v_add_u32 v13, v27, s38 // (op 214) 

v_mov_b32 v14, v13 // (op 214) 
buffer_load_ubyte v26, v14, s[48:51], 0 offen offset: 0 // (op 214) Load value





s_lshl_b32 s19, s12, 3 // (op 609) 
v_add_u32 v15, v15, s19 // (op 609) 

v_mov_b64 v[44:45], s[24:25] // (op 216) convert

s_lshl_b32 s19, s24, 2 // (op 216) 
s_mul_i32 s38, 3, s19 // (op 216) 

v_add_u32 v27, v12, s38 // (op 216) 

v_mov_b32 v13, v27 // (op 216) 
buffer_load_ubyte v23, v13, s[52:55], 0 offen offset: 0 // (op 216) Load value




s_barrier  // (op 993) 
s_lshl_b32 s19, s24, 3 // (op 673) 


v_add_u32 v20, v20, s19 // (op 673) 

s_lshl_b64 s[38:39], s[8:9], 7 // (op 414) 
s_mul_i32 s57, 3, s39 // (op 414) most significant: low of low * high
s_mul_i32 s56, 3, s38 // (op 414) least significant
s_mul_hi_u32 s19, 3, s38 // (op 414) most significant: high of low * low
s_add_u32 s57, s57, s19 // (op 414) most significant: sum


v_mov_b32 v27, s56 // (op 414) 
v_mov_b32 v12, s57 // (op 414) 
v_add_co_u32 v44, s[38:39], v8, v27 // (op 414) least significant half
v_addc_co_u32 v45, s[38:39], v9, v12, s[38:39] // (op 414) most significant half




s_add_u32 s19, 8192, s15 // (op 414) 

s_mov_b32 m0, s19 // (op 414) reg expression
buffer_load_dwordx4 v44, s[44:47], 0 offen offset: 0 lds // (op 414) Load value direct to lds



s_barrier  // (op 987) 
s_lshl_b64 s[38:39], s[8:9], 7 // (op 545) 


s_lshl_b64 s[56:57], s[38:39], 1 // (op 545) 

v_mov_b32 v27, s56 // (op 545) 
v_mov_b32 v8, s57 // (op 545) 
s_lshl_b64 s[38:39], s[2:3], 7 // (op 409) 
s_mul_i32 s59, 3, s39 // (op 409) most significant: low of low * high
s_mul_i32 s58, 3, s38 // (op 409) least significant
s_mul_hi_u32 s19, 3, s38 // (op 409) most significant: high of low * low
s_add_u32 s59, s59, s19 // (op 409) most significant: sum


v_mov_b32 v9, s58 // (op 409) 
v_mov_b32 v12, s59 // (op 409) 
v_add_co_u32 v18, s[38:39], v18, v27 // (op 545) least significant half
v_addc_co_u32 v19, s[38:39], v19, v8, s[38:39] // (op 545) most significant half




v_add_co_u32 v44, s[38:39], v16, v9 // (op 409) least significant half
v_addc_co_u32 v45, s[38:39], v17, v12, s[38:39] // (op 409) most significant half




s_add_u32 s19, 12288, s14 // (op 409) 

s_mov_b32 m0, s19 // (op 409) reg expression
buffer_load_dwordx4 v44, s[40:43], 0 offen offset: 0 lds // (op 409) Load value direct to lds



s_lshl_b64 s[38:39], s[2:3], 7 // (op 481) 
s_lshl_b64 s[56:57], s[38:39], 1 // (op 481) 

v_mov_b32 v27, s56 // (op 481) 
v_mov_b32 v8, s57 // (op 481) 
v_add_co_u32 v10, s[38:39], v10, v27 // (op 481) least significant half
v_addc_co_u32 v11, s[38:39], v11, v8, s[38:39] // (op 481) most significant half




s_waitcnt vmcnt(5)
s_barrier  // (op 977) 



ds_read_b64_tr_b8 v[28:29], v4 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v4 offset:256 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v4 offset:2048 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v4 offset:2304 // (op 65) Transpose load local data 
s_waitcnt vmcnt(4)
s_barrier  // (op 961) 



ds_read_b64_tr_b8 v[36:37], v0 offset:4096 // (op 60) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v0 offset:4352 // (op 60) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v0 offset:6144 // (op 60) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v0 offset:6400 // (op 60) Transpose load local data 
s_ashr_i64 s[38:39], s[6:7], 7 // (op 50) 
s_mov_b32 s19, s38 // (op 50) convert

s_and_b32 s38, s19, 4294967294 // (op 50) 

s_cmp_lt_u32 s18, s38 // (op 50) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoop_50 // (op 50) Condition: Bottom (jump to RR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoop_50 if true)
RR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoop_50:











s_barrier  // (op 1000) 
v_and_b32 v27, -2, v1 // (op 444) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 444) 
v_sub_u32 v8, v1, v27 // (op 444) 

s_lshr_b32 s14, s0, 1 // (op 444) 

v_lshlrev_b32 v27, 4, v8 // (op 444) 

v_lshl_add_u32 v8, s16, 5, v27 // (op 444) 

v_mov_b32 v25, 0 // (op 444) convert
v_mov_b32 v24, v8 // (op 444) convert

v_lshrrev_b32 v27, 1, v1 // (op 444) 
v_lshl_add_u32 v8, s14, 8, v27 // (op 444) 


v_mul_lo_u32 v11, v8, s3 // (op 444) most significant: low of low * high
v_mul_hi_u32 v9, v8, s2 // (op 444) most significant: high of low * low
v_mul_lo_u32 v10, v8, s2 // (op 444) least significant: low of low * low
v_add_u32 v11, v11, v9 // (op 444) most significant: sum


v_add_co_u32 v8, s[0:1], v24, v10 // (op 444) least significant half
v_addc_co_u32 v9, s[0:1], v25, v11, s[0:1] // (op 444) most significant half



s_mul_i32 s1, s2, s7 // (op 446) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 446) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 446) least significant
s_mul_hi_u32 s15, s2, s6 // (op 446) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 446) most significant: sum
s_add_u32 s1, s1, s15 // (op 446) most significant: sum


s_mov_b32 s40, s4 // (op 446) reg expression
s_mov_b32 s41, s5 // (op 446) reg expression
s_mov_b32 s42, s0 // (op 446) reg expression
s_mov_b32 s43, 131072 // (op 446) call()

v_lshrrev_b32 v27, 1, v1 // (op 511) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 511) 
v_and_b32 v10, -2, v1 // (op 511) 
s_lshr_b32 s14, s0, 1 // (op 511) 

v_sub_u32 v11, v1, v10 // (op 511) 

v_lshl_add_u32 v10, s14, 8, v27 // (op 511) 


v_lshlrev_b32 v27, 4, v11 // (op 511) 

v_lshl_add_u32 v11, s17, 5, v27 // (op 511) 

v_mov_b32 v25, 0 // (op 511) convert
v_mov_b32 v24, v11 // (op 511) convert

v_mul_lo_u32 v13, v10, s9 // (op 511) most significant: low of low * high
v_mul_hi_u32 v11, v10, s8 // (op 511) most significant: high of low * low
v_mul_lo_u32 v12, v10, s8 // (op 511) least significant: low of low * low
v_add_u32 v13, v13, v11 // (op 511) most significant: sum

v_add_co_u32 v12, s[0:1], v12, v24 // (op 511) least significant half
v_addc_co_u32 v13, s[0:1], v13, v25, s[0:1] // (op 511) most significant half



s_mul_i32 s1, s8, s7 // (op 513) most significant: low of low * high
s_mul_i32 s14, s9, s6 // (op 513) most significant: low of high * low
s_mul_i32 s0, s8, s6 // (op 513) least significant
s_mul_hi_u32 s15, s8, s6 // (op 513) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 513) most significant: sum
s_add_u32 s1, s1, s15 // (op 513) most significant: sum


s_mov_b32 s44, s10 // (op 513) reg expression
s_mov_b32 s45, s11 // (op 513) reg expression
s_mov_b32 s46, s0 // (op 513) reg expression
s_mov_b32 s47, 131072 // (op 513) call()

v_lshrrev_b32 v27, 7, v1 // (op 575) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 575) 
v_and_b32 v10, 15, v1 // (op 575) 
s_lshr_b32 s14, s0, 1 // (op 575) 

v_lshl_add_u32 v11, v27, 4, v10 // (op 575) 


v_lshl_add_u32 v10, s16, 5, v11 // (op 575) 

v_and_b32 v11, 63, v1 // (op 575) 
v_lshrrev_b32 v27, 4, v11 // (op 575) 

v_lshl_add_u32 v11, s14, 3, v27 // (op 575) 


v_mul_lo_u32 v27, v11, s12 // (op 575) 

v_add_u32 v11, v10, v27 // (op 575) 


s_mul_i32 s1, s12, s23 // (op 577) most significant: low of low * high
s_mul_i32 s14, s13, s22 // (op 577) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 577) least significant
s_mul_hi_u32 s15, s12, s22 // (op 577) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 577) most significant: sum
s_add_u32 s1, s1, s15 // (op 577) most significant: sum


s_mov_b32 s48, s20 // (op 577) reg expression
s_mov_b32 s49, s21 // (op 577) reg expression
s_mov_b32 s50, s0 // (op 577) reg expression
s_mov_b32 s51, 131072 // (op 577) call()

v_and_b32 v10, 63, v1 // (op 639) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 639) 
v_lshrrev_b32 v27, 4, v10 // (op 639) 

s_lshr_b32 s14, s0, 1 // (op 639) 

v_lshrrev_b32 v10, 6, v1 // (op 639) 
v_lshl_add_u32 v14, s14, 3, v27 // (op 639) 


v_and_b32 v27, 1, v10 // (op 639) 

v_and_b32 v10, 15, v1 // (op 639) 
v_lshl_add_u32 v15, v27, 4, v10 // (op 639) 


v_lshl_add_u32 v10, s17, 5, v15 // (op 639) 

v_mul_lo_u32 v27, v14, s24 // (op 639) 
v_add_u32 v27, v27, v10 // (op 639) 


s_mul_i32 s1, s24, s23 // (op 641) most significant: low of low * high
s_mul_i32 s14, s25, s22 // (op 641) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 641) least significant
s_mul_hi_u32 s15, s24, s22 // (op 641) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 641) most significant: sum
s_add_u32 s1, s1, s15 // (op 641) most significant: sum


s_mov_b32 s52, s26 // (op 641) reg expression
s_mov_b32 s53, s27 // (op 641) reg expression
s_mov_b32 s54, s0 // (op 641) reg expression
s_mov_b32 s55, 131072 // (op 641) call()

v_and_b32 v10, -2, v1 // (op 732) 
v_sub_u32 v14, v1, v10 // (op 732) 

v_lshlrev_b32 v10, 4, v14 // (op 732) 

v_lshrrev_b32 v14, 1, v1 // (op 732) 
v_lshl_add_u32 v15, v14, 5, v10 // (op 732) 


s_nop 0
v_readfirstlane_b32 s0, v15 // (op 732) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v10, -2, v1 // (op 784) 
v_sub_u32 v14, v1, v10 // (op 784) 

v_lshlrev_b32 v10, 4, v14 // (op 784) 

v_lshrrev_b32 v14, 1, v1 // (op 784) 
v_lshl_add_u32 v15, v14, 5, v10 // (op 784) 


s_nop 0
v_readfirstlane_b32 s1, v15 // (op 784) Wait state hazard: VALU Write Readlane Hazard

s_ashr_i64 s[14:15], s[6:7], 7 // (op 153) 
s_and_b32 s18, s14, 4294967294 // (op 153) 

s_ashr_i64 s[14:15], s[6:7], 7 // (op 152) 
s_mov_b32 s19, s14 // (op 152) convert

s_cmp_lt_u32 s18, s19 // (op 152) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoopTail_152 // (op 152) Condition: Top (jump to RR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoopTail_152 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoopTail_152:
s_barrier  // (op 958) 
s_add_u32 s18, s18, 1 // (op 154) 


v_mov_b64 v[14:15], v[8:9] // (op 397) 
v_mov_b64 v[16:17], v[14:15] // (op 397) reg expression
s_add_u32 s19, 4096, s0 // (op 397) 

s_mov_b32 m0, s19 // (op 397) reg expression
buffer_load_dwordx4 v16, s[40:43], 0 offen offset: 0 lds // (op 397) Load value direct to lds



s_lshl_b64 s[14:15], s[2:3], 7 // (op 434) 
v_mov_b32 v10, s14 // (op 434) 
v_mov_b32 v14, s15 // (op 434) 
v_add_co_u32 v8, s[38:39], v8, v10 // (op 434) least significant half
v_addc_co_u32 v9, s[38:39], v9, v14, s[38:39] // (op 434) most significant half




s_barrier  // (op 974) 


v_mov_b64 v[14:15], v[12:13] // (op 403) 
v_mov_b64 v[16:17], v[14:15] // (op 403) reg expression

s_mov_b32 m0, s1 // (op 403) reg expression
buffer_load_dwordx4 v16, s[44:47], 0 offen offset: 0 lds // (op 403) Load value direct to lds


s_lshl_b64 s[14:15], s[8:9], 7 // (op 503) 
v_mov_b32 v10, s14 // (op 503) 
v_mov_b32 v14, s15 // (op 503) 
v_add_co_u32 v12, s[38:39], v12, v10 // (op 503) least significant half
v_addc_co_u32 v13, s[38:39], v13, v14, s[38:39] // (op 503) most significant half




s_waitcnt vmcnt(1)
s_barrier  // (op 964) 



ds_read_b64_tr_b8 v[14:15], v0 offset:4096 // (op 160) Transpose load local data 
ds_read_b64_tr_b8 v[16:17], v0 offset:4352 // (op 160) Transpose load local data 
ds_read_b64_tr_b8 v[18:19], v0 offset:6144 // (op 160) Transpose load local data 
ds_read_b64_tr_b8 v[20:21], v0 offset:6400 // (op 160) Transpose load local data 
s_waitcnt vmcnt(0)
s_barrier  // (op 980) 



ds_read_b64_tr_b8 v[28:29], v4 // (op 162) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v4 offset:256 // (op 162) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v4 offset:2048 // (op 162) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v4 offset:2304 // (op 162) Transpose load local data 

v_mov_b64 v[22:23], s[12:13] // (op 164) convert

v_mov_b32 v10, v11 // (op 164) 
v_mov_b32 v24, v10 // (op 164) reg expression
v_mov_b32 v25, v24 // (op 164) 
buffer_load_ubyte v26, v25, s[48:51], 0 offen offset: 0 // (op 164) Load value




s_lshl_b32 s19, s12, 2 // (op 567) 
v_add_u32 v11, v11, s19 // (op 567) 


v_mov_b64 v[22:23], s[24:25] // (op 166) convert

v_mov_b32 v10, v27 // (op 166) 
v_mov_b32 v24, v10 // (op 166) reg expression
v_mov_b32 v25, v24 // (op 166) 
buffer_load_ubyte v36, v25, s[52:55], 0 offen offset: 0 // (op 166) Load value




s_lshl_b32 s19, s24, 2 // (op 631) 
v_add_u32 v27, v27, s19 // (op 631) 

s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[14:21], v[28:35], a[0:3], v26, v36 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 167) 




s_ashr_i64 s[14:15], s[6:7], 7 // (op 152) 
s_mov_b32 s19, s14 // (op 152) convert

s_cmp_lt_u32 s18, s19 // (op 152) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoopTail_152 // (op 152) Condition: Bottom (jump to RR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoopTail_152 if true)
RR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoopTail_152:





















s_barrier  // (op 1022) 




v_cmp_eq_f32 vcc, 0.00000, v5 // (op 114) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalFalse_0_beta_114 // (op 114) Condition: False, jump to RR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalFalse_0_beta_114
s_nop 0
v_accvgpr_read v8, a0 // (op 42) Wait state hazard: XDL Write Hazard
v_accvgpr_read v9, a1 // (op 42) 
v_accvgpr_read v10, a2 // (op 42) 
v_accvgpr_read v11, a3 // (op 42) 
v_mul_f32 v12, v7, v8 // (op 42) 
v_mul_f32 v13, v7, v9 // (op 42) 
v_mul_f32 v14, v7, v10 // (op 42) 
v_mul_f32 v15, v7, v11 // (op 42) 

v_lshrrev_b32 v0, 7, v1 // (op 825) 
v_and_b32 v6, 63, v1 // (op 825) 
v_lshrrev_b32 v3, 4, v6 // (op 825) 
v_lshlrev_b32 v4, 2, v3 // (op 825) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 825) 


v_lshl_add_u32 v0, s16, 5, v3 // (op 825) 

v_lshrrev_b32 v3, 2, v6 // (op 825) 

v_and_b32 v4, 3, v3 // (op 825) 

v_and_b32 v3, 3, v1 // (op 825) 
v_lshl_add_u32 v6, v4, 2, v3 // (op 825) 


v_lshrrev_b32 v3, 6, v1 // (op 825) 
v_and_b32 v4, 1, v3 // (op 825) 

v_lshl_add_u32 v3, v4, 4, v6 // (op 825) 


v_lshl_add_u32 v4, s17, 5, v3 // (op 825) 

v_mul_lo_u32 v3, v4, s28 // (op 825) 

v_add_lshl_u32 v4, v0, v3, 2 // (op 825) 


s_mul_i32 s1, s28, s33 // (op 827) most significant: low of low * high
s_mul_i32 s2, s29, s32 // (op 827) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 827) least significant
s_mul_hi_u32 s3, s28, s32 // (op 827) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 827) most significant: sum
s_add_u32 s1, s1, s3 // (op 827) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 2 // (op 827) 

s_mov_b32 s4, s30 // (op 827) reg expression
s_mov_b32 s5, s31 // (op 827) reg expression
s_mov_b32 s6, s2 // (op 827) reg expression
s_mov_b32 s7, 131072 // (op 827) call()




buffer_store_dwordx4 v[12:15], v4, s[4:7], 0 offen // (op 48) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalBottom_0_beta_114 // (op 114) Condition: Done, jump to RR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalBottom_0_beta_114
RR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalFalse_0_beta_114:
v_lshrrev_b32 v0, 7, v1 // (op 702) 
v_and_b32 v6, 63, v1 // (op 702) 
v_lshrrev_b32 v3, 4, v6 // (op 702) 
v_lshlrev_b32 v4, 2, v3 // (op 702) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 702) 


v_lshl_add_u32 v0, s16, 5, v3 // (op 702) 

v_lshrrev_b32 v3, 2, v6 // (op 702) 

v_and_b32 v4, 3, v3 // (op 702) 

v_and_b32 v3, 3, v1 // (op 702) 
v_lshl_add_u32 v6, v4, 2, v3 // (op 702) 


v_lshrrev_b32 v3, 6, v1 // (op 702) 
v_and_b32 v4, 1, v3 // (op 702) 

v_lshl_add_u32 v3, v4, 4, v6 // (op 702) 


v_lshl_add_u32 v4, s17, 5, v3 // (op 702) 

v_mul_lo_u32 v3, v4, s34 // (op 702) 

v_add_lshl_u32 v4, v0, v3, 2 // (op 702) 


s_mul_i32 s1, s34, s33 // (op 704) most significant: low of low * high
s_mul_i32 s2, s35, s32 // (op 704) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 704) least significant
s_mul_hi_u32 s3, s34, s32 // (op 704) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 704) most significant: sum
s_add_u32 s1, s1, s3 // (op 704) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 2 // (op 704) 

s_mov_b32 s8, s36 // (op 704) reg expression
s_mov_b32 s9, s37 // (op 704) reg expression
s_mov_b32 s10, s2 // (op 704) reg expression
s_mov_b32 s11, 131072 // (op 704) call()





buffer_load_dwordx4 v[8:11], v4, s[8:11], 0 offen // (op 126) Load value



v_accvgpr_read v12, a0 // (op 123) 
v_accvgpr_read v13, a1 // (op 123) 
v_accvgpr_read v14, a2 // (op 123) 
v_accvgpr_read v15, a3 // (op 123) 
v_mul_f32 v16, v7, v12 // (op 123) 
v_mul_f32 v17, v7, v13 // (op 123) 
v_mul_f32 v18, v7, v14 // (op 123) 
v_mul_f32 v19, v7, v15 // (op 123) 

s_waitcnt vmcnt(0)
v_mul_f32 v12, v5, v8 // (op 127) 
v_mul_f32 v13, v5, v9 // (op 127) 
v_mul_f32 v14, v5, v10 // (op 127) 
v_mul_f32 v15, v5, v11 // (op 127) 

v_add_f32 v8, v12, v16 // (op 124) 
v_add_f32 v9, v13, v17 // (op 124) 
v_add_f32 v10, v14, v18 // (op 124) 
v_add_f32 v11, v15, v19 // (op 124) 


v_lshrrev_b32 v0, 7, v1 // (op 842) 
v_and_b32 v6, 63, v1 // (op 842) 
v_lshrrev_b32 v3, 4, v6 // (op 842) 
v_lshlrev_b32 v4, 2, v3 // (op 842) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 842) 


v_lshl_add_u32 v0, s16, 5, v3 // (op 842) 

v_lshrrev_b32 v3, 2, v6 // (op 842) 

v_and_b32 v4, 3, v3 // (op 842) 

v_and_b32 v3, 3, v1 // (op 842) 
v_lshl_add_u32 v6, v4, 2, v3 // (op 842) 


v_lshrrev_b32 v3, 6, v1 // (op 842) 
v_and_b32 v4, 1, v3 // (op 842) 

v_lshl_add_u32 v3, v4, 4, v6 // (op 842) 


v_lshl_add_u32 v4, s17, 5, v3 // (op 842) 

v_mul_lo_u32 v3, v4, s28 // (op 842) 

v_add_lshl_u32 v4, v0, v3, 2 // (op 842) 


s_mul_i32 s1, s28, s33 // (op 844) most significant: low of low * high
s_mul_i32 s2, s29, s32 // (op 844) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 844) least significant
s_mul_hi_u32 s3, s28, s32 // (op 844) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 844) most significant: sum
s_add_u32 s1, s1, s3 // (op 844) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 2 // (op 844) 

s_mov_b32 s4, s30 // (op 844) reg expression
s_mov_b32 s5, s31 // (op 844) reg expression
s_mov_b32 s6, s2 // (op 844) reg expression
s_mov_b32 s7, 131072 // (op 844) call()




buffer_store_dwordx4 v[8:11], v4, s[4:7], 0 offen // (op 125) Store value






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalBottom_0_beta_114:



s_endpgm  // End of RR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2
.LRR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_end:
.size RR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2, .LRR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_end-RR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2
  .amdhsa_next_free_vgpr 52
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 16384 // lds bytes
  .amdhsa_accum_offset 48
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
    DataTypeB: F8
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
    macrotile: [32, 32, 128]
    threads: [256, 1, 1]
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
  - .name: RR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2
    .symbol: RR_GEMM_NT_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 16384
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 60
    .vgpr_count: 46
    .agpr_count: 4
    .max_flat_workgroup_size: 256
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

