// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2
.p2align 8
.type RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2,@function
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 759) 
v_and_b32 v3, 15, v1 // (op 759) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 759) 


v_and_b32 v0, 48, v1 // (op 759) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 759) 


v_lshrrev_b32 v0, 6, v1 // (op 811) 
v_and_b32 v4, 1, v0 // (op 811) 

v_and_b32 v0, 15, v1 // (op 811) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 811) 


v_and_b32 v0, 48, v1 // (op 811) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 811) 


v_lshrrev_b32 v0, 7, v1 // (op 884) 
v_and_b32 v5, 15, v1 // (op 884) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 884) 


v_and_b32 v0, 48, v1 // (op 884) 
v_lshl_add_u32 v5, v6, 7, v0 // (op 884) 


v_lshrrev_b32 v0, 6, v1 // (op 923) 
v_and_b32 v6, 1, v0 // (op 923) 

v_and_b32 v0, 15, v1 // (op 923) 
v_lshl_add_u32 v7, v6, 4, v0 // (op 923) 


v_and_b32 v0, 48, v1 // (op 923) 
v_lshl_add_u32 v6, v7, 7, v0 // (op 923) 


v_mov_b32 v0, s14 // (op 2) Move value
v_mov_b32 v7, s15 // (op 35) Move value


v_accvgpr_write a0, 0.00000 // (op 55) call()
v_accvgpr_write a1, 0.00000 // (op 55) call()
v_accvgpr_write a2, 0.00000 // (op 55) call()
v_accvgpr_write a3, 0.00000 // (op 55) call()
v_lshrrev_b32 v8, 3, v1 // (op 470) 
v_lshl_add_u32 v9, s16, 5, v8 // (op 470) 

v_and_b32 v8, -8, v1 // (op 470) 
v_sub_u32 v10, v1, v8 // (op 470) 

v_lshlrev_b32 v8, 4, v10 // (op 470) 

v_mov_b32 v11, 0 // (op 470) convert
v_mov_b32 v10, v8 // (op 470) convert

v_mul_lo_u32 v13, v9, s3 // (op 470) most significant: low of low * high
v_mul_hi_u32 v8, v9, s2 // (op 470) most significant: high of low * low
v_mul_lo_u32 v12, v9, s2 // (op 470) least significant: low of low * low
v_add_u32 v13, v13, v8 // (op 470) most significant: sum

v_add_co_u32 v12, s[0:1], v12, v10 // (op 470) least significant half
v_addc_co_u32 v13, s[0:1], v13, v11, s[0:1] // (op 470) most significant half



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

v_lshrrev_b32 v8, 3, v1 // (op 489) 
v_lshl_add_u32 v9, s16, 5, v8 // (op 489) 

v_and_b32 v8, -8, v1 // (op 489) 
v_sub_u32 v10, v1, v8 // (op 489) 

v_lshlrev_b32 v8, 4, v10 // (op 489) 

v_mov_b32 v11, 0 // (op 489) convert
v_mov_b32 v10, v8 // (op 489) convert

v_mul_lo_u32 v15, v9, s3 // (op 489) most significant: low of low * high
v_mul_hi_u32 v8, v9, s2 // (op 489) most significant: high of low * low
v_mul_lo_u32 v14, v9, s2 // (op 489) least significant: low of low * low
v_add_u32 v15, v15, v8 // (op 489) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v10 // (op 489) least significant half
v_addc_co_u32 v15, s[0:1], v15, v11, s[0:1] // (op 489) most significant half



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

v_and_b32 v8, -8, v1 // (op 534) 
v_sub_u32 v9, v1, v8 // (op 534) 

v_lshlrev_b32 v8, 4, v9 // (op 534) 

v_mov_b32 v11, 0 // (op 534) convert
v_mov_b32 v10, v8 // (op 534) convert

v_lshrrev_b32 v8, 3, v1 // (op 534) 
v_lshl_add_u32 v9, s17, 5, v8 // (op 534) 

v_mul_lo_u32 v17, v9, s9 // (op 534) most significant: low of low * high
v_mul_hi_u32 v8, v9, s8 // (op 534) most significant: high of low * low
v_mul_lo_u32 v16, v9, s8 // (op 534) least significant: low of low * low
v_add_u32 v17, v17, v8 // (op 534) most significant: sum


v_add_co_u32 v8, s[0:1], v10, v16 // (op 534) least significant half
v_addc_co_u32 v9, s[0:1], v11, v17, s[0:1] // (op 534) most significant half



s_mul_i32 s1, s8, s13 // (op 536) most significant: low of low * high
s_mul_i32 s14, s9, s12 // (op 536) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 536) least significant
s_mul_hi_u32 s15, s8, s12 // (op 536) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 536) most significant: sum
s_add_u32 s1, s1, s15 // (op 536) most significant: sum


s_mov_b32 s44, s10 // (op 536) reg expression
s_mov_b32 s45, s11 // (op 536) reg expression
s_mov_b32 s46, s0 // (op 536) reg expression
s_mov_b32 s47, 131072 // (op 536) call()

v_and_b32 v10, -8, v1 // (op 553) 
v_sub_u32 v11, v1, v10 // (op 553) 

v_lshlrev_b32 v10, 4, v11 // (op 553) 

v_mov_b32 v17, 0 // (op 553) convert
v_mov_b32 v16, v10 // (op 553) convert

v_lshrrev_b32 v10, 3, v1 // (op 553) 
v_lshl_add_u32 v11, s17, 5, v10 // (op 553) 

v_mul_lo_u32 v19, v11, s9 // (op 553) most significant: low of low * high
v_mul_hi_u32 v10, v11, s8 // (op 553) most significant: high of low * low
v_mul_lo_u32 v18, v11, s8 // (op 553) least significant: low of low * low
v_add_u32 v19, v19, v10 // (op 553) most significant: sum


v_add_co_u32 v10, s[0:1], v16, v18 // (op 553) least significant half
v_addc_co_u32 v11, s[0:1], v17, v19, s[0:1] // (op 553) most significant half



s_mul_i32 s1, s8, s13 // (op 555) most significant: low of low * high
s_mul_i32 s14, s9, s12 // (op 555) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 555) least significant
s_mul_hi_u32 s15, s8, s12 // (op 555) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 555) most significant: sum
s_add_u32 s1, s1, s15 // (op 555) most significant: sum


s_mov_b32 s44, s10 // (op 555) reg expression
s_mov_b32 s45, s11 // (op 555) reg expression
s_mov_b32 s46, s0 // (op 555) reg expression
s_mov_b32 s47, 131072 // (op 555) call()

v_lshrrev_b32 v16, 7, v1 // (op 598) 
v_and_b32 v17, 15, v1 // (op 598) 
v_lshl_add_u32 v18, v16, 4, v17 // (op 598) 


v_lshl_add_u32 v16, s16, 5, v18 // (op 598) 

v_and_b32 v17, 63, v1 // (op 598) 
v_lshrrev_b32 v18, 4, v17 // (op 598) 


s_waitcnt lgkmcnt(0)
v_mul_lo_u32 v17, v16, s20 // (op 598) 
v_add_u32 v17, v17, v18 // (op 598) 


s_mul_i32 s1, s20, s7 // (op 600) most significant: low of low * high
s_mul_i32 s14, s21, s6 // (op 600) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 600) least significant
s_mul_hi_u32 s15, s20, s6 // (op 600) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 600) most significant: sum
s_add_u32 s1, s1, s15 // (op 600) most significant: sum


s_mov_b32 s48, s22 // (op 600) reg expression
s_mov_b32 s49, s23 // (op 600) reg expression
s_mov_b32 s50, s0 // (op 600) reg expression
s_mov_b32 s51, 131072 // (op 600) call()

v_lshrrev_b32 v16, 7, v1 // (op 617) 
v_and_b32 v18, 15, v1 // (op 617) 
v_lshl_add_u32 v19, v16, 4, v18 // (op 617) 


v_lshl_add_u32 v16, s16, 5, v19 // (op 617) 

v_and_b32 v18, 63, v1 // (op 617) 
v_lshrrev_b32 v19, 4, v18 // (op 617) 

v_mul_lo_u32 v18, v16, s20 // (op 617) 
v_add_u32 v18, v18, v19 // (op 617) 


s_mul_i32 s1, s20, s7 // (op 619) most significant: low of low * high
s_mul_i32 s14, s21, s6 // (op 619) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 619) least significant
s_mul_hi_u32 s15, s20, s6 // (op 619) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 619) most significant: sum
s_add_u32 s1, s1, s15 // (op 619) most significant: sum


s_mov_b32 s48, s22 // (op 619) reg expression
s_mov_b32 s49, s23 // (op 619) reg expression
s_mov_b32 s50, s0 // (op 619) reg expression
s_mov_b32 s51, 131072 // (op 619) call()

v_lshrrev_b32 v16, 6, v1 // (op 662) 
v_and_b32 v19, 1, v16 // (op 662) 

v_and_b32 v16, 15, v1 // (op 662) 
v_lshl_add_u32 v20, v19, 4, v16 // (op 662) 


v_lshl_add_u32 v16, s17, 5, v20 // (op 662) 

v_mul_lo_u32 v19, v16, s24 // (op 662) 

v_and_b32 v16, 63, v1 // (op 662) 
v_lshrrev_b32 v20, 4, v16 // (op 662) 

v_add_u32 v16, v20, v19 // (op 662) 


s_mul_i32 s1, s24, s13 // (op 664) most significant: low of low * high
s_mul_i32 s14, s25, s12 // (op 664) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 664) least significant
s_mul_hi_u32 s15, s24, s12 // (op 664) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 664) most significant: sum
s_add_u32 s1, s1, s15 // (op 664) most significant: sum


s_mov_b32 s52, s26 // (op 664) reg expression
s_mov_b32 s53, s27 // (op 664) reg expression
s_mov_b32 s54, s0 // (op 664) reg expression
s_mov_b32 s55, 131072 // (op 664) call()

v_lshrrev_b32 v19, 6, v1 // (op 681) 
v_and_b32 v20, 1, v19 // (op 681) 

v_and_b32 v19, 15, v1 // (op 681) 
v_lshl_add_u32 v21, v20, 4, v19 // (op 681) 


v_lshl_add_u32 v19, s17, 5, v21 // (op 681) 

v_mul_lo_u32 v20, v19, s24 // (op 681) 

v_and_b32 v19, 63, v1 // (op 681) 
v_lshrrev_b32 v21, 4, v19 // (op 681) 

v_add_u32 v19, v21, v20 // (op 681) 


s_mul_i32 s1, s24, s13 // (op 683) most significant: low of low * high
s_mul_i32 s14, s25, s12 // (op 683) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 683) least significant
s_mul_hi_u32 s15, s24, s12 // (op 683) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 683) most significant: sum
s_add_u32 s1, s1, s15 // (op 683) most significant: sum


s_mov_b32 s52, s26 // (op 683) reg expression
s_mov_b32 s53, s27 // (op 683) reg expression
s_mov_b32 s54, s0 // (op 683) reg expression
s_mov_b32 s55, 131072 // (op 683) call()




buffer_load_ubyte v20, v17, s[48:51], 0 offen offset: 0 // (op 308) Load value



buffer_load_ubyte v21, v16, s[52:55], 0 offen offset: 0 // (op 313) Load value
v_and_b32 v22, -8, v1 // (op 799) 
v_sub_u32 v23, v1, v22 // (op 799) 

v_lshlrev_b32 v22, 4, v23 // (op 799) 

v_lshrrev_b32 v23, 3, v1 // (op 799) 
v_lshl_add_u32 v24, v23, 7, v22 // (op 799) 


s_nop 0
v_readfirstlane_b32 s0, v24 // (op 799) Wait state hazard: VALU Write Readlane Hazard




s_nop 1
s_mov_b32 m0, s0 // (op 419) reg expression
buffer_load_dwordx4 v8, s[44:47], 0 offen offset: 0 lds // (op 419) Load value direct to lds

v_and_b32 v22, -8, v1 // (op 747) 
v_sub_u32 v23, v1, v22 // (op 747) 

v_lshlrev_b32 v22, 4, v23 // (op 747) 

v_lshrrev_b32 v23, 3, v1 // (op 747) 
v_lshl_add_u32 v24, v23, 7, v22 // (op 747) 


s_nop 0
v_readfirstlane_b32 s0, v24 // (op 747) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s1, 4096, s0 // (op 424) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 424) reg expression
buffer_load_dwordx4 v12, s[40:43], 0 offen offset: 0 lds // (op 424) Load value direct to lds





buffer_load_ubyte v22, v17, s[48:51], 0 offen offset:4 // (op 324) Load value




buffer_load_ubyte v17, v16, s[52:55], 0 offen offset:4 // (op 329) Load value

v_and_b32 v16, -8, v1 // (op 911) 
v_sub_u32 v23, v1, v16 // (op 911) 

v_lshlrev_b32 v16, 4, v23 // (op 911) 

v_lshrrev_b32 v23, 3, v1 // (op 911) 
v_lshl_add_u32 v24, v23, 7, v16 // (op 911) 


s_nop 0
v_readfirstlane_b32 s0, v24 // (op 911) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s1, 8192, s0 // (op 429) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 429) reg expression
s_mov_b32 s14, 128 // (op 429) call()
buffer_load_dwordx4 v8, s[44:47], s14 offen offset: 0 lds // (op 429) Load value direct to lds




v_and_b32 v16, -8, v1 // (op 872) 
v_sub_u32 v8, v1, v16 // (op 872) 

v_lshlrev_b32 v9, 4, v8 // (op 872) 

v_lshrrev_b32 v8, 3, v1 // (op 872) 
v_lshl_add_u32 v16, v8, 7, v9 // (op 872) 


s_nop 0
v_readfirstlane_b32 s0, v16 // (op 872) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s1, 12288, s0 // (op 431) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 431) reg expression
s_mov_b32 s14, 128 // (op 431) call()
buffer_load_dwordx4 v12, s[40:43], s14 offen offset: 0 lds // (op 431) Load value direct to lds




s_waitcnt vmcnt(5)
s_barrier  // (op 968) 



ds_read_b128 v[24:27], v4 // (op 354) Load local data 
ds_read_b128 v[28:31], v4 offset:64 // (op 354) Load local data 
s_waitcnt vmcnt(4)
s_barrier  // (op 952) 



ds_read_b128 v[32:35], v3 offset:4096 // (op 351) Load local data 
ds_read_b128 v[36:39], v3 offset:4160 // (op 351) Load local data 
v_and_b32 v16, -8, v1 // (op 719) 
v_sub_u32 v23, v1, v16 // (op 719) 

v_lshlrev_b32 v16, 4, v23 // (op 719) 

v_lshrrev_b32 v23, 3, v1 // (op 719) 
v_lshl_add_u32 v8, v23, 7, v16 // (op 719) 


s_nop 0
v_readfirstlane_b32 s0, v8 // (op 719) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v16, -8, v1 // (op 771) 
v_sub_u32 v23, v1, v16 // (op 771) 

v_lshlrev_b32 v16, 4, v23 // (op 771) 

v_lshrrev_b32 v23, 3, v1 // (op 771) 
v_lshl_add_u32 v8, v23, 7, v16 // (op 771) 


s_nop 0
v_readfirstlane_b32 s1, v8 // (op 771) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v16, -8, v1 // (op 857) 
v_sub_u32 v23, v1, v16 // (op 857) 

v_lshlrev_b32 v16, 4, v23 // (op 857) 

v_lshrrev_b32 v23, 3, v1 // (op 857) 
v_lshl_add_u32 v8, v23, 7, v16 // (op 857) 


s_nop 0
v_readfirstlane_b32 s14, v8 // (op 857) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v16, -8, v1 // (op 896) 
v_sub_u32 v23, v1, v16 // (op 896) 

v_lshlrev_b32 v16, 4, v23 // (op 896) 

v_lshrrev_b32 v23, 3, v1 // (op 896) 
v_lshl_add_u32 v8, v23, 7, v16 // (op 896) 


s_nop 0
v_readfirstlane_b32 s15, v8 // (op 896) Wait state hazard: VALU Write Readlane Hazard

s_mov_b32 s18, 0 // (op 51) call()
s_ashr_i64 s[38:39], s[28:29], 7 // (op 50) 
s_mov_b32 s19, s38 // (op 50) convert

s_and_b32 s38, s19, 4294967294 // (op 50) 

s_cmp_lt_u32 s18, s38 // (op 50) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoop_50 // (op 50) Condition: Top (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoop_50 if false)
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoop_50:
s_add_u32 s18, s18, 2 // (op 52) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[32:39], v[24:31], a[0:3], v20, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 76) 


v_mov_b32 v16, v18 // (op 70) 

buffer_load_ubyte v20, v16, s[48:51], 0 offen offset:8 // (op 70) Load value


v_mov_b32 v23, v19 // (op 74) 

buffer_load_ubyte v21, v23, s[52:55], 0 offen offset:8 // (op 74) Load value
s_barrier  // (op 971) 


v_mov_b64 v[8:9], v[10:11] // (op 392) 

s_mov_b32 m0, s1 // (op 392) reg expression
s_mov_b32 s19, 256 // (op 392) call()
buffer_load_dwordx4 v8, s[44:47], s19 offen offset: 0 lds // (op 392) Load value direct to lds

s_barrier  // (op 955) 


v_mov_b64 v[12:13], v[14:15] // (op 387) 
s_add_u32 s19, 4096, s0 // (op 387) 

s_mov_b32 m0, s19 // (op 387) reg expression
s_mov_b32 s38, 256 // (op 387) call()
buffer_load_dwordx4 v12, s[40:43], s38 offen offset: 0 lds // (op 387) Load value direct to lds


s_waitcnt vmcnt(5)
s_barrier  // (op 990) 



ds_read_b128 v[24:27], v6 offset:8192 // (op 212) Load local data 
ds_read_b128 v[28:31], v6 offset:8256 // (op 212) Load local data 
s_waitcnt vmcnt(4)
s_barrier  // (op 984) 



ds_read_b128 v[32:35], v5 offset:12288 // (op 210) Load local data 
ds_read_b128 v[36:39], v5 offset:12352 // (op 210) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[32:39], v[24:31], a[0:3], v22, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 217) 



buffer_load_ubyte v22, v16, s[48:51], 0 offen offset:12 // (op 214) Load value




v_add_u32 v18, 8, v18 // (op 609) 
buffer_load_ubyte v17, v23, s[52:55], 0 offen offset:12 // (op 216) Load value

s_barrier  // (op 993) 
v_add_u32 v19, 8, v19 // (op 673) 


s_add_u32 s19, 8192, s15 // (op 414) 

s_mov_b32 m0, s19 // (op 414) reg expression
s_mov_b32 s38, 384 // (op 414) call()
buffer_load_dwordx4 v8, s[44:47], s38 offen offset: 0 lds // (op 414) Load value direct to lds



s_barrier  // (op 987) 
v_add_co_u32 v10, vcc, 256, v10 // (op 545) least significant half


v_addc_co_u32 v11, vcc, 0, v11, vcc // (op 545) most significant half
s_add_u32 s19, 12288, s14 // (op 409) 

s_mov_b32 m0, s19 // (op 409) reg expression
s_mov_b32 s38, 384 // (op 409) call()
buffer_load_dwordx4 v12, s[40:43], s38 offen offset: 0 lds // (op 409) Load value direct to lds



v_add_co_u32 v14, vcc, 256, v14 // (op 481) least significant half
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 481) most significant half
s_waitcnt vmcnt(5)
s_barrier  // (op 977) 



ds_read_b128 v[24:27], v4 // (op 65) Load local data 
ds_read_b128 v[28:31], v4 offset:64 // (op 65) Load local data 
s_waitcnt vmcnt(4)
s_barrier  // (op 961) 



ds_read_b128 v[32:35], v3 offset:4096 // (op 60) Load local data 
ds_read_b128 v[36:39], v3 offset:4160 // (op 60) Load local data 
s_ashr_i64 s[38:39], s[28:29], 7 // (op 50) 
s_mov_b32 s19, s38 // (op 50) convert

s_and_b32 s38, s19, 4294967294 // (op 50) 

s_cmp_lt_u32 s18, s38 // (op 50) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoop_50 // (op 50) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoop_50 if true)
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoop_50:











s_barrier  // (op 1000) 
v_lshrrev_b32 v23, 3, v1 // (op 444) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 444) 
v_lshl_add_u32 v8, s16, 5, v23 // (op 444) 

s_lshr_b32 s14, s0, 1 // (op 444) 

v_and_b32 v23, -8, v1 // (op 444) 
v_sub_u32 v9, v1, v23 // (op 444) 

v_lshlrev_b32 v23, 4, v9 // (op 444) 

v_lshl_add_u32 v9, s14, 8, v23 // (op 444) 


v_mov_b32 v11, 0 // (op 444) convert
v_mov_b32 v10, v9 // (op 444) convert

v_mul_lo_u32 v13, v8, s3 // (op 444) most significant: low of low * high
v_mul_hi_u32 v9, v8, s2 // (op 444) most significant: high of low * low
v_mul_lo_u32 v12, v8, s2 // (op 444) least significant: low of low * low
v_add_u32 v13, v13, v9 // (op 444) most significant: sum

v_add_co_u32 v12, s[0:1], v12, v10 // (op 444) least significant half
v_addc_co_u32 v13, s[0:1], v13, v11, s[0:1] // (op 444) most significant half



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

v_and_b32 v23, -8, v1 // (op 511) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 511) 
v_sub_u32 v8, v1, v23 // (op 511) 

s_lshr_b32 s14, s0, 1 // (op 511) 

v_lshlrev_b32 v23, 4, v8 // (op 511) 

v_lshl_add_u32 v8, s14, 8, v23 // (op 511) 


v_mov_b32 v11, 0 // (op 511) convert
v_mov_b32 v10, v8 // (op 511) convert

v_lshrrev_b32 v23, 3, v1 // (op 511) 
v_lshl_add_u32 v8, s17, 5, v23 // (op 511) 

v_mul_lo_u32 v15, v8, s9 // (op 511) most significant: low of low * high
v_mul_hi_u32 v9, v8, s8 // (op 511) most significant: high of low * low
v_mul_lo_u32 v14, v8, s8 // (op 511) least significant: low of low * low
v_add_u32 v15, v15, v9 // (op 511) most significant: sum


v_add_co_u32 v8, s[0:1], v10, v14 // (op 511) least significant half
v_addc_co_u32 v9, s[0:1], v11, v15, s[0:1] // (op 511) most significant half



s_mul_i32 s1, s8, s13 // (op 513) most significant: low of low * high
s_mul_i32 s14, s9, s12 // (op 513) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 513) least significant
s_mul_hi_u32 s15, s8, s12 // (op 513) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 513) most significant: sum
s_add_u32 s1, s1, s15 // (op 513) most significant: sum


s_mov_b32 s44, s10 // (op 513) reg expression
s_mov_b32 s45, s11 // (op 513) reg expression
s_mov_b32 s46, s0 // (op 513) reg expression
s_mov_b32 s47, 131072 // (op 513) call()

v_lshrrev_b32 v23, 7, v1 // (op 575) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 575) 
v_and_b32 v10, 15, v1 // (op 575) 
s_lshr_b32 s14, s0, 1 // (op 575) 

v_lshl_add_u32 v11, v23, 4, v10 // (op 575) 


v_lshl_add_u32 v10, s16, 5, v11 // (op 575) 

v_and_b32 v11, 63, v1 // (op 575) 
v_lshrrev_b32 v23, 4, v11 // (op 575) 

v_lshl_add_u32 v11, s14, 3, v23 // (op 575) 


v_mul_lo_u32 v23, v10, s20 // (op 575) 
v_add_u32 v23, v23, v11 // (op 575) 


s_mul_i32 s1, s20, s7 // (op 577) most significant: low of low * high
s_mul_i32 s14, s21, s6 // (op 577) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 577) least significant
s_mul_hi_u32 s15, s20, s6 // (op 577) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 577) most significant: sum
s_add_u32 s1, s1, s15 // (op 577) most significant: sum


s_mov_b32 s48, s22 // (op 577) reg expression
s_mov_b32 s49, s23 // (op 577) reg expression
s_mov_b32 s50, s0 // (op 577) reg expression
s_mov_b32 s51, 131072 // (op 577) call()

v_and_b32 v10, 63, v1 // (op 639) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 639) 
v_lshrrev_b32 v11, 4, v10 // (op 639) 

s_lshr_b32 s14, s0, 1 // (op 639) 

v_lshrrev_b32 v10, 6, v1 // (op 639) 
v_lshl_add_u32 v14, s14, 3, v11 // (op 639) 


v_and_b32 v11, 1, v10 // (op 639) 

v_and_b32 v10, 15, v1 // (op 639) 
v_lshl_add_u32 v15, v11, 4, v10 // (op 639) 


v_lshl_add_u32 v16, s17, 5, v15 // (op 639) 

v_mul_lo_u32 v15, v16, s24 // (op 639) 

v_add_u32 v16, v14, v15 // (op 639) 


s_mul_i32 s1, s24, s13 // (op 641) most significant: low of low * high
s_mul_i32 s14, s25, s12 // (op 641) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 641) least significant
s_mul_hi_u32 s15, s24, s12 // (op 641) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 641) most significant: sum
s_add_u32 s1, s1, s15 // (op 641) most significant: sum


s_mov_b32 s52, s26 // (op 641) reg expression
s_mov_b32 s53, s27 // (op 641) reg expression
s_mov_b32 s54, s0 // (op 641) reg expression
s_mov_b32 s55, 131072 // (op 641) call()

v_and_b32 v10, -8, v1 // (op 732) 
v_sub_u32 v11, v1, v10 // (op 732) 

v_lshlrev_b32 v10, 4, v11 // (op 732) 

v_lshrrev_b32 v11, 3, v1 // (op 732) 
v_lshl_add_u32 v14, v11, 7, v10 // (op 732) 


s_nop 0
v_readfirstlane_b32 s0, v14 // (op 732) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v10, -8, v1 // (op 784) 
v_sub_u32 v11, v1, v10 // (op 784) 

v_lshlrev_b32 v10, 4, v11 // (op 784) 

v_lshrrev_b32 v11, 3, v1 // (op 784) 
v_lshl_add_u32 v14, v11, 7, v10 // (op 784) 


s_nop 0
v_readfirstlane_b32 s1, v14 // (op 784) Wait state hazard: VALU Write Readlane Hazard

s_ashr_i64 s[14:15], s[28:29], 7 // (op 153) 
s_and_b32 s18, s14, 4294967294 // (op 153) 

s_ashr_i64 s[14:15], s[28:29], 7 // (op 152) 
s_mov_b32 s19, s14 // (op 152) convert

s_cmp_lt_u32 s18, s19 // (op 152) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoopTail_152 // (op 152) Condition: Top (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoopTail_152 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoopTail_152:
s_barrier  // (op 958) 
s_add_u32 s18, s18, 1 // (op 154) 


v_mov_b64 v[10:11], v[12:13] // (op 397) 
s_add_u32 s19, 4096, s0 // (op 397) 

s_mov_b32 m0, s19 // (op 397) reg expression
buffer_load_dwordx4 v10, s[40:43], 0 offen offset: 0 lds // (op 397) Load value direct to lds


v_add_co_u32 v12, vcc, 128, v12 // (op 434) least significant half
v_addc_co_u32 v13, vcc, 0, v13, vcc // (op 434) most significant half
s_barrier  // (op 974) 


v_mov_b64 v[10:11], v[8:9] // (op 403) 

s_mov_b32 m0, s1 // (op 403) reg expression
buffer_load_dwordx4 v10, s[44:47], 0 offen offset: 0 lds // (op 403) Load value direct to lds

v_add_co_u32 v8, vcc, 128, v8 // (op 503) least significant half
v_addc_co_u32 v9, vcc, 0, v9, vcc // (op 503) most significant half
s_waitcnt vmcnt(1)
s_barrier  // (op 964) 



ds_read_b128 v[24:27], v3 offset:4096 // (op 160) Load local data 
ds_read_b128 v[28:31], v3 offset:4160 // (op 160) Load local data 
s_waitcnt vmcnt(0)
s_barrier  // (op 980) 



ds_read_b128 v[32:35], v4 // (op 162) Load local data 
ds_read_b128 v[36:39], v4 offset:64 // (op 162) Load local data 


v_mov_b32 v10, v23 // (op 164) 

buffer_load_ubyte v11, v10, s[48:51], 0 offen offset: 0 // (op 164) Load value

v_add_u32 v23, 4, v23 // (op 567) 


v_mov_b32 v10, v16 // (op 166) 

buffer_load_ubyte v14, v10, s[52:55], 0 offen offset: 0 // (op 166) Load value

v_add_u32 v16, 4, v16 // (op 631) 
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[24:31], v[32:39], a[0:3], v11, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 167) 




s_ashr_i64 s[14:15], s[28:29], 7 // (op 152) 
s_mov_b32 s19, s14 // (op 152) convert

s_cmp_lt_u32 s18, s19 // (op 152) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoopTail_152 // (op 152) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoopTail_152 if true)
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoopTail_152:





















s_barrier  // (op 1022) 




v_cmp_eq_f32 vcc, 0.00000, v0 // (op 114) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalFalse_0_beta_114 // (op 114) Condition: False, jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalFalse_0_beta_114
s_nop 0
v_accvgpr_read v8, a0 // (op 42) Wait state hazard: XDL Write Hazard
v_accvgpr_read v9, a1 // (op 42) 
v_accvgpr_read v10, a2 // (op 42) 
v_accvgpr_read v11, a3 // (op 42) 
v_mul_f32 v12, v7, v8 // (op 42) 
v_mul_f32 v13, v7, v9 // (op 42) 
v_mul_f32 v14, v7, v10 // (op 42) 
v_mul_f32 v15, v7, v11 // (op 42) 

v_lshrrev_b32 v3, 7, v1 // (op 825) 
v_and_b32 v4, 63, v1 // (op 825) 
v_lshrrev_b32 v5, 4, v4 // (op 825) 
v_lshlrev_b32 v6, 2, v5 // (op 825) 

v_lshl_add_u32 v5, v3, 4, v6 // (op 825) 


v_lshl_add_u32 v3, s16, 5, v5 // (op 825) 

v_lshrrev_b32 v5, 2, v4 // (op 825) 

v_and_b32 v4, 3, v5 // (op 825) 

v_and_b32 v5, 3, v1 // (op 825) 
v_lshl_add_u32 v6, v4, 2, v5 // (op 825) 


v_lshrrev_b32 v4, 6, v1 // (op 825) 
v_and_b32 v5, 1, v4 // (op 825) 

v_lshl_add_u32 v4, v5, 4, v6 // (op 825) 


v_lshl_add_u32 v5, s17, 5, v4 // (op 825) 

v_mul_lo_u32 v4, v5, s30 // (op 825) 

v_add_lshl_u32 v5, v3, v4, 2 // (op 825) 


s_mul_i32 s15, s30, s13 // (op 827) most significant: low of low * high
s_mul_i32 s0, s31, s12 // (op 827) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 827) least significant
s_mul_hi_u32 s1, s30, s12 // (op 827) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 827) most significant: sum
s_add_u32 s15, s15, s1 // (op 827) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 2 // (op 827) 

s_mov_b32 s8, s32 // (op 827) reg expression
s_mov_b32 s9, s33 // (op 827) reg expression
s_mov_b32 s10, s0 // (op 827) reg expression
s_mov_b32 s11, 131072 // (op 827) call()




buffer_store_dwordx4 v[12:15], v5, s[8:11], 0 offen // (op 48) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalBottom_0_beta_114 // (op 114) Condition: Done, jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalBottom_0_beta_114
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalFalse_0_beta_114:
v_lshrrev_b32 v3, 7, v1 // (op 702) 
v_and_b32 v4, 63, v1 // (op 702) 
v_lshrrev_b32 v5, 4, v4 // (op 702) 
v_lshlrev_b32 v6, 2, v5 // (op 702) 

v_lshl_add_u32 v5, v3, 4, v6 // (op 702) 


v_lshl_add_u32 v3, s16, 5, v5 // (op 702) 

v_lshrrev_b32 v5, 2, v4 // (op 702) 

v_and_b32 v4, 3, v5 // (op 702) 

v_and_b32 v5, 3, v1 // (op 702) 
v_lshl_add_u32 v6, v4, 2, v5 // (op 702) 


v_lshrrev_b32 v4, 6, v1 // (op 702) 
v_and_b32 v5, 1, v4 // (op 702) 

v_lshl_add_u32 v4, v5, 4, v6 // (op 702) 


v_lshl_add_u32 v5, s17, 5, v4 // (op 702) 

v_mul_lo_u32 v4, v5, s34 // (op 702) 

v_add_lshl_u32 v5, v3, v4, 2 // (op 702) 


s_mul_i32 s15, s34, s13 // (op 704) most significant: low of low * high
s_mul_i32 s0, s35, s12 // (op 704) most significant: low of high * low
s_mul_i32 s14, s34, s12 // (op 704) least significant
s_mul_hi_u32 s1, s34, s12 // (op 704) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 704) most significant: sum
s_add_u32 s15, s15, s1 // (op 704) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 2 // (op 704) 

s_mov_b32 s4, s36 // (op 704) reg expression
s_mov_b32 s5, s37 // (op 704) reg expression
s_mov_b32 s6, s0 // (op 704) reg expression
s_mov_b32 s7, 131072 // (op 704) call()





buffer_load_dwordx4 v[8:11], v5, s[4:7], 0 offen // (op 126) Load value



v_accvgpr_read v12, a0 // (op 123) 
v_accvgpr_read v13, a1 // (op 123) 
v_accvgpr_read v14, a2 // (op 123) 
v_accvgpr_read v15, a3 // (op 123) 
v_mul_f32 v16, v7, v12 // (op 123) 
v_mul_f32 v17, v7, v13 // (op 123) 
v_mul_f32 v18, v7, v14 // (op 123) 
v_mul_f32 v19, v7, v15 // (op 123) 

s_waitcnt vmcnt(0)
v_mul_f32 v12, v0, v8 // (op 127) 
v_mul_f32 v13, v0, v9 // (op 127) 
v_mul_f32 v14, v0, v10 // (op 127) 
v_mul_f32 v15, v0, v11 // (op 127) 

v_add_f32 v8, v12, v16 // (op 124) 
v_add_f32 v9, v13, v17 // (op 124) 
v_add_f32 v10, v14, v18 // (op 124) 
v_add_f32 v11, v15, v19 // (op 124) 


v_lshrrev_b32 v3, 7, v1 // (op 842) 
v_and_b32 v4, 63, v1 // (op 842) 
v_lshrrev_b32 v5, 4, v4 // (op 842) 
v_lshlrev_b32 v6, 2, v5 // (op 842) 

v_lshl_add_u32 v5, v3, 4, v6 // (op 842) 


v_lshl_add_u32 v3, s16, 5, v5 // (op 842) 

v_lshrrev_b32 v5, 2, v4 // (op 842) 

v_and_b32 v4, 3, v5 // (op 842) 

v_and_b32 v5, 3, v1 // (op 842) 
v_lshl_add_u32 v6, v4, 2, v5 // (op 842) 


v_lshrrev_b32 v4, 6, v1 // (op 842) 
v_and_b32 v5, 1, v4 // (op 842) 

v_lshl_add_u32 v4, v5, 4, v6 // (op 842) 


v_lshl_add_u32 v5, s17, 5, v4 // (op 842) 

v_mul_lo_u32 v4, v5, s30 // (op 842) 

v_add_lshl_u32 v5, v3, v4, 2 // (op 842) 


s_mul_i32 s15, s30, s13 // (op 844) most significant: low of low * high
s_mul_i32 s0, s31, s12 // (op 844) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 844) least significant
s_mul_hi_u32 s1, s30, s12 // (op 844) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 844) most significant: sum
s_add_u32 s15, s15, s1 // (op 844) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 2 // (op 844) 

s_mov_b32 s8, s32 // (op 844) reg expression
s_mov_b32 s9, s33 // (op 844) reg expression
s_mov_b32 s10, s0 // (op 844) reg expression
s_mov_b32 s11, 131072 // (op 844) call()




buffer_store_dwordx4 v[8:11], v5, s[8:11], 0 offen // (op 125) Store value





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalBottom_0_beta_114:




s_endpgm  // End of RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2
.LRR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_end:
.size RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2, .LRR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_end-RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2
  .amdhsa_next_free_vgpr 44
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 16384 // lds bytes
  .amdhsa_accum_offset 40
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
    DataTypeB: F8
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
    macrotile: [32, 32, 128]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2
    .symbol: RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 16384
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 56
    .vgpr_count: 40
    .agpr_count: 4
    .max_flat_workgroup_size: 256
    .workgroup_size: [256, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, {is-null: true}]
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
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: FP8, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP8, pointerType: PointerGlobal}
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

