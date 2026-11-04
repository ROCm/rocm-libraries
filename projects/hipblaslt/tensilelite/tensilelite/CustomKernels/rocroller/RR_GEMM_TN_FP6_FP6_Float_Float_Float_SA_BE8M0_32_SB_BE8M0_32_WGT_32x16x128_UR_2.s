// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT



.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2
.p2align 8
.type RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2,@function
RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 6, v1 // (op 739) 
v_and_b32 v3, 15, v1 // (op 739) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 739) 


v_and_b32 v0, 63, v1 // (op 739) 
v_lshrrev_b32 v3, 4, v0 // (op 739) 

v_lshlrev_b32 v0, 5, v3 // (op 739) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 739) 


v_mul_lo_u32 v0, v3, 6 // (op 739) 

v_lshrrev_b32 v3, 3, v0 // (op 739) 

v_and_b32 v0, 15, v1 // (op 792) 
v_and_b32 v4, 63, v1 // (op 792) 
v_lshrrev_b32 v5, 4, v4 // (op 792) 

v_lshlrev_b32 v4, 5, v5 // (op 792) 

v_lshl_add_u32 v5, v0, 7, v4 // (op 792) 


v_mul_lo_u32 v0, v5, 6 // (op 792) 

v_lshrrev_b32 v4, 3, v0 // (op 792) 

v_lshrrev_b32 v0, 6, v1 // (op 845) 
v_and_b32 v5, 15, v1 // (op 845) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 845) 


v_and_b32 v0, 63, v1 // (op 845) 
v_lshrrev_b32 v5, 4, v0 // (op 845) 

v_lshl_add_u32 v0, v6, 2, v5 // (op 845) 


v_lshrrev_b32 v5, 6, v1 // (op 903) 
v_and_b32 v6, 15, v1 // (op 903) 
v_lshl_add_u32 v7, v5, 4, v6 // (op 903) 


v_and_b32 v5, 63, v1 // (op 903) 
v_lshrrev_b32 v6, 4, v5 // (op 903) 

v_lshlrev_b32 v5, 5, v6 // (op 903) 

v_lshl_add_u32 v6, v7, 7, v5 // (op 903) 


v_mul_lo_u32 v5, v6, 6 // (op 903) 

v_lshrrev_b32 v6, 3, v5 // (op 903) 

v_and_b32 v5, 15, v1 // (op 927) 
v_and_b32 v7, 63, v1 // (op 927) 
v_lshrrev_b32 v8, 4, v7 // (op 927) 

v_lshlrev_b32 v7, 5, v8 // (op 927) 

v_lshl_add_u32 v8, v5, 7, v7 // (op 927) 


v_mul_lo_u32 v5, v8, 6 // (op 927) 

v_lshrrev_b32 v7, 3, v5 // (op 927) 

v_lshrrev_b32 v5, 6, v1 // (op 951) 
v_and_b32 v8, 15, v1 // (op 951) 
v_lshl_add_u32 v9, v5, 4, v8 // (op 951) 


v_and_b32 v5, 63, v1 // (op 951) 
v_lshrrev_b32 v8, 4, v5 // (op 951) 

v_lshl_add_u32 v5, v9, 2, v8 // (op 951) 


v_mov_b32 v8, s14 // (op 2) Move value
v_mov_b32 v9, s15 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 59) call()
v_accvgpr_write a1, 0.00000 // (op 59) call()
v_accvgpr_write a2, 0.00000 // (op 59) call()
v_accvgpr_write a3, 0.00000 // (op 59) call()
v_lshrrev_b32 v10, 2, v1 // (op 449) 
v_lshl_add_u32 v11, s16, 5, v10 // (op 449) 

v_and_b32 v10, -4, v1 // (op 449) 
v_sub_u32 v12, v1, v10 // (op 449) 

v_lshlrev_b32 v10, 5, v12 // (op 449) 

v_mov_b32 v13, 0 // (op 449) convert
v_mov_b32 v12, v10 // (op 449) convert

v_mul_lo_u32 v15, v11, s3 // (op 449) most significant: low of low * high
v_mul_hi_u32 v10, v11, s2 // (op 449) most significant: high of low * low
v_mul_lo_u32 v14, v11, s2 // (op 449) least significant: low of low * low
v_add_u32 v15, v15, v10 // (op 449) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v12 // (op 449) least significant half
v_addc_co_u32 v15, s[0:1], v15, v13, s[0:1] // (op 449) most significant half



v_mul_lo_u32 v11, v15, 6 // (op 449) most significant: low of high * low
v_mul_hi_u32 v12, v14, 6 // (op 449) most significant: high of low * low
v_mul_lo_u32 v10, v14, 6 // (op 449) least significant: low of low * low
v_add_u32 v11, v11, v12 // (op 449) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 449) 

v_mov_b32 v10, v12 // (op 449) convert

s_mul_i32 s1, s7, 6 // (op 451) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 451) least significant
s_mul_hi_u32 s14, s6, 6 // (op 451) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 451) most significant: sum

s_mul_i32 s15, s2, s1 // (op 451) most significant: low of low * high
s_mul_i32 s18, s3, s0 // (op 451) most significant: low of high * low
s_mul_i32 s14, s2, s0 // (op 451) least significant
s_mul_hi_u32 s19, s2, s0 // (op 451) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 451) most significant: sum
s_add_u32 s15, s15, s19 // (op 451) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 451) 

s_mov_b32 s40, s4 // (op 451) reg expression
s_mov_b32 s41, s5 // (op 451) reg expression
s_mov_b32 s42, s0 // (op 451) reg expression
s_mov_b32 s43, 131072 // (op 451) call()

v_lshrrev_b32 v11, 2, v1 // (op 468) 
v_lshl_add_u32 v12, s16, 5, v11 // (op 468) 

v_and_b32 v11, -4, v1 // (op 468) 
v_sub_u32 v13, v1, v11 // (op 468) 

v_lshlrev_b32 v11, 5, v13 // (op 468) 

v_mov_b32 v15, 0 // (op 468) convert
v_mov_b32 v14, v11 // (op 468) convert

v_mul_lo_u32 v17, v12, s3 // (op 468) most significant: low of low * high
v_mul_hi_u32 v11, v12, s2 // (op 468) most significant: high of low * low
v_mul_lo_u32 v16, v12, s2 // (op 468) least significant: low of low * low
v_add_u32 v17, v17, v11 // (op 468) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 468) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 468) most significant half



v_mul_lo_u32 v15, v17, 6 // (op 468) most significant: low of high * low
v_mul_hi_u32 v11, v16, 6 // (op 468) most significant: high of low * low
v_mul_lo_u32 v14, v16, 6 // (op 468) least significant: low of low * low
v_add_u32 v15, v15, v11 // (op 468) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 468) 

v_mov_b32 v11, v12 // (op 468) convert

s_mul_i32 s1, s7, 6 // (op 470) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 470) least significant
s_mul_hi_u32 s14, s6, 6 // (op 470) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 470) most significant: sum

s_mul_i32 s15, s2, s1 // (op 470) most significant: low of low * high
s_mul_i32 s18, s3, s0 // (op 470) most significant: low of high * low
s_mul_i32 s14, s2, s0 // (op 470) least significant
s_mul_hi_u32 s19, s2, s0 // (op 470) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 470) most significant: sum
s_add_u32 s15, s15, s19 // (op 470) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 470) 

s_mov_b32 s40, s4 // (op 470) reg expression
s_mov_b32 s41, s5 // (op 470) reg expression
s_mov_b32 s42, s0 // (op 470) reg expression
s_mov_b32 s43, 131072 // (op 470) call()

v_and_b32 v12, -8, v1 // (op 513) 
v_sub_u32 v13, v1, v12 // (op 513) 

v_lshlrev_b32 v12, 4, v13 // (op 513) 

v_mov_b32 v15, 0 // (op 513) convert
v_mov_b32 v14, v12 // (op 513) convert

v_lshrrev_b32 v12, 3, v1 // (op 513) 
v_lshl_add_u32 v13, s17, 4, v12 // (op 513) 

v_mul_lo_u32 v17, v13, s9 // (op 513) most significant: low of low * high
v_mul_hi_u32 v12, v13, s8 // (op 513) most significant: high of low * low
v_mul_lo_u32 v16, v13, s8 // (op 513) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 513) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v16 // (op 513) least significant half
v_addc_co_u32 v13, s[0:1], v15, v17, s[0:1] // (op 513) most significant half



v_mul_lo_u32 v15, v13, 6 // (op 513) most significant: low of high * low
v_mul_hi_u32 v16, v12, 6 // (op 513) most significant: high of low * low
v_mul_lo_u32 v14, v12, 6 // (op 513) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 513) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 513) 

v_mov_b32 v14, v12 // (op 513) convert

s_mul_i32 s1, s13, 6 // (op 515) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 515) least significant
s_mul_hi_u32 s14, s12, 6 // (op 515) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 515) most significant: sum

s_mul_i32 s15, s8, s1 // (op 515) most significant: low of low * high
s_mul_i32 s18, s9, s0 // (op 515) most significant: low of high * low
s_mul_i32 s14, s8, s0 // (op 515) least significant
s_mul_hi_u32 s19, s8, s0 // (op 515) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 515) most significant: sum
s_add_u32 s15, s15, s19 // (op 515) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 515) 

s_mov_b32 s44, s10 // (op 515) reg expression
s_mov_b32 s45, s11 // (op 515) reg expression
s_mov_b32 s46, s0 // (op 515) reg expression
s_mov_b32 s47, 131072 // (op 515) call()

v_and_b32 v12, -8, v1 // (op 532) 
v_sub_u32 v13, v1, v12 // (op 532) 

v_lshlrev_b32 v12, 4, v13 // (op 532) 

v_mov_b32 v17, 0 // (op 532) convert
v_mov_b32 v16, v12 // (op 532) convert

v_lshrrev_b32 v15, 3, v1 // (op 532) 
v_lshl_add_u32 v12, s17, 4, v15 // (op 532) 

v_mul_lo_u32 v19, v12, s9 // (op 532) most significant: low of low * high
v_mul_hi_u32 v13, v12, s8 // (op 532) most significant: high of low * low
v_mul_lo_u32 v18, v12, s8 // (op 532) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 532) most significant: sum


v_add_co_u32 v12, s[0:1], v16, v18 // (op 532) least significant half
v_addc_co_u32 v13, s[0:1], v17, v19, s[0:1] // (op 532) most significant half



v_mul_lo_u32 v17, v13, 6 // (op 532) most significant: low of high * low
v_mul_hi_u32 v15, v12, 6 // (op 532) most significant: high of low * low
v_mul_lo_u32 v16, v12, 6 // (op 532) least significant: low of low * low
v_add_u32 v17, v17, v15 // (op 532) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[16:17] // (op 532) 

v_mov_b32 v15, v12 // (op 532) convert

s_mul_i32 s1, s13, 6 // (op 534) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 534) least significant
s_mul_hi_u32 s14, s12, 6 // (op 534) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 534) most significant: sum

s_mul_i32 s15, s8, s1 // (op 534) most significant: low of low * high
s_mul_i32 s18, s9, s0 // (op 534) most significant: low of high * low
s_mul_i32 s14, s8, s0 // (op 534) least significant
s_mul_hi_u32 s19, s8, s0 // (op 534) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 534) most significant: sum
s_add_u32 s15, s15, s19 // (op 534) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 534) 

s_mov_b32 s44, s10 // (op 534) reg expression
s_mov_b32 s45, s11 // (op 534) reg expression
s_mov_b32 s46, s0 // (op 534) reg expression
s_mov_b32 s47, 131072 // (op 534) call()

v_lshrrev_b32 v12, 2, v1 // (op 577) 
v_lshl_add_u32 v13, s16, 5, v12 // (op 577) 

v_and_b32 v12, -4, v1 // (op 577) 
v_sub_u32 v16, v1, v12 // (op 577) 


s_waitcnt lgkmcnt(0)
v_mul_lo_u32 v12, v13, s20 // (op 577) 
v_add_u32 v12, v12, v16 // (op 577) 


s_mul_i32 s1, s20, s7 // (op 579) most significant: low of low * high
s_mul_i32 s14, s21, s6 // (op 579) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 579) least significant
s_mul_hi_u32 s15, s20, s6 // (op 579) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 579) most significant: sum
s_add_u32 s1, s1, s15 // (op 579) most significant: sum


s_mov_b32 s48, s22 // (op 579) reg expression
s_mov_b32 s49, s23 // (op 579) reg expression
s_mov_b32 s50, s0 // (op 579) reg expression
s_mov_b32 s51, 131072 // (op 579) call()

v_lshrrev_b32 v13, 2, v1 // (op 596) 
v_lshl_add_u32 v16, s16, 5, v13 // (op 596) 

v_and_b32 v13, -4, v1 // (op 596) 
v_sub_u32 v17, v1, v13 // (op 596) 

v_mul_lo_u32 v13, v16, s20 // (op 596) 
v_add_u32 v13, v13, v17 // (op 596) 


s_mul_i32 s1, s20, s7 // (op 598) most significant: low of low * high
s_mul_i32 s14, s21, s6 // (op 598) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 598) least significant
s_mul_hi_u32 s15, s20, s6 // (op 598) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 598) most significant: sum
s_add_u32 s1, s1, s15 // (op 598) most significant: sum


s_mov_b32 s48, s22 // (op 598) reg expression
s_mov_b32 s49, s23 // (op 598) reg expression
s_mov_b32 s50, s0 // (op 598) reg expression
s_mov_b32 s51, 131072 // (op 598) call()

v_and_b32 v16, 15, v1 // (op 641) 
v_lshl_add_u32 v17, s17, 4, v16 // (op 641) 

v_mul_lo_u32 v16, v17, s24 // (op 641) 

v_and_b32 v17, 63, v1 // (op 641) 
v_lshrrev_b32 v18, 4, v17 // (op 641) 

v_add_u32 v17, v18, v16 // (op 641) 


s_mul_i32 s1, s24, s13 // (op 643) most significant: low of low * high
s_mul_i32 s14, s25, s12 // (op 643) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 643) least significant
s_mul_hi_u32 s15, s24, s12 // (op 643) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 643) most significant: sum
s_add_u32 s1, s1, s15 // (op 643) most significant: sum


s_mov_b32 s52, s26 // (op 643) reg expression
s_mov_b32 s53, s27 // (op 643) reg expression
s_mov_b32 s54, s0 // (op 643) reg expression
s_mov_b32 s55, 131072 // (op 643) call()

v_and_b32 v16, 15, v1 // (op 660) 
v_lshl_add_u32 v18, s17, 4, v16 // (op 660) 

v_mul_lo_u32 v16, v18, s24 // (op 660) 

v_and_b32 v18, 63, v1 // (op 660) 
v_lshrrev_b32 v19, 4, v18 // (op 660) 

v_add_u32 v18, v19, v16 // (op 660) 


s_mul_i32 s1, s24, s13 // (op 662) most significant: low of low * high
s_mul_i32 s14, s25, s12 // (op 662) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 662) least significant
s_mul_hi_u32 s15, s24, s12 // (op 662) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 662) most significant: sum
s_add_u32 s1, s1, s15 // (op 662) most significant: sum


s_mov_b32 s52, s26 // (op 662) reg expression
s_mov_b32 s53, s27 // (op 662) reg expression
s_mov_b32 s54, s0 // (op 662) reg expression
s_mov_b32 s55, 131072 // (op 662) call()




buffer_load_ubyte v16, v17, s[52:55], 0 offen offset: 0 // (op 337) Load value



buffer_load_dwordx3 v[20:22], v14, s[44:47], 0 offen // (op 341) Load value



buffer_load_dwordx4 v[24:27], v10, s[40:43], 0 offen // (op 344) Load value
buffer_load_dwordx2 v[28:29], v10, s[40:43], 0 offen offset:16 // (op 344) Load value



buffer_load_ubyte v19, v12, s[48:51], 0 offen offset: 0 // (op 347) Load value



buffer_load_ubyte v23, v17, s[52:55], 0 offen offset:4 // (op 351) Load value




buffer_load_dwordx3 v[30:32], v14, s[44:47], 0 offen offset:96 // (op 355) Load value




buffer_load_dwordx4 v[34:37], v10, s[40:43], 0 offen offset:96 // (op 358) Load value
buffer_load_dwordx2 v[38:39], v10, s[40:43], 0 offen offset:112 // (op 358) Load value




buffer_load_ubyte v10, v12, s[48:51], 0 offen offset:4 // (op 361) Load value

v_and_b32 v12, -8, v1 // (op 780) 
v_sub_u32 v14, v1, v12 // (op 780) 

v_lshlrev_b32 v12, 4, v14 // (op 780) 

v_lshrrev_b32 v14, 3, v1 // (op 780) 
v_lshl_add_u32 v17, v14, 7, v12 // (op 780) 


v_mul_lo_u32 v12, v17, 6 // (op 780) 

v_lshrrev_b32 v14, 3, v12 // (op 780) 




s_waitcnt vmcnt(8)
ds_write_b96 v14, v[20:22] // (op 363) Store local data 

v_and_b32 v12, -4, v1 // (op 727) 
v_sub_u32 v14, v1, v12 // (op 727) 

v_lshlrev_b32 v12, 5, v14 // (op 727) 

v_lshrrev_b32 v14, 2, v1 // (op 727) 
v_lshl_add_u32 v17, v14, 7, v12 // (op 727) 


v_mul_lo_u32 v12, v17, 6 // (op 727) 

v_lshrrev_b32 v14, 3, v12 // (op 727) 




s_waitcnt vmcnt(7)
ds_write_b128 v14, v[24:27] offset:1536 // (op 364) Store local data 
s_waitcnt vmcnt(6)
ds_write_b64 v14, v[28:29] offset:1552 // (op 364) Store local data 

v_and_b32 v12, -4, v1 // (op 833) 
v_sub_u32 v14, v1, v12 // (op 833) 

v_and_b32 v12, 4294967292, v1 // (op 833) 
v_add_u32 v17, v12, v14 // (op 833) 





s_waitcnt vmcnt(5)
ds_write_b8 v17, v19 offset:4608 // (op 365) Store local data 

v_and_b32 v12, -4, v1 // (op 698) 
v_sub_u32 v14, v1, v12 // (op 698) 

v_lshlrev_b32 v12, 5, v14 // (op 698) 

v_lshrrev_b32 v14, 2, v1 // (op 698) 
v_lshl_add_u32 v17, v14, 7, v12 // (op 698) 


v_mul_lo_u32 v12, v17, 6 // (op 698) 

v_lshrrev_b32 v14, 3, v12 // (op 698) 

v_and_b32 v12, -8, v1 // (op 751) 
v_sub_u32 v17, v1, v12 // (op 751) 

v_lshlrev_b32 v12, 4, v17 // (op 751) 

v_lshrrev_b32 v17, 3, v1 // (op 751) 
v_lshl_add_u32 v33, v17, 7, v12 // (op 751) 


v_mul_lo_u32 v12, v33, 6 // (op 751) 

v_lshrrev_b32 v17, 3, v12 // (op 751) 

v_and_b32 v12, -4, v1 // (op 804) 
v_sub_u32 v33, v1, v12 // (op 804) 

v_and_b32 v12, 4294967292, v1 // (op 804) 
v_add_u32 v40, v12, v33 // (op 804) 


v_and_b32 v12, -4, v1 // (op 891) 
v_sub_u32 v33, v1, v12 // (op 891) 

v_lshlrev_b32 v12, 5, v33 // (op 891) 

v_lshrrev_b32 v33, 2, v1 // (op 891) 
v_lshl_add_u32 v41, v33, 7, v12 // (op 891) 


v_mul_lo_u32 v12, v41, 6 // (op 891) 

v_lshrrev_b32 v33, 3, v12 // (op 891) 

v_and_b32 v12, -8, v1 // (op 915) 
v_sub_u32 v41, v1, v12 // (op 915) 

v_lshlrev_b32 v12, 4, v41 // (op 915) 

v_lshrrev_b32 v41, 3, v1 // (op 915) 
v_lshl_add_u32 v42, v41, 7, v12 // (op 915) 


v_mul_lo_u32 v12, v42, 6 // (op 915) 

v_lshrrev_b32 v41, 3, v12 // (op 915) 

v_and_b32 v12, -4, v1 // (op 939) 
v_sub_u32 v42, v1, v12 // (op 939) 

v_and_b32 v12, 4294967292, v1 // (op 939) 
v_add_u32 v43, v12, v42 // (op 939) 


s_mov_b32 s0, 0 // (op 55) call()
s_ashr_i64 s[14:15], s[28:29], 7 // (op 54) 
s_mov_b32 s1, s14 // (op 54) convert

s_and_b32 s14, s1, 4294967294 // (op 54) 

s_cmp_lt_u32 s0, s14 // (op 54) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoop_54 // (op 54) Condition: Top (jump to RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoop_54 if false)
RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoop_54:
s_add_u32 s0, s0, 2 // (op 56) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 984) 



ds_read_b128 v[44:47], v3 offset:1536 // (op 65) Load local data 
ds_read_b64 v[48:49], v3 offset:1552 // (op 65) Load local data 
s_barrier  // (op 997) 



ds_read_b128 v[50:53], v4 // (op 70) Load local data 
ds_read_b64 v[54:55], v4 offset:16 // (op 70) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1011) 



ds_read_u8 v12, v0 offset:4608 // (op 75) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[44:49], v[50:55], a[0:3], v12, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 82) 





v_mov_b32 v12, v18 // (op 80) 

buffer_load_ubyte v16, v12, s[52:55], 0 offen offset:8 // (op 80) Load value


v_mov_b32 v42, v15 // (op 10) 

buffer_load_dwordx3 v[20:22], v42, s[44:47], 0 offen offset:192 // (op 10) Load value


v_mov_b32 v44, v11 // (op 4) 

buffer_load_dwordx4 v[24:27], v44, s[40:43], 0 offen offset:192 // (op 4) Load value
buffer_load_dwordx2 v[28:29], v44, s[40:43], 0 offen offset:208 // (op 4) Load value


v_mov_b32 v45, v13 // (op 17) 

buffer_load_ubyte v19, v45, s[48:51], 0 offen offset:8 // (op 17) Load value
s_barrier  // (op 1034) 



s_waitcnt vmcnt(8)
ds_write_b96 v41, v[30:32] offset:4736 // (op 235) Store local data 
s_barrier  // (op 1028) 



s_waitcnt vmcnt(7)
ds_write_b128 v33, v[34:37] offset:6272 // (op 225) Store local data 
s_waitcnt vmcnt(6)
ds_write_b64 v33, v[38:39] offset:6288 // (op 225) Store local data 
s_barrier  // (op 1040) 



s_waitcnt vmcnt(5)
ds_write_b8 v43, v10 offset:9344 // (op 236) Store local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1025) 



ds_read_b128 v[46:49], v6 offset:6272 // (op 227) Load local data 
ds_read_b64 v[50:51], v6 offset:6288 // (op 227) Load local data 
s_barrier  // (op 1031) 
s_barrier  // (op 1091) 



ds_read_b128 v[52:55], v7 offset:4736 // (op 229) Load local data 
ds_read_b64 v[56:57], v7 offset:4752 // (op 229) Load local data 
s_barrier  // (op 1097) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1037) 



ds_read_u8 v58, v5 offset:9344 // (op 231) Load local data 
s_barrier  // (op 1103) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[46:51], v[52:57], a[0:3], v58, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 234) 






buffer_load_ubyte v23, v12, s[52:55], 0 offen offset:12 // (op 233) Load value




v_add_u32 v18, 8, v18 // (op 652) 
buffer_load_dwordx3 v[30:32], v42, s[44:47], 0 offen offset:288 // (op 237) Load value




v_add_u32 v15, 192, v15 // (op 524) 
buffer_load_dwordx4 v[34:37], v44, s[40:43], 0 offen offset:288 // (op 224) Load value
buffer_load_dwordx2 v[38:39], v44, s[40:43], 0 offen offset:304 // (op 224) Load value




v_add_u32 v11, 192, v11 // (op 460) 
buffer_load_ubyte v10, v45, s[48:51], 0 offen offset:12 // (op 238) Load value

s_barrier  // (op 1000) 
v_add_u32 v13, 8, v13 // (op 588) 



s_waitcnt vmcnt(8)
ds_write_b96 v17, v[20:22] // (op 13) Store local data 
s_barrier  // (op 987) 



s_waitcnt vmcnt(7)
ds_write_b128 v14, v[24:27] offset:1536 // (op 6) Store local data 
s_waitcnt vmcnt(6)
ds_write_b64 v14, v[28:29] offset:1552 // (op 6) Store local data 
s_barrier  // (op 1014) 



s_waitcnt vmcnt(5)
ds_write_b8 v40, v19 offset:4608 // (op 21) Store local data 
s_ashr_i64 s[14:15], s[28:29], 7 // (op 54) 
s_mov_b32 s1, s14 // (op 54) convert

s_and_b32 s14, s1, 4294967294 // (op 54) 

s_cmp_lt_u32 s0, s14 // (op 54) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoop_54 // (op 54) Condition: Bottom (jump to RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoop_54 if true)
RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoop_54:















v_lshrrev_b32 v33, 2, v1 // (op 423) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 423) 
v_lshl_add_u32 v11, s16, 5, v33 // (op 423) 

s_lshr_b32 s14, s0, 1 // (op 423) 

v_and_b32 v33, -4, v1 // (op 423) 
v_sub_u32 v12, v1, v33 // (op 423) 

v_lshlrev_b32 v33, 5, v12 // (op 423) 

v_lshl_add_u32 v12, s14, 8, v33 // (op 423) 


v_mov_b32 v25, 0 // (op 423) convert
v_mov_b32 v24, v12 // (op 423) convert

v_mul_lo_u32 v13, v11, s3 // (op 423) most significant: low of low * high
v_mul_hi_u32 v33, v11, s2 // (op 423) most significant: high of low * low
v_mul_lo_u32 v12, v11, s2 // (op 423) least significant: low of low * low
v_add_u32 v13, v13, v33 // (op 423) most significant: sum

v_add_co_u32 v12, s[0:1], v12, v24 // (op 423) least significant half
v_addc_co_u32 v13, s[0:1], v13, v25, s[0:1] // (op 423) most significant half



v_mul_lo_u32 v15, v13, 6 // (op 423) most significant: low of high * low
v_mul_hi_u32 v11, v12, 6 // (op 423) most significant: high of low * low
v_mul_lo_u32 v14, v12, 6 // (op 423) least significant: low of low * low
v_add_u32 v15, v15, v11 // (op 423) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 423) 

v_mov_b32 v11, v12 // (op 423) convert

s_mul_i32 s1, s7, 6 // (op 425) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 425) least significant
s_mul_hi_u32 s14, s6, 6 // (op 425) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 425) most significant: sum

s_mul_i32 s15, s2, s1 // (op 425) most significant: low of low * high
s_mul_i32 s18, s3, s0 // (op 425) most significant: low of high * low
s_mul_i32 s14, s2, s0 // (op 425) least significant
s_mul_hi_u32 s19, s2, s0 // (op 425) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 425) most significant: sum
s_add_u32 s15, s15, s19 // (op 425) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 425) 

s_mov_b32 s40, s4 // (op 425) reg expression
s_mov_b32 s41, s5 // (op 425) reg expression
s_mov_b32 s42, s0 // (op 425) reg expression
s_mov_b32 s43, 131072 // (op 425) call()

v_and_b32 v33, -8, v1 // (op 490) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 490) 
v_sub_u32 v12, v1, v33 // (op 490) 

s_lshr_b32 s14, s0, 1 // (op 490) 

v_lshlrev_b32 v33, 4, v12 // (op 490) 

v_lshl_add_u32 v12, s14, 8, v33 // (op 490) 


v_mov_b32 v25, 0 // (op 490) convert
v_mov_b32 v24, v12 // (op 490) convert

v_lshrrev_b32 v33, 3, v1 // (op 490) 
v_lshl_add_u32 v12, s17, 4, v33 // (op 490) 

v_mul_lo_u32 v27, v12, s9 // (op 490) most significant: low of low * high
v_mul_hi_u32 v33, v12, s8 // (op 490) most significant: high of low * low
v_mul_lo_u32 v26, v12, s8 // (op 490) least significant: low of low * low
v_add_u32 v27, v27, v33 // (op 490) most significant: sum


v_add_co_u32 v28, s[0:1], v24, v26 // (op 490) least significant half
v_addc_co_u32 v29, s[0:1], v25, v27, s[0:1] // (op 490) most significant half



v_mul_lo_u32 v13, v29, 6 // (op 490) most significant: low of high * low
v_mul_hi_u32 v33, v28, 6 // (op 490) most significant: high of low * low
v_mul_lo_u32 v12, v28, 6 // (op 490) least significant: low of low * low
v_add_u32 v13, v13, v33 // (op 490) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 490) 

v_mov_b32 v33, v14 // (op 490) convert

s_mul_i32 s1, s13, 6 // (op 492) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 492) least significant
s_mul_hi_u32 s14, s12, 6 // (op 492) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 492) most significant: sum

s_mul_i32 s15, s8, s1 // (op 492) most significant: low of low * high
s_mul_i32 s18, s9, s0 // (op 492) most significant: low of high * low
s_mul_i32 s14, s8, s0 // (op 492) least significant
s_mul_hi_u32 s19, s8, s0 // (op 492) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 492) most significant: sum
s_add_u32 s15, s15, s19 // (op 492) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 492) 

s_mov_b32 s44, s10 // (op 492) reg expression
s_mov_b32 s45, s11 // (op 492) reg expression
s_mov_b32 s46, s0 // (op 492) reg expression
s_mov_b32 s47, 131072 // (op 492) call()

v_lshrrev_b32 v12, 2, v1 // (op 554) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 554) 
v_lshl_add_u32 v13, s16, 5, v12 // (op 554) 

s_lshr_b32 s14, s0, 1 // (op 554) 

v_and_b32 v12, -4, v1 // (op 554) 
v_sub_u32 v14, v1, v12 // (op 554) 

v_lshl_add_u32 v12, s14, 3, v14 // (op 554) 


v_mul_lo_u32 v14, v13, s20 // (op 554) 
v_add_u32 v14, v14, v12 // (op 554) 


s_mul_i32 s1, s20, s7 // (op 556) most significant: low of low * high
s_mul_i32 s14, s21, s6 // (op 556) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 556) least significant
s_mul_hi_u32 s15, s20, s6 // (op 556) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 556) most significant: sum
s_add_u32 s1, s1, s15 // (op 556) most significant: sum


s_mov_b32 s48, s22 // (op 556) reg expression
s_mov_b32 s49, s23 // (op 556) reg expression
s_mov_b32 s50, s0 // (op 556) reg expression
s_mov_b32 s51, 131072 // (op 556) call()

v_and_b32 v12, 63, v1 // (op 618) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 618) 
v_lshrrev_b32 v13, 4, v12 // (op 618) 

s_lshr_b32 s14, s0, 1 // (op 618) 

v_and_b32 v12, 15, v1 // (op 618) 
v_lshl_add_u32 v15, s14, 3, v13 // (op 618) 


v_lshl_add_u32 v13, s17, 4, v12 // (op 618) 

v_mul_lo_u32 v12, v13, s24 // (op 618) 

v_add_u32 v13, v15, v12 // (op 618) 


s_mul_i32 s1, s24, s13 // (op 620) most significant: low of low * high
s_mul_i32 s14, s25, s12 // (op 620) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 620) least significant
s_mul_hi_u32 s15, s24, s12 // (op 620) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 620) most significant: sum
s_add_u32 s1, s1, s15 // (op 620) most significant: sum


s_mov_b32 s52, s26 // (op 620) reg expression
s_mov_b32 s53, s27 // (op 620) reg expression
s_mov_b32 s54, s0 // (op 620) reg expression
s_mov_b32 s55, 131072 // (op 620) call()

v_and_b32 v12, -4, v1 // (op 711) 
v_sub_u32 v15, v1, v12 // (op 711) 

v_lshlrev_b32 v12, 5, v15 // (op 711) 

v_lshrrev_b32 v15, 2, v1 // (op 711) 
v_lshl_add_u32 v16, v15, 7, v12 // (op 711) 


v_mul_lo_u32 v12, v16, 6 // (op 711) 

v_lshrrev_b32 v15, 3, v12 // (op 711) 

v_and_b32 v12, -8, v1 // (op 764) 
v_sub_u32 v16, v1, v12 // (op 764) 

v_lshlrev_b32 v12, 4, v16 // (op 764) 

v_lshrrev_b32 v16, 3, v1 // (op 764) 
v_lshl_add_u32 v17, v16, 7, v12 // (op 764) 


v_mul_lo_u32 v12, v17, 6 // (op 764) 

v_lshrrev_b32 v16, 3, v12 // (op 764) 

v_and_b32 v12, -4, v1 // (op 817) 
v_sub_u32 v17, v1, v12 // (op 817) 

v_and_b32 v12, 4294967292, v1 // (op 817) 
v_add_u32 v18, v12, v17 // (op 817) 


s_ashr_i64 s[0:1], s[28:29], 7 // (op 162) 
s_and_b32 s14, s0, 4294967294 // (op 162) 

s_ashr_i64 s[0:1], s[28:29], 7 // (op 161) 
s_mov_b32 s15, s0 // (op 161) convert

s_cmp_lt_u32 s14, s15 // (op 161) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoopTail_161 // (op 161) Condition: Top (jump to RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoopTail_161 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoopTail_161:


s_add_u32 s14, s14, 1 // (op 163) 
v_mov_b32 v10, v11 // (op 166) 

buffer_load_dwordx4 v[20:23], v10, s[40:43], 0 offen // (op 166) Load value
buffer_load_dwordx2 v[24:25], v10, s[40:43], 0 offen offset:16 // (op 166) Load value

v_add_u32 v11, 96, v11 // (op 413) 


s_barrier  // (op 990) 
v_mov_b32 v10, v33 // (op 179) 




buffer_load_dwordx3 v[26:28], v10, s[44:47], 0 offen // (op 179) Load value

v_add_u32 v33, 96, v33 // (op 482) 
s_waitcnt vmcnt(2)
ds_write_b128 v15, v[20:23] offset:1536 // (op 167) Store local data 


v_mov_b32 v10, v14 // (op 180) 

buffer_load_ubyte v12, v10, s[48:51], 0 offen offset: 0 // (op 180) Load value

v_add_u32 v14, 4, v14 // (op 546) 
s_waitcnt vmcnt(2)
ds_write_b64 v15, v[24:25] offset:1552 // (op 167) Store local data 

s_barrier  // (op 1003) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 993) 






ds_read_b128 v[20:23], v3 offset:1536 // (op 169) Load local data 
ds_read_b64 v[24:25], v3 offset:1552 // (op 169) Load local data 
s_waitcnt vmcnt(1)
ds_write_b96 v16, v[26:28] // (op 177) Store local data 

s_barrier  // (op 1017) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1007) 






ds_read_b128 v[26:29], v4 // (op 171) Load local data 
ds_read_b64 v[30:31], v4 offset:16 // (op 171) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v18, v12 offset:4608 // (op 178) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 1021) 



ds_read_u8 v10, v0 offset:4608 // (op 173) Load local data 


v_mov_b32 v12, v13 // (op 175) 

buffer_load_ubyte v17, v12, s[52:55], 0 offen offset: 0 // (op 175) Load value

v_add_u32 v13, 4, v13 // (op 610) 
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:25], v[26:31], a[0:3], v10, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 176) 




s_ashr_i64 s[0:1], s[28:29], 7 // (op 161) 
s_mov_b32 s15, s0 // (op 161) convert

s_cmp_lt_u32 s14, s15 // (op 161) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoopTail_161 // (op 161) Condition: Bottom (jump to RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoopTail_161 if true)
RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoopTail_161:






















s_barrier  // (op 1085) 






v_cmp_eq_f32 vcc, 0.00000, v8 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalFalse_0_beta_123
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

v_mul_lo_u32 v4, v3, s30 // (op 859) 

v_add_lshl_u32 v3, v0, v4, 2 // (op 859) 


s_mul_i32 s15, s30, s13 // (op 861) most significant: low of low * high
s_mul_i32 s0, s31, s12 // (op 861) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 861) least significant
s_mul_hi_u32 s1, s30, s12 // (op 861) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 861) most significant: sum
s_add_u32 s15, s15, s1 // (op 861) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 2 // (op 861) 

s_mov_b32 s8, s32 // (op 861) reg expression
s_mov_b32 s9, s33 // (op 861) reg expression
s_mov_b32 s10, s0 // (op 861) reg expression
s_mov_b32 s11, 131072 // (op 861) call()




buffer_store_dwordx4 v[10:13], v3, s[8:11], 0 offen // (op 52) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalBottom_0_beta_123
RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalFalse_0_beta_123:
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


s_mul_i32 s15, s34, s13 // (op 683) most significant: low of low * high
s_mul_i32 s0, s35, s12 // (op 683) most significant: low of high * low
s_mul_i32 s14, s34, s12 // (op 683) least significant
s_mul_hi_u32 s1, s34, s12 // (op 683) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 683) most significant: sum
s_add_u32 s15, s15, s1 // (op 683) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 2 // (op 683) 

s_mov_b32 s4, s36 // (op 683) reg expression
s_mov_b32 s5, s37 // (op 683) reg expression
s_mov_b32 s6, s0 // (op 683) reg expression
s_mov_b32 s7, 131072 // (op 683) call()





buffer_load_dwordx4 v[4:7], v3, s[4:7], 0 offen // (op 135) Load value



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

v_mul_lo_u32 v10, v3, s30 // (op 876) 

v_add_lshl_u32 v3, v0, v10, 2 // (op 876) 


s_mul_i32 s15, s30, s13 // (op 878) most significant: low of low * high
s_mul_i32 s0, s31, s12 // (op 878) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 878) least significant
s_mul_hi_u32 s1, s30, s12 // (op 878) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 878) most significant: sum
s_add_u32 s15, s15, s1 // (op 878) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 2 // (op 878) 

s_mov_b32 s8, s32 // (op 878) reg expression
s_mov_b32 s9, s33 // (op 878) reg expression
s_mov_b32 s10, s0 // (op 878) reg expression
s_mov_b32 s11, 131072 // (op 878) call()




buffer_store_dwordx4 v[4:7], v3, s[8:11], 0 offen // (op 134) Store value





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalBottom_0_beta_123:




s_endpgm  // End of RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2
.LRR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_end:
.size RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2, .LRR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_end-RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2
  .amdhsa_next_free_vgpr 64
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 9472 // lds bytes
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
  ProblemType:
    OperationType: GEMM
    DataType: F6
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
    macrotile: [32, 16, 128]
    threads: [128, 1, 1]
    grid: [TilesX, TilesY, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [2, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2
    .symbol: RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 9472
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 56
    .vgpr_count: 59
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
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: FP6, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP6, pointerType: PointerGlobal}
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

