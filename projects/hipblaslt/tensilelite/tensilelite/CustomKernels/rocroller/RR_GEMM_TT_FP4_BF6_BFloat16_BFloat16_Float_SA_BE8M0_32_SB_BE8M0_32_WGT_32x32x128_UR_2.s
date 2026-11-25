// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2
.p2align 8
.type RR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2,@function
RR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_exec_begin:


s_load_dwordx2 s[20:21], s[0:1], 56 // Load scalar value
s_load_dwordx2 s[22:23], s[0:1], 64 // Load scalar value
s_load_dwordx2 s[24:25], s[0:1], 72 // Load scalar value
s_load_dwordx2 s[26:27], s[0:1], 80 // Load scalar value
s_load_dwordx2 s[28:29], s[0:1], 88 // Load scalar value
s_load_dwordx2 s[30:31], s[0:1], 96 // Load scalar value
s_load_dwordx2 s[32:33], s[0:1], 104 // Load scalar value
s_load_dwordx2 s[34:35], s[0:1], 112 // Load scalar value
s_load_dwordx2 s[36:37], s[0:1], 120 // Load scalar value
s_load_dwordx2 s[38:39], s[0:1], 128 // Load scalar value

v_bfe_u32 v1, v0, 0, 10
v_bfe_u32 v2, v0, 10, 10


v_lshrrev_b32 v0, 7, v1 // (op 712) 
v_and_b32 v3, 15, v1 // (op 712) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 712) 


v_and_b32 v0, 63, v1 // (op 712) 
v_lshrrev_b32 v3, 4, v0 // (op 712) 

v_lshlrev_b32 v0, 5, v3 // (op 712) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 712) 


v_lshlrev_b32 v0, 2, v3 // (op 712) 

v_lshrrev_b32 v3, 3, v0 // (op 712) 

v_and_b32 v0, 63, v1 // (op 765) 
v_and_b32 v4, 48, v1 // (op 765) 
v_sub_u32 v5, v0, v4 // (op 765) 

v_lshrrev_b32 v4, 4, v0 // (op 765) 

v_lshl_add_u32 v0, v4, 5, v5 // (op 765) 


v_lshrrev_b32 v4, 6, v1 // (op 765) 
v_and_b32 v5, 1, v4 // (op 765) 

v_lshlrev_b32 v4, 4, v5 // (op 765) 

v_lshl_add_u32 v5, v0, 5, v4 // (op 765) 


v_mul_lo_u32 v0, v5, 6 // (op 765) 
v_lshrrev_b32 v4, 3, v0 // (op 765) 

v_lshrrev_b32 v0, 4, v5 // (op 765) 

v_lshlrev_b32 v5, 2, v0 // (op 765) 

v_add_u32 v0, v4, v5 // (op 765) 


v_lshrrev_b32 v4, 7, v1 // (op 823) 
v_and_b32 v5, 15, v1 // (op 823) 
v_lshl_add_u32 v6, v4, 4, v5 // (op 823) 


v_and_b32 v4, 63, v1 // (op 823) 
v_lshrrev_b32 v5, 4, v4 // (op 823) 

v_lshlrev_b32 v4, 5, v5 // (op 823) 

v_lshl_add_u32 v5, v6, 7, v4 // (op 823) 


v_lshlrev_b32 v4, 2, v5 // (op 823) 

v_lshrrev_b32 v5, 3, v4 // (op 823) 

v_and_b32 v4, 63, v1 // (op 847) 
v_and_b32 v6, 48, v1 // (op 847) 
v_sub_u32 v7, v4, v6 // (op 847) 

v_lshrrev_b32 v6, 4, v4 // (op 847) 

v_lshl_add_u32 v4, v6, 5, v7 // (op 847) 


v_lshrrev_b32 v6, 6, v1 // (op 847) 
v_and_b32 v7, 1, v6 // (op 847) 

v_lshlrev_b32 v6, 4, v7 // (op 847) 

v_lshl_add_u32 v7, v4, 5, v6 // (op 847) 


v_mul_lo_u32 v4, v7, 6 // (op 847) 
v_lshrrev_b32 v6, 3, v4 // (op 847) 

v_lshrrev_b32 v4, 4, v7 // (op 847) 

v_lshlrev_b32 v7, 2, v4 // (op 847) 

v_add_u32 v4, v6, v7 // (op 847) 


v_mov_b32 v6, s14 // (op 2) Move value
v_mov_b32 v7, s15 // (op 35) Move value


v_accvgpr_write a0, 0.00000 // (op 57) call()
v_accvgpr_write a1, 0.00000 // (op 57) call()
v_accvgpr_write a2, 0.00000 // (op 57) call()
v_accvgpr_write a3, 0.00000 // (op 57) call()
v_lshrrev_b32 v8, 3, v1 // (op 422) 
v_lshl_add_u32 v9, s16, 5, v8 // (op 422) 

v_and_b32 v8, -8, v1 // (op 422) 
v_sub_u32 v10, v1, v8 // (op 422) 

v_lshlrev_b32 v8, 4, v10 // (op 422) 

v_mov_b32 v11, 0 // (op 422) convert
v_mov_b32 v10, v8 // (op 422) convert

v_mul_lo_u32 v13, v9, s3 // (op 422) most significant: low of low * high
v_mul_hi_u32 v8, v9, s2 // (op 422) most significant: high of low * low
v_mul_lo_u32 v12, v9, s2 // (op 422) least significant: low of low * low
v_add_u32 v13, v13, v8 // (op 422) most significant: sum

v_add_co_u32 v12, s[0:1], v12, v10 // (op 422) least significant half
v_addc_co_u32 v13, s[0:1], v13, v11, s[0:1] // (op 422) most significant half



v_lshlrev_b64 v[8:9], 2, v[12:13] // (op 422) 

v_lshrrev_b64 v[10:11], 3, v[8:9] // (op 422) 

v_mov_b32 v8, v10 // (op 422) convert

s_mul_i32 s1, s2, s7 // (op 424) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 424) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 424) least significant
s_mul_hi_u32 s15, s2, s6 // (op 424) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 424) most significant: sum
s_add_u32 s1, s1, s15 // (op 424) most significant: sum


s_lshl_b64 s[14:15], s[0:1], 2 // (op 424) 

s_ashr_i64 s[0:1], s[14:15], 3 // (op 424) 

s_mov_b32 s40, s4 // (op 424) reg expression
s_mov_b32 s41, s5 // (op 424) reg expression
s_mov_b32 s42, s0 // (op 424) reg expression
s_mov_b32 s43, 131072 // (op 424) call()

v_lshrrev_b32 v9, 3, v1 // (op 441) 
v_lshl_add_u32 v10, s16, 5, v9 // (op 441) 

v_and_b32 v9, -8, v1 // (op 441) 
v_sub_u32 v11, v1, v9 // (op 441) 

v_lshlrev_b32 v9, 4, v11 // (op 441) 

v_mov_b32 v13, 0 // (op 441) convert
v_mov_b32 v12, v9 // (op 441) convert

v_mul_lo_u32 v15, v10, s3 // (op 441) most significant: low of low * high
v_mul_hi_u32 v9, v10, s2 // (op 441) most significant: high of low * low
v_mul_lo_u32 v14, v10, s2 // (op 441) least significant: low of low * low
v_add_u32 v15, v15, v9 // (op 441) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v12 // (op 441) least significant half
v_addc_co_u32 v15, s[0:1], v15, v13, s[0:1] // (op 441) most significant half



v_lshlrev_b64 v[12:13], 2, v[14:15] // (op 441) 

v_lshrrev_b64 v[10:11], 3, v[12:13] // (op 441) 

v_mov_b32 v9, v10 // (op 441) convert

s_mul_i32 s1, s2, s7 // (op 443) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 443) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 443) least significant
s_mul_hi_u32 s15, s2, s6 // (op 443) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 443) most significant: sum
s_add_u32 s1, s1, s15 // (op 443) most significant: sum


s_lshl_b64 s[14:15], s[0:1], 2 // (op 443) 

s_ashr_i64 s[0:1], s[14:15], 3 // (op 443) 

s_mov_b32 s40, s4 // (op 443) reg expression
s_mov_b32 s41, s5 // (op 443) reg expression
s_mov_b32 s42, s0 // (op 443) reg expression
s_mov_b32 s43, 131072 // (op 443) call()

v_and_b32 v10, -2, v1 // (op 486) 
v_sub_u32 v11, v1, v10 // (op 486) 

v_lshlrev_b32 v10, 4, v11 // (op 486) 

v_lshl_add_u32 v11, s17, 5, v10 // (op 486) 

v_mov_b32 v13, 0 // (op 486) convert
v_mov_b32 v12, v11 // (op 486) convert

v_lshrrev_b32 v10, 1, v1 // (op 486) 
v_mul_lo_u32 v15, v10, s9 // (op 486) most significant: low of low * high
v_mul_hi_u32 v11, v10, s8 // (op 486) most significant: high of low * low
v_mul_lo_u32 v14, v10, s8 // (op 486) least significant: low of low * low
v_add_u32 v15, v15, v11 // (op 486) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v12 // (op 486) least significant half
v_addc_co_u32 v15, s[0:1], v15, v13, s[0:1] // (op 486) most significant half



v_mul_lo_u32 v11, v15, 6 // (op 486) most significant: low of high * low
v_mul_hi_u32 v12, v14, 6 // (op 486) most significant: high of low * low
v_mul_lo_u32 v10, v14, 6 // (op 486) least significant: low of low * low
v_add_u32 v11, v11, v12 // (op 486) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 486) 

v_mov_b32 v10, v12 // (op 486) convert

s_mul_i32 s1, s13, 6 // (op 488) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 488) least significant
s_mul_hi_u32 s14, s12, 6 // (op 488) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 488) most significant: sum

s_mul_i32 s15, s8, s1 // (op 488) most significant: low of low * high
s_mul_i32 s18, s9, s0 // (op 488) most significant: low of high * low
s_mul_i32 s14, s8, s0 // (op 488) least significant
s_mul_hi_u32 s19, s8, s0 // (op 488) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 488) most significant: sum
s_add_u32 s15, s15, s19 // (op 488) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 488) 

s_mov_b32 s44, s10 // (op 488) reg expression
s_mov_b32 s45, s11 // (op 488) reg expression
s_mov_b32 s46, s0 // (op 488) reg expression
s_mov_b32 s47, 131072 // (op 488) call()

v_and_b32 v11, -2, v1 // (op 505) 
v_sub_u32 v12, v1, v11 // (op 505) 

v_lshlrev_b32 v11, 4, v12 // (op 505) 

v_lshl_add_u32 v12, s17, 5, v11 // (op 505) 

v_mov_b32 v15, 0 // (op 505) convert
v_mov_b32 v14, v12 // (op 505) convert

v_lshrrev_b32 v11, 1, v1 // (op 505) 
v_mul_lo_u32 v13, v11, s9 // (op 505) most significant: low of low * high
v_mul_hi_u32 v16, v11, s8 // (op 505) most significant: high of low * low
v_mul_lo_u32 v12, v11, s8 // (op 505) least significant: low of low * low
v_add_u32 v13, v13, v16 // (op 505) most significant: sum

v_add_co_u32 v12, s[0:1], v12, v14 // (op 505) least significant half
v_addc_co_u32 v13, s[0:1], v13, v15, s[0:1] // (op 505) most significant half



v_mul_lo_u32 v15, v13, 6 // (op 505) most significant: low of high * low
v_mul_hi_u32 v11, v12, 6 // (op 505) most significant: high of low * low
v_mul_lo_u32 v14, v12, 6 // (op 505) least significant: low of low * low
v_add_u32 v15, v15, v11 // (op 505) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 505) 

v_mov_b32 v11, v12 // (op 505) convert

s_mul_i32 s1, s13, 6 // (op 507) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 507) least significant
s_mul_hi_u32 s14, s12, 6 // (op 507) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 507) most significant: sum

s_mul_i32 s15, s8, s1 // (op 507) most significant: low of low * high
s_mul_i32 s18, s9, s0 // (op 507) most significant: low of high * low
s_mul_i32 s14, s8, s0 // (op 507) least significant
s_mul_hi_u32 s19, s8, s0 // (op 507) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 507) most significant: sum
s_add_u32 s15, s15, s19 // (op 507) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 507) 

s_mov_b32 s44, s10 // (op 507) reg expression
s_mov_b32 s45, s11 // (op 507) reg expression
s_mov_b32 s46, s0 // (op 507) reg expression
s_mov_b32 s47, 131072 // (op 507) call()

v_lshrrev_b32 v12, 7, v1 // (op 550) 
v_and_b32 v13, 15, v1 // (op 550) 
v_lshl_add_u32 v14, v12, 4, v13 // (op 550) 


v_lshl_add_u32 v12, s16, 5, v14 // (op 550) 

v_and_b32 v13, 63, v1 // (op 550) 
v_lshrrev_b32 v14, 4, v13 // (op 550) 


s_waitcnt lgkmcnt(0)
v_mul_lo_u32 v13, v12, s20 // (op 550) 
v_add_u32 v13, v13, v14 // (op 550) 


s_mul_i32 s1, s20, s7 // (op 552) most significant: low of low * high
s_mul_i32 s14, s21, s6 // (op 552) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 552) least significant
s_mul_hi_u32 s15, s20, s6 // (op 552) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 552) most significant: sum
s_add_u32 s1, s1, s15 // (op 552) most significant: sum


s_mov_b32 s48, s22 // (op 552) reg expression
s_mov_b32 s49, s23 // (op 552) reg expression
s_mov_b32 s50, s0 // (op 552) reg expression
s_mov_b32 s51, 131072 // (op 552) call()

v_lshrrev_b32 v12, 7, v1 // (op 569) 
v_and_b32 v14, 15, v1 // (op 569) 
v_lshl_add_u32 v15, v12, 4, v14 // (op 569) 


v_lshl_add_u32 v12, s16, 5, v15 // (op 569) 

v_and_b32 v14, 63, v1 // (op 569) 
v_lshrrev_b32 v15, 4, v14 // (op 569) 

v_mul_lo_u32 v14, v12, s20 // (op 569) 
v_add_u32 v14, v14, v15 // (op 569) 


s_mul_i32 s1, s20, s7 // (op 571) most significant: low of low * high
s_mul_i32 s14, s21, s6 // (op 571) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 571) least significant
s_mul_hi_u32 s15, s20, s6 // (op 571) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 571) most significant: sum
s_add_u32 s1, s1, s15 // (op 571) most significant: sum


s_mov_b32 s48, s22 // (op 571) reg expression
s_mov_b32 s49, s23 // (op 571) reg expression
s_mov_b32 s50, s0 // (op 571) reg expression
s_mov_b32 s51, 131072 // (op 571) call()

v_lshrrev_b32 v12, 6, v1 // (op 614) 
v_and_b32 v15, 1, v12 // (op 614) 

v_and_b32 v12, 15, v1 // (op 614) 
v_lshl_add_u32 v16, v15, 4, v12 // (op 614) 


v_lshl_add_u32 v12, s17, 5, v16 // (op 614) 

v_and_b32 v15, 63, v1 // (op 614) 
v_lshrrev_b32 v16, 4, v15 // (op 614) 

v_mul_lo_u32 v15, v16, s24 // (op 614) 
v_add_u32 v15, v15, v12 // (op 614) 


s_mul_i32 s1, s24, s29 // (op 616) most significant: low of low * high
s_mul_i32 s14, s25, s28 // (op 616) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 616) least significant
s_mul_hi_u32 s15, s24, s28 // (op 616) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 616) most significant: sum
s_add_u32 s1, s1, s15 // (op 616) most significant: sum


s_mov_b32 s52, s26 // (op 616) reg expression
s_mov_b32 s53, s27 // (op 616) reg expression
s_mov_b32 s54, s0 // (op 616) reg expression
s_mov_b32 s55, 131072 // (op 616) call()

v_lshrrev_b32 v12, 6, v1 // (op 633) 
v_and_b32 v16, 1, v12 // (op 633) 

v_and_b32 v12, 15, v1 // (op 633) 
v_lshl_add_u32 v17, v16, 4, v12 // (op 633) 


v_lshl_add_u32 v12, s17, 5, v17 // (op 633) 

v_and_b32 v16, 63, v1 // (op 633) 
v_lshrrev_b32 v17, 4, v16 // (op 633) 

v_mul_lo_u32 v16, v17, s24 // (op 633) 
v_add_u32 v16, v16, v12 // (op 633) 


s_mul_i32 s1, s24, s29 // (op 635) most significant: low of low * high
s_mul_i32 s14, s25, s28 // (op 635) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 635) least significant
s_mul_hi_u32 s15, s24, s28 // (op 635) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 635) most significant: sum
s_add_u32 s1, s1, s15 // (op 635) most significant: sum


s_mov_b32 s52, s26 // (op 635) reg expression
s_mov_b32 s53, s27 // (op 635) reg expression
s_mov_b32 s54, s0 // (op 635) reg expression
s_mov_b32 s55, 131072 // (op 635) call()




buffer_load_ubyte v12, v13, s[48:51], 0 offen offset: 0 // (op 312) Load value

v_mov_b64 v[18:19], s[24:25] // (op 317) convert

v_mov_b32 v17, v15 // (op 317) reg expression
v_mov_b32 v20, v17 // (op 317) 
buffer_load_ubyte v21, v20, s[52:55], 0 offen offset: 0 // (op 317) Load value





v_mov_b32 v17, v10 // (op 321) reg expression

buffer_load_dwordx3 v[18:20], v17, s[44:47], 0 offen // (op 321) Load value




buffer_load_dwordx2 v[22:23], v8, s[40:43], 0 offen // (op 324) Load value



buffer_load_ubyte v17, v13, s[48:51], 0 offen offset:4 // (op 328) Load value


v_mov_b64 v[24:25], s[24:25] // (op 333) convert

s_lshl_b32 s0, s24, 2 // (op 333) 
v_add_u32 v13, v15, s0 // (op 333) 

v_mov_b32 v26, v13 // (op 333) 
buffer_load_ubyte v27, v26, s[52:55], 0 offen offset: 0 // (op 333) Load value






s_lshl_b64 s[0:1], s[8:9], 7 // (op 337) 
s_mul_i32 s15, s1, 6 // (op 337) most significant: low of high * low
s_mul_i32 s14, s0, 6 // (op 337) least significant
s_mul_hi_u32 s18, s0, 6 // (op 337) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 337) most significant: sum


s_lshr_b64 s[0:1], s[14:15], 3 // (op 337) 

v_add_u32 v13, v10, s0 // (op 337) 


buffer_load_dwordx3 v[24:26], v13, s[44:47], 0 offen // (op 337) Load value





buffer_load_dwordx2 v[28:29], v8, s[40:43], 0 offen offset:64 // (op 340) Load value

v_and_b32 v8, -2, v1 // (op 753) 
v_sub_u32 v10, v1, v8 // (op 753) 

v_lshlrev_b32 v8, 4, v10 // (op 753) 

v_lshrrev_b32 v10, 1, v1 // (op 753) 
v_lshl_add_u32 v13, v10, 5, v8 // (op 753) 


v_mul_lo_u32 v8, v13, 6 // (op 753) 
v_lshrrev_b32 v10, 3, v8 // (op 753) 

v_lshrrev_b32 v8, 4, v13 // (op 753) 

v_lshlrev_b32 v13, 2, v8 // (op 753) 

v_add_u32 v8, v10, v13 // (op 753) 





s_waitcnt vmcnt(5)
ds_write_b96 v8, v[18:20] // (op 342) Store local data 

v_and_b32 v8, -8, v1 // (op 700) 
v_sub_u32 v10, v1, v8 // (op 700) 

v_lshlrev_b32 v8, 4, v10 // (op 700) 

v_lshrrev_b32 v10, 3, v1 // (op 700) 
v_lshl_add_u32 v13, v10, 7, v8 // (op 700) 


v_lshlrev_b32 v8, 2, v13 // (op 700) 

v_lshrrev_b32 v10, 3, v8 // (op 700) 




s_waitcnt vmcnt(4)
ds_write_b64 v10, v[22:23] offset:4096 // (op 343) Store local data 

v_and_b32 v8, -8, v1 // (op 671) 
v_sub_u32 v10, v1, v8 // (op 671) 

v_lshlrev_b32 v8, 4, v10 // (op 671) 

v_lshrrev_b32 v10, 3, v1 // (op 671) 
v_lshl_add_u32 v13, v10, 7, v8 // (op 671) 


v_lshlrev_b32 v8, 2, v13 // (op 671) 

v_lshrrev_b32 v10, 3, v8 // (op 671) 

v_and_b32 v8, -2, v1 // (op 724) 
v_sub_u32 v13, v1, v8 // (op 724) 

v_lshlrev_b32 v8, 4, v13 // (op 724) 

v_lshrrev_b32 v13, 1, v1 // (op 724) 
v_lshl_add_u32 v15, v13, 5, v8 // (op 724) 


v_mul_lo_u32 v8, v15, 6 // (op 724) 
v_lshrrev_b32 v13, 3, v8 // (op 724) 

v_lshrrev_b32 v8, 4, v15 // (op 724) 

v_lshlrev_b32 v15, 2, v8 // (op 724) 

v_add_u32 v8, v13, v15 // (op 724) 


v_and_b32 v13, -8, v1 // (op 811) 
v_sub_u32 v15, v1, v13 // (op 811) 

v_lshlrev_b32 v13, 4, v15 // (op 811) 

v_lshrrev_b32 v15, 3, v1 // (op 811) 
v_lshl_add_u32 v30, v15, 7, v13 // (op 811) 


v_lshlrev_b32 v13, 2, v30 // (op 811) 

v_lshrrev_b32 v15, 3, v13 // (op 811) 

v_and_b32 v13, -2, v1 // (op 835) 
v_sub_u32 v30, v1, v13 // (op 835) 

v_lshlrev_b32 v13, 4, v30 // (op 835) 

v_lshrrev_b32 v30, 1, v1 // (op 835) 
v_lshl_add_u32 v31, v30, 5, v13 // (op 835) 


v_mul_lo_u32 v13, v31, 6 // (op 835) 
v_lshrrev_b32 v30, 3, v13 // (op 835) 

v_lshrrev_b32 v13, 4, v31 // (op 835) 

v_lshlrev_b32 v31, 2, v13 // (op 835) 

v_add_u32 v13, v30, v31 // (op 835) 


s_mov_b32 s0, 0 // (op 53) call()
s_ashr_i64 s[14:15], s[12:13], 7 // (op 52) 
s_mov_b32 s1, s14 // (op 52) convert

s_and_b32 s14, s1, 4294967294 // (op 52) 

s_cmp_lt_u32 s0, s14 // (op 52) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoop_52 // (op 52) Condition: Top (jump to RR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoop_52 if false)
RR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoop_52:
s_add_u32 s0, s0, 2 // (op 54) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 876) 



ds_read_b128 v[30:33], v3 offset:4096 // (op 62) Load local data 
s_barrier  // (op 889) 



ds_read_b96_tr_b6 v[34:36], v0 // (op 67) Transpose load local data 
ds_read_b96_tr_b6 v[37:39], v0 offset:512 // (op 67) Transpose load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[30:33], v[34:39], a[0:3], v12, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 78) 




v_mov_b32 v30, v14 // (op 72) 

buffer_load_ubyte v12, v30, s[48:51], 0 offen offset:8 // (op 72) Load value

v_mov_b64 v[32:33], s[24:25] // (op 76) convert

v_mov_b32 v31, v16 // (op 76) 
s_lshl_b32 s1, s24, 3 // (op 76) 
v_add_u32 v34, v31, s1 // (op 76) 

v_mov_b32 v35, v34 // (op 76) 
buffer_load_ubyte v21, v35, s[52:55], 0 offen offset: 0 // (op 76) Load value





v_mov_b32 v32, v11 // (op 10) 
s_lshl_b64 s[14:15], s[8:9], 7 // (op 10) 
s_mul_i32 s19, s15, 6 // (op 10) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 10) least significant
s_mul_hi_u32 s1, s14, 6 // (op 10) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 10) most significant: sum


s_lshr_b64 s[14:15], s[18:19], 3 // (op 10) 

s_lshl_b32 s1, s14, 1 // (op 10) 

v_add_u32 v33, v32, s1 // (op 10) 


buffer_load_dwordx3 v[18:20], v33, s[44:47], 0 offen // (op 10) Load value



v_mov_b32 v33, v9 // (op 4) 

buffer_load_dwordx2 v[22:23], v33, s[40:43], 0 offen offset:128 // (op 4) Load value
s_barrier  // (op 912) 



s_waitcnt vmcnt(5)
ds_write_b96 v13, v[24:26] offset:6144 // (op 222) Store local data 
s_barrier  // (op 906) 



s_waitcnt vmcnt(4)
ds_write_b64 v15, v[28:29] offset:10240 // (op 212) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 903) 



ds_read_b128 v[34:37], v5 offset:10240 // (op 214) Load local data 
s_barrier  // (op 909) 
s_barrier  // (op 963) 



ds_read_b96_tr_b6 v[38:40], v4 offset:6144 // (op 216) Transpose load local data 
ds_read_b96_tr_b6 v[41:43], v4 offset:6656 // (op 216) Transpose load local data 
s_barrier  // (op 969) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:37], v[38:43], a[0:3], v17, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 221) 





buffer_load_ubyte v17, v30, s[48:51], 0 offen offset:12 // (op 218) Load value


v_add_u32 v14, 8, v14 // (op 561) 
v_mov_b64 v[34:35], s[24:25] // (op 220) convert

s_lshl_b32 s1, s24, 2 // (op 220) 
s_mul_i32 s14, 3, s1 // (op 220) 

v_add_u32 v30, v31, s14 // (op 220) 

v_mov_b32 v36, v30 // (op 220) 
buffer_load_ubyte v27, v36, s[52:55], 0 offen offset: 0 // (op 220) Load value






s_lshl_b32 s1, s24, 3 // (op 625) 
v_add_u32 v16, v16, s1 // (op 625) 

s_lshl_b64 s[14:15], s[8:9], 7 // (op 223) 
s_mul_i32 s19, s15, 6 // (op 223) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 223) least significant
s_mul_hi_u32 s1, s14, 6 // (op 223) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 223) most significant: sum


s_lshr_b64 s[14:15], s[18:19], 3 // (op 223) 

s_mul_i32 s1, 3, s14 // (op 223) 

v_add_u32 v30, v32, s1 // (op 223) 


buffer_load_dwordx3 v[24:26], v30, s[44:47], 0 offen // (op 223) Load value





s_lshl_b64 s[14:15], s[8:9], 7 // (op 497) 
s_mul_i32 s19, s15, 6 // (op 497) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 497) least significant
s_mul_hi_u32 s1, s14, 6 // (op 497) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 497) most significant: sum


s_lshr_b64 s[14:15], s[18:19], 3 // (op 497) 

s_lshl_b32 s1, s14, 1 // (op 497) 

v_add_u32 v11, v11, s1 // (op 497) 

buffer_load_dwordx2 v[28:29], v33, s[40:43], 0 offen offset:192 // (op 211) Load value

s_barrier  // (op 892) 
v_add_u32 v9, 128, v9 // (op 433) 



s_waitcnt vmcnt(5)
ds_write_b96 v8, v[18:20] // (op 13) Store local data 
s_barrier  // (op 879) 



s_waitcnt vmcnt(4)
ds_write_b64 v10, v[22:23] offset:4096 // (op 6) Store local data 
s_ashr_i64 s[14:15], s[12:13], 7 // (op 52) 
s_mov_b32 s1, s14 // (op 52) convert

s_and_b32 s14, s1, 4294967294 // (op 52) 

s_cmp_lt_u32 s0, s14 // (op 52) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoop_52 // (op 52) Condition: Bottom (jump to RR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoop_52 if true)
RR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoop_52:













v_lshrrev_b32 v8, 3, v1 // (op 396) 
s_ashr_i64 s[0:1], s[12:13], 7 // (op 396) 
v_lshl_add_u32 v9, s16, 5, v8 // (op 396) 

s_lshr_b32 s14, s0, 1 // (op 396) 

v_and_b32 v8, -8, v1 // (op 396) 
v_sub_u32 v10, v1, v8 // (op 396) 

v_lshlrev_b32 v8, 4, v10 // (op 396) 

v_lshl_add_u32 v10, s14, 8, v8 // (op 396) 


v_mov_b32 v19, 0 // (op 396) convert
v_mov_b32 v18, v10 // (op 396) convert

v_mul_lo_u32 v11, v9, s3 // (op 396) most significant: low of low * high
v_mul_hi_u32 v8, v9, s2 // (op 396) most significant: high of low * low
v_mul_lo_u32 v10, v9, s2 // (op 396) least significant: low of low * low
v_add_u32 v11, v11, v8 // (op 396) most significant: sum

v_add_co_u32 v10, s[0:1], v10, v18 // (op 396) least significant half
v_addc_co_u32 v11, s[0:1], v11, v19, s[0:1] // (op 396) most significant half



v_lshlrev_b64 v[8:9], 2, v[10:11] // (op 396) 

v_lshrrev_b64 v[10:11], 3, v[8:9] // (op 396) 

v_mov_b32 v8, v10 // (op 396) convert

s_mul_i32 s1, s2, s7 // (op 398) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 398) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 398) least significant
s_mul_hi_u32 s15, s2, s6 // (op 398) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 398) most significant: sum
s_add_u32 s1, s1, s15 // (op 398) most significant: sum


s_lshl_b64 s[14:15], s[0:1], 2 // (op 398) 

s_ashr_i64 s[0:1], s[14:15], 3 // (op 398) 

s_mov_b32 s40, s4 // (op 398) reg expression
s_mov_b32 s41, s5 // (op 398) reg expression
s_mov_b32 s42, s0 // (op 398) reg expression
s_mov_b32 s43, 131072 // (op 398) call()

v_lshrrev_b32 v9, 1, v1 // (op 463) 
s_ashr_i64 s[0:1], s[12:13], 7 // (op 463) 
v_and_b32 v10, -2, v1 // (op 463) 
s_lshr_b32 s14, s0, 1 // (op 463) 

v_sub_u32 v11, v1, v10 // (op 463) 

v_lshl_add_u32 v10, s14, 8, v9 // (op 463) 


v_lshlrev_b32 v9, 4, v11 // (op 463) 

v_lshl_add_u32 v11, s17, 5, v9 // (op 463) 

v_mov_b32 v13, 0 // (op 463) convert
v_mov_b32 v12, v11 // (op 463) convert

v_mul_lo_u32 v15, v10, s9 // (op 463) most significant: low of low * high
v_mul_hi_u32 v9, v10, s8 // (op 463) most significant: high of low * low
v_mul_lo_u32 v14, v10, s8 // (op 463) least significant: low of low * low
v_add_u32 v15, v15, v9 // (op 463) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v12 // (op 463) least significant half
v_addc_co_u32 v15, s[0:1], v15, v13, s[0:1] // (op 463) most significant half



v_mul_lo_u32 v13, v15, 6 // (op 463) most significant: low of high * low
v_mul_hi_u32 v16, v14, 6 // (op 463) most significant: high of low * low
v_mul_lo_u32 v12, v14, 6 // (op 463) least significant: low of low * low
v_add_u32 v13, v13, v16 // (op 463) most significant: sum


v_lshrrev_b64 v[10:11], 3, v[12:13] // (op 463) 

v_mov_b32 v9, v10 // (op 463) convert

s_mul_i32 s1, s13, 6 // (op 465) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 465) least significant
s_mul_hi_u32 s14, s12, 6 // (op 465) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 465) most significant: sum

s_mul_i32 s15, s8, s1 // (op 465) most significant: low of low * high
s_mul_i32 s18, s9, s0 // (op 465) most significant: low of high * low
s_mul_i32 s14, s8, s0 // (op 465) least significant
s_mul_hi_u32 s19, s8, s0 // (op 465) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 465) most significant: sum
s_add_u32 s15, s15, s19 // (op 465) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 465) 

s_mov_b32 s44, s10 // (op 465) reg expression
s_mov_b32 s45, s11 // (op 465) reg expression
s_mov_b32 s46, s0 // (op 465) reg expression
s_mov_b32 s47, 131072 // (op 465) call()

v_lshrrev_b32 v10, 7, v1 // (op 527) 
s_ashr_i64 s[0:1], s[12:13], 7 // (op 527) 
v_and_b32 v11, 15, v1 // (op 527) 
s_lshr_b32 s14, s0, 1 // (op 527) 

v_lshl_add_u32 v12, v10, 4, v11 // (op 527) 


v_lshl_add_u32 v10, s16, 5, v12 // (op 527) 

v_and_b32 v11, 63, v1 // (op 527) 
v_lshrrev_b32 v12, 4, v11 // (op 527) 

v_lshl_add_u32 v11, s14, 3, v12 // (op 527) 


v_mul_lo_u32 v12, v10, s20 // (op 527) 
v_add_u32 v12, v12, v11 // (op 527) 


s_mul_i32 s1, s20, s7 // (op 529) most significant: low of low * high
s_mul_i32 s14, s21, s6 // (op 529) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 529) least significant
s_mul_hi_u32 s15, s20, s6 // (op 529) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 529) most significant: sum
s_add_u32 s1, s1, s15 // (op 529) most significant: sum


s_mov_b32 s48, s22 // (op 529) reg expression
s_mov_b32 s49, s23 // (op 529) reg expression
s_mov_b32 s50, s0 // (op 529) reg expression
s_mov_b32 s51, 131072 // (op 529) call()

v_and_b32 v10, 63, v1 // (op 591) 
s_ashr_i64 s[0:1], s[12:13], 7 // (op 591) 
v_lshrrev_b32 v11, 4, v10 // (op 591) 

s_lshr_b32 s14, s0, 1 // (op 591) 

v_lshrrev_b32 v10, 6, v1 // (op 591) 
v_lshl_add_u32 v13, s14, 3, v11 // (op 591) 


v_and_b32 v11, 1, v10 // (op 591) 

v_and_b32 v10, 15, v1 // (op 591) 
v_lshl_add_u32 v14, v11, 4, v10 // (op 591) 


v_lshl_add_u32 v10, s17, 5, v14 // (op 591) 

v_mul_lo_u32 v11, v13, s24 // (op 591) 
v_add_u32 v11, v11, v10 // (op 591) 


s_mul_i32 s1, s24, s29 // (op 593) most significant: low of low * high
s_mul_i32 s14, s25, s28 // (op 593) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 593) least significant
s_mul_hi_u32 s15, s24, s28 // (op 593) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 593) most significant: sum
s_add_u32 s1, s1, s15 // (op 593) most significant: sum


s_mov_b32 s52, s26 // (op 593) reg expression
s_mov_b32 s53, s27 // (op 593) reg expression
s_mov_b32 s54, s0 // (op 593) reg expression
s_mov_b32 s55, 131072 // (op 593) call()

v_and_b32 v10, -8, v1 // (op 684) 
v_sub_u32 v13, v1, v10 // (op 684) 

v_lshlrev_b32 v10, 4, v13 // (op 684) 

v_lshrrev_b32 v13, 3, v1 // (op 684) 
v_lshl_add_u32 v14, v13, 7, v10 // (op 684) 


v_lshlrev_b32 v10, 2, v14 // (op 684) 

v_lshrrev_b32 v13, 3, v10 // (op 684) 

v_and_b32 v10, -2, v1 // (op 737) 
v_sub_u32 v14, v1, v10 // (op 737) 

v_lshlrev_b32 v10, 4, v14 // (op 737) 

v_lshrrev_b32 v14, 1, v1 // (op 737) 
v_lshl_add_u32 v15, v14, 5, v10 // (op 737) 


v_mul_lo_u32 v10, v15, 6 // (op 737) 
v_lshrrev_b32 v14, 3, v10 // (op 737) 

v_lshrrev_b32 v10, 4, v15 // (op 737) 

v_lshlrev_b32 v15, 2, v10 // (op 737) 

v_add_u32 v10, v14, v15 // (op 737) 


s_ashr_i64 s[0:1], s[12:13], 7 // (op 157) 
s_and_b32 s14, s0, 4294967294 // (op 157) 

s_ashr_i64 s[0:1], s[12:13], 7 // (op 156) 
s_mov_b32 s15, s0 // (op 156) convert

s_cmp_lt_u32 s14, s15 // (op 156) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoopTail_156 // (op 156) Condition: Top (jump to RR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoopTail_156 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoopTail_156:


s_add_u32 s14, s14, 1 // (op 158) 
v_mov_b32 v14, v8 // (op 161) 

buffer_load_dwordx2 v[16:17], v14, s[40:43], 0 offen // (op 161) Load value

v_add_u32 v8, 64, v8 // (op 386) 


s_barrier  // (op 882) 
v_mov_b32 v14, v9 // (op 173) 



v_mov_b32 v15, v14 // (op 173) reg expression

buffer_load_dwordx3 v[18:20], v15, s[44:47], 0 offen // (op 173) Load value


s_lshl_b64 s[0:1], s[8:9], 7 // (op 455) 
s_mul_i32 s19, s1, 6 // (op 455) most significant: low of high * low
s_mul_i32 s18, s0, 6 // (op 455) least significant
s_mul_hi_u32 s15, s0, 6 // (op 455) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 455) most significant: sum


s_lshr_b64 s[0:1], s[18:19], 3 // (op 455) 

v_add_u32 v9, v9, s0 // (op 455) 

s_waitcnt vmcnt(1)
ds_write_b64 v13, v[16:17] offset:4096 // (op 162) Store local data 

s_barrier  // (op 895) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 885) 






ds_read_b128 v[14:17], v3 offset:4096 // (op 164) Load local data 
s_waitcnt vmcnt(0)
ds_write_b96 v10, v[18:20] // (op 172) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 899) 



ds_read_b96_tr_b6 v[18:20], v0 // (op 166) Transpose load local data 
ds_read_b96_tr_b6 v[21:23], v0 offset:512 // (op 166) Transpose load local data 


v_mov_b32 v24, v12 // (op 168) 

buffer_load_ubyte v25, v24, s[48:51], 0 offen offset: 0 // (op 168) Load value

v_add_u32 v12, 4, v12 // (op 519) 

v_mov_b64 v[26:27], s[24:25] // (op 170) convert

v_mov_b32 v24, v11 // (op 170) 
v_mov_b32 v28, v24 // (op 170) reg expression
v_mov_b32 v29, v28 // (op 170) 
buffer_load_ubyte v30, v29, s[52:55], 0 offen offset: 0 // (op 170) Load value




s_lshl_b32 s15, s24, 2 // (op 583) 
v_add_u32 v11, v11, s15 // (op 583) 

s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[14:17], v[18:23], a[0:3], v25, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 171) 




s_ashr_i64 s[0:1], s[12:13], 7 // (op 156) 
s_mov_b32 s15, s0 // (op 156) convert

s_cmp_lt_u32 s14, s15 // (op 156) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoopTail_156 // (op 156) Condition: Bottom (jump to RR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoopTail_156 if true)
RR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoopTail_156:






















s_barrier  // (op 957) 




v_cmp_eq_f32 vcc, 0.00000, v6 // (op 116) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalFalse_0_beta_116 // (op 116) Condition: False, jump to RR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalFalse_0_beta_116
s_nop 0
v_accvgpr_read v8, a0 // (op 42) Wait state hazard: XDL Write Hazard
v_accvgpr_read v9, a1 // (op 42) 
v_accvgpr_read v10, a2 // (op 42) 
v_accvgpr_read v11, a3 // (op 42) 
v_mul_f32 v12, v7, v8 // (op 42) 
v_mul_f32 v13, v7, v9 // (op 42) 
v_mul_f32 v14, v7, v10 // (op 42) 
v_mul_f32 v15, v7, v11 // (op 42) 

v_lshrrev_b32 v8, 16, v12 // (op 48) 
v_lshrrev_b32 v9, 16, v13 // (op 48) 
v_lshrrev_b32 v10, 16, v14 // (op 48) 
v_lshrrev_b32 v11, 16, v15 // (op 48) 

v_lshrrev_b32 v0, 7, v1 // (op 779) 
v_and_b32 v3, 63, v1 // (op 779) 
v_lshrrev_b32 v4, 4, v3 // (op 779) 
v_lshlrev_b32 v5, 2, v4 // (op 779) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 779) 


v_lshl_add_u32 v0, s16, 5, v4 // (op 779) 

v_lshrrev_b32 v4, 2, v3 // (op 779) 

v_and_b32 v3, 3, v4 // (op 779) 

v_and_b32 v4, 3, v1 // (op 779) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 779) 


v_lshrrev_b32 v3, 6, v1 // (op 779) 
v_and_b32 v4, 1, v3 // (op 779) 

v_lshl_add_u32 v3, v4, 4, v5 // (op 779) 


v_lshl_add_u32 v4, s17, 5, v3 // (op 779) 

v_mul_lo_u32 v3, v4, s30 // (op 779) 

v_add_lshl_u32 v4, v0, v3, 1 // (op 779) 


s_mul_i32 s1, s30, s35 // (op 781) most significant: low of low * high
s_mul_i32 s2, s31, s34 // (op 781) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 781) least significant
s_mul_hi_u32 s3, s30, s34 // (op 781) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 781) most significant: sum
s_add_u32 s1, s1, s3 // (op 781) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 1 // (op 781) 

s_mov_b32 s4, s32 // (op 781) reg expression
s_mov_b32 s5, s33 // (op 781) reg expression
s_mov_b32 s6, s2 // (op 781) reg expression
s_mov_b32 s7, 131072 // (op 781) call()


v_pack_b32_f16 v12, v8, v9 // (op 50) 
v_pack_b32_f16 v13, v10, v11 // (op 50) 


buffer_store_dwordx2 v[12:13], v4, s[4:7], 0 offen // (op 50) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalBottom_0_beta_116 // (op 116) Condition: Done, jump to RR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalBottom_0_beta_116
RR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalFalse_0_beta_116:
v_lshrrev_b32 v0, 7, v1 // (op 654) 
v_and_b32 v3, 63, v1 // (op 654) 
v_lshrrev_b32 v4, 4, v3 // (op 654) 
v_lshlrev_b32 v5, 2, v4 // (op 654) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 654) 


v_lshl_add_u32 v0, s16, 5, v4 // (op 654) 

v_lshrrev_b32 v4, 2, v3 // (op 654) 

v_and_b32 v3, 3, v4 // (op 654) 

v_and_b32 v4, 3, v1 // (op 654) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 654) 


v_lshrrev_b32 v3, 6, v1 // (op 654) 
v_and_b32 v4, 1, v3 // (op 654) 

v_lshl_add_u32 v3, v4, 4, v5 // (op 654) 


v_lshl_add_u32 v4, s17, 5, v3 // (op 654) 

v_mul_lo_u32 v3, v4, s36 // (op 654) 

v_add_lshl_u32 v4, v0, v3, 1 // (op 654) 


s_mul_i32 s1, s36, s35 // (op 656) most significant: low of low * high
s_mul_i32 s2, s37, s34 // (op 656) most significant: low of high * low
s_mul_i32 s0, s36, s34 // (op 656) least significant
s_mul_hi_u32 s3, s36, s34 // (op 656) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 656) most significant: sum
s_add_u32 s1, s1, s3 // (op 656) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 1 // (op 656) 

s_mov_b32 s8, s38 // (op 656) reg expression
s_mov_b32 s9, s39 // (op 656) reg expression
s_mov_b32 s10, s2 // (op 656) reg expression
s_mov_b32 s11, 131072 // (op 656) call()





buffer_load_dwordx2 v[8:9], v4, s[8:11], 0 offen // (op 129) Load value



v_accvgpr_read v10, a0 // (op 125) 
v_accvgpr_read v11, a1 // (op 125) 
v_accvgpr_read v12, a2 // (op 125) 
v_accvgpr_read v13, a3 // (op 125) 
v_mul_f32 v14, v7, v10 // (op 125) 
v_mul_f32 v15, v7, v11 // (op 125) 
v_mul_f32 v16, v7, v12 // (op 125) 
v_mul_f32 v17, v7, v13 // (op 125) 

s_waitcnt vmcnt(0)
v_and_b32 v4, 65535, v8 // (op 130) 
v_lshrrev_b32 v5, 16, v8 // (op 130) 
v_lshlrev_b32 v4, 16, v4 // (op 130) 
v_lshlrev_b32 v5, 16, v5 // (op 130) 
v_mul_f32 v10, v6, v4 // (op 130) 
v_mul_f32 v11, v6, v5 // (op 130) 
v_and_b32 v4, 65535, v9 // (op 130) 
v_lshrrev_b32 v5, 16, v9 // (op 130) 
v_lshlrev_b32 v4, 16, v4 // (op 130) 
v_lshlrev_b32 v5, 16, v5 // (op 130) 
v_mul_f32 v12, v6, v4 // (op 130) 
v_mul_f32 v13, v6, v5 // (op 130) 


v_add_f32 v18, v10, v14 // (op 126) 
v_add_f32 v19, v11, v15 // (op 126) 
v_add_f32 v20, v12, v16 // (op 126) 
v_add_f32 v21, v13, v17 // (op 126) 


v_lshrrev_b32 v8, 16, v18 // (op 127) 
v_lshrrev_b32 v9, 16, v19 // (op 127) 
v_lshrrev_b32 v10, 16, v20 // (op 127) 
v_lshrrev_b32 v11, 16, v21 // (op 127) 

v_lshrrev_b32 v0, 7, v1 // (op 796) 
v_and_b32 v3, 63, v1 // (op 796) 
v_lshrrev_b32 v4, 4, v3 // (op 796) 
v_lshlrev_b32 v5, 2, v4 // (op 796) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 796) 


v_lshl_add_u32 v0, s16, 5, v4 // (op 796) 

v_lshrrev_b32 v4, 2, v3 // (op 796) 

v_and_b32 v3, 3, v4 // (op 796) 

v_and_b32 v4, 3, v1 // (op 796) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 796) 


v_lshrrev_b32 v3, 6, v1 // (op 796) 
v_and_b32 v4, 1, v3 // (op 796) 

v_lshl_add_u32 v3, v4, 4, v5 // (op 796) 


v_lshl_add_u32 v4, s17, 5, v3 // (op 796) 

v_mul_lo_u32 v3, v4, s30 // (op 796) 

v_add_lshl_u32 v4, v0, v3, 1 // (op 796) 


s_mul_i32 s1, s30, s35 // (op 798) most significant: low of low * high
s_mul_i32 s2, s31, s34 // (op 798) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 798) least significant
s_mul_hi_u32 s3, s30, s34 // (op 798) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 798) most significant: sum
s_add_u32 s1, s1, s3 // (op 798) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 1 // (op 798) 

s_mov_b32 s4, s32 // (op 798) reg expression
s_mov_b32 s5, s33 // (op 798) reg expression
s_mov_b32 s6, s2 // (op 798) reg expression
s_mov_b32 s7, 131072 // (op 798) call()


v_pack_b32_f16 v12, v8, v9 // (op 128) 
v_pack_b32_f16 v13, v10, v11 // (op 128) 


buffer_store_dwordx2 v[12:13], v4, s[4:7], 0 offen // (op 128) Store value







s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalBottom_0_beta_116:



s_endpgm  // End of RR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2
.LRR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_end:
.size RR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2, .LRR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_end-RR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2
  .amdhsa_next_free_vgpr 48
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 12288 // lds bytes
  .amdhsa_accum_offset 44
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
    DataTypeB: B6
    DestDataType: b
    ComputeDataType: s
    HighPrecisionAccumulate: True
    TransposeA: True
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
            { type: int64, semantic: SizeFree0 },
            { type: int64, semantic: StrideB0 },
            { type: address, semantic: AddressB },
            { type: int64, semantic: SizeSum },
            { type: float32, semantic: Beta },
            { type: float32, semantic: Alpha },
            { type: int64, semantic: StrideScaleA0 },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: StrideScaleB0 },
            { type: address, semantic: AddressMXScaleB },
            { type: int64, semantic: SizeSumScaleA },
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
  - .name: RR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2
    .symbol: RR_GEMM_TT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2.kd
    .kernarg_segment_size: 136
    .group_segment_fixed_size: 12288
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 56
    .vgpr_count: 44
    .agpr_count: 4
    .max_flat_workgroup_size: 256
    .wavefront_size: 64
    .args:
      - .name: Tensor_0_stride_0_0
        .size: 8
        .offset: 0
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 8
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_0_1
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
      - .name: Tensor_0_size_1_4
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
      - .name: Tensor_4_stride_0_6
        .size: 8
        .offset: 56
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 64
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
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
      - .name: Tensor_4_size_1_9
        .size: 8
        .offset: 88
        .value_kind: by_value
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
      - .name: Tensor_2_size_1_12
        .size: 8
        .offset: 112
        .value_kind: by_value
      - .name: Tensor_10_stride_1_11
        .size: 8
        .offset: 120
        .value_kind: by_value
      - .name: Tensor_10_pointer
        .size: 8
        .offset: 128
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
...

.end_amdgpu_metadata

