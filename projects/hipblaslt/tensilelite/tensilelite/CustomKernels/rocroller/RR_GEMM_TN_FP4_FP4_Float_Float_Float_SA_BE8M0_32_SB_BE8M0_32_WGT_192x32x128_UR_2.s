// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2
.p2align 8
.type RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2,@function
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 6, v1 // (op 3011) 
v_and_b32 v3, 1, v0 // (op 3011) 

v_and_b32 v0, 15, v1 // (op 3011) 
v_lshl_add_u32 v4, v3, 4, v0 // (op 3011) 


v_and_b32 v0, 63, v1 // (op 3011) 
v_lshrrev_b32 v3, 4, v0 // (op 3011) 

v_lshlrev_b32 v0, 5, v3 // (op 3011) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 3011) 


v_lshlrev_b32 v0, 2, v3 // (op 3011) 

v_lshrrev_b32 v3, 3, v0 // (op 3011) 

v_lshrrev_b32 v0, 6, v1 // (op 3140) 
v_and_b32 v4, 1, v0 // (op 3140) 

v_and_b32 v0, 15, v1 // (op 3140) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 3140) 


v_and_b32 v0, 63, v1 // (op 3140) 
v_lshrrev_b32 v4, 4, v0 // (op 3140) 

v_lshlrev_b32 v0, 5, v4 // (op 3140) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 3140) 


v_lshlrev_b32 v0, 2, v4 // (op 3140) 

v_lshrrev_b32 v4, 3, v0 // (op 3140) 

v_lshrrev_b32 v0, 7, v1 // (op 2896) 
v_and_b32 v5, 63, v1 // (op 2896) 
v_lshrrev_b32 v6, 4, v5 // (op 2896) 
v_lshlrev_b32 v7, 2, v6 // (op 2896) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 2896) 


v_mov_b32 v0, 192 // (op 2896) 
v_mul_lo_u32 v7, s16, v0 // (op 2896) 

v_add_u32 v7, v7, v6 // (op 2896) 

v_lshrrev_b32 v0, 2, v5 // (op 2896) 

v_and_b32 v5, 3, v0 // (op 2896) 

v_and_b32 v0, 3, v1 // (op 2896) 
v_lshl_add_u32 v6, v5, 2, v0 // (op 2896) 


v_lshrrev_b32 v0, 6, v1 // (op 2896) 
v_and_b32 v5, 1, v0 // (op 2896) 

v_lshl_add_u32 v0, v5, 4, v6 // (op 2896) 


v_lshl_add_u32 v5, s17, 5, v0 // (op 2896) 

v_mul_lo_u32 v0, v5, s2 // (op 2896) 

v_add_lshl_u32 v5, v7, v0, 2 // (op 2896) 


v_lshrrev_b32 v0, 7, v1 // (op 2958) 
v_and_b32 v6, 15, v1 // (op 2958) 
v_lshl_add_u32 v7, v0, 4, v6 // (op 2958) 


v_and_b32 v0, 63, v1 // (op 2958) 
v_lshrrev_b32 v6, 4, v0 // (op 2958) 

v_lshlrev_b32 v0, 5, v6 // (op 2958) 

v_lshl_add_u32 v6, v7, 7, v0 // (op 2958) 


v_lshlrev_b32 v0, 2, v6 // (op 2958) 

v_lshrrev_b32 v6, 3, v0 // (op 2958) 

v_lshrrev_b32 v0, 7, v1 // (op 3069) 
v_and_b32 v7, 15, v1 // (op 3069) 
s_mul_i32 s1, s2, s9 // (op 2898) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 2898) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 2898) least significant
s_mul_hi_u32 s19, s2, s8 // (op 2898) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2898) most significant: sum
s_add_u32 s1, s1, s19 // (op 2898) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 2898) 

v_lshl_add_u32 v8, v0, 4, v7 // (op 3069) 


v_and_b32 v0, 63, v1 // (op 3069) 
v_lshrrev_b32 v7, 4, v0 // (op 3069) 

v_lshl_add_u32 v0, v8, 2, v7 // (op 3069) 


v_lshrrev_b32 v7, 7, v1 // (op 3085) 
v_and_b32 v8, 63, v1 // (op 3085) 
v_lshrrev_b32 v9, 4, v8 // (op 3085) 
v_lshlrev_b32 v10, 2, v9 // (op 3085) 

v_lshl_add_u32 v9, v7, 4, v10 // (op 3085) 


v_mov_b32 v7, 192 // (op 3085) 
v_mul_lo_u32 v10, s16, v7 // (op 3085) 

v_add_u32 v10, v10, v9 // (op 3085) 

v_lshrrev_b32 v7, 2, v8 // (op 3085) 

v_and_b32 v8, 3, v7 // (op 3085) 

v_and_b32 v7, 3, v1 // (op 3085) 
v_lshl_add_u32 v9, v8, 2, v7 // (op 3085) 


v_lshrrev_b32 v7, 6, v1 // (op 3085) 
v_and_b32 v8, 1, v7 // (op 3085) 

v_lshl_add_u32 v7, v8, 4, v9 // (op 3085) 


v_lshl_add_u32 v8, s17, 5, v7 // (op 3085) 

v_mul_lo_u32 v7, v8, s4 // (op 3085) 

v_add_lshl_u32 v8, v10, v7, 2 // (op 3085) 


v_lshrrev_b32 v7, 7, v1 // (op 3116) 
v_and_b32 v9, 15, v1 // (op 3116) 
v_lshl_add_u32 v10, v7, 4, v9 // (op 3116) 


v_and_b32 v7, 63, v1 // (op 3116) 
v_lshrrev_b32 v9, 4, v7 // (op 3116) 

v_lshlrev_b32 v7, 5, v9 // (op 3116) 

v_lshl_add_u32 v9, v10, 7, v7 // (op 3116) 


v_lshlrev_b32 v7, 2, v9 // (op 3116) 

v_lshrrev_b32 v9, 3, v7 // (op 3116) 

v_lshrrev_b32 v7, 7, v1 // (op 3169) 
v_and_b32 v10, 15, v1 // (op 3169) 
s_mul_i32 s1, s4, s9 // (op 3087) most significant: low of low * high
v_lshl_add_u32 v11, v7, 4, v10 // (op 3169) 


v_and_b32 v7, 63, v1 // (op 3169) 
v_lshrrev_b32 v10, 4, v7 // (op 3169) 

v_lshl_add_u32 v7, v11, 2, v10 // (op 3169) 


s_mul_i32 s38, s5, s8 // (op 3087) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 3087) least significant
s_mul_hi_u32 s39, s4, s8 // (op 3087) most significant: high of low * low
s_add_u32 s1, s1, s38 // (op 3087) most significant: sum
s_add_u32 s1, s1, s39 // (op 3087) most significant: sum


s_lshl_b64 s[38:39], s[0:1], 2 // (op 3087) 

s_mov_b32 s40, s6 // (op 2898) reg expression
s_mov_b32 s41, s7 // (op 2898) reg expression
s_mov_b32 s42, s18 // (op 2898) reg expression
s_mov_b32 s43, 131072 // (op 2898) call()

s_mov_b32 s44, s10 // (op 3087) reg expression

s_mov_b32 s45, s11 // (op 3087) reg expression
s_mov_b32 s46, s38 // (op 3087) reg expression
s_mov_b32 s47, 131072 // (op 3087) call()


v_mov_b32 v10, s14 // (op 2) Move value
v_mov_b32 v11, s15 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 59) call()
v_accvgpr_write a1, 0.00000 // (op 59) call()
v_accvgpr_write a2, 0.00000 // (op 59) call()
v_accvgpr_write a3, 0.00000 // (op 59) call()
v_accvgpr_write a4, 0.00000 // (op 309) call()
v_accvgpr_write a5, 0.00000 // (op 309) call()
v_accvgpr_write a6, 0.00000 // (op 309) call()
v_accvgpr_write a7, 0.00000 // (op 309) call()
v_accvgpr_write a8, 0.00000 // (op 471) call()
v_accvgpr_write a9, 0.00000 // (op 471) call()
v_accvgpr_write a10, 0.00000 // (op 471) call()
v_accvgpr_write a11, 0.00000 // (op 471) call()
v_accvgpr_write a12, 0.00000 // (op 633) call()
v_accvgpr_write a13, 0.00000 // (op 633) call()
v_accvgpr_write a14, 0.00000 // (op 633) call()
v_accvgpr_write a15, 0.00000 // (op 633) call()
v_accvgpr_write a16, 0.00000 // (op 795) call()
v_accvgpr_write a17, 0.00000 // (op 795) call()
v_accvgpr_write a18, 0.00000 // (op 795) call()
v_accvgpr_write a19, 0.00000 // (op 795) call()
v_accvgpr_write a20, 0.00000 // (op 957) call()
v_accvgpr_write a21, 0.00000 // (op 957) call()
v_accvgpr_write a22, 0.00000 // (op 957) call()
v_accvgpr_write a23, 0.00000 // (op 957) call()
v_lshrrev_b32 v12, 2, v1 // (op 2663) 
v_mul_lo_u32 v13, v12, 3 // (op 2663) 

v_mov_b32 v12, 192 // (op 2663) 
v_mul_lo_u32 v14, s16, v12 // (op 2663) 

v_add_u32 v14, v14, v13 // (op 2663) 

v_and_b32 v12, -4, v1 // (op 2663) 
v_sub_u32 v13, v1, v12 // (op 2663) 

v_lshlrev_b32 v12, 5, v13 // (op 2663) 

v_mov_b32 v17, 0 // (op 2663) convert
v_mov_b32 v16, v12 // (op 2663) convert

v_mul_lo_u32 v13, v14, s13 // (op 2663) most significant: low of low * high
v_mul_hi_u32 v15, v14, s12 // (op 2663) most significant: high of low * low
v_mul_lo_u32 v12, v14, s12 // (op 2663) least significant: low of low * low
v_add_u32 v13, v13, v15 // (op 2663) most significant: sum

v_add_co_u32 v12, s[0:1], v12, v16 // (op 2663) least significant half
v_addc_co_u32 v13, s[0:1], v13, v17, s[0:1] // (op 2663) most significant half



v_lshlrev_b64 v[14:15], 2, v[12:13] // (op 2663) 

v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 2663) 

v_mov_b32 v14, v12 // (op 2663) convert


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 2665) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 2665) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 2665) least significant
s_mul_hi_u32 s7, s12, s22 // (op 2665) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2665) most significant: sum
s_add_u32 s1, s1, s7 // (op 2665) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 2665) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 2665) 

s_mov_b32 s48, s20 // (op 2665) reg expression
s_mov_b32 s49, s21 // (op 2665) reg expression
s_mov_b32 s50, s0 // (op 2665) reg expression
s_mov_b32 s51, 131072 // (op 2665) call()

v_lshrrev_b32 v12, 2, v1 // (op 2682) 
v_mul_lo_u32 v13, v12, 3 // (op 2682) 

v_mov_b32 v12, 192 // (op 2682) 
v_mul_lo_u32 v15, s16, v12 // (op 2682) 

v_add_u32 v15, v15, v13 // (op 2682) 

v_and_b32 v12, -4, v1 // (op 2682) 
v_sub_u32 v13, v1, v12 // (op 2682) 

v_lshlrev_b32 v12, 5, v13 // (op 2682) 

v_mov_b32 v17, 0 // (op 2682) convert
v_mov_b32 v16, v12 // (op 2682) convert

v_mul_lo_u32 v13, v15, s13 // (op 2682) most significant: low of low * high
v_mul_hi_u32 v18, v15, s12 // (op 2682) most significant: high of low * low
v_mul_lo_u32 v12, v15, s12 // (op 2682) least significant: low of low * low
v_add_u32 v13, v13, v18 // (op 2682) most significant: sum

v_add_co_u32 v12, s[0:1], v12, v16 // (op 2682) least significant half
v_addc_co_u32 v13, s[0:1], v13, v17, s[0:1] // (op 2682) most significant half



v_lshlrev_b64 v[16:17], 2, v[12:13] // (op 2682) 

v_lshrrev_b64 v[12:13], 3, v[16:17] // (op 2682) 

v_mov_b32 v15, v12 // (op 2682) convert

s_mul_i32 s1, s12, s23 // (op 2684) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 2684) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 2684) least significant
s_mul_hi_u32 s7, s12, s22 // (op 2684) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2684) most significant: sum
s_add_u32 s1, s1, s7 // (op 2684) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 2684) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 2684) 

s_mov_b32 s48, s20 // (op 2684) reg expression
s_mov_b32 s49, s21 // (op 2684) reg expression
s_mov_b32 s50, s0 // (op 2684) reg expression
s_mov_b32 s51, 131072 // (op 2684) call()

v_and_b32 v12, -8, v1 // (op 2727) 
v_sub_u32 v13, v1, v12 // (op 2727) 

v_lshlrev_b32 v12, 4, v13 // (op 2727) 

v_mov_b32 v17, 0 // (op 2727) convert
v_mov_b32 v16, v12 // (op 2727) convert

v_lshrrev_b32 v12, 3, v1 // (op 2727) 
v_lshl_add_u32 v13, s17, 5, v12 // (op 2727) 

v_mul_lo_u32 v19, v13, s25 // (op 2727) most significant: low of low * high
v_mul_hi_u32 v12, v13, s24 // (op 2727) most significant: high of low * low
v_mul_lo_u32 v18, v13, s24 // (op 2727) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 2727) most significant: sum


v_add_co_u32 v12, s[0:1], v16, v18 // (op 2727) least significant half
v_addc_co_u32 v13, s[0:1], v17, v19, s[0:1] // (op 2727) most significant half

v_lshlrev_b64 v[12:13], 2, v[12:13] // (op 2727) 


v_lshrrev_b64 v[16:17], 3, v[12:13] // (op 2727) 

v_mov_b32 v12, v16 // (op 2727) convert

s_mul_i32 s1, s24, s9 // (op 2729) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 2729) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 2729) least significant
s_mul_hi_u32 s7, s24, s8 // (op 2729) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2729) most significant: sum
s_add_u32 s1, s1, s7 // (op 2729) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 2729) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 2729) 

s_mov_b32 s52, s26 // (op 2729) reg expression
s_mov_b32 s53, s27 // (op 2729) reg expression
s_mov_b32 s54, s0 // (op 2729) reg expression
s_mov_b32 s55, 131072 // (op 2729) call()

v_and_b32 v13, -8, v1 // (op 2746) 
v_sub_u32 v16, v1, v13 // (op 2746) 

v_lshlrev_b32 v13, 4, v16 // (op 2746) 

v_mov_b32 v17, 0 // (op 2746) convert
v_mov_b32 v16, v13 // (op 2746) convert

v_lshrrev_b32 v13, 3, v1 // (op 2746) 
v_lshl_add_u32 v18, s17, 5, v13 // (op 2746) 

v_mul_lo_u32 v21, v18, s25 // (op 2746) most significant: low of low * high
v_mul_hi_u32 v13, v18, s24 // (op 2746) most significant: high of low * low
v_mul_lo_u32 v20, v18, s24 // (op 2746) least significant: low of low * low
v_add_u32 v21, v21, v13 // (op 2746) most significant: sum


v_add_co_u32 v18, s[0:1], v16, v20 // (op 2746) least significant half
v_addc_co_u32 v19, s[0:1], v17, v21, s[0:1] // (op 2746) most significant half

v_lshlrev_b64 v[18:19], 2, v[18:19] // (op 2746) 


v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 2746) 

v_mov_b32 v13, v16 // (op 2746) convert

s_mul_i32 s1, s24, s9 // (op 2748) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 2748) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 2748) least significant
s_mul_hi_u32 s7, s24, s8 // (op 2748) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2748) most significant: sum
s_add_u32 s1, s1, s7 // (op 2748) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 2748) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 2748) 

s_mov_b32 s52, s26 // (op 2748) reg expression
s_mov_b32 s53, s27 // (op 2748) reg expression
s_mov_b32 s54, s0 // (op 2748) reg expression
s_mov_b32 s55, 131072 // (op 2748) call()

v_lshrrev_b32 v16, 2, v1 // (op 2791) 
v_mul_lo_u32 v17, v16, 3 // (op 2791) 

v_mov_b32 v16, 192 // (op 2791) 
v_mul_lo_u32 v18, s16, v16 // (op 2791) 

v_add_u32 v18, v18, v17 // (op 2791) 

v_and_b32 v16, -4, v1 // (op 2791) 
v_sub_u32 v17, v1, v16 // (op 2791) 

v_mul_lo_u32 v16, v18, s28 // (op 2791) 
v_add_u32 v16, v16, v17 // (op 2791) 


s_mul_i32 s1, s28, s23 // (op 2793) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 2793) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 2793) least significant
s_mul_hi_u32 s7, s28, s22 // (op 2793) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2793) most significant: sum
s_add_u32 s1, s1, s7 // (op 2793) most significant: sum


s_mov_b32 s56, s30 // (op 2793) reg expression
s_mov_b32 s57, s31 // (op 2793) reg expression
s_mov_b32 s58, s0 // (op 2793) reg expression
s_mov_b32 s59, 131072 // (op 2793) call()

v_lshrrev_b32 v17, 2, v1 // (op 2810) 
v_mul_lo_u32 v18, v17, 3 // (op 2810) 

v_mov_b32 v17, 192 // (op 2810) 
v_mul_lo_u32 v19, s16, v17 // (op 2810) 

v_add_u32 v19, v19, v18 // (op 2810) 

v_and_b32 v17, -4, v1 // (op 2810) 
v_sub_u32 v18, v1, v17 // (op 2810) 

v_mul_lo_u32 v17, v19, s28 // (op 2810) 
v_add_u32 v17, v17, v18 // (op 2810) 


s_mul_i32 s1, s28, s23 // (op 2812) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 2812) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 2812) least significant
s_mul_hi_u32 s7, s28, s22 // (op 2812) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2812) most significant: sum
s_add_u32 s1, s1, s7 // (op 2812) most significant: sum


s_mov_b32 s56, s30 // (op 2812) reg expression
s_mov_b32 s57, s31 // (op 2812) reg expression
s_mov_b32 s58, s0 // (op 2812) reg expression
s_mov_b32 s59, 131072 // (op 2812) call()

v_lshrrev_b32 v18, 6, v1 // (op 2855) 
v_and_b32 v19, 1, v18 // (op 2855) 

v_and_b32 v18, 15, v1 // (op 2855) 
v_lshl_add_u32 v20, v19, 4, v18 // (op 2855) 


v_lshl_add_u32 v18, s17, 5, v20 // (op 2855) 

v_mul_lo_u32 v19, v18, s32 // (op 2855) 

v_and_b32 v18, 63, v1 // (op 2855) 
v_lshrrev_b32 v20, 4, v18 // (op 2855) 

v_add_u32 v18, v20, v19 // (op 2855) 


s_mul_i32 s1, s32, s9 // (op 2857) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 2857) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 2857) least significant
s_mul_hi_u32 s7, s32, s8 // (op 2857) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2857) most significant: sum
s_add_u32 s1, s1, s7 // (op 2857) most significant: sum


s_mov_b32 s60, s34 // (op 2857) reg expression
s_mov_b32 s61, s35 // (op 2857) reg expression
s_mov_b32 s62, s0 // (op 2857) reg expression
s_mov_b32 s63, 131072 // (op 2857) call()

v_lshrrev_b32 v19, 6, v1 // (op 2874) 
v_and_b32 v20, 1, v19 // (op 2874) 

v_and_b32 v19, 15, v1 // (op 2874) 
v_lshl_add_u32 v21, v20, 4, v19 // (op 2874) 


v_lshl_add_u32 v19, s17, 5, v21 // (op 2874) 

v_mul_lo_u32 v20, v19, s32 // (op 2874) 

v_and_b32 v19, 63, v1 // (op 2874) 
v_lshrrev_b32 v21, 4, v19 // (op 2874) 

v_add_u32 v19, v21, v20 // (op 2874) 


s_mul_i32 s1, s32, s9 // (op 2876) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 2876) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 2876) least significant
s_mul_hi_u32 s7, s32, s8 // (op 2876) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2876) most significant: sum
s_add_u32 s1, s1, s7 // (op 2876) most significant: sum


s_mov_b32 s60, s34 // (op 2876) reg expression
s_mov_b32 s61, s35 // (op 2876) reg expression
s_mov_b32 s62, s0 // (op 2876) reg expression
s_mov_b32 s63, 131072 // (op 2876) call()




buffer_load_ubyte v20, v18, s[60:63], 0 offen offset: 0 // (op 2426) Load value



buffer_load_dwordx2 v[22:23], v12, s[52:55], 0 offen // (op 2430) Load value

s_lshl_b64 s[0:1], s[12:13], 2 // (op 2433) 
v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 2433) 


v_mov_b32 v21, v14 // (op 2433) 
buffer_load_dwordx4 v[26:29], v21, s[48:51], 0 offen // (op 2433) Load value
v_add_u32 v21, v21, v24 // (op 2433) 
buffer_load_dwordx4 v[30:33], v21, s[48:51], 0 offen // (op 2433) Load value
v_add_u32 v21, v21, v24 // (op 2433) 
buffer_load_dwordx4 v[34:37], v21, s[48:51], 0 offen // (op 2433) Load value



v_mov_b64 v[24:25], s[28:29] // (op 2436) convert

v_mov_b32 v21, v16 // (op 2436) 
buffer_load_ubyte v38, v21, s[56:59], 0 offen offset: 0 // (op 2436) Load value
v_add_u32 v21, v21, v24 // (op 2436) 
buffer_load_ubyte v39, v21, s[56:59], 0 offen offset: 0 // (op 2436) Load value
v_add_u32 v21, v21, v24 // (op 2436) 
buffer_load_ubyte v40, v21, s[56:59], 0 offen offset: 0 // (op 2436) Load value





buffer_load_ubyte v21, v18, s[60:63], 0 offen offset:4 // (op 2440) Load value




buffer_load_dwordx2 v[24:25], v12, s[52:55], 0 offen offset:64 // (op 2444) Load value


s_lshl_b64 s[0:1], s[12:13], 2 // (op 2447) 
v_lshrrev_b64 v[42:43], 3, s[0:1] // (op 2447) 


v_mov_b32 v12, v14 // (op 2447) 
buffer_load_dwordx4 v[44:47], v12, s[48:51], 0 offen offset:64 // (op 2447) Load value
v_add_u32 v12, v12, v42 // (op 2447) 
buffer_load_dwordx4 v[48:51], v12, s[48:51], 0 offen offset:64 // (op 2447) Load value
v_add_u32 v12, v12, v42 // (op 2447) 
buffer_load_dwordx4 v[52:55], v12, s[48:51], 0 offen offset:64 // (op 2447) Load value




v_mov_b64 v[42:43], s[28:29] // (op 2450) convert

v_mov_b32 v12, v16 // (op 2450) 
buffer_load_ubyte v56, v12, s[56:59], 0 offen offset:4 // (op 2450) Load value
v_add_u32 v12, v12, v42 // (op 2450) 
buffer_load_ubyte v57, v12, s[56:59], 0 offen offset:4 // (op 2450) Load value
v_add_u32 v12, v12, v42 // (op 2450) 
buffer_load_ubyte v58, v12, s[56:59], 0 offen offset:4 // (op 2450) Load value



v_and_b32 v12, -8, v1 // (op 2999) 
v_sub_u32 v14, v1, v12 // (op 2999) 

v_lshlrev_b32 v12, 4, v14 // (op 2999) 

v_lshrrev_b32 v14, 3, v1 // (op 2999) 
v_lshl_add_u32 v16, v14, 7, v12 // (op 2999) 


v_lshlrev_b32 v12, 2, v16 // (op 2999) 

v_lshrrev_b32 v14, 3, v12 // (op 2999) 




s_waitcnt vmcnt(14)
ds_write_b64 v14, v[22:23] // (op 2452) Store local data 

v_and_b32 v12, -4, v1 // (op 2941) 
v_sub_u32 v14, v1, v12 // (op 2941) 

v_lshlrev_b32 v12, 5, v14 // (op 2941) 

v_lshrrev_b32 v14, 2, v1 // (op 2941) 
v_mul_lo_u32 v16, v14, 3 // (op 2941) 

v_lshl_add_u32 v14, v16, 7, v12 // (op 2941) 


v_lshlrev_b32 v12, 2, v14 // (op 2941) 

v_lshrrev_b32 v14, 3, v12 // (op 2941) 




s_waitcnt vmcnt(13)
ds_write_b128 v14, v[26:29] offset:2048 // (op 2453) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v14, v[30:33] offset:2112 // (op 2453) Store local data 
s_waitcnt vmcnt(11)
ds_write_b128 v14, v[34:37] offset:2176 // (op 2453) Store local data 

v_and_b32 v12, -4, v1 // (op 3052) 
v_sub_u32 v14, v1, v12 // (op 3052) 

v_lshrrev_b32 v12, 2, v1 // (op 3052) 
v_mul_lo_u32 v16, v12, 3 // (op 3052) 

v_lshl_add_u32 v12, v16, 2, v14 // (op 3052) 





s_waitcnt vmcnt(10)
ds_write_b8 v12, v38 offset:14336 // (op 2454) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v12, v39 offset:14340 // (op 2454) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v12, v40 offset:14344 // (op 2454) Store local data 

v_and_b32 v12, -4, v1 // (op 2912) 
v_sub_u32 v14, v1, v12 // (op 2912) 

v_lshlrev_b32 v12, 5, v14 // (op 2912) 

v_lshrrev_b32 v14, 2, v1 // (op 2912) 
v_mul_lo_u32 v16, v14, 3 // (op 2912) 

v_lshl_add_u32 v14, v16, 7, v12 // (op 2912) 


v_lshlrev_b32 v12, 2, v14 // (op 2912) 

v_lshrrev_b32 v14, 3, v12 // (op 2912) 

v_and_b32 v12, -8, v1 // (op 2970) 
v_sub_u32 v16, v1, v12 // (op 2970) 

v_lshlrev_b32 v12, 4, v16 // (op 2970) 

v_lshrrev_b32 v16, 3, v1 // (op 2970) 
v_lshl_add_u32 v18, v16, 7, v12 // (op 2970) 


v_lshlrev_b32 v12, 2, v18 // (op 2970) 

v_lshrrev_b32 v16, 3, v12 // (op 2970) 

v_and_b32 v12, -4, v1 // (op 3023) 
v_sub_u32 v18, v1, v12 // (op 3023) 

v_lshrrev_b32 v12, 2, v1 // (op 3023) 
v_mul_lo_u32 v41, v12, 3 // (op 3023) 

v_lshl_add_u32 v12, v41, 2, v18 // (op 3023) 


v_and_b32 v18, -4, v1 // (op 3099) 
v_sub_u32 v41, v1, v18 // (op 3099) 

v_lshlrev_b32 v18, 5, v41 // (op 3099) 

v_lshrrev_b32 v41, 2, v1 // (op 3099) 
v_mul_lo_u32 v42, v41, 3 // (op 3099) 

v_lshl_add_u32 v41, v42, 7, v18 // (op 3099) 


v_lshlrev_b32 v18, 2, v41 // (op 3099) 

v_lshrrev_b32 v41, 3, v18 // (op 3099) 

v_and_b32 v18, -8, v1 // (op 3128) 
v_sub_u32 v42, v1, v18 // (op 3128) 

v_lshlrev_b32 v18, 4, v42 // (op 3128) 

v_lshrrev_b32 v42, 3, v1 // (op 3128) 
v_lshl_add_u32 v43, v42, 7, v18 // (op 3128) 


v_lshlrev_b32 v18, 2, v43 // (op 3128) 

v_lshrrev_b32 v42, 3, v18 // (op 3128) 

v_and_b32 v18, -4, v1 // (op 3152) 
v_sub_u32 v43, v1, v18 // (op 3152) 

v_lshrrev_b32 v18, 2, v1 // (op 3152) 
v_mul_lo_u32 v59, v18, 3 // (op 3152) 

v_lshl_add_u32 v18, v59, 2, v43 // (op 3152) 


s_mov_b32 s0, 0 // (op 959) call()
s_ashr_i64 s[6:7], s[36:37], 7 // (op 958) 
s_mov_b32 s1, s6 // (op 958) convert

s_and_b32 s6, s1, 4294967294 // (op 958) 

s_cmp_lt_u32 s0, s6 // (op 958) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2_ForLoopBottom_KLoop_958 // (op 958) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2_ForLoopBottom_KLoop_958 if false)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2_ForLoopTop_KLoop_958:
s_add_u32 s0, s0, 2 // (op 960) 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 3222) 



ds_read_b128 v[60:63], v6 offset:2048 // (op 65) Load local data 
s_barrier  // (op 3235) 



ds_read_b128 v[64:67], v3 // (op 70) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 3249) 



ds_read_u8 v43, v0 offset:14336 // (op 75) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[60:63], v[64:67], a[0:3], v43, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 82) 
ds_read_b128 v[68:71], v6 offset:4096 // (op 340) Load local data 





ds_read_u8 v43, v0 offset:14464 // (op 344) Load local data 



ds_read_b128 v[60:63], v6 offset:6144 // (op 502) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[68:71], v[64:67], a[4:7], v43, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 346) 





ds_read_u8 v43, v0 offset:14592 // (op 506) Load local data 



ds_read_b128 v[68:71], v6 offset:8192 // (op 664) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[60:63], v[64:67], a[8:11], v43, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 508) 





ds_read_u8 v43, v0 offset:14720 // (op 668) Load local data 



ds_read_b128 v[60:63], v6 offset:10240 // (op 826) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[68:71], v[64:67], a[12:15], v43, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 670) 





ds_read_u8 v43, v0 offset:14848 // (op 830) Load local data 



ds_read_b128 v[68:71], v6 offset:12288 // (op 988) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[60:63], v[64:67], a[16:19], v43, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 832) 





ds_read_u8 v43, v0 offset:14976 // (op 992) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[68:71], v[64:67], a[20:23], v43, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 994) 





v_mov_b32 v43, v19 // (op 80) 

buffer_load_ubyte v20, v43, s[60:63], 0 offen offset:8 // (op 80) Load value


v_mov_b32 v59, v13 // (op 10) 

buffer_load_dwordx2 v[22:23], v59, s[52:55], 0 offen offset:128 // (op 10) Load value

s_lshl_b64 s[6:7], s[12:13], 2 // (op 4) 
v_lshrrev_b64 v[60:61], 3, s[6:7] // (op 4) 


v_mov_b32 v62, v15 // (op 4) 
v_mov_b32 v63, v62 // (op 4) 
buffer_load_dwordx4 v[26:29], v63, s[48:51], 0 offen offset:128 // (op 4) Load value
v_add_u32 v63, v63, v60 // (op 4) 
buffer_load_dwordx4 v[30:33], v63, s[48:51], 0 offen offset:128 // (op 4) Load value
v_add_u32 v63, v63, v60 // (op 4) 
buffer_load_dwordx4 v[34:37], v63, s[48:51], 0 offen offset:128 // (op 4) Load value



v_mov_b64 v[60:61], s[28:29] // (op 17) convert

v_mov_b32 v63, v17 // (op 17) 
v_mov_b32 v64, v63 // (op 17) 
buffer_load_ubyte v38, v64, s[56:59], 0 offen offset:8 // (op 17) Load value
v_add_u32 v64, v64, v60 // (op 17) 
buffer_load_ubyte v39, v64, s[56:59], 0 offen offset:8 // (op 17) Load value
v_add_u32 v64, v64, v60 // (op 17) 
buffer_load_ubyte v40, v64, s[56:59], 0 offen offset:8 // (op 17) Load value


s_barrier  // (op 3272) 



s_waitcnt vmcnt(14)
ds_write_b64 v42, v[24:25] offset:15104 // (op 235) Store local data 
s_barrier  // (op 3266) 



s_waitcnt vmcnt(13)
ds_write_b128 v41, v[44:47] offset:17152 // (op 225) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v41, v[48:51] offset:17216 // (op 225) Store local data 
s_waitcnt vmcnt(11)
ds_write_b128 v41, v[52:55] offset:17280 // (op 225) Store local data 
s_barrier  // (op 3278) 



s_waitcnt vmcnt(10)
ds_write_b8 v18, v56 offset:29440 // (op 236) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v18, v57 offset:29444 // (op 236) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v18, v58 offset:29448 // (op 236) Store local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 3263) 



ds_read_b128 v[64:67], v9 offset:17152 // (op 227) Load local data 
s_barrier  // (op 3269) 



ds_read_b128 v[68:71], v4 offset:15104 // (op 229) Load local data 
s_barrier  // (op 3315) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 3275) 



ds_read_u8 v60, v7 offset:29440 // (op 231) Load local data 



ds_read_b128 v[72:75], v9 offset:19200 // (op 353) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[64:67], v[68:71], a[0:3], v60, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 234) 





ds_read_u8 v60, v7 offset:29568 // (op 357) Load local data 



ds_read_b128 v[64:67], v9 offset:21248 // (op 515) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[72:75], v[68:71], a[4:7], v60, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 347) 





ds_read_u8 v60, v7 offset:29696 // (op 519) Load local data 



ds_read_b128 v[72:75], v9 offset:23296 // (op 677) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[64:67], v[68:71], a[8:11], v60, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 509) 





ds_read_u8 v60, v7 offset:29824 // (op 681) Load local data 



ds_read_b128 v[64:67], v9 offset:25344 // (op 839) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[72:75], v[68:71], a[12:15], v60, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 671) 





ds_read_u8 v60, v7 offset:29952 // (op 843) Load local data 



ds_read_b128 v[72:75], v9 offset:27392 // (op 1001) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[64:67], v[68:71], a[16:19], v60, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 833) 





s_barrier  // (op 3373) 
ds_read_u8 v60, v7 offset:30080 // (op 1005) Load local data 
s_barrier  // (op 3377) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[72:75], v[68:71], a[20:23], v60, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 995) 






buffer_load_ubyte v21, v43, s[60:63], 0 offen offset:12 // (op 233) Load value




v_add_u32 v19, 8, v19 // (op 2866) 
buffer_load_dwordx2 v[24:25], v59, s[52:55], 0 offen offset:192 // (op 237) Load value


v_add_u32 v13, 128, v13 // (op 2738) 
s_lshl_b64 s[6:7], s[12:13], 2 // (op 224) 
v_lshrrev_b64 v[60:61], 3, s[6:7] // (op 224) 


v_mov_b32 v43, v62 // (op 224) 
buffer_load_dwordx4 v[44:47], v43, s[48:51], 0 offen offset:192 // (op 224) Load value
v_add_u32 v43, v43, v60 // (op 224) 
buffer_load_dwordx4 v[48:51], v43, s[48:51], 0 offen offset:192 // (op 224) Load value
v_add_u32 v43, v43, v60 // (op 224) 
buffer_load_dwordx4 v[52:55], v43, s[48:51], 0 offen offset:192 // (op 224) Load value




v_add_u32 v15, 128, v15 // (op 2674) 
v_mov_b64 v[60:61], s[28:29] // (op 238) convert

v_mov_b32 v43, v63 // (op 238) 
buffer_load_ubyte v56, v43, s[56:59], 0 offen offset:12 // (op 238) Load value
v_add_u32 v43, v43, v60 // (op 238) 
buffer_load_ubyte v57, v43, s[56:59], 0 offen offset:12 // (op 238) Load value
v_add_u32 v43, v43, v60 // (op 238) 
buffer_load_ubyte v58, v43, s[56:59], 0 offen offset:12 // (op 238) Load value



s_barrier  // (op 3238) 
v_add_u32 v17, 8, v17 // (op 2802) 



s_waitcnt vmcnt(14)
ds_write_b64 v16, v[22:23] // (op 13) Store local data 
s_barrier  // (op 3225) 



s_waitcnt vmcnt(13)
ds_write_b128 v14, v[26:29] offset:2048 // (op 6) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v14, v[30:33] offset:2112 // (op 6) Store local data 
s_waitcnt vmcnt(11)
ds_write_b128 v14, v[34:37] offset:2176 // (op 6) Store local data 
s_barrier  // (op 3252) 



s_waitcnt vmcnt(10)
ds_write_b8 v12, v38 offset:14336 // (op 21) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v12, v39 offset:14340 // (op 21) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v12, v40 offset:14344 // (op 21) Store local data 
s_ashr_i64 s[6:7], s[36:37], 7 // (op 958) 
s_mov_b32 s1, s6 // (op 958) convert

s_and_b32 s6, s1, 4294967294 // (op 958) 

s_cmp_lt_u32 s0, s6 // (op 958) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2_ForLoopTop_KLoop_958 // (op 958) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2_ForLoopTop_KLoop_958 if true)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2_ForLoopBottom_KLoop_958:















v_lshrrev_b32 v12, 2, v1 // (op 2632) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 2632) 
v_mul_lo_u32 v13, v12, 3 // (op 2632) 

s_lshr_b32 s6, s0, 1 // (op 2632) 

v_mov_b32 v12, 192 // (op 2632) 
v_mul_lo_u32 v14, s16, v12 // (op 2632) 

v_add_u32 v14, v14, v13 // (op 2632) 

v_and_b32 v12, -4, v1 // (op 2632) 
v_sub_u32 v13, v1, v12 // (op 2632) 

v_lshlrev_b32 v12, 5, v13 // (op 2632) 

v_lshl_add_u32 v13, s6, 8, v12 // (op 2632) 


v_mov_b32 v23, 0 // (op 2632) convert
v_mov_b32 v22, v13 // (op 2632) convert

v_mul_lo_u32 v13, v14, s13 // (op 2632) most significant: low of low * high
v_mul_hi_u32 v15, v14, s12 // (op 2632) most significant: high of low * low
v_mul_lo_u32 v12, v14, s12 // (op 2632) least significant: low of low * low
v_add_u32 v13, v13, v15 // (op 2632) most significant: sum

v_add_co_u32 v12, s[0:1], v12, v22 // (op 2632) least significant half
v_addc_co_u32 v13, s[0:1], v13, v23, s[0:1] // (op 2632) most significant half



v_lshlrev_b64 v[22:23], 2, v[12:13] // (op 2632) 

v_lshrrev_b64 v[12:13], 3, v[22:23] // (op 2632) 

v_mov_b32 v14, v12 // (op 2632) convert

s_mul_i32 s1, s12, s23 // (op 2634) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 2634) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 2634) least significant
s_mul_hi_u32 s7, s12, s22 // (op 2634) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2634) most significant: sum
s_add_u32 s1, s1, s7 // (op 2634) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 2634) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 2634) 

s_mov_b32 s48, s20 // (op 2634) reg expression
s_mov_b32 s49, s21 // (op 2634) reg expression
s_mov_b32 s50, s0 // (op 2634) reg expression
s_mov_b32 s51, 131072 // (op 2634) call()

v_and_b32 v12, -8, v1 // (op 2704) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 2704) 
v_sub_u32 v13, v1, v12 // (op 2704) 

s_lshr_b32 s6, s0, 1 // (op 2704) 

v_lshlrev_b32 v12, 4, v13 // (op 2704) 

v_lshl_add_u32 v13, s6, 8, v12 // (op 2704) 


v_mov_b32 v23, 0 // (op 2704) convert
v_mov_b32 v22, v13 // (op 2704) convert

v_lshrrev_b32 v12, 3, v1 // (op 2704) 
v_lshl_add_u32 v13, s17, 5, v12 // (op 2704) 

v_mul_lo_u32 v27, v13, s25 // (op 2704) most significant: low of low * high
v_mul_hi_u32 v12, v13, s24 // (op 2704) most significant: high of low * low
v_mul_lo_u32 v26, v13, s24 // (op 2704) least significant: low of low * low
v_add_u32 v27, v27, v12 // (op 2704) most significant: sum


v_add_co_u32 v12, s[0:1], v22, v26 // (op 2704) least significant half
v_addc_co_u32 v13, s[0:1], v23, v27, s[0:1] // (op 2704) most significant half

v_lshlrev_b64 v[12:13], 2, v[12:13] // (op 2704) 


v_lshrrev_b64 v[22:23], 3, v[12:13] // (op 2704) 

v_mov_b32 v12, v22 // (op 2704) convert

s_mul_i32 s1, s24, s9 // (op 2706) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 2706) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 2706) least significant
s_mul_hi_u32 s7, s24, s8 // (op 2706) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2706) most significant: sum
s_add_u32 s1, s1, s7 // (op 2706) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 2706) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 2706) 

s_mov_b32 s52, s26 // (op 2706) reg expression
s_mov_b32 s53, s27 // (op 2706) reg expression
s_mov_b32 s54, s0 // (op 2706) reg expression
s_mov_b32 s55, 131072 // (op 2706) call()

v_lshrrev_b32 v13, 2, v1 // (op 2768) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 2768) 
v_mul_lo_u32 v15, v13, 3 // (op 2768) 

s_lshr_b32 s6, s0, 1 // (op 2768) 

v_mov_b32 v13, 192 // (op 2768) 
v_mul_lo_u32 v16, s16, v13 // (op 2768) 

v_add_u32 v16, v16, v15 // (op 2768) 

v_and_b32 v13, -4, v1 // (op 2768) 
v_sub_u32 v15, v1, v13 // (op 2768) 

v_lshl_add_u32 v13, s6, 3, v15 // (op 2768) 


v_mul_lo_u32 v15, v16, s28 // (op 2768) 
v_add_u32 v15, v15, v13 // (op 2768) 


s_mul_i32 s1, s28, s23 // (op 2770) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 2770) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 2770) least significant
s_mul_hi_u32 s7, s28, s22 // (op 2770) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2770) most significant: sum
s_add_u32 s1, s1, s7 // (op 2770) most significant: sum


s_mov_b32 s56, s30 // (op 2770) reg expression
s_mov_b32 s57, s31 // (op 2770) reg expression
s_mov_b32 s58, s0 // (op 2770) reg expression
s_mov_b32 s59, 131072 // (op 2770) call()

v_and_b32 v13, 63, v1 // (op 2832) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 2832) 
v_lshrrev_b32 v16, 4, v13 // (op 2832) 

s_lshr_b32 s6, s0, 1 // (op 2832) 

v_lshrrev_b32 v13, 6, v1 // (op 2832) 
v_lshl_add_u32 v17, s6, 3, v16 // (op 2832) 


v_and_b32 v16, 1, v13 // (op 2832) 

v_and_b32 v13, 15, v1 // (op 2832) 
v_lshl_add_u32 v18, v16, 4, v13 // (op 2832) 


v_lshl_add_u32 v13, s17, 5, v18 // (op 2832) 

v_mul_lo_u32 v16, v13, s32 // (op 2832) 

v_add_u32 v13, v17, v16 // (op 2832) 


s_mul_i32 s1, s32, s9 // (op 2834) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 2834) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 2834) least significant
s_mul_hi_u32 s7, s32, s8 // (op 2834) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2834) most significant: sum
s_add_u32 s1, s1, s7 // (op 2834) most significant: sum


s_mov_b32 s60, s34 // (op 2834) reg expression
s_mov_b32 s61, s35 // (op 2834) reg expression
s_mov_b32 s62, s0 // (op 2834) reg expression
s_mov_b32 s63, 131072 // (op 2834) call()

v_and_b32 v16, -4, v1 // (op 2925) 
v_sub_u32 v17, v1, v16 // (op 2925) 

v_lshlrev_b32 v16, 5, v17 // (op 2925) 

v_lshrrev_b32 v17, 2, v1 // (op 2925) 
v_mul_lo_u32 v18, v17, 3 // (op 2925) 

v_lshl_add_u32 v17, v18, 7, v16 // (op 2925) 


v_lshlrev_b32 v16, 2, v17 // (op 2925) 

v_lshrrev_b32 v17, 3, v16 // (op 2925) 

v_and_b32 v16, -8, v1 // (op 2983) 
v_sub_u32 v18, v1, v16 // (op 2983) 

v_lshlrev_b32 v16, 4, v18 // (op 2983) 

v_lshrrev_b32 v18, 3, v1 // (op 2983) 
v_lshl_add_u32 v19, v18, 7, v16 // (op 2983) 


v_lshlrev_b32 v16, 2, v19 // (op 2983) 

v_lshrrev_b32 v18, 3, v16 // (op 2983) 

v_and_b32 v16, -4, v1 // (op 3036) 
v_sub_u32 v19, v1, v16 // (op 3036) 

v_lshrrev_b32 v16, 2, v1 // (op 3036) 
v_mul_lo_u32 v20, v16, 3 // (op 3036) 

v_lshl_add_u32 v16, v20, 2, v19 // (op 3036) 


s_ashr_i64 s[0:1], s[36:37], 7 // (op 964) 
s_and_b32 s6, s0, 4294967294 // (op 964) 

s_ashr_i64 s[0:1], s[36:37], 7 // (op 963) 
s_mov_b32 s7, s0 // (op 963) convert

s_cmp_lt_u32 s6, s7 // (op 963) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2_ForLoopBottom_KLoopTail_963 // (op 963) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2_ForLoopBottom_KLoopTail_963 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2_ForLoopTop_KLoopTail_963:

s_add_u32 s6, s6, 1 // (op 965) 
s_lshl_b64 s[0:1], s[12:13], 2 // (op 166) 
v_lshrrev_b64 v[20:21], 3, s[0:1] // (op 166) 


v_mov_b32 v19, v14 // (op 166) 
v_mov_b32 v22, v19 // (op 166) 
buffer_load_dwordx4 v[24:27], v22, s[48:51], 0 offen // (op 166) Load value
v_add_u32 v22, v22, v20 // (op 166) 
buffer_load_dwordx4 v[28:31], v22, s[48:51], 0 offen // (op 166) Load value
v_add_u32 v22, v22, v20 // (op 166) 
buffer_load_dwordx4 v[32:35], v22, s[48:51], 0 offen // (op 166) Load value



v_add_u32 v14, 64, v14 // (op 2622) 


s_barrier  // (op 3228) 
v_mov_b32 v19, v12 // (op 179) 




s_waitcnt vmcnt(2)
ds_write_b128 v17, v[24:27] offset:2048 // (op 167) Store local data 
buffer_load_dwordx2 v[20:21], v19, s[52:55], 0 offen // (op 179) Load value

v_add_u32 v12, 64, v12 // (op 2696) 
s_waitcnt vmcnt(2)
ds_write_b128 v17, v[28:31] offset:2112 // (op 167) Store local data 

v_mov_b64 v[22:23], s[28:29] // (op 180) convert

v_mov_b32 v19, v15 // (op 180) 
v_mov_b32 v36, v19 // (op 180) 
buffer_load_ubyte v38, v36, s[56:59], 0 offen offset: 0 // (op 180) Load value
v_add_u32 v36, v36, v22 // (op 180) 
buffer_load_ubyte v39, v36, s[56:59], 0 offen offset: 0 // (op 180) Load value
v_add_u32 v36, v36, v22 // (op 180) 
s_waitcnt vmcnt(3)
ds_write_b128 v17, v[32:35] offset:2176 // (op 167) Store local data 
buffer_load_ubyte v40, v36, s[56:59], 0 offen offset: 0 // (op 180) Load value




v_add_u32 v15, 4, v15 // (op 2760) 
s_barrier  // (op 3241) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 3231) 






ds_read_b128 v[22:25], v6 offset:2048 // (op 169) Load local data 
s_waitcnt vmcnt(3)
ds_write_b64 v18, v[20:21] // (op 177) Store local data 

s_barrier  // (op 3255) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 3245) 






ds_read_b128 v[26:29], v3 // (op 171) Load local data 
s_waitcnt vmcnt(2)
ds_write_b8 v16, v38 offset:14336 // (op 178) Store local data 
s_waitcnt vmcnt(1)
ds_write_b8 v16, v39 offset:14340 // (op 178) Store local data 
s_waitcnt vmcnt(0)
ds_write_b8 v16, v40 offset:14344 // (op 178) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 3259) 



ds_read_u8 v19, v0 offset:14336 // (op 173) Load local data 


v_mov_b32 v20, v13 // (op 175) 

buffer_load_ubyte v21, v20, s[60:63], 0 offen offset: 0 // (op 175) Load value

v_add_u32 v13, 4, v13 // (op 2824) 



ds_read_b128 v[30:33], v6 offset:4096 // (op 321) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:25], v[26:29], a[0:3], v19, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 176) 





ds_read_u8 v19, v0 offset:14464 // (op 325) Load local data 



ds_read_b128 v[22:25], v6 offset:6144 // (op 483) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[30:33], v[26:29], a[4:7], v19, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 327) 





ds_read_u8 v19, v0 offset:14592 // (op 487) Load local data 



ds_read_b128 v[30:33], v6 offset:8192 // (op 645) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[22:25], v[26:29], a[8:11], v19, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 489) 





ds_read_u8 v19, v0 offset:14720 // (op 649) Load local data 



ds_read_b128 v[22:25], v6 offset:10240 // (op 807) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[30:33], v[26:29], a[12:15], v19, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 651) 





ds_read_u8 v19, v0 offset:14848 // (op 811) Load local data 



ds_read_b128 v[30:33], v6 offset:12288 // (op 969) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[22:25], v[26:29], a[16:19], v19, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 813) 





ds_read_u8 v19, v0 offset:14976 // (op 973) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[30:33], v[26:29], a[20:23], v19, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 975) 




s_ashr_i64 s[0:1], s[36:37], 7 // (op 963) 
s_mov_b32 s7, s0 // (op 963) convert

s_cmp_lt_u32 s6, s7 // (op 963) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2_ForLoopTop_KLoopTail_963 // (op 963) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2_ForLoopTop_KLoopTail_963 if true)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2_ForLoopBottom_KLoopTail_963:






















s_barrier  // (op 3361) 







v_cmp_eq_f32 vcc, 0.00000, v10 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2_ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2_ConditionalFalse_0_beta_123
v_accvgpr_read v12, a0 // (op 46) 
v_accvgpr_read v13, a1 // (op 46) 
v_accvgpr_read v14, a2 // (op 46) 
v_accvgpr_read v15, a3 // (op 46) 
v_mul_f32 v16, v11, v12 // (op 46) 
v_mul_f32 v17, v11, v13 // (op 46) 
v_mul_f32 v18, v11, v14 // (op 46) 
v_mul_f32 v19, v11, v15 // (op 46) 




buffer_store_dwordx4 v[16:19], v8, s[44:47], 0 offen // (op 52) Store value

v_accvgpr_read v12, a4 // (op 1249) 
v_accvgpr_read v13, a5 // (op 1249) 
v_accvgpr_read v14, a6 // (op 1249) 
v_accvgpr_read v15, a7 // (op 1249) 
v_mul_f32 v16, v11, v12 // (op 1249) 
v_mul_f32 v17, v11, v13 // (op 1249) 
v_mul_f32 v18, v11, v14 // (op 1249) 
v_mul_f32 v19, v11, v15 // (op 1249) 




buffer_store_dwordx4 v[16:19], v8, s[44:47], 0 offen offset:128 // (op 1251) Store value

v_accvgpr_read v12, a8 // (op 1267) 
v_accvgpr_read v13, a9 // (op 1267) 
v_accvgpr_read v14, a10 // (op 1267) 
v_accvgpr_read v15, a11 // (op 1267) 
v_mul_f32 v16, v11, v12 // (op 1267) 
v_mul_f32 v17, v11, v13 // (op 1267) 
v_mul_f32 v18, v11, v14 // (op 1267) 
v_mul_f32 v19, v11, v15 // (op 1267) 




buffer_store_dwordx4 v[16:19], v8, s[44:47], 0 offen offset:256 // (op 1269) Store value

v_accvgpr_read v12, a12 // (op 1285) 
v_accvgpr_read v13, a13 // (op 1285) 
v_accvgpr_read v14, a14 // (op 1285) 
v_accvgpr_read v15, a15 // (op 1285) 
v_mul_f32 v16, v11, v12 // (op 1285) 
v_mul_f32 v17, v11, v13 // (op 1285) 
v_mul_f32 v18, v11, v14 // (op 1285) 
v_mul_f32 v19, v11, v15 // (op 1285) 




buffer_store_dwordx4 v[16:19], v8, s[44:47], 0 offen offset:384 // (op 1287) Store value

v_accvgpr_read v12, a16 // (op 1303) 
v_accvgpr_read v13, a17 // (op 1303) 
v_accvgpr_read v14, a18 // (op 1303) 
v_accvgpr_read v15, a19 // (op 1303) 
v_mul_f32 v16, v11, v12 // (op 1303) 
v_mul_f32 v17, v11, v13 // (op 1303) 
v_mul_f32 v18, v11, v14 // (op 1303) 
v_mul_f32 v19, v11, v15 // (op 1303) 




buffer_store_dwordx4 v[16:19], v8, s[44:47], 0 offen offset:512 // (op 1305) Store value

v_accvgpr_read v12, a20 // (op 1321) 
v_accvgpr_read v13, a21 // (op 1321) 
v_accvgpr_read v14, a22 // (op 1321) 
v_accvgpr_read v15, a23 // (op 1321) 
v_mul_f32 v16, v11, v12 // (op 1321) 
v_mul_f32 v17, v11, v13 // (op 1321) 
v_mul_f32 v18, v11, v14 // (op 1321) 
v_mul_f32 v19, v11, v15 // (op 1321) 




buffer_store_dwordx4 v[16:19], v8, s[44:47], 0 offen offset:640 // (op 1323) Store value

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2_ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2_ConditionalBottom_0_beta_123
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2_ConditionalFalse_0_beta_123:



buffer_load_dwordx4 v[12:15], v5, s[40:43], 0 offen offset:640 // (op 1443) Load value
v_accvgpr_read v16, a20 // (op 1440) 
v_accvgpr_read v17, a21 // (op 1440) 
v_accvgpr_read v18, a22 // (op 1440) 
v_accvgpr_read v19, a23 // (op 1440) 
v_mul_f32 v20, v11, v16 // (op 1440) 
v_mul_f32 v21, v11, v17 // (op 1440) 
v_mul_f32 v22, v11, v18 // (op 1440) 
v_mul_f32 v23, v11, v19 // (op 1440) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v10, v12 // (op 1444) 
v_mul_f32 v17, v10, v13 // (op 1444) 
v_mul_f32 v18, v10, v14 // (op 1444) 
v_mul_f32 v19, v10, v15 // (op 1444) 

v_add_f32 v12, v16, v20 // (op 1441) 
v_add_f32 v13, v17, v21 // (op 1441) 
v_add_f32 v14, v18, v22 // (op 1441) 
v_add_f32 v15, v19, v23 // (op 1441) 





buffer_store_dwordx4 v[12:15], v8, s[44:47], 0 offen offset:640 // (op 1442) Store value




s_nop 1
buffer_load_dwordx4 v[12:15], v5, s[40:43], 0 offen offset:512 // (op 1425) Load value
v_accvgpr_read v16, a16 // (op 1422) 
v_accvgpr_read v17, a17 // (op 1422) 
v_accvgpr_read v18, a18 // (op 1422) 
v_accvgpr_read v19, a19 // (op 1422) 
v_mul_f32 v20, v11, v16 // (op 1422) 
v_mul_f32 v21, v11, v17 // (op 1422) 
v_mul_f32 v22, v11, v18 // (op 1422) 
v_mul_f32 v23, v11, v19 // (op 1422) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v10, v12 // (op 1426) 
v_mul_f32 v17, v10, v13 // (op 1426) 
v_mul_f32 v18, v10, v14 // (op 1426) 
v_mul_f32 v19, v10, v15 // (op 1426) 

v_add_f32 v12, v16, v20 // (op 1423) 
v_add_f32 v13, v17, v21 // (op 1423) 
v_add_f32 v14, v18, v22 // (op 1423) 
v_add_f32 v15, v19, v23 // (op 1423) 





buffer_store_dwordx4 v[12:15], v8, s[44:47], 0 offen offset:512 // (op 1424) Store value




s_nop 1
buffer_load_dwordx4 v[12:15], v5, s[40:43], 0 offen offset:384 // (op 1407) Load value
v_accvgpr_read v16, a12 // (op 1404) 
v_accvgpr_read v17, a13 // (op 1404) 
v_accvgpr_read v18, a14 // (op 1404) 
v_accvgpr_read v19, a15 // (op 1404) 
v_mul_f32 v20, v11, v16 // (op 1404) 
v_mul_f32 v21, v11, v17 // (op 1404) 
v_mul_f32 v22, v11, v18 // (op 1404) 
v_mul_f32 v23, v11, v19 // (op 1404) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v10, v12 // (op 1408) 
v_mul_f32 v17, v10, v13 // (op 1408) 
v_mul_f32 v18, v10, v14 // (op 1408) 
v_mul_f32 v19, v10, v15 // (op 1408) 

v_add_f32 v12, v16, v20 // (op 1405) 
v_add_f32 v13, v17, v21 // (op 1405) 
v_add_f32 v14, v18, v22 // (op 1405) 
v_add_f32 v15, v19, v23 // (op 1405) 





buffer_store_dwordx4 v[12:15], v8, s[44:47], 0 offen offset:384 // (op 1406) Store value




s_nop 1
buffer_load_dwordx4 v[12:15], v5, s[40:43], 0 offen offset:256 // (op 1389) Load value
v_accvgpr_read v16, a8 // (op 1386) 
v_accvgpr_read v17, a9 // (op 1386) 
v_accvgpr_read v18, a10 // (op 1386) 
v_accvgpr_read v19, a11 // (op 1386) 
v_mul_f32 v20, v11, v16 // (op 1386) 
v_mul_f32 v21, v11, v17 // (op 1386) 
v_mul_f32 v22, v11, v18 // (op 1386) 
v_mul_f32 v23, v11, v19 // (op 1386) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v10, v12 // (op 1390) 
v_mul_f32 v17, v10, v13 // (op 1390) 
v_mul_f32 v18, v10, v14 // (op 1390) 
v_mul_f32 v19, v10, v15 // (op 1390) 

v_add_f32 v12, v16, v20 // (op 1387) 
v_add_f32 v13, v17, v21 // (op 1387) 
v_add_f32 v14, v18, v22 // (op 1387) 
v_add_f32 v15, v19, v23 // (op 1387) 





buffer_store_dwordx4 v[12:15], v8, s[44:47], 0 offen offset:256 // (op 1388) Store value




s_nop 1
buffer_load_dwordx4 v[12:15], v5, s[40:43], 0 offen offset:128 // (op 1371) Load value
v_accvgpr_read v16, a4 // (op 1368) 
v_accvgpr_read v17, a5 // (op 1368) 
v_accvgpr_read v18, a6 // (op 1368) 
v_accvgpr_read v19, a7 // (op 1368) 
v_mul_f32 v20, v11, v16 // (op 1368) 
v_mul_f32 v21, v11, v17 // (op 1368) 
v_mul_f32 v22, v11, v18 // (op 1368) 
v_mul_f32 v23, v11, v19 // (op 1368) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v10, v12 // (op 1372) 
v_mul_f32 v17, v10, v13 // (op 1372) 
v_mul_f32 v18, v10, v14 // (op 1372) 
v_mul_f32 v19, v10, v15 // (op 1372) 

v_add_f32 v12, v16, v20 // (op 1369) 
v_add_f32 v13, v17, v21 // (op 1369) 
v_add_f32 v14, v18, v22 // (op 1369) 
v_add_f32 v15, v19, v23 // (op 1369) 





buffer_store_dwordx4 v[12:15], v8, s[44:47], 0 offen offset:128 // (op 1370) Store value




s_nop 1
buffer_load_dwordx4 v[12:15], v5, s[40:43], 0 offen // (op 135) Load value
v_accvgpr_read v16, a0 // (op 132) 
v_accvgpr_read v17, a1 // (op 132) 
v_accvgpr_read v18, a2 // (op 132) 
v_accvgpr_read v19, a3 // (op 132) 
v_mul_f32 v20, v11, v16 // (op 132) 
v_mul_f32 v21, v11, v17 // (op 132) 
v_mul_f32 v22, v11, v18 // (op 132) 
v_mul_f32 v23, v11, v19 // (op 132) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v10, v12 // (op 136) 
v_mul_f32 v17, v10, v13 // (op 136) 
v_mul_f32 v18, v10, v14 // (op 136) 
v_mul_f32 v19, v10, v15 // (op 136) 

v_add_f32 v12, v16, v20 // (op 133) 
v_add_f32 v13, v17, v21 // (op 133) 
v_add_f32 v14, v18, v22 // (op 133) 
v_add_f32 v15, v19, v23 // (op 133) 





buffer_store_dwordx4 v[12:15], v8, s[44:47], 0 offen // (op 134) Store value

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2_ConditionalBottom_0_beta_123:














s_endpgm  // End of RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2
.LRR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2_end:
.size RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2, .LRR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2_end-RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2
  .amdhsa_next_free_vgpr 100
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 30208 // lds bytes
  .amdhsa_accum_offset 76
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
    args: [ { type: int64, semantic: StrideC0 },
            { type: int64, semantic: StrideD0 },
            { type: address, semantic: AddressC },
            { type: int64, semantic: SizeFree1 },
            { type: address, semantic: AddressD },
            { type: int64, semantic: StrideA0 },
            { type: float32, semantic: Beta },
            { type: float32, semantic: Alpha },
            { type: address, semantic: AddressA },
            { type: int64, semantic: SizeFree0 },
            { type: int64, semantic: StrideB0 },
            { type: address, semantic: AddressB },
            { type: int64, semantic: StrideScaleA0 },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: StrideScaleB0 },
            { type: address, semantic: AddressMXScaleB },
            { type: int64, semantic: SizeSum } ]
    macrotile: [192, 32, 128]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [6, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2
    .symbol: RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 30208
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 64
    .vgpr_count: 76
    .agpr_count: 24
    .max_flat_workgroup_size: 256
    .workgroup_size: [256, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, {is-null: true}]
    .dynamic_sharedmemory_bytes: {type: LiteralValue.UInt32, dataType: UInt32, value: 0}
    .args:
      - .name: Tensor_10_stride_1_10
        .size: 8
        .offset: 0
        .expression: {type: CommandArgument, size: 8, offset: 192, name: Tensor_10_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_21_stride_1_16
        .size: 8
        .offset: 8
        .expression: {type: CommandArgument, size: 8, offset: 240, name: Tensor_21_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_10_pointer
        .size: 8
        .offset: 16
        .expression: {type: CommandArgument, size: 8, offset: 160, name: Tensor_10_pointer, variableType: {dataType: Float, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Float, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_2_size_1_4
        .size: 8
        .offset: 24
        .expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 32
        .expression: {type: CommandArgument, size: 8, offset: 208, name: Tensor_21_pointer, variableType: {dataType: Float, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Float, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_stride_0_0
        .size: 8
        .offset: 40
        .expression: {type: CommandArgument, size: 8, offset: 24, name: Tensor_0_stride_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
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
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 56
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: FP4, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP4, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_0_1
        .size: 8
        .offset: 64
        .expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_2_stride_1_3
        .size: 8
        .offset: 72
        .expression: {type: CommandArgument, size: 8, offset: 72, name: Tensor_2_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 80
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: FP4, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP4, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_stride_0_6
        .size: 8
        .offset: 88
        .expression: {type: CommandArgument, size: 8, offset: 104, name: Tensor_4_stride_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 96
        .expression: {type: CommandArgument, size: 8, offset: 80, name: Tensor_4_pointer, variableType: {dataType: E8M0, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: E8M0, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_7_stride_1_8
        .size: 8
        .offset: 104
        .expression: {type: CommandArgument, size: 8, offset: 152, name: Tensor_7_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 112
        .expression: {type: CommandArgument, size: 8, offset: 120, name: Tensor_7_pointer, variableType: {dataType: E8M0, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: E8M0, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_1_14
        .size: 8
        .offset: 120
        .expression: {type: CommandArgument, size: 8, offset: 16, name: Tensor_0_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
...

.end_amdgpu_metadata

