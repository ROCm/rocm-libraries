// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2
.p2align 8
.type RR_GEMM_TN_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2,@function
RR_GEMM_TN_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_exec_begin:


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


v_and_b32 v0, 63, v1 // (op 2859) 
v_lshrrev_b32 v3, 4, v0 // (op 2859) 

v_and_b32 v0, 1, v3 // (op 2859) 

v_and_b32 v3, 15, v1 // (op 2859) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 2859) 


v_lshrrev_b32 v0, 6, v1 // (op 2859) 
v_and_b32 v3, 1, v0 // (op 2859) 

v_lshl_add_u32 v0, v3, 5, v4 // (op 2859) 


v_and_b32 v3, 32, v1 // (op 2859) 
v_lshl_add_u32 v4, v0, 7, v3 // (op 2859) 


v_mul_lo_u32 v0, v4, 6 // (op 2859) 

v_lshrrev_b32 v3, 3, v0 // (op 2859) 

v_lshrrev_b32 v0, 6, v1 // (op 2970) 
v_and_b32 v4, 1, v0 // (op 2970) 

v_and_b32 v0, 31, v1 // (op 2970) 
v_lshl_add_u32 v5, v4, 5, v0 // (op 2970) 


v_and_b32 v0, 63, v1 // (op 2970) 
v_lshrrev_b32 v4, 5, v0 // (op 2970) 

v_lshl_add_u32 v0, v5, 2, v4 // (op 2970) 


v_and_b32 v4, 63, v1 // (op 3041) 
v_lshrrev_b32 v5, 4, v4 // (op 3041) 

v_and_b32 v4, 1, v5 // (op 3041) 

v_and_b32 v5, 15, v1 // (op 3041) 
v_lshl_add_u32 v6, v4, 4, v5 // (op 3041) 


v_lshrrev_b32 v4, 6, v1 // (op 3041) 
v_and_b32 v5, 1, v4 // (op 3041) 

v_lshl_add_u32 v4, v5, 5, v6 // (op 3041) 


v_and_b32 v5, 32, v1 // (op 3041) 
v_lshl_add_u32 v6, v4, 7, v5 // (op 3041) 


v_mul_lo_u32 v4, v6, 6 // (op 3041) 

v_lshrrev_b32 v5, 3, v4 // (op 3041) 

v_lshrrev_b32 v4, 6, v1 // (op 3094) 
v_and_b32 v6, 1, v4 // (op 3094) 

v_and_b32 v4, 31, v1 // (op 3094) 
v_lshl_add_u32 v7, v6, 5, v4 // (op 3094) 


v_and_b32 v4, 63, v1 // (op 3094) 
v_lshrrev_b32 v6, 5, v4 // (op 3094) 

v_lshl_add_u32 v4, v7, 2, v6 // (op 3094) 


v_lshrrev_b32 v6, 7, v1 // (op 2744) 
v_and_b32 v7, 63, v1 // (op 2744) 
v_lshrrev_b32 v8, 5, v7 // (op 2744) 
v_lshlrev_b32 v9, 2, v8 // (op 2744) 

v_lshl_add_u32 v8, v6, 5, v9 // (op 2744) 


v_lshl_add_u32 v6, s16, 8, v8 // (op 2744) 

v_lshrrev_b32 v8, 2, v7 // (op 2744) 

v_and_b32 v7, 7, v8 // (op 2744) 

v_and_b32 v8, 3, v1 // (op 2744) 
v_lshl_add_u32 v9, v7, 2, v8 // (op 2744) 


v_lshrrev_b32 v7, 6, v1 // (op 2744) 
v_and_b32 v8, 1, v7 // (op 2744) 

v_lshl_add_u32 v7, v8, 5, v9 // (op 2744) 


v_lshl_add_u32 v8, s17, 6, v7 // (op 2744) 

v_mul_lo_u32 v7, v8, s2 // (op 2744) 

v_add_lshl_u32 v8, v6, v7, 1 // (op 2744) 


v_and_b32 v6, 63, v1 // (op 2806) 
v_lshrrev_b32 v7, 4, v6 // (op 2806) 

v_and_b32 v6, 1, v7 // (op 2806) 

v_and_b32 v7, 15, v1 // (op 2806) 
v_lshl_add_u32 v9, v6, 4, v7 // (op 2806) 


v_lshrrev_b32 v6, 7, v1 // (op 2806) 
v_lshl_add_u32 v7, v6, 5, v9 // (op 2806) 


v_and_b32 v6, 32, v1 // (op 2806) 
v_lshl_add_u32 v9, v7, 7, v6 // (op 2806) 


v_lshlrev_b32 v6, 2, v9 // (op 2806) 

v_lshrrev_b32 v7, 3, v6 // (op 2806) 

v_lshrrev_b32 v6, 7, v1 // (op 2917) 
v_and_b32 v9, 31, v1 // (op 2917) 
s_mul_i32 s1, s2, s9 // (op 2746) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 2746) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 2746) least significant
s_mul_hi_u32 s19, s2, s8 // (op 2746) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2746) most significant: sum
s_add_u32 s1, s1, s19 // (op 2746) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 2746) 

v_lshl_add_u32 v10, v6, 5, v9 // (op 2917) 


v_and_b32 v6, 63, v1 // (op 2917) 
v_lshrrev_b32 v9, 5, v6 // (op 2917) 

v_lshl_add_u32 v6, v10, 2, v9 // (op 2917) 


v_lshrrev_b32 v9, 7, v1 // (op 2986) 
v_and_b32 v10, 63, v1 // (op 2986) 
v_lshrrev_b32 v11, 5, v10 // (op 2986) 
v_lshlrev_b32 v12, 2, v11 // (op 2986) 

v_lshl_add_u32 v11, v9, 5, v12 // (op 2986) 


v_lshl_add_u32 v9, s16, 8, v11 // (op 2986) 

v_lshrrev_b32 v11, 2, v10 // (op 2986) 

v_and_b32 v10, 7, v11 // (op 2986) 

v_and_b32 v11, 3, v1 // (op 2986) 
v_lshl_add_u32 v12, v10, 2, v11 // (op 2986) 


v_lshrrev_b32 v10, 6, v1 // (op 2986) 
v_and_b32 v11, 1, v10 // (op 2986) 

v_lshl_add_u32 v10, v11, 5, v12 // (op 2986) 


v_lshl_add_u32 v11, s17, 6, v10 // (op 2986) 

v_mul_lo_u32 v10, v11, s4 // (op 2986) 

v_add_lshl_u32 v11, v9, v10, 1 // (op 2986) 


v_and_b32 v9, 63, v1 // (op 3017) 
v_lshrrev_b32 v10, 4, v9 // (op 3017) 

v_and_b32 v9, 1, v10 // (op 3017) 

v_and_b32 v10, 15, v1 // (op 3017) 
v_lshl_add_u32 v12, v9, 4, v10 // (op 3017) 


v_lshrrev_b32 v9, 7, v1 // (op 3017) 
v_lshl_add_u32 v10, v9, 5, v12 // (op 3017) 


v_and_b32 v9, 32, v1 // (op 3017) 
v_lshl_add_u32 v12, v10, 7, v9 // (op 3017) 


v_lshlrev_b32 v9, 2, v12 // (op 3017) 

v_lshrrev_b32 v10, 3, v9 // (op 3017) 

v_lshrrev_b32 v9, 7, v1 // (op 3070) 
v_and_b32 v12, 31, v1 // (op 3070) 
s_mul_i32 s1, s4, s9 // (op 2988) most significant: low of low * high
v_lshl_add_u32 v13, v9, 5, v12 // (op 3070) 


v_and_b32 v9, 63, v1 // (op 3070) 
v_lshrrev_b32 v12, 5, v9 // (op 3070) 

v_lshl_add_u32 v9, v13, 2, v12 // (op 3070) 


s_mul_i32 s38, s5, s8 // (op 2988) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 2988) least significant
s_mul_hi_u32 s39, s4, s8 // (op 2988) most significant: high of low * low
s_add_u32 s1, s1, s38 // (op 2988) most significant: sum
s_add_u32 s1, s1, s39 // (op 2988) most significant: sum


s_lshl_b64 s[38:39], s[0:1], 1 // (op 2988) 

s_mov_b32 s40, s6 // (op 2746) reg expression
s_mov_b32 s41, s7 // (op 2746) reg expression
s_mov_b32 s42, s18 // (op 2746) reg expression
s_mov_b32 s43, 131072 // (op 2746) call()

s_mov_b32 s44, s10 // (op 2988) reg expression

s_mov_b32 s45, s11 // (op 2988) reg expression
s_mov_b32 s46, s38 // (op 2988) reg expression
s_mov_b32 s47, 131072 // (op 2988) call()


v_mov_b32 v12, s14 // (op 2) Move value
v_mov_b32 v13, s15 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 65) call()
v_accvgpr_write a1, 0.00000 // (op 65) call()
v_accvgpr_write a2, 0.00000 // (op 65) call()
v_accvgpr_write a3, 0.00000 // (op 65) call()
v_accvgpr_write a4, 0.00000 // (op 65) call()
v_accvgpr_write a5, 0.00000 // (op 65) call()
v_accvgpr_write a6, 0.00000 // (op 65) call()
v_accvgpr_write a7, 0.00000 // (op 65) call()
v_accvgpr_write a8, 0.00000 // (op 65) call()
v_accvgpr_write a9, 0.00000 // (op 65) call()
v_accvgpr_write a10, 0.00000 // (op 65) call()
v_accvgpr_write a11, 0.00000 // (op 65) call()
v_accvgpr_write a12, 0.00000 // (op 65) call()
v_accvgpr_write a13, 0.00000 // (op 65) call()
v_accvgpr_write a14, 0.00000 // (op 65) call()
v_accvgpr_write a15, 0.00000 // (op 65) call()
v_accvgpr_write a16, 0.00000 // (op 405) call()
v_accvgpr_write a17, 0.00000 // (op 405) call()
v_accvgpr_write a18, 0.00000 // (op 405) call()
v_accvgpr_write a19, 0.00000 // (op 405) call()
v_accvgpr_write a20, 0.00000 // (op 405) call()
v_accvgpr_write a21, 0.00000 // (op 405) call()
v_accvgpr_write a22, 0.00000 // (op 405) call()
v_accvgpr_write a23, 0.00000 // (op 405) call()
v_accvgpr_write a24, 0.00000 // (op 405) call()
v_accvgpr_write a25, 0.00000 // (op 405) call()
v_accvgpr_write a26, 0.00000 // (op 405) call()
v_accvgpr_write a27, 0.00000 // (op 405) call()
v_accvgpr_write a28, 0.00000 // (op 405) call()
v_accvgpr_write a29, 0.00000 // (op 405) call()
v_accvgpr_write a30, 0.00000 // (op 405) call()
v_accvgpr_write a31, 0.00000 // (op 405) call()
v_accvgpr_write a32, 0.00000 // (op 645) call()
v_accvgpr_write a33, 0.00000 // (op 645) call()
v_accvgpr_write a34, 0.00000 // (op 645) call()
v_accvgpr_write a35, 0.00000 // (op 645) call()
v_accvgpr_write a36, 0.00000 // (op 645) call()
v_accvgpr_write a37, 0.00000 // (op 645) call()
v_accvgpr_write a38, 0.00000 // (op 645) call()
v_accvgpr_write a39, 0.00000 // (op 645) call()
v_accvgpr_write a40, 0.00000 // (op 645) call()
v_accvgpr_write a41, 0.00000 // (op 645) call()
v_accvgpr_write a42, 0.00000 // (op 645) call()
v_accvgpr_write a43, 0.00000 // (op 645) call()
v_accvgpr_write a44, 0.00000 // (op 645) call()
v_accvgpr_write a45, 0.00000 // (op 645) call()
v_accvgpr_write a46, 0.00000 // (op 645) call()
v_accvgpr_write a47, 0.00000 // (op 645) call()
v_accvgpr_write a48, 0.00000 // (op 885) call()
v_accvgpr_write a49, 0.00000 // (op 885) call()
v_accvgpr_write a50, 0.00000 // (op 885) call()
v_accvgpr_write a51, 0.00000 // (op 885) call()
v_accvgpr_write a52, 0.00000 // (op 885) call()
v_accvgpr_write a53, 0.00000 // (op 885) call()
v_accvgpr_write a54, 0.00000 // (op 885) call()
v_accvgpr_write a55, 0.00000 // (op 885) call()
v_accvgpr_write a56, 0.00000 // (op 885) call()
v_accvgpr_write a57, 0.00000 // (op 885) call()
v_accvgpr_write a58, 0.00000 // (op 885) call()
v_accvgpr_write a59, 0.00000 // (op 885) call()
v_accvgpr_write a60, 0.00000 // (op 885) call()
v_accvgpr_write a61, 0.00000 // (op 885) call()
v_accvgpr_write a62, 0.00000 // (op 885) call()
v_accvgpr_write a63, 0.00000 // (op 885) call()
v_and_b32 v14, 4294967292, v1 // (op 2511) 
v_lshl_add_u32 v15, s16, 8, v14 // (op 2511) 

v_and_b32 v14, -4, v1 // (op 2511) 
v_sub_u32 v16, v1, v14 // (op 2511) 

v_lshlrev_b32 v14, 5, v16 // (op 2511) 

v_mov_b32 v17, 0 // (op 2511) convert
v_mov_b32 v16, v14 // (op 2511) convert

v_mul_lo_u32 v19, v15, s13 // (op 2511) most significant: low of low * high
v_mul_hi_u32 v14, v15, s12 // (op 2511) most significant: high of low * low
v_mul_lo_u32 v18, v15, s12 // (op 2511) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 2511) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 2511) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 2511) most significant half



v_lshlrev_b64 v[14:15], 2, v[18:19] // (op 2511) 

v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 2511) 

v_mov_b32 v14, v16 // (op 2511) convert


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 2513) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 2513) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 2513) least significant
s_mul_hi_u32 s7, s12, s22 // (op 2513) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2513) most significant: sum
s_add_u32 s1, s1, s7 // (op 2513) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 2513) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 2513) 

s_mov_b32 s48, s20 // (op 2513) reg expression
s_mov_b32 s49, s21 // (op 2513) reg expression
s_mov_b32 s50, s0 // (op 2513) reg expression
s_mov_b32 s51, 131072 // (op 2513) call()

v_and_b32 v15, 4294967292, v1 // (op 2530) 
v_lshl_add_u32 v16, s16, 8, v15 // (op 2530) 

v_and_b32 v15, -4, v1 // (op 2530) 
v_sub_u32 v17, v1, v15 // (op 2530) 

v_lshlrev_b32 v15, 5, v17 // (op 2530) 

v_mov_b32 v19, 0 // (op 2530) convert
v_mov_b32 v18, v15 // (op 2530) convert

v_mul_lo_u32 v21, v16, s13 // (op 2530) most significant: low of low * high
v_mul_hi_u32 v15, v16, s12 // (op 2530) most significant: high of low * low
v_mul_lo_u32 v20, v16, s12 // (op 2530) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 2530) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v18 // (op 2530) least significant half
v_addc_co_u32 v21, s[0:1], v21, v19, s[0:1] // (op 2530) most significant half



v_lshlrev_b64 v[18:19], 2, v[20:21] // (op 2530) 

v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 2530) 

v_mov_b32 v15, v16 // (op 2530) convert

s_mul_i32 s1, s12, s23 // (op 2532) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 2532) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 2532) least significant
s_mul_hi_u32 s7, s12, s22 // (op 2532) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2532) most significant: sum
s_add_u32 s1, s1, s7 // (op 2532) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 2532) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 2532) 

s_mov_b32 s48, s20 // (op 2532) reg expression
s_mov_b32 s49, s21 // (op 2532) reg expression
s_mov_b32 s50, s0 // (op 2532) reg expression
s_mov_b32 s51, 131072 // (op 2532) call()

v_and_b32 v16, -4, v1 // (op 2575) 
v_sub_u32 v17, v1, v16 // (op 2575) 

v_lshlrev_b32 v16, 5, v17 // (op 2575) 

v_mov_b32 v19, 0 // (op 2575) convert
v_mov_b32 v18, v16 // (op 2575) convert

v_lshrrev_b32 v16, 2, v1 // (op 2575) 
v_lshl_add_u32 v17, s17, 6, v16 // (op 2575) 

v_mul_lo_u32 v21, v17, s25 // (op 2575) most significant: low of low * high
v_mul_hi_u32 v16, v17, s24 // (op 2575) most significant: high of low * low
v_mul_lo_u32 v20, v17, s24 // (op 2575) least significant: low of low * low
v_add_u32 v21, v21, v16 // (op 2575) most significant: sum


v_add_co_u32 v16, s[0:1], v18, v20 // (op 2575) least significant half
v_addc_co_u32 v17, s[0:1], v19, v21, s[0:1] // (op 2575) most significant half



v_mul_lo_u32 v19, v17, 6 // (op 2575) most significant: low of high * low
v_mul_hi_u32 v20, v16, 6 // (op 2575) most significant: high of low * low
v_mul_lo_u32 v18, v16, 6 // (op 2575) least significant: low of low * low
v_add_u32 v19, v19, v20 // (op 2575) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 2575) 

v_mov_b32 v18, v16 // (op 2575) convert

s_mul_i32 s1, s9, 6 // (op 2577) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 2577) least significant
s_mul_hi_u32 s6, s8, 6 // (op 2577) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2577) most significant: sum

s_mul_i32 s7, s24, s1 // (op 2577) most significant: low of low * high
s_mul_i32 s10, s25, s0 // (op 2577) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 2577) least significant
s_mul_hi_u32 s11, s24, s0 // (op 2577) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 2577) most significant: sum
s_add_u32 s7, s7, s11 // (op 2577) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 2577) 

s_mov_b32 s52, s26 // (op 2577) reg expression
s_mov_b32 s53, s27 // (op 2577) reg expression
s_mov_b32 s54, s0 // (op 2577) reg expression
s_mov_b32 s55, 131072 // (op 2577) call()

v_and_b32 v16, -4, v1 // (op 2594) 
v_sub_u32 v17, v1, v16 // (op 2594) 

v_lshlrev_b32 v16, 5, v17 // (op 2594) 

v_mov_b32 v21, 0 // (op 2594) convert
v_mov_b32 v20, v16 // (op 2594) convert

v_lshrrev_b32 v19, 2, v1 // (op 2594) 
v_lshl_add_u32 v16, s17, 6, v19 // (op 2594) 

v_mul_lo_u32 v23, v16, s25 // (op 2594) most significant: low of low * high
v_mul_hi_u32 v17, v16, s24 // (op 2594) most significant: high of low * low
v_mul_lo_u32 v22, v16, s24 // (op 2594) least significant: low of low * low
v_add_u32 v23, v23, v17 // (op 2594) most significant: sum


v_add_co_u32 v16, s[0:1], v20, v22 // (op 2594) least significant half
v_addc_co_u32 v17, s[0:1], v21, v23, s[0:1] // (op 2594) most significant half



v_mul_lo_u32 v21, v17, 6 // (op 2594) most significant: low of high * low
v_mul_hi_u32 v19, v16, 6 // (op 2594) most significant: high of low * low
v_mul_lo_u32 v20, v16, 6 // (op 2594) least significant: low of low * low
v_add_u32 v21, v21, v19 // (op 2594) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[20:21] // (op 2594) 

v_mov_b32 v19, v16 // (op 2594) convert

s_mul_i32 s1, s9, 6 // (op 2596) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 2596) least significant
s_mul_hi_u32 s6, s8, 6 // (op 2596) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2596) most significant: sum

s_mul_i32 s7, s24, s1 // (op 2596) most significant: low of low * high
s_mul_i32 s10, s25, s0 // (op 2596) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 2596) least significant
s_mul_hi_u32 s11, s24, s0 // (op 2596) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 2596) most significant: sum
s_add_u32 s7, s7, s11 // (op 2596) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 2596) 

s_mov_b32 s52, s26 // (op 2596) reg expression
s_mov_b32 s53, s27 // (op 2596) reg expression
s_mov_b32 s54, s0 // (op 2596) reg expression
s_mov_b32 s55, 131072 // (op 2596) call()

v_lshl_add_u32 v16, s16, 8, v1 // (op 2639) 
v_mul_lo_u32 v17, v16, s28 // (op 2639) 

s_mul_i32 s1, s28, s23 // (op 2641) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 2641) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 2641) least significant
s_mul_hi_u32 s7, s28, s22 // (op 2641) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2641) most significant: sum
s_add_u32 s1, s1, s7 // (op 2641) most significant: sum


s_mov_b32 s56, s30 // (op 2641) reg expression
s_mov_b32 s57, s31 // (op 2641) reg expression
s_mov_b32 s58, s0 // (op 2641) reg expression
s_mov_b32 s59, 131072 // (op 2641) call()

v_lshl_add_u32 v16, s16, 8, v1 // (op 2658) 
v_mul_lo_u32 v20, v16, s28 // (op 2658) 

s_mul_i32 s1, s28, s23 // (op 2660) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 2660) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 2660) least significant
s_mul_hi_u32 s7, s28, s22 // (op 2660) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2660) most significant: sum
s_add_u32 s1, s1, s7 // (op 2660) most significant: sum


s_mov_b32 s56, s30 // (op 2660) reg expression
s_mov_b32 s57, s31 // (op 2660) reg expression
s_mov_b32 s58, s0 // (op 2660) reg expression
s_mov_b32 s59, 131072 // (op 2660) call()

v_and_b32 v16, -4, v1 // (op 2703) 
v_sub_u32 v21, v1, v16 // (op 2703) 

v_lshrrev_b32 v16, 2, v1 // (op 2703) 
v_lshl_add_u32 v22, s17, 6, v16 // (op 2703) 

v_mul_lo_u32 v16, v22, s32 // (op 2703) 

v_add_u32 v22, v21, v16 // (op 2703) 


s_mul_i32 s1, s32, s9 // (op 2705) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 2705) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 2705) least significant
s_mul_hi_u32 s7, s32, s8 // (op 2705) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2705) most significant: sum
s_add_u32 s1, s1, s7 // (op 2705) most significant: sum


s_mov_b32 s60, s34 // (op 2705) reg expression
s_mov_b32 s61, s35 // (op 2705) reg expression
s_mov_b32 s62, s0 // (op 2705) reg expression
s_mov_b32 s63, 131072 // (op 2705) call()

v_and_b32 v16, -4, v1 // (op 2722) 
v_sub_u32 v21, v1, v16 // (op 2722) 

v_lshrrev_b32 v16, 2, v1 // (op 2722) 
v_lshl_add_u32 v23, s17, 6, v16 // (op 2722) 

v_mul_lo_u32 v16, v23, s32 // (op 2722) 

v_add_u32 v23, v21, v16 // (op 2722) 


s_mul_i32 s1, s32, s9 // (op 2724) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 2724) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 2724) least significant
s_mul_hi_u32 s7, s32, s8 // (op 2724) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2724) most significant: sum
s_add_u32 s1, s1, s7 // (op 2724) most significant: sum


s_mov_b32 s60, s34 // (op 2724) reg expression
s_mov_b32 s61, s35 // (op 2724) reg expression
s_mov_b32 s62, s0 // (op 2724) reg expression
s_mov_b32 s63, 131072 // (op 2724) call()




buffer_load_dwordx4 v[24:27], v18, s[52:55], 0 offen // (op 2249) Load value
buffer_load_dwordx2 v[28:29], v18, s[52:55], 0 offen offset:16 // (op 2249) Load value



buffer_load_ubyte v16, v22, s[60:63], 0 offen offset: 0 // (op 2252) Load value

s_lshl_b64 s[0:1], s[12:13], 2 // (op 2255) 
v_lshrrev_b64 v[30:31], 3, s[0:1] // (op 2255) 


v_mov_b32 v21, v14 // (op 2255) 
buffer_load_dwordx4 v[32:35], v21, s[48:51], 0 offen // (op 2255) Load value
v_add_u32 v21, v21, v30 // (op 2255) 
buffer_load_dwordx4 v[36:39], v21, s[48:51], 0 offen // (op 2255) Load value
v_add_u32 v21, v21, v30 // (op 2255) 
buffer_load_dwordx4 v[40:43], v21, s[48:51], 0 offen // (op 2255) Load value
v_add_u32 v21, v21, v30 // (op 2255) 
buffer_load_dwordx4 v[44:47], v21, s[48:51], 0 offen // (op 2255) Load value





buffer_load_dword v21, v17, s[56:59], 0 offen // (op 2258) Load value



buffer_load_dwordx4 v[48:51], v18, s[52:55], 0 offen offset:96 // (op 2261) Load value
buffer_load_dwordx2 v[52:53], v18, s[52:55], 0 offen offset:112 // (op 2261) Load value




buffer_load_ubyte v18, v22, s[60:63], 0 offen offset:4 // (op 2264) Load value


s_lshl_b64 s[0:1], s[12:13], 2 // (op 2267) 
v_lshrrev_b64 v[30:31], 3, s[0:1] // (op 2267) 


v_mov_b32 v22, v14 // (op 2267) 
buffer_load_dwordx4 v[54:57], v22, s[48:51], 0 offen offset:64 // (op 2267) Load value
v_add_u32 v22, v22, v30 // (op 2267) 
buffer_load_dwordx4 v[58:61], v22, s[48:51], 0 offen offset:64 // (op 2267) Load value
v_add_u32 v22, v22, v30 // (op 2267) 
buffer_load_dwordx4 v[62:65], v22, s[48:51], 0 offen offset:64 // (op 2267) Load value
v_add_u32 v22, v22, v30 // (op 2267) 
buffer_load_dwordx4 v[66:69], v22, s[48:51], 0 offen offset:64 // (op 2267) Load value






buffer_load_dword v14, v17, s[56:59], 0 offen offset:4 // (op 2270) Load value

v_and_b32 v17, -4, v1 // (op 2847) 
v_sub_u32 v22, v1, v17 // (op 2847) 

v_lshlrev_b32 v17, 5, v22 // (op 2847) 

v_lshrrev_b32 v22, 2, v1 // (op 2847) 
v_lshl_add_u32 v30, v22, 7, v17 // (op 2847) 


v_mul_lo_u32 v17, v30, 6 // (op 2847) 

v_lshrrev_b32 v22, 3, v17 // (op 2847) 




s_waitcnt vmcnt(15)
ds_write_b128 v22, v[24:27] // (op 2272) Store local data 
s_waitcnt vmcnt(14)
ds_write_b64 v22, v[28:29] offset:16 // (op 2272) Store local data 

v_and_b32 v17, -4, v1 // (op 2958) 
v_sub_u32 v22, v1, v17 // (op 2958) 

v_and_b32 v17, 4294967292, v1 // (op 2958) 
v_add_u32 v30, v17, v22 // (op 2958) 





s_waitcnt vmcnt(13)
ds_write_b8 v30, v16 offset:6144 // (op 2273) Store local data 

v_and_b32 v17, -4, v1 // (op 2789) 
v_sub_u32 v22, v1, v17 // (op 2789) 

v_lshlrev_b32 v17, 5, v22 // (op 2789) 

v_lshrrev_b32 v22, 2, v1 // (op 2789) 
v_lshl_add_u32 v30, v22, 9, v17 // (op 2789) 


v_lshlrev_b32 v17, 2, v30 // (op 2789) 

v_lshrrev_b32 v22, 3, v17 // (op 2789) 




s_waitcnt vmcnt(12)
ds_write_b128 v22, v[32:35] offset:6400 // (op 2274) Store local data 
s_waitcnt vmcnt(11)
ds_write_b128 v22, v[36:39] offset:6464 // (op 2274) Store local data 
s_waitcnt vmcnt(10)
ds_write_b128 v22, v[40:43] offset:6528 // (op 2274) Store local data 
s_waitcnt vmcnt(9)
ds_write_b128 v22, v[44:47] offset:6592 // (op 2274) Store local data 

v_lshlrev_b32 v17, 2, v1 // (op 2900) 



s_waitcnt vmcnt(8)
ds_write_b32 v17, v21 offset:22784 // (op 2275) Store local data 

s_waitcnt lgkmcnt(6)
s_barrier  // (op 3176) 



ds_read_b128 v[70:73], v3 // (op 2309) Load local data 
ds_read_b64 v[74:75], v3 offset:16 // (op 2309) Load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 3217) 



ds_read_u8 v17, v0 offset:6144 // (op 2332) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 3157) 



ds_read_b128 v[76:79], v7 offset:6400 // (op 2290) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 3197) 



ds_read_u8 v22, v6 offset:22784 // (op 2313) Load local data 



ds_read_b128 v[80:83], v7 offset:10496 // (op 2295) Load local data 



ds_read_u8 v30, v6 offset:23040 // (op 2318) Load local data 



ds_read_b128 v[84:87], v7 offset:14592 // (op 2300) Load local data 



ds_read_u8 v31, v6 offset:23296 // (op 2323) Load local data 



ds_read_b128 v[88:91], v7 offset:18688 // (op 2305) Load local data 



ds_read_u8 v92, v6 offset:23552 // (op 2328) Load local data 
v_and_b32 v93, -4, v1 // (op 2760) 
v_sub_u32 v94, v1, v93 // (op 2760) 

v_lshlrev_b32 v93, 5, v94 // (op 2760) 

v_lshrrev_b32 v94, 2, v1 // (op 2760) 
v_lshl_add_u32 v95, v94, 9, v93 // (op 2760) 


v_lshlrev_b32 v93, 2, v95 // (op 2760) 

v_lshrrev_b32 v94, 3, v93 // (op 2760) 

v_and_b32 v93, -4, v1 // (op 2818) 
v_sub_u32 v95, v1, v93 // (op 2818) 

v_lshlrev_b32 v93, 5, v95 // (op 2818) 

v_lshrrev_b32 v95, 2, v1 // (op 2818) 
v_lshl_add_u32 v96, v95, 7, v93 // (op 2818) 


v_mul_lo_u32 v93, v96, 6 // (op 2818) 

v_lshrrev_b32 v95, 3, v93 // (op 2818) 

v_lshlrev_b32 v93, 2, v1 // (op 2871) 
v_and_b32 v96, -4, v1 // (op 2929) 
v_sub_u32 v97, v1, v96 // (op 2929) 

v_and_b32 v96, 4294967292, v1 // (op 2929) 
v_add_u32 v98, v96, v97 // (op 2929) 


v_and_b32 v96, -4, v1 // (op 3000) 
v_sub_u32 v97, v1, v96 // (op 3000) 

v_lshlrev_b32 v96, 5, v97 // (op 3000) 

v_lshrrev_b32 v97, 2, v1 // (op 3000) 
v_lshl_add_u32 v99, v97, 9, v96 // (op 3000) 


v_lshlrev_b32 v96, 2, v99 // (op 3000) 

v_lshrrev_b32 v97, 3, v96 // (op 3000) 

v_and_b32 v96, -4, v1 // (op 3029) 
v_sub_u32 v99, v1, v96 // (op 3029) 

v_lshlrev_b32 v96, 5, v99 // (op 3029) 

v_lshrrev_b32 v99, 2, v1 // (op 3029) 
v_lshl_add_u32 v100, v99, 7, v96 // (op 3029) 


v_mul_lo_u32 v96, v100, 6 // (op 3029) 

v_lshrrev_b32 v99, 3, v96 // (op 3029) 

v_lshlrev_b32 v96, 2, v1 // (op 3053) 
v_and_b32 v100, -4, v1 // (op 3082) 
v_sub_u32 v101, v1, v100 // (op 3082) 

v_and_b32 v100, 4294967292, v1 // (op 3082) 
v_add_u32 v102, v100, v101 // (op 3082) 


s_mov_b32 s0, 0 // (op 887) call()
s_ashr_i64 s[6:7], s[36:37], 7 // (op 886) 
s_mov_b32 s1, s6 // (op 886) convert

s_and_b32 s6, s1, 4294967294 // (op 886) 

s_cmp_lt_u32 s0, s6 // (op 886) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ForLoopBottom_KLoop_886 // (op 886) Condition: Top (jump to RR_GEMM_TN_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ForLoopBottom_KLoop_886 if false)
RR_GEMM_TN_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ForLoopTop_KLoop_886:


s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[76:79], v[70:75], a[0:15], v22, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 90) 
s_add_u32 s0, s0, 2 // (op 888) 
v_mov_b32 v100, v19 // (op 10) 

buffer_load_dwordx4 v[24:27], v100, s[52:55], 0 offen offset:192 // (op 10) Load value
buffer_load_dwordx2 v[28:29], v100, s[52:55], 0 offen offset:208 // (op 10) Load value
s_waitcnt lgkmcnt(1)
s_barrier  // (op 3166) 





ds_read_b128 v[76:79], v7 offset:6432 // (op 93) Load local data 
v_mov_b32 v101, v23 // (op 25) 

s_barrier  // (op 3186) 
buffer_load_ubyte v16, v101, s[60:63], 0 offen offset:8 // (op 25) Load value




s_lshl_b64 s[6:7], s[12:13], 2 // (op 4) 
v_lshrrev_b64 v[104:105], 3, s[6:7] // (op 4) 


v_mov_b32 v103, v15 // (op 4) 
v_mov_b32 v106, v103 // (op 4) 
ds_read_b128 v[108:111], v3 offset:48 // (op 97) Load local data 
ds_read_b64 v[112:113], v3 offset:64 // (op 97) Load local data 
buffer_load_dwordx4 v[32:35], v106, s[48:51], 0 offen offset:128 // (op 4) Load value
v_add_u32 v106, v106, v104 // (op 4) 
buffer_load_dwordx4 v[36:39], v106, s[48:51], 0 offen offset:128 // (op 4) Load value
v_add_u32 v106, v106, v104 // (op 4) 
buffer_load_dwordx4 v[40:43], v106, s[48:51], 0 offen offset:128 // (op 4) Load value
v_add_u32 v106, v106, v104 // (op 4) 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 3207) 
buffer_load_dwordx4 v[44:47], v106, s[48:51], 0 offen offset:128 // (op 4) Load value







ds_read_u8 v22, v6 offset:22786 // (op 101) Load local data 
v_mov_b32 v104, v20 // (op 17) 

s_barrier  // (op 3227) 
buffer_load_dword v21, v104, s[56:59], 0 offen offset:8 // (op 17) Load value



s_barrier  // (op 3247) 
ds_read_u8 v105, v0 offset:6146 // (op 105) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[76:79], v[108:113], a[0:15], v22, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 107) 
s_waitcnt vmcnt(15)
ds_write_b128 v99, v[48:51] offset:23808 // (op 306) Store local data 
s_waitcnt vmcnt(14)
ds_write_b64 v99, v[52:53] offset:23824 // (op 306) Store local data 
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[80:83], v[70:75], a[16:31], v30, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 454) 



s_barrier  // (op 3259) 
ds_read_b128 v[76:79], v7 offset:10528 // (op 456) Load local data 






ds_read_u8 v22, v6 offset:23042 // (op 460) Load local data 
s_waitcnt vmcnt(13)
ds_write_b8 v102, v18 offset:29952 // (op 308) Store local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[76:79], v[108:113], a[16:31], v22, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 463) 
s_barrier  // (op 3240) 



v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[84:87], v[70:75], a[32:47], v31, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 694) 
s_waitcnt vmcnt(12)
ds_write_b128 v97, v[54:57] offset:30208 // (op 287) Store local data 



ds_read_b128 v[76:79], v7 offset:14624 // (op 696) Load local data 
s_waitcnt vmcnt(11)
ds_write_b128 v97, v[58:61] offset:30272 // (op 287) Store local data 



ds_read_u8 v22, v6 offset:23298 // (op 700) Load local data 
s_waitcnt vmcnt(10)
ds_write_b128 v97, v[62:65] offset:30336 // (op 287) Store local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[76:79], v[108:113], a[32:47], v22, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 703) 
s_waitcnt vmcnt(9)
ds_write_b128 v97, v[66:69] offset:30400 // (op 287) Store local data 
s_barrier  // (op 3253) 
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[88:91], v[70:75], a[48:63], v92, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 934) 






ds_read_b128 v[88:91], v7 offset:18720 // (op 936) Load local data 
s_waitcnt vmcnt(8)
ds_write_b32 v96, v14 offset:46592 // (op 307) Store local data 



ds_read_u8 v92, v6 offset:23554 // (op 940) Load local data 
s_barrier  // (op 3243) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[88:91], v[108:113], a[48:63], v92, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 943) 





ds_read_b128 v[70:73], v5 offset:23808 // (op 291) Load local data 
ds_read_b64 v[74:75], v5 offset:23824 // (op 291) Load local data 
s_barrier  // (op 3256) 



ds_read_u8 v17, v4 offset:29952 // (op 295) Load local data 
s_barrier  // (op 3237) 



ds_read_b128 v[76:79], v10 offset:30208 // (op 289) Load local data 
s_barrier  // (op 3250) 



ds_read_u8 v22, v9 offset:46592 // (op 293) Load local data 



ds_read_b128 v[80:83], v10 offset:34304 // (op 475) Load local data 



ds_read_u8 v30, v9 offset:46848 // (op 479) Load local data 



ds_read_b128 v[84:87], v10 offset:38400 // (op 715) Load local data 



ds_read_u8 v31, v9 offset:47104 // (op 719) Load local data 



ds_read_b128 v[106:109], v10 offset:42496 // (op 955) Load local data 



ds_read_u8 v105, v9 offset:47360 // (op 959) Load local data 



s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[76:79], v[70:75], a[0:15], v22, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 296) 
buffer_load_dwordx4 v[48:51], v100, s[52:55], 0 offen offset:288 // (op 309) Load value
buffer_load_dwordx2 v[52:53], v100, s[52:55], 0 offen offset:304 // (op 309) Load value




ds_read_b128 v[110:113], v10 offset:30240 // (op 298) Load local data 






buffer_load_ubyte v18, v101, s[60:63], 0 offen offset:12 // (op 311) Load value
v_add_u32 v19, 192, v19 // (op 2586) 

s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[80:83], v[70:75], a[16:31], v30, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 464) 

v_add_u32 v23, 8, v23 // (op 2714) 
s_lshl_b64 s[6:7], s[12:13], 2 // (op 286) 
v_lshrrev_b64 v[100:101], 3, s[6:7] // (op 286) 


v_mov_b32 v114, v103 // (op 286) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[84:87], v[70:75], a[32:47], v31, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 704) 
ds_read_b128 v[116:119], v5 offset:23856 // (op 300) Load local data 
ds_read_b64 v[120:121], v5 offset:23872 // (op 300) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[106:109], v[70:75], a[48:63], v105, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 944) 





s_barrier  // (op 3308) 
ds_read_u8 v115, v9 offset:46594 // (op 302) Load local data 
buffer_load_dwordx4 v[54:57], v114, s[48:51], 0 offen offset:192 // (op 286) Load value



v_add_u32 v114, v114, v100 // (op 286) 
buffer_load_dwordx4 v[58:61], v114, s[48:51], 0 offen offset:192 // (op 286) Load value
v_add_u32 v114, v114, v100 // (op 286) 
buffer_load_dwordx4 v[62:65], v114, s[48:51], 0 offen offset:192 // (op 286) Load value
v_add_u32 v114, v114, v100 // (op 286) 
ds_read_u8 v105, v4 offset:29954 // (op 304) Load local data 
buffer_load_dwordx4 v[66:69], v114, s[48:51], 0 offen offset:192 // (op 286) Load value









s_barrier  // (op 3312) 
buffer_load_dword v14, v104, s[56:59], 0 offen offset:12 // (op 310) Load value
v_add_u32 v15, 128, v15 // (op 2522) 

ds_read_b128 v[106:109], v10 offset:34336 // (op 466) Load local data 



s_barrier  // (op 3179) 
v_add_u32 v20, 8, v20 // (op 2650) 



ds_read_u8 v114, v9 offset:46850 // (op 470) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[110:113], v[116:121], a[0:15], v115, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 305) 





ds_read_b128 v[110:113], v10 offset:38432 // (op 706) Load local data 



s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[106:109], v[116:121], a[16:31], v114, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 473) 
ds_read_u8 v115, v9 offset:47106 // (op 710) Load local data 





ds_read_b128 v[106:109], v10 offset:42528 // (op 946) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[110:113], v[116:121], a[32:47], v115, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 713) 





s_barrier  // (op 3368) 
ds_read_u8 v100, v9 offset:47362 // (op 950) Load local data 
s_waitcnt vmcnt(15)
ds_write_b128 v95, v[24:27] // (op 13) Store local data 
s_waitcnt vmcnt(14)
ds_write_b64 v95, v[28:29] offset:16 // (op 13) Store local data 
s_barrier  // (op 3372) 
s_barrier  // (op 3220) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[106:109], v[116:121], a[48:63], v100, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 953) 







s_waitcnt vmcnt(13)
ds_write_b8 v98, v16 offset:6144 // (op 30) Store local data 
s_barrier  // (op 3160) 



s_waitcnt vmcnt(12)
ds_write_b128 v94, v[32:35] offset:6400 // (op 6) Store local data 
s_waitcnt vmcnt(11)
ds_write_b128 v94, v[36:39] offset:6464 // (op 6) Store local data 
s_waitcnt vmcnt(10)
ds_write_b128 v94, v[40:43] offset:6528 // (op 6) Store local data 
s_waitcnt vmcnt(9)
ds_write_b128 v94, v[44:47] offset:6592 // (op 6) Store local data 
s_barrier  // (op 3200) 



s_waitcnt vmcnt(8)
ds_write_b32 v93, v21 offset:22784 // (op 21) Store local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 3189) 



ds_read_b128 v[70:73], v3 // (op 77) Load local data 
ds_read_b64 v[74:75], v3 offset:16 // (op 77) Load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 3230) 



ds_read_u8 v17, v0 offset:6144 // (op 87) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 3169) 



ds_read_b128 v[76:79], v7 offset:6400 // (op 72) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 3210) 



ds_read_u8 v22, v6 offset:22784 // (op 82) Load local data 



ds_read_b128 v[80:83], v7 offset:10496 // (op 447) Load local data 



ds_read_u8 v30, v6 offset:23040 // (op 451) Load local data 



ds_read_b128 v[84:87], v7 offset:14592 // (op 687) Load local data 



ds_read_u8 v31, v6 offset:23296 // (op 691) Load local data 



ds_read_b128 v[88:91], v7 offset:18688 // (op 927) Load local data 



ds_read_u8 v92, v6 offset:23552 // (op 931) Load local data 
s_ashr_i64 s[6:7], s[36:37], 7 // (op 886) 
s_mov_b32 s1, s6 // (op 886) convert

s_and_b32 s6, s1, 4294967294 // (op 886) 

s_cmp_lt_u32 s0, s6 // (op 886) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ForLoopTop_KLoop_886 // (op 886) Condition: Bottom (jump to RR_GEMM_TN_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ForLoopTop_KLoop_886 if true)
RR_GEMM_TN_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ForLoopBottom_KLoop_886:

















v_and_b32 v15, 4294967292, v1 // (op 2482) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 2482) 
v_lshl_add_u32 v16, s16, 8, v15 // (op 2482) 

s_lshr_b32 s6, s0, 1 // (op 2482) 

v_and_b32 v15, -4, v1 // (op 2482) 
v_sub_u32 v19, v1, v15 // (op 2482) 

v_lshlrev_b32 v15, 5, v19 // (op 2482) 

v_lshl_add_u32 v19, s6, 8, v15 // (op 2482) 


v_mov_b32 v21, 0 // (op 2482) convert
v_mov_b32 v20, v19 // (op 2482) convert

v_mul_lo_u32 v29, v16, s13 // (op 2482) most significant: low of low * high
v_mul_hi_u32 v15, v16, s12 // (op 2482) most significant: high of low * low
v_mul_lo_u32 v28, v16, s12 // (op 2482) least significant: low of low * low
v_add_u32 v29, v29, v15 // (op 2482) most significant: sum

v_add_co_u32 v28, s[0:1], v28, v20 // (op 2482) least significant half
v_addc_co_u32 v29, s[0:1], v29, v21, s[0:1] // (op 2482) most significant half



v_lshlrev_b64 v[20:21], 2, v[28:29] // (op 2482) 

v_lshrrev_b64 v[28:29], 3, v[20:21] // (op 2482) 

v_mov_b32 v15, v28 // (op 2482) convert

s_mul_i32 s1, s12, s23 // (op 2484) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 2484) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 2484) least significant
s_mul_hi_u32 s7, s12, s22 // (op 2484) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2484) most significant: sum
s_add_u32 s1, s1, s7 // (op 2484) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 2484) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 2484) 

s_mov_b32 s48, s20 // (op 2484) reg expression
s_mov_b32 s49, s21 // (op 2484) reg expression
s_mov_b32 s50, s0 // (op 2484) reg expression
s_mov_b32 s51, 131072 // (op 2484) call()

v_and_b32 v16, -4, v1 // (op 2552) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 2552) 
v_sub_u32 v19, v1, v16 // (op 2552) 

s_lshr_b32 s6, s0, 1 // (op 2552) 

v_lshlrev_b32 v16, 5, v19 // (op 2552) 

v_lshl_add_u32 v19, s6, 8, v16 // (op 2552) 


v_mov_b32 v21, 0 // (op 2552) convert
v_mov_b32 v20, v19 // (op 2552) convert

v_lshrrev_b32 v16, 2, v1 // (op 2552) 
v_lshl_add_u32 v19, s17, 6, v16 // (op 2552) 

v_mul_lo_u32 v29, v19, s25 // (op 2552) most significant: low of low * high
v_mul_hi_u32 v16, v19, s24 // (op 2552) most significant: high of low * low
v_mul_lo_u32 v28, v19, s24 // (op 2552) least significant: low of low * low
v_add_u32 v29, v29, v16 // (op 2552) most significant: sum


v_add_co_u32 v26, s[0:1], v20, v28 // (op 2552) least significant half
v_addc_co_u32 v27, s[0:1], v21, v29, s[0:1] // (op 2552) most significant half



v_mul_lo_u32 v21, v27, 6 // (op 2552) most significant: low of high * low
v_mul_hi_u32 v16, v26, 6 // (op 2552) most significant: high of low * low
v_mul_lo_u32 v20, v26, 6 // (op 2552) least significant: low of low * low
v_add_u32 v21, v21, v16 // (op 2552) most significant: sum


v_lshrrev_b64 v[28:29], 3, v[20:21] // (op 2552) 

v_mov_b32 v16, v28 // (op 2552) convert

s_mul_i32 s1, s9, 6 // (op 2554) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 2554) least significant
s_mul_hi_u32 s6, s8, 6 // (op 2554) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2554) most significant: sum

s_mul_i32 s7, s24, s1 // (op 2554) most significant: low of low * high
s_mul_i32 s10, s25, s0 // (op 2554) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 2554) least significant
s_mul_hi_u32 s11, s24, s0 // (op 2554) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 2554) most significant: sum
s_add_u32 s7, s7, s11 // (op 2554) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 2554) 

s_mov_b32 s52, s26 // (op 2554) reg expression
s_mov_b32 s53, s27 // (op 2554) reg expression
s_mov_b32 s54, s0 // (op 2554) reg expression
s_mov_b32 s55, 131072 // (op 2554) call()

v_lshl_add_u32 v19, s16, 8, v1 // (op 2616) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 2616) 
s_lshr_b32 s6, s0, 1 // (op 2616) 

s_lshl_b32 s7, s6, 3 // (op 2616) 

v_mul_lo_u32 v20, v19, s28 // (op 2616) 
v_add_u32 v20, v20, s7 // (op 2616) 


s_mul_i32 s1, s28, s23 // (op 2618) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 2618) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 2618) least significant
s_mul_hi_u32 s7, s28, s22 // (op 2618) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2618) most significant: sum
s_add_u32 s1, s1, s7 // (op 2618) most significant: sum


s_mov_b32 s56, s30 // (op 2618) reg expression
s_mov_b32 s57, s31 // (op 2618) reg expression
s_mov_b32 s58, s0 // (op 2618) reg expression
s_mov_b32 s59, 131072 // (op 2618) call()

v_and_b32 v19, -4, v1 // (op 2680) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 2680) 
v_sub_u32 v21, v1, v19 // (op 2680) 

s_lshr_b32 s6, s0, 1 // (op 2680) 

v_lshrrev_b32 v19, 2, v1 // (op 2680) 
v_lshl_add_u32 v23, s6, 3, v21 // (op 2680) 


v_lshl_add_u32 v21, s17, 6, v19 // (op 2680) 

v_mul_lo_u32 v19, v21, s32 // (op 2680) 

v_add_u32 v21, v23, v19 // (op 2680) 


s_mul_i32 s1, s32, s9 // (op 2682) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 2682) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 2682) least significant
s_mul_hi_u32 s7, s32, s8 // (op 2682) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2682) most significant: sum
s_add_u32 s1, s1, s7 // (op 2682) most significant: sum


s_mov_b32 s60, s34 // (op 2682) reg expression
s_mov_b32 s61, s35 // (op 2682) reg expression
s_mov_b32 s62, s0 // (op 2682) reg expression
s_mov_b32 s63, 131072 // (op 2682) call()

v_and_b32 v19, -4, v1 // (op 2773) 
v_sub_u32 v23, v1, v19 // (op 2773) 

v_lshlrev_b32 v19, 5, v23 // (op 2773) 

v_lshrrev_b32 v23, 2, v1 // (op 2773) 
v_lshl_add_u32 v24, v23, 9, v19 // (op 2773) 


v_lshlrev_b32 v19, 2, v24 // (op 2773) 

v_lshrrev_b32 v23, 3, v19 // (op 2773) 

v_and_b32 v19, -4, v1 // (op 2831) 
v_sub_u32 v24, v1, v19 // (op 2831) 

v_lshlrev_b32 v19, 5, v24 // (op 2831) 

v_lshrrev_b32 v24, 2, v1 // (op 2831) 
v_lshl_add_u32 v25, v24, 7, v19 // (op 2831) 


v_mul_lo_u32 v19, v25, 6 // (op 2831) 

v_lshrrev_b32 v24, 3, v19 // (op 2831) 

v_lshlrev_b32 v19, 2, v1 // (op 2884) 
v_and_b32 v25, -4, v1 // (op 2942) 
v_sub_u32 v26, v1, v25 // (op 2942) 

v_and_b32 v25, 4294967292, v1 // (op 2942) 
v_add_u32 v27, v25, v26 // (op 2942) 


s_ashr_i64 s[0:1], s[36:37], 7 // (op 892) 
s_and_b32 s6, s0, 4294967294 // (op 892) 

s_ashr_i64 s[0:1], s[36:37], 7 // (op 891) 
s_mov_b32 s7, s0 // (op 891) convert

s_cmp_lt_u32 s6, s7 // (op 891) 















s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ForLoopBottom_KLoopTail_891 // (op 891) Condition: Top (jump to RR_GEMM_TN_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ForLoopBottom_KLoopTail_891 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ForLoopTop_KLoopTail_891:

s_add_u32 s6, s6, 1 // (op 893) 
s_lshl_b64 s[0:1], s[12:13], 2 // (op 202) 
v_lshrrev_b64 v[28:29], 3, s[0:1] // (op 202) 


v_mov_b32 v14, v15 // (op 202) 
v_mov_b32 v22, v14 // (op 202) 
buffer_load_dwordx4 v[30:33], v22, s[48:51], 0 offen // (op 202) Load value
v_add_u32 v22, v22, v28 // (op 202) 
buffer_load_dwordx4 v[34:37], v22, s[48:51], 0 offen // (op 202) Load value
v_add_u32 v22, v22, v28 // (op 202) 
buffer_load_dwordx4 v[38:41], v22, s[48:51], 0 offen // (op 202) Load value
v_add_u32 v22, v22, v28 // (op 202) 
buffer_load_dwordx4 v[42:45], v22, s[48:51], 0 offen // (op 202) Load value



v_add_u32 v15, 64, v15 // (op 2472) 


s_barrier  // (op 3163) 
v_mov_b32 v14, v16 // (op 225) 




buffer_load_dwordx4 v[46:49], v14, s[52:55], 0 offen // (op 225) Load value
buffer_load_dwordx2 v[50:51], v14, s[52:55], 0 offen offset:16 // (op 225) Load value

v_add_u32 v16, 96, v16 // (op 2544) 
s_waitcnt vmcnt(5)
ds_write_b128 v23, v[30:33] offset:6400 // (op 203) Store local data 


v_mov_b32 v14, v20 // (op 226) 

s_waitcnt vmcnt(4)
ds_write_b128 v23, v[34:37] offset:6464 // (op 203) Store local data 
s_waitcnt vmcnt(3)
ds_write_b128 v23, v[38:41] offset:6528 // (op 203) Store local data 
s_waitcnt vmcnt(2)
ds_write_b128 v23, v[42:45] offset:6592 // (op 203) Store local data 
buffer_load_dword v22, v14, s[56:59], 0 offen // (op 226) Load value


v_add_u32 v20, 4, v20 // (op 2608) 


s_barrier  // (op 3182) 
v_mov_b32 v14, v21 // (op 227) 




buffer_load_ubyte v17, v14, s[60:63], 0 offen offset: 0 // (op 227) Load value

v_add_u32 v21, 4, v21 // (op 2672) 
s_waitcnt vmcnt(3)
ds_write_b128 v24, v[46:49] // (op 222) Store local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 3172) 
s_waitcnt vmcnt(2)
ds_write_b64 v24, v[50:51] offset:16 // (op 222) Store local data 




ds_read_b128 v[28:31], v7 offset:6400 // (op 205) Load local data 
s_barrier  // (op 3203) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 3193) 






ds_read_b128 v[32:35], v3 // (op 207) Load local data 
ds_read_b64 v[36:37], v3 offset:16 // (op 207) Load local data 
s_waitcnt vmcnt(1)
ds_write_b32 v19, v22 offset:22784 // (op 223) Store local data 

s_barrier  // (op 3223) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 3213) 






ds_read_u8 v14, v6 offset:22784 // (op 209) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v27, v17 offset:6144 // (op 224) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 3233) 



ds_read_u8 v22, v0 offset:6144 // (op 211) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[28:31], v[32:37], a[0:15], v14, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 212) 





ds_read_b128 v[28:31], v7 offset:6432 // (op 214) Load local data 



ds_read_b128 v[38:41], v3 offset:48 // (op 216) Load local data 
ds_read_b64 v[42:43], v3 offset:64 // (op 216) Load local data 



ds_read_u8 v14, v6 offset:22786 // (op 218) Load local data 



ds_read_u8 v17, v0 offset:6146 // (op 220) Load local data 



ds_read_b128 v[44:47], v7 offset:10496 // (op 417) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[28:31], v[38:43], a[0:15], v14, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 221) 





ds_read_u8 v14, v6 offset:23040 // (op 421) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[44:47], v[32:37], a[16:31], v14, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 424) 





ds_read_b128 v[28:31], v7 offset:10528 // (op 426) Load local data 



ds_read_u8 v14, v6 offset:23042 // (op 430) Load local data 



ds_read_b128 v[44:47], v7 offset:14592 // (op 657) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[28:31], v[38:43], a[16:31], v14, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 433) 





ds_read_u8 v14, v6 offset:23296 // (op 661) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[44:47], v[32:37], a[32:47], v14, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 664) 





ds_read_b128 v[28:31], v7 offset:14624 // (op 666) Load local data 



ds_read_u8 v14, v6 offset:23298 // (op 670) Load local data 



ds_read_b128 v[44:47], v7 offset:18688 // (op 897) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[28:31], v[38:43], a[32:47], v14, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 673) 





ds_read_u8 v14, v6 offset:23552 // (op 901) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[44:47], v[32:37], a[48:63], v14, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 904) 







ds_read_b128 v[28:31], v7 offset:18720 // (op 906) Load local data 



ds_read_u8 v14, v6 offset:23554 // (op 910) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[28:31], v[38:43], a[48:63], v14, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 913) 




s_ashr_i64 s[0:1], s[36:37], 7 // (op 891) 
s_mov_b32 s7, s0 // (op 891) convert

s_cmp_lt_u32 s6, s7 // (op 891) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ForLoopTop_KLoopTail_891 // (op 891) Condition: Bottom (jump to RR_GEMM_TN_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ForLoopTop_KLoopTail_891 if true)
RR_GEMM_TN_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ForLoopBottom_KLoopTail_891:























s_barrier  // (op 3354) 









v_cmp_eq_f32 vcc, 0.00000, v12 // (op 157) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ConditionalFalse_0_beta_157 // (op 157) Condition: False, jump to RR_GEMM_TN_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ConditionalFalse_0_beta_157
v_accvgpr_read v14, a0 // (op 50) 
v_accvgpr_read v15, a1 // (op 50) 
v_accvgpr_read v16, a2 // (op 50) 
v_accvgpr_read v17, a3 // (op 50) 
v_accvgpr_read v18, a4 // (op 50) 
v_accvgpr_read v19, a5 // (op 50) 
v_accvgpr_read v20, a6 // (op 50) 
v_accvgpr_read v21, a7 // (op 50) 
v_accvgpr_read v22, a8 // (op 50) 
v_accvgpr_read v23, a9 // (op 50) 
v_accvgpr_read v24, a10 // (op 50) 
v_accvgpr_read v25, a11 // (op 50) 
v_accvgpr_read v26, a12 // (op 50) 
v_accvgpr_read v27, a13 // (op 50) 
v_accvgpr_read v28, a14 // (op 50) 
v_accvgpr_read v29, a15 // (op 50) 
v_mul_f32 v30, v13, v14 // (op 50) 
v_mul_f32 v31, v13, v15 // (op 50) 
v_mul_f32 v32, v13, v16 // (op 50) 
v_mul_f32 v33, v13, v17 // (op 50) 
v_mul_f32 v34, v13, v18 // (op 50) 
v_mul_f32 v35, v13, v19 // (op 50) 
v_mul_f32 v36, v13, v20 // (op 50) 
v_mul_f32 v37, v13, v21 // (op 50) 
v_mul_f32 v38, v13, v22 // (op 50) 
v_mul_f32 v39, v13, v23 // (op 50) 
v_mul_f32 v40, v13, v24 // (op 50) 
v_mul_f32 v41, v13, v25 // (op 50) 
v_mul_f32 v42, v13, v26 // (op 50) 
v_mul_f32 v43, v13, v27 // (op 50) 
v_mul_f32 v44, v13, v28 // (op 50) 
v_mul_f32 v45, v13, v29 // (op 50) 

v_lshrrev_b32 v14, 16, v30 // (op 56) 
v_lshrrev_b32 v15, 16, v31 // (op 56) 
v_lshrrev_b32 v16, 16, v32 // (op 56) 
v_lshrrev_b32 v17, 16, v33 // (op 56) 
v_lshrrev_b32 v18, 16, v34 // (op 56) 
v_lshrrev_b32 v19, 16, v35 // (op 56) 
v_lshrrev_b32 v20, 16, v36 // (op 56) 
v_lshrrev_b32 v21, 16, v37 // (op 56) 
v_lshrrev_b32 v22, 16, v38 // (op 56) 
v_lshrrev_b32 v23, 16, v39 // (op 56) 
v_lshrrev_b32 v24, 16, v40 // (op 56) 
v_lshrrev_b32 v25, 16, v41 // (op 56) 
v_lshrrev_b32 v26, 16, v42 // (op 56) 
v_lshrrev_b32 v27, 16, v43 // (op 56) 
v_lshrrev_b32 v28, 16, v44 // (op 56) 
v_lshrrev_b32 v29, 16, v45 // (op 56) 


v_pack_b32_f16 v30, v14, v15 // (op 58) 
v_pack_b32_f16 v31, v16, v17 // (op 58) 
v_pack_b32_f16 v32, v18, v19 // (op 58) 
v_pack_b32_f16 v33, v20, v21 // (op 58) 
v_pack_b32_f16 v34, v22, v23 // (op 58) 
v_pack_b32_f16 v35, v24, v25 // (op 58) 
v_pack_b32_f16 v36, v26, v27 // (op 58) 
v_pack_b32_f16 v37, v28, v29 // (op 58) 


buffer_store_dwordx2 v[30:31], v11, s[44:47], 0 offen // (op 58) Store value
buffer_store_dwordx2 v[32:33], v11, s[44:47], 0 offen offset:16 // (op 58) Store value
buffer_store_dwordx2 v[34:35], v11, s[44:47], 0 offen offset:32 // (op 58) Store value
buffer_store_dwordx2 v[36:37], v11, s[44:47], 0 offen offset:48 // (op 58) Store value


v_accvgpr_read v14, a16 // (op 1283) 
v_accvgpr_read v15, a17 // (op 1283) 
v_accvgpr_read v16, a18 // (op 1283) 
v_accvgpr_read v17, a19 // (op 1283) 
v_accvgpr_read v18, a20 // (op 1283) 
v_accvgpr_read v19, a21 // (op 1283) 
v_accvgpr_read v20, a22 // (op 1283) 
v_accvgpr_read v21, a23 // (op 1283) 
v_accvgpr_read v22, a24 // (op 1283) 
v_accvgpr_read v23, a25 // (op 1283) 
v_accvgpr_read v24, a26 // (op 1283) 
v_accvgpr_read v25, a27 // (op 1283) 
v_accvgpr_read v26, a28 // (op 1283) 
v_accvgpr_read v27, a29 // (op 1283) 
v_accvgpr_read v28, a30 // (op 1283) 
v_accvgpr_read v29, a31 // (op 1283) 
v_mul_f32 v30, v13, v14 // (op 1283) 
v_mul_f32 v31, v13, v15 // (op 1283) 
v_mul_f32 v32, v13, v16 // (op 1283) 
v_mul_f32 v33, v13, v17 // (op 1283) 
v_mul_f32 v34, v13, v18 // (op 1283) 
v_mul_f32 v35, v13, v19 // (op 1283) 
v_mul_f32 v36, v13, v20 // (op 1283) 
v_mul_f32 v37, v13, v21 // (op 1283) 
v_mul_f32 v38, v13, v22 // (op 1283) 
v_mul_f32 v39, v13, v23 // (op 1283) 
v_mul_f32 v40, v13, v24 // (op 1283) 
v_mul_f32 v41, v13, v25 // (op 1283) 
v_mul_f32 v42, v13, v26 // (op 1283) 
v_mul_f32 v43, v13, v27 // (op 1283) 
v_mul_f32 v44, v13, v28 // (op 1283) 
v_mul_f32 v45, v13, v29 // (op 1283) 

v_lshrrev_b32 v14, 16, v30 // (op 1285) 
v_lshrrev_b32 v15, 16, v31 // (op 1285) 
v_lshrrev_b32 v16, 16, v32 // (op 1285) 
v_lshrrev_b32 v17, 16, v33 // (op 1285) 
v_lshrrev_b32 v18, 16, v34 // (op 1285) 
v_lshrrev_b32 v19, 16, v35 // (op 1285) 
v_lshrrev_b32 v20, 16, v36 // (op 1285) 
v_lshrrev_b32 v21, 16, v37 // (op 1285) 
v_lshrrev_b32 v22, 16, v38 // (op 1285) 
v_lshrrev_b32 v23, 16, v39 // (op 1285) 
v_lshrrev_b32 v24, 16, v40 // (op 1285) 
v_lshrrev_b32 v25, 16, v41 // (op 1285) 
v_lshrrev_b32 v26, 16, v42 // (op 1285) 
v_lshrrev_b32 v27, 16, v43 // (op 1285) 
v_lshrrev_b32 v28, 16, v44 // (op 1285) 
v_lshrrev_b32 v29, 16, v45 // (op 1285) 


v_pack_b32_f16 v30, v14, v15 // (op 1286) 
v_pack_b32_f16 v31, v16, v17 // (op 1286) 
v_pack_b32_f16 v32, v18, v19 // (op 1286) 
v_pack_b32_f16 v33, v20, v21 // (op 1286) 
v_pack_b32_f16 v34, v22, v23 // (op 1286) 
v_pack_b32_f16 v35, v24, v25 // (op 1286) 
v_pack_b32_f16 v36, v26, v27 // (op 1286) 
v_pack_b32_f16 v37, v28, v29 // (op 1286) 


buffer_store_dwordx2 v[30:31], v11, s[44:47], 0 offen offset:128 // (op 1286) Store value
buffer_store_dwordx2 v[32:33], v11, s[44:47], 0 offen offset:144 // (op 1286) Store value
buffer_store_dwordx2 v[34:35], v11, s[44:47], 0 offen offset:160 // (op 1286) Store value
buffer_store_dwordx2 v[36:37], v11, s[44:47], 0 offen offset:176 // (op 1286) Store value


v_accvgpr_read v14, a32 // (op 1303) 
v_accvgpr_read v15, a33 // (op 1303) 
v_accvgpr_read v16, a34 // (op 1303) 
v_accvgpr_read v17, a35 // (op 1303) 
v_accvgpr_read v18, a36 // (op 1303) 
v_accvgpr_read v19, a37 // (op 1303) 
v_accvgpr_read v20, a38 // (op 1303) 
v_accvgpr_read v21, a39 // (op 1303) 
v_accvgpr_read v22, a40 // (op 1303) 
v_accvgpr_read v23, a41 // (op 1303) 
v_accvgpr_read v24, a42 // (op 1303) 
v_accvgpr_read v25, a43 // (op 1303) 
v_accvgpr_read v26, a44 // (op 1303) 
v_accvgpr_read v27, a45 // (op 1303) 
v_accvgpr_read v28, a46 // (op 1303) 
v_accvgpr_read v29, a47 // (op 1303) 
v_mul_f32 v30, v13, v14 // (op 1303) 
v_mul_f32 v31, v13, v15 // (op 1303) 
v_mul_f32 v32, v13, v16 // (op 1303) 
v_mul_f32 v33, v13, v17 // (op 1303) 
v_mul_f32 v34, v13, v18 // (op 1303) 
v_mul_f32 v35, v13, v19 // (op 1303) 
v_mul_f32 v36, v13, v20 // (op 1303) 
v_mul_f32 v37, v13, v21 // (op 1303) 
v_mul_f32 v38, v13, v22 // (op 1303) 
v_mul_f32 v39, v13, v23 // (op 1303) 
v_mul_f32 v40, v13, v24 // (op 1303) 
v_mul_f32 v41, v13, v25 // (op 1303) 
v_mul_f32 v42, v13, v26 // (op 1303) 
v_mul_f32 v43, v13, v27 // (op 1303) 
v_mul_f32 v44, v13, v28 // (op 1303) 
v_mul_f32 v45, v13, v29 // (op 1303) 

v_lshrrev_b32 v14, 16, v30 // (op 1305) 
v_lshrrev_b32 v15, 16, v31 // (op 1305) 
v_lshrrev_b32 v16, 16, v32 // (op 1305) 
v_lshrrev_b32 v17, 16, v33 // (op 1305) 
v_lshrrev_b32 v18, 16, v34 // (op 1305) 
v_lshrrev_b32 v19, 16, v35 // (op 1305) 
v_lshrrev_b32 v20, 16, v36 // (op 1305) 
v_lshrrev_b32 v21, 16, v37 // (op 1305) 
v_lshrrev_b32 v22, 16, v38 // (op 1305) 
v_lshrrev_b32 v23, 16, v39 // (op 1305) 
v_lshrrev_b32 v24, 16, v40 // (op 1305) 
v_lshrrev_b32 v25, 16, v41 // (op 1305) 
v_lshrrev_b32 v26, 16, v42 // (op 1305) 
v_lshrrev_b32 v27, 16, v43 // (op 1305) 
v_lshrrev_b32 v28, 16, v44 // (op 1305) 
v_lshrrev_b32 v29, 16, v45 // (op 1305) 


v_pack_b32_f16 v30, v14, v15 // (op 1306) 
v_pack_b32_f16 v31, v16, v17 // (op 1306) 
v_pack_b32_f16 v32, v18, v19 // (op 1306) 
v_pack_b32_f16 v33, v20, v21 // (op 1306) 
v_pack_b32_f16 v34, v22, v23 // (op 1306) 
v_pack_b32_f16 v35, v24, v25 // (op 1306) 
v_pack_b32_f16 v36, v26, v27 // (op 1306) 
v_pack_b32_f16 v37, v28, v29 // (op 1306) 


buffer_store_dwordx2 v[30:31], v11, s[44:47], 0 offen offset:256 // (op 1306) Store value
buffer_store_dwordx2 v[32:33], v11, s[44:47], 0 offen offset:272 // (op 1306) Store value
buffer_store_dwordx2 v[34:35], v11, s[44:47], 0 offen offset:288 // (op 1306) Store value
buffer_store_dwordx2 v[36:37], v11, s[44:47], 0 offen offset:304 // (op 1306) Store value


v_accvgpr_read v14, a48 // (op 1323) 
v_accvgpr_read v15, a49 // (op 1323) 
v_accvgpr_read v16, a50 // (op 1323) 
v_accvgpr_read v17, a51 // (op 1323) 
v_accvgpr_read v18, a52 // (op 1323) 
v_accvgpr_read v19, a53 // (op 1323) 
v_accvgpr_read v20, a54 // (op 1323) 
v_accvgpr_read v21, a55 // (op 1323) 
v_accvgpr_read v22, a56 // (op 1323) 
v_accvgpr_read v23, a57 // (op 1323) 
v_accvgpr_read v24, a58 // (op 1323) 
v_accvgpr_read v25, a59 // (op 1323) 
v_accvgpr_read v26, a60 // (op 1323) 
v_accvgpr_read v27, a61 // (op 1323) 
v_accvgpr_read v28, a62 // (op 1323) 
v_accvgpr_read v29, a63 // (op 1323) 
v_mul_f32 v30, v13, v14 // (op 1323) 
v_mul_f32 v31, v13, v15 // (op 1323) 
v_mul_f32 v32, v13, v16 // (op 1323) 
v_mul_f32 v33, v13, v17 // (op 1323) 
v_mul_f32 v34, v13, v18 // (op 1323) 
v_mul_f32 v35, v13, v19 // (op 1323) 
v_mul_f32 v36, v13, v20 // (op 1323) 
v_mul_f32 v37, v13, v21 // (op 1323) 
v_mul_f32 v38, v13, v22 // (op 1323) 
v_mul_f32 v39, v13, v23 // (op 1323) 
v_mul_f32 v40, v13, v24 // (op 1323) 
v_mul_f32 v41, v13, v25 // (op 1323) 
v_mul_f32 v42, v13, v26 // (op 1323) 
v_mul_f32 v43, v13, v27 // (op 1323) 
v_mul_f32 v44, v13, v28 // (op 1323) 
v_mul_f32 v45, v13, v29 // (op 1323) 

v_lshrrev_b32 v14, 16, v30 // (op 1325) 
v_lshrrev_b32 v15, 16, v31 // (op 1325) 
v_lshrrev_b32 v16, 16, v32 // (op 1325) 
v_lshrrev_b32 v17, 16, v33 // (op 1325) 
v_lshrrev_b32 v18, 16, v34 // (op 1325) 
v_lshrrev_b32 v19, 16, v35 // (op 1325) 
v_lshrrev_b32 v20, 16, v36 // (op 1325) 
v_lshrrev_b32 v21, 16, v37 // (op 1325) 
v_lshrrev_b32 v22, 16, v38 // (op 1325) 
v_lshrrev_b32 v23, 16, v39 // (op 1325) 
v_lshrrev_b32 v24, 16, v40 // (op 1325) 
v_lshrrev_b32 v25, 16, v41 // (op 1325) 
v_lshrrev_b32 v26, 16, v42 // (op 1325) 
v_lshrrev_b32 v27, 16, v43 // (op 1325) 
v_lshrrev_b32 v28, 16, v44 // (op 1325) 
v_lshrrev_b32 v29, 16, v45 // (op 1325) 


v_pack_b32_f16 v30, v14, v15 // (op 1326) 
v_pack_b32_f16 v31, v16, v17 // (op 1326) 
v_pack_b32_f16 v32, v18, v19 // (op 1326) 
v_pack_b32_f16 v33, v20, v21 // (op 1326) 
v_pack_b32_f16 v34, v22, v23 // (op 1326) 
v_pack_b32_f16 v35, v24, v25 // (op 1326) 
v_pack_b32_f16 v36, v26, v27 // (op 1326) 
v_pack_b32_f16 v37, v28, v29 // (op 1326) 


buffer_store_dwordx2 v[30:31], v11, s[44:47], 0 offen offset:384 // (op 1326) Store value
buffer_store_dwordx2 v[32:33], v11, s[44:47], 0 offen offset:400 // (op 1326) Store value
buffer_store_dwordx2 v[34:35], v11, s[44:47], 0 offen offset:416 // (op 1326) Store value
buffer_store_dwordx2 v[36:37], v11, s[44:47], 0 offen offset:432 // (op 1326) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ConditionalBottom_0_beta_157 // (op 157) Condition: Done, jump to RR_GEMM_TN_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ConditionalBottom_0_beta_157
RR_GEMM_TN_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ConditionalFalse_0_beta_157:



buffer_load_dwordx2 v[14:15], v8, s[40:43], 0 offen offset:384 // (op 1406) Load value
buffer_load_dwordx2 v[16:17], v8, s[40:43], 0 offen offset:400 // (op 1406) Load value
buffer_load_dwordx2 v[18:19], v8, s[40:43], 0 offen offset:416 // (op 1406) Load value
v_accvgpr_read v22, a48 // (op 1402) 
v_accvgpr_read v23, a49 // (op 1402) 
v_accvgpr_read v24, a50 // (op 1402) 
v_accvgpr_read v25, a51 // (op 1402) 
v_accvgpr_read v26, a52 // (op 1402) 
v_accvgpr_read v27, a53 // (op 1402) 
v_accvgpr_read v28, a54 // (op 1402) 
v_accvgpr_read v29, a55 // (op 1402) 
v_accvgpr_read v30, a56 // (op 1402) 
v_accvgpr_read v31, a57 // (op 1402) 
v_accvgpr_read v32, a58 // (op 1402) 
v_accvgpr_read v33, a59 // (op 1402) 
v_accvgpr_read v34, a60 // (op 1402) 
v_accvgpr_read v35, a61 // (op 1402) 
v_accvgpr_read v36, a62 // (op 1402) 
v_accvgpr_read v37, a63 // (op 1402) 
v_mul_f32 v38, v13, v22 // (op 1402) 
v_mul_f32 v39, v13, v23 // (op 1402) 
v_mul_f32 v40, v13, v24 // (op 1402) 
v_mul_f32 v41, v13, v25 // (op 1402) 
v_mul_f32 v42, v13, v26 // (op 1402) 
v_mul_f32 v43, v13, v27 // (op 1402) 
v_mul_f32 v44, v13, v28 // (op 1402) 
v_mul_f32 v45, v13, v29 // (op 1402) 
v_mul_f32 v46, v13, v30 // (op 1402) 
v_mul_f32 v47, v13, v31 // (op 1402) 
v_mul_f32 v48, v13, v32 // (op 1402) 
v_mul_f32 v49, v13, v33 // (op 1402) 
v_mul_f32 v50, v13, v34 // (op 1402) 
v_mul_f32 v51, v13, v35 // (op 1402) 
v_mul_f32 v52, v13, v36 // (op 1402) 
v_mul_f32 v53, v13, v37 // (op 1402) 

buffer_load_dwordx2 v[20:21], v8, s[40:43], 0 offen offset:432 // (op 1406) Load value
s_waitcnt vmcnt(3)
v_and_b32 v6, 65535, v14 // (op 1407) 
v_lshrrev_b32 v7, 16, v14 // (op 1407) 
v_lshlrev_b32 v6, 16, v6 // (op 1407) 
v_lshlrev_b32 v7, 16, v7 // (op 1407) 
v_mul_f32 v22, v12, v6 // (op 1407) 
v_mul_f32 v23, v12, v7 // (op 1407) 
v_and_b32 v6, 65535, v15 // (op 1407) 
v_lshrrev_b32 v7, 16, v15 // (op 1407) 
v_lshlrev_b32 v6, 16, v6 // (op 1407) 
v_lshlrev_b32 v7, 16, v7 // (op 1407) 
v_mul_f32 v24, v12, v6 // (op 1407) 
v_mul_f32 v25, v12, v7 // (op 1407) 
s_waitcnt vmcnt(2)
v_and_b32 v6, 65535, v16 // (op 1407) 
v_lshrrev_b32 v7, 16, v16 // (op 1407) 
v_lshlrev_b32 v6, 16, v6 // (op 1407) 
v_lshlrev_b32 v7, 16, v7 // (op 1407) 
v_mul_f32 v26, v12, v6 // (op 1407) 
v_mul_f32 v27, v12, v7 // (op 1407) 
v_and_b32 v6, 65535, v17 // (op 1407) 
v_lshrrev_b32 v7, 16, v17 // (op 1407) 
v_lshlrev_b32 v6, 16, v6 // (op 1407) 
v_lshlrev_b32 v7, 16, v7 // (op 1407) 
v_mul_f32 v28, v12, v6 // (op 1407) 
v_mul_f32 v29, v12, v7 // (op 1407) 
s_waitcnt vmcnt(1)
v_and_b32 v6, 65535, v18 // (op 1407) 
v_lshrrev_b32 v7, 16, v18 // (op 1407) 
v_lshlrev_b32 v6, 16, v6 // (op 1407) 
v_lshlrev_b32 v7, 16, v7 // (op 1407) 
v_mul_f32 v30, v12, v6 // (op 1407) 
v_mul_f32 v31, v12, v7 // (op 1407) 
v_and_b32 v6, 65535, v19 // (op 1407) 
v_lshrrev_b32 v7, 16, v19 // (op 1407) 
v_lshlrev_b32 v6, 16, v6 // (op 1407) 
v_lshlrev_b32 v7, 16, v7 // (op 1407) 
v_mul_f32 v32, v12, v6 // (op 1407) 
v_mul_f32 v33, v12, v7 // (op 1407) 
s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v20 // (op 1407) 
v_lshrrev_b32 v7, 16, v20 // (op 1407) 
v_lshlrev_b32 v6, 16, v6 // (op 1407) 
v_lshlrev_b32 v7, 16, v7 // (op 1407) 
v_mul_f32 v34, v12, v6 // (op 1407) 
v_mul_f32 v35, v12, v7 // (op 1407) 
v_and_b32 v6, 65535, v21 // (op 1407) 
v_lshrrev_b32 v7, 16, v21 // (op 1407) 
v_lshlrev_b32 v6, 16, v6 // (op 1407) 
v_lshlrev_b32 v7, 16, v7 // (op 1407) 
v_mul_f32 v36, v12, v6 // (op 1407) 
v_mul_f32 v37, v12, v7 // (op 1407) 


v_add_f32 v54, v22, v38 // (op 1403) 
v_add_f32 v55, v23, v39 // (op 1403) 
v_add_f32 v56, v24, v40 // (op 1403) 
v_add_f32 v57, v25, v41 // (op 1403) 
v_add_f32 v58, v26, v42 // (op 1403) 
v_add_f32 v59, v27, v43 // (op 1403) 
v_add_f32 v60, v28, v44 // (op 1403) 
v_add_f32 v61, v29, v45 // (op 1403) 
v_add_f32 v62, v30, v46 // (op 1403) 
v_add_f32 v63, v31, v47 // (op 1403) 
v_add_f32 v64, v32, v48 // (op 1403) 
v_add_f32 v65, v33, v49 // (op 1403) 
v_add_f32 v66, v34, v50 // (op 1403) 
v_add_f32 v67, v35, v51 // (op 1403) 
v_add_f32 v68, v36, v52 // (op 1403) 
v_add_f32 v69, v37, v53 // (op 1403) 


v_lshrrev_b32 v14, 16, v54 // (op 1404) 
v_lshrrev_b32 v15, 16, v55 // (op 1404) 
v_lshrrev_b32 v16, 16, v56 // (op 1404) 
v_lshrrev_b32 v17, 16, v57 // (op 1404) 
v_lshrrev_b32 v18, 16, v58 // (op 1404) 
v_lshrrev_b32 v19, 16, v59 // (op 1404) 
v_lshrrev_b32 v20, 16, v60 // (op 1404) 
v_lshrrev_b32 v21, 16, v61 // (op 1404) 
v_lshrrev_b32 v22, 16, v62 // (op 1404) 
v_lshrrev_b32 v23, 16, v63 // (op 1404) 
v_lshrrev_b32 v24, 16, v64 // (op 1404) 
v_lshrrev_b32 v25, 16, v65 // (op 1404) 
v_lshrrev_b32 v26, 16, v66 // (op 1404) 
v_lshrrev_b32 v27, 16, v67 // (op 1404) 
v_lshrrev_b32 v28, 16, v68 // (op 1404) 
v_lshrrev_b32 v29, 16, v69 // (op 1404) 


v_pack_b32_f16 v30, v14, v15 // (op 1405) 
v_pack_b32_f16 v31, v16, v17 // (op 1405) 
v_pack_b32_f16 v32, v18, v19 // (op 1405) 
v_pack_b32_f16 v33, v20, v21 // (op 1405) 
v_pack_b32_f16 v34, v22, v23 // (op 1405) 
v_pack_b32_f16 v35, v24, v25 // (op 1405) 
v_pack_b32_f16 v36, v26, v27 // (op 1405) 
v_pack_b32_f16 v37, v28, v29 // (op 1405) 


buffer_store_dwordx2 v[30:31], v11, s[44:47], 0 offen offset:384 // (op 1405) Store value
buffer_store_dwordx2 v[32:33], v11, s[44:47], 0 offen offset:400 // (op 1405) Store value
buffer_store_dwordx2 v[34:35], v11, s[44:47], 0 offen offset:416 // (op 1405) Store value
buffer_store_dwordx2 v[36:37], v11, s[44:47], 0 offen offset:432 // (op 1405) Store value





buffer_load_dwordx2 v[14:15], v8, s[40:43], 0 offen offset:256 // (op 1386) Load value
buffer_load_dwordx2 v[16:17], v8, s[40:43], 0 offen offset:272 // (op 1386) Load value
v_accvgpr_read v22, a32 // (op 1382) 
v_accvgpr_read v23, a33 // (op 1382) 
v_accvgpr_read v24, a34 // (op 1382) 
v_accvgpr_read v25, a35 // (op 1382) 
v_accvgpr_read v26, a36 // (op 1382) 
v_accvgpr_read v27, a37 // (op 1382) 
v_accvgpr_read v28, a38 // (op 1382) 
v_accvgpr_read v29, a39 // (op 1382) 
v_accvgpr_read v30, a40 // (op 1382) 
v_accvgpr_read v31, a41 // (op 1382) 
v_accvgpr_read v32, a42 // (op 1382) 
v_accvgpr_read v33, a43 // (op 1382) 
v_accvgpr_read v34, a44 // (op 1382) 
v_accvgpr_read v35, a45 // (op 1382) 
v_accvgpr_read v36, a46 // (op 1382) 
v_accvgpr_read v37, a47 // (op 1382) 
v_mul_f32 v38, v13, v22 // (op 1382) 
v_mul_f32 v39, v13, v23 // (op 1382) 
v_mul_f32 v40, v13, v24 // (op 1382) 
v_mul_f32 v41, v13, v25 // (op 1382) 
v_mul_f32 v42, v13, v26 // (op 1382) 
v_mul_f32 v43, v13, v27 // (op 1382) 
v_mul_f32 v44, v13, v28 // (op 1382) 
v_mul_f32 v45, v13, v29 // (op 1382) 
v_mul_f32 v46, v13, v30 // (op 1382) 
v_mul_f32 v47, v13, v31 // (op 1382) 
v_mul_f32 v48, v13, v32 // (op 1382) 
v_mul_f32 v49, v13, v33 // (op 1382) 
v_mul_f32 v50, v13, v34 // (op 1382) 
v_mul_f32 v51, v13, v35 // (op 1382) 
v_mul_f32 v52, v13, v36 // (op 1382) 
v_mul_f32 v53, v13, v37 // (op 1382) 

buffer_load_dwordx2 v[18:19], v8, s[40:43], 0 offen offset:288 // (op 1386) Load value
buffer_load_dwordx2 v[20:21], v8, s[40:43], 0 offen offset:304 // (op 1386) Load value
s_waitcnt vmcnt(3)
v_and_b32 v6, 65535, v14 // (op 1387) 
v_lshrrev_b32 v7, 16, v14 // (op 1387) 
v_lshlrev_b32 v6, 16, v6 // (op 1387) 
v_lshlrev_b32 v7, 16, v7 // (op 1387) 
v_mul_f32 v22, v12, v6 // (op 1387) 
v_mul_f32 v23, v12, v7 // (op 1387) 
v_and_b32 v6, 65535, v15 // (op 1387) 
v_lshrrev_b32 v7, 16, v15 // (op 1387) 
v_lshlrev_b32 v6, 16, v6 // (op 1387) 
v_lshlrev_b32 v7, 16, v7 // (op 1387) 
v_mul_f32 v24, v12, v6 // (op 1387) 
v_mul_f32 v25, v12, v7 // (op 1387) 
s_waitcnt vmcnt(2)
v_and_b32 v6, 65535, v16 // (op 1387) 
v_lshrrev_b32 v7, 16, v16 // (op 1387) 
v_lshlrev_b32 v6, 16, v6 // (op 1387) 
v_lshlrev_b32 v7, 16, v7 // (op 1387) 
v_mul_f32 v26, v12, v6 // (op 1387) 
v_mul_f32 v27, v12, v7 // (op 1387) 
v_and_b32 v6, 65535, v17 // (op 1387) 
v_lshrrev_b32 v7, 16, v17 // (op 1387) 
v_lshlrev_b32 v6, 16, v6 // (op 1387) 
v_lshlrev_b32 v7, 16, v7 // (op 1387) 
v_mul_f32 v28, v12, v6 // (op 1387) 
v_mul_f32 v29, v12, v7 // (op 1387) 
s_waitcnt vmcnt(1)
v_and_b32 v6, 65535, v18 // (op 1387) 
v_lshrrev_b32 v7, 16, v18 // (op 1387) 
v_lshlrev_b32 v6, 16, v6 // (op 1387) 
v_lshlrev_b32 v7, 16, v7 // (op 1387) 
v_mul_f32 v30, v12, v6 // (op 1387) 
v_mul_f32 v31, v12, v7 // (op 1387) 
v_and_b32 v6, 65535, v19 // (op 1387) 
v_lshrrev_b32 v7, 16, v19 // (op 1387) 
v_lshlrev_b32 v6, 16, v6 // (op 1387) 
v_lshlrev_b32 v7, 16, v7 // (op 1387) 
v_mul_f32 v32, v12, v6 // (op 1387) 
v_mul_f32 v33, v12, v7 // (op 1387) 
s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v20 // (op 1387) 
v_lshrrev_b32 v7, 16, v20 // (op 1387) 
v_lshlrev_b32 v6, 16, v6 // (op 1387) 
v_lshlrev_b32 v7, 16, v7 // (op 1387) 
v_mul_f32 v34, v12, v6 // (op 1387) 
v_mul_f32 v35, v12, v7 // (op 1387) 
v_and_b32 v6, 65535, v21 // (op 1387) 
v_lshrrev_b32 v7, 16, v21 // (op 1387) 
v_lshlrev_b32 v6, 16, v6 // (op 1387) 
v_lshlrev_b32 v7, 16, v7 // (op 1387) 
v_mul_f32 v36, v12, v6 // (op 1387) 
v_mul_f32 v37, v12, v7 // (op 1387) 


v_add_f32 v54, v22, v38 // (op 1383) 
v_add_f32 v55, v23, v39 // (op 1383) 
v_add_f32 v56, v24, v40 // (op 1383) 
v_add_f32 v57, v25, v41 // (op 1383) 
v_add_f32 v58, v26, v42 // (op 1383) 
v_add_f32 v59, v27, v43 // (op 1383) 
v_add_f32 v60, v28, v44 // (op 1383) 
v_add_f32 v61, v29, v45 // (op 1383) 
v_add_f32 v62, v30, v46 // (op 1383) 
v_add_f32 v63, v31, v47 // (op 1383) 
v_add_f32 v64, v32, v48 // (op 1383) 
v_add_f32 v65, v33, v49 // (op 1383) 
v_add_f32 v66, v34, v50 // (op 1383) 
v_add_f32 v67, v35, v51 // (op 1383) 
v_add_f32 v68, v36, v52 // (op 1383) 
v_add_f32 v69, v37, v53 // (op 1383) 


v_lshrrev_b32 v14, 16, v54 // (op 1384) 
v_lshrrev_b32 v15, 16, v55 // (op 1384) 
v_lshrrev_b32 v16, 16, v56 // (op 1384) 
v_lshrrev_b32 v17, 16, v57 // (op 1384) 
v_lshrrev_b32 v18, 16, v58 // (op 1384) 
v_lshrrev_b32 v19, 16, v59 // (op 1384) 
v_lshrrev_b32 v20, 16, v60 // (op 1384) 
v_lshrrev_b32 v21, 16, v61 // (op 1384) 
v_lshrrev_b32 v22, 16, v62 // (op 1384) 
v_lshrrev_b32 v23, 16, v63 // (op 1384) 
v_lshrrev_b32 v24, 16, v64 // (op 1384) 
v_lshrrev_b32 v25, 16, v65 // (op 1384) 
v_lshrrev_b32 v26, 16, v66 // (op 1384) 
v_lshrrev_b32 v27, 16, v67 // (op 1384) 
v_lshrrev_b32 v28, 16, v68 // (op 1384) 
v_lshrrev_b32 v29, 16, v69 // (op 1384) 


v_pack_b32_f16 v30, v14, v15 // (op 1385) 
v_pack_b32_f16 v31, v16, v17 // (op 1385) 
v_pack_b32_f16 v32, v18, v19 // (op 1385) 
v_pack_b32_f16 v33, v20, v21 // (op 1385) 
v_pack_b32_f16 v34, v22, v23 // (op 1385) 
v_pack_b32_f16 v35, v24, v25 // (op 1385) 
v_pack_b32_f16 v36, v26, v27 // (op 1385) 
v_pack_b32_f16 v37, v28, v29 // (op 1385) 


buffer_store_dwordx2 v[30:31], v11, s[44:47], 0 offen offset:256 // (op 1385) Store value
buffer_store_dwordx2 v[32:33], v11, s[44:47], 0 offen offset:272 // (op 1385) Store value
buffer_store_dwordx2 v[34:35], v11, s[44:47], 0 offen offset:288 // (op 1385) Store value
buffer_store_dwordx2 v[36:37], v11, s[44:47], 0 offen offset:304 // (op 1385) Store value





buffer_load_dwordx2 v[14:15], v8, s[40:43], 0 offen offset:128 // (op 1366) Load value
buffer_load_dwordx2 v[16:17], v8, s[40:43], 0 offen offset:144 // (op 1366) Load value
v_accvgpr_read v22, a16 // (op 1362) 
v_accvgpr_read v23, a17 // (op 1362) 
v_accvgpr_read v24, a18 // (op 1362) 
v_accvgpr_read v25, a19 // (op 1362) 
v_accvgpr_read v26, a20 // (op 1362) 
v_accvgpr_read v27, a21 // (op 1362) 
v_accvgpr_read v28, a22 // (op 1362) 
v_accvgpr_read v29, a23 // (op 1362) 
v_accvgpr_read v30, a24 // (op 1362) 
v_accvgpr_read v31, a25 // (op 1362) 
v_accvgpr_read v32, a26 // (op 1362) 
v_accvgpr_read v33, a27 // (op 1362) 
v_accvgpr_read v34, a28 // (op 1362) 
v_accvgpr_read v35, a29 // (op 1362) 
v_accvgpr_read v36, a30 // (op 1362) 
v_accvgpr_read v37, a31 // (op 1362) 
v_mul_f32 v38, v13, v22 // (op 1362) 
v_mul_f32 v39, v13, v23 // (op 1362) 
v_mul_f32 v40, v13, v24 // (op 1362) 
v_mul_f32 v41, v13, v25 // (op 1362) 
v_mul_f32 v42, v13, v26 // (op 1362) 
v_mul_f32 v43, v13, v27 // (op 1362) 
v_mul_f32 v44, v13, v28 // (op 1362) 
v_mul_f32 v45, v13, v29 // (op 1362) 
v_mul_f32 v46, v13, v30 // (op 1362) 
v_mul_f32 v47, v13, v31 // (op 1362) 
v_mul_f32 v48, v13, v32 // (op 1362) 
v_mul_f32 v49, v13, v33 // (op 1362) 
v_mul_f32 v50, v13, v34 // (op 1362) 
v_mul_f32 v51, v13, v35 // (op 1362) 
v_mul_f32 v52, v13, v36 // (op 1362) 
v_mul_f32 v53, v13, v37 // (op 1362) 

buffer_load_dwordx2 v[18:19], v8, s[40:43], 0 offen offset:160 // (op 1366) Load value
buffer_load_dwordx2 v[20:21], v8, s[40:43], 0 offen offset:176 // (op 1366) Load value
s_waitcnt vmcnt(3)
v_and_b32 v6, 65535, v14 // (op 1367) 
v_lshrrev_b32 v7, 16, v14 // (op 1367) 
v_lshlrev_b32 v6, 16, v6 // (op 1367) 
v_lshlrev_b32 v7, 16, v7 // (op 1367) 
v_mul_f32 v22, v12, v6 // (op 1367) 
v_mul_f32 v23, v12, v7 // (op 1367) 
v_and_b32 v6, 65535, v15 // (op 1367) 
v_lshrrev_b32 v7, 16, v15 // (op 1367) 
v_lshlrev_b32 v6, 16, v6 // (op 1367) 
v_lshlrev_b32 v7, 16, v7 // (op 1367) 
v_mul_f32 v24, v12, v6 // (op 1367) 
v_mul_f32 v25, v12, v7 // (op 1367) 
s_waitcnt vmcnt(2)
v_and_b32 v6, 65535, v16 // (op 1367) 
v_lshrrev_b32 v7, 16, v16 // (op 1367) 
v_lshlrev_b32 v6, 16, v6 // (op 1367) 
v_lshlrev_b32 v7, 16, v7 // (op 1367) 
v_mul_f32 v26, v12, v6 // (op 1367) 
v_mul_f32 v27, v12, v7 // (op 1367) 
v_and_b32 v6, 65535, v17 // (op 1367) 
v_lshrrev_b32 v7, 16, v17 // (op 1367) 
v_lshlrev_b32 v6, 16, v6 // (op 1367) 
v_lshlrev_b32 v7, 16, v7 // (op 1367) 
v_mul_f32 v28, v12, v6 // (op 1367) 
v_mul_f32 v29, v12, v7 // (op 1367) 
s_waitcnt vmcnt(1)
v_and_b32 v6, 65535, v18 // (op 1367) 
v_lshrrev_b32 v7, 16, v18 // (op 1367) 
v_lshlrev_b32 v6, 16, v6 // (op 1367) 
v_lshlrev_b32 v7, 16, v7 // (op 1367) 
v_mul_f32 v30, v12, v6 // (op 1367) 
v_mul_f32 v31, v12, v7 // (op 1367) 
v_and_b32 v6, 65535, v19 // (op 1367) 
v_lshrrev_b32 v7, 16, v19 // (op 1367) 
v_lshlrev_b32 v6, 16, v6 // (op 1367) 
v_lshlrev_b32 v7, 16, v7 // (op 1367) 
v_mul_f32 v32, v12, v6 // (op 1367) 
v_mul_f32 v33, v12, v7 // (op 1367) 
s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v20 // (op 1367) 
v_lshrrev_b32 v7, 16, v20 // (op 1367) 
v_lshlrev_b32 v6, 16, v6 // (op 1367) 
v_lshlrev_b32 v7, 16, v7 // (op 1367) 
v_mul_f32 v34, v12, v6 // (op 1367) 
v_mul_f32 v35, v12, v7 // (op 1367) 
v_and_b32 v6, 65535, v21 // (op 1367) 
v_lshrrev_b32 v7, 16, v21 // (op 1367) 
v_lshlrev_b32 v6, 16, v6 // (op 1367) 
v_lshlrev_b32 v7, 16, v7 // (op 1367) 
v_mul_f32 v36, v12, v6 // (op 1367) 
v_mul_f32 v37, v12, v7 // (op 1367) 


v_add_f32 v54, v22, v38 // (op 1363) 
v_add_f32 v55, v23, v39 // (op 1363) 
v_add_f32 v56, v24, v40 // (op 1363) 
v_add_f32 v57, v25, v41 // (op 1363) 
v_add_f32 v58, v26, v42 // (op 1363) 
v_add_f32 v59, v27, v43 // (op 1363) 
v_add_f32 v60, v28, v44 // (op 1363) 
v_add_f32 v61, v29, v45 // (op 1363) 
v_add_f32 v62, v30, v46 // (op 1363) 
v_add_f32 v63, v31, v47 // (op 1363) 
v_add_f32 v64, v32, v48 // (op 1363) 
v_add_f32 v65, v33, v49 // (op 1363) 
v_add_f32 v66, v34, v50 // (op 1363) 
v_add_f32 v67, v35, v51 // (op 1363) 
v_add_f32 v68, v36, v52 // (op 1363) 
v_add_f32 v69, v37, v53 // (op 1363) 


v_lshrrev_b32 v14, 16, v54 // (op 1364) 
v_lshrrev_b32 v15, 16, v55 // (op 1364) 
v_lshrrev_b32 v16, 16, v56 // (op 1364) 
v_lshrrev_b32 v17, 16, v57 // (op 1364) 
v_lshrrev_b32 v18, 16, v58 // (op 1364) 
v_lshrrev_b32 v19, 16, v59 // (op 1364) 
v_lshrrev_b32 v20, 16, v60 // (op 1364) 
v_lshrrev_b32 v21, 16, v61 // (op 1364) 
v_lshrrev_b32 v22, 16, v62 // (op 1364) 
v_lshrrev_b32 v23, 16, v63 // (op 1364) 
v_lshrrev_b32 v24, 16, v64 // (op 1364) 
v_lshrrev_b32 v25, 16, v65 // (op 1364) 
v_lshrrev_b32 v26, 16, v66 // (op 1364) 
v_lshrrev_b32 v27, 16, v67 // (op 1364) 
v_lshrrev_b32 v28, 16, v68 // (op 1364) 
v_lshrrev_b32 v29, 16, v69 // (op 1364) 


v_pack_b32_f16 v30, v14, v15 // (op 1365) 
v_pack_b32_f16 v31, v16, v17 // (op 1365) 
v_pack_b32_f16 v32, v18, v19 // (op 1365) 
v_pack_b32_f16 v33, v20, v21 // (op 1365) 
v_pack_b32_f16 v34, v22, v23 // (op 1365) 
v_pack_b32_f16 v35, v24, v25 // (op 1365) 
v_pack_b32_f16 v36, v26, v27 // (op 1365) 
v_pack_b32_f16 v37, v28, v29 // (op 1365) 


buffer_store_dwordx2 v[30:31], v11, s[44:47], 0 offen offset:128 // (op 1365) Store value
buffer_store_dwordx2 v[32:33], v11, s[44:47], 0 offen offset:144 // (op 1365) Store value
buffer_store_dwordx2 v[34:35], v11, s[44:47], 0 offen offset:160 // (op 1365) Store value
buffer_store_dwordx2 v[36:37], v11, s[44:47], 0 offen offset:176 // (op 1365) Store value





buffer_load_dwordx2 v[14:15], v8, s[40:43], 0 offen // (op 170) Load value
buffer_load_dwordx2 v[16:17], v8, s[40:43], 0 offen offset:16 // (op 170) Load value
v_accvgpr_read v22, a0 // (op 166) 
v_accvgpr_read v23, a1 // (op 166) 
v_accvgpr_read v24, a2 // (op 166) 
v_accvgpr_read v25, a3 // (op 166) 
v_accvgpr_read v26, a4 // (op 166) 
v_accvgpr_read v27, a5 // (op 166) 
v_accvgpr_read v28, a6 // (op 166) 
v_accvgpr_read v29, a7 // (op 166) 
v_accvgpr_read v30, a8 // (op 166) 
v_accvgpr_read v31, a9 // (op 166) 
v_accvgpr_read v32, a10 // (op 166) 
v_accvgpr_read v33, a11 // (op 166) 
v_accvgpr_read v34, a12 // (op 166) 
v_accvgpr_read v35, a13 // (op 166) 
v_accvgpr_read v36, a14 // (op 166) 
v_accvgpr_read v37, a15 // (op 166) 
v_mul_f32 v38, v13, v22 // (op 166) 
v_mul_f32 v39, v13, v23 // (op 166) 
v_mul_f32 v40, v13, v24 // (op 166) 
v_mul_f32 v41, v13, v25 // (op 166) 
v_mul_f32 v42, v13, v26 // (op 166) 
v_mul_f32 v43, v13, v27 // (op 166) 
v_mul_f32 v44, v13, v28 // (op 166) 
v_mul_f32 v45, v13, v29 // (op 166) 
v_mul_f32 v46, v13, v30 // (op 166) 
v_mul_f32 v47, v13, v31 // (op 166) 
v_mul_f32 v48, v13, v32 // (op 166) 
v_mul_f32 v49, v13, v33 // (op 166) 
v_mul_f32 v50, v13, v34 // (op 166) 
v_mul_f32 v51, v13, v35 // (op 166) 
v_mul_f32 v52, v13, v36 // (op 166) 
v_mul_f32 v53, v13, v37 // (op 166) 

buffer_load_dwordx2 v[18:19], v8, s[40:43], 0 offen offset:32 // (op 170) Load value
buffer_load_dwordx2 v[20:21], v8, s[40:43], 0 offen offset:48 // (op 170) Load value
s_waitcnt vmcnt(3)
v_and_b32 v6, 65535, v14 // (op 171) 
v_lshrrev_b32 v7, 16, v14 // (op 171) 
v_lshlrev_b32 v6, 16, v6 // (op 171) 
v_lshlrev_b32 v7, 16, v7 // (op 171) 
v_mul_f32 v22, v12, v6 // (op 171) 
v_mul_f32 v23, v12, v7 // (op 171) 
v_and_b32 v6, 65535, v15 // (op 171) 
v_lshrrev_b32 v7, 16, v15 // (op 171) 
v_lshlrev_b32 v6, 16, v6 // (op 171) 
v_lshlrev_b32 v7, 16, v7 // (op 171) 
v_mul_f32 v24, v12, v6 // (op 171) 
v_mul_f32 v25, v12, v7 // (op 171) 
s_waitcnt vmcnt(2)
v_and_b32 v6, 65535, v16 // (op 171) 
v_lshrrev_b32 v7, 16, v16 // (op 171) 
v_lshlrev_b32 v6, 16, v6 // (op 171) 
v_lshlrev_b32 v7, 16, v7 // (op 171) 
v_mul_f32 v26, v12, v6 // (op 171) 
v_mul_f32 v27, v12, v7 // (op 171) 
v_and_b32 v6, 65535, v17 // (op 171) 
v_lshrrev_b32 v7, 16, v17 // (op 171) 
v_lshlrev_b32 v6, 16, v6 // (op 171) 
v_lshlrev_b32 v7, 16, v7 // (op 171) 
v_mul_f32 v28, v12, v6 // (op 171) 
v_mul_f32 v29, v12, v7 // (op 171) 
s_waitcnt vmcnt(1)
v_and_b32 v6, 65535, v18 // (op 171) 
v_lshrrev_b32 v7, 16, v18 // (op 171) 
v_lshlrev_b32 v6, 16, v6 // (op 171) 
v_lshlrev_b32 v7, 16, v7 // (op 171) 
v_mul_f32 v30, v12, v6 // (op 171) 
v_mul_f32 v31, v12, v7 // (op 171) 
v_and_b32 v6, 65535, v19 // (op 171) 
v_lshrrev_b32 v7, 16, v19 // (op 171) 
v_lshlrev_b32 v6, 16, v6 // (op 171) 
v_lshlrev_b32 v7, 16, v7 // (op 171) 
v_mul_f32 v32, v12, v6 // (op 171) 
v_mul_f32 v33, v12, v7 // (op 171) 
s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v20 // (op 171) 
v_lshrrev_b32 v7, 16, v20 // (op 171) 
v_lshlrev_b32 v6, 16, v6 // (op 171) 
v_lshlrev_b32 v7, 16, v7 // (op 171) 
v_mul_f32 v34, v12, v6 // (op 171) 
v_mul_f32 v35, v12, v7 // (op 171) 
v_and_b32 v6, 65535, v21 // (op 171) 
v_lshrrev_b32 v7, 16, v21 // (op 171) 
v_lshlrev_b32 v6, 16, v6 // (op 171) 
v_lshlrev_b32 v7, 16, v7 // (op 171) 
v_mul_f32 v36, v12, v6 // (op 171) 
v_mul_f32 v37, v12, v7 // (op 171) 


v_add_f32 v54, v22, v38 // (op 167) 
v_add_f32 v55, v23, v39 // (op 167) 
v_add_f32 v56, v24, v40 // (op 167) 
v_add_f32 v57, v25, v41 // (op 167) 
v_add_f32 v58, v26, v42 // (op 167) 
v_add_f32 v59, v27, v43 // (op 167) 
v_add_f32 v60, v28, v44 // (op 167) 
v_add_f32 v61, v29, v45 // (op 167) 
v_add_f32 v62, v30, v46 // (op 167) 
v_add_f32 v63, v31, v47 // (op 167) 
v_add_f32 v64, v32, v48 // (op 167) 
v_add_f32 v65, v33, v49 // (op 167) 
v_add_f32 v66, v34, v50 // (op 167) 
v_add_f32 v67, v35, v51 // (op 167) 
v_add_f32 v68, v36, v52 // (op 167) 
v_add_f32 v69, v37, v53 // (op 167) 


v_lshrrev_b32 v14, 16, v54 // (op 168) 
v_lshrrev_b32 v15, 16, v55 // (op 168) 
v_lshrrev_b32 v16, 16, v56 // (op 168) 
v_lshrrev_b32 v17, 16, v57 // (op 168) 
v_lshrrev_b32 v18, 16, v58 // (op 168) 
v_lshrrev_b32 v19, 16, v59 // (op 168) 
v_lshrrev_b32 v20, 16, v60 // (op 168) 
v_lshrrev_b32 v21, 16, v61 // (op 168) 
v_lshrrev_b32 v22, 16, v62 // (op 168) 
v_lshrrev_b32 v23, 16, v63 // (op 168) 
v_lshrrev_b32 v24, 16, v64 // (op 168) 
v_lshrrev_b32 v25, 16, v65 // (op 168) 
v_lshrrev_b32 v26, 16, v66 // (op 168) 
v_lshrrev_b32 v27, 16, v67 // (op 168) 
v_lshrrev_b32 v28, 16, v68 // (op 168) 
v_lshrrev_b32 v29, 16, v69 // (op 168) 


v_pack_b32_f16 v30, v14, v15 // (op 169) 
v_pack_b32_f16 v31, v16, v17 // (op 169) 
v_pack_b32_f16 v32, v18, v19 // (op 169) 
v_pack_b32_f16 v33, v20, v21 // (op 169) 
v_pack_b32_f16 v34, v22, v23 // (op 169) 
v_pack_b32_f16 v35, v24, v25 // (op 169) 
v_pack_b32_f16 v36, v26, v27 // (op 169) 
v_pack_b32_f16 v37, v28, v29 // (op 169) 


buffer_store_dwordx2 v[30:31], v11, s[44:47], 0 offen // (op 169) Store value
buffer_store_dwordx2 v[32:33], v11, s[44:47], 0 offen offset:16 // (op 169) Store value
buffer_store_dwordx2 v[34:35], v11, s[44:47], 0 offen offset:32 // (op 169) Store value
buffer_store_dwordx2 v[36:37], v11, s[44:47], 0 offen offset:48 // (op 169) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ConditionalBottom_0_beta_157:












s_endpgm  // End of RR_GEMM_TN_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2
.LRR_GEMM_TN_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_end:
.size RR_GEMM_TN_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2, .LRR_GEMM_TN_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_end-RR_GEMM_TN_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2
  .amdhsa_next_free_vgpr 188
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 47616 // lds bytes
  .amdhsa_accum_offset 124
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
    macrotile: [256, 64, 128]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [32, 32, 64, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [4, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2
    .symbol: RR_GEMM_TN_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 47616
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 64
    .vgpr_count: 122
    .agpr_count: 64
    .max_flat_workgroup_size: 256
    .wavefront_size: 64
    .args:
      - .name: Tensor_10_stride_1_10
        .size: 8
        .offset: 0
        .value_kind: by_value
      - .name: Tensor_21_stride_1_16
        .size: 8
        .offset: 8
        .value_kind: by_value
      - .name: Tensor_10_pointer
        .size: 8
        .offset: 16
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_2_size_1_4
        .size: 8
        .offset: 24
        .value_kind: by_value
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 32
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_stride_0_0
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
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 56
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_0_1
        .size: 8
        .offset: 64
        .value_kind: by_value
      - .name: Tensor_2_stride_1_3
        .size: 8
        .offset: 72
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 80
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_stride_0_6
        .size: 8
        .offset: 88
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 96
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_7_stride_1_8
        .size: 8
        .offset: 104
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 112
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_1_14
        .size: 8
        .offset: 120
        .value_kind: by_value
...

.end_amdgpu_metadata

