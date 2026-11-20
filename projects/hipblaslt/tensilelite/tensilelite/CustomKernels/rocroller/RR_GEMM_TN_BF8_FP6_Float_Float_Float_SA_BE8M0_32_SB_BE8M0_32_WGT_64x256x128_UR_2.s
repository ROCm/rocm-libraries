// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2
.p2align 8
.type RR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2,@function
RR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_exec_begin:


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


v_and_b32 v0, 63, v1 // (op 2727) 
v_lshrrev_b32 v3, 4, v0 // (op 2727) 
v_and_b32 v4, 1, v3 // (op 2727) 

v_and_b32 v3, 15, v1 // (op 2727) 
v_lshl_add_u32 v5, v4, 4, v3 // (op 2727) 


v_lshrrev_b32 v3, 7, v1 // (op 2727) 
v_lshl_add_u32 v4, v3, 5, v5 // (op 2727) 


v_lshrrev_b32 v3, 5, v0 // (op 2727) 

v_lshlrev_b32 v0, 4, v3 // (op 2727) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 2727) 


v_lshrrev_b32 v0, 7, v1 // (op 2838) 
v_and_b32 v4, 31, v1 // (op 2838) 
v_lshl_add_u32 v5, v0, 5, v4 // (op 2838) 


v_and_b32 v0, 63, v1 // (op 2838) 
v_lshrrev_b32 v4, 5, v0 // (op 2838) 

v_lshl_add_u32 v0, v5, 2, v4 // (op 2838) 


v_and_b32 v4, 63, v1 // (op 2953) 
v_lshrrev_b32 v5, 4, v4 // (op 2953) 
v_and_b32 v6, 1, v5 // (op 2953) 

v_and_b32 v5, 15, v1 // (op 2953) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 2953) 


v_lshrrev_b32 v5, 7, v1 // (op 2953) 
v_lshl_add_u32 v6, v5, 5, v7 // (op 2953) 


v_lshrrev_b32 v5, 5, v4 // (op 2953) 

v_lshlrev_b32 v4, 4, v5 // (op 2953) 

v_lshl_add_u32 v5, v6, 7, v4 // (op 2953) 


v_lshrrev_b32 v4, 7, v1 // (op 3006) 
v_and_b32 v6, 31, v1 // (op 3006) 
v_lshl_add_u32 v7, v4, 5, v6 // (op 3006) 


v_and_b32 v4, 63, v1 // (op 3006) 
v_lshrrev_b32 v6, 5, v4 // (op 3006) 

v_lshl_add_u32 v4, v7, 2, v6 // (op 3006) 


v_lshrrev_b32 v6, 7, v1 // (op 2671) 
v_and_b32 v7, 63, v1 // (op 2671) 
v_lshrrev_b32 v8, 5, v7 // (op 2671) 
v_lshlrev_b32 v9, 2, v8 // (op 2671) 

v_lshl_add_u32 v8, v6, 5, v9 // (op 2671) 


v_lshl_add_u32 v6, s16, 6, v8 // (op 2671) 

v_lshrrev_b32 v8, 2, v7 // (op 2671) 

v_and_b32 v7, 7, v8 // (op 2671) 

v_and_b32 v8, 3, v1 // (op 2671) 
v_lshl_add_u32 v9, v7, 2, v8 // (op 2671) 


v_lshrrev_b32 v7, 6, v1 // (op 2671) 
v_and_b32 v8, 1, v7 // (op 2671) 

v_lshl_add_u32 v7, v8, 5, v9 // (op 2671) 


v_lshl_add_u32 v8, s17, 8, v7 // (op 2671) 

v_mul_lo_u32 v7, v8, s2 // (op 2671) 

v_add_lshl_u32 v8, v6, v7, 2 // (op 2671) 


v_and_b32 v6, 63, v1 // (op 2785) 
v_lshrrev_b32 v7, 4, v6 // (op 2785) 

v_and_b32 v6, 1, v7 // (op 2785) 

v_and_b32 v7, 15, v1 // (op 2785) 
v_lshl_add_u32 v9, v6, 4, v7 // (op 2785) 


v_lshrrev_b32 v6, 6, v1 // (op 2785) 
v_and_b32 v7, 1, v6 // (op 2785) 

v_lshl_add_u32 v6, v7, 5, v9 // (op 2785) 


v_and_b32 v7, 32, v1 // (op 2785) 
v_lshl_add_u32 v9, v6, 7, v7 // (op 2785) 


v_mul_lo_u32 v6, v9, 6 // (op 2785) 

v_lshrrev_b32 v7, 3, v6 // (op 2785) 

v_lshrrev_b32 v6, 6, v1 // (op 2896) 
v_and_b32 v9, 1, v6 // (op 2896) 

v_and_b32 v6, 31, v1 // (op 2896) 
s_mul_i32 s1, s2, s9 // (op 2673) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 2673) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 2673) least significant
s_mul_hi_u32 s19, s2, s8 // (op 2673) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2673) most significant: sum
s_add_u32 s1, s1, s19 // (op 2673) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 2673) 

v_lshl_add_u32 v10, v9, 5, v6 // (op 2896) 


v_and_b32 v6, 63, v1 // (op 2896) 
v_lshrrev_b32 v9, 5, v6 // (op 2896) 

v_lshl_add_u32 v6, v10, 2, v9 // (op 2896) 


v_lshrrev_b32 v9, 7, v1 // (op 2912) 
v_and_b32 v10, 63, v1 // (op 2912) 
v_lshrrev_b32 v11, 5, v10 // (op 2912) 
v_lshlrev_b32 v12, 2, v11 // (op 2912) 

v_lshl_add_u32 v11, v9, 5, v12 // (op 2912) 


v_lshl_add_u32 v9, s16, 6, v11 // (op 2912) 

v_lshrrev_b32 v11, 2, v10 // (op 2912) 

v_and_b32 v10, 7, v11 // (op 2912) 

v_and_b32 v11, 3, v1 // (op 2912) 
v_lshl_add_u32 v12, v10, 2, v11 // (op 2912) 


v_lshrrev_b32 v10, 6, v1 // (op 2912) 
v_and_b32 v11, 1, v10 // (op 2912) 

v_lshl_add_u32 v10, v11, 5, v12 // (op 2912) 


v_lshl_add_u32 v11, s17, 8, v10 // (op 2912) 

v_mul_lo_u32 v10, v11, s4 // (op 2912) 

v_add_lshl_u32 v11, v9, v10, 2 // (op 2912) 


v_and_b32 v9, 63, v1 // (op 2982) 
v_lshrrev_b32 v10, 4, v9 // (op 2982) 

v_and_b32 v9, 1, v10 // (op 2982) 

v_and_b32 v10, 15, v1 // (op 2982) 
v_lshl_add_u32 v12, v9, 4, v10 // (op 2982) 


v_lshrrev_b32 v9, 6, v1 // (op 2982) 
v_and_b32 v10, 1, v9 // (op 2982) 

v_lshl_add_u32 v9, v10, 5, v12 // (op 2982) 


v_and_b32 v10, 32, v1 // (op 2982) 
v_lshl_add_u32 v12, v9, 7, v10 // (op 2982) 


v_mul_lo_u32 v9, v12, 6 // (op 2982) 

v_lshrrev_b32 v10, 3, v9 // (op 2982) 

v_lshrrev_b32 v9, 6, v1 // (op 3035) 
v_and_b32 v12, 1, v9 // (op 3035) 

v_and_b32 v9, 31, v1 // (op 3035) 
s_mul_i32 s1, s4, s9 // (op 2914) most significant: low of low * high
v_lshl_add_u32 v13, v12, 5, v9 // (op 3035) 


v_and_b32 v9, 63, v1 // (op 3035) 
v_lshrrev_b32 v12, 5, v9 // (op 3035) 

v_lshl_add_u32 v9, v13, 2, v12 // (op 3035) 


s_mul_i32 s38, s5, s8 // (op 2914) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 2914) least significant
s_mul_hi_u32 s39, s4, s8 // (op 2914) most significant: high of low * low
s_add_u32 s1, s1, s38 // (op 2914) most significant: sum
s_add_u32 s1, s1, s39 // (op 2914) most significant: sum


s_lshl_b64 s[38:39], s[0:1], 2 // (op 2914) 

s_mov_b32 s40, s6 // (op 2673) reg expression
s_mov_b32 s41, s7 // (op 2673) reg expression
s_mov_b32 s42, s18 // (op 2673) reg expression
s_mov_b32 s43, 131072 // (op 2673) call()

s_mov_b32 s44, s10 // (op 2914) reg expression

s_mov_b32 s45, s11 // (op 2914) reg expression
s_mov_b32 s46, s38 // (op 2914) reg expression
s_mov_b32 s47, 131072 // (op 2914) call()


v_mov_b32 v12, s14 // (op 2) Move value
v_mov_b32 v13, s15 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 63) call()
v_accvgpr_write a1, 0.00000 // (op 63) call()
v_accvgpr_write a2, 0.00000 // (op 63) call()
v_accvgpr_write a3, 0.00000 // (op 63) call()
v_accvgpr_write a4, 0.00000 // (op 63) call()
v_accvgpr_write a5, 0.00000 // (op 63) call()
v_accvgpr_write a6, 0.00000 // (op 63) call()
v_accvgpr_write a7, 0.00000 // (op 63) call()
v_accvgpr_write a8, 0.00000 // (op 63) call()
v_accvgpr_write a9, 0.00000 // (op 63) call()
v_accvgpr_write a10, 0.00000 // (op 63) call()
v_accvgpr_write a11, 0.00000 // (op 63) call()
v_accvgpr_write a12, 0.00000 // (op 63) call()
v_accvgpr_write a13, 0.00000 // (op 63) call()
v_accvgpr_write a14, 0.00000 // (op 63) call()
v_accvgpr_write a15, 0.00000 // (op 63) call()
v_accvgpr_write a16, 0.00000 // (op 398) call()
v_accvgpr_write a17, 0.00000 // (op 398) call()
v_accvgpr_write a18, 0.00000 // (op 398) call()
v_accvgpr_write a19, 0.00000 // (op 398) call()
v_accvgpr_write a20, 0.00000 // (op 398) call()
v_accvgpr_write a21, 0.00000 // (op 398) call()
v_accvgpr_write a22, 0.00000 // (op 398) call()
v_accvgpr_write a23, 0.00000 // (op 398) call()
v_accvgpr_write a24, 0.00000 // (op 398) call()
v_accvgpr_write a25, 0.00000 // (op 398) call()
v_accvgpr_write a26, 0.00000 // (op 398) call()
v_accvgpr_write a27, 0.00000 // (op 398) call()
v_accvgpr_write a28, 0.00000 // (op 398) call()
v_accvgpr_write a29, 0.00000 // (op 398) call()
v_accvgpr_write a30, 0.00000 // (op 398) call()
v_accvgpr_write a31, 0.00000 // (op 398) call()
v_accvgpr_write a32, 0.00000 // (op 632) call()
v_accvgpr_write a33, 0.00000 // (op 632) call()
v_accvgpr_write a34, 0.00000 // (op 632) call()
v_accvgpr_write a35, 0.00000 // (op 632) call()
v_accvgpr_write a36, 0.00000 // (op 632) call()
v_accvgpr_write a37, 0.00000 // (op 632) call()
v_accvgpr_write a38, 0.00000 // (op 632) call()
v_accvgpr_write a39, 0.00000 // (op 632) call()
v_accvgpr_write a40, 0.00000 // (op 632) call()
v_accvgpr_write a41, 0.00000 // (op 632) call()
v_accvgpr_write a42, 0.00000 // (op 632) call()
v_accvgpr_write a43, 0.00000 // (op 632) call()
v_accvgpr_write a44, 0.00000 // (op 632) call()
v_accvgpr_write a45, 0.00000 // (op 632) call()
v_accvgpr_write a46, 0.00000 // (op 632) call()
v_accvgpr_write a47, 0.00000 // (op 632) call()
v_accvgpr_write a48, 0.00000 // (op 866) call()
v_accvgpr_write a49, 0.00000 // (op 866) call()
v_accvgpr_write a50, 0.00000 // (op 866) call()
v_accvgpr_write a51, 0.00000 // (op 866) call()
v_accvgpr_write a52, 0.00000 // (op 866) call()
v_accvgpr_write a53, 0.00000 // (op 866) call()
v_accvgpr_write a54, 0.00000 // (op 866) call()
v_accvgpr_write a55, 0.00000 // (op 866) call()
v_accvgpr_write a56, 0.00000 // (op 866) call()
v_accvgpr_write a57, 0.00000 // (op 866) call()
v_accvgpr_write a58, 0.00000 // (op 866) call()
v_accvgpr_write a59, 0.00000 // (op 866) call()
v_accvgpr_write a60, 0.00000 // (op 866) call()
v_accvgpr_write a61, 0.00000 // (op 866) call()
v_accvgpr_write a62, 0.00000 // (op 866) call()
v_accvgpr_write a63, 0.00000 // (op 866) call()
v_lshrrev_b32 v14, 3, v1 // (op 2438) 
v_lshl_add_u32 v15, s16, 6, v14 // (op 2438) 

v_and_b32 v14, -8, v1 // (op 2438) 
v_sub_u32 v16, v1, v14 // (op 2438) 

v_lshlrev_b32 v14, 4, v16 // (op 2438) 

v_mov_b32 v17, 0 // (op 2438) convert
v_mov_b32 v16, v14 // (op 2438) convert

v_mul_lo_u32 v19, v15, s13 // (op 2438) most significant: low of low * high
v_mul_hi_u32 v14, v15, s12 // (op 2438) most significant: high of low * low
v_mul_lo_u32 v18, v15, s12 // (op 2438) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 2438) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 2438) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 2438) most significant half




s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 2440) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 2440) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 2440) least significant
s_mul_hi_u32 s7, s12, s22 // (op 2440) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2440) most significant: sum
s_add_u32 s1, s1, s7 // (op 2440) most significant: sum


s_mov_b32 s48, s20 // (op 2440) reg expression
s_mov_b32 s49, s21 // (op 2440) reg expression
s_mov_b32 s50, s0 // (op 2440) reg expression
s_mov_b32 s51, 131072 // (op 2440) call()

v_lshrrev_b32 v14, 3, v1 // (op 2457) 
v_lshl_add_u32 v15, s16, 6, v14 // (op 2457) 

v_and_b32 v14, -8, v1 // (op 2457) 
v_sub_u32 v16, v1, v14 // (op 2457) 

v_lshlrev_b32 v14, 4, v16 // (op 2457) 

v_mov_b32 v17, 0 // (op 2457) convert
v_mov_b32 v16, v14 // (op 2457) convert

v_mul_lo_u32 v21, v15, s13 // (op 2457) most significant: low of low * high
v_mul_hi_u32 v14, v15, s12 // (op 2457) most significant: high of low * low
v_mul_lo_u32 v20, v15, s12 // (op 2457) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 2457) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v16 // (op 2457) least significant half
v_addc_co_u32 v21, s[0:1], v21, v17, s[0:1] // (op 2457) most significant half



s_mul_i32 s1, s12, s23 // (op 2459) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 2459) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 2459) least significant
s_mul_hi_u32 s7, s12, s22 // (op 2459) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2459) most significant: sum
s_add_u32 s1, s1, s7 // (op 2459) most significant: sum


s_mov_b32 s48, s20 // (op 2459) reg expression
s_mov_b32 s49, s21 // (op 2459) reg expression
s_mov_b32 s50, s0 // (op 2459) reg expression
s_mov_b32 s51, 131072 // (op 2459) call()

v_and_b32 v14, -2, v1 // (op 2502) 
v_sub_u32 v15, v1, v14 // (op 2502) 

v_lshlrev_b32 v14, 6, v15 // (op 2502) 

v_mov_b32 v17, 0 // (op 2502) convert
v_mov_b32 v16, v14 // (op 2502) convert

v_and_b32 v14, 4294967294, v1 // (op 2502) 
v_lshl_add_u32 v15, s17, 8, v14 // (op 2502) 

v_mul_lo_u32 v23, v15, s25 // (op 2502) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 2502) most significant: high of low * low
v_mul_lo_u32 v22, v15, s24 // (op 2502) least significant: low of low * low
v_add_u32 v23, v23, v14 // (op 2502) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v22 // (op 2502) least significant half
v_addc_co_u32 v15, s[0:1], v17, v23, s[0:1] // (op 2502) most significant half



v_mul_lo_u32 v17, v15, 6 // (op 2502) most significant: low of high * low
v_mul_hi_u32 v22, v14, 6 // (op 2502) most significant: high of low * low
v_mul_lo_u32 v16, v14, 6 // (op 2502) least significant: low of low * low
v_add_u32 v17, v17, v22 // (op 2502) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 2502) 

v_mov_b32 v16, v14 // (op 2502) convert

s_mul_i32 s1, s9, 6 // (op 2504) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 2504) least significant
s_mul_hi_u32 s6, s8, 6 // (op 2504) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2504) most significant: sum

s_mul_i32 s7, s24, s1 // (op 2504) most significant: low of low * high
s_mul_i32 s10, s25, s0 // (op 2504) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 2504) least significant
s_mul_hi_u32 s11, s24, s0 // (op 2504) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 2504) most significant: sum
s_add_u32 s7, s7, s11 // (op 2504) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 2504) 

s_mov_b32 s52, s26 // (op 2504) reg expression
s_mov_b32 s53, s27 // (op 2504) reg expression
s_mov_b32 s54, s0 // (op 2504) reg expression
s_mov_b32 s55, 131072 // (op 2504) call()

v_and_b32 v17, -2, v1 // (op 2521) 
v_sub_u32 v14, v1, v17 // (op 2521) 

v_lshlrev_b32 v15, 6, v14 // (op 2521) 

v_mov_b32 v23, 0 // (op 2521) convert
v_mov_b32 v22, v15 // (op 2521) convert

v_and_b32 v17, 4294967294, v1 // (op 2521) 
v_lshl_add_u32 v14, s17, 8, v17 // (op 2521) 

v_mul_lo_u32 v25, v14, s25 // (op 2521) most significant: low of low * high
v_mul_hi_u32 v15, v14, s24 // (op 2521) most significant: high of low * low
v_mul_lo_u32 v24, v14, s24 // (op 2521) least significant: low of low * low
v_add_u32 v25, v25, v15 // (op 2521) most significant: sum


v_add_co_u32 v14, s[0:1], v22, v24 // (op 2521) least significant half
v_addc_co_u32 v15, s[0:1], v23, v25, s[0:1] // (op 2521) most significant half



v_mul_lo_u32 v23, v15, 6 // (op 2521) most significant: low of high * low
v_mul_hi_u32 v17, v14, 6 // (op 2521) most significant: high of low * low
v_mul_lo_u32 v22, v14, 6 // (op 2521) least significant: low of low * low
v_add_u32 v23, v23, v17 // (op 2521) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[22:23] // (op 2521) 

v_mov_b32 v17, v14 // (op 2521) convert

s_mul_i32 s1, s9, 6 // (op 2523) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 2523) least significant
s_mul_hi_u32 s6, s8, 6 // (op 2523) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2523) most significant: sum

s_mul_i32 s7, s24, s1 // (op 2523) most significant: low of low * high
s_mul_i32 s10, s25, s0 // (op 2523) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 2523) least significant
s_mul_hi_u32 s11, s24, s0 // (op 2523) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 2523) most significant: sum
s_add_u32 s7, s7, s11 // (op 2523) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 2523) 

s_mov_b32 s52, s26 // (op 2523) reg expression
s_mov_b32 s53, s27 // (op 2523) reg expression
s_mov_b32 s54, s0 // (op 2523) reg expression
s_mov_b32 s55, 131072 // (op 2523) call()

v_lshrrev_b32 v14, 2, v1 // (op 2566) 
v_lshl_add_u32 v15, s16, 6, v14 // (op 2566) 

v_and_b32 v14, -4, v1 // (op 2566) 
v_sub_u32 v22, v1, v14 // (op 2566) 

v_mul_lo_u32 v14, v15, s28 // (op 2566) 
v_add_u32 v14, v14, v22 // (op 2566) 


s_mul_i32 s1, s28, s23 // (op 2568) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 2568) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 2568) least significant
s_mul_hi_u32 s7, s28, s22 // (op 2568) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2568) most significant: sum
s_add_u32 s1, s1, s7 // (op 2568) most significant: sum


s_mov_b32 s56, s30 // (op 2568) reg expression
s_mov_b32 s57, s31 // (op 2568) reg expression
s_mov_b32 s58, s0 // (op 2568) reg expression
s_mov_b32 s59, 131072 // (op 2568) call()

v_lshrrev_b32 v15, 2, v1 // (op 2585) 
v_lshl_add_u32 v22, s16, 6, v15 // (op 2585) 

v_and_b32 v15, -4, v1 // (op 2585) 
v_sub_u32 v23, v1, v15 // (op 2585) 

v_mul_lo_u32 v15, v22, s28 // (op 2585) 
v_add_u32 v15, v15, v23 // (op 2585) 


s_mul_i32 s1, s28, s23 // (op 2587) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 2587) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 2587) least significant
s_mul_hi_u32 s7, s28, s22 // (op 2587) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2587) most significant: sum
s_add_u32 s1, s1, s7 // (op 2587) most significant: sum


s_mov_b32 s56, s30 // (op 2587) reg expression
s_mov_b32 s57, s31 // (op 2587) reg expression
s_mov_b32 s58, s0 // (op 2587) reg expression
s_mov_b32 s59, 131072 // (op 2587) call()

v_lshl_add_u32 v22, s17, 8, v1 // (op 2630) 
v_mul_lo_u32 v23, v22, s32 // (op 2630) 

s_mul_i32 s1, s32, s9 // (op 2632) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 2632) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 2632) least significant
s_mul_hi_u32 s7, s32, s8 // (op 2632) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2632) most significant: sum
s_add_u32 s1, s1, s7 // (op 2632) most significant: sum


s_mov_b32 s60, s34 // (op 2632) reg expression
s_mov_b32 s61, s35 // (op 2632) reg expression
s_mov_b32 s62, s0 // (op 2632) reg expression
s_mov_b32 s63, 131072 // (op 2632) call()

v_lshl_add_u32 v22, s17, 8, v1 // (op 2649) 
v_mul_lo_u32 v24, v22, s32 // (op 2649) 

s_mul_i32 s1, s32, s9 // (op 2651) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 2651) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 2651) least significant
s_mul_hi_u32 s7, s32, s8 // (op 2651) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2651) most significant: sum
s_add_u32 s1, s1, s7 // (op 2651) most significant: sum


s_mov_b32 s60, s34 // (op 2651) reg expression
s_mov_b32 s61, s35 // (op 2651) reg expression
s_mov_b32 s62, s0 // (op 2651) reg expression
s_mov_b32 s63, 131072 // (op 2651) call()


s_mul_i32 s1, s25, 6 // (op 2153) most significant: low of high * low
s_mul_i32 s0, s24, 6 // (op 2153) least significant
s_mul_hi_u32 s6, s24, 6 // (op 2153) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2153) most significant: sum

v_lshrrev_b64 v[26:27], 3, s[0:1] // (op 2153) 


v_mov_b32 v22, v16 // (op 2153) 
buffer_load_dwordx4 v[28:31], v22, s[52:55], 0 offen // (op 2153) Load value
buffer_load_dwordx2 v[32:33], v22, s[52:55], 0 offen offset:16 // (op 2153) Load value
buffer_load_dwordx4 v[34:37], v22, s[52:55], 0 offen offset:24 // (op 2153) Load value
buffer_load_dwordx2 v[38:39], v22, s[52:55], 0 offen offset:40 // (op 2153) Load value
v_add_u32 v22, v22, v26 // (op 2153) 
buffer_load_dwordx4 v[40:43], v22, s[52:55], 0 offen // (op 2153) Load value
buffer_load_dwordx2 v[44:45], v22, s[52:55], 0 offen offset:16 // (op 2153) Load value
buffer_load_dwordx4 v[46:49], v22, s[52:55], 0 offen offset:24 // (op 2153) Load value
buffer_load_dwordx2 v[50:51], v22, s[52:55], 0 offen offset:40 // (op 2153) Load value





buffer_load_dword v22, v23, s[60:63], 0 offen // (op 2156) Load value
v_and_b32 v25, -8, v1 // (op 2715) 
v_sub_u32 v26, v1, v25 // (op 2715) 

v_lshlrev_b32 v25, 4, v26 // (op 2715) 

v_lshrrev_b32 v26, 3, v1 // (op 2715) 
v_lshl_add_u32 v27, v26, 7, v25 // (op 2715) 


s_nop 0
v_readfirstlane_b32 s0, v27 // (op 2715) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 5 // (op 2391) 
v_mov_b64 v[26:27], s[6:7] // (op 2391) convert


v_mov_b64 v[52:53], v[18:19] // (op 2391) 
s_mov_b32 m0, s0 // (op 2391) reg expression
buffer_load_dwordx4 v52, s[48:51], 0 offen offset: 0 lds // (op 2391) Load value direct to lds
v_add_co_u32 v52, s[6:7], v52, v26 // (op 2391) least significant half
v_addc_co_u32 v53, s[6:7], 0, v53, s[6:7] // (op 2391) most significant half

s_add_u32 m0, m0, 4096 // (op 2391) 
buffer_load_dwordx4 v52, s[48:51], 0 offen offset: 0 lds // (op 2391) Load value direct to lds






buffer_load_ubyte v25, v14, s[56:59], 0 offen offset: 0 // (op 2162) Load value

s_mul_i32 s1, s25, 6 // (op 2165) most significant: low of high * low
s_mul_i32 s0, s24, 6 // (op 2165) least significant
s_mul_hi_u32 s6, s24, 6 // (op 2165) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2165) most significant: sum

v_lshrrev_b64 v[26:27], 3, s[0:1] // (op 2165) 


v_mov_b32 v52, v16 // (op 2165) 
buffer_load_dwordx4 v[54:57], v52, s[52:55], 0 offen offset:96 // (op 2165) Load value
buffer_load_dwordx2 v[58:59], v52, s[52:55], 0 offen offset:112 // (op 2165) Load value
buffer_load_dwordx4 v[60:63], v52, s[52:55], 0 offen offset:120 // (op 2165) Load value
buffer_load_dwordx2 v[64:65], v52, s[52:55], 0 offen offset:136 // (op 2165) Load value
v_add_u32 v52, v52, v26 // (op 2165) 
buffer_load_dwordx4 v[66:69], v52, s[52:55], 0 offen offset:96 // (op 2165) Load value
buffer_load_dwordx2 v[70:71], v52, s[52:55], 0 offen offset:112 // (op 2165) Load value
buffer_load_dwordx4 v[72:75], v52, s[52:55], 0 offen offset:120 // (op 2165) Load value
buffer_load_dwordx2 v[76:77], v52, s[52:55], 0 offen offset:136 // (op 2165) Load value






buffer_load_dword v16, v23, s[60:63], 0 offen offset:4 // (op 2168) Load value

v_and_b32 v23, -8, v1 // (op 2941) 
v_sub_u32 v26, v1, v23 // (op 2941) 

v_lshlrev_b32 v23, 4, v26 // (op 2941) 

v_lshrrev_b32 v26, 3, v1 // (op 2941) 
v_lshl_add_u32 v27, v26, 7, v23 // (op 2941) 


s_nop 0
v_readfirstlane_b32 s0, v27 // (op 2941) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 5 // (op 2396) 
v_mov_b64 v[26:27], s[6:7] // (op 2396) convert


v_mov_b64 v[52:53], v[18:19] // (op 2396) 
s_add_u32 s1, 8192, s0 // (op 2396) 
s_mov_b32 m0, s1 // (op 2396) reg expression
s_mov_b32 s6, 128 // (op 2396) call()
buffer_load_dwordx4 v52, s[48:51], s6 offen offset: 0 lds // (op 2396) Load value direct to lds
v_add_co_u32 v52, s[10:11], v52, v26 // (op 2396) least significant half
v_addc_co_u32 v53, s[10:11], 0, v53, s[10:11] // (op 2396) most significant half

s_add_u32 m0, m0, 4096 // (op 2396) 
buffer_load_dwordx4 v52, s[48:51], s6 offen offset: 0 lds // (op 2396) Load value direct to lds









buffer_load_ubyte v23, v14, s[56:59], 0 offen offset:4 // (op 2174) Load value

v_and_b32 v14, -2, v1 // (op 2768) 
v_sub_u32 v18, v1, v14 // (op 2768) 

v_lshlrev_b32 v14, 6, v18 // (op 2768) 

v_lshrrev_b32 v18, 1, v1 // (op 2768) 
v_lshl_add_u32 v19, v18, 8, v14 // (op 2768) 


v_mul_lo_u32 v14, v19, 6 // (op 2768) 

v_lshrrev_b32 v18, 3, v14 // (op 2768) 




s_waitcnt vmcnt(23)
ds_write_b128 v18, v[28:31] offset:16384 // (op 2176) Store local data 
s_waitcnt vmcnt(22)
ds_write_b64 v18, v[32:33] offset:16400 // (op 2176) Store local data 
s_waitcnt vmcnt(21)
ds_write_b128 v18, v[34:37] offset:16408 // (op 2176) Store local data 
s_waitcnt vmcnt(20)
ds_write_b64 v18, v[38:39] offset:16424 // (op 2176) Store local data 
s_waitcnt vmcnt(19)
ds_write_b128 v18, v[40:43] offset:16480 // (op 2176) Store local data 
s_waitcnt vmcnt(18)
ds_write_b64 v18, v[44:45] offset:16496 // (op 2176) Store local data 
s_waitcnt vmcnt(17)
ds_write_b128 v18, v[46:49] offset:16504 // (op 2176) Store local data 
s_waitcnt vmcnt(16)
ds_write_b64 v18, v[50:51] offset:16520 // (op 2176) Store local data 

v_lshlrev_b32 v14, 2, v1 // (op 2879) 



s_waitcnt vmcnt(15)
ds_write_b32 v14, v22 offset:40960 // (op 2177) Store local data 

v_and_b32 v14, -4, v1 // (op 2826) 
v_sub_u32 v18, v1, v14 // (op 2826) 

v_and_b32 v14, 4294967292, v1 // (op 2826) 
v_add_u32 v19, v14, v18 // (op 2826) 





s_waitcnt vmcnt(12)
ds_write_b8 v19, v25 offset:41984 // (op 2179) Store local data 

s_waitcnt lgkmcnt(2)
s_barrier  // (op 3117) 



ds_read_b128 v[78:81], v7 offset:16384 // (op 2197) Load local data 
ds_read_b64 v[82:83], v7 offset:16400 // (op 2197) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 3158) 



ds_read_u8 v14, v6 offset:40960 // (op 2220) Load local data 



ds_read_b128 v[84:87], v7 offset:22528 // (op 2202) Load local data 
ds_read_b64 v[88:89], v7 offset:22544 // (op 2202) Load local data 



ds_read_u8 v18, v6 offset:41216 // (op 2225) Load local data 



ds_read_b128 v[90:93], v7 offset:28672 // (op 2207) Load local data 
ds_read_b64 v[94:95], v7 offset:28688 // (op 2207) Load local data 



ds_read_u8 v19, v6 offset:41472 // (op 2230) Load local data 



ds_read_b128 v[96:99], v7 offset:34816 // (op 2212) Load local data 
ds_read_b64 v[100:101], v7 offset:34832 // (op 2212) Load local data 



ds_read_u8 v26, v6 offset:41728 // (op 2235) Load local data 
s_barrier  // (op 3098) 



ds_read_b128 v[102:105], v3 // (op 2193) Load local data 
ds_read_b128 v[106:109], v3 offset:32 // (op 2193) Load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 3138) 



ds_read_u8 v27, v0 offset:41984 // (op 2216) Load local data 
v_and_b32 v52, -8, v1 // (op 2687) 
v_sub_u32 v53, v1, v52 // (op 2687) 

v_lshlrev_b32 v52, 4, v53 // (op 2687) 

v_lshrrev_b32 v53, 3, v1 // (op 2687) 
v_lshl_add_u32 v110, v53, 7, v52 // (op 2687) 


s_nop 0
v_readfirstlane_b32 s0, v110 // (op 2687) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v52, -2, v1 // (op 2739) 
v_sub_u32 v53, v1, v52 // (op 2739) 

v_lshlrev_b32 v52, 6, v53 // (op 2739) 

v_lshrrev_b32 v53, 1, v1 // (op 2739) 
v_lshl_add_u32 v110, v53, 8, v52 // (op 2739) 


v_mul_lo_u32 v52, v110, 6 // (op 2739) 

v_lshrrev_b32 v53, 3, v52 // (op 2739) 

v_and_b32 v52, -4, v1 // (op 2797) 
v_sub_u32 v110, v1, v52 // (op 2797) 

v_and_b32 v52, 4294967292, v1 // (op 2797) 
v_add_u32 v111, v52, v110 // (op 2797) 


v_lshlrev_b32 v52, 2, v1 // (op 2850) 
v_and_b32 v110, -8, v1 // (op 2926) 
v_sub_u32 v112, v1, v110 // (op 2926) 

v_lshlrev_b32 v110, 4, v112 // (op 2926) 

v_lshrrev_b32 v112, 3, v1 // (op 2926) 
v_lshl_add_u32 v113, v112, 7, v110 // (op 2926) 


s_nop 0
v_readfirstlane_b32 s1, v113 // (op 2926) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v110, -2, v1 // (op 2965) 
v_sub_u32 v112, v1, v110 // (op 2965) 

v_lshlrev_b32 v110, 6, v112 // (op 2965) 

v_lshrrev_b32 v112, 1, v1 // (op 2965) 
v_lshl_add_u32 v113, v112, 8, v110 // (op 2965) 


v_mul_lo_u32 v110, v113, 6 // (op 2965) 

v_lshrrev_b32 v112, 3, v110 // (op 2965) 

v_and_b32 v110, -4, v1 // (op 2994) 
v_sub_u32 v113, v1, v110 // (op 2994) 

v_and_b32 v110, 4294967292, v1 // (op 2994) 
v_add_u32 v114, v110, v113 // (op 2994) 


v_lshlrev_b32 v110, 2, v1 // (op 3018) 
s_mov_b32 s6, 0 // (op 868) call()
s_ashr_i64 s[10:11], s[36:37], 7 // (op 867) 
s_mov_b32 s7, s10 // (op 867) convert

s_and_b32 s10, s7, 4294967294 // (op 867) 

s_cmp_lt_u32 s6, s10 // (op 867) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ForLoopBottom_KLoop_867 // (op 867) Condition: Top (jump to RR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ForLoopBottom_KLoop_867 if false)
RR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ForLoopTop_KLoop_867:

s_add_u32 s6, s6, 2 // (op 869) 
s_mul_i32 s11, s25, 6 // (op 10) most significant: low of high * low
s_mul_i32 s10, s24, 6 // (op 10) least significant
s_mul_hi_u32 s7, s24, 6 // (op 10) most significant: high of low * low
s_add_u32 s11, s11, s7 // (op 10) most significant: sum

v_lshrrev_b64 v[116:117], 3, s[10:11] // (op 10) 


v_mov_b32 v113, v17 // (op 10) 
v_mov_b32 v115, v113 // (op 10) 
buffer_load_dwordx4 v[28:31], v115, s[52:55], 0 offen offset:192 // (op 10) Load value
buffer_load_dwordx2 v[32:33], v115, s[52:55], 0 offen offset:208 // (op 10) Load value
buffer_load_dwordx4 v[34:37], v115, s[52:55], 0 offen offset:216 // (op 10) Load value
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[102:109], v[78:83], a[0:15], v27, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 88) 
buffer_load_dwordx2 v[38:39], v115, s[52:55], 0 offen offset:232 // (op 10) Load value
s_barrier  // (op 3107) 
v_add_u32 v115, v115, v116 // (op 10) 



buffer_load_dwordx4 v[40:43], v115, s[52:55], 0 offen offset:192 // (op 10) Load value
buffer_load_dwordx2 v[44:45], v115, s[52:55], 0 offen offset:208 // (op 10) Load value
ds_read_b128 v[118:121], v3 offset:64 // (op 91) Load local data 
ds_read_b128 v[122:125], v3 offset:96 // (op 91) Load local data 
buffer_load_dwordx4 v[46:49], v115, s[52:55], 0 offen offset:216 // (op 10) Load value
s_barrier  // (op 3127) 
buffer_load_dwordx2 v[50:51], v115, s[52:55], 0 offen offset:232 // (op 10) Load value







ds_read_b128 v[78:81], v7 offset:16432 // (op 95) Load local data 
ds_read_b64 v[82:83], v7 offset:16448 // (op 95) Load local data 
v_mov_b32 v115, v24 // (op 25) 

s_barrier  // (op 3148) 
buffer_load_dword v22, v115, s[60:63], 0 offen offset:8 // (op 25) Load value



ds_read_u8 v116, v0 offset:41986 // (op 99) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 3101) 

s_barrier  // (op 3168) 
s_lshl_b64 s[10:11], s[12:13], 5 // (op 2375) 



v_mov_b64 v[126:127], s[10:11] // (op 2375) convert


v_mov_b64 v[128:129], v[20:21] // (op 2375) 
v_mov_b64 v[130:131], v[128:129] // (op 2375) 
s_mov_b32 m0, s0 // (op 2375) reg expression
s_mov_b32 s7, 256 // (op 2375) call()
ds_read_u8 v14, v6 offset:40962 // (op 103) Load local data 
buffer_load_dwordx4 v130, s[48:51], s7 offen offset: 0 lds // (op 2375) Load value direct to lds
v_add_co_u32 v130, s[10:11], v130, v126 // (op 2375) least significant half
v_addc_co_u32 v131, s[10:11], 0, v131, s[10:11] // (op 2375) most significant half

s_add_u32 m0, m0, 4096 // (op 2375) 
buffer_load_dwordx4 v130, s[48:51], s7 offen offset: 0 lds // (op 2375) Load value direct to lds



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[118:125], v[78:83], a[0:15], v116, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 105) 


v_mov_b32 v117, v15 // (op 17) 

buffer_load_ubyte v25, v117, s[56:59], 0 offen offset:8 // (op 17) Load value
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[102:109], v[84:89], a[16:31], v27, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 447) 



s_barrier  // (op 3188) 
ds_read_b128 v[78:81], v7 offset:22576 // (op 451) Load local data 



ds_read_b64 v[82:83], v7 offset:22592 // (op 451) Load local data 
s_waitcnt vmcnt(23)
ds_write_b128 v112, v[54:57] offset:42240 // (op 302) Store local data 



ds_read_u8 v14, v6 offset:41218 // (op 455) Load local data 
s_waitcnt vmcnt(22)
ds_write_b64 v112, v[58:59] offset:42256 // (op 302) Store local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[118:125], v[78:83], a[16:31], v116, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 456) 
s_waitcnt vmcnt(21)
ds_write_b128 v112, v[60:63] offset:42264 // (op 302) Store local data 
s_waitcnt vmcnt(20)
ds_write_b64 v112, v[64:65] offset:42280 // (op 302) Store local data 
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[102:109], v[90:95], a[32:47], v27, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 681) 
s_waitcnt vmcnt(19)
ds_write_b128 v112, v[66:69] offset:42336 // (op 302) Store local data 



ds_read_b128 v[78:81], v7 offset:28720 // (op 685) Load local data 
ds_read_b64 v[82:83], v7 offset:28736 // (op 685) Load local data 
s_waitcnt vmcnt(18)
ds_write_b64 v112, v[70:71] offset:42352 // (op 302) Store local data 



ds_read_u8 v14, v6 offset:41474 // (op 689) Load local data 
s_waitcnt vmcnt(17)
ds_write_b128 v112, v[72:75] offset:42360 // (op 302) Store local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[118:125], v[78:83], a[32:47], v116, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 690) 
s_waitcnt vmcnt(16)
ds_write_b64 v112, v[76:77] offset:42376 // (op 302) Store local data 
s_barrier  // (op 3200) 
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[102:109], v[96:101], a[48:63], v27, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 915) 






ds_read_b128 v[96:99], v7 offset:34864 // (op 919) Load local data 
ds_read_b64 v[100:101], v7 offset:34880 // (op 919) Load local data 
v_add_u32 v126, 66816, v110 // (op 304) 



ds_read_u8 v26, v6 offset:41730 // (op 923) Load local data 
s_waitcnt vmcnt(15)
ds_write_b32 v126, v16 // (op 304) Store local data 

s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[118:125], v[96:101], a[48:63], v116, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 924) 


s_barrier  // (op 3194) 



v_add_u32 v116, 67840, v114 // (op 303) 
s_waitcnt vmcnt(12)
ds_write_b8 v116, v23 // (op 303) Store local data 

s_barrier  // (op 3184) 



ds_read_b128 v[78:81], v10 offset:42240 // (op 287) Load local data 
ds_read_b64 v[82:83], v10 offset:42256 // (op 287) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 3197) 



v_add_u32 v116, 66816, v9 // (op 291) 
ds_read_u8 v14, v116 // (op 291) Load local data 




ds_read_b128 v[84:87], v10 offset:48384 // (op 470) Load local data 
ds_read_b64 v[88:89], v10 offset:48400 // (op 470) Load local data 



v_add_u32 v116, 67072, v9 // (op 474) 
ds_read_u8 v18, v116 // (op 474) Load local data 




ds_read_b128 v[90:93], v10 offset:54528 // (op 704) Load local data 
ds_read_b64 v[94:95], v10 offset:54544 // (op 704) Load local data 



v_add_u32 v116, 67328, v9 // (op 708) 
ds_read_u8 v19, v116 // (op 708) Load local data 




ds_read_b128 v[118:121], v10 offset:60672 // (op 938) Load local data 
ds_read_b64 v[122:123], v10 offset:60688 // (op 938) Load local data 



v_add_u32 v116, 67584, v9 // (op 942) 
ds_read_u8 v124, v116 // (op 942) Load local data 

s_barrier  // (op 3178) 



ds_read_b128 v[102:105], v5 offset:8192 // (op 285) Load local data 
ds_read_b128 v[106:109], v5 offset:8224 // (op 285) Load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 3191) 



v_add_u32 v116, 67840, v4 // (op 289) 
ds_read_u8 v27, v116 // (op 289) Load local data 


s_mul_i32 s11, s25, 6 // (op 305) most significant: low of high * low
s_mul_i32 s10, s24, 6 // (op 305) least significant
s_mul_hi_u32 s7, s24, 6 // (op 305) most significant: high of low * low
s_add_u32 s11, s11, s7 // (op 305) most significant: sum

v_lshrrev_b64 v[126:127], 3, s[10:11] // (op 305) 


v_mov_b32 v116, v113 // (op 305) 
buffer_load_dwordx4 v[54:57], v116, s[52:55], 0 offen offset:288 // (op 305) Load value
buffer_load_dwordx2 v[58:59], v116, s[52:55], 0 offen offset:304 // (op 305) Load value
buffer_load_dwordx4 v[60:63], v116, s[52:55], 0 offen offset:312 // (op 305) Load value
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[102:109], v[78:83], a[0:15], v27, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 292) 
buffer_load_dwordx2 v[64:65], v116, s[52:55], 0 offen offset:328 // (op 305) Load value
v_add_u32 v116, v116, v126 // (op 305) 



v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[102:109], v[84:89], a[16:31], v27, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 457) 
buffer_load_dwordx4 v[66:69], v116, s[52:55], 0 offen offset:288 // (op 305) Load value
buffer_load_dwordx2 v[70:71], v116, s[52:55], 0 offen offset:304 // (op 305) Load value
ds_read_b128 v[130:133], v5 offset:8256 // (op 294) Load local data 
ds_read_b128 v[134:137], v5 offset:8288 // (op 294) Load local data 
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[102:109], v[90:95], a[32:47], v27, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 691) 



ds_read_b128 v[138:141], v10 offset:42288 // (op 296) Load local data 
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[102:109], v[118:123], a[48:63], v27, v124 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 925) 
ds_read_b64 v[142:143], v10 offset:42304 // (op 296) Load local data 





v_add_u32 v118, 67842, v4 // (op 298) 
ds_read_u8 v119, v118 // (op 298) Load local data 

buffer_load_dwordx4 v[72:75], v116, s[52:55], 0 offen offset:312 // (op 305) Load value
buffer_load_dwordx2 v[76:77], v116, s[52:55], 0 offen offset:328 // (op 305) Load value






s_barrier  // (op 3239) 



buffer_load_dword v16, v115, s[60:63], 0 offen offset:12 // (op 307) Load value
v_add_u32 v17, 192, v17 // (op 2513) 

v_add_u32 v113, 66818, v9 // (op 300) 
v_add_u32 v24, 8, v24 // (op 2641) 
ds_read_u8 v118, v113 // (op 300) Load local data 

s_waitcnt lgkmcnt(4)
s_barrier  // (op 3181) 




s_lshl_b64 s[10:11], s[12:13], 5 // (op 2386) 
v_mov_b64 v[120:121], s[10:11] // (op 2386) convert


v_mov_b64 v[122:123], v[128:129] // (op 2386) 
s_add_u32 s7, 8192, s1 // (op 2386) 
s_mov_b32 m0, s7 // (op 2386) reg expression
s_mov_b32 s10, 384 // (op 2386) call()
ds_read_b128 v[144:147], v10 offset:48432 // (op 461) Load local data 
ds_read_b64 v[148:149], v10 offset:48448 // (op 461) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[130:137], v[138:143], a[0:15], v119, v118 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 301) 





v_add_u32 v113, 67074, v9 // (op 465) 
ds_read_u8 v118, v113 // (op 465) Load local data 

buffer_load_dwordx4 v122, s[48:51], s10 offen offset: 0 lds // (op 2386) Load value direct to lds



v_add_co_u32 v122, s[14:15], v122, v120 // (op 2386) least significant half
v_addc_co_u32 v123, s[14:15], 0, v123, s[14:15] // (op 2386) most significant half

s_add_u32 m0, m0, 4096 // (op 2386) 
buffer_load_dwordx4 v122, s[48:51], s10 offen offset: 0 lds // (op 2386) Load value direct to lds




ds_read_b128 v[138:141], v10 offset:54576 // (op 695) Load local data 

ds_read_b64 v[142:143], v10 offset:54592 // (op 695) Load local data 






buffer_load_ubyte v23, v117, s[56:59], 0 offen offset:12 // (op 306) Load value
v_add_co_u32 v20, vcc, 256, v20 // (op 2449) least significant half

v_addc_co_u32 v21, vcc, 0, v21, vcc // (op 2449) most significant half
s_barrier  // (op 3120) 
v_add_u32 v15, 8, v15 // (op 2577) 



v_add_u32 v113, 67330, v9 // (op 699) 
ds_read_u8 v115, v113 // (op 699) Load local data 

s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[130:137], v[144:149], a[16:31], v119, v118 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 466) 





ds_read_b128 v[120:123], v10 offset:60720 // (op 929) Load local data 
ds_read_b64 v[124:125], v10 offset:60736 // (op 929) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[130:137], v[138:143], a[32:47], v119, v115 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 700) 





s_barrier  // (op 3297) 
v_add_u32 v113, 67586, v9 // (op 933) 
ds_read_u8 v115, v113 // (op 933) Load local data 

s_waitcnt vmcnt(23)
ds_write_b128 v53, v[28:31] offset:16384 // (op 13) Store local data 
s_waitcnt vmcnt(22)
ds_write_b64 v53, v[32:33] offset:16400 // (op 13) Store local data 
s_barrier  // (op 3301) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[130:137], v[120:125], a[48:63], v119, v115 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 934) 
s_waitcnt vmcnt(21)
ds_write_b128 v53, v[34:37] offset:16408 // (op 13) Store local data 




s_waitcnt vmcnt(20)
ds_write_b64 v53, v[38:39] offset:16424 // (op 13) Store local data 
s_waitcnt vmcnt(19)
ds_write_b128 v53, v[40:43] offset:16480 // (op 13) Store local data 
s_waitcnt vmcnt(18)
ds_write_b64 v53, v[44:45] offset:16496 // (op 13) Store local data 
s_waitcnt vmcnt(17)
ds_write_b128 v53, v[46:49] offset:16504 // (op 13) Store local data 
s_waitcnt vmcnt(16)
ds_write_b64 v53, v[50:51] offset:16520 // (op 13) Store local data 
s_barrier  // (op 3161) 



s_waitcnt vmcnt(15)
ds_write_b32 v52, v22 offset:40960 // (op 30) Store local data 
s_barrier  // (op 3141) 



s_waitcnt vmcnt(12)
ds_write_b8 v111, v25 offset:41984 // (op 21) Store local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 3130) 



ds_read_b128 v[78:81], v7 offset:16384 // (op 75) Load local data 
ds_read_b64 v[82:83], v7 offset:16400 // (op 75) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 3171) 



ds_read_u8 v14, v6 offset:40960 // (op 85) Load local data 



ds_read_b128 v[84:87], v7 offset:22528 // (op 442) Load local data 
ds_read_b64 v[88:89], v7 offset:22544 // (op 442) Load local data 



ds_read_u8 v18, v6 offset:41216 // (op 446) Load local data 



ds_read_b128 v[90:93], v7 offset:28672 // (op 676) Load local data 
ds_read_b64 v[94:95], v7 offset:28688 // (op 676) Load local data 



ds_read_u8 v19, v6 offset:41472 // (op 680) Load local data 



ds_read_b128 v[96:99], v7 offset:34816 // (op 910) Load local data 
ds_read_b64 v[100:101], v7 offset:34832 // (op 910) Load local data 



ds_read_u8 v26, v6 offset:41728 // (op 914) Load local data 
s_barrier  // (op 3110) 



ds_read_b128 v[102:105], v3 // (op 70) Load local data 
ds_read_b128 v[106:109], v3 offset:32 // (op 70) Load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 3151) 



ds_read_u8 v27, v0 offset:41984 // (op 80) Load local data 
s_ashr_i64 s[10:11], s[36:37], 7 // (op 867) 
s_mov_b32 s7, s10 // (op 867) convert

s_and_b32 s10, s7, 4294967294 // (op 867) 

s_cmp_lt_u32 s6, s10 // (op 867) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ForLoopTop_KLoop_867 // (op 867) Condition: Bottom (jump to RR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ForLoopTop_KLoop_867 if true)
RR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ForLoopBottom_KLoop_867:
















s_barrier  // (op 3279) 
v_lshrrev_b32 v15, 3, v1 // (op 2409) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 2409) 
v_lshl_add_u32 v17, s16, 6, v15 // (op 2409) 

s_lshr_b32 s6, s0, 1 // (op 2409) 

v_and_b32 v15, -8, v1 // (op 2409) 
v_sub_u32 v20, v1, v15 // (op 2409) 

v_lshlrev_b32 v15, 4, v20 // (op 2409) 

v_lshl_add_u32 v20, s6, 8, v15 // (op 2409) 


v_mov_b32 v25, 0 // (op 2409) convert
v_mov_b32 v24, v20 // (op 2409) convert

v_mul_lo_u32 v21, v17, s13 // (op 2409) most significant: low of low * high
v_mul_hi_u32 v15, v17, s12 // (op 2409) most significant: high of low * low
v_mul_lo_u32 v20, v17, s12 // (op 2409) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 2409) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v24 // (op 2409) least significant half
v_addc_co_u32 v21, s[0:1], v21, v25, s[0:1] // (op 2409) most significant half



s_mul_i32 s1, s12, s23 // (op 2411) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 2411) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 2411) least significant
s_mul_hi_u32 s7, s12, s22 // (op 2411) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2411) most significant: sum
s_add_u32 s1, s1, s7 // (op 2411) most significant: sum


s_mov_b32 s48, s20 // (op 2411) reg expression
s_mov_b32 s49, s21 // (op 2411) reg expression
s_mov_b32 s50, s0 // (op 2411) reg expression
s_mov_b32 s51, 131072 // (op 2411) call()

v_and_b32 v15, -2, v1 // (op 2479) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 2479) 
v_sub_u32 v17, v1, v15 // (op 2479) 

s_lshr_b32 s6, s0, 1 // (op 2479) 

v_lshlrev_b32 v15, 6, v17 // (op 2479) 

v_lshl_add_u32 v17, s6, 8, v15 // (op 2479) 


v_mov_b32 v25, 0 // (op 2479) convert
v_mov_b32 v24, v17 // (op 2479) convert

v_and_b32 v15, 4294967294, v1 // (op 2479) 
v_lshl_add_u32 v17, s17, 8, v15 // (op 2479) 

v_mul_lo_u32 v29, v17, s25 // (op 2479) most significant: low of low * high
v_mul_hi_u32 v15, v17, s24 // (op 2479) most significant: high of low * low
v_mul_lo_u32 v28, v17, s24 // (op 2479) least significant: low of low * low
v_add_u32 v29, v29, v15 // (op 2479) most significant: sum


v_add_co_u32 v30, s[0:1], v24, v28 // (op 2479) least significant half
v_addc_co_u32 v31, s[0:1], v25, v29, s[0:1] // (op 2479) most significant half



v_mul_lo_u32 v25, v31, 6 // (op 2479) most significant: low of high * low
v_mul_hi_u32 v15, v30, 6 // (op 2479) most significant: high of low * low
v_mul_lo_u32 v24, v30, 6 // (op 2479) least significant: low of low * low
v_add_u32 v25, v25, v15 // (op 2479) most significant: sum


v_lshrrev_b64 v[28:29], 3, v[24:25] // (op 2479) 

v_mov_b32 v15, v28 // (op 2479) convert

s_mul_i32 s1, s9, 6 // (op 2481) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 2481) least significant
s_mul_hi_u32 s6, s8, 6 // (op 2481) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2481) most significant: sum

s_mul_i32 s7, s24, s1 // (op 2481) most significant: low of low * high
s_mul_i32 s10, s25, s0 // (op 2481) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 2481) least significant
s_mul_hi_u32 s11, s24, s0 // (op 2481) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 2481) most significant: sum
s_add_u32 s7, s7, s11 // (op 2481) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 2481) 

s_mov_b32 s52, s26 // (op 2481) reg expression
s_mov_b32 s53, s27 // (op 2481) reg expression
s_mov_b32 s54, s0 // (op 2481) reg expression
s_mov_b32 s55, 131072 // (op 2481) call()

v_lshrrev_b32 v17, 2, v1 // (op 2543) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 2543) 
v_lshl_add_u32 v22, s16, 6, v17 // (op 2543) 

s_lshr_b32 s6, s0, 1 // (op 2543) 

v_and_b32 v17, -4, v1 // (op 2543) 
v_sub_u32 v24, v1, v17 // (op 2543) 

v_lshl_add_u32 v17, s6, 3, v24 // (op 2543) 


v_mul_lo_u32 v24, v22, s28 // (op 2543) 
v_add_u32 v24, v24, v17 // (op 2543) 


s_mul_i32 s1, s28, s23 // (op 2545) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 2545) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 2545) least significant
s_mul_hi_u32 s7, s28, s22 // (op 2545) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2545) most significant: sum
s_add_u32 s1, s1, s7 // (op 2545) most significant: sum


s_mov_b32 s56, s30 // (op 2545) reg expression
s_mov_b32 s57, s31 // (op 2545) reg expression
s_mov_b32 s58, s0 // (op 2545) reg expression
s_mov_b32 s59, 131072 // (op 2545) call()

v_lshl_add_u32 v17, s17, 8, v1 // (op 2607) 
v_mul_lo_u32 v22, v17, s32 // (op 2607) 

s_ashr_i64 s[0:1], s[36:37], 7 // (op 2607) 
s_lshr_b32 s6, s0, 1 // (op 2607) 

v_lshl_add_u32 v17, s6, 3, v22 // (op 2607) 


s_mul_i32 s1, s32, s9 // (op 2609) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 2609) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 2609) least significant
s_mul_hi_u32 s7, s32, s8 // (op 2609) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2609) most significant: sum
s_add_u32 s1, s1, s7 // (op 2609) most significant: sum


s_mov_b32 s60, s34 // (op 2609) reg expression
s_mov_b32 s61, s35 // (op 2609) reg expression
s_mov_b32 s62, s0 // (op 2609) reg expression
s_mov_b32 s63, 131072 // (op 2609) call()

v_and_b32 v22, -8, v1 // (op 2700) 
v_sub_u32 v25, v1, v22 // (op 2700) 

v_lshlrev_b32 v22, 4, v25 // (op 2700) 

v_lshrrev_b32 v25, 3, v1 // (op 2700) 
v_lshl_add_u32 v28, v25, 7, v22 // (op 2700) 


s_nop 0
v_readfirstlane_b32 s0, v28 // (op 2700) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v22, -2, v1 // (op 2752) 
v_sub_u32 v25, v1, v22 // (op 2752) 

v_lshlrev_b32 v22, 6, v25 // (op 2752) 

v_lshrrev_b32 v25, 1, v1 // (op 2752) 
v_lshl_add_u32 v28, v25, 8, v22 // (op 2752) 


v_mul_lo_u32 v22, v28, 6 // (op 2752) 

v_lshrrev_b32 v25, 3, v22 // (op 2752) 

v_and_b32 v22, -4, v1 // (op 2810) 
v_sub_u32 v28, v1, v22 // (op 2810) 

v_and_b32 v22, 4294967292, v1 // (op 2810) 
v_add_u32 v29, v22, v28 // (op 2810) 


v_lshlrev_b32 v22, 2, v1 // (op 2863) 
s_ashr_i64 s[6:7], s[36:37], 7 // (op 873) 
s_and_b32 s1, s6, 4294967294 // (op 873) 

s_ashr_i64 s[6:7], s[36:37], 7 // (op 872) 
s_mov_b32 s10, s6 // (op 872) convert

s_cmp_lt_u32 s1, s10 // (op 872) 














s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ForLoopBottom_KLoopTail_872 // (op 872) Condition: Top (jump to RR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ForLoopBottom_KLoopTail_872 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ForLoopTop_KLoopTail_872:
s_barrier  // (op 3104) 
s_add_u32 s1, s1, 1 // (op 874) 

s_lshl_b64 s[6:7], s[12:13], 5 // (op 2380) 
v_mov_b64 v[18:19], s[6:7] // (op 2380) convert


v_mov_b64 v[26:27], v[20:21] // (op 2380) 
v_mov_b64 v[30:31], v[26:27] // (op 2380) 
s_mov_b32 m0, s0 // (op 2380) reg expression
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 2380) Load value direct to lds
v_add_co_u32 v30, s[6:7], v30, v18 // (op 2380) least significant half
v_addc_co_u32 v31, s[6:7], 0, v31, s[6:7] // (op 2380) most significant half

s_add_u32 m0, m0, 4096 // (op 2380) 
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 2380) Load value direct to lds



v_add_co_u32 v20, vcc, 128, v20 // (op 2399) least significant half
v_addc_co_u32 v21, vcc, 0, v21, vcc // (op 2399) most significant half

s_mul_i32 s7, s25, 6 // (op 221) most significant: low of high * low
s_mul_i32 s6, s24, 6 // (op 221) least significant
s_mul_hi_u32 s10, s24, 6 // (op 221) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 221) most significant: sum

v_lshrrev_b64 v[18:19], 3, s[6:7] // (op 221) 


v_mov_b32 v14, v15 // (op 221) 
v_mov_b32 v16, v14 // (op 221) 
buffer_load_dwordx4 v[30:33], v16, s[52:55], 0 offen // (op 221) Load value
buffer_load_dwordx2 v[34:35], v16, s[52:55], 0 offen offset:16 // (op 221) Load value
buffer_load_dwordx4 v[36:39], v16, s[52:55], 0 offen offset:24 // (op 221) Load value
buffer_load_dwordx2 v[40:41], v16, s[52:55], 0 offen offset:40 // (op 221) Load value
v_add_u32 v16, v16, v18 // (op 221) 
buffer_load_dwordx4 v[42:45], v16, s[52:55], 0 offen // (op 221) Load value
buffer_load_dwordx2 v[46:47], v16, s[52:55], 0 offen offset:16 // (op 221) Load value
buffer_load_dwordx4 v[48:51], v16, s[52:55], 0 offen offset:24 // (op 221) Load value
buffer_load_dwordx2 v[52:53], v16, s[52:55], 0 offen offset:40 // (op 221) Load value



v_add_u32 v15, 96, v15 // (op 2471) 


s_barrier  // (op 3123) 
v_mov_b32 v14, v24 // (op 222) 




buffer_load_ubyte v16, v14, s[56:59], 0 offen offset: 0 // (op 222) Load value

v_add_u32 v24, 4, v24 // (op 2535) 
s_waitcnt vmcnt(8)
ds_write_b128 v25, v[30:33] offset:16384 // (op 218) Store local data 


v_mov_b32 v14, v17 // (op 223) 

buffer_load_dword v23, v14, s[60:63], 0 offen // (op 223) Load value

v_add_u32 v17, 4, v17 // (op 2599) 
s_waitcnt vmcnt(8)
ds_write_b64 v25, v[34:35] offset:16400 // (op 218) Store local data 
s_barrier  // (op 3113) 
s_waitcnt vmcnt(7)
ds_write_b128 v25, v[36:39] offset:16408 // (op 218) Store local data 



ds_read_b128 v[54:57], v3 // (op 201) Load local data 
ds_read_b128 v[58:61], v3 offset:32 // (op 201) Load local data 
s_waitcnt vmcnt(6)
ds_write_b64 v25, v[40:41] offset:16424 // (op 218) Store local data 
s_waitcnt vmcnt(5)
ds_write_b128 v25, v[42:45] offset:16480 // (op 218) Store local data 
s_waitcnt vmcnt(4)
ds_write_b64 v25, v[46:47] offset:16496 // (op 218) Store local data 
s_waitcnt vmcnt(3)
ds_write_b128 v25, v[48:51] offset:16504 // (op 218) Store local data 
s_waitcnt vmcnt(2)
ds_write_b64 v25, v[52:53] offset:16520 // (op 218) Store local data 

s_barrier  // (op 3144) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 3134) 






ds_read_b128 v[30:33], v7 offset:16384 // (op 203) Load local data 
ds_read_b64 v[34:35], v7 offset:16400 // (op 203) Load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v29, v16 offset:41984 // (op 219) Store local data 

s_barrier  // (op 3164) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 3154) 






ds_read_u8 v14, v0 offset:41984 // (op 205) Load local data 
s_waitcnt vmcnt(0)
ds_write_b32 v22, v23 offset:40960 // (op 220) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 3174) 



ds_read_u8 v16, v6 offset:40960 // (op 207) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[54:61], v[30:35], a[0:15], v14, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 208) 





ds_read_b128 v[30:33], v3 offset:64 // (op 210) Load local data 
ds_read_b128 v[34:37], v3 offset:96 // (op 210) Load local data 



ds_read_b128 v[38:41], v7 offset:16432 // (op 212) Load local data 
ds_read_b64 v[42:43], v7 offset:16448 // (op 212) Load local data 



ds_read_u8 v16, v0 offset:41986 // (op 214) Load local data 



ds_read_u8 v23, v6 offset:40962 // (op 216) Load local data 



ds_read_b128 v[44:47], v7 offset:22528 // (op 412) Load local data 
ds_read_b64 v[48:49], v7 offset:22544 // (op 412) Load local data 



ds_read_u8 v18, v6 offset:41216 // (op 416) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[54:61], v[44:49], a[16:31], v14, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 417) 





ds_read_b128 v[44:47], v7 offset:22576 // (op 421) Load local data 
ds_read_b64 v[48:49], v7 offset:22592 // (op 421) Load local data 



ds_read_u8 v18, v6 offset:41218 // (op 425) Load local data 



ds_read_b128 v[62:65], v7 offset:28672 // (op 646) Load local data 
ds_read_b64 v[66:67], v7 offset:28688 // (op 646) Load local data 



ds_read_u8 v19, v6 offset:41472 // (op 650) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[54:61], v[62:67], a[32:47], v14, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 651) 





ds_read_b128 v[62:65], v7 offset:28720 // (op 655) Load local data 
ds_read_b64 v[66:67], v7 offset:28736 // (op 655) Load local data 



ds_read_u8 v19, v6 offset:41474 // (op 659) Load local data 



ds_read_b128 v[68:71], v7 offset:34816 // (op 880) Load local data 
ds_read_b64 v[72:73], v7 offset:34832 // (op 880) Load local data 



ds_read_u8 v26, v6 offset:41728 // (op 884) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[54:61], v[68:73], a[48:63], v14, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 885) 







ds_read_b128 v[50:53], v7 offset:34864 // (op 889) Load local data 
ds_read_b64 v[54:55], v7 offset:34880 // (op 889) Load local data 
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[30:37], v[38:43], a[0:15], v16, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 217) 





ds_read_u8 v14, v6 offset:41730 // (op 893) Load local data 
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[30:37], v[44:49], a[16:31], v16, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 426) 


v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[30:37], v[62:67], a[32:47], v16, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 660) 


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[30:37], v[50:55], a[48:63], v16, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 894) 




s_ashr_i64 s[6:7], s[36:37], 7 // (op 872) 
s_mov_b32 s10, s6 // (op 872) convert

s_cmp_lt_u32 s1, s10 // (op 872) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ForLoopTop_KLoopTail_872 // (op 872) Condition: Bottom (jump to RR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ForLoopTop_KLoopTail_872 if true)
RR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ForLoopBottom_KLoopTail_872:























s_barrier  // (op 3283) 









v_cmp_eq_f32 vcc, 0.00000, v12 // (op 155) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ConditionalFalse_0_beta_155 // (op 155) Condition: False, jump to RR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ConditionalFalse_0_beta_155
s_nop 0
v_accvgpr_read v14, a0 // (op 50) Wait state hazard: XDL Write Hazard
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



v_mov_b32 v0, v11 // (op 56) reg expression

buffer_store_dwordx4 v[30:33], v0, s[44:47], 0 offen // (op 56) Store value
buffer_store_dwordx4 v[34:37], v0, s[44:47], 0 offen offset:32 // (op 56) Store value
buffer_store_dwordx4 v[38:41], v0, s[44:47], 0 offen offset:64 // (op 56) Store value
buffer_store_dwordx4 v[42:45], v0, s[44:47], 0 offen offset:96 // (op 56) Store value


v_accvgpr_read v14, a16 // (op 1258) 
v_accvgpr_read v15, a17 // (op 1258) 
v_accvgpr_read v16, a18 // (op 1258) 
v_accvgpr_read v17, a19 // (op 1258) 
v_accvgpr_read v18, a20 // (op 1258) 
v_accvgpr_read v19, a21 // (op 1258) 
v_accvgpr_read v20, a22 // (op 1258) 
v_accvgpr_read v21, a23 // (op 1258) 
v_accvgpr_read v22, a24 // (op 1258) 
v_accvgpr_read v23, a25 // (op 1258) 
v_accvgpr_read v24, a26 // (op 1258) 
v_accvgpr_read v25, a27 // (op 1258) 
v_accvgpr_read v26, a28 // (op 1258) 
v_accvgpr_read v27, a29 // (op 1258) 
v_accvgpr_read v28, a30 // (op 1258) 
v_accvgpr_read v29, a31 // (op 1258) 
v_mul_f32 v30, v13, v14 // (op 1258) 
v_mul_f32 v31, v13, v15 // (op 1258) 
v_mul_f32 v32, v13, v16 // (op 1258) 
v_mul_f32 v33, v13, v17 // (op 1258) 
v_mul_f32 v34, v13, v18 // (op 1258) 
v_mul_f32 v35, v13, v19 // (op 1258) 
v_mul_f32 v36, v13, v20 // (op 1258) 
v_mul_f32 v37, v13, v21 // (op 1258) 
v_mul_f32 v38, v13, v22 // (op 1258) 
v_mul_f32 v39, v13, v23 // (op 1258) 
v_mul_f32 v40, v13, v24 // (op 1258) 
v_mul_f32 v41, v13, v25 // (op 1258) 
v_mul_f32 v42, v13, v26 // (op 1258) 
v_mul_f32 v43, v13, v27 // (op 1258) 
v_mul_f32 v44, v13, v28 // (op 1258) 
v_mul_f32 v45, v13, v29 // (op 1258) 



s_lshl_b32 s0, s4, 8 // (op 1260) 
v_add_u32 v0, v11, s0 // (op 1260) 


buffer_store_dwordx4 v[30:33], v0, s[44:47], 0 offen // (op 1260) Store value
buffer_store_dwordx4 v[34:37], v0, s[44:47], 0 offen offset:32 // (op 1260) Store value
buffer_store_dwordx4 v[38:41], v0, s[44:47], 0 offen offset:64 // (op 1260) Store value
buffer_store_dwordx4 v[42:45], v0, s[44:47], 0 offen offset:96 // (op 1260) Store value


v_accvgpr_read v14, a32 // (op 1270) 
v_accvgpr_read v15, a33 // (op 1270) 
v_accvgpr_read v16, a34 // (op 1270) 
v_accvgpr_read v17, a35 // (op 1270) 
v_accvgpr_read v18, a36 // (op 1270) 
v_accvgpr_read v19, a37 // (op 1270) 
v_accvgpr_read v20, a38 // (op 1270) 
v_accvgpr_read v21, a39 // (op 1270) 
v_accvgpr_read v22, a40 // (op 1270) 
v_accvgpr_read v23, a41 // (op 1270) 
v_accvgpr_read v24, a42 // (op 1270) 
v_accvgpr_read v25, a43 // (op 1270) 
v_accvgpr_read v26, a44 // (op 1270) 
v_accvgpr_read v27, a45 // (op 1270) 
v_accvgpr_read v28, a46 // (op 1270) 
v_accvgpr_read v29, a47 // (op 1270) 
v_mul_f32 v30, v13, v14 // (op 1270) 
v_mul_f32 v31, v13, v15 // (op 1270) 
v_mul_f32 v32, v13, v16 // (op 1270) 
v_mul_f32 v33, v13, v17 // (op 1270) 
v_mul_f32 v34, v13, v18 // (op 1270) 
v_mul_f32 v35, v13, v19 // (op 1270) 
v_mul_f32 v36, v13, v20 // (op 1270) 
v_mul_f32 v37, v13, v21 // (op 1270) 
v_mul_f32 v38, v13, v22 // (op 1270) 
v_mul_f32 v39, v13, v23 // (op 1270) 
v_mul_f32 v40, v13, v24 // (op 1270) 
v_mul_f32 v41, v13, v25 // (op 1270) 
v_mul_f32 v42, v13, v26 // (op 1270) 
v_mul_f32 v43, v13, v27 // (op 1270) 
v_mul_f32 v44, v13, v28 // (op 1270) 
v_mul_f32 v45, v13, v29 // (op 1270) 



s_lshl_b32 s0, s4, 9 // (op 1272) 
v_add_u32 v0, v11, s0 // (op 1272) 


buffer_store_dwordx4 v[30:33], v0, s[44:47], 0 offen // (op 1272) Store value
buffer_store_dwordx4 v[34:37], v0, s[44:47], 0 offen offset:32 // (op 1272) Store value
buffer_store_dwordx4 v[38:41], v0, s[44:47], 0 offen offset:64 // (op 1272) Store value
buffer_store_dwordx4 v[42:45], v0, s[44:47], 0 offen offset:96 // (op 1272) Store value


v_accvgpr_read v14, a48 // (op 1282) 
v_accvgpr_read v15, a49 // (op 1282) 
v_accvgpr_read v16, a50 // (op 1282) 
v_accvgpr_read v17, a51 // (op 1282) 
v_accvgpr_read v18, a52 // (op 1282) 
v_accvgpr_read v19, a53 // (op 1282) 
v_accvgpr_read v20, a54 // (op 1282) 
v_accvgpr_read v21, a55 // (op 1282) 
v_accvgpr_read v22, a56 // (op 1282) 
v_accvgpr_read v23, a57 // (op 1282) 
v_accvgpr_read v24, a58 // (op 1282) 
v_accvgpr_read v25, a59 // (op 1282) 
v_accvgpr_read v26, a60 // (op 1282) 
v_accvgpr_read v27, a61 // (op 1282) 
v_accvgpr_read v28, a62 // (op 1282) 
v_accvgpr_read v29, a63 // (op 1282) 
v_mul_f32 v30, v13, v14 // (op 1282) 
v_mul_f32 v31, v13, v15 // (op 1282) 
v_mul_f32 v32, v13, v16 // (op 1282) 
v_mul_f32 v33, v13, v17 // (op 1282) 
v_mul_f32 v34, v13, v18 // (op 1282) 
v_mul_f32 v35, v13, v19 // (op 1282) 
v_mul_f32 v36, v13, v20 // (op 1282) 
v_mul_f32 v37, v13, v21 // (op 1282) 
v_mul_f32 v38, v13, v22 // (op 1282) 
v_mul_f32 v39, v13, v23 // (op 1282) 
v_mul_f32 v40, v13, v24 // (op 1282) 
v_mul_f32 v41, v13, v25 // (op 1282) 
v_mul_f32 v42, v13, v26 // (op 1282) 
v_mul_f32 v43, v13, v27 // (op 1282) 
v_mul_f32 v44, v13, v28 // (op 1282) 
v_mul_f32 v45, v13, v29 // (op 1282) 



s_lshl_b32 s0, s4, 8 // (op 1284) 
s_mul_i32 s1, 3, s0 // (op 1284) 

v_add_u32 v0, v11, s1 // (op 1284) 


buffer_store_dwordx4 v[30:33], v0, s[44:47], 0 offen // (op 1284) Store value
buffer_store_dwordx4 v[34:37], v0, s[44:47], 0 offen offset:32 // (op 1284) Store value
buffer_store_dwordx4 v[38:41], v0, s[44:47], 0 offen offset:64 // (op 1284) Store value
buffer_store_dwordx4 v[42:45], v0, s[44:47], 0 offen offset:96 // (op 1284) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ConditionalBottom_0_beta_155 // (op 155) Condition: Done, jump to RR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ConditionalBottom_0_beta_155
RR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ConditionalFalse_0_beta_155:


v_mov_b32 v0, v8 // (op 167) reg expression

buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 167) Load value
buffer_load_dwordx4 v[18:21], v0, s[40:43], 0 offen offset:32 // (op 167) Load value
buffer_load_dwordx4 v[22:25], v0, s[40:43], 0 offen offset:64 // (op 167) Load value
v_accvgpr_read v30, a0 // (op 164) 
v_accvgpr_read v31, a1 // (op 164) 
v_accvgpr_read v32, a2 // (op 164) 
v_accvgpr_read v33, a3 // (op 164) 
v_accvgpr_read v34, a4 // (op 164) 
v_accvgpr_read v35, a5 // (op 164) 
v_accvgpr_read v36, a6 // (op 164) 
v_accvgpr_read v37, a7 // (op 164) 
v_accvgpr_read v38, a8 // (op 164) 
v_accvgpr_read v39, a9 // (op 164) 
v_accvgpr_read v40, a10 // (op 164) 
v_accvgpr_read v41, a11 // (op 164) 
v_accvgpr_read v42, a12 // (op 164) 
v_accvgpr_read v43, a13 // (op 164) 
v_accvgpr_read v44, a14 // (op 164) 
v_accvgpr_read v45, a15 // (op 164) 
v_mul_f32 v46, v13, v30 // (op 164) 
v_mul_f32 v47, v13, v31 // (op 164) 
v_mul_f32 v48, v13, v32 // (op 164) 
v_mul_f32 v49, v13, v33 // (op 164) 
v_mul_f32 v50, v13, v34 // (op 164) 
v_mul_f32 v51, v13, v35 // (op 164) 
v_mul_f32 v52, v13, v36 // (op 164) 
v_mul_f32 v53, v13, v37 // (op 164) 
v_mul_f32 v54, v13, v38 // (op 164) 
v_mul_f32 v55, v13, v39 // (op 164) 
v_mul_f32 v56, v13, v40 // (op 164) 
v_mul_f32 v57, v13, v41 // (op 164) 
v_mul_f32 v58, v13, v42 // (op 164) 
v_mul_f32 v59, v13, v43 // (op 164) 
v_mul_f32 v60, v13, v44 // (op 164) 
v_mul_f32 v61, v13, v45 // (op 164) 

buffer_load_dwordx4 v[26:29], v0, s[40:43], 0 offen offset:96 // (op 167) Load value

s_waitcnt vmcnt(3)
v_mul_f32 v30, v12, v14 // (op 168) 
v_mul_f32 v31, v12, v15 // (op 168) 
v_mul_f32 v32, v12, v16 // (op 168) 
v_mul_f32 v33, v12, v17 // (op 168) 
s_waitcnt vmcnt(2)
v_mul_f32 v34, v12, v18 // (op 168) 
v_mul_f32 v35, v12, v19 // (op 168) 
v_mul_f32 v36, v12, v20 // (op 168) 
v_mul_f32 v37, v12, v21 // (op 168) 
s_waitcnt vmcnt(1)
v_mul_f32 v38, v12, v22 // (op 168) 
v_mul_f32 v39, v12, v23 // (op 168) 
v_mul_f32 v40, v12, v24 // (op 168) 
v_mul_f32 v41, v12, v25 // (op 168) 
s_waitcnt vmcnt(0)
v_mul_f32 v42, v12, v26 // (op 168) 
v_mul_f32 v43, v12, v27 // (op 168) 
v_mul_f32 v44, v12, v28 // (op 168) 
v_mul_f32 v45, v12, v29 // (op 168) 

v_add_f32 v14, v30, v46 // (op 165) 
v_add_f32 v15, v31, v47 // (op 165) 
v_add_f32 v16, v32, v48 // (op 165) 
v_add_f32 v17, v33, v49 // (op 165) 
v_add_f32 v18, v34, v50 // (op 165) 
v_add_f32 v19, v35, v51 // (op 165) 
v_add_f32 v20, v36, v52 // (op 165) 
v_add_f32 v21, v37, v53 // (op 165) 
v_add_f32 v22, v38, v54 // (op 165) 
v_add_f32 v23, v39, v55 // (op 165) 
v_add_f32 v24, v40, v56 // (op 165) 
v_add_f32 v25, v41, v57 // (op 165) 
v_add_f32 v26, v42, v58 // (op 165) 
v_add_f32 v27, v43, v59 // (op 165) 
v_add_f32 v28, v44, v60 // (op 165) 
v_add_f32 v29, v45, v61 // (op 165) 




v_mov_b32 v0, v11 // (op 166) reg expression

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 166) Store value
buffer_store_dwordx4 v[18:21], v0, s[44:47], 0 offen offset:32 // (op 166) Store value
buffer_store_dwordx4 v[22:25], v0, s[44:47], 0 offen offset:64 // (op 166) Store value
buffer_store_dwordx4 v[26:29], v0, s[44:47], 0 offen offset:96 // (op 166) Store value




s_lshl_b32 s0, s2, 8 // (op 1316) 
v_add_u32 v0, v8, s0 // (op 1316) 


buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 1316) Load value
buffer_load_dwordx4 v[18:21], v0, s[40:43], 0 offen offset:32 // (op 1316) Load value
v_accvgpr_read v30, a16 // (op 1313) 
v_accvgpr_read v31, a17 // (op 1313) 
v_accvgpr_read v32, a18 // (op 1313) 
v_accvgpr_read v33, a19 // (op 1313) 
v_accvgpr_read v34, a20 // (op 1313) 
v_accvgpr_read v35, a21 // (op 1313) 
v_accvgpr_read v36, a22 // (op 1313) 
v_accvgpr_read v37, a23 // (op 1313) 
v_accvgpr_read v38, a24 // (op 1313) 
v_accvgpr_read v39, a25 // (op 1313) 
v_accvgpr_read v40, a26 // (op 1313) 
v_accvgpr_read v41, a27 // (op 1313) 
v_accvgpr_read v42, a28 // (op 1313) 
v_accvgpr_read v43, a29 // (op 1313) 
v_accvgpr_read v44, a30 // (op 1313) 
v_accvgpr_read v45, a31 // (op 1313) 
v_mul_f32 v46, v13, v30 // (op 1313) 
v_mul_f32 v47, v13, v31 // (op 1313) 
v_mul_f32 v48, v13, v32 // (op 1313) 
v_mul_f32 v49, v13, v33 // (op 1313) 
v_mul_f32 v50, v13, v34 // (op 1313) 
v_mul_f32 v51, v13, v35 // (op 1313) 
v_mul_f32 v52, v13, v36 // (op 1313) 
v_mul_f32 v53, v13, v37 // (op 1313) 
v_mul_f32 v54, v13, v38 // (op 1313) 
v_mul_f32 v55, v13, v39 // (op 1313) 
v_mul_f32 v56, v13, v40 // (op 1313) 
v_mul_f32 v57, v13, v41 // (op 1313) 
v_mul_f32 v58, v13, v42 // (op 1313) 
v_mul_f32 v59, v13, v43 // (op 1313) 
v_mul_f32 v60, v13, v44 // (op 1313) 
v_mul_f32 v61, v13, v45 // (op 1313) 

buffer_load_dwordx4 v[22:25], v0, s[40:43], 0 offen offset:64 // (op 1316) Load value
buffer_load_dwordx4 v[26:29], v0, s[40:43], 0 offen offset:96 // (op 1316) Load value

s_waitcnt vmcnt(3)
v_mul_f32 v30, v12, v14 // (op 1317) 
v_mul_f32 v31, v12, v15 // (op 1317) 
v_mul_f32 v32, v12, v16 // (op 1317) 
v_mul_f32 v33, v12, v17 // (op 1317) 
s_waitcnt vmcnt(2)
v_mul_f32 v34, v12, v18 // (op 1317) 
v_mul_f32 v35, v12, v19 // (op 1317) 
v_mul_f32 v36, v12, v20 // (op 1317) 
v_mul_f32 v37, v12, v21 // (op 1317) 
s_waitcnt vmcnt(1)
v_mul_f32 v38, v12, v22 // (op 1317) 
v_mul_f32 v39, v12, v23 // (op 1317) 
v_mul_f32 v40, v12, v24 // (op 1317) 
v_mul_f32 v41, v12, v25 // (op 1317) 
s_waitcnt vmcnt(0)
v_mul_f32 v42, v12, v26 // (op 1317) 
v_mul_f32 v43, v12, v27 // (op 1317) 
v_mul_f32 v44, v12, v28 // (op 1317) 
v_mul_f32 v45, v12, v29 // (op 1317) 

v_add_f32 v14, v30, v46 // (op 1314) 
v_add_f32 v15, v31, v47 // (op 1314) 
v_add_f32 v16, v32, v48 // (op 1314) 
v_add_f32 v17, v33, v49 // (op 1314) 
v_add_f32 v18, v34, v50 // (op 1314) 
v_add_f32 v19, v35, v51 // (op 1314) 
v_add_f32 v20, v36, v52 // (op 1314) 
v_add_f32 v21, v37, v53 // (op 1314) 
v_add_f32 v22, v38, v54 // (op 1314) 
v_add_f32 v23, v39, v55 // (op 1314) 
v_add_f32 v24, v40, v56 // (op 1314) 
v_add_f32 v25, v41, v57 // (op 1314) 
v_add_f32 v26, v42, v58 // (op 1314) 
v_add_f32 v27, v43, v59 // (op 1314) 
v_add_f32 v28, v44, v60 // (op 1314) 
v_add_f32 v29, v45, v61 // (op 1314) 




s_lshl_b32 s0, s4, 8 // (op 1315) 
v_add_u32 v0, v11, s0 // (op 1315) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 1315) Store value
buffer_store_dwordx4 v[18:21], v0, s[44:47], 0 offen offset:32 // (op 1315) Store value
buffer_store_dwordx4 v[22:25], v0, s[44:47], 0 offen offset:64 // (op 1315) Store value
buffer_store_dwordx4 v[26:29], v0, s[44:47], 0 offen offset:96 // (op 1315) Store value




s_lshl_b32 s0, s2, 9 // (op 1328) 
v_add_u32 v0, v8, s0 // (op 1328) 


buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 1328) Load value
buffer_load_dwordx4 v[18:21], v0, s[40:43], 0 offen offset:32 // (op 1328) Load value
v_accvgpr_read v30, a32 // (op 1325) 
v_accvgpr_read v31, a33 // (op 1325) 
v_accvgpr_read v32, a34 // (op 1325) 
v_accvgpr_read v33, a35 // (op 1325) 
v_accvgpr_read v34, a36 // (op 1325) 
v_accvgpr_read v35, a37 // (op 1325) 
v_accvgpr_read v36, a38 // (op 1325) 
v_accvgpr_read v37, a39 // (op 1325) 
v_accvgpr_read v38, a40 // (op 1325) 
v_accvgpr_read v39, a41 // (op 1325) 
v_accvgpr_read v40, a42 // (op 1325) 
v_accvgpr_read v41, a43 // (op 1325) 
v_accvgpr_read v42, a44 // (op 1325) 
v_accvgpr_read v43, a45 // (op 1325) 
v_accvgpr_read v44, a46 // (op 1325) 
v_accvgpr_read v45, a47 // (op 1325) 
v_mul_f32 v46, v13, v30 // (op 1325) 
v_mul_f32 v47, v13, v31 // (op 1325) 
v_mul_f32 v48, v13, v32 // (op 1325) 
v_mul_f32 v49, v13, v33 // (op 1325) 
v_mul_f32 v50, v13, v34 // (op 1325) 
v_mul_f32 v51, v13, v35 // (op 1325) 
v_mul_f32 v52, v13, v36 // (op 1325) 
v_mul_f32 v53, v13, v37 // (op 1325) 
v_mul_f32 v54, v13, v38 // (op 1325) 
v_mul_f32 v55, v13, v39 // (op 1325) 
v_mul_f32 v56, v13, v40 // (op 1325) 
v_mul_f32 v57, v13, v41 // (op 1325) 
v_mul_f32 v58, v13, v42 // (op 1325) 
v_mul_f32 v59, v13, v43 // (op 1325) 
v_mul_f32 v60, v13, v44 // (op 1325) 
v_mul_f32 v61, v13, v45 // (op 1325) 

buffer_load_dwordx4 v[22:25], v0, s[40:43], 0 offen offset:64 // (op 1328) Load value
buffer_load_dwordx4 v[26:29], v0, s[40:43], 0 offen offset:96 // (op 1328) Load value

s_waitcnt vmcnt(3)
v_mul_f32 v30, v12, v14 // (op 1329) 
v_mul_f32 v31, v12, v15 // (op 1329) 
v_mul_f32 v32, v12, v16 // (op 1329) 
v_mul_f32 v33, v12, v17 // (op 1329) 
s_waitcnt vmcnt(2)
v_mul_f32 v34, v12, v18 // (op 1329) 
v_mul_f32 v35, v12, v19 // (op 1329) 
v_mul_f32 v36, v12, v20 // (op 1329) 
v_mul_f32 v37, v12, v21 // (op 1329) 
s_waitcnt vmcnt(1)
v_mul_f32 v38, v12, v22 // (op 1329) 
v_mul_f32 v39, v12, v23 // (op 1329) 
v_mul_f32 v40, v12, v24 // (op 1329) 
v_mul_f32 v41, v12, v25 // (op 1329) 
s_waitcnt vmcnt(0)
v_mul_f32 v42, v12, v26 // (op 1329) 
v_mul_f32 v43, v12, v27 // (op 1329) 
v_mul_f32 v44, v12, v28 // (op 1329) 
v_mul_f32 v45, v12, v29 // (op 1329) 

v_add_f32 v14, v30, v46 // (op 1326) 
v_add_f32 v15, v31, v47 // (op 1326) 
v_add_f32 v16, v32, v48 // (op 1326) 
v_add_f32 v17, v33, v49 // (op 1326) 
v_add_f32 v18, v34, v50 // (op 1326) 
v_add_f32 v19, v35, v51 // (op 1326) 
v_add_f32 v20, v36, v52 // (op 1326) 
v_add_f32 v21, v37, v53 // (op 1326) 
v_add_f32 v22, v38, v54 // (op 1326) 
v_add_f32 v23, v39, v55 // (op 1326) 
v_add_f32 v24, v40, v56 // (op 1326) 
v_add_f32 v25, v41, v57 // (op 1326) 
v_add_f32 v26, v42, v58 // (op 1326) 
v_add_f32 v27, v43, v59 // (op 1326) 
v_add_f32 v28, v44, v60 // (op 1326) 
v_add_f32 v29, v45, v61 // (op 1326) 




s_lshl_b32 s0, s4, 9 // (op 1327) 
v_add_u32 v0, v11, s0 // (op 1327) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 1327) Store value
buffer_store_dwordx4 v[18:21], v0, s[44:47], 0 offen offset:32 // (op 1327) Store value
buffer_store_dwordx4 v[22:25], v0, s[44:47], 0 offen offset:64 // (op 1327) Store value
buffer_store_dwordx4 v[26:29], v0, s[44:47], 0 offen offset:96 // (op 1327) Store value




s_lshl_b32 s0, s2, 8 // (op 1340) 
s_mul_i32 s1, 3, s0 // (op 1340) 

v_add_u32 v0, v8, s1 // (op 1340) 


buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 1340) Load value
buffer_load_dwordx4 v[18:21], v0, s[40:43], 0 offen offset:32 // (op 1340) Load value
v_accvgpr_read v30, a48 // (op 1337) 
v_accvgpr_read v31, a49 // (op 1337) 
v_accvgpr_read v32, a50 // (op 1337) 
v_accvgpr_read v33, a51 // (op 1337) 
v_accvgpr_read v34, a52 // (op 1337) 
v_accvgpr_read v35, a53 // (op 1337) 
v_accvgpr_read v36, a54 // (op 1337) 
v_accvgpr_read v37, a55 // (op 1337) 
v_accvgpr_read v38, a56 // (op 1337) 
v_accvgpr_read v39, a57 // (op 1337) 
v_accvgpr_read v40, a58 // (op 1337) 
v_accvgpr_read v41, a59 // (op 1337) 
v_accvgpr_read v42, a60 // (op 1337) 
v_accvgpr_read v43, a61 // (op 1337) 
v_accvgpr_read v44, a62 // (op 1337) 
v_accvgpr_read v45, a63 // (op 1337) 
v_mul_f32 v46, v13, v30 // (op 1337) 
v_mul_f32 v47, v13, v31 // (op 1337) 
v_mul_f32 v48, v13, v32 // (op 1337) 
v_mul_f32 v49, v13, v33 // (op 1337) 
v_mul_f32 v50, v13, v34 // (op 1337) 
v_mul_f32 v51, v13, v35 // (op 1337) 
v_mul_f32 v52, v13, v36 // (op 1337) 
v_mul_f32 v53, v13, v37 // (op 1337) 
v_mul_f32 v54, v13, v38 // (op 1337) 
v_mul_f32 v55, v13, v39 // (op 1337) 
v_mul_f32 v56, v13, v40 // (op 1337) 
v_mul_f32 v57, v13, v41 // (op 1337) 
v_mul_f32 v58, v13, v42 // (op 1337) 
v_mul_f32 v59, v13, v43 // (op 1337) 
v_mul_f32 v60, v13, v44 // (op 1337) 
v_mul_f32 v61, v13, v45 // (op 1337) 

buffer_load_dwordx4 v[22:25], v0, s[40:43], 0 offen offset:64 // (op 1340) Load value
buffer_load_dwordx4 v[26:29], v0, s[40:43], 0 offen offset:96 // (op 1340) Load value

s_waitcnt vmcnt(3)
v_mul_f32 v30, v12, v14 // (op 1341) 
v_mul_f32 v31, v12, v15 // (op 1341) 
v_mul_f32 v32, v12, v16 // (op 1341) 
v_mul_f32 v33, v12, v17 // (op 1341) 
s_waitcnt vmcnt(2)
v_mul_f32 v34, v12, v18 // (op 1341) 
v_mul_f32 v35, v12, v19 // (op 1341) 
v_mul_f32 v36, v12, v20 // (op 1341) 
v_mul_f32 v37, v12, v21 // (op 1341) 
s_waitcnt vmcnt(1)
v_mul_f32 v38, v12, v22 // (op 1341) 
v_mul_f32 v39, v12, v23 // (op 1341) 
v_mul_f32 v40, v12, v24 // (op 1341) 
v_mul_f32 v41, v12, v25 // (op 1341) 
s_waitcnt vmcnt(0)
v_mul_f32 v42, v12, v26 // (op 1341) 
v_mul_f32 v43, v12, v27 // (op 1341) 
v_mul_f32 v44, v12, v28 // (op 1341) 
v_mul_f32 v45, v12, v29 // (op 1341) 

v_add_f32 v14, v30, v46 // (op 1338) 
v_add_f32 v15, v31, v47 // (op 1338) 
v_add_f32 v16, v32, v48 // (op 1338) 
v_add_f32 v17, v33, v49 // (op 1338) 
v_add_f32 v18, v34, v50 // (op 1338) 
v_add_f32 v19, v35, v51 // (op 1338) 
v_add_f32 v20, v36, v52 // (op 1338) 
v_add_f32 v21, v37, v53 // (op 1338) 
v_add_f32 v22, v38, v54 // (op 1338) 
v_add_f32 v23, v39, v55 // (op 1338) 
v_add_f32 v24, v40, v56 // (op 1338) 
v_add_f32 v25, v41, v57 // (op 1338) 
v_add_f32 v26, v42, v58 // (op 1338) 
v_add_f32 v27, v43, v59 // (op 1338) 
v_add_f32 v28, v44, v60 // (op 1338) 
v_add_f32 v29, v45, v61 // (op 1338) 




s_lshl_b32 s0, s4, 8 // (op 1339) 
s_mul_i32 s1, 3, s0 // (op 1339) 

v_add_u32 v0, v11, s1 // (op 1339) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 1339) Store value
buffer_store_dwordx4 v[18:21], v0, s[44:47], 0 offen offset:32 // (op 1339) Store value
buffer_store_dwordx4 v[22:25], v0, s[44:47], 0 offen offset:64 // (op 1339) Store value
buffer_store_dwordx4 v[26:29], v0, s[44:47], 0 offen offset:96 // (op 1339) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ConditionalBottom_0_beta_155:












s_endpgm  // End of RR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2
.LRR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_end:
.size RR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2, .LRR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_end-RR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2
  .amdhsa_next_free_vgpr 216
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 68096 // lds bytes
  .amdhsa_accum_offset 152
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
    DataTypeB: F6
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
    macrotile: [64, 256, 128]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [32, 32, 64, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 4]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2
    .symbol: RR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 68096
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 64
    .vgpr_count: 150
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

