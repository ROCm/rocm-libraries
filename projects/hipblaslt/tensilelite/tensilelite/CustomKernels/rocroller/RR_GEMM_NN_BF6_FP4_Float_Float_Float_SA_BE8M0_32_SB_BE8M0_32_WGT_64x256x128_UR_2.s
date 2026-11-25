// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2
.p2align 8
.type RR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2,@function
RR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_exec_begin:


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


v_and_b32 v0, 63, v1 // (op 2705) 
v_and_b32 v3, 48, v1 // (op 2705) 
v_sub_u32 v4, v0, v3 // (op 2705) 

v_lshrrev_b32 v3, 5, v0 // (op 2705) 
v_lshl_add_u32 v5, v3, 5, v4 // (op 2705) 


v_lshrrev_b32 v3, 4, v0 // (op 2705) 

v_and_b32 v0, -2, v3 // (op 2705) 
v_sub_u32 v4, v3, v0 // (op 2705) 


v_lshlrev_b32 v0, 4, v4 // (op 2705) 

v_lshrrev_b32 v3, 7, v1 // (op 2705) 
v_lshl_add_u32 v4, v3, 5, v0 // (op 2705) 


v_lshl_add_u32 v0, v5, 6, v4 // (op 2705) 


v_mul_lo_u32 v3, v0, 6 // (op 2705) 
v_lshrrev_b32 v4, 3, v3 // (op 2705) 

v_lshrrev_b32 v3, 4, v0 // (op 2705) 

v_lshlrev_b32 v0, 2, v3 // (op 2705) 

v_add_u32 v3, v4, v0 // (op 2705) 


v_lshrrev_b32 v0, 7, v1 // (op 2816) 
v_and_b32 v4, 31, v1 // (op 2816) 
v_lshl_add_u32 v5, v0, 5, v4 // (op 2816) 


v_and_b32 v0, 63, v1 // (op 2816) 
v_lshrrev_b32 v4, 5, v0 // (op 2816) 

v_lshl_add_u32 v0, v4, 6, v5 // (op 2816) 


v_and_b32 v4, 63, v1 // (op 2916) 
v_and_b32 v5, 48, v1 // (op 2916) 
v_sub_u32 v6, v4, v5 // (op 2916) 

v_lshrrev_b32 v5, 5, v4 // (op 2916) 
v_lshl_add_u32 v7, v5, 5, v6 // (op 2916) 


v_lshrrev_b32 v5, 4, v4 // (op 2916) 

v_and_b32 v4, -2, v5 // (op 2916) 
v_sub_u32 v6, v5, v4 // (op 2916) 


v_lshlrev_b32 v4, 4, v6 // (op 2916) 

v_lshrrev_b32 v5, 7, v1 // (op 2916) 
v_lshl_add_u32 v6, v5, 5, v4 // (op 2916) 


v_lshl_add_u32 v4, v7, 6, v6 // (op 2916) 


v_mul_lo_u32 v5, v4, 6 // (op 2916) 
v_lshrrev_b32 v6, 3, v5 // (op 2916) 

v_lshrrev_b32 v5, 4, v4 // (op 2916) 

v_lshlrev_b32 v4, 2, v5 // (op 2916) 

v_add_u32 v5, v6, v4 // (op 2916) 


v_lshrrev_b32 v4, 7, v1 // (op 2969) 
v_and_b32 v6, 31, v1 // (op 2969) 
v_lshl_add_u32 v7, v4, 5, v6 // (op 2969) 


v_and_b32 v4, 63, v1 // (op 2969) 
v_lshrrev_b32 v6, 5, v4 // (op 2969) 

v_lshl_add_u32 v4, v6, 6, v7 // (op 2969) 


v_lshrrev_b32 v6, 7, v1 // (op 2648) 
v_and_b32 v7, 63, v1 // (op 2648) 
v_lshrrev_b32 v8, 5, v7 // (op 2648) 
v_lshlrev_b32 v9, 2, v8 // (op 2648) 

v_lshl_add_u32 v8, v6, 5, v9 // (op 2648) 


v_lshl_add_u32 v6, s16, 6, v8 // (op 2648) 

v_lshrrev_b32 v8, 2, v7 // (op 2648) 

v_and_b32 v7, 7, v8 // (op 2648) 

v_and_b32 v8, 3, v1 // (op 2648) 
v_lshl_add_u32 v9, v7, 2, v8 // (op 2648) 


v_lshrrev_b32 v7, 6, v1 // (op 2648) 
v_and_b32 v8, 1, v7 // (op 2648) 

v_lshl_add_u32 v7, v8, 5, v9 // (op 2648) 


v_lshl_add_u32 v8, s17, 8, v7 // (op 2648) 

v_mul_lo_u32 v7, v8, s2 // (op 2648) 

v_add_lshl_u32 v8, v6, v7, 2 // (op 2648) 


v_and_b32 v6, 63, v1 // (op 2763) 
v_lshrrev_b32 v7, 4, v6 // (op 2763) 

v_and_b32 v6, 1, v7 // (op 2763) 

v_and_b32 v7, 15, v1 // (op 2763) 
v_lshl_add_u32 v9, v6, 4, v7 // (op 2763) 


v_lshrrev_b32 v6, 6, v1 // (op 2763) 
v_and_b32 v7, 1, v6 // (op 2763) 

v_lshl_add_u32 v6, v7, 5, v9 // (op 2763) 


v_and_b32 v7, 32, v1 // (op 2763) 
v_lshl_add_u32 v9, v6, 7, v7 // (op 2763) 


v_lshlrev_b32 v6, 2, v9 // (op 2763) 

v_lshrrev_b32 v7, 3, v6 // (op 2763) 

v_lshrrev_b32 v6, 6, v1 // (op 2874) 
v_and_b32 v9, 1, v6 // (op 2874) 

v_and_b32 v6, 31, v1 // (op 2874) 
s_mul_i32 s1, s2, s9 // (op 2650) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 2650) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 2650) least significant
s_mul_hi_u32 s19, s2, s8 // (op 2650) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2650) most significant: sum
s_add_u32 s1, s1, s19 // (op 2650) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 2650) 

v_lshl_add_u32 v10, v9, 5, v6 // (op 2874) 


v_and_b32 v6, 63, v1 // (op 2874) 
v_lshrrev_b32 v9, 5, v6 // (op 2874) 

v_lshl_add_u32 v6, v10, 2, v9 // (op 2874) 


v_lshrrev_b32 v9, 7, v1 // (op 2890) 
v_and_b32 v10, 63, v1 // (op 2890) 
v_lshrrev_b32 v11, 5, v10 // (op 2890) 
v_lshlrev_b32 v12, 2, v11 // (op 2890) 

v_lshl_add_u32 v11, v9, 5, v12 // (op 2890) 


v_lshl_add_u32 v9, s16, 6, v11 // (op 2890) 

v_lshrrev_b32 v11, 2, v10 // (op 2890) 

v_and_b32 v10, 7, v11 // (op 2890) 

v_and_b32 v11, 3, v1 // (op 2890) 
v_lshl_add_u32 v12, v10, 2, v11 // (op 2890) 


v_lshrrev_b32 v10, 6, v1 // (op 2890) 
v_and_b32 v11, 1, v10 // (op 2890) 

v_lshl_add_u32 v10, v11, 5, v12 // (op 2890) 


v_lshl_add_u32 v11, s17, 8, v10 // (op 2890) 

v_mul_lo_u32 v10, v11, s4 // (op 2890) 

v_add_lshl_u32 v11, v9, v10, 2 // (op 2890) 


v_and_b32 v9, 63, v1 // (op 2945) 
v_lshrrev_b32 v10, 4, v9 // (op 2945) 

v_and_b32 v9, 1, v10 // (op 2945) 

v_and_b32 v10, 15, v1 // (op 2945) 
v_lshl_add_u32 v12, v9, 4, v10 // (op 2945) 


v_lshrrev_b32 v9, 6, v1 // (op 2945) 
v_and_b32 v10, 1, v9 // (op 2945) 

v_lshl_add_u32 v9, v10, 5, v12 // (op 2945) 


v_and_b32 v10, 32, v1 // (op 2945) 
v_lshl_add_u32 v12, v9, 7, v10 // (op 2945) 


v_lshlrev_b32 v9, 2, v12 // (op 2945) 

v_lshrrev_b32 v10, 3, v9 // (op 2945) 

v_lshrrev_b32 v9, 6, v1 // (op 2998) 
v_and_b32 v12, 1, v9 // (op 2998) 

v_and_b32 v9, 31, v1 // (op 2998) 
s_mul_i32 s1, s4, s9 // (op 2892) most significant: low of low * high
v_lshl_add_u32 v13, v12, 5, v9 // (op 2998) 


v_and_b32 v9, 63, v1 // (op 2998) 
v_lshrrev_b32 v12, 5, v9 // (op 2998) 

v_lshl_add_u32 v9, v13, 2, v12 // (op 2998) 


s_mul_i32 s38, s5, s8 // (op 2892) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 2892) least significant
s_mul_hi_u32 s39, s4, s8 // (op 2892) most significant: high of low * low
s_add_u32 s1, s1, s38 // (op 2892) most significant: sum
s_add_u32 s1, s1, s39 // (op 2892) most significant: sum


s_lshl_b64 s[38:39], s[0:1], 2 // (op 2892) 

s_mov_b32 s40, s6 // (op 2650) reg expression
s_mov_b32 s41, s7 // (op 2650) reg expression
s_mov_b32 s42, s18 // (op 2650) reg expression
s_mov_b32 s43, 131072 // (op 2650) call()

s_mov_b32 s44, s10 // (op 2892) reg expression

s_mov_b32 s45, s11 // (op 2892) reg expression
s_mov_b32 s46, s38 // (op 2892) reg expression
s_mov_b32 s47, 131072 // (op 2892) call()


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
v_and_b32 v14, -2, v1 // (op 2415) 
v_sub_u32 v15, v1, v14 // (op 2415) 

v_lshlrev_b32 v14, 5, v15 // (op 2415) 

v_lshl_add_u32 v15, s16, 6, v14 // (op 2415) 

v_mov_b32 v17, 0 // (op 2415) convert
v_mov_b32 v16, v15 // (op 2415) convert

v_lshrrev_b32 v14, 1, v1 // (op 2415) 
v_mul_lo_u32 v19, v14, s13 // (op 2415) most significant: low of low * high
v_mul_hi_u32 v15, v14, s12 // (op 2415) most significant: high of low * low
v_mul_lo_u32 v18, v14, s12 // (op 2415) least significant: low of low * low
v_add_u32 v19, v19, v15 // (op 2415) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v18 // (op 2415) least significant half
v_addc_co_u32 v15, s[0:1], v17, v19, s[0:1] // (op 2415) most significant half



v_mul_lo_u32 v17, v15, 6 // (op 2415) most significant: low of high * low
v_mul_hi_u32 v18, v14, 6 // (op 2415) most significant: high of low * low
v_mul_lo_u32 v16, v14, 6 // (op 2415) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 2415) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 2415) 

v_mov_b32 v16, v14 // (op 2415) convert


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s23, 6 // (op 2417) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 2417) least significant
s_mul_hi_u32 s6, s22, 6 // (op 2417) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2417) most significant: sum

s_mul_i32 s7, s12, s1 // (op 2417) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 2417) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 2417) least significant
s_mul_hi_u32 s11, s12, s0 // (op 2417) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 2417) most significant: sum
s_add_u32 s7, s7, s11 // (op 2417) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 2417) 

s_mov_b32 s48, s20 // (op 2417) reg expression
s_mov_b32 s49, s21 // (op 2417) reg expression
s_mov_b32 s50, s0 // (op 2417) reg expression
s_mov_b32 s51, 131072 // (op 2417) call()

v_and_b32 v14, -2, v1 // (op 2434) 
v_sub_u32 v15, v1, v14 // (op 2434) 

v_lshlrev_b32 v14, 5, v15 // (op 2434) 

v_lshl_add_u32 v15, s16, 6, v14 // (op 2434) 

v_mov_b32 v19, 0 // (op 2434) convert
v_mov_b32 v18, v15 // (op 2434) convert

v_lshrrev_b32 v17, 1, v1 // (op 2434) 
v_mul_lo_u32 v15, v17, s13 // (op 2434) most significant: low of low * high
v_mul_hi_u32 v20, v17, s12 // (op 2434) most significant: high of low * low
v_mul_lo_u32 v14, v17, s12 // (op 2434) least significant: low of low * low
v_add_u32 v15, v15, v20 // (op 2434) most significant: sum


v_add_co_u32 v20, s[0:1], v18, v14 // (op 2434) least significant half
v_addc_co_u32 v21, s[0:1], v19, v15, s[0:1] // (op 2434) most significant half



v_mul_lo_u32 v15, v21, 6 // (op 2434) most significant: low of high * low
v_mul_hi_u32 v17, v20, 6 // (op 2434) most significant: high of low * low
v_mul_lo_u32 v14, v20, 6 // (op 2434) least significant: low of low * low
v_add_u32 v15, v15, v17 // (op 2434) most significant: sum


v_lshrrev_b64 v[18:19], 3, v[14:15] // (op 2434) 

v_mov_b32 v17, v18 // (op 2434) convert

s_mul_i32 s1, s23, 6 // (op 2436) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 2436) least significant
s_mul_hi_u32 s6, s22, 6 // (op 2436) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2436) most significant: sum

s_mul_i32 s7, s12, s1 // (op 2436) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 2436) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 2436) least significant
s_mul_hi_u32 s11, s12, s0 // (op 2436) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 2436) most significant: sum
s_add_u32 s7, s7, s11 // (op 2436) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 2436) 

s_mov_b32 s48, s20 // (op 2436) reg expression
s_mov_b32 s49, s21 // (op 2436) reg expression
s_mov_b32 s50, s0 // (op 2436) reg expression
s_mov_b32 s51, 131072 // (op 2436) call()

v_and_b32 v14, -4, v1 // (op 2479) 
v_sub_u32 v15, v1, v14 // (op 2479) 

v_lshlrev_b32 v14, 5, v15 // (op 2479) 

v_mov_b32 v19, 0 // (op 2479) convert
v_mov_b32 v18, v14 // (op 2479) convert

v_and_b32 v14, 4294967292, v1 // (op 2479) 
v_lshl_add_u32 v15, s17, 8, v14 // (op 2479) 

v_mul_lo_u32 v21, v15, s25 // (op 2479) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 2479) most significant: high of low * low
v_mul_lo_u32 v20, v15, s24 // (op 2479) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 2479) most significant: sum


v_add_co_u32 v14, s[0:1], v18, v20 // (op 2479) least significant half
v_addc_co_u32 v15, s[0:1], v19, v21, s[0:1] // (op 2479) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 2479) 


v_lshrrev_b64 v[18:19], 3, v[14:15] // (op 2479) 

v_mov_b32 v14, v18 // (op 2479) convert

s_mul_i32 s1, s24, s9 // (op 2481) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 2481) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 2481) least significant
s_mul_hi_u32 s7, s24, s8 // (op 2481) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2481) most significant: sum
s_add_u32 s1, s1, s7 // (op 2481) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 2481) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 2481) 

s_mov_b32 s52, s26 // (op 2481) reg expression
s_mov_b32 s53, s27 // (op 2481) reg expression
s_mov_b32 s54, s0 // (op 2481) reg expression
s_mov_b32 s55, 131072 // (op 2481) call()

v_and_b32 v15, -4, v1 // (op 2498) 
v_sub_u32 v18, v1, v15 // (op 2498) 

v_lshlrev_b32 v15, 5, v18 // (op 2498) 

v_mov_b32 v19, 0 // (op 2498) convert
v_mov_b32 v18, v15 // (op 2498) convert

v_and_b32 v15, 4294967292, v1 // (op 2498) 
v_lshl_add_u32 v20, s17, 8, v15 // (op 2498) 

v_mul_lo_u32 v23, v20, s25 // (op 2498) most significant: low of low * high
v_mul_hi_u32 v15, v20, s24 // (op 2498) most significant: high of low * low
v_mul_lo_u32 v22, v20, s24 // (op 2498) least significant: low of low * low
v_add_u32 v23, v23, v15 // (op 2498) most significant: sum


v_add_co_u32 v20, s[0:1], v18, v22 // (op 2498) least significant half
v_addc_co_u32 v21, s[0:1], v19, v23, s[0:1] // (op 2498) most significant half

v_lshlrev_b64 v[20:21], 2, v[20:21] // (op 2498) 


v_lshrrev_b64 v[18:19], 3, v[20:21] // (op 2498) 

v_mov_b32 v15, v18 // (op 2498) convert

s_mul_i32 s1, s24, s9 // (op 2500) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 2500) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 2500) least significant
s_mul_hi_u32 s7, s24, s8 // (op 2500) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2500) most significant: sum
s_add_u32 s1, s1, s7 // (op 2500) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 2500) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 2500) 

s_mov_b32 s52, s26 // (op 2500) reg expression
s_mov_b32 s53, s27 // (op 2500) reg expression
s_mov_b32 s54, s0 // (op 2500) reg expression
s_mov_b32 s55, 131072 // (op 2500) call()

v_and_b32 v18, -64, v1 // (op 2543) 
v_sub_u32 v19, v1, v18 // (op 2543) 

v_lshl_add_u32 v18, s16, 6, v19 // (op 2543) 

v_lshrrev_b32 v19, 6, v1 // (op 2543) 
v_mul_lo_u32 v20, v19, s28 // (op 2543) 

v_add_u32 v19, v18, v20 // (op 2543) 


s_mul_i32 s1, s28, s33 // (op 2545) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 2545) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 2545) least significant
s_mul_hi_u32 s7, s28, s32 // (op 2545) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2545) most significant: sum
s_add_u32 s1, s1, s7 // (op 2545) most significant: sum


s_mov_b32 s56, s30 // (op 2545) reg expression
s_mov_b32 s57, s31 // (op 2545) reg expression
s_mov_b32 s58, s0 // (op 2545) reg expression
s_mov_b32 s59, 131072 // (op 2545) call()

v_and_b32 v18, -64, v1 // (op 2562) 
v_sub_u32 v20, v1, v18 // (op 2562) 

v_lshl_add_u32 v18, s16, 6, v20 // (op 2562) 

v_lshrrev_b32 v20, 6, v1 // (op 2562) 
v_mul_lo_u32 v21, v20, s28 // (op 2562) 

v_add_u32 v20, v18, v21 // (op 2562) 


s_mul_i32 s1, s28, s33 // (op 2564) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 2564) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 2564) least significant
s_mul_hi_u32 s7, s28, s32 // (op 2564) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2564) most significant: sum
s_add_u32 s1, s1, s7 // (op 2564) most significant: sum


s_mov_b32 s56, s30 // (op 2564) reg expression
s_mov_b32 s57, s31 // (op 2564) reg expression
s_mov_b32 s58, s0 // (op 2564) reg expression
s_mov_b32 s59, 131072 // (op 2564) call()

v_lshl_add_u32 v18, s17, 8, v1 // (op 2607) 
v_mul_lo_u32 v21, v18, s34 // (op 2607) 

s_mul_i32 s1, s34, s9 // (op 2609) most significant: low of low * high
s_mul_i32 s6, s35, s8 // (op 2609) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 2609) least significant
s_mul_hi_u32 s7, s34, s8 // (op 2609) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2609) most significant: sum
s_add_u32 s1, s1, s7 // (op 2609) most significant: sum


s_mov_b32 s60, s36 // (op 2609) reg expression
s_mov_b32 s61, s37 // (op 2609) reg expression
s_mov_b32 s62, s0 // (op 2609) reg expression
s_mov_b32 s63, 131072 // (op 2609) call()

v_lshl_add_u32 v18, s17, 8, v1 // (op 2626) 
v_mul_lo_u32 v22, v18, s34 // (op 2626) 

s_mul_i32 s1, s34, s9 // (op 2628) most significant: low of low * high
s_mul_i32 s6, s35, s8 // (op 2628) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 2628) least significant
s_mul_hi_u32 s7, s34, s8 // (op 2628) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2628) most significant: sum
s_add_u32 s1, s1, s7 // (op 2628) most significant: sum


s_mov_b32 s60, s36 // (op 2628) reg expression
s_mov_b32 s61, s37 // (op 2628) reg expression
s_mov_b32 s62, s0 // (op 2628) reg expression
s_mov_b32 s63, 131072 // (op 2628) call()


s_lshl_b64 s[0:1], s[24:25], 2 // (op 2153) 
v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 2153) 


v_mov_b32 v18, v14 // (op 2153) 
buffer_load_dwordx4 v[26:29], v18, s[52:55], 0 offen // (op 2153) Load value
v_add_u32 v18, v18, v24 // (op 2153) 
buffer_load_dwordx4 v[30:33], v18, s[52:55], 0 offen // (op 2153) Load value
v_add_u32 v18, v18, v24 // (op 2153) 
buffer_load_dwordx4 v[34:37], v18, s[52:55], 0 offen // (op 2153) Load value
v_add_u32 v18, v18, v24 // (op 2153) 
buffer_load_dwordx4 v[38:41], v18, s[52:55], 0 offen // (op 2153) Load value





buffer_load_dword v18, v21, s[60:63], 0 offen // (op 2156) Load value


v_mov_b32 v23, v16 // (op 2159) reg expression

buffer_load_dwordx3 v[42:44], v23, s[48:51], 0 offen // (op 2159) Load value
buffer_load_dwordx3 v[46:48], v23, s[48:51], 0 offen offset:12 // (op 2159) Load value



v_mov_b32 v45, v19 // (op 2162) reg expression

buffer_load_ubyte v23, v45, s[56:59], 0 offen offset: 0 // (op 2162) Load value


s_lshl_b64 s[0:1], s[24:25], 2 // (op 2165) 
v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 2165) 


v_mov_b32 v45, v14 // (op 2165) 
buffer_load_dwordx4 v[50:53], v45, s[52:55], 0 offen offset:64 // (op 2165) Load value
v_add_u32 v45, v45, v24 // (op 2165) 
buffer_load_dwordx4 v[54:57], v45, s[52:55], 0 offen offset:64 // (op 2165) Load value
v_add_u32 v45, v45, v24 // (op 2165) 
buffer_load_dwordx4 v[58:61], v45, s[52:55], 0 offen offset:64 // (op 2165) Load value
v_add_u32 v45, v45, v24 // (op 2165) 
buffer_load_dwordx4 v[62:65], v45, s[52:55], 0 offen offset:64 // (op 2165) Load value






buffer_load_dword v14, v21, s[60:63], 0 offen offset:4 // (op 2168) Load value



s_lshl_b64 s[0:1], s[12:13], 7 // (op 2171) 
s_mul_i32 s7, s1, 6 // (op 2171) most significant: low of high * low
s_mul_i32 s6, s0, 6 // (op 2171) least significant
s_mul_hi_u32 s10, s0, 6 // (op 2171) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 2171) most significant: sum


s_lshr_b64 s[0:1], s[6:7], 3 // (op 2171) 

v_add_u32 v21, v16, s0 // (op 2171) 


buffer_load_dwordx3 v[66:68], v21, s[48:51], 0 offen // (op 2171) Load value
buffer_load_dwordx3 v[70:72], v21, s[48:51], 0 offen offset:12 // (op 2171) Load value




s_lshl_b32 s0, s28, 2 // (op 2174) 
v_add_u32 v16, v19, s0 // (op 2174) 


buffer_load_ubyte v21, v16, s[56:59], 0 offen offset: 0 // (op 2174) Load value


v_and_b32 v16, -4, v1 // (op 2746) 
v_sub_u32 v19, v1, v16 // (op 2746) 

v_lshlrev_b32 v16, 5, v19 // (op 2746) 

v_lshrrev_b32 v19, 2, v1 // (op 2746) 
v_lshl_add_u32 v45, v19, 9, v16 // (op 2746) 


v_lshlrev_b32 v16, 2, v45 // (op 2746) 

v_lshrrev_b32 v19, 3, v16 // (op 2746) 




s_waitcnt vmcnt(15)
ds_write_b128 v19, v[26:29] // (op 2176) Store local data 
s_waitcnt vmcnt(14)
ds_write_b128 v19, v[30:33] offset:64 // (op 2176) Store local data 
s_waitcnt vmcnt(13)
ds_write_b128 v19, v[34:37] offset:128 // (op 2176) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v19, v[38:41] offset:192 // (op 2176) Store local data 

v_lshlrev_b32 v16, 2, v1 // (op 2857) 



s_waitcnt vmcnt(11)
ds_write_b32 v16, v18 offset:16384 // (op 2177) Store local data 

v_and_b32 v16, -2, v1 // (op 2693) 
v_sub_u32 v19, v1, v16 // (op 2693) 

v_lshlrev_b32 v16, 5, v19 // (op 2693) 

v_lshrrev_b32 v19, 1, v1 // (op 2693) 
v_lshl_add_u32 v45, v19, 6, v16 // (op 2693) 


v_mul_lo_u32 v16, v45, 6 // (op 2693) 
v_lshrrev_b32 v19, 3, v16 // (op 2693) 

v_lshrrev_b32 v16, 4, v45 // (op 2693) 

v_lshlrev_b32 v45, 2, v16 // (op 2693) 

v_add_u32 v16, v19, v45 // (op 2693) 





s_waitcnt vmcnt(10)
ds_write_b96 v16, v[42:44] offset:17408 // (op 2178) Store local data 
s_waitcnt vmcnt(9)
ds_write_b96 v16, v[46:48] offset:17424 // (op 2178) Store local data 

v_and_b32 v16, -64, v1 // (op 2804) 
v_sub_u32 v19, v1, v16 // (op 2804) 

v_and_b32 v16, 4294967232, v1 // (op 2804) 
v_add_u32 v45, v16, v19 // (op 2804) 





s_waitcnt vmcnt(8)
ds_write_b8 v45, v23 offset:25600 // (op 2179) Store local data 

s_waitcnt lgkmcnt(4)
s_barrier  // (op 3080) 



ds_read_b128 v[74:77], v7 // (op 2197) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 3121) 



ds_read_u8 v16, v6 offset:16384 // (op 2220) Load local data 



ds_read_b128 v[78:81], v7 offset:4096 // (op 2202) Load local data 



ds_read_u8 v19, v6 offset:16640 // (op 2225) Load local data 



ds_read_b128 v[82:85], v7 offset:8192 // (op 2207) Load local data 



ds_read_u8 v45, v6 offset:16896 // (op 2230) Load local data 



ds_read_b128 v[86:89], v7 offset:12288 // (op 2212) Load local data 



ds_read_u8 v49, v6 offset:17152 // (op 2235) Load local data 
s_waitcnt lgkmcnt(9)
s_barrier  // (op 3061) 



ds_read_b96_tr_b6 v[90:92], v3 offset:17408 // (op 2193) Transpose load local data 
ds_read_b96_tr_b6 v[93:95], v3 offset:18432 // (op 2193) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 3101) 


ds_read_u8 v69, v0 offset:25600 // (op 2216) Load local data 
v_and_b32 v73, -2, v1 // (op 2664) 
v_sub_u32 v24, v1, v73 // (op 2664) 

v_lshlrev_b32 v25, 5, v24 // (op 2664) 

v_lshrrev_b32 v24, 1, v1 // (op 2664) 
v_lshl_add_u32 v73, v24, 6, v25 // (op 2664) 


v_mul_lo_u32 v24, v73, 6 // (op 2664) 
v_lshrrev_b32 v25, 3, v24 // (op 2664) 

v_lshrrev_b32 v24, 4, v73 // (op 2664) 

v_lshlrev_b32 v73, 2, v24 // (op 2664) 

v_add_u32 v24, v25, v73 // (op 2664) 


v_and_b32 v25, -4, v1 // (op 2717) 
v_sub_u32 v73, v1, v25 // (op 2717) 

v_lshlrev_b32 v25, 5, v73 // (op 2717) 

v_lshrrev_b32 v73, 2, v1 // (op 2717) 
v_lshl_add_u32 v96, v73, 9, v25 // (op 2717) 


v_lshlrev_b32 v25, 2, v96 // (op 2717) 

v_lshrrev_b32 v73, 3, v25 // (op 2717) 

v_and_b32 v25, -64, v1 // (op 2775) 
v_sub_u32 v96, v1, v25 // (op 2775) 

v_and_b32 v25, 4294967232, v1 // (op 2775) 
v_add_u32 v97, v25, v96 // (op 2775) 


v_lshlrev_b32 v25, 2, v1 // (op 2828) 
v_and_b32 v96, -2, v1 // (op 2904) 
v_sub_u32 v98, v1, v96 // (op 2904) 

v_lshlrev_b32 v96, 5, v98 // (op 2904) 

v_lshrrev_b32 v98, 1, v1 // (op 2904) 
v_lshl_add_u32 v99, v98, 6, v96 // (op 2904) 


v_mul_lo_u32 v96, v99, 6 // (op 2904) 
v_lshrrev_b32 v98, 3, v96 // (op 2904) 

v_lshrrev_b32 v96, 4, v99 // (op 2904) 

v_lshlrev_b32 v99, 2, v96 // (op 2904) 

v_add_u32 v96, v98, v99 // (op 2904) 


v_and_b32 v98, -4, v1 // (op 2928) 
v_sub_u32 v99, v1, v98 // (op 2928) 

v_lshlrev_b32 v98, 5, v99 // (op 2928) 

v_lshrrev_b32 v99, 2, v1 // (op 2928) 
v_lshl_add_u32 v100, v99, 9, v98 // (op 2928) 


v_lshlrev_b32 v98, 2, v100 // (op 2928) 

v_lshrrev_b32 v99, 3, v98 // (op 2928) 

v_and_b32 v98, -64, v1 // (op 2957) 
v_sub_u32 v100, v1, v98 // (op 2957) 

v_and_b32 v98, 4294967232, v1 // (op 2957) 
v_add_u32 v101, v98, v100 // (op 2957) 


v_lshlrev_b32 v98, 2, v1 // (op 2981) 
s_mov_b32 s0, 0 // (op 868) call()
s_ashr_i64 s[6:7], s[22:23], 7 // (op 867) 
s_mov_b32 s1, s6 // (op 867) convert

s_and_b32 s6, s1, 4294967294 // (op 867) 

s_cmp_lt_u32 s0, s6 // (op 867) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ForLoopBottom_KLoop_867 // (op 867) Condition: Top (jump to RR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ForLoopBottom_KLoop_867 if false)
RR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ForLoopTop_KLoop_867:

s_add_u32 s0, s0, 2 // (op 869) 
s_lshl_b64 s[6:7], s[24:25], 2 // (op 10) 
v_lshrrev_b64 v[102:103], 3, s[6:7] // (op 10) 


v_mov_b32 v100, v15 // (op 10) 
v_mov_b32 v104, v100 // (op 10) 
buffer_load_dwordx4 v[26:29], v104, s[52:55], 0 offen offset:128 // (op 10) Load value
v_add_u32 v104, v104, v102 // (op 10) 
buffer_load_dwordx4 v[30:33], v104, s[52:55], 0 offen offset:128 // (op 10) Load value
v_add_u32 v104, v104, v102 // (op 10) 
buffer_load_dwordx4 v[34:37], v104, s[52:55], 0 offen offset:128 // (op 10) Load value
v_add_u32 v104, v104, v102 // (op 10) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[90:95], v[74:77], a[0:15], v69, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 88) 
buffer_load_dwordx4 v[38:41], v104, s[52:55], 0 offen offset:128 // (op 10) Load value




s_barrier  // (op 3070) 
v_mov_b32 v102, v22 // (op 25) 




buffer_load_dword v18, v102, s[60:63], 0 offen offset:8 // (op 25) Load value
ds_read_b96_tr_b6 v[104:106], v3 offset:21504 // (op 91) Transpose load local data 


ds_read_b96_tr_b6 v[107:109], v3 offset:22528 // (op 91) Transpose load local data 
v_mov_b32 v103, v17 // (op 4) 
s_barrier  // (op 3090) 
s_lshl_b64 s[6:7], s[12:13], 7 // (op 4) 



ds_read_b128 v[74:77], v7 offset:32 // (op 95) Load local data 
s_mul_i32 s11, s7, 6 // (op 4) most significant: low of high * low
s_barrier  // (op 3111) 
s_mul_i32 s10, s6, 6 // (op 4) least significant


s_mul_hi_u32 s1, s6, 6 // (op 4) most significant: high of low * low
s_add_u32 s11, s11, s1 // (op 4) most significant: sum


s_lshr_b64 s[6:7], s[10:11], 3 // (op 4) 

s_lshl_b32 s1, s6, 1 // (op 4) 

v_add_u32 v110, v103, s1 // (op 4) 


buffer_load_dwordx3 v[42:44], v110, s[48:51], 0 offen // (op 4) Load value
ds_read_u8 v111, v0 offset:25728 // (op 99) Load local data 
buffer_load_dwordx3 v[46:48], v110, s[48:51], 0 offen offset:12 // (op 4) Load value



s_barrier  // (op 3131) 
v_mov_b32 v110, v20 // (op 17) 



ds_read_u8 v16, v6 offset:16386 // (op 103) Load local data 
s_lshl_b32 s1, s28, 3 // (op 17) 
v_add_u32 v112, v110, s1 // (op 17) 


buffer_load_ubyte v23, v112, s[56:59], 0 offen offset: 0 // (op 17) Load value

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[104:109], v[74:77], a[0:15], v111, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 105) 
s_barrier  // (op 3151) 



v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[90:95], v[78:81], a[16:31], v69, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 447) 
s_waitcnt vmcnt(15)
ds_write_b128 v99, v[50:53] offset:25856 // (op 302) Store local data 



ds_read_b128 v[74:77], v7 offset:4128 // (op 451) Load local data 
s_waitcnt vmcnt(14)
ds_write_b128 v99, v[54:57] offset:25920 // (op 302) Store local data 



ds_read_u8 v16, v6 offset:16642 // (op 455) Load local data 
s_waitcnt vmcnt(13)
ds_write_b128 v99, v[58:61] offset:25984 // (op 302) Store local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[104:109], v[74:77], a[16:31], v111, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 456) 
s_waitcnt vmcnt(12)
ds_write_b128 v99, v[62:65] offset:26048 // (op 302) Store local data 
s_barrier  // (op 3163) 
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[90:95], v[82:85], a[32:47], v69, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 681) 






ds_read_b128 v[74:77], v7 offset:8224 // (op 685) Load local data 
s_waitcnt vmcnt(11)
ds_write_b32 v98, v14 offset:42240 // (op 304) Store local data 



ds_read_u8 v16, v6 offset:16898 // (op 689) Load local data 
s_barrier  // (op 3144) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[104:109], v[74:77], a[32:47], v111, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 690) 



s_waitcnt vmcnt(10)
ds_write_b96 v96, v[66:68] offset:43264 // (op 283) Store local data 
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[90:95], v[86:89], a[48:63], v69, v49 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 915) 
s_waitcnt vmcnt(9)
ds_write_b96 v96, v[70:72] offset:43280 // (op 283) Store local data 



ds_read_b128 v[86:89], v7 offset:12320 // (op 919) Load local data 



s_barrier  // (op 3157) 
ds_read_u8 v49, v6 offset:17154 // (op 923) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[104:109], v[86:89], a[48:63], v111, v49 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 924) 
s_waitcnt vmcnt(8)
ds_write_b8 v101, v21 offset:51456 // (op 303) Store local data 


s_barrier  // (op 3147) 



ds_read_b128 v[74:77], v10 offset:25856 // (op 287) Load local data 
s_barrier  // (op 3160) 



ds_read_u8 v16, v9 offset:42240 // (op 291) Load local data 



ds_read_b128 v[78:81], v10 offset:29952 // (op 470) Load local data 



ds_read_u8 v19, v9 offset:42496 // (op 474) Load local data 



ds_read_b128 v[82:85], v10 offset:34048 // (op 704) Load local data 



ds_read_u8 v45, v9 offset:42752 // (op 708) Load local data 



ds_read_b128 v[104:107], v10 offset:38144 // (op 938) Load local data 



ds_read_u8 v108, v9 offset:43008 // (op 942) Load local data 
s_barrier  // (op 3141) 



ds_read_b96_tr_b6 v[90:92], v5 offset:43264 // (op 285) Transpose load local data 
ds_read_b96_tr_b6 v[93:95], v5 offset:44288 // (op 285) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 3154) 


ds_read_u8 v69, v4 offset:51456 // (op 289) Load local data 

s_lshl_b64 s[6:7], s[24:25], 2 // (op 305) 
v_lshrrev_b64 v[112:113], 3, s[6:7] // (op 305) 


v_mov_b32 v109, v100 // (op 305) 
buffer_load_dwordx4 v[50:53], v109, s[52:55], 0 offen offset:192 // (op 305) Load value
v_add_u32 v109, v109, v112 // (op 305) 
buffer_load_dwordx4 v[54:57], v109, s[52:55], 0 offen offset:192 // (op 305) Load value
v_add_u32 v109, v109, v112 // (op 305) 
buffer_load_dwordx4 v[58:61], v109, s[52:55], 0 offen offset:192 // (op 305) Load value
v_add_u32 v109, v109, v112 // (op 305) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[90:95], v[74:77], a[0:15], v69, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 292) 
buffer_load_dwordx4 v[62:65], v109, s[52:55], 0 offen offset:192 // (op 305) Load value









v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[90:95], v[78:81], a[16:31], v69, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 457) 
buffer_load_dword v14, v102, s[60:63], 0 offen offset:12 // (op 307) Load value



v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[90:95], v[82:85], a[32:47], v69, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 691) 
v_add_u32 v15, 128, v15 // (op 2490) 
v_add_u32 v22, 8, v22 // (op 2618) 
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[90:95], v[104:107], a[48:63], v69, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 925) 
s_lshl_b64 s[6:7], s[12:13], 7 // (op 282) 


s_mul_i32 s11, s7, 6 // (op 282) most significant: low of high * low
s_mul_i32 s10, s6, 6 // (op 282) least significant
s_mul_hi_u32 s1, s6, 6 // (op 282) most significant: high of low * low
s_add_u32 s11, s11, s1 // (op 282) most significant: sum


s_lshr_b64 s[6:7], s[10:11], 3 // (op 282) 

s_mul_i32 s1, 3, s6 // (op 282) 

v_add_u32 v100, v103, s1 // (op 282) 


buffer_load_dwordx3 v[66:68], v100, s[48:51], 0 offen // (op 282) Load value
ds_read_b96_tr_b6 v[104:106], v5 offset:47360 // (op 294) Transpose load local data 
ds_read_b96_tr_b6 v[107:109], v5 offset:48384 // (op 294) Transpose load local data 
buffer_load_dwordx3 v[70:72], v100, s[48:51], 0 offen offset:12 // (op 282) Load value







s_barrier  // (op 3208) 
s_lshl_b32 s1, s28, 2 // (op 306) 
s_mul_i32 s6, 3, s1 // (op 306) 

v_add_u32 v100, v110, s6 // (op 306) 


buffer_load_ubyte v21, v100, s[56:59], 0 offen offset: 0 // (op 306) Load value

s_lshl_b64 s[6:7], s[12:13], 7 // (op 2426) 

s_mul_i32 s11, s7, 6 // (op 2426) most significant: low of high * low
s_barrier  // (op 3083) 
s_mul_i32 s10, s6, 6 // (op 2426) least significant



s_mul_hi_u32 s1, s6, 6 // (op 2426) most significant: high of low * low
s_add_u32 s11, s11, s1 // (op 2426) most significant: sum


s_lshl_b32 s1, s28, 3 // (op 2554) 
v_add_u32 v20, v20, s1 // (op 2554) 

s_lshr_b64 s[6:7], s[10:11], 3 // (op 2426) 

s_lshl_b32 s1, s6, 1 // (op 2426) 

v_add_u32 v17, v17, s1 // (op 2426) 

ds_read_b128 v[110:113], v10 offset:25888 // (op 296) Load local data 
s_waitcnt vmcnt(15)
ds_write_b128 v73, v[26:29] // (op 13) Store local data 


ds_read_u8 v100, v4 offset:51584 // (op 298) Load local data 
s_waitcnt vmcnt(14)
ds_write_b128 v73, v[30:33] offset:64 // (op 13) Store local data 
s_waitcnt vmcnt(13)
ds_write_b128 v73, v[34:37] offset:128 // (op 13) Store local data 



s_barrier  // (op 3212) 
ds_read_u8 v102, v9 offset:42242 // (op 300) Load local data 
s_waitcnt vmcnt(12)
ds_write_b128 v73, v[38:41] offset:192 // (op 13) Store local data 



ds_read_b128 v[114:117], v10 offset:29984 // (op 461) Load local data 



s_barrier  // (op 3124) 
ds_read_u8 v103, v9 offset:42498 // (op 465) Load local data 






s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[104:109], v[110:113], a[0:15], v100, v102 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 301) 
ds_read_b128 v[118:121], v10 offset:34080 // (op 695) Load local data 





ds_read_u8 v102, v9 offset:42754 // (op 699) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[104:109], v[114:117], a[16:31], v100, v103 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 466) 





ds_read_b128 v[110:113], v10 offset:38176 // (op 929) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[104:109], v[118:121], a[32:47], v100, v102 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 700) 





s_barrier  // (op 3268) 
ds_read_u8 v102, v9 offset:43010 // (op 933) Load local data 
s_waitcnt vmcnt(11)
ds_write_b32 v25, v18 offset:16384 // (op 30) Store local data 
s_barrier  // (op 3272) 
s_barrier  // (op 3064) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[104:109], v[110:113], a[48:63], v100, v102 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 934) 







s_waitcnt vmcnt(10)
ds_write_b96 v24, v[42:44] offset:17408 // (op 6) Store local data 
s_waitcnt vmcnt(9)
ds_write_b96 v24, v[46:48] offset:17424 // (op 6) Store local data 
s_barrier  // (op 3104) 



s_waitcnt vmcnt(8)
ds_write_b8 v97, v23 offset:25600 // (op 21) Store local data 
s_barrier  // (op 3093) 



ds_read_b128 v[74:77], v7 // (op 75) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 3134) 



ds_read_u8 v16, v6 offset:16384 // (op 85) Load local data 



ds_read_b128 v[78:81], v7 offset:4096 // (op 442) Load local data 



ds_read_u8 v19, v6 offset:16640 // (op 446) Load local data 



ds_read_b128 v[82:85], v7 offset:8192 // (op 676) Load local data 



ds_read_u8 v45, v6 offset:16896 // (op 680) Load local data 



ds_read_b128 v[86:89], v7 offset:12288 // (op 910) Load local data 



ds_read_u8 v49, v6 offset:17152 // (op 914) Load local data 
s_waitcnt lgkmcnt(9)
s_barrier  // (op 3073) 



ds_read_b96_tr_b6 v[90:92], v3 offset:17408 // (op 70) Transpose load local data 
ds_read_b96_tr_b6 v[93:95], v3 offset:18432 // (op 70) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 3114) 


ds_read_u8 v69, v0 offset:25600 // (op 80) Load local data 
s_ashr_i64 s[6:7], s[22:23], 7 // (op 867) 
s_mov_b32 s1, s6 // (op 867) convert

s_and_b32 s6, s1, 4294967294 // (op 867) 

s_cmp_lt_u32 s0, s6 // (op 867) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ForLoopTop_KLoop_867 // (op 867) Condition: Bottom (jump to RR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ForLoopTop_KLoop_867 if true)
RR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ForLoopBottom_KLoop_867:

















v_and_b32 v15, -2, v1 // (op 2386) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 2386) 
v_sub_u32 v20, v1, v15 // (op 2386) 

s_lshr_b32 s6, s0, 1 // (op 2386) 

v_lshlrev_b32 v15, 5, v20 // (op 2386) 

v_lshl_add_u32 v20, s16, 6, v15 // (op 2386) 

v_mov_b32 v23, 0 // (op 2386) convert
v_mov_b32 v22, v20 // (op 2386) convert

v_lshrrev_b32 v15, 1, v1 // (op 2386) 
v_lshl_add_u32 v20, s6, 8, v15 // (op 2386) 


v_mul_lo_u32 v25, v20, s13 // (op 2386) most significant: low of low * high
v_mul_hi_u32 v15, v20, s12 // (op 2386) most significant: high of low * low
v_mul_lo_u32 v24, v20, s12 // (op 2386) least significant: low of low * low
v_add_u32 v25, v25, v15 // (op 2386) most significant: sum


v_add_co_u32 v26, s[0:1], v22, v24 // (op 2386) least significant half
v_addc_co_u32 v27, s[0:1], v23, v25, s[0:1] // (op 2386) most significant half



v_mul_lo_u32 v23, v27, 6 // (op 2386) most significant: low of high * low
v_mul_hi_u32 v15, v26, 6 // (op 2386) most significant: high of low * low
v_mul_lo_u32 v22, v26, 6 // (op 2386) least significant: low of low * low
v_add_u32 v23, v23, v15 // (op 2386) most significant: sum


v_lshrrev_b64 v[24:25], 3, v[22:23] // (op 2386) 

v_mov_b32 v15, v24 // (op 2386) convert

s_mul_i32 s1, s23, 6 // (op 2388) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 2388) least significant
s_mul_hi_u32 s6, s22, 6 // (op 2388) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2388) most significant: sum

s_mul_i32 s7, s12, s1 // (op 2388) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 2388) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 2388) least significant
s_mul_hi_u32 s11, s12, s0 // (op 2388) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 2388) most significant: sum
s_add_u32 s7, s7, s11 // (op 2388) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 2388) 

s_mov_b32 s48, s20 // (op 2388) reg expression
s_mov_b32 s49, s21 // (op 2388) reg expression
s_mov_b32 s50, s0 // (op 2388) reg expression
s_mov_b32 s51, 131072 // (op 2388) call()

v_and_b32 v20, -4, v1 // (op 2456) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 2456) 
v_sub_u32 v73, v1, v20 // (op 2456) 

s_lshr_b32 s6, s0, 1 // (op 2456) 

v_lshlrev_b32 v20, 5, v73 // (op 2456) 

v_lshl_add_u32 v73, s6, 8, v20 // (op 2456) 


v_mov_b32 v23, 0 // (op 2456) convert
v_mov_b32 v22, v73 // (op 2456) convert

v_and_b32 v20, 4294967292, v1 // (op 2456) 
v_lshl_add_u32 v73, s17, 8, v20 // (op 2456) 

v_mul_lo_u32 v25, v73, s25 // (op 2456) most significant: low of low * high
v_mul_hi_u32 v20, v73, s24 // (op 2456) most significant: high of low * low
v_mul_lo_u32 v24, v73, s24 // (op 2456) least significant: low of low * low
v_add_u32 v25, v25, v20 // (op 2456) most significant: sum


v_add_co_u32 v26, s[0:1], v22, v24 // (op 2456) least significant half
v_addc_co_u32 v27, s[0:1], v23, v25, s[0:1] // (op 2456) most significant half

v_lshlrev_b64 v[26:27], 2, v[26:27] // (op 2456) 


v_lshrrev_b64 v[22:23], 3, v[26:27] // (op 2456) 

v_mov_b32 v20, v22 // (op 2456) convert

s_mul_i32 s1, s24, s9 // (op 2458) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 2458) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 2458) least significant
s_mul_hi_u32 s7, s24, s8 // (op 2458) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2458) most significant: sum
s_add_u32 s1, s1, s7 // (op 2458) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 2458) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 2458) 

s_mov_b32 s52, s26 // (op 2458) reg expression
s_mov_b32 s53, s27 // (op 2458) reg expression
s_mov_b32 s54, s0 // (op 2458) reg expression
s_mov_b32 s55, 131072 // (op 2458) call()

v_and_b32 v73, -64, v1 // (op 2520) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 2520) 
v_sub_u32 v17, v1, v73 // (op 2520) 

s_lshr_b32 s6, s0, 1 // (op 2520) 

v_lshl_add_u32 v18, s16, 6, v17 // (op 2520) 

v_lshrrev_b32 v17, 6, v1 // (op 2520) 
v_lshl_add_u32 v73, s6, 3, v17 // (op 2520) 


v_mul_lo_u32 v17, v73, s28 // (op 2520) 

v_add_u32 v73, v18, v17 // (op 2520) 


s_mul_i32 s1, s28, s33 // (op 2522) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 2522) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 2522) least significant
s_mul_hi_u32 s7, s28, s32 // (op 2522) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2522) most significant: sum
s_add_u32 s1, s1, s7 // (op 2522) most significant: sum


s_mov_b32 s56, s30 // (op 2522) reg expression
s_mov_b32 s57, s31 // (op 2522) reg expression
s_mov_b32 s58, s0 // (op 2522) reg expression
s_mov_b32 s59, 131072 // (op 2522) call()

v_lshl_add_u32 v17, s17, 8, v1 // (op 2584) 
v_mul_lo_u32 v18, v17, s34 // (op 2584) 

s_ashr_i64 s[0:1], s[22:23], 7 // (op 2584) 
s_lshr_b32 s6, s0, 1 // (op 2584) 

v_lshl_add_u32 v17, s6, 3, v18 // (op 2584) 


s_mul_i32 s1, s34, s9 // (op 2586) most significant: low of low * high
s_mul_i32 s6, s35, s8 // (op 2586) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 2586) least significant
s_mul_hi_u32 s7, s34, s8 // (op 2586) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2586) most significant: sum
s_add_u32 s1, s1, s7 // (op 2586) most significant: sum


s_mov_b32 s60, s36 // (op 2586) reg expression
s_mov_b32 s61, s37 // (op 2586) reg expression
s_mov_b32 s62, s0 // (op 2586) reg expression
s_mov_b32 s63, 131072 // (op 2586) call()

v_and_b32 v18, -2, v1 // (op 2677) 
v_sub_u32 v22, v1, v18 // (op 2677) 

v_lshlrev_b32 v18, 5, v22 // (op 2677) 

v_lshrrev_b32 v22, 1, v1 // (op 2677) 
v_lshl_add_u32 v23, v22, 6, v18 // (op 2677) 


v_mul_lo_u32 v18, v23, 6 // (op 2677) 
v_lshrrev_b32 v22, 3, v18 // (op 2677) 

v_lshrrev_b32 v18, 4, v23 // (op 2677) 

v_lshlrev_b32 v23, 2, v18 // (op 2677) 

v_add_u32 v18, v22, v23 // (op 2677) 


v_and_b32 v22, -4, v1 // (op 2730) 
v_sub_u32 v23, v1, v22 // (op 2730) 

v_lshlrev_b32 v22, 5, v23 // (op 2730) 

v_lshrrev_b32 v23, 2, v1 // (op 2730) 
v_lshl_add_u32 v24, v23, 9, v22 // (op 2730) 


v_lshlrev_b32 v22, 2, v24 // (op 2730) 

v_lshrrev_b32 v23, 3, v22 // (op 2730) 

v_and_b32 v22, -64, v1 // (op 2788) 
v_sub_u32 v24, v1, v22 // (op 2788) 

v_and_b32 v22, 4294967232, v1 // (op 2788) 
v_add_u32 v25, v22, v24 // (op 2788) 


v_lshlrev_b32 v22, 2, v1 // (op 2841) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 873) 
s_and_b32 s6, s0, 4294967294 // (op 873) 

s_ashr_i64 s[0:1], s[22:23], 7 // (op 872) 
s_mov_b32 s7, s0 // (op 872) convert

s_cmp_lt_u32 s6, s7 // (op 872) 















s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ForLoopBottom_KLoopTail_872 // (op 872) Condition: Top (jump to RR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ForLoopBottom_KLoopTail_872 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ForLoopTop_KLoopTail_872:


s_add_u32 s6, s6, 1 // (op 874) 
v_mov_b32 v14, v15 // (op 198) 
v_mov_b32 v16, v14 // (op 198) reg expression

buffer_load_dwordx3 v[26:28], v16, s[48:51], 0 offen // (op 198) Load value
buffer_load_dwordx3 v[30:32], v16, s[48:51], 0 offen offset:12 // (op 198) Load value


s_lshl_b64 s[0:1], s[12:13], 7 // (op 2376) 
s_mul_i32 s11, s1, 6 // (op 2376) most significant: low of high * low
s_mul_i32 s10, s0, 6 // (op 2376) least significant
s_mul_hi_u32 s7, s0, 6 // (op 2376) most significant: high of low * low
s_add_u32 s11, s11, s7 // (op 2376) most significant: sum


s_lshr_b64 s[0:1], s[10:11], 3 // (op 2376) 

v_add_u32 v15, v15, s0 // (op 2376) 


s_barrier  // (op 3067) 
s_lshl_b64 s[0:1], s[24:25], 2 // (op 221) 



v_lshrrev_b64 v[34:35], 3, s[0:1] // (op 221) 


v_mov_b32 v14, v20 // (op 221) 
v_mov_b32 v16, v14 // (op 221) 
buffer_load_dwordx4 v[36:39], v16, s[52:55], 0 offen // (op 221) Load value
v_add_u32 v16, v16, v34 // (op 221) 
s_waitcnt vmcnt(2)
ds_write_b96 v18, v[26:28] offset:17408 // (op 199) Store local data 
buffer_load_dwordx4 v[40:43], v16, s[52:55], 0 offen // (op 221) Load value
v_add_u32 v16, v16, v34 // (op 221) 
buffer_load_dwordx4 v[44:47], v16, s[52:55], 0 offen // (op 221) Load value
v_add_u32 v16, v16, v34 // (op 221) 
buffer_load_dwordx4 v[48:51], v16, s[52:55], 0 offen // (op 221) Load value



v_add_u32 v20, 64, v20 // (op 2448) 
s_waitcnt vmcnt(4)
ds_write_b96 v18, v[30:32] offset:17424 // (op 199) Store local data 



v_mov_b32 v14, v73 // (op 222) 
s_barrier  // (op 3086) 
v_mov_b32 v16, v14 // (op 222) reg expression




s_waitcnt vmcnt(3)
ds_write_b128 v23, v[36:39] // (op 218) Store local data 
s_waitcnt vmcnt(2)
ds_write_b128 v23, v[40:43] offset:64 // (op 218) Store local data 
buffer_load_ubyte v19, v16, s[56:59], 0 offen offset: 0 // (op 222) Load value


s_lshl_b32 s7, s28, 2 // (op 2512) 
v_add_u32 v73, v73, s7 // (op 2512) 

s_waitcnt vmcnt(2)
ds_write_b128 v23, v[44:47] offset:128 // (op 218) Store local data 


v_mov_b32 v14, v17 // (op 223) 

buffer_load_dword v16, v14, s[60:63], 0 offen // (op 223) Load value

v_add_u32 v17, 4, v17 // (op 2576) 
s_waitcnt vmcnt(2)
ds_write_b128 v23, v[48:51] offset:192 // (op 218) Store local data 

s_waitcnt lgkmcnt(4)
s_barrier  // (op 3076) 



s_barrier  // (op 3107) 
ds_read_b96_tr_b6 v[26:28], v3 offset:17408 // (op 201) Transpose load local data 



ds_read_b96_tr_b6 v[29:31], v3 offset:18432 // (op 201) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v25, v19 offset:25600 // (op 219) Store local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 3097) 



s_barrier  // (op 3127) 
ds_read_b128 v[32:35], v7 // (op 203) Load local data 



s_waitcnt lgkmcnt(1)
s_barrier  // (op 3117) 
s_waitcnt vmcnt(0)
ds_write_b32 v22, v16 offset:16384 // (op 220) Store local data 



ds_read_u8 v14, v0 offset:25600 // (op 205) Load local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 3137) 



ds_read_u8 v16, v6 offset:16384 // (op 207) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[26:31], v[32:35], a[0:15], v14, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 208) 





ds_read_b96_tr_b6 v[32:34], v3 offset:21504 // (op 210) Transpose load local data 
ds_read_b96_tr_b6 v[35:37], v3 offset:22528 // (op 210) Transpose load local data 



ds_read_b128 v[38:41], v7 offset:32 // (op 212) Load local data 


ds_read_u8 v16, v0 offset:25728 // (op 214) Load local data 



ds_read_u8 v19, v6 offset:16386 // (op 216) Load local data 



ds_read_b128 v[42:45], v7 offset:4096 // (op 412) Load local data 



ds_read_u8 v21, v6 offset:16640 // (op 416) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[26:31], v[42:45], a[16:31], v14, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 417) 





ds_read_b128 v[42:45], v7 offset:4128 // (op 421) Load local data 



ds_read_u8 v21, v6 offset:16642 // (op 425) Load local data 



ds_read_b128 v[46:49], v7 offset:8192 // (op 646) Load local data 



ds_read_u8 v24, v6 offset:16896 // (op 650) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[26:31], v[46:49], a[32:47], v14, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 651) 





ds_read_b128 v[46:49], v7 offset:8224 // (op 655) Load local data 



ds_read_u8 v24, v6 offset:16898 // (op 659) Load local data 



ds_read_b128 v[50:53], v7 offset:12288 // (op 880) Load local data 



ds_read_u8 v54, v6 offset:17152 // (op 884) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[26:31], v[50:53], a[48:63], v14, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 885) 







v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[32:37], v[38:41], a[0:15], v16, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 217) 
ds_read_b128 v[26:29], v7 offset:12320 // (op 889) Load local data 





v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[32:37], v[42:45], a[16:31], v16, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 426) 
ds_read_u8 v14, v6 offset:17154 // (op 893) Load local data 


v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[32:37], v[46:49], a[32:47], v16, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 660) 


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[32:37], v[26:29], a[48:63], v16, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 894) 




s_ashr_i64 s[0:1], s[22:23], 7 // (op 872) 
s_mov_b32 s7, s0 // (op 872) convert

s_cmp_lt_u32 s6, s7 // (op 872) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ForLoopTop_KLoopTail_872 // (op 872) Condition: Bottom (jump to RR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ForLoopTop_KLoopTail_872 if true)
RR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ForLoopBottom_KLoopTail_872:























s_barrier  // (op 3254) 









v_cmp_eq_f32 vcc, 0.00000, v12 // (op 155) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ConditionalFalse_0_beta_155 // (op 155) Condition: False, jump to RR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ConditionalFalse_0_beta_155
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
s_branch RR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ConditionalBottom_0_beta_155 // (op 155) Condition: Done, jump to RR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ConditionalBottom_0_beta_155
RR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ConditionalFalse_0_beta_155:


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
RR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ConditionalBottom_0_beta_155:












s_endpgm  // End of RR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2
.LRR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_end:
.size RR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2, .LRR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_end-RR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2
  .amdhsa_next_free_vgpr 188
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 51712 // lds bytes
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
    DataType: B6
    DataTypeA: B6
    DataTypeB: F4
    DestDataType: s
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
    args: [ { type: int64, semantic: StrideC0 },
            { type: int64, semantic: StrideD0 },
            { type: address, semantic: AddressC },
            { type: int64, semantic: SizeFree1 },
            { type: address, semantic: AddressD },
            { type: int64, semantic: StrideA0 },
            { type: float32, semantic: Beta },
            { type: float32, semantic: Alpha },
            { type: address, semantic: AddressA },
            { type: int64, semantic: SizeSum },
            { type: int64, semantic: StrideB0 },
            { type: address, semantic: AddressB },
            { type: int64, semantic: StrideScaleA0 },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: SizeSumScaleA },
            { type: int64, semantic: StrideScaleB0 },
            { type: address, semantic: AddressMXScaleB } ]
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
  - .name: RR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2
    .symbol: RR_GEMM_NN_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 51712
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 64
    .vgpr_count: 122
    .agpr_count: 64
    .max_flat_workgroup_size: 256
    .wavefront_size: 64
    .args:
      - .name: Tensor_10_stride_1_11
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
      - .name: Tensor_0_stride_1_0
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
      - .name: Tensor_0_size_1_1
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
      - .name: Tensor_4_stride_1_6
        .size: 8
        .offset: 88
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 96
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_size_1_7
        .size: 8
        .offset: 104
        .value_kind: by_value
      - .name: Tensor_7_stride_1_9
        .size: 8
        .offset: 112
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 120
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
...

.end_amdgpu_metadata

