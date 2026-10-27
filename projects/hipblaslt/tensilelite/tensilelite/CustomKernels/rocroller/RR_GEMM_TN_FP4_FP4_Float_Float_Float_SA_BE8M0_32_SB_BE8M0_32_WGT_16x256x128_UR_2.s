// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2
.p2align 8
.type RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2,@function
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 3717) 
v_and_b32 v3, 15, v1 // (op 3717) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 3717) 


v_and_b32 v0, 63, v1 // (op 3717) 
v_lshrrev_b32 v3, 4, v0 // (op 3717) 

v_lshlrev_b32 v0, 5, v3 // (op 3717) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 3717) 


v_lshlrev_b32 v0, 2, v3 // (op 3717) 

v_lshrrev_b32 v3, 3, v0 // (op 3717) 

v_lshrrev_b32 v0, 7, v1 // (op 3875) 
v_and_b32 v4, 15, v1 // (op 3875) 
v_lshl_add_u32 v5, v0, 4, v4 // (op 3875) 


v_and_b32 v0, 63, v1 // (op 3875) 
v_lshrrev_b32 v4, 4, v0 // (op 3875) 

v_lshlrev_b32 v0, 5, v4 // (op 3875) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 3875) 


v_lshlrev_b32 v0, 2, v4 // (op 3875) 

v_lshrrev_b32 v4, 3, v0 // (op 3875) 

v_lshrrev_b32 v0, 7, v1 // (op 3660) 
v_and_b32 v5, 63, v1 // (op 3660) 
v_lshrrev_b32 v6, 4, v5 // (op 3660) 
v_lshlrev_b32 v7, 2, v6 // (op 3660) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 3660) 


v_lshl_add_u32 v0, s16, 4, v6 // (op 3660) 

v_lshrrev_b32 v6, 2, v5 // (op 3660) 

v_and_b32 v5, 3, v6 // (op 3660) 

v_and_b32 v6, 3, v1 // (op 3660) 
v_lshl_add_u32 v7, v5, 2, v6 // (op 3660) 


v_lshrrev_b32 v5, 6, v1 // (op 3660) 
v_and_b32 v6, 1, v5 // (op 3660) 

v_lshl_add_u32 v5, v6, 4, v7 // (op 3660) 


v_lshl_add_u32 v6, s17, 8, v5 // (op 3660) 

v_mul_lo_u32 v5, v6, s2 // (op 3660) 

v_add_lshl_u32 v6, v0, v5, 2 // (op 3660) 


v_lshrrev_b32 v0, 6, v1 // (op 3775) 
v_and_b32 v5, 1, v0 // (op 3775) 

v_and_b32 v0, 15, v1 // (op 3775) 
v_lshl_add_u32 v7, v5, 4, v0 // (op 3775) 


v_and_b32 v0, 63, v1 // (op 3775) 
v_lshrrev_b32 v5, 4, v0 // (op 3775) 

v_lshlrev_b32 v0, 5, v5 // (op 3775) 

v_lshl_add_u32 v5, v7, 7, v0 // (op 3775) 


v_lshlrev_b32 v0, 2, v5 // (op 3775) 

v_lshrrev_b32 v5, 3, v0 // (op 3775) 

v_lshrrev_b32 v0, 6, v1 // (op 3833) 
v_and_b32 v7, 1, v0 // (op 3833) 

v_and_b32 v0, 15, v1 // (op 3833) 
s_mul_i32 s1, s2, s9 // (op 3662) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 3662) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 3662) least significant
s_mul_hi_u32 s19, s2, s8 // (op 3662) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3662) most significant: sum
s_add_u32 s1, s1, s19 // (op 3662) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 3662) 

v_lshl_add_u32 v8, v7, 4, v0 // (op 3833) 


v_and_b32 v0, 63, v1 // (op 3833) 
v_lshrrev_b32 v7, 4, v0 // (op 3833) 

v_lshl_add_u32 v0, v8, 2, v7 // (op 3833) 


v_lshrrev_b32 v7, 7, v1 // (op 3849) 
v_and_b32 v8, 63, v1 // (op 3849) 
v_lshrrev_b32 v9, 4, v8 // (op 3849) 
v_lshlrev_b32 v10, 2, v9 // (op 3849) 

v_lshl_add_u32 v9, v7, 4, v10 // (op 3849) 


v_lshl_add_u32 v7, s16, 4, v9 // (op 3849) 

v_lshrrev_b32 v9, 2, v8 // (op 3849) 

v_and_b32 v8, 3, v9 // (op 3849) 

v_and_b32 v9, 3, v1 // (op 3849) 
v_lshl_add_u32 v10, v8, 2, v9 // (op 3849) 


v_lshrrev_b32 v8, 6, v1 // (op 3849) 
v_and_b32 v9, 1, v8 // (op 3849) 

v_lshl_add_u32 v8, v9, 4, v10 // (op 3849) 


v_lshl_add_u32 v9, s17, 8, v8 // (op 3849) 

v_mul_lo_u32 v8, v9, s4 // (op 3849) 

v_add_lshl_u32 v9, v7, v8, 2 // (op 3849) 


v_lshrrev_b32 v7, 6, v1 // (op 3904) 
v_and_b32 v8, 1, v7 // (op 3904) 

v_and_b32 v7, 15, v1 // (op 3904) 
v_lshl_add_u32 v10, v8, 4, v7 // (op 3904) 


v_and_b32 v7, 63, v1 // (op 3904) 
v_lshrrev_b32 v8, 4, v7 // (op 3904) 

v_lshlrev_b32 v7, 5, v8 // (op 3904) 

v_lshl_add_u32 v8, v10, 7, v7 // (op 3904) 


v_lshlrev_b32 v7, 2, v8 // (op 3904) 

v_lshrrev_b32 v8, 3, v7 // (op 3904) 

v_lshrrev_b32 v7, 6, v1 // (op 3933) 
v_and_b32 v10, 1, v7 // (op 3933) 

v_and_b32 v7, 15, v1 // (op 3933) 
s_mul_i32 s1, s4, s9 // (op 3851) most significant: low of low * high
v_lshl_add_u32 v11, v10, 4, v7 // (op 3933) 


v_and_b32 v7, 63, v1 // (op 3933) 
v_lshrrev_b32 v10, 4, v7 // (op 3933) 

v_lshl_add_u32 v7, v11, 2, v10 // (op 3933) 


s_mul_i32 s38, s5, s8 // (op 3851) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 3851) least significant
s_mul_hi_u32 s39, s4, s8 // (op 3851) most significant: high of low * low
s_add_u32 s1, s1, s38 // (op 3851) most significant: sum
s_add_u32 s1, s1, s39 // (op 3851) most significant: sum


s_lshl_b64 s[38:39], s[0:1], 2 // (op 3851) 

s_mov_b32 s40, s6 // (op 3662) reg expression
s_mov_b32 s41, s7 // (op 3662) reg expression
s_mov_b32 s42, s18 // (op 3662) reg expression
s_mov_b32 s43, 131072 // (op 3662) call()

s_mov_b32 s44, s10 // (op 3851) reg expression

s_mov_b32 s45, s11 // (op 3851) reg expression
s_mov_b32 s46, s38 // (op 3851) reg expression
s_mov_b32 s47, 131072 // (op 3851) call()


v_mov_b32 v10, s14 // (op 2) Move value
v_mov_b32 v11, s15 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 59) call()
v_accvgpr_write a1, 0.00000 // (op 59) call()
v_accvgpr_write a2, 0.00000 // (op 59) call()
v_accvgpr_write a3, 0.00000 // (op 59) call()
v_accvgpr_write a4, 0.00000 // (op 306) call()
v_accvgpr_write a5, 0.00000 // (op 306) call()
v_accvgpr_write a6, 0.00000 // (op 306) call()
v_accvgpr_write a7, 0.00000 // (op 306) call()
v_accvgpr_write a8, 0.00000 // (op 462) call()
v_accvgpr_write a9, 0.00000 // (op 462) call()
v_accvgpr_write a10, 0.00000 // (op 462) call()
v_accvgpr_write a11, 0.00000 // (op 462) call()
v_accvgpr_write a12, 0.00000 // (op 618) call()
v_accvgpr_write a13, 0.00000 // (op 618) call()
v_accvgpr_write a14, 0.00000 // (op 618) call()
v_accvgpr_write a15, 0.00000 // (op 618) call()
v_accvgpr_write a16, 0.00000 // (op 774) call()
v_accvgpr_write a17, 0.00000 // (op 774) call()
v_accvgpr_write a18, 0.00000 // (op 774) call()
v_accvgpr_write a19, 0.00000 // (op 774) call()
v_accvgpr_write a20, 0.00000 // (op 930) call()
v_accvgpr_write a21, 0.00000 // (op 930) call()
v_accvgpr_write a22, 0.00000 // (op 930) call()
v_accvgpr_write a23, 0.00000 // (op 930) call()
v_accvgpr_write a24, 0.00000 // (op 1086) call()
v_accvgpr_write a25, 0.00000 // (op 1086) call()
v_accvgpr_write a26, 0.00000 // (op 1086) call()
v_accvgpr_write a27, 0.00000 // (op 1086) call()
v_accvgpr_write a28, 0.00000 // (op 1242) call()
v_accvgpr_write a29, 0.00000 // (op 1242) call()
v_accvgpr_write a30, 0.00000 // (op 1242) call()
v_accvgpr_write a31, 0.00000 // (op 1242) call()
v_lshrrev_b32 v12, 3, v1 // (op 3427) 
v_lshl_add_u32 v13, s16, 4, v12 // (op 3427) 

v_and_b32 v12, -8, v1 // (op 3427) 
v_sub_u32 v14, v1, v12 // (op 3427) 

v_lshlrev_b32 v12, 4, v14 // (op 3427) 

v_mov_b32 v15, 0 // (op 3427) convert
v_mov_b32 v14, v12 // (op 3427) convert

v_mul_lo_u32 v17, v13, s13 // (op 3427) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 3427) most significant: high of low * low
v_mul_lo_u32 v16, v13, s12 // (op 3427) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 3427) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 3427) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 3427) most significant half



v_lshlrev_b64 v[12:13], 2, v[16:17] // (op 3427) 

v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 3427) 

v_mov_b32 v12, v14 // (op 3427) convert


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 3429) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 3429) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 3429) least significant
s_mul_hi_u32 s7, s12, s22 // (op 3429) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3429) most significant: sum
s_add_u32 s1, s1, s7 // (op 3429) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 3429) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 3429) 

s_mov_b32 s48, s20 // (op 3429) reg expression
s_mov_b32 s49, s21 // (op 3429) reg expression
s_mov_b32 s50, s0 // (op 3429) reg expression
s_mov_b32 s51, 131072 // (op 3429) call()

v_lshrrev_b32 v13, 3, v1 // (op 3446) 
v_lshl_add_u32 v14, s16, 4, v13 // (op 3446) 

v_and_b32 v13, -8, v1 // (op 3446) 
v_sub_u32 v15, v1, v13 // (op 3446) 

v_lshlrev_b32 v13, 4, v15 // (op 3446) 

v_mov_b32 v17, 0 // (op 3446) convert
v_mov_b32 v16, v13 // (op 3446) convert

v_mul_lo_u32 v19, v14, s13 // (op 3446) most significant: low of low * high
v_mul_hi_u32 v13, v14, s12 // (op 3446) most significant: high of low * low
v_mul_lo_u32 v18, v14, s12 // (op 3446) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 3446) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 3446) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 3446) most significant half



v_lshlrev_b64 v[16:17], 2, v[18:19] // (op 3446) 

v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 3446) 

v_mov_b32 v13, v14 // (op 3446) convert

s_mul_i32 s1, s12, s23 // (op 3448) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 3448) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 3448) least significant
s_mul_hi_u32 s7, s12, s22 // (op 3448) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3448) most significant: sum
s_add_u32 s1, s1, s7 // (op 3448) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 3448) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 3448) 

s_mov_b32 s48, s20 // (op 3448) reg expression
s_mov_b32 s49, s21 // (op 3448) reg expression
s_mov_b32 s50, s0 // (op 3448) reg expression
s_mov_b32 s51, 131072 // (op 3448) call()

v_and_b32 v14, -4, v1 // (op 3491) 
v_sub_u32 v15, v1, v14 // (op 3491) 

v_lshlrev_b32 v14, 5, v15 // (op 3491) 

v_mov_b32 v17, 0 // (op 3491) convert
v_mov_b32 v16, v14 // (op 3491) convert

v_lshrrev_b32 v14, 2, v1 // (op 3491) 
v_lshlrev_b32 v15, 3, v14 // (op 3491) 

v_lshl_add_u32 v14, s17, 8, v15 // (op 3491) 

v_mul_lo_u32 v19, v14, s25 // (op 3491) most significant: low of low * high
v_mul_hi_u32 v15, v14, s24 // (op 3491) most significant: high of low * low
v_mul_lo_u32 v18, v14, s24 // (op 3491) least significant: low of low * low
v_add_u32 v19, v19, v15 // (op 3491) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v18 // (op 3491) least significant half
v_addc_co_u32 v15, s[0:1], v17, v19, s[0:1] // (op 3491) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 3491) 


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 3491) 

v_mov_b32 v14, v16 // (op 3491) convert

s_mul_i32 s1, s24, s9 // (op 3493) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 3493) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 3493) least significant
s_mul_hi_u32 s7, s24, s8 // (op 3493) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3493) most significant: sum
s_add_u32 s1, s1, s7 // (op 3493) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 3493) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 3493) 

s_mov_b32 s52, s26 // (op 3493) reg expression
s_mov_b32 s53, s27 // (op 3493) reg expression
s_mov_b32 s54, s0 // (op 3493) reg expression
s_mov_b32 s55, 131072 // (op 3493) call()

v_and_b32 v15, -4, v1 // (op 3510) 
v_sub_u32 v16, v1, v15 // (op 3510) 

v_lshlrev_b32 v15, 5, v16 // (op 3510) 

v_mov_b32 v17, 0 // (op 3510) convert
v_mov_b32 v16, v15 // (op 3510) convert

v_lshrrev_b32 v15, 2, v1 // (op 3510) 
v_lshlrev_b32 v18, 3, v15 // (op 3510) 

v_lshl_add_u32 v15, s17, 8, v18 // (op 3510) 

v_mul_lo_u32 v19, v15, s25 // (op 3510) most significant: low of low * high
v_mul_hi_u32 v20, v15, s24 // (op 3510) most significant: high of low * low
v_mul_lo_u32 v18, v15, s24 // (op 3510) least significant: low of low * low
v_add_u32 v19, v19, v20 // (op 3510) most significant: sum


v_add_co_u32 v20, s[0:1], v16, v18 // (op 3510) least significant half
v_addc_co_u32 v21, s[0:1], v17, v19, s[0:1] // (op 3510) most significant half

v_lshlrev_b64 v[20:21], 2, v[20:21] // (op 3510) 


v_lshrrev_b64 v[18:19], 3, v[20:21] // (op 3510) 

v_mov_b32 v15, v18 // (op 3510) convert

s_mul_i32 s1, s24, s9 // (op 3512) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 3512) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 3512) least significant
s_mul_hi_u32 s7, s24, s8 // (op 3512) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3512) most significant: sum
s_add_u32 s1, s1, s7 // (op 3512) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 3512) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 3512) 

s_mov_b32 s52, s26 // (op 3512) reg expression
s_mov_b32 s53, s27 // (op 3512) reg expression
s_mov_b32 s54, s0 // (op 3512) reg expression
s_mov_b32 s55, 131072 // (op 3512) call()

v_lshrrev_b32 v16, 7, v1 // (op 3555) 
v_and_b32 v17, 15, v1 // (op 3555) 
v_lshl_add_u32 v18, v16, 4, v17 // (op 3555) 


v_lshl_add_u32 v16, s16, 4, v18 // (op 3555) 

v_and_b32 v17, 63, v1 // (op 3555) 
v_lshrrev_b32 v18, 4, v17 // (op 3555) 

v_mul_lo_u32 v17, v16, s28 // (op 3555) 
v_add_u32 v17, v17, v18 // (op 3555) 


s_mul_i32 s1, s28, s23 // (op 3557) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 3557) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 3557) least significant
s_mul_hi_u32 s7, s28, s22 // (op 3557) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3557) most significant: sum
s_add_u32 s1, s1, s7 // (op 3557) most significant: sum


s_mov_b32 s56, s30 // (op 3557) reg expression
s_mov_b32 s57, s31 // (op 3557) reg expression
s_mov_b32 s58, s0 // (op 3557) reg expression
s_mov_b32 s59, 131072 // (op 3557) call()

v_lshrrev_b32 v16, 7, v1 // (op 3574) 
v_and_b32 v18, 15, v1 // (op 3574) 
v_lshl_add_u32 v19, v16, 4, v18 // (op 3574) 


v_lshl_add_u32 v16, s16, 4, v19 // (op 3574) 

v_and_b32 v18, 63, v1 // (op 3574) 
v_lshrrev_b32 v19, 4, v18 // (op 3574) 

v_mul_lo_u32 v18, v16, s28 // (op 3574) 
v_add_u32 v18, v18, v19 // (op 3574) 


s_mul_i32 s1, s28, s23 // (op 3576) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 3576) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 3576) least significant
s_mul_hi_u32 s7, s28, s22 // (op 3576) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3576) most significant: sum
s_add_u32 s1, s1, s7 // (op 3576) most significant: sum


s_mov_b32 s56, s30 // (op 3576) reg expression
s_mov_b32 s57, s31 // (op 3576) reg expression
s_mov_b32 s58, s0 // (op 3576) reg expression
s_mov_b32 s59, 131072 // (op 3576) call()

v_lshlrev_b32 v16, 1, v1 // (op 3619) 
v_lshl_add_u32 v19, s17, 8, v16 // (op 3619) 

v_mul_lo_u32 v16, v19, s32 // (op 3619) 

s_mul_i32 s1, s32, s9 // (op 3621) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 3621) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 3621) least significant
s_mul_hi_u32 s7, s32, s8 // (op 3621) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3621) most significant: sum
s_add_u32 s1, s1, s7 // (op 3621) most significant: sum


s_mov_b32 s60, s34 // (op 3621) reg expression
s_mov_b32 s61, s35 // (op 3621) reg expression
s_mov_b32 s62, s0 // (op 3621) reg expression
s_mov_b32 s63, 131072 // (op 3621) call()

v_lshlrev_b32 v19, 1, v1 // (op 3638) 
v_lshl_add_u32 v20, s17, 8, v19 // (op 3638) 

v_mul_lo_u32 v19, v20, s32 // (op 3638) 

s_mul_i32 s1, s32, s9 // (op 3640) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 3640) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 3640) least significant
s_mul_hi_u32 s7, s32, s8 // (op 3640) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3640) most significant: sum
s_add_u32 s1, s1, s7 // (op 3640) most significant: sum


s_mov_b32 s60, s34 // (op 3640) reg expression
s_mov_b32 s61, s35 // (op 3640) reg expression
s_mov_b32 s62, s0 // (op 3640) reg expression
s_mov_b32 s63, 131072 // (op 3640) call()




buffer_load_ubyte v20, v17, s[56:59], 0 offen offset: 0 // (op 3108) Load value

s_lshl_b64 s[0:1], s[24:25], 2 // (op 3112) 
v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 3112) 


v_mov_b32 v21, v14 // (op 3112) 
buffer_load_dwordx4 v[24:27], v21, s[52:55], 0 offen // (op 3112) Load value
v_add_u32 v21, v21, v22 // (op 3112) 
buffer_load_dwordx4 v[28:31], v21, s[52:55], 0 offen // (op 3112) Load value
v_add_u32 v21, v21, v22 // (op 3112) 
buffer_load_dwordx4 v[32:35], v21, s[52:55], 0 offen // (op 3112) Load value
v_add_u32 v21, v21, v22 // (op 3112) 
buffer_load_dwordx4 v[36:39], v21, s[52:55], 0 offen // (op 3112) Load value
v_add_u32 v21, v21, v22 // (op 3112) 
buffer_load_dwordx4 v[40:43], v21, s[52:55], 0 offen // (op 3112) Load value
v_add_u32 v21, v21, v22 // (op 3112) 
buffer_load_dwordx4 v[44:47], v21, s[52:55], 0 offen // (op 3112) Load value
v_add_u32 v21, v21, v22 // (op 3112) 
buffer_load_dwordx4 v[48:51], v21, s[52:55], 0 offen // (op 3112) Load value
v_add_u32 v21, v21, v22 // (op 3112) 
buffer_load_dwordx4 v[52:55], v21, s[52:55], 0 offen // (op 3112) Load value



v_mov_b64 v[22:23], s[32:33] // (op 3115) convert

v_mov_b32 v21, v16 // (op 3115) 
buffer_load_dword v56, v21, s[60:63], 0 offen // (op 3115) Load value
v_add_u32 v21, v21, v22 // (op 3115) 
buffer_load_dword v57, v21, s[60:63], 0 offen // (op 3115) Load value





buffer_load_dwordx2 v[22:23], v12, s[48:51], 0 offen // (op 3118) Load value



buffer_load_ubyte v21, v17, s[56:59], 0 offen offset:4 // (op 3122) Load value


s_lshl_b64 s[0:1], s[24:25], 2 // (op 3126) 
v_lshrrev_b64 v[58:59], 3, s[0:1] // (op 3126) 


v_mov_b32 v17, v14 // (op 3126) 
buffer_load_dwordx4 v[60:63], v17, s[52:55], 0 offen offset:64 // (op 3126) Load value
v_add_u32 v17, v17, v58 // (op 3126) 
buffer_load_dwordx4 v[64:67], v17, s[52:55], 0 offen offset:64 // (op 3126) Load value
v_add_u32 v17, v17, v58 // (op 3126) 
buffer_load_dwordx4 v[68:71], v17, s[52:55], 0 offen offset:64 // (op 3126) Load value
v_add_u32 v17, v17, v58 // (op 3126) 
buffer_load_dwordx4 v[72:75], v17, s[52:55], 0 offen offset:64 // (op 3126) Load value
v_add_u32 v17, v17, v58 // (op 3126) 
buffer_load_dwordx4 v[76:79], v17, s[52:55], 0 offen offset:64 // (op 3126) Load value
v_add_u32 v17, v17, v58 // (op 3126) 
buffer_load_dwordx4 v[80:83], v17, s[52:55], 0 offen offset:64 // (op 3126) Load value
v_add_u32 v17, v17, v58 // (op 3126) 
buffer_load_dwordx4 v[84:87], v17, s[52:55], 0 offen offset:64 // (op 3126) Load value
v_add_u32 v17, v17, v58 // (op 3126) 
buffer_load_dwordx4 v[88:91], v17, s[52:55], 0 offen offset:64 // (op 3126) Load value




v_mov_b64 v[58:59], s[32:33] // (op 3129) convert

v_mov_b32 v14, v16 // (op 3129) 
buffer_load_dword v92, v14, s[60:63], 0 offen offset:4 // (op 3129) Load value
v_add_u32 v14, v14, v58 // (op 3129) 
buffer_load_dword v93, v14, s[60:63], 0 offen offset:4 // (op 3129) Load value






buffer_load_dwordx2 v[16:17], v12, s[48:51], 0 offen offset:64 // (op 3132) Load value

v_and_b32 v12, -4, v1 // (op 3758) 
v_sub_u32 v14, v1, v12 // (op 3758) 

v_lshlrev_b32 v12, 5, v14 // (op 3758) 

v_lshrrev_b32 v14, 2, v1 // (op 3758) 
v_lshl_add_u32 v58, v14, 10, v12 // (op 3758) 


v_lshlrev_b32 v12, 2, v58 // (op 3758) 

v_lshrrev_b32 v14, 3, v12 // (op 3758) 




s_waitcnt vmcnt(22)
ds_write_b128 v14, v[24:27] // (op 3134) Store local data 
s_waitcnt vmcnt(21)
ds_write_b128 v14, v[28:31] offset:64 // (op 3134) Store local data 
s_waitcnt vmcnt(20)
ds_write_b128 v14, v[32:35] offset:128 // (op 3134) Store local data 
s_waitcnt vmcnt(19)
ds_write_b128 v14, v[36:39] offset:192 // (op 3134) Store local data 
s_waitcnt vmcnt(18)
ds_write_b128 v14, v[40:43] offset:256 // (op 3134) Store local data 
s_waitcnt vmcnt(17)
ds_write_b128 v14, v[44:47] offset:320 // (op 3134) Store local data 
s_waitcnt vmcnt(16)
ds_write_b128 v14, v[48:51] offset:384 // (op 3134) Store local data 
s_waitcnt vmcnt(15)
ds_write_b128 v14, v[52:55] offset:448 // (op 3134) Store local data 

v_lshlrev_b32 v12, 3, v1 // (op 3816) 



s_waitcnt vmcnt(14)
ds_write_b32 v12, v56 offset:16384 // (op 3135) Store local data 
s_waitcnt vmcnt(13)
ds_write_b32 v12, v57 offset:16388 // (op 3135) Store local data 

v_and_b32 v12, -8, v1 // (op 3705) 
v_sub_u32 v14, v1, v12 // (op 3705) 

v_lshlrev_b32 v12, 4, v14 // (op 3705) 

v_lshrrev_b32 v14, 3, v1 // (op 3705) 
v_lshl_add_u32 v58, v14, 7, v12 // (op 3705) 


v_lshlrev_b32 v12, 2, v58 // (op 3705) 

v_lshrrev_b32 v14, 3, v12 // (op 3705) 




s_waitcnt vmcnt(12)
ds_write_b64 v14, v[22:23] offset:17408 // (op 3136) Store local data 

v_and_b32 v12, -8, v1 // (op 3676) 
v_sub_u32 v14, v1, v12 // (op 3676) 

v_lshlrev_b32 v12, 4, v14 // (op 3676) 

v_lshrrev_b32 v14, 3, v1 // (op 3676) 
v_lshl_add_u32 v58, v14, 7, v12 // (op 3676) 


v_lshlrev_b32 v12, 2, v58 // (op 3676) 

v_lshrrev_b32 v14, 3, v12 // (op 3676) 

v_and_b32 v12, -4, v1 // (op 3729) 
v_sub_u32 v58, v1, v12 // (op 3729) 

v_lshlrev_b32 v12, 5, v58 // (op 3729) 

v_lshrrev_b32 v58, 2, v1 // (op 3729) 
v_lshl_add_u32 v59, v58, 10, v12 // (op 3729) 


v_lshlrev_b32 v12, 2, v59 // (op 3729) 

v_lshrrev_b32 v58, 3, v12 // (op 3729) 

v_lshlrev_b32 v12, 3, v1 // (op 3787) 
v_and_b32 v59, -8, v1 // (op 3863) 
v_sub_u32 v94, v1, v59 // (op 3863) 

v_lshlrev_b32 v59, 4, v94 // (op 3863) 

v_lshrrev_b32 v94, 3, v1 // (op 3863) 
v_lshl_add_u32 v95, v94, 7, v59 // (op 3863) 


v_lshlrev_b32 v59, 2, v95 // (op 3863) 

v_lshrrev_b32 v94, 3, v59 // (op 3863) 

v_and_b32 v59, -4, v1 // (op 3887) 
v_sub_u32 v95, v1, v59 // (op 3887) 

v_lshlrev_b32 v59, 5, v95 // (op 3887) 

v_lshrrev_b32 v95, 2, v1 // (op 3887) 
v_lshl_add_u32 v96, v95, 10, v59 // (op 3887) 


v_lshlrev_b32 v59, 2, v96 // (op 3887) 

v_lshrrev_b32 v95, 3, v59 // (op 3887) 

v_lshlrev_b32 v59, 3, v1 // (op 3916) 
s_mov_b32 s0, 0 // (op 1244) call()
s_ashr_i64 s[6:7], s[36:37], 7 // (op 1243) 
s_mov_b32 s1, s6 // (op 1243) convert

s_and_b32 s6, s1, 4294967294 // (op 1243) 

s_cmp_lt_u32 s0, s6 // (op 1243) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_ForLoopBottom_KLoop_1243 // (op 1243) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_ForLoopBottom_KLoop_1243 if false)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_ForLoopTop_KLoop_1243:
s_add_u32 s0, s0, 2 // (op 1245) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 3992) 



ds_read_b128 v[96:99], v3 offset:17408 // (op 65) Load local data 
s_barrier  // (op 4005) 



ds_read_b128 v[100:103], v5 // (op 70) Load local data 
s_barrier  // (op 4019) 



ds_read_u8 v104, v0 offset:16384 // (op 79) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[96:99], v[100:103], a[0:3], v20, v104 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 82) 
ds_read_b128 v[106:109], v5 offset:2048 // (op 339) Load local data 





ds_read_u8 v100, v0 offset:16512 // (op 342) Load local data 



ds_read_b128 v[102:105], v5 offset:4096 // (op 495) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[96:99], v[106:109], a[4:7], v20, v100 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 343) 





ds_read_u8 v100, v0 offset:16640 // (op 498) Load local data 



ds_read_b128 v[106:109], v5 offset:6144 // (op 651) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[96:99], v[102:105], a[8:11], v20, v100 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 499) 





ds_read_u8 v100, v0 offset:16768 // (op 654) Load local data 



ds_read_b128 v[102:105], v5 offset:8192 // (op 807) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[96:99], v[106:109], a[12:15], v20, v100 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 655) 





ds_read_u8 v100, v0 offset:16896 // (op 810) Load local data 



ds_read_b128 v[106:109], v5 offset:10240 // (op 963) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[96:99], v[102:105], a[16:19], v20, v100 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 811) 





ds_read_u8 v100, v0 offset:17024 // (op 966) Load local data 



ds_read_b128 v[102:105], v5 offset:12288 // (op 1119) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[96:99], v[106:109], a[20:23], v20, v100 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 967) 





ds_read_u8 v100, v0 offset:17152 // (op 1122) Load local data 



ds_read_b128 v[106:109], v5 offset:14336 // (op 1275) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[96:99], v[102:105], a[24:27], v20, v100 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1123) 





ds_read_u8 v100, v0 offset:17280 // (op 1278) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[96:99], v[106:109], a[28:31], v20, v100 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1279) 





v_mov_b32 v96, v18 // (op 75) 

buffer_load_ubyte v20, v96, s[56:59], 0 offen offset:8 // (op 75) Load value

s_lshl_b64 s[6:7], s[24:25], 2 // (op 10) 
v_lshrrev_b64 v[98:99], 3, s[6:7] // (op 10) 


v_mov_b32 v97, v15 // (op 10) 
v_mov_b32 v100, v97 // (op 10) 
buffer_load_dwordx4 v[24:27], v100, s[52:55], 0 offen offset:128 // (op 10) Load value
v_add_u32 v100, v100, v98 // (op 10) 
buffer_load_dwordx4 v[28:31], v100, s[52:55], 0 offen offset:128 // (op 10) Load value
v_add_u32 v100, v100, v98 // (op 10) 
buffer_load_dwordx4 v[32:35], v100, s[52:55], 0 offen offset:128 // (op 10) Load value
v_add_u32 v100, v100, v98 // (op 10) 
buffer_load_dwordx4 v[36:39], v100, s[52:55], 0 offen offset:128 // (op 10) Load value
v_add_u32 v100, v100, v98 // (op 10) 
buffer_load_dwordx4 v[40:43], v100, s[52:55], 0 offen offset:128 // (op 10) Load value
v_add_u32 v100, v100, v98 // (op 10) 
buffer_load_dwordx4 v[44:47], v100, s[52:55], 0 offen offset:128 // (op 10) Load value
v_add_u32 v100, v100, v98 // (op 10) 
buffer_load_dwordx4 v[48:51], v100, s[52:55], 0 offen offset:128 // (op 10) Load value
v_add_u32 v100, v100, v98 // (op 10) 
buffer_load_dwordx4 v[52:55], v100, s[52:55], 0 offen offset:128 // (op 10) Load value



v_mov_b64 v[98:99], s[32:33] // (op 21) convert

v_mov_b32 v100, v19 // (op 21) 
v_mov_b32 v101, v100 // (op 21) 
buffer_load_dword v56, v101, s[60:63], 0 offen offset:8 // (op 21) Load value
v_add_u32 v101, v101, v98 // (op 21) 
buffer_load_dword v57, v101, s[60:63], 0 offen offset:8 // (op 21) Load value




v_mov_b32 v98, v13 // (op 4) 

buffer_load_dwordx2 v[22:23], v98, s[48:51], 0 offen offset:128 // (op 4) Load value
s_barrier  // (op 4042) 



s_waitcnt vmcnt(22)
ds_write_b128 v95, v[60:63] offset:18432 // (op 235) Store local data 
s_waitcnt vmcnt(21)
ds_write_b128 v95, v[64:67] offset:18496 // (op 235) Store local data 
s_waitcnt vmcnt(20)
ds_write_b128 v95, v[68:71] offset:18560 // (op 235) Store local data 
s_waitcnt vmcnt(19)
ds_write_b128 v95, v[72:75] offset:18624 // (op 235) Store local data 
s_waitcnt vmcnt(18)
ds_write_b128 v95, v[76:79] offset:18688 // (op 235) Store local data 
s_waitcnt vmcnt(17)
ds_write_b128 v95, v[80:83] offset:18752 // (op 235) Store local data 
s_waitcnt vmcnt(16)
ds_write_b128 v95, v[84:87] offset:18816 // (op 235) Store local data 
s_waitcnt vmcnt(15)
ds_write_b128 v95, v[88:91] offset:18880 // (op 235) Store local data 
s_barrier  // (op 4048) 



s_waitcnt vmcnt(14)
ds_write_b32 v59, v92 offset:34816 // (op 236) Store local data 
s_waitcnt vmcnt(13)
ds_write_b32 v59, v93 offset:34820 // (op 236) Store local data 
s_barrier  // (op 4036) 



s_waitcnt vmcnt(12)
ds_write_b64 v94, v[16:17] offset:35840 // (op 225) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 4033) 



ds_read_b128 v[102:105], v4 offset:35840 // (op 227) Load local data 
s_barrier  // (op 4039) 
s_barrier  // (op 4085) 



ds_read_b128 v[106:109], v8 offset:18432 // (op 229) Load local data 
s_barrier  // (op 4045) 



ds_read_u8 v99, v7 offset:34816 // (op 233) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[102:105], v[106:109], a[0:3], v21, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 234) 
ds_read_b128 v[110:113], v8 offset:20480 // (op 348) Load local data 





ds_read_u8 v99, v7 offset:34944 // (op 351) Load local data 



ds_read_b128 v[106:109], v8 offset:22528 // (op 504) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[102:105], v[110:113], a[4:7], v21, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 344) 





ds_read_u8 v99, v7 offset:35072 // (op 507) Load local data 



ds_read_b128 v[110:113], v8 offset:24576 // (op 660) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[102:105], v[106:109], a[8:11], v21, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 500) 





ds_read_u8 v99, v7 offset:35200 // (op 663) Load local data 



ds_read_b128 v[106:109], v8 offset:26624 // (op 816) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[102:105], v[110:113], a[12:15], v21, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 656) 





ds_read_u8 v99, v7 offset:35328 // (op 819) Load local data 



ds_read_b128 v[110:113], v8 offset:28672 // (op 972) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[102:105], v[106:109], a[16:19], v21, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 812) 





ds_read_u8 v99, v7 offset:35456 // (op 975) Load local data 



ds_read_b128 v[106:109], v8 offset:30720 // (op 1128) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[102:105], v[110:113], a[20:23], v21, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 968) 





ds_read_u8 v99, v7 offset:35584 // (op 1131) Load local data 



ds_read_b128 v[110:113], v8 offset:32768 // (op 1284) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[102:105], v[106:109], a[24:27], v21, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1124) 





s_barrier  // (op 4155) 
ds_read_u8 v99, v7 offset:35712 // (op 1287) Load local data 
s_barrier  // (op 4159) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[102:105], v[110:113], a[28:31], v21, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1280) 






buffer_load_ubyte v21, v96, s[56:59], 0 offen offset:12 // (op 231) Load value


v_add_u32 v18, 8, v18 // (op 3566) 
s_lshl_b64 s[6:7], s[24:25], 2 // (op 237) 
v_lshrrev_b64 v[102:103], 3, s[6:7] // (op 237) 


v_mov_b32 v96, v97 // (op 237) 
buffer_load_dwordx4 v[60:63], v96, s[52:55], 0 offen offset:192 // (op 237) Load value
v_add_u32 v96, v96, v102 // (op 237) 
buffer_load_dwordx4 v[64:67], v96, s[52:55], 0 offen offset:192 // (op 237) Load value
v_add_u32 v96, v96, v102 // (op 237) 
buffer_load_dwordx4 v[68:71], v96, s[52:55], 0 offen offset:192 // (op 237) Load value
v_add_u32 v96, v96, v102 // (op 237) 
buffer_load_dwordx4 v[72:75], v96, s[52:55], 0 offen offset:192 // (op 237) Load value
v_add_u32 v96, v96, v102 // (op 237) 
buffer_load_dwordx4 v[76:79], v96, s[52:55], 0 offen offset:192 // (op 237) Load value
v_add_u32 v96, v96, v102 // (op 237) 
buffer_load_dwordx4 v[80:83], v96, s[52:55], 0 offen offset:192 // (op 237) Load value
v_add_u32 v96, v96, v102 // (op 237) 
buffer_load_dwordx4 v[84:87], v96, s[52:55], 0 offen offset:192 // (op 237) Load value
v_add_u32 v96, v96, v102 // (op 237) 
buffer_load_dwordx4 v[88:91], v96, s[52:55], 0 offen offset:192 // (op 237) Load value




v_add_u32 v15, 128, v15 // (op 3502) 
v_mov_b64 v[96:97], s[32:33] // (op 238) convert

v_mov_b32 v99, v100 // (op 238) 
buffer_load_dword v92, v99, s[60:63], 0 offen offset:12 // (op 238) Load value
v_add_u32 v99, v99, v96 // (op 238) 
buffer_load_dword v93, v99, s[60:63], 0 offen offset:12 // (op 238) Load value






v_add_u32 v19, 8, v19 // (op 3630) 
buffer_load_dwordx2 v[16:17], v98, s[48:51], 0 offen offset:192 // (op 224) Load value

s_barrier  // (op 4008) 
v_add_u32 v13, 128, v13 // (op 3438) 



s_waitcnt vmcnt(22)
ds_write_b128 v58, v[24:27] // (op 13) Store local data 
s_waitcnt vmcnt(21)
ds_write_b128 v58, v[28:31] offset:64 // (op 13) Store local data 
s_waitcnt vmcnt(20)
ds_write_b128 v58, v[32:35] offset:128 // (op 13) Store local data 
s_waitcnt vmcnt(19)
ds_write_b128 v58, v[36:39] offset:192 // (op 13) Store local data 
s_waitcnt vmcnt(18)
ds_write_b128 v58, v[40:43] offset:256 // (op 13) Store local data 
s_waitcnt vmcnt(17)
ds_write_b128 v58, v[44:47] offset:320 // (op 13) Store local data 
s_waitcnt vmcnt(16)
ds_write_b128 v58, v[48:51] offset:384 // (op 13) Store local data 
s_waitcnt vmcnt(15)
ds_write_b128 v58, v[52:55] offset:448 // (op 13) Store local data 
s_barrier  // (op 4022) 



s_waitcnt vmcnt(14)
ds_write_b32 v12, v56 offset:16384 // (op 26) Store local data 
s_waitcnt vmcnt(13)
ds_write_b32 v12, v57 offset:16388 // (op 26) Store local data 
s_barrier  // (op 3995) 



s_waitcnt vmcnt(12)
ds_write_b64 v14, v[22:23] offset:17408 // (op 6) Store local data 
s_ashr_i64 s[6:7], s[36:37], 7 // (op 1243) 
s_mov_b32 s1, s6 // (op 1243) convert

s_and_b32 s6, s1, 4294967294 // (op 1243) 

s_cmp_lt_u32 s0, s6 // (op 1243) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_ForLoopTop_KLoop_1243 // (op 1243) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_ForLoopTop_KLoop_1243 if true)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_ForLoopBottom_KLoop_1243:















v_lshrrev_b32 v12, 3, v1 // (op 3394) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 3394) 
v_lshl_add_u32 v13, s16, 4, v12 // (op 3394) 

s_lshr_b32 s6, s0, 1 // (op 3394) 

v_and_b32 v12, -8, v1 // (op 3394) 
v_sub_u32 v14, v1, v12 // (op 3394) 

v_lshlrev_b32 v12, 4, v14 // (op 3394) 

v_lshl_add_u32 v14, s6, 8, v12 // (op 3394) 


v_mov_b32 v19, 0 // (op 3394) convert
v_mov_b32 v18, v14 // (op 3394) convert

v_mul_lo_u32 v15, v13, s13 // (op 3394) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 3394) most significant: high of low * low
v_mul_lo_u32 v14, v13, s12 // (op 3394) least significant: low of low * low
v_add_u32 v15, v15, v12 // (op 3394) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v18 // (op 3394) least significant half
v_addc_co_u32 v15, s[0:1], v15, v19, s[0:1] // (op 3394) most significant half



v_lshlrev_b64 v[12:13], 2, v[14:15] // (op 3394) 

v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 3394) 

v_mov_b32 v12, v14 // (op 3394) convert

s_mul_i32 s1, s12, s23 // (op 3396) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 3396) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 3396) least significant
s_mul_hi_u32 s7, s12, s22 // (op 3396) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3396) most significant: sum
s_add_u32 s1, s1, s7 // (op 3396) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 3396) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 3396) 

s_mov_b32 s48, s20 // (op 3396) reg expression
s_mov_b32 s49, s21 // (op 3396) reg expression
s_mov_b32 s50, s0 // (op 3396) reg expression
s_mov_b32 s51, 131072 // (op 3396) call()

v_and_b32 v13, -4, v1 // (op 3468) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 3468) 
v_sub_u32 v14, v1, v13 // (op 3468) 

s_lshr_b32 s6, s0, 1 // (op 3468) 

v_lshlrev_b32 v13, 5, v14 // (op 3468) 

v_lshl_add_u32 v14, s6, 8, v13 // (op 3468) 


v_mov_b32 v19, 0 // (op 3468) convert
v_mov_b32 v18, v14 // (op 3468) convert

v_lshrrev_b32 v20, 2, v1 // (op 3468) 
v_lshlrev_b32 v13, 3, v20 // (op 3468) 

v_lshl_add_u32 v20, s17, 8, v13 // (op 3468) 

v_mul_lo_u32 v15, v20, s25 // (op 3468) most significant: low of low * high
v_mul_hi_u32 v13, v20, s24 // (op 3468) most significant: high of low * low
v_mul_lo_u32 v14, v20, s24 // (op 3468) least significant: low of low * low
v_add_u32 v15, v15, v13 // (op 3468) most significant: sum


v_add_co_u32 v22, s[0:1], v18, v14 // (op 3468) least significant half
v_addc_co_u32 v23, s[0:1], v19, v15, s[0:1] // (op 3468) most significant half

v_lshlrev_b64 v[22:23], 2, v[22:23] // (op 3468) 


v_lshrrev_b64 v[14:15], 3, v[22:23] // (op 3468) 

v_mov_b32 v13, v14 // (op 3468) convert

s_mul_i32 s1, s24, s9 // (op 3470) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 3470) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 3470) least significant
s_mul_hi_u32 s7, s24, s8 // (op 3470) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3470) most significant: sum
s_add_u32 s1, s1, s7 // (op 3470) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 3470) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 3470) 

s_mov_b32 s52, s26 // (op 3470) reg expression
s_mov_b32 s53, s27 // (op 3470) reg expression
s_mov_b32 s54, s0 // (op 3470) reg expression
s_mov_b32 s55, 131072 // (op 3470) call()

v_lshrrev_b32 v14, 7, v1 // (op 3532) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 3532) 
v_and_b32 v15, 15, v1 // (op 3532) 
s_lshr_b32 s6, s0, 1 // (op 3532) 

v_lshl_add_u32 v18, v14, 4, v15 // (op 3532) 


v_lshl_add_u32 v14, s16, 4, v18 // (op 3532) 

v_and_b32 v15, 63, v1 // (op 3532) 
v_lshrrev_b32 v18, 4, v15 // (op 3532) 

v_lshl_add_u32 v15, s6, 3, v18 // (op 3532) 


v_mul_lo_u32 v18, v14, s28 // (op 3532) 
v_add_u32 v18, v18, v15 // (op 3532) 


s_mul_i32 s1, s28, s23 // (op 3534) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 3534) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 3534) least significant
s_mul_hi_u32 s7, s28, s22 // (op 3534) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3534) most significant: sum
s_add_u32 s1, s1, s7 // (op 3534) most significant: sum


s_mov_b32 s56, s30 // (op 3534) reg expression
s_mov_b32 s57, s31 // (op 3534) reg expression
s_mov_b32 s58, s0 // (op 3534) reg expression
s_mov_b32 s59, 131072 // (op 3534) call()

v_lshlrev_b32 v14, 1, v1 // (op 3596) 
v_lshl_add_u32 v15, s17, 8, v14 // (op 3596) 

s_ashr_i64 s[0:1], s[36:37], 7 // (op 3596) 
v_mul_lo_u32 v14, v15, s32 // (op 3596) 

s_lshr_b32 s6, s0, 1 // (op 3596) 

v_lshl_add_u32 v15, s6, 3, v14 // (op 3596) 


s_mul_i32 s1, s32, s9 // (op 3598) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 3598) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 3598) least significant
s_mul_hi_u32 s7, s32, s8 // (op 3598) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3598) most significant: sum
s_add_u32 s1, s1, s7 // (op 3598) most significant: sum


s_mov_b32 s60, s34 // (op 3598) reg expression
s_mov_b32 s61, s35 // (op 3598) reg expression
s_mov_b32 s62, s0 // (op 3598) reg expression
s_mov_b32 s63, 131072 // (op 3598) call()

v_and_b32 v14, -8, v1 // (op 3689) 
v_sub_u32 v19, v1, v14 // (op 3689) 

v_lshlrev_b32 v14, 4, v19 // (op 3689) 

v_lshrrev_b32 v19, 3, v1 // (op 3689) 
v_lshl_add_u32 v20, v19, 7, v14 // (op 3689) 


v_lshlrev_b32 v14, 2, v20 // (op 3689) 

v_lshrrev_b32 v19, 3, v14 // (op 3689) 

v_and_b32 v14, -4, v1 // (op 3742) 
v_sub_u32 v20, v1, v14 // (op 3742) 

v_lshlrev_b32 v14, 5, v20 // (op 3742) 

v_lshrrev_b32 v20, 2, v1 // (op 3742) 
v_lshl_add_u32 v22, v20, 10, v14 // (op 3742) 


v_lshlrev_b32 v14, 2, v22 // (op 3742) 

v_lshrrev_b32 v20, 3, v14 // (op 3742) 

v_lshlrev_b32 v14, 3, v1 // (op 3800) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1249) 
s_and_b32 s6, s0, 4294967294 // (op 1249) 

s_ashr_i64 s[0:1], s[36:37], 7 // (op 1248) 
s_mov_b32 s7, s0 // (op 1248) convert

s_cmp_lt_u32 s6, s7 // (op 1248) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_ForLoopBottom_KLoopTail_1248 // (op 1248) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_ForLoopBottom_KLoopTail_1248 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_ForLoopTop_KLoopTail_1248:


s_add_u32 s6, s6, 1 // (op 1250) 
v_mov_b32 v16, v12 // (op 166) 

buffer_load_dwordx2 v[22:23], v16, s[48:51], 0 offen // (op 166) Load value

v_add_u32 v12, 64, v12 // (op 3384) 

s_barrier  // (op 3998) 
s_lshl_b64 s[0:1], s[24:25], 2 // (op 179) 



v_lshrrev_b64 v[16:17], 3, s[0:1] // (op 179) 


v_mov_b32 v21, v13 // (op 179) 
v_mov_b32 v24, v21 // (op 179) 
buffer_load_dwordx4 v[26:29], v24, s[52:55], 0 offen // (op 179) Load value
v_add_u32 v24, v24, v16 // (op 179) 
buffer_load_dwordx4 v[30:33], v24, s[52:55], 0 offen // (op 179) Load value
v_add_u32 v24, v24, v16 // (op 179) 
s_waitcnt vmcnt(2)
ds_write_b64 v19, v[22:23] offset:17408 // (op 167) Store local data 
buffer_load_dwordx4 v[34:37], v24, s[52:55], 0 offen // (op 179) Load value

v_add_u32 v24, v24, v16 // (op 179) 
buffer_load_dwordx4 v[38:41], v24, s[52:55], 0 offen // (op 179) Load value
v_add_u32 v24, v24, v16 // (op 179) 
buffer_load_dwordx4 v[42:45], v24, s[52:55], 0 offen // (op 179) Load value
v_add_u32 v24, v24, v16 // (op 179) 
buffer_load_dwordx4 v[46:49], v24, s[52:55], 0 offen // (op 179) Load value
v_add_u32 v24, v24, v16 // (op 179) 
buffer_load_dwordx4 v[50:53], v24, s[52:55], 0 offen // (op 179) Load value
v_add_u32 v24, v24, v16 // (op 179) 
buffer_load_dwordx4 v[54:57], v24, s[52:55], 0 offen // (op 179) Load value



v_add_u32 v13, 64, v13 // (op 3460) 

s_barrier  // (op 4011) 
v_mov_b64 v[16:17], s[32:33] // (op 180) convert




v_mov_b32 v21, v15 // (op 180) 
v_mov_b32 v22, v21 // (op 180) 
s_waitcnt vmcnt(7)
ds_write_b128 v20, v[26:29] // (op 177) Store local data 
s_waitcnt vmcnt(6)
ds_write_b128 v20, v[30:33] offset:64 // (op 177) Store local data 
s_waitcnt vmcnt(5)
ds_write_b128 v20, v[34:37] offset:128 // (op 177) Store local data 
s_waitcnt vmcnt(4)
ds_write_b128 v20, v[38:41] offset:192 // (op 177) Store local data 
s_waitcnt vmcnt(3)
ds_write_b128 v20, v[42:45] offset:256 // (op 177) Store local data 
s_waitcnt vmcnt(2)
ds_write_b128 v20, v[46:49] offset:320 // (op 177) Store local data 
buffer_load_dword v24, v22, s[60:63], 0 offen // (op 180) Load value
v_add_u32 v22, v22, v16 // (op 180) 
buffer_load_dword v25, v22, s[60:63], 0 offen // (op 180) Load value



v_add_u32 v15, 4, v15 // (op 3588) 
s_waitcnt vmcnt(3)
ds_write_b128 v20, v[50:53] offset:384 // (op 177) Store local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 4001) 
s_waitcnt vmcnt(2)
ds_write_b128 v20, v[54:57] offset:448 // (op 177) Store local data 




ds_read_b128 v[26:29], v3 offset:17408 // (op 169) Load local data 
s_barrier  // (op 4025) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 4015) 






ds_read_b128 v[30:33], v5 // (op 171) Load local data 
s_waitcnt vmcnt(1)
ds_write_b32 v14, v24 offset:16384 // (op 178) Store local data 


v_mov_b32 v16, v18 // (op 173) 

buffer_load_ubyte v17, v16, s[56:59], 0 offen offset: 0 // (op 173) Load value

v_add_u32 v18, 4, v18 // (op 3524) 
s_waitcnt vmcnt(1)
ds_write_b32 v14, v25 offset:16388 // (op 178) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 4029) 



ds_read_u8 v16, v0 offset:16384 // (op 175) Load local data 



ds_read_b128 v[22:25], v5 offset:2048 // (op 320) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:29], v[30:33], a[0:3], v17, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 176) 





ds_read_u8 v16, v0 offset:16512 // (op 323) Load local data 



ds_read_b128 v[30:33], v5 offset:4096 // (op 476) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[26:29], v[22:25], a[4:7], v17, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 324) 





ds_read_u8 v16, v0 offset:16640 // (op 479) Load local data 



ds_read_b128 v[22:25], v5 offset:6144 // (op 632) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[26:29], v[30:33], a[8:11], v17, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 480) 





ds_read_u8 v16, v0 offset:16768 // (op 635) Load local data 



ds_read_b128 v[30:33], v5 offset:8192 // (op 788) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[26:29], v[22:25], a[12:15], v17, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 636) 





ds_read_u8 v16, v0 offset:16896 // (op 791) Load local data 



ds_read_b128 v[22:25], v5 offset:10240 // (op 944) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[26:29], v[30:33], a[16:19], v17, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 792) 





ds_read_u8 v16, v0 offset:17024 // (op 947) Load local data 



ds_read_b128 v[30:33], v5 offset:12288 // (op 1100) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[26:29], v[22:25], a[20:23], v17, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 948) 





ds_read_u8 v16, v0 offset:17152 // (op 1103) Load local data 



ds_read_b128 v[22:25], v5 offset:14336 // (op 1256) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[26:29], v[30:33], a[24:27], v17, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1104) 





ds_read_u8 v16, v0 offset:17280 // (op 1259) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[26:29], v[22:25], a[28:31], v17, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1260) 




s_ashr_i64 s[0:1], s[36:37], 7 // (op 1248) 
s_mov_b32 s7, s0 // (op 1248) convert

s_cmp_lt_u32 s6, s7 // (op 1248) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_ForLoopTop_KLoopTail_1248 // (op 1248) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_ForLoopTop_KLoopTail_1248 if true)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_ForLoopBottom_KLoopTail_1248:






















s_barrier  // (op 4143) 







v_cmp_eq_f32 vcc, 0.00000, v10 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_ConditionalFalse_0_beta_123
v_accvgpr_read v12, a0 // (op 46) 
v_accvgpr_read v13, a1 // (op 46) 
v_accvgpr_read v14, a2 // (op 46) 
v_accvgpr_read v15, a3 // (op 46) 
v_mul_f32 v16, v11, v12 // (op 46) 
v_mul_f32 v17, v11, v13 // (op 46) 
v_mul_f32 v18, v11, v14 // (op 46) 
v_mul_f32 v19, v11, v15 // (op 46) 



v_mov_b32 v0, v9 // (op 52) reg expression

buffer_store_dwordx4 v[16:19], v0, s[44:47], 0 offen // (op 52) Store value


v_accvgpr_read v12, a4 // (op 1572) 
v_accvgpr_read v13, a5 // (op 1572) 
v_accvgpr_read v14, a6 // (op 1572) 
v_accvgpr_read v15, a7 // (op 1572) 
v_mul_f32 v16, v11, v12 // (op 1572) 
v_mul_f32 v17, v11, v13 // (op 1572) 
v_mul_f32 v18, v11, v14 // (op 1572) 
v_mul_f32 v19, v11, v15 // (op 1572) 



s_lshl_b32 s0, s4, 7 // (op 1574) 
v_add_u32 v0, v9, s0 // (op 1574) 


buffer_store_dwordx4 v[16:19], v0, s[44:47], 0 offen // (op 1574) Store value


v_accvgpr_read v12, a8 // (op 1584) 
v_accvgpr_read v13, a9 // (op 1584) 
v_accvgpr_read v14, a10 // (op 1584) 
v_accvgpr_read v15, a11 // (op 1584) 
v_mul_f32 v16, v11, v12 // (op 1584) 
v_mul_f32 v17, v11, v13 // (op 1584) 
v_mul_f32 v18, v11, v14 // (op 1584) 
v_mul_f32 v19, v11, v15 // (op 1584) 



s_lshl_b32 s0, s4, 8 // (op 1586) 
v_add_u32 v0, v9, s0 // (op 1586) 


buffer_store_dwordx4 v[16:19], v0, s[44:47], 0 offen // (op 1586) Store value


v_accvgpr_read v12, a12 // (op 1596) 
v_accvgpr_read v13, a13 // (op 1596) 
v_accvgpr_read v14, a14 // (op 1596) 
v_accvgpr_read v15, a15 // (op 1596) 
v_mul_f32 v16, v11, v12 // (op 1596) 
v_mul_f32 v17, v11, v13 // (op 1596) 
v_mul_f32 v18, v11, v14 // (op 1596) 
v_mul_f32 v19, v11, v15 // (op 1596) 



s_lshl_b32 s0, s4, 7 // (op 1598) 
s_mul_i32 s1, 3, s0 // (op 1598) 

v_add_u32 v0, v9, s1 // (op 1598) 


buffer_store_dwordx4 v[16:19], v0, s[44:47], 0 offen // (op 1598) Store value


v_accvgpr_read v12, a16 // (op 1608) 
v_accvgpr_read v13, a17 // (op 1608) 
v_accvgpr_read v14, a18 // (op 1608) 
v_accvgpr_read v15, a19 // (op 1608) 
v_mul_f32 v16, v11, v12 // (op 1608) 
v_mul_f32 v17, v11, v13 // (op 1608) 
v_mul_f32 v18, v11, v14 // (op 1608) 
v_mul_f32 v19, v11, v15 // (op 1608) 



s_lshl_b32 s0, s4, 9 // (op 1610) 
v_add_u32 v0, v9, s0 // (op 1610) 


buffer_store_dwordx4 v[16:19], v0, s[44:47], 0 offen // (op 1610) Store value


v_accvgpr_read v12, a20 // (op 1620) 
v_accvgpr_read v13, a21 // (op 1620) 
v_accvgpr_read v14, a22 // (op 1620) 
v_accvgpr_read v15, a23 // (op 1620) 
v_mul_f32 v16, v11, v12 // (op 1620) 
v_mul_f32 v17, v11, v13 // (op 1620) 
v_mul_f32 v18, v11, v14 // (op 1620) 
v_mul_f32 v19, v11, v15 // (op 1620) 



s_lshl_b32 s0, s4, 7 // (op 1622) 
s_mul_i32 s1, 5, s0 // (op 1622) 

v_add_u32 v0, v9, s1 // (op 1622) 


buffer_store_dwordx4 v[16:19], v0, s[44:47], 0 offen // (op 1622) Store value


v_accvgpr_read v12, a24 // (op 1632) 
v_accvgpr_read v13, a25 // (op 1632) 
v_accvgpr_read v14, a26 // (op 1632) 
v_accvgpr_read v15, a27 // (op 1632) 
v_mul_f32 v16, v11, v12 // (op 1632) 
v_mul_f32 v17, v11, v13 // (op 1632) 
v_mul_f32 v18, v11, v14 // (op 1632) 
v_mul_f32 v19, v11, v15 // (op 1632) 



s_lshl_b32 s0, s4, 7 // (op 1634) 
s_mul_i32 s1, 6, s0 // (op 1634) 

v_add_u32 v0, v9, s1 // (op 1634) 


buffer_store_dwordx4 v[16:19], v0, s[44:47], 0 offen // (op 1634) Store value


v_accvgpr_read v12, a28 // (op 1644) 
v_accvgpr_read v13, a29 // (op 1644) 
v_accvgpr_read v14, a30 // (op 1644) 
v_accvgpr_read v15, a31 // (op 1644) 
v_mul_f32 v16, v11, v12 // (op 1644) 
v_mul_f32 v17, v11, v13 // (op 1644) 
v_mul_f32 v18, v11, v14 // (op 1644) 
v_mul_f32 v19, v11, v15 // (op 1644) 



s_lshl_b32 s0, s4, 7 // (op 1646) 
s_mul_i32 s1, 7, s0 // (op 1646) 

v_add_u32 v0, v9, s1 // (op 1646) 


buffer_store_dwordx4 v[16:19], v0, s[44:47], 0 offen // (op 1646) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_ConditionalBottom_0_beta_123
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_ConditionalFalse_0_beta_123:


v_mov_b32 v0, v6 // (op 135) reg expression

buffer_load_dwordx4 v[12:15], v0, s[40:43], 0 offen // (op 135) Load value

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




v_mov_b32 v0, v9 // (op 134) reg expression

buffer_store_dwordx4 v[12:15], v0, s[44:47], 0 offen // (op 134) Store value




s_lshl_b32 s0, s2, 7 // (op 1698) 
v_add_u32 v0, v6, s0 // (op 1698) 


buffer_load_dwordx4 v[12:15], v0, s[40:43], 0 offen // (op 1698) Load value

v_accvgpr_read v16, a4 // (op 1695) 
v_accvgpr_read v17, a5 // (op 1695) 
v_accvgpr_read v18, a6 // (op 1695) 
v_accvgpr_read v19, a7 // (op 1695) 
v_mul_f32 v20, v11, v16 // (op 1695) 
v_mul_f32 v21, v11, v17 // (op 1695) 
v_mul_f32 v22, v11, v18 // (op 1695) 
v_mul_f32 v23, v11, v19 // (op 1695) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v10, v12 // (op 1699) 
v_mul_f32 v17, v10, v13 // (op 1699) 
v_mul_f32 v18, v10, v14 // (op 1699) 
v_mul_f32 v19, v10, v15 // (op 1699) 

v_add_f32 v12, v16, v20 // (op 1696) 
v_add_f32 v13, v17, v21 // (op 1696) 
v_add_f32 v14, v18, v22 // (op 1696) 
v_add_f32 v15, v19, v23 // (op 1696) 




s_lshl_b32 s0, s4, 7 // (op 1697) 
v_add_u32 v0, v9, s0 // (op 1697) 


buffer_store_dwordx4 v[12:15], v0, s[44:47], 0 offen // (op 1697) Store value




s_lshl_b32 s0, s2, 8 // (op 1710) 
v_add_u32 v0, v6, s0 // (op 1710) 


buffer_load_dwordx4 v[12:15], v0, s[40:43], 0 offen // (op 1710) Load value

v_accvgpr_read v16, a8 // (op 1707) 
v_accvgpr_read v17, a9 // (op 1707) 
v_accvgpr_read v18, a10 // (op 1707) 
v_accvgpr_read v19, a11 // (op 1707) 
v_mul_f32 v20, v11, v16 // (op 1707) 
v_mul_f32 v21, v11, v17 // (op 1707) 
v_mul_f32 v22, v11, v18 // (op 1707) 
v_mul_f32 v23, v11, v19 // (op 1707) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v10, v12 // (op 1711) 
v_mul_f32 v17, v10, v13 // (op 1711) 
v_mul_f32 v18, v10, v14 // (op 1711) 
v_mul_f32 v19, v10, v15 // (op 1711) 

v_add_f32 v12, v16, v20 // (op 1708) 
v_add_f32 v13, v17, v21 // (op 1708) 
v_add_f32 v14, v18, v22 // (op 1708) 
v_add_f32 v15, v19, v23 // (op 1708) 




s_lshl_b32 s0, s4, 8 // (op 1709) 
v_add_u32 v0, v9, s0 // (op 1709) 


buffer_store_dwordx4 v[12:15], v0, s[44:47], 0 offen // (op 1709) Store value




s_lshl_b32 s0, s2, 7 // (op 1722) 
s_mul_i32 s1, 3, s0 // (op 1722) 

v_add_u32 v0, v6, s1 // (op 1722) 


buffer_load_dwordx4 v[12:15], v0, s[40:43], 0 offen // (op 1722) Load value

v_accvgpr_read v16, a12 // (op 1719) 
v_accvgpr_read v17, a13 // (op 1719) 
v_accvgpr_read v18, a14 // (op 1719) 
v_accvgpr_read v19, a15 // (op 1719) 
v_mul_f32 v20, v11, v16 // (op 1719) 
v_mul_f32 v21, v11, v17 // (op 1719) 
v_mul_f32 v22, v11, v18 // (op 1719) 
v_mul_f32 v23, v11, v19 // (op 1719) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v10, v12 // (op 1723) 
v_mul_f32 v17, v10, v13 // (op 1723) 
v_mul_f32 v18, v10, v14 // (op 1723) 
v_mul_f32 v19, v10, v15 // (op 1723) 

v_add_f32 v12, v16, v20 // (op 1720) 
v_add_f32 v13, v17, v21 // (op 1720) 
v_add_f32 v14, v18, v22 // (op 1720) 
v_add_f32 v15, v19, v23 // (op 1720) 




s_lshl_b32 s0, s4, 7 // (op 1721) 
s_mul_i32 s1, 3, s0 // (op 1721) 

v_add_u32 v0, v9, s1 // (op 1721) 


buffer_store_dwordx4 v[12:15], v0, s[44:47], 0 offen // (op 1721) Store value




s_lshl_b32 s0, s2, 9 // (op 1734) 
v_add_u32 v0, v6, s0 // (op 1734) 


buffer_load_dwordx4 v[12:15], v0, s[40:43], 0 offen // (op 1734) Load value

v_accvgpr_read v16, a16 // (op 1731) 
v_accvgpr_read v17, a17 // (op 1731) 
v_accvgpr_read v18, a18 // (op 1731) 
v_accvgpr_read v19, a19 // (op 1731) 
v_mul_f32 v20, v11, v16 // (op 1731) 
v_mul_f32 v21, v11, v17 // (op 1731) 
v_mul_f32 v22, v11, v18 // (op 1731) 
v_mul_f32 v23, v11, v19 // (op 1731) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v10, v12 // (op 1735) 
v_mul_f32 v17, v10, v13 // (op 1735) 
v_mul_f32 v18, v10, v14 // (op 1735) 
v_mul_f32 v19, v10, v15 // (op 1735) 

v_add_f32 v12, v16, v20 // (op 1732) 
v_add_f32 v13, v17, v21 // (op 1732) 
v_add_f32 v14, v18, v22 // (op 1732) 
v_add_f32 v15, v19, v23 // (op 1732) 




s_lshl_b32 s0, s4, 9 // (op 1733) 
v_add_u32 v0, v9, s0 // (op 1733) 


buffer_store_dwordx4 v[12:15], v0, s[44:47], 0 offen // (op 1733) Store value




s_lshl_b32 s0, s2, 7 // (op 1746) 
s_mul_i32 s1, 5, s0 // (op 1746) 

v_add_u32 v0, v6, s1 // (op 1746) 


buffer_load_dwordx4 v[12:15], v0, s[40:43], 0 offen // (op 1746) Load value

v_accvgpr_read v16, a20 // (op 1743) 
v_accvgpr_read v17, a21 // (op 1743) 
v_accvgpr_read v18, a22 // (op 1743) 
v_accvgpr_read v19, a23 // (op 1743) 
v_mul_f32 v20, v11, v16 // (op 1743) 
v_mul_f32 v21, v11, v17 // (op 1743) 
v_mul_f32 v22, v11, v18 // (op 1743) 
v_mul_f32 v23, v11, v19 // (op 1743) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v10, v12 // (op 1747) 
v_mul_f32 v17, v10, v13 // (op 1747) 
v_mul_f32 v18, v10, v14 // (op 1747) 
v_mul_f32 v19, v10, v15 // (op 1747) 

v_add_f32 v12, v16, v20 // (op 1744) 
v_add_f32 v13, v17, v21 // (op 1744) 
v_add_f32 v14, v18, v22 // (op 1744) 
v_add_f32 v15, v19, v23 // (op 1744) 




s_lshl_b32 s0, s4, 7 // (op 1745) 
s_mul_i32 s1, 5, s0 // (op 1745) 

v_add_u32 v0, v9, s1 // (op 1745) 


buffer_store_dwordx4 v[12:15], v0, s[44:47], 0 offen // (op 1745) Store value




s_lshl_b32 s0, s2, 7 // (op 1758) 
s_mul_i32 s1, 6, s0 // (op 1758) 

v_add_u32 v0, v6, s1 // (op 1758) 


buffer_load_dwordx4 v[12:15], v0, s[40:43], 0 offen // (op 1758) Load value

v_accvgpr_read v16, a24 // (op 1755) 
v_accvgpr_read v17, a25 // (op 1755) 
v_accvgpr_read v18, a26 // (op 1755) 
v_accvgpr_read v19, a27 // (op 1755) 
v_mul_f32 v20, v11, v16 // (op 1755) 
v_mul_f32 v21, v11, v17 // (op 1755) 
v_mul_f32 v22, v11, v18 // (op 1755) 
v_mul_f32 v23, v11, v19 // (op 1755) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v10, v12 // (op 1759) 
v_mul_f32 v17, v10, v13 // (op 1759) 
v_mul_f32 v18, v10, v14 // (op 1759) 
v_mul_f32 v19, v10, v15 // (op 1759) 

v_add_f32 v12, v16, v20 // (op 1756) 
v_add_f32 v13, v17, v21 // (op 1756) 
v_add_f32 v14, v18, v22 // (op 1756) 
v_add_f32 v15, v19, v23 // (op 1756) 




s_lshl_b32 s0, s4, 7 // (op 1757) 
s_mul_i32 s1, 6, s0 // (op 1757) 

v_add_u32 v0, v9, s1 // (op 1757) 


buffer_store_dwordx4 v[12:15], v0, s[44:47], 0 offen // (op 1757) Store value




s_lshl_b32 s0, s2, 7 // (op 1770) 
s_mul_i32 s1, 7, s0 // (op 1770) 

v_add_u32 v0, v6, s1 // (op 1770) 


buffer_load_dwordx4 v[12:15], v0, s[40:43], 0 offen // (op 1770) Load value

v_accvgpr_read v16, a28 // (op 1767) 
v_accvgpr_read v17, a29 // (op 1767) 
v_accvgpr_read v18, a30 // (op 1767) 
v_accvgpr_read v19, a31 // (op 1767) 
v_mul_f32 v20, v11, v16 // (op 1767) 
v_mul_f32 v21, v11, v17 // (op 1767) 
v_mul_f32 v22, v11, v18 // (op 1767) 
v_mul_f32 v23, v11, v19 // (op 1767) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v10, v12 // (op 1771) 
v_mul_f32 v17, v10, v13 // (op 1771) 
v_mul_f32 v18, v10, v14 // (op 1771) 
v_mul_f32 v19, v10, v15 // (op 1771) 

v_add_f32 v12, v16, v20 // (op 1768) 
v_add_f32 v13, v17, v21 // (op 1768) 
v_add_f32 v14, v18, v22 // (op 1768) 
v_add_f32 v15, v19, v23 // (op 1768) 




s_lshl_b32 s0, s4, 7 // (op 1769) 
s_mul_i32 s1, 7, s0 // (op 1769) 

v_add_u32 v0, v9, s1 // (op 1769) 


buffer_store_dwordx4 v[12:15], v0, s[44:47], 0 offen // (op 1769) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_ConditionalBottom_0_beta_123:
















s_endpgm  // End of RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2
.LRR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_end:
.size RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2, .LRR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_end-RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2
  .amdhsa_next_free_vgpr 148
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 36864 // lds bytes
  .amdhsa_accum_offset 116
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
    macrotile: [16, 256, 128]
    threads: [128, 1, 1]
    grid: [TilesX, TilesY, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 8]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2
    .symbol: RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 36864
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 64
    .vgpr_count: 114
    .agpr_count: 32
    .max_flat_workgroup_size: 128
    .workgroup_size: [128, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, {is-null: true}]
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

