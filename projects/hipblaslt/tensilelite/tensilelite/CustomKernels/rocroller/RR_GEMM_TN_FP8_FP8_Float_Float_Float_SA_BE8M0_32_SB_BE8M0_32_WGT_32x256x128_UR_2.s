// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2
.p2align 8
.type RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2,@function
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_exec_begin:


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


v_and_b32 v0, 48, v1 // (op 3717) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 3717) 


v_lshrrev_b32 v0, 7, v1 // (op 3889) 
v_and_b32 v4, 15, v1 // (op 3889) 
v_lshl_add_u32 v5, v0, 4, v4 // (op 3889) 


v_and_b32 v0, 48, v1 // (op 3889) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 3889) 


v_lshrrev_b32 v0, 7, v1 // (op 3661) 
v_and_b32 v5, 63, v1 // (op 3661) 
v_lshrrev_b32 v6, 4, v5 // (op 3661) 
v_lshlrev_b32 v7, 2, v6 // (op 3661) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 3661) 


v_lshl_add_u32 v0, s16, 5, v6 // (op 3661) 

v_lshrrev_b32 v6, 2, v5 // (op 3661) 

v_and_b32 v5, 3, v6 // (op 3661) 

v_and_b32 v6, 3, v1 // (op 3661) 
v_lshl_add_u32 v7, v5, 2, v6 // (op 3661) 


v_lshrrev_b32 v5, 6, v1 // (op 3661) 
v_and_b32 v6, 1, v5 // (op 3661) 

v_lshl_add_u32 v5, v6, 4, v7 // (op 3661) 


v_lshl_add_u32 v6, s17, 8, v5 // (op 3661) 

v_mul_lo_u32 v5, v6, s2 // (op 3661) 

v_add_lshl_u32 v6, v0, v5, 2 // (op 3661) 


v_lshrrev_b32 v0, 6, v1 // (op 3774) 
v_and_b32 v5, 1, v0 // (op 3774) 

v_and_b32 v0, 15, v1 // (op 3774) 
v_lshl_add_u32 v7, v5, 4, v0 // (op 3774) 


v_and_b32 v0, 48, v1 // (op 3774) 
v_lshl_add_u32 v5, v7, 7, v0 // (op 3774) 


v_lshrrev_b32 v0, 6, v1 // (op 3832) 
v_and_b32 v7, 1, v0 // (op 3832) 

v_and_b32 v0, 15, v1 // (op 3832) 
s_mul_i32 s1, s2, s9 // (op 3663) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 3663) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 3663) least significant
s_mul_hi_u32 s19, s2, s8 // (op 3663) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3663) most significant: sum
s_add_u32 s1, s1, s19 // (op 3663) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 3663) 

v_lshl_add_u32 v8, v7, 4, v0 // (op 3832) 


v_and_b32 v0, 63, v1 // (op 3832) 
v_lshrrev_b32 v7, 4, v0 // (op 3832) 

v_lshl_add_u32 v0, v8, 2, v7 // (op 3832) 


v_lshrrev_b32 v7, 7, v1 // (op 3848) 
v_and_b32 v8, 63, v1 // (op 3848) 
v_lshrrev_b32 v9, 4, v8 // (op 3848) 
v_lshlrev_b32 v10, 2, v9 // (op 3848) 

v_lshl_add_u32 v9, v7, 4, v10 // (op 3848) 


v_lshl_add_u32 v7, s16, 5, v9 // (op 3848) 

v_lshrrev_b32 v9, 2, v8 // (op 3848) 

v_and_b32 v8, 3, v9 // (op 3848) 

v_and_b32 v9, 3, v1 // (op 3848) 
v_lshl_add_u32 v10, v8, 2, v9 // (op 3848) 


v_lshrrev_b32 v8, 6, v1 // (op 3848) 
v_and_b32 v9, 1, v8 // (op 3848) 

v_lshl_add_u32 v8, v9, 4, v10 // (op 3848) 


v_lshl_add_u32 v9, s17, 8, v8 // (op 3848) 

v_mul_lo_u32 v8, v9, s4 // (op 3848) 

v_add_lshl_u32 v9, v7, v8, 2 // (op 3848) 


v_lshrrev_b32 v7, 6, v1 // (op 3933) 
v_and_b32 v8, 1, v7 // (op 3933) 

v_and_b32 v7, 15, v1 // (op 3933) 
v_lshl_add_u32 v10, v8, 4, v7 // (op 3933) 


v_and_b32 v7, 48, v1 // (op 3933) 
v_lshl_add_u32 v8, v10, 7, v7 // (op 3933) 


v_lshrrev_b32 v7, 6, v1 // (op 3962) 
v_and_b32 v10, 1, v7 // (op 3962) 

v_and_b32 v7, 15, v1 // (op 3962) 
s_mul_i32 s1, s4, s9 // (op 3850) most significant: low of low * high
v_lshl_add_u32 v11, v10, 4, v7 // (op 3962) 


v_and_b32 v7, 63, v1 // (op 3962) 
v_lshrrev_b32 v10, 4, v7 // (op 3962) 

v_lshl_add_u32 v7, v11, 2, v10 // (op 3962) 


s_mul_i32 s38, s5, s8 // (op 3850) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 3850) least significant
s_mul_hi_u32 s39, s4, s8 // (op 3850) most significant: high of low * low
s_add_u32 s1, s1, s38 // (op 3850) most significant: sum
s_add_u32 s1, s1, s39 // (op 3850) most significant: sum


s_lshl_b64 s[38:39], s[0:1], 2 // (op 3850) 

s_mov_b32 s40, s6 // (op 3663) reg expression
s_mov_b32 s41, s7 // (op 3663) reg expression
s_mov_b32 s42, s18 // (op 3663) reg expression
s_mov_b32 s43, 131072 // (op 3663) call()

s_mov_b32 s44, s10 // (op 3850) reg expression

s_mov_b32 s45, s11 // (op 3850) reg expression
s_mov_b32 s46, s38 // (op 3850) reg expression
s_mov_b32 s47, 131072 // (op 3850) call()


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
v_lshrrev_b32 v12, 3, v1 // (op 3428) 
v_lshl_add_u32 v13, s16, 5, v12 // (op 3428) 

v_and_b32 v12, -8, v1 // (op 3428) 
v_sub_u32 v14, v1, v12 // (op 3428) 

v_lshlrev_b32 v12, 4, v14 // (op 3428) 

v_mov_b32 v15, 0 // (op 3428) convert
v_mov_b32 v14, v12 // (op 3428) convert

v_mul_lo_u32 v17, v13, s13 // (op 3428) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 3428) most significant: high of low * low
v_mul_lo_u32 v16, v13, s12 // (op 3428) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 3428) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 3428) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 3428) most significant half




s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 3430) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 3430) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 3430) least significant
s_mul_hi_u32 s7, s12, s22 // (op 3430) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3430) most significant: sum
s_add_u32 s1, s1, s7 // (op 3430) most significant: sum


s_mov_b32 s48, s20 // (op 3430) reg expression
s_mov_b32 s49, s21 // (op 3430) reg expression
s_mov_b32 s50, s0 // (op 3430) reg expression
s_mov_b32 s51, 131072 // (op 3430) call()

v_lshrrev_b32 v12, 3, v1 // (op 3447) 
v_lshl_add_u32 v13, s16, 5, v12 // (op 3447) 

v_and_b32 v12, -8, v1 // (op 3447) 
v_sub_u32 v14, v1, v12 // (op 3447) 

v_lshlrev_b32 v12, 4, v14 // (op 3447) 

v_mov_b32 v15, 0 // (op 3447) convert
v_mov_b32 v14, v12 // (op 3447) convert

v_mul_lo_u32 v19, v13, s13 // (op 3447) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 3447) most significant: high of low * low
v_mul_lo_u32 v18, v13, s12 // (op 3447) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 3447) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v14 // (op 3447) least significant half
v_addc_co_u32 v19, s[0:1], v19, v15, s[0:1] // (op 3447) most significant half



s_mul_i32 s1, s12, s23 // (op 3449) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 3449) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 3449) least significant
s_mul_hi_u32 s7, s12, s22 // (op 3449) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3449) most significant: sum
s_add_u32 s1, s1, s7 // (op 3449) most significant: sum


s_mov_b32 s48, s20 // (op 3449) reg expression
s_mov_b32 s49, s21 // (op 3449) reg expression
s_mov_b32 s50, s0 // (op 3449) reg expression
s_mov_b32 s51, 131072 // (op 3449) call()

v_and_b32 v12, -8, v1 // (op 3492) 
v_sub_u32 v13, v1, v12 // (op 3492) 

v_lshlrev_b32 v12, 4, v13 // (op 3492) 

v_mov_b32 v15, 0 // (op 3492) convert
v_mov_b32 v14, v12 // (op 3492) convert

v_lshrrev_b32 v12, 3, v1 // (op 3492) 
v_lshl_add_u32 v13, s17, 8, v12 // (op 3492) 

v_mul_lo_u32 v21, v13, s25 // (op 3492) most significant: low of low * high
v_mul_hi_u32 v12, v13, s24 // (op 3492) most significant: high of low * low
v_mul_lo_u32 v20, v13, s24 // (op 3492) least significant: low of low * low
v_add_u32 v21, v21, v12 // (op 3492) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v20 // (op 3492) least significant half
v_addc_co_u32 v13, s[0:1], v15, v21, s[0:1] // (op 3492) most significant half



s_mul_i32 s1, s24, s9 // (op 3494) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 3494) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 3494) least significant
s_mul_hi_u32 s7, s24, s8 // (op 3494) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3494) most significant: sum
s_add_u32 s1, s1, s7 // (op 3494) most significant: sum


s_mov_b32 s52, s26 // (op 3494) reg expression
s_mov_b32 s53, s27 // (op 3494) reg expression
s_mov_b32 s54, s0 // (op 3494) reg expression
s_mov_b32 s55, 131072 // (op 3494) call()

v_and_b32 v14, -8, v1 // (op 3511) 
v_sub_u32 v15, v1, v14 // (op 3511) 

v_lshlrev_b32 v14, 4, v15 // (op 3511) 

v_mov_b32 v21, 0 // (op 3511) convert
v_mov_b32 v20, v14 // (op 3511) convert

v_lshrrev_b32 v14, 3, v1 // (op 3511) 
v_lshl_add_u32 v15, s17, 8, v14 // (op 3511) 

v_mul_lo_u32 v23, v15, s25 // (op 3511) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 3511) most significant: high of low * low
v_mul_lo_u32 v22, v15, s24 // (op 3511) least significant: low of low * low
v_add_u32 v23, v23, v14 // (op 3511) most significant: sum


v_add_co_u32 v14, s[0:1], v20, v22 // (op 3511) least significant half
v_addc_co_u32 v15, s[0:1], v21, v23, s[0:1] // (op 3511) most significant half



s_mul_i32 s1, s24, s9 // (op 3513) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 3513) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 3513) least significant
s_mul_hi_u32 s7, s24, s8 // (op 3513) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3513) most significant: sum
s_add_u32 s1, s1, s7 // (op 3513) most significant: sum


s_mov_b32 s52, s26 // (op 3513) reg expression
s_mov_b32 s53, s27 // (op 3513) reg expression
s_mov_b32 s54, s0 // (op 3513) reg expression
s_mov_b32 s55, 131072 // (op 3513) call()

v_lshrrev_b32 v20, 7, v1 // (op 3556) 
v_and_b32 v21, 15, v1 // (op 3556) 
v_lshl_add_u32 v22, v20, 4, v21 // (op 3556) 


v_lshl_add_u32 v20, s16, 5, v22 // (op 3556) 

v_and_b32 v21, 63, v1 // (op 3556) 
v_lshrrev_b32 v22, 4, v21 // (op 3556) 

v_mul_lo_u32 v21, v20, s28 // (op 3556) 
v_add_u32 v21, v21, v22 // (op 3556) 


s_mul_i32 s1, s28, s23 // (op 3558) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 3558) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 3558) least significant
s_mul_hi_u32 s7, s28, s22 // (op 3558) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3558) most significant: sum
s_add_u32 s1, s1, s7 // (op 3558) most significant: sum


s_mov_b32 s56, s30 // (op 3558) reg expression
s_mov_b32 s57, s31 // (op 3558) reg expression
s_mov_b32 s58, s0 // (op 3558) reg expression
s_mov_b32 s59, 131072 // (op 3558) call()

v_lshrrev_b32 v20, 7, v1 // (op 3575) 
v_and_b32 v22, 15, v1 // (op 3575) 
v_lshl_add_u32 v23, v20, 4, v22 // (op 3575) 


v_lshl_add_u32 v20, s16, 5, v23 // (op 3575) 

v_and_b32 v22, 63, v1 // (op 3575) 
v_lshrrev_b32 v23, 4, v22 // (op 3575) 

v_mul_lo_u32 v22, v20, s28 // (op 3575) 
v_add_u32 v22, v22, v23 // (op 3575) 


s_mul_i32 s1, s28, s23 // (op 3577) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 3577) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 3577) least significant
s_mul_hi_u32 s7, s28, s22 // (op 3577) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3577) most significant: sum
s_add_u32 s1, s1, s7 // (op 3577) most significant: sum


s_mov_b32 s56, s30 // (op 3577) reg expression
s_mov_b32 s57, s31 // (op 3577) reg expression
s_mov_b32 s58, s0 // (op 3577) reg expression
s_mov_b32 s59, 131072 // (op 3577) call()

v_lshl_add_u32 v20, s17, 8, v1 // (op 3620) 
v_mul_lo_u32 v23, v20, s32 // (op 3620) 

s_mul_i32 s1, s32, s9 // (op 3622) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 3622) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 3622) least significant
s_mul_hi_u32 s7, s32, s8 // (op 3622) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3622) most significant: sum
s_add_u32 s1, s1, s7 // (op 3622) most significant: sum


s_mov_b32 s60, s34 // (op 3622) reg expression
s_mov_b32 s61, s35 // (op 3622) reg expression
s_mov_b32 s62, s0 // (op 3622) reg expression
s_mov_b32 s63, 131072 // (op 3622) call()

v_lshl_add_u32 v20, s17, 8, v1 // (op 3639) 
v_mul_lo_u32 v24, v20, s32 // (op 3639) 

s_mul_i32 s1, s32, s9 // (op 3641) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 3641) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 3641) least significant
s_mul_hi_u32 s7, s32, s8 // (op 3641) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3641) most significant: sum
s_add_u32 s1, s1, s7 // (op 3641) most significant: sum


s_mov_b32 s60, s34 // (op 3641) reg expression
s_mov_b32 s61, s35 // (op 3641) reg expression
s_mov_b32 s62, s0 // (op 3641) reg expression
s_mov_b32 s63, 131072 // (op 3641) call()




buffer_load_ubyte v20, v21, s[56:59], 0 offen offset: 0 // (op 3108) Load value
v_and_b32 v25, -8, v1 // (op 3757) 
v_sub_u32 v26, v1, v25 // (op 3757) 

v_lshlrev_b32 v25, 4, v26 // (op 3757) 

v_lshrrev_b32 v26, 3, v1 // (op 3757) 
v_lshl_add_u32 v27, v26, 7, v25 // (op 3757) 


s_nop 0
v_readfirstlane_b32 s0, v27 // (op 3757) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 3370) 
v_mov_b64 v[26:27], s[6:7] // (op 3370) convert


v_mov_b64 v[28:29], v[12:13] // (op 3370) 
s_mov_b32 m0, s0 // (op 3370) reg expression
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3370) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 3370) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3370) most significant half

s_add_u32 m0, m0, 4096 // (op 3370) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3370) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 3370) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3370) most significant half

s_add_u32 m0, m0, 4096 // (op 3370) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3370) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 3370) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3370) most significant half

s_add_u32 m0, m0, 4096 // (op 3370) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3370) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 3370) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3370) most significant half

s_add_u32 m0, m0, 4096 // (op 3370) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3370) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 3370) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3370) most significant half

s_add_u32 m0, m0, 4096 // (op 3370) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3370) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 3370) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3370) most significant half

s_add_u32 m0, m0, 4096 // (op 3370) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3370) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 3370) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3370) most significant half

s_add_u32 m0, m0, 4096 // (op 3370) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3370) Load value direct to lds






buffer_load_dword v25, v23, s[60:63], 0 offen // (op 3115) Load value
v_and_b32 v26, -8, v1 // (op 3705) 
v_sub_u32 v27, v1, v26 // (op 3705) 

v_lshlrev_b32 v26, 4, v27 // (op 3705) 

v_lshrrev_b32 v27, 3, v1 // (op 3705) 
v_lshl_add_u32 v28, v27, 7, v26 // (op 3705) 


s_nop 0
v_readfirstlane_b32 s0, v28 // (op 3705) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s1, 32768, s0 // (op 3375) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 3375) reg expression
buffer_load_dwordx4 v16, s[48:51], 0 offen offset: 0 lds // (op 3375) Load value direct to lds





buffer_load_ubyte v26, v21, s[56:59], 0 offen offset:4 // (op 3122) Load value

v_and_b32 v21, -8, v1 // (op 3916) 
v_sub_u32 v27, v1, v21 // (op 3916) 

v_lshlrev_b32 v21, 4, v27 // (op 3916) 

v_lshrrev_b32 v27, 3, v1 // (op 3916) 
v_lshl_add_u32 v28, v27, 7, v21 // (op 3916) 


s_nop 0
v_readfirstlane_b32 s0, v28 // (op 3916) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 3380) 
v_mov_b64 v[28:29], s[6:7] // (op 3380) convert


v_mov_b64 v[30:31], v[12:13] // (op 3380) 
s_add_u32 s1, 36864, s0 // (op 3380) 
s_mov_b32 m0, s1 // (op 3380) reg expression
s_mov_b32 s6, 128 // (op 3380) call()
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 3380) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v28 // (op 3380) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 3380) most significant half

s_add_u32 m0, m0, 4096 // (op 3380) 
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 3380) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v28 // (op 3380) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 3380) most significant half

s_add_u32 m0, m0, 4096 // (op 3380) 
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 3380) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v28 // (op 3380) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 3380) most significant half

s_add_u32 m0, m0, 4096 // (op 3380) 
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 3380) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v28 // (op 3380) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 3380) most significant half

s_add_u32 m0, m0, 4096 // (op 3380) 
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 3380) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v28 // (op 3380) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 3380) most significant half

s_add_u32 m0, m0, 4096 // (op 3380) 
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 3380) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v28 // (op 3380) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 3380) most significant half

s_add_u32 m0, m0, 4096 // (op 3380) 
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 3380) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v28 // (op 3380) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 3380) most significant half

s_add_u32 m0, m0, 4096 // (op 3380) 
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 3380) Load value direct to lds









buffer_load_dword v21, v23, s[60:63], 0 offen offset:4 // (op 3129) Load value

v_and_b32 v23, -8, v1 // (op 3877) 
v_sub_u32 v12, v1, v23 // (op 3877) 

v_lshlrev_b32 v13, 4, v12 // (op 3877) 

v_lshrrev_b32 v12, 3, v1 // (op 3877) 
v_lshl_add_u32 v23, v12, 7, v13 // (op 3877) 


s_nop 0
v_readfirstlane_b32 s0, v23 // (op 3877) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s1, 69632, s0 // (op 3382) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 3382) reg expression
s_mov_b32 s6, 128 // (op 3382) call()
buffer_load_dwordx4 v16, s[48:51], s6 offen offset: 0 lds // (op 3382) Load value direct to lds




v_lshlrev_b32 v23, 2, v1 // (op 3815) 



v_add_u32 v12, 73728, v23 // (op 3135) 
s_waitcnt vmcnt(12)
ds_write_b32 v12, v25 // (op 3135) Store local data 


s_barrier  // (op 4037) 



ds_read_b128 v[28:31], v5 // (op 3153) Load local data 
ds_read_b128 v[32:35], v5 offset:64 // (op 3153) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 4053) 



v_add_u32 v23, 73728, v0 // (op 3193) 
ds_read_u8 v27, v23 // (op 3193) Load local data 




ds_read_b128 v[36:39], v5 offset:4096 // (op 3158) Load local data 
ds_read_b128 v[40:43], v5 offset:4160 // (op 3158) Load local data 



v_add_u32 v23, 73856, v0 // (op 3198) 
ds_read_u8 v12, v23 // (op 3198) Load local data 




ds_read_b128 v[44:47], v5 offset:8192 // (op 3163) Load local data 
ds_read_b128 v[48:51], v5 offset:8256 // (op 3163) Load local data 



v_add_u32 v13, 73984, v0 // (op 3203) 
ds_read_u8 v23, v13 // (op 3203) Load local data 




ds_read_b128 v[52:55], v5 offset:12288 // (op 3168) Load local data 
ds_read_b128 v[56:59], v5 offset:12352 // (op 3168) Load local data 



v_add_u32 v13, 74112, v0 // (op 3208) 
ds_read_u8 v16, v13 // (op 3208) Load local data 




ds_read_b128 v[60:63], v5 offset:16384 // (op 3173) Load local data 
ds_read_b128 v[64:67], v5 offset:16448 // (op 3173) Load local data 



v_add_u32 v13, 74240, v0 // (op 3213) 
ds_read_u8 v17, v13 // (op 3213) Load local data 




ds_read_b128 v[68:71], v5 offset:20480 // (op 3178) Load local data 
ds_read_b128 v[72:75], v5 offset:20544 // (op 3178) Load local data 



v_add_u32 v13, 74368, v0 // (op 3218) 
ds_read_u8 v76, v13 // (op 3218) Load local data 




ds_read_b128 v[78:81], v5 offset:24576 // (op 3183) Load local data 
ds_read_b128 v[82:85], v5 offset:24640 // (op 3183) Load local data 



v_add_u32 v13, 74496, v0 // (op 3223) 
ds_read_u8 v77, v13 // (op 3223) Load local data 




ds_read_b128 v[86:89], v5 offset:28672 // (op 3188) Load local data 
ds_read_b128 v[90:93], v5 offset:28736 // (op 3188) Load local data 



v_add_u32 v13, 74624, v0 // (op 3228) 
ds_read_u8 v94, v13 // (op 3228) Load local data 

s_waitcnt vmcnt(11)
s_barrier  // (op 4021) 



ds_read_b128 v[96:99], v3 offset:32768 // (op 3149) Load local data 
ds_read_b128 v[100:103], v3 offset:32832 // (op 3149) Load local data 
v_and_b32 v13, -8, v1 // (op 3677) 
v_sub_u32 v95, v1, v13 // (op 3677) 

v_lshlrev_b32 v13, 4, v95 // (op 3677) 

v_lshrrev_b32 v95, 3, v1 // (op 3677) 
v_lshl_add_u32 v104, v95, 7, v13 // (op 3677) 


s_nop 0
v_readfirstlane_b32 s0, v104 // (op 3677) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v13, -8, v1 // (op 3729) 
v_sub_u32 v95, v1, v13 // (op 3729) 

v_lshlrev_b32 v13, 4, v95 // (op 3729) 

v_lshrrev_b32 v95, 3, v1 // (op 3729) 
v_lshl_add_u32 v104, v95, 7, v13 // (op 3729) 


s_nop 0
v_readfirstlane_b32 s1, v104 // (op 3729) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v13, 2, v1 // (op 3786) 
v_and_b32 v95, -8, v1 // (op 3862) 
v_sub_u32 v104, v1, v95 // (op 3862) 

v_lshlrev_b32 v95, 4, v104 // (op 3862) 

v_lshrrev_b32 v104, 3, v1 // (op 3862) 
v_lshl_add_u32 v105, v104, 7, v95 // (op 3862) 


s_nop 0
v_readfirstlane_b32 s6, v105 // (op 3862) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v95, -8, v1 // (op 3901) 
v_sub_u32 v104, v1, v95 // (op 3901) 

v_lshlrev_b32 v95, 4, v104 // (op 3901) 

v_lshrrev_b32 v104, 3, v1 // (op 3901) 
v_lshl_add_u32 v105, v104, 7, v95 // (op 3901) 


s_nop 0
v_readfirstlane_b32 s7, v105 // (op 3901) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v95, 2, v1 // (op 3945) 
s_mov_b32 s10, 0 // (op 1244) call()
s_ashr_i64 s[14:15], s[36:37], 7 // (op 1243) 
s_mov_b32 s11, s14 // (op 1243) convert

s_and_b32 s14, s11, 4294967294 // (op 1243) 

s_cmp_lt_u32 s10, s14 // (op 1243) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopBottom_KLoop_1243 // (op 1243) Condition: Top (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopBottom_KLoop_1243 if false)
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopTop_KLoop_1243:
s_add_u32 s10, s10, 2 // (op 1245) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[96:103], v[28:35], a[0:3], v20, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 82) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[96:103], v[36:43], a[4:7], v20, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 343) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[96:103], v[44:51], a[8:11], v20, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 499) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[96:103], v[52:59], a[12:15], v20, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 655) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[96:103], v[60:67], a[16:19], v20, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 811) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[96:103], v[68:75], a[20:23], v20, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 967) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[96:103], v[78:85], a[24:27], v20, v77 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1123) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[96:103], v[86:93], a[28:31], v20, v94 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1279) 


v_mov_b32 v104, v22 // (op 75) 

buffer_load_ubyte v20, v104, s[56:59], 0 offen offset:8 // (op 75) Load value
s_barrier  // (op 4040) 

s_lshl_b64 s[14:15], s[24:25], 5 // (op 3342) 
v_mov_b64 v[106:107], s[14:15] // (op 3342) convert


v_mov_b64 v[108:109], v[14:15] // (op 3342) 
v_mov_b64 v[110:111], v[108:109] // (op 3342) 
s_mov_b32 m0, s1 // (op 3342) reg expression
s_mov_b32 s11, 256 // (op 3342) call()
buffer_load_dwordx4 v110, s[52:55], s11 offen offset: 0 lds // (op 3342) Load value direct to lds
v_add_co_u32 v110, s[14:15], v110, v106 // (op 3342) least significant half
v_addc_co_u32 v111, s[14:15], 0, v111, s[14:15] // (op 3342) most significant half

s_add_u32 m0, m0, 4096 // (op 3342) 
buffer_load_dwordx4 v110, s[52:55], s11 offen offset: 0 lds // (op 3342) Load value direct to lds
v_add_co_u32 v110, s[14:15], v110, v106 // (op 3342) least significant half
v_addc_co_u32 v111, s[14:15], 0, v111, s[14:15] // (op 3342) most significant half

s_add_u32 m0, m0, 4096 // (op 3342) 
buffer_load_dwordx4 v110, s[52:55], s11 offen offset: 0 lds // (op 3342) Load value direct to lds
v_add_co_u32 v110, s[14:15], v110, v106 // (op 3342) least significant half
v_addc_co_u32 v111, s[14:15], 0, v111, s[14:15] // (op 3342) most significant half

s_add_u32 m0, m0, 4096 // (op 3342) 
buffer_load_dwordx4 v110, s[52:55], s11 offen offset: 0 lds // (op 3342) Load value direct to lds
v_add_co_u32 v110, s[14:15], v110, v106 // (op 3342) least significant half
v_addc_co_u32 v111, s[14:15], 0, v111, s[14:15] // (op 3342) most significant half

s_add_u32 m0, m0, 4096 // (op 3342) 
buffer_load_dwordx4 v110, s[52:55], s11 offen offset: 0 lds // (op 3342) Load value direct to lds
v_add_co_u32 v110, s[14:15], v110, v106 // (op 3342) least significant half
v_addc_co_u32 v111, s[14:15], 0, v111, s[14:15] // (op 3342) most significant half

s_add_u32 m0, m0, 4096 // (op 3342) 
buffer_load_dwordx4 v110, s[52:55], s11 offen offset: 0 lds // (op 3342) Load value direct to lds
v_add_co_u32 v110, s[14:15], v110, v106 // (op 3342) least significant half
v_addc_co_u32 v111, s[14:15], 0, v111, s[14:15] // (op 3342) most significant half

s_add_u32 m0, m0, 4096 // (op 3342) 
buffer_load_dwordx4 v110, s[52:55], s11 offen offset: 0 lds // (op 3342) Load value direct to lds
v_add_co_u32 v110, s[14:15], v110, v106 // (op 3342) least significant half
v_addc_co_u32 v111, s[14:15], 0, v111, s[14:15] // (op 3342) most significant half

s_add_u32 m0, m0, 4096 // (op 3342) 
buffer_load_dwordx4 v110, s[52:55], s11 offen offset: 0 lds // (op 3342) Load value direct to lds





v_mov_b32 v105, v24 // (op 21) 

buffer_load_dword v25, v105, s[60:63], 0 offen offset:8 // (op 21) Load value
s_barrier  // (op 4024) 


v_mov_b64 v[106:107], v[18:19] // (op 3337) 
s_add_u32 s11, 32768, s0 // (op 3337) 

s_mov_b32 m0, s11 // (op 3337) reg expression
s_mov_b32 s14, 256 // (op 3337) call()
buffer_load_dwordx4 v106, s[48:51], s14 offen offset: 0 lds // (op 3337) Load value direct to lds


s_barrier  // (op 4085) 



v_add_u32 v110, 74752, v95 // (op 236) 
s_waitcnt vmcnt(12)
ds_write_b32 v110, v21 // (op 236) Store local data 

s_barrier  // (op 4076) 



ds_read_b128 v[28:31], v8 offset:36864 // (op 229) Load local data 
ds_read_b128 v[32:35], v8 offset:36928 // (op 229) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 4082) 



v_add_u32 v110, 74752, v7 // (op 233) 
ds_read_u8 v27, v110 // (op 233) Load local data 




ds_read_b128 v[36:39], v8 offset:40960 // (op 348) Load local data 
ds_read_b128 v[40:43], v8 offset:41024 // (op 348) Load local data 



v_add_u32 v110, 74880, v7 // (op 351) 
ds_read_u8 v12, v110 // (op 351) Load local data 




ds_read_b128 v[44:47], v8 offset:45056 // (op 504) Load local data 
ds_read_b128 v[48:51], v8 offset:45120 // (op 504) Load local data 



v_add_u32 v110, 75008, v7 // (op 507) 
ds_read_u8 v23, v110 // (op 507) Load local data 




ds_read_b128 v[52:55], v8 offset:49152 // (op 660) Load local data 
ds_read_b128 v[56:59], v8 offset:49216 // (op 660) Load local data 



v_add_u32 v110, 75136, v7 // (op 663) 
ds_read_u8 v16, v110 // (op 663) Load local data 




ds_read_b128 v[60:63], v8 offset:53248 // (op 816) Load local data 
ds_read_b128 v[64:67], v8 offset:53312 // (op 816) Load local data 



v_add_u32 v110, 75264, v7 // (op 819) 
ds_read_u8 v17, v110 // (op 819) Load local data 




ds_read_b128 v[68:71], v8 offset:57344 // (op 972) Load local data 
ds_read_b128 v[72:75], v8 offset:57408 // (op 972) Load local data 



v_add_u32 v110, 75392, v7 // (op 975) 
ds_read_u8 v76, v110 // (op 975) Load local data 




ds_read_b128 v[78:81], v8 offset:61440 // (op 1128) Load local data 
ds_read_b128 v[82:85], v8 offset:61504 // (op 1128) Load local data 



v_add_u32 v110, 75520, v7 // (op 1131) 
ds_read_u8 v77, v110 // (op 1131) Load local data 




v_add_u32 v110, 65536, v8 // (op 1284) 
ds_read_b128 v[86:89], v110 // (op 1284) Load local data 

v_add_u32 v110, 65600, v8 // (op 1284) 
ds_read_b128 v[90:93], v110 // (op 1284) Load local data 




v_add_u32 v110, 75648, v7 // (op 1287) 
ds_read_u8 v94, v110 // (op 1287) Load local data 

s_barrier  // (op 4160) 
s_waitcnt vmcnt(11)
s_barrier  // (op 4070) 



v_add_u32 v110, 69632, v4 // (op 227) 
ds_read_b128 v[96:99], v110 // (op 227) Load local data 

v_add_u32 v110, 69696, v4 // (op 227) 
ds_read_b128 v[100:103], v110 // (op 227) Load local data 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[96:103], v[28:35], a[0:3], v26, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 234) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[96:103], v[36:43], a[4:7], v26, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 344) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[96:103], v[44:51], a[8:11], v26, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 500) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[96:103], v[52:59], a[12:15], v26, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 656) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[96:103], v[60:67], a[16:19], v26, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 812) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[96:103], v[68:75], a[20:23], v26, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 968) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[96:103], v[78:85], a[24:27], v26, v77 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1124) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[96:103], v[86:93], a[28:31], v26, v94 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1280) 



buffer_load_ubyte v26, v104, s[56:59], 0 offen offset:12 // (op 231) Load value

s_barrier  // (op 4079) 
v_add_u32 v22, 8, v22 // (op 3567) 

s_lshl_b64 s[14:15], s[24:25], 5 // (op 3365) 
v_mov_b64 v[110:111], s[14:15] // (op 3365) convert


v_mov_b64 v[112:113], v[108:109] // (op 3365) 
s_add_u32 s11, 36864, s7 // (op 3365) 
s_mov_b32 m0, s11 // (op 3365) reg expression
s_mov_b32 s14, 384 // (op 3365) call()
buffer_load_dwordx4 v112, s[52:55], s14 offen offset: 0 lds // (op 3365) Load value direct to lds
v_add_co_u32 v112, s[18:19], v112, v110 // (op 3365) least significant half
v_addc_co_u32 v113, s[18:19], 0, v113, s[18:19] // (op 3365) most significant half

s_add_u32 m0, m0, 4096 // (op 3365) 
buffer_load_dwordx4 v112, s[52:55], s14 offen offset: 0 lds // (op 3365) Load value direct to lds
v_add_co_u32 v112, s[18:19], v112, v110 // (op 3365) least significant half
v_addc_co_u32 v113, s[18:19], 0, v113, s[18:19] // (op 3365) most significant half

s_add_u32 m0, m0, 4096 // (op 3365) 
buffer_load_dwordx4 v112, s[52:55], s14 offen offset: 0 lds // (op 3365) Load value direct to lds
v_add_co_u32 v112, s[18:19], v112, v110 // (op 3365) least significant half
v_addc_co_u32 v113, s[18:19], 0, v113, s[18:19] // (op 3365) most significant half

s_add_u32 m0, m0, 4096 // (op 3365) 
buffer_load_dwordx4 v112, s[52:55], s14 offen offset: 0 lds // (op 3365) Load value direct to lds
v_add_co_u32 v112, s[18:19], v112, v110 // (op 3365) least significant half
v_addc_co_u32 v113, s[18:19], 0, v113, s[18:19] // (op 3365) most significant half

s_add_u32 m0, m0, 4096 // (op 3365) 
buffer_load_dwordx4 v112, s[52:55], s14 offen offset: 0 lds // (op 3365) Load value direct to lds
v_add_co_u32 v112, s[18:19], v112, v110 // (op 3365) least significant half
v_addc_co_u32 v113, s[18:19], 0, v113, s[18:19] // (op 3365) most significant half

s_add_u32 m0, m0, 4096 // (op 3365) 
buffer_load_dwordx4 v112, s[52:55], s14 offen offset: 0 lds // (op 3365) Load value direct to lds
v_add_co_u32 v112, s[18:19], v112, v110 // (op 3365) least significant half
v_addc_co_u32 v113, s[18:19], 0, v113, s[18:19] // (op 3365) most significant half

s_add_u32 m0, m0, 4096 // (op 3365) 
buffer_load_dwordx4 v112, s[52:55], s14 offen offset: 0 lds // (op 3365) Load value direct to lds
v_add_co_u32 v112, s[18:19], v112, v110 // (op 3365) least significant half
v_addc_co_u32 v113, s[18:19], 0, v113, s[18:19] // (op 3365) most significant half

s_add_u32 m0, m0, 4096 // (op 3365) 
buffer_load_dwordx4 v112, s[52:55], s14 offen offset: 0 lds // (op 3365) Load value direct to lds








v_add_co_u32 v14, vcc, 256, v14 // (op 3503) least significant half
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 3503) most significant half
buffer_load_dword v21, v105, s[60:63], 0 offen offset:12 // (op 238) Load value

s_barrier  // (op 4073) 
v_add_u32 v24, 8, v24 // (op 3631) 


s_add_u32 s11, 69632, s6 // (op 3360) 

s_mov_b32 m0, s11 // (op 3360) reg expression
s_mov_b32 s14, 384 // (op 3360) call()
buffer_load_dwordx4 v106, s[48:51], s14 offen offset: 0 lds // (op 3360) Load value direct to lds



s_barrier  // (op 4056) 
v_add_co_u32 v18, vcc, 256, v18 // (op 3439) least significant half



v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 3439) most significant half
v_add_u32 v104, 73728, v13 // (op 26) 
s_waitcnt vmcnt(12)
ds_write_b32 v104, v25 // (op 26) Store local data 

s_barrier  // (op 4046) 



ds_read_b128 v[28:31], v5 // (op 70) Load local data 
ds_read_b128 v[32:35], v5 offset:64 // (op 70) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 4063) 



v_add_u32 v104, 73728, v0 // (op 79) 
ds_read_u8 v27, v104 // (op 79) Load local data 




ds_read_b128 v[36:39], v5 offset:4096 // (op 339) Load local data 
ds_read_b128 v[40:43], v5 offset:4160 // (op 339) Load local data 



v_add_u32 v104, 73856, v0 // (op 342) 
ds_read_u8 v12, v104 // (op 342) Load local data 




ds_read_b128 v[44:47], v5 offset:8192 // (op 495) Load local data 
ds_read_b128 v[48:51], v5 offset:8256 // (op 495) Load local data 



v_add_u32 v104, 73984, v0 // (op 498) 
ds_read_u8 v23, v104 // (op 498) Load local data 




ds_read_b128 v[52:55], v5 offset:12288 // (op 651) Load local data 
ds_read_b128 v[56:59], v5 offset:12352 // (op 651) Load local data 



v_add_u32 v104, 74112, v0 // (op 654) 
ds_read_u8 v16, v104 // (op 654) Load local data 




ds_read_b128 v[60:63], v5 offset:16384 // (op 807) Load local data 
ds_read_b128 v[64:67], v5 offset:16448 // (op 807) Load local data 



v_add_u32 v104, 74240, v0 // (op 810) 
ds_read_u8 v17, v104 // (op 810) Load local data 




ds_read_b128 v[68:71], v5 offset:20480 // (op 963) Load local data 
ds_read_b128 v[72:75], v5 offset:20544 // (op 963) Load local data 



v_add_u32 v104, 74368, v0 // (op 966) 
ds_read_u8 v76, v104 // (op 966) Load local data 




ds_read_b128 v[78:81], v5 offset:24576 // (op 1119) Load local data 
ds_read_b128 v[82:85], v5 offset:24640 // (op 1119) Load local data 



v_add_u32 v104, 74496, v0 // (op 1122) 
ds_read_u8 v77, v104 // (op 1122) Load local data 




ds_read_b128 v[86:89], v5 offset:28672 // (op 1275) Load local data 
ds_read_b128 v[90:93], v5 offset:28736 // (op 1275) Load local data 



v_add_u32 v104, 74624, v0 // (op 1278) 
ds_read_u8 v94, v104 // (op 1278) Load local data 

s_waitcnt vmcnt(11)
s_barrier  // (op 4030) 



ds_read_b128 v[96:99], v3 offset:32768 // (op 65) Load local data 
ds_read_b128 v[100:103], v3 offset:32832 // (op 65) Load local data 
s_ashr_i64 s[14:15], s[36:37], 7 // (op 1243) 
s_mov_b32 s11, s14 // (op 1243) convert

s_and_b32 s14, s11, 4294967294 // (op 1243) 

s_cmp_lt_u32 s10, s14 // (op 1243) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopTop_KLoop_1243 // (op 1243) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopTop_KLoop_1243 if true)
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopBottom_KLoop_1243:




















s_barrier  // (op 4144) 
v_lshrrev_b32 v95, 3, v1 // (op 3395) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 3395) 
v_lshl_add_u32 v12, s16, 5, v95 // (op 3395) 

s_lshr_b32 s6, s0, 1 // (op 3395) 

v_and_b32 v95, -8, v1 // (op 3395) 
v_sub_u32 v13, v1, v95 // (op 3395) 

v_lshlrev_b32 v95, 4, v13 // (op 3395) 

v_lshl_add_u32 v13, s6, 8, v95 // (op 3395) 


v_mov_b32 v15, 0 // (op 3395) convert
v_mov_b32 v14, v13 // (op 3395) convert

v_mul_lo_u32 v19, v12, s13 // (op 3395) most significant: low of low * high
v_mul_hi_u32 v13, v12, s12 // (op 3395) most significant: high of low * low
v_mul_lo_u32 v18, v12, s12 // (op 3395) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 3395) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v14 // (op 3395) least significant half
v_addc_co_u32 v19, s[0:1], v19, v15, s[0:1] // (op 3395) most significant half



s_mul_i32 s1, s12, s23 // (op 3397) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 3397) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 3397) least significant
s_mul_hi_u32 s7, s12, s22 // (op 3397) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3397) most significant: sum
s_add_u32 s1, s1, s7 // (op 3397) most significant: sum


s_mov_b32 s48, s20 // (op 3397) reg expression
s_mov_b32 s49, s21 // (op 3397) reg expression
s_mov_b32 s50, s0 // (op 3397) reg expression
s_mov_b32 s51, 131072 // (op 3397) call()

v_and_b32 v20, -8, v1 // (op 3469) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 3469) 
v_sub_u32 v95, v1, v20 // (op 3469) 

s_lshr_b32 s6, s0, 1 // (op 3469) 

v_lshlrev_b32 v20, 4, v95 // (op 3469) 

v_lshl_add_u32 v95, s6, 8, v20 // (op 3469) 


v_mov_b32 v13, 0 // (op 3469) convert
v_mov_b32 v12, v95 // (op 3469) convert

v_lshrrev_b32 v20, 3, v1 // (op 3469) 
v_lshl_add_u32 v95, s17, 8, v20 // (op 3469) 

v_mul_lo_u32 v15, v95, s25 // (op 3469) most significant: low of low * high
v_mul_hi_u32 v20, v95, s24 // (op 3469) most significant: high of low * low
v_mul_lo_u32 v14, v95, s24 // (op 3469) least significant: low of low * low
v_add_u32 v15, v15, v20 // (op 3469) most significant: sum


v_add_co_u32 v22, s[0:1], v12, v14 // (op 3469) least significant half
v_addc_co_u32 v23, s[0:1], v13, v15, s[0:1] // (op 3469) most significant half



s_mul_i32 s1, s24, s9 // (op 3471) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 3471) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 3471) least significant
s_mul_hi_u32 s7, s24, s8 // (op 3471) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3471) most significant: sum
s_add_u32 s1, s1, s7 // (op 3471) most significant: sum


s_mov_b32 s52, s26 // (op 3471) reg expression
s_mov_b32 s53, s27 // (op 3471) reg expression
s_mov_b32 s54, s0 // (op 3471) reg expression
s_mov_b32 s55, 131072 // (op 3471) call()

v_lshrrev_b32 v20, 7, v1 // (op 3533) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 3533) 
v_and_b32 v95, 15, v1 // (op 3533) 
s_lshr_b32 s6, s0, 1 // (op 3533) 

v_lshl_add_u32 v12, v20, 4, v95 // (op 3533) 


v_lshl_add_u32 v20, s16, 5, v12 // (op 3533) 

v_and_b32 v95, 63, v1 // (op 3533) 
v_lshrrev_b32 v12, 4, v95 // (op 3533) 

v_lshl_add_u32 v95, s6, 3, v12 // (op 3533) 


v_mul_lo_u32 v12, v20, s28 // (op 3533) 
v_add_u32 v12, v12, v95 // (op 3533) 


s_mul_i32 s1, s28, s23 // (op 3535) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 3535) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 3535) least significant
s_mul_hi_u32 s7, s28, s22 // (op 3535) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3535) most significant: sum
s_add_u32 s1, s1, s7 // (op 3535) most significant: sum


s_mov_b32 s56, s30 // (op 3535) reg expression
s_mov_b32 s57, s31 // (op 3535) reg expression
s_mov_b32 s58, s0 // (op 3535) reg expression
s_mov_b32 s59, 131072 // (op 3535) call()

v_lshl_add_u32 v20, s17, 8, v1 // (op 3597) 
v_mul_lo_u32 v95, v20, s32 // (op 3597) 

s_ashr_i64 s[0:1], s[36:37], 7 // (op 3597) 
s_lshr_b32 s6, s0, 1 // (op 3597) 

v_lshl_add_u32 v20, s6, 3, v95 // (op 3597) 


s_mul_i32 s1, s32, s9 // (op 3599) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 3599) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 3599) least significant
s_mul_hi_u32 s7, s32, s8 // (op 3599) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3599) most significant: sum
s_add_u32 s1, s1, s7 // (op 3599) most significant: sum


s_mov_b32 s60, s34 // (op 3599) reg expression
s_mov_b32 s61, s35 // (op 3599) reg expression
s_mov_b32 s62, s0 // (op 3599) reg expression
s_mov_b32 s63, 131072 // (op 3599) call()

v_and_b32 v95, -8, v1 // (op 3690) 
v_sub_u32 v13, v1, v95 // (op 3690) 

v_lshlrev_b32 v95, 4, v13 // (op 3690) 

v_lshrrev_b32 v13, 3, v1 // (op 3690) 
v_lshl_add_u32 v14, v13, 7, v95 // (op 3690) 


s_nop 0
v_readfirstlane_b32 s0, v14 // (op 3690) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v95, -8, v1 // (op 3742) 
v_sub_u32 v13, v1, v95 // (op 3742) 

v_lshlrev_b32 v95, 4, v13 // (op 3742) 

v_lshrrev_b32 v13, 3, v1 // (op 3742) 
v_lshl_add_u32 v14, v13, 7, v95 // (op 3742) 


s_nop 0
v_readfirstlane_b32 s1, v14 // (op 3742) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v95, 2, v1 // (op 3799) 
s_ashr_i64 s[6:7], s[36:37], 7 // (op 1249) 
s_and_b32 s10, s6, 4294967294 // (op 1249) 

s_ashr_i64 s[6:7], s[36:37], 7 // (op 1248) 
s_mov_b32 s11, s6 // (op 1248) convert

s_cmp_lt_u32 s10, s11 // (op 1248) 













s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopBottom_KLoopTail_1248 // (op 1248) Condition: Top (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopBottom_KLoopTail_1248 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopTop_KLoopTail_1248:
s_barrier  // (op 4027) 
s_add_u32 s10, s10, 1 // (op 1250) 


v_mov_b64 v[16:17], v[18:19] // (op 3347) 
s_add_u32 s11, 32768, s0 // (op 3347) 

s_mov_b32 m0, s11 // (op 3347) reg expression
buffer_load_dwordx4 v16, s[48:51], 0 offen offset: 0 lds // (op 3347) Load value direct to lds


v_add_co_u32 v18, vcc, 128, v18 // (op 3385) least significant half
v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 3385) most significant half
s_barrier  // (op 4043) 

s_lshl_b64 s[6:7], s[24:25], 5 // (op 3353) 
v_mov_b64 v[16:17], s[6:7] // (op 3353) convert


v_mov_b64 v[14:15], v[22:23] // (op 3353) 
v_mov_b64 v[24:25], v[14:15] // (op 3353) 
s_mov_b32 m0, s1 // (op 3353) reg expression
buffer_load_dwordx4 v24, s[52:55], 0 offen offset: 0 lds // (op 3353) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v16 // (op 3353) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 3353) most significant half

s_add_u32 m0, m0, 4096 // (op 3353) 
buffer_load_dwordx4 v24, s[52:55], 0 offen offset: 0 lds // (op 3353) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v16 // (op 3353) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 3353) most significant half

s_add_u32 m0, m0, 4096 // (op 3353) 
buffer_load_dwordx4 v24, s[52:55], 0 offen offset: 0 lds // (op 3353) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v16 // (op 3353) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 3353) most significant half

s_add_u32 m0, m0, 4096 // (op 3353) 
buffer_load_dwordx4 v24, s[52:55], 0 offen offset: 0 lds // (op 3353) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v16 // (op 3353) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 3353) most significant half

s_add_u32 m0, m0, 4096 // (op 3353) 
buffer_load_dwordx4 v24, s[52:55], 0 offen offset: 0 lds // (op 3353) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v16 // (op 3353) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 3353) most significant half

s_add_u32 m0, m0, 4096 // (op 3353) 
buffer_load_dwordx4 v24, s[52:55], 0 offen offset: 0 lds // (op 3353) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v16 // (op 3353) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 3353) most significant half

s_add_u32 m0, m0, 4096 // (op 3353) 
buffer_load_dwordx4 v24, s[52:55], 0 offen offset: 0 lds // (op 3353) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v16 // (op 3353) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 3353) most significant half

s_add_u32 m0, m0, 4096 // (op 3353) 
buffer_load_dwordx4 v24, s[52:55], 0 offen offset: 0 lds // (op 3353) Load value direct to lds



v_add_co_u32 v22, vcc, 128, v22 // (op 3461) least significant half
v_addc_co_u32 v23, vcc, 0, v23, vcc // (op 3461) most significant half


v_mov_b32 v21, v20 // (op 180) 

buffer_load_dword v13, v21, s[60:63], 0 offen // (op 180) Load value

v_add_u32 v20, 4, v20 // (op 3589) 
s_barrier  // (op 4059) 
s_waitcnt vmcnt(9)
s_barrier  // (op 4033) 






v_add_u32 v21, 73728, v95 // (op 178) 
ds_read_b128 v[24:27], v3 offset:32768 // (op 169) Load local data 
ds_read_b128 v[28:31], v3 offset:32832 // (op 169) Load local data 
s_waitcnt vmcnt(0)
ds_write_b32 v21, v13 // (op 178) Store local data 


s_barrier  // (op 4049) 



ds_read_b128 v[32:35], v5 // (op 171) Load local data 
ds_read_b128 v[36:39], v5 offset:64 // (op 171) Load local data 


v_mov_b32 v21, v12 // (op 173) 

buffer_load_ubyte v13, v21, s[56:59], 0 offen offset: 0 // (op 173) Load value

v_add_u32 v12, 4, v12 // (op 3525) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 4066) 



v_add_u32 v21, 73728, v0 // (op 175) 
ds_read_u8 v14, v21 // (op 175) Load local data 




s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[24:31], v[32:39], a[0:3], v13, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 176) 
ds_read_b128 v[40:43], v5 offset:4096 // (op 320) Load local data 


ds_read_b128 v[44:47], v5 offset:4160 // (op 320) Load local data 



v_add_u32 v21, 73856, v0 // (op 323) 
ds_read_u8 v14, v21 // (op 323) Load local data 




s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[24:31], v[40:47], a[4:7], v13, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 324) 
ds_read_b128 v[32:35], v5 offset:8192 // (op 476) Load local data 


ds_read_b128 v[36:39], v5 offset:8256 // (op 476) Load local data 



v_add_u32 v21, 73984, v0 // (op 479) 
ds_read_u8 v14, v21 // (op 479) Load local data 




s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[24:31], v[32:39], a[8:11], v13, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 480) 
ds_read_b128 v[40:43], v5 offset:12288 // (op 632) Load local data 


ds_read_b128 v[44:47], v5 offset:12352 // (op 632) Load local data 



v_add_u32 v21, 74112, v0 // (op 635) 
ds_read_u8 v14, v21 // (op 635) Load local data 




s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[24:31], v[40:47], a[12:15], v13, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 636) 
ds_read_b128 v[32:35], v5 offset:16384 // (op 788) Load local data 


ds_read_b128 v[36:39], v5 offset:16448 // (op 788) Load local data 



v_add_u32 v21, 74240, v0 // (op 791) 
ds_read_u8 v14, v21 // (op 791) Load local data 




s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[24:31], v[32:39], a[16:19], v13, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 792) 
ds_read_b128 v[40:43], v5 offset:20480 // (op 944) Load local data 


ds_read_b128 v[44:47], v5 offset:20544 // (op 944) Load local data 



v_add_u32 v21, 74368, v0 // (op 947) 
ds_read_u8 v14, v21 // (op 947) Load local data 




s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[24:31], v[40:47], a[20:23], v13, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 948) 
ds_read_b128 v[32:35], v5 offset:24576 // (op 1100) Load local data 


ds_read_b128 v[36:39], v5 offset:24640 // (op 1100) Load local data 



v_add_u32 v21, 74496, v0 // (op 1103) 
ds_read_u8 v14, v21 // (op 1103) Load local data 




s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[24:31], v[32:39], a[24:27], v13, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1104) 
ds_read_b128 v[40:43], v5 offset:28672 // (op 1256) Load local data 


ds_read_b128 v[44:47], v5 offset:28736 // (op 1256) Load local data 



v_add_u32 v21, 74624, v0 // (op 1259) 
ds_read_u8 v14, v21 // (op 1259) Load local data 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[24:31], v[40:47], a[28:31], v13, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1260) 




s_ashr_i64 s[6:7], s[36:37], 7 // (op 1248) 
s_mov_b32 s11, s6 // (op 1248) convert

s_cmp_lt_u32 s10, s11 // (op 1248) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopTop_KLoopTail_1248 // (op 1248) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopTop_KLoopTail_1248 if true)
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopBottom_KLoopTail_1248:






















s_barrier  // (op 4148) 







v_cmp_eq_f32 vcc, 0.00000, v10 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ConditionalFalse_0_beta_123
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
s_branch RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ConditionalBottom_0_beta_123
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ConditionalFalse_0_beta_123:


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
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ConditionalBottom_0_beta_123:
















s_endpgm  // End of RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2
.LRR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_end:
.size RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2, .LRR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_end-RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2
  .amdhsa_next_free_vgpr 148
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 75776 // lds bytes
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
    macrotile: [32, 256, 128]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 8]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2
    .symbol: RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 75776
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 64
    .vgpr_count: 114
    .agpr_count: 32
    .max_flat_workgroup_size: 256
    .workgroup_size: [256, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, {is-null: true}]
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
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: FP8, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP8, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: FP8, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP8, pointerType: PointerGlobal}
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

