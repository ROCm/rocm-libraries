

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2
.p2align 8
.type RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2,@function
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_exec_begin:


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


v_lshl_add_u32 v0, s16, 4, v6 // (op 3661) 

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


v_lshl_add_u32 v7, s16, 4, v9 // (op 3848) 

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
v_lshl_add_u32 v13, s16, 4, v12 // (op 3428) 

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
v_lshl_add_u32 v13, s16, 4, v12 // (op 3447) 

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


v_lshl_add_u32 v20, s16, 4, v22 // (op 3556) 

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


v_lshl_add_u32 v20, s16, 4, v23 // (op 3575) 

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

v_lshlrev_b32 v20, 1, v1 // (op 3620) 
v_lshl_add_u32 v23, s17, 8, v20 // (op 3620) 

v_mul_lo_u32 v20, v23, s32 // (op 3620) 

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

v_lshlrev_b32 v23, 1, v1 // (op 3639) 
v_lshl_add_u32 v24, s17, 8, v23 // (op 3639) 

v_mul_lo_u32 v23, v24, s32 // (op 3639) 

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




buffer_load_ubyte v24, v21, s[56:59], 0 offen offset: 0 // (op 3108) Load value
v_and_b32 v25, -8, v1 // (op 3757) 
v_sub_u32 v26, v1, v25 // (op 3757) 

v_lshlrev_b32 v25, 4, v26 // (op 3757) 

v_lshrrev_b32 v26, 3, v1 // (op 3757) 
v_lshl_add_u32 v27, v26, 7, v25 // (op 3757) 


s_nop 0
v_readfirstlane_b32 s0, v27 // (op 3757) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 4 // (op 3370) 
v_mov_b64 v[26:27], s[6:7] // (op 3370) convert


v_mov_b64 v[28:29], v[12:13] // (op 3370) 
s_mov_b32 m0, s0 // (op 3370) reg expression
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3370) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 3370) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3370) most significant half

s_add_u32 m0, m0, 2048 // (op 3370) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3370) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 3370) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3370) most significant half

s_add_u32 m0, m0, 2048 // (op 3370) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3370) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 3370) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3370) most significant half

s_add_u32 m0, m0, 2048 // (op 3370) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3370) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 3370) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3370) most significant half

s_add_u32 m0, m0, 2048 // (op 3370) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3370) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 3370) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3370) most significant half

s_add_u32 m0, m0, 2048 // (op 3370) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3370) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 3370) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3370) most significant half

s_add_u32 m0, m0, 2048 // (op 3370) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3370) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 3370) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3370) most significant half

s_add_u32 m0, m0, 2048 // (op 3370) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3370) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 3370) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3370) most significant half

s_add_u32 m0, m0, 2048 // (op 3370) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3370) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 3370) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3370) most significant half

s_add_u32 m0, m0, 2048 // (op 3370) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3370) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 3370) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3370) most significant half

s_add_u32 m0, m0, 2048 // (op 3370) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3370) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 3370) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3370) most significant half

s_add_u32 m0, m0, 2048 // (op 3370) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3370) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 3370) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3370) most significant half

s_add_u32 m0, m0, 2048 // (op 3370) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3370) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 3370) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3370) most significant half

s_add_u32 m0, m0, 2048 // (op 3370) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3370) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 3370) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3370) most significant half

s_add_u32 m0, m0, 2048 // (op 3370) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3370) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 3370) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3370) most significant half

s_add_u32 m0, m0, 2048 // (op 3370) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3370) Load value direct to lds




v_mov_b64 v[26:27], s[32:33] // (op 3115) convert

v_mov_b32 v25, v20 // (op 3115) 
buffer_load_dword v28, v25, s[60:63], 0 offen // (op 3115) Load value
v_add_u32 v25, v25, v26 // (op 3115) 
buffer_load_dword v29, v25, s[60:63], 0 offen // (op 3115) Load value


v_and_b32 v25, -8, v1 // (op 3705) 
v_sub_u32 v26, v1, v25 // (op 3705) 

v_lshlrev_b32 v25, 4, v26 // (op 3705) 

v_lshrrev_b32 v26, 3, v1 // (op 3705) 
v_lshl_add_u32 v27, v26, 7, v25 // (op 3705) 


s_nop 0
v_readfirstlane_b32 s0, v27 // (op 3705) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s1, 32768, s0 // (op 3375) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 3375) reg expression
buffer_load_dwordx4 v16, s[48:51], 0 offen offset: 0 lds // (op 3375) Load value direct to lds





buffer_load_ubyte v25, v21, s[56:59], 0 offen offset:4 // (op 3122) Load value

v_and_b32 v21, -8, v1 // (op 3916) 
v_sub_u32 v26, v1, v21 // (op 3916) 

v_lshlrev_b32 v21, 4, v26 // (op 3916) 

v_lshrrev_b32 v26, 3, v1 // (op 3916) 
v_lshl_add_u32 v27, v26, 7, v21 // (op 3916) 


s_nop 0
v_readfirstlane_b32 s0, v27 // (op 3916) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 4 // (op 3380) 
v_mov_b64 v[26:27], s[6:7] // (op 3380) convert


v_mov_b64 v[30:31], v[12:13] // (op 3380) 
s_add_u32 s1, 34816, s0 // (op 3380) 
s_mov_b32 m0, s1 // (op 3380) reg expression
s_mov_b32 s6, 128 // (op 3380) call()
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 3380) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v26 // (op 3380) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 3380) most significant half

s_add_u32 m0, m0, 2048 // (op 3380) 
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 3380) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v26 // (op 3380) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 3380) most significant half

s_add_u32 m0, m0, 2048 // (op 3380) 
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 3380) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v26 // (op 3380) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 3380) most significant half

s_add_u32 m0, m0, 2048 // (op 3380) 
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 3380) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v26 // (op 3380) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 3380) most significant half

s_add_u32 m0, m0, 2048 // (op 3380) 
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 3380) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v26 // (op 3380) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 3380) most significant half

s_add_u32 m0, m0, 2048 // (op 3380) 
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 3380) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v26 // (op 3380) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 3380) most significant half

s_add_u32 m0, m0, 2048 // (op 3380) 
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 3380) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v26 // (op 3380) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 3380) most significant half

s_add_u32 m0, m0, 2048 // (op 3380) 
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 3380) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v26 // (op 3380) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 3380) most significant half

s_add_u32 m0, m0, 2048 // (op 3380) 
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 3380) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v26 // (op 3380) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 3380) most significant half

s_add_u32 m0, m0, 2048 // (op 3380) 
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 3380) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v26 // (op 3380) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 3380) most significant half

s_add_u32 m0, m0, 2048 // (op 3380) 
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 3380) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v26 // (op 3380) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 3380) most significant half

s_add_u32 m0, m0, 2048 // (op 3380) 
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 3380) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v26 // (op 3380) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 3380) most significant half

s_add_u32 m0, m0, 2048 // (op 3380) 
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 3380) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v26 // (op 3380) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 3380) most significant half

s_add_u32 m0, m0, 2048 // (op 3380) 
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 3380) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v26 // (op 3380) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 3380) most significant half

s_add_u32 m0, m0, 2048 // (op 3380) 
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 3380) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v26 // (op 3380) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 3380) most significant half

s_add_u32 m0, m0, 2048 // (op 3380) 
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 3380) Load value direct to lds







v_mov_b64 v[12:13], s[32:33] // (op 3129) convert

v_mov_b32 v21, v20 // (op 3129) 
buffer_load_dword v26, v21, s[60:63], 0 offen offset:4 // (op 3129) Load value
v_add_u32 v21, v21, v12 // (op 3129) 
buffer_load_dword v27, v21, s[60:63], 0 offen offset:4 // (op 3129) Load value



v_and_b32 v12, -8, v1 // (op 3877) 
v_sub_u32 v13, v1, v12 // (op 3877) 

v_lshlrev_b32 v12, 4, v13 // (op 3877) 

v_lshrrev_b32 v13, 3, v1 // (op 3877) 
v_lshl_add_u32 v20, v13, 7, v12 // (op 3877) 


s_nop 0
v_readfirstlane_b32 s0, v20 // (op 3877) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s1, 67584, s0 // (op 3382) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 3382) reg expression
s_mov_b32 s6, 128 // (op 3382) call()
buffer_load_dwordx4 v16, s[48:51], s6 offen offset: 0 lds // (op 3382) Load value direct to lds




v_lshlrev_b32 v12, 3, v1 // (op 3815) 



v_add_u32 v13, 69632, v12 // (op 3135) 
s_waitcnt vmcnt(22)
ds_write_b32 v13, v28 // (op 3135) Store local data 

v_add_u32 v13, 69636, v12 // (op 3135) 
s_waitcnt vmcnt(21)
ds_write_b32 v13, v29 // (op 3135) Store local data 


s_barrier  // (op 4037) 



ds_read_b128 v[30:33], v5 // (op 3153) Load local data 
ds_read_b128 v[34:37], v5 offset:64 // (op 3153) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 4053) 



v_add_u32 v12, 69632, v0 // (op 3193) 
ds_read_u8 v13, v12 // (op 3193) Load local data 




ds_read_b128 v[38:41], v5 offset:4096 // (op 3158) Load local data 
ds_read_b128 v[42:45], v5 offset:4160 // (op 3158) Load local data 



v_add_u32 v12, 69760, v0 // (op 3198) 
ds_read_u8 v16, v12 // (op 3198) Load local data 




ds_read_b128 v[46:49], v5 offset:8192 // (op 3163) Load local data 
ds_read_b128 v[50:53], v5 offset:8256 // (op 3163) Load local data 



v_add_u32 v12, 69888, v0 // (op 3203) 
ds_read_u8 v17, v12 // (op 3203) Load local data 




ds_read_b128 v[54:57], v5 offset:12288 // (op 3168) Load local data 
ds_read_b128 v[58:61], v5 offset:12352 // (op 3168) Load local data 



v_add_u32 v12, 70016, v0 // (op 3208) 
ds_read_u8 v20, v12 // (op 3208) Load local data 




ds_read_b128 v[62:65], v5 offset:16384 // (op 3173) Load local data 
ds_read_b128 v[66:69], v5 offset:16448 // (op 3173) Load local data 



v_add_u32 v12, 70144, v0 // (op 3213) 
ds_read_u8 v21, v12 // (op 3213) Load local data 




ds_read_b128 v[70:73], v5 offset:20480 // (op 3178) Load local data 
ds_read_b128 v[74:77], v5 offset:20544 // (op 3178) Load local data 



v_add_u32 v12, 70272, v0 // (op 3218) 
ds_read_u8 v78, v12 // (op 3218) Load local data 




ds_read_b128 v[80:83], v5 offset:24576 // (op 3183) Load local data 
ds_read_b128 v[84:87], v5 offset:24640 // (op 3183) Load local data 



v_add_u32 v12, 70400, v0 // (op 3223) 
ds_read_u8 v79, v12 // (op 3223) Load local data 




ds_read_b128 v[88:91], v5 offset:28672 // (op 3188) Load local data 
ds_read_b128 v[92:95], v5 offset:28736 // (op 3188) Load local data 



v_add_u32 v12, 70528, v0 // (op 3228) 
ds_read_u8 v96, v12 // (op 3228) Load local data 

s_waitcnt vmcnt(20)
s_barrier  // (op 4021) 



ds_read_b128 v[98:101], v3 offset:32768 // (op 3149) Load local data 
ds_read_b128 v[102:105], v3 offset:32832 // (op 3149) Load local data 
v_and_b32 v12, -8, v1 // (op 3677) 
v_sub_u32 v97, v1, v12 // (op 3677) 

v_lshlrev_b32 v12, 4, v97 // (op 3677) 

v_lshrrev_b32 v97, 3, v1 // (op 3677) 
v_lshl_add_u32 v106, v97, 7, v12 // (op 3677) 


s_nop 0
v_readfirstlane_b32 s0, v106 // (op 3677) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -8, v1 // (op 3729) 
v_sub_u32 v97, v1, v12 // (op 3729) 

v_lshlrev_b32 v12, 4, v97 // (op 3729) 

v_lshrrev_b32 v97, 3, v1 // (op 3729) 
v_lshl_add_u32 v106, v97, 7, v12 // (op 3729) 


s_nop 0
v_readfirstlane_b32 s1, v106 // (op 3729) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v12, 3, v1 // (op 3786) 
v_and_b32 v97, -8, v1 // (op 3862) 
v_sub_u32 v106, v1, v97 // (op 3862) 

v_lshlrev_b32 v97, 4, v106 // (op 3862) 

v_lshrrev_b32 v106, 3, v1 // (op 3862) 
v_lshl_add_u32 v107, v106, 7, v97 // (op 3862) 


s_nop 0
v_readfirstlane_b32 s6, v107 // (op 3862) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v97, -8, v1 // (op 3901) 
v_sub_u32 v106, v1, v97 // (op 3901) 

v_lshlrev_b32 v97, 4, v106 // (op 3901) 

v_lshrrev_b32 v106, 3, v1 // (op 3901) 
v_lshl_add_u32 v107, v106, 7, v97 // (op 3901) 


s_nop 0
v_readfirstlane_b32 s7, v107 // (op 3901) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v97, 3, v1 // (op 3945) 
s_mov_b32 s10, 0 // (op 1244) call()
s_ashr_i64 s[14:15], s[36:37], 7 // (op 1243) 
s_mov_b32 s11, s14 // (op 1243) convert

s_and_b32 s14, s11, 4294967294 // (op 1243) 

s_cmp_lt_u32 s10, s14 // (op 1243) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_ForLoopBottom_KLoop_1243 // (op 1243) Condition: Top (jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_ForLoopBottom_KLoop_1243 if false)
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_ForLoopTop_KLoop_1243:
s_add_u32 s10, s10, 2 // (op 1245) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[98:105], v[30:37], a[0:3], v24, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 82) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[98:105], v[38:45], a[4:7], v24, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 343) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[98:105], v[46:53], a[8:11], v24, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 499) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[98:105], v[54:61], a[12:15], v24, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 655) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[98:105], v[62:69], a[16:19], v24, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 811) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[98:105], v[70:77], a[20:23], v24, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 967) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[98:105], v[80:87], a[24:27], v24, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1123) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[98:105], v[88:95], a[28:31], v24, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1279) 


v_mov_b32 v106, v22 // (op 75) 

buffer_load_ubyte v24, v106, s[56:59], 0 offen offset:8 // (op 75) Load value
s_barrier  // (op 4040) 

s_lshl_b64 s[14:15], s[24:25], 4 // (op 3342) 
v_mov_b64 v[108:109], s[14:15] // (op 3342) convert


v_mov_b64 v[110:111], v[14:15] // (op 3342) 
v_mov_b64 v[112:113], v[110:111] // (op 3342) 
s_mov_b32 m0, s1 // (op 3342) reg expression
s_mov_b32 s11, 256 // (op 3342) call()
buffer_load_dwordx4 v112, s[52:55], s11 offen offset: 0 lds // (op 3342) Load value direct to lds
v_add_co_u32 v112, s[14:15], v112, v108 // (op 3342) least significant half
v_addc_co_u32 v113, s[14:15], 0, v113, s[14:15] // (op 3342) most significant half

s_add_u32 m0, m0, 2048 // (op 3342) 
buffer_load_dwordx4 v112, s[52:55], s11 offen offset: 0 lds // (op 3342) Load value direct to lds
v_add_co_u32 v112, s[14:15], v112, v108 // (op 3342) least significant half
v_addc_co_u32 v113, s[14:15], 0, v113, s[14:15] // (op 3342) most significant half

s_add_u32 m0, m0, 2048 // (op 3342) 
buffer_load_dwordx4 v112, s[52:55], s11 offen offset: 0 lds // (op 3342) Load value direct to lds
v_add_co_u32 v112, s[14:15], v112, v108 // (op 3342) least significant half
v_addc_co_u32 v113, s[14:15], 0, v113, s[14:15] // (op 3342) most significant half

s_add_u32 m0, m0, 2048 // (op 3342) 
buffer_load_dwordx4 v112, s[52:55], s11 offen offset: 0 lds // (op 3342) Load value direct to lds
v_add_co_u32 v112, s[14:15], v112, v108 // (op 3342) least significant half
v_addc_co_u32 v113, s[14:15], 0, v113, s[14:15] // (op 3342) most significant half

s_add_u32 m0, m0, 2048 // (op 3342) 
buffer_load_dwordx4 v112, s[52:55], s11 offen offset: 0 lds // (op 3342) Load value direct to lds
v_add_co_u32 v112, s[14:15], v112, v108 // (op 3342) least significant half
v_addc_co_u32 v113, s[14:15], 0, v113, s[14:15] // (op 3342) most significant half

s_add_u32 m0, m0, 2048 // (op 3342) 
buffer_load_dwordx4 v112, s[52:55], s11 offen offset: 0 lds // (op 3342) Load value direct to lds
v_add_co_u32 v112, s[14:15], v112, v108 // (op 3342) least significant half
v_addc_co_u32 v113, s[14:15], 0, v113, s[14:15] // (op 3342) most significant half

s_add_u32 m0, m0, 2048 // (op 3342) 
buffer_load_dwordx4 v112, s[52:55], s11 offen offset: 0 lds // (op 3342) Load value direct to lds
v_add_co_u32 v112, s[14:15], v112, v108 // (op 3342) least significant half
v_addc_co_u32 v113, s[14:15], 0, v113, s[14:15] // (op 3342) most significant half

s_add_u32 m0, m0, 2048 // (op 3342) 
buffer_load_dwordx4 v112, s[52:55], s11 offen offset: 0 lds // (op 3342) Load value direct to lds
v_add_co_u32 v112, s[14:15], v112, v108 // (op 3342) least significant half
v_addc_co_u32 v113, s[14:15], 0, v113, s[14:15] // (op 3342) most significant half

s_add_u32 m0, m0, 2048 // (op 3342) 
buffer_load_dwordx4 v112, s[52:55], s11 offen offset: 0 lds // (op 3342) Load value direct to lds
v_add_co_u32 v112, s[14:15], v112, v108 // (op 3342) least significant half
v_addc_co_u32 v113, s[14:15], 0, v113, s[14:15] // (op 3342) most significant half

s_add_u32 m0, m0, 2048 // (op 3342) 
buffer_load_dwordx4 v112, s[52:55], s11 offen offset: 0 lds // (op 3342) Load value direct to lds
v_add_co_u32 v112, s[14:15], v112, v108 // (op 3342) least significant half
v_addc_co_u32 v113, s[14:15], 0, v113, s[14:15] // (op 3342) most significant half

s_add_u32 m0, m0, 2048 // (op 3342) 
buffer_load_dwordx4 v112, s[52:55], s11 offen offset: 0 lds // (op 3342) Load value direct to lds
v_add_co_u32 v112, s[14:15], v112, v108 // (op 3342) least significant half
v_addc_co_u32 v113, s[14:15], 0, v113, s[14:15] // (op 3342) most significant half

s_add_u32 m0, m0, 2048 // (op 3342) 
buffer_load_dwordx4 v112, s[52:55], s11 offen offset: 0 lds // (op 3342) Load value direct to lds
v_add_co_u32 v112, s[14:15], v112, v108 // (op 3342) least significant half
v_addc_co_u32 v113, s[14:15], 0, v113, s[14:15] // (op 3342) most significant half

s_add_u32 m0, m0, 2048 // (op 3342) 
buffer_load_dwordx4 v112, s[52:55], s11 offen offset: 0 lds // (op 3342) Load value direct to lds
v_add_co_u32 v112, s[14:15], v112, v108 // (op 3342) least significant half
v_addc_co_u32 v113, s[14:15], 0, v113, s[14:15] // (op 3342) most significant half

s_add_u32 m0, m0, 2048 // (op 3342) 
buffer_load_dwordx4 v112, s[52:55], s11 offen offset: 0 lds // (op 3342) Load value direct to lds
v_add_co_u32 v112, s[14:15], v112, v108 // (op 3342) least significant half
v_addc_co_u32 v113, s[14:15], 0, v113, s[14:15] // (op 3342) most significant half

s_add_u32 m0, m0, 2048 // (op 3342) 
buffer_load_dwordx4 v112, s[52:55], s11 offen offset: 0 lds // (op 3342) Load value direct to lds
v_add_co_u32 v112, s[14:15], v112, v108 // (op 3342) least significant half
v_addc_co_u32 v113, s[14:15], 0, v113, s[14:15] // (op 3342) most significant half

s_add_u32 m0, m0, 2048 // (op 3342) 
buffer_load_dwordx4 v112, s[52:55], s11 offen offset: 0 lds // (op 3342) Load value direct to lds




v_mov_b64 v[108:109], s[32:33] // (op 21) convert

v_mov_b32 v107, v23 // (op 21) 
v_mov_b32 v112, v107 // (op 21) 
buffer_load_dword v28, v112, s[60:63], 0 offen offset:8 // (op 21) Load value
v_add_u32 v112, v112, v108 // (op 21) 
buffer_load_dword v29, v112, s[60:63], 0 offen offset:8 // (op 21) Load value


s_barrier  // (op 4024) 


v_mov_b64 v[108:109], v[18:19] // (op 3337) 
s_add_u32 s11, 32768, s0 // (op 3337) 

s_mov_b32 m0, s11 // (op 3337) reg expression
s_mov_b32 s14, 256 // (op 3337) call()
buffer_load_dwordx4 v108, s[48:51], s14 offen offset: 0 lds // (op 3337) Load value direct to lds


s_barrier  // (op 4085) 



v_add_u32 v112, 70656, v97 // (op 236) 
s_waitcnt vmcnt(22)
ds_write_b32 v112, v26 // (op 236) Store local data 

v_add_u32 v112, 70660, v97 // (op 236) 
s_waitcnt vmcnt(21)
ds_write_b32 v112, v27 // (op 236) Store local data 

s_barrier  // (op 4076) 



ds_read_b128 v[30:33], v8 offset:34816 // (op 229) Load local data 
ds_read_b128 v[34:37], v8 offset:34880 // (op 229) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 4082) 



v_add_u32 v112, 70656, v7 // (op 233) 
ds_read_u8 v13, v112 // (op 233) Load local data 




ds_read_b128 v[38:41], v8 offset:38912 // (op 348) Load local data 
ds_read_b128 v[42:45], v8 offset:38976 // (op 348) Load local data 



v_add_u32 v112, 70784, v7 // (op 351) 
ds_read_u8 v16, v112 // (op 351) Load local data 




ds_read_b128 v[46:49], v8 offset:43008 // (op 504) Load local data 
ds_read_b128 v[50:53], v8 offset:43072 // (op 504) Load local data 



v_add_u32 v112, 70912, v7 // (op 507) 
ds_read_u8 v17, v112 // (op 507) Load local data 




ds_read_b128 v[54:57], v8 offset:47104 // (op 660) Load local data 
ds_read_b128 v[58:61], v8 offset:47168 // (op 660) Load local data 



v_add_u32 v112, 71040, v7 // (op 663) 
ds_read_u8 v20, v112 // (op 663) Load local data 




ds_read_b128 v[62:65], v8 offset:51200 // (op 816) Load local data 
ds_read_b128 v[66:69], v8 offset:51264 // (op 816) Load local data 



v_add_u32 v112, 71168, v7 // (op 819) 
ds_read_u8 v21, v112 // (op 819) Load local data 




ds_read_b128 v[70:73], v8 offset:55296 // (op 972) Load local data 
ds_read_b128 v[74:77], v8 offset:55360 // (op 972) Load local data 



v_add_u32 v112, 71296, v7 // (op 975) 
ds_read_u8 v78, v112 // (op 975) Load local data 




ds_read_b128 v[80:83], v8 offset:59392 // (op 1128) Load local data 
ds_read_b128 v[84:87], v8 offset:59456 // (op 1128) Load local data 



v_add_u32 v112, 71424, v7 // (op 1131) 
ds_read_u8 v79, v112 // (op 1131) Load local data 




ds_read_b128 v[88:91], v8 offset:63488 // (op 1284) Load local data 
ds_read_b128 v[92:95], v8 offset:63552 // (op 1284) Load local data 



v_add_u32 v112, 71552, v7 // (op 1287) 
ds_read_u8 v96, v112 // (op 1287) Load local data 

s_barrier  // (op 4160) 
s_waitcnt vmcnt(20)
s_barrier  // (op 4070) 



v_add_u32 v112, 67584, v4 // (op 227) 
ds_read_b128 v[98:101], v112 // (op 227) Load local data 

v_add_u32 v112, 67648, v4 // (op 227) 
ds_read_b128 v[102:105], v112 // (op 227) Load local data 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[98:105], v[30:37], a[0:3], v25, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 234) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[98:105], v[38:45], a[4:7], v25, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 344) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[98:105], v[46:53], a[8:11], v25, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 500) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[98:105], v[54:61], a[12:15], v25, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 656) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[98:105], v[62:69], a[16:19], v25, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 812) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[98:105], v[70:77], a[20:23], v25, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 968) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[98:105], v[80:87], a[24:27], v25, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1124) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[98:105], v[88:95], a[28:31], v25, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1280) 



buffer_load_ubyte v25, v106, s[56:59], 0 offen offset:12 // (op 231) Load value

s_barrier  // (op 4079) 
v_add_u32 v22, 8, v22 // (op 3567) 

s_lshl_b64 s[14:15], s[24:25], 4 // (op 3365) 
v_mov_b64 v[112:113], s[14:15] // (op 3365) convert


v_mov_b64 v[114:115], v[110:111] // (op 3365) 
s_add_u32 s11, 34816, s7 // (op 3365) 
s_mov_b32 m0, s11 // (op 3365) reg expression
s_mov_b32 s14, 384 // (op 3365) call()
buffer_load_dwordx4 v114, s[52:55], s14 offen offset: 0 lds // (op 3365) Load value direct to lds
v_add_co_u32 v114, s[18:19], v114, v112 // (op 3365) least significant half
v_addc_co_u32 v115, s[18:19], 0, v115, s[18:19] // (op 3365) most significant half

s_add_u32 m0, m0, 2048 // (op 3365) 
buffer_load_dwordx4 v114, s[52:55], s14 offen offset: 0 lds // (op 3365) Load value direct to lds
v_add_co_u32 v114, s[18:19], v114, v112 // (op 3365) least significant half
v_addc_co_u32 v115, s[18:19], 0, v115, s[18:19] // (op 3365) most significant half

s_add_u32 m0, m0, 2048 // (op 3365) 
buffer_load_dwordx4 v114, s[52:55], s14 offen offset: 0 lds // (op 3365) Load value direct to lds
v_add_co_u32 v114, s[18:19], v114, v112 // (op 3365) least significant half
v_addc_co_u32 v115, s[18:19], 0, v115, s[18:19] // (op 3365) most significant half

s_add_u32 m0, m0, 2048 // (op 3365) 
buffer_load_dwordx4 v114, s[52:55], s14 offen offset: 0 lds // (op 3365) Load value direct to lds
v_add_co_u32 v114, s[18:19], v114, v112 // (op 3365) least significant half
v_addc_co_u32 v115, s[18:19], 0, v115, s[18:19] // (op 3365) most significant half

s_add_u32 m0, m0, 2048 // (op 3365) 
buffer_load_dwordx4 v114, s[52:55], s14 offen offset: 0 lds // (op 3365) Load value direct to lds
v_add_co_u32 v114, s[18:19], v114, v112 // (op 3365) least significant half
v_addc_co_u32 v115, s[18:19], 0, v115, s[18:19] // (op 3365) most significant half

s_add_u32 m0, m0, 2048 // (op 3365) 
buffer_load_dwordx4 v114, s[52:55], s14 offen offset: 0 lds // (op 3365) Load value direct to lds
v_add_co_u32 v114, s[18:19], v114, v112 // (op 3365) least significant half
v_addc_co_u32 v115, s[18:19], 0, v115, s[18:19] // (op 3365) most significant half

s_add_u32 m0, m0, 2048 // (op 3365) 
buffer_load_dwordx4 v114, s[52:55], s14 offen offset: 0 lds // (op 3365) Load value direct to lds
v_add_co_u32 v114, s[18:19], v114, v112 // (op 3365) least significant half
v_addc_co_u32 v115, s[18:19], 0, v115, s[18:19] // (op 3365) most significant half

s_add_u32 m0, m0, 2048 // (op 3365) 
buffer_load_dwordx4 v114, s[52:55], s14 offen offset: 0 lds // (op 3365) Load value direct to lds
v_add_co_u32 v114, s[18:19], v114, v112 // (op 3365) least significant half
v_addc_co_u32 v115, s[18:19], 0, v115, s[18:19] // (op 3365) most significant half

s_add_u32 m0, m0, 2048 // (op 3365) 
buffer_load_dwordx4 v114, s[52:55], s14 offen offset: 0 lds // (op 3365) Load value direct to lds
v_add_co_u32 v114, s[18:19], v114, v112 // (op 3365) least significant half
v_addc_co_u32 v115, s[18:19], 0, v115, s[18:19] // (op 3365) most significant half

s_add_u32 m0, m0, 2048 // (op 3365) 
buffer_load_dwordx4 v114, s[52:55], s14 offen offset: 0 lds // (op 3365) Load value direct to lds
v_add_co_u32 v114, s[18:19], v114, v112 // (op 3365) least significant half
v_addc_co_u32 v115, s[18:19], 0, v115, s[18:19] // (op 3365) most significant half

s_add_u32 m0, m0, 2048 // (op 3365) 
buffer_load_dwordx4 v114, s[52:55], s14 offen offset: 0 lds // (op 3365) Load value direct to lds
v_add_co_u32 v114, s[18:19], v114, v112 // (op 3365) least significant half
v_addc_co_u32 v115, s[18:19], 0, v115, s[18:19] // (op 3365) most significant half

s_add_u32 m0, m0, 2048 // (op 3365) 
buffer_load_dwordx4 v114, s[52:55], s14 offen offset: 0 lds // (op 3365) Load value direct to lds
v_add_co_u32 v114, s[18:19], v114, v112 // (op 3365) least significant half
v_addc_co_u32 v115, s[18:19], 0, v115, s[18:19] // (op 3365) most significant half

s_add_u32 m0, m0, 2048 // (op 3365) 
buffer_load_dwordx4 v114, s[52:55], s14 offen offset: 0 lds // (op 3365) Load value direct to lds
v_add_co_u32 v114, s[18:19], v114, v112 // (op 3365) least significant half
v_addc_co_u32 v115, s[18:19], 0, v115, s[18:19] // (op 3365) most significant half

s_add_u32 m0, m0, 2048 // (op 3365) 
buffer_load_dwordx4 v114, s[52:55], s14 offen offset: 0 lds // (op 3365) Load value direct to lds
v_add_co_u32 v114, s[18:19], v114, v112 // (op 3365) least significant half
v_addc_co_u32 v115, s[18:19], 0, v115, s[18:19] // (op 3365) most significant half

s_add_u32 m0, m0, 2048 // (op 3365) 
buffer_load_dwordx4 v114, s[52:55], s14 offen offset: 0 lds // (op 3365) Load value direct to lds
v_add_co_u32 v114, s[18:19], v114, v112 // (op 3365) least significant half
v_addc_co_u32 v115, s[18:19], 0, v115, s[18:19] // (op 3365) most significant half

s_add_u32 m0, m0, 2048 // (op 3365) 
buffer_load_dwordx4 v114, s[52:55], s14 offen offset: 0 lds // (op 3365) Load value direct to lds






v_add_co_u32 v14, vcc, 256, v14 // (op 3503) least significant half
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 3503) most significant half
v_mov_b64 v[110:111], s[32:33] // (op 238) convert

v_mov_b32 v106, v107 // (op 238) 
buffer_load_dword v26, v106, s[60:63], 0 offen offset:12 // (op 238) Load value
v_add_u32 v106, v106, v110 // (op 238) 
buffer_load_dword v27, v106, s[60:63], 0 offen offset:12 // (op 238) Load value



s_barrier  // (op 4073) 
v_add_u32 v23, 8, v23 // (op 3631) 


s_add_u32 s11, 67584, s6 // (op 3360) 

s_mov_b32 m0, s11 // (op 3360) reg expression
s_mov_b32 s14, 384 // (op 3360) call()
buffer_load_dwordx4 v108, s[48:51], s14 offen offset: 0 lds // (op 3360) Load value direct to lds



s_barrier  // (op 4056) 
v_add_co_u32 v18, vcc, 256, v18 // (op 3439) least significant half



v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 3439) most significant half
v_add_u32 v106, 69632, v12 // (op 26) 
s_waitcnt vmcnt(22)
ds_write_b32 v106, v28 // (op 26) Store local data 

v_add_u32 v106, 69636, v12 // (op 26) 
s_waitcnt vmcnt(21)
ds_write_b32 v106, v29 // (op 26) Store local data 

s_barrier  // (op 4046) 



ds_read_b128 v[30:33], v5 // (op 70) Load local data 
ds_read_b128 v[34:37], v5 offset:64 // (op 70) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 4063) 



v_add_u32 v106, 69632, v0 // (op 79) 
ds_read_u8 v13, v106 // (op 79) Load local data 




ds_read_b128 v[38:41], v5 offset:4096 // (op 339) Load local data 
ds_read_b128 v[42:45], v5 offset:4160 // (op 339) Load local data 



v_add_u32 v106, 69760, v0 // (op 342) 
ds_read_u8 v16, v106 // (op 342) Load local data 




ds_read_b128 v[46:49], v5 offset:8192 // (op 495) Load local data 
ds_read_b128 v[50:53], v5 offset:8256 // (op 495) Load local data 



v_add_u32 v106, 69888, v0 // (op 498) 
ds_read_u8 v17, v106 // (op 498) Load local data 




ds_read_b128 v[54:57], v5 offset:12288 // (op 651) Load local data 
ds_read_b128 v[58:61], v5 offset:12352 // (op 651) Load local data 



v_add_u32 v106, 70016, v0 // (op 654) 
ds_read_u8 v20, v106 // (op 654) Load local data 




ds_read_b128 v[62:65], v5 offset:16384 // (op 807) Load local data 
ds_read_b128 v[66:69], v5 offset:16448 // (op 807) Load local data 



v_add_u32 v106, 70144, v0 // (op 810) 
ds_read_u8 v21, v106 // (op 810) Load local data 




ds_read_b128 v[70:73], v5 offset:20480 // (op 963) Load local data 
ds_read_b128 v[74:77], v5 offset:20544 // (op 963) Load local data 



v_add_u32 v106, 70272, v0 // (op 966) 
ds_read_u8 v78, v106 // (op 966) Load local data 




ds_read_b128 v[80:83], v5 offset:24576 // (op 1119) Load local data 
ds_read_b128 v[84:87], v5 offset:24640 // (op 1119) Load local data 



v_add_u32 v106, 70400, v0 // (op 1122) 
ds_read_u8 v79, v106 // (op 1122) Load local data 




ds_read_b128 v[88:91], v5 offset:28672 // (op 1275) Load local data 
ds_read_b128 v[92:95], v5 offset:28736 // (op 1275) Load local data 



v_add_u32 v106, 70528, v0 // (op 1278) 
ds_read_u8 v96, v106 // (op 1278) Load local data 

s_waitcnt vmcnt(20)
s_barrier  // (op 4030) 



ds_read_b128 v[98:101], v3 offset:32768 // (op 65) Load local data 
ds_read_b128 v[102:105], v3 offset:32832 // (op 65) Load local data 
s_ashr_i64 s[14:15], s[36:37], 7 // (op 1243) 
s_mov_b32 s11, s14 // (op 1243) convert

s_and_b32 s14, s11, 4294967294 // (op 1243) 

s_cmp_lt_u32 s10, s14 // (op 1243) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_ForLoopTop_KLoop_1243 // (op 1243) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_ForLoopTop_KLoop_1243 if true)
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_ForLoopBottom_KLoop_1243:




















s_barrier  // (op 4144) 
v_lshrrev_b32 v97, 3, v1 // (op 3395) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 3395) 
v_lshl_add_u32 v12, s16, 4, v97 // (op 3395) 

s_lshr_b32 s6, s0, 1 // (op 3395) 

v_and_b32 v97, -8, v1 // (op 3395) 
v_sub_u32 v13, v1, v97 // (op 3395) 

v_lshlrev_b32 v97, 4, v13 // (op 3395) 

v_lshl_add_u32 v13, s6, 8, v97 // (op 3395) 


v_mov_b32 v15, 0 // (op 3395) convert
v_mov_b32 v14, v13 // (op 3395) convert

v_mul_lo_u32 v17, v12, s13 // (op 3395) most significant: low of low * high
v_mul_hi_u32 v13, v12, s12 // (op 3395) most significant: high of low * low
v_mul_lo_u32 v16, v12, s12 // (op 3395) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 3395) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 3395) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 3395) most significant half



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

v_and_b32 v97, -8, v1 // (op 3469) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 3469) 
v_sub_u32 v12, v1, v97 // (op 3469) 

s_lshr_b32 s6, s0, 1 // (op 3469) 

v_lshlrev_b32 v97, 4, v12 // (op 3469) 

v_lshl_add_u32 v12, s6, 8, v97 // (op 3469) 


v_mov_b32 v15, 0 // (op 3469) convert
v_mov_b32 v14, v12 // (op 3469) convert

v_lshrrev_b32 v97, 3, v1 // (op 3469) 
v_lshl_add_u32 v12, s17, 8, v97 // (op 3469) 

v_mul_lo_u32 v19, v12, s25 // (op 3469) most significant: low of low * high
v_mul_hi_u32 v13, v12, s24 // (op 3469) most significant: high of low * low
v_mul_lo_u32 v18, v12, s24 // (op 3469) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 3469) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v18 // (op 3469) least significant half
v_addc_co_u32 v13, s[0:1], v15, v19, s[0:1] // (op 3469) most significant half



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

v_lshrrev_b32 v97, 7, v1 // (op 3533) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 3533) 
v_and_b32 v14, 15, v1 // (op 3533) 
s_lshr_b32 s6, s0, 1 // (op 3533) 

v_lshl_add_u32 v15, v97, 4, v14 // (op 3533) 


v_lshl_add_u32 v14, s16, 4, v15 // (op 3533) 

v_and_b32 v15, 63, v1 // (op 3533) 
v_lshrrev_b32 v97, 4, v15 // (op 3533) 

v_lshl_add_u32 v15, s6, 3, v97 // (op 3533) 


v_mul_lo_u32 v97, v14, s28 // (op 3533) 
v_add_u32 v97, v97, v15 // (op 3533) 


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

v_lshlrev_b32 v14, 1, v1 // (op 3597) 
v_lshl_add_u32 v15, s17, 8, v14 // (op 3597) 

s_ashr_i64 s[0:1], s[36:37], 7 // (op 3597) 
v_mul_lo_u32 v14, v15, s32 // (op 3597) 

s_lshr_b32 s6, s0, 1 // (op 3597) 

v_lshl_add_u32 v15, s6, 3, v14 // (op 3597) 


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

v_and_b32 v14, -8, v1 // (op 3690) 
v_sub_u32 v18, v1, v14 // (op 3690) 

v_lshlrev_b32 v14, 4, v18 // (op 3690) 

v_lshrrev_b32 v18, 3, v1 // (op 3690) 
v_lshl_add_u32 v19, v18, 7, v14 // (op 3690) 


s_nop 0
v_readfirstlane_b32 s0, v19 // (op 3690) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -8, v1 // (op 3742) 
v_sub_u32 v18, v1, v14 // (op 3742) 

v_lshlrev_b32 v14, 4, v18 // (op 3742) 

v_lshrrev_b32 v18, 3, v1 // (op 3742) 
v_lshl_add_u32 v19, v18, 7, v14 // (op 3742) 


s_nop 0
v_readfirstlane_b32 s1, v19 // (op 3742) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v14, 3, v1 // (op 3799) 
s_ashr_i64 s[6:7], s[36:37], 7 // (op 1249) 
s_and_b32 s10, s6, 4294967294 // (op 1249) 

s_ashr_i64 s[6:7], s[36:37], 7 // (op 1248) 
s_mov_b32 s11, s6 // (op 1248) convert

s_cmp_lt_u32 s10, s11 // (op 1248) 













s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_ForLoopBottom_KLoopTail_1248 // (op 1248) Condition: Top (jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_ForLoopBottom_KLoopTail_1248 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_ForLoopTop_KLoopTail_1248:
s_barrier  // (op 4027) 
s_add_u32 s10, s10, 1 // (op 1250) 


v_mov_b64 v[18:19], v[16:17] // (op 3347) 
s_add_u32 s11, 32768, s0 // (op 3347) 

s_mov_b32 m0, s11 // (op 3347) reg expression
buffer_load_dwordx4 v18, s[48:51], 0 offen offset: 0 lds // (op 3347) Load value direct to lds


v_add_co_u32 v16, vcc, 128, v16 // (op 3385) least significant half
v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 3385) most significant half
s_barrier  // (op 4043) 

s_lshl_b64 s[6:7], s[24:25], 4 // (op 3353) 
v_mov_b64 v[18:19], s[6:7] // (op 3353) convert


v_mov_b64 v[20:21], v[12:13] // (op 3353) 
v_mov_b64 v[22:23], v[20:21] // (op 3353) 
s_mov_b32 m0, s1 // (op 3353) reg expression
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 3353) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v18 // (op 3353) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 3353) most significant half

s_add_u32 m0, m0, 2048 // (op 3353) 
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 3353) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v18 // (op 3353) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 3353) most significant half

s_add_u32 m0, m0, 2048 // (op 3353) 
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 3353) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v18 // (op 3353) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 3353) most significant half

s_add_u32 m0, m0, 2048 // (op 3353) 
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 3353) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v18 // (op 3353) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 3353) most significant half

s_add_u32 m0, m0, 2048 // (op 3353) 
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 3353) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v18 // (op 3353) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 3353) most significant half

s_add_u32 m0, m0, 2048 // (op 3353) 
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 3353) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v18 // (op 3353) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 3353) most significant half

s_add_u32 m0, m0, 2048 // (op 3353) 
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 3353) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v18 // (op 3353) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 3353) most significant half

s_add_u32 m0, m0, 2048 // (op 3353) 
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 3353) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v18 // (op 3353) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 3353) most significant half

s_add_u32 m0, m0, 2048 // (op 3353) 
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 3353) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v18 // (op 3353) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 3353) most significant half

s_add_u32 m0, m0, 2048 // (op 3353) 
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 3353) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v18 // (op 3353) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 3353) most significant half

s_add_u32 m0, m0, 2048 // (op 3353) 
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 3353) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v18 // (op 3353) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 3353) most significant half

s_add_u32 m0, m0, 2048 // (op 3353) 
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 3353) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v18 // (op 3353) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 3353) most significant half

s_add_u32 m0, m0, 2048 // (op 3353) 
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 3353) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v18 // (op 3353) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 3353) most significant half

s_add_u32 m0, m0, 2048 // (op 3353) 
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 3353) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v18 // (op 3353) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 3353) most significant half

s_add_u32 m0, m0, 2048 // (op 3353) 
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 3353) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v18 // (op 3353) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 3353) most significant half

s_add_u32 m0, m0, 2048 // (op 3353) 
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 3353) Load value direct to lds



v_add_co_u32 v12, vcc, 128, v12 // (op 3461) least significant half
v_addc_co_u32 v13, vcc, 0, v13, vcc // (op 3461) most significant half

v_mov_b64 v[18:19], s[32:33] // (op 180) convert

v_mov_b32 v20, v15 // (op 180) 
v_mov_b32 v21, v20 // (op 180) 
buffer_load_dword v22, v21, s[60:63], 0 offen // (op 180) Load value
v_add_u32 v21, v21, v18 // (op 180) 
buffer_load_dword v23, v21, s[60:63], 0 offen // (op 180) Load value



v_add_u32 v15, 4, v15 // (op 3589) 
s_barrier  // (op 4059) 
s_waitcnt vmcnt(18)
s_barrier  // (op 4033) 






v_add_u32 v18, 69632, v14 // (op 178) 
ds_read_b128 v[24:27], v3 offset:32768 // (op 169) Load local data 
ds_read_b128 v[28:31], v3 offset:32832 // (op 169) Load local data 
s_waitcnt vmcnt(1)
ds_write_b32 v18, v22 // (op 178) Store local data 

s_barrier  // (op 4049) 
v_add_u32 v18, 69636, v14 // (op 178) 



ds_read_b128 v[32:35], v5 // (op 171) Load local data 
ds_read_b128 v[36:39], v5 offset:64 // (op 171) Load local data 
s_waitcnt vmcnt(0)
ds_write_b32 v18, v23 // (op 178) Store local data 




v_mov_b32 v18, v97 // (op 173) 

buffer_load_ubyte v19, v18, s[56:59], 0 offen offset: 0 // (op 173) Load value

v_add_u32 v97, 4, v97 // (op 3525) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 4066) 



v_add_u32 v18, 69632, v0 // (op 175) 
ds_read_u8 v20, v18 // (op 175) Load local data 




ds_read_b128 v[40:43], v5 offset:4096 // (op 320) Load local data 
ds_read_b128 v[44:47], v5 offset:4160 // (op 320) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[24:31], v[32:39], a[0:3], v19, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 176) 





v_add_u32 v18, 69760, v0 // (op 323) 
ds_read_u8 v20, v18 // (op 323) Load local data 




ds_read_b128 v[32:35], v5 offset:8192 // (op 476) Load local data 
ds_read_b128 v[36:39], v5 offset:8256 // (op 476) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[24:31], v[40:47], a[4:7], v19, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 324) 





v_add_u32 v18, 69888, v0 // (op 479) 
ds_read_u8 v20, v18 // (op 479) Load local data 




ds_read_b128 v[40:43], v5 offset:12288 // (op 632) Load local data 
ds_read_b128 v[44:47], v5 offset:12352 // (op 632) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[24:31], v[32:39], a[8:11], v19, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 480) 





v_add_u32 v18, 70016, v0 // (op 635) 
ds_read_u8 v20, v18 // (op 635) Load local data 




ds_read_b128 v[32:35], v5 offset:16384 // (op 788) Load local data 
ds_read_b128 v[36:39], v5 offset:16448 // (op 788) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[24:31], v[40:47], a[12:15], v19, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 636) 





v_add_u32 v18, 70144, v0 // (op 791) 
ds_read_u8 v20, v18 // (op 791) Load local data 




ds_read_b128 v[40:43], v5 offset:20480 // (op 944) Load local data 
ds_read_b128 v[44:47], v5 offset:20544 // (op 944) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[24:31], v[32:39], a[16:19], v19, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 792) 





v_add_u32 v18, 70272, v0 // (op 947) 
ds_read_u8 v20, v18 // (op 947) Load local data 




ds_read_b128 v[32:35], v5 offset:24576 // (op 1100) Load local data 
ds_read_b128 v[36:39], v5 offset:24640 // (op 1100) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[24:31], v[40:47], a[20:23], v19, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 948) 





v_add_u32 v18, 70400, v0 // (op 1103) 
ds_read_u8 v20, v18 // (op 1103) Load local data 




ds_read_b128 v[40:43], v5 offset:28672 // (op 1256) Load local data 
ds_read_b128 v[44:47], v5 offset:28736 // (op 1256) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[24:31], v[32:39], a[24:27], v19, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1104) 





v_add_u32 v18, 70528, v0 // (op 1259) 
ds_read_u8 v20, v18 // (op 1259) Load local data 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[24:31], v[40:47], a[28:31], v19, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1260) 




s_ashr_i64 s[6:7], s[36:37], 7 // (op 1248) 
s_mov_b32 s11, s6 // (op 1248) convert

s_cmp_lt_u32 s10, s11 // (op 1248) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_ForLoopTop_KLoopTail_1248 // (op 1248) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_ForLoopTop_KLoopTail_1248 if true)
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_ForLoopBottom_KLoopTail_1248:






















s_barrier  // (op 4148) 







v_cmp_eq_f32 vcc, 0.00000, v10 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_ConditionalFalse_0_beta_123
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
s_branch RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_ConditionalBottom_0_beta_123
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_ConditionalFalse_0_beta_123:


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
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_ConditionalBottom_0_beta_123:
















s_endpgm  // End of RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2
.LRR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_end:
.size RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2, .LRR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_end-RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2
  .amdhsa_next_free_vgpr 148
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 71680 // lds bytes
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
    DataType: B8
    DataTypeA: B8
    DataTypeB: B8
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
  - .name: RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2
    .symbol: RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 71680
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 64
    .vgpr_count: 116
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
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: BF8, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BF8, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: BF8, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BF8, pointerType: PointerGlobal}
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

