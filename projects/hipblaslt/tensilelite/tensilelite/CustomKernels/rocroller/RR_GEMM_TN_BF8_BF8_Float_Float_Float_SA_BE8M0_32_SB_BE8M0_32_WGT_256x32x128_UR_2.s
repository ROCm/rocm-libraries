

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2
.p2align 8
.type RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2,@function
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 6, v1 // (op 3918) 
v_and_b32 v3, 1, v0 // (op 3918) 

v_and_b32 v0, 15, v1 // (op 3918) 
v_lshl_add_u32 v4, v3, 4, v0 // (op 3918) 


v_and_b32 v0, 48, v1 // (op 3918) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 3918) 


v_lshrrev_b32 v0, 6, v1 // (op 4077) 
v_and_b32 v4, 1, v0 // (op 4077) 

v_and_b32 v0, 15, v1 // (op 4077) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 4077) 


v_and_b32 v0, 48, v1 // (op 4077) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 4077) 


v_lshrrev_b32 v0, 7, v1 // (op 3805) 
v_and_b32 v5, 63, v1 // (op 3805) 
v_lshrrev_b32 v6, 4, v5 // (op 3805) 
v_lshlrev_b32 v7, 2, v6 // (op 3805) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 3805) 


v_lshl_add_u32 v0, s16, 8, v6 // (op 3805) 

v_lshrrev_b32 v6, 2, v5 // (op 3805) 

v_and_b32 v5, 3, v6 // (op 3805) 

v_and_b32 v6, 3, v1 // (op 3805) 
v_lshl_add_u32 v7, v5, 2, v6 // (op 3805) 


v_lshrrev_b32 v5, 6, v1 // (op 3805) 
v_and_b32 v6, 1, v5 // (op 3805) 

v_lshl_add_u32 v5, v6, 4, v7 // (op 3805) 


v_lshl_add_u32 v6, s17, 5, v5 // (op 3805) 

v_mul_lo_u32 v5, v6, s2 // (op 3805) 

v_add_lshl_u32 v6, v0, v5, 2 // (op 3805) 


v_lshrrev_b32 v0, 7, v1 // (op 3866) 
v_and_b32 v5, 15, v1 // (op 3866) 
v_lshl_add_u32 v7, v0, 4, v5 // (op 3866) 


v_and_b32 v0, 48, v1 // (op 3866) 
v_lshl_add_u32 v5, v7, 7, v0 // (op 3866) 


v_lshrrev_b32 v0, 7, v1 // (op 3976) 
v_and_b32 v7, 15, v1 // (op 3976) 
s_mul_i32 s1, s2, s9 // (op 3807) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 3807) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 3807) least significant
s_mul_hi_u32 s19, s2, s8 // (op 3807) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3807) most significant: sum
s_add_u32 s1, s1, s19 // (op 3807) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 3807) 

v_lshl_add_u32 v8, v0, 4, v7 // (op 3976) 


v_and_b32 v0, 63, v1 // (op 3976) 
v_lshrrev_b32 v7, 4, v0 // (op 3976) 

v_lshl_add_u32 v0, v8, 2, v7 // (op 3976) 


v_lshrrev_b32 v7, 7, v1 // (op 3992) 
v_and_b32 v8, 63, v1 // (op 3992) 
v_lshrrev_b32 v9, 4, v8 // (op 3992) 
v_lshlrev_b32 v10, 2, v9 // (op 3992) 

v_lshl_add_u32 v9, v7, 4, v10 // (op 3992) 


v_lshl_add_u32 v7, s16, 8, v9 // (op 3992) 

v_lshrrev_b32 v9, 2, v8 // (op 3992) 

v_and_b32 v8, 3, v9 // (op 3992) 

v_and_b32 v9, 3, v1 // (op 3992) 
v_lshl_add_u32 v10, v8, 2, v9 // (op 3992) 


v_lshrrev_b32 v8, 6, v1 // (op 3992) 
v_and_b32 v9, 1, v8 // (op 3992) 

v_lshl_add_u32 v8, v9, 4, v10 // (op 3992) 


v_lshl_add_u32 v9, s17, 5, v8 // (op 3992) 

v_mul_lo_u32 v8, v9, s4 // (op 3992) 

v_add_lshl_u32 v9, v7, v8, 2 // (op 3992) 


v_lshrrev_b32 v7, 7, v1 // (op 4038) 
v_and_b32 v8, 15, v1 // (op 4038) 
v_lshl_add_u32 v10, v7, 4, v8 // (op 4038) 


v_and_b32 v7, 48, v1 // (op 4038) 
v_lshl_add_u32 v8, v10, 7, v7 // (op 4038) 


v_lshrrev_b32 v7, 7, v1 // (op 4106) 
v_and_b32 v10, 15, v1 // (op 4106) 
s_mul_i32 s1, s4, s9 // (op 3994) most significant: low of low * high
v_lshl_add_u32 v11, v7, 4, v10 // (op 4106) 


v_and_b32 v7, 63, v1 // (op 4106) 
v_lshrrev_b32 v10, 4, v7 // (op 4106) 

v_lshl_add_u32 v7, v11, 2, v10 // (op 4106) 


s_mul_i32 s38, s5, s8 // (op 3994) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 3994) least significant
s_mul_hi_u32 s39, s4, s8 // (op 3994) most significant: high of low * low
s_add_u32 s1, s1, s38 // (op 3994) most significant: sum
s_add_u32 s1, s1, s39 // (op 3994) most significant: sum


s_lshl_b64 s[38:39], s[0:1], 2 // (op 3994) 

s_mov_b32 s40, s6 // (op 3807) reg expression
s_mov_b32 s41, s7 // (op 3807) reg expression
s_mov_b32 s42, s18 // (op 3807) reg expression
s_mov_b32 s43, 131072 // (op 3807) call()

s_mov_b32 s44, s10 // (op 3994) reg expression

s_mov_b32 s45, s11 // (op 3994) reg expression
s_mov_b32 s46, s38 // (op 3994) reg expression
s_mov_b32 s47, 131072 // (op 3994) call()


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
v_accvgpr_write a24, 0.00000 // (op 1119) call()
v_accvgpr_write a25, 0.00000 // (op 1119) call()
v_accvgpr_write a26, 0.00000 // (op 1119) call()
v_accvgpr_write a27, 0.00000 // (op 1119) call()
v_accvgpr_write a28, 0.00000 // (op 1281) call()
v_accvgpr_write a29, 0.00000 // (op 1281) call()
v_accvgpr_write a30, 0.00000 // (op 1281) call()
v_accvgpr_write a31, 0.00000 // (op 1281) call()
v_lshrrev_b32 v12, 3, v1 // (op 3572) 
v_lshl_add_u32 v13, s16, 8, v12 // (op 3572) 

v_and_b32 v12, -8, v1 // (op 3572) 
v_sub_u32 v14, v1, v12 // (op 3572) 

v_lshlrev_b32 v12, 4, v14 // (op 3572) 

v_mov_b32 v15, 0 // (op 3572) convert
v_mov_b32 v14, v12 // (op 3572) convert

v_mul_lo_u32 v17, v13, s13 // (op 3572) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 3572) most significant: high of low * low
v_mul_lo_u32 v16, v13, s12 // (op 3572) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 3572) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 3572) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 3572) most significant half




s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 3574) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 3574) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 3574) least significant
s_mul_hi_u32 s7, s12, s22 // (op 3574) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3574) most significant: sum
s_add_u32 s1, s1, s7 // (op 3574) most significant: sum


s_mov_b32 s48, s20 // (op 3574) reg expression
s_mov_b32 s49, s21 // (op 3574) reg expression
s_mov_b32 s50, s0 // (op 3574) reg expression
s_mov_b32 s51, 131072 // (op 3574) call()

v_lshrrev_b32 v12, 3, v1 // (op 3591) 
v_lshl_add_u32 v13, s16, 8, v12 // (op 3591) 

v_and_b32 v12, -8, v1 // (op 3591) 
v_sub_u32 v14, v1, v12 // (op 3591) 

v_lshlrev_b32 v12, 4, v14 // (op 3591) 

v_mov_b32 v15, 0 // (op 3591) convert
v_mov_b32 v14, v12 // (op 3591) convert

v_mul_lo_u32 v19, v13, s13 // (op 3591) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 3591) most significant: high of low * low
v_mul_lo_u32 v18, v13, s12 // (op 3591) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 3591) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v14 // (op 3591) least significant half
v_addc_co_u32 v19, s[0:1], v19, v15, s[0:1] // (op 3591) most significant half



s_mul_i32 s1, s12, s23 // (op 3593) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 3593) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 3593) least significant
s_mul_hi_u32 s7, s12, s22 // (op 3593) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3593) most significant: sum
s_add_u32 s1, s1, s7 // (op 3593) most significant: sum


s_mov_b32 s48, s20 // (op 3593) reg expression
s_mov_b32 s49, s21 // (op 3593) reg expression
s_mov_b32 s50, s0 // (op 3593) reg expression
s_mov_b32 s51, 131072 // (op 3593) call()

v_and_b32 v12, -8, v1 // (op 3636) 
v_sub_u32 v13, v1, v12 // (op 3636) 

v_lshlrev_b32 v12, 4, v13 // (op 3636) 

v_mov_b32 v15, 0 // (op 3636) convert
v_mov_b32 v14, v12 // (op 3636) convert

v_lshrrev_b32 v12, 3, v1 // (op 3636) 
v_lshl_add_u32 v13, s17, 5, v12 // (op 3636) 

v_mul_lo_u32 v21, v13, s25 // (op 3636) most significant: low of low * high
v_mul_hi_u32 v12, v13, s24 // (op 3636) most significant: high of low * low
v_mul_lo_u32 v20, v13, s24 // (op 3636) least significant: low of low * low
v_add_u32 v21, v21, v12 // (op 3636) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v20 // (op 3636) least significant half
v_addc_co_u32 v13, s[0:1], v15, v21, s[0:1] // (op 3636) most significant half



s_mul_i32 s1, s24, s9 // (op 3638) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 3638) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 3638) least significant
s_mul_hi_u32 s7, s24, s8 // (op 3638) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3638) most significant: sum
s_add_u32 s1, s1, s7 // (op 3638) most significant: sum


s_mov_b32 s52, s26 // (op 3638) reg expression
s_mov_b32 s53, s27 // (op 3638) reg expression
s_mov_b32 s54, s0 // (op 3638) reg expression
s_mov_b32 s55, 131072 // (op 3638) call()

v_and_b32 v14, -8, v1 // (op 3655) 
v_sub_u32 v15, v1, v14 // (op 3655) 

v_lshlrev_b32 v14, 4, v15 // (op 3655) 

v_mov_b32 v21, 0 // (op 3655) convert
v_mov_b32 v20, v14 // (op 3655) convert

v_lshrrev_b32 v14, 3, v1 // (op 3655) 
v_lshl_add_u32 v15, s17, 5, v14 // (op 3655) 

v_mul_lo_u32 v23, v15, s25 // (op 3655) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 3655) most significant: high of low * low
v_mul_lo_u32 v22, v15, s24 // (op 3655) least significant: low of low * low
v_add_u32 v23, v23, v14 // (op 3655) most significant: sum


v_add_co_u32 v14, s[0:1], v20, v22 // (op 3655) least significant half
v_addc_co_u32 v15, s[0:1], v21, v23, s[0:1] // (op 3655) most significant half



s_mul_i32 s1, s24, s9 // (op 3657) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 3657) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 3657) least significant
s_mul_hi_u32 s7, s24, s8 // (op 3657) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3657) most significant: sum
s_add_u32 s1, s1, s7 // (op 3657) most significant: sum


s_mov_b32 s52, s26 // (op 3657) reg expression
s_mov_b32 s53, s27 // (op 3657) reg expression
s_mov_b32 s54, s0 // (op 3657) reg expression
s_mov_b32 s55, 131072 // (op 3657) call()

v_lshl_add_u32 v20, s16, 8, v1 // (op 3700) 
v_mul_lo_u32 v21, v20, s28 // (op 3700) 

s_mul_i32 s1, s28, s23 // (op 3702) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 3702) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 3702) least significant
s_mul_hi_u32 s7, s28, s22 // (op 3702) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3702) most significant: sum
s_add_u32 s1, s1, s7 // (op 3702) most significant: sum


s_mov_b32 s56, s30 // (op 3702) reg expression
s_mov_b32 s57, s31 // (op 3702) reg expression
s_mov_b32 s58, s0 // (op 3702) reg expression
s_mov_b32 s59, 131072 // (op 3702) call()

v_lshl_add_u32 v20, s16, 8, v1 // (op 3719) 
v_mul_lo_u32 v22, v20, s28 // (op 3719) 

s_mul_i32 s1, s28, s23 // (op 3721) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 3721) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 3721) least significant
s_mul_hi_u32 s7, s28, s22 // (op 3721) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3721) most significant: sum
s_add_u32 s1, s1, s7 // (op 3721) most significant: sum


s_mov_b32 s56, s30 // (op 3721) reg expression
s_mov_b32 s57, s31 // (op 3721) reg expression
s_mov_b32 s58, s0 // (op 3721) reg expression
s_mov_b32 s59, 131072 // (op 3721) call()

v_lshrrev_b32 v20, 6, v1 // (op 3764) 
v_and_b32 v23, 1, v20 // (op 3764) 

v_and_b32 v20, 15, v1 // (op 3764) 
v_lshl_add_u32 v24, v23, 4, v20 // (op 3764) 


v_lshl_add_u32 v20, s17, 5, v24 // (op 3764) 

v_mul_lo_u32 v23, v20, s32 // (op 3764) 

v_and_b32 v20, 63, v1 // (op 3764) 
v_lshrrev_b32 v24, 4, v20 // (op 3764) 

v_add_u32 v20, v24, v23 // (op 3764) 


s_mul_i32 s1, s32, s9 // (op 3766) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 3766) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 3766) least significant
s_mul_hi_u32 s7, s32, s8 // (op 3766) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3766) most significant: sum
s_add_u32 s1, s1, s7 // (op 3766) most significant: sum


s_mov_b32 s60, s34 // (op 3766) reg expression
s_mov_b32 s61, s35 // (op 3766) reg expression
s_mov_b32 s62, s0 // (op 3766) reg expression
s_mov_b32 s63, 131072 // (op 3766) call()

v_lshrrev_b32 v23, 6, v1 // (op 3783) 
v_and_b32 v24, 1, v23 // (op 3783) 

v_and_b32 v23, 15, v1 // (op 3783) 
v_lshl_add_u32 v25, v24, 4, v23 // (op 3783) 


v_lshl_add_u32 v23, s17, 5, v25 // (op 3783) 

v_mul_lo_u32 v24, v23, s32 // (op 3783) 

v_and_b32 v23, 63, v1 // (op 3783) 
v_lshrrev_b32 v25, 4, v23 // (op 3783) 

v_add_u32 v23, v25, v24 // (op 3783) 


s_mul_i32 s1, s32, s9 // (op 3785) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 3785) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 3785) least significant
s_mul_hi_u32 s7, s32, s8 // (op 3785) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3785) most significant: sum
s_add_u32 s1, s1, s7 // (op 3785) most significant: sum


s_mov_b32 s60, s34 // (op 3785) reg expression
s_mov_b32 s61, s35 // (op 3785) reg expression
s_mov_b32 s62, s0 // (op 3785) reg expression
s_mov_b32 s63, 131072 // (op 3785) call()




buffer_load_ubyte v24, v20, s[60:63], 0 offen offset: 0 // (op 3252) Load value
v_and_b32 v25, -8, v1 // (op 3906) 
v_sub_u32 v26, v1, v25 // (op 3906) 

v_lshlrev_b32 v25, 4, v26 // (op 3906) 

v_lshrrev_b32 v26, 3, v1 // (op 3906) 
v_lshl_add_u32 v27, v26, 7, v25 // (op 3906) 


s_nop 0
v_readfirstlane_b32 s0, v27 // (op 3906) Wait state hazard: VALU Write Readlane Hazard




s_nop 1
s_mov_b32 m0, s0 // (op 3514) reg expression
buffer_load_dwordx4 v12, s[52:55], 0 offen offset: 0 lds // (op 3514) Load value direct to lds

v_and_b32 v25, -8, v1 // (op 3849) 
v_sub_u32 v26, v1, v25 // (op 3849) 

v_lshlrev_b32 v25, 4, v26 // (op 3849) 

v_lshrrev_b32 v26, 3, v1 // (op 3849) 
v_lshl_add_u32 v27, v26, 7, v25 // (op 3849) 


s_nop 0
v_readfirstlane_b32 s0, v27 // (op 3849) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 5 // (op 3519) 
v_mov_b64 v[26:27], s[6:7] // (op 3519) convert


v_mov_b64 v[28:29], v[16:17] // (op 3519) 
s_add_u32 s1, 4096, s0 // (op 3519) 
s_mov_b32 m0, s1 // (op 3519) reg expression
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 3519) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 3519) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3519) most significant half

s_add_u32 m0, m0, 4096 // (op 3519) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 3519) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 3519) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3519) most significant half

s_add_u32 m0, m0, 4096 // (op 3519) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 3519) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 3519) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3519) most significant half

s_add_u32 m0, m0, 4096 // (op 3519) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 3519) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 3519) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3519) most significant half

s_add_u32 m0, m0, 4096 // (op 3519) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 3519) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 3519) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3519) most significant half

s_add_u32 m0, m0, 4096 // (op 3519) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 3519) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 3519) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3519) most significant half

s_add_u32 m0, m0, 4096 // (op 3519) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 3519) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 3519) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3519) most significant half

s_add_u32 m0, m0, 4096 // (op 3519) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 3519) Load value direct to lds







buffer_load_dword v25, v21, s[56:59], 0 offen // (op 3262) Load value



buffer_load_ubyte v26, v20, s[60:63], 0 offen offset:4 // (op 3266) Load value

v_and_b32 v20, -8, v1 // (op 4065) 
v_sub_u32 v27, v1, v20 // (op 4065) 

v_lshlrev_b32 v20, 4, v27 // (op 4065) 

v_lshrrev_b32 v27, 3, v1 // (op 4065) 
v_lshl_add_u32 v28, v27, 7, v20 // (op 4065) 


s_nop 0
v_readfirstlane_b32 s0, v28 // (op 4065) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s1, 36864, s0 // (op 3524) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 3524) reg expression
s_mov_b32 s6, 128 // (op 3524) call()
buffer_load_dwordx4 v12, s[52:55], s6 offen offset: 0 lds // (op 3524) Load value direct to lds




v_and_b32 v20, -8, v1 // (op 4021) 
v_sub_u32 v12, v1, v20 // (op 4021) 

v_lshlrev_b32 v13, 4, v12 // (op 4021) 

v_lshrrev_b32 v12, 3, v1 // (op 4021) 
v_lshl_add_u32 v20, v12, 7, v13 // (op 4021) 


s_nop 0
v_readfirstlane_b32 s0, v20 // (op 4021) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 5 // (op 3526) 
v_mov_b64 v[12:13], s[6:7] // (op 3526) convert


v_mov_b64 v[28:29], v[16:17] // (op 3526) 
s_add_u32 s1, 40960, s0 // (op 3526) 
s_mov_b32 m0, s1 // (op 3526) reg expression
s_mov_b32 s6, 128 // (op 3526) call()
buffer_load_dwordx4 v28, s[48:51], s6 offen offset: 0 lds // (op 3526) Load value direct to lds
v_add_co_u32 v28, s[10:11], v28, v12 // (op 3526) least significant half
v_addc_co_u32 v29, s[10:11], 0, v29, s[10:11] // (op 3526) most significant half

s_add_u32 m0, m0, 4096 // (op 3526) 
buffer_load_dwordx4 v28, s[48:51], s6 offen offset: 0 lds // (op 3526) Load value direct to lds
v_add_co_u32 v28, s[10:11], v28, v12 // (op 3526) least significant half
v_addc_co_u32 v29, s[10:11], 0, v29, s[10:11] // (op 3526) most significant half

s_add_u32 m0, m0, 4096 // (op 3526) 
buffer_load_dwordx4 v28, s[48:51], s6 offen offset: 0 lds // (op 3526) Load value direct to lds
v_add_co_u32 v28, s[10:11], v28, v12 // (op 3526) least significant half
v_addc_co_u32 v29, s[10:11], 0, v29, s[10:11] // (op 3526) most significant half

s_add_u32 m0, m0, 4096 // (op 3526) 
buffer_load_dwordx4 v28, s[48:51], s6 offen offset: 0 lds // (op 3526) Load value direct to lds
v_add_co_u32 v28, s[10:11], v28, v12 // (op 3526) least significant half
v_addc_co_u32 v29, s[10:11], 0, v29, s[10:11] // (op 3526) most significant half

s_add_u32 m0, m0, 4096 // (op 3526) 
buffer_load_dwordx4 v28, s[48:51], s6 offen offset: 0 lds // (op 3526) Load value direct to lds
v_add_co_u32 v28, s[10:11], v28, v12 // (op 3526) least significant half
v_addc_co_u32 v29, s[10:11], 0, v29, s[10:11] // (op 3526) most significant half

s_add_u32 m0, m0, 4096 // (op 3526) 
buffer_load_dwordx4 v28, s[48:51], s6 offen offset: 0 lds // (op 3526) Load value direct to lds
v_add_co_u32 v28, s[10:11], v28, v12 // (op 3526) least significant half
v_addc_co_u32 v29, s[10:11], 0, v29, s[10:11] // (op 3526) most significant half

s_add_u32 m0, m0, 4096 // (op 3526) 
buffer_load_dwordx4 v28, s[48:51], s6 offen offset: 0 lds // (op 3526) Load value direct to lds
v_add_co_u32 v28, s[10:11], v28, v12 // (op 3526) least significant half
v_addc_co_u32 v29, s[10:11], 0, v29, s[10:11] // (op 3526) most significant half

s_add_u32 m0, m0, 4096 // (op 3526) 
buffer_load_dwordx4 v28, s[48:51], s6 offen offset: 0 lds // (op 3526) Load value direct to lds









buffer_load_dword v20, v21, s[56:59], 0 offen offset:4 // (op 3276) Load value

v_lshlrev_b32 v21, 2, v1 // (op 3959) 



v_add_u32 v12, 73728, v21 // (op 3280) 
s_waitcnt vmcnt(11)
ds_write_b32 v12, v25 // (op 3280) Store local data 


s_barrier  // (op 4181) 



ds_read_b128 v[28:31], v3 // (op 3333) Load local data 
ds_read_b128 v[32:35], v3 offset:64 // (op 3333) Load local data 
s_barrier  // (op 4165) 



ds_read_b128 v[36:39], v5 offset:4096 // (op 3294) Load local data 
ds_read_b128 v[40:43], v5 offset:4160 // (op 3294) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 4197) 



v_add_u32 v21, 73728, v0 // (op 3337) 
ds_read_u8 v27, v21 // (op 3337) Load local data 




ds_read_b128 v[44:47], v5 offset:8192 // (op 3299) Load local data 
ds_read_b128 v[48:51], v5 offset:8256 // (op 3299) Load local data 



v_add_u32 v21, 73856, v0 // (op 3342) 
ds_read_u8 v12, v21 // (op 3342) Load local data 




ds_read_b128 v[52:55], v5 offset:12288 // (op 3304) Load local data 
ds_read_b128 v[56:59], v5 offset:12352 // (op 3304) Load local data 



v_add_u32 v13, 73984, v0 // (op 3347) 
ds_read_u8 v21, v13 // (op 3347) Load local data 




ds_read_b128 v[60:63], v5 offset:16384 // (op 3309) Load local data 
ds_read_b128 v[64:67], v5 offset:16448 // (op 3309) Load local data 



v_add_u32 v13, 74112, v0 // (op 3352) 
ds_read_u8 v16, v13 // (op 3352) Load local data 




ds_read_b128 v[68:71], v5 offset:20480 // (op 3314) Load local data 
ds_read_b128 v[72:75], v5 offset:20544 // (op 3314) Load local data 



v_add_u32 v13, 74240, v0 // (op 3357) 
ds_read_u8 v17, v13 // (op 3357) Load local data 




ds_read_b128 v[76:79], v5 offset:24576 // (op 3319) Load local data 
ds_read_b128 v[80:83], v5 offset:24640 // (op 3319) Load local data 



v_add_u32 v13, 74368, v0 // (op 3362) 
ds_read_u8 v84, v13 // (op 3362) Load local data 




ds_read_b128 v[86:89], v5 offset:28672 // (op 3324) Load local data 
ds_read_b128 v[90:93], v5 offset:28736 // (op 3324) Load local data 



v_add_u32 v13, 74496, v0 // (op 3367) 
ds_read_u8 v85, v13 // (op 3367) Load local data 




ds_read_b128 v[94:97], v5 offset:32768 // (op 3329) Load local data 
ds_read_b128 v[98:101], v5 offset:32832 // (op 3329) Load local data 



v_add_u32 v13, 74624, v0 // (op 3372) 
ds_read_u8 v102, v13 // (op 3372) Load local data 

v_and_b32 v13, -8, v1 // (op 3821) 
v_sub_u32 v103, v1, v13 // (op 3821) 

v_lshlrev_b32 v13, 4, v103 // (op 3821) 

v_lshrrev_b32 v103, 3, v1 // (op 3821) 
v_lshl_add_u32 v104, v103, 7, v13 // (op 3821) 


s_nop 0
v_readfirstlane_b32 s0, v104 // (op 3821) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v13, -8, v1 // (op 3878) 
v_sub_u32 v103, v1, v13 // (op 3878) 

v_lshlrev_b32 v13, 4, v103 // (op 3878) 

v_lshrrev_b32 v103, 3, v1 // (op 3878) 
v_lshl_add_u32 v104, v103, 7, v13 // (op 3878) 


s_nop 0
v_readfirstlane_b32 s1, v104 // (op 3878) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v13, 2, v1 // (op 3930) 
v_and_b32 v103, -8, v1 // (op 4006) 
v_sub_u32 v104, v1, v103 // (op 4006) 

v_lshlrev_b32 v103, 4, v104 // (op 4006) 

v_lshrrev_b32 v104, 3, v1 // (op 4006) 
v_lshl_add_u32 v105, v104, 7, v103 // (op 4006) 


s_nop 0
v_readfirstlane_b32 s6, v105 // (op 4006) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v103, -8, v1 // (op 4050) 
v_sub_u32 v104, v1, v103 // (op 4050) 

v_lshlrev_b32 v103, 4, v104 // (op 4050) 

v_lshrrev_b32 v104, 3, v1 // (op 4050) 
v_lshl_add_u32 v105, v104, 7, v103 // (op 4050) 


s_nop 0
v_readfirstlane_b32 s7, v105 // (op 4050) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v103, 2, v1 // (op 4089) 
s_mov_b32 s10, 0 // (op 1283) call()
s_ashr_i64 s[14:15], s[36:37], 7 // (op 1282) 
s_mov_b32 s11, s14 // (op 1282) convert

s_and_b32 s14, s11, 4294967294 // (op 1282) 

s_cmp_lt_u32 s10, s14 // (op 1282) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ForLoopBottom_KLoop_1282 // (op 1282) Condition: Top (jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ForLoopBottom_KLoop_1282 if false)
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ForLoopTop_KLoop_1282:
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:43], v[28:35], a[0:3], v27, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 82) 
s_add_u32 s10, s10, 2 // (op 1284) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[44:51], v[28:35], a[4:7], v12, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 346) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[52:59], v[28:35], a[8:11], v21, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 508) 
s_waitcnt lgkmcnt(12)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[60:67], v[28:35], a[12:15], v16, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 670) 
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[68:75], v[28:35], a[16:19], v17, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 832) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[76:83], v[28:35], a[20:23], v84, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 994) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[86:93], v[28:35], a[24:27], v85, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1156) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[94:101], v[28:35], a[28:31], v102, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1318) 


v_mov_b32 v104, v23 // (op 80) 

buffer_load_ubyte v24, v104, s[60:63], 0 offen offset:8 // (op 80) Load value
s_barrier  // (op 4184) 


v_mov_b64 v[106:107], v[14:15] // (op 3486) 

s_mov_b32 m0, s1 // (op 3486) reg expression
s_mov_b32 s11, 256 // (op 3486) call()
buffer_load_dwordx4 v106, s[52:55], s11 offen offset: 0 lds // (op 3486) Load value direct to lds

s_barrier  // (op 4168) 

s_lshl_b64 s[14:15], s[12:13], 5 // (op 3481) 
v_mov_b64 v[108:109], s[14:15] // (op 3481) convert


v_mov_b64 v[110:111], v[18:19] // (op 3481) 
v_mov_b64 v[112:113], v[110:111] // (op 3481) 
s_add_u32 s11, 4096, s0 // (op 3481) 
s_mov_b32 m0, s11 // (op 3481) reg expression
s_mov_b32 s14, 256 // (op 3481) call()
buffer_load_dwordx4 v112, s[48:51], s14 offen offset: 0 lds // (op 3481) Load value direct to lds
v_add_co_u32 v112, s[18:19], v112, v108 // (op 3481) least significant half
v_addc_co_u32 v113, s[18:19], 0, v113, s[18:19] // (op 3481) most significant half

s_add_u32 m0, m0, 4096 // (op 3481) 
buffer_load_dwordx4 v112, s[48:51], s14 offen offset: 0 lds // (op 3481) Load value direct to lds
v_add_co_u32 v112, s[18:19], v112, v108 // (op 3481) least significant half
v_addc_co_u32 v113, s[18:19], 0, v113, s[18:19] // (op 3481) most significant half

s_add_u32 m0, m0, 4096 // (op 3481) 
buffer_load_dwordx4 v112, s[48:51], s14 offen offset: 0 lds // (op 3481) Load value direct to lds
v_add_co_u32 v112, s[18:19], v112, v108 // (op 3481) least significant half
v_addc_co_u32 v113, s[18:19], 0, v113, s[18:19] // (op 3481) most significant half

s_add_u32 m0, m0, 4096 // (op 3481) 
buffer_load_dwordx4 v112, s[48:51], s14 offen offset: 0 lds // (op 3481) Load value direct to lds
v_add_co_u32 v112, s[18:19], v112, v108 // (op 3481) least significant half
v_addc_co_u32 v113, s[18:19], 0, v113, s[18:19] // (op 3481) most significant half

s_add_u32 m0, m0, 4096 // (op 3481) 
buffer_load_dwordx4 v112, s[48:51], s14 offen offset: 0 lds // (op 3481) Load value direct to lds
v_add_co_u32 v112, s[18:19], v112, v108 // (op 3481) least significant half
v_addc_co_u32 v113, s[18:19], 0, v113, s[18:19] // (op 3481) most significant half

s_add_u32 m0, m0, 4096 // (op 3481) 
buffer_load_dwordx4 v112, s[48:51], s14 offen offset: 0 lds // (op 3481) Load value direct to lds
v_add_co_u32 v112, s[18:19], v112, v108 // (op 3481) least significant half
v_addc_co_u32 v113, s[18:19], 0, v113, s[18:19] // (op 3481) most significant half

s_add_u32 m0, m0, 4096 // (op 3481) 
buffer_load_dwordx4 v112, s[48:51], s14 offen offset: 0 lds // (op 3481) Load value direct to lds
v_add_co_u32 v112, s[18:19], v112, v108 // (op 3481) least significant half
v_addc_co_u32 v113, s[18:19], 0, v113, s[18:19] // (op 3481) most significant half

s_add_u32 m0, m0, 4096 // (op 3481) 
buffer_load_dwordx4 v112, s[48:51], s14 offen offset: 0 lds // (op 3481) Load value direct to lds






v_mov_b32 v105, v22 // (op 17) 

buffer_load_dword v25, v105, s[56:59], 0 offen offset:8 // (op 17) Load value
s_barrier  // (op 4229) 



v_add_u32 v108, 74752, v103 // (op 236) 
s_waitcnt vmcnt(11)
ds_write_b32 v108, v20 // (op 236) Store local data 

s_barrier  // (op 4220) 



ds_read_b128 v[28:31], v4 offset:36864 // (op 229) Load local data 
ds_read_b128 v[32:35], v4 offset:36928 // (op 229) Load local data 
s_barrier  // (op 4214) 



ds_read_b128 v[36:39], v8 offset:40960 // (op 227) Load local data 
ds_read_b128 v[40:43], v8 offset:41024 // (op 227) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 4226) 



v_add_u32 v108, 74752, v7 // (op 231) 
ds_read_u8 v27, v108 // (op 231) Load local data 




ds_read_b128 v[44:47], v8 offset:45056 // (op 353) Load local data 
ds_read_b128 v[48:51], v8 offset:45120 // (op 353) Load local data 



v_add_u32 v108, 74880, v7 // (op 357) 
ds_read_u8 v12, v108 // (op 357) Load local data 




ds_read_b128 v[52:55], v8 offset:49152 // (op 515) Load local data 
ds_read_b128 v[56:59], v8 offset:49216 // (op 515) Load local data 



v_add_u32 v108, 75008, v7 // (op 519) 
ds_read_u8 v21, v108 // (op 519) Load local data 




ds_read_b128 v[60:63], v8 offset:53248 // (op 677) Load local data 
ds_read_b128 v[64:67], v8 offset:53312 // (op 677) Load local data 



v_add_u32 v108, 75136, v7 // (op 681) 
ds_read_u8 v16, v108 // (op 681) Load local data 




ds_read_b128 v[68:71], v8 offset:57344 // (op 839) Load local data 
ds_read_b128 v[72:75], v8 offset:57408 // (op 839) Load local data 



v_add_u32 v108, 75264, v7 // (op 843) 
ds_read_u8 v17, v108 // (op 843) Load local data 




ds_read_b128 v[76:79], v8 offset:61440 // (op 1001) Load local data 
ds_read_b128 v[80:83], v8 offset:61504 // (op 1001) Load local data 



v_add_u32 v108, 75392, v7 // (op 1005) 
ds_read_u8 v84, v108 // (op 1005) Load local data 




v_add_u32 v108, 65536, v8 // (op 1163) 
ds_read_b128 v[86:89], v108 // (op 1163) Load local data 

v_add_u32 v108, 65600, v8 // (op 1163) 
ds_read_b128 v[90:93], v108 // (op 1163) Load local data 




v_add_u32 v108, 75520, v7 // (op 1167) 
ds_read_u8 v85, v108 // (op 1167) Load local data 




v_add_u32 v108, 69632, v8 // (op 1325) 
ds_read_b128 v[94:97], v108 // (op 1325) Load local data 

v_add_u32 v108, 69696, v8 // (op 1325) 
ds_read_b128 v[98:101], v108 // (op 1325) Load local data 




v_add_u32 v108, 75648, v7 // (op 1329) 
ds_read_u8 v102, v108 // (op 1329) Load local data 

s_barrier  // (op 4304) 
s_waitcnt lgkmcnt(15)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:43], v[28:35], a[0:3], v27, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 234) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[44:51], v[28:35], a[4:7], v12, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 347) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[52:59], v[28:35], a[8:11], v21, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 509) 
s_waitcnt lgkmcnt(12)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[60:67], v[28:35], a[12:15], v16, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 671) 
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[68:75], v[28:35], a[16:19], v17, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 833) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[76:83], v[28:35], a[20:23], v84, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 995) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[86:93], v[28:35], a[24:27], v85, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1157) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[94:101], v[28:35], a[28:31], v102, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1319) 



buffer_load_ubyte v26, v104, s[60:63], 0 offen offset:12 // (op 233) Load value

s_barrier  // (op 4223) 
v_add_u32 v23, 8, v23 // (op 3775) 


s_add_u32 s11, 36864, s7 // (op 3509) 

s_mov_b32 m0, s11 // (op 3509) reg expression
s_mov_b32 s14, 384 // (op 3509) call()
buffer_load_dwordx4 v106, s[52:55], s14 offen offset: 0 lds // (op 3509) Load value direct to lds



s_barrier  // (op 4217) 
v_add_co_u32 v14, vcc, 256, v14 // (op 3647) least significant half

v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 3647) most significant half
s_lshl_b64 s[14:15], s[12:13], 5 // (op 3504) 
v_mov_b64 v[106:107], s[14:15] // (op 3504) convert


v_mov_b64 v[108:109], v[110:111] // (op 3504) 
s_add_u32 s11, 40960, s6 // (op 3504) 
s_mov_b32 m0, s11 // (op 3504) reg expression
s_mov_b32 s14, 384 // (op 3504) call()
buffer_load_dwordx4 v108, s[48:51], s14 offen offset: 0 lds // (op 3504) Load value direct to lds
v_add_co_u32 v108, s[18:19], v108, v106 // (op 3504) least significant half
v_addc_co_u32 v109, s[18:19], 0, v109, s[18:19] // (op 3504) most significant half

s_add_u32 m0, m0, 4096 // (op 3504) 
buffer_load_dwordx4 v108, s[48:51], s14 offen offset: 0 lds // (op 3504) Load value direct to lds
v_add_co_u32 v108, s[18:19], v108, v106 // (op 3504) least significant half
v_addc_co_u32 v109, s[18:19], 0, v109, s[18:19] // (op 3504) most significant half

s_add_u32 m0, m0, 4096 // (op 3504) 
buffer_load_dwordx4 v108, s[48:51], s14 offen offset: 0 lds // (op 3504) Load value direct to lds
v_add_co_u32 v108, s[18:19], v108, v106 // (op 3504) least significant half
v_addc_co_u32 v109, s[18:19], 0, v109, s[18:19] // (op 3504) most significant half

s_add_u32 m0, m0, 4096 // (op 3504) 
buffer_load_dwordx4 v108, s[48:51], s14 offen offset: 0 lds // (op 3504) Load value direct to lds
v_add_co_u32 v108, s[18:19], v108, v106 // (op 3504) least significant half
v_addc_co_u32 v109, s[18:19], 0, v109, s[18:19] // (op 3504) most significant half

s_add_u32 m0, m0, 4096 // (op 3504) 
buffer_load_dwordx4 v108, s[48:51], s14 offen offset: 0 lds // (op 3504) Load value direct to lds
v_add_co_u32 v108, s[18:19], v108, v106 // (op 3504) least significant half
v_addc_co_u32 v109, s[18:19], 0, v109, s[18:19] // (op 3504) most significant half

s_add_u32 m0, m0, 4096 // (op 3504) 
buffer_load_dwordx4 v108, s[48:51], s14 offen offset: 0 lds // (op 3504) Load value direct to lds
v_add_co_u32 v108, s[18:19], v108, v106 // (op 3504) least significant half
v_addc_co_u32 v109, s[18:19], 0, v109, s[18:19] // (op 3504) most significant half

s_add_u32 m0, m0, 4096 // (op 3504) 
buffer_load_dwordx4 v108, s[48:51], s14 offen offset: 0 lds // (op 3504) Load value direct to lds
v_add_co_u32 v108, s[18:19], v108, v106 // (op 3504) least significant half
v_addc_co_u32 v109, s[18:19], 0, v109, s[18:19] // (op 3504) most significant half

s_add_u32 m0, m0, 4096 // (op 3504) 
buffer_load_dwordx4 v108, s[48:51], s14 offen offset: 0 lds // (op 3504) Load value direct to lds








v_add_co_u32 v18, vcc, 256, v18 // (op 3583) least significant half
v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 3583) most significant half
buffer_load_dword v20, v105, s[56:59], 0 offen offset:12 // (op 238) Load value

v_add_u32 v22, 8, v22 // (op 3711) 
s_barrier  // (op 4200) 



v_add_u32 v104, 73728, v13 // (op 21) 
s_waitcnt vmcnt(11)
ds_write_b32 v104, v25 // (op 21) Store local data 

s_barrier  // (op 4190) 



ds_read_b128 v[28:31], v3 // (op 70) Load local data 
ds_read_b128 v[32:35], v3 offset:64 // (op 70) Load local data 
s_barrier  // (op 4174) 



ds_read_b128 v[36:39], v5 offset:4096 // (op 65) Load local data 
ds_read_b128 v[40:43], v5 offset:4160 // (op 65) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 4207) 



v_add_u32 v104, 73728, v0 // (op 75) 
ds_read_u8 v27, v104 // (op 75) Load local data 




ds_read_b128 v[44:47], v5 offset:8192 // (op 340) Load local data 
ds_read_b128 v[48:51], v5 offset:8256 // (op 340) Load local data 



v_add_u32 v104, 73856, v0 // (op 344) 
ds_read_u8 v12, v104 // (op 344) Load local data 




ds_read_b128 v[52:55], v5 offset:12288 // (op 502) Load local data 
ds_read_b128 v[56:59], v5 offset:12352 // (op 502) Load local data 



v_add_u32 v104, 73984, v0 // (op 506) 
ds_read_u8 v21, v104 // (op 506) Load local data 




ds_read_b128 v[60:63], v5 offset:16384 // (op 664) Load local data 
ds_read_b128 v[64:67], v5 offset:16448 // (op 664) Load local data 



v_add_u32 v104, 74112, v0 // (op 668) 
ds_read_u8 v16, v104 // (op 668) Load local data 




ds_read_b128 v[68:71], v5 offset:20480 // (op 826) Load local data 
ds_read_b128 v[72:75], v5 offset:20544 // (op 826) Load local data 



v_add_u32 v104, 74240, v0 // (op 830) 
ds_read_u8 v17, v104 // (op 830) Load local data 




ds_read_b128 v[76:79], v5 offset:24576 // (op 988) Load local data 
ds_read_b128 v[80:83], v5 offset:24640 // (op 988) Load local data 



v_add_u32 v104, 74368, v0 // (op 992) 
ds_read_u8 v84, v104 // (op 992) Load local data 




ds_read_b128 v[86:89], v5 offset:28672 // (op 1150) Load local data 
ds_read_b128 v[90:93], v5 offset:28736 // (op 1150) Load local data 



v_add_u32 v104, 74496, v0 // (op 1154) 
ds_read_u8 v85, v104 // (op 1154) Load local data 




ds_read_b128 v[94:97], v5 offset:32768 // (op 1312) Load local data 
ds_read_b128 v[98:101], v5 offset:32832 // (op 1312) Load local data 



v_add_u32 v104, 74624, v0 // (op 1316) 
ds_read_u8 v102, v104 // (op 1316) Load local data 

s_ashr_i64 s[14:15], s[36:37], 7 // (op 1282) 
s_mov_b32 s11, s14 // (op 1282) convert

s_and_b32 s14, s11, 4294967294 // (op 1282) 

s_cmp_lt_u32 s10, s14 // (op 1282) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ForLoopTop_KLoop_1282 // (op 1282) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ForLoopTop_KLoop_1282 if true)
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ForLoopBottom_KLoop_1282:




















s_barrier  // (op 4288) 
v_lshrrev_b32 v12, 3, v1 // (op 3539) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 3539) 
v_lshl_add_u32 v13, s16, 8, v12 // (op 3539) 

s_lshr_b32 s6, s0, 1 // (op 3539) 

v_and_b32 v12, -8, v1 // (op 3539) 
v_sub_u32 v14, v1, v12 // (op 3539) 

v_lshlrev_b32 v12, 4, v14 // (op 3539) 

v_lshl_add_u32 v14, s6, 8, v12 // (op 3539) 


v_mov_b32 v19, 0 // (op 3539) convert
v_mov_b32 v18, v14 // (op 3539) convert

v_mul_lo_u32 v15, v13, s13 // (op 3539) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 3539) most significant: high of low * low
v_mul_lo_u32 v14, v13, s12 // (op 3539) least significant: low of low * low
v_add_u32 v15, v15, v12 // (op 3539) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v18 // (op 3539) least significant half
v_addc_co_u32 v15, s[0:1], v15, v19, s[0:1] // (op 3539) most significant half



s_mul_i32 s1, s12, s23 // (op 3541) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 3541) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 3541) least significant
s_mul_hi_u32 s7, s12, s22 // (op 3541) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3541) most significant: sum
s_add_u32 s1, s1, s7 // (op 3541) most significant: sum


s_mov_b32 s48, s20 // (op 3541) reg expression
s_mov_b32 s49, s21 // (op 3541) reg expression
s_mov_b32 s50, s0 // (op 3541) reg expression
s_mov_b32 s51, 131072 // (op 3541) call()

v_and_b32 v12, -8, v1 // (op 3613) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 3613) 
v_sub_u32 v13, v1, v12 // (op 3613) 

s_lshr_b32 s6, s0, 1 // (op 3613) 

v_lshlrev_b32 v12, 4, v13 // (op 3613) 

v_lshl_add_u32 v13, s6, 8, v12 // (op 3613) 


v_mov_b32 v19, 0 // (op 3613) convert
v_mov_b32 v18, v13 // (op 3613) convert

v_lshrrev_b32 v12, 3, v1 // (op 3613) 
v_lshl_add_u32 v13, s17, 5, v12 // (op 3613) 

v_mul_lo_u32 v25, v13, s25 // (op 3613) most significant: low of low * high
v_mul_hi_u32 v12, v13, s24 // (op 3613) most significant: high of low * low
v_mul_lo_u32 v24, v13, s24 // (op 3613) least significant: low of low * low
v_add_u32 v25, v25, v12 // (op 3613) most significant: sum


v_add_co_u32 v12, s[0:1], v18, v24 // (op 3613) least significant half
v_addc_co_u32 v13, s[0:1], v19, v25, s[0:1] // (op 3613) most significant half



s_mul_i32 s1, s24, s9 // (op 3615) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 3615) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 3615) least significant
s_mul_hi_u32 s7, s24, s8 // (op 3615) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3615) most significant: sum
s_add_u32 s1, s1, s7 // (op 3615) most significant: sum


s_mov_b32 s52, s26 // (op 3615) reg expression
s_mov_b32 s53, s27 // (op 3615) reg expression
s_mov_b32 s54, s0 // (op 3615) reg expression
s_mov_b32 s55, 131072 // (op 3615) call()

v_lshl_add_u32 v18, s16, 8, v1 // (op 3677) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 3677) 
s_lshr_b32 s6, s0, 1 // (op 3677) 

s_lshl_b32 s7, s6, 3 // (op 3677) 

v_mul_lo_u32 v19, v18, s28 // (op 3677) 
v_add_u32 v19, v19, s7 // (op 3677) 


s_mul_i32 s1, s28, s23 // (op 3679) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 3679) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 3679) least significant
s_mul_hi_u32 s7, s28, s22 // (op 3679) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3679) most significant: sum
s_add_u32 s1, s1, s7 // (op 3679) most significant: sum


s_mov_b32 s56, s30 // (op 3679) reg expression
s_mov_b32 s57, s31 // (op 3679) reg expression
s_mov_b32 s58, s0 // (op 3679) reg expression
s_mov_b32 s59, 131072 // (op 3679) call()

v_and_b32 v18, 63, v1 // (op 3741) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 3741) 
v_lshrrev_b32 v21, 4, v18 // (op 3741) 

s_lshr_b32 s6, s0, 1 // (op 3741) 

v_lshrrev_b32 v18, 6, v1 // (op 3741) 
v_lshl_add_u32 v22, s6, 3, v21 // (op 3741) 


v_and_b32 v21, 1, v18 // (op 3741) 

v_and_b32 v18, 15, v1 // (op 3741) 
v_lshl_add_u32 v23, v21, 4, v18 // (op 3741) 


v_lshl_add_u32 v18, s17, 5, v23 // (op 3741) 

v_mul_lo_u32 v21, v18, s32 // (op 3741) 

v_add_u32 v18, v22, v21 // (op 3741) 


s_mul_i32 s1, s32, s9 // (op 3743) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 3743) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 3743) least significant
s_mul_hi_u32 s7, s32, s8 // (op 3743) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3743) most significant: sum
s_add_u32 s1, s1, s7 // (op 3743) most significant: sum


s_mov_b32 s60, s34 // (op 3743) reg expression
s_mov_b32 s61, s35 // (op 3743) reg expression
s_mov_b32 s62, s0 // (op 3743) reg expression
s_mov_b32 s63, 131072 // (op 3743) call()

v_and_b32 v21, -8, v1 // (op 3834) 
v_sub_u32 v22, v1, v21 // (op 3834) 

v_lshlrev_b32 v21, 4, v22 // (op 3834) 

v_lshrrev_b32 v22, 3, v1 // (op 3834) 
v_lshl_add_u32 v23, v22, 7, v21 // (op 3834) 


s_nop 0
v_readfirstlane_b32 s0, v23 // (op 3834) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v21, -8, v1 // (op 3891) 
v_sub_u32 v22, v1, v21 // (op 3891) 

v_lshlrev_b32 v21, 4, v22 // (op 3891) 

v_lshrrev_b32 v22, 3, v1 // (op 3891) 
v_lshl_add_u32 v23, v22, 7, v21 // (op 3891) 


s_nop 0
v_readfirstlane_b32 s1, v23 // (op 3891) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v21, 2, v1 // (op 3943) 
s_ashr_i64 s[6:7], s[36:37], 7 // (op 1288) 
s_and_b32 s10, s6, 4294967294 // (op 1288) 

s_ashr_i64 s[6:7], s[36:37], 7 // (op 1287) 
s_mov_b32 s11, s6 // (op 1287) convert

s_cmp_lt_u32 s10, s11 // (op 1287) 













s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ForLoopBottom_KLoopTail_1287 // (op 1287) Condition: Top (jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ForLoopBottom_KLoopTail_1287 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ForLoopTop_KLoopTail_1287:
s_barrier  // (op 4171) 
s_add_u32 s10, s10, 1 // (op 1289) 

s_lshl_b64 s[6:7], s[12:13], 5 // (op 3491) 
v_mov_b64 v[16:17], s[6:7] // (op 3491) convert


v_mov_b64 v[22:23], v[14:15] // (op 3491) 
v_mov_b64 v[24:25], v[22:23] // (op 3491) 
s_add_u32 s11, 4096, s0 // (op 3491) 
s_mov_b32 m0, s11 // (op 3491) reg expression
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 3491) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v16 // (op 3491) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 3491) most significant half

s_add_u32 m0, m0, 4096 // (op 3491) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 3491) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v16 // (op 3491) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 3491) most significant half

s_add_u32 m0, m0, 4096 // (op 3491) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 3491) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v16 // (op 3491) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 3491) most significant half

s_add_u32 m0, m0, 4096 // (op 3491) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 3491) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v16 // (op 3491) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 3491) most significant half

s_add_u32 m0, m0, 4096 // (op 3491) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 3491) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v16 // (op 3491) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 3491) most significant half

s_add_u32 m0, m0, 4096 // (op 3491) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 3491) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v16 // (op 3491) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 3491) most significant half

s_add_u32 m0, m0, 4096 // (op 3491) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 3491) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v16 // (op 3491) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 3491) most significant half

s_add_u32 m0, m0, 4096 // (op 3491) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 3491) Load value direct to lds




v_add_co_u32 v14, vcc, 128, v14 // (op 3529) least significant half
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 3529) most significant half
s_barrier  // (op 4187) 


v_mov_b64 v[16:17], v[12:13] // (op 3497) 

s_mov_b32 m0, s1 // (op 3497) reg expression
buffer_load_dwordx4 v16, s[52:55], 0 offen offset: 0 lds // (op 3497) Load value direct to lds

v_add_co_u32 v12, vcc, 128, v12 // (op 3605) least significant half
v_addc_co_u32 v13, vcc, 0, v13, vcc // (op 3605) most significant half


v_mov_b32 v20, v19 // (op 180) 

buffer_load_dword v16, v20, s[56:59], 0 offen // (op 180) Load value

v_add_u32 v19, 4, v19 // (op 3669) 
s_barrier  // (op 4203) 
s_waitcnt vmcnt(2)
s_barrier  // (op 4177) 






v_add_u32 v17, 73728, v21 // (op 178) 
ds_read_b128 v[22:25], v5 offset:4096 // (op 169) Load local data 
ds_read_b128 v[26:29], v5 offset:4160 // (op 169) Load local data 
s_waitcnt vmcnt(0)
ds_write_b32 v17, v16 // (op 178) Store local data 


s_barrier  // (op 4193) 



ds_read_b128 v[30:33], v3 // (op 171) Load local data 
ds_read_b128 v[34:37], v3 offset:64 // (op 171) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 4210) 



v_add_u32 v20, 73728, v0 // (op 173) 
ds_read_u8 v16, v20 // (op 173) Load local data 



v_mov_b32 v17, v18 // (op 175) 

buffer_load_ubyte v20, v17, s[60:63], 0 offen offset: 0 // (op 175) Load value

v_add_u32 v18, 4, v18 // (op 3733) 



s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:29], v[30:37], a[0:3], v16, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 176) 
ds_read_b128 v[38:41], v5 offset:8192 // (op 321) Load local data 


ds_read_b128 v[42:45], v5 offset:8256 // (op 321) Load local data 



v_add_u32 v16, 73856, v0 // (op 325) 
ds_read_u8 v17, v16 // (op 325) Load local data 




s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[38:45], v[30:37], a[4:7], v17, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 327) 
ds_read_b128 v[22:25], v5 offset:12288 // (op 483) Load local data 


ds_read_b128 v[26:29], v5 offset:12352 // (op 483) Load local data 



v_add_u32 v16, 73984, v0 // (op 487) 
ds_read_u8 v17, v16 // (op 487) Load local data 




s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[22:29], v[30:37], a[8:11], v17, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 489) 
ds_read_b128 v[38:41], v5 offset:16384 // (op 645) Load local data 


ds_read_b128 v[42:45], v5 offset:16448 // (op 645) Load local data 



v_add_u32 v16, 74112, v0 // (op 649) 
ds_read_u8 v17, v16 // (op 649) Load local data 




s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[38:45], v[30:37], a[12:15], v17, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 651) 
ds_read_b128 v[22:25], v5 offset:20480 // (op 807) Load local data 


ds_read_b128 v[26:29], v5 offset:20544 // (op 807) Load local data 



v_add_u32 v16, 74240, v0 // (op 811) 
ds_read_u8 v17, v16 // (op 811) Load local data 




s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[22:29], v[30:37], a[16:19], v17, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 813) 
ds_read_b128 v[38:41], v5 offset:24576 // (op 969) Load local data 


ds_read_b128 v[42:45], v5 offset:24640 // (op 969) Load local data 



v_add_u32 v16, 74368, v0 // (op 973) 
ds_read_u8 v17, v16 // (op 973) Load local data 




s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[38:45], v[30:37], a[20:23], v17, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 975) 
ds_read_b128 v[22:25], v5 offset:28672 // (op 1131) Load local data 


ds_read_b128 v[26:29], v5 offset:28736 // (op 1131) Load local data 



v_add_u32 v16, 74496, v0 // (op 1135) 
ds_read_u8 v17, v16 // (op 1135) Load local data 




s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[22:29], v[30:37], a[24:27], v17, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1137) 
ds_read_b128 v[38:41], v5 offset:32768 // (op 1293) Load local data 


ds_read_b128 v[42:45], v5 offset:32832 // (op 1293) Load local data 



v_add_u32 v16, 74624, v0 // (op 1297) 
ds_read_u8 v17, v16 // (op 1297) Load local data 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[38:45], v[30:37], a[28:31], v17, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1299) 




s_ashr_i64 s[6:7], s[36:37], 7 // (op 1287) 
s_mov_b32 s11, s6 // (op 1287) convert

s_cmp_lt_u32 s10, s11 // (op 1287) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ForLoopTop_KLoopTail_1287 // (op 1287) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ForLoopTop_KLoopTail_1287 if true)
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ForLoopBottom_KLoopTail_1287:






















s_barrier  // (op 4292) 







v_cmp_eq_f32 vcc, 0.00000, v10 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ConditionalFalse_0_beta_123
v_accvgpr_read v12, a0 // (op 46) 
v_accvgpr_read v13, a1 // (op 46) 
v_accvgpr_read v14, a2 // (op 46) 
v_accvgpr_read v15, a3 // (op 46) 
v_mul_f32 v16, v11, v12 // (op 46) 
v_mul_f32 v17, v11, v13 // (op 46) 
v_mul_f32 v18, v11, v14 // (op 46) 
v_mul_f32 v19, v11, v15 // (op 46) 




buffer_store_dwordx4 v[16:19], v9, s[44:47], 0 offen // (op 52) Store value

v_accvgpr_read v12, a4 // (op 1617) 
v_accvgpr_read v13, a5 // (op 1617) 
v_accvgpr_read v14, a6 // (op 1617) 
v_accvgpr_read v15, a7 // (op 1617) 
v_mul_f32 v16, v11, v12 // (op 1617) 
v_mul_f32 v17, v11, v13 // (op 1617) 
v_mul_f32 v18, v11, v14 // (op 1617) 
v_mul_f32 v19, v11, v15 // (op 1617) 




buffer_store_dwordx4 v[16:19], v9, s[44:47], 0 offen offset:128 // (op 1619) Store value

v_accvgpr_read v12, a8 // (op 1635) 
v_accvgpr_read v13, a9 // (op 1635) 
v_accvgpr_read v14, a10 // (op 1635) 
v_accvgpr_read v15, a11 // (op 1635) 
v_mul_f32 v16, v11, v12 // (op 1635) 
v_mul_f32 v17, v11, v13 // (op 1635) 
v_mul_f32 v18, v11, v14 // (op 1635) 
v_mul_f32 v19, v11, v15 // (op 1635) 




buffer_store_dwordx4 v[16:19], v9, s[44:47], 0 offen offset:256 // (op 1637) Store value

v_accvgpr_read v12, a12 // (op 1653) 
v_accvgpr_read v13, a13 // (op 1653) 
v_accvgpr_read v14, a14 // (op 1653) 
v_accvgpr_read v15, a15 // (op 1653) 
v_mul_f32 v16, v11, v12 // (op 1653) 
v_mul_f32 v17, v11, v13 // (op 1653) 
v_mul_f32 v18, v11, v14 // (op 1653) 
v_mul_f32 v19, v11, v15 // (op 1653) 




buffer_store_dwordx4 v[16:19], v9, s[44:47], 0 offen offset:384 // (op 1655) Store value

v_accvgpr_read v12, a16 // (op 1671) 
v_accvgpr_read v13, a17 // (op 1671) 
v_accvgpr_read v14, a18 // (op 1671) 
v_accvgpr_read v15, a19 // (op 1671) 
v_mul_f32 v16, v11, v12 // (op 1671) 
v_mul_f32 v17, v11, v13 // (op 1671) 
v_mul_f32 v18, v11, v14 // (op 1671) 
v_mul_f32 v19, v11, v15 // (op 1671) 




buffer_store_dwordx4 v[16:19], v9, s[44:47], 0 offen offset:512 // (op 1673) Store value

v_accvgpr_read v12, a20 // (op 1689) 
v_accvgpr_read v13, a21 // (op 1689) 
v_accvgpr_read v14, a22 // (op 1689) 
v_accvgpr_read v15, a23 // (op 1689) 
v_mul_f32 v16, v11, v12 // (op 1689) 
v_mul_f32 v17, v11, v13 // (op 1689) 
v_mul_f32 v18, v11, v14 // (op 1689) 
v_mul_f32 v19, v11, v15 // (op 1689) 




buffer_store_dwordx4 v[16:19], v9, s[44:47], 0 offen offset:640 // (op 1691) Store value

v_accvgpr_read v12, a24 // (op 1707) 
v_accvgpr_read v13, a25 // (op 1707) 
v_accvgpr_read v14, a26 // (op 1707) 
v_accvgpr_read v15, a27 // (op 1707) 
v_mul_f32 v16, v11, v12 // (op 1707) 
v_mul_f32 v17, v11, v13 // (op 1707) 
v_mul_f32 v18, v11, v14 // (op 1707) 
v_mul_f32 v19, v11, v15 // (op 1707) 




buffer_store_dwordx4 v[16:19], v9, s[44:47], 0 offen offset:768 // (op 1709) Store value

v_accvgpr_read v12, a28 // (op 1725) 
v_accvgpr_read v13, a29 // (op 1725) 
v_accvgpr_read v14, a30 // (op 1725) 
v_accvgpr_read v15, a31 // (op 1725) 
v_mul_f32 v16, v11, v12 // (op 1725) 
v_mul_f32 v17, v11, v13 // (op 1725) 
v_mul_f32 v18, v11, v14 // (op 1725) 
v_mul_f32 v19, v11, v15 // (op 1725) 




buffer_store_dwordx4 v[16:19], v9, s[44:47], 0 offen offset:896 // (op 1727) Store value

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ConditionalBottom_0_beta_123
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ConditionalFalse_0_beta_123:



buffer_load_dwordx4 v[12:15], v6, s[40:43], 0 offen offset:896 // (op 1893) Load value
v_accvgpr_read v16, a28 // (op 1890) 
v_accvgpr_read v17, a29 // (op 1890) 
v_accvgpr_read v18, a30 // (op 1890) 
v_accvgpr_read v19, a31 // (op 1890) 
v_mul_f32 v20, v11, v16 // (op 1890) 
v_mul_f32 v21, v11, v17 // (op 1890) 
v_mul_f32 v22, v11, v18 // (op 1890) 
v_mul_f32 v23, v11, v19 // (op 1890) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v10, v12 // (op 1894) 
v_mul_f32 v17, v10, v13 // (op 1894) 
v_mul_f32 v18, v10, v14 // (op 1894) 
v_mul_f32 v19, v10, v15 // (op 1894) 

v_add_f32 v12, v16, v20 // (op 1891) 
v_add_f32 v13, v17, v21 // (op 1891) 
v_add_f32 v14, v18, v22 // (op 1891) 
v_add_f32 v15, v19, v23 // (op 1891) 





buffer_store_dwordx4 v[12:15], v9, s[44:47], 0 offen offset:896 // (op 1892) Store value




s_nop 1
buffer_load_dwordx4 v[12:15], v6, s[40:43], 0 offen offset:768 // (op 1875) Load value
v_accvgpr_read v16, a24 // (op 1872) 
v_accvgpr_read v17, a25 // (op 1872) 
v_accvgpr_read v18, a26 // (op 1872) 
v_accvgpr_read v19, a27 // (op 1872) 
v_mul_f32 v20, v11, v16 // (op 1872) 
v_mul_f32 v21, v11, v17 // (op 1872) 
v_mul_f32 v22, v11, v18 // (op 1872) 
v_mul_f32 v23, v11, v19 // (op 1872) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v10, v12 // (op 1876) 
v_mul_f32 v17, v10, v13 // (op 1876) 
v_mul_f32 v18, v10, v14 // (op 1876) 
v_mul_f32 v19, v10, v15 // (op 1876) 

v_add_f32 v12, v16, v20 // (op 1873) 
v_add_f32 v13, v17, v21 // (op 1873) 
v_add_f32 v14, v18, v22 // (op 1873) 
v_add_f32 v15, v19, v23 // (op 1873) 





buffer_store_dwordx4 v[12:15], v9, s[44:47], 0 offen offset:768 // (op 1874) Store value




s_nop 1
buffer_load_dwordx4 v[12:15], v6, s[40:43], 0 offen offset:640 // (op 1857) Load value
v_accvgpr_read v16, a20 // (op 1854) 
v_accvgpr_read v17, a21 // (op 1854) 
v_accvgpr_read v18, a22 // (op 1854) 
v_accvgpr_read v19, a23 // (op 1854) 
v_mul_f32 v20, v11, v16 // (op 1854) 
v_mul_f32 v21, v11, v17 // (op 1854) 
v_mul_f32 v22, v11, v18 // (op 1854) 
v_mul_f32 v23, v11, v19 // (op 1854) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v10, v12 // (op 1858) 
v_mul_f32 v17, v10, v13 // (op 1858) 
v_mul_f32 v18, v10, v14 // (op 1858) 
v_mul_f32 v19, v10, v15 // (op 1858) 

v_add_f32 v12, v16, v20 // (op 1855) 
v_add_f32 v13, v17, v21 // (op 1855) 
v_add_f32 v14, v18, v22 // (op 1855) 
v_add_f32 v15, v19, v23 // (op 1855) 





buffer_store_dwordx4 v[12:15], v9, s[44:47], 0 offen offset:640 // (op 1856) Store value




s_nop 1
buffer_load_dwordx4 v[12:15], v6, s[40:43], 0 offen offset:512 // (op 1839) Load value
v_accvgpr_read v16, a16 // (op 1836) 
v_accvgpr_read v17, a17 // (op 1836) 
v_accvgpr_read v18, a18 // (op 1836) 
v_accvgpr_read v19, a19 // (op 1836) 
v_mul_f32 v20, v11, v16 // (op 1836) 
v_mul_f32 v21, v11, v17 // (op 1836) 
v_mul_f32 v22, v11, v18 // (op 1836) 
v_mul_f32 v23, v11, v19 // (op 1836) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v10, v12 // (op 1840) 
v_mul_f32 v17, v10, v13 // (op 1840) 
v_mul_f32 v18, v10, v14 // (op 1840) 
v_mul_f32 v19, v10, v15 // (op 1840) 

v_add_f32 v12, v16, v20 // (op 1837) 
v_add_f32 v13, v17, v21 // (op 1837) 
v_add_f32 v14, v18, v22 // (op 1837) 
v_add_f32 v15, v19, v23 // (op 1837) 





buffer_store_dwordx4 v[12:15], v9, s[44:47], 0 offen offset:512 // (op 1838) Store value




s_nop 1
buffer_load_dwordx4 v[12:15], v6, s[40:43], 0 offen offset:384 // (op 1821) Load value
v_accvgpr_read v16, a12 // (op 1818) 
v_accvgpr_read v17, a13 // (op 1818) 
v_accvgpr_read v18, a14 // (op 1818) 
v_accvgpr_read v19, a15 // (op 1818) 
v_mul_f32 v20, v11, v16 // (op 1818) 
v_mul_f32 v21, v11, v17 // (op 1818) 
v_mul_f32 v22, v11, v18 // (op 1818) 
v_mul_f32 v23, v11, v19 // (op 1818) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v10, v12 // (op 1822) 
v_mul_f32 v17, v10, v13 // (op 1822) 
v_mul_f32 v18, v10, v14 // (op 1822) 
v_mul_f32 v19, v10, v15 // (op 1822) 

v_add_f32 v12, v16, v20 // (op 1819) 
v_add_f32 v13, v17, v21 // (op 1819) 
v_add_f32 v14, v18, v22 // (op 1819) 
v_add_f32 v15, v19, v23 // (op 1819) 





buffer_store_dwordx4 v[12:15], v9, s[44:47], 0 offen offset:384 // (op 1820) Store value




s_nop 1
buffer_load_dwordx4 v[12:15], v6, s[40:43], 0 offen offset:256 // (op 1803) Load value
v_accvgpr_read v16, a8 // (op 1800) 
v_accvgpr_read v17, a9 // (op 1800) 
v_accvgpr_read v18, a10 // (op 1800) 
v_accvgpr_read v19, a11 // (op 1800) 
v_mul_f32 v20, v11, v16 // (op 1800) 
v_mul_f32 v21, v11, v17 // (op 1800) 
v_mul_f32 v22, v11, v18 // (op 1800) 
v_mul_f32 v23, v11, v19 // (op 1800) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v10, v12 // (op 1804) 
v_mul_f32 v17, v10, v13 // (op 1804) 
v_mul_f32 v18, v10, v14 // (op 1804) 
v_mul_f32 v19, v10, v15 // (op 1804) 

v_add_f32 v12, v16, v20 // (op 1801) 
v_add_f32 v13, v17, v21 // (op 1801) 
v_add_f32 v14, v18, v22 // (op 1801) 
v_add_f32 v15, v19, v23 // (op 1801) 





buffer_store_dwordx4 v[12:15], v9, s[44:47], 0 offen offset:256 // (op 1802) Store value




s_nop 1
buffer_load_dwordx4 v[12:15], v6, s[40:43], 0 offen offset:128 // (op 1785) Load value
v_accvgpr_read v16, a4 // (op 1782) 
v_accvgpr_read v17, a5 // (op 1782) 
v_accvgpr_read v18, a6 // (op 1782) 
v_accvgpr_read v19, a7 // (op 1782) 
v_mul_f32 v20, v11, v16 // (op 1782) 
v_mul_f32 v21, v11, v17 // (op 1782) 
v_mul_f32 v22, v11, v18 // (op 1782) 
v_mul_f32 v23, v11, v19 // (op 1782) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v10, v12 // (op 1786) 
v_mul_f32 v17, v10, v13 // (op 1786) 
v_mul_f32 v18, v10, v14 // (op 1786) 
v_mul_f32 v19, v10, v15 // (op 1786) 

v_add_f32 v12, v16, v20 // (op 1783) 
v_add_f32 v13, v17, v21 // (op 1783) 
v_add_f32 v14, v18, v22 // (op 1783) 
v_add_f32 v15, v19, v23 // (op 1783) 





buffer_store_dwordx4 v[12:15], v9, s[44:47], 0 offen offset:128 // (op 1784) Store value




s_nop 1
buffer_load_dwordx4 v[12:15], v6, s[40:43], 0 offen // (op 135) Load value
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





buffer_store_dwordx4 v[12:15], v9, s[44:47], 0 offen // (op 134) Store value

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ConditionalBottom_0_beta_123:
















s_endpgm  // End of RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2
.LRR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_end:
.size RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2, .LRR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_end-RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2
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
    macrotile: [256, 32, 128]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [8, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2
    .symbol: RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2.kd
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
    .workitem_count: [{type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, {is-null: true}]
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

