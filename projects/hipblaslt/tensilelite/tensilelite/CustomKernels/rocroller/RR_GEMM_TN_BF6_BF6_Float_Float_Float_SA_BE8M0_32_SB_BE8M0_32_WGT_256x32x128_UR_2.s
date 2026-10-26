

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2
.p2align 8
.type RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2,@function
RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 6, v1 // (op 3915) 
v_and_b32 v3, 1, v0 // (op 3915) 

v_and_b32 v0, 15, v1 // (op 3915) 
v_lshl_add_u32 v4, v3, 4, v0 // (op 3915) 


v_and_b32 v0, 63, v1 // (op 3915) 
v_lshrrev_b32 v3, 4, v0 // (op 3915) 

v_lshlrev_b32 v0, 5, v3 // (op 3915) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 3915) 


v_mul_lo_u32 v0, v3, 6 // (op 3915) 

v_lshrrev_b32 v3, 3, v0 // (op 3915) 

v_lshrrev_b32 v0, 6, v1 // (op 4044) 
v_and_b32 v4, 1, v0 // (op 4044) 

v_and_b32 v0, 15, v1 // (op 4044) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 4044) 


v_and_b32 v0, 63, v1 // (op 4044) 
v_lshrrev_b32 v4, 4, v0 // (op 4044) 

v_lshlrev_b32 v0, 5, v4 // (op 4044) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 4044) 


v_mul_lo_u32 v0, v4, 6 // (op 4044) 

v_lshrrev_b32 v4, 3, v0 // (op 4044) 

v_lshrrev_b32 v0, 7, v1 // (op 3800) 
v_and_b32 v5, 63, v1 // (op 3800) 
v_lshrrev_b32 v6, 4, v5 // (op 3800) 
v_lshlrev_b32 v7, 2, v6 // (op 3800) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 3800) 


v_lshl_add_u32 v0, s16, 8, v6 // (op 3800) 

v_lshrrev_b32 v6, 2, v5 // (op 3800) 

v_and_b32 v5, 3, v6 // (op 3800) 

v_and_b32 v6, 3, v1 // (op 3800) 
v_lshl_add_u32 v7, v5, 2, v6 // (op 3800) 


v_lshrrev_b32 v5, 6, v1 // (op 3800) 
v_and_b32 v6, 1, v5 // (op 3800) 

v_lshl_add_u32 v5, v6, 4, v7 // (op 3800) 


v_lshl_add_u32 v6, s17, 5, v5 // (op 3800) 

v_mul_lo_u32 v5, v6, s2 // (op 3800) 

v_add_lshl_u32 v6, v0, v5, 2 // (op 3800) 


v_lshrrev_b32 v0, 7, v1 // (op 3862) 
v_and_b32 v5, 15, v1 // (op 3862) 
v_lshl_add_u32 v7, v0, 4, v5 // (op 3862) 


v_and_b32 v0, 63, v1 // (op 3862) 
v_lshrrev_b32 v5, 4, v0 // (op 3862) 

v_lshlrev_b32 v0, 5, v5 // (op 3862) 

v_lshl_add_u32 v5, v7, 7, v0 // (op 3862) 


v_mul_lo_u32 v0, v5, 6 // (op 3862) 

v_lshrrev_b32 v5, 3, v0 // (op 3862) 

v_lshrrev_b32 v0, 7, v1 // (op 3973) 
v_and_b32 v7, 15, v1 // (op 3973) 
s_mul_i32 s1, s2, s9 // (op 3802) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 3802) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 3802) least significant
s_mul_hi_u32 s19, s2, s8 // (op 3802) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3802) most significant: sum
s_add_u32 s1, s1, s19 // (op 3802) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 3802) 

v_lshl_add_u32 v8, v0, 4, v7 // (op 3973) 


v_and_b32 v0, 63, v1 // (op 3973) 
v_lshrrev_b32 v7, 4, v0 // (op 3973) 

v_lshl_add_u32 v0, v8, 2, v7 // (op 3973) 


v_lshrrev_b32 v7, 7, v1 // (op 3989) 
v_and_b32 v8, 63, v1 // (op 3989) 
v_lshrrev_b32 v9, 4, v8 // (op 3989) 
v_lshlrev_b32 v10, 2, v9 // (op 3989) 

v_lshl_add_u32 v9, v7, 4, v10 // (op 3989) 


v_lshl_add_u32 v7, s16, 8, v9 // (op 3989) 

v_lshrrev_b32 v9, 2, v8 // (op 3989) 

v_and_b32 v8, 3, v9 // (op 3989) 

v_and_b32 v9, 3, v1 // (op 3989) 
v_lshl_add_u32 v10, v8, 2, v9 // (op 3989) 


v_lshrrev_b32 v8, 6, v1 // (op 3989) 
v_and_b32 v9, 1, v8 // (op 3989) 

v_lshl_add_u32 v8, v9, 4, v10 // (op 3989) 


v_lshl_add_u32 v9, s17, 5, v8 // (op 3989) 

v_mul_lo_u32 v8, v9, s4 // (op 3989) 

v_add_lshl_u32 v9, v7, v8, 2 // (op 3989) 


v_lshrrev_b32 v7, 7, v1 // (op 4020) 
v_and_b32 v8, 15, v1 // (op 4020) 
v_lshl_add_u32 v10, v7, 4, v8 // (op 4020) 


v_and_b32 v7, 63, v1 // (op 4020) 
v_lshrrev_b32 v8, 4, v7 // (op 4020) 

v_lshlrev_b32 v7, 5, v8 // (op 4020) 

v_lshl_add_u32 v8, v10, 7, v7 // (op 4020) 


v_mul_lo_u32 v7, v8, 6 // (op 4020) 

v_lshrrev_b32 v8, 3, v7 // (op 4020) 

v_lshrrev_b32 v7, 7, v1 // (op 4073) 
v_and_b32 v10, 15, v1 // (op 4073) 
s_mul_i32 s1, s4, s9 // (op 3991) most significant: low of low * high
v_lshl_add_u32 v11, v7, 4, v10 // (op 4073) 


v_and_b32 v7, 63, v1 // (op 4073) 
v_lshrrev_b32 v10, 4, v7 // (op 4073) 

v_lshl_add_u32 v7, v11, 2, v10 // (op 4073) 


s_mul_i32 s38, s5, s8 // (op 3991) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 3991) least significant
s_mul_hi_u32 s39, s4, s8 // (op 3991) most significant: high of low * low
s_add_u32 s1, s1, s38 // (op 3991) most significant: sum
s_add_u32 s1, s1, s39 // (op 3991) most significant: sum


s_lshl_b64 s[38:39], s[0:1], 2 // (op 3991) 

s_mov_b32 s40, s6 // (op 3802) reg expression
s_mov_b32 s41, s7 // (op 3802) reg expression
s_mov_b32 s42, s18 // (op 3802) reg expression
s_mov_b32 s43, 131072 // (op 3802) call()

s_mov_b32 s44, s10 // (op 3991) reg expression

s_mov_b32 s45, s11 // (op 3991) reg expression
s_mov_b32 s46, s38 // (op 3991) reg expression
s_mov_b32 s47, 131072 // (op 3991) call()


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
v_and_b32 v12, 4294967294, v1 // (op 3567) 
v_lshl_add_u32 v13, s16, 8, v12 // (op 3567) 

v_and_b32 v12, -2, v1 // (op 3567) 
v_sub_u32 v14, v1, v12 // (op 3567) 

v_lshlrev_b32 v12, 6, v14 // (op 3567) 

v_mov_b32 v15, 0 // (op 3567) convert
v_mov_b32 v14, v12 // (op 3567) convert

v_mul_lo_u32 v17, v13, s13 // (op 3567) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 3567) most significant: high of low * low
v_mul_lo_u32 v16, v13, s12 // (op 3567) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 3567) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 3567) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 3567) most significant half



v_mul_lo_u32 v13, v17, 6 // (op 3567) most significant: low of high * low
v_mul_hi_u32 v14, v16, 6 // (op 3567) most significant: high of low * low
v_mul_lo_u32 v12, v16, 6 // (op 3567) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 3567) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 3567) 

v_mov_b32 v12, v14 // (op 3567) convert


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s23, 6 // (op 3569) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 3569) least significant
s_mul_hi_u32 s6, s22, 6 // (op 3569) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3569) most significant: sum

s_mul_i32 s7, s12, s1 // (op 3569) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 3569) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 3569) least significant
s_mul_hi_u32 s11, s12, s0 // (op 3569) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 3569) most significant: sum
s_add_u32 s7, s7, s11 // (op 3569) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 3569) 

s_mov_b32 s48, s20 // (op 3569) reg expression
s_mov_b32 s49, s21 // (op 3569) reg expression
s_mov_b32 s50, s0 // (op 3569) reg expression
s_mov_b32 s51, 131072 // (op 3569) call()

v_and_b32 v13, 4294967294, v1 // (op 3586) 
v_lshl_add_u32 v14, s16, 8, v13 // (op 3586) 

v_and_b32 v13, -2, v1 // (op 3586) 
v_sub_u32 v15, v1, v13 // (op 3586) 

v_lshlrev_b32 v13, 6, v15 // (op 3586) 

v_mov_b32 v17, 0 // (op 3586) convert
v_mov_b32 v16, v13 // (op 3586) convert

v_mul_lo_u32 v19, v14, s13 // (op 3586) most significant: low of low * high
v_mul_hi_u32 v13, v14, s12 // (op 3586) most significant: high of low * low
v_mul_lo_u32 v18, v14, s12 // (op 3586) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 3586) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 3586) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 3586) most significant half



v_mul_lo_u32 v17, v19, 6 // (op 3586) most significant: low of high * low
v_mul_hi_u32 v13, v18, 6 // (op 3586) most significant: high of low * low
v_mul_lo_u32 v16, v18, 6 // (op 3586) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 3586) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 3586) 

v_mov_b32 v13, v14 // (op 3586) convert

s_mul_i32 s1, s23, 6 // (op 3588) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 3588) least significant
s_mul_hi_u32 s6, s22, 6 // (op 3588) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3588) most significant: sum

s_mul_i32 s7, s12, s1 // (op 3588) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 3588) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 3588) least significant
s_mul_hi_u32 s11, s12, s0 // (op 3588) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 3588) most significant: sum
s_add_u32 s7, s7, s11 // (op 3588) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 3588) 

s_mov_b32 s48, s20 // (op 3588) reg expression
s_mov_b32 s49, s21 // (op 3588) reg expression
s_mov_b32 s50, s0 // (op 3588) reg expression
s_mov_b32 s51, 131072 // (op 3588) call()

v_and_b32 v14, -8, v1 // (op 3631) 
v_sub_u32 v15, v1, v14 // (op 3631) 

v_lshlrev_b32 v14, 4, v15 // (op 3631) 

v_mov_b32 v17, 0 // (op 3631) convert
v_mov_b32 v16, v14 // (op 3631) convert

v_lshrrev_b32 v14, 3, v1 // (op 3631) 
v_lshl_add_u32 v15, s17, 5, v14 // (op 3631) 

v_mul_lo_u32 v19, v15, s25 // (op 3631) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 3631) most significant: high of low * low
v_mul_lo_u32 v18, v15, s24 // (op 3631) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 3631) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v18 // (op 3631) least significant half
v_addc_co_u32 v15, s[0:1], v17, v19, s[0:1] // (op 3631) most significant half



v_mul_lo_u32 v17, v15, 6 // (op 3631) most significant: low of high * low
v_mul_hi_u32 v18, v14, 6 // (op 3631) most significant: high of low * low
v_mul_lo_u32 v16, v14, 6 // (op 3631) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 3631) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 3631) 

v_mov_b32 v16, v14 // (op 3631) convert

s_mul_i32 s1, s9, 6 // (op 3633) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 3633) least significant
s_mul_hi_u32 s6, s8, 6 // (op 3633) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3633) most significant: sum

s_mul_i32 s7, s24, s1 // (op 3633) most significant: low of low * high
s_mul_i32 s10, s25, s0 // (op 3633) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 3633) least significant
s_mul_hi_u32 s11, s24, s0 // (op 3633) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 3633) most significant: sum
s_add_u32 s7, s7, s11 // (op 3633) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 3633) 

s_mov_b32 s52, s26 // (op 3633) reg expression
s_mov_b32 s53, s27 // (op 3633) reg expression
s_mov_b32 s54, s0 // (op 3633) reg expression
s_mov_b32 s55, 131072 // (op 3633) call()

v_and_b32 v14, -8, v1 // (op 3650) 
v_sub_u32 v15, v1, v14 // (op 3650) 

v_lshlrev_b32 v14, 4, v15 // (op 3650) 

v_mov_b32 v19, 0 // (op 3650) convert
v_mov_b32 v18, v14 // (op 3650) convert

v_lshrrev_b32 v17, 3, v1 // (op 3650) 
v_lshl_add_u32 v14, s17, 5, v17 // (op 3650) 

v_mul_lo_u32 v21, v14, s25 // (op 3650) most significant: low of low * high
v_mul_hi_u32 v15, v14, s24 // (op 3650) most significant: high of low * low
v_mul_lo_u32 v20, v14, s24 // (op 3650) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 3650) most significant: sum


v_add_co_u32 v14, s[0:1], v18, v20 // (op 3650) least significant half
v_addc_co_u32 v15, s[0:1], v19, v21, s[0:1] // (op 3650) most significant half



v_mul_lo_u32 v19, v15, 6 // (op 3650) most significant: low of high * low
v_mul_hi_u32 v17, v14, 6 // (op 3650) most significant: high of low * low
v_mul_lo_u32 v18, v14, 6 // (op 3650) least significant: low of low * low
v_add_u32 v19, v19, v17 // (op 3650) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[18:19] // (op 3650) 

v_mov_b32 v17, v14 // (op 3650) convert

s_mul_i32 s1, s9, 6 // (op 3652) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 3652) least significant
s_mul_hi_u32 s6, s8, 6 // (op 3652) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3652) most significant: sum

s_mul_i32 s7, s24, s1 // (op 3652) most significant: low of low * high
s_mul_i32 s10, s25, s0 // (op 3652) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 3652) least significant
s_mul_hi_u32 s11, s24, s0 // (op 3652) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 3652) most significant: sum
s_add_u32 s7, s7, s11 // (op 3652) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 3652) 

s_mov_b32 s52, s26 // (op 3652) reg expression
s_mov_b32 s53, s27 // (op 3652) reg expression
s_mov_b32 s54, s0 // (op 3652) reg expression
s_mov_b32 s55, 131072 // (op 3652) call()

v_lshl_add_u32 v14, s16, 8, v1 // (op 3695) 
v_mul_lo_u32 v15, v14, s28 // (op 3695) 

s_mul_i32 s1, s28, s23 // (op 3697) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 3697) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 3697) least significant
s_mul_hi_u32 s7, s28, s22 // (op 3697) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3697) most significant: sum
s_add_u32 s1, s1, s7 // (op 3697) most significant: sum


s_mov_b32 s56, s30 // (op 3697) reg expression
s_mov_b32 s57, s31 // (op 3697) reg expression
s_mov_b32 s58, s0 // (op 3697) reg expression
s_mov_b32 s59, 131072 // (op 3697) call()

v_lshl_add_u32 v14, s16, 8, v1 // (op 3714) 
v_mul_lo_u32 v18, v14, s28 // (op 3714) 

s_mul_i32 s1, s28, s23 // (op 3716) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 3716) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 3716) least significant
s_mul_hi_u32 s7, s28, s22 // (op 3716) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3716) most significant: sum
s_add_u32 s1, s1, s7 // (op 3716) most significant: sum


s_mov_b32 s56, s30 // (op 3716) reg expression
s_mov_b32 s57, s31 // (op 3716) reg expression
s_mov_b32 s58, s0 // (op 3716) reg expression
s_mov_b32 s59, 131072 // (op 3716) call()

v_lshrrev_b32 v14, 6, v1 // (op 3759) 
v_and_b32 v19, 1, v14 // (op 3759) 

v_and_b32 v14, 15, v1 // (op 3759) 
v_lshl_add_u32 v20, v19, 4, v14 // (op 3759) 


v_lshl_add_u32 v14, s17, 5, v20 // (op 3759) 

v_mul_lo_u32 v19, v14, s32 // (op 3759) 

v_and_b32 v14, 63, v1 // (op 3759) 
v_lshrrev_b32 v20, 4, v14 // (op 3759) 

v_add_u32 v14, v20, v19 // (op 3759) 


s_mul_i32 s1, s32, s9 // (op 3761) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 3761) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 3761) least significant
s_mul_hi_u32 s7, s32, s8 // (op 3761) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3761) most significant: sum
s_add_u32 s1, s1, s7 // (op 3761) most significant: sum


s_mov_b32 s60, s34 // (op 3761) reg expression
s_mov_b32 s61, s35 // (op 3761) reg expression
s_mov_b32 s62, s0 // (op 3761) reg expression
s_mov_b32 s63, 131072 // (op 3761) call()

v_lshrrev_b32 v19, 6, v1 // (op 3778) 
v_and_b32 v20, 1, v19 // (op 3778) 

v_and_b32 v19, 15, v1 // (op 3778) 
v_lshl_add_u32 v21, v20, 4, v19 // (op 3778) 


v_lshl_add_u32 v19, s17, 5, v21 // (op 3778) 

v_mul_lo_u32 v20, v19, s32 // (op 3778) 

v_and_b32 v19, 63, v1 // (op 3778) 
v_lshrrev_b32 v21, 4, v19 // (op 3778) 

v_add_u32 v19, v21, v20 // (op 3778) 


s_mul_i32 s1, s32, s9 // (op 3780) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 3780) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 3780) least significant
s_mul_hi_u32 s7, s32, s8 // (op 3780) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3780) most significant: sum
s_add_u32 s1, s1, s7 // (op 3780) most significant: sum


s_mov_b32 s60, s34 // (op 3780) reg expression
s_mov_b32 s61, s35 // (op 3780) reg expression
s_mov_b32 s62, s0 // (op 3780) reg expression
s_mov_b32 s63, 131072 // (op 3780) call()




buffer_load_ubyte v20, v14, s[60:63], 0 offen offset: 0 // (op 3252) Load value



buffer_load_dwordx3 v[22:24], v16, s[52:55], 0 offen // (op 3256) Load value

s_mul_i32 s1, s13, 6 // (op 3259) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 3259) least significant
s_mul_hi_u32 s6, s12, 6 // (op 3259) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3259) most significant: sum

v_lshrrev_b64 v[26:27], 3, s[0:1] // (op 3259) 


v_mov_b32 v21, v12 // (op 3259) 
buffer_load_dwordx4 v[28:31], v21, s[48:51], 0 offen // (op 3259) Load value
buffer_load_dwordx2 v[32:33], v21, s[48:51], 0 offen offset:16 // (op 3259) Load value
buffer_load_dwordx4 v[34:37], v21, s[48:51], 0 offen offset:24 // (op 3259) Load value
buffer_load_dwordx2 v[38:39], v21, s[48:51], 0 offen offset:40 // (op 3259) Load value
v_add_u32 v21, v21, v26 // (op 3259) 
buffer_load_dwordx4 v[40:43], v21, s[48:51], 0 offen // (op 3259) Load value
buffer_load_dwordx2 v[44:45], v21, s[48:51], 0 offen offset:16 // (op 3259) Load value
buffer_load_dwordx4 v[46:49], v21, s[48:51], 0 offen offset:24 // (op 3259) Load value
buffer_load_dwordx2 v[50:51], v21, s[48:51], 0 offen offset:40 // (op 3259) Load value





buffer_load_dword v21, v15, s[56:59], 0 offen // (op 3262) Load value



buffer_load_ubyte v25, v14, s[60:63], 0 offen offset:4 // (op 3266) Load value




buffer_load_dwordx3 v[52:54], v16, s[52:55], 0 offen offset:96 // (op 3270) Load value


s_mul_i32 s1, s13, 6 // (op 3273) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 3273) least significant
s_mul_hi_u32 s6, s12, 6 // (op 3273) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3273) most significant: sum

v_lshrrev_b64 v[26:27], 3, s[0:1] // (op 3273) 


v_mov_b32 v14, v12 // (op 3273) 
buffer_load_dwordx4 v[56:59], v14, s[48:51], 0 offen offset:96 // (op 3273) Load value
buffer_load_dwordx2 v[60:61], v14, s[48:51], 0 offen offset:112 // (op 3273) Load value
buffer_load_dwordx4 v[62:65], v14, s[48:51], 0 offen offset:120 // (op 3273) Load value
buffer_load_dwordx2 v[66:67], v14, s[48:51], 0 offen offset:136 // (op 3273) Load value
v_add_u32 v14, v14, v26 // (op 3273) 
buffer_load_dwordx4 v[68:71], v14, s[48:51], 0 offen offset:96 // (op 3273) Load value
buffer_load_dwordx2 v[72:73], v14, s[48:51], 0 offen offset:112 // (op 3273) Load value
buffer_load_dwordx4 v[74:77], v14, s[48:51], 0 offen offset:120 // (op 3273) Load value
buffer_load_dwordx2 v[78:79], v14, s[48:51], 0 offen offset:136 // (op 3273) Load value






buffer_load_dword v12, v15, s[56:59], 0 offen offset:4 // (op 3276) Load value

v_and_b32 v55, -8, v1 // (op 3903) 
v_sub_u32 v14, v1, v55 // (op 3903) 

v_lshlrev_b32 v55, 4, v14 // (op 3903) 

v_lshrrev_b32 v14, 3, v1 // (op 3903) 
v_lshl_add_u32 v15, v14, 7, v55 // (op 3903) 


v_mul_lo_u32 v14, v15, 6 // (op 3903) 

v_lshrrev_b32 v55, 3, v14 // (op 3903) 




s_waitcnt vmcnt(20)
ds_write_b96 v55, v[22:24] // (op 3278) Store local data 

v_and_b32 v55, -2, v1 // (op 3845) 
v_sub_u32 v14, v1, v55 // (op 3845) 

v_lshlrev_b32 v55, 6, v14 // (op 3845) 

v_lshrrev_b32 v14, 1, v1 // (op 3845) 
v_lshl_add_u32 v15, v14, 8, v55 // (op 3845) 


v_mul_lo_u32 v14, v15, 6 // (op 3845) 

v_lshrrev_b32 v55, 3, v14 // (op 3845) 




s_waitcnt vmcnt(19)
ds_write_b128 v55, v[28:31] offset:3072 // (op 3279) Store local data 
s_waitcnt vmcnt(18)
ds_write_b64 v55, v[32:33] offset:3088 // (op 3279) Store local data 
s_waitcnt vmcnt(17)
ds_write_b128 v55, v[34:37] offset:3096 // (op 3279) Store local data 
s_waitcnt vmcnt(16)
ds_write_b64 v55, v[38:39] offset:3112 // (op 3279) Store local data 
s_waitcnt vmcnt(15)
ds_write_b128 v55, v[40:43] offset:3168 // (op 3279) Store local data 
s_waitcnt vmcnt(14)
ds_write_b64 v55, v[44:45] offset:3184 // (op 3279) Store local data 
s_waitcnt vmcnt(13)
ds_write_b128 v55, v[46:49] offset:3192 // (op 3279) Store local data 
s_waitcnt vmcnt(12)
ds_write_b64 v55, v[50:51] offset:3208 // (op 3279) Store local data 

v_lshlrev_b32 v55, 2, v1 // (op 3956) 



s_waitcnt vmcnt(11)
ds_write_b32 v55, v21 offset:27648 // (op 3280) Store local data 

v_and_b32 v55, -2, v1 // (op 3816) 
v_sub_u32 v14, v1, v55 // (op 3816) 

v_lshlrev_b32 v55, 6, v14 // (op 3816) 

v_lshrrev_b32 v14, 1, v1 // (op 3816) 
v_lshl_add_u32 v15, v14, 8, v55 // (op 3816) 


v_mul_lo_u32 v14, v15, 6 // (op 3816) 

v_lshrrev_b32 v55, 3, v14 // (op 3816) 

v_and_b32 v14, -8, v1 // (op 3874) 
v_sub_u32 v15, v1, v14 // (op 3874) 

v_lshlrev_b32 v14, 4, v15 // (op 3874) 

v_lshrrev_b32 v15, 3, v1 // (op 3874) 
v_lshl_add_u32 v16, v15, 7, v14 // (op 3874) 


v_mul_lo_u32 v14, v16, 6 // (op 3874) 

v_lshrrev_b32 v15, 3, v14 // (op 3874) 

v_lshlrev_b32 v14, 2, v1 // (op 3927) 
v_and_b32 v16, -2, v1 // (op 4003) 
v_sub_u32 v26, v1, v16 // (op 4003) 

v_lshlrev_b32 v16, 6, v26 // (op 4003) 

v_lshrrev_b32 v26, 1, v1 // (op 4003) 
v_lshl_add_u32 v27, v26, 8, v16 // (op 4003) 


v_mul_lo_u32 v16, v27, 6 // (op 4003) 

v_lshrrev_b32 v26, 3, v16 // (op 4003) 

v_and_b32 v16, -8, v1 // (op 4032) 
v_sub_u32 v27, v1, v16 // (op 4032) 

v_lshlrev_b32 v16, 4, v27 // (op 4032) 

v_lshrrev_b32 v27, 3, v1 // (op 4032) 
v_lshl_add_u32 v80, v27, 7, v16 // (op 4032) 


v_mul_lo_u32 v16, v80, 6 // (op 4032) 

v_lshrrev_b32 v27, 3, v16 // (op 4032) 

v_lshlrev_b32 v16, 2, v1 // (op 4056) 
s_mov_b32 s0, 0 // (op 1283) call()
s_ashr_i64 s[6:7], s[36:37], 7 // (op 1282) 
s_mov_b32 s1, s6 // (op 1282) convert

s_and_b32 s6, s1, 4294967294 // (op 1282) 

s_cmp_lt_u32 s0, s6 // (op 1282) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ForLoopBottom_KLoop_1282 // (op 1282) Condition: Top (jump to RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ForLoopBottom_KLoop_1282 if false)
RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ForLoopTop_KLoop_1282:
s_add_u32 s0, s0, 2 // (op 1284) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 4132) 



ds_read_b128 v[80:83], v5 offset:3072 // (op 65) Load local data 
ds_read_b64 v[84:85], v5 offset:3088 // (op 65) Load local data 
s_barrier  // (op 4145) 



ds_read_b128 v[86:89], v3 // (op 70) Load local data 
ds_read_b64 v[90:91], v3 offset:16 // (op 70) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 4159) 



ds_read_u8 v92, v0 offset:27648 // (op 75) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[80:85], v[86:91], a[0:3], v92, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 82) 
ds_read_b128 v[94:97], v5 offset:6144 // (op 340) Load local data 


ds_read_b64 v[98:99], v5 offset:6160 // (op 340) Load local data 



ds_read_u8 v80, v0 offset:27776 // (op 344) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[94:99], v[86:91], a[4:7], v80, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 346) 
ds_read_b128 v[100:103], v5 offset:9216 // (op 502) Load local data 


ds_read_b64 v[104:105], v5 offset:9232 // (op 502) Load local data 



ds_read_u8 v80, v0 offset:27904 // (op 506) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[100:105], v[86:91], a[8:11], v80, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 508) 
ds_read_b128 v[92:95], v5 offset:12288 // (op 664) Load local data 


ds_read_b64 v[96:97], v5 offset:12304 // (op 664) Load local data 



ds_read_u8 v80, v0 offset:28032 // (op 668) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[92:97], v[86:91], a[12:15], v80, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 670) 
ds_read_b128 v[98:101], v5 offset:15360 // (op 826) Load local data 


ds_read_b64 v[102:103], v5 offset:15376 // (op 826) Load local data 



ds_read_u8 v80, v0 offset:28160 // (op 830) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[98:103], v[86:91], a[16:19], v80, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 832) 
ds_read_b128 v[92:95], v5 offset:18432 // (op 988) Load local data 


ds_read_b64 v[96:97], v5 offset:18448 // (op 988) Load local data 



ds_read_u8 v80, v0 offset:28288 // (op 992) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[92:97], v[86:91], a[20:23], v80, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 994) 
ds_read_b128 v[98:101], v5 offset:21504 // (op 1150) Load local data 


ds_read_b64 v[102:103], v5 offset:21520 // (op 1150) Load local data 



ds_read_u8 v80, v0 offset:28416 // (op 1154) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[98:103], v[86:91], a[24:27], v80, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 1156) 
ds_read_b128 v[92:95], v5 offset:24576 // (op 1312) Load local data 


ds_read_b64 v[96:97], v5 offset:24592 // (op 1312) Load local data 



ds_read_u8 v80, v0 offset:28544 // (op 1316) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[92:97], v[86:91], a[28:31], v80, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 1318) 





v_mov_b32 v80, v19 // (op 80) 

buffer_load_ubyte v20, v80, s[60:63], 0 offen offset:8 // (op 80) Load value


v_mov_b32 v81, v17 // (op 10) 

buffer_load_dwordx3 v[22:24], v81, s[52:55], 0 offen offset:192 // (op 10) Load value

s_mul_i32 s7, s13, 6 // (op 4) most significant: low of high * low
s_mul_i32 s6, s12, 6 // (op 4) least significant
s_mul_hi_u32 s1, s12, 6 // (op 4) most significant: high of low * low
s_add_u32 s7, s7, s1 // (op 4) most significant: sum

v_lshrrev_b64 v[82:83], 3, s[6:7] // (op 4) 


v_mov_b32 v84, v13 // (op 4) 
v_mov_b32 v85, v84 // (op 4) 
buffer_load_dwordx4 v[28:31], v85, s[48:51], 0 offen offset:192 // (op 4) Load value
buffer_load_dwordx2 v[32:33], v85, s[48:51], 0 offen offset:208 // (op 4) Load value
buffer_load_dwordx4 v[34:37], v85, s[48:51], 0 offen offset:216 // (op 4) Load value
buffer_load_dwordx2 v[38:39], v85, s[48:51], 0 offen offset:232 // (op 4) Load value
v_add_u32 v85, v85, v82 // (op 4) 
buffer_load_dwordx4 v[40:43], v85, s[48:51], 0 offen offset:192 // (op 4) Load value
buffer_load_dwordx2 v[44:45], v85, s[48:51], 0 offen offset:208 // (op 4) Load value
buffer_load_dwordx4 v[46:49], v85, s[48:51], 0 offen offset:216 // (op 4) Load value
buffer_load_dwordx2 v[50:51], v85, s[48:51], 0 offen offset:232 // (op 4) Load value




v_mov_b32 v82, v18 // (op 17) 

buffer_load_dword v21, v82, s[56:59], 0 offen offset:8 // (op 17) Load value
s_barrier  // (op 4182) 



s_waitcnt vmcnt(20)
ds_write_b96 v27, v[52:54] offset:28672 // (op 235) Store local data 
s_barrier  // (op 4176) 



s_waitcnt vmcnt(19)
ds_write_b128 v26, v[56:59] offset:31744 // (op 225) Store local data 
s_waitcnt vmcnt(18)
ds_write_b64 v26, v[60:61] offset:31760 // (op 225) Store local data 
s_waitcnt vmcnt(17)
ds_write_b128 v26, v[62:65] offset:31768 // (op 225) Store local data 
s_waitcnt vmcnt(16)
ds_write_b64 v26, v[66:67] offset:31784 // (op 225) Store local data 
s_waitcnt vmcnt(15)
ds_write_b128 v26, v[68:71] offset:31840 // (op 225) Store local data 
s_waitcnt vmcnt(14)
ds_write_b64 v26, v[72:73] offset:31856 // (op 225) Store local data 
s_waitcnt vmcnt(13)
ds_write_b128 v26, v[74:77] offset:31864 // (op 225) Store local data 
s_waitcnt vmcnt(12)
ds_write_b64 v26, v[78:79] offset:31880 // (op 225) Store local data 
s_barrier  // (op 4188) 



s_waitcnt vmcnt(11)
ds_write_b32 v16, v12 offset:56320 // (op 236) Store local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 4173) 



ds_read_b128 v[86:89], v8 offset:31744 // (op 227) Load local data 
ds_read_b64 v[90:91], v8 offset:31760 // (op 227) Load local data 
s_barrier  // (op 4179) 



ds_read_b128 v[92:95], v4 offset:28672 // (op 229) Load local data 
ds_read_b64 v[96:97], v4 offset:28688 // (op 229) Load local data 
s_barrier  // (op 4225) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 4185) 



ds_read_u8 v83, v7 offset:56320 // (op 231) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[86:91], v[92:97], a[0:3], v83, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 234) 
ds_read_b128 v[98:101], v8 offset:34816 // (op 353) Load local data 


ds_read_b64 v[102:103], v8 offset:34832 // (op 353) Load local data 



ds_read_u8 v83, v7 offset:56448 // (op 357) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[98:103], v[92:97], a[4:7], v83, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 347) 
ds_read_b128 v[86:89], v8 offset:37888 // (op 515) Load local data 


ds_read_b64 v[90:91], v8 offset:37904 // (op 515) Load local data 



ds_read_u8 v83, v7 offset:56576 // (op 519) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[86:91], v[92:97], a[8:11], v83, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 509) 
ds_read_b128 v[98:101], v8 offset:40960 // (op 677) Load local data 


ds_read_b64 v[102:103], v8 offset:40976 // (op 677) Load local data 



ds_read_u8 v83, v7 offset:56704 // (op 681) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[98:103], v[92:97], a[12:15], v83, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 671) 
ds_read_b128 v[86:89], v8 offset:44032 // (op 839) Load local data 


ds_read_b64 v[90:91], v8 offset:44048 // (op 839) Load local data 



ds_read_u8 v83, v7 offset:56832 // (op 843) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[86:91], v[92:97], a[16:19], v83, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 833) 
ds_read_b128 v[98:101], v8 offset:47104 // (op 1001) Load local data 


ds_read_b64 v[102:103], v8 offset:47120 // (op 1001) Load local data 



ds_read_u8 v83, v7 offset:56960 // (op 1005) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[98:103], v[92:97], a[20:23], v83, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 995) 
ds_read_b128 v[86:89], v8 offset:50176 // (op 1163) Load local data 


ds_read_b64 v[90:91], v8 offset:50192 // (op 1163) Load local data 



ds_read_u8 v83, v7 offset:57088 // (op 1167) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[86:91], v[92:97], a[24:27], v83, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 1157) 
ds_read_b128 v[98:101], v8 offset:53248 // (op 1325) Load local data 


ds_read_b64 v[102:103], v8 offset:53264 // (op 1325) Load local data 



s_barrier  // (op 4295) 
ds_read_u8 v83, v7 offset:57216 // (op 1329) Load local data 
s_barrier  // (op 4299) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[98:103], v[92:97], a[28:31], v83, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 1319) 






buffer_load_ubyte v25, v80, s[60:63], 0 offen offset:12 // (op 233) Load value




v_add_u32 v19, 8, v19 // (op 3770) 
buffer_load_dwordx3 v[52:54], v81, s[52:55], 0 offen offset:288 // (op 237) Load value


v_add_u32 v17, 192, v17 // (op 3642) 
s_mul_i32 s7, s13, 6 // (op 224) most significant: low of high * low
s_mul_i32 s6, s12, 6 // (op 224) least significant
s_mul_hi_u32 s1, s12, 6 // (op 224) most significant: high of low * low
s_add_u32 s7, s7, s1 // (op 224) most significant: sum

v_lshrrev_b64 v[80:81], 3, s[6:7] // (op 224) 


v_mov_b32 v83, v84 // (op 224) 
buffer_load_dwordx4 v[56:59], v83, s[48:51], 0 offen offset:288 // (op 224) Load value
buffer_load_dwordx2 v[60:61], v83, s[48:51], 0 offen offset:304 // (op 224) Load value
buffer_load_dwordx4 v[62:65], v83, s[48:51], 0 offen offset:312 // (op 224) Load value
buffer_load_dwordx2 v[66:67], v83, s[48:51], 0 offen offset:328 // (op 224) Load value
v_add_u32 v83, v83, v80 // (op 224) 
buffer_load_dwordx4 v[68:71], v83, s[48:51], 0 offen offset:288 // (op 224) Load value
buffer_load_dwordx2 v[72:73], v83, s[48:51], 0 offen offset:304 // (op 224) Load value
buffer_load_dwordx4 v[74:77], v83, s[48:51], 0 offen offset:312 // (op 224) Load value
buffer_load_dwordx2 v[78:79], v83, s[48:51], 0 offen offset:328 // (op 224) Load value






v_add_u32 v13, 192, v13 // (op 3578) 
buffer_load_dword v12, v82, s[56:59], 0 offen offset:12 // (op 238) Load value

s_barrier  // (op 4148) 
v_add_u32 v18, 8, v18 // (op 3706) 



s_waitcnt vmcnt(20)
ds_write_b96 v15, v[22:24] // (op 13) Store local data 
s_barrier  // (op 4135) 



s_waitcnt vmcnt(19)
ds_write_b128 v55, v[28:31] offset:3072 // (op 6) Store local data 
s_waitcnt vmcnt(18)
ds_write_b64 v55, v[32:33] offset:3088 // (op 6) Store local data 
s_waitcnt vmcnt(17)
ds_write_b128 v55, v[34:37] offset:3096 // (op 6) Store local data 
s_waitcnt vmcnt(16)
ds_write_b64 v55, v[38:39] offset:3112 // (op 6) Store local data 
s_waitcnt vmcnt(15)
ds_write_b128 v55, v[40:43] offset:3168 // (op 6) Store local data 
s_waitcnt vmcnt(14)
ds_write_b64 v55, v[44:45] offset:3184 // (op 6) Store local data 
s_waitcnt vmcnt(13)
ds_write_b128 v55, v[46:49] offset:3192 // (op 6) Store local data 
s_waitcnt vmcnt(12)
ds_write_b64 v55, v[50:51] offset:3208 // (op 6) Store local data 
s_barrier  // (op 4162) 



s_waitcnt vmcnt(11)
ds_write_b32 v14, v21 offset:27648 // (op 21) Store local data 
s_ashr_i64 s[6:7], s[36:37], 7 // (op 1282) 
s_mov_b32 s1, s6 // (op 1282) convert

s_and_b32 s6, s1, 4294967294 // (op 1282) 

s_cmp_lt_u32 s0, s6 // (op 1282) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ForLoopTop_KLoop_1282 // (op 1282) Condition: Bottom (jump to RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ForLoopTop_KLoop_1282 if true)
RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ForLoopBottom_KLoop_1282:















v_and_b32 v55, 4294967294, v1 // (op 3534) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 3534) 
v_lshl_add_u32 v13, s16, 8, v55 // (op 3534) 

s_lshr_b32 s6, s0, 1 // (op 3534) 

v_and_b32 v55, -2, v1 // (op 3534) 
v_sub_u32 v14, v1, v55 // (op 3534) 

v_lshlrev_b32 v55, 6, v14 // (op 3534) 

v_lshl_add_u32 v14, s6, 8, v55 // (op 3534) 


v_mov_b32 v27, 0 // (op 3534) convert
v_mov_b32 v26, v14 // (op 3534) convert

v_mul_lo_u32 v15, v13, s13 // (op 3534) most significant: low of low * high
v_mul_hi_u32 v55, v13, s12 // (op 3534) most significant: high of low * low
v_mul_lo_u32 v14, v13, s12 // (op 3534) least significant: low of low * low
v_add_u32 v15, v15, v55 // (op 3534) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v26 // (op 3534) least significant half
v_addc_co_u32 v15, s[0:1], v15, v27, s[0:1] // (op 3534) most significant half



v_mul_lo_u32 v17, v15, 6 // (op 3534) most significant: low of high * low
v_mul_hi_u32 v13, v14, 6 // (op 3534) most significant: high of low * low
v_mul_lo_u32 v16, v14, 6 // (op 3534) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 3534) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 3534) 

v_mov_b32 v13, v14 // (op 3534) convert

s_mul_i32 s1, s23, 6 // (op 3536) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 3536) least significant
s_mul_hi_u32 s6, s22, 6 // (op 3536) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3536) most significant: sum

s_mul_i32 s7, s12, s1 // (op 3536) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 3536) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 3536) least significant
s_mul_hi_u32 s11, s12, s0 // (op 3536) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 3536) most significant: sum
s_add_u32 s7, s7, s11 // (op 3536) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 3536) 

s_mov_b32 s48, s20 // (op 3536) reg expression
s_mov_b32 s49, s21 // (op 3536) reg expression
s_mov_b32 s50, s0 // (op 3536) reg expression
s_mov_b32 s51, 131072 // (op 3536) call()

v_and_b32 v55, -8, v1 // (op 3608) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 3608) 
v_sub_u32 v14, v1, v55 // (op 3608) 

s_lshr_b32 s6, s0, 1 // (op 3608) 

v_lshlrev_b32 v55, 4, v14 // (op 3608) 

v_lshl_add_u32 v14, s6, 8, v55 // (op 3608) 


v_mov_b32 v27, 0 // (op 3608) convert
v_mov_b32 v26, v14 // (op 3608) convert

v_lshrrev_b32 v55, 3, v1 // (op 3608) 
v_lshl_add_u32 v14, s17, 5, v55 // (op 3608) 

v_mul_lo_u32 v29, v14, s25 // (op 3608) most significant: low of low * high
v_mul_hi_u32 v55, v14, s24 // (op 3608) most significant: high of low * low
v_mul_lo_u32 v28, v14, s24 // (op 3608) least significant: low of low * low
v_add_u32 v29, v29, v55 // (op 3608) most significant: sum


v_add_co_u32 v14, s[0:1], v26, v28 // (op 3608) least significant half
v_addc_co_u32 v15, s[0:1], v27, v29, s[0:1] // (op 3608) most significant half



v_mul_lo_u32 v17, v15, 6 // (op 3608) most significant: low of high * low
v_mul_hi_u32 v55, v14, 6 // (op 3608) most significant: high of low * low
v_mul_lo_u32 v16, v14, 6 // (op 3608) least significant: low of low * low
v_add_u32 v17, v17, v55 // (op 3608) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 3608) 

v_mov_b32 v55, v14 // (op 3608) convert

s_mul_i32 s1, s9, 6 // (op 3610) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 3610) least significant
s_mul_hi_u32 s6, s8, 6 // (op 3610) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3610) most significant: sum

s_mul_i32 s7, s24, s1 // (op 3610) most significant: low of low * high
s_mul_i32 s10, s25, s0 // (op 3610) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 3610) least significant
s_mul_hi_u32 s11, s24, s0 // (op 3610) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 3610) most significant: sum
s_add_u32 s7, s7, s11 // (op 3610) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 3610) 

s_mov_b32 s52, s26 // (op 3610) reg expression
s_mov_b32 s53, s27 // (op 3610) reg expression
s_mov_b32 s54, s0 // (op 3610) reg expression
s_mov_b32 s55, 131072 // (op 3610) call()

v_lshl_add_u32 v14, s16, 8, v1 // (op 3672) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 3672) 
s_lshr_b32 s6, s0, 1 // (op 3672) 

s_lshl_b32 s7, s6, 3 // (op 3672) 

v_mul_lo_u32 v15, v14, s28 // (op 3672) 
v_add_u32 v15, v15, s7 // (op 3672) 


s_mul_i32 s1, s28, s23 // (op 3674) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 3674) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 3674) least significant
s_mul_hi_u32 s7, s28, s22 // (op 3674) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3674) most significant: sum
s_add_u32 s1, s1, s7 // (op 3674) most significant: sum


s_mov_b32 s56, s30 // (op 3674) reg expression
s_mov_b32 s57, s31 // (op 3674) reg expression
s_mov_b32 s58, s0 // (op 3674) reg expression
s_mov_b32 s59, 131072 // (op 3674) call()

v_and_b32 v14, 63, v1 // (op 3736) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 3736) 
v_lshrrev_b32 v16, 4, v14 // (op 3736) 

s_lshr_b32 s6, s0, 1 // (op 3736) 

v_lshrrev_b32 v14, 6, v1 // (op 3736) 
v_lshl_add_u32 v17, s6, 3, v16 // (op 3736) 


v_and_b32 v16, 1, v14 // (op 3736) 

v_and_b32 v14, 15, v1 // (op 3736) 
v_lshl_add_u32 v18, v16, 4, v14 // (op 3736) 


v_lshl_add_u32 v14, s17, 5, v18 // (op 3736) 

v_mul_lo_u32 v16, v14, s32 // (op 3736) 

v_add_u32 v14, v17, v16 // (op 3736) 


s_mul_i32 s1, s32, s9 // (op 3738) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 3738) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 3738) least significant
s_mul_hi_u32 s7, s32, s8 // (op 3738) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3738) most significant: sum
s_add_u32 s1, s1, s7 // (op 3738) most significant: sum


s_mov_b32 s60, s34 // (op 3738) reg expression
s_mov_b32 s61, s35 // (op 3738) reg expression
s_mov_b32 s62, s0 // (op 3738) reg expression
s_mov_b32 s63, 131072 // (op 3738) call()

v_and_b32 v16, -2, v1 // (op 3829) 
v_sub_u32 v17, v1, v16 // (op 3829) 

v_lshlrev_b32 v16, 6, v17 // (op 3829) 

v_lshrrev_b32 v17, 1, v1 // (op 3829) 
v_lshl_add_u32 v18, v17, 8, v16 // (op 3829) 


v_mul_lo_u32 v16, v18, 6 // (op 3829) 

v_lshrrev_b32 v17, 3, v16 // (op 3829) 

v_and_b32 v16, -8, v1 // (op 3887) 
v_sub_u32 v18, v1, v16 // (op 3887) 

v_lshlrev_b32 v16, 4, v18 // (op 3887) 

v_lshrrev_b32 v18, 3, v1 // (op 3887) 
v_lshl_add_u32 v19, v18, 7, v16 // (op 3887) 


v_mul_lo_u32 v16, v19, 6 // (op 3887) 

v_lshrrev_b32 v18, 3, v16 // (op 3887) 

v_lshlrev_b32 v16, 2, v1 // (op 3940) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1288) 
s_and_b32 s6, s0, 4294967294 // (op 1288) 

s_ashr_i64 s[0:1], s[36:37], 7 // (op 1287) 
s_mov_b32 s7, s0 // (op 1287) convert

s_cmp_lt_u32 s6, s7 // (op 1287) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ForLoopBottom_KLoopTail_1287 // (op 1287) Condition: Top (jump to RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ForLoopBottom_KLoopTail_1287 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ForLoopTop_KLoopTail_1287:

s_add_u32 s6, s6, 1 // (op 1289) 
s_mul_i32 s1, s13, 6 // (op 166) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 166) least significant
s_mul_hi_u32 s7, s12, 6 // (op 166) most significant: high of low * low
s_add_u32 s1, s1, s7 // (op 166) most significant: sum

v_lshrrev_b64 v[20:21], 3, s[0:1] // (op 166) 


v_mov_b32 v12, v13 // (op 166) 
v_mov_b32 v19, v12 // (op 166) 
buffer_load_dwordx4 v[22:25], v19, s[48:51], 0 offen // (op 166) Load value
buffer_load_dwordx2 v[26:27], v19, s[48:51], 0 offen offset:16 // (op 166) Load value
buffer_load_dwordx4 v[28:31], v19, s[48:51], 0 offen offset:24 // (op 166) Load value
buffer_load_dwordx2 v[32:33], v19, s[48:51], 0 offen offset:40 // (op 166) Load value
v_add_u32 v19, v19, v20 // (op 166) 
buffer_load_dwordx4 v[34:37], v19, s[48:51], 0 offen // (op 166) Load value
buffer_load_dwordx2 v[38:39], v19, s[48:51], 0 offen offset:16 // (op 166) Load value
buffer_load_dwordx4 v[40:43], v19, s[48:51], 0 offen offset:24 // (op 166) Load value
buffer_load_dwordx2 v[44:45], v19, s[48:51], 0 offen offset:40 // (op 166) Load value



v_add_u32 v13, 96, v13 // (op 3524) 


s_barrier  // (op 4138) 
v_mov_b32 v12, v55 // (op 179) 




buffer_load_dwordx3 v[46:48], v12, s[52:55], 0 offen // (op 179) Load value

v_add_u32 v55, 96, v55 // (op 3600) 
s_waitcnt vmcnt(8)
ds_write_b128 v17, v[22:25] offset:3072 // (op 167) Store local data 


v_mov_b32 v12, v15 // (op 180) 

s_waitcnt vmcnt(7)
ds_write_b64 v17, v[26:27] offset:3088 // (op 167) Store local data 
s_waitcnt vmcnt(6)
ds_write_b128 v17, v[28:31] offset:3096 // (op 167) Store local data 
s_waitcnt vmcnt(5)
ds_write_b64 v17, v[32:33] offset:3112 // (op 167) Store local data 
s_waitcnt vmcnt(4)
ds_write_b128 v17, v[34:37] offset:3168 // (op 167) Store local data 
s_waitcnt vmcnt(3)
ds_write_b64 v17, v[38:39] offset:3184 // (op 167) Store local data 
s_waitcnt vmcnt(2)
ds_write_b128 v17, v[40:43] offset:3192 // (op 167) Store local data 
buffer_load_dword v19, v12, s[56:59], 0 offen // (op 180) Load value

v_add_u32 v15, 4, v15 // (op 3664) 
s_waitcnt vmcnt(2)
ds_write_b64 v17, v[44:45] offset:3208 // (op 167) Store local data 

s_barrier  // (op 4151) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 4141) 






ds_read_b128 v[20:23], v5 offset:3072 // (op 169) Load local data 
ds_read_b64 v[24:25], v5 offset:3088 // (op 169) Load local data 
s_waitcnt vmcnt(1)
ds_write_b96 v18, v[46:48] // (op 177) Store local data 

s_barrier  // (op 4165) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 4155) 






ds_read_b128 v[26:29], v3 // (op 171) Load local data 
ds_read_b64 v[30:31], v3 offset:16 // (op 171) Load local data 
s_waitcnt vmcnt(0)
ds_write_b32 v16, v19 offset:27648 // (op 178) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 4169) 



ds_read_u8 v12, v0 offset:27648 // (op 173) Load local data 


v_mov_b32 v19, v14 // (op 175) 

buffer_load_ubyte v32, v19, s[60:63], 0 offen offset: 0 // (op 175) Load value

v_add_u32 v14, 4, v14 // (op 3728) 



ds_read_b128 v[34:37], v5 offset:6144 // (op 321) Load local data 
ds_read_b64 v[38:39], v5 offset:6160 // (op 321) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:25], v[26:31], a[0:3], v12, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 176) 





ds_read_u8 v12, v0 offset:27776 // (op 325) Load local data 



ds_read_b128 v[20:23], v5 offset:9216 // (op 483) Load local data 
ds_read_b64 v[24:25], v5 offset:9232 // (op 483) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[34:39], v[26:31], a[4:7], v12, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 327) 





ds_read_u8 v12, v0 offset:27904 // (op 487) Load local data 



ds_read_b128 v[34:37], v5 offset:12288 // (op 645) Load local data 
ds_read_b64 v[38:39], v5 offset:12304 // (op 645) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[20:25], v[26:31], a[8:11], v12, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 489) 





ds_read_u8 v12, v0 offset:28032 // (op 649) Load local data 



ds_read_b128 v[20:23], v5 offset:15360 // (op 807) Load local data 
ds_read_b64 v[24:25], v5 offset:15376 // (op 807) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[34:39], v[26:31], a[12:15], v12, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 651) 





ds_read_u8 v12, v0 offset:28160 // (op 811) Load local data 



ds_read_b128 v[34:37], v5 offset:18432 // (op 969) Load local data 
ds_read_b64 v[38:39], v5 offset:18448 // (op 969) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[20:25], v[26:31], a[16:19], v12, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 813) 





ds_read_u8 v12, v0 offset:28288 // (op 973) Load local data 



ds_read_b128 v[20:23], v5 offset:21504 // (op 1131) Load local data 
ds_read_b64 v[24:25], v5 offset:21520 // (op 1131) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[34:39], v[26:31], a[20:23], v12, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 975) 





ds_read_u8 v12, v0 offset:28416 // (op 1135) Load local data 



ds_read_b128 v[34:37], v5 offset:24576 // (op 1293) Load local data 
ds_read_b64 v[38:39], v5 offset:24592 // (op 1293) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[20:25], v[26:31], a[24:27], v12, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 1137) 





ds_read_u8 v12, v0 offset:28544 // (op 1297) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[34:39], v[26:31], a[28:31], v12, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 1299) 




s_ashr_i64 s[0:1], s[36:37], 7 // (op 1287) 
s_mov_b32 s7, s0 // (op 1287) convert

s_cmp_lt_u32 s6, s7 // (op 1287) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ForLoopTop_KLoopTail_1287 // (op 1287) Condition: Bottom (jump to RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ForLoopTop_KLoopTail_1287 if true)
RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ForLoopBottom_KLoopTail_1287:






















s_barrier  // (op 4283) 







v_cmp_eq_f32 vcc, 0.00000, v10 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ConditionalFalse_0_beta_123
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
s_branch RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ConditionalBottom_0_beta_123
RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ConditionalFalse_0_beta_123:



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
RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ConditionalBottom_0_beta_123:
















s_endpgm  // End of RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2
.LRR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_end:
.size RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2, .LRR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_end-RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2
  .amdhsa_next_free_vgpr 140
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 57344 // lds bytes
  .amdhsa_accum_offset 108
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
    DataType: B6
    DataTypeA: B6
    DataTypeB: B6
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
  - .name: RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2
    .symbol: RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 57344
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 64
    .vgpr_count: 106
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
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: BF6, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BF6, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: BF6, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BF6, pointerType: PointerGlobal}
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

