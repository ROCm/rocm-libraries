

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2
.p2align 8
.type RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2,@function
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 6943) 
v_and_b32 v3, 15, v1 // (op 6943) 
v_lshl_add_u32 v4, v0, 6, v3 // (op 6943) 


v_and_b32 v0, 48, v1 // (op 6943) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 6943) 


v_lshrrev_b32 v0, 6, v1 // (op 7000) 
v_and_b32 v4, 1, v0 // (op 7000) 

v_and_b32 v0, 15, v1 // (op 7000) 
v_lshl_add_u32 v5, v4, 6, v0 // (op 7000) 


v_and_b32 v0, 48, v1 // (op 7000) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 7000) 


v_lshrrev_b32 v0, 7, v1 // (op 7067) 
v_and_b32 v5, 15, v1 // (op 7067) 
v_lshl_add_u32 v6, v0, 6, v5 // (op 7067) 


v_and_b32 v0, 48, v1 // (op 7067) 
v_lshl_add_u32 v5, v6, 7, v0 // (op 7067) 


v_lshrrev_b32 v0, 6, v1 // (op 7111) 
v_and_b32 v6, 1, v0 // (op 7111) 

v_and_b32 v0, 15, v1 // (op 7111) 
v_lshl_add_u32 v7, v6, 6, v0 // (op 7111) 


v_and_b32 v0, 48, v1 // (op 7111) 
v_lshl_add_u32 v6, v7, 7, v0 // (op 7111) 


v_lshrrev_b32 v0, 7, v1 // (op 6882) 
v_and_b32 v7, 63, v1 // (op 6882) 
v_lshrrev_b32 v8, 4, v7 // (op 6882) 
v_lshlrev_b32 v9, 2, v8 // (op 6882) 

v_lshl_add_u32 v8, v0, 6, v9 // (op 6882) 


v_lshl_add_u32 v0, s16, 7, v8 // (op 6882) 

v_lshrrev_b32 v8, 2, v7 // (op 6882) 

v_and_b32 v7, 3, v8 // (op 6882) 

v_and_b32 v8, 3, v1 // (op 6882) 
v_lshl_add_u32 v9, v7, 2, v8 // (op 6882) 


v_lshrrev_b32 v7, 6, v1 // (op 6882) 
v_and_b32 v8, 1, v7 // (op 6882) 

v_lshl_add_u32 v7, v8, 6, v9 // (op 6882) 


v_lshl_add_u32 v8, s17, 7, v7 // (op 6882) 

v_mul_lo_u32 v7, v8, s2 // (op 6882) 

v_add_lshl_u32 v8, v0, v7, 2 // (op 6882) 


v_lshrrev_b32 v0, 7, v1 // (op 7021) 
v_and_b32 v7, 63, v1 // (op 7021) 
v_lshrrev_b32 v9, 4, v7 // (op 7021) 
s_mul_i32 s1, s2, s9 // (op 6884) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 6884) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 6884) least significant
s_mul_hi_u32 s19, s2, s8 // (op 6884) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 6884) most significant: sum
s_add_u32 s1, s1, s19 // (op 6884) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 6884) 

v_lshlrev_b32 v10, 2, v9 // (op 7021) 

v_lshl_add_u32 v9, v0, 6, v10 // (op 7021) 


v_lshl_add_u32 v0, s16, 7, v9 // (op 7021) 

v_lshrrev_b32 v9, 2, v7 // (op 7021) 

v_and_b32 v7, 3, v9 // (op 7021) 

v_and_b32 v9, 3, v1 // (op 7021) 
v_lshl_add_u32 v10, v7, 2, v9 // (op 7021) 


v_lshrrev_b32 v7, 6, v1 // (op 7021) 
v_and_b32 v9, 1, v7 // (op 7021) 

v_lshl_add_u32 v7, v9, 6, v10 // (op 7021) 


v_lshl_add_u32 v9, s17, 7, v7 // (op 7021) 

v_mul_lo_u32 v7, v9, s4 // (op 7021) 

v_add_lshl_u32 v9, v0, v7, 2 // (op 7021) 


s_mov_b32 s40, s6 // (op 6884) reg expression
s_mov_b32 s41, s7 // (op 6884) reg expression
s_mov_b32 s42, s18 // (op 6884) reg expression
s_mov_b32 s43, 131072 // (op 6884) call()

s_mul_i32 s1, s4, s9 // (op 7023) most significant: low of low * high

s_mul_i32 s6, s5, s8 // (op 7023) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 7023) least significant
s_mul_hi_u32 s7, s4, s8 // (op 7023) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 7023) most significant: sum
s_add_u32 s1, s1, s7 // (op 7023) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 7023) 

s_mov_b32 s44, s10 // (op 7023) reg expression
s_mov_b32 s45, s11 // (op 7023) reg expression
s_mov_b32 s46, s6 // (op 7023) reg expression
s_mov_b32 s47, 131072 // (op 7023) call()


v_mov_b32 v0, s14 // (op 2) Move value
v_mov_b32 v7, s15 // (op 35) Move value


v_accvgpr_write a0, 0.00000 // (op 55) call()
v_accvgpr_write a1, 0.00000 // (op 55) call()
v_accvgpr_write a2, 0.00000 // (op 55) call()
v_accvgpr_write a3, 0.00000 // (op 55) call()
v_accvgpr_write a4, 0.00000 // (op 277) call()
v_accvgpr_write a5, 0.00000 // (op 277) call()
v_accvgpr_write a6, 0.00000 // (op 277) call()
v_accvgpr_write a7, 0.00000 // (op 277) call()
v_accvgpr_write a8, 0.00000 // (op 413) call()
v_accvgpr_write a9, 0.00000 // (op 413) call()
v_accvgpr_write a10, 0.00000 // (op 413) call()
v_accvgpr_write a11, 0.00000 // (op 413) call()
v_accvgpr_write a12, 0.00000 // (op 549) call()
v_accvgpr_write a13, 0.00000 // (op 549) call()
v_accvgpr_write a14, 0.00000 // (op 549) call()
v_accvgpr_write a15, 0.00000 // (op 549) call()
v_accvgpr_write a16, 0.00000 // (op 773) call()
v_accvgpr_write a17, 0.00000 // (op 773) call()
v_accvgpr_write a18, 0.00000 // (op 773) call()
v_accvgpr_write a19, 0.00000 // (op 773) call()
v_accvgpr_write a20, 0.00000 // (op 1034) call()
v_accvgpr_write a21, 0.00000 // (op 1034) call()
v_accvgpr_write a22, 0.00000 // (op 1034) call()
v_accvgpr_write a23, 0.00000 // (op 1034) call()
v_accvgpr_write a24, 0.00000 // (op 1035) call()
v_accvgpr_write a25, 0.00000 // (op 1035) call()
v_accvgpr_write a26, 0.00000 // (op 1035) call()
v_accvgpr_write a27, 0.00000 // (op 1035) call()
v_accvgpr_write a28, 0.00000 // (op 1036) call()
v_accvgpr_write a29, 0.00000 // (op 1036) call()
v_accvgpr_write a30, 0.00000 // (op 1036) call()
v_accvgpr_write a31, 0.00000 // (op 1036) call()
v_accvgpr_write a32, 0.00000 // (op 1383) call()
v_accvgpr_write a33, 0.00000 // (op 1383) call()
v_accvgpr_write a34, 0.00000 // (op 1383) call()
v_accvgpr_write a35, 0.00000 // (op 1383) call()
v_accvgpr_write a36, 0.00000 // (op 1644) call()
v_accvgpr_write a37, 0.00000 // (op 1644) call()
v_accvgpr_write a38, 0.00000 // (op 1644) call()
v_accvgpr_write a39, 0.00000 // (op 1644) call()
v_accvgpr_write a40, 0.00000 // (op 1645) call()
v_accvgpr_write a41, 0.00000 // (op 1645) call()
v_accvgpr_write a42, 0.00000 // (op 1645) call()
v_accvgpr_write a43, 0.00000 // (op 1645) call()
v_accvgpr_write a44, 0.00000 // (op 1646) call()
v_accvgpr_write a45, 0.00000 // (op 1646) call()
v_accvgpr_write a46, 0.00000 // (op 1646) call()
v_accvgpr_write a47, 0.00000 // (op 1646) call()
v_accvgpr_write a48, 0.00000 // (op 1993) call()
v_accvgpr_write a49, 0.00000 // (op 1993) call()
v_accvgpr_write a50, 0.00000 // (op 1993) call()
v_accvgpr_write a51, 0.00000 // (op 1993) call()
v_accvgpr_write a52, 0.00000 // (op 2254) call()
v_accvgpr_write a53, 0.00000 // (op 2254) call()
v_accvgpr_write a54, 0.00000 // (op 2254) call()
v_accvgpr_write a55, 0.00000 // (op 2254) call()
v_accvgpr_write a56, 0.00000 // (op 2255) call()
v_accvgpr_write a57, 0.00000 // (op 2255) call()
v_accvgpr_write a58, 0.00000 // (op 2255) call()
v_accvgpr_write a59, 0.00000 // (op 2255) call()
v_accvgpr_write a60, 0.00000 // (op 2256) call()
v_accvgpr_write a61, 0.00000 // (op 2256) call()
v_accvgpr_write a62, 0.00000 // (op 2256) call()
v_accvgpr_write a63, 0.00000 // (op 2256) call()
v_lshrrev_b32 v10, 3, v1 // (op 6772) 
v_lshl_add_u32 v11, s16, 7, v10 // (op 6772) 

v_and_b32 v10, -8, v1 // (op 6772) 
v_sub_u32 v12, v1, v10 // (op 6772) 

v_lshlrev_b32 v10, 4, v12 // (op 6772) 

v_mov_b32 v13, 0 // (op 6772) convert
v_mov_b32 v12, v10 // (op 6772) convert

v_mul_lo_u32 v15, v11, s13 // (op 6772) most significant: low of low * high
v_mul_hi_u32 v10, v11, s12 // (op 6772) most significant: high of low * low
v_mul_lo_u32 v14, v11, s12 // (op 6772) least significant: low of low * low
v_add_u32 v15, v15, v10 // (op 6772) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v12 // (op 6772) least significant half
v_addc_co_u32 v15, s[0:1], v15, v13, s[0:1] // (op 6772) most significant half




s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 6774) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 6774) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 6774) least significant
s_mul_hi_u32 s7, s12, s22 // (op 6774) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6774) most significant: sum
s_add_u32 s1, s1, s7 // (op 6774) most significant: sum


s_mov_b32 s48, s20 // (op 6774) reg expression
s_mov_b32 s49, s21 // (op 6774) reg expression
s_mov_b32 s50, s0 // (op 6774) reg expression
s_mov_b32 s51, 131072 // (op 6774) call()

v_lshrrev_b32 v10, 3, v1 // (op 6791) 
v_lshl_add_u32 v11, s16, 7, v10 // (op 6791) 

v_and_b32 v10, -8, v1 // (op 6791) 
v_sub_u32 v12, v1, v10 // (op 6791) 

v_lshlrev_b32 v10, 4, v12 // (op 6791) 

v_mov_b32 v13, 0 // (op 6791) convert
v_mov_b32 v12, v10 // (op 6791) convert

v_mul_lo_u32 v17, v11, s13 // (op 6791) most significant: low of low * high
v_mul_hi_u32 v10, v11, s12 // (op 6791) most significant: high of low * low
v_mul_lo_u32 v16, v11, s12 // (op 6791) least significant: low of low * low
v_add_u32 v17, v17, v10 // (op 6791) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v12 // (op 6791) least significant half
v_addc_co_u32 v17, s[0:1], v17, v13, s[0:1] // (op 6791) most significant half



s_mul_i32 s1, s12, s23 // (op 6793) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 6793) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 6793) least significant
s_mul_hi_u32 s7, s12, s22 // (op 6793) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6793) most significant: sum
s_add_u32 s1, s1, s7 // (op 6793) most significant: sum


s_mov_b32 s48, s20 // (op 6793) reg expression
s_mov_b32 s49, s21 // (op 6793) reg expression
s_mov_b32 s50, s0 // (op 6793) reg expression
s_mov_b32 s51, 131072 // (op 6793) call()

v_and_b32 v10, -8, v1 // (op 6836) 
v_sub_u32 v11, v1, v10 // (op 6836) 

v_lshlrev_b32 v10, 4, v11 // (op 6836) 

v_mov_b32 v13, 0 // (op 6836) convert
v_mov_b32 v12, v10 // (op 6836) convert

v_lshrrev_b32 v10, 3, v1 // (op 6836) 
v_lshl_add_u32 v11, s17, 7, v10 // (op 6836) 

v_mul_lo_u32 v19, v11, s25 // (op 6836) most significant: low of low * high
v_mul_hi_u32 v10, v11, s24 // (op 6836) most significant: high of low * low
v_mul_lo_u32 v18, v11, s24 // (op 6836) least significant: low of low * low
v_add_u32 v19, v19, v10 // (op 6836) most significant: sum


v_add_co_u32 v10, s[0:1], v12, v18 // (op 6836) least significant half
v_addc_co_u32 v11, s[0:1], v13, v19, s[0:1] // (op 6836) most significant half



s_mul_i32 s1, s24, s9 // (op 6838) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 6838) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 6838) least significant
s_mul_hi_u32 s7, s24, s8 // (op 6838) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6838) most significant: sum
s_add_u32 s1, s1, s7 // (op 6838) most significant: sum


s_mov_b32 s52, s26 // (op 6838) reg expression
s_mov_b32 s53, s27 // (op 6838) reg expression
s_mov_b32 s54, s0 // (op 6838) reg expression
s_mov_b32 s55, 131072 // (op 6838) call()

v_and_b32 v12, -8, v1 // (op 6855) 
v_sub_u32 v13, v1, v12 // (op 6855) 

v_lshlrev_b32 v12, 4, v13 // (op 6855) 

v_mov_b32 v19, 0 // (op 6855) convert
v_mov_b32 v18, v12 // (op 6855) convert

v_lshrrev_b32 v12, 3, v1 // (op 6855) 
v_lshl_add_u32 v13, s17, 7, v12 // (op 6855) 

v_mul_lo_u32 v21, v13, s25 // (op 6855) most significant: low of low * high
v_mul_hi_u32 v12, v13, s24 // (op 6855) most significant: high of low * low
v_mul_lo_u32 v20, v13, s24 // (op 6855) least significant: low of low * low
v_add_u32 v21, v21, v12 // (op 6855) most significant: sum


v_add_co_u32 v12, s[0:1], v18, v20 // (op 6855) least significant half
v_addc_co_u32 v13, s[0:1], v19, v21, s[0:1] // (op 6855) most significant half



s_mul_i32 s1, s24, s9 // (op 6857) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 6857) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 6857) least significant
s_mul_hi_u32 s7, s24, s8 // (op 6857) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6857) most significant: sum
s_add_u32 s1, s1, s7 // (op 6857) most significant: sum


s_mov_b32 s52, s26 // (op 6857) reg expression
s_mov_b32 s53, s27 // (op 6857) reg expression
s_mov_b32 s54, s0 // (op 6857) reg expression
s_mov_b32 s55, 131072 // (op 6857) call()

v_and_b32 v18, 63, v1 // (op 7133) 
v_lshrrev_b32 v19, 4, v18 // (op 7133) 
v_and_b32 v20, 3, v19 // (op 7133) 

v_and_b32 v19, 15, v1 // (op 7133) 
v_lshl_add_u32 v21, v20, 4, v19 // (op 7133) 


v_lshrrev_b32 v19, 7, v1 // (op 7133) 
v_lshl_add_u32 v20, v19, 6, v21 // (op 7133) 


v_lshl_add_u32 v19, s16, 7, v20 // (op 7133) 

v_lshrrev_b32 v20, 6, v18 // (op 7133) 

v_lshlrev_b32 v18, 2, v20 // (op 7133) 

v_mul_lo_u32 v20, v19, s28 // (op 7133) 
v_add_u32 v20, v20, v18 // (op 7133) 


s_mul_i32 s1, s28, s23 // (op 7135) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 7135) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 7135) least significant
s_mul_hi_u32 s7, s28, s22 // (op 7135) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 7135) most significant: sum
s_add_u32 s1, s1, s7 // (op 7135) most significant: sum


s_mov_b32 s56, s30 // (op 7135) reg expression
s_mov_b32 s57, s31 // (op 7135) reg expression
s_mov_b32 s58, s0 // (op 7135) reg expression
s_mov_b32 s59, 131072 // (op 7135) call()

v_and_b32 v18, 63, v1 // (op 7154) 
v_lshrrev_b32 v19, 4, v18 // (op 7154) 
v_and_b32 v21, 3, v19 // (op 7154) 

v_and_b32 v19, 15, v1 // (op 7154) 
v_lshl_add_u32 v22, v21, 4, v19 // (op 7154) 


v_lshrrev_b32 v19, 7, v1 // (op 7154) 
v_lshl_add_u32 v21, v19, 6, v22 // (op 7154) 


v_lshl_add_u32 v19, s16, 7, v21 // (op 7154) 

v_lshrrev_b32 v21, 6, v18 // (op 7154) 

v_lshlrev_b32 v18, 2, v21 // (op 7154) 

v_mul_lo_u32 v21, v19, s28 // (op 7154) 
v_add_u32 v21, v21, v18 // (op 7154) 


s_mul_i32 s1, s28, s23 // (op 7156) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 7156) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 7156) least significant
s_mul_hi_u32 s7, s28, s22 // (op 7156) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 7156) most significant: sum
s_add_u32 s1, s1, s7 // (op 7156) most significant: sum


s_mov_b32 s56, s30 // (op 7156) reg expression
s_mov_b32 s57, s31 // (op 7156) reg expression
s_mov_b32 s58, s0 // (op 7156) reg expression
s_mov_b32 s59, 131072 // (op 7156) call()

v_and_b32 v18, 63, v1 // (op 7185) 
v_lshrrev_b32 v19, 4, v18 // (op 7185) 
v_and_b32 v22, 3, v19 // (op 7185) 

v_and_b32 v19, 15, v1 // (op 7185) 
v_lshl_add_u32 v23, v22, 4, v19 // (op 7185) 


v_lshrrev_b32 v19, 6, v1 // (op 7185) 
v_and_b32 v22, 1, v19 // (op 7185) 

v_lshl_add_u32 v19, v22, 6, v23 // (op 7185) 


v_lshl_add_u32 v22, s17, 7, v19 // (op 7185) 

v_mul_lo_u32 v19, v22, s32 // (op 7185) 

v_lshrrev_b32 v22, 6, v18 // (op 7185) 

v_lshl_add_u32 v18, v22, 2, v19 // (op 7185) 


s_mul_i32 s1, s32, s9 // (op 7187) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 7187) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 7187) least significant
s_mul_hi_u32 s7, s32, s8 // (op 7187) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 7187) most significant: sum
s_add_u32 s1, s1, s7 // (op 7187) most significant: sum


s_mov_b32 s60, s34 // (op 7187) reg expression
s_mov_b32 s61, s35 // (op 7187) reg expression
s_mov_b32 s62, s0 // (op 7187) reg expression
s_mov_b32 s63, 131072 // (op 7187) call()

v_and_b32 v19, 63, v1 // (op 7206) 
v_lshrrev_b32 v22, 4, v19 // (op 7206) 
v_and_b32 v23, 3, v22 // (op 7206) 

v_and_b32 v22, 15, v1 // (op 7206) 
v_lshl_add_u32 v24, v23, 4, v22 // (op 7206) 


v_lshrrev_b32 v22, 6, v1 // (op 7206) 
v_and_b32 v23, 1, v22 // (op 7206) 

v_lshl_add_u32 v22, v23, 6, v24 // (op 7206) 


v_lshl_add_u32 v23, s17, 7, v22 // (op 7206) 

v_mul_lo_u32 v22, v23, s32 // (op 7206) 

v_lshrrev_b32 v23, 6, v19 // (op 7206) 

v_lshl_add_u32 v19, v23, 2, v22 // (op 7206) 


s_mul_i32 s1, s32, s9 // (op 7208) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 7208) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 7208) least significant
s_mul_hi_u32 s7, s32, s8 // (op 7208) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 7208) most significant: sum
s_add_u32 s1, s1, s7 // (op 7208) most significant: sum


s_mov_b32 s60, s34 // (op 7208) reg expression
s_mov_b32 s61, s35 // (op 7208) reg expression
s_mov_b32 s62, s0 // (op 7208) reg expression
s_mov_b32 s63, 131072 // (op 7208) call()

v_and_b32 v22, -8, v1 // (op 6983) 
v_sub_u32 v23, v1, v22 // (op 6983) 

v_lshlrev_b32 v22, 4, v23 // (op 6983) 

v_lshrrev_b32 v23, 3, v1 // (op 6983) 
v_lshl_add_u32 v24, v23, 7, v22 // (op 6983) 


s_nop 0
v_readfirstlane_b32 s0, v24 // (op 6983) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 6706) 
v_mov_b64 v[22:23], s[6:7] // (op 6706) convert


v_mov_b64 v[24:25], v[10:11] // (op 6706) 
s_mov_b32 m0, s0 // (op 6706) reg expression
buffer_load_dwordx4 v24, s[52:55], 0 offen offset: 0 lds // (op 6706) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v22 // (op 6706) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 6706) most significant half

s_add_u32 m0, m0, 4096 // (op 6706) 
buffer_load_dwordx4 v24, s[52:55], 0 offen offset: 0 lds // (op 6706) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v22 // (op 6706) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 6706) most significant half

s_add_u32 m0, m0, 4096 // (op 6706) 
buffer_load_dwordx4 v24, s[52:55], 0 offen offset: 0 lds // (op 6706) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v22 // (op 6706) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 6706) most significant half

s_add_u32 m0, m0, 4096 // (op 6706) 
buffer_load_dwordx4 v24, s[52:55], 0 offen offset: 0 lds // (op 6706) Load value direct to lds



v_and_b32 v22, -8, v1 // (op 6926) 
v_sub_u32 v23, v1, v22 // (op 6926) 

v_lshlrev_b32 v22, 4, v23 // (op 6926) 

v_lshrrev_b32 v23, 3, v1 // (op 6926) 
v_lshl_add_u32 v24, v23, 7, v22 // (op 6926) 


s_nop 0
v_readfirstlane_b32 s0, v24 // (op 6926) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 5 // (op 6711) 
v_mov_b64 v[22:23], s[6:7] // (op 6711) convert


v_mov_b64 v[24:25], v[14:15] // (op 6711) 
s_add_u32 s1, 16384, s0 // (op 6711) 
s_mov_b32 m0, s1 // (op 6711) reg expression
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 6711) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v22 // (op 6711) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 6711) most significant half

s_add_u32 m0, m0, 4096 // (op 6711) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 6711) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v22 // (op 6711) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 6711) most significant half

s_add_u32 m0, m0, 4096 // (op 6711) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 6711) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v22 // (op 6711) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 6711) most significant half

s_add_u32 m0, m0, 4096 // (op 6711) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 6711) Load value direct to lds






v_mov_b32 v22, v18 // (op 74) reg expression

buffer_load_dwordx2 v[24:25], v22, s[60:63], 0 offen // (op 74) Load value




v_mov_b32 v18, v20 // (op 70) reg expression

buffer_load_dwordx2 v[22:23], v18, s[56:59], 0 offen // (op 70) Load value


v_and_b32 v18, -8, v1 // (op 7094) 
v_sub_u32 v20, v1, v18 // (op 7094) 

v_lshlrev_b32 v18, 4, v20 // (op 7094) 

v_lshrrev_b32 v20, 3, v1 // (op 7094) 
v_lshl_add_u32 v26, v20, 7, v18 // (op 7094) 


s_nop 0
v_readfirstlane_b32 s0, v26 // (op 7094) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 6716) 
v_mov_b64 v[26:27], s[6:7] // (op 6716) convert


v_mov_b64 v[28:29], v[10:11] // (op 6716) 
s_add_u32 s1, 32768, s0 // (op 6716) 
s_mov_b32 m0, s1 // (op 6716) reg expression
s_mov_b32 s6, 128 // (op 6716) call()
buffer_load_dwordx4 v28, s[52:55], s6 offen offset: 0 lds // (op 6716) Load value direct to lds
v_add_co_u32 v28, s[10:11], v28, v26 // (op 6716) least significant half
v_addc_co_u32 v29, s[10:11], 0, v29, s[10:11] // (op 6716) most significant half

s_add_u32 m0, m0, 4096 // (op 6716) 
buffer_load_dwordx4 v28, s[52:55], s6 offen offset: 0 lds // (op 6716) Load value direct to lds
v_add_co_u32 v28, s[10:11], v28, v26 // (op 6716) least significant half
v_addc_co_u32 v29, s[10:11], 0, v29, s[10:11] // (op 6716) most significant half

s_add_u32 m0, m0, 4096 // (op 6716) 
buffer_load_dwordx4 v28, s[52:55], s6 offen offset: 0 lds // (op 6716) Load value direct to lds
v_add_co_u32 v28, s[10:11], v28, v26 // (op 6716) least significant half
v_addc_co_u32 v29, s[10:11], 0, v29, s[10:11] // (op 6716) most significant half

s_add_u32 m0, m0, 4096 // (op 6716) 
buffer_load_dwordx4 v28, s[52:55], s6 offen offset: 0 lds // (op 6716) Load value direct to lds






v_and_b32 v18, -8, v1 // (op 7050) 
v_sub_u32 v20, v1, v18 // (op 7050) 

v_lshlrev_b32 v18, 4, v20 // (op 7050) 

v_lshrrev_b32 v20, 3, v1 // (op 7050) 
v_lshl_add_u32 v10, v20, 7, v18 // (op 7050) 


s_nop 0
v_readfirstlane_b32 s0, v10 // (op 7050) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 5 // (op 6718) 
v_mov_b64 v[10:11], s[6:7] // (op 6718) convert


v_mov_b64 v[26:27], v[14:15] // (op 6718) 
s_add_u32 s1, 49152, s0 // (op 6718) 
s_mov_b32 m0, s1 // (op 6718) reg expression
s_mov_b32 s6, 128 // (op 6718) call()
buffer_load_dwordx4 v26, s[48:51], s6 offen offset: 0 lds // (op 6718) Load value direct to lds
v_add_co_u32 v26, s[10:11], v26, v10 // (op 6718) least significant half
v_addc_co_u32 v27, s[10:11], 0, v27, s[10:11] // (op 6718) most significant half

s_add_u32 m0, m0, 4096 // (op 6718) 
buffer_load_dwordx4 v26, s[48:51], s6 offen offset: 0 lds // (op 6718) Load value direct to lds
v_add_co_u32 v26, s[10:11], v26, v10 // (op 6718) least significant half
v_addc_co_u32 v27, s[10:11], 0, v27, s[10:11] // (op 6718) most significant half

s_add_u32 m0, m0, 4096 // (op 6718) 
buffer_load_dwordx4 v26, s[48:51], s6 offen offset: 0 lds // (op 6718) Load value direct to lds
v_add_co_u32 v26, s[10:11], v26, v10 // (op 6718) least significant half
v_addc_co_u32 v27, s[10:11], 0, v27, s[10:11] // (op 6718) most significant half

s_add_u32 m0, m0, 4096 // (op 6718) 
buffer_load_dwordx4 v26, s[48:51], s6 offen offset: 0 lds // (op 6718) Load value direct to lds






s_waitcnt vmcnt(14)
s_barrier  // (op 7399) 



ds_read_b128 v[26:29], v4 // (op 6454) Load local data 
ds_read_b128 v[30:33], v4 offset:64 // (op 6454) Load local data 



ds_read_b128 v[34:37], v4 offset:2048 // (op 6459) Load local data 
ds_read_b128 v[38:41], v4 offset:2112 // (op 6459) Load local data 



ds_read_b128 v[42:45], v4 offset:4096 // (op 6464) Load local data 
ds_read_b128 v[46:49], v4 offset:4160 // (op 6464) Load local data 



ds_read_b128 v[50:53], v4 offset:6144 // (op 6469) Load local data 
ds_read_b128 v[54:57], v4 offset:6208 // (op 6469) Load local data 
s_waitcnt vmcnt(10)
s_barrier  // (op 7382) 



ds_read_b128 v[58:61], v3 offset:16384 // (op 6434) Load local data 
ds_read_b128 v[62:65], v3 offset:16448 // (op 6434) Load local data 



ds_read_b128 v[66:69], v3 offset:18432 // (op 6439) Load local data 
ds_read_b128 v[70:73], v3 offset:18496 // (op 6439) Load local data 



ds_read_b128 v[74:77], v3 offset:20480 // (op 6444) Load local data 
ds_read_b128 v[78:81], v3 offset:20544 // (op 6444) Load local data 



ds_read_b128 v[82:85], v3 offset:22528 // (op 6449) Load local data 
ds_read_b128 v[86:89], v3 offset:22592 // (op 6449) Load local data 
v_and_b32 v18, -8, v1 // (op 6911) 
v_sub_u32 v20, v1, v18 // (op 6911) 

v_lshlrev_b32 v18, 4, v20 // (op 6911) 

v_lshrrev_b32 v20, 3, v1 // (op 6911) 
v_lshl_add_u32 v10, v20, 7, v18 // (op 6911) 


s_nop 0
v_readfirstlane_b32 s0, v10 // (op 6911) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v18, -8, v1 // (op 6968) 
v_sub_u32 v20, v1, v18 // (op 6968) 

v_lshlrev_b32 v18, 4, v20 // (op 6968) 

v_lshrrev_b32 v20, 3, v1 // (op 6968) 
v_lshl_add_u32 v10, v20, 7, v18 // (op 6968) 


s_nop 0
v_readfirstlane_b32 s1, v10 // (op 6968) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v18, -8, v1 // (op 7035) 
v_sub_u32 v20, v1, v18 // (op 7035) 

v_lshlrev_b32 v18, 4, v20 // (op 7035) 

v_lshrrev_b32 v20, 3, v1 // (op 7035) 
v_lshl_add_u32 v10, v20, 7, v18 // (op 7035) 


s_nop 0
v_readfirstlane_b32 s6, v10 // (op 7035) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v18, -8, v1 // (op 7079) 
v_sub_u32 v20, v1, v18 // (op 7079) 

v_lshlrev_b32 v18, 4, v20 // (op 7079) 

v_lshrrev_b32 v20, 3, v1 // (op 7079) 
v_lshl_add_u32 v10, v20, 7, v18 // (op 7079) 


s_nop 0
v_readfirstlane_b32 s7, v10 // (op 7079) Wait state hazard: VALU Write Readlane Hazard

s_mov_b32 s10, 0 // (op 1466) call()
s_ashr_i64 s[14:15], s[36:37], 7 // (op 1465) 
s_mov_b32 s11, s14 // (op 1465) convert

s_and_b32 s14, s11, 4294967294 // (op 1465) 

s_cmp_lt_u32 s10, s14 // (op 1465) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopBottom_KLoop_1465 // (op 1465) Condition: Top (jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopBottom_KLoop_1465 if false)
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopTop_KLoop_1465:
s_add_u32 s10, s10, 2 // (op 1467) 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 7402) 
s_waitcnt vmcnt(9)
v_mov_b32 v10, v24 // (op 6630) reg expression

v_mov_b32 v11, v25 // (op 6630) reg expression
s_lshl_b64 s[14:15], s[24:25], 5 // (op 6679) 
v_mov_b64 v[14:15], s[14:15] // (op 6679) convert


v_mov_b64 v[90:91], v[12:13] // (op 6679) 
v_mov_b64 v[92:93], v[90:91] // (op 6679) 
s_mov_b32 m0, s1 // (op 6679) reg expression
s_mov_b32 s11, 256 // (op 6679) call()
buffer_load_dwordx4 v92, s[52:55], s11 offen offset: 0 lds // (op 6679) Load value direct to lds
v_add_co_u32 v92, s[14:15], v92, v14 // (op 6679) least significant half
v_addc_co_u32 v93, s[14:15], 0, v93, s[14:15] // (op 6679) most significant half

s_add_u32 m0, m0, 4096 // (op 6679) 
buffer_load_dwordx4 v92, s[52:55], s11 offen offset: 0 lds // (op 6679) Load value direct to lds
v_add_co_u32 v92, s[14:15], v92, v14 // (op 6679) least significant half
v_addc_co_u32 v93, s[14:15], 0, v93, s[14:15] // (op 6679) most significant half

s_add_u32 m0, m0, 4096 // (op 6679) 
buffer_load_dwordx4 v92, s[52:55], s11 offen offset: 0 lds // (op 6679) Load value direct to lds
v_add_co_u32 v92, s[14:15], v92, v14 // (op 6679) least significant half
v_addc_co_u32 v93, s[14:15], 0, v93, s[14:15] // (op 6679) most significant half

s_add_u32 m0, m0, 4096 // (op 6679) 
s_waitcnt vmcnt(11)
v_mov_b32 v94, v22 // (op 6643) reg expression
v_mov_b32 v95, v23 // (op 6643) reg expression
buffer_load_dwordx4 v92, s[52:55], s11 offen offset: 0 lds // (op 6679) Load value direct to lds



v_bfe_u32 v96, v94, 0, 8 // (op 6057) 
v_bfe_u32 v97, v94, 8, 8 // (op 6057) 
v_bfe_u32 v98, v94, 16, 8 // (op 6057) 
v_bfe_u32 v99, v94, 24, 8 // (op 6057) 
v_permlane16_swap_b32 v96, v97 // (op 6057) 
s_nop 0
v_permlane16_swap_b32 v98, v99 // (op 6057) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v96, v98 // (op 6057) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v97, v99 // (op 6057) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 7385) 

s_lshl_b64 s[14:15], s[12:13], 5 // (op 6674) 
v_mov_b64 v[14:15], s[14:15] // (op 6674) convert


v_mov_b64 v[92:93], v[16:17] // (op 6674) 
v_mov_b64 v[100:101], v[92:93] // (op 6674) 
s_add_u32 s11, 16384, s0 // (op 6674) 
s_mov_b32 m0, s11 // (op 6674) reg expression
s_mov_b32 s14, 256 // (op 6674) call()
buffer_load_dwordx4 v100, s[48:51], s14 offen offset: 0 lds // (op 6674) Load value direct to lds
v_add_co_u32 v100, s[18:19], v100, v14 // (op 6674) least significant half
v_addc_co_u32 v101, s[18:19], 0, v101, s[18:19] // (op 6674) most significant half

s_add_u32 m0, m0, 4096 // (op 6674) 
buffer_load_dwordx4 v100, s[48:51], s14 offen offset: 0 lds // (op 6674) Load value direct to lds
v_add_co_u32 v100, s[18:19], v100, v14 // (op 6674) least significant half
v_addc_co_u32 v101, s[18:19], 0, v101, s[18:19] // (op 6674) most significant half

s_add_u32 m0, m0, 4096 // (op 6674) 
v_bfe_u32 v102, v10, 0, 8 // (op 6120) 
v_bfe_u32 v103, v10, 8, 8 // (op 6120) 
v_bfe_u32 v104, v10, 16, 8 // (op 6120) 
v_bfe_u32 v105, v10, 24, 8 // (op 6120) 
v_permlane16_swap_b32 v102, v103 // (op 6120) 
s_nop 0
v_permlane16_swap_b32 v104, v105 // (op 6120) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v102, v104 // (op 6120) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v103, v105 // (op 6120) 
buffer_load_dwordx4 v100, s[48:51], s14 offen offset: 0 lds // (op 6674) Load value direct to lds
v_add_co_u32 v100, s[18:19], v100, v14 // (op 6674) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[58:65], v[26:33], a[0:3], v96, v102 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 76) 
v_addc_co_u32 v101, s[18:19], 0, v101, s[18:19] // (op 6674) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[58:65], v[34:41], a[4:7], v96, v103 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 311) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[58:65], v[42:49], a[8:11], v96, v104 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 447) 
s_add_u32 m0, m0, 4096 // (op 6674) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[58:65], v[50:57], a[12:15], v96, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 583) 
buffer_load_dwordx4 v100, s[48:51], s14 offen offset: 0 lds // (op 6674) Load value direct to lds






v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[66:73], v[26:33], a[16:19], v97, v102 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1003) 
v_mov_b32 v18, v19 // (op 6638) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[66:73], v[34:41], a[20:23], v97, v103 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 965) 
v_add_u32 v20, 8, v18 // (op 6638) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[66:73], v[42:49], a[24:27], v97, v104 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 927) 
buffer_load_dwordx2 v[24:25], v20, s[60:63], 0 offen // (op 6638) Load value


v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[66:73], v[50:57], a[28:31], v97, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 889) 
v_add_u32 v19, 8, v19 // (op 7196) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[74:81], v[26:33], a[32:35], v98, v102 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1613) 
v_mov_b32 v18, v21 // (op 6651) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[74:81], v[34:41], a[36:39], v98, v103 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1575) 
v_add_u32 v20, 8, v18 // (op 6651) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[74:81], v[42:49], a[40:43], v98, v104 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1537) 
s_waitcnt vmcnt(13)
s_barrier  // (op 7422) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[74:81], v[50:57], a[44:47], v98, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1499) 
ds_read_b128 v[106:109], v6 offset:32768 // (op 212) Load local data 
ds_read_b128 v[110:113], v6 offset:32832 // (op 212) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[82:89], v[26:33], a[48:51], v99, v102 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2223) 
ds_read_b128 v[114:117], v6 offset:34816 // (op 320) Load local data 
ds_read_b128 v[118:121], v6 offset:34880 // (op 320) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[82:89], v[34:41], a[52:55], v99, v103 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2185) 
ds_read_b128 v[122:125], v6 offset:36864 // (op 456) Load local data 
ds_read_b128 v[126:129], v6 offset:36928 // (op 456) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[82:89], v[42:49], a[56:59], v99, v104 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2147) 
ds_read_b128 v[130:133], v6 offset:38912 // (op 592) Load local data 
ds_read_b128 v[134:137], v6 offset:38976 // (op 592) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[82:89], v[50:57], a[60:63], v99, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2109) 


s_waitcnt vmcnt(9)
s_barrier  // (op 7415) 
buffer_load_dwordx2 v[22:23], v20, s[56:59], 0 offen // (op 6651) Load value





ds_read_b128 v[58:61], v5 offset:49152 // (op 210) Load local data 
ds_read_b128 v[62:65], v5 offset:49216 // (op 210) Load local data 
v_add_u32 v21, 8, v21 // (op 7144) 



ds_read_b128 v[66:69], v5 offset:51200 // (op 897) Load local data 
ds_read_b128 v[70:73], v5 offset:51264 // (op 897) Load local data 



ds_read_b128 v[74:77], v5 offset:53248 // (op 1507) Load local data 
ds_read_b128 v[78:81], v5 offset:53312 // (op 1507) Load local data 



ds_read_b128 v[96:99], v5 offset:55296 // (op 2117) Load local data 
ds_read_b128 v[100:103], v5 offset:55360 // (op 2117) Load local data 
v_bfe_u32 v138, v95, 0, 8 // (op 6072) 
v_bfe_u32 v139, v95, 8, 8 // (op 6072) 
v_bfe_u32 v140, v95, 16, 8 // (op 6072) 
v_bfe_u32 v141, v95, 24, 8 // (op 6072) 
v_permlane16_swap_b32 v138, v139 // (op 6072) 
s_nop 0
v_permlane16_swap_b32 v140, v141 // (op 6072) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v138, v140 // (op 6072) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v139, v141 // (op 6072) 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 7426) 


s_lshl_b64 s[14:15], s[24:25], 5 // (op 6701) 
v_mov_b64 v[14:15], s[14:15] // (op 6701) convert


v_mov_b64 v[94:95], v[90:91] // (op 6701) 
s_add_u32 s11, 32768, s7 // (op 6701) 
s_mov_b32 m0, s11 // (op 6701) reg expression
s_mov_b32 s14, 384 // (op 6701) call()
buffer_load_dwordx4 v94, s[52:55], s14 offen offset: 0 lds // (op 6701) Load value direct to lds
v_add_co_u32 v94, s[18:19], v94, v14 // (op 6701) least significant half
v_addc_co_u32 v95, s[18:19], 0, v95, s[18:19] // (op 6701) most significant half

s_add_u32 m0, m0, 4096 // (op 6701) 
buffer_load_dwordx4 v94, s[52:55], s14 offen offset: 0 lds // (op 6701) Load value direct to lds
v_add_co_u32 v94, s[18:19], v94, v14 // (op 6701) least significant half
v_addc_co_u32 v95, s[18:19], 0, v95, s[18:19] // (op 6701) most significant half

s_add_u32 m0, m0, 4096 // (op 6701) 
v_bfe_u32 v142, v11, 0, 8 // (op 6135) 
v_bfe_u32 v143, v11, 8, 8 // (op 6135) 
v_bfe_u32 v144, v11, 16, 8 // (op 6135) 
v_bfe_u32 v145, v11, 24, 8 // (op 6135) 
v_permlane16_swap_b32 v142, v143 // (op 6135) 
s_nop 0
v_permlane16_swap_b32 v144, v145 // (op 6135) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v142, v144 // (op 6135) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v143, v145 // (op 6135) 
buffer_load_dwordx4 v94, s[52:55], s14 offen offset: 0 lds // (op 6701) Load value direct to lds

v_add_co_u32 v94, s[18:19], v94, v14 // (op 6701) least significant half
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[58:65], v[106:113], a[0:3], v138, v142 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 217) 
v_addc_co_u32 v95, s[18:19], 0, v95, s[18:19] // (op 6701) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[58:65], v[114:121], a[4:7], v138, v143 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 312) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[58:65], v[122:129], a[8:11], v138, v144 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 448) 
s_add_u32 m0, m0, 4096 // (op 6701) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[58:65], v[130:137], a[12:15], v138, v145 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 584) 
buffer_load_dwordx4 v94, s[52:55], s14 offen offset: 0 lds // (op 6701) Load value direct to lds





s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[66:73], v[106:113], a[16:19], v139, v142 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1004) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[66:73], v[114:121], a[20:23], v139, v143 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 966) 
v_add_co_u32 v12, vcc, 256, v12 // (op 6847) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[66:73], v[122:129], a[24:27], v139, v144 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 928) 
v_addc_co_u32 v13, vcc, 0, v13, vcc // (op 6847) most significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[66:73], v[130:137], a[28:31], v139, v145 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 890) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 7419) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[74:81], v[106:113], a[32:35], v140, v142 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1614) 
s_lshl_b64 s[14:15], s[12:13], 5 // (op 6696) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[74:81], v[114:121], a[36:39], v140, v143 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1576) 
v_mov_b64 v[10:11], s[14:15] // (op 6696) convert


v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[74:81], v[122:129], a[40:43], v140, v144 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1538) 
v_mov_b64 v[14:15], v[92:93] // (op 6696) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[74:81], v[130:137], a[44:47], v140, v145 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1500) 
s_add_u32 s11, 49152, s6 // (op 6696) 
s_mov_b32 m0, s11 // (op 6696) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[96:103], v[106:113], a[48:51], v141, v142 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2224) 
s_mov_b32 s14, 384 // (op 6696) call()

buffer_load_dwordx4 v14, s[48:51], s14 offen offset: 0 lds // (op 6696) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[96:103], v[114:121], a[52:55], v141, v143 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2186) 
v_add_co_u32 v14, s[18:19], v14, v10 // (op 6696) least significant half

v_addc_co_u32 v15, s[18:19], 0, v15, s[18:19] // (op 6696) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[96:103], v[122:129], a[56:59], v141, v144 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2148) 

s_add_u32 m0, m0, 4096 // (op 6696) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[96:103], v[130:137], a[60:63], v141, v145 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2110) 




s_waitcnt vmcnt(11)
s_barrier  // (op 7408) 
buffer_load_dwordx4 v14, s[48:51], s14 offen offset: 0 lds // (op 6696) Load value direct to lds



ds_read_b128 v[26:29], v4 // (op 65) Load local data 
ds_read_b128 v[30:33], v4 offset:64 // (op 65) Load local data 
v_add_co_u32 v14, s[18:19], v14, v10 // (op 6696) least significant half



ds_read_b128 v[34:37], v4 offset:2048 // (op 308) Load local data 
v_addc_co_u32 v15, s[18:19], 0, v15, s[18:19] // (op 6696) most significant half

s_add_u32 m0, m0, 4096 // (op 6696) 
ds_read_b128 v[38:41], v4 offset:2112 // (op 308) Load local data 
buffer_load_dwordx4 v14, s[48:51], s14 offen offset: 0 lds // (op 6696) Load value direct to lds



ds_read_b128 v[42:45], v4 offset:4096 // (op 444) Load local data 
ds_read_b128 v[46:49], v4 offset:4160 // (op 444) Load local data 
v_add_co_u32 v14, s[18:19], v14, v10 // (op 6696) least significant half



ds_read_b128 v[50:53], v4 offset:6144 // (op 580) Load local data 
v_addc_co_u32 v15, s[18:19], 0, v15, s[18:19] // (op 6696) most significant half

s_add_u32 m0, m0, 4096 // (op 6696) 
buffer_load_dwordx4 v14, s[48:51], s14 offen offset: 0 lds // (op 6696) Load value direct to lds




ds_read_b128 v[54:57], v4 offset:6208 // (op 580) Load local data 

v_add_co_u32 v16, vcc, 256, v16 // (op 6783) least significant half
v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 6783) most significant half
s_waitcnt vmcnt(10)
s_barrier  // (op 7391) 



ds_read_b128 v[58:61], v3 offset:16384 // (op 60) Load local data 
ds_read_b128 v[62:65], v3 offset:16448 // (op 60) Load local data 



ds_read_b128 v[66:69], v3 offset:18432 // (op 885) Load local data 
ds_read_b128 v[70:73], v3 offset:18496 // (op 885) Load local data 



ds_read_b128 v[74:77], v3 offset:20480 // (op 1495) Load local data 
ds_read_b128 v[78:81], v3 offset:20544 // (op 1495) Load local data 



ds_read_b128 v[82:85], v3 offset:22528 // (op 2105) Load local data 
ds_read_b128 v[86:89], v3 offset:22592 // (op 2105) Load local data 
s_ashr_i64 s[14:15], s[36:37], 7 // (op 1465) 
s_mov_b32 s11, s14 // (op 1465) convert

s_and_b32 s14, s11, 4294967294 // (op 1465) 

s_cmp_lt_u32 s10, s14 // (op 1465) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopTop_KLoop_1465 // (op 1465) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopTop_KLoop_1465 if true)
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopBottom_KLoop_1465:











s_barrier  // (op 7476) 
v_lshrrev_b32 v10, 3, v1 // (op 6731) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 6731) 
v_lshl_add_u32 v11, s16, 7, v10 // (op 6731) 

s_lshr_b32 s6, s0, 1 // (op 6731) 

v_and_b32 v10, -8, v1 // (op 6731) 
v_sub_u32 v12, v1, v10 // (op 6731) 

v_lshlrev_b32 v10, 4, v12 // (op 6731) 

v_lshl_add_u32 v12, s6, 8, v10 // (op 6731) 


v_mov_b32 v21, 0 // (op 6731) convert
v_mov_b32 v20, v12 // (op 6731) convert

v_mul_lo_u32 v13, v11, s13 // (op 6731) most significant: low of low * high
v_mul_hi_u32 v10, v11, s12 // (op 6731) most significant: high of low * low
v_mul_lo_u32 v12, v11, s12 // (op 6731) least significant: low of low * low
v_add_u32 v13, v13, v10 // (op 6731) most significant: sum

v_add_co_u32 v12, s[0:1], v12, v20 // (op 6731) least significant half
v_addc_co_u32 v13, s[0:1], v13, v21, s[0:1] // (op 6731) most significant half



s_mul_i32 s1, s12, s23 // (op 6733) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 6733) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 6733) least significant
s_mul_hi_u32 s7, s12, s22 // (op 6733) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6733) most significant: sum
s_add_u32 s1, s1, s7 // (op 6733) most significant: sum


s_mov_b32 s48, s20 // (op 6733) reg expression
s_mov_b32 s49, s21 // (op 6733) reg expression
s_mov_b32 s50, s0 // (op 6733) reg expression
s_mov_b32 s51, 131072 // (op 6733) call()

v_and_b32 v10, -8, v1 // (op 6813) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 6813) 
v_sub_u32 v11, v1, v10 // (op 6813) 

s_lshr_b32 s6, s0, 1 // (op 6813) 

v_lshlrev_b32 v10, 4, v11 // (op 6813) 

v_lshl_add_u32 v11, s6, 8, v10 // (op 6813) 


v_mov_b32 v15, 0 // (op 6813) convert
v_mov_b32 v14, v11 // (op 6813) convert

v_lshrrev_b32 v10, 3, v1 // (op 6813) 
v_lshl_add_u32 v11, s17, 7, v10 // (op 6813) 

v_mul_lo_u32 v17, v11, s25 // (op 6813) most significant: low of low * high
v_mul_hi_u32 v10, v11, s24 // (op 6813) most significant: high of low * low
v_mul_lo_u32 v16, v11, s24 // (op 6813) least significant: low of low * low
v_add_u32 v17, v17, v10 // (op 6813) most significant: sum


v_add_co_u32 v10, s[0:1], v14, v16 // (op 6813) least significant half
v_addc_co_u32 v11, s[0:1], v15, v17, s[0:1] // (op 6813) most significant half



s_mul_i32 s1, s24, s9 // (op 6815) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 6815) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 6815) least significant
s_mul_hi_u32 s7, s24, s8 // (op 6815) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6815) most significant: sum
s_add_u32 s1, s1, s7 // (op 6815) most significant: sum


s_mov_b32 s52, s26 // (op 6815) reg expression
s_mov_b32 s53, s27 // (op 6815) reg expression
s_mov_b32 s54, s0 // (op 6815) reg expression
s_mov_b32 s55, 131072 // (op 6815) call()

v_and_b32 v14, -8, v1 // (op 6896) 
v_sub_u32 v15, v1, v14 // (op 6896) 

v_lshlrev_b32 v14, 4, v15 // (op 6896) 

v_lshrrev_b32 v15, 3, v1 // (op 6896) 
v_lshl_add_u32 v16, v15, 7, v14 // (op 6896) 


s_nop 0
v_readfirstlane_b32 s0, v16 // (op 6896) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -8, v1 // (op 6955) 
v_sub_u32 v15, v1, v14 // (op 6955) 

v_lshlrev_b32 v14, 4, v15 // (op 6955) 

v_lshrrev_b32 v15, 3, v1 // (op 6955) 
v_lshl_add_u32 v16, v15, 7, v14 // (op 6955) 


s_nop 0
v_readfirstlane_b32 s1, v16 // (op 6955) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, 63, v1 // (op 7233) 
s_ashr_i64 s[6:7], s[36:37], 7 // (op 7233) 
v_lshrrev_b32 v15, 4, v14 // (op 7233) 
s_lshr_b32 s10, s6, 1 // (op 7233) 

v_and_b32 v16, 3, v15 // (op 7233) 

v_and_b32 v15, 15, v1 // (op 7233) 
v_lshl_add_u32 v17, v16, 4, v15 // (op 7233) 


v_lshrrev_b32 v15, 7, v1 // (op 7233) 
v_lshl_add_u32 v16, v15, 6, v17 // (op 7233) 


v_lshl_add_u32 v15, s16, 7, v16 // (op 7233) 

v_lshrrev_b32 v16, 6, v14 // (op 7233) 

v_lshlrev_b32 v14, 2, v16 // (op 7233) 

v_lshl_add_u32 v16, s10, 3, v14 // (op 7233) 


v_mul_lo_u32 v14, v15, s28 // (op 7233) 
v_add_u32 v14, v14, v16 // (op 7233) 


s_mul_i32 s7, s28, s23 // (op 7235) most significant: low of low * high
s_mul_i32 s10, s29, s22 // (op 7235) most significant: low of high * low
s_mul_i32 s6, s28, s22 // (op 7235) least significant
s_mul_hi_u32 s11, s28, s22 // (op 7235) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 7235) most significant: sum
s_add_u32 s7, s7, s11 // (op 7235) most significant: sum


s_mov_b32 s56, s30 // (op 7235) reg expression
s_mov_b32 s57, s31 // (op 7235) reg expression
s_mov_b32 s58, s6 // (op 7235) reg expression
s_mov_b32 s59, 131072 // (op 7235) call()

v_and_b32 v15, 63, v1 // (op 7260) 
s_ashr_i64 s[6:7], s[36:37], 7 // (op 7260) 
v_lshrrev_b32 v16, 6, v15 // (op 7260) 
s_lshr_b32 s10, s6, 1 // (op 7260) 

v_lshlrev_b32 v17, 2, v16 // (op 7260) 

v_lshl_add_u32 v16, s10, 3, v17 // (op 7260) 


v_lshrrev_b32 v17, 4, v15 // (op 7260) 

v_and_b32 v15, 3, v17 // (op 7260) 

v_and_b32 v17, 15, v1 // (op 7260) 
v_lshl_add_u32 v18, v15, 4, v17 // (op 7260) 


v_lshrrev_b32 v15, 6, v1 // (op 7260) 
v_and_b32 v17, 1, v15 // (op 7260) 

v_lshl_add_u32 v15, v17, 6, v18 // (op 7260) 


v_lshl_add_u32 v17, s17, 7, v15 // (op 7260) 

v_mul_lo_u32 v15, v17, s32 // (op 7260) 

v_add_u32 v17, v16, v15 // (op 7260) 


s_mul_i32 s7, s32, s9 // (op 7262) most significant: low of low * high
s_mul_i32 s10, s33, s8 // (op 7262) most significant: low of high * low
s_mul_i32 s6, s32, s8 // (op 7262) least significant
s_mul_hi_u32 s11, s32, s8 // (op 7262) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 7262) most significant: sum
s_add_u32 s7, s7, s11 // (op 7262) most significant: sum


s_mov_b32 s60, s34 // (op 7262) reg expression
s_mov_b32 s61, s35 // (op 7262) reg expression
s_mov_b32 s62, s6 // (op 7262) reg expression
s_mov_b32 s63, 131072 // (op 7262) call()

s_ashr_i64 s[6:7], s[36:37], 7 // (op 153) 
s_and_b32 s10, s6, 4294967294 // (op 153) 

s_ashr_i64 s[6:7], s[36:37], 7 // (op 152) 
s_mov_b32 s11, s6 // (op 152) convert

s_cmp_lt_u32 s10, s11 // (op 152) 











s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopBottom_KLoopTail_152 // (op 152) Condition: Top (jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopBottom_KLoopTail_152 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopTop_KLoopTail_152:
s_barrier  // (op 7388) 
s_add_u32 s10, s10, 1 // (op 154) 

s_lshl_b64 s[6:7], s[12:13], 5 // (op 6684) 
v_mov_b64 v[18:19], s[6:7] // (op 6684) convert


v_mov_b64 v[20:21], v[12:13] // (op 6684) 
v_mov_b64 v[22:23], v[20:21] // (op 6684) 
s_add_u32 s11, 16384, s0 // (op 6684) 
s_mov_b32 m0, s11 // (op 6684) reg expression
buffer_load_dwordx4 v22, s[48:51], 0 offen offset: 0 lds // (op 6684) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v18 // (op 6684) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 6684) most significant half

s_add_u32 m0, m0, 4096 // (op 6684) 
buffer_load_dwordx4 v22, s[48:51], 0 offen offset: 0 lds // (op 6684) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v18 // (op 6684) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 6684) most significant half

s_add_u32 m0, m0, 4096 // (op 6684) 
buffer_load_dwordx4 v22, s[48:51], 0 offen offset: 0 lds // (op 6684) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v18 // (op 6684) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 6684) most significant half

s_add_u32 m0, m0, 4096 // (op 6684) 
buffer_load_dwordx4 v22, s[48:51], 0 offen offset: 0 lds // (op 6684) Load value direct to lds




v_add_co_u32 v12, vcc, 128, v12 // (op 6721) least significant half
v_addc_co_u32 v13, vcc, 0, v13, vcc // (op 6721) most significant half
s_barrier  // (op 7405) 

s_lshl_b64 s[6:7], s[24:25], 5 // (op 6690) 
v_mov_b64 v[18:19], s[6:7] // (op 6690) convert


v_mov_b64 v[20:21], v[10:11] // (op 6690) 
v_mov_b64 v[22:23], v[20:21] // (op 6690) 
s_mov_b32 m0, s1 // (op 6690) reg expression
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 6690) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v18 // (op 6690) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 6690) most significant half

s_add_u32 m0, m0, 4096 // (op 6690) 
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 6690) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v18 // (op 6690) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 6690) most significant half

s_add_u32 m0, m0, 4096 // (op 6690) 
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 6690) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v18 // (op 6690) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 6690) most significant half

s_add_u32 m0, m0, 4096 // (op 6690) 
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 6690) Load value direct to lds



v_add_co_u32 v10, vcc, 128, v10 // (op 6805) least significant half
v_addc_co_u32 v11, vcc, 0, v11, vcc // (op 6805) most significant half
s_waitcnt vmcnt(4)
s_barrier  // (op 7395) 



ds_read_b128 v[18:21], v3 offset:16384 // (op 160) Load local data 
ds_read_b128 v[22:25], v3 offset:16448 // (op 160) Load local data 
s_waitcnt vmcnt(0)
s_barrier  // (op 7411) 



ds_read_b128 v[26:29], v4 // (op 162) Load local data 
ds_read_b128 v[30:33], v4 offset:64 // (op 162) Load local data 


v_mov_b32 v15, v14 // (op 164) 
v_mov_b32 v16, v15 // (op 164) reg expression

buffer_load_dwordx2 v[34:35], v16, s[56:59], 0 offen // (op 164) Load value


v_add_u32 v14, 4, v14 // (op 7223) 


v_mov_b32 v15, v17 // (op 166) 
v_mov_b32 v16, v15 // (op 166) reg expression

buffer_load_dwordx2 v[36:37], v16, s[60:63], 0 offen // (op 166) Load value


v_add_u32 v17, 4, v17 // (op 7250) 
s_waitcnt vmcnt(1)
v_bfe_u32 v38, v34, 0, 8 // (op 6184) 



v_bfe_u32 v39, v34, 8, 8 // (op 6184) 
v_bfe_u32 v40, v34, 16, 8 // (op 6184) 
v_bfe_u32 v41, v34, 24, 8 // (op 6184) 
v_permlane16_swap_b32 v38, v39 // (op 6184) 
ds_read_b128 v[42:45], v4 offset:2048 // (op 291) Load local data 
ds_read_b128 v[46:49], v4 offset:2112 // (op 291) Load local data 
v_permlane16_swap_b32 v40, v41 // (op 6184) 



s_nop 1
v_permlane32_swap_b32 v38, v40 // (op 6184) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v39, v41 // (op 6184) 
ds_read_b128 v[50:53], v4 offset:4096 // (op 427) Load local data 

ds_read_b128 v[54:57], v4 offset:4160 // (op 427) Load local data 



ds_read_b128 v[58:61], v4 offset:6144 // (op 563) Load local data 
s_waitcnt vmcnt(0)
v_bfe_u32 v66, v36, 0, 8 // (op 6214) 
v_bfe_u32 v67, v36, 8, 8 // (op 6214) 
v_bfe_u32 v68, v36, 16, 8 // (op 6214) 
v_bfe_u32 v69, v36, 24, 8 // (op 6214) 
v_permlane16_swap_b32 v66, v67 // (op 6214) 
s_nop 0
v_permlane16_swap_b32 v68, v69 // (op 6214) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v66, v68 // (op 6214) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v67, v69 // (op 6214) 
ds_read_b128 v[62:65], v4 offset:6208 // (op 563) Load local data 




s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:25], v[26:33], a[0:3], v38, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 167) 
ds_read_b128 v[70:73], v3 offset:18432 // (op 785) Load local data 
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[18:25], v[42:49], a[4:7], v38, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 294) 
ds_read_b128 v[74:77], v3 offset:18496 // (op 785) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:25], v[50:57], a[8:11], v38, v68 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 430) 
ds_read_b128 v[78:81], v3 offset:20480 // (op 1395) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[18:25], v[58:65], a[12:15], v38, v69 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 566) 
ds_read_b128 v[82:85], v3 offset:20544 // (op 1395) Load local data 




s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[70:77], v[26:33], a[16:19], v39, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 789) 
ds_read_b128 v[18:21], v3 offset:22528 // (op 2005) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[70:77], v[50:57], a[24:27], v39, v68 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 843) 
ds_read_b128 v[22:25], v3 offset:22592 // (op 2005) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[70:77], v[58:65], a[28:31], v39, v69 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 870) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[70:77], v[42:49], a[20:23], v39, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 816) 

s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[78:85], v[26:33], a[32:35], v40, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1399) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[78:85], v[50:57], a[40:43], v40, v68 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1453) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[78:85], v[58:65], a[44:47], v40, v69 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1480) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[78:85], v[42:49], a[36:39], v40, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1426) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[18:25], v[26:33], a[48:51], v41, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2009) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[18:25], v[50:57], a[56:59], v41, v68 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2063) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[18:25], v[58:65], a[60:63], v41, v69 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2090) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[18:25], v[42:49], a[52:55], v41, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2036) 




s_ashr_i64 s[6:7], s[36:37], 7 // (op 152) 
s_mov_b32 s11, s6 // (op 152) convert

s_cmp_lt_u32 s10, s11 // (op 152) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopTop_KLoopTail_152 // (op 152) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopTop_KLoopTail_152 if true)
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopBottom_KLoopTail_152:





















s_barrier  // (op 7452) 





v_cmp_eq_f32 vcc, 0.00000, v0 // (op 114) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ConditionalFalse_0_beta_114 // (op 114) Condition: False, jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ConditionalFalse_0_beta_114
v_accvgpr_read v10, a0 // (op 42) 
v_accvgpr_read v11, a1 // (op 42) 
v_accvgpr_read v12, a2 // (op 42) 
v_accvgpr_read v13, a3 // (op 42) 
v_mul_f32 v14, v7, v10 // (op 42) 
v_mul_f32 v15, v7, v11 // (op 42) 
v_mul_f32 v16, v7, v12 // (op 42) 
v_mul_f32 v17, v7, v13 // (op 42) 



v_mov_b32 v3, v9 // (op 48) reg expression

buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 48) Store value


v_accvgpr_read v10, a4 // (op 2902) 
v_accvgpr_read v11, a5 // (op 2902) 
v_accvgpr_read v12, a6 // (op 2902) 
v_accvgpr_read v13, a7 // (op 2902) 
v_mul_f32 v14, v7, v10 // (op 2902) 
v_mul_f32 v15, v7, v11 // (op 2902) 
v_mul_f32 v16, v7, v12 // (op 2902) 
v_mul_f32 v17, v7, v13 // (op 2902) 



s_lshl_b32 s0, s4, 6 // (op 2904) 
v_add_u32 v3, v9, s0 // (op 2904) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 2904) Store value


v_accvgpr_read v10, a8 // (op 2914) 
v_accvgpr_read v11, a9 // (op 2914) 
v_accvgpr_read v12, a10 // (op 2914) 
v_accvgpr_read v13, a11 // (op 2914) 
v_mul_f32 v14, v7, v10 // (op 2914) 
v_mul_f32 v15, v7, v11 // (op 2914) 
v_mul_f32 v16, v7, v12 // (op 2914) 
v_mul_f32 v17, v7, v13 // (op 2914) 



s_lshl_b32 s0, s4, 7 // (op 2916) 
v_add_u32 v3, v9, s0 // (op 2916) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 2916) Store value


v_accvgpr_read v10, a12 // (op 2926) 
v_accvgpr_read v11, a13 // (op 2926) 
v_accvgpr_read v12, a14 // (op 2926) 
v_accvgpr_read v13, a15 // (op 2926) 
v_mul_f32 v14, v7, v10 // (op 2926) 
v_mul_f32 v15, v7, v11 // (op 2926) 
v_mul_f32 v16, v7, v12 // (op 2926) 
v_mul_f32 v17, v7, v13 // (op 2926) 



s_lshl_b32 s0, s4, 6 // (op 2928) 
s_mul_i32 s1, 3, s0 // (op 2928) 

v_add_u32 v3, v9, s1 // (op 2928) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 2928) Store value


v_accvgpr_read v10, a16 // (op 2960) 
v_accvgpr_read v11, a17 // (op 2960) 
v_accvgpr_read v12, a18 // (op 2960) 
v_accvgpr_read v13, a19 // (op 2960) 
v_mul_f32 v14, v7, v10 // (op 2960) 
v_mul_f32 v15, v7, v11 // (op 2960) 
v_mul_f32 v16, v7, v12 // (op 2960) 
v_mul_f32 v17, v7, v13 // (op 2960) 



v_add_u32 v3, 64, v9 // (op 2963) 

buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 2963) Store value


v_accvgpr_read v10, a20 // (op 2973) 
v_accvgpr_read v11, a21 // (op 2973) 
v_accvgpr_read v12, a22 // (op 2973) 
v_accvgpr_read v13, a23 // (op 2973) 
v_mul_f32 v14, v7, v10 // (op 2973) 
v_mul_f32 v15, v7, v11 // (op 2973) 
v_mul_f32 v16, v7, v12 // (op 2973) 
v_mul_f32 v17, v7, v13 // (op 2973) 



s_lshl_b32 s0, s4, 6 // (op 2965) 
s_add_u32 s0, s0, 64 // (op 2965) 
v_add_u32 v3, v9, s0 // (op 2965) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 2965) Store value


v_accvgpr_read v10, a24 // (op 2978) 
v_accvgpr_read v11, a25 // (op 2978) 
v_accvgpr_read v12, a26 // (op 2978) 
v_accvgpr_read v13, a27 // (op 2978) 
v_mul_f32 v14, v7, v10 // (op 2978) 
v_mul_f32 v15, v7, v11 // (op 2978) 
v_mul_f32 v16, v7, v12 // (op 2978) 
v_mul_f32 v17, v7, v13 // (op 2978) 



s_lshl_b32 s0, s4, 7 // (op 2967) 
s_add_u32 s0, s0, 64 // (op 2967) 
v_add_u32 v3, v9, s0 // (op 2967) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 2967) Store value


v_accvgpr_read v10, a28 // (op 2983) 
v_accvgpr_read v11, a29 // (op 2983) 
v_accvgpr_read v12, a30 // (op 2983) 
v_accvgpr_read v13, a31 // (op 2983) 
v_mul_f32 v14, v7, v10 // (op 2983) 
v_mul_f32 v15, v7, v11 // (op 2983) 
v_mul_f32 v16, v7, v12 // (op 2983) 
v_mul_f32 v17, v7, v13 // (op 2983) 



s_lshl_b32 s0, s4, 6 // (op 2969) 
s_mul_i32 s1, 3, s0 // (op 2969) 
s_add_u32 s1, s1, 64 // (op 2969) 

v_add_u32 v3, v9, s1 // (op 2969) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 2969) Store value


v_accvgpr_read v10, a32 // (op 3028) 
v_accvgpr_read v11, a33 // (op 3028) 
v_accvgpr_read v12, a34 // (op 3028) 
v_accvgpr_read v13, a35 // (op 3028) 
v_mul_f32 v14, v7, v10 // (op 3028) 
v_mul_f32 v15, v7, v11 // (op 3028) 
v_mul_f32 v16, v7, v12 // (op 3028) 
v_mul_f32 v17, v7, v13 // (op 3028) 



v_add_u32 v3, 128, v9 // (op 3031) 

buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 3031) Store value


v_accvgpr_read v10, a36 // (op 3041) 
v_accvgpr_read v11, a37 // (op 3041) 
v_accvgpr_read v12, a38 // (op 3041) 
v_accvgpr_read v13, a39 // (op 3041) 
v_mul_f32 v14, v7, v10 // (op 3041) 
v_mul_f32 v15, v7, v11 // (op 3041) 
v_mul_f32 v16, v7, v12 // (op 3041) 
v_mul_f32 v17, v7, v13 // (op 3041) 



s_lshl_b32 s0, s4, 6 // (op 3033) 
s_add_u32 s0, s0, 128 // (op 3033) 
v_add_u32 v3, v9, s0 // (op 3033) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 3033) Store value


v_accvgpr_read v10, a40 // (op 3046) 
v_accvgpr_read v11, a41 // (op 3046) 
v_accvgpr_read v12, a42 // (op 3046) 
v_accvgpr_read v13, a43 // (op 3046) 
v_mul_f32 v14, v7, v10 // (op 3046) 
v_mul_f32 v15, v7, v11 // (op 3046) 
v_mul_f32 v16, v7, v12 // (op 3046) 
v_mul_f32 v17, v7, v13 // (op 3046) 



s_lshl_b32 s0, s4, 7 // (op 3035) 
s_add_u32 s0, s0, 128 // (op 3035) 
v_add_u32 v3, v9, s0 // (op 3035) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 3035) Store value


v_accvgpr_read v10, a44 // (op 3051) 
v_accvgpr_read v11, a45 // (op 3051) 
v_accvgpr_read v12, a46 // (op 3051) 
v_accvgpr_read v13, a47 // (op 3051) 
v_mul_f32 v14, v7, v10 // (op 3051) 
v_mul_f32 v15, v7, v11 // (op 3051) 
v_mul_f32 v16, v7, v12 // (op 3051) 
v_mul_f32 v17, v7, v13 // (op 3051) 



s_lshl_b32 s0, s4, 6 // (op 3037) 
s_mul_i32 s1, 3, s0 // (op 3037) 
s_add_u32 s1, s1, 128 // (op 3037) 

v_add_u32 v3, v9, s1 // (op 3037) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 3037) Store value


v_accvgpr_read v10, a48 // (op 3096) 
v_accvgpr_read v11, a49 // (op 3096) 
v_accvgpr_read v12, a50 // (op 3096) 
v_accvgpr_read v13, a51 // (op 3096) 
v_mul_f32 v14, v7, v10 // (op 3096) 
v_mul_f32 v15, v7, v11 // (op 3096) 
v_mul_f32 v16, v7, v12 // (op 3096) 
v_mul_f32 v17, v7, v13 // (op 3096) 



v_add_u32 v3, 192, v9 // (op 3099) 

buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 3099) Store value


v_accvgpr_read v10, a52 // (op 3109) 
v_accvgpr_read v11, a53 // (op 3109) 
v_accvgpr_read v12, a54 // (op 3109) 
v_accvgpr_read v13, a55 // (op 3109) 
v_mul_f32 v14, v7, v10 // (op 3109) 
v_mul_f32 v15, v7, v11 // (op 3109) 
v_mul_f32 v16, v7, v12 // (op 3109) 
v_mul_f32 v17, v7, v13 // (op 3109) 



s_lshl_b32 s0, s4, 6 // (op 3101) 
s_add_u32 s0, s0, 192 // (op 3101) 
v_add_u32 v3, v9, s0 // (op 3101) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 3101) Store value


v_accvgpr_read v10, a56 // (op 3114) 
v_accvgpr_read v11, a57 // (op 3114) 
v_accvgpr_read v12, a58 // (op 3114) 
v_accvgpr_read v13, a59 // (op 3114) 
v_mul_f32 v14, v7, v10 // (op 3114) 
v_mul_f32 v15, v7, v11 // (op 3114) 
v_mul_f32 v16, v7, v12 // (op 3114) 
v_mul_f32 v17, v7, v13 // (op 3114) 



s_lshl_b32 s0, s4, 7 // (op 3103) 
s_add_u32 s0, s0, 192 // (op 3103) 
v_add_u32 v3, v9, s0 // (op 3103) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 3103) Store value


v_accvgpr_read v10, a60 // (op 3119) 
v_accvgpr_read v11, a61 // (op 3119) 
v_accvgpr_read v12, a62 // (op 3119) 
v_accvgpr_read v13, a63 // (op 3119) 
v_mul_f32 v14, v7, v10 // (op 3119) 
v_mul_f32 v15, v7, v11 // (op 3119) 
v_mul_f32 v16, v7, v12 // (op 3119) 
v_mul_f32 v17, v7, v13 // (op 3119) 



s_lshl_b32 s0, s4, 6 // (op 3105) 
s_mul_i32 s1, 3, s0 // (op 3105) 
s_add_u32 s1, s1, 192 // (op 3105) 

v_add_u32 v3, v9, s1 // (op 3105) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 3105) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ConditionalBottom_0_beta_114 // (op 114) Condition: Done, jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ConditionalBottom_0_beta_114
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ConditionalFalse_0_beta_114:


v_add_u32 v3, 128, v8 // (op 3384) 

buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 3384) Load value

v_accvgpr_read v14, a32 // (op 3372) 
v_accvgpr_read v15, a33 // (op 3372) 
v_accvgpr_read v16, a34 // (op 3372) 
v_accvgpr_read v17, a35 // (op 3372) 
v_mul_f32 v18, v7, v14 // (op 3372) 
v_mul_f32 v19, v7, v15 // (op 3372) 
v_mul_f32 v20, v7, v16 // (op 3372) 
v_mul_f32 v21, v7, v17 // (op 3372) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 3383) 
v_mul_f32 v15, v0, v11 // (op 3383) 
v_mul_f32 v16, v0, v12 // (op 3383) 
v_mul_f32 v17, v0, v13 // (op 3383) 

v_add_f32 v10, v14, v18 // (op 3373) 
v_add_f32 v11, v15, v19 // (op 3373) 
v_add_f32 v12, v16, v20 // (op 3373) 
v_add_f32 v13, v17, v21 // (op 3373) 




v_add_u32 v3, 128, v9 // (op 3375) 

buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 3375) Store value




s_lshl_b32 s0, s2, 6 // (op 3388) 
s_add_u32 s0, s0, 128 // (op 3388) 
v_add_u32 v3, v8, s0 // (op 3388) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 3388) Load value

v_accvgpr_read v14, a36 // (op 3398) 
v_accvgpr_read v15, a37 // (op 3398) 
v_accvgpr_read v16, a38 // (op 3398) 
v_accvgpr_read v17, a39 // (op 3398) 
v_mul_f32 v18, v7, v14 // (op 3398) 
v_mul_f32 v19, v7, v15 // (op 3398) 
v_mul_f32 v20, v7, v16 // (op 3398) 
v_mul_f32 v21, v7, v17 // (op 3398) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 3386) 
v_mul_f32 v15, v0, v11 // (op 3386) 
v_mul_f32 v16, v0, v12 // (op 3386) 
v_mul_f32 v17, v0, v13 // (op 3386) 

v_add_f32 v10, v14, v18 // (op 3387) 
v_add_f32 v11, v15, v19 // (op 3387) 
v_add_f32 v12, v16, v20 // (op 3387) 
v_add_f32 v13, v17, v21 // (op 3387) 




s_lshl_b32 s0, s4, 6 // (op 3377) 
s_add_u32 s0, s0, 128 // (op 3377) 
v_add_u32 v3, v9, s0 // (op 3377) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 3377) Store value




s_lshl_b32 s0, s2, 7 // (op 3392) 
s_add_u32 s0, s0, 128 // (op 3392) 
v_add_u32 v3, v8, s0 // (op 3392) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 3392) Load value

v_accvgpr_read v14, a40 // (op 3400) 
v_accvgpr_read v15, a41 // (op 3400) 
v_accvgpr_read v16, a42 // (op 3400) 
v_accvgpr_read v17, a43 // (op 3400) 
v_mul_f32 v18, v7, v14 // (op 3400) 
v_mul_f32 v19, v7, v15 // (op 3400) 
v_mul_f32 v20, v7, v16 // (op 3400) 
v_mul_f32 v21, v7, v17 // (op 3400) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 3390) 
v_mul_f32 v15, v0, v11 // (op 3390) 
v_mul_f32 v16, v0, v12 // (op 3390) 
v_mul_f32 v17, v0, v13 // (op 3390) 

v_add_f32 v10, v14, v18 // (op 3391) 
v_add_f32 v11, v15, v19 // (op 3391) 
v_add_f32 v12, v16, v20 // (op 3391) 
v_add_f32 v13, v17, v21 // (op 3391) 




s_lshl_b32 s0, s4, 7 // (op 3379) 
s_add_u32 s0, s0, 128 // (op 3379) 
v_add_u32 v3, v9, s0 // (op 3379) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 3379) Store value




s_lshl_b32 s0, s2, 6 // (op 3396) 
s_mul_i32 s1, 3, s0 // (op 3396) 
s_add_u32 s1, s1, 128 // (op 3396) 

v_add_u32 v3, v8, s1 // (op 3396) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 3396) Load value

v_accvgpr_read v14, a44 // (op 3402) 
v_accvgpr_read v15, a45 // (op 3402) 
v_accvgpr_read v16, a46 // (op 3402) 
v_accvgpr_read v17, a47 // (op 3402) 
v_mul_f32 v18, v7, v14 // (op 3402) 
v_mul_f32 v19, v7, v15 // (op 3402) 
v_mul_f32 v20, v7, v16 // (op 3402) 
v_mul_f32 v21, v7, v17 // (op 3402) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 3394) 
v_mul_f32 v15, v0, v11 // (op 3394) 
v_mul_f32 v16, v0, v12 // (op 3394) 
v_mul_f32 v17, v0, v13 // (op 3394) 

v_add_f32 v10, v14, v18 // (op 3395) 
v_add_f32 v11, v15, v19 // (op 3395) 
v_add_f32 v12, v16, v20 // (op 3395) 
v_add_f32 v13, v17, v21 // (op 3395) 




s_lshl_b32 s0, s4, 6 // (op 3381) 
s_mul_i32 s1, 3, s0 // (op 3381) 
s_add_u32 s1, s1, 128 // (op 3381) 

v_add_u32 v3, v9, s1 // (op 3381) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 3381) Store value




v_add_u32 v3, 64, v8 // (op 3305) 

s_nop 0
buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 3305) Load value

v_accvgpr_read v14, a16 // (op 3293) 
v_accvgpr_read v15, a17 // (op 3293) 
v_accvgpr_read v16, a18 // (op 3293) 
v_accvgpr_read v17, a19 // (op 3293) 
v_mul_f32 v18, v7, v14 // (op 3293) 
v_mul_f32 v19, v7, v15 // (op 3293) 
v_mul_f32 v20, v7, v16 // (op 3293) 
v_mul_f32 v21, v7, v17 // (op 3293) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 3304) 
v_mul_f32 v15, v0, v11 // (op 3304) 
v_mul_f32 v16, v0, v12 // (op 3304) 
v_mul_f32 v17, v0, v13 // (op 3304) 

v_add_f32 v10, v14, v18 // (op 3294) 
v_add_f32 v11, v15, v19 // (op 3294) 
v_add_f32 v12, v16, v20 // (op 3294) 
v_add_f32 v13, v17, v21 // (op 3294) 




v_add_u32 v3, 64, v9 // (op 3296) 

buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 3296) Store value




s_lshl_b32 s0, s2, 6 // (op 3309) 
s_add_u32 s0, s0, 64 // (op 3309) 
v_add_u32 v3, v8, s0 // (op 3309) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 3309) Load value

v_accvgpr_read v14, a20 // (op 3319) 
v_accvgpr_read v15, a21 // (op 3319) 
v_accvgpr_read v16, a22 // (op 3319) 
v_accvgpr_read v17, a23 // (op 3319) 
v_mul_f32 v18, v7, v14 // (op 3319) 
v_mul_f32 v19, v7, v15 // (op 3319) 
v_mul_f32 v20, v7, v16 // (op 3319) 
v_mul_f32 v21, v7, v17 // (op 3319) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 3307) 
v_mul_f32 v15, v0, v11 // (op 3307) 
v_mul_f32 v16, v0, v12 // (op 3307) 
v_mul_f32 v17, v0, v13 // (op 3307) 

v_add_f32 v10, v14, v18 // (op 3308) 
v_add_f32 v11, v15, v19 // (op 3308) 
v_add_f32 v12, v16, v20 // (op 3308) 
v_add_f32 v13, v17, v21 // (op 3308) 




s_lshl_b32 s0, s4, 6 // (op 3298) 
s_add_u32 s0, s0, 64 // (op 3298) 
v_add_u32 v3, v9, s0 // (op 3298) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 3298) Store value




s_lshl_b32 s0, s2, 7 // (op 3313) 
s_add_u32 s0, s0, 64 // (op 3313) 
v_add_u32 v3, v8, s0 // (op 3313) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 3313) Load value

v_accvgpr_read v14, a24 // (op 3321) 
v_accvgpr_read v15, a25 // (op 3321) 
v_accvgpr_read v16, a26 // (op 3321) 
v_accvgpr_read v17, a27 // (op 3321) 
v_mul_f32 v18, v7, v14 // (op 3321) 
v_mul_f32 v19, v7, v15 // (op 3321) 
v_mul_f32 v20, v7, v16 // (op 3321) 
v_mul_f32 v21, v7, v17 // (op 3321) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 3311) 
v_mul_f32 v15, v0, v11 // (op 3311) 
v_mul_f32 v16, v0, v12 // (op 3311) 
v_mul_f32 v17, v0, v13 // (op 3311) 

v_add_f32 v10, v14, v18 // (op 3312) 
v_add_f32 v11, v15, v19 // (op 3312) 
v_add_f32 v12, v16, v20 // (op 3312) 
v_add_f32 v13, v17, v21 // (op 3312) 




s_lshl_b32 s0, s4, 7 // (op 3300) 
s_add_u32 s0, s0, 64 // (op 3300) 
v_add_u32 v3, v9, s0 // (op 3300) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 3300) Store value




s_lshl_b32 s0, s2, 6 // (op 3317) 
s_mul_i32 s1, 3, s0 // (op 3317) 
s_add_u32 s1, s1, 64 // (op 3317) 

v_add_u32 v3, v8, s1 // (op 3317) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 3317) Load value

v_accvgpr_read v14, a28 // (op 3323) 
v_accvgpr_read v15, a29 // (op 3323) 
v_accvgpr_read v16, a30 // (op 3323) 
v_accvgpr_read v17, a31 // (op 3323) 
v_mul_f32 v18, v7, v14 // (op 3323) 
v_mul_f32 v19, v7, v15 // (op 3323) 
v_mul_f32 v20, v7, v16 // (op 3323) 
v_mul_f32 v21, v7, v17 // (op 3323) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 3315) 
v_mul_f32 v15, v0, v11 // (op 3315) 
v_mul_f32 v16, v0, v12 // (op 3315) 
v_mul_f32 v17, v0, v13 // (op 3315) 

v_add_f32 v10, v14, v18 // (op 3316) 
v_add_f32 v11, v15, v19 // (op 3316) 
v_add_f32 v12, v16, v20 // (op 3316) 
v_add_f32 v13, v17, v21 // (op 3316) 




s_lshl_b32 s0, s4, 6 // (op 3302) 
s_mul_i32 s1, 3, s0 // (op 3302) 
s_add_u32 s1, s1, 64 // (op 3302) 

v_add_u32 v3, v9, s1 // (op 3302) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 3302) Store value




v_add_u32 v3, 192, v8 // (op 3463) 

s_nop 0
buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 3463) Load value

v_accvgpr_read v14, a48 // (op 3451) 
v_accvgpr_read v15, a49 // (op 3451) 
v_accvgpr_read v16, a50 // (op 3451) 
v_accvgpr_read v17, a51 // (op 3451) 
v_mul_f32 v18, v7, v14 // (op 3451) 
v_mul_f32 v19, v7, v15 // (op 3451) 
v_mul_f32 v20, v7, v16 // (op 3451) 
v_mul_f32 v21, v7, v17 // (op 3451) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 3462) 
v_mul_f32 v15, v0, v11 // (op 3462) 
v_mul_f32 v16, v0, v12 // (op 3462) 
v_mul_f32 v17, v0, v13 // (op 3462) 

v_add_f32 v10, v14, v18 // (op 3452) 
v_add_f32 v11, v15, v19 // (op 3452) 
v_add_f32 v12, v16, v20 // (op 3452) 
v_add_f32 v13, v17, v21 // (op 3452) 




v_add_u32 v3, 192, v9 // (op 3454) 

buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 3454) Store value




s_lshl_b32 s0, s2, 6 // (op 3467) 
s_add_u32 s0, s0, 192 // (op 3467) 
v_add_u32 v3, v8, s0 // (op 3467) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 3467) Load value

v_accvgpr_read v14, a52 // (op 3477) 
v_accvgpr_read v15, a53 // (op 3477) 
v_accvgpr_read v16, a54 // (op 3477) 
v_accvgpr_read v17, a55 // (op 3477) 
v_mul_f32 v18, v7, v14 // (op 3477) 
v_mul_f32 v19, v7, v15 // (op 3477) 
v_mul_f32 v20, v7, v16 // (op 3477) 
v_mul_f32 v21, v7, v17 // (op 3477) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 3465) 
v_mul_f32 v15, v0, v11 // (op 3465) 
v_mul_f32 v16, v0, v12 // (op 3465) 
v_mul_f32 v17, v0, v13 // (op 3465) 

v_add_f32 v10, v14, v18 // (op 3466) 
v_add_f32 v11, v15, v19 // (op 3466) 
v_add_f32 v12, v16, v20 // (op 3466) 
v_add_f32 v13, v17, v21 // (op 3466) 




s_lshl_b32 s0, s4, 6 // (op 3456) 
s_add_u32 s0, s0, 192 // (op 3456) 
v_add_u32 v3, v9, s0 // (op 3456) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 3456) Store value




s_lshl_b32 s0, s2, 7 // (op 3471) 
s_add_u32 s0, s0, 192 // (op 3471) 
v_add_u32 v3, v8, s0 // (op 3471) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 3471) Load value

v_accvgpr_read v14, a56 // (op 3479) 
v_accvgpr_read v15, a57 // (op 3479) 
v_accvgpr_read v16, a58 // (op 3479) 
v_accvgpr_read v17, a59 // (op 3479) 
v_mul_f32 v18, v7, v14 // (op 3479) 
v_mul_f32 v19, v7, v15 // (op 3479) 
v_mul_f32 v20, v7, v16 // (op 3479) 
v_mul_f32 v21, v7, v17 // (op 3479) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 3469) 
v_mul_f32 v15, v0, v11 // (op 3469) 
v_mul_f32 v16, v0, v12 // (op 3469) 
v_mul_f32 v17, v0, v13 // (op 3469) 

v_add_f32 v10, v14, v18 // (op 3470) 
v_add_f32 v11, v15, v19 // (op 3470) 
v_add_f32 v12, v16, v20 // (op 3470) 
v_add_f32 v13, v17, v21 // (op 3470) 




s_lshl_b32 s0, s4, 7 // (op 3458) 
s_add_u32 s0, s0, 192 // (op 3458) 
v_add_u32 v3, v9, s0 // (op 3458) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 3458) Store value




s_lshl_b32 s0, s2, 6 // (op 3475) 
s_mul_i32 s1, 3, s0 // (op 3475) 
s_add_u32 s1, s1, 192 // (op 3475) 

v_add_u32 v3, v8, s1 // (op 3475) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 3475) Load value

v_accvgpr_read v14, a60 // (op 3481) 
v_accvgpr_read v15, a61 // (op 3481) 
v_accvgpr_read v16, a62 // (op 3481) 
v_accvgpr_read v17, a63 // (op 3481) 
v_mul_f32 v18, v7, v14 // (op 3481) 
v_mul_f32 v19, v7, v15 // (op 3481) 
v_mul_f32 v20, v7, v16 // (op 3481) 
v_mul_f32 v21, v7, v17 // (op 3481) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 3473) 
v_mul_f32 v15, v0, v11 // (op 3473) 
v_mul_f32 v16, v0, v12 // (op 3473) 
v_mul_f32 v17, v0, v13 // (op 3473) 

v_add_f32 v10, v14, v18 // (op 3474) 
v_add_f32 v11, v15, v19 // (op 3474) 
v_add_f32 v12, v16, v20 // (op 3474) 
v_add_f32 v13, v17, v21 // (op 3474) 




s_lshl_b32 s0, s4, 6 // (op 3460) 
s_mul_i32 s1, 3, s0 // (op 3460) 
s_add_u32 s1, s1, 192 // (op 3460) 

v_add_u32 v3, v9, s1 // (op 3460) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 3460) Store value




v_mov_b32 v3, v8 // (op 126) reg expression

s_nop 0
buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 126) Load value

v_accvgpr_read v14, a0 // (op 123) 
v_accvgpr_read v15, a1 // (op 123) 
v_accvgpr_read v16, a2 // (op 123) 
v_accvgpr_read v17, a3 // (op 123) 
v_mul_f32 v18, v7, v14 // (op 123) 
v_mul_f32 v19, v7, v15 // (op 123) 
v_mul_f32 v20, v7, v16 // (op 123) 
v_mul_f32 v21, v7, v17 // (op 123) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 127) 
v_mul_f32 v15, v0, v11 // (op 127) 
v_mul_f32 v16, v0, v12 // (op 127) 
v_mul_f32 v17, v0, v13 // (op 127) 

v_add_f32 v10, v14, v18 // (op 124) 
v_add_f32 v11, v15, v19 // (op 124) 
v_add_f32 v12, v16, v20 // (op 124) 
v_add_f32 v13, v17, v21 // (op 124) 




v_mov_b32 v3, v9 // (op 125) reg expression

buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 125) Store value




s_lshl_b32 s0, s2, 6 // (op 3219) 
v_add_u32 v3, v8, s0 // (op 3219) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 3219) Load value

v_accvgpr_read v14, a4 // (op 3216) 
v_accvgpr_read v15, a5 // (op 3216) 
v_accvgpr_read v16, a6 // (op 3216) 
v_accvgpr_read v17, a7 // (op 3216) 
v_mul_f32 v18, v7, v14 // (op 3216) 
v_mul_f32 v19, v7, v15 // (op 3216) 
v_mul_f32 v20, v7, v16 // (op 3216) 
v_mul_f32 v21, v7, v17 // (op 3216) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 3220) 
v_mul_f32 v15, v0, v11 // (op 3220) 
v_mul_f32 v16, v0, v12 // (op 3220) 
v_mul_f32 v17, v0, v13 // (op 3220) 

v_add_f32 v10, v14, v18 // (op 3217) 
v_add_f32 v11, v15, v19 // (op 3217) 
v_add_f32 v12, v16, v20 // (op 3217) 
v_add_f32 v13, v17, v21 // (op 3217) 




s_lshl_b32 s0, s4, 6 // (op 3218) 
v_add_u32 v3, v9, s0 // (op 3218) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 3218) Store value




s_lshl_b32 s0, s2, 7 // (op 3231) 
v_add_u32 v3, v8, s0 // (op 3231) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 3231) Load value

v_accvgpr_read v14, a8 // (op 3228) 
v_accvgpr_read v15, a9 // (op 3228) 
v_accvgpr_read v16, a10 // (op 3228) 
v_accvgpr_read v17, a11 // (op 3228) 
v_mul_f32 v18, v7, v14 // (op 3228) 
v_mul_f32 v19, v7, v15 // (op 3228) 
v_mul_f32 v20, v7, v16 // (op 3228) 
v_mul_f32 v21, v7, v17 // (op 3228) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 3232) 
v_mul_f32 v15, v0, v11 // (op 3232) 
v_mul_f32 v16, v0, v12 // (op 3232) 
v_mul_f32 v17, v0, v13 // (op 3232) 

v_add_f32 v10, v14, v18 // (op 3229) 
v_add_f32 v11, v15, v19 // (op 3229) 
v_add_f32 v12, v16, v20 // (op 3229) 
v_add_f32 v13, v17, v21 // (op 3229) 




s_lshl_b32 s0, s4, 7 // (op 3230) 
v_add_u32 v3, v9, s0 // (op 3230) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 3230) Store value




s_lshl_b32 s0, s2, 6 // (op 3243) 
s_mul_i32 s1, 3, s0 // (op 3243) 

v_add_u32 v3, v8, s1 // (op 3243) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 3243) Load value

v_accvgpr_read v14, a12 // (op 3240) 
v_accvgpr_read v15, a13 // (op 3240) 
v_accvgpr_read v16, a14 // (op 3240) 
v_accvgpr_read v17, a15 // (op 3240) 
v_mul_f32 v18, v7, v14 // (op 3240) 
v_mul_f32 v19, v7, v15 // (op 3240) 
v_mul_f32 v20, v7, v16 // (op 3240) 
v_mul_f32 v21, v7, v17 // (op 3240) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 3244) 
v_mul_f32 v15, v0, v11 // (op 3244) 
v_mul_f32 v16, v0, v12 // (op 3244) 
v_mul_f32 v17, v0, v13 // (op 3244) 

v_add_f32 v10, v14, v18 // (op 3241) 
v_add_f32 v11, v15, v19 // (op 3241) 
v_add_f32 v12, v16, v20 // (op 3241) 
v_add_f32 v13, v17, v21 // (op 3241) 




s_lshl_b32 s0, s4, 6 // (op 3242) 
s_mul_i32 s1, 3, s0 // (op 3242) 

v_add_u32 v3, v9, s1 // (op 3242) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 3242) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ConditionalBottom_0_beta_114:
























s_endpgm  // End of RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2
.LRR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_end:
.size RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2, .LRR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_end-RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2
  .amdhsa_next_free_vgpr 212
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 65536 // lds bytes
  .amdhsa_accum_offset 148
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
    macrotile: [128, 128, 128]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [4, 4]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2
    .symbol: RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 65536
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 64
    .vgpr_count: 146
    .agpr_count: 64
    .max_flat_workgroup_size: 256
    .workgroup_size: [256, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, {is-null: true}]
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

