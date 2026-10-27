// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2
.p2align 8
.type RR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2,@function
RR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 6969) 
v_and_b32 v3, 15, v1 // (op 6969) 
v_lshl_add_u32 v4, v0, 6, v3 // (op 6969) 


v_and_b32 v0, 63, v1 // (op 6969) 
v_lshrrev_b32 v3, 4, v0 // (op 6969) 

v_lshlrev_b32 v0, 5, v3 // (op 6969) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 6969) 


v_lshlrev_b32 v0, 2, v3 // (op 6969) 

v_lshrrev_b32 v3, 3, v0 // (op 6969) 

v_lshrrev_b32 v0, 6, v1 // (op 7027) 
v_and_b32 v4, 1, v0 // (op 7027) 

v_and_b32 v0, 15, v1 // (op 7027) 
v_lshl_add_u32 v5, v4, 6, v0 // (op 7027) 


v_and_b32 v0, 63, v1 // (op 7027) 
v_lshrrev_b32 v4, 4, v0 // (op 7027) 

v_lshlrev_b32 v0, 5, v4 // (op 7027) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 7027) 


v_mul_lo_u32 v0, v4, 6 // (op 7027) 

v_lshrrev_b32 v4, 3, v0 // (op 7027) 

v_lshrrev_b32 v0, 7, v1 // (op 7094) 
v_and_b32 v5, 15, v1 // (op 7094) 
v_lshl_add_u32 v6, v0, 6, v5 // (op 7094) 


v_and_b32 v0, 63, v1 // (op 7094) 
v_lshrrev_b32 v5, 4, v0 // (op 7094) 

v_lshlrev_b32 v0, 5, v5 // (op 7094) 

v_lshl_add_u32 v5, v6, 7, v0 // (op 7094) 


v_lshlrev_b32 v0, 2, v5 // (op 7094) 

v_lshrrev_b32 v5, 3, v0 // (op 7094) 

v_lshrrev_b32 v0, 6, v1 // (op 7123) 
v_and_b32 v6, 1, v0 // (op 7123) 

v_and_b32 v0, 15, v1 // (op 7123) 
v_lshl_add_u32 v7, v6, 6, v0 // (op 7123) 


v_and_b32 v0, 63, v1 // (op 7123) 
v_lshrrev_b32 v6, 4, v0 // (op 7123) 

v_lshlrev_b32 v0, 5, v6 // (op 7123) 

v_lshl_add_u32 v6, v7, 7, v0 // (op 7123) 


v_mul_lo_u32 v0, v6, 6 // (op 7123) 

v_lshrrev_b32 v6, 3, v0 // (op 7123) 

v_lshrrev_b32 v0, 7, v1 // (op 6908) 
v_and_b32 v7, 63, v1 // (op 6908) 
v_lshrrev_b32 v8, 4, v7 // (op 6908) 
v_lshlrev_b32 v9, 2, v8 // (op 6908) 

v_lshl_add_u32 v8, v0, 6, v9 // (op 6908) 


v_lshl_add_u32 v0, s16, 7, v8 // (op 6908) 

v_lshrrev_b32 v8, 2, v7 // (op 6908) 

v_and_b32 v7, 3, v8 // (op 6908) 

v_and_b32 v8, 3, v1 // (op 6908) 
v_lshl_add_u32 v9, v7, 2, v8 // (op 6908) 


v_lshrrev_b32 v7, 6, v1 // (op 6908) 
v_and_b32 v8, 1, v7 // (op 6908) 

v_lshl_add_u32 v7, v8, 6, v9 // (op 6908) 


v_lshl_add_u32 v8, s17, 7, v7 // (op 6908) 

v_mul_lo_u32 v7, v8, s2 // (op 6908) 

v_add_lshl_u32 v8, v0, v7, 2 // (op 6908) 


v_lshrrev_b32 v0, 7, v1 // (op 7048) 
v_and_b32 v7, 63, v1 // (op 7048) 
v_lshrrev_b32 v9, 4, v7 // (op 7048) 
s_mul_i32 s1, s2, s9 // (op 6910) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 6910) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 6910) least significant
s_mul_hi_u32 s19, s2, s8 // (op 6910) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 6910) most significant: sum
s_add_u32 s1, s1, s19 // (op 6910) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 6910) 

v_lshlrev_b32 v10, 2, v9 // (op 7048) 

v_lshl_add_u32 v9, v0, 6, v10 // (op 7048) 


v_lshl_add_u32 v0, s16, 7, v9 // (op 7048) 

v_lshrrev_b32 v9, 2, v7 // (op 7048) 

v_and_b32 v7, 3, v9 // (op 7048) 

v_and_b32 v9, 3, v1 // (op 7048) 
v_lshl_add_u32 v10, v7, 2, v9 // (op 7048) 


v_lshrrev_b32 v7, 6, v1 // (op 7048) 
v_and_b32 v9, 1, v7 // (op 7048) 

v_lshl_add_u32 v7, v9, 6, v10 // (op 7048) 


v_lshl_add_u32 v9, s17, 7, v7 // (op 7048) 

v_mul_lo_u32 v7, v9, s4 // (op 7048) 

v_add_lshl_u32 v9, v0, v7, 2 // (op 7048) 


s_mov_b32 s40, s6 // (op 6910) reg expression
s_mov_b32 s41, s7 // (op 6910) reg expression
s_mov_b32 s42, s18 // (op 6910) reg expression
s_mov_b32 s43, 131072 // (op 6910) call()

s_mul_i32 s1, s4, s9 // (op 7050) most significant: low of low * high

s_mul_i32 s6, s5, s8 // (op 7050) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 7050) least significant
s_mul_hi_u32 s7, s4, s8 // (op 7050) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 7050) most significant: sum
s_add_u32 s1, s1, s7 // (op 7050) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 7050) 

s_mov_b32 s44, s10 // (op 7050) reg expression
s_mov_b32 s45, s11 // (op 7050) reg expression
s_mov_b32 s46, s6 // (op 7050) reg expression
s_mov_b32 s47, 131072 // (op 7050) call()


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
v_lshrrev_b32 v10, 2, v1 // (op 6798) 
v_lshl_add_u32 v11, s16, 7, v10 // (op 6798) 

v_and_b32 v10, -4, v1 // (op 6798) 
v_sub_u32 v12, v1, v10 // (op 6798) 

v_lshlrev_b32 v10, 5, v12 // (op 6798) 

v_mov_b32 v13, 0 // (op 6798) convert
v_mov_b32 v12, v10 // (op 6798) convert

v_mul_lo_u32 v15, v11, s13 // (op 6798) most significant: low of low * high
v_mul_hi_u32 v10, v11, s12 // (op 6798) most significant: high of low * low
v_mul_lo_u32 v14, v11, s12 // (op 6798) least significant: low of low * low
v_add_u32 v15, v15, v10 // (op 6798) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v12 // (op 6798) least significant half
v_addc_co_u32 v15, s[0:1], v15, v13, s[0:1] // (op 6798) most significant half



v_lshlrev_b64 v[10:11], 2, v[14:15] // (op 6798) 

v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 6798) 


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 6800) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 6800) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 6800) least significant
s_mul_hi_u32 s7, s12, s22 // (op 6800) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6800) most significant: sum
s_add_u32 s1, s1, s7 // (op 6800) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 6800) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 6800) 

s_mov_b32 s48, s20 // (op 6800) reg expression
s_mov_b32 s49, s21 // (op 6800) reg expression
s_mov_b32 s50, s0 // (op 6800) reg expression
s_mov_b32 s51, 131072 // (op 6800) call()

v_lshrrev_b32 v10, 2, v1 // (op 6817) 
v_lshl_add_u32 v11, s16, 7, v10 // (op 6817) 

v_and_b32 v10, -4, v1 // (op 6817) 
v_sub_u32 v14, v1, v10 // (op 6817) 

v_lshlrev_b32 v10, 5, v14 // (op 6817) 

v_mov_b32 v15, 0 // (op 6817) convert
v_mov_b32 v14, v10 // (op 6817) convert

v_mul_lo_u32 v17, v11, s13 // (op 6817) most significant: low of low * high
v_mul_hi_u32 v10, v11, s12 // (op 6817) most significant: high of low * low
v_mul_lo_u32 v16, v11, s12 // (op 6817) least significant: low of low * low
v_add_u32 v17, v17, v10 // (op 6817) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 6817) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 6817) most significant half



v_lshlrev_b64 v[10:11], 2, v[16:17] // (op 6817) 

v_lshrrev_b64 v[14:15], 3, v[10:11] // (op 6817) 

s_mul_i32 s1, s12, s23 // (op 6819) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 6819) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 6819) least significant
s_mul_hi_u32 s7, s12, s22 // (op 6819) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6819) most significant: sum
s_add_u32 s1, s1, s7 // (op 6819) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 6819) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 6819) 

s_mov_b32 s48, s20 // (op 6819) reg expression
s_mov_b32 s49, s21 // (op 6819) reg expression
s_mov_b32 s50, s0 // (op 6819) reg expression
s_mov_b32 s51, 131072 // (op 6819) call()

v_and_b32 v10, -2, v1 // (op 6862) 
v_sub_u32 v11, v1, v10 // (op 6862) 

v_lshlrev_b32 v10, 6, v11 // (op 6862) 

v_mov_b32 v17, 0 // (op 6862) convert
v_mov_b32 v16, v10 // (op 6862) convert

v_lshrrev_b32 v10, 1, v1 // (op 6862) 
v_lshl_add_u32 v11, s17, 7, v10 // (op 6862) 

v_mul_lo_u32 v19, v11, s25 // (op 6862) most significant: low of low * high
v_mul_hi_u32 v10, v11, s24 // (op 6862) most significant: high of low * low
v_mul_lo_u32 v18, v11, s24 // (op 6862) least significant: low of low * low
v_add_u32 v19, v19, v10 // (op 6862) most significant: sum


v_add_co_u32 v10, s[0:1], v16, v18 // (op 6862) least significant half
v_addc_co_u32 v11, s[0:1], v17, v19, s[0:1] // (op 6862) most significant half



v_mul_lo_u32 v17, v11, 6 // (op 6862) most significant: low of high * low
v_mul_hi_u32 v18, v10, 6 // (op 6862) most significant: high of low * low
v_mul_lo_u32 v16, v10, 6 // (op 6862) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 6862) most significant: sum


v_lshrrev_b64 v[10:11], 3, v[16:17] // (op 6862) 

v_mov_b32 v16, v10 // (op 6862) convert

s_mul_i32 s1, s9, 6 // (op 6864) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 6864) least significant
s_mul_hi_u32 s6, s8, 6 // (op 6864) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6864) most significant: sum

s_mul_i32 s7, s24, s1 // (op 6864) most significant: low of low * high
s_mul_i32 s10, s25, s0 // (op 6864) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 6864) least significant
s_mul_hi_u32 s11, s24, s0 // (op 6864) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 6864) most significant: sum
s_add_u32 s7, s7, s11 // (op 6864) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 6864) 

s_mov_b32 s52, s26 // (op 6864) reg expression
s_mov_b32 s53, s27 // (op 6864) reg expression
s_mov_b32 s54, s0 // (op 6864) reg expression
s_mov_b32 s55, 131072 // (op 6864) call()

v_and_b32 v10, -2, v1 // (op 6881) 
v_sub_u32 v11, v1, v10 // (op 6881) 

v_lshlrev_b32 v10, 6, v11 // (op 6881) 

v_mov_b32 v19, 0 // (op 6881) convert
v_mov_b32 v18, v10 // (op 6881) convert

v_lshrrev_b32 v17, 1, v1 // (op 6881) 
v_lshl_add_u32 v10, s17, 7, v17 // (op 6881) 

v_mul_lo_u32 v21, v10, s25 // (op 6881) most significant: low of low * high
v_mul_hi_u32 v11, v10, s24 // (op 6881) most significant: high of low * low
v_mul_lo_u32 v20, v10, s24 // (op 6881) least significant: low of low * low
v_add_u32 v21, v21, v11 // (op 6881) most significant: sum


v_add_co_u32 v10, s[0:1], v18, v20 // (op 6881) least significant half
v_addc_co_u32 v11, s[0:1], v19, v21, s[0:1] // (op 6881) most significant half



v_mul_lo_u32 v19, v11, 6 // (op 6881) most significant: low of high * low
v_mul_hi_u32 v17, v10, 6 // (op 6881) most significant: high of low * low
v_mul_lo_u32 v18, v10, 6 // (op 6881) least significant: low of low * low
v_add_u32 v19, v19, v17 // (op 6881) most significant: sum


v_lshrrev_b64 v[10:11], 3, v[18:19] // (op 6881) 

v_mov_b32 v17, v10 // (op 6881) convert

s_mul_i32 s1, s9, 6 // (op 6883) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 6883) least significant
s_mul_hi_u32 s6, s8, 6 // (op 6883) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6883) most significant: sum

s_mul_i32 s7, s24, s1 // (op 6883) most significant: low of low * high
s_mul_i32 s10, s25, s0 // (op 6883) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 6883) least significant
s_mul_hi_u32 s11, s24, s0 // (op 6883) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 6883) most significant: sum
s_add_u32 s7, s7, s11 // (op 6883) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 6883) 

s_mov_b32 s52, s26 // (op 6883) reg expression
s_mov_b32 s53, s27 // (op 6883) reg expression
s_mov_b32 s54, s0 // (op 6883) reg expression
s_mov_b32 s55, 131072 // (op 6883) call()

v_and_b32 v10, 63, v1 // (op 7145) 
v_lshrrev_b32 v11, 4, v10 // (op 7145) 
v_and_b32 v18, 3, v11 // (op 7145) 

v_and_b32 v11, 15, v1 // (op 7145) 
v_lshl_add_u32 v19, v18, 4, v11 // (op 7145) 


v_lshrrev_b32 v11, 7, v1 // (op 7145) 
v_lshl_add_u32 v18, v11, 6, v19 // (op 7145) 


v_lshl_add_u32 v11, s16, 7, v18 // (op 7145) 

v_lshrrev_b32 v18, 6, v10 // (op 7145) 

v_lshlrev_b32 v10, 2, v18 // (op 7145) 

v_mul_lo_u32 v18, v11, s28 // (op 7145) 
v_add_u32 v18, v18, v10 // (op 7145) 


s_mul_i32 s1, s28, s23 // (op 7147) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 7147) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 7147) least significant
s_mul_hi_u32 s7, s28, s22 // (op 7147) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 7147) most significant: sum
s_add_u32 s1, s1, s7 // (op 7147) most significant: sum


s_mov_b32 s56, s30 // (op 7147) reg expression
s_mov_b32 s57, s31 // (op 7147) reg expression
s_mov_b32 s58, s0 // (op 7147) reg expression
s_mov_b32 s59, 131072 // (op 7147) call()

v_and_b32 v10, 63, v1 // (op 7166) 
v_lshrrev_b32 v11, 4, v10 // (op 7166) 
v_and_b32 v19, 3, v11 // (op 7166) 

v_and_b32 v11, 15, v1 // (op 7166) 
v_lshl_add_u32 v20, v19, 4, v11 // (op 7166) 


v_lshrrev_b32 v11, 7, v1 // (op 7166) 
v_lshl_add_u32 v19, v11, 6, v20 // (op 7166) 


v_lshl_add_u32 v11, s16, 7, v19 // (op 7166) 

v_lshrrev_b32 v19, 6, v10 // (op 7166) 

v_lshlrev_b32 v10, 2, v19 // (op 7166) 

v_mul_lo_u32 v19, v11, s28 // (op 7166) 
v_add_u32 v19, v19, v10 // (op 7166) 


s_mul_i32 s1, s28, s23 // (op 7168) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 7168) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 7168) least significant
s_mul_hi_u32 s7, s28, s22 // (op 7168) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 7168) most significant: sum
s_add_u32 s1, s1, s7 // (op 7168) most significant: sum


s_mov_b32 s56, s30 // (op 7168) reg expression
s_mov_b32 s57, s31 // (op 7168) reg expression
s_mov_b32 s58, s0 // (op 7168) reg expression
s_mov_b32 s59, 131072 // (op 7168) call()

v_and_b32 v10, 63, v1 // (op 7197) 
v_lshrrev_b32 v11, 4, v10 // (op 7197) 
v_and_b32 v20, 3, v11 // (op 7197) 

v_and_b32 v11, 15, v1 // (op 7197) 
v_lshl_add_u32 v21, v20, 4, v11 // (op 7197) 


v_lshrrev_b32 v11, 6, v1 // (op 7197) 
v_and_b32 v20, 1, v11 // (op 7197) 

v_lshl_add_u32 v11, v20, 6, v21 // (op 7197) 


v_lshl_add_u32 v20, s17, 7, v11 // (op 7197) 

v_mul_lo_u32 v11, v20, s32 // (op 7197) 

v_lshrrev_b32 v20, 6, v10 // (op 7197) 

v_lshl_add_u32 v10, v20, 2, v11 // (op 7197) 


s_mul_i32 s1, s32, s9 // (op 7199) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 7199) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 7199) least significant
s_mul_hi_u32 s7, s32, s8 // (op 7199) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 7199) most significant: sum
s_add_u32 s1, s1, s7 // (op 7199) most significant: sum


s_mov_b32 s60, s34 // (op 7199) reg expression
s_mov_b32 s61, s35 // (op 7199) reg expression
s_mov_b32 s62, s0 // (op 7199) reg expression
s_mov_b32 s63, 131072 // (op 7199) call()

v_and_b32 v11, 63, v1 // (op 7218) 
v_lshrrev_b32 v20, 4, v11 // (op 7218) 
v_and_b32 v21, 3, v20 // (op 7218) 

v_and_b32 v20, 15, v1 // (op 7218) 
v_lshl_add_u32 v22, v21, 4, v20 // (op 7218) 


v_lshrrev_b32 v20, 6, v1 // (op 7218) 
v_and_b32 v21, 1, v20 // (op 7218) 

v_lshl_add_u32 v20, v21, 6, v22 // (op 7218) 


v_lshl_add_u32 v21, s17, 7, v20 // (op 7218) 

v_mul_lo_u32 v20, v21, s32 // (op 7218) 

v_lshrrev_b32 v21, 6, v11 // (op 7218) 

v_lshl_add_u32 v11, v21, 2, v20 // (op 7218) 


s_mul_i32 s1, s32, s9 // (op 7220) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 7220) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 7220) least significant
s_mul_hi_u32 s7, s32, s8 // (op 7220) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 7220) most significant: sum
s_add_u32 s1, s1, s7 // (op 7220) most significant: sum


s_mov_b32 s60, s34 // (op 7220) reg expression
s_mov_b32 s61, s35 // (op 7220) reg expression
s_mov_b32 s62, s0 // (op 7220) reg expression
s_mov_b32 s63, 131072 // (op 7220) call()




buffer_load_dwordx4 v[20:23], v16, s[52:55], 0 offen // (op 6413) Load value
buffer_load_dwordx2 v[24:25], v16, s[52:55], 0 offen offset:16 // (op 6413) Load value
buffer_load_dwordx4 v[26:29], v16, s[52:55], 0 offen offset:24 // (op 6413) Load value
buffer_load_dwordx2 v[30:31], v16, s[52:55], 0 offen offset:40 // (op 6413) Load value
v_and_b32 v32, -4, v1 // (op 6952) 
v_sub_u32 v33, v1, v32 // (op 6952) 

v_lshlrev_b32 v32, 5, v33 // (op 6952) 

v_lshrrev_b32 v33, 2, v1 // (op 6952) 
v_lshl_add_u32 v34, v33, 7, v32 // (op 6952) 


v_lshlrev_b32 v32, 2, v34 // (op 6952) 

v_lshrrev_b32 v33, 3, v32 // (op 6952) 

s_nop 0
v_readfirstlane_b32 s0, v33 // (op 6952) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 6 // (op 6739) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 6739) 

v_lshrrev_b64 v[32:33], 3, s[10:11] // (op 6739) 


v_mov_b64 v[34:35], v[12:13] // (op 6739) 
s_mov_b32 m0, s0 // (op 6739) reg expression
buffer_load_dwordx4 v34, s[48:51], 0 offen offset: 0 lds // (op 6739) Load value direct to lds
v_add_co_u32 v34, s[6:7], v34, v32 // (op 6739) least significant half
v_addc_co_u32 v35, s[6:7], 0, v35, s[6:7] // (op 6739) most significant half

s_add_u32 m0, m0, 4096 // (op 6739) 
buffer_load_dwordx4 v34, s[48:51], 0 offen offset: 0 lds // (op 6739) Load value direct to lds





v_mov_b32 v32, v10 // (op 74) reg expression

buffer_load_dwordx2 v[34:35], v32, s[60:63], 0 offen // (op 74) Load value




v_mov_b32 v10, v18 // (op 70) reg expression

buffer_load_dwordx2 v[32:33], v10, s[56:59], 0 offen // (op 70) Load value





buffer_load_dwordx4 v[36:39], v16, s[52:55], 0 offen offset:96 // (op 6419) Load value
buffer_load_dwordx2 v[40:41], v16, s[52:55], 0 offen offset:112 // (op 6419) Load value
buffer_load_dwordx4 v[42:45], v16, s[52:55], 0 offen offset:120 // (op 6419) Load value
buffer_load_dwordx2 v[46:47], v16, s[52:55], 0 offen offset:136 // (op 6419) Load value

v_and_b32 v10, -4, v1 // (op 7077) 
v_sub_u32 v16, v1, v10 // (op 7077) 

v_lshlrev_b32 v10, 5, v16 // (op 7077) 

v_lshrrev_b32 v16, 2, v1 // (op 7077) 
v_lshl_add_u32 v18, v16, 7, v10 // (op 7077) 


v_lshlrev_b32 v10, 2, v18 // (op 7077) 

v_lshrrev_b32 v16, 3, v10 // (op 7077) 

s_nop 0
v_readfirstlane_b32 s0, v16 // (op 7077) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 6 // (op 6744) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 6744) 

v_lshrrev_b64 v[48:49], 3, s[10:11] // (op 6744) 


v_mov_b64 v[50:51], v[12:13] // (op 6744) 
s_add_u32 s1, 8192, s0 // (op 6744) 
s_mov_b32 m0, s1 // (op 6744) reg expression
buffer_load_dwordx4 v50, s[48:51], 64 offen offset: 0 lds // (op 6744) Load value direct to lds
v_add_co_u32 v50, s[6:7], v50, v48 // (op 6744) least significant half
v_addc_co_u32 v51, s[6:7], 0, v51, s[6:7] // (op 6744) most significant half

s_add_u32 m0, m0, 4096 // (op 6744) 
buffer_load_dwordx4 v50, s[48:51], 64 offen offset: 0 lds // (op 6744) Load value direct to lds





v_and_b32 v10, -2, v1 // (op 7010) 
v_sub_u32 v16, v1, v10 // (op 7010) 

v_lshlrev_b32 v10, 6, v16 // (op 7010) 

v_lshrrev_b32 v16, 1, v1 // (op 7010) 
v_lshl_add_u32 v18, v16, 7, v10 // (op 7010) 


v_mul_lo_u32 v10, v18, 6 // (op 7010) 

v_lshrrev_b32 v16, 3, v10 // (op 7010) 




s_waitcnt vmcnt(13)
ds_write_b128 v16, v[20:23] offset:16384 // (op 6424) Store local data 
s_waitcnt vmcnt(12)
ds_write_b64 v16, v[24:25] offset:16400 // (op 6424) Store local data 
s_waitcnt vmcnt(11)
ds_write_b128 v16, v[26:29] offset:16408 // (op 6424) Store local data 
s_waitcnt vmcnt(10)
ds_write_b64 v16, v[30:31] offset:16424 // (op 6424) Store local data 

v_and_b32 v10, -4, v1 // (op 6937) 
v_sub_u32 v16, v1, v10 // (op 6937) 

v_lshlrev_b32 v10, 5, v16 // (op 6937) 

v_lshrrev_b32 v16, 2, v1 // (op 6937) 
v_lshl_add_u32 v18, v16, 7, v10 // (op 6937) 


v_lshlrev_b32 v10, 2, v18 // (op 6937) 

v_lshrrev_b32 v16, 3, v10 // (op 6937) 

s_nop 0
v_readfirstlane_b32 s0, v16 // (op 6937) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v10, -2, v1 // (op 6994) 
v_sub_u32 v16, v1, v10 // (op 6994) 

v_lshlrev_b32 v10, 6, v16 // (op 6994) 

v_lshrrev_b32 v16, 1, v1 // (op 6994) 
v_lshl_add_u32 v18, v16, 7, v10 // (op 6994) 


v_mul_lo_u32 v10, v18, 6 // (op 6994) 

v_lshrrev_b32 v16, 3, v10 // (op 6994) 

v_and_b32 v10, -4, v1 // (op 7062) 
v_sub_u32 v18, v1, v10 // (op 7062) 

v_lshlrev_b32 v10, 5, v18 // (op 7062) 

v_lshrrev_b32 v18, 2, v1 // (op 7062) 
v_lshl_add_u32 v12, v18, 7, v10 // (op 7062) 


v_lshlrev_b32 v10, 2, v12 // (op 7062) 

v_lshrrev_b32 v18, 3, v10 // (op 7062) 

s_nop 0
v_readfirstlane_b32 s1, v18 // (op 7062) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v10, -2, v1 // (op 7106) 
v_sub_u32 v18, v1, v10 // (op 7106) 

v_lshlrev_b32 v10, 6, v18 // (op 7106) 

v_lshrrev_b32 v18, 1, v1 // (op 7106) 
v_lshl_add_u32 v12, v18, 7, v10 // (op 7106) 


v_mul_lo_u32 v10, v12, 6 // (op 7106) 

v_lshrrev_b32 v18, 3, v10 // (op 7106) 

s_mov_b32 s6, 0 // (op 1466) call()
s_ashr_i64 s[10:11], s[36:37], 7 // (op 1465) 
s_mov_b32 s7, s10 // (op 1465) convert

s_and_b32 s10, s7, 4294967294 // (op 1465) 

s_cmp_lt_u32 s6, s10 // (op 1465) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopBottom_KLoop_1465 // (op 1465) Condition: Top (jump to RR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopBottom_KLoop_1465 if false)
RR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopTop_KLoop_1465:
s_add_u32 s6, s6, 2 // (op 1467) 
s_waitcnt vmcnt(8)
s_barrier  // (op 7362) 



ds_read_b128 v[48:51], v3 // (op 60) Load local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 7378) 



ds_read_b128 v[52:55], v4 offset:16384 // (op 65) Load local data 
ds_read_b64 v[56:57], v4 offset:16400 // (op 65) Load local data 



ds_read_b128 v[58:61], v4 offset:17920 // (op 308) Load local data 


v_mov_b32 v10, v17 // (op 10) 

buffer_load_dwordx4 v[20:23], v10, s[52:55], 0 offen offset:192 // (op 10) Load value
s_waitcnt vmcnt(8)
v_mov_b32 v12, v34 // (op 6679) reg expression
v_mov_b32 v13, v35 // (op 6679) reg expression
s_waitcnt vmcnt(7)
v_mov_b32 v64, v32 // (op 6692) reg expression
v_mov_b32 v65, v33 // (op 6692) reg expression
ds_read_b64 v[62:63], v4 offset:17936 // (op 308) Load local data 



v_bfe_u32 v66, v64, 0, 8 // (op 6057) 
v_bfe_u32 v67, v64, 8, 8 // (op 6057) 
v_bfe_u32 v68, v64, 16, 8 // (op 6057) 
v_bfe_u32 v69, v64, 24, 8 // (op 6057) 
v_permlane16_swap_b32 v66, v67 // (op 6057) 
ds_read_b128 v[70:73], v4 offset:19456 // (op 444) Load local data 
v_permlane16_swap_b32 v68, v69 // (op 6057) 
s_nop 1
v_permlane32_swap_b32 v66, v68 // (op 6057) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v67, v69 // (op 6057) 
ds_read_b64 v[74:75], v4 offset:19472 // (op 444) Load local data 



v_bfe_u32 v76, v12, 0, 8 // (op 6120) 
v_bfe_u32 v77, v12, 8, 8 // (op 6120) 
v_bfe_u32 v78, v12, 16, 8 // (op 6120) 
v_bfe_u32 v79, v12, 24, 8 // (op 6120) 
v_permlane16_swap_b32 v76, v77 // (op 6120) 
s_nop 0
v_permlane16_swap_b32 v78, v79 // (op 6120) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v76, v78 // (op 6120) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v77, v79 // (op 6120) 
ds_read_b128 v[80:83], v4 offset:20992 // (op 580) Load local data 
ds_read_b64 v[84:85], v4 offset:21008 // (op 580) Load local data 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[48:51], v[52:57], a[0:3], v66, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 76) 
s_barrier  // (op 7371) 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[48:51], v[58:63], a[4:7], v66, v77 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 311) 
ds_read_b128 v[86:89], v3 offset:1024 // (op 885) Load local data 



s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[48:51], v[70:75], a[8:11], v66, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 447) 
ds_read_b128 v[90:93], v3 offset:2048 // (op 1495) Load local data 



s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[48:51], v[80:85], a[12:15], v66, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 583) 
ds_read_b128 v[94:97], v3 offset:3072 // (op 2105) Load local data 

buffer_load_dwordx2 v[24:25], v10, s[52:55], 0 offen offset:208 // (op 10) Load value
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[86:89], v[58:63], a[20:23], v67, v77 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 965) 
buffer_load_dwordx4 v[26:29], v10, s[52:55], 0 offen offset:216 // (op 10) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[86:89], v[70:75], a[24:27], v67, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 927) 
buffer_load_dwordx2 v[30:31], v10, s[52:55], 0 offen offset:232 // (op 10) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[86:89], v[80:85], a[28:31], v67, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 889) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 7365) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[86:89], v[52:57], a[16:19], v67, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 1003) 
s_lshl_b64 s[10:11], s[12:13], 6 // (op 6723) 

s_lshl_b64 s[14:15], s[10:11], 2 // (op 6723) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[90:93], v[52:57], a[32:35], v68, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 1613) 
v_lshrrev_b64 v[48:49], 3, s[14:15] // (op 6723) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[90:93], v[58:63], a[36:39], v68, v77 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 1575) 
v_mov_b64 v[50:51], v[14:15] // (op 6723) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[90:93], v[70:75], a[40:43], v68, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 1537) 
v_mov_b64 v[86:87], v[50:51] // (op 6723) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[90:93], v[80:85], a[44:47], v68, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 1499) 
s_mov_b32 m0, s0 // (op 6723) reg expression

s_mov_b32 s7, 128 // (op 6723) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[94:97], v[52:57], a[48:51], v69, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 2223) 
buffer_load_dwordx4 v86, s[48:51], s7 offen offset: 0 lds // (op 6723) Load value direct to lds

v_add_co_u32 v86, s[10:11], v86, v48 // (op 6723) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[94:97], v[58:63], a[52:55], v69, v77 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 2185) 
v_addc_co_u32 v87, s[10:11], 0, v87, s[10:11] // (op 6723) most significant half


v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[94:97], v[70:75], a[56:59], v69, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 2147) 
s_add_u32 m0, m0, 4096 // (op 6723) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[94:97], v[80:85], a[60:63], v69, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 2109) 
buffer_load_dwordx4 v86, s[48:51], s7 offen offset: 0 lds // (op 6723) Load value direct to lds









s_barrier  // (op 7404) 
v_mov_b32 v48, v11 // (op 6687) 



v_add_u32 v49, 8, v48 // (op 6687) 

s_waitcnt vmcnt(11)
ds_write_b128 v18, v[36:39] offset:28672 // (op 218) Store local data 
s_waitcnt vmcnt(10)
ds_write_b64 v18, v[40:41] offset:28688 // (op 218) Store local data 
s_waitcnt vmcnt(9)
ds_write_b128 v18, v[42:45] offset:28696 // (op 218) Store local data 
s_waitcnt vmcnt(8)
ds_write_b64 v18, v[46:47] offset:28712 // (op 218) Store local data 
buffer_load_dwordx2 v[34:35], v49, s[60:63], 0 offen // (op 6687) Load value


v_add_u32 v11, 8, v11 // (op 7208) 


v_mov_b32 v48, v19 // (op 6700) 
v_add_u32 v49, 8, v48 // (op 6700) 

s_waitcnt vmcnt(7)
s_barrier  // (op 7392) 
buffer_load_dwordx2 v[32:33], v49, s[56:59], 0 offen // (op 6700) Load value





v_add_u32 v19, 8, v19 // (op 7156) 
ds_read_b128 v[52:55], v5 offset:8192 // (op 210) Load local data 



buffer_load_dwordx4 v[36:39], v10, s[52:55], 0 offen offset:288 // (op 219) Load value
s_waitcnt lgkmcnt(1)
s_barrier  // (op 7401) 
buffer_load_dwordx2 v[40:41], v10, s[52:55], 0 offen offset:304 // (op 219) Load value



ds_read_b128 v[56:59], v6 offset:28672 // (op 212) Load local data 
ds_read_b64 v[60:61], v6 offset:28688 // (op 212) Load local data 
buffer_load_dwordx4 v[42:45], v10, s[52:55], 0 offen offset:312 // (op 219) Load value



v_bfe_u32 v66, v65, 0, 8 // (op 6072) 
v_bfe_u32 v67, v65, 8, 8 // (op 6072) 
v_bfe_u32 v68, v65, 16, 8 // (op 6072) 
v_bfe_u32 v69, v65, 24, 8 // (op 6072) 
v_permlane16_swap_b32 v66, v67 // (op 6072) 
ds_read_b128 v[70:73], v6 offset:30208 // (op 320) Load local data 
ds_read_b64 v[74:75], v6 offset:30224 // (op 320) Load local data 
v_permlane16_swap_b32 v68, v69 // (op 6072) 



ds_read_b128 v[76:79], v6 offset:31744 // (op 456) Load local data 
s_nop 0
v_permlane32_swap_b32 v66, v68 // (op 6072) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v67, v69 // (op 6072) 
buffer_load_dwordx2 v[46:47], v10, s[52:55], 0 offen offset:328 // (op 219) Load value


v_add_u32 v17, 192, v17 // (op 6873) 
v_bfe_u32 v62, v13, 0, 8 // (op 6135) 
v_bfe_u32 v63, v13, 8, 8 // (op 6135) 
v_bfe_u32 v64, v13, 16, 8 // (op 6135) 
v_bfe_u32 v65, v13, 24, 8 // (op 6135) 
v_permlane16_swap_b32 v62, v63 // (op 6135) 
s_nop 0
v_permlane16_swap_b32 v64, v65 // (op 6135) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v62, v64 // (op 6135) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v63, v65 // (op 6135) 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 7395) 


s_lshl_b64 s[10:11], s[12:13], 6 // (op 6734) 
s_lshl_b64 s[14:15], s[10:11], 2 // (op 6734) 

v_lshrrev_b64 v[12:13], 3, s[14:15] // (op 6734) 


v_mov_b64 v[48:49], v[50:51] // (op 6734) 
s_add_u32 s7, 8192, s1 // (op 6734) 
s_mov_b32 m0, s7 // (op 6734) reg expression
s_mov_b32 s10, 192 // (op 6734) call()
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[52:55], v[56:61], a[0:3], v66, v62 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 217) 
ds_read_b64 v[80:81], v6 offset:31760 // (op 456) Load local data 



ds_read_b128 v[82:85], v6 offset:33280 // (op 592) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[52:55], v[70:75], a[4:7], v66, v63 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 312) 
ds_read_b64 v[86:87], v6 offset:33296 // (op 592) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[52:55], v[76:81], a[8:11], v66, v64 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 448) 
s_barrier  // (op 7398) 
s_barrier  // (op 7451) 



ds_read_b128 v[88:91], v5 offset:9216 // (op 897) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[52:55], v[82:87], a[12:15], v66, v65 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 584) 




ds_read_b128 v[52:55], v5 offset:10240 // (op 1507) Load local data 



ds_read_b128 v[92:95], v5 offset:11264 // (op 2117) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[88:91], v[56:61], a[16:19], v67, v62 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 1004) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[88:91], v[76:81], a[24:27], v67, v64 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 928) 
buffer_load_dwordx4 v48, s[48:51], s10 offen offset: 0 lds // (op 6734) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[88:91], v[82:87], a[28:31], v67, v65 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 890) 
v_add_co_u32 v48, s[14:15], v48, v12 // (op 6734) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[88:91], v[70:75], a[20:23], v67, v63 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 966) 
v_addc_co_u32 v49, s[14:15], 0, v49, s[14:15] // (op 6734) most significant half


s_add_u32 m0, m0, 4096 // (op 6734) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[52:55], v[56:61], a[32:35], v68, v62 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 1614) 
buffer_load_dwordx4 v48, s[48:51], s10 offen offset: 0 lds // (op 6734) Load value direct to lds





v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[52:55], v[76:81], a[40:43], v68, v64 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 1538) 
s_barrier  // (op 7381) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[52:55], v[82:87], a[44:47], v68, v65 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 1500) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[52:55], v[70:75], a[36:39], v68, v63 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 1576) 
v_add_co_u32 v14, vcc, 128, v14 // (op 6809) least significant half

v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 6809) most significant half
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[92:95], v[56:61], a[48:51], v69, v62 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 2224) 
s_waitcnt vmcnt(13)
ds_write_b128 v16, v[20:23] offset:16384 // (op 13) Store local data 

s_waitcnt vmcnt(12)
ds_write_b64 v16, v[24:25] offset:16400 // (op 13) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[92:95], v[76:81], a[56:59], v69, v64 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 2148) 
s_waitcnt vmcnt(11)
ds_write_b128 v16, v[26:29] offset:16408 // (op 13) Store local data 

s_waitcnt vmcnt(10)
ds_write_b64 v16, v[30:31] offset:16424 // (op 13) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[92:95], v[82:87], a[60:63], v69, v65 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 2110) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[92:95], v[70:75], a[52:55], v69, v63 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 2186) 




s_ashr_i64 s[10:11], s[36:37], 7 // (op 1465) 
s_mov_b32 s7, s10 // (op 1465) convert

s_and_b32 s10, s7, 4294967294 // (op 1465) 

s_cmp_lt_u32 s6, s10 // (op 1465) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopTop_KLoop_1465 // (op 1465) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopTop_KLoop_1465 if true)
RR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopBottom_KLoop_1465:












s_barrier  // (op 7463) 
v_lshrrev_b32 v10, 2, v1 // (op 6757) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 6757) 
v_lshl_add_u32 v11, s16, 7, v10 // (op 6757) 

s_lshr_b32 s6, s0, 1 // (op 6757) 

v_and_b32 v10, -4, v1 // (op 6757) 
v_sub_u32 v12, v1, v10 // (op 6757) 

v_lshlrev_b32 v10, 5, v12 // (op 6757) 

v_lshl_add_u32 v12, s6, 8, v10 // (op 6757) 


v_mov_b32 v31, 0 // (op 6757) convert
v_mov_b32 v30, v12 // (op 6757) convert

v_mul_lo_u32 v13, v11, s13 // (op 6757) most significant: low of low * high
v_mul_hi_u32 v10, v11, s12 // (op 6757) most significant: high of low * low
v_mul_lo_u32 v12, v11, s12 // (op 6757) least significant: low of low * low
v_add_u32 v13, v13, v10 // (op 6757) most significant: sum

v_add_co_u32 v12, s[0:1], v12, v30 // (op 6757) least significant half
v_addc_co_u32 v13, s[0:1], v13, v31, s[0:1] // (op 6757) most significant half



v_lshlrev_b64 v[10:11], 2, v[12:13] // (op 6757) 

v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 6757) 

s_mul_i32 s1, s12, s23 // (op 6759) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 6759) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 6759) least significant
s_mul_hi_u32 s7, s12, s22 // (op 6759) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6759) most significant: sum
s_add_u32 s1, s1, s7 // (op 6759) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 6759) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 6759) 

s_mov_b32 s48, s20 // (op 6759) reg expression
s_mov_b32 s49, s21 // (op 6759) reg expression
s_mov_b32 s50, s0 // (op 6759) reg expression
s_mov_b32 s51, 131072 // (op 6759) call()

v_and_b32 v10, -2, v1 // (op 6839) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 6839) 
v_sub_u32 v11, v1, v10 // (op 6839) 

s_lshr_b32 s6, s0, 1 // (op 6839) 

v_lshlrev_b32 v10, 6, v11 // (op 6839) 

v_lshl_add_u32 v11, s6, 8, v10 // (op 6839) 


v_mov_b32 v15, 0 // (op 6839) convert
v_mov_b32 v14, v11 // (op 6839) convert

v_lshrrev_b32 v10, 1, v1 // (op 6839) 
v_lshl_add_u32 v11, s17, 7, v10 // (op 6839) 

v_mul_lo_u32 v17, v11, s25 // (op 6839) most significant: low of low * high
v_mul_hi_u32 v10, v11, s24 // (op 6839) most significant: high of low * low
v_mul_lo_u32 v16, v11, s24 // (op 6839) least significant: low of low * low
v_add_u32 v17, v17, v10 // (op 6839) most significant: sum


v_add_co_u32 v10, s[0:1], v14, v16 // (op 6839) least significant half
v_addc_co_u32 v11, s[0:1], v15, v17, s[0:1] // (op 6839) most significant half



v_mul_lo_u32 v15, v11, 6 // (op 6839) most significant: low of high * low
v_mul_hi_u32 v16, v10, 6 // (op 6839) most significant: high of low * low
v_mul_lo_u32 v14, v10, 6 // (op 6839) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 6839) most significant: sum


v_lshrrev_b64 v[10:11], 3, v[14:15] // (op 6839) 

v_mov_b32 v14, v10 // (op 6839) convert

s_mul_i32 s1, s9, 6 // (op 6841) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 6841) least significant
s_mul_hi_u32 s6, s8, 6 // (op 6841) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6841) most significant: sum

s_mul_i32 s7, s24, s1 // (op 6841) most significant: low of low * high
s_mul_i32 s10, s25, s0 // (op 6841) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 6841) least significant
s_mul_hi_u32 s11, s24, s0 // (op 6841) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 6841) most significant: sum
s_add_u32 s7, s7, s11 // (op 6841) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 6841) 

s_mov_b32 s52, s26 // (op 6841) reg expression
s_mov_b32 s53, s27 // (op 6841) reg expression
s_mov_b32 s54, s0 // (op 6841) reg expression
s_mov_b32 s55, 131072 // (op 6841) call()

v_and_b32 v10, -4, v1 // (op 6922) 
v_sub_u32 v11, v1, v10 // (op 6922) 

v_lshlrev_b32 v10, 5, v11 // (op 6922) 

v_lshrrev_b32 v11, 2, v1 // (op 6922) 
v_lshl_add_u32 v15, v11, 7, v10 // (op 6922) 


v_lshlrev_b32 v10, 2, v15 // (op 6922) 

v_lshrrev_b32 v11, 3, v10 // (op 6922) 

s_nop 0
v_readfirstlane_b32 s0, v11 // (op 6922) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v10, -2, v1 // (op 6981) 
v_sub_u32 v11, v1, v10 // (op 6981) 

v_lshlrev_b32 v10, 6, v11 // (op 6981) 

v_lshrrev_b32 v11, 1, v1 // (op 6981) 
v_lshl_add_u32 v15, v11, 7, v10 // (op 6981) 


v_mul_lo_u32 v10, v15, 6 // (op 6981) 

v_lshrrev_b32 v11, 3, v10 // (op 6981) 

v_and_b32 v10, 63, v1 // (op 7245) 
s_ashr_i64 s[6:7], s[36:37], 7 // (op 7245) 
v_lshrrev_b32 v15, 4, v10 // (op 7245) 
s_lshr_b32 s1, s6, 1 // (op 7245) 

v_and_b32 v16, 3, v15 // (op 7245) 

v_and_b32 v15, 15, v1 // (op 7245) 
v_lshl_add_u32 v17, v16, 4, v15 // (op 7245) 


v_lshrrev_b32 v15, 7, v1 // (op 7245) 
v_lshl_add_u32 v16, v15, 6, v17 // (op 7245) 


v_lshl_add_u32 v15, s16, 7, v16 // (op 7245) 

v_lshrrev_b32 v16, 6, v10 // (op 7245) 

v_lshlrev_b32 v10, 2, v16 // (op 7245) 

v_lshl_add_u32 v16, s1, 3, v10 // (op 7245) 


v_mul_lo_u32 v10, v15, s28 // (op 7245) 
v_add_u32 v10, v10, v16 // (op 7245) 


s_mul_i32 s7, s28, s23 // (op 7247) most significant: low of low * high
s_mul_i32 s1, s29, s22 // (op 7247) most significant: low of high * low
s_mul_i32 s6, s28, s22 // (op 7247) least significant
s_mul_hi_u32 s10, s28, s22 // (op 7247) most significant: high of low * low
s_add_u32 s7, s7, s1 // (op 7247) most significant: sum
s_add_u32 s7, s7, s10 // (op 7247) most significant: sum


s_mov_b32 s56, s30 // (op 7247) reg expression
s_mov_b32 s57, s31 // (op 7247) reg expression
s_mov_b32 s58, s6 // (op 7247) reg expression
s_mov_b32 s59, 131072 // (op 7247) call()

v_and_b32 v15, 63, v1 // (op 7272) 
s_ashr_i64 s[6:7], s[36:37], 7 // (op 7272) 
v_lshrrev_b32 v16, 6, v15 // (op 7272) 
s_lshr_b32 s1, s6, 1 // (op 7272) 

v_lshlrev_b32 v17, 2, v16 // (op 7272) 

v_lshl_add_u32 v16, s1, 3, v17 // (op 7272) 


v_lshrrev_b32 v17, 4, v15 // (op 7272) 

v_and_b32 v15, 3, v17 // (op 7272) 

v_and_b32 v17, 15, v1 // (op 7272) 
v_lshl_add_u32 v18, v15, 4, v17 // (op 7272) 


v_lshrrev_b32 v15, 6, v1 // (op 7272) 
v_and_b32 v17, 1, v15 // (op 7272) 

v_lshl_add_u32 v15, v17, 6, v18 // (op 7272) 


v_lshl_add_u32 v17, s17, 7, v15 // (op 7272) 

v_mul_lo_u32 v15, v17, s32 // (op 7272) 

v_add_u32 v17, v16, v15 // (op 7272) 


s_mul_i32 s7, s32, s9 // (op 7274) most significant: low of low * high
s_mul_i32 s1, s33, s8 // (op 7274) most significant: low of high * low
s_mul_i32 s6, s32, s8 // (op 7274) least significant
s_mul_hi_u32 s10, s32, s8 // (op 7274) most significant: high of low * low
s_add_u32 s7, s7, s1 // (op 7274) most significant: sum
s_add_u32 s7, s7, s10 // (op 7274) most significant: sum


s_mov_b32 s60, s34 // (op 7274) reg expression
s_mov_b32 s61, s35 // (op 7274) reg expression
s_mov_b32 s62, s6 // (op 7274) reg expression
s_mov_b32 s63, 131072 // (op 7274) call()

s_ashr_i64 s[6:7], s[36:37], 7 // (op 153) 
s_and_b32 s1, s6, 4294967294 // (op 153) 

s_ashr_i64 s[6:7], s[36:37], 7 // (op 152) 
s_mov_b32 s10, s6 // (op 152) convert

s_cmp_lt_u32 s1, s10 // (op 152) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopBottom_KLoopTail_152 // (op 152) Condition: Top (jump to RR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopBottom_KLoopTail_152 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopTop_KLoopTail_152:
s_barrier  // (op 7368) 
s_add_u32 s1, s1, 1 // (op 154) 

s_lshl_b64 s[6:7], s[12:13], 6 // (op 6728) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 6728) 

v_lshrrev_b64 v[18:19], 3, s[10:11] // (op 6728) 


v_mov_b64 v[20:21], v[12:13] // (op 6728) 
v_mov_b64 v[22:23], v[20:21] // (op 6728) 
s_mov_b32 m0, s0 // (op 6728) reg expression
buffer_load_dwordx4 v22, s[48:51], 0 offen offset: 0 lds // (op 6728) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v18 // (op 6728) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 6728) most significant half

s_add_u32 m0, m0, 4096 // (op 6728) 
buffer_load_dwordx4 v22, s[48:51], 0 offen offset: 0 lds // (op 6728) Load value direct to lds



v_add_co_u32 v12, s[6:7], 64, v12 // (op 6747) least significant half
v_addc_co_u32 v13, s[6:7], 0, v13, s[6:7] // (op 6747) most significant half



v_mov_b32 v15, v14 // (op 169) 

buffer_load_dwordx4 v[18:21], v15, s[52:55], 0 offen // (op 169) Load value
buffer_load_dwordx2 v[22:23], v15, s[52:55], 0 offen offset:16 // (op 169) Load value
buffer_load_dwordx4 v[24:27], v15, s[52:55], 0 offen offset:24 // (op 169) Load value
buffer_load_dwordx2 v[28:29], v15, s[52:55], 0 offen offset:40 // (op 169) Load value

v_add_u32 v14, 96, v14 // (op 6831) 
s_barrier  // (op 7384) 
s_waitcnt vmcnt(4)
s_barrier  // (op 7374) 






ds_read_b128 v[30:33], v3 // (op 160) Load local data 
s_waitcnt vmcnt(3)
ds_write_b128 v11, v[18:21] offset:16384 // (op 168) Store local data 
s_waitcnt vmcnt(2)
ds_write_b64 v11, v[22:23] offset:16400 // (op 168) Store local data 
s_waitcnt vmcnt(1)
ds_write_b128 v11, v[24:27] offset:16408 // (op 168) Store local data 
s_waitcnt vmcnt(0)
ds_write_b64 v11, v[28:29] offset:16424 // (op 168) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 7388) 



ds_read_b128 v[18:21], v4 offset:16384 // (op 162) Load local data 
ds_read_b64 v[22:23], v4 offset:16400 // (op 162) Load local data 


v_mov_b32 v15, v10 // (op 164) 
v_mov_b32 v16, v15 // (op 164) reg expression

buffer_load_dwordx2 v[24:25], v16, s[56:59], 0 offen // (op 164) Load value


v_add_u32 v10, 4, v10 // (op 7235) 


v_mov_b32 v15, v17 // (op 166) 
v_mov_b32 v16, v15 // (op 166) reg expression

buffer_load_dwordx2 v[26:27], v16, s[60:63], 0 offen // (op 166) Load value


v_add_u32 v17, 4, v17 // (op 7262) 
s_waitcnt vmcnt(1)
v_bfe_u32 v34, v24, 0, 8 // (op 6184) 



v_bfe_u32 v35, v24, 8, 8 // (op 6184) 
v_bfe_u32 v36, v24, 16, 8 // (op 6184) 
v_bfe_u32 v37, v24, 24, 8 // (op 6184) 
v_permlane16_swap_b32 v34, v35 // (op 6184) 
ds_read_b128 v[38:41], v4 offset:17920 // (op 291) Load local data 
v_permlane16_swap_b32 v36, v37 // (op 6184) 
s_nop 1
v_permlane32_swap_b32 v34, v36 // (op 6184) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v35, v37 // (op 6184) 
ds_read_b64 v[42:43], v4 offset:17936 // (op 291) Load local data 




ds_read_b128 v[44:47], v4 offset:19456 // (op 427) Load local data 
ds_read_b64 v[48:49], v4 offset:19472 // (op 427) Load local data 
s_waitcnt vmcnt(0)
v_bfe_u32 v50, v26, 0, 8 // (op 6214) 



v_bfe_u32 v51, v26, 8, 8 // (op 6214) 
v_bfe_u32 v52, v26, 16, 8 // (op 6214) 
v_bfe_u32 v53, v26, 24, 8 // (op 6214) 
v_permlane16_swap_b32 v50, v51 // (op 6214) 
s_nop 0
v_permlane16_swap_b32 v52, v53 // (op 6214) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v50, v52 // (op 6214) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v51, v53 // (op 6214) 
ds_read_b128 v[54:57], v4 offset:20992 // (op 563) Load local data 

ds_read_b64 v[58:59], v4 offset:21008 // (op 563) Load local data 



s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[30:33], v[18:23], a[0:3], v34, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 167) 
ds_read_b128 v[24:27], v3 offset:1024 // (op 785) Load local data 



s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[30:33], v[38:43], a[4:7], v34, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 294) 
ds_read_b128 v[60:63], v3 offset:2048 // (op 1395) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[30:33], v[44:49], a[8:11], v34, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 430) 
ds_read_b128 v[64:67], v3 offset:3072 // (op 2005) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[30:33], v[54:59], a[12:15], v34, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 566) 

s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[24:27], v[18:23], a[16:19], v35, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 789) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[24:27], v[44:49], a[24:27], v35, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 843) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[24:27], v[54:59], a[28:31], v35, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 870) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[24:27], v[38:43], a[20:23], v35, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 816) 

s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[60:63], v[18:23], a[32:35], v36, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 1399) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[60:63], v[44:49], a[40:43], v36, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 1453) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[60:63], v[54:59], a[44:47], v36, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 1480) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[60:63], v[38:43], a[36:39], v36, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 1426) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[64:67], v[18:23], a[48:51], v37, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 2009) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[64:67], v[44:49], a[56:59], v37, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 2063) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[64:67], v[54:59], a[60:63], v37, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 2090) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[64:67], v[38:43], a[52:55], v37, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 2036) 




s_ashr_i64 s[6:7], s[36:37], 7 // (op 152) 
s_mov_b32 s10, s6 // (op 152) convert

s_cmp_lt_u32 s1, s10 // (op 152) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopTop_KLoopTail_152 // (op 152) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopTop_KLoopTail_152 if true)
RR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopBottom_KLoopTail_152:





















s_barrier  // (op 7435) 





v_cmp_eq_f32 vcc, 0.00000, v0 // (op 114) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ConditionalFalse_0_beta_114 // (op 114) Condition: False, jump to RR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ConditionalFalse_0_beta_114
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
s_branch RR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ConditionalBottom_0_beta_114 // (op 114) Condition: Done, jump to RR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ConditionalBottom_0_beta_114
RR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ConditionalFalse_0_beta_114:


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
RR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ConditionalBottom_0_beta_114:
























s_endpgm  // End of RR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2
.LRR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_end:
.size RR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2, .LRR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_end-RR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2
  .amdhsa_next_free_vgpr 164
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 40960 // lds bytes
  .amdhsa_accum_offset 100
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
  - .name: RR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2
    .symbol: RR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 40960
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 64
    .vgpr_count: 98
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
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: FP6, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP6, pointerType: PointerGlobal}
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

