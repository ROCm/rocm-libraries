// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2
.p2align 8
.type RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2,@function
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 8913) 
v_and_b32 v3, 15, v1 // (op 8913) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 8913) 


v_and_b32 v0, 48, v1 // (op 8913) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 8913) 


v_lshrrev_b32 v0, 6, v1 // (op 8970) 
v_and_b32 v4, 1, v0 // (op 8970) 

v_and_b32 v0, 15, v1 // (op 8970) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 8970) 


v_and_b32 v0, 48, v1 // (op 8970) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 8970) 


v_lshrrev_b32 v0, 7, v1 // (op 9028) 
v_and_b32 v5, 15, v1 // (op 9028) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 9028) 


v_and_b32 v0, 63, v1 // (op 9028) 
v_lshrrev_b32 v5, 4, v0 // (op 9028) 

v_lshl_add_u32 v0, v6, 2, v5 // (op 9028) 


v_lshrrev_b32 v5, 6, v1 // (op 9086) 
v_and_b32 v6, 1, v5 // (op 9086) 

v_and_b32 v5, 15, v1 // (op 9086) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 9086) 


v_and_b32 v5, 63, v1 // (op 9086) 
v_lshrrev_b32 v6, 4, v5 // (op 9086) 

v_lshl_add_u32 v5, v7, 2, v6 // (op 9086) 


v_lshrrev_b32 v6, 7, v1 // (op 9153) 
v_and_b32 v7, 15, v1 // (op 9153) 
v_lshl_add_u32 v8, v6, 4, v7 // (op 9153) 


v_and_b32 v6, 48, v1 // (op 9153) 
v_lshl_add_u32 v7, v8, 7, v6 // (op 9153) 


v_lshrrev_b32 v6, 6, v1 // (op 9197) 
v_and_b32 v8, 1, v6 // (op 9197) 

v_and_b32 v6, 15, v1 // (op 9197) 
v_lshl_add_u32 v9, v8, 4, v6 // (op 9197) 


v_and_b32 v6, 48, v1 // (op 9197) 
v_lshl_add_u32 v8, v9, 7, v6 // (op 9197) 


v_lshrrev_b32 v6, 7, v1 // (op 9226) 
v_and_b32 v9, 15, v1 // (op 9226) 
v_lshl_add_u32 v10, v6, 4, v9 // (op 9226) 


v_and_b32 v6, 63, v1 // (op 9226) 
v_lshrrev_b32 v9, 4, v6 // (op 9226) 

v_lshl_add_u32 v6, v10, 2, v9 // (op 9226) 


v_lshrrev_b32 v9, 6, v1 // (op 9255) 
v_and_b32 v10, 1, v9 // (op 9255) 

v_and_b32 v9, 15, v1 // (op 9255) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 9255) 


v_and_b32 v9, 63, v1 // (op 9255) 
v_lshrrev_b32 v10, 4, v9 // (op 9255) 

v_lshl_add_u32 v9, v11, 2, v10 // (op 9255) 


v_lshrrev_b32 v10, 7, v1 // (op 8852) 
v_and_b32 v11, 63, v1 // (op 8852) 
v_lshrrev_b32 v12, 4, v11 // (op 8852) 
v_lshlrev_b32 v13, 2, v12 // (op 8852) 

v_lshl_add_u32 v12, v10, 4, v13 // (op 8852) 


v_lshl_add_u32 v10, s16, 8, v12 // (op 8852) 

v_lshrrev_b32 v12, 2, v11 // (op 8852) 

v_and_b32 v11, 3, v12 // (op 8852) 

v_and_b32 v12, 3, v1 // (op 8852) 
v_lshl_add_u32 v13, v11, 2, v12 // (op 8852) 


v_lshrrev_b32 v11, 6, v1 // (op 8852) 
v_and_b32 v12, 1, v11 // (op 8852) 

v_lshl_add_u32 v11, v12, 4, v13 // (op 8852) 


v_lshl_add_u32 v12, s17, 6, v11 // (op 8852) 

v_mul_lo_u32 v11, v12, s2 // (op 8852) 

v_add_lshl_u32 v12, v10, v11, 2 // (op 8852) 


v_lshrrev_b32 v10, 7, v1 // (op 9107) 
v_and_b32 v11, 63, v1 // (op 9107) 
v_lshrrev_b32 v13, 4, v11 // (op 9107) 
s_mul_i32 s1, s2, s9 // (op 8854) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 8854) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 8854) least significant
s_mul_hi_u32 s19, s2, s8 // (op 8854) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8854) most significant: sum
s_add_u32 s1, s1, s19 // (op 8854) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 8854) 

v_lshlrev_b32 v14, 2, v13 // (op 9107) 

v_lshl_add_u32 v13, v10, 4, v14 // (op 9107) 


v_lshl_add_u32 v10, s16, 8, v13 // (op 9107) 

v_lshrrev_b32 v13, 2, v11 // (op 9107) 

v_and_b32 v11, 3, v13 // (op 9107) 

v_and_b32 v13, 3, v1 // (op 9107) 
v_lshl_add_u32 v14, v11, 2, v13 // (op 9107) 


v_lshrrev_b32 v11, 6, v1 // (op 9107) 
v_and_b32 v13, 1, v11 // (op 9107) 

v_lshl_add_u32 v11, v13, 4, v14 // (op 9107) 


v_lshl_add_u32 v13, s17, 6, v11 // (op 9107) 

v_mul_lo_u32 v11, v13, s4 // (op 9107) 

v_add_lshl_u32 v13, v10, v11, 2 // (op 9107) 


s_mov_b32 s40, s6 // (op 8854) reg expression
s_mov_b32 s41, s7 // (op 8854) reg expression
s_mov_b32 s42, s18 // (op 8854) reg expression
s_mov_b32 s43, 131072 // (op 8854) call()

s_mul_i32 s1, s4, s9 // (op 9109) most significant: low of low * high

s_mul_i32 s6, s5, s8 // (op 9109) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 9109) least significant
s_mul_hi_u32 s7, s4, s8 // (op 9109) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 9109) most significant: sum
s_add_u32 s1, s1, s7 // (op 9109) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 9109) 

s_mov_b32 s44, s10 // (op 9109) reg expression
s_mov_b32 s45, s11 // (op 9109) reg expression
s_mov_b32 s46, s6 // (op 9109) reg expression
s_mov_b32 s47, 131072 // (op 9109) call()


v_mov_b32 v10, s14 // (op 2) Move value
v_mov_b32 v11, s15 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 63) call()
v_accvgpr_write a1, 0.00000 // (op 63) call()
v_accvgpr_write a2, 0.00000 // (op 63) call()
v_accvgpr_write a3, 0.00000 // (op 63) call()
v_accvgpr_write a4, 0.00000 // (op 336) call()
v_accvgpr_write a5, 0.00000 // (op 336) call()
v_accvgpr_write a6, 0.00000 // (op 336) call()
v_accvgpr_write a7, 0.00000 // (op 336) call()
v_accvgpr_write a8, 0.00000 // (op 556) call()
v_accvgpr_write a9, 0.00000 // (op 556) call()
v_accvgpr_write a10, 0.00000 // (op 556) call()
v_accvgpr_write a11, 0.00000 // (op 556) call()
v_accvgpr_write a12, 0.00000 // (op 711) call()
v_accvgpr_write a13, 0.00000 // (op 711) call()
v_accvgpr_write a14, 0.00000 // (op 711) call()
v_accvgpr_write a15, 0.00000 // (op 711) call()
v_accvgpr_write a16, 0.00000 // (op 942) call()
v_accvgpr_write a17, 0.00000 // (op 942) call()
v_accvgpr_write a18, 0.00000 // (op 942) call()
v_accvgpr_write a19, 0.00000 // (op 942) call()
v_accvgpr_write a20, 0.00000 // (op 1097) call()
v_accvgpr_write a21, 0.00000 // (op 1097) call()
v_accvgpr_write a22, 0.00000 // (op 1097) call()
v_accvgpr_write a23, 0.00000 // (op 1097) call()
v_accvgpr_write a24, 0.00000 // (op 1328) call()
v_accvgpr_write a25, 0.00000 // (op 1328) call()
v_accvgpr_write a26, 0.00000 // (op 1328) call()
v_accvgpr_write a27, 0.00000 // (op 1328) call()
v_accvgpr_write a28, 0.00000 // (op 1483) call()
v_accvgpr_write a29, 0.00000 // (op 1483) call()
v_accvgpr_write a30, 0.00000 // (op 1483) call()
v_accvgpr_write a31, 0.00000 // (op 1483) call()
v_accvgpr_write a32, 0.00000 // (op 1714) call()
v_accvgpr_write a33, 0.00000 // (op 1714) call()
v_accvgpr_write a34, 0.00000 // (op 1714) call()
v_accvgpr_write a35, 0.00000 // (op 1714) call()
v_accvgpr_write a36, 0.00000 // (op 1869) call()
v_accvgpr_write a37, 0.00000 // (op 1869) call()
v_accvgpr_write a38, 0.00000 // (op 1869) call()
v_accvgpr_write a39, 0.00000 // (op 1869) call()
v_accvgpr_write a40, 0.00000 // (op 2100) call()
v_accvgpr_write a41, 0.00000 // (op 2100) call()
v_accvgpr_write a42, 0.00000 // (op 2100) call()
v_accvgpr_write a43, 0.00000 // (op 2100) call()
v_accvgpr_write a44, 0.00000 // (op 2255) call()
v_accvgpr_write a45, 0.00000 // (op 2255) call()
v_accvgpr_write a46, 0.00000 // (op 2255) call()
v_accvgpr_write a47, 0.00000 // (op 2255) call()
v_accvgpr_write a48, 0.00000 // (op 2486) call()
v_accvgpr_write a49, 0.00000 // (op 2486) call()
v_accvgpr_write a50, 0.00000 // (op 2486) call()
v_accvgpr_write a51, 0.00000 // (op 2486) call()
v_accvgpr_write a52, 0.00000 // (op 2641) call()
v_accvgpr_write a53, 0.00000 // (op 2641) call()
v_accvgpr_write a54, 0.00000 // (op 2641) call()
v_accvgpr_write a55, 0.00000 // (op 2641) call()
v_accvgpr_write a56, 0.00000 // (op 2872) call()
v_accvgpr_write a57, 0.00000 // (op 2872) call()
v_accvgpr_write a58, 0.00000 // (op 2872) call()
v_accvgpr_write a59, 0.00000 // (op 2872) call()
v_accvgpr_write a60, 0.00000 // (op 3027) call()
v_accvgpr_write a61, 0.00000 // (op 3027) call()
v_accvgpr_write a62, 0.00000 // (op 3027) call()
v_accvgpr_write a63, 0.00000 // (op 3027) call()
v_lshrrev_b32 v14, 3, v1 // (op 8614) 
v_lshl_add_u32 v15, s16, 8, v14 // (op 8614) 

v_and_b32 v14, -8, v1 // (op 8614) 
v_sub_u32 v16, v1, v14 // (op 8614) 

v_lshlrev_b32 v14, 4, v16 // (op 8614) 

v_mov_b32 v17, 0 // (op 8614) convert
v_mov_b32 v16, v14 // (op 8614) convert

v_mul_lo_u32 v19, v15, s13 // (op 8614) most significant: low of low * high
v_mul_hi_u32 v14, v15, s12 // (op 8614) most significant: high of low * low
v_mul_lo_u32 v18, v15, s12 // (op 8614) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 8614) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 8614) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 8614) most significant half




s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 8616) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 8616) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 8616) least significant
s_mul_hi_u32 s7, s12, s22 // (op 8616) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8616) most significant: sum
s_add_u32 s1, s1, s7 // (op 8616) most significant: sum


s_mov_b32 s48, s20 // (op 8616) reg expression
s_mov_b32 s49, s21 // (op 8616) reg expression
s_mov_b32 s50, s0 // (op 8616) reg expression
s_mov_b32 s51, 131072 // (op 8616) call()

v_lshrrev_b32 v14, 3, v1 // (op 8633) 
v_lshl_add_u32 v15, s16, 8, v14 // (op 8633) 

v_and_b32 v14, -8, v1 // (op 8633) 
v_sub_u32 v16, v1, v14 // (op 8633) 

v_lshlrev_b32 v14, 4, v16 // (op 8633) 

v_mov_b32 v17, 0 // (op 8633) convert
v_mov_b32 v16, v14 // (op 8633) convert

v_mul_lo_u32 v21, v15, s13 // (op 8633) most significant: low of low * high
v_mul_hi_u32 v14, v15, s12 // (op 8633) most significant: high of low * low
v_mul_lo_u32 v20, v15, s12 // (op 8633) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 8633) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v16 // (op 8633) least significant half
v_addc_co_u32 v21, s[0:1], v21, v17, s[0:1] // (op 8633) most significant half



s_mul_i32 s1, s12, s23 // (op 8635) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 8635) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 8635) least significant
s_mul_hi_u32 s7, s12, s22 // (op 8635) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8635) most significant: sum
s_add_u32 s1, s1, s7 // (op 8635) most significant: sum


s_mov_b32 s48, s20 // (op 8635) reg expression
s_mov_b32 s49, s21 // (op 8635) reg expression
s_mov_b32 s50, s0 // (op 8635) reg expression
s_mov_b32 s51, 131072 // (op 8635) call()

v_and_b32 v14, -8, v1 // (op 8678) 
v_sub_u32 v15, v1, v14 // (op 8678) 

v_lshlrev_b32 v14, 4, v15 // (op 8678) 

v_mov_b32 v17, 0 // (op 8678) convert
v_mov_b32 v16, v14 // (op 8678) convert

v_lshrrev_b32 v14, 3, v1 // (op 8678) 
v_lshl_add_u32 v15, s17, 6, v14 // (op 8678) 

v_mul_lo_u32 v23, v15, s25 // (op 8678) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 8678) most significant: high of low * low
v_mul_lo_u32 v22, v15, s24 // (op 8678) least significant: low of low * low
v_add_u32 v23, v23, v14 // (op 8678) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v22 // (op 8678) least significant half
v_addc_co_u32 v15, s[0:1], v17, v23, s[0:1] // (op 8678) most significant half



s_mul_i32 s1, s24, s9 // (op 8680) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 8680) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 8680) least significant
s_mul_hi_u32 s7, s24, s8 // (op 8680) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8680) most significant: sum
s_add_u32 s1, s1, s7 // (op 8680) most significant: sum


s_mov_b32 s52, s26 // (op 8680) reg expression
s_mov_b32 s53, s27 // (op 8680) reg expression
s_mov_b32 s54, s0 // (op 8680) reg expression
s_mov_b32 s55, 131072 // (op 8680) call()

v_and_b32 v16, -8, v1 // (op 8697) 
v_sub_u32 v17, v1, v16 // (op 8697) 

v_lshlrev_b32 v16, 4, v17 // (op 8697) 

v_mov_b32 v23, 0 // (op 8697) convert
v_mov_b32 v22, v16 // (op 8697) convert

v_lshrrev_b32 v16, 3, v1 // (op 8697) 
v_lshl_add_u32 v17, s17, 6, v16 // (op 8697) 

v_mul_lo_u32 v25, v17, s25 // (op 8697) most significant: low of low * high
v_mul_hi_u32 v16, v17, s24 // (op 8697) most significant: high of low * low
v_mul_lo_u32 v24, v17, s24 // (op 8697) least significant: low of low * low
v_add_u32 v25, v25, v16 // (op 8697) most significant: sum


v_add_co_u32 v16, s[0:1], v22, v24 // (op 8697) least significant half
v_addc_co_u32 v17, s[0:1], v23, v25, s[0:1] // (op 8697) most significant half



s_mul_i32 s1, s24, s9 // (op 8699) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 8699) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 8699) least significant
s_mul_hi_u32 s7, s24, s8 // (op 8699) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8699) most significant: sum
s_add_u32 s1, s1, s7 // (op 8699) most significant: sum


s_mov_b32 s52, s26 // (op 8699) reg expression
s_mov_b32 s53, s27 // (op 8699) reg expression
s_mov_b32 s54, s0 // (op 8699) reg expression
s_mov_b32 s55, 131072 // (op 8699) call()

v_lshl_add_u32 v22, s16, 8, v1 // (op 8742) 
v_mul_lo_u32 v23, v22, s28 // (op 8742) 

s_mul_i32 s1, s28, s23 // (op 8744) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 8744) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 8744) least significant
s_mul_hi_u32 s7, s28, s22 // (op 8744) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8744) most significant: sum
s_add_u32 s1, s1, s7 // (op 8744) most significant: sum


s_mov_b32 s56, s30 // (op 8744) reg expression
s_mov_b32 s57, s31 // (op 8744) reg expression
s_mov_b32 s58, s0 // (op 8744) reg expression
s_mov_b32 s59, 131072 // (op 8744) call()

v_lshl_add_u32 v22, s16, 8, v1 // (op 8761) 
v_mul_lo_u32 v24, v22, s28 // (op 8761) 

s_mul_i32 s1, s28, s23 // (op 8763) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 8763) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 8763) least significant
s_mul_hi_u32 s7, s28, s22 // (op 8763) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8763) most significant: sum
s_add_u32 s1, s1, s7 // (op 8763) most significant: sum


s_mov_b32 s56, s30 // (op 8763) reg expression
s_mov_b32 s57, s31 // (op 8763) reg expression
s_mov_b32 s58, s0 // (op 8763) reg expression
s_mov_b32 s59, 131072 // (op 8763) call()

v_and_b32 v22, -4, v1 // (op 8806) 
v_sub_u32 v25, v1, v22 // (op 8806) 

v_lshrrev_b32 v22, 2, v1 // (op 8806) 
v_lshl_add_u32 v26, s17, 6, v22 // (op 8806) 

v_mul_lo_u32 v22, v26, s32 // (op 8806) 

v_add_u32 v26, v25, v22 // (op 8806) 


s_mul_i32 s1, s32, s9 // (op 8808) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 8808) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 8808) least significant
s_mul_hi_u32 s7, s32, s8 // (op 8808) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8808) most significant: sum
s_add_u32 s1, s1, s7 // (op 8808) most significant: sum


s_mov_b32 s60, s34 // (op 8808) reg expression
s_mov_b32 s61, s35 // (op 8808) reg expression
s_mov_b32 s62, s0 // (op 8808) reg expression
s_mov_b32 s63, 131072 // (op 8808) call()

v_and_b32 v22, -4, v1 // (op 8825) 
v_sub_u32 v25, v1, v22 // (op 8825) 

v_lshrrev_b32 v22, 2, v1 // (op 8825) 
v_lshl_add_u32 v27, s17, 6, v22 // (op 8825) 

v_mul_lo_u32 v22, v27, s32 // (op 8825) 

v_add_u32 v27, v25, v22 // (op 8825) 


s_mul_i32 s1, s32, s9 // (op 8827) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 8827) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 8827) least significant
s_mul_hi_u32 s7, s32, s8 // (op 8827) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8827) most significant: sum
s_add_u32 s1, s1, s7 // (op 8827) most significant: sum


s_mov_b32 s60, s34 // (op 8827) reg expression
s_mov_b32 s61, s35 // (op 8827) reg expression
s_mov_b32 s62, s0 // (op 8827) reg expression
s_mov_b32 s63, 131072 // (op 8827) call()

v_and_b32 v22, -8, v1 // (op 8953) 
v_sub_u32 v25, v1, v22 // (op 8953) 

v_lshlrev_b32 v22, 4, v25 // (op 8953) 

v_lshrrev_b32 v25, 3, v1 // (op 8953) 
v_lshl_add_u32 v28, v25, 7, v22 // (op 8953) 


s_nop 0
v_readfirstlane_b32 s0, v28 // (op 8953) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 8548) 
v_mov_b64 v[28:29], s[6:7] // (op 8548) convert


v_mov_b64 v[30:31], v[14:15] // (op 8548) 
s_mov_b32 m0, s0 // (op 8548) reg expression
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 8548) Load value direct to lds
v_add_co_u32 v30, s[6:7], v30, v28 // (op 8548) least significant half
v_addc_co_u32 v31, s[6:7], 0, v31, s[6:7] // (op 8548) most significant half

s_add_u32 m0, m0, 4096 // (op 8548) 
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 8548) Load value direct to lds






buffer_load_ubyte v22, v26, s[60:63], 0 offen offset: 0 // (op 8234) Load value
v_and_b32 v25, -8, v1 // (op 8896) 
v_sub_u32 v28, v1, v25 // (op 8896) 

v_lshlrev_b32 v25, 4, v28 // (op 8896) 

v_lshrrev_b32 v28, 3, v1 // (op 8896) 
v_lshl_add_u32 v29, v28, 7, v25 // (op 8896) 


s_nop 0
v_readfirstlane_b32 s0, v29 // (op 8896) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 5 // (op 8553) 
v_mov_b64 v[28:29], s[6:7] // (op 8553) convert


v_mov_b64 v[30:31], v[18:19] // (op 8553) 
s_add_u32 s1, 8192, s0 // (op 8553) 
s_mov_b32 m0, s1 // (op 8553) reg expression
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 8553) Load value direct to lds
v_add_co_u32 v30, s[6:7], v30, v28 // (op 8553) least significant half
v_addc_co_u32 v31, s[6:7], 0, v31, s[6:7] // (op 8553) most significant half

s_add_u32 m0, m0, 4096 // (op 8553) 
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 8553) Load value direct to lds
v_add_co_u32 v30, s[6:7], v30, v28 // (op 8553) least significant half
v_addc_co_u32 v31, s[6:7], 0, v31, s[6:7] // (op 8553) most significant half

s_add_u32 m0, m0, 4096 // (op 8553) 
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 8553) Load value direct to lds
v_add_co_u32 v30, s[6:7], v30, v28 // (op 8553) least significant half
v_addc_co_u32 v31, s[6:7], 0, v31, s[6:7] // (op 8553) most significant half

s_add_u32 m0, m0, 4096 // (op 8553) 
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 8553) Load value direct to lds
v_add_co_u32 v30, s[6:7], v30, v28 // (op 8553) least significant half
v_addc_co_u32 v31, s[6:7], 0, v31, s[6:7] // (op 8553) most significant half

s_add_u32 m0, m0, 4096 // (op 8553) 
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 8553) Load value direct to lds
v_add_co_u32 v30, s[6:7], v30, v28 // (op 8553) least significant half
v_addc_co_u32 v31, s[6:7], 0, v31, s[6:7] // (op 8553) most significant half

s_add_u32 m0, m0, 4096 // (op 8553) 
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 8553) Load value direct to lds
v_add_co_u32 v30, s[6:7], v30, v28 // (op 8553) least significant half
v_addc_co_u32 v31, s[6:7], 0, v31, s[6:7] // (op 8553) most significant half

s_add_u32 m0, m0, 4096 // (op 8553) 
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 8553) Load value direct to lds
v_add_co_u32 v30, s[6:7], v30, v28 // (op 8553) least significant half
v_addc_co_u32 v31, s[6:7], 0, v31, s[6:7] // (op 8553) most significant half

s_add_u32 m0, m0, 4096 // (op 8553) 
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 8553) Load value direct to lds







buffer_load_dword v25, v23, s[56:59], 0 offen // (op 8240) Load value
v_and_b32 v28, -8, v1 // (op 9180) 
v_sub_u32 v29, v1, v28 // (op 9180) 

v_lshlrev_b32 v28, 4, v29 // (op 9180) 

v_lshrrev_b32 v29, 3, v1 // (op 9180) 
v_lshl_add_u32 v30, v29, 7, v28 // (op 9180) 


s_nop 0
v_readfirstlane_b32 s0, v30 // (op 9180) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 8558) 
v_mov_b64 v[28:29], s[6:7] // (op 8558) convert


v_mov_b64 v[30:31], v[14:15] // (op 8558) 
s_add_u32 s1, 40960, s0 // (op 8558) 
s_mov_b32 m0, s1 // (op 8558) reg expression
s_mov_b32 s6, 128 // (op 8558) call()
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 8558) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v28 // (op 8558) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 8558) most significant half

s_add_u32 m0, m0, 4096 // (op 8558) 
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 8558) Load value direct to lds









buffer_load_ubyte v14, v26, s[60:63], 0 offen offset:4 // (op 8246) Load value

v_and_b32 v15, -8, v1 // (op 9136) 
v_sub_u32 v26, v1, v15 // (op 9136) 

v_lshlrev_b32 v15, 4, v26 // (op 9136) 

v_lshrrev_b32 v26, 3, v1 // (op 9136) 
v_lshl_add_u32 v28, v26, 7, v15 // (op 9136) 


s_nop 0
v_readfirstlane_b32 s0, v28 // (op 9136) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 5 // (op 8560) 
v_mov_b64 v[28:29], s[6:7] // (op 8560) convert


v_mov_b64 v[30:31], v[18:19] // (op 8560) 
s_add_u32 s1, 49152, s0 // (op 8560) 
s_mov_b32 m0, s1 // (op 8560) reg expression
s_mov_b32 s6, 128 // (op 8560) call()
buffer_load_dwordx4 v30, s[48:51], s6 offen offset: 0 lds // (op 8560) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v28 // (op 8560) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 8560) most significant half

s_add_u32 m0, m0, 4096 // (op 8560) 
buffer_load_dwordx4 v30, s[48:51], s6 offen offset: 0 lds // (op 8560) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v28 // (op 8560) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 8560) most significant half

s_add_u32 m0, m0, 4096 // (op 8560) 
buffer_load_dwordx4 v30, s[48:51], s6 offen offset: 0 lds // (op 8560) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v28 // (op 8560) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 8560) most significant half

s_add_u32 m0, m0, 4096 // (op 8560) 
buffer_load_dwordx4 v30, s[48:51], s6 offen offset: 0 lds // (op 8560) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v28 // (op 8560) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 8560) most significant half

s_add_u32 m0, m0, 4096 // (op 8560) 
buffer_load_dwordx4 v30, s[48:51], s6 offen offset: 0 lds // (op 8560) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v28 // (op 8560) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 8560) most significant half

s_add_u32 m0, m0, 4096 // (op 8560) 
buffer_load_dwordx4 v30, s[48:51], s6 offen offset: 0 lds // (op 8560) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v28 // (op 8560) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 8560) most significant half

s_add_u32 m0, m0, 4096 // (op 8560) 
buffer_load_dwordx4 v30, s[48:51], s6 offen offset: 0 lds // (op 8560) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v28 // (op 8560) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 8560) most significant half

s_add_u32 m0, m0, 4096 // (op 8560) 
buffer_load_dwordx4 v30, s[48:51], s6 offen offset: 0 lds // (op 8560) Load value direct to lds









buffer_load_dword v15, v23, s[56:59], 0 offen offset:4 // (op 8252) Load value

v_and_b32 v23, -4, v1 // (op 9069) 
v_sub_u32 v26, v1, v23 // (op 9069) 

v_and_b32 v23, 4294967292, v1 // (op 9069) 
v_add_u32 v18, v23, v26 // (op 9069) 





v_add_u32 v19, 81920, v18 // (op 8255) 
s_waitcnt vmcnt(21)
ds_write_b8 v19, v22 // (op 8255) Store local data 


v_lshlrev_b32 v23, 2, v1 // (op 9011) 



v_add_u32 v26, 82176, v23 // (op 8257) 
s_waitcnt vmcnt(12)
ds_write_b32 v26, v25 // (op 8257) Store local data 


s_barrier  // (op 9396) 



ds_read_b128 v[28:31], v4 // (op 8312) Load local data 
ds_read_b128 v[32:35], v4 offset:64 // (op 8312) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 9431) 



v_add_u32 v23, 81920, v5 // (op 8362) 
ds_read_u8 v26, v23 // (op 8362) Load local data 




ds_read_b128 v[36:39], v4 offset:4096 // (op 8317) Load local data 
ds_read_b128 v[40:43], v4 offset:4160 // (op 8317) Load local data 



v_add_u32 v23, 82048, v5 // (op 8367) 
ds_read_u8 v18, v23 // (op 8367) Load local data 

s_barrier  // (op 9380) 



ds_read_b128 v[44:47], v3 offset:8192 // (op 8272) Load local data 
ds_read_b128 v[48:51], v3 offset:8256 // (op 8272) Load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 9413) 



v_add_u32 v19, 82176, v0 // (op 8322) 
ds_read_u8 v23, v19 // (op 8322) Load local data 




ds_read_b128 v[52:55], v3 offset:12288 // (op 8277) Load local data 
ds_read_b128 v[56:59], v3 offset:12352 // (op 8277) Load local data 



v_add_u32 v19, 82304, v0 // (op 8327) 
ds_read_u8 v60, v19 // (op 8327) Load local data 




ds_read_b128 v[62:65], v3 offset:16384 // (op 8282) Load local data 
ds_read_b128 v[66:69], v3 offset:16448 // (op 8282) Load local data 



v_add_u32 v19, 82432, v0 // (op 8332) 
ds_read_u8 v61, v19 // (op 8332) Load local data 




ds_read_b128 v[70:73], v3 offset:20480 // (op 8287) Load local data 
ds_read_b128 v[74:77], v3 offset:20544 // (op 8287) Load local data 



v_add_u32 v19, 82560, v0 // (op 8337) 
ds_read_u8 v78, v19 // (op 8337) Load local data 




ds_read_b128 v[80:83], v3 offset:24576 // (op 8292) Load local data 
ds_read_b128 v[84:87], v3 offset:24640 // (op 8292) Load local data 



v_add_u32 v19, 82688, v0 // (op 8342) 
ds_read_u8 v79, v19 // (op 8342) Load local data 




ds_read_b128 v[88:91], v3 offset:28672 // (op 8297) Load local data 
ds_read_b128 v[92:95], v3 offset:28736 // (op 8297) Load local data 



v_add_u32 v19, 82816, v0 // (op 8347) 
ds_read_u8 v96, v19 // (op 8347) Load local data 




ds_read_b128 v[98:101], v3 offset:32768 // (op 8302) Load local data 
ds_read_b128 v[102:105], v3 offset:32832 // (op 8302) Load local data 



v_add_u32 v19, 82944, v0 // (op 8352) 
ds_read_u8 v97, v19 // (op 8352) Load local data 




ds_read_b128 v[106:109], v3 offset:36864 // (op 8307) Load local data 
ds_read_b128 v[110:113], v3 offset:36928 // (op 8307) Load local data 



v_add_u32 v19, 83072, v0 // (op 8357) 
ds_read_u8 v114, v19 // (op 8357) Load local data 

v_and_b32 v19, -8, v1 // (op 8881) 
v_sub_u32 v115, v1, v19 // (op 8881) 

v_lshlrev_b32 v19, 4, v115 // (op 8881) 

v_lshrrev_b32 v115, 3, v1 // (op 8881) 
v_lshl_add_u32 v116, v115, 7, v19 // (op 8881) 


s_nop 0
v_readfirstlane_b32 s0, v116 // (op 8881) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v19, -8, v1 // (op 8938) 
v_sub_u32 v115, v1, v19 // (op 8938) 

v_lshlrev_b32 v19, 4, v115 // (op 8938) 

v_lshrrev_b32 v115, 3, v1 // (op 8938) 
v_lshl_add_u32 v116, v115, 7, v19 // (op 8938) 


s_nop 0
v_readfirstlane_b32 s1, v116 // (op 8938) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v19, 2, v1 // (op 8995) 
v_and_b32 v115, -4, v1 // (op 9053) 
v_sub_u32 v116, v1, v115 // (op 9053) 

v_and_b32 v115, 4294967292, v1 // (op 9053) 
v_add_u32 v117, v115, v116 // (op 9053) 


v_and_b32 v115, -8, v1 // (op 9121) 
v_sub_u32 v116, v1, v115 // (op 9121) 

v_lshlrev_b32 v115, 4, v116 // (op 9121) 

v_lshrrev_b32 v116, 3, v1 // (op 9121) 
v_lshl_add_u32 v118, v116, 7, v115 // (op 9121) 


s_nop 0
v_readfirstlane_b32 s6, v118 // (op 9121) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v115, -8, v1 // (op 9165) 
v_sub_u32 v116, v1, v115 // (op 9165) 

v_lshlrev_b32 v115, 4, v116 // (op 9165) 

v_lshrrev_b32 v116, 3, v1 // (op 9165) 
v_lshl_add_u32 v118, v116, 7, v115 // (op 9165) 


s_nop 0
v_readfirstlane_b32 s7, v118 // (op 9165) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v115, 2, v1 // (op 9209) 
v_and_b32 v116, -4, v1 // (op 9238) 
v_sub_u32 v118, v1, v116 // (op 9238) 

v_and_b32 v116, 4294967292, v1 // (op 9238) 
v_add_u32 v119, v116, v118 // (op 9238) 


s_mov_b32 s10, 0 // (op 2905) call()
s_ashr_i64 s[14:15], s[36:37], 7 // (op 2904) 
s_mov_b32 s11, s14 // (op 2904) convert

s_and_b32 s14, s11, 4294967294 // (op 2904) 

s_cmp_lt_u32 s10, s14 // (op 2904) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ForLoopBottom_KLoop_2904 // (op 2904) Condition: Top (jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ForLoopBottom_KLoop_2904 if false)
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ForLoopTop_KLoop_2904:
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[44:51], v[28:35], a[0:3], v23, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 88) 
s_barrier  // (op 9399) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[44:51], v[36:43], a[4:7], v23, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 376) 
s_add_u32 s10, s10, 2 // (op 2906) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[52:59], v[28:35], a[8:11], v60, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 673) 
s_lshl_b64 s[14:15], s[24:25], 5 // (op 8520) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[52:59], v[36:43], a[12:15], v60, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 627) 
v_mov_b64 v[120:121], s[14:15] // (op 8520) convert


v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[62:69], v[28:35], a[16:19], v61, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1059) 
v_mov_b64 v[122:123], v[16:17] // (op 8520) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[62:69], v[36:43], a[20:23], v61, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1013) 
v_mov_b64 v[124:125], v[122:123] // (op 8520) 
s_waitcnt lgkmcnt(12)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[70:77], v[28:35], a[24:27], v78, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1445) 
s_mov_b32 m0, s1 // (op 8520) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[70:77], v[36:43], a[28:31], v78, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1399) 
s_mov_b32 s11, 256 // (op 8520) call()
buffer_load_dwordx4 v124, s[52:55], s11 offen offset: 0 lds // (op 8520) Load value direct to lds
v_add_co_u32 v124, s[14:15], v124, v120 // (op 8520) least significant half
v_addc_co_u32 v125, s[14:15], 0, v125, s[14:15] // (op 8520) most significant half

s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[80:87], v[28:35], a[32:35], v79, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1831) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[80:87], v[36:43], a[36:39], v79, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1785) 
s_add_u32 m0, m0, 4096 // (op 8520) 
buffer_load_dwordx4 v124, s[52:55], s11 offen offset: 0 lds // (op 8520) Load value direct to lds



s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[88:95], v[28:35], a[40:43], v96, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2217) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[88:95], v[36:43], a[44:47], v96, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2171) 
v_mov_b32 v116, v27 // (op 25) 

buffer_load_ubyte v22, v116, s[60:63], 0 offen offset:8 // (op 25) Load value
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[98:105], v[28:35], a[48:51], v97, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2603) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[98:105], v[36:43], a[52:55], v97, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2557) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 9383) 

s_lshl_b64 s[14:15], s[12:13], 5 // (op 8515) 
v_mov_b64 v[120:121], s[14:15] // (op 8515) convert


v_mov_b64 v[124:125], v[20:21] // (op 8515) 
v_mov_b64 v[126:127], v[124:125] // (op 8515) 
s_add_u32 s11, 8192, s0 // (op 8515) 
s_mov_b32 m0, s11 // (op 8515) reg expression
s_mov_b32 s14, 256 // (op 8515) call()
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[106:113], v[28:35], a[56:59], v114, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2989) 
buffer_load_dwordx4 v126, s[48:51], s14 offen offset: 0 lds // (op 8515) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[106:113], v[36:43], a[60:63], v114, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2943) 
v_add_co_u32 v126, s[18:19], v126, v120 // (op 8515) least significant half
v_addc_co_u32 v127, s[18:19], 0, v127, s[18:19] // (op 8515) most significant half

s_add_u32 m0, m0, 4096 // (op 8515) 
buffer_load_dwordx4 v126, s[48:51], s14 offen offset: 0 lds // (op 8515) Load value direct to lds
v_add_co_u32 v126, s[18:19], v126, v120 // (op 8515) least significant half
v_addc_co_u32 v127, s[18:19], 0, v127, s[18:19] // (op 8515) most significant half

s_add_u32 m0, m0, 4096 // (op 8515) 
buffer_load_dwordx4 v126, s[48:51], s14 offen offset: 0 lds // (op 8515) Load value direct to lds
v_add_co_u32 v126, s[18:19], v126, v120 // (op 8515) least significant half
v_addc_co_u32 v127, s[18:19], 0, v127, s[18:19] // (op 8515) most significant half

s_add_u32 m0, m0, 4096 // (op 8515) 
buffer_load_dwordx4 v126, s[48:51], s14 offen offset: 0 lds // (op 8515) Load value direct to lds
v_add_co_u32 v126, s[18:19], v126, v120 // (op 8515) least significant half
v_addc_co_u32 v127, s[18:19], 0, v127, s[18:19] // (op 8515) most significant half

s_add_u32 m0, m0, 4096 // (op 8515) 
buffer_load_dwordx4 v126, s[48:51], s14 offen offset: 0 lds // (op 8515) Load value direct to lds
v_add_co_u32 v126, s[18:19], v126, v120 // (op 8515) least significant half
v_addc_co_u32 v127, s[18:19], 0, v127, s[18:19] // (op 8515) most significant half

s_add_u32 m0, m0, 4096 // (op 8515) 
buffer_load_dwordx4 v126, s[48:51], s14 offen offset: 0 lds // (op 8515) Load value direct to lds
v_add_co_u32 v126, s[18:19], v126, v120 // (op 8515) least significant half
v_addc_co_u32 v127, s[18:19], 0, v127, s[18:19] // (op 8515) most significant half

s_add_u32 m0, m0, 4096 // (op 8515) 
buffer_load_dwordx4 v126, s[48:51], s14 offen offset: 0 lds // (op 8515) Load value direct to lds
v_add_co_u32 v126, s[18:19], v126, v120 // (op 8515) least significant half
v_addc_co_u32 v127, s[18:19], 0, v127, s[18:19] // (op 8515) most significant half

s_add_u32 m0, m0, 4096 // (op 8515) 
buffer_load_dwordx4 v126, s[48:51], s14 offen offset: 0 lds // (op 8515) Load value direct to lds






v_mov_b32 v118, v24 // (op 17) 

buffer_load_dword v25, v118, s[56:59], 0 offen offset:8 // (op 17) Load value
s_barrier  // (op 9471) 



v_add_u32 v120, 83200, v119 // (op 255) 
s_waitcnt vmcnt(21)
ds_write_b8 v120, v14 // (op 255) Store local data 

s_barrier  // (op 9465) 



v_add_u32 v120, 83456, v115 // (op 254) 
s_waitcnt vmcnt(12)
ds_write_b32 v120, v15 // (op 254) Store local data 

s_barrier  // (op 9454) 



ds_read_b128 v[126:129], v8 offset:40960 // (op 247) Load local data 
ds_read_b128 v[130:133], v8 offset:41024 // (op 247) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 9468) 



v_add_u32 v120, 83200, v9 // (op 251) 
ds_read_u8 v121, v120 // (op 251) Load local data 




ds_read_b128 v[134:137], v8 offset:45056 // (op 381) Load local data 
ds_read_b128 v[138:141], v8 offset:45120 // (op 381) Load local data 



v_add_u32 v120, 83328, v9 // (op 385) 
ds_read_u8 v142, v120 // (op 385) Load local data 

s_barrier  // (op 9448) 
s_barrier  // (op 9512) 



ds_read_b128 v[44:47], v7 offset:49152 // (op 245) Load local data 
ds_read_b128 v[48:51], v7 offset:49216 // (op 245) Load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 9461) 



v_add_u32 v120, 83456, v6 // (op 249) 
ds_read_u8 v23, v120 // (op 249) Load local data 




ds_read_b128 v[52:55], v7 offset:53248 // (op 630) Load local data 
ds_read_b128 v[56:59], v7 offset:53312 // (op 630) Load local data 



v_add_u32 v120, 83584, v6 // (op 633) 
ds_read_u8 v60, v120 // (op 633) Load local data 




ds_read_b128 v[62:65], v7 offset:57344 // (op 1016) Load local data 
ds_read_b128 v[66:69], v7 offset:57408 // (op 1016) Load local data 



v_add_u32 v120, 83712, v6 // (op 1019) 
ds_read_u8 v61, v120 // (op 1019) Load local data 




ds_read_b128 v[70:73], v7 offset:61440 // (op 1402) Load local data 
ds_read_b128 v[74:77], v7 offset:61504 // (op 1402) Load local data 



v_add_u32 v120, 83840, v6 // (op 1405) 
ds_read_u8 v78, v120 // (op 1405) Load local data 




v_add_u32 v120, 65536, v7 // (op 1788) 
ds_read_b128 v[80:83], v120 // (op 1788) Load local data 

v_add_u32 v120, 65600, v7 // (op 1788) 
ds_read_b128 v[84:87], v120 // (op 1788) Load local data 




v_add_u32 v120, 83968, v6 // (op 1791) 
ds_read_u8 v79, v120 // (op 1791) Load local data 




v_add_u32 v120, 69632, v7 // (op 2174) 
ds_read_b128 v[88:91], v120 // (op 2174) Load local data 

v_add_u32 v120, 69696, v7 // (op 2174) 
ds_read_b128 v[92:95], v120 // (op 2174) Load local data 




v_add_u32 v120, 84096, v6 // (op 2177) 
ds_read_u8 v96, v120 // (op 2177) Load local data 




v_add_u32 v120, 73728, v7 // (op 2560) 
ds_read_b128 v[98:101], v120 // (op 2560) Load local data 

v_add_u32 v120, 73792, v7 // (op 2560) 
ds_read_b128 v[102:105], v120 // (op 2560) Load local data 




v_add_u32 v120, 84224, v6 // (op 2563) 
ds_read_u8 v97, v120 // (op 2563) Load local data 




v_add_u32 v120, 77824, v7 // (op 2946) 
ds_read_b128 v[106:109], v120 // (op 2946) Load local data 

v_add_u32 v120, 77888, v7 // (op 2946) 
ds_read_b128 v[110:113], v120 // (op 2946) Load local data 




v_add_u32 v120, 84352, v6 // (op 2949) 
ds_read_u8 v114, v120 // (op 2949) Load local data 

s_barrier  // (op 9558) 
s_waitcnt lgkmcnt(15)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[44:51], v[126:133], a[0:3], v23, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 252) 
s_barrier  // (op 9458) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[44:51], v[134:141], a[4:7], v23, v142 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 377) 
s_lshl_b64 s[14:15], s[24:25], 5 // (op 8543) 
v_mov_b64 v[144:145], s[14:15] // (op 8543) convert


v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[52:59], v[126:133], a[8:11], v60, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 674) 
v_mov_b64 v[146:147], v[122:123] // (op 8543) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[52:59], v[134:141], a[12:15], v60, v142 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 628) 
s_add_u32 s11, 40960, s7 // (op 8543) 
s_mov_b32 m0, s11 // (op 8543) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[62:69], v[126:133], a[16:19], v61, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1060) 
s_mov_b32 s14, 384 // (op 8543) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[62:69], v[134:141], a[20:23], v61, v142 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1014) 
buffer_load_dwordx4 v146, s[52:55], s14 offen offset: 0 lds // (op 8543) Load value direct to lds
v_add_co_u32 v146, s[18:19], v146, v144 // (op 8543) least significant half
s_waitcnt lgkmcnt(12)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[70:77], v[126:133], a[24:27], v78, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1446) 
v_addc_co_u32 v147, s[18:19], 0, v147, s[18:19] // (op 8543) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[70:77], v[134:141], a[28:31], v78, v142 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1400) 
s_add_u32 m0, m0, 4096 // (op 8543) 
buffer_load_dwordx4 v146, s[52:55], s14 offen offset: 0 lds // (op 8543) Load value direct to lds




s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[80:87], v[126:133], a[32:35], v79, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1832) 




v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[80:87], v[134:141], a[36:39], v79, v142 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1786) 
buffer_load_ubyte v14, v116, s[60:63], 0 offen offset:12 // (op 258) Load value

v_add_co_u32 v16, vcc, 256, v16 // (op 8689) least significant half
v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 8689) most significant half
v_add_u32 v27, 8, v27 // (op 8817) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[88:95], v[126:133], a[40:43], v96, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2218) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 9451) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[88:95], v[134:141], a[44:47], v96, v142 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2172) 
s_lshl_b64 s[14:15], s[12:13], 5 // (op 8538) 
v_mov_b64 v[122:123], s[14:15] // (op 8538) convert


v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[98:105], v[126:133], a[48:51], v97, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2604) 
v_mov_b64 v[144:145], v[124:125] // (op 8538) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[98:105], v[134:141], a[52:55], v97, v142 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2558) 
s_add_u32 s11, 49152, s6 // (op 8538) 
s_mov_b32 m0, s11 // (op 8538) reg expression
s_mov_b32 s14, 384 // (op 8538) call()
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[106:113], v[126:133], a[56:59], v114, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2990) 
buffer_load_dwordx4 v144, s[48:51], s14 offen offset: 0 lds // (op 8538) Load value direct to lds


v_add_co_u32 v144, s[18:19], v144, v122 // (op 8538) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[106:113], v[134:141], a[60:63], v114, v142 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2944) 
v_addc_co_u32 v145, s[18:19], 0, v145, s[18:19] // (op 8538) most significant half



s_add_u32 m0, m0, 4096 // (op 8538) 
buffer_load_dwordx4 v144, s[48:51], s14 offen offset: 0 lds // (op 8538) Load value direct to lds
v_add_co_u32 v144, s[18:19], v144, v122 // (op 8538) least significant half
v_addc_co_u32 v145, s[18:19], 0, v145, s[18:19] // (op 8538) most significant half

s_add_u32 m0, m0, 4096 // (op 8538) 
buffer_load_dwordx4 v144, s[48:51], s14 offen offset: 0 lds // (op 8538) Load value direct to lds
v_add_co_u32 v144, s[18:19], v144, v122 // (op 8538) least significant half
v_addc_co_u32 v145, s[18:19], 0, v145, s[18:19] // (op 8538) most significant half

s_add_u32 m0, m0, 4096 // (op 8538) 
buffer_load_dwordx4 v144, s[48:51], s14 offen offset: 0 lds // (op 8538) Load value direct to lds
v_add_co_u32 v144, s[18:19], v144, v122 // (op 8538) least significant half
v_addc_co_u32 v145, s[18:19], 0, v145, s[18:19] // (op 8538) most significant half

s_add_u32 m0, m0, 4096 // (op 8538) 
buffer_load_dwordx4 v144, s[48:51], s14 offen offset: 0 lds // (op 8538) Load value direct to lds
v_add_co_u32 v144, s[18:19], v144, v122 // (op 8538) least significant half
v_addc_co_u32 v145, s[18:19], 0, v145, s[18:19] // (op 8538) most significant half

s_add_u32 m0, m0, 4096 // (op 8538) 
buffer_load_dwordx4 v144, s[48:51], s14 offen offset: 0 lds // (op 8538) Load value direct to lds
v_add_co_u32 v144, s[18:19], v144, v122 // (op 8538) least significant half
v_addc_co_u32 v145, s[18:19], 0, v145, s[18:19] // (op 8538) most significant half

s_add_u32 m0, m0, 4096 // (op 8538) 
buffer_load_dwordx4 v144, s[48:51], s14 offen offset: 0 lds // (op 8538) Load value direct to lds
v_add_co_u32 v144, s[18:19], v144, v122 // (op 8538) least significant half
v_addc_co_u32 v145, s[18:19], 0, v145, s[18:19] // (op 8538) most significant half

s_add_u32 m0, m0, 4096 // (op 8538) 
buffer_load_dwordx4 v144, s[48:51], s14 offen offset: 0 lds // (op 8538) Load value direct to lds








v_add_co_u32 v20, vcc, 256, v20 // (op 8625) least significant half
v_addc_co_u32 v21, vcc, 0, v21, vcc // (op 8625) most significant half
buffer_load_dword v15, v118, s[56:59], 0 offen offset:12 // (op 257) Load value

s_barrier  // (op 9434) 
v_add_u32 v24, 8, v24 // (op 8753) 



v_add_u32 v116, 81920, v117 // (op 30) 
s_waitcnt vmcnt(21)
ds_write_b8 v116, v22 // (op 30) Store local data 

s_barrier  // (op 9416) 



v_add_u32 v116, 82176, v19 // (op 21) 
s_waitcnt vmcnt(12)
ds_write_b32 v116, v25 // (op 21) Store local data 

s_barrier  // (op 9405) 



ds_read_b128 v[28:31], v4 // (op 75) Load local data 
ds_read_b128 v[32:35], v4 offset:64 // (op 75) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 9441) 



v_add_u32 v116, 81920, v5 // (op 85) 
ds_read_u8 v26, v116 // (op 85) Load local data 




ds_read_b128 v[36:39], v4 offset:4096 // (op 371) Load local data 
ds_read_b128 v[40:43], v4 offset:4160 // (op 371) Load local data 



v_add_u32 v116, 82048, v5 // (op 375) 
ds_read_u8 v18, v116 // (op 375) Load local data 

s_barrier  // (op 9389) 



ds_read_b128 v[44:47], v3 offset:8192 // (op 70) Load local data 
ds_read_b128 v[48:51], v3 offset:8256 // (op 70) Load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 9423) 



v_add_u32 v116, 82176, v0 // (op 80) 
ds_read_u8 v23, v116 // (op 80) Load local data 




ds_read_b128 v[52:55], v3 offset:12288 // (op 622) Load local data 
ds_read_b128 v[56:59], v3 offset:12352 // (op 622) Load local data 



v_add_u32 v116, 82304, v0 // (op 625) 
ds_read_u8 v60, v116 // (op 625) Load local data 




ds_read_b128 v[62:65], v3 offset:16384 // (op 1008) Load local data 
ds_read_b128 v[66:69], v3 offset:16448 // (op 1008) Load local data 



v_add_u32 v116, 82432, v0 // (op 1011) 
ds_read_u8 v61, v116 // (op 1011) Load local data 




ds_read_b128 v[70:73], v3 offset:20480 // (op 1394) Load local data 
ds_read_b128 v[74:77], v3 offset:20544 // (op 1394) Load local data 



v_add_u32 v116, 82560, v0 // (op 1397) 
ds_read_u8 v78, v116 // (op 1397) Load local data 




ds_read_b128 v[80:83], v3 offset:24576 // (op 1780) Load local data 
ds_read_b128 v[84:87], v3 offset:24640 // (op 1780) Load local data 



v_add_u32 v116, 82688, v0 // (op 1783) 
ds_read_u8 v79, v116 // (op 1783) Load local data 




ds_read_b128 v[88:91], v3 offset:28672 // (op 2166) Load local data 
ds_read_b128 v[92:95], v3 offset:28736 // (op 2166) Load local data 



v_add_u32 v116, 82816, v0 // (op 2169) 
ds_read_u8 v96, v116 // (op 2169) Load local data 




ds_read_b128 v[98:101], v3 offset:32768 // (op 2552) Load local data 
ds_read_b128 v[102:105], v3 offset:32832 // (op 2552) Load local data 



v_add_u32 v116, 82944, v0 // (op 2555) 
ds_read_u8 v97, v116 // (op 2555) Load local data 




ds_read_b128 v[106:109], v3 offset:36864 // (op 2938) Load local data 
ds_read_b128 v[110:113], v3 offset:36928 // (op 2938) Load local data 



v_add_u32 v116, 83072, v0 // (op 2941) 
ds_read_u8 v114, v116 // (op 2941) Load local data 

s_ashr_i64 s[14:15], s[36:37], 7 // (op 2904) 
s_mov_b32 s11, s14 // (op 2904) convert

s_and_b32 s14, s11, 4294967294 // (op 2904) 

s_cmp_lt_u32 s10, s14 // (op 2904) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ForLoopTop_KLoop_2904 // (op 2904) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ForLoopTop_KLoop_2904 if true)
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ForLoopBottom_KLoop_2904:

























s_barrier  // (op 9548) 
v_lshrrev_b32 v16, 3, v1 // (op 8573) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 8573) 
v_lshl_add_u32 v17, s16, 8, v16 // (op 8573) 

s_lshr_b32 s6, s0, 1 // (op 8573) 

v_and_b32 v16, -8, v1 // (op 8573) 
v_sub_u32 v18, v1, v16 // (op 8573) 

v_lshlrev_b32 v16, 4, v18 // (op 8573) 

v_lshl_add_u32 v18, s6, 8, v16 // (op 8573) 


v_mov_b32 v69, 0 // (op 8573) convert
v_mov_b32 v68, v18 // (op 8573) convert

v_mul_lo_u32 v19, v17, s13 // (op 8573) most significant: low of low * high
v_mul_hi_u32 v16, v17, s12 // (op 8573) most significant: high of low * low
v_mul_lo_u32 v18, v17, s12 // (op 8573) least significant: low of low * low
v_add_u32 v19, v19, v16 // (op 8573) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v68 // (op 8573) least significant half
v_addc_co_u32 v19, s[0:1], v19, v69, s[0:1] // (op 8573) most significant half



s_mul_i32 s1, s12, s23 // (op 8575) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 8575) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 8575) least significant
s_mul_hi_u32 s7, s12, s22 // (op 8575) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8575) most significant: sum
s_add_u32 s1, s1, s7 // (op 8575) most significant: sum


s_mov_b32 s48, s20 // (op 8575) reg expression
s_mov_b32 s49, s21 // (op 8575) reg expression
s_mov_b32 s50, s0 // (op 8575) reg expression
s_mov_b32 s51, 131072 // (op 8575) call()

v_and_b32 v16, -8, v1 // (op 8655) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 8655) 
v_sub_u32 v17, v1, v16 // (op 8655) 

s_lshr_b32 s6, s0, 1 // (op 8655) 

v_lshlrev_b32 v16, 4, v17 // (op 8655) 

v_lshl_add_u32 v17, s6, 8, v16 // (op 8655) 


v_mov_b32 v21, 0 // (op 8655) convert
v_mov_b32 v20, v17 // (op 8655) convert

v_lshrrev_b32 v16, 3, v1 // (op 8655) 
v_lshl_add_u32 v17, s17, 6, v16 // (op 8655) 

v_mul_lo_u32 v23, v17, s25 // (op 8655) most significant: low of low * high
v_mul_hi_u32 v16, v17, s24 // (op 8655) most significant: high of low * low
v_mul_lo_u32 v22, v17, s24 // (op 8655) least significant: low of low * low
v_add_u32 v23, v23, v16 // (op 8655) most significant: sum


v_add_co_u32 v16, s[0:1], v20, v22 // (op 8655) least significant half
v_addc_co_u32 v17, s[0:1], v21, v23, s[0:1] // (op 8655) most significant half



s_mul_i32 s1, s24, s9 // (op 8657) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 8657) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 8657) least significant
s_mul_hi_u32 s7, s24, s8 // (op 8657) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8657) most significant: sum
s_add_u32 s1, s1, s7 // (op 8657) most significant: sum


s_mov_b32 s52, s26 // (op 8657) reg expression
s_mov_b32 s53, s27 // (op 8657) reg expression
s_mov_b32 s54, s0 // (op 8657) reg expression
s_mov_b32 s55, 131072 // (op 8657) call()

v_lshl_add_u32 v20, s16, 8, v1 // (op 8719) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 8719) 
s_lshr_b32 s6, s0, 1 // (op 8719) 

s_lshl_b32 s7, s6, 3 // (op 8719) 

v_mul_lo_u32 v21, v20, s28 // (op 8719) 
v_add_u32 v21, v21, s7 // (op 8719) 


s_mul_i32 s1, s28, s23 // (op 8721) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 8721) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 8721) least significant
s_mul_hi_u32 s7, s28, s22 // (op 8721) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8721) most significant: sum
s_add_u32 s1, s1, s7 // (op 8721) most significant: sum


s_mov_b32 s56, s30 // (op 8721) reg expression
s_mov_b32 s57, s31 // (op 8721) reg expression
s_mov_b32 s58, s0 // (op 8721) reg expression
s_mov_b32 s59, 131072 // (op 8721) call()

v_and_b32 v20, -4, v1 // (op 8783) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 8783) 
v_sub_u32 v22, v1, v20 // (op 8783) 

s_lshr_b32 s6, s0, 1 // (op 8783) 

v_lshrrev_b32 v20, 2, v1 // (op 8783) 
v_lshl_add_u32 v23, s6, 3, v22 // (op 8783) 


v_lshl_add_u32 v22, s17, 6, v20 // (op 8783) 

v_mul_lo_u32 v20, v22, s32 // (op 8783) 

v_add_u32 v22, v23, v20 // (op 8783) 


s_mul_i32 s1, s32, s9 // (op 8785) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 8785) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 8785) least significant
s_mul_hi_u32 s7, s32, s8 // (op 8785) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8785) most significant: sum
s_add_u32 s1, s1, s7 // (op 8785) most significant: sum


s_mov_b32 s60, s34 // (op 8785) reg expression
s_mov_b32 s61, s35 // (op 8785) reg expression
s_mov_b32 s62, s0 // (op 8785) reg expression
s_mov_b32 s63, 131072 // (op 8785) call()

v_and_b32 v20, -8, v1 // (op 8866) 
v_sub_u32 v23, v1, v20 // (op 8866) 

v_lshlrev_b32 v20, 4, v23 // (op 8866) 

v_lshrrev_b32 v23, 3, v1 // (op 8866) 
v_lshl_add_u32 v24, v23, 7, v20 // (op 8866) 


s_nop 0
v_readfirstlane_b32 s0, v24 // (op 8866) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v20, -8, v1 // (op 8925) 
v_sub_u32 v23, v1, v20 // (op 8925) 

v_lshlrev_b32 v20, 4, v23 // (op 8925) 

v_lshrrev_b32 v23, 3, v1 // (op 8925) 
v_lshl_add_u32 v24, v23, 7, v20 // (op 8925) 


s_nop 0
v_readfirstlane_b32 s1, v24 // (op 8925) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v20, 2, v1 // (op 8982) 
v_and_b32 v23, -4, v1 // (op 9040) 
v_sub_u32 v24, v1, v23 // (op 9040) 

v_and_b32 v23, 4294967292, v1 // (op 9040) 
v_add_u32 v25, v23, v24 // (op 9040) 


s_ashr_i64 s[6:7], s[36:37], 7 // (op 2879) 
s_and_b32 s10, s6, 4294967294 // (op 2879) 

s_ashr_i64 s[6:7], s[36:37], 7 // (op 2878) 
s_mov_b32 s11, s6 // (op 2878) convert

s_cmp_lt_u32 s10, s11 // (op 2878) 













s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ForLoopBottom_KLoopTail_2878 // (op 2878) Condition: Top (jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ForLoopBottom_KLoopTail_2878 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ForLoopTop_KLoopTail_2878:
s_barrier  // (op 9386) 
s_add_u32 s10, s10, 1 // (op 2880) 

s_lshl_b64 s[6:7], s[12:13], 5 // (op 8525) 
v_mov_b64 v[14:15], s[6:7] // (op 8525) convert


v_mov_b64 v[26:27], v[18:19] // (op 8525) 
v_mov_b64 v[28:29], v[26:27] // (op 8525) 
s_add_u32 s11, 8192, s0 // (op 8525) 
s_mov_b32 m0, s11 // (op 8525) reg expression
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 8525) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v14 // (op 8525) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 8525) most significant half

s_add_u32 m0, m0, 4096 // (op 8525) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 8525) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v14 // (op 8525) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 8525) most significant half

s_add_u32 m0, m0, 4096 // (op 8525) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 8525) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v14 // (op 8525) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 8525) most significant half

s_add_u32 m0, m0, 4096 // (op 8525) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 8525) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v14 // (op 8525) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 8525) most significant half

s_add_u32 m0, m0, 4096 // (op 8525) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 8525) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v14 // (op 8525) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 8525) most significant half

s_add_u32 m0, m0, 4096 // (op 8525) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 8525) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v14 // (op 8525) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 8525) most significant half

s_add_u32 m0, m0, 4096 // (op 8525) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 8525) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v14 // (op 8525) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 8525) most significant half

s_add_u32 m0, m0, 4096 // (op 8525) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 8525) Load value direct to lds




v_add_co_u32 v18, vcc, 128, v18 // (op 8563) least significant half
v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 8563) most significant half
s_barrier  // (op 9402) 

s_lshl_b64 s[6:7], s[24:25], 5 // (op 8531) 
v_mov_b64 v[14:15], s[6:7] // (op 8531) convert


v_mov_b64 v[26:27], v[16:17] // (op 8531) 
v_mov_b64 v[28:29], v[26:27] // (op 8531) 
s_mov_b32 m0, s1 // (op 8531) reg expression
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 8531) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v14 // (op 8531) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 8531) most significant half

s_add_u32 m0, m0, 4096 // (op 8531) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 8531) Load value direct to lds



v_add_co_u32 v16, vcc, 128, v16 // (op 8647) least significant half
v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 8647) most significant half


v_mov_b32 v14, v21 // (op 191) 

buffer_load_dword v15, v14, s[56:59], 0 offen // (op 191) Load value

v_add_u32 v21, 4, v21 // (op 8711) 


s_barrier  // (op 9419) 
v_mov_b32 v14, v22 // (op 192) 




v_add_u32 v23, 82176, v20 // (op 188) 
buffer_load_ubyte v24, v14, s[60:63], 0 offen offset: 0 // (op 192) Load value

v_add_u32 v22, 4, v22 // (op 8775) 
s_waitcnt vmcnt(1)
ds_write_b32 v23, v15 // (op 188) Store local data 


s_barrier  // (op 9392) 



s_barrier  // (op 9437) 
ds_read_b128 v[26:29], v3 offset:8192 // (op 179) Load local data 



ds_read_b128 v[30:33], v3 offset:8256 // (op 179) Load local data 
v_add_u32 v23, 81920, v25 // (op 189) 
s_barrier  // (op 9409) 
s_waitcnt vmcnt(0)
ds_write_b8 v23, v24 // (op 189) Store local data 





ds_read_b128 v[34:37], v4 // (op 181) Load local data 
ds_read_b128 v[38:41], v4 offset:64 // (op 181) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 9427) 



v_add_u32 v14, 82176, v0 // (op 183) 
ds_read_u8 v15, v14 // (op 183) Load local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 9444) 



v_add_u32 v14, 81920, v5 // (op 185) 
ds_read_u8 v23, v14 // (op 185) Load local data 




ds_read_b128 v[42:45], v4 offset:4096 // (op 350) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:33], v[34:41], a[0:3], v15, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 186) 
ds_read_b128 v[46:49], v4 offset:4160 // (op 350) Load local data 



v_add_u32 v14, 82048, v5 // (op 354) 
ds_read_u8 v24, v14 // (op 354) Load local data 




ds_read_b128 v[50:53], v3 offset:12288 // (op 568) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[26:33], v[42:49], a[4:7], v15, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 355) 
ds_read_b128 v[54:57], v3 offset:12352 // (op 568) Load local data 





v_add_u32 v14, 82304, v0 // (op 571) 
ds_read_u8 v15, v14 // (op 571) Load local data 




ds_read_b128 v[26:29], v3 offset:16384 // (op 954) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[50:57], v[34:41], a[8:11], v15, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 573) 
ds_read_b128 v[30:33], v3 offset:16448 // (op 954) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[50:57], v[42:49], a[12:15], v15, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 604) 
v_add_u32 v14, 82432, v0 // (op 957) 


ds_read_u8 v15, v14 // (op 957) Load local data 




ds_read_b128 v[50:53], v3 offset:20480 // (op 1340) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[26:33], v[34:41], a[16:19], v15, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 959) 
ds_read_b128 v[54:57], v3 offset:20544 // (op 1340) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[26:33], v[42:49], a[20:23], v15, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 990) 
v_add_u32 v14, 82560, v0 // (op 1343) 


ds_read_u8 v15, v14 // (op 1343) Load local data 




ds_read_b128 v[26:29], v3 offset:24576 // (op 1726) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[50:57], v[34:41], a[24:27], v15, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1345) 
ds_read_b128 v[30:33], v3 offset:24640 // (op 1726) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[50:57], v[42:49], a[28:31], v15, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1376) 
v_add_u32 v14, 82688, v0 // (op 1729) 


ds_read_u8 v15, v14 // (op 1729) Load local data 




ds_read_b128 v[50:53], v3 offset:28672 // (op 2112) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[26:33], v[34:41], a[32:35], v15, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1731) 
ds_read_b128 v[54:57], v3 offset:28736 // (op 2112) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[26:33], v[42:49], a[36:39], v15, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1762) 
v_add_u32 v14, 82816, v0 // (op 2115) 


ds_read_u8 v15, v14 // (op 2115) Load local data 




ds_read_b128 v[26:29], v3 offset:32768 // (op 2498) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[50:57], v[34:41], a[40:43], v15, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2117) 
ds_read_b128 v[30:33], v3 offset:32832 // (op 2498) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[50:57], v[42:49], a[44:47], v15, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2148) 
v_add_u32 v14, 82944, v0 // (op 2501) 


ds_read_u8 v15, v14 // (op 2501) Load local data 




ds_read_b128 v[50:53], v3 offset:36864 // (op 2884) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[26:33], v[34:41], a[48:51], v15, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2503) 
ds_read_b128 v[54:57], v3 offset:36928 // (op 2884) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[26:33], v[42:49], a[52:55], v15, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2534) 
v_add_u32 v14, 83072, v0 // (op 2887) 


ds_read_u8 v15, v14 // (op 2887) Load local data 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[50:57], v[34:41], a[56:59], v15, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2889) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[50:57], v[42:49], a[60:63], v15, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2920) 




s_ashr_i64 s[6:7], s[36:37], 7 // (op 2878) 
s_mov_b32 s11, s6 // (op 2878) convert

s_cmp_lt_u32 s10, s11 // (op 2878) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ForLoopTop_KLoopTail_2878 // (op 2878) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ForLoopTop_KLoopTail_2878 if true)
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ForLoopBottom_KLoopTail_2878:























s_barrier  // (op 9540) 









v_cmp_eq_f32 vcc, 0.00000, v10 // (op 133) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ConditionalFalse_0_beta_133 // (op 133) Condition: False, jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ConditionalFalse_0_beta_133
v_accvgpr_read v6, a0 // (op 50) 
v_accvgpr_read v7, a1 // (op 50) 
v_accvgpr_read v8, a2 // (op 50) 
v_accvgpr_read v9, a3 // (op 50) 
v_mul_f32 v14, v11, v6 // (op 50) 
v_mul_f32 v15, v11, v7 // (op 50) 
v_mul_f32 v16, v11, v8 // (op 50) 
v_mul_f32 v17, v11, v9 // (op 50) 



v_mov_b32 v0, v13 // (op 56) reg expression

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 56) Store value


v_accvgpr_read v6, a4 // (op 3573) 
v_accvgpr_read v7, a5 // (op 3573) 
v_accvgpr_read v8, a6 // (op 3573) 
v_accvgpr_read v9, a7 // (op 3573) 
v_mul_f32 v14, v11, v6 // (op 3573) 
v_mul_f32 v15, v11, v7 // (op 3573) 
v_mul_f32 v16, v11, v8 // (op 3573) 
v_mul_f32 v17, v11, v9 // (op 3573) 



s_lshl_b32 s0, s4, 7 // (op 3575) 
v_add_u32 v0, v13, s0 // (op 3575) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3575) Store value


v_accvgpr_read v6, a8 // (op 3597) 
v_accvgpr_read v7, a9 // (op 3597) 
v_accvgpr_read v8, a10 // (op 3597) 
v_accvgpr_read v9, a11 // (op 3597) 
v_mul_f32 v14, v11, v6 // (op 3597) 
v_mul_f32 v15, v11, v7 // (op 3597) 
v_mul_f32 v16, v11, v8 // (op 3597) 
v_mul_f32 v17, v11, v9 // (op 3597) 



v_add_u32 v0, 128, v13 // (op 3600) 

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3600) Store value


v_accvgpr_read v6, a12 // (op 3606) 
v_accvgpr_read v7, a13 // (op 3606) 
v_accvgpr_read v8, a14 // (op 3606) 
v_accvgpr_read v9, a15 // (op 3606) 
v_mul_f32 v14, v11, v6 // (op 3606) 
v_mul_f32 v15, v11, v7 // (op 3606) 
v_mul_f32 v16, v11, v8 // (op 3606) 
v_mul_f32 v17, v11, v9 // (op 3606) 



s_lshl_b32 s0, s4, 7 // (op 3602) 
s_add_u32 s0, s0, 128 // (op 3602) 
v_add_u32 v0, v13, s0 // (op 3602) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3602) Store value


v_accvgpr_read v6, a16 // (op 3633) 
v_accvgpr_read v7, a17 // (op 3633) 
v_accvgpr_read v8, a18 // (op 3633) 
v_accvgpr_read v9, a19 // (op 3633) 
v_mul_f32 v14, v11, v6 // (op 3633) 
v_mul_f32 v15, v11, v7 // (op 3633) 
v_mul_f32 v16, v11, v8 // (op 3633) 
v_mul_f32 v17, v11, v9 // (op 3633) 



v_add_u32 v0, 256, v13 // (op 3636) 

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3636) Store value


v_accvgpr_read v6, a20 // (op 3642) 
v_accvgpr_read v7, a21 // (op 3642) 
v_accvgpr_read v8, a22 // (op 3642) 
v_accvgpr_read v9, a23 // (op 3642) 
v_mul_f32 v14, v11, v6 // (op 3642) 
v_mul_f32 v15, v11, v7 // (op 3642) 
v_mul_f32 v16, v11, v8 // (op 3642) 
v_mul_f32 v17, v11, v9 // (op 3642) 



s_lshl_b32 s0, s4, 7 // (op 3638) 
s_add_u32 s0, s0, 256 // (op 3638) 
v_add_u32 v0, v13, s0 // (op 3638) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3638) Store value


v_accvgpr_read v6, a24 // (op 3669) 
v_accvgpr_read v7, a25 // (op 3669) 
v_accvgpr_read v8, a26 // (op 3669) 
v_accvgpr_read v9, a27 // (op 3669) 
v_mul_f32 v14, v11, v6 // (op 3669) 
v_mul_f32 v15, v11, v7 // (op 3669) 
v_mul_f32 v16, v11, v8 // (op 3669) 
v_mul_f32 v17, v11, v9 // (op 3669) 



v_add_u32 v0, 384, v13 // (op 3672) 

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3672) Store value


v_accvgpr_read v6, a28 // (op 3678) 
v_accvgpr_read v7, a29 // (op 3678) 
v_accvgpr_read v8, a30 // (op 3678) 
v_accvgpr_read v9, a31 // (op 3678) 
v_mul_f32 v14, v11, v6 // (op 3678) 
v_mul_f32 v15, v11, v7 // (op 3678) 
v_mul_f32 v16, v11, v8 // (op 3678) 
v_mul_f32 v17, v11, v9 // (op 3678) 



s_lshl_b32 s0, s4, 7 // (op 3674) 
s_add_u32 s0, s0, 384 // (op 3674) 
v_add_u32 v0, v13, s0 // (op 3674) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3674) Store value


v_accvgpr_read v6, a32 // (op 3705) 
v_accvgpr_read v7, a33 // (op 3705) 
v_accvgpr_read v8, a34 // (op 3705) 
v_accvgpr_read v9, a35 // (op 3705) 
v_mul_f32 v14, v11, v6 // (op 3705) 
v_mul_f32 v15, v11, v7 // (op 3705) 
v_mul_f32 v16, v11, v8 // (op 3705) 
v_mul_f32 v17, v11, v9 // (op 3705) 



v_add_u32 v0, 512, v13 // (op 3708) 

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3708) Store value


v_accvgpr_read v6, a36 // (op 3714) 
v_accvgpr_read v7, a37 // (op 3714) 
v_accvgpr_read v8, a38 // (op 3714) 
v_accvgpr_read v9, a39 // (op 3714) 
v_mul_f32 v14, v11, v6 // (op 3714) 
v_mul_f32 v15, v11, v7 // (op 3714) 
v_mul_f32 v16, v11, v8 // (op 3714) 
v_mul_f32 v17, v11, v9 // (op 3714) 



s_lshl_b32 s0, s4, 7 // (op 3710) 
s_add_u32 s0, s0, 512 // (op 3710) 
v_add_u32 v0, v13, s0 // (op 3710) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3710) Store value


v_accvgpr_read v6, a40 // (op 3741) 
v_accvgpr_read v7, a41 // (op 3741) 
v_accvgpr_read v8, a42 // (op 3741) 
v_accvgpr_read v9, a43 // (op 3741) 
v_mul_f32 v14, v11, v6 // (op 3741) 
v_mul_f32 v15, v11, v7 // (op 3741) 
v_mul_f32 v16, v11, v8 // (op 3741) 
v_mul_f32 v17, v11, v9 // (op 3741) 



v_add_u32 v0, 640, v13 // (op 3744) 

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3744) Store value


v_accvgpr_read v6, a44 // (op 3750) 
v_accvgpr_read v7, a45 // (op 3750) 
v_accvgpr_read v8, a46 // (op 3750) 
v_accvgpr_read v9, a47 // (op 3750) 
v_mul_f32 v14, v11, v6 // (op 3750) 
v_mul_f32 v15, v11, v7 // (op 3750) 
v_mul_f32 v16, v11, v8 // (op 3750) 
v_mul_f32 v17, v11, v9 // (op 3750) 



s_lshl_b32 s0, s4, 7 // (op 3746) 
s_add_u32 s0, s0, 640 // (op 3746) 
v_add_u32 v0, v13, s0 // (op 3746) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3746) Store value


v_accvgpr_read v6, a48 // (op 3777) 
v_accvgpr_read v7, a49 // (op 3777) 
v_accvgpr_read v8, a50 // (op 3777) 
v_accvgpr_read v9, a51 // (op 3777) 
v_mul_f32 v14, v11, v6 // (op 3777) 
v_mul_f32 v15, v11, v7 // (op 3777) 
v_mul_f32 v16, v11, v8 // (op 3777) 
v_mul_f32 v17, v11, v9 // (op 3777) 



v_add_u32 v0, 768, v13 // (op 3780) 

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3780) Store value


v_accvgpr_read v6, a52 // (op 3786) 
v_accvgpr_read v7, a53 // (op 3786) 
v_accvgpr_read v8, a54 // (op 3786) 
v_accvgpr_read v9, a55 // (op 3786) 
v_mul_f32 v14, v11, v6 // (op 3786) 
v_mul_f32 v15, v11, v7 // (op 3786) 
v_mul_f32 v16, v11, v8 // (op 3786) 
v_mul_f32 v17, v11, v9 // (op 3786) 



s_lshl_b32 s0, s4, 7 // (op 3782) 
s_add_u32 s0, s0, 768 // (op 3782) 
v_add_u32 v0, v13, s0 // (op 3782) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3782) Store value


v_accvgpr_read v6, a56 // (op 3813) 
v_accvgpr_read v7, a57 // (op 3813) 
v_accvgpr_read v8, a58 // (op 3813) 
v_accvgpr_read v9, a59 // (op 3813) 
v_mul_f32 v14, v11, v6 // (op 3813) 
v_mul_f32 v15, v11, v7 // (op 3813) 
v_mul_f32 v16, v11, v8 // (op 3813) 
v_mul_f32 v17, v11, v9 // (op 3813) 



v_add_u32 v0, 896, v13 // (op 3816) 

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3816) Store value


v_accvgpr_read v6, a60 // (op 3822) 
v_accvgpr_read v7, a61 // (op 3822) 
v_accvgpr_read v8, a62 // (op 3822) 
v_accvgpr_read v9, a63 // (op 3822) 
v_mul_f32 v14, v11, v6 // (op 3822) 
v_mul_f32 v15, v11, v7 // (op 3822) 
v_mul_f32 v16, v11, v8 // (op 3822) 
v_mul_f32 v17, v11, v9 // (op 3822) 



s_lshl_b32 s0, s4, 7 // (op 3818) 
s_add_u32 s0, s0, 896 // (op 3818) 
v_add_u32 v0, v13, s0 // (op 3818) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3818) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ConditionalBottom_0_beta_133 // (op 133) Condition: Done, jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ConditionalBottom_0_beta_133
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ConditionalFalse_0_beta_133:


v_add_u32 v0, 896, v12 // (op 4196) 

buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 4196) Load value

v_accvgpr_read v14, a56 // (op 4188) 
v_accvgpr_read v15, a57 // (op 4188) 
v_accvgpr_read v16, a58 // (op 4188) 
v_accvgpr_read v17, a59 // (op 4188) 
v_mul_f32 v18, v11, v14 // (op 4188) 
v_mul_f32 v19, v11, v15 // (op 4188) 
v_mul_f32 v20, v11, v16 // (op 4188) 
v_mul_f32 v21, v11, v17 // (op 4188) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4195) 
v_mul_f32 v15, v10, v7 // (op 4195) 
v_mul_f32 v16, v10, v8 // (op 4195) 
v_mul_f32 v17, v10, v9 // (op 4195) 

v_add_f32 v6, v14, v18 // (op 4189) 
v_add_f32 v7, v15, v19 // (op 4189) 
v_add_f32 v8, v16, v20 // (op 4189) 
v_add_f32 v9, v17, v21 // (op 4189) 




v_add_u32 v0, 896, v13 // (op 4191) 

buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 4191) Store value




s_lshl_b32 s0, s2, 7 // (op 4200) 
s_add_u32 s0, s0, 896 // (op 4200) 
v_add_u32 v0, v12, s0 // (op 4200) 


buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 4200) Load value

v_accvgpr_read v14, a60 // (op 4202) 
v_accvgpr_read v15, a61 // (op 4202) 
v_accvgpr_read v16, a62 // (op 4202) 
v_accvgpr_read v17, a63 // (op 4202) 
v_mul_f32 v18, v11, v14 // (op 4202) 
v_mul_f32 v19, v11, v15 // (op 4202) 
v_mul_f32 v20, v11, v16 // (op 4202) 
v_mul_f32 v21, v11, v17 // (op 4202) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4198) 
v_mul_f32 v15, v10, v7 // (op 4198) 
v_mul_f32 v16, v10, v8 // (op 4198) 
v_mul_f32 v17, v10, v9 // (op 4198) 

v_add_f32 v6, v14, v18 // (op 4199) 
v_add_f32 v7, v15, v19 // (op 4199) 
v_add_f32 v8, v16, v20 // (op 4199) 
v_add_f32 v9, v17, v21 // (op 4199) 




s_lshl_b32 s0, s4, 7 // (op 4193) 
s_add_u32 s0, s0, 896 // (op 4193) 
v_add_u32 v0, v13, s0 // (op 4193) 


buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 4193) Store value




v_add_u32 v0, 768, v12 // (op 4155) 

s_nop 0
buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 4155) Load value

v_accvgpr_read v14, a48 // (op 4147) 
v_accvgpr_read v15, a49 // (op 4147) 
v_accvgpr_read v16, a50 // (op 4147) 
v_accvgpr_read v17, a51 // (op 4147) 
v_mul_f32 v18, v11, v14 // (op 4147) 
v_mul_f32 v19, v11, v15 // (op 4147) 
v_mul_f32 v20, v11, v16 // (op 4147) 
v_mul_f32 v21, v11, v17 // (op 4147) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4154) 
v_mul_f32 v15, v10, v7 // (op 4154) 
v_mul_f32 v16, v10, v8 // (op 4154) 
v_mul_f32 v17, v10, v9 // (op 4154) 

v_add_f32 v6, v14, v18 // (op 4148) 
v_add_f32 v7, v15, v19 // (op 4148) 
v_add_f32 v8, v16, v20 // (op 4148) 
v_add_f32 v9, v17, v21 // (op 4148) 




v_add_u32 v0, 768, v13 // (op 4150) 

buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 4150) Store value




s_lshl_b32 s0, s2, 7 // (op 4159) 
s_add_u32 s0, s0, 768 // (op 4159) 
v_add_u32 v0, v12, s0 // (op 4159) 


buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 4159) Load value

v_accvgpr_read v14, a52 // (op 4161) 
v_accvgpr_read v15, a53 // (op 4161) 
v_accvgpr_read v16, a54 // (op 4161) 
v_accvgpr_read v17, a55 // (op 4161) 
v_mul_f32 v18, v11, v14 // (op 4161) 
v_mul_f32 v19, v11, v15 // (op 4161) 
v_mul_f32 v20, v11, v16 // (op 4161) 
v_mul_f32 v21, v11, v17 // (op 4161) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4157) 
v_mul_f32 v15, v10, v7 // (op 4157) 
v_mul_f32 v16, v10, v8 // (op 4157) 
v_mul_f32 v17, v10, v9 // (op 4157) 

v_add_f32 v6, v14, v18 // (op 4158) 
v_add_f32 v7, v15, v19 // (op 4158) 
v_add_f32 v8, v16, v20 // (op 4158) 
v_add_f32 v9, v17, v21 // (op 4158) 




s_lshl_b32 s0, s4, 7 // (op 4152) 
s_add_u32 s0, s0, 768 // (op 4152) 
v_add_u32 v0, v13, s0 // (op 4152) 


buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 4152) Store value




v_add_u32 v0, 640, v12 // (op 4114) 

s_nop 0
buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 4114) Load value

v_accvgpr_read v14, a40 // (op 4106) 
v_accvgpr_read v15, a41 // (op 4106) 
v_accvgpr_read v16, a42 // (op 4106) 
v_accvgpr_read v17, a43 // (op 4106) 
v_mul_f32 v18, v11, v14 // (op 4106) 
v_mul_f32 v19, v11, v15 // (op 4106) 
v_mul_f32 v20, v11, v16 // (op 4106) 
v_mul_f32 v21, v11, v17 // (op 4106) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4113) 
v_mul_f32 v15, v10, v7 // (op 4113) 
v_mul_f32 v16, v10, v8 // (op 4113) 
v_mul_f32 v17, v10, v9 // (op 4113) 

v_add_f32 v6, v14, v18 // (op 4107) 
v_add_f32 v7, v15, v19 // (op 4107) 
v_add_f32 v8, v16, v20 // (op 4107) 
v_add_f32 v9, v17, v21 // (op 4107) 




v_add_u32 v0, 640, v13 // (op 4109) 

buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 4109) Store value




s_lshl_b32 s0, s2, 7 // (op 4118) 
s_add_u32 s0, s0, 640 // (op 4118) 
v_add_u32 v0, v12, s0 // (op 4118) 


buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 4118) Load value

v_accvgpr_read v14, a44 // (op 4120) 
v_accvgpr_read v15, a45 // (op 4120) 
v_accvgpr_read v16, a46 // (op 4120) 
v_accvgpr_read v17, a47 // (op 4120) 
v_mul_f32 v18, v11, v14 // (op 4120) 
v_mul_f32 v19, v11, v15 // (op 4120) 
v_mul_f32 v20, v11, v16 // (op 4120) 
v_mul_f32 v21, v11, v17 // (op 4120) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4116) 
v_mul_f32 v15, v10, v7 // (op 4116) 
v_mul_f32 v16, v10, v8 // (op 4116) 
v_mul_f32 v17, v10, v9 // (op 4116) 

v_add_f32 v6, v14, v18 // (op 4117) 
v_add_f32 v7, v15, v19 // (op 4117) 
v_add_f32 v8, v16, v20 // (op 4117) 
v_add_f32 v9, v17, v21 // (op 4117) 




s_lshl_b32 s0, s4, 7 // (op 4111) 
s_add_u32 s0, s0, 640 // (op 4111) 
v_add_u32 v0, v13, s0 // (op 4111) 


buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 4111) Store value




v_add_u32 v0, 512, v12 // (op 4073) 

s_nop 0
buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 4073) Load value

v_accvgpr_read v14, a32 // (op 4065) 
v_accvgpr_read v15, a33 // (op 4065) 
v_accvgpr_read v16, a34 // (op 4065) 
v_accvgpr_read v17, a35 // (op 4065) 
v_mul_f32 v18, v11, v14 // (op 4065) 
v_mul_f32 v19, v11, v15 // (op 4065) 
v_mul_f32 v20, v11, v16 // (op 4065) 
v_mul_f32 v21, v11, v17 // (op 4065) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4072) 
v_mul_f32 v15, v10, v7 // (op 4072) 
v_mul_f32 v16, v10, v8 // (op 4072) 
v_mul_f32 v17, v10, v9 // (op 4072) 

v_add_f32 v6, v14, v18 // (op 4066) 
v_add_f32 v7, v15, v19 // (op 4066) 
v_add_f32 v8, v16, v20 // (op 4066) 
v_add_f32 v9, v17, v21 // (op 4066) 




v_add_u32 v0, 512, v13 // (op 4068) 

buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 4068) Store value




s_lshl_b32 s0, s2, 7 // (op 4077) 
s_add_u32 s0, s0, 512 // (op 4077) 
v_add_u32 v0, v12, s0 // (op 4077) 


buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 4077) Load value

v_accvgpr_read v14, a36 // (op 4079) 
v_accvgpr_read v15, a37 // (op 4079) 
v_accvgpr_read v16, a38 // (op 4079) 
v_accvgpr_read v17, a39 // (op 4079) 
v_mul_f32 v18, v11, v14 // (op 4079) 
v_mul_f32 v19, v11, v15 // (op 4079) 
v_mul_f32 v20, v11, v16 // (op 4079) 
v_mul_f32 v21, v11, v17 // (op 4079) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4075) 
v_mul_f32 v15, v10, v7 // (op 4075) 
v_mul_f32 v16, v10, v8 // (op 4075) 
v_mul_f32 v17, v10, v9 // (op 4075) 

v_add_f32 v6, v14, v18 // (op 4076) 
v_add_f32 v7, v15, v19 // (op 4076) 
v_add_f32 v8, v16, v20 // (op 4076) 
v_add_f32 v9, v17, v21 // (op 4076) 




s_lshl_b32 s0, s4, 7 // (op 4070) 
s_add_u32 s0, s0, 512 // (op 4070) 
v_add_u32 v0, v13, s0 // (op 4070) 


buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 4070) Store value




v_add_u32 v0, 384, v12 // (op 4032) 

s_nop 0
buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 4032) Load value

v_accvgpr_read v14, a24 // (op 4024) 
v_accvgpr_read v15, a25 // (op 4024) 
v_accvgpr_read v16, a26 // (op 4024) 
v_accvgpr_read v17, a27 // (op 4024) 
v_mul_f32 v18, v11, v14 // (op 4024) 
v_mul_f32 v19, v11, v15 // (op 4024) 
v_mul_f32 v20, v11, v16 // (op 4024) 
v_mul_f32 v21, v11, v17 // (op 4024) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4031) 
v_mul_f32 v15, v10, v7 // (op 4031) 
v_mul_f32 v16, v10, v8 // (op 4031) 
v_mul_f32 v17, v10, v9 // (op 4031) 

v_add_f32 v6, v14, v18 // (op 4025) 
v_add_f32 v7, v15, v19 // (op 4025) 
v_add_f32 v8, v16, v20 // (op 4025) 
v_add_f32 v9, v17, v21 // (op 4025) 




v_add_u32 v0, 384, v13 // (op 4027) 

buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 4027) Store value




s_lshl_b32 s0, s2, 7 // (op 4036) 
s_add_u32 s0, s0, 384 // (op 4036) 
v_add_u32 v0, v12, s0 // (op 4036) 


buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 4036) Load value

v_accvgpr_read v14, a28 // (op 4038) 
v_accvgpr_read v15, a29 // (op 4038) 
v_accvgpr_read v16, a30 // (op 4038) 
v_accvgpr_read v17, a31 // (op 4038) 
v_mul_f32 v18, v11, v14 // (op 4038) 
v_mul_f32 v19, v11, v15 // (op 4038) 
v_mul_f32 v20, v11, v16 // (op 4038) 
v_mul_f32 v21, v11, v17 // (op 4038) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4034) 
v_mul_f32 v15, v10, v7 // (op 4034) 
v_mul_f32 v16, v10, v8 // (op 4034) 
v_mul_f32 v17, v10, v9 // (op 4034) 

v_add_f32 v6, v14, v18 // (op 4035) 
v_add_f32 v7, v15, v19 // (op 4035) 
v_add_f32 v8, v16, v20 // (op 4035) 
v_add_f32 v9, v17, v21 // (op 4035) 




s_lshl_b32 s0, s4, 7 // (op 4029) 
s_add_u32 s0, s0, 384 // (op 4029) 
v_add_u32 v0, v13, s0 // (op 4029) 


buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 4029) Store value




v_add_u32 v0, 256, v12 // (op 3991) 

s_nop 0
buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 3991) Load value

v_accvgpr_read v14, a16 // (op 3983) 
v_accvgpr_read v15, a17 // (op 3983) 
v_accvgpr_read v16, a18 // (op 3983) 
v_accvgpr_read v17, a19 // (op 3983) 
v_mul_f32 v18, v11, v14 // (op 3983) 
v_mul_f32 v19, v11, v15 // (op 3983) 
v_mul_f32 v20, v11, v16 // (op 3983) 
v_mul_f32 v21, v11, v17 // (op 3983) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 3990) 
v_mul_f32 v15, v10, v7 // (op 3990) 
v_mul_f32 v16, v10, v8 // (op 3990) 
v_mul_f32 v17, v10, v9 // (op 3990) 

v_add_f32 v6, v14, v18 // (op 3984) 
v_add_f32 v7, v15, v19 // (op 3984) 
v_add_f32 v8, v16, v20 // (op 3984) 
v_add_f32 v9, v17, v21 // (op 3984) 




v_add_u32 v0, 256, v13 // (op 3986) 

buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 3986) Store value




s_lshl_b32 s0, s2, 7 // (op 3995) 
s_add_u32 s0, s0, 256 // (op 3995) 
v_add_u32 v0, v12, s0 // (op 3995) 


buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 3995) Load value

v_accvgpr_read v14, a20 // (op 3997) 
v_accvgpr_read v15, a21 // (op 3997) 
v_accvgpr_read v16, a22 // (op 3997) 
v_accvgpr_read v17, a23 // (op 3997) 
v_mul_f32 v18, v11, v14 // (op 3997) 
v_mul_f32 v19, v11, v15 // (op 3997) 
v_mul_f32 v20, v11, v16 // (op 3997) 
v_mul_f32 v21, v11, v17 // (op 3997) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 3993) 
v_mul_f32 v15, v10, v7 // (op 3993) 
v_mul_f32 v16, v10, v8 // (op 3993) 
v_mul_f32 v17, v10, v9 // (op 3993) 

v_add_f32 v6, v14, v18 // (op 3994) 
v_add_f32 v7, v15, v19 // (op 3994) 
v_add_f32 v8, v16, v20 // (op 3994) 
v_add_f32 v9, v17, v21 // (op 3994) 




s_lshl_b32 s0, s4, 7 // (op 3988) 
s_add_u32 s0, s0, 256 // (op 3988) 
v_add_u32 v0, v13, s0 // (op 3988) 


buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 3988) Store value




v_add_u32 v0, 128, v12 // (op 3950) 

s_nop 0
buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 3950) Load value

v_accvgpr_read v14, a8 // (op 3942) 
v_accvgpr_read v15, a9 // (op 3942) 
v_accvgpr_read v16, a10 // (op 3942) 
v_accvgpr_read v17, a11 // (op 3942) 
v_mul_f32 v18, v11, v14 // (op 3942) 
v_mul_f32 v19, v11, v15 // (op 3942) 
v_mul_f32 v20, v11, v16 // (op 3942) 
v_mul_f32 v21, v11, v17 // (op 3942) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 3949) 
v_mul_f32 v15, v10, v7 // (op 3949) 
v_mul_f32 v16, v10, v8 // (op 3949) 
v_mul_f32 v17, v10, v9 // (op 3949) 

v_add_f32 v6, v14, v18 // (op 3943) 
v_add_f32 v7, v15, v19 // (op 3943) 
v_add_f32 v8, v16, v20 // (op 3943) 
v_add_f32 v9, v17, v21 // (op 3943) 




v_add_u32 v0, 128, v13 // (op 3945) 

buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 3945) Store value




s_lshl_b32 s0, s2, 7 // (op 3954) 
s_add_u32 s0, s0, 128 // (op 3954) 
v_add_u32 v0, v12, s0 // (op 3954) 


buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 3954) Load value

v_accvgpr_read v14, a12 // (op 3956) 
v_accvgpr_read v15, a13 // (op 3956) 
v_accvgpr_read v16, a14 // (op 3956) 
v_accvgpr_read v17, a15 // (op 3956) 
v_mul_f32 v18, v11, v14 // (op 3956) 
v_mul_f32 v19, v11, v15 // (op 3956) 
v_mul_f32 v20, v11, v16 // (op 3956) 
v_mul_f32 v21, v11, v17 // (op 3956) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 3952) 
v_mul_f32 v15, v10, v7 // (op 3952) 
v_mul_f32 v16, v10, v8 // (op 3952) 
v_mul_f32 v17, v10, v9 // (op 3952) 

v_add_f32 v6, v14, v18 // (op 3953) 
v_add_f32 v7, v15, v19 // (op 3953) 
v_add_f32 v8, v16, v20 // (op 3953) 
v_add_f32 v9, v17, v21 // (op 3953) 




s_lshl_b32 s0, s4, 7 // (op 3947) 
s_add_u32 s0, s0, 128 // (op 3947) 
v_add_u32 v0, v13, s0 // (op 3947) 


buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 3947) Store value




v_mov_b32 v0, v12 // (op 145) reg expression

s_nop 0
buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 145) Load value

v_accvgpr_read v14, a0 // (op 142) 
v_accvgpr_read v15, a1 // (op 142) 
v_accvgpr_read v16, a2 // (op 142) 
v_accvgpr_read v17, a3 // (op 142) 
v_mul_f32 v18, v11, v14 // (op 142) 
v_mul_f32 v19, v11, v15 // (op 142) 
v_mul_f32 v20, v11, v16 // (op 142) 
v_mul_f32 v21, v11, v17 // (op 142) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 146) 
v_mul_f32 v15, v10, v7 // (op 146) 
v_mul_f32 v16, v10, v8 // (op 146) 
v_mul_f32 v17, v10, v9 // (op 146) 

v_add_f32 v6, v14, v18 // (op 143) 
v_add_f32 v7, v15, v19 // (op 143) 
v_add_f32 v8, v16, v20 // (op 143) 
v_add_f32 v9, v17, v21 // (op 143) 




v_mov_b32 v0, v13 // (op 144) reg expression

buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 144) Store value




s_lshl_b32 s0, s2, 7 // (op 3912) 
v_add_u32 v0, v12, s0 // (op 3912) 


buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 3912) Load value

v_accvgpr_read v14, a4 // (op 3909) 
v_accvgpr_read v15, a5 // (op 3909) 
v_accvgpr_read v16, a6 // (op 3909) 
v_accvgpr_read v17, a7 // (op 3909) 
v_mul_f32 v18, v11, v14 // (op 3909) 
v_mul_f32 v19, v11, v15 // (op 3909) 
v_mul_f32 v20, v11, v16 // (op 3909) 
v_mul_f32 v21, v11, v17 // (op 3909) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 3913) 
v_mul_f32 v15, v10, v7 // (op 3913) 
v_mul_f32 v16, v10, v8 // (op 3913) 
v_mul_f32 v17, v10, v9 // (op 3913) 

v_add_f32 v6, v14, v18 // (op 3910) 
v_add_f32 v7, v15, v19 // (op 3910) 
v_add_f32 v8, v16, v20 // (op 3910) 
v_add_f32 v9, v17, v21 // (op 3910) 




s_lshl_b32 s0, s4, 7 // (op 3911) 
v_add_u32 v0, v13, s0 // (op 3911) 


buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 3911) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ConditionalBottom_0_beta_133:
























s_endpgm  // End of RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2
.LRR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_end:
.size RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2, .LRR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_end-RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2
  .amdhsa_next_free_vgpr 212
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 84480 // lds bytes
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
    macrotile: [256, 64, 128]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [8, 2]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2
    .symbol: RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 84480
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 64
    .vgpr_count: 148
    .agpr_count: 64
    .max_flat_workgroup_size: 256
    .workgroup_size: [256, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, {is-null: true}]
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

