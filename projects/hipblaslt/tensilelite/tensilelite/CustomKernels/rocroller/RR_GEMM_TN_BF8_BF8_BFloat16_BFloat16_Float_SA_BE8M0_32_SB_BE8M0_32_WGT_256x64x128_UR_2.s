

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2
.p2align 8
.type RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2,@function
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 8971) 
v_and_b32 v3, 15, v1 // (op 8971) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 8971) 


v_and_b32 v0, 48, v1 // (op 8971) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 8971) 


v_lshrrev_b32 v0, 6, v1 // (op 9028) 
v_and_b32 v4, 1, v0 // (op 9028) 

v_and_b32 v0, 15, v1 // (op 9028) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 9028) 


v_and_b32 v0, 48, v1 // (op 9028) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 9028) 


v_lshrrev_b32 v0, 7, v1 // (op 9086) 
v_and_b32 v5, 15, v1 // (op 9086) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 9086) 


v_and_b32 v0, 63, v1 // (op 9086) 
v_lshrrev_b32 v5, 4, v0 // (op 9086) 

v_lshl_add_u32 v0, v6, 2, v5 // (op 9086) 


v_lshrrev_b32 v5, 6, v1 // (op 9144) 
v_and_b32 v6, 1, v5 // (op 9144) 

v_and_b32 v5, 15, v1 // (op 9144) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 9144) 


v_and_b32 v5, 63, v1 // (op 9144) 
v_lshrrev_b32 v6, 4, v5 // (op 9144) 

v_lshl_add_u32 v5, v7, 2, v6 // (op 9144) 


v_lshrrev_b32 v6, 7, v1 // (op 9211) 
v_and_b32 v7, 15, v1 // (op 9211) 
v_lshl_add_u32 v8, v6, 4, v7 // (op 9211) 


v_and_b32 v6, 48, v1 // (op 9211) 
v_lshl_add_u32 v7, v8, 7, v6 // (op 9211) 


v_lshrrev_b32 v6, 6, v1 // (op 9255) 
v_and_b32 v8, 1, v6 // (op 9255) 

v_and_b32 v6, 15, v1 // (op 9255) 
v_lshl_add_u32 v9, v8, 4, v6 // (op 9255) 


v_and_b32 v6, 48, v1 // (op 9255) 
v_lshl_add_u32 v8, v9, 7, v6 // (op 9255) 


v_lshrrev_b32 v6, 7, v1 // (op 9284) 
v_and_b32 v9, 15, v1 // (op 9284) 
v_lshl_add_u32 v10, v6, 4, v9 // (op 9284) 


v_and_b32 v6, 63, v1 // (op 9284) 
v_lshrrev_b32 v9, 4, v6 // (op 9284) 

v_lshl_add_u32 v6, v10, 2, v9 // (op 9284) 


v_lshrrev_b32 v9, 6, v1 // (op 9313) 
v_and_b32 v10, 1, v9 // (op 9313) 

v_and_b32 v9, 15, v1 // (op 9313) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 9313) 


v_and_b32 v9, 63, v1 // (op 9313) 
v_lshrrev_b32 v10, 4, v9 // (op 9313) 

v_lshl_add_u32 v9, v11, 2, v10 // (op 9313) 


v_lshrrev_b32 v10, 7, v1 // (op 8910) 
v_and_b32 v11, 63, v1 // (op 8910) 
v_lshrrev_b32 v12, 4, v11 // (op 8910) 
v_lshlrev_b32 v13, 2, v12 // (op 8910) 

v_lshl_add_u32 v12, v10, 4, v13 // (op 8910) 


v_lshl_add_u32 v10, s16, 8, v12 // (op 8910) 

v_lshrrev_b32 v12, 2, v11 // (op 8910) 

v_and_b32 v11, 3, v12 // (op 8910) 

v_and_b32 v12, 3, v1 // (op 8910) 
v_lshl_add_u32 v13, v11, 2, v12 // (op 8910) 


v_lshrrev_b32 v11, 6, v1 // (op 8910) 
v_and_b32 v12, 1, v11 // (op 8910) 

v_lshl_add_u32 v11, v12, 4, v13 // (op 8910) 


v_lshl_add_u32 v12, s17, 6, v11 // (op 8910) 

v_mul_lo_u32 v11, v12, s2 // (op 8910) 

v_add_lshl_u32 v12, v10, v11, 1 // (op 8910) 


v_lshrrev_b32 v10, 7, v1 // (op 9165) 
v_and_b32 v11, 63, v1 // (op 9165) 
v_lshrrev_b32 v13, 4, v11 // (op 9165) 
s_mul_i32 s1, s2, s9 // (op 8912) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 8912) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 8912) least significant
s_mul_hi_u32 s19, s2, s8 // (op 8912) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8912) most significant: sum
s_add_u32 s1, s1, s19 // (op 8912) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 8912) 

v_lshlrev_b32 v14, 2, v13 // (op 9165) 

v_lshl_add_u32 v13, v10, 4, v14 // (op 9165) 


v_lshl_add_u32 v10, s16, 8, v13 // (op 9165) 

v_lshrrev_b32 v13, 2, v11 // (op 9165) 

v_and_b32 v11, 3, v13 // (op 9165) 

v_and_b32 v13, 3, v1 // (op 9165) 
v_lshl_add_u32 v14, v11, 2, v13 // (op 9165) 


v_lshrrev_b32 v11, 6, v1 // (op 9165) 
v_and_b32 v13, 1, v11 // (op 9165) 

v_lshl_add_u32 v11, v13, 4, v14 // (op 9165) 


v_lshl_add_u32 v13, s17, 6, v11 // (op 9165) 

v_mul_lo_u32 v11, v13, s4 // (op 9165) 

v_add_lshl_u32 v13, v10, v11, 1 // (op 9165) 


s_mov_b32 s40, s6 // (op 8912) reg expression
s_mov_b32 s41, s7 // (op 8912) reg expression
s_mov_b32 s42, s18 // (op 8912) reg expression
s_mov_b32 s43, 131072 // (op 8912) call()

s_mul_i32 s1, s4, s9 // (op 9167) most significant: low of low * high

s_mul_i32 s6, s5, s8 // (op 9167) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 9167) least significant
s_mul_hi_u32 s7, s4, s8 // (op 9167) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 9167) most significant: sum
s_add_u32 s1, s1, s7 // (op 9167) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 1 // (op 9167) 

s_mov_b32 s44, s10 // (op 9167) reg expression
s_mov_b32 s45, s11 // (op 9167) reg expression
s_mov_b32 s46, s6 // (op 9167) reg expression
s_mov_b32 s47, 131072 // (op 9167) call()


v_mov_b32 v10, s14 // (op 2) Move value
v_mov_b32 v11, s15 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 65) call()
v_accvgpr_write a1, 0.00000 // (op 65) call()
v_accvgpr_write a2, 0.00000 // (op 65) call()
v_accvgpr_write a3, 0.00000 // (op 65) call()
v_accvgpr_write a4, 0.00000 // (op 340) call()
v_accvgpr_write a5, 0.00000 // (op 340) call()
v_accvgpr_write a6, 0.00000 // (op 340) call()
v_accvgpr_write a7, 0.00000 // (op 340) call()
v_accvgpr_write a8, 0.00000 // (op 560) call()
v_accvgpr_write a9, 0.00000 // (op 560) call()
v_accvgpr_write a10, 0.00000 // (op 560) call()
v_accvgpr_write a11, 0.00000 // (op 560) call()
v_accvgpr_write a12, 0.00000 // (op 715) call()
v_accvgpr_write a13, 0.00000 // (op 715) call()
v_accvgpr_write a14, 0.00000 // (op 715) call()
v_accvgpr_write a15, 0.00000 // (op 715) call()
v_accvgpr_write a16, 0.00000 // (op 946) call()
v_accvgpr_write a17, 0.00000 // (op 946) call()
v_accvgpr_write a18, 0.00000 // (op 946) call()
v_accvgpr_write a19, 0.00000 // (op 946) call()
v_accvgpr_write a20, 0.00000 // (op 1101) call()
v_accvgpr_write a21, 0.00000 // (op 1101) call()
v_accvgpr_write a22, 0.00000 // (op 1101) call()
v_accvgpr_write a23, 0.00000 // (op 1101) call()
v_accvgpr_write a24, 0.00000 // (op 1332) call()
v_accvgpr_write a25, 0.00000 // (op 1332) call()
v_accvgpr_write a26, 0.00000 // (op 1332) call()
v_accvgpr_write a27, 0.00000 // (op 1332) call()
v_accvgpr_write a28, 0.00000 // (op 1487) call()
v_accvgpr_write a29, 0.00000 // (op 1487) call()
v_accvgpr_write a30, 0.00000 // (op 1487) call()
v_accvgpr_write a31, 0.00000 // (op 1487) call()
v_accvgpr_write a32, 0.00000 // (op 1718) call()
v_accvgpr_write a33, 0.00000 // (op 1718) call()
v_accvgpr_write a34, 0.00000 // (op 1718) call()
v_accvgpr_write a35, 0.00000 // (op 1718) call()
v_accvgpr_write a36, 0.00000 // (op 1873) call()
v_accvgpr_write a37, 0.00000 // (op 1873) call()
v_accvgpr_write a38, 0.00000 // (op 1873) call()
v_accvgpr_write a39, 0.00000 // (op 1873) call()
v_accvgpr_write a40, 0.00000 // (op 2104) call()
v_accvgpr_write a41, 0.00000 // (op 2104) call()
v_accvgpr_write a42, 0.00000 // (op 2104) call()
v_accvgpr_write a43, 0.00000 // (op 2104) call()
v_accvgpr_write a44, 0.00000 // (op 2259) call()
v_accvgpr_write a45, 0.00000 // (op 2259) call()
v_accvgpr_write a46, 0.00000 // (op 2259) call()
v_accvgpr_write a47, 0.00000 // (op 2259) call()
v_accvgpr_write a48, 0.00000 // (op 2490) call()
v_accvgpr_write a49, 0.00000 // (op 2490) call()
v_accvgpr_write a50, 0.00000 // (op 2490) call()
v_accvgpr_write a51, 0.00000 // (op 2490) call()
v_accvgpr_write a52, 0.00000 // (op 2645) call()
v_accvgpr_write a53, 0.00000 // (op 2645) call()
v_accvgpr_write a54, 0.00000 // (op 2645) call()
v_accvgpr_write a55, 0.00000 // (op 2645) call()
v_accvgpr_write a56, 0.00000 // (op 2876) call()
v_accvgpr_write a57, 0.00000 // (op 2876) call()
v_accvgpr_write a58, 0.00000 // (op 2876) call()
v_accvgpr_write a59, 0.00000 // (op 2876) call()
v_accvgpr_write a60, 0.00000 // (op 3031) call()
v_accvgpr_write a61, 0.00000 // (op 3031) call()
v_accvgpr_write a62, 0.00000 // (op 3031) call()
v_accvgpr_write a63, 0.00000 // (op 3031) call()
v_lshrrev_b32 v14, 3, v1 // (op 8672) 
v_lshl_add_u32 v15, s16, 8, v14 // (op 8672) 

v_and_b32 v14, -8, v1 // (op 8672) 
v_sub_u32 v16, v1, v14 // (op 8672) 

v_lshlrev_b32 v14, 4, v16 // (op 8672) 

v_mov_b32 v17, 0 // (op 8672) convert
v_mov_b32 v16, v14 // (op 8672) convert

v_mul_lo_u32 v19, v15, s13 // (op 8672) most significant: low of low * high
v_mul_hi_u32 v14, v15, s12 // (op 8672) most significant: high of low * low
v_mul_lo_u32 v18, v15, s12 // (op 8672) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 8672) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 8672) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 8672) most significant half




s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 8674) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 8674) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 8674) least significant
s_mul_hi_u32 s7, s12, s22 // (op 8674) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8674) most significant: sum
s_add_u32 s1, s1, s7 // (op 8674) most significant: sum


s_mov_b32 s48, s20 // (op 8674) reg expression
s_mov_b32 s49, s21 // (op 8674) reg expression
s_mov_b32 s50, s0 // (op 8674) reg expression
s_mov_b32 s51, 131072 // (op 8674) call()

v_lshrrev_b32 v14, 3, v1 // (op 8691) 
v_lshl_add_u32 v15, s16, 8, v14 // (op 8691) 

v_and_b32 v14, -8, v1 // (op 8691) 
v_sub_u32 v16, v1, v14 // (op 8691) 

v_lshlrev_b32 v14, 4, v16 // (op 8691) 

v_mov_b32 v17, 0 // (op 8691) convert
v_mov_b32 v16, v14 // (op 8691) convert

v_mul_lo_u32 v21, v15, s13 // (op 8691) most significant: low of low * high
v_mul_hi_u32 v14, v15, s12 // (op 8691) most significant: high of low * low
v_mul_lo_u32 v20, v15, s12 // (op 8691) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 8691) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v16 // (op 8691) least significant half
v_addc_co_u32 v21, s[0:1], v21, v17, s[0:1] // (op 8691) most significant half



s_mul_i32 s1, s12, s23 // (op 8693) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 8693) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 8693) least significant
s_mul_hi_u32 s7, s12, s22 // (op 8693) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8693) most significant: sum
s_add_u32 s1, s1, s7 // (op 8693) most significant: sum


s_mov_b32 s48, s20 // (op 8693) reg expression
s_mov_b32 s49, s21 // (op 8693) reg expression
s_mov_b32 s50, s0 // (op 8693) reg expression
s_mov_b32 s51, 131072 // (op 8693) call()

v_and_b32 v14, -8, v1 // (op 8736) 
v_sub_u32 v15, v1, v14 // (op 8736) 

v_lshlrev_b32 v14, 4, v15 // (op 8736) 

v_mov_b32 v17, 0 // (op 8736) convert
v_mov_b32 v16, v14 // (op 8736) convert

v_lshrrev_b32 v14, 3, v1 // (op 8736) 
v_lshl_add_u32 v15, s17, 6, v14 // (op 8736) 

v_mul_lo_u32 v23, v15, s25 // (op 8736) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 8736) most significant: high of low * low
v_mul_lo_u32 v22, v15, s24 // (op 8736) least significant: low of low * low
v_add_u32 v23, v23, v14 // (op 8736) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v22 // (op 8736) least significant half
v_addc_co_u32 v15, s[0:1], v17, v23, s[0:1] // (op 8736) most significant half



s_mul_i32 s1, s24, s9 // (op 8738) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 8738) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 8738) least significant
s_mul_hi_u32 s7, s24, s8 // (op 8738) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8738) most significant: sum
s_add_u32 s1, s1, s7 // (op 8738) most significant: sum


s_mov_b32 s52, s26 // (op 8738) reg expression
s_mov_b32 s53, s27 // (op 8738) reg expression
s_mov_b32 s54, s0 // (op 8738) reg expression
s_mov_b32 s55, 131072 // (op 8738) call()

v_and_b32 v16, -8, v1 // (op 8755) 
v_sub_u32 v17, v1, v16 // (op 8755) 

v_lshlrev_b32 v16, 4, v17 // (op 8755) 

v_mov_b32 v23, 0 // (op 8755) convert
v_mov_b32 v22, v16 // (op 8755) convert

v_lshrrev_b32 v16, 3, v1 // (op 8755) 
v_lshl_add_u32 v17, s17, 6, v16 // (op 8755) 

v_mul_lo_u32 v25, v17, s25 // (op 8755) most significant: low of low * high
v_mul_hi_u32 v16, v17, s24 // (op 8755) most significant: high of low * low
v_mul_lo_u32 v24, v17, s24 // (op 8755) least significant: low of low * low
v_add_u32 v25, v25, v16 // (op 8755) most significant: sum


v_add_co_u32 v16, s[0:1], v22, v24 // (op 8755) least significant half
v_addc_co_u32 v17, s[0:1], v23, v25, s[0:1] // (op 8755) most significant half



s_mul_i32 s1, s24, s9 // (op 8757) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 8757) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 8757) least significant
s_mul_hi_u32 s7, s24, s8 // (op 8757) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8757) most significant: sum
s_add_u32 s1, s1, s7 // (op 8757) most significant: sum


s_mov_b32 s52, s26 // (op 8757) reg expression
s_mov_b32 s53, s27 // (op 8757) reg expression
s_mov_b32 s54, s0 // (op 8757) reg expression
s_mov_b32 s55, 131072 // (op 8757) call()

v_lshl_add_u32 v22, s16, 8, v1 // (op 8800) 
v_mul_lo_u32 v23, v22, s28 // (op 8800) 

s_mul_i32 s1, s28, s23 // (op 8802) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 8802) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 8802) least significant
s_mul_hi_u32 s7, s28, s22 // (op 8802) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8802) most significant: sum
s_add_u32 s1, s1, s7 // (op 8802) most significant: sum


s_mov_b32 s56, s30 // (op 8802) reg expression
s_mov_b32 s57, s31 // (op 8802) reg expression
s_mov_b32 s58, s0 // (op 8802) reg expression
s_mov_b32 s59, 131072 // (op 8802) call()

v_lshl_add_u32 v22, s16, 8, v1 // (op 8819) 
v_mul_lo_u32 v24, v22, s28 // (op 8819) 

s_mul_i32 s1, s28, s23 // (op 8821) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 8821) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 8821) least significant
s_mul_hi_u32 s7, s28, s22 // (op 8821) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8821) most significant: sum
s_add_u32 s1, s1, s7 // (op 8821) most significant: sum


s_mov_b32 s56, s30 // (op 8821) reg expression
s_mov_b32 s57, s31 // (op 8821) reg expression
s_mov_b32 s58, s0 // (op 8821) reg expression
s_mov_b32 s59, 131072 // (op 8821) call()

v_and_b32 v22, -4, v1 // (op 8864) 
v_sub_u32 v25, v1, v22 // (op 8864) 

v_lshrrev_b32 v22, 2, v1 // (op 8864) 
v_lshl_add_u32 v26, s17, 6, v22 // (op 8864) 

v_mul_lo_u32 v22, v26, s32 // (op 8864) 

v_add_u32 v26, v25, v22 // (op 8864) 


s_mul_i32 s1, s32, s9 // (op 8866) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 8866) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 8866) least significant
s_mul_hi_u32 s7, s32, s8 // (op 8866) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8866) most significant: sum
s_add_u32 s1, s1, s7 // (op 8866) most significant: sum


s_mov_b32 s60, s34 // (op 8866) reg expression
s_mov_b32 s61, s35 // (op 8866) reg expression
s_mov_b32 s62, s0 // (op 8866) reg expression
s_mov_b32 s63, 131072 // (op 8866) call()

v_and_b32 v22, -4, v1 // (op 8883) 
v_sub_u32 v25, v1, v22 // (op 8883) 

v_lshrrev_b32 v22, 2, v1 // (op 8883) 
v_lshl_add_u32 v27, s17, 6, v22 // (op 8883) 

v_mul_lo_u32 v22, v27, s32 // (op 8883) 

v_add_u32 v27, v25, v22 // (op 8883) 


s_mul_i32 s1, s32, s9 // (op 8885) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 8885) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 8885) least significant
s_mul_hi_u32 s7, s32, s8 // (op 8885) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8885) most significant: sum
s_add_u32 s1, s1, s7 // (op 8885) most significant: sum


s_mov_b32 s60, s34 // (op 8885) reg expression
s_mov_b32 s61, s35 // (op 8885) reg expression
s_mov_b32 s62, s0 // (op 8885) reg expression
s_mov_b32 s63, 131072 // (op 8885) call()

v_and_b32 v22, -8, v1 // (op 9011) 
v_sub_u32 v25, v1, v22 // (op 9011) 

v_lshlrev_b32 v22, 4, v25 // (op 9011) 

v_lshrrev_b32 v25, 3, v1 // (op 9011) 
v_lshl_add_u32 v28, v25, 7, v22 // (op 9011) 


s_nop 0
v_readfirstlane_b32 s0, v28 // (op 9011) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 8606) 
v_mov_b64 v[28:29], s[6:7] // (op 8606) convert


v_mov_b64 v[30:31], v[14:15] // (op 8606) 
s_mov_b32 m0, s0 // (op 8606) reg expression
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 8606) Load value direct to lds
v_add_co_u32 v30, s[6:7], v30, v28 // (op 8606) least significant half
v_addc_co_u32 v31, s[6:7], 0, v31, s[6:7] // (op 8606) most significant half

s_add_u32 m0, m0, 4096 // (op 8606) 
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 8606) Load value direct to lds






buffer_load_ubyte v22, v26, s[60:63], 0 offen offset: 0 // (op 8292) Load value
v_and_b32 v25, -8, v1 // (op 8954) 
v_sub_u32 v28, v1, v25 // (op 8954) 

v_lshlrev_b32 v25, 4, v28 // (op 8954) 

v_lshrrev_b32 v28, 3, v1 // (op 8954) 
v_lshl_add_u32 v29, v28, 7, v25 // (op 8954) 


s_nop 0
v_readfirstlane_b32 s0, v29 // (op 8954) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 5 // (op 8611) 
v_mov_b64 v[28:29], s[6:7] // (op 8611) convert


v_mov_b64 v[30:31], v[18:19] // (op 8611) 
s_add_u32 s1, 8192, s0 // (op 8611) 
s_mov_b32 m0, s1 // (op 8611) reg expression
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 8611) Load value direct to lds
v_add_co_u32 v30, s[6:7], v30, v28 // (op 8611) least significant half
v_addc_co_u32 v31, s[6:7], 0, v31, s[6:7] // (op 8611) most significant half

s_add_u32 m0, m0, 4096 // (op 8611) 
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 8611) Load value direct to lds
v_add_co_u32 v30, s[6:7], v30, v28 // (op 8611) least significant half
v_addc_co_u32 v31, s[6:7], 0, v31, s[6:7] // (op 8611) most significant half

s_add_u32 m0, m0, 4096 // (op 8611) 
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 8611) Load value direct to lds
v_add_co_u32 v30, s[6:7], v30, v28 // (op 8611) least significant half
v_addc_co_u32 v31, s[6:7], 0, v31, s[6:7] // (op 8611) most significant half

s_add_u32 m0, m0, 4096 // (op 8611) 
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 8611) Load value direct to lds
v_add_co_u32 v30, s[6:7], v30, v28 // (op 8611) least significant half
v_addc_co_u32 v31, s[6:7], 0, v31, s[6:7] // (op 8611) most significant half

s_add_u32 m0, m0, 4096 // (op 8611) 
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 8611) Load value direct to lds
v_add_co_u32 v30, s[6:7], v30, v28 // (op 8611) least significant half
v_addc_co_u32 v31, s[6:7], 0, v31, s[6:7] // (op 8611) most significant half

s_add_u32 m0, m0, 4096 // (op 8611) 
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 8611) Load value direct to lds
v_add_co_u32 v30, s[6:7], v30, v28 // (op 8611) least significant half
v_addc_co_u32 v31, s[6:7], 0, v31, s[6:7] // (op 8611) most significant half

s_add_u32 m0, m0, 4096 // (op 8611) 
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 8611) Load value direct to lds
v_add_co_u32 v30, s[6:7], v30, v28 // (op 8611) least significant half
v_addc_co_u32 v31, s[6:7], 0, v31, s[6:7] // (op 8611) most significant half

s_add_u32 m0, m0, 4096 // (op 8611) 
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 8611) Load value direct to lds







buffer_load_dword v25, v23, s[56:59], 0 offen // (op 8298) Load value
v_and_b32 v28, -8, v1 // (op 9238) 
v_sub_u32 v29, v1, v28 // (op 9238) 

v_lshlrev_b32 v28, 4, v29 // (op 9238) 

v_lshrrev_b32 v29, 3, v1 // (op 9238) 
v_lshl_add_u32 v30, v29, 7, v28 // (op 9238) 


s_nop 0
v_readfirstlane_b32 s0, v30 // (op 9238) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 8616) 
v_mov_b64 v[28:29], s[6:7] // (op 8616) convert


v_mov_b64 v[30:31], v[14:15] // (op 8616) 
s_add_u32 s1, 40960, s0 // (op 8616) 
s_mov_b32 m0, s1 // (op 8616) reg expression
s_mov_b32 s6, 128 // (op 8616) call()
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 8616) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v28 // (op 8616) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 8616) most significant half

s_add_u32 m0, m0, 4096 // (op 8616) 
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 8616) Load value direct to lds









buffer_load_ubyte v14, v26, s[60:63], 0 offen offset:4 // (op 8304) Load value

v_and_b32 v15, -8, v1 // (op 9194) 
v_sub_u32 v26, v1, v15 // (op 9194) 

v_lshlrev_b32 v15, 4, v26 // (op 9194) 

v_lshrrev_b32 v26, 3, v1 // (op 9194) 
v_lshl_add_u32 v28, v26, 7, v15 // (op 9194) 


s_nop 0
v_readfirstlane_b32 s0, v28 // (op 9194) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 5 // (op 8618) 
v_mov_b64 v[28:29], s[6:7] // (op 8618) convert


v_mov_b64 v[30:31], v[18:19] // (op 8618) 
s_add_u32 s1, 49152, s0 // (op 8618) 
s_mov_b32 m0, s1 // (op 8618) reg expression
s_mov_b32 s6, 128 // (op 8618) call()
buffer_load_dwordx4 v30, s[48:51], s6 offen offset: 0 lds // (op 8618) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v28 // (op 8618) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 8618) most significant half

s_add_u32 m0, m0, 4096 // (op 8618) 
buffer_load_dwordx4 v30, s[48:51], s6 offen offset: 0 lds // (op 8618) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v28 // (op 8618) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 8618) most significant half

s_add_u32 m0, m0, 4096 // (op 8618) 
buffer_load_dwordx4 v30, s[48:51], s6 offen offset: 0 lds // (op 8618) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v28 // (op 8618) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 8618) most significant half

s_add_u32 m0, m0, 4096 // (op 8618) 
buffer_load_dwordx4 v30, s[48:51], s6 offen offset: 0 lds // (op 8618) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v28 // (op 8618) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 8618) most significant half

s_add_u32 m0, m0, 4096 // (op 8618) 
buffer_load_dwordx4 v30, s[48:51], s6 offen offset: 0 lds // (op 8618) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v28 // (op 8618) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 8618) most significant half

s_add_u32 m0, m0, 4096 // (op 8618) 
buffer_load_dwordx4 v30, s[48:51], s6 offen offset: 0 lds // (op 8618) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v28 // (op 8618) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 8618) most significant half

s_add_u32 m0, m0, 4096 // (op 8618) 
buffer_load_dwordx4 v30, s[48:51], s6 offen offset: 0 lds // (op 8618) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v28 // (op 8618) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 8618) most significant half

s_add_u32 m0, m0, 4096 // (op 8618) 
buffer_load_dwordx4 v30, s[48:51], s6 offen offset: 0 lds // (op 8618) Load value direct to lds









buffer_load_dword v15, v23, s[56:59], 0 offen offset:4 // (op 8310) Load value

v_and_b32 v23, -4, v1 // (op 9127) 
v_sub_u32 v26, v1, v23 // (op 9127) 

v_and_b32 v23, 4294967292, v1 // (op 9127) 
v_add_u32 v18, v23, v26 // (op 9127) 





v_add_u32 v19, 81920, v18 // (op 8313) 
s_waitcnt vmcnt(21)
ds_write_b8 v19, v22 // (op 8313) Store local data 


v_lshlrev_b32 v23, 2, v1 // (op 9069) 



v_add_u32 v26, 82176, v23 // (op 8315) 
s_waitcnt vmcnt(12)
ds_write_b32 v26, v25 // (op 8315) Store local data 


s_barrier  // (op 9454) 



ds_read_b128 v[28:31], v4 // (op 8370) Load local data 
ds_read_b128 v[32:35], v4 offset:64 // (op 8370) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 9489) 



v_add_u32 v23, 81920, v5 // (op 8420) 
ds_read_u8 v26, v23 // (op 8420) Load local data 




ds_read_b128 v[36:39], v4 offset:4096 // (op 8375) Load local data 
ds_read_b128 v[40:43], v4 offset:4160 // (op 8375) Load local data 



v_add_u32 v23, 82048, v5 // (op 8425) 
ds_read_u8 v18, v23 // (op 8425) Load local data 

s_barrier  // (op 9438) 



ds_read_b128 v[44:47], v3 offset:8192 // (op 8330) Load local data 
ds_read_b128 v[48:51], v3 offset:8256 // (op 8330) Load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 9471) 



v_add_u32 v19, 82176, v0 // (op 8380) 
ds_read_u8 v23, v19 // (op 8380) Load local data 




ds_read_b128 v[52:55], v3 offset:12288 // (op 8335) Load local data 
ds_read_b128 v[56:59], v3 offset:12352 // (op 8335) Load local data 



v_add_u32 v19, 82304, v0 // (op 8385) 
ds_read_u8 v60, v19 // (op 8385) Load local data 




ds_read_b128 v[62:65], v3 offset:16384 // (op 8340) Load local data 
ds_read_b128 v[66:69], v3 offset:16448 // (op 8340) Load local data 



v_add_u32 v19, 82432, v0 // (op 8390) 
ds_read_u8 v61, v19 // (op 8390) Load local data 




ds_read_b128 v[70:73], v3 offset:20480 // (op 8345) Load local data 
ds_read_b128 v[74:77], v3 offset:20544 // (op 8345) Load local data 



v_add_u32 v19, 82560, v0 // (op 8395) 
ds_read_u8 v78, v19 // (op 8395) Load local data 




ds_read_b128 v[80:83], v3 offset:24576 // (op 8350) Load local data 
ds_read_b128 v[84:87], v3 offset:24640 // (op 8350) Load local data 



v_add_u32 v19, 82688, v0 // (op 8400) 
ds_read_u8 v79, v19 // (op 8400) Load local data 




ds_read_b128 v[88:91], v3 offset:28672 // (op 8355) Load local data 
ds_read_b128 v[92:95], v3 offset:28736 // (op 8355) Load local data 



v_add_u32 v19, 82816, v0 // (op 8405) 
ds_read_u8 v96, v19 // (op 8405) Load local data 




ds_read_b128 v[98:101], v3 offset:32768 // (op 8360) Load local data 
ds_read_b128 v[102:105], v3 offset:32832 // (op 8360) Load local data 



v_add_u32 v19, 82944, v0 // (op 8410) 
ds_read_u8 v97, v19 // (op 8410) Load local data 




ds_read_b128 v[106:109], v3 offset:36864 // (op 8365) Load local data 
ds_read_b128 v[110:113], v3 offset:36928 // (op 8365) Load local data 



v_add_u32 v19, 83072, v0 // (op 8415) 
ds_read_u8 v114, v19 // (op 8415) Load local data 

v_and_b32 v19, -8, v1 // (op 8939) 
v_sub_u32 v115, v1, v19 // (op 8939) 

v_lshlrev_b32 v19, 4, v115 // (op 8939) 

v_lshrrev_b32 v115, 3, v1 // (op 8939) 
v_lshl_add_u32 v116, v115, 7, v19 // (op 8939) 


s_nop 0
v_readfirstlane_b32 s0, v116 // (op 8939) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v19, -8, v1 // (op 8996) 
v_sub_u32 v115, v1, v19 // (op 8996) 

v_lshlrev_b32 v19, 4, v115 // (op 8996) 

v_lshrrev_b32 v115, 3, v1 // (op 8996) 
v_lshl_add_u32 v116, v115, 7, v19 // (op 8996) 


s_nop 0
v_readfirstlane_b32 s1, v116 // (op 8996) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v19, 2, v1 // (op 9053) 
v_and_b32 v115, -4, v1 // (op 9111) 
v_sub_u32 v116, v1, v115 // (op 9111) 

v_and_b32 v115, 4294967292, v1 // (op 9111) 
v_add_u32 v117, v115, v116 // (op 9111) 


v_and_b32 v115, -8, v1 // (op 9179) 
v_sub_u32 v116, v1, v115 // (op 9179) 

v_lshlrev_b32 v115, 4, v116 // (op 9179) 

v_lshrrev_b32 v116, 3, v1 // (op 9179) 
v_lshl_add_u32 v118, v116, 7, v115 // (op 9179) 


s_nop 0
v_readfirstlane_b32 s6, v118 // (op 9179) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v115, -8, v1 // (op 9223) 
v_sub_u32 v116, v1, v115 // (op 9223) 

v_lshlrev_b32 v115, 4, v116 // (op 9223) 

v_lshrrev_b32 v116, 3, v1 // (op 9223) 
v_lshl_add_u32 v118, v116, 7, v115 // (op 9223) 


s_nop 0
v_readfirstlane_b32 s7, v118 // (op 9223) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v115, 2, v1 // (op 9267) 
v_and_b32 v116, -4, v1 // (op 9296) 
v_sub_u32 v118, v1, v116 // (op 9296) 

v_and_b32 v116, 4294967292, v1 // (op 9296) 
v_add_u32 v119, v116, v118 // (op 9296) 


s_mov_b32 s10, 0 // (op 2909) call()
s_ashr_i64 s[14:15], s[36:37], 7 // (op 2908) 
s_mov_b32 s11, s14 // (op 2908) convert

s_and_b32 s14, s11, 4294967294 // (op 2908) 

s_cmp_lt_u32 s10, s14 // (op 2908) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ForLoopBottom_KLoop_2908 // (op 2908) Condition: Top (jump to RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ForLoopBottom_KLoop_2908 if false)
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ForLoopTop_KLoop_2908:
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[44:51], v[28:35], a[0:3], v23, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 90) 
s_barrier  // (op 9457) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[44:51], v[36:43], a[4:7], v23, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 380) 
s_add_u32 s10, s10, 2 // (op 2910) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[52:59], v[28:35], a[8:11], v60, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 677) 
s_lshl_b64 s[14:15], s[24:25], 5 // (op 8578) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[52:59], v[36:43], a[12:15], v60, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 631) 
v_mov_b64 v[120:121], s[14:15] // (op 8578) convert


v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[62:69], v[28:35], a[16:19], v61, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1063) 
v_mov_b64 v[122:123], v[16:17] // (op 8578) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[62:69], v[36:43], a[20:23], v61, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1017) 
v_mov_b64 v[124:125], v[122:123] // (op 8578) 
s_waitcnt lgkmcnt(12)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[70:77], v[28:35], a[24:27], v78, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1449) 
s_mov_b32 m0, s1 // (op 8578) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[70:77], v[36:43], a[28:31], v78, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1403) 
s_mov_b32 s11, 256 // (op 8578) call()
buffer_load_dwordx4 v124, s[52:55], s11 offen offset: 0 lds // (op 8578) Load value direct to lds
v_add_co_u32 v124, s[14:15], v124, v120 // (op 8578) least significant half
v_addc_co_u32 v125, s[14:15], 0, v125, s[14:15] // (op 8578) most significant half

s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[80:87], v[28:35], a[32:35], v79, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1835) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[80:87], v[36:43], a[36:39], v79, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1789) 
s_add_u32 m0, m0, 4096 // (op 8578) 
buffer_load_dwordx4 v124, s[52:55], s11 offen offset: 0 lds // (op 8578) Load value direct to lds



s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[88:95], v[28:35], a[40:43], v96, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2221) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[88:95], v[36:43], a[44:47], v96, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2175) 
v_mov_b32 v116, v27 // (op 25) 

buffer_load_ubyte v22, v116, s[60:63], 0 offen offset:8 // (op 25) Load value
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[98:105], v[28:35], a[48:51], v97, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2607) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[98:105], v[36:43], a[52:55], v97, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2561) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 9441) 

s_lshl_b64 s[14:15], s[12:13], 5 // (op 8573) 
v_mov_b64 v[120:121], s[14:15] // (op 8573) convert


v_mov_b64 v[124:125], v[20:21] // (op 8573) 
v_mov_b64 v[126:127], v[124:125] // (op 8573) 
s_add_u32 s11, 8192, s0 // (op 8573) 
s_mov_b32 m0, s11 // (op 8573) reg expression
s_mov_b32 s14, 256 // (op 8573) call()
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[106:113], v[28:35], a[56:59], v114, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2993) 
buffer_load_dwordx4 v126, s[48:51], s14 offen offset: 0 lds // (op 8573) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[106:113], v[36:43], a[60:63], v114, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2947) 
v_add_co_u32 v126, s[18:19], v126, v120 // (op 8573) least significant half
v_addc_co_u32 v127, s[18:19], 0, v127, s[18:19] // (op 8573) most significant half

s_add_u32 m0, m0, 4096 // (op 8573) 
buffer_load_dwordx4 v126, s[48:51], s14 offen offset: 0 lds // (op 8573) Load value direct to lds
v_add_co_u32 v126, s[18:19], v126, v120 // (op 8573) least significant half
v_addc_co_u32 v127, s[18:19], 0, v127, s[18:19] // (op 8573) most significant half

s_add_u32 m0, m0, 4096 // (op 8573) 
buffer_load_dwordx4 v126, s[48:51], s14 offen offset: 0 lds // (op 8573) Load value direct to lds
v_add_co_u32 v126, s[18:19], v126, v120 // (op 8573) least significant half
v_addc_co_u32 v127, s[18:19], 0, v127, s[18:19] // (op 8573) most significant half

s_add_u32 m0, m0, 4096 // (op 8573) 
buffer_load_dwordx4 v126, s[48:51], s14 offen offset: 0 lds // (op 8573) Load value direct to lds
v_add_co_u32 v126, s[18:19], v126, v120 // (op 8573) least significant half
v_addc_co_u32 v127, s[18:19], 0, v127, s[18:19] // (op 8573) most significant half

s_add_u32 m0, m0, 4096 // (op 8573) 
buffer_load_dwordx4 v126, s[48:51], s14 offen offset: 0 lds // (op 8573) Load value direct to lds
v_add_co_u32 v126, s[18:19], v126, v120 // (op 8573) least significant half
v_addc_co_u32 v127, s[18:19], 0, v127, s[18:19] // (op 8573) most significant half

s_add_u32 m0, m0, 4096 // (op 8573) 
buffer_load_dwordx4 v126, s[48:51], s14 offen offset: 0 lds // (op 8573) Load value direct to lds
v_add_co_u32 v126, s[18:19], v126, v120 // (op 8573) least significant half
v_addc_co_u32 v127, s[18:19], 0, v127, s[18:19] // (op 8573) most significant half

s_add_u32 m0, m0, 4096 // (op 8573) 
buffer_load_dwordx4 v126, s[48:51], s14 offen offset: 0 lds // (op 8573) Load value direct to lds
v_add_co_u32 v126, s[18:19], v126, v120 // (op 8573) least significant half
v_addc_co_u32 v127, s[18:19], 0, v127, s[18:19] // (op 8573) most significant half

s_add_u32 m0, m0, 4096 // (op 8573) 
buffer_load_dwordx4 v126, s[48:51], s14 offen offset: 0 lds // (op 8573) Load value direct to lds






v_mov_b32 v118, v24 // (op 17) 

buffer_load_dword v25, v118, s[56:59], 0 offen offset:8 // (op 17) Load value
s_barrier  // (op 9529) 



v_add_u32 v120, 83200, v119 // (op 259) 
s_waitcnt vmcnt(21)
ds_write_b8 v120, v14 // (op 259) Store local data 

s_barrier  // (op 9523) 



v_add_u32 v120, 83456, v115 // (op 258) 
s_waitcnt vmcnt(12)
ds_write_b32 v120, v15 // (op 258) Store local data 

s_barrier  // (op 9512) 



ds_read_b128 v[126:129], v8 offset:40960 // (op 251) Load local data 
ds_read_b128 v[130:133], v8 offset:41024 // (op 251) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 9526) 



v_add_u32 v120, 83200, v9 // (op 255) 
ds_read_u8 v121, v120 // (op 255) Load local data 




ds_read_b128 v[134:137], v8 offset:45056 // (op 385) Load local data 
ds_read_b128 v[138:141], v8 offset:45120 // (op 385) Load local data 



v_add_u32 v120, 83328, v9 // (op 389) 
ds_read_u8 v142, v120 // (op 389) Load local data 

s_barrier  // (op 9506) 
s_barrier  // (op 9574) 



ds_read_b128 v[44:47], v7 offset:49152 // (op 249) Load local data 
ds_read_b128 v[48:51], v7 offset:49216 // (op 249) Load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 9519) 



v_add_u32 v120, 83456, v6 // (op 253) 
ds_read_u8 v23, v120 // (op 253) Load local data 




ds_read_b128 v[52:55], v7 offset:53248 // (op 634) Load local data 
ds_read_b128 v[56:59], v7 offset:53312 // (op 634) Load local data 



v_add_u32 v120, 83584, v6 // (op 637) 
ds_read_u8 v60, v120 // (op 637) Load local data 




ds_read_b128 v[62:65], v7 offset:57344 // (op 1020) Load local data 
ds_read_b128 v[66:69], v7 offset:57408 // (op 1020) Load local data 



v_add_u32 v120, 83712, v6 // (op 1023) 
ds_read_u8 v61, v120 // (op 1023) Load local data 




ds_read_b128 v[70:73], v7 offset:61440 // (op 1406) Load local data 
ds_read_b128 v[74:77], v7 offset:61504 // (op 1406) Load local data 



v_add_u32 v120, 83840, v6 // (op 1409) 
ds_read_u8 v78, v120 // (op 1409) Load local data 




v_add_u32 v120, 65536, v7 // (op 1792) 
ds_read_b128 v[80:83], v120 // (op 1792) Load local data 

v_add_u32 v120, 65600, v7 // (op 1792) 
ds_read_b128 v[84:87], v120 // (op 1792) Load local data 




v_add_u32 v120, 83968, v6 // (op 1795) 
ds_read_u8 v79, v120 // (op 1795) Load local data 




v_add_u32 v120, 69632, v7 // (op 2178) 
ds_read_b128 v[88:91], v120 // (op 2178) Load local data 

v_add_u32 v120, 69696, v7 // (op 2178) 
ds_read_b128 v[92:95], v120 // (op 2178) Load local data 




v_add_u32 v120, 84096, v6 // (op 2181) 
ds_read_u8 v96, v120 // (op 2181) Load local data 




v_add_u32 v120, 73728, v7 // (op 2564) 
ds_read_b128 v[98:101], v120 // (op 2564) Load local data 

v_add_u32 v120, 73792, v7 // (op 2564) 
ds_read_b128 v[102:105], v120 // (op 2564) Load local data 




v_add_u32 v120, 84224, v6 // (op 2567) 
ds_read_u8 v97, v120 // (op 2567) Load local data 




v_add_u32 v120, 77824, v7 // (op 2950) 
ds_read_b128 v[106:109], v120 // (op 2950) Load local data 

v_add_u32 v120, 77888, v7 // (op 2950) 
ds_read_b128 v[110:113], v120 // (op 2950) Load local data 




v_add_u32 v120, 84352, v6 // (op 2953) 
ds_read_u8 v114, v120 // (op 2953) Load local data 

s_barrier  // (op 9620) 
s_waitcnt lgkmcnt(15)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[44:51], v[126:133], a[0:3], v23, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 256) 
s_barrier  // (op 9516) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[44:51], v[134:141], a[4:7], v23, v142 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 381) 
s_lshl_b64 s[14:15], s[24:25], 5 // (op 8601) 
v_mov_b64 v[144:145], s[14:15] // (op 8601) convert


v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[52:59], v[126:133], a[8:11], v60, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 678) 
v_mov_b64 v[146:147], v[122:123] // (op 8601) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[52:59], v[134:141], a[12:15], v60, v142 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 632) 
s_add_u32 s11, 40960, s7 // (op 8601) 
s_mov_b32 m0, s11 // (op 8601) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[62:69], v[126:133], a[16:19], v61, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1064) 
s_mov_b32 s14, 384 // (op 8601) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[62:69], v[134:141], a[20:23], v61, v142 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1018) 
buffer_load_dwordx4 v146, s[52:55], s14 offen offset: 0 lds // (op 8601) Load value direct to lds
v_add_co_u32 v146, s[18:19], v146, v144 // (op 8601) least significant half
s_waitcnt lgkmcnt(12)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[70:77], v[126:133], a[24:27], v78, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1450) 
v_addc_co_u32 v147, s[18:19], 0, v147, s[18:19] // (op 8601) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[70:77], v[134:141], a[28:31], v78, v142 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1404) 
s_add_u32 m0, m0, 4096 // (op 8601) 
buffer_load_dwordx4 v146, s[52:55], s14 offen offset: 0 lds // (op 8601) Load value direct to lds




s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[80:87], v[126:133], a[32:35], v79, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1836) 




v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[80:87], v[134:141], a[36:39], v79, v142 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1790) 
buffer_load_ubyte v14, v116, s[60:63], 0 offen offset:12 // (op 262) Load value

v_add_co_u32 v16, vcc, 256, v16 // (op 8747) least significant half
v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 8747) most significant half
v_add_u32 v27, 8, v27 // (op 8875) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[88:95], v[126:133], a[40:43], v96, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2222) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 9509) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[88:95], v[134:141], a[44:47], v96, v142 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2176) 
s_lshl_b64 s[14:15], s[12:13], 5 // (op 8596) 
v_mov_b64 v[122:123], s[14:15] // (op 8596) convert


v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[98:105], v[126:133], a[48:51], v97, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2608) 
v_mov_b64 v[144:145], v[124:125] // (op 8596) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[98:105], v[134:141], a[52:55], v97, v142 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2562) 
s_add_u32 s11, 49152, s6 // (op 8596) 
s_mov_b32 m0, s11 // (op 8596) reg expression
s_mov_b32 s14, 384 // (op 8596) call()
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[106:113], v[126:133], a[56:59], v114, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2994) 
buffer_load_dwordx4 v144, s[48:51], s14 offen offset: 0 lds // (op 8596) Load value direct to lds


v_add_co_u32 v144, s[18:19], v144, v122 // (op 8596) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[106:113], v[134:141], a[60:63], v114, v142 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2948) 
v_addc_co_u32 v145, s[18:19], 0, v145, s[18:19] // (op 8596) most significant half



s_add_u32 m0, m0, 4096 // (op 8596) 
buffer_load_dwordx4 v144, s[48:51], s14 offen offset: 0 lds // (op 8596) Load value direct to lds
v_add_co_u32 v144, s[18:19], v144, v122 // (op 8596) least significant half
v_addc_co_u32 v145, s[18:19], 0, v145, s[18:19] // (op 8596) most significant half

s_add_u32 m0, m0, 4096 // (op 8596) 
buffer_load_dwordx4 v144, s[48:51], s14 offen offset: 0 lds // (op 8596) Load value direct to lds
v_add_co_u32 v144, s[18:19], v144, v122 // (op 8596) least significant half
v_addc_co_u32 v145, s[18:19], 0, v145, s[18:19] // (op 8596) most significant half

s_add_u32 m0, m0, 4096 // (op 8596) 
buffer_load_dwordx4 v144, s[48:51], s14 offen offset: 0 lds // (op 8596) Load value direct to lds
v_add_co_u32 v144, s[18:19], v144, v122 // (op 8596) least significant half
v_addc_co_u32 v145, s[18:19], 0, v145, s[18:19] // (op 8596) most significant half

s_add_u32 m0, m0, 4096 // (op 8596) 
buffer_load_dwordx4 v144, s[48:51], s14 offen offset: 0 lds // (op 8596) Load value direct to lds
v_add_co_u32 v144, s[18:19], v144, v122 // (op 8596) least significant half
v_addc_co_u32 v145, s[18:19], 0, v145, s[18:19] // (op 8596) most significant half

s_add_u32 m0, m0, 4096 // (op 8596) 
buffer_load_dwordx4 v144, s[48:51], s14 offen offset: 0 lds // (op 8596) Load value direct to lds
v_add_co_u32 v144, s[18:19], v144, v122 // (op 8596) least significant half
v_addc_co_u32 v145, s[18:19], 0, v145, s[18:19] // (op 8596) most significant half

s_add_u32 m0, m0, 4096 // (op 8596) 
buffer_load_dwordx4 v144, s[48:51], s14 offen offset: 0 lds // (op 8596) Load value direct to lds
v_add_co_u32 v144, s[18:19], v144, v122 // (op 8596) least significant half
v_addc_co_u32 v145, s[18:19], 0, v145, s[18:19] // (op 8596) most significant half

s_add_u32 m0, m0, 4096 // (op 8596) 
buffer_load_dwordx4 v144, s[48:51], s14 offen offset: 0 lds // (op 8596) Load value direct to lds








v_add_co_u32 v20, vcc, 256, v20 // (op 8683) least significant half
v_addc_co_u32 v21, vcc, 0, v21, vcc // (op 8683) most significant half
buffer_load_dword v15, v118, s[56:59], 0 offen offset:12 // (op 261) Load value

s_barrier  // (op 9492) 
v_add_u32 v24, 8, v24 // (op 8811) 



v_add_u32 v116, 81920, v117 // (op 30) 
s_waitcnt vmcnt(21)
ds_write_b8 v116, v22 // (op 30) Store local data 

s_barrier  // (op 9474) 



v_add_u32 v116, 82176, v19 // (op 21) 
s_waitcnt vmcnt(12)
ds_write_b32 v116, v25 // (op 21) Store local data 

s_barrier  // (op 9463) 



ds_read_b128 v[28:31], v4 // (op 77) Load local data 
ds_read_b128 v[32:35], v4 offset:64 // (op 77) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 9499) 



v_add_u32 v116, 81920, v5 // (op 87) 
ds_read_u8 v26, v116 // (op 87) Load local data 




ds_read_b128 v[36:39], v4 offset:4096 // (op 375) Load local data 
ds_read_b128 v[40:43], v4 offset:4160 // (op 375) Load local data 



v_add_u32 v116, 82048, v5 // (op 379) 
ds_read_u8 v18, v116 // (op 379) Load local data 

s_barrier  // (op 9447) 



ds_read_b128 v[44:47], v3 offset:8192 // (op 72) Load local data 
ds_read_b128 v[48:51], v3 offset:8256 // (op 72) Load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 9481) 



v_add_u32 v116, 82176, v0 // (op 82) 
ds_read_u8 v23, v116 // (op 82) Load local data 




ds_read_b128 v[52:55], v3 offset:12288 // (op 626) Load local data 
ds_read_b128 v[56:59], v3 offset:12352 // (op 626) Load local data 



v_add_u32 v116, 82304, v0 // (op 629) 
ds_read_u8 v60, v116 // (op 629) Load local data 




ds_read_b128 v[62:65], v3 offset:16384 // (op 1012) Load local data 
ds_read_b128 v[66:69], v3 offset:16448 // (op 1012) Load local data 



v_add_u32 v116, 82432, v0 // (op 1015) 
ds_read_u8 v61, v116 // (op 1015) Load local data 




ds_read_b128 v[70:73], v3 offset:20480 // (op 1398) Load local data 
ds_read_b128 v[74:77], v3 offset:20544 // (op 1398) Load local data 



v_add_u32 v116, 82560, v0 // (op 1401) 
ds_read_u8 v78, v116 // (op 1401) Load local data 




ds_read_b128 v[80:83], v3 offset:24576 // (op 1784) Load local data 
ds_read_b128 v[84:87], v3 offset:24640 // (op 1784) Load local data 



v_add_u32 v116, 82688, v0 // (op 1787) 
ds_read_u8 v79, v116 // (op 1787) Load local data 




ds_read_b128 v[88:91], v3 offset:28672 // (op 2170) Load local data 
ds_read_b128 v[92:95], v3 offset:28736 // (op 2170) Load local data 



v_add_u32 v116, 82816, v0 // (op 2173) 
ds_read_u8 v96, v116 // (op 2173) Load local data 




ds_read_b128 v[98:101], v3 offset:32768 // (op 2556) Load local data 
ds_read_b128 v[102:105], v3 offset:32832 // (op 2556) Load local data 



v_add_u32 v116, 82944, v0 // (op 2559) 
ds_read_u8 v97, v116 // (op 2559) Load local data 




ds_read_b128 v[106:109], v3 offset:36864 // (op 2942) Load local data 
ds_read_b128 v[110:113], v3 offset:36928 // (op 2942) Load local data 



v_add_u32 v116, 83072, v0 // (op 2945) 
ds_read_u8 v114, v116 // (op 2945) Load local data 

s_ashr_i64 s[14:15], s[36:37], 7 // (op 2908) 
s_mov_b32 s11, s14 // (op 2908) convert

s_and_b32 s14, s11, 4294967294 // (op 2908) 

s_cmp_lt_u32 s10, s14 // (op 2908) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ForLoopTop_KLoop_2908 // (op 2908) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ForLoopTop_KLoop_2908 if true)
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ForLoopBottom_KLoop_2908:

























s_barrier  // (op 9610) 
v_lshrrev_b32 v16, 3, v1 // (op 8631) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 8631) 
v_lshl_add_u32 v17, s16, 8, v16 // (op 8631) 

s_lshr_b32 s6, s0, 1 // (op 8631) 

v_and_b32 v16, -8, v1 // (op 8631) 
v_sub_u32 v18, v1, v16 // (op 8631) 

v_lshlrev_b32 v16, 4, v18 // (op 8631) 

v_lshl_add_u32 v18, s6, 8, v16 // (op 8631) 


v_mov_b32 v69, 0 // (op 8631) convert
v_mov_b32 v68, v18 // (op 8631) convert

v_mul_lo_u32 v19, v17, s13 // (op 8631) most significant: low of low * high
v_mul_hi_u32 v16, v17, s12 // (op 8631) most significant: high of low * low
v_mul_lo_u32 v18, v17, s12 // (op 8631) least significant: low of low * low
v_add_u32 v19, v19, v16 // (op 8631) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v68 // (op 8631) least significant half
v_addc_co_u32 v19, s[0:1], v19, v69, s[0:1] // (op 8631) most significant half



s_mul_i32 s1, s12, s23 // (op 8633) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 8633) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 8633) least significant
s_mul_hi_u32 s7, s12, s22 // (op 8633) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8633) most significant: sum
s_add_u32 s1, s1, s7 // (op 8633) most significant: sum


s_mov_b32 s48, s20 // (op 8633) reg expression
s_mov_b32 s49, s21 // (op 8633) reg expression
s_mov_b32 s50, s0 // (op 8633) reg expression
s_mov_b32 s51, 131072 // (op 8633) call()

v_and_b32 v16, -8, v1 // (op 8713) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 8713) 
v_sub_u32 v17, v1, v16 // (op 8713) 

s_lshr_b32 s6, s0, 1 // (op 8713) 

v_lshlrev_b32 v16, 4, v17 // (op 8713) 

v_lshl_add_u32 v17, s6, 8, v16 // (op 8713) 


v_mov_b32 v21, 0 // (op 8713) convert
v_mov_b32 v20, v17 // (op 8713) convert

v_lshrrev_b32 v16, 3, v1 // (op 8713) 
v_lshl_add_u32 v17, s17, 6, v16 // (op 8713) 

v_mul_lo_u32 v23, v17, s25 // (op 8713) most significant: low of low * high
v_mul_hi_u32 v16, v17, s24 // (op 8713) most significant: high of low * low
v_mul_lo_u32 v22, v17, s24 // (op 8713) least significant: low of low * low
v_add_u32 v23, v23, v16 // (op 8713) most significant: sum


v_add_co_u32 v16, s[0:1], v20, v22 // (op 8713) least significant half
v_addc_co_u32 v17, s[0:1], v21, v23, s[0:1] // (op 8713) most significant half



s_mul_i32 s1, s24, s9 // (op 8715) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 8715) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 8715) least significant
s_mul_hi_u32 s7, s24, s8 // (op 8715) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8715) most significant: sum
s_add_u32 s1, s1, s7 // (op 8715) most significant: sum


s_mov_b32 s52, s26 // (op 8715) reg expression
s_mov_b32 s53, s27 // (op 8715) reg expression
s_mov_b32 s54, s0 // (op 8715) reg expression
s_mov_b32 s55, 131072 // (op 8715) call()

v_lshl_add_u32 v20, s16, 8, v1 // (op 8777) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 8777) 
s_lshr_b32 s6, s0, 1 // (op 8777) 

s_lshl_b32 s7, s6, 3 // (op 8777) 

v_mul_lo_u32 v21, v20, s28 // (op 8777) 
v_add_u32 v21, v21, s7 // (op 8777) 


s_mul_i32 s1, s28, s23 // (op 8779) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 8779) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 8779) least significant
s_mul_hi_u32 s7, s28, s22 // (op 8779) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8779) most significant: sum
s_add_u32 s1, s1, s7 // (op 8779) most significant: sum


s_mov_b32 s56, s30 // (op 8779) reg expression
s_mov_b32 s57, s31 // (op 8779) reg expression
s_mov_b32 s58, s0 // (op 8779) reg expression
s_mov_b32 s59, 131072 // (op 8779) call()

v_and_b32 v20, -4, v1 // (op 8841) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 8841) 
v_sub_u32 v22, v1, v20 // (op 8841) 

s_lshr_b32 s6, s0, 1 // (op 8841) 

v_lshrrev_b32 v20, 2, v1 // (op 8841) 
v_lshl_add_u32 v23, s6, 3, v22 // (op 8841) 


v_lshl_add_u32 v22, s17, 6, v20 // (op 8841) 

v_mul_lo_u32 v20, v22, s32 // (op 8841) 

v_add_u32 v22, v23, v20 // (op 8841) 


s_mul_i32 s1, s32, s9 // (op 8843) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 8843) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 8843) least significant
s_mul_hi_u32 s7, s32, s8 // (op 8843) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8843) most significant: sum
s_add_u32 s1, s1, s7 // (op 8843) most significant: sum


s_mov_b32 s60, s34 // (op 8843) reg expression
s_mov_b32 s61, s35 // (op 8843) reg expression
s_mov_b32 s62, s0 // (op 8843) reg expression
s_mov_b32 s63, 131072 // (op 8843) call()

v_and_b32 v20, -8, v1 // (op 8924) 
v_sub_u32 v23, v1, v20 // (op 8924) 

v_lshlrev_b32 v20, 4, v23 // (op 8924) 

v_lshrrev_b32 v23, 3, v1 // (op 8924) 
v_lshl_add_u32 v24, v23, 7, v20 // (op 8924) 


s_nop 0
v_readfirstlane_b32 s0, v24 // (op 8924) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v20, -8, v1 // (op 8983) 
v_sub_u32 v23, v1, v20 // (op 8983) 

v_lshlrev_b32 v20, 4, v23 // (op 8983) 

v_lshrrev_b32 v23, 3, v1 // (op 8983) 
v_lshl_add_u32 v24, v23, 7, v20 // (op 8983) 


s_nop 0
v_readfirstlane_b32 s1, v24 // (op 8983) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v20, 2, v1 // (op 9040) 
v_and_b32 v23, -4, v1 // (op 9098) 
v_sub_u32 v24, v1, v23 // (op 9098) 

v_and_b32 v23, 4294967292, v1 // (op 9098) 
v_add_u32 v25, v23, v24 // (op 9098) 


s_ashr_i64 s[6:7], s[36:37], 7 // (op 2883) 
s_and_b32 s10, s6, 4294967294 // (op 2883) 

s_ashr_i64 s[6:7], s[36:37], 7 // (op 2882) 
s_mov_b32 s11, s6 // (op 2882) convert

s_cmp_lt_u32 s10, s11 // (op 2882) 













s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ForLoopBottom_KLoopTail_2882 // (op 2882) Condition: Top (jump to RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ForLoopBottom_KLoopTail_2882 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ForLoopTop_KLoopTail_2882:
s_barrier  // (op 9444) 
s_add_u32 s10, s10, 1 // (op 2884) 

s_lshl_b64 s[6:7], s[12:13], 5 // (op 8583) 
v_mov_b64 v[14:15], s[6:7] // (op 8583) convert


v_mov_b64 v[26:27], v[18:19] // (op 8583) 
v_mov_b64 v[28:29], v[26:27] // (op 8583) 
s_add_u32 s11, 8192, s0 // (op 8583) 
s_mov_b32 m0, s11 // (op 8583) reg expression
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 8583) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v14 // (op 8583) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 8583) most significant half

s_add_u32 m0, m0, 4096 // (op 8583) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 8583) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v14 // (op 8583) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 8583) most significant half

s_add_u32 m0, m0, 4096 // (op 8583) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 8583) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v14 // (op 8583) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 8583) most significant half

s_add_u32 m0, m0, 4096 // (op 8583) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 8583) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v14 // (op 8583) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 8583) most significant half

s_add_u32 m0, m0, 4096 // (op 8583) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 8583) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v14 // (op 8583) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 8583) most significant half

s_add_u32 m0, m0, 4096 // (op 8583) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 8583) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v14 // (op 8583) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 8583) most significant half

s_add_u32 m0, m0, 4096 // (op 8583) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 8583) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v14 // (op 8583) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 8583) most significant half

s_add_u32 m0, m0, 4096 // (op 8583) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 8583) Load value direct to lds




v_add_co_u32 v18, vcc, 128, v18 // (op 8621) least significant half
v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 8621) most significant half
s_barrier  // (op 9460) 

s_lshl_b64 s[6:7], s[24:25], 5 // (op 8589) 
v_mov_b64 v[14:15], s[6:7] // (op 8589) convert


v_mov_b64 v[26:27], v[16:17] // (op 8589) 
v_mov_b64 v[28:29], v[26:27] // (op 8589) 
s_mov_b32 m0, s1 // (op 8589) reg expression
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 8589) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v14 // (op 8589) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 8589) most significant half

s_add_u32 m0, m0, 4096 // (op 8589) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 8589) Load value direct to lds



v_add_co_u32 v16, vcc, 128, v16 // (op 8705) least significant half
v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 8705) most significant half


v_mov_b32 v14, v21 // (op 195) 

buffer_load_dword v15, v14, s[56:59], 0 offen // (op 195) Load value

v_add_u32 v21, 4, v21 // (op 8769) 


s_barrier  // (op 9477) 
v_mov_b32 v14, v22 // (op 196) 




v_add_u32 v23, 82176, v20 // (op 192) 
buffer_load_ubyte v24, v14, s[60:63], 0 offen offset: 0 // (op 196) Load value

v_add_u32 v22, 4, v22 // (op 8833) 
s_waitcnt vmcnt(1)
ds_write_b32 v23, v15 // (op 192) Store local data 


s_barrier  // (op 9450) 



s_barrier  // (op 9495) 
ds_read_b128 v[26:29], v3 offset:8192 // (op 183) Load local data 



ds_read_b128 v[30:33], v3 offset:8256 // (op 183) Load local data 
v_add_u32 v23, 81920, v25 // (op 193) 
s_barrier  // (op 9467) 
s_waitcnt vmcnt(0)
ds_write_b8 v23, v24 // (op 193) Store local data 





ds_read_b128 v[34:37], v4 // (op 185) Load local data 
ds_read_b128 v[38:41], v4 offset:64 // (op 185) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 9485) 



v_add_u32 v14, 82176, v0 // (op 187) 
ds_read_u8 v15, v14 // (op 187) Load local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 9502) 



v_add_u32 v14, 81920, v5 // (op 189) 
ds_read_u8 v23, v14 // (op 189) Load local data 




ds_read_b128 v[42:45], v4 offset:4096 // (op 354) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:33], v[34:41], a[0:3], v15, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 190) 
ds_read_b128 v[46:49], v4 offset:4160 // (op 354) Load local data 



v_add_u32 v14, 82048, v5 // (op 358) 
ds_read_u8 v24, v14 // (op 358) Load local data 




ds_read_b128 v[50:53], v3 offset:12288 // (op 572) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[26:33], v[42:49], a[4:7], v15, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 359) 
ds_read_b128 v[54:57], v3 offset:12352 // (op 572) Load local data 





v_add_u32 v14, 82304, v0 // (op 575) 
ds_read_u8 v15, v14 // (op 575) Load local data 




ds_read_b128 v[26:29], v3 offset:16384 // (op 958) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[50:57], v[34:41], a[8:11], v15, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 577) 
ds_read_b128 v[30:33], v3 offset:16448 // (op 958) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[50:57], v[42:49], a[12:15], v15, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 608) 
v_add_u32 v14, 82432, v0 // (op 961) 


ds_read_u8 v15, v14 // (op 961) Load local data 




ds_read_b128 v[50:53], v3 offset:20480 // (op 1344) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[26:33], v[34:41], a[16:19], v15, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 963) 
ds_read_b128 v[54:57], v3 offset:20544 // (op 1344) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[26:33], v[42:49], a[20:23], v15, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 994) 
v_add_u32 v14, 82560, v0 // (op 1347) 


ds_read_u8 v15, v14 // (op 1347) Load local data 




ds_read_b128 v[26:29], v3 offset:24576 // (op 1730) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[50:57], v[34:41], a[24:27], v15, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1349) 
ds_read_b128 v[30:33], v3 offset:24640 // (op 1730) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[50:57], v[42:49], a[28:31], v15, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1380) 
v_add_u32 v14, 82688, v0 // (op 1733) 


ds_read_u8 v15, v14 // (op 1733) Load local data 




ds_read_b128 v[50:53], v3 offset:28672 // (op 2116) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[26:33], v[34:41], a[32:35], v15, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1735) 
ds_read_b128 v[54:57], v3 offset:28736 // (op 2116) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[26:33], v[42:49], a[36:39], v15, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1766) 
v_add_u32 v14, 82816, v0 // (op 2119) 


ds_read_u8 v15, v14 // (op 2119) Load local data 




ds_read_b128 v[26:29], v3 offset:32768 // (op 2502) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[50:57], v[34:41], a[40:43], v15, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2121) 
ds_read_b128 v[30:33], v3 offset:32832 // (op 2502) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[50:57], v[42:49], a[44:47], v15, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2152) 
v_add_u32 v14, 82944, v0 // (op 2505) 


ds_read_u8 v15, v14 // (op 2505) Load local data 




ds_read_b128 v[50:53], v3 offset:36864 // (op 2888) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[26:33], v[34:41], a[48:51], v15, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2507) 
ds_read_b128 v[54:57], v3 offset:36928 // (op 2888) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[26:33], v[42:49], a[52:55], v15, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2538) 
v_add_u32 v14, 83072, v0 // (op 2891) 


ds_read_u8 v15, v14 // (op 2891) Load local data 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[50:57], v[34:41], a[56:59], v15, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2893) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[50:57], v[42:49], a[60:63], v15, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2924) 




s_ashr_i64 s[6:7], s[36:37], 7 // (op 2882) 
s_mov_b32 s11, s6 // (op 2882) convert

s_cmp_lt_u32 s10, s11 // (op 2882) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ForLoopTop_KLoopTail_2882 // (op 2882) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ForLoopTop_KLoopTail_2882 if true)
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ForLoopBottom_KLoopTail_2882:























s_barrier  // (op 9602) 









v_cmp_eq_f32 vcc, 0.00000, v10 // (op 135) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ConditionalFalse_0_beta_135 // (op 135) Condition: False, jump to RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ConditionalFalse_0_beta_135
v_accvgpr_read v6, a0 // (op 50) 
v_accvgpr_read v7, a1 // (op 50) 
v_accvgpr_read v8, a2 // (op 50) 
v_accvgpr_read v9, a3 // (op 50) 
v_mul_f32 v14, v11, v6 // (op 50) 
v_mul_f32 v15, v11, v7 // (op 50) 
v_mul_f32 v16, v11, v8 // (op 50) 
v_mul_f32 v17, v11, v9 // (op 50) 

v_lshrrev_b32 v6, 16, v14 // (op 56) 
v_lshrrev_b32 v7, 16, v15 // (op 56) 
v_lshrrev_b32 v8, 16, v16 // (op 56) 
v_lshrrev_b32 v9, 16, v17 // (op 56) 


v_pack_b32_f16 v4, v6, v7 // (op 58) 
v_pack_b32_f16 v5, v8, v9 // (op 58) 

v_mov_b32 v0, v13 // (op 58) reg expression

buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 58) Store value



v_accvgpr_read v6, a4 // (op 3577) 
v_accvgpr_read v7, a5 // (op 3577) 
v_accvgpr_read v8, a6 // (op 3577) 
v_accvgpr_read v9, a7 // (op 3577) 
v_mul_f32 v14, v11, v6 // (op 3577) 
v_mul_f32 v15, v11, v7 // (op 3577) 
v_mul_f32 v16, v11, v8 // (op 3577) 
v_mul_f32 v17, v11, v9 // (op 3577) 

v_lshrrev_b32 v6, 16, v14 // (op 3579) 
v_lshrrev_b32 v7, 16, v15 // (op 3579) 
v_lshrrev_b32 v8, 16, v16 // (op 3579) 
v_lshrrev_b32 v9, 16, v17 // (op 3579) 


v_pack_b32_f16 v4, v6, v7 // (op 3580) 
v_pack_b32_f16 v5, v8, v9 // (op 3580) 

s_lshl_b32 s0, s4, 6 // (op 3580) 
v_add_u32 v0, v13, s0 // (op 3580) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3580) Store value



v_accvgpr_read v6, a8 // (op 3603) 
v_accvgpr_read v7, a9 // (op 3603) 
v_accvgpr_read v8, a10 // (op 3603) 
v_accvgpr_read v9, a11 // (op 3603) 
v_mul_f32 v14, v11, v6 // (op 3603) 
v_mul_f32 v15, v11, v7 // (op 3603) 
v_mul_f32 v16, v11, v8 // (op 3603) 
v_mul_f32 v17, v11, v9 // (op 3603) 

v_lshrrev_b32 v6, 16, v14 // (op 3605) 
v_lshrrev_b32 v7, 16, v15 // (op 3605) 
v_lshrrev_b32 v8, 16, v16 // (op 3605) 
v_lshrrev_b32 v9, 16, v17 // (op 3605) 


v_pack_b32_f16 v4, v6, v7 // (op 3607) 
v_pack_b32_f16 v5, v8, v9 // (op 3607) 

v_add_u32 v0, 64, v13 // (op 3607) 

buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3607) Store value



v_accvgpr_read v6, a12 // (op 3613) 
v_accvgpr_read v7, a13 // (op 3613) 
v_accvgpr_read v8, a14 // (op 3613) 
v_accvgpr_read v9, a15 // (op 3613) 
v_mul_f32 v14, v11, v6 // (op 3613) 
v_mul_f32 v15, v11, v7 // (op 3613) 
v_mul_f32 v16, v11, v8 // (op 3613) 
v_mul_f32 v17, v11, v9 // (op 3613) 

v_lshrrev_b32 v6, 16, v14 // (op 3615) 
v_lshrrev_b32 v7, 16, v15 // (op 3615) 
v_lshrrev_b32 v8, 16, v16 // (op 3615) 
v_lshrrev_b32 v9, 16, v17 // (op 3615) 


v_pack_b32_f16 v4, v6, v7 // (op 3609) 
v_pack_b32_f16 v5, v8, v9 // (op 3609) 

s_lshl_b32 s0, s4, 6 // (op 3609) 
s_add_u32 s0, s0, 64 // (op 3609) 
v_add_u32 v0, v13, s0 // (op 3609) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3609) Store value



v_accvgpr_read v6, a16 // (op 3643) 
v_accvgpr_read v7, a17 // (op 3643) 
v_accvgpr_read v8, a18 // (op 3643) 
v_accvgpr_read v9, a19 // (op 3643) 
v_mul_f32 v14, v11, v6 // (op 3643) 
v_mul_f32 v15, v11, v7 // (op 3643) 
v_mul_f32 v16, v11, v8 // (op 3643) 
v_mul_f32 v17, v11, v9 // (op 3643) 

v_lshrrev_b32 v6, 16, v14 // (op 3645) 
v_lshrrev_b32 v7, 16, v15 // (op 3645) 
v_lshrrev_b32 v8, 16, v16 // (op 3645) 
v_lshrrev_b32 v9, 16, v17 // (op 3645) 


v_pack_b32_f16 v4, v6, v7 // (op 3647) 
v_pack_b32_f16 v5, v8, v9 // (op 3647) 

v_add_u32 v0, 128, v13 // (op 3647) 

buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3647) Store value



v_accvgpr_read v6, a20 // (op 3653) 
v_accvgpr_read v7, a21 // (op 3653) 
v_accvgpr_read v8, a22 // (op 3653) 
v_accvgpr_read v9, a23 // (op 3653) 
v_mul_f32 v14, v11, v6 // (op 3653) 
v_mul_f32 v15, v11, v7 // (op 3653) 
v_mul_f32 v16, v11, v8 // (op 3653) 
v_mul_f32 v17, v11, v9 // (op 3653) 

v_lshrrev_b32 v6, 16, v14 // (op 3655) 
v_lshrrev_b32 v7, 16, v15 // (op 3655) 
v_lshrrev_b32 v8, 16, v16 // (op 3655) 
v_lshrrev_b32 v9, 16, v17 // (op 3655) 


v_pack_b32_f16 v4, v6, v7 // (op 3649) 
v_pack_b32_f16 v5, v8, v9 // (op 3649) 

s_lshl_b32 s0, s4, 6 // (op 3649) 
s_add_u32 s0, s0, 128 // (op 3649) 
v_add_u32 v0, v13, s0 // (op 3649) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3649) Store value



v_accvgpr_read v6, a24 // (op 3683) 
v_accvgpr_read v7, a25 // (op 3683) 
v_accvgpr_read v8, a26 // (op 3683) 
v_accvgpr_read v9, a27 // (op 3683) 
v_mul_f32 v14, v11, v6 // (op 3683) 
v_mul_f32 v15, v11, v7 // (op 3683) 
v_mul_f32 v16, v11, v8 // (op 3683) 
v_mul_f32 v17, v11, v9 // (op 3683) 

v_lshrrev_b32 v6, 16, v14 // (op 3685) 
v_lshrrev_b32 v7, 16, v15 // (op 3685) 
v_lshrrev_b32 v8, 16, v16 // (op 3685) 
v_lshrrev_b32 v9, 16, v17 // (op 3685) 


v_pack_b32_f16 v4, v6, v7 // (op 3687) 
v_pack_b32_f16 v5, v8, v9 // (op 3687) 

v_add_u32 v0, 192, v13 // (op 3687) 

buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3687) Store value



v_accvgpr_read v6, a28 // (op 3693) 
v_accvgpr_read v7, a29 // (op 3693) 
v_accvgpr_read v8, a30 // (op 3693) 
v_accvgpr_read v9, a31 // (op 3693) 
v_mul_f32 v14, v11, v6 // (op 3693) 
v_mul_f32 v15, v11, v7 // (op 3693) 
v_mul_f32 v16, v11, v8 // (op 3693) 
v_mul_f32 v17, v11, v9 // (op 3693) 

v_lshrrev_b32 v6, 16, v14 // (op 3695) 
v_lshrrev_b32 v7, 16, v15 // (op 3695) 
v_lshrrev_b32 v8, 16, v16 // (op 3695) 
v_lshrrev_b32 v9, 16, v17 // (op 3695) 


v_pack_b32_f16 v4, v6, v7 // (op 3689) 
v_pack_b32_f16 v5, v8, v9 // (op 3689) 

s_lshl_b32 s0, s4, 6 // (op 3689) 
s_add_u32 s0, s0, 192 // (op 3689) 
v_add_u32 v0, v13, s0 // (op 3689) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3689) Store value



v_accvgpr_read v6, a32 // (op 3723) 
v_accvgpr_read v7, a33 // (op 3723) 
v_accvgpr_read v8, a34 // (op 3723) 
v_accvgpr_read v9, a35 // (op 3723) 
v_mul_f32 v14, v11, v6 // (op 3723) 
v_mul_f32 v15, v11, v7 // (op 3723) 
v_mul_f32 v16, v11, v8 // (op 3723) 
v_mul_f32 v17, v11, v9 // (op 3723) 

v_lshrrev_b32 v6, 16, v14 // (op 3725) 
v_lshrrev_b32 v7, 16, v15 // (op 3725) 
v_lshrrev_b32 v8, 16, v16 // (op 3725) 
v_lshrrev_b32 v9, 16, v17 // (op 3725) 


v_pack_b32_f16 v4, v6, v7 // (op 3727) 
v_pack_b32_f16 v5, v8, v9 // (op 3727) 

v_add_u32 v0, 256, v13 // (op 3727) 

buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3727) Store value



v_accvgpr_read v6, a36 // (op 3733) 
v_accvgpr_read v7, a37 // (op 3733) 
v_accvgpr_read v8, a38 // (op 3733) 
v_accvgpr_read v9, a39 // (op 3733) 
v_mul_f32 v14, v11, v6 // (op 3733) 
v_mul_f32 v15, v11, v7 // (op 3733) 
v_mul_f32 v16, v11, v8 // (op 3733) 
v_mul_f32 v17, v11, v9 // (op 3733) 

v_lshrrev_b32 v6, 16, v14 // (op 3735) 
v_lshrrev_b32 v7, 16, v15 // (op 3735) 
v_lshrrev_b32 v8, 16, v16 // (op 3735) 
v_lshrrev_b32 v9, 16, v17 // (op 3735) 


v_pack_b32_f16 v4, v6, v7 // (op 3729) 
v_pack_b32_f16 v5, v8, v9 // (op 3729) 

s_lshl_b32 s0, s4, 6 // (op 3729) 
s_add_u32 s0, s0, 256 // (op 3729) 
v_add_u32 v0, v13, s0 // (op 3729) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3729) Store value



v_accvgpr_read v6, a40 // (op 3763) 
v_accvgpr_read v7, a41 // (op 3763) 
v_accvgpr_read v8, a42 // (op 3763) 
v_accvgpr_read v9, a43 // (op 3763) 
v_mul_f32 v14, v11, v6 // (op 3763) 
v_mul_f32 v15, v11, v7 // (op 3763) 
v_mul_f32 v16, v11, v8 // (op 3763) 
v_mul_f32 v17, v11, v9 // (op 3763) 

v_lshrrev_b32 v6, 16, v14 // (op 3765) 
v_lshrrev_b32 v7, 16, v15 // (op 3765) 
v_lshrrev_b32 v8, 16, v16 // (op 3765) 
v_lshrrev_b32 v9, 16, v17 // (op 3765) 


v_pack_b32_f16 v4, v6, v7 // (op 3767) 
v_pack_b32_f16 v5, v8, v9 // (op 3767) 

v_add_u32 v0, 320, v13 // (op 3767) 

buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3767) Store value



v_accvgpr_read v6, a44 // (op 3773) 
v_accvgpr_read v7, a45 // (op 3773) 
v_accvgpr_read v8, a46 // (op 3773) 
v_accvgpr_read v9, a47 // (op 3773) 
v_mul_f32 v14, v11, v6 // (op 3773) 
v_mul_f32 v15, v11, v7 // (op 3773) 
v_mul_f32 v16, v11, v8 // (op 3773) 
v_mul_f32 v17, v11, v9 // (op 3773) 

v_lshrrev_b32 v6, 16, v14 // (op 3775) 
v_lshrrev_b32 v7, 16, v15 // (op 3775) 
v_lshrrev_b32 v8, 16, v16 // (op 3775) 
v_lshrrev_b32 v9, 16, v17 // (op 3775) 


v_pack_b32_f16 v4, v6, v7 // (op 3769) 
v_pack_b32_f16 v5, v8, v9 // (op 3769) 

s_lshl_b32 s0, s4, 6 // (op 3769) 
s_add_u32 s0, s0, 320 // (op 3769) 
v_add_u32 v0, v13, s0 // (op 3769) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3769) Store value



v_accvgpr_read v6, a48 // (op 3803) 
v_accvgpr_read v7, a49 // (op 3803) 
v_accvgpr_read v8, a50 // (op 3803) 
v_accvgpr_read v9, a51 // (op 3803) 
v_mul_f32 v14, v11, v6 // (op 3803) 
v_mul_f32 v15, v11, v7 // (op 3803) 
v_mul_f32 v16, v11, v8 // (op 3803) 
v_mul_f32 v17, v11, v9 // (op 3803) 

v_lshrrev_b32 v6, 16, v14 // (op 3805) 
v_lshrrev_b32 v7, 16, v15 // (op 3805) 
v_lshrrev_b32 v8, 16, v16 // (op 3805) 
v_lshrrev_b32 v9, 16, v17 // (op 3805) 


v_pack_b32_f16 v4, v6, v7 // (op 3807) 
v_pack_b32_f16 v5, v8, v9 // (op 3807) 

v_add_u32 v0, 384, v13 // (op 3807) 

buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3807) Store value



v_accvgpr_read v6, a52 // (op 3813) 
v_accvgpr_read v7, a53 // (op 3813) 
v_accvgpr_read v8, a54 // (op 3813) 
v_accvgpr_read v9, a55 // (op 3813) 
v_mul_f32 v14, v11, v6 // (op 3813) 
v_mul_f32 v15, v11, v7 // (op 3813) 
v_mul_f32 v16, v11, v8 // (op 3813) 
v_mul_f32 v17, v11, v9 // (op 3813) 

v_lshrrev_b32 v6, 16, v14 // (op 3815) 
v_lshrrev_b32 v7, 16, v15 // (op 3815) 
v_lshrrev_b32 v8, 16, v16 // (op 3815) 
v_lshrrev_b32 v9, 16, v17 // (op 3815) 


v_pack_b32_f16 v4, v6, v7 // (op 3809) 
v_pack_b32_f16 v5, v8, v9 // (op 3809) 

s_lshl_b32 s0, s4, 6 // (op 3809) 
s_add_u32 s0, s0, 384 // (op 3809) 
v_add_u32 v0, v13, s0 // (op 3809) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3809) Store value



v_accvgpr_read v6, a56 // (op 3843) 
v_accvgpr_read v7, a57 // (op 3843) 
v_accvgpr_read v8, a58 // (op 3843) 
v_accvgpr_read v9, a59 // (op 3843) 
v_mul_f32 v14, v11, v6 // (op 3843) 
v_mul_f32 v15, v11, v7 // (op 3843) 
v_mul_f32 v16, v11, v8 // (op 3843) 
v_mul_f32 v17, v11, v9 // (op 3843) 

v_lshrrev_b32 v6, 16, v14 // (op 3845) 
v_lshrrev_b32 v7, 16, v15 // (op 3845) 
v_lshrrev_b32 v8, 16, v16 // (op 3845) 
v_lshrrev_b32 v9, 16, v17 // (op 3845) 


v_pack_b32_f16 v4, v6, v7 // (op 3847) 
v_pack_b32_f16 v5, v8, v9 // (op 3847) 

v_add_u32 v0, 448, v13 // (op 3847) 

buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3847) Store value



v_accvgpr_read v6, a60 // (op 3853) 
v_accvgpr_read v7, a61 // (op 3853) 
v_accvgpr_read v8, a62 // (op 3853) 
v_accvgpr_read v9, a63 // (op 3853) 
v_mul_f32 v14, v11, v6 // (op 3853) 
v_mul_f32 v15, v11, v7 // (op 3853) 
v_mul_f32 v16, v11, v8 // (op 3853) 
v_mul_f32 v17, v11, v9 // (op 3853) 

v_lshrrev_b32 v6, 16, v14 // (op 3855) 
v_lshrrev_b32 v7, 16, v15 // (op 3855) 
v_lshrrev_b32 v8, 16, v16 // (op 3855) 
v_lshrrev_b32 v9, 16, v17 // (op 3855) 


v_pack_b32_f16 v4, v6, v7 // (op 3849) 
v_pack_b32_f16 v5, v8, v9 // (op 3849) 

s_lshl_b32 s0, s4, 6 // (op 3849) 
s_add_u32 s0, s0, 448 // (op 3849) 
v_add_u32 v0, v13, s0 // (op 3849) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3849) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ConditionalBottom_0_beta_135 // (op 135) Condition: Done, jump to RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ConditionalBottom_0_beta_135
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ConditionalFalse_0_beta_135:


v_add_u32 v0, 448, v12 // (op 4257) 

buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 4257) Load value

v_accvgpr_read v4, a56 // (op 4248) 
v_accvgpr_read v5, a57 // (op 4248) 
v_accvgpr_read v6, a58 // (op 4248) 
v_accvgpr_read v7, a59 // (op 4248) 
v_mul_f32 v14, v11, v4 // (op 4248) 
v_mul_f32 v15, v11, v5 // (op 4248) 
v_mul_f32 v16, v11, v6 // (op 4248) 
v_mul_f32 v17, v11, v7 // (op 4248) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4256) 
v_lshrrev_b32 v7, 16, v8 // (op 4256) 
v_lshlrev_b32 v6, 16, v6 // (op 4256) 
v_lshlrev_b32 v7, 16, v7 // (op 4256) 
v_mul_f32 v18, v10, v6 // (op 4256) 
v_mul_f32 v19, v10, v7 // (op 4256) 
v_and_b32 v6, 65535, v9 // (op 4256) 
v_lshrrev_b32 v7, 16, v9 // (op 4256) 
v_lshlrev_b32 v6, 16, v6 // (op 4256) 
v_lshlrev_b32 v7, 16, v7 // (op 4256) 
v_mul_f32 v20, v10, v6 // (op 4256) 
v_mul_f32 v21, v10, v7 // (op 4256) 


v_add_f32 v6, v18, v14 // (op 4249) 
v_add_f32 v7, v19, v15 // (op 4249) 
v_add_f32 v8, v20, v16 // (op 4249) 
v_add_f32 v9, v21, v17 // (op 4249) 


v_lshrrev_b32 v14, 16, v6 // (op 4250) 
v_lshrrev_b32 v15, 16, v7 // (op 4250) 
v_lshrrev_b32 v16, 16, v8 // (op 4250) 
v_lshrrev_b32 v17, 16, v9 // (op 4250) 


v_pack_b32_f16 v8, v14, v15 // (op 4252) 
v_pack_b32_f16 v9, v16, v17 // (op 4252) 

v_add_u32 v0, 448, v13 // (op 4252) 

buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 4252) Store value





s_lshl_b32 s0, s2, 6 // (op 4262) 
s_add_u32 s0, s0, 448 // (op 4262) 
v_add_u32 v0, v12, s0 // (op 4262) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 4262) Load value

v_accvgpr_read v4, a60 // (op 4264) 
v_accvgpr_read v5, a61 // (op 4264) 
v_accvgpr_read v6, a62 // (op 4264) 
v_accvgpr_read v7, a63 // (op 4264) 
v_mul_f32 v14, v11, v4 // (op 4264) 
v_mul_f32 v15, v11, v5 // (op 4264) 
v_mul_f32 v16, v11, v6 // (op 4264) 
v_mul_f32 v17, v11, v7 // (op 4264) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4259) 
v_lshrrev_b32 v7, 16, v8 // (op 4259) 
v_lshlrev_b32 v6, 16, v6 // (op 4259) 
v_lshlrev_b32 v7, 16, v7 // (op 4259) 
v_mul_f32 v18, v10, v6 // (op 4259) 
v_mul_f32 v19, v10, v7 // (op 4259) 
v_and_b32 v6, 65535, v9 // (op 4259) 
v_lshrrev_b32 v7, 16, v9 // (op 4259) 
v_lshlrev_b32 v6, 16, v6 // (op 4259) 
v_lshlrev_b32 v7, 16, v7 // (op 4259) 
v_mul_f32 v20, v10, v6 // (op 4259) 
v_mul_f32 v21, v10, v7 // (op 4259) 


v_add_f32 v6, v18, v14 // (op 4260) 
v_add_f32 v7, v19, v15 // (op 4260) 
v_add_f32 v8, v20, v16 // (op 4260) 
v_add_f32 v9, v21, v17 // (op 4260) 


v_lshrrev_b32 v14, 16, v6 // (op 4261) 
v_lshrrev_b32 v15, 16, v7 // (op 4261) 
v_lshrrev_b32 v16, 16, v8 // (op 4261) 
v_lshrrev_b32 v17, 16, v9 // (op 4261) 


v_pack_b32_f16 v8, v14, v15 // (op 4254) 
v_pack_b32_f16 v9, v16, v17 // (op 4254) 

s_lshl_b32 s0, s4, 6 // (op 4254) 
s_add_u32 s0, s0, 448 // (op 4254) 
v_add_u32 v0, v13, s0 // (op 4254) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 4254) Store value





v_add_u32 v0, 384, v12 // (op 4212) 

buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 4212) Load value

v_accvgpr_read v4, a48 // (op 4203) 
v_accvgpr_read v5, a49 // (op 4203) 
v_accvgpr_read v6, a50 // (op 4203) 
v_accvgpr_read v7, a51 // (op 4203) 
v_mul_f32 v14, v11, v4 // (op 4203) 
v_mul_f32 v15, v11, v5 // (op 4203) 
v_mul_f32 v16, v11, v6 // (op 4203) 
v_mul_f32 v17, v11, v7 // (op 4203) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4211) 
v_lshrrev_b32 v7, 16, v8 // (op 4211) 
v_lshlrev_b32 v6, 16, v6 // (op 4211) 
v_lshlrev_b32 v7, 16, v7 // (op 4211) 
v_mul_f32 v18, v10, v6 // (op 4211) 
v_mul_f32 v19, v10, v7 // (op 4211) 
v_and_b32 v6, 65535, v9 // (op 4211) 
v_lshrrev_b32 v7, 16, v9 // (op 4211) 
v_lshlrev_b32 v6, 16, v6 // (op 4211) 
v_lshlrev_b32 v7, 16, v7 // (op 4211) 
v_mul_f32 v20, v10, v6 // (op 4211) 
v_mul_f32 v21, v10, v7 // (op 4211) 


v_add_f32 v6, v18, v14 // (op 4204) 
v_add_f32 v7, v19, v15 // (op 4204) 
v_add_f32 v8, v20, v16 // (op 4204) 
v_add_f32 v9, v21, v17 // (op 4204) 


v_lshrrev_b32 v14, 16, v6 // (op 4205) 
v_lshrrev_b32 v15, 16, v7 // (op 4205) 
v_lshrrev_b32 v16, 16, v8 // (op 4205) 
v_lshrrev_b32 v17, 16, v9 // (op 4205) 


v_pack_b32_f16 v8, v14, v15 // (op 4207) 
v_pack_b32_f16 v9, v16, v17 // (op 4207) 

v_add_u32 v0, 384, v13 // (op 4207) 

buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 4207) Store value





s_lshl_b32 s0, s2, 6 // (op 4217) 
s_add_u32 s0, s0, 384 // (op 4217) 
v_add_u32 v0, v12, s0 // (op 4217) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 4217) Load value

v_accvgpr_read v4, a52 // (op 4219) 
v_accvgpr_read v5, a53 // (op 4219) 
v_accvgpr_read v6, a54 // (op 4219) 
v_accvgpr_read v7, a55 // (op 4219) 
v_mul_f32 v14, v11, v4 // (op 4219) 
v_mul_f32 v15, v11, v5 // (op 4219) 
v_mul_f32 v16, v11, v6 // (op 4219) 
v_mul_f32 v17, v11, v7 // (op 4219) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4214) 
v_lshrrev_b32 v7, 16, v8 // (op 4214) 
v_lshlrev_b32 v6, 16, v6 // (op 4214) 
v_lshlrev_b32 v7, 16, v7 // (op 4214) 
v_mul_f32 v18, v10, v6 // (op 4214) 
v_mul_f32 v19, v10, v7 // (op 4214) 
v_and_b32 v6, 65535, v9 // (op 4214) 
v_lshrrev_b32 v7, 16, v9 // (op 4214) 
v_lshlrev_b32 v6, 16, v6 // (op 4214) 
v_lshlrev_b32 v7, 16, v7 // (op 4214) 
v_mul_f32 v20, v10, v6 // (op 4214) 
v_mul_f32 v21, v10, v7 // (op 4214) 


v_add_f32 v6, v18, v14 // (op 4215) 
v_add_f32 v7, v19, v15 // (op 4215) 
v_add_f32 v8, v20, v16 // (op 4215) 
v_add_f32 v9, v21, v17 // (op 4215) 


v_lshrrev_b32 v14, 16, v6 // (op 4216) 
v_lshrrev_b32 v15, 16, v7 // (op 4216) 
v_lshrrev_b32 v16, 16, v8 // (op 4216) 
v_lshrrev_b32 v17, 16, v9 // (op 4216) 


v_pack_b32_f16 v8, v14, v15 // (op 4209) 
v_pack_b32_f16 v9, v16, v17 // (op 4209) 

s_lshl_b32 s0, s4, 6 // (op 4209) 
s_add_u32 s0, s0, 384 // (op 4209) 
v_add_u32 v0, v13, s0 // (op 4209) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 4209) Store value





v_add_u32 v0, 320, v12 // (op 4167) 

buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 4167) Load value

v_accvgpr_read v4, a40 // (op 4158) 
v_accvgpr_read v5, a41 // (op 4158) 
v_accvgpr_read v6, a42 // (op 4158) 
v_accvgpr_read v7, a43 // (op 4158) 
v_mul_f32 v14, v11, v4 // (op 4158) 
v_mul_f32 v15, v11, v5 // (op 4158) 
v_mul_f32 v16, v11, v6 // (op 4158) 
v_mul_f32 v17, v11, v7 // (op 4158) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4166) 
v_lshrrev_b32 v7, 16, v8 // (op 4166) 
v_lshlrev_b32 v6, 16, v6 // (op 4166) 
v_lshlrev_b32 v7, 16, v7 // (op 4166) 
v_mul_f32 v18, v10, v6 // (op 4166) 
v_mul_f32 v19, v10, v7 // (op 4166) 
v_and_b32 v6, 65535, v9 // (op 4166) 
v_lshrrev_b32 v7, 16, v9 // (op 4166) 
v_lshlrev_b32 v6, 16, v6 // (op 4166) 
v_lshlrev_b32 v7, 16, v7 // (op 4166) 
v_mul_f32 v20, v10, v6 // (op 4166) 
v_mul_f32 v21, v10, v7 // (op 4166) 


v_add_f32 v6, v18, v14 // (op 4159) 
v_add_f32 v7, v19, v15 // (op 4159) 
v_add_f32 v8, v20, v16 // (op 4159) 
v_add_f32 v9, v21, v17 // (op 4159) 


v_lshrrev_b32 v14, 16, v6 // (op 4160) 
v_lshrrev_b32 v15, 16, v7 // (op 4160) 
v_lshrrev_b32 v16, 16, v8 // (op 4160) 
v_lshrrev_b32 v17, 16, v9 // (op 4160) 


v_pack_b32_f16 v8, v14, v15 // (op 4162) 
v_pack_b32_f16 v9, v16, v17 // (op 4162) 

v_add_u32 v0, 320, v13 // (op 4162) 

buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 4162) Store value





s_lshl_b32 s0, s2, 6 // (op 4172) 
s_add_u32 s0, s0, 320 // (op 4172) 
v_add_u32 v0, v12, s0 // (op 4172) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 4172) Load value

v_accvgpr_read v4, a44 // (op 4174) 
v_accvgpr_read v5, a45 // (op 4174) 
v_accvgpr_read v6, a46 // (op 4174) 
v_accvgpr_read v7, a47 // (op 4174) 
v_mul_f32 v14, v11, v4 // (op 4174) 
v_mul_f32 v15, v11, v5 // (op 4174) 
v_mul_f32 v16, v11, v6 // (op 4174) 
v_mul_f32 v17, v11, v7 // (op 4174) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4169) 
v_lshrrev_b32 v7, 16, v8 // (op 4169) 
v_lshlrev_b32 v6, 16, v6 // (op 4169) 
v_lshlrev_b32 v7, 16, v7 // (op 4169) 
v_mul_f32 v18, v10, v6 // (op 4169) 
v_mul_f32 v19, v10, v7 // (op 4169) 
v_and_b32 v6, 65535, v9 // (op 4169) 
v_lshrrev_b32 v7, 16, v9 // (op 4169) 
v_lshlrev_b32 v6, 16, v6 // (op 4169) 
v_lshlrev_b32 v7, 16, v7 // (op 4169) 
v_mul_f32 v20, v10, v6 // (op 4169) 
v_mul_f32 v21, v10, v7 // (op 4169) 


v_add_f32 v6, v18, v14 // (op 4170) 
v_add_f32 v7, v19, v15 // (op 4170) 
v_add_f32 v8, v20, v16 // (op 4170) 
v_add_f32 v9, v21, v17 // (op 4170) 


v_lshrrev_b32 v14, 16, v6 // (op 4171) 
v_lshrrev_b32 v15, 16, v7 // (op 4171) 
v_lshrrev_b32 v16, 16, v8 // (op 4171) 
v_lshrrev_b32 v17, 16, v9 // (op 4171) 


v_pack_b32_f16 v8, v14, v15 // (op 4164) 
v_pack_b32_f16 v9, v16, v17 // (op 4164) 

s_lshl_b32 s0, s4, 6 // (op 4164) 
s_add_u32 s0, s0, 320 // (op 4164) 
v_add_u32 v0, v13, s0 // (op 4164) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 4164) Store value





v_add_u32 v0, 256, v12 // (op 4122) 

buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 4122) Load value

v_accvgpr_read v4, a32 // (op 4113) 
v_accvgpr_read v5, a33 // (op 4113) 
v_accvgpr_read v6, a34 // (op 4113) 
v_accvgpr_read v7, a35 // (op 4113) 
v_mul_f32 v14, v11, v4 // (op 4113) 
v_mul_f32 v15, v11, v5 // (op 4113) 
v_mul_f32 v16, v11, v6 // (op 4113) 
v_mul_f32 v17, v11, v7 // (op 4113) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4121) 
v_lshrrev_b32 v7, 16, v8 // (op 4121) 
v_lshlrev_b32 v6, 16, v6 // (op 4121) 
v_lshlrev_b32 v7, 16, v7 // (op 4121) 
v_mul_f32 v18, v10, v6 // (op 4121) 
v_mul_f32 v19, v10, v7 // (op 4121) 
v_and_b32 v6, 65535, v9 // (op 4121) 
v_lshrrev_b32 v7, 16, v9 // (op 4121) 
v_lshlrev_b32 v6, 16, v6 // (op 4121) 
v_lshlrev_b32 v7, 16, v7 // (op 4121) 
v_mul_f32 v20, v10, v6 // (op 4121) 
v_mul_f32 v21, v10, v7 // (op 4121) 


v_add_f32 v6, v18, v14 // (op 4114) 
v_add_f32 v7, v19, v15 // (op 4114) 
v_add_f32 v8, v20, v16 // (op 4114) 
v_add_f32 v9, v21, v17 // (op 4114) 


v_lshrrev_b32 v14, 16, v6 // (op 4115) 
v_lshrrev_b32 v15, 16, v7 // (op 4115) 
v_lshrrev_b32 v16, 16, v8 // (op 4115) 
v_lshrrev_b32 v17, 16, v9 // (op 4115) 


v_pack_b32_f16 v8, v14, v15 // (op 4117) 
v_pack_b32_f16 v9, v16, v17 // (op 4117) 

v_add_u32 v0, 256, v13 // (op 4117) 

buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 4117) Store value





s_lshl_b32 s0, s2, 6 // (op 4127) 
s_add_u32 s0, s0, 256 // (op 4127) 
v_add_u32 v0, v12, s0 // (op 4127) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 4127) Load value

v_accvgpr_read v4, a36 // (op 4129) 
v_accvgpr_read v5, a37 // (op 4129) 
v_accvgpr_read v6, a38 // (op 4129) 
v_accvgpr_read v7, a39 // (op 4129) 
v_mul_f32 v14, v11, v4 // (op 4129) 
v_mul_f32 v15, v11, v5 // (op 4129) 
v_mul_f32 v16, v11, v6 // (op 4129) 
v_mul_f32 v17, v11, v7 // (op 4129) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4124) 
v_lshrrev_b32 v7, 16, v8 // (op 4124) 
v_lshlrev_b32 v6, 16, v6 // (op 4124) 
v_lshlrev_b32 v7, 16, v7 // (op 4124) 
v_mul_f32 v18, v10, v6 // (op 4124) 
v_mul_f32 v19, v10, v7 // (op 4124) 
v_and_b32 v6, 65535, v9 // (op 4124) 
v_lshrrev_b32 v7, 16, v9 // (op 4124) 
v_lshlrev_b32 v6, 16, v6 // (op 4124) 
v_lshlrev_b32 v7, 16, v7 // (op 4124) 
v_mul_f32 v20, v10, v6 // (op 4124) 
v_mul_f32 v21, v10, v7 // (op 4124) 


v_add_f32 v6, v18, v14 // (op 4125) 
v_add_f32 v7, v19, v15 // (op 4125) 
v_add_f32 v8, v20, v16 // (op 4125) 
v_add_f32 v9, v21, v17 // (op 4125) 


v_lshrrev_b32 v14, 16, v6 // (op 4126) 
v_lshrrev_b32 v15, 16, v7 // (op 4126) 
v_lshrrev_b32 v16, 16, v8 // (op 4126) 
v_lshrrev_b32 v17, 16, v9 // (op 4126) 


v_pack_b32_f16 v8, v14, v15 // (op 4119) 
v_pack_b32_f16 v9, v16, v17 // (op 4119) 

s_lshl_b32 s0, s4, 6 // (op 4119) 
s_add_u32 s0, s0, 256 // (op 4119) 
v_add_u32 v0, v13, s0 // (op 4119) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 4119) Store value





v_add_u32 v0, 192, v12 // (op 4077) 

buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 4077) Load value

v_accvgpr_read v4, a24 // (op 4068) 
v_accvgpr_read v5, a25 // (op 4068) 
v_accvgpr_read v6, a26 // (op 4068) 
v_accvgpr_read v7, a27 // (op 4068) 
v_mul_f32 v14, v11, v4 // (op 4068) 
v_mul_f32 v15, v11, v5 // (op 4068) 
v_mul_f32 v16, v11, v6 // (op 4068) 
v_mul_f32 v17, v11, v7 // (op 4068) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4076) 
v_lshrrev_b32 v7, 16, v8 // (op 4076) 
v_lshlrev_b32 v6, 16, v6 // (op 4076) 
v_lshlrev_b32 v7, 16, v7 // (op 4076) 
v_mul_f32 v18, v10, v6 // (op 4076) 
v_mul_f32 v19, v10, v7 // (op 4076) 
v_and_b32 v6, 65535, v9 // (op 4076) 
v_lshrrev_b32 v7, 16, v9 // (op 4076) 
v_lshlrev_b32 v6, 16, v6 // (op 4076) 
v_lshlrev_b32 v7, 16, v7 // (op 4076) 
v_mul_f32 v20, v10, v6 // (op 4076) 
v_mul_f32 v21, v10, v7 // (op 4076) 


v_add_f32 v6, v18, v14 // (op 4069) 
v_add_f32 v7, v19, v15 // (op 4069) 
v_add_f32 v8, v20, v16 // (op 4069) 
v_add_f32 v9, v21, v17 // (op 4069) 


v_lshrrev_b32 v14, 16, v6 // (op 4070) 
v_lshrrev_b32 v15, 16, v7 // (op 4070) 
v_lshrrev_b32 v16, 16, v8 // (op 4070) 
v_lshrrev_b32 v17, 16, v9 // (op 4070) 


v_pack_b32_f16 v8, v14, v15 // (op 4072) 
v_pack_b32_f16 v9, v16, v17 // (op 4072) 

v_add_u32 v0, 192, v13 // (op 4072) 

buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 4072) Store value





s_lshl_b32 s0, s2, 6 // (op 4082) 
s_add_u32 s0, s0, 192 // (op 4082) 
v_add_u32 v0, v12, s0 // (op 4082) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 4082) Load value

v_accvgpr_read v4, a28 // (op 4084) 
v_accvgpr_read v5, a29 // (op 4084) 
v_accvgpr_read v6, a30 // (op 4084) 
v_accvgpr_read v7, a31 // (op 4084) 
v_mul_f32 v14, v11, v4 // (op 4084) 
v_mul_f32 v15, v11, v5 // (op 4084) 
v_mul_f32 v16, v11, v6 // (op 4084) 
v_mul_f32 v17, v11, v7 // (op 4084) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4079) 
v_lshrrev_b32 v7, 16, v8 // (op 4079) 
v_lshlrev_b32 v6, 16, v6 // (op 4079) 
v_lshlrev_b32 v7, 16, v7 // (op 4079) 
v_mul_f32 v18, v10, v6 // (op 4079) 
v_mul_f32 v19, v10, v7 // (op 4079) 
v_and_b32 v6, 65535, v9 // (op 4079) 
v_lshrrev_b32 v7, 16, v9 // (op 4079) 
v_lshlrev_b32 v6, 16, v6 // (op 4079) 
v_lshlrev_b32 v7, 16, v7 // (op 4079) 
v_mul_f32 v20, v10, v6 // (op 4079) 
v_mul_f32 v21, v10, v7 // (op 4079) 


v_add_f32 v6, v18, v14 // (op 4080) 
v_add_f32 v7, v19, v15 // (op 4080) 
v_add_f32 v8, v20, v16 // (op 4080) 
v_add_f32 v9, v21, v17 // (op 4080) 


v_lshrrev_b32 v14, 16, v6 // (op 4081) 
v_lshrrev_b32 v15, 16, v7 // (op 4081) 
v_lshrrev_b32 v16, 16, v8 // (op 4081) 
v_lshrrev_b32 v17, 16, v9 // (op 4081) 


v_pack_b32_f16 v8, v14, v15 // (op 4074) 
v_pack_b32_f16 v9, v16, v17 // (op 4074) 

s_lshl_b32 s0, s4, 6 // (op 4074) 
s_add_u32 s0, s0, 192 // (op 4074) 
v_add_u32 v0, v13, s0 // (op 4074) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 4074) Store value





v_add_u32 v0, 128, v12 // (op 4032) 

buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 4032) Load value

v_accvgpr_read v4, a16 // (op 4023) 
v_accvgpr_read v5, a17 // (op 4023) 
v_accvgpr_read v6, a18 // (op 4023) 
v_accvgpr_read v7, a19 // (op 4023) 
v_mul_f32 v14, v11, v4 // (op 4023) 
v_mul_f32 v15, v11, v5 // (op 4023) 
v_mul_f32 v16, v11, v6 // (op 4023) 
v_mul_f32 v17, v11, v7 // (op 4023) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4031) 
v_lshrrev_b32 v7, 16, v8 // (op 4031) 
v_lshlrev_b32 v6, 16, v6 // (op 4031) 
v_lshlrev_b32 v7, 16, v7 // (op 4031) 
v_mul_f32 v18, v10, v6 // (op 4031) 
v_mul_f32 v19, v10, v7 // (op 4031) 
v_and_b32 v6, 65535, v9 // (op 4031) 
v_lshrrev_b32 v7, 16, v9 // (op 4031) 
v_lshlrev_b32 v6, 16, v6 // (op 4031) 
v_lshlrev_b32 v7, 16, v7 // (op 4031) 
v_mul_f32 v20, v10, v6 // (op 4031) 
v_mul_f32 v21, v10, v7 // (op 4031) 


v_add_f32 v6, v18, v14 // (op 4024) 
v_add_f32 v7, v19, v15 // (op 4024) 
v_add_f32 v8, v20, v16 // (op 4024) 
v_add_f32 v9, v21, v17 // (op 4024) 


v_lshrrev_b32 v14, 16, v6 // (op 4025) 
v_lshrrev_b32 v15, 16, v7 // (op 4025) 
v_lshrrev_b32 v16, 16, v8 // (op 4025) 
v_lshrrev_b32 v17, 16, v9 // (op 4025) 


v_pack_b32_f16 v8, v14, v15 // (op 4027) 
v_pack_b32_f16 v9, v16, v17 // (op 4027) 

v_add_u32 v0, 128, v13 // (op 4027) 

buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 4027) Store value





s_lshl_b32 s0, s2, 6 // (op 4037) 
s_add_u32 s0, s0, 128 // (op 4037) 
v_add_u32 v0, v12, s0 // (op 4037) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 4037) Load value

v_accvgpr_read v4, a20 // (op 4039) 
v_accvgpr_read v5, a21 // (op 4039) 
v_accvgpr_read v6, a22 // (op 4039) 
v_accvgpr_read v7, a23 // (op 4039) 
v_mul_f32 v14, v11, v4 // (op 4039) 
v_mul_f32 v15, v11, v5 // (op 4039) 
v_mul_f32 v16, v11, v6 // (op 4039) 
v_mul_f32 v17, v11, v7 // (op 4039) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4034) 
v_lshrrev_b32 v7, 16, v8 // (op 4034) 
v_lshlrev_b32 v6, 16, v6 // (op 4034) 
v_lshlrev_b32 v7, 16, v7 // (op 4034) 
v_mul_f32 v18, v10, v6 // (op 4034) 
v_mul_f32 v19, v10, v7 // (op 4034) 
v_and_b32 v6, 65535, v9 // (op 4034) 
v_lshrrev_b32 v7, 16, v9 // (op 4034) 
v_lshlrev_b32 v6, 16, v6 // (op 4034) 
v_lshlrev_b32 v7, 16, v7 // (op 4034) 
v_mul_f32 v20, v10, v6 // (op 4034) 
v_mul_f32 v21, v10, v7 // (op 4034) 


v_add_f32 v6, v18, v14 // (op 4035) 
v_add_f32 v7, v19, v15 // (op 4035) 
v_add_f32 v8, v20, v16 // (op 4035) 
v_add_f32 v9, v21, v17 // (op 4035) 


v_lshrrev_b32 v14, 16, v6 // (op 4036) 
v_lshrrev_b32 v15, 16, v7 // (op 4036) 
v_lshrrev_b32 v16, 16, v8 // (op 4036) 
v_lshrrev_b32 v17, 16, v9 // (op 4036) 


v_pack_b32_f16 v8, v14, v15 // (op 4029) 
v_pack_b32_f16 v9, v16, v17 // (op 4029) 

s_lshl_b32 s0, s4, 6 // (op 4029) 
s_add_u32 s0, s0, 128 // (op 4029) 
v_add_u32 v0, v13, s0 // (op 4029) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 4029) Store value





v_add_u32 v0, 64, v12 // (op 3987) 

buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 3987) Load value

v_accvgpr_read v4, a8 // (op 3978) 
v_accvgpr_read v5, a9 // (op 3978) 
v_accvgpr_read v6, a10 // (op 3978) 
v_accvgpr_read v7, a11 // (op 3978) 
v_mul_f32 v14, v11, v4 // (op 3978) 
v_mul_f32 v15, v11, v5 // (op 3978) 
v_mul_f32 v16, v11, v6 // (op 3978) 
v_mul_f32 v17, v11, v7 // (op 3978) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 3986) 
v_lshrrev_b32 v7, 16, v8 // (op 3986) 
v_lshlrev_b32 v6, 16, v6 // (op 3986) 
v_lshlrev_b32 v7, 16, v7 // (op 3986) 
v_mul_f32 v18, v10, v6 // (op 3986) 
v_mul_f32 v19, v10, v7 // (op 3986) 
v_and_b32 v6, 65535, v9 // (op 3986) 
v_lshrrev_b32 v7, 16, v9 // (op 3986) 
v_lshlrev_b32 v6, 16, v6 // (op 3986) 
v_lshlrev_b32 v7, 16, v7 // (op 3986) 
v_mul_f32 v20, v10, v6 // (op 3986) 
v_mul_f32 v21, v10, v7 // (op 3986) 


v_add_f32 v6, v18, v14 // (op 3979) 
v_add_f32 v7, v19, v15 // (op 3979) 
v_add_f32 v8, v20, v16 // (op 3979) 
v_add_f32 v9, v21, v17 // (op 3979) 


v_lshrrev_b32 v14, 16, v6 // (op 3980) 
v_lshrrev_b32 v15, 16, v7 // (op 3980) 
v_lshrrev_b32 v16, 16, v8 // (op 3980) 
v_lshrrev_b32 v17, 16, v9 // (op 3980) 


v_pack_b32_f16 v8, v14, v15 // (op 3982) 
v_pack_b32_f16 v9, v16, v17 // (op 3982) 

v_add_u32 v0, 64, v13 // (op 3982) 

buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 3982) Store value





s_lshl_b32 s0, s2, 6 // (op 3992) 
s_add_u32 s0, s0, 64 // (op 3992) 
v_add_u32 v0, v12, s0 // (op 3992) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 3992) Load value

v_accvgpr_read v4, a12 // (op 3994) 
v_accvgpr_read v5, a13 // (op 3994) 
v_accvgpr_read v6, a14 // (op 3994) 
v_accvgpr_read v7, a15 // (op 3994) 
v_mul_f32 v14, v11, v4 // (op 3994) 
v_mul_f32 v15, v11, v5 // (op 3994) 
v_mul_f32 v16, v11, v6 // (op 3994) 
v_mul_f32 v17, v11, v7 // (op 3994) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 3989) 
v_lshrrev_b32 v7, 16, v8 // (op 3989) 
v_lshlrev_b32 v6, 16, v6 // (op 3989) 
v_lshlrev_b32 v7, 16, v7 // (op 3989) 
v_mul_f32 v18, v10, v6 // (op 3989) 
v_mul_f32 v19, v10, v7 // (op 3989) 
v_and_b32 v6, 65535, v9 // (op 3989) 
v_lshrrev_b32 v7, 16, v9 // (op 3989) 
v_lshlrev_b32 v6, 16, v6 // (op 3989) 
v_lshlrev_b32 v7, 16, v7 // (op 3989) 
v_mul_f32 v20, v10, v6 // (op 3989) 
v_mul_f32 v21, v10, v7 // (op 3989) 


v_add_f32 v6, v18, v14 // (op 3990) 
v_add_f32 v7, v19, v15 // (op 3990) 
v_add_f32 v8, v20, v16 // (op 3990) 
v_add_f32 v9, v21, v17 // (op 3990) 


v_lshrrev_b32 v14, 16, v6 // (op 3991) 
v_lshrrev_b32 v15, 16, v7 // (op 3991) 
v_lshrrev_b32 v16, 16, v8 // (op 3991) 
v_lshrrev_b32 v17, 16, v9 // (op 3991) 


v_pack_b32_f16 v8, v14, v15 // (op 3984) 
v_pack_b32_f16 v9, v16, v17 // (op 3984) 

s_lshl_b32 s0, s4, 6 // (op 3984) 
s_add_u32 s0, s0, 64 // (op 3984) 
v_add_u32 v0, v13, s0 // (op 3984) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 3984) Store value





v_mov_b32 v0, v12 // (op 148) reg expression

buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 148) Load value

v_accvgpr_read v4, a0 // (op 144) 
v_accvgpr_read v5, a1 // (op 144) 
v_accvgpr_read v6, a2 // (op 144) 
v_accvgpr_read v7, a3 // (op 144) 
v_mul_f32 v14, v11, v4 // (op 144) 
v_mul_f32 v15, v11, v5 // (op 144) 
v_mul_f32 v16, v11, v6 // (op 144) 
v_mul_f32 v17, v11, v7 // (op 144) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 149) 
v_lshrrev_b32 v7, 16, v8 // (op 149) 
v_lshlrev_b32 v6, 16, v6 // (op 149) 
v_lshlrev_b32 v7, 16, v7 // (op 149) 
v_mul_f32 v18, v10, v6 // (op 149) 
v_mul_f32 v19, v10, v7 // (op 149) 
v_and_b32 v6, 65535, v9 // (op 149) 
v_lshrrev_b32 v7, 16, v9 // (op 149) 
v_lshlrev_b32 v6, 16, v6 // (op 149) 
v_lshlrev_b32 v7, 16, v7 // (op 149) 
v_mul_f32 v20, v10, v6 // (op 149) 
v_mul_f32 v21, v10, v7 // (op 149) 


v_add_f32 v6, v18, v14 // (op 145) 
v_add_f32 v7, v19, v15 // (op 145) 
v_add_f32 v8, v20, v16 // (op 145) 
v_add_f32 v9, v21, v17 // (op 145) 


v_lshrrev_b32 v14, 16, v6 // (op 146) 
v_lshrrev_b32 v15, 16, v7 // (op 146) 
v_lshrrev_b32 v16, 16, v8 // (op 146) 
v_lshrrev_b32 v17, 16, v9 // (op 146) 


v_pack_b32_f16 v8, v14, v15 // (op 147) 
v_pack_b32_f16 v9, v16, v17 // (op 147) 

v_mov_b32 v0, v13 // (op 147) reg expression

buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 147) Store value





s_lshl_b32 s0, s2, 6 // (op 3947) 
v_add_u32 v0, v12, s0 // (op 3947) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 3947) Load value

v_accvgpr_read v4, a4 // (op 3943) 
v_accvgpr_read v5, a5 // (op 3943) 
v_accvgpr_read v6, a6 // (op 3943) 
v_accvgpr_read v7, a7 // (op 3943) 
v_mul_f32 v14, v11, v4 // (op 3943) 
v_mul_f32 v15, v11, v5 // (op 3943) 
v_mul_f32 v16, v11, v6 // (op 3943) 
v_mul_f32 v17, v11, v7 // (op 3943) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 3948) 
v_lshrrev_b32 v7, 16, v8 // (op 3948) 
v_lshlrev_b32 v6, 16, v6 // (op 3948) 
v_lshlrev_b32 v7, 16, v7 // (op 3948) 
v_mul_f32 v18, v10, v6 // (op 3948) 
v_mul_f32 v19, v10, v7 // (op 3948) 
v_and_b32 v6, 65535, v9 // (op 3948) 
v_lshrrev_b32 v7, 16, v9 // (op 3948) 
v_lshlrev_b32 v6, 16, v6 // (op 3948) 
v_lshlrev_b32 v7, 16, v7 // (op 3948) 
v_mul_f32 v20, v10, v6 // (op 3948) 
v_mul_f32 v21, v10, v7 // (op 3948) 


v_add_f32 v6, v18, v14 // (op 3944) 
v_add_f32 v7, v19, v15 // (op 3944) 
v_add_f32 v8, v20, v16 // (op 3944) 
v_add_f32 v9, v21, v17 // (op 3944) 


v_lshrrev_b32 v14, 16, v6 // (op 3945) 
v_lshrrev_b32 v15, 16, v7 // (op 3945) 
v_lshrrev_b32 v16, 16, v8 // (op 3945) 
v_lshrrev_b32 v17, 16, v9 // (op 3945) 


v_pack_b32_f16 v8, v14, v15 // (op 3946) 
v_pack_b32_f16 v9, v16, v17 // (op 3946) 

s_lshl_b32 s0, s4, 6 // (op 3946) 
v_add_u32 v0, v13, s0 // (op 3946) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 3946) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ConditionalBottom_0_beta_135:
























s_endpgm  // End of RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2
.LRR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_end:
.size RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2, .LRR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_end-RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2
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
    DestDataType: b
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
  - .name: RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2
    .symbol: RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2.kd
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
        .expression: {type: CommandArgument, size: 8, offset: 160, name: Tensor_10_pointer, variableType: {dataType: BFloat16, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BFloat16, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 208, name: Tensor_21_pointer, variableType: {dataType: BFloat16, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BFloat16, pointerType: PointerGlobal}
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

