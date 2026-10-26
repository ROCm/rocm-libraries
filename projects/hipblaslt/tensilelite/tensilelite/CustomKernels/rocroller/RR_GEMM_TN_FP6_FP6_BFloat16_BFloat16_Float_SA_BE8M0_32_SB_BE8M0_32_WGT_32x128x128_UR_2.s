

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2
.p2align 8
.type RR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2,@function
RR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 2029) 
v_and_b32 v3, 15, v1 // (op 2029) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 2029) 


v_and_b32 v0, 63, v1 // (op 2029) 
v_lshrrev_b32 v3, 4, v0 // (op 2029) 

v_lshlrev_b32 v0, 5, v3 // (op 2029) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 2029) 


v_mul_lo_u32 v0, v3, 6 // (op 2029) 

v_lshrrev_b32 v3, 3, v0 // (op 2029) 

v_lshrrev_b32 v0, 7, v1 // (op 2187) 
v_and_b32 v4, 15, v1 // (op 2187) 
v_lshl_add_u32 v5, v0, 4, v4 // (op 2187) 


v_and_b32 v0, 63, v1 // (op 2187) 
v_lshrrev_b32 v4, 4, v0 // (op 2187) 

v_lshlrev_b32 v0, 5, v4 // (op 2187) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 2187) 


v_mul_lo_u32 v0, v4, 6 // (op 2187) 

v_lshrrev_b32 v4, 3, v0 // (op 2187) 

v_lshrrev_b32 v0, 7, v1 // (op 1972) 
v_and_b32 v5, 63, v1 // (op 1972) 
v_lshrrev_b32 v6, 4, v5 // (op 1972) 
v_lshlrev_b32 v7, 2, v6 // (op 1972) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 1972) 


v_lshl_add_u32 v0, s16, 5, v6 // (op 1972) 

v_lshrrev_b32 v6, 2, v5 // (op 1972) 

v_and_b32 v5, 3, v6 // (op 1972) 

v_and_b32 v6, 3, v1 // (op 1972) 
v_lshl_add_u32 v7, v5, 2, v6 // (op 1972) 


v_lshrrev_b32 v5, 6, v1 // (op 1972) 
v_and_b32 v6, 1, v5 // (op 1972) 

v_lshl_add_u32 v5, v6, 4, v7 // (op 1972) 


v_lshl_add_u32 v6, s17, 7, v5 // (op 1972) 

v_mul_lo_u32 v5, v6, s2 // (op 1972) 

v_add_lshl_u32 v6, v0, v5, 1 // (op 1972) 


v_lshrrev_b32 v0, 6, v1 // (op 2087) 
v_and_b32 v5, 1, v0 // (op 2087) 

v_and_b32 v0, 15, v1 // (op 2087) 
v_lshl_add_u32 v7, v5, 4, v0 // (op 2087) 


v_and_b32 v0, 63, v1 // (op 2087) 
v_lshrrev_b32 v5, 4, v0 // (op 2087) 

v_lshlrev_b32 v0, 5, v5 // (op 2087) 

v_lshl_add_u32 v5, v7, 7, v0 // (op 2087) 


v_mul_lo_u32 v0, v5, 6 // (op 2087) 

v_lshrrev_b32 v5, 3, v0 // (op 2087) 

v_lshrrev_b32 v0, 6, v1 // (op 2145) 
v_and_b32 v7, 1, v0 // (op 2145) 

v_and_b32 v0, 15, v1 // (op 2145) 
s_mul_i32 s1, s2, s9 // (op 1974) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 1974) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 1974) least significant
s_mul_hi_u32 s19, s2, s8 // (op 1974) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1974) most significant: sum
s_add_u32 s1, s1, s19 // (op 1974) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 1974) 

v_lshl_add_u32 v8, v7, 4, v0 // (op 2145) 


v_and_b32 v0, 63, v1 // (op 2145) 
v_lshrrev_b32 v7, 4, v0 // (op 2145) 

v_lshl_add_u32 v0, v8, 2, v7 // (op 2145) 


v_lshrrev_b32 v7, 7, v1 // (op 2161) 
v_and_b32 v8, 63, v1 // (op 2161) 
v_lshrrev_b32 v9, 4, v8 // (op 2161) 
v_lshlrev_b32 v10, 2, v9 // (op 2161) 

v_lshl_add_u32 v9, v7, 4, v10 // (op 2161) 


v_lshl_add_u32 v7, s16, 5, v9 // (op 2161) 

v_lshrrev_b32 v9, 2, v8 // (op 2161) 

v_and_b32 v8, 3, v9 // (op 2161) 

v_and_b32 v9, 3, v1 // (op 2161) 
v_lshl_add_u32 v10, v8, 2, v9 // (op 2161) 


v_lshrrev_b32 v8, 6, v1 // (op 2161) 
v_and_b32 v9, 1, v8 // (op 2161) 

v_lshl_add_u32 v8, v9, 4, v10 // (op 2161) 


v_lshl_add_u32 v9, s17, 7, v8 // (op 2161) 

v_mul_lo_u32 v8, v9, s4 // (op 2161) 

v_add_lshl_u32 v9, v7, v8, 1 // (op 2161) 


v_lshrrev_b32 v7, 6, v1 // (op 2216) 
v_and_b32 v8, 1, v7 // (op 2216) 

v_and_b32 v7, 15, v1 // (op 2216) 
v_lshl_add_u32 v10, v8, 4, v7 // (op 2216) 


v_and_b32 v7, 63, v1 // (op 2216) 
v_lshrrev_b32 v8, 4, v7 // (op 2216) 

v_lshlrev_b32 v7, 5, v8 // (op 2216) 

v_lshl_add_u32 v8, v10, 7, v7 // (op 2216) 


v_mul_lo_u32 v7, v8, 6 // (op 2216) 

v_lshrrev_b32 v8, 3, v7 // (op 2216) 

v_lshrrev_b32 v7, 6, v1 // (op 2245) 
v_and_b32 v10, 1, v7 // (op 2245) 

v_and_b32 v7, 15, v1 // (op 2245) 
s_mul_i32 s1, s4, s9 // (op 2163) most significant: low of low * high
v_lshl_add_u32 v11, v10, 4, v7 // (op 2245) 


v_and_b32 v7, 63, v1 // (op 2245) 
v_lshrrev_b32 v10, 4, v7 // (op 2245) 

v_lshl_add_u32 v7, v11, 2, v10 // (op 2245) 


s_mul_i32 s38, s5, s8 // (op 2163) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 2163) least significant
s_mul_hi_u32 s39, s4, s8 // (op 2163) most significant: high of low * low
s_add_u32 s1, s1, s38 // (op 2163) most significant: sum
s_add_u32 s1, s1, s39 // (op 2163) most significant: sum


s_lshl_b64 s[38:39], s[0:1], 1 // (op 2163) 

s_mov_b32 s40, s6 // (op 1974) reg expression
s_mov_b32 s41, s7 // (op 1974) reg expression
s_mov_b32 s42, s18 // (op 1974) reg expression
s_mov_b32 s43, 131072 // (op 1974) call()

s_mov_b32 s44, s10 // (op 2163) reg expression

s_mov_b32 s45, s11 // (op 2163) reg expression
s_mov_b32 s46, s38 // (op 2163) reg expression
s_mov_b32 s47, 131072 // (op 2163) call()


v_mov_b32 v10, s14 // (op 2) Move value
v_mov_b32 v11, s15 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 61) call()
v_accvgpr_write a1, 0.00000 // (op 61) call()
v_accvgpr_write a2, 0.00000 // (op 61) call()
v_accvgpr_write a3, 0.00000 // (op 61) call()
v_accvgpr_write a4, 0.00000 // (op 310) call()
v_accvgpr_write a5, 0.00000 // (op 310) call()
v_accvgpr_write a6, 0.00000 // (op 310) call()
v_accvgpr_write a7, 0.00000 // (op 310) call()
v_accvgpr_write a8, 0.00000 // (op 466) call()
v_accvgpr_write a9, 0.00000 // (op 466) call()
v_accvgpr_write a10, 0.00000 // (op 466) call()
v_accvgpr_write a11, 0.00000 // (op 466) call()
v_accvgpr_write a12, 0.00000 // (op 622) call()
v_accvgpr_write a13, 0.00000 // (op 622) call()
v_accvgpr_write a14, 0.00000 // (op 622) call()
v_accvgpr_write a15, 0.00000 // (op 622) call()
v_lshrrev_b32 v12, 3, v1 // (op 1739) 
v_lshl_add_u32 v13, s16, 5, v12 // (op 1739) 

v_and_b32 v12, -8, v1 // (op 1739) 
v_sub_u32 v14, v1, v12 // (op 1739) 

v_lshlrev_b32 v12, 4, v14 // (op 1739) 

v_mov_b32 v15, 0 // (op 1739) convert
v_mov_b32 v14, v12 // (op 1739) convert

v_mul_lo_u32 v17, v13, s13 // (op 1739) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 1739) most significant: high of low * low
v_mul_lo_u32 v16, v13, s12 // (op 1739) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 1739) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 1739) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 1739) most significant half



v_mul_lo_u32 v13, v17, 6 // (op 1739) most significant: low of high * low
v_mul_hi_u32 v14, v16, 6 // (op 1739) most significant: high of low * low
v_mul_lo_u32 v12, v16, 6 // (op 1739) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 1739) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 1739) 

v_mov_b32 v12, v14 // (op 1739) convert


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s23, 6 // (op 1741) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 1741) least significant
s_mul_hi_u32 s6, s22, 6 // (op 1741) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1741) most significant: sum

s_mul_i32 s7, s12, s1 // (op 1741) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 1741) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 1741) least significant
s_mul_hi_u32 s11, s12, s0 // (op 1741) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 1741) most significant: sum
s_add_u32 s7, s7, s11 // (op 1741) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 1741) 

s_mov_b32 s48, s20 // (op 1741) reg expression
s_mov_b32 s49, s21 // (op 1741) reg expression
s_mov_b32 s50, s0 // (op 1741) reg expression
s_mov_b32 s51, 131072 // (op 1741) call()

v_lshrrev_b32 v13, 3, v1 // (op 1758) 
v_lshl_add_u32 v14, s16, 5, v13 // (op 1758) 

v_and_b32 v13, -8, v1 // (op 1758) 
v_sub_u32 v15, v1, v13 // (op 1758) 

v_lshlrev_b32 v13, 4, v15 // (op 1758) 

v_mov_b32 v17, 0 // (op 1758) convert
v_mov_b32 v16, v13 // (op 1758) convert

v_mul_lo_u32 v19, v14, s13 // (op 1758) most significant: low of low * high
v_mul_hi_u32 v13, v14, s12 // (op 1758) most significant: high of low * low
v_mul_lo_u32 v18, v14, s12 // (op 1758) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 1758) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 1758) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 1758) most significant half



v_mul_lo_u32 v17, v19, 6 // (op 1758) most significant: low of high * low
v_mul_hi_u32 v13, v18, 6 // (op 1758) most significant: high of low * low
v_mul_lo_u32 v16, v18, 6 // (op 1758) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 1758) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 1758) 

v_mov_b32 v13, v14 // (op 1758) convert

s_mul_i32 s1, s23, 6 // (op 1760) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 1760) least significant
s_mul_hi_u32 s6, s22, 6 // (op 1760) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1760) most significant: sum

s_mul_i32 s7, s12, s1 // (op 1760) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 1760) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 1760) least significant
s_mul_hi_u32 s11, s12, s0 // (op 1760) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 1760) most significant: sum
s_add_u32 s7, s7, s11 // (op 1760) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 1760) 

s_mov_b32 s48, s20 // (op 1760) reg expression
s_mov_b32 s49, s21 // (op 1760) reg expression
s_mov_b32 s50, s0 // (op 1760) reg expression
s_mov_b32 s51, 131072 // (op 1760) call()

v_and_b32 v14, -2, v1 // (op 1803) 
v_sub_u32 v15, v1, v14 // (op 1803) 

v_lshlrev_b32 v14, 6, v15 // (op 1803) 

v_mov_b32 v17, 0 // (op 1803) convert
v_mov_b32 v16, v14 // (op 1803) convert

v_lshrrev_b32 v14, 1, v1 // (op 1803) 
v_lshl_add_u32 v15, s17, 7, v14 // (op 1803) 

v_mul_lo_u32 v19, v15, s25 // (op 1803) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 1803) most significant: high of low * low
v_mul_lo_u32 v18, v15, s24 // (op 1803) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 1803) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v18 // (op 1803) least significant half
v_addc_co_u32 v15, s[0:1], v17, v19, s[0:1] // (op 1803) most significant half



v_mul_lo_u32 v17, v15, 6 // (op 1803) most significant: low of high * low
v_mul_hi_u32 v18, v14, 6 // (op 1803) most significant: high of low * low
v_mul_lo_u32 v16, v14, 6 // (op 1803) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 1803) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 1803) 

v_mov_b32 v16, v14 // (op 1803) convert

s_mul_i32 s1, s9, 6 // (op 1805) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 1805) least significant
s_mul_hi_u32 s6, s8, 6 // (op 1805) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1805) most significant: sum

s_mul_i32 s7, s24, s1 // (op 1805) most significant: low of low * high
s_mul_i32 s10, s25, s0 // (op 1805) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 1805) least significant
s_mul_hi_u32 s11, s24, s0 // (op 1805) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 1805) most significant: sum
s_add_u32 s7, s7, s11 // (op 1805) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 1805) 

s_mov_b32 s52, s26 // (op 1805) reg expression
s_mov_b32 s53, s27 // (op 1805) reg expression
s_mov_b32 s54, s0 // (op 1805) reg expression
s_mov_b32 s55, 131072 // (op 1805) call()

v_and_b32 v14, -2, v1 // (op 1822) 
v_sub_u32 v15, v1, v14 // (op 1822) 

v_lshlrev_b32 v14, 6, v15 // (op 1822) 

v_mov_b32 v19, 0 // (op 1822) convert
v_mov_b32 v18, v14 // (op 1822) convert

v_lshrrev_b32 v17, 1, v1 // (op 1822) 
v_lshl_add_u32 v14, s17, 7, v17 // (op 1822) 

v_mul_lo_u32 v21, v14, s25 // (op 1822) most significant: low of low * high
v_mul_hi_u32 v15, v14, s24 // (op 1822) most significant: high of low * low
v_mul_lo_u32 v20, v14, s24 // (op 1822) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 1822) most significant: sum


v_add_co_u32 v14, s[0:1], v18, v20 // (op 1822) least significant half
v_addc_co_u32 v15, s[0:1], v19, v21, s[0:1] // (op 1822) most significant half



v_mul_lo_u32 v19, v15, 6 // (op 1822) most significant: low of high * low
v_mul_hi_u32 v17, v14, 6 // (op 1822) most significant: high of low * low
v_mul_lo_u32 v18, v14, 6 // (op 1822) least significant: low of low * low
v_add_u32 v19, v19, v17 // (op 1822) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[18:19] // (op 1822) 

v_mov_b32 v17, v14 // (op 1822) convert

s_mul_i32 s1, s9, 6 // (op 1824) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 1824) least significant
s_mul_hi_u32 s6, s8, 6 // (op 1824) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1824) most significant: sum

s_mul_i32 s7, s24, s1 // (op 1824) most significant: low of low * high
s_mul_i32 s10, s25, s0 // (op 1824) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 1824) least significant
s_mul_hi_u32 s11, s24, s0 // (op 1824) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 1824) most significant: sum
s_add_u32 s7, s7, s11 // (op 1824) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 1824) 

s_mov_b32 s52, s26 // (op 1824) reg expression
s_mov_b32 s53, s27 // (op 1824) reg expression
s_mov_b32 s54, s0 // (op 1824) reg expression
s_mov_b32 s55, 131072 // (op 1824) call()

v_lshrrev_b32 v14, 7, v1 // (op 1867) 
v_and_b32 v15, 15, v1 // (op 1867) 
v_lshl_add_u32 v18, v14, 4, v15 // (op 1867) 


v_lshl_add_u32 v14, s16, 5, v18 // (op 1867) 

v_and_b32 v15, 63, v1 // (op 1867) 
v_lshrrev_b32 v18, 4, v15 // (op 1867) 

v_mul_lo_u32 v15, v14, s28 // (op 1867) 
v_add_u32 v15, v15, v18 // (op 1867) 


s_mul_i32 s1, s28, s23 // (op 1869) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 1869) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 1869) least significant
s_mul_hi_u32 s7, s28, s22 // (op 1869) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1869) most significant: sum
s_add_u32 s1, s1, s7 // (op 1869) most significant: sum


s_mov_b32 s56, s30 // (op 1869) reg expression
s_mov_b32 s57, s31 // (op 1869) reg expression
s_mov_b32 s58, s0 // (op 1869) reg expression
s_mov_b32 s59, 131072 // (op 1869) call()

v_lshrrev_b32 v14, 7, v1 // (op 1886) 
v_and_b32 v18, 15, v1 // (op 1886) 
v_lshl_add_u32 v19, v14, 4, v18 // (op 1886) 


v_lshl_add_u32 v14, s16, 5, v19 // (op 1886) 

v_and_b32 v18, 63, v1 // (op 1886) 
v_lshrrev_b32 v19, 4, v18 // (op 1886) 

v_mul_lo_u32 v18, v14, s28 // (op 1886) 
v_add_u32 v18, v18, v19 // (op 1886) 


s_mul_i32 s1, s28, s23 // (op 1888) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 1888) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 1888) least significant
s_mul_hi_u32 s7, s28, s22 // (op 1888) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1888) most significant: sum
s_add_u32 s1, s1, s7 // (op 1888) most significant: sum


s_mov_b32 s56, s30 // (op 1888) reg expression
s_mov_b32 s57, s31 // (op 1888) reg expression
s_mov_b32 s58, s0 // (op 1888) reg expression
s_mov_b32 s59, 131072 // (op 1888) call()

v_and_b32 v14, -4, v1 // (op 1931) 
v_sub_u32 v19, v1, v14 // (op 1931) 

v_lshrrev_b32 v14, 2, v1 // (op 1931) 
v_lshlrev_b32 v20, 1, v14 // (op 1931) 

v_lshl_add_u32 v14, s17, 7, v20 // (op 1931) 

v_mul_lo_u32 v20, v14, s32 // (op 1931) 

v_add_u32 v14, v19, v20 // (op 1931) 


s_mul_i32 s1, s32, s9 // (op 1933) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 1933) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 1933) least significant
s_mul_hi_u32 s7, s32, s8 // (op 1933) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1933) most significant: sum
s_add_u32 s1, s1, s7 // (op 1933) most significant: sum


s_mov_b32 s60, s34 // (op 1933) reg expression
s_mov_b32 s61, s35 // (op 1933) reg expression
s_mov_b32 s62, s0 // (op 1933) reg expression
s_mov_b32 s63, 131072 // (op 1933) call()

v_and_b32 v19, -4, v1 // (op 1950) 
v_sub_u32 v20, v1, v19 // (op 1950) 

v_lshrrev_b32 v19, 2, v1 // (op 1950) 
v_lshlrev_b32 v21, 1, v19 // (op 1950) 

v_lshl_add_u32 v19, s17, 7, v21 // (op 1950) 

v_mul_lo_u32 v21, v19, s32 // (op 1950) 

v_add_u32 v19, v20, v21 // (op 1950) 


s_mul_i32 s1, s32, s9 // (op 1952) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 1952) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 1952) least significant
s_mul_hi_u32 s7, s32, s8 // (op 1952) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1952) most significant: sum
s_add_u32 s1, s1, s7 // (op 1952) most significant: sum


s_mov_b32 s60, s34 // (op 1952) reg expression
s_mov_b32 s61, s35 // (op 1952) reg expression
s_mov_b32 s62, s0 // (op 1952) reg expression
s_mov_b32 s63, 131072 // (op 1952) call()




buffer_load_ubyte v20, v15, s[56:59], 0 offen offset: 0 // (op 1560) Load value



buffer_load_dwordx4 v[22:25], v16, s[52:55], 0 offen // (op 1564) Load value
buffer_load_dwordx2 v[26:27], v16, s[52:55], 0 offen offset:16 // (op 1564) Load value
buffer_load_dwordx4 v[28:31], v16, s[52:55], 0 offen offset:24 // (op 1564) Load value
buffer_load_dwordx2 v[32:33], v16, s[52:55], 0 offen offset:40 // (op 1564) Load value

v_mov_b64 v[34:35], s[32:33] // (op 1567) convert

v_mov_b32 v21, v14 // (op 1567) 
buffer_load_ubyte v36, v21, s[60:63], 0 offen offset: 0 // (op 1567) Load value
v_add_u32 v21, v21, v34 // (op 1567) 
buffer_load_ubyte v37, v21, s[60:63], 0 offen offset: 0 // (op 1567) Load value





buffer_load_dwordx3 v[38:40], v12, s[48:51], 0 offen // (op 1570) Load value



buffer_load_ubyte v21, v15, s[56:59], 0 offen offset:4 // (op 1574) Load value




buffer_load_dwordx4 v[42:45], v16, s[52:55], 0 offen offset:96 // (op 1578) Load value
buffer_load_dwordx2 v[46:47], v16, s[52:55], 0 offen offset:112 // (op 1578) Load value
buffer_load_dwordx4 v[48:51], v16, s[52:55], 0 offen offset:120 // (op 1578) Load value
buffer_load_dwordx2 v[52:53], v16, s[52:55], 0 offen offset:136 // (op 1578) Load value


v_mov_b64 v[34:35], s[32:33] // (op 1581) convert

v_mov_b32 v41, v14 // (op 1581) 
buffer_load_ubyte v54, v41, s[60:63], 0 offen offset:4 // (op 1581) Load value
v_add_u32 v41, v41, v34 // (op 1581) 
buffer_load_ubyte v55, v41, s[60:63], 0 offen offset:4 // (op 1581) Load value






buffer_load_dwordx3 v[14:16], v12, s[48:51], 0 offen offset:96 // (op 1584) Load value

v_and_b32 v12, -2, v1 // (op 2070) 
v_sub_u32 v41, v1, v12 // (op 2070) 

v_lshlrev_b32 v12, 6, v41 // (op 2070) 

v_lshrrev_b32 v41, 1, v1 // (op 2070) 
v_lshl_add_u32 v34, v41, 7, v12 // (op 2070) 


v_mul_lo_u32 v12, v34, 6 // (op 2070) 

v_lshrrev_b32 v41, 3, v12 // (op 2070) 




s_waitcnt vmcnt(14)
ds_write_b128 v41, v[22:25] // (op 1586) Store local data 
s_waitcnt vmcnt(13)
ds_write_b64 v41, v[26:27] offset:16 // (op 1586) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v41, v[28:31] offset:24 // (op 1586) Store local data 
s_waitcnt vmcnt(11)
ds_write_b64 v41, v[32:33] offset:40 // (op 1586) Store local data 

v_and_b32 v12, -4, v1 // (op 2128) 
v_sub_u32 v41, v1, v12 // (op 2128) 

v_lshrrev_b32 v12, 2, v1 // (op 2128) 
v_lshl_add_u32 v34, v12, 3, v41 // (op 2128) 





s_waitcnt vmcnt(10)
ds_write_b8 v34, v36 offset:12288 // (op 1587) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v34, v37 offset:12292 // (op 1587) Store local data 

v_and_b32 v12, -8, v1 // (op 2017) 
v_sub_u32 v41, v1, v12 // (op 2017) 

v_lshlrev_b32 v12, 4, v41 // (op 2017) 

v_lshrrev_b32 v41, 3, v1 // (op 2017) 
v_lshl_add_u32 v34, v41, 7, v12 // (op 2017) 


v_mul_lo_u32 v12, v34, 6 // (op 2017) 

v_lshrrev_b32 v41, 3, v12 // (op 2017) 




s_waitcnt vmcnt(8)
ds_write_b96 v41, v[38:40] offset:12800 // (op 1588) Store local data 

v_and_b32 v12, -8, v1 // (op 1988) 
v_sub_u32 v41, v1, v12 // (op 1988) 

v_lshlrev_b32 v12, 4, v41 // (op 1988) 

v_lshrrev_b32 v41, 3, v1 // (op 1988) 
v_lshl_add_u32 v34, v41, 7, v12 // (op 1988) 


v_mul_lo_u32 v12, v34, 6 // (op 1988) 

v_lshrrev_b32 v41, 3, v12 // (op 1988) 

v_and_b32 v12, -2, v1 // (op 2041) 
v_sub_u32 v34, v1, v12 // (op 2041) 

v_lshlrev_b32 v12, 6, v34 // (op 2041) 

v_lshrrev_b32 v34, 1, v1 // (op 2041) 
v_lshl_add_u32 v35, v34, 7, v12 // (op 2041) 


v_mul_lo_u32 v12, v35, 6 // (op 2041) 

v_lshrrev_b32 v34, 3, v12 // (op 2041) 

v_and_b32 v12, -4, v1 // (op 2099) 
v_sub_u32 v35, v1, v12 // (op 2099) 

v_lshrrev_b32 v12, 2, v1 // (op 2099) 
v_lshl_add_u32 v56, v12, 3, v35 // (op 2099) 


v_and_b32 v12, -8, v1 // (op 2175) 
v_sub_u32 v35, v1, v12 // (op 2175) 

v_lshlrev_b32 v12, 4, v35 // (op 2175) 

v_lshrrev_b32 v35, 3, v1 // (op 2175) 
v_lshl_add_u32 v57, v35, 7, v12 // (op 2175) 


v_mul_lo_u32 v12, v57, 6 // (op 2175) 

v_lshrrev_b32 v35, 3, v12 // (op 2175) 

v_and_b32 v12, -2, v1 // (op 2199) 
v_sub_u32 v57, v1, v12 // (op 2199) 

v_lshlrev_b32 v12, 6, v57 // (op 2199) 

v_lshrrev_b32 v57, 1, v1 // (op 2199) 
v_lshl_add_u32 v58, v57, 7, v12 // (op 2199) 


v_mul_lo_u32 v12, v58, 6 // (op 2199) 

v_lshrrev_b32 v57, 3, v12 // (op 2199) 

v_and_b32 v12, -4, v1 // (op 2228) 
v_sub_u32 v58, v1, v12 // (op 2228) 

v_lshrrev_b32 v12, 2, v1 // (op 2228) 
v_lshl_add_u32 v59, v12, 3, v58 // (op 2228) 


s_mov_b32 s0, 0 // (op 624) call()
s_ashr_i64 s[6:7], s[36:37], 7 // (op 623) 
s_mov_b32 s1, s6 // (op 623) convert

s_and_b32 s6, s1, 4294967294 // (op 623) 

s_cmp_lt_u32 s0, s6 // (op 623) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopBottom_KLoop_623 // (op 623) Condition: Top (jump to RR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopBottom_KLoop_623 if false)
RR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopTop_KLoop_623:
s_add_u32 s0, s0, 2 // (op 625) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 2292) 



ds_read_b128 v[60:63], v3 offset:12800 // (op 67) Load local data 
ds_read_b64 v[64:65], v3 offset:12816 // (op 67) Load local data 
s_barrier  // (op 2305) 



ds_read_b128 v[66:69], v5 // (op 72) Load local data 
ds_read_b64 v[70:71], v5 offset:16 // (op 72) Load local data 
s_barrier  // (op 2319) 



ds_read_u8 v12, v0 offset:12288 // (op 81) Load local data 



ds_read_b128 v[72:75], v5 offset:3072 // (op 343) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[60:65], v[66:71], a[0:3], v20, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 84) 
ds_read_b64 v[76:77], v5 offset:3088 // (op 343) Load local data 





ds_read_u8 v12, v0 offset:12416 // (op 346) Load local data 



ds_read_b128 v[66:69], v5 offset:6144 // (op 499) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[60:65], v[72:77], a[4:7], v20, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 347) 
ds_read_b64 v[70:71], v5 offset:6160 // (op 499) Load local data 





ds_read_u8 v12, v0 offset:12544 // (op 502) Load local data 



ds_read_b128 v[72:75], v5 offset:9216 // (op 655) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[60:65], v[66:71], a[8:11], v20, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 503) 
ds_read_b64 v[76:77], v5 offset:9232 // (op 655) Load local data 





ds_read_u8 v12, v0 offset:12672 // (op 658) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[60:65], v[72:77], a[12:15], v20, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 659) 





v_mov_b32 v12, v18 // (op 77) 

buffer_load_ubyte v20, v12, s[56:59], 0 offen offset:8 // (op 77) Load value


v_mov_b32 v58, v17 // (op 10) 

buffer_load_dwordx4 v[22:25], v58, s[52:55], 0 offen offset:192 // (op 10) Load value
buffer_load_dwordx2 v[26:27], v58, s[52:55], 0 offen offset:208 // (op 10) Load value
buffer_load_dwordx4 v[28:31], v58, s[52:55], 0 offen offset:216 // (op 10) Load value
buffer_load_dwordx2 v[32:33], v58, s[52:55], 0 offen offset:232 // (op 10) Load value

v_mov_b64 v[60:61], s[32:33] // (op 21) convert

v_mov_b32 v62, v19 // (op 21) 
v_mov_b32 v63, v62 // (op 21) 
buffer_load_ubyte v36, v63, s[60:63], 0 offen offset:8 // (op 21) Load value
v_add_u32 v63, v63, v60 // (op 21) 
buffer_load_ubyte v37, v63, s[60:63], 0 offen offset:8 // (op 21) Load value




v_mov_b32 v60, v13 // (op 4) 

buffer_load_dwordx3 v[38:40], v60, s[48:51], 0 offen offset:192 // (op 4) Load value
s_barrier  // (op 2342) 



s_waitcnt vmcnt(14)
ds_write_b128 v57, v[42:45] offset:15872 // (op 239) Store local data 
s_waitcnt vmcnt(13)
ds_write_b64 v57, v[46:47] offset:15888 // (op 239) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v57, v[48:51] offset:15896 // (op 239) Store local data 
s_waitcnt vmcnt(11)
ds_write_b64 v57, v[52:53] offset:15912 // (op 239) Store local data 
s_barrier  // (op 2348) 



s_waitcnt vmcnt(10)
ds_write_b8 v59, v54 offset:28160 // (op 240) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v59, v55 offset:28164 // (op 240) Store local data 
s_barrier  // (op 2336) 



s_waitcnt vmcnt(8)
ds_write_b96 v35, v[14:16] offset:28672 // (op 229) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 2333) 



ds_read_b128 v[64:67], v4 offset:28672 // (op 231) Load local data 
ds_read_b64 v[68:69], v4 offset:28688 // (op 231) Load local data 
s_barrier  // (op 2339) 
s_barrier  // (op 2389) 



ds_read_b128 v[70:73], v8 offset:15872 // (op 233) Load local data 
ds_read_b64 v[74:75], v8 offset:15888 // (op 233) Load local data 
s_barrier  // (op 2345) 



ds_read_u8 v61, v7 offset:28160 // (op 237) Load local data 



ds_read_b128 v[76:79], v8 offset:18944 // (op 352) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[64:69], v[70:75], a[0:3], v21, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 238) 
ds_read_b64 v[80:81], v8 offset:18960 // (op 352) Load local data 





ds_read_u8 v61, v7 offset:28288 // (op 355) Load local data 



ds_read_b128 v[70:73], v8 offset:22016 // (op 508) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[64:69], v[76:81], a[4:7], v21, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 348) 
ds_read_b64 v[74:75], v8 offset:22032 // (op 508) Load local data 





ds_read_u8 v61, v7 offset:28416 // (op 511) Load local data 



ds_read_b128 v[76:79], v8 offset:25088 // (op 664) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[64:69], v[70:75], a[8:11], v21, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 504) 
ds_read_b64 v[80:81], v8 offset:25104 // (op 664) Load local data 





s_barrier  // (op 2435) 
ds_read_u8 v61, v7 offset:28544 // (op 667) Load local data 
s_barrier  // (op 2439) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[64:69], v[76:81], a[12:15], v21, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 660) 






buffer_load_ubyte v21, v12, s[56:59], 0 offen offset:12 // (op 235) Load value




v_add_u32 v18, 8, v18 // (op 1878) 
buffer_load_dwordx4 v[42:45], v58, s[52:55], 0 offen offset:288 // (op 241) Load value
buffer_load_dwordx2 v[46:47], v58, s[52:55], 0 offen offset:304 // (op 241) Load value
buffer_load_dwordx4 v[48:51], v58, s[52:55], 0 offen offset:312 // (op 241) Load value
buffer_load_dwordx2 v[52:53], v58, s[52:55], 0 offen offset:328 // (op 241) Load value


v_add_u32 v17, 192, v17 // (op 1814) 
v_mov_b64 v[64:65], s[32:33] // (op 242) convert

v_mov_b32 v12, v62 // (op 242) 
buffer_load_ubyte v54, v12, s[60:63], 0 offen offset:12 // (op 242) Load value
v_add_u32 v12, v12, v64 // (op 242) 
buffer_load_ubyte v55, v12, s[60:63], 0 offen offset:12 // (op 242) Load value






v_add_u32 v19, 8, v19 // (op 1942) 
buffer_load_dwordx3 v[14:16], v60, s[48:51], 0 offen offset:288 // (op 228) Load value

s_barrier  // (op 2308) 
v_add_u32 v13, 192, v13 // (op 1750) 



s_waitcnt vmcnt(14)
ds_write_b128 v34, v[22:25] // (op 13) Store local data 
s_waitcnt vmcnt(13)
ds_write_b64 v34, v[26:27] offset:16 // (op 13) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v34, v[28:31] offset:24 // (op 13) Store local data 
s_waitcnt vmcnt(11)
ds_write_b64 v34, v[32:33] offset:40 // (op 13) Store local data 
s_barrier  // (op 2322) 



s_waitcnt vmcnt(10)
ds_write_b8 v56, v36 offset:12288 // (op 26) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v56, v37 offset:12292 // (op 26) Store local data 
s_barrier  // (op 2295) 



s_waitcnt vmcnt(8)
ds_write_b96 v41, v[38:40] offset:12800 // (op 6) Store local data 
s_ashr_i64 s[6:7], s[36:37], 7 // (op 623) 
s_mov_b32 s1, s6 // (op 623) convert

s_and_b32 s6, s1, 4294967294 // (op 623) 

s_cmp_lt_u32 s0, s6 // (op 623) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopTop_KLoop_623 // (op 623) Condition: Bottom (jump to RR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopTop_KLoop_623 if true)
RR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopBottom_KLoop_623:















v_lshrrev_b32 v12, 3, v1 // (op 1710) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1710) 
v_lshl_add_u32 v13, s16, 5, v12 // (op 1710) 

s_lshr_b32 s6, s0, 1 // (op 1710) 

v_and_b32 v12, -8, v1 // (op 1710) 
v_sub_u32 v17, v1, v12 // (op 1710) 

v_lshlrev_b32 v12, 4, v17 // (op 1710) 

v_lshl_add_u32 v17, s6, 8, v12 // (op 1710) 


v_mov_b32 v19, 0 // (op 1710) convert
v_mov_b32 v18, v17 // (op 1710) convert

v_mul_lo_u32 v23, v13, s13 // (op 1710) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 1710) most significant: high of low * low
v_mul_lo_u32 v22, v13, s12 // (op 1710) least significant: low of low * low
v_add_u32 v23, v23, v12 // (op 1710) most significant: sum

v_add_co_u32 v22, s[0:1], v22, v18 // (op 1710) least significant half
v_addc_co_u32 v23, s[0:1], v23, v19, s[0:1] // (op 1710) most significant half



v_mul_lo_u32 v13, v23, 6 // (op 1710) most significant: low of high * low
v_mul_hi_u32 v17, v22, 6 // (op 1710) most significant: high of low * low
v_mul_lo_u32 v12, v22, 6 // (op 1710) least significant: low of low * low
v_add_u32 v13, v13, v17 // (op 1710) most significant: sum


v_lshrrev_b64 v[22:23], 3, v[12:13] // (op 1710) 

v_mov_b32 v12, v22 // (op 1710) convert

s_mul_i32 s1, s23, 6 // (op 1712) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 1712) least significant
s_mul_hi_u32 s6, s22, 6 // (op 1712) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1712) most significant: sum

s_mul_i32 s7, s12, s1 // (op 1712) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 1712) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 1712) least significant
s_mul_hi_u32 s11, s12, s0 // (op 1712) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 1712) most significant: sum
s_add_u32 s7, s7, s11 // (op 1712) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 1712) 

s_mov_b32 s48, s20 // (op 1712) reg expression
s_mov_b32 s49, s21 // (op 1712) reg expression
s_mov_b32 s50, s0 // (op 1712) reg expression
s_mov_b32 s51, 131072 // (op 1712) call()

v_and_b32 v13, -2, v1 // (op 1780) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1780) 
v_sub_u32 v17, v1, v13 // (op 1780) 

s_lshr_b32 s6, s0, 1 // (op 1780) 

v_lshlrev_b32 v13, 6, v17 // (op 1780) 

v_lshl_add_u32 v17, s6, 8, v13 // (op 1780) 


v_mov_b32 v19, 0 // (op 1780) convert
v_mov_b32 v18, v17 // (op 1780) convert

v_lshrrev_b32 v13, 1, v1 // (op 1780) 
v_lshl_add_u32 v17, s17, 7, v13 // (op 1780) 

v_mul_lo_u32 v23, v17, s25 // (op 1780) most significant: low of low * high
v_mul_hi_u32 v13, v17, s24 // (op 1780) most significant: high of low * low
v_mul_lo_u32 v22, v17, s24 // (op 1780) least significant: low of low * low
v_add_u32 v23, v23, v13 // (op 1780) most significant: sum


v_add_co_u32 v24, s[0:1], v18, v22 // (op 1780) least significant half
v_addc_co_u32 v25, s[0:1], v19, v23, s[0:1] // (op 1780) most significant half



v_mul_lo_u32 v23, v25, 6 // (op 1780) most significant: low of high * low
v_mul_hi_u32 v13, v24, 6 // (op 1780) most significant: high of low * low
v_mul_lo_u32 v22, v24, 6 // (op 1780) least significant: low of low * low
v_add_u32 v23, v23, v13 // (op 1780) most significant: sum


v_lshrrev_b64 v[24:25], 3, v[22:23] // (op 1780) 

v_mov_b32 v13, v24 // (op 1780) convert

s_mul_i32 s1, s9, 6 // (op 1782) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 1782) least significant
s_mul_hi_u32 s6, s8, 6 // (op 1782) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1782) most significant: sum

s_mul_i32 s7, s24, s1 // (op 1782) most significant: low of low * high
s_mul_i32 s10, s25, s0 // (op 1782) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 1782) least significant
s_mul_hi_u32 s11, s24, s0 // (op 1782) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 1782) most significant: sum
s_add_u32 s7, s7, s11 // (op 1782) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 1782) 

s_mov_b32 s52, s26 // (op 1782) reg expression
s_mov_b32 s53, s27 // (op 1782) reg expression
s_mov_b32 s54, s0 // (op 1782) reg expression
s_mov_b32 s55, 131072 // (op 1782) call()

v_lshrrev_b32 v17, 7, v1 // (op 1844) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1844) 
v_and_b32 v18, 15, v1 // (op 1844) 
s_lshr_b32 s6, s0, 1 // (op 1844) 

v_lshl_add_u32 v19, v17, 4, v18 // (op 1844) 


v_lshl_add_u32 v20, s16, 5, v19 // (op 1844) 

v_and_b32 v17, 63, v1 // (op 1844) 
v_lshrrev_b32 v18, 4, v17 // (op 1844) 

v_lshl_add_u32 v17, s6, 3, v18 // (op 1844) 


v_mul_lo_u32 v18, v20, s28 // (op 1844) 
v_add_u32 v18, v18, v17 // (op 1844) 


s_mul_i32 s1, s28, s23 // (op 1846) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 1846) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 1846) least significant
s_mul_hi_u32 s7, s28, s22 // (op 1846) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1846) most significant: sum
s_add_u32 s1, s1, s7 // (op 1846) most significant: sum


s_mov_b32 s56, s30 // (op 1846) reg expression
s_mov_b32 s57, s31 // (op 1846) reg expression
s_mov_b32 s58, s0 // (op 1846) reg expression
s_mov_b32 s59, 131072 // (op 1846) call()

v_and_b32 v17, -4, v1 // (op 1908) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1908) 
v_sub_u32 v19, v1, v17 // (op 1908) 

s_lshr_b32 s6, s0, 1 // (op 1908) 

v_lshrrev_b32 v17, 2, v1 // (op 1908) 
v_lshl_add_u32 v20, s6, 3, v19 // (op 1908) 


v_lshlrev_b32 v19, 1, v17 // (op 1908) 

v_lshl_add_u32 v17, s17, 7, v19 // (op 1908) 

v_mul_lo_u32 v19, v17, s32 // (op 1908) 

v_add_u32 v17, v20, v19 // (op 1908) 


s_mul_i32 s1, s32, s9 // (op 1910) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 1910) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 1910) least significant
s_mul_hi_u32 s7, s32, s8 // (op 1910) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1910) most significant: sum
s_add_u32 s1, s1, s7 // (op 1910) most significant: sum


s_mov_b32 s60, s34 // (op 1910) reg expression
s_mov_b32 s61, s35 // (op 1910) reg expression
s_mov_b32 s62, s0 // (op 1910) reg expression
s_mov_b32 s63, 131072 // (op 1910) call()

v_and_b32 v19, -8, v1 // (op 2001) 
v_sub_u32 v20, v1, v19 // (op 2001) 

v_lshlrev_b32 v19, 4, v20 // (op 2001) 

v_lshrrev_b32 v20, 3, v1 // (op 2001) 
v_lshl_add_u32 v22, v20, 7, v19 // (op 2001) 


v_mul_lo_u32 v19, v22, 6 // (op 2001) 

v_lshrrev_b32 v20, 3, v19 // (op 2001) 

v_and_b32 v19, -2, v1 // (op 2054) 
v_sub_u32 v22, v1, v19 // (op 2054) 

v_lshlrev_b32 v19, 6, v22 // (op 2054) 

v_lshrrev_b32 v22, 1, v1 // (op 2054) 
v_lshl_add_u32 v23, v22, 7, v19 // (op 2054) 


v_mul_lo_u32 v19, v23, 6 // (op 2054) 

v_lshrrev_b32 v22, 3, v19 // (op 2054) 

v_and_b32 v19, -4, v1 // (op 2112) 
v_sub_u32 v23, v1, v19 // (op 2112) 

v_lshrrev_b32 v19, 2, v1 // (op 2112) 
v_lshl_add_u32 v24, v19, 3, v23 // (op 2112) 


s_ashr_i64 s[0:1], s[36:37], 7 // (op 629) 
s_and_b32 s6, s0, 4294967294 // (op 629) 

s_ashr_i64 s[0:1], s[36:37], 7 // (op 628) 
s_mov_b32 s7, s0 // (op 628) convert

s_cmp_lt_u32 s6, s7 // (op 628) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopBottom_KLoopTail_628 // (op 628) Condition: Top (jump to RR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopBottom_KLoopTail_628 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopTop_KLoopTail_628:


s_add_u32 s6, s6, 1 // (op 630) 
v_mov_b32 v19, v12 // (op 170) 

buffer_load_dwordx3 v[14:16], v19, s[48:51], 0 offen // (op 170) Load value

v_add_u32 v12, 96, v12 // (op 1700) 


s_barrier  // (op 2298) 
v_mov_b32 v19, v13 // (op 183) 




buffer_load_dwordx4 v[26:29], v19, s[52:55], 0 offen // (op 183) Load value
buffer_load_dwordx2 v[30:31], v19, s[52:55], 0 offen offset:16 // (op 183) Load value
s_waitcnt vmcnt(2)
ds_write_b96 v20, v[14:16] offset:12800 // (op 171) Store local data 
buffer_load_dwordx4 v[32:35], v19, s[52:55], 0 offen offset:24 // (op 183) Load value

buffer_load_dwordx2 v[36:37], v19, s[52:55], 0 offen offset:40 // (op 183) Load value

v_add_u32 v13, 96, v13 // (op 1772) 

s_barrier  // (op 2311) 
v_mov_b64 v[14:15], s[32:33] // (op 184) convert




v_mov_b32 v16, v17 // (op 184) 
v_mov_b32 v19, v16 // (op 184) 
buffer_load_ubyte v38, v19, s[60:63], 0 offen offset: 0 // (op 184) Load value
v_add_u32 v19, v19, v14 // (op 184) 
s_waitcnt vmcnt(4)
ds_write_b128 v22, v[26:29] // (op 181) Store local data 
s_waitcnt vmcnt(3)
ds_write_b64 v22, v[30:31] offset:16 // (op 181) Store local data 
s_waitcnt vmcnt(2)
ds_write_b128 v22, v[32:35] offset:24 // (op 181) Store local data 
buffer_load_ubyte v39, v19, s[60:63], 0 offen offset: 0 // (op 184) Load value



v_add_u32 v17, 4, v17 // (op 1900) 
s_waitcnt vmcnt(2)
ds_write_b64 v22, v[36:37] offset:40 // (op 181) Store local data 

s_waitcnt lgkmcnt(4)
s_barrier  // (op 2301) 



s_barrier  // (op 2325) 
ds_read_b128 v[32:35], v3 offset:12800 // (op 173) Load local data 



ds_read_b64 v[36:37], v3 offset:12816 // (op 173) Load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v24, v38 offset:12288 // (op 182) Store local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 2315) 
s_waitcnt vmcnt(0)
ds_write_b8 v24, v39 offset:12292 // (op 182) Store local data 




ds_read_b128 v[26:29], v5 // (op 175) Load local data 
ds_read_b64 v[30:31], v5 offset:16 // (op 175) Load local data 


v_mov_b32 v19, v18 // (op 177) 

buffer_load_ubyte v21, v19, s[56:59], 0 offen offset: 0 // (op 177) Load value

v_add_u32 v18, 4, v18 // (op 1836) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 2329) 



ds_read_u8 v19, v0 offset:12288 // (op 179) Load local data 



s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[32:37], v[26:31], a[0:3], v21, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 180) 
ds_read_b128 v[38:41], v5 offset:3072 // (op 324) Load local data 


ds_read_b64 v[42:43], v5 offset:3088 // (op 324) Load local data 



ds_read_u8 v19, v0 offset:12416 // (op 327) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[32:37], v[38:43], a[4:7], v21, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 328) 
ds_read_b128 v[26:29], v5 offset:6144 // (op 480) Load local data 


ds_read_b64 v[30:31], v5 offset:6160 // (op 480) Load local data 



ds_read_u8 v19, v0 offset:12544 // (op 483) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[32:37], v[26:31], a[8:11], v21, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 484) 
ds_read_b128 v[38:41], v5 offset:9216 // (op 636) Load local data 


ds_read_b64 v[42:43], v5 offset:9232 // (op 636) Load local data 



ds_read_u8 v19, v0 offset:12672 // (op 639) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[32:37], v[38:43], a[12:15], v21, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 640) 




s_ashr_i64 s[0:1], s[36:37], 7 // (op 628) 
s_mov_b32 s7, s0 // (op 628) convert

s_cmp_lt_u32 s6, s7 // (op 628) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopTop_KLoopTail_628 // (op 628) Condition: Bottom (jump to RR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopTop_KLoopTail_628 if true)
RR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopBottom_KLoopTail_628:






















s_barrier  // (op 2423) 







v_cmp_eq_f32 vcc, 0.00000, v10 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ConditionalFalse_0_beta_125
v_accvgpr_read v12, a0 // (op 46) 
v_accvgpr_read v13, a1 // (op 46) 
v_accvgpr_read v14, a2 // (op 46) 
v_accvgpr_read v15, a3 // (op 46) 
v_mul_f32 v16, v11, v12 // (op 46) 
v_mul_f32 v17, v11, v13 // (op 46) 
v_mul_f32 v18, v11, v14 // (op 46) 
v_mul_f32 v19, v11, v15 // (op 46) 

v_lshrrev_b32 v12, 16, v16 // (op 52) 
v_lshrrev_b32 v13, 16, v17 // (op 52) 
v_lshrrev_b32 v14, 16, v18 // (op 52) 
v_lshrrev_b32 v15, 16, v19 // (op 52) 


v_pack_b32_f16 v4, v12, v13 // (op 54) 
v_pack_b32_f16 v5, v14, v15 // (op 54) 

v_mov_b32 v0, v9 // (op 54) reg expression

buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 54) Store value



v_accvgpr_read v12, a4 // (op 864) 
v_accvgpr_read v13, a5 // (op 864) 
v_accvgpr_read v14, a6 // (op 864) 
v_accvgpr_read v15, a7 // (op 864) 
v_mul_f32 v16, v11, v12 // (op 864) 
v_mul_f32 v17, v11, v13 // (op 864) 
v_mul_f32 v18, v11, v14 // (op 864) 
v_mul_f32 v19, v11, v15 // (op 864) 

v_lshrrev_b32 v12, 16, v16 // (op 866) 
v_lshrrev_b32 v13, 16, v17 // (op 866) 
v_lshrrev_b32 v14, 16, v18 // (op 866) 
v_lshrrev_b32 v15, 16, v19 // (op 866) 


v_pack_b32_f16 v4, v12, v13 // (op 867) 
v_pack_b32_f16 v5, v14, v15 // (op 867) 

s_lshl_b32 s0, s4, 6 // (op 867) 
v_add_u32 v0, v9, s0 // (op 867) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 867) Store value



v_accvgpr_read v12, a8 // (op 878) 
v_accvgpr_read v13, a9 // (op 878) 
v_accvgpr_read v14, a10 // (op 878) 
v_accvgpr_read v15, a11 // (op 878) 
v_mul_f32 v16, v11, v12 // (op 878) 
v_mul_f32 v17, v11, v13 // (op 878) 
v_mul_f32 v18, v11, v14 // (op 878) 
v_mul_f32 v19, v11, v15 // (op 878) 

v_lshrrev_b32 v12, 16, v16 // (op 880) 
v_lshrrev_b32 v13, 16, v17 // (op 880) 
v_lshrrev_b32 v14, 16, v18 // (op 880) 
v_lshrrev_b32 v15, 16, v19 // (op 880) 


v_pack_b32_f16 v4, v12, v13 // (op 881) 
v_pack_b32_f16 v5, v14, v15 // (op 881) 

s_lshl_b32 s0, s4, 7 // (op 881) 
v_add_u32 v0, v9, s0 // (op 881) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 881) Store value



v_accvgpr_read v12, a12 // (op 892) 
v_accvgpr_read v13, a13 // (op 892) 
v_accvgpr_read v14, a14 // (op 892) 
v_accvgpr_read v15, a15 // (op 892) 
v_mul_f32 v16, v11, v12 // (op 892) 
v_mul_f32 v17, v11, v13 // (op 892) 
v_mul_f32 v18, v11, v14 // (op 892) 
v_mul_f32 v19, v11, v15 // (op 892) 

v_lshrrev_b32 v12, 16, v16 // (op 894) 
v_lshrrev_b32 v13, 16, v17 // (op 894) 
v_lshrrev_b32 v14, 16, v18 // (op 894) 
v_lshrrev_b32 v15, 16, v19 // (op 894) 


v_pack_b32_f16 v4, v12, v13 // (op 895) 
v_pack_b32_f16 v5, v14, v15 // (op 895) 

s_lshl_b32 s0, s4, 6 // (op 895) 
s_mul_i32 s1, 3, s0 // (op 895) 

v_add_u32 v0, v9, s1 // (op 895) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 895) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ConditionalBottom_0_beta_125
RR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ConditionalFalse_0_beta_125:


v_mov_b32 v0, v6 // (op 138) reg expression

buffer_load_dwordx2 v[4:5], v0, s[40:43], 0 offen // (op 138) Load value

v_accvgpr_read v12, a0 // (op 134) 
v_accvgpr_read v13, a1 // (op 134) 
v_accvgpr_read v14, a2 // (op 134) 
v_accvgpr_read v15, a3 // (op 134) 
v_mul_f32 v16, v11, v12 // (op 134) 
v_mul_f32 v17, v11, v13 // (op 134) 
v_mul_f32 v18, v11, v14 // (op 134) 
v_mul_f32 v19, v11, v15 // (op 134) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v4 // (op 139) 
v_lshrrev_b32 v13, 16, v4 // (op 139) 
v_lshlrev_b32 v12, 16, v12 // (op 139) 
v_lshlrev_b32 v13, 16, v13 // (op 139) 
v_mul_f32 v20, v10, v12 // (op 139) 
v_mul_f32 v21, v10, v13 // (op 139) 
v_and_b32 v12, 65535, v5 // (op 139) 
v_lshrrev_b32 v13, 16, v5 // (op 139) 
v_lshlrev_b32 v12, 16, v12 // (op 139) 
v_lshlrev_b32 v13, 16, v13 // (op 139) 
v_mul_f32 v22, v10, v12 // (op 139) 
v_mul_f32 v23, v10, v13 // (op 139) 


v_add_f32 v12, v20, v16 // (op 135) 
v_add_f32 v13, v21, v17 // (op 135) 
v_add_f32 v14, v22, v18 // (op 135) 
v_add_f32 v15, v23, v19 // (op 135) 


v_lshrrev_b32 v16, 16, v12 // (op 136) 
v_lshrrev_b32 v17, 16, v13 // (op 136) 
v_lshrrev_b32 v18, 16, v14 // (op 136) 
v_lshrrev_b32 v19, 16, v15 // (op 136) 


v_pack_b32_f16 v4, v16, v17 // (op 137) 
v_pack_b32_f16 v5, v18, v19 // (op 137) 

v_mov_b32 v0, v9 // (op 137) reg expression

buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 137) Store value





s_lshl_b32 s0, s2, 6 // (op 929) 
v_add_u32 v0, v6, s0 // (op 929) 


buffer_load_dwordx2 v[4:5], v0, s[40:43], 0 offen // (op 929) Load value

v_accvgpr_read v12, a4 // (op 925) 
v_accvgpr_read v13, a5 // (op 925) 
v_accvgpr_read v14, a6 // (op 925) 
v_accvgpr_read v15, a7 // (op 925) 
v_mul_f32 v16, v11, v12 // (op 925) 
v_mul_f32 v17, v11, v13 // (op 925) 
v_mul_f32 v18, v11, v14 // (op 925) 
v_mul_f32 v19, v11, v15 // (op 925) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v4 // (op 930) 
v_lshrrev_b32 v13, 16, v4 // (op 930) 
v_lshlrev_b32 v12, 16, v12 // (op 930) 
v_lshlrev_b32 v13, 16, v13 // (op 930) 
v_mul_f32 v20, v10, v12 // (op 930) 
v_mul_f32 v21, v10, v13 // (op 930) 
v_and_b32 v12, 65535, v5 // (op 930) 
v_lshrrev_b32 v13, 16, v5 // (op 930) 
v_lshlrev_b32 v12, 16, v12 // (op 930) 
v_lshlrev_b32 v13, 16, v13 // (op 930) 
v_mul_f32 v22, v10, v12 // (op 930) 
v_mul_f32 v23, v10, v13 // (op 930) 


v_add_f32 v12, v20, v16 // (op 926) 
v_add_f32 v13, v21, v17 // (op 926) 
v_add_f32 v14, v22, v18 // (op 926) 
v_add_f32 v15, v23, v19 // (op 926) 


v_lshrrev_b32 v16, 16, v12 // (op 927) 
v_lshrrev_b32 v17, 16, v13 // (op 927) 
v_lshrrev_b32 v18, 16, v14 // (op 927) 
v_lshrrev_b32 v19, 16, v15 // (op 927) 


v_pack_b32_f16 v4, v16, v17 // (op 928) 
v_pack_b32_f16 v5, v18, v19 // (op 928) 

s_lshl_b32 s0, s4, 6 // (op 928) 
v_add_u32 v0, v9, s0 // (op 928) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 928) Store value





s_lshl_b32 s0, s2, 7 // (op 943) 
v_add_u32 v0, v6, s0 // (op 943) 


buffer_load_dwordx2 v[4:5], v0, s[40:43], 0 offen // (op 943) Load value

v_accvgpr_read v12, a8 // (op 939) 
v_accvgpr_read v13, a9 // (op 939) 
v_accvgpr_read v14, a10 // (op 939) 
v_accvgpr_read v15, a11 // (op 939) 
v_mul_f32 v16, v11, v12 // (op 939) 
v_mul_f32 v17, v11, v13 // (op 939) 
v_mul_f32 v18, v11, v14 // (op 939) 
v_mul_f32 v19, v11, v15 // (op 939) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v4 // (op 944) 
v_lshrrev_b32 v13, 16, v4 // (op 944) 
v_lshlrev_b32 v12, 16, v12 // (op 944) 
v_lshlrev_b32 v13, 16, v13 // (op 944) 
v_mul_f32 v20, v10, v12 // (op 944) 
v_mul_f32 v21, v10, v13 // (op 944) 
v_and_b32 v12, 65535, v5 // (op 944) 
v_lshrrev_b32 v13, 16, v5 // (op 944) 
v_lshlrev_b32 v12, 16, v12 // (op 944) 
v_lshlrev_b32 v13, 16, v13 // (op 944) 
v_mul_f32 v22, v10, v12 // (op 944) 
v_mul_f32 v23, v10, v13 // (op 944) 


v_add_f32 v12, v20, v16 // (op 940) 
v_add_f32 v13, v21, v17 // (op 940) 
v_add_f32 v14, v22, v18 // (op 940) 
v_add_f32 v15, v23, v19 // (op 940) 


v_lshrrev_b32 v16, 16, v12 // (op 941) 
v_lshrrev_b32 v17, 16, v13 // (op 941) 
v_lshrrev_b32 v18, 16, v14 // (op 941) 
v_lshrrev_b32 v19, 16, v15 // (op 941) 


v_pack_b32_f16 v4, v16, v17 // (op 942) 
v_pack_b32_f16 v5, v18, v19 // (op 942) 

s_lshl_b32 s0, s4, 7 // (op 942) 
v_add_u32 v0, v9, s0 // (op 942) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 942) Store value





s_lshl_b32 s0, s2, 6 // (op 957) 
s_mul_i32 s1, 3, s0 // (op 957) 

v_add_u32 v0, v6, s1 // (op 957) 


buffer_load_dwordx2 v[4:5], v0, s[40:43], 0 offen // (op 957) Load value

v_accvgpr_read v12, a12 // (op 953) 
v_accvgpr_read v13, a13 // (op 953) 
v_accvgpr_read v14, a14 // (op 953) 
v_accvgpr_read v15, a15 // (op 953) 
v_mul_f32 v16, v11, v12 // (op 953) 
v_mul_f32 v17, v11, v13 // (op 953) 
v_mul_f32 v18, v11, v14 // (op 953) 
v_mul_f32 v19, v11, v15 // (op 953) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v4 // (op 958) 
v_lshrrev_b32 v13, 16, v4 // (op 958) 
v_lshlrev_b32 v12, 16, v12 // (op 958) 
v_lshlrev_b32 v13, 16, v13 // (op 958) 
v_mul_f32 v20, v10, v12 // (op 958) 
v_mul_f32 v21, v10, v13 // (op 958) 
v_and_b32 v12, 65535, v5 // (op 958) 
v_lshrrev_b32 v13, 16, v5 // (op 958) 
v_lshlrev_b32 v12, 16, v12 // (op 958) 
v_lshlrev_b32 v13, 16, v13 // (op 958) 
v_mul_f32 v22, v10, v12 // (op 958) 
v_mul_f32 v23, v10, v13 // (op 958) 


v_add_f32 v12, v20, v16 // (op 954) 
v_add_f32 v13, v21, v17 // (op 954) 
v_add_f32 v14, v22, v18 // (op 954) 
v_add_f32 v15, v23, v19 // (op 954) 


v_lshrrev_b32 v16, 16, v12 // (op 955) 
v_lshrrev_b32 v17, 16, v13 // (op 955) 
v_lshrrev_b32 v18, 16, v14 // (op 955) 
v_lshrrev_b32 v19, 16, v15 // (op 955) 


v_pack_b32_f16 v4, v16, v17 // (op 956) 
v_pack_b32_f16 v5, v18, v19 // (op 956) 

s_lshl_b32 s0, s4, 6 // (op 956) 
s_mul_i32 s1, 3, s0 // (op 956) 

v_add_u32 v0, v9, s1 // (op 956) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 956) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ConditionalBottom_0_beta_125:












s_endpgm  // End of RR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2
.LRR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_end:
.size RR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2, .LRR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_end-RR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2
  .amdhsa_next_free_vgpr 100
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 31744 // lds bytes
  .amdhsa_accum_offset 84
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
    DataType: F6
    DataTypeA: F6
    DataTypeB: F6
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
    macrotile: [32, 128, 128]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 4]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2
    .symbol: RR_GEMM_TN_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 31744
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 64
    .vgpr_count: 82
    .agpr_count: 16
    .max_flat_workgroup_size: 256
    .workgroup_size: [256, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, {is-null: true}]
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
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: FP6, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP6, pointerType: PointerGlobal}
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

