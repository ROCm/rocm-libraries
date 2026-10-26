

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2
.p2align 8
.type RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2,@function
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 2085) 
v_and_b32 v3, 15, v1 // (op 2085) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 2085) 


v_and_b32 v0, 48, v1 // (op 2085) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 2085) 


v_lshrrev_b32 v0, 7, v1 // (op 2257) 
v_and_b32 v4, 15, v1 // (op 2257) 
v_lshl_add_u32 v5, v0, 4, v4 // (op 2257) 


v_and_b32 v0, 48, v1 // (op 2257) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 2257) 


v_lshrrev_b32 v0, 7, v1 // (op 2029) 
v_and_b32 v5, 63, v1 // (op 2029) 
v_lshrrev_b32 v6, 4, v5 // (op 2029) 
v_lshlrev_b32 v7, 2, v6 // (op 2029) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 2029) 


v_lshl_add_u32 v0, s16, 5, v6 // (op 2029) 

v_lshrrev_b32 v6, 2, v5 // (op 2029) 

v_and_b32 v5, 3, v6 // (op 2029) 

v_and_b32 v6, 3, v1 // (op 2029) 
v_lshl_add_u32 v7, v5, 2, v6 // (op 2029) 


v_lshrrev_b32 v5, 6, v1 // (op 2029) 
v_and_b32 v6, 1, v5 // (op 2029) 

v_lshl_add_u32 v5, v6, 4, v7 // (op 2029) 


v_lshl_add_u32 v6, s17, 7, v5 // (op 2029) 

v_mul_lo_u32 v5, v6, s2 // (op 2029) 

v_add_lshl_u32 v6, v0, v5, 1 // (op 2029) 


v_lshrrev_b32 v0, 6, v1 // (op 2142) 
v_and_b32 v5, 1, v0 // (op 2142) 

v_and_b32 v0, 15, v1 // (op 2142) 
v_lshl_add_u32 v7, v5, 4, v0 // (op 2142) 


v_and_b32 v0, 48, v1 // (op 2142) 
v_lshl_add_u32 v5, v7, 7, v0 // (op 2142) 


v_lshrrev_b32 v0, 6, v1 // (op 2200) 
v_and_b32 v7, 1, v0 // (op 2200) 

v_and_b32 v0, 15, v1 // (op 2200) 
s_mul_i32 s1, s2, s9 // (op 2031) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 2031) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 2031) least significant
s_mul_hi_u32 s19, s2, s8 // (op 2031) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2031) most significant: sum
s_add_u32 s1, s1, s19 // (op 2031) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 2031) 

v_lshl_add_u32 v8, v7, 4, v0 // (op 2200) 


v_and_b32 v0, 63, v1 // (op 2200) 
v_lshrrev_b32 v7, 4, v0 // (op 2200) 

v_lshl_add_u32 v0, v8, 2, v7 // (op 2200) 


v_lshrrev_b32 v7, 7, v1 // (op 2216) 
v_and_b32 v8, 63, v1 // (op 2216) 
v_lshrrev_b32 v9, 4, v8 // (op 2216) 
v_lshlrev_b32 v10, 2, v9 // (op 2216) 

v_lshl_add_u32 v9, v7, 4, v10 // (op 2216) 


v_lshl_add_u32 v7, s16, 5, v9 // (op 2216) 

v_lshrrev_b32 v9, 2, v8 // (op 2216) 

v_and_b32 v8, 3, v9 // (op 2216) 

v_and_b32 v9, 3, v1 // (op 2216) 
v_lshl_add_u32 v10, v8, 2, v9 // (op 2216) 


v_lshrrev_b32 v8, 6, v1 // (op 2216) 
v_and_b32 v9, 1, v8 // (op 2216) 

v_lshl_add_u32 v8, v9, 4, v10 // (op 2216) 


v_lshl_add_u32 v9, s17, 7, v8 // (op 2216) 

v_mul_lo_u32 v8, v9, s4 // (op 2216) 

v_add_lshl_u32 v9, v7, v8, 1 // (op 2216) 


v_lshrrev_b32 v7, 6, v1 // (op 2301) 
v_and_b32 v8, 1, v7 // (op 2301) 

v_and_b32 v7, 15, v1 // (op 2301) 
v_lshl_add_u32 v10, v8, 4, v7 // (op 2301) 


v_and_b32 v7, 48, v1 // (op 2301) 
v_lshl_add_u32 v8, v10, 7, v7 // (op 2301) 


v_lshrrev_b32 v7, 6, v1 // (op 2330) 
v_and_b32 v10, 1, v7 // (op 2330) 

v_and_b32 v7, 15, v1 // (op 2330) 
s_mul_i32 s1, s4, s9 // (op 2218) most significant: low of low * high
v_lshl_add_u32 v11, v10, 4, v7 // (op 2330) 


v_and_b32 v7, 63, v1 // (op 2330) 
v_lshrrev_b32 v10, 4, v7 // (op 2330) 

v_lshl_add_u32 v7, v11, 2, v10 // (op 2330) 


s_mul_i32 s38, s5, s8 // (op 2218) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 2218) least significant
s_mul_hi_u32 s39, s4, s8 // (op 2218) most significant: high of low * low
s_add_u32 s1, s1, s38 // (op 2218) most significant: sum
s_add_u32 s1, s1, s39 // (op 2218) most significant: sum


s_lshl_b64 s[38:39], s[0:1], 1 // (op 2218) 

s_mov_b32 s40, s6 // (op 2031) reg expression
s_mov_b32 s41, s7 // (op 2031) reg expression
s_mov_b32 s42, s18 // (op 2031) reg expression
s_mov_b32 s43, 131072 // (op 2031) call()

s_mov_b32 s44, s10 // (op 2218) reg expression

s_mov_b32 s45, s11 // (op 2218) reg expression
s_mov_b32 s46, s38 // (op 2218) reg expression
s_mov_b32 s47, 131072 // (op 2218) call()


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
v_lshrrev_b32 v12, 3, v1 // (op 1796) 
v_lshl_add_u32 v13, s16, 5, v12 // (op 1796) 

v_and_b32 v12, -8, v1 // (op 1796) 
v_sub_u32 v14, v1, v12 // (op 1796) 

v_lshlrev_b32 v12, 4, v14 // (op 1796) 

v_mov_b32 v15, 0 // (op 1796) convert
v_mov_b32 v14, v12 // (op 1796) convert

v_mul_lo_u32 v17, v13, s13 // (op 1796) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 1796) most significant: high of low * low
v_mul_lo_u32 v16, v13, s12 // (op 1796) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 1796) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 1796) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 1796) most significant half




s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 1798) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 1798) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 1798) least significant
s_mul_hi_u32 s7, s12, s22 // (op 1798) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1798) most significant: sum
s_add_u32 s1, s1, s7 // (op 1798) most significant: sum


s_mov_b32 s48, s20 // (op 1798) reg expression
s_mov_b32 s49, s21 // (op 1798) reg expression
s_mov_b32 s50, s0 // (op 1798) reg expression
s_mov_b32 s51, 131072 // (op 1798) call()

v_lshrrev_b32 v12, 3, v1 // (op 1815) 
v_lshl_add_u32 v13, s16, 5, v12 // (op 1815) 

v_and_b32 v12, -8, v1 // (op 1815) 
v_sub_u32 v14, v1, v12 // (op 1815) 

v_lshlrev_b32 v12, 4, v14 // (op 1815) 

v_mov_b32 v15, 0 // (op 1815) convert
v_mov_b32 v14, v12 // (op 1815) convert

v_mul_lo_u32 v19, v13, s13 // (op 1815) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 1815) most significant: high of low * low
v_mul_lo_u32 v18, v13, s12 // (op 1815) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 1815) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v14 // (op 1815) least significant half
v_addc_co_u32 v19, s[0:1], v19, v15, s[0:1] // (op 1815) most significant half



s_mul_i32 s1, s12, s23 // (op 1817) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 1817) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 1817) least significant
s_mul_hi_u32 s7, s12, s22 // (op 1817) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1817) most significant: sum
s_add_u32 s1, s1, s7 // (op 1817) most significant: sum


s_mov_b32 s48, s20 // (op 1817) reg expression
s_mov_b32 s49, s21 // (op 1817) reg expression
s_mov_b32 s50, s0 // (op 1817) reg expression
s_mov_b32 s51, 131072 // (op 1817) call()

v_and_b32 v12, -8, v1 // (op 1860) 
v_sub_u32 v13, v1, v12 // (op 1860) 

v_lshlrev_b32 v12, 4, v13 // (op 1860) 

v_mov_b32 v15, 0 // (op 1860) convert
v_mov_b32 v14, v12 // (op 1860) convert

v_lshrrev_b32 v12, 3, v1 // (op 1860) 
v_lshl_add_u32 v13, s17, 7, v12 // (op 1860) 

v_mul_lo_u32 v21, v13, s25 // (op 1860) most significant: low of low * high
v_mul_hi_u32 v12, v13, s24 // (op 1860) most significant: high of low * low
v_mul_lo_u32 v20, v13, s24 // (op 1860) least significant: low of low * low
v_add_u32 v21, v21, v12 // (op 1860) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v20 // (op 1860) least significant half
v_addc_co_u32 v13, s[0:1], v15, v21, s[0:1] // (op 1860) most significant half



s_mul_i32 s1, s24, s9 // (op 1862) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 1862) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 1862) least significant
s_mul_hi_u32 s7, s24, s8 // (op 1862) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1862) most significant: sum
s_add_u32 s1, s1, s7 // (op 1862) most significant: sum


s_mov_b32 s52, s26 // (op 1862) reg expression
s_mov_b32 s53, s27 // (op 1862) reg expression
s_mov_b32 s54, s0 // (op 1862) reg expression
s_mov_b32 s55, 131072 // (op 1862) call()

v_and_b32 v14, -8, v1 // (op 1879) 
v_sub_u32 v15, v1, v14 // (op 1879) 

v_lshlrev_b32 v14, 4, v15 // (op 1879) 

v_mov_b32 v21, 0 // (op 1879) convert
v_mov_b32 v20, v14 // (op 1879) convert

v_lshrrev_b32 v14, 3, v1 // (op 1879) 
v_lshl_add_u32 v15, s17, 7, v14 // (op 1879) 

v_mul_lo_u32 v23, v15, s25 // (op 1879) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 1879) most significant: high of low * low
v_mul_lo_u32 v22, v15, s24 // (op 1879) least significant: low of low * low
v_add_u32 v23, v23, v14 // (op 1879) most significant: sum


v_add_co_u32 v14, s[0:1], v20, v22 // (op 1879) least significant half
v_addc_co_u32 v15, s[0:1], v21, v23, s[0:1] // (op 1879) most significant half



s_mul_i32 s1, s24, s9 // (op 1881) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 1881) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 1881) least significant
s_mul_hi_u32 s7, s24, s8 // (op 1881) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1881) most significant: sum
s_add_u32 s1, s1, s7 // (op 1881) most significant: sum


s_mov_b32 s52, s26 // (op 1881) reg expression
s_mov_b32 s53, s27 // (op 1881) reg expression
s_mov_b32 s54, s0 // (op 1881) reg expression
s_mov_b32 s55, 131072 // (op 1881) call()

v_lshrrev_b32 v20, 7, v1 // (op 1924) 
v_and_b32 v21, 15, v1 // (op 1924) 
v_lshl_add_u32 v22, v20, 4, v21 // (op 1924) 


v_lshl_add_u32 v20, s16, 5, v22 // (op 1924) 

v_and_b32 v21, 63, v1 // (op 1924) 
v_lshrrev_b32 v22, 4, v21 // (op 1924) 

v_mul_lo_u32 v21, v20, s28 // (op 1924) 
v_add_u32 v21, v21, v22 // (op 1924) 


s_mul_i32 s1, s28, s23 // (op 1926) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 1926) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 1926) least significant
s_mul_hi_u32 s7, s28, s22 // (op 1926) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1926) most significant: sum
s_add_u32 s1, s1, s7 // (op 1926) most significant: sum


s_mov_b32 s56, s30 // (op 1926) reg expression
s_mov_b32 s57, s31 // (op 1926) reg expression
s_mov_b32 s58, s0 // (op 1926) reg expression
s_mov_b32 s59, 131072 // (op 1926) call()

v_lshrrev_b32 v20, 7, v1 // (op 1943) 
v_and_b32 v22, 15, v1 // (op 1943) 
v_lshl_add_u32 v23, v20, 4, v22 // (op 1943) 


v_lshl_add_u32 v20, s16, 5, v23 // (op 1943) 

v_and_b32 v22, 63, v1 // (op 1943) 
v_lshrrev_b32 v23, 4, v22 // (op 1943) 

v_mul_lo_u32 v22, v20, s28 // (op 1943) 
v_add_u32 v22, v22, v23 // (op 1943) 


s_mul_i32 s1, s28, s23 // (op 1945) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 1945) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 1945) least significant
s_mul_hi_u32 s7, s28, s22 // (op 1945) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1945) most significant: sum
s_add_u32 s1, s1, s7 // (op 1945) most significant: sum


s_mov_b32 s56, s30 // (op 1945) reg expression
s_mov_b32 s57, s31 // (op 1945) reg expression
s_mov_b32 s58, s0 // (op 1945) reg expression
s_mov_b32 s59, 131072 // (op 1945) call()

v_and_b32 v20, -4, v1 // (op 1988) 
v_sub_u32 v23, v1, v20 // (op 1988) 

v_lshrrev_b32 v20, 2, v1 // (op 1988) 
v_lshlrev_b32 v24, 1, v20 // (op 1988) 

v_lshl_add_u32 v20, s17, 7, v24 // (op 1988) 

v_mul_lo_u32 v24, v20, s32 // (op 1988) 

v_add_u32 v20, v23, v24 // (op 1988) 


s_mul_i32 s1, s32, s9 // (op 1990) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 1990) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 1990) least significant
s_mul_hi_u32 s7, s32, s8 // (op 1990) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1990) most significant: sum
s_add_u32 s1, s1, s7 // (op 1990) most significant: sum


s_mov_b32 s60, s34 // (op 1990) reg expression
s_mov_b32 s61, s35 // (op 1990) reg expression
s_mov_b32 s62, s0 // (op 1990) reg expression
s_mov_b32 s63, 131072 // (op 1990) call()

v_and_b32 v23, -4, v1 // (op 2007) 
v_sub_u32 v24, v1, v23 // (op 2007) 

v_lshrrev_b32 v23, 2, v1 // (op 2007) 
v_lshlrev_b32 v25, 1, v23 // (op 2007) 

v_lshl_add_u32 v23, s17, 7, v25 // (op 2007) 

v_mul_lo_u32 v25, v23, s32 // (op 2007) 

v_add_u32 v23, v24, v25 // (op 2007) 


s_mul_i32 s1, s32, s9 // (op 2009) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 2009) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 2009) least significant
s_mul_hi_u32 s7, s32, s8 // (op 2009) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2009) most significant: sum
s_add_u32 s1, s1, s7 // (op 2009) most significant: sum


s_mov_b32 s60, s34 // (op 2009) reg expression
s_mov_b32 s61, s35 // (op 2009) reg expression
s_mov_b32 s62, s0 // (op 2009) reg expression
s_mov_b32 s63, 131072 // (op 2009) call()




buffer_load_ubyte v24, v21, s[56:59], 0 offen offset: 0 // (op 1560) Load value
v_and_b32 v25, -8, v1 // (op 2125) 
v_sub_u32 v26, v1, v25 // (op 2125) 

v_lshlrev_b32 v25, 4, v26 // (op 2125) 

v_lshrrev_b32 v26, 3, v1 // (op 2125) 
v_lshl_add_u32 v27, v26, 7, v25 // (op 2125) 


s_nop 0
v_readfirstlane_b32 s0, v27 // (op 2125) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 1742) 
v_mov_b64 v[26:27], s[6:7] // (op 1742) convert


v_mov_b64 v[28:29], v[12:13] // (op 1742) 
s_mov_b32 m0, s0 // (op 1742) reg expression
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 1742) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 1742) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 1742) most significant half

s_add_u32 m0, m0, 4096 // (op 1742) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 1742) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 1742) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 1742) most significant half

s_add_u32 m0, m0, 4096 // (op 1742) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 1742) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 1742) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 1742) most significant half

s_add_u32 m0, m0, 4096 // (op 1742) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 1742) Load value direct to lds




v_mov_b64 v[26:27], s[32:33] // (op 1567) convert

v_mov_b32 v25, v20 // (op 1567) 
buffer_load_ubyte v28, v25, s[60:63], 0 offen offset: 0 // (op 1567) Load value
v_add_u32 v25, v25, v26 // (op 1567) 
buffer_load_ubyte v29, v25, s[60:63], 0 offen offset: 0 // (op 1567) Load value


v_and_b32 v25, -8, v1 // (op 2073) 
v_sub_u32 v26, v1, v25 // (op 2073) 

v_lshlrev_b32 v25, 4, v26 // (op 2073) 

v_lshrrev_b32 v26, 3, v1 // (op 2073) 
v_lshl_add_u32 v27, v26, 7, v25 // (op 2073) 


s_nop 0
v_readfirstlane_b32 s0, v27 // (op 2073) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s1, 16384, s0 // (op 1747) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 1747) reg expression
buffer_load_dwordx4 v16, s[48:51], 0 offen offset: 0 lds // (op 1747) Load value direct to lds





buffer_load_ubyte v25, v21, s[56:59], 0 offen offset:4 // (op 1574) Load value

v_and_b32 v21, -8, v1 // (op 2284) 
v_sub_u32 v26, v1, v21 // (op 2284) 

v_lshlrev_b32 v21, 4, v26 // (op 2284) 

v_lshrrev_b32 v26, 3, v1 // (op 2284) 
v_lshl_add_u32 v27, v26, 7, v21 // (op 2284) 


s_nop 0
v_readfirstlane_b32 s0, v27 // (op 2284) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 1752) 
v_mov_b64 v[26:27], s[6:7] // (op 1752) convert


v_mov_b64 v[30:31], v[12:13] // (op 1752) 
s_add_u32 s1, 20480, s0 // (op 1752) 
s_mov_b32 m0, s1 // (op 1752) reg expression
s_mov_b32 s6, 128 // (op 1752) call()
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 1752) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v26 // (op 1752) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 1752) most significant half

s_add_u32 m0, m0, 4096 // (op 1752) 
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 1752) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v26 // (op 1752) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 1752) most significant half

s_add_u32 m0, m0, 4096 // (op 1752) 
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 1752) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v26 // (op 1752) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 1752) most significant half

s_add_u32 m0, m0, 4096 // (op 1752) 
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 1752) Load value direct to lds







v_mov_b64 v[12:13], s[32:33] // (op 1581) convert

v_mov_b32 v21, v20 // (op 1581) 
buffer_load_ubyte v26, v21, s[60:63], 0 offen offset:4 // (op 1581) Load value
v_add_u32 v21, v21, v12 // (op 1581) 
buffer_load_ubyte v27, v21, s[60:63], 0 offen offset:4 // (op 1581) Load value



v_and_b32 v12, -8, v1 // (op 2245) 
v_sub_u32 v13, v1, v12 // (op 2245) 

v_lshlrev_b32 v12, 4, v13 // (op 2245) 

v_lshrrev_b32 v13, 3, v1 // (op 2245) 
v_lshl_add_u32 v20, v13, 7, v12 // (op 2245) 


s_nop 0
v_readfirstlane_b32 s0, v20 // (op 2245) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s1, 36864, s0 // (op 1754) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 1754) reg expression
s_mov_b32 s6, 128 // (op 1754) call()
buffer_load_dwordx4 v16, s[48:51], s6 offen offset: 0 lds // (op 1754) Load value direct to lds




v_and_b32 v12, -4, v1 // (op 2183) 
v_sub_u32 v13, v1, v12 // (op 2183) 

v_lshrrev_b32 v12, 2, v1 // (op 2183) 
v_lshl_add_u32 v16, v12, 3, v13 // (op 2183) 





s_waitcnt vmcnt(10)
ds_write_b8 v16, v28 offset:40960 // (op 1587) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v16, v29 offset:40964 // (op 1587) Store local data 

s_barrier  // (op 2393) 



ds_read_b128 v[30:33], v5 // (op 1605) Load local data 
ds_read_b128 v[34:37], v5 offset:64 // (op 1605) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 2409) 



ds_read_u8 v12, v0 offset:40960 // (op 1625) Load local data 



ds_read_b128 v[38:41], v5 offset:4096 // (op 1610) Load local data 
ds_read_b128 v[42:45], v5 offset:4160 // (op 1610) Load local data 



ds_read_u8 v13, v0 offset:41088 // (op 1630) Load local data 



ds_read_b128 v[46:49], v5 offset:8192 // (op 1615) Load local data 
ds_read_b128 v[50:53], v5 offset:8256 // (op 1615) Load local data 



ds_read_u8 v16, v0 offset:41216 // (op 1635) Load local data 



ds_read_b128 v[54:57], v5 offset:12288 // (op 1620) Load local data 
ds_read_b128 v[58:61], v5 offset:12352 // (op 1620) Load local data 



ds_read_u8 v17, v0 offset:41344 // (op 1640) Load local data 
s_waitcnt vmcnt(8)
s_barrier  // (op 2377) 



ds_read_b128 v[62:65], v3 offset:16384 // (op 1601) Load local data 
ds_read_b128 v[66:69], v3 offset:16448 // (op 1601) Load local data 
v_and_b32 v20, -8, v1 // (op 2045) 
v_sub_u32 v21, v1, v20 // (op 2045) 

v_lshlrev_b32 v20, 4, v21 // (op 2045) 

v_lshrrev_b32 v21, 3, v1 // (op 2045) 
v_lshl_add_u32 v70, v21, 7, v20 // (op 2045) 


s_nop 0
v_readfirstlane_b32 s0, v70 // (op 2045) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v20, -8, v1 // (op 2097) 
v_sub_u32 v21, v1, v20 // (op 2097) 

v_lshlrev_b32 v20, 4, v21 // (op 2097) 

v_lshrrev_b32 v21, 3, v1 // (op 2097) 
v_lshl_add_u32 v70, v21, 7, v20 // (op 2097) 


s_nop 0
v_readfirstlane_b32 s1, v70 // (op 2097) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v20, -4, v1 // (op 2154) 
v_sub_u32 v21, v1, v20 // (op 2154) 

v_lshrrev_b32 v20, 2, v1 // (op 2154) 
v_lshl_add_u32 v70, v20, 3, v21 // (op 2154) 


v_and_b32 v20, -8, v1 // (op 2230) 
v_sub_u32 v21, v1, v20 // (op 2230) 

v_lshlrev_b32 v20, 4, v21 // (op 2230) 

v_lshrrev_b32 v21, 3, v1 // (op 2230) 
v_lshl_add_u32 v71, v21, 7, v20 // (op 2230) 


s_nop 0
v_readfirstlane_b32 s6, v71 // (op 2230) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v20, -8, v1 // (op 2269) 
v_sub_u32 v21, v1, v20 // (op 2269) 

v_lshlrev_b32 v20, 4, v21 // (op 2269) 

v_lshrrev_b32 v21, 3, v1 // (op 2269) 
v_lshl_add_u32 v71, v21, 7, v20 // (op 2269) 


s_nop 0
v_readfirstlane_b32 s7, v71 // (op 2269) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v20, -4, v1 // (op 2313) 
v_sub_u32 v21, v1, v20 // (op 2313) 

v_lshrrev_b32 v20, 2, v1 // (op 2313) 
v_lshl_add_u32 v71, v20, 3, v21 // (op 2313) 


s_mov_b32 s10, 0 // (op 624) call()
s_ashr_i64 s[14:15], s[36:37], 7 // (op 623) 
s_mov_b32 s11, s14 // (op 623) convert

s_and_b32 s14, s11, 4294967294 // (op 623) 

s_cmp_lt_u32 s10, s14 // (op 623) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopBottom_KLoop_623 // (op 623) Condition: Top (jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopBottom_KLoop_623 if false)
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopTop_KLoop_623:
s_add_u32 s10, s10, 2 // (op 625) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:69], v[30:37], a[0:3], v24, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 84) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[62:69], v[38:45], a[4:7], v24, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 347) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[62:69], v[46:53], a[8:11], v24, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 503) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[62:69], v[54:61], a[12:15], v24, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 659) 


v_mov_b32 v20, v22 // (op 77) 

buffer_load_ubyte v24, v20, s[56:59], 0 offen offset:8 // (op 77) Load value
s_barrier  // (op 2396) 

s_lshl_b64 s[14:15], s[24:25], 5 // (op 1714) 
v_mov_b64 v[72:73], s[14:15] // (op 1714) convert


v_mov_b64 v[74:75], v[14:15] // (op 1714) 
v_mov_b64 v[76:77], v[74:75] // (op 1714) 
s_mov_b32 m0, s1 // (op 1714) reg expression
s_mov_b32 s11, 256 // (op 1714) call()
buffer_load_dwordx4 v76, s[52:55], s11 offen offset: 0 lds // (op 1714) Load value direct to lds
v_add_co_u32 v76, s[14:15], v76, v72 // (op 1714) least significant half
v_addc_co_u32 v77, s[14:15], 0, v77, s[14:15] // (op 1714) most significant half

s_add_u32 m0, m0, 4096 // (op 1714) 
buffer_load_dwordx4 v76, s[52:55], s11 offen offset: 0 lds // (op 1714) Load value direct to lds
v_add_co_u32 v76, s[14:15], v76, v72 // (op 1714) least significant half
v_addc_co_u32 v77, s[14:15], 0, v77, s[14:15] // (op 1714) most significant half

s_add_u32 m0, m0, 4096 // (op 1714) 
buffer_load_dwordx4 v76, s[52:55], s11 offen offset: 0 lds // (op 1714) Load value direct to lds
v_add_co_u32 v76, s[14:15], v76, v72 // (op 1714) least significant half
v_addc_co_u32 v77, s[14:15], 0, v77, s[14:15] // (op 1714) most significant half

s_add_u32 m0, m0, 4096 // (op 1714) 
buffer_load_dwordx4 v76, s[52:55], s11 offen offset: 0 lds // (op 1714) Load value direct to lds




v_mov_b64 v[72:73], s[32:33] // (op 21) convert

v_mov_b32 v21, v23 // (op 21) 
v_mov_b32 v76, v21 // (op 21) 
buffer_load_ubyte v28, v76, s[60:63], 0 offen offset:8 // (op 21) Load value
v_add_u32 v76, v76, v72 // (op 21) 
buffer_load_ubyte v29, v76, s[60:63], 0 offen offset:8 // (op 21) Load value


s_barrier  // (op 2380) 


v_mov_b64 v[72:73], v[18:19] // (op 1709) 
s_add_u32 s11, 16384, s0 // (op 1709) 

s_mov_b32 m0, s11 // (op 1709) reg expression
s_mov_b32 s14, 256 // (op 1709) call()
buffer_load_dwordx4 v72, s[48:51], s14 offen offset: 0 lds // (op 1709) Load value direct to lds


s_barrier  // (op 2441) 



s_waitcnt vmcnt(10)
ds_write_b8 v71, v26 offset:41472 // (op 240) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v71, v27 offset:41476 // (op 240) Store local data 
s_barrier  // (op 2432) 



ds_read_b128 v[30:33], v8 offset:20480 // (op 233) Load local data 
ds_read_b128 v[34:37], v8 offset:20544 // (op 233) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 2438) 



ds_read_u8 v12, v7 offset:41472 // (op 237) Load local data 



ds_read_b128 v[38:41], v8 offset:24576 // (op 352) Load local data 
ds_read_b128 v[42:45], v8 offset:24640 // (op 352) Load local data 



ds_read_u8 v13, v7 offset:41600 // (op 355) Load local data 



ds_read_b128 v[46:49], v8 offset:28672 // (op 508) Load local data 
ds_read_b128 v[50:53], v8 offset:28736 // (op 508) Load local data 



ds_read_u8 v16, v7 offset:41728 // (op 511) Load local data 



ds_read_b128 v[54:57], v8 offset:32768 // (op 664) Load local data 
ds_read_b128 v[58:61], v8 offset:32832 // (op 664) Load local data 



ds_read_u8 v17, v7 offset:41856 // (op 667) Load local data 
s_barrier  // (op 2504) 
s_waitcnt vmcnt(8)
s_barrier  // (op 2426) 



ds_read_b128 v[62:65], v4 offset:36864 // (op 231) Load local data 
ds_read_b128 v[66:69], v4 offset:36928 // (op 231) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:69], v[30:37], a[0:3], v25, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 238) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[62:69], v[38:45], a[4:7], v25, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 348) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[62:69], v[46:53], a[8:11], v25, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 504) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[62:69], v[54:61], a[12:15], v25, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 660) 



buffer_load_ubyte v25, v20, s[56:59], 0 offen offset:12 // (op 235) Load value

s_barrier  // (op 2435) 
v_add_u32 v22, 8, v22 // (op 1935) 

s_lshl_b64 s[14:15], s[24:25], 5 // (op 1737) 
v_mov_b64 v[76:77], s[14:15] // (op 1737) convert


v_mov_b64 v[78:79], v[74:75] // (op 1737) 
s_add_u32 s11, 20480, s7 // (op 1737) 
s_mov_b32 m0, s11 // (op 1737) reg expression
s_mov_b32 s14, 384 // (op 1737) call()
buffer_load_dwordx4 v78, s[52:55], s14 offen offset: 0 lds // (op 1737) Load value direct to lds
v_add_co_u32 v78, s[18:19], v78, v76 // (op 1737) least significant half
v_addc_co_u32 v79, s[18:19], 0, v79, s[18:19] // (op 1737) most significant half

s_add_u32 m0, m0, 4096 // (op 1737) 
buffer_load_dwordx4 v78, s[52:55], s14 offen offset: 0 lds // (op 1737) Load value direct to lds
v_add_co_u32 v78, s[18:19], v78, v76 // (op 1737) least significant half
v_addc_co_u32 v79, s[18:19], 0, v79, s[18:19] // (op 1737) most significant half

s_add_u32 m0, m0, 4096 // (op 1737) 
buffer_load_dwordx4 v78, s[52:55], s14 offen offset: 0 lds // (op 1737) Load value direct to lds
v_add_co_u32 v78, s[18:19], v78, v76 // (op 1737) least significant half
v_addc_co_u32 v79, s[18:19], 0, v79, s[18:19] // (op 1737) most significant half

s_add_u32 m0, m0, 4096 // (op 1737) 
buffer_load_dwordx4 v78, s[52:55], s14 offen offset: 0 lds // (op 1737) Load value direct to lds






v_add_co_u32 v14, vcc, 256, v14 // (op 1871) least significant half
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 1871) most significant half
v_mov_b64 v[74:75], s[32:33] // (op 242) convert

v_mov_b32 v20, v21 // (op 242) 
buffer_load_ubyte v26, v20, s[60:63], 0 offen offset:12 // (op 242) Load value
v_add_u32 v20, v20, v74 // (op 242) 
buffer_load_ubyte v27, v20, s[60:63], 0 offen offset:12 // (op 242) Load value



s_barrier  // (op 2429) 
v_add_u32 v23, 8, v23 // (op 1999) 


s_add_u32 s11, 36864, s6 // (op 1732) 

s_mov_b32 m0, s11 // (op 1732) reg expression
s_mov_b32 s14, 384 // (op 1732) call()
buffer_load_dwordx4 v72, s[48:51], s14 offen offset: 0 lds // (op 1732) Load value direct to lds



s_barrier  // (op 2412) 
v_add_co_u32 v18, vcc, 256, v18 // (op 1807) least significant half



v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 1807) most significant half
s_waitcnt vmcnt(10)
ds_write_b8 v70, v28 offset:40960 // (op 26) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v70, v29 offset:40964 // (op 26) Store local data 
s_barrier  // (op 2402) 



ds_read_b128 v[30:33], v5 // (op 72) Load local data 
ds_read_b128 v[34:37], v5 offset:64 // (op 72) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 2419) 



ds_read_u8 v12, v0 offset:40960 // (op 81) Load local data 



ds_read_b128 v[38:41], v5 offset:4096 // (op 343) Load local data 
ds_read_b128 v[42:45], v5 offset:4160 // (op 343) Load local data 



ds_read_u8 v13, v0 offset:41088 // (op 346) Load local data 



ds_read_b128 v[46:49], v5 offset:8192 // (op 499) Load local data 
ds_read_b128 v[50:53], v5 offset:8256 // (op 499) Load local data 



ds_read_u8 v16, v0 offset:41216 // (op 502) Load local data 



ds_read_b128 v[54:57], v5 offset:12288 // (op 655) Load local data 
ds_read_b128 v[58:61], v5 offset:12352 // (op 655) Load local data 



ds_read_u8 v17, v0 offset:41344 // (op 658) Load local data 
s_waitcnt vmcnt(8)
s_barrier  // (op 2386) 



ds_read_b128 v[62:65], v3 offset:16384 // (op 67) Load local data 
ds_read_b128 v[66:69], v3 offset:16448 // (op 67) Load local data 
s_ashr_i64 s[14:15], s[36:37], 7 // (op 623) 
s_mov_b32 s11, s14 // (op 623) convert

s_and_b32 s14, s11, 4294967294 // (op 623) 

s_cmp_lt_u32 s10, s14 // (op 623) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopTop_KLoop_623 // (op 623) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopTop_KLoop_623 if true)
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopBottom_KLoop_623:













s_barrier  // (op 2488) 
v_lshrrev_b32 v14, 3, v1 // (op 1767) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1767) 
v_lshl_add_u32 v15, s16, 5, v14 // (op 1767) 

s_lshr_b32 s6, s0, 1 // (op 1767) 

v_and_b32 v14, -8, v1 // (op 1767) 
v_sub_u32 v18, v1, v14 // (op 1767) 

v_lshlrev_b32 v14, 4, v18 // (op 1767) 

v_lshl_add_u32 v18, s6, 8, v14 // (op 1767) 


v_mov_b32 v29, 0 // (op 1767) convert
v_mov_b32 v28, v18 // (op 1767) convert

v_mul_lo_u32 v19, v15, s13 // (op 1767) most significant: low of low * high
v_mul_hi_u32 v14, v15, s12 // (op 1767) most significant: high of low * low
v_mul_lo_u32 v18, v15, s12 // (op 1767) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 1767) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v28 // (op 1767) least significant half
v_addc_co_u32 v19, s[0:1], v19, v29, s[0:1] // (op 1767) most significant half



s_mul_i32 s1, s12, s23 // (op 1769) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 1769) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 1769) least significant
s_mul_hi_u32 s7, s12, s22 // (op 1769) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1769) most significant: sum
s_add_u32 s1, s1, s7 // (op 1769) most significant: sum


s_mov_b32 s48, s20 // (op 1769) reg expression
s_mov_b32 s49, s21 // (op 1769) reg expression
s_mov_b32 s50, s0 // (op 1769) reg expression
s_mov_b32 s51, 131072 // (op 1769) call()

v_and_b32 v14, -8, v1 // (op 1837) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1837) 
v_sub_u32 v15, v1, v14 // (op 1837) 

s_lshr_b32 s6, s0, 1 // (op 1837) 

v_lshlrev_b32 v14, 4, v15 // (op 1837) 

v_lshl_add_u32 v15, s6, 8, v14 // (op 1837) 


v_mov_b32 v29, 0 // (op 1837) convert
v_mov_b32 v28, v15 // (op 1837) convert

v_lshrrev_b32 v14, 3, v1 // (op 1837) 
v_lshl_add_u32 v15, s17, 7, v14 // (op 1837) 

v_mul_lo_u32 v21, v15, s25 // (op 1837) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 1837) most significant: high of low * low
v_mul_lo_u32 v20, v15, s24 // (op 1837) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 1837) most significant: sum


v_add_co_u32 v14, s[0:1], v28, v20 // (op 1837) least significant half
v_addc_co_u32 v15, s[0:1], v29, v21, s[0:1] // (op 1837) most significant half



s_mul_i32 s1, s24, s9 // (op 1839) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 1839) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 1839) least significant
s_mul_hi_u32 s7, s24, s8 // (op 1839) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1839) most significant: sum
s_add_u32 s1, s1, s7 // (op 1839) most significant: sum


s_mov_b32 s52, s26 // (op 1839) reg expression
s_mov_b32 s53, s27 // (op 1839) reg expression
s_mov_b32 s54, s0 // (op 1839) reg expression
s_mov_b32 s55, 131072 // (op 1839) call()

v_lshrrev_b32 v20, 7, v1 // (op 1901) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1901) 
v_and_b32 v21, 15, v1 // (op 1901) 
s_lshr_b32 s6, s0, 1 // (op 1901) 

v_lshl_add_u32 v22, v20, 4, v21 // (op 1901) 


v_lshl_add_u32 v20, s16, 5, v22 // (op 1901) 

v_and_b32 v21, 63, v1 // (op 1901) 
v_lshrrev_b32 v22, 4, v21 // (op 1901) 

v_lshl_add_u32 v21, s6, 3, v22 // (op 1901) 


v_mul_lo_u32 v22, v20, s28 // (op 1901) 
v_add_u32 v22, v22, v21 // (op 1901) 


s_mul_i32 s1, s28, s23 // (op 1903) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 1903) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 1903) least significant
s_mul_hi_u32 s7, s28, s22 // (op 1903) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1903) most significant: sum
s_add_u32 s1, s1, s7 // (op 1903) most significant: sum


s_mov_b32 s56, s30 // (op 1903) reg expression
s_mov_b32 s57, s31 // (op 1903) reg expression
s_mov_b32 s58, s0 // (op 1903) reg expression
s_mov_b32 s59, 131072 // (op 1903) call()

v_and_b32 v20, -4, v1 // (op 1965) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1965) 
v_sub_u32 v21, v1, v20 // (op 1965) 

s_lshr_b32 s6, s0, 1 // (op 1965) 

v_lshrrev_b32 v20, 2, v1 // (op 1965) 
v_lshl_add_u32 v23, s6, 3, v21 // (op 1965) 


v_lshlrev_b32 v21, 1, v20 // (op 1965) 

v_lshl_add_u32 v20, s17, 7, v21 // (op 1965) 

v_mul_lo_u32 v21, v20, s32 // (op 1965) 

v_add_u32 v20, v23, v21 // (op 1965) 


s_mul_i32 s1, s32, s9 // (op 1967) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 1967) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 1967) least significant
s_mul_hi_u32 s7, s32, s8 // (op 1967) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1967) most significant: sum
s_add_u32 s1, s1, s7 // (op 1967) most significant: sum


s_mov_b32 s60, s34 // (op 1967) reg expression
s_mov_b32 s61, s35 // (op 1967) reg expression
s_mov_b32 s62, s0 // (op 1967) reg expression
s_mov_b32 s63, 131072 // (op 1967) call()

v_and_b32 v21, -8, v1 // (op 2058) 
v_sub_u32 v23, v1, v21 // (op 2058) 

v_lshlrev_b32 v21, 4, v23 // (op 2058) 

v_lshrrev_b32 v23, 3, v1 // (op 2058) 
v_lshl_add_u32 v24, v23, 7, v21 // (op 2058) 


s_nop 0
v_readfirstlane_b32 s0, v24 // (op 2058) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v21, -8, v1 // (op 2110) 
v_sub_u32 v23, v1, v21 // (op 2110) 

v_lshlrev_b32 v21, 4, v23 // (op 2110) 

v_lshrrev_b32 v23, 3, v1 // (op 2110) 
v_lshl_add_u32 v24, v23, 7, v21 // (op 2110) 


s_nop 0
v_readfirstlane_b32 s1, v24 // (op 2110) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v21, -4, v1 // (op 2167) 
v_sub_u32 v23, v1, v21 // (op 2167) 

v_lshrrev_b32 v21, 2, v1 // (op 2167) 
v_lshl_add_u32 v24, v21, 3, v23 // (op 2167) 


s_ashr_i64 s[6:7], s[36:37], 7 // (op 629) 
s_and_b32 s10, s6, 4294967294 // (op 629) 

s_ashr_i64 s[6:7], s[36:37], 7 // (op 628) 
s_mov_b32 s11, s6 // (op 628) convert

s_cmp_lt_u32 s10, s11 // (op 628) 












s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopBottom_KLoopTail_628 // (op 628) Condition: Top (jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopBottom_KLoopTail_628 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopTop_KLoopTail_628:
s_barrier  // (op 2383) 
s_add_u32 s10, s10, 1 // (op 630) 


v_mov_b64 v[12:13], v[18:19] // (op 1719) 
s_add_u32 s11, 16384, s0 // (op 1719) 

s_mov_b32 m0, s11 // (op 1719) reg expression
buffer_load_dwordx4 v12, s[48:51], 0 offen offset: 0 lds // (op 1719) Load value direct to lds


v_add_co_u32 v18, vcc, 128, v18 // (op 1757) least significant half
v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 1757) most significant half
s_barrier  // (op 2399) 

s_lshl_b64 s[6:7], s[24:25], 5 // (op 1725) 
v_mov_b64 v[12:13], s[6:7] // (op 1725) convert


v_mov_b64 v[16:17], v[14:15] // (op 1725) 
v_mov_b64 v[26:27], v[16:17] // (op 1725) 
s_mov_b32 m0, s1 // (op 1725) reg expression
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 1725) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v12 // (op 1725) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 1725) most significant half

s_add_u32 m0, m0, 4096 // (op 1725) 
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 1725) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v12 // (op 1725) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 1725) most significant half

s_add_u32 m0, m0, 4096 // (op 1725) 
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 1725) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v12 // (op 1725) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 1725) most significant half

s_add_u32 m0, m0, 4096 // (op 1725) 
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 1725) Load value direct to lds



v_add_co_u32 v14, vcc, 128, v14 // (op 1829) least significant half
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 1829) most significant half

v_mov_b64 v[12:13], s[32:33] // (op 184) convert

v_mov_b32 v21, v20 // (op 184) 
v_mov_b32 v23, v21 // (op 184) 
buffer_load_ubyte v16, v23, s[60:63], 0 offen offset: 0 // (op 184) Load value
v_add_u32 v23, v23, v12 // (op 184) 
buffer_load_ubyte v17, v23, s[60:63], 0 offen offset: 0 // (op 184) Load value



v_add_u32 v20, 4, v20 // (op 1957) 
s_barrier  // (op 2415) 
s_waitcnt vmcnt(6)
s_barrier  // (op 2389) 






ds_read_b128 v[26:29], v3 offset:16384 // (op 173) Load local data 
ds_read_b128 v[30:33], v3 offset:16448 // (op 173) Load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v24, v16 offset:40960 // (op 182) Store local data 
s_barrier  // (op 2405) 
s_waitcnt vmcnt(0)
ds_write_b8 v24, v17 offset:40964 // (op 182) Store local data 




ds_read_b128 v[34:37], v5 // (op 175) Load local data 
ds_read_b128 v[38:41], v5 offset:64 // (op 175) Load local data 


v_mov_b32 v21, v22 // (op 177) 

buffer_load_ubyte v23, v21, s[56:59], 0 offen offset: 0 // (op 177) Load value

v_add_u32 v22, 4, v22 // (op 1893) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 2422) 



ds_read_u8 v21, v0 offset:40960 // (op 179) Load local data 



ds_read_b128 v[42:45], v5 offset:4096 // (op 324) Load local data 
ds_read_b128 v[46:49], v5 offset:4160 // (op 324) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:33], v[34:41], a[0:3], v23, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 180) 





ds_read_u8 v21, v0 offset:41088 // (op 327) Load local data 



ds_read_b128 v[34:37], v5 offset:8192 // (op 480) Load local data 
ds_read_b128 v[38:41], v5 offset:8256 // (op 480) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[26:33], v[42:49], a[4:7], v23, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 328) 





ds_read_u8 v21, v0 offset:41216 // (op 483) Load local data 



ds_read_b128 v[42:45], v5 offset:12288 // (op 636) Load local data 
ds_read_b128 v[46:49], v5 offset:12352 // (op 636) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[26:33], v[34:41], a[8:11], v23, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 484) 





ds_read_u8 v21, v0 offset:41344 // (op 639) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[26:33], v[42:49], a[12:15], v23, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 640) 




s_ashr_i64 s[6:7], s[36:37], 7 // (op 628) 
s_mov_b32 s11, s6 // (op 628) convert

s_cmp_lt_u32 s10, s11 // (op 628) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopTop_KLoopTail_628 // (op 628) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopTop_KLoopTail_628 if true)
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopBottom_KLoopTail_628:






















s_barrier  // (op 2492) 







v_cmp_eq_f32 vcc, 0.00000, v10 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ConditionalFalse_0_beta_125
v_accvgpr_read v12, a0 // (op 46) 
v_accvgpr_read v13, a1 // (op 46) 
v_accvgpr_read v14, a2 // (op 46) 
v_accvgpr_read v15, a3 // (op 46) 
v_mul_f32 v16, v11, v12 // (op 46) 
v_mul_f32 v17, v11, v13 // (op 46) 
v_mul_f32 v18, v11, v14 // (op 46) 
v_mul_f32 v19, v11, v15 // (op 46) 

v_cvt_f16_f32 v12, v16 // (op 52) 
v_cvt_f16_f32 v13, v17 // (op 52) 
v_cvt_f16_f32 v14, v18 // (op 52) 
v_cvt_f16_f32 v15, v19 // (op 52) 


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

v_cvt_f16_f32 v12, v16 // (op 866) 
v_cvt_f16_f32 v13, v17 // (op 866) 
v_cvt_f16_f32 v14, v18 // (op 866) 
v_cvt_f16_f32 v15, v19 // (op 866) 


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

v_cvt_f16_f32 v12, v16 // (op 880) 
v_cvt_f16_f32 v13, v17 // (op 880) 
v_cvt_f16_f32 v14, v18 // (op 880) 
v_cvt_f16_f32 v15, v19 // (op 880) 


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

v_cvt_f16_f32 v12, v16 // (op 894) 
v_cvt_f16_f32 v13, v17 // (op 894) 
v_cvt_f16_f32 v14, v18 // (op 894) 
v_cvt_f16_f32 v15, v19 // (op 894) 


v_pack_b32_f16 v4, v12, v13 // (op 895) 
v_pack_b32_f16 v5, v14, v15 // (op 895) 

s_lshl_b32 s0, s4, 6 // (op 895) 
s_mul_i32 s1, 3, s0 // (op 895) 

v_add_u32 v0, v9, s1 // (op 895) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 895) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ConditionalBottom_0_beta_125
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ConditionalFalse_0_beta_125:


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
v_cvt_f32_f16 v12, v4 // (op 139) 
v_lshrrev_b32 v13, 16, v4 // (op 139) 
v_cvt_f32_f16 v13, v13 // (op 139) 
v_mul_f32 v20, v10, v12 // (op 139) 
v_mul_f32 v21, v10, v13 // (op 139) 
v_cvt_f32_f16 v12, v5 // (op 139) 
v_lshrrev_b32 v13, 16, v5 // (op 139) 
v_cvt_f32_f16 v13, v13 // (op 139) 
v_mul_f32 v22, v10, v12 // (op 139) 
v_mul_f32 v23, v10, v13 // (op 139) 


v_add_f32 v12, v20, v16 // (op 135) 
v_add_f32 v13, v21, v17 // (op 135) 
v_add_f32 v14, v22, v18 // (op 135) 
v_add_f32 v15, v23, v19 // (op 135) 


v_cvt_f16_f32 v16, v12 // (op 136) 
v_cvt_f16_f32 v17, v13 // (op 136) 
v_cvt_f16_f32 v18, v14 // (op 136) 
v_cvt_f16_f32 v19, v15 // (op 136) 


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
v_cvt_f32_f16 v12, v4 // (op 930) 
v_lshrrev_b32 v13, 16, v4 // (op 930) 
v_cvt_f32_f16 v13, v13 // (op 930) 
v_mul_f32 v20, v10, v12 // (op 930) 
v_mul_f32 v21, v10, v13 // (op 930) 
v_cvt_f32_f16 v12, v5 // (op 930) 
v_lshrrev_b32 v13, 16, v5 // (op 930) 
v_cvt_f32_f16 v13, v13 // (op 930) 
v_mul_f32 v22, v10, v12 // (op 930) 
v_mul_f32 v23, v10, v13 // (op 930) 


v_add_f32 v12, v20, v16 // (op 926) 
v_add_f32 v13, v21, v17 // (op 926) 
v_add_f32 v14, v22, v18 // (op 926) 
v_add_f32 v15, v23, v19 // (op 926) 


v_cvt_f16_f32 v16, v12 // (op 927) 
v_cvt_f16_f32 v17, v13 // (op 927) 
v_cvt_f16_f32 v18, v14 // (op 927) 
v_cvt_f16_f32 v19, v15 // (op 927) 


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
v_cvt_f32_f16 v12, v4 // (op 944) 
v_lshrrev_b32 v13, 16, v4 // (op 944) 
v_cvt_f32_f16 v13, v13 // (op 944) 
v_mul_f32 v20, v10, v12 // (op 944) 
v_mul_f32 v21, v10, v13 // (op 944) 
v_cvt_f32_f16 v12, v5 // (op 944) 
v_lshrrev_b32 v13, 16, v5 // (op 944) 
v_cvt_f32_f16 v13, v13 // (op 944) 
v_mul_f32 v22, v10, v12 // (op 944) 
v_mul_f32 v23, v10, v13 // (op 944) 


v_add_f32 v12, v20, v16 // (op 940) 
v_add_f32 v13, v21, v17 // (op 940) 
v_add_f32 v14, v22, v18 // (op 940) 
v_add_f32 v15, v23, v19 // (op 940) 


v_cvt_f16_f32 v16, v12 // (op 941) 
v_cvt_f16_f32 v17, v13 // (op 941) 
v_cvt_f16_f32 v18, v14 // (op 941) 
v_cvt_f16_f32 v19, v15 // (op 941) 


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
v_cvt_f32_f16 v12, v4 // (op 958) 
v_lshrrev_b32 v13, 16, v4 // (op 958) 
v_cvt_f32_f16 v13, v13 // (op 958) 
v_mul_f32 v20, v10, v12 // (op 958) 
v_mul_f32 v21, v10, v13 // (op 958) 
v_cvt_f32_f16 v12, v5 // (op 958) 
v_lshrrev_b32 v13, 16, v5 // (op 958) 
v_cvt_f32_f16 v13, v13 // (op 958) 
v_mul_f32 v22, v10, v12 // (op 958) 
v_mul_f32 v23, v10, v13 // (op 958) 


v_add_f32 v12, v20, v16 // (op 954) 
v_add_f32 v13, v21, v17 // (op 954) 
v_add_f32 v14, v22, v18 // (op 954) 
v_add_f32 v15, v23, v19 // (op 954) 


v_cvt_f16_f32 v16, v12 // (op 955) 
v_cvt_f16_f32 v17, v13 // (op 955) 
v_cvt_f16_f32 v18, v14 // (op 955) 
v_cvt_f16_f32 v19, v15 // (op 955) 


v_pack_b32_f16 v4, v16, v17 // (op 956) 
v_pack_b32_f16 v5, v18, v19 // (op 956) 

s_lshl_b32 s0, s4, 6 // (op 956) 
s_mul_i32 s1, 3, s0 // (op 956) 

v_add_u32 v0, v9, s1 // (op 956) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 956) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ConditionalBottom_0_beta_125:












s_endpgm  // End of RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2
.LRR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_end:
.size RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2, .LRR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_end-RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2
  .amdhsa_next_free_vgpr 96
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 41984 // lds bytes
  .amdhsa_accum_offset 80
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
    DestDataType: h
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
  - .name: RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2
    .symbol: RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 41984
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 64
    .vgpr_count: 80
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
        .expression: {type: CommandArgument, size: 8, offset: 160, name: Tensor_10_pointer, variableType: {dataType: Half, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Half, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 208, name: Tensor_21_pointer, variableType: {dataType: Half, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Half, pointerType: PointerGlobal}
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

