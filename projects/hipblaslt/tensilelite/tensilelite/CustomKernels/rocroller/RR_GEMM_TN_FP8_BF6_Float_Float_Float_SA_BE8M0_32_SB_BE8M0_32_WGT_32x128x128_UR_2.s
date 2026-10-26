

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2
.p2align 8
.type RR_GEMM_TN_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2,@function
RR_GEMM_TN_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 2035) 
v_and_b32 v3, 15, v1 // (op 2035) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 2035) 


v_and_b32 v0, 48, v1 // (op 2035) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 2035) 


v_lshrrev_b32 v0, 7, v1 // (op 2208) 
v_and_b32 v4, 15, v1 // (op 2208) 
v_lshl_add_u32 v5, v0, 4, v4 // (op 2208) 


v_and_b32 v0, 48, v1 // (op 2208) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 2208) 


v_lshrrev_b32 v0, 7, v1 // (op 1979) 
v_and_b32 v5, 63, v1 // (op 1979) 
v_lshrrev_b32 v6, 4, v5 // (op 1979) 
v_lshlrev_b32 v7, 2, v6 // (op 1979) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 1979) 


v_lshl_add_u32 v0, s16, 5, v6 // (op 1979) 

v_lshrrev_b32 v6, 2, v5 // (op 1979) 

v_and_b32 v5, 3, v6 // (op 1979) 

v_and_b32 v6, 3, v1 // (op 1979) 
v_lshl_add_u32 v7, v5, 2, v6 // (op 1979) 


v_lshrrev_b32 v5, 6, v1 // (op 1979) 
v_and_b32 v6, 1, v5 // (op 1979) 

v_lshl_add_u32 v5, v6, 4, v7 // (op 1979) 


v_lshl_add_u32 v6, s17, 7, v5 // (op 1979) 

v_mul_lo_u32 v5, v6, s2 // (op 1979) 

v_add_lshl_u32 v6, v0, v5, 2 // (op 1979) 


v_lshrrev_b32 v0, 6, v1 // (op 2093) 
v_and_b32 v5, 1, v0 // (op 2093) 

v_and_b32 v0, 15, v1 // (op 2093) 
v_lshl_add_u32 v7, v5, 4, v0 // (op 2093) 


v_and_b32 v0, 63, v1 // (op 2093) 
v_lshrrev_b32 v5, 4, v0 // (op 2093) 

v_lshlrev_b32 v0, 5, v5 // (op 2093) 

v_lshl_add_u32 v5, v7, 7, v0 // (op 2093) 


v_mul_lo_u32 v0, v5, 6 // (op 2093) 

v_lshrrev_b32 v5, 3, v0 // (op 2093) 

v_lshrrev_b32 v0, 6, v1 // (op 2151) 
v_and_b32 v7, 1, v0 // (op 2151) 

v_and_b32 v0, 15, v1 // (op 2151) 
s_mul_i32 s1, s2, s9 // (op 1981) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 1981) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 1981) least significant
s_mul_hi_u32 s19, s2, s8 // (op 1981) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1981) most significant: sum
s_add_u32 s1, s1, s19 // (op 1981) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 1981) 

v_lshl_add_u32 v8, v7, 4, v0 // (op 2151) 


v_and_b32 v0, 63, v1 // (op 2151) 
v_lshrrev_b32 v7, 4, v0 // (op 2151) 

v_lshl_add_u32 v0, v8, 2, v7 // (op 2151) 


v_lshrrev_b32 v7, 7, v1 // (op 2167) 
v_and_b32 v8, 63, v1 // (op 2167) 
v_lshrrev_b32 v9, 4, v8 // (op 2167) 
v_lshlrev_b32 v10, 2, v9 // (op 2167) 

v_lshl_add_u32 v9, v7, 4, v10 // (op 2167) 


v_lshl_add_u32 v7, s16, 5, v9 // (op 2167) 

v_lshrrev_b32 v9, 2, v8 // (op 2167) 

v_and_b32 v8, 3, v9 // (op 2167) 

v_and_b32 v9, 3, v1 // (op 2167) 
v_lshl_add_u32 v10, v8, 2, v9 // (op 2167) 


v_lshrrev_b32 v8, 6, v1 // (op 2167) 
v_and_b32 v9, 1, v8 // (op 2167) 

v_lshl_add_u32 v8, v9, 4, v10 // (op 2167) 


v_lshl_add_u32 v9, s17, 7, v8 // (op 2167) 

v_mul_lo_u32 v8, v9, s4 // (op 2167) 

v_add_lshl_u32 v9, v7, v8, 2 // (op 2167) 


v_lshrrev_b32 v7, 6, v1 // (op 2237) 
v_and_b32 v8, 1, v7 // (op 2237) 

v_and_b32 v7, 15, v1 // (op 2237) 
v_lshl_add_u32 v10, v8, 4, v7 // (op 2237) 


v_and_b32 v7, 63, v1 // (op 2237) 
v_lshrrev_b32 v8, 4, v7 // (op 2237) 

v_lshlrev_b32 v7, 5, v8 // (op 2237) 

v_lshl_add_u32 v8, v10, 7, v7 // (op 2237) 


v_mul_lo_u32 v7, v8, 6 // (op 2237) 

v_lshrrev_b32 v8, 3, v7 // (op 2237) 

v_lshrrev_b32 v7, 6, v1 // (op 2266) 
v_and_b32 v10, 1, v7 // (op 2266) 

v_and_b32 v7, 15, v1 // (op 2266) 
s_mul_i32 s1, s4, s9 // (op 2169) most significant: low of low * high
v_lshl_add_u32 v11, v10, 4, v7 // (op 2266) 


v_and_b32 v7, 63, v1 // (op 2266) 
v_lshrrev_b32 v10, 4, v7 // (op 2266) 

v_lshl_add_u32 v7, v11, 2, v10 // (op 2266) 


s_mul_i32 s38, s5, s8 // (op 2169) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 2169) least significant
s_mul_hi_u32 s39, s4, s8 // (op 2169) most significant: high of low * low
s_add_u32 s1, s1, s38 // (op 2169) most significant: sum
s_add_u32 s1, s1, s39 // (op 2169) most significant: sum


s_lshl_b64 s[38:39], s[0:1], 2 // (op 2169) 

s_mov_b32 s40, s6 // (op 1981) reg expression
s_mov_b32 s41, s7 // (op 1981) reg expression
s_mov_b32 s42, s18 // (op 1981) reg expression
s_mov_b32 s43, 131072 // (op 1981) call()

s_mov_b32 s44, s10 // (op 2169) reg expression

s_mov_b32 s45, s11 // (op 2169) reg expression
s_mov_b32 s46, s38 // (op 2169) reg expression
s_mov_b32 s47, 131072 // (op 2169) call()


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
v_lshrrev_b32 v12, 3, v1 // (op 1746) 
v_lshl_add_u32 v13, s16, 5, v12 // (op 1746) 

v_and_b32 v12, -8, v1 // (op 1746) 
v_sub_u32 v14, v1, v12 // (op 1746) 

v_lshlrev_b32 v12, 4, v14 // (op 1746) 

v_mov_b32 v15, 0 // (op 1746) convert
v_mov_b32 v14, v12 // (op 1746) convert

v_mul_lo_u32 v17, v13, s13 // (op 1746) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 1746) most significant: high of low * low
v_mul_lo_u32 v16, v13, s12 // (op 1746) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 1746) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 1746) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 1746) most significant half




s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 1748) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 1748) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 1748) least significant
s_mul_hi_u32 s7, s12, s22 // (op 1748) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1748) most significant: sum
s_add_u32 s1, s1, s7 // (op 1748) most significant: sum


s_mov_b32 s48, s20 // (op 1748) reg expression
s_mov_b32 s49, s21 // (op 1748) reg expression
s_mov_b32 s50, s0 // (op 1748) reg expression
s_mov_b32 s51, 131072 // (op 1748) call()

v_lshrrev_b32 v12, 3, v1 // (op 1765) 
v_lshl_add_u32 v13, s16, 5, v12 // (op 1765) 

v_and_b32 v12, -8, v1 // (op 1765) 
v_sub_u32 v14, v1, v12 // (op 1765) 

v_lshlrev_b32 v12, 4, v14 // (op 1765) 

v_mov_b32 v15, 0 // (op 1765) convert
v_mov_b32 v14, v12 // (op 1765) convert

v_mul_lo_u32 v19, v13, s13 // (op 1765) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 1765) most significant: high of low * low
v_mul_lo_u32 v18, v13, s12 // (op 1765) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 1765) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v14 // (op 1765) least significant half
v_addc_co_u32 v19, s[0:1], v19, v15, s[0:1] // (op 1765) most significant half



s_mul_i32 s1, s12, s23 // (op 1767) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 1767) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 1767) least significant
s_mul_hi_u32 s7, s12, s22 // (op 1767) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1767) most significant: sum
s_add_u32 s1, s1, s7 // (op 1767) most significant: sum


s_mov_b32 s48, s20 // (op 1767) reg expression
s_mov_b32 s49, s21 // (op 1767) reg expression
s_mov_b32 s50, s0 // (op 1767) reg expression
s_mov_b32 s51, 131072 // (op 1767) call()

v_and_b32 v12, -2, v1 // (op 1810) 
v_sub_u32 v13, v1, v12 // (op 1810) 

v_lshlrev_b32 v12, 6, v13 // (op 1810) 

v_mov_b32 v15, 0 // (op 1810) convert
v_mov_b32 v14, v12 // (op 1810) convert

v_lshrrev_b32 v12, 1, v1 // (op 1810) 
v_lshl_add_u32 v13, s17, 7, v12 // (op 1810) 

v_mul_lo_u32 v21, v13, s25 // (op 1810) most significant: low of low * high
v_mul_hi_u32 v12, v13, s24 // (op 1810) most significant: high of low * low
v_mul_lo_u32 v20, v13, s24 // (op 1810) least significant: low of low * low
v_add_u32 v21, v21, v12 // (op 1810) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v20 // (op 1810) least significant half
v_addc_co_u32 v13, s[0:1], v15, v21, s[0:1] // (op 1810) most significant half



v_mul_lo_u32 v15, v13, 6 // (op 1810) most significant: low of high * low
v_mul_hi_u32 v20, v12, 6 // (op 1810) most significant: high of low * low
v_mul_lo_u32 v14, v12, 6 // (op 1810) least significant: low of low * low
v_add_u32 v15, v15, v20 // (op 1810) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 1810) 

v_mov_b32 v14, v12 // (op 1810) convert

s_mul_i32 s1, s9, 6 // (op 1812) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 1812) least significant
s_mul_hi_u32 s6, s8, 6 // (op 1812) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1812) most significant: sum

s_mul_i32 s7, s24, s1 // (op 1812) most significant: low of low * high
s_mul_i32 s10, s25, s0 // (op 1812) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 1812) least significant
s_mul_hi_u32 s11, s24, s0 // (op 1812) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 1812) most significant: sum
s_add_u32 s7, s7, s11 // (op 1812) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 1812) 

s_mov_b32 s52, s26 // (op 1812) reg expression
s_mov_b32 s53, s27 // (op 1812) reg expression
s_mov_b32 s54, s0 // (op 1812) reg expression
s_mov_b32 s55, 131072 // (op 1812) call()

v_and_b32 v15, -2, v1 // (op 1829) 
v_sub_u32 v12, v1, v15 // (op 1829) 

v_lshlrev_b32 v13, 6, v12 // (op 1829) 

v_mov_b32 v21, 0 // (op 1829) convert
v_mov_b32 v20, v13 // (op 1829) convert

v_lshrrev_b32 v15, 1, v1 // (op 1829) 
v_lshl_add_u32 v12, s17, 7, v15 // (op 1829) 

v_mul_lo_u32 v23, v12, s25 // (op 1829) most significant: low of low * high
v_mul_hi_u32 v13, v12, s24 // (op 1829) most significant: high of low * low
v_mul_lo_u32 v22, v12, s24 // (op 1829) least significant: low of low * low
v_add_u32 v23, v23, v13 // (op 1829) most significant: sum


v_add_co_u32 v12, s[0:1], v20, v22 // (op 1829) least significant half
v_addc_co_u32 v13, s[0:1], v21, v23, s[0:1] // (op 1829) most significant half



v_mul_lo_u32 v21, v13, 6 // (op 1829) most significant: low of high * low
v_mul_hi_u32 v15, v12, 6 // (op 1829) most significant: high of low * low
v_mul_lo_u32 v20, v12, 6 // (op 1829) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 1829) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[20:21] // (op 1829) 

v_mov_b32 v15, v12 // (op 1829) convert

s_mul_i32 s1, s9, 6 // (op 1831) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 1831) least significant
s_mul_hi_u32 s6, s8, 6 // (op 1831) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1831) most significant: sum

s_mul_i32 s7, s24, s1 // (op 1831) most significant: low of low * high
s_mul_i32 s10, s25, s0 // (op 1831) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 1831) least significant
s_mul_hi_u32 s11, s24, s0 // (op 1831) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 1831) most significant: sum
s_add_u32 s7, s7, s11 // (op 1831) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 1831) 

s_mov_b32 s52, s26 // (op 1831) reg expression
s_mov_b32 s53, s27 // (op 1831) reg expression
s_mov_b32 s54, s0 // (op 1831) reg expression
s_mov_b32 s55, 131072 // (op 1831) call()

v_lshrrev_b32 v12, 7, v1 // (op 1874) 
v_and_b32 v13, 15, v1 // (op 1874) 
v_lshl_add_u32 v20, v12, 4, v13 // (op 1874) 


v_lshl_add_u32 v12, s16, 5, v20 // (op 1874) 

v_and_b32 v13, 63, v1 // (op 1874) 
v_lshrrev_b32 v20, 4, v13 // (op 1874) 

v_mul_lo_u32 v13, v12, s28 // (op 1874) 
v_add_u32 v13, v13, v20 // (op 1874) 


s_mul_i32 s1, s28, s23 // (op 1876) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 1876) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 1876) least significant
s_mul_hi_u32 s7, s28, s22 // (op 1876) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1876) most significant: sum
s_add_u32 s1, s1, s7 // (op 1876) most significant: sum


s_mov_b32 s56, s30 // (op 1876) reg expression
s_mov_b32 s57, s31 // (op 1876) reg expression
s_mov_b32 s58, s0 // (op 1876) reg expression
s_mov_b32 s59, 131072 // (op 1876) call()

v_lshrrev_b32 v12, 7, v1 // (op 1893) 
v_and_b32 v20, 15, v1 // (op 1893) 
v_lshl_add_u32 v21, v12, 4, v20 // (op 1893) 


v_lshl_add_u32 v12, s16, 5, v21 // (op 1893) 

v_and_b32 v20, 63, v1 // (op 1893) 
v_lshrrev_b32 v21, 4, v20 // (op 1893) 

v_mul_lo_u32 v20, v12, s28 // (op 1893) 
v_add_u32 v20, v20, v21 // (op 1893) 


s_mul_i32 s1, s28, s23 // (op 1895) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 1895) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 1895) least significant
s_mul_hi_u32 s7, s28, s22 // (op 1895) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1895) most significant: sum
s_add_u32 s1, s1, s7 // (op 1895) most significant: sum


s_mov_b32 s56, s30 // (op 1895) reg expression
s_mov_b32 s57, s31 // (op 1895) reg expression
s_mov_b32 s58, s0 // (op 1895) reg expression
s_mov_b32 s59, 131072 // (op 1895) call()

v_and_b32 v12, -4, v1 // (op 1938) 
v_sub_u32 v21, v1, v12 // (op 1938) 

v_lshrrev_b32 v12, 2, v1 // (op 1938) 
v_lshlrev_b32 v22, 1, v12 // (op 1938) 

v_lshl_add_u32 v12, s17, 7, v22 // (op 1938) 

v_mul_lo_u32 v22, v12, s32 // (op 1938) 

v_add_u32 v12, v21, v22 // (op 1938) 


s_mul_i32 s1, s32, s9 // (op 1940) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 1940) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 1940) least significant
s_mul_hi_u32 s7, s32, s8 // (op 1940) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1940) most significant: sum
s_add_u32 s1, s1, s7 // (op 1940) most significant: sum


s_mov_b32 s60, s34 // (op 1940) reg expression
s_mov_b32 s61, s35 // (op 1940) reg expression
s_mov_b32 s62, s0 // (op 1940) reg expression
s_mov_b32 s63, 131072 // (op 1940) call()

v_and_b32 v21, -4, v1 // (op 1957) 
v_sub_u32 v22, v1, v21 // (op 1957) 

v_lshrrev_b32 v21, 2, v1 // (op 1957) 
v_lshlrev_b32 v23, 1, v21 // (op 1957) 

v_lshl_add_u32 v21, s17, 7, v23 // (op 1957) 

v_mul_lo_u32 v23, v21, s32 // (op 1957) 

v_add_u32 v21, v22, v23 // (op 1957) 


s_mul_i32 s1, s32, s9 // (op 1959) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 1959) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 1959) least significant
s_mul_hi_u32 s7, s32, s8 // (op 1959) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1959) most significant: sum
s_add_u32 s1, s1, s7 // (op 1959) most significant: sum


s_mov_b32 s60, s34 // (op 1959) reg expression
s_mov_b32 s61, s35 // (op 1959) reg expression
s_mov_b32 s62, s0 // (op 1959) reg expression
s_mov_b32 s63, 131072 // (op 1959) call()




buffer_load_ubyte v22, v13, s[56:59], 0 offen offset: 0 // (op 1544) Load value



buffer_load_dwordx4 v[24:27], v14, s[52:55], 0 offen // (op 1548) Load value
buffer_load_dwordx2 v[28:29], v14, s[52:55], 0 offen offset:16 // (op 1548) Load value
buffer_load_dwordx4 v[30:33], v14, s[52:55], 0 offen offset:24 // (op 1548) Load value
buffer_load_dwordx2 v[34:35], v14, s[52:55], 0 offen offset:40 // (op 1548) Load value

v_mov_b64 v[36:37], s[32:33] // (op 1551) convert

v_mov_b32 v23, v12 // (op 1551) 
buffer_load_ubyte v38, v23, s[60:63], 0 offen offset: 0 // (op 1551) Load value
v_add_u32 v23, v23, v36 // (op 1551) 
buffer_load_ubyte v39, v23, s[60:63], 0 offen offset: 0 // (op 1551) Load value


v_and_b32 v23, -8, v1 // (op 2023) 
v_sub_u32 v36, v1, v23 // (op 2023) 

v_lshlrev_b32 v23, 4, v36 // (op 2023) 

v_lshrrev_b32 v36, 3, v1 // (op 2023) 
v_lshl_add_u32 v37, v36, 7, v23 // (op 2023) 


s_nop 0
v_readfirstlane_b32 s0, v37 // (op 2023) Wait state hazard: VALU Write Readlane Hazard




s_nop 1
s_mov_b32 m0, s0 // (op 1699) reg expression
buffer_load_dwordx4 v16, s[48:51], 0 offen offset: 0 lds // (op 1699) Load value direct to lds




buffer_load_ubyte v23, v13, s[56:59], 0 offen offset:4 // (op 1558) Load value




buffer_load_dwordx4 v[40:43], v14, s[52:55], 0 offen offset:96 // (op 1562) Load value
buffer_load_dwordx2 v[44:45], v14, s[52:55], 0 offen offset:112 // (op 1562) Load value
buffer_load_dwordx4 v[46:49], v14, s[52:55], 0 offen offset:120 // (op 1562) Load value
buffer_load_dwordx2 v[50:51], v14, s[52:55], 0 offen offset:136 // (op 1562) Load value


v_mov_b64 v[36:37], s[32:33] // (op 1565) convert

v_mov_b32 v13, v12 // (op 1565) 
buffer_load_ubyte v52, v13, s[60:63], 0 offen offset:4 // (op 1565) Load value
v_add_u32 v13, v13, v36 // (op 1565) 
buffer_load_ubyte v53, v13, s[60:63], 0 offen offset:4 // (op 1565) Load value



v_and_b32 v12, -8, v1 // (op 2196) 
v_sub_u32 v13, v1, v12 // (op 2196) 

v_lshlrev_b32 v12, 4, v13 // (op 2196) 

v_lshrrev_b32 v13, 3, v1 // (op 2196) 
v_lshl_add_u32 v14, v13, 7, v12 // (op 2196) 


s_nop 0
v_readfirstlane_b32 s0, v14 // (op 2196) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s1, 4096, s0 // (op 1704) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 1704) reg expression
s_mov_b32 s6, 128 // (op 1704) call()
buffer_load_dwordx4 v16, s[48:51], s6 offen offset: 0 lds // (op 1704) Load value direct to lds




v_and_b32 v12, -2, v1 // (op 2076) 
v_sub_u32 v13, v1, v12 // (op 2076) 

v_lshlrev_b32 v12, 6, v13 // (op 2076) 

v_lshrrev_b32 v13, 1, v1 // (op 2076) 
v_lshl_add_u32 v14, v13, 7, v12 // (op 2076) 


v_mul_lo_u32 v12, v14, 6 // (op 2076) 

v_lshrrev_b32 v13, 3, v12 // (op 2076) 




s_waitcnt vmcnt(14)
ds_write_b128 v13, v[24:27] offset:8192 // (op 1570) Store local data 
s_waitcnt vmcnt(13)
ds_write_b64 v13, v[28:29] offset:8208 // (op 1570) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v13, v[30:33] offset:8216 // (op 1570) Store local data 
s_waitcnt vmcnt(11)
ds_write_b64 v13, v[34:35] offset:8232 // (op 1570) Store local data 

v_and_b32 v12, -4, v1 // (op 2134) 
v_sub_u32 v13, v1, v12 // (op 2134) 

v_lshrrev_b32 v12, 2, v1 // (op 2134) 
v_lshl_add_u32 v14, v12, 3, v13 // (op 2134) 





s_waitcnt vmcnt(10)
ds_write_b8 v14, v38 offset:20480 // (op 1571) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v14, v39 offset:20484 // (op 1571) Store local data 

v_and_b32 v12, -8, v1 // (op 1995) 
v_sub_u32 v13, v1, v12 // (op 1995) 

v_lshlrev_b32 v12, 4, v13 // (op 1995) 

v_lshrrev_b32 v13, 3, v1 // (op 1995) 
v_lshl_add_u32 v14, v13, 7, v12 // (op 1995) 


s_nop 0
v_readfirstlane_b32 s0, v14 // (op 1995) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -2, v1 // (op 2047) 
v_sub_u32 v13, v1, v12 // (op 2047) 

v_lshlrev_b32 v12, 6, v13 // (op 2047) 

v_lshrrev_b32 v13, 1, v1 // (op 2047) 
v_lshl_add_u32 v14, v13, 7, v12 // (op 2047) 


v_mul_lo_u32 v12, v14, 6 // (op 2047) 

v_lshrrev_b32 v13, 3, v12 // (op 2047) 

v_and_b32 v12, -4, v1 // (op 2105) 
v_sub_u32 v14, v1, v12 // (op 2105) 

v_lshrrev_b32 v12, 2, v1 // (op 2105) 
v_lshl_add_u32 v16, v12, 3, v14 // (op 2105) 


v_and_b32 v12, -8, v1 // (op 2181) 
v_sub_u32 v14, v1, v12 // (op 2181) 

v_lshlrev_b32 v12, 4, v14 // (op 2181) 

v_lshrrev_b32 v14, 3, v1 // (op 2181) 
v_lshl_add_u32 v17, v14, 7, v12 // (op 2181) 


s_nop 0
v_readfirstlane_b32 s1, v17 // (op 2181) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -2, v1 // (op 2220) 
v_sub_u32 v14, v1, v12 // (op 2220) 

v_lshlrev_b32 v12, 6, v14 // (op 2220) 

v_lshrrev_b32 v14, 1, v1 // (op 2220) 
v_lshl_add_u32 v17, v14, 7, v12 // (op 2220) 


v_mul_lo_u32 v12, v17, 6 // (op 2220) 

v_lshrrev_b32 v14, 3, v12 // (op 2220) 

v_and_b32 v12, -4, v1 // (op 2249) 
v_sub_u32 v17, v1, v12 // (op 2249) 

v_lshrrev_b32 v12, 2, v1 // (op 2249) 
v_lshl_add_u32 v36, v12, 3, v17 // (op 2249) 


s_mov_b32 s6, 0 // (op 620) call()
s_ashr_i64 s[10:11], s[36:37], 7 // (op 619) 
s_mov_b32 s7, s10 // (op 619) convert

s_and_b32 s10, s7, 4294967294 // (op 619) 

s_cmp_lt_u32 s6, s10 // (op 619) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopBottom_KLoop_619 // (op 619) Condition: Top (jump to RR_GEMM_TN_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopBottom_KLoop_619 if false)
RR_GEMM_TN_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopTop_KLoop_619:
s_add_u32 s6, s6, 2 // (op 621) 
s_waitcnt vmcnt(8)
s_barrier  // (op 2313) 



ds_read_b128 v[54:57], v3 // (op 65) Load local data 
ds_read_b128 v[58:61], v3 offset:64 // (op 65) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 2326) 



ds_read_b128 v[62:65], v5 offset:8192 // (op 70) Load local data 
ds_read_b64 v[66:67], v5 offset:8208 // (op 70) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 2340) 



ds_read_u8 v12, v0 offset:20480 // (op 79) Load local data 



ds_read_b128 v[68:71], v5 offset:11264 // (op 339) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[54:61], v[62:67], a[0:3], v22, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 82) 
ds_read_b64 v[72:73], v5 offset:11280 // (op 339) Load local data 





ds_read_u8 v12, v0 offset:20608 // (op 342) Load local data 



ds_read_b128 v[62:65], v5 offset:14336 // (op 495) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[54:61], v[68:73], a[4:7], v22, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 343) 
ds_read_b64 v[66:67], v5 offset:14352 // (op 495) Load local data 





ds_read_u8 v12, v0 offset:20736 // (op 498) Load local data 



ds_read_b128 v[68:71], v5 offset:17408 // (op 651) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[54:61], v[62:67], a[8:11], v22, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 499) 
ds_read_b64 v[72:73], v5 offset:17424 // (op 651) Load local data 





ds_read_u8 v12, v0 offset:20864 // (op 654) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[54:61], v[68:73], a[12:15], v22, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 655) 





v_mov_b32 v12, v20 // (op 75) 

buffer_load_ubyte v22, v12, s[56:59], 0 offen offset:8 // (op 75) Load value


v_mov_b32 v17, v15 // (op 10) 

buffer_load_dwordx4 v[24:27], v17, s[52:55], 0 offen offset:192 // (op 10) Load value
buffer_load_dwordx2 v[28:29], v17, s[52:55], 0 offen offset:208 // (op 10) Load value
buffer_load_dwordx4 v[30:33], v17, s[52:55], 0 offen offset:216 // (op 10) Load value
buffer_load_dwordx2 v[34:35], v17, s[52:55], 0 offen offset:232 // (op 10) Load value

v_mov_b64 v[54:55], s[32:33] // (op 21) convert

v_mov_b32 v37, v21 // (op 21) 
v_mov_b32 v56, v37 // (op 21) 
buffer_load_ubyte v38, v56, s[60:63], 0 offen offset:8 // (op 21) Load value
v_add_u32 v56, v56, v54 // (op 21) 
buffer_load_ubyte v39, v56, s[60:63], 0 offen offset:8 // (op 21) Load value


s_barrier  // (op 2316) 


v_mov_b64 v[54:55], v[18:19] // (op 1683) 

s_mov_b32 m0, s0 // (op 1683) reg expression
s_mov_b32 s7, 256 // (op 1683) call()
buffer_load_dwordx4 v54, s[48:51], s7 offen offset: 0 lds // (op 1683) Load value direct to lds

s_barrier  // (op 2363) 



s_waitcnt vmcnt(14)
ds_write_b128 v14, v[40:43] offset:20992 // (op 235) Store local data 
s_waitcnt vmcnt(13)
ds_write_b64 v14, v[44:45] offset:21008 // (op 235) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v14, v[46:49] offset:21016 // (op 235) Store local data 
s_waitcnt vmcnt(11)
ds_write_b64 v14, v[50:51] offset:21032 // (op 235) Store local data 
s_barrier  // (op 2369) 



s_waitcnt vmcnt(10)
ds_write_b8 v36, v52 offset:33280 // (op 236) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v36, v53 offset:33284 // (op 236) Store local data 
s_waitcnt vmcnt(8)
s_barrier  // (op 2354) 



ds_read_b128 v[56:59], v4 offset:4096 // (op 227) Load local data 
ds_read_b128 v[60:63], v4 offset:4160 // (op 227) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 2360) 



ds_read_b128 v[64:67], v8 offset:20992 // (op 229) Load local data 
ds_read_b64 v[68:69], v8 offset:21008 // (op 229) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 2366) 



ds_read_u8 v70, v7 offset:33280 // (op 233) Load local data 



ds_read_b128 v[72:75], v8 offset:24064 // (op 348) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[56:63], v[64:69], a[0:3], v23, v70 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 234) 
ds_read_b64 v[76:77], v8 offset:24080 // (op 348) Load local data 





ds_read_u8 v64, v7 offset:33408 // (op 351) Load local data 



ds_read_b128 v[66:69], v8 offset:27136 // (op 504) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[56:63], v[72:77], a[4:7], v23, v64 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 344) 
ds_read_b64 v[70:71], v8 offset:27152 // (op 504) Load local data 





ds_read_u8 v64, v7 offset:33536 // (op 507) Load local data 



ds_read_b128 v[72:75], v8 offset:30208 // (op 660) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[56:63], v[66:71], a[8:11], v23, v64 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 500) 
ds_read_b64 v[76:77], v8 offset:30224 // (op 660) Load local data 





s_barrier  // (op 2442) 
ds_read_u8 v64, v7 offset:33664 // (op 663) Load local data 
s_barrier  // (op 2446) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[56:63], v[72:77], a[12:15], v23, v64 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 656) 






buffer_load_ubyte v23, v12, s[56:59], 0 offen offset:12 // (op 231) Load value




v_add_u32 v20, 8, v20 // (op 1885) 
buffer_load_dwordx4 v[40:43], v17, s[52:55], 0 offen offset:288 // (op 237) Load value
buffer_load_dwordx2 v[44:45], v17, s[52:55], 0 offen offset:304 // (op 237) Load value
buffer_load_dwordx4 v[46:49], v17, s[52:55], 0 offen offset:312 // (op 237) Load value
buffer_load_dwordx2 v[50:51], v17, s[52:55], 0 offen offset:328 // (op 237) Load value


v_add_u32 v15, 192, v15 // (op 1821) 
v_mov_b64 v[56:57], s[32:33] // (op 238) convert

v_mov_b32 v12, v37 // (op 238) 
buffer_load_ubyte v52, v12, s[60:63], 0 offen offset:12 // (op 238) Load value
v_add_u32 v12, v12, v56 // (op 238) 
buffer_load_ubyte v53, v12, s[60:63], 0 offen offset:12 // (op 238) Load value



s_barrier  // (op 2357) 
v_add_u32 v21, 8, v21 // (op 1949) 


s_add_u32 s7, 4096, s1 // (op 1694) 

s_mov_b32 m0, s7 // (op 1694) reg expression
s_mov_b32 s10, 384 // (op 1694) call()
buffer_load_dwordx4 v54, s[48:51], s10 offen offset: 0 lds // (op 1694) Load value direct to lds



s_barrier  // (op 2329) 
v_add_co_u32 v18, vcc, 256, v18 // (op 1757) least significant half



v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 1757) most significant half
s_waitcnt vmcnt(14)
ds_write_b128 v13, v[24:27] offset:8192 // (op 13) Store local data 
s_waitcnt vmcnt(13)
ds_write_b64 v13, v[28:29] offset:8208 // (op 13) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v13, v[30:33] offset:8216 // (op 13) Store local data 
s_waitcnt vmcnt(11)
ds_write_b64 v13, v[34:35] offset:8232 // (op 13) Store local data 
s_barrier  // (op 2343) 



s_waitcnt vmcnt(10)
ds_write_b8 v16, v38 offset:20480 // (op 26) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v16, v39 offset:20484 // (op 26) Store local data 
s_ashr_i64 s[10:11], s[36:37], 7 // (op 619) 
s_mov_b32 s7, s10 // (op 619) convert

s_and_b32 s10, s7, 4294967294 // (op 619) 

s_cmp_lt_u32 s6, s10 // (op 619) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopTop_KLoop_619 // (op 619) Condition: Bottom (jump to RR_GEMM_TN_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopTop_KLoop_619 if true)
RR_GEMM_TN_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopBottom_KLoop_619:














s_barrier  // (op 2426) 
v_lshrrev_b32 v12, 3, v1 // (op 1717) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1717) 
v_lshl_add_u32 v13, s16, 5, v12 // (op 1717) 

s_lshr_b32 s6, s0, 1 // (op 1717) 

v_and_b32 v12, -8, v1 // (op 1717) 
v_sub_u32 v14, v1, v12 // (op 1717) 

v_lshlrev_b32 v12, 4, v14 // (op 1717) 

v_lshl_add_u32 v14, s6, 8, v12 // (op 1717) 


v_mov_b32 v25, 0 // (op 1717) convert
v_mov_b32 v24, v14 // (op 1717) convert

v_mul_lo_u32 v15, v13, s13 // (op 1717) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 1717) most significant: high of low * low
v_mul_lo_u32 v14, v13, s12 // (op 1717) least significant: low of low * low
v_add_u32 v15, v15, v12 // (op 1717) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v24 // (op 1717) least significant half
v_addc_co_u32 v15, s[0:1], v15, v25, s[0:1] // (op 1717) most significant half



s_mul_i32 s1, s12, s23 // (op 1719) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 1719) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 1719) least significant
s_mul_hi_u32 s7, s12, s22 // (op 1719) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1719) most significant: sum
s_add_u32 s1, s1, s7 // (op 1719) most significant: sum


s_mov_b32 s48, s20 // (op 1719) reg expression
s_mov_b32 s49, s21 // (op 1719) reg expression
s_mov_b32 s50, s0 // (op 1719) reg expression
s_mov_b32 s51, 131072 // (op 1719) call()

v_and_b32 v12, -2, v1 // (op 1787) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1787) 
v_sub_u32 v13, v1, v12 // (op 1787) 

s_lshr_b32 s6, s0, 1 // (op 1787) 

v_lshlrev_b32 v12, 6, v13 // (op 1787) 

v_lshl_add_u32 v13, s6, 8, v12 // (op 1787) 


v_mov_b32 v17, 0 // (op 1787) convert
v_mov_b32 v16, v13 // (op 1787) convert

v_lshrrev_b32 v12, 1, v1 // (op 1787) 
v_lshl_add_u32 v13, s17, 7, v12 // (op 1787) 

v_mul_lo_u32 v19, v13, s25 // (op 1787) most significant: low of low * high
v_mul_hi_u32 v12, v13, s24 // (op 1787) most significant: high of low * low
v_mul_lo_u32 v18, v13, s24 // (op 1787) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 1787) most significant: sum


v_add_co_u32 v12, s[0:1], v16, v18 // (op 1787) least significant half
v_addc_co_u32 v13, s[0:1], v17, v19, s[0:1] // (op 1787) most significant half



v_mul_lo_u32 v17, v13, 6 // (op 1787) most significant: low of high * low
v_mul_hi_u32 v18, v12, 6 // (op 1787) most significant: high of low * low
v_mul_lo_u32 v16, v12, 6 // (op 1787) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 1787) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[16:17] // (op 1787) 

v_mov_b32 v16, v12 // (op 1787) convert

s_mul_i32 s1, s9, 6 // (op 1789) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 1789) least significant
s_mul_hi_u32 s6, s8, 6 // (op 1789) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1789) most significant: sum

s_mul_i32 s7, s24, s1 // (op 1789) most significant: low of low * high
s_mul_i32 s10, s25, s0 // (op 1789) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 1789) least significant
s_mul_hi_u32 s11, s24, s0 // (op 1789) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 1789) most significant: sum
s_add_u32 s7, s7, s11 // (op 1789) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 1789) 

s_mov_b32 s52, s26 // (op 1789) reg expression
s_mov_b32 s53, s27 // (op 1789) reg expression
s_mov_b32 s54, s0 // (op 1789) reg expression
s_mov_b32 s55, 131072 // (op 1789) call()

v_lshrrev_b32 v12, 7, v1 // (op 1851) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1851) 
v_and_b32 v13, 15, v1 // (op 1851) 
s_lshr_b32 s6, s0, 1 // (op 1851) 

v_lshl_add_u32 v17, v12, 4, v13 // (op 1851) 


v_lshl_add_u32 v12, s16, 5, v17 // (op 1851) 

v_and_b32 v13, 63, v1 // (op 1851) 
v_lshrrev_b32 v17, 4, v13 // (op 1851) 

v_lshl_add_u32 v13, s6, 3, v17 // (op 1851) 


v_mul_lo_u32 v17, v12, s28 // (op 1851) 
v_add_u32 v17, v17, v13 // (op 1851) 


s_mul_i32 s1, s28, s23 // (op 1853) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 1853) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 1853) least significant
s_mul_hi_u32 s7, s28, s22 // (op 1853) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1853) most significant: sum
s_add_u32 s1, s1, s7 // (op 1853) most significant: sum


s_mov_b32 s56, s30 // (op 1853) reg expression
s_mov_b32 s57, s31 // (op 1853) reg expression
s_mov_b32 s58, s0 // (op 1853) reg expression
s_mov_b32 s59, 131072 // (op 1853) call()

v_and_b32 v12, -4, v1 // (op 1915) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1915) 
v_sub_u32 v13, v1, v12 // (op 1915) 

s_lshr_b32 s6, s0, 1 // (op 1915) 

v_lshrrev_b32 v12, 2, v1 // (op 1915) 
v_lshl_add_u32 v18, s6, 3, v13 // (op 1915) 


v_lshlrev_b32 v13, 1, v12 // (op 1915) 

v_lshl_add_u32 v12, s17, 7, v13 // (op 1915) 

v_mul_lo_u32 v13, v12, s32 // (op 1915) 

v_add_u32 v12, v18, v13 // (op 1915) 


s_mul_i32 s1, s32, s9 // (op 1917) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 1917) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 1917) least significant
s_mul_hi_u32 s7, s32, s8 // (op 1917) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1917) most significant: sum
s_add_u32 s1, s1, s7 // (op 1917) most significant: sum


s_mov_b32 s60, s34 // (op 1917) reg expression
s_mov_b32 s61, s35 // (op 1917) reg expression
s_mov_b32 s62, s0 // (op 1917) reg expression
s_mov_b32 s63, 131072 // (op 1917) call()

v_and_b32 v13, -8, v1 // (op 2008) 
v_sub_u32 v18, v1, v13 // (op 2008) 

v_lshlrev_b32 v13, 4, v18 // (op 2008) 

v_lshrrev_b32 v18, 3, v1 // (op 2008) 
v_lshl_add_u32 v19, v18, 7, v13 // (op 2008) 


s_nop 0
v_readfirstlane_b32 s0, v19 // (op 2008) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v13, -2, v1 // (op 2060) 
v_sub_u32 v18, v1, v13 // (op 2060) 

v_lshlrev_b32 v13, 6, v18 // (op 2060) 

v_lshrrev_b32 v18, 1, v1 // (op 2060) 
v_lshl_add_u32 v19, v18, 7, v13 // (op 2060) 


v_mul_lo_u32 v13, v19, 6 // (op 2060) 

v_lshrrev_b32 v18, 3, v13 // (op 2060) 

v_and_b32 v13, -4, v1 // (op 2118) 
v_sub_u32 v19, v1, v13 // (op 2118) 

v_lshrrev_b32 v13, 2, v1 // (op 2118) 
v_lshl_add_u32 v20, v13, 3, v19 // (op 2118) 


s_ashr_i64 s[6:7], s[36:37], 7 // (op 625) 
s_and_b32 s1, s6, 4294967294 // (op 625) 

s_ashr_i64 s[6:7], s[36:37], 7 // (op 624) 
s_mov_b32 s10, s6 // (op 624) convert

s_cmp_lt_u32 s1, s10 // (op 624) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopBottom_KLoopTail_624 // (op 624) Condition: Top (jump to RR_GEMM_TN_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopBottom_KLoopTail_624 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopTop_KLoopTail_624:
s_barrier  // (op 2319) 
s_add_u32 s1, s1, 1 // (op 626) 


v_mov_b64 v[22:23], v[14:15] // (op 1688) 

s_mov_b32 m0, s0 // (op 1688) reg expression
buffer_load_dwordx4 v22, s[48:51], 0 offen offset: 0 lds // (op 1688) Load value direct to lds

v_add_co_u32 v14, vcc, 128, v14 // (op 1707) least significant half
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 1707) most significant half


v_mov_b32 v13, v16 // (op 179) 

buffer_load_dwordx4 v[22:25], v13, s[52:55], 0 offen // (op 179) Load value
buffer_load_dwordx2 v[26:27], v13, s[52:55], 0 offen offset:16 // (op 179) Load value
buffer_load_dwordx4 v[28:31], v13, s[52:55], 0 offen offset:24 // (op 179) Load value
buffer_load_dwordx2 v[32:33], v13, s[52:55], 0 offen offset:40 // (op 179) Load value

v_add_u32 v16, 96, v16 // (op 1779) 

s_barrier  // (op 2332) 
v_mov_b64 v[34:35], s[32:33] // (op 180) convert




v_mov_b32 v13, v12 // (op 180) 
v_mov_b32 v19, v13 // (op 180) 
buffer_load_ubyte v36, v19, s[60:63], 0 offen offset: 0 // (op 180) Load value
v_add_u32 v19, v19, v34 // (op 180) 
s_waitcnt vmcnt(4)
ds_write_b128 v18, v[22:25] offset:8192 // (op 177) Store local data 
s_waitcnt vmcnt(3)
ds_write_b64 v18, v[26:27] offset:8208 // (op 177) Store local data 
s_waitcnt vmcnt(2)
ds_write_b128 v18, v[28:31] offset:8216 // (op 177) Store local data 
buffer_load_ubyte v37, v19, s[60:63], 0 offen offset: 0 // (op 180) Load value



v_add_u32 v12, 4, v12 // (op 1907) 
s_waitcnt vmcnt(2)
ds_write_b64 v18, v[32:33] offset:8232 // (op 177) Store local data 

s_barrier  // (op 2322) 



s_barrier  // (op 2346) 
ds_read_b128 v[28:31], v3 // (op 169) Load local data 



ds_read_b128 v[32:35], v3 offset:64 // (op 169) Load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v20, v36 offset:20480 // (op 178) Store local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 2336) 
s_waitcnt vmcnt(0)
ds_write_b8 v20, v37 offset:20484 // (op 178) Store local data 




ds_read_b128 v[22:25], v5 offset:8192 // (op 171) Load local data 
ds_read_b64 v[26:27], v5 offset:8208 // (op 171) Load local data 


v_mov_b32 v13, v17 // (op 173) 

buffer_load_ubyte v19, v13, s[56:59], 0 offen offset: 0 // (op 173) Load value

v_add_u32 v17, 4, v17 // (op 1843) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 2350) 



ds_read_u8 v13, v0 offset:20480 // (op 175) Load local data 



ds_read_b128 v[36:39], v5 offset:11264 // (op 320) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[28:35], v[22:27], a[0:3], v19, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 176) 
ds_read_b64 v[40:41], v5 offset:11280 // (op 320) Load local data 





ds_read_u8 v13, v0 offset:20608 // (op 323) Load local data 



ds_read_b128 v[22:25], v5 offset:14336 // (op 476) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[28:35], v[36:41], a[4:7], v19, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 324) 
ds_read_b64 v[26:27], v5 offset:14352 // (op 476) Load local data 





ds_read_u8 v13, v0 offset:20736 // (op 479) Load local data 



ds_read_b128 v[36:39], v5 offset:17408 // (op 632) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[28:35], v[22:27], a[8:11], v19, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 480) 
ds_read_b64 v[40:41], v5 offset:17424 // (op 632) Load local data 





ds_read_u8 v13, v0 offset:20864 // (op 635) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[28:35], v[36:41], a[12:15], v19, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 636) 




s_ashr_i64 s[6:7], s[36:37], 7 // (op 624) 
s_mov_b32 s10, s6 // (op 624) convert

s_cmp_lt_u32 s1, s10 // (op 624) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopTop_KLoopTail_624 // (op 624) Condition: Bottom (jump to RR_GEMM_TN_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopTop_KLoopTail_624 if true)
RR_GEMM_TN_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopBottom_KLoopTail_624:






















s_barrier  // (op 2430) 







v_cmp_eq_f32 vcc, 0.00000, v10 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_TN_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ConditionalFalse_0_beta_123
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


v_accvgpr_read v12, a4 // (op 860) 
v_accvgpr_read v13, a5 // (op 860) 
v_accvgpr_read v14, a6 // (op 860) 
v_accvgpr_read v15, a7 // (op 860) 
v_mul_f32 v16, v11, v12 // (op 860) 
v_mul_f32 v17, v11, v13 // (op 860) 
v_mul_f32 v18, v11, v14 // (op 860) 
v_mul_f32 v19, v11, v15 // (op 860) 



s_lshl_b32 s0, s4, 7 // (op 862) 
v_add_u32 v0, v9, s0 // (op 862) 


buffer_store_dwordx4 v[16:19], v0, s[44:47], 0 offen // (op 862) Store value


v_accvgpr_read v12, a8 // (op 872) 
v_accvgpr_read v13, a9 // (op 872) 
v_accvgpr_read v14, a10 // (op 872) 
v_accvgpr_read v15, a11 // (op 872) 
v_mul_f32 v16, v11, v12 // (op 872) 
v_mul_f32 v17, v11, v13 // (op 872) 
v_mul_f32 v18, v11, v14 // (op 872) 
v_mul_f32 v19, v11, v15 // (op 872) 



s_lshl_b32 s0, s4, 8 // (op 874) 
v_add_u32 v0, v9, s0 // (op 874) 


buffer_store_dwordx4 v[16:19], v0, s[44:47], 0 offen // (op 874) Store value


v_accvgpr_read v12, a12 // (op 884) 
v_accvgpr_read v13, a13 // (op 884) 
v_accvgpr_read v14, a14 // (op 884) 
v_accvgpr_read v15, a15 // (op 884) 
v_mul_f32 v16, v11, v12 // (op 884) 
v_mul_f32 v17, v11, v13 // (op 884) 
v_mul_f32 v18, v11, v14 // (op 884) 
v_mul_f32 v19, v11, v15 // (op 884) 



s_lshl_b32 s0, s4, 7 // (op 886) 
s_mul_i32 s1, 3, s0 // (op 886) 

v_add_u32 v0, v9, s1 // (op 886) 


buffer_store_dwordx4 v[16:19], v0, s[44:47], 0 offen // (op 886) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_TN_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ConditionalBottom_0_beta_123
RR_GEMM_TN_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ConditionalFalse_0_beta_123:


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




s_lshl_b32 s0, s2, 7 // (op 918) 
v_add_u32 v0, v6, s0 // (op 918) 


buffer_load_dwordx4 v[12:15], v0, s[40:43], 0 offen // (op 918) Load value

v_accvgpr_read v16, a4 // (op 915) 
v_accvgpr_read v17, a5 // (op 915) 
v_accvgpr_read v18, a6 // (op 915) 
v_accvgpr_read v19, a7 // (op 915) 
v_mul_f32 v20, v11, v16 // (op 915) 
v_mul_f32 v21, v11, v17 // (op 915) 
v_mul_f32 v22, v11, v18 // (op 915) 
v_mul_f32 v23, v11, v19 // (op 915) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v10, v12 // (op 919) 
v_mul_f32 v17, v10, v13 // (op 919) 
v_mul_f32 v18, v10, v14 // (op 919) 
v_mul_f32 v19, v10, v15 // (op 919) 

v_add_f32 v12, v16, v20 // (op 916) 
v_add_f32 v13, v17, v21 // (op 916) 
v_add_f32 v14, v18, v22 // (op 916) 
v_add_f32 v15, v19, v23 // (op 916) 




s_lshl_b32 s0, s4, 7 // (op 917) 
v_add_u32 v0, v9, s0 // (op 917) 


buffer_store_dwordx4 v[12:15], v0, s[44:47], 0 offen // (op 917) Store value




s_lshl_b32 s0, s2, 8 // (op 930) 
v_add_u32 v0, v6, s0 // (op 930) 


buffer_load_dwordx4 v[12:15], v0, s[40:43], 0 offen // (op 930) Load value

v_accvgpr_read v16, a8 // (op 927) 
v_accvgpr_read v17, a9 // (op 927) 
v_accvgpr_read v18, a10 // (op 927) 
v_accvgpr_read v19, a11 // (op 927) 
v_mul_f32 v20, v11, v16 // (op 927) 
v_mul_f32 v21, v11, v17 // (op 927) 
v_mul_f32 v22, v11, v18 // (op 927) 
v_mul_f32 v23, v11, v19 // (op 927) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v10, v12 // (op 931) 
v_mul_f32 v17, v10, v13 // (op 931) 
v_mul_f32 v18, v10, v14 // (op 931) 
v_mul_f32 v19, v10, v15 // (op 931) 

v_add_f32 v12, v16, v20 // (op 928) 
v_add_f32 v13, v17, v21 // (op 928) 
v_add_f32 v14, v18, v22 // (op 928) 
v_add_f32 v15, v19, v23 // (op 928) 




s_lshl_b32 s0, s4, 8 // (op 929) 
v_add_u32 v0, v9, s0 // (op 929) 


buffer_store_dwordx4 v[12:15], v0, s[44:47], 0 offen // (op 929) Store value




s_lshl_b32 s0, s2, 7 // (op 942) 
s_mul_i32 s1, 3, s0 // (op 942) 

v_add_u32 v0, v6, s1 // (op 942) 


buffer_load_dwordx4 v[12:15], v0, s[40:43], 0 offen // (op 942) Load value

v_accvgpr_read v16, a12 // (op 939) 
v_accvgpr_read v17, a13 // (op 939) 
v_accvgpr_read v18, a14 // (op 939) 
v_accvgpr_read v19, a15 // (op 939) 
v_mul_f32 v20, v11, v16 // (op 939) 
v_mul_f32 v21, v11, v17 // (op 939) 
v_mul_f32 v22, v11, v18 // (op 939) 
v_mul_f32 v23, v11, v19 // (op 939) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v10, v12 // (op 943) 
v_mul_f32 v17, v10, v13 // (op 943) 
v_mul_f32 v18, v10, v14 // (op 943) 
v_mul_f32 v19, v10, v15 // (op 943) 

v_add_f32 v12, v16, v20 // (op 940) 
v_add_f32 v13, v17, v21 // (op 940) 
v_add_f32 v14, v18, v22 // (op 940) 
v_add_f32 v15, v19, v23 // (op 940) 




s_lshl_b32 s0, s4, 7 // (op 941) 
s_mul_i32 s1, 3, s0 // (op 941) 

v_add_u32 v0, v9, s1 // (op 941) 


buffer_store_dwordx4 v[12:15], v0, s[44:47], 0 offen // (op 941) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ConditionalBottom_0_beta_123:












s_endpgm  // End of RR_GEMM_TN_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2
.LRR_GEMM_TN_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_end:
.size RR_GEMM_TN_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2, .LRR_GEMM_TN_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_end-RR_GEMM_TN_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2
  .amdhsa_next_free_vgpr 96
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 33792 // lds bytes
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
    DataType: F8
    DataTypeA: F8
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
  - .name: RR_GEMM_TN_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2
    .symbol: RR_GEMM_TN_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 33792
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 64
    .vgpr_count: 78
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

