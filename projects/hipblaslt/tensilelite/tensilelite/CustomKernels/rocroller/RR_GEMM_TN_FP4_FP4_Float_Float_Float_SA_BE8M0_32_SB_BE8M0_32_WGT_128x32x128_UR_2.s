

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2
.p2align 8
.type RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2,@function
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 6, v1 // (op 2121) 
v_and_b32 v3, 1, v0 // (op 2121) 

v_and_b32 v0, 15, v1 // (op 2121) 
v_lshl_add_u32 v4, v3, 4, v0 // (op 2121) 


v_and_b32 v0, 63, v1 // (op 2121) 
v_lshrrev_b32 v3, 4, v0 // (op 2121) 

v_lshlrev_b32 v0, 5, v3 // (op 2121) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 2121) 


v_lshlrev_b32 v0, 2, v3 // (op 2121) 

v_lshrrev_b32 v3, 3, v0 // (op 2121) 

v_lshrrev_b32 v0, 6, v1 // (op 2250) 
v_and_b32 v4, 1, v0 // (op 2250) 

v_and_b32 v0, 15, v1 // (op 2250) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 2250) 


v_and_b32 v0, 63, v1 // (op 2250) 
v_lshrrev_b32 v4, 4, v0 // (op 2250) 

v_lshlrev_b32 v0, 5, v4 // (op 2250) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 2250) 


v_lshlrev_b32 v0, 2, v4 // (op 2250) 

v_lshrrev_b32 v4, 3, v0 // (op 2250) 

v_lshrrev_b32 v0, 7, v1 // (op 2006) 
v_and_b32 v5, 63, v1 // (op 2006) 
v_lshrrev_b32 v6, 4, v5 // (op 2006) 
v_lshlrev_b32 v7, 2, v6 // (op 2006) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 2006) 


v_lshl_add_u32 v0, s16, 7, v6 // (op 2006) 

v_lshrrev_b32 v6, 2, v5 // (op 2006) 

v_and_b32 v5, 3, v6 // (op 2006) 

v_and_b32 v6, 3, v1 // (op 2006) 
v_lshl_add_u32 v7, v5, 2, v6 // (op 2006) 


v_lshrrev_b32 v5, 6, v1 // (op 2006) 
v_and_b32 v6, 1, v5 // (op 2006) 

v_lshl_add_u32 v5, v6, 4, v7 // (op 2006) 


v_lshl_add_u32 v6, s17, 5, v5 // (op 2006) 

v_mul_lo_u32 v5, v6, s2 // (op 2006) 

v_add_lshl_u32 v6, v0, v5, 2 // (op 2006) 


v_lshrrev_b32 v0, 7, v1 // (op 2068) 
v_and_b32 v5, 15, v1 // (op 2068) 
v_lshl_add_u32 v7, v0, 4, v5 // (op 2068) 


v_and_b32 v0, 63, v1 // (op 2068) 
v_lshrrev_b32 v5, 4, v0 // (op 2068) 

v_lshlrev_b32 v0, 5, v5 // (op 2068) 

v_lshl_add_u32 v5, v7, 7, v0 // (op 2068) 


v_lshlrev_b32 v0, 2, v5 // (op 2068) 

v_lshrrev_b32 v5, 3, v0 // (op 2068) 

v_lshrrev_b32 v0, 7, v1 // (op 2179) 
v_and_b32 v7, 15, v1 // (op 2179) 
s_mul_i32 s1, s2, s9 // (op 2008) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 2008) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 2008) least significant
s_mul_hi_u32 s19, s2, s8 // (op 2008) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2008) most significant: sum
s_add_u32 s1, s1, s19 // (op 2008) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 2008) 

v_lshl_add_u32 v8, v0, 4, v7 // (op 2179) 


v_and_b32 v0, 63, v1 // (op 2179) 
v_lshrrev_b32 v7, 4, v0 // (op 2179) 

v_lshl_add_u32 v0, v8, 2, v7 // (op 2179) 


v_lshrrev_b32 v7, 7, v1 // (op 2195) 
v_and_b32 v8, 63, v1 // (op 2195) 
v_lshrrev_b32 v9, 4, v8 // (op 2195) 
v_lshlrev_b32 v10, 2, v9 // (op 2195) 

v_lshl_add_u32 v9, v7, 4, v10 // (op 2195) 


v_lshl_add_u32 v7, s16, 7, v9 // (op 2195) 

v_lshrrev_b32 v9, 2, v8 // (op 2195) 

v_and_b32 v8, 3, v9 // (op 2195) 

v_and_b32 v9, 3, v1 // (op 2195) 
v_lshl_add_u32 v10, v8, 2, v9 // (op 2195) 


v_lshrrev_b32 v8, 6, v1 // (op 2195) 
v_and_b32 v9, 1, v8 // (op 2195) 

v_lshl_add_u32 v8, v9, 4, v10 // (op 2195) 


v_lshl_add_u32 v9, s17, 5, v8 // (op 2195) 

v_mul_lo_u32 v8, v9, s4 // (op 2195) 

v_add_lshl_u32 v9, v7, v8, 2 // (op 2195) 


v_lshrrev_b32 v7, 7, v1 // (op 2226) 
v_and_b32 v8, 15, v1 // (op 2226) 
v_lshl_add_u32 v10, v7, 4, v8 // (op 2226) 


v_and_b32 v7, 63, v1 // (op 2226) 
v_lshrrev_b32 v8, 4, v7 // (op 2226) 

v_lshlrev_b32 v7, 5, v8 // (op 2226) 

v_lshl_add_u32 v8, v10, 7, v7 // (op 2226) 


v_lshlrev_b32 v7, 2, v8 // (op 2226) 

v_lshrrev_b32 v8, 3, v7 // (op 2226) 

v_lshrrev_b32 v7, 7, v1 // (op 2279) 
v_and_b32 v10, 15, v1 // (op 2279) 
s_mul_i32 s1, s4, s9 // (op 2197) most significant: low of low * high
v_lshl_add_u32 v11, v7, 4, v10 // (op 2279) 


v_and_b32 v7, 63, v1 // (op 2279) 
v_lshrrev_b32 v10, 4, v7 // (op 2279) 

v_lshl_add_u32 v7, v11, 2, v10 // (op 2279) 


s_mul_i32 s38, s5, s8 // (op 2197) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 2197) least significant
s_mul_hi_u32 s39, s4, s8 // (op 2197) most significant: high of low * low
s_add_u32 s1, s1, s38 // (op 2197) most significant: sum
s_add_u32 s1, s1, s39 // (op 2197) most significant: sum


s_lshl_b64 s[38:39], s[0:1], 2 // (op 2197) 

s_mov_b32 s40, s6 // (op 2008) reg expression
s_mov_b32 s41, s7 // (op 2008) reg expression
s_mov_b32 s42, s18 // (op 2008) reg expression
s_mov_b32 s43, 131072 // (op 2008) call()

s_mov_b32 s44, s10 // (op 2197) reg expression

s_mov_b32 s45, s11 // (op 2197) reg expression
s_mov_b32 s46, s38 // (op 2197) reg expression
s_mov_b32 s47, 131072 // (op 2197) call()


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
v_lshrrev_b32 v12, 2, v1 // (op 1773) 
v_lshlrev_b32 v13, 1, v12 // (op 1773) 

v_lshl_add_u32 v12, s16, 7, v13 // (op 1773) 

v_and_b32 v13, -4, v1 // (op 1773) 
v_sub_u32 v14, v1, v13 // (op 1773) 

v_lshlrev_b32 v13, 5, v14 // (op 1773) 

v_mov_b32 v15, 0 // (op 1773) convert
v_mov_b32 v14, v13 // (op 1773) convert

v_mul_lo_u32 v17, v12, s13 // (op 1773) most significant: low of low * high
v_mul_hi_u32 v13, v12, s12 // (op 1773) most significant: high of low * low
v_mul_lo_u32 v16, v12, s12 // (op 1773) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 1773) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 1773) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 1773) most significant half



v_lshlrev_b64 v[12:13], 2, v[16:17] // (op 1773) 

v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 1773) 

v_mov_b32 v12, v14 // (op 1773) convert


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 1775) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 1775) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 1775) least significant
s_mul_hi_u32 s7, s12, s22 // (op 1775) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1775) most significant: sum
s_add_u32 s1, s1, s7 // (op 1775) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 1775) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 1775) 

s_mov_b32 s48, s20 // (op 1775) reg expression
s_mov_b32 s49, s21 // (op 1775) reg expression
s_mov_b32 s50, s0 // (op 1775) reg expression
s_mov_b32 s51, 131072 // (op 1775) call()

v_lshrrev_b32 v13, 2, v1 // (op 1792) 
v_lshlrev_b32 v14, 1, v13 // (op 1792) 

v_lshl_add_u32 v13, s16, 7, v14 // (op 1792) 

v_and_b32 v14, -4, v1 // (op 1792) 
v_sub_u32 v15, v1, v14 // (op 1792) 

v_lshlrev_b32 v14, 5, v15 // (op 1792) 

v_mov_b32 v17, 0 // (op 1792) convert
v_mov_b32 v16, v14 // (op 1792) convert

v_mul_lo_u32 v15, v13, s13 // (op 1792) most significant: low of low * high
v_mul_hi_u32 v18, v13, s12 // (op 1792) most significant: high of low * low
v_mul_lo_u32 v14, v13, s12 // (op 1792) least significant: low of low * low
v_add_u32 v15, v15, v18 // (op 1792) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v16 // (op 1792) least significant half
v_addc_co_u32 v15, s[0:1], v15, v17, s[0:1] // (op 1792) most significant half



v_lshlrev_b64 v[16:17], 2, v[14:15] // (op 1792) 

v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 1792) 

v_mov_b32 v13, v14 // (op 1792) convert

s_mul_i32 s1, s12, s23 // (op 1794) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 1794) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 1794) least significant
s_mul_hi_u32 s7, s12, s22 // (op 1794) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1794) most significant: sum
s_add_u32 s1, s1, s7 // (op 1794) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 1794) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 1794) 

s_mov_b32 s48, s20 // (op 1794) reg expression
s_mov_b32 s49, s21 // (op 1794) reg expression
s_mov_b32 s50, s0 // (op 1794) reg expression
s_mov_b32 s51, 131072 // (op 1794) call()

v_and_b32 v14, -8, v1 // (op 1837) 
v_sub_u32 v15, v1, v14 // (op 1837) 

v_lshlrev_b32 v14, 4, v15 // (op 1837) 

v_mov_b32 v17, 0 // (op 1837) convert
v_mov_b32 v16, v14 // (op 1837) convert

v_lshrrev_b32 v14, 3, v1 // (op 1837) 
v_lshl_add_u32 v15, s17, 5, v14 // (op 1837) 

v_mul_lo_u32 v19, v15, s25 // (op 1837) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 1837) most significant: high of low * low
v_mul_lo_u32 v18, v15, s24 // (op 1837) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 1837) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v18 // (op 1837) least significant half
v_addc_co_u32 v15, s[0:1], v17, v19, s[0:1] // (op 1837) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 1837) 


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 1837) 

v_mov_b32 v14, v16 // (op 1837) convert

s_mul_i32 s1, s24, s9 // (op 1839) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 1839) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 1839) least significant
s_mul_hi_u32 s7, s24, s8 // (op 1839) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1839) most significant: sum
s_add_u32 s1, s1, s7 // (op 1839) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 1839) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 1839) 

s_mov_b32 s52, s26 // (op 1839) reg expression
s_mov_b32 s53, s27 // (op 1839) reg expression
s_mov_b32 s54, s0 // (op 1839) reg expression
s_mov_b32 s55, 131072 // (op 1839) call()

v_and_b32 v15, -8, v1 // (op 1856) 
v_sub_u32 v16, v1, v15 // (op 1856) 

v_lshlrev_b32 v15, 4, v16 // (op 1856) 

v_mov_b32 v17, 0 // (op 1856) convert
v_mov_b32 v16, v15 // (op 1856) convert

v_lshrrev_b32 v15, 3, v1 // (op 1856) 
v_lshl_add_u32 v18, s17, 5, v15 // (op 1856) 

v_mul_lo_u32 v21, v18, s25 // (op 1856) most significant: low of low * high
v_mul_hi_u32 v15, v18, s24 // (op 1856) most significant: high of low * low
v_mul_lo_u32 v20, v18, s24 // (op 1856) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 1856) most significant: sum


v_add_co_u32 v18, s[0:1], v16, v20 // (op 1856) least significant half
v_addc_co_u32 v19, s[0:1], v17, v21, s[0:1] // (op 1856) most significant half

v_lshlrev_b64 v[18:19], 2, v[18:19] // (op 1856) 


v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 1856) 

v_mov_b32 v15, v16 // (op 1856) convert

s_mul_i32 s1, s24, s9 // (op 1858) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 1858) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 1858) least significant
s_mul_hi_u32 s7, s24, s8 // (op 1858) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1858) most significant: sum
s_add_u32 s1, s1, s7 // (op 1858) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 1858) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 1858) 

s_mov_b32 s52, s26 // (op 1858) reg expression
s_mov_b32 s53, s27 // (op 1858) reg expression
s_mov_b32 s54, s0 // (op 1858) reg expression
s_mov_b32 s55, 131072 // (op 1858) call()

v_lshrrev_b32 v16, 2, v1 // (op 1901) 
v_lshlrev_b32 v17, 1, v16 // (op 1901) 

v_lshl_add_u32 v16, s16, 7, v17 // (op 1901) 

v_and_b32 v17, -4, v1 // (op 1901) 
v_sub_u32 v18, v1, v17 // (op 1901) 

v_mul_lo_u32 v17, v16, s28 // (op 1901) 
v_add_u32 v17, v17, v18 // (op 1901) 


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

v_lshrrev_b32 v16, 2, v1 // (op 1920) 
v_lshlrev_b32 v18, 1, v16 // (op 1920) 

v_lshl_add_u32 v16, s16, 7, v18 // (op 1920) 

v_and_b32 v18, -4, v1 // (op 1920) 
v_sub_u32 v19, v1, v18 // (op 1920) 

v_mul_lo_u32 v18, v16, s28 // (op 1920) 
v_add_u32 v18, v18, v19 // (op 1920) 


s_mul_i32 s1, s28, s23 // (op 1922) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 1922) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 1922) least significant
s_mul_hi_u32 s7, s28, s22 // (op 1922) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1922) most significant: sum
s_add_u32 s1, s1, s7 // (op 1922) most significant: sum


s_mov_b32 s56, s30 // (op 1922) reg expression
s_mov_b32 s57, s31 // (op 1922) reg expression
s_mov_b32 s58, s0 // (op 1922) reg expression
s_mov_b32 s59, 131072 // (op 1922) call()

v_lshrrev_b32 v16, 6, v1 // (op 1965) 
v_and_b32 v19, 1, v16 // (op 1965) 

v_and_b32 v16, 15, v1 // (op 1965) 
v_lshl_add_u32 v20, v19, 4, v16 // (op 1965) 


v_lshl_add_u32 v16, s17, 5, v20 // (op 1965) 

v_mul_lo_u32 v19, v16, s32 // (op 1965) 

v_and_b32 v16, 63, v1 // (op 1965) 
v_lshrrev_b32 v20, 4, v16 // (op 1965) 

v_add_u32 v16, v20, v19 // (op 1965) 


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

v_lshrrev_b32 v19, 6, v1 // (op 1984) 
v_and_b32 v20, 1, v19 // (op 1984) 

v_and_b32 v19, 15, v1 // (op 1984) 
v_lshl_add_u32 v21, v20, 4, v19 // (op 1984) 


v_lshl_add_u32 v19, s17, 5, v21 // (op 1984) 

v_mul_lo_u32 v20, v19, s32 // (op 1984) 

v_and_b32 v19, 63, v1 // (op 1984) 
v_lshrrev_b32 v21, 4, v19 // (op 1984) 

v_add_u32 v19, v21, v20 // (op 1984) 


s_mul_i32 s1, s32, s9 // (op 1986) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 1986) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 1986) least significant
s_mul_hi_u32 s7, s32, s8 // (op 1986) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1986) most significant: sum
s_add_u32 s1, s1, s7 // (op 1986) most significant: sum


s_mov_b32 s60, s34 // (op 1986) reg expression
s_mov_b32 s61, s35 // (op 1986) reg expression
s_mov_b32 s62, s0 // (op 1986) reg expression
s_mov_b32 s63, 131072 // (op 1986) call()




buffer_load_ubyte v20, v16, s[60:63], 0 offen offset: 0 // (op 1598) Load value



buffer_load_dwordx2 v[22:23], v14, s[52:55], 0 offen // (op 1602) Load value

s_lshl_b64 s[0:1], s[12:13], 2 // (op 1605) 
v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 1605) 


v_mov_b32 v21, v12 // (op 1605) 
buffer_load_dwordx4 v[26:29], v21, s[48:51], 0 offen // (op 1605) Load value
v_add_u32 v21, v21, v24 // (op 1605) 
buffer_load_dwordx4 v[30:33], v21, s[48:51], 0 offen // (op 1605) Load value



v_mov_b64 v[24:25], s[28:29] // (op 1608) convert

v_mov_b32 v21, v17 // (op 1608) 
buffer_load_ubyte v34, v21, s[56:59], 0 offen offset: 0 // (op 1608) Load value
v_add_u32 v21, v21, v24 // (op 1608) 
buffer_load_ubyte v35, v21, s[56:59], 0 offen offset: 0 // (op 1608) Load value





buffer_load_ubyte v21, v16, s[60:63], 0 offen offset:4 // (op 1612) Load value




buffer_load_dwordx2 v[24:25], v14, s[52:55], 0 offen offset:64 // (op 1616) Load value


s_lshl_b64 s[0:1], s[12:13], 2 // (op 1619) 
v_lshrrev_b64 v[36:37], 3, s[0:1] // (op 1619) 


v_mov_b32 v14, v12 // (op 1619) 
buffer_load_dwordx4 v[38:41], v14, s[48:51], 0 offen offset:64 // (op 1619) Load value
v_add_u32 v14, v14, v36 // (op 1619) 
buffer_load_dwordx4 v[42:45], v14, s[48:51], 0 offen offset:64 // (op 1619) Load value




v_mov_b64 v[36:37], s[28:29] // (op 1622) convert

v_mov_b32 v12, v17 // (op 1622) 
buffer_load_ubyte v46, v12, s[56:59], 0 offen offset:4 // (op 1622) Load value
v_add_u32 v12, v12, v36 // (op 1622) 
buffer_load_ubyte v47, v12, s[56:59], 0 offen offset:4 // (op 1622) Load value



v_and_b32 v12, -8, v1 // (op 2109) 
v_sub_u32 v14, v1, v12 // (op 2109) 

v_lshlrev_b32 v12, 4, v14 // (op 2109) 

v_lshrrev_b32 v14, 3, v1 // (op 2109) 
v_lshl_add_u32 v16, v14, 7, v12 // (op 2109) 


v_lshlrev_b32 v12, 2, v16 // (op 2109) 

v_lshrrev_b32 v14, 3, v12 // (op 2109) 




s_waitcnt vmcnt(10)
ds_write_b64 v14, v[22:23] // (op 1624) Store local data 

v_and_b32 v12, -4, v1 // (op 2051) 
v_sub_u32 v14, v1, v12 // (op 2051) 

v_lshlrev_b32 v12, 5, v14 // (op 2051) 

v_lshrrev_b32 v14, 2, v1 // (op 2051) 
v_lshl_add_u32 v16, v14, 8, v12 // (op 2051) 


v_lshlrev_b32 v12, 2, v16 // (op 2051) 

v_lshrrev_b32 v14, 3, v12 // (op 2051) 




s_waitcnt vmcnt(9)
ds_write_b128 v14, v[26:29] offset:2048 // (op 1625) Store local data 
s_waitcnt vmcnt(8)
ds_write_b128 v14, v[30:33] offset:2112 // (op 1625) Store local data 

v_and_b32 v12, -4, v1 // (op 2162) 
v_sub_u32 v14, v1, v12 // (op 2162) 

v_lshrrev_b32 v12, 2, v1 // (op 2162) 
v_lshl_add_u32 v16, v12, 3, v14 // (op 2162) 





s_waitcnt vmcnt(7)
ds_write_b8 v16, v34 offset:10240 // (op 1626) Store local data 
s_waitcnt vmcnt(6)
ds_write_b8 v16, v35 offset:10244 // (op 1626) Store local data 

v_and_b32 v12, -4, v1 // (op 2022) 
v_sub_u32 v14, v1, v12 // (op 2022) 

v_lshlrev_b32 v12, 5, v14 // (op 2022) 

v_lshrrev_b32 v14, 2, v1 // (op 2022) 
v_lshl_add_u32 v16, v14, 8, v12 // (op 2022) 


v_lshlrev_b32 v12, 2, v16 // (op 2022) 

v_lshrrev_b32 v14, 3, v12 // (op 2022) 

v_and_b32 v12, -8, v1 // (op 2080) 
v_sub_u32 v16, v1, v12 // (op 2080) 

v_lshlrev_b32 v12, 4, v16 // (op 2080) 

v_lshrrev_b32 v16, 3, v1 // (op 2080) 
v_lshl_add_u32 v17, v16, 7, v12 // (op 2080) 


v_lshlrev_b32 v12, 2, v17 // (op 2080) 

v_lshrrev_b32 v16, 3, v12 // (op 2080) 

v_and_b32 v12, -4, v1 // (op 2133) 
v_sub_u32 v17, v1, v12 // (op 2133) 

v_lshrrev_b32 v12, 2, v1 // (op 2133) 
v_lshl_add_u32 v36, v12, 3, v17 // (op 2133) 


v_and_b32 v12, -4, v1 // (op 2209) 
v_sub_u32 v17, v1, v12 // (op 2209) 

v_lshlrev_b32 v12, 5, v17 // (op 2209) 

v_lshrrev_b32 v17, 2, v1 // (op 2209) 
v_lshl_add_u32 v37, v17, 8, v12 // (op 2209) 


v_lshlrev_b32 v12, 2, v37 // (op 2209) 

v_lshrrev_b32 v17, 3, v12 // (op 2209) 

v_and_b32 v12, -8, v1 // (op 2238) 
v_sub_u32 v37, v1, v12 // (op 2238) 

v_lshlrev_b32 v12, 4, v37 // (op 2238) 

v_lshrrev_b32 v37, 3, v1 // (op 2238) 
v_lshl_add_u32 v48, v37, 7, v12 // (op 2238) 


v_lshlrev_b32 v12, 2, v48 // (op 2238) 

v_lshrrev_b32 v37, 3, v12 // (op 2238) 

v_and_b32 v12, -4, v1 // (op 2262) 
v_sub_u32 v48, v1, v12 // (op 2262) 

v_lshrrev_b32 v12, 2, v1 // (op 2262) 
v_lshl_add_u32 v49, v12, 3, v48 // (op 2262) 


s_mov_b32 s0, 0 // (op 635) call()
s_ashr_i64 s[6:7], s[36:37], 7 // (op 634) 
s_mov_b32 s1, s6 // (op 634) convert

s_and_b32 s6, s1, 4294967294 // (op 634) 

s_cmp_lt_u32 s0, s6 // (op 634) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopBottom_KLoop_634 // (op 634) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopBottom_KLoop_634 if false)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopTop_KLoop_634:
s_add_u32 s0, s0, 2 // (op 636) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 2326) 



ds_read_b128 v[50:53], v5 offset:2048 // (op 65) Load local data 
s_barrier  // (op 2339) 



ds_read_b128 v[54:57], v3 // (op 70) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 2353) 



ds_read_u8 v12, v0 offset:10240 // (op 75) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[50:53], v[54:57], a[0:3], v12, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 82) 
ds_read_b128 v[58:61], v5 offset:4096 // (op 340) Load local data 





ds_read_u8 v12, v0 offset:10368 // (op 344) Load local data 



ds_read_b128 v[50:53], v5 offset:6144 // (op 502) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[58:61], v[54:57], a[4:7], v12, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 346) 





ds_read_u8 v12, v0 offset:10496 // (op 506) Load local data 



ds_read_b128 v[58:61], v5 offset:8192 // (op 664) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[50:53], v[54:57], a[8:11], v12, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 508) 





ds_read_u8 v12, v0 offset:10624 // (op 668) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[58:61], v[54:57], a[12:15], v12, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 670) 





v_mov_b32 v12, v19 // (op 80) 

buffer_load_ubyte v20, v12, s[60:63], 0 offen offset:8 // (op 80) Load value


v_mov_b32 v48, v15 // (op 10) 

buffer_load_dwordx2 v[22:23], v48, s[52:55], 0 offen offset:128 // (op 10) Load value

s_lshl_b64 s[6:7], s[12:13], 2 // (op 4) 
v_lshrrev_b64 v[50:51], 3, s[6:7] // (op 4) 


v_mov_b32 v52, v13 // (op 4) 
v_mov_b32 v53, v52 // (op 4) 
buffer_load_dwordx4 v[26:29], v53, s[48:51], 0 offen offset:128 // (op 4) Load value
v_add_u32 v53, v53, v50 // (op 4) 
buffer_load_dwordx4 v[30:33], v53, s[48:51], 0 offen offset:128 // (op 4) Load value



v_mov_b64 v[50:51], s[28:29] // (op 17) convert

v_mov_b32 v53, v18 // (op 17) 
v_mov_b32 v54, v53 // (op 17) 
buffer_load_ubyte v34, v54, s[56:59], 0 offen offset:8 // (op 17) Load value
v_add_u32 v54, v54, v50 // (op 17) 
buffer_load_ubyte v35, v54, s[56:59], 0 offen offset:8 // (op 17) Load value


s_barrier  // (op 2376) 



s_waitcnt vmcnt(10)
ds_write_b64 v37, v[24:25] offset:10752 // (op 235) Store local data 
s_barrier  // (op 2370) 



s_waitcnt vmcnt(9)
ds_write_b128 v17, v[38:41] offset:12800 // (op 225) Store local data 
s_waitcnt vmcnt(8)
ds_write_b128 v17, v[42:45] offset:12864 // (op 225) Store local data 
s_barrier  // (op 2382) 



s_waitcnt vmcnt(7)
ds_write_b8 v49, v46 offset:20992 // (op 236) Store local data 
s_waitcnt vmcnt(6)
ds_write_b8 v49, v47 offset:20996 // (op 236) Store local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 2367) 



ds_read_b128 v[54:57], v8 offset:12800 // (op 227) Load local data 
s_barrier  // (op 2373) 



ds_read_b128 v[58:61], v4 offset:10752 // (op 229) Load local data 
s_barrier  // (op 2419) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 2379) 



ds_read_u8 v50, v7 offset:20992 // (op 231) Load local data 



ds_read_b128 v[62:65], v8 offset:14848 // (op 353) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[54:57], v[58:61], a[0:3], v50, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 234) 





ds_read_u8 v50, v7 offset:21120 // (op 357) Load local data 



ds_read_b128 v[54:57], v8 offset:16896 // (op 515) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[62:65], v[58:61], a[4:7], v50, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 347) 





ds_read_u8 v50, v7 offset:21248 // (op 519) Load local data 



ds_read_b128 v[62:65], v8 offset:18944 // (op 677) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[54:57], v[58:61], a[8:11], v50, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 509) 





s_barrier  // (op 2465) 
ds_read_u8 v50, v7 offset:21376 // (op 681) Load local data 
s_barrier  // (op 2469) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[62:65], v[58:61], a[12:15], v50, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 671) 






buffer_load_ubyte v21, v12, s[60:63], 0 offen offset:12 // (op 233) Load value




v_add_u32 v19, 8, v19 // (op 1976) 
buffer_load_dwordx2 v[24:25], v48, s[52:55], 0 offen offset:192 // (op 237) Load value


v_add_u32 v15, 128, v15 // (op 1848) 
s_lshl_b64 s[6:7], s[12:13], 2 // (op 224) 
v_lshrrev_b64 v[50:51], 3, s[6:7] // (op 224) 


v_mov_b32 v12, v52 // (op 224) 
buffer_load_dwordx4 v[38:41], v12, s[48:51], 0 offen offset:192 // (op 224) Load value
v_add_u32 v12, v12, v50 // (op 224) 
buffer_load_dwordx4 v[42:45], v12, s[48:51], 0 offen offset:192 // (op 224) Load value




v_add_u32 v13, 128, v13 // (op 1784) 
v_mov_b64 v[50:51], s[28:29] // (op 238) convert

v_mov_b32 v12, v53 // (op 238) 
buffer_load_ubyte v46, v12, s[56:59], 0 offen offset:12 // (op 238) Load value
v_add_u32 v12, v12, v50 // (op 238) 
buffer_load_ubyte v47, v12, s[56:59], 0 offen offset:12 // (op 238) Load value



s_barrier  // (op 2342) 
v_add_u32 v18, 8, v18 // (op 1912) 



s_waitcnt vmcnt(10)
ds_write_b64 v16, v[22:23] // (op 13) Store local data 
s_barrier  // (op 2329) 



s_waitcnt vmcnt(9)
ds_write_b128 v14, v[26:29] offset:2048 // (op 6) Store local data 
s_waitcnt vmcnt(8)
ds_write_b128 v14, v[30:33] offset:2112 // (op 6) Store local data 
s_barrier  // (op 2356) 



s_waitcnt vmcnt(7)
ds_write_b8 v36, v34 offset:10240 // (op 21) Store local data 
s_waitcnt vmcnt(6)
ds_write_b8 v36, v35 offset:10244 // (op 21) Store local data 
s_ashr_i64 s[6:7], s[36:37], 7 // (op 634) 
s_mov_b32 s1, s6 // (op 634) convert

s_and_b32 s6, s1, 4294967294 // (op 634) 

s_cmp_lt_u32 s0, s6 // (op 634) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopTop_KLoop_634 // (op 634) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopTop_KLoop_634 if true)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopBottom_KLoop_634:















v_lshrrev_b32 v12, 2, v1 // (op 1744) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1744) 
v_lshlrev_b32 v13, 1, v12 // (op 1744) 

s_lshr_b32 s6, s0, 1 // (op 1744) 

v_lshl_add_u32 v12, s16, 7, v13 // (op 1744) 

v_and_b32 v13, -4, v1 // (op 1744) 
v_sub_u32 v14, v1, v13 // (op 1744) 

v_lshlrev_b32 v13, 5, v14 // (op 1744) 

v_lshl_add_u32 v14, s6, 8, v13 // (op 1744) 


v_mov_b32 v23, 0 // (op 1744) convert
v_mov_b32 v22, v14 // (op 1744) convert

v_mul_lo_u32 v27, v12, s13 // (op 1744) most significant: low of low * high
v_mul_hi_u32 v13, v12, s12 // (op 1744) most significant: high of low * low
v_mul_lo_u32 v26, v12, s12 // (op 1744) least significant: low of low * low
v_add_u32 v27, v27, v13 // (op 1744) most significant: sum

v_add_co_u32 v26, s[0:1], v26, v22 // (op 1744) least significant half
v_addc_co_u32 v27, s[0:1], v27, v23, s[0:1] // (op 1744) most significant half



v_lshlrev_b64 v[22:23], 2, v[26:27] // (op 1744) 

v_lshrrev_b64 v[12:13], 3, v[22:23] // (op 1744) 

v_mov_b32 v14, v12 // (op 1744) convert

s_mul_i32 s1, s12, s23 // (op 1746) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 1746) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 1746) least significant
s_mul_hi_u32 s7, s12, s22 // (op 1746) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1746) most significant: sum
s_add_u32 s1, s1, s7 // (op 1746) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 1746) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 1746) 

s_mov_b32 s48, s20 // (op 1746) reg expression
s_mov_b32 s49, s21 // (op 1746) reg expression
s_mov_b32 s50, s0 // (op 1746) reg expression
s_mov_b32 s51, 131072 // (op 1746) call()

v_and_b32 v12, -8, v1 // (op 1814) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1814) 
v_sub_u32 v13, v1, v12 // (op 1814) 

s_lshr_b32 s6, s0, 1 // (op 1814) 

v_lshlrev_b32 v12, 4, v13 // (op 1814) 

v_lshl_add_u32 v13, s6, 8, v12 // (op 1814) 


v_mov_b32 v23, 0 // (op 1814) convert
v_mov_b32 v22, v13 // (op 1814) convert

v_lshrrev_b32 v12, 3, v1 // (op 1814) 
v_lshl_add_u32 v13, s17, 5, v12 // (op 1814) 

v_mul_lo_u32 v27, v13, s25 // (op 1814) most significant: low of low * high
v_mul_hi_u32 v12, v13, s24 // (op 1814) most significant: high of low * low
v_mul_lo_u32 v26, v13, s24 // (op 1814) least significant: low of low * low
v_add_u32 v27, v27, v12 // (op 1814) most significant: sum


v_add_co_u32 v12, s[0:1], v22, v26 // (op 1814) least significant half
v_addc_co_u32 v13, s[0:1], v23, v27, s[0:1] // (op 1814) most significant half

v_lshlrev_b64 v[12:13], 2, v[12:13] // (op 1814) 


v_lshrrev_b64 v[22:23], 3, v[12:13] // (op 1814) 

v_mov_b32 v12, v22 // (op 1814) convert

s_mul_i32 s1, s24, s9 // (op 1816) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 1816) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 1816) least significant
s_mul_hi_u32 s7, s24, s8 // (op 1816) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1816) most significant: sum
s_add_u32 s1, s1, s7 // (op 1816) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 1816) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 1816) 

s_mov_b32 s52, s26 // (op 1816) reg expression
s_mov_b32 s53, s27 // (op 1816) reg expression
s_mov_b32 s54, s0 // (op 1816) reg expression
s_mov_b32 s55, 131072 // (op 1816) call()

v_lshrrev_b32 v13, 2, v1 // (op 1878) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1878) 
v_lshlrev_b32 v15, 1, v13 // (op 1878) 

s_lshr_b32 s6, s0, 1 // (op 1878) 

v_lshl_add_u32 v13, s16, 7, v15 // (op 1878) 

v_and_b32 v15, -4, v1 // (op 1878) 
v_sub_u32 v16, v1, v15 // (op 1878) 

v_lshl_add_u32 v15, s6, 3, v16 // (op 1878) 


v_mul_lo_u32 v16, v13, s28 // (op 1878) 
v_add_u32 v16, v16, v15 // (op 1878) 


s_mul_i32 s1, s28, s23 // (op 1880) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 1880) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 1880) least significant
s_mul_hi_u32 s7, s28, s22 // (op 1880) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1880) most significant: sum
s_add_u32 s1, s1, s7 // (op 1880) most significant: sum


s_mov_b32 s56, s30 // (op 1880) reg expression
s_mov_b32 s57, s31 // (op 1880) reg expression
s_mov_b32 s58, s0 // (op 1880) reg expression
s_mov_b32 s59, 131072 // (op 1880) call()

v_and_b32 v13, 63, v1 // (op 1942) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1942) 
v_lshrrev_b32 v15, 4, v13 // (op 1942) 

s_lshr_b32 s6, s0, 1 // (op 1942) 

v_lshrrev_b32 v13, 6, v1 // (op 1942) 
v_lshl_add_u32 v17, s6, 3, v15 // (op 1942) 


v_and_b32 v15, 1, v13 // (op 1942) 

v_and_b32 v13, 15, v1 // (op 1942) 
v_lshl_add_u32 v18, v15, 4, v13 // (op 1942) 


v_lshl_add_u32 v13, s17, 5, v18 // (op 1942) 

v_mul_lo_u32 v15, v13, s32 // (op 1942) 

v_add_u32 v13, v17, v15 // (op 1942) 


s_mul_i32 s1, s32, s9 // (op 1944) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 1944) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 1944) least significant
s_mul_hi_u32 s7, s32, s8 // (op 1944) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1944) most significant: sum
s_add_u32 s1, s1, s7 // (op 1944) most significant: sum


s_mov_b32 s60, s34 // (op 1944) reg expression
s_mov_b32 s61, s35 // (op 1944) reg expression
s_mov_b32 s62, s0 // (op 1944) reg expression
s_mov_b32 s63, 131072 // (op 1944) call()

v_and_b32 v15, -4, v1 // (op 2035) 
v_sub_u32 v17, v1, v15 // (op 2035) 

v_lshlrev_b32 v15, 5, v17 // (op 2035) 

v_lshrrev_b32 v17, 2, v1 // (op 2035) 
v_lshl_add_u32 v18, v17, 8, v15 // (op 2035) 


v_lshlrev_b32 v15, 2, v18 // (op 2035) 

v_lshrrev_b32 v17, 3, v15 // (op 2035) 

v_and_b32 v15, -8, v1 // (op 2093) 
v_sub_u32 v18, v1, v15 // (op 2093) 

v_lshlrev_b32 v15, 4, v18 // (op 2093) 

v_lshrrev_b32 v18, 3, v1 // (op 2093) 
v_lshl_add_u32 v19, v18, 7, v15 // (op 2093) 


v_lshlrev_b32 v15, 2, v19 // (op 2093) 

v_lshrrev_b32 v18, 3, v15 // (op 2093) 

v_and_b32 v15, -4, v1 // (op 2146) 
v_sub_u32 v19, v1, v15 // (op 2146) 

v_lshrrev_b32 v15, 2, v1 // (op 2146) 
v_lshl_add_u32 v20, v15, 3, v19 // (op 2146) 


s_ashr_i64 s[0:1], s[36:37], 7 // (op 640) 
s_and_b32 s6, s0, 4294967294 // (op 640) 

s_ashr_i64 s[0:1], s[36:37], 7 // (op 639) 
s_mov_b32 s7, s0 // (op 639) convert

s_cmp_lt_u32 s6, s7 // (op 639) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopBottom_KLoopTail_639 // (op 639) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopBottom_KLoopTail_639 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopTop_KLoopTail_639:

s_add_u32 s6, s6, 1 // (op 641) 
s_lshl_b64 s[0:1], s[12:13], 2 // (op 166) 
v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 166) 


v_mov_b32 v15, v14 // (op 166) 
v_mov_b32 v19, v15 // (op 166) 
buffer_load_dwordx4 v[24:27], v19, s[48:51], 0 offen // (op 166) Load value
v_add_u32 v19, v19, v22 // (op 166) 
buffer_load_dwordx4 v[28:31], v19, s[48:51], 0 offen // (op 166) Load value



v_add_u32 v14, 64, v14 // (op 1734) 


s_barrier  // (op 2332) 
v_mov_b32 v15, v12 // (op 179) 




buffer_load_dwordx2 v[22:23], v15, s[52:55], 0 offen // (op 179) Load value

v_add_u32 v12, 64, v12 // (op 1806) 
s_waitcnt vmcnt(2)
ds_write_b128 v17, v[24:27] offset:2048 // (op 167) Store local data 

v_mov_b64 v[32:33], s[28:29] // (op 180) convert

v_mov_b32 v15, v16 // (op 180) 
v_mov_b32 v19, v15 // (op 180) 
buffer_load_ubyte v34, v19, s[56:59], 0 offen offset: 0 // (op 180) Load value
v_add_u32 v19, v19, v32 // (op 180) 
buffer_load_ubyte v35, v19, s[56:59], 0 offen offset: 0 // (op 180) Load value



v_add_u32 v16, 4, v16 // (op 1870) 
s_waitcnt vmcnt(3)
ds_write_b128 v17, v[28:31] offset:2112 // (op 167) Store local data 

s_barrier  // (op 2345) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 2335) 






ds_read_b128 v[24:27], v5 offset:2048 // (op 169) Load local data 
s_waitcnt vmcnt(2)
ds_write_b64 v18, v[22:23] // (op 177) Store local data 

s_barrier  // (op 2359) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 2349) 






ds_read_b128 v[28:31], v3 // (op 171) Load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v20, v34 offset:10240 // (op 178) Store local data 
s_waitcnt vmcnt(0)
ds_write_b8 v20, v35 offset:10244 // (op 178) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 2363) 



ds_read_u8 v15, v0 offset:10240 // (op 173) Load local data 


v_mov_b32 v19, v13 // (op 175) 

buffer_load_ubyte v21, v19, s[60:63], 0 offen offset: 0 // (op 175) Load value

v_add_u32 v13, 4, v13 // (op 1934) 



ds_read_b128 v[32:35], v5 offset:4096 // (op 321) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[24:27], v[28:31], a[0:3], v15, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 176) 





ds_read_u8 v15, v0 offset:10368 // (op 325) Load local data 



ds_read_b128 v[22:25], v5 offset:6144 // (op 483) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[32:35], v[28:31], a[4:7], v15, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 327) 





ds_read_u8 v15, v0 offset:10496 // (op 487) Load local data 



ds_read_b128 v[32:35], v5 offset:8192 // (op 645) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[22:25], v[28:31], a[8:11], v15, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 489) 





ds_read_u8 v15, v0 offset:10624 // (op 649) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[32:35], v[28:31], a[12:15], v15, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 651) 




s_ashr_i64 s[0:1], s[36:37], 7 // (op 639) 
s_mov_b32 s7, s0 // (op 639) convert

s_cmp_lt_u32 s6, s7 // (op 639) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopTop_KLoopTail_639 // (op 639) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopTop_KLoopTail_639 if true)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopBottom_KLoopTail_639:






















s_barrier  // (op 2453) 







v_cmp_eq_f32 vcc, 0.00000, v10 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ConditionalFalse_0_beta_123
v_accvgpr_read v12, a0 // (op 46) 
v_accvgpr_read v13, a1 // (op 46) 
v_accvgpr_read v14, a2 // (op 46) 
v_accvgpr_read v15, a3 // (op 46) 
v_mul_f32 v16, v11, v12 // (op 46) 
v_mul_f32 v17, v11, v13 // (op 46) 
v_mul_f32 v18, v11, v14 // (op 46) 
v_mul_f32 v19, v11, v15 // (op 46) 




buffer_store_dwordx4 v[16:19], v9, s[44:47], 0 offen // (op 52) Store value

v_accvgpr_read v12, a4 // (op 881) 
v_accvgpr_read v13, a5 // (op 881) 
v_accvgpr_read v14, a6 // (op 881) 
v_accvgpr_read v15, a7 // (op 881) 
v_mul_f32 v16, v11, v12 // (op 881) 
v_mul_f32 v17, v11, v13 // (op 881) 
v_mul_f32 v18, v11, v14 // (op 881) 
v_mul_f32 v19, v11, v15 // (op 881) 




buffer_store_dwordx4 v[16:19], v9, s[44:47], 0 offen offset:128 // (op 883) Store value

v_accvgpr_read v12, a8 // (op 899) 
v_accvgpr_read v13, a9 // (op 899) 
v_accvgpr_read v14, a10 // (op 899) 
v_accvgpr_read v15, a11 // (op 899) 
v_mul_f32 v16, v11, v12 // (op 899) 
v_mul_f32 v17, v11, v13 // (op 899) 
v_mul_f32 v18, v11, v14 // (op 899) 
v_mul_f32 v19, v11, v15 // (op 899) 




buffer_store_dwordx4 v[16:19], v9, s[44:47], 0 offen offset:256 // (op 901) Store value

v_accvgpr_read v12, a12 // (op 917) 
v_accvgpr_read v13, a13 // (op 917) 
v_accvgpr_read v14, a14 // (op 917) 
v_accvgpr_read v15, a15 // (op 917) 
v_mul_f32 v16, v11, v12 // (op 917) 
v_mul_f32 v17, v11, v13 // (op 917) 
v_mul_f32 v18, v11, v14 // (op 917) 
v_mul_f32 v19, v11, v15 // (op 917) 




buffer_store_dwordx4 v[16:19], v9, s[44:47], 0 offen offset:384 // (op 919) Store value

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ConditionalBottom_0_beta_123
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ConditionalFalse_0_beta_123:



buffer_load_dwordx4 v[12:15], v6, s[40:43], 0 offen offset:256 // (op 975) Load value
v_accvgpr_read v16, a8 // (op 972) 
v_accvgpr_read v17, a9 // (op 972) 
v_accvgpr_read v18, a10 // (op 972) 
v_accvgpr_read v19, a11 // (op 972) 
v_mul_f32 v20, v11, v16 // (op 972) 
v_mul_f32 v21, v11, v17 // (op 972) 
v_mul_f32 v22, v11, v18 // (op 972) 
v_mul_f32 v23, v11, v19 // (op 972) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v10, v12 // (op 976) 
v_mul_f32 v17, v10, v13 // (op 976) 
v_mul_f32 v18, v10, v14 // (op 976) 
v_mul_f32 v19, v10, v15 // (op 976) 

v_add_f32 v12, v16, v20 // (op 973) 
v_add_f32 v13, v17, v21 // (op 973) 
v_add_f32 v14, v18, v22 // (op 973) 
v_add_f32 v15, v19, v23 // (op 973) 





buffer_store_dwordx4 v[12:15], v9, s[44:47], 0 offen offset:256 // (op 974) Store value




s_nop 1
buffer_load_dwordx4 v[12:15], v6, s[40:43], 0 offen offset:128 // (op 957) Load value
v_accvgpr_read v16, a4 // (op 954) 
v_accvgpr_read v17, a5 // (op 954) 
v_accvgpr_read v18, a6 // (op 954) 
v_accvgpr_read v19, a7 // (op 954) 
v_mul_f32 v20, v11, v16 // (op 954) 
v_mul_f32 v21, v11, v17 // (op 954) 
v_mul_f32 v22, v11, v18 // (op 954) 
v_mul_f32 v23, v11, v19 // (op 954) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v10, v12 // (op 958) 
v_mul_f32 v17, v10, v13 // (op 958) 
v_mul_f32 v18, v10, v14 // (op 958) 
v_mul_f32 v19, v10, v15 // (op 958) 

v_add_f32 v12, v16, v20 // (op 955) 
v_add_f32 v13, v17, v21 // (op 955) 
v_add_f32 v14, v18, v22 // (op 955) 
v_add_f32 v15, v19, v23 // (op 955) 





buffer_store_dwordx4 v[12:15], v9, s[44:47], 0 offen offset:128 // (op 956) Store value




s_nop 1
buffer_load_dwordx4 v[12:15], v6, s[40:43], 0 offen offset:384 // (op 993) Load value
v_accvgpr_read v16, a12 // (op 990) 
v_accvgpr_read v17, a13 // (op 990) 
v_accvgpr_read v18, a14 // (op 990) 
v_accvgpr_read v19, a15 // (op 990) 
v_mul_f32 v20, v11, v16 // (op 990) 
v_mul_f32 v21, v11, v17 // (op 990) 
v_mul_f32 v22, v11, v18 // (op 990) 
v_mul_f32 v23, v11, v19 // (op 990) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v10, v12 // (op 994) 
v_mul_f32 v17, v10, v13 // (op 994) 
v_mul_f32 v18, v10, v14 // (op 994) 
v_mul_f32 v19, v10, v15 // (op 994) 

v_add_f32 v12, v16, v20 // (op 991) 
v_add_f32 v13, v17, v21 // (op 991) 
v_add_f32 v14, v18, v22 // (op 991) 
v_add_f32 v15, v19, v23 // (op 991) 





buffer_store_dwordx4 v[12:15], v9, s[44:47], 0 offen offset:384 // (op 992) Store value




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
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ConditionalBottom_0_beta_123:












s_endpgm  // End of RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2
.LRR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_end:
.size RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2, .LRR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_end-RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2
  .amdhsa_next_free_vgpr 84
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 21504 // lds bytes
  .amdhsa_accum_offset 68
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
    DataTypeB: F4
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
    macrotile: [128, 32, 128]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [4, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2
    .symbol: RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 21504
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 64
    .vgpr_count: 66
    .agpr_count: 16
    .max_flat_workgroup_size: 256
    .workgroup_size: [256, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, {is-null: true}]
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
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: FP4, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP4, pointerType: PointerGlobal}
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

