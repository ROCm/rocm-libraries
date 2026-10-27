// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2
.p2align 8
.type RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2,@function
RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_exec_begin:


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


v_mul_lo_u32 v0, v3, 6 // (op 2121) 

v_lshrrev_b32 v3, 3, v0 // (op 2121) 

v_lshrrev_b32 v0, 6, v1 // (op 2250) 
v_and_b32 v4, 1, v0 // (op 2250) 

v_and_b32 v0, 15, v1 // (op 2250) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 2250) 


v_and_b32 v0, 63, v1 // (op 2250) 
v_lshrrev_b32 v4, 4, v0 // (op 2250) 

v_lshlrev_b32 v0, 5, v4 // (op 2250) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 2250) 


v_mul_lo_u32 v0, v4, 6 // (op 2250) 

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


v_mul_lo_u32 v0, v5, 6 // (op 2068) 

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


v_mul_lo_u32 v7, v8, 6 // (op 2226) 

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
v_lshrrev_b32 v12, 1, v1 // (op 1773) 
v_lshl_add_u32 v13, s16, 7, v12 // (op 1773) 

v_and_b32 v12, -2, v1 // (op 1773) 
v_sub_u32 v14, v1, v12 // (op 1773) 

v_lshlrev_b32 v12, 6, v14 // (op 1773) 

v_mov_b32 v15, 0 // (op 1773) convert
v_mov_b32 v14, v12 // (op 1773) convert

v_mul_lo_u32 v17, v13, s13 // (op 1773) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 1773) most significant: high of low * low
v_mul_lo_u32 v16, v13, s12 // (op 1773) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 1773) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 1773) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 1773) most significant half



v_mul_lo_u32 v13, v17, 6 // (op 1773) most significant: low of high * low
v_mul_hi_u32 v14, v16, 6 // (op 1773) most significant: high of low * low
v_mul_lo_u32 v12, v16, 6 // (op 1773) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 1773) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 1773) 

v_mov_b32 v12, v14 // (op 1773) convert


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s23, 6 // (op 1775) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 1775) least significant
s_mul_hi_u32 s6, s22, 6 // (op 1775) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1775) most significant: sum

s_mul_i32 s7, s12, s1 // (op 1775) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 1775) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 1775) least significant
s_mul_hi_u32 s11, s12, s0 // (op 1775) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 1775) most significant: sum
s_add_u32 s7, s7, s11 // (op 1775) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 1775) 

s_mov_b32 s48, s20 // (op 1775) reg expression
s_mov_b32 s49, s21 // (op 1775) reg expression
s_mov_b32 s50, s0 // (op 1775) reg expression
s_mov_b32 s51, 131072 // (op 1775) call()

v_lshrrev_b32 v13, 1, v1 // (op 1792) 
v_lshl_add_u32 v14, s16, 7, v13 // (op 1792) 

v_and_b32 v13, -2, v1 // (op 1792) 
v_sub_u32 v15, v1, v13 // (op 1792) 

v_lshlrev_b32 v13, 6, v15 // (op 1792) 

v_mov_b32 v17, 0 // (op 1792) convert
v_mov_b32 v16, v13 // (op 1792) convert

v_mul_lo_u32 v19, v14, s13 // (op 1792) most significant: low of low * high
v_mul_hi_u32 v13, v14, s12 // (op 1792) most significant: high of low * low
v_mul_lo_u32 v18, v14, s12 // (op 1792) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 1792) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 1792) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 1792) most significant half



v_mul_lo_u32 v17, v19, 6 // (op 1792) most significant: low of high * low
v_mul_hi_u32 v13, v18, 6 // (op 1792) most significant: high of low * low
v_mul_lo_u32 v16, v18, 6 // (op 1792) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 1792) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 1792) 

v_mov_b32 v13, v14 // (op 1792) convert

s_mul_i32 s1, s23, 6 // (op 1794) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 1794) least significant
s_mul_hi_u32 s6, s22, 6 // (op 1794) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1794) most significant: sum

s_mul_i32 s7, s12, s1 // (op 1794) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 1794) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 1794) least significant
s_mul_hi_u32 s11, s12, s0 // (op 1794) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 1794) most significant: sum
s_add_u32 s7, s7, s11 // (op 1794) most significant: sum



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



v_mul_lo_u32 v17, v15, 6 // (op 1837) most significant: low of high * low
v_mul_hi_u32 v18, v14, 6 // (op 1837) most significant: high of low * low
v_mul_lo_u32 v16, v14, 6 // (op 1837) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 1837) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 1837) 

v_mov_b32 v16, v14 // (op 1837) convert

s_mul_i32 s1, s9, 6 // (op 1839) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 1839) least significant
s_mul_hi_u32 s6, s8, 6 // (op 1839) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1839) most significant: sum

s_mul_i32 s7, s24, s1 // (op 1839) most significant: low of low * high
s_mul_i32 s10, s25, s0 // (op 1839) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 1839) least significant
s_mul_hi_u32 s11, s24, s0 // (op 1839) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 1839) most significant: sum
s_add_u32 s7, s7, s11 // (op 1839) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 1839) 

s_mov_b32 s52, s26 // (op 1839) reg expression
s_mov_b32 s53, s27 // (op 1839) reg expression
s_mov_b32 s54, s0 // (op 1839) reg expression
s_mov_b32 s55, 131072 // (op 1839) call()

v_and_b32 v14, -8, v1 // (op 1856) 
v_sub_u32 v15, v1, v14 // (op 1856) 

v_lshlrev_b32 v14, 4, v15 // (op 1856) 

v_mov_b32 v19, 0 // (op 1856) convert
v_mov_b32 v18, v14 // (op 1856) convert

v_lshrrev_b32 v17, 3, v1 // (op 1856) 
v_lshl_add_u32 v14, s17, 5, v17 // (op 1856) 

v_mul_lo_u32 v21, v14, s25 // (op 1856) most significant: low of low * high
v_mul_hi_u32 v15, v14, s24 // (op 1856) most significant: high of low * low
v_mul_lo_u32 v20, v14, s24 // (op 1856) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 1856) most significant: sum


v_add_co_u32 v14, s[0:1], v18, v20 // (op 1856) least significant half
v_addc_co_u32 v15, s[0:1], v19, v21, s[0:1] // (op 1856) most significant half



v_mul_lo_u32 v19, v15, 6 // (op 1856) most significant: low of high * low
v_mul_hi_u32 v17, v14, 6 // (op 1856) most significant: high of low * low
v_mul_lo_u32 v18, v14, 6 // (op 1856) least significant: low of low * low
v_add_u32 v19, v19, v17 // (op 1856) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[18:19] // (op 1856) 

v_mov_b32 v17, v14 // (op 1856) convert

s_mul_i32 s1, s9, 6 // (op 1858) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 1858) least significant
s_mul_hi_u32 s6, s8, 6 // (op 1858) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1858) most significant: sum

s_mul_i32 s7, s24, s1 // (op 1858) most significant: low of low * high
s_mul_i32 s10, s25, s0 // (op 1858) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 1858) least significant
s_mul_hi_u32 s11, s24, s0 // (op 1858) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 1858) most significant: sum
s_add_u32 s7, s7, s11 // (op 1858) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 1858) 

s_mov_b32 s52, s26 // (op 1858) reg expression
s_mov_b32 s53, s27 // (op 1858) reg expression
s_mov_b32 s54, s0 // (op 1858) reg expression
s_mov_b32 s55, 131072 // (op 1858) call()

v_lshrrev_b32 v14, 2, v1 // (op 1901) 
v_lshlrev_b32 v15, 1, v14 // (op 1901) 

v_lshl_add_u32 v14, s16, 7, v15 // (op 1901) 

v_and_b32 v15, -4, v1 // (op 1901) 
v_sub_u32 v18, v1, v15 // (op 1901) 

v_mul_lo_u32 v15, v14, s28 // (op 1901) 
v_add_u32 v15, v15, v18 // (op 1901) 


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

v_lshrrev_b32 v14, 2, v1 // (op 1920) 
v_lshlrev_b32 v18, 1, v14 // (op 1920) 

v_lshl_add_u32 v14, s16, 7, v18 // (op 1920) 

v_and_b32 v18, -4, v1 // (op 1920) 
v_sub_u32 v19, v1, v18 // (op 1920) 

v_mul_lo_u32 v18, v14, s28 // (op 1920) 
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

v_lshrrev_b32 v14, 6, v1 // (op 1965) 
v_and_b32 v19, 1, v14 // (op 1965) 

v_and_b32 v14, 15, v1 // (op 1965) 
v_lshl_add_u32 v20, v19, 4, v14 // (op 1965) 


v_lshl_add_u32 v14, s17, 5, v20 // (op 1965) 

v_mul_lo_u32 v19, v14, s32 // (op 1965) 

v_and_b32 v14, 63, v1 // (op 1965) 
v_lshrrev_b32 v20, 4, v14 // (op 1965) 

v_add_u32 v14, v20, v19 // (op 1965) 


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




buffer_load_ubyte v20, v14, s[60:63], 0 offen offset: 0 // (op 1598) Load value



buffer_load_dwordx3 v[22:24], v16, s[52:55], 0 offen // (op 1602) Load value



buffer_load_dwordx4 v[26:29], v12, s[48:51], 0 offen // (op 1605) Load value
buffer_load_dwordx2 v[30:31], v12, s[48:51], 0 offen offset:16 // (op 1605) Load value
buffer_load_dwordx4 v[32:35], v12, s[48:51], 0 offen offset:24 // (op 1605) Load value
buffer_load_dwordx2 v[36:37], v12, s[48:51], 0 offen offset:40 // (op 1605) Load value

v_mov_b64 v[38:39], s[28:29] // (op 1608) convert

v_mov_b32 v21, v15 // (op 1608) 
buffer_load_ubyte v40, v21, s[56:59], 0 offen offset: 0 // (op 1608) Load value
v_add_u32 v21, v21, v38 // (op 1608) 
buffer_load_ubyte v41, v21, s[56:59], 0 offen offset: 0 // (op 1608) Load value





buffer_load_ubyte v21, v14, s[60:63], 0 offen offset:4 // (op 1612) Load value




buffer_load_dwordx3 v[42:44], v16, s[52:55], 0 offen offset:96 // (op 1616) Load value




buffer_load_dwordx4 v[46:49], v12, s[48:51], 0 offen offset:96 // (op 1619) Load value
buffer_load_dwordx2 v[50:51], v12, s[48:51], 0 offen offset:112 // (op 1619) Load value
buffer_load_dwordx4 v[52:55], v12, s[48:51], 0 offen offset:120 // (op 1619) Load value
buffer_load_dwordx2 v[56:57], v12, s[48:51], 0 offen offset:136 // (op 1619) Load value


v_mov_b64 v[38:39], s[28:29] // (op 1622) convert

v_mov_b32 v12, v15 // (op 1622) 
buffer_load_ubyte v58, v12, s[56:59], 0 offen offset:4 // (op 1622) Load value
v_add_u32 v12, v12, v38 // (op 1622) 
buffer_load_ubyte v59, v12, s[56:59], 0 offen offset:4 // (op 1622) Load value



v_and_b32 v12, -8, v1 // (op 2109) 
v_sub_u32 v25, v1, v12 // (op 2109) 

v_lshlrev_b32 v12, 4, v25 // (op 2109) 

v_lshrrev_b32 v25, 3, v1 // (op 2109) 
v_lshl_add_u32 v45, v25, 7, v12 // (op 2109) 


v_mul_lo_u32 v12, v45, 6 // (op 2109) 

v_lshrrev_b32 v25, 3, v12 // (op 2109) 




s_waitcnt vmcnt(14)
ds_write_b96 v25, v[22:24] // (op 1624) Store local data 

v_and_b32 v12, -2, v1 // (op 2051) 
v_sub_u32 v25, v1, v12 // (op 2051) 

v_lshlrev_b32 v12, 6, v25 // (op 2051) 

v_lshrrev_b32 v25, 1, v1 // (op 2051) 
v_lshl_add_u32 v45, v25, 7, v12 // (op 2051) 


v_mul_lo_u32 v12, v45, 6 // (op 2051) 

v_lshrrev_b32 v25, 3, v12 // (op 2051) 




s_waitcnt vmcnt(13)
ds_write_b128 v25, v[26:29] offset:3072 // (op 1625) Store local data 
s_waitcnt vmcnt(12)
ds_write_b64 v25, v[30:31] offset:3088 // (op 1625) Store local data 
s_waitcnt vmcnt(11)
ds_write_b128 v25, v[32:35] offset:3096 // (op 1625) Store local data 
s_waitcnt vmcnt(10)
ds_write_b64 v25, v[36:37] offset:3112 // (op 1625) Store local data 

v_and_b32 v12, -4, v1 // (op 2162) 
v_sub_u32 v25, v1, v12 // (op 2162) 

v_lshrrev_b32 v12, 2, v1 // (op 2162) 
v_lshl_add_u32 v45, v12, 3, v25 // (op 2162) 





s_waitcnt vmcnt(9)
ds_write_b8 v45, v40 offset:15360 // (op 1626) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v45, v41 offset:15364 // (op 1626) Store local data 

v_and_b32 v12, -2, v1 // (op 2022) 
v_sub_u32 v25, v1, v12 // (op 2022) 

v_lshlrev_b32 v12, 6, v25 // (op 2022) 

v_lshrrev_b32 v25, 1, v1 // (op 2022) 
v_lshl_add_u32 v45, v25, 7, v12 // (op 2022) 


v_mul_lo_u32 v12, v45, 6 // (op 2022) 

v_lshrrev_b32 v25, 3, v12 // (op 2022) 

v_and_b32 v12, -8, v1 // (op 2080) 
v_sub_u32 v45, v1, v12 // (op 2080) 

v_lshlrev_b32 v12, 4, v45 // (op 2080) 

v_lshrrev_b32 v45, 3, v1 // (op 2080) 
v_lshl_add_u32 v14, v45, 7, v12 // (op 2080) 


v_mul_lo_u32 v12, v14, 6 // (op 2080) 

v_lshrrev_b32 v45, 3, v12 // (op 2080) 

v_and_b32 v12, -4, v1 // (op 2133) 
v_sub_u32 v14, v1, v12 // (op 2133) 

v_lshrrev_b32 v12, 2, v1 // (op 2133) 
v_lshl_add_u32 v15, v12, 3, v14 // (op 2133) 


v_and_b32 v12, -2, v1 // (op 2209) 
v_sub_u32 v14, v1, v12 // (op 2209) 

v_lshlrev_b32 v12, 6, v14 // (op 2209) 

v_lshrrev_b32 v14, 1, v1 // (op 2209) 
v_lshl_add_u32 v16, v14, 7, v12 // (op 2209) 


v_mul_lo_u32 v12, v16, 6 // (op 2209) 

v_lshrrev_b32 v14, 3, v12 // (op 2209) 

v_and_b32 v12, -8, v1 // (op 2238) 
v_sub_u32 v16, v1, v12 // (op 2238) 

v_lshlrev_b32 v12, 4, v16 // (op 2238) 

v_lshrrev_b32 v16, 3, v1 // (op 2238) 
v_lshl_add_u32 v38, v16, 7, v12 // (op 2238) 


v_mul_lo_u32 v12, v38, 6 // (op 2238) 

v_lshrrev_b32 v16, 3, v12 // (op 2238) 

v_and_b32 v12, -4, v1 // (op 2262) 
v_sub_u32 v38, v1, v12 // (op 2262) 

v_lshrrev_b32 v12, 2, v1 // (op 2262) 
v_lshl_add_u32 v39, v12, 3, v38 // (op 2262) 


s_mov_b32 s0, 0 // (op 635) call()
s_ashr_i64 s[6:7], s[36:37], 7 // (op 634) 
s_mov_b32 s1, s6 // (op 634) convert

s_and_b32 s6, s1, 4294967294 // (op 634) 

s_cmp_lt_u32 s0, s6 // (op 634) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopBottom_KLoop_634 // (op 634) Condition: Top (jump to RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopBottom_KLoop_634 if false)
RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopTop_KLoop_634:
s_add_u32 s0, s0, 2 // (op 636) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 2326) 



ds_read_b128 v[60:63], v5 offset:3072 // (op 65) Load local data 
ds_read_b64 v[64:65], v5 offset:3088 // (op 65) Load local data 
s_barrier  // (op 2339) 



ds_read_b128 v[66:69], v3 // (op 70) Load local data 
ds_read_b64 v[70:71], v3 offset:16 // (op 70) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 2353) 



ds_read_u8 v12, v0 offset:15360 // (op 75) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[60:65], v[66:71], a[0:3], v12, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 82) 
ds_read_b128 v[72:75], v5 offset:6144 // (op 340) Load local data 


ds_read_b64 v[76:77], v5 offset:6160 // (op 340) Load local data 



ds_read_u8 v12, v0 offset:15488 // (op 344) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[72:77], v[66:71], a[4:7], v12, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 346) 
ds_read_b128 v[60:63], v5 offset:9216 // (op 502) Load local data 


ds_read_b64 v[64:65], v5 offset:9232 // (op 502) Load local data 



ds_read_u8 v12, v0 offset:15616 // (op 506) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[60:65], v[66:71], a[8:11], v12, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 508) 
ds_read_b128 v[72:75], v5 offset:12288 // (op 664) Load local data 


ds_read_b64 v[76:77], v5 offset:12304 // (op 664) Load local data 



ds_read_u8 v12, v0 offset:15744 // (op 668) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[72:77], v[66:71], a[12:15], v12, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 670) 





v_mov_b32 v12, v19 // (op 80) 

buffer_load_ubyte v20, v12, s[60:63], 0 offen offset:8 // (op 80) Load value


v_mov_b32 v38, v17 // (op 10) 

buffer_load_dwordx3 v[22:24], v38, s[52:55], 0 offen offset:192 // (op 10) Load value


v_mov_b32 v60, v13 // (op 4) 

buffer_load_dwordx4 v[26:29], v60, s[48:51], 0 offen offset:192 // (op 4) Load value
buffer_load_dwordx2 v[30:31], v60, s[48:51], 0 offen offset:208 // (op 4) Load value
buffer_load_dwordx4 v[32:35], v60, s[48:51], 0 offen offset:216 // (op 4) Load value
buffer_load_dwordx2 v[36:37], v60, s[48:51], 0 offen offset:232 // (op 4) Load value

v_mov_b64 v[62:63], s[28:29] // (op 17) convert

v_mov_b32 v61, v18 // (op 17) 
v_mov_b32 v64, v61 // (op 17) 
buffer_load_ubyte v40, v64, s[56:59], 0 offen offset:8 // (op 17) Load value
v_add_u32 v64, v64, v62 // (op 17) 
buffer_load_ubyte v41, v64, s[56:59], 0 offen offset:8 // (op 17) Load value


s_barrier  // (op 2376) 



s_waitcnt vmcnt(14)
ds_write_b96 v16, v[42:44] offset:15872 // (op 235) Store local data 
s_barrier  // (op 2370) 



s_waitcnt vmcnt(13)
ds_write_b128 v14, v[46:49] offset:18944 // (op 225) Store local data 
s_waitcnt vmcnt(12)
ds_write_b64 v14, v[50:51] offset:18960 // (op 225) Store local data 
s_waitcnt vmcnt(11)
ds_write_b128 v14, v[52:55] offset:18968 // (op 225) Store local data 
s_waitcnt vmcnt(10)
ds_write_b64 v14, v[56:57] offset:18984 // (op 225) Store local data 
s_barrier  // (op 2382) 



s_waitcnt vmcnt(9)
ds_write_b8 v39, v58 offset:31232 // (op 236) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v39, v59 offset:31236 // (op 236) Store local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 2367) 



ds_read_b128 v[62:65], v8 offset:18944 // (op 227) Load local data 
ds_read_b64 v[66:67], v8 offset:18960 // (op 227) Load local data 
s_barrier  // (op 2373) 



ds_read_b128 v[68:71], v4 offset:15872 // (op 229) Load local data 
ds_read_b64 v[72:73], v4 offset:15888 // (op 229) Load local data 
s_barrier  // (op 2419) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 2379) 



ds_read_u8 v74, v7 offset:31232 // (op 231) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:67], v[68:73], a[0:3], v74, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 234) 
ds_read_b128 v[76:79], v8 offset:22016 // (op 353) Load local data 


ds_read_b64 v[80:81], v8 offset:22032 // (op 353) Load local data 



ds_read_u8 v62, v7 offset:31360 // (op 357) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[76:81], v[68:73], a[4:7], v62, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 347) 
ds_read_b128 v[82:85], v8 offset:25088 // (op 515) Load local data 


ds_read_b64 v[86:87], v8 offset:25104 // (op 515) Load local data 



ds_read_u8 v62, v7 offset:31488 // (op 519) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[82:87], v[68:73], a[8:11], v62, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 509) 
ds_read_b128 v[74:77], v8 offset:28160 // (op 677) Load local data 


ds_read_b64 v[78:79], v8 offset:28176 // (op 677) Load local data 



s_barrier  // (op 2465) 
ds_read_u8 v62, v7 offset:31616 // (op 681) Load local data 
s_barrier  // (op 2469) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[74:79], v[68:73], a[12:15], v62, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 671) 






buffer_load_ubyte v21, v12, s[60:63], 0 offen offset:12 // (op 233) Load value




v_add_u32 v19, 8, v19 // (op 1976) 
buffer_load_dwordx3 v[42:44], v38, s[52:55], 0 offen offset:288 // (op 237) Load value




v_add_u32 v17, 192, v17 // (op 1848) 
buffer_load_dwordx4 v[46:49], v60, s[48:51], 0 offen offset:288 // (op 224) Load value
buffer_load_dwordx2 v[50:51], v60, s[48:51], 0 offen offset:304 // (op 224) Load value
buffer_load_dwordx4 v[52:55], v60, s[48:51], 0 offen offset:312 // (op 224) Load value
buffer_load_dwordx2 v[56:57], v60, s[48:51], 0 offen offset:328 // (op 224) Load value


v_add_u32 v13, 192, v13 // (op 1784) 
v_mov_b64 v[62:63], s[28:29] // (op 238) convert

v_mov_b32 v12, v61 // (op 238) 
buffer_load_ubyte v58, v12, s[56:59], 0 offen offset:12 // (op 238) Load value
v_add_u32 v12, v12, v62 // (op 238) 
buffer_load_ubyte v59, v12, s[56:59], 0 offen offset:12 // (op 238) Load value



s_barrier  // (op 2342) 
v_add_u32 v18, 8, v18 // (op 1912) 



s_waitcnt vmcnt(14)
ds_write_b96 v45, v[22:24] // (op 13) Store local data 
s_barrier  // (op 2329) 



s_waitcnt vmcnt(13)
ds_write_b128 v25, v[26:29] offset:3072 // (op 6) Store local data 
s_waitcnt vmcnt(12)
ds_write_b64 v25, v[30:31] offset:3088 // (op 6) Store local data 
s_waitcnt vmcnt(11)
ds_write_b128 v25, v[32:35] offset:3096 // (op 6) Store local data 
s_waitcnt vmcnt(10)
ds_write_b64 v25, v[36:37] offset:3112 // (op 6) Store local data 
s_barrier  // (op 2356) 



s_waitcnt vmcnt(9)
ds_write_b8 v15, v40 offset:15360 // (op 21) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v15, v41 offset:15364 // (op 21) Store local data 
s_ashr_i64 s[6:7], s[36:37], 7 // (op 634) 
s_mov_b32 s1, s6 // (op 634) convert

s_and_b32 s6, s1, 4294967294 // (op 634) 

s_cmp_lt_u32 s0, s6 // (op 634) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopTop_KLoop_634 // (op 634) Condition: Bottom (jump to RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopTop_KLoop_634 if true)
RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopBottom_KLoop_634:















v_lshrrev_b32 v45, 1, v1 // (op 1744) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1744) 
v_lshl_add_u32 v12, s16, 7, v45 // (op 1744) 

s_lshr_b32 s6, s0, 1 // (op 1744) 

v_and_b32 v45, -2, v1 // (op 1744) 
v_sub_u32 v13, v1, v45 // (op 1744) 

v_lshlrev_b32 v45, 6, v13 // (op 1744) 

v_lshl_add_u32 v13, s6, 8, v45 // (op 1744) 


v_mov_b32 v15, 0 // (op 1744) convert
v_mov_b32 v14, v13 // (op 1744) convert

v_mul_lo_u32 v17, v12, s13 // (op 1744) most significant: low of low * high
v_mul_hi_u32 v13, v12, s12 // (op 1744) most significant: high of low * low
v_mul_lo_u32 v16, v12, s12 // (op 1744) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 1744) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 1744) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 1744) most significant half



v_mul_lo_u32 v13, v17, 6 // (op 1744) most significant: low of high * low
v_mul_hi_u32 v45, v16, 6 // (op 1744) most significant: high of low * low
v_mul_lo_u32 v12, v16, 6 // (op 1744) least significant: low of low * low
v_add_u32 v13, v13, v45 // (op 1744) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 1744) 

v_mov_b32 v45, v14 // (op 1744) convert

s_mul_i32 s1, s23, 6 // (op 1746) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 1746) least significant
s_mul_hi_u32 s6, s22, 6 // (op 1746) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1746) most significant: sum

s_mul_i32 s7, s12, s1 // (op 1746) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 1746) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 1746) least significant
s_mul_hi_u32 s11, s12, s0 // (op 1746) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 1746) most significant: sum
s_add_u32 s7, s7, s11 // (op 1746) most significant: sum



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


v_mov_b32 v15, 0 // (op 1814) convert
v_mov_b32 v14, v13 // (op 1814) convert

v_lshrrev_b32 v12, 3, v1 // (op 1814) 
v_lshl_add_u32 v13, s17, 5, v12 // (op 1814) 

v_mul_lo_u32 v17, v13, s25 // (op 1814) most significant: low of low * high
v_mul_hi_u32 v12, v13, s24 // (op 1814) most significant: high of low * low
v_mul_lo_u32 v16, v13, s24 // (op 1814) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 1814) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v16 // (op 1814) least significant half
v_addc_co_u32 v13, s[0:1], v15, v17, s[0:1] // (op 1814) most significant half



v_mul_lo_u32 v15, v13, 6 // (op 1814) most significant: low of high * low
v_mul_hi_u32 v16, v12, 6 // (op 1814) most significant: high of low * low
v_mul_lo_u32 v14, v12, 6 // (op 1814) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 1814) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 1814) 

v_mov_b32 v14, v12 // (op 1814) convert

s_mul_i32 s1, s9, 6 // (op 1816) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 1816) least significant
s_mul_hi_u32 s6, s8, 6 // (op 1816) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1816) most significant: sum

s_mul_i32 s7, s24, s1 // (op 1816) most significant: low of low * high
s_mul_i32 s10, s25, s0 // (op 1816) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 1816) least significant
s_mul_hi_u32 s11, s24, s0 // (op 1816) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 1816) most significant: sum
s_add_u32 s7, s7, s11 // (op 1816) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 1816) 

s_mov_b32 s52, s26 // (op 1816) reg expression
s_mov_b32 s53, s27 // (op 1816) reg expression
s_mov_b32 s54, s0 // (op 1816) reg expression
s_mov_b32 s55, 131072 // (op 1816) call()

v_lshrrev_b32 v12, 2, v1 // (op 1878) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1878) 
v_lshlrev_b32 v13, 1, v12 // (op 1878) 

s_lshr_b32 s6, s0, 1 // (op 1878) 

v_lshl_add_u32 v12, s16, 7, v13 // (op 1878) 

v_and_b32 v13, -4, v1 // (op 1878) 
v_sub_u32 v15, v1, v13 // (op 1878) 

v_lshl_add_u32 v13, s6, 3, v15 // (op 1878) 


v_mul_lo_u32 v15, v12, s28 // (op 1878) 
v_add_u32 v15, v15, v13 // (op 1878) 


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

v_and_b32 v12, 63, v1 // (op 1942) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1942) 
v_lshrrev_b32 v13, 4, v12 // (op 1942) 

s_lshr_b32 s6, s0, 1 // (op 1942) 

v_lshrrev_b32 v12, 6, v1 // (op 1942) 
v_lshl_add_u32 v16, s6, 3, v13 // (op 1942) 


v_and_b32 v13, 1, v12 // (op 1942) 

v_and_b32 v12, 15, v1 // (op 1942) 
v_lshl_add_u32 v17, v13, 4, v12 // (op 1942) 


v_lshl_add_u32 v12, s17, 5, v17 // (op 1942) 

v_mul_lo_u32 v13, v12, s32 // (op 1942) 

v_add_u32 v12, v16, v13 // (op 1942) 


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

v_and_b32 v13, -2, v1 // (op 2035) 
v_sub_u32 v16, v1, v13 // (op 2035) 

v_lshlrev_b32 v13, 6, v16 // (op 2035) 

v_lshrrev_b32 v16, 1, v1 // (op 2035) 
v_lshl_add_u32 v17, v16, 7, v13 // (op 2035) 


v_mul_lo_u32 v13, v17, 6 // (op 2035) 

v_lshrrev_b32 v16, 3, v13 // (op 2035) 

v_and_b32 v13, -8, v1 // (op 2093) 
v_sub_u32 v17, v1, v13 // (op 2093) 

v_lshlrev_b32 v13, 4, v17 // (op 2093) 

v_lshrrev_b32 v17, 3, v1 // (op 2093) 
v_lshl_add_u32 v18, v17, 7, v13 // (op 2093) 


v_mul_lo_u32 v13, v18, 6 // (op 2093) 

v_lshrrev_b32 v17, 3, v13 // (op 2093) 

v_and_b32 v13, -4, v1 // (op 2146) 
v_sub_u32 v18, v1, v13 // (op 2146) 

v_lshrrev_b32 v13, 2, v1 // (op 2146) 
v_lshl_add_u32 v19, v13, 3, v18 // (op 2146) 


s_ashr_i64 s[0:1], s[36:37], 7 // (op 640) 
s_and_b32 s6, s0, 4294967294 // (op 640) 

s_ashr_i64 s[0:1], s[36:37], 7 // (op 639) 
s_mov_b32 s7, s0 // (op 639) convert

s_cmp_lt_u32 s6, s7 // (op 639) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopBottom_KLoopTail_639 // (op 639) Condition: Top (jump to RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopBottom_KLoopTail_639 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopTop_KLoopTail_639:


s_add_u32 s6, s6, 1 // (op 641) 
v_mov_b32 v13, v45 // (op 166) 

buffer_load_dwordx4 v[20:23], v13, s[48:51], 0 offen // (op 166) Load value
buffer_load_dwordx2 v[24:25], v13, s[48:51], 0 offen offset:16 // (op 166) Load value
buffer_load_dwordx4 v[26:29], v13, s[48:51], 0 offen offset:24 // (op 166) Load value
buffer_load_dwordx2 v[30:31], v13, s[48:51], 0 offen offset:40 // (op 166) Load value

v_add_u32 v45, 96, v45 // (op 1734) 


s_barrier  // (op 2332) 
v_mov_b32 v13, v14 // (op 179) 




buffer_load_dwordx3 v[32:34], v13, s[52:55], 0 offen // (op 179) Load value

v_add_u32 v14, 96, v14 // (op 1806) 
s_waitcnt vmcnt(4)
ds_write_b128 v16, v[20:23] offset:3072 // (op 167) Store local data 

v_mov_b64 v[36:37], s[28:29] // (op 180) convert

v_mov_b32 v13, v15 // (op 180) 
v_mov_b32 v18, v13 // (op 180) 
buffer_load_ubyte v38, v18, s[56:59], 0 offen offset: 0 // (op 180) Load value
v_add_u32 v18, v18, v36 // (op 180) 
s_waitcnt vmcnt(4)
ds_write_b64 v16, v[24:25] offset:3088 // (op 167) Store local data 
s_waitcnt vmcnt(3)
ds_write_b128 v16, v[26:29] offset:3096 // (op 167) Store local data 
s_waitcnt vmcnt(2)
ds_write_b64 v16, v[30:31] offset:3112 // (op 167) Store local data 
buffer_load_ubyte v39, v18, s[56:59], 0 offen offset: 0 // (op 180) Load value




v_add_u32 v15, 4, v15 // (op 1870) 
s_barrier  // (op 2345) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 2335) 






ds_read_b128 v[20:23], v5 offset:3072 // (op 169) Load local data 
ds_read_b64 v[24:25], v5 offset:3088 // (op 169) Load local data 
s_waitcnt vmcnt(2)
ds_write_b96 v17, v[32:34] // (op 177) Store local data 

s_barrier  // (op 2359) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 2349) 






ds_read_b128 v[26:29], v3 // (op 171) Load local data 
ds_read_b64 v[30:31], v3 offset:16 // (op 171) Load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v19, v38 offset:15360 // (op 178) Store local data 
s_waitcnt vmcnt(0)
ds_write_b8 v19, v39 offset:15364 // (op 178) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 2363) 



ds_read_u8 v13, v0 offset:15360 // (op 173) Load local data 


v_mov_b32 v18, v12 // (op 175) 

buffer_load_ubyte v32, v18, s[60:63], 0 offen offset: 0 // (op 175) Load value

v_add_u32 v12, 4, v12 // (op 1934) 



ds_read_b128 v[34:37], v5 offset:6144 // (op 321) Load local data 
ds_read_b64 v[38:39], v5 offset:6160 // (op 321) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:25], v[26:31], a[0:3], v13, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 176) 





ds_read_u8 v13, v0 offset:15488 // (op 325) Load local data 



ds_read_b128 v[20:23], v5 offset:9216 // (op 483) Load local data 
ds_read_b64 v[24:25], v5 offset:9232 // (op 483) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[34:39], v[26:31], a[4:7], v13, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 327) 





ds_read_u8 v13, v0 offset:15616 // (op 487) Load local data 



ds_read_b128 v[34:37], v5 offset:12288 // (op 645) Load local data 
ds_read_b64 v[38:39], v5 offset:12304 // (op 645) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[20:25], v[26:31], a[8:11], v13, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 489) 





ds_read_u8 v13, v0 offset:15744 // (op 649) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[34:39], v[26:31], a[12:15], v13, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 651) 




s_ashr_i64 s[0:1], s[36:37], 7 // (op 639) 
s_mov_b32 s7, s0 // (op 639) convert

s_cmp_lt_u32 s6, s7 // (op 639) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopTop_KLoopTail_639 // (op 639) Condition: Bottom (jump to RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopTop_KLoopTail_639 if true)
RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopBottom_KLoopTail_639:






















s_barrier  // (op 2453) 







v_cmp_eq_f32 vcc, 0.00000, v10 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ConditionalFalse_0_beta_123
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
s_branch RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ConditionalBottom_0_beta_123
RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ConditionalFalse_0_beta_123:



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
RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ConditionalBottom_0_beta_123:












s_endpgm  // End of RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2
.LRR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_end:
.size RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2, .LRR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_end-RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2
  .amdhsa_next_free_vgpr 104
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 31744 // lds bytes
  .amdhsa_accum_offset 88
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
  - .name: RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2
    .symbol: RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 31744
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 64
    .vgpr_count: 88
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

