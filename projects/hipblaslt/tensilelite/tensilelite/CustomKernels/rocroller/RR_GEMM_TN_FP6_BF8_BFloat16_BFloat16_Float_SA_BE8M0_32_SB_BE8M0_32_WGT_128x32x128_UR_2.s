// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2
.p2align 8
.type RR_GEMM_TN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2,@function
RR_GEMM_TN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 6, v1 // (op 2162) 
v_and_b32 v3, 1, v0 // (op 2162) 

v_and_b32 v0, 15, v1 // (op 2162) 
v_lshl_add_u32 v4, v3, 4, v0 // (op 2162) 


v_and_b32 v0, 48, v1 // (op 2162) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 2162) 


v_lshrrev_b32 v0, 6, v1 // (op 2306) 
v_and_b32 v4, 1, v0 // (op 2306) 

v_and_b32 v0, 15, v1 // (op 2306) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 2306) 


v_and_b32 v0, 48, v1 // (op 2306) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 2306) 


v_lshrrev_b32 v0, 7, v1 // (op 2048) 
v_and_b32 v5, 63, v1 // (op 2048) 
v_lshrrev_b32 v6, 4, v5 // (op 2048) 
v_lshlrev_b32 v7, 2, v6 // (op 2048) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 2048) 


v_lshl_add_u32 v0, s16, 7, v6 // (op 2048) 

v_lshrrev_b32 v6, 2, v5 // (op 2048) 

v_and_b32 v5, 3, v6 // (op 2048) 

v_and_b32 v6, 3, v1 // (op 2048) 
v_lshl_add_u32 v7, v5, 2, v6 // (op 2048) 


v_lshrrev_b32 v5, 6, v1 // (op 2048) 
v_and_b32 v6, 1, v5 // (op 2048) 

v_lshl_add_u32 v5, v6, 4, v7 // (op 2048) 


v_lshl_add_u32 v6, s17, 5, v5 // (op 2048) 

v_mul_lo_u32 v5, v6, s2 // (op 2048) 

v_add_lshl_u32 v6, v0, v5, 1 // (op 2048) 


v_lshrrev_b32 v0, 7, v1 // (op 2110) 
v_and_b32 v5, 15, v1 // (op 2110) 
v_lshl_add_u32 v7, v0, 4, v5 // (op 2110) 


v_and_b32 v0, 63, v1 // (op 2110) 
v_lshrrev_b32 v5, 4, v0 // (op 2110) 

v_lshlrev_b32 v0, 5, v5 // (op 2110) 

v_lshl_add_u32 v5, v7, 7, v0 // (op 2110) 


v_mul_lo_u32 v0, v5, 6 // (op 2110) 

v_lshrrev_b32 v5, 3, v0 // (op 2110) 

v_lshrrev_b32 v0, 7, v1 // (op 2220) 
v_and_b32 v7, 15, v1 // (op 2220) 
s_mul_i32 s1, s2, s9 // (op 2050) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 2050) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 2050) least significant
s_mul_hi_u32 s19, s2, s8 // (op 2050) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2050) most significant: sum
s_add_u32 s1, s1, s19 // (op 2050) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 2050) 

v_lshl_add_u32 v8, v0, 4, v7 // (op 2220) 


v_and_b32 v0, 63, v1 // (op 2220) 
v_lshrrev_b32 v7, 4, v0 // (op 2220) 

v_lshl_add_u32 v0, v8, 2, v7 // (op 2220) 


v_lshrrev_b32 v7, 7, v1 // (op 2236) 
v_and_b32 v8, 63, v1 // (op 2236) 
v_lshrrev_b32 v9, 4, v8 // (op 2236) 
v_lshlrev_b32 v10, 2, v9 // (op 2236) 

v_lshl_add_u32 v9, v7, 4, v10 // (op 2236) 


v_lshl_add_u32 v7, s16, 7, v9 // (op 2236) 

v_lshrrev_b32 v9, 2, v8 // (op 2236) 

v_and_b32 v8, 3, v9 // (op 2236) 

v_and_b32 v9, 3, v1 // (op 2236) 
v_lshl_add_u32 v10, v8, 2, v9 // (op 2236) 


v_lshrrev_b32 v8, 6, v1 // (op 2236) 
v_and_b32 v9, 1, v8 // (op 2236) 

v_lshl_add_u32 v8, v9, 4, v10 // (op 2236) 


v_lshl_add_u32 v9, s17, 5, v8 // (op 2236) 

v_mul_lo_u32 v8, v9, s4 // (op 2236) 

v_add_lshl_u32 v9, v7, v8, 1 // (op 2236) 


v_lshrrev_b32 v7, 7, v1 // (op 2267) 
v_and_b32 v8, 15, v1 // (op 2267) 
v_lshl_add_u32 v10, v7, 4, v8 // (op 2267) 


v_and_b32 v7, 63, v1 // (op 2267) 
v_lshrrev_b32 v8, 4, v7 // (op 2267) 

v_lshlrev_b32 v7, 5, v8 // (op 2267) 

v_lshl_add_u32 v8, v10, 7, v7 // (op 2267) 


v_mul_lo_u32 v7, v8, 6 // (op 2267) 

v_lshrrev_b32 v8, 3, v7 // (op 2267) 

v_lshrrev_b32 v7, 7, v1 // (op 2335) 
v_and_b32 v10, 15, v1 // (op 2335) 
s_mul_i32 s1, s4, s9 // (op 2238) most significant: low of low * high
v_lshl_add_u32 v11, v7, 4, v10 // (op 2335) 


v_and_b32 v7, 63, v1 // (op 2335) 
v_lshrrev_b32 v10, 4, v7 // (op 2335) 

v_lshl_add_u32 v7, v11, 2, v10 // (op 2335) 


s_mul_i32 s38, s5, s8 // (op 2238) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 2238) least significant
s_mul_hi_u32 s39, s4, s8 // (op 2238) most significant: high of low * low
s_add_u32 s1, s1, s38 // (op 2238) most significant: sum
s_add_u32 s1, s1, s39 // (op 2238) most significant: sum


s_lshl_b64 s[38:39], s[0:1], 1 // (op 2238) 

s_mov_b32 s40, s6 // (op 2050) reg expression
s_mov_b32 s41, s7 // (op 2050) reg expression
s_mov_b32 s42, s18 // (op 2050) reg expression
s_mov_b32 s43, 131072 // (op 2050) call()

s_mov_b32 s44, s10 // (op 2238) reg expression

s_mov_b32 s45, s11 // (op 2238) reg expression
s_mov_b32 s46, s38 // (op 2238) reg expression
s_mov_b32 s47, 131072 // (op 2238) call()


v_mov_b32 v10, s14 // (op 2) Move value
v_mov_b32 v11, s15 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 61) call()
v_accvgpr_write a1, 0.00000 // (op 61) call()
v_accvgpr_write a2, 0.00000 // (op 61) call()
v_accvgpr_write a3, 0.00000 // (op 61) call()
v_accvgpr_write a4, 0.00000 // (op 313) call()
v_accvgpr_write a5, 0.00000 // (op 313) call()
v_accvgpr_write a6, 0.00000 // (op 313) call()
v_accvgpr_write a7, 0.00000 // (op 313) call()
v_accvgpr_write a8, 0.00000 // (op 475) call()
v_accvgpr_write a9, 0.00000 // (op 475) call()
v_accvgpr_write a10, 0.00000 // (op 475) call()
v_accvgpr_write a11, 0.00000 // (op 475) call()
v_accvgpr_write a12, 0.00000 // (op 637) call()
v_accvgpr_write a13, 0.00000 // (op 637) call()
v_accvgpr_write a14, 0.00000 // (op 637) call()
v_accvgpr_write a15, 0.00000 // (op 637) call()
v_lshrrev_b32 v12, 1, v1 // (op 1815) 
v_lshl_add_u32 v13, s16, 7, v12 // (op 1815) 

v_and_b32 v12, -2, v1 // (op 1815) 
v_sub_u32 v14, v1, v12 // (op 1815) 

v_lshlrev_b32 v12, 6, v14 // (op 1815) 

v_mov_b32 v15, 0 // (op 1815) convert
v_mov_b32 v14, v12 // (op 1815) convert

v_mul_lo_u32 v17, v13, s13 // (op 1815) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 1815) most significant: high of low * low
v_mul_lo_u32 v16, v13, s12 // (op 1815) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 1815) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 1815) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 1815) most significant half



v_mul_lo_u32 v13, v17, 6 // (op 1815) most significant: low of high * low
v_mul_hi_u32 v14, v16, 6 // (op 1815) most significant: high of low * low
v_mul_lo_u32 v12, v16, 6 // (op 1815) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 1815) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 1815) 

v_mov_b32 v12, v14 // (op 1815) convert


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s23, 6 // (op 1817) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 1817) least significant
s_mul_hi_u32 s6, s22, 6 // (op 1817) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1817) most significant: sum

s_mul_i32 s7, s12, s1 // (op 1817) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 1817) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 1817) least significant
s_mul_hi_u32 s11, s12, s0 // (op 1817) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 1817) most significant: sum
s_add_u32 s7, s7, s11 // (op 1817) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 1817) 

s_mov_b32 s48, s20 // (op 1817) reg expression
s_mov_b32 s49, s21 // (op 1817) reg expression
s_mov_b32 s50, s0 // (op 1817) reg expression
s_mov_b32 s51, 131072 // (op 1817) call()

v_lshrrev_b32 v13, 1, v1 // (op 1834) 
v_lshl_add_u32 v14, s16, 7, v13 // (op 1834) 

v_and_b32 v13, -2, v1 // (op 1834) 
v_sub_u32 v15, v1, v13 // (op 1834) 

v_lshlrev_b32 v13, 6, v15 // (op 1834) 

v_mov_b32 v17, 0 // (op 1834) convert
v_mov_b32 v16, v13 // (op 1834) convert

v_mul_lo_u32 v19, v14, s13 // (op 1834) most significant: low of low * high
v_mul_hi_u32 v13, v14, s12 // (op 1834) most significant: high of low * low
v_mul_lo_u32 v18, v14, s12 // (op 1834) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 1834) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 1834) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 1834) most significant half



v_mul_lo_u32 v17, v19, 6 // (op 1834) most significant: low of high * low
v_mul_hi_u32 v13, v18, 6 // (op 1834) most significant: high of low * low
v_mul_lo_u32 v16, v18, 6 // (op 1834) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 1834) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 1834) 

v_mov_b32 v13, v14 // (op 1834) convert

s_mul_i32 s1, s23, 6 // (op 1836) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 1836) least significant
s_mul_hi_u32 s6, s22, 6 // (op 1836) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1836) most significant: sum

s_mul_i32 s7, s12, s1 // (op 1836) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 1836) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 1836) least significant
s_mul_hi_u32 s11, s12, s0 // (op 1836) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 1836) most significant: sum
s_add_u32 s7, s7, s11 // (op 1836) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 1836) 

s_mov_b32 s48, s20 // (op 1836) reg expression
s_mov_b32 s49, s21 // (op 1836) reg expression
s_mov_b32 s50, s0 // (op 1836) reg expression
s_mov_b32 s51, 131072 // (op 1836) call()

v_and_b32 v14, -8, v1 // (op 1879) 
v_sub_u32 v15, v1, v14 // (op 1879) 

v_lshlrev_b32 v14, 4, v15 // (op 1879) 

v_mov_b32 v17, 0 // (op 1879) convert
v_mov_b32 v16, v14 // (op 1879) convert

v_lshrrev_b32 v14, 3, v1 // (op 1879) 
v_lshl_add_u32 v15, s17, 5, v14 // (op 1879) 

v_mul_lo_u32 v19, v15, s25 // (op 1879) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 1879) most significant: high of low * low
v_mul_lo_u32 v18, v15, s24 // (op 1879) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 1879) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v18 // (op 1879) least significant half
v_addc_co_u32 v15, s[0:1], v17, v19, s[0:1] // (op 1879) most significant half



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

v_and_b32 v16, -8, v1 // (op 1898) 
v_sub_u32 v17, v1, v16 // (op 1898) 

v_lshlrev_b32 v16, 4, v17 // (op 1898) 

v_mov_b32 v19, 0 // (op 1898) convert
v_mov_b32 v18, v16 // (op 1898) convert

v_lshrrev_b32 v16, 3, v1 // (op 1898) 
v_lshl_add_u32 v17, s17, 5, v16 // (op 1898) 

v_mul_lo_u32 v21, v17, s25 // (op 1898) most significant: low of low * high
v_mul_hi_u32 v16, v17, s24 // (op 1898) most significant: high of low * low
v_mul_lo_u32 v20, v17, s24 // (op 1898) least significant: low of low * low
v_add_u32 v21, v21, v16 // (op 1898) most significant: sum


v_add_co_u32 v16, s[0:1], v18, v20 // (op 1898) least significant half
v_addc_co_u32 v17, s[0:1], v19, v21, s[0:1] // (op 1898) most significant half



s_mul_i32 s1, s24, s9 // (op 1900) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 1900) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 1900) least significant
s_mul_hi_u32 s7, s24, s8 // (op 1900) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1900) most significant: sum
s_add_u32 s1, s1, s7 // (op 1900) most significant: sum


s_mov_b32 s52, s26 // (op 1900) reg expression
s_mov_b32 s53, s27 // (op 1900) reg expression
s_mov_b32 s54, s0 // (op 1900) reg expression
s_mov_b32 s55, 131072 // (op 1900) call()

v_lshrrev_b32 v18, 2, v1 // (op 1943) 
v_lshlrev_b32 v19, 1, v18 // (op 1943) 

v_lshl_add_u32 v18, s16, 7, v19 // (op 1943) 

v_and_b32 v19, -4, v1 // (op 1943) 
v_sub_u32 v20, v1, v19 // (op 1943) 

v_mul_lo_u32 v19, v18, s28 // (op 1943) 
v_add_u32 v19, v19, v20 // (op 1943) 


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

v_lshrrev_b32 v18, 2, v1 // (op 1962) 
v_lshlrev_b32 v20, 1, v18 // (op 1962) 

v_lshl_add_u32 v18, s16, 7, v20 // (op 1962) 

v_and_b32 v20, -4, v1 // (op 1962) 
v_sub_u32 v21, v1, v20 // (op 1962) 

v_mul_lo_u32 v20, v18, s28 // (op 1962) 
v_add_u32 v20, v20, v21 // (op 1962) 


s_mul_i32 s1, s28, s23 // (op 1964) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 1964) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 1964) least significant
s_mul_hi_u32 s7, s28, s22 // (op 1964) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1964) most significant: sum
s_add_u32 s1, s1, s7 // (op 1964) most significant: sum


s_mov_b32 s56, s30 // (op 1964) reg expression
s_mov_b32 s57, s31 // (op 1964) reg expression
s_mov_b32 s58, s0 // (op 1964) reg expression
s_mov_b32 s59, 131072 // (op 1964) call()

v_lshrrev_b32 v18, 6, v1 // (op 2007) 
v_and_b32 v21, 1, v18 // (op 2007) 

v_and_b32 v18, 15, v1 // (op 2007) 
v_lshl_add_u32 v22, v21, 4, v18 // (op 2007) 


v_lshl_add_u32 v18, s17, 5, v22 // (op 2007) 

v_mul_lo_u32 v21, v18, s32 // (op 2007) 

v_and_b32 v18, 63, v1 // (op 2007) 
v_lshrrev_b32 v22, 4, v18 // (op 2007) 

v_add_u32 v18, v22, v21 // (op 2007) 


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

v_lshrrev_b32 v21, 6, v1 // (op 2026) 
v_and_b32 v22, 1, v21 // (op 2026) 

v_and_b32 v21, 15, v1 // (op 2026) 
v_lshl_add_u32 v23, v22, 4, v21 // (op 2026) 


v_lshl_add_u32 v21, s17, 5, v23 // (op 2026) 

v_mul_lo_u32 v22, v21, s32 // (op 2026) 

v_and_b32 v21, 63, v1 // (op 2026) 
v_lshrrev_b32 v23, 4, v21 // (op 2026) 

v_add_u32 v21, v23, v22 // (op 2026) 


s_mul_i32 s1, s32, s9 // (op 2028) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 2028) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 2028) least significant
s_mul_hi_u32 s7, s32, s8 // (op 2028) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2028) most significant: sum
s_add_u32 s1, s1, s7 // (op 2028) most significant: sum


s_mov_b32 s60, s34 // (op 2028) reg expression
s_mov_b32 s61, s35 // (op 2028) reg expression
s_mov_b32 s62, s0 // (op 2028) reg expression
s_mov_b32 s63, 131072 // (op 2028) call()




buffer_load_ubyte v22, v18, s[60:63], 0 offen offset: 0 // (op 1616) Load value
v_and_b32 v23, -8, v1 // (op 2150) 
v_sub_u32 v24, v1, v23 // (op 2150) 

v_lshlrev_b32 v23, 4, v24 // (op 2150) 

v_lshrrev_b32 v24, 3, v1 // (op 2150) 
v_lshl_add_u32 v25, v24, 7, v23 // (op 2150) 


s_nop 0
v_readfirstlane_b32 s0, v25 // (op 2150) Wait state hazard: VALU Write Readlane Hazard




s_nop 1
s_mov_b32 m0, s0 // (op 1768) reg expression
buffer_load_dwordx4 v14, s[52:55], 0 offen offset: 0 lds // (op 1768) Load value direct to lds




buffer_load_dwordx4 v[24:27], v12, s[48:51], 0 offen // (op 1623) Load value
buffer_load_dwordx2 v[28:29], v12, s[48:51], 0 offen offset:16 // (op 1623) Load value
buffer_load_dwordx4 v[30:33], v12, s[48:51], 0 offen offset:24 // (op 1623) Load value
buffer_load_dwordx2 v[34:35], v12, s[48:51], 0 offen offset:40 // (op 1623) Load value

v_mov_b64 v[36:37], s[28:29] // (op 1626) convert

v_mov_b32 v23, v19 // (op 1626) 
buffer_load_ubyte v38, v23, s[56:59], 0 offen offset: 0 // (op 1626) Load value
v_add_u32 v23, v23, v36 // (op 1626) 
buffer_load_ubyte v39, v23, s[56:59], 0 offen offset: 0 // (op 1626) Load value





buffer_load_ubyte v23, v18, s[60:63], 0 offen offset:4 // (op 1630) Load value

v_and_b32 v18, -8, v1 // (op 2294) 
v_sub_u32 v36, v1, v18 // (op 2294) 

v_lshlrev_b32 v18, 4, v36 // (op 2294) 

v_lshrrev_b32 v36, 3, v1 // (op 2294) 
v_lshl_add_u32 v37, v36, 7, v18 // (op 2294) 


s_nop 0
v_readfirstlane_b32 s0, v37 // (op 2294) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s1, 4096, s0 // (op 1773) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 1773) reg expression
s_mov_b32 s6, 128 // (op 1773) call()
buffer_load_dwordx4 v14, s[52:55], s6 offen offset: 0 lds // (op 1773) Load value direct to lds







buffer_load_dwordx4 v[40:43], v12, s[48:51], 0 offen offset:96 // (op 1637) Load value
buffer_load_dwordx2 v[44:45], v12, s[48:51], 0 offen offset:112 // (op 1637) Load value
buffer_load_dwordx4 v[46:49], v12, s[48:51], 0 offen offset:120 // (op 1637) Load value
buffer_load_dwordx2 v[50:51], v12, s[48:51], 0 offen offset:136 // (op 1637) Load value


v_mov_b64 v[14:15], s[28:29] // (op 1640) convert

v_mov_b32 v12, v19 // (op 1640) 
buffer_load_ubyte v36, v12, s[56:59], 0 offen offset:4 // (op 1640) Load value
v_add_u32 v12, v12, v14 // (op 1640) 
buffer_load_ubyte v37, v12, s[56:59], 0 offen offset:4 // (op 1640) Load value



v_and_b32 v12, -2, v1 // (op 2093) 
v_sub_u32 v14, v1, v12 // (op 2093) 

v_lshlrev_b32 v12, 6, v14 // (op 2093) 

v_lshrrev_b32 v14, 1, v1 // (op 2093) 
v_lshl_add_u32 v15, v14, 7, v12 // (op 2093) 


v_mul_lo_u32 v12, v15, 6 // (op 2093) 

v_lshrrev_b32 v14, 3, v12 // (op 2093) 




s_waitcnt vmcnt(13)
ds_write_b128 v14, v[24:27] offset:8192 // (op 1643) Store local data 
s_waitcnt vmcnt(12)
ds_write_b64 v14, v[28:29] offset:8208 // (op 1643) Store local data 
s_waitcnt vmcnt(11)
ds_write_b128 v14, v[30:33] offset:8216 // (op 1643) Store local data 
s_waitcnt vmcnt(10)
ds_write_b64 v14, v[34:35] offset:8232 // (op 1643) Store local data 

v_and_b32 v12, -4, v1 // (op 2203) 
v_sub_u32 v14, v1, v12 // (op 2203) 

v_lshrrev_b32 v12, 2, v1 // (op 2203) 
v_lshl_add_u32 v15, v12, 3, v14 // (op 2203) 





s_waitcnt vmcnt(9)
ds_write_b8 v15, v38 offset:20480 // (op 1644) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v15, v39 offset:20484 // (op 1644) Store local data 

v_and_b32 v12, -2, v1 // (op 2064) 
v_sub_u32 v14, v1, v12 // (op 2064) 

v_lshlrev_b32 v12, 6, v14 // (op 2064) 

v_lshrrev_b32 v14, 1, v1 // (op 2064) 
v_lshl_add_u32 v15, v14, 7, v12 // (op 2064) 


v_mul_lo_u32 v12, v15, 6 // (op 2064) 

v_lshrrev_b32 v14, 3, v12 // (op 2064) 

v_and_b32 v12, -8, v1 // (op 2122) 
v_sub_u32 v15, v1, v12 // (op 2122) 

v_lshlrev_b32 v12, 4, v15 // (op 2122) 

v_lshrrev_b32 v15, 3, v1 // (op 2122) 
v_lshl_add_u32 v18, v15, 7, v12 // (op 2122) 


s_nop 0
v_readfirstlane_b32 s0, v18 // (op 2122) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -4, v1 // (op 2174) 
v_sub_u32 v15, v1, v12 // (op 2174) 

v_lshrrev_b32 v12, 2, v1 // (op 2174) 
v_lshl_add_u32 v18, v12, 3, v15 // (op 2174) 


v_and_b32 v12, -2, v1 // (op 2250) 
v_sub_u32 v15, v1, v12 // (op 2250) 

v_lshlrev_b32 v12, 6, v15 // (op 2250) 

v_lshrrev_b32 v15, 1, v1 // (op 2250) 
v_lshl_add_u32 v19, v15, 7, v12 // (op 2250) 


v_mul_lo_u32 v12, v19, 6 // (op 2250) 

v_lshrrev_b32 v15, 3, v12 // (op 2250) 

v_and_b32 v12, -8, v1 // (op 2279) 
v_sub_u32 v19, v1, v12 // (op 2279) 

v_lshlrev_b32 v12, 4, v19 // (op 2279) 

v_lshrrev_b32 v19, 3, v1 // (op 2279) 
v_lshl_add_u32 v52, v19, 7, v12 // (op 2279) 


s_nop 0
v_readfirstlane_b32 s1, v52 // (op 2279) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -4, v1 // (op 2318) 
v_sub_u32 v19, v1, v12 // (op 2318) 

v_lshrrev_b32 v12, 2, v1 // (op 2318) 
v_lshl_add_u32 v52, v12, 3, v19 // (op 2318) 


s_mov_b32 s6, 0 // (op 639) call()
s_ashr_i64 s[10:11], s[36:37], 7 // (op 638) 
s_mov_b32 s7, s10 // (op 638) convert

s_and_b32 s10, s7, 4294967294 // (op 638) 

s_cmp_lt_u32 s6, s10 // (op 638) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopBottom_KLoop_638 // (op 638) Condition: Top (jump to RR_GEMM_TN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopBottom_KLoop_638 if false)
RR_GEMM_TN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopTop_KLoop_638:
s_add_u32 s6, s6, 2 // (op 640) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 2382) 



ds_read_b128 v[54:57], v5 offset:8192 // (op 67) Load local data 
ds_read_b64 v[58:59], v5 offset:8208 // (op 67) Load local data 
s_barrier  // (op 2395) 



ds_read_b128 v[60:63], v3 // (op 72) Load local data 
ds_read_b128 v[64:67], v3 offset:64 // (op 72) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 2408) 



ds_read_u8 v12, v0 offset:20480 // (op 77) Load local data 



ds_read_b128 v[68:71], v5 offset:11264 // (op 344) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[54:59], v[60:67], a[0:3], v12, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 84) 
ds_read_b64 v[72:73], v5 offset:11280 // (op 344) Load local data 





ds_read_u8 v12, v0 offset:20608 // (op 348) Load local data 



ds_read_b128 v[54:57], v5 offset:14336 // (op 506) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[68:73], v[60:67], a[4:7], v12, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 350) 
ds_read_b64 v[58:59], v5 offset:14352 // (op 506) Load local data 





ds_read_u8 v12, v0 offset:20736 // (op 510) Load local data 



ds_read_b128 v[68:71], v5 offset:17408 // (op 668) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[54:59], v[60:67], a[8:11], v12, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 512) 
ds_read_b64 v[72:73], v5 offset:17424 // (op 668) Load local data 





ds_read_u8 v12, v0 offset:20864 // (op 672) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[68:73], v[60:67], a[12:15], v12, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 674) 





v_mov_b32 v12, v21 // (op 82) 

buffer_load_ubyte v22, v12, s[60:63], 0 offen offset:8 // (op 82) Load value
s_barrier  // (op 2398) 


v_mov_b64 v[54:55], v[16:17] // (op 1751) 

s_mov_b32 m0, s0 // (op 1751) reg expression
s_mov_b32 s7, 256 // (op 1751) call()
buffer_load_dwordx4 v54, s[52:55], s7 offen offset: 0 lds // (op 1751) Load value direct to lds



v_mov_b32 v19, v13 // (op 4) 

buffer_load_dwordx4 v[24:27], v19, s[48:51], 0 offen offset:192 // (op 4) Load value
buffer_load_dwordx2 v[28:29], v19, s[48:51], 0 offen offset:208 // (op 4) Load value
buffer_load_dwordx4 v[30:33], v19, s[48:51], 0 offen offset:216 // (op 4) Load value
buffer_load_dwordx2 v[34:35], v19, s[48:51], 0 offen offset:232 // (op 4) Load value

v_mov_b64 v[56:57], s[28:29] // (op 17) convert

v_mov_b32 v53, v20 // (op 17) 
v_mov_b32 v58, v53 // (op 17) 
buffer_load_ubyte v38, v58, s[56:59], 0 offen offset:8 // (op 17) Load value
v_add_u32 v58, v58, v56 // (op 17) 
buffer_load_ubyte v39, v58, s[56:59], 0 offen offset:8 // (op 17) Load value


s_barrier  // (op 2425) 



s_waitcnt vmcnt(13)
ds_write_b128 v15, v[40:43] offset:20992 // (op 229) Store local data 
s_waitcnt vmcnt(12)
ds_write_b64 v15, v[44:45] offset:21008 // (op 229) Store local data 
s_waitcnt vmcnt(11)
ds_write_b128 v15, v[46:49] offset:21016 // (op 229) Store local data 
s_waitcnt vmcnt(10)
ds_write_b64 v15, v[50:51] offset:21032 // (op 229) Store local data 
s_barrier  // (op 2437) 



s_waitcnt vmcnt(9)
ds_write_b8 v52, v36 offset:33280 // (op 240) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v52, v37 offset:33284 // (op 240) Store local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 2422) 



ds_read_b128 v[56:59], v8 offset:20992 // (op 231) Load local data 
ds_read_b64 v[60:61], v8 offset:21008 // (op 231) Load local data 
s_barrier  // (op 2428) 



ds_read_b128 v[62:65], v4 offset:4096 // (op 233) Load local data 
ds_read_b128 v[66:69], v4 offset:4160 // (op 233) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 2434) 



ds_read_u8 v70, v7 offset:33280 // (op 235) Load local data 



ds_read_b128 v[72:75], v8 offset:24064 // (op 357) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[56:61], v[62:69], a[0:3], v70, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 238) 
ds_read_b64 v[76:77], v8 offset:24080 // (op 357) Load local data 





ds_read_u8 v56, v7 offset:33408 // (op 361) Load local data 



ds_read_b128 v[78:81], v8 offset:27136 // (op 519) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[72:77], v[62:69], a[4:7], v56, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 351) 
ds_read_b64 v[82:83], v8 offset:27152 // (op 519) Load local data 





ds_read_u8 v56, v7 offset:33536 // (op 523) Load local data 



ds_read_b128 v[70:73], v8 offset:30208 // (op 681) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[78:83], v[62:69], a[8:11], v56, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 513) 
ds_read_b64 v[74:75], v8 offset:30224 // (op 681) Load local data 





s_barrier  // (op 2514) 
ds_read_u8 v56, v7 offset:33664 // (op 685) Load local data 
s_barrier  // (op 2518) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[70:75], v[62:69], a[12:15], v56, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 675) 






buffer_load_ubyte v23, v12, s[60:63], 0 offen offset:12 // (op 237) Load value

s_barrier  // (op 2431) 
v_add_u32 v21, 8, v21 // (op 2018) 


s_add_u32 s7, 4096, s1 // (op 1763) 

s_mov_b32 m0, s7 // (op 1763) reg expression
s_mov_b32 s10, 384 // (op 1763) call()
buffer_load_dwordx4 v54, s[52:55], s10 offen offset: 0 lds // (op 1763) Load value direct to lds






v_add_co_u32 v16, vcc, 256, v16 // (op 1890) least significant half
v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 1890) most significant half
buffer_load_dwordx4 v[40:43], v19, s[48:51], 0 offen offset:288 // (op 228) Load value
buffer_load_dwordx2 v[44:45], v19, s[48:51], 0 offen offset:304 // (op 228) Load value
buffer_load_dwordx4 v[46:49], v19, s[48:51], 0 offen offset:312 // (op 228) Load value
buffer_load_dwordx2 v[50:51], v19, s[48:51], 0 offen offset:328 // (op 228) Load value


v_add_u32 v13, 192, v13 // (op 1826) 
v_mov_b64 v[54:55], s[28:29] // (op 242) convert

v_mov_b32 v12, v53 // (op 242) 
buffer_load_ubyte v36, v12, s[56:59], 0 offen offset:12 // (op 242) Load value
v_add_u32 v12, v12, v54 // (op 242) 
buffer_load_ubyte v37, v12, s[56:59], 0 offen offset:12 // (op 242) Load value



s_barrier  // (op 2385) 
v_add_u32 v20, 8, v20 // (op 1954) 



s_waitcnt vmcnt(13)
ds_write_b128 v14, v[24:27] offset:8192 // (op 6) Store local data 
s_waitcnt vmcnt(12)
ds_write_b64 v14, v[28:29] offset:8208 // (op 6) Store local data 
s_waitcnt vmcnt(11)
ds_write_b128 v14, v[30:33] offset:8216 // (op 6) Store local data 
s_waitcnt vmcnt(10)
ds_write_b64 v14, v[34:35] offset:8232 // (op 6) Store local data 
s_barrier  // (op 2411) 



s_waitcnt vmcnt(9)
ds_write_b8 v18, v38 offset:20480 // (op 21) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v18, v39 offset:20484 // (op 21) Store local data 
s_ashr_i64 s[10:11], s[36:37], 7 // (op 638) 
s_mov_b32 s7, s10 // (op 638) convert

s_and_b32 s10, s7, 4294967294 // (op 638) 

s_cmp_lt_u32 s6, s10 // (op 638) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopTop_KLoop_638 // (op 638) Condition: Bottom (jump to RR_GEMM_TN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopTop_KLoop_638 if true)
RR_GEMM_TN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopBottom_KLoop_638:














s_barrier  // (op 2498) 
v_lshrrev_b32 v12, 1, v1 // (op 1786) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1786) 
v_lshl_add_u32 v13, s16, 7, v12 // (op 1786) 

s_lshr_b32 s6, s0, 1 // (op 1786) 

v_and_b32 v12, -2, v1 // (op 1786) 
v_sub_u32 v14, v1, v12 // (op 1786) 

v_lshlrev_b32 v12, 6, v14 // (op 1786) 

v_lshl_add_u32 v14, s6, 8, v12 // (op 1786) 


v_mov_b32 v39, 0 // (op 1786) convert
v_mov_b32 v38, v14 // (op 1786) convert

v_mul_lo_u32 v15, v13, s13 // (op 1786) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 1786) most significant: high of low * low
v_mul_lo_u32 v14, v13, s12 // (op 1786) least significant: low of low * low
v_add_u32 v15, v15, v12 // (op 1786) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v38 // (op 1786) least significant half
v_addc_co_u32 v15, s[0:1], v15, v39, s[0:1] // (op 1786) most significant half



v_mul_lo_u32 v13, v15, 6 // (op 1786) most significant: low of high * low
v_mul_hi_u32 v16, v14, 6 // (op 1786) most significant: high of low * low
v_mul_lo_u32 v12, v14, 6 // (op 1786) least significant: low of low * low
v_add_u32 v13, v13, v16 // (op 1786) most significant: sum


v_lshrrev_b64 v[38:39], 3, v[12:13] // (op 1786) 

v_mov_b32 v12, v38 // (op 1786) convert

s_mul_i32 s1, s23, 6 // (op 1788) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 1788) least significant
s_mul_hi_u32 s6, s22, 6 // (op 1788) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1788) most significant: sum

s_mul_i32 s7, s12, s1 // (op 1788) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 1788) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 1788) least significant
s_mul_hi_u32 s11, s12, s0 // (op 1788) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 1788) most significant: sum
s_add_u32 s7, s7, s11 // (op 1788) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 1788) 

s_mov_b32 s48, s20 // (op 1788) reg expression
s_mov_b32 s49, s21 // (op 1788) reg expression
s_mov_b32 s50, s0 // (op 1788) reg expression
s_mov_b32 s51, 131072 // (op 1788) call()

v_and_b32 v13, -8, v1 // (op 1856) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1856) 
v_sub_u32 v14, v1, v13 // (op 1856) 

s_lshr_b32 s6, s0, 1 // (op 1856) 

v_lshlrev_b32 v13, 4, v14 // (op 1856) 

v_lshl_add_u32 v14, s6, 8, v13 // (op 1856) 


v_mov_b32 v39, 0 // (op 1856) convert
v_mov_b32 v38, v14 // (op 1856) convert

v_lshrrev_b32 v13, 3, v1 // (op 1856) 
v_lshl_add_u32 v14, s17, 5, v13 // (op 1856) 

v_mul_lo_u32 v25, v14, s25 // (op 1856) most significant: low of low * high
v_mul_hi_u32 v13, v14, s24 // (op 1856) most significant: high of low * low
v_mul_lo_u32 v24, v14, s24 // (op 1856) least significant: low of low * low
v_add_u32 v25, v25, v13 // (op 1856) most significant: sum


v_add_co_u32 v26, s[0:1], v38, v24 // (op 1856) least significant half
v_addc_co_u32 v27, s[0:1], v39, v25, s[0:1] // (op 1856) most significant half



s_mul_i32 s1, s24, s9 // (op 1858) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 1858) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 1858) least significant
s_mul_hi_u32 s7, s24, s8 // (op 1858) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1858) most significant: sum
s_add_u32 s1, s1, s7 // (op 1858) most significant: sum


s_mov_b32 s52, s26 // (op 1858) reg expression
s_mov_b32 s53, s27 // (op 1858) reg expression
s_mov_b32 s54, s0 // (op 1858) reg expression
s_mov_b32 s55, 131072 // (op 1858) call()

v_lshrrev_b32 v13, 2, v1 // (op 1920) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1920) 
v_lshlrev_b32 v14, 1, v13 // (op 1920) 

s_lshr_b32 s6, s0, 1 // (op 1920) 

v_lshl_add_u32 v13, s16, 7, v14 // (op 1920) 

v_and_b32 v14, -4, v1 // (op 1920) 
v_sub_u32 v15, v1, v14 // (op 1920) 

v_lshl_add_u32 v14, s6, 3, v15 // (op 1920) 


v_mul_lo_u32 v15, v13, s28 // (op 1920) 
v_add_u32 v15, v15, v14 // (op 1920) 


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

v_and_b32 v13, 63, v1 // (op 1984) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1984) 
v_lshrrev_b32 v14, 4, v13 // (op 1984) 

s_lshr_b32 s6, s0, 1 // (op 1984) 

v_lshrrev_b32 v13, 6, v1 // (op 1984) 
v_lshl_add_u32 v16, s6, 3, v14 // (op 1984) 


v_and_b32 v14, 1, v13 // (op 1984) 

v_and_b32 v13, 15, v1 // (op 1984) 
v_lshl_add_u32 v17, v14, 4, v13 // (op 1984) 


v_lshl_add_u32 v13, s17, 5, v17 // (op 1984) 

v_mul_lo_u32 v14, v13, s32 // (op 1984) 

v_add_u32 v13, v16, v14 // (op 1984) 


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

v_and_b32 v14, -2, v1 // (op 2077) 
v_sub_u32 v16, v1, v14 // (op 2077) 

v_lshlrev_b32 v14, 6, v16 // (op 2077) 

v_lshrrev_b32 v16, 1, v1 // (op 2077) 
v_lshl_add_u32 v17, v16, 7, v14 // (op 2077) 


v_mul_lo_u32 v14, v17, 6 // (op 2077) 

v_lshrrev_b32 v16, 3, v14 // (op 2077) 

v_and_b32 v14, -8, v1 // (op 2135) 
v_sub_u32 v17, v1, v14 // (op 2135) 

v_lshlrev_b32 v14, 4, v17 // (op 2135) 

v_lshrrev_b32 v17, 3, v1 // (op 2135) 
v_lshl_add_u32 v18, v17, 7, v14 // (op 2135) 


s_nop 0
v_readfirstlane_b32 s0, v18 // (op 2135) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 2187) 
v_sub_u32 v17, v1, v14 // (op 2187) 

v_lshrrev_b32 v14, 2, v1 // (op 2187) 
v_lshl_add_u32 v18, v14, 3, v17 // (op 2187) 


s_ashr_i64 s[6:7], s[36:37], 7 // (op 644) 
s_and_b32 s1, s6, 4294967294 // (op 644) 

s_ashr_i64 s[6:7], s[36:37], 7 // (op 643) 
s_mov_b32 s10, s6 // (op 643) convert

s_cmp_lt_u32 s1, s10 // (op 643) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopBottom_KLoopTail_643 // (op 643) Condition: Top (jump to RR_GEMM_TN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopBottom_KLoopTail_643 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopTop_KLoopTail_643:


s_add_u32 s1, s1, 1 // (op 645) 
v_mov_b32 v14, v12 // (op 170) 

buffer_load_dwordx4 v[28:31], v14, s[48:51], 0 offen // (op 170) Load value
buffer_load_dwordx2 v[32:33], v14, s[48:51], 0 offen offset:16 // (op 170) Load value
buffer_load_dwordx4 v[34:37], v14, s[48:51], 0 offen offset:24 // (op 170) Load value
buffer_load_dwordx2 v[38:39], v14, s[48:51], 0 offen offset:40 // (op 170) Load value

v_add_u32 v12, 96, v12 // (op 1776) 
s_barrier  // (op 2388) 
s_barrier  // (op 2401) 





v_mov_b64 v[24:25], v[26:27] // (op 1756) 

s_mov_b32 m0, s0 // (op 1756) reg expression
buffer_load_dwordx4 v24, s[52:55], 0 offen offset: 0 lds // (op 1756) Load value direct to lds

v_add_co_u32 v26, vcc, 128, v26 // (op 1848) least significant half
v_addc_co_u32 v27, vcc, 0, v27, vcc // (op 1848) most significant half
s_waitcnt vmcnt(4)
ds_write_b128 v16, v[28:31] offset:8192 // (op 171) Store local data 

v_mov_b64 v[24:25], s[28:29] // (op 184) convert

v_mov_b32 v14, v15 // (op 184) 
v_mov_b32 v17, v14 // (op 184) 
buffer_load_ubyte v22, v17, s[56:59], 0 offen offset: 0 // (op 184) Load value
v_add_u32 v17, v17, v24 // (op 184) 
s_waitcnt vmcnt(4)
ds_write_b64 v16, v[32:33] offset:8208 // (op 171) Store local data 
s_waitcnt vmcnt(3)
ds_write_b128 v16, v[34:37] offset:8216 // (op 171) Store local data 
s_waitcnt vmcnt(2)
ds_write_b64 v16, v[38:39] offset:8232 // (op 171) Store local data 
buffer_load_ubyte v23, v17, s[56:59], 0 offen offset: 0 // (op 184) Load value




v_add_u32 v15, 4, v15 // (op 1912) 
s_barrier  // (op 2414) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 2391) 






ds_read_b128 v[28:31], v5 offset:8192 // (op 173) Load local data 
ds_read_b64 v[32:33], v5 offset:8208 // (op 173) Load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v18, v22 offset:20480 // (op 182) Store local data 
s_barrier  // (op 2404) 
s_waitcnt vmcnt(0)
ds_write_b8 v18, v23 offset:20484 // (op 182) Store local data 




ds_read_b128 v[34:37], v3 // (op 175) Load local data 
ds_read_b128 v[38:41], v3 offset:64 // (op 175) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 2418) 



ds_read_u8 v14, v0 offset:20480 // (op 177) Load local data 


v_mov_b32 v17, v13 // (op 179) 

buffer_load_ubyte v19, v17, s[60:63], 0 offen offset: 0 // (op 179) Load value

v_add_u32 v13, 4, v13 // (op 1976) 



ds_read_b128 v[20:23], v5 offset:11264 // (op 325) Load local data 
ds_read_b64 v[24:25], v5 offset:11280 // (op 325) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[28:33], v[34:41], a[0:3], v14, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 180) 





ds_read_u8 v14, v0 offset:20608 // (op 329) Load local data 



ds_read_b128 v[28:31], v5 offset:14336 // (op 487) Load local data 
ds_read_b64 v[32:33], v5 offset:14352 // (op 487) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[20:25], v[34:41], a[4:7], v14, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 331) 





ds_read_u8 v14, v0 offset:20736 // (op 491) Load local data 



ds_read_b128 v[20:23], v5 offset:17408 // (op 649) Load local data 
ds_read_b64 v[24:25], v5 offset:17424 // (op 649) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[28:33], v[34:41], a[8:11], v14, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 493) 





ds_read_u8 v14, v0 offset:20864 // (op 653) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[20:25], v[34:41], a[12:15], v14, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 655) 




s_ashr_i64 s[6:7], s[36:37], 7 // (op 643) 
s_mov_b32 s10, s6 // (op 643) convert

s_cmp_lt_u32 s1, s10 // (op 643) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopTop_KLoopTail_643 // (op 643) Condition: Bottom (jump to RR_GEMM_TN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopTop_KLoopTail_643 if true)
RR_GEMM_TN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopBottom_KLoopTail_643:






















s_barrier  // (op 2502) 







v_cmp_eq_f32 vcc, 0.00000, v10 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_TN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ConditionalFalse_0_beta_125
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


buffer_store_dwordx2 v[4:5], v9, s[44:47], 0 offen // (op 54) Store value


v_accvgpr_read v12, a4 // (op 885) 
v_accvgpr_read v13, a5 // (op 885) 
v_accvgpr_read v14, a6 // (op 885) 
v_accvgpr_read v15, a7 // (op 885) 
v_mul_f32 v16, v11, v12 // (op 885) 
v_mul_f32 v17, v11, v13 // (op 885) 
v_mul_f32 v18, v11, v14 // (op 885) 
v_mul_f32 v19, v11, v15 // (op 885) 

v_lshrrev_b32 v12, 16, v16 // (op 887) 
v_lshrrev_b32 v13, 16, v17 // (op 887) 
v_lshrrev_b32 v14, 16, v18 // (op 887) 
v_lshrrev_b32 v15, 16, v19 // (op 887) 


v_pack_b32_f16 v4, v12, v13 // (op 888) 
v_pack_b32_f16 v5, v14, v15 // (op 888) 


buffer_store_dwordx2 v[4:5], v9, s[44:47], 0 offen offset:64 // (op 888) Store value


v_accvgpr_read v12, a8 // (op 905) 
v_accvgpr_read v13, a9 // (op 905) 
v_accvgpr_read v14, a10 // (op 905) 
v_accvgpr_read v15, a11 // (op 905) 
v_mul_f32 v16, v11, v12 // (op 905) 
v_mul_f32 v17, v11, v13 // (op 905) 
v_mul_f32 v18, v11, v14 // (op 905) 
v_mul_f32 v19, v11, v15 // (op 905) 

v_lshrrev_b32 v12, 16, v16 // (op 907) 
v_lshrrev_b32 v13, 16, v17 // (op 907) 
v_lshrrev_b32 v14, 16, v18 // (op 907) 
v_lshrrev_b32 v15, 16, v19 // (op 907) 


v_pack_b32_f16 v4, v12, v13 // (op 908) 
v_pack_b32_f16 v5, v14, v15 // (op 908) 


buffer_store_dwordx2 v[4:5], v9, s[44:47], 0 offen offset:128 // (op 908) Store value


v_accvgpr_read v12, a12 // (op 925) 
v_accvgpr_read v13, a13 // (op 925) 
v_accvgpr_read v14, a14 // (op 925) 
v_accvgpr_read v15, a15 // (op 925) 
v_mul_f32 v16, v11, v12 // (op 925) 
v_mul_f32 v17, v11, v13 // (op 925) 
v_mul_f32 v18, v11, v14 // (op 925) 
v_mul_f32 v19, v11, v15 // (op 925) 

v_lshrrev_b32 v12, 16, v16 // (op 927) 
v_lshrrev_b32 v13, 16, v17 // (op 927) 
v_lshrrev_b32 v14, 16, v18 // (op 927) 
v_lshrrev_b32 v15, 16, v19 // (op 927) 


v_pack_b32_f16 v4, v12, v13 // (op 928) 
v_pack_b32_f16 v5, v14, v15 // (op 928) 


buffer_store_dwordx2 v[4:5], v9, s[44:47], 0 offen offset:192 // (op 928) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_TN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ConditionalBottom_0_beta_125
RR_GEMM_TN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ConditionalFalse_0_beta_125:



buffer_load_dwordx2 v[4:5], v6, s[40:43], 0 offen offset:192 // (op 1008) Load value
v_accvgpr_read v12, a12 // (op 1004) 
v_accvgpr_read v13, a13 // (op 1004) 
v_accvgpr_read v14, a14 // (op 1004) 
v_accvgpr_read v15, a15 // (op 1004) 
v_mul_f32 v16, v11, v12 // (op 1004) 
v_mul_f32 v17, v11, v13 // (op 1004) 
v_mul_f32 v18, v11, v14 // (op 1004) 
v_mul_f32 v19, v11, v15 // (op 1004) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v4 // (op 1009) 
v_lshrrev_b32 v13, 16, v4 // (op 1009) 
v_lshlrev_b32 v12, 16, v12 // (op 1009) 
v_lshlrev_b32 v13, 16, v13 // (op 1009) 
v_mul_f32 v20, v10, v12 // (op 1009) 
v_mul_f32 v21, v10, v13 // (op 1009) 
v_and_b32 v12, 65535, v5 // (op 1009) 
v_lshrrev_b32 v13, 16, v5 // (op 1009) 
v_lshlrev_b32 v12, 16, v12 // (op 1009) 
v_lshlrev_b32 v13, 16, v13 // (op 1009) 
v_mul_f32 v22, v10, v12 // (op 1009) 
v_mul_f32 v23, v10, v13 // (op 1009) 


v_add_f32 v12, v20, v16 // (op 1005) 
v_add_f32 v13, v21, v17 // (op 1005) 
v_add_f32 v14, v22, v18 // (op 1005) 
v_add_f32 v15, v23, v19 // (op 1005) 


v_lshrrev_b32 v16, 16, v12 // (op 1006) 
v_lshrrev_b32 v17, 16, v13 // (op 1006) 
v_lshrrev_b32 v18, 16, v14 // (op 1006) 
v_lshrrev_b32 v19, 16, v15 // (op 1006) 


v_pack_b32_f16 v4, v16, v17 // (op 1007) 
v_pack_b32_f16 v5, v18, v19 // (op 1007) 


buffer_store_dwordx2 v[4:5], v9, s[44:47], 0 offen offset:192 // (op 1007) Store value





buffer_load_dwordx2 v[4:5], v6, s[40:43], 0 offen offset:128 // (op 988) Load value
v_accvgpr_read v12, a8 // (op 984) 
v_accvgpr_read v13, a9 // (op 984) 
v_accvgpr_read v14, a10 // (op 984) 
v_accvgpr_read v15, a11 // (op 984) 
v_mul_f32 v16, v11, v12 // (op 984) 
v_mul_f32 v17, v11, v13 // (op 984) 
v_mul_f32 v18, v11, v14 // (op 984) 
v_mul_f32 v19, v11, v15 // (op 984) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v4 // (op 989) 
v_lshrrev_b32 v13, 16, v4 // (op 989) 
v_lshlrev_b32 v12, 16, v12 // (op 989) 
v_lshlrev_b32 v13, 16, v13 // (op 989) 
v_mul_f32 v20, v10, v12 // (op 989) 
v_mul_f32 v21, v10, v13 // (op 989) 
v_and_b32 v12, 65535, v5 // (op 989) 
v_lshrrev_b32 v13, 16, v5 // (op 989) 
v_lshlrev_b32 v12, 16, v12 // (op 989) 
v_lshlrev_b32 v13, 16, v13 // (op 989) 
v_mul_f32 v22, v10, v12 // (op 989) 
v_mul_f32 v23, v10, v13 // (op 989) 


v_add_f32 v12, v20, v16 // (op 985) 
v_add_f32 v13, v21, v17 // (op 985) 
v_add_f32 v14, v22, v18 // (op 985) 
v_add_f32 v15, v23, v19 // (op 985) 


v_lshrrev_b32 v16, 16, v12 // (op 986) 
v_lshrrev_b32 v17, 16, v13 // (op 986) 
v_lshrrev_b32 v18, 16, v14 // (op 986) 
v_lshrrev_b32 v19, 16, v15 // (op 986) 


v_pack_b32_f16 v4, v16, v17 // (op 987) 
v_pack_b32_f16 v5, v18, v19 // (op 987) 


buffer_store_dwordx2 v[4:5], v9, s[44:47], 0 offen offset:128 // (op 987) Store value





buffer_load_dwordx2 v[4:5], v6, s[40:43], 0 offen offset:64 // (op 968) Load value
v_accvgpr_read v12, a4 // (op 964) 
v_accvgpr_read v13, a5 // (op 964) 
v_accvgpr_read v14, a6 // (op 964) 
v_accvgpr_read v15, a7 // (op 964) 
v_mul_f32 v16, v11, v12 // (op 964) 
v_mul_f32 v17, v11, v13 // (op 964) 
v_mul_f32 v18, v11, v14 // (op 964) 
v_mul_f32 v19, v11, v15 // (op 964) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v4 // (op 969) 
v_lshrrev_b32 v13, 16, v4 // (op 969) 
v_lshlrev_b32 v12, 16, v12 // (op 969) 
v_lshlrev_b32 v13, 16, v13 // (op 969) 
v_mul_f32 v20, v10, v12 // (op 969) 
v_mul_f32 v21, v10, v13 // (op 969) 
v_and_b32 v12, 65535, v5 // (op 969) 
v_lshrrev_b32 v13, 16, v5 // (op 969) 
v_lshlrev_b32 v12, 16, v12 // (op 969) 
v_lshlrev_b32 v13, 16, v13 // (op 969) 
v_mul_f32 v22, v10, v12 // (op 969) 
v_mul_f32 v23, v10, v13 // (op 969) 


v_add_f32 v12, v20, v16 // (op 965) 
v_add_f32 v13, v21, v17 // (op 965) 
v_add_f32 v14, v22, v18 // (op 965) 
v_add_f32 v15, v23, v19 // (op 965) 


v_lshrrev_b32 v16, 16, v12 // (op 966) 
v_lshrrev_b32 v17, 16, v13 // (op 966) 
v_lshrrev_b32 v18, 16, v14 // (op 966) 
v_lshrrev_b32 v19, 16, v15 // (op 966) 


v_pack_b32_f16 v4, v16, v17 // (op 967) 
v_pack_b32_f16 v5, v18, v19 // (op 967) 


buffer_store_dwordx2 v[4:5], v9, s[44:47], 0 offen offset:64 // (op 967) Store value





buffer_load_dwordx2 v[4:5], v6, s[40:43], 0 offen // (op 138) Load value
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


buffer_store_dwordx2 v[4:5], v9, s[44:47], 0 offen // (op 137) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ConditionalBottom_0_beta_125:












s_endpgm  // End of RR_GEMM_TN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2
.LRR_GEMM_TN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_end:
.size RR_GEMM_TN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2, .LRR_GEMM_TN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_end-RR_GEMM_TN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2
  .amdhsa_next_free_vgpr 100
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 33792 // lds bytes
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
  MXScaleFormat: NoSwizzle
  ProblemType:
    OperationType: GEMM
    DataType: F6
    DataTypeA: F6
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
    macrotile: [128, 32, 128]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [4, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2
    .symbol: RR_GEMM_TN_FP6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 33792
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 64
    .vgpr_count: 84
    .agpr_count: 16
    .max_flat_workgroup_size: 256
    .wavefront_size: 64
    .args:
      - .name: Tensor_10_stride_1_10
        .size: 8
        .offset: 0
        .value_kind: by_value
      - .name: Tensor_21_stride_1_16
        .size: 8
        .offset: 8
        .value_kind: by_value
      - .name: Tensor_10_pointer
        .size: 8
        .offset: 16
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_2_size_1_4
        .size: 8
        .offset: 24
        .value_kind: by_value
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 32
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_stride_0_0
        .size: 8
        .offset: 40
        .value_kind: by_value
      - .name: user_Float_Value_14
        .size: 4
        .offset: 48
        .value_kind: by_value
      - .name: user_Float_Value_12
        .size: 4
        .offset: 52
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 56
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_0_1
        .size: 8
        .offset: 64
        .value_kind: by_value
      - .name: Tensor_2_stride_1_3
        .size: 8
        .offset: 72
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 80
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_stride_0_6
        .size: 8
        .offset: 88
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 96
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_7_stride_1_8
        .size: 8
        .offset: 104
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 112
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_1_14
        .size: 8
        .offset: 120
        .value_kind: by_value
...

.end_amdgpu_metadata

