// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2
.p2align 8
.type RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2,@function
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 2069) 
v_and_b32 v3, 15, v1 // (op 2069) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 2069) 


v_and_b32 v0, 48, v1 // (op 2069) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 2069) 


v_lshrrev_b32 v0, 7, v1 // (op 2241) 
v_and_b32 v4, 15, v1 // (op 2241) 
v_lshl_add_u32 v5, v0, 4, v4 // (op 2241) 


v_and_b32 v0, 48, v1 // (op 2241) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 2241) 


v_lshrrev_b32 v0, 7, v1 // (op 2013) 
v_and_b32 v5, 63, v1 // (op 2013) 
v_lshrrev_b32 v6, 4, v5 // (op 2013) 
v_lshlrev_b32 v7, 2, v6 // (op 2013) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 2013) 


v_lshl_add_u32 v0, s16, 5, v6 // (op 2013) 

v_lshrrev_b32 v6, 2, v5 // (op 2013) 

v_and_b32 v5, 3, v6 // (op 2013) 

v_and_b32 v6, 3, v1 // (op 2013) 
v_lshl_add_u32 v7, v5, 2, v6 // (op 2013) 


v_lshrrev_b32 v5, 6, v1 // (op 2013) 
v_and_b32 v6, 1, v5 // (op 2013) 

v_lshl_add_u32 v5, v6, 4, v7 // (op 2013) 


v_lshl_add_u32 v6, s17, 7, v5 // (op 2013) 

v_mul_lo_u32 v5, v6, s2 // (op 2013) 

v_add_lshl_u32 v6, v0, v5, 2 // (op 2013) 


v_lshrrev_b32 v0, 6, v1 // (op 2126) 
v_and_b32 v5, 1, v0 // (op 2126) 

v_and_b32 v0, 15, v1 // (op 2126) 
v_lshl_add_u32 v7, v5, 4, v0 // (op 2126) 


v_and_b32 v0, 48, v1 // (op 2126) 
v_lshl_add_u32 v5, v7, 7, v0 // (op 2126) 


v_lshrrev_b32 v0, 6, v1 // (op 2184) 
v_and_b32 v7, 1, v0 // (op 2184) 

v_and_b32 v0, 15, v1 // (op 2184) 
s_mul_i32 s1, s2, s9 // (op 2015) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 2015) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 2015) least significant
s_mul_hi_u32 s19, s2, s8 // (op 2015) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2015) most significant: sum
s_add_u32 s1, s1, s19 // (op 2015) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 2015) 

v_lshl_add_u32 v8, v7, 4, v0 // (op 2184) 


v_and_b32 v0, 63, v1 // (op 2184) 
v_lshrrev_b32 v7, 4, v0 // (op 2184) 

v_lshl_add_u32 v0, v8, 2, v7 // (op 2184) 


v_lshrrev_b32 v7, 7, v1 // (op 2200) 
v_and_b32 v8, 63, v1 // (op 2200) 
v_lshrrev_b32 v9, 4, v8 // (op 2200) 
v_lshlrev_b32 v10, 2, v9 // (op 2200) 

v_lshl_add_u32 v9, v7, 4, v10 // (op 2200) 


v_lshl_add_u32 v7, s16, 5, v9 // (op 2200) 

v_lshrrev_b32 v9, 2, v8 // (op 2200) 

v_and_b32 v8, 3, v9 // (op 2200) 

v_and_b32 v9, 3, v1 // (op 2200) 
v_lshl_add_u32 v10, v8, 2, v9 // (op 2200) 


v_lshrrev_b32 v8, 6, v1 // (op 2200) 
v_and_b32 v9, 1, v8 // (op 2200) 

v_lshl_add_u32 v8, v9, 4, v10 // (op 2200) 


v_lshl_add_u32 v9, s17, 7, v8 // (op 2200) 

v_mul_lo_u32 v8, v9, s4 // (op 2200) 

v_add_lshl_u32 v9, v7, v8, 2 // (op 2200) 


v_lshrrev_b32 v7, 6, v1 // (op 2285) 
v_and_b32 v8, 1, v7 // (op 2285) 

v_and_b32 v7, 15, v1 // (op 2285) 
v_lshl_add_u32 v10, v8, 4, v7 // (op 2285) 


v_and_b32 v7, 48, v1 // (op 2285) 
v_lshl_add_u32 v8, v10, 7, v7 // (op 2285) 


v_lshrrev_b32 v7, 6, v1 // (op 2314) 
v_and_b32 v10, 1, v7 // (op 2314) 

v_and_b32 v7, 15, v1 // (op 2314) 
s_mul_i32 s1, s4, s9 // (op 2202) most significant: low of low * high
v_lshl_add_u32 v11, v10, 4, v7 // (op 2314) 


v_and_b32 v7, 63, v1 // (op 2314) 
v_lshrrev_b32 v10, 4, v7 // (op 2314) 

v_lshl_add_u32 v7, v11, 2, v10 // (op 2314) 


s_mul_i32 s38, s5, s8 // (op 2202) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 2202) least significant
s_mul_hi_u32 s39, s4, s8 // (op 2202) most significant: high of low * low
s_add_u32 s1, s1, s38 // (op 2202) most significant: sum
s_add_u32 s1, s1, s39 // (op 2202) most significant: sum


s_lshl_b64 s[38:39], s[0:1], 2 // (op 2202) 

s_mov_b32 s40, s6 // (op 2015) reg expression
s_mov_b32 s41, s7 // (op 2015) reg expression
s_mov_b32 s42, s18 // (op 2015) reg expression
s_mov_b32 s43, 131072 // (op 2015) call()

s_mov_b32 s44, s10 // (op 2202) reg expression

s_mov_b32 s45, s11 // (op 2202) reg expression
s_mov_b32 s46, s38 // (op 2202) reg expression
s_mov_b32 s47, 131072 // (op 2202) call()


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
v_lshrrev_b32 v12, 3, v1 // (op 1780) 
v_lshl_add_u32 v13, s16, 5, v12 // (op 1780) 

v_and_b32 v12, -8, v1 // (op 1780) 
v_sub_u32 v14, v1, v12 // (op 1780) 

v_lshlrev_b32 v12, 4, v14 // (op 1780) 

v_mov_b32 v15, 0 // (op 1780) convert
v_mov_b32 v14, v12 // (op 1780) convert

v_mul_lo_u32 v17, v13, s13 // (op 1780) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 1780) most significant: high of low * low
v_mul_lo_u32 v16, v13, s12 // (op 1780) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 1780) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 1780) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 1780) most significant half




s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 1782) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 1782) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 1782) least significant
s_mul_hi_u32 s7, s12, s22 // (op 1782) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1782) most significant: sum
s_add_u32 s1, s1, s7 // (op 1782) most significant: sum


s_mov_b32 s48, s20 // (op 1782) reg expression
s_mov_b32 s49, s21 // (op 1782) reg expression
s_mov_b32 s50, s0 // (op 1782) reg expression
s_mov_b32 s51, 131072 // (op 1782) call()

v_lshrrev_b32 v12, 3, v1 // (op 1799) 
v_lshl_add_u32 v13, s16, 5, v12 // (op 1799) 

v_and_b32 v12, -8, v1 // (op 1799) 
v_sub_u32 v14, v1, v12 // (op 1799) 

v_lshlrev_b32 v12, 4, v14 // (op 1799) 

v_mov_b32 v15, 0 // (op 1799) convert
v_mov_b32 v14, v12 // (op 1799) convert

v_mul_lo_u32 v19, v13, s13 // (op 1799) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 1799) most significant: high of low * low
v_mul_lo_u32 v18, v13, s12 // (op 1799) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 1799) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v14 // (op 1799) least significant half
v_addc_co_u32 v19, s[0:1], v19, v15, s[0:1] // (op 1799) most significant half



s_mul_i32 s1, s12, s23 // (op 1801) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 1801) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 1801) least significant
s_mul_hi_u32 s7, s12, s22 // (op 1801) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1801) most significant: sum
s_add_u32 s1, s1, s7 // (op 1801) most significant: sum


s_mov_b32 s48, s20 // (op 1801) reg expression
s_mov_b32 s49, s21 // (op 1801) reg expression
s_mov_b32 s50, s0 // (op 1801) reg expression
s_mov_b32 s51, 131072 // (op 1801) call()

v_and_b32 v12, -8, v1 // (op 1844) 
v_sub_u32 v13, v1, v12 // (op 1844) 

v_lshlrev_b32 v12, 4, v13 // (op 1844) 

v_mov_b32 v15, 0 // (op 1844) convert
v_mov_b32 v14, v12 // (op 1844) convert

v_lshrrev_b32 v12, 3, v1 // (op 1844) 
v_lshl_add_u32 v13, s17, 7, v12 // (op 1844) 

v_mul_lo_u32 v21, v13, s25 // (op 1844) most significant: low of low * high
v_mul_hi_u32 v12, v13, s24 // (op 1844) most significant: high of low * low
v_mul_lo_u32 v20, v13, s24 // (op 1844) least significant: low of low * low
v_add_u32 v21, v21, v12 // (op 1844) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v20 // (op 1844) least significant half
v_addc_co_u32 v13, s[0:1], v15, v21, s[0:1] // (op 1844) most significant half



s_mul_i32 s1, s24, s9 // (op 1846) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 1846) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 1846) least significant
s_mul_hi_u32 s7, s24, s8 // (op 1846) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1846) most significant: sum
s_add_u32 s1, s1, s7 // (op 1846) most significant: sum


s_mov_b32 s52, s26 // (op 1846) reg expression
s_mov_b32 s53, s27 // (op 1846) reg expression
s_mov_b32 s54, s0 // (op 1846) reg expression
s_mov_b32 s55, 131072 // (op 1846) call()

v_and_b32 v14, -8, v1 // (op 1863) 
v_sub_u32 v15, v1, v14 // (op 1863) 

v_lshlrev_b32 v14, 4, v15 // (op 1863) 

v_mov_b32 v21, 0 // (op 1863) convert
v_mov_b32 v20, v14 // (op 1863) convert

v_lshrrev_b32 v14, 3, v1 // (op 1863) 
v_lshl_add_u32 v15, s17, 7, v14 // (op 1863) 

v_mul_lo_u32 v23, v15, s25 // (op 1863) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 1863) most significant: high of low * low
v_mul_lo_u32 v22, v15, s24 // (op 1863) least significant: low of low * low
v_add_u32 v23, v23, v14 // (op 1863) most significant: sum


v_add_co_u32 v14, s[0:1], v20, v22 // (op 1863) least significant half
v_addc_co_u32 v15, s[0:1], v21, v23, s[0:1] // (op 1863) most significant half



s_mul_i32 s1, s24, s9 // (op 1865) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 1865) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 1865) least significant
s_mul_hi_u32 s7, s24, s8 // (op 1865) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1865) most significant: sum
s_add_u32 s1, s1, s7 // (op 1865) most significant: sum


s_mov_b32 s52, s26 // (op 1865) reg expression
s_mov_b32 s53, s27 // (op 1865) reg expression
s_mov_b32 s54, s0 // (op 1865) reg expression
s_mov_b32 s55, 131072 // (op 1865) call()

v_lshrrev_b32 v20, 7, v1 // (op 1908) 
v_and_b32 v21, 15, v1 // (op 1908) 
v_lshl_add_u32 v22, v20, 4, v21 // (op 1908) 


v_lshl_add_u32 v20, s16, 5, v22 // (op 1908) 

v_and_b32 v21, 63, v1 // (op 1908) 
v_lshrrev_b32 v22, 4, v21 // (op 1908) 

v_mul_lo_u32 v21, v20, s28 // (op 1908) 
v_add_u32 v21, v21, v22 // (op 1908) 


s_mul_i32 s1, s28, s23 // (op 1910) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 1910) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 1910) least significant
s_mul_hi_u32 s7, s28, s22 // (op 1910) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1910) most significant: sum
s_add_u32 s1, s1, s7 // (op 1910) most significant: sum


s_mov_b32 s56, s30 // (op 1910) reg expression
s_mov_b32 s57, s31 // (op 1910) reg expression
s_mov_b32 s58, s0 // (op 1910) reg expression
s_mov_b32 s59, 131072 // (op 1910) call()

v_lshrrev_b32 v20, 7, v1 // (op 1927) 
v_and_b32 v22, 15, v1 // (op 1927) 
v_lshl_add_u32 v23, v20, 4, v22 // (op 1927) 


v_lshl_add_u32 v20, s16, 5, v23 // (op 1927) 

v_and_b32 v22, 63, v1 // (op 1927) 
v_lshrrev_b32 v23, 4, v22 // (op 1927) 

v_mul_lo_u32 v22, v20, s28 // (op 1927) 
v_add_u32 v22, v22, v23 // (op 1927) 


s_mul_i32 s1, s28, s23 // (op 1929) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 1929) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 1929) least significant
s_mul_hi_u32 s7, s28, s22 // (op 1929) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1929) most significant: sum
s_add_u32 s1, s1, s7 // (op 1929) most significant: sum


s_mov_b32 s56, s30 // (op 1929) reg expression
s_mov_b32 s57, s31 // (op 1929) reg expression
s_mov_b32 s58, s0 // (op 1929) reg expression
s_mov_b32 s59, 131072 // (op 1929) call()

v_and_b32 v20, -4, v1 // (op 1972) 
v_sub_u32 v23, v1, v20 // (op 1972) 

v_lshrrev_b32 v20, 2, v1 // (op 1972) 
v_lshlrev_b32 v24, 1, v20 // (op 1972) 

v_lshl_add_u32 v20, s17, 7, v24 // (op 1972) 

v_mul_lo_u32 v24, v20, s32 // (op 1972) 

v_add_u32 v20, v23, v24 // (op 1972) 


s_mul_i32 s1, s32, s9 // (op 1974) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 1974) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 1974) least significant
s_mul_hi_u32 s7, s32, s8 // (op 1974) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1974) most significant: sum
s_add_u32 s1, s1, s7 // (op 1974) most significant: sum


s_mov_b32 s60, s34 // (op 1974) reg expression
s_mov_b32 s61, s35 // (op 1974) reg expression
s_mov_b32 s62, s0 // (op 1974) reg expression
s_mov_b32 s63, 131072 // (op 1974) call()

v_and_b32 v23, -4, v1 // (op 1991) 
v_sub_u32 v24, v1, v23 // (op 1991) 

v_lshrrev_b32 v23, 2, v1 // (op 1991) 
v_lshlrev_b32 v25, 1, v23 // (op 1991) 

v_lshl_add_u32 v23, s17, 7, v25 // (op 1991) 

v_mul_lo_u32 v25, v23, s32 // (op 1991) 

v_add_u32 v23, v24, v25 // (op 1991) 


s_mul_i32 s1, s32, s9 // (op 1993) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 1993) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 1993) least significant
s_mul_hi_u32 s7, s32, s8 // (op 1993) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1993) most significant: sum
s_add_u32 s1, s1, s7 // (op 1993) most significant: sum


s_mov_b32 s60, s34 // (op 1993) reg expression
s_mov_b32 s61, s35 // (op 1993) reg expression
s_mov_b32 s62, s0 // (op 1993) reg expression
s_mov_b32 s63, 131072 // (op 1993) call()




buffer_load_ubyte v24, v21, s[56:59], 0 offen offset: 0 // (op 1544) Load value
v_and_b32 v25, -8, v1 // (op 2109) 
v_sub_u32 v26, v1, v25 // (op 2109) 

v_lshlrev_b32 v25, 4, v26 // (op 2109) 

v_lshrrev_b32 v26, 3, v1 // (op 2109) 
v_lshl_add_u32 v27, v26, 7, v25 // (op 2109) 


s_nop 0
v_readfirstlane_b32 s0, v27 // (op 2109) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 1726) 
v_mov_b64 v[26:27], s[6:7] // (op 1726) convert


v_mov_b64 v[28:29], v[12:13] // (op 1726) 
s_mov_b32 m0, s0 // (op 1726) reg expression
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 1726) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 1726) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 1726) most significant half

s_add_u32 m0, m0, 4096 // (op 1726) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 1726) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 1726) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 1726) most significant half

s_add_u32 m0, m0, 4096 // (op 1726) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 1726) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 1726) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 1726) most significant half

s_add_u32 m0, m0, 4096 // (op 1726) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 1726) Load value direct to lds




v_mov_b64 v[26:27], s[32:33] // (op 1551) convert

v_mov_b32 v25, v20 // (op 1551) 
buffer_load_ubyte v28, v25, s[60:63], 0 offen offset: 0 // (op 1551) Load value
v_add_u32 v25, v25, v26 // (op 1551) 
buffer_load_ubyte v29, v25, s[60:63], 0 offen offset: 0 // (op 1551) Load value


v_and_b32 v25, -8, v1 // (op 2057) 
v_sub_u32 v26, v1, v25 // (op 2057) 

v_lshlrev_b32 v25, 4, v26 // (op 2057) 

v_lshrrev_b32 v26, 3, v1 // (op 2057) 
v_lshl_add_u32 v27, v26, 7, v25 // (op 2057) 


s_nop 0
v_readfirstlane_b32 s0, v27 // (op 2057) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s1, 16384, s0 // (op 1731) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 1731) reg expression
buffer_load_dwordx4 v16, s[48:51], 0 offen offset: 0 lds // (op 1731) Load value direct to lds





buffer_load_ubyte v25, v21, s[56:59], 0 offen offset:4 // (op 1558) Load value

v_and_b32 v21, -8, v1 // (op 2268) 
v_sub_u32 v26, v1, v21 // (op 2268) 

v_lshlrev_b32 v21, 4, v26 // (op 2268) 

v_lshrrev_b32 v26, 3, v1 // (op 2268) 
v_lshl_add_u32 v27, v26, 7, v21 // (op 2268) 


s_nop 0
v_readfirstlane_b32 s0, v27 // (op 2268) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 1736) 
v_mov_b64 v[26:27], s[6:7] // (op 1736) convert


v_mov_b64 v[30:31], v[12:13] // (op 1736) 
s_add_u32 s1, 20480, s0 // (op 1736) 
s_mov_b32 m0, s1 // (op 1736) reg expression
s_mov_b32 s6, 128 // (op 1736) call()
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 1736) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v26 // (op 1736) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 1736) most significant half

s_add_u32 m0, m0, 4096 // (op 1736) 
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 1736) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v26 // (op 1736) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 1736) most significant half

s_add_u32 m0, m0, 4096 // (op 1736) 
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 1736) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v26 // (op 1736) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 1736) most significant half

s_add_u32 m0, m0, 4096 // (op 1736) 
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 1736) Load value direct to lds







v_mov_b64 v[12:13], s[32:33] // (op 1565) convert

v_mov_b32 v21, v20 // (op 1565) 
buffer_load_ubyte v26, v21, s[60:63], 0 offen offset:4 // (op 1565) Load value
v_add_u32 v21, v21, v12 // (op 1565) 
buffer_load_ubyte v27, v21, s[60:63], 0 offen offset:4 // (op 1565) Load value



v_and_b32 v12, -8, v1 // (op 2229) 
v_sub_u32 v13, v1, v12 // (op 2229) 

v_lshlrev_b32 v12, 4, v13 // (op 2229) 

v_lshrrev_b32 v13, 3, v1 // (op 2229) 
v_lshl_add_u32 v20, v13, 7, v12 // (op 2229) 


s_nop 0
v_readfirstlane_b32 s0, v20 // (op 2229) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s1, 36864, s0 // (op 1738) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 1738) reg expression
s_mov_b32 s6, 128 // (op 1738) call()
buffer_load_dwordx4 v16, s[48:51], s6 offen offset: 0 lds // (op 1738) Load value direct to lds




v_and_b32 v12, -4, v1 // (op 2167) 
v_sub_u32 v13, v1, v12 // (op 2167) 

v_lshrrev_b32 v12, 2, v1 // (op 2167) 
v_lshl_add_u32 v16, v12, 3, v13 // (op 2167) 





s_waitcnt vmcnt(10)
ds_write_b8 v16, v28 offset:40960 // (op 1571) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v16, v29 offset:40964 // (op 1571) Store local data 

s_barrier  // (op 2377) 



ds_read_b128 v[30:33], v5 // (op 1589) Load local data 
ds_read_b128 v[34:37], v5 offset:64 // (op 1589) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 2393) 



ds_read_u8 v12, v0 offset:40960 // (op 1609) Load local data 



ds_read_b128 v[38:41], v5 offset:4096 // (op 1594) Load local data 
ds_read_b128 v[42:45], v5 offset:4160 // (op 1594) Load local data 



ds_read_u8 v13, v0 offset:41088 // (op 1614) Load local data 



ds_read_b128 v[46:49], v5 offset:8192 // (op 1599) Load local data 
ds_read_b128 v[50:53], v5 offset:8256 // (op 1599) Load local data 



ds_read_u8 v16, v0 offset:41216 // (op 1619) Load local data 



ds_read_b128 v[54:57], v5 offset:12288 // (op 1604) Load local data 
ds_read_b128 v[58:61], v5 offset:12352 // (op 1604) Load local data 



ds_read_u8 v17, v0 offset:41344 // (op 1624) Load local data 
s_waitcnt vmcnt(8)
s_barrier  // (op 2361) 



ds_read_b128 v[62:65], v3 offset:16384 // (op 1585) Load local data 
ds_read_b128 v[66:69], v3 offset:16448 // (op 1585) Load local data 
v_and_b32 v20, -8, v1 // (op 2029) 
v_sub_u32 v21, v1, v20 // (op 2029) 

v_lshlrev_b32 v20, 4, v21 // (op 2029) 

v_lshrrev_b32 v21, 3, v1 // (op 2029) 
v_lshl_add_u32 v70, v21, 7, v20 // (op 2029) 


s_nop 0
v_readfirstlane_b32 s0, v70 // (op 2029) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v20, -8, v1 // (op 2081) 
v_sub_u32 v21, v1, v20 // (op 2081) 

v_lshlrev_b32 v20, 4, v21 // (op 2081) 

v_lshrrev_b32 v21, 3, v1 // (op 2081) 
v_lshl_add_u32 v70, v21, 7, v20 // (op 2081) 


s_nop 0
v_readfirstlane_b32 s1, v70 // (op 2081) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v20, -4, v1 // (op 2138) 
v_sub_u32 v21, v1, v20 // (op 2138) 

v_lshrrev_b32 v20, 2, v1 // (op 2138) 
v_lshl_add_u32 v70, v20, 3, v21 // (op 2138) 


v_and_b32 v20, -8, v1 // (op 2214) 
v_sub_u32 v21, v1, v20 // (op 2214) 

v_lshlrev_b32 v20, 4, v21 // (op 2214) 

v_lshrrev_b32 v21, 3, v1 // (op 2214) 
v_lshl_add_u32 v71, v21, 7, v20 // (op 2214) 


s_nop 0
v_readfirstlane_b32 s6, v71 // (op 2214) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v20, -8, v1 // (op 2253) 
v_sub_u32 v21, v1, v20 // (op 2253) 

v_lshlrev_b32 v20, 4, v21 // (op 2253) 

v_lshrrev_b32 v21, 3, v1 // (op 2253) 
v_lshl_add_u32 v71, v21, 7, v20 // (op 2253) 


s_nop 0
v_readfirstlane_b32 s7, v71 // (op 2253) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v20, -4, v1 // (op 2297) 
v_sub_u32 v21, v1, v20 // (op 2297) 

v_lshrrev_b32 v20, 2, v1 // (op 2297) 
v_lshl_add_u32 v71, v20, 3, v21 // (op 2297) 


s_mov_b32 s10, 0 // (op 620) call()
s_ashr_i64 s[14:15], s[36:37], 7 // (op 619) 
s_mov_b32 s11, s14 // (op 619) convert

s_and_b32 s14, s11, 4294967294 // (op 619) 

s_cmp_lt_u32 s10, s14 // (op 619) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopBottom_KLoop_619 // (op 619) Condition: Top (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopBottom_KLoop_619 if false)
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopTop_KLoop_619:
s_add_u32 s10, s10, 2 // (op 621) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:69], v[30:37], a[0:3], v24, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 82) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[62:69], v[38:45], a[4:7], v24, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 343) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[62:69], v[46:53], a[8:11], v24, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 499) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[62:69], v[54:61], a[12:15], v24, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 655) 


v_mov_b32 v20, v22 // (op 75) 

buffer_load_ubyte v24, v20, s[56:59], 0 offen offset:8 // (op 75) Load value
s_barrier  // (op 2380) 

s_lshl_b64 s[14:15], s[24:25], 5 // (op 1698) 
v_mov_b64 v[72:73], s[14:15] // (op 1698) convert


v_mov_b64 v[74:75], v[14:15] // (op 1698) 
v_mov_b64 v[76:77], v[74:75] // (op 1698) 
s_mov_b32 m0, s1 // (op 1698) reg expression
s_mov_b32 s11, 256 // (op 1698) call()
buffer_load_dwordx4 v76, s[52:55], s11 offen offset: 0 lds // (op 1698) Load value direct to lds
v_add_co_u32 v76, s[14:15], v76, v72 // (op 1698) least significant half
v_addc_co_u32 v77, s[14:15], 0, v77, s[14:15] // (op 1698) most significant half

s_add_u32 m0, m0, 4096 // (op 1698) 
buffer_load_dwordx4 v76, s[52:55], s11 offen offset: 0 lds // (op 1698) Load value direct to lds
v_add_co_u32 v76, s[14:15], v76, v72 // (op 1698) least significant half
v_addc_co_u32 v77, s[14:15], 0, v77, s[14:15] // (op 1698) most significant half

s_add_u32 m0, m0, 4096 // (op 1698) 
buffer_load_dwordx4 v76, s[52:55], s11 offen offset: 0 lds // (op 1698) Load value direct to lds
v_add_co_u32 v76, s[14:15], v76, v72 // (op 1698) least significant half
v_addc_co_u32 v77, s[14:15], 0, v77, s[14:15] // (op 1698) most significant half

s_add_u32 m0, m0, 4096 // (op 1698) 
buffer_load_dwordx4 v76, s[52:55], s11 offen offset: 0 lds // (op 1698) Load value direct to lds




v_mov_b64 v[72:73], s[32:33] // (op 21) convert

v_mov_b32 v21, v23 // (op 21) 
v_mov_b32 v76, v21 // (op 21) 
buffer_load_ubyte v28, v76, s[60:63], 0 offen offset:8 // (op 21) Load value
v_add_u32 v76, v76, v72 // (op 21) 
buffer_load_ubyte v29, v76, s[60:63], 0 offen offset:8 // (op 21) Load value


s_barrier  // (op 2364) 


v_mov_b64 v[72:73], v[18:19] // (op 1693) 
s_add_u32 s11, 16384, s0 // (op 1693) 

s_mov_b32 m0, s11 // (op 1693) reg expression
s_mov_b32 s14, 256 // (op 1693) call()
buffer_load_dwordx4 v72, s[48:51], s14 offen offset: 0 lds // (op 1693) Load value direct to lds


s_barrier  // (op 2425) 



s_waitcnt vmcnt(10)
ds_write_b8 v71, v26 offset:41472 // (op 236) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v71, v27 offset:41476 // (op 236) Store local data 
s_barrier  // (op 2416) 



ds_read_b128 v[30:33], v8 offset:20480 // (op 229) Load local data 
ds_read_b128 v[34:37], v8 offset:20544 // (op 229) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 2422) 



ds_read_u8 v12, v7 offset:41472 // (op 233) Load local data 



ds_read_b128 v[38:41], v8 offset:24576 // (op 348) Load local data 
ds_read_b128 v[42:45], v8 offset:24640 // (op 348) Load local data 



ds_read_u8 v13, v7 offset:41600 // (op 351) Load local data 



ds_read_b128 v[46:49], v8 offset:28672 // (op 504) Load local data 
ds_read_b128 v[50:53], v8 offset:28736 // (op 504) Load local data 



ds_read_u8 v16, v7 offset:41728 // (op 507) Load local data 



ds_read_b128 v[54:57], v8 offset:32768 // (op 660) Load local data 
ds_read_b128 v[58:61], v8 offset:32832 // (op 660) Load local data 



ds_read_u8 v17, v7 offset:41856 // (op 663) Load local data 
s_barrier  // (op 2484) 
s_waitcnt vmcnt(8)
s_barrier  // (op 2410) 



ds_read_b128 v[62:65], v4 offset:36864 // (op 227) Load local data 
ds_read_b128 v[66:69], v4 offset:36928 // (op 227) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:69], v[30:37], a[0:3], v25, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 234) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[62:69], v[38:45], a[4:7], v25, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 344) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[62:69], v[46:53], a[8:11], v25, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 500) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[62:69], v[54:61], a[12:15], v25, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 656) 



buffer_load_ubyte v25, v20, s[56:59], 0 offen offset:12 // (op 231) Load value

s_barrier  // (op 2419) 
v_add_u32 v22, 8, v22 // (op 1919) 

s_lshl_b64 s[14:15], s[24:25], 5 // (op 1721) 
v_mov_b64 v[76:77], s[14:15] // (op 1721) convert


v_mov_b64 v[78:79], v[74:75] // (op 1721) 
s_add_u32 s11, 20480, s7 // (op 1721) 
s_mov_b32 m0, s11 // (op 1721) reg expression
s_mov_b32 s14, 384 // (op 1721) call()
buffer_load_dwordx4 v78, s[52:55], s14 offen offset: 0 lds // (op 1721) Load value direct to lds
v_add_co_u32 v78, s[18:19], v78, v76 // (op 1721) least significant half
v_addc_co_u32 v79, s[18:19], 0, v79, s[18:19] // (op 1721) most significant half

s_add_u32 m0, m0, 4096 // (op 1721) 
buffer_load_dwordx4 v78, s[52:55], s14 offen offset: 0 lds // (op 1721) Load value direct to lds
v_add_co_u32 v78, s[18:19], v78, v76 // (op 1721) least significant half
v_addc_co_u32 v79, s[18:19], 0, v79, s[18:19] // (op 1721) most significant half

s_add_u32 m0, m0, 4096 // (op 1721) 
buffer_load_dwordx4 v78, s[52:55], s14 offen offset: 0 lds // (op 1721) Load value direct to lds
v_add_co_u32 v78, s[18:19], v78, v76 // (op 1721) least significant half
v_addc_co_u32 v79, s[18:19], 0, v79, s[18:19] // (op 1721) most significant half

s_add_u32 m0, m0, 4096 // (op 1721) 
buffer_load_dwordx4 v78, s[52:55], s14 offen offset: 0 lds // (op 1721) Load value direct to lds






v_add_co_u32 v14, vcc, 256, v14 // (op 1855) least significant half
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 1855) most significant half
v_mov_b64 v[74:75], s[32:33] // (op 238) convert

v_mov_b32 v20, v21 // (op 238) 
buffer_load_ubyte v26, v20, s[60:63], 0 offen offset:12 // (op 238) Load value
v_add_u32 v20, v20, v74 // (op 238) 
buffer_load_ubyte v27, v20, s[60:63], 0 offen offset:12 // (op 238) Load value



s_barrier  // (op 2413) 
v_add_u32 v23, 8, v23 // (op 1983) 


s_add_u32 s11, 36864, s6 // (op 1716) 

s_mov_b32 m0, s11 // (op 1716) reg expression
s_mov_b32 s14, 384 // (op 1716) call()
buffer_load_dwordx4 v72, s[48:51], s14 offen offset: 0 lds // (op 1716) Load value direct to lds



s_barrier  // (op 2396) 
v_add_co_u32 v18, vcc, 256, v18 // (op 1791) least significant half



v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 1791) most significant half
s_waitcnt vmcnt(10)
ds_write_b8 v70, v28 offset:40960 // (op 26) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v70, v29 offset:40964 // (op 26) Store local data 
s_barrier  // (op 2386) 



ds_read_b128 v[30:33], v5 // (op 70) Load local data 
ds_read_b128 v[34:37], v5 offset:64 // (op 70) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 2403) 



ds_read_u8 v12, v0 offset:40960 // (op 79) Load local data 



ds_read_b128 v[38:41], v5 offset:4096 // (op 339) Load local data 
ds_read_b128 v[42:45], v5 offset:4160 // (op 339) Load local data 



ds_read_u8 v13, v0 offset:41088 // (op 342) Load local data 



ds_read_b128 v[46:49], v5 offset:8192 // (op 495) Load local data 
ds_read_b128 v[50:53], v5 offset:8256 // (op 495) Load local data 



ds_read_u8 v16, v0 offset:41216 // (op 498) Load local data 



ds_read_b128 v[54:57], v5 offset:12288 // (op 651) Load local data 
ds_read_b128 v[58:61], v5 offset:12352 // (op 651) Load local data 



ds_read_u8 v17, v0 offset:41344 // (op 654) Load local data 
s_waitcnt vmcnt(8)
s_barrier  // (op 2370) 



ds_read_b128 v[62:65], v3 offset:16384 // (op 65) Load local data 
ds_read_b128 v[66:69], v3 offset:16448 // (op 65) Load local data 
s_ashr_i64 s[14:15], s[36:37], 7 // (op 619) 
s_mov_b32 s11, s14 // (op 619) convert

s_and_b32 s14, s11, 4294967294 // (op 619) 

s_cmp_lt_u32 s10, s14 // (op 619) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopTop_KLoop_619 // (op 619) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopTop_KLoop_619 if true)
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopBottom_KLoop_619:













s_barrier  // (op 2468) 
v_lshrrev_b32 v14, 3, v1 // (op 1751) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1751) 
v_lshl_add_u32 v15, s16, 5, v14 // (op 1751) 

s_lshr_b32 s6, s0, 1 // (op 1751) 

v_and_b32 v14, -8, v1 // (op 1751) 
v_sub_u32 v18, v1, v14 // (op 1751) 

v_lshlrev_b32 v14, 4, v18 // (op 1751) 

v_lshl_add_u32 v18, s6, 8, v14 // (op 1751) 


v_mov_b32 v29, 0 // (op 1751) convert
v_mov_b32 v28, v18 // (op 1751) convert

v_mul_lo_u32 v19, v15, s13 // (op 1751) most significant: low of low * high
v_mul_hi_u32 v14, v15, s12 // (op 1751) most significant: high of low * low
v_mul_lo_u32 v18, v15, s12 // (op 1751) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 1751) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v28 // (op 1751) least significant half
v_addc_co_u32 v19, s[0:1], v19, v29, s[0:1] // (op 1751) most significant half



s_mul_i32 s1, s12, s23 // (op 1753) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 1753) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 1753) least significant
s_mul_hi_u32 s7, s12, s22 // (op 1753) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1753) most significant: sum
s_add_u32 s1, s1, s7 // (op 1753) most significant: sum


s_mov_b32 s48, s20 // (op 1753) reg expression
s_mov_b32 s49, s21 // (op 1753) reg expression
s_mov_b32 s50, s0 // (op 1753) reg expression
s_mov_b32 s51, 131072 // (op 1753) call()

v_and_b32 v14, -8, v1 // (op 1821) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1821) 
v_sub_u32 v15, v1, v14 // (op 1821) 

s_lshr_b32 s6, s0, 1 // (op 1821) 

v_lshlrev_b32 v14, 4, v15 // (op 1821) 

v_lshl_add_u32 v15, s6, 8, v14 // (op 1821) 


v_mov_b32 v29, 0 // (op 1821) convert
v_mov_b32 v28, v15 // (op 1821) convert

v_lshrrev_b32 v14, 3, v1 // (op 1821) 
v_lshl_add_u32 v15, s17, 7, v14 // (op 1821) 

v_mul_lo_u32 v21, v15, s25 // (op 1821) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 1821) most significant: high of low * low
v_mul_lo_u32 v20, v15, s24 // (op 1821) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 1821) most significant: sum


v_add_co_u32 v14, s[0:1], v28, v20 // (op 1821) least significant half
v_addc_co_u32 v15, s[0:1], v29, v21, s[0:1] // (op 1821) most significant half



s_mul_i32 s1, s24, s9 // (op 1823) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 1823) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 1823) least significant
s_mul_hi_u32 s7, s24, s8 // (op 1823) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1823) most significant: sum
s_add_u32 s1, s1, s7 // (op 1823) most significant: sum


s_mov_b32 s52, s26 // (op 1823) reg expression
s_mov_b32 s53, s27 // (op 1823) reg expression
s_mov_b32 s54, s0 // (op 1823) reg expression
s_mov_b32 s55, 131072 // (op 1823) call()

v_lshrrev_b32 v20, 7, v1 // (op 1885) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1885) 
v_and_b32 v21, 15, v1 // (op 1885) 
s_lshr_b32 s6, s0, 1 // (op 1885) 

v_lshl_add_u32 v22, v20, 4, v21 // (op 1885) 


v_lshl_add_u32 v20, s16, 5, v22 // (op 1885) 

v_and_b32 v21, 63, v1 // (op 1885) 
v_lshrrev_b32 v22, 4, v21 // (op 1885) 

v_lshl_add_u32 v21, s6, 3, v22 // (op 1885) 


v_mul_lo_u32 v22, v20, s28 // (op 1885) 
v_add_u32 v22, v22, v21 // (op 1885) 


s_mul_i32 s1, s28, s23 // (op 1887) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 1887) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 1887) least significant
s_mul_hi_u32 s7, s28, s22 // (op 1887) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1887) most significant: sum
s_add_u32 s1, s1, s7 // (op 1887) most significant: sum


s_mov_b32 s56, s30 // (op 1887) reg expression
s_mov_b32 s57, s31 // (op 1887) reg expression
s_mov_b32 s58, s0 // (op 1887) reg expression
s_mov_b32 s59, 131072 // (op 1887) call()

v_and_b32 v20, -4, v1 // (op 1949) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1949) 
v_sub_u32 v21, v1, v20 // (op 1949) 

s_lshr_b32 s6, s0, 1 // (op 1949) 

v_lshrrev_b32 v20, 2, v1 // (op 1949) 
v_lshl_add_u32 v23, s6, 3, v21 // (op 1949) 


v_lshlrev_b32 v21, 1, v20 // (op 1949) 

v_lshl_add_u32 v20, s17, 7, v21 // (op 1949) 

v_mul_lo_u32 v21, v20, s32 // (op 1949) 

v_add_u32 v20, v23, v21 // (op 1949) 


s_mul_i32 s1, s32, s9 // (op 1951) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 1951) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 1951) least significant
s_mul_hi_u32 s7, s32, s8 // (op 1951) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1951) most significant: sum
s_add_u32 s1, s1, s7 // (op 1951) most significant: sum


s_mov_b32 s60, s34 // (op 1951) reg expression
s_mov_b32 s61, s35 // (op 1951) reg expression
s_mov_b32 s62, s0 // (op 1951) reg expression
s_mov_b32 s63, 131072 // (op 1951) call()

v_and_b32 v21, -8, v1 // (op 2042) 
v_sub_u32 v23, v1, v21 // (op 2042) 

v_lshlrev_b32 v21, 4, v23 // (op 2042) 

v_lshrrev_b32 v23, 3, v1 // (op 2042) 
v_lshl_add_u32 v24, v23, 7, v21 // (op 2042) 


s_nop 0
v_readfirstlane_b32 s0, v24 // (op 2042) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v21, -8, v1 // (op 2094) 
v_sub_u32 v23, v1, v21 // (op 2094) 

v_lshlrev_b32 v21, 4, v23 // (op 2094) 

v_lshrrev_b32 v23, 3, v1 // (op 2094) 
v_lshl_add_u32 v24, v23, 7, v21 // (op 2094) 


s_nop 0
v_readfirstlane_b32 s1, v24 // (op 2094) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v21, -4, v1 // (op 2151) 
v_sub_u32 v23, v1, v21 // (op 2151) 

v_lshrrev_b32 v21, 2, v1 // (op 2151) 
v_lshl_add_u32 v24, v21, 3, v23 // (op 2151) 


s_ashr_i64 s[6:7], s[36:37], 7 // (op 625) 
s_and_b32 s10, s6, 4294967294 // (op 625) 

s_ashr_i64 s[6:7], s[36:37], 7 // (op 624) 
s_mov_b32 s11, s6 // (op 624) convert

s_cmp_lt_u32 s10, s11 // (op 624) 












s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopBottom_KLoopTail_624 // (op 624) Condition: Top (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopBottom_KLoopTail_624 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopTop_KLoopTail_624:
s_barrier  // (op 2367) 
s_add_u32 s10, s10, 1 // (op 626) 


v_mov_b64 v[12:13], v[18:19] // (op 1703) 
s_add_u32 s11, 16384, s0 // (op 1703) 

s_mov_b32 m0, s11 // (op 1703) reg expression
buffer_load_dwordx4 v12, s[48:51], 0 offen offset: 0 lds // (op 1703) Load value direct to lds


v_add_co_u32 v18, vcc, 128, v18 // (op 1741) least significant half
v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 1741) most significant half
s_barrier  // (op 2383) 

s_lshl_b64 s[6:7], s[24:25], 5 // (op 1709) 
v_mov_b64 v[12:13], s[6:7] // (op 1709) convert


v_mov_b64 v[16:17], v[14:15] // (op 1709) 
v_mov_b64 v[26:27], v[16:17] // (op 1709) 
s_mov_b32 m0, s1 // (op 1709) reg expression
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 1709) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v12 // (op 1709) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 1709) most significant half

s_add_u32 m0, m0, 4096 // (op 1709) 
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 1709) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v12 // (op 1709) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 1709) most significant half

s_add_u32 m0, m0, 4096 // (op 1709) 
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 1709) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v12 // (op 1709) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 1709) most significant half

s_add_u32 m0, m0, 4096 // (op 1709) 
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 1709) Load value direct to lds



v_add_co_u32 v14, vcc, 128, v14 // (op 1813) least significant half
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 1813) most significant half

v_mov_b64 v[12:13], s[32:33] // (op 180) convert

v_mov_b32 v21, v20 // (op 180) 
v_mov_b32 v23, v21 // (op 180) 
buffer_load_ubyte v16, v23, s[60:63], 0 offen offset: 0 // (op 180) Load value
v_add_u32 v23, v23, v12 // (op 180) 
buffer_load_ubyte v17, v23, s[60:63], 0 offen offset: 0 // (op 180) Load value



v_add_u32 v20, 4, v20 // (op 1941) 
s_barrier  // (op 2399) 
s_waitcnt vmcnt(6)
s_barrier  // (op 2373) 






ds_read_b128 v[26:29], v3 offset:16384 // (op 169) Load local data 
ds_read_b128 v[30:33], v3 offset:16448 // (op 169) Load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v24, v16 offset:40960 // (op 178) Store local data 
s_barrier  // (op 2389) 
s_waitcnt vmcnt(0)
ds_write_b8 v24, v17 offset:40964 // (op 178) Store local data 




ds_read_b128 v[34:37], v5 // (op 171) Load local data 
ds_read_b128 v[38:41], v5 offset:64 // (op 171) Load local data 


v_mov_b32 v21, v22 // (op 173) 

buffer_load_ubyte v23, v21, s[56:59], 0 offen offset: 0 // (op 173) Load value

v_add_u32 v22, 4, v22 // (op 1877) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 2406) 



ds_read_u8 v21, v0 offset:40960 // (op 175) Load local data 



ds_read_b128 v[42:45], v5 offset:4096 // (op 320) Load local data 
ds_read_b128 v[46:49], v5 offset:4160 // (op 320) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:33], v[34:41], a[0:3], v23, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 176) 





ds_read_u8 v21, v0 offset:41088 // (op 323) Load local data 



ds_read_b128 v[34:37], v5 offset:8192 // (op 476) Load local data 
ds_read_b128 v[38:41], v5 offset:8256 // (op 476) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[26:33], v[42:49], a[4:7], v23, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 324) 





ds_read_u8 v21, v0 offset:41216 // (op 479) Load local data 



ds_read_b128 v[42:45], v5 offset:12288 // (op 632) Load local data 
ds_read_b128 v[46:49], v5 offset:12352 // (op 632) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[26:33], v[34:41], a[8:11], v23, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 480) 





ds_read_u8 v21, v0 offset:41344 // (op 635) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[26:33], v[42:49], a[12:15], v23, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 636) 




s_ashr_i64 s[6:7], s[36:37], 7 // (op 624) 
s_mov_b32 s11, s6 // (op 624) convert

s_cmp_lt_u32 s10, s11 // (op 624) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopTop_KLoopTail_624 // (op 624) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopTop_KLoopTail_624 if true)
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopBottom_KLoopTail_624:






















s_barrier  // (op 2472) 







v_cmp_eq_f32 vcc, 0.00000, v10 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ConditionalFalse_0_beta_123
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
s_branch RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ConditionalBottom_0_beta_123
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ConditionalFalse_0_beta_123:


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
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ConditionalBottom_0_beta_123:












s_endpgm  // End of RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2
.LRR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_end:
.size RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2, .LRR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_end-RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2
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
    DataType: F8
    DataTypeA: F8
    DataTypeB: F8
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
  - .name: RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2
    .symbol: RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2.kd
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
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: FP8, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP8, pointerType: PointerGlobal}
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

