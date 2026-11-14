// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2
.p2align 8
.type RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2,@function
RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 2053) 
v_and_b32 v3, 15, v1 // (op 2053) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 2053) 


v_and_b32 v0, 63, v1 // (op 2053) 
v_lshrrev_b32 v3, 4, v0 // (op 2053) 

v_lshlrev_b32 v0, 5, v3 // (op 2053) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 2053) 


v_mul_lo_u32 v0, v3, 6 // (op 2053) 

v_lshrrev_b32 v3, 3, v0 // (op 2053) 

v_lshrrev_b32 v0, 7, v1 // (op 2210) 
v_and_b32 v4, 15, v1 // (op 2210) 
v_lshl_add_u32 v5, v0, 4, v4 // (op 2210) 


v_and_b32 v0, 63, v1 // (op 2210) 
v_lshrrev_b32 v4, 4, v0 // (op 2210) 

v_lshlrev_b32 v0, 5, v4 // (op 2210) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 2210) 


v_mul_lo_u32 v0, v4, 6 // (op 2210) 

v_lshrrev_b32 v4, 3, v0 // (op 2210) 

v_lshrrev_b32 v0, 7, v1 // (op 1996) 
v_and_b32 v5, 63, v1 // (op 1996) 
v_lshrrev_b32 v6, 4, v5 // (op 1996) 
v_lshlrev_b32 v7, 2, v6 // (op 1996) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 1996) 


v_lshl_add_u32 v0, s16, 5, v6 // (op 1996) 

v_lshrrev_b32 v6, 2, v5 // (op 1996) 

v_and_b32 v5, 3, v6 // (op 1996) 

v_and_b32 v6, 3, v1 // (op 1996) 
v_lshl_add_u32 v7, v5, 2, v6 // (op 1996) 


v_lshrrev_b32 v5, 6, v1 // (op 1996) 
v_and_b32 v6, 1, v5 // (op 1996) 

v_lshl_add_u32 v5, v6, 4, v7 // (op 1996) 


v_lshl_add_u32 v6, s17, 7, v5 // (op 1996) 

v_mul_lo_u32 v5, v6, s2 // (op 1996) 

v_add_lshl_u32 v6, v0, v5, 1 // (op 1996) 


v_lshrrev_b32 v0, 6, v1 // (op 2110) 
v_and_b32 v5, 1, v0 // (op 2110) 

v_and_b32 v0, 15, v1 // (op 2110) 
v_lshl_add_u32 v7, v5, 4, v0 // (op 2110) 


v_and_b32 v0, 48, v1 // (op 2110) 
v_lshl_add_u32 v5, v7, 7, v0 // (op 2110) 


v_lshrrev_b32 v0, 6, v1 // (op 2168) 
v_and_b32 v7, 1, v0 // (op 2168) 

v_and_b32 v0, 15, v1 // (op 2168) 
s_mul_i32 s1, s2, s9 // (op 1998) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 1998) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 1998) least significant
s_mul_hi_u32 s19, s2, s8 // (op 1998) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1998) most significant: sum
s_add_u32 s1, s1, s19 // (op 1998) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 1998) 

v_lshl_add_u32 v8, v7, 4, v0 // (op 2168) 


v_and_b32 v0, 63, v1 // (op 2168) 
v_lshrrev_b32 v7, 4, v0 // (op 2168) 

v_lshl_add_u32 v0, v8, 2, v7 // (op 2168) 


v_lshrrev_b32 v7, 7, v1 // (op 2184) 
v_and_b32 v8, 63, v1 // (op 2184) 
v_lshrrev_b32 v9, 4, v8 // (op 2184) 
v_lshlrev_b32 v10, 2, v9 // (op 2184) 

v_lshl_add_u32 v9, v7, 4, v10 // (op 2184) 


v_lshl_add_u32 v7, s16, 5, v9 // (op 2184) 

v_lshrrev_b32 v9, 2, v8 // (op 2184) 

v_and_b32 v8, 3, v9 // (op 2184) 

v_and_b32 v9, 3, v1 // (op 2184) 
v_lshl_add_u32 v10, v8, 2, v9 // (op 2184) 


v_lshrrev_b32 v8, 6, v1 // (op 2184) 
v_and_b32 v9, 1, v8 // (op 2184) 

v_lshl_add_u32 v8, v9, 4, v10 // (op 2184) 


v_lshl_add_u32 v9, s17, 7, v8 // (op 2184) 

v_mul_lo_u32 v8, v9, s4 // (op 2184) 

v_add_lshl_u32 v9, v7, v8, 1 // (op 2184) 


v_lshrrev_b32 v7, 6, v1 // (op 2254) 
v_and_b32 v8, 1, v7 // (op 2254) 

v_and_b32 v7, 15, v1 // (op 2254) 
v_lshl_add_u32 v10, v8, 4, v7 // (op 2254) 


v_and_b32 v7, 48, v1 // (op 2254) 
v_lshl_add_u32 v8, v10, 7, v7 // (op 2254) 


v_lshrrev_b32 v7, 6, v1 // (op 2283) 
v_and_b32 v10, 1, v7 // (op 2283) 

v_and_b32 v7, 15, v1 // (op 2283) 
s_mul_i32 s1, s4, s9 // (op 2186) most significant: low of low * high
v_lshl_add_u32 v11, v10, 4, v7 // (op 2283) 


v_and_b32 v7, 63, v1 // (op 2283) 
v_lshrrev_b32 v10, 4, v7 // (op 2283) 

v_lshl_add_u32 v7, v11, 2, v10 // (op 2283) 


s_mul_i32 s38, s5, s8 // (op 2186) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 2186) least significant
s_mul_hi_u32 s39, s4, s8 // (op 2186) most significant: high of low * low
s_add_u32 s1, s1, s38 // (op 2186) most significant: sum
s_add_u32 s1, s1, s39 // (op 2186) most significant: sum


s_lshl_b64 s[38:39], s[0:1], 1 // (op 2186) 

s_mov_b32 s40, s6 // (op 1998) reg expression
s_mov_b32 s41, s7 // (op 1998) reg expression
s_mov_b32 s42, s18 // (op 1998) reg expression
s_mov_b32 s43, 131072 // (op 1998) call()

s_mov_b32 s44, s10 // (op 2186) reg expression

s_mov_b32 s45, s11 // (op 2186) reg expression
s_mov_b32 s46, s38 // (op 2186) reg expression
s_mov_b32 s47, 131072 // (op 2186) call()


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
v_lshrrev_b32 v12, 3, v1 // (op 1763) 
v_lshl_add_u32 v13, s16, 5, v12 // (op 1763) 

v_and_b32 v12, -8, v1 // (op 1763) 
v_sub_u32 v14, v1, v12 // (op 1763) 

v_lshlrev_b32 v12, 4, v14 // (op 1763) 

v_mov_b32 v15, 0 // (op 1763) convert
v_mov_b32 v14, v12 // (op 1763) convert

v_mul_lo_u32 v17, v13, s13 // (op 1763) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 1763) most significant: high of low * low
v_mul_lo_u32 v16, v13, s12 // (op 1763) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 1763) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 1763) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 1763) most significant half



v_mul_lo_u32 v13, v17, 6 // (op 1763) most significant: low of high * low
v_mul_hi_u32 v14, v16, 6 // (op 1763) most significant: high of low * low
v_mul_lo_u32 v12, v16, 6 // (op 1763) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 1763) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 1763) 

v_mov_b32 v12, v14 // (op 1763) convert


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s23, 6 // (op 1765) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 1765) least significant
s_mul_hi_u32 s6, s22, 6 // (op 1765) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1765) most significant: sum

s_mul_i32 s7, s12, s1 // (op 1765) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 1765) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 1765) least significant
s_mul_hi_u32 s11, s12, s0 // (op 1765) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 1765) most significant: sum
s_add_u32 s7, s7, s11 // (op 1765) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 1765) 

s_mov_b32 s48, s20 // (op 1765) reg expression
s_mov_b32 s49, s21 // (op 1765) reg expression
s_mov_b32 s50, s0 // (op 1765) reg expression
s_mov_b32 s51, 131072 // (op 1765) call()

v_lshrrev_b32 v13, 3, v1 // (op 1782) 
v_lshl_add_u32 v14, s16, 5, v13 // (op 1782) 

v_and_b32 v13, -8, v1 // (op 1782) 
v_sub_u32 v15, v1, v13 // (op 1782) 

v_lshlrev_b32 v13, 4, v15 // (op 1782) 

v_mov_b32 v17, 0 // (op 1782) convert
v_mov_b32 v16, v13 // (op 1782) convert

v_mul_lo_u32 v19, v14, s13 // (op 1782) most significant: low of low * high
v_mul_hi_u32 v13, v14, s12 // (op 1782) most significant: high of low * low
v_mul_lo_u32 v18, v14, s12 // (op 1782) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 1782) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 1782) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 1782) most significant half



v_mul_lo_u32 v17, v19, 6 // (op 1782) most significant: low of high * low
v_mul_hi_u32 v13, v18, 6 // (op 1782) most significant: high of low * low
v_mul_lo_u32 v16, v18, 6 // (op 1782) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 1782) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 1782) 

v_mov_b32 v13, v14 // (op 1782) convert

s_mul_i32 s1, s23, 6 // (op 1784) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 1784) least significant
s_mul_hi_u32 s6, s22, 6 // (op 1784) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1784) most significant: sum

s_mul_i32 s7, s12, s1 // (op 1784) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 1784) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 1784) least significant
s_mul_hi_u32 s11, s12, s0 // (op 1784) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 1784) most significant: sum
s_add_u32 s7, s7, s11 // (op 1784) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 1784) 

s_mov_b32 s48, s20 // (op 1784) reg expression
s_mov_b32 s49, s21 // (op 1784) reg expression
s_mov_b32 s50, s0 // (op 1784) reg expression
s_mov_b32 s51, 131072 // (op 1784) call()

v_and_b32 v14, -8, v1 // (op 1827) 
v_sub_u32 v15, v1, v14 // (op 1827) 

v_lshlrev_b32 v14, 4, v15 // (op 1827) 

v_mov_b32 v17, 0 // (op 1827) convert
v_mov_b32 v16, v14 // (op 1827) convert

v_lshrrev_b32 v14, 3, v1 // (op 1827) 
v_lshl_add_u32 v15, s17, 7, v14 // (op 1827) 

v_mul_lo_u32 v19, v15, s25 // (op 1827) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 1827) most significant: high of low * low
v_mul_lo_u32 v18, v15, s24 // (op 1827) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 1827) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v18 // (op 1827) least significant half
v_addc_co_u32 v15, s[0:1], v17, v19, s[0:1] // (op 1827) most significant half



s_mul_i32 s1, s24, s9 // (op 1829) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 1829) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 1829) least significant
s_mul_hi_u32 s7, s24, s8 // (op 1829) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1829) most significant: sum
s_add_u32 s1, s1, s7 // (op 1829) most significant: sum


s_mov_b32 s52, s26 // (op 1829) reg expression
s_mov_b32 s53, s27 // (op 1829) reg expression
s_mov_b32 s54, s0 // (op 1829) reg expression
s_mov_b32 s55, 131072 // (op 1829) call()

v_and_b32 v16, -8, v1 // (op 1846) 
v_sub_u32 v17, v1, v16 // (op 1846) 

v_lshlrev_b32 v16, 4, v17 // (op 1846) 

v_mov_b32 v19, 0 // (op 1846) convert
v_mov_b32 v18, v16 // (op 1846) convert

v_lshrrev_b32 v16, 3, v1 // (op 1846) 
v_lshl_add_u32 v17, s17, 7, v16 // (op 1846) 

v_mul_lo_u32 v21, v17, s25 // (op 1846) most significant: low of low * high
v_mul_hi_u32 v16, v17, s24 // (op 1846) most significant: high of low * low
v_mul_lo_u32 v20, v17, s24 // (op 1846) least significant: low of low * low
v_add_u32 v21, v21, v16 // (op 1846) most significant: sum


v_add_co_u32 v16, s[0:1], v18, v20 // (op 1846) least significant half
v_addc_co_u32 v17, s[0:1], v19, v21, s[0:1] // (op 1846) most significant half



s_mul_i32 s1, s24, s9 // (op 1848) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 1848) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 1848) least significant
s_mul_hi_u32 s7, s24, s8 // (op 1848) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1848) most significant: sum
s_add_u32 s1, s1, s7 // (op 1848) most significant: sum


s_mov_b32 s52, s26 // (op 1848) reg expression
s_mov_b32 s53, s27 // (op 1848) reg expression
s_mov_b32 s54, s0 // (op 1848) reg expression
s_mov_b32 s55, 131072 // (op 1848) call()

v_lshrrev_b32 v18, 7, v1 // (op 1891) 
v_and_b32 v19, 15, v1 // (op 1891) 
v_lshl_add_u32 v20, v18, 4, v19 // (op 1891) 


v_lshl_add_u32 v18, s16, 5, v20 // (op 1891) 

v_and_b32 v19, 63, v1 // (op 1891) 
v_lshrrev_b32 v20, 4, v19 // (op 1891) 

v_mul_lo_u32 v19, v18, s28 // (op 1891) 
v_add_u32 v19, v19, v20 // (op 1891) 


s_mul_i32 s1, s28, s23 // (op 1893) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 1893) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 1893) least significant
s_mul_hi_u32 s7, s28, s22 // (op 1893) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1893) most significant: sum
s_add_u32 s1, s1, s7 // (op 1893) most significant: sum


s_mov_b32 s56, s30 // (op 1893) reg expression
s_mov_b32 s57, s31 // (op 1893) reg expression
s_mov_b32 s58, s0 // (op 1893) reg expression
s_mov_b32 s59, 131072 // (op 1893) call()

v_lshrrev_b32 v18, 7, v1 // (op 1910) 
v_and_b32 v20, 15, v1 // (op 1910) 
v_lshl_add_u32 v21, v18, 4, v20 // (op 1910) 


v_lshl_add_u32 v18, s16, 5, v21 // (op 1910) 

v_and_b32 v20, 63, v1 // (op 1910) 
v_lshrrev_b32 v21, 4, v20 // (op 1910) 

v_mul_lo_u32 v20, v18, s28 // (op 1910) 
v_add_u32 v20, v20, v21 // (op 1910) 


s_mul_i32 s1, s28, s23 // (op 1912) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 1912) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 1912) least significant
s_mul_hi_u32 s7, s28, s22 // (op 1912) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1912) most significant: sum
s_add_u32 s1, s1, s7 // (op 1912) most significant: sum


s_mov_b32 s56, s30 // (op 1912) reg expression
s_mov_b32 s57, s31 // (op 1912) reg expression
s_mov_b32 s58, s0 // (op 1912) reg expression
s_mov_b32 s59, 131072 // (op 1912) call()

v_and_b32 v18, -4, v1 // (op 1955) 
v_sub_u32 v21, v1, v18 // (op 1955) 

v_lshrrev_b32 v18, 2, v1 // (op 1955) 
v_lshlrev_b32 v22, 1, v18 // (op 1955) 

v_lshl_add_u32 v18, s17, 7, v22 // (op 1955) 

v_mul_lo_u32 v22, v18, s32 // (op 1955) 

v_add_u32 v18, v21, v22 // (op 1955) 


s_mul_i32 s1, s32, s9 // (op 1957) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 1957) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 1957) least significant
s_mul_hi_u32 s7, s32, s8 // (op 1957) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1957) most significant: sum
s_add_u32 s1, s1, s7 // (op 1957) most significant: sum


s_mov_b32 s60, s34 // (op 1957) reg expression
s_mov_b32 s61, s35 // (op 1957) reg expression
s_mov_b32 s62, s0 // (op 1957) reg expression
s_mov_b32 s63, 131072 // (op 1957) call()

v_and_b32 v21, -4, v1 // (op 1974) 
v_sub_u32 v22, v1, v21 // (op 1974) 

v_lshrrev_b32 v21, 2, v1 // (op 1974) 
v_lshlrev_b32 v23, 1, v21 // (op 1974) 

v_lshl_add_u32 v21, s17, 7, v23 // (op 1974) 

v_mul_lo_u32 v23, v21, s32 // (op 1974) 

v_add_u32 v21, v22, v23 // (op 1974) 


s_mul_i32 s1, s32, s9 // (op 1976) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 1976) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 1976) least significant
s_mul_hi_u32 s7, s32, s8 // (op 1976) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1976) most significant: sum
s_add_u32 s1, s1, s7 // (op 1976) most significant: sum


s_mov_b32 s60, s34 // (op 1976) reg expression
s_mov_b32 s61, s35 // (op 1976) reg expression
s_mov_b32 s62, s0 // (op 1976) reg expression
s_mov_b32 s63, 131072 // (op 1976) call()




buffer_load_ubyte v22, v19, s[56:59], 0 offen offset: 0 // (op 1560) Load value
v_and_b32 v23, -8, v1 // (op 2093) 
v_sub_u32 v24, v1, v23 // (op 2093) 

v_lshlrev_b32 v23, 4, v24 // (op 2093) 

v_lshrrev_b32 v24, 3, v1 // (op 2093) 
v_lshl_add_u32 v25, v24, 7, v23 // (op 2093) 


s_nop 0
v_readfirstlane_b32 s0, v25 // (op 2093) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 1716) 
v_mov_b64 v[24:25], s[6:7] // (op 1716) convert


v_mov_b64 v[26:27], v[14:15] // (op 1716) 
s_mov_b32 m0, s0 // (op 1716) reg expression
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 1716) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v24 // (op 1716) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 1716) most significant half

s_add_u32 m0, m0, 4096 // (op 1716) 
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 1716) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v24 // (op 1716) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 1716) most significant half

s_add_u32 m0, m0, 4096 // (op 1716) 
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 1716) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v24 // (op 1716) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 1716) most significant half

s_add_u32 m0, m0, 4096 // (op 1716) 
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 1716) Load value direct to lds




v_mov_b64 v[24:25], s[32:33] // (op 1567) convert

v_mov_b32 v23, v18 // (op 1567) 
buffer_load_ubyte v26, v23, s[60:63], 0 offen offset: 0 // (op 1567) Load value
v_add_u32 v23, v23, v24 // (op 1567) 
buffer_load_ubyte v27, v23, s[60:63], 0 offen offset: 0 // (op 1567) Load value





buffer_load_dwordx3 v[28:30], v12, s[48:51], 0 offen // (op 1570) Load value



buffer_load_ubyte v23, v19, s[56:59], 0 offen offset:4 // (op 1574) Load value

v_and_b32 v19, -8, v1 // (op 2237) 
v_sub_u32 v24, v1, v19 // (op 2237) 

v_lshlrev_b32 v19, 4, v24 // (op 2237) 

v_lshrrev_b32 v24, 3, v1 // (op 2237) 
v_lshl_add_u32 v25, v24, 7, v19 // (op 2237) 


s_nop 0
v_readfirstlane_b32 s0, v25 // (op 2237) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 1721) 
v_mov_b64 v[24:25], s[6:7] // (op 1721) convert


v_mov_b64 v[32:33], v[14:15] // (op 1721) 
s_add_u32 s1, 16384, s0 // (op 1721) 
s_mov_b32 m0, s1 // (op 1721) reg expression
s_mov_b32 s6, 128 // (op 1721) call()
buffer_load_dwordx4 v32, s[52:55], s6 offen offset: 0 lds // (op 1721) Load value direct to lds
v_add_co_u32 v32, s[10:11], v32, v24 // (op 1721) least significant half
v_addc_co_u32 v33, s[10:11], 0, v33, s[10:11] // (op 1721) most significant half

s_add_u32 m0, m0, 4096 // (op 1721) 
buffer_load_dwordx4 v32, s[52:55], s6 offen offset: 0 lds // (op 1721) Load value direct to lds
v_add_co_u32 v32, s[10:11], v32, v24 // (op 1721) least significant half
v_addc_co_u32 v33, s[10:11], 0, v33, s[10:11] // (op 1721) most significant half

s_add_u32 m0, m0, 4096 // (op 1721) 
buffer_load_dwordx4 v32, s[52:55], s6 offen offset: 0 lds // (op 1721) Load value direct to lds
v_add_co_u32 v32, s[10:11], v32, v24 // (op 1721) least significant half
v_addc_co_u32 v33, s[10:11], 0, v33, s[10:11] // (op 1721) most significant half

s_add_u32 m0, m0, 4096 // (op 1721) 
buffer_load_dwordx4 v32, s[52:55], s6 offen offset: 0 lds // (op 1721) Load value direct to lds







v_mov_b64 v[14:15], s[32:33] // (op 1581) convert

v_mov_b32 v19, v18 // (op 1581) 
buffer_load_ubyte v24, v19, s[60:63], 0 offen offset:4 // (op 1581) Load value
v_add_u32 v19, v19, v14 // (op 1581) 
buffer_load_ubyte v25, v19, s[60:63], 0 offen offset:4 // (op 1581) Load value






buffer_load_dwordx3 v[32:34], v12, s[48:51], 0 offen offset:96 // (op 1584) Load value

v_and_b32 v12, -4, v1 // (op 2151) 
v_sub_u32 v31, v1, v12 // (op 2151) 

v_lshrrev_b32 v12, 2, v1 // (op 2151) 
v_lshl_add_u32 v14, v12, 3, v31 // (op 2151) 





s_waitcnt vmcnt(10)
ds_write_b8 v14, v26 offset:32768 // (op 1587) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v14, v27 offset:32772 // (op 1587) Store local data 

v_and_b32 v12, -8, v1 // (op 2041) 
v_sub_u32 v31, v1, v12 // (op 2041) 

v_lshlrev_b32 v12, 4, v31 // (op 2041) 

v_lshrrev_b32 v31, 3, v1 // (op 2041) 
v_lshl_add_u32 v14, v31, 7, v12 // (op 2041) 


v_mul_lo_u32 v12, v14, 6 // (op 2041) 

v_lshrrev_b32 v31, 3, v12 // (op 2041) 




s_waitcnt vmcnt(8)
ds_write_b96 v31, v[28:30] offset:33280 // (op 1588) Store local data 

v_and_b32 v12, -8, v1 // (op 2012) 
v_sub_u32 v31, v1, v12 // (op 2012) 

v_lshlrev_b32 v12, 4, v31 // (op 2012) 

v_lshrrev_b32 v31, 3, v1 // (op 2012) 
v_lshl_add_u32 v14, v31, 7, v12 // (op 2012) 


v_mul_lo_u32 v12, v14, 6 // (op 2012) 

v_lshrrev_b32 v31, 3, v12 // (op 2012) 

v_and_b32 v12, -8, v1 // (op 2065) 
v_sub_u32 v14, v1, v12 // (op 2065) 

v_lshlrev_b32 v12, 4, v14 // (op 2065) 

v_lshrrev_b32 v14, 3, v1 // (op 2065) 
v_lshl_add_u32 v15, v14, 7, v12 // (op 2065) 


s_nop 0
v_readfirstlane_b32 s0, v15 // (op 2065) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -4, v1 // (op 2122) 
v_sub_u32 v14, v1, v12 // (op 2122) 

v_lshrrev_b32 v12, 2, v1 // (op 2122) 
v_lshl_add_u32 v15, v12, 3, v14 // (op 2122) 


v_and_b32 v12, -8, v1 // (op 2198) 
v_sub_u32 v14, v1, v12 // (op 2198) 

v_lshlrev_b32 v12, 4, v14 // (op 2198) 

v_lshrrev_b32 v14, 3, v1 // (op 2198) 
v_lshl_add_u32 v18, v14, 7, v12 // (op 2198) 


v_mul_lo_u32 v12, v18, 6 // (op 2198) 

v_lshrrev_b32 v14, 3, v12 // (op 2198) 

v_and_b32 v12, -8, v1 // (op 2222) 
v_sub_u32 v18, v1, v12 // (op 2222) 

v_lshlrev_b32 v12, 4, v18 // (op 2222) 

v_lshrrev_b32 v18, 3, v1 // (op 2222) 
v_lshl_add_u32 v19, v18, 7, v12 // (op 2222) 


s_nop 0
v_readfirstlane_b32 s1, v19 // (op 2222) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -4, v1 // (op 2266) 
v_sub_u32 v18, v1, v12 // (op 2266) 

v_lshrrev_b32 v12, 2, v1 // (op 2266) 
v_lshl_add_u32 v19, v12, 3, v18 // (op 2266) 


s_mov_b32 s6, 0 // (op 624) call()
s_ashr_i64 s[10:11], s[36:37], 7 // (op 623) 
s_mov_b32 s7, s10 // (op 623) convert

s_and_b32 s10, s7, 4294967294 // (op 623) 

s_cmp_lt_u32 s6, s10 // (op 623) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopBottom_KLoop_623 // (op 623) Condition: Top (jump to RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopBottom_KLoop_623 if false)
RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopTop_KLoop_623:
s_add_u32 s6, s6, 2 // (op 625) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 2330) 



ds_read_b128 v[36:39], v3 offset:33280 // (op 67) Load local data 
ds_read_b64 v[40:41], v3 offset:33296 // (op 67) Load local data 
s_barrier  // (op 2343) 



ds_read_b128 v[42:45], v5 // (op 72) Load local data 
ds_read_b128 v[46:49], v5 offset:64 // (op 72) Load local data 
s_barrier  // (op 2356) 



ds_read_u8 v12, v0 offset:32768 // (op 81) Load local data 



ds_read_b128 v[50:53], v5 offset:4096 // (op 343) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:41], v[42:49], a[0:3], v22, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 84) 
ds_read_b128 v[54:57], v5 offset:4160 // (op 343) Load local data 





ds_read_u8 v12, v0 offset:32896 // (op 346) Load local data 



ds_read_b128 v[42:45], v5 offset:8192 // (op 499) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[36:41], v[50:57], a[4:7], v22, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 347) 
ds_read_b128 v[46:49], v5 offset:8256 // (op 499) Load local data 





ds_read_u8 v12, v0 offset:33024 // (op 502) Load local data 



ds_read_b128 v[50:53], v5 offset:12288 // (op 655) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[36:41], v[42:49], a[8:11], v22, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 503) 
ds_read_b128 v[54:57], v5 offset:12352 // (op 655) Load local data 





ds_read_u8 v12, v0 offset:33152 // (op 658) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[36:41], v[50:57], a[12:15], v22, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 659) 





v_mov_b32 v12, v20 // (op 77) 

buffer_load_ubyte v22, v12, s[56:59], 0 offen offset:8 // (op 77) Load value
s_barrier  // (op 2346) 

s_lshl_b64 s[10:11], s[24:25], 5 // (op 1699) 
v_mov_b64 v[36:37], s[10:11] // (op 1699) convert


v_mov_b64 v[38:39], v[16:17] // (op 1699) 
v_mov_b64 v[40:41], v[38:39] // (op 1699) 
s_mov_b32 m0, s0 // (op 1699) reg expression
s_mov_b32 s7, 256 // (op 1699) call()
buffer_load_dwordx4 v40, s[52:55], s7 offen offset: 0 lds // (op 1699) Load value direct to lds
v_add_co_u32 v40, s[10:11], v40, v36 // (op 1699) least significant half
v_addc_co_u32 v41, s[10:11], 0, v41, s[10:11] // (op 1699) most significant half

s_add_u32 m0, m0, 4096 // (op 1699) 
buffer_load_dwordx4 v40, s[52:55], s7 offen offset: 0 lds // (op 1699) Load value direct to lds
v_add_co_u32 v40, s[10:11], v40, v36 // (op 1699) least significant half
v_addc_co_u32 v41, s[10:11], 0, v41, s[10:11] // (op 1699) most significant half

s_add_u32 m0, m0, 4096 // (op 1699) 
buffer_load_dwordx4 v40, s[52:55], s7 offen offset: 0 lds // (op 1699) Load value direct to lds
v_add_co_u32 v40, s[10:11], v40, v36 // (op 1699) least significant half
v_addc_co_u32 v41, s[10:11], 0, v41, s[10:11] // (op 1699) most significant half

s_add_u32 m0, m0, 4096 // (op 1699) 
buffer_load_dwordx4 v40, s[52:55], s7 offen offset: 0 lds // (op 1699) Load value direct to lds




v_mov_b64 v[36:37], s[32:33] // (op 21) convert

v_mov_b32 v18, v21 // (op 21) 
v_mov_b32 v35, v18 // (op 21) 
buffer_load_ubyte v26, v35, s[60:63], 0 offen offset:8 // (op 21) Load value
v_add_u32 v35, v35, v36 // (op 21) 
buffer_load_ubyte v27, v35, s[60:63], 0 offen offset:8 // (op 21) Load value




v_mov_b32 v35, v13 // (op 4) 

buffer_load_dwordx3 v[28:30], v35, s[48:51], 0 offen offset:192 // (op 4) Load value
s_barrier  // (op 2385) 



s_waitcnt vmcnt(10)
ds_write_b8 v19, v24 offset:36352 // (op 240) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v19, v25 offset:36356 // (op 240) Store local data 
s_barrier  // (op 2373) 



s_waitcnt vmcnt(8)
ds_write_b96 v14, v[32:34] offset:36864 // (op 229) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 2370) 



ds_read_b128 v[40:43], v4 offset:36864 // (op 231) Load local data 
ds_read_b64 v[44:45], v4 offset:36880 // (op 231) Load local data 
s_barrier  // (op 2376) 
s_barrier  // (op 2420) 



ds_read_b128 v[46:49], v8 offset:16384 // (op 233) Load local data 
ds_read_b128 v[50:53], v8 offset:16448 // (op 233) Load local data 
s_barrier  // (op 2382) 



ds_read_u8 v36, v7 offset:36352 // (op 237) Load local data 



ds_read_b128 v[54:57], v8 offset:20480 // (op 352) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[40:45], v[46:53], a[0:3], v23, v36 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 238) 
ds_read_b128 v[58:61], v8 offset:20544 // (op 352) Load local data 





ds_read_u8 v36, v7 offset:36480 // (op 355) Load local data 



ds_read_b128 v[46:49], v8 offset:24576 // (op 508) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[40:45], v[54:61], a[4:7], v23, v36 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 348) 
ds_read_b128 v[50:53], v8 offset:24640 // (op 508) Load local data 





ds_read_u8 v36, v7 offset:36608 // (op 511) Load local data 



ds_read_b128 v[54:57], v8 offset:28672 // (op 664) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[40:45], v[46:53], a[8:11], v23, v36 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 504) 
ds_read_b128 v[58:61], v8 offset:28736 // (op 664) Load local data 





ds_read_u8 v36, v7 offset:36736 // (op 667) Load local data 
s_barrier  // (op 2468) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[40:45], v[54:61], a[12:15], v23, v36 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 660) 






buffer_load_ubyte v23, v12, s[56:59], 0 offen offset:12 // (op 235) Load value

s_barrier  // (op 2379) 
v_add_u32 v20, 8, v20 // (op 1902) 

s_lshl_b64 s[10:11], s[24:25], 5 // (op 1711) 
v_mov_b64 v[36:37], s[10:11] // (op 1711) convert


v_mov_b64 v[40:41], v[38:39] // (op 1711) 
s_add_u32 s7, 16384, s1 // (op 1711) 
s_mov_b32 m0, s7 // (op 1711) reg expression
s_mov_b32 s10, 384 // (op 1711) call()
buffer_load_dwordx4 v40, s[52:55], s10 offen offset: 0 lds // (op 1711) Load value direct to lds
v_add_co_u32 v40, s[14:15], v40, v36 // (op 1711) least significant half
v_addc_co_u32 v41, s[14:15], 0, v41, s[14:15] // (op 1711) most significant half

s_add_u32 m0, m0, 4096 // (op 1711) 
buffer_load_dwordx4 v40, s[52:55], s10 offen offset: 0 lds // (op 1711) Load value direct to lds
v_add_co_u32 v40, s[14:15], v40, v36 // (op 1711) least significant half
v_addc_co_u32 v41, s[14:15], 0, v41, s[14:15] // (op 1711) most significant half

s_add_u32 m0, m0, 4096 // (op 1711) 
buffer_load_dwordx4 v40, s[52:55], s10 offen offset: 0 lds // (op 1711) Load value direct to lds
v_add_co_u32 v40, s[14:15], v40, v36 // (op 1711) least significant half
v_addc_co_u32 v41, s[14:15], 0, v41, s[14:15] // (op 1711) most significant half

s_add_u32 m0, m0, 4096 // (op 1711) 
buffer_load_dwordx4 v40, s[52:55], s10 offen offset: 0 lds // (op 1711) Load value direct to lds






v_add_co_u32 v16, vcc, 256, v16 // (op 1838) least significant half
v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 1838) most significant half
v_mov_b64 v[36:37], s[32:33] // (op 242) convert

v_mov_b32 v12, v18 // (op 242) 
buffer_load_ubyte v24, v12, s[60:63], 0 offen offset:12 // (op 242) Load value
v_add_u32 v12, v12, v36 // (op 242) 
buffer_load_ubyte v25, v12, s[60:63], 0 offen offset:12 // (op 242) Load value






v_add_u32 v21, 8, v21 // (op 1966) 
buffer_load_dwordx3 v[32:34], v35, s[48:51], 0 offen offset:288 // (op 228) Load value

s_barrier  // (op 2359) 
v_add_u32 v13, 192, v13 // (op 1774) 



s_waitcnt vmcnt(10)
ds_write_b8 v15, v26 offset:32768 // (op 26) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v15, v27 offset:32772 // (op 26) Store local data 
s_barrier  // (op 2333) 



s_waitcnt vmcnt(8)
ds_write_b96 v31, v[28:30] offset:33280 // (op 6) Store local data 
s_ashr_i64 s[10:11], s[36:37], 7 // (op 623) 
s_mov_b32 s7, s10 // (op 623) convert

s_and_b32 s10, s7, 4294967294 // (op 623) 

s_cmp_lt_u32 s6, s10 // (op 623) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopTop_KLoop_623 // (op 623) Condition: Bottom (jump to RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopTop_KLoop_623 if true)
RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopBottom_KLoop_623:














s_barrier  // (op 2450) 
v_lshrrev_b32 v12, 3, v1 // (op 1734) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1734) 
v_lshl_add_u32 v13, s16, 5, v12 // (op 1734) 

s_lshr_b32 s6, s0, 1 // (op 1734) 

v_and_b32 v12, -8, v1 // (op 1734) 
v_sub_u32 v14, v1, v12 // (op 1734) 

v_lshlrev_b32 v12, 4, v14 // (op 1734) 

v_lshl_add_u32 v14, s6, 8, v12 // (op 1734) 


v_mov_b32 v27, 0 // (op 1734) convert
v_mov_b32 v26, v14 // (op 1734) convert

v_mul_lo_u32 v15, v13, s13 // (op 1734) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 1734) most significant: high of low * low
v_mul_lo_u32 v14, v13, s12 // (op 1734) least significant: low of low * low
v_add_u32 v15, v15, v12 // (op 1734) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v26 // (op 1734) least significant half
v_addc_co_u32 v15, s[0:1], v15, v27, s[0:1] // (op 1734) most significant half



v_mul_lo_u32 v13, v15, 6 // (op 1734) most significant: low of high * low
v_mul_hi_u32 v16, v14, 6 // (op 1734) most significant: high of low * low
v_mul_lo_u32 v12, v14, 6 // (op 1734) least significant: low of low * low
v_add_u32 v13, v13, v16 // (op 1734) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 1734) 

v_mov_b32 v12, v14 // (op 1734) convert

s_mul_i32 s1, s23, 6 // (op 1736) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 1736) least significant
s_mul_hi_u32 s6, s22, 6 // (op 1736) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1736) most significant: sum

s_mul_i32 s7, s12, s1 // (op 1736) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 1736) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 1736) least significant
s_mul_hi_u32 s11, s12, s0 // (op 1736) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 1736) most significant: sum
s_add_u32 s7, s7, s11 // (op 1736) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 1736) 

s_mov_b32 s48, s20 // (op 1736) reg expression
s_mov_b32 s49, s21 // (op 1736) reg expression
s_mov_b32 s50, s0 // (op 1736) reg expression
s_mov_b32 s51, 131072 // (op 1736) call()

v_and_b32 v13, -8, v1 // (op 1804) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1804) 
v_sub_u32 v14, v1, v13 // (op 1804) 

s_lshr_b32 s6, s0, 1 // (op 1804) 

v_lshlrev_b32 v13, 4, v14 // (op 1804) 

v_lshl_add_u32 v14, s6, 8, v13 // (op 1804) 


v_mov_b32 v27, 0 // (op 1804) convert
v_mov_b32 v26, v14 // (op 1804) convert

v_lshrrev_b32 v13, 3, v1 // (op 1804) 
v_lshl_add_u32 v14, s17, 7, v13 // (op 1804) 

v_mul_lo_u32 v29, v14, s25 // (op 1804) most significant: low of low * high
v_mul_hi_u32 v13, v14, s24 // (op 1804) most significant: high of low * low
v_mul_lo_u32 v28, v14, s24 // (op 1804) least significant: low of low * low
v_add_u32 v29, v29, v13 // (op 1804) most significant: sum


v_add_co_u32 v30, s[0:1], v26, v28 // (op 1804) least significant half
v_addc_co_u32 v31, s[0:1], v27, v29, s[0:1] // (op 1804) most significant half



s_mul_i32 s1, s24, s9 // (op 1806) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 1806) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 1806) least significant
s_mul_hi_u32 s7, s24, s8 // (op 1806) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1806) most significant: sum
s_add_u32 s1, s1, s7 // (op 1806) most significant: sum


s_mov_b32 s52, s26 // (op 1806) reg expression
s_mov_b32 s53, s27 // (op 1806) reg expression
s_mov_b32 s54, s0 // (op 1806) reg expression
s_mov_b32 s55, 131072 // (op 1806) call()

v_lshrrev_b32 v13, 7, v1 // (op 1868) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1868) 
v_and_b32 v14, 15, v1 // (op 1868) 
s_lshr_b32 s6, s0, 1 // (op 1868) 

v_lshl_add_u32 v15, v13, 4, v14 // (op 1868) 


v_lshl_add_u32 v13, s16, 5, v15 // (op 1868) 

v_and_b32 v14, 63, v1 // (op 1868) 
v_lshrrev_b32 v15, 4, v14 // (op 1868) 

v_lshl_add_u32 v14, s6, 3, v15 // (op 1868) 


v_mul_lo_u32 v15, v13, s28 // (op 1868) 
v_add_u32 v15, v15, v14 // (op 1868) 


s_mul_i32 s1, s28, s23 // (op 1870) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 1870) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 1870) least significant
s_mul_hi_u32 s7, s28, s22 // (op 1870) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1870) most significant: sum
s_add_u32 s1, s1, s7 // (op 1870) most significant: sum


s_mov_b32 s56, s30 // (op 1870) reg expression
s_mov_b32 s57, s31 // (op 1870) reg expression
s_mov_b32 s58, s0 // (op 1870) reg expression
s_mov_b32 s59, 131072 // (op 1870) call()

v_and_b32 v13, -4, v1 // (op 1932) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1932) 
v_sub_u32 v14, v1, v13 // (op 1932) 

s_lshr_b32 s6, s0, 1 // (op 1932) 

v_lshrrev_b32 v13, 2, v1 // (op 1932) 
v_lshl_add_u32 v16, s6, 3, v14 // (op 1932) 


v_lshlrev_b32 v14, 1, v13 // (op 1932) 

v_lshl_add_u32 v13, s17, 7, v14 // (op 1932) 

v_mul_lo_u32 v14, v13, s32 // (op 1932) 

v_add_u32 v13, v16, v14 // (op 1932) 


s_mul_i32 s1, s32, s9 // (op 1934) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 1934) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 1934) least significant
s_mul_hi_u32 s7, s32, s8 // (op 1934) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1934) most significant: sum
s_add_u32 s1, s1, s7 // (op 1934) most significant: sum


s_mov_b32 s60, s34 // (op 1934) reg expression
s_mov_b32 s61, s35 // (op 1934) reg expression
s_mov_b32 s62, s0 // (op 1934) reg expression
s_mov_b32 s63, 131072 // (op 1934) call()

v_and_b32 v14, -8, v1 // (op 2025) 
v_sub_u32 v16, v1, v14 // (op 2025) 

v_lshlrev_b32 v14, 4, v16 // (op 2025) 

v_lshrrev_b32 v16, 3, v1 // (op 2025) 
v_lshl_add_u32 v17, v16, 7, v14 // (op 2025) 


v_mul_lo_u32 v14, v17, 6 // (op 2025) 

v_lshrrev_b32 v16, 3, v14 // (op 2025) 

v_and_b32 v14, -8, v1 // (op 2078) 
v_sub_u32 v17, v1, v14 // (op 2078) 

v_lshlrev_b32 v14, 4, v17 // (op 2078) 

v_lshrrev_b32 v17, 3, v1 // (op 2078) 
v_lshl_add_u32 v18, v17, 7, v14 // (op 2078) 


s_nop 0
v_readfirstlane_b32 s0, v18 // (op 2078) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 2135) 
v_sub_u32 v17, v1, v14 // (op 2135) 

v_lshrrev_b32 v14, 2, v1 // (op 2135) 
v_lshl_add_u32 v18, v14, 3, v17 // (op 2135) 


s_ashr_i64 s[6:7], s[36:37], 7 // (op 629) 
s_and_b32 s1, s6, 4294967294 // (op 629) 

s_ashr_i64 s[6:7], s[36:37], 7 // (op 628) 
s_mov_b32 s10, s6 // (op 628) convert

s_cmp_lt_u32 s1, s10 // (op 628) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopBottom_KLoopTail_628 // (op 628) Condition: Top (jump to RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopBottom_KLoopTail_628 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopTop_KLoopTail_628:


s_add_u32 s1, s1, 1 // (op 630) 
v_mov_b32 v14, v12 // (op 170) 

buffer_load_dwordx3 v[20:22], v14, s[48:51], 0 offen // (op 170) Load value

v_add_u32 v12, 96, v12 // (op 1724) 
s_barrier  // (op 2336) 
s_barrier  // (op 2349) 




s_lshl_b64 s[6:7], s[24:25], 5 // (op 1704) 
v_mov_b64 v[28:29], s[6:7] // (op 1704) convert


v_mov_b64 v[26:27], v[30:31] // (op 1704) 
v_mov_b64 v[24:25], v[26:27] // (op 1704) 
s_mov_b32 m0, s0 // (op 1704) reg expression
buffer_load_dwordx4 v24, s[52:55], 0 offen offset: 0 lds // (op 1704) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v28 // (op 1704) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 1704) most significant half

s_add_u32 m0, m0, 4096 // (op 1704) 
buffer_load_dwordx4 v24, s[52:55], 0 offen offset: 0 lds // (op 1704) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v28 // (op 1704) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 1704) most significant half

s_add_u32 m0, m0, 4096 // (op 1704) 
s_waitcnt vmcnt(2)
ds_write_b96 v16, v[20:22] offset:33280 // (op 171) Store local data 
buffer_load_dwordx4 v24, s[52:55], 0 offen offset: 0 lds // (op 1704) Load value direct to lds

v_add_co_u32 v24, s[6:7], v24, v28 // (op 1704) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 1704) most significant half

s_add_u32 m0, m0, 4096 // (op 1704) 
buffer_load_dwordx4 v24, s[52:55], 0 offen offset: 0 lds // (op 1704) Load value direct to lds



v_add_co_u32 v30, vcc, 128, v30 // (op 1796) least significant half
v_addc_co_u32 v31, vcc, 0, v31, vcc // (op 1796) most significant half

v_mov_b64 v[28:29], s[32:33] // (op 184) convert

v_mov_b32 v14, v13 // (op 184) 
v_mov_b32 v17, v14 // (op 184) 
buffer_load_ubyte v26, v17, s[60:63], 0 offen offset: 0 // (op 184) Load value
v_add_u32 v17, v17, v28 // (op 184) 
buffer_load_ubyte v27, v17, s[60:63], 0 offen offset: 0 // (op 184) Load value



v_add_u32 v13, 4, v13 // (op 1924) 
s_barrier  // (op 2362) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 2339) 






ds_read_b128 v[20:23], v3 offset:33280 // (op 173) Load local data 
ds_read_b64 v[24:25], v3 offset:33296 // (op 173) Load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v18, v26 offset:32768 // (op 182) Store local data 
s_barrier  // (op 2352) 
s_waitcnt vmcnt(0)
ds_write_b8 v18, v27 offset:32772 // (op 182) Store local data 




ds_read_b128 v[32:35], v5 // (op 175) Load local data 
ds_read_b128 v[36:39], v5 offset:64 // (op 175) Load local data 


v_mov_b32 v14, v15 // (op 177) 

buffer_load_ubyte v17, v14, s[56:59], 0 offen offset: 0 // (op 177) Load value

v_add_u32 v15, 4, v15 // (op 1860) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 2366) 



ds_read_u8 v14, v0 offset:32768 // (op 179) Load local data 



ds_read_b128 v[40:43], v5 offset:4096 // (op 324) Load local data 
ds_read_b128 v[44:47], v5 offset:4160 // (op 324) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:25], v[32:39], a[0:3], v17, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 180) 





ds_read_u8 v14, v0 offset:32896 // (op 327) Load local data 



ds_read_b128 v[32:35], v5 offset:8192 // (op 480) Load local data 
ds_read_b128 v[36:39], v5 offset:8256 // (op 480) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[20:25], v[40:47], a[4:7], v17, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 328) 





ds_read_u8 v14, v0 offset:33024 // (op 483) Load local data 



ds_read_b128 v[40:43], v5 offset:12288 // (op 636) Load local data 
ds_read_b128 v[44:47], v5 offset:12352 // (op 636) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[20:25], v[32:39], a[8:11], v17, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 484) 





ds_read_u8 v14, v0 offset:33152 // (op 639) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[20:25], v[40:47], a[12:15], v17, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 640) 




s_ashr_i64 s[6:7], s[36:37], 7 // (op 628) 
s_mov_b32 s10, s6 // (op 628) convert

s_cmp_lt_u32 s1, s10 // (op 628) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopTop_KLoopTail_628 // (op 628) Condition: Bottom (jump to RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopTop_KLoopTail_628 if true)
RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopBottom_KLoopTail_628:






















s_barrier  // (op 2454) 







v_cmp_eq_f32 vcc, 0.00000, v10 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ConditionalFalse_0_beta_125
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
s_branch RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ConditionalBottom_0_beta_125
RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ConditionalFalse_0_beta_125:


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
RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ConditionalBottom_0_beta_125:












s_endpgm  // End of RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2
.LRR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_end:
.size RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2, .LRR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_end-RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2
  .amdhsa_next_free_vgpr 80
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 39936 // lds bytes
  .amdhsa_accum_offset 64
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
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 4]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2
    .symbol: RR_GEMM_TN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 39936
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 64
    .vgpr_count: 62
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

