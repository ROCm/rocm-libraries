// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2
.p2align 8
.type RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2,@function
RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 6, v1 // (op 2144) 
v_and_b32 v3, 1, v0 // (op 2144) 

v_and_b32 v0, 15, v1 // (op 2144) 
v_lshl_add_u32 v4, v3, 4, v0 // (op 2144) 


v_and_b32 v0, 48, v1 // (op 2144) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 2144) 


v_lshrrev_b32 v0, 6, v1 // (op 2288) 
v_and_b32 v4, 1, v0 // (op 2288) 

v_and_b32 v0, 15, v1 // (op 2288) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 2288) 


v_and_b32 v0, 48, v1 // (op 2288) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 2288) 


v_lshrrev_b32 v0, 7, v1 // (op 2030) 
v_and_b32 v5, 63, v1 // (op 2030) 
v_lshrrev_b32 v6, 4, v5 // (op 2030) 
v_lshlrev_b32 v7, 2, v6 // (op 2030) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 2030) 


v_lshl_add_u32 v0, s16, 7, v6 // (op 2030) 

v_lshrrev_b32 v6, 2, v5 // (op 2030) 

v_and_b32 v5, 3, v6 // (op 2030) 

v_and_b32 v6, 3, v1 // (op 2030) 
v_lshl_add_u32 v7, v5, 2, v6 // (op 2030) 


v_lshrrev_b32 v5, 6, v1 // (op 2030) 
v_and_b32 v6, 1, v5 // (op 2030) 

v_lshl_add_u32 v5, v6, 4, v7 // (op 2030) 


v_lshl_add_u32 v6, s17, 5, v5 // (op 2030) 

v_mul_lo_u32 v5, v6, s2 // (op 2030) 

v_add_lshl_u32 v6, v0, v5, 2 // (op 2030) 


v_lshrrev_b32 v0, 7, v1 // (op 2092) 
v_and_b32 v5, 15, v1 // (op 2092) 
v_lshl_add_u32 v7, v0, 4, v5 // (op 2092) 


v_and_b32 v0, 63, v1 // (op 2092) 
v_lshrrev_b32 v5, 4, v0 // (op 2092) 

v_lshlrev_b32 v0, 5, v5 // (op 2092) 

v_lshl_add_u32 v5, v7, 7, v0 // (op 2092) 


v_mul_lo_u32 v0, v5, 6 // (op 2092) 

v_lshrrev_b32 v5, 3, v0 // (op 2092) 

v_lshrrev_b32 v0, 7, v1 // (op 2202) 
v_and_b32 v7, 15, v1 // (op 2202) 
s_mul_i32 s1, s2, s9 // (op 2032) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 2032) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 2032) least significant
s_mul_hi_u32 s19, s2, s8 // (op 2032) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2032) most significant: sum
s_add_u32 s1, s1, s19 // (op 2032) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 2032) 

v_lshl_add_u32 v8, v0, 4, v7 // (op 2202) 


v_and_b32 v0, 63, v1 // (op 2202) 
v_lshrrev_b32 v7, 4, v0 // (op 2202) 

v_lshl_add_u32 v0, v8, 2, v7 // (op 2202) 


v_lshrrev_b32 v7, 7, v1 // (op 2218) 
v_and_b32 v8, 63, v1 // (op 2218) 
v_lshrrev_b32 v9, 4, v8 // (op 2218) 
v_lshlrev_b32 v10, 2, v9 // (op 2218) 

v_lshl_add_u32 v9, v7, 4, v10 // (op 2218) 


v_lshl_add_u32 v7, s16, 7, v9 // (op 2218) 

v_lshrrev_b32 v9, 2, v8 // (op 2218) 

v_and_b32 v8, 3, v9 // (op 2218) 

v_and_b32 v9, 3, v1 // (op 2218) 
v_lshl_add_u32 v10, v8, 2, v9 // (op 2218) 


v_lshrrev_b32 v8, 6, v1 // (op 2218) 
v_and_b32 v9, 1, v8 // (op 2218) 

v_lshl_add_u32 v8, v9, 4, v10 // (op 2218) 


v_lshl_add_u32 v9, s17, 5, v8 // (op 2218) 

v_mul_lo_u32 v8, v9, s4 // (op 2218) 

v_add_lshl_u32 v9, v7, v8, 2 // (op 2218) 


v_lshrrev_b32 v7, 7, v1 // (op 2249) 
v_and_b32 v8, 15, v1 // (op 2249) 
v_lshl_add_u32 v10, v7, 4, v8 // (op 2249) 


v_and_b32 v7, 63, v1 // (op 2249) 
v_lshrrev_b32 v8, 4, v7 // (op 2249) 

v_lshlrev_b32 v7, 5, v8 // (op 2249) 

v_lshl_add_u32 v8, v10, 7, v7 // (op 2249) 


v_mul_lo_u32 v7, v8, 6 // (op 2249) 

v_lshrrev_b32 v8, 3, v7 // (op 2249) 

v_lshrrev_b32 v7, 7, v1 // (op 2317) 
v_and_b32 v10, 15, v1 // (op 2317) 
s_mul_i32 s1, s4, s9 // (op 2220) most significant: low of low * high
v_lshl_add_u32 v11, v7, 4, v10 // (op 2317) 


v_and_b32 v7, 63, v1 // (op 2317) 
v_lshrrev_b32 v10, 4, v7 // (op 2317) 

v_lshl_add_u32 v7, v11, 2, v10 // (op 2317) 


s_mul_i32 s38, s5, s8 // (op 2220) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 2220) least significant
s_mul_hi_u32 s39, s4, s8 // (op 2220) most significant: high of low * low
s_add_u32 s1, s1, s38 // (op 2220) most significant: sum
s_add_u32 s1, s1, s39 // (op 2220) most significant: sum


s_lshl_b64 s[38:39], s[0:1], 2 // (op 2220) 

s_mov_b32 s40, s6 // (op 2032) reg expression
s_mov_b32 s41, s7 // (op 2032) reg expression
s_mov_b32 s42, s18 // (op 2032) reg expression
s_mov_b32 s43, 131072 // (op 2032) call()

s_mov_b32 s44, s10 // (op 2220) reg expression

s_mov_b32 s45, s11 // (op 2220) reg expression
s_mov_b32 s46, s38 // (op 2220) reg expression
s_mov_b32 s47, 131072 // (op 2220) call()


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
v_lshrrev_b32 v12, 1, v1 // (op 1797) 
v_lshl_add_u32 v13, s16, 7, v12 // (op 1797) 

v_and_b32 v12, -2, v1 // (op 1797) 
v_sub_u32 v14, v1, v12 // (op 1797) 

v_lshlrev_b32 v12, 6, v14 // (op 1797) 

v_mov_b32 v15, 0 // (op 1797) convert
v_mov_b32 v14, v12 // (op 1797) convert

v_mul_lo_u32 v17, v13, s13 // (op 1797) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 1797) most significant: high of low * low
v_mul_lo_u32 v16, v13, s12 // (op 1797) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 1797) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 1797) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 1797) most significant half



v_mul_lo_u32 v13, v17, 6 // (op 1797) most significant: low of high * low
v_mul_hi_u32 v14, v16, 6 // (op 1797) most significant: high of low * low
v_mul_lo_u32 v12, v16, 6 // (op 1797) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 1797) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 1797) 

v_mov_b32 v12, v14 // (op 1797) convert


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s23, 6 // (op 1799) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 1799) least significant
s_mul_hi_u32 s6, s22, 6 // (op 1799) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1799) most significant: sum

s_mul_i32 s7, s12, s1 // (op 1799) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 1799) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 1799) least significant
s_mul_hi_u32 s11, s12, s0 // (op 1799) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 1799) most significant: sum
s_add_u32 s7, s7, s11 // (op 1799) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 1799) 

s_mov_b32 s48, s20 // (op 1799) reg expression
s_mov_b32 s49, s21 // (op 1799) reg expression
s_mov_b32 s50, s0 // (op 1799) reg expression
s_mov_b32 s51, 131072 // (op 1799) call()

v_lshrrev_b32 v13, 1, v1 // (op 1816) 
v_lshl_add_u32 v14, s16, 7, v13 // (op 1816) 

v_and_b32 v13, -2, v1 // (op 1816) 
v_sub_u32 v15, v1, v13 // (op 1816) 

v_lshlrev_b32 v13, 6, v15 // (op 1816) 

v_mov_b32 v17, 0 // (op 1816) convert
v_mov_b32 v16, v13 // (op 1816) convert

v_mul_lo_u32 v19, v14, s13 // (op 1816) most significant: low of low * high
v_mul_hi_u32 v13, v14, s12 // (op 1816) most significant: high of low * low
v_mul_lo_u32 v18, v14, s12 // (op 1816) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 1816) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 1816) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 1816) most significant half



v_mul_lo_u32 v17, v19, 6 // (op 1816) most significant: low of high * low
v_mul_hi_u32 v13, v18, 6 // (op 1816) most significant: high of low * low
v_mul_lo_u32 v16, v18, 6 // (op 1816) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 1816) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 1816) 

v_mov_b32 v13, v14 // (op 1816) convert

s_mul_i32 s1, s23, 6 // (op 1818) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 1818) least significant
s_mul_hi_u32 s6, s22, 6 // (op 1818) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1818) most significant: sum

s_mul_i32 s7, s12, s1 // (op 1818) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 1818) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 1818) least significant
s_mul_hi_u32 s11, s12, s0 // (op 1818) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 1818) most significant: sum
s_add_u32 s7, s7, s11 // (op 1818) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 1818) 

s_mov_b32 s48, s20 // (op 1818) reg expression
s_mov_b32 s49, s21 // (op 1818) reg expression
s_mov_b32 s50, s0 // (op 1818) reg expression
s_mov_b32 s51, 131072 // (op 1818) call()

v_and_b32 v14, -8, v1 // (op 1861) 
v_sub_u32 v15, v1, v14 // (op 1861) 

v_lshlrev_b32 v14, 4, v15 // (op 1861) 

v_mov_b32 v17, 0 // (op 1861) convert
v_mov_b32 v16, v14 // (op 1861) convert

v_lshrrev_b32 v14, 3, v1 // (op 1861) 
v_lshl_add_u32 v15, s17, 5, v14 // (op 1861) 

v_mul_lo_u32 v19, v15, s25 // (op 1861) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 1861) most significant: high of low * low
v_mul_lo_u32 v18, v15, s24 // (op 1861) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 1861) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v18 // (op 1861) least significant half
v_addc_co_u32 v15, s[0:1], v17, v19, s[0:1] // (op 1861) most significant half



s_mul_i32 s1, s24, s9 // (op 1863) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 1863) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 1863) least significant
s_mul_hi_u32 s7, s24, s8 // (op 1863) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1863) most significant: sum
s_add_u32 s1, s1, s7 // (op 1863) most significant: sum


s_mov_b32 s52, s26 // (op 1863) reg expression
s_mov_b32 s53, s27 // (op 1863) reg expression
s_mov_b32 s54, s0 // (op 1863) reg expression
s_mov_b32 s55, 131072 // (op 1863) call()

v_and_b32 v16, -8, v1 // (op 1880) 
v_sub_u32 v17, v1, v16 // (op 1880) 

v_lshlrev_b32 v16, 4, v17 // (op 1880) 

v_mov_b32 v19, 0 // (op 1880) convert
v_mov_b32 v18, v16 // (op 1880) convert

v_lshrrev_b32 v16, 3, v1 // (op 1880) 
v_lshl_add_u32 v17, s17, 5, v16 // (op 1880) 

v_mul_lo_u32 v21, v17, s25 // (op 1880) most significant: low of low * high
v_mul_hi_u32 v16, v17, s24 // (op 1880) most significant: high of low * low
v_mul_lo_u32 v20, v17, s24 // (op 1880) least significant: low of low * low
v_add_u32 v21, v21, v16 // (op 1880) most significant: sum


v_add_co_u32 v16, s[0:1], v18, v20 // (op 1880) least significant half
v_addc_co_u32 v17, s[0:1], v19, v21, s[0:1] // (op 1880) most significant half



s_mul_i32 s1, s24, s9 // (op 1882) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 1882) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 1882) least significant
s_mul_hi_u32 s7, s24, s8 // (op 1882) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1882) most significant: sum
s_add_u32 s1, s1, s7 // (op 1882) most significant: sum


s_mov_b32 s52, s26 // (op 1882) reg expression
s_mov_b32 s53, s27 // (op 1882) reg expression
s_mov_b32 s54, s0 // (op 1882) reg expression
s_mov_b32 s55, 131072 // (op 1882) call()

v_lshrrev_b32 v18, 2, v1 // (op 1925) 
v_lshlrev_b32 v19, 1, v18 // (op 1925) 

v_lshl_add_u32 v18, s16, 7, v19 // (op 1925) 

v_and_b32 v19, -4, v1 // (op 1925) 
v_sub_u32 v20, v1, v19 // (op 1925) 

v_mul_lo_u32 v19, v18, s28 // (op 1925) 
v_add_u32 v19, v19, v20 // (op 1925) 


s_mul_i32 s1, s28, s23 // (op 1927) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 1927) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 1927) least significant
s_mul_hi_u32 s7, s28, s22 // (op 1927) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1927) most significant: sum
s_add_u32 s1, s1, s7 // (op 1927) most significant: sum


s_mov_b32 s56, s30 // (op 1927) reg expression
s_mov_b32 s57, s31 // (op 1927) reg expression
s_mov_b32 s58, s0 // (op 1927) reg expression
s_mov_b32 s59, 131072 // (op 1927) call()

v_lshrrev_b32 v18, 2, v1 // (op 1944) 
v_lshlrev_b32 v20, 1, v18 // (op 1944) 

v_lshl_add_u32 v18, s16, 7, v20 // (op 1944) 

v_and_b32 v20, -4, v1 // (op 1944) 
v_sub_u32 v21, v1, v20 // (op 1944) 

v_mul_lo_u32 v20, v18, s28 // (op 1944) 
v_add_u32 v20, v20, v21 // (op 1944) 


s_mul_i32 s1, s28, s23 // (op 1946) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 1946) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 1946) least significant
s_mul_hi_u32 s7, s28, s22 // (op 1946) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1946) most significant: sum
s_add_u32 s1, s1, s7 // (op 1946) most significant: sum


s_mov_b32 s56, s30 // (op 1946) reg expression
s_mov_b32 s57, s31 // (op 1946) reg expression
s_mov_b32 s58, s0 // (op 1946) reg expression
s_mov_b32 s59, 131072 // (op 1946) call()

v_lshrrev_b32 v18, 6, v1 // (op 1989) 
v_and_b32 v21, 1, v18 // (op 1989) 

v_and_b32 v18, 15, v1 // (op 1989) 
v_lshl_add_u32 v22, v21, 4, v18 // (op 1989) 


v_lshl_add_u32 v18, s17, 5, v22 // (op 1989) 

v_mul_lo_u32 v21, v18, s32 // (op 1989) 

v_and_b32 v18, 63, v1 // (op 1989) 
v_lshrrev_b32 v22, 4, v18 // (op 1989) 

v_add_u32 v18, v22, v21 // (op 1989) 


s_mul_i32 s1, s32, s9 // (op 1991) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 1991) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 1991) least significant
s_mul_hi_u32 s7, s32, s8 // (op 1991) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1991) most significant: sum
s_add_u32 s1, s1, s7 // (op 1991) most significant: sum


s_mov_b32 s60, s34 // (op 1991) reg expression
s_mov_b32 s61, s35 // (op 1991) reg expression
s_mov_b32 s62, s0 // (op 1991) reg expression
s_mov_b32 s63, 131072 // (op 1991) call()

v_lshrrev_b32 v21, 6, v1 // (op 2008) 
v_and_b32 v22, 1, v21 // (op 2008) 

v_and_b32 v21, 15, v1 // (op 2008) 
v_lshl_add_u32 v23, v22, 4, v21 // (op 2008) 


v_lshl_add_u32 v21, s17, 5, v23 // (op 2008) 

v_mul_lo_u32 v22, v21, s32 // (op 2008) 

v_and_b32 v21, 63, v1 // (op 2008) 
v_lshrrev_b32 v23, 4, v21 // (op 2008) 

v_add_u32 v21, v23, v22 // (op 2008) 


s_mul_i32 s1, s32, s9 // (op 2010) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 2010) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 2010) least significant
s_mul_hi_u32 s7, s32, s8 // (op 2010) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2010) most significant: sum
s_add_u32 s1, s1, s7 // (op 2010) most significant: sum


s_mov_b32 s60, s34 // (op 2010) reg expression
s_mov_b32 s61, s35 // (op 2010) reg expression
s_mov_b32 s62, s0 // (op 2010) reg expression
s_mov_b32 s63, 131072 // (op 2010) call()




buffer_load_ubyte v22, v18, s[60:63], 0 offen offset: 0 // (op 1598) Load value
v_and_b32 v23, -8, v1 // (op 2132) 
v_sub_u32 v24, v1, v23 // (op 2132) 

v_lshlrev_b32 v23, 4, v24 // (op 2132) 

v_lshrrev_b32 v24, 3, v1 // (op 2132) 
v_lshl_add_u32 v25, v24, 7, v23 // (op 2132) 


s_nop 0
v_readfirstlane_b32 s0, v25 // (op 2132) Wait state hazard: VALU Write Readlane Hazard




s_nop 1
s_mov_b32 m0, s0 // (op 1750) reg expression
buffer_load_dwordx4 v14, s[52:55], 0 offen offset: 0 lds // (op 1750) Load value direct to lds




buffer_load_dwordx4 v[24:27], v12, s[48:51], 0 offen // (op 1605) Load value
buffer_load_dwordx2 v[28:29], v12, s[48:51], 0 offen offset:16 // (op 1605) Load value
buffer_load_dwordx4 v[30:33], v12, s[48:51], 0 offen offset:24 // (op 1605) Load value
buffer_load_dwordx2 v[34:35], v12, s[48:51], 0 offen offset:40 // (op 1605) Load value

v_mov_b64 v[36:37], s[28:29] // (op 1608) convert

v_mov_b32 v23, v19 // (op 1608) 
buffer_load_ubyte v38, v23, s[56:59], 0 offen offset: 0 // (op 1608) Load value
v_add_u32 v23, v23, v36 // (op 1608) 
buffer_load_ubyte v39, v23, s[56:59], 0 offen offset: 0 // (op 1608) Load value





buffer_load_ubyte v23, v18, s[60:63], 0 offen offset:4 // (op 1612) Load value

v_and_b32 v18, -8, v1 // (op 2276) 
v_sub_u32 v36, v1, v18 // (op 2276) 

v_lshlrev_b32 v18, 4, v36 // (op 2276) 

v_lshrrev_b32 v36, 3, v1 // (op 2276) 
v_lshl_add_u32 v37, v36, 7, v18 // (op 2276) 


s_nop 0
v_readfirstlane_b32 s0, v37 // (op 2276) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s1, 4096, s0 // (op 1755) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 1755) reg expression
s_mov_b32 s6, 128 // (op 1755) call()
buffer_load_dwordx4 v14, s[52:55], s6 offen offset: 0 lds // (op 1755) Load value direct to lds







buffer_load_dwordx4 v[40:43], v12, s[48:51], 0 offen offset:96 // (op 1619) Load value
buffer_load_dwordx2 v[44:45], v12, s[48:51], 0 offen offset:112 // (op 1619) Load value
buffer_load_dwordx4 v[46:49], v12, s[48:51], 0 offen offset:120 // (op 1619) Load value
buffer_load_dwordx2 v[50:51], v12, s[48:51], 0 offen offset:136 // (op 1619) Load value


v_mov_b64 v[14:15], s[28:29] // (op 1622) convert

v_mov_b32 v12, v19 // (op 1622) 
buffer_load_ubyte v36, v12, s[56:59], 0 offen offset:4 // (op 1622) Load value
v_add_u32 v12, v12, v14 // (op 1622) 
buffer_load_ubyte v37, v12, s[56:59], 0 offen offset:4 // (op 1622) Load value



v_and_b32 v12, -2, v1 // (op 2075) 
v_sub_u32 v14, v1, v12 // (op 2075) 

v_lshlrev_b32 v12, 6, v14 // (op 2075) 

v_lshrrev_b32 v14, 1, v1 // (op 2075) 
v_lshl_add_u32 v15, v14, 7, v12 // (op 2075) 


v_mul_lo_u32 v12, v15, 6 // (op 2075) 

v_lshrrev_b32 v14, 3, v12 // (op 2075) 




s_waitcnt vmcnt(13)
ds_write_b128 v14, v[24:27] offset:8192 // (op 1625) Store local data 
s_waitcnt vmcnt(12)
ds_write_b64 v14, v[28:29] offset:8208 // (op 1625) Store local data 
s_waitcnt vmcnt(11)
ds_write_b128 v14, v[30:33] offset:8216 // (op 1625) Store local data 
s_waitcnt vmcnt(10)
ds_write_b64 v14, v[34:35] offset:8232 // (op 1625) Store local data 

v_and_b32 v12, -4, v1 // (op 2185) 
v_sub_u32 v14, v1, v12 // (op 2185) 

v_lshrrev_b32 v12, 2, v1 // (op 2185) 
v_lshl_add_u32 v15, v12, 3, v14 // (op 2185) 





s_waitcnt vmcnt(9)
ds_write_b8 v15, v38 offset:20480 // (op 1626) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v15, v39 offset:20484 // (op 1626) Store local data 

v_and_b32 v12, -2, v1 // (op 2046) 
v_sub_u32 v14, v1, v12 // (op 2046) 

v_lshlrev_b32 v12, 6, v14 // (op 2046) 

v_lshrrev_b32 v14, 1, v1 // (op 2046) 
v_lshl_add_u32 v15, v14, 7, v12 // (op 2046) 


v_mul_lo_u32 v12, v15, 6 // (op 2046) 

v_lshrrev_b32 v14, 3, v12 // (op 2046) 

v_and_b32 v12, -8, v1 // (op 2104) 
v_sub_u32 v15, v1, v12 // (op 2104) 

v_lshlrev_b32 v12, 4, v15 // (op 2104) 

v_lshrrev_b32 v15, 3, v1 // (op 2104) 
v_lshl_add_u32 v18, v15, 7, v12 // (op 2104) 


s_nop 0
v_readfirstlane_b32 s0, v18 // (op 2104) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -4, v1 // (op 2156) 
v_sub_u32 v15, v1, v12 // (op 2156) 

v_lshrrev_b32 v12, 2, v1 // (op 2156) 
v_lshl_add_u32 v18, v12, 3, v15 // (op 2156) 


v_and_b32 v12, -2, v1 // (op 2232) 
v_sub_u32 v15, v1, v12 // (op 2232) 

v_lshlrev_b32 v12, 6, v15 // (op 2232) 

v_lshrrev_b32 v15, 1, v1 // (op 2232) 
v_lshl_add_u32 v19, v15, 7, v12 // (op 2232) 


v_mul_lo_u32 v12, v19, 6 // (op 2232) 

v_lshrrev_b32 v15, 3, v12 // (op 2232) 

v_and_b32 v12, -8, v1 // (op 2261) 
v_sub_u32 v19, v1, v12 // (op 2261) 

v_lshlrev_b32 v12, 4, v19 // (op 2261) 

v_lshrrev_b32 v19, 3, v1 // (op 2261) 
v_lshl_add_u32 v52, v19, 7, v12 // (op 2261) 


s_nop 0
v_readfirstlane_b32 s1, v52 // (op 2261) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -4, v1 // (op 2300) 
v_sub_u32 v19, v1, v12 // (op 2300) 

v_lshrrev_b32 v12, 2, v1 // (op 2300) 
v_lshl_add_u32 v52, v12, 3, v19 // (op 2300) 


s_mov_b32 s6, 0 // (op 635) call()
s_ashr_i64 s[10:11], s[36:37], 7 // (op 634) 
s_mov_b32 s7, s10 // (op 634) convert

s_and_b32 s10, s7, 4294967294 // (op 634) 

s_cmp_lt_u32 s6, s10 // (op 634) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopBottom_KLoop_634 // (op 634) Condition: Top (jump to RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopBottom_KLoop_634 if false)
RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopTop_KLoop_634:
s_add_u32 s6, s6, 2 // (op 636) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 2364) 



ds_read_b128 v[54:57], v5 offset:8192 // (op 65) Load local data 
ds_read_b64 v[58:59], v5 offset:8208 // (op 65) Load local data 
s_barrier  // (op 2377) 



ds_read_b128 v[60:63], v3 // (op 70) Load local data 
ds_read_b128 v[64:67], v3 offset:64 // (op 70) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 2390) 



ds_read_u8 v12, v0 offset:20480 // (op 75) Load local data 



ds_read_b128 v[68:71], v5 offset:11264 // (op 340) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[54:59], v[60:67], a[0:3], v12, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 82) 
ds_read_b64 v[72:73], v5 offset:11280 // (op 340) Load local data 





ds_read_u8 v12, v0 offset:20608 // (op 344) Load local data 



ds_read_b128 v[54:57], v5 offset:14336 // (op 502) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[68:73], v[60:67], a[4:7], v12, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 346) 
ds_read_b64 v[58:59], v5 offset:14352 // (op 502) Load local data 





ds_read_u8 v12, v0 offset:20736 // (op 506) Load local data 



ds_read_b128 v[68:71], v5 offset:17408 // (op 664) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[54:59], v[60:67], a[8:11], v12, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 508) 
ds_read_b64 v[72:73], v5 offset:17424 // (op 664) Load local data 





ds_read_u8 v12, v0 offset:20864 // (op 668) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[68:73], v[60:67], a[12:15], v12, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 670) 





v_mov_b32 v12, v21 // (op 80) 

buffer_load_ubyte v22, v12, s[60:63], 0 offen offset:8 // (op 80) Load value
s_barrier  // (op 2380) 


v_mov_b64 v[54:55], v[16:17] // (op 1733) 

s_mov_b32 m0, s0 // (op 1733) reg expression
s_mov_b32 s7, 256 // (op 1733) call()
buffer_load_dwordx4 v54, s[52:55], s7 offen offset: 0 lds // (op 1733) Load value direct to lds



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


s_barrier  // (op 2407) 



s_waitcnt vmcnt(13)
ds_write_b128 v15, v[40:43] offset:20992 // (op 225) Store local data 
s_waitcnt vmcnt(12)
ds_write_b64 v15, v[44:45] offset:21008 // (op 225) Store local data 
s_waitcnt vmcnt(11)
ds_write_b128 v15, v[46:49] offset:21016 // (op 225) Store local data 
s_waitcnt vmcnt(10)
ds_write_b64 v15, v[50:51] offset:21032 // (op 225) Store local data 
s_barrier  // (op 2419) 



s_waitcnt vmcnt(9)
ds_write_b8 v52, v36 offset:33280 // (op 236) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v52, v37 offset:33284 // (op 236) Store local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 2404) 



ds_read_b128 v[56:59], v8 offset:20992 // (op 227) Load local data 
ds_read_b64 v[60:61], v8 offset:21008 // (op 227) Load local data 
s_barrier  // (op 2410) 



ds_read_b128 v[62:65], v4 offset:4096 // (op 229) Load local data 
ds_read_b128 v[66:69], v4 offset:4160 // (op 229) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 2416) 



ds_read_u8 v70, v7 offset:33280 // (op 231) Load local data 



ds_read_b128 v[72:75], v8 offset:24064 // (op 353) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[56:61], v[62:69], a[0:3], v70, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 234) 
ds_read_b64 v[76:77], v8 offset:24080 // (op 353) Load local data 





ds_read_u8 v56, v7 offset:33408 // (op 357) Load local data 



ds_read_b128 v[78:81], v8 offset:27136 // (op 515) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[72:77], v[62:69], a[4:7], v56, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 347) 
ds_read_b64 v[82:83], v8 offset:27152 // (op 515) Load local data 





ds_read_u8 v56, v7 offset:33536 // (op 519) Load local data 



ds_read_b128 v[70:73], v8 offset:30208 // (op 677) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[78:83], v[62:69], a[8:11], v56, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 509) 
ds_read_b64 v[74:75], v8 offset:30224 // (op 677) Load local data 





s_barrier  // (op 2492) 
ds_read_u8 v56, v7 offset:33664 // (op 681) Load local data 
s_barrier  // (op 2496) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[70:75], v[62:69], a[12:15], v56, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 671) 






buffer_load_ubyte v23, v12, s[60:63], 0 offen offset:12 // (op 233) Load value

s_barrier  // (op 2413) 
v_add_u32 v21, 8, v21 // (op 2000) 


s_add_u32 s7, 4096, s1 // (op 1745) 

s_mov_b32 m0, s7 // (op 1745) reg expression
s_mov_b32 s10, 384 // (op 1745) call()
buffer_load_dwordx4 v54, s[52:55], s10 offen offset: 0 lds // (op 1745) Load value direct to lds






v_add_co_u32 v16, vcc, 256, v16 // (op 1872) least significant half
v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 1872) most significant half
buffer_load_dwordx4 v[40:43], v19, s[48:51], 0 offen offset:288 // (op 224) Load value
buffer_load_dwordx2 v[44:45], v19, s[48:51], 0 offen offset:304 // (op 224) Load value
buffer_load_dwordx4 v[46:49], v19, s[48:51], 0 offen offset:312 // (op 224) Load value
buffer_load_dwordx2 v[50:51], v19, s[48:51], 0 offen offset:328 // (op 224) Load value


v_add_u32 v13, 192, v13 // (op 1808) 
v_mov_b64 v[54:55], s[28:29] // (op 238) convert

v_mov_b32 v12, v53 // (op 238) 
buffer_load_ubyte v36, v12, s[56:59], 0 offen offset:12 // (op 238) Load value
v_add_u32 v12, v12, v54 // (op 238) 
buffer_load_ubyte v37, v12, s[56:59], 0 offen offset:12 // (op 238) Load value



s_barrier  // (op 2367) 
v_add_u32 v20, 8, v20 // (op 1936) 



s_waitcnt vmcnt(13)
ds_write_b128 v14, v[24:27] offset:8192 // (op 6) Store local data 
s_waitcnt vmcnt(12)
ds_write_b64 v14, v[28:29] offset:8208 // (op 6) Store local data 
s_waitcnt vmcnt(11)
ds_write_b128 v14, v[30:33] offset:8216 // (op 6) Store local data 
s_waitcnt vmcnt(10)
ds_write_b64 v14, v[34:35] offset:8232 // (op 6) Store local data 
s_barrier  // (op 2393) 



s_waitcnt vmcnt(9)
ds_write_b8 v18, v38 offset:20480 // (op 21) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v18, v39 offset:20484 // (op 21) Store local data 
s_ashr_i64 s[10:11], s[36:37], 7 // (op 634) 
s_mov_b32 s7, s10 // (op 634) convert

s_and_b32 s10, s7, 4294967294 // (op 634) 

s_cmp_lt_u32 s6, s10 // (op 634) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopTop_KLoop_634 // (op 634) Condition: Bottom (jump to RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopTop_KLoop_634 if true)
RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopBottom_KLoop_634:














s_barrier  // (op 2476) 
v_lshrrev_b32 v12, 1, v1 // (op 1768) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1768) 
v_lshl_add_u32 v13, s16, 7, v12 // (op 1768) 

s_lshr_b32 s6, s0, 1 // (op 1768) 

v_and_b32 v12, -2, v1 // (op 1768) 
v_sub_u32 v14, v1, v12 // (op 1768) 

v_lshlrev_b32 v12, 6, v14 // (op 1768) 

v_lshl_add_u32 v14, s6, 8, v12 // (op 1768) 


v_mov_b32 v39, 0 // (op 1768) convert
v_mov_b32 v38, v14 // (op 1768) convert

v_mul_lo_u32 v15, v13, s13 // (op 1768) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 1768) most significant: high of low * low
v_mul_lo_u32 v14, v13, s12 // (op 1768) least significant: low of low * low
v_add_u32 v15, v15, v12 // (op 1768) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v38 // (op 1768) least significant half
v_addc_co_u32 v15, s[0:1], v15, v39, s[0:1] // (op 1768) most significant half



v_mul_lo_u32 v13, v15, 6 // (op 1768) most significant: low of high * low
v_mul_hi_u32 v16, v14, 6 // (op 1768) most significant: high of low * low
v_mul_lo_u32 v12, v14, 6 // (op 1768) least significant: low of low * low
v_add_u32 v13, v13, v16 // (op 1768) most significant: sum


v_lshrrev_b64 v[38:39], 3, v[12:13] // (op 1768) 

v_mov_b32 v12, v38 // (op 1768) convert

s_mul_i32 s1, s23, 6 // (op 1770) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 1770) least significant
s_mul_hi_u32 s6, s22, 6 // (op 1770) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1770) most significant: sum

s_mul_i32 s7, s12, s1 // (op 1770) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 1770) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 1770) least significant
s_mul_hi_u32 s11, s12, s0 // (op 1770) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 1770) most significant: sum
s_add_u32 s7, s7, s11 // (op 1770) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 1770) 

s_mov_b32 s48, s20 // (op 1770) reg expression
s_mov_b32 s49, s21 // (op 1770) reg expression
s_mov_b32 s50, s0 // (op 1770) reg expression
s_mov_b32 s51, 131072 // (op 1770) call()

v_and_b32 v13, -8, v1 // (op 1838) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1838) 
v_sub_u32 v14, v1, v13 // (op 1838) 

s_lshr_b32 s6, s0, 1 // (op 1838) 

v_lshlrev_b32 v13, 4, v14 // (op 1838) 

v_lshl_add_u32 v14, s6, 8, v13 // (op 1838) 


v_mov_b32 v39, 0 // (op 1838) convert
v_mov_b32 v38, v14 // (op 1838) convert

v_lshrrev_b32 v13, 3, v1 // (op 1838) 
v_lshl_add_u32 v14, s17, 5, v13 // (op 1838) 

v_mul_lo_u32 v25, v14, s25 // (op 1838) most significant: low of low * high
v_mul_hi_u32 v13, v14, s24 // (op 1838) most significant: high of low * low
v_mul_lo_u32 v24, v14, s24 // (op 1838) least significant: low of low * low
v_add_u32 v25, v25, v13 // (op 1838) most significant: sum


v_add_co_u32 v26, s[0:1], v38, v24 // (op 1838) least significant half
v_addc_co_u32 v27, s[0:1], v39, v25, s[0:1] // (op 1838) most significant half



s_mul_i32 s1, s24, s9 // (op 1840) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 1840) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 1840) least significant
s_mul_hi_u32 s7, s24, s8 // (op 1840) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1840) most significant: sum
s_add_u32 s1, s1, s7 // (op 1840) most significant: sum


s_mov_b32 s52, s26 // (op 1840) reg expression
s_mov_b32 s53, s27 // (op 1840) reg expression
s_mov_b32 s54, s0 // (op 1840) reg expression
s_mov_b32 s55, 131072 // (op 1840) call()

v_lshrrev_b32 v13, 2, v1 // (op 1902) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1902) 
v_lshlrev_b32 v14, 1, v13 // (op 1902) 

s_lshr_b32 s6, s0, 1 // (op 1902) 

v_lshl_add_u32 v13, s16, 7, v14 // (op 1902) 

v_and_b32 v14, -4, v1 // (op 1902) 
v_sub_u32 v15, v1, v14 // (op 1902) 

v_lshl_add_u32 v14, s6, 3, v15 // (op 1902) 


v_mul_lo_u32 v15, v13, s28 // (op 1902) 
v_add_u32 v15, v15, v14 // (op 1902) 


s_mul_i32 s1, s28, s23 // (op 1904) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 1904) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 1904) least significant
s_mul_hi_u32 s7, s28, s22 // (op 1904) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1904) most significant: sum
s_add_u32 s1, s1, s7 // (op 1904) most significant: sum


s_mov_b32 s56, s30 // (op 1904) reg expression
s_mov_b32 s57, s31 // (op 1904) reg expression
s_mov_b32 s58, s0 // (op 1904) reg expression
s_mov_b32 s59, 131072 // (op 1904) call()

v_and_b32 v13, 63, v1 // (op 1966) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1966) 
v_lshrrev_b32 v14, 4, v13 // (op 1966) 

s_lshr_b32 s6, s0, 1 // (op 1966) 

v_lshrrev_b32 v13, 6, v1 // (op 1966) 
v_lshl_add_u32 v16, s6, 3, v14 // (op 1966) 


v_and_b32 v14, 1, v13 // (op 1966) 

v_and_b32 v13, 15, v1 // (op 1966) 
v_lshl_add_u32 v17, v14, 4, v13 // (op 1966) 


v_lshl_add_u32 v13, s17, 5, v17 // (op 1966) 

v_mul_lo_u32 v14, v13, s32 // (op 1966) 

v_add_u32 v13, v16, v14 // (op 1966) 


s_mul_i32 s1, s32, s9 // (op 1968) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 1968) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 1968) least significant
s_mul_hi_u32 s7, s32, s8 // (op 1968) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1968) most significant: sum
s_add_u32 s1, s1, s7 // (op 1968) most significant: sum


s_mov_b32 s60, s34 // (op 1968) reg expression
s_mov_b32 s61, s35 // (op 1968) reg expression
s_mov_b32 s62, s0 // (op 1968) reg expression
s_mov_b32 s63, 131072 // (op 1968) call()

v_and_b32 v14, -2, v1 // (op 2059) 
v_sub_u32 v16, v1, v14 // (op 2059) 

v_lshlrev_b32 v14, 6, v16 // (op 2059) 

v_lshrrev_b32 v16, 1, v1 // (op 2059) 
v_lshl_add_u32 v17, v16, 7, v14 // (op 2059) 


v_mul_lo_u32 v14, v17, 6 // (op 2059) 

v_lshrrev_b32 v16, 3, v14 // (op 2059) 

v_and_b32 v14, -8, v1 // (op 2117) 
v_sub_u32 v17, v1, v14 // (op 2117) 

v_lshlrev_b32 v14, 4, v17 // (op 2117) 

v_lshrrev_b32 v17, 3, v1 // (op 2117) 
v_lshl_add_u32 v18, v17, 7, v14 // (op 2117) 


s_nop 0
v_readfirstlane_b32 s0, v18 // (op 2117) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 2169) 
v_sub_u32 v17, v1, v14 // (op 2169) 

v_lshrrev_b32 v14, 2, v1 // (op 2169) 
v_lshl_add_u32 v18, v14, 3, v17 // (op 2169) 


s_ashr_i64 s[6:7], s[36:37], 7 // (op 640) 
s_and_b32 s1, s6, 4294967294 // (op 640) 

s_ashr_i64 s[6:7], s[36:37], 7 // (op 639) 
s_mov_b32 s10, s6 // (op 639) convert

s_cmp_lt_u32 s1, s10 // (op 639) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopBottom_KLoopTail_639 // (op 639) Condition: Top (jump to RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopBottom_KLoopTail_639 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopTop_KLoopTail_639:


s_add_u32 s1, s1, 1 // (op 641) 
v_mov_b32 v14, v12 // (op 166) 

buffer_load_dwordx4 v[28:31], v14, s[48:51], 0 offen // (op 166) Load value
buffer_load_dwordx2 v[32:33], v14, s[48:51], 0 offen offset:16 // (op 166) Load value
buffer_load_dwordx4 v[34:37], v14, s[48:51], 0 offen offset:24 // (op 166) Load value
buffer_load_dwordx2 v[38:39], v14, s[48:51], 0 offen offset:40 // (op 166) Load value

v_add_u32 v12, 96, v12 // (op 1758) 
s_barrier  // (op 2370) 
s_barrier  // (op 2383) 





v_mov_b64 v[24:25], v[26:27] // (op 1738) 

s_mov_b32 m0, s0 // (op 1738) reg expression
buffer_load_dwordx4 v24, s[52:55], 0 offen offset: 0 lds // (op 1738) Load value direct to lds

v_add_co_u32 v26, vcc, 128, v26 // (op 1830) least significant half
v_addc_co_u32 v27, vcc, 0, v27, vcc // (op 1830) most significant half
s_waitcnt vmcnt(4)
ds_write_b128 v16, v[28:31] offset:8192 // (op 167) Store local data 

v_mov_b64 v[24:25], s[28:29] // (op 180) convert

v_mov_b32 v14, v15 // (op 180) 
v_mov_b32 v17, v14 // (op 180) 
buffer_load_ubyte v22, v17, s[56:59], 0 offen offset: 0 // (op 180) Load value
v_add_u32 v17, v17, v24 // (op 180) 
s_waitcnt vmcnt(4)
ds_write_b64 v16, v[32:33] offset:8208 // (op 167) Store local data 
s_waitcnt vmcnt(3)
ds_write_b128 v16, v[34:37] offset:8216 // (op 167) Store local data 
s_waitcnt vmcnt(2)
ds_write_b64 v16, v[38:39] offset:8232 // (op 167) Store local data 
buffer_load_ubyte v23, v17, s[56:59], 0 offen offset: 0 // (op 180) Load value




v_add_u32 v15, 4, v15 // (op 1894) 
s_barrier  // (op 2396) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 2373) 






ds_read_b128 v[28:31], v5 offset:8192 // (op 169) Load local data 
ds_read_b64 v[32:33], v5 offset:8208 // (op 169) Load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v18, v22 offset:20480 // (op 178) Store local data 
s_barrier  // (op 2386) 
s_waitcnt vmcnt(0)
ds_write_b8 v18, v23 offset:20484 // (op 178) Store local data 




ds_read_b128 v[34:37], v3 // (op 171) Load local data 
ds_read_b128 v[38:41], v3 offset:64 // (op 171) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 2400) 



ds_read_u8 v14, v0 offset:20480 // (op 173) Load local data 


v_mov_b32 v17, v13 // (op 175) 

buffer_load_ubyte v19, v17, s[60:63], 0 offen offset: 0 // (op 175) Load value

v_add_u32 v13, 4, v13 // (op 1958) 



ds_read_b128 v[20:23], v5 offset:11264 // (op 321) Load local data 
ds_read_b64 v[24:25], v5 offset:11280 // (op 321) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[28:33], v[34:41], a[0:3], v14, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 176) 





ds_read_u8 v14, v0 offset:20608 // (op 325) Load local data 



ds_read_b128 v[28:31], v5 offset:14336 // (op 483) Load local data 
ds_read_b64 v[32:33], v5 offset:14352 // (op 483) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[20:25], v[34:41], a[4:7], v14, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 327) 





ds_read_u8 v14, v0 offset:20736 // (op 487) Load local data 



ds_read_b128 v[20:23], v5 offset:17408 // (op 645) Load local data 
ds_read_b64 v[24:25], v5 offset:17424 // (op 645) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[28:33], v[34:41], a[8:11], v14, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 489) 





ds_read_u8 v14, v0 offset:20864 // (op 649) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[20:25], v[34:41], a[12:15], v14, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 651) 




s_ashr_i64 s[6:7], s[36:37], 7 // (op 639) 
s_mov_b32 s10, s6 // (op 639) convert

s_cmp_lt_u32 s1, s10 // (op 639) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopTop_KLoopTail_639 // (op 639) Condition: Bottom (jump to RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopTop_KLoopTail_639 if true)
RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopBottom_KLoopTail_639:






















s_barrier  // (op 2480) 







v_cmp_eq_f32 vcc, 0.00000, v10 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ConditionalFalse_0_beta_123
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
s_branch RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ConditionalBottom_0_beta_123
RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ConditionalFalse_0_beta_123:



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
RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ConditionalBottom_0_beta_123:












s_endpgm  // End of RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2
.LRR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_end:
.size RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2, .LRR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_end-RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2
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
    DataType: B6
    DataTypeA: B6
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
  - .name: RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2
    .symbol: RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2.kd
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

