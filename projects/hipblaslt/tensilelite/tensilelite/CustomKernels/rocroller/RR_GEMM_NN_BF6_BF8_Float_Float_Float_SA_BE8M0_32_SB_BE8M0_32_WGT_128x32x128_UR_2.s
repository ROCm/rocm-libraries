// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2
.p2align 8
.type RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2,@function
RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_exec_begin:


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


v_and_b32 v0, 63, v1 // (op 2092) 
v_and_b32 v5, 48, v1 // (op 2092) 
v_sub_u32 v7, v0, v5 // (op 2092) 

v_lshrrev_b32 v5, 4, v0 // (op 2092) 

v_lshl_add_u32 v0, v5, 5, v7 // (op 2092) 


v_lshrrev_b32 v5, 7, v1 // (op 2092) 
v_lshlrev_b32 v7, 4, v5 // (op 2092) 

v_lshl_add_u32 v5, v0, 7, v7 // (op 2092) 


v_mul_lo_u32 v0, v5, 6 // (op 2092) 
v_lshrrev_b32 v7, 3, v0 // (op 2092) 

v_lshrrev_b32 v0, 4, v5 // (op 2092) 

v_lshlrev_b32 v5, 2, v0 // (op 2092) 

v_add_u32 v0, v7, v5 // (op 2092) 


v_lshrrev_b32 v5, 7, v1 // (op 2202) 
v_and_b32 v7, 15, v1 // (op 2202) 
s_mul_i32 s1, s2, s9 // (op 2032) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 2032) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 2032) least significant
s_mul_hi_u32 s19, s2, s8 // (op 2032) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2032) most significant: sum
s_add_u32 s1, s1, s19 // (op 2032) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 2032) 

v_lshl_add_u32 v8, v5, 4, v7 // (op 2202) 


v_and_b32 v5, 63, v1 // (op 2202) 
v_lshrrev_b32 v7, 4, v5 // (op 2202) 

v_lshl_add_u32 v5, v7, 7, v8 // (op 2202) 


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


v_and_b32 v7, 63, v1 // (op 2249) 
v_and_b32 v8, 48, v1 // (op 2249) 
v_sub_u32 v10, v7, v8 // (op 2249) 

v_lshrrev_b32 v8, 4, v7 // (op 2249) 

v_lshl_add_u32 v7, v8, 5, v10 // (op 2249) 


v_lshrrev_b32 v8, 7, v1 // (op 2249) 
v_lshlrev_b32 v10, 4, v8 // (op 2249) 

v_lshl_add_u32 v8, v7, 7, v10 // (op 2249) 


v_mul_lo_u32 v7, v8, 6 // (op 2249) 
v_lshrrev_b32 v10, 3, v7 // (op 2249) 

v_lshrrev_b32 v7, 4, v8 // (op 2249) 

v_lshlrev_b32 v8, 2, v7 // (op 2249) 

v_add_u32 v7, v10, v8 // (op 2249) 


v_lshrrev_b32 v8, 7, v1 // (op 2317) 
v_and_b32 v10, 15, v1 // (op 2317) 
s_mul_i32 s1, s4, s9 // (op 2220) most significant: low of low * high
v_lshl_add_u32 v11, v8, 4, v10 // (op 2317) 


v_and_b32 v8, 63, v1 // (op 2317) 
v_lshrrev_b32 v10, 4, v8 // (op 2317) 

v_lshl_add_u32 v8, v10, 7, v11 // (op 2317) 


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
v_and_b32 v12, -2, v1 // (op 1797) 
v_sub_u32 v13, v1, v12 // (op 1797) 

v_lshlrev_b32 v12, 6, v13 // (op 1797) 

v_lshl_add_u32 v13, s16, 7, v12 // (op 1797) 

v_mov_b32 v15, 0 // (op 1797) convert
v_mov_b32 v14, v13 // (op 1797) convert

v_lshrrev_b32 v12, 1, v1 // (op 1797) 
v_mul_lo_u32 v17, v12, s13 // (op 1797) most significant: low of low * high
v_mul_hi_u32 v13, v12, s12 // (op 1797) most significant: high of low * low
v_mul_lo_u32 v16, v12, s12 // (op 1797) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 1797) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v16 // (op 1797) least significant half
v_addc_co_u32 v13, s[0:1], v15, v17, s[0:1] // (op 1797) most significant half



v_mul_lo_u32 v15, v13, 6 // (op 1797) most significant: low of high * low
v_mul_hi_u32 v16, v12, 6 // (op 1797) most significant: high of low * low
v_mul_lo_u32 v14, v12, 6 // (op 1797) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 1797) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 1797) 

v_mov_b32 v14, v12 // (op 1797) convert


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

v_and_b32 v12, -2, v1 // (op 1816) 
v_sub_u32 v13, v1, v12 // (op 1816) 

v_lshlrev_b32 v12, 6, v13 // (op 1816) 

v_lshl_add_u32 v13, s16, 7, v12 // (op 1816) 

v_mov_b32 v17, 0 // (op 1816) convert
v_mov_b32 v16, v13 // (op 1816) convert

v_lshrrev_b32 v15, 1, v1 // (op 1816) 
v_mul_lo_u32 v13, v15, s13 // (op 1816) most significant: low of low * high
v_mul_hi_u32 v18, v15, s12 // (op 1816) most significant: high of low * low
v_mul_lo_u32 v12, v15, s12 // (op 1816) least significant: low of low * low
v_add_u32 v13, v13, v18 // (op 1816) most significant: sum


v_add_co_u32 v18, s[0:1], v16, v12 // (op 1816) least significant half
v_addc_co_u32 v19, s[0:1], v17, v13, s[0:1] // (op 1816) most significant half



v_mul_lo_u32 v13, v19, 6 // (op 1816) most significant: low of high * low
v_mul_hi_u32 v15, v18, 6 // (op 1816) most significant: high of low * low
v_mul_lo_u32 v12, v18, 6 // (op 1816) least significant: low of low * low
v_add_u32 v13, v13, v15 // (op 1816) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[12:13] // (op 1816) 

v_mov_b32 v15, v16 // (op 1816) convert

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

v_and_b32 v12, -8, v1 // (op 1861) 
v_sub_u32 v13, v1, v12 // (op 1861) 

v_lshlrev_b32 v12, 4, v13 // (op 1861) 

v_mov_b32 v17, 0 // (op 1861) convert
v_mov_b32 v16, v12 // (op 1861) convert

v_lshrrev_b32 v12, 3, v1 // (op 1861) 
v_lshl_add_u32 v13, s17, 5, v12 // (op 1861) 

v_mul_lo_u32 v19, v13, s25 // (op 1861) most significant: low of low * high
v_mul_hi_u32 v12, v13, s24 // (op 1861) most significant: high of low * low
v_mul_lo_u32 v18, v13, s24 // (op 1861) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 1861) most significant: sum


v_add_co_u32 v12, s[0:1], v16, v18 // (op 1861) least significant half
v_addc_co_u32 v13, s[0:1], v17, v19, s[0:1] // (op 1861) most significant half



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

v_and_b32 v18, -128, v1 // (op 1925) 
v_sub_u32 v19, v1, v18 // (op 1925) 

v_lshl_add_u32 v18, s16, 7, v19 // (op 1925) 

v_lshrrev_b32 v19, 7, v1 // (op 1925) 
v_lshlrev_b32 v20, 1, v19 // (op 1925) 

v_mul_lo_u32 v19, v20, s28 // (op 1925) 

v_add_u32 v20, v18, v19 // (op 1925) 


s_mul_i32 s1, s28, s33 // (op 1927) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 1927) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1927) least significant
s_mul_hi_u32 s7, s28, s32 // (op 1927) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1927) most significant: sum
s_add_u32 s1, s1, s7 // (op 1927) most significant: sum


s_mov_b32 s56, s30 // (op 1927) reg expression
s_mov_b32 s57, s31 // (op 1927) reg expression
s_mov_b32 s58, s0 // (op 1927) reg expression
s_mov_b32 s59, 131072 // (op 1927) call()

v_and_b32 v18, -128, v1 // (op 1944) 
v_sub_u32 v19, v1, v18 // (op 1944) 

v_lshl_add_u32 v18, s16, 7, v19 // (op 1944) 

v_lshrrev_b32 v19, 7, v1 // (op 1944) 
v_lshlrev_b32 v21, 1, v19 // (op 1944) 

v_mul_lo_u32 v19, v21, s28 // (op 1944) 

v_add_u32 v21, v18, v19 // (op 1944) 


s_mul_i32 s1, s28, s33 // (op 1946) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 1946) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1946) least significant
s_mul_hi_u32 s7, s28, s32 // (op 1946) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1946) most significant: sum
s_add_u32 s1, s1, s7 // (op 1946) most significant: sum


s_mov_b32 s56, s30 // (op 1946) reg expression
s_mov_b32 s57, s31 // (op 1946) reg expression
s_mov_b32 s58, s0 // (op 1946) reg expression
s_mov_b32 s59, 131072 // (op 1946) call()

v_lshrrev_b32 v18, 6, v1 // (op 1989) 
v_and_b32 v19, 1, v18 // (op 1989) 

v_and_b32 v18, 15, v1 // (op 1989) 
v_lshl_add_u32 v22, v19, 4, v18 // (op 1989) 


v_lshl_add_u32 v18, s17, 5, v22 // (op 1989) 

v_mul_lo_u32 v19, v18, s34 // (op 1989) 

v_and_b32 v18, 63, v1 // (op 1989) 
v_lshrrev_b32 v22, 4, v18 // (op 1989) 

v_add_u32 v18, v22, v19 // (op 1989) 


s_mul_i32 s1, s34, s9 // (op 1991) most significant: low of low * high
s_mul_i32 s6, s35, s8 // (op 1991) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 1991) least significant
s_mul_hi_u32 s7, s34, s8 // (op 1991) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1991) most significant: sum
s_add_u32 s1, s1, s7 // (op 1991) most significant: sum


s_mov_b32 s60, s36 // (op 1991) reg expression
s_mov_b32 s61, s37 // (op 1991) reg expression
s_mov_b32 s62, s0 // (op 1991) reg expression
s_mov_b32 s63, 131072 // (op 1991) call()

v_lshrrev_b32 v19, 6, v1 // (op 2008) 
v_and_b32 v22, 1, v19 // (op 2008) 

v_and_b32 v19, 15, v1 // (op 2008) 
v_lshl_add_u32 v23, v22, 4, v19 // (op 2008) 


v_lshl_add_u32 v19, s17, 5, v23 // (op 2008) 

v_mul_lo_u32 v22, v19, s34 // (op 2008) 

v_and_b32 v19, 63, v1 // (op 2008) 
v_lshrrev_b32 v23, 4, v19 // (op 2008) 

v_add_u32 v19, v23, v22 // (op 2008) 


s_mul_i32 s1, s34, s9 // (op 2010) most significant: low of low * high
s_mul_i32 s6, s35, s8 // (op 2010) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 2010) least significant
s_mul_hi_u32 s7, s34, s8 // (op 2010) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2010) most significant: sum
s_add_u32 s1, s1, s7 // (op 2010) most significant: sum


s_mov_b32 s60, s36 // (op 2010) reg expression
s_mov_b32 s61, s37 // (op 2010) reg expression
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
buffer_load_dwordx4 v12, s[52:55], 0 offen offset: 0 lds // (op 1750) Load value direct to lds



v_mov_b32 v23, v14 // (op 1605) reg expression

buffer_load_dwordx3 v[24:26], v23, s[48:51], 0 offen // (op 1605) Load value
buffer_load_dwordx3 v[28:30], v23, s[48:51], 0 offen offset:12 // (op 1605) Load value
buffer_load_dwordx3 v[32:34], v23, s[48:51], 0 offen offset:24 // (op 1605) Load value
buffer_load_dwordx3 v[36:38], v23, s[48:51], 0 offen offset:36 // (op 1605) Load value


v_mov_b64 v[40:41], s[28:29] // (op 1608) convert

v_mov_b32 v23, v20 // (op 1608) reg expression
buffer_load_ubyte v42, v23, s[56:59], 0 offen offset: 0 // (op 1608) Load value
v_add_u32 v23, v23, v40 // (op 1608) 
buffer_load_ubyte v43, v23, s[56:59], 0 offen offset: 0 // (op 1608) Load value





buffer_load_ubyte v23, v18, s[60:63], 0 offen offset:4 // (op 1612) Load value

v_and_b32 v18, -8, v1 // (op 2276) 
v_sub_u32 v27, v1, v18 // (op 2276) 

v_lshlrev_b32 v18, 4, v27 // (op 2276) 

v_lshrrev_b32 v27, 3, v1 // (op 2276) 
v_lshl_add_u32 v31, v27, 7, v18 // (op 2276) 


s_nop 0
v_readfirstlane_b32 s0, v31 // (op 2276) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s1, 4096, s0 // (op 1755) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 1755) reg expression
s_mov_b32 s6, 128 // (op 1755) call()
buffer_load_dwordx4 v12, s[52:55], s6 offen offset: 0 lds // (op 1755) Load value direct to lds






s_lshl_b64 s[0:1], s[12:13], 7 // (op 1619) 
s_mul_i32 s7, s1, 6 // (op 1619) most significant: low of high * low
s_mul_i32 s6, s0, 6 // (op 1619) least significant
s_mul_hi_u32 s10, s0, 6 // (op 1619) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 1619) most significant: sum


s_lshr_b64 s[0:1], s[6:7], 3 // (op 1619) 

v_add_u32 v18, v14, s0 // (op 1619) 


buffer_load_dwordx3 v[44:46], v18, s[48:51], 0 offen // (op 1619) Load value
buffer_load_dwordx3 v[48:50], v18, s[48:51], 0 offen offset:12 // (op 1619) Load value
buffer_load_dwordx3 v[52:54], v18, s[48:51], 0 offen offset:24 // (op 1619) Load value
buffer_load_dwordx3 v[56:58], v18, s[48:51], 0 offen offset:36 // (op 1619) Load value



v_mov_b64 v[40:41], s[28:29] // (op 1622) convert

s_lshl_b32 s0, s28, 2 // (op 1622) 
v_add_u32 v18, v20, s0 // (op 1622) 

buffer_load_ubyte v12, v18, s[56:59], 0 offen offset: 0 // (op 1622) Load value
v_add_u32 v18, v18, v40 // (op 1622) 
buffer_load_ubyte v13, v18, s[56:59], 0 offen offset: 0 // (op 1622) Load value



v_and_b32 v14, -2, v1 // (op 2075) 
v_sub_u32 v18, v1, v14 // (op 2075) 

v_lshlrev_b32 v14, 6, v18 // (op 2075) 

v_lshrrev_b32 v18, 1, v1 // (op 2075) 
v_lshl_add_u32 v20, v18, 7, v14 // (op 2075) 


v_mul_lo_u32 v14, v20, 6 // (op 2075) 
v_lshrrev_b32 v18, 3, v14 // (op 2075) 

v_lshrrev_b32 v14, 4, v20 // (op 2075) 

v_lshlrev_b32 v20, 2, v14 // (op 2075) 

v_add_u32 v14, v18, v20 // (op 2075) 





s_waitcnt vmcnt(13)
ds_write_b96 v14, v[24:26] offset:8192 // (op 1625) Store local data 
s_waitcnt vmcnt(12)
ds_write_b96 v14, v[28:30] offset:8208 // (op 1625) Store local data 
s_waitcnt vmcnt(11)
ds_write_b96 v14, v[32:34] offset:8224 // (op 1625) Store local data 
s_waitcnt vmcnt(10)
ds_write_b96 v14, v[36:38] offset:8240 // (op 1625) Store local data 

v_and_b32 v14, -128, v1 // (op 2185) 
v_sub_u32 v18, v1, v14 // (op 2185) 

v_lshrrev_b32 v14, 7, v1 // (op 2185) 
v_lshl_add_u32 v20, v14, 8, v18 // (op 2185) 





s_waitcnt vmcnt(9)
ds_write_b8 v20, v42 offset:24576 // (op 1626) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v20, v43 offset:24704 // (op 1626) Store local data 

v_and_b32 v14, -2, v1 // (op 2046) 
v_sub_u32 v18, v1, v14 // (op 2046) 

v_lshlrev_b32 v14, 6, v18 // (op 2046) 

v_lshrrev_b32 v18, 1, v1 // (op 2046) 
v_lshl_add_u32 v20, v18, 7, v14 // (op 2046) 


v_mul_lo_u32 v14, v20, 6 // (op 2046) 
v_lshrrev_b32 v18, 3, v14 // (op 2046) 

v_lshrrev_b32 v14, 4, v20 // (op 2046) 

v_lshlrev_b32 v20, 2, v14 // (op 2046) 

v_add_u32 v14, v18, v20 // (op 2046) 


v_and_b32 v18, -8, v1 // (op 2104) 
v_sub_u32 v20, v1, v18 // (op 2104) 

v_lshlrev_b32 v18, 4, v20 // (op 2104) 

v_lshrrev_b32 v20, 3, v1 // (op 2104) 
v_lshl_add_u32 v27, v20, 7, v18 // (op 2104) 


s_nop 0
v_readfirstlane_b32 s0, v27 // (op 2104) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v18, -128, v1 // (op 2156) 
v_sub_u32 v20, v1, v18 // (op 2156) 

v_lshrrev_b32 v18, 7, v1 // (op 2156) 
v_lshl_add_u32 v27, v18, 8, v20 // (op 2156) 


v_and_b32 v18, -2, v1 // (op 2232) 
v_sub_u32 v20, v1, v18 // (op 2232) 

v_lshlrev_b32 v18, 6, v20 // (op 2232) 

v_lshrrev_b32 v20, 1, v1 // (op 2232) 
v_lshl_add_u32 v31, v20, 7, v18 // (op 2232) 


v_mul_lo_u32 v18, v31, 6 // (op 2232) 
v_lshrrev_b32 v20, 3, v18 // (op 2232) 

v_lshrrev_b32 v18, 4, v31 // (op 2232) 

v_lshlrev_b32 v31, 2, v18 // (op 2232) 

v_add_u32 v18, v20, v31 // (op 2232) 


v_and_b32 v20, -8, v1 // (op 2261) 
v_sub_u32 v31, v1, v20 // (op 2261) 

v_lshlrev_b32 v20, 4, v31 // (op 2261) 

v_lshrrev_b32 v31, 3, v1 // (op 2261) 
v_lshl_add_u32 v35, v31, 7, v20 // (op 2261) 


s_nop 0
v_readfirstlane_b32 s1, v35 // (op 2261) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v20, -128, v1 // (op 2300) 
v_sub_u32 v31, v1, v20 // (op 2300) 

v_lshrrev_b32 v20, 7, v1 // (op 2300) 
v_lshl_add_u32 v35, v20, 8, v31 // (op 2300) 


s_mov_b32 s6, 0 // (op 635) call()
s_ashr_i64 s[10:11], s[22:23], 7 // (op 634) 
s_mov_b32 s7, s10 // (op 634) convert

s_and_b32 s10, s7, 4294967294 // (op 634) 

s_cmp_lt_u32 s6, s10 // (op 634) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopBottom_KLoop_634 // (op 634) Condition: Top (jump to RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopBottom_KLoop_634 if false)
RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopTop_KLoop_634:
s_add_u32 s6, s6, 2 // (op 636) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 2364) 



ds_read_b96_tr_b6 v[60:62], v0 offset:8192 // (op 65) Transpose load local data 
ds_read_b96_tr_b6 v[63:65], v0 offset:10240 // (op 65) Transpose load local data 
s_barrier  // (op 2377) 



ds_read_b128 v[66:69], v3 // (op 70) Load local data 
ds_read_b128 v[70:73], v3 offset:64 // (op 70) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 2390) 


ds_read_u8 v20, v5 offset:24576 // (op 75) Load local data 



ds_read_b96_tr_b6 v[74:76], v0 offset:8224 // (op 340) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[60:65], v[66:73], a[0:3], v20, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 82) 
ds_read_b96_tr_b6 v[77:79], v0 offset:10272 // (op 340) Transpose load local data 




ds_read_u8 v20, v5 offset:24608 // (op 344) Load local data 



ds_read_b96_tr_b6 v[60:62], v0 offset:8256 // (op 502) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[74:79], v[66:73], a[4:7], v20, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 346) 
ds_read_b96_tr_b6 v[63:65], v0 offset:10304 // (op 502) Transpose load local data 




ds_read_u8 v20, v5 offset:24640 // (op 506) Load local data 



ds_read_b96_tr_b6 v[74:76], v0 offset:8288 // (op 664) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[60:65], v[66:73], a[8:11], v20, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 508) 
ds_read_b96_tr_b6 v[77:79], v0 offset:10336 // (op 664) Transpose load local data 




ds_read_u8 v20, v5 offset:24672 // (op 668) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[74:79], v[66:73], a[12:15], v20, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 670) 





v_mov_b32 v20, v19 // (op 80) 

buffer_load_ubyte v22, v20, s[60:63], 0 offen offset:8 // (op 80) Load value
s_barrier  // (op 2380) 


v_mov_b64 v[40:41], v[16:17] // (op 1733) 

s_mov_b32 m0, s0 // (op 1733) reg expression
s_mov_b32 s7, 256 // (op 1733) call()
buffer_load_dwordx4 v40, s[52:55], s7 offen offset: 0 lds // (op 1733) Load value direct to lds



v_mov_b32 v31, v15 // (op 4) 
s_lshl_b64 s[10:11], s[12:13], 7 // (op 4) 
s_mul_i32 s15, s11, 6 // (op 4) most significant: low of high * low
s_mul_i32 s14, s10, 6 // (op 4) least significant
s_mul_hi_u32 s7, s10, 6 // (op 4) most significant: high of low * low
s_add_u32 s15, s15, s7 // (op 4) most significant: sum


s_lshr_b64 s[10:11], s[14:15], 3 // (op 4) 

s_lshl_b32 s7, s10, 1 // (op 4) 

v_add_u32 v39, v31, s7 // (op 4) 


buffer_load_dwordx3 v[24:26], v39, s[48:51], 0 offen // (op 4) Load value
buffer_load_dwordx3 v[28:30], v39, s[48:51], 0 offen offset:12 // (op 4) Load value
buffer_load_dwordx3 v[32:34], v39, s[48:51], 0 offen offset:24 // (op 4) Load value
buffer_load_dwordx3 v[36:38], v39, s[48:51], 0 offen offset:36 // (op 4) Load value


v_mov_b64 v[60:61], s[28:29] // (op 17) convert

v_mov_b32 v39, v21 // (op 17) 
s_lshl_b32 s7, s28, 3 // (op 17) 
v_add_u32 v47, v39, s7 // (op 17) 

buffer_load_ubyte v42, v47, s[56:59], 0 offen offset: 0 // (op 17) Load value
v_add_u32 v47, v47, v60 // (op 17) 
buffer_load_ubyte v43, v47, s[56:59], 0 offen offset: 0 // (op 17) Load value


s_barrier  // (op 2407) 



s_waitcnt vmcnt(13)
ds_write_b96 v18, v[44:46] offset:25088 // (op 225) Store local data 
s_waitcnt vmcnt(12)
ds_write_b96 v18, v[48:50] offset:25104 // (op 225) Store local data 
s_waitcnt vmcnt(11)
ds_write_b96 v18, v[52:54] offset:25120 // (op 225) Store local data 
s_waitcnt vmcnt(10)
ds_write_b96 v18, v[56:58] offset:25136 // (op 225) Store local data 
s_barrier  // (op 2419) 



s_waitcnt vmcnt(9)
ds_write_b8 v35, v12 offset:41472 // (op 236) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v35, v13 offset:41600 // (op 236) Store local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 2404) 



ds_read_b96_tr_b6 v[60:62], v7 offset:25088 // (op 227) Transpose load local data 
ds_read_b96_tr_b6 v[63:65], v7 offset:27136 // (op 227) Transpose load local data 
s_barrier  // (op 2410) 



ds_read_b128 v[66:69], v4 offset:4096 // (op 229) Load local data 
ds_read_b128 v[70:73], v4 offset:4160 // (op 229) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 2416) 


ds_read_u8 v47, v8 offset:41472 // (op 231) Load local data 



ds_read_b96_tr_b6 v[74:76], v7 offset:25120 // (op 353) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[60:65], v[66:73], a[0:3], v47, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 234) 
ds_read_b96_tr_b6 v[77:79], v7 offset:27168 // (op 353) Transpose load local data 




ds_read_u8 v47, v8 offset:41504 // (op 357) Load local data 



ds_read_b96_tr_b6 v[60:62], v7 offset:25152 // (op 515) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[74:79], v[66:73], a[4:7], v47, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 347) 
ds_read_b96_tr_b6 v[63:65], v7 offset:27200 // (op 515) Transpose load local data 




ds_read_u8 v47, v8 offset:41536 // (op 519) Load local data 



ds_read_b96_tr_b6 v[74:76], v7 offset:25184 // (op 677) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[60:65], v[66:73], a[8:11], v47, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 509) 
ds_read_b96_tr_b6 v[77:79], v7 offset:27232 // (op 677) Transpose load local data 




s_barrier  // (op 2492) 
ds_read_u8 v47, v8 offset:41568 // (op 681) Load local data 
s_barrier  // (op 2496) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[74:79], v[66:73], a[12:15], v47, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 671) 






buffer_load_ubyte v23, v20, s[60:63], 0 offen offset:12 // (op 233) Load value

s_barrier  // (op 2413) 
v_add_u32 v19, 8, v19 // (op 2000) 


s_add_u32 s7, 4096, s1 // (op 1745) 

s_mov_b32 m0, s7 // (op 1745) reg expression
s_mov_b32 s10, 384 // (op 1745) call()
buffer_load_dwordx4 v40, s[52:55], s10 offen offset: 0 lds // (op 1745) Load value direct to lds





v_add_co_u32 v16, vcc, 256, v16 // (op 1872) least significant half
v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 1872) most significant half
s_lshl_b64 s[10:11], s[12:13], 7 // (op 224) 
s_mul_i32 s15, s11, 6 // (op 224) most significant: low of high * low
s_mul_i32 s14, s10, 6 // (op 224) least significant
s_mul_hi_u32 s7, s10, 6 // (op 224) most significant: high of low * low
s_add_u32 s15, s15, s7 // (op 224) most significant: sum


s_lshr_b64 s[10:11], s[14:15], 3 // (op 224) 

s_mul_i32 s7, 3, s10 // (op 224) 

v_add_u32 v20, v31, s7 // (op 224) 


buffer_load_dwordx3 v[44:46], v20, s[48:51], 0 offen // (op 224) Load value
buffer_load_dwordx3 v[48:50], v20, s[48:51], 0 offen offset:12 // (op 224) Load value
buffer_load_dwordx3 v[52:54], v20, s[48:51], 0 offen offset:24 // (op 224) Load value
buffer_load_dwordx3 v[56:58], v20, s[48:51], 0 offen offset:36 // (op 224) Load value



v_mov_b64 v[40:41], s[28:29] // (op 238) convert

s_lshl_b32 s7, s28, 2 // (op 238) 
s_mul_i32 s10, 3, s7 // (op 238) 

v_add_u32 v20, v39, s10 // (op 238) 

s_lshl_b64 s[10:11], s[12:13], 7 // (op 1808) 
s_mul_i32 s15, s11, 6 // (op 1808) most significant: low of high * low
s_mul_i32 s14, s10, 6 // (op 1808) least significant
s_mul_hi_u32 s7, s10, 6 // (op 1808) most significant: high of low * low
s_add_u32 s15, s15, s7 // (op 1808) most significant: sum


s_lshr_b64 s[10:11], s[14:15], 3 // (op 1808) 

s_lshl_b32 s7, s10, 1 // (op 1808) 

v_add_u32 v15, v15, s7 // (op 1808) 

buffer_load_ubyte v12, v20, s[56:59], 0 offen offset: 0 // (op 238) Load value
v_add_u32 v20, v20, v40 // (op 238) 
buffer_load_ubyte v13, v20, s[56:59], 0 offen offset: 0 // (op 238) Load value



s_barrier  // (op 2367) 
s_lshl_b32 s7, s28, 3 // (op 1936) 



v_add_u32 v21, v21, s7 // (op 1936) 

s_waitcnt vmcnt(13)
ds_write_b96 v14, v[24:26] offset:8192 // (op 6) Store local data 
s_waitcnt vmcnt(12)
ds_write_b96 v14, v[28:30] offset:8208 // (op 6) Store local data 
s_waitcnt vmcnt(11)
ds_write_b96 v14, v[32:34] offset:8224 // (op 6) Store local data 
s_waitcnt vmcnt(10)
ds_write_b96 v14, v[36:38] offset:8240 // (op 6) Store local data 
s_barrier  // (op 2393) 



s_waitcnt vmcnt(9)
ds_write_b8 v27, v42 offset:24576 // (op 21) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v27, v43 offset:24704 // (op 21) Store local data 
s_ashr_i64 s[10:11], s[22:23], 7 // (op 634) 
s_mov_b32 s7, s10 // (op 634) convert

s_and_b32 s10, s7, 4294967294 // (op 634) 

s_cmp_lt_u32 s6, s10 // (op 634) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopTop_KLoop_634 // (op 634) Condition: Bottom (jump to RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopTop_KLoop_634 if true)
RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopBottom_KLoop_634:














s_barrier  // (op 2476) 
v_and_b32 v47, -2, v1 // (op 1768) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 1768) 
v_sub_u32 v51, v1, v47 // (op 1768) 

s_lshr_b32 s6, s0, 1 // (op 1768) 

v_lshlrev_b32 v47, 6, v51 // (op 1768) 

v_lshl_add_u32 v51, s16, 7, v47 // (op 1768) 

v_mov_b32 v15, 0 // (op 1768) convert
v_mov_b32 v14, v51 // (op 1768) convert

v_lshrrev_b32 v47, 1, v1 // (op 1768) 
v_lshl_add_u32 v51, s6, 8, v47 // (op 1768) 


v_mul_lo_u32 v17, v51, s13 // (op 1768) most significant: low of low * high
v_mul_hi_u32 v47, v51, s12 // (op 1768) most significant: high of low * low
v_mul_lo_u32 v16, v51, s12 // (op 1768) least significant: low of low * low
v_add_u32 v17, v17, v47 // (op 1768) most significant: sum


v_add_co_u32 v18, s[0:1], v14, v16 // (op 1768) least significant half
v_addc_co_u32 v19, s[0:1], v15, v17, s[0:1] // (op 1768) most significant half



v_mul_lo_u32 v15, v19, 6 // (op 1768) most significant: low of high * low
v_mul_hi_u32 v47, v18, 6 // (op 1768) most significant: high of low * low
v_mul_lo_u32 v14, v18, 6 // (op 1768) least significant: low of low * low
v_add_u32 v15, v15, v47 // (op 1768) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 1768) 

v_mov_b32 v47, v16 // (op 1768) convert

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

v_and_b32 v51, -8, v1 // (op 1838) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 1838) 
v_sub_u32 v55, v1, v51 // (op 1838) 

s_lshr_b32 s6, s0, 1 // (op 1838) 

v_lshlrev_b32 v51, 4, v55 // (op 1838) 

v_lshl_add_u32 v55, s6, 8, v51 // (op 1838) 


v_mov_b32 v15, 0 // (op 1838) convert
v_mov_b32 v14, v55 // (op 1838) convert

v_lshrrev_b32 v51, 3, v1 // (op 1838) 
v_lshl_add_u32 v55, s17, 5, v51 // (op 1838) 

v_mul_lo_u32 v17, v55, s25 // (op 1838) most significant: low of low * high
v_mul_hi_u32 v51, v55, s24 // (op 1838) most significant: high of low * low
v_mul_lo_u32 v16, v55, s24 // (op 1838) least significant: low of low * low
v_add_u32 v17, v17, v51 // (op 1838) most significant: sum


v_add_co_u32 v18, s[0:1], v14, v16 // (op 1838) least significant half
v_addc_co_u32 v19, s[0:1], v15, v17, s[0:1] // (op 1838) most significant half



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

v_and_b32 v51, -128, v1 // (op 1902) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 1902) 
v_sub_u32 v55, v1, v51 // (op 1902) 

s_lshr_b32 s6, s0, 1 // (op 1902) 

v_lshl_add_u32 v51, s16, 7, v55 // (op 1902) 

v_lshrrev_b32 v55, 7, v1 // (op 1902) 
v_lshlrev_b32 v14, 1, v55 // (op 1902) 

v_lshl_add_u32 v55, s6, 3, v14 // (op 1902) 


v_mul_lo_u32 v14, v55, s28 // (op 1902) 

v_add_u32 v55, v51, v14 // (op 1902) 


s_mul_i32 s1, s28, s33 // (op 1904) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 1904) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1904) least significant
s_mul_hi_u32 s7, s28, s32 // (op 1904) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1904) most significant: sum
s_add_u32 s1, s1, s7 // (op 1904) most significant: sum


s_mov_b32 s56, s30 // (op 1904) reg expression
s_mov_b32 s57, s31 // (op 1904) reg expression
s_mov_b32 s58, s0 // (op 1904) reg expression
s_mov_b32 s59, 131072 // (op 1904) call()

v_and_b32 v51, 63, v1 // (op 1966) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 1966) 
v_lshrrev_b32 v14, 4, v51 // (op 1966) 

s_lshr_b32 s6, s0, 1 // (op 1966) 

v_lshrrev_b32 v51, 6, v1 // (op 1966) 
v_lshl_add_u32 v15, s6, 3, v14 // (op 1966) 


v_and_b32 v14, 1, v51 // (op 1966) 

v_and_b32 v51, 15, v1 // (op 1966) 
v_lshl_add_u32 v16, v14, 4, v51 // (op 1966) 


v_lshl_add_u32 v14, s17, 5, v16 // (op 1966) 

v_mul_lo_u32 v51, v14, s34 // (op 1966) 

v_add_u32 v14, v15, v51 // (op 1966) 


s_mul_i32 s1, s34, s9 // (op 1968) most significant: low of low * high
s_mul_i32 s6, s35, s8 // (op 1968) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 1968) least significant
s_mul_hi_u32 s7, s34, s8 // (op 1968) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1968) most significant: sum
s_add_u32 s1, s1, s7 // (op 1968) most significant: sum


s_mov_b32 s60, s36 // (op 1968) reg expression
s_mov_b32 s61, s37 // (op 1968) reg expression
s_mov_b32 s62, s0 // (op 1968) reg expression
s_mov_b32 s63, 131072 // (op 1968) call()

v_and_b32 v51, -2, v1 // (op 2059) 
v_sub_u32 v15, v1, v51 // (op 2059) 

v_lshlrev_b32 v51, 6, v15 // (op 2059) 

v_lshrrev_b32 v15, 1, v1 // (op 2059) 
v_lshl_add_u32 v16, v15, 7, v51 // (op 2059) 


v_mul_lo_u32 v15, v16, 6 // (op 2059) 
v_lshrrev_b32 v17, 3, v15 // (op 2059) 

v_lshrrev_b32 v15, 4, v16 // (op 2059) 

v_lshlrev_b32 v16, 2, v15 // (op 2059) 

v_add_u32 v15, v17, v16 // (op 2059) 


v_and_b32 v51, -8, v1 // (op 2117) 
v_sub_u32 v16, v1, v51 // (op 2117) 

v_lshlrev_b32 v17, 4, v16 // (op 2117) 

v_lshrrev_b32 v16, 3, v1 // (op 2117) 
v_lshl_add_u32 v51, v16, 7, v17 // (op 2117) 


s_nop 0
v_readfirstlane_b32 s0, v51 // (op 2117) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v51, -128, v1 // (op 2169) 
v_sub_u32 v16, v1, v51 // (op 2169) 

v_lshrrev_b32 v17, 7, v1 // (op 2169) 
v_lshl_add_u32 v51, v17, 8, v16 // (op 2169) 


s_ashr_i64 s[6:7], s[22:23], 7 // (op 640) 
s_and_b32 s1, s6, 4294967294 // (op 640) 

s_ashr_i64 s[6:7], s[22:23], 7 // (op 639) 
s_mov_b32 s10, s6 // (op 639) convert

s_cmp_lt_u32 s1, s10 // (op 639) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopBottom_KLoopTail_639 // (op 639) Condition: Top (jump to RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopBottom_KLoopTail_639 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopTop_KLoopTail_639:


s_add_u32 s1, s1, 1 // (op 641) 
v_mov_b32 v12, v47 // (op 166) 
v_mov_b32 v13, v12 // (op 166) reg expression

buffer_load_dwordx3 v[20:22], v13, s[48:51], 0 offen // (op 166) Load value
buffer_load_dwordx3 v[24:26], v13, s[48:51], 0 offen offset:12 // (op 166) Load value
buffer_load_dwordx3 v[28:30], v13, s[48:51], 0 offen offset:24 // (op 166) Load value
buffer_load_dwordx3 v[32:34], v13, s[48:51], 0 offen offset:36 // (op 166) Load value


s_lshl_b64 s[6:7], s[12:13], 7 // (op 1758) 
s_mul_i32 s11, s7, 6 // (op 1758) most significant: low of high * low
s_mul_i32 s10, s6, 6 // (op 1758) least significant
s_mul_hi_u32 s14, s6, 6 // (op 1758) most significant: high of low * low
s_add_u32 s11, s11, s14 // (op 1758) most significant: sum


s_lshr_b64 s[6:7], s[10:11], 3 // (op 1758) 

v_add_u32 v47, v47, s6 // (op 1758) 

s_barrier  // (op 2370) 
s_barrier  // (op 2383) 





v_mov_b64 v[12:13], v[18:19] // (op 1738) 

s_mov_b32 m0, s0 // (op 1738) reg expression
buffer_load_dwordx4 v12, s[52:55], 0 offen offset: 0 lds // (op 1738) Load value direct to lds

v_add_co_u32 v18, vcc, 128, v18 // (op 1830) least significant half
v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 1830) most significant half
s_waitcnt vmcnt(4)
ds_write_b96 v15, v[20:22] offset:8192 // (op 167) Store local data 

v_mov_b64 v[12:13], s[28:29] // (op 180) convert

v_mov_b32 v23, v55 // (op 180) 
v_mov_b32 v27, v23 // (op 180) reg expression
buffer_load_ubyte v16, v27, s[56:59], 0 offen offset: 0 // (op 180) Load value
v_add_u32 v27, v27, v12 // (op 180) 
s_waitcnt vmcnt(4)
ds_write_b96 v15, v[24:26] offset:8208 // (op 167) Store local data 
s_waitcnt vmcnt(3)
ds_write_b96 v15, v[28:30] offset:8224 // (op 167) Store local data 
s_waitcnt vmcnt(2)
ds_write_b96 v15, v[32:34] offset:8240 // (op 167) Store local data 
buffer_load_ubyte v17, v27, s[56:59], 0 offen offset: 0 // (op 180) Load value




s_lshl_b32 s6, s28, 2 // (op 1894) 
v_add_u32 v55, v55, s6 // (op 1894) 

s_barrier  // (op 2396) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 2373) 






ds_read_b96_tr_b6 v[20:22], v0 offset:8192 // (op 169) Transpose load local data 
ds_read_b96_tr_b6 v[23:25], v0 offset:10240 // (op 169) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v51, v16 offset:24576 // (op 178) Store local data 
s_barrier  // (op 2386) 
s_waitcnt vmcnt(0)
ds_write_b8 v51, v17 offset:24704 // (op 178) Store local data 




ds_read_b128 v[26:29], v3 // (op 171) Load local data 
ds_read_b128 v[30:33], v3 offset:64 // (op 171) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 2400) 


ds_read_u8 v12, v5 offset:24576 // (op 173) Load local data 


v_mov_b32 v13, v14 // (op 175) 

buffer_load_ubyte v16, v13, s[60:63], 0 offen offset: 0 // (op 175) Load value

v_add_u32 v14, 4, v14 // (op 1958) 



ds_read_b96_tr_b6 v[34:36], v0 offset:8224 // (op 321) Transpose load local data 
ds_read_b96_tr_b6 v[37:39], v0 offset:10272 // (op 321) Transpose load local data 
s_waitcnt vmcnt(0) lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:25], v[26:33], a[0:3], v12, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 176) 




ds_read_u8 v17, v5 offset:24608 // (op 325) Load local data 



ds_read_b96_tr_b6 v[20:22], v0 offset:8256 // (op 483) Transpose load local data 
ds_read_b96_tr_b6 v[23:25], v0 offset:10304 // (op 483) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[34:39], v[26:33], a[4:7], v17, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 327) 




ds_read_u8 v17, v5 offset:24640 // (op 487) Load local data 



ds_read_b96_tr_b6 v[34:36], v0 offset:8288 // (op 645) Transpose load local data 
ds_read_b96_tr_b6 v[37:39], v0 offset:10336 // (op 645) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[20:25], v[26:33], a[8:11], v17, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 489) 




ds_read_u8 v17, v5 offset:24672 // (op 649) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[34:39], v[26:33], a[12:15], v17, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 651) 




s_ashr_i64 s[6:7], s[22:23], 7 // (op 639) 
s_mov_b32 s10, s6 // (op 639) convert

s_cmp_lt_u32 s1, s10 // (op 639) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopTop_KLoopTail_639 // (op 639) Condition: Bottom (jump to RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopTop_KLoopTail_639 if true)
RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopBottom_KLoopTail_639:






















s_barrier  // (op 2480) 







v_cmp_eq_f32 vcc, 0.00000, v10 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ConditionalFalse_0_beta_123
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
s_branch RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ConditionalBottom_0_beta_123
RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ConditionalFalse_0_beta_123:



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
RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ConditionalBottom_0_beta_123:












s_endpgm  // End of RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2
.LRR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_end:
.size RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2, .LRR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_end-RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2
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
  MXScaleFormat: NoSwizzle
  ProblemType:
    OperationType: GEMM
    DataType: B6
    DataTypeA: B6
    DataTypeB: B8
    DestDataType: s
    ComputeDataType: s
    HighPrecisionAccumulate: True
    TransposeA: False
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
            { type: int64, semantic: SizeSum },
            { type: int64, semantic: StrideB0 },
            { type: address, semantic: AddressB },
            { type: int64, semantic: StrideScaleA0 },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: SizeSumScaleA },
            { type: int64, semantic: StrideScaleB0 },
            { type: address, semantic: AddressMXScaleB } ]
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
  - .name: RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2
    .symbol: RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 41984
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 64
    .vgpr_count: 80
    .agpr_count: 16
    .max_flat_workgroup_size: 256
    .wavefront_size: 64
    .args:
      - .name: Tensor_10_stride_1_11
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
      - .name: Tensor_0_stride_1_0
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
      - .name: Tensor_0_size_1_1
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
      - .name: Tensor_4_stride_1_6
        .size: 8
        .offset: 88
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 96
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_size_1_7
        .size: 8
        .offset: 104
        .value_kind: by_value
      - .name: Tensor_7_stride_1_9
        .size: 8
        .offset: 112
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 120
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
...

.end_amdgpu_metadata

