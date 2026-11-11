// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2
.p2align 8
.type RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2,@function
RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_exec_begin:


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


v_and_b32 v0, 63, v1 // (op 2143) 
v_and_b32 v3, 48, v1 // (op 2143) 
v_sub_u32 v4, v0, v3 // (op 2143) 

v_lshrrev_b32 v3, 4, v0 // (op 2143) 

v_lshl_add_u32 v0, v3, 5, v4 // (op 2143) 


v_lshrrev_b32 v3, 6, v1 // (op 2143) 
v_and_b32 v4, 1, v3 // (op 2143) 

v_lshlrev_b32 v3, 4, v4 // (op 2143) 

v_lshl_add_u32 v4, v0, 5, v3 // (op 2143) 


v_mul_lo_u32 v0, v4, 6 // (op 2143) 
v_lshrrev_b32 v3, 3, v0 // (op 2143) 

v_lshrrev_b32 v0, 4, v4 // (op 2143) 

v_lshlrev_b32 v4, 2, v0 // (op 2143) 

v_add_u32 v0, v3, v4 // (op 2143) 


v_and_b32 v3, 63, v1 // (op 2287) 
v_and_b32 v4, 48, v1 // (op 2287) 
v_sub_u32 v5, v3, v4 // (op 2287) 

v_lshrrev_b32 v4, 4, v3 // (op 2287) 

v_lshl_add_u32 v3, v4, 5, v5 // (op 2287) 


v_lshrrev_b32 v4, 6, v1 // (op 2287) 
v_and_b32 v5, 1, v4 // (op 2287) 

v_lshlrev_b32 v4, 4, v5 // (op 2287) 

v_lshl_add_u32 v5, v3, 5, v4 // (op 2287) 


v_mul_lo_u32 v3, v5, 6 // (op 2287) 
v_lshrrev_b32 v4, 3, v3 // (op 2287) 

v_lshrrev_b32 v3, 4, v5 // (op 2287) 

v_lshlrev_b32 v5, 2, v3 // (op 2287) 

v_add_u32 v3, v4, v5 // (op 2287) 


v_lshrrev_b32 v4, 7, v1 // (op 2029) 
v_and_b32 v5, 63, v1 // (op 2029) 
v_lshrrev_b32 v6, 4, v5 // (op 2029) 
v_lshlrev_b32 v7, 2, v6 // (op 2029) 

v_lshl_add_u32 v6, v4, 4, v7 // (op 2029) 


v_lshl_add_u32 v4, s16, 7, v6 // (op 2029) 

v_lshrrev_b32 v6, 2, v5 // (op 2029) 

v_and_b32 v5, 3, v6 // (op 2029) 

v_and_b32 v6, 3, v1 // (op 2029) 
v_lshl_add_u32 v7, v5, 2, v6 // (op 2029) 


v_lshrrev_b32 v5, 6, v1 // (op 2029) 
v_and_b32 v6, 1, v5 // (op 2029) 

v_lshl_add_u32 v5, v6, 4, v7 // (op 2029) 


v_lshl_add_u32 v6, s17, 5, v5 // (op 2029) 

v_mul_lo_u32 v5, v6, s2 // (op 2029) 

v_add_lshl_u32 v6, v4, v5, 2 // (op 2029) 


v_and_b32 v4, 48, v1 // (op 2090) 
v_and_b32 v5, 63, v1 // (op 2090) 
v_sub_u32 v7, v5, v4 // (op 2090) 

v_lshrrev_b32 v5, 1, v7 // (op 2090) 
s_mul_i32 s1, s2, s9 // (op 2031) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 2031) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 2031) least significant
s_mul_hi_u32 s19, s2, s8 // (op 2031) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2031) most significant: sum
s_add_u32 s1, s1, s19 // (op 2031) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 2031) 

v_add_lshl_u32 v8, v4, v5, 7 // (op 2090) 


v_and_b32 v4, 1, v7 // (op 2090) 

v_lshlrev_b32 v5, 3, v4 // (op 2090) 

v_lshrrev_b32 v4, 7, v1 // (op 2090) 
v_lshl_add_u32 v7, v4, 4, v5 // (op 2090) 


v_add_u32 v4, v8, v7 // (op 2090) 


v_lshrrev_b32 v5, 7, v1 // (op 2201) 
v_and_b32 v7, 15, v1 // (op 2201) 
v_lshl_add_u32 v8, v5, 4, v7 // (op 2201) 


v_and_b32 v5, 63, v1 // (op 2201) 
v_lshrrev_b32 v7, 4, v5 // (op 2201) 

v_lshl_add_u32 v5, v7, 7, v8 // (op 2201) 


v_lshrrev_b32 v7, 7, v1 // (op 2217) 
v_and_b32 v8, 63, v1 // (op 2217) 
v_lshrrev_b32 v9, 4, v8 // (op 2217) 
v_lshlrev_b32 v10, 2, v9 // (op 2217) 

v_lshl_add_u32 v9, v7, 4, v10 // (op 2217) 


v_lshl_add_u32 v7, s16, 7, v9 // (op 2217) 

v_lshrrev_b32 v9, 2, v8 // (op 2217) 

v_and_b32 v8, 3, v9 // (op 2217) 

v_and_b32 v9, 3, v1 // (op 2217) 
v_lshl_add_u32 v10, v8, 2, v9 // (op 2217) 


v_lshrrev_b32 v8, 6, v1 // (op 2217) 
v_and_b32 v9, 1, v8 // (op 2217) 

v_lshl_add_u32 v8, v9, 4, v10 // (op 2217) 


v_lshl_add_u32 v9, s17, 5, v8 // (op 2217) 

v_mul_lo_u32 v8, v9, s4 // (op 2217) 

v_add_lshl_u32 v9, v7, v8, 2 // (op 2217) 


v_and_b32 v7, 48, v1 // (op 2263) 
v_and_b32 v8, 63, v1 // (op 2263) 
v_sub_u32 v10, v8, v7 // (op 2263) 

v_lshrrev_b32 v8, 1, v10 // (op 2263) 
s_mul_i32 s1, s4, s9 // (op 2219) most significant: low of low * high
v_add_lshl_u32 v11, v7, v8, 7 // (op 2263) 


v_and_b32 v7, 1, v10 // (op 2263) 

v_lshlrev_b32 v8, 3, v7 // (op 2263) 

v_lshrrev_b32 v7, 7, v1 // (op 2263) 
v_lshl_add_u32 v10, v7, 4, v8 // (op 2263) 


v_add_u32 v7, v11, v10 // (op 2263) 


v_lshrrev_b32 v8, 7, v1 // (op 2316) 
v_and_b32 v10, 15, v1 // (op 2316) 
v_lshl_add_u32 v11, v8, 4, v10 // (op 2316) 


v_and_b32 v8, 63, v1 // (op 2316) 
v_lshrrev_b32 v10, 4, v8 // (op 2316) 

v_lshl_add_u32 v8, v10, 7, v11 // (op 2316) 


s_mul_i32 s38, s5, s8 // (op 2219) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 2219) least significant
s_mul_hi_u32 s39, s4, s8 // (op 2219) most significant: high of low * low
s_add_u32 s1, s1, s38 // (op 2219) most significant: sum
s_add_u32 s1, s1, s39 // (op 2219) most significant: sum


s_lshl_b64 s[38:39], s[0:1], 2 // (op 2219) 

s_mov_b32 s40, s6 // (op 2031) reg expression
s_mov_b32 s41, s7 // (op 2031) reg expression
s_mov_b32 s42, s18 // (op 2031) reg expression
s_mov_b32 s43, 131072 // (op 2031) call()

s_mov_b32 s44, s10 // (op 2219) reg expression

s_mov_b32 s45, s11 // (op 2219) reg expression
s_mov_b32 s46, s38 // (op 2219) reg expression
s_mov_b32 s47, 131072 // (op 2219) call()



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
v_and_b32 v12, -8, v1 // (op 1796) 
v_sub_u32 v13, v1, v12 // (op 1796) 

v_lshlrev_b32 v12, 4, v13 // (op 1796) 

v_lshl_add_u32 v13, s16, 7, v12 // (op 1796) 

v_mov_b32 v15, 0 // (op 1796) convert
v_mov_b32 v14, v13 // (op 1796) convert

v_lshrrev_b32 v12, 3, v1 // (op 1796) 
v_mul_lo_u32 v17, v12, s13 // (op 1796) most significant: low of low * high
v_mul_hi_u32 v13, v12, s12 // (op 1796) most significant: high of low * low
v_mul_lo_u32 v16, v12, s12 // (op 1796) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 1796) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v16 // (op 1796) least significant half
v_addc_co_u32 v13, s[0:1], v15, v17, s[0:1] // (op 1796) most significant half




s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 1798) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 1798) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 1798) least significant
s_mul_hi_u32 s7, s12, s22 // (op 1798) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1798) most significant: sum
s_add_u32 s1, s1, s7 // (op 1798) most significant: sum


s_mov_b32 s8, s20 // (op 1798) reg expression
s_mov_b32 s9, s21 // (op 1798) reg expression
s_mov_b32 s10, s0 // (op 1798) reg expression
s_mov_b32 s11, 131072 // (op 1798) call()

v_and_b32 v14, -8, v1 // (op 1815) 
v_sub_u32 v15, v1, v14 // (op 1815) 

v_lshlrev_b32 v14, 4, v15 // (op 1815) 

v_lshl_add_u32 v15, s16, 7, v14 // (op 1815) 

v_mov_b32 v17, 0 // (op 1815) convert
v_mov_b32 v16, v15 // (op 1815) convert

v_lshrrev_b32 v14, 3, v1 // (op 1815) 
v_mul_lo_u32 v19, v14, s13 // (op 1815) most significant: low of low * high
v_mul_hi_u32 v15, v14, s12 // (op 1815) most significant: high of low * low
v_mul_lo_u32 v18, v14, s12 // (op 1815) least significant: low of low * low
v_add_u32 v19, v19, v15 // (op 1815) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v18 // (op 1815) least significant half
v_addc_co_u32 v15, s[0:1], v17, v19, s[0:1] // (op 1815) most significant half



s_mul_i32 s1, s12, s23 // (op 1817) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 1817) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 1817) least significant
s_mul_hi_u32 s7, s12, s22 // (op 1817) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1817) most significant: sum
s_add_u32 s1, s1, s7 // (op 1817) most significant: sum


s_mov_b32 s8, s20 // (op 1817) reg expression
s_mov_b32 s9, s21 // (op 1817) reg expression
s_mov_b32 s10, s0 // (op 1817) reg expression
s_mov_b32 s11, 131072 // (op 1817) call()

v_and_b32 v16, -2, v1 // (op 1860) 
v_sub_u32 v17, v1, v16 // (op 1860) 

v_lshlrev_b32 v16, 4, v17 // (op 1860) 

v_lshl_add_u32 v17, s17, 5, v16 // (op 1860) 

v_mov_b32 v19, 0 // (op 1860) convert
v_mov_b32 v18, v17 // (op 1860) convert

v_lshrrev_b32 v16, 1, v1 // (op 1860) 
v_mul_lo_u32 v21, v16, s25 // (op 1860) most significant: low of low * high
v_mul_hi_u32 v17, v16, s24 // (op 1860) most significant: high of low * low
v_mul_lo_u32 v20, v16, s24 // (op 1860) least significant: low of low * low
v_add_u32 v21, v21, v17 // (op 1860) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v18 // (op 1860) least significant half
v_addc_co_u32 v21, s[0:1], v21, v19, s[0:1] // (op 1860) most significant half



v_mul_lo_u32 v17, v21, 6 // (op 1860) most significant: low of high * low
v_mul_hi_u32 v18, v20, 6 // (op 1860) most significant: high of low * low
v_mul_lo_u32 v16, v20, 6 // (op 1860) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 1860) most significant: sum


v_lshrrev_b64 v[18:19], 3, v[16:17] // (op 1860) 

v_mov_b32 v16, v18 // (op 1860) convert

s_mul_i32 s1, s23, 6 // (op 1862) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 1862) least significant
s_mul_hi_u32 s6, s22, 6 // (op 1862) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1862) most significant: sum

s_mul_i32 s7, s24, s1 // (op 1862) most significant: low of low * high
s_mul_i32 s14, s25, s0 // (op 1862) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 1862) least significant
s_mul_hi_u32 s15, s24, s0 // (op 1862) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 1862) most significant: sum
s_add_u32 s7, s7, s15 // (op 1862) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 1862) 

s_mov_b32 s48, s26 // (op 1862) reg expression
s_mov_b32 s49, s27 // (op 1862) reg expression
s_mov_b32 s50, s0 // (op 1862) reg expression
s_mov_b32 s51, 131072 // (op 1862) call()

v_and_b32 v17, -2, v1 // (op 1879) 
v_sub_u32 v18, v1, v17 // (op 1879) 

v_lshlrev_b32 v17, 4, v18 // (op 1879) 

v_lshl_add_u32 v18, s17, 5, v17 // (op 1879) 

v_mov_b32 v21, 0 // (op 1879) convert
v_mov_b32 v20, v18 // (op 1879) convert

v_lshrrev_b32 v17, 1, v1 // (op 1879) 
v_mul_lo_u32 v19, v17, s25 // (op 1879) most significant: low of low * high
v_mul_hi_u32 v22, v17, s24 // (op 1879) most significant: high of low * low
v_mul_lo_u32 v18, v17, s24 // (op 1879) least significant: low of low * low
v_add_u32 v19, v19, v22 // (op 1879) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v20 // (op 1879) least significant half
v_addc_co_u32 v19, s[0:1], v19, v21, s[0:1] // (op 1879) most significant half



v_mul_lo_u32 v21, v19, 6 // (op 1879) most significant: low of high * low
v_mul_hi_u32 v17, v18, 6 // (op 1879) most significant: high of low * low
v_mul_lo_u32 v20, v18, 6 // (op 1879) least significant: low of low * low
v_add_u32 v21, v21, v17 // (op 1879) most significant: sum


v_lshrrev_b64 v[18:19], 3, v[20:21] // (op 1879) 

v_mov_b32 v17, v18 // (op 1879) convert

s_mul_i32 s1, s23, 6 // (op 1881) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 1881) least significant
s_mul_hi_u32 s6, s22, 6 // (op 1881) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1881) most significant: sum

s_mul_i32 s7, s24, s1 // (op 1881) most significant: low of low * high
s_mul_i32 s14, s25, s0 // (op 1881) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 1881) least significant
s_mul_hi_u32 s15, s24, s0 // (op 1881) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 1881) most significant: sum
s_add_u32 s7, s7, s15 // (op 1881) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 1881) 

s_mov_b32 s48, s26 // (op 1881) reg expression
s_mov_b32 s49, s27 // (op 1881) reg expression
s_mov_b32 s50, s0 // (op 1881) reg expression
s_mov_b32 s51, 131072 // (op 1881) call()

v_and_b32 v18, -128, v1 // (op 1924) 
v_sub_u32 v19, v1, v18 // (op 1924) 

v_lshl_add_u32 v18, s16, 7, v19 // (op 1924) 

v_lshrrev_b32 v19, 7, v1 // (op 1924) 
v_lshlrev_b32 v20, 1, v19 // (op 1924) 

v_mul_lo_u32 v19, v20, s28 // (op 1924) 

v_add_u32 v20, v18, v19 // (op 1924) 


s_mul_i32 s1, s28, s33 // (op 1926) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 1926) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1926) least significant
s_mul_hi_u32 s7, s28, s32 // (op 1926) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1926) most significant: sum
s_add_u32 s1, s1, s7 // (op 1926) most significant: sum


s_mov_b32 s52, s30 // (op 1926) reg expression
s_mov_b32 s53, s31 // (op 1926) reg expression
s_mov_b32 s54, s0 // (op 1926) reg expression
s_mov_b32 s55, 131072 // (op 1926) call()

v_and_b32 v18, -128, v1 // (op 1943) 
v_sub_u32 v19, v1, v18 // (op 1943) 

v_lshl_add_u32 v18, s16, 7, v19 // (op 1943) 

v_lshrrev_b32 v19, 7, v1 // (op 1943) 
v_lshlrev_b32 v21, 1, v19 // (op 1943) 

v_mul_lo_u32 v19, v21, s28 // (op 1943) 

v_add_u32 v21, v18, v19 // (op 1943) 


s_mul_i32 s1, s28, s33 // (op 1945) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 1945) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1945) least significant
s_mul_hi_u32 s7, s28, s32 // (op 1945) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1945) most significant: sum
s_add_u32 s1, s1, s7 // (op 1945) most significant: sum


s_mov_b32 s52, s30 // (op 1945) reg expression
s_mov_b32 s53, s31 // (op 1945) reg expression
s_mov_b32 s54, s0 // (op 1945) reg expression
s_mov_b32 s55, 131072 // (op 1945) call()

v_lshrrev_b32 v18, 6, v1 // (op 1988) 
v_and_b32 v19, 1, v18 // (op 1988) 

v_and_b32 v18, 15, v1 // (op 1988) 
v_lshl_add_u32 v22, v19, 4, v18 // (op 1988) 


v_lshl_add_u32 v18, s17, 5, v22 // (op 1988) 

v_and_b32 v19, 63, v1 // (op 1988) 
v_lshrrev_b32 v22, 4, v19 // (op 1988) 

v_mul_lo_u32 v19, v22, s34 // (op 1988) 
v_add_u32 v19, v19, v18 // (op 1988) 


s_mul_i32 s1, s34, s33 // (op 1990) most significant: low of low * high
s_mul_i32 s6, s35, s32 // (op 1990) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 1990) least significant
s_mul_hi_u32 s7, s34, s32 // (op 1990) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1990) most significant: sum
s_add_u32 s1, s1, s7 // (op 1990) most significant: sum


s_mov_b32 s56, s36 // (op 1990) reg expression
s_mov_b32 s57, s37 // (op 1990) reg expression
s_mov_b32 s58, s0 // (op 1990) reg expression
s_mov_b32 s59, 131072 // (op 1990) call()

v_lshrrev_b32 v18, 6, v1 // (op 2007) 
v_and_b32 v22, 1, v18 // (op 2007) 

v_and_b32 v18, 15, v1 // (op 2007) 
v_lshl_add_u32 v23, v22, 4, v18 // (op 2007) 


v_lshl_add_u32 v18, s17, 5, v23 // (op 2007) 

v_and_b32 v22, 63, v1 // (op 2007) 
v_lshrrev_b32 v23, 4, v22 // (op 2007) 

v_mul_lo_u32 v22, v23, s34 // (op 2007) 
v_add_u32 v22, v22, v18 // (op 2007) 


s_mul_i32 s1, s34, s33 // (op 2009) most significant: low of low * high
s_mul_i32 s6, s35, s32 // (op 2009) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 2009) least significant
s_mul_hi_u32 s7, s34, s32 // (op 2009) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2009) most significant: sum
s_add_u32 s1, s1, s7 // (op 2009) most significant: sum


s_mov_b32 s56, s36 // (op 2009) reg expression
s_mov_b32 s57, s37 // (op 2009) reg expression
s_mov_b32 s58, s0 // (op 2009) reg expression
s_mov_b32 s59, 131072 // (op 2009) call()


v_mov_b64 v[24:25], s[34:35] // (op 1598) convert

v_mov_b32 v18, v19 // (op 1598) reg expression
v_mov_b32 v23, v18 // (op 1598) 
buffer_load_ubyte v26, v23, s[56:59], 0 offen offset: 0 // (op 1598) Load value





v_mov_b32 v18, v16 // (op 1602) reg expression

buffer_load_dwordx3 v[28:30], v18, s[48:51], 0 offen // (op 1602) Load value

v_and_b32 v18, -8, v1 // (op 2073) 
v_sub_u32 v27, v1, v18 // (op 2073) 

v_lshlrev_b32 v18, 4, v27 // (op 2073) 

v_lshrrev_b32 v27, 3, v1 // (op 2073) 
v_lshl_add_u32 v23, v27, 7, v18 // (op 2073) 


s_nop 0
v_readfirstlane_b32 s0, v23 // (op 2073) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 5 // (op 1749) 
v_mov_b64 v[24:25], s[6:7] // (op 1749) convert


v_mov_b64 v[32:33], v[12:13] // (op 1749) reg expression
s_mov_b32 m0, s0 // (op 1749) reg expression
buffer_load_dwordx4 v32, s[8:11], 0 offen offset: 0 lds // (op 1749) Load value direct to lds
v_add_co_u32 v32, s[6:7], v32, v24 // (op 1749) least significant half
v_addc_co_u32 v33, s[6:7], 0, v33, s[6:7] // (op 1749) most significant half

s_add_u32 m0, m0, 4096 // (op 1749) 
buffer_load_dwordx4 v32, s[8:11], 0 offen offset: 0 lds // (op 1749) Load value direct to lds
v_add_co_u32 v32, s[6:7], v32, v24 // (op 1749) least significant half
v_addc_co_u32 v33, s[6:7], 0, v33, s[6:7] // (op 1749) most significant half

s_add_u32 m0, m0, 4096 // (op 1749) 
buffer_load_dwordx4 v32, s[8:11], 0 offen offset: 0 lds // (op 1749) Load value direct to lds
v_add_co_u32 v32, s[6:7], v32, v24 // (op 1749) least significant half
v_addc_co_u32 v33, s[6:7], 0, v33, s[6:7] // (op 1749) most significant half

s_add_u32 m0, m0, 4096 // (op 1749) 
buffer_load_dwordx4 v32, s[8:11], 0 offen offset: 0 lds // (op 1749) Load value direct to lds




v_mov_b64 v[24:25], s[28:29] // (op 1608) convert

v_mov_b32 v18, v20 // (op 1608) reg expression
buffer_load_ubyte v32, v18, s[52:55], 0 offen offset: 0 // (op 1608) Load value
v_add_u32 v18, v18, v24 // (op 1608) 
buffer_load_ubyte v33, v18, s[52:55], 0 offen offset: 0 // (op 1608) Load value



v_mov_b64 v[24:25], s[34:35] // (op 1612) convert

s_lshl_b32 s0, s34, 2 // (op 1612) 
v_add_u32 v18, v19, s0 // (op 1612) 

v_mov_b32 v23, v18 // (op 1612) 
buffer_load_ubyte v27, v23, s[56:59], 0 offen offset: 0 // (op 1612) Load value






s_lshl_b64 s[0:1], s[24:25], 7 // (op 1616) 
s_mul_i32 s7, s1, 6 // (op 1616) most significant: low of high * low
s_mul_i32 s6, s0, 6 // (op 1616) least significant
s_mul_hi_u32 s14, s0, 6 // (op 1616) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 1616) most significant: sum


s_lshr_b64 s[0:1], s[6:7], 3 // (op 1616) 

v_add_u32 v31, v16, s0 // (op 1616) 


buffer_load_dwordx3 v[34:36], v31, s[48:51], 0 offen // (op 1616) Load value


v_and_b32 v16, -8, v1 // (op 2246) 
v_sub_u32 v31, v1, v16 // (op 2246) 

v_lshlrev_b32 v16, 4, v31 // (op 2246) 

v_lshrrev_b32 v31, 3, v1 // (op 2246) 
v_lshl_add_u32 v18, v31, 7, v16 // (op 2246) 


s_nop 0
v_readfirstlane_b32 s0, v18 // (op 2246) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 5 // (op 1754) 
v_mov_b64 v[18:19], s[6:7] // (op 1754) convert


s_lshl_b64 s[6:7], s[12:13], 7 // (op 1754) 
v_mov_b32 v16, s6 // (op 1754) 
v_mov_b32 v31, s7 // (op 1754) 
v_add_co_u32 v24, s[14:15], v12, v16 // (op 1754) least significant half
v_addc_co_u32 v25, s[14:15], v13, v31, s[14:15] // (op 1754) most significant half




s_add_u32 s1, 16384, s0 // (op 1754) 
s_mov_b32 m0, s1 // (op 1754) reg expression
buffer_load_dwordx4 v24, s[8:11], 0 offen offset: 0 lds // (op 1754) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v18 // (op 1754) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 1754) most significant half

s_add_u32 m0, m0, 4096 // (op 1754) 
buffer_load_dwordx4 v24, s[8:11], 0 offen offset: 0 lds // (op 1754) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v18 // (op 1754) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 1754) most significant half

s_add_u32 m0, m0, 4096 // (op 1754) 
buffer_load_dwordx4 v24, s[8:11], 0 offen offset: 0 lds // (op 1754) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v18 // (op 1754) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 1754) most significant half

s_add_u32 m0, m0, 4096 // (op 1754) 
buffer_load_dwordx4 v24, s[8:11], 0 offen offset: 0 lds // (op 1754) Load value direct to lds






v_mov_b64 v[12:13], s[28:29] // (op 1622) convert

s_lshl_b32 s0, s28, 2 // (op 1622) 
v_add_u32 v16, v20, s0 // (op 1622) 

buffer_load_ubyte v18, v16, s[52:55], 0 offen offset: 0 // (op 1622) Load value
v_add_u32 v16, v16, v12 // (op 1622) 
buffer_load_ubyte v19, v16, s[52:55], 0 offen offset: 0 // (op 1622) Load value



v_and_b32 v16, -2, v1 // (op 2131) 
v_sub_u32 v20, v1, v16 // (op 2131) 

v_lshlrev_b32 v16, 4, v20 // (op 2131) 

v_lshrrev_b32 v20, 1, v1 // (op 2131) 
v_lshl_add_u32 v31, v20, 5, v16 // (op 2131) 


v_mul_lo_u32 v16, v31, 6 // (op 2131) 
v_lshrrev_b32 v20, 3, v16 // (op 2131) 

v_lshrrev_b32 v16, 4, v31 // (op 2131) 

v_lshlrev_b32 v31, 2, v16 // (op 2131) 

v_add_u32 v16, v20, v31 // (op 2131) 





s_waitcnt vmcnt(14)
ds_write_b96 v16, v[28:30] offset:32768 // (op 1624) Store local data 

v_and_b32 v16, -128, v1 // (op 2184) 
v_sub_u32 v20, v1, v16 // (op 2184) 

v_lshrrev_b32 v16, 7, v1 // (op 2184) 
v_lshl_add_u32 v31, v16, 8, v20 // (op 2184) 





s_waitcnt vmcnt(9)
ds_write_b8 v31, v32 offset:36864 // (op 1626) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v31, v33 offset:36992 // (op 1626) Store local data 

v_and_b32 v16, -8, v1 // (op 2045) 
v_sub_u32 v20, v1, v16 // (op 2045) 

v_lshlrev_b32 v16, 4, v20 // (op 2045) 

v_lshrrev_b32 v20, 3, v1 // (op 2045) 
v_lshl_add_u32 v31, v20, 7, v16 // (op 2045) 


s_nop 0
v_readfirstlane_b32 s0, v31 // (op 2045) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v16, -2, v1 // (op 2102) 
v_sub_u32 v20, v1, v16 // (op 2102) 

v_lshlrev_b32 v16, 4, v20 // (op 2102) 

v_lshrrev_b32 v20, 1, v1 // (op 2102) 
v_lshl_add_u32 v31, v20, 5, v16 // (op 2102) 


v_mul_lo_u32 v16, v31, 6 // (op 2102) 
v_lshrrev_b32 v20, 3, v16 // (op 2102) 

v_lshrrev_b32 v16, 4, v31 // (op 2102) 

v_lshlrev_b32 v31, 2, v16 // (op 2102) 

v_add_u32 v16, v20, v31 // (op 2102) 


v_and_b32 v20, -128, v1 // (op 2155) 
v_sub_u32 v31, v1, v20 // (op 2155) 

v_lshrrev_b32 v20, 7, v1 // (op 2155) 
v_lshl_add_u32 v12, v20, 8, v31 // (op 2155) 


v_and_b32 v13, -8, v1 // (op 2231) 
v_sub_u32 v20, v1, v13 // (op 2231) 

v_lshlrev_b32 v13, 4, v20 // (op 2231) 

v_lshrrev_b32 v20, 3, v1 // (op 2231) 
v_lshl_add_u32 v31, v20, 7, v13 // (op 2231) 


s_nop 0
v_readfirstlane_b32 s1, v31 // (op 2231) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v13, -2, v1 // (op 2275) 
v_sub_u32 v20, v1, v13 // (op 2275) 

v_lshlrev_b32 v13, 4, v20 // (op 2275) 

v_lshrrev_b32 v20, 1, v1 // (op 2275) 
v_lshl_add_u32 v31, v20, 5, v13 // (op 2275) 


v_mul_lo_u32 v13, v31, 6 // (op 2275) 
v_lshrrev_b32 v20, 3, v13 // (op 2275) 

v_lshrrev_b32 v13, 4, v31 // (op 2275) 

v_lshlrev_b32 v31, 2, v13 // (op 2275) 

v_add_u32 v13, v20, v31 // (op 2275) 


v_and_b32 v20, -128, v1 // (op 2299) 
v_sub_u32 v31, v1, v20 // (op 2299) 

v_lshrrev_b32 v20, 7, v1 // (op 2299) 
v_lshl_add_u32 v23, v20, 8, v31 // (op 2299) 


s_mov_b32 s6, 0 // (op 635) call()
s_ashr_i64 s[14:15], s[22:23], 7 // (op 634) 
s_mov_b32 s7, s14 // (op 634) convert

s_and_b32 s14, s7, 4294967294 // (op 634) 

s_cmp_lt_u32 s6, s14 // (op 634) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopBottom_KLoop_634 // (op 634) Condition: Top (jump to RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopBottom_KLoop_634 if false)
RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopTop_KLoop_634:
s_barrier  // (op 2363) 
s_add_u32 s6, s6, 2 // (op 636) 



ds_read_b64_tr_b8 v[38:39], v4 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v4 offset:1024 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v4 offset:8192 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v4 offset:9216 // (op 65) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 2376) 



ds_read_b96_tr_b6 v[46:48], v0 offset:32768 // (op 70) Transpose load local data 
ds_read_b96_tr_b6 v[49:51], v0 offset:33280 // (op 70) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 2390) 


ds_read_u8 v20, v5 offset:36864 // (op 75) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[38:45], v[46:51], a[0:3], v20, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 82) 
ds_read_b64_tr_b8 v[52:53], v4 offset:32 // (op 340) Transpose load local data 


ds_read_b64_tr_b8 v[54:55], v4 offset:1056 // (op 340) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v4 offset:8224 // (op 340) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v4 offset:9248 // (op 340) Transpose load local data 


ds_read_u8 v20, v5 offset:36896 // (op 344) Load local data 



ds_read_b64_tr_b8 v[38:39], v4 offset:64 // (op 502) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[52:59], v[46:51], a[4:7], v20, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 346) 
ds_read_b64_tr_b8 v[40:41], v4 offset:1088 // (op 502) Transpose load local data 


ds_read_b64_tr_b8 v[42:43], v4 offset:8256 // (op 502) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v4 offset:9280 // (op 502) Transpose load local data 


ds_read_u8 v20, v5 offset:36928 // (op 506) Load local data 



ds_read_b64_tr_b8 v[52:53], v4 offset:96 // (op 664) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v4 offset:1120 // (op 664) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[38:45], v[46:51], a[8:11], v20, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 508) 
ds_read_b64_tr_b8 v[56:57], v4 offset:8288 // (op 664) Transpose load local data 


ds_read_b64_tr_b8 v[58:59], v4 offset:9312 // (op 664) Transpose load local data 


ds_read_u8 v20, v5 offset:36960 // (op 668) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[52:59], v[46:51], a[12:15], v20, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 670) 




v_mov_b64 v[24:25], s[34:35] // (op 80) convert

v_mov_b32 v20, v22 // (op 80) 
s_lshl_b32 s7, s34, 3 // (op 80) 
v_add_u32 v31, v20, s7 // (op 80) 

v_mov_b32 v37, v31 // (op 80) 
buffer_load_ubyte v26, v37, s[56:59], 0 offen offset: 0 // (op 80) Load value





v_mov_b32 v31, v17 // (op 10) 
s_lshl_b64 s[14:15], s[24:25], 7 // (op 10) 
s_mul_i32 s19, s15, 6 // (op 10) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 10) least significant
s_mul_hi_u32 s7, s14, 6 // (op 10) most significant: high of low * low
s_add_u32 s19, s19, s7 // (op 10) most significant: sum


s_lshr_b64 s[14:15], s[18:19], 3 // (op 10) 

s_lshl_b32 s7, s14, 1 // (op 10) 

v_add_u32 v24, v31, s7 // (op 10) 


buffer_load_dwordx3 v[28:30], v24, s[48:51], 0 offen // (op 10) Load value

s_barrier  // (op 2366) 

s_lshl_b64 s[14:15], s[12:13], 5 // (op 1733) 
v_mov_b64 v[24:25], s[14:15] // (op 1733) convert


v_mov_b64 v[38:39], v[14:15] // (op 1733) 
s_lshl_b64 s[14:15], s[12:13], 7 // (op 1733) 
s_lshl_b64 s[18:19], s[14:15], 1 // (op 1733) 

v_mov_b32 v37, s18 // (op 1733) 
v_mov_b32 v40, s19 // (op 1733) 
v_add_co_u32 v42, s[14:15], v38, v37 // (op 1733) least significant half
v_addc_co_u32 v43, s[14:15], v39, v40, s[14:15] // (op 1733) most significant half




s_mov_b32 m0, s0 // (op 1733) reg expression
buffer_load_dwordx4 v42, s[8:11], 0 offen offset: 0 lds // (op 1733) Load value direct to lds
v_add_co_u32 v42, s[14:15], v42, v24 // (op 1733) least significant half
v_addc_co_u32 v43, s[14:15], 0, v43, s[14:15] // (op 1733) most significant half

s_add_u32 m0, m0, 4096 // (op 1733) 
buffer_load_dwordx4 v42, s[8:11], 0 offen offset: 0 lds // (op 1733) Load value direct to lds
v_add_co_u32 v42, s[14:15], v42, v24 // (op 1733) least significant half
v_addc_co_u32 v43, s[14:15], 0, v43, s[14:15] // (op 1733) most significant half

s_add_u32 m0, m0, 4096 // (op 1733) 
buffer_load_dwordx4 v42, s[8:11], 0 offen offset: 0 lds // (op 1733) Load value direct to lds
v_add_co_u32 v42, s[14:15], v42, v24 // (op 1733) least significant half
v_addc_co_u32 v43, s[14:15], 0, v43, s[14:15] // (op 1733) most significant half

s_add_u32 m0, m0, 4096 // (op 1733) 
buffer_load_dwordx4 v42, s[8:11], 0 offen offset: 0 lds // (op 1733) Load value direct to lds



v_mov_b64 v[24:25], s[28:29] // (op 17) convert

v_mov_b32 v37, v21 // (op 17) 
s_lshl_b32 s7, s28, 3 // (op 17) 
v_add_u32 v40, v37, s7 // (op 17) 

buffer_load_ubyte v32, v40, s[52:55], 0 offen offset: 0 // (op 17) Load value
v_add_u32 v40, v40, v24 // (op 17) 
buffer_load_ubyte v33, v40, s[52:55], 0 offen offset: 0 // (op 17) Load value


s_barrier  // (op 2413) 



s_waitcnt vmcnt(14)
ds_write_b96 v13, v[34:36] offset:37376 // (op 235) Store local data 
s_barrier  // (op 2419) 



s_waitcnt vmcnt(9)
ds_write_b8 v23, v18 offset:41472 // (op 236) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v23, v19 offset:41600 // (op 236) Store local data 
s_barrier  // (op 2404) 



ds_read_b64_tr_b8 v[40:41], v7 offset:16384 // (op 227) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v7 offset:17408 // (op 227) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v7 offset:24576 // (op 227) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v7 offset:25600 // (op 227) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 2410) 



ds_read_b96_tr_b6 v[48:50], v3 offset:37376 // (op 229) Transpose load local data 
ds_read_b96_tr_b6 v[51:53], v3 offset:37888 // (op 229) Transpose load local data 
s_barrier  // (op 2448) 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 2416) 


ds_read_u8 v24, v8 offset:41472 // (op 231) Load local data 



ds_read_b64_tr_b8 v[54:55], v7 offset:16416 // (op 353) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v7 offset:17440 // (op 353) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[40:47], v[48:53], a[0:3], v24, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 234) 
ds_read_b64_tr_b8 v[58:59], v7 offset:24608 // (op 353) Transpose load local data 


ds_read_b64_tr_b8 v[60:61], v7 offset:25632 // (op 353) Transpose load local data 


ds_read_u8 v24, v8 offset:41504 // (op 357) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[54:61], v[48:53], a[4:7], v24, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 347) 
ds_read_b64_tr_b8 v[40:41], v7 offset:16448 // (op 515) Transpose load local data 


ds_read_b64_tr_b8 v[42:43], v7 offset:17472 // (op 515) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v7 offset:24640 // (op 515) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v7 offset:25664 // (op 515) Transpose load local data 


ds_read_u8 v24, v8 offset:41536 // (op 519) Load local data 



ds_read_b64_tr_b8 v[54:55], v7 offset:16480 // (op 677) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[40:47], v[48:53], a[8:11], v24, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 509) 
ds_read_b64_tr_b8 v[56:57], v7 offset:17504 // (op 677) Transpose load local data 


ds_read_b64_tr_b8 v[58:59], v7 offset:24672 // (op 677) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v7 offset:25696 // (op 677) Transpose load local data 


ds_read_u8 v24, v8 offset:41568 // (op 681) Load local data 
s_barrier  // (op 2498) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[54:61], v[48:53], a[12:15], v24, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 671) 




v_mov_b64 v[24:25], s[34:35] // (op 233) convert

s_lshl_b32 s7, s34, 2 // (op 233) 
s_mul_i32 s14, 3, s7 // (op 233) 

v_add_u32 v40, v20, s14 // (op 233) 

v_mov_b32 v41, v40 // (op 233) 
buffer_load_ubyte v27, v41, s[56:59], 0 offen offset: 0 // (op 233) Load value






s_lshl_b32 s7, s34, 3 // (op 1999) 
v_add_u32 v22, v22, s7 // (op 1999) 

s_lshl_b64 s[14:15], s[24:25], 7 // (op 237) 
s_mul_i32 s19, s15, 6 // (op 237) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 237) least significant
s_mul_hi_u32 s7, s14, 6 // (op 237) most significant: high of low * low
s_add_u32 s19, s19, s7 // (op 237) most significant: sum


s_lshr_b64 s[14:15], s[18:19], 3 // (op 237) 

s_mul_i32 s7, 3, s14 // (op 237) 

v_add_u32 v20, v31, s7 // (op 237) 


buffer_load_dwordx3 v[34:36], v20, s[48:51], 0 offen // (op 237) Load value


s_barrier  // (op 2407) 
s_lshl_b64 s[14:15], s[24:25], 7 // (op 1871) 

s_mul_i32 s19, s15, 6 // (op 1871) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 1871) least significant
s_mul_hi_u32 s7, s14, 6 // (op 1871) most significant: high of low * low
s_add_u32 s19, s19, s7 // (op 1871) most significant: sum


s_lshr_b64 s[14:15], s[18:19], 3 // (op 1871) 

s_lshl_b32 s7, s14, 1 // (op 1871) 

v_add_u32 v17, v17, s7 // (op 1871) 

s_lshl_b64 s[14:15], s[12:13], 5 // (op 1744) 
v_mov_b64 v[24:25], s[14:15] // (op 1744) convert


s_lshl_b64 s[14:15], s[12:13], 7 // (op 1744) 
s_mul_i32 s19, 3, s15 // (op 1744) most significant: low of low * high
s_mul_i32 s18, 3, s14 // (op 1744) least significant
s_mul_hi_u32 s7, 3, s14 // (op 1744) most significant: high of low * low
s_add_u32 s19, s19, s7 // (op 1744) most significant: sum


v_mov_b32 v20, s18 // (op 1744) 
v_mov_b32 v31, s19 // (op 1744) 
v_add_co_u32 v40, s[14:15], v38, v20 // (op 1744) least significant half
v_addc_co_u32 v41, s[14:15], v39, v31, s[14:15] // (op 1744) most significant half




s_add_u32 s7, 16384, s1 // (op 1744) 
s_mov_b32 m0, s7 // (op 1744) reg expression
buffer_load_dwordx4 v40, s[8:11], 0 offen offset: 0 lds // (op 1744) Load value direct to lds
v_add_co_u32 v40, s[14:15], v40, v24 // (op 1744) least significant half
v_addc_co_u32 v41, s[14:15], 0, v41, s[14:15] // (op 1744) most significant half

s_add_u32 m0, m0, 4096 // (op 1744) 
buffer_load_dwordx4 v40, s[8:11], 0 offen offset: 0 lds // (op 1744) Load value direct to lds
v_add_co_u32 v40, s[14:15], v40, v24 // (op 1744) least significant half
v_addc_co_u32 v41, s[14:15], 0, v41, s[14:15] // (op 1744) most significant half

s_add_u32 m0, m0, 4096 // (op 1744) 
buffer_load_dwordx4 v40, s[8:11], 0 offen offset: 0 lds // (op 1744) Load value direct to lds
v_add_co_u32 v40, s[14:15], v40, v24 // (op 1744) least significant half
v_addc_co_u32 v41, s[14:15], 0, v41, s[14:15] // (op 1744) most significant half

s_add_u32 m0, m0, 4096 // (op 1744) 
buffer_load_dwordx4 v40, s[8:11], 0 offen offset: 0 lds // (op 1744) Load value direct to lds





v_mov_b64 v[24:25], s[28:29] // (op 238) convert

s_lshl_b32 s7, s28, 2 // (op 238) 
s_mul_i32 s14, 3, s7 // (op 238) 

v_add_u32 v20, v37, s14 // (op 238) 

s_lshl_b64 s[14:15], s[12:13], 7 // (op 1807) 
s_lshl_b64 s[18:19], s[14:15], 1 // (op 1807) 

v_mov_b32 v31, s18 // (op 1807) 
v_mov_b32 v38, s19 // (op 1807) 
v_add_co_u32 v14, s[14:15], v14, v31 // (op 1807) least significant half
v_addc_co_u32 v15, s[14:15], v15, v38, s[14:15] // (op 1807) most significant half




buffer_load_ubyte v18, v20, s[52:55], 0 offen offset: 0 // (op 238) Load value
v_add_u32 v20, v20, v24 // (op 238) 
buffer_load_ubyte v19, v20, s[52:55], 0 offen offset: 0 // (op 238) Load value



s_barrier  // (op 2379) 
s_lshl_b32 s7, s28, 3 // (op 1935) 



v_add_u32 v21, v21, s7 // (op 1935) 

s_waitcnt vmcnt(14)
ds_write_b96 v16, v[28:30] offset:32768 // (op 13) Store local data 
s_barrier  // (op 2393) 



s_waitcnt vmcnt(9)
ds_write_b8 v12, v32 offset:36864 // (op 21) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v12, v33 offset:36992 // (op 21) Store local data 
s_ashr_i64 s[14:15], s[22:23], 7 // (op 634) 
s_mov_b32 s7, s14 // (op 634) convert

s_and_b32 s14, s7, 4294967294 // (op 634) 

s_cmp_lt_u32 s6, s14 // (op 634) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopTop_KLoop_634 // (op 634) Condition: Bottom (jump to RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopTop_KLoop_634 if true)
RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopBottom_KLoop_634:














s_barrier  // (op 2480) 
v_and_b32 v12, -8, v1 // (op 1767) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 1767) 
v_sub_u32 v13, v1, v12 // (op 1767) 

s_lshr_b32 s6, s0, 1 // (op 1767) 

v_lshlrev_b32 v12, 4, v13 // (op 1767) 

v_lshl_add_u32 v13, s16, 7, v12 // (op 1767) 

v_mov_b32 v15, 0 // (op 1767) convert
v_mov_b32 v14, v13 // (op 1767) convert

v_lshrrev_b32 v12, 3, v1 // (op 1767) 
v_lshl_add_u32 v13, s6, 8, v12 // (op 1767) 


v_mul_lo_u32 v17, v13, s13 // (op 1767) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 1767) most significant: high of low * low
v_mul_lo_u32 v16, v13, s12 // (op 1767) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 1767) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v16 // (op 1767) least significant half
v_addc_co_u32 v13, s[0:1], v15, v17, s[0:1] // (op 1767) most significant half



s_mul_i32 s1, s12, s23 // (op 1769) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 1769) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 1769) least significant
s_mul_hi_u32 s7, s12, s22 // (op 1769) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1769) most significant: sum
s_add_u32 s1, s1, s7 // (op 1769) most significant: sum


s_mov_b32 s8, s20 // (op 1769) reg expression
s_mov_b32 s9, s21 // (op 1769) reg expression
s_mov_b32 s10, s0 // (op 1769) reg expression
s_mov_b32 s11, 131072 // (op 1769) call()

v_lshrrev_b32 v14, 1, v1 // (op 1837) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 1837) 
v_and_b32 v15, -2, v1 // (op 1837) 
s_lshr_b32 s6, s0, 1 // (op 1837) 

v_sub_u32 v16, v1, v15 // (op 1837) 

v_lshl_add_u32 v15, s6, 8, v14 // (op 1837) 


v_lshlrev_b32 v14, 4, v16 // (op 1837) 

v_lshl_add_u32 v16, s17, 5, v14 // (op 1837) 

v_mov_b32 v21, 0 // (op 1837) convert
v_mov_b32 v20, v16 // (op 1837) convert

v_mul_lo_u32 v17, v15, s25 // (op 1837) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 1837) most significant: high of low * low
v_mul_lo_u32 v16, v15, s24 // (op 1837) least significant: low of low * low
v_add_u32 v17, v17, v14 // (op 1837) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v20 // (op 1837) least significant half
v_addc_co_u32 v17, s[0:1], v17, v21, s[0:1] // (op 1837) most significant half



v_mul_lo_u32 v15, v17, 6 // (op 1837) most significant: low of high * low
v_mul_hi_u32 v20, v16, 6 // (op 1837) most significant: high of low * low
v_mul_lo_u32 v14, v16, 6 // (op 1837) least significant: low of low * low
v_add_u32 v15, v15, v20 // (op 1837) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 1837) 

v_mov_b32 v14, v16 // (op 1837) convert

s_mul_i32 s1, s23, 6 // (op 1839) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 1839) least significant
s_mul_hi_u32 s6, s22, 6 // (op 1839) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1839) most significant: sum

s_mul_i32 s7, s24, s1 // (op 1839) most significant: low of low * high
s_mul_i32 s14, s25, s0 // (op 1839) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 1839) least significant
s_mul_hi_u32 s15, s24, s0 // (op 1839) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 1839) most significant: sum
s_add_u32 s7, s7, s15 // (op 1839) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 1839) 

s_mov_b32 s48, s26 // (op 1839) reg expression
s_mov_b32 s49, s27 // (op 1839) reg expression
s_mov_b32 s50, s0 // (op 1839) reg expression
s_mov_b32 s51, 131072 // (op 1839) call()

v_and_b32 v15, -128, v1 // (op 1901) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 1901) 
v_sub_u32 v16, v1, v15 // (op 1901) 

s_lshr_b32 s6, s0, 1 // (op 1901) 

v_lshl_add_u32 v15, s16, 7, v16 // (op 1901) 

v_lshrrev_b32 v16, 7, v1 // (op 1901) 
v_lshlrev_b32 v17, 1, v16 // (op 1901) 

v_lshl_add_u32 v16, s6, 3, v17 // (op 1901) 


v_mul_lo_u32 v17, v16, s28 // (op 1901) 

v_add_u32 v16, v15, v17 // (op 1901) 


s_mul_i32 s1, s28, s33 // (op 1903) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 1903) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1903) least significant
s_mul_hi_u32 s7, s28, s32 // (op 1903) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1903) most significant: sum
s_add_u32 s1, s1, s7 // (op 1903) most significant: sum


s_mov_b32 s52, s30 // (op 1903) reg expression
s_mov_b32 s53, s31 // (op 1903) reg expression
s_mov_b32 s54, s0 // (op 1903) reg expression
s_mov_b32 s55, 131072 // (op 1903) call()

v_and_b32 v15, 63, v1 // (op 1965) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 1965) 
v_lshrrev_b32 v17, 4, v15 // (op 1965) 

s_lshr_b32 s6, s0, 1 // (op 1965) 

v_lshrrev_b32 v15, 6, v1 // (op 1965) 
v_lshl_add_u32 v20, s6, 3, v17 // (op 1965) 


v_and_b32 v17, 1, v15 // (op 1965) 

v_and_b32 v15, 15, v1 // (op 1965) 
v_lshl_add_u32 v21, v17, 4, v15 // (op 1965) 


v_lshl_add_u32 v15, s17, 5, v21 // (op 1965) 

v_mul_lo_u32 v17, v20, s34 // (op 1965) 
v_add_u32 v17, v17, v15 // (op 1965) 


s_mul_i32 s1, s34, s33 // (op 1967) most significant: low of low * high
s_mul_i32 s6, s35, s32 // (op 1967) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 1967) least significant
s_mul_hi_u32 s7, s34, s32 // (op 1967) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1967) most significant: sum
s_add_u32 s1, s1, s7 // (op 1967) most significant: sum


s_mov_b32 s56, s36 // (op 1967) reg expression
s_mov_b32 s57, s37 // (op 1967) reg expression
s_mov_b32 s58, s0 // (op 1967) reg expression
s_mov_b32 s59, 131072 // (op 1967) call()

v_and_b32 v15, -8, v1 // (op 2058) 
v_sub_u32 v20, v1, v15 // (op 2058) 

v_lshlrev_b32 v15, 4, v20 // (op 2058) 

v_lshrrev_b32 v20, 3, v1 // (op 2058) 
v_lshl_add_u32 v21, v20, 7, v15 // (op 2058) 


s_nop 0
v_readfirstlane_b32 s0, v21 // (op 2058) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v15, -2, v1 // (op 2115) 
v_sub_u32 v20, v1, v15 // (op 2115) 

v_lshlrev_b32 v15, 4, v20 // (op 2115) 

v_lshrrev_b32 v20, 1, v1 // (op 2115) 
v_lshl_add_u32 v21, v20, 5, v15 // (op 2115) 


v_mul_lo_u32 v15, v21, 6 // (op 2115) 
v_lshrrev_b32 v20, 3, v15 // (op 2115) 

v_lshrrev_b32 v15, 4, v21 // (op 2115) 

v_lshlrev_b32 v21, 2, v15 // (op 2115) 

v_add_u32 v15, v20, v21 // (op 2115) 


v_and_b32 v20, -128, v1 // (op 2168) 
v_sub_u32 v21, v1, v20 // (op 2168) 

v_lshrrev_b32 v20, 7, v1 // (op 2168) 
v_lshl_add_u32 v22, v20, 8, v21 // (op 2168) 


s_ashr_i64 s[6:7], s[22:23], 7 // (op 640) 
s_and_b32 s1, s6, 4294967294 // (op 640) 

s_ashr_i64 s[6:7], s[22:23], 7 // (op 639) 
s_mov_b32 s14, s6 // (op 639) convert

s_cmp_lt_u32 s1, s14 // (op 639) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopBottom_KLoopTail_639 // (op 639) Condition: Top (jump to RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopBottom_KLoopTail_639 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopTop_KLoopTail_639:
s_barrier  // (op 2369) 
s_add_u32 s1, s1, 1 // (op 641) 

s_lshl_b64 s[6:7], s[12:13], 5 // (op 1738) 
v_mov_b64 v[18:19], s[6:7] // (op 1738) convert


v_mov_b64 v[20:21], v[12:13] // (op 1738) 
v_mov_b64 v[24:25], v[20:21] // (op 1738) reg expression
s_mov_b32 m0, s0 // (op 1738) reg expression
buffer_load_dwordx4 v24, s[8:11], 0 offen offset: 0 lds // (op 1738) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v18 // (op 1738) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 1738) most significant half

s_add_u32 m0, m0, 4096 // (op 1738) 
buffer_load_dwordx4 v24, s[8:11], 0 offen offset: 0 lds // (op 1738) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v18 // (op 1738) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 1738) most significant half

s_add_u32 m0, m0, 4096 // (op 1738) 
buffer_load_dwordx4 v24, s[8:11], 0 offen offset: 0 lds // (op 1738) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v18 // (op 1738) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 1738) most significant half

s_add_u32 m0, m0, 4096 // (op 1738) 
buffer_load_dwordx4 v24, s[8:11], 0 offen offset: 0 lds // (op 1738) Load value direct to lds



s_lshl_b64 s[6:7], s[12:13], 7 // (op 1757) 
v_mov_b32 v18, s6 // (op 1757) 
v_mov_b32 v19, s7 // (op 1757) 
v_add_co_u32 v12, s[14:15], v12, v18 // (op 1757) least significant half
v_addc_co_u32 v13, s[14:15], v13, v19, s[14:15] // (op 1757) most significant half






v_mov_b32 v18, v14 // (op 179) 
v_mov_b32 v19, v18 // (op 179) reg expression

buffer_load_dwordx3 v[24:26], v19, s[48:51], 0 offen // (op 179) Load value


s_lshl_b64 s[6:7], s[24:25], 7 // (op 1829) 
s_mul_i32 s15, s7, 6 // (op 1829) most significant: low of high * low
s_mul_i32 s14, s6, 6 // (op 1829) least significant
s_mul_hi_u32 s18, s6, 6 // (op 1829) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 1829) most significant: sum


s_lshr_b64 s[6:7], s[14:15], 3 // (op 1829) 

v_add_u32 v14, v14, s6 // (op 1829) 


s_barrier  // (op 2382) 
v_mov_b64 v[18:19], s[28:29] // (op 180) convert




v_mov_b32 v23, v16 // (op 180) 
v_mov_b32 v20, v23 // (op 180) reg expression
buffer_load_ubyte v28, v20, s[52:55], 0 offen offset: 0 // (op 180) Load value
v_add_u32 v20, v20, v18 // (op 180) 
s_waitcnt vmcnt(1)
ds_write_b96 v15, v[24:26] offset:32768 // (op 177) Store local data 
buffer_load_ubyte v29, v20, s[52:55], 0 offen offset: 0 // (op 180) Load value




s_lshl_b32 s6, s28, 2 // (op 1893) 
v_add_u32 v16, v16, s6 // (op 1893) 

s_barrier  // (op 2372) 
s_barrier  // (op 2396) 






ds_read_b64_tr_b8 v[30:31], v4 // (op 169) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v4 offset:1024 // (op 169) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v4 offset:8192 // (op 169) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v4 offset:9216 // (op 169) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v22, v28 offset:36864 // (op 178) Store local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 2386) 
s_waitcnt vmcnt(0)
ds_write_b8 v22, v29 offset:36992 // (op 178) Store local data 




ds_read_b96_tr_b6 v[24:26], v0 offset:32768 // (op 171) Transpose load local data 
ds_read_b96_tr_b6 v[27:29], v0 offset:33280 // (op 171) Transpose load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 2400) 


ds_read_u8 v23, v5 offset:36864 // (op 173) Load local data 

v_mov_b64 v[18:19], s[34:35] // (op 175) convert

v_mov_b32 v20, v17 // (op 175) 
v_mov_b32 v21, v20 // (op 175) reg expression
v_mov_b32 v38, v21 // (op 175) 
buffer_load_ubyte v39, v38, s[56:59], 0 offen offset: 0 // (op 175) Load value




s_lshl_b32 s6, s34, 2 // (op 1957) 
v_add_u32 v17, v17, s6 // (op 1957) 




ds_read_b64_tr_b8 v[40:41], v4 offset:32 // (op 321) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v4 offset:1056 // (op 321) Transpose load local data 
s_waitcnt vmcnt(0) lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[30:37], v[24:29], a[0:3], v23, v39 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 176) 
ds_read_b64_tr_b8 v[44:45], v4 offset:8224 // (op 321) Transpose load local data 


ds_read_b64_tr_b8 v[46:47], v4 offset:9248 // (op 321) Transpose load local data 


ds_read_u8 v23, v5 offset:36896 // (op 325) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[40:47], v[24:29], a[4:7], v23, v39 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 327) 
ds_read_b64_tr_b8 v[30:31], v4 offset:64 // (op 483) Transpose load local data 


ds_read_b64_tr_b8 v[32:33], v4 offset:1088 // (op 483) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v4 offset:8256 // (op 483) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v4 offset:9280 // (op 483) Transpose load local data 


ds_read_u8 v23, v5 offset:36928 // (op 487) Load local data 



ds_read_b64_tr_b8 v[40:41], v4 offset:96 // (op 645) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[30:37], v[24:29], a[8:11], v23, v39 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 489) 
ds_read_b64_tr_b8 v[42:43], v4 offset:1120 // (op 645) Transpose load local data 


ds_read_b64_tr_b8 v[44:45], v4 offset:8288 // (op 645) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v4 offset:9312 // (op 645) Transpose load local data 


ds_read_u8 v23, v5 offset:36960 // (op 649) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[40:47], v[24:29], a[12:15], v23, v39 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 651) 




s_ashr_i64 s[6:7], s[22:23], 7 // (op 639) 
s_mov_b32 s14, s6 // (op 639) convert

s_cmp_lt_u32 s1, s14 // (op 639) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopTop_KLoopTail_639 // (op 639) Condition: Bottom (jump to RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopTop_KLoopTail_639 if true)
RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopBottom_KLoopTail_639:






















s_barrier  // (op 2484) 






v_cmp_eq_f32 vcc, 0.00000, v10 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ConditionalFalse_0_beta_123
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
s_branch RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ConditionalBottom_0_beta_123
RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ConditionalFalse_0_beta_123:



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
RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ConditionalBottom_0_beta_123:












s_endpgm  // End of RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2
.LRR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_end:
.size RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2, .LRR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_end-RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2
  .amdhsa_next_free_vgpr 80
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 41984 // lds bytes
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
    DataType: F8
    DataTypeA: F8
    DataTypeB: F6
    DestDataType: s
    ComputeDataType: s
    HighPrecisionAccumulate: True
    TransposeA: False
    TransposeB: True
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
  - .name: RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2
    .symbol: RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 41984
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 60
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
      - .name: Tensor_2_size_1_11
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
      - .name: Tensor_2_stride_0_3
        .size: 8
        .offset: 72
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 80
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_stride_1_5
        .size: 8
        .offset: 88
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 96
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_size_1_6
        .size: 8
        .offset: 104
        .value_kind: by_value
      - .name: Tensor_7_stride_0_8
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

