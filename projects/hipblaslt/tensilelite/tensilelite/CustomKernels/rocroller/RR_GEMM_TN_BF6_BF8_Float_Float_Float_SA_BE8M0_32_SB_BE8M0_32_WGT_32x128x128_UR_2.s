// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2
.p2align 8
.type RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2,@function
RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 2037) 
v_and_b32 v3, 15, v1 // (op 2037) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 2037) 


v_and_b32 v0, 63, v1 // (op 2037) 
v_lshrrev_b32 v3, 4, v0 // (op 2037) 

v_lshlrev_b32 v0, 5, v3 // (op 2037) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 2037) 


v_mul_lo_u32 v0, v3, 6 // (op 2037) 

v_lshrrev_b32 v3, 3, v0 // (op 2037) 

v_lshrrev_b32 v0, 7, v1 // (op 2194) 
v_and_b32 v4, 15, v1 // (op 2194) 
v_lshl_add_u32 v5, v0, 4, v4 // (op 2194) 


v_and_b32 v0, 63, v1 // (op 2194) 
v_lshrrev_b32 v4, 4, v0 // (op 2194) 

v_lshlrev_b32 v0, 5, v4 // (op 2194) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 2194) 


v_mul_lo_u32 v0, v4, 6 // (op 2194) 

v_lshrrev_b32 v4, 3, v0 // (op 2194) 

v_lshrrev_b32 v0, 7, v1 // (op 1980) 
v_and_b32 v5, 63, v1 // (op 1980) 
v_lshrrev_b32 v6, 4, v5 // (op 1980) 
v_lshlrev_b32 v7, 2, v6 // (op 1980) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 1980) 


v_lshl_add_u32 v0, s16, 5, v6 // (op 1980) 

v_lshrrev_b32 v6, 2, v5 // (op 1980) 

v_and_b32 v5, 3, v6 // (op 1980) 

v_and_b32 v6, 3, v1 // (op 1980) 
v_lshl_add_u32 v7, v5, 2, v6 // (op 1980) 


v_lshrrev_b32 v5, 6, v1 // (op 1980) 
v_and_b32 v6, 1, v5 // (op 1980) 

v_lshl_add_u32 v5, v6, 4, v7 // (op 1980) 


v_lshl_add_u32 v6, s17, 7, v5 // (op 1980) 

v_mul_lo_u32 v5, v6, s2 // (op 1980) 

v_add_lshl_u32 v6, v0, v5, 2 // (op 1980) 


v_lshrrev_b32 v0, 6, v1 // (op 2094) 
v_and_b32 v5, 1, v0 // (op 2094) 

v_and_b32 v0, 15, v1 // (op 2094) 
v_lshl_add_u32 v7, v5, 4, v0 // (op 2094) 


v_and_b32 v0, 48, v1 // (op 2094) 
v_lshl_add_u32 v5, v7, 7, v0 // (op 2094) 


v_lshrrev_b32 v0, 6, v1 // (op 2152) 
v_and_b32 v7, 1, v0 // (op 2152) 

v_and_b32 v0, 15, v1 // (op 2152) 
s_mul_i32 s1, s2, s9 // (op 1982) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 1982) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 1982) least significant
s_mul_hi_u32 s19, s2, s8 // (op 1982) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1982) most significant: sum
s_add_u32 s1, s1, s19 // (op 1982) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 1982) 

v_lshl_add_u32 v8, v7, 4, v0 // (op 2152) 


v_and_b32 v0, 63, v1 // (op 2152) 
v_lshrrev_b32 v7, 4, v0 // (op 2152) 

v_lshl_add_u32 v0, v8, 2, v7 // (op 2152) 


v_lshrrev_b32 v7, 7, v1 // (op 2168) 
v_and_b32 v8, 63, v1 // (op 2168) 
v_lshrrev_b32 v9, 4, v8 // (op 2168) 
v_lshlrev_b32 v10, 2, v9 // (op 2168) 

v_lshl_add_u32 v9, v7, 4, v10 // (op 2168) 


v_lshl_add_u32 v7, s16, 5, v9 // (op 2168) 

v_lshrrev_b32 v9, 2, v8 // (op 2168) 

v_and_b32 v8, 3, v9 // (op 2168) 

v_and_b32 v9, 3, v1 // (op 2168) 
v_lshl_add_u32 v10, v8, 2, v9 // (op 2168) 


v_lshrrev_b32 v8, 6, v1 // (op 2168) 
v_and_b32 v9, 1, v8 // (op 2168) 

v_lshl_add_u32 v8, v9, 4, v10 // (op 2168) 


v_lshl_add_u32 v9, s17, 7, v8 // (op 2168) 

v_mul_lo_u32 v8, v9, s4 // (op 2168) 

v_add_lshl_u32 v9, v7, v8, 2 // (op 2168) 


v_lshrrev_b32 v7, 6, v1 // (op 2238) 
v_and_b32 v8, 1, v7 // (op 2238) 

v_and_b32 v7, 15, v1 // (op 2238) 
v_lshl_add_u32 v10, v8, 4, v7 // (op 2238) 


v_and_b32 v7, 48, v1 // (op 2238) 
v_lshl_add_u32 v8, v10, 7, v7 // (op 2238) 


v_lshrrev_b32 v7, 6, v1 // (op 2267) 
v_and_b32 v10, 1, v7 // (op 2267) 

v_and_b32 v7, 15, v1 // (op 2267) 
s_mul_i32 s1, s4, s9 // (op 2170) most significant: low of low * high
v_lshl_add_u32 v11, v10, 4, v7 // (op 2267) 


v_and_b32 v7, 63, v1 // (op 2267) 
v_lshrrev_b32 v10, 4, v7 // (op 2267) 

v_lshl_add_u32 v7, v11, 2, v10 // (op 2267) 


s_mul_i32 s38, s5, s8 // (op 2170) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 2170) least significant
s_mul_hi_u32 s39, s4, s8 // (op 2170) most significant: high of low * low
s_add_u32 s1, s1, s38 // (op 2170) most significant: sum
s_add_u32 s1, s1, s39 // (op 2170) most significant: sum


s_lshl_b64 s[38:39], s[0:1], 2 // (op 2170) 

s_mov_b32 s40, s6 // (op 1982) reg expression
s_mov_b32 s41, s7 // (op 1982) reg expression
s_mov_b32 s42, s18 // (op 1982) reg expression
s_mov_b32 s43, 131072 // (op 1982) call()

s_mov_b32 s44, s10 // (op 2170) reg expression

s_mov_b32 s45, s11 // (op 2170) reg expression
s_mov_b32 s46, s38 // (op 2170) reg expression
s_mov_b32 s47, 131072 // (op 2170) call()


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
v_lshrrev_b32 v12, 3, v1 // (op 1747) 
v_lshl_add_u32 v13, s16, 5, v12 // (op 1747) 

v_and_b32 v12, -8, v1 // (op 1747) 
v_sub_u32 v14, v1, v12 // (op 1747) 

v_lshlrev_b32 v12, 4, v14 // (op 1747) 

v_mov_b32 v15, 0 // (op 1747) convert
v_mov_b32 v14, v12 // (op 1747) convert

v_mul_lo_u32 v17, v13, s13 // (op 1747) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 1747) most significant: high of low * low
v_mul_lo_u32 v16, v13, s12 // (op 1747) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 1747) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 1747) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 1747) most significant half



v_mul_lo_u32 v13, v17, 6 // (op 1747) most significant: low of high * low
v_mul_hi_u32 v14, v16, 6 // (op 1747) most significant: high of low * low
v_mul_lo_u32 v12, v16, 6 // (op 1747) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 1747) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 1747) 

v_mov_b32 v12, v14 // (op 1747) convert


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s23, 6 // (op 1749) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 1749) least significant
s_mul_hi_u32 s6, s22, 6 // (op 1749) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1749) most significant: sum

s_mul_i32 s7, s12, s1 // (op 1749) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 1749) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 1749) least significant
s_mul_hi_u32 s11, s12, s0 // (op 1749) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 1749) most significant: sum
s_add_u32 s7, s7, s11 // (op 1749) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 1749) 

s_mov_b32 s48, s20 // (op 1749) reg expression
s_mov_b32 s49, s21 // (op 1749) reg expression
s_mov_b32 s50, s0 // (op 1749) reg expression
s_mov_b32 s51, 131072 // (op 1749) call()

v_lshrrev_b32 v13, 3, v1 // (op 1766) 
v_lshl_add_u32 v14, s16, 5, v13 // (op 1766) 

v_and_b32 v13, -8, v1 // (op 1766) 
v_sub_u32 v15, v1, v13 // (op 1766) 

v_lshlrev_b32 v13, 4, v15 // (op 1766) 

v_mov_b32 v17, 0 // (op 1766) convert
v_mov_b32 v16, v13 // (op 1766) convert

v_mul_lo_u32 v19, v14, s13 // (op 1766) most significant: low of low * high
v_mul_hi_u32 v13, v14, s12 // (op 1766) most significant: high of low * low
v_mul_lo_u32 v18, v14, s12 // (op 1766) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 1766) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 1766) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 1766) most significant half



v_mul_lo_u32 v17, v19, 6 // (op 1766) most significant: low of high * low
v_mul_hi_u32 v13, v18, 6 // (op 1766) most significant: high of low * low
v_mul_lo_u32 v16, v18, 6 // (op 1766) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 1766) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 1766) 

v_mov_b32 v13, v14 // (op 1766) convert

s_mul_i32 s1, s23, 6 // (op 1768) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 1768) least significant
s_mul_hi_u32 s6, s22, 6 // (op 1768) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1768) most significant: sum

s_mul_i32 s7, s12, s1 // (op 1768) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 1768) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 1768) least significant
s_mul_hi_u32 s11, s12, s0 // (op 1768) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 1768) most significant: sum
s_add_u32 s7, s7, s11 // (op 1768) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 1768) 

s_mov_b32 s48, s20 // (op 1768) reg expression
s_mov_b32 s49, s21 // (op 1768) reg expression
s_mov_b32 s50, s0 // (op 1768) reg expression
s_mov_b32 s51, 131072 // (op 1768) call()

v_and_b32 v14, -8, v1 // (op 1811) 
v_sub_u32 v15, v1, v14 // (op 1811) 

v_lshlrev_b32 v14, 4, v15 // (op 1811) 

v_mov_b32 v17, 0 // (op 1811) convert
v_mov_b32 v16, v14 // (op 1811) convert

v_lshrrev_b32 v14, 3, v1 // (op 1811) 
v_lshl_add_u32 v15, s17, 7, v14 // (op 1811) 

v_mul_lo_u32 v19, v15, s25 // (op 1811) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 1811) most significant: high of low * low
v_mul_lo_u32 v18, v15, s24 // (op 1811) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 1811) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v18 // (op 1811) least significant half
v_addc_co_u32 v15, s[0:1], v17, v19, s[0:1] // (op 1811) most significant half



s_mul_i32 s1, s24, s9 // (op 1813) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 1813) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 1813) least significant
s_mul_hi_u32 s7, s24, s8 // (op 1813) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1813) most significant: sum
s_add_u32 s1, s1, s7 // (op 1813) most significant: sum


s_mov_b32 s52, s26 // (op 1813) reg expression
s_mov_b32 s53, s27 // (op 1813) reg expression
s_mov_b32 s54, s0 // (op 1813) reg expression
s_mov_b32 s55, 131072 // (op 1813) call()

v_and_b32 v16, -8, v1 // (op 1830) 
v_sub_u32 v17, v1, v16 // (op 1830) 

v_lshlrev_b32 v16, 4, v17 // (op 1830) 

v_mov_b32 v19, 0 // (op 1830) convert
v_mov_b32 v18, v16 // (op 1830) convert

v_lshrrev_b32 v16, 3, v1 // (op 1830) 
v_lshl_add_u32 v17, s17, 7, v16 // (op 1830) 

v_mul_lo_u32 v21, v17, s25 // (op 1830) most significant: low of low * high
v_mul_hi_u32 v16, v17, s24 // (op 1830) most significant: high of low * low
v_mul_lo_u32 v20, v17, s24 // (op 1830) least significant: low of low * low
v_add_u32 v21, v21, v16 // (op 1830) most significant: sum


v_add_co_u32 v16, s[0:1], v18, v20 // (op 1830) least significant half
v_addc_co_u32 v17, s[0:1], v19, v21, s[0:1] // (op 1830) most significant half



s_mul_i32 s1, s24, s9 // (op 1832) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 1832) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 1832) least significant
s_mul_hi_u32 s7, s24, s8 // (op 1832) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1832) most significant: sum
s_add_u32 s1, s1, s7 // (op 1832) most significant: sum


s_mov_b32 s52, s26 // (op 1832) reg expression
s_mov_b32 s53, s27 // (op 1832) reg expression
s_mov_b32 s54, s0 // (op 1832) reg expression
s_mov_b32 s55, 131072 // (op 1832) call()

v_lshrrev_b32 v18, 7, v1 // (op 1875) 
v_and_b32 v19, 15, v1 // (op 1875) 
v_lshl_add_u32 v20, v18, 4, v19 // (op 1875) 


v_lshl_add_u32 v18, s16, 5, v20 // (op 1875) 

v_and_b32 v19, 63, v1 // (op 1875) 
v_lshrrev_b32 v20, 4, v19 // (op 1875) 

v_mul_lo_u32 v19, v18, s28 // (op 1875) 
v_add_u32 v19, v19, v20 // (op 1875) 


s_mul_i32 s1, s28, s23 // (op 1877) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 1877) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 1877) least significant
s_mul_hi_u32 s7, s28, s22 // (op 1877) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1877) most significant: sum
s_add_u32 s1, s1, s7 // (op 1877) most significant: sum


s_mov_b32 s56, s30 // (op 1877) reg expression
s_mov_b32 s57, s31 // (op 1877) reg expression
s_mov_b32 s58, s0 // (op 1877) reg expression
s_mov_b32 s59, 131072 // (op 1877) call()

v_lshrrev_b32 v18, 7, v1 // (op 1894) 
v_and_b32 v20, 15, v1 // (op 1894) 
v_lshl_add_u32 v21, v18, 4, v20 // (op 1894) 


v_lshl_add_u32 v18, s16, 5, v21 // (op 1894) 

v_and_b32 v20, 63, v1 // (op 1894) 
v_lshrrev_b32 v21, 4, v20 // (op 1894) 

v_mul_lo_u32 v20, v18, s28 // (op 1894) 
v_add_u32 v20, v20, v21 // (op 1894) 


s_mul_i32 s1, s28, s23 // (op 1896) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 1896) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 1896) least significant
s_mul_hi_u32 s7, s28, s22 // (op 1896) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1896) most significant: sum
s_add_u32 s1, s1, s7 // (op 1896) most significant: sum


s_mov_b32 s56, s30 // (op 1896) reg expression
s_mov_b32 s57, s31 // (op 1896) reg expression
s_mov_b32 s58, s0 // (op 1896) reg expression
s_mov_b32 s59, 131072 // (op 1896) call()

v_and_b32 v18, -4, v1 // (op 1939) 
v_sub_u32 v21, v1, v18 // (op 1939) 

v_lshrrev_b32 v18, 2, v1 // (op 1939) 
v_lshlrev_b32 v22, 1, v18 // (op 1939) 

v_lshl_add_u32 v18, s17, 7, v22 // (op 1939) 

v_mul_lo_u32 v22, v18, s32 // (op 1939) 

v_add_u32 v18, v21, v22 // (op 1939) 


s_mul_i32 s1, s32, s9 // (op 1941) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 1941) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 1941) least significant
s_mul_hi_u32 s7, s32, s8 // (op 1941) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1941) most significant: sum
s_add_u32 s1, s1, s7 // (op 1941) most significant: sum


s_mov_b32 s60, s34 // (op 1941) reg expression
s_mov_b32 s61, s35 // (op 1941) reg expression
s_mov_b32 s62, s0 // (op 1941) reg expression
s_mov_b32 s63, 131072 // (op 1941) call()

v_and_b32 v21, -4, v1 // (op 1958) 
v_sub_u32 v22, v1, v21 // (op 1958) 

v_lshrrev_b32 v21, 2, v1 // (op 1958) 
v_lshlrev_b32 v23, 1, v21 // (op 1958) 

v_lshl_add_u32 v21, s17, 7, v23 // (op 1958) 

v_mul_lo_u32 v23, v21, s32 // (op 1958) 

v_add_u32 v21, v22, v23 // (op 1958) 


s_mul_i32 s1, s32, s9 // (op 1960) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 1960) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 1960) least significant
s_mul_hi_u32 s7, s32, s8 // (op 1960) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1960) most significant: sum
s_add_u32 s1, s1, s7 // (op 1960) most significant: sum


s_mov_b32 s60, s34 // (op 1960) reg expression
s_mov_b32 s61, s35 // (op 1960) reg expression
s_mov_b32 s62, s0 // (op 1960) reg expression
s_mov_b32 s63, 131072 // (op 1960) call()




buffer_load_ubyte v22, v19, s[56:59], 0 offen offset: 0 // (op 1544) Load value
v_and_b32 v23, -8, v1 // (op 2077) 
v_sub_u32 v24, v1, v23 // (op 2077) 

v_lshlrev_b32 v23, 4, v24 // (op 2077) 

v_lshrrev_b32 v24, 3, v1 // (op 2077) 
v_lshl_add_u32 v25, v24, 7, v23 // (op 2077) 


s_nop 0
v_readfirstlane_b32 s0, v25 // (op 2077) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 1700) 
v_mov_b64 v[24:25], s[6:7] // (op 1700) convert


v_mov_b64 v[26:27], v[14:15] // (op 1700) 
s_mov_b32 m0, s0 // (op 1700) reg expression
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 1700) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v24 // (op 1700) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 1700) most significant half

s_add_u32 m0, m0, 4096 // (op 1700) 
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 1700) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v24 // (op 1700) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 1700) most significant half

s_add_u32 m0, m0, 4096 // (op 1700) 
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 1700) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v24 // (op 1700) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 1700) most significant half

s_add_u32 m0, m0, 4096 // (op 1700) 
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 1700) Load value direct to lds




v_mov_b64 v[24:25], s[32:33] // (op 1551) convert

v_mov_b32 v23, v18 // (op 1551) 
buffer_load_ubyte v26, v23, s[60:63], 0 offen offset: 0 // (op 1551) Load value
v_add_u32 v23, v23, v24 // (op 1551) 
buffer_load_ubyte v27, v23, s[60:63], 0 offen offset: 0 // (op 1551) Load value





buffer_load_dwordx3 v[28:30], v12, s[48:51], 0 offen // (op 1554) Load value



buffer_load_ubyte v23, v19, s[56:59], 0 offen offset:4 // (op 1558) Load value

v_and_b32 v19, -8, v1 // (op 2221) 
v_sub_u32 v24, v1, v19 // (op 2221) 

v_lshlrev_b32 v19, 4, v24 // (op 2221) 

v_lshrrev_b32 v24, 3, v1 // (op 2221) 
v_lshl_add_u32 v25, v24, 7, v19 // (op 2221) 


s_nop 0
v_readfirstlane_b32 s0, v25 // (op 2221) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 1705) 
v_mov_b64 v[24:25], s[6:7] // (op 1705) convert


v_mov_b64 v[32:33], v[14:15] // (op 1705) 
s_add_u32 s1, 16384, s0 // (op 1705) 
s_mov_b32 m0, s1 // (op 1705) reg expression
s_mov_b32 s6, 128 // (op 1705) call()
buffer_load_dwordx4 v32, s[52:55], s6 offen offset: 0 lds // (op 1705) Load value direct to lds
v_add_co_u32 v32, s[10:11], v32, v24 // (op 1705) least significant half
v_addc_co_u32 v33, s[10:11], 0, v33, s[10:11] // (op 1705) most significant half

s_add_u32 m0, m0, 4096 // (op 1705) 
buffer_load_dwordx4 v32, s[52:55], s6 offen offset: 0 lds // (op 1705) Load value direct to lds
v_add_co_u32 v32, s[10:11], v32, v24 // (op 1705) least significant half
v_addc_co_u32 v33, s[10:11], 0, v33, s[10:11] // (op 1705) most significant half

s_add_u32 m0, m0, 4096 // (op 1705) 
buffer_load_dwordx4 v32, s[52:55], s6 offen offset: 0 lds // (op 1705) Load value direct to lds
v_add_co_u32 v32, s[10:11], v32, v24 // (op 1705) least significant half
v_addc_co_u32 v33, s[10:11], 0, v33, s[10:11] // (op 1705) most significant half

s_add_u32 m0, m0, 4096 // (op 1705) 
buffer_load_dwordx4 v32, s[52:55], s6 offen offset: 0 lds // (op 1705) Load value direct to lds







v_mov_b64 v[14:15], s[32:33] // (op 1565) convert

v_mov_b32 v19, v18 // (op 1565) 
buffer_load_ubyte v24, v19, s[60:63], 0 offen offset:4 // (op 1565) Load value
v_add_u32 v19, v19, v14 // (op 1565) 
buffer_load_ubyte v25, v19, s[60:63], 0 offen offset:4 // (op 1565) Load value






buffer_load_dwordx3 v[32:34], v12, s[48:51], 0 offen offset:96 // (op 1568) Load value

v_and_b32 v12, -4, v1 // (op 2135) 
v_sub_u32 v31, v1, v12 // (op 2135) 

v_lshrrev_b32 v12, 2, v1 // (op 2135) 
v_lshl_add_u32 v14, v12, 3, v31 // (op 2135) 





s_waitcnt vmcnt(10)
ds_write_b8 v14, v26 offset:32768 // (op 1571) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v14, v27 offset:32772 // (op 1571) Store local data 

v_and_b32 v12, -8, v1 // (op 2025) 
v_sub_u32 v31, v1, v12 // (op 2025) 

v_lshlrev_b32 v12, 4, v31 // (op 2025) 

v_lshrrev_b32 v31, 3, v1 // (op 2025) 
v_lshl_add_u32 v14, v31, 7, v12 // (op 2025) 


v_mul_lo_u32 v12, v14, 6 // (op 2025) 

v_lshrrev_b32 v31, 3, v12 // (op 2025) 




s_waitcnt vmcnt(8)
ds_write_b96 v31, v[28:30] offset:33280 // (op 1572) Store local data 

v_and_b32 v12, -8, v1 // (op 1996) 
v_sub_u32 v31, v1, v12 // (op 1996) 

v_lshlrev_b32 v12, 4, v31 // (op 1996) 

v_lshrrev_b32 v31, 3, v1 // (op 1996) 
v_lshl_add_u32 v14, v31, 7, v12 // (op 1996) 


v_mul_lo_u32 v12, v14, 6 // (op 1996) 

v_lshrrev_b32 v31, 3, v12 // (op 1996) 

v_and_b32 v12, -8, v1 // (op 2049) 
v_sub_u32 v14, v1, v12 // (op 2049) 

v_lshlrev_b32 v12, 4, v14 // (op 2049) 

v_lshrrev_b32 v14, 3, v1 // (op 2049) 
v_lshl_add_u32 v15, v14, 7, v12 // (op 2049) 


s_nop 0
v_readfirstlane_b32 s0, v15 // (op 2049) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -4, v1 // (op 2106) 
v_sub_u32 v14, v1, v12 // (op 2106) 

v_lshrrev_b32 v12, 2, v1 // (op 2106) 
v_lshl_add_u32 v15, v12, 3, v14 // (op 2106) 


v_and_b32 v12, -8, v1 // (op 2182) 
v_sub_u32 v14, v1, v12 // (op 2182) 

v_lshlrev_b32 v12, 4, v14 // (op 2182) 

v_lshrrev_b32 v14, 3, v1 // (op 2182) 
v_lshl_add_u32 v18, v14, 7, v12 // (op 2182) 


v_mul_lo_u32 v12, v18, 6 // (op 2182) 

v_lshrrev_b32 v14, 3, v12 // (op 2182) 

v_and_b32 v12, -8, v1 // (op 2206) 
v_sub_u32 v18, v1, v12 // (op 2206) 

v_lshlrev_b32 v12, 4, v18 // (op 2206) 

v_lshrrev_b32 v18, 3, v1 // (op 2206) 
v_lshl_add_u32 v19, v18, 7, v12 // (op 2206) 


s_nop 0
v_readfirstlane_b32 s1, v19 // (op 2206) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -4, v1 // (op 2250) 
v_sub_u32 v18, v1, v12 // (op 2250) 

v_lshrrev_b32 v12, 2, v1 // (op 2250) 
v_lshl_add_u32 v19, v12, 3, v18 // (op 2250) 


s_mov_b32 s6, 0 // (op 620) call()
s_ashr_i64 s[10:11], s[36:37], 7 // (op 619) 
s_mov_b32 s7, s10 // (op 619) convert

s_and_b32 s10, s7, 4294967294 // (op 619) 

s_cmp_lt_u32 s6, s10 // (op 619) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopBottom_KLoop_619 // (op 619) Condition: Top (jump to RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopBottom_KLoop_619 if false)
RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopTop_KLoop_619:
s_add_u32 s6, s6, 2 // (op 621) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 2314) 



ds_read_b128 v[36:39], v3 offset:33280 // (op 65) Load local data 
ds_read_b64 v[40:41], v3 offset:33296 // (op 65) Load local data 
s_barrier  // (op 2327) 



ds_read_b128 v[42:45], v5 // (op 70) Load local data 
ds_read_b128 v[46:49], v5 offset:64 // (op 70) Load local data 
s_barrier  // (op 2340) 



ds_read_u8 v12, v0 offset:32768 // (op 79) Load local data 



ds_read_b128 v[50:53], v5 offset:4096 // (op 339) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:41], v[42:49], a[0:3], v22, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 82) 
ds_read_b128 v[54:57], v5 offset:4160 // (op 339) Load local data 





ds_read_u8 v12, v0 offset:32896 // (op 342) Load local data 



ds_read_b128 v[42:45], v5 offset:8192 // (op 495) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[36:41], v[50:57], a[4:7], v22, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 343) 
ds_read_b128 v[46:49], v5 offset:8256 // (op 495) Load local data 





ds_read_u8 v12, v0 offset:33024 // (op 498) Load local data 



ds_read_b128 v[50:53], v5 offset:12288 // (op 651) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[36:41], v[42:49], a[8:11], v22, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 499) 
ds_read_b128 v[54:57], v5 offset:12352 // (op 651) Load local data 





ds_read_u8 v12, v0 offset:33152 // (op 654) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[36:41], v[50:57], a[12:15], v22, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 655) 





v_mov_b32 v12, v20 // (op 75) 

buffer_load_ubyte v22, v12, s[56:59], 0 offen offset:8 // (op 75) Load value
s_barrier  // (op 2330) 

s_lshl_b64 s[10:11], s[24:25], 5 // (op 1683) 
v_mov_b64 v[36:37], s[10:11] // (op 1683) convert


v_mov_b64 v[38:39], v[16:17] // (op 1683) 
v_mov_b64 v[40:41], v[38:39] // (op 1683) 
s_mov_b32 m0, s0 // (op 1683) reg expression
s_mov_b32 s7, 256 // (op 1683) call()
buffer_load_dwordx4 v40, s[52:55], s7 offen offset: 0 lds // (op 1683) Load value direct to lds
v_add_co_u32 v40, s[10:11], v40, v36 // (op 1683) least significant half
v_addc_co_u32 v41, s[10:11], 0, v41, s[10:11] // (op 1683) most significant half

s_add_u32 m0, m0, 4096 // (op 1683) 
buffer_load_dwordx4 v40, s[52:55], s7 offen offset: 0 lds // (op 1683) Load value direct to lds
v_add_co_u32 v40, s[10:11], v40, v36 // (op 1683) least significant half
v_addc_co_u32 v41, s[10:11], 0, v41, s[10:11] // (op 1683) most significant half

s_add_u32 m0, m0, 4096 // (op 1683) 
buffer_load_dwordx4 v40, s[52:55], s7 offen offset: 0 lds // (op 1683) Load value direct to lds
v_add_co_u32 v40, s[10:11], v40, v36 // (op 1683) least significant half
v_addc_co_u32 v41, s[10:11], 0, v41, s[10:11] // (op 1683) most significant half

s_add_u32 m0, m0, 4096 // (op 1683) 
buffer_load_dwordx4 v40, s[52:55], s7 offen offset: 0 lds // (op 1683) Load value direct to lds




v_mov_b64 v[36:37], s[32:33] // (op 21) convert

v_mov_b32 v18, v21 // (op 21) 
v_mov_b32 v35, v18 // (op 21) 
buffer_load_ubyte v26, v35, s[60:63], 0 offen offset:8 // (op 21) Load value
v_add_u32 v35, v35, v36 // (op 21) 
buffer_load_ubyte v27, v35, s[60:63], 0 offen offset:8 // (op 21) Load value




v_mov_b32 v35, v13 // (op 4) 

buffer_load_dwordx3 v[28:30], v35, s[48:51], 0 offen offset:192 // (op 4) Load value
s_barrier  // (op 2369) 



s_waitcnt vmcnt(10)
ds_write_b8 v19, v24 offset:36352 // (op 236) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v19, v25 offset:36356 // (op 236) Store local data 
s_barrier  // (op 2357) 



s_waitcnt vmcnt(8)
ds_write_b96 v14, v[32:34] offset:36864 // (op 225) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 2354) 



ds_read_b128 v[40:43], v4 offset:36864 // (op 227) Load local data 
ds_read_b64 v[44:45], v4 offset:36880 // (op 227) Load local data 
s_barrier  // (op 2360) 
s_barrier  // (op 2400) 



ds_read_b128 v[46:49], v8 offset:16384 // (op 229) Load local data 
ds_read_b128 v[50:53], v8 offset:16448 // (op 229) Load local data 
s_barrier  // (op 2366) 



ds_read_u8 v36, v7 offset:36352 // (op 233) Load local data 



ds_read_b128 v[54:57], v8 offset:20480 // (op 348) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[40:45], v[46:53], a[0:3], v23, v36 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 234) 
ds_read_b128 v[58:61], v8 offset:20544 // (op 348) Load local data 





ds_read_u8 v36, v7 offset:36480 // (op 351) Load local data 



ds_read_b128 v[46:49], v8 offset:24576 // (op 504) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[40:45], v[54:61], a[4:7], v23, v36 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 344) 
ds_read_b128 v[50:53], v8 offset:24640 // (op 504) Load local data 





ds_read_u8 v36, v7 offset:36608 // (op 507) Load local data 



ds_read_b128 v[54:57], v8 offset:28672 // (op 660) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[40:45], v[46:53], a[8:11], v23, v36 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 500) 
ds_read_b128 v[58:61], v8 offset:28736 // (op 660) Load local data 





ds_read_u8 v36, v7 offset:36736 // (op 663) Load local data 
s_barrier  // (op 2448) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[40:45], v[54:61], a[12:15], v23, v36 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 656) 






buffer_load_ubyte v23, v12, s[56:59], 0 offen offset:12 // (op 231) Load value

s_barrier  // (op 2363) 
v_add_u32 v20, 8, v20 // (op 1886) 

s_lshl_b64 s[10:11], s[24:25], 5 // (op 1695) 
v_mov_b64 v[36:37], s[10:11] // (op 1695) convert


v_mov_b64 v[40:41], v[38:39] // (op 1695) 
s_add_u32 s7, 16384, s1 // (op 1695) 
s_mov_b32 m0, s7 // (op 1695) reg expression
s_mov_b32 s10, 384 // (op 1695) call()
buffer_load_dwordx4 v40, s[52:55], s10 offen offset: 0 lds // (op 1695) Load value direct to lds
v_add_co_u32 v40, s[14:15], v40, v36 // (op 1695) least significant half
v_addc_co_u32 v41, s[14:15], 0, v41, s[14:15] // (op 1695) most significant half

s_add_u32 m0, m0, 4096 // (op 1695) 
buffer_load_dwordx4 v40, s[52:55], s10 offen offset: 0 lds // (op 1695) Load value direct to lds
v_add_co_u32 v40, s[14:15], v40, v36 // (op 1695) least significant half
v_addc_co_u32 v41, s[14:15], 0, v41, s[14:15] // (op 1695) most significant half

s_add_u32 m0, m0, 4096 // (op 1695) 
buffer_load_dwordx4 v40, s[52:55], s10 offen offset: 0 lds // (op 1695) Load value direct to lds
v_add_co_u32 v40, s[14:15], v40, v36 // (op 1695) least significant half
v_addc_co_u32 v41, s[14:15], 0, v41, s[14:15] // (op 1695) most significant half

s_add_u32 m0, m0, 4096 // (op 1695) 
buffer_load_dwordx4 v40, s[52:55], s10 offen offset: 0 lds // (op 1695) Load value direct to lds






v_add_co_u32 v16, vcc, 256, v16 // (op 1822) least significant half
v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 1822) most significant half
v_mov_b64 v[36:37], s[32:33] // (op 238) convert

v_mov_b32 v12, v18 // (op 238) 
buffer_load_ubyte v24, v12, s[60:63], 0 offen offset:12 // (op 238) Load value
v_add_u32 v12, v12, v36 // (op 238) 
buffer_load_ubyte v25, v12, s[60:63], 0 offen offset:12 // (op 238) Load value






v_add_u32 v21, 8, v21 // (op 1950) 
buffer_load_dwordx3 v[32:34], v35, s[48:51], 0 offen offset:288 // (op 224) Load value

s_barrier  // (op 2343) 
v_add_u32 v13, 192, v13 // (op 1758) 



s_waitcnt vmcnt(10)
ds_write_b8 v15, v26 offset:32768 // (op 26) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v15, v27 offset:32772 // (op 26) Store local data 
s_barrier  // (op 2317) 



s_waitcnt vmcnt(8)
ds_write_b96 v31, v[28:30] offset:33280 // (op 6) Store local data 
s_ashr_i64 s[10:11], s[36:37], 7 // (op 619) 
s_mov_b32 s7, s10 // (op 619) convert

s_and_b32 s10, s7, 4294967294 // (op 619) 

s_cmp_lt_u32 s6, s10 // (op 619) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopTop_KLoop_619 // (op 619) Condition: Bottom (jump to RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopTop_KLoop_619 if true)
RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopBottom_KLoop_619:














s_barrier  // (op 2430) 
v_lshrrev_b32 v12, 3, v1 // (op 1718) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1718) 
v_lshl_add_u32 v13, s16, 5, v12 // (op 1718) 

s_lshr_b32 s6, s0, 1 // (op 1718) 

v_and_b32 v12, -8, v1 // (op 1718) 
v_sub_u32 v14, v1, v12 // (op 1718) 

v_lshlrev_b32 v12, 4, v14 // (op 1718) 

v_lshl_add_u32 v14, s6, 8, v12 // (op 1718) 


v_mov_b32 v27, 0 // (op 1718) convert
v_mov_b32 v26, v14 // (op 1718) convert

v_mul_lo_u32 v15, v13, s13 // (op 1718) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 1718) most significant: high of low * low
v_mul_lo_u32 v14, v13, s12 // (op 1718) least significant: low of low * low
v_add_u32 v15, v15, v12 // (op 1718) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v26 // (op 1718) least significant half
v_addc_co_u32 v15, s[0:1], v15, v27, s[0:1] // (op 1718) most significant half



v_mul_lo_u32 v13, v15, 6 // (op 1718) most significant: low of high * low
v_mul_hi_u32 v16, v14, 6 // (op 1718) most significant: high of low * low
v_mul_lo_u32 v12, v14, 6 // (op 1718) least significant: low of low * low
v_add_u32 v13, v13, v16 // (op 1718) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 1718) 

v_mov_b32 v12, v14 // (op 1718) convert

s_mul_i32 s1, s23, 6 // (op 1720) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 1720) least significant
s_mul_hi_u32 s6, s22, 6 // (op 1720) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1720) most significant: sum

s_mul_i32 s7, s12, s1 // (op 1720) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 1720) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 1720) least significant
s_mul_hi_u32 s11, s12, s0 // (op 1720) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 1720) most significant: sum
s_add_u32 s7, s7, s11 // (op 1720) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 1720) 

s_mov_b32 s48, s20 // (op 1720) reg expression
s_mov_b32 s49, s21 // (op 1720) reg expression
s_mov_b32 s50, s0 // (op 1720) reg expression
s_mov_b32 s51, 131072 // (op 1720) call()

v_and_b32 v13, -8, v1 // (op 1788) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1788) 
v_sub_u32 v14, v1, v13 // (op 1788) 

s_lshr_b32 s6, s0, 1 // (op 1788) 

v_lshlrev_b32 v13, 4, v14 // (op 1788) 

v_lshl_add_u32 v14, s6, 8, v13 // (op 1788) 


v_mov_b32 v27, 0 // (op 1788) convert
v_mov_b32 v26, v14 // (op 1788) convert

v_lshrrev_b32 v13, 3, v1 // (op 1788) 
v_lshl_add_u32 v14, s17, 7, v13 // (op 1788) 

v_mul_lo_u32 v29, v14, s25 // (op 1788) most significant: low of low * high
v_mul_hi_u32 v13, v14, s24 // (op 1788) most significant: high of low * low
v_mul_lo_u32 v28, v14, s24 // (op 1788) least significant: low of low * low
v_add_u32 v29, v29, v13 // (op 1788) most significant: sum


v_add_co_u32 v30, s[0:1], v26, v28 // (op 1788) least significant half
v_addc_co_u32 v31, s[0:1], v27, v29, s[0:1] // (op 1788) most significant half



s_mul_i32 s1, s24, s9 // (op 1790) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 1790) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 1790) least significant
s_mul_hi_u32 s7, s24, s8 // (op 1790) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1790) most significant: sum
s_add_u32 s1, s1, s7 // (op 1790) most significant: sum


s_mov_b32 s52, s26 // (op 1790) reg expression
s_mov_b32 s53, s27 // (op 1790) reg expression
s_mov_b32 s54, s0 // (op 1790) reg expression
s_mov_b32 s55, 131072 // (op 1790) call()

v_lshrrev_b32 v13, 7, v1 // (op 1852) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1852) 
v_and_b32 v14, 15, v1 // (op 1852) 
s_lshr_b32 s6, s0, 1 // (op 1852) 

v_lshl_add_u32 v15, v13, 4, v14 // (op 1852) 


v_lshl_add_u32 v13, s16, 5, v15 // (op 1852) 

v_and_b32 v14, 63, v1 // (op 1852) 
v_lshrrev_b32 v15, 4, v14 // (op 1852) 

v_lshl_add_u32 v14, s6, 3, v15 // (op 1852) 


v_mul_lo_u32 v15, v13, s28 // (op 1852) 
v_add_u32 v15, v15, v14 // (op 1852) 


s_mul_i32 s1, s28, s23 // (op 1854) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 1854) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 1854) least significant
s_mul_hi_u32 s7, s28, s22 // (op 1854) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1854) most significant: sum
s_add_u32 s1, s1, s7 // (op 1854) most significant: sum


s_mov_b32 s56, s30 // (op 1854) reg expression
s_mov_b32 s57, s31 // (op 1854) reg expression
s_mov_b32 s58, s0 // (op 1854) reg expression
s_mov_b32 s59, 131072 // (op 1854) call()

v_and_b32 v13, -4, v1 // (op 1916) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1916) 
v_sub_u32 v14, v1, v13 // (op 1916) 

s_lshr_b32 s6, s0, 1 // (op 1916) 

v_lshrrev_b32 v13, 2, v1 // (op 1916) 
v_lshl_add_u32 v16, s6, 3, v14 // (op 1916) 


v_lshlrev_b32 v14, 1, v13 // (op 1916) 

v_lshl_add_u32 v13, s17, 7, v14 // (op 1916) 

v_mul_lo_u32 v14, v13, s32 // (op 1916) 

v_add_u32 v13, v16, v14 // (op 1916) 


s_mul_i32 s1, s32, s9 // (op 1918) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 1918) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 1918) least significant
s_mul_hi_u32 s7, s32, s8 // (op 1918) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1918) most significant: sum
s_add_u32 s1, s1, s7 // (op 1918) most significant: sum


s_mov_b32 s60, s34 // (op 1918) reg expression
s_mov_b32 s61, s35 // (op 1918) reg expression
s_mov_b32 s62, s0 // (op 1918) reg expression
s_mov_b32 s63, 131072 // (op 1918) call()

v_and_b32 v14, -8, v1 // (op 2009) 
v_sub_u32 v16, v1, v14 // (op 2009) 

v_lshlrev_b32 v14, 4, v16 // (op 2009) 

v_lshrrev_b32 v16, 3, v1 // (op 2009) 
v_lshl_add_u32 v17, v16, 7, v14 // (op 2009) 


v_mul_lo_u32 v14, v17, 6 // (op 2009) 

v_lshrrev_b32 v16, 3, v14 // (op 2009) 

v_and_b32 v14, -8, v1 // (op 2062) 
v_sub_u32 v17, v1, v14 // (op 2062) 

v_lshlrev_b32 v14, 4, v17 // (op 2062) 

v_lshrrev_b32 v17, 3, v1 // (op 2062) 
v_lshl_add_u32 v18, v17, 7, v14 // (op 2062) 


s_nop 0
v_readfirstlane_b32 s0, v18 // (op 2062) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 2119) 
v_sub_u32 v17, v1, v14 // (op 2119) 

v_lshrrev_b32 v14, 2, v1 // (op 2119) 
v_lshl_add_u32 v18, v14, 3, v17 // (op 2119) 


s_ashr_i64 s[6:7], s[36:37], 7 // (op 625) 
s_and_b32 s1, s6, 4294967294 // (op 625) 

s_ashr_i64 s[6:7], s[36:37], 7 // (op 624) 
s_mov_b32 s10, s6 // (op 624) convert

s_cmp_lt_u32 s1, s10 // (op 624) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopBottom_KLoopTail_624 // (op 624) Condition: Top (jump to RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopBottom_KLoopTail_624 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopTop_KLoopTail_624:


s_add_u32 s1, s1, 1 // (op 626) 
v_mov_b32 v14, v12 // (op 166) 

buffer_load_dwordx3 v[20:22], v14, s[48:51], 0 offen // (op 166) Load value

v_add_u32 v12, 96, v12 // (op 1708) 
s_barrier  // (op 2320) 
s_barrier  // (op 2333) 




s_lshl_b64 s[6:7], s[24:25], 5 // (op 1688) 
v_mov_b64 v[28:29], s[6:7] // (op 1688) convert


v_mov_b64 v[26:27], v[30:31] // (op 1688) 
v_mov_b64 v[24:25], v[26:27] // (op 1688) 
s_mov_b32 m0, s0 // (op 1688) reg expression
buffer_load_dwordx4 v24, s[52:55], 0 offen offset: 0 lds // (op 1688) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v28 // (op 1688) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 1688) most significant half

s_add_u32 m0, m0, 4096 // (op 1688) 
buffer_load_dwordx4 v24, s[52:55], 0 offen offset: 0 lds // (op 1688) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v28 // (op 1688) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 1688) most significant half

s_add_u32 m0, m0, 4096 // (op 1688) 
s_waitcnt vmcnt(2)
ds_write_b96 v16, v[20:22] offset:33280 // (op 167) Store local data 
buffer_load_dwordx4 v24, s[52:55], 0 offen offset: 0 lds // (op 1688) Load value direct to lds

v_add_co_u32 v24, s[6:7], v24, v28 // (op 1688) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 1688) most significant half

s_add_u32 m0, m0, 4096 // (op 1688) 
buffer_load_dwordx4 v24, s[52:55], 0 offen offset: 0 lds // (op 1688) Load value direct to lds



v_add_co_u32 v30, vcc, 128, v30 // (op 1780) least significant half
v_addc_co_u32 v31, vcc, 0, v31, vcc // (op 1780) most significant half

v_mov_b64 v[28:29], s[32:33] // (op 180) convert

v_mov_b32 v14, v13 // (op 180) 
v_mov_b32 v17, v14 // (op 180) 
buffer_load_ubyte v26, v17, s[60:63], 0 offen offset: 0 // (op 180) Load value
v_add_u32 v17, v17, v28 // (op 180) 
buffer_load_ubyte v27, v17, s[60:63], 0 offen offset: 0 // (op 180) Load value



v_add_u32 v13, 4, v13 // (op 1908) 
s_barrier  // (op 2346) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 2323) 






ds_read_b128 v[20:23], v3 offset:33280 // (op 169) Load local data 
ds_read_b64 v[24:25], v3 offset:33296 // (op 169) Load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v18, v26 offset:32768 // (op 178) Store local data 
s_barrier  // (op 2336) 
s_waitcnt vmcnt(0)
ds_write_b8 v18, v27 offset:32772 // (op 178) Store local data 




ds_read_b128 v[32:35], v5 // (op 171) Load local data 
ds_read_b128 v[36:39], v5 offset:64 // (op 171) Load local data 


v_mov_b32 v14, v15 // (op 173) 

buffer_load_ubyte v17, v14, s[56:59], 0 offen offset: 0 // (op 173) Load value

v_add_u32 v15, 4, v15 // (op 1844) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 2350) 



ds_read_u8 v14, v0 offset:32768 // (op 175) Load local data 



ds_read_b128 v[40:43], v5 offset:4096 // (op 320) Load local data 
ds_read_b128 v[44:47], v5 offset:4160 // (op 320) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:25], v[32:39], a[0:3], v17, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 176) 





ds_read_u8 v14, v0 offset:32896 // (op 323) Load local data 



ds_read_b128 v[32:35], v5 offset:8192 // (op 476) Load local data 
ds_read_b128 v[36:39], v5 offset:8256 // (op 476) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[20:25], v[40:47], a[4:7], v17, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 324) 





ds_read_u8 v14, v0 offset:33024 // (op 479) Load local data 



ds_read_b128 v[40:43], v5 offset:12288 // (op 632) Load local data 
ds_read_b128 v[44:47], v5 offset:12352 // (op 632) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[20:25], v[32:39], a[8:11], v17, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 480) 





ds_read_u8 v14, v0 offset:33152 // (op 635) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[20:25], v[40:47], a[12:15], v17, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 636) 




s_ashr_i64 s[6:7], s[36:37], 7 // (op 624) 
s_mov_b32 s10, s6 // (op 624) convert

s_cmp_lt_u32 s1, s10 // (op 624) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopTop_KLoopTail_624 // (op 624) Condition: Bottom (jump to RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopTop_KLoopTail_624 if true)
RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopBottom_KLoopTail_624:






















s_barrier  // (op 2434) 







v_cmp_eq_f32 vcc, 0.00000, v10 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ConditionalFalse_0_beta_123
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
s_branch RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ConditionalBottom_0_beta_123
RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ConditionalFalse_0_beta_123:


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
RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ConditionalBottom_0_beta_123:












s_endpgm  // End of RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2
.LRR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_end:
.size RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2, .LRR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_end-RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2
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
  - .name: RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2
    .symbol: RR_GEMM_TN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2.kd
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

