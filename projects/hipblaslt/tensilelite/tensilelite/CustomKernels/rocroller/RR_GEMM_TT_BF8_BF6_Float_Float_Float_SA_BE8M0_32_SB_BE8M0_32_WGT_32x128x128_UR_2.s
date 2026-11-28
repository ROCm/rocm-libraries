// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2
.p2align 8
.type RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2,@function
RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_exec_begin:


s_load_dwordx2 s[20:21], s[0:1], 56 // Load scalar value
s_load_dwordx2 s[22:23], s[0:1], 64 // Load scalar value
s_load_dwordx2 s[24:25], s[0:1], 72 // Load scalar value
s_load_dwordx2 s[26:27], s[0:1], 80 // Load scalar value
s_load_dwordx2 s[28:29], s[0:1], 88 // Load scalar value
s_load_dwordx2 s[30:31], s[0:1], 96 // Load scalar value
s_load_dwordx2 s[32:33], s[0:1], 104 // Load scalar value
s_load_dwordx2 s[34:35], s[0:1], 112 // Load scalar value
s_load_dwordx2 s[36:37], s[0:1], 120 // Load scalar value
s_load_dwordx2 s[38:39], s[0:1], 128 // Load scalar value

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


v_and_b32 v0, 63, v1 // (op 2093) 
v_and_b32 v5, 48, v1 // (op 2093) 
v_sub_u32 v7, v0, v5 // (op 2093) 

v_lshrrev_b32 v5, 4, v0 // (op 2093) 

v_lshl_add_u32 v0, v5, 5, v7 // (op 2093) 


v_lshrrev_b32 v5, 6, v1 // (op 2093) 
v_and_b32 v7, 1, v5 // (op 2093) 

v_lshlrev_b32 v5, 4, v7 // (op 2093) 

v_lshl_add_u32 v7, v0, 7, v5 // (op 2093) 


v_mul_lo_u32 v0, v7, 6 // (op 2093) 
v_lshrrev_b32 v5, 3, v0 // (op 2093) 

v_lshrrev_b32 v0, 4, v7 // (op 2093) 

v_lshlrev_b32 v7, 2, v0 // (op 2093) 

v_add_u32 v0, v5, v7 // (op 2093) 


v_lshrrev_b32 v5, 6, v1 // (op 2151) 
v_and_b32 v7, 1, v5 // (op 2151) 

v_and_b32 v5, 15, v1 // (op 2151) 
s_mul_i32 s1, s2, s9 // (op 1981) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 1981) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 1981) least significant
s_mul_hi_u32 s19, s2, s8 // (op 1981) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1981) most significant: sum
s_add_u32 s1, s1, s19 // (op 1981) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 1981) 

v_lshl_add_u32 v8, v7, 4, v5 // (op 2151) 


v_and_b32 v5, 63, v1 // (op 2151) 
v_lshrrev_b32 v7, 4, v5 // (op 2151) 

v_lshl_add_u32 v5, v7, 7, v8 // (op 2151) 


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


v_and_b32 v7, 63, v1 // (op 2237) 
v_and_b32 v8, 48, v1 // (op 2237) 
v_sub_u32 v10, v7, v8 // (op 2237) 

v_lshrrev_b32 v8, 4, v7 // (op 2237) 

v_lshl_add_u32 v7, v8, 5, v10 // (op 2237) 


v_lshrrev_b32 v8, 6, v1 // (op 2237) 
v_and_b32 v10, 1, v8 // (op 2237) 

v_lshlrev_b32 v8, 4, v10 // (op 2237) 

v_lshl_add_u32 v10, v7, 7, v8 // (op 2237) 


v_mul_lo_u32 v7, v10, 6 // (op 2237) 
v_lshrrev_b32 v8, 3, v7 // (op 2237) 

v_lshrrev_b32 v7, 4, v10 // (op 2237) 

v_lshlrev_b32 v10, 2, v7 // (op 2237) 

v_add_u32 v7, v8, v10 // (op 2237) 


v_lshrrev_b32 v8, 6, v1 // (op 2266) 
v_and_b32 v10, 1, v8 // (op 2266) 

v_and_b32 v8, 15, v1 // (op 2266) 
s_mul_i32 s1, s4, s9 // (op 2169) most significant: low of low * high
v_lshl_add_u32 v11, v10, 4, v8 // (op 2266) 


v_and_b32 v8, 63, v1 // (op 2266) 
v_lshrrev_b32 v10, 4, v8 // (op 2266) 

v_lshl_add_u32 v8, v10, 7, v11 // (op 2266) 


s_mul_i32 s40, s5, s8 // (op 2169) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 2169) least significant
s_mul_hi_u32 s41, s4, s8 // (op 2169) most significant: high of low * low
s_add_u32 s1, s1, s40 // (op 2169) most significant: sum
s_add_u32 s1, s1, s41 // (op 2169) most significant: sum


s_lshl_b64 s[40:41], s[0:1], 2 // (op 2169) 

s_mov_b32 s44, s6 // (op 1981) reg expression
s_mov_b32 s45, s7 // (op 1981) reg expression
s_mov_b32 s46, s18 // (op 1981) reg expression
s_mov_b32 s47, 131072 // (op 1981) call()

s_mov_b32 s48, s10 // (op 2169) reg expression

s_mov_b32 s49, s11 // (op 2169) reg expression
s_mov_b32 s50, s40 // (op 2169) reg expression
s_mov_b32 s51, 131072 // (op 2169) call()



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


s_mov_b32 s8, s20 // (op 1748) reg expression
s_mov_b32 s9, s21 // (op 1748) reg expression
s_mov_b32 s10, s0 // (op 1748) reg expression
s_mov_b32 s11, 131072 // (op 1748) call()

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


s_mov_b32 s8, s20 // (op 1767) reg expression
s_mov_b32 s9, s21 // (op 1767) reg expression
s_mov_b32 s10, s0 // (op 1767) reg expression
s_mov_b32 s11, 131072 // (op 1767) call()

v_and_b32 v12, -2, v1 // (op 1810) 
v_sub_u32 v13, v1, v12 // (op 1810) 

v_lshlrev_b32 v12, 6, v13 // (op 1810) 

v_lshl_add_u32 v13, s17, 7, v12 // (op 1810) 

v_mov_b32 v15, 0 // (op 1810) convert
v_mov_b32 v14, v13 // (op 1810) convert

v_lshrrev_b32 v12, 1, v1 // (op 1810) 
v_mul_lo_u32 v21, v12, s25 // (op 1810) most significant: low of low * high
v_mul_hi_u32 v13, v12, s24 // (op 1810) most significant: high of low * low
v_mul_lo_u32 v20, v12, s24 // (op 1810) least significant: low of low * low
v_add_u32 v21, v21, v13 // (op 1810) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v14 // (op 1810) least significant half
v_addc_co_u32 v21, s[0:1], v21, v15, s[0:1] // (op 1810) most significant half



v_mul_lo_u32 v13, v21, 6 // (op 1810) most significant: low of high * low
v_mul_hi_u32 v14, v20, 6 // (op 1810) most significant: high of low * low
v_mul_lo_u32 v12, v20, 6 // (op 1810) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 1810) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 1810) 

v_mov_b32 v12, v14 // (op 1810) convert

s_mul_i32 s1, s29, 6 // (op 1812) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 1812) least significant
s_mul_hi_u32 s6, s28, 6 // (op 1812) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1812) most significant: sum

s_mul_i32 s7, s24, s1 // (op 1812) most significant: low of low * high
s_mul_i32 s14, s25, s0 // (op 1812) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 1812) least significant
s_mul_hi_u32 s15, s24, s0 // (op 1812) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 1812) most significant: sum
s_add_u32 s7, s7, s15 // (op 1812) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 1812) 

s_mov_b32 s40, s26 // (op 1812) reg expression
s_mov_b32 s41, s27 // (op 1812) reg expression
s_mov_b32 s42, s0 // (op 1812) reg expression
s_mov_b32 s43, 131072 // (op 1812) call()

v_and_b32 v13, -2, v1 // (op 1829) 
v_sub_u32 v14, v1, v13 // (op 1829) 

v_lshlrev_b32 v13, 6, v14 // (op 1829) 

v_lshl_add_u32 v14, s17, 7, v13 // (op 1829) 

v_mov_b32 v21, 0 // (op 1829) convert
v_mov_b32 v20, v14 // (op 1829) convert

v_lshrrev_b32 v13, 1, v1 // (op 1829) 
v_mul_lo_u32 v15, v13, s25 // (op 1829) most significant: low of low * high
v_mul_hi_u32 v22, v13, s24 // (op 1829) most significant: high of low * low
v_mul_lo_u32 v14, v13, s24 // (op 1829) least significant: low of low * low
v_add_u32 v15, v15, v22 // (op 1829) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v20 // (op 1829) least significant half
v_addc_co_u32 v15, s[0:1], v15, v21, s[0:1] // (op 1829) most significant half



v_mul_lo_u32 v21, v15, 6 // (op 1829) most significant: low of high * low
v_mul_hi_u32 v13, v14, 6 // (op 1829) most significant: high of low * low
v_mul_lo_u32 v20, v14, 6 // (op 1829) least significant: low of low * low
v_add_u32 v21, v21, v13 // (op 1829) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[20:21] // (op 1829) 

v_mov_b32 v13, v14 // (op 1829) convert

s_mul_i32 s1, s29, 6 // (op 1831) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 1831) least significant
s_mul_hi_u32 s6, s28, 6 // (op 1831) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1831) most significant: sum

s_mul_i32 s7, s24, s1 // (op 1831) most significant: low of low * high
s_mul_i32 s14, s25, s0 // (op 1831) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 1831) least significant
s_mul_hi_u32 s15, s24, s0 // (op 1831) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 1831) most significant: sum
s_add_u32 s7, s7, s15 // (op 1831) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 1831) 

s_mov_b32 s40, s26 // (op 1831) reg expression
s_mov_b32 s41, s27 // (op 1831) reg expression
s_mov_b32 s42, s0 // (op 1831) reg expression
s_mov_b32 s43, 131072 // (op 1831) call()

v_lshrrev_b32 v14, 7, v1 // (op 1874) 
v_and_b32 v15, 15, v1 // (op 1874) 
v_lshl_add_u32 v20, v14, 4, v15 // (op 1874) 


v_lshl_add_u32 v14, s16, 5, v20 // (op 1874) 

v_and_b32 v15, 63, v1 // (op 1874) 
v_lshrrev_b32 v20, 4, v15 // (op 1874) 

v_mul_lo_u32 v15, v14, s30 // (op 1874) 
v_add_u32 v15, v15, v20 // (op 1874) 


s_mul_i32 s1, s30, s23 // (op 1876) most significant: low of low * high
s_mul_i32 s6, s31, s22 // (op 1876) most significant: low of high * low
s_mul_i32 s0, s30, s22 // (op 1876) least significant
s_mul_hi_u32 s7, s30, s22 // (op 1876) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1876) most significant: sum
s_add_u32 s1, s1, s7 // (op 1876) most significant: sum


s_mov_b32 s52, s32 // (op 1876) reg expression
s_mov_b32 s53, s33 // (op 1876) reg expression
s_mov_b32 s54, s0 // (op 1876) reg expression
s_mov_b32 s55, 131072 // (op 1876) call()

v_lshrrev_b32 v14, 7, v1 // (op 1893) 
v_and_b32 v20, 15, v1 // (op 1893) 
v_lshl_add_u32 v21, v14, 4, v20 // (op 1893) 


v_lshl_add_u32 v14, s16, 5, v21 // (op 1893) 

v_and_b32 v20, 63, v1 // (op 1893) 
v_lshrrev_b32 v21, 4, v20 // (op 1893) 

v_mul_lo_u32 v20, v14, s30 // (op 1893) 
v_add_u32 v20, v20, v21 // (op 1893) 


s_mul_i32 s1, s30, s23 // (op 1895) most significant: low of low * high
s_mul_i32 s6, s31, s22 // (op 1895) most significant: low of high * low
s_mul_i32 s0, s30, s22 // (op 1895) least significant
s_mul_hi_u32 s7, s30, s22 // (op 1895) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1895) most significant: sum
s_add_u32 s1, s1, s7 // (op 1895) most significant: sum


s_mov_b32 s52, s32 // (op 1895) reg expression
s_mov_b32 s53, s33 // (op 1895) reg expression
s_mov_b32 s54, s0 // (op 1895) reg expression
s_mov_b32 s55, 131072 // (op 1895) call()

v_and_b32 v14, -128, v1 // (op 1938) 
v_sub_u32 v21, v1, v14 // (op 1938) 

v_lshl_add_u32 v14, s17, 7, v21 // (op 1938) 

v_lshrrev_b32 v21, 7, v1 // (op 1938) 
v_lshlrev_b32 v22, 1, v21 // (op 1938) 

v_mul_lo_u32 v21, v22, s34 // (op 1938) 
v_add_u32 v21, v21, v14 // (op 1938) 


s_mul_i32 s1, s34, s39 // (op 1940) most significant: low of low * high
s_mul_i32 s6, s35, s38 // (op 1940) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 1940) least significant
s_mul_hi_u32 s7, s34, s38 // (op 1940) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1940) most significant: sum
s_add_u32 s1, s1, s7 // (op 1940) most significant: sum


s_mov_b32 s56, s36 // (op 1940) reg expression
s_mov_b32 s57, s37 // (op 1940) reg expression
s_mov_b32 s58, s0 // (op 1940) reg expression
s_mov_b32 s59, 131072 // (op 1940) call()

v_and_b32 v14, -128, v1 // (op 1957) 
v_sub_u32 v22, v1, v14 // (op 1957) 

v_lshl_add_u32 v14, s17, 7, v22 // (op 1957) 

v_lshrrev_b32 v22, 7, v1 // (op 1957) 
v_lshlrev_b32 v23, 1, v22 // (op 1957) 

v_mul_lo_u32 v22, v23, s34 // (op 1957) 
v_add_u32 v22, v22, v14 // (op 1957) 


s_mul_i32 s1, s34, s39 // (op 1959) most significant: low of low * high
s_mul_i32 s6, s35, s38 // (op 1959) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 1959) least significant
s_mul_hi_u32 s7, s34, s38 // (op 1959) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1959) most significant: sum
s_add_u32 s1, s1, s7 // (op 1959) most significant: sum


s_mov_b32 s56, s36 // (op 1959) reg expression
s_mov_b32 s57, s37 // (op 1959) reg expression
s_mov_b32 s58, s0 // (op 1959) reg expression
s_mov_b32 s59, 131072 // (op 1959) call()




buffer_load_ubyte v14, v15, s[52:55], 0 offen offset: 0 // (op 1544) Load value


v_mov_b32 v23, v12 // (op 1548) reg expression

buffer_load_dwordx3 v[24:26], v23, s[40:43], 0 offen // (op 1548) Load value
buffer_load_dwordx3 v[28:30], v23, s[40:43], 0 offen offset:12 // (op 1548) Load value
buffer_load_dwordx3 v[32:34], v23, s[40:43], 0 offen offset:24 // (op 1548) Load value
buffer_load_dwordx3 v[36:38], v23, s[40:43], 0 offen offset:36 // (op 1548) Load value


v_mov_b64 v[40:41], s[34:35] // (op 1551) convert

v_mov_b32 v23, v21 // (op 1551) reg expression
buffer_load_ubyte v42, v23, s[56:59], 0 offen offset: 0 // (op 1551) Load value
v_add_u32 v23, v23, v40 // (op 1551) 
buffer_load_ubyte v43, v23, s[56:59], 0 offen offset: 0 // (op 1551) Load value


v_and_b32 v23, -8, v1 // (op 2023) 
v_sub_u32 v27, v1, v23 // (op 2023) 

v_lshlrev_b32 v23, 4, v27 // (op 2023) 

v_lshrrev_b32 v27, 3, v1 // (op 2023) 
v_lshl_add_u32 v31, v27, 7, v23 // (op 2023) 


s_nop 0
v_readfirstlane_b32 s0, v31 // (op 2023) Wait state hazard: VALU Write Readlane Hazard




s_nop 1
s_mov_b32 m0, s0 // (op 1699) reg expression
buffer_load_dwordx4 v16, s[8:11], 0 offen offset: 0 lds // (op 1699) Load value direct to lds




buffer_load_ubyte v23, v15, s[52:55], 0 offen offset:4 // (op 1558) Load value



s_lshl_b64 s[0:1], s[24:25], 7 // (op 1562) 
s_mul_i32 s7, s1, 6 // (op 1562) most significant: low of high * low
s_mul_i32 s6, s0, 6 // (op 1562) least significant
s_mul_hi_u32 s14, s0, 6 // (op 1562) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 1562) most significant: sum


s_lshr_b64 s[0:1], s[6:7], 3 // (op 1562) 

v_add_u32 v15, v12, s0 // (op 1562) 


buffer_load_dwordx3 v[44:46], v15, s[40:43], 0 offen // (op 1562) Load value
buffer_load_dwordx3 v[48:50], v15, s[40:43], 0 offen offset:12 // (op 1562) Load value
buffer_load_dwordx3 v[52:54], v15, s[40:43], 0 offen offset:24 // (op 1562) Load value
buffer_load_dwordx3 v[56:58], v15, s[40:43], 0 offen offset:36 // (op 1562) Load value



v_mov_b64 v[40:41], s[34:35] // (op 1565) convert

s_lshl_b32 s0, s34, 2 // (op 1565) 
v_add_u32 v12, v21, s0 // (op 1565) 

buffer_load_ubyte v60, v12, s[56:59], 0 offen offset: 0 // (op 1565) Load value
v_add_u32 v12, v12, v40 // (op 1565) 
buffer_load_ubyte v61, v12, s[56:59], 0 offen offset: 0 // (op 1565) Load value



v_and_b32 v12, -8, v1 // (op 2196) 
v_sub_u32 v15, v1, v12 // (op 2196) 

v_lshlrev_b32 v12, 4, v15 // (op 2196) 

v_lshrrev_b32 v15, 3, v1 // (op 2196) 
v_lshl_add_u32 v21, v15, 7, v12 // (op 2196) 


s_nop 0
v_readfirstlane_b32 s0, v21 // (op 2196) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s1, 4096, s0 // (op 1704) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 1704) reg expression
s_mov_b32 s6, 128 // (op 1704) call()
buffer_load_dwordx4 v16, s[8:11], s6 offen offset: 0 lds // (op 1704) Load value direct to lds




v_and_b32 v12, -2, v1 // (op 2076) 
v_sub_u32 v21, v1, v12 // (op 2076) 

v_lshlrev_b32 v12, 6, v21 // (op 2076) 

v_lshrrev_b32 v21, 1, v1 // (op 2076) 
v_lshl_add_u32 v27, v21, 7, v12 // (op 2076) 


v_mul_lo_u32 v12, v27, 6 // (op 2076) 
v_lshrrev_b32 v21, 3, v12 // (op 2076) 

v_lshrrev_b32 v12, 4, v27 // (op 2076) 

v_lshlrev_b32 v27, 2, v12 // (op 2076) 

v_add_u32 v12, v21, v27 // (op 2076) 





s_waitcnt vmcnt(14)
ds_write_b96 v12, v[24:26] offset:8192 // (op 1570) Store local data 
s_waitcnt vmcnt(13)
ds_write_b96 v12, v[28:30] offset:8208 // (op 1570) Store local data 
s_waitcnt vmcnt(12)
ds_write_b96 v12, v[32:34] offset:8224 // (op 1570) Store local data 
s_waitcnt vmcnt(11)
ds_write_b96 v12, v[36:38] offset:8240 // (op 1570) Store local data 

v_and_b32 v12, -128, v1 // (op 2134) 
v_sub_u32 v21, v1, v12 // (op 2134) 

v_lshrrev_b32 v12, 7, v1 // (op 2134) 
v_lshl_add_u32 v27, v12, 8, v21 // (op 2134) 





s_waitcnt vmcnt(10)
ds_write_b8 v27, v42 offset:24576 // (op 1571) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v27, v43 offset:24704 // (op 1571) Store local data 

v_and_b32 v12, -8, v1 // (op 1995) 
v_sub_u32 v21, v1, v12 // (op 1995) 

v_lshlrev_b32 v12, 4, v21 // (op 1995) 

v_lshrrev_b32 v21, 3, v1 // (op 1995) 
v_lshl_add_u32 v27, v21, 7, v12 // (op 1995) 


s_nop 0
v_readfirstlane_b32 s0, v27 // (op 1995) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -2, v1 // (op 2047) 
v_sub_u32 v21, v1, v12 // (op 2047) 

v_lshlrev_b32 v12, 6, v21 // (op 2047) 

v_lshrrev_b32 v21, 1, v1 // (op 2047) 
v_lshl_add_u32 v27, v21, 7, v12 // (op 2047) 


v_mul_lo_u32 v12, v27, 6 // (op 2047) 
v_lshrrev_b32 v21, 3, v12 // (op 2047) 

v_lshrrev_b32 v12, 4, v27 // (op 2047) 

v_lshlrev_b32 v27, 2, v12 // (op 2047) 

v_add_u32 v12, v21, v27 // (op 2047) 


v_and_b32 v21, -128, v1 // (op 2105) 
v_sub_u32 v27, v1, v21 // (op 2105) 

v_lshrrev_b32 v21, 7, v1 // (op 2105) 
v_lshl_add_u32 v31, v21, 8, v27 // (op 2105) 


v_and_b32 v21, -8, v1 // (op 2181) 
v_sub_u32 v27, v1, v21 // (op 2181) 

v_lshlrev_b32 v21, 4, v27 // (op 2181) 

v_lshrrev_b32 v27, 3, v1 // (op 2181) 
v_lshl_add_u32 v35, v27, 7, v21 // (op 2181) 


s_nop 0
v_readfirstlane_b32 s1, v35 // (op 2181) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v21, -2, v1 // (op 2220) 
v_sub_u32 v27, v1, v21 // (op 2220) 

v_lshlrev_b32 v21, 6, v27 // (op 2220) 

v_lshrrev_b32 v27, 1, v1 // (op 2220) 
v_lshl_add_u32 v35, v27, 7, v21 // (op 2220) 


v_mul_lo_u32 v21, v35, 6 // (op 2220) 
v_lshrrev_b32 v27, 3, v21 // (op 2220) 

v_lshrrev_b32 v21, 4, v35 // (op 2220) 

v_lshlrev_b32 v35, 2, v21 // (op 2220) 

v_add_u32 v21, v27, v35 // (op 2220) 


v_and_b32 v27, -128, v1 // (op 2249) 
v_sub_u32 v35, v1, v27 // (op 2249) 

v_lshrrev_b32 v27, 7, v1 // (op 2249) 
v_lshl_add_u32 v47, v27, 8, v35 // (op 2249) 


s_mov_b32 s6, 0 // (op 620) call()
s_ashr_i64 s[14:15], s[28:29], 7 // (op 619) 
s_mov_b32 s7, s14 // (op 619) convert

s_and_b32 s14, s7, 4294967294 // (op 619) 

s_cmp_lt_u32 s6, s14 // (op 619) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopBottom_KLoop_619 // (op 619) Condition: Top (jump to RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopBottom_KLoop_619 if false)
RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopTop_KLoop_619:
s_add_u32 s6, s6, 2 // (op 621) 
s_waitcnt vmcnt(8)
s_barrier  // (op 2313) 



ds_read_b128 v[62:65], v3 // (op 65) Load local data 
ds_read_b128 v[66:69], v3 offset:64 // (op 65) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 2326) 



ds_read_b96_tr_b6 v[70:72], v0 offset:8192 // (op 70) Transpose load local data 
ds_read_b96_tr_b6 v[73:75], v0 offset:10240 // (op 70) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 2340) 


ds_read_u8 v27, v5 offset:24576 // (op 79) Load local data 



ds_read_b96_tr_b6 v[76:78], v0 offset:8224 // (op 339) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:69], v[70:75], a[0:3], v14, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 82) 
ds_read_b96_tr_b6 v[79:81], v0 offset:10272 // (op 339) Transpose load local data 




ds_read_u8 v27, v5 offset:24608 // (op 342) Load local data 



ds_read_b96_tr_b6 v[70:72], v0 offset:8256 // (op 495) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[62:69], v[76:81], a[4:7], v14, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 343) 
ds_read_b96_tr_b6 v[73:75], v0 offset:10304 // (op 495) Transpose load local data 




ds_read_u8 v27, v5 offset:24640 // (op 498) Load local data 



ds_read_b96_tr_b6 v[76:78], v0 offset:8288 // (op 651) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[62:69], v[70:75], a[8:11], v14, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 499) 
ds_read_b96_tr_b6 v[79:81], v0 offset:10336 // (op 651) Transpose load local data 




ds_read_u8 v27, v5 offset:24672 // (op 654) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[62:69], v[76:81], a[12:15], v14, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 655) 





v_mov_b32 v27, v20 // (op 75) 

buffer_load_ubyte v14, v27, s[52:55], 0 offen offset:8 // (op 75) Load value


v_mov_b32 v35, v13 // (op 10) 
s_lshl_b64 s[14:15], s[24:25], 7 // (op 10) 
s_mul_i32 s19, s15, 6 // (op 10) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 10) least significant
s_mul_hi_u32 s7, s14, 6 // (op 10) most significant: high of low * low
s_add_u32 s19, s19, s7 // (op 10) most significant: sum


s_lshr_b64 s[14:15], s[18:19], 3 // (op 10) 

s_lshl_b32 s7, s14, 1 // (op 10) 

v_add_u32 v51, v35, s7 // (op 10) 


buffer_load_dwordx3 v[24:26], v51, s[40:43], 0 offen // (op 10) Load value
buffer_load_dwordx3 v[28:30], v51, s[40:43], 0 offen offset:12 // (op 10) Load value
buffer_load_dwordx3 v[32:34], v51, s[40:43], 0 offen offset:24 // (op 10) Load value
buffer_load_dwordx3 v[36:38], v51, s[40:43], 0 offen offset:36 // (op 10) Load value


v_mov_b64 v[16:17], s[34:35] // (op 21) convert

v_mov_b32 v15, v22 // (op 21) 
s_lshl_b32 s7, s34, 3 // (op 21) 
v_add_u32 v51, v15, s7 // (op 21) 

buffer_load_ubyte v42, v51, s[56:59], 0 offen offset: 0 // (op 21) Load value
v_add_u32 v51, v51, v16 // (op 21) 
buffer_load_ubyte v43, v51, s[56:59], 0 offen offset: 0 // (op 21) Load value


s_barrier  // (op 2316) 


v_mov_b64 v[16:17], v[18:19] // (op 1683) 

s_mov_b32 m0, s0 // (op 1683) reg expression
s_mov_b32 s7, 256 // (op 1683) call()
buffer_load_dwordx4 v16, s[8:11], s7 offen offset: 0 lds // (op 1683) Load value direct to lds

s_barrier  // (op 2363) 



s_waitcnt vmcnt(14)
ds_write_b96 v21, v[44:46] offset:25088 // (op 235) Store local data 
s_waitcnt vmcnt(13)
ds_write_b96 v21, v[48:50] offset:25104 // (op 235) Store local data 
s_waitcnt vmcnt(12)
ds_write_b96 v21, v[52:54] offset:25120 // (op 235) Store local data 
s_waitcnt vmcnt(11)
ds_write_b96 v21, v[56:58] offset:25136 // (op 235) Store local data 
s_barrier  // (op 2369) 



s_waitcnt vmcnt(10)
ds_write_b8 v47, v60 offset:41472 // (op 236) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v47, v61 offset:41600 // (op 236) Store local data 
s_waitcnt vmcnt(8)
s_barrier  // (op 2354) 



ds_read_b128 v[62:65], v4 offset:4096 // (op 227) Load local data 
ds_read_b128 v[66:69], v4 offset:4160 // (op 227) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 2360) 



ds_read_b96_tr_b6 v[70:72], v7 offset:25088 // (op 229) Transpose load local data 
ds_read_b96_tr_b6 v[73:75], v7 offset:27136 // (op 229) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 2366) 


ds_read_u8 v51, v8 offset:41472 // (op 233) Load local data 



ds_read_b96_tr_b6 v[76:78], v7 offset:25120 // (op 348) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:69], v[70:75], a[0:3], v23, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 234) 
ds_read_b96_tr_b6 v[79:81], v7 offset:27168 // (op 348) Transpose load local data 




ds_read_u8 v51, v8 offset:41504 // (op 351) Load local data 



ds_read_b96_tr_b6 v[70:72], v7 offset:25152 // (op 504) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[62:69], v[76:81], a[4:7], v23, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 344) 
ds_read_b96_tr_b6 v[73:75], v7 offset:27200 // (op 504) Transpose load local data 




ds_read_u8 v51, v8 offset:41536 // (op 507) Load local data 



ds_read_b96_tr_b6 v[76:78], v7 offset:25184 // (op 660) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[62:69], v[70:75], a[8:11], v23, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 500) 
ds_read_b96_tr_b6 v[79:81], v7 offset:27232 // (op 660) Transpose load local data 




s_barrier  // (op 2442) 
ds_read_u8 v51, v8 offset:41568 // (op 663) Load local data 
s_barrier  // (op 2446) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[62:69], v[76:81], a[12:15], v23, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 656) 






buffer_load_ubyte v23, v27, s[52:55], 0 offen offset:12 // (op 231) Load value



v_add_u32 v20, 8, v20 // (op 1885) 
s_lshl_b64 s[14:15], s[24:25], 7 // (op 237) 
s_mul_i32 s19, s15, 6 // (op 237) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 237) least significant
s_mul_hi_u32 s7, s14, 6 // (op 237) most significant: high of low * low
s_add_u32 s19, s19, s7 // (op 237) most significant: sum


s_lshr_b64 s[14:15], s[18:19], 3 // (op 237) 

s_mul_i32 s7, 3, s14 // (op 237) 

v_add_u32 v27, v35, s7 // (op 237) 


buffer_load_dwordx3 v[44:46], v27, s[40:43], 0 offen // (op 237) Load value
buffer_load_dwordx3 v[48:50], v27, s[40:43], 0 offen offset:12 // (op 237) Load value
buffer_load_dwordx3 v[52:54], v27, s[40:43], 0 offen offset:24 // (op 237) Load value
buffer_load_dwordx3 v[56:58], v27, s[40:43], 0 offen offset:36 // (op 237) Load value



v_mov_b64 v[40:41], s[34:35] // (op 238) convert

s_lshl_b32 s7, s34, 2 // (op 238) 
s_mul_i32 s14, 3, s7 // (op 238) 

v_add_u32 v27, v15, s14 // (op 238) 

s_lshl_b64 s[14:15], s[24:25], 7 // (op 1821) 
s_mul_i32 s19, s15, 6 // (op 1821) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 1821) least significant
s_mul_hi_u32 s7, s14, 6 // (op 1821) most significant: high of low * low
s_add_u32 s19, s19, s7 // (op 1821) most significant: sum


buffer_load_ubyte v60, v27, s[56:59], 0 offen offset: 0 // (op 238) Load value
v_add_u32 v27, v27, v40 // (op 238) 
s_lshr_b64 s[14:15], s[18:19], 3 // (op 1821) 

s_lshl_b32 s7, s14, 1 // (op 1821) 

v_add_u32 v13, v13, s7 // (op 1821) 

buffer_load_ubyte v61, v27, s[56:59], 0 offen offset: 0 // (op 238) Load value



s_barrier  // (op 2357) 
s_lshl_b32 s7, s34, 3 // (op 1949) 


v_add_u32 v22, v22, s7 // (op 1949) 

s_add_u32 s7, 4096, s1 // (op 1694) 

s_mov_b32 m0, s7 // (op 1694) reg expression
s_mov_b32 s14, 384 // (op 1694) call()
buffer_load_dwordx4 v16, s[8:11], s14 offen offset: 0 lds // (op 1694) Load value direct to lds



s_barrier  // (op 2329) 
v_add_co_u32 v18, vcc, 256, v18 // (op 1757) least significant half



v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 1757) most significant half
s_waitcnt vmcnt(14)
ds_write_b96 v12, v[24:26] offset:8192 // (op 13) Store local data 
s_waitcnt vmcnt(13)
ds_write_b96 v12, v[28:30] offset:8208 // (op 13) Store local data 
s_waitcnt vmcnt(12)
ds_write_b96 v12, v[32:34] offset:8224 // (op 13) Store local data 
s_waitcnt vmcnt(11)
ds_write_b96 v12, v[36:38] offset:8240 // (op 13) Store local data 
s_barrier  // (op 2343) 



s_waitcnt vmcnt(10)
ds_write_b8 v31, v42 offset:24576 // (op 26) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v31, v43 offset:24704 // (op 26) Store local data 
s_ashr_i64 s[14:15], s[28:29], 7 // (op 619) 
s_mov_b32 s7, s14 // (op 619) convert

s_and_b32 s14, s7, 4294967294 // (op 619) 

s_cmp_lt_u32 s6, s14 // (op 619) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopTop_KLoop_619 // (op 619) Condition: Bottom (jump to RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopTop_KLoop_619 if true)
RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopBottom_KLoop_619:














s_barrier  // (op 2426) 
v_lshrrev_b32 v47, 3, v1 // (op 1717) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 1717) 
v_lshl_add_u32 v51, s16, 5, v47 // (op 1717) 

s_lshr_b32 s6, s0, 1 // (op 1717) 

v_and_b32 v47, -8, v1 // (op 1717) 
v_sub_u32 v55, v1, v47 // (op 1717) 

v_lshlrev_b32 v47, 4, v55 // (op 1717) 

v_lshl_add_u32 v55, s6, 8, v47 // (op 1717) 


v_mov_b32 v13, 0 // (op 1717) convert
v_mov_b32 v12, v55 // (op 1717) convert

v_mul_lo_u32 v15, v51, s13 // (op 1717) most significant: low of low * high
v_mul_hi_u32 v47, v51, s12 // (op 1717) most significant: high of low * low
v_mul_lo_u32 v14, v51, s12 // (op 1717) least significant: low of low * low
v_add_u32 v15, v15, v47 // (op 1717) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v12 // (op 1717) least significant half
v_addc_co_u32 v15, s[0:1], v15, v13, s[0:1] // (op 1717) most significant half



s_mul_i32 s1, s12, s23 // (op 1719) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 1719) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 1719) least significant
s_mul_hi_u32 s7, s12, s22 // (op 1719) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1719) most significant: sum
s_add_u32 s1, s1, s7 // (op 1719) most significant: sum


s_mov_b32 s8, s20 // (op 1719) reg expression
s_mov_b32 s9, s21 // (op 1719) reg expression
s_mov_b32 s10, s0 // (op 1719) reg expression
s_mov_b32 s11, 131072 // (op 1719) call()

v_lshrrev_b32 v47, 1, v1 // (op 1787) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 1787) 
v_and_b32 v51, -2, v1 // (op 1787) 
s_lshr_b32 s6, s0, 1 // (op 1787) 

v_sub_u32 v55, v1, v51 // (op 1787) 

v_lshl_add_u32 v51, s6, 8, v47 // (op 1787) 


v_lshlrev_b32 v47, 6, v55 // (op 1787) 

v_lshl_add_u32 v55, s17, 7, v47 // (op 1787) 

v_mov_b32 v13, 0 // (op 1787) convert
v_mov_b32 v12, v55 // (op 1787) convert

v_mul_lo_u32 v17, v51, s25 // (op 1787) most significant: low of low * high
v_mul_hi_u32 v47, v51, s24 // (op 1787) most significant: high of low * low
v_mul_lo_u32 v16, v51, s24 // (op 1787) least significant: low of low * low
v_add_u32 v17, v17, v47 // (op 1787) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v12 // (op 1787) least significant half
v_addc_co_u32 v17, s[0:1], v17, v13, s[0:1] // (op 1787) most significant half



v_mul_lo_u32 v13, v17, 6 // (op 1787) most significant: low of high * low
v_mul_hi_u32 v47, v16, 6 // (op 1787) most significant: high of low * low
v_mul_lo_u32 v12, v16, 6 // (op 1787) least significant: low of low * low
v_add_u32 v13, v13, v47 // (op 1787) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[12:13] // (op 1787) 

v_mov_b32 v47, v16 // (op 1787) convert

s_mul_i32 s1, s29, 6 // (op 1789) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 1789) least significant
s_mul_hi_u32 s6, s28, 6 // (op 1789) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1789) most significant: sum

s_mul_i32 s7, s24, s1 // (op 1789) most significant: low of low * high
s_mul_i32 s14, s25, s0 // (op 1789) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 1789) least significant
s_mul_hi_u32 s15, s24, s0 // (op 1789) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 1789) most significant: sum
s_add_u32 s7, s7, s15 // (op 1789) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 1789) 

s_mov_b32 s40, s26 // (op 1789) reg expression
s_mov_b32 s41, s27 // (op 1789) reg expression
s_mov_b32 s42, s0 // (op 1789) reg expression
s_mov_b32 s43, 131072 // (op 1789) call()

v_lshrrev_b32 v51, 7, v1 // (op 1851) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 1851) 
v_and_b32 v55, 15, v1 // (op 1851) 
s_lshr_b32 s6, s0, 1 // (op 1851) 

v_lshl_add_u32 v59, v51, 4, v55 // (op 1851) 


v_lshl_add_u32 v51, s16, 5, v59 // (op 1851) 

v_and_b32 v55, 63, v1 // (op 1851) 
v_lshrrev_b32 v59, 4, v55 // (op 1851) 

v_lshl_add_u32 v55, s6, 3, v59 // (op 1851) 


v_mul_lo_u32 v59, v51, s30 // (op 1851) 
v_add_u32 v59, v59, v55 // (op 1851) 


s_mul_i32 s1, s30, s23 // (op 1853) most significant: low of low * high
s_mul_i32 s6, s31, s22 // (op 1853) most significant: low of high * low
s_mul_i32 s0, s30, s22 // (op 1853) least significant
s_mul_hi_u32 s7, s30, s22 // (op 1853) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1853) most significant: sum
s_add_u32 s1, s1, s7 // (op 1853) most significant: sum


s_mov_b32 s52, s32 // (op 1853) reg expression
s_mov_b32 s53, s33 // (op 1853) reg expression
s_mov_b32 s54, s0 // (op 1853) reg expression
s_mov_b32 s55, 131072 // (op 1853) call()

v_lshrrev_b32 v51, 7, v1 // (op 1915) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 1915) 
v_lshlrev_b32 v55, 1, v51 // (op 1915) 

s_lshr_b32 s6, s0, 1 // (op 1915) 

v_and_b32 v51, -128, v1 // (op 1915) 
v_lshl_add_u32 v12, s6, 3, v55 // (op 1915) 


v_sub_u32 v13, v1, v51 // (op 1915) 

v_lshl_add_u32 v51, s17, 7, v13 // (op 1915) 

v_mul_lo_u32 v13, v12, s34 // (op 1915) 
v_add_u32 v13, v13, v51 // (op 1915) 


s_mul_i32 s1, s34, s39 // (op 1917) most significant: low of low * high
s_mul_i32 s6, s35, s38 // (op 1917) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 1917) least significant
s_mul_hi_u32 s7, s34, s38 // (op 1917) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1917) most significant: sum
s_add_u32 s1, s1, s7 // (op 1917) most significant: sum


s_mov_b32 s56, s36 // (op 1917) reg expression
s_mov_b32 s57, s37 // (op 1917) reg expression
s_mov_b32 s58, s0 // (op 1917) reg expression
s_mov_b32 s59, 131072 // (op 1917) call()

v_and_b32 v12, -8, v1 // (op 2008) 
v_sub_u32 v51, v1, v12 // (op 2008) 

v_lshlrev_b32 v12, 4, v51 // (op 2008) 

v_lshrrev_b32 v51, 3, v1 // (op 2008) 
v_lshl_add_u32 v55, v51, 7, v12 // (op 2008) 


s_nop 0
v_readfirstlane_b32 s0, v55 // (op 2008) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -2, v1 // (op 2060) 
v_sub_u32 v51, v1, v12 // (op 2060) 

v_lshlrev_b32 v12, 6, v51 // (op 2060) 

v_lshrrev_b32 v51, 1, v1 // (op 2060) 
v_lshl_add_u32 v55, v51, 7, v12 // (op 2060) 


v_mul_lo_u32 v12, v55, 6 // (op 2060) 
v_lshrrev_b32 v51, 3, v12 // (op 2060) 

v_lshrrev_b32 v12, 4, v55 // (op 2060) 

v_lshlrev_b32 v55, 2, v12 // (op 2060) 

v_add_u32 v12, v51, v55 // (op 2060) 


v_and_b32 v51, -128, v1 // (op 2118) 
v_sub_u32 v55, v1, v51 // (op 2118) 

v_lshrrev_b32 v51, 7, v1 // (op 2118) 
v_lshl_add_u32 v16, v51, 8, v55 // (op 2118) 


s_ashr_i64 s[6:7], s[28:29], 7 // (op 625) 
s_and_b32 s1, s6, 4294967294 // (op 625) 

s_ashr_i64 s[6:7], s[28:29], 7 // (op 624) 
s_mov_b32 s14, s6 // (op 624) convert

s_cmp_lt_u32 s1, s14 // (op 624) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopBottom_KLoopTail_624 // (op 624) Condition: Top (jump to RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopBottom_KLoopTail_624 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopTop_KLoopTail_624:
s_barrier  // (op 2319) 
s_add_u32 s1, s1, 1 // (op 626) 


v_mov_b64 v[48:49], v[14:15] // (op 1688) 

s_mov_b32 m0, s0 // (op 1688) reg expression
buffer_load_dwordx4 v48, s[8:11], 0 offen offset: 0 lds // (op 1688) Load value direct to lds

v_add_co_u32 v14, vcc, 128, v14 // (op 1707) least significant half
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 1707) most significant half


v_mov_b32 v17, v47 // (op 179) 
v_mov_b32 v18, v17 // (op 179) reg expression

buffer_load_dwordx3 v[20:22], v18, s[40:43], 0 offen // (op 179) Load value
buffer_load_dwordx3 v[24:26], v18, s[40:43], 0 offen offset:12 // (op 179) Load value
buffer_load_dwordx3 v[28:30], v18, s[40:43], 0 offen offset:24 // (op 179) Load value
buffer_load_dwordx3 v[32:34], v18, s[40:43], 0 offen offset:36 // (op 179) Load value


s_lshl_b64 s[6:7], s[24:25], 7 // (op 1779) 
s_mul_i32 s15, s7, 6 // (op 1779) most significant: low of high * low
s_mul_i32 s14, s6, 6 // (op 1779) least significant
s_mul_hi_u32 s18, s6, 6 // (op 1779) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 1779) most significant: sum


s_lshr_b64 s[6:7], s[14:15], 3 // (op 1779) 

v_add_u32 v47, v47, s6 // (op 1779) 


s_barrier  // (op 2332) 
v_mov_b64 v[18:19], s[34:35] // (op 180) convert




v_mov_b32 v17, v13 // (op 180) 
v_mov_b32 v23, v17 // (op 180) reg expression
buffer_load_ubyte v48, v23, s[56:59], 0 offen offset: 0 // (op 180) Load value
v_add_u32 v23, v23, v18 // (op 180) 
s_waitcnt vmcnt(4)
ds_write_b96 v12, v[20:22] offset:8192 // (op 177) Store local data 
s_waitcnt vmcnt(3)
ds_write_b96 v12, v[24:26] offset:8208 // (op 177) Store local data 
s_waitcnt vmcnt(2)
ds_write_b96 v12, v[28:30] offset:8224 // (op 177) Store local data 
buffer_load_ubyte v49, v23, s[56:59], 0 offen offset: 0 // (op 180) Load value



s_lshl_b32 s6, s34, 2 // (op 1907) 
v_add_u32 v13, v13, s6 // (op 1907) 

s_waitcnt vmcnt(2)
ds_write_b96 v12, v[32:34] offset:8240 // (op 177) Store local data 

s_barrier  // (op 2322) 



s_barrier  // (op 2346) 
ds_read_b128 v[50:53], v3 // (op 169) Load local data 



ds_read_b128 v[54:57], v3 offset:64 // (op 169) Load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v16, v48 offset:24576 // (op 178) Store local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 2336) 
s_waitcnt vmcnt(0)
ds_write_b8 v16, v49 offset:24704 // (op 178) Store local data 




ds_read_b96_tr_b6 v[18:20], v0 offset:8192 // (op 171) Transpose load local data 
ds_read_b96_tr_b6 v[21:23], v0 offset:10240 // (op 171) Transpose load local data 


v_mov_b32 v17, v59 // (op 173) 

buffer_load_ubyte v58, v17, s[52:55], 0 offen offset: 0 // (op 173) Load value

v_add_u32 v59, 4, v59 // (op 1843) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 2350) 


ds_read_u8 v17, v5 offset:24576 // (op 175) Load local data 



ds_read_b96_tr_b6 v[24:26], v0 offset:8224 // (op 320) Transpose load local data 
s_waitcnt vmcnt(0) lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[50:57], v[18:23], a[0:3], v58, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 176) 
ds_read_b96_tr_b6 v[27:29], v0 offset:10272 // (op 320) Transpose load local data 




ds_read_u8 v17, v5 offset:24608 // (op 323) Load local data 



ds_read_b96_tr_b6 v[18:20], v0 offset:8256 // (op 476) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[50:57], v[24:29], a[4:7], v58, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 324) 
ds_read_b96_tr_b6 v[21:23], v0 offset:10304 // (op 476) Transpose load local data 




ds_read_u8 v17, v5 offset:24640 // (op 479) Load local data 



ds_read_b96_tr_b6 v[24:26], v0 offset:8288 // (op 632) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[50:57], v[18:23], a[8:11], v58, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 480) 
ds_read_b96_tr_b6 v[27:29], v0 offset:10336 // (op 632) Transpose load local data 




ds_read_u8 v17, v5 offset:24672 // (op 635) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[50:57], v[24:29], a[12:15], v58, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 636) 




s_ashr_i64 s[6:7], s[28:29], 7 // (op 624) 
s_mov_b32 s14, s6 // (op 624) convert

s_cmp_lt_u32 s1, s14 // (op 624) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopTop_KLoopTail_624 // (op 624) Condition: Bottom (jump to RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopTop_KLoopTail_624 if true)
RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopBottom_KLoopTail_624:























s_barrier  // (op 2430) 






v_cmp_eq_f32 vcc, 0.00000, v10 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ConditionalFalse_0_beta_123
v_accvgpr_read v12, a0 // (op 46) 
v_accvgpr_read v13, a1 // (op 46) 
v_accvgpr_read v14, a2 // (op 46) 
v_accvgpr_read v15, a3 // (op 46) 
v_mul_f32 v16, v11, v12 // (op 46) 
v_mul_f32 v17, v11, v13 // (op 46) 
v_mul_f32 v18, v11, v14 // (op 46) 
v_mul_f32 v19, v11, v15 // (op 46) 



v_mov_b32 v0, v9 // (op 52) reg expression

buffer_store_dwordx4 v[16:19], v0, s[48:51], 0 offen // (op 52) Store value


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


buffer_store_dwordx4 v[16:19], v0, s[48:51], 0 offen // (op 862) Store value


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


buffer_store_dwordx4 v[16:19], v0, s[48:51], 0 offen // (op 874) Store value


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


buffer_store_dwordx4 v[16:19], v0, s[48:51], 0 offen // (op 886) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ConditionalBottom_0_beta_123
RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ConditionalFalse_0_beta_123:


v_mov_b32 v0, v6 // (op 135) reg expression

buffer_load_dwordx4 v[12:15], v0, s[44:47], 0 offen // (op 135) Load value

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

buffer_store_dwordx4 v[12:15], v0, s[48:51], 0 offen // (op 134) Store value




s_lshl_b32 s0, s2, 7 // (op 918) 
v_add_u32 v0, v6, s0 // (op 918) 


buffer_load_dwordx4 v[12:15], v0, s[44:47], 0 offen // (op 918) Load value

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


buffer_store_dwordx4 v[12:15], v0, s[48:51], 0 offen // (op 917) Store value




s_lshl_b32 s0, s2, 8 // (op 930) 
v_add_u32 v0, v6, s0 // (op 930) 


buffer_load_dwordx4 v[12:15], v0, s[44:47], 0 offen // (op 930) Load value

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


buffer_store_dwordx4 v[12:15], v0, s[48:51], 0 offen // (op 929) Store value




s_lshl_b32 s0, s2, 7 // (op 942) 
s_mul_i32 s1, 3, s0 // (op 942) 

v_add_u32 v0, v6, s1 // (op 942) 


buffer_load_dwordx4 v[12:15], v0, s[44:47], 0 offen // (op 942) Load value

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


buffer_store_dwordx4 v[12:15], v0, s[48:51], 0 offen // (op 941) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ConditionalBottom_0_beta_123:












s_endpgm  // End of RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2
.LRR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_end:
.size RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2, .LRR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_end-RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2
  .amdhsa_next_free_vgpr 100
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 41984 // lds bytes
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
    DataType: B8
    DataTypeA: B8
    DataTypeB: B6
    DestDataType: s
    ComputeDataType: s
    HighPrecisionAccumulate: True
    TransposeA: True
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
            { type: int64, semantic: SizeFree0 },
            { type: int64, semantic: StrideB0 },
            { type: address, semantic: AddressB },
            { type: int64, semantic: SizeSum },
            { type: int64, semantic: StrideScaleA0 },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: StrideScaleB0 },
            { type: address, semantic: AddressMXScaleB },
            { type: int64, semantic: SizeSumScaleA } ]
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
  - .name: RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2
    .symbol: RR_GEMM_TT_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2.kd
    .kernarg_segment_size: 136
    .group_segment_fixed_size: 41984
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 60
    .vgpr_count: 82
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
      - .name: Tensor_2_size_1_12
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
      - .name: Tensor_0_size_1_4
        .size: 8
        .offset: 88
        .value_kind: by_value
      - .name: Tensor_4_stride_0_6
        .size: 8
        .offset: 96
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 104
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
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
      - .name: Tensor_4_size_1_9
        .size: 8
        .offset: 128
        .value_kind: by_value
...

.end_amdgpu_metadata

