// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2
.p2align 8
.type RR_GEMM_NT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2,@function
RR_GEMM_NT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_exec_begin:


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


v_and_b32 v0, 63, v1 // (op 2161) 
v_and_b32 v3, 48, v1 // (op 2161) 
v_sub_u32 v4, v0, v3 // (op 2161) 

v_lshrrev_b32 v3, 4, v0 // (op 2161) 

v_lshl_add_u32 v0, v3, 5, v4 // (op 2161) 


v_lshrrev_b32 v3, 6, v1 // (op 2161) 
v_and_b32 v4, 1, v3 // (op 2161) 

v_lshlrev_b32 v3, 4, v4 // (op 2161) 

v_lshl_add_u32 v4, v0, 5, v3 // (op 2161) 


v_mul_lo_u32 v0, v4, 6 // (op 2161) 
v_lshrrev_b32 v3, 3, v0 // (op 2161) 

v_lshrrev_b32 v0, 4, v4 // (op 2161) 

v_lshlrev_b32 v4, 2, v0 // (op 2161) 

v_add_u32 v0, v3, v4 // (op 2161) 


v_and_b32 v3, 63, v1 // (op 2305) 
v_and_b32 v4, 48, v1 // (op 2305) 
v_sub_u32 v5, v3, v4 // (op 2305) 

v_lshrrev_b32 v4, 4, v3 // (op 2305) 

v_lshl_add_u32 v3, v4, 5, v5 // (op 2305) 


v_lshrrev_b32 v4, 6, v1 // (op 2305) 
v_and_b32 v5, 1, v4 // (op 2305) 

v_lshlrev_b32 v4, 4, v5 // (op 2305) 

v_lshl_add_u32 v5, v3, 5, v4 // (op 2305) 


v_mul_lo_u32 v3, v5, 6 // (op 2305) 
v_lshrrev_b32 v4, 3, v3 // (op 2305) 

v_lshrrev_b32 v3, 4, v5 // (op 2305) 

v_lshlrev_b32 v5, 2, v3 // (op 2305) 

v_add_u32 v3, v4, v5 // (op 2305) 


v_lshrrev_b32 v4, 7, v1 // (op 2047) 
v_and_b32 v5, 63, v1 // (op 2047) 
v_lshrrev_b32 v6, 4, v5 // (op 2047) 
v_lshlrev_b32 v7, 2, v6 // (op 2047) 

v_lshl_add_u32 v6, v4, 4, v7 // (op 2047) 


v_lshl_add_u32 v4, s16, 7, v6 // (op 2047) 

v_lshrrev_b32 v6, 2, v5 // (op 2047) 

v_and_b32 v5, 3, v6 // (op 2047) 

v_and_b32 v6, 3, v1 // (op 2047) 
v_lshl_add_u32 v7, v5, 2, v6 // (op 2047) 


v_lshrrev_b32 v5, 6, v1 // (op 2047) 
v_and_b32 v6, 1, v5 // (op 2047) 

v_lshl_add_u32 v5, v6, 4, v7 // (op 2047) 


v_lshl_add_u32 v6, s17, 5, v5 // (op 2047) 

v_mul_lo_u32 v5, v6, s2 // (op 2047) 

v_add_lshl_u32 v6, v4, v5, 1 // (op 2047) 


v_and_b32 v4, 48, v1 // (op 2108) 
v_and_b32 v5, 63, v1 // (op 2108) 
v_sub_u32 v7, v5, v4 // (op 2108) 

v_lshrrev_b32 v5, 1, v7 // (op 2108) 
s_mul_i32 s1, s2, s9 // (op 2049) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 2049) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 2049) least significant
s_mul_hi_u32 s19, s2, s8 // (op 2049) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2049) most significant: sum
s_add_u32 s1, s1, s19 // (op 2049) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 2049) 

v_add_lshl_u32 v8, v4, v5, 7 // (op 2108) 


v_and_b32 v4, 1, v7 // (op 2108) 

v_lshlrev_b32 v5, 3, v4 // (op 2108) 

v_lshrrev_b32 v4, 7, v1 // (op 2108) 
v_lshl_add_u32 v7, v4, 4, v5 // (op 2108) 


v_add_u32 v4, v8, v7 // (op 2108) 


v_lshrrev_b32 v5, 7, v1 // (op 2219) 
v_and_b32 v7, 15, v1 // (op 2219) 
v_lshl_add_u32 v8, v5, 4, v7 // (op 2219) 


v_and_b32 v5, 63, v1 // (op 2219) 
v_lshrrev_b32 v7, 4, v5 // (op 2219) 

v_lshl_add_u32 v5, v7, 7, v8 // (op 2219) 


v_lshrrev_b32 v7, 7, v1 // (op 2235) 
v_and_b32 v8, 63, v1 // (op 2235) 
v_lshrrev_b32 v9, 4, v8 // (op 2235) 
v_lshlrev_b32 v10, 2, v9 // (op 2235) 

v_lshl_add_u32 v9, v7, 4, v10 // (op 2235) 


v_lshl_add_u32 v7, s16, 7, v9 // (op 2235) 

v_lshrrev_b32 v9, 2, v8 // (op 2235) 

v_and_b32 v8, 3, v9 // (op 2235) 

v_and_b32 v9, 3, v1 // (op 2235) 
v_lshl_add_u32 v10, v8, 2, v9 // (op 2235) 


v_lshrrev_b32 v8, 6, v1 // (op 2235) 
v_and_b32 v9, 1, v8 // (op 2235) 

v_lshl_add_u32 v8, v9, 4, v10 // (op 2235) 


v_lshl_add_u32 v9, s17, 5, v8 // (op 2235) 

v_mul_lo_u32 v8, v9, s4 // (op 2235) 

v_add_lshl_u32 v9, v7, v8, 1 // (op 2235) 


v_and_b32 v7, 48, v1 // (op 2281) 
v_and_b32 v8, 63, v1 // (op 2281) 
v_sub_u32 v10, v8, v7 // (op 2281) 

v_lshrrev_b32 v8, 1, v10 // (op 2281) 
s_mul_i32 s1, s4, s9 // (op 2237) most significant: low of low * high
v_add_lshl_u32 v11, v7, v8, 7 // (op 2281) 


v_and_b32 v7, 1, v10 // (op 2281) 

v_lshlrev_b32 v8, 3, v7 // (op 2281) 

v_lshrrev_b32 v7, 7, v1 // (op 2281) 
v_lshl_add_u32 v10, v7, 4, v8 // (op 2281) 


v_add_u32 v7, v11, v10 // (op 2281) 


v_lshrrev_b32 v8, 7, v1 // (op 2334) 
v_and_b32 v10, 15, v1 // (op 2334) 
v_lshl_add_u32 v11, v8, 4, v10 // (op 2334) 


v_and_b32 v8, 63, v1 // (op 2334) 
v_lshrrev_b32 v10, 4, v8 // (op 2334) 

v_lshl_add_u32 v8, v10, 7, v11 // (op 2334) 


s_mul_i32 s38, s5, s8 // (op 2237) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 2237) least significant
s_mul_hi_u32 s39, s4, s8 // (op 2237) most significant: high of low * low
s_add_u32 s1, s1, s38 // (op 2237) most significant: sum
s_add_u32 s1, s1, s39 // (op 2237) most significant: sum


s_lshl_b64 s[38:39], s[0:1], 1 // (op 2237) 

s_mov_b32 s40, s6 // (op 2049) reg expression
s_mov_b32 s41, s7 // (op 2049) reg expression
s_mov_b32 s42, s18 // (op 2049) reg expression
s_mov_b32 s43, 131072 // (op 2049) call()

s_mov_b32 s44, s10 // (op 2237) reg expression

s_mov_b32 s45, s11 // (op 2237) reg expression
s_mov_b32 s46, s38 // (op 2237) reg expression
s_mov_b32 s47, 131072 // (op 2237) call()



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
v_and_b32 v12, -8, v1 // (op 1814) 
v_sub_u32 v13, v1, v12 // (op 1814) 

v_lshlrev_b32 v12, 4, v13 // (op 1814) 

v_lshl_add_u32 v13, s16, 7, v12 // (op 1814) 

v_mov_b32 v15, 0 // (op 1814) convert
v_mov_b32 v14, v13 // (op 1814) convert

v_lshrrev_b32 v12, 3, v1 // (op 1814) 
v_mul_lo_u32 v17, v12, s13 // (op 1814) most significant: low of low * high
v_mul_hi_u32 v13, v12, s12 // (op 1814) most significant: high of low * low
v_mul_lo_u32 v16, v12, s12 // (op 1814) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 1814) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v16 // (op 1814) least significant half
v_addc_co_u32 v13, s[0:1], v15, v17, s[0:1] // (op 1814) most significant half




s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 1816) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 1816) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 1816) least significant
s_mul_hi_u32 s7, s12, s22 // (op 1816) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1816) most significant: sum
s_add_u32 s1, s1, s7 // (op 1816) most significant: sum


s_mov_b32 s8, s20 // (op 1816) reg expression
s_mov_b32 s9, s21 // (op 1816) reg expression
s_mov_b32 s10, s0 // (op 1816) reg expression
s_mov_b32 s11, 131072 // (op 1816) call()

v_and_b32 v14, -8, v1 // (op 1833) 
v_sub_u32 v15, v1, v14 // (op 1833) 

v_lshlrev_b32 v14, 4, v15 // (op 1833) 

v_lshl_add_u32 v15, s16, 7, v14 // (op 1833) 

v_mov_b32 v17, 0 // (op 1833) convert
v_mov_b32 v16, v15 // (op 1833) convert

v_lshrrev_b32 v14, 3, v1 // (op 1833) 
v_mul_lo_u32 v19, v14, s13 // (op 1833) most significant: low of low * high
v_mul_hi_u32 v15, v14, s12 // (op 1833) most significant: high of low * low
v_mul_lo_u32 v18, v14, s12 // (op 1833) least significant: low of low * low
v_add_u32 v19, v19, v15 // (op 1833) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v18 // (op 1833) least significant half
v_addc_co_u32 v15, s[0:1], v17, v19, s[0:1] // (op 1833) most significant half



s_mul_i32 s1, s12, s23 // (op 1835) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 1835) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 1835) least significant
s_mul_hi_u32 s7, s12, s22 // (op 1835) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1835) most significant: sum
s_add_u32 s1, s1, s7 // (op 1835) most significant: sum


s_mov_b32 s8, s20 // (op 1835) reg expression
s_mov_b32 s9, s21 // (op 1835) reg expression
s_mov_b32 s10, s0 // (op 1835) reg expression
s_mov_b32 s11, 131072 // (op 1835) call()

v_and_b32 v16, -2, v1 // (op 1878) 
v_sub_u32 v17, v1, v16 // (op 1878) 

v_lshlrev_b32 v16, 4, v17 // (op 1878) 

v_lshl_add_u32 v17, s17, 5, v16 // (op 1878) 

v_mov_b32 v19, 0 // (op 1878) convert
v_mov_b32 v18, v17 // (op 1878) convert

v_lshrrev_b32 v16, 1, v1 // (op 1878) 
v_mul_lo_u32 v21, v16, s25 // (op 1878) most significant: low of low * high
v_mul_hi_u32 v17, v16, s24 // (op 1878) most significant: high of low * low
v_mul_lo_u32 v20, v16, s24 // (op 1878) least significant: low of low * low
v_add_u32 v21, v21, v17 // (op 1878) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v18 // (op 1878) least significant half
v_addc_co_u32 v21, s[0:1], v21, v19, s[0:1] // (op 1878) most significant half



v_mul_lo_u32 v17, v21, 6 // (op 1878) most significant: low of high * low
v_mul_hi_u32 v18, v20, 6 // (op 1878) most significant: high of low * low
v_mul_lo_u32 v16, v20, 6 // (op 1878) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 1878) most significant: sum


v_lshrrev_b64 v[18:19], 3, v[16:17] // (op 1878) 

v_mov_b32 v16, v18 // (op 1878) convert

s_mul_i32 s1, s23, 6 // (op 1880) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 1880) least significant
s_mul_hi_u32 s6, s22, 6 // (op 1880) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1880) most significant: sum

s_mul_i32 s7, s24, s1 // (op 1880) most significant: low of low * high
s_mul_i32 s14, s25, s0 // (op 1880) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 1880) least significant
s_mul_hi_u32 s15, s24, s0 // (op 1880) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 1880) most significant: sum
s_add_u32 s7, s7, s15 // (op 1880) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 1880) 

s_mov_b32 s48, s26 // (op 1880) reg expression
s_mov_b32 s49, s27 // (op 1880) reg expression
s_mov_b32 s50, s0 // (op 1880) reg expression
s_mov_b32 s51, 131072 // (op 1880) call()

v_and_b32 v17, -2, v1 // (op 1897) 
v_sub_u32 v18, v1, v17 // (op 1897) 

v_lshlrev_b32 v17, 4, v18 // (op 1897) 

v_lshl_add_u32 v18, s17, 5, v17 // (op 1897) 

v_mov_b32 v21, 0 // (op 1897) convert
v_mov_b32 v20, v18 // (op 1897) convert

v_lshrrev_b32 v17, 1, v1 // (op 1897) 
v_mul_lo_u32 v19, v17, s25 // (op 1897) most significant: low of low * high
v_mul_hi_u32 v22, v17, s24 // (op 1897) most significant: high of low * low
v_mul_lo_u32 v18, v17, s24 // (op 1897) least significant: low of low * low
v_add_u32 v19, v19, v22 // (op 1897) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v20 // (op 1897) least significant half
v_addc_co_u32 v19, s[0:1], v19, v21, s[0:1] // (op 1897) most significant half



v_mul_lo_u32 v21, v19, 6 // (op 1897) most significant: low of high * low
v_mul_hi_u32 v17, v18, 6 // (op 1897) most significant: high of low * low
v_mul_lo_u32 v20, v18, 6 // (op 1897) least significant: low of low * low
v_add_u32 v21, v21, v17 // (op 1897) most significant: sum


v_lshrrev_b64 v[18:19], 3, v[20:21] // (op 1897) 

v_mov_b32 v17, v18 // (op 1897) convert

s_mul_i32 s1, s23, 6 // (op 1899) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 1899) least significant
s_mul_hi_u32 s6, s22, 6 // (op 1899) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1899) most significant: sum

s_mul_i32 s7, s24, s1 // (op 1899) most significant: low of low * high
s_mul_i32 s14, s25, s0 // (op 1899) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 1899) least significant
s_mul_hi_u32 s15, s24, s0 // (op 1899) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 1899) most significant: sum
s_add_u32 s7, s7, s15 // (op 1899) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 1899) 

s_mov_b32 s48, s26 // (op 1899) reg expression
s_mov_b32 s49, s27 // (op 1899) reg expression
s_mov_b32 s50, s0 // (op 1899) reg expression
s_mov_b32 s51, 131072 // (op 1899) call()

v_and_b32 v18, -128, v1 // (op 1942) 
v_sub_u32 v19, v1, v18 // (op 1942) 

v_lshl_add_u32 v18, s16, 7, v19 // (op 1942) 

v_lshrrev_b32 v19, 7, v1 // (op 1942) 
v_lshlrev_b32 v20, 1, v19 // (op 1942) 

v_mul_lo_u32 v19, v20, s28 // (op 1942) 

v_add_u32 v20, v18, v19 // (op 1942) 


s_mul_i32 s1, s28, s33 // (op 1944) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 1944) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1944) least significant
s_mul_hi_u32 s7, s28, s32 // (op 1944) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1944) most significant: sum
s_add_u32 s1, s1, s7 // (op 1944) most significant: sum


s_mov_b32 s52, s30 // (op 1944) reg expression
s_mov_b32 s53, s31 // (op 1944) reg expression
s_mov_b32 s54, s0 // (op 1944) reg expression
s_mov_b32 s55, 131072 // (op 1944) call()

v_and_b32 v18, -128, v1 // (op 1961) 
v_sub_u32 v19, v1, v18 // (op 1961) 

v_lshl_add_u32 v18, s16, 7, v19 // (op 1961) 

v_lshrrev_b32 v19, 7, v1 // (op 1961) 
v_lshlrev_b32 v21, 1, v19 // (op 1961) 

v_mul_lo_u32 v19, v21, s28 // (op 1961) 

v_add_u32 v21, v18, v19 // (op 1961) 


s_mul_i32 s1, s28, s33 // (op 1963) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 1963) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1963) least significant
s_mul_hi_u32 s7, s28, s32 // (op 1963) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1963) most significant: sum
s_add_u32 s1, s1, s7 // (op 1963) most significant: sum


s_mov_b32 s52, s30 // (op 1963) reg expression
s_mov_b32 s53, s31 // (op 1963) reg expression
s_mov_b32 s54, s0 // (op 1963) reg expression
s_mov_b32 s55, 131072 // (op 1963) call()

v_lshrrev_b32 v18, 6, v1 // (op 2006) 
v_and_b32 v19, 1, v18 // (op 2006) 

v_and_b32 v18, 15, v1 // (op 2006) 
v_lshl_add_u32 v22, v19, 4, v18 // (op 2006) 


v_lshl_add_u32 v18, s17, 5, v22 // (op 2006) 

v_and_b32 v19, 63, v1 // (op 2006) 
v_lshrrev_b32 v22, 4, v19 // (op 2006) 

v_mul_lo_u32 v19, v22, s34 // (op 2006) 
v_add_u32 v19, v19, v18 // (op 2006) 


s_mul_i32 s1, s34, s33 // (op 2008) most significant: low of low * high
s_mul_i32 s6, s35, s32 // (op 2008) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 2008) least significant
s_mul_hi_u32 s7, s34, s32 // (op 2008) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2008) most significant: sum
s_add_u32 s1, s1, s7 // (op 2008) most significant: sum


s_mov_b32 s56, s36 // (op 2008) reg expression
s_mov_b32 s57, s37 // (op 2008) reg expression
s_mov_b32 s58, s0 // (op 2008) reg expression
s_mov_b32 s59, 131072 // (op 2008) call()

v_lshrrev_b32 v18, 6, v1 // (op 2025) 
v_and_b32 v22, 1, v18 // (op 2025) 

v_and_b32 v18, 15, v1 // (op 2025) 
v_lshl_add_u32 v23, v22, 4, v18 // (op 2025) 


v_lshl_add_u32 v18, s17, 5, v23 // (op 2025) 

v_and_b32 v22, 63, v1 // (op 2025) 
v_lshrrev_b32 v23, 4, v22 // (op 2025) 

v_mul_lo_u32 v22, v23, s34 // (op 2025) 
v_add_u32 v22, v22, v18 // (op 2025) 


s_mul_i32 s1, s34, s33 // (op 2027) most significant: low of low * high
s_mul_i32 s6, s35, s32 // (op 2027) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 2027) least significant
s_mul_hi_u32 s7, s34, s32 // (op 2027) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2027) most significant: sum
s_add_u32 s1, s1, s7 // (op 2027) most significant: sum


s_mov_b32 s56, s36 // (op 2027) reg expression
s_mov_b32 s57, s37 // (op 2027) reg expression
s_mov_b32 s58, s0 // (op 2027) reg expression
s_mov_b32 s59, 131072 // (op 2027) call()


v_mov_b64 v[24:25], s[34:35] // (op 1616) convert

v_mov_b32 v18, v19 // (op 1616) reg expression
v_mov_b32 v23, v18 // (op 1616) 
buffer_load_ubyte v26, v23, s[56:59], 0 offen offset: 0 // (op 1616) Load value





v_mov_b32 v18, v16 // (op 1620) reg expression

buffer_load_dwordx3 v[28:30], v18, s[48:51], 0 offen // (op 1620) Load value

v_and_b32 v18, -8, v1 // (op 2091) 
v_sub_u32 v27, v1, v18 // (op 2091) 

v_lshlrev_b32 v18, 4, v27 // (op 2091) 

v_lshrrev_b32 v27, 3, v1 // (op 2091) 
v_lshl_add_u32 v23, v27, 7, v18 // (op 2091) 


s_nop 0
v_readfirstlane_b32 s0, v23 // (op 2091) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 5 // (op 1767) 
v_mov_b64 v[24:25], s[6:7] // (op 1767) convert


v_mov_b64 v[32:33], v[12:13] // (op 1767) reg expression
s_mov_b32 m0, s0 // (op 1767) reg expression
buffer_load_dwordx4 v32, s[8:11], 0 offen offset: 0 lds // (op 1767) Load value direct to lds
v_add_co_u32 v32, s[6:7], v32, v24 // (op 1767) least significant half
v_addc_co_u32 v33, s[6:7], 0, v33, s[6:7] // (op 1767) most significant half

s_add_u32 m0, m0, 4096 // (op 1767) 
buffer_load_dwordx4 v32, s[8:11], 0 offen offset: 0 lds // (op 1767) Load value direct to lds
v_add_co_u32 v32, s[6:7], v32, v24 // (op 1767) least significant half
v_addc_co_u32 v33, s[6:7], 0, v33, s[6:7] // (op 1767) most significant half

s_add_u32 m0, m0, 4096 // (op 1767) 
buffer_load_dwordx4 v32, s[8:11], 0 offen offset: 0 lds // (op 1767) Load value direct to lds
v_add_co_u32 v32, s[6:7], v32, v24 // (op 1767) least significant half
v_addc_co_u32 v33, s[6:7], 0, v33, s[6:7] // (op 1767) most significant half

s_add_u32 m0, m0, 4096 // (op 1767) 
buffer_load_dwordx4 v32, s[8:11], 0 offen offset: 0 lds // (op 1767) Load value direct to lds




v_mov_b64 v[24:25], s[28:29] // (op 1626) convert

v_mov_b32 v18, v20 // (op 1626) reg expression
buffer_load_ubyte v32, v18, s[52:55], 0 offen offset: 0 // (op 1626) Load value
v_add_u32 v18, v18, v24 // (op 1626) 
buffer_load_ubyte v33, v18, s[52:55], 0 offen offset: 0 // (op 1626) Load value



v_mov_b64 v[24:25], s[34:35] // (op 1630) convert

s_lshl_b32 s0, s34, 2 // (op 1630) 
v_add_u32 v18, v19, s0 // (op 1630) 

v_mov_b32 v23, v18 // (op 1630) 
buffer_load_ubyte v27, v23, s[56:59], 0 offen offset: 0 // (op 1630) Load value






s_lshl_b64 s[0:1], s[24:25], 7 // (op 1634) 
s_mul_i32 s7, s1, 6 // (op 1634) most significant: low of high * low
s_mul_i32 s6, s0, 6 // (op 1634) least significant
s_mul_hi_u32 s14, s0, 6 // (op 1634) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 1634) most significant: sum


s_lshr_b64 s[0:1], s[6:7], 3 // (op 1634) 

v_add_u32 v31, v16, s0 // (op 1634) 


buffer_load_dwordx3 v[34:36], v31, s[48:51], 0 offen // (op 1634) Load value


v_and_b32 v16, -8, v1 // (op 2264) 
v_sub_u32 v31, v1, v16 // (op 2264) 

v_lshlrev_b32 v16, 4, v31 // (op 2264) 

v_lshrrev_b32 v31, 3, v1 // (op 2264) 
v_lshl_add_u32 v18, v31, 7, v16 // (op 2264) 


s_nop 0
v_readfirstlane_b32 s0, v18 // (op 2264) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 5 // (op 1772) 
v_mov_b64 v[18:19], s[6:7] // (op 1772) convert


s_lshl_b64 s[6:7], s[12:13], 7 // (op 1772) 
v_mov_b32 v16, s6 // (op 1772) 
v_mov_b32 v31, s7 // (op 1772) 
v_add_co_u32 v24, s[14:15], v12, v16 // (op 1772) least significant half
v_addc_co_u32 v25, s[14:15], v13, v31, s[14:15] // (op 1772) most significant half




s_add_u32 s1, 16384, s0 // (op 1772) 
s_mov_b32 m0, s1 // (op 1772) reg expression
buffer_load_dwordx4 v24, s[8:11], 0 offen offset: 0 lds // (op 1772) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v18 // (op 1772) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 1772) most significant half

s_add_u32 m0, m0, 4096 // (op 1772) 
buffer_load_dwordx4 v24, s[8:11], 0 offen offset: 0 lds // (op 1772) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v18 // (op 1772) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 1772) most significant half

s_add_u32 m0, m0, 4096 // (op 1772) 
buffer_load_dwordx4 v24, s[8:11], 0 offen offset: 0 lds // (op 1772) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v18 // (op 1772) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 1772) most significant half

s_add_u32 m0, m0, 4096 // (op 1772) 
buffer_load_dwordx4 v24, s[8:11], 0 offen offset: 0 lds // (op 1772) Load value direct to lds






v_mov_b64 v[12:13], s[28:29] // (op 1640) convert

s_lshl_b32 s0, s28, 2 // (op 1640) 
v_add_u32 v16, v20, s0 // (op 1640) 

buffer_load_ubyte v18, v16, s[52:55], 0 offen offset: 0 // (op 1640) Load value
v_add_u32 v16, v16, v12 // (op 1640) 
buffer_load_ubyte v19, v16, s[52:55], 0 offen offset: 0 // (op 1640) Load value



v_and_b32 v16, -2, v1 // (op 2149) 
v_sub_u32 v20, v1, v16 // (op 2149) 

v_lshlrev_b32 v16, 4, v20 // (op 2149) 

v_lshrrev_b32 v20, 1, v1 // (op 2149) 
v_lshl_add_u32 v31, v20, 5, v16 // (op 2149) 


v_mul_lo_u32 v16, v31, 6 // (op 2149) 
v_lshrrev_b32 v20, 3, v16 // (op 2149) 

v_lshrrev_b32 v16, 4, v31 // (op 2149) 

v_lshlrev_b32 v31, 2, v16 // (op 2149) 

v_add_u32 v16, v20, v31 // (op 2149) 





s_waitcnt vmcnt(14)
ds_write_b96 v16, v[28:30] offset:32768 // (op 1642) Store local data 

v_and_b32 v16, -128, v1 // (op 2202) 
v_sub_u32 v20, v1, v16 // (op 2202) 

v_lshrrev_b32 v16, 7, v1 // (op 2202) 
v_lshl_add_u32 v31, v16, 8, v20 // (op 2202) 





s_waitcnt vmcnt(9)
ds_write_b8 v31, v32 offset:36864 // (op 1644) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v31, v33 offset:36992 // (op 1644) Store local data 

v_and_b32 v16, -8, v1 // (op 2063) 
v_sub_u32 v20, v1, v16 // (op 2063) 

v_lshlrev_b32 v16, 4, v20 // (op 2063) 

v_lshrrev_b32 v20, 3, v1 // (op 2063) 
v_lshl_add_u32 v31, v20, 7, v16 // (op 2063) 


s_nop 0
v_readfirstlane_b32 s0, v31 // (op 2063) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v16, -2, v1 // (op 2120) 
v_sub_u32 v20, v1, v16 // (op 2120) 

v_lshlrev_b32 v16, 4, v20 // (op 2120) 

v_lshrrev_b32 v20, 1, v1 // (op 2120) 
v_lshl_add_u32 v31, v20, 5, v16 // (op 2120) 


v_mul_lo_u32 v16, v31, 6 // (op 2120) 
v_lshrrev_b32 v20, 3, v16 // (op 2120) 

v_lshrrev_b32 v16, 4, v31 // (op 2120) 

v_lshlrev_b32 v31, 2, v16 // (op 2120) 

v_add_u32 v16, v20, v31 // (op 2120) 


v_and_b32 v20, -128, v1 // (op 2173) 
v_sub_u32 v31, v1, v20 // (op 2173) 

v_lshrrev_b32 v20, 7, v1 // (op 2173) 
v_lshl_add_u32 v12, v20, 8, v31 // (op 2173) 


v_and_b32 v13, -8, v1 // (op 2249) 
v_sub_u32 v20, v1, v13 // (op 2249) 

v_lshlrev_b32 v13, 4, v20 // (op 2249) 

v_lshrrev_b32 v20, 3, v1 // (op 2249) 
v_lshl_add_u32 v31, v20, 7, v13 // (op 2249) 


s_nop 0
v_readfirstlane_b32 s1, v31 // (op 2249) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v13, -2, v1 // (op 2293) 
v_sub_u32 v20, v1, v13 // (op 2293) 

v_lshlrev_b32 v13, 4, v20 // (op 2293) 

v_lshrrev_b32 v20, 1, v1 // (op 2293) 
v_lshl_add_u32 v31, v20, 5, v13 // (op 2293) 


v_mul_lo_u32 v13, v31, 6 // (op 2293) 
v_lshrrev_b32 v20, 3, v13 // (op 2293) 

v_lshrrev_b32 v13, 4, v31 // (op 2293) 

v_lshlrev_b32 v31, 2, v13 // (op 2293) 

v_add_u32 v13, v20, v31 // (op 2293) 


v_and_b32 v20, -128, v1 // (op 2317) 
v_sub_u32 v31, v1, v20 // (op 2317) 

v_lshrrev_b32 v20, 7, v1 // (op 2317) 
v_lshl_add_u32 v23, v20, 8, v31 // (op 2317) 


s_mov_b32 s6, 0 // (op 639) call()
s_ashr_i64 s[14:15], s[22:23], 7 // (op 638) 
s_mov_b32 s7, s14 // (op 638) convert

s_and_b32 s14, s7, 4294967294 // (op 638) 

s_cmp_lt_u32 s6, s14 // (op 638) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopBottom_KLoop_638 // (op 638) Condition: Top (jump to RR_GEMM_NT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopBottom_KLoop_638 if false)
RR_GEMM_NT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopTop_KLoop_638:
s_barrier  // (op 2381) 
s_add_u32 s6, s6, 2 // (op 640) 



ds_read_b64_tr_b8 v[38:39], v4 // (op 67) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v4 offset:1024 // (op 67) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v4 offset:8192 // (op 67) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v4 offset:9216 // (op 67) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 2394) 



ds_read_b96_tr_b6 v[46:48], v0 offset:32768 // (op 72) Transpose load local data 
ds_read_b96_tr_b6 v[49:51], v0 offset:33280 // (op 72) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 2408) 


ds_read_u8 v20, v5 offset:36864 // (op 77) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[38:45], v[46:51], a[0:3], v20, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 84) 
ds_read_b64_tr_b8 v[52:53], v4 offset:32 // (op 344) Transpose load local data 


ds_read_b64_tr_b8 v[54:55], v4 offset:1056 // (op 344) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v4 offset:8224 // (op 344) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v4 offset:9248 // (op 344) Transpose load local data 


ds_read_u8 v20, v5 offset:36896 // (op 348) Load local data 



ds_read_b64_tr_b8 v[38:39], v4 offset:64 // (op 506) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[52:59], v[46:51], a[4:7], v20, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 350) 
ds_read_b64_tr_b8 v[40:41], v4 offset:1088 // (op 506) Transpose load local data 


ds_read_b64_tr_b8 v[42:43], v4 offset:8256 // (op 506) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v4 offset:9280 // (op 506) Transpose load local data 


ds_read_u8 v20, v5 offset:36928 // (op 510) Load local data 



ds_read_b64_tr_b8 v[52:53], v4 offset:96 // (op 668) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v4 offset:1120 // (op 668) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[38:45], v[46:51], a[8:11], v20, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 512) 
ds_read_b64_tr_b8 v[56:57], v4 offset:8288 // (op 668) Transpose load local data 


ds_read_b64_tr_b8 v[58:59], v4 offset:9312 // (op 668) Transpose load local data 


ds_read_u8 v20, v5 offset:36960 // (op 672) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[52:59], v[46:51], a[12:15], v20, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 674) 




v_mov_b64 v[24:25], s[34:35] // (op 82) convert

v_mov_b32 v20, v22 // (op 82) 
s_lshl_b32 s7, s34, 3 // (op 82) 
v_add_u32 v31, v20, s7 // (op 82) 

v_mov_b32 v37, v31 // (op 82) 
buffer_load_ubyte v26, v37, s[56:59], 0 offen offset: 0 // (op 82) Load value





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

s_barrier  // (op 2384) 

s_lshl_b64 s[14:15], s[12:13], 5 // (op 1751) 
v_mov_b64 v[24:25], s[14:15] // (op 1751) convert


v_mov_b64 v[38:39], v[14:15] // (op 1751) 
s_lshl_b64 s[14:15], s[12:13], 7 // (op 1751) 
s_lshl_b64 s[18:19], s[14:15], 1 // (op 1751) 

v_mov_b32 v37, s18 // (op 1751) 
v_mov_b32 v40, s19 // (op 1751) 
v_add_co_u32 v42, s[14:15], v38, v37 // (op 1751) least significant half
v_addc_co_u32 v43, s[14:15], v39, v40, s[14:15] // (op 1751) most significant half




s_mov_b32 m0, s0 // (op 1751) reg expression
buffer_load_dwordx4 v42, s[8:11], 0 offen offset: 0 lds // (op 1751) Load value direct to lds
v_add_co_u32 v42, s[14:15], v42, v24 // (op 1751) least significant half
v_addc_co_u32 v43, s[14:15], 0, v43, s[14:15] // (op 1751) most significant half

s_add_u32 m0, m0, 4096 // (op 1751) 
buffer_load_dwordx4 v42, s[8:11], 0 offen offset: 0 lds // (op 1751) Load value direct to lds
v_add_co_u32 v42, s[14:15], v42, v24 // (op 1751) least significant half
v_addc_co_u32 v43, s[14:15], 0, v43, s[14:15] // (op 1751) most significant half

s_add_u32 m0, m0, 4096 // (op 1751) 
buffer_load_dwordx4 v42, s[8:11], 0 offen offset: 0 lds // (op 1751) Load value direct to lds
v_add_co_u32 v42, s[14:15], v42, v24 // (op 1751) least significant half
v_addc_co_u32 v43, s[14:15], 0, v43, s[14:15] // (op 1751) most significant half

s_add_u32 m0, m0, 4096 // (op 1751) 
buffer_load_dwordx4 v42, s[8:11], 0 offen offset: 0 lds // (op 1751) Load value direct to lds



v_mov_b64 v[24:25], s[28:29] // (op 17) convert

v_mov_b32 v37, v21 // (op 17) 
s_lshl_b32 s7, s28, 3 // (op 17) 
v_add_u32 v40, v37, s7 // (op 17) 

buffer_load_ubyte v32, v40, s[52:55], 0 offen offset: 0 // (op 17) Load value
v_add_u32 v40, v40, v24 // (op 17) 
buffer_load_ubyte v33, v40, s[52:55], 0 offen offset: 0 // (op 17) Load value


s_barrier  // (op 2431) 



s_waitcnt vmcnt(14)
ds_write_b96 v13, v[34:36] offset:37376 // (op 239) Store local data 
s_barrier  // (op 2437) 



s_waitcnt vmcnt(9)
ds_write_b8 v23, v18 offset:41472 // (op 240) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v23, v19 offset:41600 // (op 240) Store local data 
s_barrier  // (op 2422) 



ds_read_b64_tr_b8 v[40:41], v7 offset:16384 // (op 231) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v7 offset:17408 // (op 231) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v7 offset:24576 // (op 231) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v7 offset:25600 // (op 231) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 2428) 



ds_read_b96_tr_b6 v[48:50], v3 offset:37376 // (op 233) Transpose load local data 
ds_read_b96_tr_b6 v[51:53], v3 offset:37888 // (op 233) Transpose load local data 
s_barrier  // (op 2470) 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 2434) 


ds_read_u8 v24, v8 offset:41472 // (op 235) Load local data 



ds_read_b64_tr_b8 v[54:55], v7 offset:16416 // (op 357) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v7 offset:17440 // (op 357) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[40:47], v[48:53], a[0:3], v24, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 238) 
ds_read_b64_tr_b8 v[58:59], v7 offset:24608 // (op 357) Transpose load local data 


ds_read_b64_tr_b8 v[60:61], v7 offset:25632 // (op 357) Transpose load local data 


ds_read_u8 v24, v8 offset:41504 // (op 361) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[54:61], v[48:53], a[4:7], v24, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 351) 
ds_read_b64_tr_b8 v[40:41], v7 offset:16448 // (op 519) Transpose load local data 


ds_read_b64_tr_b8 v[42:43], v7 offset:17472 // (op 519) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v7 offset:24640 // (op 519) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v7 offset:25664 // (op 519) Transpose load local data 


ds_read_u8 v24, v8 offset:41536 // (op 523) Load local data 



ds_read_b64_tr_b8 v[54:55], v7 offset:16480 // (op 681) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[40:47], v[48:53], a[8:11], v24, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 513) 
ds_read_b64_tr_b8 v[56:57], v7 offset:17504 // (op 681) Transpose load local data 


ds_read_b64_tr_b8 v[58:59], v7 offset:24672 // (op 681) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v7 offset:25696 // (op 681) Transpose load local data 


ds_read_u8 v24, v8 offset:41568 // (op 685) Load local data 
s_barrier  // (op 2520) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[54:61], v[48:53], a[12:15], v24, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 675) 




v_mov_b64 v[24:25], s[34:35] // (op 237) convert

s_lshl_b32 s7, s34, 2 // (op 237) 
s_mul_i32 s14, 3, s7 // (op 237) 

v_add_u32 v40, v20, s14 // (op 237) 

v_mov_b32 v41, v40 // (op 237) 
buffer_load_ubyte v27, v41, s[56:59], 0 offen offset: 0 // (op 237) Load value






s_lshl_b32 s7, s34, 3 // (op 2017) 
v_add_u32 v22, v22, s7 // (op 2017) 

s_lshl_b64 s[14:15], s[24:25], 7 // (op 241) 
s_mul_i32 s19, s15, 6 // (op 241) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 241) least significant
s_mul_hi_u32 s7, s14, 6 // (op 241) most significant: high of low * low
s_add_u32 s19, s19, s7 // (op 241) most significant: sum


s_lshr_b64 s[14:15], s[18:19], 3 // (op 241) 

s_mul_i32 s7, 3, s14 // (op 241) 

v_add_u32 v20, v31, s7 // (op 241) 


buffer_load_dwordx3 v[34:36], v20, s[48:51], 0 offen // (op 241) Load value


s_barrier  // (op 2425) 
s_lshl_b64 s[14:15], s[24:25], 7 // (op 1889) 

s_mul_i32 s19, s15, 6 // (op 1889) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 1889) least significant
s_mul_hi_u32 s7, s14, 6 // (op 1889) most significant: high of low * low
s_add_u32 s19, s19, s7 // (op 1889) most significant: sum


s_lshr_b64 s[14:15], s[18:19], 3 // (op 1889) 

s_lshl_b32 s7, s14, 1 // (op 1889) 

v_add_u32 v17, v17, s7 // (op 1889) 

s_lshl_b64 s[14:15], s[12:13], 5 // (op 1762) 
v_mov_b64 v[24:25], s[14:15] // (op 1762) convert


s_lshl_b64 s[14:15], s[12:13], 7 // (op 1762) 
s_mul_i32 s19, 3, s15 // (op 1762) most significant: low of low * high
s_mul_i32 s18, 3, s14 // (op 1762) least significant
s_mul_hi_u32 s7, 3, s14 // (op 1762) most significant: high of low * low
s_add_u32 s19, s19, s7 // (op 1762) most significant: sum


v_mov_b32 v20, s18 // (op 1762) 
v_mov_b32 v31, s19 // (op 1762) 
v_add_co_u32 v40, s[14:15], v38, v20 // (op 1762) least significant half
v_addc_co_u32 v41, s[14:15], v39, v31, s[14:15] // (op 1762) most significant half




s_add_u32 s7, 16384, s1 // (op 1762) 
s_mov_b32 m0, s7 // (op 1762) reg expression
buffer_load_dwordx4 v40, s[8:11], 0 offen offset: 0 lds // (op 1762) Load value direct to lds
v_add_co_u32 v40, s[14:15], v40, v24 // (op 1762) least significant half
v_addc_co_u32 v41, s[14:15], 0, v41, s[14:15] // (op 1762) most significant half

s_add_u32 m0, m0, 4096 // (op 1762) 
buffer_load_dwordx4 v40, s[8:11], 0 offen offset: 0 lds // (op 1762) Load value direct to lds
v_add_co_u32 v40, s[14:15], v40, v24 // (op 1762) least significant half
v_addc_co_u32 v41, s[14:15], 0, v41, s[14:15] // (op 1762) most significant half

s_add_u32 m0, m0, 4096 // (op 1762) 
buffer_load_dwordx4 v40, s[8:11], 0 offen offset: 0 lds // (op 1762) Load value direct to lds
v_add_co_u32 v40, s[14:15], v40, v24 // (op 1762) least significant half
v_addc_co_u32 v41, s[14:15], 0, v41, s[14:15] // (op 1762) most significant half

s_add_u32 m0, m0, 4096 // (op 1762) 
buffer_load_dwordx4 v40, s[8:11], 0 offen offset: 0 lds // (op 1762) Load value direct to lds





v_mov_b64 v[24:25], s[28:29] // (op 242) convert

s_lshl_b32 s7, s28, 2 // (op 242) 
s_mul_i32 s14, 3, s7 // (op 242) 

v_add_u32 v20, v37, s14 // (op 242) 

s_lshl_b64 s[14:15], s[12:13], 7 // (op 1825) 
s_lshl_b64 s[18:19], s[14:15], 1 // (op 1825) 

v_mov_b32 v31, s18 // (op 1825) 
v_mov_b32 v38, s19 // (op 1825) 
v_add_co_u32 v14, s[14:15], v14, v31 // (op 1825) least significant half
v_addc_co_u32 v15, s[14:15], v15, v38, s[14:15] // (op 1825) most significant half




buffer_load_ubyte v18, v20, s[52:55], 0 offen offset: 0 // (op 242) Load value
v_add_u32 v20, v20, v24 // (op 242) 
buffer_load_ubyte v19, v20, s[52:55], 0 offen offset: 0 // (op 242) Load value



s_barrier  // (op 2397) 
s_lshl_b32 s7, s28, 3 // (op 1953) 



v_add_u32 v21, v21, s7 // (op 1953) 

s_waitcnt vmcnt(14)
ds_write_b96 v16, v[28:30] offset:32768 // (op 13) Store local data 
s_barrier  // (op 2411) 



s_waitcnt vmcnt(9)
ds_write_b8 v12, v32 offset:36864 // (op 21) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v12, v33 offset:36992 // (op 21) Store local data 
s_ashr_i64 s[14:15], s[22:23], 7 // (op 638) 
s_mov_b32 s7, s14 // (op 638) convert

s_and_b32 s14, s7, 4294967294 // (op 638) 

s_cmp_lt_u32 s6, s14 // (op 638) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopTop_KLoop_638 // (op 638) Condition: Bottom (jump to RR_GEMM_NT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopTop_KLoop_638 if true)
RR_GEMM_NT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopBottom_KLoop_638:














s_barrier  // (op 2502) 
v_and_b32 v12, -8, v1 // (op 1785) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 1785) 
v_sub_u32 v13, v1, v12 // (op 1785) 

s_lshr_b32 s6, s0, 1 // (op 1785) 

v_lshlrev_b32 v12, 4, v13 // (op 1785) 

v_lshl_add_u32 v13, s16, 7, v12 // (op 1785) 

v_mov_b32 v15, 0 // (op 1785) convert
v_mov_b32 v14, v13 // (op 1785) convert

v_lshrrev_b32 v12, 3, v1 // (op 1785) 
v_lshl_add_u32 v13, s6, 8, v12 // (op 1785) 


v_mul_lo_u32 v17, v13, s13 // (op 1785) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 1785) most significant: high of low * low
v_mul_lo_u32 v16, v13, s12 // (op 1785) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 1785) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v16 // (op 1785) least significant half
v_addc_co_u32 v13, s[0:1], v15, v17, s[0:1] // (op 1785) most significant half



s_mul_i32 s1, s12, s23 // (op 1787) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 1787) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 1787) least significant
s_mul_hi_u32 s7, s12, s22 // (op 1787) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1787) most significant: sum
s_add_u32 s1, s1, s7 // (op 1787) most significant: sum


s_mov_b32 s8, s20 // (op 1787) reg expression
s_mov_b32 s9, s21 // (op 1787) reg expression
s_mov_b32 s10, s0 // (op 1787) reg expression
s_mov_b32 s11, 131072 // (op 1787) call()

v_lshrrev_b32 v14, 1, v1 // (op 1855) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 1855) 
v_and_b32 v15, -2, v1 // (op 1855) 
s_lshr_b32 s6, s0, 1 // (op 1855) 

v_sub_u32 v16, v1, v15 // (op 1855) 

v_lshl_add_u32 v15, s6, 8, v14 // (op 1855) 


v_lshlrev_b32 v14, 4, v16 // (op 1855) 

v_lshl_add_u32 v16, s17, 5, v14 // (op 1855) 

v_mov_b32 v21, 0 // (op 1855) convert
v_mov_b32 v20, v16 // (op 1855) convert

v_mul_lo_u32 v17, v15, s25 // (op 1855) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 1855) most significant: high of low * low
v_mul_lo_u32 v16, v15, s24 // (op 1855) least significant: low of low * low
v_add_u32 v17, v17, v14 // (op 1855) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v20 // (op 1855) least significant half
v_addc_co_u32 v17, s[0:1], v17, v21, s[0:1] // (op 1855) most significant half



v_mul_lo_u32 v15, v17, 6 // (op 1855) most significant: low of high * low
v_mul_hi_u32 v20, v16, 6 // (op 1855) most significant: high of low * low
v_mul_lo_u32 v14, v16, 6 // (op 1855) least significant: low of low * low
v_add_u32 v15, v15, v20 // (op 1855) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 1855) 

v_mov_b32 v14, v16 // (op 1855) convert

s_mul_i32 s1, s23, 6 // (op 1857) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 1857) least significant
s_mul_hi_u32 s6, s22, 6 // (op 1857) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1857) most significant: sum

s_mul_i32 s7, s24, s1 // (op 1857) most significant: low of low * high
s_mul_i32 s14, s25, s0 // (op 1857) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 1857) least significant
s_mul_hi_u32 s15, s24, s0 // (op 1857) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 1857) most significant: sum
s_add_u32 s7, s7, s15 // (op 1857) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 1857) 

s_mov_b32 s48, s26 // (op 1857) reg expression
s_mov_b32 s49, s27 // (op 1857) reg expression
s_mov_b32 s50, s0 // (op 1857) reg expression
s_mov_b32 s51, 131072 // (op 1857) call()

v_and_b32 v15, -128, v1 // (op 1919) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 1919) 
v_sub_u32 v16, v1, v15 // (op 1919) 

s_lshr_b32 s6, s0, 1 // (op 1919) 

v_lshl_add_u32 v15, s16, 7, v16 // (op 1919) 

v_lshrrev_b32 v16, 7, v1 // (op 1919) 
v_lshlrev_b32 v17, 1, v16 // (op 1919) 

v_lshl_add_u32 v16, s6, 3, v17 // (op 1919) 


v_mul_lo_u32 v17, v16, s28 // (op 1919) 

v_add_u32 v16, v15, v17 // (op 1919) 


s_mul_i32 s1, s28, s33 // (op 1921) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 1921) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1921) least significant
s_mul_hi_u32 s7, s28, s32 // (op 1921) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1921) most significant: sum
s_add_u32 s1, s1, s7 // (op 1921) most significant: sum


s_mov_b32 s52, s30 // (op 1921) reg expression
s_mov_b32 s53, s31 // (op 1921) reg expression
s_mov_b32 s54, s0 // (op 1921) reg expression
s_mov_b32 s55, 131072 // (op 1921) call()

v_and_b32 v15, 63, v1 // (op 1983) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 1983) 
v_lshrrev_b32 v17, 4, v15 // (op 1983) 

s_lshr_b32 s6, s0, 1 // (op 1983) 

v_lshrrev_b32 v15, 6, v1 // (op 1983) 
v_lshl_add_u32 v20, s6, 3, v17 // (op 1983) 


v_and_b32 v17, 1, v15 // (op 1983) 

v_and_b32 v15, 15, v1 // (op 1983) 
v_lshl_add_u32 v21, v17, 4, v15 // (op 1983) 


v_lshl_add_u32 v15, s17, 5, v21 // (op 1983) 

v_mul_lo_u32 v17, v20, s34 // (op 1983) 
v_add_u32 v17, v17, v15 // (op 1983) 


s_mul_i32 s1, s34, s33 // (op 1985) most significant: low of low * high
s_mul_i32 s6, s35, s32 // (op 1985) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 1985) least significant
s_mul_hi_u32 s7, s34, s32 // (op 1985) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1985) most significant: sum
s_add_u32 s1, s1, s7 // (op 1985) most significant: sum


s_mov_b32 s56, s36 // (op 1985) reg expression
s_mov_b32 s57, s37 // (op 1985) reg expression
s_mov_b32 s58, s0 // (op 1985) reg expression
s_mov_b32 s59, 131072 // (op 1985) call()

v_and_b32 v15, -8, v1 // (op 2076) 
v_sub_u32 v20, v1, v15 // (op 2076) 

v_lshlrev_b32 v15, 4, v20 // (op 2076) 

v_lshrrev_b32 v20, 3, v1 // (op 2076) 
v_lshl_add_u32 v21, v20, 7, v15 // (op 2076) 


s_nop 0
v_readfirstlane_b32 s0, v21 // (op 2076) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v15, -2, v1 // (op 2133) 
v_sub_u32 v20, v1, v15 // (op 2133) 

v_lshlrev_b32 v15, 4, v20 // (op 2133) 

v_lshrrev_b32 v20, 1, v1 // (op 2133) 
v_lshl_add_u32 v21, v20, 5, v15 // (op 2133) 


v_mul_lo_u32 v15, v21, 6 // (op 2133) 
v_lshrrev_b32 v20, 3, v15 // (op 2133) 

v_lshrrev_b32 v15, 4, v21 // (op 2133) 

v_lshlrev_b32 v21, 2, v15 // (op 2133) 

v_add_u32 v15, v20, v21 // (op 2133) 


v_and_b32 v20, -128, v1 // (op 2186) 
v_sub_u32 v21, v1, v20 // (op 2186) 

v_lshrrev_b32 v20, 7, v1 // (op 2186) 
v_lshl_add_u32 v22, v20, 8, v21 // (op 2186) 


s_ashr_i64 s[6:7], s[22:23], 7 // (op 644) 
s_and_b32 s1, s6, 4294967294 // (op 644) 

s_ashr_i64 s[6:7], s[22:23], 7 // (op 643) 
s_mov_b32 s14, s6 // (op 643) convert

s_cmp_lt_u32 s1, s14 // (op 643) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopBottom_KLoopTail_643 // (op 643) Condition: Top (jump to RR_GEMM_NT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopBottom_KLoopTail_643 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopTop_KLoopTail_643:
s_barrier  // (op 2387) 
s_add_u32 s1, s1, 1 // (op 645) 

s_lshl_b64 s[6:7], s[12:13], 5 // (op 1756) 
v_mov_b64 v[18:19], s[6:7] // (op 1756) convert


v_mov_b64 v[20:21], v[12:13] // (op 1756) 
v_mov_b64 v[24:25], v[20:21] // (op 1756) reg expression
s_mov_b32 m0, s0 // (op 1756) reg expression
buffer_load_dwordx4 v24, s[8:11], 0 offen offset: 0 lds // (op 1756) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v18 // (op 1756) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 1756) most significant half

s_add_u32 m0, m0, 4096 // (op 1756) 
buffer_load_dwordx4 v24, s[8:11], 0 offen offset: 0 lds // (op 1756) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v18 // (op 1756) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 1756) most significant half

s_add_u32 m0, m0, 4096 // (op 1756) 
buffer_load_dwordx4 v24, s[8:11], 0 offen offset: 0 lds // (op 1756) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v18 // (op 1756) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 1756) most significant half

s_add_u32 m0, m0, 4096 // (op 1756) 
buffer_load_dwordx4 v24, s[8:11], 0 offen offset: 0 lds // (op 1756) Load value direct to lds



s_lshl_b64 s[6:7], s[12:13], 7 // (op 1775) 
v_mov_b32 v18, s6 // (op 1775) 
v_mov_b32 v19, s7 // (op 1775) 
v_add_co_u32 v12, s[14:15], v12, v18 // (op 1775) least significant half
v_addc_co_u32 v13, s[14:15], v13, v19, s[14:15] // (op 1775) most significant half






v_mov_b32 v18, v14 // (op 183) 
v_mov_b32 v19, v18 // (op 183) reg expression

buffer_load_dwordx3 v[24:26], v19, s[48:51], 0 offen // (op 183) Load value


s_lshl_b64 s[6:7], s[24:25], 7 // (op 1847) 
s_mul_i32 s15, s7, 6 // (op 1847) most significant: low of high * low
s_mul_i32 s14, s6, 6 // (op 1847) least significant
s_mul_hi_u32 s18, s6, 6 // (op 1847) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 1847) most significant: sum


s_lshr_b64 s[6:7], s[14:15], 3 // (op 1847) 

v_add_u32 v14, v14, s6 // (op 1847) 


s_barrier  // (op 2400) 
v_mov_b64 v[18:19], s[28:29] // (op 184) convert




v_mov_b32 v23, v16 // (op 184) 
v_mov_b32 v20, v23 // (op 184) reg expression
buffer_load_ubyte v28, v20, s[52:55], 0 offen offset: 0 // (op 184) Load value
v_add_u32 v20, v20, v18 // (op 184) 
s_waitcnt vmcnt(1)
ds_write_b96 v15, v[24:26] offset:32768 // (op 181) Store local data 
buffer_load_ubyte v29, v20, s[52:55], 0 offen offset: 0 // (op 184) Load value




s_lshl_b32 s6, s28, 2 // (op 1911) 
v_add_u32 v16, v16, s6 // (op 1911) 

s_barrier  // (op 2390) 
s_barrier  // (op 2414) 






ds_read_b64_tr_b8 v[30:31], v4 // (op 173) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v4 offset:1024 // (op 173) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v4 offset:8192 // (op 173) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v4 offset:9216 // (op 173) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v22, v28 offset:36864 // (op 182) Store local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 2404) 
s_waitcnt vmcnt(0)
ds_write_b8 v22, v29 offset:36992 // (op 182) Store local data 




ds_read_b96_tr_b6 v[24:26], v0 offset:32768 // (op 175) Transpose load local data 
ds_read_b96_tr_b6 v[27:29], v0 offset:33280 // (op 175) Transpose load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 2418) 


ds_read_u8 v23, v5 offset:36864 // (op 177) Load local data 

v_mov_b64 v[18:19], s[34:35] // (op 179) convert

v_mov_b32 v20, v17 // (op 179) 
v_mov_b32 v21, v20 // (op 179) reg expression
v_mov_b32 v38, v21 // (op 179) 
buffer_load_ubyte v39, v38, s[56:59], 0 offen offset: 0 // (op 179) Load value




s_lshl_b32 s6, s34, 2 // (op 1975) 
v_add_u32 v17, v17, s6 // (op 1975) 




ds_read_b64_tr_b8 v[40:41], v4 offset:32 // (op 325) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v4 offset:1056 // (op 325) Transpose load local data 
s_waitcnt vmcnt(0) lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[30:37], v[24:29], a[0:3], v23, v39 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 180) 
ds_read_b64_tr_b8 v[44:45], v4 offset:8224 // (op 325) Transpose load local data 


ds_read_b64_tr_b8 v[46:47], v4 offset:9248 // (op 325) Transpose load local data 


ds_read_u8 v23, v5 offset:36896 // (op 329) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[40:47], v[24:29], a[4:7], v23, v39 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 331) 
ds_read_b64_tr_b8 v[30:31], v4 offset:64 // (op 487) Transpose load local data 


ds_read_b64_tr_b8 v[32:33], v4 offset:1088 // (op 487) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v4 offset:8256 // (op 487) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v4 offset:9280 // (op 487) Transpose load local data 


ds_read_u8 v23, v5 offset:36928 // (op 491) Load local data 



ds_read_b64_tr_b8 v[40:41], v4 offset:96 // (op 649) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[30:37], v[24:29], a[8:11], v23, v39 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 493) 
ds_read_b64_tr_b8 v[42:43], v4 offset:1120 // (op 649) Transpose load local data 


ds_read_b64_tr_b8 v[44:45], v4 offset:8288 // (op 649) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v4 offset:9312 // (op 649) Transpose load local data 


ds_read_u8 v23, v5 offset:36960 // (op 653) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[40:47], v[24:29], a[12:15], v23, v39 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 655) 




s_ashr_i64 s[6:7], s[22:23], 7 // (op 643) 
s_mov_b32 s14, s6 // (op 643) convert

s_cmp_lt_u32 s1, s14 // (op 643) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopTop_KLoopTail_643 // (op 643) Condition: Bottom (jump to RR_GEMM_NT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopTop_KLoopTail_643 if true)
RR_GEMM_NT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ForLoopBottom_KLoopTail_643:






















s_barrier  // (op 2506) 






v_cmp_eq_f32 vcc, 0.00000, v10 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_NT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ConditionalFalse_0_beta_125
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


buffer_store_dwordx2 v[4:5], v9, s[44:47], 0 offen // (op 54) Store value


v_accvgpr_read v12, a4 // (op 885) 
v_accvgpr_read v13, a5 // (op 885) 
v_accvgpr_read v14, a6 // (op 885) 
v_accvgpr_read v15, a7 // (op 885) 
v_mul_f32 v16, v11, v12 // (op 885) 
v_mul_f32 v17, v11, v13 // (op 885) 
v_mul_f32 v18, v11, v14 // (op 885) 
v_mul_f32 v19, v11, v15 // (op 885) 

v_cvt_f16_f32 v12, v16 // (op 887) 
v_cvt_f16_f32 v13, v17 // (op 887) 
v_cvt_f16_f32 v14, v18 // (op 887) 
v_cvt_f16_f32 v15, v19 // (op 887) 


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

v_cvt_f16_f32 v12, v16 // (op 907) 
v_cvt_f16_f32 v13, v17 // (op 907) 
v_cvt_f16_f32 v14, v18 // (op 907) 
v_cvt_f16_f32 v15, v19 // (op 907) 


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

v_cvt_f16_f32 v12, v16 // (op 927) 
v_cvt_f16_f32 v13, v17 // (op 927) 
v_cvt_f16_f32 v14, v18 // (op 927) 
v_cvt_f16_f32 v15, v19 // (op 927) 


v_pack_b32_f16 v4, v12, v13 // (op 928) 
v_pack_b32_f16 v5, v14, v15 // (op 928) 


buffer_store_dwordx2 v[4:5], v9, s[44:47], 0 offen offset:192 // (op 928) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_NT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ConditionalBottom_0_beta_125
RR_GEMM_NT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ConditionalFalse_0_beta_125:



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
v_cvt_f32_f16 v12, v4 // (op 1009) 
v_lshrrev_b32 v13, 16, v4 // (op 1009) 
v_cvt_f32_f16 v13, v13 // (op 1009) 
v_mul_f32 v20, v10, v12 // (op 1009) 
v_mul_f32 v21, v10, v13 // (op 1009) 
v_cvt_f32_f16 v12, v5 // (op 1009) 
v_lshrrev_b32 v13, 16, v5 // (op 1009) 
v_cvt_f32_f16 v13, v13 // (op 1009) 
v_mul_f32 v22, v10, v12 // (op 1009) 
v_mul_f32 v23, v10, v13 // (op 1009) 


v_add_f32 v12, v20, v16 // (op 1005) 
v_add_f32 v13, v21, v17 // (op 1005) 
v_add_f32 v14, v22, v18 // (op 1005) 
v_add_f32 v15, v23, v19 // (op 1005) 


v_cvt_f16_f32 v16, v12 // (op 1006) 
v_cvt_f16_f32 v17, v13 // (op 1006) 
v_cvt_f16_f32 v18, v14 // (op 1006) 
v_cvt_f16_f32 v19, v15 // (op 1006) 


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
v_cvt_f32_f16 v12, v4 // (op 989) 
v_lshrrev_b32 v13, 16, v4 // (op 989) 
v_cvt_f32_f16 v13, v13 // (op 989) 
v_mul_f32 v20, v10, v12 // (op 989) 
v_mul_f32 v21, v10, v13 // (op 989) 
v_cvt_f32_f16 v12, v5 // (op 989) 
v_lshrrev_b32 v13, 16, v5 // (op 989) 
v_cvt_f32_f16 v13, v13 // (op 989) 
v_mul_f32 v22, v10, v12 // (op 989) 
v_mul_f32 v23, v10, v13 // (op 989) 


v_add_f32 v12, v20, v16 // (op 985) 
v_add_f32 v13, v21, v17 // (op 985) 
v_add_f32 v14, v22, v18 // (op 985) 
v_add_f32 v15, v23, v19 // (op 985) 


v_cvt_f16_f32 v16, v12 // (op 986) 
v_cvt_f16_f32 v17, v13 // (op 986) 
v_cvt_f16_f32 v18, v14 // (op 986) 
v_cvt_f16_f32 v19, v15 // (op 986) 


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
v_cvt_f32_f16 v12, v4 // (op 969) 
v_lshrrev_b32 v13, 16, v4 // (op 969) 
v_cvt_f32_f16 v13, v13 // (op 969) 
v_mul_f32 v20, v10, v12 // (op 969) 
v_mul_f32 v21, v10, v13 // (op 969) 
v_cvt_f32_f16 v12, v5 // (op 969) 
v_lshrrev_b32 v13, 16, v5 // (op 969) 
v_cvt_f32_f16 v13, v13 // (op 969) 
v_mul_f32 v22, v10, v12 // (op 969) 
v_mul_f32 v23, v10, v13 // (op 969) 


v_add_f32 v12, v20, v16 // (op 965) 
v_add_f32 v13, v21, v17 // (op 965) 
v_add_f32 v14, v22, v18 // (op 965) 
v_add_f32 v15, v23, v19 // (op 965) 


v_cvt_f16_f32 v16, v12 // (op 966) 
v_cvt_f16_f32 v17, v13 // (op 966) 
v_cvt_f16_f32 v18, v14 // (op 966) 
v_cvt_f16_f32 v19, v15 // (op 966) 


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


buffer_store_dwordx2 v[4:5], v9, s[44:47], 0 offen // (op 137) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_ConditionalBottom_0_beta_125:












s_endpgm  // End of RR_GEMM_NT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2
.LRR_GEMM_NT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_end:
.size RR_GEMM_NT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2, .LRR_GEMM_NT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_end-RR_GEMM_NT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2
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
    DataTypeB: B6
    DestDataType: h
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
  - .name: RR_GEMM_NT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2
    .symbol: RR_GEMM_NT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2.kd
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

