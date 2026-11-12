// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2
.p2align 8
.type RR_GEMM_TT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2,@function
RR_GEMM_TT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 2408) 
v_and_b32 v3, 15, v1 // (op 2408) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 2408) 


v_and_b32 v0, 63, v1 // (op 2408) 
v_lshrrev_b32 v3, 4, v0 // (op 2408) 

v_lshlrev_b32 v0, 5, v3 // (op 2408) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 2408) 


v_lshlrev_b32 v0, 2, v3 // (op 2408) 

v_lshrrev_b32 v3, 3, v0 // (op 2408) 

v_and_b32 v0, 48, v1 // (op 2466) 
v_and_b32 v4, 63, v1 // (op 2466) 
v_sub_u32 v5, v4, v0 // (op 2466) 

v_lshrrev_b32 v4, 1, v5 // (op 2466) 
v_add_lshl_u32 v6, v0, v4, 6 // (op 2466) 


v_and_b32 v0, 1, v5 // (op 2466) 

v_lshlrev_b32 v4, 3, v0 // (op 2466) 

v_lshrrev_b32 v0, 6, v1 // (op 2466) 
v_and_b32 v5, 1, v0 // (op 2466) 

v_lshl_add_u32 v0, v5, 4, v4 // (op 2466) 


v_add_u32 v4, v6, v0 // (op 2466) 


v_lshrrev_b32 v0, 7, v1 // (op 2524) 
v_and_b32 v5, 15, v1 // (op 2524) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 2524) 


v_and_b32 v0, 63, v1 // (op 2524) 
v_lshrrev_b32 v5, 4, v0 // (op 2524) 

v_lshl_add_u32 v0, v6, 2, v5 // (op 2524) 


v_lshrrev_b32 v5, 6, v1 // (op 2582) 
v_and_b32 v6, 1, v5 // (op 2582) 

v_and_b32 v5, 15, v1 // (op 2582) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 2582) 


v_and_b32 v5, 63, v1 // (op 2582) 
v_lshrrev_b32 v6, 4, v5 // (op 2582) 

v_lshl_add_u32 v5, v6, 6, v7 // (op 2582) 


v_lshrrev_b32 v6, 7, v1 // (op 2634) 
v_and_b32 v7, 15, v1 // (op 2634) 
v_lshl_add_u32 v8, v6, 4, v7 // (op 2634) 


v_and_b32 v6, 63, v1 // (op 2634) 
v_lshrrev_b32 v7, 4, v6 // (op 2634) 

v_lshlrev_b32 v6, 5, v7 // (op 2634) 

v_lshl_add_u32 v7, v8, 7, v6 // (op 2634) 


v_lshlrev_b32 v6, 2, v7 // (op 2634) 

v_lshrrev_b32 v7, 3, v6 // (op 2634) 

v_and_b32 v6, 48, v1 // (op 2663) 
v_and_b32 v8, 63, v1 // (op 2663) 
v_sub_u32 v9, v8, v6 // (op 2663) 

v_lshrrev_b32 v8, 1, v9 // (op 2663) 
v_add_lshl_u32 v10, v6, v8, 6 // (op 2663) 


v_and_b32 v6, 1, v9 // (op 2663) 

v_lshlrev_b32 v8, 3, v6 // (op 2663) 

v_lshrrev_b32 v6, 6, v1 // (op 2663) 
v_and_b32 v9, 1, v6 // (op 2663) 

v_lshl_add_u32 v6, v9, 4, v8 // (op 2663) 


v_add_u32 v8, v10, v6 // (op 2663) 


v_lshrrev_b32 v6, 7, v1 // (op 2692) 
v_and_b32 v9, 15, v1 // (op 2692) 
v_lshl_add_u32 v10, v6, 4, v9 // (op 2692) 


v_and_b32 v6, 63, v1 // (op 2692) 
v_lshrrev_b32 v9, 4, v6 // (op 2692) 

v_lshl_add_u32 v6, v10, 2, v9 // (op 2692) 


v_lshrrev_b32 v9, 6, v1 // (op 2721) 
v_and_b32 v10, 1, v9 // (op 2721) 

v_and_b32 v9, 15, v1 // (op 2721) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 2721) 


v_and_b32 v9, 63, v1 // (op 2721) 
v_lshrrev_b32 v10, 4, v9 // (op 2721) 

v_lshl_add_u32 v9, v10, 6, v11 // (op 2721) 


v_lshrrev_b32 v10, 7, v1 // (op 2346) 
v_and_b32 v11, 63, v1 // (op 2346) 
v_lshrrev_b32 v12, 4, v11 // (op 2346) 
v_lshlrev_b32 v13, 2, v12 // (op 2346) 

v_lshl_add_u32 v12, v10, 4, v13 // (op 2346) 


v_lshl_add_u32 v10, s16, 6, v12 // (op 2346) 

v_lshrrev_b32 v12, 2, v11 // (op 2346) 

v_and_b32 v11, 3, v12 // (op 2346) 

v_and_b32 v12, 3, v1 // (op 2346) 
v_lshl_add_u32 v13, v11, 2, v12 // (op 2346) 


v_lshrrev_b32 v11, 6, v1 // (op 2346) 
v_and_b32 v12, 1, v11 // (op 2346) 

v_lshl_add_u32 v11, v12, 4, v13 // (op 2346) 


v_lshl_add_u32 v12, s17, 6, v11 // (op 2346) 

v_mul_lo_u32 v11, v12, s2 // (op 2346) 

v_add_lshl_u32 v12, v10, v11, 2 // (op 2346) 


v_lshrrev_b32 v10, 7, v1 // (op 2603) 
v_and_b32 v11, 63, v1 // (op 2603) 
v_lshrrev_b32 v13, 4, v11 // (op 2603) 
s_mul_i32 s1, s2, s9 // (op 2348) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 2348) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 2348) least significant
s_mul_hi_u32 s19, s2, s8 // (op 2348) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2348) most significant: sum
s_add_u32 s1, s1, s19 // (op 2348) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 2348) 

v_lshlrev_b32 v14, 2, v13 // (op 2603) 

v_lshl_add_u32 v13, v10, 4, v14 // (op 2603) 


v_lshl_add_u32 v10, s16, 6, v13 // (op 2603) 

v_lshrrev_b32 v13, 2, v11 // (op 2603) 

v_and_b32 v11, 3, v13 // (op 2603) 

v_and_b32 v13, 3, v1 // (op 2603) 
v_lshl_add_u32 v14, v11, 2, v13 // (op 2603) 


v_lshrrev_b32 v11, 6, v1 // (op 2603) 
v_and_b32 v13, 1, v11 // (op 2603) 

v_lshl_add_u32 v11, v13, 4, v14 // (op 2603) 


v_lshl_add_u32 v13, s17, 6, v11 // (op 2603) 

v_mul_lo_u32 v11, v13, s4 // (op 2603) 

v_add_lshl_u32 v13, v10, v11, 2 // (op 2603) 


s_mov_b32 s40, s6 // (op 2348) reg expression
s_mov_b32 s41, s7 // (op 2348) reg expression
s_mov_b32 s42, s18 // (op 2348) reg expression
s_mov_b32 s43, 131072 // (op 2348) call()

s_mul_i32 s1, s4, s9 // (op 2605) most significant: low of low * high

s_mul_i32 s6, s5, s8 // (op 2605) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 2605) least significant
s_mul_hi_u32 s7, s4, s8 // (op 2605) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2605) most significant: sum
s_add_u32 s1, s1, s7 // (op 2605) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 2605) 

s_mov_b32 s44, s10 // (op 2605) reg expression
s_mov_b32 s45, s11 // (op 2605) reg expression
s_mov_b32 s46, s6 // (op 2605) reg expression
s_mov_b32 s47, 131072 // (op 2605) call()



v_mov_b32 v10, s14 // (op 2) Move value
v_mov_b32 v11, s15 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 63) call()
v_accvgpr_write a1, 0.00000 // (op 63) call()
v_accvgpr_write a2, 0.00000 // (op 63) call()
v_accvgpr_write a3, 0.00000 // (op 63) call()
v_accvgpr_write a4, 0.00000 // (op 336) call()
v_accvgpr_write a5, 0.00000 // (op 336) call()
v_accvgpr_write a6, 0.00000 // (op 336) call()
v_accvgpr_write a7, 0.00000 // (op 336) call()
v_accvgpr_write a8, 0.00000 // (op 556) call()
v_accvgpr_write a9, 0.00000 // (op 556) call()
v_accvgpr_write a10, 0.00000 // (op 556) call()
v_accvgpr_write a11, 0.00000 // (op 556) call()
v_accvgpr_write a12, 0.00000 // (op 711) call()
v_accvgpr_write a13, 0.00000 // (op 711) call()
v_accvgpr_write a14, 0.00000 // (op 711) call()
v_accvgpr_write a15, 0.00000 // (op 711) call()
v_lshrrev_b32 v14, 2, v1 // (op 2108) 
v_lshl_add_u32 v15, s16, 6, v14 // (op 2108) 

v_and_b32 v14, -4, v1 // (op 2108) 
v_sub_u32 v16, v1, v14 // (op 2108) 

v_lshlrev_b32 v14, 5, v16 // (op 2108) 

v_mov_b32 v17, 0 // (op 2108) convert
v_mov_b32 v16, v14 // (op 2108) convert

v_mul_lo_u32 v19, v15, s13 // (op 2108) most significant: low of low * high
v_mul_hi_u32 v14, v15, s12 // (op 2108) most significant: high of low * low
v_mul_lo_u32 v18, v15, s12 // (op 2108) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 2108) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 2108) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 2108) most significant half



v_lshlrev_b64 v[14:15], 2, v[18:19] // (op 2108) 

v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 2108) 

v_mov_b32 v14, v16 // (op 2108) convert


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 2110) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 2110) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 2110) least significant
s_mul_hi_u32 s7, s12, s22 // (op 2110) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2110) most significant: sum
s_add_u32 s1, s1, s7 // (op 2110) most significant: sum


s_lshl_b64 s[14:15], s[0:1], 2 // (op 2110) 

s_ashr_i64 s[0:1], s[14:15], 3 // (op 2110) 

s_mov_b32 s8, s20 // (op 2110) reg expression
s_mov_b32 s9, s21 // (op 2110) reg expression
s_mov_b32 s10, s0 // (op 2110) reg expression
s_mov_b32 s11, 131072 // (op 2110) call()

v_lshrrev_b32 v15, 2, v1 // (op 2127) 
v_lshl_add_u32 v16, s16, 6, v15 // (op 2127) 

v_and_b32 v15, -4, v1 // (op 2127) 
v_sub_u32 v17, v1, v15 // (op 2127) 

v_lshlrev_b32 v15, 5, v17 // (op 2127) 

v_mov_b32 v19, 0 // (op 2127) convert
v_mov_b32 v18, v15 // (op 2127) convert

v_mul_lo_u32 v21, v16, s13 // (op 2127) most significant: low of low * high
v_mul_hi_u32 v15, v16, s12 // (op 2127) most significant: high of low * low
v_mul_lo_u32 v20, v16, s12 // (op 2127) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 2127) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v18 // (op 2127) least significant half
v_addc_co_u32 v21, s[0:1], v21, v19, s[0:1] // (op 2127) most significant half



v_lshlrev_b64 v[18:19], 2, v[20:21] // (op 2127) 

v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 2127) 

v_mov_b32 v15, v16 // (op 2127) convert

s_mul_i32 s1, s12, s23 // (op 2129) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 2129) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 2129) least significant
s_mul_hi_u32 s7, s12, s22 // (op 2129) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2129) most significant: sum
s_add_u32 s1, s1, s7 // (op 2129) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 2129) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 2129) 

s_mov_b32 s8, s20 // (op 2129) reg expression
s_mov_b32 s9, s21 // (op 2129) reg expression
s_mov_b32 s10, s0 // (op 2129) reg expression
s_mov_b32 s11, 131072 // (op 2129) call()

v_and_b32 v16, -4, v1 // (op 2172) 
v_sub_u32 v17, v1, v16 // (op 2172) 

v_lshlrev_b32 v16, 4, v17 // (op 2172) 

v_lshl_add_u32 v17, s17, 6, v16 // (op 2172) 

v_lshrrev_b32 v16, 2, v1 // (op 2172) 
v_lshlrev_b32 v18, 1, v16 // (op 2172) 

v_mul_lo_u32 v16, v18, s24 // (op 2172) 
v_add_u32 v16, v16, v17 // (op 2172) 


s_mul_i32 s1, s24, s29 // (op 2174) most significant: low of low * high
s_mul_i32 s6, s25, s28 // (op 2174) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 2174) least significant
s_mul_hi_u32 s7, s24, s28 // (op 2174) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2174) most significant: sum
s_add_u32 s1, s1, s7 // (op 2174) most significant: sum


s_mov_b32 s48, s26 // (op 2174) reg expression
s_mov_b32 s49, s27 // (op 2174) reg expression
s_mov_b32 s50, s0 // (op 2174) reg expression
s_mov_b32 s51, 131072 // (op 2174) call()

v_and_b32 v17, -4, v1 // (op 2191) 
v_sub_u32 v18, v1, v17 // (op 2191) 

v_lshlrev_b32 v17, 4, v18 // (op 2191) 

v_lshl_add_u32 v18, s17, 6, v17 // (op 2191) 

v_lshrrev_b32 v17, 2, v1 // (op 2191) 
v_lshlrev_b32 v19, 1, v17 // (op 2191) 

v_mul_lo_u32 v17, v19, s24 // (op 2191) 
v_add_u32 v17, v17, v18 // (op 2191) 


s_mul_i32 s1, s24, s29 // (op 2193) most significant: low of low * high
s_mul_i32 s6, s25, s28 // (op 2193) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 2193) least significant
s_mul_hi_u32 s7, s24, s28 // (op 2193) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2193) most significant: sum
s_add_u32 s1, s1, s7 // (op 2193) most significant: sum


s_mov_b32 s48, s26 // (op 2193) reg expression
s_mov_b32 s49, s27 // (op 2193) reg expression
s_mov_b32 s50, s0 // (op 2193) reg expression
s_mov_b32 s51, 131072 // (op 2193) call()

v_lshrrev_b32 v18, 2, v1 // (op 2236) 
v_lshl_add_u32 v19, s16, 6, v18 // (op 2236) 

v_and_b32 v18, -4, v1 // (op 2236) 
v_sub_u32 v20, v1, v18 // (op 2236) 

v_mul_lo_u32 v18, v19, s30 // (op 2236) 
v_add_u32 v18, v18, v20 // (op 2236) 


s_mul_i32 s1, s30, s23 // (op 2238) most significant: low of low * high
s_mul_i32 s6, s31, s22 // (op 2238) most significant: low of high * low
s_mul_i32 s0, s30, s22 // (op 2238) least significant
s_mul_hi_u32 s7, s30, s22 // (op 2238) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2238) most significant: sum
s_add_u32 s1, s1, s7 // (op 2238) most significant: sum


s_mov_b32 s52, s32 // (op 2238) reg expression
s_mov_b32 s53, s33 // (op 2238) reg expression
s_mov_b32 s54, s0 // (op 2238) reg expression
s_mov_b32 s55, 131072 // (op 2238) call()

v_lshrrev_b32 v19, 2, v1 // (op 2255) 
v_lshl_add_u32 v20, s16, 6, v19 // (op 2255) 

v_and_b32 v19, -4, v1 // (op 2255) 
v_sub_u32 v21, v1, v19 // (op 2255) 

v_mul_lo_u32 v19, v20, s30 // (op 2255) 
v_add_u32 v19, v19, v21 // (op 2255) 


s_mul_i32 s1, s30, s23 // (op 2257) most significant: low of low * high
s_mul_i32 s6, s31, s22 // (op 2257) most significant: low of high * low
s_mul_i32 s0, s30, s22 // (op 2257) least significant
s_mul_hi_u32 s7, s30, s22 // (op 2257) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2257) most significant: sum
s_add_u32 s1, s1, s7 // (op 2257) most significant: sum


s_mov_b32 s52, s32 // (op 2257) reg expression
s_mov_b32 s53, s33 // (op 2257) reg expression
s_mov_b32 s54, s0 // (op 2257) reg expression
s_mov_b32 s55, 131072 // (op 2257) call()

v_and_b32 v20, -64, v1 // (op 2300) 
v_sub_u32 v21, v1, v20 // (op 2300) 

v_lshl_add_u32 v20, s17, 6, v21 // (op 2300) 

v_lshrrev_b32 v21, 6, v1 // (op 2300) 
v_mul_lo_u32 v22, v21, s34 // (op 2300) 
v_add_u32 v22, v22, v20 // (op 2300) 


s_mul_i32 s1, s34, s39 // (op 2302) most significant: low of low * high
s_mul_i32 s6, s35, s38 // (op 2302) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 2302) least significant
s_mul_hi_u32 s7, s34, s38 // (op 2302) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2302) most significant: sum
s_add_u32 s1, s1, s7 // (op 2302) most significant: sum


s_mov_b32 s56, s36 // (op 2302) reg expression
s_mov_b32 s57, s37 // (op 2302) reg expression
s_mov_b32 s58, s0 // (op 2302) reg expression
s_mov_b32 s59, 131072 // (op 2302) call()

v_and_b32 v20, -64, v1 // (op 2319) 
v_sub_u32 v21, v1, v20 // (op 2319) 

v_lshl_add_u32 v20, s17, 6, v21 // (op 2319) 

v_lshrrev_b32 v21, 6, v1 // (op 2319) 
v_mul_lo_u32 v23, v21, s34 // (op 2319) 
v_add_u32 v23, v23, v20 // (op 2319) 


s_mul_i32 s1, s34, s39 // (op 2321) most significant: low of low * high
s_mul_i32 s6, s35, s38 // (op 2321) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 2321) least significant
s_mul_hi_u32 s7, s34, s38 // (op 2321) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2321) most significant: sum
s_add_u32 s1, s1, s7 // (op 2321) most significant: sum


s_mov_b32 s56, s36 // (op 2321) reg expression
s_mov_b32 s57, s37 // (op 2321) reg expression
s_mov_b32 s58, s0 // (op 2321) reg expression
s_mov_b32 s59, 131072 // (op 2321) call()


v_mov_b64 v[20:21], s[24:25] // (op 1940) convert

v_mov_b32 v24, v16 // (op 1940) reg expression
buffer_load_dwordx4 v[26:29], v24, s[48:51], 0 offen // (op 1940) Load value
v_add_u32 v24, v24, v20 // (op 1940) 
buffer_load_dwordx4 v[30:33], v24, s[48:51], 0 offen // (op 1940) Load value




v_mov_b32 v20, v22 // (op 1943) reg expression

buffer_load_ubyte v21, v20, s[56:59], 0 offen offset: 0 // (op 1943) Load value




buffer_load_dwordx4 v[34:37], v14, s[8:11], 0 offen // (op 1946) Load value



buffer_load_ubyte v20, v18, s[52:55], 0 offen offset: 0 // (op 1949) Load value

v_mov_b64 v[24:25], s[24:25] // (op 1952) convert

s_lshl_b32 s0, s24, 7 // (op 1952) 
v_add_u32 v38, v16, s0 // (op 1952) 

buffer_load_dwordx4 v[40:43], v38, s[48:51], 0 offen // (op 1952) Load value
v_add_u32 v38, v38, v24 // (op 1952) 
buffer_load_dwordx4 v[44:47], v38, s[48:51], 0 offen // (op 1952) Load value





s_lshl_b32 s0, s34, 2 // (op 1955) 
v_add_u32 v16, v22, s0 // (op 1955) 


buffer_load_ubyte v24, v16, s[56:59], 0 offen offset: 0 // (op 1955) Load value





buffer_load_dwordx4 v[48:51], v14, s[8:11], 0 offen offset:64 // (op 1958) Load value




buffer_load_ubyte v14, v18, s[52:55], 0 offen offset:4 // (op 1961) Load value

v_and_b32 v16, -4, v1 // (op 2449) 
v_sub_u32 v18, v1, v16 // (op 2449) 

v_lshlrev_b32 v16, 4, v18 // (op 2449) 

v_lshrrev_b32 v18, 2, v1 // (op 2449) 
v_lshl_add_u32 v22, v18, 7, v16 // (op 2449) 





s_waitcnt vmcnt(9)
ds_write_b128 v22, v[26:29] // (op 1963) Store local data 
s_waitcnt vmcnt(8)
ds_write_b128 v22, v[30:33] offset:64 // (op 1963) Store local data 

v_and_b32 v16, -64, v1 // (op 2565) 
v_sub_u32 v18, v1, v16 // (op 2565) 

v_and_b32 v16, 4294967232, v1 // (op 2565) 
v_add_u32 v22, v16, v18 // (op 2565) 





s_waitcnt vmcnt(7)
ds_write_b8 v22, v21 offset:8192 // (op 1964) Store local data 

v_and_b32 v16, -4, v1 // (op 2391) 
v_sub_u32 v18, v1, v16 // (op 2391) 

v_lshlrev_b32 v16, 5, v18 // (op 2391) 

v_lshrrev_b32 v18, 2, v1 // (op 2391) 
v_lshl_add_u32 v22, v18, 7, v16 // (op 2391) 


v_lshlrev_b32 v16, 2, v22 // (op 2391) 

v_lshrrev_b32 v18, 3, v16 // (op 2391) 




s_waitcnt vmcnt(6)
ds_write_b128 v18, v[34:37] offset:8448 // (op 1965) Store local data 

v_and_b32 v16, -4, v1 // (op 2507) 
v_sub_u32 v18, v1, v16 // (op 2507) 

v_and_b32 v16, 4294967292, v1 // (op 2507) 
v_add_u32 v22, v16, v18 // (op 2507) 





s_waitcnt vmcnt(5)
ds_write_b8 v22, v20 offset:12544 // (op 1966) Store local data 

v_and_b32 v16, -4, v1 // (op 2362) 
v_sub_u32 v18, v1, v16 // (op 2362) 

v_lshlrev_b32 v16, 5, v18 // (op 2362) 

v_lshrrev_b32 v18, 2, v1 // (op 2362) 
v_lshl_add_u32 v22, v18, 7, v16 // (op 2362) 


v_lshlrev_b32 v16, 2, v22 // (op 2362) 

v_lshrrev_b32 v18, 3, v16 // (op 2362) 

v_and_b32 v16, -4, v1 // (op 2420) 
v_sub_u32 v22, v1, v16 // (op 2420) 

v_lshlrev_b32 v16, 4, v22 // (op 2420) 

v_lshrrev_b32 v22, 2, v1 // (op 2420) 
v_lshl_add_u32 v25, v22, 7, v16 // (op 2420) 


v_and_b32 v16, -4, v1 // (op 2478) 
v_sub_u32 v22, v1, v16 // (op 2478) 

v_and_b32 v16, 4294967292, v1 // (op 2478) 
v_add_u32 v38, v16, v22 // (op 2478) 


v_and_b32 v16, -64, v1 // (op 2536) 
v_sub_u32 v22, v1, v16 // (op 2536) 

v_and_b32 v16, 4294967232, v1 // (op 2536) 
v_add_u32 v39, v16, v22 // (op 2536) 


v_and_b32 v16, -4, v1 // (op 2617) 
v_sub_u32 v22, v1, v16 // (op 2617) 

v_lshlrev_b32 v16, 5, v22 // (op 2617) 

v_lshrrev_b32 v22, 2, v1 // (op 2617) 
v_lshl_add_u32 v52, v22, 7, v16 // (op 2617) 


v_lshlrev_b32 v16, 2, v52 // (op 2617) 

v_lshrrev_b32 v22, 3, v16 // (op 2617) 

v_and_b32 v16, -4, v1 // (op 2646) 
v_sub_u32 v52, v1, v16 // (op 2646) 

v_lshlrev_b32 v16, 4, v52 // (op 2646) 

v_lshrrev_b32 v52, 2, v1 // (op 2646) 
v_lshl_add_u32 v53, v52, 7, v16 // (op 2646) 


v_and_b32 v16, -4, v1 // (op 2675) 
v_sub_u32 v52, v1, v16 // (op 2675) 

v_and_b32 v16, 4294967292, v1 // (op 2675) 
v_add_u32 v54, v16, v52 // (op 2675) 


v_and_b32 v16, -64, v1 // (op 2704) 
v_sub_u32 v52, v1, v16 // (op 2704) 

v_and_b32 v16, 4294967232, v1 // (op 2704) 
v_add_u32 v55, v16, v52 // (op 2704) 


s_mov_b32 s0, 0 // (op 589) call()
s_ashr_i64 s[6:7], s[28:29], 7 // (op 588) 
s_mov_b32 s1, s6 // (op 588) convert

s_and_b32 s6, s1, 4294967294 // (op 588) 

s_cmp_lt_u32 s0, s6 // (op 588) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_ForLoopBottom_KLoop_588 // (op 588) Condition: Top (jump to RR_GEMM_TT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_ForLoopBottom_KLoop_588 if false)
RR_GEMM_TT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_ForLoopTop_KLoop_588:

s_add_u32 s0, s0, 2 // (op 590) 
v_mov_b64 v[56:57], s[24:25] // (op 10) convert

v_mov_b32 v16, v17 // (op 10) 
s_lshl_b32 s1, s24, 8 // (op 10) 
v_add_u32 v52, v16, s1 // (op 10) 

buffer_load_dwordx4 v[26:29], v52, s[48:51], 0 offen // (op 10) Load value
v_add_u32 v52, v52, v56 // (op 10) 
buffer_load_dwordx4 v[30:33], v52, s[48:51], 0 offen // (op 10) Load value


s_waitcnt lgkmcnt(1)
s_barrier  // (op 2772) 





v_mov_b32 v52, v23 // (op 25) 
s_lshl_b32 s1, s34, 3 // (op 25) 
v_add_u32 v56, v52, s1 // (op 25) 


ds_read_b128 v[58:61], v3 offset:8448 // (op 70) Load local data 
buffer_load_ubyte v21, v56, s[56:59], 0 offen offset: 0 // (op 25) Load value



s_barrier  // (op 2785) 
v_mov_b32 v56, v15 // (op 4) 




ds_read_b64_tr_b8 v[62:63], v4 // (op 75) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v4 offset:512 // (op 75) Transpose load local data 
ds_read_b64_tr_b8 v[66:67], v4 offset:4096 // (op 75) Transpose load local data 
ds_read_b64_tr_b8 v[68:69], v4 offset:4608 // (op 75) Transpose load local data 
buffer_load_dwordx4 v[34:37], v56, s[8:11], 0 offen offset:128 // (op 4) Load value


v_mov_b32 v57, v19 // (op 17) 

buffer_load_ubyte v20, v57, s[52:55], 0 offen offset:8 // (op 17) Load value
s_waitcnt lgkmcnt(5)
s_barrier  // (op 2799) 



s_barrier  // (op 2836) 
ds_read_u8 v70, v0 offset:12544 // (op 80) Load local data 



s_barrier  // (op 2813) 
s_waitcnt vmcnt(9)
ds_write_b128 v53, v[40:43] offset:12800 // (op 253) Store local data 


ds_read_u8 v71, v5 offset:8192 // (op 85) Load local data 
s_waitcnt vmcnt(8)
ds_write_b128 v53, v[44:47] offset:12864 // (op 253) Store local data 



s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[58:61], v[62:69], a[0:3], v70, v71 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 88) 
s_barrier  // (op 2848) 
ds_read_b64_tr_b8 v[72:73], v4 offset:32 // (op 371) Transpose load local data 



ds_read_b64_tr_b8 v[74:75], v4 offset:544 // (op 371) Transpose load local data 
ds_read_b64_tr_b8 v[76:77], v4 offset:4128 // (op 371) Transpose load local data 
ds_read_b64_tr_b8 v[78:79], v4 offset:4640 // (op 371) Transpose load local data 
s_waitcnt vmcnt(7)
ds_write_b8 v55, v24 offset:20992 // (op 255) Store local data 


ds_read_u8 v80, v5 offset:8224 // (op 375) Load local data 



s_barrier  // (op 2830) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[58:61], v[72:79], a[4:7], v70, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 376) 





ds_read_b128 v[58:61], v3 offset:10496 // (op 622) Load local data 
s_waitcnt vmcnt(6)
ds_write_b128 v22, v[48:51] offset:21248 // (op 243) Store local data 



ds_read_u8 v70, v0 offset:12672 // (op 625) Load local data 
s_barrier  // (op 2842) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[58:61], v[62:69], a[8:11], v70, v71 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 673) 





s_waitcnt vmcnt(5)
ds_write_b8 v54, v14 offset:25344 // (op 254) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[58:61], v[72:79], a[12:15], v70, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 627) 





s_barrier  // (op 2827) 
v_mov_b64 v[58:59], s[24:25] // (op 256) convert




s_lshl_b32 s1, s24, 7 // (op 256) 
s_mul_i32 s6, 3, s1 // (op 256) 

v_add_u32 v60, v16, s6 // (op 256) 

ds_read_b128 v[62:65], v7 offset:21248 // (op 245) Load local data 
buffer_load_dwordx4 v[40:43], v60, s[48:51], 0 offen // (op 256) Load value
s_barrier  // (op 2833) 
v_add_u32 v60, v60, v58 // (op 256) 



ds_read_b64_tr_b8 v[66:67], v8 offset:12800 // (op 247) Transpose load local data 
ds_read_b64_tr_b8 v[68:69], v8 offset:13312 // (op 247) Transpose load local data 
ds_read_b64_tr_b8 v[70:71], v8 offset:16896 // (op 247) Transpose load local data 
ds_read_b64_tr_b8 v[72:73], v8 offset:17408 // (op 247) Transpose load local data 
buffer_load_dwordx4 v[44:47], v60, s[48:51], 0 offen // (op 256) Load value



s_waitcnt lgkmcnt(5)
s_barrier  // (op 2839) 





ds_read_u8 v16, v6 offset:25344 // (op 249) Load local data 
s_lshl_b32 s1, s34, 2 // (op 258) 
s_barrier  // (op 2845) 
s_mul_i32 s6, 3, s1 // (op 258) 



s_lshl_b32 s1, s24, 8 // (op 2183) 
v_add_u32 v17, v17, s1 // (op 2183) 

v_add_u32 v58, v52, s6 // (op 258) 


buffer_load_ubyte v24, v58, s[56:59], 0 offen offset: 0 // (op 258) Load value

ds_read_u8 v58, v9 offset:20992 // (op 251) Load local data 







s_lshl_b32 s1, s34, 3 // (op 2311) 
v_add_u32 v23, v23, s1 // (op 2311) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:65], v[66:73], a[0:3], v16, v58 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 252) 
ds_read_b64_tr_b8 v[74:75], v8 offset:12832 // (op 381) Transpose load local data 
ds_read_b64_tr_b8 v[76:77], v8 offset:13344 // (op 381) Transpose load local data 
ds_read_b64_tr_b8 v[78:79], v8 offset:16928 // (op 381) Transpose load local data 
ds_read_b64_tr_b8 v[80:81], v8 offset:17440 // (op 381) Transpose load local data 
buffer_load_dwordx4 v[48:51], v56, s[8:11], 0 offen offset:192 // (op 242) Load value






s_barrier  // (op 2905) 
v_add_u32 v15, 128, v15 // (op 2119) 
ds_read_u8 v52, v9 offset:21024 // (op 385) Load local data 
buffer_load_ubyte v14, v57, s[52:55], 0 offen offset:12 // (op 257) Load value




s_barrier  // (op 2909) 
s_barrier  // (op 2788) 
v_add_u32 v19, 8, v19 // (op 2247) 



ds_read_b128 v[82:85], v7 offset:23296 // (op 630) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[62:65], v[74:81], a[4:7], v16, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 377) 





s_barrier  // (op 2929) 
ds_read_u8 v16, v6 offset:25472 // (op 633) Load local data 
s_waitcnt vmcnt(9)
ds_write_b128 v25, v[26:29] // (op 13) Store local data 
s_waitcnt vmcnt(8)
ds_write_b128 v25, v[30:33] offset:64 // (op 13) Store local data 
s_barrier  // (op 2933) 
s_barrier  // (op 2816) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[82:85], v[66:73], a[8:11], v16, v58 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 674) 





s_waitcnt vmcnt(7)
ds_write_b8 v39, v21 offset:8192 // (op 30) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[82:85], v[74:81], a[12:15], v16, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 628) 




s_barrier  // (op 2775) 



s_waitcnt vmcnt(6)
ds_write_b128 v18, v[34:37] offset:8448 // (op 6) Store local data 
s_barrier  // (op 2802) 



s_waitcnt vmcnt(5)
ds_write_b8 v38, v20 offset:12544 // (op 21) Store local data 
s_ashr_i64 s[6:7], s[28:29], 7 // (op 588) 
s_mov_b32 s1, s6 // (op 588) convert

s_and_b32 s6, s1, 4294967294 // (op 588) 

s_cmp_lt_u32 s0, s6 // (op 588) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_ForLoopTop_KLoop_588 // (op 588) Condition: Bottom (jump to RR_GEMM_TT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_ForLoopTop_KLoop_588 if true)
RR_GEMM_TT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_ForLoopBottom_KLoop_588:

















v_lshrrev_b32 v15, 2, v1 // (op 2079) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 2079) 
v_lshl_add_u32 v16, s16, 6, v15 // (op 2079) 

s_lshr_b32 s6, s0, 1 // (op 2079) 

v_and_b32 v15, -4, v1 // (op 2079) 
v_sub_u32 v17, v1, v15 // (op 2079) 

v_lshlrev_b32 v15, 5, v17 // (op 2079) 

v_lshl_add_u32 v17, s6, 8, v15 // (op 2079) 


v_mov_b32 v19, 0 // (op 2079) convert
v_mov_b32 v18, v17 // (op 2079) convert

v_mul_lo_u32 v21, v16, s13 // (op 2079) most significant: low of low * high
v_mul_hi_u32 v15, v16, s12 // (op 2079) most significant: high of low * low
v_mul_lo_u32 v20, v16, s12 // (op 2079) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 2079) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v18 // (op 2079) least significant half
v_addc_co_u32 v21, s[0:1], v21, v19, s[0:1] // (op 2079) most significant half



v_lshlrev_b64 v[22:23], 2, v[20:21] // (op 2079) 

v_lshrrev_b64 v[20:21], 3, v[22:23] // (op 2079) 

v_mov_b32 v15, v20 // (op 2079) convert

s_mul_i32 s1, s12, s23 // (op 2081) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 2081) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 2081) least significant
s_mul_hi_u32 s7, s12, s22 // (op 2081) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2081) most significant: sum
s_add_u32 s1, s1, s7 // (op 2081) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 2081) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 2081) 

s_mov_b32 s8, s20 // (op 2081) reg expression
s_mov_b32 s9, s21 // (op 2081) reg expression
s_mov_b32 s10, s0 // (op 2081) reg expression
s_mov_b32 s11, 131072 // (op 2081) call()

v_lshrrev_b32 v16, 2, v1 // (op 2149) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 2149) 
v_lshlrev_b32 v17, 1, v16 // (op 2149) 

s_lshr_b32 s6, s0, 1 // (op 2149) 

v_and_b32 v16, -4, v1 // (op 2149) 
v_lshl_add_u32 v18, s6, 8, v17 // (op 2149) 


v_sub_u32 v17, v1, v16 // (op 2149) 

v_lshlrev_b32 v16, 4, v17 // (op 2149) 

v_lshl_add_u32 v17, s17, 6, v16 // (op 2149) 

v_mul_lo_u32 v16, v18, s24 // (op 2149) 
v_add_u32 v16, v16, v17 // (op 2149) 


s_mul_i32 s1, s24, s29 // (op 2151) most significant: low of low * high
s_mul_i32 s6, s25, s28 // (op 2151) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 2151) least significant
s_mul_hi_u32 s7, s24, s28 // (op 2151) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2151) most significant: sum
s_add_u32 s1, s1, s7 // (op 2151) most significant: sum


s_mov_b32 s48, s26 // (op 2151) reg expression
s_mov_b32 s49, s27 // (op 2151) reg expression
s_mov_b32 s50, s0 // (op 2151) reg expression
s_mov_b32 s51, 131072 // (op 2151) call()

v_lshrrev_b32 v17, 2, v1 // (op 2213) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 2213) 
v_lshl_add_u32 v18, s16, 6, v17 // (op 2213) 

s_lshr_b32 s6, s0, 1 // (op 2213) 

v_and_b32 v17, -4, v1 // (op 2213) 
v_sub_u32 v19, v1, v17 // (op 2213) 

v_lshl_add_u32 v17, s6, 3, v19 // (op 2213) 


v_mul_lo_u32 v19, v18, s30 // (op 2213) 
v_add_u32 v19, v19, v17 // (op 2213) 


s_mul_i32 s1, s30, s23 // (op 2215) most significant: low of low * high
s_mul_i32 s6, s31, s22 // (op 2215) most significant: low of high * low
s_mul_i32 s0, s30, s22 // (op 2215) least significant
s_mul_hi_u32 s7, s30, s22 // (op 2215) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2215) most significant: sum
s_add_u32 s1, s1, s7 // (op 2215) most significant: sum


s_mov_b32 s52, s32 // (op 2215) reg expression
s_mov_b32 s53, s33 // (op 2215) reg expression
s_mov_b32 s54, s0 // (op 2215) reg expression
s_mov_b32 s55, 131072 // (op 2215) call()

v_lshrrev_b32 v17, 6, v1 // (op 2277) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 2277) 
v_and_b32 v18, -64, v1 // (op 2277) 
s_lshr_b32 s6, s0, 1 // (op 2277) 

v_sub_u32 v20, v1, v18 // (op 2277) 

v_lshl_add_u32 v18, s6, 3, v17 // (op 2277) 


v_lshl_add_u32 v17, s17, 6, v20 // (op 2277) 

v_mul_lo_u32 v20, v18, s34 // (op 2277) 
v_add_u32 v20, v20, v17 // (op 2277) 


s_mul_i32 s1, s34, s39 // (op 2279) most significant: low of low * high
s_mul_i32 s6, s35, s38 // (op 2279) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 2279) least significant
s_mul_hi_u32 s7, s34, s38 // (op 2279) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2279) most significant: sum
s_add_u32 s1, s1, s7 // (op 2279) most significant: sum


s_mov_b32 s56, s36 // (op 2279) reg expression
s_mov_b32 s57, s37 // (op 2279) reg expression
s_mov_b32 s58, s0 // (op 2279) reg expression
s_mov_b32 s59, 131072 // (op 2279) call()

v_and_b32 v17, -4, v1 // (op 2375) 
v_sub_u32 v18, v1, v17 // (op 2375) 

v_lshlrev_b32 v17, 5, v18 // (op 2375) 

v_lshrrev_b32 v18, 2, v1 // (op 2375) 
v_lshl_add_u32 v21, v18, 7, v17 // (op 2375) 


v_lshlrev_b32 v17, 2, v21 // (op 2375) 

v_lshrrev_b32 v18, 3, v17 // (op 2375) 

v_and_b32 v17, -4, v1 // (op 2433) 
v_sub_u32 v21, v1, v17 // (op 2433) 

v_lshlrev_b32 v17, 4, v21 // (op 2433) 

v_lshrrev_b32 v21, 2, v1 // (op 2433) 
v_lshl_add_u32 v22, v21, 7, v17 // (op 2433) 


v_and_b32 v17, -4, v1 // (op 2491) 
v_sub_u32 v21, v1, v17 // (op 2491) 

v_and_b32 v17, 4294967292, v1 // (op 2491) 
v_add_u32 v23, v17, v21 // (op 2491) 


v_and_b32 v17, -64, v1 // (op 2549) 
v_sub_u32 v21, v1, v17 // (op 2549) 

v_and_b32 v17, 4294967232, v1 // (op 2549) 
v_add_u32 v25, v17, v21 // (op 2549) 


s_ashr_i64 s[0:1], s[28:29], 7 // (op 594) 
s_and_b32 s6, s0, 4294967294 // (op 594) 

s_ashr_i64 s[0:1], s[28:29], 7 // (op 593) 
s_mov_b32 s7, s0 // (op 593) convert

s_cmp_lt_u32 s6, s7 // (op 593) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_ForLoopBottom_KLoopTail_593 // (op 593) Condition: Top (jump to RR_GEMM_TT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_ForLoopBottom_KLoopTail_593 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_ForLoopTop_KLoopTail_593:


s_add_u32 s6, s6, 1 // (op 595) 
v_mov_b32 v14, v15 // (op 176) 

buffer_load_dwordx4 v[26:29], v14, s[8:11], 0 offen // (op 176) Load value

v_add_u32 v15, 64, v15 // (op 2069) 

s_barrier  // (op 2778) 
v_mov_b64 v[30:31], s[24:25] // (op 190) convert




v_mov_b32 v14, v16 // (op 190) 
v_mov_b32 v17, v14 // (op 190) reg expression
buffer_load_dwordx4 v[32:35], v17, s[48:51], 0 offen // (op 190) Load value
v_add_u32 v17, v17, v30 // (op 190) 
buffer_load_dwordx4 v[36:39], v17, s[48:51], 0 offen // (op 190) Load value



s_lshl_b32 s7, s24, 7 // (op 2141) 
v_add_u32 v16, v16, s7 // (op 2141) 

s_waitcnt vmcnt(2)
ds_write_b128 v18, v[26:29] offset:8448 // (op 177) Store local data 



v_mov_b32 v14, v19 // (op 191) 

s_barrier  // (op 2791) 
buffer_load_ubyte v17, v14, s[52:55], 0 offen offset: 0 // (op 191) Load value




v_add_u32 v19, 4, v19 // (op 2205) 
s_waitcnt vmcnt(2)
ds_write_b128 v22, v[32:35] // (op 187) Store local data 


v_mov_b32 v14, v20 // (op 192) 
v_mov_b32 v21, v14 // (op 192) reg expression

buffer_load_ubyte v24, v21, s[56:59], 0 offen offset: 0 // (op 192) Load value


s_lshl_b32 s7, s34, 2 // (op 2269) 
v_add_u32 v20, v20, s7 // (op 2269) 

s_waitcnt vmcnt(2)
ds_write_b128 v22, v[36:39] offset:64 // (op 187) Store local data 

s_waitcnt lgkmcnt(2)
s_barrier  // (op 2781) 



s_barrier  // (op 2805) 
ds_read_b128 v[26:29], v3 offset:8448 // (op 179) Load local data 



s_waitcnt lgkmcnt(1)
s_barrier  // (op 2795) 
s_waitcnt vmcnt(1)
ds_write_b8 v23, v17 offset:12544 // (op 188) Store local data 




ds_read_b64_tr_b8 v[30:31], v4 // (op 181) Transpose load local data 
s_barrier  // (op 2819) 
ds_read_b64_tr_b8 v[32:33], v4 offset:512 // (op 181) Transpose load local data 



ds_read_b64_tr_b8 v[34:35], v4 offset:4096 // (op 181) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v4 offset:4608 // (op 181) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v25, v24 offset:8192 // (op 189) Store local data 

s_waitcnt lgkmcnt(5)
s_barrier  // (op 2809) 



ds_read_u8 v14, v0 offset:12544 // (op 183) Load local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 2823) 


ds_read_u8 v17, v5 offset:8192 // (op 185) Load local data 



ds_read_b64_tr_b8 v[38:39], v4 offset:32 // (op 350) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:29], v[30:37], a[0:3], v14, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 186) 
ds_read_b64_tr_b8 v[40:41], v4 offset:544 // (op 350) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v4 offset:4128 // (op 350) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v4 offset:4640 // (op 350) Transpose load local data 


ds_read_u8 v21, v5 offset:8224 // (op 354) Load local data 



ds_read_b128 v[46:49], v3 offset:10496 // (op 568) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[26:29], v[38:45], a[4:7], v14, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 355) 





ds_read_u8 v14, v0 offset:12672 // (op 571) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[46:49], v[30:37], a[8:11], v14, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 573) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[46:49], v[38:45], a[12:15], v14, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 604) 




s_ashr_i64 s[0:1], s[28:29], 7 // (op 593) 
s_mov_b32 s7, s0 // (op 593) convert

s_cmp_lt_u32 s6, s7 // (op 593) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_ForLoopTop_KLoopTail_593 // (op 593) Condition: Bottom (jump to RR_GEMM_TT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_ForLoopTop_KLoopTail_593 if true)
RR_GEMM_TT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_ForLoopBottom_KLoopTail_593:
























s_barrier  // (op 2917) 








v_cmp_eq_f32 vcc, 0.00000, v10 // (op 133) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_ConditionalFalse_0_beta_133 // (op 133) Condition: False, jump to RR_GEMM_TT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_ConditionalFalse_0_beta_133
v_accvgpr_read v6, a0 // (op 50) 
v_accvgpr_read v7, a1 // (op 50) 
v_accvgpr_read v8, a2 // (op 50) 
v_accvgpr_read v9, a3 // (op 50) 
v_mul_f32 v14, v11, v6 // (op 50) 
v_mul_f32 v15, v11, v7 // (op 50) 
v_mul_f32 v16, v11, v8 // (op 50) 
v_mul_f32 v17, v11, v9 // (op 50) 



v_mov_b32 v0, v13 // (op 56) reg expression

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 56) Store value


v_accvgpr_read v6, a4 // (op 1017) 
v_accvgpr_read v7, a5 // (op 1017) 
v_accvgpr_read v8, a6 // (op 1017) 
v_accvgpr_read v9, a7 // (op 1017) 
v_mul_f32 v14, v11, v6 // (op 1017) 
v_mul_f32 v15, v11, v7 // (op 1017) 
v_mul_f32 v16, v11, v8 // (op 1017) 
v_mul_f32 v17, v11, v9 // (op 1017) 



s_lshl_b32 s0, s4, 7 // (op 1019) 
v_add_u32 v0, v13, s0 // (op 1019) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 1019) Store value


v_accvgpr_read v6, a8 // (op 1041) 
v_accvgpr_read v7, a9 // (op 1041) 
v_accvgpr_read v8, a10 // (op 1041) 
v_accvgpr_read v9, a11 // (op 1041) 
v_mul_f32 v14, v11, v6 // (op 1041) 
v_mul_f32 v15, v11, v7 // (op 1041) 
v_mul_f32 v16, v11, v8 // (op 1041) 
v_mul_f32 v17, v11, v9 // (op 1041) 



v_add_u32 v0, 128, v13 // (op 1044) 

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 1044) Store value


v_accvgpr_read v6, a12 // (op 1050) 
v_accvgpr_read v7, a13 // (op 1050) 
v_accvgpr_read v8, a14 // (op 1050) 
v_accvgpr_read v9, a15 // (op 1050) 
v_mul_f32 v14, v11, v6 // (op 1050) 
v_mul_f32 v15, v11, v7 // (op 1050) 
v_mul_f32 v16, v11, v8 // (op 1050) 
v_mul_f32 v17, v11, v9 // (op 1050) 



s_lshl_b32 s0, s4, 7 // (op 1046) 
s_add_u32 s0, s0, 128 // (op 1046) 
v_add_u32 v0, v13, s0 // (op 1046) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 1046) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_ConditionalBottom_0_beta_133 // (op 133) Condition: Done, jump to RR_GEMM_TT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_ConditionalBottom_0_beta_133
RR_GEMM_TT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_ConditionalFalse_0_beta_133:


v_add_u32 v0, 128, v12 // (op 1130) 

buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 1130) Load value

v_accvgpr_read v14, a8 // (op 1122) 
v_accvgpr_read v15, a9 // (op 1122) 
v_accvgpr_read v16, a10 // (op 1122) 
v_accvgpr_read v17, a11 // (op 1122) 
v_mul_f32 v18, v11, v14 // (op 1122) 
v_mul_f32 v19, v11, v15 // (op 1122) 
v_mul_f32 v20, v11, v16 // (op 1122) 
v_mul_f32 v21, v11, v17 // (op 1122) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 1129) 
v_mul_f32 v15, v10, v7 // (op 1129) 
v_mul_f32 v16, v10, v8 // (op 1129) 
v_mul_f32 v17, v10, v9 // (op 1129) 

v_add_f32 v6, v14, v18 // (op 1123) 
v_add_f32 v7, v15, v19 // (op 1123) 
v_add_f32 v8, v16, v20 // (op 1123) 
v_add_f32 v9, v17, v21 // (op 1123) 




v_add_u32 v0, 128, v13 // (op 1125) 

buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 1125) Store value




s_lshl_b32 s0, s2, 7 // (op 1134) 
s_add_u32 s0, s0, 128 // (op 1134) 
v_add_u32 v0, v12, s0 // (op 1134) 


buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 1134) Load value

v_accvgpr_read v14, a12 // (op 1136) 
v_accvgpr_read v15, a13 // (op 1136) 
v_accvgpr_read v16, a14 // (op 1136) 
v_accvgpr_read v17, a15 // (op 1136) 
v_mul_f32 v18, v11, v14 // (op 1136) 
v_mul_f32 v19, v11, v15 // (op 1136) 
v_mul_f32 v20, v11, v16 // (op 1136) 
v_mul_f32 v21, v11, v17 // (op 1136) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 1132) 
v_mul_f32 v15, v10, v7 // (op 1132) 
v_mul_f32 v16, v10, v8 // (op 1132) 
v_mul_f32 v17, v10, v9 // (op 1132) 

v_add_f32 v6, v14, v18 // (op 1133) 
v_add_f32 v7, v15, v19 // (op 1133) 
v_add_f32 v8, v16, v20 // (op 1133) 
v_add_f32 v9, v17, v21 // (op 1133) 




s_lshl_b32 s0, s4, 7 // (op 1127) 
s_add_u32 s0, s0, 128 // (op 1127) 
v_add_u32 v0, v13, s0 // (op 1127) 


buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 1127) Store value




v_mov_b32 v0, v12 // (op 145) reg expression

s_nop 0
buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 145) Load value

v_accvgpr_read v14, a0 // (op 142) 
v_accvgpr_read v15, a1 // (op 142) 
v_accvgpr_read v16, a2 // (op 142) 
v_accvgpr_read v17, a3 // (op 142) 
v_mul_f32 v18, v11, v14 // (op 142) 
v_mul_f32 v19, v11, v15 // (op 142) 
v_mul_f32 v20, v11, v16 // (op 142) 
v_mul_f32 v21, v11, v17 // (op 142) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 146) 
v_mul_f32 v15, v10, v7 // (op 146) 
v_mul_f32 v16, v10, v8 // (op 146) 
v_mul_f32 v17, v10, v9 // (op 146) 

v_add_f32 v6, v14, v18 // (op 143) 
v_add_f32 v7, v15, v19 // (op 143) 
v_add_f32 v8, v16, v20 // (op 143) 
v_add_f32 v9, v17, v21 // (op 143) 




v_mov_b32 v0, v13 // (op 144) reg expression

buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 144) Store value




s_lshl_b32 s0, s2, 7 // (op 1092) 
v_add_u32 v0, v12, s0 // (op 1092) 


buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 1092) Load value

v_accvgpr_read v14, a4 // (op 1089) 
v_accvgpr_read v15, a5 // (op 1089) 
v_accvgpr_read v16, a6 // (op 1089) 
v_accvgpr_read v17, a7 // (op 1089) 
v_mul_f32 v18, v11, v14 // (op 1089) 
v_mul_f32 v19, v11, v15 // (op 1089) 
v_mul_f32 v20, v11, v16 // (op 1089) 
v_mul_f32 v21, v11, v17 // (op 1089) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 1093) 
v_mul_f32 v15, v10, v7 // (op 1093) 
v_mul_f32 v16, v10, v8 // (op 1093) 
v_mul_f32 v17, v10, v9 // (op 1093) 

v_add_f32 v6, v14, v18 // (op 1090) 
v_add_f32 v7, v15, v19 // (op 1090) 
v_add_f32 v8, v16, v20 // (op 1090) 
v_add_f32 v9, v17, v21 // (op 1090) 




s_lshl_b32 s0, s4, 7 // (op 1091) 
v_add_u32 v0, v13, s0 // (op 1091) 


buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 1091) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_ConditionalBottom_0_beta_133:












s_endpgm  // End of RR_GEMM_TT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2
.LRR_GEMM_TT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_end:
.size RR_GEMM_TT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2, .LRR_GEMM_TT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_end-RR_GEMM_TT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2
  .amdhsa_next_free_vgpr 104
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 25600 // lds bytes
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
  MXScaleFormat: NoSwizzle
  ProblemType:
    OperationType: GEMM
    DataType: F4
    DataTypeA: F4
    DataTypeB: B8
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
    macrotile: [64, 64, 128]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [2, 2]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2
    .symbol: RR_GEMM_TT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2.kd
    .kernarg_segment_size: 136
    .group_segment_fixed_size: 25600
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 60
    .vgpr_count: 86
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

