// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC
.p2align 8
.type RR_GEMM_NN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC,@function
RR_GEMM_NN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC_exec_begin:


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


v_and_b32 v0, 63, v1 // (op 2424) 
v_and_b32 v3, 48, v1 // (op 2424) 
v_sub_u32 v4, v0, v3 // (op 2424) 

v_lshrrev_b32 v3, 4, v0 // (op 2424) 

v_lshl_add_u32 v0, v3, 5, v4 // (op 2424) 


v_lshrrev_b32 v3, 7, v1 // (op 2424) 
v_lshlrev_b32 v4, 4, v3 // (op 2424) 

v_lshl_add_u32 v3, v0, 6, v4 // (op 2424) 


v_mul_lo_u32 v0, v3, 6 // (op 2424) 
v_lshrrev_b32 v4, 3, v0 // (op 2424) 

v_lshrrev_b32 v0, 4, v3 // (op 2424) 

v_lshlrev_b32 v3, 2, v0 // (op 2424) 

v_add_u32 v0, v4, v3 // (op 2424) 


v_lshrrev_b32 v3, 6, v1 // (op 2482) 
v_and_b32 v4, 1, v3 // (op 2482) 

v_and_b32 v3, 15, v1 // (op 2482) 
v_lshl_add_u32 v5, v4, 4, v3 // (op 2482) 


v_and_b32 v3, 63, v1 // (op 2482) 
v_lshrrev_b32 v4, 4, v3 // (op 2482) 

v_lshlrev_b32 v3, 5, v4 // (op 2482) 

v_lshl_add_u32 v4, v5, 7, v3 // (op 2482) 


v_mul_lo_u32 v3, v4, 6 // (op 2482) 

v_lshrrev_b32 v4, 3, v3 // (op 2482) 

v_lshrrev_b32 v3, 7, v1 // (op 2540) 
v_and_b32 v5, 15, v1 // (op 2540) 
v_lshl_add_u32 v6, v3, 4, v5 // (op 2540) 


v_and_b32 v3, 63, v1 // (op 2540) 
v_lshrrev_b32 v5, 4, v3 // (op 2540) 

v_lshl_add_u32 v3, v5, 6, v6 // (op 2540) 


v_lshrrev_b32 v5, 6, v1 // (op 2598) 
v_and_b32 v6, 1, v5 // (op 2598) 

v_and_b32 v5, 15, v1 // (op 2598) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 2598) 


v_and_b32 v5, 63, v1 // (op 2598) 
v_lshrrev_b32 v6, 4, v5 // (op 2598) 

v_lshl_add_u32 v5, v7, 2, v6 // (op 2598) 


v_and_b32 v6, 63, v1 // (op 2650) 
v_and_b32 v7, 48, v1 // (op 2650) 
v_sub_u32 v8, v6, v7 // (op 2650) 

v_lshrrev_b32 v7, 4, v6 // (op 2650) 

v_lshl_add_u32 v6, v7, 5, v8 // (op 2650) 


v_lshrrev_b32 v7, 7, v1 // (op 2650) 
v_lshlrev_b32 v8, 4, v7 // (op 2650) 

v_lshl_add_u32 v7, v6, 6, v8 // (op 2650) 


v_mul_lo_u32 v6, v7, 6 // (op 2650) 
v_lshrrev_b32 v8, 3, v6 // (op 2650) 

v_lshrrev_b32 v6, 4, v7 // (op 2650) 

v_lshlrev_b32 v7, 2, v6 // (op 2650) 

v_add_u32 v6, v8, v7 // (op 2650) 


v_lshrrev_b32 v7, 6, v1 // (op 2679) 
v_and_b32 v8, 1, v7 // (op 2679) 

v_and_b32 v7, 15, v1 // (op 2679) 
v_lshl_add_u32 v9, v8, 4, v7 // (op 2679) 


v_and_b32 v7, 63, v1 // (op 2679) 
v_lshrrev_b32 v8, 4, v7 // (op 2679) 

v_lshlrev_b32 v7, 5, v8 // (op 2679) 

v_lshl_add_u32 v8, v9, 7, v7 // (op 2679) 


v_mul_lo_u32 v7, v8, 6 // (op 2679) 

v_lshrrev_b32 v8, 3, v7 // (op 2679) 

v_lshrrev_b32 v7, 7, v1 // (op 2708) 
v_and_b32 v9, 15, v1 // (op 2708) 
v_lshl_add_u32 v10, v7, 4, v9 // (op 2708) 


v_and_b32 v7, 63, v1 // (op 2708) 
v_lshrrev_b32 v9, 4, v7 // (op 2708) 

v_lshl_add_u32 v7, v9, 6, v10 // (op 2708) 


v_lshrrev_b32 v9, 6, v1 // (op 2737) 
v_and_b32 v10, 1, v9 // (op 2737) 

v_and_b32 v9, 15, v1 // (op 2737) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 2737) 


v_and_b32 v9, 63, v1 // (op 2737) 
v_lshrrev_b32 v10, 4, v9 // (op 2737) 

v_lshl_add_u32 v9, v11, 2, v10 // (op 2737) 


v_lshrrev_b32 v10, 7, v1 // (op 2362) 
v_and_b32 v11, 63, v1 // (op 2362) 
v_lshrrev_b32 v12, 4, v11 // (op 2362) 
v_lshlrev_b32 v13, 2, v12 // (op 2362) 

v_lshl_add_u32 v12, v10, 4, v13 // (op 2362) 


v_lshl_add_u32 v10, s16, 6, v12 // (op 2362) 

v_lshrrev_b32 v12, 2, v11 // (op 2362) 

v_and_b32 v11, 3, v12 // (op 2362) 

v_and_b32 v12, 3, v1 // (op 2362) 
v_lshl_add_u32 v13, v11, 2, v12 // (op 2362) 


v_lshrrev_b32 v11, 6, v1 // (op 2362) 
v_and_b32 v12, 1, v11 // (op 2362) 

v_lshl_add_u32 v11, v12, 4, v13 // (op 2362) 


v_lshl_add_u32 v12, s17, 6, v11 // (op 2362) 

v_mul_lo_u32 v11, v12, s2 // (op 2362) 

v_add_lshl_u32 v12, v10, v11, 1 // (op 2362) 


v_lshrrev_b32 v10, 7, v1 // (op 2619) 
v_and_b32 v11, 63, v1 // (op 2619) 
v_lshrrev_b32 v13, 4, v11 // (op 2619) 
s_mul_i32 s1, s2, s9 // (op 2364) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 2364) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 2364) least significant
s_mul_hi_u32 s19, s2, s8 // (op 2364) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2364) most significant: sum
s_add_u32 s1, s1, s19 // (op 2364) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 2364) 

v_lshlrev_b32 v14, 2, v13 // (op 2619) 

v_lshl_add_u32 v13, v10, 4, v14 // (op 2619) 


v_lshl_add_u32 v10, s16, 6, v13 // (op 2619) 

v_lshrrev_b32 v13, 2, v11 // (op 2619) 

v_and_b32 v11, 3, v13 // (op 2619) 

v_and_b32 v13, 3, v1 // (op 2619) 
v_lshl_add_u32 v14, v11, 2, v13 // (op 2619) 


v_lshrrev_b32 v11, 6, v1 // (op 2619) 
v_and_b32 v13, 1, v11 // (op 2619) 

v_lshl_add_u32 v11, v13, 4, v14 // (op 2619) 


v_lshl_add_u32 v13, s17, 6, v11 // (op 2619) 

v_mul_lo_u32 v11, v13, s4 // (op 2619) 

v_add_lshl_u32 v13, v10, v11, 1 // (op 2619) 


s_mov_b32 s40, s6 // (op 2364) reg expression
s_mov_b32 s41, s7 // (op 2364) reg expression
s_mov_b32 s42, s18 // (op 2364) reg expression
s_mov_b32 s43, 131072 // (op 2364) call()

s_mul_i32 s1, s4, s9 // (op 2621) most significant: low of low * high

s_mul_i32 s6, s5, s8 // (op 2621) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 2621) least significant
s_mul_hi_u32 s7, s4, s8 // (op 2621) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2621) most significant: sum
s_add_u32 s1, s1, s7 // (op 2621) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 1 // (op 2621) 

s_mov_b32 s44, s10 // (op 2621) reg expression
s_mov_b32 s45, s11 // (op 2621) reg expression
s_mov_b32 s46, s6 // (op 2621) reg expression
s_mov_b32 s47, 131072 // (op 2621) call()


v_mov_b32 v10, s14 // (op 2) Move value
v_mov_b32 v11, s15 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 65) call()
v_accvgpr_write a1, 0.00000 // (op 65) call()
v_accvgpr_write a2, 0.00000 // (op 65) call()
v_accvgpr_write a3, 0.00000 // (op 65) call()
v_accvgpr_write a4, 0.00000 // (op 340) call()
v_accvgpr_write a5, 0.00000 // (op 340) call()
v_accvgpr_write a6, 0.00000 // (op 340) call()
v_accvgpr_write a7, 0.00000 // (op 340) call()
v_accvgpr_write a8, 0.00000 // (op 560) call()
v_accvgpr_write a9, 0.00000 // (op 560) call()
v_accvgpr_write a10, 0.00000 // (op 560) call()
v_accvgpr_write a11, 0.00000 // (op 560) call()
v_accvgpr_write a12, 0.00000 // (op 715) call()
v_accvgpr_write a13, 0.00000 // (op 715) call()
v_accvgpr_write a14, 0.00000 // (op 715) call()
v_accvgpr_write a15, 0.00000 // (op 715) call()
v_and_b32 v14, -2, v1 // (op 2124) 
v_sub_u32 v15, v1, v14 // (op 2124) 

v_lshlrev_b32 v14, 5, v15 // (op 2124) 

v_lshl_add_u32 v15, s16, 6, v14 // (op 2124) 

v_mov_b32 v17, 0 // (op 2124) convert
v_mov_b32 v16, v15 // (op 2124) convert

v_lshrrev_b32 v14, 1, v1 // (op 2124) 
v_mul_lo_u32 v19, v14, s13 // (op 2124) most significant: low of low * high
v_mul_hi_u32 v15, v14, s12 // (op 2124) most significant: high of low * low
v_mul_lo_u32 v18, v14, s12 // (op 2124) least significant: low of low * low
v_add_u32 v19, v19, v15 // (op 2124) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v18 // (op 2124) least significant half
v_addc_co_u32 v15, s[0:1], v17, v19, s[0:1] // (op 2124) most significant half



v_mul_lo_u32 v17, v15, 6 // (op 2124) most significant: low of high * low
v_mul_hi_u32 v18, v14, 6 // (op 2124) most significant: high of low * low
v_mul_lo_u32 v16, v14, 6 // (op 2124) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 2124) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 2124) 

v_mov_b32 v16, v14 // (op 2124) convert


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s23, 6 // (op 2126) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 2126) least significant
s_mul_hi_u32 s6, s22, 6 // (op 2126) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2126) most significant: sum

s_mul_i32 s7, s12, s1 // (op 2126) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 2126) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 2126) least significant
s_mul_hi_u32 s11, s12, s0 // (op 2126) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 2126) most significant: sum
s_add_u32 s7, s7, s11 // (op 2126) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 2126) 

s_mov_b32 s48, s20 // (op 2126) reg expression
s_mov_b32 s49, s21 // (op 2126) reg expression
s_mov_b32 s50, s0 // (op 2126) reg expression
s_mov_b32 s51, 131072 // (op 2126) call()

v_and_b32 v14, -2, v1 // (op 2143) 
v_sub_u32 v15, v1, v14 // (op 2143) 

v_lshlrev_b32 v14, 5, v15 // (op 2143) 

v_lshl_add_u32 v15, s16, 6, v14 // (op 2143) 

v_mov_b32 v19, 0 // (op 2143) convert
v_mov_b32 v18, v15 // (op 2143) convert

v_lshrrev_b32 v17, 1, v1 // (op 2143) 
v_mul_lo_u32 v15, v17, s13 // (op 2143) most significant: low of low * high
v_mul_hi_u32 v20, v17, s12 // (op 2143) most significant: high of low * low
v_mul_lo_u32 v14, v17, s12 // (op 2143) least significant: low of low * low
v_add_u32 v15, v15, v20 // (op 2143) most significant: sum


v_add_co_u32 v20, s[0:1], v18, v14 // (op 2143) least significant half
v_addc_co_u32 v21, s[0:1], v19, v15, s[0:1] // (op 2143) most significant half



v_mul_lo_u32 v15, v21, 6 // (op 2143) most significant: low of high * low
v_mul_hi_u32 v17, v20, 6 // (op 2143) most significant: high of low * low
v_mul_lo_u32 v14, v20, 6 // (op 2143) least significant: low of low * low
v_add_u32 v15, v15, v17 // (op 2143) most significant: sum


v_lshrrev_b64 v[18:19], 3, v[14:15] // (op 2143) 

v_mov_b32 v17, v18 // (op 2143) convert

s_mul_i32 s1, s23, 6 // (op 2145) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 2145) least significant
s_mul_hi_u32 s6, s22, 6 // (op 2145) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2145) most significant: sum

s_mul_i32 s7, s12, s1 // (op 2145) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 2145) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 2145) least significant
s_mul_hi_u32 s11, s12, s0 // (op 2145) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 2145) most significant: sum
s_add_u32 s7, s7, s11 // (op 2145) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 2145) 

s_mov_b32 s48, s20 // (op 2145) reg expression
s_mov_b32 s49, s21 // (op 2145) reg expression
s_mov_b32 s50, s0 // (op 2145) reg expression
s_mov_b32 s51, 131072 // (op 2145) call()

v_and_b32 v14, -4, v1 // (op 2188) 
v_sub_u32 v15, v1, v14 // (op 2188) 

v_lshlrev_b32 v14, 5, v15 // (op 2188) 

v_mov_b32 v19, 0 // (op 2188) convert
v_mov_b32 v18, v14 // (op 2188) convert

v_lshrrev_b32 v14, 2, v1 // (op 2188) 
v_lshl_add_u32 v15, s17, 6, v14 // (op 2188) 

v_mul_lo_u32 v21, v15, s25 // (op 2188) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 2188) most significant: high of low * low
v_mul_lo_u32 v20, v15, s24 // (op 2188) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 2188) most significant: sum


v_add_co_u32 v14, s[0:1], v18, v20 // (op 2188) least significant half
v_addc_co_u32 v15, s[0:1], v19, v21, s[0:1] // (op 2188) most significant half



v_mul_lo_u32 v19, v15, 6 // (op 2188) most significant: low of high * low
v_mul_hi_u32 v20, v14, 6 // (op 2188) most significant: high of low * low
v_mul_lo_u32 v18, v14, 6 // (op 2188) least significant: low of low * low
v_add_u32 v19, v19, v20 // (op 2188) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[18:19] // (op 2188) 

v_mov_b32 v18, v14 // (op 2188) convert

s_mul_i32 s1, s9, 6 // (op 2190) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 2190) least significant
s_mul_hi_u32 s6, s8, 6 // (op 2190) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2190) most significant: sum

s_mul_i32 s7, s24, s1 // (op 2190) most significant: low of low * high
s_mul_i32 s10, s25, s0 // (op 2190) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 2190) least significant
s_mul_hi_u32 s11, s24, s0 // (op 2190) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 2190) most significant: sum
s_add_u32 s7, s7, s11 // (op 2190) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 2190) 

s_mov_b32 s52, s26 // (op 2190) reg expression
s_mov_b32 s53, s27 // (op 2190) reg expression
s_mov_b32 s54, s0 // (op 2190) reg expression
s_mov_b32 s55, 131072 // (op 2190) call()

v_and_b32 v14, -4, v1 // (op 2207) 
v_sub_u32 v15, v1, v14 // (op 2207) 

v_lshlrev_b32 v14, 5, v15 // (op 2207) 

v_mov_b32 v21, 0 // (op 2207) convert
v_mov_b32 v20, v14 // (op 2207) convert

v_lshrrev_b32 v19, 2, v1 // (op 2207) 
v_lshl_add_u32 v14, s17, 6, v19 // (op 2207) 

v_mul_lo_u32 v23, v14, s25 // (op 2207) most significant: low of low * high
v_mul_hi_u32 v15, v14, s24 // (op 2207) most significant: high of low * low
v_mul_lo_u32 v22, v14, s24 // (op 2207) least significant: low of low * low
v_add_u32 v23, v23, v15 // (op 2207) most significant: sum


v_add_co_u32 v14, s[0:1], v20, v22 // (op 2207) least significant half
v_addc_co_u32 v15, s[0:1], v21, v23, s[0:1] // (op 2207) most significant half



v_mul_lo_u32 v21, v15, 6 // (op 2207) most significant: low of high * low
v_mul_hi_u32 v19, v14, 6 // (op 2207) most significant: high of low * low
v_mul_lo_u32 v20, v14, 6 // (op 2207) least significant: low of low * low
v_add_u32 v21, v21, v19 // (op 2207) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[20:21] // (op 2207) 

v_mov_b32 v19, v14 // (op 2207) convert

s_mul_i32 s1, s9, 6 // (op 2209) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 2209) least significant
s_mul_hi_u32 s6, s8, 6 // (op 2209) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2209) most significant: sum

s_mul_i32 s7, s24, s1 // (op 2209) most significant: low of low * high
s_mul_i32 s10, s25, s0 // (op 2209) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 2209) least significant
s_mul_hi_u32 s11, s24, s0 // (op 2209) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 2209) most significant: sum
s_add_u32 s7, s7, s11 // (op 2209) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 2209) 

s_mov_b32 s52, s26 // (op 2209) reg expression
s_mov_b32 s53, s27 // (op 2209) reg expression
s_mov_b32 s54, s0 // (op 2209) reg expression
s_mov_b32 s55, 131072 // (op 2209) call()

v_and_b32 v14, -64, v1 // (op 2252) 
v_sub_u32 v15, v1, v14 // (op 2252) 

v_lshl_add_u32 v14, s16, 6, v15 // (op 2252) 

v_lshrrev_b32 v15, 6, v1 // (op 2252) 
v_mul_lo_u32 v20, v15, s28 // (op 2252) 

v_add_u32 v15, v14, v20 // (op 2252) 


s_mul_i32 s1, s28, s33 // (op 2254) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 2254) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 2254) least significant
s_mul_hi_u32 s7, s28, s32 // (op 2254) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2254) most significant: sum
s_add_u32 s1, s1, s7 // (op 2254) most significant: sum


s_mov_b32 s56, s30 // (op 2254) reg expression
s_mov_b32 s57, s31 // (op 2254) reg expression
s_mov_b32 s58, s0 // (op 2254) reg expression
s_mov_b32 s59, 131072 // (op 2254) call()

v_and_b32 v14, -64, v1 // (op 2271) 
v_sub_u32 v20, v1, v14 // (op 2271) 

v_lshl_add_u32 v14, s16, 6, v20 // (op 2271) 

v_lshrrev_b32 v20, 6, v1 // (op 2271) 
v_mul_lo_u32 v21, v20, s28 // (op 2271) 

v_add_u32 v20, v14, v21 // (op 2271) 


s_mul_i32 s1, s28, s33 // (op 2273) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 2273) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 2273) least significant
s_mul_hi_u32 s7, s28, s32 // (op 2273) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2273) most significant: sum
s_add_u32 s1, s1, s7 // (op 2273) most significant: sum


s_mov_b32 s56, s30 // (op 2273) reg expression
s_mov_b32 s57, s31 // (op 2273) reg expression
s_mov_b32 s58, s0 // (op 2273) reg expression
s_mov_b32 s59, 131072 // (op 2273) call()

v_and_b32 v14, -4, v1 // (op 2316) 
v_sub_u32 v21, v1, v14 // (op 2316) 

v_lshrrev_b32 v14, 2, v1 // (op 2316) 
v_lshl_add_u32 v22, s17, 6, v14 // (op 2316) 

v_mul_lo_u32 v14, v22, s34 // (op 2316) 

v_add_u32 v22, v21, v14 // (op 2316) 


s_mul_i32 s1, s34, s9 // (op 2318) most significant: low of low * high
s_mul_i32 s6, s35, s8 // (op 2318) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 2318) least significant
s_mul_hi_u32 s7, s34, s8 // (op 2318) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2318) most significant: sum
s_add_u32 s1, s1, s7 // (op 2318) most significant: sum


s_mov_b32 s60, s36 // (op 2318) reg expression
s_mov_b32 s61, s37 // (op 2318) reg expression
s_mov_b32 s62, s0 // (op 2318) reg expression
s_mov_b32 s63, 131072 // (op 2318) call()

v_and_b32 v14, -4, v1 // (op 2335) 
v_sub_u32 v21, v1, v14 // (op 2335) 

v_lshrrev_b32 v14, 2, v1 // (op 2335) 
v_lshl_add_u32 v23, s17, 6, v14 // (op 2335) 

v_mul_lo_u32 v14, v23, s34 // (op 2335) 

v_add_u32 v23, v21, v14 // (op 2335) 


s_mul_i32 s1, s34, s9 // (op 2337) most significant: low of low * high
s_mul_i32 s6, s35, s8 // (op 2337) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 2337) least significant
s_mul_hi_u32 s7, s34, s8 // (op 2337) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2337) most significant: sum
s_add_u32 s1, s1, s7 // (op 2337) most significant: sum


s_mov_b32 s60, s36 // (op 2337) reg expression
s_mov_b32 s61, s37 // (op 2337) reg expression
s_mov_b32 s62, s0 // (op 2337) reg expression
s_mov_b32 s63, 131072 // (op 2337) call()




buffer_load_dwordx4 v[24:27], v18, s[52:55], 0 offen // (op 1956) Load value
buffer_load_dwordx2 v[28:29], v18, s[52:55], 0 offen offset:16 // (op 1956) Load value



buffer_load_ubyte v14, v22, s[60:63], 0 offen offset: 0 // (op 1959) Load value


v_mov_b32 v21, v16 // (op 1962) reg expression

buffer_load_dwordx3 v[30:32], v21, s[48:51], 0 offen // (op 1962) Load value
buffer_load_dwordx3 v[34:36], v21, s[48:51], 0 offen offset:12 // (op 1962) Load value



v_mov_b32 v21, v15 // (op 1965) reg expression

buffer_load_ubyte v33, v21, s[56:59], 0 offen offset: 0 // (op 1965) Load value




buffer_load_dwordx4 v[38:41], v18, s[52:55], 0 offen offset:96 // (op 1968) Load value
buffer_load_dwordx2 v[42:43], v18, s[52:55], 0 offen offset:112 // (op 1968) Load value




buffer_load_ubyte v18, v22, s[60:63], 0 offen offset:4 // (op 1971) Load value



s_lshl_b64 s[0:1], s[12:13], 7 // (op 1974) 
s_mul_i32 s7, s1, 6 // (op 1974) most significant: low of high * low
s_mul_i32 s6, s0, 6 // (op 1974) least significant
s_mul_hi_u32 s10, s0, 6 // (op 1974) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 1974) most significant: sum


s_lshr_b64 s[0:1], s[6:7], 3 // (op 1974) 

v_add_u32 v37, v16, s0 // (op 1974) 


buffer_load_dwordx3 v[44:46], v37, s[48:51], 0 offen // (op 1974) Load value
buffer_load_dwordx3 v[48:50], v37, s[48:51], 0 offen offset:12 // (op 1974) Load value




s_lshl_b32 s0, s28, 2 // (op 1977) 
v_add_u32 v16, v15, s0 // (op 1977) 


buffer_load_ubyte v37, v16, s[56:59], 0 offen offset: 0 // (op 1977) Load value


v_and_b32 v47, -4, v1 // (op 2465) 
v_sub_u32 v15, v1, v47 // (op 2465) 

v_lshlrev_b32 v16, 5, v15 // (op 2465) 

v_lshrrev_b32 v15, 2, v1 // (op 2465) 
v_lshl_add_u32 v47, v15, 7, v16 // (op 2465) 


v_mul_lo_u32 v15, v47, 6 // (op 2465) 

v_lshrrev_b32 v16, 3, v15 // (op 2465) 




s_waitcnt vmcnt(11)
ds_write_b128 v16, v[24:27] // (op 1979) Store local data 
s_waitcnt vmcnt(10)
ds_write_b64 v16, v[28:29] offset:16 // (op 1979) Store local data 

v_and_b32 v47, -4, v1 // (op 2581) 
v_sub_u32 v15, v1, v47 // (op 2581) 

v_and_b32 v16, 4294967292, v1 // (op 2581) 
v_add_u32 v47, v16, v15 // (op 2581) 





s_waitcnt vmcnt(9)
ds_write_b8 v47, v14 offset:6144 // (op 1980) Store local data 

v_and_b32 v47, -2, v1 // (op 2407) 
v_sub_u32 v15, v1, v47 // (op 2407) 

v_lshlrev_b32 v16, 5, v15 // (op 2407) 

v_lshrrev_b32 v15, 1, v1 // (op 2407) 
v_lshl_add_u32 v47, v15, 6, v16 // (op 2407) 


v_mul_lo_u32 v15, v47, 6 // (op 2407) 
v_lshrrev_b32 v16, 3, v15 // (op 2407) 

v_lshrrev_b32 v15, 4, v47 // (op 2407) 

v_lshlrev_b32 v47, 2, v15 // (op 2407) 

v_add_u32 v15, v16, v47 // (op 2407) 





s_waitcnt vmcnt(8)
ds_write_b96 v15, v[30:32] offset:6400 // (op 1981) Store local data 
s_waitcnt vmcnt(7)
ds_write_b96 v15, v[34:36] offset:6416 // (op 1981) Store local data 

v_and_b32 v47, -64, v1 // (op 2523) 
v_sub_u32 v15, v1, v47 // (op 2523) 

v_and_b32 v16, 4294967232, v1 // (op 2523) 
v_add_u32 v47, v16, v15 // (op 2523) 





s_waitcnt vmcnt(6)
ds_write_b8 v47, v33 offset:14592 // (op 1982) Store local data 

v_and_b32 v47, -2, v1 // (op 2378) 
v_sub_u32 v15, v1, v47 // (op 2378) 

v_lshlrev_b32 v16, 5, v15 // (op 2378) 

v_lshrrev_b32 v15, 1, v1 // (op 2378) 
v_lshl_add_u32 v47, v15, 6, v16 // (op 2378) 


v_mul_lo_u32 v15, v47, 6 // (op 2378) 
v_lshrrev_b32 v16, 3, v15 // (op 2378) 

v_lshrrev_b32 v15, 4, v47 // (op 2378) 

v_lshlrev_b32 v47, 2, v15 // (op 2378) 

v_add_u32 v15, v16, v47 // (op 2378) 


v_and_b32 v16, -4, v1 // (op 2436) 
v_sub_u32 v47, v1, v16 // (op 2436) 

v_lshlrev_b32 v16, 5, v47 // (op 2436) 

v_lshrrev_b32 v47, 2, v1 // (op 2436) 
v_lshl_add_u32 v21, v47, 7, v16 // (op 2436) 


v_mul_lo_u32 v16, v21, 6 // (op 2436) 

v_lshrrev_b32 v47, 3, v16 // (op 2436) 

v_and_b32 v16, -64, v1 // (op 2494) 
v_sub_u32 v21, v1, v16 // (op 2494) 

v_and_b32 v16, 4294967232, v1 // (op 2494) 
v_add_u32 v22, v16, v21 // (op 2494) 


v_and_b32 v16, -4, v1 // (op 2552) 
v_sub_u32 v21, v1, v16 // (op 2552) 

v_and_b32 v16, 4294967292, v1 // (op 2552) 
v_add_u32 v51, v16, v21 // (op 2552) 


v_and_b32 v16, -2, v1 // (op 2633) 
v_sub_u32 v21, v1, v16 // (op 2633) 

v_lshlrev_b32 v16, 5, v21 // (op 2633) 

v_lshrrev_b32 v21, 1, v1 // (op 2633) 
v_lshl_add_u32 v52, v21, 6, v16 // (op 2633) 


v_mul_lo_u32 v16, v52, 6 // (op 2633) 
v_lshrrev_b32 v21, 3, v16 // (op 2633) 

v_lshrrev_b32 v16, 4, v52 // (op 2633) 

v_lshlrev_b32 v52, 2, v16 // (op 2633) 

v_add_u32 v16, v21, v52 // (op 2633) 


v_and_b32 v21, -4, v1 // (op 2662) 
v_sub_u32 v52, v1, v21 // (op 2662) 

v_lshlrev_b32 v21, 5, v52 // (op 2662) 

v_lshrrev_b32 v52, 2, v1 // (op 2662) 
v_lshl_add_u32 v53, v52, 7, v21 // (op 2662) 


v_mul_lo_u32 v21, v53, 6 // (op 2662) 

v_lshrrev_b32 v52, 3, v21 // (op 2662) 

v_and_b32 v21, -64, v1 // (op 2691) 
v_sub_u32 v53, v1, v21 // (op 2691) 

v_and_b32 v21, 4294967232, v1 // (op 2691) 
v_add_u32 v54, v21, v53 // (op 2691) 


v_and_b32 v21, -4, v1 // (op 2720) 
v_sub_u32 v53, v1, v21 // (op 2720) 

v_and_b32 v21, 4294967292, v1 // (op 2720) 
v_add_u32 v55, v21, v53 // (op 2720) 


s_mov_b32 s0, 0 // (op 593) call()
s_ashr_i64 s[6:7], s[22:23], 7 // (op 592) 
s_mov_b32 s1, s6 // (op 592) convert

s_and_b32 s6, s1, 4294967294 // (op 592) 

s_cmp_lt_u32 s0, s6 // (op 592) 

s_waitcnt lgkmcnt(0)// Sync LDS queue before KLoop entry
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC_ForLoopBottom_KLoop_592 // (op 592) Condition: Top (jump to RR_GEMM_NN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC_ForLoopBottom_KLoop_592 if false)
RR_GEMM_NN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC_ForLoopTop_KLoop_592:


s_barrier  // (op 2788) 
s_add_u32 s0, s0, 2 // (op 594) 



v_mov_b32 v21, v19 // (op 10) 

ds_read_b96_tr_b6 v[56:58], v0 offset:6400 // (op 72) Transpose load local data 
ds_read_b96_tr_b6 v[59:61], v0 offset:7424 // (op 72) Transpose load local data 
buffer_load_dwordx4 v[24:27], v21, s[52:55], 0 offen offset:192 // (op 10) Load value
s_barrier  // (op 2801) 
buffer_load_dwordx2 v[28:29], v21, s[52:55], 0 offen offset:208 // (op 10) Load value





v_mov_b32 v53, v23 // (op 25) 

buffer_load_ubyte v14, v53, s[60:63], 0 offen offset:8 // (op 25) Load value
ds_read_b128 v[62:65], v4 // (op 77) Load local data 


ds_read_b64 v[66:67], v4 offset:16 // (op 77) Load local data 
v_mov_b32 v68, v17 // (op 4) 
s_barrier  // (op 2815) 
s_lshl_b64 s[6:7], s[12:13], 7 // (op 4) 


ds_read_u8 v69, v3 offset:14592 // (op 82) Load local data 
s_mul_i32 s11, s7, 6 // (op 4) most significant: low of high * low
s_barrier  // (op 2829) 
s_mul_i32 s10, s6, 6 // (op 4) least significant



s_mul_hi_u32 s1, s6, 6 // (op 4) most significant: high of low * low
s_add_u32 s11, s11, s1 // (op 4) most significant: sum


s_lshr_b64 s[6:7], s[10:11], 3 // (op 4) 

s_lshl_b32 s1, s6, 1 // (op 4) 

v_add_u32 v70, v68, s1 // (op 4) 


ds_read_u8 v71, v5 offset:6144 // (op 87) Load local data 
buffer_load_dwordx3 v[30:32], v70, s[48:51], 0 offen // (op 4) Load value



ds_read_b128 v[72:75], v4 offset:3072 // (op 375) Load local data 
ds_read_b64 v[76:77], v4 offset:3088 // (op 375) Load local data 
buffer_load_dwordx3 v[34:36], v70, s[48:51], 0 offen offset:12 // (op 4) Load value






ds_read_u8 v70, v5 offset:6272 // (op 379) Load local data 
v_mov_b32 v78, v20 // (op 17) 



s_lshl_b32 s1, s28, 3 // (op 17) 
v_add_u32 v79, v78, s1 // (op 17) 


buffer_load_ubyte v33, v79, s[56:59], 0 offen offset: 0 // (op 17) Load value

s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[56:61], v[62:67], a[0:3], v69, v71 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 90) 
s_barrier  // (op 2852) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[56:61], v[72:77], a[4:7], v69, v70 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 380) 





ds_read_b96_tr_b6 v[56:58], v0 offset:6432 // (op 626) Transpose load local data 
ds_read_b96_tr_b6 v[59:61], v0 offset:7456 // (op 626) Transpose load local data 
s_waitcnt vmcnt(11)
ds_write_b128 v52, v[38:41] offset:14848 // (op 257) Store local data 


ds_read_u8 v69, v3 offset:14624 // (op 629) Load local data 
s_waitcnt vmcnt(10)
ds_write_b64 v52, v[42:43] offset:14864 // (op 257) Store local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[56:61], v[62:67], a[8:11], v69, v71 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 677) 


s_barrier  // (op 2864) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[56:61], v[72:77], a[12:15], v69, v70 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 631) 
s_waitcnt vmcnt(9)
ds_write_b8 v55, v18 offset:20992 // (op 259) Store local data 




s_barrier  // (op 2846) 



s_waitcnt vmcnt(8)
ds_write_b96 v16, v[44:46] offset:21248 // (op 247) Store local data 
s_waitcnt vmcnt(7)
ds_write_b96 v16, v[48:50] offset:21264 // (op 247) Store local data 
s_barrier  // (op 2858) 



s_waitcnt vmcnt(6)
ds_write_b8 v54, v37 offset:29440 // (op 258) Store local data 



s_waitcnt lgkmcnt(1)
s_barrier  // (op 2843) 
buffer_load_dwordx4 v[38:41], v21, s[52:55], 0 offen offset:288 // (op 260) Load value



ds_read_b96_tr_b6 v[56:58], v6 offset:21248 // (op 249) Transpose load local data 
ds_read_b96_tr_b6 v[59:61], v6 offset:22272 // (op 249) Transpose load local data 
buffer_load_dwordx2 v[42:43], v21, s[52:55], 0 offen offset:304 // (op 260) Load value

s_barrier  // (op 2849) 






v_add_u32 v19, 192, v19 // (op 2199) 
ds_read_b128 v[62:65], v8 offset:14848 // (op 251) Load local data 
buffer_load_ubyte v18, v53, s[60:63], 0 offen offset:12 // (op 262) Load value
ds_read_b64 v[66:67], v8 offset:14864 // (op 251) Load local data 



v_add_u32 v23, 8, v23 // (op 2327) 
s_lshl_b64 s[6:7], s[12:13], 7 // (op 246) 
s_mul_i32 s11, s7, 6 // (op 246) most significant: low of high * low
s_mul_i32 s10, s6, 6 // (op 246) least significant
s_mul_hi_u32 s1, s6, 6 // (op 246) most significant: high of low * low
s_add_u32 s11, s11, s1 // (op 246) most significant: sum


s_lshr_b64 s[6:7], s[10:11], 3 // (op 246) 

s_mul_i32 s1, 3, s6 // (op 246) 

v_add_u32 v21, v68, s1 // (op 246) 


s_waitcnt lgkmcnt(4)
s_barrier  // (op 2855) 
buffer_load_dwordx3 v[44:46], v21, s[48:51], 0 offen // (op 246) Load value


ds_read_u8 v53, v7 offset:29440 // (op 253) Load local data 
buffer_load_dwordx3 v[48:50], v21, s[48:51], 0 offen offset:12 // (op 246) Load value


s_barrier  // (op 2861) 





ds_read_u8 v21, v9 offset:20992 // (op 255) Load local data 
s_lshl_b32 s1, s28, 2 // (op 261) 



s_mul_i32 s6, 3, s1 // (op 261) 

v_add_u32 v68, v78, s6 // (op 261) 


buffer_load_ubyte v37, v68, s[56:59], 0 offen offset: 0 // (op 261) Load value

s_lshl_b64 s[6:7], s[12:13], 7 // (op 2135) 

s_mul_i32 s11, s7, 6 // (op 2135) most significant: low of high * low
s_barrier  // (op 2804) 
s_mul_i32 s10, s6, 6 // (op 2135) least significant



s_mul_hi_u32 s1, s6, 6 // (op 2135) most significant: high of low * low
s_add_u32 s11, s11, s1 // (op 2135) most significant: sum


s_lshl_b32 s1, s28, 3 // (op 2263) 
v_add_u32 v20, v20, s1 // (op 2263) 

s_lshr_b64 s[6:7], s[10:11], 3 // (op 2135) 

s_lshl_b32 s1, s6, 1 // (op 2135) 

v_add_u32 v17, v17, s1 // (op 2135) 

ds_read_b128 v[68:71], v8 offset:17920 // (op 385) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[56:61], v[62:67], a[0:3], v53, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 256) 
ds_read_b64 v[72:73], v8 offset:17936 // (op 385) Load local data 
s_waitcnt vmcnt(11)
ds_write_b128 v47, v[24:27] // (op 13) Store local data 
s_waitcnt vmcnt(10)
ds_write_b64 v47, v[28:29] offset:16 // (op 13) Store local data 



s_barrier  // (op 2925) 
s_barrier  // (op 2832) 
ds_read_u8 v74, v9 offset:21120 // (op 389) Load local data 



s_waitcnt vmcnt(9)
ds_write_b8 v51, v14 offset:6144 // (op 30) Store local data 



s_barrier  // (op 2929) 
s_barrier  // (op 2791) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[56:61], v[68:73], a[4:7], v53, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 381) 





ds_read_b96_tr_b6 v[56:58], v6 offset:21280 // (op 634) Transpose load local data 
ds_read_b96_tr_b6 v[59:61], v6 offset:22304 // (op 634) Transpose load local data 
s_waitcnt vmcnt(8)
ds_write_b96 v15, v[30:32] offset:6400 // (op 6) Store local data 
s_waitcnt vmcnt(7)
ds_write_b96 v15, v[34:36] offset:6416 // (op 6) Store local data 


s_barrier  // (op 2949) 
s_barrier  // (op 2818) 
ds_read_u8 v53, v7 offset:29472 // (op 637) Load local data 



s_waitcnt vmcnt(6)
ds_write_b8 v22, v33 offset:14592 // (op 21) Store local data 
s_barrier  // (op 2953) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[56:61], v[62:67], a[8:11], v53, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 678) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[56:61], v[68:73], a[12:15], v53, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 632) 




s_ashr_i64 s[6:7], s[22:23], 7 // (op 592) 
s_mov_b32 s1, s6 // (op 592) convert

s_and_b32 s6, s1, 4294967294 // (op 592) 

s_cmp_lt_u32 s0, s6 // (op 592) 

s_waitcnt lgkmcnt(0)// Sync LDS queue before KLoop back-edge
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC_ForLoopTop_KLoop_592 // (op 592) Condition: Bottom (jump to RR_GEMM_NN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC_ForLoopTop_KLoop_592 if true)
RR_GEMM_NN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC_ForLoopBottom_KLoop_592:

















v_and_b32 v47, -2, v1 // (op 2095) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 2095) 
v_sub_u32 v14, v1, v47 // (op 2095) 

s_lshr_b32 s6, s0, 1 // (op 2095) 

v_lshlrev_b32 v47, 5, v14 // (op 2095) 

v_lshl_add_u32 v14, s16, 6, v47 // (op 2095) 

v_mov_b32 v17, 0 // (op 2095) convert
v_mov_b32 v16, v14 // (op 2095) convert

v_lshrrev_b32 v47, 1, v1 // (op 2095) 
v_lshl_add_u32 v14, s6, 8, v47 // (op 2095) 


v_mul_lo_u32 v21, v14, s13 // (op 2095) most significant: low of low * high
v_mul_hi_u32 v15, v14, s12 // (op 2095) most significant: high of low * low
v_mul_lo_u32 v20, v14, s12 // (op 2095) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 2095) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v20 // (op 2095) least significant half
v_addc_co_u32 v15, s[0:1], v17, v21, s[0:1] // (op 2095) most significant half



v_mul_lo_u32 v17, v15, 6 // (op 2095) most significant: low of high * low
v_mul_hi_u32 v47, v14, 6 // (op 2095) most significant: high of low * low
v_mul_lo_u32 v16, v14, 6 // (op 2095) least significant: low of low * low
v_add_u32 v17, v17, v47 // (op 2095) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 2095) 

v_mov_b32 v47, v14 // (op 2095) convert

s_mul_i32 s1, s23, 6 // (op 2097) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 2097) least significant
s_mul_hi_u32 s6, s22, 6 // (op 2097) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2097) most significant: sum

s_mul_i32 s7, s12, s1 // (op 2097) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 2097) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 2097) least significant
s_mul_hi_u32 s11, s12, s0 // (op 2097) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 2097) most significant: sum
s_add_u32 s7, s7, s11 // (op 2097) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 2097) 

s_mov_b32 s48, s20 // (op 2097) reg expression
s_mov_b32 s49, s21 // (op 2097) reg expression
s_mov_b32 s50, s0 // (op 2097) reg expression
s_mov_b32 s51, 131072 // (op 2097) call()

v_and_b32 v14, -4, v1 // (op 2165) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 2165) 
v_sub_u32 v15, v1, v14 // (op 2165) 

s_lshr_b32 s6, s0, 1 // (op 2165) 

v_lshlrev_b32 v14, 5, v15 // (op 2165) 

v_lshl_add_u32 v15, s6, 8, v14 // (op 2165) 


v_mov_b32 v17, 0 // (op 2165) convert
v_mov_b32 v16, v15 // (op 2165) convert

v_lshrrev_b32 v14, 2, v1 // (op 2165) 
v_lshl_add_u32 v15, s17, 6, v14 // (op 2165) 

v_mul_lo_u32 v21, v15, s25 // (op 2165) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 2165) most significant: high of low * low
v_mul_lo_u32 v20, v15, s24 // (op 2165) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 2165) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v20 // (op 2165) least significant half
v_addc_co_u32 v15, s[0:1], v17, v21, s[0:1] // (op 2165) most significant half



v_mul_lo_u32 v17, v15, 6 // (op 2165) most significant: low of high * low
v_mul_hi_u32 v19, v14, 6 // (op 2165) most significant: high of low * low
v_mul_lo_u32 v16, v14, 6 // (op 2165) least significant: low of low * low
v_add_u32 v17, v17, v19 // (op 2165) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 2165) 

v_mov_b32 v16, v14 // (op 2165) convert

s_mul_i32 s1, s9, 6 // (op 2167) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 2167) least significant
s_mul_hi_u32 s6, s8, 6 // (op 2167) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2167) most significant: sum

s_mul_i32 s7, s24, s1 // (op 2167) most significant: low of low * high
s_mul_i32 s10, s25, s0 // (op 2167) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 2167) least significant
s_mul_hi_u32 s11, s24, s0 // (op 2167) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 2167) most significant: sum
s_add_u32 s7, s7, s11 // (op 2167) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 2167) 

s_mov_b32 s52, s26 // (op 2167) reg expression
s_mov_b32 s53, s27 // (op 2167) reg expression
s_mov_b32 s54, s0 // (op 2167) reg expression
s_mov_b32 s55, 131072 // (op 2167) call()

v_and_b32 v17, -64, v1 // (op 2229) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 2229) 
v_sub_u32 v14, v1, v17 // (op 2229) 

s_lshr_b32 s6, s0, 1 // (op 2229) 

v_lshl_add_u32 v15, s16, 6, v14 // (op 2229) 

v_lshrrev_b32 v14, 6, v1 // (op 2229) 
v_lshl_add_u32 v17, s6, 3, v14 // (op 2229) 


v_mul_lo_u32 v14, v17, s28 // (op 2229) 

v_add_u32 v17, v15, v14 // (op 2229) 


s_mul_i32 s1, s28, s33 // (op 2231) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 2231) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 2231) least significant
s_mul_hi_u32 s7, s28, s32 // (op 2231) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2231) most significant: sum
s_add_u32 s1, s1, s7 // (op 2231) most significant: sum


s_mov_b32 s56, s30 // (op 2231) reg expression
s_mov_b32 s57, s31 // (op 2231) reg expression
s_mov_b32 s58, s0 // (op 2231) reg expression
s_mov_b32 s59, 131072 // (op 2231) call()

v_and_b32 v14, -4, v1 // (op 2293) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 2293) 
v_sub_u32 v15, v1, v14 // (op 2293) 

s_lshr_b32 s6, s0, 1 // (op 2293) 

v_lshrrev_b32 v14, 2, v1 // (op 2293) 
v_lshl_add_u32 v19, s6, 3, v15 // (op 2293) 


v_lshl_add_u32 v15, s17, 6, v14 // (op 2293) 

v_mul_lo_u32 v14, v15, s34 // (op 2293) 

v_add_u32 v15, v19, v14 // (op 2293) 


s_mul_i32 s1, s34, s9 // (op 2295) most significant: low of low * high
s_mul_i32 s6, s35, s8 // (op 2295) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 2295) least significant
s_mul_hi_u32 s7, s34, s8 // (op 2295) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2295) most significant: sum
s_add_u32 s1, s1, s7 // (op 2295) most significant: sum


s_mov_b32 s60, s36 // (op 2295) reg expression
s_mov_b32 s61, s37 // (op 2295) reg expression
s_mov_b32 s62, s0 // (op 2295) reg expression
s_mov_b32 s63, 131072 // (op 2295) call()

v_and_b32 v14, -2, v1 // (op 2391) 
v_sub_u32 v19, v1, v14 // (op 2391) 

v_lshlrev_b32 v14, 5, v19 // (op 2391) 

v_lshrrev_b32 v19, 1, v1 // (op 2391) 
v_lshl_add_u32 v20, v19, 6, v14 // (op 2391) 


v_mul_lo_u32 v14, v20, 6 // (op 2391) 
v_lshrrev_b32 v19, 3, v14 // (op 2391) 

v_lshrrev_b32 v14, 4, v20 // (op 2391) 

v_lshlrev_b32 v20, 2, v14 // (op 2391) 

v_add_u32 v14, v19, v20 // (op 2391) 


v_and_b32 v19, -4, v1 // (op 2449) 
v_sub_u32 v20, v1, v19 // (op 2449) 

v_lshlrev_b32 v19, 5, v20 // (op 2449) 

v_lshrrev_b32 v20, 2, v1 // (op 2449) 
v_lshl_add_u32 v21, v20, 7, v19 // (op 2449) 


v_mul_lo_u32 v19, v21, 6 // (op 2449) 

v_lshrrev_b32 v20, 3, v19 // (op 2449) 

v_and_b32 v19, -64, v1 // (op 2507) 
v_sub_u32 v21, v1, v19 // (op 2507) 

v_and_b32 v19, 4294967232, v1 // (op 2507) 
v_add_u32 v22, v19, v21 // (op 2507) 


v_and_b32 v19, -4, v1 // (op 2565) 
v_sub_u32 v21, v1, v19 // (op 2565) 

v_and_b32 v19, 4294967292, v1 // (op 2565) 
v_add_u32 v23, v19, v21 // (op 2565) 


s_ashr_i64 s[0:1], s[22:23], 7 // (op 598) 
s_and_b32 s6, s0, 4294967294 // (op 598) 

s_ashr_i64 s[0:1], s[22:23], 7 // (op 597) 
s_mov_b32 s7, s0 // (op 597) convert

s_cmp_lt_u32 s6, s7 // (op 597) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC_ForLoopBottom_KLoopTail_597 // (op 597) Condition: Top (jump to RR_GEMM_NN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC_ForLoopBottom_KLoopTail_597 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC_ForLoopTop_KLoopTail_597:


s_add_u32 s6, s6, 1 // (op 599) 
v_mov_b32 v21, v47 // (op 180) 
v_mov_b32 v18, v21 // (op 180) reg expression

buffer_load_dwordx3 v[24:26], v18, s[48:51], 0 offen // (op 180) Load value
buffer_load_dwordx3 v[28:30], v18, s[48:51], 0 offen offset:12 // (op 180) Load value


s_lshl_b64 s[0:1], s[12:13], 7 // (op 2085) 
s_mul_i32 s11, s1, 6 // (op 2085) most significant: low of high * low
s_mul_i32 s10, s0, 6 // (op 2085) least significant
s_mul_hi_u32 s7, s0, 6 // (op 2085) most significant: high of low * low
s_add_u32 s11, s11, s7 // (op 2085) most significant: sum


s_lshr_b64 s[0:1], s[10:11], 3 // (op 2085) 

v_add_u32 v47, v47, s0 // (op 2085) 



s_barrier  // (op 2794) 
v_mov_b32 v21, v16 // (op 194) 




buffer_load_dwordx4 v[32:35], v21, s[52:55], 0 offen // (op 194) Load value
s_waitcnt vmcnt(2)
ds_write_b96 v14, v[24:26] offset:6400 // (op 181) Store local data 
buffer_load_dwordx2 v[36:37], v21, s[52:55], 0 offen offset:16 // (op 194) Load value

v_add_u32 v16, 96, v16 // (op 2157) 
s_waitcnt vmcnt(2)
ds_write_b96 v14, v[28:30] offset:6416 // (op 181) Store local data 



v_mov_b32 v21, v17 // (op 195) 
s_barrier  // (op 2807) 
v_mov_b32 v18, v21 // (op 195) reg expression




buffer_load_ubyte v19, v18, s[56:59], 0 offen offset: 0 // (op 195) Load value


s_lshl_b32 s7, s28, 2 // (op 2221) 
v_add_u32 v17, v17, s7 // (op 2221) 

s_waitcnt vmcnt(2)
ds_write_b128 v20, v[32:35] // (op 191) Store local data 


v_mov_b32 v18, v15 // (op 196) 

buffer_load_ubyte v21, v18, s[60:63], 0 offen offset: 0 // (op 196) Load value

v_add_u32 v15, 4, v15 // (op 2285) 
s_waitcnt vmcnt(2)
ds_write_b64 v20, v[36:37] offset:16 // (op 191) Store local data 

s_waitcnt lgkmcnt(2)
s_barrier  // (op 2797) 



s_barrier  // (op 2821) 
ds_read_b96_tr_b6 v[24:26], v0 offset:6400 // (op 183) Transpose load local data 



ds_read_b96_tr_b6 v[27:29], v0 offset:7424 // (op 183) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v22, v19 offset:14592 // (op 192) Store local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 2811) 



s_barrier  // (op 2835) 
ds_read_b128 v[30:33], v4 // (op 185) Load local data 



ds_read_b64 v[34:35], v4 offset:16 // (op 185) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v23, v21 offset:6144 // (op 193) Store local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 2825) 


ds_read_u8 v21, v3 offset:14592 // (op 187) Load local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 2839) 



ds_read_u8 v18, v5 offset:6144 // (op 189) Load local data 



ds_read_b128 v[36:39], v4 offset:3072 // (op 354) Load local data 
ds_read_b64 v[40:41], v4 offset:3088 // (op 354) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[24:29], v[30:35], a[0:3], v21, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 190) 



ds_read_u8 v19, v5 offset:6272 // (op 358) Load local data 



ds_read_b96_tr_b6 v[48:50], v0 offset:6432 // (op 572) Transpose load local data 
ds_read_b96_tr_b6 v[51:53], v0 offset:7456 // (op 572) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[24:29], v[36:41], a[4:7], v21, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 359) 




ds_read_u8 v21, v3 offset:14624 // (op 575) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[48:53], v[30:35], a[8:11], v21, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 577) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[48:53], v[36:41], a[12:15], v21, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 608) 




s_ashr_i64 s[0:1], s[22:23], 7 // (op 597) 
s_mov_b32 s7, s0 // (op 597) convert

s_cmp_lt_u32 s6, s7 // (op 597) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC_ForLoopTop_KLoopTail_597 // (op 597) Condition: Bottom (jump to RR_GEMM_NN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC_ForLoopTop_KLoopTail_597 if true)
RR_GEMM_NN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC_ForLoopBottom_KLoopTail_597:























s_barrier  // (op 2937) 









v_cmp_eq_f32 vcc, 0.00000, v10 // (op 135) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC_ConditionalFalse_0_beta_135 // (op 135) Condition: False, jump to RR_GEMM_NN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC_ConditionalFalse_0_beta_135
v_accvgpr_read v6, a0 // (op 50) 
v_accvgpr_read v7, a1 // (op 50) 
v_accvgpr_read v8, a2 // (op 50) 
v_accvgpr_read v9, a3 // (op 50) 
v_mul_f32 v14, v11, v6 // (op 50) 
v_mul_f32 v15, v11, v7 // (op 50) 
v_mul_f32 v16, v11, v8 // (op 50) 
v_mul_f32 v17, v11, v9 // (op 50) 

v_cvt_f16_f32 v6, v14 // (op 56) 
v_cvt_f16_f32 v7, v15 // (op 56) 
v_cvt_f16_f32 v8, v16 // (op 56) 
v_cvt_f16_f32 v9, v17 // (op 56) 


v_pack_b32_f16 v4, v6, v7 // (op 58) 
v_pack_b32_f16 v5, v8, v9 // (op 58) 

v_mov_b32 v0, v13 // (op 58) reg expression

buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 58) Store value



v_accvgpr_read v6, a4 // (op 1021) 
v_accvgpr_read v7, a5 // (op 1021) 
v_accvgpr_read v8, a6 // (op 1021) 
v_accvgpr_read v9, a7 // (op 1021) 
v_mul_f32 v14, v11, v6 // (op 1021) 
v_mul_f32 v15, v11, v7 // (op 1021) 
v_mul_f32 v16, v11, v8 // (op 1021) 
v_mul_f32 v17, v11, v9 // (op 1021) 

v_cvt_f16_f32 v6, v14 // (op 1023) 
v_cvt_f16_f32 v7, v15 // (op 1023) 
v_cvt_f16_f32 v8, v16 // (op 1023) 
v_cvt_f16_f32 v9, v17 // (op 1023) 


v_pack_b32_f16 v4, v6, v7 // (op 1024) 
v_pack_b32_f16 v5, v8, v9 // (op 1024) 

s_lshl_b32 s0, s4, 6 // (op 1024) 
v_add_u32 v0, v13, s0 // (op 1024) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 1024) Store value



v_accvgpr_read v6, a8 // (op 1047) 
v_accvgpr_read v7, a9 // (op 1047) 
v_accvgpr_read v8, a10 // (op 1047) 
v_accvgpr_read v9, a11 // (op 1047) 
v_mul_f32 v14, v11, v6 // (op 1047) 
v_mul_f32 v15, v11, v7 // (op 1047) 
v_mul_f32 v16, v11, v8 // (op 1047) 
v_mul_f32 v17, v11, v9 // (op 1047) 

v_cvt_f16_f32 v6, v14 // (op 1049) 
v_cvt_f16_f32 v7, v15 // (op 1049) 
v_cvt_f16_f32 v8, v16 // (op 1049) 
v_cvt_f16_f32 v9, v17 // (op 1049) 


v_pack_b32_f16 v4, v6, v7 // (op 1051) 
v_pack_b32_f16 v5, v8, v9 // (op 1051) 

v_add_u32 v0, 64, v13 // (op 1051) 

buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 1051) Store value



v_accvgpr_read v6, a12 // (op 1057) 
v_accvgpr_read v7, a13 // (op 1057) 
v_accvgpr_read v8, a14 // (op 1057) 
v_accvgpr_read v9, a15 // (op 1057) 
v_mul_f32 v14, v11, v6 // (op 1057) 
v_mul_f32 v15, v11, v7 // (op 1057) 
v_mul_f32 v16, v11, v8 // (op 1057) 
v_mul_f32 v17, v11, v9 // (op 1057) 

v_cvt_f16_f32 v6, v14 // (op 1059) 
v_cvt_f16_f32 v7, v15 // (op 1059) 
v_cvt_f16_f32 v8, v16 // (op 1059) 
v_cvt_f16_f32 v9, v17 // (op 1059) 


v_pack_b32_f16 v4, v6, v7 // (op 1053) 
v_pack_b32_f16 v5, v8, v9 // (op 1053) 

s_lshl_b32 s0, s4, 6 // (op 1053) 
s_add_u32 s0, s0, 64 // (op 1053) 
v_add_u32 v0, v13, s0 // (op 1053) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 1053) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC_ConditionalBottom_0_beta_135 // (op 135) Condition: Done, jump to RR_GEMM_NN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC_ConditionalBottom_0_beta_135
RR_GEMM_NN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC_ConditionalFalse_0_beta_135:


v_add_u32 v0, 64, v12 // (op 1143) 

buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 1143) Load value

v_accvgpr_read v4, a8 // (op 1134) 
v_accvgpr_read v5, a9 // (op 1134) 
v_accvgpr_read v6, a10 // (op 1134) 
v_accvgpr_read v7, a11 // (op 1134) 
v_mul_f32 v14, v11, v4 // (op 1134) 
v_mul_f32 v15, v11, v5 // (op 1134) 
v_mul_f32 v16, v11, v6 // (op 1134) 
v_mul_f32 v17, v11, v7 // (op 1134) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v8 // (op 1142) 
v_lshrrev_b32 v7, 16, v8 // (op 1142) 
v_cvt_f32_f16 v7, v7 // (op 1142) 
v_mul_f32 v18, v10, v6 // (op 1142) 
v_mul_f32 v19, v10, v7 // (op 1142) 
v_cvt_f32_f16 v6, v9 // (op 1142) 
v_lshrrev_b32 v7, 16, v9 // (op 1142) 
v_cvt_f32_f16 v7, v7 // (op 1142) 
v_mul_f32 v20, v10, v6 // (op 1142) 
v_mul_f32 v21, v10, v7 // (op 1142) 


v_add_f32 v6, v18, v14 // (op 1135) 
v_add_f32 v7, v19, v15 // (op 1135) 
v_add_f32 v8, v20, v16 // (op 1135) 
v_add_f32 v9, v21, v17 // (op 1135) 


v_cvt_f16_f32 v14, v6 // (op 1136) 
v_cvt_f16_f32 v15, v7 // (op 1136) 
v_cvt_f16_f32 v16, v8 // (op 1136) 
v_cvt_f16_f32 v17, v9 // (op 1136) 


v_pack_b32_f16 v8, v14, v15 // (op 1138) 
v_pack_b32_f16 v9, v16, v17 // (op 1138) 

v_add_u32 v0, 64, v13 // (op 1138) 

buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 1138) Store value





s_lshl_b32 s0, s2, 6 // (op 1148) 
s_add_u32 s0, s0, 64 // (op 1148) 
v_add_u32 v0, v12, s0 // (op 1148) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 1148) Load value

v_accvgpr_read v4, a12 // (op 1150) 
v_accvgpr_read v5, a13 // (op 1150) 
v_accvgpr_read v6, a14 // (op 1150) 
v_accvgpr_read v7, a15 // (op 1150) 
v_mul_f32 v14, v11, v4 // (op 1150) 
v_mul_f32 v15, v11, v5 // (op 1150) 
v_mul_f32 v16, v11, v6 // (op 1150) 
v_mul_f32 v17, v11, v7 // (op 1150) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v8 // (op 1145) 
v_lshrrev_b32 v7, 16, v8 // (op 1145) 
v_cvt_f32_f16 v7, v7 // (op 1145) 
v_mul_f32 v18, v10, v6 // (op 1145) 
v_mul_f32 v19, v10, v7 // (op 1145) 
v_cvt_f32_f16 v6, v9 // (op 1145) 
v_lshrrev_b32 v7, 16, v9 // (op 1145) 
v_cvt_f32_f16 v7, v7 // (op 1145) 
v_mul_f32 v20, v10, v6 // (op 1145) 
v_mul_f32 v21, v10, v7 // (op 1145) 


v_add_f32 v6, v18, v14 // (op 1146) 
v_add_f32 v7, v19, v15 // (op 1146) 
v_add_f32 v8, v20, v16 // (op 1146) 
v_add_f32 v9, v21, v17 // (op 1146) 


v_cvt_f16_f32 v14, v6 // (op 1147) 
v_cvt_f16_f32 v15, v7 // (op 1147) 
v_cvt_f16_f32 v16, v8 // (op 1147) 
v_cvt_f16_f32 v17, v9 // (op 1147) 


v_pack_b32_f16 v8, v14, v15 // (op 1140) 
v_pack_b32_f16 v9, v16, v17 // (op 1140) 

s_lshl_b32 s0, s4, 6 // (op 1140) 
s_add_u32 s0, s0, 64 // (op 1140) 
v_add_u32 v0, v13, s0 // (op 1140) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 1140) Store value





v_mov_b32 v0, v12 // (op 148) reg expression

buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 148) Load value

v_accvgpr_read v4, a0 // (op 144) 
v_accvgpr_read v5, a1 // (op 144) 
v_accvgpr_read v6, a2 // (op 144) 
v_accvgpr_read v7, a3 // (op 144) 
v_mul_f32 v14, v11, v4 // (op 144) 
v_mul_f32 v15, v11, v5 // (op 144) 
v_mul_f32 v16, v11, v6 // (op 144) 
v_mul_f32 v17, v11, v7 // (op 144) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v8 // (op 149) 
v_lshrrev_b32 v7, 16, v8 // (op 149) 
v_cvt_f32_f16 v7, v7 // (op 149) 
v_mul_f32 v18, v10, v6 // (op 149) 
v_mul_f32 v19, v10, v7 // (op 149) 
v_cvt_f32_f16 v6, v9 // (op 149) 
v_lshrrev_b32 v7, 16, v9 // (op 149) 
v_cvt_f32_f16 v7, v7 // (op 149) 
v_mul_f32 v20, v10, v6 // (op 149) 
v_mul_f32 v21, v10, v7 // (op 149) 


v_add_f32 v6, v18, v14 // (op 145) 
v_add_f32 v7, v19, v15 // (op 145) 
v_add_f32 v8, v20, v16 // (op 145) 
v_add_f32 v9, v21, v17 // (op 145) 


v_cvt_f16_f32 v14, v6 // (op 146) 
v_cvt_f16_f32 v15, v7 // (op 146) 
v_cvt_f16_f32 v16, v8 // (op 146) 
v_cvt_f16_f32 v17, v9 // (op 146) 


v_pack_b32_f16 v8, v14, v15 // (op 147) 
v_pack_b32_f16 v9, v16, v17 // (op 147) 

v_mov_b32 v0, v13 // (op 147) reg expression

buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 147) Store value





s_lshl_b32 s0, s2, 6 // (op 1103) 
v_add_u32 v0, v12, s0 // (op 1103) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 1103) Load value

v_accvgpr_read v4, a4 // (op 1099) 
v_accvgpr_read v5, a5 // (op 1099) 
v_accvgpr_read v6, a6 // (op 1099) 
v_accvgpr_read v7, a7 // (op 1099) 
v_mul_f32 v14, v11, v4 // (op 1099) 
v_mul_f32 v15, v11, v5 // (op 1099) 
v_mul_f32 v16, v11, v6 // (op 1099) 
v_mul_f32 v17, v11, v7 // (op 1099) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v8 // (op 1104) 
v_lshrrev_b32 v7, 16, v8 // (op 1104) 
v_cvt_f32_f16 v7, v7 // (op 1104) 
v_mul_f32 v18, v10, v6 // (op 1104) 
v_mul_f32 v19, v10, v7 // (op 1104) 
v_cvt_f32_f16 v6, v9 // (op 1104) 
v_lshrrev_b32 v7, 16, v9 // (op 1104) 
v_cvt_f32_f16 v7, v7 // (op 1104) 
v_mul_f32 v20, v10, v6 // (op 1104) 
v_mul_f32 v21, v10, v7 // (op 1104) 


v_add_f32 v6, v18, v14 // (op 1100) 
v_add_f32 v7, v19, v15 // (op 1100) 
v_add_f32 v8, v20, v16 // (op 1100) 
v_add_f32 v9, v21, v17 // (op 1100) 


v_cvt_f16_f32 v14, v6 // (op 1101) 
v_cvt_f16_f32 v15, v7 // (op 1101) 
v_cvt_f16_f32 v16, v8 // (op 1101) 
v_cvt_f16_f32 v17, v9 // (op 1101) 


v_pack_b32_f16 v8, v14, v15 // (op 1102) 
v_pack_b32_f16 v9, v16, v17 // (op 1102) 

s_lshl_b32 s0, s4, 6 // (op 1102) 
v_add_u32 v0, v13, s0 // (op 1102) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 1102) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC_ConditionalBottom_0_beta_135:












s_endpgm  // End of RR_GEMM_NN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC
.LRR_GEMM_NN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC_end:
.size RR_GEMM_NN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC, .LRR_GEMM_NN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC_end-RR_GEMM_NN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC
  .amdhsa_next_free_vgpr 96
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 29696 // lds bytes
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
    DataType: F6
    DataTypeA: F6
    DataTypeB: B6
    DestDataType: h
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
  - .name: RR_GEMM_NN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC
    .symbol: RR_GEMM_NN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 29696
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

