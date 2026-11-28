// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC
.p2align 8
.type RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC,@function
RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC_exec_begin:


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


v_lshlrev_b32 v0, 2, v3 // (op 2424) 

v_lshrrev_b32 v3, 3, v0 // (op 2424) 

v_and_b32 v0, 48, v1 // (op 2482) 
v_and_b32 v4, 63, v1 // (op 2482) 
v_sub_u32 v5, v4, v0 // (op 2482) 

v_lshrrev_b32 v4, 1, v5 // (op 2482) 
v_add_lshl_u32 v6, v0, v4, 6 // (op 2482) 


v_and_b32 v0, 1, v5 // (op 2482) 

v_lshlrev_b32 v4, 3, v0 // (op 2482) 

v_lshrrev_b32 v0, 6, v1 // (op 2482) 
v_and_b32 v5, 1, v0 // (op 2482) 

v_lshl_add_u32 v0, v5, 4, v4 // (op 2482) 


v_add_u32 v4, v6, v0 // (op 2482) 


v_lshrrev_b32 v0, 7, v1 // (op 2540) 
v_and_b32 v5, 15, v1 // (op 2540) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 2540) 


v_and_b32 v0, 63, v1 // (op 2540) 
v_lshrrev_b32 v5, 4, v0 // (op 2540) 

v_lshl_add_u32 v0, v5, 6, v6 // (op 2540) 


v_lshrrev_b32 v5, 6, v1 // (op 2598) 
v_and_b32 v6, 1, v5 // (op 2598) 

v_and_b32 v5, 15, v1 // (op 2598) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 2598) 


v_and_b32 v5, 63, v1 // (op 2598) 
v_lshrrev_b32 v6, 4, v5 // (op 2598) 

v_lshl_add_u32 v5, v6, 6, v7 // (op 2598) 


v_and_b32 v6, 63, v1 // (op 2650) 
v_and_b32 v7, 48, v1 // (op 2650) 
v_sub_u32 v8, v6, v7 // (op 2650) 

v_lshrrev_b32 v7, 4, v6 // (op 2650) 

v_lshl_add_u32 v6, v7, 5, v8 // (op 2650) 


v_lshrrev_b32 v7, 7, v1 // (op 2650) 
v_lshlrev_b32 v8, 4, v7 // (op 2650) 

v_lshl_add_u32 v7, v6, 6, v8 // (op 2650) 


v_lshlrev_b32 v6, 2, v7 // (op 2650) 

v_lshrrev_b32 v7, 3, v6 // (op 2650) 

v_and_b32 v6, 48, v1 // (op 2679) 
v_and_b32 v8, 63, v1 // (op 2679) 
v_sub_u32 v9, v8, v6 // (op 2679) 

v_lshrrev_b32 v8, 1, v9 // (op 2679) 
v_add_lshl_u32 v10, v6, v8, 6 // (op 2679) 


v_and_b32 v6, 1, v9 // (op 2679) 

v_lshlrev_b32 v8, 3, v6 // (op 2679) 

v_lshrrev_b32 v6, 6, v1 // (op 2679) 
v_and_b32 v9, 1, v6 // (op 2679) 

v_lshl_add_u32 v6, v9, 4, v8 // (op 2679) 


v_add_u32 v8, v10, v6 // (op 2679) 


v_lshrrev_b32 v6, 7, v1 // (op 2708) 
v_and_b32 v9, 15, v1 // (op 2708) 
v_lshl_add_u32 v10, v6, 4, v9 // (op 2708) 


v_and_b32 v6, 63, v1 // (op 2708) 
v_lshrrev_b32 v9, 4, v6 // (op 2708) 

v_lshl_add_u32 v6, v9, 6, v10 // (op 2708) 


v_lshrrev_b32 v9, 6, v1 // (op 2737) 
v_and_b32 v10, 1, v9 // (op 2737) 

v_and_b32 v9, 15, v1 // (op 2737) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 2737) 


v_and_b32 v9, 63, v1 // (op 2737) 
v_lshrrev_b32 v10, 4, v9 // (op 2737) 

v_lshl_add_u32 v9, v10, 6, v11 // (op 2737) 


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

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 2124) 


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 2124) 

v_mov_b32 v14, v16 // (op 2124) convert


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 2126) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 2126) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 2126) least significant
s_mul_hi_u32 s7, s12, s22 // (op 2126) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2126) most significant: sum
s_add_u32 s1, s1, s7 // (op 2126) most significant: sum


s_lshl_b64 s[14:15], s[0:1], 2 // (op 2126) 

s_ashr_i64 s[0:1], s[14:15], 3 // (op 2126) 

s_mov_b32 s8, s20 // (op 2126) reg expression
s_mov_b32 s9, s21 // (op 2126) reg expression
s_mov_b32 s10, s0 // (op 2126) reg expression
s_mov_b32 s11, 131072 // (op 2126) call()

v_and_b32 v15, -2, v1 // (op 2143) 
v_sub_u32 v16, v1, v15 // (op 2143) 

v_lshlrev_b32 v15, 5, v16 // (op 2143) 

v_lshl_add_u32 v16, s16, 6, v15 // (op 2143) 

v_mov_b32 v19, 0 // (op 2143) convert
v_mov_b32 v18, v16 // (op 2143) convert

v_lshrrev_b32 v15, 1, v1 // (op 2143) 
v_mul_lo_u32 v17, v15, s13 // (op 2143) most significant: low of low * high
v_mul_hi_u32 v20, v15, s12 // (op 2143) most significant: high of low * low
v_mul_lo_u32 v16, v15, s12 // (op 2143) least significant: low of low * low
v_add_u32 v17, v17, v20 // (op 2143) most significant: sum


v_add_co_u32 v20, s[0:1], v18, v16 // (op 2143) least significant half
v_addc_co_u32 v21, s[0:1], v19, v17, s[0:1] // (op 2143) most significant half

v_lshlrev_b64 v[20:21], 2, v[20:21] // (op 2143) 


v_lshrrev_b64 v[18:19], 3, v[20:21] // (op 2143) 

v_mov_b32 v15, v18 // (op 2143) convert

s_mul_i32 s1, s12, s23 // (op 2145) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 2145) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 2145) least significant
s_mul_hi_u32 s7, s12, s22 // (op 2145) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2145) most significant: sum
s_add_u32 s1, s1, s7 // (op 2145) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 2145) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 2145) 

s_mov_b32 s8, s20 // (op 2145) reg expression
s_mov_b32 s9, s21 // (op 2145) reg expression
s_mov_b32 s10, s0 // (op 2145) reg expression
s_mov_b32 s11, 131072 // (op 2145) call()

v_and_b32 v16, -4, v1 // (op 2188) 
v_sub_u32 v17, v1, v16 // (op 2188) 

v_lshlrev_b32 v16, 4, v17 // (op 2188) 

v_lshl_add_u32 v17, s17, 6, v16 // (op 2188) 

v_lshrrev_b32 v16, 2, v1 // (op 2188) 
v_lshlrev_b32 v18, 1, v16 // (op 2188) 

v_mul_lo_u32 v16, v18, s24 // (op 2188) 
v_add_u32 v16, v16, v17 // (op 2188) 


s_mul_i32 s1, s24, s23 // (op 2190) most significant: low of low * high
s_mul_i32 s6, s25, s22 // (op 2190) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 2190) least significant
s_mul_hi_u32 s7, s24, s22 // (op 2190) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2190) most significant: sum
s_add_u32 s1, s1, s7 // (op 2190) most significant: sum


s_mov_b32 s48, s26 // (op 2190) reg expression
s_mov_b32 s49, s27 // (op 2190) reg expression
s_mov_b32 s50, s0 // (op 2190) reg expression
s_mov_b32 s51, 131072 // (op 2190) call()

v_and_b32 v17, -4, v1 // (op 2207) 
v_sub_u32 v18, v1, v17 // (op 2207) 

v_lshlrev_b32 v17, 4, v18 // (op 2207) 

v_lshl_add_u32 v18, s17, 6, v17 // (op 2207) 

v_lshrrev_b32 v17, 2, v1 // (op 2207) 
v_lshlrev_b32 v19, 1, v17 // (op 2207) 

v_mul_lo_u32 v17, v19, s24 // (op 2207) 
v_add_u32 v17, v17, v18 // (op 2207) 


s_mul_i32 s1, s24, s23 // (op 2209) most significant: low of low * high
s_mul_i32 s6, s25, s22 // (op 2209) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 2209) least significant
s_mul_hi_u32 s7, s24, s22 // (op 2209) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2209) most significant: sum
s_add_u32 s1, s1, s7 // (op 2209) most significant: sum


s_mov_b32 s48, s26 // (op 2209) reg expression
s_mov_b32 s49, s27 // (op 2209) reg expression
s_mov_b32 s50, s0 // (op 2209) reg expression
s_mov_b32 s51, 131072 // (op 2209) call()

v_and_b32 v18, -64, v1 // (op 2252) 
v_sub_u32 v19, v1, v18 // (op 2252) 

v_lshl_add_u32 v18, s16, 6, v19 // (op 2252) 

v_lshrrev_b32 v19, 6, v1 // (op 2252) 
v_mul_lo_u32 v20, v19, s28 // (op 2252) 

v_add_u32 v19, v18, v20 // (op 2252) 


s_mul_i32 s1, s28, s33 // (op 2254) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 2254) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 2254) least significant
s_mul_hi_u32 s7, s28, s32 // (op 2254) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2254) most significant: sum
s_add_u32 s1, s1, s7 // (op 2254) most significant: sum


s_mov_b32 s52, s30 // (op 2254) reg expression
s_mov_b32 s53, s31 // (op 2254) reg expression
s_mov_b32 s54, s0 // (op 2254) reg expression
s_mov_b32 s55, 131072 // (op 2254) call()

v_and_b32 v18, -64, v1 // (op 2271) 
v_sub_u32 v20, v1, v18 // (op 2271) 

v_lshl_add_u32 v18, s16, 6, v20 // (op 2271) 

v_lshrrev_b32 v20, 6, v1 // (op 2271) 
v_mul_lo_u32 v21, v20, s28 // (op 2271) 

v_add_u32 v20, v18, v21 // (op 2271) 


s_mul_i32 s1, s28, s33 // (op 2273) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 2273) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 2273) least significant
s_mul_hi_u32 s7, s28, s32 // (op 2273) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2273) most significant: sum
s_add_u32 s1, s1, s7 // (op 2273) most significant: sum


s_mov_b32 s52, s30 // (op 2273) reg expression
s_mov_b32 s53, s31 // (op 2273) reg expression
s_mov_b32 s54, s0 // (op 2273) reg expression
s_mov_b32 s55, 131072 // (op 2273) call()

v_and_b32 v18, -64, v1 // (op 2316) 
v_sub_u32 v21, v1, v18 // (op 2316) 

v_lshl_add_u32 v18, s17, 6, v21 // (op 2316) 

v_lshrrev_b32 v21, 6, v1 // (op 2316) 
v_mul_lo_u32 v22, v21, s34 // (op 2316) 
v_add_u32 v22, v22, v18 // (op 2316) 


s_mul_i32 s1, s34, s33 // (op 2318) most significant: low of low * high
s_mul_i32 s6, s35, s32 // (op 2318) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 2318) least significant
s_mul_hi_u32 s7, s34, s32 // (op 2318) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2318) most significant: sum
s_add_u32 s1, s1, s7 // (op 2318) most significant: sum


s_mov_b32 s56, s36 // (op 2318) reg expression
s_mov_b32 s57, s37 // (op 2318) reg expression
s_mov_b32 s58, s0 // (op 2318) reg expression
s_mov_b32 s59, 131072 // (op 2318) call()

v_and_b32 v18, -64, v1 // (op 2335) 
v_sub_u32 v21, v1, v18 // (op 2335) 

v_lshl_add_u32 v18, s17, 6, v21 // (op 2335) 

v_lshrrev_b32 v21, 6, v1 // (op 2335) 
v_mul_lo_u32 v23, v21, s34 // (op 2335) 
v_add_u32 v23, v23, v18 // (op 2335) 


s_mul_i32 s1, s34, s33 // (op 2337) most significant: low of low * high
s_mul_i32 s6, s35, s32 // (op 2337) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 2337) least significant
s_mul_hi_u32 s7, s34, s32 // (op 2337) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2337) most significant: sum
s_add_u32 s1, s1, s7 // (op 2337) most significant: sum


s_mov_b32 s56, s36 // (op 2337) reg expression
s_mov_b32 s57, s37 // (op 2337) reg expression
s_mov_b32 s58, s0 // (op 2337) reg expression
s_mov_b32 s59, 131072 // (op 2337) call()


v_mov_b64 v[24:25], s[24:25] // (op 1956) convert

v_mov_b32 v18, v16 // (op 1956) reg expression
buffer_load_dwordx4 v[26:29], v18, s[48:51], 0 offen // (op 1956) Load value
v_add_u32 v18, v18, v24 // (op 1956) 
buffer_load_dwordx4 v[30:33], v18, s[48:51], 0 offen // (op 1956) Load value




v_mov_b32 v18, v22 // (op 1959) reg expression

buffer_load_ubyte v21, v18, s[56:59], 0 offen offset: 0 // (op 1959) Load value



v_mov_b32 v18, v14 // (op 1962) reg expression

buffer_load_dwordx4 v[34:37], v18, s[8:11], 0 offen // (op 1962) Load value



v_mov_b32 v18, v19 // (op 1965) reg expression

buffer_load_ubyte v24, v18, s[52:55], 0 offen offset: 0 // (op 1965) Load value


v_mov_b64 v[38:39], s[24:25] // (op 1968) convert

s_lshl_b32 s0, s24, 7 // (op 1968) 
v_add_u32 v18, v16, s0 // (op 1968) 

buffer_load_dwordx4 v[40:43], v18, s[48:51], 0 offen // (op 1968) Load value
v_add_u32 v18, v18, v38 // (op 1968) 
buffer_load_dwordx4 v[44:47], v18, s[48:51], 0 offen // (op 1968) Load value





s_lshl_b32 s0, s34, 2 // (op 1971) 
v_add_u32 v16, v22, s0 // (op 1971) 


buffer_load_ubyte v18, v16, s[56:59], 0 offen offset: 0 // (op 1971) Load value




s_lshl_b64 s[0:1], s[12:13], 7 // (op 1974) 
s_lshl_b64 s[6:7], s[0:1], 2 // (op 1974) 

s_lshr_b64 s[0:1], s[6:7], 3 // (op 1974) 

v_add_u32 v16, v14, s0 // (op 1974) 


buffer_load_dwordx4 v[48:51], v16, s[8:11], 0 offen // (op 1974) Load value




s_lshl_b32 s0, s28, 2 // (op 1977) 
v_add_u32 v14, v19, s0 // (op 1977) 


buffer_load_ubyte v16, v14, s[52:55], 0 offen offset: 0 // (op 1977) Load value


v_and_b32 v14, -4, v1 // (op 2465) 
v_sub_u32 v19, v1, v14 // (op 2465) 

v_lshlrev_b32 v14, 4, v19 // (op 2465) 

v_lshrrev_b32 v19, 2, v1 // (op 2465) 
v_lshl_add_u32 v22, v19, 7, v14 // (op 2465) 





s_waitcnt vmcnt(9)
ds_write_b128 v22, v[26:29] // (op 1979) Store local data 
s_waitcnt vmcnt(8)
ds_write_b128 v22, v[30:33] offset:64 // (op 1979) Store local data 

v_and_b32 v14, -64, v1 // (op 2581) 
v_sub_u32 v19, v1, v14 // (op 2581) 

v_and_b32 v14, 4294967232, v1 // (op 2581) 
v_add_u32 v22, v14, v19 // (op 2581) 





s_waitcnt vmcnt(7)
ds_write_b8 v22, v21 offset:8192 // (op 1980) Store local data 

v_and_b32 v14, -2, v1 // (op 2407) 
v_sub_u32 v19, v1, v14 // (op 2407) 

v_lshlrev_b32 v14, 5, v19 // (op 2407) 

v_lshrrev_b32 v19, 1, v1 // (op 2407) 
v_lshl_add_u32 v22, v19, 6, v14 // (op 2407) 


v_lshlrev_b32 v14, 2, v22 // (op 2407) 

v_lshrrev_b32 v19, 3, v14 // (op 2407) 




s_waitcnt vmcnt(6)
ds_write_b128 v19, v[34:37] offset:8448 // (op 1981) Store local data 

v_and_b32 v14, -64, v1 // (op 2523) 
v_sub_u32 v19, v1, v14 // (op 2523) 

v_and_b32 v14, 4294967232, v1 // (op 2523) 
v_add_u32 v22, v14, v19 // (op 2523) 





s_waitcnt vmcnt(5)
ds_write_b8 v22, v24 offset:12544 // (op 1982) Store local data 

v_and_b32 v14, -2, v1 // (op 2378) 
v_sub_u32 v19, v1, v14 // (op 2378) 

v_lshlrev_b32 v14, 5, v19 // (op 2378) 

v_lshrrev_b32 v19, 1, v1 // (op 2378) 
v_lshl_add_u32 v22, v19, 6, v14 // (op 2378) 


v_lshlrev_b32 v14, 2, v22 // (op 2378) 

v_lshrrev_b32 v19, 3, v14 // (op 2378) 

v_and_b32 v14, -4, v1 // (op 2436) 
v_sub_u32 v22, v1, v14 // (op 2436) 

v_lshlrev_b32 v14, 4, v22 // (op 2436) 

v_lshrrev_b32 v22, 2, v1 // (op 2436) 
v_lshl_add_u32 v25, v22, 7, v14 // (op 2436) 


v_and_b32 v14, -64, v1 // (op 2494) 
v_sub_u32 v22, v1, v14 // (op 2494) 

v_and_b32 v14, 4294967232, v1 // (op 2494) 
v_add_u32 v38, v14, v22 // (op 2494) 


v_and_b32 v14, -64, v1 // (op 2552) 
v_sub_u32 v22, v1, v14 // (op 2552) 

v_and_b32 v14, 4294967232, v1 // (op 2552) 
v_add_u32 v39, v14, v22 // (op 2552) 


v_and_b32 v14, -2, v1 // (op 2633) 
v_sub_u32 v22, v1, v14 // (op 2633) 

v_lshlrev_b32 v14, 5, v22 // (op 2633) 

v_lshrrev_b32 v22, 1, v1 // (op 2633) 
v_lshl_add_u32 v52, v22, 6, v14 // (op 2633) 


v_lshlrev_b32 v14, 2, v52 // (op 2633) 

v_lshrrev_b32 v22, 3, v14 // (op 2633) 

v_and_b32 v14, -4, v1 // (op 2662) 
v_sub_u32 v52, v1, v14 // (op 2662) 

v_lshlrev_b32 v14, 4, v52 // (op 2662) 

v_lshrrev_b32 v52, 2, v1 // (op 2662) 
v_lshl_add_u32 v53, v52, 7, v14 // (op 2662) 


v_and_b32 v14, -64, v1 // (op 2691) 
v_sub_u32 v52, v1, v14 // (op 2691) 

v_and_b32 v14, 4294967232, v1 // (op 2691) 
v_add_u32 v54, v14, v52 // (op 2691) 


v_and_b32 v14, -64, v1 // (op 2720) 
v_sub_u32 v52, v1, v14 // (op 2720) 

v_and_b32 v14, 4294967232, v1 // (op 2720) 
v_add_u32 v55, v14, v52 // (op 2720) 


s_mov_b32 s0, 0 // (op 593) call()
s_ashr_i64 s[6:7], s[22:23], 7 // (op 592) 
s_mov_b32 s1, s6 // (op 592) convert

s_and_b32 s6, s1, 4294967294 // (op 592) 

s_cmp_lt_u32 s0, s6 // (op 592) 

s_waitcnt lgkmcnt(0)// Sync LDS queue before KLoop entry
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC_ForLoopBottom_KLoop_592 // (op 592) Condition: Top (jump to RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC_ForLoopBottom_KLoop_592 if false)
RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC_ForLoopTop_KLoop_592:

s_barrier  // (op 2788) 
s_add_u32 s0, s0, 2 // (op 594) 



v_mov_b64 v[56:57], s[24:25] // (op 10) convert

v_mov_b32 v14, v17 // (op 10) 
s_lshl_b32 s1, s24, 8 // (op 10) 
v_add_u32 v52, v14, s1 // (op 10) 

buffer_load_dwordx4 v[26:29], v52, s[48:51], 0 offen // (op 10) Load value
v_add_u32 v52, v52, v56 // (op 10) 
buffer_load_dwordx4 v[30:33], v52, s[48:51], 0 offen // (op 10) Load value


ds_read_b64_tr_b4 v[56:57], v3 offset:8448 // (op 72) Transpose load local data 


ds_read_b64_tr_b4 v[58:59], v3 offset:8960 // (op 72) Transpose load local data 
v_mov_b32 v52, v23 // (op 25) 
s_barrier  // (op 2801) 
s_lshl_b32 s1, s34, 3 // (op 25) 



v_add_u32 v60, v52, s1 // (op 25) 


ds_read_b64_tr_b8 v[62:63], v4 // (op 77) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v4 offset:512 // (op 77) Transpose load local data 
ds_read_b64_tr_b8 v[66:67], v4 offset:4096 // (op 77) Transpose load local data 
ds_read_b64_tr_b8 v[68:69], v4 offset:4608 // (op 77) Transpose load local data 
buffer_load_ubyte v21, v60, s[56:59], 0 offen offset: 0 // (op 25) Load value



s_barrier  // (op 2815) 
v_mov_b32 v60, v15 // (op 4) 


ds_read_u8 v61, v0 offset:12544 // (op 82) Load local data 
s_lshl_b64 s[6:7], s[12:13], 7 // (op 4) 
s_barrier  // (op 2829) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 4) 



ds_read_u8 v70, v5 offset:8192 // (op 87) Load local data 
s_lshr_b64 s[6:7], s[14:15], 3 // (op 4) 




s_lshl_b32 s1, s6, 1 // (op 4) 

v_add_u32 v71, v60, s1 // (op 4) 


ds_read_b64_tr_b8 v[72:73], v4 offset:32 // (op 375) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[56:59], v[62:69], a[0:3], v61, v70 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 90) 
ds_read_b64_tr_b8 v[74:75], v4 offset:544 // (op 375) Transpose load local data 
ds_read_b64_tr_b8 v[76:77], v4 offset:4128 // (op 375) Transpose load local data 
ds_read_b64_tr_b8 v[78:79], v4 offset:4640 // (op 375) Transpose load local data 
buffer_load_dwordx4 v[34:37], v71, s[8:11], 0 offen // (op 4) Load value





ds_read_u8 v71, v5 offset:8224 // (op 379) Load local data 
v_mov_b32 v80, v20 // (op 17) 



s_lshl_b32 s1, s28, 3 // (op 17) 
v_add_u32 v81, v80, s1 // (op 17) 


buffer_load_ubyte v24, v81, s[52:55], 0 offen offset: 0 // (op 17) Load value

ds_read_b64_tr_b4 v[82:83], v3 offset:8464 // (op 626) Transpose load local data 
s_barrier  // (op 2852) 
ds_read_b64_tr_b4 v[84:85], v3 offset:8976 // (op 626) Transpose load local data 





s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[56:59], v[72:79], a[4:7], v61, v71 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 380) 
ds_read_u8 v81, v0 offset:12576 // (op 629) Load local data 


s_waitcnt vmcnt(9)
ds_write_b128 v53, v[40:43] offset:12800 // (op 257) Store local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[82:85], v[62:69], a[8:11], v81, v70 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 677) 
s_waitcnt vmcnt(8)
ds_write_b128 v53, v[44:47] offset:12864 // (op 257) Store local data 


s_barrier  // (op 2864) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[82:85], v[72:79], a[12:15], v81, v71 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 631) 







s_waitcnt vmcnt(7)
ds_write_b8 v55, v18 offset:20992 // (op 259) Store local data 
s_barrier  // (op 2846) 



s_waitcnt vmcnt(6)
ds_write_b128 v22, v[48:51] offset:21248 // (op 247) Store local data 
s_barrier  // (op 2858) 



s_waitcnt vmcnt(5)
ds_write_b8 v54, v16 offset:25344 // (op 258) Store local data 

v_mov_b64 v[56:57], s[24:25] // (op 260) convert

s_lshl_b32 s1, s24, 7 // (op 260) 
s_mul_i32 s6, 3, s1 // (op 260) 

v_add_u32 v58, v14, s6 // (op 260) 

s_waitcnt lgkmcnt(1)
s_barrier  // (op 2843) 
buffer_load_dwordx4 v[40:43], v58, s[48:51], 0 offen // (op 260) Load value



v_add_u32 v58, v58, v56 // (op 260) 
ds_read_b64_tr_b4 v[76:77], v7 offset:21248 // (op 249) Transpose load local data 
ds_read_b64_tr_b4 v[78:79], v7 offset:21760 // (op 249) Transpose load local data 
buffer_load_dwordx4 v[44:47], v58, s[48:51], 0 offen // (op 260) Load value



s_barrier  // (op 2849) 





s_lshl_b32 s1, s34, 2 // (op 262) 
s_mul_i32 s6, 3, s1 // (op 262) 

s_lshl_b32 s1, s24, 8 // (op 2199) 
v_add_u32 v17, v17, s1 // (op 2199) 

v_add_u32 v14, v52, s6 // (op 262) 


buffer_load_ubyte v18, v14, s[56:59], 0 offen offset: 0 // (op 262) Load value

ds_read_b64_tr_b8 v[68:69], v8 offset:12800 // (op 251) Transpose load local data 

ds_read_b64_tr_b8 v[70:71], v8 offset:13312 // (op 251) Transpose load local data 


ds_read_b64_tr_b8 v[72:73], v8 offset:16896 // (op 251) Transpose load local data 
s_lshl_b32 s1, s34, 3 // (op 2327) 
v_add_u32 v23, v23, s1 // (op 2327) 

s_lshl_b64 s[6:7], s[12:13], 7 // (op 246) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 246) 

s_lshr_b64 s[6:7], s[14:15], 3 // (op 246) 

s_mul_i32 s1, 3, s6 // (op 246) 

v_add_u32 v14, v60, s1 // (op 246) 


ds_read_b64_tr_b8 v[74:75], v8 offset:17408 // (op 251) Transpose load local data 
buffer_load_dwordx4 v[48:51], v14, s[8:11], 0 offen // (op 246) Load value


s_waitcnt lgkmcnt(6)
s_barrier  // (op 2855) 




ds_read_u8 v14, v6 offset:25344 // (op 253) Load local data 
s_lshl_b32 s1, s28, 2 // (op 261) 
s_barrier  // (op 2861) 
s_mul_i32 s6, 3, s1 // (op 261) 



ds_read_u8 v52, v9 offset:20992 // (op 255) Load local data 
v_add_u32 v56, v80, s6 // (op 261) 





s_lshl_b64 s[6:7], s[12:13], 7 // (op 2135) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 2135) 

s_lshr_b64 s[6:7], s[14:15], 3 // (op 2135) 

s_lshl_b32 s1, s6, 1 // (op 2135) 

v_add_u32 v15, v15, s1 // (op 2135) 

ds_read_b64_tr_b8 v[60:61], v8 offset:12832 // (op 385) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[76:79], v[68:75], a[0:3], v14, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 256) 
ds_read_b64_tr_b8 v[62:63], v8 offset:13344 // (op 385) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v8 offset:16928 // (op 385) Transpose load local data 
ds_read_b64_tr_b8 v[66:67], v8 offset:17440 // (op 385) Transpose load local data 
buffer_load_ubyte v16, v56, s[52:55], 0 offen offset: 0 // (op 261) Load value




s_barrier  // (op 2925) 
s_barrier  // (op 2804) 
ds_read_u8 v56, v9 offset:21024 // (op 389) Load local data 



s_lshl_b32 s1, s28, 3 // (op 2263) 



s_barrier  // (op 2929) 
v_add_u32 v20, v20, s1 // (op 2263) 

ds_read_b64_tr_b4 v[80:81], v7 offset:21264 // (op 634) Transpose load local data 
ds_read_b64_tr_b4 v[82:83], v7 offset:21776 // (op 634) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[76:79], v[60:67], a[4:7], v14, v56 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 381) 




s_barrier  // (op 2949) 
ds_read_u8 v14, v6 offset:25376 // (op 637) Load local data 
s_waitcnt vmcnt(9)
ds_write_b128 v25, v[26:29] // (op 13) Store local data 
s_waitcnt vmcnt(8)
ds_write_b128 v25, v[30:33] offset:64 // (op 13) Store local data 
s_barrier  // (op 2953) 
s_barrier  // (op 2832) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[80:83], v[68:75], a[8:11], v14, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 678) 





s_waitcnt vmcnt(7)
ds_write_b8 v39, v21 offset:8192 // (op 30) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[80:83], v[60:67], a[12:15], v14, v56 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 632) 




s_barrier  // (op 2791) 



s_waitcnt vmcnt(6)
ds_write_b128 v19, v[34:37] offset:8448 // (op 6) Store local data 
s_barrier  // (op 2818) 



s_waitcnt vmcnt(5)
ds_write_b8 v38, v24 offset:12544 // (op 21) Store local data 
s_ashr_i64 s[6:7], s[22:23], 7 // (op 592) 
s_mov_b32 s1, s6 // (op 592) convert

s_and_b32 s6, s1, 4294967294 // (op 592) 

s_cmp_lt_u32 s0, s6 // (op 592) 

s_waitcnt lgkmcnt(0)// Sync LDS queue before KLoop back-edge
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC_ForLoopTop_KLoop_592 // (op 592) Condition: Bottom (jump to RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC_ForLoopTop_KLoop_592 if true)
RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC_ForLoopBottom_KLoop_592:

















v_and_b32 v17, -2, v1 // (op 2095) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 2095) 
v_sub_u32 v14, v1, v17 // (op 2095) 

s_lshr_b32 s6, s0, 1 // (op 2095) 

v_lshlrev_b32 v15, 5, v14 // (op 2095) 

v_lshl_add_u32 v14, s16, 6, v15 // (op 2095) 

v_mov_b32 v39, 0 // (op 2095) convert
v_mov_b32 v38, v14 // (op 2095) convert

v_lshrrev_b32 v17, 1, v1 // (op 2095) 
v_lshl_add_u32 v14, s6, 8, v17 // (op 2095) 


v_mul_lo_u32 v37, v14, s13 // (op 2095) most significant: low of low * high
v_mul_hi_u32 v15, v14, s12 // (op 2095) most significant: high of low * low
v_mul_lo_u32 v36, v14, s12 // (op 2095) least significant: low of low * low
v_add_u32 v37, v37, v15 // (op 2095) most significant: sum


v_add_co_u32 v14, s[0:1], v38, v36 // (op 2095) least significant half
v_addc_co_u32 v15, s[0:1], v39, v37, s[0:1] // (op 2095) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 2095) 


v_lshrrev_b64 v[38:39], 3, v[14:15] // (op 2095) 

v_mov_b32 v17, v38 // (op 2095) convert

s_mul_i32 s1, s12, s23 // (op 2097) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 2097) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 2097) least significant
s_mul_hi_u32 s7, s12, s22 // (op 2097) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2097) most significant: sum
s_add_u32 s1, s1, s7 // (op 2097) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 2097) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 2097) 

s_mov_b32 s8, s20 // (op 2097) reg expression
s_mov_b32 s9, s21 // (op 2097) reg expression
s_mov_b32 s10, s0 // (op 2097) reg expression
s_mov_b32 s11, 131072 // (op 2097) call()

v_lshrrev_b32 v14, 2, v1 // (op 2165) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 2165) 
v_lshlrev_b32 v15, 1, v14 // (op 2165) 

s_lshr_b32 s6, s0, 1 // (op 2165) 

v_and_b32 v14, -4, v1 // (op 2165) 
v_lshl_add_u32 v19, s6, 8, v15 // (op 2165) 


v_sub_u32 v15, v1, v14 // (op 2165) 

v_lshlrev_b32 v14, 4, v15 // (op 2165) 

v_lshl_add_u32 v15, s17, 6, v14 // (op 2165) 

v_mul_lo_u32 v14, v19, s24 // (op 2165) 
v_add_u32 v14, v14, v15 // (op 2165) 


s_mul_i32 s1, s24, s23 // (op 2167) most significant: low of low * high
s_mul_i32 s6, s25, s22 // (op 2167) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 2167) least significant
s_mul_hi_u32 s7, s24, s22 // (op 2167) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2167) most significant: sum
s_add_u32 s1, s1, s7 // (op 2167) most significant: sum


s_mov_b32 s48, s26 // (op 2167) reg expression
s_mov_b32 s49, s27 // (op 2167) reg expression
s_mov_b32 s50, s0 // (op 2167) reg expression
s_mov_b32 s51, 131072 // (op 2167) call()

v_and_b32 v15, -64, v1 // (op 2229) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 2229) 
v_sub_u32 v19, v1, v15 // (op 2229) 

s_lshr_b32 s6, s0, 1 // (op 2229) 

v_lshl_add_u32 v15, s16, 6, v19 // (op 2229) 

v_lshrrev_b32 v19, 6, v1 // (op 2229) 
v_lshl_add_u32 v20, s6, 3, v19 // (op 2229) 


v_mul_lo_u32 v19, v20, s28 // (op 2229) 

v_add_u32 v20, v15, v19 // (op 2229) 


s_mul_i32 s1, s28, s33 // (op 2231) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 2231) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 2231) least significant
s_mul_hi_u32 s7, s28, s32 // (op 2231) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2231) most significant: sum
s_add_u32 s1, s1, s7 // (op 2231) most significant: sum


s_mov_b32 s52, s30 // (op 2231) reg expression
s_mov_b32 s53, s31 // (op 2231) reg expression
s_mov_b32 s54, s0 // (op 2231) reg expression
s_mov_b32 s55, 131072 // (op 2231) call()

v_lshrrev_b32 v15, 6, v1 // (op 2293) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 2293) 
v_and_b32 v19, -64, v1 // (op 2293) 
s_lshr_b32 s6, s0, 1 // (op 2293) 

v_sub_u32 v21, v1, v19 // (op 2293) 

v_lshl_add_u32 v19, s6, 3, v15 // (op 2293) 


v_lshl_add_u32 v15, s17, 6, v21 // (op 2293) 

v_mul_lo_u32 v21, v19, s34 // (op 2293) 
v_add_u32 v21, v21, v15 // (op 2293) 


s_mul_i32 s1, s34, s33 // (op 2295) most significant: low of low * high
s_mul_i32 s6, s35, s32 // (op 2295) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 2295) least significant
s_mul_hi_u32 s7, s34, s32 // (op 2295) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2295) most significant: sum
s_add_u32 s1, s1, s7 // (op 2295) most significant: sum


s_mov_b32 s56, s36 // (op 2295) reg expression
s_mov_b32 s57, s37 // (op 2295) reg expression
s_mov_b32 s58, s0 // (op 2295) reg expression
s_mov_b32 s59, 131072 // (op 2295) call()

v_and_b32 v15, -2, v1 // (op 2391) 
v_sub_u32 v19, v1, v15 // (op 2391) 

v_lshlrev_b32 v15, 5, v19 // (op 2391) 

v_lshrrev_b32 v19, 1, v1 // (op 2391) 
v_lshl_add_u32 v22, v19, 6, v15 // (op 2391) 


v_lshlrev_b32 v15, 2, v22 // (op 2391) 

v_lshrrev_b32 v19, 3, v15 // (op 2391) 

v_and_b32 v15, -4, v1 // (op 2449) 
v_sub_u32 v22, v1, v15 // (op 2449) 

v_lshlrev_b32 v15, 4, v22 // (op 2449) 

v_lshrrev_b32 v22, 2, v1 // (op 2449) 
v_lshl_add_u32 v23, v22, 7, v15 // (op 2449) 


v_and_b32 v15, -64, v1 // (op 2507) 
v_sub_u32 v22, v1, v15 // (op 2507) 

v_and_b32 v15, 4294967232, v1 // (op 2507) 
v_add_u32 v24, v15, v22 // (op 2507) 


v_and_b32 v15, -64, v1 // (op 2565) 
v_sub_u32 v22, v1, v15 // (op 2565) 

v_and_b32 v15, 4294967232, v1 // (op 2565) 
v_add_u32 v25, v15, v22 // (op 2565) 


s_ashr_i64 s[0:1], s[22:23], 7 // (op 598) 
s_and_b32 s6, s0, 4294967294 // (op 598) 

s_ashr_i64 s[0:1], s[22:23], 7 // (op 597) 
s_mov_b32 s7, s0 // (op 597) convert

s_cmp_lt_u32 s6, s7 // (op 597) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC_ForLoopBottom_KLoopTail_597 // (op 597) Condition: Top (jump to RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC_ForLoopBottom_KLoopTail_597 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC_ForLoopTop_KLoopTail_597:


s_add_u32 s6, s6, 1 // (op 599) 
v_mov_b32 v18, v17 // (op 180) 
v_mov_b32 v22, v18 // (op 180) reg expression

buffer_load_dwordx4 v[26:29], v22, s[8:11], 0 offen // (op 180) Load value


s_lshl_b64 s[0:1], s[12:13], 7 // (op 2085) 
s_lshl_b64 s[14:15], s[0:1], 2 // (op 2085) 

s_lshr_b64 s[0:1], s[14:15], 3 // (op 2085) 

v_add_u32 v17, v17, s0 // (op 2085) 


s_barrier  // (op 2794) 
v_mov_b64 v[30:31], s[24:25] // (op 194) convert




v_mov_b32 v18, v14 // (op 194) 
v_mov_b32 v22, v18 // (op 194) reg expression
buffer_load_dwordx4 v[32:35], v22, s[48:51], 0 offen // (op 194) Load value
v_add_u32 v22, v22, v30 // (op 194) 
buffer_load_dwordx4 v[36:39], v22, s[48:51], 0 offen // (op 194) Load value



s_lshl_b32 s7, s24, 7 // (op 2157) 
v_add_u32 v14, v14, s7 // (op 2157) 

s_waitcnt vmcnt(2)
ds_write_b128 v19, v[26:29] offset:8448 // (op 181) Store local data 



v_mov_b32 v18, v20 // (op 195) 
s_barrier  // (op 2807) 
v_mov_b32 v22, v18 // (op 195) reg expression




buffer_load_ubyte v15, v22, s[52:55], 0 offen offset: 0 // (op 195) Load value


s_lshl_b32 s7, s28, 2 // (op 2221) 
v_add_u32 v20, v20, s7 // (op 2221) 

s_waitcnt vmcnt(2)
ds_write_b128 v23, v[32:35] // (op 191) Store local data 


v_mov_b32 v16, v21 // (op 196) 
v_mov_b32 v18, v16 // (op 196) reg expression

buffer_load_ubyte v22, v18, s[56:59], 0 offen offset: 0 // (op 196) Load value


s_lshl_b32 s7, s34, 2 // (op 2285) 
v_add_u32 v21, v21, s7 // (op 2285) 

s_waitcnt vmcnt(2)
ds_write_b128 v23, v[36:39] offset:64 // (op 191) Store local data 

s_waitcnt lgkmcnt(2)
s_barrier  // (op 2797) 



s_barrier  // (op 2821) 
ds_read_b64_tr_b4 v[26:27], v3 offset:8448 // (op 183) Transpose load local data 



ds_read_b64_tr_b4 v[28:29], v3 offset:8960 // (op 183) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v24, v15 offset:12544 // (op 192) Store local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 2811) 



s_barrier  // (op 2835) 
ds_read_b64_tr_b8 v[30:31], v4 // (op 185) Transpose load local data 



ds_read_b64_tr_b8 v[32:33], v4 offset:512 // (op 185) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v4 offset:4096 // (op 185) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v4 offset:4608 // (op 185) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v25, v22 offset:8192 // (op 193) Store local data 

s_waitcnt lgkmcnt(5)
s_barrier  // (op 2825) 


ds_read_u8 v18, v0 offset:12544 // (op 187) Load local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 2839) 


ds_read_u8 v22, v5 offset:8192 // (op 189) Load local data 



ds_read_b64_tr_b8 v[38:39], v4 offset:32 // (op 354) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:29], v[30:37], a[0:3], v18, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 190) 
ds_read_b64_tr_b8 v[40:41], v4 offset:544 // (op 354) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v4 offset:4128 // (op 354) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v4 offset:4640 // (op 354) Transpose load local data 


ds_read_u8 v15, v5 offset:8224 // (op 358) Load local data 



ds_read_b64_tr_b4 v[46:47], v3 offset:8464 // (op 572) Transpose load local data 
ds_read_b64_tr_b4 v[48:49], v3 offset:8976 // (op 572) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[26:29], v[38:45], a[4:7], v18, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 359) 




ds_read_u8 v16, v0 offset:12576 // (op 575) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[46:49], v[30:37], a[8:11], v16, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 577) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[46:49], v[38:45], a[12:15], v16, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 608) 




s_ashr_i64 s[0:1], s[22:23], 7 // (op 597) 
s_mov_b32 s7, s0 // (op 597) convert

s_cmp_lt_u32 s6, s7 // (op 597) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC_ForLoopTop_KLoopTail_597 // (op 597) Condition: Bottom (jump to RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC_ForLoopTop_KLoopTail_597 if true)
RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC_ForLoopBottom_KLoopTail_597:























s_barrier  // (op 2937) 








v_cmp_eq_f32 vcc, 0.00000, v10 // (op 135) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC_ConditionalFalse_0_beta_135 // (op 135) Condition: False, jump to RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC_ConditionalFalse_0_beta_135
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
s_branch RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC_ConditionalBottom_0_beta_135 // (op 135) Condition: Done, jump to RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC_ConditionalBottom_0_beta_135
RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC_ConditionalFalse_0_beta_135:


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
RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC_ConditionalBottom_0_beta_135:












s_endpgm  // End of RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC
.LRR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC_end:
.size RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC, .LRR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC_end-RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC
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
  - .name: RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC
    .symbol: RR_GEMM_NT_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_LDSSYNC.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 25600
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 60
    .vgpr_count: 86
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

