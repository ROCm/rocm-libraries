// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2
.p2align 8
.type RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2,@function
RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_exec_begin:


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
s_load_dwordx2 s[40:41], s[0:1], 136 // Load scalar value
s_load_dwordx2 s[42:43], s[0:1], 144 // Load scalar value
s_load_dwordx2 s[44:45], s[0:1], 152 // Load scalar value
s_load_dwordx2 s[46:47], s[0:1], 160 // Load scalar value
s_load_dwordx2 s[48:49], s[0:1], 168 // Load scalar value
s_load_dwordx2 s[50:51], s[0:1], 176 // Load scalar value

v_bfe_u32 v1, v0, 0, 10
v_bfe_u32 v2, v0, 10, 10


v_and_b32 v0, 48, v1 // (op 1908) 
v_and_b32 v3, 63, v1 // (op 1908) 
v_sub_u32 v4, v3, v0 // (op 1908) 

v_lshrrev_b32 v3, 1, v4 // (op 1908) 
v_add_lshl_u32 v5, v0, v3, 4 // (op 1908) 


v_and_b32 v0, 1, v4 // (op 1908) 

v_lshlrev_b32 v3, 3, v0 // (op 1908) 

v_lshrrev_b32 v0, 6, v1 // (op 1908) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 1908) 


v_add_u32 v0, v5, v4 // (op 1908) 


v_and_b32 v3, 48, v1 // (op 1973) 
v_and_b32 v4, 63, v1 // (op 1973) 
v_sub_u32 v5, v4, v3 // (op 1973) 

v_lshrrev_b32 v4, 1, v5 // (op 1973) 
v_add_lshl_u32 v6, v3, v4, 4 // (op 1973) 


v_and_b32 v3, 1, v5 // (op 1973) 

v_lshlrev_b32 v4, 3, v3 // (op 1973) 

v_add_u32 v3, v6, v4 // (op 1973) 


v_lshrrev_b32 v4, 6, v1 // (op 2039) 
v_and_b32 v5, 15, v1 // (op 2039) 
v_lshl_add_u32 v6, v4, 4, v5 // (op 2039) 


v_and_b32 v4, 48, v1 // (op 2039) 
v_add_u32 v5, v4, v6 // (op 2039) 


v_and_b32 v4, 48, v1 // (op 2105) 
v_and_b32 v6, 15, v1 // (op 2105) 
v_add_u32 v7, v4, v6 // (op 2105) 


v_and_b32 v4, 48, v1 // (op 2240) 
v_and_b32 v6, 63, v1 // (op 2240) 
v_sub_u32 v8, v6, v4 // (op 2240) 

v_lshrrev_b32 v6, 1, v8 // (op 2240) 
v_add_lshl_u32 v9, v4, v6, 4 // (op 2240) 


v_and_b32 v4, 1, v8 // (op 2240) 

v_lshlrev_b32 v6, 3, v4 // (op 2240) 

v_lshrrev_b32 v4, 6, v1 // (op 2240) 
v_lshl_add_u32 v8, v4, 4, v6 // (op 2240) 


v_add_u32 v4, v9, v8 // (op 2240) 


v_and_b32 v6, 48, v1 // (op 2305) 
v_and_b32 v8, 63, v1 // (op 2305) 
v_sub_u32 v9, v8, v6 // (op 2305) 

v_lshrrev_b32 v8, 1, v9 // (op 2305) 
v_add_lshl_u32 v10, v6, v8, 4 // (op 2305) 


v_and_b32 v6, 1, v9 // (op 2305) 

v_lshlrev_b32 v8, 3, v6 // (op 2305) 

v_add_u32 v6, v10, v8 // (op 2305) 


v_lshrrev_b32 v8, 6, v1 // (op 2371) 
v_and_b32 v9, 15, v1 // (op 2371) 
v_lshl_add_u32 v10, v8, 4, v9 // (op 2371) 


v_and_b32 v8, 48, v1 // (op 2371) 
v_add_u32 v9, v8, v10 // (op 2371) 


v_and_b32 v8, 48, v1 // (op 2437) 
v_and_b32 v10, 15, v1 // (op 2437) 
v_add_u32 v11, v8, v10 // (op 2437) 


v_and_b32 v8, 48, v1 // (op 2523) 
v_and_b32 v10, 63, v1 // (op 2523) 
v_sub_u32 v12, v10, v8 // (op 2523) 

v_lshrrev_b32 v10, 1, v12 // (op 2523) 
v_add_lshl_u32 v13, v8, v10, 4 // (op 2523) 


v_and_b32 v8, 1, v12 // (op 2523) 

v_lshlrev_b32 v10, 3, v8 // (op 2523) 

v_lshrrev_b32 v8, 6, v1 // (op 2523) 
v_lshl_add_u32 v12, v8, 4, v10 // (op 2523) 


v_add_u32 v8, v13, v12 // (op 2523) 


v_and_b32 v10, 48, v1 // (op 2562) 
v_and_b32 v12, 63, v1 // (op 2562) 
v_sub_u32 v13, v12, v10 // (op 2562) 

v_lshrrev_b32 v12, 1, v13 // (op 2562) 
v_add_lshl_u32 v14, v10, v12, 4 // (op 2562) 


v_and_b32 v10, 1, v13 // (op 2562) 

v_lshlrev_b32 v12, 3, v10 // (op 2562) 

v_add_u32 v10, v14, v12 // (op 2562) 


v_lshrrev_b32 v12, 6, v1 // (op 2586) 
v_and_b32 v13, 15, v1 // (op 2586) 
v_lshl_add_u32 v14, v12, 4, v13 // (op 2586) 


v_and_b32 v12, 48, v1 // (op 2586) 
v_add_u32 v13, v12, v14 // (op 2586) 


v_and_b32 v12, 48, v1 // (op 2610) 
v_and_b32 v14, 15, v1 // (op 2610) 
v_add_u32 v15, v12, v14 // (op 2610) 


v_and_b32 v12, 48, v1 // (op 2649) 
v_and_b32 v14, 63, v1 // (op 2649) 
v_sub_u32 v16, v14, v12 // (op 2649) 

v_lshrrev_b32 v14, 1, v16 // (op 2649) 
v_add_lshl_u32 v17, v12, v14, 4 // (op 2649) 


v_and_b32 v12, 1, v16 // (op 2649) 

v_lshlrev_b32 v14, 3, v12 // (op 2649) 

v_lshrrev_b32 v12, 6, v1 // (op 2649) 
v_lshl_add_u32 v16, v12, 4, v14 // (op 2649) 


v_add_u32 v12, v17, v16 // (op 2649) 


v_and_b32 v14, 48, v1 // (op 2688) 
v_and_b32 v16, 63, v1 // (op 2688) 
v_sub_u32 v17, v16, v14 // (op 2688) 

v_lshrrev_b32 v16, 1, v17 // (op 2688) 
v_add_lshl_u32 v18, v14, v16, 4 // (op 2688) 


v_and_b32 v14, 1, v17 // (op 2688) 

v_lshlrev_b32 v16, 3, v14 // (op 2688) 

v_add_u32 v14, v18, v16 // (op 2688) 


v_lshrrev_b32 v16, 6, v1 // (op 2712) 
v_and_b32 v17, 15, v1 // (op 2712) 
v_lshl_add_u32 v18, v16, 4, v17 // (op 2712) 


v_and_b32 v16, 48, v1 // (op 2712) 
v_add_u32 v17, v16, v18 // (op 2712) 


v_and_b32 v16, 48, v1 // (op 2736) 
v_and_b32 v18, 15, v1 // (op 2736) 
v_add_u32 v19, v16, v18 // (op 2736) 


v_mov_b32 v16, s6 // (op 2) Move value
v_mov_b32 v18, s7 // (op 43) Move value


s_mov_b32 s0, 0 // (op 435) call()
s_mov_b32 s1, 0 // (op 436) call()
s_mov_b32 s6, 0 // (op 437) call()
s_mov_b32 s7, 0 // (op 438) call()
s_mov_b32 s18, 0 // (op 439) call()
s_mov_b32 s19, 0 // (op 440) call()
s_mov_b32 s0, 0 // (op 335) call()

s_waitcnt lgkmcnt(0)// DEBUG: Wait for scalar queue
s_mul_i32 s52, s8, s16 // (op 334) 
s_cmp_gt_u32 s9, s52 // (op 334) 
s_mov_b32 s53, scc // (op 334) 
s_sub_u32 s54, s9, s52 // (op 334) 

s_cmp_eq_i32 s53, 0 // (op 334) 
s_cselect_b32 s52, 0, s54 // (op 334) 


s_cmp_lt_u32 s8, s52 // (op 334) 
s_mov_b32 s53, scc // (op 334) 
s_cmp_eq_i32 s53, 0 // (op 334) 
s_cselect_b32 s54, s52, s8 // (op 334) 


s_cmp_lt_u32 s0, s54 // (op 334) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopBottom_SKStreamTileLoop_334 // (op 334) Condition: Top (jump to RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopBottom_SKStreamTileLoop_334 if false)
RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopTop_SKStreamTileLoop_334:
v_lshrrev_b32 v20, 6, v1 // (op 1834) 
s_mul_i32 s52, s16, s8 // (op 1834) 
s_add_u32 s52, s52, s0 // (op 1834) 
v_and_b32 v21, 63, v1 // (op 1834) 
s_mul_hi_u32 s53, s52, s13 // (op 1834) 
v_lshrrev_b32 v22, 4, v21 // (op 1834) 
s_sub_u32 s54, s52, s53 // (op 1834) 
v_lshlrev_b32 v23, 2, v22 // (op 1834) 

v_lshl_add_u32 v22, v20, 4, v23 // (op 1834) 


s_lshr_b32 s55, s54, 1 // (op 1834) 

s_add_u32 s54, s55, s53 // (op 1834) 


v_lshrrev_b32 v20, 2, v21 // (op 1834) 

s_lshr_b32 s53, s54, s11 // (op 1834) 

v_and_b32 v21, 3, v20 // (op 1834) 

s_and_b32 s54, s11, 2147483648 // (op 1834) 
v_and_b32 v20, 3, v1 // (op 1834) 
s_cmp_eq_i32 s54, 0 // (op 1834) 
s_mov_b32 s55, scc // (op 1834) 

v_lshl_add_u32 v23, v21, 2, v20 // (op 1834) 


s_cmp_eq_i32 s55, 0 // (op 1834) 
s_cselect_b32 s54, s52, s53 // (op 1834) 



s_mul_hi_u32 s52, s54, s12 // (op 1834) 
s_sub_u32 s53, s54, s52 // (op 1834) 
s_lshr_b32 s55, s53, 1 // (op 1834) 

s_add_u32 s53, s55, s52 // (op 1834) 


s_lshr_b32 s52, s53, s10 // (op 1834) 

s_and_b32 s53, s10, 2147483648 // (op 1834) 
s_cmp_eq_i32 s53, 0 // (op 1834) 
s_mov_b32 s55, scc // (op 1834) 

s_cmp_eq_i32 s55, 0 // (op 1834) 
s_cselect_b32 s53, s54, s52 // (op 1834) 


s_mul_i32 s52, s53, s14 // (op 1834) 
v_lshl_add_u32 v20, s53, 4, v22 // (op 1834) 


s_sub_u32 s53, s54, s52 // (op 1834) 


v_lshl_add_u32 v21, s53, 4, v23 // (op 1834) 


v_mul_lo_u32 v22, v21, s4 // (op 1834) 

v_add_lshl_u32 v21, v20, v22, 1 // (op 1834) 


s_mul_i32 s53, s4, s23 // (op 1836) most significant: low of low * high
s_mul_i32 s54, s5, s22 // (op 1836) most significant: low of high * low
s_mul_i32 s52, s4, s22 // (op 1836) least significant
s_mul_hi_u32 s55, s4, s22 // (op 1836) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1836) most significant: sum
s_add_u32 s53, s53, s55 // (op 1836) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 1 // (op 1836) 

s_mov_b32 s56, s20 // (op 1836) reg expression
s_mov_b32 s57, s21 // (op 1836) reg expression
s_mov_b32 s58, s54 // (op 1836) reg expression
s_mov_b32 s59, 131072 // (op 1836) call()

v_and_b32 v20, 48, v1 // (op 1920) 
v_and_b32 v22, 63, v1 // (op 1920) 
v_sub_u32 v23, v22, v20 // (op 1920) 

v_lshrrev_b32 v22, 1, v23 // (op 1920) 
v_add_lshl_u32 v24, v20, v22, 4 // (op 1920) 


v_and_b32 v20, 1, v23 // (op 1920) 

v_lshlrev_b32 v22, 3, v20 // (op 1920) 

v_lshrrev_b32 v20, 6, v1 // (op 1920) 
v_lshl_add_u32 v23, v20, 4, v22 // (op 1920) 


v_add_u32 v20, v24, v23 // (op 1920) 


v_and_b32 v22, 48, v1 // (op 1985) 
v_and_b32 v23, 63, v1 // (op 1985) 
v_sub_u32 v24, v23, v22 // (op 1985) 

v_lshrrev_b32 v23, 1, v24 // (op 1985) 
v_add_lshl_u32 v25, v22, v23, 4 // (op 1985) 


v_and_b32 v22, 1, v24 // (op 1985) 

v_lshlrev_b32 v23, 3, v22 // (op 1985) 

v_add_u32 v22, v25, v23 // (op 1985) 


v_lshrrev_b32 v23, 6, v1 // (op 2051) 
v_and_b32 v24, 15, v1 // (op 2051) 
v_lshl_add_u32 v25, v23, 4, v24 // (op 2051) 


v_and_b32 v23, 48, v1 // (op 2051) 
v_add_u32 v24, v23, v25 // (op 2051) 


v_and_b32 v23, 48, v1 // (op 2117) 
v_and_b32 v25, 15, v1 // (op 2117) 
v_add_u32 v26, v23, v25 // (op 2117) 


s_mul_i32 s6, s8, s16 // (op 339) 
s_add_u32 s6, s6, s0 // (op 339) 
s_mul_hi_u32 s52, s6, s13 // (op 340) 
s_sub_u32 s53, s6, s52 // (op 340) 
s_lshr_b32 s54, s53, 1 // (op 340) 

s_add_u32 s53, s54, s52 // (op 340) 


s_lshr_b32 s52, s53, s11 // (op 340) 

s_and_b32 s53, s11, 2147483648 // (op 340) 
s_cmp_eq_i32 s53, 0 // (op 340) 
s_mov_b32 s54, scc // (op 340) 

s_cmp_eq_i32 s54, 0 // (op 340) 
s_cselect_b32 s53, s6, s52 // (op 340) 


s_ashr_i64 s[54:55], s[2:3], 8 // (op 340) 
s_mul_i32 s52, s53, s54 // (op 340) 


s_sub_u32 s18, s6, s52 // (op 340) 

s_ashr_i64 s[52:53], s[2:3], 8 // (op 341) 
s_cmp_gt_u32 s52, s18 // (op 341) 
s_mov_b32 s54, scc // (op 341) 
s_sub_u32 s55, s52, s18 // (op 341) 

s_cmp_eq_i32 s54, 0 // (op 341) 
s_cselect_b32 s52, 0, s55 // (op 341) 


s_cmp_gt_u32 s8, s0 // (op 341) 
s_mov_b32 s53, scc // (op 341) 
s_sub_u32 s54, s8, s0 // (op 341) 
s_cmp_eq_i32 s53, 0 // (op 341) 
s_cselect_b32 s55, 0, s54 // (op 341) 


s_cmp_lt_u32 s52, s55 // (op 341) 
s_mov_b32 s53, scc // (op 341) 
s_cmp_eq_i32 s53, 0 // (op 341) 
s_cselect_b32 s54, s55, s52 // (op 341) 



s_cmp_gt_u32 s9, s6 // (op 341) 
s_mov_b32 s55, scc // (op 341) 
s_sub_u32 s52, s9, s6 // (op 341) 
s_cmp_eq_i32 s55, 0 // (op 341) 
s_cselect_b32 s53, 0, s52 // (op 341) 


s_cmp_lt_u32 s54, s53 // (op 341) 
s_mov_b32 s52, scc // (op 341) 
s_cmp_eq_i32 s52, 0 // (op 341) 
s_cselect_b32 s7, s53, s54 // (op 341) 



s_add_u32 s52, s18, s7 // (op 342) 
s_sub_u32 s19, s52, 1 // (op 342) 

v_accvgpr_write a0, 0.00000 // (op 65) call()
v_accvgpr_write a1, 0.00000 // (op 65) call()
v_accvgpr_write a2, 0.00000 // (op 65) call()
v_accvgpr_write a3, 0.00000 // (op 65) call()
s_mul_i32 s52, s16, s8 // (op 1366) 
s_add_u32 s52, s52, s0 // (op 1366) 
s_mul_hi_u32 s53, s52, s13 // (op 1366) 
s_sub_u32 s54, s52, s53 // (op 1366) 
s_lshr_b32 s55, s54, 1 // (op 1366) 

s_add_u32 s54, s55, s53 // (op 1366) 


s_lshr_b32 s53, s54, s11 // (op 1366) 

s_and_b32 s54, s11, 2147483648 // (op 1366) 
s_cmp_eq_i32 s54, 0 // (op 1366) 
s_mov_b32 s55, scc // (op 1366) 

s_cmp_eq_i32 s55, 0 // (op 1366) 
s_cselect_b32 s54, s52, s53 // (op 1366) 


s_mul_hi_u32 s53, s54, s12 // (op 1366) 
s_sub_u32 s55, s54, s53 // (op 1366) 
s_lshr_b32 s60, s55, 1 // (op 1366) 

s_add_u32 s55, s60, s53 // (op 1366) 


s_lshr_b32 s53, s55, s10 // (op 1366) 

s_and_b32 s55, s10, 2147483648 // (op 1366) 
s_cmp_eq_i32 s55, 0 // (op 1366) 
s_mov_b32 s60, scc // (op 1366) 

s_cmp_eq_i32 s60, 0 // (op 1366) 
s_cselect_b32 s55, s54, s53 // (op 1366) 


s_lshl_b32 s53, s55, 4 // (op 1366) 

s_mov_b32 s61, 0 // (op 1366) convert
s_mov_b32 s60, s53 // (op 1366) convert

s_ashr_i64 s[62:63], s[2:3], 8 // (op 1366) 
s_mul_i32 s53, s54, s62 // (op 1366) 


s_sub_u32 s54, s52, s53 // (op 1366) 


v_lshl_add_u32 v23, s54, 8, v1 // (op 1366) 

v_mul_lo_u32 v29, v23, s25 // (op 1366) most significant: low of low * high
v_mul_hi_u32 v25, v23, s24 // (op 1366) most significant: high of low * low
v_mul_lo_u32 v28, v23, s24 // (op 1366) least significant: low of low * low
v_add_u32 v29, v29, v25 // (op 1366) most significant: sum


v_mov_b32 v23, s60 // (op 1366) 
v_mov_b32 v25, s61 // (op 1366) 
v_add_co_u32 v30, s[52:53], v23, v28 // (op 1366) least significant half
v_addc_co_u32 v31, s[52:53], v25, v29, s[52:53] // (op 1366) most significant half





s_mul_i32 s53, s24, s3 // (op 1368) most significant: low of low * high
s_mul_i32 s54, s25, s2 // (op 1368) most significant: low of high * low
s_mul_i32 s52, s24, s2 // (op 1368) least significant
s_mul_hi_u32 s55, s24, s2 // (op 1368) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1368) most significant: sum
s_add_u32 s53, s53, s55 // (op 1368) most significant: sum


s_mov_b32 s60, s26 // (op 1368) reg expression
s_mov_b32 s61, s27 // (op 1368) reg expression
s_mov_b32 s62, s52 // (op 1368) reg expression
s_mov_b32 s63, 131072 // (op 1368) call()

s_mul_i32 s52, s16, s8 // (op 1385) 
s_add_u32 s52, s52, s0 // (op 1385) 
s_mul_hi_u32 s53, s52, s13 // (op 1385) 
s_sub_u32 s54, s52, s53 // (op 1385) 
s_lshr_b32 s55, s54, 1 // (op 1385) 

s_add_u32 s54, s55, s53 // (op 1385) 


s_lshr_b32 s53, s54, s11 // (op 1385) 

s_and_b32 s54, s11, 2147483648 // (op 1385) 
s_cmp_eq_i32 s54, 0 // (op 1385) 
s_mov_b32 s55, scc // (op 1385) 

s_cmp_eq_i32 s55, 0 // (op 1385) 
s_cselect_b32 s54, s52, s53 // (op 1385) 


s_mul_hi_u32 s53, s54, s12 // (op 1385) 
s_sub_u32 s55, s54, s53 // (op 1385) 
s_lshr_b32 s64, s55, 1 // (op 1385) 

s_add_u32 s55, s64, s53 // (op 1385) 


s_lshr_b32 s53, s55, s10 // (op 1385) 

s_and_b32 s55, s10, 2147483648 // (op 1385) 
s_cmp_eq_i32 s55, 0 // (op 1385) 
s_mov_b32 s64, scc // (op 1385) 

s_cmp_eq_i32 s64, 0 // (op 1385) 
s_cselect_b32 s55, s54, s53 // (op 1385) 


s_lshl_b32 s53, s55, 4 // (op 1385) 

s_mov_b32 s65, 0 // (op 1385) convert
s_mov_b32 s64, s53 // (op 1385) convert

s_ashr_i64 s[66:67], s[2:3], 8 // (op 1385) 
s_mul_i32 s53, s54, s66 // (op 1385) 


s_sub_u32 s54, s52, s53 // (op 1385) 


v_lshl_add_u32 v23, s54, 8, v1 // (op 1385) 

v_mul_lo_u32 v29, v23, s25 // (op 1385) most significant: low of low * high
v_mul_hi_u32 v25, v23, s24 // (op 1385) most significant: high of low * low
v_mul_lo_u32 v28, v23, s24 // (op 1385) least significant: low of low * low
v_add_u32 v29, v29, v25 // (op 1385) most significant: sum


v_mov_b32 v23, s64 // (op 1385) 
v_mov_b32 v25, s65 // (op 1385) 
v_add_co_u32 v32, s[52:53], v23, v28 // (op 1385) least significant half
v_addc_co_u32 v33, s[52:53], v25, v29, s[52:53] // (op 1385) most significant half





s_mul_i32 s53, s24, s3 // (op 1387) most significant: low of low * high
s_mul_i32 s54, s25, s2 // (op 1387) most significant: low of high * low
s_mul_i32 s52, s24, s2 // (op 1387) least significant
s_mul_hi_u32 s55, s24, s2 // (op 1387) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1387) most significant: sum
s_add_u32 s53, s53, s55 // (op 1387) most significant: sum


s_mov_b32 s60, s26 // (op 1387) reg expression
s_mov_b32 s61, s27 // (op 1387) reg expression
s_mov_b32 s62, s52 // (op 1387) reg expression
s_mov_b32 s63, 131072 // (op 1387) call()

s_mul_i32 s52, s16, s8 // (op 1497) 
s_add_u32 s52, s52, s0 // (op 1497) 
s_mul_hi_u32 s53, s52, s13 // (op 1497) 
s_sub_u32 s54, s52, s53 // (op 1497) 
s_lshr_b32 s55, s54, 1 // (op 1497) 

s_add_u32 s54, s55, s53 // (op 1497) 


s_lshr_b32 s53, s54, s11 // (op 1497) 

s_and_b32 s54, s11, 2147483648 // (op 1497) 
s_cmp_eq_i32 s54, 0 // (op 1497) 
s_mov_b32 s55, scc // (op 1497) 

s_cmp_eq_i32 s55, 0 // (op 1497) 
s_cselect_b32 s54, s52, s53 // (op 1497) 


s_mul_hi_u32 s53, s54, s12 // (op 1497) 
s_sub_u32 s55, s54, s53 // (op 1497) 
s_lshr_b32 s64, s55, 1 // (op 1497) 

s_add_u32 s55, s64, s53 // (op 1497) 


s_lshr_b32 s53, s55, s10 // (op 1497) 

s_and_b32 s55, s10, 2147483648 // (op 1497) 
s_cmp_eq_i32 s55, 0 // (op 1497) 
s_mov_b32 s64, scc // (op 1497) 

s_cmp_eq_i32 s64, 0 // (op 1497) 
s_cselect_b32 s55, s54, s53 // (op 1497) 


s_mul_i32 s53, s55, s14 // (op 1497) 

s_sub_u32 s55, s54, s53 // (op 1497) 

s_lshl_b32 s53, s55, 4 // (op 1497) 

s_mov_b32 s65, 0 // (op 1497) convert
s_mov_b32 s64, s53 // (op 1497) convert

s_ashr_i64 s[66:67], s[2:3], 8 // (op 1497) 
s_mul_i32 s53, s54, s66 // (op 1497) 


s_sub_u32 s54, s52, s53 // (op 1497) 


v_lshl_add_u32 v23, s54, 8, v1 // (op 1497) 

v_mul_lo_u32 v29, v23, s29 // (op 1497) most significant: low of low * high
v_mul_hi_u32 v25, v23, s28 // (op 1497) most significant: high of low * low
v_mul_lo_u32 v28, v23, s28 // (op 1497) least significant: low of low * low
v_add_u32 v29, v29, v25 // (op 1497) most significant: sum

v_mov_b32 v25, s64 // (op 1497) 
v_mov_b32 v27, s65 // (op 1497) 
v_add_co_u32 v28, s[52:53], v28, v25 // (op 1497) least significant half
v_addc_co_u32 v29, s[52:53], v29, v27, s[52:53] // (op 1497) most significant half





s_mul_i32 s53, s28, s3 // (op 1499) most significant: low of low * high
s_mul_i32 s54, s29, s2 // (op 1499) most significant: low of high * low
s_mul_i32 s52, s28, s2 // (op 1499) least significant
s_mul_hi_u32 s55, s28, s2 // (op 1499) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1499) most significant: sum
s_add_u32 s53, s53, s55 // (op 1499) most significant: sum


s_mov_b32 s64, s30 // (op 1499) reg expression
s_mov_b32 s65, s31 // (op 1499) reg expression
s_mov_b32 s66, s52 // (op 1499) reg expression
s_mov_b32 s67, 131072 // (op 1499) call()

s_mul_i32 s52, s16, s8 // (op 1516) 
s_add_u32 s52, s52, s0 // (op 1516) 
s_mul_hi_u32 s53, s52, s13 // (op 1516) 
s_sub_u32 s54, s52, s53 // (op 1516) 
s_lshr_b32 s55, s54, 1 // (op 1516) 

s_add_u32 s54, s55, s53 // (op 1516) 


s_lshr_b32 s53, s54, s11 // (op 1516) 

s_and_b32 s54, s11, 2147483648 // (op 1516) 
s_cmp_eq_i32 s54, 0 // (op 1516) 
s_mov_b32 s55, scc // (op 1516) 

s_cmp_eq_i32 s55, 0 // (op 1516) 
s_cselect_b32 s54, s52, s53 // (op 1516) 


s_mul_hi_u32 s53, s54, s12 // (op 1516) 
s_sub_u32 s55, s54, s53 // (op 1516) 
s_lshr_b32 s68, s55, 1 // (op 1516) 

s_add_u32 s55, s68, s53 // (op 1516) 


s_lshr_b32 s53, s55, s10 // (op 1516) 

s_and_b32 s55, s10, 2147483648 // (op 1516) 
s_cmp_eq_i32 s55, 0 // (op 1516) 
s_mov_b32 s68, scc // (op 1516) 

s_cmp_eq_i32 s68, 0 // (op 1516) 
s_cselect_b32 s55, s54, s53 // (op 1516) 


s_mul_i32 s53, s55, s14 // (op 1516) 

s_sub_u32 s55, s54, s53 // (op 1516) 

s_lshl_b32 s53, s55, 4 // (op 1516) 

s_mov_b32 s69, 0 // (op 1516) convert
s_mov_b32 s68, s53 // (op 1516) convert

s_ashr_i64 s[70:71], s[2:3], 8 // (op 1516) 
s_mul_i32 s53, s54, s70 // (op 1516) 


s_sub_u32 s54, s52, s53 // (op 1516) 


v_lshl_add_u32 v23, s54, 8, v1 // (op 1516) 

v_mul_lo_u32 v35, v23, s29 // (op 1516) most significant: low of low * high
v_mul_hi_u32 v25, v23, s28 // (op 1516) most significant: high of low * low
v_mul_lo_u32 v34, v23, s28 // (op 1516) least significant: low of low * low
v_add_u32 v35, v35, v25 // (op 1516) most significant: sum

v_mov_b32 v25, s68 // (op 1516) 
v_mov_b32 v27, s69 // (op 1516) 
v_add_co_u32 v34, s[52:53], v34, v25 // (op 1516) least significant half
v_addc_co_u32 v35, s[52:53], v35, v27, s[52:53] // (op 1516) most significant half





s_mul_i32 s53, s28, s3 // (op 1518) most significant: low of low * high
s_mul_i32 s54, s29, s2 // (op 1518) most significant: low of high * low
s_mul_i32 s52, s28, s2 // (op 1518) least significant
s_mul_hi_u32 s55, s28, s2 // (op 1518) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1518) most significant: sum
s_add_u32 s53, s53, s55 // (op 1518) most significant: sum


s_mov_b32 s64, s30 // (op 1518) reg expression
s_mov_b32 s65, s31 // (op 1518) reg expression
s_mov_b32 s66, s52 // (op 1518) reg expression
s_mov_b32 s67, 131072 // (op 1518) call()

v_and_b32 v23, -16, v1 // (op 1625) 
s_mul_i32 s52, s16, s8 // (op 1625) 
s_add_u32 s52, s52, s0 // (op 1625) 
v_sub_u32 v25, v1, v23 // (op 1625) 

s_mul_hi_u32 s53, s52, s13 // (op 1625) 
v_lshrrev_b32 v23, 4, v1 // (op 1625) 
s_sub_u32 s54, s52, s53 // (op 1625) 
v_lshlrev_b32 v27, 1, v23 // (op 1625) 

s_lshr_b32 s55, s54, 1 // (op 1625) 

s_add_u32 s54, s55, s53 // (op 1625) 


s_lshr_b32 s53, s54, s11 // (op 1625) 

s_and_b32 s54, s11, 2147483648 // (op 1625) 
s_cmp_eq_i32 s54, 0 // (op 1625) 
s_mov_b32 s55, scc // (op 1625) 

s_cmp_eq_i32 s55, 0 // (op 1625) 
s_cselect_b32 s54, s52, s53 // (op 1625) 


s_mul_hi_u32 s53, s54, s12 // (op 1625) 
s_sub_u32 s55, s54, s53 // (op 1625) 
s_lshr_b32 s68, s55, 1 // (op 1625) 

s_add_u32 s55, s68, s53 // (op 1625) 


s_lshr_b32 s53, s55, s10 // (op 1625) 

s_and_b32 s55, s10, 2147483648 // (op 1625) 
s_cmp_eq_i32 s55, 0 // (op 1625) 
s_mov_b32 s68, scc // (op 1625) 

s_cmp_eq_i32 s68, 0 // (op 1625) 
s_cselect_b32 s55, s54, s53 // (op 1625) 


v_lshl_add_u32 v23, s55, 4, v25 // (op 1625) 


s_ashr_i64 s[68:69], s[2:3], 8 // (op 1625) 
s_mul_i32 s53, s54, s68 // (op 1625) 


s_sub_u32 s54, s52, s53 // (op 1625) 


v_lshl_add_u32 v25, s54, 3, v27 // (op 1625) 


v_mul_lo_u32 v27, v25, s32 // (op 1625) 

v_add_u32 v25, v23, v27 // (op 1625) 


s_mul_i32 s53, s32, s37 // (op 1627) most significant: low of low * high
s_mul_i32 s54, s33, s36 // (op 1627) most significant: low of high * low
s_mul_i32 s52, s32, s36 // (op 1627) least significant
s_mul_hi_u32 s55, s32, s36 // (op 1627) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1627) most significant: sum
s_add_u32 s53, s53, s55 // (op 1627) most significant: sum


s_mov_b32 s68, s34 // (op 1627) reg expression
s_mov_b32 s69, s35 // (op 1627) reg expression
s_mov_b32 s70, s52 // (op 1627) reg expression
s_mov_b32 s71, 131072 // (op 1627) call()

v_and_b32 v23, -16, v1 // (op 1644) 
s_mul_i32 s52, s16, s8 // (op 1644) 
s_add_u32 s52, s52, s0 // (op 1644) 
v_sub_u32 v27, v1, v23 // (op 1644) 

s_mul_hi_u32 s53, s52, s13 // (op 1644) 
v_lshrrev_b32 v23, 4, v1 // (op 1644) 
s_sub_u32 s54, s52, s53 // (op 1644) 
v_lshlrev_b32 v36, 1, v23 // (op 1644) 

s_lshr_b32 s55, s54, 1 // (op 1644) 

s_add_u32 s54, s55, s53 // (op 1644) 


s_lshr_b32 s53, s54, s11 // (op 1644) 

s_and_b32 s54, s11, 2147483648 // (op 1644) 
s_cmp_eq_i32 s54, 0 // (op 1644) 
s_mov_b32 s55, scc // (op 1644) 

s_cmp_eq_i32 s55, 0 // (op 1644) 
s_cselect_b32 s54, s52, s53 // (op 1644) 


s_mul_hi_u32 s53, s54, s12 // (op 1644) 
s_sub_u32 s55, s54, s53 // (op 1644) 
s_lshr_b32 s72, s55, 1 // (op 1644) 

s_add_u32 s55, s72, s53 // (op 1644) 


s_lshr_b32 s53, s55, s10 // (op 1644) 

s_and_b32 s55, s10, 2147483648 // (op 1644) 
s_cmp_eq_i32 s55, 0 // (op 1644) 
s_mov_b32 s72, scc // (op 1644) 

s_cmp_eq_i32 s72, 0 // (op 1644) 
s_cselect_b32 s55, s54, s53 // (op 1644) 


v_lshl_add_u32 v23, s55, 4, v27 // (op 1644) 


s_ashr_i64 s[72:73], s[2:3], 8 // (op 1644) 
s_mul_i32 s53, s54, s72 // (op 1644) 


s_sub_u32 s54, s52, s53 // (op 1644) 


v_lshl_add_u32 v27, s54, 3, v36 // (op 1644) 


v_mul_lo_u32 v36, v27, s32 // (op 1644) 

v_add_u32 v27, v23, v36 // (op 1644) 


s_mul_i32 s53, s32, s37 // (op 1646) most significant: low of low * high
s_mul_i32 s54, s33, s36 // (op 1646) most significant: low of high * low
s_mul_i32 s52, s32, s36 // (op 1646) least significant
s_mul_hi_u32 s55, s32, s36 // (op 1646) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1646) most significant: sum
s_add_u32 s53, s53, s55 // (op 1646) most significant: sum


s_mov_b32 s68, s34 // (op 1646) reg expression
s_mov_b32 s69, s35 // (op 1646) reg expression
s_mov_b32 s70, s52 // (op 1646) reg expression
s_mov_b32 s71, 131072 // (op 1646) call()

v_and_b32 v23, -16, v1 // (op 1753) 
s_mul_i32 s52, s16, s8 // (op 1753) 
s_add_u32 s52, s52, s0 // (op 1753) 
v_sub_u32 v36, v1, v23 // (op 1753) 

s_mul_hi_u32 s53, s52, s13 // (op 1753) 
v_lshrrev_b32 v23, 4, v1 // (op 1753) 
s_sub_u32 s54, s52, s53 // (op 1753) 
v_lshlrev_b32 v37, 1, v23 // (op 1753) 

s_lshr_b32 s55, s54, 1 // (op 1753) 

s_add_u32 s54, s55, s53 // (op 1753) 


s_lshr_b32 s53, s54, s11 // (op 1753) 

s_and_b32 s54, s11, 2147483648 // (op 1753) 
s_cmp_eq_i32 s54, 0 // (op 1753) 
s_mov_b32 s55, scc // (op 1753) 

s_cmp_eq_i32 s55, 0 // (op 1753) 
s_cselect_b32 s54, s52, s53 // (op 1753) 


s_mul_hi_u32 s53, s54, s12 // (op 1753) 
s_sub_u32 s55, s54, s53 // (op 1753) 
s_lshr_b32 s72, s55, 1 // (op 1753) 

s_add_u32 s55, s72, s53 // (op 1753) 


s_lshr_b32 s53, s55, s10 // (op 1753) 

s_and_b32 s55, s10, 2147483648 // (op 1753) 
s_cmp_eq_i32 s55, 0 // (op 1753) 
s_mov_b32 s72, scc // (op 1753) 

s_cmp_eq_i32 s72, 0 // (op 1753) 
s_cselect_b32 s55, s54, s53 // (op 1753) 


s_mul_i32 s53, s55, s14 // (op 1753) 

s_sub_u32 s55, s54, s53 // (op 1753) 

v_lshl_add_u32 v23, s55, 4, v36 // (op 1753) 


s_ashr_i64 s[72:73], s[2:3], 8 // (op 1753) 
s_mul_i32 s53, s54, s72 // (op 1753) 


s_sub_u32 s54, s52, s53 // (op 1753) 


v_lshl_add_u32 v36, s54, 3, v37 // (op 1753) 


v_mul_lo_u32 v37, v36, s38 // (op 1753) 
v_add_u32 v37, v37, v23 // (op 1753) 


s_mul_i32 s53, s38, s37 // (op 1755) most significant: low of low * high
s_mul_i32 s54, s39, s36 // (op 1755) most significant: low of high * low
s_mul_i32 s52, s38, s36 // (op 1755) least significant
s_mul_hi_u32 s55, s38, s36 // (op 1755) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1755) most significant: sum
s_add_u32 s53, s53, s55 // (op 1755) most significant: sum


s_mov_b32 s72, s40 // (op 1755) reg expression
s_mov_b32 s73, s41 // (op 1755) reg expression
s_mov_b32 s74, s52 // (op 1755) reg expression
s_mov_b32 s75, 131072 // (op 1755) call()

v_and_b32 v23, -16, v1 // (op 1772) 
s_mul_i32 s52, s16, s8 // (op 1772) 
s_add_u32 s52, s52, s0 // (op 1772) 
v_sub_u32 v36, v1, v23 // (op 1772) 

s_mul_hi_u32 s53, s52, s13 // (op 1772) 
v_lshrrev_b32 v23, 4, v1 // (op 1772) 
s_sub_u32 s54, s52, s53 // (op 1772) 
v_lshlrev_b32 v38, 1, v23 // (op 1772) 

s_lshr_b32 s55, s54, 1 // (op 1772) 

s_add_u32 s54, s55, s53 // (op 1772) 


s_lshr_b32 s53, s54, s11 // (op 1772) 

s_and_b32 s54, s11, 2147483648 // (op 1772) 
s_cmp_eq_i32 s54, 0 // (op 1772) 
s_mov_b32 s55, scc // (op 1772) 

s_cmp_eq_i32 s55, 0 // (op 1772) 
s_cselect_b32 s54, s52, s53 // (op 1772) 


s_mul_hi_u32 s53, s54, s12 // (op 1772) 
s_sub_u32 s55, s54, s53 // (op 1772) 
s_lshr_b32 s76, s55, 1 // (op 1772) 

s_add_u32 s55, s76, s53 // (op 1772) 


s_lshr_b32 s53, s55, s10 // (op 1772) 

s_and_b32 s55, s10, 2147483648 // (op 1772) 
s_cmp_eq_i32 s55, 0 // (op 1772) 
s_mov_b32 s76, scc // (op 1772) 

s_cmp_eq_i32 s76, 0 // (op 1772) 
s_cselect_b32 s55, s54, s53 // (op 1772) 


s_mul_i32 s53, s55, s14 // (op 1772) 

s_sub_u32 s55, s54, s53 // (op 1772) 

v_lshl_add_u32 v23, s55, 4, v36 // (op 1772) 


s_ashr_i64 s[76:77], s[2:3], 8 // (op 1772) 
s_mul_i32 s53, s54, s76 // (op 1772) 


s_sub_u32 s54, s52, s53 // (op 1772) 


v_lshl_add_u32 v36, s54, 3, v38 // (op 1772) 


v_mul_lo_u32 v38, v36, s38 // (op 1772) 
v_add_u32 v38, v38, v23 // (op 1772) 


s_mul_i32 s53, s38, s37 // (op 1774) most significant: low of low * high
s_mul_i32 s54, s39, s36 // (op 1774) most significant: low of high * low
s_mul_i32 s52, s38, s36 // (op 1774) least significant
s_mul_hi_u32 s55, s38, s36 // (op 1774) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1774) most significant: sum
s_add_u32 s53, s53, s55 // (op 1774) most significant: sum


s_mov_b32 s72, s40 // (op 1774) reg expression
s_mov_b32 s73, s41 // (op 1774) reg expression
s_mov_b32 s74, s52 // (op 1774) reg expression
s_mov_b32 s75, 131072 // (op 1774) call()

v_lshlrev_b32 v23, 4, v1 // (op 1961) 
s_nop 0
v_readfirstlane_b32 s52, v23 // (op 1961) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 2870) 

s_lshl_b64 s[54:55], s[28:29], 6 // (op 1277) 
v_mov_b64 v[40:41], s[54:55] // (op 1277) convert


v_mov_b64 v[42:43], v[28:29] // (op 1277) reg expression
s_mov_b32 m0, s52 // (op 1277) reg expression
buffer_load_dwordx4 v42, s[64:67], 0 offen offset: 0 lds // (op 1277) Load value direct to lds
v_add_co_u32 v42, s[54:55], v42, v40 // (op 1277) least significant half
v_addc_co_u32 v43, s[54:55], 0, v43, s[54:55] // (op 1277) most significant half

s_add_u32 m0, m0, 1024 // (op 1277) 
buffer_load_dwordx4 v42, s[64:67], 0 offen offset: 0 lds // (op 1277) Load value direct to lds
v_add_co_u32 v42, s[54:55], v42, v40 // (op 1277) least significant half
v_addc_co_u32 v43, s[54:55], 0, v43, s[54:55] // (op 1277) most significant half

s_add_u32 m0, m0, 1024 // (op 1277) 
buffer_load_dwordx4 v42, s[64:67], 0 offen offset: 0 lds // (op 1277) Load value direct to lds
v_add_co_u32 v42, s[54:55], v42, v40 // (op 1277) least significant half
v_addc_co_u32 v43, s[54:55], 0, v43, s[54:55] // (op 1277) most significant half

s_add_u32 m0, m0, 1024 // (op 1277) 
buffer_load_dwordx4 v42, s[64:67], 0 offen offset: 0 lds // (op 1277) Load value direct to lds




v_mov_b64 v[40:41], s[38:39] // (op 976) convert

v_mov_b32 v23, v37 // (op 976) reg expression
buffer_load_ubyte v42, v23, s[72:75], 0 offen offset: 0 // (op 976) Load value
v_add_u32 v23, v23, v40 // (op 976) 
buffer_load_ubyte v43, v23, s[72:75], 0 offen offset: 0 // (op 976) Load value


v_lshlrev_b32 v23, 4, v1 // (op 1896) 
s_nop 0
v_readfirstlane_b32 s52, v23 // (op 1896) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 2851) 

s_lshl_b64 s[54:55], s[24:25], 6 // (op 1282) 
v_mov_b64 v[40:41], s[54:55] // (op 1282) convert


v_mov_b64 v[44:45], v[30:31] // (op 1282) reg expression
s_add_u32 s53, 4096, s52 // (op 1282) 
s_mov_b32 m0, s53 // (op 1282) reg expression
buffer_load_dwordx4 v44, s[60:63], 0 offen offset: 0 lds // (op 1282) Load value direct to lds
v_add_co_u32 v44, s[54:55], v44, v40 // (op 1282) least significant half
v_addc_co_u32 v45, s[54:55], 0, v45, s[54:55] // (op 1282) most significant half

s_add_u32 m0, m0, 1024 // (op 1282) 
buffer_load_dwordx4 v44, s[60:63], 0 offen offset: 0 lds // (op 1282) Load value direct to lds
v_add_co_u32 v44, s[54:55], v44, v40 // (op 1282) least significant half
v_addc_co_u32 v45, s[54:55], 0, v45, s[54:55] // (op 1282) most significant half

s_add_u32 m0, m0, 1024 // (op 1282) 
buffer_load_dwordx4 v44, s[60:63], 0 offen offset: 0 lds // (op 1282) Load value direct to lds
v_add_co_u32 v44, s[54:55], v44, v40 // (op 1282) least significant half
v_addc_co_u32 v45, s[54:55], 0, v45, s[54:55] // (op 1282) most significant half

s_add_u32 m0, m0, 1024 // (op 1282) 
buffer_load_dwordx4 v44, s[60:63], 0 offen offset: 0 lds // (op 1282) Load value direct to lds





v_mov_b64 v[40:41], s[32:33] // (op 982) convert

v_mov_b32 v23, v25 // (op 982) reg expression
buffer_load_ubyte v44, v23, s[68:71], 0 offen offset: 0 // (op 982) Load value
v_add_u32 v23, v23, v40 // (op 982) 
buffer_load_ubyte v45, v23, s[68:71], 0 offen offset: 0 // (op 982) Load value


v_lshlrev_b32 v23, 4, v1 // (op 2676) 
s_nop 0
v_readfirstlane_b32 s52, v23 // (op 2676) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 3050) 

s_lshl_b64 s[54:55], s[28:29], 6 // (op 1287) 
v_mov_b64 v[40:41], s[54:55] // (op 1287) convert


s_lshl_b64 s[54:55], s[28:29], 8 // (op 1287) 
v_mov_b32 v23, s54 // (op 1287) 
v_mov_b32 v36, s55 // (op 1287) 
v_add_co_u32 v46, s[76:77], v28, v23 // (op 1287) least significant half
v_addc_co_u32 v47, s[76:77], v29, v36, s[76:77] // (op 1287) most significant half




s_add_u32 s53, 8192, s52 // (op 1287) 
s_mov_b32 m0, s53 // (op 1287) reg expression
buffer_load_dwordx4 v46, s[64:67], 0 offen offset: 0 lds // (op 1287) Load value direct to lds
v_add_co_u32 v46, s[54:55], v46, v40 // (op 1287) least significant half
v_addc_co_u32 v47, s[54:55], 0, v47, s[54:55] // (op 1287) most significant half

s_add_u32 m0, m0, 1024 // (op 1287) 
buffer_load_dwordx4 v46, s[64:67], 0 offen offset: 0 lds // (op 1287) Load value direct to lds
v_add_co_u32 v46, s[54:55], v46, v40 // (op 1287) least significant half
v_addc_co_u32 v47, s[54:55], 0, v47, s[54:55] // (op 1287) most significant half

s_add_u32 m0, m0, 1024 // (op 1287) 
buffer_load_dwordx4 v46, s[64:67], 0 offen offset: 0 lds // (op 1287) Load value direct to lds
v_add_co_u32 v46, s[54:55], v46, v40 // (op 1287) least significant half
v_addc_co_u32 v47, s[54:55], 0, v47, s[54:55] // (op 1287) most significant half

s_add_u32 m0, m0, 1024 // (op 1287) 
buffer_load_dwordx4 v46, s[64:67], 0 offen offset: 0 lds // (op 1287) Load value direct to lds






v_mov_b64 v[28:29], s[38:39] // (op 988) convert

s_lshl_b32 s52, s38, 3 // (op 988) 
v_add_u32 v23, v37, s52 // (op 988) 

buffer_load_ubyte v40, v23, s[72:75], 0 offen offset: 0 // (op 988) Load value
v_add_u32 v23, v23, v28 // (op 988) 
buffer_load_ubyte v41, v23, s[72:75], 0 offen offset: 0 // (op 988) Load value



v_lshlrev_b32 v23, 4, v1 // (op 2637) 
s_nop 0
v_readfirstlane_b32 s52, v23 // (op 2637) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 3040) 

s_lshl_b64 s[54:55], s[24:25], 6 // (op 1289) 
v_mov_b64 v[28:29], s[54:55] // (op 1289) convert


s_lshl_b64 s[54:55], s[24:25], 8 // (op 1289) 
v_mov_b32 v23, s54 // (op 1289) 
v_mov_b32 v39, s55 // (op 1289) 
v_add_co_u32 v36, s[76:77], v30, v23 // (op 1289) least significant half
v_addc_co_u32 v37, s[76:77], v31, v39, s[76:77] // (op 1289) most significant half




s_add_u32 s53, 12288, s52 // (op 1289) 
s_mov_b32 m0, s53 // (op 1289) reg expression
buffer_load_dwordx4 v36, s[60:63], 0 offen offset: 0 lds // (op 1289) Load value direct to lds
v_add_co_u32 v36, s[54:55], v36, v28 // (op 1289) least significant half
v_addc_co_u32 v37, s[54:55], 0, v37, s[54:55] // (op 1289) most significant half

s_add_u32 m0, m0, 1024 // (op 1289) 
buffer_load_dwordx4 v36, s[60:63], 0 offen offset: 0 lds // (op 1289) Load value direct to lds
v_add_co_u32 v36, s[54:55], v36, v28 // (op 1289) least significant half
v_addc_co_u32 v37, s[54:55], 0, v37, s[54:55] // (op 1289) most significant half

s_add_u32 m0, m0, 1024 // (op 1289) 
buffer_load_dwordx4 v36, s[60:63], 0 offen offset: 0 lds // (op 1289) Load value direct to lds
v_add_co_u32 v36, s[54:55], v36, v28 // (op 1289) least significant half
v_addc_co_u32 v37, s[54:55], 0, v37, s[54:55] // (op 1289) most significant half

s_add_u32 m0, m0, 1024 // (op 1289) 
buffer_load_dwordx4 v36, s[60:63], 0 offen offset: 0 lds // (op 1289) Load value direct to lds






v_mov_b64 v[36:37], s[32:33] // (op 994) convert

s_lshl_b32 s52, s32, 3 // (op 994) 
v_add_u32 v23, v25, s52 // (op 994) 

buffer_load_ubyte v28, v23, s[68:71], 0 offen offset: 0 // (op 994) Load value
v_add_u32 v23, v23, v36 // (op 994) 
buffer_load_ubyte v29, v23, s[68:71], 0 offen offset: 0 // (op 994) Load value



v_and_b32 v23, -16, v1 // (op 2093) 
v_sub_u32 v25, v1, v23 // (op 2093) 

v_lshrrev_b32 v23, 4, v1 // (op 2093) 
v_lshl_add_u32 v39, v23, 5, v25 // (op 2093) 


s_barrier  // (op 2910) 



s_waitcnt vmcnt(19)
ds_write_b8 v39, v42 offset:16384 // (op 997) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v39, v43 offset:16400 // (op 997) Store local data 

v_and_b32 v23, -16, v1 // (op 2027) 
v_sub_u32 v25, v1, v23 // (op 2027) 

v_lshrrev_b32 v23, 4, v1 // (op 2027) 
v_lshl_add_u32 v39, v23, 5, v25 // (op 2027) 


s_barrier  // (op 2890) 



s_waitcnt vmcnt(13)
ds_write_b8 v39, v44 offset:16512 // (op 999) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v39, v45 offset:16528 // (op 999) Store local data 

s_barrier  // (op 2867) 



ds_read_b64_tr_b8 v[46:47], v22 // (op 1019) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v22 offset:128 // (op 1019) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v22 offset:1024 // (op 1019) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v22 offset:1152 // (op 1019) Transpose load local data 



ds_read_b64_tr_b8 v[54:55], v22 offset:2048 // (op 1022) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v22 offset:2176 // (op 1022) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v22 offset:3072 // (op 1022) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v22 offset:3200 // (op 1022) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 2907) 


ds_read_u8 v23, v26 offset:16384 // (op 1031) Load local data 


ds_read_u8 v25, v26 offset:16448 // (op 1034) Load local data 
s_barrier  // (op 2848) 



ds_read_b64_tr_b8 v[62:63], v20 offset:4096 // (op 1013) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v20 offset:4224 // (op 1013) Transpose load local data 
ds_read_b64_tr_b8 v[66:67], v20 offset:5120 // (op 1013) Transpose load local data 
ds_read_b64_tr_b8 v[68:69], v20 offset:5248 // (op 1013) Transpose load local data 



ds_read_b64_tr_b8 v[70:71], v20 offset:6144 // (op 1016) Transpose load local data 
ds_read_b64_tr_b8 v[72:73], v20 offset:6272 // (op 1016) Transpose load local data 
ds_read_b64_tr_b8 v[74:75], v20 offset:7168 // (op 1016) Transpose load local data 
ds_read_b64_tr_b8 v[76:77], v20 offset:7296 // (op 1016) Transpose load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 2887) 


ds_read_u8 v39, v24 offset:16512 // (op 1025) Load local data 


ds_read_u8 v30, v24 offset:16576 // (op 1028) Load local data 
v_lshlrev_b32 v31, 4, v1 // (op 1868) 
s_nop 0
v_readfirstlane_b32 s52, v31 // (op 1868) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v31, 4, v1 // (op 1933) 
s_nop 0
v_readfirstlane_b32 s53, v31 // (op 1933) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v31, -16, v1 // (op 1998) 
v_sub_u32 v36, v1, v31 // (op 1998) 

v_lshrrev_b32 v31, 4, v1 // (op 1998) 
v_lshl_add_u32 v37, v31, 5, v36 // (op 1998) 


v_and_b32 v31, -16, v1 // (op 2064) 
v_sub_u32 v36, v1, v31 // (op 2064) 

v_lshrrev_b32 v31, 4, v1 // (op 2064) 
v_lshl_add_u32 v78, v31, 5, v36 // (op 2064) 


v_lshlrev_b32 v31, 4, v1 // (op 2622) 
s_nop 0
v_readfirstlane_b32 s54, v31 // (op 2622) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v31, 4, v1 // (op 2661) 
s_nop 0
v_readfirstlane_b32 s55, v31 // (op 2661) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v31, -16, v1 // (op 2700) 
v_sub_u32 v36, v1, v31 // (op 2700) 

v_lshrrev_b32 v31, 4, v1 // (op 2700) 
v_lshl_add_u32 v79, v31, 5, v36 // (op 2700) 


v_and_b32 v31, -16, v1 // (op 2724) 
v_sub_u32 v36, v1, v31 // (op 2724) 

v_lshrrev_b32 v31, 4, v1 // (op 2724) 
v_lshl_add_u32 v80, v31, 5, v36 // (op 2724) 


s_mov_b32 s1, 0 // (op 61) call()
s_and_b32 s76, s7, 4294967294 // (op 60) 
s_cmp_lt_u32 s1, s76 // (op 60) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopBottom_KLoop_60 // (op 60) Condition: Top (jump to RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopBottom_KLoop_60 if false)
RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopTop_KLoop_60:
s_add_u32 s1, s1, 2 // (op 62) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:69], v[46:53], a[0:3], v39, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 90) 
s_barrier  // (op 2873) 

s_lshl_b64 s[76:77], s[28:29], 6 // (op 1216) 
v_mov_b64 v[82:83], s[76:77] // (op 1216) convert


v_mov_b64 v[84:85], v[34:35] // (op 1216) 
s_lshl_b64 s[76:77], s[28:29], 8 // (op 1216) 
s_lshl_b64 s[78:79], s[76:77], 1 // (op 1216) 

v_mov_b32 v31, s78 // (op 1216) 
v_mov_b32 v36, s79 // (op 1216) 
v_add_co_u32 v86, s[76:77], v84, v31 // (op 1216) least significant half
v_addc_co_u32 v87, s[76:77], v85, v36, s[76:77] // (op 1216) most significant half




s_mov_b32 m0, s53 // (op 1216) reg expression
buffer_load_dwordx4 v86, s[64:67], 0 offen offset: 0 lds // (op 1216) Load value direct to lds
v_add_co_u32 v86, s[76:77], v86, v82 // (op 1216) least significant half
v_addc_co_u32 v87, s[76:77], 0, v87, s[76:77] // (op 1216) most significant half

s_add_u32 m0, m0, 1024 // (op 1216) 
buffer_load_dwordx4 v86, s[64:67], 0 offen offset: 0 lds // (op 1216) Load value direct to lds
v_add_co_u32 v86, s[76:77], v86, v82 // (op 1216) least significant half
v_addc_co_u32 v87, s[76:77], 0, v87, s[76:77] // (op 1216) most significant half

s_add_u32 m0, m0, 1024 // (op 1216) 
buffer_load_dwordx4 v86, s[64:67], 0 offen offset: 0 lds // (op 1216) Load value direct to lds
v_add_co_u32 v86, s[76:77], v86, v82 // (op 1216) least significant half
v_addc_co_u32 v87, s[76:77], 0, v87, s[76:77] // (op 1216) most significant half

s_add_u32 m0, m0, 1024 // (op 1216) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[70:77], v[54:61], a[0:3], v30, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 107) 
buffer_load_dwordx4 v86, s[64:67], 0 offen offset: 0 lds // (op 1216) Load value direct to lds



v_mov_b64 v[82:83], s[38:39] // (op 25) convert

v_mov_b32 v31, v38 // (op 25) 
s_lshl_b32 s76, s38, 4 // (op 25) 
v_add_u32 v36, v31, s76 // (op 25) 

buffer_load_ubyte v42, v36, s[72:75], 0 offen offset: 0 // (op 25) Load value
v_add_u32 v36, v36, v82 // (op 25) 
buffer_load_ubyte v43, v36, s[72:75], 0 offen offset: 0 // (op 25) Load value


s_barrier  // (op 2854) 

s_lshl_b64 s[76:77], s[24:25], 6 // (op 1211) 
v_mov_b64 v[82:83], s[76:77] // (op 1211) convert


v_mov_b64 v[86:87], v[32:33] // (op 1211) 
s_lshl_b64 s[76:77], s[24:25], 8 // (op 1211) 
s_lshl_b64 s[78:79], s[76:77], 1 // (op 1211) 

v_mov_b32 v36, s78 // (op 1211) 
v_mov_b32 v81, s79 // (op 1211) 
v_add_co_u32 v88, s[76:77], v86, v36 // (op 1211) least significant half
v_addc_co_u32 v89, s[76:77], v87, v81, s[76:77] // (op 1211) most significant half




s_add_u32 s76, 4096, s52 // (op 1211) 
s_nop 0
s_mov_b32 m0, s76 // (op 1211) reg expression
buffer_load_dwordx4 v88, s[60:63], 0 offen offset: 0 lds // (op 1211) Load value direct to lds
v_add_co_u32 v88, s[78:79], v88, v82 // (op 1211) least significant half
v_addc_co_u32 v89, s[78:79], 0, v89, s[78:79] // (op 1211) most significant half

s_add_u32 m0, m0, 1024 // (op 1211) 
buffer_load_dwordx4 v88, s[60:63], 0 offen offset: 0 lds // (op 1211) Load value direct to lds
v_add_co_u32 v88, s[78:79], v88, v82 // (op 1211) least significant half
v_addc_co_u32 v89, s[78:79], 0, v89, s[78:79] // (op 1211) most significant half

s_add_u32 m0, m0, 1024 // (op 1211) 
buffer_load_dwordx4 v88, s[60:63], 0 offen offset: 0 lds // (op 1211) Load value direct to lds
v_add_co_u32 v88, s[78:79], v88, v82 // (op 1211) least significant half
v_addc_co_u32 v89, s[78:79], 0, v89, s[78:79] // (op 1211) most significant half

s_add_u32 m0, m0, 1024 // (op 1211) 
buffer_load_dwordx4 v88, s[60:63], 0 offen offset: 0 lds // (op 1211) Load value direct to lds




v_mov_b64 v[82:83], s[32:33] // (op 17) convert

v_mov_b32 v36, v27 // (op 17) 
s_lshl_b32 s76, s32, 4 // (op 17) 
v_add_u32 v81, v36, s76 // (op 17) 

buffer_load_ubyte v44, v81, s[68:71], 0 offen offset: 0 // (op 17) Load value
v_add_u32 v81, v81, v82 // (op 17) 
buffer_load_ubyte v45, v81, s[68:71], 0 offen offset: 0 // (op 17) Load value


s_barrier  // (op 3065) 



s_waitcnt vmcnt(19)
ds_write_b8 v80, v40 offset:16640 // (op 813) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v80, v41 offset:16656 // (op 813) Store local data 
s_barrier  // (op 3059) 



s_waitcnt vmcnt(13)
ds_write_b8 v79, v28 offset:16768 // (op 812) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v79, v29 offset:16784 // (op 812) Store local data 
s_barrier  // (op 3046) 



ds_read_b64_tr_b8 v[46:47], v14 offset:8192 // (op 796) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v14 offset:8320 // (op 796) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v14 offset:9216 // (op 796) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v14 offset:9344 // (op 796) Transpose load local data 



ds_read_b64_tr_b8 v[54:55], v14 offset:10240 // (op 805) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v14 offset:10368 // (op 805) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v14 offset:11264 // (op 805) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v14 offset:11392 // (op 805) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 3062) 


ds_read_u8 v23, v19 offset:16640 // (op 800) Load local data 


ds_read_u8 v25, v19 offset:16704 // (op 809) Load local data 
s_barrier  // (op 3037) 
s_barrier  // (op 3220) 



ds_read_b64_tr_b8 v[62:63], v12 offset:12288 // (op 794) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v12 offset:12416 // (op 794) Transpose load local data 
ds_read_b64_tr_b8 v[66:67], v12 offset:13312 // (op 794) Transpose load local data 
ds_read_b64_tr_b8 v[68:69], v12 offset:13440 // (op 794) Transpose load local data 



ds_read_b64_tr_b8 v[70:71], v12 offset:14336 // (op 803) Transpose load local data 
ds_read_b64_tr_b8 v[72:73], v12 offset:14464 // (op 803) Transpose load local data 
ds_read_b64_tr_b8 v[74:75], v12 offset:15360 // (op 803) Transpose load local data 
ds_read_b64_tr_b8 v[76:77], v12 offset:15488 // (op 803) Transpose load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 3056) 


ds_read_u8 v39, v17 offset:16768 // (op 798) Load local data 


ds_read_u8 v30, v17 offset:16832 // (op 807) Load local data 
s_barrier  // (op 3216) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:69], v[46:53], a[0:3], v39, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 801) 
s_barrier  // (op 3053) 

s_lshl_b64 s[76:77], s[28:29], 6 // (op 1272) 
v_mov_b64 v[82:83], s[76:77] // (op 1272) convert


s_lshl_b64 s[76:77], s[28:29], 8 // (op 1272) 
s_mul_i32 s79, 3, s77 // (op 1272) most significant: low of low * high
s_mul_i32 s78, 3, s76 // (op 1272) least significant
s_mul_hi_u32 s80, 3, s76 // (op 1272) most significant: high of low * low
s_add_u32 s79, s79, s80 // (op 1272) most significant: sum


v_mov_b32 v81, s78 // (op 1272) 
v_mov_b32 v88, s79 // (op 1272) 
v_add_co_u32 v90, s[76:77], v84, v81 // (op 1272) least significant half
v_addc_co_u32 v91, s[76:77], v85, v88, s[76:77] // (op 1272) most significant half




s_add_u32 s76, 8192, s55 // (op 1272) 
s_nop 0
s_mov_b32 m0, s76 // (op 1272) reg expression
buffer_load_dwordx4 v90, s[64:67], 0 offen offset: 0 lds // (op 1272) Load value direct to lds
v_add_co_u32 v90, s[78:79], v90, v82 // (op 1272) least significant half
v_addc_co_u32 v91, s[78:79], 0, v91, s[78:79] // (op 1272) most significant half

s_add_u32 m0, m0, 1024 // (op 1272) 
buffer_load_dwordx4 v90, s[64:67], 0 offen offset: 0 lds // (op 1272) Load value direct to lds
v_add_co_u32 v90, s[78:79], v90, v82 // (op 1272) least significant half
v_addc_co_u32 v91, s[78:79], 0, v91, s[78:79] // (op 1272) most significant half

s_add_u32 m0, m0, 1024 // (op 1272) 
buffer_load_dwordx4 v90, s[64:67], 0 offen offset: 0 lds // (op 1272) Load value direct to lds
v_add_co_u32 v90, s[78:79], v90, v82 // (op 1272) least significant half
v_addc_co_u32 v91, s[78:79], 0, v91, s[78:79] // (op 1272) most significant half

s_add_u32 m0, m0, 1024 // (op 1272) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[70:77], v[54:61], a[0:3], v30, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 810) 
buffer_load_dwordx4 v90, s[64:67], 0 offen offset: 0 lds // (op 1272) Load value direct to lds





v_mov_b64 v[84:85], s[38:39] // (op 816) convert

s_lshl_b32 s76, s38, 3 // (op 816) 
s_mul_i32 s77, 3, s76 // (op 816) 

v_add_u32 v81, v31, s77 // (op 816) 

buffer_load_ubyte v40, v81, s[72:75], 0 offen offset: 0 // (op 816) Load value
v_add_u32 v81, v81, v84 // (op 816) 
s_lshl_b64 s[76:77], s[28:29], 8 // (op 1508) 
buffer_load_ubyte v41, v81, s[72:75], 0 offen offset: 0 // (op 816) Load value


s_lshl_b64 s[78:79], s[76:77], 1 // (op 1508) 


v_mov_b32 v31, s78 // (op 1508) 
s_barrier  // (op 3043) 
s_lshl_b32 s76, s38, 4 // (op 1764) 

v_add_u32 v38, v38, s76 // (op 1764) 

v_mov_b32 v81, s79 // (op 1508) 
s_lshl_b64 s[76:77], s[24:25], 6 // (op 1267) 
v_mov_b64 v[82:83], s[76:77] // (op 1267) convert


s_lshl_b64 s[76:77], s[24:25], 8 // (op 1267) 
s_mul_i32 s81, 3, s77 // (op 1267) most significant: low of low * high
s_mul_i32 s80, 3, s76 // (op 1267) least significant
s_mul_hi_u32 s82, 3, s76 // (op 1267) most significant: high of low * low
s_add_u32 s81, s81, s82 // (op 1267) most significant: sum


v_mov_b32 v84, s80 // (op 1267) 
v_mov_b32 v85, s81 // (op 1267) 
v_add_co_u32 v34, s[76:77], v34, v31 // (op 1508) least significant half
v_addc_co_u32 v35, s[76:77], v35, v81, s[76:77] // (op 1508) most significant half




v_add_co_u32 v88, s[76:77], v86, v84 // (op 1267) least significant half
v_addc_co_u32 v89, s[76:77], v87, v85, s[76:77] // (op 1267) most significant half




s_add_u32 s76, 12288, s54 // (op 1267) 
s_nop 0
s_mov_b32 m0, s76 // (op 1267) reg expression
buffer_load_dwordx4 v88, s[60:63], 0 offen offset: 0 lds // (op 1267) Load value direct to lds
v_add_co_u32 v88, s[78:79], v88, v82 // (op 1267) least significant half
v_addc_co_u32 v89, s[78:79], 0, v89, s[78:79] // (op 1267) most significant half

s_add_u32 m0, m0, 1024 // (op 1267) 
buffer_load_dwordx4 v88, s[60:63], 0 offen offset: 0 lds // (op 1267) Load value direct to lds
v_add_co_u32 v88, s[78:79], v88, v82 // (op 1267) least significant half
v_addc_co_u32 v89, s[78:79], 0, v89, s[78:79] // (op 1267) most significant half

s_add_u32 m0, m0, 1024 // (op 1267) 
buffer_load_dwordx4 v88, s[60:63], 0 offen offset: 0 lds // (op 1267) Load value direct to lds
v_add_co_u32 v88, s[78:79], v88, v82 // (op 1267) least significant half
v_addc_co_u32 v89, s[78:79], 0, v89, s[78:79] // (op 1267) most significant half

s_add_u32 m0, m0, 1024 // (op 1267) 
buffer_load_dwordx4 v88, s[60:63], 0 offen offset: 0 lds // (op 1267) Load value direct to lds





v_mov_b64 v[82:83], s[32:33] // (op 815) convert

s_lshl_b32 s76, s32, 3 // (op 815) 
s_mul_i32 s77, 3, s76 // (op 815) 

v_add_u32 v31, v36, s77 // (op 815) 

buffer_load_ubyte v28, v31, s[68:71], 0 offen offset: 0 // (op 815) Load value
v_add_u32 v31, v31, v82 // (op 815) 
s_lshl_b64 s[76:77], s[24:25], 8 // (op 1377) 
s_lshl_b64 s[78:79], s[76:77], 1 // (op 1377) 

buffer_load_ubyte v29, v31, s[68:71], 0 offen offset: 0 // (op 815) Load value


v_mov_b32 v31, s78 // (op 1377) 

v_mov_b32 v36, s79 // (op 1377) 
s_barrier  // (op 2913) 
s_lshl_b32 s76, s32, 4 // (op 1636) 



v_add_u32 v27, v27, s76 // (op 1636) 

v_add_co_u32 v32, s[76:77], v32, v31 // (op 1377) least significant half
v_addc_co_u32 v33, s[76:77], v33, v36, s[76:77] // (op 1377) most significant half




s_waitcnt vmcnt(19)
ds_write_b8 v78, v42 offset:16384 // (op 30) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v78, v43 offset:16400 // (op 30) Store local data 
s_barrier  // (op 2893) 



s_waitcnt vmcnt(13)
ds_write_b8 v37, v44 offset:16512 // (op 21) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v37, v45 offset:16528 // (op 21) Store local data 
s_barrier  // (op 2879) 



ds_read_b64_tr_b8 v[46:47], v3 // (op 77) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v3 offset:128 // (op 77) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v3 offset:1024 // (op 77) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v3 offset:1152 // (op 77) Transpose load local data 



ds_read_b64_tr_b8 v[54:55], v3 offset:2048 // (op 97) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v3 offset:2176 // (op 97) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v3 offset:3072 // (op 97) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v3 offset:3200 // (op 97) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 2920) 


ds_read_u8 v23, v7 offset:16384 // (op 87) Load local data 


ds_read_u8 v25, v7 offset:16448 // (op 105) Load local data 
s_barrier  // (op 2860) 



ds_read_b64_tr_b8 v[62:63], v0 offset:4096 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v0 offset:4224 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[66:67], v0 offset:5120 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[68:69], v0 offset:5248 // (op 72) Transpose load local data 



ds_read_b64_tr_b8 v[70:71], v0 offset:6144 // (op 93) Transpose load local data 
ds_read_b64_tr_b8 v[72:73], v0 offset:6272 // (op 93) Transpose load local data 
ds_read_b64_tr_b8 v[74:75], v0 offset:7168 // (op 93) Transpose load local data 
ds_read_b64_tr_b8 v[76:77], v0 offset:7296 // (op 93) Transpose load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 2900) 


ds_read_u8 v39, v5 offset:16512 // (op 82) Load local data 


ds_read_u8 v30, v5 offset:16576 // (op 101) Load local data 
s_and_b32 s76, s7, 4294967294 // (op 60) 
s_cmp_lt_u32 s1, s76 // (op 60) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopTop_KLoop_60 // (op 60) Condition: Bottom (jump to RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopTop_KLoop_60 if true)
RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopBottom_KLoop_60:















s_barrier  // (op 3078) 
s_and_b32 s52, s7, 4294967294 // (op 1340) 
s_add_u32 s53, s0, s52 // (op 1340) 

s_mul_i32 s52, s16, s8 // (op 1340) 
s_add_u32 s52, s52, s53 // (op 1340) 

s_mul_hi_u32 s53, s52, s13 // (op 1340) 
s_sub_u32 s54, s52, s53 // (op 1340) 
s_lshr_b32 s55, s54, 1 // (op 1340) 

s_add_u32 s54, s55, s53 // (op 1340) 


s_lshr_b32 s53, s54, s11 // (op 1340) 

s_and_b32 s54, s11, 2147483648 // (op 1340) 
s_cmp_eq_i32 s54, 0 // (op 1340) 
s_mov_b32 s55, scc // (op 1340) 

s_cmp_eq_i32 s55, 0 // (op 1340) 
s_cselect_b32 s54, s52, s53 // (op 1340) 


s_mul_hi_u32 s53, s54, s12 // (op 1340) 
s_sub_u32 s55, s54, s53 // (op 1340) 
s_lshr_b32 s76, s55, 1 // (op 1340) 

s_add_u32 s55, s76, s53 // (op 1340) 


s_lshr_b32 s53, s55, s10 // (op 1340) 

s_and_b32 s55, s10, 2147483648 // (op 1340) 
s_cmp_eq_i32 s55, 0 // (op 1340) 
s_mov_b32 s76, scc // (op 1340) 

s_cmp_eq_i32 s76, 0 // (op 1340) 
s_cselect_b32 s55, s54, s53 // (op 1340) 


s_lshl_b32 s53, s55, 4 // (op 1340) 

s_mov_b32 s77, 0 // (op 1340) convert
s_mov_b32 s76, s53 // (op 1340) convert

s_ashr_i64 s[78:79], s[2:3], 8 // (op 1340) 
s_mul_i32 s53, s54, s78 // (op 1340) 


s_sub_u32 s54, s52, s53 // (op 1340) 


v_lshl_add_u32 v27, s54, 8, v1 // (op 1340) 

v_mul_lo_u32 v43, v27, s25 // (op 1340) most significant: low of low * high
v_mul_hi_u32 v31, v27, s24 // (op 1340) most significant: high of low * low
v_mul_lo_u32 v42, v27, s24 // (op 1340) least significant: low of low * low
v_add_u32 v43, v43, v31 // (op 1340) most significant: sum


v_mov_b32 v27, s76 // (op 1340) 
v_mov_b32 v31, s77 // (op 1340) 
v_add_co_u32 v32, s[52:53], v27, v42 // (op 1340) least significant half
v_addc_co_u32 v33, s[52:53], v31, v43, s[52:53] // (op 1340) most significant half





s_mul_i32 s53, s24, s3 // (op 1342) most significant: low of low * high
s_mul_i32 s54, s25, s2 // (op 1342) most significant: low of high * low
s_mul_i32 s52, s24, s2 // (op 1342) least significant
s_mul_hi_u32 s55, s24, s2 // (op 1342) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1342) most significant: sum
s_add_u32 s53, s53, s55 // (op 1342) most significant: sum


s_mov_b32 s60, s26 // (op 1342) reg expression
s_mov_b32 s61, s27 // (op 1342) reg expression
s_mov_b32 s62, s52 // (op 1342) reg expression
s_mov_b32 s63, 131072 // (op 1342) call()

s_and_b32 s52, s7, 4294967294 // (op 1474) 
s_add_u32 s53, s0, s52 // (op 1474) 

s_mul_i32 s52, s16, s8 // (op 1474) 
s_add_u32 s52, s52, s53 // (op 1474) 

s_mul_hi_u32 s53, s52, s13 // (op 1474) 
s_sub_u32 s54, s52, s53 // (op 1474) 
s_lshr_b32 s55, s54, 1 // (op 1474) 

s_add_u32 s54, s55, s53 // (op 1474) 


s_lshr_b32 s53, s54, s11 // (op 1474) 

s_and_b32 s54, s11, 2147483648 // (op 1474) 
s_cmp_eq_i32 s54, 0 // (op 1474) 
s_mov_b32 s55, scc // (op 1474) 

s_cmp_eq_i32 s55, 0 // (op 1474) 
s_cselect_b32 s54, s52, s53 // (op 1474) 


s_mul_hi_u32 s53, s54, s12 // (op 1474) 
s_sub_u32 s55, s54, s53 // (op 1474) 
s_lshr_b32 s76, s55, 1 // (op 1474) 

s_add_u32 s55, s76, s53 // (op 1474) 


s_lshr_b32 s53, s55, s10 // (op 1474) 

s_and_b32 s55, s10, 2147483648 // (op 1474) 
s_cmp_eq_i32 s55, 0 // (op 1474) 
s_mov_b32 s76, scc // (op 1474) 

s_cmp_eq_i32 s76, 0 // (op 1474) 
s_cselect_b32 s55, s54, s53 // (op 1474) 


s_mul_i32 s53, s55, s14 // (op 1474) 

s_sub_u32 s55, s54, s53 // (op 1474) 

s_lshl_b32 s53, s55, 4 // (op 1474) 

s_mov_b32 s77, 0 // (op 1474) convert
s_mov_b32 s76, s53 // (op 1474) convert

s_ashr_i64 s[78:79], s[2:3], 8 // (op 1474) 
s_mul_i32 s53, s54, s78 // (op 1474) 


s_sub_u32 s54, s52, s53 // (op 1474) 


v_lshl_add_u32 v27, s54, 8, v1 // (op 1474) 

v_mul_lo_u32 v35, v27, s29 // (op 1474) most significant: low of low * high
v_mul_hi_u32 v31, v27, s28 // (op 1474) most significant: high of low * low
v_mul_lo_u32 v34, v27, s28 // (op 1474) least significant: low of low * low
v_add_u32 v35, v35, v31 // (op 1474) most significant: sum

v_mov_b32 v31, s76 // (op 1474) 
v_mov_b32 v36, s77 // (op 1474) 
v_add_co_u32 v34, s[52:53], v34, v31 // (op 1474) least significant half
v_addc_co_u32 v35, s[52:53], v35, v36, s[52:53] // (op 1474) most significant half





s_mul_i32 s53, s28, s3 // (op 1476) most significant: low of low * high
s_mul_i32 s54, s29, s2 // (op 1476) most significant: low of high * low
s_mul_i32 s52, s28, s2 // (op 1476) least significant
s_mul_hi_u32 s55, s28, s2 // (op 1476) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1476) most significant: sum
s_add_u32 s53, s53, s55 // (op 1476) most significant: sum


s_mov_b32 s64, s30 // (op 1476) reg expression
s_mov_b32 s65, s31 // (op 1476) reg expression
s_mov_b32 s66, s52 // (op 1476) reg expression
s_mov_b32 s67, 131072 // (op 1476) call()

s_and_b32 s52, s7, 4294967294 // (op 1602) 
v_and_b32 v27, -16, v1 // (op 1602) 
s_add_u32 s53, s0, s52 // (op 1602) 

v_sub_u32 v31, v1, v27 // (op 1602) 

v_lshrrev_b32 v27, 4, v1 // (op 1602) 
s_mul_i32 s52, s16, s8 // (op 1602) 
s_add_u32 s52, s52, s53 // (op 1602) 

v_lshlrev_b32 v36, 1, v27 // (op 1602) 

s_mul_hi_u32 s53, s52, s13 // (op 1602) 
s_sub_u32 s54, s52, s53 // (op 1602) 
s_lshr_b32 s55, s54, 1 // (op 1602) 

s_add_u32 s54, s55, s53 // (op 1602) 


s_lshr_b32 s53, s54, s11 // (op 1602) 

s_and_b32 s54, s11, 2147483648 // (op 1602) 
s_cmp_eq_i32 s54, 0 // (op 1602) 
s_mov_b32 s55, scc // (op 1602) 

s_cmp_eq_i32 s55, 0 // (op 1602) 
s_cselect_b32 s54, s52, s53 // (op 1602) 


s_mul_hi_u32 s53, s54, s12 // (op 1602) 
s_sub_u32 s55, s54, s53 // (op 1602) 
s_lshr_b32 s76, s55, 1 // (op 1602) 

s_add_u32 s55, s76, s53 // (op 1602) 


s_lshr_b32 s53, s55, s10 // (op 1602) 

s_and_b32 s55, s10, 2147483648 // (op 1602) 
s_cmp_eq_i32 s55, 0 // (op 1602) 
s_mov_b32 s76, scc // (op 1602) 

s_cmp_eq_i32 s76, 0 // (op 1602) 
s_cselect_b32 s55, s54, s53 // (op 1602) 


v_lshl_add_u32 v27, s55, 4, v31 // (op 1602) 


s_ashr_i64 s[76:77], s[2:3], 8 // (op 1602) 
s_mul_i32 s53, s54, s76 // (op 1602) 


s_sub_u32 s54, s52, s53 // (op 1602) 


v_lshl_add_u32 v31, s54, 3, v36 // (op 1602) 


v_mul_lo_u32 v36, v31, s32 // (op 1602) 

v_add_u32 v31, v27, v36 // (op 1602) 


s_mul_i32 s53, s32, s37 // (op 1604) most significant: low of low * high
s_mul_i32 s54, s33, s36 // (op 1604) most significant: low of high * low
s_mul_i32 s52, s32, s36 // (op 1604) least significant
s_mul_hi_u32 s55, s32, s36 // (op 1604) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1604) most significant: sum
s_add_u32 s53, s53, s55 // (op 1604) most significant: sum


s_mov_b32 s68, s34 // (op 1604) reg expression
s_mov_b32 s69, s35 // (op 1604) reg expression
s_mov_b32 s70, s52 // (op 1604) reg expression
s_mov_b32 s71, 131072 // (op 1604) call()

s_and_b32 s52, s7, 4294967294 // (op 1730) 
v_and_b32 v27, -16, v1 // (op 1730) 
s_add_u32 s53, s0, s52 // (op 1730) 

v_sub_u32 v36, v1, v27 // (op 1730) 

v_lshrrev_b32 v27, 4, v1 // (op 1730) 
s_mul_i32 s52, s16, s8 // (op 1730) 
s_add_u32 s52, s52, s53 // (op 1730) 

v_lshlrev_b32 v37, 1, v27 // (op 1730) 

s_mul_hi_u32 s53, s52, s13 // (op 1730) 
s_sub_u32 s54, s52, s53 // (op 1730) 
s_lshr_b32 s55, s54, 1 // (op 1730) 

s_add_u32 s54, s55, s53 // (op 1730) 


s_lshr_b32 s53, s54, s11 // (op 1730) 

s_and_b32 s54, s11, 2147483648 // (op 1730) 
s_cmp_eq_i32 s54, 0 // (op 1730) 
s_mov_b32 s55, scc // (op 1730) 

s_cmp_eq_i32 s55, 0 // (op 1730) 
s_cselect_b32 s54, s52, s53 // (op 1730) 


s_mul_hi_u32 s53, s54, s12 // (op 1730) 
s_sub_u32 s55, s54, s53 // (op 1730) 
s_lshr_b32 s76, s55, 1 // (op 1730) 

s_add_u32 s55, s76, s53 // (op 1730) 


s_lshr_b32 s53, s55, s10 // (op 1730) 

s_and_b32 s55, s10, 2147483648 // (op 1730) 
s_cmp_eq_i32 s55, 0 // (op 1730) 
s_mov_b32 s76, scc // (op 1730) 

s_cmp_eq_i32 s76, 0 // (op 1730) 
s_cselect_b32 s55, s54, s53 // (op 1730) 


s_mul_i32 s53, s55, s14 // (op 1730) 

s_sub_u32 s55, s54, s53 // (op 1730) 

v_lshl_add_u32 v27, s55, 4, v36 // (op 1730) 


s_ashr_i64 s[76:77], s[2:3], 8 // (op 1730) 
s_mul_i32 s53, s54, s76 // (op 1730) 


s_sub_u32 s54, s52, s53 // (op 1730) 


v_lshl_add_u32 v36, s54, 3, v37 // (op 1730) 


v_mul_lo_u32 v37, v36, s38 // (op 1730) 
v_add_u32 v37, v37, v27 // (op 1730) 


s_mul_i32 s53, s38, s37 // (op 1732) most significant: low of low * high
s_mul_i32 s54, s39, s36 // (op 1732) most significant: low of high * low
s_mul_i32 s52, s38, s36 // (op 1732) least significant
s_mul_hi_u32 s55, s38, s36 // (op 1732) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1732) most significant: sum
s_add_u32 s53, s53, s55 // (op 1732) most significant: sum


s_mov_b32 s72, s40 // (op 1732) reg expression
s_mov_b32 s73, s41 // (op 1732) reg expression
s_mov_b32 s74, s52 // (op 1732) reg expression
s_mov_b32 s75, 131072 // (op 1732) call()

v_lshlrev_b32 v27, 4, v1 // (op 1881) 
s_nop 0
v_readfirstlane_b32 s52, v27 // (op 1881) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v27, 4, v1 // (op 1946) 
s_nop 0
v_readfirstlane_b32 s53, v27 // (op 1946) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v27, -16, v1 // (op 2011) 
v_sub_u32 v36, v1, v27 // (op 2011) 

v_lshrrev_b32 v27, 4, v1 // (op 2011) 
v_lshl_add_u32 v38, v27, 5, v36 // (op 2011) 


v_and_b32 v27, -16, v1 // (op 2077) 
v_sub_u32 v36, v1, v27 // (op 2077) 

v_lshrrev_b32 v27, 4, v1 // (op 2077) 
v_lshl_add_u32 v42, v27, 5, v36 // (op 2077) 


s_and_b32 s54, s7, 4294967294 // (op 703) 
s_cmp_lt_u32 s54, s7 // (op 702) 









s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopBottom_KLoopTail_702 // (op 702) Condition: Top (jump to RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopBottom_KLoopTail_702 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopTop_KLoopTail_702:
s_barrier  // (op 2857) 
s_add_u32 s54, s54, 1 // (op 704) 

s_lshl_b64 s[76:77], s[24:25], 6 // (op 1254) 
v_mov_b64 v[40:41], s[76:77] // (op 1254) convert


v_mov_b64 v[28:29], v[32:33] // (op 1254) 
v_mov_b64 v[44:45], v[28:29] // (op 1254) reg expression
s_add_u32 s55, 4096, s52 // (op 1254) 
s_mov_b32 m0, s55 // (op 1254) reg expression
buffer_load_dwordx4 v44, s[60:63], 0 offen offset: 0 lds // (op 1254) Load value direct to lds
v_add_co_u32 v44, s[76:77], v44, v40 // (op 1254) least significant half
v_addc_co_u32 v45, s[76:77], 0, v45, s[76:77] // (op 1254) most significant half

s_add_u32 m0, m0, 1024 // (op 1254) 
buffer_load_dwordx4 v44, s[60:63], 0 offen offset: 0 lds // (op 1254) Load value direct to lds
v_add_co_u32 v44, s[76:77], v44, v40 // (op 1254) least significant half
v_addc_co_u32 v45, s[76:77], 0, v45, s[76:77] // (op 1254) most significant half

s_add_u32 m0, m0, 1024 // (op 1254) 
buffer_load_dwordx4 v44, s[60:63], 0 offen offset: 0 lds // (op 1254) Load value direct to lds
v_add_co_u32 v44, s[76:77], v44, v40 // (op 1254) least significant half
v_addc_co_u32 v45, s[76:77], 0, v45, s[76:77] // (op 1254) most significant half

s_add_u32 m0, m0, 1024 // (op 1254) 
buffer_load_dwordx4 v44, s[60:63], 0 offen offset: 0 lds // (op 1254) Load value direct to lds




s_lshl_b64 s[76:77], s[24:25], 8 // (op 1330) 
v_mov_b32 v23, s76 // (op 1330) 
v_mov_b32 v25, s77 // (op 1330) 
v_add_co_u32 v32, s[78:79], v32, v23 // (op 1330) least significant half
v_addc_co_u32 v33, s[78:79], v33, v25, s[78:79] // (op 1330) most significant half




s_barrier  // (op 2876) 

s_lshl_b64 s[76:77], s[28:29], 6 // (op 1260) 
v_mov_b64 v[40:41], s[76:77] // (op 1260) convert


v_mov_b64 v[28:29], v[34:35] // (op 1260) 
v_mov_b64 v[44:45], v[28:29] // (op 1260) reg expression
s_mov_b32 m0, s53 // (op 1260) reg expression
buffer_load_dwordx4 v44, s[64:67], 0 offen offset: 0 lds // (op 1260) Load value direct to lds
v_add_co_u32 v44, s[76:77], v44, v40 // (op 1260) least significant half
v_addc_co_u32 v45, s[76:77], 0, v45, s[76:77] // (op 1260) most significant half

s_add_u32 m0, m0, 1024 // (op 1260) 
buffer_load_dwordx4 v44, s[64:67], 0 offen offset: 0 lds // (op 1260) Load value direct to lds
v_add_co_u32 v44, s[76:77], v44, v40 // (op 1260) least significant half
v_addc_co_u32 v45, s[76:77], 0, v45, s[76:77] // (op 1260) most significant half

s_add_u32 m0, m0, 1024 // (op 1260) 
buffer_load_dwordx4 v44, s[64:67], 0 offen offset: 0 lds // (op 1260) Load value direct to lds
v_add_co_u32 v44, s[76:77], v44, v40 // (op 1260) least significant half
v_addc_co_u32 v45, s[76:77], 0, v45, s[76:77] // (op 1260) most significant half

s_add_u32 m0, m0, 1024 // (op 1260) 
buffer_load_dwordx4 v44, s[64:67], 0 offen offset: 0 lds // (op 1260) Load value direct to lds



s_lshl_b64 s[76:77], s[28:29], 8 // (op 1466) 
v_mov_b32 v23, s76 // (op 1466) 
v_mov_b32 v25, s77 // (op 1466) 
v_add_co_u32 v34, s[78:79], v34, v23 // (op 1466) least significant half
v_addc_co_u32 v35, s[78:79], v35, v25, s[78:79] // (op 1466) most significant half





v_mov_b64 v[40:41], s[32:33] // (op 731) convert

v_mov_b32 v23, v31 // (op 731) 
v_mov_b32 v25, v23 // (op 731) reg expression
buffer_load_ubyte v28, v25, s[68:71], 0 offen offset: 0 // (op 731) Load value
v_add_u32 v25, v25, v40 // (op 731) 
buffer_load_ubyte v29, v25, s[68:71], 0 offen offset: 0 // (op 731) Load value



s_lshl_b32 s55, s32, 3 // (op 1594) 
v_add_u32 v31, v31, s55 // (op 1594) 


s_barrier  // (op 2896) 
v_mov_b64 v[40:41], s[38:39] // (op 732) convert




v_mov_b32 v23, v37 // (op 732) 
v_mov_b32 v25, v23 // (op 732) reg expression
buffer_load_ubyte v44, v25, s[72:75], 0 offen offset: 0 // (op 732) Load value
v_add_u32 v25, v25, v40 // (op 732) 
buffer_load_ubyte v45, v25, s[72:75], 0 offen offset: 0 // (op 732) Load value



s_lshl_b32 s55, s38, 3 // (op 1722) 
v_add_u32 v37, v37, s55 // (op 1722) 

s_waitcnt vmcnt(3)
ds_write_b8 v38, v28 offset:16512 // (op 728) Store local data 
s_barrier  // (op 2863) 
s_waitcnt vmcnt(2)
ds_write_b8 v38, v29 offset:16528 // (op 728) Store local data 




ds_read_b64_tr_b8 v[46:47], v0 offset:4096 // (op 710) Transpose load local data 
s_barrier  // (op 2916) 
ds_read_b64_tr_b8 v[48:49], v0 offset:4224 // (op 710) Transpose load local data 



ds_read_b64_tr_b8 v[50:51], v0 offset:5120 // (op 710) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v0 offset:5248 // (op 710) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v42, v44 offset:16384 // (op 729) Store local data 
s_barrier  // (op 2883) 
s_waitcnt vmcnt(0)
ds_write_b8 v42, v45 offset:16400 // (op 729) Store local data 




ds_read_b64_tr_b8 v[54:55], v3 // (op 712) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v3 offset:128 // (op 712) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v3 offset:1024 // (op 712) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v3 offset:1152 // (op 712) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 2903) 


ds_read_u8 v23, v5 offset:16512 // (op 714) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 2923) 


ds_read_u8 v25, v7 offset:16384 // (op 716) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[46:53], v[54:61], a[0:3], v23, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 717) 







ds_read_b64_tr_b8 v[44:45], v0 offset:6144 // (op 719) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v0 offset:6272 // (op 719) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v0 offset:7168 // (op 719) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v0 offset:7296 // (op 719) Transpose load local data 



ds_read_b64_tr_b8 v[52:53], v3 offset:2048 // (op 721) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v3 offset:2176 // (op 721) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v3 offset:3072 // (op 721) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v3 offset:3200 // (op 721) Transpose load local data 


ds_read_u8 v23, v5 offset:16576 // (op 723) Load local data 


ds_read_u8 v25, v7 offset:16448 // (op 725) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[44:51], v[52:59], a[0:3], v23, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 726) 




s_cmp_lt_u32 s54, s7 // (op 702) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopTop_KLoopTail_702 // (op 702) Condition: Bottom (jump to RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopTop_KLoopTail_702 if true)
RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopBottom_KLoopTail_702:









s_barrier  // (op 3206) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_cmp_lg_u32 s18, 0 // (op 362) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalFalse_Send_Tile_362 // (op 362) Condition: False, jump to RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalFalse_Send_Tile_362
v_lshrrev_b32 v20, 4, v1 // (op 2464) 
v_lshlrev_b32 v22, 2, v20 // (op 2464) 

v_lshl_add_u32 v20, s16, 4, v22 // (op 2464) 

v_and_b32 v22, -16, v1 // (op 2464) 
v_sub_u32 v23, v1, v22 // (op 2464) 

v_lshl_add_u32 v22, v20, 4, v23 // (op 2464) 


v_lshlrev_b32 v20, 2, v22 // (op 2464) 

s_lshl_b32 s52, s15, 10 // (op 2466) 
s_mov_b32 s76, s42 // (op 2466) reg expression
s_mov_b32 s77, s43 // (op 2466) reg expression
s_mov_b32 s78, s52 // (op 2466) reg expression
s_mov_b32 s79, 131072 // (op 2466) call()




buffer_store_dword a0, v20, s[76:79], 0 offen sc1 // (op 359) Store value
buffer_store_dword a1, v20, s[76:79], 0 offen offset:64 sc1 // (op 359) Store value
buffer_store_dword a2, v20, s[76:79], 0 offen offset:128 sc1 // (op 359) Store value
buffer_store_dword a3, v20, s[76:79], 0 offen offset:192 sc1 // (op 359) Store value

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_barrier  // (op 363) 
v_cmp_eq_i32 vcc, v1, 0 // (op 367) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalFalse_Wave0_Store_Flag_367 // (op 367) Condition: False, jump to RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalFalse_Wave0_Store_Flag_367
s_mov_b32 s52, 1 // (op 365) call()
s_mov_b32 s55, 0 // (op 366) convert
s_mov_b32 s54, s16 // (op 366) convert
s_lshl_b64 s[54:55], s[54:55], 2 // (op 366) 
s_add_u32 s80, s44, s54 // (op 366) Start of Int64 add, locking SCC
s_addc_u32 s81, s45, s55 // (op 366) End of Int64 add, unlocking SCC
s_store_dword s52, s[80:81], 0 glc // (op 366) Store scalar value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalBottom_Wave0_Store_Flag_367 // (op 367) Condition: Done, jump to RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalBottom_Wave0_Store_Flag_367
RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalFalse_Wave0_Store_Flag_367:
RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalBottom_Wave0_Store_Flag_367:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalBottom_Send_Tile_362 // (op 362) Condition: Done, jump to RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalBottom_Send_Tile_362
RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalFalse_Send_Tile_362:
RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalBottom_Send_Tile_362:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_cmp_eq_i32 s18, 0 // (op 378) 
s_mov_b32 s52, scc // (op 378) 
s_ashr_i64 s[54:55], s[2:3], 8 // (op 378) 
s_mov_b32 s53, s54 // (op 378) convert

s_sub_u32 s54, s53, 1 // (op 378) 

s_cmp_lt_u32 s19, s54 // (op 378) 
s_mov_b32 s53, scc // (op 378) 

s_and_b32 s54, s52, s53 // (op 378) 
s_cmp_lg_u32 s54, 0 // (op 378) 



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalFalse_Receive_Tile_378 // (op 378) Condition: False, jump to RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalFalse_Receive_Tile_378
s_mov_b32 s52, 0 // (op 344) call()
s_mul_i32 s53, s52, s8 // (op 343) 
s_ashr_i64 s[54:55], s[2:3], 8 // (op 343) 
s_mov_b32 s80, s54 // (op 343) convert

s_sub_u32 s54, s80, s19 // (op 343) 

s_sub_u32 s55, s54, 1 // (op 343) 

s_cmp_lt_u32 s53, s55 // (op 343) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopBottom_ReceiveTileLoop_343 // (op 343) Condition: Top (jump to RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopBottom_ReceiveTileLoop_343 if false)
RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopTop_ReceiveTileLoop_343:
s_add_u32 s53, s16, 1 // (op 2481) 
v_lshrrev_b32 v20, 4, v1 // (op 2481) 
s_add_u32 s54, s53, s52 // (op 2481) 
s_lshl_b32 s54, s54, 4 // (op 2481) 

v_lshlrev_b32 v22, 2, v20 // (op 2481) 

v_add_lshl_u32 v20, s54, v22, 4 // (op 2481) 


v_and_b32 v22, -16, v1 // (op 2481) 
v_sub_u32 v23, v1, v22 // (op 2481) 

v_add_lshl_u32 v22, v20, v23, 2 // (op 2481) 


s_lshl_b32 s53, s15, 10 // (op 2483) 
s_mov_b32 s76, s42 // (op 2483) reg expression
s_mov_b32 s77, s43 // (op 2483) reg expression
s_mov_b32 s78, s53 // (op 2483) reg expression
s_mov_b32 s79, 131072 // (op 2483) call()

RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_DoWhileTop_Global_sync_spin_loop_382:
s_add_u32 s53, s16, 1 // (op 380) 
s_add_u32 s54, s53, s52 // (op 380) Start of Int64 add, locking SCC
s_addc_u32 s55, 0, 0 // (op 380) End of Int64 add, unlocking SCC
s_lshl_b64 s[54:55], s[54:55], 2 // (op 380) 

s_add_u32 s80, s44, s54 // (op 380) Start of Int64 add, locking SCC
s_addc_u32 s81, s45, s55 // (op 380) End of Int64 add, unlocking SCC
s_load_dword s53, s[80:81], 0 glc // (op 380) Load scalar value


s_waitcnt lgkmcnt(0)
s_cmp_eq_i32 s53, 0 // (op 382) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_DoWhileTop_Global_sync_spin_loop_382 // (op 382) Condition: Bottom (jump to RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_DoWhileTop_Global_sync_spin_loop_382 if true)
s_barrier  // (op 386) 
v_cmp_eq_i32 vcc, v1, 0 // (op 385) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalFalse_Wave0_Reset_Flag_385 // (op 385) Condition: False, jump to RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalFalse_Wave0_Reset_Flag_385
s_mov_b32 s53, 0 // (op 383) call()
s_add_u32 s54, s16, 1 // (op 384) 
s_add_u32 s80, s54, s52 // (op 384) Start of Int64 add, locking SCC
s_addc_u32 s81, 0, 0 // (op 384) End of Int64 add, unlocking SCC
s_lshl_b64 s[80:81], s[80:81], 2 // (op 384) 

s_add_u32 s54, s44, s80 // (op 384) Start of Int64 add, locking SCC
s_addc_u32 s55, s45, s81 // (op 384) End of Int64 add, unlocking SCC
s_store_dword s53, s[54:55], 0 glc // (op 384) Store scalar value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalBottom_Wave0_Reset_Flag_385 // (op 385) Condition: Done, jump to RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalBottom_Wave0_Reset_Flag_385
RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalFalse_Wave0_Reset_Flag_385:
RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalBottom_Wave0_Reset_Flag_385:




buffer_load_dword v24, v22, s[76:79], 0 offen // (op 397) Load value
buffer_load_dword v25, v22, s[76:79], 0 offen offset:64 // (op 397) Load value
buffer_load_dword v26, v22, s[76:79], 0 offen offset:128 // (op 397) Load value
buffer_load_dword v27, v22, s[76:79], 0 offen offset:192 // (op 397) Load value
v_accvgpr_read v28, a0 // (op 400) reg expression
v_accvgpr_read v29, a1 // (op 400) reg expression
v_accvgpr_read v30, a2 // (op 400) reg expression
v_accvgpr_read v31, a3 // (op 400) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v28, v28, v24 // (op 402) 
s_waitcnt vmcnt(2)
v_add_f32 v29, v29, v25 // (op 402) 
s_waitcnt vmcnt(1)
v_add_f32 v30, v30, v26 // (op 402) 
s_waitcnt vmcnt(0)
v_add_f32 v31, v31, v27 // (op 402) 

v_accvgpr_write a0, v28 // (op 404) reg expression
v_accvgpr_write a1, v29 // (op 404) reg expression
v_accvgpr_write a2, v30 // (op 404) reg expression
v_accvgpr_write a3, v31 // (op 404) reg expression

s_add_u32 s52, s52, 1 // (op 345) 

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_mul_i32 s53, s52, s8 // (op 343) 
s_ashr_i64 s[54:55], s[2:3], 8 // (op 343) 
s_mov_b32 s80, s54 // (op 343) convert

s_sub_u32 s54, s80, s19 // (op 343) 

s_sub_u32 s55, s54, 1 // (op 343) 

s_cmp_lt_u32 s53, s55 // (op 343) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopTop_ReceiveTileLoop_343 // (op 343) Condition: Bottom (jump to RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopTop_ReceiveTileLoop_343 if true)
RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopBottom_ReceiveTileLoop_343:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalBottom_Receive_Tile_378 // (op 378) Condition: Done, jump to RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalBottom_Receive_Tile_378
RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalFalse_Receive_Tile_378:
RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalBottom_Receive_Tile_378:

s_cmp_eq_i32 s18, 0 // (op 421) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalFalse_Postmaccumulation_Condition_421 // (op 421) Condition: False, jump to RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalFalse_Postmaccumulation_Condition_421
v_cmp_eq_f32 vcc, 0.00000, v16 // (op 157) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalFalse_0_beta_157 // (op 157) Condition: False, jump to RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalFalse_0_beta_157
v_accvgpr_read v22, a0 // (op 50) 
v_accvgpr_read v23, a1 // (op 50) 
v_accvgpr_read v24, a2 // (op 50) 
v_accvgpr_read v25, a3 // (op 50) 
v_mul_f32 v26, v18, v22 // (op 50) 
v_mul_f32 v27, v18, v23 // (op 50) 
v_mul_f32 v28, v18, v24 // (op 50) 
v_mul_f32 v29, v18, v25 // (op 50) 

v_lshrrev_b32 v22, 16, v26 // (op 56) 
v_lshrrev_b32 v23, 16, v27 // (op 56) 
v_lshrrev_b32 v24, 16, v28 // (op 56) 
v_lshrrev_b32 v25, 16, v29 // (op 56) 

v_lshrrev_b32 v20, 6, v1 // (op 2132) 
s_mul_i32 s52, s16, s8 // (op 2132) 
s_add_u32 s52, s52, s0 // (op 2132) 
v_and_b32 v26, 63, v1 // (op 2132) 
s_mul_hi_u32 s53, s52, s13 // (op 2132) 
v_lshrrev_b32 v27, 4, v26 // (op 2132) 
s_sub_u32 s54, s52, s53 // (op 2132) 
v_lshlrev_b32 v28, 2, v27 // (op 2132) 

s_lshr_b32 s55, s54, 1 // (op 2132) 

v_lshl_add_u32 v27, v20, 4, v28 // (op 2132) 


s_add_u32 s54, s55, s53 // (op 2132) 


v_lshrrev_b32 v20, 2, v26 // (op 2132) 

s_lshr_b32 s53, s54, s11 // (op 2132) 

v_and_b32 v26, 3, v20 // (op 2132) 

s_and_b32 s54, s11, 2147483648 // (op 2132) 
v_and_b32 v20, 3, v1 // (op 2132) 
s_cmp_eq_i32 s54, 0 // (op 2132) 
s_mov_b32 s55, scc // (op 2132) 

v_lshl_add_u32 v28, v26, 2, v20 // (op 2132) 


s_cmp_eq_i32 s55, 0 // (op 2132) 
s_cselect_b32 s54, s52, s53 // (op 2132) 



s_mul_hi_u32 s55, s54, s12 // (op 2132) 
s_sub_u32 s52, s54, s55 // (op 2132) 
s_lshr_b32 s53, s52, 1 // (op 2132) 

s_add_u32 s52, s53, s55 // (op 2132) 


s_lshr_b32 s53, s52, s10 // (op 2132) 

s_and_b32 s52, s10, 2147483648 // (op 2132) 
s_cmp_eq_i32 s52, 0 // (op 2132) 
s_mov_b32 s55, scc // (op 2132) 

s_cmp_eq_i32 s55, 0 // (op 2132) 
s_cselect_b32 s52, s54, s53 // (op 2132) 


s_mul_i32 s53, s52, s14 // (op 2132) 
v_lshl_add_u32 v20, s52, 4, v27 // (op 2132) 


s_sub_u32 s52, s54, s53 // (op 2132) 


v_lshl_add_u32 v26, s52, 4, v28 // (op 2132) 


v_mul_lo_u32 v27, v26, s46 // (op 2132) 

v_add_lshl_u32 v26, v20, v27, 1 // (op 2132) 


s_mul_i32 s53, s46, s23 // (op 2134) most significant: low of low * high
s_mul_i32 s54, s47, s22 // (op 2134) most significant: low of high * low
s_mul_i32 s52, s46, s22 // (op 2134) least significant
s_mul_hi_u32 s55, s46, s22 // (op 2134) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2134) most significant: sum
s_add_u32 s53, s53, s55 // (op 2134) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 1 // (op 2134) 

s_mov_b32 s76, s48 // (op 2134) reg expression
s_mov_b32 s77, s49 // (op 2134) reg expression
s_mov_b32 s78, s54 // (op 2134) reg expression
s_mov_b32 s79, 131072 // (op 2134) call()


v_pack_b32_f16 v28, v22, v23 // (op 58) 
v_pack_b32_f16 v29, v24, v25 // (op 58) 


buffer_store_dwordx2 v[28:29], v26, s[76:79], 0 offen // (op 58) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalBottom_0_beta_157 // (op 157) Condition: Done, jump to RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalBottom_0_beta_157
RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalFalse_0_beta_157:



buffer_load_dwordx2 v[22:23], v21, s[56:59], 0 offen // (op 170) Load value
v_accvgpr_read v24, a0 // (op 166) 
v_accvgpr_read v25, a1 // (op 166) 
v_accvgpr_read v26, a2 // (op 166) 
v_accvgpr_read v27, a3 // (op 166) 
v_mul_f32 v28, v18, v24 // (op 166) 
v_mul_f32 v29, v18, v25 // (op 166) 
v_mul_f32 v30, v18, v26 // (op 166) 
v_mul_f32 v31, v18, v27 // (op 166) 

s_waitcnt vmcnt(0)
v_and_b32 v24, 65535, v22 // (op 171) 
v_lshrrev_b32 v25, 16, v22 // (op 171) 
v_lshlrev_b32 v24, 16, v24 // (op 171) 
v_lshlrev_b32 v25, 16, v25 // (op 171) 
v_mul_f32 v32, v16, v24 // (op 171) 
v_mul_f32 v33, v16, v25 // (op 171) 
v_and_b32 v24, 65535, v23 // (op 171) 
v_lshrrev_b32 v25, 16, v23 // (op 171) 
v_lshlrev_b32 v24, 16, v24 // (op 171) 
v_lshlrev_b32 v25, 16, v25 // (op 171) 
v_mul_f32 v34, v16, v24 // (op 171) 
v_mul_f32 v35, v16, v25 // (op 171) 


v_add_f32 v22, v32, v28 // (op 167) 
v_add_f32 v23, v33, v29 // (op 167) 
v_add_f32 v24, v34, v30 // (op 167) 
v_add_f32 v25, v35, v31 // (op 167) 


v_lshrrev_b32 v26, 16, v22 // (op 168) 
v_lshrrev_b32 v27, 16, v23 // (op 168) 
v_lshrrev_b32 v28, 16, v24 // (op 168) 
v_lshrrev_b32 v29, 16, v25 // (op 168) 

v_lshrrev_b32 v20, 6, v1 // (op 2149) 
s_mul_i32 s52, s16, s8 // (op 2149) 
s_add_u32 s52, s52, s0 // (op 2149) 
v_and_b32 v22, 63, v1 // (op 2149) 
s_mul_hi_u32 s53, s52, s13 // (op 2149) 
v_lshrrev_b32 v23, 4, v22 // (op 2149) 
s_sub_u32 s54, s52, s53 // (op 2149) 
v_lshlrev_b32 v24, 2, v23 // (op 2149) 

s_lshr_b32 s55, s54, 1 // (op 2149) 

v_lshl_add_u32 v23, v20, 4, v24 // (op 2149) 


s_add_u32 s54, s55, s53 // (op 2149) 


v_lshrrev_b32 v20, 2, v22 // (op 2149) 

s_lshr_b32 s53, s54, s11 // (op 2149) 

v_and_b32 v22, 3, v20 // (op 2149) 

s_and_b32 s54, s11, 2147483648 // (op 2149) 
v_and_b32 v20, 3, v1 // (op 2149) 
s_cmp_eq_i32 s54, 0 // (op 2149) 
s_mov_b32 s55, scc // (op 2149) 

v_lshl_add_u32 v24, v22, 2, v20 // (op 2149) 


s_cmp_eq_i32 s55, 0 // (op 2149) 
s_cselect_b32 s54, s52, s53 // (op 2149) 



s_mul_hi_u32 s55, s54, s12 // (op 2149) 
s_sub_u32 s52, s54, s55 // (op 2149) 
s_lshr_b32 s53, s52, 1 // (op 2149) 

s_add_u32 s52, s53, s55 // (op 2149) 


s_lshr_b32 s53, s52, s10 // (op 2149) 

s_and_b32 s52, s10, 2147483648 // (op 2149) 
s_cmp_eq_i32 s52, 0 // (op 2149) 
s_mov_b32 s55, scc // (op 2149) 

s_cmp_eq_i32 s55, 0 // (op 2149) 
s_cselect_b32 s52, s54, s53 // (op 2149) 


s_mul_i32 s53, s52, s14 // (op 2149) 
v_lshl_add_u32 v20, s52, 4, v23 // (op 2149) 


s_sub_u32 s52, s54, s53 // (op 2149) 


v_lshl_add_u32 v25, s52, 4, v24 // (op 2149) 


v_mul_lo_u32 v22, v25, s46 // (op 2149) 

v_add_lshl_u32 v23, v20, v22, 1 // (op 2149) 


s_mul_i32 s53, s46, s23 // (op 2151) most significant: low of low * high
s_mul_i32 s54, s47, s22 // (op 2151) most significant: low of high * low
s_mul_i32 s52, s46, s22 // (op 2151) least significant
s_mul_hi_u32 s55, s46, s22 // (op 2151) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2151) most significant: sum
s_add_u32 s53, s53, s55 // (op 2151) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 1 // (op 2151) 

s_mov_b32 s76, s48 // (op 2151) reg expression
s_mov_b32 s77, s49 // (op 2151) reg expression
s_mov_b32 s78, s54 // (op 2151) reg expression
s_mov_b32 s79, 131072 // (op 2151) call()


v_pack_b32_f16 v24, v26, v27 // (op 169) 
v_pack_b32_f16 v25, v28, v29 // (op 169) 


buffer_store_dwordx2 v[24:25], v23, s[76:79], 0 offen // (op 169) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalBottom_0_beta_157:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalBottom_Postmaccumulation_Condition_421 // (op 421) Condition: Done, jump to RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalBottom_Postmaccumulation_Condition_421
RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalFalse_Postmaccumulation_Condition_421:
RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalBottom_Postmaccumulation_Condition_421:

s_add_u32 s0, s0, s7 // (op 336) 
s_mul_i32 s52, s8, s16 // (op 334) 
s_cmp_gt_u32 s9, s52 // (op 334) 
s_mov_b32 s53, scc // (op 334) 
s_sub_u32 s54, s9, s52 // (op 334) 

s_cmp_eq_i32 s53, 0 // (op 334) 
s_cselect_b32 s52, 0, s54 // (op 334) 


s_cmp_lt_u32 s8, s52 // (op 334) 
s_mov_b32 s53, scc // (op 334) 
s_cmp_eq_i32 s53, 0 // (op 334) 
s_cselect_b32 s54, s52, s8 // (op 334) 


s_cmp_lt_u32 s0, s54 // (op 334) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopTop_SKStreamTileLoop_334 // (op 334) Condition: Bottom (jump to RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopTop_SKStreamTileLoop_334 if true)
RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopBottom_SKStreamTileLoop_334:

















s_mov_b32 s0, 0 // (op 475) call()
s_mov_b32 s1, 0 // (op 476) call()
s_mov_b32 s0, 0 // (op 490) call()
s_mul_i32 s15, s50, s16 // (op 489) 
s_cmp_gt_u32 s51, s15 // (op 489) 
s_mov_b32 s6, scc // (op 489) 
s_sub_u32 s7, s51, s15 // (op 489) 

s_cmp_eq_i32 s6, 0 // (op 489) 
s_cselect_b32 s8, 0, s7 // (op 489) 


s_cmp_lt_u32 s50, s8 // (op 489) 
s_mov_b32 s15, scc // (op 489) 
s_cmp_eq_i32 s15, 0 // (op 489) 
s_cselect_b32 s6, s8, s50 // (op 489) 


s_cmp_lt_u32 s0, s6 // (op 489) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopBottom_DPStreamTileLoop_489 // (op 489) Condition: Top (jump to RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopBottom_DPStreamTileLoop_489 if false)
RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopTop_DPStreamTileLoop_489:
v_lshrrev_b32 v0, 6, v1 // (op 1851) 
s_mul_i32 s15, s16, s50 // (op 1851) 
s_add_u32 s15, s15, s0 // (op 1851) 
v_and_b32 v3, 63, v1 // (op 1851) 
s_add_u32 s6, s15, s9 // (op 1851) 

v_lshrrev_b32 v5, 4, v3 // (op 1851) 
s_mul_hi_u32 s15, s6, s13 // (op 1851) 
v_lshlrev_b32 v7, 2, v5 // (op 1851) 

s_sub_u32 s7, s6, s15 // (op 1851) 
v_lshl_add_u32 v5, v0, 4, v7 // (op 1851) 


s_lshr_b32 s8, s7, 1 // (op 1851) 

v_lshrrev_b32 v0, 2, v3 // (op 1851) 

s_add_u32 s7, s8, s15 // (op 1851) 


v_and_b32 v3, 3, v0 // (op 1851) 

s_lshr_b32 s8, s7, s11 // (op 1851) 

v_and_b32 v0, 3, v1 // (op 1851) 
s_and_b32 s7, s11, 2147483648 // (op 1851) 
v_lshl_add_u32 v7, v3, 2, v0 // (op 1851) 


s_cmp_eq_i32 s7, 0 // (op 1851) 
s_mov_b32 s15, scc // (op 1851) 

s_cmp_eq_i32 s15, 0 // (op 1851) 
s_cselect_b32 s7, s6, s8 // (op 1851) 



s_mul_hi_u32 s6, s7, s12 // (op 1851) 
s_sub_u32 s8, s7, s6 // (op 1851) 
s_lshr_b32 s15, s8, 1 // (op 1851) 

s_add_u32 s8, s15, s6 // (op 1851) 


s_lshr_b32 s6, s8, s10 // (op 1851) 

s_and_b32 s8, s10, 2147483648 // (op 1851) 
s_cmp_eq_i32 s8, 0 // (op 1851) 
s_mov_b32 s15, scc // (op 1851) 

s_cmp_eq_i32 s15, 0 // (op 1851) 
s_cselect_b32 s8, s7, s6 // (op 1851) 


s_mul_i32 s6, s8, s14 // (op 1851) 
v_lshl_add_u32 v0, s8, 4, v5 // (op 1851) 


s_sub_u32 s8, s7, s6 // (op 1851) 


v_lshl_add_u32 v3, s8, 4, v7 // (op 1851) 


v_mul_lo_u32 v5, v3, s4 // (op 1851) 

v_add_lshl_u32 v3, v0, v5, 1 // (op 1851) 


s_mul_i32 s19, s4, s23 // (op 1853) most significant: low of low * high
s_mul_i32 s15, s5, s22 // (op 1853) most significant: low of high * low
s_mul_i32 s18, s4, s22 // (op 1853) least significant
s_mul_hi_u32 s6, s4, s22 // (op 1853) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1853) most significant: sum
s_add_u32 s19, s19, s6 // (op 1853) most significant: sum


s_lshl_b64 s[6:7], s[18:19], 1 // (op 1853) 

s_mov_b32 s56, s20 // (op 1853) reg expression
s_mov_b32 s57, s21 // (op 1853) reg expression
s_mov_b32 s58, s6 // (op 1853) reg expression
s_mov_b32 s59, 131072 // (op 1853) call()

v_and_b32 v0, 48, v1 // (op 2252) 
v_and_b32 v5, 63, v1 // (op 2252) 
v_sub_u32 v7, v5, v0 // (op 2252) 

v_lshrrev_b32 v5, 1, v7 // (op 2252) 
v_add_lshl_u32 v12, v0, v5, 4 // (op 2252) 


v_and_b32 v0, 1, v7 // (op 2252) 

v_lshlrev_b32 v5, 3, v0 // (op 2252) 

v_lshrrev_b32 v0, 6, v1 // (op 2252) 
v_lshl_add_u32 v7, v0, 4, v5 // (op 2252) 


v_add_u32 v0, v12, v7 // (op 2252) 


v_and_b32 v5, 48, v1 // (op 2317) 
v_and_b32 v7, 63, v1 // (op 2317) 
v_sub_u32 v12, v7, v5 // (op 2317) 

v_lshrrev_b32 v7, 1, v12 // (op 2317) 
v_add_lshl_u32 v14, v5, v7, 4 // (op 2317) 


v_and_b32 v5, 1, v12 // (op 2317) 

v_lshlrev_b32 v7, 3, v5 // (op 2317) 

v_add_u32 v5, v14, v7 // (op 2317) 


v_lshrrev_b32 v7, 6, v1 // (op 2383) 
v_and_b32 v12, 15, v1 // (op 2383) 
v_lshl_add_u32 v14, v7, 4, v12 // (op 2383) 


v_and_b32 v7, 48, v1 // (op 2383) 
v_add_u32 v12, v7, v14 // (op 2383) 


v_and_b32 v7, 48, v1 // (op 2449) 
v_and_b32 v14, 15, v1 // (op 2449) 
v_add_u32 v17, v7, v14 // (op 2449) 


v_accvgpr_write a0, 0.00000 // (op 206) call()
v_accvgpr_write a1, 0.00000 // (op 206) call()
v_accvgpr_write a2, 0.00000 // (op 206) call()
v_accvgpr_write a3, 0.00000 // (op 206) call()
s_mul_i32 s15, s16, s50 // (op 1411) 
s_add_u32 s15, s15, s0 // (op 1411) 
s_add_u32 s6, s15, s9 // (op 1411) 

s_mul_hi_u32 s15, s6, s13 // (op 1411) 
s_sub_u32 s7, s6, s15 // (op 1411) 
s_lshr_b32 s8, s7, 1 // (op 1411) 

s_add_u32 s7, s8, s15 // (op 1411) 


s_lshr_b32 s8, s7, s11 // (op 1411) 

s_and_b32 s7, s11, 2147483648 // (op 1411) 
s_cmp_eq_i32 s7, 0 // (op 1411) 
s_mov_b32 s15, scc // (op 1411) 

s_cmp_eq_i32 s15, 0 // (op 1411) 
s_cselect_b32 s7, s6, s8 // (op 1411) 


s_mul_hi_u32 s8, s7, s12 // (op 1411) 
s_sub_u32 s15, s7, s8 // (op 1411) 
s_lshr_b32 s18, s15, 1 // (op 1411) 

s_add_u32 s15, s18, s8 // (op 1411) 


s_lshr_b32 s8, s15, s10 // (op 1411) 

s_and_b32 s15, s10, 2147483648 // (op 1411) 
s_cmp_eq_i32 s15, 0 // (op 1411) 
s_mov_b32 s18, scc // (op 1411) 

s_cmp_eq_i32 s18, 0 // (op 1411) 
s_cselect_b32 s15, s7, s8 // (op 1411) 


s_lshl_b32 s8, s15, 4 // (op 1411) 

s_mov_b32 s19, 0 // (op 1411) convert
s_mov_b32 s18, s8 // (op 1411) convert

s_ashr_i64 s[42:43], s[2:3], 8 // (op 1411) 
s_mul_i32 s8, s7, s42 // (op 1411) 


s_sub_u32 s7, s6, s8 // (op 1411) 


v_lshl_add_u32 v7, s7, 8, v1 // (op 1411) 

v_mul_lo_u32 v21, v7, s25 // (op 1411) most significant: low of low * high
v_mul_hi_u32 v14, v7, s24 // (op 1411) most significant: high of low * low
v_mul_lo_u32 v20, v7, s24 // (op 1411) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 1411) most significant: sum


v_mov_b32 v7, s18 // (op 1411) 
v_mov_b32 v14, s19 // (op 1411) 
v_add_co_u32 v22, s[6:7], v7, v20 // (op 1411) least significant half
v_addc_co_u32 v23, s[6:7], v14, v21, s[6:7] // (op 1411) most significant half





s_mul_i32 s19, s24, s3 // (op 1413) most significant: low of low * high
s_mul_i32 s15, s25, s2 // (op 1413) most significant: low of high * low
s_mul_i32 s18, s24, s2 // (op 1413) least significant
s_mul_hi_u32 s6, s24, s2 // (op 1413) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1413) most significant: sum
s_add_u32 s19, s19, s6 // (op 1413) most significant: sum


s_mov_b32 s60, s26 // (op 1413) reg expression
s_mov_b32 s61, s27 // (op 1413) reg expression
s_mov_b32 s62, s18 // (op 1413) reg expression
s_mov_b32 s63, 131072 // (op 1413) call()

s_mul_i32 s15, s16, s50 // (op 1430) 
s_add_u32 s15, s15, s0 // (op 1430) 
s_add_u32 s6, s15, s9 // (op 1430) 

s_mul_hi_u32 s15, s6, s13 // (op 1430) 
s_sub_u32 s7, s6, s15 // (op 1430) 
s_lshr_b32 s8, s7, 1 // (op 1430) 

s_add_u32 s7, s8, s15 // (op 1430) 


s_lshr_b32 s8, s7, s11 // (op 1430) 

s_and_b32 s7, s11, 2147483648 // (op 1430) 
s_cmp_eq_i32 s7, 0 // (op 1430) 
s_mov_b32 s15, scc // (op 1430) 

s_cmp_eq_i32 s15, 0 // (op 1430) 
s_cselect_b32 s7, s6, s8 // (op 1430) 


s_mul_hi_u32 s8, s7, s12 // (op 1430) 
s_sub_u32 s15, s7, s8 // (op 1430) 
s_lshr_b32 s18, s15, 1 // (op 1430) 

s_add_u32 s15, s18, s8 // (op 1430) 


s_lshr_b32 s8, s15, s10 // (op 1430) 

s_and_b32 s15, s10, 2147483648 // (op 1430) 
s_cmp_eq_i32 s15, 0 // (op 1430) 
s_mov_b32 s18, scc // (op 1430) 

s_cmp_eq_i32 s18, 0 // (op 1430) 
s_cselect_b32 s15, s7, s8 // (op 1430) 


s_lshl_b32 s8, s15, 4 // (op 1430) 

s_mov_b32 s19, 0 // (op 1430) convert
s_mov_b32 s18, s8 // (op 1430) convert

s_ashr_i64 s[42:43], s[2:3], 8 // (op 1430) 
s_mul_i32 s8, s7, s42 // (op 1430) 


s_sub_u32 s7, s6, s8 // (op 1430) 


v_lshl_add_u32 v7, s7, 8, v1 // (op 1430) 

v_mul_lo_u32 v21, v7, s25 // (op 1430) most significant: low of low * high
v_mul_hi_u32 v14, v7, s24 // (op 1430) most significant: high of low * low
v_mul_lo_u32 v20, v7, s24 // (op 1430) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 1430) most significant: sum


v_mov_b32 v7, s18 // (op 1430) 
v_mov_b32 v14, s19 // (op 1430) 
v_add_co_u32 v24, s[6:7], v7, v20 // (op 1430) least significant half
v_addc_co_u32 v25, s[6:7], v14, v21, s[6:7] // (op 1430) most significant half





s_mul_i32 s19, s24, s3 // (op 1432) most significant: low of low * high
s_mul_i32 s15, s25, s2 // (op 1432) most significant: low of high * low
s_mul_i32 s18, s24, s2 // (op 1432) least significant
s_mul_hi_u32 s6, s24, s2 // (op 1432) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1432) most significant: sum
s_add_u32 s19, s19, s6 // (op 1432) most significant: sum


s_mov_b32 s60, s26 // (op 1432) reg expression
s_mov_b32 s61, s27 // (op 1432) reg expression
s_mov_b32 s62, s18 // (op 1432) reg expression
s_mov_b32 s63, 131072 // (op 1432) call()

s_mul_i32 s15, s16, s50 // (op 1539) 
s_add_u32 s15, s15, s0 // (op 1539) 
s_add_u32 s6, s15, s9 // (op 1539) 

s_mul_hi_u32 s15, s6, s13 // (op 1539) 
s_sub_u32 s7, s6, s15 // (op 1539) 
s_lshr_b32 s8, s7, 1 // (op 1539) 

s_add_u32 s7, s8, s15 // (op 1539) 


s_lshr_b32 s8, s7, s11 // (op 1539) 

s_and_b32 s7, s11, 2147483648 // (op 1539) 
s_cmp_eq_i32 s7, 0 // (op 1539) 
s_mov_b32 s15, scc // (op 1539) 

s_cmp_eq_i32 s15, 0 // (op 1539) 
s_cselect_b32 s7, s6, s8 // (op 1539) 


s_mul_hi_u32 s8, s7, s12 // (op 1539) 
s_sub_u32 s15, s7, s8 // (op 1539) 
s_lshr_b32 s18, s15, 1 // (op 1539) 

s_add_u32 s15, s18, s8 // (op 1539) 


s_lshr_b32 s8, s15, s10 // (op 1539) 

s_and_b32 s15, s10, 2147483648 // (op 1539) 
s_cmp_eq_i32 s15, 0 // (op 1539) 
s_mov_b32 s18, scc // (op 1539) 

s_cmp_eq_i32 s18, 0 // (op 1539) 
s_cselect_b32 s15, s7, s8 // (op 1539) 


s_mul_i32 s8, s15, s14 // (op 1539) 

s_sub_u32 s15, s7, s8 // (op 1539) 

s_lshl_b32 s8, s15, 4 // (op 1539) 

s_mov_b32 s19, 0 // (op 1539) convert
s_mov_b32 s18, s8 // (op 1539) convert

s_ashr_i64 s[42:43], s[2:3], 8 // (op 1539) 
s_mul_i32 s8, s7, s42 // (op 1539) 


s_sub_u32 s7, s6, s8 // (op 1539) 


v_lshl_add_u32 v7, s7, 8, v1 // (op 1539) 

v_mul_lo_u32 v21, v7, s29 // (op 1539) most significant: low of low * high
v_mul_hi_u32 v14, v7, s28 // (op 1539) most significant: high of low * low
v_mul_lo_u32 v20, v7, s28 // (op 1539) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 1539) most significant: sum

v_mov_b32 v14, s18 // (op 1539) 
v_mov_b32 v19, s19 // (op 1539) 
v_add_co_u32 v20, s[6:7], v20, v14 // (op 1539) least significant half
v_addc_co_u32 v21, s[6:7], v21, v19, s[6:7] // (op 1539) most significant half





s_mul_i32 s19, s28, s3 // (op 1541) most significant: low of low * high
s_mul_i32 s15, s29, s2 // (op 1541) most significant: low of high * low
s_mul_i32 s18, s28, s2 // (op 1541) least significant
s_mul_hi_u32 s6, s28, s2 // (op 1541) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1541) most significant: sum
s_add_u32 s19, s19, s6 // (op 1541) most significant: sum


s_mov_b32 s64, s30 // (op 1541) reg expression
s_mov_b32 s65, s31 // (op 1541) reg expression
s_mov_b32 s66, s18 // (op 1541) reg expression
s_mov_b32 s67, 131072 // (op 1541) call()

s_mul_i32 s15, s16, s50 // (op 1558) 
s_add_u32 s15, s15, s0 // (op 1558) 
s_add_u32 s6, s15, s9 // (op 1558) 

s_mul_hi_u32 s15, s6, s13 // (op 1558) 
s_sub_u32 s7, s6, s15 // (op 1558) 
s_lshr_b32 s8, s7, 1 // (op 1558) 

s_add_u32 s7, s8, s15 // (op 1558) 


s_lshr_b32 s8, s7, s11 // (op 1558) 

s_and_b32 s7, s11, 2147483648 // (op 1558) 
s_cmp_eq_i32 s7, 0 // (op 1558) 
s_mov_b32 s15, scc // (op 1558) 

s_cmp_eq_i32 s15, 0 // (op 1558) 
s_cselect_b32 s7, s6, s8 // (op 1558) 


s_mul_hi_u32 s8, s7, s12 // (op 1558) 
s_sub_u32 s15, s7, s8 // (op 1558) 
s_lshr_b32 s18, s15, 1 // (op 1558) 

s_add_u32 s15, s18, s8 // (op 1558) 


s_lshr_b32 s8, s15, s10 // (op 1558) 

s_and_b32 s15, s10, 2147483648 // (op 1558) 
s_cmp_eq_i32 s15, 0 // (op 1558) 
s_mov_b32 s18, scc // (op 1558) 

s_cmp_eq_i32 s18, 0 // (op 1558) 
s_cselect_b32 s15, s7, s8 // (op 1558) 


s_mul_i32 s8, s15, s14 // (op 1558) 

s_sub_u32 s15, s7, s8 // (op 1558) 

s_lshl_b32 s8, s15, 4 // (op 1558) 

s_mov_b32 s19, 0 // (op 1558) convert
s_mov_b32 s18, s8 // (op 1558) convert

s_ashr_i64 s[42:43], s[2:3], 8 // (op 1558) 
s_mul_i32 s8, s7, s42 // (op 1558) 


s_sub_u32 s7, s6, s8 // (op 1558) 


v_lshl_add_u32 v7, s7, 8, v1 // (op 1558) 

v_mul_lo_u32 v27, v7, s29 // (op 1558) most significant: low of low * high
v_mul_hi_u32 v14, v7, s28 // (op 1558) most significant: high of low * low
v_mul_lo_u32 v26, v7, s28 // (op 1558) least significant: low of low * low
v_add_u32 v27, v27, v14 // (op 1558) most significant: sum

v_mov_b32 v14, s18 // (op 1558) 
v_mov_b32 v19, s19 // (op 1558) 
v_add_co_u32 v26, s[6:7], v26, v14 // (op 1558) least significant half
v_addc_co_u32 v27, s[6:7], v27, v19, s[6:7] // (op 1558) most significant half





s_mul_i32 s19, s28, s3 // (op 1560) most significant: low of low * high
s_mul_i32 s15, s29, s2 // (op 1560) most significant: low of high * low
s_mul_i32 s18, s28, s2 // (op 1560) least significant
s_mul_hi_u32 s6, s28, s2 // (op 1560) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1560) most significant: sum
s_add_u32 s19, s19, s6 // (op 1560) most significant: sum


s_mov_b32 s64, s30 // (op 1560) reg expression
s_mov_b32 s65, s31 // (op 1560) reg expression
s_mov_b32 s66, s18 // (op 1560) reg expression
s_mov_b32 s67, 131072 // (op 1560) call()

v_and_b32 v7, -16, v1 // (op 1667) 
s_mul_i32 s15, s16, s50 // (op 1667) 
s_add_u32 s15, s15, s0 // (op 1667) 
v_sub_u32 v14, v1, v7 // (op 1667) 

s_add_u32 s6, s15, s9 // (op 1667) 

v_lshrrev_b32 v7, 4, v1 // (op 1667) 
s_mul_hi_u32 s15, s6, s13 // (op 1667) 
v_lshlrev_b32 v19, 1, v7 // (op 1667) 

s_sub_u32 s7, s6, s15 // (op 1667) 
s_lshr_b32 s8, s7, 1 // (op 1667) 

s_add_u32 s7, s8, s15 // (op 1667) 


s_lshr_b32 s8, s7, s11 // (op 1667) 

s_and_b32 s7, s11, 2147483648 // (op 1667) 
s_cmp_eq_i32 s7, 0 // (op 1667) 
s_mov_b32 s15, scc // (op 1667) 

s_cmp_eq_i32 s15, 0 // (op 1667) 
s_cselect_b32 s7, s6, s8 // (op 1667) 


s_mul_hi_u32 s8, s7, s12 // (op 1667) 
s_sub_u32 s15, s7, s8 // (op 1667) 
s_lshr_b32 s18, s15, 1 // (op 1667) 

s_add_u32 s15, s18, s8 // (op 1667) 


s_lshr_b32 s8, s15, s10 // (op 1667) 

s_and_b32 s15, s10, 2147483648 // (op 1667) 
s_cmp_eq_i32 s15, 0 // (op 1667) 
s_mov_b32 s18, scc // (op 1667) 

s_cmp_eq_i32 s18, 0 // (op 1667) 
s_cselect_b32 s15, s7, s8 // (op 1667) 


v_lshl_add_u32 v7, s15, 4, v14 // (op 1667) 


s_ashr_i64 s[18:19], s[2:3], 8 // (op 1667) 
s_mul_i32 s8, s7, s18 // (op 1667) 


s_sub_u32 s7, s6, s8 // (op 1667) 


v_lshl_add_u32 v14, s7, 3, v19 // (op 1667) 


v_mul_lo_u32 v19, v14, s32 // (op 1667) 

v_add_u32 v14, v7, v19 // (op 1667) 


s_mul_i32 s19, s32, s37 // (op 1669) most significant: low of low * high
s_mul_i32 s15, s33, s36 // (op 1669) most significant: low of high * low
s_mul_i32 s18, s32, s36 // (op 1669) least significant
s_mul_hi_u32 s6, s32, s36 // (op 1669) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1669) most significant: sum
s_add_u32 s19, s19, s6 // (op 1669) most significant: sum


s_mov_b32 s68, s34 // (op 1669) reg expression
s_mov_b32 s69, s35 // (op 1669) reg expression
s_mov_b32 s70, s18 // (op 1669) reg expression
s_mov_b32 s71, 131072 // (op 1669) call()

v_and_b32 v7, -16, v1 // (op 1686) 
s_mul_i32 s15, s16, s50 // (op 1686) 
s_add_u32 s15, s15, s0 // (op 1686) 
v_sub_u32 v19, v1, v7 // (op 1686) 

s_add_u32 s6, s15, s9 // (op 1686) 

v_lshrrev_b32 v7, 4, v1 // (op 1686) 
s_mul_hi_u32 s15, s6, s13 // (op 1686) 
v_lshlrev_b32 v28, 1, v7 // (op 1686) 

s_sub_u32 s7, s6, s15 // (op 1686) 
s_lshr_b32 s8, s7, 1 // (op 1686) 

s_add_u32 s7, s8, s15 // (op 1686) 


s_lshr_b32 s8, s7, s11 // (op 1686) 

s_and_b32 s7, s11, 2147483648 // (op 1686) 
s_cmp_eq_i32 s7, 0 // (op 1686) 
s_mov_b32 s15, scc // (op 1686) 

s_cmp_eq_i32 s15, 0 // (op 1686) 
s_cselect_b32 s7, s6, s8 // (op 1686) 


s_mul_hi_u32 s8, s7, s12 // (op 1686) 
s_sub_u32 s15, s7, s8 // (op 1686) 
s_lshr_b32 s18, s15, 1 // (op 1686) 

s_add_u32 s15, s18, s8 // (op 1686) 


s_lshr_b32 s8, s15, s10 // (op 1686) 

s_and_b32 s15, s10, 2147483648 // (op 1686) 
s_cmp_eq_i32 s15, 0 // (op 1686) 
s_mov_b32 s18, scc // (op 1686) 

s_cmp_eq_i32 s18, 0 // (op 1686) 
s_cselect_b32 s15, s7, s8 // (op 1686) 


v_lshl_add_u32 v7, s15, 4, v19 // (op 1686) 


s_ashr_i64 s[18:19], s[2:3], 8 // (op 1686) 
s_mul_i32 s8, s7, s18 // (op 1686) 


s_sub_u32 s7, s6, s8 // (op 1686) 


v_lshl_add_u32 v19, s7, 3, v28 // (op 1686) 


v_mul_lo_u32 v28, v19, s32 // (op 1686) 

v_add_u32 v19, v7, v28 // (op 1686) 


s_mul_i32 s19, s32, s37 // (op 1688) most significant: low of low * high
s_mul_i32 s15, s33, s36 // (op 1688) most significant: low of high * low
s_mul_i32 s18, s32, s36 // (op 1688) least significant
s_mul_hi_u32 s6, s32, s36 // (op 1688) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1688) most significant: sum
s_add_u32 s19, s19, s6 // (op 1688) most significant: sum


s_mov_b32 s68, s34 // (op 1688) reg expression
s_mov_b32 s69, s35 // (op 1688) reg expression
s_mov_b32 s70, s18 // (op 1688) reg expression
s_mov_b32 s71, 131072 // (op 1688) call()

v_and_b32 v7, -16, v1 // (op 1795) 
s_mul_i32 s15, s16, s50 // (op 1795) 
s_add_u32 s15, s15, s0 // (op 1795) 
v_sub_u32 v28, v1, v7 // (op 1795) 

s_add_u32 s6, s15, s9 // (op 1795) 

v_lshrrev_b32 v7, 4, v1 // (op 1795) 
s_mul_hi_u32 s15, s6, s13 // (op 1795) 
v_lshlrev_b32 v29, 1, v7 // (op 1795) 

s_sub_u32 s7, s6, s15 // (op 1795) 
s_lshr_b32 s8, s7, 1 // (op 1795) 

s_add_u32 s7, s8, s15 // (op 1795) 


s_lshr_b32 s8, s7, s11 // (op 1795) 

s_and_b32 s7, s11, 2147483648 // (op 1795) 
s_cmp_eq_i32 s7, 0 // (op 1795) 
s_mov_b32 s15, scc // (op 1795) 

s_cmp_eq_i32 s15, 0 // (op 1795) 
s_cselect_b32 s7, s6, s8 // (op 1795) 


s_mul_hi_u32 s8, s7, s12 // (op 1795) 
s_sub_u32 s15, s7, s8 // (op 1795) 
s_lshr_b32 s18, s15, 1 // (op 1795) 

s_add_u32 s15, s18, s8 // (op 1795) 


s_lshr_b32 s8, s15, s10 // (op 1795) 

s_and_b32 s15, s10, 2147483648 // (op 1795) 
s_cmp_eq_i32 s15, 0 // (op 1795) 
s_mov_b32 s18, scc // (op 1795) 

s_cmp_eq_i32 s18, 0 // (op 1795) 
s_cselect_b32 s15, s7, s8 // (op 1795) 


s_mul_i32 s8, s15, s14 // (op 1795) 

s_sub_u32 s15, s7, s8 // (op 1795) 

v_lshl_add_u32 v7, s15, 4, v28 // (op 1795) 


s_ashr_i64 s[18:19], s[2:3], 8 // (op 1795) 
s_mul_i32 s8, s7, s18 // (op 1795) 


s_sub_u32 s7, s6, s8 // (op 1795) 


v_lshl_add_u32 v28, s7, 3, v29 // (op 1795) 


v_mul_lo_u32 v29, v28, s38 // (op 1795) 
v_add_u32 v29, v29, v7 // (op 1795) 


s_mul_i32 s19, s38, s37 // (op 1797) most significant: low of low * high
s_mul_i32 s15, s39, s36 // (op 1797) most significant: low of high * low
s_mul_i32 s18, s38, s36 // (op 1797) least significant
s_mul_hi_u32 s6, s38, s36 // (op 1797) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1797) most significant: sum
s_add_u32 s19, s19, s6 // (op 1797) most significant: sum


s_mov_b32 s72, s40 // (op 1797) reg expression
s_mov_b32 s73, s41 // (op 1797) reg expression
s_mov_b32 s74, s18 // (op 1797) reg expression
s_mov_b32 s75, 131072 // (op 1797) call()

v_and_b32 v7, -16, v1 // (op 1814) 
s_mul_i32 s15, s16, s50 // (op 1814) 
s_add_u32 s15, s15, s0 // (op 1814) 
v_sub_u32 v28, v1, v7 // (op 1814) 

s_add_u32 s6, s15, s9 // (op 1814) 

v_lshrrev_b32 v7, 4, v1 // (op 1814) 
s_mul_hi_u32 s15, s6, s13 // (op 1814) 
v_lshlrev_b32 v30, 1, v7 // (op 1814) 

s_sub_u32 s7, s6, s15 // (op 1814) 
s_lshr_b32 s8, s7, 1 // (op 1814) 

s_add_u32 s7, s8, s15 // (op 1814) 


s_lshr_b32 s8, s7, s11 // (op 1814) 

s_and_b32 s7, s11, 2147483648 // (op 1814) 
s_cmp_eq_i32 s7, 0 // (op 1814) 
s_mov_b32 s15, scc // (op 1814) 

s_cmp_eq_i32 s15, 0 // (op 1814) 
s_cselect_b32 s7, s6, s8 // (op 1814) 


s_mul_hi_u32 s8, s7, s12 // (op 1814) 
s_sub_u32 s15, s7, s8 // (op 1814) 
s_lshr_b32 s18, s15, 1 // (op 1814) 

s_add_u32 s15, s18, s8 // (op 1814) 


s_lshr_b32 s8, s15, s10 // (op 1814) 

s_and_b32 s15, s10, 2147483648 // (op 1814) 
s_cmp_eq_i32 s15, 0 // (op 1814) 
s_mov_b32 s18, scc // (op 1814) 

s_cmp_eq_i32 s18, 0 // (op 1814) 
s_cselect_b32 s15, s7, s8 // (op 1814) 


s_mul_i32 s8, s15, s14 // (op 1814) 

s_sub_u32 s15, s7, s8 // (op 1814) 

v_lshl_add_u32 v7, s15, 4, v28 // (op 1814) 


s_ashr_i64 s[18:19], s[2:3], 8 // (op 1814) 
s_mul_i32 s8, s7, s18 // (op 1814) 


s_sub_u32 s7, s6, s8 // (op 1814) 


v_lshl_add_u32 v28, s7, 3, v30 // (op 1814) 


v_mul_lo_u32 v30, v28, s38 // (op 1814) 
v_add_u32 v30, v30, v7 // (op 1814) 


s_mul_i32 s19, s38, s37 // (op 1816) most significant: low of low * high
s_mul_i32 s15, s39, s36 // (op 1816) most significant: low of high * low
s_mul_i32 s18, s38, s36 // (op 1816) least significant
s_mul_hi_u32 s6, s38, s36 // (op 1816) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1816) most significant: sum
s_add_u32 s19, s19, s6 // (op 1816) most significant: sum


s_mov_b32 s72, s40 // (op 1816) reg expression
s_mov_b32 s73, s41 // (op 1816) reg expression
s_mov_b32 s74, s18 // (op 1816) reg expression
s_mov_b32 s75, 131072 // (op 1816) call()

v_lshlrev_b32 v7, 4, v1 // (op 2293) 
s_nop 0
v_readfirstlane_b32 s15, v7 // (op 2293) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 2949) 

s_lshl_b64 s[18:19], s[28:29], 6 // (op 1291) 
v_mov_b64 v[32:33], s[18:19] // (op 1291) convert


v_mov_b64 v[34:35], v[20:21] // (op 1291) reg expression
s_mov_b32 m0, s15 // (op 1291) reg expression
buffer_load_dwordx4 v34, s[64:67], 0 offen offset: 0 lds // (op 1291) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v32 // (op 1291) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 1291) most significant half

s_add_u32 m0, m0, 1024 // (op 1291) 
buffer_load_dwordx4 v34, s[64:67], 0 offen offset: 0 lds // (op 1291) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v32 // (op 1291) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 1291) most significant half

s_add_u32 m0, m0, 1024 // (op 1291) 
buffer_load_dwordx4 v34, s[64:67], 0 offen offset: 0 lds // (op 1291) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v32 // (op 1291) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 1291) most significant half

s_add_u32 m0, m0, 1024 // (op 1291) 
buffer_load_dwordx4 v34, s[64:67], 0 offen offset: 0 lds // (op 1291) Load value direct to lds




v_mov_b64 v[32:33], s[38:39] // (op 1098) convert

v_mov_b32 v7, v29 // (op 1098) reg expression
buffer_load_ubyte v34, v7, s[72:75], 0 offen offset: 0 // (op 1098) Load value
v_add_u32 v7, v7, v32 // (op 1098) 
buffer_load_ubyte v35, v7, s[72:75], 0 offen offset: 0 // (op 1098) Load value


v_lshlrev_b32 v7, 4, v1 // (op 2228) 
s_nop 0
v_readfirstlane_b32 s15, v7 // (op 2228) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 2930) 

s_lshl_b64 s[18:19], s[24:25], 6 // (op 1296) 
v_mov_b64 v[32:33], s[18:19] // (op 1296) convert


v_mov_b64 v[36:37], v[22:23] // (op 1296) reg expression
s_add_u32 s6, 4096, s15 // (op 1296) 
s_mov_b32 m0, s6 // (op 1296) reg expression
buffer_load_dwordx4 v36, s[60:63], 0 offen offset: 0 lds // (op 1296) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v32 // (op 1296) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 1296) most significant half

s_add_u32 m0, m0, 1024 // (op 1296) 
buffer_load_dwordx4 v36, s[60:63], 0 offen offset: 0 lds // (op 1296) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v32 // (op 1296) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 1296) most significant half

s_add_u32 m0, m0, 1024 // (op 1296) 
buffer_load_dwordx4 v36, s[60:63], 0 offen offset: 0 lds // (op 1296) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v32 // (op 1296) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 1296) most significant half

s_add_u32 m0, m0, 1024 // (op 1296) 
buffer_load_dwordx4 v36, s[60:63], 0 offen offset: 0 lds // (op 1296) Load value direct to lds





v_mov_b64 v[32:33], s[32:33] // (op 1104) convert

v_mov_b32 v7, v14 // (op 1104) reg expression
buffer_load_ubyte v36, v7, s[68:71], 0 offen offset: 0 // (op 1104) Load value
v_add_u32 v7, v7, v32 // (op 1104) 
buffer_load_ubyte v37, v7, s[68:71], 0 offen offset: 0 // (op 1104) Load value


v_lshlrev_b32 v7, 4, v1 // (op 2550) 
s_nop 0
v_readfirstlane_b32 s15, v7 // (op 2550) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 3019) 

s_lshl_b64 s[18:19], s[28:29], 6 // (op 1301) 
v_mov_b64 v[32:33], s[18:19] // (op 1301) convert


s_lshl_b64 s[18:19], s[28:29], 8 // (op 1301) 
v_mov_b32 v7, s18 // (op 1301) 
v_mov_b32 v28, s19 // (op 1301) 
v_add_co_u32 v38, s[6:7], v20, v7 // (op 1301) least significant half
v_addc_co_u32 v39, s[6:7], v21, v28, s[6:7] // (op 1301) most significant half




s_add_u32 s6, 8192, s15 // (op 1301) 
s_nop 0
s_mov_b32 m0, s6 // (op 1301) reg expression
buffer_load_dwordx4 v38, s[64:67], 0 offen offset: 0 lds // (op 1301) Load value direct to lds
v_add_co_u32 v38, s[18:19], v38, v32 // (op 1301) least significant half
v_addc_co_u32 v39, s[18:19], 0, v39, s[18:19] // (op 1301) most significant half

s_add_u32 m0, m0, 1024 // (op 1301) 
buffer_load_dwordx4 v38, s[64:67], 0 offen offset: 0 lds // (op 1301) Load value direct to lds
v_add_co_u32 v38, s[18:19], v38, v32 // (op 1301) least significant half
v_addc_co_u32 v39, s[18:19], 0, v39, s[18:19] // (op 1301) most significant half

s_add_u32 m0, m0, 1024 // (op 1301) 
buffer_load_dwordx4 v38, s[64:67], 0 offen offset: 0 lds // (op 1301) Load value direct to lds
v_add_co_u32 v38, s[18:19], v38, v32 // (op 1301) least significant half
v_addc_co_u32 v39, s[18:19], 0, v39, s[18:19] // (op 1301) most significant half

s_add_u32 m0, m0, 1024 // (op 1301) 
buffer_load_dwordx4 v38, s[64:67], 0 offen offset: 0 lds // (op 1301) Load value direct to lds






v_mov_b64 v[20:21], s[38:39] // (op 1110) convert

s_lshl_b32 s15, s38, 3 // (op 1110) 
v_add_u32 v7, v29, s15 // (op 1110) 

buffer_load_ubyte v32, v7, s[72:75], 0 offen offset: 0 // (op 1110) Load value
v_add_u32 v7, v7, v20 // (op 1110) 
buffer_load_ubyte v33, v7, s[72:75], 0 offen offset: 0 // (op 1110) Load value



v_lshlrev_b32 v7, 4, v1 // (op 2511) 
s_nop 0
v_readfirstlane_b32 s15, v7 // (op 2511) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 3009) 

s_lshl_b64 s[18:19], s[24:25], 6 // (op 1303) 
v_mov_b64 v[20:21], s[18:19] // (op 1303) convert


s_lshl_b64 s[18:19], s[24:25], 8 // (op 1303) 
v_mov_b32 v7, s18 // (op 1303) 
v_mov_b32 v31, s19 // (op 1303) 
v_add_co_u32 v28, s[6:7], v22, v7 // (op 1303) least significant half
v_addc_co_u32 v29, s[6:7], v23, v31, s[6:7] // (op 1303) most significant half




s_add_u32 s6, 12288, s15 // (op 1303) 
s_nop 0
s_mov_b32 m0, s6 // (op 1303) reg expression
buffer_load_dwordx4 v28, s[60:63], 0 offen offset: 0 lds // (op 1303) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v20 // (op 1303) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 1303) most significant half

s_add_u32 m0, m0, 1024 // (op 1303) 
buffer_load_dwordx4 v28, s[60:63], 0 offen offset: 0 lds // (op 1303) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v20 // (op 1303) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 1303) most significant half

s_add_u32 m0, m0, 1024 // (op 1303) 
buffer_load_dwordx4 v28, s[60:63], 0 offen offset: 0 lds // (op 1303) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v20 // (op 1303) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 1303) most significant half

s_add_u32 m0, m0, 1024 // (op 1303) 
buffer_load_dwordx4 v28, s[60:63], 0 offen offset: 0 lds // (op 1303) Load value direct to lds






v_mov_b64 v[28:29], s[32:33] // (op 1116) convert

s_lshl_b32 s15, s32, 3 // (op 1116) 
v_add_u32 v7, v14, s15 // (op 1116) 

buffer_load_ubyte v20, v7, s[68:71], 0 offen offset: 0 // (op 1116) Load value
v_add_u32 v7, v7, v28 // (op 1116) 
buffer_load_ubyte v21, v7, s[68:71], 0 offen offset: 0 // (op 1116) Load value



v_and_b32 v7, -16, v1 // (op 2425) 
v_sub_u32 v14, v1, v7 // (op 2425) 

v_lshrrev_b32 v7, 4, v1 // (op 2425) 
v_lshl_add_u32 v31, v7, 5, v14 // (op 2425) 


s_barrier  // (op 2989) 



s_waitcnt vmcnt(19)
ds_write_b8 v31, v34 offset:16384 // (op 1119) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v31, v35 offset:16400 // (op 1119) Store local data 

v_and_b32 v7, -16, v1 // (op 2359) 
v_sub_u32 v14, v1, v7 // (op 2359) 

v_lshrrev_b32 v7, 4, v1 // (op 2359) 
v_lshl_add_u32 v31, v7, 5, v14 // (op 2359) 


s_barrier  // (op 2969) 



s_waitcnt vmcnt(13)
ds_write_b8 v31, v36 offset:16512 // (op 1121) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v31, v37 offset:16528 // (op 1121) Store local data 

s_barrier  // (op 2946) 



ds_read_b64_tr_b8 v[38:39], v5 // (op 1141) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v5 offset:128 // (op 1141) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v5 offset:1024 // (op 1141) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v5 offset:1152 // (op 1141) Transpose load local data 



ds_read_b64_tr_b8 v[46:47], v5 offset:2048 // (op 1144) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v5 offset:2176 // (op 1144) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v5 offset:3072 // (op 1144) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v5 offset:3200 // (op 1144) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 2986) 


ds_read_u8 v7, v17 offset:16384 // (op 1153) Load local data 


ds_read_u8 v14, v17 offset:16448 // (op 1156) Load local data 
s_barrier  // (op 2927) 



ds_read_b64_tr_b8 v[54:55], v0 offset:4096 // (op 1135) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v0 offset:4224 // (op 1135) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v0 offset:5120 // (op 1135) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v0 offset:5248 // (op 1135) Transpose load local data 



ds_read_b64_tr_b8 v[62:63], v0 offset:6144 // (op 1138) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v0 offset:6272 // (op 1138) Transpose load local data 
ds_read_b64_tr_b8 v[66:67], v0 offset:7168 // (op 1138) Transpose load local data 
ds_read_b64_tr_b8 v[68:69], v0 offset:7296 // (op 1138) Transpose load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 2966) 


ds_read_u8 v31, v12 offset:16512 // (op 1147) Load local data 


ds_read_u8 v22, v12 offset:16576 // (op 1150) Load local data 
v_lshlrev_b32 v23, 4, v1 // (op 2200) 
s_nop 0
v_readfirstlane_b32 s15, v23 // (op 2200) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v23, 4, v1 // (op 2265) 
s_nop 0
v_readfirstlane_b32 s6, v23 // (op 2265) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v23, -16, v1 // (op 2330) 
v_sub_u32 v28, v1, v23 // (op 2330) 

v_lshrrev_b32 v23, 4, v1 // (op 2330) 
v_lshl_add_u32 v29, v23, 5, v28 // (op 2330) 


v_and_b32 v23, -16, v1 // (op 2396) 
v_sub_u32 v28, v1, v23 // (op 2396) 

v_lshrrev_b32 v23, 4, v1 // (op 2396) 
v_lshl_add_u32 v70, v23, 5, v28 // (op 2396) 


v_lshlrev_b32 v23, 4, v1 // (op 2496) 
s_nop 0
v_readfirstlane_b32 s7, v23 // (op 2496) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v23, 4, v1 // (op 2535) 
s_nop 0
v_readfirstlane_b32 s8, v23 // (op 2535) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v23, -16, v1 // (op 2574) 
v_sub_u32 v28, v1, v23 // (op 2574) 

v_lshrrev_b32 v23, 4, v1 // (op 2574) 
v_lshl_add_u32 v71, v23, 5, v28 // (op 2574) 


v_and_b32 v23, -16, v1 // (op 2598) 
v_sub_u32 v28, v1, v23 // (op 2598) 

v_lshrrev_b32 v23, 4, v1 // (op 2598) 
v_lshl_add_u32 v72, v23, 5, v28 // (op 2598) 


s_mov_b32 s1, 0 // (op 208) call()
s_ashr_i64 s[18:19], s[2:3], 8 // (op 207) 
s_mov_b32 s42, s18 // (op 207) convert

s_and_b32 s18, s42, 4294967294 // (op 207) 

s_cmp_lt_u32 s1, s18 // (op 207) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopBottom_KLoop_207 // (op 207) Condition: Top (jump to RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopBottom_KLoop_207 if false)
RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopTop_KLoop_207:
s_add_u32 s1, s1, 2 // (op 209) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[54:61], v[38:45], a[0:3], v31, v7 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 220) 
s_barrier  // (op 2952) 

s_lshl_b64 s[18:19], s[28:29], 6 // (op 1226) 
v_mov_b64 v[74:75], s[18:19] // (op 1226) convert


v_mov_b64 v[76:77], v[26:27] // (op 1226) 
s_lshl_b64 s[18:19], s[28:29], 8 // (op 1226) 
s_lshl_b64 s[42:43], s[18:19], 1 // (op 1226) 

v_mov_b32 v23, s42 // (op 1226) 
v_mov_b32 v28, s43 // (op 1226) 
v_add_co_u32 v78, s[18:19], v76, v23 // (op 1226) least significant half
v_addc_co_u32 v79, s[18:19], v77, v28, s[18:19] // (op 1226) most significant half




s_mov_b32 m0, s6 // (op 1226) reg expression
buffer_load_dwordx4 v78, s[64:67], 0 offen offset: 0 lds // (op 1226) Load value direct to lds
v_add_co_u32 v78, s[18:19], v78, v74 // (op 1226) least significant half
v_addc_co_u32 v79, s[18:19], 0, v79, s[18:19] // (op 1226) most significant half

s_add_u32 m0, m0, 1024 // (op 1226) 
buffer_load_dwordx4 v78, s[64:67], 0 offen offset: 0 lds // (op 1226) Load value direct to lds
v_add_co_u32 v78, s[18:19], v78, v74 // (op 1226) least significant half
v_addc_co_u32 v79, s[18:19], 0, v79, s[18:19] // (op 1226) most significant half

s_add_u32 m0, m0, 1024 // (op 1226) 
buffer_load_dwordx4 v78, s[64:67], 0 offen offset: 0 lds // (op 1226) Load value direct to lds
v_add_co_u32 v78, s[18:19], v78, v74 // (op 1226) least significant half
v_addc_co_u32 v79, s[18:19], 0, v79, s[18:19] // (op 1226) most significant half

s_add_u32 m0, m0, 1024 // (op 1226) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:69], v[46:53], a[0:3], v22, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 229) 
buffer_load_dwordx4 v78, s[64:67], 0 offen offset: 0 lds // (op 1226) Load value direct to lds



v_mov_b64 v[74:75], s[38:39] // (op 235) convert

v_mov_b32 v23, v30 // (op 235) 
s_lshl_b32 s18, s38, 4 // (op 235) 
v_add_u32 v28, v23, s18 // (op 235) 

buffer_load_ubyte v34, v28, s[72:75], 0 offen offset: 0 // (op 235) Load value
v_add_u32 v28, v28, v74 // (op 235) 
buffer_load_ubyte v35, v28, s[72:75], 0 offen offset: 0 // (op 235) Load value


s_barrier  // (op 2933) 

s_lshl_b64 s[18:19], s[24:25], 6 // (op 1221) 
v_mov_b64 v[74:75], s[18:19] // (op 1221) convert


v_mov_b64 v[78:79], v[24:25] // (op 1221) 
s_lshl_b64 s[18:19], s[24:25], 8 // (op 1221) 
s_lshl_b64 s[42:43], s[18:19], 1 // (op 1221) 

v_mov_b32 v28, s42 // (op 1221) 
v_mov_b32 v73, s43 // (op 1221) 
v_add_co_u32 v80, s[18:19], v78, v28 // (op 1221) least significant half
v_addc_co_u32 v81, s[18:19], v79, v73, s[18:19] // (op 1221) most significant half




s_add_u32 s18, 4096, s15 // (op 1221) 
s_nop 0
s_mov_b32 m0, s18 // (op 1221) reg expression
buffer_load_dwordx4 v80, s[60:63], 0 offen offset: 0 lds // (op 1221) Load value direct to lds
v_add_co_u32 v80, s[42:43], v80, v74 // (op 1221) least significant half
v_addc_co_u32 v81, s[42:43], 0, v81, s[42:43] // (op 1221) most significant half

s_add_u32 m0, m0, 1024 // (op 1221) 
buffer_load_dwordx4 v80, s[60:63], 0 offen offset: 0 lds // (op 1221) Load value direct to lds
v_add_co_u32 v80, s[42:43], v80, v74 // (op 1221) least significant half
v_addc_co_u32 v81, s[42:43], 0, v81, s[42:43] // (op 1221) most significant half

s_add_u32 m0, m0, 1024 // (op 1221) 
buffer_load_dwordx4 v80, s[60:63], 0 offen offset: 0 lds // (op 1221) Load value direct to lds
v_add_co_u32 v80, s[42:43], v80, v74 // (op 1221) least significant half
v_addc_co_u32 v81, s[42:43], 0, v81, s[42:43] // (op 1221) most significant half

s_add_u32 m0, m0, 1024 // (op 1221) 
buffer_load_dwordx4 v80, s[60:63], 0 offen offset: 0 lds // (op 1221) Load value direct to lds




v_mov_b64 v[74:75], s[32:33] // (op 234) convert

v_mov_b32 v28, v19 // (op 234) 
s_lshl_b32 s18, s32, 4 // (op 234) 
v_add_u32 v73, v28, s18 // (op 234) 

buffer_load_ubyte v36, v73, s[68:71], 0 offen offset: 0 // (op 234) Load value
v_add_u32 v73, v73, v74 // (op 234) 
buffer_load_ubyte v37, v73, s[68:71], 0 offen offset: 0 // (op 234) Load value


s_barrier  // (op 3034) 



s_waitcnt vmcnt(19)
ds_write_b8 v72, v32 offset:16640 // (op 608) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v72, v33 offset:16656 // (op 608) Store local data 
s_barrier  // (op 3028) 



s_waitcnt vmcnt(13)
ds_write_b8 v71, v20 offset:16768 // (op 607) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v71, v21 offset:16784 // (op 607) Store local data 
s_barrier  // (op 3015) 



ds_read_b64_tr_b8 v[38:39], v10 offset:8192 // (op 591) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v10 offset:8320 // (op 591) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v10 offset:9216 // (op 591) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v10 offset:9344 // (op 591) Transpose load local data 



ds_read_b64_tr_b8 v[46:47], v10 offset:10240 // (op 600) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v10 offset:10368 // (op 600) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v10 offset:11264 // (op 600) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v10 offset:11392 // (op 600) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 3031) 


ds_read_u8 v7, v15 offset:16640 // (op 595) Load local data 


ds_read_u8 v14, v15 offset:16704 // (op 604) Load local data 
s_barrier  // (op 3006) 
s_barrier  // (op 3174) 



ds_read_b64_tr_b8 v[54:55], v8 offset:12288 // (op 589) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v8 offset:12416 // (op 589) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v8 offset:13312 // (op 589) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v8 offset:13440 // (op 589) Transpose load local data 



ds_read_b64_tr_b8 v[62:63], v8 offset:14336 // (op 598) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v8 offset:14464 // (op 598) Transpose load local data 
ds_read_b64_tr_b8 v[66:67], v8 offset:15360 // (op 598) Transpose load local data 
ds_read_b64_tr_b8 v[68:69], v8 offset:15488 // (op 598) Transpose load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 3025) 


ds_read_u8 v31, v13 offset:16768 // (op 593) Load local data 


ds_read_u8 v22, v13 offset:16832 // (op 602) Load local data 
s_barrier  // (op 3170) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[54:61], v[38:45], a[0:3], v31, v7 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 596) 
s_barrier  // (op 3022) 

s_lshl_b64 s[18:19], s[28:29], 6 // (op 1249) 
v_mov_b64 v[74:75], s[18:19] // (op 1249) convert


s_lshl_b64 s[18:19], s[28:29], 8 // (op 1249) 
s_mul_i32 s43, 3, s19 // (op 1249) most significant: low of low * high
s_mul_i32 s42, 3, s18 // (op 1249) least significant
s_mul_hi_u32 s44, 3, s18 // (op 1249) most significant: high of low * low
s_add_u32 s43, s43, s44 // (op 1249) most significant: sum


v_mov_b32 v73, s42 // (op 1249) 
v_mov_b32 v80, s43 // (op 1249) 
v_add_co_u32 v82, s[18:19], v76, v73 // (op 1249) least significant half
v_addc_co_u32 v83, s[18:19], v77, v80, s[18:19] // (op 1249) most significant half




s_add_u32 s18, 8192, s8 // (op 1249) 
s_nop 0
s_mov_b32 m0, s18 // (op 1249) reg expression
buffer_load_dwordx4 v82, s[64:67], 0 offen offset: 0 lds // (op 1249) Load value direct to lds
v_add_co_u32 v82, s[42:43], v82, v74 // (op 1249) least significant half
v_addc_co_u32 v83, s[42:43], 0, v83, s[42:43] // (op 1249) most significant half

s_add_u32 m0, m0, 1024 // (op 1249) 
buffer_load_dwordx4 v82, s[64:67], 0 offen offset: 0 lds // (op 1249) Load value direct to lds
v_add_co_u32 v82, s[42:43], v82, v74 // (op 1249) least significant half
v_addc_co_u32 v83, s[42:43], 0, v83, s[42:43] // (op 1249) most significant half

s_add_u32 m0, m0, 1024 // (op 1249) 
buffer_load_dwordx4 v82, s[64:67], 0 offen offset: 0 lds // (op 1249) Load value direct to lds
v_add_co_u32 v82, s[42:43], v82, v74 // (op 1249) least significant half
v_addc_co_u32 v83, s[42:43], 0, v83, s[42:43] // (op 1249) most significant half

s_add_u32 m0, m0, 1024 // (op 1249) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:69], v[46:53], a[0:3], v22, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 605) 
buffer_load_dwordx4 v82, s[64:67], 0 offen offset: 0 lds // (op 1249) Load value direct to lds





v_mov_b64 v[76:77], s[38:39] // (op 611) convert

s_lshl_b32 s18, s38, 3 // (op 611) 
s_mul_i32 s19, 3, s18 // (op 611) 

v_add_u32 v73, v23, s19 // (op 611) 

buffer_load_ubyte v32, v73, s[72:75], 0 offen offset: 0 // (op 611) Load value
v_add_u32 v73, v73, v76 // (op 611) 
s_lshl_b64 s[18:19], s[28:29], 8 // (op 1550) 
buffer_load_ubyte v33, v73, s[72:75], 0 offen offset: 0 // (op 611) Load value


s_lshl_b64 s[42:43], s[18:19], 1 // (op 1550) 


v_mov_b32 v23, s42 // (op 1550) 
s_barrier  // (op 3012) 
s_lshl_b32 s18, s38, 4 // (op 1806) 

v_add_u32 v30, v30, s18 // (op 1806) 

v_mov_b32 v73, s43 // (op 1550) 
s_lshl_b64 s[18:19], s[24:25], 6 // (op 1244) 
v_mov_b64 v[74:75], s[18:19] // (op 1244) convert


s_lshl_b64 s[18:19], s[24:25], 8 // (op 1244) 
s_mul_i32 s45, 3, s19 // (op 1244) most significant: low of low * high
s_mul_i32 s44, 3, s18 // (op 1244) least significant
s_mul_hi_u32 s52, 3, s18 // (op 1244) most significant: high of low * low
s_add_u32 s45, s45, s52 // (op 1244) most significant: sum


v_mov_b32 v76, s44 // (op 1244) 
v_mov_b32 v77, s45 // (op 1244) 
v_add_co_u32 v26, s[18:19], v26, v23 // (op 1550) least significant half
v_addc_co_u32 v27, s[18:19], v27, v73, s[18:19] // (op 1550) most significant half




v_add_co_u32 v80, s[18:19], v78, v76 // (op 1244) least significant half
v_addc_co_u32 v81, s[18:19], v79, v77, s[18:19] // (op 1244) most significant half




s_add_u32 s18, 12288, s7 // (op 1244) 
s_nop 0
s_mov_b32 m0, s18 // (op 1244) reg expression
buffer_load_dwordx4 v80, s[60:63], 0 offen offset: 0 lds // (op 1244) Load value direct to lds
v_add_co_u32 v80, s[42:43], v80, v74 // (op 1244) least significant half
v_addc_co_u32 v81, s[42:43], 0, v81, s[42:43] // (op 1244) most significant half

s_add_u32 m0, m0, 1024 // (op 1244) 
buffer_load_dwordx4 v80, s[60:63], 0 offen offset: 0 lds // (op 1244) Load value direct to lds
v_add_co_u32 v80, s[42:43], v80, v74 // (op 1244) least significant half
v_addc_co_u32 v81, s[42:43], 0, v81, s[42:43] // (op 1244) most significant half

s_add_u32 m0, m0, 1024 // (op 1244) 
buffer_load_dwordx4 v80, s[60:63], 0 offen offset: 0 lds // (op 1244) Load value direct to lds
v_add_co_u32 v80, s[42:43], v80, v74 // (op 1244) least significant half
v_addc_co_u32 v81, s[42:43], 0, v81, s[42:43] // (op 1244) most significant half

s_add_u32 m0, m0, 1024 // (op 1244) 
buffer_load_dwordx4 v80, s[60:63], 0 offen offset: 0 lds // (op 1244) Load value direct to lds





v_mov_b64 v[74:75], s[32:33] // (op 610) convert

s_lshl_b32 s18, s32, 3 // (op 610) 
s_mul_i32 s19, 3, s18 // (op 610) 

v_add_u32 v23, v28, s19 // (op 610) 

buffer_load_ubyte v20, v23, s[68:71], 0 offen offset: 0 // (op 610) Load value
v_add_u32 v23, v23, v74 // (op 610) 
s_lshl_b64 s[18:19], s[24:25], 8 // (op 1422) 
s_lshl_b64 s[42:43], s[18:19], 1 // (op 1422) 

buffer_load_ubyte v21, v23, s[68:71], 0 offen offset: 0 // (op 610) Load value


v_mov_b32 v23, s42 // (op 1422) 

v_mov_b32 v28, s43 // (op 1422) 
s_barrier  // (op 2992) 
s_lshl_b32 s18, s32, 4 // (op 1678) 



v_add_u32 v19, v19, s18 // (op 1678) 

v_add_co_u32 v24, s[18:19], v24, v23 // (op 1422) least significant half
v_addc_co_u32 v25, s[18:19], v25, v28, s[18:19] // (op 1422) most significant half




s_waitcnt vmcnt(19)
ds_write_b8 v70, v34 offset:16384 // (op 232) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v70, v35 offset:16400 // (op 232) Store local data 
s_barrier  // (op 2972) 



s_waitcnt vmcnt(13)
ds_write_b8 v29, v36 offset:16512 // (op 231) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v29, v37 offset:16528 // (op 231) Store local data 
s_barrier  // (op 2958) 



ds_read_b64_tr_b8 v[38:39], v6 // (op 215) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v6 offset:128 // (op 215) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v6 offset:1024 // (op 215) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v6 offset:1152 // (op 215) Transpose load local data 



ds_read_b64_tr_b8 v[46:47], v6 offset:2048 // (op 224) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v6 offset:2176 // (op 224) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v6 offset:3072 // (op 224) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v6 offset:3200 // (op 224) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 2999) 


ds_read_u8 v7, v11 offset:16384 // (op 219) Load local data 


ds_read_u8 v14, v11 offset:16448 // (op 228) Load local data 
s_barrier  // (op 2939) 



ds_read_b64_tr_b8 v[54:55], v4 offset:4096 // (op 213) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v4 offset:4224 // (op 213) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v4 offset:5120 // (op 213) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v4 offset:5248 // (op 213) Transpose load local data 



ds_read_b64_tr_b8 v[62:63], v4 offset:6144 // (op 222) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v4 offset:6272 // (op 222) Transpose load local data 
ds_read_b64_tr_b8 v[66:67], v4 offset:7168 // (op 222) Transpose load local data 
ds_read_b64_tr_b8 v[68:69], v4 offset:7296 // (op 222) Transpose load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 2979) 


ds_read_u8 v31, v9 offset:16512 // (op 217) Load local data 


ds_read_u8 v22, v9 offset:16576 // (op 226) Load local data 
s_ashr_i64 s[18:19], s[2:3], 8 // (op 207) 
s_mov_b32 s42, s18 // (op 207) convert

s_and_b32 s18, s42, 4294967294 // (op 207) 

s_cmp_lt_u32 s1, s18 // (op 207) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopTop_KLoop_207 // (op 207) Condition: Bottom (jump to RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopTop_KLoop_207 if true)
RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopBottom_KLoop_207:















s_barrier  // (op 3092) 
s_ashr_i64 s[18:19], s[2:3], 8 // (op 1316) 
s_and_b32 s15, s18, 4294967294 // (op 1316) 
s_add_u32 s6, s0, s15 // (op 1316) 

s_mul_i32 s15, s16, s50 // (op 1316) 
s_add_u32 s15, s15, s6 // (op 1316) 

s_add_u32 s6, s15, s9 // (op 1316) 

s_mul_hi_u32 s15, s6, s13 // (op 1316) 
s_sub_u32 s7, s6, s15 // (op 1316) 
s_lshr_b32 s8, s7, 1 // (op 1316) 

s_add_u32 s7, s8, s15 // (op 1316) 


s_lshr_b32 s8, s7, s11 // (op 1316) 

s_and_b32 s7, s11, 2147483648 // (op 1316) 
s_cmp_eq_i32 s7, 0 // (op 1316) 
s_mov_b32 s15, scc // (op 1316) 

s_cmp_eq_i32 s15, 0 // (op 1316) 
s_cselect_b32 s7, s6, s8 // (op 1316) 


s_mul_hi_u32 s8, s7, s12 // (op 1316) 
s_sub_u32 s15, s7, s8 // (op 1316) 
s_lshr_b32 s42, s15, 1 // (op 1316) 

s_add_u32 s15, s42, s8 // (op 1316) 


s_lshr_b32 s8, s15, s10 // (op 1316) 

s_and_b32 s15, s10, 2147483648 // (op 1316) 
s_cmp_eq_i32 s15, 0 // (op 1316) 
s_mov_b32 s42, scc // (op 1316) 

s_cmp_eq_i32 s42, 0 // (op 1316) 
s_cselect_b32 s15, s7, s8 // (op 1316) 


s_lshl_b32 s8, s15, 4 // (op 1316) 

s_mov_b32 s43, 0 // (op 1316) convert
s_mov_b32 s42, s8 // (op 1316) convert

s_mul_i32 s8, s7, s18 // (op 1316) 


s_sub_u32 s7, s6, s8 // (op 1316) 


v_lshl_add_u32 v19, s7, 8, v1 // (op 1316) 

v_mul_lo_u32 v35, v19, s25 // (op 1316) most significant: low of low * high
v_mul_hi_u32 v23, v19, s24 // (op 1316) most significant: high of low * low
v_mul_lo_u32 v34, v19, s24 // (op 1316) least significant: low of low * low
v_add_u32 v35, v35, v23 // (op 1316) most significant: sum


v_mov_b32 v19, s42 // (op 1316) 
v_mov_b32 v23, s43 // (op 1316) 
v_add_co_u32 v24, s[18:19], v19, v34 // (op 1316) least significant half
v_addc_co_u32 v25, s[18:19], v23, v35, s[18:19] // (op 1316) most significant half





s_mul_i32 s19, s24, s3 // (op 1318) most significant: low of low * high
s_mul_i32 s15, s25, s2 // (op 1318) most significant: low of high * low
s_mul_i32 s18, s24, s2 // (op 1318) least significant
s_mul_hi_u32 s6, s24, s2 // (op 1318) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1318) most significant: sum
s_add_u32 s19, s19, s6 // (op 1318) most significant: sum


s_mov_b32 s60, s26 // (op 1318) reg expression
s_mov_b32 s61, s27 // (op 1318) reg expression
s_mov_b32 s62, s18 // (op 1318) reg expression
s_mov_b32 s63, 131072 // (op 1318) call()

s_ashr_i64 s[18:19], s[2:3], 8 // (op 1452) 
s_and_b32 s15, s18, 4294967294 // (op 1452) 
s_add_u32 s6, s0, s15 // (op 1452) 

s_mul_i32 s15, s16, s50 // (op 1452) 
s_add_u32 s15, s15, s6 // (op 1452) 

s_add_u32 s6, s15, s9 // (op 1452) 

s_mul_hi_u32 s15, s6, s13 // (op 1452) 
s_sub_u32 s7, s6, s15 // (op 1452) 
s_lshr_b32 s8, s7, 1 // (op 1452) 

s_add_u32 s7, s8, s15 // (op 1452) 


s_lshr_b32 s8, s7, s11 // (op 1452) 

s_and_b32 s7, s11, 2147483648 // (op 1452) 
s_cmp_eq_i32 s7, 0 // (op 1452) 
s_mov_b32 s15, scc // (op 1452) 

s_cmp_eq_i32 s15, 0 // (op 1452) 
s_cselect_b32 s7, s6, s8 // (op 1452) 


s_mul_hi_u32 s8, s7, s12 // (op 1452) 
s_sub_u32 s15, s7, s8 // (op 1452) 
s_lshr_b32 s42, s15, 1 // (op 1452) 

s_add_u32 s15, s42, s8 // (op 1452) 


s_lshr_b32 s8, s15, s10 // (op 1452) 

s_and_b32 s15, s10, 2147483648 // (op 1452) 
s_cmp_eq_i32 s15, 0 // (op 1452) 
s_mov_b32 s42, scc // (op 1452) 

s_cmp_eq_i32 s42, 0 // (op 1452) 
s_cselect_b32 s15, s7, s8 // (op 1452) 


s_mul_i32 s8, s15, s14 // (op 1452) 

s_sub_u32 s15, s7, s8 // (op 1452) 

s_lshl_b32 s8, s15, 4 // (op 1452) 

s_mov_b32 s43, 0 // (op 1452) convert
s_mov_b32 s42, s8 // (op 1452) convert

s_mul_i32 s8, s7, s18 // (op 1452) 


s_sub_u32 s7, s6, s8 // (op 1452) 


v_lshl_add_u32 v19, s7, 8, v1 // (op 1452) 

v_mul_lo_u32 v27, v19, s29 // (op 1452) most significant: low of low * high
v_mul_hi_u32 v23, v19, s28 // (op 1452) most significant: high of low * low
v_mul_lo_u32 v26, v19, s28 // (op 1452) least significant: low of low * low
v_add_u32 v27, v27, v23 // (op 1452) most significant: sum

v_mov_b32 v23, s42 // (op 1452) 
v_mov_b32 v28, s43 // (op 1452) 
v_add_co_u32 v26, s[18:19], v26, v23 // (op 1452) least significant half
v_addc_co_u32 v27, s[18:19], v27, v28, s[18:19] // (op 1452) most significant half





s_mul_i32 s19, s28, s3 // (op 1454) most significant: low of low * high
s_mul_i32 s15, s29, s2 // (op 1454) most significant: low of high * low
s_mul_i32 s18, s28, s2 // (op 1454) least significant
s_mul_hi_u32 s6, s28, s2 // (op 1454) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1454) most significant: sum
s_add_u32 s19, s19, s6 // (op 1454) most significant: sum


s_mov_b32 s64, s30 // (op 1454) reg expression
s_mov_b32 s65, s31 // (op 1454) reg expression
s_mov_b32 s66, s18 // (op 1454) reg expression
s_mov_b32 s67, 131072 // (op 1454) call()

v_and_b32 v19, -16, v1 // (op 1580) 
s_ashr_i64 s[18:19], s[2:3], 8 // (op 1580) 
v_sub_u32 v23, v1, v19 // (op 1580) 

s_and_b32 s15, s18, 4294967294 // (op 1580) 
v_lshrrev_b32 v19, 4, v1 // (op 1580) 
s_add_u32 s6, s0, s15 // (op 1580) 

v_lshlrev_b32 v28, 1, v19 // (op 1580) 

s_mul_i32 s15, s16, s50 // (op 1580) 
s_add_u32 s15, s15, s6 // (op 1580) 

s_add_u32 s6, s15, s9 // (op 1580) 

s_mul_hi_u32 s15, s6, s13 // (op 1580) 
s_sub_u32 s7, s6, s15 // (op 1580) 
s_lshr_b32 s8, s7, 1 // (op 1580) 

s_add_u32 s7, s8, s15 // (op 1580) 


s_lshr_b32 s8, s7, s11 // (op 1580) 

s_and_b32 s7, s11, 2147483648 // (op 1580) 
s_cmp_eq_i32 s7, 0 // (op 1580) 
s_mov_b32 s15, scc // (op 1580) 

s_cmp_eq_i32 s15, 0 // (op 1580) 
s_cselect_b32 s7, s6, s8 // (op 1580) 


s_mul_hi_u32 s8, s7, s12 // (op 1580) 
s_sub_u32 s15, s7, s8 // (op 1580) 
s_lshr_b32 s42, s15, 1 // (op 1580) 

s_add_u32 s15, s42, s8 // (op 1580) 


s_lshr_b32 s8, s15, s10 // (op 1580) 

s_and_b32 s15, s10, 2147483648 // (op 1580) 
s_cmp_eq_i32 s15, 0 // (op 1580) 
s_mov_b32 s42, scc // (op 1580) 

s_cmp_eq_i32 s42, 0 // (op 1580) 
s_cselect_b32 s15, s7, s8 // (op 1580) 


s_mul_i32 s8, s7, s18 // (op 1580) 


v_lshl_add_u32 v19, s15, 4, v23 // (op 1580) 


s_sub_u32 s7, s6, s8 // (op 1580) 


v_lshl_add_u32 v23, s7, 3, v28 // (op 1580) 


v_mul_lo_u32 v28, v23, s32 // (op 1580) 

v_add_u32 v23, v19, v28 // (op 1580) 


s_mul_i32 s19, s32, s37 // (op 1582) most significant: low of low * high
s_mul_i32 s15, s33, s36 // (op 1582) most significant: low of high * low
s_mul_i32 s18, s32, s36 // (op 1582) least significant
s_mul_hi_u32 s6, s32, s36 // (op 1582) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1582) most significant: sum
s_add_u32 s19, s19, s6 // (op 1582) most significant: sum


s_mov_b32 s68, s34 // (op 1582) reg expression
s_mov_b32 s69, s35 // (op 1582) reg expression
s_mov_b32 s70, s18 // (op 1582) reg expression
s_mov_b32 s71, 131072 // (op 1582) call()

v_and_b32 v19, -16, v1 // (op 1708) 
s_ashr_i64 s[18:19], s[2:3], 8 // (op 1708) 
v_sub_u32 v28, v1, v19 // (op 1708) 

s_and_b32 s15, s18, 4294967294 // (op 1708) 
v_lshrrev_b32 v19, 4, v1 // (op 1708) 
s_add_u32 s6, s0, s15 // (op 1708) 

v_lshlrev_b32 v29, 1, v19 // (op 1708) 

s_mul_i32 s15, s16, s50 // (op 1708) 
s_add_u32 s15, s15, s6 // (op 1708) 

s_add_u32 s6, s15, s9 // (op 1708) 

s_mul_hi_u32 s15, s6, s13 // (op 1708) 
s_sub_u32 s7, s6, s15 // (op 1708) 
s_lshr_b32 s8, s7, 1 // (op 1708) 

s_add_u32 s7, s8, s15 // (op 1708) 


s_lshr_b32 s8, s7, s11 // (op 1708) 

s_and_b32 s7, s11, 2147483648 // (op 1708) 
s_cmp_eq_i32 s7, 0 // (op 1708) 
s_mov_b32 s15, scc // (op 1708) 

s_cmp_eq_i32 s15, 0 // (op 1708) 
s_cselect_b32 s7, s6, s8 // (op 1708) 


s_mul_hi_u32 s8, s7, s12 // (op 1708) 
s_sub_u32 s15, s7, s8 // (op 1708) 
s_lshr_b32 s42, s15, 1 // (op 1708) 

s_add_u32 s15, s42, s8 // (op 1708) 


s_lshr_b32 s8, s15, s10 // (op 1708) 

s_and_b32 s15, s10, 2147483648 // (op 1708) 
s_cmp_eq_i32 s15, 0 // (op 1708) 
s_mov_b32 s42, scc // (op 1708) 

s_cmp_eq_i32 s42, 0 // (op 1708) 
s_cselect_b32 s15, s7, s8 // (op 1708) 


s_mul_i32 s8, s15, s14 // (op 1708) 

s_sub_u32 s15, s7, s8 // (op 1708) 

s_mul_i32 s8, s7, s18 // (op 1708) 


v_lshl_add_u32 v19, s15, 4, v28 // (op 1708) 


s_sub_u32 s7, s6, s8 // (op 1708) 


v_lshl_add_u32 v28, s7, 3, v29 // (op 1708) 


v_mul_lo_u32 v29, v28, s38 // (op 1708) 
v_add_u32 v29, v29, v19 // (op 1708) 


s_mul_i32 s19, s38, s37 // (op 1710) most significant: low of low * high
s_mul_i32 s15, s39, s36 // (op 1710) most significant: low of high * low
s_mul_i32 s18, s38, s36 // (op 1710) least significant
s_mul_hi_u32 s6, s38, s36 // (op 1710) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1710) most significant: sum
s_add_u32 s19, s19, s6 // (op 1710) most significant: sum


s_mov_b32 s72, s40 // (op 1710) reg expression
s_mov_b32 s73, s41 // (op 1710) reg expression
s_mov_b32 s74, s18 // (op 1710) reg expression
s_mov_b32 s75, 131072 // (op 1710) call()

v_lshlrev_b32 v19, 4, v1 // (op 2213) 
s_nop 0
v_readfirstlane_b32 s15, v19 // (op 2213) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v19, 4, v1 // (op 2278) 
s_nop 0
v_readfirstlane_b32 s6, v19 // (op 2278) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v19, -16, v1 // (op 2343) 
v_sub_u32 v28, v1, v19 // (op 2343) 

v_lshrrev_b32 v19, 4, v1 // (op 2343) 
v_lshl_add_u32 v30, v19, 5, v28 // (op 2343) 


v_and_b32 v19, -16, v1 // (op 2409) 
v_sub_u32 v28, v1, v19 // (op 2409) 

v_lshrrev_b32 v19, 4, v1 // (op 2409) 
v_lshl_add_u32 v34, v19, 5, v28 // (op 2409) 


s_ashr_i64 s[18:19], s[2:3], 8 // (op 498) 
s_and_b32 s7, s18, 4294967294 // (op 498) 

s_ashr_i64 s[18:19], s[2:3], 8 // (op 497) 
s_mov_b32 s8, s18 // (op 497) convert

s_cmp_lt_u32 s7, s8 // (op 497) 










s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopBottom_KLoopTail_497 // (op 497) Condition: Top (jump to RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopBottom_KLoopTail_497 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopTop_KLoopTail_497:
s_barrier  // (op 2936) 
s_add_u32 s7, s7, 1 // (op 499) 

s_lshl_b64 s[18:19], s[24:25], 6 // (op 1231) 
v_mov_b64 v[32:33], s[18:19] // (op 1231) convert


v_mov_b64 v[20:21], v[24:25] // (op 1231) 
v_mov_b64 v[36:37], v[20:21] // (op 1231) reg expression
s_add_u32 s8, 4096, s15 // (op 1231) 
s_mov_b32 m0, s8 // (op 1231) reg expression
buffer_load_dwordx4 v36, s[60:63], 0 offen offset: 0 lds // (op 1231) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v32 // (op 1231) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 1231) most significant half

s_add_u32 m0, m0, 1024 // (op 1231) 
buffer_load_dwordx4 v36, s[60:63], 0 offen offset: 0 lds // (op 1231) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v32 // (op 1231) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 1231) most significant half

s_add_u32 m0, m0, 1024 // (op 1231) 
buffer_load_dwordx4 v36, s[60:63], 0 offen offset: 0 lds // (op 1231) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v32 // (op 1231) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 1231) most significant half

s_add_u32 m0, m0, 1024 // (op 1231) 
buffer_load_dwordx4 v36, s[60:63], 0 offen offset: 0 lds // (op 1231) Load value direct to lds




s_lshl_b64 s[18:19], s[24:25], 8 // (op 1306) 
v_mov_b32 v7, s18 // (op 1306) 
v_mov_b32 v14, s19 // (op 1306) 
v_add_co_u32 v24, s[42:43], v24, v7 // (op 1306) least significant half
v_addc_co_u32 v25, s[42:43], v25, v14, s[42:43] // (op 1306) most significant half




s_barrier  // (op 2955) 

s_lshl_b64 s[18:19], s[28:29], 6 // (op 1237) 
v_mov_b64 v[32:33], s[18:19] // (op 1237) convert


v_mov_b64 v[20:21], v[26:27] // (op 1237) 
v_mov_b64 v[36:37], v[20:21] // (op 1237) reg expression
s_mov_b32 m0, s6 // (op 1237) reg expression
buffer_load_dwordx4 v36, s[64:67], 0 offen offset: 0 lds // (op 1237) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v32 // (op 1237) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 1237) most significant half

s_add_u32 m0, m0, 1024 // (op 1237) 
buffer_load_dwordx4 v36, s[64:67], 0 offen offset: 0 lds // (op 1237) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v32 // (op 1237) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 1237) most significant half

s_add_u32 m0, m0, 1024 // (op 1237) 
buffer_load_dwordx4 v36, s[64:67], 0 offen offset: 0 lds // (op 1237) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v32 // (op 1237) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 1237) most significant half

s_add_u32 m0, m0, 1024 // (op 1237) 
buffer_load_dwordx4 v36, s[64:67], 0 offen offset: 0 lds // (op 1237) Load value direct to lds



s_lshl_b64 s[18:19], s[28:29], 8 // (op 1444) 
v_mov_b32 v7, s18 // (op 1444) 
v_mov_b32 v14, s19 // (op 1444) 
v_add_co_u32 v26, s[42:43], v26, v7 // (op 1444) least significant half
v_addc_co_u32 v27, s[42:43], v27, v14, s[42:43] // (op 1444) most significant half





v_mov_b64 v[32:33], s[32:33] // (op 526) convert

v_mov_b32 v7, v23 // (op 526) 
v_mov_b32 v14, v7 // (op 526) reg expression
buffer_load_ubyte v20, v14, s[68:71], 0 offen offset: 0 // (op 526) Load value
v_add_u32 v14, v14, v32 // (op 526) 
buffer_load_ubyte v21, v14, s[68:71], 0 offen offset: 0 // (op 526) Load value



s_lshl_b32 s8, s32, 3 // (op 1572) 
v_add_u32 v23, v23, s8 // (op 1572) 


s_barrier  // (op 2975) 
v_mov_b64 v[32:33], s[38:39] // (op 527) convert




v_mov_b32 v7, v29 // (op 527) 
v_mov_b32 v14, v7 // (op 527) reg expression
buffer_load_ubyte v36, v14, s[72:75], 0 offen offset: 0 // (op 527) Load value
v_add_u32 v14, v14, v32 // (op 527) 
buffer_load_ubyte v37, v14, s[72:75], 0 offen offset: 0 // (op 527) Load value



s_lshl_b32 s8, s38, 3 // (op 1700) 
v_add_u32 v29, v29, s8 // (op 1700) 

s_waitcnt vmcnt(3)
ds_write_b8 v30, v20 offset:16512 // (op 523) Store local data 
s_barrier  // (op 2942) 
s_waitcnt vmcnt(2)
ds_write_b8 v30, v21 offset:16528 // (op 523) Store local data 




ds_read_b64_tr_b8 v[38:39], v4 offset:4096 // (op 505) Transpose load local data 
s_barrier  // (op 2995) 
ds_read_b64_tr_b8 v[40:41], v4 offset:4224 // (op 505) Transpose load local data 



ds_read_b64_tr_b8 v[42:43], v4 offset:5120 // (op 505) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v4 offset:5248 // (op 505) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v34, v36 offset:16384 // (op 524) Store local data 
s_barrier  // (op 2962) 
s_waitcnt vmcnt(0)
ds_write_b8 v34, v37 offset:16400 // (op 524) Store local data 




ds_read_b64_tr_b8 v[46:47], v6 // (op 507) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v6 offset:128 // (op 507) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v6 offset:1024 // (op 507) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v6 offset:1152 // (op 507) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 2982) 


ds_read_u8 v7, v9 offset:16512 // (op 509) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 3002) 


ds_read_u8 v14, v11 offset:16384 // (op 511) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[38:45], v[46:53], a[0:3], v7, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 512) 







ds_read_b64_tr_b8 v[36:37], v4 offset:6144 // (op 514) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v4 offset:6272 // (op 514) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v4 offset:7168 // (op 514) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v4 offset:7296 // (op 514) Transpose load local data 



ds_read_b64_tr_b8 v[44:45], v6 offset:2048 // (op 516) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v6 offset:2176 // (op 516) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v6 offset:3072 // (op 516) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v6 offset:3200 // (op 516) Transpose load local data 


ds_read_u8 v7, v9 offset:16576 // (op 518) Load local data 


ds_read_u8 v14, v11 offset:16448 // (op 520) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:43], v[44:51], a[0:3], v7, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 521) 




s_ashr_i64 s[18:19], s[2:3], 8 // (op 497) 
s_mov_b32 s8, s18 // (op 497) convert

s_cmp_lt_u32 s7, s8 // (op 497) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopTop_KLoopTail_497 // (op 497) Condition: Bottom (jump to RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopTop_KLoopTail_497 if true)
RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopBottom_KLoopTail_497:









s_barrier  // (op 3160) 




v_cmp_eq_f32 vcc, 0.00000, v16 // (op 278) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalFalse_0_beta_278 // (op 278) Condition: False, jump to RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalFalse_0_beta_278
s_nop 0
v_accvgpr_read v20, a0 // (op 286) Wait state hazard: XDL Write Hazard
v_accvgpr_read v21, a1 // (op 286) 
v_accvgpr_read v22, a2 // (op 286) 
v_accvgpr_read v23, a3 // (op 286) 
v_mul_f32 v24, v18, v20 // (op 286) 
v_mul_f32 v25, v18, v21 // (op 286) 
v_mul_f32 v26, v18, v22 // (op 286) 
v_mul_f32 v27, v18, v23 // (op 286) 

v_lshrrev_b32 v20, 16, v24 // (op 288) 
v_lshrrev_b32 v21, 16, v25 // (op 288) 
v_lshrrev_b32 v22, 16, v26 // (op 288) 
v_lshrrev_b32 v23, 16, v27 // (op 288) 

v_lshrrev_b32 v0, 6, v1 // (op 2166) 
s_mul_i32 s15, s16, s50 // (op 2166) 
s_add_u32 s15, s15, s0 // (op 2166) 
v_and_b32 v5, 63, v1 // (op 2166) 
s_add_u32 s6, s15, s9 // (op 2166) 

v_lshrrev_b32 v7, 4, v5 // (op 2166) 
s_mul_hi_u32 s15, s6, s13 // (op 2166) 
v_lshlrev_b32 v12, 2, v7 // (op 2166) 

s_sub_u32 s7, s6, s15 // (op 2166) 
v_lshl_add_u32 v7, v0, 4, v12 // (op 2166) 


s_lshr_b32 s8, s7, 1 // (op 2166) 

v_lshrrev_b32 v0, 2, v5 // (op 2166) 

s_add_u32 s7, s8, s15 // (op 2166) 


v_and_b32 v5, 3, v0 // (op 2166) 

s_lshr_b32 s8, s7, s11 // (op 2166) 

v_and_b32 v0, 3, v1 // (op 2166) 
s_and_b32 s7, s11, 2147483648 // (op 2166) 
v_lshl_add_u32 v12, v5, 2, v0 // (op 2166) 


s_cmp_eq_i32 s7, 0 // (op 2166) 
s_mov_b32 s15, scc // (op 2166) 

s_cmp_eq_i32 s15, 0 // (op 2166) 
s_cselect_b32 s7, s6, s8 // (op 2166) 



s_mul_hi_u32 s6, s7, s12 // (op 2166) 
s_sub_u32 s8, s7, s6 // (op 2166) 
s_lshr_b32 s15, s8, 1 // (op 2166) 

s_add_u32 s8, s15, s6 // (op 2166) 


s_lshr_b32 s6, s8, s10 // (op 2166) 

s_and_b32 s8, s10, 2147483648 // (op 2166) 
s_cmp_eq_i32 s8, 0 // (op 2166) 
s_mov_b32 s15, scc // (op 2166) 

s_cmp_eq_i32 s15, 0 // (op 2166) 
s_cselect_b32 s8, s7, s6 // (op 2166) 


s_mul_i32 s6, s8, s14 // (op 2166) 
v_lshl_add_u32 v0, s8, 4, v7 // (op 2166) 


s_sub_u32 s8, s7, s6 // (op 2166) 


v_lshl_add_u32 v5, s8, 4, v12 // (op 2166) 


v_mul_lo_u32 v7, v5, s46 // (op 2166) 

v_add_lshl_u32 v5, v0, v7, 1 // (op 2166) 


s_mul_i32 s19, s46, s23 // (op 2168) most significant: low of low * high
s_mul_i32 s15, s47, s22 // (op 2168) most significant: low of high * low
s_mul_i32 s18, s46, s22 // (op 2168) least significant
s_mul_hi_u32 s6, s46, s22 // (op 2168) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2168) most significant: sum
s_add_u32 s19, s19, s6 // (op 2168) most significant: sum


s_lshl_b64 s[6:7], s[18:19], 1 // (op 2168) 

s_mov_b32 s76, s48 // (op 2168) reg expression
s_mov_b32 s77, s49 // (op 2168) reg expression
s_mov_b32 s78, s6 // (op 2168) reg expression
s_mov_b32 s79, 131072 // (op 2168) call()


v_pack_b32_f16 v24, v20, v21 // (op 289) 
v_pack_b32_f16 v25, v22, v23 // (op 289) 


buffer_store_dwordx2 v[24:25], v5, s[76:79], 0 offen // (op 289) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalBottom_0_beta_278 // (op 278) Condition: Done, jump to RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalBottom_0_beta_278
RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalFalse_0_beta_278:



buffer_load_dwordx2 v[20:21], v3, s[56:59], 0 offen // (op 303) Load value
v_accvgpr_read v22, a0 // (op 299) 
v_accvgpr_read v23, a1 // (op 299) 
v_accvgpr_read v24, a2 // (op 299) 
v_accvgpr_read v25, a3 // (op 299) 
v_mul_f32 v26, v18, v22 // (op 299) 
v_mul_f32 v27, v18, v23 // (op 299) 
v_mul_f32 v28, v18, v24 // (op 299) 
v_mul_f32 v29, v18, v25 // (op 299) 

s_waitcnt vmcnt(0)
v_and_b32 v22, 65535, v20 // (op 304) 
v_lshrrev_b32 v23, 16, v20 // (op 304) 
v_lshlrev_b32 v22, 16, v22 // (op 304) 
v_lshlrev_b32 v23, 16, v23 // (op 304) 
v_mul_f32 v30, v16, v22 // (op 304) 
v_mul_f32 v31, v16, v23 // (op 304) 
v_and_b32 v22, 65535, v21 // (op 304) 
v_lshrrev_b32 v23, 16, v21 // (op 304) 
v_lshlrev_b32 v22, 16, v22 // (op 304) 
v_lshlrev_b32 v23, 16, v23 // (op 304) 
v_mul_f32 v32, v16, v22 // (op 304) 
v_mul_f32 v33, v16, v23 // (op 304) 


v_add_f32 v22, v30, v26 // (op 300) 
v_add_f32 v23, v31, v27 // (op 300) 
v_add_f32 v24, v32, v28 // (op 300) 
v_add_f32 v25, v33, v29 // (op 300) 


v_lshrrev_b32 v26, 16, v22 // (op 301) 
v_lshrrev_b32 v27, 16, v23 // (op 301) 
v_lshrrev_b32 v28, 16, v24 // (op 301) 
v_lshrrev_b32 v29, 16, v25 // (op 301) 

v_lshrrev_b32 v0, 6, v1 // (op 2183) 
s_mul_i32 s15, s16, s50 // (op 2183) 
s_add_u32 s15, s15, s0 // (op 2183) 
v_and_b32 v5, 63, v1 // (op 2183) 
s_add_u32 s6, s15, s9 // (op 2183) 

v_lshrrev_b32 v7, 4, v5 // (op 2183) 
s_mul_hi_u32 s15, s6, s13 // (op 2183) 
v_lshlrev_b32 v12, 2, v7 // (op 2183) 

s_sub_u32 s7, s6, s15 // (op 2183) 
v_lshl_add_u32 v7, v0, 4, v12 // (op 2183) 


s_lshr_b32 s8, s7, 1 // (op 2183) 

v_lshrrev_b32 v0, 2, v5 // (op 2183) 

s_add_u32 s7, s8, s15 // (op 2183) 


v_and_b32 v5, 3, v0 // (op 2183) 

s_lshr_b32 s8, s7, s11 // (op 2183) 

v_and_b32 v0, 3, v1 // (op 2183) 
s_and_b32 s7, s11, 2147483648 // (op 2183) 
v_lshl_add_u32 v12, v5, 2, v0 // (op 2183) 


s_cmp_eq_i32 s7, 0 // (op 2183) 
s_mov_b32 s15, scc // (op 2183) 

s_cmp_eq_i32 s15, 0 // (op 2183) 
s_cselect_b32 s7, s6, s8 // (op 2183) 



s_mul_hi_u32 s6, s7, s12 // (op 2183) 
s_sub_u32 s8, s7, s6 // (op 2183) 
s_lshr_b32 s15, s8, 1 // (op 2183) 

s_add_u32 s8, s15, s6 // (op 2183) 


s_lshr_b32 s6, s8, s10 // (op 2183) 

s_and_b32 s8, s10, 2147483648 // (op 2183) 
s_cmp_eq_i32 s8, 0 // (op 2183) 
s_mov_b32 s15, scc // (op 2183) 

s_cmp_eq_i32 s15, 0 // (op 2183) 
s_cselect_b32 s8, s7, s6 // (op 2183) 


s_mul_i32 s6, s8, s14 // (op 2183) 
v_lshl_add_u32 v0, s8, 4, v7 // (op 2183) 


s_sub_u32 s8, s7, s6 // (op 2183) 


v_lshl_add_u32 v5, s8, 4, v12 // (op 2183) 


v_mul_lo_u32 v7, v5, s46 // (op 2183) 

v_add_lshl_u32 v5, v0, v7, 1 // (op 2183) 


s_mul_i32 s19, s46, s23 // (op 2185) most significant: low of low * high
s_mul_i32 s15, s47, s22 // (op 2185) most significant: low of high * low
s_mul_i32 s18, s46, s22 // (op 2185) least significant
s_mul_hi_u32 s6, s46, s22 // (op 2185) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2185) most significant: sum
s_add_u32 s19, s19, s6 // (op 2185) most significant: sum


s_lshl_b64 s[6:7], s[18:19], 1 // (op 2185) 

s_mov_b32 s76, s48 // (op 2185) reg expression
s_mov_b32 s77, s49 // (op 2185) reg expression
s_mov_b32 s78, s6 // (op 2185) reg expression
s_mov_b32 s79, 131072 // (op 2185) call()


v_pack_b32_f16 v24, v26, v27 // (op 302) 
v_pack_b32_f16 v25, v28, v29 // (op 302) 


buffer_store_dwordx2 v[24:25], v5, s[76:79], 0 offen // (op 302) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalBottom_0_beta_278:

s_add_u32 s0, s0, s1 // (op 491) 
s_mul_i32 s15, s50, s16 // (op 489) 
s_cmp_gt_u32 s51, s15 // (op 489) 
s_mov_b32 s6, scc // (op 489) 
s_sub_u32 s7, s51, s15 // (op 489) 

s_cmp_eq_i32 s6, 0 // (op 489) 
s_cselect_b32 s8, 0, s7 // (op 489) 


s_cmp_lt_u32 s50, s8 // (op 489) 
s_mov_b32 s15, scc // (op 489) 
s_cmp_eq_i32 s15, 0 // (op 489) 
s_cselect_b32 s6, s8, s50 // (op 489) 


s_cmp_lt_u32 s0, s6 // (op 489) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopTop_DPStreamTileLoop_489 // (op 489) Condition: Bottom (jump to RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopTop_DPStreamTileLoop_489 if true)
RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopBottom_DPStreamTileLoop_489:










































s_endpgm  // End of RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2
.LRR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_end:
.size RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2, .LRR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_end-RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2
  .amdhsa_next_free_vgpr 96
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 16896 // lds bytes
  .amdhsa_accum_offset 92
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
    DataTypeB: F8
    DestDataType: b
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
    args: [ { type: int64, semantic: SizeSum },
            { type: int64, semantic: StrideC0 },
            { type: float32, semantic: Beta },
            { type: float32, semantic: Alpha },
            { type: uint32, semantic: RR_SKNumSKTilesPerWG },
            { type: uint32, semantic: RR_SKNumSKTiles },
            { type: int32, semantic: RR_SKMagicShiftsNumTilesN },
            { type: int32, semantic: RR_SKMagicShiftsIters },
            { type: uint32, semantic: RR_SKMagicMultipleNumTilesN },
            { type: uint32, semantic: RR_SKMagicMultipleIters },
            { type: uint32, semantic: RR_SKNumTilesN },
            { type: uint32, semantic: RR_SKNumWGs },
            { type: address, semantic: AddressC },
            { type: int64, semantic: SizeFree1 },
            { type: int64, semantic: StrideA0 },
            { type: address, semantic: AddressA },
            { type: int64, semantic: StrideB0 },
            { type: address, semantic: AddressB },
            { type: int64, semantic: StrideScaleA0 },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: SizeSumScaleA },
            { type: int64, semantic: StrideScaleB0 },
            { type: address, semantic: AddressMXScaleB },
            { type: address, semantic: RR_SKAddressPartials },
            { type: address, semantic: RR_SKAddressFlags },
            { type: int64, semantic: StrideD0 },
            { type: address, semantic: AddressD },
            { type: uint32, semantic: RR_SKNumDPTilesPerWG },
            { type: uint32, semantic: RR_SKNumDPTiles } ]
    macrotile: [16, 16, 256]
    threads: [64, 1, 1]
    grid: [RocRollerStreamK, One, One]
    occupancy: 9
    workspaceType: RocRollerStreamK
    workspaceSizePerElemC: 4
    workspaceSizePerElemBias: 0
    unroll: 2
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2
    .symbol: RR_GEMM_NT_BF8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2.kd
    .kernarg_segment_size: 184
    .group_segment_fixed_size: 16896
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 83
    .vgpr_count: 92
    .agpr_count: 4
    .max_flat_workgroup_size: 64
    .wavefront_size: 64
    .args:
      - .name: Tensor_0_size_1_9
        .size: 8
        .offset: 0
        .value_kind: by_value
      - .name: Tensor_10_stride_1_18
        .size: 8
        .offset: 8
        .value_kind: by_value
      - .name: user_Float_Value_14
        .size: 4
        .offset: 16
        .value_kind: by_value
      - .name: user_Float_Value_12
        .size: 4
        .offset: 20
        .value_kind: by_value
      - .name: numSKTilesPerWG
        .size: 4
        .offset: 24
        .value_kind: by_value
      - .name: numSKTiles
        .size: 4
        .offset: 28
        .value_kind: by_value
      - .name: MagicShifts_31
        .size: 4
        .offset: 32
        .value_kind: by_value
      - .name: MagicShifts_29
        .size: 4
        .offset: 36
        .value_kind: by_value
      - .name: MagicMultiple_30
        .size: 4
        .offset: 40
        .value_kind: by_value
      - .name: MagicMultiple_28
        .size: 4
        .offset: 44
        .value_kind: by_value
      - .name: numTiles1
        .size: 4
        .offset: 48
        .value_kind: by_value
      - .name: numWGs
        .size: 4
        .offset: 52
        .value_kind: by_value
      - .name: Tensor_10_pointer
        .size: 8
        .offset: 56
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_2_size_1_19
        .size: 8
        .offset: 64
        .value_kind: by_value
      - .name: Tensor_0_stride_1_8
        .size: 8
        .offset: 72
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 80
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_2_stride_0_11
        .size: 8
        .offset: 88
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 96
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_stride_1_13
        .size: 8
        .offset: 104
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 112
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_size_1_14
        .size: 8
        .offset: 120
        .value_kind: by_value
      - .name: Tensor_7_stride_0_16
        .size: 8
        .offset: 128
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 136
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: SCRATCH_None
        .size: 8
        .offset: 144
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: SCRATCH_ZeroedBeforeAndAfter
        .size: 8
        .offset: 152
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_21_stride_1_24
        .size: 8
        .offset: 160
        .value_kind: by_value
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 168
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: numDPTilesPerWG
        .size: 4
        .offset: 176
        .value_kind: by_value
      - .name: numDPTiles
        .size: 4
        .offset: 180
        .value_kind: by_value
...

.end_amdgpu_metadata

