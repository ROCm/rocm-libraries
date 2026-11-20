// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2
.p2align 8
.type RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2,@function
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_exec_begin:


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
s_load_dwordx2 s[52:53], s[0:1], 184 // Load scalar value

v_bfe_u32 v1, v0, 0, 10
v_bfe_u32 v2, v0, 10, 10


v_lshrrev_b32 v0, 6, v1 // (op 1900) 
v_and_b32 v3, 15, v1 // (op 1900) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 1900) 


v_and_b32 v0, 48, v1 // (op 1900) 
v_lshl_add_u32 v3, v4, 8, v0 // (op 1900) 


v_and_b32 v0, 48, v1 // (op 1965) 
v_and_b32 v4, 63, v1 // (op 1965) 
v_sub_u32 v5, v4, v0 // (op 1965) 

v_lshrrev_b32 v4, 1, v5 // (op 1965) 
v_add_lshl_u32 v6, v0, v4, 4 // (op 1965) 


v_and_b32 v0, 1, v5 // (op 1965) 

v_lshlrev_b32 v4, 3, v0 // (op 1965) 

v_add_u32 v0, v6, v4 // (op 1965) 


v_lshrrev_b32 v4, 6, v1 // (op 2031) 
v_and_b32 v5, 15, v1 // (op 2031) 
v_lshl_add_u32 v6, v4, 4, v5 // (op 2031) 


v_and_b32 v4, 63, v1 // (op 2031) 
v_lshrrev_b32 v5, 4, v4 // (op 2031) 

v_lshl_add_u32 v4, v6, 3, v5 // (op 2031) 


v_and_b32 v5, 48, v1 // (op 2097) 
v_and_b32 v6, 15, v1 // (op 2097) 
v_add_u32 v7, v5, v6 // (op 2097) 


v_lshrrev_b32 v5, 6, v1 // (op 2232) 
v_and_b32 v6, 15, v1 // (op 2232) 
v_lshl_add_u32 v8, v5, 4, v6 // (op 2232) 


v_and_b32 v5, 48, v1 // (op 2232) 
v_lshl_add_u32 v6, v8, 8, v5 // (op 2232) 


v_and_b32 v5, 48, v1 // (op 2297) 
v_and_b32 v8, 63, v1 // (op 2297) 
v_sub_u32 v9, v8, v5 // (op 2297) 

v_lshrrev_b32 v8, 1, v9 // (op 2297) 
v_add_lshl_u32 v10, v5, v8, 4 // (op 2297) 


v_and_b32 v5, 1, v9 // (op 2297) 

v_lshlrev_b32 v8, 3, v5 // (op 2297) 

v_add_u32 v5, v10, v8 // (op 2297) 


v_lshrrev_b32 v8, 6, v1 // (op 2363) 
v_and_b32 v9, 15, v1 // (op 2363) 
v_lshl_add_u32 v10, v8, 4, v9 // (op 2363) 


v_and_b32 v8, 63, v1 // (op 2363) 
v_lshrrev_b32 v9, 4, v8 // (op 2363) 

v_lshl_add_u32 v8, v10, 3, v9 // (op 2363) 


v_and_b32 v9, 48, v1 // (op 2429) 
v_and_b32 v10, 15, v1 // (op 2429) 
v_add_u32 v11, v9, v10 // (op 2429) 


v_lshrrev_b32 v9, 6, v1 // (op 2515) 
v_and_b32 v10, 15, v1 // (op 2515) 
v_lshl_add_u32 v12, v9, 4, v10 // (op 2515) 


v_and_b32 v9, 48, v1 // (op 2515) 
v_lshl_add_u32 v10, v12, 8, v9 // (op 2515) 


v_and_b32 v9, 48, v1 // (op 2554) 
v_and_b32 v12, 63, v1 // (op 2554) 
v_sub_u32 v13, v12, v9 // (op 2554) 

v_lshrrev_b32 v12, 1, v13 // (op 2554) 
v_add_lshl_u32 v14, v9, v12, 4 // (op 2554) 


v_and_b32 v9, 1, v13 // (op 2554) 

v_lshlrev_b32 v12, 3, v9 // (op 2554) 

v_add_u32 v9, v14, v12 // (op 2554) 


v_lshrrev_b32 v12, 6, v1 // (op 2578) 
v_and_b32 v13, 15, v1 // (op 2578) 
v_lshl_add_u32 v14, v12, 4, v13 // (op 2578) 


v_and_b32 v12, 63, v1 // (op 2578) 
v_lshrrev_b32 v13, 4, v12 // (op 2578) 

v_lshl_add_u32 v12, v14, 3, v13 // (op 2578) 


v_and_b32 v13, 48, v1 // (op 2602) 
v_and_b32 v14, 15, v1 // (op 2602) 
v_add_u32 v15, v13, v14 // (op 2602) 


v_lshrrev_b32 v13, 6, v1 // (op 2641) 
v_and_b32 v14, 15, v1 // (op 2641) 
v_lshl_add_u32 v16, v13, 4, v14 // (op 2641) 


v_and_b32 v13, 48, v1 // (op 2641) 
v_lshl_add_u32 v14, v16, 8, v13 // (op 2641) 


v_and_b32 v13, 48, v1 // (op 2680) 
v_and_b32 v16, 63, v1 // (op 2680) 
v_sub_u32 v17, v16, v13 // (op 2680) 

v_lshrrev_b32 v16, 1, v17 // (op 2680) 
v_add_lshl_u32 v18, v13, v16, 4 // (op 2680) 


v_and_b32 v13, 1, v17 // (op 2680) 

v_lshlrev_b32 v16, 3, v13 // (op 2680) 

v_add_u32 v13, v18, v16 // (op 2680) 


v_lshrrev_b32 v16, 6, v1 // (op 2704) 
v_and_b32 v17, 15, v1 // (op 2704) 
v_lshl_add_u32 v18, v16, 4, v17 // (op 2704) 


v_and_b32 v16, 63, v1 // (op 2704) 
v_lshrrev_b32 v17, 4, v16 // (op 2704) 

v_lshl_add_u32 v16, v18, 3, v17 // (op 2704) 


v_and_b32 v17, 48, v1 // (op 2728) 
v_and_b32 v18, 15, v1 // (op 2728) 
v_add_u32 v19, v17, v18 // (op 2728) 


v_mov_b32 v17, s6 // (op 2) Move value
v_mov_b32 v18, s7 // (op 43) Move value


s_mov_b32 s0, 0 // (op 427) call()
s_mov_b32 s1, 0 // (op 428) call()
s_mov_b32 s6, 0 // (op 429) call()
s_mov_b32 s7, 0 // (op 430) call()
s_mov_b32 s18, 0 // (op 431) call()
s_mov_b32 s19, 0 // (op 432) call()
s_mov_b32 s0, 0 // (op 327) call()

s_waitcnt lgkmcnt(0)// DEBUG: Wait for scalar queue
s_mul_i32 s54, s8, s16 // (op 326) 
s_cmp_gt_u32 s9, s54 // (op 326) 
s_mov_b32 s55, scc // (op 326) 
s_sub_u32 s56, s9, s54 // (op 326) 

s_cmp_eq_i32 s55, 0 // (op 326) 
s_cselect_b32 s54, 0, s56 // (op 326) 


s_cmp_lt_u32 s8, s54 // (op 326) 
s_mov_b32 s55, scc // (op 326) 
s_cmp_eq_i32 s55, 0 // (op 326) 
s_cselect_b32 s56, s54, s8 // (op 326) 


s_cmp_lt_u32 s0, s56 // (op 326) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopBottom_SKStreamTileLoop_326 // (op 326) Condition: Top (jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopBottom_SKStreamTileLoop_326 if false)
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopTop_SKStreamTileLoop_326:
v_lshrrev_b32 v20, 6, v1 // (op 1826) 
s_mul_i32 s54, s16, s8 // (op 1826) 
s_add_u32 s54, s54, s0 // (op 1826) 
v_and_b32 v21, 63, v1 // (op 1826) 
s_mul_hi_u32 s55, s54, s13 // (op 1826) 
v_lshrrev_b32 v22, 4, v21 // (op 1826) 
s_sub_u32 s56, s54, s55 // (op 1826) 
v_lshlrev_b32 v23, 2, v22 // (op 1826) 

v_lshl_add_u32 v22, v20, 4, v23 // (op 1826) 


s_lshr_b32 s57, s56, 1 // (op 1826) 

s_add_u32 s56, s57, s55 // (op 1826) 


v_lshrrev_b32 v20, 2, v21 // (op 1826) 

s_lshr_b32 s55, s56, s11 // (op 1826) 

v_and_b32 v21, 3, v20 // (op 1826) 

s_and_b32 s56, s11, 2147483648 // (op 1826) 
v_and_b32 v20, 3, v1 // (op 1826) 
s_cmp_eq_i32 s56, 0 // (op 1826) 
s_mov_b32 s57, scc // (op 1826) 

v_lshl_add_u32 v23, v21, 2, v20 // (op 1826) 


s_cmp_eq_i32 s57, 0 // (op 1826) 
s_cselect_b32 s56, s54, s55 // (op 1826) 



s_mul_hi_u32 s54, s56, s12 // (op 1826) 
s_sub_u32 s55, s56, s54 // (op 1826) 
s_lshr_b32 s57, s55, 1 // (op 1826) 

s_add_u32 s55, s57, s54 // (op 1826) 


s_lshr_b32 s54, s55, s10 // (op 1826) 

s_and_b32 s55, s10, 2147483648 // (op 1826) 
s_cmp_eq_i32 s55, 0 // (op 1826) 
s_mov_b32 s57, scc // (op 1826) 

s_cmp_eq_i32 s57, 0 // (op 1826) 
s_cselect_b32 s55, s56, s54 // (op 1826) 


s_mul_i32 s54, s55, s14 // (op 1826) 
v_lshl_add_u32 v20, s55, 4, v22 // (op 1826) 


s_sub_u32 s55, s56, s54 // (op 1826) 


v_lshl_add_u32 v21, s55, 4, v23 // (op 1826) 


v_mul_lo_u32 v22, v21, s4 // (op 1826) 

v_add_lshl_u32 v21, v20, v22, 2 // (op 1826) 


s_mul_i32 s55, s4, s23 // (op 1828) most significant: low of low * high
s_mul_i32 s56, s5, s22 // (op 1828) most significant: low of high * low
s_mul_i32 s54, s4, s22 // (op 1828) least significant
s_mul_hi_u32 s57, s4, s22 // (op 1828) most significant: high of low * low
s_add_u32 s55, s55, s56 // (op 1828) most significant: sum
s_add_u32 s55, s55, s57 // (op 1828) most significant: sum


s_lshl_b64 s[56:57], s[54:55], 2 // (op 1828) 

s_mov_b32 s60, s20 // (op 1828) reg expression
s_mov_b32 s61, s21 // (op 1828) reg expression
s_mov_b32 s62, s56 // (op 1828) reg expression
s_mov_b32 s63, 131072 // (op 1828) call()

v_lshrrev_b32 v20, 6, v1 // (op 1912) 
v_and_b32 v22, 15, v1 // (op 1912) 
v_lshl_add_u32 v23, v20, 4, v22 // (op 1912) 


v_and_b32 v20, 48, v1 // (op 1912) 
v_lshl_add_u32 v22, v23, 8, v20 // (op 1912) 


v_and_b32 v20, 48, v1 // (op 1977) 
v_and_b32 v23, 63, v1 // (op 1977) 
v_sub_u32 v24, v23, v20 // (op 1977) 

v_lshrrev_b32 v23, 1, v24 // (op 1977) 
v_add_lshl_u32 v25, v20, v23, 4 // (op 1977) 


v_and_b32 v20, 1, v24 // (op 1977) 

v_lshlrev_b32 v23, 3, v20 // (op 1977) 

v_add_u32 v20, v25, v23 // (op 1977) 


v_lshrrev_b32 v23, 6, v1 // (op 2043) 
v_and_b32 v24, 15, v1 // (op 2043) 
v_lshl_add_u32 v25, v23, 4, v24 // (op 2043) 


v_and_b32 v23, 63, v1 // (op 2043) 
v_lshrrev_b32 v24, 4, v23 // (op 2043) 

v_lshl_add_u32 v23, v25, 3, v24 // (op 2043) 


v_and_b32 v24, 48, v1 // (op 2109) 
v_and_b32 v25, 15, v1 // (op 2109) 
v_add_u32 v26, v24, v25 // (op 2109) 


s_mul_i32 s6, s8, s16 // (op 331) 
s_add_u32 s6, s6, s0 // (op 331) 
s_mul_hi_u32 s54, s6, s13 // (op 332) 
s_sub_u32 s55, s6, s54 // (op 332) 
s_lshr_b32 s56, s55, 1 // (op 332) 

s_add_u32 s55, s56, s54 // (op 332) 


s_lshr_b32 s54, s55, s11 // (op 332) 

s_and_b32 s55, s11, 2147483648 // (op 332) 
s_cmp_eq_i32 s55, 0 // (op 332) 
s_mov_b32 s56, scc // (op 332) 

s_cmp_eq_i32 s56, 0 // (op 332) 
s_cselect_b32 s55, s6, s54 // (op 332) 


s_ashr_i64 s[56:57], s[2:3], 8 // (op 332) 
s_mul_i32 s54, s55, s56 // (op 332) 


s_sub_u32 s18, s6, s54 // (op 332) 

s_ashr_i64 s[54:55], s[2:3], 8 // (op 333) 
s_cmp_gt_u32 s54, s18 // (op 333) 
s_mov_b32 s56, scc // (op 333) 
s_sub_u32 s57, s54, s18 // (op 333) 

s_cmp_eq_i32 s56, 0 // (op 333) 
s_cselect_b32 s54, 0, s57 // (op 333) 


s_cmp_gt_u32 s8, s0 // (op 333) 
s_mov_b32 s55, scc // (op 333) 
s_sub_u32 s56, s8, s0 // (op 333) 
s_cmp_eq_i32 s55, 0 // (op 333) 
s_cselect_b32 s57, 0, s56 // (op 333) 


s_cmp_lt_u32 s54, s57 // (op 333) 
s_mov_b32 s55, scc // (op 333) 
s_cmp_eq_i32 s55, 0 // (op 333) 
s_cselect_b32 s56, s57, s54 // (op 333) 



s_cmp_gt_u32 s9, s6 // (op 333) 
s_mov_b32 s54, scc // (op 333) 
s_sub_u32 s55, s9, s6 // (op 333) 
s_cmp_eq_i32 s54, 0 // (op 333) 
s_cselect_b32 s57, 0, s55 // (op 333) 


s_cmp_lt_u32 s56, s57 // (op 333) 
s_mov_b32 s54, scc // (op 333) 
s_cmp_eq_i32 s54, 0 // (op 333) 
s_cselect_b32 s7, s57, s56 // (op 333) 



s_add_u32 s54, s18, s7 // (op 334) 
s_sub_u32 s19, s54, 1 // (op 334) 

v_accvgpr_write a0, 0.00000 // (op 63) call()
v_accvgpr_write a1, 0.00000 // (op 63) call()
v_accvgpr_write a2, 0.00000 // (op 63) call()
v_accvgpr_write a3, 0.00000 // (op 63) call()
v_lshrrev_b32 v24, 4, v1 // (op 1358) 
s_mul_i32 s54, s16, s8 // (op 1358) 
s_add_u32 s54, s54, s0 // (op 1358) 
v_and_b32 v25, -16, v1 // (op 1358) 
s_mul_hi_u32 s55, s54, s13 // (op 1358) 
v_sub_u32 v27, v1, v25 // (op 1358) 

s_sub_u32 s56, s54, s55 // (op 1358) 
v_lshlrev_b32 v25, 4, v27 // (op 1358) 

s_lshr_b32 s57, s56, 1 // (op 1358) 

s_add_u32 s56, s57, s55 // (op 1358) 


s_lshr_b32 s55, s56, s11 // (op 1358) 

s_and_b32 s56, s11, 2147483648 // (op 1358) 
s_cmp_eq_i32 s56, 0 // (op 1358) 
s_mov_b32 s57, scc // (op 1358) 

s_cmp_eq_i32 s57, 0 // (op 1358) 
s_cselect_b32 s56, s54, s55 // (op 1358) 


s_mul_hi_u32 s55, s56, s12 // (op 1358) 
s_sub_u32 s57, s56, s55 // (op 1358) 
s_lshr_b32 s58, s57, 1 // (op 1358) 

s_add_u32 s57, s58, s55 // (op 1358) 


s_lshr_b32 s55, s57, s10 // (op 1358) 

s_and_b32 s57, s10, 2147483648 // (op 1358) 
s_cmp_eq_i32 s57, 0 // (op 1358) 
s_mov_b32 s58, scc // (op 1358) 

s_cmp_eq_i32 s58, 0 // (op 1358) 
s_cselect_b32 s57, s56, s55 // (op 1358) 


v_lshl_add_u32 v27, s57, 4, v24 // (op 1358) 


s_ashr_i64 s[58:59], s[2:3], 8 // (op 1358) 
s_mul_i32 s55, s56, s58 // (op 1358) 


s_sub_u32 s56, s54, s55 // (op 1358) 


v_lshl_add_u32 v24, s56, 8, v25 // (op 1358) 


v_mov_b32 v29, 0 // (op 1358) convert
v_mov_b32 v28, v24 // (op 1358) convert

v_mul_lo_u32 v25, v27, s25 // (op 1358) most significant: low of low * high
v_mul_hi_u32 v30, v27, s24 // (op 1358) most significant: high of low * low
v_mul_lo_u32 v24, v27, s24 // (op 1358) least significant: low of low * low
v_add_u32 v25, v25, v30 // (op 1358) most significant: sum

v_add_co_u32 v24, s[54:55], v24, v28 // (op 1358) least significant half
v_addc_co_u32 v25, s[54:55], v25, v29, s[54:55] // (op 1358) most significant half



s_mul_i32 s55, s24, s29 // (op 1360) most significant: low of low * high
s_mul_i32 s56, s25, s28 // (op 1360) most significant: low of high * low
s_mul_i32 s54, s24, s28 // (op 1360) least significant
s_mul_hi_u32 s57, s24, s28 // (op 1360) most significant: high of low * low
s_add_u32 s55, s55, s56 // (op 1360) most significant: sum
s_add_u32 s55, s55, s57 // (op 1360) most significant: sum


s_mov_b32 s56, s26 // (op 1360) reg expression
s_mov_b32 s57, s27 // (op 1360) reg expression
s_mov_b32 s58, s54 // (op 1360) reg expression
s_mov_b32 s59, 131072 // (op 1360) call()

v_lshrrev_b32 v27, 4, v1 // (op 1377) 
s_mul_i32 s54, s16, s8 // (op 1377) 
s_add_u32 s54, s54, s0 // (op 1377) 
v_and_b32 v28, -16, v1 // (op 1377) 
s_mul_hi_u32 s55, s54, s13 // (op 1377) 
v_sub_u32 v29, v1, v28 // (op 1377) 

v_lshlrev_b32 v28, 4, v29 // (op 1377) 

s_sub_u32 s64, s54, s55 // (op 1377) 
s_lshr_b32 s65, s64, 1 // (op 1377) 

s_add_u32 s64, s65, s55 // (op 1377) 


s_lshr_b32 s55, s64, s11 // (op 1377) 

s_and_b32 s64, s11, 2147483648 // (op 1377) 
s_cmp_eq_i32 s64, 0 // (op 1377) 
s_mov_b32 s65, scc // (op 1377) 

s_cmp_eq_i32 s65, 0 // (op 1377) 
s_cselect_b32 s64, s54, s55 // (op 1377) 


s_mul_hi_u32 s55, s64, s12 // (op 1377) 
s_sub_u32 s65, s64, s55 // (op 1377) 
s_lshr_b32 s66, s65, 1 // (op 1377) 

s_add_u32 s65, s66, s55 // (op 1377) 


s_lshr_b32 s55, s65, s10 // (op 1377) 

s_and_b32 s65, s10, 2147483648 // (op 1377) 
s_cmp_eq_i32 s65, 0 // (op 1377) 
s_mov_b32 s66, scc // (op 1377) 

s_cmp_eq_i32 s66, 0 // (op 1377) 
s_cselect_b32 s65, s64, s55 // (op 1377) 


v_lshl_add_u32 v29, s65, 4, v27 // (op 1377) 


s_ashr_i64 s[66:67], s[2:3], 8 // (op 1377) 
s_mul_i32 s55, s64, s66 // (op 1377) 


s_sub_u32 s64, s54, s55 // (op 1377) 


v_lshl_add_u32 v27, s64, 8, v28 // (op 1377) 


v_mov_b32 v31, 0 // (op 1377) convert
v_mov_b32 v30, v27 // (op 1377) convert

v_mul_lo_u32 v33, v29, s25 // (op 1377) most significant: low of low * high
v_mul_hi_u32 v27, v29, s24 // (op 1377) most significant: high of low * low
v_mul_lo_u32 v32, v29, s24 // (op 1377) least significant: low of low * low
v_add_u32 v33, v33, v27 // (op 1377) most significant: sum

v_add_co_u32 v32, s[54:55], v32, v30 // (op 1377) least significant half
v_addc_co_u32 v33, s[54:55], v33, v31, s[54:55] // (op 1377) most significant half



s_mul_i32 s55, s24, s29 // (op 1379) most significant: low of low * high
s_mul_i32 s64, s25, s28 // (op 1379) most significant: low of high * low
s_mul_i32 s54, s24, s28 // (op 1379) least significant
s_mul_hi_u32 s65, s24, s28 // (op 1379) most significant: high of low * low
s_add_u32 s55, s55, s64 // (op 1379) most significant: sum
s_add_u32 s55, s55, s65 // (op 1379) most significant: sum


s_mov_b32 s56, s26 // (op 1379) reg expression
s_mov_b32 s57, s27 // (op 1379) reg expression
s_mov_b32 s58, s54 // (op 1379) reg expression
s_mov_b32 s59, 131072 // (op 1379) call()

s_mul_i32 s54, s16, s8 // (op 1489) 
s_add_u32 s54, s54, s0 // (op 1489) 
s_mul_hi_u32 s55, s54, s13 // (op 1489) 
s_sub_u32 s64, s54, s55 // (op 1489) 
s_lshr_b32 s65, s64, 1 // (op 1489) 

s_add_u32 s64, s65, s55 // (op 1489) 


s_lshr_b32 s55, s64, s11 // (op 1489) 

s_and_b32 s64, s11, 2147483648 // (op 1489) 
s_cmp_eq_i32 s64, 0 // (op 1489) 
s_mov_b32 s65, scc // (op 1489) 

s_cmp_eq_i32 s65, 0 // (op 1489) 
s_cselect_b32 s64, s54, s55 // (op 1489) 


s_mul_hi_u32 s55, s64, s12 // (op 1489) 
s_sub_u32 s65, s64, s55 // (op 1489) 
s_lshr_b32 s66, s65, 1 // (op 1489) 

s_add_u32 s65, s66, s55 // (op 1489) 


s_lshr_b32 s55, s65, s10 // (op 1489) 

s_and_b32 s65, s10, 2147483648 // (op 1489) 
s_cmp_eq_i32 s65, 0 // (op 1489) 
s_mov_b32 s66, scc // (op 1489) 

s_cmp_eq_i32 s66, 0 // (op 1489) 
s_cselect_b32 s65, s64, s55 // (op 1489) 


s_mul_i32 s55, s65, s14 // (op 1489) 

s_sub_u32 s65, s64, s55 // (op 1489) 

s_lshl_b32 s55, s65, 4 // (op 1489) 

s_mov_b32 s67, 0 // (op 1489) convert
s_mov_b32 s66, s55 // (op 1489) convert

s_ashr_i64 s[68:69], s[2:3], 8 // (op 1489) 
s_mul_i32 s55, s64, s68 // (op 1489) 


s_sub_u32 s64, s54, s55 // (op 1489) 


v_lshl_add_u32 v27, s64, 8, v1 // (op 1489) 

v_mul_lo_u32 v29, v27, s31 // (op 1489) most significant: low of low * high
v_mul_hi_u32 v30, v27, s30 // (op 1489) most significant: high of low * low
v_mul_lo_u32 v28, v27, s30 // (op 1489) least significant: low of low * low
v_add_u32 v29, v29, v30 // (op 1489) most significant: sum

v_mov_b32 v30, s66 // (op 1489) 
v_mov_b32 v31, s67 // (op 1489) 
v_add_co_u32 v28, s[54:55], v28, v30 // (op 1489) least significant half
v_addc_co_u32 v29, s[54:55], v29, v31, s[54:55] // (op 1489) most significant half





s_mul_i32 s55, s30, s3 // (op 1491) most significant: low of low * high
s_mul_i32 s64, s31, s2 // (op 1491) most significant: low of high * low
s_mul_i32 s54, s30, s2 // (op 1491) least significant
s_mul_hi_u32 s65, s30, s2 // (op 1491) most significant: high of low * low
s_add_u32 s55, s55, s64 // (op 1491) most significant: sum
s_add_u32 s55, s55, s65 // (op 1491) most significant: sum


s_mov_b32 s64, s32 // (op 1491) reg expression
s_mov_b32 s65, s33 // (op 1491) reg expression
s_mov_b32 s66, s54 // (op 1491) reg expression
s_mov_b32 s67, 131072 // (op 1491) call()

s_mul_i32 s54, s16, s8 // (op 1508) 
s_add_u32 s54, s54, s0 // (op 1508) 
s_mul_hi_u32 s55, s54, s13 // (op 1508) 
s_sub_u32 s68, s54, s55 // (op 1508) 
s_lshr_b32 s69, s68, 1 // (op 1508) 

s_add_u32 s68, s69, s55 // (op 1508) 


s_lshr_b32 s55, s68, s11 // (op 1508) 

s_and_b32 s68, s11, 2147483648 // (op 1508) 
s_cmp_eq_i32 s68, 0 // (op 1508) 
s_mov_b32 s69, scc // (op 1508) 

s_cmp_eq_i32 s69, 0 // (op 1508) 
s_cselect_b32 s68, s54, s55 // (op 1508) 


s_mul_hi_u32 s55, s68, s12 // (op 1508) 
s_sub_u32 s69, s68, s55 // (op 1508) 
s_lshr_b32 s70, s69, 1 // (op 1508) 

s_add_u32 s69, s70, s55 // (op 1508) 


s_lshr_b32 s55, s69, s10 // (op 1508) 

s_and_b32 s69, s10, 2147483648 // (op 1508) 
s_cmp_eq_i32 s69, 0 // (op 1508) 
s_mov_b32 s70, scc // (op 1508) 

s_cmp_eq_i32 s70, 0 // (op 1508) 
s_cselect_b32 s69, s68, s55 // (op 1508) 


s_mul_i32 s55, s69, s14 // (op 1508) 

s_sub_u32 s69, s68, s55 // (op 1508) 

s_lshl_b32 s55, s69, 4 // (op 1508) 

s_mov_b32 s71, 0 // (op 1508) convert
s_mov_b32 s70, s55 // (op 1508) convert

s_ashr_i64 s[72:73], s[2:3], 8 // (op 1508) 
s_mul_i32 s55, s68, s72 // (op 1508) 


s_sub_u32 s68, s54, s55 // (op 1508) 


v_lshl_add_u32 v27, s68, 8, v1 // (op 1508) 

v_mul_lo_u32 v31, v27, s31 // (op 1508) most significant: low of low * high
v_mul_hi_u32 v34, v27, s30 // (op 1508) most significant: high of low * low
v_mul_lo_u32 v30, v27, s30 // (op 1508) least significant: low of low * low
v_add_u32 v31, v31, v34 // (op 1508) most significant: sum

v_mov_b32 v34, s70 // (op 1508) 
v_mov_b32 v35, s71 // (op 1508) 
v_add_co_u32 v30, s[54:55], v30, v34 // (op 1508) least significant half
v_addc_co_u32 v31, s[54:55], v31, v35, s[54:55] // (op 1508) most significant half





s_mul_i32 s55, s30, s3 // (op 1510) most significant: low of low * high
s_mul_i32 s68, s31, s2 // (op 1510) most significant: low of high * low
s_mul_i32 s54, s30, s2 // (op 1510) least significant
s_mul_hi_u32 s69, s30, s2 // (op 1510) most significant: high of low * low
s_add_u32 s55, s55, s68 // (op 1510) most significant: sum
s_add_u32 s55, s55, s69 // (op 1510) most significant: sum


s_mov_b32 s64, s32 // (op 1510) reg expression
s_mov_b32 s65, s33 // (op 1510) reg expression
s_mov_b32 s66, s54 // (op 1510) reg expression
s_mov_b32 s67, 131072 // (op 1510) call()

v_lshrrev_b32 v27, 3, v1 // (op 1617) 
s_mul_i32 s54, s16, s8 // (op 1617) 
s_add_u32 s54, s54, s0 // (op 1617) 
v_lshlrev_b32 v34, 1, v27 // (op 1617) 

s_mul_hi_u32 s55, s54, s13 // (op 1617) 
v_and_b32 v27, -8, v1 // (op 1617) 
s_sub_u32 s68, s54, s55 // (op 1617) 
v_sub_u32 v35, v1, v27 // (op 1617) 

s_lshr_b32 s69, s68, 1 // (op 1617) 

s_add_u32 s68, s69, s55 // (op 1617) 


s_lshr_b32 s55, s68, s11 // (op 1617) 

s_and_b32 s68, s11, 2147483648 // (op 1617) 
s_cmp_eq_i32 s68, 0 // (op 1617) 
s_mov_b32 s69, scc // (op 1617) 

s_cmp_eq_i32 s69, 0 // (op 1617) 
s_cselect_b32 s68, s54, s55 // (op 1617) 


s_mul_hi_u32 s55, s68, s12 // (op 1617) 
s_sub_u32 s69, s68, s55 // (op 1617) 
s_lshr_b32 s70, s69, 1 // (op 1617) 

s_add_u32 s69, s70, s55 // (op 1617) 


s_lshr_b32 s55, s69, s10 // (op 1617) 

s_and_b32 s69, s10, 2147483648 // (op 1617) 
s_cmp_eq_i32 s69, 0 // (op 1617) 
s_mov_b32 s70, scc // (op 1617) 

s_cmp_eq_i32 s70, 0 // (op 1617) 
s_cselect_b32 s69, s68, s55 // (op 1617) 


v_lshl_add_u32 v27, s69, 4, v34 // (op 1617) 


s_ashr_i64 s[70:71], s[2:3], 8 // (op 1617) 
s_mul_i32 s55, s68, s70 // (op 1617) 


s_sub_u32 s68, s54, s55 // (op 1617) 


v_lshl_add_u32 v34, s68, 3, v35 // (op 1617) 


v_mul_lo_u32 v35, v27, s34 // (op 1617) 
v_add_u32 v35, v35, v34 // (op 1617) 


s_mul_i32 s55, s34, s29 // (op 1619) most significant: low of low * high
s_mul_i32 s68, s35, s28 // (op 1619) most significant: low of high * low
s_mul_i32 s54, s34, s28 // (op 1619) least significant
s_mul_hi_u32 s69, s34, s28 // (op 1619) most significant: high of low * low
s_add_u32 s55, s55, s68 // (op 1619) most significant: sum
s_add_u32 s55, s55, s69 // (op 1619) most significant: sum


s_mov_b32 s68, s36 // (op 1619) reg expression
s_mov_b32 s69, s37 // (op 1619) reg expression
s_mov_b32 s70, s54 // (op 1619) reg expression
s_mov_b32 s71, 131072 // (op 1619) call()

v_lshrrev_b32 v27, 3, v1 // (op 1636) 
s_mul_i32 s54, s16, s8 // (op 1636) 
s_add_u32 s54, s54, s0 // (op 1636) 
v_lshlrev_b32 v34, 1, v27 // (op 1636) 

s_mul_hi_u32 s55, s54, s13 // (op 1636) 
v_and_b32 v27, -8, v1 // (op 1636) 
s_sub_u32 s72, s54, s55 // (op 1636) 
v_sub_u32 v36, v1, v27 // (op 1636) 

s_lshr_b32 s73, s72, 1 // (op 1636) 

s_add_u32 s72, s73, s55 // (op 1636) 


s_lshr_b32 s55, s72, s11 // (op 1636) 

s_and_b32 s72, s11, 2147483648 // (op 1636) 
s_cmp_eq_i32 s72, 0 // (op 1636) 
s_mov_b32 s73, scc // (op 1636) 

s_cmp_eq_i32 s73, 0 // (op 1636) 
s_cselect_b32 s72, s54, s55 // (op 1636) 


s_mul_hi_u32 s55, s72, s12 // (op 1636) 
s_sub_u32 s73, s72, s55 // (op 1636) 
s_lshr_b32 s74, s73, 1 // (op 1636) 

s_add_u32 s73, s74, s55 // (op 1636) 


s_lshr_b32 s55, s73, s10 // (op 1636) 

s_and_b32 s73, s10, 2147483648 // (op 1636) 
s_cmp_eq_i32 s73, 0 // (op 1636) 
s_mov_b32 s74, scc // (op 1636) 

s_cmp_eq_i32 s74, 0 // (op 1636) 
s_cselect_b32 s73, s72, s55 // (op 1636) 


v_lshl_add_u32 v27, s73, 4, v34 // (op 1636) 


s_ashr_i64 s[74:75], s[2:3], 8 // (op 1636) 
s_mul_i32 s55, s72, s74 // (op 1636) 


s_sub_u32 s72, s54, s55 // (op 1636) 


v_lshl_add_u32 v34, s72, 3, v36 // (op 1636) 


v_mul_lo_u32 v36, v27, s34 // (op 1636) 
v_add_u32 v36, v36, v34 // (op 1636) 


s_mul_i32 s55, s34, s29 // (op 1638) most significant: low of low * high
s_mul_i32 s72, s35, s28 // (op 1638) most significant: low of high * low
s_mul_i32 s54, s34, s28 // (op 1638) least significant
s_mul_hi_u32 s73, s34, s28 // (op 1638) most significant: high of low * low
s_add_u32 s55, s55, s72 // (op 1638) most significant: sum
s_add_u32 s55, s55, s73 // (op 1638) most significant: sum


s_mov_b32 s68, s36 // (op 1638) reg expression
s_mov_b32 s69, s37 // (op 1638) reg expression
s_mov_b32 s70, s54 // (op 1638) reg expression
s_mov_b32 s71, 131072 // (op 1638) call()

v_and_b32 v27, -16, v1 // (op 1745) 
s_mul_i32 s54, s16, s8 // (op 1745) 
s_add_u32 s54, s54, s0 // (op 1745) 
v_sub_u32 v34, v1, v27 // (op 1745) 

s_mul_hi_u32 s55, s54, s13 // (op 1745) 
v_lshrrev_b32 v27, 4, v1 // (op 1745) 
s_sub_u32 s72, s54, s55 // (op 1745) 
v_lshlrev_b32 v37, 1, v27 // (op 1745) 

s_lshr_b32 s73, s72, 1 // (op 1745) 

s_add_u32 s72, s73, s55 // (op 1745) 


s_lshr_b32 s55, s72, s11 // (op 1745) 

s_and_b32 s72, s11, 2147483648 // (op 1745) 
s_cmp_eq_i32 s72, 0 // (op 1745) 
s_mov_b32 s73, scc // (op 1745) 

s_cmp_eq_i32 s73, 0 // (op 1745) 
s_cselect_b32 s72, s54, s55 // (op 1745) 


s_mul_hi_u32 s55, s72, s12 // (op 1745) 
s_sub_u32 s73, s72, s55 // (op 1745) 
s_lshr_b32 s74, s73, 1 // (op 1745) 

s_add_u32 s73, s74, s55 // (op 1745) 


s_lshr_b32 s55, s73, s10 // (op 1745) 

s_and_b32 s73, s10, 2147483648 // (op 1745) 
s_cmp_eq_i32 s73, 0 // (op 1745) 
s_mov_b32 s74, scc // (op 1745) 

s_cmp_eq_i32 s74, 0 // (op 1745) 
s_cselect_b32 s73, s72, s55 // (op 1745) 


s_mul_i32 s55, s73, s14 // (op 1745) 

s_sub_u32 s73, s72, s55 // (op 1745) 

v_lshl_add_u32 v27, s73, 4, v34 // (op 1745) 


s_ashr_i64 s[74:75], s[2:3], 8 // (op 1745) 
s_mul_i32 s55, s72, s74 // (op 1745) 


s_sub_u32 s72, s54, s55 // (op 1745) 


v_lshl_add_u32 v34, s72, 3, v37 // (op 1745) 


v_mul_lo_u32 v37, v34, s38 // (op 1745) 
v_add_u32 v37, v37, v27 // (op 1745) 


s_mul_i32 s55, s38, s43 // (op 1747) most significant: low of low * high
s_mul_i32 s72, s39, s42 // (op 1747) most significant: low of high * low
s_mul_i32 s54, s38, s42 // (op 1747) least significant
s_mul_hi_u32 s73, s38, s42 // (op 1747) most significant: high of low * low
s_add_u32 s55, s55, s72 // (op 1747) most significant: sum
s_add_u32 s55, s55, s73 // (op 1747) most significant: sum


s_mov_b32 s72, s40 // (op 1747) reg expression
s_mov_b32 s73, s41 // (op 1747) reg expression
s_mov_b32 s74, s54 // (op 1747) reg expression
s_mov_b32 s75, 131072 // (op 1747) call()

v_and_b32 v27, -16, v1 // (op 1764) 
s_mul_i32 s54, s16, s8 // (op 1764) 
s_add_u32 s54, s54, s0 // (op 1764) 
v_sub_u32 v34, v1, v27 // (op 1764) 

s_mul_hi_u32 s55, s54, s13 // (op 1764) 
v_lshrrev_b32 v27, 4, v1 // (op 1764) 
v_lshlrev_b32 v38, 1, v27 // (op 1764) 

s_sub_u32 s76, s54, s55 // (op 1764) 
s_lshr_b32 s77, s76, 1 // (op 1764) 

s_add_u32 s76, s77, s55 // (op 1764) 


s_lshr_b32 s55, s76, s11 // (op 1764) 

s_and_b32 s76, s11, 2147483648 // (op 1764) 
s_cmp_eq_i32 s76, 0 // (op 1764) 
s_mov_b32 s77, scc // (op 1764) 

s_cmp_eq_i32 s77, 0 // (op 1764) 
s_cselect_b32 s76, s54, s55 // (op 1764) 


s_mul_hi_u32 s55, s76, s12 // (op 1764) 
s_sub_u32 s77, s76, s55 // (op 1764) 
s_lshr_b32 s78, s77, 1 // (op 1764) 

s_add_u32 s77, s78, s55 // (op 1764) 


s_lshr_b32 s55, s77, s10 // (op 1764) 

s_and_b32 s77, s10, 2147483648 // (op 1764) 
s_cmp_eq_i32 s77, 0 // (op 1764) 
s_mov_b32 s78, scc // (op 1764) 

s_cmp_eq_i32 s78, 0 // (op 1764) 
s_cselect_b32 s77, s76, s55 // (op 1764) 


s_mul_i32 s55, s77, s14 // (op 1764) 

s_sub_u32 s77, s76, s55 // (op 1764) 

v_lshl_add_u32 v27, s77, 4, v34 // (op 1764) 


s_ashr_i64 s[78:79], s[2:3], 8 // (op 1764) 
s_mul_i32 s55, s76, s78 // (op 1764) 


s_sub_u32 s76, s54, s55 // (op 1764) 


v_lshl_add_u32 v34, s76, 3, v38 // (op 1764) 


v_mul_lo_u32 v38, v34, s38 // (op 1764) 
v_add_u32 v38, v38, v27 // (op 1764) 


s_mul_i32 s55, s38, s43 // (op 1766) most significant: low of low * high
s_mul_i32 s76, s39, s42 // (op 1766) most significant: low of high * low
s_mul_i32 s54, s38, s42 // (op 1766) least significant
s_mul_hi_u32 s77, s38, s42 // (op 1766) most significant: high of low * low
s_add_u32 s55, s55, s76 // (op 1766) most significant: sum
s_add_u32 s55, s55, s77 // (op 1766) most significant: sum


s_mov_b32 s72, s40 // (op 1766) reg expression
s_mov_b32 s73, s41 // (op 1766) reg expression
s_mov_b32 s74, s54 // (op 1766) reg expression
s_mov_b32 s75, 131072 // (op 1766) call()

v_lshlrev_b32 v27, 4, v1 // (op 1953) 
s_nop 0
v_readfirstlane_b32 s54, v27 // (op 1953) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 2866) 

s_lshl_b64 s[76:77], s[30:31], 6 // (op 1269) 
v_mov_b64 v[40:41], s[76:77] // (op 1269) convert


v_mov_b64 v[42:43], v[28:29] // (op 1269) reg expression
s_mov_b32 m0, s54 // (op 1269) reg expression
buffer_load_dwordx4 v42, s[64:67], 0 offen offset: 0 lds // (op 1269) Load value direct to lds
v_add_co_u32 v42, s[76:77], v42, v40 // (op 1269) least significant half
v_addc_co_u32 v43, s[76:77], 0, v43, s[76:77] // (op 1269) most significant half

s_add_u32 m0, m0, 1024 // (op 1269) 
buffer_load_dwordx4 v42, s[64:67], 0 offen offset: 0 lds // (op 1269) Load value direct to lds
v_add_co_u32 v42, s[76:77], v42, v40 // (op 1269) least significant half
v_addc_co_u32 v43, s[76:77], 0, v43, s[76:77] // (op 1269) most significant half

s_add_u32 m0, m0, 1024 // (op 1269) 
buffer_load_dwordx4 v42, s[64:67], 0 offen offset: 0 lds // (op 1269) Load value direct to lds
v_add_co_u32 v42, s[76:77], v42, v40 // (op 1269) least significant half
v_addc_co_u32 v43, s[76:77], 0, v43, s[76:77] // (op 1269) most significant half

s_add_u32 m0, m0, 1024 // (op 1269) 
buffer_load_dwordx4 v42, s[64:67], 0 offen offset: 0 lds // (op 1269) Load value direct to lds




v_mov_b64 v[40:41], s[38:39] // (op 968) convert

v_mov_b32 v27, v37 // (op 968) reg expression
buffer_load_ubyte v42, v27, s[72:75], 0 offen offset: 0 // (op 968) Load value
v_add_u32 v27, v27, v40 // (op 968) 
buffer_load_ubyte v43, v27, s[72:75], 0 offen offset: 0 // (op 968) Load value


v_and_b32 v27, -16, v1 // (op 1888) 
v_sub_u32 v34, v1, v27 // (op 1888) 

v_lshlrev_b32 v27, 4, v34 // (op 1888) 

v_lshrrev_b32 v34, 4, v1 // (op 1888) 
v_lshl_add_u32 v39, v34, 8, v27 // (op 1888) 


s_nop 0
v_readfirstlane_b32 s54, v39 // (op 1888) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 2847) 

s_lshl_b64 s[76:77], s[24:25], 2 // (op 1274) 
v_mov_b64 v[40:41], s[76:77] // (op 1274) convert


v_mov_b64 v[44:45], v[24:25] // (op 1274) 
s_add_u32 s55, 4096, s54 // (op 1274) 
s_mov_b32 m0, s55 // (op 1274) reg expression
buffer_load_dwordx4 v44, s[56:59], 0 offen offset: 0 lds // (op 1274) Load value direct to lds
v_add_co_u32 v44, s[76:77], v44, v40 // (op 1274) least significant half
v_addc_co_u32 v45, s[76:77], 0, v45, s[76:77] // (op 1274) most significant half

s_add_u32 m0, m0, 1024 // (op 1274) 
buffer_load_dwordx4 v44, s[56:59], 0 offen offset: 0 lds // (op 1274) Load value direct to lds
v_add_co_u32 v44, s[76:77], v44, v40 // (op 1274) least significant half
v_addc_co_u32 v45, s[76:77], 0, v45, s[76:77] // (op 1274) most significant half

s_add_u32 m0, m0, 1024 // (op 1274) 
buffer_load_dwordx4 v44, s[56:59], 0 offen offset: 0 lds // (op 1274) Load value direct to lds
v_add_co_u32 v44, s[76:77], v44, v40 // (op 1274) least significant half
v_addc_co_u32 v45, s[76:77], 0, v45, s[76:77] // (op 1274) most significant half

s_add_u32 m0, m0, 1024 // (op 1274) 
buffer_load_dwordx4 v44, s[56:59], 0 offen offset: 0 lds // (op 1274) Load value direct to lds





v_mov_b64 v[40:41], s[34:35] // (op 974) convert

v_mov_b32 v27, v35 // (op 974) 
buffer_load_ubyte v44, v27, s[68:71], 0 offen offset: 0 // (op 974) Load value
v_add_u32 v27, v27, v40 // (op 974) 
buffer_load_ubyte v45, v27, s[68:71], 0 offen offset: 0 // (op 974) Load value


v_lshlrev_b32 v27, 4, v1 // (op 2668) 
s_nop 0
v_readfirstlane_b32 s54, v27 // (op 2668) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 3046) 

s_lshl_b64 s[76:77], s[30:31], 6 // (op 1279) 
v_mov_b64 v[40:41], s[76:77] // (op 1279) convert


s_lshl_b64 s[76:77], s[30:31], 8 // (op 1279) 
v_mov_b32 v27, s76 // (op 1279) 
v_mov_b32 v34, s77 // (op 1279) 
v_add_co_u32 v46, s[78:79], v28, v27 // (op 1279) least significant half
v_addc_co_u32 v47, s[78:79], v29, v34, s[78:79] // (op 1279) most significant half




s_add_u32 s55, 8192, s54 // (op 1279) 
s_mov_b32 m0, s55 // (op 1279) reg expression
buffer_load_dwordx4 v46, s[64:67], 0 offen offset: 0 lds // (op 1279) Load value direct to lds
v_add_co_u32 v46, s[76:77], v46, v40 // (op 1279) least significant half
v_addc_co_u32 v47, s[76:77], 0, v47, s[76:77] // (op 1279) most significant half

s_add_u32 m0, m0, 1024 // (op 1279) 
buffer_load_dwordx4 v46, s[64:67], 0 offen offset: 0 lds // (op 1279) Load value direct to lds
v_add_co_u32 v46, s[76:77], v46, v40 // (op 1279) least significant half
v_addc_co_u32 v47, s[76:77], 0, v47, s[76:77] // (op 1279) most significant half

s_add_u32 m0, m0, 1024 // (op 1279) 
buffer_load_dwordx4 v46, s[64:67], 0 offen offset: 0 lds // (op 1279) Load value direct to lds
v_add_co_u32 v46, s[76:77], v46, v40 // (op 1279) least significant half
v_addc_co_u32 v47, s[76:77], 0, v47, s[76:77] // (op 1279) most significant half

s_add_u32 m0, m0, 1024 // (op 1279) 
buffer_load_dwordx4 v46, s[64:67], 0 offen offset: 0 lds // (op 1279) Load value direct to lds






v_mov_b64 v[28:29], s[38:39] // (op 980) convert

s_lshl_b32 s54, s38, 3 // (op 980) 
v_add_u32 v27, v37, s54 // (op 980) 

buffer_load_ubyte v40, v27, s[72:75], 0 offen offset: 0 // (op 980) Load value
v_add_u32 v27, v27, v28 // (op 980) 
buffer_load_ubyte v41, v27, s[72:75], 0 offen offset: 0 // (op 980) Load value



v_and_b32 v34, -16, v1 // (op 2629) 
v_sub_u32 v37, v1, v34 // (op 2629) 

v_lshlrev_b32 v34, 4, v37 // (op 2629) 

v_lshrrev_b32 v37, 4, v1 // (op 2629) 
v_lshl_add_u32 v39, v37, 8, v34 // (op 2629) 


s_nop 0
v_readfirstlane_b32 s54, v39 // (op 2629) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 3036) 

s_lshl_b64 s[76:77], s[24:25], 2 // (op 1281) 
v_mov_b64 v[28:29], s[76:77] // (op 1281) convert


v_mov_b64 v[46:47], v[24:25] // (op 1281) 
s_add_u32 s55, 12288, s54 // (op 1281) 
s_mov_b32 m0, s55 // (op 1281) reg expression
s_mov_b32 s76, 256 // (op 1281) call()
buffer_load_dwordx4 v46, s[56:59], s76 offen offset: 0 lds // (op 1281) Load value direct to lds
v_add_co_u32 v46, s[78:79], v46, v28 // (op 1281) least significant half
v_addc_co_u32 v47, s[78:79], 0, v47, s[78:79] // (op 1281) most significant half

s_add_u32 m0, m0, 1024 // (op 1281) 
buffer_load_dwordx4 v46, s[56:59], s76 offen offset: 0 lds // (op 1281) Load value direct to lds
v_add_co_u32 v46, s[78:79], v46, v28 // (op 1281) least significant half
v_addc_co_u32 v47, s[78:79], 0, v47, s[78:79] // (op 1281) most significant half

s_add_u32 m0, m0, 1024 // (op 1281) 
buffer_load_dwordx4 v46, s[56:59], s76 offen offset: 0 lds // (op 1281) Load value direct to lds
v_add_co_u32 v46, s[78:79], v46, v28 // (op 1281) least significant half
v_addc_co_u32 v47, s[78:79], 0, v47, s[78:79] // (op 1281) most significant half

s_add_u32 m0, m0, 1024 // (op 1281) 
buffer_load_dwordx4 v46, s[56:59], s76 offen offset: 0 lds // (op 1281) Load value direct to lds







v_mov_b64 v[24:25], s[34:35] // (op 986) convert

v_mov_b32 v34, v35 // (op 986) 
buffer_load_ubyte v28, v34, s[68:71], 0 offen offset:8 // (op 986) Load value
v_add_u32 v34, v34, v24 // (op 986) 
buffer_load_ubyte v29, v34, s[68:71], 0 offen offset:8 // (op 986) Load value



v_and_b32 v27, -16, v1 // (op 2085) 
v_sub_u32 v37, v1, v27 // (op 2085) 

v_lshrrev_b32 v27, 4, v1 // (op 2085) 
v_lshl_add_u32 v39, v27, 5, v37 // (op 2085) 


s_barrier  // (op 2906) 



s_waitcnt vmcnt(19)
ds_write_b8 v39, v42 offset:16384 // (op 989) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v39, v43 offset:16400 // (op 989) Store local data 

v_and_b32 v27, -8, v1 // (op 2019) 
v_sub_u32 v37, v1, v27 // (op 2019) 

v_lshrrev_b32 v27, 3, v1 // (op 2019) 
v_lshl_add_u32 v39, v27, 4, v37 // (op 2019) 


s_barrier  // (op 2886) 



s_waitcnt vmcnt(13)
ds_write_b8 v39, v44 offset:16512 // (op 991) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v39, v45 offset:16520 // (op 991) Store local data 

s_barrier  // (op 2863) 



ds_read_b64_tr_b8 v[46:47], v20 // (op 1011) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v20 offset:128 // (op 1011) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v20 offset:1024 // (op 1011) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v20 offset:1152 // (op 1011) Transpose load local data 



ds_read_b64_tr_b8 v[54:55], v20 offset:2048 // (op 1014) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v20 offset:2176 // (op 1014) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v20 offset:3072 // (op 1014) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v20 offset:3200 // (op 1014) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 2903) 


ds_read_u8 v27, v26 offset:16384 // (op 1023) Load local data 


ds_read_u8 v37, v26 offset:16448 // (op 1026) Load local data 
s_barrier  // (op 2844) 



ds_read_b128 v[62:65], v22 offset:4096 // (op 1005) Load local data 
ds_read_b128 v[66:69], v22 offset:4160 // (op 1005) Load local data 



ds_read_b128 v[70:73], v22 offset:4224 // (op 1008) Load local data 
ds_read_b128 v[74:77], v22 offset:4288 // (op 1008) Load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 2883) 



ds_read_u8 v39, v23 offset:16512 // (op 1017) Load local data 



ds_read_u8 v24, v23 offset:16516 // (op 1020) Load local data 
v_and_b32 v25, -16, v1 // (op 1860) 
v_sub_u32 v34, v1, v25 // (op 1860) 

v_lshlrev_b32 v25, 4, v34 // (op 1860) 

v_lshrrev_b32 v34, 4, v1 // (op 1860) 
v_lshl_add_u32 v35, v34, 8, v25 // (op 1860) 


s_nop 0
v_readfirstlane_b32 s54, v35 // (op 1860) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v25, 4, v1 // (op 1925) 
s_nop 0
v_readfirstlane_b32 s55, v25 // (op 1925) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v25, -8, v1 // (op 1990) 
v_sub_u32 v34, v1, v25 // (op 1990) 

v_lshrrev_b32 v25, 3, v1 // (op 1990) 
v_lshl_add_u32 v35, v25, 4, v34 // (op 1990) 


v_and_b32 v25, -16, v1 // (op 2056) 
v_sub_u32 v34, v1, v25 // (op 2056) 

v_lshrrev_b32 v25, 4, v1 // (op 2056) 
v_lshl_add_u32 v78, v25, 5, v34 // (op 2056) 


v_and_b32 v25, -16, v1 // (op 2614) 
v_sub_u32 v34, v1, v25 // (op 2614) 

v_lshlrev_b32 v25, 4, v34 // (op 2614) 

v_lshrrev_b32 v34, 4, v1 // (op 2614) 
v_lshl_add_u32 v79, v34, 8, v25 // (op 2614) 


s_nop 0
v_readfirstlane_b32 s76, v79 // (op 2614) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v25, 4, v1 // (op 2653) 
s_nop 0
v_readfirstlane_b32 s77, v25 // (op 2653) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v25, -8, v1 // (op 2692) 
v_sub_u32 v34, v1, v25 // (op 2692) 

v_lshrrev_b32 v25, 3, v1 // (op 2692) 
v_lshl_add_u32 v79, v25, 4, v34 // (op 2692) 


v_and_b32 v25, -16, v1 // (op 2716) 
v_sub_u32 v34, v1, v25 // (op 2716) 

v_lshrrev_b32 v25, 4, v1 // (op 2716) 
v_lshl_add_u32 v80, v25, 5, v34 // (op 2716) 


s_mov_b32 s1, 0 // (op 59) call()
s_and_b32 s78, s7, 4294967294 // (op 58) 
s_cmp_lt_u32 s1, s78 // (op 58) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopBottom_KLoop_58 // (op 58) Condition: Top (jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopBottom_KLoop_58 if false)
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopTop_KLoop_58:
s_add_u32 s1, s1, 2 // (op 60) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:69], v[46:53], a[0:3], v39, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 88) 
s_barrier  // (op 2869) 

s_lshl_b64 s[78:79], s[30:31], 6 // (op 1208) 
v_mov_b64 v[82:83], s[78:79] // (op 1208) convert


v_mov_b64 v[84:85], v[30:31] // (op 1208) 
s_lshl_b64 s[78:79], s[30:31], 8 // (op 1208) 
s_lshl_b64 s[80:81], s[78:79], 1 // (op 1208) 

v_mov_b32 v25, s80 // (op 1208) 
v_mov_b32 v34, s81 // (op 1208) 
v_add_co_u32 v86, s[78:79], v84, v25 // (op 1208) least significant half
v_addc_co_u32 v87, s[78:79], v85, v34, s[78:79] // (op 1208) most significant half




s_mov_b32 m0, s55 // (op 1208) reg expression
buffer_load_dwordx4 v86, s[64:67], 0 offen offset: 0 lds // (op 1208) Load value direct to lds
v_add_co_u32 v86, s[78:79], v86, v82 // (op 1208) least significant half
v_addc_co_u32 v87, s[78:79], 0, v87, s[78:79] // (op 1208) most significant half

s_add_u32 m0, m0, 1024 // (op 1208) 
buffer_load_dwordx4 v86, s[64:67], 0 offen offset: 0 lds // (op 1208) Load value direct to lds
v_add_co_u32 v86, s[78:79], v86, v82 // (op 1208) least significant half
v_addc_co_u32 v87, s[78:79], 0, v87, s[78:79] // (op 1208) most significant half

s_add_u32 m0, m0, 1024 // (op 1208) 
buffer_load_dwordx4 v86, s[64:67], 0 offen offset: 0 lds // (op 1208) Load value direct to lds
v_add_co_u32 v86, s[78:79], v86, v82 // (op 1208) least significant half
v_addc_co_u32 v87, s[78:79], 0, v87, s[78:79] // (op 1208) most significant half

s_add_u32 m0, m0, 1024 // (op 1208) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[70:77], v[54:61], a[0:3], v24, v37 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 105) 
buffer_load_dwordx4 v86, s[64:67], 0 offen offset: 0 lds // (op 1208) Load value direct to lds



v_mov_b64 v[82:83], s[38:39] // (op 25) convert

v_mov_b32 v25, v38 // (op 25) 
s_lshl_b32 s78, s38, 4 // (op 25) 
v_add_u32 v34, v25, s78 // (op 25) 

buffer_load_ubyte v42, v34, s[72:75], 0 offen offset: 0 // (op 25) Load value
v_add_u32 v34, v34, v82 // (op 25) 
buffer_load_ubyte v43, v34, s[72:75], 0 offen offset: 0 // (op 25) Load value


s_barrier  // (op 2850) 

s_lshl_b64 s[78:79], s[24:25], 2 // (op 1203) 
v_mov_b64 v[82:83], s[78:79] // (op 1203) convert


v_mov_b64 v[86:87], v[32:33] // (op 1203) 
v_mov_b64 v[88:89], v[86:87] // (op 1203) 
s_add_u32 s78, 4096, s54 // (op 1203) 
s_mov_b32 m0, s78 // (op 1203) reg expression
s_mov_b32 s79, 512 // (op 1203) call()
buffer_load_dwordx4 v88, s[56:59], s79 offen offset: 0 lds // (op 1203) Load value direct to lds
v_add_co_u32 v88, s[80:81], v88, v82 // (op 1203) least significant half
v_addc_co_u32 v89, s[80:81], 0, v89, s[80:81] // (op 1203) most significant half

s_add_u32 m0, m0, 1024 // (op 1203) 
buffer_load_dwordx4 v88, s[56:59], s79 offen offset: 0 lds // (op 1203) Load value direct to lds
v_add_co_u32 v88, s[80:81], v88, v82 // (op 1203) least significant half
v_addc_co_u32 v89, s[80:81], 0, v89, s[80:81] // (op 1203) most significant half

s_add_u32 m0, m0, 1024 // (op 1203) 
buffer_load_dwordx4 v88, s[56:59], s79 offen offset: 0 lds // (op 1203) Load value direct to lds
v_add_co_u32 v88, s[80:81], v88, v82 // (op 1203) least significant half
v_addc_co_u32 v89, s[80:81], 0, v89, s[80:81] // (op 1203) most significant half

s_add_u32 m0, m0, 1024 // (op 1203) 
buffer_load_dwordx4 v88, s[56:59], s79 offen offset: 0 lds // (op 1203) Load value direct to lds





v_mov_b64 v[82:83], s[34:35] // (op 17) convert

v_mov_b32 v34, v36 // (op 17) 
v_mov_b32 v81, v34 // (op 17) 
buffer_load_ubyte v44, v81, s[68:71], 0 offen offset:16 // (op 17) Load value
v_add_u32 v81, v81, v82 // (op 17) 
buffer_load_ubyte v45, v81, s[68:71], 0 offen offset:16 // (op 17) Load value


s_barrier  // (op 3061) 



s_waitcnt vmcnt(19)
ds_write_b8 v80, v40 offset:16640 // (op 805) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v80, v41 offset:16656 // (op 805) Store local data 
s_barrier  // (op 3055) 



s_waitcnt vmcnt(13)
ds_write_b8 v79, v28 offset:16768 // (op 804) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v79, v29 offset:16776 // (op 804) Store local data 
s_barrier  // (op 3042) 



ds_read_b64_tr_b8 v[46:47], v13 offset:8192 // (op 788) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v13 offset:8320 // (op 788) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v13 offset:9216 // (op 788) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v13 offset:9344 // (op 788) Transpose load local data 



ds_read_b64_tr_b8 v[88:89], v13 offset:10240 // (op 797) Transpose load local data 
ds_read_b64_tr_b8 v[90:91], v13 offset:10368 // (op 797) Transpose load local data 
ds_read_b64_tr_b8 v[92:93], v13 offset:11264 // (op 797) Transpose load local data 
ds_read_b64_tr_b8 v[94:95], v13 offset:11392 // (op 797) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 3058) 


ds_read_u8 v27, v19 offset:16640 // (op 792) Load local data 


ds_read_u8 v81, v19 offset:16704 // (op 801) Load local data 
s_barrier  // (op 3033) 
s_barrier  // (op 3208) 



ds_read_b128 v[62:65], v14 offset:12288 // (op 786) Load local data 
ds_read_b128 v[66:69], v14 offset:12352 // (op 786) Load local data 



ds_read_b128 v[70:73], v14 offset:12416 // (op 795) Load local data 
ds_read_b128 v[74:77], v14 offset:12480 // (op 795) Load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 3052) 



ds_read_u8 v39, v16 offset:16768 // (op 790) Load local data 



ds_read_u8 v24, v16 offset:16772 // (op 799) Load local data 
s_barrier  // (op 3204) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:69], v[46:53], a[0:3], v39, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 793) 
s_barrier  // (op 3049) 

s_lshl_b64 s[78:79], s[30:31], 6 // (op 1264) 
v_mov_b64 v[82:83], s[78:79] // (op 1264) convert


s_lshl_b64 s[78:79], s[30:31], 8 // (op 1264) 
s_mul_i32 s81, 3, s79 // (op 1264) most significant: low of low * high
s_mul_i32 s80, 3, s78 // (op 1264) least significant
s_mul_hi_u32 s82, 3, s78 // (op 1264) most significant: high of low * low
s_add_u32 s81, s81, s82 // (op 1264) most significant: sum


v_mov_b32 v96, s80 // (op 1264) 
v_mov_b32 v97, s81 // (op 1264) 
v_add_co_u32 v98, s[78:79], v84, v96 // (op 1264) least significant half
v_addc_co_u32 v99, s[78:79], v85, v97, s[78:79] // (op 1264) most significant half




s_add_u32 s78, 8192, s77 // (op 1264) 
s_nop 0
s_mov_b32 m0, s78 // (op 1264) reg expression
buffer_load_dwordx4 v98, s[64:67], 0 offen offset: 0 lds // (op 1264) Load value direct to lds
v_add_co_u32 v98, s[80:81], v98, v82 // (op 1264) least significant half
v_addc_co_u32 v99, s[80:81], 0, v99, s[80:81] // (op 1264) most significant half

s_add_u32 m0, m0, 1024 // (op 1264) 
buffer_load_dwordx4 v98, s[64:67], 0 offen offset: 0 lds // (op 1264) Load value direct to lds
v_add_co_u32 v98, s[80:81], v98, v82 // (op 1264) least significant half
v_addc_co_u32 v99, s[80:81], 0, v99, s[80:81] // (op 1264) most significant half

s_add_u32 m0, m0, 1024 // (op 1264) 
buffer_load_dwordx4 v98, s[64:67], 0 offen offset: 0 lds // (op 1264) Load value direct to lds
v_add_co_u32 v98, s[80:81], v98, v82 // (op 1264) least significant half
v_addc_co_u32 v99, s[80:81], 0, v99, s[80:81] // (op 1264) most significant half

s_add_u32 m0, m0, 1024 // (op 1264) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[70:77], v[88:95], a[0:3], v24, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 802) 
buffer_load_dwordx4 v98, s[64:67], 0 offen offset: 0 lds // (op 1264) Load value direct to lds







v_mov_b64 v[84:85], s[38:39] // (op 808) convert

s_lshl_b32 s78, s38, 3 // (op 808) 
s_mul_i32 s79, 3, s78 // (op 808) 

v_add_u32 v81, v25, s79 // (op 808) 

buffer_load_ubyte v40, v81, s[72:75], 0 offen offset: 0 // (op 808) Load value
v_add_u32 v81, v81, v84 // (op 808) 
s_lshl_b64 s[78:79], s[30:31], 8 // (op 1500) 
buffer_load_ubyte v41, v81, s[72:75], 0 offen offset: 0 // (op 808) Load value


s_lshl_b64 s[80:81], s[78:79], 1 // (op 1500) 


v_mov_b32 v25, s80 // (op 1500) 
s_barrier  // (op 3039) 
s_lshl_b32 s78, s38, 4 // (op 1756) 

v_add_u32 v38, v38, s78 // (op 1756) 

v_mov_b32 v81, s81 // (op 1500) 
s_lshl_b64 s[78:79], s[24:25], 2 // (op 1259) 
v_mov_b64 v[82:83], s[78:79] // (op 1259) convert


v_mov_b64 v[84:85], v[86:87] // (op 1259) 
s_add_u32 s78, 12288, s76 // (op 1259) 
s_mov_b32 m0, s78 // (op 1259) reg expression
s_mov_b32 s79, 768 // (op 1259) call()
v_add_co_u32 v30, s[82:83], v30, v25 // (op 1500) least significant half
v_addc_co_u32 v31, s[82:83], v31, v81, s[82:83] // (op 1500) most significant half




buffer_load_dwordx4 v84, s[56:59], s79 offen offset: 0 lds // (op 1259) Load value direct to lds
v_add_co_u32 v84, s[80:81], v84, v82 // (op 1259) least significant half
v_addc_co_u32 v85, s[80:81], 0, v85, s[80:81] // (op 1259) most significant half

s_add_u32 m0, m0, 1024 // (op 1259) 
buffer_load_dwordx4 v84, s[56:59], s79 offen offset: 0 lds // (op 1259) Load value direct to lds
v_add_co_u32 v84, s[80:81], v84, v82 // (op 1259) least significant half
v_addc_co_u32 v85, s[80:81], 0, v85, s[80:81] // (op 1259) most significant half

s_add_u32 m0, m0, 1024 // (op 1259) 
buffer_load_dwordx4 v84, s[56:59], s79 offen offset: 0 lds // (op 1259) Load value direct to lds
v_add_co_u32 v84, s[80:81], v84, v82 // (op 1259) least significant half
v_addc_co_u32 v85, s[80:81], 0, v85, s[80:81] // (op 1259) most significant half

s_add_u32 m0, m0, 1024 // (op 1259) 
buffer_load_dwordx4 v84, s[56:59], s79 offen offset: 0 lds // (op 1259) Load value direct to lds






v_add_co_u32 v32, vcc, 512, v32 // (op 1369) least significant half
v_addc_co_u32 v33, vcc, 0, v33, vcc // (op 1369) most significant half
v_mov_b64 v[82:83], s[34:35] // (op 807) convert

v_mov_b32 v25, v34 // (op 807) 
buffer_load_ubyte v28, v25, s[68:71], 0 offen offset:24 // (op 807) Load value
v_add_u32 v25, v25, v82 // (op 807) 
buffer_load_ubyte v29, v25, s[68:71], 0 offen offset:24 // (op 807) Load value



s_barrier  // (op 2909) 
v_add_u32 v36, 16, v36 // (op 1628) 



s_waitcnt vmcnt(19)
ds_write_b8 v78, v42 offset:16384 // (op 30) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v78, v43 offset:16400 // (op 30) Store local data 
s_barrier  // (op 2889) 



s_waitcnt vmcnt(13)
ds_write_b8 v35, v44 offset:16512 // (op 21) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v35, v45 offset:16520 // (op 21) Store local data 
s_barrier  // (op 2875) 



ds_read_b64_tr_b8 v[46:47], v0 // (op 75) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v0 offset:128 // (op 75) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v0 offset:1024 // (op 75) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v0 offset:1152 // (op 75) Transpose load local data 



ds_read_b64_tr_b8 v[54:55], v0 offset:2048 // (op 95) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v0 offset:2176 // (op 95) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v0 offset:3072 // (op 95) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v0 offset:3200 // (op 95) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 2916) 


ds_read_u8 v27, v7 offset:16384 // (op 85) Load local data 


ds_read_u8 v37, v7 offset:16448 // (op 103) Load local data 
s_barrier  // (op 2856) 



ds_read_b128 v[62:65], v3 offset:4096 // (op 70) Load local data 
ds_read_b128 v[66:69], v3 offset:4160 // (op 70) Load local data 



ds_read_b128 v[70:73], v3 offset:4224 // (op 91) Load local data 
ds_read_b128 v[74:77], v3 offset:4288 // (op 91) Load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 2896) 



ds_read_u8 v39, v4 offset:16512 // (op 80) Load local data 



ds_read_u8 v24, v4 offset:16516 // (op 99) Load local data 
s_and_b32 s78, s7, 4294967294 // (op 58) 
s_cmp_lt_u32 s1, s78 // (op 58) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopTop_KLoop_58 // (op 58) Condition: Bottom (jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopTop_KLoop_58 if true)
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopBottom_KLoop_58:














s_barrier  // (op 3072) 
s_and_b32 s54, s7, 4294967294 // (op 1332) 
v_lshrrev_b32 v25, 4, v1 // (op 1332) 
s_add_u32 s55, s0, s54 // (op 1332) 

v_and_b32 v38, -16, v1 // (op 1332) 
v_sub_u32 v30, v1, v38 // (op 1332) 

s_mul_i32 s54, s16, s8 // (op 1332) 
s_add_u32 s54, s54, s55 // (op 1332) 

v_lshlrev_b32 v38, 4, v30 // (op 1332) 

s_mul_hi_u32 s55, s54, s13 // (op 1332) 
s_sub_u32 s76, s54, s55 // (op 1332) 
s_lshr_b32 s77, s76, 1 // (op 1332) 

s_add_u32 s76, s77, s55 // (op 1332) 


s_lshr_b32 s55, s76, s11 // (op 1332) 

s_and_b32 s76, s11, 2147483648 // (op 1332) 
s_cmp_eq_i32 s76, 0 // (op 1332) 
s_mov_b32 s77, scc // (op 1332) 

s_cmp_eq_i32 s77, 0 // (op 1332) 
s_cselect_b32 s76, s54, s55 // (op 1332) 


s_mul_hi_u32 s55, s76, s12 // (op 1332) 
s_sub_u32 s77, s76, s55 // (op 1332) 
s_lshr_b32 s78, s77, 1 // (op 1332) 

s_add_u32 s77, s78, s55 // (op 1332) 


s_lshr_b32 s55, s77, s10 // (op 1332) 

s_and_b32 s77, s10, 2147483648 // (op 1332) 
s_cmp_eq_i32 s77, 0 // (op 1332) 
s_mov_b32 s78, scc // (op 1332) 

s_cmp_eq_i32 s78, 0 // (op 1332) 
s_cselect_b32 s77, s76, s55 // (op 1332) 


v_lshl_add_u32 v30, s77, 4, v25 // (op 1332) 


s_ashr_i64 s[78:79], s[2:3], 8 // (op 1332) 
s_mul_i32 s55, s76, s78 // (op 1332) 


s_sub_u32 s76, s54, s55 // (op 1332) 


v_lshl_add_u32 v25, s76, 8, v38 // (op 1332) 


v_mov_b32 v43, 0 // (op 1332) convert
v_mov_b32 v42, v25 // (op 1332) convert

v_mul_lo_u32 v45, v30, s25 // (op 1332) most significant: low of low * high
v_mul_hi_u32 v25, v30, s24 // (op 1332) most significant: high of low * low
v_mul_lo_u32 v44, v30, s24 // (op 1332) least significant: low of low * low
v_add_u32 v45, v45, v25 // (op 1332) most significant: sum

v_add_co_u32 v44, s[54:55], v44, v42 // (op 1332) least significant half
v_addc_co_u32 v45, s[54:55], v45, v43, s[54:55] // (op 1332) most significant half



s_mul_i32 s55, s24, s29 // (op 1334) most significant: low of low * high
s_mul_i32 s76, s25, s28 // (op 1334) most significant: low of high * low
s_mul_i32 s54, s24, s28 // (op 1334) least significant
s_mul_hi_u32 s77, s24, s28 // (op 1334) most significant: high of low * low
s_add_u32 s55, s55, s76 // (op 1334) most significant: sum
s_add_u32 s55, s55, s77 // (op 1334) most significant: sum


s_mov_b32 s56, s26 // (op 1334) reg expression
s_mov_b32 s57, s27 // (op 1334) reg expression
s_mov_b32 s58, s54 // (op 1334) reg expression
s_mov_b32 s59, 131072 // (op 1334) call()

s_and_b32 s54, s7, 4294967294 // (op 1466) 
s_add_u32 s55, s0, s54 // (op 1466) 

s_mul_i32 s54, s16, s8 // (op 1466) 
s_add_u32 s54, s54, s55 // (op 1466) 

s_mul_hi_u32 s55, s54, s13 // (op 1466) 
s_sub_u32 s76, s54, s55 // (op 1466) 
s_lshr_b32 s77, s76, 1 // (op 1466) 

s_add_u32 s76, s77, s55 // (op 1466) 


s_lshr_b32 s55, s76, s11 // (op 1466) 

s_and_b32 s76, s11, 2147483648 // (op 1466) 
s_cmp_eq_i32 s76, 0 // (op 1466) 
s_mov_b32 s77, scc // (op 1466) 

s_cmp_eq_i32 s77, 0 // (op 1466) 
s_cselect_b32 s76, s54, s55 // (op 1466) 


s_mul_hi_u32 s55, s76, s12 // (op 1466) 
s_sub_u32 s77, s76, s55 // (op 1466) 
s_lshr_b32 s78, s77, 1 // (op 1466) 

s_add_u32 s77, s78, s55 // (op 1466) 


s_lshr_b32 s55, s77, s10 // (op 1466) 

s_and_b32 s77, s10, 2147483648 // (op 1466) 
s_cmp_eq_i32 s77, 0 // (op 1466) 
s_mov_b32 s78, scc // (op 1466) 

s_cmp_eq_i32 s78, 0 // (op 1466) 
s_cselect_b32 s77, s76, s55 // (op 1466) 


s_mul_i32 s55, s77, s14 // (op 1466) 

s_sub_u32 s77, s76, s55 // (op 1466) 

s_lshl_b32 s55, s77, 4 // (op 1466) 

s_mov_b32 s79, 0 // (op 1466) convert
s_mov_b32 s78, s55 // (op 1466) convert

s_ashr_i64 s[80:81], s[2:3], 8 // (op 1466) 
s_mul_i32 s55, s76, s80 // (op 1466) 


s_sub_u32 s76, s54, s55 // (op 1466) 


v_lshl_add_u32 v25, s76, 8, v1 // (op 1466) 

v_mul_lo_u32 v43, v25, s31 // (op 1466) most significant: low of low * high
v_mul_hi_u32 v38, v25, s30 // (op 1466) most significant: high of low * low
v_mul_lo_u32 v42, v25, s30 // (op 1466) least significant: low of low * low
v_add_u32 v43, v43, v38 // (op 1466) most significant: sum

v_mov_b32 v38, s78 // (op 1466) 
v_mov_b32 v30, s79 // (op 1466) 
v_add_co_u32 v42, s[54:55], v42, v38 // (op 1466) least significant half
v_addc_co_u32 v43, s[54:55], v43, v30, s[54:55] // (op 1466) most significant half





s_mul_i32 s55, s30, s3 // (op 1468) most significant: low of low * high
s_mul_i32 s76, s31, s2 // (op 1468) most significant: low of high * low
s_mul_i32 s54, s30, s2 // (op 1468) least significant
s_mul_hi_u32 s77, s30, s2 // (op 1468) most significant: high of low * low
s_add_u32 s55, s55, s76 // (op 1468) most significant: sum
s_add_u32 s55, s55, s77 // (op 1468) most significant: sum


s_mov_b32 s64, s32 // (op 1468) reg expression
s_mov_b32 s65, s33 // (op 1468) reg expression
s_mov_b32 s66, s54 // (op 1468) reg expression
s_mov_b32 s67, 131072 // (op 1468) call()

s_and_b32 s54, s7, 4294967294 // (op 1594) 
v_lshrrev_b32 v25, 3, v1 // (op 1594) 
s_add_u32 s55, s0, s54 // (op 1594) 

v_lshlrev_b32 v38, 1, v25 // (op 1594) 

v_and_b32 v25, -8, v1 // (op 1594) 
s_mul_i32 s54, s16, s8 // (op 1594) 
s_add_u32 s54, s54, s55 // (op 1594) 

v_sub_u32 v30, v1, v25 // (op 1594) 

s_mul_hi_u32 s55, s54, s13 // (op 1594) 
s_sub_u32 s76, s54, s55 // (op 1594) 
s_lshr_b32 s77, s76, 1 // (op 1594) 

s_add_u32 s76, s77, s55 // (op 1594) 


s_lshr_b32 s55, s76, s11 // (op 1594) 

s_and_b32 s76, s11, 2147483648 // (op 1594) 
s_cmp_eq_i32 s76, 0 // (op 1594) 
s_mov_b32 s77, scc // (op 1594) 

s_cmp_eq_i32 s77, 0 // (op 1594) 
s_cselect_b32 s76, s54, s55 // (op 1594) 


s_mul_hi_u32 s55, s76, s12 // (op 1594) 
s_sub_u32 s77, s76, s55 // (op 1594) 
s_lshr_b32 s78, s77, 1 // (op 1594) 

s_add_u32 s77, s78, s55 // (op 1594) 


s_lshr_b32 s55, s77, s10 // (op 1594) 

s_and_b32 s77, s10, 2147483648 // (op 1594) 
s_cmp_eq_i32 s77, 0 // (op 1594) 
s_mov_b32 s78, scc // (op 1594) 

s_cmp_eq_i32 s78, 0 // (op 1594) 
s_cselect_b32 s77, s76, s55 // (op 1594) 


v_lshl_add_u32 v25, s77, 4, v38 // (op 1594) 


s_ashr_i64 s[78:79], s[2:3], 8 // (op 1594) 
s_mul_i32 s55, s76, s78 // (op 1594) 


s_sub_u32 s76, s54, s55 // (op 1594) 


v_lshl_add_u32 v38, s76, 3, v30 // (op 1594) 


v_mul_lo_u32 v30, v25, s34 // (op 1594) 
v_add_u32 v30, v30, v38 // (op 1594) 


s_mul_i32 s55, s34, s29 // (op 1596) most significant: low of low * high
s_mul_i32 s76, s35, s28 // (op 1596) most significant: low of high * low
s_mul_i32 s54, s34, s28 // (op 1596) least significant
s_mul_hi_u32 s77, s34, s28 // (op 1596) most significant: high of low * low
s_add_u32 s55, s55, s76 // (op 1596) most significant: sum
s_add_u32 s55, s55, s77 // (op 1596) most significant: sum


s_mov_b32 s68, s36 // (op 1596) reg expression
s_mov_b32 s69, s37 // (op 1596) reg expression
s_mov_b32 s70, s54 // (op 1596) reg expression
s_mov_b32 s71, 131072 // (op 1596) call()

s_and_b32 s54, s7, 4294967294 // (op 1722) 
v_and_b32 v25, -16, v1 // (op 1722) 
s_add_u32 s55, s0, s54 // (op 1722) 

v_sub_u32 v38, v1, v25 // (op 1722) 

v_lshrrev_b32 v25, 4, v1 // (op 1722) 
s_mul_i32 s54, s16, s8 // (op 1722) 
s_add_u32 s54, s54, s55 // (op 1722) 

v_lshlrev_b32 v31, 1, v25 // (op 1722) 

s_mul_hi_u32 s55, s54, s13 // (op 1722) 
s_sub_u32 s76, s54, s55 // (op 1722) 
s_lshr_b32 s77, s76, 1 // (op 1722) 

s_add_u32 s76, s77, s55 // (op 1722) 


s_lshr_b32 s55, s76, s11 // (op 1722) 

s_and_b32 s76, s11, 2147483648 // (op 1722) 
s_cmp_eq_i32 s76, 0 // (op 1722) 
s_mov_b32 s77, scc // (op 1722) 

s_cmp_eq_i32 s77, 0 // (op 1722) 
s_cselect_b32 s76, s54, s55 // (op 1722) 


s_mul_hi_u32 s55, s76, s12 // (op 1722) 
s_sub_u32 s77, s76, s55 // (op 1722) 
s_lshr_b32 s78, s77, 1 // (op 1722) 

s_add_u32 s77, s78, s55 // (op 1722) 


s_lshr_b32 s55, s77, s10 // (op 1722) 

s_and_b32 s77, s10, 2147483648 // (op 1722) 
s_cmp_eq_i32 s77, 0 // (op 1722) 
s_mov_b32 s78, scc // (op 1722) 

s_cmp_eq_i32 s78, 0 // (op 1722) 
s_cselect_b32 s77, s76, s55 // (op 1722) 


s_mul_i32 s55, s77, s14 // (op 1722) 

s_sub_u32 s77, s76, s55 // (op 1722) 

v_lshl_add_u32 v25, s77, 4, v38 // (op 1722) 


s_ashr_i64 s[78:79], s[2:3], 8 // (op 1722) 
s_mul_i32 s55, s76, s78 // (op 1722) 


s_sub_u32 s76, s54, s55 // (op 1722) 


v_lshl_add_u32 v38, s76, 3, v31 // (op 1722) 


v_mul_lo_u32 v31, v38, s38 // (op 1722) 
v_add_u32 v31, v31, v25 // (op 1722) 


s_mul_i32 s55, s38, s43 // (op 1724) most significant: low of low * high
s_mul_i32 s76, s39, s42 // (op 1724) most significant: low of high * low
s_mul_i32 s54, s38, s42 // (op 1724) least significant
s_mul_hi_u32 s77, s38, s42 // (op 1724) most significant: high of low * low
s_add_u32 s55, s55, s76 // (op 1724) most significant: sum
s_add_u32 s55, s55, s77 // (op 1724) most significant: sum


s_mov_b32 s72, s40 // (op 1724) reg expression
s_mov_b32 s73, s41 // (op 1724) reg expression
s_mov_b32 s74, s54 // (op 1724) reg expression
s_mov_b32 s75, 131072 // (op 1724) call()

v_and_b32 v25, -16, v1 // (op 1873) 
v_sub_u32 v38, v1, v25 // (op 1873) 

v_lshlrev_b32 v25, 4, v38 // (op 1873) 

v_lshrrev_b32 v38, 4, v1 // (op 1873) 
v_lshl_add_u32 v32, v38, 8, v25 // (op 1873) 


s_nop 0
v_readfirstlane_b32 s54, v32 // (op 1873) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v25, 4, v1 // (op 1938) 
s_nop 0
v_readfirstlane_b32 s55, v25 // (op 1938) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v25, -8, v1 // (op 2003) 
v_sub_u32 v38, v1, v25 // (op 2003) 

v_lshrrev_b32 v25, 3, v1 // (op 2003) 
v_lshl_add_u32 v32, v25, 4, v38 // (op 2003) 


v_and_b32 v25, -16, v1 // (op 2069) 
v_sub_u32 v38, v1, v25 // (op 2069) 

v_lshrrev_b32 v25, 4, v1 // (op 2069) 
v_lshl_add_u32 v33, v25, 5, v38 // (op 2069) 


s_and_b32 s76, s7, 4294967294 // (op 695) 
s_cmp_lt_u32 s76, s7 // (op 694) 










s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopBottom_KLoopTail_694 // (op 694) Condition: Top (jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopBottom_KLoopTail_694 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopTop_KLoopTail_694:
s_barrier  // (op 2853) 
s_add_u32 s76, s76, 1 // (op 696) 

s_lshl_b64 s[78:79], s[24:25], 2 // (op 1246) 
v_mov_b64 v[24:25], s[78:79] // (op 1246) convert


v_mov_b64 v[28:29], v[44:45] // (op 1246) 
v_mov_b64 v[34:35], v[28:29] // (op 1246) 
s_add_u32 s77, 4096, s54 // (op 1246) 
s_mov_b32 m0, s77 // (op 1246) reg expression
buffer_load_dwordx4 v34, s[56:59], 0 offen offset: 0 lds // (op 1246) Load value direct to lds
v_add_co_u32 v34, s[78:79], v34, v24 // (op 1246) least significant half
v_addc_co_u32 v35, s[78:79], 0, v35, s[78:79] // (op 1246) most significant half

s_add_u32 m0, m0, 1024 // (op 1246) 
buffer_load_dwordx4 v34, s[56:59], 0 offen offset: 0 lds // (op 1246) Load value direct to lds
v_add_co_u32 v34, s[78:79], v34, v24 // (op 1246) least significant half
v_addc_co_u32 v35, s[78:79], 0, v35, s[78:79] // (op 1246) most significant half

s_add_u32 m0, m0, 1024 // (op 1246) 
buffer_load_dwordx4 v34, s[56:59], 0 offen offset: 0 lds // (op 1246) Load value direct to lds
v_add_co_u32 v34, s[78:79], v34, v24 // (op 1246) least significant half
v_addc_co_u32 v35, s[78:79], 0, v35, s[78:79] // (op 1246) most significant half

s_add_u32 m0, m0, 1024 // (op 1246) 
buffer_load_dwordx4 v34, s[56:59], 0 offen offset: 0 lds // (op 1246) Load value direct to lds




v_add_co_u32 v44, vcc, 256, v44 // (op 1322) least significant half
v_addc_co_u32 v45, vcc, 0, v45, vcc // (op 1322) most significant half
s_barrier  // (op 2872) 

s_lshl_b64 s[78:79], s[30:31], 6 // (op 1252) 
v_mov_b64 v[24:25], s[78:79] // (op 1252) convert


v_mov_b64 v[28:29], v[42:43] // (op 1252) 
v_mov_b64 v[34:35], v[28:29] // (op 1252) reg expression
s_mov_b32 m0, s55 // (op 1252) reg expression
buffer_load_dwordx4 v34, s[64:67], 0 offen offset: 0 lds // (op 1252) Load value direct to lds
v_add_co_u32 v34, s[78:79], v34, v24 // (op 1252) least significant half
v_addc_co_u32 v35, s[78:79], 0, v35, s[78:79] // (op 1252) most significant half

s_add_u32 m0, m0, 1024 // (op 1252) 
buffer_load_dwordx4 v34, s[64:67], 0 offen offset: 0 lds // (op 1252) Load value direct to lds
v_add_co_u32 v34, s[78:79], v34, v24 // (op 1252) least significant half
v_addc_co_u32 v35, s[78:79], 0, v35, s[78:79] // (op 1252) most significant half

s_add_u32 m0, m0, 1024 // (op 1252) 
buffer_load_dwordx4 v34, s[64:67], 0 offen offset: 0 lds // (op 1252) Load value direct to lds
v_add_co_u32 v34, s[78:79], v34, v24 // (op 1252) least significant half
v_addc_co_u32 v35, s[78:79], 0, v35, s[78:79] // (op 1252) most significant half

s_add_u32 m0, m0, 1024 // (op 1252) 
buffer_load_dwordx4 v34, s[64:67], 0 offen offset: 0 lds // (op 1252) Load value direct to lds



s_lshl_b64 s[78:79], s[30:31], 8 // (op 1458) 
v_mov_b32 v24, s78 // (op 1458) 
v_mov_b32 v25, s79 // (op 1458) 
v_add_co_u32 v42, s[80:81], v42, v24 // (op 1458) least significant half
v_addc_co_u32 v43, s[80:81], v43, v25, s[80:81] // (op 1458) most significant half





v_mov_b64 v[24:25], s[34:35] // (op 723) convert

v_mov_b32 v27, v30 // (op 723) 
v_mov_b32 v28, v27 // (op 723) 
buffer_load_ubyte v34, v28, s[68:71], 0 offen offset: 0 // (op 723) Load value
v_add_u32 v28, v28, v24 // (op 723) 
buffer_load_ubyte v35, v28, s[68:71], 0 offen offset: 0 // (op 723) Load value



v_add_u32 v30, 8, v30 // (op 1586) 

s_barrier  // (op 2892) 
v_mov_b64 v[24:25], s[38:39] // (op 724) convert




v_mov_b32 v27, v31 // (op 724) 
v_mov_b32 v28, v27 // (op 724) reg expression
buffer_load_ubyte v36, v28, s[72:75], 0 offen offset: 0 // (op 724) Load value
v_add_u32 v28, v28, v24 // (op 724) 
buffer_load_ubyte v37, v28, s[72:75], 0 offen offset: 0 // (op 724) Load value



s_lshl_b32 s77, s38, 3 // (op 1714) 
v_add_u32 v31, v31, s77 // (op 1714) 

s_waitcnt vmcnt(3)
ds_write_b8 v32, v34 offset:16512 // (op 720) Store local data 
s_barrier  // (op 2859) 
s_waitcnt vmcnt(2)
ds_write_b8 v32, v35 offset:16520 // (op 720) Store local data 




ds_read_b128 v[46:49], v3 offset:4096 // (op 702) Load local data 
s_barrier  // (op 2912) 
ds_read_b128 v[50:53], v3 offset:4160 // (op 702) Load local data 



s_barrier  // (op 2879) 
s_waitcnt vmcnt(1)
ds_write_b8 v33, v36 offset:16384 // (op 721) Store local data 



ds_read_b64_tr_b8 v[54:55], v0 // (op 704) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v0 offset:128 // (op 704) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v0 offset:1024 // (op 704) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v0 offset:1152 // (op 704) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v33, v37 offset:16400 // (op 721) Store local data 

s_waitcnt lgkmcnt(8)
s_barrier  // (op 2899) 



ds_read_u8 v24, v4 offset:16512 // (op 706) Load local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 2919) 


ds_read_u8 v25, v7 offset:16384 // (op 708) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[46:53], v[54:61], a[0:3], v24, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 709) 







ds_read_b128 v[34:37], v3 offset:4224 // (op 711) Load local data 
ds_read_b128 v[38:41], v3 offset:4288 // (op 711) Load local data 



ds_read_b64_tr_b8 v[46:47], v0 offset:2048 // (op 713) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v0 offset:2176 // (op 713) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v0 offset:3072 // (op 713) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v0 offset:3200 // (op 713) Transpose load local data 



ds_read_u8 v24, v4 offset:16516 // (op 715) Load local data 


ds_read_u8 v25, v7 offset:16448 // (op 717) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:41], v[46:53], a[0:3], v24, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 718) 




s_cmp_lt_u32 s76, s7 // (op 694) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopTop_KLoopTail_694 // (op 694) Condition: Bottom (jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopTop_KLoopTail_694 if true)
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopBottom_KLoopTail_694:









s_barrier  // (op 3194) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_cmp_lg_u32 s18, 0 // (op 354) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalFalse_Send_Tile_354 // (op 354) Condition: False, jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalFalse_Send_Tile_354
v_lshrrev_b32 v20, 4, v1 // (op 2456) 
v_lshlrev_b32 v22, 2, v20 // (op 2456) 

v_lshl_add_u32 v20, s16, 4, v22 // (op 2456) 

v_and_b32 v22, -16, v1 // (op 2456) 
v_sub_u32 v23, v1, v22 // (op 2456) 

v_lshl_add_u32 v22, v20, 4, v23 // (op 2456) 


v_lshlrev_b32 v20, 2, v22 // (op 2456) 

s_lshl_b32 s54, s15, 10 // (op 2458) 
s_mov_b32 s76, s44 // (op 2458) reg expression
s_mov_b32 s77, s45 // (op 2458) reg expression
s_mov_b32 s78, s54 // (op 2458) reg expression
s_mov_b32 s79, 131072 // (op 2458) call()




buffer_store_dword a0, v20, s[76:79], 0 offen sc1 // (op 351) Store value
buffer_store_dword a1, v20, s[76:79], 0 offen offset:64 sc1 // (op 351) Store value
buffer_store_dword a2, v20, s[76:79], 0 offen offset:128 sc1 // (op 351) Store value
buffer_store_dword a3, v20, s[76:79], 0 offen offset:192 sc1 // (op 351) Store value

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_barrier  // (op 355) 
v_cmp_eq_i32 vcc, v1, 0 // (op 359) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalFalse_Wave0_Store_Flag_359 // (op 359) Condition: False, jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalFalse_Wave0_Store_Flag_359
s_mov_b32 s54, 1 // (op 357) call()
s_mov_b32 s81, 0 // (op 358) convert
s_mov_b32 s80, s16 // (op 358) convert
s_lshl_b64 s[80:81], s[80:81], 2 // (op 358) 
s_add_u32 s82, s46, s80 // (op 358) Start of Int64 add, locking SCC
s_addc_u32 s83, s47, s81 // (op 358) End of Int64 add, unlocking SCC
s_store_dword s54, s[82:83], 0 glc // (op 358) Store scalar value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalBottom_Wave0_Store_Flag_359 // (op 359) Condition: Done, jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalBottom_Wave0_Store_Flag_359
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalFalse_Wave0_Store_Flag_359:
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalBottom_Wave0_Store_Flag_359:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalBottom_Send_Tile_354 // (op 354) Condition: Done, jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalBottom_Send_Tile_354
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalFalse_Send_Tile_354:
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalBottom_Send_Tile_354:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_cmp_eq_i32 s18, 0 // (op 370) 
s_mov_b32 s54, scc // (op 370) 
s_ashr_i64 s[80:81], s[2:3], 8 // (op 370) 
s_mov_b32 s55, s80 // (op 370) convert

s_sub_u32 s80, s55, 1 // (op 370) 

s_cmp_lt_u32 s19, s80 // (op 370) 
s_mov_b32 s55, scc // (op 370) 

s_and_b32 s80, s54, s55 // (op 370) 
s_cmp_lg_u32 s80, 0 // (op 370) 



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalFalse_Receive_Tile_370 // (op 370) Condition: False, jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalFalse_Receive_Tile_370
s_mov_b32 s54, 0 // (op 336) call()
s_mul_i32 s55, s54, s8 // (op 335) 
s_ashr_i64 s[80:81], s[2:3], 8 // (op 335) 
s_mov_b32 s82, s80 // (op 335) convert

s_sub_u32 s80, s82, s19 // (op 335) 

s_sub_u32 s81, s80, 1 // (op 335) 

s_cmp_lt_u32 s55, s81 // (op 335) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopBottom_ReceiveTileLoop_335 // (op 335) Condition: Top (jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopBottom_ReceiveTileLoop_335 if false)
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopTop_ReceiveTileLoop_335:
s_add_u32 s55, s16, 1 // (op 2473) 
v_lshrrev_b32 v20, 4, v1 // (op 2473) 
s_add_u32 s80, s55, s54 // (op 2473) 
s_lshl_b32 s80, s80, 4 // (op 2473) 

v_lshlrev_b32 v22, 2, v20 // (op 2473) 

v_add_lshl_u32 v20, s80, v22, 4 // (op 2473) 


v_and_b32 v22, -16, v1 // (op 2473) 
v_sub_u32 v23, v1, v22 // (op 2473) 

v_add_lshl_u32 v22, v20, v23, 2 // (op 2473) 


s_lshl_b32 s55, s15, 10 // (op 2475) 
s_mov_b32 s76, s44 // (op 2475) reg expression
s_mov_b32 s77, s45 // (op 2475) reg expression
s_mov_b32 s78, s55 // (op 2475) reg expression
s_mov_b32 s79, 131072 // (op 2475) call()

RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_DoWhileTop_Global_sync_spin_loop_374:
s_add_u32 s55, s16, 1 // (op 372) 
s_add_u32 s80, s55, s54 // (op 372) Start of Int64 add, locking SCC
s_addc_u32 s81, 0, 0 // (op 372) End of Int64 add, unlocking SCC
s_lshl_b64 s[80:81], s[80:81], 2 // (op 372) 

s_add_u32 s82, s46, s80 // (op 372) Start of Int64 add, locking SCC
s_addc_u32 s83, s47, s81 // (op 372) End of Int64 add, unlocking SCC
s_load_dword s55, s[82:83], 0 glc // (op 372) Load scalar value


s_waitcnt lgkmcnt(0)
s_cmp_eq_i32 s55, 0 // (op 374) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_DoWhileTop_Global_sync_spin_loop_374 // (op 374) Condition: Bottom (jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_DoWhileTop_Global_sync_spin_loop_374 if true)
s_barrier  // (op 378) 
v_cmp_eq_i32 vcc, v1, 0 // (op 377) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalFalse_Wave0_Reset_Flag_377 // (op 377) Condition: False, jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalFalse_Wave0_Reset_Flag_377
s_mov_b32 s55, 0 // (op 375) call()
s_add_u32 s80, s16, 1 // (op 376) 
s_add_u32 s82, s80, s54 // (op 376) Start of Int64 add, locking SCC
s_addc_u32 s83, 0, 0 // (op 376) End of Int64 add, unlocking SCC
s_lshl_b64 s[82:83], s[82:83], 2 // (op 376) 

s_add_u32 s80, s46, s82 // (op 376) Start of Int64 add, locking SCC
s_addc_u32 s81, s47, s83 // (op 376) End of Int64 add, unlocking SCC
s_store_dword s55, s[80:81], 0 glc // (op 376) Store scalar value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalBottom_Wave0_Reset_Flag_377 // (op 377) Condition: Done, jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalBottom_Wave0_Reset_Flag_377
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalFalse_Wave0_Reset_Flag_377:
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalBottom_Wave0_Reset_Flag_377:




buffer_load_dword v24, v22, s[76:79], 0 offen // (op 389) Load value
buffer_load_dword v25, v22, s[76:79], 0 offen offset:64 // (op 389) Load value
buffer_load_dword v26, v22, s[76:79], 0 offen offset:128 // (op 389) Load value
buffer_load_dword v27, v22, s[76:79], 0 offen offset:192 // (op 389) Load value
v_accvgpr_read v28, a0 // (op 392) reg expression
v_accvgpr_read v29, a1 // (op 392) reg expression
v_accvgpr_read v30, a2 // (op 392) reg expression
v_accvgpr_read v31, a3 // (op 392) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v28, v28, v24 // (op 394) 
s_waitcnt vmcnt(2)
v_add_f32 v29, v29, v25 // (op 394) 
s_waitcnt vmcnt(1)
v_add_f32 v30, v30, v26 // (op 394) 
s_waitcnt vmcnt(0)
v_add_f32 v31, v31, v27 // (op 394) 

v_accvgpr_write a0, v28 // (op 396) reg expression
v_accvgpr_write a1, v29 // (op 396) reg expression
v_accvgpr_write a2, v30 // (op 396) reg expression
v_accvgpr_write a3, v31 // (op 396) reg expression

s_add_u32 s54, s54, 1 // (op 337) 

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_mul_i32 s55, s54, s8 // (op 335) 
s_ashr_i64 s[80:81], s[2:3], 8 // (op 335) 
s_mov_b32 s82, s80 // (op 335) convert

s_sub_u32 s80, s82, s19 // (op 335) 

s_sub_u32 s81, s80, 1 // (op 335) 

s_cmp_lt_u32 s55, s81 // (op 335) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopTop_ReceiveTileLoop_335 // (op 335) Condition: Bottom (jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopTop_ReceiveTileLoop_335 if true)
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopBottom_ReceiveTileLoop_335:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalBottom_Receive_Tile_370 // (op 370) Condition: Done, jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalBottom_Receive_Tile_370
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalFalse_Receive_Tile_370:
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalBottom_Receive_Tile_370:

s_cmp_eq_i32 s18, 0 // (op 413) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalFalse_Postmaccumulation_Condition_413 // (op 413) Condition: False, jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalFalse_Postmaccumulation_Condition_413
v_cmp_eq_f32 vcc, 0.00000, v17 // (op 155) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalFalse_0_beta_155 // (op 155) Condition: False, jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalFalse_0_beta_155
v_accvgpr_read v22, a0 // (op 50) 
v_accvgpr_read v23, a1 // (op 50) 
v_accvgpr_read v24, a2 // (op 50) 
v_accvgpr_read v25, a3 // (op 50) 
v_mul_f32 v26, v18, v22 // (op 50) 
v_mul_f32 v27, v18, v23 // (op 50) 
v_mul_f32 v28, v18, v24 // (op 50) 
v_mul_f32 v29, v18, v25 // (op 50) 

v_lshrrev_b32 v20, 6, v1 // (op 2124) 
s_mul_i32 s54, s16, s8 // (op 2124) 
s_add_u32 s54, s54, s0 // (op 2124) 
v_and_b32 v22, 63, v1 // (op 2124) 
s_mul_hi_u32 s55, s54, s13 // (op 2124) 
v_lshrrev_b32 v23, 4, v22 // (op 2124) 
s_sub_u32 s76, s54, s55 // (op 2124) 
v_lshlrev_b32 v24, 2, v23 // (op 2124) 

s_lshr_b32 s77, s76, 1 // (op 2124) 

v_lshl_add_u32 v23, v20, 4, v24 // (op 2124) 


s_add_u32 s76, s77, s55 // (op 2124) 


v_lshrrev_b32 v20, 2, v22 // (op 2124) 

s_lshr_b32 s55, s76, s11 // (op 2124) 

v_and_b32 v22, 3, v20 // (op 2124) 

s_and_b32 s76, s11, 2147483648 // (op 2124) 
v_and_b32 v20, 3, v1 // (op 2124) 
s_cmp_eq_i32 s76, 0 // (op 2124) 
s_mov_b32 s77, scc // (op 2124) 

v_lshl_add_u32 v24, v22, 2, v20 // (op 2124) 


s_cmp_eq_i32 s77, 0 // (op 2124) 
s_cselect_b32 s76, s54, s55 // (op 2124) 



s_mul_hi_u32 s54, s76, s12 // (op 2124) 
s_sub_u32 s55, s76, s54 // (op 2124) 
s_lshr_b32 s77, s55, 1 // (op 2124) 

s_add_u32 s55, s77, s54 // (op 2124) 


s_lshr_b32 s54, s55, s10 // (op 2124) 

s_and_b32 s55, s10, 2147483648 // (op 2124) 
s_cmp_eq_i32 s55, 0 // (op 2124) 
s_mov_b32 s77, scc // (op 2124) 

s_cmp_eq_i32 s77, 0 // (op 2124) 
s_cselect_b32 s55, s76, s54 // (op 2124) 


s_mul_i32 s54, s55, s14 // (op 2124) 
v_lshl_add_u32 v20, s55, 4, v23 // (op 2124) 


s_sub_u32 s55, s76, s54 // (op 2124) 


v_lshl_add_u32 v25, s55, 4, v24 // (op 2124) 


v_mul_lo_u32 v22, v25, s48 // (op 2124) 

v_add_lshl_u32 v23, v20, v22, 2 // (op 2124) 


s_mul_i32 s55, s48, s23 // (op 2126) most significant: low of low * high
s_mul_i32 s76, s49, s22 // (op 2126) most significant: low of high * low
s_mul_i32 s54, s48, s22 // (op 2126) least significant
s_mul_hi_u32 s77, s48, s22 // (op 2126) most significant: high of low * low
s_add_u32 s55, s55, s76 // (op 2126) most significant: sum
s_add_u32 s55, s55, s77 // (op 2126) most significant: sum


s_lshl_b64 s[76:77], s[54:55], 2 // (op 2126) 

s_mov_b32 s80, s50 // (op 2126) reg expression
s_mov_b32 s81, s51 // (op 2126) reg expression
s_mov_b32 s82, s76 // (op 2126) reg expression
s_mov_b32 s83, 131072 // (op 2126) call()




buffer_store_dwordx4 v[26:29], v23, s[80:83], 0 offen // (op 56) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalBottom_0_beta_155 // (op 155) Condition: Done, jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalBottom_0_beta_155
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalFalse_0_beta_155:



buffer_load_dwordx4 v[22:25], v21, s[60:63], 0 offen // (op 167) Load value
v_accvgpr_read v26, a0 // (op 164) 
v_accvgpr_read v27, a1 // (op 164) 
v_accvgpr_read v28, a2 // (op 164) 
v_accvgpr_read v29, a3 // (op 164) 
v_mul_f32 v30, v18, v26 // (op 164) 
v_mul_f32 v31, v18, v27 // (op 164) 
v_mul_f32 v32, v18, v28 // (op 164) 
v_mul_f32 v33, v18, v29 // (op 164) 

s_waitcnt vmcnt(0)
v_mul_f32 v26, v17, v22 // (op 168) 
v_mul_f32 v27, v17, v23 // (op 168) 
v_mul_f32 v28, v17, v24 // (op 168) 
v_mul_f32 v29, v17, v25 // (op 168) 

v_add_f32 v22, v26, v30 // (op 165) 
v_add_f32 v23, v27, v31 // (op 165) 
v_add_f32 v24, v28, v32 // (op 165) 
v_add_f32 v25, v29, v33 // (op 165) 


v_lshrrev_b32 v20, 6, v1 // (op 2141) 
s_mul_i32 s54, s16, s8 // (op 2141) 
s_add_u32 s54, s54, s0 // (op 2141) 
v_and_b32 v26, 63, v1 // (op 2141) 
s_mul_hi_u32 s55, s54, s13 // (op 2141) 
v_lshrrev_b32 v27, 4, v26 // (op 2141) 
s_sub_u32 s76, s54, s55 // (op 2141) 
v_lshlrev_b32 v28, 2, v27 // (op 2141) 

s_lshr_b32 s77, s76, 1 // (op 2141) 

v_lshl_add_u32 v27, v20, 4, v28 // (op 2141) 


s_add_u32 s76, s77, s55 // (op 2141) 


v_lshrrev_b32 v20, 2, v26 // (op 2141) 

s_lshr_b32 s55, s76, s11 // (op 2141) 

v_and_b32 v26, 3, v20 // (op 2141) 

s_and_b32 s76, s11, 2147483648 // (op 2141) 
v_and_b32 v20, 3, v1 // (op 2141) 
s_cmp_eq_i32 s76, 0 // (op 2141) 
s_mov_b32 s77, scc // (op 2141) 

v_lshl_add_u32 v28, v26, 2, v20 // (op 2141) 


s_cmp_eq_i32 s77, 0 // (op 2141) 
s_cselect_b32 s76, s54, s55 // (op 2141) 



s_mul_hi_u32 s54, s76, s12 // (op 2141) 
s_sub_u32 s55, s76, s54 // (op 2141) 
s_lshr_b32 s77, s55, 1 // (op 2141) 

s_add_u32 s55, s77, s54 // (op 2141) 


s_lshr_b32 s54, s55, s10 // (op 2141) 

s_and_b32 s55, s10, 2147483648 // (op 2141) 
s_cmp_eq_i32 s55, 0 // (op 2141) 
s_mov_b32 s77, scc // (op 2141) 

s_cmp_eq_i32 s77, 0 // (op 2141) 
s_cselect_b32 s55, s76, s54 // (op 2141) 


s_mul_i32 s54, s55, s14 // (op 2141) 
v_lshl_add_u32 v20, s55, 4, v27 // (op 2141) 


s_sub_u32 s55, s76, s54 // (op 2141) 


v_lshl_add_u32 v26, s55, 4, v28 // (op 2141) 


v_mul_lo_u32 v27, v26, s48 // (op 2141) 

v_add_lshl_u32 v26, v20, v27, 2 // (op 2141) 


s_mul_i32 s55, s48, s23 // (op 2143) most significant: low of low * high
s_mul_i32 s76, s49, s22 // (op 2143) most significant: low of high * low
s_mul_i32 s54, s48, s22 // (op 2143) least significant
s_mul_hi_u32 s77, s48, s22 // (op 2143) most significant: high of low * low
s_add_u32 s55, s55, s76 // (op 2143) most significant: sum
s_add_u32 s55, s55, s77 // (op 2143) most significant: sum


s_lshl_b64 s[76:77], s[54:55], 2 // (op 2143) 

s_mov_b32 s80, s50 // (op 2143) reg expression
s_mov_b32 s81, s51 // (op 2143) reg expression
s_mov_b32 s82, s76 // (op 2143) reg expression
s_mov_b32 s83, 131072 // (op 2143) call()




buffer_store_dwordx4 v[22:25], v26, s[80:83], 0 offen // (op 166) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalBottom_0_beta_155:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalBottom_Postmaccumulation_Condition_413 // (op 413) Condition: Done, jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalBottom_Postmaccumulation_Condition_413
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalFalse_Postmaccumulation_Condition_413:
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalBottom_Postmaccumulation_Condition_413:

s_add_u32 s0, s0, s7 // (op 328) 
s_mul_i32 s54, s8, s16 // (op 326) 
s_cmp_gt_u32 s9, s54 // (op 326) 
s_mov_b32 s55, scc // (op 326) 
s_sub_u32 s76, s9, s54 // (op 326) 

s_cmp_eq_i32 s55, 0 // (op 326) 
s_cselect_b32 s54, 0, s76 // (op 326) 


s_cmp_lt_u32 s8, s54 // (op 326) 
s_mov_b32 s55, scc // (op 326) 
s_cmp_eq_i32 s55, 0 // (op 326) 
s_cselect_b32 s76, s54, s8 // (op 326) 


s_cmp_lt_u32 s0, s76 // (op 326) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopTop_SKStreamTileLoop_326 // (op 326) Condition: Bottom (jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopTop_SKStreamTileLoop_326 if true)
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopBottom_SKStreamTileLoop_326:

















s_mov_b32 s0, 0 // (op 467) call()
s_mov_b32 s1, 0 // (op 468) call()
s_mov_b32 s0, 0 // (op 482) call()
s_mul_i32 s15, s52, s16 // (op 481) 
s_cmp_gt_u32 s53, s15 // (op 481) 
s_mov_b32 s6, scc // (op 481) 
s_sub_u32 s7, s53, s15 // (op 481) 

s_cmp_eq_i32 s6, 0 // (op 481) 
s_cselect_b32 s8, 0, s7 // (op 481) 


s_cmp_lt_u32 s52, s8 // (op 481) 
s_mov_b32 s15, scc // (op 481) 
s_cmp_eq_i32 s15, 0 // (op 481) 
s_cselect_b32 s6, s8, s52 // (op 481) 


s_cmp_lt_u32 s0, s6 // (op 481) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopBottom_DPStreamTileLoop_481 // (op 481) Condition: Top (jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopBottom_DPStreamTileLoop_481 if false)
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopTop_DPStreamTileLoop_481:
v_lshrrev_b32 v0, 6, v1 // (op 1843) 
s_mul_i32 s15, s16, s52 // (op 1843) 
s_add_u32 s15, s15, s0 // (op 1843) 
v_and_b32 v7, 63, v1 // (op 1843) 
s_add_u32 s6, s15, s9 // (op 1843) 

v_lshrrev_b32 v16, 4, v7 // (op 1843) 
s_mul_hi_u32 s15, s6, s13 // (op 1843) 
v_lshlrev_b32 v3, 2, v16 // (op 1843) 

s_sub_u32 s7, s6, s15 // (op 1843) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 1843) 


s_lshr_b32 s8, s7, 1 // (op 1843) 

v_lshrrev_b32 v0, 2, v7 // (op 1843) 

s_add_u32 s7, s8, s15 // (op 1843) 


v_and_b32 v3, 3, v0 // (op 1843) 

s_lshr_b32 s8, s7, s11 // (op 1843) 

v_and_b32 v0, 3, v1 // (op 1843) 
s_and_b32 s7, s11, 2147483648 // (op 1843) 
v_lshl_add_u32 v7, v3, 2, v0 // (op 1843) 


s_cmp_eq_i32 s7, 0 // (op 1843) 
s_mov_b32 s15, scc // (op 1843) 

s_cmp_eq_i32 s15, 0 // (op 1843) 
s_cselect_b32 s7, s6, s8 // (op 1843) 



s_mul_hi_u32 s6, s7, s12 // (op 1843) 
s_sub_u32 s8, s7, s6 // (op 1843) 
s_lshr_b32 s15, s8, 1 // (op 1843) 

s_add_u32 s8, s15, s6 // (op 1843) 


s_lshr_b32 s6, s8, s10 // (op 1843) 

s_and_b32 s8, s10, 2147483648 // (op 1843) 
s_cmp_eq_i32 s8, 0 // (op 1843) 
s_mov_b32 s15, scc // (op 1843) 

s_cmp_eq_i32 s15, 0 // (op 1843) 
s_cselect_b32 s8, s7, s6 // (op 1843) 


s_mul_i32 s6, s8, s14 // (op 1843) 
v_lshl_add_u32 v0, s8, 4, v4 // (op 1843) 


s_sub_u32 s8, s7, s6 // (op 1843) 


v_lshl_add_u32 v16, s8, 4, v7 // (op 1843) 


v_mul_lo_u32 v7, v16, s4 // (op 1843) 

v_add_lshl_u32 v16, v0, v7, 2 // (op 1843) 


s_mul_i32 s19, s4, s23 // (op 1845) most significant: low of low * high
s_mul_i32 s15, s5, s22 // (op 1845) most significant: low of high * low
s_mul_i32 s18, s4, s22 // (op 1845) least significant
s_mul_hi_u32 s6, s4, s22 // (op 1845) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1845) most significant: sum
s_add_u32 s19, s19, s6 // (op 1845) most significant: sum


s_lshl_b64 s[54:55], s[18:19], 2 // (op 1845) 

s_mov_b32 s60, s20 // (op 1845) reg expression
s_mov_b32 s61, s21 // (op 1845) reg expression
s_mov_b32 s62, s54 // (op 1845) reg expression
s_mov_b32 s63, 131072 // (op 1845) call()

v_lshrrev_b32 v0, 6, v1 // (op 2244) 
v_and_b32 v7, 15, v1 // (op 2244) 
v_lshl_add_u32 v3, v0, 4, v7 // (op 2244) 


v_and_b32 v0, 48, v1 // (op 2244) 
v_lshl_add_u32 v4, v3, 8, v0 // (op 2244) 


v_and_b32 v0, 48, v1 // (op 2309) 
v_and_b32 v3, 63, v1 // (op 2309) 
v_sub_u32 v7, v3, v0 // (op 2309) 

v_lshrrev_b32 v3, 1, v7 // (op 2309) 
v_add_lshl_u32 v13, v0, v3, 4 // (op 2309) 


v_and_b32 v0, 1, v7 // (op 2309) 

v_lshlrev_b32 v3, 3, v0 // (op 2309) 

v_add_u32 v0, v13, v3 // (op 2309) 


v_lshrrev_b32 v3, 6, v1 // (op 2375) 
v_and_b32 v7, 15, v1 // (op 2375) 
v_lshl_add_u32 v13, v3, 4, v7 // (op 2375) 


v_and_b32 v3, 63, v1 // (op 2375) 
v_lshrrev_b32 v7, 4, v3 // (op 2375) 

v_lshl_add_u32 v3, v13, 3, v7 // (op 2375) 


v_and_b32 v7, 48, v1 // (op 2441) 
v_and_b32 v13, 15, v1 // (op 2441) 
v_add_u32 v14, v7, v13 // (op 2441) 


v_accvgpr_write a0, 0.00000 // (op 202) call()
v_accvgpr_write a1, 0.00000 // (op 202) call()
v_accvgpr_write a2, 0.00000 // (op 202) call()
v_accvgpr_write a3, 0.00000 // (op 202) call()
v_lshrrev_b32 v7, 4, v1 // (op 1403) 
s_mul_i32 s15, s16, s52 // (op 1403) 
s_add_u32 s15, s15, s0 // (op 1403) 
v_and_b32 v13, -16, v1 // (op 1403) 
s_add_u32 s6, s15, s9 // (op 1403) 

v_sub_u32 v19, v1, v13 // (op 1403) 

s_mul_hi_u32 s15, s6, s13 // (op 1403) 
v_lshlrev_b32 v13, 4, v19 // (op 1403) 

s_sub_u32 s7, s6, s15 // (op 1403) 
s_lshr_b32 s8, s7, 1 // (op 1403) 

s_add_u32 s7, s8, s15 // (op 1403) 


s_lshr_b32 s8, s7, s11 // (op 1403) 

s_and_b32 s7, s11, 2147483648 // (op 1403) 
s_cmp_eq_i32 s7, 0 // (op 1403) 
s_mov_b32 s15, scc // (op 1403) 

s_cmp_eq_i32 s15, 0 // (op 1403) 
s_cselect_b32 s7, s6, s8 // (op 1403) 


s_mul_hi_u32 s8, s7, s12 // (op 1403) 
s_sub_u32 s15, s7, s8 // (op 1403) 
s_lshr_b32 s18, s15, 1 // (op 1403) 

s_add_u32 s15, s18, s8 // (op 1403) 


s_lshr_b32 s8, s15, s10 // (op 1403) 

s_and_b32 s15, s10, 2147483648 // (op 1403) 
s_cmp_eq_i32 s15, 0 // (op 1403) 
s_mov_b32 s18, scc // (op 1403) 

s_cmp_eq_i32 s18, 0 // (op 1403) 
s_cselect_b32 s15, s7, s8 // (op 1403) 


v_lshl_add_u32 v19, s15, 4, v7 // (op 1403) 


s_ashr_i64 s[18:19], s[2:3], 8 // (op 1403) 
s_mul_i32 s8, s7, s18 // (op 1403) 


s_sub_u32 s7, s6, s8 // (op 1403) 


v_lshl_add_u32 v7, s7, 8, v13 // (op 1403) 


v_mov_b32 v21, 0 // (op 1403) convert
v_mov_b32 v20, v7 // (op 1403) convert

v_mul_lo_u32 v23, v19, s25 // (op 1403) most significant: low of low * high
v_mul_hi_u32 v7, v19, s24 // (op 1403) most significant: high of low * low
v_mul_lo_u32 v22, v19, s24 // (op 1403) least significant: low of low * low
v_add_u32 v23, v23, v7 // (op 1403) most significant: sum

v_add_co_u32 v22, s[18:19], v22, v20 // (op 1403) least significant half
v_addc_co_u32 v23, s[18:19], v23, v21, s[18:19] // (op 1403) most significant half



s_mul_i32 s19, s24, s29 // (op 1405) most significant: low of low * high
s_mul_i32 s15, s25, s28 // (op 1405) most significant: low of high * low
s_mul_i32 s18, s24, s28 // (op 1405) least significant
s_mul_hi_u32 s6, s24, s28 // (op 1405) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1405) most significant: sum
s_add_u32 s19, s19, s6 // (op 1405) most significant: sum


s_mov_b32 s56, s26 // (op 1405) reg expression
s_mov_b32 s57, s27 // (op 1405) reg expression
s_mov_b32 s58, s18 // (op 1405) reg expression
s_mov_b32 s59, 131072 // (op 1405) call()

v_lshrrev_b32 v7, 4, v1 // (op 1422) 
s_mul_i32 s15, s16, s52 // (op 1422) 
s_add_u32 s15, s15, s0 // (op 1422) 
v_and_b32 v13, -16, v1 // (op 1422) 
s_add_u32 s6, s15, s9 // (op 1422) 

v_sub_u32 v19, v1, v13 // (op 1422) 

s_mul_hi_u32 s15, s6, s13 // (op 1422) 
v_lshlrev_b32 v13, 4, v19 // (op 1422) 

s_sub_u32 s7, s6, s15 // (op 1422) 
s_lshr_b32 s8, s7, 1 // (op 1422) 

s_add_u32 s7, s8, s15 // (op 1422) 


s_lshr_b32 s8, s7, s11 // (op 1422) 

s_and_b32 s7, s11, 2147483648 // (op 1422) 
s_cmp_eq_i32 s7, 0 // (op 1422) 
s_mov_b32 s15, scc // (op 1422) 

s_cmp_eq_i32 s15, 0 // (op 1422) 
s_cselect_b32 s7, s6, s8 // (op 1422) 


s_mul_hi_u32 s8, s7, s12 // (op 1422) 
s_sub_u32 s15, s7, s8 // (op 1422) 
s_lshr_b32 s18, s15, 1 // (op 1422) 

s_add_u32 s15, s18, s8 // (op 1422) 


s_lshr_b32 s8, s15, s10 // (op 1422) 

s_and_b32 s15, s10, 2147483648 // (op 1422) 
s_cmp_eq_i32 s15, 0 // (op 1422) 
s_mov_b32 s18, scc // (op 1422) 

s_cmp_eq_i32 s18, 0 // (op 1422) 
s_cselect_b32 s15, s7, s8 // (op 1422) 


v_lshl_add_u32 v19, s15, 4, v7 // (op 1422) 


s_ashr_i64 s[18:19], s[2:3], 8 // (op 1422) 
s_mul_i32 s8, s7, s18 // (op 1422) 


s_sub_u32 s7, s6, s8 // (op 1422) 


v_lshl_add_u32 v7, s7, 8, v13 // (op 1422) 


v_mov_b32 v21, 0 // (op 1422) convert
v_mov_b32 v20, v7 // (op 1422) convert

v_mul_lo_u32 v25, v19, s25 // (op 1422) most significant: low of low * high
v_mul_hi_u32 v7, v19, s24 // (op 1422) most significant: high of low * low
v_mul_lo_u32 v24, v19, s24 // (op 1422) least significant: low of low * low
v_add_u32 v25, v25, v7 // (op 1422) most significant: sum

v_add_co_u32 v24, s[18:19], v24, v20 // (op 1422) least significant half
v_addc_co_u32 v25, s[18:19], v25, v21, s[18:19] // (op 1422) most significant half



s_mul_i32 s19, s24, s29 // (op 1424) most significant: low of low * high
s_mul_i32 s15, s25, s28 // (op 1424) most significant: low of high * low
s_mul_i32 s18, s24, s28 // (op 1424) least significant
s_mul_hi_u32 s6, s24, s28 // (op 1424) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1424) most significant: sum
s_add_u32 s19, s19, s6 // (op 1424) most significant: sum


s_mov_b32 s56, s26 // (op 1424) reg expression
s_mov_b32 s57, s27 // (op 1424) reg expression
s_mov_b32 s58, s18 // (op 1424) reg expression
s_mov_b32 s59, 131072 // (op 1424) call()

s_mul_i32 s15, s16, s52 // (op 1531) 
s_add_u32 s15, s15, s0 // (op 1531) 
s_add_u32 s6, s15, s9 // (op 1531) 

s_mul_hi_u32 s15, s6, s13 // (op 1531) 
s_sub_u32 s7, s6, s15 // (op 1531) 
s_lshr_b32 s8, s7, 1 // (op 1531) 

s_add_u32 s7, s8, s15 // (op 1531) 


s_lshr_b32 s8, s7, s11 // (op 1531) 

s_and_b32 s7, s11, 2147483648 // (op 1531) 
s_cmp_eq_i32 s7, 0 // (op 1531) 
s_mov_b32 s15, scc // (op 1531) 

s_cmp_eq_i32 s15, 0 // (op 1531) 
s_cselect_b32 s7, s6, s8 // (op 1531) 


s_mul_hi_u32 s8, s7, s12 // (op 1531) 
s_sub_u32 s15, s7, s8 // (op 1531) 
s_lshr_b32 s18, s15, 1 // (op 1531) 

s_add_u32 s15, s18, s8 // (op 1531) 


s_lshr_b32 s8, s15, s10 // (op 1531) 

s_and_b32 s15, s10, 2147483648 // (op 1531) 
s_cmp_eq_i32 s15, 0 // (op 1531) 
s_mov_b32 s18, scc // (op 1531) 

s_cmp_eq_i32 s18, 0 // (op 1531) 
s_cselect_b32 s15, s7, s8 // (op 1531) 


s_mul_i32 s8, s15, s14 // (op 1531) 

s_sub_u32 s15, s7, s8 // (op 1531) 

s_lshl_b32 s8, s15, 4 // (op 1531) 

s_mov_b32 s19, 0 // (op 1531) convert
s_mov_b32 s18, s8 // (op 1531) convert

s_ashr_i64 s[54:55], s[2:3], 8 // (op 1531) 
s_mul_i32 s8, s7, s54 // (op 1531) 


s_sub_u32 s7, s6, s8 // (op 1531) 


v_lshl_add_u32 v7, s7, 8, v1 // (op 1531) 

v_mul_lo_u32 v21, v7, s31 // (op 1531) most significant: low of low * high
v_mul_hi_u32 v13, v7, s30 // (op 1531) most significant: high of low * low
v_mul_lo_u32 v20, v7, s30 // (op 1531) least significant: low of low * low
v_add_u32 v21, v21, v13 // (op 1531) most significant: sum

v_mov_b32 v13, s18 // (op 1531) 
v_mov_b32 v19, s19 // (op 1531) 
v_add_co_u32 v20, s[54:55], v20, v13 // (op 1531) least significant half
v_addc_co_u32 v21, s[54:55], v21, v19, s[54:55] // (op 1531) most significant half





s_mul_i32 s19, s30, s3 // (op 1533) most significant: low of low * high
s_mul_i32 s15, s31, s2 // (op 1533) most significant: low of high * low
s_mul_i32 s18, s30, s2 // (op 1533) least significant
s_mul_hi_u32 s6, s30, s2 // (op 1533) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1533) most significant: sum
s_add_u32 s19, s19, s6 // (op 1533) most significant: sum


s_mov_b32 s64, s32 // (op 1533) reg expression
s_mov_b32 s65, s33 // (op 1533) reg expression
s_mov_b32 s66, s18 // (op 1533) reg expression
s_mov_b32 s67, 131072 // (op 1533) call()

s_mul_i32 s15, s16, s52 // (op 1550) 
s_add_u32 s15, s15, s0 // (op 1550) 
s_add_u32 s6, s15, s9 // (op 1550) 

s_mul_hi_u32 s15, s6, s13 // (op 1550) 
s_sub_u32 s7, s6, s15 // (op 1550) 
s_lshr_b32 s8, s7, 1 // (op 1550) 

s_add_u32 s7, s8, s15 // (op 1550) 


s_lshr_b32 s8, s7, s11 // (op 1550) 

s_and_b32 s7, s11, 2147483648 // (op 1550) 
s_cmp_eq_i32 s7, 0 // (op 1550) 
s_mov_b32 s15, scc // (op 1550) 

s_cmp_eq_i32 s15, 0 // (op 1550) 
s_cselect_b32 s7, s6, s8 // (op 1550) 


s_mul_hi_u32 s8, s7, s12 // (op 1550) 
s_sub_u32 s15, s7, s8 // (op 1550) 
s_lshr_b32 s18, s15, 1 // (op 1550) 

s_add_u32 s15, s18, s8 // (op 1550) 


s_lshr_b32 s8, s15, s10 // (op 1550) 

s_and_b32 s15, s10, 2147483648 // (op 1550) 
s_cmp_eq_i32 s15, 0 // (op 1550) 
s_mov_b32 s18, scc // (op 1550) 

s_cmp_eq_i32 s18, 0 // (op 1550) 
s_cselect_b32 s15, s7, s8 // (op 1550) 


s_mul_i32 s8, s15, s14 // (op 1550) 

s_sub_u32 s15, s7, s8 // (op 1550) 

s_lshl_b32 s8, s15, 4 // (op 1550) 

s_mov_b32 s19, 0 // (op 1550) convert
s_mov_b32 s18, s8 // (op 1550) convert

s_ashr_i64 s[54:55], s[2:3], 8 // (op 1550) 
s_mul_i32 s8, s7, s54 // (op 1550) 


s_sub_u32 s7, s6, s8 // (op 1550) 


v_lshl_add_u32 v7, s7, 8, v1 // (op 1550) 

v_mul_lo_u32 v27, v7, s31 // (op 1550) most significant: low of low * high
v_mul_hi_u32 v13, v7, s30 // (op 1550) most significant: high of low * low
v_mul_lo_u32 v26, v7, s30 // (op 1550) least significant: low of low * low
v_add_u32 v27, v27, v13 // (op 1550) most significant: sum

v_mov_b32 v13, s18 // (op 1550) 
v_mov_b32 v19, s19 // (op 1550) 
v_add_co_u32 v26, s[54:55], v26, v13 // (op 1550) least significant half
v_addc_co_u32 v27, s[54:55], v27, v19, s[54:55] // (op 1550) most significant half





s_mul_i32 s19, s30, s3 // (op 1552) most significant: low of low * high
s_mul_i32 s15, s31, s2 // (op 1552) most significant: low of high * low
s_mul_i32 s18, s30, s2 // (op 1552) least significant
s_mul_hi_u32 s6, s30, s2 // (op 1552) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1552) most significant: sum
s_add_u32 s19, s19, s6 // (op 1552) most significant: sum


s_mov_b32 s64, s32 // (op 1552) reg expression
s_mov_b32 s65, s33 // (op 1552) reg expression
s_mov_b32 s66, s18 // (op 1552) reg expression
s_mov_b32 s67, 131072 // (op 1552) call()

v_lshrrev_b32 v7, 3, v1 // (op 1659) 
s_mul_i32 s15, s16, s52 // (op 1659) 
s_add_u32 s15, s15, s0 // (op 1659) 
v_lshlrev_b32 v13, 1, v7 // (op 1659) 

s_add_u32 s6, s15, s9 // (op 1659) 

v_and_b32 v7, -8, v1 // (op 1659) 
s_mul_hi_u32 s15, s6, s13 // (op 1659) 
v_sub_u32 v19, v1, v7 // (op 1659) 

s_sub_u32 s7, s6, s15 // (op 1659) 
s_lshr_b32 s8, s7, 1 // (op 1659) 

s_add_u32 s7, s8, s15 // (op 1659) 


s_lshr_b32 s8, s7, s11 // (op 1659) 

s_and_b32 s7, s11, 2147483648 // (op 1659) 
s_cmp_eq_i32 s7, 0 // (op 1659) 
s_mov_b32 s15, scc // (op 1659) 

s_cmp_eq_i32 s15, 0 // (op 1659) 
s_cselect_b32 s7, s6, s8 // (op 1659) 


s_mul_hi_u32 s8, s7, s12 // (op 1659) 
s_sub_u32 s15, s7, s8 // (op 1659) 
s_lshr_b32 s18, s15, 1 // (op 1659) 

s_add_u32 s15, s18, s8 // (op 1659) 


s_lshr_b32 s8, s15, s10 // (op 1659) 

s_and_b32 s15, s10, 2147483648 // (op 1659) 
s_cmp_eq_i32 s15, 0 // (op 1659) 
s_mov_b32 s18, scc // (op 1659) 

s_cmp_eq_i32 s18, 0 // (op 1659) 
s_cselect_b32 s15, s7, s8 // (op 1659) 


v_lshl_add_u32 v7, s15, 4, v13 // (op 1659) 


s_ashr_i64 s[18:19], s[2:3], 8 // (op 1659) 
s_mul_i32 s8, s7, s18 // (op 1659) 


s_sub_u32 s7, s6, s8 // (op 1659) 


v_lshl_add_u32 v13, s7, 3, v19 // (op 1659) 


v_mul_lo_u32 v19, v7, s34 // (op 1659) 
v_add_u32 v19, v19, v13 // (op 1659) 


s_mul_i32 s19, s34, s29 // (op 1661) most significant: low of low * high
s_mul_i32 s15, s35, s28 // (op 1661) most significant: low of high * low
s_mul_i32 s18, s34, s28 // (op 1661) least significant
s_mul_hi_u32 s6, s34, s28 // (op 1661) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1661) most significant: sum
s_add_u32 s19, s19, s6 // (op 1661) most significant: sum


s_mov_b32 s68, s36 // (op 1661) reg expression
s_mov_b32 s69, s37 // (op 1661) reg expression
s_mov_b32 s70, s18 // (op 1661) reg expression
s_mov_b32 s71, 131072 // (op 1661) call()

v_lshrrev_b32 v7, 3, v1 // (op 1678) 
s_mul_i32 s15, s16, s52 // (op 1678) 
s_add_u32 s15, s15, s0 // (op 1678) 
v_lshlrev_b32 v13, 1, v7 // (op 1678) 

s_add_u32 s6, s15, s9 // (op 1678) 

v_and_b32 v7, -8, v1 // (op 1678) 
s_mul_hi_u32 s15, s6, s13 // (op 1678) 
v_sub_u32 v28, v1, v7 // (op 1678) 

s_sub_u32 s7, s6, s15 // (op 1678) 
s_lshr_b32 s8, s7, 1 // (op 1678) 

s_add_u32 s7, s8, s15 // (op 1678) 


s_lshr_b32 s8, s7, s11 // (op 1678) 

s_and_b32 s7, s11, 2147483648 // (op 1678) 
s_cmp_eq_i32 s7, 0 // (op 1678) 
s_mov_b32 s15, scc // (op 1678) 

s_cmp_eq_i32 s15, 0 // (op 1678) 
s_cselect_b32 s7, s6, s8 // (op 1678) 


s_mul_hi_u32 s8, s7, s12 // (op 1678) 
s_sub_u32 s15, s7, s8 // (op 1678) 
s_lshr_b32 s18, s15, 1 // (op 1678) 

s_add_u32 s15, s18, s8 // (op 1678) 


s_lshr_b32 s8, s15, s10 // (op 1678) 

s_and_b32 s15, s10, 2147483648 // (op 1678) 
s_cmp_eq_i32 s15, 0 // (op 1678) 
s_mov_b32 s18, scc // (op 1678) 

s_cmp_eq_i32 s18, 0 // (op 1678) 
s_cselect_b32 s15, s7, s8 // (op 1678) 


v_lshl_add_u32 v7, s15, 4, v13 // (op 1678) 


s_ashr_i64 s[18:19], s[2:3], 8 // (op 1678) 
s_mul_i32 s8, s7, s18 // (op 1678) 


s_sub_u32 s7, s6, s8 // (op 1678) 


v_lshl_add_u32 v13, s7, 3, v28 // (op 1678) 


v_mul_lo_u32 v28, v7, s34 // (op 1678) 
v_add_u32 v28, v28, v13 // (op 1678) 


s_mul_i32 s19, s34, s29 // (op 1680) most significant: low of low * high
s_mul_i32 s15, s35, s28 // (op 1680) most significant: low of high * low
s_mul_i32 s18, s34, s28 // (op 1680) least significant
s_mul_hi_u32 s6, s34, s28 // (op 1680) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1680) most significant: sum
s_add_u32 s19, s19, s6 // (op 1680) most significant: sum


s_mov_b32 s68, s36 // (op 1680) reg expression
s_mov_b32 s69, s37 // (op 1680) reg expression
s_mov_b32 s70, s18 // (op 1680) reg expression
s_mov_b32 s71, 131072 // (op 1680) call()

v_and_b32 v7, -16, v1 // (op 1787) 
s_mul_i32 s15, s16, s52 // (op 1787) 
s_add_u32 s15, s15, s0 // (op 1787) 
v_sub_u32 v13, v1, v7 // (op 1787) 

s_add_u32 s6, s15, s9 // (op 1787) 

v_lshrrev_b32 v7, 4, v1 // (op 1787) 
s_mul_hi_u32 s15, s6, s13 // (op 1787) 
v_lshlrev_b32 v29, 1, v7 // (op 1787) 

s_sub_u32 s7, s6, s15 // (op 1787) 
s_lshr_b32 s8, s7, 1 // (op 1787) 

s_add_u32 s7, s8, s15 // (op 1787) 


s_lshr_b32 s8, s7, s11 // (op 1787) 

s_and_b32 s7, s11, 2147483648 // (op 1787) 
s_cmp_eq_i32 s7, 0 // (op 1787) 
s_mov_b32 s15, scc // (op 1787) 

s_cmp_eq_i32 s15, 0 // (op 1787) 
s_cselect_b32 s7, s6, s8 // (op 1787) 


s_mul_hi_u32 s8, s7, s12 // (op 1787) 
s_sub_u32 s15, s7, s8 // (op 1787) 
s_lshr_b32 s18, s15, 1 // (op 1787) 

s_add_u32 s15, s18, s8 // (op 1787) 


s_lshr_b32 s8, s15, s10 // (op 1787) 

s_and_b32 s15, s10, 2147483648 // (op 1787) 
s_cmp_eq_i32 s15, 0 // (op 1787) 
s_mov_b32 s18, scc // (op 1787) 

s_cmp_eq_i32 s18, 0 // (op 1787) 
s_cselect_b32 s15, s7, s8 // (op 1787) 


s_mul_i32 s8, s15, s14 // (op 1787) 

s_sub_u32 s15, s7, s8 // (op 1787) 

v_lshl_add_u32 v7, s15, 4, v13 // (op 1787) 


s_ashr_i64 s[18:19], s[2:3], 8 // (op 1787) 
s_mul_i32 s8, s7, s18 // (op 1787) 


s_sub_u32 s7, s6, s8 // (op 1787) 


v_lshl_add_u32 v13, s7, 3, v29 // (op 1787) 


v_mul_lo_u32 v29, v13, s38 // (op 1787) 
v_add_u32 v29, v29, v7 // (op 1787) 


s_mul_i32 s19, s38, s43 // (op 1789) most significant: low of low * high
s_mul_i32 s15, s39, s42 // (op 1789) most significant: low of high * low
s_mul_i32 s18, s38, s42 // (op 1789) least significant
s_mul_hi_u32 s6, s38, s42 // (op 1789) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1789) most significant: sum
s_add_u32 s19, s19, s6 // (op 1789) most significant: sum


s_mov_b32 s72, s40 // (op 1789) reg expression
s_mov_b32 s73, s41 // (op 1789) reg expression
s_mov_b32 s74, s18 // (op 1789) reg expression
s_mov_b32 s75, 131072 // (op 1789) call()

v_and_b32 v7, -16, v1 // (op 1806) 
s_mul_i32 s15, s16, s52 // (op 1806) 
s_add_u32 s15, s15, s0 // (op 1806) 
v_sub_u32 v13, v1, v7 // (op 1806) 

s_add_u32 s6, s15, s9 // (op 1806) 

v_lshrrev_b32 v7, 4, v1 // (op 1806) 
s_mul_hi_u32 s15, s6, s13 // (op 1806) 
v_lshlrev_b32 v30, 1, v7 // (op 1806) 

s_sub_u32 s7, s6, s15 // (op 1806) 
s_lshr_b32 s8, s7, 1 // (op 1806) 

s_add_u32 s7, s8, s15 // (op 1806) 


s_lshr_b32 s8, s7, s11 // (op 1806) 

s_and_b32 s7, s11, 2147483648 // (op 1806) 
s_cmp_eq_i32 s7, 0 // (op 1806) 
s_mov_b32 s15, scc // (op 1806) 

s_cmp_eq_i32 s15, 0 // (op 1806) 
s_cselect_b32 s7, s6, s8 // (op 1806) 


s_mul_hi_u32 s8, s7, s12 // (op 1806) 
s_sub_u32 s15, s7, s8 // (op 1806) 
s_lshr_b32 s18, s15, 1 // (op 1806) 

s_add_u32 s15, s18, s8 // (op 1806) 


s_lshr_b32 s8, s15, s10 // (op 1806) 

s_and_b32 s15, s10, 2147483648 // (op 1806) 
s_cmp_eq_i32 s15, 0 // (op 1806) 
s_mov_b32 s18, scc // (op 1806) 

s_cmp_eq_i32 s18, 0 // (op 1806) 
s_cselect_b32 s15, s7, s8 // (op 1806) 


s_mul_i32 s8, s15, s14 // (op 1806) 

s_sub_u32 s15, s7, s8 // (op 1806) 

v_lshl_add_u32 v7, s15, 4, v13 // (op 1806) 


s_ashr_i64 s[18:19], s[2:3], 8 // (op 1806) 
s_mul_i32 s8, s7, s18 // (op 1806) 


s_sub_u32 s7, s6, s8 // (op 1806) 


v_lshl_add_u32 v13, s7, 3, v30 // (op 1806) 


v_mul_lo_u32 v30, v13, s38 // (op 1806) 
v_add_u32 v30, v30, v7 // (op 1806) 


s_mul_i32 s19, s38, s43 // (op 1808) most significant: low of low * high
s_mul_i32 s15, s39, s42 // (op 1808) most significant: low of high * low
s_mul_i32 s18, s38, s42 // (op 1808) least significant
s_mul_hi_u32 s6, s38, s42 // (op 1808) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1808) most significant: sum
s_add_u32 s19, s19, s6 // (op 1808) most significant: sum


s_mov_b32 s72, s40 // (op 1808) reg expression
s_mov_b32 s73, s41 // (op 1808) reg expression
s_mov_b32 s74, s18 // (op 1808) reg expression
s_mov_b32 s75, 131072 // (op 1808) call()

v_lshlrev_b32 v7, 4, v1 // (op 2285) 
s_nop 0
v_readfirstlane_b32 s15, v7 // (op 2285) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 2945) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 1283) 
v_mov_b64 v[32:33], s[18:19] // (op 1283) convert


v_mov_b64 v[34:35], v[20:21] // (op 1283) reg expression
s_mov_b32 m0, s15 // (op 1283) reg expression
buffer_load_dwordx4 v34, s[64:67], 0 offen offset: 0 lds // (op 1283) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v32 // (op 1283) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 1283) most significant half

s_add_u32 m0, m0, 1024 // (op 1283) 
buffer_load_dwordx4 v34, s[64:67], 0 offen offset: 0 lds // (op 1283) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v32 // (op 1283) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 1283) most significant half

s_add_u32 m0, m0, 1024 // (op 1283) 
buffer_load_dwordx4 v34, s[64:67], 0 offen offset: 0 lds // (op 1283) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v32 // (op 1283) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 1283) most significant half

s_add_u32 m0, m0, 1024 // (op 1283) 
buffer_load_dwordx4 v34, s[64:67], 0 offen offset: 0 lds // (op 1283) Load value direct to lds




v_mov_b64 v[32:33], s[38:39] // (op 1090) convert

v_mov_b32 v7, v29 // (op 1090) reg expression
buffer_load_ubyte v34, v7, s[72:75], 0 offen offset: 0 // (op 1090) Load value
v_add_u32 v7, v7, v32 // (op 1090) 
buffer_load_ubyte v35, v7, s[72:75], 0 offen offset: 0 // (op 1090) Load value


v_and_b32 v7, -16, v1 // (op 2220) 
v_sub_u32 v13, v1, v7 // (op 2220) 

v_lshlrev_b32 v7, 4, v13 // (op 2220) 

v_lshrrev_b32 v13, 4, v1 // (op 2220) 
v_lshl_add_u32 v31, v13, 8, v7 // (op 2220) 


s_nop 0
v_readfirstlane_b32 s15, v31 // (op 2220) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 2926) 

s_lshl_b64 s[18:19], s[24:25], 2 // (op 1288) 
v_mov_b64 v[32:33], s[18:19] // (op 1288) convert


v_mov_b64 v[36:37], v[22:23] // (op 1288) 
s_add_u32 s6, 4096, s15 // (op 1288) 
s_mov_b32 m0, s6 // (op 1288) reg expression
buffer_load_dwordx4 v36, s[56:59], 0 offen offset: 0 lds // (op 1288) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v32 // (op 1288) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 1288) most significant half

s_add_u32 m0, m0, 1024 // (op 1288) 
buffer_load_dwordx4 v36, s[56:59], 0 offen offset: 0 lds // (op 1288) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v32 // (op 1288) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 1288) most significant half

s_add_u32 m0, m0, 1024 // (op 1288) 
buffer_load_dwordx4 v36, s[56:59], 0 offen offset: 0 lds // (op 1288) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v32 // (op 1288) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 1288) most significant half

s_add_u32 m0, m0, 1024 // (op 1288) 
buffer_load_dwordx4 v36, s[56:59], 0 offen offset: 0 lds // (op 1288) Load value direct to lds





v_mov_b64 v[32:33], s[34:35] // (op 1096) convert

v_mov_b32 v7, v19 // (op 1096) 
buffer_load_ubyte v36, v7, s[68:71], 0 offen offset: 0 // (op 1096) Load value
v_add_u32 v7, v7, v32 // (op 1096) 
buffer_load_ubyte v37, v7, s[68:71], 0 offen offset: 0 // (op 1096) Load value


v_lshlrev_b32 v7, 4, v1 // (op 2542) 
s_nop 0
v_readfirstlane_b32 s15, v7 // (op 2542) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 3015) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 1293) 
v_mov_b64 v[32:33], s[18:19] // (op 1293) convert


s_lshl_b64 s[18:19], s[30:31], 8 // (op 1293) 
v_mov_b32 v7, s18 // (op 1293) 
v_mov_b32 v13, s19 // (op 1293) 
v_add_co_u32 v38, s[54:55], v20, v7 // (op 1293) least significant half
v_addc_co_u32 v39, s[54:55], v21, v13, s[54:55] // (op 1293) most significant half




s_add_u32 s6, 8192, s15 // (op 1293) 
s_mov_b32 m0, s6 // (op 1293) reg expression
buffer_load_dwordx4 v38, s[64:67], 0 offen offset: 0 lds // (op 1293) Load value direct to lds
v_add_co_u32 v38, s[18:19], v38, v32 // (op 1293) least significant half
v_addc_co_u32 v39, s[18:19], 0, v39, s[18:19] // (op 1293) most significant half

s_add_u32 m0, m0, 1024 // (op 1293) 
buffer_load_dwordx4 v38, s[64:67], 0 offen offset: 0 lds // (op 1293) Load value direct to lds
v_add_co_u32 v38, s[18:19], v38, v32 // (op 1293) least significant half
v_addc_co_u32 v39, s[18:19], 0, v39, s[18:19] // (op 1293) most significant half

s_add_u32 m0, m0, 1024 // (op 1293) 
buffer_load_dwordx4 v38, s[64:67], 0 offen offset: 0 lds // (op 1293) Load value direct to lds
v_add_co_u32 v38, s[18:19], v38, v32 // (op 1293) least significant half
v_addc_co_u32 v39, s[18:19], 0, v39, s[18:19] // (op 1293) most significant half

s_add_u32 m0, m0, 1024 // (op 1293) 
buffer_load_dwordx4 v38, s[64:67], 0 offen offset: 0 lds // (op 1293) Load value direct to lds






v_mov_b64 v[20:21], s[38:39] // (op 1102) convert

s_lshl_b32 s15, s38, 3 // (op 1102) 
v_add_u32 v7, v29, s15 // (op 1102) 

buffer_load_ubyte v32, v7, s[72:75], 0 offen offset: 0 // (op 1102) Load value
v_add_u32 v7, v7, v20 // (op 1102) 
buffer_load_ubyte v33, v7, s[72:75], 0 offen offset: 0 // (op 1102) Load value



v_and_b32 v7, -16, v1 // (op 2503) 
v_sub_u32 v13, v1, v7 // (op 2503) 

v_lshlrev_b32 v7, 4, v13 // (op 2503) 

v_lshrrev_b32 v13, 4, v1 // (op 2503) 
v_lshl_add_u32 v29, v13, 8, v7 // (op 2503) 


s_nop 0
v_readfirstlane_b32 s15, v29 // (op 2503) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 3005) 

s_lshl_b64 s[18:19], s[24:25], 2 // (op 1295) 
v_mov_b64 v[20:21], s[18:19] // (op 1295) convert


v_mov_b64 v[38:39], v[22:23] // (op 1295) 
s_add_u32 s6, 12288, s15 // (op 1295) 
s_mov_b32 m0, s6 // (op 1295) reg expression
s_mov_b32 s7, 256 // (op 1295) call()
buffer_load_dwordx4 v38, s[56:59], s7 offen offset: 0 lds // (op 1295) Load value direct to lds
v_add_co_u32 v38, s[18:19], v38, v20 // (op 1295) least significant half
v_addc_co_u32 v39, s[18:19], 0, v39, s[18:19] // (op 1295) most significant half

s_add_u32 m0, m0, 1024 // (op 1295) 
buffer_load_dwordx4 v38, s[56:59], s7 offen offset: 0 lds // (op 1295) Load value direct to lds
v_add_co_u32 v38, s[18:19], v38, v20 // (op 1295) least significant half
v_addc_co_u32 v39, s[18:19], 0, v39, s[18:19] // (op 1295) most significant half

s_add_u32 m0, m0, 1024 // (op 1295) 
buffer_load_dwordx4 v38, s[56:59], s7 offen offset: 0 lds // (op 1295) Load value direct to lds
v_add_co_u32 v38, s[18:19], v38, v20 // (op 1295) least significant half
v_addc_co_u32 v39, s[18:19], 0, v39, s[18:19] // (op 1295) most significant half

s_add_u32 m0, m0, 1024 // (op 1295) 
buffer_load_dwordx4 v38, s[56:59], s7 offen offset: 0 lds // (op 1295) Load value direct to lds







v_mov_b64 v[20:21], s[34:35] // (op 1108) convert

v_mov_b32 v7, v19 // (op 1108) 
buffer_load_ubyte v22, v7, s[68:71], 0 offen offset:8 // (op 1108) Load value
v_add_u32 v7, v7, v20 // (op 1108) 
buffer_load_ubyte v23, v7, s[68:71], 0 offen offset:8 // (op 1108) Load value



v_and_b32 v7, -16, v1 // (op 2417) 
v_sub_u32 v13, v1, v7 // (op 2417) 

v_lshrrev_b32 v7, 4, v1 // (op 2417) 
v_lshl_add_u32 v29, v7, 5, v13 // (op 2417) 


s_barrier  // (op 2985) 



s_waitcnt vmcnt(19)
ds_write_b8 v29, v34 offset:16384 // (op 1111) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v29, v35 offset:16400 // (op 1111) Store local data 

v_and_b32 v7, -8, v1 // (op 2351) 
v_sub_u32 v13, v1, v7 // (op 2351) 

v_lshrrev_b32 v7, 3, v1 // (op 2351) 
v_lshl_add_u32 v29, v7, 4, v13 // (op 2351) 


s_barrier  // (op 2965) 



s_waitcnt vmcnt(13)
ds_write_b8 v29, v36 offset:16512 // (op 1113) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v29, v37 offset:16520 // (op 1113) Store local data 

s_barrier  // (op 2942) 



ds_read_b64_tr_b8 v[38:39], v0 // (op 1133) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v0 offset:128 // (op 1133) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v0 offset:1024 // (op 1133) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v0 offset:1152 // (op 1133) Transpose load local data 



ds_read_b64_tr_b8 v[46:47], v0 offset:2048 // (op 1136) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v0 offset:2176 // (op 1136) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v0 offset:3072 // (op 1136) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v0 offset:3200 // (op 1136) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 2982) 


ds_read_u8 v7, v14 offset:16384 // (op 1145) Load local data 


ds_read_u8 v13, v14 offset:16448 // (op 1148) Load local data 
s_barrier  // (op 2923) 



ds_read_b128 v[54:57], v4 offset:4096 // (op 1127) Load local data 
ds_read_b128 v[58:61], v4 offset:4160 // (op 1127) Load local data 



ds_read_b128 v[62:65], v4 offset:4224 // (op 1130) Load local data 
ds_read_b128 v[66:69], v4 offset:4288 // (op 1130) Load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 2962) 



ds_read_u8 v29, v3 offset:16512 // (op 1139) Load local data 



ds_read_u8 v31, v3 offset:16516 // (op 1142) Load local data 
v_and_b32 v19, -16, v1 // (op 2192) 
v_sub_u32 v20, v1, v19 // (op 2192) 

v_lshlrev_b32 v19, 4, v20 // (op 2192) 

v_lshrrev_b32 v20, 4, v1 // (op 2192) 
v_lshl_add_u32 v21, v20, 8, v19 // (op 2192) 


s_nop 0
v_readfirstlane_b32 s15, v21 // (op 2192) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v19, 4, v1 // (op 2257) 
s_nop 0
v_readfirstlane_b32 s6, v19 // (op 2257) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v19, -8, v1 // (op 2322) 
v_sub_u32 v20, v1, v19 // (op 2322) 

v_lshrrev_b32 v19, 3, v1 // (op 2322) 
v_lshl_add_u32 v21, v19, 4, v20 // (op 2322) 


v_and_b32 v19, -16, v1 // (op 2388) 
v_sub_u32 v20, v1, v19 // (op 2388) 

v_lshrrev_b32 v19, 4, v1 // (op 2388) 
v_lshl_add_u32 v70, v19, 5, v20 // (op 2388) 


v_and_b32 v19, -16, v1 // (op 2488) 
v_sub_u32 v20, v1, v19 // (op 2488) 

v_lshlrev_b32 v19, 4, v20 // (op 2488) 

v_lshrrev_b32 v20, 4, v1 // (op 2488) 
v_lshl_add_u32 v71, v20, 8, v19 // (op 2488) 


s_nop 0
v_readfirstlane_b32 s7, v71 // (op 2488) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v19, 4, v1 // (op 2527) 
s_nop 0
v_readfirstlane_b32 s8, v19 // (op 2527) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v19, -8, v1 // (op 2566) 
v_sub_u32 v20, v1, v19 // (op 2566) 

v_lshrrev_b32 v19, 3, v1 // (op 2566) 
v_lshl_add_u32 v71, v19, 4, v20 // (op 2566) 


v_and_b32 v19, -16, v1 // (op 2590) 
v_sub_u32 v20, v1, v19 // (op 2590) 

v_lshrrev_b32 v19, 4, v1 // (op 2590) 
v_lshl_add_u32 v72, v19, 5, v20 // (op 2590) 


s_mov_b32 s1, 0 // (op 204) call()
s_ashr_i64 s[18:19], s[2:3], 8 // (op 203) 
s_mov_b32 s44, s18 // (op 203) convert

s_and_b32 s18, s44, 4294967294 // (op 203) 

s_cmp_lt_u32 s1, s18 // (op 203) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopBottom_KLoop_203 // (op 203) Condition: Top (jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopBottom_KLoop_203 if false)
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopTop_KLoop_203:
s_add_u32 s1, s1, 2 // (op 205) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[54:61], v[38:45], a[0:3], v29, v7 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 216) 
s_barrier  // (op 2948) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 1218) 
v_mov_b64 v[74:75], s[18:19] // (op 1218) convert


v_mov_b64 v[76:77], v[26:27] // (op 1218) 
s_lshl_b64 s[18:19], s[30:31], 8 // (op 1218) 
s_lshl_b64 s[54:55], s[18:19], 1 // (op 1218) 

v_mov_b32 v73, s54 // (op 1218) 
v_mov_b32 v19, s55 // (op 1218) 
v_add_co_u32 v78, s[18:19], v76, v73 // (op 1218) least significant half
v_addc_co_u32 v79, s[18:19], v77, v19, s[18:19] // (op 1218) most significant half




s_mov_b32 m0, s6 // (op 1218) reg expression
buffer_load_dwordx4 v78, s[64:67], 0 offen offset: 0 lds // (op 1218) Load value direct to lds
v_add_co_u32 v78, s[18:19], v78, v74 // (op 1218) least significant half
v_addc_co_u32 v79, s[18:19], 0, v79, s[18:19] // (op 1218) most significant half

s_add_u32 m0, m0, 1024 // (op 1218) 
buffer_load_dwordx4 v78, s[64:67], 0 offen offset: 0 lds // (op 1218) Load value direct to lds
v_add_co_u32 v78, s[18:19], v78, v74 // (op 1218) least significant half
v_addc_co_u32 v79, s[18:19], 0, v79, s[18:19] // (op 1218) most significant half

s_add_u32 m0, m0, 1024 // (op 1218) 
buffer_load_dwordx4 v78, s[64:67], 0 offen offset: 0 lds // (op 1218) Load value direct to lds
v_add_co_u32 v78, s[18:19], v78, v74 // (op 1218) least significant half
v_addc_co_u32 v79, s[18:19], 0, v79, s[18:19] // (op 1218) most significant half

s_add_u32 m0, m0, 1024 // (op 1218) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:69], v[46:53], a[0:3], v31, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 225) 
buffer_load_dwordx4 v78, s[64:67], 0 offen offset: 0 lds // (op 1218) Load value direct to lds



v_mov_b64 v[74:75], s[38:39] // (op 231) convert

v_mov_b32 v73, v30 // (op 231) 
s_lshl_b32 s18, s38, 4 // (op 231) 
v_add_u32 v19, v73, s18 // (op 231) 

buffer_load_ubyte v34, v19, s[72:75], 0 offen offset: 0 // (op 231) Load value
v_add_u32 v19, v19, v74 // (op 231) 
buffer_load_ubyte v35, v19, s[72:75], 0 offen offset: 0 // (op 231) Load value


s_barrier  // (op 2929) 

s_lshl_b64 s[18:19], s[24:25], 2 // (op 1213) 
v_mov_b64 v[74:75], s[18:19] // (op 1213) convert


v_mov_b64 v[78:79], v[24:25] // (op 1213) 
v_mov_b64 v[80:81], v[78:79] // (op 1213) 
s_add_u32 s18, 4096, s15 // (op 1213) 
s_mov_b32 m0, s18 // (op 1213) reg expression
s_mov_b32 s19, 512 // (op 1213) call()
buffer_load_dwordx4 v80, s[56:59], s19 offen offset: 0 lds // (op 1213) Load value direct to lds
v_add_co_u32 v80, s[54:55], v80, v74 // (op 1213) least significant half
v_addc_co_u32 v81, s[54:55], 0, v81, s[54:55] // (op 1213) most significant half

s_add_u32 m0, m0, 1024 // (op 1213) 
buffer_load_dwordx4 v80, s[56:59], s19 offen offset: 0 lds // (op 1213) Load value direct to lds
v_add_co_u32 v80, s[54:55], v80, v74 // (op 1213) least significant half
v_addc_co_u32 v81, s[54:55], 0, v81, s[54:55] // (op 1213) most significant half

s_add_u32 m0, m0, 1024 // (op 1213) 
buffer_load_dwordx4 v80, s[56:59], s19 offen offset: 0 lds // (op 1213) Load value direct to lds
v_add_co_u32 v80, s[54:55], v80, v74 // (op 1213) least significant half
v_addc_co_u32 v81, s[54:55], 0, v81, s[54:55] // (op 1213) most significant half

s_add_u32 m0, m0, 1024 // (op 1213) 
buffer_load_dwordx4 v80, s[56:59], s19 offen offset: 0 lds // (op 1213) Load value direct to lds





v_mov_b64 v[74:75], s[34:35] // (op 230) convert

v_mov_b32 v19, v28 // (op 230) 
v_mov_b32 v20, v19 // (op 230) 
buffer_load_ubyte v36, v20, s[68:71], 0 offen offset:16 // (op 230) Load value
v_add_u32 v20, v20, v74 // (op 230) 
buffer_load_ubyte v37, v20, s[68:71], 0 offen offset:16 // (op 230) Load value


s_barrier  // (op 3030) 



s_waitcnt vmcnt(19)
ds_write_b8 v72, v32 offset:16640 // (op 600) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v72, v33 offset:16656 // (op 600) Store local data 
s_barrier  // (op 3024) 



s_waitcnt vmcnt(13)
ds_write_b8 v71, v22 offset:16768 // (op 599) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v71, v23 offset:16776 // (op 599) Store local data 
s_barrier  // (op 3011) 



ds_read_b64_tr_b8 v[38:39], v9 offset:8192 // (op 583) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v9 offset:8320 // (op 583) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v9 offset:9216 // (op 583) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v9 offset:9344 // (op 583) Transpose load local data 



ds_read_b64_tr_b8 v[80:81], v9 offset:10240 // (op 592) Transpose load local data 
ds_read_b64_tr_b8 v[82:83], v9 offset:10368 // (op 592) Transpose load local data 
ds_read_b64_tr_b8 v[84:85], v9 offset:11264 // (op 592) Transpose load local data 
ds_read_b64_tr_b8 v[86:87], v9 offset:11392 // (op 592) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 3027) 


ds_read_u8 v7, v15 offset:16640 // (op 587) Load local data 


ds_read_u8 v20, v15 offset:16704 // (op 596) Load local data 
s_barrier  // (op 3002) 
s_barrier  // (op 3162) 



ds_read_b128 v[54:57], v10 offset:12288 // (op 581) Load local data 
ds_read_b128 v[58:61], v10 offset:12352 // (op 581) Load local data 



ds_read_b128 v[62:65], v10 offset:12416 // (op 590) Load local data 
ds_read_b128 v[66:69], v10 offset:12480 // (op 590) Load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 3021) 



ds_read_u8 v29, v12 offset:16768 // (op 585) Load local data 



ds_read_u8 v31, v12 offset:16772 // (op 594) Load local data 
s_barrier  // (op 3158) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[54:61], v[38:45], a[0:3], v29, v7 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 588) 
s_barrier  // (op 3018) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 1241) 
v_mov_b64 v[74:75], s[18:19] // (op 1241) convert


s_lshl_b64 s[18:19], s[30:31], 8 // (op 1241) 
s_mul_i32 s55, 3, s19 // (op 1241) most significant: low of low * high
s_mul_i32 s54, 3, s18 // (op 1241) least significant
s_mul_hi_u32 s44, 3, s18 // (op 1241) most significant: high of low * low
s_add_u32 s55, s55, s44 // (op 1241) most significant: sum


v_mov_b32 v88, s54 // (op 1241) 
v_mov_b32 v89, s55 // (op 1241) 
v_add_co_u32 v90, s[18:19], v76, v88 // (op 1241) least significant half
v_addc_co_u32 v91, s[18:19], v77, v89, s[18:19] // (op 1241) most significant half




s_add_u32 s18, 8192, s8 // (op 1241) 
s_nop 0
s_mov_b32 m0, s18 // (op 1241) reg expression
buffer_load_dwordx4 v90, s[64:67], 0 offen offset: 0 lds // (op 1241) Load value direct to lds
v_add_co_u32 v90, s[54:55], v90, v74 // (op 1241) least significant half
v_addc_co_u32 v91, s[54:55], 0, v91, s[54:55] // (op 1241) most significant half

s_add_u32 m0, m0, 1024 // (op 1241) 
buffer_load_dwordx4 v90, s[64:67], 0 offen offset: 0 lds // (op 1241) Load value direct to lds
v_add_co_u32 v90, s[54:55], v90, v74 // (op 1241) least significant half
v_addc_co_u32 v91, s[54:55], 0, v91, s[54:55] // (op 1241) most significant half

s_add_u32 m0, m0, 1024 // (op 1241) 
buffer_load_dwordx4 v90, s[64:67], 0 offen offset: 0 lds // (op 1241) Load value direct to lds
v_add_co_u32 v90, s[54:55], v90, v74 // (op 1241) least significant half
v_addc_co_u32 v91, s[54:55], 0, v91, s[54:55] // (op 1241) most significant half

s_add_u32 m0, m0, 1024 // (op 1241) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:69], v[80:87], a[0:3], v31, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 597) 
buffer_load_dwordx4 v90, s[64:67], 0 offen offset: 0 lds // (op 1241) Load value direct to lds







v_mov_b64 v[74:75], s[38:39] // (op 603) convert

s_lshl_b32 s18, s38, 3 // (op 603) 
s_mul_i32 s19, 3, s18 // (op 603) 

v_add_u32 v20, v73, s19 // (op 603) 

buffer_load_ubyte v32, v20, s[72:75], 0 offen offset: 0 // (op 603) Load value
v_add_u32 v20, v20, v74 // (op 603) 
s_lshl_b64 s[18:19], s[30:31], 8 // (op 1542) 
buffer_load_ubyte v33, v20, s[72:75], 0 offen offset: 0 // (op 603) Load value


s_lshl_b64 s[54:55], s[18:19], 1 // (op 1542) 


v_mov_b32 v20, s54 // (op 1542) 
s_barrier  // (op 3008) 
s_lshl_b32 s18, s38, 4 // (op 1798) 

v_add_u32 v30, v30, s18 // (op 1798) 

v_mov_b32 v73, s55 // (op 1542) 
s_lshl_b64 s[18:19], s[24:25], 2 // (op 1236) 
v_mov_b64 v[74:75], s[18:19] // (op 1236) convert


v_mov_b64 v[76:77], v[78:79] // (op 1236) 
s_add_u32 s18, 12288, s7 // (op 1236) 
s_mov_b32 m0, s18 // (op 1236) reg expression
s_mov_b32 s19, 768 // (op 1236) call()
v_add_co_u32 v26, s[44:45], v26, v20 // (op 1542) least significant half
v_addc_co_u32 v27, s[44:45], v27, v73, s[44:45] // (op 1542) most significant half




buffer_load_dwordx4 v76, s[56:59], s19 offen offset: 0 lds // (op 1236) Load value direct to lds
v_add_co_u32 v76, s[54:55], v76, v74 // (op 1236) least significant half
v_addc_co_u32 v77, s[54:55], 0, v77, s[54:55] // (op 1236) most significant half

s_add_u32 m0, m0, 1024 // (op 1236) 
buffer_load_dwordx4 v76, s[56:59], s19 offen offset: 0 lds // (op 1236) Load value direct to lds
v_add_co_u32 v76, s[54:55], v76, v74 // (op 1236) least significant half
v_addc_co_u32 v77, s[54:55], 0, v77, s[54:55] // (op 1236) most significant half

s_add_u32 m0, m0, 1024 // (op 1236) 
buffer_load_dwordx4 v76, s[56:59], s19 offen offset: 0 lds // (op 1236) Load value direct to lds
v_add_co_u32 v76, s[54:55], v76, v74 // (op 1236) least significant half
v_addc_co_u32 v77, s[54:55], 0, v77, s[54:55] // (op 1236) most significant half

s_add_u32 m0, m0, 1024 // (op 1236) 
buffer_load_dwordx4 v76, s[56:59], s19 offen offset: 0 lds // (op 1236) Load value direct to lds






v_add_co_u32 v24, vcc, 512, v24 // (op 1414) least significant half
v_addc_co_u32 v25, vcc, 0, v25, vcc // (op 1414) most significant half
v_mov_b64 v[74:75], s[34:35] // (op 602) convert

v_mov_b32 v20, v19 // (op 602) 
buffer_load_ubyte v22, v20, s[68:71], 0 offen offset:24 // (op 602) Load value
v_add_u32 v20, v20, v74 // (op 602) 
buffer_load_ubyte v23, v20, s[68:71], 0 offen offset:24 // (op 602) Load value



s_barrier  // (op 2988) 
v_add_u32 v28, 16, v28 // (op 1670) 



s_waitcnt vmcnt(19)
ds_write_b8 v70, v34 offset:16384 // (op 228) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v70, v35 offset:16400 // (op 228) Store local data 
s_barrier  // (op 2968) 



s_waitcnt vmcnt(13)
ds_write_b8 v21, v36 offset:16512 // (op 227) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v21, v37 offset:16520 // (op 227) Store local data 
s_barrier  // (op 2954) 



ds_read_b64_tr_b8 v[38:39], v5 // (op 211) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v5 offset:128 // (op 211) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v5 offset:1024 // (op 211) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v5 offset:1152 // (op 211) Transpose load local data 



ds_read_b64_tr_b8 v[46:47], v5 offset:2048 // (op 220) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v5 offset:2176 // (op 220) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v5 offset:3072 // (op 220) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v5 offset:3200 // (op 220) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 2995) 


ds_read_u8 v7, v11 offset:16384 // (op 215) Load local data 


ds_read_u8 v13, v11 offset:16448 // (op 224) Load local data 
s_barrier  // (op 2935) 



ds_read_b128 v[54:57], v6 offset:4096 // (op 209) Load local data 
ds_read_b128 v[58:61], v6 offset:4160 // (op 209) Load local data 



ds_read_b128 v[62:65], v6 offset:4224 // (op 218) Load local data 
ds_read_b128 v[66:69], v6 offset:4288 // (op 218) Load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 2975) 



ds_read_u8 v29, v8 offset:16512 // (op 213) Load local data 



ds_read_u8 v31, v8 offset:16516 // (op 222) Load local data 
s_ashr_i64 s[18:19], s[2:3], 8 // (op 203) 
s_mov_b32 s44, s18 // (op 203) convert

s_and_b32 s18, s44, 4294967294 // (op 203) 

s_cmp_lt_u32 s1, s18 // (op 203) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopTop_KLoop_203 // (op 203) Condition: Bottom (jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopTop_KLoop_203 if true)
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopBottom_KLoop_203:














s_barrier  // (op 3084) 
v_lshrrev_b32 v30, 4, v1 // (op 1308) 
s_ashr_i64 s[18:19], s[2:3], 8 // (op 1308) 
v_and_b32 v19, -16, v1 // (op 1308) 
s_and_b32 s15, s18, 4294967294 // (op 1308) 
v_sub_u32 v20, v1, v19 // (op 1308) 

s_add_u32 s6, s0, s15 // (op 1308) 

v_lshlrev_b32 v19, 4, v20 // (op 1308) 

s_mul_i32 s15, s16, s52 // (op 1308) 
s_add_u32 s15, s15, s6 // (op 1308) 

s_add_u32 s6, s15, s9 // (op 1308) 

s_mul_hi_u32 s15, s6, s13 // (op 1308) 
s_sub_u32 s7, s6, s15 // (op 1308) 
s_lshr_b32 s8, s7, 1 // (op 1308) 

s_add_u32 s7, s8, s15 // (op 1308) 


s_lshr_b32 s8, s7, s11 // (op 1308) 

s_and_b32 s7, s11, 2147483648 // (op 1308) 
s_cmp_eq_i32 s7, 0 // (op 1308) 
s_mov_b32 s15, scc // (op 1308) 

s_cmp_eq_i32 s15, 0 // (op 1308) 
s_cselect_b32 s7, s6, s8 // (op 1308) 


s_mul_hi_u32 s8, s7, s12 // (op 1308) 
s_sub_u32 s15, s7, s8 // (op 1308) 
s_lshr_b32 s44, s15, 1 // (op 1308) 

s_add_u32 s15, s44, s8 // (op 1308) 


s_lshr_b32 s8, s15, s10 // (op 1308) 

s_and_b32 s15, s10, 2147483648 // (op 1308) 
s_cmp_eq_i32 s15, 0 // (op 1308) 
s_mov_b32 s44, scc // (op 1308) 

s_cmp_eq_i32 s44, 0 // (op 1308) 
s_cselect_b32 s15, s7, s8 // (op 1308) 


s_mul_i32 s8, s7, s18 // (op 1308) 


v_lshl_add_u32 v20, s15, 4, v30 // (op 1308) 


s_sub_u32 s7, s6, s8 // (op 1308) 


v_lshl_add_u32 v21, s7, 8, v19 // (op 1308) 


v_mov_b32 v25, 0 // (op 1308) convert
v_mov_b32 v24, v21 // (op 1308) convert

v_mul_lo_u32 v27, v20, s25 // (op 1308) most significant: low of low * high
v_mul_hi_u32 v19, v20, s24 // (op 1308) most significant: high of low * low
v_mul_lo_u32 v26, v20, s24 // (op 1308) least significant: low of low * low
v_add_u32 v27, v27, v19 // (op 1308) most significant: sum

v_add_co_u32 v26, s[18:19], v26, v24 // (op 1308) least significant half
v_addc_co_u32 v27, s[18:19], v27, v25, s[18:19] // (op 1308) most significant half



s_mul_i32 s19, s24, s29 // (op 1310) most significant: low of low * high
s_mul_i32 s15, s25, s28 // (op 1310) most significant: low of high * low
s_mul_i32 s18, s24, s28 // (op 1310) least significant
s_mul_hi_u32 s6, s24, s28 // (op 1310) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1310) most significant: sum
s_add_u32 s19, s19, s6 // (op 1310) most significant: sum


s_mov_b32 s56, s26 // (op 1310) reg expression
s_mov_b32 s57, s27 // (op 1310) reg expression
s_mov_b32 s58, s18 // (op 1310) reg expression
s_mov_b32 s59, 131072 // (op 1310) call()

s_ashr_i64 s[18:19], s[2:3], 8 // (op 1444) 
s_and_b32 s15, s18, 4294967294 // (op 1444) 
s_add_u32 s6, s0, s15 // (op 1444) 

s_mul_i32 s15, s16, s52 // (op 1444) 
s_add_u32 s15, s15, s6 // (op 1444) 

s_add_u32 s6, s15, s9 // (op 1444) 

s_mul_hi_u32 s15, s6, s13 // (op 1444) 
s_sub_u32 s7, s6, s15 // (op 1444) 
s_lshr_b32 s8, s7, 1 // (op 1444) 

s_add_u32 s7, s8, s15 // (op 1444) 


s_lshr_b32 s8, s7, s11 // (op 1444) 

s_and_b32 s7, s11, 2147483648 // (op 1444) 
s_cmp_eq_i32 s7, 0 // (op 1444) 
s_mov_b32 s15, scc // (op 1444) 

s_cmp_eq_i32 s15, 0 // (op 1444) 
s_cselect_b32 s7, s6, s8 // (op 1444) 


s_mul_hi_u32 s8, s7, s12 // (op 1444) 
s_sub_u32 s15, s7, s8 // (op 1444) 
s_lshr_b32 s44, s15, 1 // (op 1444) 

s_add_u32 s15, s44, s8 // (op 1444) 


s_lshr_b32 s8, s15, s10 // (op 1444) 

s_and_b32 s15, s10, 2147483648 // (op 1444) 
s_cmp_eq_i32 s15, 0 // (op 1444) 
s_mov_b32 s44, scc // (op 1444) 

s_cmp_eq_i32 s44, 0 // (op 1444) 
s_cselect_b32 s15, s7, s8 // (op 1444) 


s_mul_i32 s8, s15, s14 // (op 1444) 

s_sub_u32 s15, s7, s8 // (op 1444) 

s_lshl_b32 s8, s15, 4 // (op 1444) 

s_mov_b32 s55, 0 // (op 1444) convert
s_mov_b32 s54, s8 // (op 1444) convert

s_mul_i32 s8, s7, s18 // (op 1444) 


s_sub_u32 s7, s6, s8 // (op 1444) 


v_lshl_add_u32 v28, s7, 8, v1 // (op 1444) 

v_mul_lo_u32 v21, v28, s31 // (op 1444) most significant: low of low * high
v_mul_hi_u32 v19, v28, s30 // (op 1444) most significant: high of low * low
v_mul_lo_u32 v20, v28, s30 // (op 1444) least significant: low of low * low
v_add_u32 v21, v21, v19 // (op 1444) most significant: sum

v_mov_b32 v19, s54 // (op 1444) 
v_mov_b32 v30, s55 // (op 1444) 
v_add_co_u32 v20, s[18:19], v20, v19 // (op 1444) least significant half
v_addc_co_u32 v21, s[18:19], v21, v30, s[18:19] // (op 1444) most significant half





s_mul_i32 s19, s30, s3 // (op 1446) most significant: low of low * high
s_mul_i32 s15, s31, s2 // (op 1446) most significant: low of high * low
s_mul_i32 s18, s30, s2 // (op 1446) least significant
s_mul_hi_u32 s6, s30, s2 // (op 1446) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1446) most significant: sum
s_add_u32 s19, s19, s6 // (op 1446) most significant: sum


s_mov_b32 s64, s32 // (op 1446) reg expression
s_mov_b32 s65, s33 // (op 1446) reg expression
s_mov_b32 s66, s18 // (op 1446) reg expression
s_mov_b32 s67, 131072 // (op 1446) call()

v_lshrrev_b32 v19, 3, v1 // (op 1572) 
s_ashr_i64 s[18:19], s[2:3], 8 // (op 1572) 
v_lshlrev_b32 v28, 1, v19 // (op 1572) 

s_and_b32 s15, s18, 4294967294 // (op 1572) 
v_and_b32 v19, -8, v1 // (op 1572) 
s_add_u32 s6, s0, s15 // (op 1572) 

v_sub_u32 v30, v1, v19 // (op 1572) 

s_mul_i32 s15, s16, s52 // (op 1572) 
s_add_u32 s15, s15, s6 // (op 1572) 

s_add_u32 s6, s15, s9 // (op 1572) 

s_mul_hi_u32 s15, s6, s13 // (op 1572) 
s_sub_u32 s7, s6, s15 // (op 1572) 
s_lshr_b32 s8, s7, 1 // (op 1572) 

s_add_u32 s7, s8, s15 // (op 1572) 


s_lshr_b32 s8, s7, s11 // (op 1572) 

s_and_b32 s7, s11, 2147483648 // (op 1572) 
s_cmp_eq_i32 s7, 0 // (op 1572) 
s_mov_b32 s15, scc // (op 1572) 

s_cmp_eq_i32 s15, 0 // (op 1572) 
s_cselect_b32 s7, s6, s8 // (op 1572) 


s_mul_hi_u32 s8, s7, s12 // (op 1572) 
s_sub_u32 s15, s7, s8 // (op 1572) 
s_lshr_b32 s44, s15, 1 // (op 1572) 

s_add_u32 s15, s44, s8 // (op 1572) 


s_lshr_b32 s8, s15, s10 // (op 1572) 

s_and_b32 s15, s10, 2147483648 // (op 1572) 
s_cmp_eq_i32 s15, 0 // (op 1572) 
s_mov_b32 s44, scc // (op 1572) 

s_cmp_eq_i32 s44, 0 // (op 1572) 
s_cselect_b32 s15, s7, s8 // (op 1572) 


s_mul_i32 s8, s7, s18 // (op 1572) 


v_lshl_add_u32 v19, s15, 4, v28 // (op 1572) 


s_sub_u32 s7, s6, s8 // (op 1572) 


v_lshl_add_u32 v28, s7, 3, v30 // (op 1572) 


v_mul_lo_u32 v30, v19, s34 // (op 1572) 
v_add_u32 v30, v30, v28 // (op 1572) 


s_mul_i32 s19, s34, s29 // (op 1574) most significant: low of low * high
s_mul_i32 s15, s35, s28 // (op 1574) most significant: low of high * low
s_mul_i32 s18, s34, s28 // (op 1574) least significant
s_mul_hi_u32 s6, s34, s28 // (op 1574) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1574) most significant: sum
s_add_u32 s19, s19, s6 // (op 1574) most significant: sum


s_mov_b32 s68, s36 // (op 1574) reg expression
s_mov_b32 s69, s37 // (op 1574) reg expression
s_mov_b32 s70, s18 // (op 1574) reg expression
s_mov_b32 s71, 131072 // (op 1574) call()

v_and_b32 v19, -16, v1 // (op 1700) 
s_ashr_i64 s[18:19], s[2:3], 8 // (op 1700) 
v_sub_u32 v28, v1, v19 // (op 1700) 

s_and_b32 s15, s18, 4294967294 // (op 1700) 
v_lshrrev_b32 v19, 4, v1 // (op 1700) 
s_add_u32 s6, s0, s15 // (op 1700) 

v_lshlrev_b32 v24, 1, v19 // (op 1700) 

s_mul_i32 s15, s16, s52 // (op 1700) 
s_add_u32 s15, s15, s6 // (op 1700) 

s_add_u32 s6, s15, s9 // (op 1700) 

s_mul_hi_u32 s15, s6, s13 // (op 1700) 
s_sub_u32 s7, s6, s15 // (op 1700) 
s_lshr_b32 s8, s7, 1 // (op 1700) 

s_add_u32 s7, s8, s15 // (op 1700) 


s_lshr_b32 s8, s7, s11 // (op 1700) 

s_and_b32 s7, s11, 2147483648 // (op 1700) 
s_cmp_eq_i32 s7, 0 // (op 1700) 
s_mov_b32 s15, scc // (op 1700) 

s_cmp_eq_i32 s15, 0 // (op 1700) 
s_cselect_b32 s7, s6, s8 // (op 1700) 


s_mul_hi_u32 s8, s7, s12 // (op 1700) 
s_sub_u32 s15, s7, s8 // (op 1700) 
s_lshr_b32 s44, s15, 1 // (op 1700) 

s_add_u32 s15, s44, s8 // (op 1700) 


s_lshr_b32 s8, s15, s10 // (op 1700) 

s_and_b32 s15, s10, 2147483648 // (op 1700) 
s_cmp_eq_i32 s15, 0 // (op 1700) 
s_mov_b32 s44, scc // (op 1700) 

s_cmp_eq_i32 s44, 0 // (op 1700) 
s_cselect_b32 s15, s7, s8 // (op 1700) 


s_mul_i32 s8, s15, s14 // (op 1700) 

s_sub_u32 s15, s7, s8 // (op 1700) 

s_mul_i32 s8, s7, s18 // (op 1700) 


v_lshl_add_u32 v19, s15, 4, v28 // (op 1700) 


s_sub_u32 s7, s6, s8 // (op 1700) 


v_lshl_add_u32 v25, s7, 3, v24 // (op 1700) 


v_mul_lo_u32 v24, v25, s38 // (op 1700) 
v_add_u32 v24, v24, v19 // (op 1700) 


s_mul_i32 s19, s38, s43 // (op 1702) most significant: low of low * high
s_mul_i32 s15, s39, s42 // (op 1702) most significant: low of high * low
s_mul_i32 s18, s38, s42 // (op 1702) least significant
s_mul_hi_u32 s6, s38, s42 // (op 1702) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1702) most significant: sum
s_add_u32 s19, s19, s6 // (op 1702) most significant: sum


s_mov_b32 s72, s40 // (op 1702) reg expression
s_mov_b32 s73, s41 // (op 1702) reg expression
s_mov_b32 s74, s18 // (op 1702) reg expression
s_mov_b32 s75, 131072 // (op 1702) call()

v_and_b32 v19, -16, v1 // (op 2205) 
v_sub_u32 v25, v1, v19 // (op 2205) 

v_lshlrev_b32 v19, 4, v25 // (op 2205) 

v_lshrrev_b32 v25, 4, v1 // (op 2205) 
v_lshl_add_u32 v28, v25, 8, v19 // (op 2205) 


s_nop 0
v_readfirstlane_b32 s15, v28 // (op 2205) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v19, 4, v1 // (op 2270) 
s_nop 0
v_readfirstlane_b32 s6, v19 // (op 2270) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v19, -8, v1 // (op 2335) 
v_sub_u32 v25, v1, v19 // (op 2335) 

v_lshrrev_b32 v19, 3, v1 // (op 2335) 
v_lshl_add_u32 v28, v19, 4, v25 // (op 2335) 


v_and_b32 v19, -16, v1 // (op 2401) 
v_sub_u32 v25, v1, v19 // (op 2401) 

v_lshrrev_b32 v19, 4, v1 // (op 2401) 
v_lshl_add_u32 v34, v19, 5, v25 // (op 2401) 


s_ashr_i64 s[18:19], s[2:3], 8 // (op 490) 
s_and_b32 s7, s18, 4294967294 // (op 490) 

s_ashr_i64 s[18:19], s[2:3], 8 // (op 489) 
s_mov_b32 s8, s18 // (op 489) convert

s_cmp_lt_u32 s7, s8 // (op 489) 











s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopBottom_KLoopTail_489 // (op 489) Condition: Top (jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopBottom_KLoopTail_489 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopTop_KLoopTail_489:
s_barrier  // (op 2932) 
s_add_u32 s7, s7, 1 // (op 491) 

s_lshl_b64 s[18:19], s[24:25], 2 // (op 1223) 
v_mov_b64 v[22:23], s[18:19] // (op 1223) convert


v_mov_b64 v[32:33], v[26:27] // (op 1223) 
v_mov_b64 v[36:37], v[32:33] // (op 1223) 
s_add_u32 s8, 4096, s15 // (op 1223) 
s_mov_b32 m0, s8 // (op 1223) reg expression
buffer_load_dwordx4 v36, s[56:59], 0 offen offset: 0 lds // (op 1223) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v22 // (op 1223) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 1223) most significant half

s_add_u32 m0, m0, 1024 // (op 1223) 
buffer_load_dwordx4 v36, s[56:59], 0 offen offset: 0 lds // (op 1223) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v22 // (op 1223) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 1223) most significant half

s_add_u32 m0, m0, 1024 // (op 1223) 
buffer_load_dwordx4 v36, s[56:59], 0 offen offset: 0 lds // (op 1223) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v22 // (op 1223) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 1223) most significant half

s_add_u32 m0, m0, 1024 // (op 1223) 
buffer_load_dwordx4 v36, s[56:59], 0 offen offset: 0 lds // (op 1223) Load value direct to lds




v_add_co_u32 v26, vcc, 256, v26 // (op 1298) least significant half
v_addc_co_u32 v27, vcc, 0, v27, vcc // (op 1298) most significant half
s_barrier  // (op 2951) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 1229) 
v_mov_b64 v[22:23], s[18:19] // (op 1229) convert


v_mov_b64 v[32:33], v[20:21] // (op 1229) 
v_mov_b64 v[36:37], v[32:33] // (op 1229) reg expression
s_mov_b32 m0, s6 // (op 1229) reg expression
buffer_load_dwordx4 v36, s[64:67], 0 offen offset: 0 lds // (op 1229) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v22 // (op 1229) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 1229) most significant half

s_add_u32 m0, m0, 1024 // (op 1229) 
buffer_load_dwordx4 v36, s[64:67], 0 offen offset: 0 lds // (op 1229) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v22 // (op 1229) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 1229) most significant half

s_add_u32 m0, m0, 1024 // (op 1229) 
buffer_load_dwordx4 v36, s[64:67], 0 offen offset: 0 lds // (op 1229) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v22 // (op 1229) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 1229) most significant half

s_add_u32 m0, m0, 1024 // (op 1229) 
buffer_load_dwordx4 v36, s[64:67], 0 offen offset: 0 lds // (op 1229) Load value direct to lds



s_lshl_b64 s[18:19], s[30:31], 8 // (op 1436) 
v_mov_b32 v7, s18 // (op 1436) 
v_mov_b32 v13, s19 // (op 1436) 
v_add_co_u32 v20, s[54:55], v20, v7 // (op 1436) least significant half
v_addc_co_u32 v21, s[54:55], v21, v13, s[54:55] // (op 1436) most significant half





v_mov_b64 v[22:23], s[34:35] // (op 518) convert

v_mov_b32 v7, v30 // (op 518) 
v_mov_b32 v13, v7 // (op 518) 
buffer_load_ubyte v32, v13, s[68:71], 0 offen offset: 0 // (op 518) Load value
v_add_u32 v13, v13, v22 // (op 518) 
buffer_load_ubyte v33, v13, s[68:71], 0 offen offset: 0 // (op 518) Load value



v_add_u32 v30, 8, v30 // (op 1564) 

s_barrier  // (op 2971) 
v_mov_b64 v[22:23], s[38:39] // (op 519) convert




v_mov_b32 v7, v24 // (op 519) 
v_mov_b32 v13, v7 // (op 519) reg expression
buffer_load_ubyte v36, v13, s[72:75], 0 offen offset: 0 // (op 519) Load value
v_add_u32 v13, v13, v22 // (op 519) 
buffer_load_ubyte v37, v13, s[72:75], 0 offen offset: 0 // (op 519) Load value



s_lshl_b32 s8, s38, 3 // (op 1692) 
v_add_u32 v24, v24, s8 // (op 1692) 

s_waitcnt vmcnt(3)
ds_write_b8 v28, v32 offset:16512 // (op 515) Store local data 
s_barrier  // (op 2938) 
s_waitcnt vmcnt(2)
ds_write_b8 v28, v33 offset:16520 // (op 515) Store local data 




ds_read_b128 v[38:41], v6 offset:4096 // (op 497) Load local data 
s_barrier  // (op 2991) 
ds_read_b128 v[42:45], v6 offset:4160 // (op 497) Load local data 



s_barrier  // (op 2958) 
s_waitcnt vmcnt(1)
ds_write_b8 v34, v36 offset:16384 // (op 516) Store local data 



ds_read_b64_tr_b8 v[46:47], v5 // (op 499) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v5 offset:128 // (op 499) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v5 offset:1024 // (op 499) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v5 offset:1152 // (op 499) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v34, v37 offset:16400 // (op 516) Store local data 

s_waitcnt lgkmcnt(8)
s_barrier  // (op 2978) 



ds_read_u8 v7, v8 offset:16512 // (op 501) Load local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 2998) 


ds_read_u8 v13, v11 offset:16384 // (op 503) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[38:45], v[46:53], a[0:3], v7, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 504) 







ds_read_b128 v[36:39], v6 offset:4224 // (op 506) Load local data 
ds_read_b128 v[40:43], v6 offset:4288 // (op 506) Load local data 



ds_read_b64_tr_b8 v[44:45], v5 offset:2048 // (op 508) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v5 offset:2176 // (op 508) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v5 offset:3072 // (op 508) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v5 offset:3200 // (op 508) Transpose load local data 



ds_read_u8 v7, v8 offset:16516 // (op 510) Load local data 


ds_read_u8 v13, v11 offset:16448 // (op 512) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:43], v[44:51], a[0:3], v7, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 513) 




s_ashr_i64 s[18:19], s[2:3], 8 // (op 489) 
s_mov_b32 s8, s18 // (op 489) convert

s_cmp_lt_u32 s7, s8 // (op 489) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopTop_KLoopTail_489 // (op 489) Condition: Bottom (jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopTop_KLoopTail_489 if true)
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopBottom_KLoopTail_489:









s_barrier  // (op 3148) 




v_cmp_eq_f32 vcc, 0.00000, v17 // (op 274) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalFalse_0_beta_274 // (op 274) Condition: False, jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalFalse_0_beta_274
s_nop 0
v_accvgpr_read v20, a0 // (op 282) Wait state hazard: XDL Write Hazard
v_accvgpr_read v21, a1 // (op 282) 
v_accvgpr_read v22, a2 // (op 282) 
v_accvgpr_read v23, a3 // (op 282) 
v_mul_f32 v24, v18, v20 // (op 282) 
v_mul_f32 v25, v18, v21 // (op 282) 
v_mul_f32 v26, v18, v22 // (op 282) 
v_mul_f32 v27, v18, v23 // (op 282) 

v_lshrrev_b32 v0, 6, v1 // (op 2158) 
s_mul_i32 s15, s16, s52 // (op 2158) 
s_add_u32 s15, s15, s0 // (op 2158) 
v_and_b32 v7, 63, v1 // (op 2158) 
s_add_u32 s6, s15, s9 // (op 2158) 

v_lshrrev_b32 v3, 4, v7 // (op 2158) 
s_mul_hi_u32 s15, s6, s13 // (op 2158) 
v_lshlrev_b32 v4, 2, v3 // (op 2158) 

s_sub_u32 s7, s6, s15 // (op 2158) 
v_lshl_add_u32 v3, v0, 4, v4 // (op 2158) 


s_lshr_b32 s8, s7, 1 // (op 2158) 

v_lshrrev_b32 v0, 2, v7 // (op 2158) 

s_add_u32 s7, s8, s15 // (op 2158) 


v_and_b32 v4, 3, v0 // (op 2158) 

s_lshr_b32 s8, s7, s11 // (op 2158) 

v_and_b32 v0, 3, v1 // (op 2158) 
s_and_b32 s7, s11, 2147483648 // (op 2158) 
v_lshl_add_u32 v7, v4, 2, v0 // (op 2158) 


s_cmp_eq_i32 s7, 0 // (op 2158) 
s_mov_b32 s15, scc // (op 2158) 

s_cmp_eq_i32 s15, 0 // (op 2158) 
s_cselect_b32 s7, s6, s8 // (op 2158) 



s_mul_hi_u32 s6, s7, s12 // (op 2158) 
s_sub_u32 s8, s7, s6 // (op 2158) 
s_lshr_b32 s15, s8, 1 // (op 2158) 

s_add_u32 s8, s15, s6 // (op 2158) 


s_lshr_b32 s6, s8, s10 // (op 2158) 

s_and_b32 s8, s10, 2147483648 // (op 2158) 
s_cmp_eq_i32 s8, 0 // (op 2158) 
s_mov_b32 s15, scc // (op 2158) 

s_cmp_eq_i32 s15, 0 // (op 2158) 
s_cselect_b32 s8, s7, s6 // (op 2158) 


s_mul_i32 s6, s8, s14 // (op 2158) 
v_lshl_add_u32 v0, s8, 4, v3 // (op 2158) 


s_sub_u32 s8, s7, s6 // (op 2158) 


v_lshl_add_u32 v3, s8, 4, v7 // (op 2158) 


v_mul_lo_u32 v4, v3, s48 // (op 2158) 

v_add_lshl_u32 v3, v0, v4, 2 // (op 2158) 


s_mul_i32 s19, s48, s23 // (op 2160) most significant: low of low * high
s_mul_i32 s15, s49, s22 // (op 2160) most significant: low of high * low
s_mul_i32 s18, s48, s22 // (op 2160) least significant
s_mul_hi_u32 s6, s48, s22 // (op 2160) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2160) most significant: sum
s_add_u32 s19, s19, s6 // (op 2160) most significant: sum


s_lshl_b64 s[54:55], s[18:19], 2 // (op 2160) 

s_mov_b32 s80, s50 // (op 2160) reg expression
s_mov_b32 s81, s51 // (op 2160) reg expression
s_mov_b32 s82, s54 // (op 2160) reg expression
s_mov_b32 s83, 131072 // (op 2160) call()




buffer_store_dwordx4 v[24:27], v3, s[80:83], 0 offen // (op 284) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalBottom_0_beta_274 // (op 274) Condition: Done, jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalBottom_0_beta_274
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalFalse_0_beta_274:



buffer_load_dwordx4 v[20:23], v16, s[60:63], 0 offen // (op 297) Load value
v_accvgpr_read v24, a0 // (op 294) 
v_accvgpr_read v25, a1 // (op 294) 
v_accvgpr_read v26, a2 // (op 294) 
v_accvgpr_read v27, a3 // (op 294) 
v_mul_f32 v28, v18, v24 // (op 294) 
v_mul_f32 v29, v18, v25 // (op 294) 
v_mul_f32 v30, v18, v26 // (op 294) 
v_mul_f32 v31, v18, v27 // (op 294) 

s_waitcnt vmcnt(0)
v_mul_f32 v24, v17, v20 // (op 298) 
v_mul_f32 v25, v17, v21 // (op 298) 
v_mul_f32 v26, v17, v22 // (op 298) 
v_mul_f32 v27, v17, v23 // (op 298) 

v_add_f32 v20, v24, v28 // (op 295) 
v_add_f32 v21, v25, v29 // (op 295) 
v_add_f32 v22, v26, v30 // (op 295) 
v_add_f32 v23, v27, v31 // (op 295) 


v_lshrrev_b32 v0, 6, v1 // (op 2175) 
s_mul_i32 s15, s16, s52 // (op 2175) 
s_add_u32 s15, s15, s0 // (op 2175) 
v_and_b32 v7, 63, v1 // (op 2175) 
s_add_u32 s6, s15, s9 // (op 2175) 

v_lshrrev_b32 v19, 4, v7 // (op 2175) 
s_mul_hi_u32 s15, s6, s13 // (op 2175) 
v_lshlrev_b32 v3, 2, v19 // (op 2175) 

s_sub_u32 s7, s6, s15 // (op 2175) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 2175) 


s_lshr_b32 s8, s7, 1 // (op 2175) 

v_lshrrev_b32 v0, 2, v7 // (op 2175) 

s_add_u32 s7, s8, s15 // (op 2175) 


v_and_b32 v3, 3, v0 // (op 2175) 

s_lshr_b32 s8, s7, s11 // (op 2175) 

v_and_b32 v0, 3, v1 // (op 2175) 
s_and_b32 s7, s11, 2147483648 // (op 2175) 
v_lshl_add_u32 v7, v3, 2, v0 // (op 2175) 


s_cmp_eq_i32 s7, 0 // (op 2175) 
s_mov_b32 s15, scc // (op 2175) 

s_cmp_eq_i32 s15, 0 // (op 2175) 
s_cselect_b32 s7, s6, s8 // (op 2175) 



s_mul_hi_u32 s6, s7, s12 // (op 2175) 
s_sub_u32 s8, s7, s6 // (op 2175) 
s_lshr_b32 s15, s8, 1 // (op 2175) 

s_add_u32 s8, s15, s6 // (op 2175) 


s_lshr_b32 s6, s8, s10 // (op 2175) 

s_and_b32 s8, s10, 2147483648 // (op 2175) 
s_cmp_eq_i32 s8, 0 // (op 2175) 
s_mov_b32 s15, scc // (op 2175) 

s_cmp_eq_i32 s15, 0 // (op 2175) 
s_cselect_b32 s8, s7, s6 // (op 2175) 


s_mul_i32 s6, s8, s14 // (op 2175) 
v_lshl_add_u32 v0, s8, 4, v4 // (op 2175) 


s_sub_u32 s8, s7, s6 // (op 2175) 


v_lshl_add_u32 v19, s8, 4, v7 // (op 2175) 


v_mul_lo_u32 v7, v19, s48 // (op 2175) 

v_add_lshl_u32 v19, v0, v7, 2 // (op 2175) 


s_mul_i32 s19, s48, s23 // (op 2177) most significant: low of low * high
s_mul_i32 s15, s49, s22 // (op 2177) most significant: low of high * low
s_mul_i32 s18, s48, s22 // (op 2177) least significant
s_mul_hi_u32 s6, s48, s22 // (op 2177) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2177) most significant: sum
s_add_u32 s19, s19, s6 // (op 2177) most significant: sum


s_lshl_b64 s[54:55], s[18:19], 2 // (op 2177) 

s_mov_b32 s80, s50 // (op 2177) reg expression
s_mov_b32 s81, s51 // (op 2177) reg expression
s_mov_b32 s82, s54 // (op 2177) reg expression
s_mov_b32 s83, 131072 // (op 2177) call()




buffer_store_dwordx4 v[20:23], v19, s[80:83], 0 offen // (op 296) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ConditionalBottom_0_beta_274:

s_add_u32 s0, s0, s1 // (op 483) 
s_mul_i32 s15, s52, s16 // (op 481) 
s_cmp_gt_u32 s53, s15 // (op 481) 
s_mov_b32 s6, scc // (op 481) 
s_sub_u32 s7, s53, s15 // (op 481) 

s_cmp_eq_i32 s6, 0 // (op 481) 
s_cselect_b32 s8, 0, s7 // (op 481) 


s_cmp_lt_u32 s52, s8 // (op 481) 
s_mov_b32 s15, scc // (op 481) 
s_cmp_eq_i32 s15, 0 // (op 481) 
s_cselect_b32 s6, s8, s52 // (op 481) 


s_cmp_lt_u32 s0, s6 // (op 481) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopTop_DPStreamTileLoop_481 // (op 481) Condition: Bottom (jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopTop_DPStreamTileLoop_481 if true)
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_ForLoopBottom_DPStreamTileLoop_481:











































s_endpgm  // End of RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2
.LRR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_end:
.size RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2, .LRR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2_end-RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2
  .amdhsa_next_free_vgpr 104
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 16896 // lds bytes
  .amdhsa_accum_offset 100
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
    DataTypeB: F8
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
            { type: int64, semantic: SizeFree0 },
            { type: int64, semantic: StrideB0 },
            { type: address, semantic: AddressB },
            { type: int64, semantic: StrideScaleA0 },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: StrideScaleB0 },
            { type: address, semantic: AddressMXScaleB },
            { type: int64, semantic: SizeSumScaleA },
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
  - .name: RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2
    .symbol: RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x256_UR_2.kd
    .kernarg_segment_size: 192
    .group_segment_fixed_size: 16896
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 84
    .vgpr_count: 100
    .agpr_count: 4
    .max_flat_workgroup_size: 64
    .wavefront_size: 64
    .args:
      - .name: Tensor_0_size_1_12
        .size: 8
        .offset: 0
        .value_kind: by_value
      - .name: Tensor_10_stride_1_19
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
      - .name: Tensor_2_size_1_20
        .size: 8
        .offset: 64
        .value_kind: by_value
      - .name: Tensor_0_stride_0_8
        .size: 8
        .offset: 72
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 80
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_0_9
        .size: 8
        .offset: 88
        .value_kind: by_value
      - .name: Tensor_2_stride_0_11
        .size: 8
        .offset: 96
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 104
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_stride_0_14
        .size: 8
        .offset: 112
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 120
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
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
      - .name: Tensor_4_size_1_17
        .size: 8
        .offset: 144
        .value_kind: by_value
      - .name: SCRATCH_None
        .size: 8
        .offset: 152
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: SCRATCH_ZeroedBeforeAndAfter
        .size: 8
        .offset: 160
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_21_stride_1_24
        .size: 8
        .offset: 168
        .value_kind: by_value
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 176
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: numDPTilesPerWG
        .size: 4
        .offset: 184
        .value_kind: by_value
      - .name: numDPTiles
        .size: 4
        .offset: 188
        .value_kind: by_value
...

.end_amdgpu_metadata

