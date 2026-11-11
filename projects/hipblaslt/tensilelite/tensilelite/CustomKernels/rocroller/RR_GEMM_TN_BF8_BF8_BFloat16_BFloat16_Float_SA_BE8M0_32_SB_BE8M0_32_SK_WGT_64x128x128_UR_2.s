// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2
.p2align 8
.type RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2,@function
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 9522) 
v_and_b32 v3, 15, v1 // (op 9522) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 9522) 


v_and_b32 v0, 48, v1 // (op 9522) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 9522) 


v_lshrrev_b32 v0, 6, v1 // (op 9597) 
v_and_b32 v4, 1, v0 // (op 9597) 

v_and_b32 v0, 15, v1 // (op 9597) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 9597) 


v_and_b32 v0, 48, v1 // (op 9597) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 9597) 


v_lshrrev_b32 v0, 7, v1 // (op 9673) 
v_and_b32 v5, 15, v1 // (op 9673) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 9673) 


v_and_b32 v0, 63, v1 // (op 9673) 
v_lshrrev_b32 v5, 4, v0 // (op 9673) 

v_lshl_add_u32 v0, v6, 2, v5 // (op 9673) 


v_lshrrev_b32 v5, 6, v1 // (op 9749) 
v_and_b32 v6, 1, v5 // (op 9749) 

v_and_b32 v5, 15, v1 // (op 9749) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 9749) 


v_and_b32 v5, 63, v1 // (op 9749) 
v_lshrrev_b32 v6, 4, v5 // (op 9749) 

v_lshl_add_u32 v5, v7, 2, v6 // (op 9749) 


v_lshrrev_b32 v6, 7, v1 // (op 9876) 
v_and_b32 v7, 15, v1 // (op 9876) 
v_lshl_add_u32 v8, v6, 4, v7 // (op 9876) 


v_and_b32 v6, 48, v1 // (op 9876) 
v_lshl_add_u32 v7, v8, 7, v6 // (op 9876) 


v_lshrrev_b32 v6, 6, v1 // (op 9951) 
v_and_b32 v8, 1, v6 // (op 9951) 

v_and_b32 v6, 15, v1 // (op 9951) 
v_lshl_add_u32 v9, v8, 4, v6 // (op 9951) 


v_and_b32 v6, 48, v1 // (op 9951) 
v_lshl_add_u32 v8, v9, 7, v6 // (op 9951) 


v_lshrrev_b32 v6, 7, v1 // (op 10027) 
v_and_b32 v9, 15, v1 // (op 10027) 
v_lshl_add_u32 v10, v6, 4, v9 // (op 10027) 


v_and_b32 v6, 63, v1 // (op 10027) 
v_lshrrev_b32 v9, 4, v6 // (op 10027) 

v_lshl_add_u32 v6, v10, 2, v9 // (op 10027) 


v_lshrrev_b32 v9, 6, v1 // (op 10103) 
v_and_b32 v10, 1, v9 // (op 10103) 

v_and_b32 v9, 15, v1 // (op 10103) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 10103) 


v_and_b32 v9, 63, v1 // (op 10103) 
v_lshrrev_b32 v10, 4, v9 // (op 10103) 

v_lshl_add_u32 v9, v11, 2, v10 // (op 10103) 


v_lshrrev_b32 v10, 7, v1 // (op 10217) 
v_and_b32 v11, 15, v1 // (op 10217) 
v_lshl_add_u32 v12, v10, 4, v11 // (op 10217) 


v_and_b32 v10, 48, v1 // (op 10217) 
v_lshl_add_u32 v11, v12, 7, v10 // (op 10217) 


v_lshrrev_b32 v10, 6, v1 // (op 10261) 
v_and_b32 v12, 1, v10 // (op 10261) 

v_and_b32 v10, 15, v1 // (op 10261) 
v_lshl_add_u32 v13, v12, 4, v10 // (op 10261) 


v_and_b32 v10, 48, v1 // (op 10261) 
v_lshl_add_u32 v12, v13, 7, v10 // (op 10261) 


v_lshrrev_b32 v10, 7, v1 // (op 10290) 
v_and_b32 v13, 15, v1 // (op 10290) 
v_lshl_add_u32 v14, v10, 4, v13 // (op 10290) 


v_and_b32 v10, 63, v1 // (op 10290) 
v_lshrrev_b32 v13, 4, v10 // (op 10290) 

v_lshl_add_u32 v10, v14, 2, v13 // (op 10290) 


v_lshrrev_b32 v13, 6, v1 // (op 10319) 
v_and_b32 v14, 1, v13 // (op 10319) 

v_and_b32 v13, 15, v1 // (op 10319) 
v_lshl_add_u32 v15, v14, 4, v13 // (op 10319) 


v_and_b32 v13, 63, v1 // (op 10319) 
v_lshrrev_b32 v14, 4, v13 // (op 10319) 

v_lshl_add_u32 v13, v15, 2, v14 // (op 10319) 


v_lshrrev_b32 v14, 7, v1 // (op 10363) 
v_and_b32 v15, 15, v1 // (op 10363) 
v_lshl_add_u32 v16, v14, 4, v15 // (op 10363) 


v_and_b32 v14, 48, v1 // (op 10363) 
v_lshl_add_u32 v15, v16, 7, v14 // (op 10363) 


v_lshrrev_b32 v14, 6, v1 // (op 10407) 
v_and_b32 v16, 1, v14 // (op 10407) 

v_and_b32 v14, 15, v1 // (op 10407) 
v_lshl_add_u32 v17, v16, 4, v14 // (op 10407) 


v_and_b32 v14, 48, v1 // (op 10407) 
v_lshl_add_u32 v16, v17, 7, v14 // (op 10407) 


v_lshrrev_b32 v14, 7, v1 // (op 10436) 
v_and_b32 v17, 15, v1 // (op 10436) 
v_lshl_add_u32 v18, v14, 4, v17 // (op 10436) 


v_and_b32 v14, 63, v1 // (op 10436) 
v_lshrrev_b32 v17, 4, v14 // (op 10436) 

v_lshl_add_u32 v14, v18, 2, v17 // (op 10436) 


v_lshrrev_b32 v17, 6, v1 // (op 10465) 
v_and_b32 v18, 1, v17 // (op 10465) 

v_and_b32 v17, 15, v1 // (op 10465) 
v_lshl_add_u32 v19, v18, 4, v17 // (op 10465) 


v_and_b32 v17, 63, v1 // (op 10465) 
v_lshrrev_b32 v18, 4, v17 // (op 10465) 

v_lshl_add_u32 v17, v19, 2, v18 // (op 10465) 


v_mov_b32 v18, s6 // (op 2) Move value
v_mov_b32 v19, s7 // (op 43) Move value


s_mov_b32 s0, 0 // (op 395) call()
s_mov_b32 s1, 0 // (op 396) call()
s_mov_b32 s6, 0 // (op 397) call()
s_mov_b32 s7, 0 // (op 398) call()
s_mov_b32 s18, 0 // (op 399) call()
s_mov_b32 s19, 0 // (op 400) call()
s_mov_b32 s0, 0 // (op 295) call()

s_waitcnt lgkmcnt(0)// DEBUG: Wait for scalar queue
s_mul_i32 s52, s8, s16 // (op 294) 
s_cmp_gt_u32 s9, s52 // (op 294) 
s_mov_b32 s53, scc // (op 294) 
s_sub_u32 s54, s9, s52 // (op 294) 

s_cmp_eq_i32 s53, 0 // (op 294) 
s_cselect_b32 s52, 0, s54 // (op 294) 


s_cmp_lt_u32 s8, s52 // (op 294) 
s_mov_b32 s53, scc // (op 294) 
s_cmp_eq_i32 s53, 0 // (op 294) 
s_cselect_b32 s54, s52, s8 // (op 294) 


s_cmp_lt_u32 s0, s54 // (op 294) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopBottom_SKStreamTileLoop_294 // (op 294) Condition: Top (jump to RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopBottom_SKStreamTileLoop_294 if false)
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopTop_SKStreamTileLoop_294:
v_lshrrev_b32 v20, 7, v1 // (op 9433) 
s_mul_i32 s52, s16, s8 // (op 9433) 
s_add_u32 s52, s52, s0 // (op 9433) 
v_and_b32 v21, 63, v1 // (op 9433) 
s_mul_hi_u32 s53, s52, s13 // (op 9433) 
v_lshrrev_b32 v22, 4, v21 // (op 9433) 
s_sub_u32 s54, s52, s53 // (op 9433) 
v_lshlrev_b32 v23, 2, v22 // (op 9433) 

v_lshl_add_u32 v22, v20, 4, v23 // (op 9433) 


s_lshr_b32 s55, s54, 1 // (op 9433) 

s_add_u32 s54, s55, s53 // (op 9433) 


v_lshrrev_b32 v20, 2, v21 // (op 9433) 

s_lshr_b32 s53, s54, s11 // (op 9433) 

v_and_b32 v21, 3, v20 // (op 9433) 

s_and_b32 s54, s11, 2147483648 // (op 9433) 
v_and_b32 v20, 3, v1 // (op 9433) 
s_cmp_eq_i32 s54, 0 // (op 9433) 
s_mov_b32 s55, scc // (op 9433) 

v_lshl_add_u32 v23, v21, 2, v20 // (op 9433) 


s_cmp_eq_i32 s55, 0 // (op 9433) 
s_cselect_b32 s54, s52, s53 // (op 9433) 



v_lshrrev_b32 v20, 6, v1 // (op 9433) 
s_mul_hi_u32 s52, s54, s12 // (op 9433) 
v_and_b32 v21, 1, v20 // (op 9433) 

s_sub_u32 s53, s54, s52 // (op 9433) 
v_lshl_add_u32 v20, v21, 4, v23 // (op 9433) 


s_lshr_b32 s55, s53, 1 // (op 9433) 

s_add_u32 s53, s55, s52 // (op 9433) 


s_lshr_b32 s52, s53, s10 // (op 9433) 

s_and_b32 s53, s10, 2147483648 // (op 9433) 
s_cmp_eq_i32 s53, 0 // (op 9433) 
s_mov_b32 s55, scc // (op 9433) 

s_cmp_eq_i32 s55, 0 // (op 9433) 
s_cselect_b32 s53, s54, s52 // (op 9433) 


s_mul_i32 s52, s53, s14 // (op 9433) 
v_lshl_add_u32 v21, s53, 6, v22 // (op 9433) 


s_sub_u32 s53, s54, s52 // (op 9433) 


v_lshl_add_u32 v22, s53, 7, v20 // (op 9433) 


v_mul_lo_u32 v20, v22, s4 // (op 9433) 

v_add_lshl_u32 v22, v21, v20, 1 // (op 9433) 


s_mul_i32 s53, s4, s23 // (op 9435) most significant: low of low * high
s_mul_i32 s54, s5, s22 // (op 9435) most significant: low of high * low
s_mul_i32 s52, s4, s22 // (op 9435) least significant
s_mul_hi_u32 s55, s4, s22 // (op 9435) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 9435) most significant: sum
s_add_u32 s53, s53, s55 // (op 9435) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 1 // (op 9435) 

s_mov_b32 s56, s20 // (op 9435) reg expression
s_mov_b32 s57, s21 // (op 9435) reg expression
s_mov_b32 s58, s54 // (op 9435) reg expression
s_mov_b32 s59, 131072 // (op 9435) call()

v_lshrrev_b32 v20, 7, v1 // (op 9539) 
v_and_b32 v21, 15, v1 // (op 9539) 
v_lshl_add_u32 v23, v20, 4, v21 // (op 9539) 


v_and_b32 v20, 48, v1 // (op 9539) 
v_lshl_add_u32 v21, v23, 7, v20 // (op 9539) 


v_lshrrev_b32 v20, 6, v1 // (op 9614) 
v_and_b32 v23, 1, v20 // (op 9614) 

v_and_b32 v20, 15, v1 // (op 9614) 
v_lshl_add_u32 v24, v23, 4, v20 // (op 9614) 


v_and_b32 v20, 48, v1 // (op 9614) 
v_lshl_add_u32 v23, v24, 7, v20 // (op 9614) 


v_lshrrev_b32 v20, 7, v1 // (op 9690) 
v_and_b32 v24, 15, v1 // (op 9690) 
v_lshl_add_u32 v25, v20, 4, v24 // (op 9690) 


v_and_b32 v20, 63, v1 // (op 9690) 
v_lshrrev_b32 v24, 4, v20 // (op 9690) 

v_lshl_add_u32 v20, v25, 2, v24 // (op 9690) 


v_lshrrev_b32 v24, 6, v1 // (op 9766) 
v_and_b32 v25, 1, v24 // (op 9766) 

v_and_b32 v24, 15, v1 // (op 9766) 
v_lshl_add_u32 v26, v25, 4, v24 // (op 9766) 


v_and_b32 v24, 63, v1 // (op 9766) 
v_lshrrev_b32 v25, 4, v24 // (op 9766) 

v_lshl_add_u32 v24, v26, 2, v25 // (op 9766) 


v_lshrrev_b32 v25, 7, v1 // (op 9789) 
s_mul_i32 s52, s16, s8 // (op 9789) 
s_add_u32 s52, s52, s0 // (op 9789) 
v_and_b32 v26, 63, v1 // (op 9789) 
s_mul_hi_u32 s53, s52, s13 // (op 9789) 
v_lshrrev_b32 v27, 4, v26 // (op 9789) 
s_sub_u32 s54, s52, s53 // (op 9789) 
v_lshlrev_b32 v28, 2, v27 // (op 9789) 

s_lshr_b32 s55, s54, 1 // (op 9789) 

v_lshl_add_u32 v27, v25, 4, v28 // (op 9789) 


s_add_u32 s54, s55, s53 // (op 9789) 


v_lshrrev_b32 v25, 2, v26 // (op 9789) 

s_lshr_b32 s53, s54, s11 // (op 9789) 

v_and_b32 v26, 3, v25 // (op 9789) 

s_and_b32 s54, s11, 2147483648 // (op 9789) 
v_and_b32 v25, 3, v1 // (op 9789) 
s_cmp_eq_i32 s54, 0 // (op 9789) 
s_mov_b32 s55, scc // (op 9789) 

v_lshl_add_u32 v28, v26, 2, v25 // (op 9789) 


s_cmp_eq_i32 s55, 0 // (op 9789) 
s_cselect_b32 s54, s52, s53 // (op 9789) 



v_lshrrev_b32 v25, 6, v1 // (op 9789) 
s_mul_hi_u32 s55, s54, s12 // (op 9789) 
v_and_b32 v26, 1, v25 // (op 9789) 

s_sub_u32 s52, s54, s55 // (op 9789) 
v_lshl_add_u32 v25, v26, 4, v28 // (op 9789) 


s_lshr_b32 s53, s52, 1 // (op 9789) 

s_add_u32 s52, s53, s55 // (op 9789) 


s_lshr_b32 s53, s52, s10 // (op 9789) 

s_and_b32 s52, s10, 2147483648 // (op 9789) 
s_cmp_eq_i32 s52, 0 // (op 9789) 
s_mov_b32 s55, scc // (op 9789) 

s_cmp_eq_i32 s55, 0 // (op 9789) 
s_cselect_b32 s52, s54, s53 // (op 9789) 


s_mul_i32 s53, s52, s14 // (op 9789) 
v_lshl_add_u32 v26, s52, 6, v27 // (op 9789) 


s_sub_u32 s52, s54, s53 // (op 9789) 


v_lshl_add_u32 v27, s52, 7, v25 // (op 9789) 


v_mul_lo_u32 v25, v27, s24 // (op 9789) 

v_add_lshl_u32 v27, v26, v25, 1 // (op 9789) 


s_mul_i32 s53, s24, s23 // (op 9791) most significant: low of low * high
s_mul_i32 s54, s25, s22 // (op 9791) most significant: low of high * low
s_mul_i32 s52, s24, s22 // (op 9791) least significant
s_mul_hi_u32 s55, s24, s22 // (op 9791) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 9791) most significant: sum
s_add_u32 s53, s53, s55 // (op 9791) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 1 // (op 9791) 

s_mov_b32 s60, s26 // (op 9791) reg expression
s_mov_b32 s61, s27 // (op 9791) reg expression
s_mov_b32 s62, s54 // (op 9791) reg expression
s_mov_b32 s63, 131072 // (op 9791) call()

v_lshrrev_b32 v25, 5, v1 // (op 10143) 
v_lshlrev_b32 v26, 2, v25 // (op 10143) 

v_lshl_add_u32 v25, s16, 6, v26 // (op 10143) 

v_and_b32 v26, -32, v1 // (op 10143) 
v_sub_u32 v28, v1, v26 // (op 10143) 

v_lshl_add_u32 v26, v25, 7, v28 // (op 10143) 


v_lshlrev_b32 v25, 2, v26 // (op 10143) 

s_lshl_b32 s52, s15, 15 // (op 10145) 
s_mov_b32 s64, s28 // (op 10145) reg expression
s_mov_b32 s65, s29 // (op 10145) reg expression
s_mov_b32 s66, s52 // (op 10145) reg expression
s_mov_b32 s67, 131072 // (op 10145) call()

s_mul_i32 s6, s8, s16 // (op 299) 
s_add_u32 s6, s6, s0 // (op 299) 
s_mul_hi_u32 s52, s6, s13 // (op 300) 
s_sub_u32 s53, s6, s52 // (op 300) 
s_lshr_b32 s54, s53, 1 // (op 300) 

s_add_u32 s53, s54, s52 // (op 300) 


s_lshr_b32 s52, s53, s11 // (op 300) 

s_and_b32 s53, s11, 2147483648 // (op 300) 
s_cmp_eq_i32 s53, 0 // (op 300) 
s_mov_b32 s54, scc // (op 300) 

s_cmp_eq_i32 s54, 0 // (op 300) 
s_cselect_b32 s53, s6, s52 // (op 300) 


s_ashr_i64 s[54:55], s[2:3], 7 // (op 300) 
s_mul_i32 s52, s53, s54 // (op 300) 


s_sub_u32 s18, s6, s52 // (op 300) 

s_ashr_i64 s[52:53], s[2:3], 7 // (op 301) 
s_cmp_gt_u32 s52, s18 // (op 301) 
s_mov_b32 s54, scc // (op 301) 
s_sub_u32 s55, s52, s18 // (op 301) 

s_cmp_eq_i32 s54, 0 // (op 301) 
s_cselect_b32 s52, 0, s55 // (op 301) 


s_cmp_gt_u32 s8, s0 // (op 301) 
s_mov_b32 s53, scc // (op 301) 
s_sub_u32 s54, s8, s0 // (op 301) 
s_cmp_eq_i32 s53, 0 // (op 301) 
s_cselect_b32 s55, 0, s54 // (op 301) 


s_cmp_lt_u32 s52, s55 // (op 301) 
s_mov_b32 s53, scc // (op 301) 
s_cmp_eq_i32 s53, 0 // (op 301) 
s_cselect_b32 s54, s55, s52 // (op 301) 



s_cmp_gt_u32 s9, s6 // (op 301) 
s_mov_b32 s55, scc // (op 301) 
s_sub_u32 s52, s9, s6 // (op 301) 
s_cmp_eq_i32 s55, 0 // (op 301) 
s_cselect_b32 s53, 0, s52 // (op 301) 


s_cmp_lt_u32 s54, s53 // (op 301) 
s_mov_b32 s52, scc // (op 301) 
s_cmp_eq_i32 s52, 0 // (op 301) 
s_cselect_b32 s7, s53, s54 // (op 301) 



s_add_u32 s52, s18, s7 // (op 302) 
s_sub_u32 s19, s52, 1 // (op 302) 

v_accvgpr_write a0, 0.00000 // (op 65) call()
v_accvgpr_write a1, 0.00000 // (op 65) call()
v_accvgpr_write a2, 0.00000 // (op 65) call()
v_accvgpr_write a3, 0.00000 // (op 65) call()
v_accvgpr_write a4, 0.00000 // (op 2702) call()
v_accvgpr_write a5, 0.00000 // (op 2702) call()
v_accvgpr_write a6, 0.00000 // (op 2702) call()
v_accvgpr_write a7, 0.00000 // (op 2702) call()
v_accvgpr_write a8, 0.00000 // (op 2878) call()
v_accvgpr_write a9, 0.00000 // (op 2878) call()
v_accvgpr_write a10, 0.00000 // (op 2878) call()
v_accvgpr_write a11, 0.00000 // (op 2878) call()
v_accvgpr_write a12, 0.00000 // (op 3054) call()
v_accvgpr_write a13, 0.00000 // (op 3054) call()
v_accvgpr_write a14, 0.00000 // (op 3054) call()
v_accvgpr_write a15, 0.00000 // (op 3054) call()
v_accvgpr_write a16, 0.00000 // (op 3318) call()
v_accvgpr_write a17, 0.00000 // (op 3318) call()
v_accvgpr_write a18, 0.00000 // (op 3318) call()
v_accvgpr_write a19, 0.00000 // (op 3318) call()
v_accvgpr_write a20, 0.00000 // (op 3627) call()
v_accvgpr_write a21, 0.00000 // (op 3627) call()
v_accvgpr_write a22, 0.00000 // (op 3627) call()
v_accvgpr_write a23, 0.00000 // (op 3627) call()
v_accvgpr_write a24, 0.00000 // (op 3628) call()
v_accvgpr_write a25, 0.00000 // (op 3628) call()
v_accvgpr_write a26, 0.00000 // (op 3628) call()
v_accvgpr_write a27, 0.00000 // (op 3628) call()
v_accvgpr_write a28, 0.00000 // (op 3629) call()
v_accvgpr_write a29, 0.00000 // (op 3629) call()
v_accvgpr_write a30, 0.00000 // (op 3629) call()
v_accvgpr_write a31, 0.00000 // (op 3629) call()
v_lshrrev_b32 v26, 3, v1 // (op 9000) 
s_mul_i32 s52, s16, s8 // (op 9000) 
s_add_u32 s52, s52, s0 // (op 9000) 
v_and_b32 v28, -8, v1 // (op 9000) 
s_mul_hi_u32 s53, s52, s13 // (op 9000) 
v_sub_u32 v29, v1, v28 // (op 9000) 

s_sub_u32 s54, s52, s53 // (op 9000) 
v_lshlrev_b32 v28, 4, v29 // (op 9000) 

s_lshr_b32 s55, s54, 1 // (op 9000) 

s_add_u32 s54, s55, s53 // (op 9000) 


s_lshr_b32 s53, s54, s11 // (op 9000) 

s_and_b32 s54, s11, 2147483648 // (op 9000) 
s_cmp_eq_i32 s54, 0 // (op 9000) 
s_mov_b32 s55, scc // (op 9000) 

s_cmp_eq_i32 s55, 0 // (op 9000) 
s_cselect_b32 s54, s52, s53 // (op 9000) 


s_mul_hi_u32 s53, s54, s12 // (op 9000) 
s_sub_u32 s55, s54, s53 // (op 9000) 
s_lshr_b32 s68, s55, 1 // (op 9000) 

s_add_u32 s55, s68, s53 // (op 9000) 


s_lshr_b32 s53, s55, s10 // (op 9000) 

s_and_b32 s55, s10, 2147483648 // (op 9000) 
s_cmp_eq_i32 s55, 0 // (op 9000) 
s_mov_b32 s68, scc // (op 9000) 

s_cmp_eq_i32 s68, 0 // (op 9000) 
s_cselect_b32 s55, s54, s53 // (op 9000) 


v_lshl_add_u32 v29, s55, 6, v26 // (op 9000) 


s_ashr_i64 s[68:69], s[2:3], 7 // (op 9000) 
s_mul_i32 s53, s54, s68 // (op 9000) 


s_sub_u32 s54, s52, s53 // (op 9000) 


v_lshl_add_u32 v26, s54, 7, v28 // (op 9000) 


v_mov_b32 v31, 0 // (op 9000) convert
v_mov_b32 v30, v26 // (op 9000) convert

v_mul_lo_u32 v33, v29, s31 // (op 9000) most significant: low of low * high
v_mul_hi_u32 v26, v29, s30 // (op 9000) most significant: high of low * low
v_mul_lo_u32 v32, v29, s30 // (op 9000) least significant: low of low * low
v_add_u32 v33, v33, v26 // (op 9000) most significant: sum

v_add_co_u32 v32, s[52:53], v32, v30 // (op 9000) least significant half
v_addc_co_u32 v33, s[52:53], v33, v31, s[52:53] // (op 9000) most significant half



s_mul_i32 s53, s30, s35 // (op 9002) most significant: low of low * high
s_mul_i32 s54, s31, s34 // (op 9002) most significant: low of high * low
s_mul_i32 s52, s30, s34 // (op 9002) least significant
s_mul_hi_u32 s55, s30, s34 // (op 9002) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 9002) most significant: sum
s_add_u32 s53, s53, s55 // (op 9002) most significant: sum


s_mov_b32 s68, s32 // (op 9002) reg expression
s_mov_b32 s69, s33 // (op 9002) reg expression
s_mov_b32 s70, s52 // (op 9002) reg expression
s_mov_b32 s71, 131072 // (op 9002) call()

v_lshrrev_b32 v26, 3, v1 // (op 9019) 
s_mul_i32 s52, s16, s8 // (op 9019) 
s_add_u32 s52, s52, s0 // (op 9019) 
v_and_b32 v28, -8, v1 // (op 9019) 
s_mul_hi_u32 s53, s52, s13 // (op 9019) 
v_sub_u32 v29, v1, v28 // (op 9019) 

s_sub_u32 s54, s52, s53 // (op 9019) 
v_lshlrev_b32 v28, 4, v29 // (op 9019) 

s_lshr_b32 s55, s54, 1 // (op 9019) 

s_add_u32 s54, s55, s53 // (op 9019) 


s_lshr_b32 s53, s54, s11 // (op 9019) 

s_and_b32 s54, s11, 2147483648 // (op 9019) 
s_cmp_eq_i32 s54, 0 // (op 9019) 
s_mov_b32 s55, scc // (op 9019) 

s_cmp_eq_i32 s55, 0 // (op 9019) 
s_cselect_b32 s54, s52, s53 // (op 9019) 


s_mul_hi_u32 s53, s54, s12 // (op 9019) 
s_sub_u32 s55, s54, s53 // (op 9019) 
s_lshr_b32 s72, s55, 1 // (op 9019) 

s_add_u32 s55, s72, s53 // (op 9019) 


s_lshr_b32 s53, s55, s10 // (op 9019) 

s_and_b32 s55, s10, 2147483648 // (op 9019) 
s_cmp_eq_i32 s55, 0 // (op 9019) 
s_mov_b32 s72, scc // (op 9019) 

s_cmp_eq_i32 s72, 0 // (op 9019) 
s_cselect_b32 s55, s54, s53 // (op 9019) 


v_lshl_add_u32 v29, s55, 6, v26 // (op 9019) 


s_ashr_i64 s[72:73], s[2:3], 7 // (op 9019) 
s_mul_i32 s53, s54, s72 // (op 9019) 


s_sub_u32 s54, s52, s53 // (op 9019) 


v_lshl_add_u32 v26, s54, 7, v28 // (op 9019) 


v_mov_b32 v31, 0 // (op 9019) convert
v_mov_b32 v30, v26 // (op 9019) convert

v_mul_lo_u32 v35, v29, s31 // (op 9019) most significant: low of low * high
v_mul_hi_u32 v26, v29, s30 // (op 9019) most significant: high of low * low
v_mul_lo_u32 v34, v29, s30 // (op 9019) least significant: low of low * low
v_add_u32 v35, v35, v26 // (op 9019) most significant: sum

v_add_co_u32 v34, s[52:53], v34, v30 // (op 9019) least significant half
v_addc_co_u32 v35, s[52:53], v35, v31, s[52:53] // (op 9019) most significant half



s_mul_i32 s53, s30, s35 // (op 9021) most significant: low of low * high
s_mul_i32 s54, s31, s34 // (op 9021) most significant: low of high * low
s_mul_i32 s52, s30, s34 // (op 9021) least significant
s_mul_hi_u32 s55, s30, s34 // (op 9021) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 9021) most significant: sum
s_add_u32 s53, s53, s55 // (op 9021) most significant: sum


s_mov_b32 s68, s32 // (op 9021) reg expression
s_mov_b32 s69, s33 // (op 9021) reg expression
s_mov_b32 s70, s52 // (op 9021) reg expression
s_mov_b32 s71, 131072 // (op 9021) call()

v_lshrrev_b32 v26, 3, v1 // (op 9128) 
s_mul_i32 s52, s16, s8 // (op 9128) 
s_add_u32 s52, s52, s0 // (op 9128) 
v_and_b32 v28, -8, v1 // (op 9128) 
s_mul_hi_u32 s53, s52, s13 // (op 9128) 
v_sub_u32 v29, v1, v28 // (op 9128) 

s_sub_u32 s54, s52, s53 // (op 9128) 
v_lshlrev_b32 v28, 4, v29 // (op 9128) 

s_lshr_b32 s55, s54, 1 // (op 9128) 

s_add_u32 s54, s55, s53 // (op 9128) 


s_lshr_b32 s53, s54, s11 // (op 9128) 

s_and_b32 s54, s11, 2147483648 // (op 9128) 
s_cmp_eq_i32 s54, 0 // (op 9128) 
s_mov_b32 s55, scc // (op 9128) 

s_cmp_eq_i32 s55, 0 // (op 9128) 
s_cselect_b32 s54, s52, s53 // (op 9128) 


s_mul_hi_u32 s53, s54, s12 // (op 9128) 
s_sub_u32 s55, s54, s53 // (op 9128) 
s_lshr_b32 s72, s55, 1 // (op 9128) 

s_add_u32 s55, s72, s53 // (op 9128) 


s_lshr_b32 s53, s55, s10 // (op 9128) 

s_and_b32 s55, s10, 2147483648 // (op 9128) 
s_cmp_eq_i32 s55, 0 // (op 9128) 
s_mov_b32 s72, scc // (op 9128) 

s_cmp_eq_i32 s72, 0 // (op 9128) 
s_cselect_b32 s55, s54, s53 // (op 9128) 


s_mul_i32 s53, s55, s14 // (op 9128) 

s_sub_u32 s55, s54, s53 // (op 9128) 

v_lshl_add_u32 v29, s55, 7, v26 // (op 9128) 


v_mul_lo_u32 v31, v29, s37 // (op 9128) most significant: low of low * high
v_mul_hi_u32 v26, v29, s36 // (op 9128) most significant: high of low * low
v_mul_lo_u32 v30, v29, s36 // (op 9128) least significant: low of low * low
v_add_u32 v31, v31, v26 // (op 9128) most significant: sum


s_ashr_i64 s[72:73], s[2:3], 7 // (op 9128) 
s_mul_i32 s53, s54, s72 // (op 9128) 


s_sub_u32 s54, s52, s53 // (op 9128) 


v_lshl_add_u32 v26, s54, 7, v28 // (op 9128) 


v_mov_b32 v29, 0 // (op 9128) convert
v_mov_b32 v28, v26 // (op 9128) convert

v_add_co_u32 v36, s[52:53], v28, v30 // (op 9128) least significant half
v_addc_co_u32 v37, s[52:53], v29, v31, s[52:53] // (op 9128) most significant half



s_mul_i32 s53, s36, s23 // (op 9130) most significant: low of low * high
s_mul_i32 s54, s37, s22 // (op 9130) most significant: low of high * low
s_mul_i32 s52, s36, s22 // (op 9130) least significant
s_mul_hi_u32 s55, s36, s22 // (op 9130) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 9130) most significant: sum
s_add_u32 s53, s53, s55 // (op 9130) most significant: sum


s_mov_b32 s72, s38 // (op 9130) reg expression
s_mov_b32 s73, s39 // (op 9130) reg expression
s_mov_b32 s74, s52 // (op 9130) reg expression
s_mov_b32 s75, 131072 // (op 9130) call()

v_lshrrev_b32 v26, 3, v1 // (op 9147) 
s_mul_i32 s52, s16, s8 // (op 9147) 
s_add_u32 s52, s52, s0 // (op 9147) 
v_and_b32 v28, -8, v1 // (op 9147) 
s_mul_hi_u32 s53, s52, s13 // (op 9147) 
v_sub_u32 v29, v1, v28 // (op 9147) 

s_sub_u32 s54, s52, s53 // (op 9147) 
v_lshlrev_b32 v28, 4, v29 // (op 9147) 

s_lshr_b32 s55, s54, 1 // (op 9147) 

s_add_u32 s54, s55, s53 // (op 9147) 


s_lshr_b32 s53, s54, s11 // (op 9147) 

s_and_b32 s54, s11, 2147483648 // (op 9147) 
s_cmp_eq_i32 s54, 0 // (op 9147) 
s_mov_b32 s55, scc // (op 9147) 

s_cmp_eq_i32 s55, 0 // (op 9147) 
s_cselect_b32 s54, s52, s53 // (op 9147) 


s_mul_hi_u32 s53, s54, s12 // (op 9147) 
s_sub_u32 s55, s54, s53 // (op 9147) 
s_lshr_b32 s76, s55, 1 // (op 9147) 

s_add_u32 s55, s76, s53 // (op 9147) 


s_lshr_b32 s53, s55, s10 // (op 9147) 

s_and_b32 s55, s10, 2147483648 // (op 9147) 
s_cmp_eq_i32 s55, 0 // (op 9147) 
s_mov_b32 s76, scc // (op 9147) 

s_cmp_eq_i32 s76, 0 // (op 9147) 
s_cselect_b32 s55, s54, s53 // (op 9147) 


s_mul_i32 s53, s55, s14 // (op 9147) 

s_sub_u32 s55, s54, s53 // (op 9147) 

v_lshl_add_u32 v29, s55, 7, v26 // (op 9147) 


v_mul_lo_u32 v31, v29, s37 // (op 9147) most significant: low of low * high
v_mul_hi_u32 v26, v29, s36 // (op 9147) most significant: high of low * low
v_mul_lo_u32 v30, v29, s36 // (op 9147) least significant: low of low * low
v_add_u32 v31, v31, v26 // (op 9147) most significant: sum


s_ashr_i64 s[76:77], s[2:3], 7 // (op 9147) 
s_mul_i32 s53, s54, s76 // (op 9147) 


s_sub_u32 s54, s52, s53 // (op 9147) 


v_lshl_add_u32 v26, s54, 7, v28 // (op 9147) 


v_mov_b32 v29, 0 // (op 9147) convert
v_mov_b32 v28, v26 // (op 9147) convert

v_add_co_u32 v38, s[52:53], v28, v30 // (op 9147) least significant half
v_addc_co_u32 v39, s[52:53], v29, v31, s[52:53] // (op 9147) most significant half



s_mul_i32 s53, s36, s23 // (op 9149) most significant: low of low * high
s_mul_i32 s54, s37, s22 // (op 9149) most significant: low of high * low
s_mul_i32 s52, s36, s22 // (op 9149) least significant
s_mul_hi_u32 s55, s36, s22 // (op 9149) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 9149) most significant: sum
s_add_u32 s53, s53, s55 // (op 9149) most significant: sum


s_mov_b32 s72, s38 // (op 9149) reg expression
s_mov_b32 s73, s39 // (op 9149) reg expression
s_mov_b32 s74, s52 // (op 9149) reg expression
s_mov_b32 s75, 131072 // (op 9149) call()

v_lshrrev_b32 v26, 2, v1 // (op 9256) 
s_mul_i32 s52, s16, s8 // (op 9256) 
s_add_u32 s52, s52, s0 // (op 9256) 
v_and_b32 v28, -4, v1 // (op 9256) 
s_mul_hi_u32 s53, s52, s13 // (op 9256) 
v_sub_u32 v29, v1, v28 // (op 9256) 

s_sub_u32 s54, s52, s53 // (op 9256) 
s_lshr_b32 s55, s54, 1 // (op 9256) 

s_add_u32 s54, s55, s53 // (op 9256) 


s_lshr_b32 s53, s54, s11 // (op 9256) 

s_and_b32 s54, s11, 2147483648 // (op 9256) 
s_cmp_eq_i32 s54, 0 // (op 9256) 
s_mov_b32 s55, scc // (op 9256) 

s_cmp_eq_i32 s55, 0 // (op 9256) 
s_cselect_b32 s54, s52, s53 // (op 9256) 


s_mul_hi_u32 s53, s54, s12 // (op 9256) 
s_sub_u32 s55, s54, s53 // (op 9256) 
s_lshr_b32 s76, s55, 1 // (op 9256) 

s_add_u32 s55, s76, s53 // (op 9256) 


s_lshr_b32 s53, s55, s10 // (op 9256) 

s_and_b32 s55, s10, 2147483648 // (op 9256) 
s_cmp_eq_i32 s55, 0 // (op 9256) 
s_mov_b32 s76, scc // (op 9256) 

s_cmp_eq_i32 s76, 0 // (op 9256) 
s_cselect_b32 s55, s54, s53 // (op 9256) 


v_lshl_add_u32 v28, s55, 6, v26 // (op 9256) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 9256) 
s_mul_i32 s53, s54, s76 // (op 9256) 


s_sub_u32 s54, s52, s53 // (op 9256) 


v_lshl_add_u32 v26, s54, 2, v29 // (op 9256) 


v_mul_lo_u32 v29, v28, s40 // (op 9256) 
v_add_u32 v29, v29, v26 // (op 9256) 


s_mul_i32 s53, s40, s35 // (op 9258) most significant: low of low * high
s_mul_i32 s54, s41, s34 // (op 9258) most significant: low of high * low
s_mul_i32 s52, s40, s34 // (op 9258) least significant
s_mul_hi_u32 s55, s40, s34 // (op 9258) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 9258) most significant: sum
s_add_u32 s53, s53, s55 // (op 9258) most significant: sum


s_mov_b32 s76, s42 // (op 9258) reg expression
s_mov_b32 s77, s43 // (op 9258) reg expression
s_mov_b32 s78, s52 // (op 9258) reg expression
s_mov_b32 s79, 131072 // (op 9258) call()

v_lshrrev_b32 v26, 2, v1 // (op 9275) 
s_mul_i32 s52, s16, s8 // (op 9275) 
s_add_u32 s52, s52, s0 // (op 9275) 
v_and_b32 v28, -4, v1 // (op 9275) 
s_mul_hi_u32 s53, s52, s13 // (op 9275) 
v_sub_u32 v30, v1, v28 // (op 9275) 

s_sub_u32 s54, s52, s53 // (op 9275) 
s_lshr_b32 s55, s54, 1 // (op 9275) 

s_add_u32 s54, s55, s53 // (op 9275) 


s_lshr_b32 s53, s54, s11 // (op 9275) 

s_and_b32 s54, s11, 2147483648 // (op 9275) 
s_cmp_eq_i32 s54, 0 // (op 9275) 
s_mov_b32 s55, scc // (op 9275) 

s_cmp_eq_i32 s55, 0 // (op 9275) 
s_cselect_b32 s54, s52, s53 // (op 9275) 


s_mul_hi_u32 s53, s54, s12 // (op 9275) 
s_sub_u32 s55, s54, s53 // (op 9275) 
s_lshr_b32 s80, s55, 1 // (op 9275) 

s_add_u32 s55, s80, s53 // (op 9275) 


s_lshr_b32 s53, s55, s10 // (op 9275) 

s_and_b32 s55, s10, 2147483648 // (op 9275) 
s_cmp_eq_i32 s55, 0 // (op 9275) 
s_mov_b32 s80, scc // (op 9275) 

s_cmp_eq_i32 s80, 0 // (op 9275) 
s_cselect_b32 s55, s54, s53 // (op 9275) 


v_lshl_add_u32 v28, s55, 6, v26 // (op 9275) 


s_ashr_i64 s[80:81], s[2:3], 7 // (op 9275) 
s_mul_i32 s53, s54, s80 // (op 9275) 


s_sub_u32 s54, s52, s53 // (op 9275) 


v_lshl_add_u32 v26, s54, 2, v30 // (op 9275) 


v_mul_lo_u32 v30, v28, s40 // (op 9275) 
v_add_u32 v30, v30, v26 // (op 9275) 


s_mul_i32 s53, s40, s35 // (op 9277) most significant: low of low * high
s_mul_i32 s54, s41, s34 // (op 9277) most significant: low of high * low
s_mul_i32 s52, s40, s34 // (op 9277) least significant
s_mul_hi_u32 s55, s40, s34 // (op 9277) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 9277) most significant: sum
s_add_u32 s53, s53, s55 // (op 9277) most significant: sum


s_mov_b32 s76, s42 // (op 9277) reg expression
s_mov_b32 s77, s43 // (op 9277) reg expression
s_mov_b32 s78, s52 // (op 9277) reg expression
s_mov_b32 s79, 131072 // (op 9277) call()

v_lshrrev_b32 v26, 2, v1 // (op 9384) 
s_mul_i32 s52, s16, s8 // (op 9384) 
s_add_u32 s52, s52, s0 // (op 9384) 
v_lshlrev_b32 v28, 1, v26 // (op 9384) 

s_mul_hi_u32 s53, s52, s13 // (op 9384) 
v_and_b32 v26, -4, v1 // (op 9384) 
s_sub_u32 s54, s52, s53 // (op 9384) 
v_sub_u32 v31, v1, v26 // (op 9384) 

s_lshr_b32 s55, s54, 1 // (op 9384) 

s_add_u32 s54, s55, s53 // (op 9384) 


s_lshr_b32 s53, s54, s11 // (op 9384) 

s_and_b32 s54, s11, 2147483648 // (op 9384) 
s_cmp_eq_i32 s54, 0 // (op 9384) 
s_mov_b32 s55, scc // (op 9384) 

s_cmp_eq_i32 s55, 0 // (op 9384) 
s_cselect_b32 s54, s52, s53 // (op 9384) 


s_mul_hi_u32 s53, s54, s12 // (op 9384) 
s_sub_u32 s55, s54, s53 // (op 9384) 
s_lshr_b32 s80, s55, 1 // (op 9384) 

s_add_u32 s55, s80, s53 // (op 9384) 


s_lshr_b32 s53, s55, s10 // (op 9384) 

s_and_b32 s55, s10, 2147483648 // (op 9384) 
s_cmp_eq_i32 s55, 0 // (op 9384) 
s_mov_b32 s80, scc // (op 9384) 

s_cmp_eq_i32 s80, 0 // (op 9384) 
s_cselect_b32 s55, s54, s53 // (op 9384) 


s_mul_i32 s53, s55, s14 // (op 9384) 

s_sub_u32 s55, s54, s53 // (op 9384) 

v_lshl_add_u32 v26, s55, 7, v28 // (op 9384) 


v_mul_lo_u32 v28, v26, s44 // (op 9384) 

s_ashr_i64 s[80:81], s[2:3], 7 // (op 9384) 
s_mul_i32 s53, s54, s80 // (op 9384) 


s_sub_u32 s54, s52, s53 // (op 9384) 


v_lshl_add_u32 v26, s54, 2, v31 // (op 9384) 


v_add_u32 v31, v26, v28 // (op 9384) 


s_mul_i32 s53, s44, s23 // (op 9386) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 9386) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 9386) least significant
s_mul_hi_u32 s55, s44, s22 // (op 9386) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 9386) most significant: sum
s_add_u32 s53, s53, s55 // (op 9386) most significant: sum


s_mov_b32 s80, s46 // (op 9386) reg expression
s_mov_b32 s81, s47 // (op 9386) reg expression
s_mov_b32 s82, s52 // (op 9386) reg expression
s_mov_b32 s83, 131072 // (op 9386) call()

v_lshrrev_b32 v26, 2, v1 // (op 9403) 
s_mul_i32 s52, s16, s8 // (op 9403) 
s_add_u32 s52, s52, s0 // (op 9403) 
v_lshlrev_b32 v28, 1, v26 // (op 9403) 

s_mul_hi_u32 s53, s52, s13 // (op 9403) 
v_and_b32 v26, -4, v1 // (op 9403) 
s_sub_u32 s54, s52, s53 // (op 9403) 
v_sub_u32 v40, v1, v26 // (op 9403) 

s_lshr_b32 s55, s54, 1 // (op 9403) 

s_add_u32 s54, s55, s53 // (op 9403) 


s_lshr_b32 s53, s54, s11 // (op 9403) 

s_and_b32 s54, s11, 2147483648 // (op 9403) 
s_cmp_eq_i32 s54, 0 // (op 9403) 
s_mov_b32 s55, scc // (op 9403) 

s_cmp_eq_i32 s55, 0 // (op 9403) 
s_cselect_b32 s54, s52, s53 // (op 9403) 


s_mul_hi_u32 s53, s54, s12 // (op 9403) 
s_sub_u32 s55, s54, s53 // (op 9403) 
s_lshr_b32 s84, s55, 1 // (op 9403) 

s_add_u32 s55, s84, s53 // (op 9403) 


s_lshr_b32 s53, s55, s10 // (op 9403) 

s_and_b32 s55, s10, 2147483648 // (op 9403) 
s_cmp_eq_i32 s55, 0 // (op 9403) 
s_mov_b32 s84, scc // (op 9403) 

s_cmp_eq_i32 s84, 0 // (op 9403) 
s_cselect_b32 s55, s54, s53 // (op 9403) 


s_mul_i32 s53, s55, s14 // (op 9403) 

s_sub_u32 s55, s54, s53 // (op 9403) 

v_lshl_add_u32 v26, s55, 7, v28 // (op 9403) 


v_mul_lo_u32 v28, v26, s44 // (op 9403) 

s_ashr_i64 s[84:85], s[2:3], 7 // (op 9403) 
s_mul_i32 s53, s54, s84 // (op 9403) 


s_sub_u32 s54, s52, s53 // (op 9403) 


v_lshl_add_u32 v26, s54, 2, v40 // (op 9403) 


v_add_u32 v40, v26, v28 // (op 9403) 


s_mul_i32 s53, s44, s23 // (op 9405) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 9405) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 9405) least significant
s_mul_hi_u32 s55, s44, s22 // (op 9405) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 9405) most significant: sum
s_add_u32 s53, s53, s55 // (op 9405) most significant: sum


s_mov_b32 s80, s46 // (op 9405) reg expression
s_mov_b32 s81, s47 // (op 9405) reg expression
s_mov_b32 s82, s52 // (op 9405) reg expression
s_mov_b32 s83, 131072 // (op 9405) call()

v_and_b32 v26, -8, v1 // (op 9580) 
v_sub_u32 v28, v1, v26 // (op 9580) 

v_lshlrev_b32 v26, 4, v28 // (op 9580) 

v_lshrrev_b32 v28, 3, v1 // (op 9580) 
v_lshl_add_u32 v41, v28, 7, v26 // (op 9580) 


s_nop 0
v_readfirstlane_b32 s52, v41 // (op 9580) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 10655) 

s_lshl_b64 s[54:55], s[36:37], 5 // (op 8866) 
v_mov_b64 v[42:43], s[54:55] // (op 8866) convert


v_mov_b64 v[44:45], v[36:37] // (op 8866) 
s_mov_b32 m0, s52 // (op 8866) reg expression
buffer_load_dwordx4 v44, s[72:75], 0 offen offset: 0 lds // (op 8866) Load value direct to lds
v_add_co_u32 v44, s[54:55], v44, v42 // (op 8866) least significant half
v_addc_co_u32 v45, s[54:55], 0, v45, s[54:55] // (op 8866) most significant half

s_add_u32 m0, m0, 4096 // (op 8866) 
buffer_load_dwordx4 v44, s[72:75], 0 offen offset: 0 lds // (op 8866) Load value direct to lds
v_add_co_u32 v44, s[54:55], v44, v42 // (op 8866) least significant half
v_addc_co_u32 v45, s[54:55], 0, v45, s[54:55] // (op 8866) most significant half

s_add_u32 m0, m0, 4096 // (op 8866) 
buffer_load_dwordx4 v44, s[72:75], 0 offen offset: 0 lds // (op 8866) Load value direct to lds
v_add_co_u32 v44, s[54:55], v44, v42 // (op 8866) least significant half
v_addc_co_u32 v45, s[54:55], 0, v45, s[54:55] // (op 8866) most significant half

s_add_u32 m0, m0, 4096 // (op 8866) 
buffer_load_dwordx4 v44, s[72:75], 0 offen offset: 0 lds // (op 8866) Load value direct to lds




v_mov_b64 v[42:43], s[44:45] // (op 8601) convert

v_mov_b32 v26, v31 // (op 8601) 
buffer_load_ubyte v44, v26, s[80:83], 0 offen offset: 0 // (op 8601) Load value
v_add_u32 v26, v26, v42 // (op 8601) 
buffer_load_ubyte v45, v26, s[80:83], 0 offen offset: 0 // (op 8601) Load value


v_and_b32 v26, -8, v1 // (op 9505) 
v_sub_u32 v28, v1, v26 // (op 9505) 

v_lshlrev_b32 v26, 4, v28 // (op 9505) 

v_lshrrev_b32 v28, 3, v1 // (op 9505) 
v_lshl_add_u32 v41, v28, 7, v26 // (op 9505) 


s_nop 0
v_readfirstlane_b32 s52, v41 // (op 9505) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 10636) 

s_lshl_b64 s[54:55], s[30:31], 5 // (op 8871) 
v_mov_b64 v[42:43], s[54:55] // (op 8871) convert


v_mov_b64 v[46:47], v[32:33] // (op 8871) 
s_add_u32 s53, 16384, s52 // (op 8871) 
s_mov_b32 m0, s53 // (op 8871) reg expression
buffer_load_dwordx4 v46, s[68:71], 0 offen offset: 0 lds // (op 8871) Load value direct to lds
v_add_co_u32 v46, s[54:55], v46, v42 // (op 8871) least significant half
v_addc_co_u32 v47, s[54:55], 0, v47, s[54:55] // (op 8871) most significant half

s_add_u32 m0, m0, 4096 // (op 8871) 
buffer_load_dwordx4 v46, s[68:71], 0 offen offset: 0 lds // (op 8871) Load value direct to lds







buffer_load_ubyte v26, v29, s[76:79], 0 offen offset: 0 // (op 8607) Load value
v_and_b32 v28, -8, v1 // (op 10390) 
v_sub_u32 v41, v1, v28 // (op 10390) 

v_lshlrev_b32 v28, 4, v41 // (op 10390) 

v_lshrrev_b32 v41, 3, v1 // (op 10390) 
v_lshl_add_u32 v42, v41, 7, v28 // (op 10390) 


s_nop 0
v_readfirstlane_b32 s52, v42 // (op 10390) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 10838) 

s_lshl_b64 s[54:55], s[36:37], 5 // (op 8876) 
v_mov_b64 v[42:43], s[54:55] // (op 8876) convert


v_mov_b64 v[46:47], v[36:37] // (op 8876) 
s_add_u32 s53, 24576, s52 // (op 8876) 
s_mov_b32 m0, s53 // (op 8876) reg expression
s_mov_b32 s54, 128 // (op 8876) call()
buffer_load_dwordx4 v46, s[72:75], s54 offen offset: 0 lds // (op 8876) Load value direct to lds
v_add_co_u32 v46, s[84:85], v46, v42 // (op 8876) least significant half
v_addc_co_u32 v47, s[84:85], 0, v47, s[84:85] // (op 8876) most significant half

s_add_u32 m0, m0, 4096 // (op 8876) 
buffer_load_dwordx4 v46, s[72:75], s54 offen offset: 0 lds // (op 8876) Load value direct to lds
v_add_co_u32 v46, s[84:85], v46, v42 // (op 8876) least significant half
v_addc_co_u32 v47, s[84:85], 0, v47, s[84:85] // (op 8876) most significant half

s_add_u32 m0, m0, 4096 // (op 8876) 
buffer_load_dwordx4 v46, s[72:75], s54 offen offset: 0 lds // (op 8876) Load value direct to lds
v_add_co_u32 v46, s[84:85], v46, v42 // (op 8876) least significant half
v_addc_co_u32 v47, s[84:85], 0, v47, s[84:85] // (op 8876) most significant half

s_add_u32 m0, m0, 4096 // (op 8876) 
buffer_load_dwordx4 v46, s[72:75], s54 offen offset: 0 lds // (op 8876) Load value direct to lds







v_mov_b64 v[36:37], s[44:45] // (op 8613) convert

v_mov_b32 v28, v31 // (op 8613) 
buffer_load_ubyte v42, v28, s[80:83], 0 offen offset:4 // (op 8613) Load value
v_add_u32 v28, v28, v36 // (op 8613) 
buffer_load_ubyte v43, v28, s[80:83], 0 offen offset:4 // (op 8613) Load value



v_and_b32 v28, -8, v1 // (op 10346) 
v_sub_u32 v31, v1, v28 // (op 10346) 

v_lshlrev_b32 v28, 4, v31 // (op 10346) 

v_lshrrev_b32 v31, 3, v1 // (op 10346) 
v_lshl_add_u32 v41, v31, 7, v28 // (op 10346) 


s_nop 0
v_readfirstlane_b32 s52, v41 // (op 10346) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 10828) 

s_lshl_b64 s[54:55], s[30:31], 5 // (op 8878) 
v_mov_b64 v[36:37], s[54:55] // (op 8878) convert


v_mov_b64 v[46:47], v[32:33] // (op 8878) 
s_add_u32 s53, 40960, s52 // (op 8878) 
s_mov_b32 m0, s53 // (op 8878) reg expression
s_mov_b32 s54, 128 // (op 8878) call()
buffer_load_dwordx4 v46, s[68:71], s54 offen offset: 0 lds // (op 8878) Load value direct to lds
v_add_co_u32 v46, s[84:85], v46, v36 // (op 8878) least significant half
v_addc_co_u32 v47, s[84:85], 0, v47, s[84:85] // (op 8878) most significant half

s_add_u32 m0, m0, 4096 // (op 8878) 
buffer_load_dwordx4 v46, s[68:71], s54 offen offset: 0 lds // (op 8878) Load value direct to lds









buffer_load_ubyte v28, v29, s[76:79], 0 offen offset:4 // (op 8619) Load value

v_and_b32 v29, -4, v1 // (op 9732) 
v_sub_u32 v41, v1, v29 // (op 9732) 

v_lshrrev_b32 v29, 2, v1 // (op 9732) 
v_lshl_add_u32 v31, v29, 3, v41 // (op 9732) 


s_barrier  // (op 10696) 



s_waitcnt vmcnt(13)
ds_write_b8 v31, v44 offset:49152 // (op 8622) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v31, v45 offset:49156 // (op 8622) Store local data 

v_and_b32 v29, -4, v1 // (op 9656) 
v_sub_u32 v41, v1, v29 // (op 9656) 

v_and_b32 v29, 4294967292, v1 // (op 9656) 
v_add_u32 v31, v29, v41 // (op 9656) 


s_barrier  // (op 10675) 



s_waitcnt vmcnt(9)
ds_write_b8 v31, v26 offset:49664 // (op 8624) Store local data 

s_barrier  // (op 10652) 



ds_read_b128 v[46:49], v23 // (op 8649) Load local data 
ds_read_b128 v[50:53], v23 offset:64 // (op 8649) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 10693) 



ds_read_u8 v29, v24 offset:49152 // (op 8679) Load local data 



ds_read_b128 v[54:57], v23 offset:4096 // (op 8654) Load local data 
ds_read_b128 v[58:61], v23 offset:4160 // (op 8654) Load local data 



ds_read_u8 v41, v24 offset:49280 // (op 8684) Load local data 



ds_read_b128 v[62:65], v23 offset:8192 // (op 8659) Load local data 
ds_read_b128 v[66:69], v23 offset:8256 // (op 8659) Load local data 



ds_read_u8 v31, v24 offset:49408 // (op 8689) Load local data 



ds_read_b128 v[70:73], v23 offset:12288 // (op 8664) Load local data 
ds_read_b128 v[74:77], v23 offset:12352 // (op 8664) Load local data 



ds_read_u8 v32, v24 offset:49536 // (op 8694) Load local data 
s_barrier  // (op 10633) 



ds_read_b128 v[78:81], v21 offset:16384 // (op 8639) Load local data 
ds_read_b128 v[82:85], v21 offset:16448 // (op 8639) Load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 10672) 



ds_read_u8 v33, v20 offset:49664 // (op 8669) Load local data 



ds_read_b128 v[86:89], v21 offset:20480 // (op 8644) Load local data 
ds_read_b128 v[90:93], v21 offset:20544 // (op 8644) Load local data 



ds_read_u8 v36, v20 offset:49792 // (op 8674) Load local data 
v_and_b32 v37, -8, v1 // (op 9477) 
v_sub_u32 v94, v1, v37 // (op 9477) 

v_lshlrev_b32 v37, 4, v94 // (op 9477) 

v_lshrrev_b32 v94, 3, v1 // (op 9477) 
v_lshl_add_u32 v95, v94, 7, v37 // (op 9477) 


s_nop 0
v_readfirstlane_b32 s52, v95 // (op 9477) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v37, -8, v1 // (op 9552) 
v_sub_u32 v94, v1, v37 // (op 9552) 

v_lshlrev_b32 v37, 4, v94 // (op 9552) 

v_lshrrev_b32 v94, 3, v1 // (op 9552) 
v_lshl_add_u32 v95, v94, 7, v37 // (op 9552) 


s_nop 0
v_readfirstlane_b32 s53, v95 // (op 9552) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v37, -4, v1 // (op 9627) 
v_sub_u32 v94, v1, v37 // (op 9627) 

v_and_b32 v37, 4294967292, v1 // (op 9627) 
v_add_u32 v95, v37, v94 // (op 9627) 


v_and_b32 v37, -4, v1 // (op 9703) 
v_sub_u32 v94, v1, v37 // (op 9703) 

v_lshrrev_b32 v37, 2, v1 // (op 9703) 
v_lshl_add_u32 v96, v37, 3, v94 // (op 9703) 


v_and_b32 v37, -8, v1 // (op 10331) 
v_sub_u32 v94, v1, v37 // (op 10331) 

v_lshlrev_b32 v37, 4, v94 // (op 10331) 

v_lshrrev_b32 v94, 3, v1 // (op 10331) 
v_lshl_add_u32 v97, v94, 7, v37 // (op 10331) 


s_nop 0
v_readfirstlane_b32 s54, v97 // (op 10331) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v37, -8, v1 // (op 10375) 
v_sub_u32 v94, v1, v37 // (op 10375) 

v_lshlrev_b32 v37, 4, v94 // (op 10375) 

v_lshrrev_b32 v94, 3, v1 // (op 10375) 
v_lshl_add_u32 v97, v94, 7, v37 // (op 10375) 


s_nop 0
v_readfirstlane_b32 s55, v97 // (op 10375) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v37, -4, v1 // (op 10419) 
v_sub_u32 v94, v1, v37 // (op 10419) 

v_and_b32 v37, 4294967292, v1 // (op 10419) 
v_add_u32 v97, v37, v94 // (op 10419) 


v_and_b32 v37, -4, v1 // (op 10448) 
v_sub_u32 v94, v1, v37 // (op 10448) 

v_lshrrev_b32 v37, 2, v1 // (op 10448) 
v_lshl_add_u32 v98, v37, 3, v94 // (op 10448) 


s_mov_b32 s1, 0 // (op 3382) call()
s_and_b32 s84, s7, 4294967294 // (op 3381) 
s_cmp_lt_u32 s1, s84 // (op 3381) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopBottom_KLoop_3381 // (op 3381) Condition: Top (jump to RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopBottom_KLoop_3381 if false)
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopTop_KLoop_3381:
s_add_u32 s1, s1, 2 // (op 3383) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[78:85], v[46:53], a[0:3], v33, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 90) 
s_barrier  // (op 10658) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[78:85], v[54:61], a[4:7], v33, v41 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2742) 
s_lshl_b64 s[84:85], s[36:37], 5 // (op 8791) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[78:85], v[62:69], a[8:11], v33, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2918) 
v_mov_b64 v[100:101], s[84:85] // (op 8791) convert


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[78:85], v[70:77], a[12:15], v33, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 3094) 
v_mov_b64 v[102:103], v[38:39] // (op 8791) 
v_mov_b64 v[104:105], v[102:103] // (op 8791) 
s_mov_b32 m0, s53 // (op 8791) reg expression
s_mov_b32 s84, 256 // (op 8791) call()
buffer_load_dwordx4 v104, s[72:75], s84 offen offset: 0 lds // (op 8791) Load value direct to lds
v_add_co_u32 v104, s[86:87], v104, v100 // (op 8791) least significant half
v_addc_co_u32 v105, s[86:87], 0, v105, s[86:87] // (op 8791) most significant half

s_add_u32 m0, m0, 4096 // (op 8791) 
buffer_load_dwordx4 v104, s[72:75], s84 offen offset: 0 lds // (op 8791) Load value direct to lds
v_add_co_u32 v104, s[86:87], v104, v100 // (op 8791) least significant half
v_addc_co_u32 v105, s[86:87], 0, v105, s[86:87] // (op 8791) most significant half

s_add_u32 m0, m0, 4096 // (op 8791) 
buffer_load_dwordx4 v104, s[72:75], s84 offen offset: 0 lds // (op 8791) Load value direct to lds
v_add_co_u32 v104, s[86:87], v104, v100 // (op 8791) least significant half
v_addc_co_u32 v105, s[86:87], 0, v105, s[86:87] // (op 8791) most significant half

s_add_u32 m0, m0, 4096 // (op 8791) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[86:93], v[46:53], a[16:19], v36, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 3589) 
buffer_load_dwordx4 v104, s[72:75], s84 offen offset: 0 lds // (op 8791) Load value direct to lds




v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[86:93], v[54:61], a[20:23], v36, v41 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 3543) 
v_mov_b64 v[100:101], s[44:45] // (op 25) convert

v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[86:93], v[62:69], a[24:27], v36, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 3497) 
v_mov_b32 v37, v40 // (op 25) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[86:93], v[70:77], a[28:31], v36, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 3451) 
v_mov_b32 v94, v37 // (op 25) 
buffer_load_ubyte v44, v94, s[80:83], 0 offen offset:8 // (op 25) Load value
v_add_u32 v94, v94, v100 // (op 25) 
buffer_load_ubyte v45, v94, s[80:83], 0 offen offset:8 // (op 25) Load value


s_barrier  // (op 10639) 

s_lshl_b64 s[84:85], s[30:31], 5 // (op 8786) 
v_mov_b64 v[100:101], s[84:85] // (op 8786) convert


v_mov_b64 v[104:105], v[34:35] // (op 8786) 
v_mov_b64 v[106:107], v[104:105] // (op 8786) 
s_add_u32 s84, 16384, s52 // (op 8786) 
s_mov_b32 m0, s84 // (op 8786) reg expression
s_mov_b32 s85, 256 // (op 8786) call()
buffer_load_dwordx4 v106, s[68:71], s85 offen offset: 0 lds // (op 8786) Load value direct to lds
v_add_co_u32 v106, s[86:87], v106, v100 // (op 8786) least significant half
v_addc_co_u32 v107, s[86:87], 0, v107, s[86:87] // (op 8786) most significant half

s_add_u32 m0, m0, 4096 // (op 8786) 
buffer_load_dwordx4 v106, s[68:71], s85 offen offset: 0 lds // (op 8786) Load value direct to lds






v_mov_b32 v94, v30 // (op 17) 

buffer_load_ubyte v26, v94, s[76:79], 0 offen offset:8 // (op 17) Load value
s_barrier  // (op 10854) 



s_waitcnt vmcnt(13)
ds_write_b8 v98, v42 offset:49920 // (op 2621) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v98, v43 offset:49924 // (op 2621) Store local data 
s_barrier  // (op 10848) 



s_waitcnt vmcnt(9)
ds_write_b8 v97, v28 offset:50432 // (op 2620) Store local data 
s_barrier  // (op 10834) 



ds_read_b128 v[106:109], v16 offset:24576 // (op 2613) Load local data 
ds_read_b128 v[110:113], v16 offset:24640 // (op 2613) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 10851) 



ds_read_u8 v99, v17 offset:49920 // (op 2617) Load local data 



ds_read_b128 v[114:117], v16 offset:28672 // (op 2747) Load local data 
ds_read_b128 v[118:121], v16 offset:28736 // (op 2747) Load local data 



ds_read_u8 v100, v17 offset:50048 // (op 2751) Load local data 



ds_read_b128 v[122:125], v16 offset:32768 // (op 2923) Load local data 
ds_read_b128 v[126:129], v16 offset:32832 // (op 2923) Load local data 



ds_read_u8 v101, v17 offset:50176 // (op 2927) Load local data 



ds_read_b128 v[130:133], v16 offset:36864 // (op 3099) Load local data 
ds_read_b128 v[134:137], v16 offset:36928 // (op 3099) Load local data 



ds_read_u8 v138, v17 offset:50304 // (op 3103) Load local data 
s_barrier  // (op 10825) 
s_barrier  // (op 11093) 



ds_read_b128 v[78:81], v15 offset:40960 // (op 2611) Load local data 
ds_read_b128 v[82:85], v15 offset:41024 // (op 2611) Load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 10844) 



ds_read_u8 v33, v14 offset:50432 // (op 2615) Load local data 



ds_read_b128 v[86:89], v15 offset:45056 // (op 3454) Load local data 
ds_read_b128 v[90:93], v15 offset:45120 // (op 3454) Load local data 



ds_read_u8 v36, v14 offset:50560 // (op 3457) Load local data 
s_barrier  // (op 11119) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[78:85], v[106:113], a[0:3], v33, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2618) 
s_barrier  // (op 10841) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[78:85], v[114:121], a[4:7], v33, v100 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2743) 
s_lshl_b64 s[84:85], s[36:37], 5 // (op 8847) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[78:85], v[122:129], a[8:11], v33, v101 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2919) 
v_mov_b64 v[140:141], s[84:85] // (op 8847) convert


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[78:85], v[130:137], a[12:15], v33, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 3095) 
v_mov_b64 v[142:143], v[102:103] // (op 8847) 
s_add_u32 s84, 24576, s55 // (op 8847) 
s_mov_b32 m0, s84 // (op 8847) reg expression
s_mov_b32 s85, 384 // (op 8847) call()
buffer_load_dwordx4 v142, s[72:75], s85 offen offset: 0 lds // (op 8847) Load value direct to lds
v_add_co_u32 v142, s[86:87], v142, v140 // (op 8847) least significant half
v_addc_co_u32 v143, s[86:87], 0, v143, s[86:87] // (op 8847) most significant half

s_add_u32 m0, m0, 4096 // (op 8847) 
buffer_load_dwordx4 v142, s[72:75], s85 offen offset: 0 lds // (op 8847) Load value direct to lds
v_add_co_u32 v142, s[86:87], v142, v140 // (op 8847) least significant half
v_addc_co_u32 v143, s[86:87], 0, v143, s[86:87] // (op 8847) most significant half

s_add_u32 m0, m0, 4096 // (op 8847) 
buffer_load_dwordx4 v142, s[72:75], s85 offen offset: 0 lds // (op 8847) Load value direct to lds
v_add_co_u32 v142, s[86:87], v142, v140 // (op 8847) least significant half
v_addc_co_u32 v143, s[86:87], 0, v143, s[86:87] // (op 8847) most significant half

s_add_u32 m0, m0, 4096 // (op 8847) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[86:93], v[106:113], a[16:19], v36, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 3590) 
buffer_load_dwordx4 v142, s[72:75], s85 offen offset: 0 lds // (op 8847) Load value direct to lds








v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[86:93], v[114:121], a[20:23], v36, v100 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 3544) 
v_add_co_u32 v38, vcc, 256, v38 // (op 9139) least significant half


v_addc_co_u32 v39, vcc, 0, v39, vcc // (op 9139) most significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[86:93], v[122:129], a[24:27], v36, v101 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 3498) 
v_mov_b64 v[102:103], s[44:45] // (op 2624) convert



v_mov_b32 v99, v37 // (op 2624) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[86:93], v[130:137], a[28:31], v36, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 3452) 
buffer_load_ubyte v42, v99, s[80:83], 0 offen offset:12 // (op 2624) Load value


v_add_u32 v99, v99, v102 // (op 2624) 
buffer_load_ubyte v43, v99, s[80:83], 0 offen offset:12 // (op 2624) Load value



s_barrier  // (op 10831) 
v_add_u32 v40, 8, v40 // (op 9395) 

s_lshl_b64 s[84:85], s[30:31], 5 // (op 8842) 
v_mov_b64 v[102:103], s[84:85] // (op 8842) convert


v_mov_b64 v[100:101], v[104:105] // (op 8842) 
s_add_u32 s84, 40960, s54 // (op 8842) 
s_mov_b32 m0, s84 // (op 8842) reg expression
s_mov_b32 s85, 384 // (op 8842) call()
buffer_load_dwordx4 v100, s[68:71], s85 offen offset: 0 lds // (op 8842) Load value direct to lds
v_add_co_u32 v100, s[86:87], v100, v102 // (op 8842) least significant half
v_addc_co_u32 v101, s[86:87], 0, v101, s[86:87] // (op 8842) most significant half

s_add_u32 m0, m0, 4096 // (op 8842) 
buffer_load_dwordx4 v100, s[68:71], s85 offen offset: 0 lds // (op 8842) Load value direct to lds








v_add_co_u32 v34, vcc, 256, v34 // (op 9011) least significant half
buffer_load_ubyte v28, v94, s[76:79], 0 offen offset:12 // (op 2623) Load value
v_addc_co_u32 v35, vcc, 0, v35, vcc // (op 9011) most significant half

s_barrier  // (op 10699) 
v_add_u32 v30, 8, v30 // (op 9267) 



s_waitcnt vmcnt(13)
ds_write_b8 v96, v44 offset:49152 // (op 30) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v96, v45 offset:49156 // (op 30) Store local data 
s_barrier  // (op 10678) 



s_waitcnt vmcnt(9)
ds_write_b8 v95, v26 offset:49664 // (op 21) Store local data 
s_barrier  // (op 10664) 



ds_read_b128 v[46:49], v4 // (op 77) Load local data 
ds_read_b128 v[50:53], v4 offset:64 // (op 77) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 10706) 



ds_read_u8 v29, v5 offset:49152 // (op 87) Load local data 



ds_read_b128 v[54:57], v4 offset:4096 // (op 2737) Load local data 
ds_read_b128 v[58:61], v4 offset:4160 // (op 2737) Load local data 



ds_read_u8 v41, v5 offset:49280 // (op 2741) Load local data 



ds_read_b128 v[62:65], v4 offset:8192 // (op 2913) Load local data 
ds_read_b128 v[66:69], v4 offset:8256 // (op 2913) Load local data 



ds_read_u8 v31, v5 offset:49408 // (op 2917) Load local data 



ds_read_b128 v[70:73], v4 offset:12288 // (op 3089) Load local data 
ds_read_b128 v[74:77], v4 offset:12352 // (op 3089) Load local data 



ds_read_u8 v32, v5 offset:49536 // (op 3093) Load local data 
s_barrier  // (op 10645) 



ds_read_b128 v[78:81], v3 offset:16384 // (op 72) Load local data 
ds_read_b128 v[82:85], v3 offset:16448 // (op 72) Load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 10685) 



ds_read_u8 v33, v0 offset:49664 // (op 82) Load local data 



ds_read_b128 v[86:89], v3 offset:20480 // (op 3446) Load local data 
ds_read_b128 v[90:93], v3 offset:20544 // (op 3446) Load local data 



ds_read_u8 v36, v0 offset:49792 // (op 3449) Load local data 
s_and_b32 s84, s7, 4294967294 // (op 3381) 
s_cmp_lt_u32 s1, s84 // (op 3381) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopTop_KLoop_3381 // (op 3381) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopTop_KLoop_3381 if true)
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopBottom_KLoop_3381:
















s_barrier  // (op 11101) 
s_and_b32 s52, s7, 4294967294 // (op 8915) 
v_lshrrev_b32 v26, 3, v1 // (op 8915) 
s_add_u32 s53, s0, s52 // (op 8915) 

v_and_b32 v29, -8, v1 // (op 8915) 
v_sub_u32 v30, v1, v29 // (op 8915) 

s_mul_i32 s52, s16, s8 // (op 8915) 
s_add_u32 s52, s52, s53 // (op 8915) 

v_lshlrev_b32 v29, 4, v30 // (op 8915) 

s_mul_hi_u32 s53, s52, s13 // (op 8915) 
s_sub_u32 s54, s52, s53 // (op 8915) 
s_lshr_b32 s55, s54, 1 // (op 8915) 

s_add_u32 s54, s55, s53 // (op 8915) 


s_lshr_b32 s53, s54, s11 // (op 8915) 

s_and_b32 s54, s11, 2147483648 // (op 8915) 
s_cmp_eq_i32 s54, 0 // (op 8915) 
s_mov_b32 s55, scc // (op 8915) 

s_cmp_eq_i32 s55, 0 // (op 8915) 
s_cselect_b32 s54, s52, s53 // (op 8915) 


s_mul_hi_u32 s53, s54, s12 // (op 8915) 
s_sub_u32 s55, s54, s53 // (op 8915) 
s_lshr_b32 s84, s55, 1 // (op 8915) 

s_add_u32 s55, s84, s53 // (op 8915) 


s_lshr_b32 s53, s55, s10 // (op 8915) 

s_and_b32 s55, s10, 2147483648 // (op 8915) 
s_cmp_eq_i32 s55, 0 // (op 8915) 
s_mov_b32 s84, scc // (op 8915) 

s_cmp_eq_i32 s84, 0 // (op 8915) 
s_cselect_b32 s55, s54, s53 // (op 8915) 


v_lshl_add_u32 v30, s55, 6, v26 // (op 8915) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 8915) 
s_mul_i32 s53, s54, s84 // (op 8915) 


s_sub_u32 s54, s52, s53 // (op 8915) 


v_lshl_add_u32 v26, s54, 7, v29 // (op 8915) 


v_mov_b32 v35, 0 // (op 8915) convert
v_mov_b32 v34, v26 // (op 8915) convert

v_mul_lo_u32 v45, v30, s31 // (op 8915) most significant: low of low * high
v_mul_hi_u32 v26, v30, s30 // (op 8915) most significant: high of low * low
v_mul_lo_u32 v44, v30, s30 // (op 8915) least significant: low of low * low
v_add_u32 v45, v45, v26 // (op 8915) most significant: sum

v_add_co_u32 v44, s[52:53], v44, v34 // (op 8915) least significant half
v_addc_co_u32 v45, s[52:53], v45, v35, s[52:53] // (op 8915) most significant half



s_mul_i32 s53, s30, s35 // (op 8917) most significant: low of low * high
s_mul_i32 s54, s31, s34 // (op 8917) most significant: low of high * low
s_mul_i32 s52, s30, s34 // (op 8917) least significant
s_mul_hi_u32 s55, s30, s34 // (op 8917) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 8917) most significant: sum
s_add_u32 s53, s53, s55 // (op 8917) most significant: sum


s_mov_b32 s68, s32 // (op 8917) reg expression
s_mov_b32 s69, s33 // (op 8917) reg expression
s_mov_b32 s70, s52 // (op 8917) reg expression
s_mov_b32 s71, 131072 // (op 8917) call()

s_and_b32 s52, s7, 4294967294 // (op 9063) 
v_lshrrev_b32 v26, 3, v1 // (op 9063) 
s_add_u32 s53, s0, s52 // (op 9063) 

v_and_b32 v29, -8, v1 // (op 9063) 
v_sub_u32 v30, v1, v29 // (op 9063) 

s_mul_i32 s52, s16, s8 // (op 9063) 
s_add_u32 s52, s52, s53 // (op 9063) 

v_lshlrev_b32 v29, 4, v30 // (op 9063) 

s_mul_hi_u32 s53, s52, s13 // (op 9063) 
s_sub_u32 s54, s52, s53 // (op 9063) 
s_lshr_b32 s55, s54, 1 // (op 9063) 

s_add_u32 s54, s55, s53 // (op 9063) 


s_lshr_b32 s53, s54, s11 // (op 9063) 

s_and_b32 s54, s11, 2147483648 // (op 9063) 
s_cmp_eq_i32 s54, 0 // (op 9063) 
s_mov_b32 s55, scc // (op 9063) 

s_cmp_eq_i32 s55, 0 // (op 9063) 
s_cselect_b32 s54, s52, s53 // (op 9063) 


s_mul_hi_u32 s53, s54, s12 // (op 9063) 
s_sub_u32 s55, s54, s53 // (op 9063) 
s_lshr_b32 s84, s55, 1 // (op 9063) 

s_add_u32 s55, s84, s53 // (op 9063) 


s_lshr_b32 s53, s55, s10 // (op 9063) 

s_and_b32 s55, s10, 2147483648 // (op 9063) 
s_cmp_eq_i32 s55, 0 // (op 9063) 
s_mov_b32 s84, scc // (op 9063) 

s_cmp_eq_i32 s84, 0 // (op 9063) 
s_cselect_b32 s55, s54, s53 // (op 9063) 


s_mul_i32 s53, s55, s14 // (op 9063) 

s_sub_u32 s55, s54, s53 // (op 9063) 

v_lshl_add_u32 v30, s55, 7, v26 // (op 9063) 


v_mul_lo_u32 v35, v30, s37 // (op 9063) most significant: low of low * high
v_mul_hi_u32 v26, v30, s36 // (op 9063) most significant: high of low * low
v_mul_lo_u32 v34, v30, s36 // (op 9063) least significant: low of low * low
v_add_u32 v35, v35, v26 // (op 9063) most significant: sum


s_ashr_i64 s[84:85], s[2:3], 7 // (op 9063) 
s_mul_i32 s53, s54, s84 // (op 9063) 


s_sub_u32 s54, s52, s53 // (op 9063) 


v_lshl_add_u32 v26, s54, 7, v29 // (op 9063) 


v_mov_b32 v47, 0 // (op 9063) convert
v_mov_b32 v46, v26 // (op 9063) convert

v_add_co_u32 v48, s[52:53], v46, v34 // (op 9063) least significant half
v_addc_co_u32 v49, s[52:53], v47, v35, s[52:53] // (op 9063) most significant half



s_mul_i32 s53, s36, s23 // (op 9065) most significant: low of low * high
s_mul_i32 s54, s37, s22 // (op 9065) most significant: low of high * low
s_mul_i32 s52, s36, s22 // (op 9065) least significant
s_mul_hi_u32 s55, s36, s22 // (op 9065) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 9065) most significant: sum
s_add_u32 s53, s53, s55 // (op 9065) most significant: sum


s_mov_b32 s72, s38 // (op 9065) reg expression
s_mov_b32 s73, s39 // (op 9065) reg expression
s_mov_b32 s74, s52 // (op 9065) reg expression
s_mov_b32 s75, 131072 // (op 9065) call()

s_and_b32 s52, s7, 4294967294 // (op 9191) 
v_lshrrev_b32 v26, 2, v1 // (op 9191) 
s_add_u32 s53, s0, s52 // (op 9191) 

v_and_b32 v29, -4, v1 // (op 9191) 
v_sub_u32 v30, v1, v29 // (op 9191) 

s_mul_i32 s52, s16, s8 // (op 9191) 
s_add_u32 s52, s52, s53 // (op 9191) 

s_mul_hi_u32 s53, s52, s13 // (op 9191) 
s_sub_u32 s54, s52, s53 // (op 9191) 
s_lshr_b32 s55, s54, 1 // (op 9191) 

s_add_u32 s54, s55, s53 // (op 9191) 


s_lshr_b32 s53, s54, s11 // (op 9191) 

s_and_b32 s54, s11, 2147483648 // (op 9191) 
s_cmp_eq_i32 s54, 0 // (op 9191) 
s_mov_b32 s55, scc // (op 9191) 

s_cmp_eq_i32 s55, 0 // (op 9191) 
s_cselect_b32 s54, s52, s53 // (op 9191) 


s_mul_hi_u32 s53, s54, s12 // (op 9191) 
s_sub_u32 s55, s54, s53 // (op 9191) 
s_lshr_b32 s84, s55, 1 // (op 9191) 

s_add_u32 s55, s84, s53 // (op 9191) 


s_lshr_b32 s53, s55, s10 // (op 9191) 

s_and_b32 s55, s10, 2147483648 // (op 9191) 
s_cmp_eq_i32 s55, 0 // (op 9191) 
s_mov_b32 s84, scc // (op 9191) 

s_cmp_eq_i32 s84, 0 // (op 9191) 
s_cselect_b32 s55, s54, s53 // (op 9191) 


v_lshl_add_u32 v29, s55, 6, v26 // (op 9191) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 9191) 
s_mul_i32 s53, s54, s84 // (op 9191) 


s_sub_u32 s54, s52, s53 // (op 9191) 


v_lshl_add_u32 v26, s54, 2, v30 // (op 9191) 


v_mul_lo_u32 v30, v29, s40 // (op 9191) 
v_add_u32 v30, v30, v26 // (op 9191) 


s_mul_i32 s53, s40, s35 // (op 9193) most significant: low of low * high
s_mul_i32 s54, s41, s34 // (op 9193) most significant: low of high * low
s_mul_i32 s52, s40, s34 // (op 9193) least significant
s_mul_hi_u32 s55, s40, s34 // (op 9193) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 9193) most significant: sum
s_add_u32 s53, s53, s55 // (op 9193) most significant: sum


s_mov_b32 s76, s42 // (op 9193) reg expression
s_mov_b32 s77, s43 // (op 9193) reg expression
s_mov_b32 s78, s52 // (op 9193) reg expression
s_mov_b32 s79, 131072 // (op 9193) call()

s_and_b32 s52, s7, 4294967294 // (op 9319) 
v_lshrrev_b32 v26, 2, v1 // (op 9319) 
s_add_u32 s53, s0, s52 // (op 9319) 

v_lshlrev_b32 v29, 1, v26 // (op 9319) 

v_and_b32 v26, -4, v1 // (op 9319) 
s_mul_i32 s52, s16, s8 // (op 9319) 
s_add_u32 s52, s52, s53 // (op 9319) 

v_sub_u32 v34, v1, v26 // (op 9319) 

s_mul_hi_u32 s53, s52, s13 // (op 9319) 
s_sub_u32 s54, s52, s53 // (op 9319) 
s_lshr_b32 s55, s54, 1 // (op 9319) 

s_add_u32 s54, s55, s53 // (op 9319) 


s_lshr_b32 s53, s54, s11 // (op 9319) 

s_and_b32 s54, s11, 2147483648 // (op 9319) 
s_cmp_eq_i32 s54, 0 // (op 9319) 
s_mov_b32 s55, scc // (op 9319) 

s_cmp_eq_i32 s55, 0 // (op 9319) 
s_cselect_b32 s54, s52, s53 // (op 9319) 


s_mul_hi_u32 s53, s54, s12 // (op 9319) 
s_sub_u32 s55, s54, s53 // (op 9319) 
s_lshr_b32 s84, s55, 1 // (op 9319) 

s_add_u32 s55, s84, s53 // (op 9319) 


s_lshr_b32 s53, s55, s10 // (op 9319) 

s_and_b32 s55, s10, 2147483648 // (op 9319) 
s_cmp_eq_i32 s55, 0 // (op 9319) 
s_mov_b32 s84, scc // (op 9319) 

s_cmp_eq_i32 s84, 0 // (op 9319) 
s_cselect_b32 s55, s54, s53 // (op 9319) 


s_mul_i32 s53, s55, s14 // (op 9319) 

s_sub_u32 s55, s54, s53 // (op 9319) 

v_lshl_add_u32 v26, s55, 7, v29 // (op 9319) 


v_mul_lo_u32 v29, v26, s44 // (op 9319) 

s_ashr_i64 s[84:85], s[2:3], 7 // (op 9319) 
s_mul_i32 s53, s54, s84 // (op 9319) 


s_sub_u32 s54, s52, s53 // (op 9319) 


v_lshl_add_u32 v26, s54, 2, v34 // (op 9319) 


v_add_u32 v34, v26, v29 // (op 9319) 


s_mul_i32 s53, s44, s23 // (op 9321) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 9321) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 9321) least significant
s_mul_hi_u32 s55, s44, s22 // (op 9321) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 9321) most significant: sum
s_add_u32 s53, s53, s55 // (op 9321) most significant: sum


s_mov_b32 s80, s46 // (op 9321) reg expression
s_mov_b32 s81, s47 // (op 9321) reg expression
s_mov_b32 s82, s52 // (op 9321) reg expression
s_mov_b32 s83, 131072 // (op 9321) call()

v_and_b32 v26, -8, v1 // (op 9490) 
v_sub_u32 v29, v1, v26 // (op 9490) 

v_lshlrev_b32 v26, 4, v29 // (op 9490) 

v_lshrrev_b32 v29, 3, v1 // (op 9490) 
v_lshl_add_u32 v35, v29, 7, v26 // (op 9490) 


s_nop 0
v_readfirstlane_b32 s52, v35 // (op 9490) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v26, -8, v1 // (op 9565) 
v_sub_u32 v29, v1, v26 // (op 9565) 

v_lshlrev_b32 v26, 4, v29 // (op 9565) 

v_lshrrev_b32 v29, 3, v1 // (op 9565) 
v_lshl_add_u32 v35, v29, 7, v26 // (op 9565) 


s_nop 0
v_readfirstlane_b32 s53, v35 // (op 9565) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v26, -4, v1 // (op 9640) 
v_sub_u32 v29, v1, v26 // (op 9640) 

v_and_b32 v26, 4294967292, v1 // (op 9640) 
v_add_u32 v35, v26, v29 // (op 9640) 


v_and_b32 v26, -4, v1 // (op 9716) 
v_sub_u32 v29, v1, v26 // (op 9716) 

v_lshrrev_b32 v26, 2, v1 // (op 9716) 
v_lshl_add_u32 v37, v26, 3, v29 // (op 9716) 


s_and_b32 s54, s7, 4294967294 // (op 3387) 
s_cmp_lt_u32 s54, s7 // (op 3386) 












s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopBottom_KLoopTail_3386 // (op 3386) Condition: Top (jump to RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopBottom_KLoopTail_3386 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopTop_KLoopTail_3386:
s_barrier  // (op 10642) 
s_add_u32 s54, s54, 1 // (op 3388) 

s_lshl_b64 s[84:85], s[30:31], 5 // (op 8829) 
v_mov_b64 v[28:29], s[84:85] // (op 8829) convert


v_mov_b64 v[32:33], v[44:45] // (op 8829) 
v_mov_b64 v[46:47], v[32:33] // (op 8829) 
s_add_u32 s55, 16384, s52 // (op 8829) 
s_mov_b32 m0, s55 // (op 8829) reg expression
buffer_load_dwordx4 v46, s[68:71], 0 offen offset: 0 lds // (op 8829) Load value direct to lds
v_add_co_u32 v46, s[84:85], v46, v28 // (op 8829) least significant half
v_addc_co_u32 v47, s[84:85], 0, v47, s[84:85] // (op 8829) most significant half

s_add_u32 m0, m0, 4096 // (op 8829) 
buffer_load_dwordx4 v46, s[68:71], 0 offen offset: 0 lds // (op 8829) Load value direct to lds




v_add_co_u32 v44, vcc, 128, v44 // (op 8905) least significant half
v_addc_co_u32 v45, vcc, 0, v45, vcc // (op 8905) most significant half
s_barrier  // (op 10661) 

s_lshl_b64 s[84:85], s[36:37], 5 // (op 8835) 
v_mov_b64 v[28:29], s[84:85] // (op 8835) convert


v_mov_b64 v[32:33], v[48:49] // (op 8835) 
v_mov_b64 v[46:47], v[32:33] // (op 8835) 
s_mov_b32 m0, s53 // (op 8835) reg expression
buffer_load_dwordx4 v46, s[72:75], 0 offen offset: 0 lds // (op 8835) Load value direct to lds
v_add_co_u32 v46, s[84:85], v46, v28 // (op 8835) least significant half
v_addc_co_u32 v47, s[84:85], 0, v47, s[84:85] // (op 8835) most significant half

s_add_u32 m0, m0, 4096 // (op 8835) 
buffer_load_dwordx4 v46, s[72:75], 0 offen offset: 0 lds // (op 8835) Load value direct to lds
v_add_co_u32 v46, s[84:85], v46, v28 // (op 8835) least significant half
v_addc_co_u32 v47, s[84:85], 0, v47, s[84:85] // (op 8835) most significant half

s_add_u32 m0, m0, 4096 // (op 8835) 
buffer_load_dwordx4 v46, s[72:75], 0 offen offset: 0 lds // (op 8835) Load value direct to lds
v_add_co_u32 v46, s[84:85], v46, v28 // (op 8835) least significant half
v_addc_co_u32 v47, s[84:85], 0, v47, s[84:85] // (op 8835) most significant half

s_add_u32 m0, m0, 4096 // (op 8835) 
buffer_load_dwordx4 v46, s[72:75], 0 offen offset: 0 lds // (op 8835) Load value direct to lds



v_add_co_u32 v48, vcc, 128, v48 // (op 9055) least significant half
v_addc_co_u32 v49, vcc, 0, v49, vcc // (op 9055) most significant half


v_mov_b32 v26, v30 // (op 2557) 

buffer_load_ubyte v36, v26, s[76:79], 0 offen offset: 0 // (op 2557) Load value

v_add_u32 v30, 4, v30 // (op 9183) 

s_barrier  // (op 10681) 
v_mov_b64 v[28:29], s[44:45] // (op 2558) convert




v_mov_b32 v26, v34 // (op 2558) 
v_mov_b32 v31, v26 // (op 2558) 
buffer_load_ubyte v32, v31, s[80:83], 0 offen offset: 0 // (op 2558) Load value
v_add_u32 v31, v31, v28 // (op 2558) 
buffer_load_ubyte v33, v31, s[80:83], 0 offen offset: 0 // (op 2558) Load value



v_add_u32 v34, 4, v34 // (op 9311) 
s_waitcnt vmcnt(2)
ds_write_b8 v35, v36 offset:49664 // (op 2554) Store local data 

s_barrier  // (op 10648) 



s_barrier  // (op 10702) 
ds_read_b128 v[50:53], v3 offset:16384 // (op 2545) Load local data 



ds_read_b128 v[54:57], v3 offset:16448 // (op 2545) Load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v37, v32 offset:49152 // (op 2555) Store local data 
s_barrier  // (op 10668) 
s_waitcnt vmcnt(0)
ds_write_b8 v37, v33 offset:49156 // (op 2555) Store local data 




ds_read_b128 v[58:61], v4 // (op 2547) Load local data 
ds_read_b128 v[62:65], v4 offset:64 // (op 2547) Load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 10689) 



ds_read_u8 v26, v0 offset:49664 // (op 2549) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 10709) 



ds_read_u8 v36, v5 offset:49152 // (op 2551) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[50:57], v[58:65], a[0:3], v26, v36 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2552) 
ds_read_b128 v[66:69], v4 offset:4096 // (op 2716) Load local data 
ds_read_b128 v[70:73], v4 offset:4160 // (op 2716) Load local data 



ds_read_u8 v28, v5 offset:49280 // (op 2720) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[50:57], v[66:73], a[4:7], v26, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2721) 
ds_read_b128 v[74:77], v4 offset:8192 // (op 2892) Load local data 
ds_read_b128 v[78:81], v4 offset:8256 // (op 2892) Load local data 



ds_read_u8 v29, v5 offset:49408 // (op 2896) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[50:57], v[74:81], a[8:11], v26, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2897) 
ds_read_b128 v[82:85], v4 offset:12288 // (op 3068) Load local data 
ds_read_b128 v[86:89], v4 offset:12352 // (op 3068) Load local data 



ds_read_u8 v31, v5 offset:49536 // (op 3072) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[50:57], v[82:89], a[12:15], v26, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 3073) 
ds_read_b128 v[90:93], v3 offset:20480 // (op 3330) Load local data 


ds_read_b128 v[94:97], v3 offset:20544 // (op 3330) Load local data 



ds_read_u8 v26, v0 offset:49792 // (op 3333) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[90:97], v[58:65], a[16:19], v26, v36 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 3335) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[90:97], v[66:73], a[20:23], v26, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 3366) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[90:97], v[74:81], a[24:27], v26, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 3397) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[90:97], v[82:89], a[28:31], v26, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 3428) 




s_cmp_lt_u32 s54, s7 // (op 3386) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopTop_KLoopTail_3386 // (op 3386) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopTop_KLoopTail_3386 if true)
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopBottom_KLoopTail_3386:









s_barrier  // (op 11105) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_add_u32 s0, s0, s7 // (op 296) 
s_cmp_lg_u32 s18, 0 // (op 322) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalFalse_Send_Tile_322 // (op 322) Condition: False, jump to RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalFalse_Send_Tile_322



v_add_u32 v26, 16384, v25 // (op 4260) 
buffer_store_dword a16, v26, s[64:67], 0 offen sc1 // (op 4260) Store value

v_add_u32 v26, 16896, v25 // (op 4260) 
buffer_store_dword a17, v26, s[64:67], 0 offen sc1 // (op 4260) Store value

v_add_u32 v26, 17408, v25 // (op 4260) 
buffer_store_dword a18, v26, s[64:67], 0 offen sc1 // (op 4260) Store value

v_add_u32 v26, 17920, v25 // (op 4260) 
buffer_store_dword a19, v26, s[64:67], 0 offen sc1 // (op 4260) Store value




v_add_u32 v26, 16512, v25 // (op 4262) 
buffer_store_dword a20, v26, s[64:67], 0 offen sc1 // (op 4262) Store value

v_add_u32 v26, 17024, v25 // (op 4262) 
buffer_store_dword a21, v26, s[64:67], 0 offen sc1 // (op 4262) Store value

v_add_u32 v26, 17536, v25 // (op 4262) 
buffer_store_dword a22, v26, s[64:67], 0 offen sc1 // (op 4262) Store value

v_add_u32 v26, 18048, v25 // (op 4262) 
buffer_store_dword a23, v26, s[64:67], 0 offen sc1 // (op 4262) Store value




v_add_u32 v26, 16640, v25 // (op 4264) 
buffer_store_dword a24, v26, s[64:67], 0 offen sc1 // (op 4264) Store value

v_add_u32 v26, 17152, v25 // (op 4264) 
buffer_store_dword a25, v26, s[64:67], 0 offen sc1 // (op 4264) Store value

v_add_u32 v26, 17664, v25 // (op 4264) 
buffer_store_dword a26, v26, s[64:67], 0 offen sc1 // (op 4264) Store value

v_add_u32 v26, 18176, v25 // (op 4264) 
buffer_store_dword a27, v26, s[64:67], 0 offen sc1 // (op 4264) Store value




v_add_u32 v26, 16768, v25 // (op 4266) 
buffer_store_dword a28, v26, s[64:67], 0 offen sc1 // (op 4266) Store value

v_add_u32 v26, 17280, v25 // (op 4266) 
buffer_store_dword a29, v26, s[64:67], 0 offen sc1 // (op 4266) Store value

v_add_u32 v26, 17792, v25 // (op 4266) 
buffer_store_dword a30, v26, s[64:67], 0 offen sc1 // (op 4266) Store value

v_add_u32 v26, 18304, v25 // (op 4266) 
buffer_store_dword a31, v26, s[64:67], 0 offen sc1 // (op 4266) Store value




buffer_store_dword a0, v25, s[64:67], 0 offen sc1 // (op 319) Store value
buffer_store_dword a1, v25, s[64:67], 0 offen offset:512 sc1 // (op 319) Store value
buffer_store_dword a2, v25, s[64:67], 0 offen offset:1024 sc1 // (op 319) Store value
buffer_store_dword a3, v25, s[64:67], 0 offen offset:1536 sc1 // (op 319) Store value



buffer_store_dword a4, v25, s[64:67], 0 offen offset:128 sc1 // (op 4234) Store value
buffer_store_dword a5, v25, s[64:67], 0 offen offset:640 sc1 // (op 4234) Store value
buffer_store_dword a6, v25, s[64:67], 0 offen offset:1152 sc1 // (op 4234) Store value
buffer_store_dword a7, v25, s[64:67], 0 offen offset:1664 sc1 // (op 4234) Store value



buffer_store_dword a8, v25, s[64:67], 0 offen offset:256 sc1 // (op 4235) Store value
buffer_store_dword a9, v25, s[64:67], 0 offen offset:768 sc1 // (op 4235) Store value
buffer_store_dword a10, v25, s[64:67], 0 offen offset:1280 sc1 // (op 4235) Store value
buffer_store_dword a11, v25, s[64:67], 0 offen offset:1792 sc1 // (op 4235) Store value



buffer_store_dword a12, v25, s[64:67], 0 offen offset:384 sc1 // (op 4236) Store value
buffer_store_dword a13, v25, s[64:67], 0 offen offset:896 sc1 // (op 4236) Store value
buffer_store_dword a14, v25, s[64:67], 0 offen offset:1408 sc1 // (op 4236) Store value
buffer_store_dword a15, v25, s[64:67], 0 offen offset:1920 sc1 // (op 4236) Store value
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_barrier  // (op 323) 
v_cmp_eq_i32 vcc, v1, 0 // (op 327) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalFalse_Wave0_Store_Flag_327 // (op 327) Condition: False, jump to RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalFalse_Wave0_Store_Flag_327
s_mov_b32 s52, 1 // (op 325) call()
s_mov_b32 s55, 0 // (op 326) convert
s_mov_b32 s54, s16 // (op 326) convert
s_lshl_b64 s[54:55], s[54:55], 2 // (op 326) 
s_add_u32 s84, s48, s54 // (op 326) Start of Int64 add, locking SCC
s_addc_u32 s85, s49, s55 // (op 326) End of Int64 add, unlocking SCC
s_store_dword s52, s[84:85], 0 glc // (op 326) Store scalar value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalBottom_Wave0_Store_Flag_327 // (op 327) Condition: Done, jump to RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalBottom_Wave0_Store_Flag_327
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalFalse_Wave0_Store_Flag_327:
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalBottom_Wave0_Store_Flag_327:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalBottom_Send_Tile_322 // (op 322) Condition: Done, jump to RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalBottom_Send_Tile_322
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalFalse_Send_Tile_322:
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalBottom_Send_Tile_322:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_cmp_eq_i32 s18, 0 // (op 338) 
s_mov_b32 s52, scc // (op 338) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 338) 
s_mov_b32 s53, s54 // (op 338) convert

s_sub_u32 s54, s53, 1 // (op 338) 

s_cmp_lt_u32 s19, s54 // (op 338) 
s_mov_b32 s53, scc // (op 338) 

s_and_b32 s54, s52, s53 // (op 338) 
s_cmp_lg_u32 s54, 0 // (op 338) 



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalFalse_Receive_Tile_338 // (op 338) Condition: False, jump to RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalFalse_Receive_Tile_338
s_mov_b32 s52, 0 // (op 304) call()
s_mul_i32 s53, s52, s8 // (op 303) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 303) 
s_mov_b32 s84, s54 // (op 303) convert

s_sub_u32 s54, s84, s19 // (op 303) 

s_sub_u32 s55, s54, 1 // (op 303) 

s_cmp_lt_u32 s53, s55 // (op 303) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopBottom_ReceiveTileLoop_303 // (op 303) Condition: Top (jump to RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopBottom_ReceiveTileLoop_303 if false)
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopTop_ReceiveTileLoop_303:
s_add_u32 s53, s16, 1 // (op 10170) 
v_lshrrev_b32 v20, 5, v1 // (op 10170) 
s_add_u32 s54, s53, s52 // (op 10170) 
s_lshl_b32 s54, s54, 6 // (op 10170) 

v_lshlrev_b32 v21, 2, v20 // (op 10170) 

v_add_lshl_u32 v20, s54, v21, 7 // (op 10170) 


v_and_b32 v21, -32, v1 // (op 10170) 
v_sub_u32 v23, v1, v21 // (op 10170) 

v_add_lshl_u32 v21, v20, v23, 2 // (op 10170) 


s_lshl_b32 s53, s15, 15 // (op 10172) 
s_mov_b32 s64, s28 // (op 10172) reg expression
s_mov_b32 s65, s29 // (op 10172) reg expression
s_mov_b32 s66, s53 // (op 10172) reg expression
s_mov_b32 s67, 131072 // (op 10172) call()

RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_DoWhileTop_Global_sync_spin_loop_342:
s_add_u32 s53, s16, 1 // (op 340) 
s_add_u32 s54, s53, s52 // (op 340) Start of Int64 add, locking SCC
s_addc_u32 s55, 0, 0 // (op 340) End of Int64 add, unlocking SCC
s_lshl_b64 s[54:55], s[54:55], 2 // (op 340) 

s_add_u32 s84, s48, s54 // (op 340) Start of Int64 add, locking SCC
s_addc_u32 s85, s49, s55 // (op 340) End of Int64 add, unlocking SCC
s_load_dword s53, s[84:85], 0 glc // (op 340) Load scalar value


s_waitcnt lgkmcnt(0)
s_cmp_eq_i32 s53, 0 // (op 342) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_DoWhileTop_Global_sync_spin_loop_342 // (op 342) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_DoWhileTop_Global_sync_spin_loop_342 if true)
s_barrier  // (op 346) 
v_cmp_eq_i32 vcc, v1, 0 // (op 345) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_345 // (op 345) Condition: False, jump to RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_345
s_mov_b32 s53, 0 // (op 343) call()
s_add_u32 s54, s16, 1 // (op 344) 
s_add_u32 s84, s54, s52 // (op 344) Start of Int64 add, locking SCC
s_addc_u32 s85, 0, 0 // (op 344) End of Int64 add, unlocking SCC
s_lshl_b64 s[84:85], s[84:85], 2 // (op 344) 

s_add_u32 s54, s48, s84 // (op 344) Start of Int64 add, locking SCC
s_addc_u32 s55, s49, s85 // (op 344) End of Int64 add, unlocking SCC
s_store_dword s53, s[54:55], 0 glc // (op 344) Store scalar value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_345 // (op 345) Condition: Done, jump to RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_345
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_345:
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_345:




buffer_load_dword v28, v21, s[64:67], 0 offen // (op 357) Load value
buffer_load_dword v29, v21, s[64:67], 0 offen offset:512 // (op 357) Load value
buffer_load_dword v30, v21, s[64:67], 0 offen offset:1024 // (op 357) Load value
buffer_load_dword v31, v21, s[64:67], 0 offen offset:1536 // (op 357) Load value
v_accvgpr_read v32, a0 // (op 360) reg expression
v_accvgpr_read v33, a1 // (op 360) reg expression
v_accvgpr_read v34, a2 // (op 360) reg expression
v_accvgpr_read v35, a3 // (op 360) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 362) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 362) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 362) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 362) 

v_accvgpr_write a0, v32 // (op 364) reg expression
v_accvgpr_write a1, v33 // (op 364) reg expression
v_accvgpr_write a2, v34 // (op 364) reg expression
v_accvgpr_write a3, v35 // (op 364) reg expression




buffer_load_dword v28, v21, s[64:67], 0 offen offset:128 // (op 4307) Load value
buffer_load_dword v29, v21, s[64:67], 0 offen offset:640 // (op 4307) Load value
buffer_load_dword v30, v21, s[64:67], 0 offen offset:1152 // (op 4307) Load value
buffer_load_dword v31, v21, s[64:67], 0 offen offset:1664 // (op 4307) Load value
v_accvgpr_read v32, a4 // (op 4308) reg expression
v_accvgpr_read v33, a5 // (op 4308) reg expression
v_accvgpr_read v34, a6 // (op 4308) reg expression
v_accvgpr_read v35, a7 // (op 4308) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 4309) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 4309) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 4309) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 4309) 

v_accvgpr_write a4, v32 // (op 4310) reg expression
v_accvgpr_write a5, v33 // (op 4310) reg expression
v_accvgpr_write a6, v34 // (op 4310) reg expression
v_accvgpr_write a7, v35 // (op 4310) reg expression




buffer_load_dword v28, v21, s[64:67], 0 offen offset:256 // (op 4314) Load value
buffer_load_dword v29, v21, s[64:67], 0 offen offset:768 // (op 4314) Load value
buffer_load_dword v30, v21, s[64:67], 0 offen offset:1280 // (op 4314) Load value
buffer_load_dword v31, v21, s[64:67], 0 offen offset:1792 // (op 4314) Load value
v_accvgpr_read v32, a8 // (op 4315) reg expression
v_accvgpr_read v33, a9 // (op 4315) reg expression
v_accvgpr_read v34, a10 // (op 4315) reg expression
v_accvgpr_read v35, a11 // (op 4315) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 4316) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 4316) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 4316) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 4316) 

v_accvgpr_write a8, v32 // (op 4317) reg expression
v_accvgpr_write a9, v33 // (op 4317) reg expression
v_accvgpr_write a10, v34 // (op 4317) reg expression
v_accvgpr_write a11, v35 // (op 4317) reg expression




buffer_load_dword v28, v21, s[64:67], 0 offen offset:384 // (op 4321) Load value
buffer_load_dword v29, v21, s[64:67], 0 offen offset:896 // (op 4321) Load value
buffer_load_dword v30, v21, s[64:67], 0 offen offset:1408 // (op 4321) Load value
buffer_load_dword v31, v21, s[64:67], 0 offen offset:1920 // (op 4321) Load value
v_accvgpr_read v32, a12 // (op 4322) reg expression
v_accvgpr_read v33, a13 // (op 4322) reg expression
v_accvgpr_read v34, a14 // (op 4322) reg expression
v_accvgpr_read v35, a15 // (op 4322) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 4323) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 4323) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 4323) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 4323) 

v_accvgpr_write a12, v32 // (op 4324) reg expression
v_accvgpr_write a13, v33 // (op 4324) reg expression
v_accvgpr_write a14, v34 // (op 4324) reg expression
v_accvgpr_write a15, v35 // (op 4324) reg expression




v_mov_b32 v20, 16384 // (op 4358) 
v_add_i32 v23, v21, v20 // (op 4358) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 4358) Load value

v_mov_b32 v20, 16896 // (op 4358) 
v_add_i32 v23, v21, v20 // (op 4358) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 4358) Load value

v_mov_b32 v20, 17408 // (op 4358) 
v_add_i32 v23, v21, v20 // (op 4358) 

buffer_load_dword v30, v23, s[64:67], 0 offen // (op 4358) Load value

v_mov_b32 v20, 17920 // (op 4358) 
v_add_i32 v23, v21, v20 // (op 4358) 

buffer_load_dword v31, v23, s[64:67], 0 offen // (op 4358) Load value

v_accvgpr_read v32, a16 // (op 4355) reg expression
v_accvgpr_read v33, a17 // (op 4355) reg expression
v_accvgpr_read v34, a18 // (op 4355) reg expression
v_accvgpr_read v35, a19 // (op 4355) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 4356) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 4356) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 4356) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 4356) 

v_accvgpr_write a16, v32 // (op 4357) reg expression
v_accvgpr_write a17, v33 // (op 4357) reg expression
v_accvgpr_write a18, v34 // (op 4357) reg expression
v_accvgpr_write a19, v35 // (op 4357) reg expression




v_mov_b32 v20, 16512 // (op 4363) 
v_add_i32 v23, v21, v20 // (op 4363) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 4363) Load value

v_mov_b32 v20, 17024 // (op 4363) 
v_add_i32 v23, v21, v20 // (op 4363) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 4363) Load value

v_mov_b32 v20, 17536 // (op 4363) 
v_add_i32 v23, v21, v20 // (op 4363) 

buffer_load_dword v30, v23, s[64:67], 0 offen // (op 4363) Load value

v_mov_b32 v20, 18048 // (op 4363) 
v_add_i32 v23, v21, v20 // (op 4363) 

buffer_load_dword v31, v23, s[64:67], 0 offen // (op 4363) Load value

v_accvgpr_read v32, a20 // (op 4360) reg expression
v_accvgpr_read v33, a21 // (op 4360) reg expression
v_accvgpr_read v34, a22 // (op 4360) reg expression
v_accvgpr_read v35, a23 // (op 4360) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 4361) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 4361) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 4361) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 4361) 

v_accvgpr_write a20, v32 // (op 4362) reg expression
v_accvgpr_write a21, v33 // (op 4362) reg expression
v_accvgpr_write a22, v34 // (op 4362) reg expression
v_accvgpr_write a23, v35 // (op 4362) reg expression




v_mov_b32 v20, 16640 // (op 4368) 
v_add_i32 v23, v21, v20 // (op 4368) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 4368) Load value

v_mov_b32 v20, 17152 // (op 4368) 
v_add_i32 v23, v21, v20 // (op 4368) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 4368) Load value

v_mov_b32 v20, 17664 // (op 4368) 
v_add_i32 v23, v21, v20 // (op 4368) 

buffer_load_dword v30, v23, s[64:67], 0 offen // (op 4368) Load value

v_mov_b32 v20, 18176 // (op 4368) 
v_add_i32 v23, v21, v20 // (op 4368) 

buffer_load_dword v31, v23, s[64:67], 0 offen // (op 4368) Load value

v_accvgpr_read v32, a24 // (op 4365) reg expression
v_accvgpr_read v33, a25 // (op 4365) reg expression
v_accvgpr_read v34, a26 // (op 4365) reg expression
v_accvgpr_read v35, a27 // (op 4365) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 4366) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 4366) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 4366) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 4366) 

v_accvgpr_write a24, v32 // (op 4367) reg expression
v_accvgpr_write a25, v33 // (op 4367) reg expression
v_accvgpr_write a26, v34 // (op 4367) reg expression
v_accvgpr_write a27, v35 // (op 4367) reg expression




v_mov_b32 v20, 16768 // (op 4373) 
v_add_i32 v23, v21, v20 // (op 4373) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 4373) Load value

v_mov_b32 v20, 17280 // (op 4373) 
v_add_i32 v23, v21, v20 // (op 4373) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 4373) Load value

v_mov_b32 v20, 17792 // (op 4373) 
v_add_i32 v23, v21, v20 // (op 4373) 

buffer_load_dword v30, v23, s[64:67], 0 offen // (op 4373) Load value

v_mov_b32 v20, 18304 // (op 4373) 
v_add_i32 v23, v21, v20 // (op 4373) 

buffer_load_dword v31, v23, s[64:67], 0 offen // (op 4373) Load value

v_accvgpr_read v32, a28 // (op 4370) reg expression
v_accvgpr_read v33, a29 // (op 4370) reg expression
v_accvgpr_read v34, a30 // (op 4370) reg expression
v_accvgpr_read v35, a31 // (op 4370) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 4371) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 4371) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 4371) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 4371) 

v_accvgpr_write a28, v32 // (op 4372) reg expression
v_accvgpr_write a29, v33 // (op 4372) reg expression
v_accvgpr_write a30, v34 // (op 4372) reg expression
v_accvgpr_write a31, v35 // (op 4372) reg expression

s_add_u32 s52, s52, 1 // (op 305) 

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_mul_i32 s53, s52, s8 // (op 303) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 303) 
s_mov_b32 s84, s54 // (op 303) convert

s_sub_u32 s54, s84, s19 // (op 303) 

s_sub_u32 s55, s54, 1 // (op 303) 

s_cmp_lt_u32 s53, s55 // (op 303) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopTop_ReceiveTileLoop_303 // (op 303) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopTop_ReceiveTileLoop_303 if true)
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopBottom_ReceiveTileLoop_303:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalBottom_Receive_Tile_338 // (op 338) Condition: Done, jump to RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalBottom_Receive_Tile_338
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalFalse_Receive_Tile_338:
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalBottom_Receive_Tile_338:

s_cmp_eq_i32 s18, 0 // (op 381) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_381 // (op 381) Condition: False, jump to RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_381
v_cmp_eq_f32 vcc, 0.00000, v18 // (op 135) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalFalse_0_beta_135 // (op 135) Condition: False, jump to RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalFalse_0_beta_135
v_accvgpr_read v28, a0 // (op 50) 
v_accvgpr_read v29, a1 // (op 50) 
v_accvgpr_read v30, a2 // (op 50) 
v_accvgpr_read v31, a3 // (op 50) 
v_mul_f32 v32, v19, v28 // (op 50) 
v_mul_f32 v33, v19, v29 // (op 50) 
v_mul_f32 v34, v19, v30 // (op 50) 
v_mul_f32 v35, v19, v31 // (op 50) 

v_lshrrev_b32 v28, 16, v32 // (op 56) 
v_lshrrev_b32 v29, 16, v33 // (op 56) 
v_lshrrev_b32 v30, 16, v34 // (op 56) 
v_lshrrev_b32 v31, 16, v35 // (op 56) 


v_pack_b32_f16 v20, v28, v29 // (op 58) 
v_pack_b32_f16 v21, v30, v31 // (op 58) 

v_mov_b32 v23, v27 // (op 58) reg expression

buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 58) Store value



v_accvgpr_read v28, a4 // (op 4427) 
v_accvgpr_read v29, a5 // (op 4427) 
v_accvgpr_read v30, a6 // (op 4427) 
v_accvgpr_read v31, a7 // (op 4427) 
v_mul_f32 v32, v19, v28 // (op 4427) 
v_mul_f32 v33, v19, v29 // (op 4427) 
v_mul_f32 v34, v19, v30 // (op 4427) 
v_mul_f32 v35, v19, v31 // (op 4427) 

v_lshrrev_b32 v28, 16, v32 // (op 4429) 
v_lshrrev_b32 v29, 16, v33 // (op 4429) 
v_lshrrev_b32 v30, 16, v34 // (op 4429) 
v_lshrrev_b32 v31, 16, v35 // (op 4429) 


v_pack_b32_f16 v20, v28, v29 // (op 4430) 
v_pack_b32_f16 v21, v30, v31 // (op 4430) 

s_lshl_b32 s52, s24, 6 // (op 4430) 
v_add_u32 v23, v27, s52 // (op 4430) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 4430) Store value



v_accvgpr_read v28, a8 // (op 4441) 
v_accvgpr_read v29, a9 // (op 4441) 
v_accvgpr_read v30, a10 // (op 4441) 
v_accvgpr_read v31, a11 // (op 4441) 
v_mul_f32 v32, v19, v28 // (op 4441) 
v_mul_f32 v33, v19, v29 // (op 4441) 
v_mul_f32 v34, v19, v30 // (op 4441) 
v_mul_f32 v35, v19, v31 // (op 4441) 

v_lshrrev_b32 v28, 16, v32 // (op 4443) 
v_lshrrev_b32 v29, 16, v33 // (op 4443) 
v_lshrrev_b32 v30, 16, v34 // (op 4443) 
v_lshrrev_b32 v31, 16, v35 // (op 4443) 


v_pack_b32_f16 v20, v28, v29 // (op 4444) 
v_pack_b32_f16 v21, v30, v31 // (op 4444) 

s_lshl_b32 s52, s24, 7 // (op 4444) 
v_add_u32 v23, v27, s52 // (op 4444) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 4444) Store value



v_accvgpr_read v28, a12 // (op 4455) 
v_accvgpr_read v29, a13 // (op 4455) 
v_accvgpr_read v30, a14 // (op 4455) 
v_accvgpr_read v31, a15 // (op 4455) 
v_mul_f32 v32, v19, v28 // (op 4455) 
v_mul_f32 v33, v19, v29 // (op 4455) 
v_mul_f32 v34, v19, v30 // (op 4455) 
v_mul_f32 v35, v19, v31 // (op 4455) 

v_lshrrev_b32 v28, 16, v32 // (op 4457) 
v_lshrrev_b32 v29, 16, v33 // (op 4457) 
v_lshrrev_b32 v30, 16, v34 // (op 4457) 
v_lshrrev_b32 v31, 16, v35 // (op 4457) 


v_pack_b32_f16 v20, v28, v29 // (op 4458) 
v_pack_b32_f16 v21, v30, v31 // (op 4458) 

s_lshl_b32 s52, s24, 6 // (op 4458) 
s_mul_i32 s53, 3, s52 // (op 4458) 

v_add_u32 v23, v27, s53 // (op 4458) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 4458) Store value



v_accvgpr_read v28, a16 // (op 4491) 
v_accvgpr_read v29, a17 // (op 4491) 
v_accvgpr_read v30, a18 // (op 4491) 
v_accvgpr_read v31, a19 // (op 4491) 
v_mul_f32 v32, v19, v28 // (op 4491) 
v_mul_f32 v33, v19, v29 // (op 4491) 
v_mul_f32 v34, v19, v30 // (op 4491) 
v_mul_f32 v35, v19, v31 // (op 4491) 

v_lshrrev_b32 v28, 16, v32 // (op 4493) 
v_lshrrev_b32 v29, 16, v33 // (op 4493) 
v_lshrrev_b32 v30, 16, v34 // (op 4493) 
v_lshrrev_b32 v31, 16, v35 // (op 4493) 


v_pack_b32_f16 v20, v28, v29 // (op 4495) 
v_pack_b32_f16 v21, v30, v31 // (op 4495) 

v_add_u32 v23, 64, v27 // (op 4495) 

buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 4495) Store value



v_accvgpr_read v28, a20 // (op 4505) 
v_accvgpr_read v29, a21 // (op 4505) 
v_accvgpr_read v30, a22 // (op 4505) 
v_accvgpr_read v31, a23 // (op 4505) 
v_mul_f32 v32, v19, v28 // (op 4505) 
v_mul_f32 v33, v19, v29 // (op 4505) 
v_mul_f32 v34, v19, v30 // (op 4505) 
v_mul_f32 v35, v19, v31 // (op 4505) 

v_lshrrev_b32 v28, 16, v32 // (op 4507) 
v_lshrrev_b32 v29, 16, v33 // (op 4507) 
v_lshrrev_b32 v30, 16, v34 // (op 4507) 
v_lshrrev_b32 v31, 16, v35 // (op 4507) 


v_pack_b32_f16 v20, v28, v29 // (op 4497) 
v_pack_b32_f16 v21, v30, v31 // (op 4497) 

s_lshl_b32 s52, s24, 6 // (op 4497) 
s_add_u32 s52, s52, 64 // (op 4497) 
v_add_u32 v23, v27, s52 // (op 4497) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 4497) Store value



v_accvgpr_read v28, a24 // (op 4511) 
v_accvgpr_read v29, a25 // (op 4511) 
v_accvgpr_read v30, a26 // (op 4511) 
v_accvgpr_read v31, a27 // (op 4511) 
v_mul_f32 v32, v19, v28 // (op 4511) 
v_mul_f32 v33, v19, v29 // (op 4511) 
v_mul_f32 v34, v19, v30 // (op 4511) 
v_mul_f32 v35, v19, v31 // (op 4511) 

v_lshrrev_b32 v28, 16, v32 // (op 4513) 
v_lshrrev_b32 v29, 16, v33 // (op 4513) 
v_lshrrev_b32 v30, 16, v34 // (op 4513) 
v_lshrrev_b32 v31, 16, v35 // (op 4513) 


v_pack_b32_f16 v20, v28, v29 // (op 4499) 
v_pack_b32_f16 v21, v30, v31 // (op 4499) 

s_lshl_b32 s52, s24, 7 // (op 4499) 
s_add_u32 s52, s52, 64 // (op 4499) 
v_add_u32 v23, v27, s52 // (op 4499) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 4499) Store value



v_accvgpr_read v28, a28 // (op 4517) 
v_accvgpr_read v29, a29 // (op 4517) 
v_accvgpr_read v30, a30 // (op 4517) 
v_accvgpr_read v31, a31 // (op 4517) 
v_mul_f32 v32, v19, v28 // (op 4517) 
v_mul_f32 v33, v19, v29 // (op 4517) 
v_mul_f32 v34, v19, v30 // (op 4517) 
v_mul_f32 v35, v19, v31 // (op 4517) 

v_lshrrev_b32 v28, 16, v32 // (op 4519) 
v_lshrrev_b32 v29, 16, v33 // (op 4519) 
v_lshrrev_b32 v30, 16, v34 // (op 4519) 
v_lshrrev_b32 v31, 16, v35 // (op 4519) 


v_pack_b32_f16 v20, v28, v29 // (op 4501) 
v_pack_b32_f16 v21, v30, v31 // (op 4501) 

s_lshl_b32 s52, s24, 6 // (op 4501) 
s_mul_i32 s53, 3, s52 // (op 4501) 
s_add_u32 s53, s53, 64 // (op 4501) 

v_add_u32 v23, v27, s53 // (op 4501) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 4501) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalBottom_0_beta_135 // (op 135) Condition: Done, jump to RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalBottom_0_beta_135
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalFalse_0_beta_135:


v_add_u32 v20, 64, v22 // (op 4687) 

buffer_load_dwordx2 v[24:25], v20, s[56:59], 0 offen // (op 4687) Load value

v_accvgpr_read v28, a16 // (op 4674) 
v_accvgpr_read v29, a17 // (op 4674) 
v_accvgpr_read v30, a18 // (op 4674) 
v_accvgpr_read v31, a19 // (op 4674) 
v_mul_f32 v32, v19, v28 // (op 4674) 
v_mul_f32 v33, v19, v29 // (op 4674) 
v_mul_f32 v34, v19, v30 // (op 4674) 
v_mul_f32 v35, v19, v31 // (op 4674) 

s_waitcnt vmcnt(0)
v_and_b32 v20, 65535, v24 // (op 4686) 
v_lshrrev_b32 v21, 16, v24 // (op 4686) 
v_lshlrev_b32 v20, 16, v20 // (op 4686) 
v_lshlrev_b32 v21, 16, v21 // (op 4686) 
v_mul_f32 v28, v18, v20 // (op 4686) 
v_mul_f32 v29, v18, v21 // (op 4686) 
v_and_b32 v20, 65535, v25 // (op 4686) 
v_lshrrev_b32 v21, 16, v25 // (op 4686) 
v_lshlrev_b32 v20, 16, v20 // (op 4686) 
v_lshlrev_b32 v21, 16, v21 // (op 4686) 
v_mul_f32 v30, v18, v20 // (op 4686) 
v_mul_f32 v31, v18, v21 // (op 4686) 


v_add_f32 v36, v28, v32 // (op 4675) 
v_add_f32 v37, v29, v33 // (op 4675) 
v_add_f32 v38, v30, v34 // (op 4675) 
v_add_f32 v39, v31, v35 // (op 4675) 


v_lshrrev_b32 v28, 16, v36 // (op 4676) 
v_lshrrev_b32 v29, 16, v37 // (op 4676) 
v_lshrrev_b32 v30, 16, v38 // (op 4676) 
v_lshrrev_b32 v31, 16, v39 // (op 4676) 


v_pack_b32_f16 v20, v28, v29 // (op 4678) 
v_pack_b32_f16 v21, v30, v31 // (op 4678) 

v_add_u32 v23, 64, v27 // (op 4678) 

buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 4678) Store value





s_lshl_b32 s52, s4, 6 // (op 4692) 
s_add_u32 s52, s52, 64 // (op 4692) 
v_add_u32 v20, v22, s52 // (op 4692) 


buffer_load_dwordx2 v[24:25], v20, s[56:59], 0 offen // (op 4692) Load value

v_accvgpr_read v28, a20 // (op 4704) 
v_accvgpr_read v29, a21 // (op 4704) 
v_accvgpr_read v30, a22 // (op 4704) 
v_accvgpr_read v31, a23 // (op 4704) 
v_mul_f32 v32, v19, v28 // (op 4704) 
v_mul_f32 v33, v19, v29 // (op 4704) 
v_mul_f32 v34, v19, v30 // (op 4704) 
v_mul_f32 v35, v19, v31 // (op 4704) 

s_waitcnt vmcnt(0)
v_and_b32 v20, 65535, v24 // (op 4689) 
v_lshrrev_b32 v21, 16, v24 // (op 4689) 
v_lshlrev_b32 v20, 16, v20 // (op 4689) 
v_lshlrev_b32 v21, 16, v21 // (op 4689) 
v_mul_f32 v28, v18, v20 // (op 4689) 
v_mul_f32 v29, v18, v21 // (op 4689) 
v_and_b32 v20, 65535, v25 // (op 4689) 
v_lshrrev_b32 v21, 16, v25 // (op 4689) 
v_lshlrev_b32 v20, 16, v20 // (op 4689) 
v_lshlrev_b32 v21, 16, v21 // (op 4689) 
v_mul_f32 v30, v18, v20 // (op 4689) 
v_mul_f32 v31, v18, v21 // (op 4689) 


v_add_f32 v36, v28, v32 // (op 4690) 
v_add_f32 v37, v29, v33 // (op 4690) 
v_add_f32 v38, v30, v34 // (op 4690) 
v_add_f32 v39, v31, v35 // (op 4690) 


v_lshrrev_b32 v28, 16, v36 // (op 4691) 
v_lshrrev_b32 v29, 16, v37 // (op 4691) 
v_lshrrev_b32 v30, 16, v38 // (op 4691) 
v_lshrrev_b32 v31, 16, v39 // (op 4691) 


v_pack_b32_f16 v20, v28, v29 // (op 4680) 
v_pack_b32_f16 v21, v30, v31 // (op 4680) 

s_lshl_b32 s52, s24, 6 // (op 4680) 
s_add_u32 s52, s52, 64 // (op 4680) 
v_add_u32 v23, v27, s52 // (op 4680) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 4680) Store value





s_lshl_b32 s52, s4, 7 // (op 4697) 
s_add_u32 s52, s52, 64 // (op 4697) 
v_add_u32 v20, v22, s52 // (op 4697) 


buffer_load_dwordx2 v[24:25], v20, s[56:59], 0 offen // (op 4697) Load value

v_accvgpr_read v28, a24 // (op 4706) 
v_accvgpr_read v29, a25 // (op 4706) 
v_accvgpr_read v30, a26 // (op 4706) 
v_accvgpr_read v31, a27 // (op 4706) 
v_mul_f32 v32, v19, v28 // (op 4706) 
v_mul_f32 v33, v19, v29 // (op 4706) 
v_mul_f32 v34, v19, v30 // (op 4706) 
v_mul_f32 v35, v19, v31 // (op 4706) 

s_waitcnt vmcnt(0)
v_and_b32 v20, 65535, v24 // (op 4694) 
v_lshrrev_b32 v21, 16, v24 // (op 4694) 
v_lshlrev_b32 v20, 16, v20 // (op 4694) 
v_lshlrev_b32 v21, 16, v21 // (op 4694) 
v_mul_f32 v28, v18, v20 // (op 4694) 
v_mul_f32 v29, v18, v21 // (op 4694) 
v_and_b32 v20, 65535, v25 // (op 4694) 
v_lshrrev_b32 v21, 16, v25 // (op 4694) 
v_lshlrev_b32 v20, 16, v20 // (op 4694) 
v_lshlrev_b32 v21, 16, v21 // (op 4694) 
v_mul_f32 v30, v18, v20 // (op 4694) 
v_mul_f32 v31, v18, v21 // (op 4694) 


v_add_f32 v36, v28, v32 // (op 4695) 
v_add_f32 v37, v29, v33 // (op 4695) 
v_add_f32 v38, v30, v34 // (op 4695) 
v_add_f32 v39, v31, v35 // (op 4695) 


v_lshrrev_b32 v28, 16, v36 // (op 4696) 
v_lshrrev_b32 v29, 16, v37 // (op 4696) 
v_lshrrev_b32 v30, 16, v38 // (op 4696) 
v_lshrrev_b32 v31, 16, v39 // (op 4696) 


v_pack_b32_f16 v20, v28, v29 // (op 4682) 
v_pack_b32_f16 v21, v30, v31 // (op 4682) 

s_lshl_b32 s52, s24, 7 // (op 4682) 
s_add_u32 s52, s52, 64 // (op 4682) 
v_add_u32 v23, v27, s52 // (op 4682) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 4682) Store value





s_lshl_b32 s52, s4, 6 // (op 4702) 
s_mul_i32 s53, 3, s52 // (op 4702) 
s_add_u32 s53, s53, 64 // (op 4702) 

v_add_u32 v20, v22, s53 // (op 4702) 


buffer_load_dwordx2 v[24:25], v20, s[56:59], 0 offen // (op 4702) Load value

v_accvgpr_read v28, a28 // (op 4708) 
v_accvgpr_read v29, a29 // (op 4708) 
v_accvgpr_read v30, a30 // (op 4708) 
v_accvgpr_read v31, a31 // (op 4708) 
v_mul_f32 v32, v19, v28 // (op 4708) 
v_mul_f32 v33, v19, v29 // (op 4708) 
v_mul_f32 v34, v19, v30 // (op 4708) 
v_mul_f32 v35, v19, v31 // (op 4708) 

s_waitcnt vmcnt(0)
v_and_b32 v20, 65535, v24 // (op 4699) 
v_lshrrev_b32 v21, 16, v24 // (op 4699) 
v_lshlrev_b32 v20, 16, v20 // (op 4699) 
v_lshlrev_b32 v21, 16, v21 // (op 4699) 
v_mul_f32 v28, v18, v20 // (op 4699) 
v_mul_f32 v29, v18, v21 // (op 4699) 
v_and_b32 v20, 65535, v25 // (op 4699) 
v_lshrrev_b32 v21, 16, v25 // (op 4699) 
v_lshlrev_b32 v20, 16, v20 // (op 4699) 
v_lshlrev_b32 v21, 16, v21 // (op 4699) 
v_mul_f32 v30, v18, v20 // (op 4699) 
v_mul_f32 v31, v18, v21 // (op 4699) 


v_add_f32 v36, v28, v32 // (op 4700) 
v_add_f32 v37, v29, v33 // (op 4700) 
v_add_f32 v38, v30, v34 // (op 4700) 
v_add_f32 v39, v31, v35 // (op 4700) 


v_lshrrev_b32 v28, 16, v36 // (op 4701) 
v_lshrrev_b32 v29, 16, v37 // (op 4701) 
v_lshrrev_b32 v30, 16, v38 // (op 4701) 
v_lshrrev_b32 v31, 16, v39 // (op 4701) 


v_pack_b32_f16 v20, v28, v29 // (op 4684) 
v_pack_b32_f16 v21, v30, v31 // (op 4684) 

s_lshl_b32 s52, s24, 6 // (op 4684) 
s_mul_i32 s53, 3, s52 // (op 4684) 
s_add_u32 s53, s53, 64 // (op 4684) 

v_add_u32 v23, v27, s53 // (op 4684) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 4684) Store value





v_mov_b32 v20, v22 // (op 148) reg expression

buffer_load_dwordx2 v[24:25], v20, s[56:59], 0 offen // (op 148) Load value

v_accvgpr_read v28, a0 // (op 144) 
v_accvgpr_read v29, a1 // (op 144) 
v_accvgpr_read v30, a2 // (op 144) 
v_accvgpr_read v31, a3 // (op 144) 
v_mul_f32 v32, v19, v28 // (op 144) 
v_mul_f32 v33, v19, v29 // (op 144) 
v_mul_f32 v34, v19, v30 // (op 144) 
v_mul_f32 v35, v19, v31 // (op 144) 

s_waitcnt vmcnt(0)
v_and_b32 v20, 65535, v24 // (op 149) 
v_lshrrev_b32 v21, 16, v24 // (op 149) 
v_lshlrev_b32 v20, 16, v20 // (op 149) 
v_lshlrev_b32 v21, 16, v21 // (op 149) 
v_mul_f32 v28, v18, v20 // (op 149) 
v_mul_f32 v29, v18, v21 // (op 149) 
v_and_b32 v20, 65535, v25 // (op 149) 
v_lshrrev_b32 v21, 16, v25 // (op 149) 
v_lshlrev_b32 v20, 16, v20 // (op 149) 
v_lshlrev_b32 v21, 16, v21 // (op 149) 
v_mul_f32 v30, v18, v20 // (op 149) 
v_mul_f32 v31, v18, v21 // (op 149) 


v_add_f32 v36, v28, v32 // (op 145) 
v_add_f32 v37, v29, v33 // (op 145) 
v_add_f32 v38, v30, v34 // (op 145) 
v_add_f32 v39, v31, v35 // (op 145) 


v_lshrrev_b32 v28, 16, v36 // (op 146) 
v_lshrrev_b32 v29, 16, v37 // (op 146) 
v_lshrrev_b32 v30, 16, v38 // (op 146) 
v_lshrrev_b32 v31, 16, v39 // (op 146) 


v_pack_b32_f16 v20, v28, v29 // (op 147) 
v_pack_b32_f16 v21, v30, v31 // (op 147) 

v_mov_b32 v23, v27 // (op 147) reg expression

buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 147) Store value





s_lshl_b32 s52, s4, 6 // (op 4595) 
v_add_u32 v20, v22, s52 // (op 4595) 


buffer_load_dwordx2 v[24:25], v20, s[56:59], 0 offen // (op 4595) Load value

v_accvgpr_read v28, a4 // (op 4591) 
v_accvgpr_read v29, a5 // (op 4591) 
v_accvgpr_read v30, a6 // (op 4591) 
v_accvgpr_read v31, a7 // (op 4591) 
v_mul_f32 v32, v19, v28 // (op 4591) 
v_mul_f32 v33, v19, v29 // (op 4591) 
v_mul_f32 v34, v19, v30 // (op 4591) 
v_mul_f32 v35, v19, v31 // (op 4591) 

s_waitcnt vmcnt(0)
v_and_b32 v20, 65535, v24 // (op 4596) 
v_lshrrev_b32 v21, 16, v24 // (op 4596) 
v_lshlrev_b32 v20, 16, v20 // (op 4596) 
v_lshlrev_b32 v21, 16, v21 // (op 4596) 
v_mul_f32 v28, v18, v20 // (op 4596) 
v_mul_f32 v29, v18, v21 // (op 4596) 
v_and_b32 v20, 65535, v25 // (op 4596) 
v_lshrrev_b32 v21, 16, v25 // (op 4596) 
v_lshlrev_b32 v20, 16, v20 // (op 4596) 
v_lshlrev_b32 v21, 16, v21 // (op 4596) 
v_mul_f32 v30, v18, v20 // (op 4596) 
v_mul_f32 v31, v18, v21 // (op 4596) 


v_add_f32 v36, v28, v32 // (op 4592) 
v_add_f32 v37, v29, v33 // (op 4592) 
v_add_f32 v38, v30, v34 // (op 4592) 
v_add_f32 v39, v31, v35 // (op 4592) 


v_lshrrev_b32 v28, 16, v36 // (op 4593) 
v_lshrrev_b32 v29, 16, v37 // (op 4593) 
v_lshrrev_b32 v30, 16, v38 // (op 4593) 
v_lshrrev_b32 v31, 16, v39 // (op 4593) 


v_pack_b32_f16 v20, v28, v29 // (op 4594) 
v_pack_b32_f16 v21, v30, v31 // (op 4594) 

s_lshl_b32 s52, s24, 6 // (op 4594) 
v_add_u32 v23, v27, s52 // (op 4594) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 4594) Store value





s_lshl_b32 s52, s4, 7 // (op 4609) 
v_add_u32 v20, v22, s52 // (op 4609) 


buffer_load_dwordx2 v[24:25], v20, s[56:59], 0 offen // (op 4609) Load value

v_accvgpr_read v28, a8 // (op 4605) 
v_accvgpr_read v29, a9 // (op 4605) 
v_accvgpr_read v30, a10 // (op 4605) 
v_accvgpr_read v31, a11 // (op 4605) 
v_mul_f32 v32, v19, v28 // (op 4605) 
v_mul_f32 v33, v19, v29 // (op 4605) 
v_mul_f32 v34, v19, v30 // (op 4605) 
v_mul_f32 v35, v19, v31 // (op 4605) 

s_waitcnt vmcnt(0)
v_and_b32 v20, 65535, v24 // (op 4610) 
v_lshrrev_b32 v21, 16, v24 // (op 4610) 
v_lshlrev_b32 v20, 16, v20 // (op 4610) 
v_lshlrev_b32 v21, 16, v21 // (op 4610) 
v_mul_f32 v28, v18, v20 // (op 4610) 
v_mul_f32 v29, v18, v21 // (op 4610) 
v_and_b32 v20, 65535, v25 // (op 4610) 
v_lshrrev_b32 v21, 16, v25 // (op 4610) 
v_lshlrev_b32 v20, 16, v20 // (op 4610) 
v_lshlrev_b32 v21, 16, v21 // (op 4610) 
v_mul_f32 v30, v18, v20 // (op 4610) 
v_mul_f32 v31, v18, v21 // (op 4610) 


v_add_f32 v36, v28, v32 // (op 4606) 
v_add_f32 v37, v29, v33 // (op 4606) 
v_add_f32 v38, v30, v34 // (op 4606) 
v_add_f32 v39, v31, v35 // (op 4606) 


v_lshrrev_b32 v28, 16, v36 // (op 4607) 
v_lshrrev_b32 v29, 16, v37 // (op 4607) 
v_lshrrev_b32 v30, 16, v38 // (op 4607) 
v_lshrrev_b32 v31, 16, v39 // (op 4607) 


v_pack_b32_f16 v20, v28, v29 // (op 4608) 
v_pack_b32_f16 v21, v30, v31 // (op 4608) 

s_lshl_b32 s52, s24, 7 // (op 4608) 
v_add_u32 v23, v27, s52 // (op 4608) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 4608) Store value





s_lshl_b32 s52, s4, 6 // (op 4623) 
s_mul_i32 s53, 3, s52 // (op 4623) 

v_add_u32 v20, v22, s53 // (op 4623) 


buffer_load_dwordx2 v[24:25], v20, s[56:59], 0 offen // (op 4623) Load value

v_accvgpr_read v28, a12 // (op 4619) 
v_accvgpr_read v29, a13 // (op 4619) 
v_accvgpr_read v30, a14 // (op 4619) 
v_accvgpr_read v31, a15 // (op 4619) 
v_mul_f32 v32, v19, v28 // (op 4619) 
v_mul_f32 v33, v19, v29 // (op 4619) 
v_mul_f32 v34, v19, v30 // (op 4619) 
v_mul_f32 v35, v19, v31 // (op 4619) 

s_waitcnt vmcnt(0)
v_and_b32 v20, 65535, v24 // (op 4624) 
v_lshrrev_b32 v21, 16, v24 // (op 4624) 
v_lshlrev_b32 v20, 16, v20 // (op 4624) 
v_lshlrev_b32 v21, 16, v21 // (op 4624) 
v_mul_f32 v28, v18, v20 // (op 4624) 
v_mul_f32 v29, v18, v21 // (op 4624) 
v_and_b32 v20, 65535, v25 // (op 4624) 
v_lshrrev_b32 v21, 16, v25 // (op 4624) 
v_lshlrev_b32 v20, 16, v20 // (op 4624) 
v_lshlrev_b32 v21, 16, v21 // (op 4624) 
v_mul_f32 v30, v18, v20 // (op 4624) 
v_mul_f32 v31, v18, v21 // (op 4624) 


v_add_f32 v36, v28, v32 // (op 4620) 
v_add_f32 v37, v29, v33 // (op 4620) 
v_add_f32 v38, v30, v34 // (op 4620) 
v_add_f32 v39, v31, v35 // (op 4620) 


v_lshrrev_b32 v28, 16, v36 // (op 4621) 
v_lshrrev_b32 v29, 16, v37 // (op 4621) 
v_lshrrev_b32 v30, 16, v38 // (op 4621) 
v_lshrrev_b32 v31, 16, v39 // (op 4621) 


v_pack_b32_f16 v20, v28, v29 // (op 4622) 
v_pack_b32_f16 v21, v30, v31 // (op 4622) 

s_lshl_b32 s52, s24, 6 // (op 4622) 
s_mul_i32 s53, 3, s52 // (op 4622) 

v_add_u32 v23, v27, s53 // (op 4622) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 4622) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalBottom_0_beta_135:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_381 // (op 381) Condition: Done, jump to RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_381
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_381:
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_381:


s_mul_i32 s52, s8, s16 // (op 294) 
s_cmp_gt_u32 s9, s52 // (op 294) 
s_mov_b32 s53, scc // (op 294) 
s_sub_u32 s54, s9, s52 // (op 294) 

s_cmp_eq_i32 s53, 0 // (op 294) 
s_cselect_b32 s52, 0, s54 // (op 294) 


s_cmp_lt_u32 s8, s52 // (op 294) 
s_mov_b32 s53, scc // (op 294) 
s_cmp_eq_i32 s53, 0 // (op 294) 
s_cselect_b32 s54, s52, s8 // (op 294) 


s_cmp_lt_u32 s0, s54 // (op 294) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopTop_SKStreamTileLoop_294 // (op 294) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopTop_SKStreamTileLoop_294 if true)
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopBottom_SKStreamTileLoop_294:

















s_mov_b32 s0, 0 // (op 435) call()
s_mov_b32 s1, 0 // (op 436) call()
s_mov_b32 s0, 0 // (op 450) call()
s_mul_i32 s15, s50, s16 // (op 449) 
s_cmp_gt_u32 s51, s15 // (op 449) 
s_mov_b32 s6, scc // (op 449) 
s_sub_u32 s7, s51, s15 // (op 449) 

s_cmp_eq_i32 s6, 0 // (op 449) 
s_cselect_b32 s8, 0, s7 // (op 449) 


s_cmp_lt_u32 s50, s8 // (op 449) 
s_mov_b32 s15, scc // (op 449) 
s_cmp_eq_i32 s15, 0 // (op 449) 
s_cselect_b32 s6, s8, s50 // (op 449) 


s_cmp_lt_u32 s0, s6 // (op 449) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopBottom_DPStreamTileLoop_449 // (op 449) Condition: Top (jump to RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopBottom_DPStreamTileLoop_449 if false)
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopTop_DPStreamTileLoop_449:
v_lshrrev_b32 v0, 7, v1 // (op 9460) 
s_mul_i32 s15, s16, s50 // (op 9460) 
s_add_u32 s15, s15, s0 // (op 9460) 
v_and_b32 v3, 63, v1 // (op 9460) 
s_add_u32 s6, s15, s9 // (op 9460) 

v_lshrrev_b32 v4, 4, v3 // (op 9460) 
s_mul_hi_u32 s15, s6, s13 // (op 9460) 
v_lshlrev_b32 v5, 2, v4 // (op 9460) 

s_sub_u32 s7, s6, s15 // (op 9460) 
v_lshl_add_u32 v4, v0, 4, v5 // (op 9460) 


s_lshr_b32 s8, s7, 1 // (op 9460) 

v_lshrrev_b32 v0, 2, v3 // (op 9460) 

s_add_u32 s7, s8, s15 // (op 9460) 


v_and_b32 v3, 3, v0 // (op 9460) 

s_lshr_b32 s8, s7, s11 // (op 9460) 

v_and_b32 v0, 3, v1 // (op 9460) 
s_and_b32 s7, s11, 2147483648 // (op 9460) 
v_lshl_add_u32 v5, v3, 2, v0 // (op 9460) 


s_cmp_eq_i32 s7, 0 // (op 9460) 
s_mov_b32 s15, scc // (op 9460) 

v_lshrrev_b32 v0, 6, v1 // (op 9460) 
s_cmp_eq_i32 s15, 0 // (op 9460) 
s_cselect_b32 s7, s6, s8 // (op 9460) 



v_and_b32 v3, 1, v0 // (op 9460) 

s_mul_hi_u32 s6, s7, s12 // (op 9460) 
v_lshl_add_u32 v0, v3, 4, v5 // (op 9460) 


s_sub_u32 s8, s7, s6 // (op 9460) 
s_lshr_b32 s15, s8, 1 // (op 9460) 

s_add_u32 s8, s15, s6 // (op 9460) 


s_lshr_b32 s6, s8, s10 // (op 9460) 

s_and_b32 s8, s10, 2147483648 // (op 9460) 
s_cmp_eq_i32 s8, 0 // (op 9460) 
s_mov_b32 s15, scc // (op 9460) 

s_cmp_eq_i32 s15, 0 // (op 9460) 
s_cselect_b32 s8, s7, s6 // (op 9460) 


s_mul_i32 s6, s8, s14 // (op 9460) 
v_lshl_add_u32 v3, s8, 6, v4 // (op 9460) 


s_sub_u32 s8, s7, s6 // (op 9460) 


v_lshl_add_u32 v4, s8, 7, v0 // (op 9460) 


v_mul_lo_u32 v0, v4, s4 // (op 9460) 

v_add_lshl_u32 v4, v3, v0, 1 // (op 9460) 


s_mul_i32 s19, s4, s23 // (op 9462) most significant: low of low * high
s_mul_i32 s15, s5, s22 // (op 9462) most significant: low of high * low
s_mul_i32 s18, s4, s22 // (op 9462) least significant
s_mul_hi_u32 s6, s4, s22 // (op 9462) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 9462) most significant: sum
s_add_u32 s19, s19, s6 // (op 9462) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 1 // (op 9462) 

s_mov_b32 s56, s20 // (op 9462) reg expression
s_mov_b32 s57, s21 // (op 9462) reg expression
s_mov_b32 s58, s28 // (op 9462) reg expression
s_mov_b32 s59, 131072 // (op 9462) call()

v_lshrrev_b32 v0, 7, v1 // (op 9814) 
s_mul_i32 s15, s16, s50 // (op 9814) 
s_add_u32 s15, s15, s0 // (op 9814) 
v_and_b32 v3, 63, v1 // (op 9814) 
s_add_u32 s6, s15, s9 // (op 9814) 

v_lshrrev_b32 v5, 4, v3 // (op 9814) 
s_mul_hi_u32 s15, s6, s13 // (op 9814) 
v_lshlrev_b32 v14, 2, v5 // (op 9814) 

s_sub_u32 s7, s6, s15 // (op 9814) 
v_lshl_add_u32 v5, v0, 4, v14 // (op 9814) 


s_lshr_b32 s8, s7, 1 // (op 9814) 

v_lshrrev_b32 v0, 2, v3 // (op 9814) 

s_add_u32 s7, s8, s15 // (op 9814) 


v_and_b32 v3, 3, v0 // (op 9814) 

s_lshr_b32 s8, s7, s11 // (op 9814) 

v_and_b32 v0, 3, v1 // (op 9814) 
s_and_b32 s7, s11, 2147483648 // (op 9814) 
v_lshl_add_u32 v14, v3, 2, v0 // (op 9814) 


s_cmp_eq_i32 s7, 0 // (op 9814) 
s_mov_b32 s15, scc // (op 9814) 

v_lshrrev_b32 v0, 6, v1 // (op 9814) 
s_cmp_eq_i32 s15, 0 // (op 9814) 
s_cselect_b32 s7, s6, s8 // (op 9814) 



v_and_b32 v3, 1, v0 // (op 9814) 

s_mul_hi_u32 s6, s7, s12 // (op 9814) 
v_lshl_add_u32 v0, v3, 4, v14 // (op 9814) 


s_sub_u32 s8, s7, s6 // (op 9814) 
s_lshr_b32 s15, s8, 1 // (op 9814) 

s_add_u32 s8, s15, s6 // (op 9814) 


s_lshr_b32 s6, s8, s10 // (op 9814) 

s_and_b32 s8, s10, 2147483648 // (op 9814) 
s_cmp_eq_i32 s8, 0 // (op 9814) 
s_mov_b32 s15, scc // (op 9814) 

s_cmp_eq_i32 s15, 0 // (op 9814) 
s_cselect_b32 s8, s7, s6 // (op 9814) 


s_mul_i32 s6, s8, s14 // (op 9814) 
v_lshl_add_u32 v3, s8, 6, v5 // (op 9814) 


s_sub_u32 s8, s7, s6 // (op 9814) 


v_lshl_add_u32 v5, s8, 7, v0 // (op 9814) 


v_mul_lo_u32 v0, v5, s24 // (op 9814) 

v_add_lshl_u32 v5, v3, v0, 1 // (op 9814) 


s_mul_i32 s19, s24, s23 // (op 9816) most significant: low of low * high
s_mul_i32 s15, s25, s22 // (op 9816) most significant: low of high * low
s_mul_i32 s18, s24, s22 // (op 9816) least significant
s_mul_hi_u32 s6, s24, s22 // (op 9816) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 9816) most significant: sum
s_add_u32 s19, s19, s6 // (op 9816) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 1 // (op 9816) 

s_mov_b32 s60, s26 // (op 9816) reg expression
s_mov_b32 s61, s27 // (op 9816) reg expression
s_mov_b32 s62, s28 // (op 9816) reg expression
s_mov_b32 s63, 131072 // (op 9816) call()

v_lshrrev_b32 v0, 7, v1 // (op 9893) 
v_and_b32 v3, 15, v1 // (op 9893) 
v_lshl_add_u32 v14, v0, 4, v3 // (op 9893) 


v_and_b32 v0, 48, v1 // (op 9893) 
v_lshl_add_u32 v3, v14, 7, v0 // (op 9893) 


v_lshrrev_b32 v0, 6, v1 // (op 9968) 
v_and_b32 v14, 1, v0 // (op 9968) 

v_and_b32 v0, 15, v1 // (op 9968) 
v_lshl_add_u32 v15, v14, 4, v0 // (op 9968) 


v_and_b32 v0, 48, v1 // (op 9968) 
v_lshl_add_u32 v14, v15, 7, v0 // (op 9968) 


v_lshrrev_b32 v0, 7, v1 // (op 10044) 
v_and_b32 v15, 15, v1 // (op 10044) 
v_lshl_add_u32 v16, v0, 4, v15 // (op 10044) 


v_and_b32 v0, 63, v1 // (op 10044) 
v_lshrrev_b32 v15, 4, v0 // (op 10044) 

v_lshl_add_u32 v0, v16, 2, v15 // (op 10044) 


v_lshrrev_b32 v15, 6, v1 // (op 10120) 
v_and_b32 v16, 1, v15 // (op 10120) 

v_and_b32 v15, 15, v1 // (op 10120) 
v_lshl_add_u32 v17, v16, 4, v15 // (op 10120) 


v_and_b32 v15, 63, v1 // (op 10120) 
v_lshrrev_b32 v16, 4, v15 // (op 10120) 

v_lshl_add_u32 v15, v17, 2, v16 // (op 10120) 


v_accvgpr_write a0, 0.00000 // (op 184) call()
v_accvgpr_write a1, 0.00000 // (op 184) call()
v_accvgpr_write a2, 0.00000 // (op 184) call()
v_accvgpr_write a3, 0.00000 // (op 184) call()
v_accvgpr_write a4, 0.00000 // (op 622) call()
v_accvgpr_write a5, 0.00000 // (op 622) call()
v_accvgpr_write a6, 0.00000 // (op 622) call()
v_accvgpr_write a7, 0.00000 // (op 622) call()
v_accvgpr_write a8, 0.00000 // (op 798) call()
v_accvgpr_write a9, 0.00000 // (op 798) call()
v_accvgpr_write a10, 0.00000 // (op 798) call()
v_accvgpr_write a11, 0.00000 // (op 798) call()
v_accvgpr_write a12, 0.00000 // (op 974) call()
v_accvgpr_write a13, 0.00000 // (op 974) call()
v_accvgpr_write a14, 0.00000 // (op 974) call()
v_accvgpr_write a15, 0.00000 // (op 974) call()
v_accvgpr_write a16, 0.00000 // (op 1238) call()
v_accvgpr_write a17, 0.00000 // (op 1238) call()
v_accvgpr_write a18, 0.00000 // (op 1238) call()
v_accvgpr_write a19, 0.00000 // (op 1238) call()
v_accvgpr_write a20, 0.00000 // (op 1547) call()
v_accvgpr_write a21, 0.00000 // (op 1547) call()
v_accvgpr_write a22, 0.00000 // (op 1547) call()
v_accvgpr_write a23, 0.00000 // (op 1547) call()
v_accvgpr_write a24, 0.00000 // (op 1548) call()
v_accvgpr_write a25, 0.00000 // (op 1548) call()
v_accvgpr_write a26, 0.00000 // (op 1548) call()
v_accvgpr_write a27, 0.00000 // (op 1548) call()
v_accvgpr_write a28, 0.00000 // (op 1549) call()
v_accvgpr_write a29, 0.00000 // (op 1549) call()
v_accvgpr_write a30, 0.00000 // (op 1549) call()
v_accvgpr_write a31, 0.00000 // (op 1549) call()
v_lshrrev_b32 v16, 3, v1 // (op 8948) 
s_mul_i32 s15, s16, s50 // (op 8948) 
s_add_u32 s15, s15, s0 // (op 8948) 
v_and_b32 v17, -8, v1 // (op 8948) 
s_add_u32 s6, s15, s9 // (op 8948) 

v_sub_u32 v20, v1, v17 // (op 8948) 

s_mul_hi_u32 s15, s6, s13 // (op 8948) 
v_lshlrev_b32 v17, 4, v20 // (op 8948) 

s_sub_u32 s7, s6, s15 // (op 8948) 
s_lshr_b32 s8, s7, 1 // (op 8948) 

s_add_u32 s7, s8, s15 // (op 8948) 


s_lshr_b32 s8, s7, s11 // (op 8948) 

s_and_b32 s7, s11, 2147483648 // (op 8948) 
s_cmp_eq_i32 s7, 0 // (op 8948) 
s_mov_b32 s15, scc // (op 8948) 

s_cmp_eq_i32 s15, 0 // (op 8948) 
s_cselect_b32 s7, s6, s8 // (op 8948) 


s_mul_hi_u32 s8, s7, s12 // (op 8948) 
s_sub_u32 s15, s7, s8 // (op 8948) 
s_lshr_b32 s18, s15, 1 // (op 8948) 

s_add_u32 s15, s18, s8 // (op 8948) 


s_lshr_b32 s8, s15, s10 // (op 8948) 

s_and_b32 s15, s10, 2147483648 // (op 8948) 
s_cmp_eq_i32 s15, 0 // (op 8948) 
s_mov_b32 s18, scc // (op 8948) 

s_cmp_eq_i32 s18, 0 // (op 8948) 
s_cselect_b32 s15, s7, s8 // (op 8948) 


v_lshl_add_u32 v20, s15, 6, v16 // (op 8948) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 8948) 
s_mul_i32 s8, s7, s18 // (op 8948) 


s_sub_u32 s7, s6, s8 // (op 8948) 


v_lshl_add_u32 v16, s7, 7, v17 // (op 8948) 


v_mov_b32 v23, 0 // (op 8948) convert
v_mov_b32 v22, v16 // (op 8948) convert

v_mul_lo_u32 v17, v20, s31 // (op 8948) most significant: low of low * high
v_mul_hi_u32 v21, v20, s30 // (op 8948) most significant: high of low * low
v_mul_lo_u32 v16, v20, s30 // (op 8948) least significant: low of low * low
v_add_u32 v17, v17, v21 // (op 8948) most significant: sum

v_add_co_u32 v16, s[18:19], v16, v22 // (op 8948) least significant half
v_addc_co_u32 v17, s[18:19], v17, v23, s[18:19] // (op 8948) most significant half



s_mul_i32 s19, s30, s35 // (op 8950) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 8950) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 8950) least significant
s_mul_hi_u32 s6, s30, s34 // (op 8950) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 8950) most significant: sum
s_add_u32 s19, s19, s6 // (op 8950) most significant: sum


s_mov_b32 s68, s32 // (op 8950) reg expression
s_mov_b32 s69, s33 // (op 8950) reg expression
s_mov_b32 s70, s18 // (op 8950) reg expression
s_mov_b32 s71, 131072 // (op 8950) call()

v_lshrrev_b32 v20, 3, v1 // (op 8967) 
s_mul_i32 s15, s16, s50 // (op 8967) 
s_add_u32 s15, s15, s0 // (op 8967) 
v_and_b32 v21, -8, v1 // (op 8967) 
s_add_u32 s6, s15, s9 // (op 8967) 

v_sub_u32 v22, v1, v21 // (op 8967) 

s_mul_hi_u32 s15, s6, s13 // (op 8967) 
v_lshlrev_b32 v21, 4, v22 // (op 8967) 

s_sub_u32 s7, s6, s15 // (op 8967) 
s_lshr_b32 s8, s7, 1 // (op 8967) 

s_add_u32 s7, s8, s15 // (op 8967) 


s_lshr_b32 s8, s7, s11 // (op 8967) 

s_and_b32 s7, s11, 2147483648 // (op 8967) 
s_cmp_eq_i32 s7, 0 // (op 8967) 
s_mov_b32 s15, scc // (op 8967) 

s_cmp_eq_i32 s15, 0 // (op 8967) 
s_cselect_b32 s7, s6, s8 // (op 8967) 


s_mul_hi_u32 s8, s7, s12 // (op 8967) 
s_sub_u32 s15, s7, s8 // (op 8967) 
s_lshr_b32 s18, s15, 1 // (op 8967) 

s_add_u32 s15, s18, s8 // (op 8967) 


s_lshr_b32 s8, s15, s10 // (op 8967) 

s_and_b32 s15, s10, 2147483648 // (op 8967) 
s_cmp_eq_i32 s15, 0 // (op 8967) 
s_mov_b32 s18, scc // (op 8967) 

s_cmp_eq_i32 s18, 0 // (op 8967) 
s_cselect_b32 s15, s7, s8 // (op 8967) 


v_lshl_add_u32 v22, s15, 6, v20 // (op 8967) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 8967) 
s_mul_i32 s8, s7, s18 // (op 8967) 


s_sub_u32 s7, s6, s8 // (op 8967) 


v_lshl_add_u32 v20, s7, 7, v21 // (op 8967) 


v_mov_b32 v25, 0 // (op 8967) convert
v_mov_b32 v24, v20 // (op 8967) convert

v_mul_lo_u32 v21, v22, s31 // (op 8967) most significant: low of low * high
v_mul_hi_u32 v23, v22, s30 // (op 8967) most significant: high of low * low
v_mul_lo_u32 v20, v22, s30 // (op 8967) least significant: low of low * low
v_add_u32 v21, v21, v23 // (op 8967) most significant: sum

v_add_co_u32 v20, s[18:19], v20, v24 // (op 8967) least significant half
v_addc_co_u32 v21, s[18:19], v21, v25, s[18:19] // (op 8967) most significant half



s_mul_i32 s19, s30, s35 // (op 8969) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 8969) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 8969) least significant
s_mul_hi_u32 s6, s30, s34 // (op 8969) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 8969) most significant: sum
s_add_u32 s19, s19, s6 // (op 8969) most significant: sum


s_mov_b32 s68, s32 // (op 8969) reg expression
s_mov_b32 s69, s33 // (op 8969) reg expression
s_mov_b32 s70, s18 // (op 8969) reg expression
s_mov_b32 s71, 131072 // (op 8969) call()

v_lshrrev_b32 v22, 3, v1 // (op 9086) 
s_mul_i32 s15, s16, s50 // (op 9086) 
s_add_u32 s15, s15, s0 // (op 9086) 
v_and_b32 v23, -8, v1 // (op 9086) 
s_add_u32 s6, s15, s9 // (op 9086) 

v_sub_u32 v24, v1, v23 // (op 9086) 

s_mul_hi_u32 s15, s6, s13 // (op 9086) 
v_lshlrev_b32 v23, 4, v24 // (op 9086) 

s_sub_u32 s7, s6, s15 // (op 9086) 
s_lshr_b32 s8, s7, 1 // (op 9086) 

s_add_u32 s7, s8, s15 // (op 9086) 


s_lshr_b32 s8, s7, s11 // (op 9086) 

s_and_b32 s7, s11, 2147483648 // (op 9086) 
s_cmp_eq_i32 s7, 0 // (op 9086) 
s_mov_b32 s15, scc // (op 9086) 

s_cmp_eq_i32 s15, 0 // (op 9086) 
s_cselect_b32 s7, s6, s8 // (op 9086) 


s_mul_hi_u32 s8, s7, s12 // (op 9086) 
s_sub_u32 s15, s7, s8 // (op 9086) 
s_lshr_b32 s18, s15, 1 // (op 9086) 

s_add_u32 s15, s18, s8 // (op 9086) 


s_lshr_b32 s8, s15, s10 // (op 9086) 

s_and_b32 s15, s10, 2147483648 // (op 9086) 
s_cmp_eq_i32 s15, 0 // (op 9086) 
s_mov_b32 s18, scc // (op 9086) 

s_cmp_eq_i32 s18, 0 // (op 9086) 
s_cselect_b32 s15, s7, s8 // (op 9086) 


s_mul_i32 s8, s15, s14 // (op 9086) 

s_sub_u32 s15, s7, s8 // (op 9086) 

v_lshl_add_u32 v24, s15, 7, v22 // (op 9086) 


v_mul_lo_u32 v27, v24, s37 // (op 9086) most significant: low of low * high
v_mul_hi_u32 v22, v24, s36 // (op 9086) most significant: high of low * low
v_mul_lo_u32 v26, v24, s36 // (op 9086) least significant: low of low * low
v_add_u32 v27, v27, v22 // (op 9086) most significant: sum


s_ashr_i64 s[18:19], s[2:3], 7 // (op 9086) 
s_mul_i32 s8, s7, s18 // (op 9086) 


s_sub_u32 s7, s6, s8 // (op 9086) 


v_lshl_add_u32 v22, s7, 7, v23 // (op 9086) 


v_mov_b32 v25, 0 // (op 9086) convert
v_mov_b32 v24, v22 // (op 9086) convert

v_add_co_u32 v22, s[18:19], v24, v26 // (op 9086) least significant half
v_addc_co_u32 v23, s[18:19], v25, v27, s[18:19] // (op 9086) most significant half



s_mul_i32 s19, s36, s23 // (op 9088) most significant: low of low * high
s_mul_i32 s15, s37, s22 // (op 9088) most significant: low of high * low
s_mul_i32 s18, s36, s22 // (op 9088) least significant
s_mul_hi_u32 s6, s36, s22 // (op 9088) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 9088) most significant: sum
s_add_u32 s19, s19, s6 // (op 9088) most significant: sum


s_mov_b32 s72, s38 // (op 9088) reg expression
s_mov_b32 s73, s39 // (op 9088) reg expression
s_mov_b32 s74, s18 // (op 9088) reg expression
s_mov_b32 s75, 131072 // (op 9088) call()

v_lshrrev_b32 v24, 3, v1 // (op 9105) 
s_mul_i32 s15, s16, s50 // (op 9105) 
s_add_u32 s15, s15, s0 // (op 9105) 
v_and_b32 v25, -8, v1 // (op 9105) 
s_add_u32 s6, s15, s9 // (op 9105) 

v_sub_u32 v26, v1, v25 // (op 9105) 

s_mul_hi_u32 s15, s6, s13 // (op 9105) 
v_lshlrev_b32 v25, 4, v26 // (op 9105) 

s_sub_u32 s7, s6, s15 // (op 9105) 
s_lshr_b32 s8, s7, 1 // (op 9105) 

s_add_u32 s7, s8, s15 // (op 9105) 


s_lshr_b32 s8, s7, s11 // (op 9105) 

s_and_b32 s7, s11, 2147483648 // (op 9105) 
s_cmp_eq_i32 s7, 0 // (op 9105) 
s_mov_b32 s15, scc // (op 9105) 

s_cmp_eq_i32 s15, 0 // (op 9105) 
s_cselect_b32 s7, s6, s8 // (op 9105) 


s_mul_hi_u32 s8, s7, s12 // (op 9105) 
s_sub_u32 s15, s7, s8 // (op 9105) 
s_lshr_b32 s18, s15, 1 // (op 9105) 

s_add_u32 s15, s18, s8 // (op 9105) 


s_lshr_b32 s8, s15, s10 // (op 9105) 

s_and_b32 s15, s10, 2147483648 // (op 9105) 
s_cmp_eq_i32 s15, 0 // (op 9105) 
s_mov_b32 s18, scc // (op 9105) 

s_cmp_eq_i32 s18, 0 // (op 9105) 
s_cselect_b32 s15, s7, s8 // (op 9105) 


s_mul_i32 s8, s15, s14 // (op 9105) 

s_sub_u32 s15, s7, s8 // (op 9105) 

v_lshl_add_u32 v26, s15, 7, v24 // (op 9105) 


v_mul_lo_u32 v29, v26, s37 // (op 9105) most significant: low of low * high
v_mul_hi_u32 v24, v26, s36 // (op 9105) most significant: high of low * low
v_mul_lo_u32 v28, v26, s36 // (op 9105) least significant: low of low * low
v_add_u32 v29, v29, v24 // (op 9105) most significant: sum


s_ashr_i64 s[18:19], s[2:3], 7 // (op 9105) 
s_mul_i32 s8, s7, s18 // (op 9105) 


s_sub_u32 s7, s6, s8 // (op 9105) 


v_lshl_add_u32 v24, s7, 7, v25 // (op 9105) 


v_mov_b32 v27, 0 // (op 9105) convert
v_mov_b32 v26, v24 // (op 9105) convert

v_add_co_u32 v24, s[18:19], v26, v28 // (op 9105) least significant half
v_addc_co_u32 v25, s[18:19], v27, v29, s[18:19] // (op 9105) most significant half



s_mul_i32 s19, s36, s23 // (op 9107) most significant: low of low * high
s_mul_i32 s15, s37, s22 // (op 9107) most significant: low of high * low
s_mul_i32 s18, s36, s22 // (op 9107) least significant
s_mul_hi_u32 s6, s36, s22 // (op 9107) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 9107) most significant: sum
s_add_u32 s19, s19, s6 // (op 9107) most significant: sum


s_mov_b32 s72, s38 // (op 9107) reg expression
s_mov_b32 s73, s39 // (op 9107) reg expression
s_mov_b32 s74, s18 // (op 9107) reg expression
s_mov_b32 s75, 131072 // (op 9107) call()

v_lshrrev_b32 v26, 2, v1 // (op 9214) 
s_mul_i32 s15, s16, s50 // (op 9214) 
s_add_u32 s15, s15, s0 // (op 9214) 
v_and_b32 v27, -4, v1 // (op 9214) 
s_add_u32 s6, s15, s9 // (op 9214) 

v_sub_u32 v28, v1, v27 // (op 9214) 

s_mul_hi_u32 s15, s6, s13 // (op 9214) 
s_sub_u32 s7, s6, s15 // (op 9214) 
s_lshr_b32 s8, s7, 1 // (op 9214) 

s_add_u32 s7, s8, s15 // (op 9214) 


s_lshr_b32 s8, s7, s11 // (op 9214) 

s_and_b32 s7, s11, 2147483648 // (op 9214) 
s_cmp_eq_i32 s7, 0 // (op 9214) 
s_mov_b32 s15, scc // (op 9214) 

s_cmp_eq_i32 s15, 0 // (op 9214) 
s_cselect_b32 s7, s6, s8 // (op 9214) 


s_mul_hi_u32 s8, s7, s12 // (op 9214) 
s_sub_u32 s15, s7, s8 // (op 9214) 
s_lshr_b32 s18, s15, 1 // (op 9214) 

s_add_u32 s15, s18, s8 // (op 9214) 


s_lshr_b32 s8, s15, s10 // (op 9214) 

s_and_b32 s15, s10, 2147483648 // (op 9214) 
s_cmp_eq_i32 s15, 0 // (op 9214) 
s_mov_b32 s18, scc // (op 9214) 

s_cmp_eq_i32 s18, 0 // (op 9214) 
s_cselect_b32 s15, s7, s8 // (op 9214) 


v_lshl_add_u32 v27, s15, 6, v26 // (op 9214) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 9214) 
s_mul_i32 s8, s7, s18 // (op 9214) 


s_sub_u32 s7, s6, s8 // (op 9214) 


v_lshl_add_u32 v26, s7, 2, v28 // (op 9214) 


v_mul_lo_u32 v28, v27, s40 // (op 9214) 
v_add_u32 v28, v28, v26 // (op 9214) 


s_mul_i32 s19, s40, s35 // (op 9216) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 9216) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 9216) least significant
s_mul_hi_u32 s6, s40, s34 // (op 9216) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 9216) most significant: sum
s_add_u32 s19, s19, s6 // (op 9216) most significant: sum


s_mov_b32 s76, s42 // (op 9216) reg expression
s_mov_b32 s77, s43 // (op 9216) reg expression
s_mov_b32 s78, s18 // (op 9216) reg expression
s_mov_b32 s79, 131072 // (op 9216) call()

v_lshrrev_b32 v26, 2, v1 // (op 9233) 
s_mul_i32 s15, s16, s50 // (op 9233) 
s_add_u32 s15, s15, s0 // (op 9233) 
v_and_b32 v27, -4, v1 // (op 9233) 
s_add_u32 s6, s15, s9 // (op 9233) 

v_sub_u32 v29, v1, v27 // (op 9233) 

s_mul_hi_u32 s15, s6, s13 // (op 9233) 
s_sub_u32 s7, s6, s15 // (op 9233) 
s_lshr_b32 s8, s7, 1 // (op 9233) 

s_add_u32 s7, s8, s15 // (op 9233) 


s_lshr_b32 s8, s7, s11 // (op 9233) 

s_and_b32 s7, s11, 2147483648 // (op 9233) 
s_cmp_eq_i32 s7, 0 // (op 9233) 
s_mov_b32 s15, scc // (op 9233) 

s_cmp_eq_i32 s15, 0 // (op 9233) 
s_cselect_b32 s7, s6, s8 // (op 9233) 


s_mul_hi_u32 s8, s7, s12 // (op 9233) 
s_sub_u32 s15, s7, s8 // (op 9233) 
s_lshr_b32 s18, s15, 1 // (op 9233) 

s_add_u32 s15, s18, s8 // (op 9233) 


s_lshr_b32 s8, s15, s10 // (op 9233) 

s_and_b32 s15, s10, 2147483648 // (op 9233) 
s_cmp_eq_i32 s15, 0 // (op 9233) 
s_mov_b32 s18, scc // (op 9233) 

s_cmp_eq_i32 s18, 0 // (op 9233) 
s_cselect_b32 s15, s7, s8 // (op 9233) 


v_lshl_add_u32 v27, s15, 6, v26 // (op 9233) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 9233) 
s_mul_i32 s8, s7, s18 // (op 9233) 


s_sub_u32 s7, s6, s8 // (op 9233) 


v_lshl_add_u32 v26, s7, 2, v29 // (op 9233) 


v_mul_lo_u32 v29, v27, s40 // (op 9233) 
v_add_u32 v29, v29, v26 // (op 9233) 


s_mul_i32 s19, s40, s35 // (op 9235) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 9235) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 9235) least significant
s_mul_hi_u32 s6, s40, s34 // (op 9235) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 9235) most significant: sum
s_add_u32 s19, s19, s6 // (op 9235) most significant: sum


s_mov_b32 s76, s42 // (op 9235) reg expression
s_mov_b32 s77, s43 // (op 9235) reg expression
s_mov_b32 s78, s18 // (op 9235) reg expression
s_mov_b32 s79, 131072 // (op 9235) call()

v_lshrrev_b32 v26, 2, v1 // (op 9342) 
s_mul_i32 s15, s16, s50 // (op 9342) 
s_add_u32 s15, s15, s0 // (op 9342) 
v_lshlrev_b32 v27, 1, v26 // (op 9342) 

s_add_u32 s6, s15, s9 // (op 9342) 

v_and_b32 v26, -4, v1 // (op 9342) 
s_mul_hi_u32 s15, s6, s13 // (op 9342) 
v_sub_u32 v30, v1, v26 // (op 9342) 

s_sub_u32 s7, s6, s15 // (op 9342) 
s_lshr_b32 s8, s7, 1 // (op 9342) 

s_add_u32 s7, s8, s15 // (op 9342) 


s_lshr_b32 s8, s7, s11 // (op 9342) 

s_and_b32 s7, s11, 2147483648 // (op 9342) 
s_cmp_eq_i32 s7, 0 // (op 9342) 
s_mov_b32 s15, scc // (op 9342) 

s_cmp_eq_i32 s15, 0 // (op 9342) 
s_cselect_b32 s7, s6, s8 // (op 9342) 


s_mul_hi_u32 s8, s7, s12 // (op 9342) 
s_sub_u32 s15, s7, s8 // (op 9342) 
s_lshr_b32 s18, s15, 1 // (op 9342) 

s_add_u32 s15, s18, s8 // (op 9342) 


s_lshr_b32 s8, s15, s10 // (op 9342) 

s_and_b32 s15, s10, 2147483648 // (op 9342) 
s_cmp_eq_i32 s15, 0 // (op 9342) 
s_mov_b32 s18, scc // (op 9342) 

s_cmp_eq_i32 s18, 0 // (op 9342) 
s_cselect_b32 s15, s7, s8 // (op 9342) 


s_mul_i32 s8, s15, s14 // (op 9342) 

s_sub_u32 s15, s7, s8 // (op 9342) 

v_lshl_add_u32 v26, s15, 7, v27 // (op 9342) 


v_mul_lo_u32 v27, v26, s44 // (op 9342) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 9342) 
s_mul_i32 s8, s7, s18 // (op 9342) 


s_sub_u32 s7, s6, s8 // (op 9342) 


v_lshl_add_u32 v26, s7, 2, v30 // (op 9342) 


v_add_u32 v30, v26, v27 // (op 9342) 


s_mul_i32 s19, s44, s23 // (op 9344) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 9344) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 9344) least significant
s_mul_hi_u32 s6, s44, s22 // (op 9344) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 9344) most significant: sum
s_add_u32 s19, s19, s6 // (op 9344) most significant: sum


s_mov_b32 s80, s46 // (op 9344) reg expression
s_mov_b32 s81, s47 // (op 9344) reg expression
s_mov_b32 s82, s18 // (op 9344) reg expression
s_mov_b32 s83, 131072 // (op 9344) call()

v_lshrrev_b32 v26, 2, v1 // (op 9361) 
s_mul_i32 s15, s16, s50 // (op 9361) 
s_add_u32 s15, s15, s0 // (op 9361) 
v_lshlrev_b32 v27, 1, v26 // (op 9361) 

s_add_u32 s6, s15, s9 // (op 9361) 

v_and_b32 v26, -4, v1 // (op 9361) 
s_mul_hi_u32 s15, s6, s13 // (op 9361) 
v_sub_u32 v31, v1, v26 // (op 9361) 

s_sub_u32 s7, s6, s15 // (op 9361) 
s_lshr_b32 s8, s7, 1 // (op 9361) 

s_add_u32 s7, s8, s15 // (op 9361) 


s_lshr_b32 s8, s7, s11 // (op 9361) 

s_and_b32 s7, s11, 2147483648 // (op 9361) 
s_cmp_eq_i32 s7, 0 // (op 9361) 
s_mov_b32 s15, scc // (op 9361) 

s_cmp_eq_i32 s15, 0 // (op 9361) 
s_cselect_b32 s7, s6, s8 // (op 9361) 


s_mul_hi_u32 s8, s7, s12 // (op 9361) 
s_sub_u32 s15, s7, s8 // (op 9361) 
s_lshr_b32 s18, s15, 1 // (op 9361) 

s_add_u32 s15, s18, s8 // (op 9361) 


s_lshr_b32 s8, s15, s10 // (op 9361) 

s_and_b32 s15, s10, 2147483648 // (op 9361) 
s_cmp_eq_i32 s15, 0 // (op 9361) 
s_mov_b32 s18, scc // (op 9361) 

s_cmp_eq_i32 s18, 0 // (op 9361) 
s_cselect_b32 s15, s7, s8 // (op 9361) 


s_mul_i32 s8, s15, s14 // (op 9361) 

s_sub_u32 s15, s7, s8 // (op 9361) 

v_lshl_add_u32 v26, s15, 7, v27 // (op 9361) 


v_mul_lo_u32 v27, v26, s44 // (op 9361) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 9361) 
s_mul_i32 s8, s7, s18 // (op 9361) 


s_sub_u32 s7, s6, s8 // (op 9361) 


v_lshl_add_u32 v26, s7, 2, v31 // (op 9361) 


v_add_u32 v31, v26, v27 // (op 9361) 


s_mul_i32 s19, s44, s23 // (op 9363) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 9363) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 9363) least significant
s_mul_hi_u32 s6, s44, s22 // (op 9363) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 9363) most significant: sum
s_add_u32 s19, s19, s6 // (op 9363) most significant: sum


s_mov_b32 s80, s46 // (op 9363) reg expression
s_mov_b32 s81, s47 // (op 9363) reg expression
s_mov_b32 s82, s18 // (op 9363) reg expression
s_mov_b32 s83, 131072 // (op 9363) call()

v_and_b32 v26, -8, v1 // (op 9934) 
v_sub_u32 v27, v1, v26 // (op 9934) 

v_lshlrev_b32 v26, 4, v27 // (op 9934) 

v_lshrrev_b32 v27, 3, v1 // (op 9934) 
v_lshl_add_u32 v32, v27, 7, v26 // (op 9934) 


s_nop 0
v_readfirstlane_b32 s15, v32 // (op 9934) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 10735) 

s_lshl_b64 s[18:19], s[36:37], 5 // (op 8852) 
v_mov_b64 v[26:27], s[18:19] // (op 8852) convert


v_mov_b64 v[32:33], v[22:23] // (op 8852) 
s_mov_b32 m0, s15 // (op 8852) reg expression
buffer_load_dwordx4 v32, s[72:75], 0 offen offset: 0 lds // (op 8852) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v26 // (op 8852) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 8852) most significant half

s_add_u32 m0, m0, 4096 // (op 8852) 
buffer_load_dwordx4 v32, s[72:75], 0 offen offset: 0 lds // (op 8852) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v26 // (op 8852) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 8852) most significant half

s_add_u32 m0, m0, 4096 // (op 8852) 
buffer_load_dwordx4 v32, s[72:75], 0 offen offset: 0 lds // (op 8852) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v26 // (op 8852) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 8852) most significant half

s_add_u32 m0, m0, 4096 // (op 8852) 
buffer_load_dwordx4 v32, s[72:75], 0 offen offset: 0 lds // (op 8852) Load value direct to lds




v_mov_b64 v[26:27], s[44:45] // (op 8400) convert

v_mov_b32 v32, v30 // (op 8400) 
buffer_load_ubyte v34, v32, s[80:83], 0 offen offset: 0 // (op 8400) Load value
v_add_u32 v32, v32, v26 // (op 8400) 
buffer_load_ubyte v35, v32, s[80:83], 0 offen offset: 0 // (op 8400) Load value


v_and_b32 v26, -8, v1 // (op 9859) 
v_sub_u32 v27, v1, v26 // (op 9859) 

v_lshlrev_b32 v26, 4, v27 // (op 9859) 

v_lshrrev_b32 v27, 3, v1 // (op 9859) 
v_lshl_add_u32 v32, v27, 7, v26 // (op 9859) 


s_nop 0
v_readfirstlane_b32 s15, v32 // (op 9859) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 10716) 

s_lshl_b64 s[18:19], s[30:31], 5 // (op 8857) 
v_mov_b64 v[26:27], s[18:19] // (op 8857) convert


v_mov_b64 v[32:33], v[16:17] // (op 8857) 
s_add_u32 s6, 16384, s15 // (op 8857) 
s_mov_b32 m0, s6 // (op 8857) reg expression
buffer_load_dwordx4 v32, s[68:71], 0 offen offset: 0 lds // (op 8857) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v26 // (op 8857) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 8857) most significant half

s_add_u32 m0, m0, 4096 // (op 8857) 
buffer_load_dwordx4 v32, s[68:71], 0 offen offset: 0 lds // (op 8857) Load value direct to lds







buffer_load_ubyte v26, v28, s[76:79], 0 offen offset: 0 // (op 8406) Load value
v_and_b32 v27, -8, v1 // (op 10244) 
v_sub_u32 v32, v1, v27 // (op 10244) 

v_lshlrev_b32 v27, 4, v32 // (op 10244) 

v_lshrrev_b32 v32, 3, v1 // (op 10244) 
v_lshl_add_u32 v33, v32, 7, v27 // (op 10244) 


s_nop 0
v_readfirstlane_b32 s15, v33 // (op 10244) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 10806) 

s_lshl_b64 s[18:19], s[36:37], 5 // (op 8862) 
v_mov_b64 v[32:33], s[18:19] // (op 8862) convert


v_mov_b64 v[36:37], v[22:23] // (op 8862) 
s_add_u32 s6, 24576, s15 // (op 8862) 
s_mov_b32 m0, s6 // (op 8862) reg expression
s_mov_b32 s7, 128 // (op 8862) call()
buffer_load_dwordx4 v36, s[72:75], s7 offen offset: 0 lds // (op 8862) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v32 // (op 8862) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 8862) most significant half

s_add_u32 m0, m0, 4096 // (op 8862) 
buffer_load_dwordx4 v36, s[72:75], s7 offen offset: 0 lds // (op 8862) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v32 // (op 8862) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 8862) most significant half

s_add_u32 m0, m0, 4096 // (op 8862) 
buffer_load_dwordx4 v36, s[72:75], s7 offen offset: 0 lds // (op 8862) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v32 // (op 8862) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 8862) most significant half

s_add_u32 m0, m0, 4096 // (op 8862) 
buffer_load_dwordx4 v36, s[72:75], s7 offen offset: 0 lds // (op 8862) Load value direct to lds







v_mov_b64 v[22:23], s[44:45] // (op 8412) convert

v_mov_b32 v27, v30 // (op 8412) 
buffer_load_ubyte v32, v27, s[80:83], 0 offen offset:4 // (op 8412) Load value
v_add_u32 v27, v27, v22 // (op 8412) 
buffer_load_ubyte v33, v27, s[80:83], 0 offen offset:4 // (op 8412) Load value



v_and_b32 v27, -8, v1 // (op 10200) 
v_sub_u32 v30, v1, v27 // (op 10200) 

v_lshlrev_b32 v27, 4, v30 // (op 10200) 

v_lshrrev_b32 v30, 3, v1 // (op 10200) 
v_lshl_add_u32 v22, v30, 7, v27 // (op 10200) 


s_nop 0
v_readfirstlane_b32 s15, v22 // (op 10200) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 10796) 

s_lshl_b64 s[18:19], s[30:31], 5 // (op 8864) 
v_mov_b64 v[22:23], s[18:19] // (op 8864) convert


v_mov_b64 v[36:37], v[16:17] // (op 8864) 
s_add_u32 s6, 40960, s15 // (op 8864) 
s_mov_b32 m0, s6 // (op 8864) reg expression
s_mov_b32 s7, 128 // (op 8864) call()
buffer_load_dwordx4 v36, s[68:71], s7 offen offset: 0 lds // (op 8864) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v22 // (op 8864) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 8864) most significant half

s_add_u32 m0, m0, 4096 // (op 8864) 
buffer_load_dwordx4 v36, s[68:71], s7 offen offset: 0 lds // (op 8864) Load value direct to lds









buffer_load_ubyte v27, v28, s[76:79], 0 offen offset:4 // (op 8418) Load value

v_and_b32 v28, -4, v1 // (op 10086) 
v_sub_u32 v30, v1, v28 // (op 10086) 

v_lshrrev_b32 v28, 2, v1 // (op 10086) 
v_lshl_add_u32 v16, v28, 3, v30 // (op 10086) 


s_barrier  // (op 10776) 



s_waitcnt vmcnt(13)
ds_write_b8 v16, v34 offset:49152 // (op 8421) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v16, v35 offset:49156 // (op 8421) Store local data 

v_and_b32 v28, -4, v1 // (op 10010) 
v_sub_u32 v30, v1, v28 // (op 10010) 

v_and_b32 v28, 4294967292, v1 // (op 10010) 
v_add_u32 v16, v28, v30 // (op 10010) 


s_barrier  // (op 10755) 



s_waitcnt vmcnt(9)
ds_write_b8 v16, v26 offset:49664 // (op 8423) Store local data 

s_barrier  // (op 10732) 



ds_read_b128 v[36:39], v14 // (op 8448) Load local data 
ds_read_b128 v[40:43], v14 offset:64 // (op 8448) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 10773) 



ds_read_u8 v28, v15 offset:49152 // (op 8478) Load local data 



ds_read_b128 v[44:47], v14 offset:4096 // (op 8453) Load local data 
ds_read_b128 v[48:51], v14 offset:4160 // (op 8453) Load local data 



ds_read_u8 v30, v15 offset:49280 // (op 8483) Load local data 



ds_read_b128 v[52:55], v14 offset:8192 // (op 8458) Load local data 
ds_read_b128 v[56:59], v14 offset:8256 // (op 8458) Load local data 



ds_read_u8 v16, v15 offset:49408 // (op 8488) Load local data 



ds_read_b128 v[60:63], v14 offset:12288 // (op 8463) Load local data 
ds_read_b128 v[64:67], v14 offset:12352 // (op 8463) Load local data 



ds_read_u8 v17, v15 offset:49536 // (op 8493) Load local data 
s_barrier  // (op 10713) 



ds_read_b128 v[68:71], v3 offset:16384 // (op 8438) Load local data 
ds_read_b128 v[72:75], v3 offset:16448 // (op 8438) Load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 10752) 



ds_read_u8 v22, v0 offset:49664 // (op 8468) Load local data 



ds_read_b128 v[76:79], v3 offset:20480 // (op 8443) Load local data 
ds_read_b128 v[80:83], v3 offset:20544 // (op 8443) Load local data 



ds_read_u8 v23, v0 offset:49792 // (op 8473) Load local data 
v_and_b32 v84, -8, v1 // (op 9831) 
v_sub_u32 v85, v1, v84 // (op 9831) 

v_lshlrev_b32 v84, 4, v85 // (op 9831) 

v_lshrrev_b32 v85, 3, v1 // (op 9831) 
v_lshl_add_u32 v86, v85, 7, v84 // (op 9831) 


s_nop 0
v_readfirstlane_b32 s15, v86 // (op 9831) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v84, -8, v1 // (op 9906) 
v_sub_u32 v85, v1, v84 // (op 9906) 

v_lshlrev_b32 v84, 4, v85 // (op 9906) 

v_lshrrev_b32 v85, 3, v1 // (op 9906) 
v_lshl_add_u32 v86, v85, 7, v84 // (op 9906) 


s_nop 0
v_readfirstlane_b32 s6, v86 // (op 9906) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v84, -4, v1 // (op 9981) 
v_sub_u32 v85, v1, v84 // (op 9981) 

v_and_b32 v84, 4294967292, v1 // (op 9981) 
v_add_u32 v86, v84, v85 // (op 9981) 


v_and_b32 v84, -4, v1 // (op 10057) 
v_sub_u32 v85, v1, v84 // (op 10057) 

v_lshrrev_b32 v84, 2, v1 // (op 10057) 
v_lshl_add_u32 v87, v84, 3, v85 // (op 10057) 


v_and_b32 v84, -8, v1 // (op 10185) 
v_sub_u32 v85, v1, v84 // (op 10185) 

v_lshlrev_b32 v84, 4, v85 // (op 10185) 

v_lshrrev_b32 v85, 3, v1 // (op 10185) 
v_lshl_add_u32 v88, v85, 7, v84 // (op 10185) 


s_nop 0
v_readfirstlane_b32 s7, v88 // (op 10185) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v84, -8, v1 // (op 10229) 
v_sub_u32 v85, v1, v84 // (op 10229) 

v_lshlrev_b32 v84, 4, v85 // (op 10229) 

v_lshrrev_b32 v85, 3, v1 // (op 10229) 
v_lshl_add_u32 v88, v85, 7, v84 // (op 10229) 


s_nop 0
v_readfirstlane_b32 s8, v88 // (op 10229) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v84, -4, v1 // (op 10273) 
v_sub_u32 v85, v1, v84 // (op 10273) 

v_and_b32 v84, 4294967292, v1 // (op 10273) 
v_add_u32 v88, v84, v85 // (op 10273) 


v_and_b32 v84, -4, v1 // (op 10302) 
v_sub_u32 v85, v1, v84 // (op 10302) 

v_lshrrev_b32 v84, 2, v1 // (op 10302) 
v_lshl_add_u32 v89, v84, 3, v85 // (op 10302) 


s_mov_b32 s1, 0 // (op 1302) call()
s_ashr_i64 s[18:19], s[2:3], 7 // (op 1301) 
s_mov_b32 s28, s18 // (op 1301) convert

s_and_b32 s29, s28, 4294967294 // (op 1301) 

s_cmp_lt_u32 s1, s29 // (op 1301) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopBottom_KLoop_1301 // (op 1301) Condition: Top (jump to RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopBottom_KLoop_1301 if false)
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopTop_KLoop_1301:
s_add_u32 s1, s1, 2 // (op 1303) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[68:75], v[36:43], a[0:3], v22, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 198) 
s_barrier  // (op 10738) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[68:75], v[44:51], a[4:7], v22, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 662) 
s_lshl_b64 s[18:19], s[36:37], 5 // (op 8801) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[68:75], v[52:59], a[8:11], v22, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 838) 
v_mov_b64 v[84:85], s[18:19] // (op 8801) convert


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[68:75], v[60:67], a[12:15], v22, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1014) 
v_mov_b64 v[90:91], v[24:25] // (op 8801) 
v_mov_b64 v[92:93], v[90:91] // (op 8801) 
s_mov_b32 m0, s6 // (op 8801) reg expression
s_mov_b32 s18, 256 // (op 8801) call()
buffer_load_dwordx4 v92, s[72:75], s18 offen offset: 0 lds // (op 8801) Load value direct to lds
v_add_co_u32 v92, s[28:29], v92, v84 // (op 8801) least significant half
v_addc_co_u32 v93, s[28:29], 0, v93, s[28:29] // (op 8801) most significant half

s_add_u32 m0, m0, 4096 // (op 8801) 
buffer_load_dwordx4 v92, s[72:75], s18 offen offset: 0 lds // (op 8801) Load value direct to lds
v_add_co_u32 v92, s[28:29], v92, v84 // (op 8801) least significant half
v_addc_co_u32 v93, s[28:29], 0, v93, s[28:29] // (op 8801) most significant half

s_add_u32 m0, m0, 4096 // (op 8801) 
buffer_load_dwordx4 v92, s[72:75], s18 offen offset: 0 lds // (op 8801) Load value direct to lds
v_add_co_u32 v92, s[28:29], v92, v84 // (op 8801) least significant half
v_addc_co_u32 v93, s[28:29], 0, v93, s[28:29] // (op 8801) most significant half

s_add_u32 m0, m0, 4096 // (op 8801) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[76:83], v[36:43], a[16:19], v23, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1509) 
buffer_load_dwordx4 v92, s[72:75], s18 offen offset: 0 lds // (op 8801) Load value direct to lds




v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[76:83], v[44:51], a[20:23], v23, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1463) 
v_mov_b64 v[84:85], s[44:45] // (op 204) convert

v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[76:83], v[52:59], a[24:27], v23, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1417) 
v_mov_b32 v92, v31 // (op 204) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[76:83], v[60:67], a[28:31], v23, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1371) 
v_mov_b32 v93, v92 // (op 204) 
buffer_load_ubyte v34, v93, s[80:83], 0 offen offset:8 // (op 204) Load value
v_add_u32 v93, v93, v84 // (op 204) 
buffer_load_ubyte v35, v93, s[80:83], 0 offen offset:8 // (op 204) Load value


s_barrier  // (op 10719) 

s_lshl_b64 s[18:19], s[30:31], 5 // (op 8796) 
v_mov_b64 v[84:85], s[18:19] // (op 8796) convert


v_mov_b64 v[94:95], v[20:21] // (op 8796) 
v_mov_b64 v[96:97], v[94:95] // (op 8796) 
s_add_u32 s18, 16384, s15 // (op 8796) 
s_mov_b32 m0, s18 // (op 8796) reg expression
s_mov_b32 s19, 256 // (op 8796) call()
buffer_load_dwordx4 v96, s[68:71], s19 offen offset: 0 lds // (op 8796) Load value direct to lds
v_add_co_u32 v96, s[28:29], v96, v84 // (op 8796) least significant half
v_addc_co_u32 v97, s[28:29], 0, v97, s[28:29] // (op 8796) most significant half

s_add_u32 m0, m0, 4096 // (op 8796) 
buffer_load_dwordx4 v96, s[68:71], s19 offen offset: 0 lds // (op 8796) Load value direct to lds






v_mov_b32 v93, v29 // (op 203) 

buffer_load_ubyte v26, v93, s[76:79], 0 offen offset:8 // (op 203) Load value
s_barrier  // (op 10822) 



s_waitcnt vmcnt(13)
ds_write_b8 v89, v32 offset:49920 // (op 541) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v89, v33 offset:49924 // (op 541) Store local data 
s_barrier  // (op 10816) 



s_waitcnt vmcnt(9)
ds_write_b8 v88, v27 offset:50432 // (op 540) Store local data 
s_barrier  // (op 10802) 



ds_read_b128 v[96:99], v12 offset:24576 // (op 533) Load local data 
ds_read_b128 v[100:103], v12 offset:24640 // (op 533) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 10819) 



ds_read_u8 v84, v13 offset:49920 // (op 537) Load local data 



ds_read_b128 v[104:107], v12 offset:28672 // (op 667) Load local data 
ds_read_b128 v[108:111], v12 offset:28736 // (op 667) Load local data 



ds_read_u8 v85, v13 offset:50048 // (op 671) Load local data 



ds_read_b128 v[112:115], v12 offset:32768 // (op 843) Load local data 
ds_read_b128 v[116:119], v12 offset:32832 // (op 843) Load local data 



ds_read_u8 v120, v13 offset:50176 // (op 847) Load local data 



ds_read_b128 v[122:125], v12 offset:36864 // (op 1019) Load local data 
ds_read_b128 v[126:129], v12 offset:36928 // (op 1019) Load local data 



ds_read_u8 v121, v13 offset:50304 // (op 1023) Load local data 
s_barrier  // (op 10793) 
s_barrier  // (op 10941) 



ds_read_b128 v[68:71], v11 offset:40960 // (op 531) Load local data 
ds_read_b128 v[72:75], v11 offset:41024 // (op 531) Load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 10812) 



ds_read_u8 v22, v10 offset:50432 // (op 535) Load local data 



ds_read_b128 v[76:79], v11 offset:45056 // (op 1374) Load local data 
ds_read_b128 v[80:83], v11 offset:45120 // (op 1374) Load local data 



ds_read_u8 v23, v10 offset:50560 // (op 1377) Load local data 
s_barrier  // (op 10967) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[68:75], v[96:103], a[0:3], v22, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 538) 
s_barrier  // (op 10809) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[68:75], v[104:111], a[4:7], v22, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 663) 
s_lshl_b64 s[18:19], s[36:37], 5 // (op 8824) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[68:75], v[112:119], a[8:11], v22, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 839) 
v_mov_b64 v[130:131], s[18:19] // (op 8824) convert


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[68:75], v[122:129], a[12:15], v22, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1015) 
v_mov_b64 v[132:133], v[90:91] // (op 8824) 
s_add_u32 s18, 24576, s8 // (op 8824) 
s_mov_b32 m0, s18 // (op 8824) reg expression
s_mov_b32 s19, 384 // (op 8824) call()
buffer_load_dwordx4 v132, s[72:75], s19 offen offset: 0 lds // (op 8824) Load value direct to lds
v_add_co_u32 v132, s[28:29], v132, v130 // (op 8824) least significant half
v_addc_co_u32 v133, s[28:29], 0, v133, s[28:29] // (op 8824) most significant half

s_add_u32 m0, m0, 4096 // (op 8824) 
buffer_load_dwordx4 v132, s[72:75], s19 offen offset: 0 lds // (op 8824) Load value direct to lds
v_add_co_u32 v132, s[28:29], v132, v130 // (op 8824) least significant half
v_addc_co_u32 v133, s[28:29], 0, v133, s[28:29] // (op 8824) most significant half

s_add_u32 m0, m0, 4096 // (op 8824) 
buffer_load_dwordx4 v132, s[72:75], s19 offen offset: 0 lds // (op 8824) Load value direct to lds
v_add_co_u32 v132, s[28:29], v132, v130 // (op 8824) least significant half
v_addc_co_u32 v133, s[28:29], 0, v133, s[28:29] // (op 8824) most significant half

s_add_u32 m0, m0, 4096 // (op 8824) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[76:83], v[96:103], a[16:19], v23, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1510) 
buffer_load_dwordx4 v132, s[72:75], s19 offen offset: 0 lds // (op 8824) Load value direct to lds








v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[76:83], v[104:111], a[20:23], v23, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1464) 
v_add_co_u32 v24, vcc, 256, v24 // (op 9097) least significant half


v_addc_co_u32 v25, vcc, 0, v25, vcc // (op 9097) most significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[76:83], v[112:119], a[24:27], v23, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1418) 
v_mov_b64 v[84:85], s[44:45] // (op 544) convert



v_mov_b32 v90, v92 // (op 544) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[76:83], v[122:129], a[28:31], v23, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1372) 
buffer_load_ubyte v32, v90, s[80:83], 0 offen offset:12 // (op 544) Load value


v_add_u32 v90, v90, v84 // (op 544) 
buffer_load_ubyte v33, v90, s[80:83], 0 offen offset:12 // (op 544) Load value



s_barrier  // (op 10799) 
v_add_u32 v31, 8, v31 // (op 9353) 

s_lshl_b64 s[18:19], s[30:31], 5 // (op 8819) 
v_mov_b64 v[84:85], s[18:19] // (op 8819) convert


v_mov_b64 v[90:91], v[94:95] // (op 8819) 
s_add_u32 s18, 40960, s7 // (op 8819) 
s_mov_b32 m0, s18 // (op 8819) reg expression
s_mov_b32 s19, 384 // (op 8819) call()
buffer_load_dwordx4 v90, s[68:71], s19 offen offset: 0 lds // (op 8819) Load value direct to lds
v_add_co_u32 v90, s[28:29], v90, v84 // (op 8819) least significant half
v_addc_co_u32 v91, s[28:29], 0, v91, s[28:29] // (op 8819) most significant half

s_add_u32 m0, m0, 4096 // (op 8819) 
buffer_load_dwordx4 v90, s[68:71], s19 offen offset: 0 lds // (op 8819) Load value direct to lds








v_add_co_u32 v20, vcc, 256, v20 // (op 8959) least significant half
buffer_load_ubyte v27, v93, s[76:79], 0 offen offset:12 // (op 543) Load value
v_addc_co_u32 v21, vcc, 0, v21, vcc // (op 8959) most significant half

s_barrier  // (op 10779) 
v_add_u32 v29, 8, v29 // (op 9225) 



s_waitcnt vmcnt(13)
ds_write_b8 v87, v34 offset:49152 // (op 201) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v87, v35 offset:49156 // (op 201) Store local data 
s_barrier  // (op 10758) 



s_waitcnt vmcnt(9)
ds_write_b8 v86, v26 offset:49664 // (op 200) Store local data 
s_barrier  // (op 10744) 



ds_read_b128 v[36:39], v8 // (op 193) Load local data 
ds_read_b128 v[40:43], v8 offset:64 // (op 193) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 10786) 



ds_read_u8 v28, v9 offset:49152 // (op 197) Load local data 



ds_read_b128 v[44:47], v8 offset:4096 // (op 657) Load local data 
ds_read_b128 v[48:51], v8 offset:4160 // (op 657) Load local data 



ds_read_u8 v30, v9 offset:49280 // (op 661) Load local data 



ds_read_b128 v[52:55], v8 offset:8192 // (op 833) Load local data 
ds_read_b128 v[56:59], v8 offset:8256 // (op 833) Load local data 



ds_read_u8 v16, v9 offset:49408 // (op 837) Load local data 



ds_read_b128 v[60:63], v8 offset:12288 // (op 1009) Load local data 
ds_read_b128 v[64:67], v8 offset:12352 // (op 1009) Load local data 



ds_read_u8 v17, v9 offset:49536 // (op 1013) Load local data 
s_barrier  // (op 10725) 



ds_read_b128 v[68:71], v7 offset:16384 // (op 191) Load local data 
ds_read_b128 v[72:75], v7 offset:16448 // (op 191) Load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 10765) 



ds_read_u8 v22, v6 offset:49664 // (op 195) Load local data 



ds_read_b128 v[76:79], v7 offset:20480 // (op 1366) Load local data 
ds_read_b128 v[80:83], v7 offset:20544 // (op 1366) Load local data 



ds_read_u8 v23, v6 offset:49792 // (op 1369) Load local data 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 1301) 
s_mov_b32 s28, s18 // (op 1301) convert

s_and_b32 s29, s28, 4294967294 // (op 1301) 

s_cmp_lt_u32 s1, s29 // (op 1301) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopTop_KLoop_1301 // (op 1301) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopTop_KLoop_1301 if true)
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopBottom_KLoop_1301:
















s_barrier  // (op 10949) 
v_lshrrev_b32 v31, 3, v1 // (op 8891) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 8891) 
v_and_b32 v20, -8, v1 // (op 8891) 
s_and_b32 s15, s18, 4294967294 // (op 8891) 
v_sub_u32 v21, v1, v20 // (op 8891) 

s_add_u32 s6, s0, s15 // (op 8891) 

v_lshlrev_b32 v20, 4, v21 // (op 8891) 

s_mul_i32 s15, s16, s50 // (op 8891) 
s_add_u32 s15, s15, s6 // (op 8891) 

s_add_u32 s6, s15, s9 // (op 8891) 

s_mul_hi_u32 s15, s6, s13 // (op 8891) 
s_sub_u32 s7, s6, s15 // (op 8891) 
s_lshr_b32 s8, s7, 1 // (op 8891) 

s_add_u32 s7, s8, s15 // (op 8891) 


s_lshr_b32 s8, s7, s11 // (op 8891) 

s_and_b32 s7, s11, 2147483648 // (op 8891) 
s_cmp_eq_i32 s7, 0 // (op 8891) 
s_mov_b32 s15, scc // (op 8891) 

s_cmp_eq_i32 s15, 0 // (op 8891) 
s_cselect_b32 s7, s6, s8 // (op 8891) 


s_mul_hi_u32 s8, s7, s12 // (op 8891) 
s_sub_u32 s15, s7, s8 // (op 8891) 
s_lshr_b32 s28, s15, 1 // (op 8891) 

s_add_u32 s15, s28, s8 // (op 8891) 


s_lshr_b32 s8, s15, s10 // (op 8891) 

s_and_b32 s15, s10, 2147483648 // (op 8891) 
s_cmp_eq_i32 s15, 0 // (op 8891) 
s_mov_b32 s28, scc // (op 8891) 

s_cmp_eq_i32 s28, 0 // (op 8891) 
s_cselect_b32 s15, s7, s8 // (op 8891) 


s_mul_i32 s8, s7, s18 // (op 8891) 


v_lshl_add_u32 v21, s15, 6, v31 // (op 8891) 


s_sub_u32 s7, s6, s8 // (op 8891) 


v_lshl_add_u32 v31, s7, 7, v20 // (op 8891) 


v_mov_b32 v29, 0 // (op 8891) convert
v_mov_b32 v28, v31 // (op 8891) convert

v_mul_lo_u32 v25, v21, s31 // (op 8891) most significant: low of low * high
v_mul_hi_u32 v20, v21, s30 // (op 8891) most significant: high of low * low
v_mul_lo_u32 v24, v21, s30 // (op 8891) least significant: low of low * low
v_add_u32 v25, v25, v20 // (op 8891) most significant: sum

v_add_co_u32 v24, s[18:19], v24, v28 // (op 8891) least significant half
v_addc_co_u32 v25, s[18:19], v25, v29, s[18:19] // (op 8891) most significant half



s_mul_i32 s19, s30, s35 // (op 8893) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 8893) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 8893) least significant
s_mul_hi_u32 s6, s30, s34 // (op 8893) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 8893) most significant: sum
s_add_u32 s19, s19, s6 // (op 8893) most significant: sum


s_mov_b32 s68, s32 // (op 8893) reg expression
s_mov_b32 s69, s33 // (op 8893) reg expression
s_mov_b32 s70, s18 // (op 8893) reg expression
s_mov_b32 s71, 131072 // (op 8893) call()

v_lshrrev_b32 v26, 3, v1 // (op 9041) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 9041) 
v_and_b32 v31, -8, v1 // (op 9041) 
s_and_b32 s15, s18, 4294967294 // (op 9041) 
v_sub_u32 v20, v1, v31 // (op 9041) 

s_add_u32 s6, s0, s15 // (op 9041) 

v_lshlrev_b32 v21, 4, v20 // (op 9041) 

s_mul_i32 s15, s16, s50 // (op 9041) 
s_add_u32 s15, s15, s6 // (op 9041) 

s_add_u32 s6, s15, s9 // (op 9041) 

s_mul_hi_u32 s15, s6, s13 // (op 9041) 
s_sub_u32 s7, s6, s15 // (op 9041) 
s_lshr_b32 s8, s7, 1 // (op 9041) 

s_add_u32 s7, s8, s15 // (op 9041) 


s_lshr_b32 s8, s7, s11 // (op 9041) 

s_and_b32 s7, s11, 2147483648 // (op 9041) 
s_cmp_eq_i32 s7, 0 // (op 9041) 
s_mov_b32 s15, scc // (op 9041) 

s_cmp_eq_i32 s15, 0 // (op 9041) 
s_cselect_b32 s7, s6, s8 // (op 9041) 


s_mul_hi_u32 s8, s7, s12 // (op 9041) 
s_sub_u32 s15, s7, s8 // (op 9041) 
s_lshr_b32 s28, s15, 1 // (op 9041) 

s_add_u32 s15, s28, s8 // (op 9041) 


s_lshr_b32 s8, s15, s10 // (op 9041) 

s_and_b32 s15, s10, 2147483648 // (op 9041) 
s_cmp_eq_i32 s15, 0 // (op 9041) 
s_mov_b32 s28, scc // (op 9041) 

s_cmp_eq_i32 s28, 0 // (op 9041) 
s_cselect_b32 s15, s7, s8 // (op 9041) 


s_mul_i32 s8, s15, s14 // (op 9041) 

s_sub_u32 s15, s7, s8 // (op 9041) 

s_mul_i32 s8, s7, s18 // (op 9041) 


v_lshl_add_u32 v20, s15, 7, v26 // (op 9041) 


s_sub_u32 s7, s6, s8 // (op 9041) 


v_mul_lo_u32 v29, v20, s37 // (op 9041) most significant: low of low * high
v_mul_hi_u32 v26, v20, s36 // (op 9041) most significant: high of low * low
v_mul_lo_u32 v28, v20, s36 // (op 9041) least significant: low of low * low
v_add_u32 v29, v29, v26 // (op 9041) most significant: sum


v_lshl_add_u32 v20, s7, 7, v21 // (op 9041) 


v_mov_b32 v35, 0 // (op 9041) convert
v_mov_b32 v34, v20 // (op 9041) convert

v_add_co_u32 v20, s[18:19], v34, v28 // (op 9041) least significant half
v_addc_co_u32 v21, s[18:19], v35, v29, s[18:19] // (op 9041) most significant half



s_mul_i32 s19, s36, s23 // (op 9043) most significant: low of low * high
s_mul_i32 s15, s37, s22 // (op 9043) most significant: low of high * low
s_mul_i32 s18, s36, s22 // (op 9043) least significant
s_mul_hi_u32 s6, s36, s22 // (op 9043) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 9043) most significant: sum
s_add_u32 s19, s19, s6 // (op 9043) most significant: sum


s_mov_b32 s72, s38 // (op 9043) reg expression
s_mov_b32 s73, s39 // (op 9043) reg expression
s_mov_b32 s74, s18 // (op 9043) reg expression
s_mov_b32 s75, 131072 // (op 9043) call()

v_lshrrev_b32 v26, 2, v1 // (op 9169) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 9169) 
v_and_b32 v31, -4, v1 // (op 9169) 
s_and_b32 s15, s18, 4294967294 // (op 9169) 
v_sub_u32 v28, v1, v31 // (op 9169) 

s_add_u32 s6, s0, s15 // (op 9169) 

s_mul_i32 s15, s16, s50 // (op 9169) 
s_add_u32 s15, s15, s6 // (op 9169) 

s_add_u32 s6, s15, s9 // (op 9169) 

s_mul_hi_u32 s15, s6, s13 // (op 9169) 
s_sub_u32 s7, s6, s15 // (op 9169) 
s_lshr_b32 s8, s7, 1 // (op 9169) 

s_add_u32 s7, s8, s15 // (op 9169) 


s_lshr_b32 s8, s7, s11 // (op 9169) 

s_and_b32 s7, s11, 2147483648 // (op 9169) 
s_cmp_eq_i32 s7, 0 // (op 9169) 
s_mov_b32 s15, scc // (op 9169) 

s_cmp_eq_i32 s15, 0 // (op 9169) 
s_cselect_b32 s7, s6, s8 // (op 9169) 


s_mul_hi_u32 s8, s7, s12 // (op 9169) 
s_sub_u32 s15, s7, s8 // (op 9169) 
s_lshr_b32 s28, s15, 1 // (op 9169) 

s_add_u32 s15, s28, s8 // (op 9169) 


s_lshr_b32 s8, s15, s10 // (op 9169) 

s_and_b32 s15, s10, 2147483648 // (op 9169) 
s_cmp_eq_i32 s15, 0 // (op 9169) 
s_mov_b32 s28, scc // (op 9169) 

s_cmp_eq_i32 s28, 0 // (op 9169) 
s_cselect_b32 s15, s7, s8 // (op 9169) 


s_mul_i32 s8, s7, s18 // (op 9169) 


v_lshl_add_u32 v29, s15, 6, v26 // (op 9169) 


s_sub_u32 s7, s6, s8 // (op 9169) 


v_lshl_add_u32 v26, s7, 2, v28 // (op 9169) 


v_mul_lo_u32 v28, v29, s40 // (op 9169) 
v_add_u32 v28, v28, v26 // (op 9169) 


s_mul_i32 s19, s40, s35 // (op 9171) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 9171) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 9171) least significant
s_mul_hi_u32 s6, s40, s34 // (op 9171) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 9171) most significant: sum
s_add_u32 s19, s19, s6 // (op 9171) most significant: sum


s_mov_b32 s76, s42 // (op 9171) reg expression
s_mov_b32 s77, s43 // (op 9171) reg expression
s_mov_b32 s78, s18 // (op 9171) reg expression
s_mov_b32 s79, 131072 // (op 9171) call()

v_lshrrev_b32 v26, 2, v1 // (op 9297) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 9297) 
v_lshlrev_b32 v29, 1, v26 // (op 9297) 

s_and_b32 s15, s18, 4294967294 // (op 9297) 
v_and_b32 v26, -4, v1 // (op 9297) 
s_add_u32 s6, s0, s15 // (op 9297) 

v_sub_u32 v31, v1, v26 // (op 9297) 

s_mul_i32 s15, s16, s50 // (op 9297) 
s_add_u32 s15, s15, s6 // (op 9297) 

s_add_u32 s6, s15, s9 // (op 9297) 

s_mul_hi_u32 s15, s6, s13 // (op 9297) 
s_sub_u32 s7, s6, s15 // (op 9297) 
s_lshr_b32 s8, s7, 1 // (op 9297) 

s_add_u32 s7, s8, s15 // (op 9297) 


s_lshr_b32 s8, s7, s11 // (op 9297) 

s_and_b32 s7, s11, 2147483648 // (op 9297) 
s_cmp_eq_i32 s7, 0 // (op 9297) 
s_mov_b32 s15, scc // (op 9297) 

s_cmp_eq_i32 s15, 0 // (op 9297) 
s_cselect_b32 s7, s6, s8 // (op 9297) 


s_mul_hi_u32 s8, s7, s12 // (op 9297) 
s_sub_u32 s15, s7, s8 // (op 9297) 
s_lshr_b32 s28, s15, 1 // (op 9297) 

s_add_u32 s15, s28, s8 // (op 9297) 


s_lshr_b32 s8, s15, s10 // (op 9297) 

s_and_b32 s15, s10, 2147483648 // (op 9297) 
s_cmp_eq_i32 s15, 0 // (op 9297) 
s_mov_b32 s28, scc // (op 9297) 

s_cmp_eq_i32 s28, 0 // (op 9297) 
s_cselect_b32 s15, s7, s8 // (op 9297) 


s_mul_i32 s8, s15, s14 // (op 9297) 

s_sub_u32 s15, s7, s8 // (op 9297) 

v_lshl_add_u32 v26, s15, 7, v29 // (op 9297) 


s_mul_i32 s8, s7, s18 // (op 9297) 


v_mul_lo_u32 v29, v26, s44 // (op 9297) 

s_sub_u32 s7, s6, s8 // (op 9297) 


v_lshl_add_u32 v26, s7, 2, v31 // (op 9297) 


v_add_u32 v31, v26, v29 // (op 9297) 


s_mul_i32 s19, s44, s23 // (op 9299) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 9299) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 9299) least significant
s_mul_hi_u32 s6, s44, s22 // (op 9299) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 9299) most significant: sum
s_add_u32 s19, s19, s6 // (op 9299) most significant: sum


s_mov_b32 s80, s46 // (op 9299) reg expression
s_mov_b32 s81, s47 // (op 9299) reg expression
s_mov_b32 s82, s18 // (op 9299) reg expression
s_mov_b32 s83, 131072 // (op 9299) call()

v_and_b32 v26, -8, v1 // (op 9844) 
v_sub_u32 v29, v1, v26 // (op 9844) 

v_lshlrev_b32 v26, 4, v29 // (op 9844) 

v_lshrrev_b32 v29, 3, v1 // (op 9844) 
v_lshl_add_u32 v34, v29, 7, v26 // (op 9844) 


s_nop 0
v_readfirstlane_b32 s15, v34 // (op 9844) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v26, -8, v1 // (op 9919) 
v_sub_u32 v29, v1, v26 // (op 9919) 

v_lshlrev_b32 v26, 4, v29 // (op 9919) 

v_lshrrev_b32 v29, 3, v1 // (op 9919) 
v_lshl_add_u32 v34, v29, 7, v26 // (op 9919) 


s_nop 0
v_readfirstlane_b32 s6, v34 // (op 9919) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v26, -4, v1 // (op 9994) 
v_sub_u32 v29, v1, v26 // (op 9994) 

v_and_b32 v26, 4294967292, v1 // (op 9994) 
v_add_u32 v34, v26, v29 // (op 9994) 


v_and_b32 v26, -4, v1 // (op 10070) 
v_sub_u32 v29, v1, v26 // (op 10070) 

v_lshrrev_b32 v26, 2, v1 // (op 10070) 
v_lshl_add_u32 v35, v26, 3, v29 // (op 10070) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1307) 
s_and_b32 s7, s18, 4294967294 // (op 1307) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 1306) 
s_mov_b32 s8, s18 // (op 1306) convert

s_cmp_lt_u32 s7, s8 // (op 1306) 













s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopBottom_KLoopTail_1306 // (op 1306) Condition: Top (jump to RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopBottom_KLoopTail_1306 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopTop_KLoopTail_1306:
s_barrier  // (op 10722) 
s_add_u32 s7, s7, 1 // (op 1308) 

s_lshl_b64 s[18:19], s[30:31], 5 // (op 8806) 
v_mov_b64 v[16:17], s[18:19] // (op 8806) convert


v_mov_b64 v[22:23], v[24:25] // (op 8806) 
v_mov_b64 v[26:27], v[22:23] // (op 8806) 
s_add_u32 s8, 16384, s15 // (op 8806) 
s_mov_b32 m0, s8 // (op 8806) reg expression
buffer_load_dwordx4 v26, s[68:71], 0 offen offset: 0 lds // (op 8806) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v16 // (op 8806) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 8806) most significant half

s_add_u32 m0, m0, 4096 // (op 8806) 
buffer_load_dwordx4 v26, s[68:71], 0 offen offset: 0 lds // (op 8806) Load value direct to lds




v_add_co_u32 v24, vcc, 128, v24 // (op 8881) least significant half
v_addc_co_u32 v25, vcc, 0, v25, vcc // (op 8881) most significant half
s_barrier  // (op 10741) 

s_lshl_b64 s[18:19], s[36:37], 5 // (op 8812) 
v_mov_b64 v[16:17], s[18:19] // (op 8812) convert


v_mov_b64 v[22:23], v[20:21] // (op 8812) 
v_mov_b64 v[26:27], v[22:23] // (op 8812) 
s_mov_b32 m0, s6 // (op 8812) reg expression
buffer_load_dwordx4 v26, s[72:75], 0 offen offset: 0 lds // (op 8812) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v16 // (op 8812) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 8812) most significant half

s_add_u32 m0, m0, 4096 // (op 8812) 
buffer_load_dwordx4 v26, s[72:75], 0 offen offset: 0 lds // (op 8812) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v16 // (op 8812) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 8812) most significant half

s_add_u32 m0, m0, 4096 // (op 8812) 
buffer_load_dwordx4 v26, s[72:75], 0 offen offset: 0 lds // (op 8812) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v16 // (op 8812) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 8812) most significant half

s_add_u32 m0, m0, 4096 // (op 8812) 
buffer_load_dwordx4 v26, s[72:75], 0 offen offset: 0 lds // (op 8812) Load value direct to lds



v_add_co_u32 v20, vcc, 128, v20 // (op 9033) least significant half
v_addc_co_u32 v21, vcc, 0, v21, vcc // (op 9033) most significant half


v_mov_b32 v16, v28 // (op 477) 

buffer_load_ubyte v17, v16, s[76:79], 0 offen offset: 0 // (op 477) Load value

v_add_u32 v28, 4, v28 // (op 9161) 

s_barrier  // (op 10761) 
v_mov_b64 v[22:23], s[44:45] // (op 478) convert




v_mov_b32 v16, v31 // (op 478) 
v_mov_b32 v26, v16 // (op 478) 
buffer_load_ubyte v32, v26, s[80:83], 0 offen offset: 0 // (op 478) Load value
v_add_u32 v26, v26, v22 // (op 478) 
buffer_load_ubyte v33, v26, s[80:83], 0 offen offset: 0 // (op 478) Load value



v_add_u32 v31, 4, v31 // (op 9289) 
s_waitcnt vmcnt(2)
ds_write_b8 v34, v17 offset:49664 // (op 474) Store local data 

s_barrier  // (op 10728) 



s_barrier  // (op 10782) 
ds_read_b128 v[36:39], v7 offset:16384 // (op 465) Load local data 



ds_read_b128 v[40:43], v7 offset:16448 // (op 465) Load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v35, v32 offset:49152 // (op 475) Store local data 
s_barrier  // (op 10748) 
s_waitcnt vmcnt(0)
ds_write_b8 v35, v33 offset:49156 // (op 475) Store local data 




ds_read_b128 v[44:47], v8 // (op 467) Load local data 
ds_read_b128 v[48:51], v8 offset:64 // (op 467) Load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 10769) 



ds_read_u8 v16, v6 offset:49664 // (op 469) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 10789) 



ds_read_u8 v17, v9 offset:49152 // (op 471) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:43], v[44:51], a[0:3], v16, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 472) 
ds_read_b128 v[52:55], v8 offset:4096 // (op 636) Load local data 
ds_read_b128 v[56:59], v8 offset:4160 // (op 636) Load local data 



ds_read_u8 v22, v9 offset:49280 // (op 640) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[36:43], v[52:59], a[4:7], v16, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 641) 
ds_read_b128 v[60:63], v8 offset:8192 // (op 812) Load local data 
ds_read_b128 v[64:67], v8 offset:8256 // (op 812) Load local data 



ds_read_u8 v23, v9 offset:49408 // (op 816) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[36:43], v[60:67], a[8:11], v16, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 817) 
ds_read_b128 v[68:71], v8 offset:12288 // (op 988) Load local data 
ds_read_b128 v[72:75], v8 offset:12352 // (op 988) Load local data 



ds_read_u8 v26, v9 offset:49536 // (op 992) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[36:43], v[68:75], a[12:15], v16, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 993) 
ds_read_b128 v[76:79], v7 offset:20480 // (op 1250) Load local data 


ds_read_b128 v[80:83], v7 offset:20544 // (op 1250) Load local data 



ds_read_u8 v16, v6 offset:49792 // (op 1253) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[76:83], v[44:51], a[16:19], v16, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1255) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[76:83], v[52:59], a[20:23], v16, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1286) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[76:83], v[60:67], a[24:27], v16, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1317) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[76:83], v[68:75], a[28:31], v16, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1348) 




s_ashr_i64 s[18:19], s[2:3], 7 // (op 1306) 
s_mov_b32 s8, s18 // (op 1306) convert

s_cmp_lt_u32 s7, s8 // (op 1306) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopTop_KLoopTail_1306 // (op 1306) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopTop_KLoopTail_1306 if true)
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopBottom_KLoopTail_1306:









s_barrier  // (op 10953) 




v_cmp_eq_f32 vcc, 0.00000, v18 // (op 238) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalFalse_0_beta_238 // (op 238) Condition: False, jump to RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalFalse_0_beta_238
v_accvgpr_read v14, a0 // (op 246) 
v_accvgpr_read v15, a1 // (op 246) 
v_accvgpr_read v16, a2 // (op 246) 
v_accvgpr_read v17, a3 // (op 246) 
v_mul_f32 v20, v19, v14 // (op 246) 
v_mul_f32 v21, v19, v15 // (op 246) 
v_mul_f32 v22, v19, v16 // (op 246) 
v_mul_f32 v23, v19, v17 // (op 246) 

v_lshrrev_b32 v14, 16, v20 // (op 248) 
v_lshrrev_b32 v15, 16, v21 // (op 248) 
v_lshrrev_b32 v16, 16, v22 // (op 248) 
v_lshrrev_b32 v17, 16, v23 // (op 248) 


v_pack_b32_f16 v20, v14, v15 // (op 249) 
v_pack_b32_f16 v21, v16, v17 // (op 249) 

v_mov_b32 v0, v5 // (op 249) reg expression

buffer_store_dwordx2 v[20:21], v0, s[60:63], 0 offen // (op 249) Store value



v_accvgpr_read v14, a4 // (op 2155) 
v_accvgpr_read v15, a5 // (op 2155) 
v_accvgpr_read v16, a6 // (op 2155) 
v_accvgpr_read v17, a7 // (op 2155) 
v_mul_f32 v20, v19, v14 // (op 2155) 
v_mul_f32 v21, v19, v15 // (op 2155) 
v_mul_f32 v22, v19, v16 // (op 2155) 
v_mul_f32 v23, v19, v17 // (op 2155) 

v_lshrrev_b32 v14, 16, v20 // (op 2157) 
v_lshrrev_b32 v15, 16, v21 // (op 2157) 
v_lshrrev_b32 v16, 16, v22 // (op 2157) 
v_lshrrev_b32 v17, 16, v23 // (op 2157) 


v_pack_b32_f16 v20, v14, v15 // (op 2158) 
v_pack_b32_f16 v21, v16, v17 // (op 2158) 

s_lshl_b32 s15, s24, 6 // (op 2158) 
v_add_u32 v0, v5, s15 // (op 2158) 


buffer_store_dwordx2 v[20:21], v0, s[60:63], 0 offen // (op 2158) Store value



v_accvgpr_read v14, a8 // (op 2169) 
v_accvgpr_read v15, a9 // (op 2169) 
v_accvgpr_read v16, a10 // (op 2169) 
v_accvgpr_read v17, a11 // (op 2169) 
v_mul_f32 v20, v19, v14 // (op 2169) 
v_mul_f32 v21, v19, v15 // (op 2169) 
v_mul_f32 v22, v19, v16 // (op 2169) 
v_mul_f32 v23, v19, v17 // (op 2169) 

v_lshrrev_b32 v14, 16, v20 // (op 2171) 
v_lshrrev_b32 v15, 16, v21 // (op 2171) 
v_lshrrev_b32 v16, 16, v22 // (op 2171) 
v_lshrrev_b32 v17, 16, v23 // (op 2171) 


v_pack_b32_f16 v20, v14, v15 // (op 2172) 
v_pack_b32_f16 v21, v16, v17 // (op 2172) 

s_lshl_b32 s15, s24, 7 // (op 2172) 
v_add_u32 v0, v5, s15 // (op 2172) 


buffer_store_dwordx2 v[20:21], v0, s[60:63], 0 offen // (op 2172) Store value



v_accvgpr_read v14, a12 // (op 2183) 
v_accvgpr_read v15, a13 // (op 2183) 
v_accvgpr_read v16, a14 // (op 2183) 
v_accvgpr_read v17, a15 // (op 2183) 
v_mul_f32 v20, v19, v14 // (op 2183) 
v_mul_f32 v21, v19, v15 // (op 2183) 
v_mul_f32 v22, v19, v16 // (op 2183) 
v_mul_f32 v23, v19, v17 // (op 2183) 

v_lshrrev_b32 v14, 16, v20 // (op 2185) 
v_lshrrev_b32 v15, 16, v21 // (op 2185) 
v_lshrrev_b32 v16, 16, v22 // (op 2185) 
v_lshrrev_b32 v17, 16, v23 // (op 2185) 


v_pack_b32_f16 v20, v14, v15 // (op 2186) 
v_pack_b32_f16 v21, v16, v17 // (op 2186) 

s_lshl_b32 s15, s24, 6 // (op 2186) 
s_mul_i32 s6, 3, s15 // (op 2186) 

v_add_u32 v0, v5, s6 // (op 2186) 


buffer_store_dwordx2 v[20:21], v0, s[60:63], 0 offen // (op 2186) Store value



v_accvgpr_read v14, a16 // (op 2219) 
v_accvgpr_read v15, a17 // (op 2219) 
v_accvgpr_read v16, a18 // (op 2219) 
v_accvgpr_read v17, a19 // (op 2219) 
v_mul_f32 v20, v19, v14 // (op 2219) 
v_mul_f32 v21, v19, v15 // (op 2219) 
v_mul_f32 v22, v19, v16 // (op 2219) 
v_mul_f32 v23, v19, v17 // (op 2219) 

v_lshrrev_b32 v14, 16, v20 // (op 2221) 
v_lshrrev_b32 v15, 16, v21 // (op 2221) 
v_lshrrev_b32 v16, 16, v22 // (op 2221) 
v_lshrrev_b32 v17, 16, v23 // (op 2221) 


v_pack_b32_f16 v20, v14, v15 // (op 2223) 
v_pack_b32_f16 v21, v16, v17 // (op 2223) 

v_add_u32 v0, 64, v5 // (op 2223) 

buffer_store_dwordx2 v[20:21], v0, s[60:63], 0 offen // (op 2223) Store value



v_accvgpr_read v14, a20 // (op 2233) 
v_accvgpr_read v15, a21 // (op 2233) 
v_accvgpr_read v16, a22 // (op 2233) 
v_accvgpr_read v17, a23 // (op 2233) 
v_mul_f32 v20, v19, v14 // (op 2233) 
v_mul_f32 v21, v19, v15 // (op 2233) 
v_mul_f32 v22, v19, v16 // (op 2233) 
v_mul_f32 v23, v19, v17 // (op 2233) 

v_lshrrev_b32 v14, 16, v20 // (op 2235) 
v_lshrrev_b32 v15, 16, v21 // (op 2235) 
v_lshrrev_b32 v16, 16, v22 // (op 2235) 
v_lshrrev_b32 v17, 16, v23 // (op 2235) 


v_pack_b32_f16 v20, v14, v15 // (op 2225) 
v_pack_b32_f16 v21, v16, v17 // (op 2225) 

s_lshl_b32 s15, s24, 6 // (op 2225) 
s_add_u32 s15, s15, 64 // (op 2225) 
v_add_u32 v0, v5, s15 // (op 2225) 


buffer_store_dwordx2 v[20:21], v0, s[60:63], 0 offen // (op 2225) Store value



v_accvgpr_read v14, a24 // (op 2239) 
v_accvgpr_read v15, a25 // (op 2239) 
v_accvgpr_read v16, a26 // (op 2239) 
v_accvgpr_read v17, a27 // (op 2239) 
v_mul_f32 v20, v19, v14 // (op 2239) 
v_mul_f32 v21, v19, v15 // (op 2239) 
v_mul_f32 v22, v19, v16 // (op 2239) 
v_mul_f32 v23, v19, v17 // (op 2239) 

v_lshrrev_b32 v14, 16, v20 // (op 2241) 
v_lshrrev_b32 v15, 16, v21 // (op 2241) 
v_lshrrev_b32 v16, 16, v22 // (op 2241) 
v_lshrrev_b32 v17, 16, v23 // (op 2241) 


v_pack_b32_f16 v20, v14, v15 // (op 2227) 
v_pack_b32_f16 v21, v16, v17 // (op 2227) 

s_lshl_b32 s15, s24, 7 // (op 2227) 
s_add_u32 s15, s15, 64 // (op 2227) 
v_add_u32 v0, v5, s15 // (op 2227) 


buffer_store_dwordx2 v[20:21], v0, s[60:63], 0 offen // (op 2227) Store value



v_accvgpr_read v14, a28 // (op 2245) 
v_accvgpr_read v15, a29 // (op 2245) 
v_accvgpr_read v16, a30 // (op 2245) 
v_accvgpr_read v17, a31 // (op 2245) 
v_mul_f32 v20, v19, v14 // (op 2245) 
v_mul_f32 v21, v19, v15 // (op 2245) 
v_mul_f32 v22, v19, v16 // (op 2245) 
v_mul_f32 v23, v19, v17 // (op 2245) 

v_lshrrev_b32 v14, 16, v20 // (op 2247) 
v_lshrrev_b32 v15, 16, v21 // (op 2247) 
v_lshrrev_b32 v16, 16, v22 // (op 2247) 
v_lshrrev_b32 v17, 16, v23 // (op 2247) 


v_pack_b32_f16 v20, v14, v15 // (op 2229) 
v_pack_b32_f16 v21, v16, v17 // (op 2229) 

s_lshl_b32 s15, s24, 6 // (op 2229) 
s_mul_i32 s6, 3, s15 // (op 2229) 
s_add_u32 s6, s6, 64 // (op 2229) 

v_add_u32 v0, v5, s6 // (op 2229) 


buffer_store_dwordx2 v[20:21], v0, s[60:63], 0 offen // (op 2229) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalBottom_0_beta_238 // (op 238) Condition: Done, jump to RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalBottom_0_beta_238
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalFalse_0_beta_238:


v_add_u32 v0, 64, v4 // (op 2415) 

buffer_load_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 2415) Load value

v_accvgpr_read v20, a16 // (op 2402) 
v_accvgpr_read v21, a17 // (op 2402) 
v_accvgpr_read v22, a18 // (op 2402) 
v_accvgpr_read v23, a19 // (op 2402) 
v_mul_f32 v24, v19, v20 // (op 2402) 
v_mul_f32 v25, v19, v21 // (op 2402) 
v_mul_f32 v26, v19, v22 // (op 2402) 
v_mul_f32 v27, v19, v23 // (op 2402) 

s_waitcnt vmcnt(0)
v_and_b32 v16, 65535, v14 // (op 2414) 
v_lshrrev_b32 v17, 16, v14 // (op 2414) 
v_lshlrev_b32 v16, 16, v16 // (op 2414) 
v_lshlrev_b32 v17, 16, v17 // (op 2414) 
v_mul_f32 v20, v18, v16 // (op 2414) 
v_mul_f32 v21, v18, v17 // (op 2414) 
v_and_b32 v16, 65535, v15 // (op 2414) 
v_lshrrev_b32 v17, 16, v15 // (op 2414) 
v_lshlrev_b32 v16, 16, v16 // (op 2414) 
v_lshlrev_b32 v17, 16, v17 // (op 2414) 
v_mul_f32 v22, v18, v16 // (op 2414) 
v_mul_f32 v23, v18, v17 // (op 2414) 


v_add_f32 v14, v20, v24 // (op 2403) 
v_add_f32 v15, v21, v25 // (op 2403) 
v_add_f32 v16, v22, v26 // (op 2403) 
v_add_f32 v17, v23, v27 // (op 2403) 


v_lshrrev_b32 v20, 16, v14 // (op 2404) 
v_lshrrev_b32 v21, 16, v15 // (op 2404) 
v_lshrrev_b32 v22, 16, v16 // (op 2404) 
v_lshrrev_b32 v23, 16, v17 // (op 2404) 


v_pack_b32_f16 v14, v20, v21 // (op 2406) 
v_pack_b32_f16 v15, v22, v23 // (op 2406) 

v_add_u32 v0, 64, v5 // (op 2406) 

buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 2406) Store value





s_lshl_b32 s15, s4, 6 // (op 2420) 
s_add_u32 s15, s15, 64 // (op 2420) 
v_add_u32 v0, v4, s15 // (op 2420) 


buffer_load_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 2420) Load value

v_accvgpr_read v20, a20 // (op 2432) 
v_accvgpr_read v21, a21 // (op 2432) 
v_accvgpr_read v22, a22 // (op 2432) 
v_accvgpr_read v23, a23 // (op 2432) 
v_mul_f32 v24, v19, v20 // (op 2432) 
v_mul_f32 v25, v19, v21 // (op 2432) 
v_mul_f32 v26, v19, v22 // (op 2432) 
v_mul_f32 v27, v19, v23 // (op 2432) 

s_waitcnt vmcnt(0)
v_and_b32 v16, 65535, v14 // (op 2417) 
v_lshrrev_b32 v17, 16, v14 // (op 2417) 
v_lshlrev_b32 v16, 16, v16 // (op 2417) 
v_lshlrev_b32 v17, 16, v17 // (op 2417) 
v_mul_f32 v20, v18, v16 // (op 2417) 
v_mul_f32 v21, v18, v17 // (op 2417) 
v_and_b32 v16, 65535, v15 // (op 2417) 
v_lshrrev_b32 v17, 16, v15 // (op 2417) 
v_lshlrev_b32 v16, 16, v16 // (op 2417) 
v_lshlrev_b32 v17, 16, v17 // (op 2417) 
v_mul_f32 v22, v18, v16 // (op 2417) 
v_mul_f32 v23, v18, v17 // (op 2417) 


v_add_f32 v14, v20, v24 // (op 2418) 
v_add_f32 v15, v21, v25 // (op 2418) 
v_add_f32 v16, v22, v26 // (op 2418) 
v_add_f32 v17, v23, v27 // (op 2418) 


v_lshrrev_b32 v20, 16, v14 // (op 2419) 
v_lshrrev_b32 v21, 16, v15 // (op 2419) 
v_lshrrev_b32 v22, 16, v16 // (op 2419) 
v_lshrrev_b32 v23, 16, v17 // (op 2419) 


v_pack_b32_f16 v14, v20, v21 // (op 2408) 
v_pack_b32_f16 v15, v22, v23 // (op 2408) 

s_lshl_b32 s15, s24, 6 // (op 2408) 
s_add_u32 s15, s15, 64 // (op 2408) 
v_add_u32 v0, v5, s15 // (op 2408) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 2408) Store value





s_lshl_b32 s15, s4, 7 // (op 2425) 
s_add_u32 s15, s15, 64 // (op 2425) 
v_add_u32 v0, v4, s15 // (op 2425) 


buffer_load_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 2425) Load value

v_accvgpr_read v20, a24 // (op 2434) 
v_accvgpr_read v21, a25 // (op 2434) 
v_accvgpr_read v22, a26 // (op 2434) 
v_accvgpr_read v23, a27 // (op 2434) 
v_mul_f32 v24, v19, v20 // (op 2434) 
v_mul_f32 v25, v19, v21 // (op 2434) 
v_mul_f32 v26, v19, v22 // (op 2434) 
v_mul_f32 v27, v19, v23 // (op 2434) 

s_waitcnt vmcnt(0)
v_and_b32 v16, 65535, v14 // (op 2422) 
v_lshrrev_b32 v17, 16, v14 // (op 2422) 
v_lshlrev_b32 v16, 16, v16 // (op 2422) 
v_lshlrev_b32 v17, 16, v17 // (op 2422) 
v_mul_f32 v20, v18, v16 // (op 2422) 
v_mul_f32 v21, v18, v17 // (op 2422) 
v_and_b32 v16, 65535, v15 // (op 2422) 
v_lshrrev_b32 v17, 16, v15 // (op 2422) 
v_lshlrev_b32 v16, 16, v16 // (op 2422) 
v_lshlrev_b32 v17, 16, v17 // (op 2422) 
v_mul_f32 v22, v18, v16 // (op 2422) 
v_mul_f32 v23, v18, v17 // (op 2422) 


v_add_f32 v14, v20, v24 // (op 2423) 
v_add_f32 v15, v21, v25 // (op 2423) 
v_add_f32 v16, v22, v26 // (op 2423) 
v_add_f32 v17, v23, v27 // (op 2423) 


v_lshrrev_b32 v20, 16, v14 // (op 2424) 
v_lshrrev_b32 v21, 16, v15 // (op 2424) 
v_lshrrev_b32 v22, 16, v16 // (op 2424) 
v_lshrrev_b32 v23, 16, v17 // (op 2424) 


v_pack_b32_f16 v14, v20, v21 // (op 2410) 
v_pack_b32_f16 v15, v22, v23 // (op 2410) 

s_lshl_b32 s15, s24, 7 // (op 2410) 
s_add_u32 s15, s15, 64 // (op 2410) 
v_add_u32 v0, v5, s15 // (op 2410) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 2410) Store value





s_lshl_b32 s15, s4, 6 // (op 2430) 
s_mul_i32 s6, 3, s15 // (op 2430) 
s_add_u32 s6, s6, 64 // (op 2430) 

v_add_u32 v0, v4, s6 // (op 2430) 


buffer_load_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 2430) Load value

v_accvgpr_read v20, a28 // (op 2436) 
v_accvgpr_read v21, a29 // (op 2436) 
v_accvgpr_read v22, a30 // (op 2436) 
v_accvgpr_read v23, a31 // (op 2436) 
v_mul_f32 v24, v19, v20 // (op 2436) 
v_mul_f32 v25, v19, v21 // (op 2436) 
v_mul_f32 v26, v19, v22 // (op 2436) 
v_mul_f32 v27, v19, v23 // (op 2436) 

s_waitcnt vmcnt(0)
v_and_b32 v16, 65535, v14 // (op 2427) 
v_lshrrev_b32 v17, 16, v14 // (op 2427) 
v_lshlrev_b32 v16, 16, v16 // (op 2427) 
v_lshlrev_b32 v17, 16, v17 // (op 2427) 
v_mul_f32 v20, v18, v16 // (op 2427) 
v_mul_f32 v21, v18, v17 // (op 2427) 
v_and_b32 v16, 65535, v15 // (op 2427) 
v_lshrrev_b32 v17, 16, v15 // (op 2427) 
v_lshlrev_b32 v16, 16, v16 // (op 2427) 
v_lshlrev_b32 v17, 16, v17 // (op 2427) 
v_mul_f32 v22, v18, v16 // (op 2427) 
v_mul_f32 v23, v18, v17 // (op 2427) 


v_add_f32 v14, v20, v24 // (op 2428) 
v_add_f32 v15, v21, v25 // (op 2428) 
v_add_f32 v16, v22, v26 // (op 2428) 
v_add_f32 v17, v23, v27 // (op 2428) 


v_lshrrev_b32 v20, 16, v14 // (op 2429) 
v_lshrrev_b32 v21, 16, v15 // (op 2429) 
v_lshrrev_b32 v22, 16, v16 // (op 2429) 
v_lshrrev_b32 v23, 16, v17 // (op 2429) 


v_pack_b32_f16 v14, v20, v21 // (op 2412) 
v_pack_b32_f16 v15, v22, v23 // (op 2412) 

s_lshl_b32 s15, s24, 6 // (op 2412) 
s_mul_i32 s6, 3, s15 // (op 2412) 
s_add_u32 s6, s6, 64 // (op 2412) 

v_add_u32 v0, v5, s6 // (op 2412) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 2412) Store value





v_mov_b32 v0, v4 // (op 263) reg expression

buffer_load_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 263) Load value

v_accvgpr_read v20, a0 // (op 259) 
v_accvgpr_read v21, a1 // (op 259) 
v_accvgpr_read v22, a2 // (op 259) 
v_accvgpr_read v23, a3 // (op 259) 
v_mul_f32 v24, v19, v20 // (op 259) 
v_mul_f32 v25, v19, v21 // (op 259) 
v_mul_f32 v26, v19, v22 // (op 259) 
v_mul_f32 v27, v19, v23 // (op 259) 

s_waitcnt vmcnt(0)
v_and_b32 v16, 65535, v14 // (op 264) 
v_lshrrev_b32 v17, 16, v14 // (op 264) 
v_lshlrev_b32 v16, 16, v16 // (op 264) 
v_lshlrev_b32 v17, 16, v17 // (op 264) 
v_mul_f32 v20, v18, v16 // (op 264) 
v_mul_f32 v21, v18, v17 // (op 264) 
v_and_b32 v16, 65535, v15 // (op 264) 
v_lshrrev_b32 v17, 16, v15 // (op 264) 
v_lshlrev_b32 v16, 16, v16 // (op 264) 
v_lshlrev_b32 v17, 16, v17 // (op 264) 
v_mul_f32 v22, v18, v16 // (op 264) 
v_mul_f32 v23, v18, v17 // (op 264) 


v_add_f32 v14, v20, v24 // (op 260) 
v_add_f32 v15, v21, v25 // (op 260) 
v_add_f32 v16, v22, v26 // (op 260) 
v_add_f32 v17, v23, v27 // (op 260) 


v_lshrrev_b32 v20, 16, v14 // (op 261) 
v_lshrrev_b32 v21, 16, v15 // (op 261) 
v_lshrrev_b32 v22, 16, v16 // (op 261) 
v_lshrrev_b32 v23, 16, v17 // (op 261) 


v_pack_b32_f16 v14, v20, v21 // (op 262) 
v_pack_b32_f16 v15, v22, v23 // (op 262) 

v_mov_b32 v0, v5 // (op 262) reg expression

buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 262) Store value





s_lshl_b32 s15, s4, 6 // (op 2323) 
v_add_u32 v0, v4, s15 // (op 2323) 


buffer_load_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 2323) Load value

v_accvgpr_read v20, a4 // (op 2319) 
v_accvgpr_read v21, a5 // (op 2319) 
v_accvgpr_read v22, a6 // (op 2319) 
v_accvgpr_read v23, a7 // (op 2319) 
v_mul_f32 v24, v19, v20 // (op 2319) 
v_mul_f32 v25, v19, v21 // (op 2319) 
v_mul_f32 v26, v19, v22 // (op 2319) 
v_mul_f32 v27, v19, v23 // (op 2319) 

s_waitcnt vmcnt(0)
v_and_b32 v16, 65535, v14 // (op 2324) 
v_lshrrev_b32 v17, 16, v14 // (op 2324) 
v_lshlrev_b32 v16, 16, v16 // (op 2324) 
v_lshlrev_b32 v17, 16, v17 // (op 2324) 
v_mul_f32 v20, v18, v16 // (op 2324) 
v_mul_f32 v21, v18, v17 // (op 2324) 
v_and_b32 v16, 65535, v15 // (op 2324) 
v_lshrrev_b32 v17, 16, v15 // (op 2324) 
v_lshlrev_b32 v16, 16, v16 // (op 2324) 
v_lshlrev_b32 v17, 16, v17 // (op 2324) 
v_mul_f32 v22, v18, v16 // (op 2324) 
v_mul_f32 v23, v18, v17 // (op 2324) 


v_add_f32 v14, v20, v24 // (op 2320) 
v_add_f32 v15, v21, v25 // (op 2320) 
v_add_f32 v16, v22, v26 // (op 2320) 
v_add_f32 v17, v23, v27 // (op 2320) 


v_lshrrev_b32 v20, 16, v14 // (op 2321) 
v_lshrrev_b32 v21, 16, v15 // (op 2321) 
v_lshrrev_b32 v22, 16, v16 // (op 2321) 
v_lshrrev_b32 v23, 16, v17 // (op 2321) 


v_pack_b32_f16 v14, v20, v21 // (op 2322) 
v_pack_b32_f16 v15, v22, v23 // (op 2322) 

s_lshl_b32 s15, s24, 6 // (op 2322) 
v_add_u32 v0, v5, s15 // (op 2322) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 2322) Store value





s_lshl_b32 s15, s4, 7 // (op 2337) 
v_add_u32 v0, v4, s15 // (op 2337) 


buffer_load_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 2337) Load value

v_accvgpr_read v20, a8 // (op 2333) 
v_accvgpr_read v21, a9 // (op 2333) 
v_accvgpr_read v22, a10 // (op 2333) 
v_accvgpr_read v23, a11 // (op 2333) 
v_mul_f32 v24, v19, v20 // (op 2333) 
v_mul_f32 v25, v19, v21 // (op 2333) 
v_mul_f32 v26, v19, v22 // (op 2333) 
v_mul_f32 v27, v19, v23 // (op 2333) 

s_waitcnt vmcnt(0)
v_and_b32 v16, 65535, v14 // (op 2338) 
v_lshrrev_b32 v17, 16, v14 // (op 2338) 
v_lshlrev_b32 v16, 16, v16 // (op 2338) 
v_lshlrev_b32 v17, 16, v17 // (op 2338) 
v_mul_f32 v20, v18, v16 // (op 2338) 
v_mul_f32 v21, v18, v17 // (op 2338) 
v_and_b32 v16, 65535, v15 // (op 2338) 
v_lshrrev_b32 v17, 16, v15 // (op 2338) 
v_lshlrev_b32 v16, 16, v16 // (op 2338) 
v_lshlrev_b32 v17, 16, v17 // (op 2338) 
v_mul_f32 v22, v18, v16 // (op 2338) 
v_mul_f32 v23, v18, v17 // (op 2338) 


v_add_f32 v14, v20, v24 // (op 2334) 
v_add_f32 v15, v21, v25 // (op 2334) 
v_add_f32 v16, v22, v26 // (op 2334) 
v_add_f32 v17, v23, v27 // (op 2334) 


v_lshrrev_b32 v20, 16, v14 // (op 2335) 
v_lshrrev_b32 v21, 16, v15 // (op 2335) 
v_lshrrev_b32 v22, 16, v16 // (op 2335) 
v_lshrrev_b32 v23, 16, v17 // (op 2335) 


v_pack_b32_f16 v14, v20, v21 // (op 2336) 
v_pack_b32_f16 v15, v22, v23 // (op 2336) 

s_lshl_b32 s15, s24, 7 // (op 2336) 
v_add_u32 v0, v5, s15 // (op 2336) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 2336) Store value





s_lshl_b32 s15, s4, 6 // (op 2351) 
s_mul_i32 s6, 3, s15 // (op 2351) 

v_add_u32 v0, v4, s6 // (op 2351) 


buffer_load_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 2351) Load value

v_accvgpr_read v20, a12 // (op 2347) 
v_accvgpr_read v21, a13 // (op 2347) 
v_accvgpr_read v22, a14 // (op 2347) 
v_accvgpr_read v23, a15 // (op 2347) 
v_mul_f32 v24, v19, v20 // (op 2347) 
v_mul_f32 v25, v19, v21 // (op 2347) 
v_mul_f32 v26, v19, v22 // (op 2347) 
v_mul_f32 v27, v19, v23 // (op 2347) 

s_waitcnt vmcnt(0)
v_and_b32 v16, 65535, v14 // (op 2352) 
v_lshrrev_b32 v17, 16, v14 // (op 2352) 
v_lshlrev_b32 v16, 16, v16 // (op 2352) 
v_lshlrev_b32 v17, 16, v17 // (op 2352) 
v_mul_f32 v20, v18, v16 // (op 2352) 
v_mul_f32 v21, v18, v17 // (op 2352) 
v_and_b32 v16, 65535, v15 // (op 2352) 
v_lshrrev_b32 v17, 16, v15 // (op 2352) 
v_lshlrev_b32 v16, 16, v16 // (op 2352) 
v_lshlrev_b32 v17, 16, v17 // (op 2352) 
v_mul_f32 v22, v18, v16 // (op 2352) 
v_mul_f32 v23, v18, v17 // (op 2352) 


v_add_f32 v14, v20, v24 // (op 2348) 
v_add_f32 v15, v21, v25 // (op 2348) 
v_add_f32 v16, v22, v26 // (op 2348) 
v_add_f32 v17, v23, v27 // (op 2348) 


v_lshrrev_b32 v20, 16, v14 // (op 2349) 
v_lshrrev_b32 v21, 16, v15 // (op 2349) 
v_lshrrev_b32 v22, 16, v16 // (op 2349) 
v_lshrrev_b32 v23, 16, v17 // (op 2349) 


v_pack_b32_f16 v14, v20, v21 // (op 2350) 
v_pack_b32_f16 v15, v22, v23 // (op 2350) 

s_lshl_b32 s15, s24, 6 // (op 2350) 
s_mul_i32 s6, 3, s15 // (op 2350) 

v_add_u32 v0, v5, s6 // (op 2350) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 2350) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalBottom_0_beta_238:
s_add_u32 s0, s0, s1 // (op 451) 


s_mul_i32 s15, s50, s16 // (op 449) 
s_cmp_gt_u32 s51, s15 // (op 449) 
s_mov_b32 s6, scc // (op 449) 
s_sub_u32 s7, s51, s15 // (op 449) 

s_cmp_eq_i32 s6, 0 // (op 449) 
s_cselect_b32 s8, 0, s7 // (op 449) 


s_cmp_lt_u32 s50, s8 // (op 449) 
s_mov_b32 s15, scc // (op 449) 
s_cmp_eq_i32 s15, 0 // (op 449) 
s_cselect_b32 s6, s8, s50 // (op 449) 


s_cmp_lt_u32 s0, s6 // (op 449) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopTop_DPStreamTileLoop_449 // (op 449) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopTop_DPStreamTileLoop_449 if true)
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopBottom_DPStreamTileLoop_449:

















































s_endpgm  // End of RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2
.LRR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_end:
.size RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2, .LRR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_end-RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2
  .amdhsa_next_free_vgpr 176
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 50688 // lds bytes
  .amdhsa_accum_offset 144
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
    DataTypeB: B8
    DestDataType: b
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
            { type: int64, semantic: StrideD0 },
            { type: address, semantic: AddressD },
            { type: address, semantic: RR_SKAddressPartials },
            { type: int64, semantic: StrideA0 },
            { type: address, semantic: AddressA },
            { type: int64, semantic: SizeFree0 },
            { type: int64, semantic: StrideB0 },
            { type: address, semantic: AddressB },
            { type: int64, semantic: StrideScaleA0 },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: StrideScaleB0 },
            { type: address, semantic: AddressMXScaleB },
            { type: address, semantic: RR_SKAddressFlags },
            { type: uint32, semantic: RR_SKNumDPTilesPerWG },
            { type: uint32, semantic: RR_SKNumDPTiles } ]
    macrotile: [64, 128, 128]
    threads: [256, 1, 1]
    grid: [RocRollerStreamK, One, One]
    occupancy: 3
    workspaceType: RocRollerStreamK
    workspaceSizePerElemC: 4
    workspaceSizePerElemBias: 0
    unroll: 2
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [2, 4]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2
    .symbol: RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2.kd
    .kernarg_segment_size: 184
    .group_segment_fixed_size: 50688
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 88
    .vgpr_count: 144
    .agpr_count: 32
    .max_flat_workgroup_size: 256
    .wavefront_size: 64
    .args:
      - .name: Tensor_0_size_1_22
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
      - .name: Tensor_2_size_1_12
        .size: 8
        .offset: 64
        .value_kind: by_value
      - .name: Tensor_21_stride_1_24
        .size: 8
        .offset: 72
        .value_kind: by_value
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 80
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: SCRATCH_None
        .size: 8
        .offset: 88
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_stride_0_8
        .size: 8
        .offset: 96
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 104
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_0_9
        .size: 8
        .offset: 112
        .value_kind: by_value
      - .name: Tensor_2_stride_1_11
        .size: 8
        .offset: 120
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 128
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_stride_0_14
        .size: 8
        .offset: 136
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 144
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_7_stride_1_16
        .size: 8
        .offset: 152
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 160
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: SCRATCH_ZeroedBeforeAndAfter
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

